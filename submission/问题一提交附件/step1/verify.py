# 本程序及代码是在人工智能工具辅助下完成的。

# 人工智能工具名称：GPT-6 Astra

# 版本/型号：gpt-6-astra

# 开发机构/公司：OpenAI

# 版本颁布日期：2026年9月3日

"""逐样本独立核验输入保留、CTC路径、真实时序与跨模态聚合。"""
from common import ROOT, activate_project_dependencies, read_samples, save_json, sha256
activate_project_dependencies()
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def matches_recorded_code(path, expected):
    """保留原实验代码哈希；仅允许剥离本次逐字添加的AI声明。"""
    data = path.read_bytes()
    annotation = '# 本程序及代码是在人工智能工具辅助下完成的。\n\n# 人工智能工具名称：GPT-6 Astra\n\n# 版本/型号：gpt-6-astra\n\n# 开发机构/公司：OpenAI\n\n# 版本颁布日期：2026年9月3日\n\n'.encode('utf-8')
    return (hashlib.sha256(data).hexdigest() == expected or
            (data.startswith(annotation) and
             hashlib.sha256(data[len(annotation):]).hexdigest() == expected))


def direct_weights(intervals, frames, valid):
    """独立写法复算所有时间交叠，避免仅检查程序自己给出的通过标志。"""
    weights = np.zeros((len(intervals), len(frames)), np.float64)
    for k, (a, b) in enumerate(intervals):
        for j, (c, d) in enumerate(frames):
            if valid[j]:
                weights[k, j] = max(0., min(b, d)-max(a, c))
        total = weights[k].sum()
        if total > 0:
            weights[k] /= total
    return weights


def verify_sample(sample, directory, asset_hashes):
    """核验整个样本而非抽查单个位置；不要求无人脸帧具有虚构视觉值。"""
    meta = json.loads((directory/"metadata.json").read_text())
    for key in ("sample_id", "source_sha256", "raw_text", "label", "annotation", "video_id", "clip_id"):
        assert meta[key] == sample[key], (sample["sample_id"], key)
    assert meta["config_sha256"] == sha256(ROOT/"config.json")
    assert meta["model_assets_sha256"] == asset_hashes
    for name, checksum in meta["code_sha256"].items():
        assert matches_recorded_code(ROOT/name, checksum), name
    assert meta["padding_count"] == meta["truncated_words"] == 0
    with np.load(directory/"features.npz", allow_pickle=False) as z:
        for key in z.files:
            if z[key].dtype.kind in "fc":
                assert np.isfinite(z[key]).all(), key
        words = meta["words"]; n = len(words); duration = meta["duration"]
        assert z["text"].shape == (n, 768)
        assert z["audio"].shape == (n, 25)
        assert z["vision"].shape == (n, 52)
        for name in ("audio_bounds", "vision_bounds", "word_bounds", "ctc_bounds"):
            bounds = z[name]
            assert (bounds[:, 1] >= bounds[:, 0]).all(), name
            assert bounds.min() >= -1e-6 and bounds.max() <= duration + .05, name
        assert (np.diff(z["all_video_pts"]) > 0).all()
        np.testing.assert_array_equal(z["all_video_pts"][z["video_frame_indices"]], z["vision_pts"])
        np.testing.assert_allclose(z["vision_bounds"][1:, 0],z["vision_bounds"][:-1, 1])
        assert len(z["vision_pts"]) == len(z["vision_frames"])
        assert np.all(z["vision_frames"][~z["vision_frame_valid"]] == 0)
        b = z["word_bounds"][z["alignment_valid"]]
        assert np.all(b[1:, 0] >= b[:-1, 1]-1e-8)
        for w in words:
            assert meta["raw_text"][w["char_start"]:w["char_end"]] == w["original"]
            js = w["subword_indices"]
            if js:
                np.testing.assert_allclose(z["text"][w["index"]], z["subword_features"][js].mean(axis=0), rtol=1e-5, atol=1e-6)
            if w["start"] is not None:
                np.testing.assert_allclose(z["word_bounds"][w["index"]], [w["start"],w["end"]])
        for mod in ("audio", "vision"):
            weights = direct_weights(z["word_bounds"], z[mod+"_bounds"], z[mod+"_frame_valid"])
            np.testing.assert_allclose(z[mod+"_weights"], weights, rtol=1e-5, atol=1e-7)
            observed = weights.sum(axis=1) > 0
            np.testing.assert_array_equal(z[mod+"_valid"], observed)
            np.testing.assert_allclose(z[mod], weights @ z[mod+"_frames"], rtol=1e-5, atol=1e-5)
            assert (z[mod][~observed] == 0).all()
        if meta["alignment_error"] is None:
            states = z["ctc_state_path"]
            transitions = np.diff(states)
            assert np.isin(transitions, [0,1,2]).all()
            labels = np.zeros(2*len(z["ctc_target_ids"])+1, dtype=int)
            labels[1::2] = z["ctc_target_ids"]
            symbols = labels[states]
            collapsed = symbols[np.r_[True, symbols[1:] != symbols[:-1]]]
            np.testing.assert_array_equal(collapsed[collapsed != 0], z["ctc_target_ids"])
            for w in words:
                matches = np.isin((states-1)//2, np.flatnonzero(z["ctc_character_word"]==w["index"])) & (states%2==1)
                js = np.flatnonzero(matches)
                assert len(js) > 0
                np.testing.assert_allclose(z["word_bounds"][w["index"]],
                                          [z["ctc_bounds"][js[0],0],z["ctc_bounds"][js[-1],1]])
        quality_path=directory/"quality.json"
        if quality_path.exists():
            quality=json.loads(quality_path.read_text())
            assert quality["raw_text"]==meta["raw_text"]
            suspect=(z["word_confidence"].mean()<.2 or (z["word_confidence"]<.1).mean()>.5 or
                     meta["diagnostic_wer"]>.8 or meta["alignment_error"] is not None)
            assert quality["requires_transcript_review"]==bool(suspect)
            with np.load(directory/"quality_masks.npz") as q:
                expected=z["alignment_high_confidence"] & (not suspect)
                np.testing.assert_array_equal(q["timing_usable"],expected)
                for mod in ("text","audio","vision"):
                    np.testing.assert_array_equal(q[mod+"_aligned_usable"],expected & z[mod+"_valid"])
                for width in (.2,.5):
                    with np.load(directory/f"windows_{width:.1f}s.npz") as windows:
                        wb=windows["bounds"]
                        assert abs(wb[0,0])<1e-8 and abs(wb[-1,1]-duration)<1e-8
                        np.testing.assert_allclose(wb[1:,0],wb[:-1,1])
                        weights=direct_weights(wb,z["word_bounds"],q["text_aligned_usable"])
                        np.testing.assert_allclose(windows["text_word_weights"],weights,atol=1e-7)
                        np.testing.assert_array_equal(windows["text_valid"],weights.sum(axis=1)>0)
                        for mod in ("audio","vision"):
                            weights=direct_weights(wb,z[mod+"_bounds"],z[mod+"_frame_valid"])
                            np.testing.assert_allclose(windows[mod],weights@z[mod+"_frames"],rtol=1e-5,atol=1e-5)
                            np.testing.assert_array_equal(windows[mod+"_valid"],weights.sum(axis=1)>0)
        return dict(sample_id=sample["sample_id"], status="passed", words=n,
                    feature_sha256=sha256(directory/"features.npz"),
                    alignment_failed=meta["alignment_error"] is not None)


def main():
    p=argparse.ArgumentParser();p.add_argument("--results",type=Path,default=ROOT/"results")
    p.add_argument("--allow-partial",action="store_true");args=p.parse_args()
    samples=read_samples(); checks=[]
    asset_hashes={p.relative_to(ROOT).as_posix():sha256(p) for p in (ROOT/"assets").rglob("*")
                  if p.is_file() and ".cache" not in p.parts}
    actual={p.name for p in (args.results/"samples").iterdir() if (p/"features.npz").exists()}
    expected={s["sample_id"] for s in samples}
    assert actual <= expected
    if not args.allow_partial:
        assert actual == expected, f"缺少 {sorted(expected-actual)}"
        for sample in samples:
            directory=args.results/"samples"/sample["sample_id"]
            for name in ("quality.json","quality_masks.npz","windows_0.2s.npz","windows_0.5s.npz"):
                assert (directory/name).is_file(),(sample["sample_id"],name)
    for sample in samples:
        if sample["sample_id"] in actual:
            checks.append(verify_sample(sample,args.results/"samples"/sample["sample_id"],asset_hashes))
    audit=dict(expected_samples=100, checked_samples=len(checks), all_passed=True,
               source_video_hashes_verified=True, code_and_config_hashes_verified=True,
               model_asset_hashes_verified=True,
               ctc_paths_verified=True, aggregation_recomputed=True,
               note="通过仅指结构与数学映射自洽，不证明真实词界准确或最大人脸就是说话者。", samples=checks)
    save_json(args.results/"verification.json",audit)
    print(json.dumps({k:v for k,v in audit.items() if k!="samples"},ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()
