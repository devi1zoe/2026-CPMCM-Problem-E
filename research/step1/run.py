"""问题一全量处理入口：不训练情感模型，不读取附件2、3、4。"""
from common import ROOT, PROJECT, activate_project_dependencies, read_samples, save_json, sha256
activate_project_dependencies()
import argparse
import datetime
import importlib.metadata
import json
import os
import platform
import random
import sys
import time
import traceback
import numpy as np
import torch
from alignment import (extract_words, make_target, ctc_viterbi, words_from_path,
                       overlap_pool, word_error_rate)
from features import Extractors, decode_media, contact_sheet


def process_one(sample, extractors, config, destination, provenance):
    """完整保留原样本；低置信度与无观测不会导致删样本或悄悄均分词时间。"""
    began = time.monotonic()
    media = decode_media(PROJECT / sample["source_path"], config["video_sampling_hz"])
    words = extract_words(sample["raw_text"])
    text, text_valid, subwords, sub_ids, sub_map, mapping = extractors.text(sample["raw_text"], words)
    audio, audio_bounds, audio_valid, rms, silence, audio_names = extractors.acoustic(media)
    vision, vision_valid, face_count, vision_names = extractors.visual(media)
    logp, ctc_bounds, asr, convolution = extractors.emissions(media)
    targets, owners, target_text = make_target(words, extractors.processor.tokenizer.get_vocab())
    alignment_error = None
    try:
        path, score = ctc_viterbi(logp, targets, extractors.processor.tokenizer.pad_token_id)
        bounds, confidence, aligned = words_from_path(path, targets, owners, logp, words,
                                                      ctc_bounds, config["word_confidence_threshold"])
    except ValueError as exc:
        alignment_error = str(exc)
        bounds = np.zeros((len(words), 2)); confidence = np.zeros(len(words), np.float32)
        aligned = np.zeros(len(words), bool); path = np.full(len(logp), -1, np.int32); score = None
        for w in words:
            w.update(start=None, end=None, confidence=0., status="alignment_failed")
    pooled_audio, observed_audio, audio_weights = overlap_pool(bounds, audio_bounds, audio, audio_valid)
    pooled_vision, observed_vision, vision_weights = overlap_pool(bounds, media["video_bounds"], vision, vision_valid)
    high_confidence = aligned & (confidence >= config["word_confidence_threshold"])
    arrays = dict(text=text, audio=pooled_audio, vision=pooled_vision, word_bounds=bounds,
                  text_valid=text_valid, audio_valid=observed_audio, vision_valid=observed_vision,
                  alignment_valid=aligned, alignment_high_confidence=high_confidence,
                  word_confidence=confidence, subword_features=subwords, subword_ids=sub_ids,
                  subword_to_word=sub_map, audio_frames=audio, audio_bounds=audio_bounds,
                  audio_frame_valid=audio_valid, audio_rms=rms, audio_silence=silence,
                  vision_frames=vision, vision_pts=media["video_pts"], vision_bounds=media["video_bounds"],
                  vision_frame_valid=vision_valid, vision_face_count=face_count,
                  video_frame_indices=media["video_frame_indices"], all_video_pts=media["all_video_pts"],
                  audio_weights=audio_weights, vision_weights=vision_weights,
                  ctc_log_probs=logp, ctc_bounds=ctc_bounds, ctc_state_path=path,
                  ctc_target_ids=targets, ctc_character_word=owners)
    np.savez_compressed(destination / "features.npz", **arrays)
    contact_sheet(media, destination / "contact_sheet.jpg")
    metadata = dict(**sample, **provenance, duration=media["duration"], video_rate=media["video_rate"],
                    video_origin=media["video_origin"], audio_start=media["audio_start"],
                    audio_duration=len(media["waveform"])/16000,
                    audio_coverage_ratio=float(media["audio_coverage"].mean()),
                    words=words, subwords=mapping, audio_feature_names=audio_names,
                    vision_feature_names=vision_names, convolution=convolution,
                    ctc_target_text=target_text, diagnostic_asr=asr,
                    diagnostic_wer=word_error_rate(target_text.replace("|", " "), asr),
                    alignment_path_log_probability=score, alignment_error=alignment_error,
                    padding_count=0, truncated_words=0,
                    visual_time_support="nearest-sampled-frame cells, midpoint boundaries; 5 Hz sampling",
                    face_selection="largest detected face; speaker identity not verified",
                    feature_dimensions=dict(text=text.shape[1], audio=audio.shape[1], vision=vision.shape[1]),
                    statuses=dict(low_confidence_words=int((aligned & ~high_confidence).sum()),
                                  unaligned_words=int((~aligned).sum()),
                                  no_face_frames=int((face_count == 0).sum()),
                                  multiple_face_frames=int((face_count > 1).sum()),
                                  silent_audio_frames=int(silence.sum()),
                                  invalid_audio_frames=int((~audio_valid).sum())),
                    elapsed_seconds=time.monotonic()-began)
    save_json(destination / "metadata.json", metadata)
    return metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--only", help="指定一个完整sample_id")
    parser.add_argument("--output", type=__import__('pathlib').Path, default=ROOT / "results")
    args = parser.parse_args()
    config = json.loads((ROOT / "config.json").read_text())
    random.seed(config["seed"]); np.random.seed(config["seed"]); torch.manual_seed(config["seed"])
    torch.set_num_threads(4)
    if args.device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA不可用：请在沙箱外通过既有根Python运行，不能静默改为CPU")
        torch.cuda.manual_seed_all(config["seed"])
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
    args.output.mkdir(parents=True, exist_ok=True)
    samples = read_samples()
    save_json(args.output / "input_manifest.json", samples)
    provenance = dict(config_sha256=sha256(ROOT / "config.json"),
                      code_sha256={name: sha256(ROOT / name) for name in
                                   ("common.py", "alignment.py", "features.py", "run.py")})
    assets = {p.relative_to(ROOT).as_posix(): sha256(p) for p in (ROOT / "assets").rglob("*")
              if p.is_file() and ".cache" not in p.parts}
    provenance["model_assets_sha256"] = assets
    save_json(args.output / "environment.json", dict(
        created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        python=sys.executable, python_version=sys.version, platform=platform.platform(),
        torch_file=torch.__file__, torch_version=torch.__version__, cuda=torch.version.cuda,
        device=args.device, gpu=torch.cuda.get_device_name(0) if args.device.startswith("cuda") else None,
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        packages={p: importlib.metadata.version(p) for p in
                  ("numpy", "scipy", "pandas", "av", "pillow", "transformers", "mediapipe",
                   "opensmile", "openpyxl", "num2words", "matplotlib", "tokenizers")},
        asset_sha256=assets, config=config, **provenance))
    if args.only:
        samples = [s for s in samples if s["sample_id"] == args.only]
        if not samples:
            raise ValueError("sample_id不存在")
    if args.limit:
        samples = samples[:args.limit]
    extractor = Extractors(ROOT / "assets", config, args.device)
    failures = []
    try:
        for index, sample in enumerate(samples):
            directory = args.output / "samples" / sample["sample_id"]
            directory.mkdir(parents=True, exist_ok=True)
            existing = directory / "metadata.json"
            if args.resume and existing.is_file() and (directory / "features.npz").is_file():
                old = json.loads(existing.read_text())
                if all(old.get(k) == v for k, v in provenance.items()) and old["source_sha256"] == sample["source_sha256"]:
                    print(f"SKIP {sample['sample_id']}", flush=True)
                    continue
            try:
                meta = process_one(sample, extractor, config, directory, provenance)
                event = dict(sample_id=sample["sample_id"], status="processed", seconds=meta["elapsed_seconds"],
                             word_count=len(meta["words"]), **meta["statuses"])
                print(f"[{index+1}/{len(samples)}] " + json.dumps(event), flush=True)
            except Exception as exc:
                event = dict(sample_id=sample["sample_id"], status="failed", error=repr(exc), traceback=traceback.format_exc())
                failures.append(event)
                print(json.dumps(event), flush=True)
            with open(args.output / "processing_log.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(event, ensure_ascii=False) + "\n")
    finally:
        extractor.close()
    save_json(args.output / "failures.json", failures)
    if failures:
        raise SystemExit(f"{len(failures)}个样本处理失败；记录已保存，不能声称全量完成")


if __name__ == "__main__":
    main()
