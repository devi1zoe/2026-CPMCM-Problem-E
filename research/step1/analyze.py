"""只评价问题一的覆盖、可追溯性、观测质量和窗口平滑效应。"""
from common import ROOT, PROJECT, activate_project_dependencies, save_json
activate_project_dependencies()
import argparse
import html
import json
import os
from pathlib import Path
from urllib.parse import quote
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from alignment import overlap_pool, window_bounds

FONT = FontProperties(fname="/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
plt.rcParams.update({"axes.unicode_minus": False, "font.family": FONT.get_name(),
                     "figure.dpi": 130, "savefig.dpi": 160, "text.parse_math": False})
matplotlib.font_manager.fontManager.addfont(FONT.get_file())


def reconstruction_error(bounds, raw_bounds, values, valid):
    """将窗口均值映回原始帧，计算片段内逐维标准化MSE；不衡量情感预测性能。"""
    pooled, mask, _ = overlap_pool(bounds, raw_bounds, values, valid)
    reconstructed, support, _ = overlap_pool(raw_bounds, bounds, pooled, mask)
    keep = valid & support
    if not keep.any():
        return None
    sd = values[keep].std(axis=0)
    dimensions = sd > 1e-6
    if not dimensions.any():
        return 0.
    return float(np.mean(((reconstructed[keep][:, dimensions]-values[keep][:, dimensions]) /
                          sd[dimensions]) ** 2))


def sample_figure(directory, output):
    """把实际视频帧、声学曲线、词界和面部表情置于同一物理时间轴。"""
    import av
    meta = json.loads((directory / "metadata.json").read_text())
    quality_path=directory/"quality.json"
    risk=json.loads(quality_path.read_text())["requires_transcript_review"] if quality_path.exists() else False
    with np.load(directory / "features.npz") as z:
        data = {k: z[k] for k in z.files}
    words = meta["words"]
    end = min(meta["duration"], 10.)
    chosen = [w for w in words if w["start"] is not None and w["start"] < end]
    if len(chosen) > 28:
        chosen = chosen[:28]; end = min(end, chosen[-1]["end"]+.2)
    times = np.linspace(.05, end-.05, 6)
    frames, frame_times = [], []
    with av.open(str(PROJECT / meta["source_path"])) as c:
        index = 0
        for f in c.decode(video=0):
            t = float(f.pts*f.time_base)-meta["video_origin"]
            if index < len(times) and t >= times[index]:
                frames.append(f.to_image()); frame_times.append(t); index += 1
            if index == len(times):
                break
    fig = plt.figure(figsize=(16, 11), layout="constrained")
    gs = fig.add_gridspec(5, 6, height_ratios=[1.2, 1, 1, 1.6, 1.1])
    for j, im in enumerate(frames):
        ax = fig.add_subplot(gs[0, j]); ax.imshow(im); ax.axis("off")
        ax.set_title(f"实际帧 PTS={frame_times[j]:.3f}s", fontsize=10)
    centers = data["audio_bounds"].mean(axis=1)
    ax = fig.add_subplot(gs[1, :]); ax.plot(centers, data["audio_rms"], lw=.8)
    ax.fill_between(centers, 0, data["audio_rms"], where=data["audio_silence"], color="gray", alpha=.4)
    ax.set(ylabel="音频 RMS", xlim=(0, end), title="灰色标记低能量窗口（不等于无效观测）")
    ax = fig.add_subplot(gs[2, :])
    feature = next(i for i, name in enumerate(meta["audio_feature_names"]) if "F0" in name)
    ax.plot(centers, data["audio_frames"][:, feature], color="#a55d26", lw=.9)
    ax.set(ylabel="F0 半音值", xlim=(0, end), title=meta["audio_feature_names"][feature])
    ax = fig.add_subplot(gs[3, :])
    for j, w in enumerate(chosen):
        color = plt.cm.viridis(w["confidence"])
        y = j % 4
        ax.barh(y, w["end"]-w["start"], left=w["start"], height=.6, color=color)
        ax.text((w["start"]+w["end"])/2, y+.33, w["original"], fontsize=8, ha="center")
    ax.set(xlim=(0, end), ylim=(-.5, 4), yticks=[], title="CTC词区间；颜色为模型置信度（不是标注准确率）")
    fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(0,1),cmap="viridis"),ax=ax,
                 fraction=.015,pad=.01,label="CTC置信度")
    ax = fig.add_subplot(gs[4, :])
    names = meta["vision_feature_names"] or []
    for name in ["mouthSmileLeft", "mouthSmileRight", "jawOpen", "browInnerUp"]:
        if name in names:
            vals = data["vision_frames"][:, names.index(name)].copy()
            vals[~data["vision_frame_valid"]] = np.nan
            ax.plot(data["vision_pts"], vals, label=name, lw=1)
    ax.set(xlim=(0, end), xlabel="相对视频起点 / 秒", ylabel="表情系数", ylim=(-.02, 1.02),
           title="5 Hz抽样；无人脸处断线，不补造面部特征")
    if data["vision_frame_valid"].any():
        ax.legend(ncol=4, fontsize=8)
    else:
        ax.text(.5,.5,"抽样帧未检出可用人脸；不代表原画面无人",transform=ax.transAxes,ha="center",color="#885522")
    risk_text="｜转写一致性风险：词界仅为候选" if risk else ""
    fig.suptitle(f"样本 {meta['sample_id']}｜展示前 {end:.2f} 秒｜标签 {meta['label']:.3f}（仅原始标注）{risk_text}", fontsize=12)
    fig.savefig(output); plt.close(fig)


def feature_mapping(directory, output):
    """逐词列出文本向量与原生音视频帧的对应，完整权重另见NPZ。"""
    meta=json.loads((directory/"metadata.json").read_text());rows=[]
    with np.load(directory/"features.npz") as z:
        for word in meta["words"]:
            i=word["index"]
            aj=np.flatnonzero(z["audio_weights"][i]>0)
            vj=np.flatnonzero(z["vision_weights"][i]>0)
            rows.append(dict(word=word["original"],word_index=i,start=word["start"],end=word["end"],
                        confidence=word["confidence"],subword_indices=json.dumps(word["subword_indices"]),
                        text_dimension=768,text_vector_norm=float(np.linalg.norm(z["text"][i])),
                        audio_dimension=25,audio_frame_indices=json.dumps(aj.tolist()),
                        audio_first_feature=float(z["audio"][i,0]),
                        vision_dimension=52,video_frame_indices=json.dumps(z["video_frame_indices"][vj].tolist()),
                        video_pts=json.dumps(z["vision_pts"][vj].tolist()),
                        vision_observed=bool(z["vision_valid"][i])))
    pd.DataFrame(rows).to_csv(output,index=False,encoding="utf-8-sig")


def review_html(metas, output):
    """生成本地声画回看页，点词可跳转；时间仍为待核验的模型估计。"""
    examples = []
    for m in metas:
        examples.append(dict(id=m["sample_id"], text=m["raw_text"],
                             src=quote(os.path.relpath(PROJECT/m["source_path"], output.parent)),
                             words=m["words"],review=m.get("_requires_review",False)))
    payload = json.dumps(examples, ensure_ascii=False).replace("</", "<\\/")
    template = '''<!doctype html><html lang="zh"><meta charset="utf-8"><title>问题一：100条样本回看</title>
<style>body{max-width:1000px;margin:25px auto;font:16px system-ui;background:#f6f8fb;color:#172d46}video{width:100%;max-height:430px;background:#111}button{margin:4px;padding:7px;border:1px solid #aac;border-radius:5px;cursor:pointer}.low{background:#ffe2bd}.active{outline:3px solid #3975df}#text{line-height:1.8}small{color:#596675}</style>
<h1>问题一：原始素材与词级时间回看</h1><p>点击词按钮跳转至估计发音区间。橙色为低置信度；时间戳不是人工真值。未进行情感预测。</p>
<select id="select"></select><p id="text"></p><video id="video" controls preload="metadata"></video><p id="position"></p><div id="words"></div>
<script>const data=PAYLOAD;const sel=document.querySelector('#select'),vid=document.querySelector('#video');
data.forEach((x,i)=>{let o=document.createElement('option');o.value=i;o.textContent=x.id;sel.append(o)});
function load(){let x=data[sel.value];document.querySelector('#text').textContent=(x.review?'【转写一致性风险：以下词界仅为候选，待复核】 ':'')+x.text;vid.src=x.src;let host=document.querySelector('#words');host.replaceChildren();x.words.forEach(w=>{let b=document.createElement('button');b.textContent=w.original+' '+(w.start===null?'无时间':w.start.toFixed(2)+'–'+w.end.toFixed(2)+'s');b.title='模型置信度 '+w.confidence.toFixed(3);b.className=(x.review||w.status==='low_confidence')?'low':'';b.disabled=w.start===null;b.onclick=()=>{vid.currentTime=Math.max(0,w.start-.1);vid.play()};host.append(b)})}
sel.onchange=load;vid.ontimeupdate=()=>{document.querySelector('#position').textContent=vid.currentTime.toFixed(3)+' 秒';let x=data[sel.value];document.querySelectorAll('#words button').forEach((b,i)=>b.classList.toggle('active',x.words[i].start!==null&&vid.currentTime>=x.words[i].start&&vid.currentTime<=x.words[i].end))};load();</script></html>'''
    output.write_text(template.replace("PAYLOAD", payload), encoding="utf-8")


def main():
    p = argparse.ArgumentParser(); p.add_argument("--results", type=Path, default=ROOT/"results")
    args = p.parse_args(); out = args.results
    dirs = sorted((out/"samples").glob("*"))
    summaries, modality_rows, word_rows, windows, metas = [], [], [], [], []
    for directory in dirs:
        if not (directory/"metadata.json").exists():
            continue
        meta = json.loads((directory/"metadata.json").read_text()); metas.append(meta)
        quality_path=directory/"quality.json"
        meta["_requires_review"]=json.loads(quality_path.read_text())["requires_transcript_review"] if quality_path.exists() else False
        with np.load(directory/"features.npz") as z:
            n = len(meta["words"])
            row = dict(sample_id=meta["sample_id"], duration_seconds=meta["duration"], words=n,
                       requires_transcript_review=meta["_requires_review"],
                       subwords=len(z["subword_ids"]), decoded_video_frames=len(z["all_video_pts"]),
                       sampled_video_frames=len(z["vision_pts"]), acoustic_frames=len(z["audio_frames"]),
                       aligned_words=int(z["alignment_valid"].sum()),
                       low_confidence_words=int((z["alignment_valid"] & ~z["alignment_high_confidence"]).sum()),
                       mean_word_confidence=float(z["word_confidence"].mean()),
                       text_observed_words=int(z["text_valid"].sum()),
                       audio_observed_words=int(z["audio_valid"].sum()),
                       vision_observed_words=int(z["vision_valid"].sum()),
                       face_frame_ratio=float(z["vision_frame_valid"].mean()),
                       no_face_frames=int((z["vision_face_count"]==0).sum()),
                       multiple_face_frames=int((z["vision_face_count"]>1).sum()),
                       silent_audio_frames=int(z["audio_silence"].sum()),
                       audio_frames_observed=int(z["audio_frame_valid"].sum()),
                       diagnostic_wer=meta["diagnostic_wer"], padding_count=0, truncated_words=0,
                       elapsed_seconds=meta["elapsed_seconds"], feature_bytes=(directory/"features.npz").stat().st_size)
            summaries.append(row)
            for mod in ("text", "audio", "vision"):
                modality_rows.append(dict(sample_id=meta["sample_id"], modality=mod,
                                          duration_seconds=meta["duration"], feature_dimension=z[mod].shape[1],
                                          stored_positions=n, valid_positions=int(z[mod+"_valid"].sum()),
                                          alignment_granularity="original_transcript_word",
                                          padding_count=0, feature_path=f"samples/{meta['sample_id']}/features.npz"))
            for word in meta["words"]:
                word_rows.append(dict(sample_id=meta["sample_id"], **{k:v for k,v in word.items() if k != "subword_indices"},
                                      subword_indices=json.dumps(word["subword_indices"])))
            quality_file=directory/"quality_masks.npz"
            if quality_file.exists():
                with np.load(quality_file) as quality:
                    usable_text=quality["text_aligned_usable"]
            else:
                usable_text=z["alignment_high_confidence"] & z["text_valid"]
            for width in (.2, .5):
                bins = window_bounds(meta["duration"], width)
                valid_bounds = z["word_bounds"][usable_text]
                overlaps = np.maximum(0, np.minimum(valid_bounds[:, None, 1], bins[None, :, 1])-
                                      np.maximum(valid_bounds[:, None, 0], bins[None, :, 0]))
                split_words = (overlaps > 1e-8).sum(axis=1) > 1
                audio, a_valid, _ = overlap_pool(bins, z["audio_bounds"], z["audio_frames"], z["audio_frame_valid"])
                vision, v_valid, _ = overlap_pool(bins, z["vision_bounds"], z["vision_frames"], z["vision_frame_valid"])
                _, text_mask, text_weights = overlap_pool(bins, z["word_bounds"], z["text"], usable_text)
                np.savez_compressed(directory/f"windows_{width:.1f}s.npz", bounds=bins, audio=audio, vision=vision,
                                    audio_valid=a_valid, vision_valid=v_valid, text_valid=text_mask,
                                    text_word_weights=text_weights)
                windows.append(dict(sample_id=meta["sample_id"], window_seconds=width, windows=len(bins),
                                    timing_usable_words=int(usable_text.sum()),
                                    words_crossing_boundary=int(split_words.sum()),
                                    word_boundary_crossing_ratio=float(split_words.mean()) if len(split_words) else None,
                                    audio_normalized_reconstruction_mse=reconstruction_error(bins, z["audio_bounds"], z["audio_frames"], z["audio_frame_valid"]),
                                    vision_normalized_reconstruction_mse=reconstruction_error(bins, z["vision_bounds"], z["vision_frames"], z["vision_frame_valid"])))
    if not summaries:
        raise ValueError("没有实验结果")
    df = pd.DataFrame(summaries); wf = pd.DataFrame(word_rows); win = pd.DataFrame(windows)
    for frame, name in [(df,"sample_summary.csv"),(wf,"word_alignment.csv"),
                        (pd.DataFrame(modality_rows),"modality_summary.csv"),(win,"window_comparison.csv")]:
        frame.to_csv(out/name, index=False, encoding="utf-8-sig")
    totals = {name: int(df[name].sum()) for name in ["words", "subwords", "decoded_video_frames",
              "sampled_video_frames", "acoustic_frames", "aligned_words", "low_confidence_words",
              "text_observed_words", "audio_observed_words", "vision_observed_words", "no_face_frames",
              "multiple_face_frames", "silent_audio_frames", "audio_frames_observed", "padding_count", "truncated_words"]}
    metrics = dict(samples=len(df), totals=totals, total_duration_seconds=float(df.duration_seconds.sum()),
                   duration_min=float(df.duration_seconds.min()), duration_max=float(df.duration_seconds.max()),
                   median_word_confidence=float(wf.confidence.median()),
                   mean_word_confidence=float(wf.confidence.mean()),
                   mean_diagnostic_wer=float(df.diagnostic_wer.mean()),
                   elapsed_seconds=float(df.elapsed_seconds.sum()), feature_bytes=int(df.feature_bytes.sum()),
                   window_comparison=win.groupby("window_seconds").mean(numeric_only=True).reset_index().to_dict("records"),
                   lowest_confidence_samples=df.nsmallest(5, "mean_word_confidence").to_dict("records"),
                   lowest_face_coverage_samples=df.nsmallest(5,"face_frame_ratio").to_dict("records"))
    save_json(out/"metrics.json", metrics)
    figs = out/"figures"; figs.mkdir(exist_ok=True)
    fig, axs = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    axs[0,0].hist(df.duration_seconds, bins=15, color="#4778a8"); axs[0,0].set(xlabel="视频时长 / 秒", ylabel="样本数")
    axs[0,1].hist(wf.confidence, bins=20, color="#52997b"); axs[0,1].axvline(.1,color="red",ls="--");axs[0,1].set(xlabel="词级CTC置信度（非准确率）",ylabel="词数")
    axs[1,0].bar(["文本", "语音", "视觉"], [totals[k]/totals["words"] for k in
                                               ["text_observed_words","audio_observed_words","vision_observed_words"]],color=["#4778a8","#be8b55","#52997b"])
    axs[1,0].set(ylabel="词位置可观测比例",ylim=(0,1.05))
    axs[1,1].scatter(df.diagnostic_wer,df.mean_word_confidence,c=df.face_frame_ratio,cmap="viridis",s=25)
    axs[1,1].set(xlabel="贪心ASR与给定转写的词错误率",ylabel="平均词级CTC置信度",title="仅作转写一致性诊断，不验证真实时间误差")
    fig.savefig(figs/"quality_overview.png");plt.close(fig)
    fig, axs = plt.subplots(1,3,figsize=(13,4),layout="constrained")
    for ax,col,title in zip(axs,["audio_normalized_reconstruction_mse","vision_normalized_reconstruction_mse","word_boundary_crossing_ratio"],
                            ["声学标准化重建MSE","视觉标准化重建MSE","词跨窗口边界比例"]):
        ax.boxplot([win.loc[win.window_seconds==w,col].dropna() for w in (.2,.5)],tick_labels=["0.2秒","0.5秒"])
        ax.set_title(title);ax.grid(axis="y",alpha=.2)
    fig.savefig(figs/"window_comparison.png");plt.close(fig)
    # 典型图选择准则公开：在脸覆盖>=80%的样本中找时长最接近10秒者；另绘最低置信度例。
    candidates=df[(df.face_frame_ratio>=.8) & ~df.requires_transcript_review]
    typical=(candidates if len(candidates) else df).iloc[((candidates if len(candidates) else df).duration_seconds-10).abs().argsort().iloc[0]]
    difficult=df.nsmallest(1,"mean_word_confidence").iloc[0]
    for kind, row in [("typical",typical),("low_confidence",difficult)]:
        sample_figure(out/"samples"/row.sample_id,figs/f"alignment_{kind}.png")
    feature_mapping(out/"samples"/typical.sample_id,out/"typical_feature_mapping.csv")
    save_json(out/"figure_samples.json",dict(typical=typical.sample_id,low_confidence=difficult.sample_id))
    review_html(metas,out/"alignment_review.html")
    print(json.dumps({k:v for k,v in metrics.items() if not k.startswith("lowest")},ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()
