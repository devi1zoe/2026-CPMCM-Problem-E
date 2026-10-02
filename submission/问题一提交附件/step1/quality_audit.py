# 本程序及代码是在人工智能工具辅助下完成的。

# 人工智能工具名称：GPT-6 Astra

# 版本/型号：gpt-6-astra

# 开发机构/公司：OpenAI

# 版本颁布日期：2026年9月3日

"""在保留全部候选对齐的前提下，单独记录转写一致性风险及保守使用掩码。"""
from common import ROOT, activate_project_dependencies, save_json
activate_project_dependencies()
import json
import numpy as np
import pandas as pd


def main():
    """阈值为人工质控规则，不是准确率校准，也不使用情感标签选择阈值。"""
    rows=[]; sensitivity=[]
    for directory in sorted((ROOT/"results/samples").iterdir()):
        if not (directory/"metadata.json").exists():
            continue
        m=json.loads((directory/"metadata.json").read_text())
        with np.load(directory/"features.npz") as z:
            average=float(z["word_confidence"].mean())
            low_fraction=float((z["word_confidence"]<.1).mean())
            reasons=[]
            if average<.2: reasons.append("mean_CTC_confidence_below_0.2")
            if low_fraction>.5: reasons.append("more_than_half_words_below_0.1")
            if m["diagnostic_wer"]>.8: reasons.append("diagnostic_WER_above_0.8")
            if m["alignment_error"] is not None: reasons.append("no_complete_CTC_path")
            suspect=bool(reasons)
            # 风险样本不宣称局部词时间可信；文本内容仍保留为无时间保证的序列。
            usable=z["alignment_high_confidence"] & (not suspect)
            np.savez_compressed(directory/"quality_masks.npz",timing_usable=usable,
                                text_aligned_usable=usable & z["text_valid"],
                                audio_aligned_usable=usable & z["audio_valid"],
                                vision_aligned_usable=usable & z["vision_valid"])
            record=dict(sample_id=m["sample_id"],requires_transcript_review=suspect,reasons=reasons,
                        mean_confidence=average,low_confidence_fraction=low_fraction,
                        diagnostic_wer=m["diagnostic_wer"],usable_word_times=int(usable.sum()),
                        words=len(usable),raw_text=m["raw_text"],diagnostic_asr=m["diagnostic_asr"])
            save_json(directory/"quality.json",record)
            rows.append({**record,"reasons":";".join(reasons)})
            for confidence in (.1,.2,.3):
                for wer in (.6,.8,1.):
                    sensitivity.append(dict(mean_confidence_threshold=confidence,wer_threshold=wer,
                        flagged=int(average<confidence or low_fraction>.5 or m["diagnostic_wer"]>wer or
                                    m["alignment_error"] is not None)))
    assert len(rows)==100
    df=pd.DataFrame(rows);df.to_csv(ROOT/"results/alignment_quality_review.csv",index=False,encoding="utf-8-sig")
    sf=pd.DataFrame(sensitivity).groupby(["mean_confidence_threshold","wer_threshold"]).flagged.sum().reset_index()
    sf.to_csv(ROOT/"results/quality_threshold_sensitivity.csv",index=False,encoding="utf-8-sig")
    summary=dict(samples=100,requires_transcript_review=int(df.requires_transcript_review.sum()),
                 usable_word_times=int(df.usable_word_times.sum()),total_words=int(df.words.sum()),
                 thresholds=dict(mean_confidence=.2,word_low_confidence=.1,low_fraction=.5,diagnostic_wer=.8),
                 thresholds_chosen="检查实际异常后设定的保守质控规则；不以情感标签、预测成绩或人工词界调参",
                 interpretation="仅是需要复核的信号，不证明赛题标签错误，也不证明未触发规则者时间正确",
                 sensitivity=sf.to_dict("records"))
    save_json(ROOT/"results/alignment_quality_summary.json",summary)
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
