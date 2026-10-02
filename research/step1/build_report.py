"""从实际实验产物生成问题一结果文档，不写入预设的性能数值。"""
from common import ROOT
import json
import csv


def table(rows, columns):
    lines=["|"+"|".join(label for _,label in columns)+"|", "|"+"|".join("---" for _ in columns)+"|"]
    for row in rows:
        vals=[]
        for key,_ in columns:
            v=row[key]; vals.append(f"{v:.4f}" if isinstance(v,float) else str(v))
        lines.append("|"+"|".join(vals)+"|")
    return "\n".join(lines)


def main():
    """缺少全量核验时拒绝生成已完成口径的结果文档。"""
    r=ROOT/"results"
    metrics=json.loads((r/"metrics.json").read_text());t=metrics["totals"]
    audit=json.loads((r/"verification.json").read_text())
    assert audit["all_passed"] and audit["checked_samples"]==metrics["samples"]==100
    env=json.loads((r/"environment.json").read_text())
    examples=json.loads((r/"figure_samples.json").read_text())
    quality=json.loads((r/"alignment_quality_summary.json").read_text())
    repro=json.loads((r/"reproducibility.json").read_text())
    with open(r/"window_comparison.csv",encoding="utf-8-sig") as f:
        window_rows=list(csv.DictReader(f))
    counts={key:sum(bool(row[key]) for row in window_rows if row['window_seconds']=='0.2') for key in
            ('audio_normalized_reconstruction_mse','vision_normalized_reconstruction_mse','word_boundary_crossing_ratio')}
    sensitivity=table(quality['sensitivity'],[("mean_confidence_threshold","平均词置信度阈值"),
                      ("wer_threshold","WER阈值"),("flagged","待复核样本数")])
    windows=table(metrics["window_comparison"],[("window_seconds","窗口/秒"),("windows","平均窗口数"),
                 ("audio_normalized_reconstruction_mse","声学标准化重建MSE"),
                 ("vision_normalized_reconstruction_mse","视觉标准化重建MSE"),
                 ("word_boundary_crossing_ratio","平均词跨界比例")])
    lows=table(metrics["lowest_confidence_samples"],[("sample_id","样本"),("words","词数"),
               ("mean_word_confidence","平均CTC置信度"),("low_confidence_words","低置信度词数"),
               ("diagnostic_wer","转写一致性WER")])
    faces=table(metrics["lowest_face_coverage_samples"],[("sample_id","样本"),("sampled_video_frames","抽样帧数"),
                ("face_frame_ratio","人脸可观测比例"),("vision_observed_words","有视觉观测的词数")])
    def pct(a,b): return f"{100*a/b:.2f}%"
    text=f'''# 问题一实验结果与分析

## 1. 实验范围与结论

本次仅完成问题一：附件1全部100条视频的三模态特征提取、词级时间对齐、全量追溯与验证。没有训练情感分类器/回归器，没有使用附件2、3、4，没有开展问题二、问题三。以下数值均从本次实际输出计算，原始证据见`results/metrics.json`、汇总CSV、各样本NPZ/JSON和处理日志。

共处理100/100条视频，保留原始标签及转写。三模态在统一词位置输出768维文本、25维声学和52维面部表情特征。覆盖全部样本不代表每个词都具备可信音视频证据；缺失与低置信度均显式保留。

关键发现：**{quality['requires_transcript_review']}条样本触发转写一致性质控规则**，不能把它们的强制路径当作可靠词界。候选时间区间覆盖{t['words']}个词，但仅{quality['usable_word_times']}个词通过本次保守时间使用规则；这仍不是人工验证后的准确词数。风险样本保留原文、候选路径、原生音视频特征及固定窗口结果，不替换或删除赛题数据。

## 2. 原方案的实施和调整

采用Markdown首选的“词级强制对齐＋时间交叠聚合”。将示例工具MFA替换为冻结Wav2Vec2声学模型与标准CTC最大概率路径，保持“以给定转写为约束、显式词时间”的核心要求；无需另配发音词典。视觉用MediaPipe的52维blendshape代替OpenFace，明确二者语义不同，不声称得到OpenFace动作单元，也不强凑附件2的35维视觉特征。

文本使用冻结DistilBERT；声音使用openSMILE eGeMAPSv02低层描述量。0.2秒、0.5秒固定窗口作为额外对照，没有用均匀分配词时间替代CTC。所有网络均冻结，没有从100条标签拟合参数。

## 3. 实际运行环境

- GPU：{env['gpu']}；进程可见卡设置`CUDA_VISIBLE_DEVICES={env['cuda_visible_devices']}`。
- 解释器：`{env['python']}`；PyTorch `{env['torch_version']}`，CUDA `{env['cuda']}`。
- 文本和声学模型在GPU推理；媒体解码、openSMILE、MediaPipe及CTC路径计算在CPU进行。
- 原环境不修改；补充依赖使用已有项目venv，具体版本、权重哈希和代码哈希保存在环境记录中。
- 100条样本累计处理耗时：{metrics['elapsed_seconds']:.2f}秒。此数是逐样本耗时之和，不包含依赖下载、模型初始化和报告/核验时间，不是端到端墙钟时间。

## 4. 全量覆盖与观测质量

| 指标 | 实测结果 |
|---|---:|
| 已处理/应处理样本 | 100/100 |
| 解码后最长模态总时长 | {metrics['total_duration_seconds']:.3f}秒 |
| 解码后样本时长范围 | {metrics['duration_min']:.3f}～{metrics['duration_max']:.3f}秒 |
| 原视频解码帧数 | {t['decoded_video_frames']} |
| 5 Hz目标频率抽样帧数 | {t['sampled_video_frames']} |
| 原生声学窗口数 | {t['acoustic_frames']} |
| 保留原文词数 / 子词数 | {t['words']} / {t['subwords']} |
| 具有CTC候选时间区间的词 | {t['aligned_words']}（{pct(t['aligned_words'],t['words'])}） |
| CTC置信度低于0.1的词 | {t['low_confidence_words']}（占全部词{pct(t['low_confidence_words'],t['words'])}） |
| 文本可观测词位置 | {t['text_observed_words']}（{pct(t['text_observed_words'],t['words'])}） |
| 声学可观测词位置 | {t['audio_observed_words']}（{pct(t['audio_observed_words'],t['words'])}） |
| 视觉可观测词位置 | {t['vision_observed_words']}（{pct(t['vision_observed_words'],t['words'])}） |
| 未检出可用人脸的抽样帧 | {t['no_face_frames']}（{pct(t['no_face_frames'],t['sampled_video_frames'])}） |
| 多脸抽样帧 | {t['multiple_face_frames']} |
| 低能量声学窗口 | {t['silent_audio_frames']}（{pct(t['silent_audio_frames'],t['acoustic_frames'])}） |
| 填充位置 / 截断词 | {t['padding_count']} / {t['truncated_words']} |
| 需要转写复核的样本 | {quality['requires_transcript_review']} |
| 通过保守时间使用规则的词 | {quality['usable_word_times']}（{pct(quality['usable_word_times'],t['words'])}） |
| 全量features.npz大小 | {metrics['feature_bytes']/1024**2:.2f} MiB |

这里的词数由公开词规则确定，数字串在原文仍是一条词记录。时长取解码视频终点和音频终点中较大者。**实测时长范围与Word所述2.648～34.567秒不一致**：最短`-mJ2ud6oKI8$_$6`的容器时长为2.257秒，最长`-yRb-Jum7EQ$_$1`的容器时长为29.288021秒，独立读取容器也支持本次实测范围，不能把这一差异简单归因于小幅编码补齐。本次保留赛题提供的全部100个文件，使用实际PTS，不修改视频来迎合题干数字。完整媒体核查见`media_audit.json/csv`。静音、无人脸和缺失是不同状态。

平均词置信度为{metrics['mean_word_confidence']:.4f}，中位数为{metrics['median_word_confidence']:.4f}。完整CTC路径只说明给定转写可被放到一个合法单调路径上，**不等于每个词被正确对齐**。贪心ASR与给定转写的样本平均WER为{metrics['mean_diagnostic_wer']:.4f}，仅作为转写一致性诊断，不能代替时间误差。

质量规则为：平均词置信度<0.2，或低于0.1的词超过一半，或诊断WER>0.8，则片段进入待复核清单。此规则是在看到实际异常后制定，没有用情感标签或真实词界调参；按不同阈值检查，待复核数为{min(x['flagged'] for x in quality['sensitivity'])}～{max(x['flagged'] for x in quality['sensitivity'])}。严格使用时应读取`quality_masks.npz`，风险样本的词时间使用掩码全False；其他样本还需逐词通过0.1的置信度规则。原生特征和原文语义向量不受此掩码删除。

{sensitivity}

全量候选聚合表中的“可观测”只表示候选区间与有效帧有交叠，不能等同于词界正确。`alignment_quality_review.csv`提供全部100条的原文、ASR诊断、原因和分数，方便逐项回看。

![全量质量概览](results/figures/quality_overview.png)

## 5. 固定时间窗对照

对全部100条样本分别重算0.2秒、0.5秒时间窗，共200条比较记录。声学/视觉先按时间交叠聚合，再映回原生帧；每个样本内对非退化维度按原生有效帧标准差归一化，计算重建MSE，最后对样本求平均。无人脸或没有有效特征的样本不进入对应MSE均值。每种粒度声学MSE实际包含{counts['audio_normalized_reconstruction_mse']}条、视觉MSE包含{counts['vision_normalized_reconstruction_mse']}条；词跨界比例只在通过保守时间规则的词上计算，包含{counts['word_boundary_crossing_ratio']}条样本。没有可靠词时间的窗口不强行填入文本表示。

{windows}

这些指标描述时间平均带来的信号平滑和词被窗口边界切开的程度，**不是情感预测指标，也不证明某种时间粒度在下游任务最优**。0.2秒与0.5秒视觉结果还受到5 Hz抽样及近邻保持规则限制。窗口文本表示可由保存的词权重重建，未重复保存大矩阵。

本次结果显示，0.2秒窗口的声学、视觉重建MSE均低于0.5秒，较细时间划分保留了更多原生变化；但词跨界比例更高，固定短窗更容易切开词。这个实际权衡支持将词级时间作为可追溯主表示，同时保留物理时间窗供转写不可靠的片段使用，而不是宣布某一窗口全面更优。

![窗口比较](results/figures/window_comparison.png)

## 6. 典型对齐与困难样本

典型图样本：`{examples['typical']}`。选择规则为：在未触发片段级风险且抽样人脸可观测比例至少80%的样本中，选择时长最接近10秒者。该选择便于展示，不代表全体表现最好或最差。图中的视频是原始视频实际帧，标有真实PTS；声音曲线、词区间和表情系数使用同一秒级时间轴。`typical_feature_mapping.csv`逐词给出子词位置、文本向量范数、声学原始帧索引和视觉原帧PTS，完整特征及权重仍在NPZ中。

![典型样本对齐](results/figures/alignment_typical.png)

平均CTC置信度最低的5条样本如下，需优先回看给定转写、声画内容和词时间。低分原因尚不能仅凭分数确定，不能自动归因于某一噪声或转写错误。

{lows}

![低置信度样本](results/figures/alignment_low_confidence.png)

最低置信度例的原文为“Power is similar to strength”，ASR诊断却为“OF POWER EPOSE”；候选路径中部分词区间横跨低能量时段。它说明即使CTC能输出完整单调路径，也不能据此认定文本与当前声轨相符。该图中能看见人物，但5 Hz抽样未检出可用面部特征，因此这里的视觉无观测至少不能简单解释成“画面无人”。本次不据ASR擅自修改转写或标签。

人脸覆盖最低的5条样本如下。无人脸可能是画面确实没有可见脸，也可能是检测失败；代码只报告观测结果，不把两者混为一谈。

{faces}

另生成并查看了100条视频的中点实际帧总览。可以确认附件包含字幕卡、物体或风景镜头，而非始终是正面人脸；例如`-mJ2ud6oKI8$_$1`中点为器材画面，`-NFrJFQijFE$_$1`中点为山体画面。这只能证明所抽时刻的画面内容，不能据此断言整段都无人脸。面部特征缺失时原始视觉素材仍被保留，本实现没有把场景图像伪造成面部表情。

![100条原始视频中点实际帧](results/figures/source_overview.jpg)

全部100条可通过[本地回看页](results/alignment_review.html)查看，点击词按钮跳转至估计区间；橙色标记低置信度。回看页并不构成人工真值标注。本次未对全部词进行人工听辨打标。

## 7. 验证证据

- 三项核心单元测试覆盖CTC重复字符路径与穷举最优值一致、不可行路径拒绝、缺失情况下的交叠归一化。
- 全量100条核对了原始视频SHA256、原文、标签、样本ID，以及处理代码和配置哈希。
- 全量核对了词/子词回映、真实视频PTS索引、无填充/无截断记录、有限数值和掩码。
- 对每个样本的全部词位置，独立按区间交集重算声学和视觉聚合权重及结果；核验CTC状态转移、折叠目标和词时间来源。
- 核验通过数：{audit['checked_samples']}/100；机器可读记录为`results/verification.json`。
- 两条独立试跑与正式运行的全部NPZ字段进行复现对照，最大浮点绝对差为{max(s['max_absolute_difference'] for s in repro['samples']):.8f}，离散字段一致；记录为`results/reproducibility.json`。此检查只覆盖两条重复实验，不能推称已重复运行全部100条。

这些检查证明实现和记录的自洽性，没有把程序自洽冒充外部真值准确性。

## 8. 已知限制与后续接口边界

1. 没有人工词级时间标签，不能报告时间戳MAE、毫秒级对齐准确率或其他无证据指标。CTC分数低的词保留质量标记，不用预测文本纠正赛题转写。
2. 通用文本/声学表示、面部blendshape只是情感相关线索；没有验证它们能达到某个情感分类或回归性能。问题一不需要、此次也没有计算Accuracy、F1、MAE或Pearson预测成绩。
3. 5 Hz采样不能覆盖快速微表情；近邻保持区间是建模假设。多脸取最大脸没有说话者跟踪，也没有覆盖肢体动作的专门特征。
4. 标点不伪造发音时间，数字展开可能与实际读法不同；原文中的非发音说明也可能牵引强制对齐。记录可追溯但仍需针对性复核。
5. 模型权重和项目venv不属于小型交付特征包；它们保留在本地供复现。最终50MB限制适用于竞赛总附件，后续若加入其他问题材料还需重新核算。
6. 本问自提取的768/25/52维特征语义不同于附件2的768/74/35，不应直接替换附件2或混入后续任务训练。

已完成本轮问题一实现、实际运行、结果保存和分析，至此停止。未进入问题二及后续内容。
'''
    (ROOT/"实验结果与分析.md").write_text(text,encoding="utf-8")
    print("已根据100条实际结果生成 实验结果与分析.md")


if __name__=="__main__":
    main()
