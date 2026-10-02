"""由最终论文对应的既有结果整理问题一提交附件，不改动实验源文件。"""
from pathlib import Path
import csv
import hashlib
import json
import shutil

PROJECT = Path(__file__).resolve().parents[2]
SRC = PROJECT / 'step1'
OUT = PROJECT / '问题一提交附件'
ROOT = OUT / 'step1'


def copy(relative, target=None):
    destination = ROOT / (target or relative)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(SRC / relative, destination)


def write(name, text):
    (OUT / name).write_text(text.strip() + '\n', encoding='utf-8')


def main():
    OUT.mkdir(exist_ok=True)
    for name in ['plot_histograms.py', 'plot_relationships.py', 'figure_style.py', 'data_observation_samples.csv', '绘图说明.md',
                 'figures/fig_5_2_duration.pdf', 'figures/fig_5_2_transcript_length.pdf',
                 'figures/fig_5_2_duration_words.pdf', 'figures/fig_5_2_label_distribution.pdf',
                 'fonts/SimHei.ttf', 'fonts/source.json']:
        copy('paper_5_2/' + name)
    for name in ['build_acoustic_example.py', 'build_visual_example.py',
                 'acoustic_example_manifest.json', 'acoustic_example_values.csv',
                 'visual_example_manifest.json', 'visual_example_values.csv', '绘图说明.md',
                 'figures/fig_5_3_acoustic_example.pdf', 'figures/fig_5_3_visual_example.pdf']:
        copy('paper_5_3/' + name)
    copy('results/figure_samples.json')
    for name in ['plot_selected_results.py', 'selected_figures_manifest.json',
                 'annotation_preservation_check.json', '指定配图说明.md',
                 'plot_window_statistics.py', 'window_figures_manifest.json', '窗口箱线图说明.md']:
        copy('paper_results/' + name)
    for name in ['visual_coverage', 'audio_energy_share', 'rms', 'rms_zoom', 'ctc_confidence', 'diagnostic_wer',
                 'window_reconstruction', 'word_window_overlap', 'audio_window_error',
                 'vision_window_error', 'word_crossing_ratio']:
        copy(f'paper_results/figures/fig_p1_{name}.pdf')
    for relative in [
        'paper_5_2/figures/fig_5_2_source_examples.pdf',
        'paper_5_3/figures/fig_5_3_text_mapping.pdf',
        'paper_5_4/figures/fig_5_4_ctc_principle.pdf',
        'paper_5_4/figures/fig_5_4_overlap_principle.pdf',
        'paper_results/readable_figures/fig_p1_quality_flow.pdf',
        'paper_results/figures/fig_p1_multimodal_timeline.pdf',
    ]:
        copy(relative)
    for name in ['common.py', 'alignment.py', 'features.py', 'run.py',
                 'quality_audit.py', 'analyze.py', 'verify.py', 'audit_media.py',
                 'check_reproducibility.py', 'download_assets.py', 'test_alignment.py',
                 'config.json', 'assets_manifest.json', 'requirements-project.txt',
                 'requirements-project-resolved.txt']:
        copy(name)
    per_sample = ['features.npz', 'metadata.json', 'quality.json', 'quality_masks.npz',
                  'windows_0.2s.npz', 'windows_0.5s.npz']
    samples = sorted((SRC / 'results/samples').iterdir())
    assert len(samples) == 100
    for sample in samples:
        for name in per_sample:
            copy(f'results/samples/{sample.name}/{name}')
    for name in ['input_manifest.json', 'sample_summary.csv', 'modality_summary.csv',
                 'word_alignment.csv', 'processing_log.jsonl', 'failures.json',
                 'media_audit.csv', 'media_audit.json', 'quality_threshold_sensitivity.csv',
                 'alignment_quality_review.csv', 'alignment_quality_summary.json',
                 'window_comparison.csv', 'metrics.json', 'typical_feature_mapping.csv',
                 'verification.json', 'reproducibility.json']:
        copy('results/' + name)
    copy('paper_results/全部样本特征交付清单.csv', 'results/全部样本特征交付清单.csv')
    environment = json.loads((SRC / 'results/environment.json').read_text())
    # 软件版本与模型/代码哈希不变；不提交本机账户和安装路径。
    for key in ['python', 'torch_file']:
        environment.pop(key)
    (ROOT / 'results/environment.json').write_text(json.dumps(environment, ensure_ascii=False, indent=2) + '\n')
    for name in ['data_audit.json', 'audited_window_comparison.csv', 'analysis_statistics.json']:
        copy('paper_results/' + name, 'validation/' + name)

    # 独立复算入口沿用既有实现，用CSV核对全量表，不依赖论文草稿或旧图。
    audit = (SRC / 'paper_results/audit_paper_data.py').read_text()
    a = audit.index("for line in (HERE/'all_samples_table.tex')")
    b = audit.index('statistics={};paired={}', a)
    audit = audit[:a] + '''catalog=rows('全部样本特征交付清单.csv')
assert len(catalog)==100 and {r['样本编号'] for r in catalog}==set(labels)
for row in catalog:
 r=next(x for x in audit_rows if x['sample_id']==row['样本编号'])
 for col,key in [('片段时长（秒）','D'),('音频时长（秒）','Da'),('原文词数','L'),('视觉可观测词数','V'),('通过对齐筛选词数','U'),('其中三种模态都可观测词数','J')]:
  assert np.isclose(float(row[col]),r[key]),(row['样本编号'],col)
''' + audit[b:]
    a = audit.index('# 独立核对正文解读中新增的实例数值与旧图的实例清单。')
    b = audit.index("directory=DATA/'samples'/'-egA8-b7-3M$_$1'", a)
    audit = audit[:a] + '# 核对论文典型样本的原生记录映射。\nexamples={}\n' + audit[b:]
    (ROOT / 'validation/audit_results.py').write_text(audit)
    stats = (SRC / 'paper_results/check_analysis_statistics.py').read_text().replace(
        '核对学术精简稿新增的规则贡献和非退化样本统计，不改变原特征。',
        '核对论文中的筛选条件、非恒定样本统计及固定窗口差异。')
    (ROOT / 'validation/check_analysis_statistics.py').write_text(stats)

    with (SRC / 'paper_results/人工审核_21条样本.csv').open(encoding='utf-8-sig') as f:
        review = list(csv.DictReader(f))
    fields = ['sample_id', 'raw_text', 'manual_review_status', 'manual_category',
              'review_note', 'words', 'usable_word_times', 'word_boundaries_annotated']
    with (ROOT / 'results/人工审核_21条样本.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in review:
            record = {k: row[k] for k in fields if k != 'review_note'}
            record['review_note'] = row['original_review_note']
            writer.writerow(record)
    assert len(review) == 21 and sum(int(r['words']) for r in review) == 317

    write('附件说明.md', '''
# 问题一：多模态特征与时序对应附件

本附件提供附件1全部100条样本的自生成多模态时序特征、原生特征、时间对应记录、质量标记，以及处理代码、参数、日志和复现说明。样本编号采用 `video_id$_$clip_id`，与给定视频和标签一一对应；`$_$`为实际目录名中的字符。

## 文件导航

| 路径 | 内容 |
|---|---|
| `step1/results/samples/` | 100个样本目录，每个目录包含6个结果文件 |
| `step1/results/全部样本特征交付清单.csv` | 对应论文全量表的100条样本、媒体时长、词数及质量统计 |
| `step1/results/sample_summary.csv` | 100条样本的提取与诊断汇总 |
| `step1/results/modality_summary.csv` | 300条样本—模态记录，含维数、序列长度、有效位置数及特征路径 |
| `step1/results/word_alignment.csv` | 1932个原文词的候选时间及状态 |
| `step1/results/人工审核_21条样本.csv` | 21条复核样本的内容审核结论和说明 |
| `step1/results/processing_log.jsonl` | 全量处理日志，一条样本一条记录；失败清单为 `failures.json` |
| `step1/results/environment.json` | 实际运行环境、软件版本、配置与模型哈希 |
| `step1/results/verification.json`、`reproducibility.json` | 全量计算核验与两条样本重复提取对照记录 |
| `step1/validation/` | 独立复算程序、全量统计与200条固定窗口对照结果 |
| `step1/paper_5_2/` | 四张数据观察统计图PDF、Python绘图脚本、数据表及绘图说明 |
| `step1/paper_5_3/绘图说明.md`、`step1/paper_results/指定配图说明.md` | 声学、视觉实例与八张覆盖率、诊断及固定窗口图的PDF、Python绘图脚本、来源记录和复现说明 |
| `step1/paper_results/窗口箱线图说明.md` | 三张声学/视觉重建误差、词跨界比例箱线图及独立Python脚本、统计核对记录 |
| `step1/paper_5_2/figures/fig_5_2_source_examples.pdf`、`step1/paper_5_3/figures/`、`step1/paper_5_4/figures/`、`step1/paper_results/` | 六张原始样例、文本映射、CTC与交叠原理、质量筛选、多模态时间轴配图，绘制文字统一为纯黑，方框统一为直角 |
| `step1/*.py`、`config.json`、`assets_manifest.json` | 特征提取、对齐、质量检查的核心代码、参数及冻结模型版本 |
| `特征字段与读取.md`、`复现实验说明.md` | 字段定义、有效位置规则、读取方法与运行顺序 |
| `文件校验清单.json`、`核验附件.py` | 附件文件哈希、离线完整性检查入口 |

## 结果口径

全部100条样本共保留1932个原文词位置。文本、声学、视觉特征分别为768、25、52维，采用变长词级序列，无统一长度填充或截断。1545个词通过对齐质量筛选，其中1387个词同时具有三模态观测。原生声学记录77261条，抽样视频帧3973帧，其中3253帧具有可用面部特征。

样本覆盖、模态可观测和对齐质量是独立记录的状态。无面部观测的零向量仅占位；低能量音频不自动判为无效；整段对应受限的21条样本保留全部原文、标签和候选记录。筛选通过比例不作为词界准确率。媒体时长采用实际解码口径，不表示有效语音累计时长。

审核表中的 `review_note` 保留人工审核记录，`word_boundaries_annotated=False` 表示未标注人工词界；人工审核没有恢复受限样本的时间使用标记。固定窗口统计采用样本等权均值，详见 `step1/validation/data_audit.json` 与 `audited_window_comparison.csv`。

查阅已提交特征仅需Python和NumPy，无需原始视频或GPU。复现提取过程需使用赛题原始附件1，并按说明获取已锁定版本的公开预训练权重。原始视频、公开权重及软件环境不在本附件中重复存放；本问未训练情感预测模型。
''')
    write('特征字段与读取.md', '''
# 特征字段与读取

## 每条样本的文件

| 文件 | 内容 |
|---|---|
| `features.npz` | 全量词级与原生特征、时间戳、有效标记、聚合权重和CTC中间记录 |
| `metadata.json` | 原文、原始标签、原始素材相对路径与哈希、词和子词映射、特征名称及处理信息 |
| `quality.json` | 片段级诊断、复核原因与可用词数 |
| `quality_masks.npz` | 对齐筛选及各模态的联合有效位置标记 |
| `windows_0.2s.npz`、`windows_0.5s.npz` | 固定窗口边界、视听特征、有效标记与文本词权重 |

## 主要数组

记L为原文词数，Na为声学记录数，Nv为抽样帧数，S为子词数。所有索引从0开始；时间以秒为单位，以视频起点为共同原点。

| `features.npz`字段 | 形状或说明 |
|---|---|
| `text` / `audio` / `vision` | L×768 / L×25 / L×52，同一行对应同一原文词 |
| `word_bounds` | L×2，候选词起止时间；须结合对齐标记读取 |
| `text_valid` / `audio_valid` / `vision_valid` | L，模态内容或观测是否存在 |
| `alignment_valid` / `alignment_high_confidence` | L，是否产生候选时间 / 是否同时达到逐词置信度阈值；未包含片段级排除条件 |
| `word_confidence` | L，CTC路径文字字符帧概率的几何平均 |
| `subword_features` / `subword_ids` / `subword_to_word` | S×768 / S / S；独立标点的词归属为−1，仍参与文本编码 |
| `audio_frames` / `audio_bounds` / `audio_frame_valid` | Na×25 / Na×2 / Na，原生声学记录及时间支持近似 |
| `audio_rms` / `audio_silence` | Na，短时RMS / 低于0.0031622776601683794的标记；silence字段仅表示低能量 |
| `vision_frames` / `vision_pts` / `vision_bounds` | Nv×52 / Nv / Nv×2，面部系数、真实帧时刻及中点划分支持区间 |
| `vision_frame_valid` / `vision_face_count` | Nv，面部观测是否有效 / 检出人脸数 |
| `video_frame_indices` / `all_video_pts` | Nv，抽样帧对应原帧索引 / 全部解码帧时间戳 |
| `audio_weights` / `vision_weights` | L×Na / L×Nv，词区间与有效原生记录交叠时长归一化的权重 |
| `ctc_log_probs` / `ctc_bounds` | CTC输出时刻×字符表大小 / CTC输出时刻×2 |
| `ctc_state_path` / `ctc_target_ids` / `ctc_character_word` | 最优扩展状态路径 / 规范化目标字符ID / 字符对应原文词索引 |

全部数值特征以原文件精度保存。各模态的具体字段名称和顺序见 `metadata.json` 的 `audio_feature_names`、`vision_feature_names`；文本采用768维隐藏表示。`words`保存原文词、字符区间、子词索引和候选时间，`subwords`保存子词与字符位置记录。字符区间左闭右开。原文数字保持原样，仅在CTC目标副本中展开；诊断识别结果不替代给定转写。

## 有效位置与异常处理

`quality_masks.npz`中的 `timing_usable` 表示通过片段级与逐词两层时间筛选，`text_aligned_usable`、`audio_aligned_usable`、`vision_aligned_usable`分别为该标记与模态观测标记的交集。三者同时为真才是论文统计的1387个位置。

片段平均词置信度低于0.2、低于0.1的词超过一半、诊断WER大于0.8，或无法形成完整CTC路径，任一满足即整段时间对应受限。其余片段中的词须有候选时间且置信度不低于0.1。阈值未用情感标签或人工词界优化。

每条样本保存L行，不在末尾补齐至50。有效位置可能不连续，不能以有效数量替代掩码。缺失视听观测采用零向量占位，原词不删除；没有候选时间的词保留无效标记，不均分时间补造词界。本批100条样本全部生成候选时间，但不意味着均有真实对应朗读。

`windows_*.npz`中的 `bounds`保存窗口边界，`audio`、`vision`保存聚合向量，对应 `*_valid` 指示观测状态；`text_word_weights`乘词级文本矩阵得到固定窗口文本表示。其文本权重仅使用通过时间筛选的词。

## 读取示例

在附件根目录运行：

```python
from pathlib import Path
import json
import numpy as np

sample_id = '-egA8-b7-3M$_$1'
folder = Path('step1/results/samples') / sample_id
meta = json.loads((folder / 'metadata.json').read_text(encoding='utf-8'))
with np.load(folder / 'features.npz', allow_pickle=False) as z, \\
     np.load(folder / 'quality_masks.npz', allow_pickle=False) as q:
    text, audio, vision = z['text'], z['audio'], z['vision']
    usable = q['text_aligned_usable'] & q['audio_aligned_usable'] & q['vision_aligned_usable']
    k = next(i for i, w in enumerate(meta['words']) if w['original'] == 'successful')
    audio_records = np.flatnonzero(z['audio_weights'][k] > 0)
    vision_records = np.flatnonzero(z['vision_weights'][k] > 0)
    print(text.shape, audio.shape, vision.shape, usable.sum())
    print(meta['words'][k], z['word_bounds'][k])
    print(audio_records, z['vision_pts'][vision_records])
    print(z['video_frame_indices'][vision_records])
    rebuilt_audio = z['audio_weights'][k] @ z['audio_frames']
    np.testing.assert_allclose(rebuilt_audio, audio[k], rtol=1e-5, atol=1e-5)
```

该样本包含20个词，18个通过筛选且三模态可观测。successful的声学来源为183—240号记录，视觉帧时刻为1.8、2.0、2.2、2.4秒。源视频由 `source_path`定位，抽样帧由原帧索引定位；边界帧中心可在词区间外，其支持区间与词区间的交叠决定权重。
''')
    write('复现实验说明.md', '''
# 问题一复现实验说明

## 环境与模型

本次环境为Python 3.10.13、PyTorch 2.5.1+cu121、CUDA 12.1，使用RTX 4090 D完成冻结模型推理。主要依赖为NumPy 1.26.4、SciPy 1.15.3、PyAV 17.0.1、Transformers 4.48.3、openSMILE 2.6.0、MediaPipe 0.10.21。完整实测版本见 `step1/results/environment.json`；补充依赖版本见 `requirements-project*.txt`。后两份清单不包含全部基础环境，基础包版本以环境记录为准。

模型及版本：

- DistilBERT：`distilbert/distilbert-base-uncased`，commit `12040accade4e8a0f71eabdb258fecc2e7e948be`。
- Wav2Vec2：`facebook/wav2vec2-base-960h`，commit `22aad52d435eb6dbaf354bdad9b0da84ce7d6156`。
- MediaPipe Face Landmarker：float16，版本1；固定下载地址及全部文件SHA256见 `step1/assets_manifest.json`。

三个模型均冻结，不使用情感标签训练或微调。公开权重由 `download_assets.py`按固定版本获取；本问无自行训练的模型参数文件。执行下载前应保留提交的 `assets_manifest.json`，以其中SHA256核对获取结果。

## 输入目录

从本附件根目录运行命令。将赛题原始附件1放置在以下位置，保留原视频及标签文件：

```text
附件根目录/
├── step1/
└── E题_材料阅读/E题数据/附件1-数据集原始多模态样本/
    └── MOSEI数据集部分原始视频-100条/
        ├── label-100.xlsx
        └── <video_id>/<clip_id>.mp4
```

输入清单 `step1/results/input_manifest.json`记录全部样本标识、原文、标签、相对路径及源视频SHA256。程序核对100条标签和100个视频一一对应。无需附件2、3、4。

## 参数与处理规则

`step1/config.json`保存随机种子20260923、16 kHz单声道音频、约5 Hz实际视频帧抽样、两项人脸阈值0.5、词置信度阈值0.1、低能量RMS阈值0.0031622776601683794、0.2/0.5秒窗口及每块最多510个子词等参数。

文本保留原文与标点，字符位置用于子词回映；CTC另用规范化文本估计候选词界。声学采用25维eGeMAPSv02低层描述量；视觉保留52维面部系数，逐帧检测，多脸取面积最大者，无可用人脸时保留位置和无效标记。最大人脸未验证为说话者。时间聚合按有效交叠长度归一化；声学记录支持区间是时间组织近似，不能解释为各描述量完整分析窗。完整字段及筛选条件见《特征字段与读取》。

## 运行顺序

仅查阅、检查提交文件时，使用已安装NumPy的Python执行：

```bash
python 核验附件.py
```

重跑时在附件的工作副本上操作，避免覆盖提交记录。选择已配置上述依赖的Python解释器；补充依赖仅安装到工作副本的项目环境。命令中的 `python`统一指向该解释器。`common.py`同时支持从 `step1/.venv/lib/python3.10/site-packages`加载项目补充依赖。

```bash
# 获取固定版本权重，核验输入并检查基础对齐逻辑
python step1/download_assets.py
python step1/audit_media.py
python -m unittest discover -s step1 -p 'test_alignment.py' -v

# 两条独立试跑和全部100条正式提取
python step1/run.py --device cuda:0 --limit 2 --output step1/smoke
python step1/run.py --device cuda:0
python step1/check_reproducibility.py

# 生成质量标记、汇总表及窗口对照，复核输入与聚合计算
python step1/quality_audit.py
python step1/analyze.py
python step1/verify.py
python step1/validation/audit_results.py
python step1/validation/check_analysis_statistics.py
```

上述提取命令使用一张可见GPU；可通过 `CUDA_VISIBLE_DEVICES`选择设备。正式提取可添加 `--resume`，仅当源视频、配置、代码和模型哈希一致时跳过已有样本。为保持核验哈希一致，核心提取代码和配置按原运行版本随附。单独的绘图输出不影响NPZ特征与质量标记。

人工审核文件记录已完成的21条内容核查，未标注人工词界；不会由自动程序生成新的人工结论。`全部样本特征交付清单.csv`与论文全量表一致，可由独立复算入口逐项检查。NPZ、原文和标签均保留全部100条样本。

## 核验记录的范围

`verification.json`记录100条样本的输入、代码、配置与模型哈希核验，以及路径和聚合复算；`validation/data_audit.json`记录媒体解码、RMS、CTC置信度、诊断WER、质量掩码及固定窗口指标的独立复算。`validation/analysis_statistics.json`记录筛选条件重叠、恒定声学样本与窗口差异统计。

`reproducibility.json`为两条独立试跑与正式提取的全部NPZ字段对照，最大绝对差为0。该检查范围为两条样本，并非全量重新提取或跨环境完全一致的保证。计算一致性及诊断筛选不替代人工词界准确性评价。
''')
    print(OUT)


if __name__ == '__main__':
    main()
