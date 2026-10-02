# 问题一实验结果：人工审核与排版修订版

本版在前文衔接稿基础上加入用户提供的21条人工审核结果，将原先独立的“特征文件、全量结果与复现说明”并入同一个“实验结果分析”小节。先分析观测覆盖，再定义对齐质量筛选并报告1545与1387的统计结果，随后加入人工审核、实例、粒度对照及全量汇总。CTC与交叠加权采用对前文方法的文字引用，不重复推导，也不依赖前文章节的固定编号。

“1932”明确为原始转写词数；“时间使用规则”改为有具体条件的“对齐质量筛选”；“具有三模态观测”改为“三种模态都可观测”。配图保持独立PDF。2026-09-26更新的视觉覆盖率、声学能量占比、RMS、RMS放大、CTC置信度、诊断WER、窗口重建、词窗交叠八张图使用SimHei黑体、DejaVu Sans英文数字与柔和配色，详见 `指定配图说明.md`；多模态时间轴图的下方坐标与图例参照 `fig_5_3_visual_example` 使用SimHei中文与DejaVu Sans英文数字，上方帧时间标签和转写单词保留Times New Roman，可运行 `build_publication_figures.py --timeline-only` 单独重绘；其余图沿用各自设置。

图内保留坐标、刻度、图例、词文本和必要的时间标签；标题、样本编号、统计量、阈值含义及结论移入LaTeX正文。图片是独立PDF，两图并排由 `minipage` 实现，不预先合成为一张图片。

## 当前文件

- `results_section.tex`：唯一正文插入入口，末尾自动调用全量汇总片段。
- `paper_preview.pdf` / `paper_preview.tex`：完整预览及编译入口。
- `mapping_table.tex`：典型词的映射记录。
- `results_delivery.tex` / `all_samples_table.tex`：同一结果分析小节中的全量汇总、文件说明及复现设置，不再构成第二个subsection。
- `build_publication_figures.py`：全量配图入口；`plot_selected_results.py`可独立重绘上述八张图。
- `publication_figures_manifest.json`：所用字体、15张图、图中文字及数据依据。
- `配图调整说明.md`：每张图的用途、对应正文解释及排版方式。

## 15张独立PDF

以下文件全部位于 `figures/`。新材料包只包含这15张，不混入旧版拼图。

| 文件 | 内容 | 正文排版 |
|---|---|---|
| `fig_p1_observation_work.pdf` | work的实际画面与词、帧代表区间 | 三例通过三个minipage并排 |
| `fig_p1_observation_day.pdf` | day的实际画面与词、帧代表区间 | 同上 |
| `fig_p1_observation_similar.pdf` | similar的实际画面与词、帧代表区间 | 同上 |
| `fig_p1_visual_coverage.pdf` | 帧级、词级面部观测覆盖 | 与低能量占比独立并排 |
| `fig_p1_audio_energy_share.pdf` | 低能量记录的全量占比 | 同上 |
| `fig_p1_rms.pdf` | 实例短时RMS | 与纵轴放大图独立并排 |
| `fig_p1_rms_zoom.pdf` | 同一RMS的低幅范围 | 同上 |
| `fig_p1_ctc_confidence.pdf` | CTC词置信度分布 | 与WER独立并排 |
| `fig_p1_diagnostic_wer.pdf` | 诊断WER分布 | 同上 |
| `fig_p1_multimodal_timeline.pdf` | 同一时间轴上的实际帧、词、声音与面部运动 | 通栏；同一片段的关联记录保留在一幅图中 |
| `fig_p1_window_reconstruction.pdf` | 响度原生曲线与窗口重建曲线 | 与跨窗示意独立并排 |
| `fig_p1_word_window_overlap.pdf` | 同一词与两种窗口的交叠 | 同上 |
| `fig_p1_audio_window_error.pdf` | 声学窗口重建误差 | 与视觉误差、词跨界比例三个minipage同排 |
| `fig_p1_vision_window_error.pdf` | 视觉窗口重建误差 | 同上 |
| `fig_p1_word_crossing_ratio.pdf` | 词跨界比例 | 同上 |

单个观测实例内的“画面+时间区间”、同一词的两种窗口，以及同一片段的视听时间轴保留关联排布，其余均拆分。三组观测实例各自有PDF和caption，三图排列由LaTeX控制。

上述八张新图采用6.4×4.8英寸画布，标签14 pt、刻度12 pt；以下为其余图的原排版设置。原半栏统计图采用4.1×2.9英寸画布，三张箱线图改用2.65×2.9英寸画布，坐标标题13 pt、刻度12 pt，按0.48倍行宽插入后约为10 pt。三例图按0.32倍行宽插入，调整了独立画布与字号；通栏图的文字与半栏图视觉尺度接近。PDF固定画布导出，避免自动裁边造成同一行两幅图的字体比例、绘图区大小不一致。

## LaTeX插入

保持 `figures/` 相对路径，将本目录的LaTeX文件放在论文主文件目录。导言区需要：

```latex
\usepackage{amsmath,amssymb,graphicx,float,longtable,array,booktabs,url}
```

在方法部分之后插入：

```latex
\input{results_section.tex}
```

不要再次单独插入 `results_delivery.tex`，否则会重复输出全量汇总。正文采用如下独立图排版方式：

```latex
\begin{figure}[H]
    \centering
    \begin{minipage}[t]{0.48\linewidth}
        \vspace{0pt}
        \centering
        \includegraphics[width=\linewidth]{figures/fig_p1_ctc_confidence.pdf}
        \caption{词级CTC置信度分布}
        \label{fig:p1_ctc_confidence}
    \end{minipage}\hfill
    \begin{minipage}[t]{0.48\linewidth}
        \vspace{0pt}
        \centering
        \includegraphics[width=\linewidth]{figures/fig_p1_diagnostic_wer.pdf}
        \caption{片段级诊断WER分布}
        \label{fig:p1_diagnostic_wer}
    \end{minipage}
\end{figure}
```

预览采用 `[H]` 保持图文阅读顺序，并入完整论文时可以改为 `[htp!]`。图片内部没有标题，caption由论文模板排版。预览正文仍使用原有模板字体；图中已嵌入指定的两种字体，与正文模板无关。

## 数据与复现

配图重用原有数据；未更改模型、特征、候选词界或质量规则。原始全量核验见 `数据核验与补充说明.md`、`data_audit.json`，新增统计见 `analysis_statistics.json`。

在工作区根目录重绘：

```bash
OPENBLAS_NUM_THREADS=1 step1/.venv/bin/python step1/paper_results/build_publication_figures.py
```

绘图需要原视频、`step1/results`和系统已安装的宋体、Times New Roman。字体文件不随材料包分发；编译现成PDF无需安装绘图字体。旧绘图脚本用于历史版本，请勿作为本版绘图入口。

本次改图前的正文、PDF和旧拼图保存在 `archive/独立配图改版前/`。当前完整包为 `问题一_实验结果分析_人工审核版.zip`，`问题一_实验结果分析_推荐稿.zip`同步更新。另提供只含15张独立PDF及排版代码的 `问题一_结果独立配图.zip`。

## 100条样本清单的用途

清单用于逐样本核对交付和使用范围，不展示高维特征元素或情感分类结果。三模态矩阵规格单独说明，全量清单仅列样本之间不同的内容：原编号、片段/音频时长、原文词数、视觉可观测词数、通过对齐筛选词数，以及其中三种模态都可观测的词数。

四组互斥：全部词满足条件32条，部分词满足46条，未触发整段复核但无面部观测1条，整段对齐受限且已完成人工审核21条。最后一列是观测与对齐筛选条件的交集，不能由前两列取较小值替代。

- `build_sample_catalog.py`：清单生成入口。
- `sample_catalog_manifest.json`：100条清单的数值依据及分组统计。
- `全部样本特征交付清单.csv`：中文列名、完整精度的可检索清单。

工作区根目录运行 `step1/.venv/bin/python step1/paper_results/build_sample_catalog.py` 可重建清单。原表备份见 `archive/样本清单重排前/`。

## 本次核验与备份

重新读取100条样本的数值文件、质量标记与元数据，核对矩阵形状、四类分组以及1932、1595、1545、1387四项总数；复算各片段筛选条件的触发数。原始视频、CTC分数和窗口误差的独立核验沿用已有 `data_audit.json`，本次未改动特征或重新运行提取模型。修订前版本保存于 `archive/前文衔接修改前/`。

## 人工审核记录与本次排版

- `build_manual_review.py`：保存用户提供的原始问题说明，规范样本编号分隔符，与既有自动筛选记录逐一核对。
- `人工审核_21条样本.csv`：全部21条审核明细，包含原始说明、分类、自动指标及审核状态。
- `manual_review_manifest.json`：审核来源、编号对应、分类计数和核验结果。
- `manual_review_table.tex`：正文中的五类结果汇总，自动由主正文插入。

审核结果为：音轨与文本不对应11条（120词）、无对应朗读5条（64词）、非英语语音3条（93词）、分段内容不对应1条（18词）、音频卡顿且局部对应1条（22词），共21条、317词。这些是用户提供的人工内容审核结果，未添加新的人工词界标注，也未改写自动质量掩码。1545与1387的统计保持不变。原始“requires_transcript_review”字段仍表示自动筛选触发情况，CSV另设“manual_review_status”记录已审核，二者不能混为待办状态。

覆盖表的文本与声学分列两行，去掉“（各）”；小表采用自然列宽居中，说明类表格收窄列宽，表注同步收窄。三张箱线图保持三个独立PDF，用三个0.32行宽的minipage同排，重新设置画布以保持印刷字号。全文统一采用“视听分析”“视听特征”等表述。

本次修改前版本见 `archive/人工审核与排版修改前/`。`merge_into_main.py`只用于历史稿件合并，不是当前正文生成入口，请勿用其覆盖现稿。

## 正文排字

结果正文与图注已去除所有 `\texttt`。文件名和样本编号保留特殊字符与断行处理，但通过 `\urlstyle{same}` 随正文字体显示，不再使用等宽字形。排版修改不改变编号、文件名或实验统计。

## 与题目要求对应的目录

1. 样本覆盖完整性与特征有效性
2. 时序对齐质量评估与人工审核
3. 典型样本的跨模态时序对齐验证
4. 三模态特征文件说明与100条样本结果汇总
5. 方案可复现性说明

固定窗口对照已并入第2项，按自动诊断、人工审核、时间粒度对照的顺序展开，不再单列目录标题。第4项说明文件结构、读取方式及全部100条样本的结果，第5项集中说明可复现性。

## 三张窗口对照箱线图的最新样式

`fig_p1_audio_window_error`、`fig_p1_vision_window_error`、`fig_p1_word_crossing_ratio` 已使用独立Python脚本 `plot_window_statistics.py` 重绘，中文为SimHei、英文数字为DejaVu Sans，0.2秒窗口为蓝色、0.5秒窗口为暖橙色。图内保留坐标与窗口标签、样本数、均值标记、箱线与所有离群点；解释文字、均值数值及具体复现方法见 `窗口箱线图说明.md`。
