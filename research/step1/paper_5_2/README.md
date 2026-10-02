# 问题一5.2：数据观察与预处理

本目录补充论文材料，不修改既有特征文件，不运行或重新训练模型。

## 使用方式

- `section_5_2.tex`：可直接纳入正文的5.2，含四张独立统计图及一张画面示例总图，不设置三级标题。前四张图按两组minipage并排：时长与词数分布一组，联合分布与类别构成一组；每图占0.48倍行宽，均保留独立caption、图号和label。画面总图单独占整行。
- `optional_raw_timeline.tex`：可选声画图及正文，可插入预处理段之后；也可移至附录。
- 将 `figures/` 复制到论文工程，保持 LaTeX 中的相对路径。导言区需要 `\usepackage{graphicx}`。章号随主文档自动生成。
- 按最新要求，交付包中的配图仅提供PDF；后续图像交付也只提供PDF。本地保留的PNG用于预览，SVG作为已有源文件存档，不再打包交付。2026-09-25更新的四张统计图使用中文SimSun（宋体）、英文与数字Times New Roman，坐标轴标签19 pt、刻度和柱上标注17 pt；画面示例也采用SimSun与Times New Roman，声画图保留原字体。四张统计图的字体与字号已按2026-09-26的新要求改为SimHei黑体和默认英文无衬线字体，标签14 pt、刻度12 pt、图例11 pt。

## 新增图与数据来源

1. 四张统计图：`fig_5_2_duration`（时长）、`fig_5_2_transcript_length`（转写词数）、`fig_5_2_duration_words`（时长与词数联合分布）、`fig_5_2_label_distribution`（原始类别构成）。2026-09-26两张直方图分别改为柔和蓝色（#4C78A8）与暖橙色（#E69A52），采用中文黑体、默认英文无衬线字体、普通四边坐标框和图例，并保留中位数虚线；由独立脚本 `plot_histograms.py` 绘制。类别柱形按负向、中性、正向分别使用柔和橙色（#E69A52）、蓝色（#4C78A8）、绿色（#68A18A）与黑色细边框，柱顶仅标注样本数；散点图改用蓝色（#4C78A8）圆点，两图均使用SimHei黑体与普通四边坐标框。没有新增情感推断或相关性显著性检验。旧版四合一图保存在 `archive/`，不再用于正文或新版材料包。
2. `fig_5_2_source_examples`：六条原视频的实际中点帧与给定转写节选，保持整张总图，转写放入统一浅色文本框，编号与时间在框外独立显示。样本由已有100帧总览中按画面差异选取，属于说明性选例，不代表随机抽样或场景比例。完整原文、帧索引与实际时刻保存在 `source_examples_manifest.json`。
3. `fig_5_2_raw_timeline`：已有典型样本的实际视频帧、音频波形与完整给定转写；仅说明预处理后的时间记录，不展示或假定词级对齐。波形包络仅用于绘图，不改变原始音频。

统计读取 `step1/results/media_audit.json` 与 `input_manifest.json`；此次重新读取原始 `label-100.xlsx`，核对所有文本、标签和编号，核对100个视频文件路径，并将词数、时长与既有 `sample_summary.csv` 逐条比较。具体结果在 `data_observation_stats.json` 和 `data_observation_samples.csv`。

原有全量解码来自 `step1/audit_media.py`，原有中点帧总览来自 `step1/preview_sources.py`，文本计词规则沿用 `step1/alignment.py`，音视频预处理沿用 `step1/features.py`。此次仅重新解码作图所需的原始片段。

## 写作口径

- 实测时长为2.257至29.288秒，与题干的2.648至34.567秒不同；正文脚注已注明，未修改原数据。
- 文本词数按字母词（允许内部撇号）及数字串计数，不是空格切分数，也不是DistilBERT子词数；说明性文字仍属于给定原文，不代表均在声轨中发音。
- 类别构成直接来自原始annotation字段，问题一没有按类别重采样或训练模型。
- 每个样本中点的一帧不能代表该片段全部内容，不能据此统计整段“无人脸比例”。
- 画面与整段转写并列不代表其中某个词对应所显示画面的时刻。
- 图中帧时刻来自原始视频显示时间戳；没有生成或美化原视频画面。
- 代码运行及图像检查不需要GPU，也不改动根Python环境。

## 复现

在项目根目录运行：

```bash
/usr/local/python3.10/bin/python3 step1/paper_5_2/build_data_observation.py
```

依赖复用现有环境与 `step1/.venv`。未安装新依赖。

仅重绘两张直方图（读取既有CSV，不重新处理数据）：

```bash
/usr/local/python3.10/bin/python3 step1/paper_5_2/plot_histograms.py
```

画面示例仍使用SimSun与Times New Roman。四张统计图均由独立脚本绘制，共用SimHei字体注册函数。独立直方图脚本采用黑体SimHei，以及DejaVu Sans英文数字字体；SimHei字体文件位于fonts目录，禁止回退到其他中文字体，详见`绘图说明.md`；默认仅输出PDF，添加`--formats pdf png svg`可更新全部本地格式。原入口`build_data_observation.py --histograms-only`已同步调用该脚本。当前字体下载来源及SHA256保存在该目录的`sources.json`。修改前脚本和两张图保存在`archive/before_histogram_font_update_20260925/`。

本地可通过 `preview.html` 直接浏览六张PNG预览；该页及PNG不包含在仅PDF配图的交付包中。当前环境未发现LaTeX编译器，正文未编译成整篇PDF；配图PDF由绘图库直接导出。

仅更新时长词数散点图与类别分布图：

```bash
/usr/local/python3.10/bin/python3 step1/paper_5_2/plot_relationships.py
```

散点图已删除“每个点对应一条视频片段”的图内说明。原入口`build_data_observation.py --relationships-only`已同步调用独立脚本。

原始画面示例图已统一为宋体与Times New Roman，子图标题22 pt，编号、时间、转写17 pt；时间信息分行，转写最多两行，超出部分以省略号表示，文本框同步缩小。六幅原始帧与完整转写记录不变。单独重绘命令：

```bash
/usr/local/python3.10/bin/python3 step1/paper_5_2/build_data_observation.py --source-examples-only
```
