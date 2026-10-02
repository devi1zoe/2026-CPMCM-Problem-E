# 5.3声学特征原理与实际实例

2026-09-26：声学、视觉实例图已统一使用SimHei黑体、DejaVu Sans英文数字、柔和配色与简洁坐标框；保留原始视频帧、波形、特征与缺失标记。最新复现与配色说明见 `绘图说明.md`。

## 视觉支路补充

- `visual_section.tex`：完整视觉小节，含52维系数组成表、映射与缺失处理公式、实际样本分析，以及5.3末尾的三模态输出衔接段。需要 amsmath、amssymb、graphicx。
- `figures/fig_5_3_visual_example.pdf`：四帧实际画面、三组代表性系数及逐帧有效性标记；配图仅交付PDF。
- `build_visual_example.py`：读取已有NPZ并解码对应原视频作图，不重新运行MediaPipe，不改变原始特征。
- `visual_example_manifest.json`、`visual_example_values.csv`：展示样本、具体帧索引、分组维度与系数来源。

视觉示例编号为 `-wny0OAz3g8$_$9`，用于展示系数变化及缺失处理，属于说明性选例。31个抽样帧中29帧有可用特征；5.60和5.80秒位置无效。所展示的5.60秒画面仍可见人物，因此不将无效标记解释为画面无人。折线只连接相邻有效观测，缺失位置断开；不根据单个系数推断真实情绪。

复现命令：

```bash
/usr/local/python3.10/bin/python3 step1/paper_5_3/build_visual_example.py
```

## 声学支路

- `acoustic_section.tex`：声学小节正文。特征组成表前移，正文保留6组核心公式，并增加两组声学观测的对照表。需要 amsmath、amssymb、graphicx。
- `acoustic_appendix.tex`：谱斜率、LPC及极点换算、次谐波累加、Jitter/Shimmer的补充推导，共5组公式。
- `archive/acoustic_section_原理展开版.tex`：改写前的原理展开版备份。
- `figures/fig_5_3_acoustic_example.pdf`：论文用矢量图。删除图内总标题及底部重复说明，增大坐标字号；正式图片仅交付PDF。
- `acoustic_example_values.csv`：展示区间内的既有25维特征及时间记录。
- `acoustic_example_manifest.json`：来源和两组示例数值。
- `build_acoustic_example.py`：重绘脚本，直接读取已有结果，不重跑特征提取。

复制两份LaTeX与figures目录到论文工程根目录即可使用。在5.3相应位置使用 `\input{acoustic_section}`，在主文档已有的 `\appendix` 后使用 `\input{acoustic_appendix}`，以使正文中的附录交叉引用完整。附录文件本身不调用 `\appendix`。未修改5.2正文和原始特征。当前环境没有LaTeX编译器，已核对环境配对并实际查看生成图，尚未完成整篇论文编译。

默认绘图命令：

```bash
/usr/local/python3.10/bin/python3 step1/paper_5_3/build_acoustic_example.py
```

脚本仅读取已有特征，并解码示例音频作图，不重新运行特征提取或使用GPU。默认只输出PDF；可使用 `--preview /tmp/acoustic_preview.png` 生成内部检查用图片，不纳入交付。此次修改前的正文、脚本与PDF保存在 `archive/调整前_*/`。

公式说明的是核心运算，包含具体实现采用的参数与后处理说明。正文区分响度和MFCC支路的频带能量标度，补齐频率索引定义。Loudness不等于RMS，基频零标记不表示整行音频缺失，Shimmer分贝转换按openSMILE实际实现描述。正文明确：工具返回区间作为声学记录的时间支持近似，不是全部内部窗口与平滑运算的完整时间支持；时间与有效标记不计入25维特征。结尾指向“后续词级时序对齐阶段”。

参考依据：

- 实际项目配置：`../.venv/lib/python3.10/site-packages/opensmile/core/config/gemaps/v01b/GeMAPSv01b_core.lld.conf.inc`与`egemaps/v02/eGeMAPSv02_core.lld.conf.inc`。
- 梅尔滤波：[cMelspec官方文档](https://audeering.github.io/opensmile/_components/cMelspec.html)。
- MFCC：[cMfcc官方文档](https://audeering.github.io/opensmile/_components/cMfcc.html)及[官方实现](https://github.com/audeering/opensmile/blob/master/src/lldcore/mfcc.cpp)。
- 听觉加权与压缩：[cPlp官方文档](https://audeering.github.io/opensmile/_components/cPlp.html)。
- 基频：[cPitchShs官方文档](https://audeering.github.io/opensmile/_components/cPitchShs.html)。
- 周期扰动：[cPitchJitter官方实现](https://github.com/audeering/opensmile/blob/master/src/lld/pitchJitter.cpp)。

尚未执行人工词界标注；本图不含词界，也不将两处局部声学差异解释为情绪改变。
