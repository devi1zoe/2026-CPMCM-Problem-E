from pathlib import Path
import re
import shutil
import json

ROOT = Path('/mnt/data/math')
SRC = Path('/home/ubuntu/.codex/attachments/35e7a791-ee10-4c9b-9b44-cecf0bfe94d8/已粘贴的文本.txt')
OUT = ROOT / 'step1/paper_question1/结果分析与表格修订'
OUT.mkdir(exist_ok=True)
source = SRC.read_text()
lines = source.splitlines()

def old(a, b=None):
    return '\n'.join(lines[a - 1:(a if b is None else b)]).strip()

def cols(widths, aligns=None):
    aligns = aligns or ['c'] * len(widths)
    names = {'c': r'\centering', 'l': r'\raggedright'}
    return '\n        '.join(
        '>{' + names[a] + r'\arraybackslash}m{\dimexpr' + str(w) +
        r'\linewidth-2\tabcolsep\relax}' for w, a in zip(widths, aligns)
    )

def table(label, title, header, rows, widths, aligns=None, note='', sep=8):
    # Natural-width columns for short fields; only prose columns have wrap widths.
    specifications = {
        'tab:p1_acoustic_groups': r'l>{\raggedright\arraybackslash}p{17em}>{\raggedright\arraybackslash}p{10em}c',
        'tab:p1_acoustic_example': 'cccccc',
        'tab:p1_visual_groups': r'l>{\raggedright\arraybackslash}p{16em}>{\raggedright\arraybackslash}p{11em}c',
        'tab:p1_result_overview': 'lrr',
        'tab:p1_manual_review': r'lcc>{\raggedright\arraybackslash}p{17em}',
        'tab:p1_mapping_records': 'lcccc',
        'tab:p1_feature_specification': 'lccc',
        'tab:p1_files': r'l>{\raggedright\arraybackslash}p{24em}',
        'tab:p1_repro_settings': r'l>{\raggedright\arraybackslash}p{27em}',
    }
    sep = 6
    note_tex = (r'\par\smallskip' + '\n' +
                r'\parbox{\wd\pOneCompactTableBox}{\footnotesize ' + note + '}\n') if note else ''
    return (r'\begin{table}[H]' + '\n' +
            r'    \centering\small' + '\n' +
            '    \\caption{' + title + '}\n' +
            '    \\label{' + label + '}\n' +
            '    \\setlength{\\tabcolsep}{' + str(sep) + 'pt}\n' +
            r'    \renewcommand{\arraystretch}{1.12}' + '\n' +
            r'    \ifdefined\pOneCompactTableBox\else\newsavebox{\pOneCompactTableBox}\fi' + '\n' +
            r'    \begin{lrbox}{\pOneCompactTableBox}' + '\n' +
            '    \\begin{tabular}{' + specifications[label] + '}\n' +
            r'        \toprule' + '\n        ' + header + r'\\' + '\n' +
            r'        \midrule' + '\n' + '\n'.join('        ' + row for row in rows) + '\n' +
            r'        \bottomrule' + '\n' +
            r'    \end{tabular}' + '\n' +
            r'    \end{lrbox}' + '\n' +
            r'    \usebox{\pOneCompactTableBox}' + '\n' + note_tex + r'\end{table}')

def rows(a, b):
    return [x.strip().replace(r'\hline', '').rstrip() for x in lines[a-1:b] if x.strip()]

tables = {}
tables['tab:p1_acoustic_groups'] = table(
    'tab:p1_acoustic_groups', '25维声学特征的组成及含义',
    '类别 & 描述量 & 主要描述内容 & 维度',
    rows(214, 219) + [r'\midrule', r'合计 & & & 25\\'],
    [.13, .46, .32, .09], ['c', 'l', 'l', 'c'])
tables['tab:p1_acoustic_example'] = table(
    'tab:p1_acoustic_example', '两组声学观测的数值与有效性对照',
    '记录区间/s & 中心时刻/s & 响度 & 半音基频 & MFCC1 & 记录有效',
    [r'$[1.49,1.51]$ & 1.50 & 0.870 & 36.368 & 36.947 & 是\\',
     r'$[1.99,2.01]$ & 2.00 & 0.363 & $0^{\ast}$ & $-4.455$ & 是\\'],
    [.23, .17, .14, .17, .15, .14],
    note=r'注：$0^{\ast}$表示未输出有声基频时保留的零标记，不作为实际音高解释。', sep=6)
tables['tab:p1_visual_groups'] = table(
    'tab:p1_visual_groups', '52维面部系数的分组及代表性含义',
    '分组 & 代表性输出 & 主要描述内容 & 维度',
    [r'额眉部 & 压眉、内侧抬眉、外侧抬眉 & 眉部位置与形态 & 5\\',
     r'眼部 & 眨眼、睁大、眯眼及视线方向 & 眼睑状态与眼部运动 & 14\\',
     r'面颊与鼻部 & 鼓腮、面颊收紧、鼻部牵动 & 面中部形态 & 5\\',
     r'嘴部 & 嘴角上扬或下压、噘嘴、嘴唇收紧等 & 嘴唇及嘴角形态 & 23\\',
     r'下颌 & 张嘴、下颌前伸及左右移动 & 下颌位置与开合 & 4\\',
     r'其他输出项 & 工具字段\_neutral & 保留输出，不作为中性情感判据 & 1\\',
     r'\midrule', r'合计 & & & 52\\'], [.17, .41, .33, .09], ['c', 'l', 'l', 'c'])
tables['tab:p1_result_overview'] = table(
    'tab:p1_result_overview', '样本处理与三种模态的观测覆盖情况',
    '统计项目 & 数量 & 占比', rows(687, 691), [.56, .22, .22],
    note='注：样本比例以100为分母，词比例以1932为分母，各行不是互斥类别。观测覆盖表示相应位置存在可用于特征计算的记录。', sep=10)
tables['tab:p1_manual_review'] = table(
    'tab:p1_manual_review', '21条复核样本的人工审核结果',
    '审核发现 & 样本数 & 原文词数 & 主要表现', rows(817, 821) + [r'\midrule', r'合计 & 21 & 317 & \\'],
    [.27, .12, .14, .47], ['c', 'c', 'c', 'l'], sep=7)
tables['tab:p1_mapping_records'] = table(
    'tab:p1_mapping_records', '典型样本的部分词级映射记录',
    '原文词 & 子词索引 & 候选区间/s & 声学记录索引 & 视觉帧时刻/s',
    rows(922, 924), [.19, .12, .24, .18, .27],
    note='注：索引从0开始；声学索引指具有正聚合权重的原生特征记录，不是波形采样点。视觉帧按其时间支持区间与候选词区间的交叠关系参与聚合，帧中心不必位于词区间内部。', sep=6)
tables['tab:p1_feature_specification'] = table(
    'tab:p1_feature_specification', '三种模态的词级特征规格',
    '模态 & 特征维数 & 样本矩阵形状 & 对齐粒度',
    [r'文本 & 768 & $L\times768$ & 原文词\\',
     r'声学 & 25 & $L\times25$ & 原文词\\',
     r'视觉（面部） & 52 & $L\times52$ & 原文词\\'], [.27, .20, .29, .24],
    note=r'注：$L$为对应样本的原文词数。各样本按实际词数存储，填充数与截断词数均为0；视觉模态不可观测的词位置采用零向量占位，并附无效标记。', sep=10)

long = old(972, 1100)
long = long.replace(r'\setlength{\tabcolsep}{3pt}', r'\setlength{\tabcolsep}{6pt}')
long = long.replace(r'\renewcommand{\arraystretch}{1.16}', r'\renewcommand{\arraystretch}{1.12}')
long = long.replace(r'\begin{longtable}{@{}lrrrrrr@{}}',
    '\\setlength{\\LTleft}{\\fill}\n\\setlength{\\LTright}{\\fill}\n\\begin{longtable}{lrrrrrr}')
long = long.replace(r'\multicolumn{7}{@{}l}', r'\multicolumn{7}{l}')
long = long.replace(r'\shortstack{片段时长\\/ s}', r'\shortstack{片段\\时长/s}')
long = long.replace(r'\shortstack{音频时长\\/ s}', r'\shortstack{音频\\时长/s}')
long = long.replace(r'\shortstack{其中三种模态\\都可观测词数}', r'\shortstack{其中三种模态\\都可观测\\词数}')
long = long.replace('\\midrule\n\\multicolumn{7}{l}', '\\addlinespace[0.5em]\n\\multicolumn{7}{l}')
tables['tab:p1_all_samples'] = long

tables['tab:p1_files'] = table(
    'tab:p1_files', '特征结果文件及其内容', '文件 & 内容', rows(1113, 1118),
    [.32, .68], ['c', 'l'], sep=10)
tables['tab:p1_repro_settings'] = table(
    'tab:p1_repro_settings', '实验环境与主要参数', '项目 & 版本或参数', rows(1138, 1144),
    [.20, .80], ['c', 'l'], sep=10)

parts = []
def add(s):
    parts.append(s.strip())

add(r'''% 实验结果分析：完整替换稿，表格代码均已内嵌。
% 导言区依赖：amsmath、amssymb、graphicx、float、array、booktabs、longtable、url。
% 图片路径沿用 figures/；不要与“问题一_全部表格.tex”重复载入。
\subsection{实验结果分析}
\label{sec:p1_results}
\urlstyle{same}

本节围绕三个问题展开：特征是否覆盖全部样本，哪些词位置满足时序对应的使用条件，以及输出能否回溯至原始记录。首先统计特征输出与模态观测情况；随后在时序对齐质量评估中，依次分析自动筛选结果、固定窗口粒度的影响及人工审核发现；最后通过典型样本和全量计算核验检查映射关系，并汇总结果文件与复现条件。本问不训练情感预测模型，以下评价限于特征提取与时序组织。

\subsubsection{样本覆盖完整性与特征有效性}
\label{sec:p1_quality_scope}

100条样本均生成了三种模态的词级特征矩阵，各矩阵的行数与原文词数一致，共保留1932个词位置，未发生序列截断。在此基础上，表~\ref{tab:p1_result_overview}进一步统计各词位置是否具有可参与特征计算的模态记录。这里的有效性沿用前文的观测标记定义，词级时间对应的质量另行评估。''')
add(tables['tab:p1_result_overview'])
add(r'''全量共得到77261条声学记录和3973帧抽样画面。文本和声学在全部词位置均有可计算记录；视觉方面，3253帧成功提取面部特征，720帧未获得可用结果，占18.12\%。经词级聚合，1595个词具有可用面部观测，其余337个词缺少相应记录。由于一个词可以关联多帧，一帧也可能参与多个词的聚合，帧级与词级的缺失比例并不相同，图~\ref{fig:p1_visual_coverage}给出了两种统计口径的结果。

声学记录的观测覆盖完整，但输入幅度存在差异。以数字音频满幅1为基准，将短时均方根幅度（RMS）低于0.0031623的记录标记为低能量，共9918条，占12.84\%，如图~\ref{fig:p1_audio_energy_share}所示。该标记描述波形幅度较低，不作为观测缺失或无语音的判据。''')
add(old(734, 750))
add(r'''以样本-iRBcNs9oI8\$\_\$9为例，其333条声学记录中有147条属于低能量记录。图~\ref{fig:p1_rms}及其局部放大图~\ref{fig:p1_rms_zoom}展示了这些记录在时间轴上的分布，虚线表示阈值，橙点表示低能量记录。此外，两条样本的解码音频波形全为零，声学特征各维均不随时间变化。因此，全部词位置具有可计算的声学记录，并不意味着每条样本都包含与转写对应的语音；还需进一步检查文本与实际音频的对应关系。''')
add(old(754, 770))
add(r'''\subsubsection{时序对齐质量评估与人工审核}
\label{sec:p1_timing_quality}

本节先利用模型诊断指标筛查候选词区间，再将固定窗口对照作为时序组织的补充实验，最后通过人工审核解释整段对应异常。前两部分分别考察模型对词级对应的支持程度与时间划分的影响，人工审核则检查给定转写是否具有相应的语音依据。

CTC强制对齐为1932个词均生成了候选时间区间，但路径满足字符约束并不能保证真实词界准确。本文采用词级置信度与片段级诊断词错误率（Word Error Rate，WER）进行自动诊断。前者为最优CTC路径中属于该词的文字字符帧概率的几何平均，不包含空白符和词界符；后者为无转写约束的贪心识别文本与规范化转写之间的最少词编辑数除以参考词数。两项指标分别描述局部字符匹配与整段文本差异，均来自同一声学模型，不构成独立的准确性验证。

如图~\ref{fig:p1_ctc_confidence}所示，词级置信度均值为0.7477，中位数为0.9585，1034个词的分数不低于0.95，但仍有350个词低于0.1，占18.12\%。图~\ref{fig:p1_diagnostic_wer}中，诊断WER均值为0.4689，21条样本超过0.8，其中9条超过1；由于编辑数包含插入操作，WER可以大于1。这些分布说明，整体较高的词置信度并未排除局部异常或整段文本差异。两图分别统计1932个词和100条样本，虚线标示后续筛选使用的0.1和0.8阈值。''')
add(old(779, 795))
add(r'''据此，本文实施片段级与逐词两层筛选。首先，将平均词置信度低于0.2、置信度低于0.1的词超过一半、诊断WER大于0.8或无法形成完整CTC路径的样本列入整段复核，暂不采用其词级时间对应。其次，在其余样本中，仅将成功获得候选起止时间且置信度不低于0.1的词计为通过筛选。阈值是在检查实际异常后设定的保守条件，未利用情感标签或人工词界标注优化；筛选仅增加质量标记，不删除原文词、特征或候选时间记录。

四项片段级条件分别标记19、20、21和0条样本，前两项标记集合均包含在诊断WER大于0.8的集合中，最终需整段复核的样本为21条，涉及317个词。其余79条样本中，另有70个词因置信度不足而未通过逐词筛选。因此，1932个词中共有1545个通过对齐质量筛选，占79.97\%。片段级筛选还限制了37个局部置信度达到阈值的词，因而未通过筛选的387个词多于低置信度词的350个。

结合上一节的模态观测状态，在1545个通过筛选的词中，1387个词的三种模态均可观测，占全部词的71.79\%；其余158个词缺少有效面部记录。该交集明确了同时满足对齐筛选与三模态观测条件的词位置范围。

为检查阈值敏感性，将片段平均置信度阈值分别取0.1、0.2、0.3，诊断WER阈值分别取0.6、0.8、1.0，其余条件不变。在三种平均置信度阈值下，待复核样本数均随WER阈值变化依次为25、21、20条，说明当前数据的筛选数量主要受WER条件影响。这一对照描述复核范围的变化；CTC置信度尚未经人工词界标注校准，诊断WER也受识别误差影响，因此不能据此确定最优阈值或计算对齐准确率。''')
add(r'''% 固定窗口子实验属于本小节，不新增小标题或独立编号。
% 若其他位置引用此标签，其节号与 sec:p1_timing_quality 相同。
\label{sec:p1_window_analysis}

在明确候选词区间的筛选范围后，进一步从时间划分角度考察时序组织。作为补充实验，本文在相同原生特征上构造0.2~s和0.5~s固定窗口，对100条样本形成两组对照。窗口从零时刻开始，末窗截取至片段终点，视听特征沿用时间交叠加权方法。该实验考察聚合粒度的影响，不作为真实词界精度的验证。

评价包括两个方面：将窗口特征映回原生记录，以各维原生标准差归一化后的重建均方误差衡量局部特征变化的保留程度；在通过对齐筛选的词中，统计与两个及以上窗口交叠的词比例，衡量词区间被窗口边界切分的频率。两项指标均先在样本内计算，再对相应有效样本等权平均。重建误差仅纳入标准差大于$10^{-6}$的维度；有记录但全部维度恒定时记0，无有效记录时不计入模态均值。

选取未触发整段复核、面部观测帧占比不低于80\%且时长最接近10~s的样本-egA8-b7-3M\$\_\$1作为图示。该样本时长10.266~s，包含20个词，其中18个通过对齐筛选。图~\ref{fig:p1_window_reconstruction}显示，窗口平均削弱了响度的局部峰谷，0.5~s窗口的平滑范围更大。图~\ref{fig:p1_word_window_overlap}中，successful分别与4个0.2~s窗口和2个0.5~s窗口交叠，在两种划分下均计为跨界词。该指标只判断是否被切分，不区分交叠窗口数量及切分幅度。该词终点约为2.40247~s，与第四个短窗仍有约0.00247~s交叠；计算时将大于$10^{-8}\,\mathrm{s}$的交叠视为有效交叠，每个跨界词只计一次。''')
add(old(843, 859))
add(r'''全量结果见图~\ref{fig:p1_audio_window_error}至图~\ref{fig:p1_word_crossing_ratio}。窗口由0.5~s缩短至0.2~s后，声学和视觉平均标准化重建误差分别由0.7467、0.4373降至0.5449、0.2357，相对降幅为27.02\%和46.10\%；平均词跨界比例则由39.50\%升至71.33\%，增加31.83个百分点。声学误差基于100条样本，视觉误差基于具有有效面部记录的87条样本，词跨界比例基于未触发整段复核的79条样本。其中，跨界比例为样本等权均值，不是将全部词合并后的比例。''')
add(old(863, 886))
add(r'''箱线图的箱体为第25至第75百分位数，中线为中位数，须线延伸至1.5倍四分位距内的最远观测，须线外圆点仍参与统计，菱形为样本均值。逐样本比较发现，98条样本的短窗声学误差更低，另2条相同；87条样本的短窗视觉误差均更低；77条样本的短窗词跨界比例更高，另2条相同。

两条声学误差相同的样本即前述全零音频样本，其原生特征不随时间变化，两种窗口误差均按规则记0。将其单独列出后，其余98条样本的0.2~s、0.5~s平均声学误差分别为0.5560和0.7619，差异方向保持一致。因此，本次对照中短窗声学误差较低的结果并非由这两条退化输入造成。

这一子实验表明，在本次对照条件下，较短窗口保留了更多原生特征变化，同时也提高了词区间被切分的比例。词级表示与固定窗口分别提供按文本单元和统一时间尺度组织特征的方式；这里仅比较两种固定窗口，不能据此断言词级表示具有更低重建误差或更高情感预测性能。''')
add(r'''上述自动诊断与时间粒度对照分别明确了候选对应的筛选范围和窗口划分的影响，但尚未解释21条整段复核样本为何异常。为此，进一步对照实际音频与给定英文转写进行人工审核，检查片段内容是否对应。审核结果按主要问题归为五类，见表~\ref{tab:p1_manual_review}，各类互不重叠，共涉及317个原文词。''')
add(tables['tab:p1_manual_review'])
add(r'''人工审核表明，低置信度或高WER具有不同来源。音轨与文本不对应、缺少对应朗读等情况，使给定转写缺少相应的语音依据；非英语语音则与所用英语声学模型及英文对齐目标不匹配。其中，3条非英语样本均来自视频-yRb-Jum7EQ，两条全零音频样本归入“无对应朗读”。因此，这些异常不能统一解释为词界定位偏差，还需结合输入内容判断强制对齐的适用条件。

样本-HwX2H8Z4hY\$\_\$9体现了局部匹配与整段对应的差别。人工审核发现其音频存在卡顿，部分词语与文本一致，但整体内容并非逐词对应。该样本平均词置信度为0.7238，仅9.09\%的词低于0.1，却因诊断WER达到0.8636进入复核范围，说明局部置信度较高仍不足以确认整段对应关系。

依据审核结果，本文继续将这21条样本的整体词级对应标记为受限，并保留原文、候选时间与原生视听特征。审核未重建词级时间边界，因此不恢复局部词的对齐标记，前述筛选统计保持不变。此次人工审核仅覆盖自动筛出的21条样本，不能据此认定其余79条均无异常，也不能计算全量词界准确率。逐样本审核说明随结果附件保存。

\subsubsection{典型样本的跨模态时序对齐验证}
\label{sec:p1_verification_examples}

在明确词级对应的使用范围后，本节继续使用固定窗口子实验中的同一样本，检查词级输出与原始记录之间的映射。这里核验的是映射可追溯性与计算一致性，真实词界精度仍需人工时间标注评价。

图~\ref{fig:p1_multimodal_result}展示了该样本在1.80--3.60~s内的局部对应。successful的候选区间内约2.11~s处出现响度峰值；张嘴系数在2.8~s附近较大，随后降低。共同时间轴将这些变化与候选词区间及抽样画面联系起来，使词级表示能够回溯至具体的视听记录。图中虚线表示候选词界，抽样画面的宽度不表示持续时间，面部曲线连线仅辅助观察。''')
add(old(900, 905))
add(r'''表~\ref{tab:p1_mapping_records}列出部分映射记录。以successful为例，其候选区间为$[1.8425,2.4025]\,\mathrm{s}$，文本表示对应子词索引3，声学表示由183--240号共58条短时记录聚合，视觉表示由1.8、2.0、2.2、2.4~s四帧聚合。边界帧仅部分覆盖词区间，因而聚合权重相应较小。由保存的原生记录及权重，可以复算对应的词级向量。''')
add(tables['tab:p1_mapping_records'])
add(r'''除实例展示外，对全部样本的交叠权重、视听聚合值和CTC置信度进行独立复算，并重新计算诊断WER，解码音视频核对时间记录与低能量标记。100条样本均与保存结果一致，说明输出符合既定计算规则。典型样本中的in和1989仍因置信度不足而未通过逐词筛选，表明片段未触发整体复核，并不意味着其中所有词均可用于对齐分析。

\subsubsection{三模态特征文件说明与100条样本结果汇总}
\label{sec:p1_delivery}

为便于逐样本查阅，表~\ref{tab:p1_feature_specification}汇总三种模态的输出规格，表~\ref{tab:p1_all_samples}按词级对应的使用范围分组，列出全部100条样本的编号、时长、词数与有效位置统计。各组划分依据前述观测状态、对齐筛选和人工审核结果，高维特征的具体数值保存在对应文件中。''')
add(tables['tab:p1_feature_specification'])
add(r'''全量表中，“原文词数”对应三个矩阵的行数，“视觉可观测词数”统计能够聚合有效面部记录的词，“通过对齐筛选词数”采用前述两层筛选条件，最后一列统计通过筛选且三种模态均可观测的词数。由于文本与声学在全部词位置均有可计算记录，最后一列等于通过对齐筛选与视觉可观测两个条件的交集。片段时长与音频时长采用实际解码记录，不表示有效语音或面部信息的累计时长。

例如，样本-s9qJ7ATP7w\$\_\$7共有28个词，其中14个视觉可观测，24个通过对齐筛选，同时满足两项条件的有12个。交集数量不能取前两项数量的较小值代替。

按上述口径，32条样本的全部词通过筛选且三种模态均可观测，46条样本仅部分词满足两项条件，1条样本的全部词通过对齐筛选但视觉模态不可观测，另21条样本经人工审核后整体词级对应仍受限。全量表按这四类依次排列，已审核的21条样本统一归入最后一类，各组互不重叠。''')
add(tables['tab:p1_all_samples'])
add(r'''各样本按编号保存数值特征与对应记录，文件内容见表~\ref{tab:p1_files}。数值数组采用NPZ格式，原文、标签及描述信息采用JSON格式；样本、模态和词级明细另以CSV汇总。''')
add(tables['tab:p1_files'])
add(r'''读取时，通过样本编号定位目录，载入features.npz中的text、audio和vision数组，相同行对应同一原文词；\nolinkurl{word_bounds}保存候选起止时间。词级视听分析同时读取\nolinkurl{quality_masks.npz}中的模态有效位置标记。例如，\nolinkurl{vision_aligned_usable}取1表示该词通过对齐筛选且视觉可观测。固定窗口的文本表示可由保存的词权重与词级文本矩阵相乘获得。

\subsubsection{方案可复现性说明}
\label{sec:p1_reproducibility}

实验环境与主要参数见表~\ref{tab:p1_repro_settings}，完整运行命令、模型版本及处理日志随结果附件提供。''')
add(tables['tab:p1_repro_settings'])
add(r'''除前述100条样本的计算一致性核验外，还将两条独立试跑样本与正式运行结果逐项比较，所得特征的最大绝对差为0，表明这两条样本在相同条件下能够复现。该检查不等同于对全部样本重复运行实验。''')

results = '\n\n'.join(parts) + '\n'
(OUT / '实验结果分析_最终版.tex').write_text(results)

# Preserve the pasted document outside the requested results and table changes.
prefix = '\n'.join(lines[:661]) + '\n'
for a, b, key in [(204,223,'tab:p1_acoustic_groups'), (282,310,'tab:p1_acoustic_example'), (359,387,'tab:p1_visual_groups')]:
    assert old(a,b) in prefix
    prefix = prefix.replace(old(a,b), tables[key])
prefix = prefix.replace('    \\label{fig:video_time}', '    % \\label{fig:video_time} % 所属旧图已注释，避免标签误指。')
full = prefix.rstrip() + '\n\n' + results + '\n' + '\n'.join(lines[1169:]) + '\n'
(OUT / '问题一_整合稿.tex').write_text(full)

table_order = ['tab:p1_acoustic_groups','tab:p1_acoustic_example','tab:p1_visual_groups',
               'tab:p1_result_overview','tab:p1_manual_review','tab:p1_mapping_records',
               'tab:p1_feature_specification','tab:p1_all_samples','tab:p1_files','tab:p1_repro_settings']
table_text = '% 逐表替换用：不要和已内嵌这些表格的正文重复载入。\n\\urlstyle{same}\n\n'
table_text += '\n\n'.join('% ===== ' + key + ' =====\n' + tables[key] for key in table_order) + '\n'
(OUT / '问题一_全部表格.tex').write_text(table_text)
(OUT / '前文三张表_替换代码.tex').write_text('\n\n'.join(tables[key] for key in table_order[:3]) + '\n')

figdir = OUT / 'figures'
figdir.mkdir(exist_ok=True)
for name in re.findall(r'\\includegraphics(?:\[[^\]]*\])?\s*\{figures/([^}]+)\}', results):
    src = ROOT / 'step1/paper_results/figures' / name
    assert src.is_file(), name
    shutil.copy2(src, figdir / name)

preamble = r'''\documentclass[UTF8,a4paper,zihao=-4,fontset=none]{ctexart}
\usepackage[margin=2.2cm]{geometry}
\usepackage{amsmath,amssymb,graphicx,float,array,booktabs,longtable,url}
\usepackage[hidelinks]{hyperref}
\setmainfont{Times New Roman}
\setCJKmainfont[AutoFakeBold=2.5,AutoFakeSlant=0.2]{SimSun}
\setCJKsansfont{Noto Sans CJK SC}
\setCJKmonofont{Noto Sans Mono CJK SC}
\setlength{\emergencystretch}{2em}
\setlength{\LTpre}{0.5em}
\setlength{\LTpost}{0.5em}
\begin{document}
'''
(OUT / '实验结果分析_预览.tex').write_text(preamble + '\\section{问题一：多模态特征提取与时序对齐}\n\\input{实验结果分析_最终版.tex}\n\\end{document}\n')
(OUT / '全部表格_预览.tex').write_text(preamble + '\\section*{问题一表格排版预览}\n\\input{问题一_全部表格.tex}\n\\end{document}\n')

# Document-integrity checks; do not rerun or alter experiment data.
before_sample_rows = re.findall(r'^\\nolinkurl\{[^\n]+', old(976,1099), re.M)
after_sample_rows = re.findall(r'^\\nolinkurl\{[^\n]+', results, re.M)
assert len(before_sample_rows) == len(after_sample_rows) == 100
assert before_sample_rows == after_sample_rows
old_headers = re.findall(r'^\\subsubsection\{[^\n]+', old(662,1151), re.M)
new_headers = re.findall(r'^\\subsubsection\{[^\n]+', results, re.M)
assert old_headers == new_headers
labels = re.findall(r'\\label\{([^}]+)\}', results)
refs = re.findall(r'\\(?:ref|eqref)\{([^}]+)\}', results)
assert len(labels) == len(set(labels))
assert set(refs) <= set(labels)
assert results.index('构造0.2~s和0.5~s固定窗口') < results.index('审核结果按主要问题归为五类')
assert results.count(r'\begin{table}') == 6
assert table_text.count(r'\begin{table}') == 9
assert table_text.count(r'\begin{longtable}') == 1
checks = {'source':str(SRC), 'tables':10, 'all_100_sample_rows_unchanged':True,
          'original_subsubsection_titles_and_order_preserved':True,
          'fixed_window_before_manual_review':True,
          'results_refs_resolved_in_section':True,
          'results_table_count':7,
          'images':len(list(figdir.glob('*.pdf')))}
(OUT / '内容核对.json').write_text(json.dumps(checks, ensure_ascii=False, indent=2))
print(json.dumps(checks, ensure_ascii=False, indent=2))
