"""将论文衔接版的五组比较图改为 minipage，各图独立图题、编号和引用。"""
from pathlib import Path
import re
import shutil
from datetime import datetime

OUT = Path(__file__).resolve().parent / '结果分析与表格修订'
target = OUT / '实验结果分析_论文衔接版.tex'
source = target.read_text()
backup = OUT.parent / 'archive' / ('minipage修改前_' + datetime.now().strftime('%Y%m%d_%H%M%S'))
backup.mkdir(parents=True)
for name in [target.name, '问题一_整合稿_论文衔接版.tex', '实验结果分析_论文衔接版_预览.pdf']:
    shutil.copy2(OUT / name, backup / name)

groups = {
    'fig_p1_observation_summary': [
        ('fig_p1_visual_coverage', '面部特征的帧级与词级覆盖', 'fig:p1_visual_coverage'),
        ('fig_p1_audio_energy_share', '声学记录的低能量占比', 'fig:p1_audio_energy_share')],
    'fig_p1_rms_explained': [
        ('fig_p1_rms', '实例音频的短时RMS', 'fig:p1_rms'),
        ('fig_p1_rms_zoom', '同一时间范围的低幅值放大', 'fig:p1_rms_zoom')],
    'fig_p1_alignment_screening': [
        ('fig_p1_ctc_confidence', '词级CTC置信度分布', 'fig:p1_ctc_confidence'),
        ('fig_p1_diagnostic_wer', '片段级诊断WER分布', 'fig:p1_diagnostic_wer')],
    'fig_p1_window_experiment': [
        ('fig_p1_window_reconstruction', '响度的固定窗口聚合与重建', 'fig:p1_window_reconstruction'),
        ('fig_p1_word_window_overlap', '候选词区间与固定窗口交叠', 'fig:p1_word_window_overlap')],
    'fig_p1_window_tradeoff': [
        ('fig_p1_audio_window_error', '声学重建误差对照', 'fig:p1_audio_window_error'),
        ('fig_p1_vision_window_error', '视觉重建误差对照', 'fig:p1_vision_window_error'),
        ('fig_p1_word_crossing_ratio', '词跨界比例对照', 'fig:p1_word_crossing_ratio')],
}


def block(items):
    width = '.48' if len(items) == 2 else '.32'
    cells = []
    for filename, caption, label in items:
        assert (OUT / 'figures' / (filename + '_single.pdf')).exists()
        cells.append(r'''    \begin{minipage}[b]{WIDTH\linewidth}
        \centering
        \includegraphics[width=\linewidth]{figures/FILENAME_single.pdf}
        \caption{CAPTION}
        \label{LABEL}
    \end{minipage}'''.replace('WIDTH', width).replace('FILENAME', filename).replace('CAPTION', caption).replace('LABEL', label))
    return '\\begin{figure}[H]\n    \\centering\n' + '\\hfill\n'.join(cells) + '\n\\end{figure}'


text = source
for filename, items in groups.items():
    matches = [m for m in re.finditer(r'\\begin\{figure\}\[H\][\s\S]*?\\end\{figure\}', text)
               if f'figures/{filename}.pdf' in m.group()]
    assert len(matches) == 1, filename
    m = matches[0]
    text = text[:m.start()] + block(items) + text[m.end():]

replacements = {
    r'图~\ref{fig:p1_observation_summary}(a)': r'图~\ref{fig:p1_visual_coverage}',
    r'图~\ref{fig:p1_observation_summary}(b)': r'图~\ref{fig:p1_audio_energy_share}',
    r'图~\ref{fig:p1_rms_explained}以样本': r'图~\ref{fig:p1_rms}与图~\ref{fig:p1_rms_zoom}以样本',
    r'图~\ref{fig:p1_alignment_screening}显示': r'图~\ref{fig:p1_ctc_confidence}与图~\ref{fig:p1_diagnostic_wer}显示',
    r'固定窗口对照的处理过程见图~\ref{fig:p1_window_experiment}：': '固定窗口对照包含聚合与重建两个步骤：',
    '图(a)以响度展示重建结果': r'图~\ref{fig:p1_window_reconstruction}以响度展示重建结果',
    '图(b)检查同一样本中的词区间': r'图~\ref{fig:p1_word_window_overlap}检查同一样本中的词区间',
    r'图~\ref{fig:p1_window_tradeoff}汇总全量对照。图(a)、(b)中的短窗箱体整体低于长窗箱体':
        r'图~\ref{fig:p1_audio_window_error}至图~\ref{fig:p1_word_crossing_ratio}汇总全量对照。声学与视觉重建误差图中的短窗箱体整体低于长窗箱体',
    '图(c)则呈相反方向': r'图~\ref{fig:p1_word_crossing_ratio}则呈相反方向',
}
for old, new in replacements.items():
    assert text.count(old) == 1, old
    text = text.replace(old, new)
text = text.replace('沿用易读版全部图表。', '比较图采用minipage独立图题与编号排版。', 1)

# 表格和非图引用正文以外的用户修改均保留，不调用旧版全文生成脚本。
table_pattern = r'\\begin\{(?:table|longtable)\}[\s\S]*?\\end\{(?:table|longtable)\}'
assert re.findall(table_pattern, source) == re.findall(table_pattern, text)
labels = re.findall(r'\\label\{([^}]+)\}', text)
assert len(labels) == len(set(labels))
assert set(re.findall(r'\\ref\{([^}]+)\}', text)) <= set(labels)
assert not re.search(r'图\([abc]\)', text)
target.write_text(text)

integrated_path = OUT / '问题一_整合稿_论文衔接版.tex'
integrated = integrated_path.read_text()
start = integrated.index(r'\subsection{实验结果分析}')
end = integrated.index(r'\subsection{本问小结}', start)
integrated_path.write_text(integrated[:start] + text[text.index(r'\subsection{实验结果分析}'):] + '\n' + integrated[end:])
(OUT / 'minipage排版说明.md').write_text('''本次将五组比较图拆成11张独立PDF，在LaTeX中用minipage排列。
两图并排采用0.48\\linewidth，三图并排采用0.32\\linewidth；各图有独立caption和label，不设共同图题。
保留筛选流程图和共用时间轴的跨模态示例为完整图，正文引用同步更新。
最新正文中的人工筛选说明、全部表格、数值及章节顺序均保留。
图文件采用_single后缀；原合图文件保留供旧版使用。

绘图导出脚本：step1/paper_results/export_results_minipage.py。
修改前版本备份目录：''' + str(backup) + '\n')
print('已改为11张独立比较图；表格与最新正文修改保留。备份：' + str(backup))
