"""将既有补充分析并入正文，保留三个与新版思路图相同的小标题。"""
from pathlib import Path
import shutil

HERE = Path(__file__).resolve().parent
ARCHIVE = HERE/'archive'/'并入正文前'
ARCHIVE.mkdir(parents=True, exist_ok=True)
for name in ['results_section.tex','results_supplement.tex','README.md']:
    if not (ARCHIVE/name).exists():
        shutil.copy2(HERE/name, ARCHIVE/name)
main=(ARCHIVE/'results_section.tex').read_text()
supp=(ARCHIVE/'results_supplement.tex').read_text()
main=main[main.index('\\subsection{'):]
main='% 正文与新版思路图的三个分析模块一一对应，无附录依赖。\n% 需要 amsmath、amssymb、graphicx；同目录放置 mapping_table.tex。\n'+main
old='本节按照技术路线中的结果分析流程，首先检查样本完整性、观测有效性和对齐时间质量，再将检查结果汇总为质量标记，通过计算核验和典型、困难样本对照说明其含义，最终明确结果的使用范围。'
new='本节从质量检查与使用范围、计算核验与实例分析、时间粒度对照三个方面评价输出结果，最后汇总特征及其质量记录。'
assert old in main
main=main.replace(old,new)
main=main.replace('% 总图：样本完整性。',r'''\subsubsection{质量检查与使用范围}
\label{sec:p1_quality_scope}

% 样本完整性、观测有效性、对齐时间质量，以及质量标记汇总。''')
old=r'改变所考察的平均置信度与WER阈值后，待复核样本数为20--25条，完整结果见附录~\ref{app:p1_result_supplement}。'
new=r'为考察阈值依赖性，将平均置信度阈值取0.1、0.2、0.3，诊断WER阈值取0.6、0.8、1.0，其余规则保持不变，结果见表~\ref{tab:p1_threshold_sensitivity}。'
assert old in main
main=main.replace(old,new)
a=supp.index('\\begin{table}')
b=supp.index('为考察时间粒度')
sensitivity=supp[a:b].strip()
main=main.replace('% 总图：汇总特征与质量记录。',sensitivity+'\n\n% 汇总特征与质量记录。')
main=main.replace('% 总图：计算与记录核验。',r'''\subsubsection{计算核验与实例分析}
\label{sec:p1_verification_examples}

% 计算与记录核验、典型与困难样本对照。''')
head,closing=main.split('% 总图：可追溯结果交付。',1)
delivery=closing.split('此外，针对交付中保留的')[0].strip()
window=supp[b:].strip()
main=head+r'''\subsubsection{时间粒度对照分析}
\label{sec:p1_window_analysis}

'''+window+'\n\n% 结果汇总与交付：收束段，不再单设小标题。\n'+delivery+'\n\n'+(
    '结合质量检查、计算核验、实例对照与粒度比较，本问完成了带有时间映射及质量信息的三模态表示。'
    '结果能够回溯至原文和原始声画记录，同时以质量掩码区分观测缺失与时间使用受限。'
    '其使用仍受到人工词界真值缺失、面部观测不完整以及视频抽样粒度的限制。\n')
assert '附录~' not in main and 'app:p1_result_supplement' not in main
assert main.count('\\subsubsection{')==3
(HERE/'results_section.tex').write_text(main,encoding='utf-8')
(HERE/'results_supplement.tex').write_text(
    '% 本文件内容已全部并入 results_section.tex，不再需要单独插入。\n'
    '% 请从主文档移除对本文件的 input；旧版本保存在 archive/并入正文前/。\n',encoding='utf-8')
print('已合并正文：三个小标题，阈值敏感性与固定窗口对照均已纳入，无附录引用。')
