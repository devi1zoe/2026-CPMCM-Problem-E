"""仅重排本轮用户提供的完整问题一正文；校验未改动原有正文、公式与图表。"""
from pathlib import Path
from collections import Counter
import json
import re

HERE = Path(__file__).resolve().parent
SOURCE = HERE / '原始正文_用户提供.txt'
OUTPUT = HERE / '问题一_结构调整稿.tex'
TITLES = {
    '特征提取覆盖与数据质量': '样本覆盖完整性与特征有效性',
    '词级对齐的自动诊断与人工审核': '时序对齐质量评估与人工审核',
    '典型样本的时序对应与计算核验': '典型样本的跨模态时序对齐验证',
    '全量结果汇总与复现检查': '三模态特征文件说明与100条样本结果汇总',
}
WINDOW_TITLE = r'\subsubsection{时间粒度对照与信息保留}'
BRIDGE = '在上述对齐质量评估与人工审核之后，进一步从时间粒度角度分析特征的组织方式。\n\n'
FORWARD = r'以第~\ref{sec:p1_verification_examples}~节展示的典型样本为例，'
REPRO_HEADING = r'\subsubsection{方案可复现性说明}'+'\n'+r'\label{sec:p1_reproducibility}'+'\n\n'


def active(text):
    return '\n'.join(line for line in text.splitlines() if not line.lstrip().startswith('%'))


def main():
    original = SOURCE.read_text()
    typical_start = original.index(r'\subsubsection{典型样本的时序对应与计算核验}')
    window_start = original.index(WINDOW_TITLE)
    window_end = original.index('% 下列文件为本小节的组成部分', window_start)
    prefix = original[:typical_start]
    typical = original[typical_start:window_start]
    window = original[window_start:window_end]
    suffix = original[window_end:]
    assert window.count(WINDOW_TITLE) == 1
    moved = window.replace(WINDOW_TITLE+'\n', '', 1)
    anchor = r'\label{sec:p1_window_analysis}'+'\n\n'
    assert moved.count(anchor) == 1
    moved = moved.replace(anchor, anchor+BRIDGE, 1)
    figure_sentence = r'图~\ref{fig:p1_window_reconstruction}显示，'
    assert moved.count(figure_sentence) == 1
    moved = moved.replace(figure_sentence, FORWARD+figure_sentence, 1)
    changed = prefix + moved + typical + suffix
    for old,new in TITLES.items():
        old_command = r'\subsubsection{'+old+'}'
        assert changed.count(old_command) == 1
        changed = changed.replace(old_command, r'\subsubsection{'+new+'}', 1)
    repro_anchor = '实验环境与主要参数见表~'
    assert changed.count(repro_anchor) == 1
    changed = changed.replace(repro_anchor, REPRO_HEADING+repro_anchor, 1)
    OUTPUT.write_text(changed)

    # 将允许的修改逆向还原，要求与本轮原稿逐字符一致。
    restored = changed
    for old,new in TITLES.items():
        restored = restored.replace(r'\subsubsection{'+new+'}', r'\subsubsection{'+old+'}', 1)
    restored = restored.replace(REPRO_HEADING, '', 1)
    restored = restored.replace(BRIDGE, '', 1).replace(FORWARD, '', 1)
    result_window_start = restored.index(r'\label{sec:p1_window_analysis}')
    result_typical_start = restored.index(r'\subsubsection{典型样本的时序对应与计算核验}')
    result_window_end = restored.index('% 下列文件为本小节的组成部分',result_typical_start)
    # 删除原标题后留下的行内容从label开始；原标题紧跟一个换行。
    moved_restored = restored[result_window_start:result_typical_start]
    restored = (restored[:result_window_start] + restored[result_typical_start:result_window_end]
                + WINDOW_TITLE+'\n'+moved_restored + restored[result_window_end:])
    assert restored == original, '存在预期之外的正文变化'

    before,after = active(original),active(changed)
    preserved={}
    for env in ['equation','figure','table','longtable']:
        pattern = r'\\begin\{'+env+r'\}.*?\\end\{'+env+r'\}'
        old=Counter(re.findall(pattern,before,re.S));new=Counter(re.findall(pattern,after,re.S))
        assert old==new,env
        preserved[env]=sum(old.values())
    def content_numbers(text):
        text=re.sub(r'\\(?:subsubsection|label|ref|eqref)\{[^}]*\}', '', text)
        return Counter(re.findall(r'\d+(?:\.\d+)?',text))
    assert content_numbers(before)==content_numbers(after)
    oldlabels=Counter(re.findall(r'\\label\{([^}]+)\}',before))
    newlabels=Counter(re.findall(r'\\label\{([^}]+)\}',after))
    assert newlabels-oldlabels==Counter({'sec:p1_reproducibility':1}) and not oldlabels-newlabels
    assert max(newlabels.values())==1
    results=changed[changed.index(r'\subsection{实验结果分析}'):changed.index(r'\subsection{本问小结}')]
    heads=re.findall(r'^\\subsubsection\{([^}]+)\}',results,re.M)
    assert heads==list(TITLES.values())+['方案可复现性说明']
    p=original.index(r'\subsection{实验结果分析}')
    assert original[:p]==changed[:p]
    assert original[original.index(r'\subsection{本问小结}'):]==changed[changed.index(r'\subsection{本问小结}'):]
    figures=re.findall(r'\\includegraphics(?:\[[^]]*\])?\s*\{([^}]+)\}',after)
    refs=set(re.findall(r'\\(?:ref|eqref)\{([^}]+)\}',after))
    unresolved=sorted(refs-set(newlabels))
    original_unresolved=set(re.findall(r'\\(?:ref|eqref)\{([^}]+)\}',before))-set(oldlabels)
    assert set(unresolved)==original_unresolved
    report={'status':'passed','source':'原始正文_用户提供.txt','output':OUTPUT.name,
            'allowed_changes':{'renamed_headings':TITLES,'removed_standalone_heading':'时间粒度对照与信息保留',
                'added_heading':'方案可复现性说明','moved_block':'时间粒度对照整体移至人工审核之后、典型样本之前',
                'added_transition':BRIDGE.strip(),'added_forward_reference':FORWARD},
            'reversing_only_allowed_changes_restores_source_exactly':True,
            'before_results_section_unchanged':True,'final_summary_unchanged':True,
            'all_original_numbers_unchanged':True,'unchanged_environments':preserved,
            'result_headings':heads,'figure_paths_unchanged':True,'figure_paths':figures,
            'preexisting_external_references':unresolved,'new_unresolved_references':[]}
    (HERE/'结构调整核对.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ['allowed_changes','figure_paths']},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
