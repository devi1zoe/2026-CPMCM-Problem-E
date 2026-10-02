"""以本次上传的最终稿为基准，仅补充已提交附件的查阅索引。"""
from pathlib import Path
from zipfile import ZipFile
import re
import json
import hashlib

PROJECT = Path(__file__).resolve().parents[2]
SOURCE = Path('/home/ubuntu/.codex/attachments/b222b131-719f-43d8-a793-72f7871ef984/已粘贴的文本.txt')
OUT = PROJECT / 'step1/paper_question1/结果分析与表格修订'
text = original = SOURCE.read_text()
changes = []


def append(starts, addition, location, separate=False):
    global text
    matches = list(re.finditer(r'^' + re.escape(starts) + r'[^\n]*', text, re.M))
    assert len(matches) == 1, starts
    m = matches[0]
    addition = addition.strip()
    insertion = ('\n\n' if separate else '') + addition
    text = text[:m.end()] + insertion + text[m.end():]
    changes.append(dict(location=location, anchor=starts, latex=addition))


append('最后，检查样本是否完整、各模态特征是否有效，', r'''
本问自生成特征及配套材料统一随\nolinkurl{问题一_提交附件.zip}提交，以下简称“问题一提交附件”。下文所列文件路径均相对于解压后的\nolinkurl{问题一提交附件/}目录，文件导航见其中的\nolinkurl{附件说明.md}。
''', '问题分析与总体思路：最后一段之后', True)

append('附件1包含100条视频片段，分布于37个', r'''
样本编号、原始素材路径及源文件哈希详见问题一提交附件中的\nolinkurl{step1/results/input_manifest.json}，逐样本媒体检查记录见同目录下的\nolinkurl{media_audit.csv}。
''', '数据观察与预处理：首段末尾')

append('逐样本比较进一步表明，98条样本的短窗声学误差更低，', r'''
两种窗口的200条逐样本指标详见问题一提交附件中的\nolinkurl{step1/validation/audited_window_comparison.csv}，剔除恒定声学样本后的统计及相对变化量见同目录下的\nolinkurl{analysis_statistics.json}。
''', '固定窗口对照：逐样本比较段落末尾')

append('依据审核结果，本文维持这21条样本整体词级对应受限的标记，', r'''
逐样本的问题分类与审核说明详见问题一提交附件中的\nolinkurl{step1/results/人工审核_21条样本.csv}。
''', '人工审核：结论段末尾')

append(r'表~\ref{tab:p1_mapping_records}列出部分映射记录。', r'''
该样本全部词的映射明细见问题一提交附件中的\nolinkurl{step1/results/typical_feature_mapping.csv}；完整向量和聚合权重保存在该样本目录的\nolinkurl{features.npz}中。
''', '典型样本验证：映射表前的说明段末尾')

append('典型样本图与映射表共同展示了从原文词到模型子词、声学记录和视频帧的可追溯链条，', r'''
全量路径与聚合核验记录见问题一提交附件中的\nolinkurl{step1/results/verification.json}，媒体解码及各项统计的独立复算记录见\nolinkurl{step1/validation/data_audit.json}。
''', '典型样本验证：全量计算核验段末尾')

append('按上述口径，32条样本的全部词通过筛选且三种模态均可观测，', r'''
与表~\ref{tab:p1_all_samples}对应的电子清单见问题一提交附件中的\nolinkurl{step1/results/全部样本特征交付清单.csv}，可按样本编号检索和核对各项统计。
''', '全量结果汇总：100条样本长表之前')

append('为使全量汇总能够落实到具体文件，', r'''
全部100条样本的完整特征位于问题一提交附件的\nolinkurl{step1/results/samples/}下，每个以样本编号命名的子目录均保存\nolinkurl{features.npz}、\nolinkurl{metadata.json}、\nolinkurl{quality.json}、\nolinkurl{quality_masks.npz}及两种固定窗口结果；汇总CSV文件位于\nolinkurl{step1/results/}目录。
''', '特征文件说明：文件内容表之前', True)

append('读取时，先通过样本编号定位目录，', r'''
完整字段释义、掩码使用规则及可执行读取示例详见问题一提交附件中的\nolinkurl{特征字段与读取.md}。
''', '特征读取说明：段落末尾')

append('为使前述处理结果能够复现，本文同时记录工具与模型版本、关键参数、配置文件、模型文件哈希及处理日志。', r'''
实际环境记录、提取参数和冻结模型版本分别见问题一提交附件中的\nolinkurl{step1/results/environment.json}、\nolinkurl{step1/config.json}与\nolinkurl{step1/assets_manifest.json}。
''', '方案可复现性说明：环境参数表之前')

old = '实际执行顺序、命令及异常记录随代码说明和日志保存，21条样本的人工审核说明单独保留，以区分程序计算结果与人工内容核查。'
new = r'具体输入目录、运行顺序和执行命令详见问题一提交附件中的\nolinkurl{复现实验说明.md}，核心实现位于\nolinkurl{step1/}目录；逐样本处理日志及失败清单分别见\nolinkurl{step1/results/processing_log.jsonl}与同目录下的\nolinkurl{failures.json}。21条样本的人工审核说明单独保留，以区分程序计算结果与人工内容核查。'
assert text.count(old) == 1
text = text.replace(old, new)
changes.append(dict(location='方案可复现性说明：完整运行流程段的末两句', anchor=old, latex=new))

append('复现检查分别覆盖输入与实现的一致性、计算结果的一致性和重复提取的一致性。', r'''
两条样本的逐字段对照记录详见问题一提交附件中的\nolinkurl{step1/results/reproducibility.json}；附件文件的完整性可依据根目录下的\nolinkurl{文件校验清单.json}，通过\nolinkurl{核验附件.py}进行离线检查。
''', '方案可复现性说明：最后一段末尾')

# 所有新增路径均核对压缩包中的实际成员；同目录短名允许匹配唯一或逐样本文件。
archive = PROJECT / '问题一_提交附件.zip'
with ZipFile(archive) as z:
    members = [name.removeprefix('问题一提交附件/') for name in z.namelist()]
paths = sorted(set(re.findall(r'\\nolinkurl\{([^}]+)\}', '\n'.join(c['latex'] for c in changes))))
for path in paths:
    if path in ['问题一_提交附件.zip', '问题一提交附件/']:
        continue
    if path.endswith('/'):
        assert any(n.startswith(path) for n in members), path
    else:
        assert path in members or any(n.endswith('/' + path) for n in members), path

# url宏包将中文当作西文字形处理；仅英文路径使用nolinkurl，中文文件名用正文中文字体。
def chinese_path(match):
    path = match.group(1)
    found = re.search(r'[\u4e00-\u9fff]', path)
    if found is None:
        return match.group()
    prefix, name = path[:found.start()], path[found.start():]
    escaped = name.replace('_', r'\_')
    return (r'\nolinkurl{' + prefix + '}' if prefix else '') + r'\textnormal{' + escaped + '}'

for change in changes:
    raw = change['latex']
    rendered = re.sub(r'\\nolinkurl\{([^}]+)\}', chinese_path, raw)
    assert text.count(raw) == 1
    text = text.replace(raw, rendered, 1)
    change['latex'] = rendered

# 公式、所有图表环境和既有样本数值不作变动。
pattern = r'\\begin\{(figure|table|longtable|equation)\}[\s\S]*?\\end\{\1\}'
assert [m.group() for m in re.finditer(pattern, original)] == [m.group() for m in re.finditer(pattern, text)]
(OUT / '问题一_最终稿_附附件说明.tex').write_text(text)
start = text.index(r'\subsection{实验结果分析}')
end = text.index(r'\subsection{本问小结}', start)
(OUT / '实验结果分析_附附件说明.tex').write_text(text[start:end])
wrapper = (OUT / '实验结果分析_论文衔接版_预览.tex').read_text()
wrapper = wrapper.replace('实验结果分析_论文衔接版.tex', '实验结果分析_附附件说明.tex')
(OUT / '实验结果分析_附附件说明_预览.tex').write_text(wrapper)
(OUT / '附件说明_新增文字.tex').write_text('\n\n'.join('% 位置：' + c['location'] + '\n' + c['latex'] for c in changes) + '\n')
(OUT / '附件索引_核对.json').write_text(json.dumps({
    'source_sha256': hashlib.sha256(original.encode()).hexdigest(),
    'submission_zip_sha256': hashlib.sha256(archive.read_bytes()).hexdigest(),
    'changes': changes, 'referenced_paths_checked': paths,
    'figures_tables_equations_unchanged': True
}, ensure_ascii=False, indent=2) + '\n')
print(f'已补充{len(changes)}处附件指引，核对{len(paths)}个路径；原公式、图表保持不变。')
