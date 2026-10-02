"""将100样本特征清单按实际词级使用范围分组，保留完整样本编号与时长。"""
from pathlib import Path
import csv
import json
from collections import Counter
import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / 'results'
GROUPS = [
    ('全部词满足', '全部词通过筛选且三种模态都可观测'),
    ('部分词满足', '部分词通过筛选且三种模态都可观测'),
    ('面部观测缺失', '全部词通过对齐筛选，但视觉模态不可观测'),
    ('整段对齐受限', '整段对齐受限（已完成人工审核）'),
]


def collect():
    records=[]
    for path in sorted((DATA/'samples').glob('*/metadata.json')):
        m=json.loads(path.read_text());quality=json.loads((path.parent/'quality.json').read_text())
        with np.load(path.parent/'features.npz') as z,np.load(path.parent/'quality_masks.npz') as q:
            n=len(m['words']);risk=bool(quality['requires_transcript_review'])
            t=q['timing_usable'];joint=q['text_aligned_usable'] & q['audio_aligned_usable'] & q['vision_aligned_usable']
            for mod,dim in [('text',768),('audio',25),('vision',52)]:assert z[mod].shape==(n,dim)
            assert z['text_valid'].all() and z['audio_valid'].all()
            assert m['padding_count']==m['truncated_words']==0
            j=int(joint.sum());v=int(z['vision_valid'].sum());u=int(t.sum())
            assert 0<=j<=min(v,u)<=n
            if risk:
                group='整段对齐受限';assert u==j==0
            elif j==n:group='全部词满足'
            elif j>0:group='部分词满足'
            else:
                group='面部观测缺失';assert v==0 and u==n
            records.append({'sample_id':m['sample_id'],'duration':m['duration'],'audio_duration':m['audio_duration'],
                'words':n,'vision_observed_words':v,'timing_usable_words':u,'joint_usable_words':j,
                'requires_review':risk,'group':group})
    assert len(records)==len({r['sample_id'] for r in records})==100
    audit=json.loads((HERE/'data_audit.json').read_text())['counts']
    for key,oldkey in [('words','words'),('vision_observed_words','vision_observed_words'),
                       ('timing_usable_words','time_usable'),('joint_usable_words','joint_usable')]:
        assert sum(r[key] for r in records)==audit[oldkey]
    counts=Counter(r['group'] for r in records)
    assert counts=={'全部词满足':32,'部分词满足':46,'面部观测缺失':1,'整段对齐受限':21}
    return records,counts


def write_table(records,counts):
    head=r'''样本编号 & \shortstack{片段时长\\/ s} & \shortstack{音频时长\\/ s} & \shortstack{原文\\词数} & \shortstack{视觉可观测\\词数} & \shortstack{通过对齐\\筛选词数} & \shortstack{其中三种模态\\都可观测词数}\\'''
    lines=[r'''% 逐样本特征交付清单：按使用范围分组；由build_sample_catalog.py生成。
\begingroup
\small
\setlength{\tabcolsep}{3pt}
\renewcommand{\arraystretch}{1.16}
\begin{longtable}{@{}lrrrrrr@{}}
\caption{附件1全部样本的特征输出与有效位置统计}\label{tab:p1_all_samples}\\
\toprule''',head,r'''\midrule
\endfirsthead
\multicolumn{7}{c}{表~\thetable\quad 特征输出与有效位置统计（续）}\\
\toprule''',head,r'''\midrule
\endhead
\midrule
\multicolumn{7}{r}{续下页}\\
\endfoot
\bottomrule
\endlastfoot''']
    for i,(group,title) in enumerate(GROUPS,1):
        lines.append(r'\multicolumn{7}{@{}l}{\textbf{'+f'{i}. {title}（{counts[group]}条）'+r'}}\\*')
        for r in records:
            if r['group']!=group:continue
            lines.append(r'\nolinkurl{'+r['sample_id']+'} & '+f"{r['duration']:.3f} & {r['audio_duration']:.3f} & {r['words']} & {r['vision_observed_words']} & {r['timing_usable_words']} & {r['joint_usable_words']}"+r'\\')
        if i<len(GROUPS):lines.append(r'\midrule')
    lines += [r'\end{longtable}',r'\endgroup']
    (HERE/'all_samples_table.tex').write_text('\n'.join(lines)+'\n')


def main():
    records,counts=collect();write_table(records,counts)
    ordered=[r for g,_ in GROUPS for r in records if r['group']==g]
    headers=['使用分组','样本编号','片段时长（秒）','音频时长（秒）','原文词数','视觉可观测词数','通过对齐筛选词数','其中三种模态都可观测词数','自动筛选是否触发整段复核']
    keys=['group','sample_id','duration','audio_duration','words','vision_observed_words','timing_usable_words','joint_usable_words','requires_review']
    with (HERE/'全部样本特征交付清单.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.writer(f);writer.writerow(headers)
        for r in ordered:writer.writerow([r[k] if k!='requires_review' else ('是' if r[k] else '否') for k in keys])
    output={'status':'passed','samples':100,'groups':dict(counts),'records':ordered,
            'totals':{k:sum(r[k] for r in records) for k in ['words','vision_observed_words','timing_usable_words','joint_usable_words']},
            'shared_specification':{'modalities':['text','audio','vision'],'dimensions':[768,25,52],'alignment':'original_transcript_word','padding':0,'truncated_words':0}}
    (HERE/'sample_catalog_manifest.json').write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in output.items() if k!='records'},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
