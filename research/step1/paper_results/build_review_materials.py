"""为按总图组织的结果分析生成诊断图和逐词映射表，只读取既有实验输出。"""
from pathlib import Path
import csv
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

HERE = Path(__file__).resolve().parent
DATA = HERE.parent/'results'
FONT = '/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc'
matplotlib.font_manager.fontManager.addfont(FONT)
plt.rcParams.update({'font.family':FontProperties(fname=FONT).get_name(),
    'font.size':10.5,'axes.unicode_minus':False,'pdf.fonttype':3,
    'text.parse_math':False,'axes.spines.top':False,'axes.spines.right':False})
with (DATA/'alignment_quality_review.csv').open(encoding='utf-8-sig') as f:
    rows=list(csv.DictReader(f))
confidence=[]
for p in sorted((DATA/'samples').glob('*/features.npz')):
    with np.load(p) as z: confidence.extend(z['word_confidence'].tolist())
c=np.array(confidence)
wer=np.array([float(r['diagnostic_wer']) for r in rows])
assert len(c)==1932 and len(wer)==100 and int((c<.1).sum())==350
fig,axs=plt.subplots(1,2,figsize=(9.3,3.6))
fig.subplots_adjust(left=.08,right=.985,bottom=.18,top=.77,wspace=.30)
axs[0].hist(c,bins=np.linspace(0,1,21),color='#52997B',edgecolor='white',linewidth=.6)
axs[0].axvline(.1,color='#B76345',ls='--',lw=1.5)
axs[0].set(xlabel='词级CTC置信度',ylabel='词数',xlim=(0,1))
axs[0].set_title('（a）1932个词的置信度分布',loc='left',fontsize=11.5,pad=35)
axs[0].text(.5,1.05,'置信度 < 0.1：350个词（18.12%）',ha='center',transform=axs[0].transAxes,fontsize=10)
axs[1].hist(wer,bins=np.linspace(0,2,21),color='#4778A8',edgecolor='white',linewidth=.6)
axs[1].axvline(.8,color='#B76345',ls='--',lw=1.5)
axs[1].set(xlabel='ASR与给定转写的诊断WER',ylabel='样本数',xlim=(0,2))
axs[1].set_title('（b）100条样本的转写一致性诊断',loc='left',fontsize=11.5,pad=35)
axs[1].text(.5,1.05,'虚线：诊断WER阈值0.8',ha='center',transform=axs[1].transAxes,fontsize=10)
for ax in axs:
    ax.grid(axis='y',alpha=.18)
    ax.set_axisbelow(True)
fig.savefig(HERE/'figures/fig_p1_alignment_diagnostics.pdf',bbox_inches='tight')
fig.savefig(HERE/'previews/fig_p1_alignment_diagnostics.png',dpi=170,bbox_inches='tight')
plt.close(fig)

# 从实际保存的权重读取“参与聚合”的记录，而不是根据帧中心推测对应关系。
p=DATA/'samples/-egA8-b7-3M$_$1'
m=json.loads((p/'metadata.json').read_text())
records=[]
with np.load(p/'features.npz') as z:
    for wid in [1,3,6]:
        w=m['words'][wid]
        a=np.flatnonzero(z['audio_weights'][wid]>0)
        v=np.flatnonzero(z['vision_weights'][wid]>0)
        assert np.array_equal(a,np.arange(a[0],a[-1]+1))
        records.append({'word':w['original'],'subword_indices':w['subword_indices'],
            'start':float(z['word_bounds'][wid,0]),'end':float(z['word_bounds'][wid,1]),
            'confidence':float(z['word_confidence'][wid]),
            'audio_first':int(a[0]),'audio_last':int(a[-1]),
            'video_pts':[float(t) for t in z['vision_pts'][v]]})
tex=[r'''% 自动读取既有特征文件生成；由 results_section.tex 插入。
\begin{table}[H]
    \centering
    \caption{典型样本的部分词级映射记录}
    \label{tab:p1_mapping_records}
    \begingroup
    \small
    \setlength{\tabcolsep}{3pt}
    \renewcommand{\arraystretch}{1.2}
    \begin{tabular}{@{}l c c c l@{}}
        \hline
        原文词 & 子词索引 & 候选区间/s & 声学记录索引 & 视觉帧时刻/s\\ \hline''']
for r in records:
    sub=','.join(str(x) for x in r['subword_indices'])
    pts=', '.join(f'{x:.1f}' for x in r['video_pts'])
    tex.append(f"        {r['word']} & {sub} & $[{r['start']:.4f},{r['end']:.4f}]$ & {r['audio_first']}--{r['audio_last']} & {pts}\\\\")
tex.append(r'''        \hline
    \end{tabular}
    \par\smallskip
    \parbox{0.95\linewidth}{\footnotesize 注：索引从0开始；声学索引指具有正聚合权重的原生特征记录，不是波形采样点。视觉帧按其时间支持区间与候选词区间的交叠关系参与聚合，帧中心不必位于词区间内部。}
    \endgroup
\end{table}''')
(HERE/'mapping_table.tex').write_text('\n'.join(tex)+'\n',encoding='utf-8')
manifest={'diagnostic_sources':['step1/results/samples/*/features.npz','step1/results/alignment_quality_review.csv'],
    'word_count':len(c),'sample_count':len(wer),'low_confidence_words':int((c<.1).sum()),
    'mean_confidence':float(c.mean()),'median_confidence':float(np.median(c)),
    'mean_diagnostic_wer':float(wer.mean()),'mapping_records':records,
    'manual_review_scope':'本次只核读保存的实际帧图、候选词界和诊断记录；未听辨并标注全部词界。'}
(HERE/'review_materials_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(manifest,ensure_ascii=False,indent=2))
