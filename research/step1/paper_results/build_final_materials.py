"""从附件一和保存结果生成全量正文表、困难实例图及核对记录。"""
from pathlib import Path
import json
import csv
import numpy as np
import openpyxl
import av
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Rectangle

HERE = Path(__file__).resolve().parent
STEP = HERE.parent
DATA = STEP/'results'
PROJECT = STEP.parent
font = '/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc'
matplotlib.font_manager.fontManager.addfont(font)
plt.rcParams.update({'font.family':FontProperties(fname=font).get_name(),
    'font.size':10.5,'axes.unicode_minus':False,'pdf.fonttype':3,
    'text.parse_math':False,'axes.spines.top':False,'axes.spines.right':False})
metas = {m['sample_id']:m for m in (json.loads(p.read_text()) for p in (DATA/'samples').glob('*/metadata.json'))}
xlsx = next((PROJECT/'E题_材料阅读/E题数据/附件1-数据集原始多模态样本').rglob('label-100.xlsx'))
wb = openpyxl.load_workbook(xlsx,read_only=True,data_only=True)
rows = list(wb['label'].values)
assert len(rows)==101 and len(metas)==100
for values in rows[1:]:
    r=dict(zip(rows[0],values));sid=str(r['video_id'])+'$_$'+str(r['clip_id'])
    assert metas[sid]['raw_text']==r['text']
    assert metas[sid]['label']==float(r['label']) and metas[sid]['annotation']==r['annotation']
wb.close()
with (DATA/'sample_summary.csv').open(encoding='utf-8-sig') as f:
    summaries={r['sample_id']:r for r in csv.DictReader(f)}
counts=dict(words=0,time_usable=0,joint_usable=0,no_vision_time_usable=0,risk_samples=0,risk_words=0,low_words=0)
lines=[r'''% 根据逐样本metadata.json、features.npz和quality_masks.npz生成，不是论文附录。
\begingroup
\footnotesize
\setlength{\tabcolsep}{5pt}
\renewcommand{\arraystretch}{1.10}
\begin{longtable}{@{}lrrrrrr@{}}
\caption{附件1全部100条样本的三模态特征结果}\label{tab:p1_all_samples}\\
\toprule
样本编号 & $D$/s & $D_a$/s & $L$ & $T/A/V$ & $U$ & $R$\\
\midrule
\endfirsthead
\multicolumn{7}{c}{表~\thetable\quad 续表}\\
\toprule
样本编号 & $D$/s & $D_a$/s & $L$ & $T/A/V$ & $U$ & $R$\\
\midrule
\endhead
\midrule
\multicolumn{7}{r}{续下页}\\
\endfoot
\bottomrule
\endlastfoot''']
for sid,m in sorted(metas.items()):
    path=DATA/'samples'/sid
    quality=json.loads((path/'quality.json').read_text())
    with np.load(path/'features.npz') as z, np.load(path/'quality_masks.npz') as q:
        n=len(m['words']);observed=[int(z[f'{mod}_valid'].sum()) for mod in ['text','audio','vision']]
        for mod,dim in [('text',768),('audio',25),('vision',52)]: assert z[mod].shape==(n,dim)
        t=q['timing_usable'];joint=q['text_aligned_usable'] & q['audio_aligned_usable'] & q['vision_aligned_usable']
        counts['words']+=n;counts['time_usable']+=int(t.sum());counts['joint_usable']+=int(joint.sum())
        counts['no_vision_time_usable']+=int((t & ~joint).sum())
        counts['risk_samples']+=int(quality['requires_transcript_review'])
        counts['risk_words']+=n*int(quality['requires_transcript_review'])
        counts['low_words']+=int((z['word_confidence']<.1).sum())
        assert observed==[int(summaries[sid][f'{mod}_observed_words']) for mod in ['text','audio','vision']]
        lines.append(r'\nolinkurl{'+sid+'} & '+f"{m['duration']:.3f} & {m['audio_duration']:.3f} & {n} & {'/'.join(map(str,observed))} & {int(t.sum())} & {int(quality['requires_transcript_review'])}"+r'\\')
lines += [r'\end{longtable}',r'\endgroup']
# 当前清单由统一生成器输出，避免历史脚本恢复旧缩写表头。
from build_sample_catalog import main as build_current_catalog
build_current_catalog()
assert counts==dict(words=1932,time_usable=1545,joint_usable=1387,no_vision_time_usable=158,risk_samples=21,risk_words=317,low_words=350)

sid='-iRBcNs9oI8$_$9';m=metas[sid];p=DATA/'samples'/sid
with np.load(p/'features.npz') as z: d={k:z[k] for k in z.files}
end=m['duration'];wanted=[0.6,1.6,2.6];js=[int(np.argmin(abs(d['vision_pts']-v))) for v in wanted]
indices={int(d['video_frame_indices'][j]):j for j in js};frames={}
with av.open(str(PROJECT/m['source_path'])) as c:
    for i,f in enumerate(c.decode(video=0)):
        if i in indices:
            j=indices[i];actual=float(f.pts*f.time_base)-m['video_origin']
            assert abs(actual-d['vision_pts'][j])<1e-7
            frames[j]=f.to_ndarray(format='rgb24')
        if len(frames)==len(js): break
assert len(frames)==3 and not d['vision_frame_valid'].any()
fig,axes=plt.subplots(3,1,figsize=(9.1,5.4),sharex=True,
    gridspec_kw={'height_ratios':[1.30,1.,1.30]})
fig.subplots_adjust(left=.10,right=.98,top=.91,bottom=.11,hspace=.77)
ax=axes[0]
for j in js:
    t=float(d['vision_pts'][j]);ax.imshow(frames[j],extent=(t-.40,t+.40,0,1),aspect='auto')
    ax.text(t,1.04,f'{t:.2f} s',ha='center',va='bottom',fontsize=9)
ax.set(ylim=(0,1.28));ax.axis('off')
ax.set_title('（a）实际抽样帧：画面有人，面部特征无有效输出',loc='left',fontsize=11.5,pad=14)
ax=axes[1];times=d['audio_bounds'].mean(axis=1)
ax.plot(times,d['audio_rms'],color='#4778A8',lw=1.)
ax.axhline(.0031622776601683794,color='#9AA3AD',ls=':',lw=.8,label='低能量阈值')
ax.set_ylabel('音频RMS');ax.grid(axis='y',alpha=.2)
ax.set_title('（b）原生声学记录：保留有效音频观测',loc='left',fontsize=11.5,pad=10)
ax.legend(loc='upper right',fontsize=8,frameon=False)
ax=axes[2]
for i,w in enumerate(m['words']):
    a,b=d['word_bounds'][i];y=1.15 if i%2==0 else .12
    ax.add_patch(Rectangle((a,y),b-a,.48,facecolor='#F4E5D1',edgecolor='#C1914B',lw=1.))
    ax.text((a+b)/2,y+.24,w['original'],ha='center',va='center',fontsize=10)
    ax.text((a+b)/2,y-.08,f"c={d['word_confidence'][i]:.1e}",ha='center',va='top',fontsize=8)
ax.set(xlim=(0,end),ylim=(-.35,1.85),yticks=[],xlabel='相对视频起点的时间 / s')
ax.spines['left'].set_visible(False)
ax.set_title('（c）候选词区间：5个词全部时间受限',loc='left',fontsize=11.5,pad=11)
for ax in axes[1:]:
    for a,b in d['word_bounds']:
        ax.axvline(a,color='#ADB6BE',ls='--',lw=.55,zorder=0)
        ax.axvline(b,color='#ADB6BE',ls='--',lw=.55,zorder=0)
for ext in ['pdf','png']:
    out=HERE/('figures' if ext=='pdf' else 'previews')/f'fig_p1_difficult_result.{ext}'
    fig.savefig(out,bbox_inches='tight',facecolor='white',dpi=170)
plt.close(fig)
verification=json.loads((DATA/'verification.json').read_text())
assert verification['all_passed'] and verification['checked_samples']==100
manifest={'sources':['附件1/label-100.xlsx','step1/results/samples/*/{metadata.json,features.npz,quality_masks.npz}','step1/results/verification.json'],
    'xlsx_rows_matched':100,'video_sources':len({m['video_id'] for m in metas.values()}),'counts':counts,
    'independent_verification':{k:v for k,v in verification.items() if k!='samples'},
    'difficult_example':{'sample_id':sid,'frame_pts':[float(d['vision_pts'][j]) for j in js],
        'mean_confidence':float(d['word_confidence'].mean()),'diagnostic_wer':m['diagnostic_wer'],
        'sampled_frames':len(d['vision_pts']),'valid_face_frames':int(d['vision_frame_valid'].sum())},
    'full_table_rows':len(metas),'manual_review_scope':'原文、保存诊断与实际画面核读；没有人工听辨或词界真值标注'}
(HERE/'final_materials_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
print(json.dumps(manifest,ensure_ascii=False,indent=2))
