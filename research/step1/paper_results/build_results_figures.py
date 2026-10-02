"""读取既有实验结果生成论文图；不重跑模型、不修改原始结果。"""
from pathlib import Path
import csv
import json
import numpy as np
import av
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Rectangle

HERE = Path(__file__).resolve().parent
STEP = HERE.parent
DATA = STEP / 'results'
FIG = HERE / 'figures'
PREVIEW = HERE / 'previews'
FIG.mkdir(exist_ok=True)
PREVIEW.mkdir(exist_ok=True)
font = '/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc'
matplotlib.font_manager.fontManager.addfont(font)
plt.rcParams.update({'font.family': FontProperties(fname=font).get_name(),
                     'font.size': 10.5, 'axes.unicode_minus': False,
                     'pdf.fonttype': 3, 'text.parse_math': False,
                     'axes.spines.top': False, 'axes.spines.right': False})
BLUE, GREEN, GOLD, GRAY = '#4778A8', '#52997B', '#C1914B', '#9AA3AD'

def save(fig, name):
    fig.savefig(FIG / (name + '.pdf'), bbox_inches='tight', facecolor='white')
    fig.savefig(PREVIEW / (name + '.png'), dpi=170, bbox_inches='tight', facecolor='white')
    plt.close(fig)

metrics = json.loads((DATA/'metrics.json').read_text())
quality = json.loads((DATA/'alignment_quality_summary.json').read_text())
verification = json.loads((DATA/'verification.json').read_text())
counts = dict(words=0, timing_usable=0, joint_usable=0, timing_only=0,
              timing_restricted=0, risk_sample_words=0, low_outside_risk=0)
for p in sorted((DATA/'samples').glob('*/quality_masks.npz')):
    q = json.loads((p.parent/'quality.json').read_text())
    with np.load(p) as z:
        t = z['timing_usable']
        joint = z['text_aligned_usable'] & z['audio_aligned_usable'] & z['vision_aligned_usable']
        assert np.all(z['text_aligned_usable'] == t)
        assert np.all(z['audio_aligned_usable'] == t)
        counts['words'] += len(t)
        counts['timing_usable'] += int(t.sum())
        counts['joint_usable'] += int(joint.sum())
        counts['timing_only'] += int((t & ~joint).sum())
        counts['timing_restricted'] += int((~t).sum())
    with np.load(p.parent/'features.npz') as z:
        if q['requires_transcript_review']:
            counts['risk_sample_words'] += len(z['word_confidence'])
        else:
            counts['low_outside_risk'] += int((z['word_confidence'] < .1).sum())
assert counts['words'] == metrics['totals']['words'] == 1932
assert counts['timing_usable'] == quality['usable_word_times'] == 1545
assert counts['joint_usable'] == 1387 and counts['timing_only'] == 158
assert counts['timing_restricted'] == counts['risk_sample_words'] + counts['low_outside_risk'] == 387

# 图1：使用保存的音视频特征、词界和实际原始帧，不进行模型推理。
sid = '-egA8-b7-3M$_$1'
src = DATA/'samples'/sid
meta = json.loads((src/'metadata.json').read_text())
with np.load(src/'features.npz') as z:
    d = {k: z[k] for k in ['audio_bounds','audio_frames','audio_frame_valid',
         'vision_pts','vision_frames','vision_frame_valid','video_frame_indices','word_bounds','word_confidence']}
start, end = 1.8, 3.6
target = [2.0, 2.4, 3.0, 3.4]
js = [int(np.argmin(abs(d['vision_pts']-t))) for t in target]
indices = {int(d['video_frame_indices'][j]): j for j in js}
frames = {}
with av.open(str(STEP.parent/meta['source_path'])) as container:
    for i, f in enumerate(container.decode(video=0)):
        if i in indices:
            j = indices[i]
            t = float(f.pts*f.time_base)-meta['video_origin']
            assert abs(t-d['vision_pts'][j]) < 1e-7
            frames[j] = f.to_ndarray(format='rgb24')
        if len(frames) == len(js): break
assert len(frames) == len(js)
fig, axes = plt.subplots(4, 1, figsize=(9.1, 6.6), sharex=True,
                         gridspec_kw={'height_ratios':[1.32,.85,1.05,1.1]})
fig.subplots_adjust(left=.11, right=.98, bottom=.10, top=.94, hspace=.67)
ax=axes[0]
for j in js:
    t=float(d['vision_pts'][j])
    ax.imshow(frames[j], extent=(t-.175,t+.175,0,1), aspect='auto')
    ax.text(t,1.05,f'{t:.2f} s',ha='center',va='bottom',fontsize=9)
ax.set_ylim(0,1.28);ax.axis('off')
ax.set_title('（a）原视频实际抽样帧',loc='left',fontsize=11.5,pad=15)
ax=axes[1]
selected=[]
for wid in [3,4,5,6]:
    w=meta['words'][wid]; a,b=d['word_bounds'][wid]
    ax.add_patch(Rectangle((a,.14),b-a,.60,facecolor='#E4F0EC',edgecolor=GREEN,lw=1.2))
    ax.text((a+b)/2,.44,w['original'],ha='center',va='center',fontsize=10)
    selected.append(dict(word=w['original'],start=float(a),end=float(b),confidence=float(d['word_confidence'][wid])))
ax.set(ylim=(0,1),yticks=[])
ax.spines[['left','bottom']].set_visible(False)
ax.tick_params(bottom=False)
ax.set_title('（b）候选词级时间区间',loc='left',fontsize=11.5,pad=9)
centers=d['audio_bounds'].mean(axis=1)
vals=d['audio_frames'][:,meta['audio_feature_names'].index('Loudness_sma3')].copy()
vals[~d['audio_frame_valid']]=np.nan
axes[2].plot(centers,vals,color=BLUE,lw=1)
axes[2].set_ylabel('响度')
axes[2].set_title('（c）原生声学特征',loc='left',fontsize=11.5,pad=9)
for name, label, color, marker in [('jawOpen','张嘴',GREEN,'o'),('mouthSmileLeft','左嘴角上扬',GOLD,'s')]:
    vals=d['vision_frames'][:,meta['vision_feature_names'].index(name)].copy()
    vals[~d['vision_frame_valid']]=np.nan
    axes[3].plot(d['vision_pts'],vals,label=label,color=color,marker=marker,ms=3,lw=1)
axes[3].set(ylabel='面部系数',xlabel='相对视频起点的时间 / s',ylim=(-.02,1.02))
axes[3].set_title('（d）原生视觉特征',loc='left',fontsize=11.5,pad=9)
axes[3].legend(loc='upper right',ncol=2,fontsize=9,frameon=False)
for ax in axes[1:]:
    ax.set_xlim(start,end)
    for w in selected:
        ax.axvline(w['start'],color='#A8B5BD',ls='--',lw=.65,zorder=0)
        ax.axvline(w['end'],color='#A8B5BD',ls='--',lw=.65,zorder=0)
for ax in axes[2:]: ax.grid(axis='y',alpha=.2)
save(fig,'fig_p1_multimodal_result')

# 图2：三个互斥类别；100%分母为全部1932个词。
fig, ax=plt.subplots(figsize=(9.1,2.65))
fig.subplots_adjust(left=.045,right=.97,top=.83,bottom=.37)
categories=[('时间规则通过且三模态可观测',counts['joint_usable'],GREEN,''),
            ('时间规则通过但缺少视觉观测',counts['timing_only'],GOLD,'//'),
            ('未通过时间使用规则',counts['timing_restricted'],GRAY,'..')]
left=0
for label,n,color,hatch in categories:
    width=100*n/counts['words']
    ax.barh(0,width,left=left,height=.56,color=color,edgecolor='white',hatch=hatch,label=label)
    ax.text(left+width/2,0,f'{n}\n{width:.2f}%',ha='center',va='center',fontsize=10,
            color='white' if color==GREEN else '#172D46',
            bbox={'facecolor':color,'edgecolor':'none','pad':2})
    left+=width
ax.set(xlim=(0,100),ylim=(-.45,.45),yticks=[],xlabel='占全部词位置的比例 / %')
ax.spines[['left','top','right']].set_visible(False)
ax.set_title('全部1932个词位置的质量构成',loc='left',fontsize=12,pad=13)
ax.legend(loc='upper center',bbox_to_anchor=(.5,-.67),ncol=1,frameon=False,fontsize=10)
save(fig,'fig_p1_quality_composition')

# 图3：只读取既有200行窗口对照结果。箱线表示分布，菱形表示样本均值。
with (DATA/'window_comparison.csv').open(encoding='utf-8-sig') as f:
    records=list(csv.DictReader(f))
assert len(records)==200
fields=[('audio_normalized_reconstruction_mse','（a）声学标准化重建误差','标准化重建MSE',100),
        ('vision_normalized_reconstruction_mse','（b）视觉标准化重建误差','标准化重建MSE',87),
        ('word_boundary_crossing_ratio','（c）词跨窗口边界比例','样本内跨界比例',79)]
fig,axes=plt.subplots(1,3,figsize=(10,3.65))
fig.subplots_adjust(left=.07,right=.985,bottom=.17,top=.75,wspace=.40)
window_stats={}
for ax,(key,title,ylabel,n) in zip(axes,fields):
    values=[np.array([float(r[key]) for r in records if float(r['window_seconds'])==width and r[key]!='']) for width in (.2,.5)]
    assert all(len(v)==n for v in values)
    bp=ax.boxplot(values,positions=[1,2],widths=.48,patch_artist=True,showfliers=True,
                  medianprops={'color':'#20344C','linewidth':1.3},
                  flierprops={'marker':'.','markersize':3,'markeredgecolor':'#84929E'})
    for patch,c in zip(bp['boxes'],[BLUE,GOLD]): patch.set(facecolor=c,alpha=.35)
    means=[float(x.mean()) for x in values]
    ax.scatter([1,2],means,color='#263D50',marker='D',s=26,zorder=4)
    ax.set_xticks([1,2],['0.2 s','0.5 s']);ax.set_ylabel(ylabel)
    ax.set_title(title,fontsize=11,loc='left',pad=40)
    ax.text(.5,1.06,f'n = {n}；均值 {means[0]:.4f} / {means[1]:.4f}',ha='center',transform=ax.transAxes,fontsize=9)
    # 声学有两个恒定序列，零误差离群点会重叠；留出下边距，避免被坐标轴遮住。
    ax.set_ylim(bottom=-.04 if key=='audio_normalized_reconstruction_mse' else 0)
    ax.grid(axis='y',alpha=.2)
    if key=='word_boundary_crossing_ratio': ax.set_ylim(0,1.05)
    window_stats[key]={'samples_per_width':n,'mean_0.2s':means[0],'mean_0.5s':means[1]}
    for width,mean in zip((.2,.5),means):
        old=next(r for r in metrics['window_comparison'] if r['window_seconds']==width)[key]
        assert np.isclose(old,mean,rtol=1e-10,atol=1e-10)
save(fig,'fig_p1_window_comparison')

manifest={'source':'step1/results 中的已保存结果；本次仅读取、汇总与绘图',
          'counts':counts,'quality_rules':quality,'verification':{k:v for k,v in verification.items() if k!='samples'},
          'example':{'sample_id':sid,'display_seconds':[start,end],'words':selected,
                     'frame_pts':[float(d['vision_pts'][j]) for j in js]},
          'window_statistics':window_stats,
          'figures':[p.name for p in sorted(FIG.glob('*.pdf'))]}
(HERE/'results_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'counts':counts,'window_statistics':window_stats,'pdfs':manifest['figures']},ensure_ascii=False,indent=2))
