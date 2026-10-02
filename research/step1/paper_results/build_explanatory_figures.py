"""用真实记录解释观测、低能量、窗口平滑；不推理模型、不改变结果。"""
from pathlib import Path
import sys,json
import numpy as np
import av
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Rectangle,Patch
HERE=Path(__file__).resolve().parent;STEP=HERE.parent;DATA=STEP/'results';PROJECT=STEP.parent
sys.path.insert(0,str(STEP))
from alignment import overlap_pool,window_bounds
font='/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc'
matplotlib.font_manager.fontManager.addfont(font)
plt.rcParams.update({'font.family':FontProperties(fname=font).get_name(),'font.size':10.5,
 'axes.unicode_minus':False,'pdf.fonttype':3,'text.parse_math':False,
 'axes.spines.top':False,'axes.spines.right':False})
GREEN='#40876C';BLUE='#356B9A';GOLD='#CB903B';GRAY='#AAB2BA';LIGHT='#EFF3F5'
manifest={}
def load(sid):
 p=DATA/'samples'/sid;m=json.loads((p/'metadata.json').read_text())
 with np.load(p/'features.npz') as z:d={k:z[k] for k in z.files}
 return m,d

def frame(m,d,t):
 j=int(np.argmin(abs(d['vision_pts']-t)));idx=int(d['video_frame_indices'][j])
 with av.open(str(PROJECT/m['source_path'])) as c:
  for i,f in enumerate(c.decode(video=0)):
   if i==idx:
    assert abs(float(f.pts*f.time_base)-m['video_origin']-d['vision_pts'][j])<1e-7
    return f.to_ndarray(format='rgb24'),float(d['vision_pts'][j])
 raise AssertionError('missing frame')

def save(fig,name):
 fig.savefig(HERE/'figures'/f'{name}.pdf',bbox_inches='tight',facecolor='white')
 fig.savefig(HERE/'previews'/f'{name}.png',bbox_inches='tight',facecolor='white',dpi=170)
 plt.close(fig)

# 帧层与词层分别计数，避免把两个分母混在一起。
counts={'words':0,'visual_words':0,'frames':0,'valid_frames':0,'audio':0,'audio_low':0,'audio_valid':0}
for p in (DATA/'samples').glob('*/features.npz'):
 with np.load(p) as z:
  counts['words']+=len(z['text']);counts['visual_words']+=int(z['vision_valid'].sum())
  counts['frames']+=len(z['vision_pts']);counts['valid_frames']+=int(z['vision_frame_valid'].sum())
  counts['audio']+=len(z['audio_frames']);counts['audio_low']+=int(z['audio_silence'].sum())
  counts['audio_valid']+=int(z['audio_frame_valid'].sum())
assert counts=={'words':1932,'visual_words':1595,'frames':3973,'valid_frames':3253,'audio':77261,'audio_low':9918,'audio_valid':77261}
fig=plt.figure(figsize=(10,7.1));gs=fig.add_gridspec(3,3,height_ratios=[1.35,1.45,1.05],hspace=.78,wspace=.3)
fig.subplots_adjust(left=.055,right=.98,top=.97,bottom=.07)
ax=fig.add_subplot(gs[0,:]);ax.set_title('（a）同一批结果的两种统计单位：帧数与词数',loc='left',pad=13)
for y,label,n,total in [(1,'抽样视频帧',3253,3973),(0,'原文词位置',1595,1932)]:
 pct=100*n/total;ax.barh(y,pct,color=GREEN,height=.52);ax.barh(y,100-pct,left=pct,color=GRAY,height=.52)
 ax.text(pct/2,y,f'有面部特征  {n}（{pct:.2f}%）',ha='center',va='center',color='white',fontsize=10)
 ax.text(pct+(100-pct)/2,y,f'无特征 {total-n}\n{100-pct:.2f}%',ha='center',va='center',fontsize=9)
 ax.text(-2,y,label,ha='right',va='center',fontsize=10)
ax.set(yticks=[],xlim=(-16,100),ylim=(-.6,1.6),xlabel='占各自统计总数的比例 / %');ax.set_xticks([0,20,40,60,80,100]);ax.spines['left'].set_visible(False)
examples=[('-s9qJ7ATP7w$_$7','work',.8,'（b）有画面，未提取出面部特征'),
          ('-s9qJ7ATP7w$_$7','day',4.,'（c）有画面，也有面部特征'),
          ('-iRBcNs9oI8$_$9','similar',1.6,'（d）人物可见，仍未获得面部特征')]
recs=[]
for k,(sid,word,t,title) in enumerate(examples):
 m,d=load(sid);wid=next(i for i,w in enumerate(m['words']) if w['original']==word)
 a,b=d['word_bounds'][wid];js=np.flatnonzero(np.maximum(0,np.minimum(b,d['vision_bounds'][:,1])-np.maximum(a,d['vision_bounds'][:,0]))>0)
 valid=d['vision_frame_valid'][js];im,pts=frame(m,d,t)
 ax=fig.add_subplot(gs[1,k]);ax.imshow(im);ax.axis('off');ax.set_title(title,loc='left',fontsize=9.3,pad=9)
 ax.text(.5,-.10,f'实际抽样帧：{pts:.4f} s',ha='center',transform=ax.transAxes,fontsize=9)
 ax=fig.add_subplot(gs[2,k]);left=d['vision_bounds'][js[0],0];right=d['vision_bounds'][js[-1],1]
 for j in js:
  u,v=d['vision_bounds'][j];c=GREEN if d['vision_frame_valid'][j] else GRAY
  ax.add_patch(Rectangle((u,.65),v-u,.28,facecolor=c,edgecolor='white'))
  ax.text((u+v)/2,.79,f'{d["vision_pts"][j]:.1f} s',ha='center',va='center',fontsize=8,color='white' if c==GREEN else 'black')
 ax.add_patch(Rectangle((a,.14),b-a,.28,facecolor='#DCE7EF',edgecolor=BLUE))
 ax.text((a+b)/2,.28,word,ha='center',va='center',fontsize=10)
 ax.set(xlim=(left-.02,right+.02),ylim=(-.15,1.10),yticks=[],xlabel='时间 / s')
 ax.set_xticks([left,right]);ax.set_xticklabels([f'{left:.2f}',f'{right:.2f}']);ax.spines['left'].set_visible(False)
 ax.set_title(f'交叠的{len(js)}帧中有效{int(valid.sum())}帧 → 词观测标记{int(valid.any())}',fontsize=9,pad=8)
 assert bool(valid.any())==bool(d['vision_valid'][wid])
 recs.append(dict(sample_id=sid,word=word,word_interval=[float(a),float(b)],frame_pts=[float(d['vision_pts'][j]) for j in js],valid_frames=int(valid.sum()),word_observed=bool(valid.any()),displayed_frame_pts=pts))
save(fig,'fig_p1_observation_explained');manifest['observation_examples']=recs

# 低能量仍是实际记录。全局计数、实际曲线与二值标记分别说明。
sid='-iRBcNs9oI8$_$9';m,d=load(sid);t=d['audio_bounds'].mean(axis=1);r=d['audio_rms'];threshold=.0031622776601683794
fig=plt.figure(figsize=(9.8,6.4));gs=fig.add_gridspec(3,2,height_ratios=[.62,1.5,.65],hspace=.9,wspace=.30)
fig.subplots_adjust(left=.085,right=.98,top=.93,bottom=.10)
ax=fig.add_subplot(gs[0,:]);n=counts['audio_low'];total=counts['audio'];pct=100*n/total
ax.barh(0,100-pct,color=BLUE,height=.6);ax.barh(0,pct,left=100-pct,color=GOLD,height=.6)
ax.text((100-pct)/2,0,f'未标记低能量：{total-n}（{100-pct:.2f}%）',ha='center',va='center',color='white',fontsize=10)
ax.text(100-pct/2,0,f'低能量 {n}\n{pct:.2f}%',ha='center',va='center',fontsize=9)
ax.set(xlim=(0,100),ylim=(-.5,.5),yticks=[],xticks=[]);ax.spines[['left','bottom']].set_visible(False)
ax.set_title('（a）77261条声学记录均有观测，其中9918条幅度较低',loc='left',fontsize=11,pad=10)
for col,zoom in [(0,False),(1,True)]:
 ax=fig.add_subplot(gs[1,col]);ax.plot(t,r,color=BLUE,lw=1.1)
 ax.axhline(threshold,color=GOLD,ls='--',lw=1.2,label='RMS阈值0.0031623')
 # each center is a record; dots match recorded classification, not inferred speech activity
 low=d['audio_silence'];ax.scatter(t[low],r[low],color=GOLD,s=6,zorder=4)
 ax.set(xlim=(0,m['duration']),ylabel='RMS（幅度的均方根）',xlabel='时间 / s')
 ax.set_title('（c）同一曲线：放大低幅值范围' if zoom else '（b）一个样本的实际RMS曲线',loc='left',fontsize=10.5,pad=12)
 ax.set_ylim((0,.010) if zoom else (0,.55));ax.grid(axis='y',alpha=.2)
 if zoom:ax.legend(loc='upper right',fontsize=8,frameon=False)
ax=fig.add_subplot(gs[2,:]);ax.set_title('（d）同一样本的两种标记：低能量不等于没有声学记录',loc='left',fontsize=11,pad=10)
# 连续条仅编码逐条记录状态；每个记录以相邻中心的中点划分显示格。
edges=np.r_[d['audio_bounds'][0,0],(t[:-1]+t[1:])/2,d['audio_bounds'][-1,1]]
for j in range(len(t)):
 ax.add_patch(Rectangle((edges[j],.7),edges[j+1]-edges[j],.22,facecolor=GREEN,edgecolor='none'))
 ax.add_patch(Rectangle((edges[j],.15),edges[j+1]-edges[j],.22,facecolor=GOLD if d['audio_silence'][j] else LIGHT,edgecolor='none'))
ax.text(m['duration']+.035,.81,'333/333条有效',va='center',fontsize=9)
ax.text(m['duration']+.035,.26,'147/333条低能量',va='center',fontsize=9)
ax.set(xlim=(0,m['duration']+.65),ylim=(0,1.05),yticks=[.26,.81],yticklabels=['低能量标记','有观测标记'],xlabel='时间 / s')
ax.set_xticks([0,.5,1,1.5,2,2.5,3]);ax.spines['left'].set_visible(False)
assert int(d['audio_silence'].sum())==147 and d['audio_frame_valid'].all()
save(fig,'fig_p1_low_energy_explained');manifest['low_energy_example']={'sample_id':sid,'records':len(t),'low_energy_records':int(d['audio_silence'].sum()),'valid_records':int(d['audio_frame_valid'].sum()),'threshold':threshold}

# 用同一条真实响度曲线展示平滑，用同一个词展示跨窗。
sid='-egA8-b7-3M$_$1';m,d=load(sid);dim=m['audio_feature_names'].index('Loudness_sma3');t=d['audio_bounds'].mean(axis=1)
fig,axs=plt.subplots(1,2,figsize=(10,3.7),gridspec_kw={'width_ratios':[1.25,1.]})
fig.subplots_adjust(left=.075,right=.98,bottom=.20,top=.79,wspace=.28)
ax=axs[0];ax.plot(t,d['audio_frames'][:,dim],color='#7A8288',lw=1,label='原生响度')
for width,color in [(.2,BLUE),(.5,GOLD)]:
 bounds=window_bounds(m['duration'],width)
 pooled,mask,_=overlap_pool(bounds,d['audio_bounds'],d['audio_frames'],d['audio_frame_valid'])
 recon,support,_=overlap_pool(d['audio_bounds'],bounds,pooled,mask)
 ax.plot(t,recon[:,dim],color=color,lw=1.4,label=f'{width:.1f} s窗口重建')
ax.set(xlim=(1.8,3.6),xlabel='时间 / s',ylabel='响度',ylim=(0,1.1));ax.grid(axis='y',alpha=.2)
ax.set_title('（a）先做窗口平均，再还原到原记录位置',fontsize=11,loc='left',pad=17)
ax.legend(loc='upper right',fontsize=8,frameon=False)
wid=3;a,b=d['word_bounds'][wid];ax=axs[1];crossings={}
for y,width,color in [(1.0,.2,BLUE),(0.,.5,GOLD)]:
 bounds=window_bounds(m['duration'],width);hit=np.maximum(0,np.minimum(b,bounds[:,1])-np.maximum(a,bounds[:,0]))>1e-8
 crossings[str(width)]=int(hit.sum())
 for j,(u,v) in enumerate(bounds):
  if v<1.7 or u>2.65:continue
  ax.add_patch(Rectangle((u,y+.01),v-u,.44,facecolor=color if hit[j] else LIGHT,alpha=.20,edgecolor=color,lw=.8))
  if 1.7<=u<=2.65:ax.plot([u,u],[y+.01,y+.45],color=color,ls=':',lw=.8)
 ax.plot([a,b],[y+.24,y+.24],color=GREEN,lw=5,solid_capstyle='butt')
 ax.text((a+b)/2,y+.57,f'{width:.1f} s窗：successful跨{hit.sum()}个窗口',ha='center',fontsize=9.5)
ax.set(xlim=(1.7,2.65),ylim=(-.1,1.9),yticks=[],xlabel='时间 / s');ax.spines['left'].set_visible(False)
for x in [a,b]:ax.axvline(x,color=GREEN,ls='--',lw=.8)
ax.set_xticks([1.8,2.,2.2,2.4,2.6]);ax.set_title('（b）同一词区间遇到不同密度的窗口边界',loc='left',fontsize=11,pad=17)
assert crossings=={'0.2':4,'0.5':2}
save(fig,'fig_p1_window_explained');manifest['window_example']={'sample_id':sid,'word':'successful','interval':[float(a),float(b)],'overlapping_windows':crossings,'acoustic_feature':'Loudness_sma3'}
manifest['counts']=counts
(HERE/'explanatory_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
print(json.dumps(manifest,ensure_ascii=False,indent=2))
