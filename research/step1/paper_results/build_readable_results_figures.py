"""读取既有结果，绘制带阅读提示的结果图；不重跑模型，不改动实验数据。"""
from pathlib import Path
import csv
import json
import hashlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch, Patch
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator, MaxNLocator
from build_publication_figures import load, frame, overlap_pool, window_bounds, DATA, HERE

OUT = HERE / 'readable_figures'
OUT.mkdir(exist_ok=True)
plt.rcParams.update({'font.size': 10.5, 'axes.labelsize': 11,
                     'axes.titlesize': 11.5, 'xtick.labelsize': 10,
                     'ytick.labelsize': 10, 'legend.fontsize': 10})
BLUE, GREEN, ORANGE, GRAY = '#79A9CE', '#78A98E', '#EDA45B', '#CBD2D8'
BE, GE, OE, DARK = '#416E91', '#497963', '#B57C42', '#263D50'
audit = json.loads((HERE / 'data_audit.json').read_text())
assert audit['status'] == 'passed'
C = audit['counts']
manifest = {'source': '已有特征、质量掩码、data_audit.json和audited_window_comparison.csv',
            'figures': {}, 'checks': {}, 'source_sha256': {}}
for path in [HERE/'data_audit.json', HERE/'audited_window_comparison.csv']:
    manifest['source_sha256'][path.name] = hashlib.sha256(path.read_bytes()).hexdigest()


def save(fig, name):
    fig.canvas.draw()
    if name == 'fig_p1_quality_flow':
        for patch in fig.findobj(FancyBboxPatch):
            patch.set_boxstyle('square', pad=patch.get_boxstyle().pad)
            patch.set_joinstyle('miter')
        for text in fig.findobj(plt.Text):
            text.set_color('#000000')
            text.set_alpha(1)
    manifest['figures'][name] = sorted({t.get_text() for t in fig.findobj(plt.Text)
                                      if t.get_visible() and t.get_text()})
    fig.savefig(OUT/f'{name}.pdf', bbox_inches='tight', pad_inches=.045)
    fig.savefig(OUT/f'{name}.png', dpi=160, bbox_inches='tight', pad_inches=.045)
    plt.close(fig)


def clean(ax, axis='y'):
    ax.set_axisbelow(True)
    ax.grid(axis=axis, color='#E8EDF1', lw=.6)
    ax.tick_params(length=3)


def box(ax, x, y, w, h, text, color='#EDF3F7', edge=BE, size=11, linespacing=1.6):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.008,rounding_size=0.015',
                               facecolor=color, edgecolor=edge, linewidth=1))
    ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=size,color=DARK,linespacing=linespacing)


def arrow(ax, a, b, color='#718392'):
    ax.annotate('',xy=b,xytext=a,arrowprops={'arrowstyle':'-|>','lw':1,'color':color})


def observation():
    fig, axes = plt.subplots(1,2,figsize=(8.5,2.65),gridspec_kw={'width_ratios':[1.35,1]})
    fig.subplots_adjust(left=.12,right=.98,bottom=.24,top=.83,wspace=.47)
    ax = axes[0]
    good = [C['valid_face_frames'],C['vision_observed_words']]
    totals = [C['sampled_frames'],C['words']]
    pct = np.array(good)/totals*100
    ax.barh([1,0],pct,color=GREEN,height=.54)
    ax.barh([1,0],100-pct,left=pct,color=GRAY,height=.54)
    for y,n,total,p,unit in zip([1,0],good,totals,pct,['帧','词']):
        ax.text(p/2,y,f'{n} {unit}',ha='center',va='center',fontsize=10)
        ax.text(p+(100-p)/2,y,f'{total-n} {unit}',ha='center',va='center',fontsize=9)
    ax.set(yticks=[1,0],yticklabels=['抽样帧','候选词区间'],xlim=(0,100),ylim=(-.55,1.55),xlabel='占各自总数的比例 / %')
    ax.set_title('（a）哪些位置有可用面部记录？',loc='left',pad=14)
    ax.set_xticks([0,25,50,75,100]); clean(ax,'x')
    ax.legend(handles=[Patch(color=GREEN,label='有面部记录'),Patch(color=GRAY,label='无面部记录')],
              loc='upper center',bbox_to_anchor=(.5,1.10),ncol=2,frameon=False,fontsize=9)
    ax = axes[1]
    ns = [C['audio_records']-C['low_energy'],C['low_energy']]
    ps = np.array(ns)/C['audio_records']*100
    ax.bar([0,1],ps,color=[BLUE,ORANGE],width=.55)
    for x,n,p in zip([0,1],ns,ps):
        ax.text(x,p+2,f'{n} 条',ha='center',va='bottom',fontsize=10)
    ax.set(xticks=[0,1],xticklabels=['非低能量','低能量'],ylim=(0,109),ylabel='声学记录占比 / %')
    ax.set_title('（b）多少声学记录幅度较低？',loc='left',pad=14)
    ax.set_yticks([0,25,50,75,100]);clean(ax)
    fig.text(.12,.025,'观测覆盖不等于对齐准确；低能量表示 RMS 低于 0.0031623，不等于没有语音。',fontsize=10,color=DARK)
    save(fig,'fig_p1_observation_summary')


def rms():
    m,d = load('-iRBcNs9oI8$_$9')
    t=d['audio_bounds'].mean(axis=1);r=d['audio_rms'];low=d['audio_silence']
    threshold=json.loads((HERE.parent/'config.json').read_text())['silence_rms_threshold']
    assert len(t)==333 and low.sum()==147
    fig,axes=plt.subplots(1,2,figsize=(8.5,3.1),sharex=True)
    fig.subplots_adjust(left=.09,right=.98,bottom=.37,top=.85,wspace=.29)
    for ax,upper,title in zip(axes,[.55,.01],['（a）整段幅度变化','（b）同一曲线：仅放大低幅值范围']):
        ax.plot(t,r,color=BE,lw=1)
        ax.axhline(threshold,color=OE,ls='--',lw=1.2)
        ax.scatter(t[low],r[low],color=ORANGE,s=9,zorder=3)
        ax.set(xlim=(0,m['duration']),ylim=(0,upper),xlabel='相对视频起点的时间 / s',ylabel='短时 RMS')
        ax.set_title(title,loc='left',pad=7);clean(ax)
        ax.xaxis.set_major_locator(MultipleLocator(1))
    axes[0].axhspan(0,.01,color=ORANGE,alpha=.2)
    axes[0].text(.03,.90,'低能量记录：147 / 333 条',transform=axes[0].transAxes,fontsize=10)
    axes[1].text(.98,threshold+.00035,'阈值 0.0031623',ha='right',fontsize=9,
                 transform=axes[1].get_yaxis_transform(),color=OE)
    fig.legend(handles=[Line2D([],[],color=BE,label='原生 RMS'),
                        Line2D([],[],color=OE,ls='--',label='低能量阈值'),
                        Line2D([],[],color=ORANGE,marker='o',ls='',label='低于阈值的记录')],
               loc='lower center',bbox_to_anchor=(.54,.10),ncol=3,frameon=False)
    fig.text(.09,.015,'两图时间范围相同；右图大于 0.01 的部分超出显示范围，并未删除。',fontsize=10,color=DARK)
    save(fig,'fig_p1_rms_explained')


def diagnostics_and_flow():
    conf=[];wer=[];counts=dict(words=0,risk_words=0,low_outside=0,passed=0,joint=0,risk_samples=0)
    for p in sorted((DATA/'samples').glob('*/metadata.json')):
        m=json.loads(p.read_text());wer.append(m['diagnostic_wer'])
        q=json.loads((p.parent/'quality.json').read_text())
        with np.load(p.parent/'features.npz') as z:
            c=z['word_confidence'];conf.extend(c.tolist());counts['words']+=len(c)
            if q['requires_transcript_review']:
                counts['risk_words']+=len(c);counts['risk_samples']+=1
            else: counts['low_outside']+=int((c<.1).sum())
        with np.load(p.parent/'quality_masks.npz') as z:
            counts['passed']+=int(z['timing_usable'].sum())
            counts['joint']+=int((z['text_aligned_usable'] & z['audio_aligned_usable'] & z['vision_aligned_usable']).sum())
    assert counts==dict(words=1932,risk_words=317,low_outside=70,passed=1545,joint=1387,risk_samples=21)
    manifest['checks']['quality_counts']=counts
    conf=np.array(conf);wer=np.array(wer)
    assert (conf<.1).sum()==350 and (wer>.8).sum()==21
    fig,axes=plt.subplots(1,2,figsize=(8.5,2.85))
    fig.subplots_adjust(left=.09,right=.98,bottom=.25,top=.78,wspace=.28)
    for ax,values,bins,threshold,title,xlabel,ylabel in [
        (axes[0],conf,np.linspace(0,1,21),.1,'（a）逐词检查：1932 个词','CTC 置信度（模型分数）','词数 / 个'),
        (axes[1],wer,np.linspace(0,2,21),.8,'（b）整段检查：100 条片段','诊断 WER（识别文本与给定转写的差异）','片段数 / 条')]:
        ax.hist(values,bins=bins,color=GREEN if ax is axes[0] else BLUE,edgecolor='white',lw=.6)
        ax.axvline(threshold,color=OE,ls='--',lw=1.2)
        ax.axvspan(0 if ax is axes[0] else threshold,threshold if ax is axes[0] else 2,color=ORANGE,alpha=.16)
        ax.set(xlim=(bins[0],bins[-1]),xlabel=xlabel,ylabel=ylabel)
        ax.set_title(title,loc='left',pad=10);clean(ax)
        ax.yaxis.set_major_locator(MaxNLocator(integer=True,nbins=5))
    axes[0].text(.18,.93,'低于 0.1：350 个词',transform=axes[0].transAxes,fontsize=10,va='top')
    axes[1].text(.43,.92,'高于 0.8：21 条片段',transform=axes[1].transAxes,fontsize=10,va='top')
    fig.text(.09,.025,'置信度不是词界准确率；高 WER 提示文本差异，具体原因由人工审核确认。',fontsize=10,color=DARK)
    save(fig,'fig_p1_alignment_screening')
    quality_flow()


def concise_individual_figures():
    """导出四张独立图：覆盖图只保留数量标注，诊断图仅一行阈值说明。"""
    def canvas():
        fig,ax=plt.subplots(figsize=(4.2,2.65))
        fig.subplots_adjust(left=.21,right=.97,bottom=.23,top=.92)
        return fig,ax

    fig,ax=canvas()
    percentages=np.array([C['valid_face_frames']/C['sampled_frames'],
                          C['vision_observed_words']/C['words']])*100
    ax.barh([1,0],percentages,color=GREEN,height=.47,label='有面部记录')
    ax.barh([1,0],100-percentages,left=percentages,color=GRAY,height=.47,label='无面部记录')
    for y,p,n,total,unit in zip([1,0],percentages,
                              [C['valid_face_frames'],C['vision_observed_words']],
                              [C['sampled_frames'],C['words']],['帧','词']):
        ax.text(p/2,y,f'{n} {unit}',ha='center',va='center',fontsize=10)
        ax.text(p+(100-p)/2,y,f'{total-n} {unit}',ha='center',va='center',fontsize=9)
    ax.set(xlim=(0,100),ylim=(-.55,1.85),yticks=[1,0],yticklabels=['抽样帧','词位置'],xlabel='占比 / %')
    ax.set_xticks([0,25,50,75,100]);clean(ax,'x')
    ax.legend(loc='upper center',ncol=2,frameon=False,fontsize=9,handlelength=1.2,columnspacing=1)
    save(fig,'fig_p1_visual_coverage')

    fig,ax=canvas()
    percentages=np.array([C['audio_records']-C['low_energy'],C['low_energy']])/C['audio_records']*100
    ax.bar([0,1],percentages,color=[BLUE,ORANGE],width=.55)
    for x,p,n in zip([0,1],percentages,[C['audio_records']-C['low_energy'],C['low_energy']]):
        ax.text(x,p+2,f'{n} 条',ha='center',va='bottom',fontsize=10)
    ax.set(xticks=[0,1],xticklabels=['非低能量','低能量'],ylim=(0,105),ylabel='声学记录占比 / %')
    ax.set_yticks([0,25,50,75,100]);clean(ax)
    save(fig,'fig_p1_audio_energy_share')

    conf=[];wer=[]
    for path in sorted((DATA/'samples').glob('*/metadata.json')):
        wer.append(json.loads(path.read_text())['diagnostic_wer'])
        with np.load(path.parent/'features.npz') as z:conf.extend(z['word_confidence'].tolist())
    assert len(conf)==1932 and len(wer)==100
    assert (np.array(conf)<.1).sum()==350 and (np.array(wer)>.8).sum()==21
    for values,bins,threshold,color,xlabel,ylabel,note,name in [
        (conf,np.linspace(0,1,21),.1,GREEN,'CTC 置信度','词数 / 个','低于 0.1：350 个词','fig_p1_ctc_confidence'),
        (wer,np.linspace(0,2,21),.8,BLUE,'诊断 WER','片段数 / 条','高于 0.8：21 条片段','fig_p1_diagnostic_wer')]:
        fig,ax=canvas()
        ax.hist(values,bins=bins,color=color,edgecolor='white',lw=.6)
        ax.axvline(threshold,color=OE,ls='--',lw=1.2)
        ax.axvspan(0 if threshold==.1 else threshold,threshold if threshold==.1 else 2,color=ORANGE,alpha=.16)
        ax.text(.18 if threshold==.1 else .43,.93,note,transform=ax.transAxes,va='top',fontsize=10)
        ax.set(xlim=(bins[0],bins[-1]),xlabel=xlabel,ylabel=ylabel)
        ax.yaxis.set_major_locator(MaxNLocator(integer=True,nbins=5));clean(ax)
        save(fig,name)


def quality_flow():
    fig,ax=plt.subplots(figsize=(8.5,2.5));fig.subplots_adjust(left=.015,right=.985,top=.98,bottom=.04)
    ax.set(xlim=(0,1),ylim=(.05,1));ax.axis('off')
    xs=[.01,.27,.53,.79];w=.20
    titles=['保留原文词位置','片段级筛选后','逐词筛选后','再检查模态观测']
    texts=['1932 个词\n100 条样本','1615 个词\n79 条样本','1545 个词\n通过对齐筛选','1387 个词\n通过筛选\n且三模态可观测']
    for x,title,txt in zip(xs,titles,texts):
        ax.text(x+w/2,.94,title,ha='center',fontsize=11.5)
        box(ax,x,.55,w,.31,txt,color='#B7DACB' if x==xs[-1] else '#C2D8EB',
            edge='#347567' if x==xs[-1] else '#3E6689',size=12,linespacing=1.25)
    for x in xs[:-1]:arrow(ax,(x+w+.008,.705),(x+.26-.015,.705),color='#000000')
    losses=['317 个词\n整段受限\n对应 21 条复核样本','70 个词\n置信度不足\n来自其余 79 条样本','158 个词\n缺少面部记录\n已通过对齐筛选']
    for x,txt in zip(xs[1:],losses):
        # 每个分支从筛选前的输入箭头分出，避免将排除数画成筛选后数量的子集。
        # 先竖直下行，再水平进入橙框左侧；箭头尖端与最后一段同向。
        center_y=.08+.30/2
        ax.plot([x-.03,x-.03],[.705,center_y],color='#000000',lw=1)
        ax.annotate('',xy=(x-.008,center_y),xytext=(x-.03,center_y),
                    arrowprops={'arrowstyle':'-|>','lw':1,'color':'#000000',
                                'shrinkA':0,'shrinkB':0})
        box(ax,x,.08,w,.30,txt,color='#F2C99D',edge='#AD7738',size=10.5,linespacing=1.25)
    save(fig,'fig_p1_quality_flow')


def window_example():
    m,d=load('-egA8-b7-3M$_$1');t=d['audio_bounds'].mean(axis=1)
    idx=m['audio_feature_names'].index('Loudness_sma3')
    native=d['audio_frames'][:,idx].astype(float);native[~d['audio_frame_valid']]=np.nan
    fig=plt.figure(figsize=(8.5,4.25))
    top=fig.add_axes([.04,.75,.93,.21]);top.axis('off')
    for x,txt in zip([.015,.355,.705],['原生视听记录\n保留局部时间变化','分别按 0.2 s / 0.5 s 分窗\n按有效交叠时长加权平均','将窗口特征映回原记录\n比较原生与重建特征']):
        box(top,x,.24,.28,.65,txt,size=10)
    arrow(top,(.30,.55),(.35,.55));arrow(top,(.65,.55),(.70,.55))
    ax=fig.add_axes([.085,.22,.39,.43]);right=fig.add_axes([.585,.22,.39,.43])
    ax.plot(t,native,color='#8C959C',lw=1,label='原生记录')
    for width,color in [(.2,BE),(.5,OE)]:
        bounds=window_bounds(m['duration'],width)
        pooled,valid,_=overlap_pool(bounds,d['audio_bounds'],d['audio_frames'],d['audio_frame_valid'])
        rebuilt,_,_=overlap_pool(d['audio_bounds'],bounds,pooled,valid)
        ax.plot(t,rebuilt[:,idx],color=color,lw=1.6,label=f'{width:.1f} s 重建')
    ax.set(xlim=(1.8,3.6),ylim=(0,1.28),xlabel='时间 / s',ylabel='响度')
    ax.set_title('（a）平均后，原有峰谷被削弱',loc='left',pad=10)
    ax.legend(loc='upper right',frameon=False,ncol=3,fontsize=8.3,handlelength=1.2,columnspacing=.5)
    ax.xaxis.set_major_locator(MultipleLocator(.4));clean(ax)
    a,b=d['word_bounds'][3];assert m['words'][3]['original']=='successful'
    crossings={}
    for y,width,color,edge in [(1.,.2,BLUE,BE),(0.,.5,ORANGE,OE)]:
        bins=window_bounds(m['duration'],width)
        hit=np.maximum(0,np.minimum(b,bins[:,1])-np.maximum(a,bins[:,0]))>1e-8
        crossings[str(width)]=int(hit.sum())
        for j,(u,v) in enumerate(bins):
            if v<1.7 or u>2.65:continue
            right.add_patch(Rectangle((u,y-.17),v-u,.34,facecolor=color if hit[j] else '#F3F5F7',
                                      edgecolor=edge,lw=.6,alpha=.65))
        right.plot([a,b],[y,y],color=GE,lw=3)
    for x in [a,b]:right.axvline(x,color=GE,ls='--',lw=.8)
    right.set(xlim=(1.7,2.65),ylim=(-.55,1.6),yticks=[0,1],yticklabels=['0.5 s','0.2 s'],xlabel='时间 / s')
    right.set_title('（b）同一词区间被窗口边界切分',loc='left',pad=10)
    right.text(2.25,1.35,'绿色线：successful',ha='center',fontsize=10,color=GE)
    right.text(2.59,.59,'4 个窗',ha='right',fontsize=9,color=BE)
    right.text(2.59,-.40,'2 个窗',ha='right',fontsize=9,color=OE)
    right.annotate('末窗仍交叠 0.00247 s',xy=(b,1),xytext=(1.79,.44),fontsize=8.5,
                   arrowprops={'arrowstyle':'->','color':BE,'lw':.8},color=BE)
    right.spines['left'].set_visible(False);right.xaxis.set_major_locator(MultipleLocator(.2))
    fig.text(.085,.085,'左：重建误差越低，越接近原生特征。右：只要与至少两个窗口交叠，就计为一个跨界词。',fontsize=10)
    fig.text(.085,.025,'示例为同一条实测样本；本实验比较两种固定窗口，不评价真实词界精度。',fontsize=10,color=DARK)
    assert crossings=={'0.2':4,'0.5':2}
    manifest['checks']['example_crossings']=crossings
    manifest['checks']['example_interval']=[float(a),float(b)]
    save(fig,'fig_p1_window_experiment')


def window_statistics():
    with (HERE/'audited_window_comparison.csv').open(encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
    fig,axes=plt.subplots(1,3,figsize=(8.5,3.3))
    fig.subplots_adjust(left=.08,right=.98,bottom=.32,top=.79,wspace=.43)
    keys=['audio_normalized_reconstruction_mse','vision_normalized_reconstruction_mse','word_boundary_crossing_ratio']
    ns=[100,87,79];titles=['（a）声学：保留多少局部变化','（b）视觉：保留多少局部变化','（c）词区间：多少被切分']
    for ax,key,n,title in zip(axes,keys,ns,titles):
        vals=[np.array([float(r[key]) for r in rows if float(r['window_seconds'])==w and r[key]!='']) for w in [.2,.5]]
        assert all(len(v)==n for v in vals)
        cross=key==keys[-1];scale=100 if cross else 1
        means=[float(v.mean()) for v in vals]
        for w,mean in zip(['0.2','0.5'],means):
            assert abs(mean-audit['window_statistics'][key][w]['mean'])<1e-10
        bp=ax.boxplot([v*scale for v in vals],positions=[1,2],widths=.45,patch_artist=True,
                      medianprops={'color':DARK,'lw':1.3},
                      flierprops={'marker':'o','markersize':2.5,'markeredgecolor':'#8393A0'},
                      whiskerprops={'color':'#73818B'},capprops={'color':'#73818B'})
        for patch,color in zip(bp['boxes'],[BLUE,ORANGE]):patch.set(facecolor=color,alpha=.7)
        ax.scatter([1,2],np.array(means)*scale,marker='D',s=25,color=DARK,zorder=3)
        ax.set(xticks=[1,2],xticklabels=['0.2 s','0.5 s'],xlabel='',
               ylabel='词跨界比例 / %' if cross else '标准化重建误差')
        ax.set_ylim((-4,106) if cross else (-.045,.93 if n==100 else .72))
        ax.set_title(title,loc='left',fontsize=10.5,pad=23);clean(ax)
        ax.text(.5,1.025,f'n = {n} 条；'+('越高，被切分的词越多' if cross else '越低，越接近原生特征'),
                transform=ax.transAxes,ha='center',fontsize=8.7)
        value_text=' / '.join(f'{v*scale:.2f}%' if cross else f'{v:.4f}' for v in means)
        ax.text(.5,-.30,'均值（短窗 / 长窗）\n'+value_text,transform=ax.transAxes,ha='center',fontsize=9,linespacing=1.5)
    fig.text(.08,.06,'横轴：固定窗口宽度。箱体：中间 50% 的样本；横线：中位数；菱形：均值。',fontsize=9.5)
    fig.text(.08,.015,'箱外圆点仍参与统计；词跨界比例先在样本内计算，再对 79 条样本等权平均。',fontsize=9.5)
    manifest['checks']['window_means_match_audit']=True
    save(fig,'fig_p1_window_tradeoff')


def timeline():
    m,d=load('-egA8-b7-3M$_$1');t=d['audio_bounds'].mean(axis=1)
    fig,axs=plt.subplots(4,1,figsize=(8.5,4.0),sharex=True,gridspec_kw={'height_ratios':[1.05,.30,.9,.9]})
    fig.subplots_adjust(left=.14,right=.98,top=.87,bottom=.15,hspace=.18)
    fig.text(.14,.96,'沿虚线向下读：候选词区间 → 同一时段的声学与面部记录',fontsize=11.5,color=DARK)
    fig.canvas.draw();area=axs[0].get_window_extent()
    for target in [2.,2.4,3.,3.4]:
        im,pts=frame(m,d,target)
        height=(.35/1.8*area.width)*(im.shape[0]/im.shape[1])/area.height*1.2
        axs[0].imshow(im,extent=(pts-.175,pts+.175,1.04-height,1.04),aspect='auto')
        axs[0].text(pts,1.075,f'{pts:.1f} s',ha='center',va='bottom',fontsize=10.5)
    axs[0].set_ylim(0,1.2);axs[0].axis('off')
    for ax,label in zip(axs[:2],['实际画面','候选词区间']):
        ax.text(-.045,.50,label,transform=ax.transAxes,ha='right',va='center',fontsize=10.5)
    for k in [3,4,5,6]:
        a,b=d['word_bounds'][k]
        axs[1].add_patch(Rectangle((a,.08),b-a,.84,facecolor='#CFE4D7' if k==3 else '#EDF3EF',edgecolor=GE,lw=.8))
        axs[1].text((a+b)/2,.50,m['words'][k]['original'],ha='center',va='center',fontsize=11)
        for ax in axs[2:]:
            for x in [a,b]:ax.axvline(x,color='#9EAEB9',ls='--',lw=.7,zorder=0)
    axs[1].set(ylim=(0,1),yticks=[]);axs[1].spines[['left','bottom']].set_visible(False);axs[1].tick_params(bottom=False)
    a,b=d['word_bounds'][3]
    for ax in axs[2:]:ax.axvspan(a,b,color=GREEN,alpha=.12,zorder=-1)
    native=d['audio_frames'][:,m['audio_feature_names'].index('Loudness_sma3')].astype(float)
    native[~d['audio_frame_valid']]=np.nan
    axs[2].plot(t,native,color=BE,lw=1);axs[2].set_ylabel('响度')
    peak=audit['examples']['typical_curve']
    axs[2].annotate('该词区间内的响度峰值',xy=(peak['loudness_peak_time'],peak['loudness_peak_value']),
                    xytext=(2.32,1.05),fontsize=9,color=BE,arrowprops={'arrowstyle':'->','lw':.7,'color':BE})
    axs[2].set_ylim(.08,1.23)
    for name,label,color,marker in [('jawOpen','张嘴',GE,'o'),('mouthSmileLeft','左嘴角上扬',OE,'s')]:
        y=d['vision_frames'][:,m['vision_feature_names'].index(name)].astype(float);y[~d['vision_frame_valid']]=np.nan
        axs[3].plot(d['vision_pts'],y,color=color,label=label,lw=1.1,marker=marker,ms=3)
    axs[3].set(ylim=(-.015,.4),yticks=[0,.2,.4],ylabel='面部系数',xlabel='相对视频起点的时间 / s')
    axs[3].legend(loc='upper right',ncol=2,frameon=False,fontsize=9)
    for ax in axs[2:]:clean(ax)
    for ax in axs:ax.set_xlim(1.8,3.6)
    axs[3].xaxis.set_major_locator(MultipleLocator(.2))
    fig.text(.14,.015,'浅绿底对应 successful；虚线为候选词界。画面按标注时刻定位，图片宽度不代表持续时间。',fontsize=9)
    save(fig,'fig_p1_multimodal_readable')


if __name__=='__main__':
    observation();rms();diagnostics_and_flow();concise_individual_figures();window_example();window_statistics();timeline()
    (OUT/'readable_figures_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'figures':list(manifest['figures']),'checks':manifest['checks']},ensure_ascii=False,indent=2))
