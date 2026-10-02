"""结果节独立配图：默认宋体与Times New Roman，时间轴采用SimHei与DejaVu Sans。

只读原生记录、已核验统计与实际视频帧；无模型推理。各统计图单独导出，
仅同一词/同一时间轴的关联记录保留在同一幅图中。
"""
from pathlib import Path
import json
import sys
import av
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Rectangle, Patch
from matplotlib.ticker import MaxNLocator, MultipleLocator

HERE = Path(__file__).resolve().parent
STEP = HERE.parent
PROJECT = STEP.parent
DATA = STEP / 'results'
FIG = HERE / 'figures'
PREVIEW = HERE / 'previews'
sys.path.insert(0, str(STEP))
from alignment import overlap_pool, window_bounds

FONT_DIR = Path.home() / '.local/share/fonts/paper'
for filename in ['Times.ttf', 'SimSun.ttc']:
    font_manager.fontManager.addfont(FONT_DIR / filename)
for family in ['Times New Roman', 'SimSun']:
    font_manager.findfont(font_manager.FontProperties(family=family), fallback_to_default=False)
plt.rcParams.update({
    'font.family': ['Times New Roman', 'SimSun'],
    'font.size': 12, 'axes.labelsize': 13, 'xtick.labelsize': 12, 'ytick.labelsize': 12,
    'legend.fontsize': 11, 'axes.unicode_minus': False, 'text.parse_math': False,
    'pdf.fonttype': 42, 'ps.fonttype': 42,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.edgecolor': '#97A5B1', 'axes.linewidth': .8,
    'axes.labelcolor': '#222222', 'text.color': '#222222',
    'xtick.color': '#333333', 'ytick.color': '#333333',
    'grid.color': '#E8EDF1', 'grid.linewidth': .6,
    'figure.facecolor': 'white', 'savefig.facecolor': 'white',
})
BLUE, GREEN, ORANGE, GRAY, LIGHT = '#79A9CE', '#78A98E', '#EDA45B', '#B8C0C7', '#F0F3F5'
BLUE_EDGE, GREEN_EDGE, ORANGE_EDGE = '#416E91', '#497963', '#B57C42'
manifest = {'fonts': {'Chinese': 'SimSun', 'Latin_and_digits': 'Times New Roman'},
            'figures': [], 'evidence': {}, 'figure_text': {}}


def load(sid):
    folder = DATA / 'samples' / sid
    meta = json.loads((folder / 'metadata.json').read_text())
    with np.load(folder / 'features.npz') as z:
        arrays = {k: z[k] for k in z.files}
    return meta, arrays


def frame(meta, arrays, time):
    j = int(np.argmin(abs(arrays['vision_pts'] - time)))
    index = int(arrays['video_frame_indices'][j])
    with av.open(str(PROJECT / meta['source_path'])) as c:
        for i, f in enumerate(c.decode(video=0)):
            if i == index:
                pts = float(f.pts * f.time_base) - meta['video_origin']
                assert abs(pts - arrays['vision_pts'][j]) < 1e-7
                return f.to_ndarray(format='rgb24'), pts
    raise AssertionError('Frame not found')


def chart(third=False):
    # 独立半栏图采用同一物理画布；在0.48\linewidth排版时坐标文字约10 pt。
    fig, ax = plt.subplots(figsize=(2.65 if third else 4.1, 2.9))
    fig.subplots_adjust(left=.30 if third else .19, right=.97, bottom=.23, top=.95)
    ax.set_axisbelow(True)
    ax.grid(axis='y')
    ax.tick_params(length=3, width=.7)
    return fig, ax


def save(fig, name):
    # 固定画布而非tight裁切，使同一minipage行的绘图区、字体缩放一致。
    fig.canvas.draw()
    if name == 'fig_p1_multimodal_timeline':
        sys.path.insert(0, str(STEP / 'paper_5_2'))
        from plot_histograms import font_families
        families = font_families()
        for text in fig.findobj(matplotlib.text.Text):
            text.set_fontfamily(families)
            text.set_color('#000000')
            text.set_alpha(1)
        # 上方帧时间和转写单词沿用原字体，仅下方坐标与图例改为无衬线字体。
        for ax in fig.axes[:2]:
            for text in ax.texts:
                text.set_fontfamily('Times New Roman')
        fig.canvas.draw()
        manifest.setdefault('fonts_by_figure', {})[name] = {
            'Chinese': 'SimHei', 'Latin_and_digits': 'DejaVu Sans',
            'frame_timestamps_and_transcript': 'Times New Roman'}
    texts = [t.get_text() for t in fig.findobj(matplotlib.text.Text) if t.get_visible() and t.get_text()]
    manifest['figure_text'][name] = sorted(set(texts))
    for ax in fig.axes:
        assert not ax.get_title(), '标题应放在LaTeX中'
    # 关联时间轴单独裁去外边留白；统计图仍使用固定画布。
    crop = {'bbox_inches': 'tight', 'pad_inches': .05} if name == 'fig_p1_multimodal_timeline' else {}
    fig.savefig(FIG / f'{name}.pdf', **crop)
    fig.savefig(PREVIEW / f'{name}.png', dpi=190, **crop)
    manifest['figures'].append(f'figures/{name}.pdf')
    plt.close(fig)


def build_multimodal_timeline():
    # 唯一的多模态总图：同一片段，共享横轴。去除子图标题及解释行。
    sid='-egA8-b7-3M$_$1';m,d=load(sid);t=d['audio_bounds'].mean(axis=1)
    fig,axs=plt.subplots(4,1,figsize=(8.2,4.05),sharex=True,
                         gridspec_kw={'height_ratios':[1.05,.27,.9,.9]})
    fig.subplots_adjust(left=.115,right=.98,bottom=.14,top=.94,hspace=.13)
    fig.canvas.draw()
    image_area = axs[0].get_window_extent()
    targets=[2.,2.4,3.,3.4]
    for target in targets:
        im,pts=frame(m,d,target)
        # 横向位置对应真实时刻，同时保持原始640×360画面的宽高比。
        image_height = (.35/1.8*image_area.width)*(im.shape[0]/im.shape[1])/image_area.height*1.2
        axs[0].imshow(im,extent=(pts-.175,pts+.175,1.04-image_height,1.04),aspect='auto')
        axs[0].text(pts,1.075,f'{pts:.1f} s',ha='center',va='bottom',fontsize=13)
    axs[0].set_ylim(0,1.20);axs[0].axis('off')
    for k in [3,4,5,6]:
        a,b=d['word_bounds'][k]
        axs[1].add_patch(Rectangle((a,.08),b-a,.84,facecolor='#E7F0EA',edgecolor=GREEN_EDGE,lw=.9))
        axs[1].text((a+b)/2,.50,m['words'][k]['original'],ha='center',va='center',fontsize=13)
        for ax in axs[2:]:
            for x in [a,b]:ax.axvline(x,color=GRAY,ls='--',lw=.7,zorder=0)
    axs[1].set(ylim=(0,1),yticks=[]);axs[1].spines[['left','bottom']].set_visible(False);axs[1].tick_params(bottom=False)
    native=d['audio_frames'][:,m['audio_feature_names'].index('Loudness_sma3')].copy()
    native[~d['audio_frame_valid']]=np.nan
    axs[2].plot(t,native,color=BLUE_EDGE,lw=1);axs[2].set_ylabel('响度')
    for name,label,color,marker in [('jawOpen','张嘴',GREEN_EDGE,'o'),('mouthSmileLeft','左嘴角上扬',ORANGE_EDGE,'s')]:
        y=d['vision_frames'][:,m['vision_feature_names'].index(name)].copy();y[~d['vision_frame_valid']]=np.nan
        axs[3].plot(d['vision_pts'],y,color=color,label=label,lw=1.15,marker=marker,ms=3)
    axs[3].set(ylim=(-.015,.4),yticks=[0,.2,.4],ylabel='面部系数',xlabel='时间 / s')
    axs[3].legend(loc='upper right',ncol=2,frameon=False,fontsize=12)
    for ax in axs[2:]:ax.grid(axis='y');ax.set_axisbelow(True)
    for ax in axs:ax.set_xlim(1.8,3.6)
    axs[3].xaxis.set_major_locator(MultipleLocator(.2))
    save(fig,'fig_p1_multimodal_timeline')
    manifest['evidence']['multimodal']={'sample_id':sid,'time_range':[1.8,3.6],'displayed_frame_pts':targets,'vision_ylim':[-.015,.4]}

    return m, d, t, native


def main():
    audit = json.loads((HERE / 'data_audit.json').read_text())
    assert audit['status'] == 'passed'
    count = audit['counts']
    # 两种统计单位和低能量占比分别导出，不在图内重复计数/结论。
    fig, ax = chart()
    val = np.array([count['valid_face_frames']/count['sampled_frames'],
                    count['vision_observed_words']/count['words']]) * 100
    ax.barh([1, 0], val, color=GREEN, height=.47, label='有面部特征')
    ax.barh([1, 0], 100-val, left=val, color=GRAY, height=.47, label='无面部特征')
    ax.set(xlim=(0,100), ylim=(-.6,1.9), yticks=[0,1], yticklabels=['词位置','抽样帧'], xlabel='占比 / %')
    ax.xaxis.set_major_locator(MultipleLocator(25))
    ax.grid(False); ax.grid(axis='x')
    ax.legend(loc='upper center', bbox_to_anchor=(.5,1.02), frameon=False, ncol=1, fontsize=11)
    save(fig, 'fig_p1_visual_coverage')

    fig, ax = chart()
    val = np.array([count['audio_records']-count['low_energy'], count['low_energy']])/count['audio_records']*100
    ax.bar([0,1], val, color=[BLUE,ORANGE], edgecolor=[BLUE_EDGE,ORANGE_EDGE], width=.6)
    ax.set(xticks=[0,1], xticklabels=['非低能量','低能量'], ylabel='声学记录占比 / %', ylim=(0,100))
    ax.yaxis.set_major_locator(MultipleLocator(20))
    save(fig, 'fig_p1_audio_energy_share')

    # 每个词实例独立成图。原画面和它的时间支持不可分离，保留上下关联。
    examples = [('-s9qJ7ATP7w$_$7','work',.8), ('-s9qJ7ATP7w$_$7','day',4.),
                ('-iRBcNs9oI8$_$9','similar',1.6)]
    examples_manifest = []
    for sid, word, target in examples:
        m, d = load(sid); k = next(i for i,w in enumerate(m['words']) if w['original']==word)
        a,b = d['word_bounds'][k]
        js = np.flatnonzero(np.maximum(0,np.minimum(b,d['vision_bounds'][:,1])-np.maximum(a,d['vision_bounds'][:,0]))>0)
        im,pts = frame(m,d,target)
        fig = plt.figure(figsize=(2.6,3.1))
        ax = fig.add_axes([.03,.48,.94,.49]); ax.imshow(im); ax.axis('off')
        ax = fig.add_axes([.15,.20,.78,.21])
        for j in js:
            u,v = d['vision_bounds'][j]; valid = bool(d['vision_frame_valid'][j])
            ax.add_patch(Rectangle((u,.61),v-u,.26,facecolor=GREEN if valid else GRAY,edgecolor='white',lw=1))
        ax.add_patch(Rectangle((a,.10),b-a,.27,facecolor=BLUE,edgecolor=BLUE_EDGE,lw=.8))
        ax.text((a+b)/2,.235,word,ha='center',va='center',fontsize=12)
        left,right=d['vision_bounds'][js[0],0],d['vision_bounds'][js[-1],1]
        ax.set(xlim=(left-.025,right+.025),ylim=(0,1),yticks=[],xlabel='时间 / s')
        ax.set_xticks(d['vision_pts'][js]); ax.set_xticklabels([f'{v:.1f}' for v in d['vision_pts'][js]])
        ax.spines['left'].set_visible(False)
        # 图中仅保留词和坐标；帧时刻、颜色含义、样本ID在正文交代。
        assert bool(d['vision_frame_valid'][js].any()) == bool(d['vision_valid'][k])
        save(fig, 'fig_p1_observation_' + word)
        examples_manifest.append({'sample_id':sid,'word':word,'word_interval':[float(a),float(b)],
                                  'frame_pts':d['vision_pts'][js].tolist(),'valid_frames':int(d['vision_frame_valid'][js].sum()),
                                  'displayed_frame_pts':pts})
    manifest['evidence']['observation_examples'] = examples_manifest

    # 同一RMS曲线的全幅和纵轴放大分别导出。
    sid='-iRBcNs9oI8$_$9';m,d=load(sid);t=d['audio_bounds'].mean(axis=1);r=d['audio_rms']
    threshold=json.loads((STEP/'config.json').read_text())['silence_rms_threshold']
    assert len(t)==333 and int(d['audio_silence'].sum())==147 and d['audio_frame_valid'].all()
    for zoom in [False,True]:
        fig,ax=chart()
        ax.plot(t,r,color=BLUE_EDGE,lw=1.05)
        ax.axhline(threshold,color=ORANGE_EDGE,ls='--',lw=1.1)
        ax.scatter(t[d['audio_silence']],r[d['audio_silence']],s=7,color=ORANGE,zorder=3)
        ax.set(xlim=(0,m['duration']),ylim=(0,.01 if zoom else .55),xlabel='时间 / s',ylabel='RMS')
        ax.xaxis.set_major_locator(MultipleLocator(1))
        if zoom: ax.yaxis.set_major_locator(MultipleLocator(.002))
        save(fig,'fig_p1_rms_zoom' if zoom else 'fig_p1_rms')
    manifest['evidence']['low_energy']={'sample_id':sid,'records':len(t),'low_energy':147,'threshold':threshold}

    confidence=[];wer=[]
    for p in sorted((DATA/'samples').glob('*/metadata.json')):
        meta=json.loads(p.read_text());wer.append(meta['diagnostic_wer'])
        with np.load(p.parent/'features.npz') as z:confidence.extend(z['word_confidence'].tolist())
    c=np.array(confidence);wer=np.array(wer)
    assert len(c)==1932 and len(wer)==100 and (c<.1).sum()==350
    for values,bins,color,edge,thresh,xlabel,ylabel,name in [
        (c,np.linspace(0,1,21),GREEN,GREEN_EDGE,.1,'CTC置信度','词数 / 个','fig_p1_ctc_confidence'),
        (wer,np.linspace(0,2,21),BLUE,BLUE_EDGE,.8,'诊断WER','样本数 / 条','fig_p1_diagnostic_wer')]:
        fig,ax=chart();ax.hist(values,bins=bins,color=color,edgecolor=edge,linewidth=.75)
        ax.axvline(thresh,color='#525D65',ls='--',lw=1.15)
        ax.set(xlabel=xlabel,ylabel=ylabel,xlim=(bins[0],bins[-1]))
        ax.yaxis.set_major_locator(MaxNLocator(integer=True,nbins=5))
        save(fig,name)
    manifest['evidence']['diagnostics']={'words':len(c),'samples':len(wer),'low_words':int((c<.1).sum())}

    m,d,t,native = build_multimodal_timeline()

    # 聚合后的响度与词跨窗示意为独立文件。
    fig,ax=chart();ax.plot(t,native,color='#8C959C',lw=1,label='原生记录')
    for width,color in [(.2,BLUE_EDGE),(.5,ORANGE_EDGE)]:
        bounds=window_bounds(m['duration'],width)
        pooled,valid,_=overlap_pool(bounds,d['audio_bounds'],d['audio_frames'],d['audio_frame_valid'])
        rebuilt,_,_=overlap_pool(d['audio_bounds'],bounds,pooled,valid)
        ax.plot(t,rebuilt[:,m['audio_feature_names'].index('Loudness_sma3')],color=color,lw=1.3,label=f'{width:.1f} s')
    ax.set(xlim=(1.8,3.6),ylim=(0,1.25),xlabel='时间 / s',ylabel='响度')
    ax.xaxis.set_major_locator(MultipleLocator(.4));ax.legend(loc='upper right',ncol=3,frameon=False,columnspacing=.8,handlelength=1.1,fontsize=10)
    save(fig,'fig_p1_window_reconstruction')

    fig,ax=chart();a,b=d['word_bounds'][3];crossings={}
    for y,width,color,edge in [(1.,.2,BLUE,BLUE_EDGE),(0.,.5,ORANGE,ORANGE_EDGE)]:
        bins=window_bounds(m['duration'],width);hit=np.maximum(0,np.minimum(b,bins[:,1])-np.maximum(a,bins[:,0]))>1e-8
        crossings[str(width)]=int(hit.sum())
        for j,(u,v) in enumerate(bins):
            if v<1.7 or u>2.65:continue
            ax.add_patch(Rectangle((u,y-.20),v-u,.4,facecolor=color if hit[j] else LIGHT,alpha=.40,edgecolor=edge,lw=.8))
        ax.plot([a,b],[y,y],color=GREEN_EDGE,lw=3.5,solid_capstyle='butt')
    for x in [a,b]:ax.axvline(x,color=GREEN_EDGE,ls='--',lw=.8)
    ax.set(xlim=(1.7,2.65),ylim=(-.55,1.6),yticks=[0,1],yticklabels=['0.5 s','0.2 s'],xlabel='时间 / s')
    ax.xaxis.set_major_locator(MultipleLocator(.2));ax.grid(False);ax.spines['left'].set_visible(False)
    ax.plot([],[],color=GREEN_EDGE,lw=3.5,label='successful');ax.legend(loc='upper center',frameon=False)
    save(fig,'fig_p1_word_window_overlap')
    assert crossings=={'0.2':4,'0.5':2}
    manifest['evidence']['crossings']=crossings

    # 三种评价量各输出独立的1/3行宽PDF，通过LaTeX minipage排为一行。
    import csv
    with (HERE/'audited_window_comparison.csv').open(encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
    manifest['evidence']['window_statistics']={}
    for key,label,name,n in [('audio_normalized_reconstruction_mse','声学标准化重建误差','fig_p1_audio_window_error',100),
                             ('vision_normalized_reconstruction_mse','视觉标准化重建误差','fig_p1_vision_window_error',87),
                             ('word_boundary_crossing_ratio','词跨界比例 / %','fig_p1_word_crossing_ratio',79)]:
        values=[np.array([float(r[key]) for r in rows if float(r['window_seconds'])==width and r[key]!='']) for width in [.2,.5]]
        assert all(len(v)==n for v in values)
        means=[float(v.mean()) for v in values]
        cross=key=='word_boundary_crossing_ratio';scale=100 if cross else 1
        fig,ax=chart(third=True)
        bp=ax.boxplot([v*scale for v in values],positions=[1,2],widths=.45,patch_artist=True,
                      medianprops={'color':'#263D50','linewidth':1.2},
                      flierprops={'marker':'.','markersize':4,'markeredgecolor':'#7E8E9C'},
                      whiskerprops={'color':'#62717D'},capprops={'color':'#62717D'})
        for patch,color in zip(bp['boxes'],[BLUE,ORANGE]):patch.set(facecolor=color,alpha=.65,edgecolor='#62717D')
        ax.scatter([1,2],np.array(means)*scale,marker='D',s=24,color='#263D50',zorder=3)
        ax.set(xticks=[1,2],xticklabels=['0.2 s','0.5 s'],xlabel='窗口宽度',ylabel=label)
        ax.set_ylim(bottom=-.04 if key.startswith('audio') else 0)
        if cross:ax.set_ylim(0,105);ax.yaxis.set_major_locator(MultipleLocator(20))
        save(fig,name)
        manifest['evidence']['window_statistics'][key]={'samples_per_width':n,'means':means,'display_scale':scale}
    (HERE/'publication_figures_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'figures':manifest['figures'],'fonts':manifest['fonts'],'status':'passed'},ensure_ascii=False,indent=2))

if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--timeline-only', action='store_true', help='仅重绘多模态时间轴图')
    parser.add_argument('--selected-style-only', action='store_true', help='仅重绘八张覆盖率、诊断与固定窗口图，采用SimHei和柔和配色')
    parser.add_argument('--window-statistics-only', action='store_true', help='仅重绘三张窗口对照箱线图，保留样本数、均值与离群点')
    args = parser.parse_args()
    if args.window_statistics_only:
        from plot_window_statistics import build
        build()
    elif args.selected_style_only:
        from plot_selected_results import build
        build()
    elif args.timeline_only:
        build_multimodal_timeline()
        manifest_path = HERE / 'publication_figures_manifest.json'
        if manifest_path.exists():
            existing = json.loads(manifest_path.read_text())
            existing['evidence']['multimodal'] = manifest['evidence']['multimodal']
            existing['figure_text']['fig_p1_multimodal_timeline'] = manifest['figure_text']['fig_p1_multimodal_timeline']
            existing.setdefault('fonts_by_figure', {}).update(manifest['fonts_by_figure'])
            manifest_path.write_text(json.dumps(existing, ensure_ascii=False, indent=2) + '\n')
        print('Updated fig_p1_multimodal_timeline.pdf and PNG preview')
    else:
        main()
        # 全量重绘时也保留八张指定图的最新字体与配色。
        from plot_selected_results import build
        build()
        from plot_window_statistics import build as build_windows
        build_windows()
