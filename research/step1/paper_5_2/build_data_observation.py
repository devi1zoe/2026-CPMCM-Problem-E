"""为论文5.2生成原始数据观察图；不调用模型、不改动既有特征结果。"""
from pathlib import Path
import argparse
import csv
import json
import re
import sys
import textwrap
from collections import Counter

HERE = Path(__file__).resolve().parent
STEP = HERE.parent
PROJECT = STEP.parent
sys.path.insert(0, str(STEP))
from common import activate_project_dependencies, DATA
activate_project_dependencies()

import av
import numpy as np
import openpyxl
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import MaxNLocator
from matplotlib.patches import Rectangle
from features import decode_media

FIG = HERE / 'figures'
FIG.mkdir(parents=True, exist_ok=True)
for weight in ['Regular', 'Bold']:
    font_manager.fontManager.addfont(f'/usr/share/fonts/opentype/noto/NotoSerifCJK-{weight}.ttc')
SERIF_NAME = font_manager.FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc').get_name()
plt.rcParams.update({
    'font.family': [SERIF_NAME, 'DejaVu Serif'], 'text.parse_math': False,
    'font.size': 12, 'axes.titlesize': 15, 'axes.labelsize': 12,
    'xtick.labelsize': 11, 'ytick.labelsize': 11,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.edgecolor': '#90A0AF', 'axes.labelcolor': '#20344C',
    'text.color': '#20344C', 'xtick.color': '#526575', 'ytick.color': '#526575',
    'pdf.fonttype': 42, 'svg.fonttype': 'path', 'axes.unicode_minus': False,
    'figure.facecolor': 'white', 'savefig.facecolor': 'white',
})
BLUE, GREEN, PURPLE, AMBER = '#76A9DF', '#69B39C', '#AE88D3', '#C38B26'


def save(fig, stem):
    if stem == 'fig_5_2_source_examples':
        fig.canvas.draw()
        for text in fig.findobj(matplotlib.text.Text):
            text.set_color('#000000')
            text.set_alpha(1)
    for extension in ['png', 'pdf', 'svg']:
        fig.savefig(FIG / f'{stem}.{extension}', dpi=240, bbox_inches='tight', pad_inches=0.14)
    plt.close(fig)


def dump(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def load_data():
    manifest = json.loads((STEP/'results/input_manifest.json').read_text())
    audit = {r['sample_id']: r for r in json.loads((STEP/'results/media_audit.json').read_text())}
    with (STEP/'results/sample_summary.csv').open(encoding='utf-8-sig') as f:
        summary = {r['sample_id']: r for r in csv.DictReader(f)}
    wb = openpyxl.load_workbook(DATA/'label-100.xlsx', read_only=True, data_only=True)
    sheet = list(wb['label'].values)
    labels = {}
    for values in sheet[1:]:
        r = dict(zip(sheet[0], values))
        labels[f"{r['video_id']}$_${r['clip_id']}"] = r
    wb.close()
    ids = {r['sample_id'] for r in manifest}
    assert len(manifest) == len(ids) == 100
    assert ids == set(labels) == set(audit) == set(summary)
    assert {PROJECT/r['source_path'] for r in manifest} == set(DATA.rglob('*.mp4'))
    records = []
    for r in manifest:
        sid = r['sample_id']
        original = labels[sid]
        assert original['text'] == r['raw_text']
        assert float(original['label']) == r['label']
        assert original['annotation'] == r['annotation']
        words = re.findall(r"[A-Za-z]+(?:['’][A-Za-z]+)*|\d+(?:[.,]\d+)*", r['raw_text'])
        assert len(words) == int(summary[sid]['words'])
        assert abs(audit[sid]['duration'] - float(summary[sid]['duration_seconds'])) < 1e-9
        records.append(dict(sample_id=sid, video_id=r['video_id'], clip_id=r['clip_id'],
                            duration_seconds=audit[sid]['duration'], word_count=len(words),
                            label=r['label'], annotation=r['annotation'], raw_text=r['raw_text'],
                            source_path=r['source_path'], decode_status=audit[sid]['status'],
                            video_pts_monotonic=audit[sid]['video_pts_monotonic']))
    records.sort(key=lambda r:r['sample_id'])
    with (HERE/'data_observation_samples.csv').open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(records[0])); w.writeheader(); w.writerows(records)
    duration = np.array([r['duration_seconds'] for r in records])
    words = np.array([r['word_count'] for r in records])
    stats = dict(sample_count=len(records), video_id_count=len({r['video_id'] for r in records}),
                 duration_seconds=dict(min=float(duration.min()), max=float(duration.max()),
                                       mean=float(duration.mean()), median=float(np.median(duration)),
                                       q1=float(np.quantile(duration,.25)), q3=float(np.quantile(duration,.75)),
                                       total=float(duration.sum())),
                 words=dict(min=int(words.min()), max=int(words.max()), mean=float(words.mean()),
                            median=float(np.median(words)), total=int(words.sum())),
                 annotation_counts=dict(Counter(r['annotation'] for r in records)),
                 decoded_count=sum(r['decode_status']=='decoded' for r in records),
                 monotonic_video_pts_count=sum(r['video_pts_monotonic'] for r in records),
                 word_count_rule="字母词（允许内部撇号）及数字串；标点不单独计词；沿用step1/alignment.py规则",
                 duration_definition="沿用完整解码审计：视频终点和音频终点的较大者，时间以视频流起点为原点",
                 provenance=['step1/results/media_audit.json', 'step1/results/input_manifest.json',
                             '原始label-100.xlsx；与manifest逐行核对', 'step1/results/sample_summary.csv；核对词数与时长'])
    dump(HERE/'data_observation_stats.json', stats)
    return records, stats


def statistical_style():
    """画面示例的字体与字号；四张统计图使用独立脚本。"""
    local_fonts = Path.home() / '.local/share/fonts/paper'
    for filename in ['Times.ttf', 'SimSun.ttc']:
        path = local_fonts / filename
        if path.is_file():
            font_manager.fontManager.addfont(path)
    # 禁止静默回退到默认字体；其他机器可使用已安装的同名字体。
    for family in ['Times New Roman', 'SimSun']:
        font_manager.findfont(font_manager.FontProperties(family=family), fallback_to_default=False)
    return plt.rc_context({'font.family': ['Times New Roman', 'SimSun'],
                         'font.size': 17, 'axes.labelsize': 19,
                         'xtick.labelsize': 17, 'ytick.labelsize': 17,
                         'axes.labelcolor': '#222222', 'text.color': '#222222',
                         'xtick.color': '#333333', 'ytick.color': '#333333'})


def histograms(records):
    from plot_histograms import plot_histograms
    plot_histograms(records, FIG, formats=('png', 'pdf', 'svg'))


def overview(records, stats):
    histograms(records)
    relationships(records, stats)


def relationships(records, stats):
    from plot_relationships import plot_relationships
    assert dict(Counter(r['annotation'] for r in records)) == stats['annotation_counts']
    plot_relationships(records, FIG, formats=('png', 'pdf', 'svg'))


def frame_at(record, fraction):
    target = record['duration_seconds']*fraction
    with av.open(str(PROJECT/record['source_path'])) as container:
        stream = container.streams.video[0]
        origin = float((stream.start_time or 0)*stream.time_base)
        picked = None
        for idx, frame in enumerate(container.decode(video=0)):
            if frame.pts is None:
                raise ValueError('原帧缺少PTS')
            t = float(frame.pts*frame.time_base)-origin
            picked = frame.to_image(), t, idx
            if t >= target:
                break
    if picked is None:
        raise ValueError('视频无可解码帧')
    return picked


def scene_examples(records):
    with statistical_style():
        _scene_examples(records)


def _scene_examples(records):
    index = {r['sample_id']:r for r in records}
    cases = [('-egA8-b7-3M$_$1', '人物近景'),
             ('-9y-fZ3swSY$_$0', '人物与板书'),
             ('-HwX2H8Z4hY$_$2', '文字卡画面'),
             ('-mJ2ud6oKI8$_$1', '器材展示'),
             ('-NFrJFQijFE$_$1', '山体景观'),
             ('-ri04Z7vwnc$_$0', '图形与文字')]
    fig = plt.figure(figsize=(14, 10.4), layout='constrained')
    grid = fig.add_gridspec(2, 3, hspace=.14, wspace=.05)
    provenance = []
    for i,(sid,description) in enumerate(cases):
        r = index[sid]
        im,t,frameidx = frame_at(r,.5)
        sub = grid[i//3,i%3].subgridspec(3,1,height_ratios=[2.3,.95,1.05],hspace=.035)
        ax = fig.add_subplot(sub[0]); ax.imshow(im); ax.axis('off')
        ax.set_title(f'（{chr(97+i)}）{description}', pad=8, fontsize=22)
        meta = fig.add_subplot(sub[1]); meta.axis('off')
        meta.text(.015,.95, 'ID  '+sid, fontsize=17, va='top', color='#222222')
        meta.text(.015,.61, f"帧时刻 {t:.3f} s", fontsize=17, va='top', color='#333333')
        meta.text(.015,.27, f"片段时长 {r['duration_seconds']:.3f} s",
                  fontsize=17, va='top', color='#333333')
        tx = fig.add_subplot(sub[2]); tx.axis('off')
        tx.add_patch(Rectangle((.005,.025),.99,.95,transform=tx.transAxes,
                              facecolor='#F2F6FA',edgecolor='#BACBDA',linewidth=1))
        tx.add_patch(Rectangle((.005,.025),.009,.95,transform=tx.transAxes,
                              facecolor='#76A9DF',edgecolor='none'))
        tx.text(.045,.89, '转写文本节选', fontsize=17, va='top', color='#333333')
        wrapped = textwrap.fill(r['raw_text'], width=36, max_lines=2, placeholder='…',
                                break_long_words=False, break_on_hyphens=False)
        excerpt = wrapped.replace('\n', ' ')
        tx.text(.045,.57, wrapped, fontsize=17, va='top', linespacing=1.2,
                fontfamily='Times New Roman', color='#222222')
        provenance.append(dict(sample_id=sid, frame_description=description, actual_pts=t,
                               original_frame_index=frameidx, requested_fraction=.5,
                               raw_text=r['raw_text'], displayed_excerpt=excerpt, source_path=r['source_path']))
    save(fig,'fig_5_2_source_examples')
    dump(HERE/'source_examples_manifest.json',provenance)


def raw_timeline(records):
    record = next(r for r in records if r['sample_id']=='-egA8-b7-3M$_$1')
    media = decode_media(PROJECT/record['source_path'],5)
    fig = plt.figure(figsize=(13.2,6.4), layout='constrained')
    grid = fig.add_gridspec(4,5,height_ratios=[1.6,.3,1.5,.9],hspace=.05)
    fractions = [.1,.3,.5,.7,.9]
    selected = []
    for i,q in enumerate(fractions):
        target=record['duration_seconds']*q
        k=int(np.argmin(abs(media['video_pts']-target)))
        t,frame = media['frames'][k]
        ax=fig.add_subplot(grid[0,i]);ax.imshow(frame);ax.axis('off')
        ax.set_title(f'{t:.3f} s',fontsize=12)
        selected.append(dict(pts=float(t), original_frame_index=int(media['video_frame_indices'][k])))
    strip=fig.add_subplot(grid[1,:]);strip.axis('off')
    strip.text(.5,.5,'从约 5 Hz 抽样序列中选取的实际视频帧',ha='center',va='center',fontsize=12)
    ax=fig.add_subplot(grid[2,:])
    signal=media['waveform']; step=160; count=len(signal)//step
    blocks=signal[:count*step].reshape(count,step)
    times=(np.arange(count)*step+step/2)/16000+media['audio_start']
    ax.fill_between(times,blocks.min(1),blocks.max(1),color=BLUE,linewidth=0)
    for frame in selected:
        ax.axvline(frame['pts'],color=PURPLE,linestyle='--',linewidth=1)
    ax.set(xlim=(0,media['duration']),xlabel='相对视频起点的时间 / s',ylabel='波形幅值')
    ax.grid(axis='x',color='#E7EDF2',linewidth=.7)
    ax.set_title('16 kHz 单声道音频波形（10 ms 内最小值—最大值包络）',loc='left',fontsize=12)
    tx=fig.add_subplot(grid[3,:]);tx.axis('off')
    tx.text(0,.95, '给定转写（尚未赋予词级时间）：',va='top',fontsize=12)
    tx.text(0,.60,textwrap.fill(record['raw_text'],110),va='top',fontsize=12,fontfamily='DejaVu Serif',linespacing=1.5)
    save(fig,'fig_5_2_raw_timeline')
    dump(HERE/'raw_timeline_manifest.json',dict(sample_id=record['sample_id'],source_path=record['source_path'],
         duration=float(media['duration']),sample_rate=16000,video_target_hz=5,selected_frames=selected,
         waveform_display='10ms min-max envelope, display only; original waveform not modified',
         raw_text=record['raw_text'],word_alignment_shown=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--histograms-only', action='store_true',
                        help='读取既有统计表，仅更新时长和转写词数两张直方图')
    parser.add_argument('--relationships-only', action='store_true',
                        help='读取既有统计表，仅更新时长词数散点图和类别分布图')
    parser.add_argument('--source-examples-only', action='store_true',
                        help='读取既有统计表，仅更新六个原始画面与转写示例')
    args = parser.parse_args()
    if args.histograms_only or args.relationships_only or args.source_examples_only:
        with (HERE/'data_observation_samples.csv').open(encoding='utf-8-sig') as f:
            records = list(csv.DictReader(f))
        for record in records:
            record['duration_seconds'] = float(record['duration_seconds'])
            record['word_count'] = int(record['word_count'])
        if args.histograms_only:
            histograms(records)
        if args.relationships_only:
            stats = json.loads((HERE/'data_observation_stats.json').read_text())
            relationships(records, stats)
        if args.source_examples_only:
            scene_examples(records)
        print('已更新指定图（PNG/PDF/SVG）；字体设置见各绘图函数。')
        return
    records,stats=load_data()
    overview(records,stats)
    scene_examples(records)
    raw_timeline(records)
    print(json.dumps(stats,ensure_ascii=False,indent=2))
    print('已生成四张独立统计图、画面示例总图、可选声画图（PNG/PDF/SVG）。')


if __name__=='__main__':
    main()
