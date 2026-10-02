"""读取已提交的原生记录，重绘八张覆盖率、诊断与固定窗口配图；不进行模型推理。"""
from pathlib import Path
import argparse
import json
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.ticker import MaxNLocator, MultipleLocator

HERE = Path(__file__).resolve().parent
STEP = HERE.parent
DATA = STEP / 'results'
sys.path.insert(0, str(STEP))
sys.path.insert(0, str(STEP / 'paper_5_2'))
from alignment import overlap_pool, window_bounds
from figure_style import apply_style, BLUE, ORANGE, TEAL, GREEN, RED


def load(sample_id):
    folder = DATA / 'samples' / sample_id
    meta = json.loads((folder / 'metadata.json').read_text())
    with np.load(folder / 'features.npz', allow_pickle=False) as z:
        arrays = {key: z[key] for key in z.files}
    return meta, arrays


def build(output_dir=None, preview_dir=None):
    output_dir = Path(output_dir) if output_dir else HERE / 'figures'
    preview_dir = Path(preview_dir) if preview_dir else HERE / 'previews'
    output_dir.mkdir(parents=True, exist_ok=True)
    preview_dir.mkdir(parents=True, exist_ok=True)
    apply_style()
    manifest = {'fonts': {'Chinese': 'SimHei', 'Latin_and_digits': 'DejaVu Sans'},
                'figures': [], 'evidence': {}, 'figure_text': {}}

    def chart():
        return plt.subplots(figsize=(6.4, 4.8), layout='constrained')

    def save(fig, name):
        fig.canvas.draw()
        manifest['figure_text'][name] = sorted({t.get_text() for t in fig.findobj(matplotlib.text.Text)
                                                if t.get_visible() and t.get_text()})
        reference_path = HERE / 'annotation_preservation_check.json'
        if reference_path.exists():
            required = json.loads(reference_path.read_text())['required_annotations'].get(name, [])
            missing = set(required) - set(manifest['figure_text'][name])
            assert not missing, f'{name}缺少原图内容：{sorted(missing)}'
        fig.savefig(output_dir / f'{name}.pdf', bbox_inches='tight', pad_inches=.10)
        fig.savefig(preview_dir / f'{name}.png', dpi=190, bbox_inches='tight', pad_inches=.10)
        manifest['figures'].append(f'figures/{name}.pdf')
        plt.close(fig)

    audit_path = HERE / 'data_audit.json'
    if not audit_path.exists():
        audit_path = STEP / 'validation/data_audit.json'
    audit = json.loads(audit_path.read_text())
    assert audit['status'] == 'passed'
    count = audit['counts']
    coverage = np.array([count['valid_face_frames'] / count['sampled_frames'],
                         count['vision_observed_words'] / count['words']]) * 100
    fig, ax = chart()
    ax.barh([1, 0], coverage, height=.47, color=GREEN, edgecolor='black',
            linewidth=.8, label='有面部记录')
    ax.barh([1, 0], 100-coverage, left=coverage, height=.47, color=ORANGE,
            edgecolor='black', linewidth=.8, label='无面部记录')
    for y, pct, good, total, unit in zip(
            [1, 0], coverage, [count['valid_face_frames'], count['vision_observed_words']],
            [count['sampled_frames'], count['words']], ['帧', '词']):
        ax.text(pct/2, y, f'{good} {unit}', ha='center', va='center', fontsize=12)
        ax.text(pct+(100-pct)/2, y, f'{total-good} {unit}', ha='center', va='center', fontsize=11)
    ax.set(xlim=(0, 100), ylim=(-.6, 1.9), yticks=[0, 1],
           yticklabels=['词位置', '抽样帧'], xlabel='占比 / %')
    ax.xaxis.set_major_locator(MultipleLocator(25))
    ax.legend(loc='upper center', ncol=2)
    save(fig, 'fig_p1_visual_coverage')

    energy = np.array([count['audio_records']-count['low_energy'], count['low_energy']]) / count['audio_records'] * 100
    fig, ax = chart()
    bars = ax.bar([0, 1], energy, color=[BLUE, ORANGE], edgecolor='black', linewidth=.8, width=.6)
    ax.bar_label(bars, labels=[f"{count['audio_records']-count['low_energy']} 条",
                              f"{count['low_energy']} 条"], padding=5, fontsize=12)
    ax.set(xticks=[0, 1], xticklabels=['非低能量', '低能量'],
           ylabel='声学记录占比 / %', ylim=(0, 100))
    ax.yaxis.set_major_locator(MultipleLocator(20))
    save(fig, 'fig_p1_audio_energy_share')
    manifest['evidence']['coverage_and_energy'] = {
        'counts': {key: count[key] for key in ['valid_face_frames', 'sampled_frames',
                   'vision_observed_words', 'words', 'audio_records', 'low_energy']},
        'visual_percentages_frame_then_word': coverage.tolist(),
        'audio_percentages_nonlow_then_low': energy.tolist(),
    }

    sid = '-iRBcNs9oI8$_$9'
    meta, data = load(sid)
    t = data['audio_bounds'].mean(axis=1)
    rms = data['audio_rms']
    low = data['audio_silence']
    threshold = json.loads((STEP / 'config.json').read_text())['silence_rms_threshold']
    assert len(t) == 333 and low.sum() == 147 and data['audio_frame_valid'].all()
    for zoom in [False, True]:
        fig, ax = chart()
        ax.plot(t, rms, color=BLUE, lw=1.2, label='原生 RMS')
        ax.axhline(threshold, color=ORANGE, ls='--', lw=1.1, label='低能量阈值')
        ax.scatter(t[low], rms[low], s=10, color=ORANGE, zorder=3, label='低能量记录')
        ax.set(xlim=(0, meta['duration']), ylim=(0, .01 if zoom else .55),
               xlabel='相对视频起点的时间 / s', ylabel='短时 RMS')
        ax.legend(loc='upper right', fontsize=10, handlelength=1.6)
        ax.xaxis.set_major_locator(MultipleLocator(1))
        if zoom:
            ax.yaxis.set_major_locator(MultipleLocator(.002))
            ax.text(.98, threshold+.00035, f'阈值 {threshold:.7f}', ha='right',
                    transform=ax.get_yaxis_transform(), fontsize=11)
        else:
            ax.axhspan(0, .01, color=ORANGE, alpha=.2)
            ax.text(.03, .94, f'低能量记录：{int(low.sum())} / {len(t)} 条',
                    transform=ax.transAxes, va='top', fontsize=11)
        save(fig, 'fig_p1_rms_zoom' if zoom else 'fig_p1_rms')
    manifest['evidence']['low_energy'] = dict(sample_id=sid, records=len(t), low_energy=int(low.sum()), threshold=threshold)

    confidence, wer = [], []
    actual_counts = dict.fromkeys(['valid_face_frames', 'sampled_frames',
                                  'vision_observed_words', 'words', 'audio_records', 'low_energy'], 0)
    for p in sorted((DATA / 'samples').glob('*/metadata.json')):
        wer.append(json.loads(p.read_text())['diagnostic_wer'])
        with np.load(p.parent / 'features.npz', allow_pickle=False) as z:
            confidence.extend(z['word_confidence'].tolist())
            actual_counts['valid_face_frames'] += int(z['vision_frame_valid'].sum())
            actual_counts['sampled_frames'] += len(z['vision_pts'])
            actual_counts['vision_observed_words'] += int(z['vision_valid'].sum())
            actual_counts['words'] += len(z['word_confidence'])
            actual_counts['audio_records'] += len(z['audio_rms'])
            actual_counts['low_energy'] += int(z['audio_silence'].sum())
    assert actual_counts == manifest['evidence']['coverage_and_energy']['counts']
    confidence, wer = np.array(confidence), np.array(wer)
    assert len(confidence) == 1932 and len(wer) == 100 and (confidence < .1).sum() == 350
    histograms = {}
    for values, bins, color, limit, xlabel, ylabel, name in [
        (confidence, np.linspace(0, 1, 21), GREEN, .1, 'CTC 置信度', '词数 / 个', 'fig_p1_ctc_confidence'),
        (wer, np.linspace(0, 2, 21), BLUE, .8, '诊断 WER', '片段数 / 条', 'fig_p1_diagnostic_wer'),
    ]:
        counts, _ = np.histogram(values, bins=bins)
        assert counts.sum() == len(values)
        fig, ax = chart()
        ax.hist(values, bins=bins, color=color, edgecolor='black', linewidth=.8)
        ax.axvline(limit, color=RED, ls='--', lw=1.1)
        is_confidence = name == 'fig_p1_ctc_confidence'
        ax.axvspan(0 if is_confidence else limit, limit if is_confidence else bins[-1],
                   color=ORANGE, alpha=.16)
        note = (f'低于 0.1：{int((values < limit).sum())} 个词' if is_confidence
                else f'高于 0.8：{int((values > limit).sum())} 条片段')
        ax.text(.18 if is_confidence else .43, .93, note,
                transform=ax.transAxes, va='top', fontsize=11)
        ax.set(xlabel=xlabel, ylabel=ylabel, xlim=(bins[0], bins[-1]))
        ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
        save(fig, name)
        histograms[name] = {'bins': bins.tolist(), 'counts': counts.tolist(), 'threshold': limit}
    manifest['evidence']['diagnostics'] = {'words': len(confidence), 'samples': len(wer),
                                          'low_words': int((confidence < .1).sum()), 'histograms': histograms}

    sid = '-egA8-b7-3M$_$1'
    meta, data = load(sid)
    t = data['audio_bounds'].mean(axis=1)
    index = meta['audio_feature_names'].index('Loudness_sma3')
    native = data['audio_frames'][:, index].copy()
    native[~data['audio_frame_valid']] = np.nan
    fig, ax = chart()
    ax.plot(t, native, color=TEAL, lw=1, label='原生记录')
    for width, color in [(.2, BLUE), (.5, ORANGE)]:
        bounds = window_bounds(meta['duration'], width)
        pooled, valid, _ = overlap_pool(bounds, data['audio_bounds'], data['audio_frames'], data['audio_frame_valid'])
        rebuilt, _, _ = overlap_pool(data['audio_bounds'], bounds, pooled, valid)
        ax.plot(t, rebuilt[:, index], color=color, lw=1.5, label=f'{width:.1f} s 重建')
    ax.set(xlim=(1.8, 3.6), ylim=(0, 1.25), xlabel='时间 / s', ylabel='响度')
    ax.xaxis.set_major_locator(MultipleLocator(.4))
    ax.legend(loc='upper right', ncol=3, fontsize=10, columnspacing=.8, handlelength=1.3)
    save(fig, 'fig_p1_window_reconstruction')

    fig, ax = chart()
    a, b = data['word_bounds'][3]
    assert meta['words'][3]['original'] == 'successful'
    crossings = {}
    for y, width, color in [(1., .2, BLUE), (0., .5, ORANGE)]:
        bins = window_bounds(meta['duration'], width)
        hit = np.maximum(0, np.minimum(b, bins[:, 1]) - np.maximum(a, bins[:, 0])) > 1e-8
        crossings[str(width)] = int(hit.sum())
        ax.text(2.59, y-.39, f'{int(hit.sum())} 个窗', ha='right', fontsize=11)
        if width == .2:
            last = bins[np.flatnonzero(hit)[-1]]
            last_overlap = min(b, last[1]) - max(a, last[0])
        # 浅色填充只标出与候选词区间相交的窗口，边界仍逐个显示。
        rgb = np.array(matplotlib.colors.to_rgb(color))
        fill = tuple(.35 * rgb + .65)
        for j, (u, v) in enumerate(bins):
            if v < 1.7 or u > 2.65:
                continue
            ax.add_patch(Rectangle((u, y-.20), v-u, .4,
                                  facecolor=fill if hit[j] else 'white', edgecolor=color, lw=.8))
        ax.plot([a, b], [y, y], color=GREEN, lw=3.5, solid_capstyle='butt')
    for x in [a, b]:
        ax.axvline(x, color=GREEN, ls='--', lw=.9)
    ax.set(xlim=(1.7, 2.65), ylim=(-.55, 1.6), yticks=[0, 1],
           yticklabels=['0.5 s', '0.2 s'], xlabel='时间 / s')
    ax.xaxis.set_major_locator(MultipleLocator(.2))
    ax.annotate(f'末窗仍交叠 {last_overlap:.5f} s', xy=(b, 1), xytext=(1.79, .44),
                fontsize=11, arrowprops={'arrowstyle': '->', 'color': BLUE, 'lw': .8})
    ax.plot([], [], color=GREEN, lw=3.5, label='绿色线：successful')
    ax.legend(loc='upper center')
    save(fig, 'fig_p1_word_window_overlap')
    assert crossings == {'0.2': 4, '0.5': 2}
    manifest['evidence']['crossings'] = crossings
    manifest['evidence']['window_example'] = {'sample_id': sid, 'word': 'successful', 'word_bounds': [float(a), float(b)]}
    (output_dir.parent / 'selected_figures_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n')
    # 旧总清单继续描述其他配图，指定图的字体与文本单独覆盖记录。
    publication_path = output_dir.parent / 'publication_figures_manifest.json'
    if publication_path.exists():
        publication = json.loads(publication_path.read_text())
        for name, texts in manifest['figure_text'].items():
            publication.setdefault('fonts_by_figure', {})[name] = manifest['fonts']
            publication.setdefault('figure_text', {})[name] = texts
        publication_path.write_text(json.dumps(publication, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({'figures': manifest['figures'], 'fonts': manifest['fonts'], 'status': 'passed'}, ensure_ascii=False))
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--preview-dir', type=Path)
    args = parser.parse_args()
    build(args.output_dir, args.preview_dir)
