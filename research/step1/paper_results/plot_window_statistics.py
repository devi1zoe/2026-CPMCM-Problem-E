"""从已核验的窗口统计表绘制简洁箱线图，保留样本数、均值标记和离群点。"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
STEP = HERE.parent
sys.path.insert(0, str(STEP / 'paper_5_2'))
from figure_style import apply_style, BLUE, ORANGE


def build(output_dir=None, preview_dir=None):
    output_dir = Path(output_dir) if output_dir else HERE / 'figures'
    preview_dir = Path(preview_dir) if preview_dir else HERE / 'previews'
    output_dir.mkdir(parents=True, exist_ok=True)
    preview_dir.mkdir(parents=True, exist_ok=True)
    source = HERE / 'audited_window_comparison.csv'
    audit_path = HERE / 'data_audit.json'
    if not source.exists():
        source = STEP / 'validation/audited_window_comparison.csv'
    if not audit_path.exists():
        audit_path = STEP / 'validation/data_audit.json'
    with source.open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    audit = json.loads(audit_path.read_text())
    assert audit['status'] == 'passed' and len(rows) == 200
    apply_style()
    manifest = {'fonts': {'Chinese': 'SimHei', 'Latin_and_digits': 'DejaVu Sans'},
                'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                'figures': [], 'figure_text': {}, 'evidence': {}}
    specs = [
        ('audio_normalized_reconstruction_mse', '声学标准化重建误差', 'fig_p1_audio_window_error', 100, (-.045, .93)),
        ('vision_normalized_reconstruction_mse', '视觉标准化重建误差', 'fig_p1_vision_window_error', 87, (-.045, .72)),
        ('word_boundary_crossing_ratio', '词跨界比例 / %', 'fig_p1_word_crossing_ratio', 79, (-4, 106)),
    ]
    for key, ylabel, name, n, ylim in specs:
        groups = [[r for r in rows if float(r['window_seconds']) == width and r[key] != '']
                  for width in [.2, .5]]
        assert all(len(group) == len({r['sample_id'] for r in group}) == n for group in groups)
        assert {r['sample_id'] for r in groups[0]} == {r['sample_id'] for r in groups[1]}
        values = [np.array([float(r[key]) for r in group]) for group in groups]
        cross = key == 'word_boundary_crossing_ratio'
        scale = 100 if cross else 1
        means = [float(v.mean()) for v in values]
        for width, v in zip(['0.2', '0.5'], values):
            expected = audit['window_statistics'][key][width]
            q1, median, q3 = np.quantile(v, [.25, .5, .75])
            outliers = ((v < q1-1.5*(q3-q1)) | (v > q3+1.5*(q3-q1))).sum()
            assert np.isfinite(v).all() and len(v) == expected['n']
            np.testing.assert_allclose([v.mean(), q1, median, q3],
                                       [expected[k] for k in ['mean', 'q1', 'median', 'q3']], atol=1e-10, rtol=0)
            assert outliers == expected['outliers']
            assert v.min()*scale >= ylim[0] and v.max()*scale <= ylim[1]

        fig, ax = plt.subplots(figsize=(6.4, 4.8), layout='constrained')
        bp = ax.boxplot([v*scale for v in values], positions=[1, 2], widths=.45,
                        patch_artist=True, showfliers=True,
                        medianprops={'color': 'black', 'linewidth': 1.3},
                        flierprops={'marker': 'o', 'markersize': 4, 'markerfacecolor': 'none', 'markeredgecolor': 'black'},
                        whiskerprops={'color': 'black', 'linewidth': .9},
                        capprops={'color': 'black', 'linewidth': .9})
        for patch, color in zip(bp['boxes'], [BLUE, ORANGE]):
            patch.set(facecolor=color, edgecolor='black', linewidth=.9)
        ax.scatter([1, 2], np.array(means)*scale, marker='D', s=35, color='black', zorder=3, label='均值')
        ax.set(xticks=[1, 2], xticklabels=['0.2 s', '0.5 s'], xlabel='窗口宽度', ylabel=ylabel, ylim=ylim)
        if cross:
            ax.set_yticks([0, 25, 50, 75, 100])
        else:
            ax.set_yticks(np.arange(0, .81 if n == 100 else .61, .2))
        note = f'n = {n}'
        ax.text(.03, .97, note, transform=ax.transAxes, ha='left', va='top', fontsize=11)
        ax.legend(loc='upper right', fontsize=10)
        fig.canvas.draw()
        texts = sorted({t.get_text() for t in fig.findobj(matplotlib.text.Text) if t.get_visible() and t.get_text()})
        assert note in texts and '均值' in texts
        fig.savefig(output_dir / f'{name}.pdf', bbox_inches='tight', pad_inches=.10)
        fig.savefig(preview_dir / f'{name}.png', dpi=190, bbox_inches='tight', pad_inches=.10)
        plt.close(fig)
        manifest['figures'].append(f'figures/{name}.pdf')
        manifest['figure_text'][name] = texts
        manifest['evidence'][key] = {'samples_per_width': n, 'means': means, 'display_scale': scale,
                                     'all_outliers_retained': True, 'required_annotations': [note, '均值']}
    (output_dir.parent / 'window_figures_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n')
    publication_path = output_dir.parent / 'publication_figures_manifest.json'
    if publication_path.exists():
        publication = json.loads(publication_path.read_text())
        for name, texts in manifest['figure_text'].items():
            publication.setdefault('fonts_by_figure', {})[name] = manifest['fonts']
            publication.setdefault('figure_text', {})[name] = texts
        publication_path.write_text(json.dumps(publication, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({'figures': manifest['figures'], 'status': 'passed', 'checks': '样本数、均值、中位数、四分位数与离群点数量均匹配审计记录'}, ensure_ascii=False))
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--preview-dir', type=Path)
    args = parser.parse_args()
    build(args.output_dir, args.preview_dir)
