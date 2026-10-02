"""从既有样本统计表绘制论文5.2的两张直方图，仅依赖NumPy和Matplotlib。"""
from pathlib import Path
import argparse
import csv

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

HERE = Path(__file__).resolve().parent


def font_families():
    for directory in [HERE / 'fonts', Path.home() / '.local/share/fonts/paper']:
        for filename in ['SimHei.ttf', 'simhei.ttf']:
            path = directory / filename
            if path.is_file():
                font_manager.fontManager.addfont(path)
    available = {font.name for font in font_manager.fontManager.ttflist}
    if 'SimHei' not in available:
        raise RuntimeError('需要SimHei黑体；请安装该字体或将SimHei.ttf放入脚本旁的fonts目录。')
    return ['DejaVu Sans', 'SimHei']


def plot_histograms(records, output_dir, formats=('pdf',)):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    durations = np.array([float(row['duration_seconds']) for row in records])
    words = np.array([int(row['word_count']) for row in records])
    if not len(records) or not np.isfinite(durations).all() or (durations < 0).any() or (words < 0).any():
        raise ValueError('统计表为空或存在无效时长、词数。')
    # 沿用原图的分箱边界，频数直接由既有记录计算。
    charts = [
        ('fig_5_2_duration', durations, np.arange(0, 33, 3),
         '实际时长 / s', (0, 30), np.arange(0, 31, 5), 5,
         f'中位数 = {np.median(durations):.3f} s', '#4C78A8'),
        ('fig_5_2_transcript_length', words, np.arange(0, 76, 5),
         '每条转写的词数', (0, 70), np.arange(0, 71, 10), 5,
         f'中位数 = {np.median(words):g}', '#E69A52'),
    ]
    style = {
        'font.family': font_families(), 'font.size': 12,
        'axes.labelsize': 14, 'xtick.labelsize': 12, 'ytick.labelsize': 12,
        'text.color': 'black', 'axes.labelcolor': 'black',
        'xtick.color': 'black', 'ytick.color': 'black',
        'axes.edgecolor': 'black', 'axes.linewidth': .8,
        'axes.spines.top': True, 'axes.spines.right': True,
        'axes.grid': False, 'axes.unicode_minus': False,
        'figure.facecolor': 'white', 'axes.facecolor': 'white',
        'savefig.facecolor': 'white', 'pdf.fonttype': 42,
        'svg.fonttype': 'path', 'text.parse_math': False,
    }
    with plt.rc_context():
        plt.rcdefaults()
        plt.rcParams.update(style)
        for stem, values, bins, xlabel, xlim, xticks, ystep, median_label, color in charts:
            counts, _ = np.histogram(values, bins=bins)
            if counts.sum() != len(records) or values.max() > xlim[1]:
                raise ValueError(f'{stem}的分箱或坐标范围未覆盖全部样本。')
            fig, ax = plt.subplots(figsize=(6.4, 4.8), layout='constrained')
            ax.hist(values, bins=bins, color=color, edgecolor='black', linewidth=.8)
            ax.axvline(np.median(values), color='black', linestyle='--',
                       linewidth=1.1, label=median_label)
            ymax = (int(counts.max()) // ystep + 1) * ystep
            ax.set(xlabel=xlabel, ylabel='样本数 / 条', xlim=xlim, ylim=(0, ymax),
                   xticks=xticks, yticks=np.arange(0, ymax + 1, ystep))
            ax.legend(loc='upper right', fontsize=11)
            for extension in formats:
                fig.savefig(output_dir / f'{stem}.{extension}', dpi=300,
                            bbox_inches='tight', pad_inches=.10)
            plt.close(fig)
            print(f'{stem}: 样本数={counts.sum()}, 分箱频数={counts.tolist()}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=HERE / 'data_observation_samples.csv')
    parser.add_argument('--output-dir', type=Path, default=HERE / 'figures')
    parser.add_argument('--formats', nargs='+', choices=['pdf', 'png', 'svg'], default=['pdf'])
    args = parser.parse_args()
    with args.data.open(encoding='utf-8-sig', newline='') as stream:
        records = list(csv.DictReader(stream))
    plot_histograms(records, args.output_dir, args.formats)


if __name__ == '__main__':
    main()
