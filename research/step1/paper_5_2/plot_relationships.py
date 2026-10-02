"""用既有样本表绘制时长—词数散点图与原始类别分布图。"""
from pathlib import Path
from collections import Counter
import argparse
import csv

from plot_histograms import font_families
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent


def plot_relationships(records, output_dir, formats=('pdf',)):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    durations = np.array([float(row['duration_seconds']) for row in records])
    words = np.array([int(row['word_count']) for row in records])
    counts = Counter(row['annotation'] for row in records)
    categories = ['Negative', 'Neutral', 'Positive']
    if (not len(records) or not np.isfinite(durations).all()
            or (durations < 0).any() or (words < 0).any()
            or set(counts) - set(categories)):
        raise ValueError('统计表为空，或存在无效时长、词数、类别。')
    if durations.max() > 31 or words.max() > 72:
        raise ValueError('原图坐标范围无法覆盖全部样本。')

    def save(fig, stem):
        for extension in formats:
            fig.savefig(output_dir / f'{stem}.{extension}', dpi=300,
                        bbox_inches='tight', pad_inches=.10)
        plt.close(fig)

    with plt.rc_context():
        plt.rcdefaults()
        plt.rcParams.update({
            'font.family': font_families(), 'font.size': 12,
            'axes.labelsize': 14, 'xtick.labelsize': 12, 'ytick.labelsize': 12,
            'axes.unicode_minus': False, 'text.parse_math': False,
            'pdf.fonttype': 42, 'svg.fonttype': 'path',
        })
        fig, ax = plt.subplots(figsize=(6.4, 4.8), layout='constrained')
        ax.scatter(durations, words, color='#4C78A8', s=36)
        ax.set(xlabel='实际时长 / s', ylabel='给定转写词数',
               xlim=(0, 31), ylim=(0, 72),
               xticks=np.arange(0, 31, 5), yticks=np.arange(0, 71, 10))
        save(fig, 'fig_5_2_duration_words')

        fig, ax = plt.subplots(figsize=(6.4, 4.8), layout='constrained')
        heights = [counts[key] for key in categories]
        bars = ax.bar(['负向', '中性', '正向'], heights,
                      color=['#E69A52', '#4C78A8', '#68A18A'],
                      edgecolor='black', linewidth=.8)
        ax.bar_label(bars, labels=[str(n) for n in heights],
                     padding=4, fontsize=11)
        ymax = max(70, (max(heights) // 10 + 2) * 10)
        ax.set(ylabel='样本数 / 条', ylim=(0, ymax), yticks=np.arange(0, ymax + 1, 10))
        save(fig, 'fig_5_2_label_distribution')
    print(f'散点数={len(records)}；类别计数={dict(counts)}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=HERE / 'data_observation_samples.csv')
    parser.add_argument('--output-dir', type=Path, default=HERE / 'figures')
    parser.add_argument('--formats', nargs='+', choices=['pdf', 'png', 'svg'], default=['pdf'])
    args = parser.parse_args()
    with args.data.open(encoding='utf-8-sig', newline='') as stream:
        records = list(csv.DictReader(stream))
    plot_relationships(records, args.output_dir, args.formats)


if __name__ == '__main__':
    main()
