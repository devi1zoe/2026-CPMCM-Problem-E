"""分别导出已核验绘图中的坐标轴，供 LaTeX minipage 独立编号排版。"""
import json
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import build_readable_results_figures as source

OUT = Path(__file__).resolve().parents[1] / 'paper_question1/结果分析与表格修订/figures'
exported = []


def export(fig, name):
    groups = {
        'fig_p1_rms_explained': ['fig_p1_rms_single', 'fig_p1_rms_zoom_single'],
        'fig_p1_window_experiment': ['fig_p1_window_reconstruction_single', 'fig_p1_word_window_overlap_single'],
        'fig_p1_window_tradeoff': ['fig_p1_audio_window_error_single', 'fig_p1_vision_window_error_single', 'fig_p1_word_crossing_ratio_single'],
    }
    names = groups.get(name, [name + '_single'])
    axes = fig.axes[1:] if name == 'fig_p1_window_experiment' else fig.axes
    assert len(axes) == len(names)
    for item in [*fig.texts, *fig.legends]:
        item.set_visible(False)
    for ax, filename in zip(axes, names):
        for other in fig.axes:
            other.set_visible(other is ax)
        ax.set_title('', loc='left')
        if name == 'fig_p1_rms_explained':
            ax.legend(handles=[Line2D([], [], color=source.BE, label='原生 RMS'),
                               Line2D([], [], color=source.OE, ls='--', label='低能量阈值'),
                               Line2D([], [], color=source.ORANGE, marker='o', ls='', label='低能量记录')],
                      loc='upper center', bbox_to_anchor=(.5, -.30), ncol=3,
                      frameon=False, fontsize=8, columnspacing=.7, handlelength=1.3)
        fig.canvas.draw()
        # 保留轴标签、图例与统计标注；不包含原合图的总标题和页脚。
        bbox = ax.get_tightbbox(fig.canvas.get_renderer()).transformed(fig.dpi_scale_trans.inverted())
        bbox = bbox.padded(.045)
        fig.savefig(OUT / f'{filename}.pdf', bbox_inches=bbox)
        fig.savefig(OUT / f'{filename}.png', bbox_inches=bbox, dpi=160)
        exported.append(filename)
    plt.close(fig)


if __name__ == '__main__':
    source.save = export
    source.concise_individual_figures()
    source.rms()
    source.window_example()
    source.window_statistics()
    (OUT.parent / 'minipage配图核对.json').write_text(json.dumps({
        'figures': exported, 'checks': source.manifest['checks'],
        'source_sha256': source.manifest['source_sha256'],
        'note': '沿用原绘图的数据与图元，分别导出矢量PDF；未运行模型或修改实验数据。'
    }, ensure_ascii=False, indent=2) + '\n')
    print('\n'.join(exported))
