# 本程序及代码是在人工智能工具辅助下完成的。

# 人工智能工具名称：GPT-6 Astra

# 版本/型号：gpt-6-astra

# 开发机构/公司：OpenAI

# 版本颁布日期：2026年9月3日

"""问题一新增配图共用的黑体与柔和配色。"""
import matplotlib.pyplot as plt
from plot_histograms import font_families

BLUE = '#4C78A8'
ORANGE = '#E69A52'
TEAL = '#3D8C95'
GREEN = '#68A18A'
RED = '#C96B65'
PURPLE = '#8D78A8'


def apply_style(labelsize=14, fontsize=12):
    plt.rcdefaults()
    plt.rcParams.update({
        'font.family': font_families(), 'font.size': fontsize,
        'axes.labelsize': labelsize, 'axes.titlesize': labelsize,
        'xtick.labelsize': fontsize, 'ytick.labelsize': fontsize,
        'legend.fontsize': 11, 'axes.unicode_minus': False,
        'text.parse_math': False, 'pdf.fonttype': 42, 'svg.fonttype': 'path',
    })
