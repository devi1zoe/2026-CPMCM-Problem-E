# 本程序及代码是在人工智能工具辅助下完成的。

# 人工智能工具名称：GPT-6 Astra

# 版本/型号：gpt-6-astra

# 开发机构/公司：OpenAI

# 版本颁布日期：2026年9月3日

"""以既有声学结果绘制论文实例，不重新提取特征或修改实验输出。"""
from pathlib import Path
import argparse
import csv
import json
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--preview', type=Path, help='可选内部检查用PNG路径；正式配图仅输出PDF')
args = parser.parse_args()
HERE = Path(__file__).resolve().parent
STEP = HERE.parent
sys.path.insert(0, str(STEP))
from common import PROJECT, activate_project_dependencies
activate_project_dependencies()
from features import decode_media

SID = json.loads((STEP / 'results/figure_samples.json').read_text())['typical']
SOURCE = STEP / 'results/samples' / SID
meta = json.loads((SOURCE / 'metadata.json').read_text())
with np.load(SOURCE / 'features.npz', allow_pickle=False) as z:
    bounds = z['audio_bounds'].copy()
    values = z['audio_frames'].copy()
    valid = z['audio_frame_valid'].copy()
media = decode_media(PROJECT / meta['source_path'], 5)
t_wave = np.arange(len(media['waveform'])) / 16000 + media['audio_start']
t = bounds.mean(axis=1)
names = meta['audio_feature_names']
start, end = 1.0, 3.2
chosen = (t >= start) & (t <= end)
wave_keep = (t_wave >= start) & (t_wave <= end)

sys.path.insert(0, str(STEP / 'paper_5_2'))
from figure_style import apply_style, BLUE, ORANGE, TEAL, GREEN, RED, PURPLE
apply_style(labelsize=12.5, fontsize=11.5)
fig, axes = plt.subplots(4, 1, figsize=(8.6, 6.0), sharex=True,
                         gridspec_kw={'height_ratios': [1.15, 1, 1, 1]})
fig.subplots_adjust(left=.14, right=.975, top=.95, bottom=.105, hspace=.38)
axes[0].plot(t_wave[wave_keep], media['waveform'][wave_keep], color=BLUE, lw=.55)
axes[0].set_ylabel('波形幅值')
axes[0].set_title('（a）原视频解码音频', loc='left', fontsize=12.5, pad=4)
field_specs = [
    ('Loudness_sma3', '响度', '（b）响度', TEAL),
    ('F0semitoneFrom27.5Hz_sma3nz', '半音值', '（c）半音基频（参考频率 27.5 Hz）', ORANGE),
    ('mfcc1_sma3', 'MFCC1', '（d）第一阶梅尔频率倒谱系数', PURPLE)]
examples = []
for sec in (1.5, 2.0):
    j = int(np.argmin(abs(t-sec)))
    examples.append({'row_index_zero_based': j, 'time_center': float(t[j]),
                     'recorded_interval': bounds[j].tolist(), 'valid': bool(valid[j]),
                     'values': {name: float(values[j, names.index(name)]) for name, *_ in field_specs}})
for ax, (field, ylabel, title, color) in zip(axes[1:], field_specs):
    y = values[:, names.index(field)].astype(float)
    y[~valid] = np.nan
    if field.startswith('F0'):
        y[y == 0] = np.nan
    ax.plot(t[chosen], y[chosen], color=color, lw=1.25)
    ax.margins(y=.20)
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc='left', fontsize=12.5, pad=4)
    for j, e in enumerate(examples):
        v = e['values'][field]
        if field.startswith('F0') and v == 0:
            ax.text(e['time_center']+.025, .13, '未输出基频',
                    transform=ax.get_xaxis_transform(), color=RED, fontsize=11)
            continue
        ax.scatter([e['time_center']], [v], s=25, color=RED, zorder=5)
        ax.annotate(f'{v:.3f}', (e['time_center'], v), xytext=(8, 6 if j == 0 else -16),
                    textcoords='offset points', fontsize=11, color=RED,
                    bbox={'facecolor':'white', 'alpha':.85, 'edgecolor':'none', 'pad':1})
for ax in axes:
    ax.set_xlim(start, end)
    ax.spines[['top','right']].set_visible(True)
    ax.grid(False)
    ax.tick_params(labelsize=11.5)
    for e in examples:
        ax.axvline(e['time_center'], color=RED, lw=.8, linestyle=(0,(4,4)), alpha=.75)
axes[-1].set_xlabel('相对视频起点的时间 / s')
axes[1].set_yticks([.25, .50, .75, 1.00])
axes[2].set_yticks([34, 36, 38])
axes[3].set_yticks([-20, 0, 20, 40])
out = HERE / 'figures'
out.mkdir(parents=True, exist_ok=True)
fig.savefig(out / 'fig_5_3_acoustic_example.pdf', facecolor='white', bbox_inches='tight', pad_inches=.05)
if args.preview:
    args.preview.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.preview, dpi=180, facecolor='white', bbox_inches='tight', pad_inches=.05)
plt.close(fig)
with (HERE / 'acoustic_example_values.csv').open('w', newline='', encoding='utf-8-sig') as f:
    writer = csv.writer(f)
    writer.writerow(['row_index_zero_based','interval_start','interval_end','time_center','valid']+names)
    for j in np.flatnonzero(chosen):
        writer.writerow([j,*bounds[j],t[j],int(valid[j]),*values[j]])
manifest = {'sample_id': SID, 'source_video': meta['source_path'],
            'feature_file': (SOURCE / 'features.npz').relative_to(PROJECT).as_posix(),
            'display_interval': [start,end], 'examples': examples,
            'note': '读取既有特征，未重新提取。F0零值仅在图中隐藏；原数据不变。输出区间并非全部内部分析窗口的完整支持范围。'}
(HERE / 'acoustic_example_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(manifest, ensure_ascii=False, indent=2))
