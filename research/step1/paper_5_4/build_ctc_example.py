"""从已保存的CTC结果绘制词级强制对齐实例；正式输出仅为PDF。"""
from pathlib import Path
import argparse
import json
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Rectangle

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--preview', type=Path)
args = parser.parse_args()
HERE = Path(__file__).resolve().parent
STEP = HERE.parent
sys.path.insert(0, str(STEP))
from common import PROJECT, activate_project_dependencies
activate_project_dependencies()
from features import decode_media

sid = '-egA8-b7-3M$_$1'
source = STEP / 'results' / 'samples' / sid
meta = json.loads((source / 'metadata.json').read_text())
vocab = json.loads((STEP / 'assets/wav2vec2/vocab.json').read_text())
inverse = {v: k for k, v in vocab.items()}
with np.load(source / 'features.npz', allow_pickle=False) as data:
    logp = data['ctc_log_probs'].copy()
    cbounds = data['ctc_bounds'].copy()
    path = data['ctc_state_path'].copy()
    targets = data['ctc_target_ids'].copy()
    owners = data['ctc_character_word'].copy()
    wbounds = data['word_bounds'].copy()
    confidence = data['word_confidence'].copy()

word_ids = [3, 4]
colors = ['#2B718D', '#B36E32']
fills = ['#DCEEF4', '#FBE9D7']
target_indices = np.arange(np.flatnonzero(owners == 3)[0],
                           np.flatnonzero(owners == 4)[-1] + 1)
row_for_target = {int(k): r for r, k in enumerate(target_indices)}
start, end = 1.80, 2.76
centers = cbounds.mean(axis=1)
frame_ids = np.flatnonzero((centers >= start) & (centers <= end))
blank = vocab['<pad>']
row_tokens = np.r_[targets[target_indices], blank]
heat = np.exp(logp[np.ix_(frame_ids, row_tokens)]).T
blank_row = len(target_indices)

# Validate the plotted word bounds and confidence against the actual saved path.
word_records = []
for wid in word_ids:
    token_ids = np.flatnonzero(owners == wid)
    selected = np.flatnonzero((path % 2 == 1) & np.isin((path - 1) // 2, token_ids))
    calculated = [cbounds[selected[0], 0], cbounds[selected[-1], 1]]
    assert np.allclose(calculated, wbounds[wid], atol=1e-10)
    assigned = targets[(path[selected] - 1) // 2]
    q = float(np.exp(logp[selected, assigned].mean()))
    assert np.isclose(q, confidence[wid], atol=1e-6)
    word_records.append({'word_index': wid, 'word': meta['words'][wid]['original'],
                         'interval': calculated, 'confidence': q,
                         'emitting_ctc_frames': selected.tolist()})

media = decode_media(PROJECT / meta['source_path'], 5)
wave_t = np.arange(len(media['waveform'])) / 16000 + media['audio_start']
wave_keep = (wave_t >= start) & (wave_t <= end)
font = '/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc'
matplotlib.font_manager.fontManager.addfont(font)
plt.rcParams.update({'font.family': FontProperties(fname=font).get_name(),
                     'font.size': 11, 'axes.labelsize': 11,
                     'axes.unicode_minus': False, 'pdf.fonttype': 42,
                     'text.parse_math': False})
fig = plt.figure(figsize=(9.0, 8.0))
gs = fig.add_gridspec(3, 2, width_ratios=[1, .025], height_ratios=[1.1, 3.9, 1.1],
                      left=.16, right=.94, bottom=.09, top=.96, hspace=.36, wspace=.045)
ax_wave = fig.add_subplot(gs[0, 0])
ax_heat = fig.add_subplot(gs[1, 0], sharex=ax_wave)
ax_word = fig.add_subplot(gs[2, 0], sharex=ax_wave)
cax = fig.add_subplot(gs[1, 1])
ax_wave.plot(wave_t[wave_keep], media['waveform'][wave_keep], lw=.65, color='#536878')
ax_wave.set_title('（a）原始音频波形', loc='left', pad=9, fontsize=12)
ax_wave.set_ylabel('幅值')
ax_wave.tick_params(labelbottom=False)
ax_wave.spines[['top', 'right']].set_visible(False)
ax_wave.grid(axis='y', color='#E6EBEF', lw=.6)

xedges = np.r_[cbounds[frame_ids, 0], cbounds[frame_ids[-1], 1]]
assert np.allclose(cbounds[frame_ids[:-1], 1], cbounds[frame_ids[1:], 0])
mesh = ax_heat.pcolormesh(xedges, np.arange(len(row_tokens) + 1) - .5,
                           heat, cmap='Greys', vmin=0, vmax=1, shading='flat', rasterized=True)
bar = fig.colorbar(mesh, cax=cax, ticks=[0, .5, 1])
bar.set_label('模型输出概率', fontsize=10)
bar.ax.tick_params(labelsize=9)
bar.outline.set_linewidth(.5)
ax_heat.set_title('（b）目标字符概率与选中位置（圆点）', loc='left', pad=9, fontsize=12)
labels = [f'{r+1:02d}  {inverse[int(targets[k])]}' for r, k in enumerate(target_indices)]
labels.append('空白符')
ax_heat.set_yticks(np.arange(len(row_tokens)), labels, fontsize=10)
ax_heat.set_ylim(len(row_tokens) - .5, -.5)
ax_heat.set_ylabel('目标字符顺序', labelpad=13)
ax_heat.tick_params(axis='y', length=0)
ax_heat.tick_params(axis='x', labelbottom=False)
ax_heat.axhline(blank_row - .5, color='#82909B', lw=.9)
separator_row = int(np.flatnonzero(owners[target_indices] == -1)[0])
ax_heat.axhline(separator_row - .5, color='#C8D0D5', lw=.6)
ax_heat.axhline(separator_row + .5, color='#C8D0D5', lw=.6)
for label, tid in zip(ax_heat.get_yticklabels(), target_indices):
    owner = int(owners[tid])
    if owner in word_ids:
        label.set_color(colors[word_ids.index(owner)])
path_records = []
for f in frame_ids:
    state = int(path[f])
    if state % 2 == 0:
        row, color = blank_row, '#7D8992'
        symbol = '<blank>'
    else:
        ti = (state - 1) // 2
        if ti not in row_for_target:
            continue
        row = row_for_target[ti]
        owner = int(owners[ti])
        color = colors[word_ids.index(owner)] if owner in word_ids else '#7D8992'
        symbol = inverse[int(targets[ti])]
    ax_heat.scatter(centers[f], row, s=23, facecolor=color,
                    edgecolor='white', linewidth=.55, zorder=4)
    path_records.append({'ctc_frame': int(f), 'time_center': float(centers[f]),
                         'expanded_state': state, 'symbol': symbol, 'plot_row': row})

ax_word.set_title('（c）按原文词合并后的候选时间区间', loc='left', pad=9, fontsize=12)
ax_word.set_ylim(-.68, .68)
ax_word.set_yticks([])
for i, item in enumerate(word_records):
    a, b = item['interval']
    color = colors[i]
    ax_word.add_patch(Rectangle((a, -.08), b-a, .62,
                               facecolor=fills[i], edgecolor=color, lw=1.25))
    ax_word.text((a+b)/2, .23, item['word'], ha='center', va='center', fontsize=12, color=color)
    ax_word.text((a+b)/2, -.28, f'[{a:.3f}, {b:.3f}] s',
                 ha='center', va='center', fontsize=10, color=color)
    for ax in [ax_wave, ax_heat]:
        for boundary in [a, b]:
            ax.axvline(boundary, color=color, ls=(0, (3, 3)), lw=.8, alpha=.8)
    ax_wave.axvspan(a, b, color=fills[i], alpha=.5, zorder=0)
ax_word.spines[['left', 'right', 'top']].set_visible(False)
ax_word.set_xlabel('相对于视频起点的时间 / s', labelpad=8)
ax_word.set_xticks(np.arange(1.8, 2.761, .1))
ax_word.set_xlim(start, end)

out = HERE / 'figures'
out.mkdir(parents=True, exist_ok=True)
pdf = out / 'fig_5_4_ctc_word_alignment.pdf'
fig.savefig(pdf, bbox_inches='tight', pad_inches=.12, facecolor='white')
if args.preview:
    args.preview.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.preview, dpi=180, bbox_inches='tight', pad_inches=.12, facecolor='white')
plt.close(fig)
manifest = {'sample_id': sid, 'source_video': meta['source_path'],
            'feature_file': str(source / 'features.npz'), 'display_interval': [start, end],
            'target_subsequence': ''.join(inverse[int(targets[k])] for k in target_indices),
            'words': word_records, 'plotted_path': path_records,
            'note': '仅读取既有模型结果并解码原音频；不重跑推理。概率热图含重复字符行及空白符行；圆点显示完整转写最优路径的局部结果。词界符不参与词边界与置信度计算。区间未经人工边界标注验证。'}
(HERE / 'ctc_example_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps({'pdf': str(pdf), 'target': manifest['target_subsequence'],
                  'words': word_records}, ensure_ascii=False, indent=2))
