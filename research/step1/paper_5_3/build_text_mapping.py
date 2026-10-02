"""从既有词/子词记录绘制论文映射示意图，并复核词向量的平均聚合。"""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.font_manager import FontProperties

HERE = Path(__file__).resolve().parent
SID = '-3g5yACwYnA$_$3'
source = HERE.parent / 'results/samples' / SID
meta = json.loads((source / 'metadata.json').read_text())
word = next(w for w in meta['words'] if w['original'] == 'adhesives')
tokens = [meta['subwords'][j] for j in word['subword_indices']]
with np.load(source / 'features.npz', allow_pickle=False) as z:
    vectors = z['subword_features'][word['subword_indices']]
    pooled = z['text'][word['index']]
    error = float(np.max(np.abs(vectors.mean(axis=0) - pooled)))
    assert vectors.shape == (3, 768) and error < 1e-6
for token in tokens:
    assert min(word['char_end'], token['char_end']) > max(word['char_start'], token['char_start'])

font_dir = Path.home() / '.local/share/fonts/paper'
for filename in ['SimSun.ttc', 'Times.ttf', 'Times Bold.ttf', 'Times Italic.ttf', 'Times Bold Italic.ttf']:
    if (font_dir / filename).is_file():
        matplotlib.font_manager.fontManager.addfont(font_dir / filename)
for family in ['SimSun', 'Times New Roman']:
    matplotlib.font_manager.findfont(FontProperties(family=family), fallback_to_default=False)
plt.rcParams.update({'font.family': ['Times New Roman', 'SimSun'],
                     'font.size': 12, 'pdf.fonttype': 42, 'svg.fonttype': 'none',
                     'axes.unicode_minus': False, 'mathtext.fontset': 'custom',
                     'mathtext.rm': 'Times New Roman',
                     'mathtext.it': 'Times New Roman:italic',
                     'mathtext.bf': 'Times New Roman:bold',
                     'mathtext.bfit': 'Times New Roman:italic:bold'})
fig, ax = plt.subplots(figsize=(12.6, 2.9))
fig.subplots_adjust(left=.015, right=.985, top=.98, bottom=.02)
ax.set(xlim=(0, 15.7), ylim=(.85, 4.38))
ax.axis('off')
ink, muted = '#243746', '#596B79'
blue, teal, amber = '#3E6689', '#347567', '#A77931'

def text(x, y, s, size=12, color=ink, **kw):
    return ax.text(x, y, s, ha='center', va='center', fontsize=size, color=color, **kw)

def box(x, y, w, h, edge, face):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='square,pad=0.02',
                               edgecolor=edge, facecolor=face, linewidth=1.1,
                               joinstyle='miter', zorder=2))

def arrow(a, b):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle='-|>', mutation_scale=10,
                                color='#000000', linewidth=1.05, zorder=3))

for x, label in [(1.35, '原词与字符位置'), (4.05, 'WordPiece 子词'),
                 (6.8, '冻结模型编码'), (9.5, '子词上下文向量'), (13.55, '词级平均聚合')]:
    text(x, 3.4 if x == 1.35 else 4.05, label, 12)

box(.25, 1.85, 2.2, 1.18, blue, '#C2D8EB')
text(1.35, 2.64, word['original'], 16)
text(1.35, 2.18, '[21, 30)', 12, muted)
text(1.35, 1.65, '原文第 6 个词', 11, muted)

ys = [3.35, 2.42, 1.49]
for i, (token, y) in enumerate(zip(tokens, ys), 1):
    box(3.15, y-.35, 1.8, .70, blue, '#C2D8EB')
    text(4.05, y+.12, token['token'], 13)
    text(4.05, y-.17, f"[{token['char_start']}, {token['char_end']})", 10, muted)
    arrow((2.48, 2.44), (3.1, y))
    arrow((4.99, y), (5.63, y))
    box(8.6, y-.35, 1.8, .70, teal, '#B7DACB')
    text(9.5, y+.12, rf'$\boldsymbol{{h}}_{{{i}}}$'.replace('\\boldsymbol', '\\mathbf'), 16)
    text(9.5, y-.18, '768 维', 10, muted)
    arrow((7.95, y), (8.55, y))
    arrow((10.45, y), (11.65, 2.43))

box(5.7, 1.00, 2.2, 2.79, blue, '#A5C6E2')
text(6.8, 2.80, 'DistilBERT', 15)
text(6.8, 2.31, '上下文编码', 12)
text(6.8, 1.80, '读取最后一层', 10, muted)

box(11.7, 1.20, 3.65, 2.47, amber, '#EED39F')
text(13.525, 3.18, r'$\mathbf{x}^{(T)}_k=\frac{\mathbf{h}_1+\mathbf{h}_2+\mathbf{h}_3}{3}$', 18)
arrow((13.525, 2.70), (13.525, 2.34))
text(13.525, 2.10, 'adhesives 的词级向量', 12)
text(13.525, 1.62, '1 × 768 → 文本矩阵中的一行', 11, muted)

out = HERE / 'figures'
out.mkdir(exist_ok=True)
fig.canvas.draw()
for text_artist in fig.findobj(matplotlib.text.Text):
    text_artist.set_color('#000000')
    text_artist.set_alpha(1)
for suffix in ('pdf', 'svg', 'png'):
    fig.savefig(out / f'fig_5_3_text_mapping.{suffix}', dpi=200, facecolor='white')
plt.close(fig)
(HERE / 'text_mapping_manifest.json').write_text(json.dumps({
    'sample_id': SID, 'raw_text': meta['raw_text'], 'word': word, 'subwords': tokens,
    'source_features': str(source / 'features.npz'),
    'pooling_max_absolute_error': error,
    'note': '图中h1、h2、h3为局部示意编号；保存的子词索引为7、8、9（从0开始）。'
}, ensure_ascii=False, indent=2), encoding='utf-8')
print(f'Exported PDF/SVG/PNG; pooling max absolute error = {error}')
