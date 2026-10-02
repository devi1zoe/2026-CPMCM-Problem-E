"""绘制CTC折叠及词时间定位原理图；示例是教学构造，不是实测结果。"""
from pathlib import Path
import argparse
from itertools import groupby
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib.offsetbox import AnnotationBbox, HPacker, TextArea

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--preview', type=Path)
args = parser.parse_args()
here = Path(__file__).resolve().parent
font_dir = Path.home() / '.local/share/fonts/paper'
for filename in ['SimSun.ttc', 'Times.ttf', 'Times Bold.ttf', 'Times Italic.ttf', 'Times Bold Italic.ttf']:
    if (font_dir / filename).is_file():
        matplotlib.font_manager.fontManager.addfont(font_dir / filename)
for family in ['SimSun', 'Times New Roman']:
    matplotlib.font_manager.findfont(FontProperties(family=family), fallback_to_default=False)
plt.rcParams.update({'font.family': ['Times New Roman', 'SimSun'],
                     'font.size': 12, 'pdf.fonttype': 42, 'axes.unicode_minus': False,
                     'mathtext.fontset': 'custom', 'mathtext.rm': 'Times New Roman',
                     'mathtext.it': 'Times New Roman:italic',
                     'mathtext.bf': 'Times New Roman:bold',
                     'mathtext.bfit': 'Times New Roman:italic:bold',
                     'mathtext.cal': 'Times New Roman',
                     'mathtext.sf': 'Times New Roman', 'mathtext.tt': 'Times New Roman'})
fig, ax = plt.subplots(figsize=(10.5, 5.2))
fig.subplots_adjust(left=.02, right=.99, top=.98, bottom=.025)
ax.set(xlim=(0, 20), ylim=(.8, 12.7))
ax.axis('off')
dark = '#263747'
gray = '#6C7881'
border = '#A7B5C0'
palette = {'H': ('#DDEBF6', '#3B709B'),
           'E': ('#DDEFE8', '#43806A'),
           'L': ('#FBE8CC', '#AB7532'),
           'O': ('#EFDFEB', '#8C6080'),
           '_': ('#F1F3F5', '#AAB3BA')}

def box(x, y, w, h, text, fill='#F5F8FB', edge=border, size=12):
    ax.add_patch(FancyBboxPatch((x,y), w,h, boxstyle='square,pad=0.04',
                                linewidth=1.1, edgecolor=edge, facecolor=fill,
                                joinstyle='miter'))
    ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=size,color=dark)

def arrow(a,b,color=gray):
    ax.annotate('',xy=b,xytext=a,arrowprops=dict(arrowstyle='-|>',lw=1.25,color=color,
                                               shrinkA=3,shrinkB=3))

box(4.5,11.5,5.9,.83,'给定转写：HELLO',fill='#EAF2F8',edge='#86ABC5',size=13)
box(11.1,11.5,8.1,.83,'音频 → Wav2Vec2字符概率',size=13)
box(7.4,9.94,9.1,.86,'在给定转写约束下搜索CTC路径',fill='#E6F0E8',edge='#82A68A',size=13)
arrow((7.45,11.5),(10.1,10.8))
arrow((15.15,11.5),(13.5,10.8))
arrow((11.95,9.94),(11.95,9.12))
ax.text(15.3,9.34,'下方为构造示例',fontsize=10.5,color=gray,ha='center')

sequence = list('_HHE__LLL_LLO_')
assert len(sequence) == 14
groups = []
offset = 0
for char, run in groupby(sequence):
    n = len(list(run))
    groups.append((char, offset, offset+n))
    offset += n
assert ''.join(c for c, _, _ in groups if c != '_') == 'HELLO'
x0, dx, h = 4.8, 1.0, .72
rows = [8.16,6.35,4.53]

def tile(char, left, right, y):
    fill, edge = palette[char]
    ax.add_patch(Rectangle((x0+left*dx+.025,y),(right-left)*dx-.05,h,
                           facecolor=fill,edgecolor=edge,lw=1.0))
    ax.text(x0+(left+right)*dx/2,y+h/2,
             r'$\varnothing$' if char=='_' else char,
             ha='center',va='center',fontsize=17,color=gray if char=='_' else dark)

for i, c in enumerate(sequence):
    tile(c,i,i+1,rows[0])
for c,a,b in groups:
    tile(c,a,b,rows[1])
    if c != '_':
        tile(c,a,b,rows[2])
    else:
        # Leave the time location open, without shifting later characters.
        ax.plot([x0+a+.07,x0+b-.07],[rows[2]+h/2]*2,color='#D6DDE3',lw=1,ls=(0,(2,3)))

labels = [('① 一条合法路径','每格对应一个输出时刻'),
          ('② 合并连续重复','仅合并相邻的相同符号'),
          ('③ 删除空白符','按顺序读出 HELLO')]
for y,(title,sub) in zip(rows,labels):
    ax.text(.22,y+.48,title,ha='left',va='center',fontsize=13,color=dark)
    ax.text(.22,y+.05,sub,ha='left',va='center',fontsize=10.5,color=gray)

for a,b in [(rows[0],rows[1]),(rows[1],rows[2])]:
    arrow((11.8,a-.1),(11.8,b+h+.08))
ax.text(14.3,7.65,'两个 L 被空白符隔开',ha='center',fontsize=10.5,color='#986623')
ax.text(14.3,5.85,'删除空白后，两个 L 均保留',ha='center',fontsize=10.5,color='#986623')

ax.text(.22,2.69,'④ 确定词时间区间',fontsize=13,color=dark,va='center')
ax.text(.22,2.24,'保留原始时间位置',fontsize=10.5,color=gray,va='center')
left,right = 1,13
ax.add_patch(Rectangle((x0+left,2.17),right-left,.78,
                       facecolor='#DCECF2',edgecolor='#397E98',lw=1.3))
ax.text(x0+(left+right)/2,2.56,'HELLO',ha='center',va='center',fontsize=17,color='#316F86')
for pos in (left,right):
    ax.plot([x0+pos]*2,[1.55,rows[0]+h+.15],color='#6E99AA',lw=.8,ls=(0,(4,4)),zorder=0)
arrow((x0+4.5,rows[2]-.1),(x0+4.5,3.03),color='#6E99AA')
arrow((x0+9.5,rows[2]-.1),(x0+9.5,3.03),color='#6E99AA')
ax.text(x0+7,3.58,'从首个 H 的起点，到 O 的终点',ha='center',va='center',fontsize=11.5,color='#316F86')
arrow((x0-.15,1.6),(x0+14.4,1.6))
for index in (0,1,13,14):
    ax.plot([x0+index]*2,[1.53,1.67],color=gray,lw=1)
    ax.text(x0+index,1.22,rf'$t_{{{index}}}$',ha='center',va='center',fontsize=12,color=dark)
interval_label = HPacker(children=[
    TextArea('词区间 ', textprops={'fontfamily': 'SimSun', 'fontsize': 12, 'color': '#316F86'}),
    TextArea(r'$I=[t_1,t_{13}]$', textprops={'fontsize': 12, 'color': '#316F86'}),
], align='baseline', pad=0, sep=2)
ax.add_artist(AnnotationBbox(interval_label, (x0+7, 1.18), frameon=False,
                            box_alignment=(.5, .5), pad=0))
ax.text(x0+14.6,1.6,'时间',ha='left',va='center',fontsize=10.5,color=gray)

out = here/'figures'
out.mkdir(parents=True,exist_ok=True)
pdf = out/'fig_5_4_ctc_principle.pdf'
fig.canvas.draw()
for text_artist in fig.findobj(matplotlib.text.Text):
    text_artist.set_color('#000000')
    text_artist.set_alpha(1)
fig.savefig(pdf,bbox_inches='tight',pad_inches=.05,facecolor='white')
if args.preview:
    fig.savefig(args.preview,dpi=180,bbox_inches='tight',pad_inches=.05,facecolor='white')
plt.close(fig)
print(pdf)
