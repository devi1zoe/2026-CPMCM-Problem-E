"""时间交叠加权聚合教学示意，不使用或替代实测数据。"""
from pathlib import Path
import argparse
import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Rectangle, FancyBboxPatch
from matplotlib.offsetbox import AnnotationBbox, HPacker, TextArea

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--preview', type=Path)
args = p.parse_args()
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
fig, ax = plt.subplots(figsize=(10.0, 4.6))
fig.subplots_adjust(left=.02, right=.98, bottom=.03, top=.97)
ax.set(xlim=(0,20), ylim=(0,8))
ax.axis('off')
blue, green, gray = '#387D9C', '#438979', '#8D949B'
dark = '#293A47'
X = lambda t: 4.2+(t-.9)/.6*14.3
start,end = 1.,1.45
intervals = [(.9,1.1),(1.1,1.3),(1.3,1.5)]
valid = [True,False,True]
overlap = [max(0,min(end,b)-max(start,a))*v for (a,b),v in zip(intervals,valid)]
weights = [d/sum(overlap) for d in overlap]
assert abs(weights[0]-.4)<1e-10 and abs(weights[2]-.6)<1e-10

def rect(a,b,y,height,face,edge,hatch=None):
    ax.add_patch(Rectangle((X(a),y),X(b)-X(a),height,facecolor=face,edgecolor=edge,
                           hatch=hatch,lw=1.2))
def txt(x,y,s,size=12,color=dark):
    if '$' in s:
        parts = [TextArea(part, textprops={'fontsize': size, 'color': color})
                 for part in re.split(r'(\$[^$]+\$)', s) if part]
        label = HPacker(children=parts, align='baseline', pad=0, sep=0)
        ax.add_artist(AnnotationBbox(label, (x,y), frameon=False,
                                    box_alignment=(.5,.5), pad=0))
    else:
        ax.text(x,y,s,ha='center',va='center',fontsize=size,color=color)

ax.annotate('',xy=(X(1.5)+.4,7.48),xytext=(X(.9)-.12,7.48),
            arrowprops=dict(arrowstyle='-|>',lw=1.1,color=gray))
for t in [.9,1.,1.1,1.2,1.3,1.4,1.45,1.5]:
    ax.plot([X(t)]*2,[7.4,7.56],color=gray,lw=.8)
    txt(X(t),7.78,f'{t:.2f}',10.5)
txt(2.05,7.5,'时间 / s',11)

for boundary in (start,end):
    ax.plot([X(boundary)]*2,[2.85,7.25],color=blue,ls=(0,(4,4)),lw=.9,zorder=0)
rect(start,end,6.3,.66,'#DCEBF4',blue)
txt((X(start)+X(end))/2,6.63,'某个词的候选时间区间：[1.00, 1.45]',12.5,blue)
txt(1.96,6.63,'① 确定词区间',12.5)

txt(1.96,5.05,'② 找到交叠记录',12.5)
for j,((a,b),v) in enumerate(zip(intervals,valid)):
    rect(a,b,4.6,.85,'#EFF6F3' if v else '#F1F2F3',green if v else gray,
         None if v else '///')
    txt((X(a)+X(b))/2,5.12,f'帧 {j+1}：'+('观测有效' if v else '观测无效'),12,
        green if v else '#6D757C')
    txt((X(a)+X(b))/2,4.82,f'代表区间 [{a:.2f}, {b:.2f}]',10.5)
    txt((X(a)+X(b))/2,5.79,rf'$t_{j+1}^{{(V)}}={(a+b)/2:.2f}\,\mathrm{{s}}$',11)
    ax.plot([X((a+b)/2)]*2,[5.47,5.64],color=gray,lw=.9)

txt(1.96,3.52,'③ 计算有效交叠',12.5)
for j,((a,b),v) in enumerate(zip(intervals,valid)):
    lo,hi = max(a,start),min(b,end)
    rect(lo,hi,3.16,.69,'#BBDCD1' if v else '#F1F2F3',green if v else gray,
         None if v else '///')
    txt((X(lo)+X(hi))/2,3.5, '0.20 s → 不计入' if not v else f'{hi-lo:.2f} s',
        11.5,green if v else '#71777D')

txt(1.96,1.78,'④ 归一化并聚合',12.5)
ax.add_patch(FancyBboxPatch((4.2,.72),14.3,1.68,
                           boxstyle='square,pad=.06',
                           facecolor='#F7FAFC',edgecolor='#A2B7C5',lw=1,
                           joinstyle='miter'))
txt(11.35,2.06,r'有效交叠总时长：$0.10+0.15=0.25\,\mathrm{s}$',12)
txt(11.35,1.57,r'权重：$w_1=0.10/0.25=0.4,\quad w_2=0,\quad w_3=0.15/0.25=0.6$',12)
txt(11.35,1.03,r'该词的视觉特征：$\widehat{\boldsymbol{v}}=0.4\boldsymbol{v}_1+0.6\boldsymbol{v}_3$',14,blue)
txt(11.35,.28,'教学构造示例；同一加权规则也用于声学记录。',10.5,gray)
out = here/'figures/fig_5_4_overlap_principle.pdf'
out.parent.mkdir(parents=True,exist_ok=True)
fig.canvas.draw()
for text_artist in fig.findobj(matplotlib.text.Text):
    text_artist.set_color('#000000')
    text_artist.set_alpha(1)
fig.savefig(out,bbox_inches='tight',pad_inches=.05,facecolor='white')
if args.preview:
    fig.savefig(args.preview,dpi=170,bbox_inches='tight',pad_inches=.05,facecolor='white')
plt.close(fig)
print(out)
