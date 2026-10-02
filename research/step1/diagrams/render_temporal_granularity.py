"""生成概念示意图；所有位置均为示意，不读取或改动实验数据。"""
from pathlib import Path
from xml.sax.saxutils import escape
import cairosvg

OUT = Path(__file__).resolve().parent
BASE = OUT / '问题一_时间粒度差异与对齐需求示意'
INK, MUTED, RED = '#20344C', '#62758A', '#B64F48'
GREEN, BLUE, PURPLE = '#69B39C', '#76A9DF', '#AE88D3'
parts = ['''<svg xmlns="http://www.w3.org/2000/svg" width="1660" height="730" viewBox="0 0 1660 730"
font-family="SimSun,宋体,STSong,Noto Serif CJK SC,serif" fill="#20344C">
<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0 0L10 5L0 10Z" fill="#62758A"/></marker></defs>
<rect width="1660" height="730" fill="white"/>
<title>三模态时间粒度差异与对齐需求示意</title>
<desc>左图以概念性的词区间、重叠声学窗口和视频帧时间点表示时间粒度差异。文本时间边界需要估计。右图表示固定窗口边界可能穿过一个词。全部位置为示意，不是实测数据。</desc>''']

def text(x, y, s, size=24, color=INK, anchor='middle', bold=False):
    parts.append(f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}" text-anchor="{anchor}" font-weight="{"bold" if bold else "normal"}">{escape(s)}</text>')

def rect(x, y, w, h, fill, stroke='none', dash=None, width=2):
    parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="{fill}" stroke="{stroke}" stroke-width="{width}"'+(f' stroke-dasharray="{dash}"' if dash else '')+'/>')

def line(x1, y1, x2, y2, color=MUTED, dash=None, arrow=False, width=2):
    parts.append(f'<path d="M{x1} {y1}L{x2} {y2}" fill="none" stroke="{color}" stroke-width="{width}"'+(f' stroke-dasharray="{dash}"' if dash else '')+(' marker-end="url(#arrow)"' if arrow else '')+'/>')

text(535, 54, '（a）三模态原生单元的时间粒度不同', 29, bold=True)
text(1365, 54, '（b）固定窗口可能切开词', 29, bold=True)
line(1090, 88, 1090, 603, '#D3DEE8')

# 共同物理时间轴；词区间仅用于解释所需的时间关系。
left, right = 248, 1036
text(45, 126, '物理时间', 25, anchor='start')
line(left, 116, right, 116, arrow=True)
text(left, 99, '0', 21)
text(right-6, 99, 'T', 21)

text(45, 210, '文本', 27, anchor='start', bold=True)
text(45, 245, '词／子词单元', 21, MUTED, anchor='start')
words = [(260, 76, 'I'), (352, 224, 'really'), (590, 123, 'like'), (737, 110, 'this'), (879, 145, 'film')]
for i, (x, w, word) in enumerate(words):
    rect(x, 178, w, 65, '#E8F7F1', GREEN, '7 5')
    text(x+w/2, 220, word, 26)
text(645, 280, '虚线词界仅作示意，需由转写与声画建立时间对应', 21, MUTED)

text(45, 357, '语音', 27, anchor='start', bold=True)
text(45, 392, '短时分析窗口', 21, MUTED, anchor='start')
for i in range(23):
    x = 260+i*32
    y = 321+(i % 2)*38
    rect(x, y, 60, 31, '#EAF4FF', BLUE, width=1.5)
text(645, 428, '连续滑动，窗口之间可重叠', 21, MUTED)

text(45, 502, '视觉', 27, anchor='start', bold=True)
text(45, 537, '离散视频帧', 21, MUTED, anchor='start')
for i in range(9):
    x = 278+i*88
    rect(x-25, 470, 50, 43, '#F3ECFC', PURPLE, width=1.5)
    # 抽象图标，只表达独立画面，不伪装成原始视频截图。
    parts.append(f'<circle cx="{x}" cy="484" r="5" fill="none" stroke="{PURPLE}" stroke-width="1.6"/>')
    parts.append(f'<path d="M{x-12} 506Q{x-12} 493 {x} 493Q{x+12} 493 {x+12} 506" fill="none" stroke="{PURPLE}" stroke-width="1.6"/>')
    line(x, 518, x, 533, PURPLE, width=1.5)
    parts.append(f'<circle cx="{x}" cy="539" r="4" fill="{PURPLE}"/>')
text(645, 580, '各帧对应独立的采样时刻', 21, MUTED)

# 独立的局部放大示意，用明确的固定窗口线说明跨界。
text(1364, 126, '局部时间区间示意', 23, MUTED)
line(1142, 163, 1597, 163, arrow=True)
for x in (1156, 1365, 1574):
    line(x, 156, x, 171)
rect(1156, 196, 209, 65, '#F4F6F8', '#B9C4CF')
rect(1365, 196, 209, 65, '#F4F6F8', '#B9C4CF')
text(1260, 238, '固定窗口 j', 23)
text(1470, 238, '固定窗口 j+1', 23)
rect(1202, 328, 316, 71, '#E8F7F1', GREEN, '7 5')
text(1360, 374, 'really（一个完整词）', 24)
line(1365, 181, 1365, 429, RED, '8 6', width=2.5)
text(1365, 472, '窗口边界穿过词内部', 24, RED, bold=True)
text(1365, 518, '同一词的声画观测', 23, MUTED)
text(1365, 552, '可能分散到相邻窗口', 23, MUTED)

rect(40, 617, 1580, 58, '#F3F6F9')
text(830, 655, '相同序列位置 ≠ 相同时间区间；固定时间窗口也不保证词的完整性。', 26, bold=True)
text(830, 714, '注：英文词、区间长度与采样密度均为概念示意，非实测结果；本图不表示原视频音画不同步。', 20, MUTED)
parts.append('</svg>')
svg = '\n'.join(parts)
BASE.with_suffix('.svg').write_text(svg, encoding='utf-8')
cairosvg.svg2png(bytestring=svg.encode(), write_to=str(BASE.with_suffix('.png')), scale=2)
cairosvg.svg2pdf(bytestring=svg.encode(), write_to=str(BASE.with_suffix('.pdf')))
print(BASE)
