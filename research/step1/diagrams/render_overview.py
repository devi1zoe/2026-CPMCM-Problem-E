"""绘制问题一精简总图；仅生成示意图，不修改实验结果。"""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch
from matplotlib.font_manager import FontProperties

OUT = Path(__file__).resolve().parent
REG = FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
BOLD = FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc')
plt.rcParams.update({'svg.fonttype': 'none', 'pdf.fonttype': 42})
W, H = 2400, 1440
fig, ax = plt.subplots(figsize=(24, 14.4))
fig.subplots_adjust(0, 0, 1, 1)
ax.set(xlim=(0, W), ylim=(H, 0))
ax.axis('off')
INK, LINE = '#20344C', '#62758A'
COLORS = {
    'neutral': ('#F3F7FC', '#8FA9C6'),
    'text': ('#E8F7F1', '#69B39C'),
    'audio': ('#EAF4FF', '#76A9DF'),
    'vision': ('#F3ECFC', '#AE88D3'),
    'align': ('#FFF5DC', '#D2A04A'),
    'quality': ('#FDF0E9', '#D69275'),
}

def text(x, y, s, size=17, bold=False, color=INK, ha='center'):
    ax.text(x, y, s, ha=ha, va='center', fontsize=size,
            fontproperties=BOLD if bold else REG, color=color, linespacing=1.5, zorder=5)

def box(x, y, w, h, lines, kind='neutral', sizes=None):
    bg, edge = COLORS[kind]
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0,rounding_size=17',
                              facecolor=bg, edgecolor=edge, linewidth=1.4, zorder=3))
    if isinstance(lines, str): lines = [lines]
    sizes = sizes or [17] * len(lines)
    gap = 36 if len(lines) > 1 else 0
    cy = y + h / 2
    for i, s in enumerate(lines):
        text(x+w/2, cy+(i-(len(lines)-1)/2)*gap, s, sizes[i], bold=(i==0))

def arrow(points, color=LINE, dashed=False, width=1.55):
    for a, b in zip(points[:-2], points[1:-1]):
        ax.plot([a[0],b[0]], [a[1],b[1]], color=color, lw=width,
                linestyle='--' if dashed else '-', zorder=2)
    ax.add_patch(FancyArrowPatch(points[-2], points[-1], arrowstyle='-|>',
                                mutation_scale=15, linewidth=width, color=color,
                                linestyle='--' if dashed else '-', zorder=2,
                                shrinkA=0, shrinkB=2))

def panel(x, w, n, title, color, bg):
    ax.add_patch(FancyBboxPatch((x, 190), w, 1150, boxstyle='round,pad=0,rounding_size=22',
                              facecolor='#FFFFFF', edgecolor='#D3DEE8', linewidth=1.25,
                              linestyle=(0,(5,4)), zorder=0))
    ax.add_patch(FancyBboxPatch((x, 190), w, 78, boxstyle='round,pad=0,rounding_size=20',
                              facecolor=bg, edgecolor='none', zorder=1))
    ax.add_patch(Circle((x+38,229), 22, facecolor=color, edgecolor='none', zorder=4))
    text(x+38, 229, str(n), 19, True, '#FFFFFF')
    text(x+78, 229, title, 20, True, color, ha='left')

text(1200, 77, '多模态情感特征提取与时序对齐', 31, True)
ax.plot([860, 1540], [123, 123], color='#D5DFEA', lw=1.5)
panel(36,420,1,'原始数据与预处理','#397DC6','#F0F6FF')
panel(482,480,2,'三模态特征提取','#338D79','#F0F8F5')
panel(988,650,3,'词级时序对齐','#C38B26','#FFFAEE')
panel(1664,700,4,'质量控制与结果输出','#C77454','#FFF5EF')
for x1,x2 in [(458,479),(964,985),(1640,1661)]:
    arrow([(x1,229),(x2,229)], color='#9BAEBF', width=1.7)

# 1. 两种来源经编号核对后并行准备三个模态。
box(60,315,184,100,['100条','原始视频'], sizes=[18,17])
box(262,315,170,100,['标签表','转写与标签'], sizes=[18,16])
arrow([(152,415),(152,446),(246,446),(246,466)])
arrow([(347,415),(347,446),(246,446),(246,466)])
box(80,466,332,73,'样本编号与完整性核对', sizes=[17])
arrow([(246,539),(246,572),(63,572),(63,650),(85,650)])
arrow([(63,650),(63,900),(85,900)])
arrow([(63,900),(63,1150),(85,1150)])
box(85,590,340,120,['文本预处理','原始转写 · 字符位置'], 'text', [19,16])
box(85,840,340,120,['音频预处理','16 kHz单声道 · 时间信息'], 'audio', [19,15.5])
box(85,1090,340,120,['视频预处理','约5 Hz抽样 · 真实PTS'], 'vision', [19,16])

# 2. 特征提取与输出维度合并，省略重复输入说明。
for y, kind in [(650,'text'),(900,'audio'),(1150,'vision')]:
    arrow([(425,y),(515,y)], COLORS[kind][1])
text(722,357,'文本 / 语音 / 视觉',19,True,color='#5A6D7F')
text(722,409,'三条并行支路',16,color='#7A8897')
box(515,557,415,186,['冻结DistilBERT','子词编码 → 词级平均','文本特征  L × 768'], 'text', [20,17,19])
box(515,807,415,186,['openSMILE · eGeMAPSv02','音高 · 响度 · 音质','声学窗口特征  $N_a$ × 25'], 'audio', [17,17,19])
box(515,1057,415,186,['MediaPipe人脸检测','52维面部blendshape','抽样帧特征  $N_v$ × 52'], 'vision', [19,17,19])

# 3. 声学对齐输入与特征聚合输入分开表达。
box(1022,315,265,76,'给定英文转写', 'text', [17])
box(1338,315,265,76,'解码音频', 'audio', [17])
arrow([(1154,391),(1154,417),(1313,417),(1313,449)])
arrow([(1470,391),(1470,417),(1313,417),(1313,449)])
box(1105,449,416,100,['Wav2Vec2 + CTC强制对齐','估计词级时间边界'], 'align', [18,17])
arrow([(1313,549),(1313,590)])
box(1105,590,416,100,['词级候选时间区间','起止时间 · 对齐置信度'], 'align', [19,17])
arrow([(1313,690),(1313,950)])
box(1022,767,268,112,['原生声学特征','窗口时间 · 有效掩码'], 'audio', [17,14.5])
box(1338,767,265,112,['原生视觉特征','帧时间 · 有效掩码'], 'vision', [17,14.5])
arrow([(1156,879),(1156,923),(1220,923),(1220,950)], COLORS['audio'][1])
arrow([(1470,879),(1470,923),(1406,923),(1406,950)], COLORS['vision'][1])
box(1093,950,440,120,['时间交叠加权聚合','词级音频25维 · 视觉52维'], 'align', [20,16])
arrow([(1313,1070),(1313,1180)])
box(1022,1095,230,64,'词级文本768维','text',[15.5])
arrow([(1137,1159),(1137,1224),(1165,1224)], COLORS['text'][1])
box(1165,1180,438,108,['统一词位置的三模态特征','L × 768 / L × 25 / L × 52'], 'text', [17,16])

# 4. 三类检查并行，计算核验与人工审核分别保留。
text(2014,330,'三模态特征与映射记录',20,True)
ax.plot([2014,2014],[363,399],color=LINE,lw=1.5,zorder=2)
ax.plot([1781,2247],[399,399],color=LINE,lw=1.5,zorder=2)
for center in [1781,2014,2247]:arrow([(center,399),(center,435)])
box(1689,435,184,174,['样本完整性','编号与覆盖','原文与标签'], 'text', [15,14,14])
box(1922,435,184,174,['观测有效性','有效音频','可用人脸'], 'audio', [15,14,14])
box(2155,435,184,174,['对齐时间质量','CTC置信度','ASR转写一致性'], 'align', [14,14,12.5])
for center in [1781,2014,2247]:
    ax.plot([center,center],[609,650],color=LINE,lw=1.5,zorder=2)
ax.plot([1781,2247],[650,650],color=LINE,lw=1.5,zorder=2)
arrow([(2014,650),(2014,703)])
box(1786,703,456,104,['汇总特征与质量记录','观测掩码 · 时间掩码 · 异常标记'], 'quality', [20,15])
arrow([(2014,807),(2014,850),(1843,850),(1843,900)])
arrow([(2014,850),(2185,850),(2185,900)])
box(1704,900,278,133,['计算与记录核验','复算权重与聚合','核对时间映射'], 'vision', [16.5,15,15])
box(2046,900,278,133,['人工审核','回看原文与声画','记录审核结论'], 'vision', [18,15,15])
arrow([(1843,1033),(1843,1095),(2014,1095),(2014,1152)])
arrow([(2185,1033),(2185,1095),(2014,1095)])
box(1774,1152,480,131,['可追溯结果交付','特征文件 · 全量汇总 · 验证图','处理日志与复现说明'], 'quality', [23,16,16])

text(1200,1391,'L：词数    $N_a$：声学窗口数    $N_v$：抽样帧数    ｜    通过自动质控不等于人工确认词界准确',14,color='#64748B')
for extension in ['svg','pdf','png']:
    fig.savefig(OUT/f'问题一_技术路线总图.{extension}', dpi=180, facecolor='white')
plt.close(fig)
