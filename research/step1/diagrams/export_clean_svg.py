from pathlib import Path
from xml.sax.saxutils import escape
import cairosvg
OUT = Path(__file__).resolve().parent
parts = ['''<svg xmlns="http://www.w3.org/2000/svg" width="2400" height="974" viewBox="0 126 2400 974" font-family="SimSun,宋体,STSong,Noto Serif CJK SC,serif" font-size="26" text-anchor="middle" fill="#20344C">
<defs><marker id="arrow" viewBox="0 0 10 10" refX="10" refY="5" markerWidth="5" markerHeight="5" orient="auto"><path d="M0 0L10 5L0 10Z" fill="#62758A"/></marker></defs>
<path d="M0 126H2400V1100H0Z" fill="#fff"/>''']
colors={'gray':('#F2F2F2','#A6A6A6'),'neutral':('#F3F7FC','#8FA9C6'),'text':('#E8F7F1','#69B39C'),'audio':('#EAF4FF','#76A9DF'),'vision':('#F3ECFC','#AE88D3'),'align':('#FFF5DC','#D2A04A'),'quality':('#FDF0E9','#D69275')}
shapes={k:[] for k in colors}
labels=[]
wires=[]
def rectangle(x,y,w,h,r=16):
    return f'M{x} {y}H{x+w}V{y+h}H{x}Z'
# 全图仅使用三档字号：分区标题30、方框主字26、方框小字22。
SECTION_SIZE, MAIN_SIZE, NOTE_SIZE = 30, 26, 22

def text(x,y,s,size=MAIN_SIZE,bold=False,color='#20344C',anchor='middle'):
    labels.append(f'<text x="{x}" y="{y}" font-size="{size}" font-weight="{"bold" if bold else "normal"}" fill="{color}" text-anchor="{anchor}">{escape(s)}</text>')

def box(x,y,w,h,lines,kind='neutral',sizes=None):
    shapes[kind].append(rectangle(x,y,w,h))
    rows=[]
    for i,line in enumerate(lines):
        for fragment in line.split('\n'):
            rows.append((fragment, MAIN_SIZE if i==0 else NOTE_SIZE, i==0))
    for i,(s,size,bold) in enumerate(rows):
        text(x+w/2,y+h/2+9+(i-(len(rows)-1)/2)*32,s,size,bold)
def wire(d,arrow=True,color='#62758A'):
    wires.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"'+(' marker-end="url(#arrow)"' if arrow else '')+'/>')
for x,w,n,title,c,bg in [(36,908,1,'数据输入与三模态特征提取','#397DC6','#F2F2F2'),(968,652,2,'词级时序对齐','#C38B26','#F2F2F2'),(1644,720,3,'质量控制与结果输出','#C77454','#F2F2F2')]:
    parts.append(f'<path d="{rectangle(x,150,w,930,20)}" fill="#fff" stroke="#D3DEE8" stroke-width="2" stroke-dasharray="9 7"/>')
    parts.append(f'<path d="M{x+62} 189A24 24 0 1 1 {x+14} 189A24 24 0 1 1 {x+62} 189Z" fill="{c}"/>')
    text(x+38,199,str(n),30,True,'#fff');text(x+76,200,title,30,True,'#000000','start')
wire('M944 189H968 M1620 189H1644')
box(200,270,240,80,['视频样本'],'gray',sizes=[30])
box(540,270,240,80,['标签表'],'gray',sizes=[30])
box(270,390,440,80,['样本校验','核对编号、文件与标签对应关系'],'gray',sizes=[30,21])
parts.append(f'<path d="{rectangle(52,510,876,546,18)}" fill="#FCFDFF" stroke="#8FA9C6" stroke-width="2"/>')
wire('M320 350V370H660V350',False)
wire('M490 370V390')
wire('M490 470V538 M201 538H779',False)
branches=[
    (76,'text','文本特征提取',[
        ['原始英文文本'],
        ['DistilBERT\n编码'],
        ['词级文本特征']]),
    (365,'audio','语音特征提取',[
        ['16 kHz单声道音频'],
        ['openSMILE'],
        ['声学窗口特征']]),
    (654,'vision','视觉特征提取',[
        ['5 Hz抽样画面'],
        ['MediaPipe\n人脸检测'],
        ['抽样帧视觉特征']])]
for x,kind,title,rows in branches:
    center=x+125
    edge=colors[kind][1]
    parts.append(f'<path d="{rectangle(x-12,558,274,482,16)}" fill="none" stroke="{edge}" stroke-width="1.6" stroke-dasharray="7 6"/>')
    wire(f'M{center} 538V558')
    text(center,592,title,30,True,'#62758A')
    wire(f'M{center} 600V622')
    for i,lines in enumerate(rows):
        y=622+i*146
        box(x,y,250,110,lines,'gray' if i in (0,2) else kind)
        if i<2:wire(f'M{center} {y+110}V{y+146}')
box(1002,270,264,76,['给定英文文本'],'gray')
box(1320,270,264,76,['解码音频'],'gray')
box(1084,382,420,92,['Wav2Vec2 + CTC强制对齐','估计词级时间边界'],'align')
box(1084,510,420,92,['词级候选时间区间','起止时间 · 对齐置信度'],'gray',[30,26])
box(1002,646,266,96,['原生声学特征','窗口时间 · 有效掩码'],'gray',[26,23])
box(1320,646,264,96,['原生视觉特征','帧时间 · 有效掩码'],'gray',[26,23])
box(1074,778,440,96,['时间交叠加权聚合','词级音频25维 · 视觉52维'],'align',[30,23])
box(1002,898,230,64,['词级文本768维'],'gray',[23])
box(1180,974,404,82,['统一词位置的三模态特征','文本 768 · 声学 25 · 视觉 52'],'gray')
wire('M1134 346V364H1452V346',False)
wire('M1294 364V382 M1294 474V510 M1294 602V778 M1294 874V974')
wire('M1135 742V778',True,colors['audio'][1]);wire('M1452 742V778',True,colors['vision'][1]);wire('M1117 962V1015H1180',True,colors['text'][1])
box(1764,248,480,64,['三模态特征与映射记录'],'gray')
box(1668,380,200,144,['样本完整性','编号与覆盖','原文与标签'],'text',[26,23,23])
box(1904,380,200,144,['观测有效性','有效音频','可用人脸'],'audio',[26,23,23])
box(2140,380,200,144,['对齐时间质量','CTC置信度','ASR与原文一致性'],'align',[26,23,21])
box(1764,600,480,96,['汇总特征与质量记录','观测掩码 · 时间掩码 · 异常标记'],'gray',[30,23])
box(1680,752,300,120,['计算与记录核验','复算权重与聚合','核对时间映射'],'gray',[26,23,23])
box(2028,752,300,120,['人工审核','回看原文与声画','记录审核结论'],'gray',[30,23,23])
box(1740,928,528,128,['可追溯结果交付','特征文件 · 全量汇总 · 验证图','处理日志与复现说明'],'gray',[34,26,26])
wire('M2004 312V340 M1768 340H2240 M1768 524V552H2240V524 M2004 524V552 M2004 696V724 M1830 724H2178 M1830 872V904H2178V872',False)
wire('M1768 340V380 M2004 340V380 M2240 340V380 M2004 552V600 M1830 724V752 M2178 724V752 M2004 904V928')
parts.extend(wires)
for kind,paths in shapes.items():
    if not paths: continue
    fill,stroke=colors[kind]
    parts.append(f'<path d="{" ".join(paths)}" fill="{fill}" stroke="{stroke}" stroke-width="2"/>')
parts.extend(labels);parts.append('</svg>')
svg='\n'.join(parts)
base=OUT/'问题一_技术路线总图_宋体精简版'
base.with_suffix('.svg').write_text(svg,encoding='utf-8')
cairosvg.svg2png(bytestring=svg.encode(),write_to=str(base.with_suffix('.png')),output_width=4800,output_height=1948)
print(base)
