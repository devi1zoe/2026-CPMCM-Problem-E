"""绘制与正文三个小标题对应的结果分析思路图及新版技术路线总图。"""
from pathlib import Path
from xml.sax.saxutils import escape
import re
import cairosvg

OUT=Path(__file__).resolve().parent
W,H=760,1140
parts=['''<svg xmlns="http://www.w3.org/2000/svg" width="760" height="1140" viewBox="0 0 760 1140" font-family="SimSun,宋体,STSong,Noto Serif CJK SC,serif" fill="#20344C" text-anchor="middle">
<defs><marker id="result_arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0 0L10 5L0 10Z" fill="#62758A"/></marker></defs>
<rect x="3" y="3" width="754" height="1134" fill="#E0F2E3" stroke="#CFDFD4" stroke-width="2"/>
<circle cx="43" cy="46" r="24" fill="#7E9F3E"/>
<text x="43" y="56" font-size="30" fill="white">3</text>
<text x="82" y="59" text-anchor="start" font-size="34" font-weight="bold" fill="#111">实验结果分析</text>''']

def text(x,y,s,size=22,bold=False,anchor='middle',color='#20344C'):
    parts.append(f'<text x="{x}" y="{y}" font-size="{size}" font-weight="{"bold" if bold else "normal"}" text-anchor="{anchor}" fill="{color}">{escape(s)}</text>')

def rect(x,y,w,h,fill='#FFFFFF',stroke='#A5B4AA',dash=None):
    parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="{fill}" stroke="{stroke}" stroke-width="1.7"'+(f' stroke-dasharray="{dash}"' if dash else '')+'/>')

def box(x,y,w,h,lines,fill='#FFFFFF'):
    rect(x,y,w,h,fill,'#D1A04C' if fill=='#FFF6DF' else '#A5B4AA')
    gap=29
    for i,s in enumerate(lines):
        text(x+w/2,y+h/2+8+(i-(len(lines)-1)/2)*gap,s,23 if i==0 else 20,i==0)

def wire(d,arrow=True):
    parts.append(f'<path d="{d}" fill="none" stroke="#62758A" stroke-width="2"'+(' marker-end="url(#result_arrow)"' if arrow else '')+'/>')

def panel(y,h,n,title):
    rect(20,y,720,h,'#F6FBF6','#BCD1BE')
    text(39,y+34,f'（{n}）{title}',25,True,'start','#334E3B')

box(135,99,490,60,['三模态特征与映射记录'])
wire('M380 159V183')
panel(185,364,1,'质量检查与使用范围')
for x,lines in [(40,['样本完整性','编号与覆盖','原文与标签']),
                (282,['观测有效性','有效音频','可用面部观测']),
                (524,['对齐时间质量','CTC置信度','ASR与原文一致性'])]:
    box(x,245,196,116,lines,'#FFF6DF')
wire('M138 361V386H622V361 M380 361V386',False)
wire('M380 386V407')
box(113,407,534,111,['质量标记与使用范围','观测掩码 · 时间掩码 · 异常标记','阈值敏感性 · 可用位置统计'])
wire('M380 549V575')
panel(577,185,2,'计算核验与实例分析')
box(40,635,326,98,['计算与记录核验','复算权重与聚合','核对词、子词及帧映射'])
box(394,635,326,98,['典型与困难样本对照','核读原文、画面与诊断','解释局部风险与观测缺失'])
wire('M203 733V748H557V733',False)
wire('M380 762V788')
panel(790,188,3,'时间粒度对照分析')
box(97,848,566,101,['0.2 s与0.5 s固定窗口对照','标准化重建误差 · 词跨界比例','分析变化保留与词完整性的权衡'],'#FFF6DF')
wire('M380 978V1005')
box(95,1007,570,96,['结果汇总与可追溯交付','特征与时间映射 · 质量记录','全量明细 · 验证图 · 使用范围'])
parts.append('</svg>')
svg='\n'.join(parts)

def export(base,content,w,h):
    base.with_suffix('.svg').write_text(content,encoding='utf-8')
    cairosvg.svg2pdf(bytestring=content.encode(),write_to=str(base.with_suffix('.pdf')))
    cairosvg.svg2png(bytestring=content.encode(),write_to=str(base.with_suffix('.png')),output_width=int(w*1.5),output_height=int(h*1.5))

export(OUT/'问题一_实验结果分析思路图_正文版',svg,W,H)

# 复用既有矢量总图的前两栏，第三栏替换为上述正文结构；另存，不覆盖旧图。
old=(OUT/'问题一_技术路线总图_宋体精简版.svg').read_text()
old=old.replace('数据输入与三模态特征提取','模态特征提取')
old=old.replace('d="M36 150H944V1080H36Z" fill="#fff"','d="M36 150H944V1080H36Z" fill="#E1EEF9"')
old=old.replace('d="M968 150H1620V1080H968Z" fill="#fff"','d="M968 150H1620V1080H968Z" fill="#FDECDD"')
left_width=round(1640*H/974)
old=re.sub(r'<svg[^>]*>',f'<svg x="0" y="0" width="{left_width}" height="{H}" viewBox="0 126 1640 974" overflow="hidden" font-family="SimSun,宋体,STSong,Noto Serif CJK SC,serif" font-size="26" text-anchor="middle" fill="#20344C">',old,count=1)
right=re.sub(r'<svg[^>]*>',f'<svg x="{left_width}" y="0" width="{W}" height="{H}" viewBox="0 0 {W} {H}" font-family="SimSun,宋体,STSong,Noto Serif CJK SC,serif" fill="#20344C" text-anchor="middle">',svg,count=1)
total_width=left_width+W
full=f'<svg xmlns="http://www.w3.org/2000/svg" width="{total_width}" height="{H}" viewBox="0 0 {total_width} {H}"><rect width="100%" height="100%" fill="white"/>{old}{right}</svg>'
export(OUT/'问题一_技术路线总图_正文分析版',full,total_width,H)
print('已生成结果分析单栏图与技术路线总图：SVG、PDF、PNG。')
