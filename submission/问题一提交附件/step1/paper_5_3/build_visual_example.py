# 本程序及代码是在人工智能工具辅助下完成的。

# 人工智能工具名称：GPT-6 Astra

# 版本/型号：gpt-6-astra

# 开发机构/公司：OpenAI

# 版本颁布日期：2026年9月3日

"""读取既有视觉结果并回取真实视频帧，仅生成论文示例PDF。"""
from pathlib import Path
import argparse
import csv
import json
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
STEP = HERE.parent
sys.path.insert(0, str(STEP))
from common import PROJECT, activate_project_dependencies
activate_project_dependencies()
from features import decode_media

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--preview', type=Path, help='可选内部检查用PNG；正式配图只交付PDF')
args = parser.parse_args()

SID = '-wny0OAz3g8$_$9'
SOURCE = STEP/'results/samples'/SID
meta = json.loads((SOURCE/'metadata.json').read_text())
names = meta['vision_feature_names']
with np.load(SOURCE/'features.npz', allow_pickle=False) as z:
    pts = z['vision_pts'].copy()
    values = z['vision_frames'].copy()
    valid = z['vision_frame_valid'].copy()
    frame_indices = z['video_frame_indices'].copy()
    face_count = z['vision_face_count'].copy()
media = decode_media(PROJECT/meta['source_path'], 5)
np.testing.assert_allclose(pts, media['video_pts'], rtol=0, atol=1e-12)
assert np.array_equal(frame_indices, media['video_frame_indices'])
assert len(names)==values.shape[1]==52

sys.path.insert(0, str(STEP / 'paper_5_2'))
from figure_style import apply_style, BLUE, ORANGE, TEAL, GREEN, RED, PURPLE
apply_style(labelsize=12.5, fontsize=11.5)
# Keep the common 0--1 scale and image aspect ratios; compact the layout only.
fig = plt.figure(figsize=(9.3,6.1))
outer = fig.add_gridspec(2,1,height_ratios=[1.2,3.7],hspace=.14,
                        left=.10,right=.98,top=.94,bottom=.10)
frame_grid = outer[0].subgridspec(1,4,wspace=.06)
curve_grid = outer[1].subgridspec(4,1,height_ratios=[1,1,1,.27],hspace=.36)
selected = []
for col,time in enumerate([1.2,2.2,4.6,5.6]):
    j=int(np.argmin(abs(pts-time)))
    assert abs(pts[j]-time)<1e-8
    ax=fig.add_subplot(frame_grid[0,col])
    ax.imshow(media['frames'][j][1]); ax.axis('off')
    ax.set_title(f'（{chr(97+col)}）{pts[j]:.2f} s',fontsize=12,pad=3)
    ax.text(.5,-.04,'有效观测' if valid[j] else '未获得可用面部特征',
            transform=ax.transAxes,ha='center',va='top',fontsize=10,
            color=GREEN if valid[j] else RED)
    selected.append(dict(time=float(pts[j]),row_index=j,original_frame_index=int(frame_indices[j]),
                         valid=bool(valid[j]),face_count=int(face_count[j]),
                         values={k:float(values[j,names.index(k)]) for k in
                                 ['jawOpen','mouthSmileLeft','mouthSmileRight','eyeBlinkLeft','eyeBlinkRight']}))

panels=[('（e）张嘴系数',[('jawOpen','jawOpen',TEAL)]),
        ('（f）左右嘴角上扬系数',[('mouthSmileLeft','左侧',BLUE),('mouthSmileRight','右侧',ORANGE)]),
        ('（g）左右眼睑闭合系数',[('eyeBlinkLeft','左侧',BLUE),('eyeBlinkRight','右侧',ORANGE)])]
axes=[]
for row,(title,fields) in enumerate(panels,1):
    ax=fig.add_subplot(curve_grid[row-1,0],sharex=axes[0] if axes else None)
    axes.append(ax)
    for field,label,color in fields:
        y=values[:,names.index(field)].astype(float)
        y[~valid]=np.nan
        ax.plot(pts,y,color=color,lw=1.6,label=label,marker='.',markersize=3.2,
                linestyle='--' if label=='右侧' else '-')
    ax.set(xlim=(0,meta['duration']),ylim=(0,1),ylabel='系数值')
    ax.set_yticks([0,.5,1]); ax.set_title(title,loc='left',fontsize=12,pad=3)
    ax.spines[['top','right']].set_visible(True)
    ax.grid(False)
    ax.tick_params(axis='x',labelbottom=False)
    for ex in selected:
        ax.axvline(ex['time'],color=TEAL,alpha=.45,linestyle='--',linewidth=.8,zorder=0)
    if len(fields)>1: ax.legend(loc='upper right',ncol=2,fontsize=10.5)

mask=fig.add_subplot(curve_grid[3,0],sharex=axes[0])
mask.scatter(pts[valid],np.zeros(sum(valid)),s=49,marker='s',color=GREEN,label='有效')
mask.scatter(pts[~valid],np.zeros(sum(~valid)),s=55,marker='x',linewidth=1.8,color=RED,label='缺失')
mask.set(ylim=(-.65,.65),yticks=[],xlabel='相对视频起点的时间 / s')
mask.set_title('（h）逐帧观测有效性',loc='left',fontsize=12,pad=3)
mask.spines[['top','right','left']].set_visible(False)
mask.legend(loc='lower right',bbox_to_anchor=(1,1.02),borderaxespad=0,
            frameon=False,ncol=2,fontsize=10.5)
mask.set_xticks(np.arange(0,6.1,1))

out=HERE/'figures'; out.mkdir(exist_ok=True)
fig.savefig(out/'fig_5_3_visual_example.pdf',bbox_inches='tight',pad_inches=.05)
if args.preview:
    args.preview.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(args.preview,dpi=170,bbox_inches='tight',pad_inches=.05)
plt.close(fig)

counts={}
for group,prefixes in [('额眉部',('brow',)),('眼部',('eye',)),('面颊与鼻部',('cheek','nose')),
                       ('嘴部',('mouth',)),('下颌',('jaw',)),('其他输出项',('_neutral',))]:
    counts[group]=sum(n.startswith(prefixes) for n in names)
assert sum(counts.values())==52
manifest=dict(sample_id=SID,source_path=meta['source_path'],duration=float(meta['duration']),
              sampling_hz=5,total_frames=len(pts),valid_frames=int(valid.sum()),
              missing_times=pts[~valid].tolist(),group_counts=counts,selected_frames=selected,
              selection_note='从已有结果中选择同时具有系数变化与少量缺失观测的样本；四帧展示不同运动线索及缺失状态，不构成整体性能评价。',
              plot_note='系数直接读取已有输出；无效位置绘图置为NaN断线，原始零占位与掩码不修改。')
(HERE/'visual_example_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
with (HERE/'visual_example_values.csv').open('w',encoding='utf-8-sig',newline='') as f:
    writer=csv.writer(f)
    writer.writerow(['time','original_frame_index','valid','face_count']+names)
    for j in range(len(pts)):
        writer.writerow([pts[j],frame_indices[j],int(valid[j]),face_count[j],*values[j]])
print(json.dumps(manifest,ensure_ascii=False,indent=2))
