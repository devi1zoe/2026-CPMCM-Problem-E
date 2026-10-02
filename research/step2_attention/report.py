from common import *
from engine import metrics
import pandas as pd
import numpy as np
import html
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from sklearn.metrics import confusion_matrix

NAMES={'old_gated':'旧TCN门控','pool':'无新增注意力','self':'模态内自注意力','cross':'自注意力+跨模态','cross_clean':'跨模态/无缺失增强'}


def md(df):
    def cell(x):
        if isinstance(x,(float,np.floating)):return '—' if pd.isna(x) else f'{x:.4f}'
        return str(x).replace('|','/').replace('$','\\$').replace('\n',' ')
    return '| '+' | '.join(df.columns)+' |\n| '+' | '.join(['---']*len(df.columns))+' |\n'+'\n'.join('| '+' | '.join(cell(x) for x in r)+' |' for r in df.itertuples(index=False,name=None))


def bootstrap(candidate,reference,c,y,ids,reps=500):
    groups={}
    for i,s in enumerate(ids):groups.setdefault(s.split('$_$')[0],[]).append(i)
    groups=list(groups.values());rng=np.random.default_rng(20260925);values=[]
    for _ in range(reps):
        ix=np.concatenate([groups[j] for j in rng.integers(len(groups),size=len(groups))])
        a=metrics(c[ix],y[ix],candidate[ix]);b=metrics(c[ix],y[ix],reference[ix]);values.append([a['macro_f1']-b['macro_f1'],a['mae']-b['mae']])
    lo,hi=np.quantile(values,[.025,.975],axis=0);a=metrics(c,y,candidate);b=metrics(c,y,reference)
    return [dict(metric=k,delta=a[k]-b[k],lower=float(lo[j]),upper=float(hi[j])) for j,k in enumerate(['macro_f1','mae'])]


def main():
    figdir=ROOT/'results/figures';figdir.mkdir(exist_ok=True)
    font='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc';font_manager.fontManager.addfont(font)
    plt.rcParams.update({'font.family':font_manager.FontProperties(fname=font).get_name(),'axes.unicode_minus':False,'text.parse_math':False,'font.size':10})
    def save(name):plt.tight_layout();plt.savefig(figdir/(name+'.png'),dpi=165,bbox_inches='tight');plt.close()
    lock=json.loads((ROOT/'selection_lock.json').read_text());chosen=lock['selected_attention'];proto=json.loads((ROOT/'protocol.json').read_text())
    df=pd.read_csv(ROOT/'results/valid_metrics.csv');ens=df[df.seed=='ensemble'];table=pd.read_csv(ROOT/'results/selection_comparison.csv').set_index('model')
    seeds=[]
    for name in NAMES:
        for condition in ['clean','continuous_average']:
            sub=df[(df.model==name)&(df.seed!='ensemble')]
            sub=sub[sub.name=='clean'] if condition=='clean' else sub[sub.primary].groupby(['seed','name'])[['macro_f1','mae','accuracy','pearson']].mean().reset_index()
            sub=sub.groupby('seed')[['accuracy','macro_f1','mae','pearson']].mean()
            row=dict(model=name,condition=condition)
            for k in sub:row[k+'_mean']=sub[k].mean();row[k+'_std']=sub[k].std(ddof=1)
            seeds.append(row)
    pd.DataFrame(seeds).to_csv(ROOT/'results/seed_summary.csv',index=False)
    x=np.arange(len(NAMES));fig,ax=plt.subplots(1,2,figsize=(12,4))
    for a,fields in zip(ax,[['clean_f1','robust_f1'],['clean_mae','robust_mae']]):
        for j,k in enumerate(fields):a.bar(x+(j-.5)*.35,table.loc[list(NAMES),k],.35,label=k)
        a.set_xticks(x,list(NAMES.values()),rotation=20,ha='right');a.legend();a.grid(axis='y',alpha=.2)
    ax[0].set_title('验证集：三分类Macro-F1');ax[1].set_title('验证集：强度MAE（越低越好）');save('01_comparison')
    rnd=ens[(ens.position=='random')&(ens.pattern=='single')&ens['sync']]
    fig,ax=plt.subplots(1,2,figsize=(12,4))
    for name in NAMES:
        t=rnd[(rnd.model==name)&(rnd.subset=='TAV')].groupby('rate')[['macro_f1','mae']].agg(['mean','std'])
        for a,k in zip(ax,['macro_f1','mae']):
            a.errorbar(t.index,t[k]['mean'],yerr=t[k]['std'],marker='o',capsize=3,label=NAMES[name]);a.set(xlabel='TAV同步连续缺失比例',ylabel=k);a.grid(alpha=.2)
    ax[0].legend(fontsize=8);ax[0].set_title('3次遮挡均值±标准差（各点为三种子集成）');save('02_missing_ablation')
    cand=ens[ens.model==chosen];effects=cand[cand.primary].groupby(['subset','rate','position'])[['accuracy','macro_f1','mae','pearson']].mean().reset_index()
    effects.to_csv(ROOT/'results/missing_effects.csv',index=False)
    fig,ax=plt.subplots(1,2,figsize=(12,4))
    for subset in ['T','A','V','TA','TV','AV','TAV']:
        g=effects[effects.subset==subset].groupby('rate')[['macro_f1','mae']].mean()
        for a,k in zip(ax,['macro_f1','mae']):a.plot(g.index,g[k],marker='o',label=subset);a.set(xlabel='缺失比例',ylabel=k);a.grid(alpha=.2)
    ax[0].legend(ncol=4);ax[0].set_title(f'注意力候选：{NAMES[chosen]}，四位置等权');save('03_modality_duration')
    fig,ax=plt.subplots(1,2,figsize=(10,4))
    for a,k in zip(ax,['macro_f1','mae']):
        g=effects[effects.subset=='TAV'].pivot(index='rate',columns='position',values=k).reindex(columns=['start','middle','end','random'])
        im=a.imshow(g,cmap='YlGnBu',aspect='auto');a.set_xticks(range(4),['开头','中间','末尾','随机']);a.set_yticks(range(4),['10%','30%','50%','70%']);a.set_title(k)
        for i in range(4):
            for j in range(4):a.text(j,i,f'{g.iloc[i,j]:.3f}',ha='center',color='white' if im.norm(g.iloc[i,j])>.55 else 'black')
    save('04_position')
    profiles=json.loads((ROOT/'results/valid_profiles.json').read_text());v=dict(np.load(ROOT/f'results/valid_{chosen}_predictions.npz'));out=v['out'].mean(1)
    raw=None
    from data import load_attachment2
    raw=load_attachment2()['valid'];c,y=v['c'],v['y'];clean=out[0]
    cm=confusion_matrix(c,clean[:,:3].argmax(-1),labels=[0,1,2]);fig,ax=plt.subplots(1,2,figsize=(10,4))
    ax[0].imshow(cm,cmap='Blues');ax[0].set_xticks(range(3),['负向','中性','正向']);ax[0].set_yticks(range(3),['负向','中性','正向']);ax[0].set(xlabel='预测',ylabel='真实',title='验证集完整输入')
    for i in range(3):
        for j in range(3):ax[0].text(j,i,str(cm[i,j]),ha='center',color='white' if cm[i,j]>.55*cm.max() else 'black')
    ax[1].scatter(y,clean[:,3],s=10,alpha=.35);ax[1].plot([-3,3],[-3,3],'k--');ax[1].set(xlabel='真实强度',ylabel='预测强度',title='强度误差');save('05_validation_errors')
    error=[]
    for name,mask in [('强负向',y<=-1.5),('弱负向',(y> -1.5)&(y<0)),('中性',y==0),('弱正向',(y>0)&(y<1.5)),('强正向',y>=1.5)]:
        error.append(dict(group=name,n=int(mask.sum()),mae=float(abs(clean[mask,3]-y[mask]).mean()),accuracy=float((clean[mask,:3].argmax(-1)==c[mask]).mean()),true_mean=float(y[mask].mean()),pred_mean=float(clean[mask,3].mean())))
    error=pd.DataFrame(error);error.to_csv(ROOT/'results/error_groups.csv',index=False,encoding='utf-8-sig')
    review=pd.DataFrame(dict(sample_id=v['ids'],raw_text=raw['raw_text'],true_class=c,true_intensity=y,predicted_class=clean[:,:3].argmax(-1),predicted_intensity=clean[:,3],abs_error=abs(clean[:,3]-y)))
    review.to_csv(ROOT/'results/validation_prediction_review.csv',index=False,encoding='utf-8-sig');review.nlargest(10,'abs_error').to_csv(ROOT/'results/representative_errors.csv',index=False,encoding='utf-8-sig')
    shape=cand[(cand.subset=='TAV')&(cand.position=='random')&cand.rate.isin([.3,.5])].copy()
    shape['condition']=np.where(shape.pattern=='multi','分区多段',np.where(shape['sync'],'同步单段','独立起点单段'))
    shape=shape.groupby(['rate','condition'])[['macro_f1','mae']].mean().reset_index();shape.to_csv(ROOT/'results/shape_comparison.csv',index=False,encoding='utf-8-sig')
    fig,ax=plt.subplots(1,2,figsize=(10,4))
    for name,g in shape.groupby('condition'):
        for a,k in zip(ax,['macro_f1','mae']):a.plot(g.rate,g[k],marker='o',label=name);a.set(xlabel='总缺失比例',ylabel=k);a.grid(alpha=.2)
    ax[0].legend();save('06_shape')
    fig,ax=plt.subplots(1,2,figsize=(11,4))
    for name in proto['models']:
        hist=json.loads((ROOT/f'results/history_{name}_17.json').read_text())
        ax[0].plot([r['epoch'] for r in hist],[r['loss'] for r in hist],label=NAMES[name]);ax[1].plot([r['epoch'] for r in hist],[r['J'] for r in hist],label=NAMES[name])
    ax[0].set(xlabel='epoch',ylabel='训练损失');ax[1].set(xlabel='epoch',ylabel='验证J');ax[0].legend(fontsize=8);save('07_training')
    fig,ax=plt.subplots(1,2,figsize=(11,4.6))
    for a,condition in zip(ax,['clean','missing']):
        mat=np.load(ROOT/f'results/attention_{condition}.npz')['cross'][0]
        im=a.imshow(mat,cmap='magma',vmin=0,vmax=max(float(np.quantile(mat,.995)),.001),aspect='equal');fig.colorbar(im,ax=a,fraction=.046)
        a.set_xticks([25,75,125],['文本键','音频键','视觉键']);a.set_yticks([25,75,125],['文本查询','音频查询','视觉查询'])
        for line in [49.5,99.5]:a.axvline(line,color='white',lw=.5);a.axhline(line,color='white',lw=.5)
        a.set_title('完整输入' if condition=='clean' else '中间连续缺失30%')
    fig.suptitle('cross结构种子17：内部注意力路由，不是因果重要性',y=1.03);save('08_attention_routing')
    boot=[];test=pd.read_csv(ROOT/'results/test_metrics.csv');testens=test[test.seed=='ensemble']
    for split in ['valid','test']:
        a=dict(np.load(ROOT/f'results/{split}_{chosen}_predictions.npz'));b=dict(np.load(ROOT/f'results/{split}_old_gated_predictions.npz'))
        for condition,idx in [('clean',0),('TAV_r50_random',next(i for i,p in enumerate(profiles) if p['name']=='TAV_r50_random' and p['repeat']==0))]:
            for r in bootstrap(a['out'][idx].mean(0),b['out'][idx].mean(0),a['c'],a['y'],a['ids']):boot.append(dict(split=split,condition=condition,**r))
    boot=pd.DataFrame(boot);boot.to_csv(ROOT/'results/paired_video_bootstrap.csv',index=False)
    special=pd.read_csv(ROOT/'results/附件3_预测审阅.csv');selected=table.loc[chosen];ref=table.loc['old_gated']
    failed=[k for k,vv in lock['reference_guardrails'][chosen].items() if not vv]
    params=pd.DataFrame([dict(model=n,trainable_parameters=json.loads((ROOT/f'results/completed_{n}_17.json').read_text())['parameters']) for n in proto['models']])
    contrasts=[]
    for a,b in [('self','pool'),('cross','self'),('cross','cross_clean')]:
        contrasts.append(dict(comparison=f'{a} − {b}',**{k:table.loc[a,k]-table.loc[b,k] for k in ['clean_f1','clean_mae','robust_f1','robust_mae']}))
    contrasts=pd.DataFrame(contrasts)
    contrasts.to_csv(ROOT/'results/ablation_differences.csv',index=False)
    testclean=testens[testens.name=='clean'].set_index('model');stress=testens[(testens.name=='TAV_r50_random')].groupby('model')[['macro_f1','mae']].mean()
    duration=effects[effects.subset=='TAV'].groupby('rate')[['macro_f1','mae']].mean()
    bestpos=effects[(effects.subset=='TAV')&(effects.rate==.5)].sort_values('macro_f1')
    result='通过旧版全部七项参考门槛；是否正式采用由用户决定。' if not failed else '未通过旧版全部七项参考门槛，不建议仅依据本轮结果替换旧版。未通过项：'+', '.join(failed)+'。'
    text=f'''# 问题二注意力实验报告

## 结论

已新建独立目录，完整阅读题目后实现模态内多头自注意力与跨模态注意力，完成4配置×3种子共12次训练、217次输入实现的完整验证、模型锁定后的测试复核和附件3全部30条独立预测。

本轮注意力候选为 **{NAMES[chosen]}（{chosen}）**，按预注册六场景验证J在增强self/cross中选择。{result}这是对当前轻量参数化和固定训练预算的结论，不能外推成所有注意力机制均有效或均无效。

完整验证集Macro-F1={selected.clean_f1:.4f}、MAE={selected.clean_mae:.4f}；旧版对应为{ref.clean_f1:.4f}、{ref.clean_mae:.4f}。112个主连续缺失场景的平均Macro-F1={selected.robust_f1:.4f}、MAE={selected.robust_mae:.4f}；旧版为{ref.robust_f1:.4f}、{ref.robust_mae:.4f}。**建成注意力方案不等于证实其优于旧方案。**

## 题意、输入和模型

问题二必须解决局部连续段缺失，而非只模拟整模态退出；同时预测负向/中性/正向三类和[-3,3]连续强度。真实强度恰为0才是中性。训练3395、验证728、既有测试727；统一aligned_50接口。逐槽缺失和padding区分处理；长度是有效词槽比例，不能当作秒。

附件3没有预计算text，所以所有输入统一经冻结DistilBERT；文本删除在编码前执行，不读取另一版本或原文本补齐。词表对4850条逐一核对一致，源视频划分不交叉。行为检查验证不可观测输入不影响输出、其梯度为0、全空回退有限、跨模态矩阵同模态块为0。缓存复编码最大数值差见 `verification.json`，属于半精度批量编码差异。

结构：三模态投影至64维，加位置/模态编码，分别做4头自注意力；cross进一步用150槽共享4头注意力，让每个查询只访问其他模态的有效键。每块有残差、LayerNorm、128维前馈层。可观测位置均值池化、三分支拼接质量统计，输出分类和3×tanh回归。参考跨模态思想来自[MulT原文](https://aclanthology.org/P19-1656/)，这里不是其完整复现。

完整公式、掩码与边界见[模型定义](模型定义.md)。代码和环境见[代码说明](代码说明.md)。新任务网络的pool对照没有注意力块，但冻结文本编码器仍含注意力，不能声称整个系统完全无注意力。

## 训练与选择规则

新配置均使用AdamW、学习率0.0003、weight decay0.001、dropout0.2、最大batch64、最多30轮、耐心6、种子17/29/43。类别加权交叉熵加2×SmoothL1；该回归系数沿用既有验证设置，不在本轮搜索。pool/self/cross共用由种子、epoch、样本ID确定的动态遮挡；cross_clean只用完整输入。去掉cross块也去掉其前馈层，参数量差异须计入解释。

J为完整输入的MAE+1−MacroF1权重0.5，加五个缺失验证场景各0.1。每个种子分别早停，概率与强度均值集成。旧版是历史参考，训练方式不同，不作为完全同条件结构消融。验证集已被既有实验反复使用，统计区间不校正选择偏差。

## 验证对照与缺失规律

{md(table.reset_index())}

新增可训练网络参数量（不含冻结文本编码器）：

{md(params)}

同条件新模型差值如下；F1为正、MAE为负表示前者更好。

{md(contrasts)}

本轮self相对pool的完整输入F1仅小幅变化，集成MAE降低；增加cross块未形成F1和MAE的同步改善。cross相对cross_clean在缺失平均指标上有小幅收益，但幅度不足以追平旧参考。三种子结果存在波动，因此这些小差异仅作为消融观察，不宣称统计显著或机制得到证明。注意力可访问更多剩余位置，并不保证学到有利于本数据泛化的联系。

![模型比较](results/figures/01_comparison.png)
![缺失增强和注意力消融](results/figures/02_missing_ablation.png)

robust均值先平均一个场景的随机重复，再对112场景等权平均。确定位置1次，随机位置3次；额外多短段/错开实验单列。三种子均值和标准差见 `seed_summary.csv`，不能把训练种子或遮挡重复当独立样本扩大样本量。

TAV缺失比例从10%到70%，四位置等权Macro-F1由{duration.loc[.1,'macro_f1']:.4f}变为{duration.loc[.7,'macro_f1']:.4f}，MAE由{duration.loc[.1,'mae']:.4f}变为{duration.loc[.7,'mae']:.4f}。50%时Macro-F1最低的位置为{bestpos.iloc[0].position}，最高为{bestpos.iloc[-1].position}；不能把槽位置差异外推为原视频秒级重要性。

![类型和长度](results/figures/03_modality_duration.png)
![位置](results/figures/04_position.png)

{md(shape)}

“独立起点”由各模态分别抽取位置，允许偶然重合，不是强制互不重叠；“分区多段”在三个分区分配删除槽，邻接的片段可能连在一起，短样本按代码规则回退为单段。

![形状](results/figures/06_shape.png)

这些均值只描述本数据、本随机缺失机制。音视频单独缺失是否略有改善，应视作需要实测的现象，不能预设模态越多越好；也不因一个小差值断言某形状普遍更难。

## 验证错误分析与注意力展示

{md(error)}

![验证错误](results/figures/05_validation_errors.png)

强情感分组的真实均值与预测均值用于核查幅度收缩；中性准确率单列，避免总体分数掩盖中性类别表现。10条最大绝对误差保存在 `representative_errors.csv`，全量真值、原文和预测在 `validation_prediction_review.csv`。原文仅用于错误审阅。组差异是误差诊断证据，不能据此证明跨模态因果冲突。

当前强负向组的平均真实强度约−2.01，预测约−1.06；强正向组约1.99，预测约0.94。两组分类准确率均接近89%，但MAE超过1：模型通常能判断方向，却低估情感幅度，这是本轮回归不足的直接表现。中性召回由旧版的{ref.neutral_recall:.3f}上升为{selected.neutral_recall:.3f}，也说明不同子指标的取舍不能仅由一个总分概括。

![训练轨迹](results/figures/07_training.png)
![注意力路由](results/figures/08_attention_routing.png)

注意力展示固定使用验证集第一条、cross种子17，比较完整输入与中间TAV30%缺失。若最终候选为self，此图明确属于cross对照。黑色同模态块和不可用连接由合法掩码决定；其余颜色是内部权重，不是情感贡献大小。本轮不进行问题三归因。

## 锁定后的既有测试复核

{md(testclean[['accuracy','macro_f1','mae','pearson','neutral_recall']].reset_index())}

TAV50%随机连续缺失三次均值：

{md(stress.reset_index())}

该测试集以前已用于其他方案，因此不是全新独立测试；本轮仅在验证选择锁定后计算，不再按这些分数调参。以下为源视频聚类500次配对bootstrap，差值=注意力−旧版；F1正向、MAE负向才是改善。这里只展示两个固定输入实现的区间，不保证覆盖所有故障。

{md(boot)}

本轮完整验证输入的MAE差值区间位于0以上，支持“当前注意力候选回归更差”的判断；完整输入F1区间包含0。既有测试上的F1点估计虽高于旧版，区间仍包含0，不能据此断言分类提升可靠。这些判断限定于当前数据划分和固定比较，仍受前述重复选择影响。

## 附件3与交付

`附件3_注意力模型预测.csv`包含全部30条，字段sample_id,polarity,intensity。分布为{special.polarity.value_counts().to_dict()}。这是新注意力候选的独立结果，既有采用文件未被覆盖。附件3无标签，不能给出专项准确率；分类/回归是两个头，非中性符号分歧有{int(special.directional_disagreement.sum())}条，中性预测也可能有非零连续强度。

代码、参数、协议、环境、全部训练记录、验证/测试矩阵、注意力可视化与结果审阅页均保留。通用预训练权重与大型可重建缓存不进入交付包；复现入口见代码说明。
'''
    (ROOT/'注意力实验报告.md').write_text(text,encoding='utf-8')
    cards=f'<div class="cards"><div>候选<b>{NAMES[chosen]}</b></div><div>验证Macro-F1<b>{selected.clean_f1:.3f}</b></div><div>验证MAE<b>{selected.clean_mae:.3f}</b></div><div>缺失平均F1<b>{selected.robust_f1:.3f}</b></div></div>'
    pictures=''.join(f'<figure><img loading="lazy" src="figures/{p.name}"><figcaption>{p.stem}</figcaption></figure>' for p in sorted(figdir.glob('*.png')))
    rows=''.join(f'<tr data-p="{r.polarity}"><td>{html.escape(r.sample_id)}</td><td>{r.polarity}</td><td>{r.intensity:.4f}</td><td>{r.intensity_seed_std:.4f}</td></tr>' for r in special.itertuples())
    page=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>问题二注意力实验</title><style>body{{font:16px/1.7 system-ui;background:#f4f6fa;color:#1c2d43;margin:0}}main{{max-width:1200px;padding:28px;margin:auto}}section,figure,.cards>div{{background:white;border-radius:12px;padding:20px;margin:18px 0}}.cards{{display:flex;gap:14px;flex-wrap:wrap}}.cards>div{{flex:1;min-width:170px}}b{{display:block;font-size:24px}}img{{max-width:100%}}table{{border-collapse:collapse;width:100%;font-size:14px}}th,td{{text-align:left;border-bottom:1px solid #ddd;padding:7px}}a{{color:#146582}}select{{font:inherit;padding:6px}}.note{{color:#945019}}</style><main><h1>问题二 · 注意力机制实验</h1><p class="note">{result}</p>{cards}<section><a href="../注意力实验报告.md">实验报告</a> · <a href="../模型定义.md">模型公式</a> · <a href="../代码说明.md">复现说明</a> · <a href="附件3_注意力模型预测.csv">30条预测</a><p>217次验证输入实现，三种子；注意力矩阵仅展示内部路由，不是因果解释。</p>{table.reset_index().round(4).to_html(index=False,border=0)}</section><section><h2>附件3候选预测</h2><select id="filter"><option value="">全部</option><option>negative</option><option>neutral</option><option>positive</option></select><table id="special"><thead><tr><th>样本</th><th>极性</th><th>强度</th><th>种子标准差</th></tr></thead><tbody>{rows}</tbody></table></section>{pictures}</main><script>document.getElementById('filter').onchange=function(){{document.querySelectorAll('#special tbody tr').forEach(r=>r.hidden=!!(this.value&&r.dataset.p!==this.value))}}</script></html>'''
    (ROOT/'results/attention_review.html').write_text(page,encoding='utf-8')
    save_json(ROOT/'results/summary.json',dict(selected=chosen,meets_previous_criteria=lock['meets_previous_replacement_criteria'],failed=failed,
        validation=selected.to_dict(),test_clean=testclean.loc[chosen,['accuracy','macro_f1','mae','pearson']].to_dict(),test_tav50=stress.loc[chosen].to_dict(),training_fits=12,validation_realizations=217,special_predictions=30))
    print('Report, figures and review generated.',flush=True)


if __name__=='__main__':main()
