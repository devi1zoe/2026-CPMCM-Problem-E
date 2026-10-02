"""Generate diagnostic and experiment reports from immutable recorded results."""
from common import *
from data import load_attachment2
from support import grouped_metrics
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import html

NAMES=['old_gated','text','gated','dual_gate','late_fusion']
METRICS=['accuracy','macro_f1','weighted_f1','mae','pearson','neutral_recall','f1_negative','f1_neutral','f1_positive','strong_negative_mae','strong_positive_mae']


def table(df):
    def fmt(v):return f'{v:.4f}' if isinstance(v,(float,np.floating)) else str(v)
    rows=df.fillna('').values.tolist()
    return '\n'.join(['| '+' | '.join(df.columns)+' |','|'+'---|'*len(df.columns)]+['| '+' | '.join(map(fmt,r))+' |' for r in rows])


def diagnostics():
    general=pd.read_csv(ROOT/'results/generalization_groups.csv')
    allrows=general[general.group=='all'].groupby(['model','split'])[['accuracy','macro_f1','mae','neutral_recall']].mean().reset_index()
    grad=pd.read_csv(ROOT/'results/task_gradients.csv')
    grad_summary=grad.groupby(['condition','block']).agg(batches=('seed','size'),mean_cosine=('cosine','mean'),negative_fraction=('negative','mean'),zero_gradient_batches=('zero_gradient','sum'),mean_norm_ratio=('norm_ratio','mean')).reset_index()
    mem=pd.DataFrame(json.loads((ROOT/'results/memory_probe_summary.json').read_text()))
    comp=pd.read_csv(ROOT/'results/complementarity.csv')
    compmetrics=['rescue_fraction','harm_fraction','text_mae','oracle_available_mae','oracle_accuracy']
    clean=comp[comp.name=='clean'][['group','n','text_wrong','text_wrong_rescued','rescue_fraction','text_correct','text_correct_harmed','harm_fraction','text_mae','oracle_available_mae']]
    primary=comp[comp.primary==True].groupby(['subset','name','group'])[compmetrics].mean().groupby(['subset','group']).mean().reset_index()
    primary.to_csv(ROOT/'results/complementarity_by_subset.csv',index=False)
    grad_summary.to_csv(ROOT/'results/gradient_summary.csv',index=False);allrows.to_csv(ROOT/'results/generalization_summary.csv',index=False)
    saturation=general[(general.split=='valid')&general.group.isin(['strong_negative','strong_positive'])].groupby(['model','group'])[['n','true_mean','pred_mean','saturation_fraction','pred_q10','pred_q50','pred_q90']].mean().reset_index()
    saturation.to_csv(ROOT/'results/saturation_summary.csv',index=False)
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for i,metric in enumerate(['macro_f1','mae']):
        t=allrows.pivot(index='model',columns='split',values=metric).reindex(['text','audio','vision','gated'])
        t.plot.bar(ax=axes[i],rot=0);axes[i].set_ylabel(metric);axes[i].set_title('Three-seed mean, eval mode')
    fig.tight_layout();fig.savefig(ROOT/'results/figures/generalization.png',dpi=150);plt.close(fig)
    memory=pd.read_csv(ROOT/'results/memory_probe_history.csv');fig,axes=plt.subplots(1,2,figsize=(11,4))
    for name,d in memory.groupby('model'):
        axes[0].plot(d.step,d.accuracy,label=name);axes[1].plot(d.step,d.mae,label=name)
    for a in axes:a.set_xlabel('Updates on 96 training samples');a.legend()
    axes[0].axhline(.95,color='gray',ls='--');axes[0].set_ylabel('Training accuracy');axes[1].axhline(.1,color='gray',ls='--');axes[1].set_ylabel('Training MAE')
    fig.tight_layout();fig.savefig(ROOT/'results/figures/memory_probe.png',dpi=150);plt.close(fig)
    body=f'''# 问题二瓶颈诊断报告

本轮先完成输入与行为检查，再进行诊断，最后才启动两种候选结构的正式训练。未发现标签、源视频划分、接口或人工掩码错误；具体断言见 `results/verification.json`。诊断只使用训练和验证材料，不使用专项测试集选择方法。

## 1. 训练表现与泛化差距

对v2文本、音频、视觉、门控模型各三个最佳检查点，在完整训练3395条和验证728条上统一使用评估模式。以下为三个单模型指标均值，不是集成指标，不能与集成F1直接混用。

{table(allrows)}

门控和文本模型在训练集上明显优于验证集，提供了泛化差距的直接观测。最佳检查点早并非判断过拟合的唯一依据；这里仍无法排除源视频分布差异等原因，也不据此宣称加大模型能解决问题。原检查点阶段和实际训练轮数见 `historical_stages.csv`。

![训练与验证](results/figures/generalization.png)

## 2. 96条小样本记忆检查

种子17，从训练集中同时具有音视频观测的样本按三类各抽32条。两分支使用同一96条样本；随机初始化现有单模态结构，关闭dropout，AdamW学习率0.001、weight decay=0、梯度裁剪5，最多300次更新，每10次检查一次，达到准确率≥0.95且MAE≤0.1即停止。分类权重仍按完整训练集频数计算。诊断模型不保存为正式候选，也不参与选择。

{table(mem[['model','steps','passed','accuracy','mae','macro_f1']])}

两个分支均能记住这一组样本，因此当前低验证性能不能简单归咎于网络完全没有拟合能力。记忆小样本并不证明音视频能跨视频泛化，也不能证明这些特征存在足够情感信号。

![小样本记忆](results/figures/memory_probe.png)

## 3. 互补性与融合损害

使用已有三个单模态集成和门控集成的验证预测。只将实际可观测专家计入救错和逐样本最小误差；每一行都使用对应缺失场景的观测掩码。完整输入如下：

{table(clean)}

`rescue_fraction` 是文本出错时、音频或视觉至少一个预测正确的比例；`harm_fraction` 是文本正确但门控预测错误的比例。`oracle_available_mae` 使用真值逐样本挑选误差最小的可用专家，全部缺失时回退训练均值。这些oracle指标是乐观参照，不是可部署模型；救错可能包含随机命中和类别偏好，不能凭该比例证明存在可学习、稳定的互补性。

以下缺失类型汇总先平均随机重复，再平均各比例与位置。完整逐场景和中性、强正负向分组见 `complementarity.csv`。

{table(primary[primary.group=='all'])}

## 4. 分类与回归梯度

固定512条训练样本，8批×64；三个门控检查点，各自检查完整输入和固定种子17/轮次1动态遮挡。关闭dropout，参数不更新；比较分类交叉熵与2×SmoothL1对共享分支、共享门控的梯度。范数比为分类梯度范数除以加权回归梯度范数；零梯度单独标记，余弦不作解释，分母为零时不定义范数比。

{table(grad_summary)}

在这些检查点和抽样批次上，梯度大多同向，没有证据支持“普遍的梯度冲突是主要瓶颈”。这不是证明任意训练时刻都没有冲突，也不排除分类与回归需要不同模态权重。本轮不使用PCGrad或其他梯度修正。[相关研究](https://arxiv.org/abs/2001.06782)

## 5. 强情感收缩与输出饱和

强负向定义为真实强度≤−1.5，强正向为真实强度≥1.5；弱情感为其余非零强度，中性严格为0。饱和诊断阈值固定为预测绝对值≥2.9，属于分析指标而非决策阈值。下表为三个单模型的组均值和分位数均值。

{table(saturation)}

平均预测向中间收缩不等于输出已经饱和。缺乏输出饱和证据时，不把误差简单归因于3×tanh，也不临时换输出函数。本轮保留连续回归目标和真实中性定义。

## 6. 对后续实验的约束

诊断支持继续进行已约定的两个有限对照：任务独立门控和冻结专家后融合。它没有证明其中任何一种必然改善，也不作为追加模块或扩大搜索的理由。全部行为检查通过，诊断完成标记绑定协议哈希；正式模型只按验证集与原七项门槛选择。
'''
    (ROOT/'问题二瓶颈诊断报告.md').write_text(body)
    return allrows,grad_summary,mem,clean


def main():
    diagnostic_tables=diagnostics();lock=json.loads((ROOT/'selection_lock.json').read_text());comparison=pd.read_csv(ROOT/'results/selection_comparison.csv')
    frames={n:pd.read_csv(ROOT/f'results/valid_{n}_metrics.csv') for n in NAMES};clean=[];seedrows=[];groups=[];errorrows=[];effects=[];paired=[]
    raw=load_attachment2()['valid'];text_lookup={str(i):str(t) for i,t in zip(raw['id'],raw['raw_text'])};predictions={}
    for name,df in frames.items():
        clean.append(dict(model=name,**df[(df.seed=='ensemble')&(df.name=='clean')][METRICS].iloc[0].to_dict()))
        for condition in ['clean','continuous_average']:
            q=df[df.name=='clean'] if condition=='clean' else df[df.primary==True].groupby(['seed','name'])[METRICS].mean().groupby('seed').mean().reset_index()
            for metric in METRICS:
                singles=q[q.seed.isin(['17','29','43'])][metric]
                seedrows.append(dict(model=name,condition=condition,metric=metric,seed_mean=singles.mean(),seed_std=singles.std(ddof=1),ensemble=q[q.seed=='ensemble'][metric].iloc[0]))
        d=dict(np.load(ROOT/f'results/valid_{name}_predictions.npz'));predictions[name]=d;p=d['out'][0].mean(0)
        groups.extend(dict(model=name,**r) for r in grouped_metrics(d['c'],d['y'],p))
        if name in ['dual_gate','late_fusion']:
            for i,sid in enumerate(d['ids']):
                errorrows.append(dict(model=name,sample_id=sid,text=text_lookup[sid],true_class=int(d['c'][i]),pred_class=int(p[i,:3].argmax()),true_intensity=float(d['y'][i]),pred_intensity=float(p[i,3]),absolute_error=float(abs(p[i,3]-d['y'][i]))))
        e=df[(df.seed=='ensemble')&(df.name!='clean')]
        per=e.groupby(['name','subset','rate','position','pattern','sync','primary'])[['macro_f1','mae']].mean().reset_index()
        for factor in ['subset','rate','position']:
            for value,g in per[per.primary==True].groupby(factor):effects.append(dict(model=name,factor=factor,value=str(value),macro_f1=g.macro_f1.mean(),mae=g.mae.mean()))
        reference=e[(e.position=='random')&(e.pattern=='single')&(e.sync==True)]
        for label,target in [('multi',e[e.pattern=='multi']),('staggered',e[(e.position=='random')&(e.sync==False)])]:
            joined=target.merge(reference,on=['subset','rate','repeat'],suffixes=('_target','_ref'));assert len(joined)==24
            paired.append(dict(model=name,comparison=label,pairs=len(joined),delta_f1=float((joined.macro_f1_target-joined.macro_f1_ref).mean()),delta_mae=float((joined.mae_target-joined.mae_ref).mean())))
    clean=pd.DataFrame(clean);groups=pd.DataFrame(groups);errors=pd.DataFrame(errorrows);effects=pd.DataFrame(effects);paired=pd.DataFrame(paired)
    for name,df in [('clean_comparison',clean),('seed_summary',pd.DataFrame(seedrows)),('intensity_groups',groups),('validation_prediction_review',errors),('missing_effects',effects),('paired_shape_sync',paired)]:df.to_csv(ROOT/f'results/{name}.csv',index=False)
    errors.sort_values('absolute_error',ascending=False).groupby('model').head(12).to_csv(ROOT/'results/representative_errors.csv',index=False)
    stages=[];trainrows=[]
    for name in ['dual_gate','late_fusion']:
        for seed in [17,29,43]:
            ck=__import__('torch').load(ROOT/f'checkpoints/{name}_{seed}.pt',map_location='cpu',weights_only=True);done=json.loads((ROOT/f'results/completed_{name}_{seed}.json').read_text())
            stages.append(dict(model=name,seed=seed,selected_stage=ck['stage'],selected_epoch=ck['epoch'],epochs_run=done['epochs'],trainable_parameters=done['trainable_parameters'],total_parameters=done['total_parameters']))
            for r in json.loads((ROOT/f'results/history_{name}_{seed}.json').read_text()):trainrows.append(dict(model=name,seed=seed,epoch=r['epoch'],train_f1=r['fixed_train']['macro_f1'],valid_f1=r['clean']['macro_f1'],train_mae=r['fixed_train']['mae'],valid_mae=r['clean']['mae'],J=r['score']))
    pd.DataFrame(stages).to_csv(ROOT/'results/checkpoint_stages.csv',index=False);pd.DataFrame(trainrows).to_csv(ROOT/'results/training_curves.csv',index=False)
    expert_cost=[]
    for name in ['text','audio','vision']:
        for seed in [17,29,43]:
            d=json.loads((PROJECT/f'step2_v2/results/completed_{name}_{seed}.json').read_text());expert_cost.append(dict(model=name,seed=seed,epochs=d['epochs'],seconds=d['seconds'],reused=True))
    pd.DataFrame(expert_cost).to_csv(ROOT/'results/expert_pretraining_cost.csv',index=False)
    gate_files=list((ROOT/'results').glob('valid_task_gate_weights_*.csv'));gates=pd.concat([pd.read_csv(p) for p in gate_files],ignore_index=True).drop_duplicates(['model','seed','name','repeat'])
    gate_cols=[f'{t}_{m}' for t in ['classification','regression'] for m in 'TAV']
    gate_summary=gates[gates.name=='clean'].groupby('model')[gate_cols].mean().reset_index();gate_summary.to_csv(ROOT/'results/clean_gate_summary.csv',index=False)
    boot=json.loads((ROOT/'results/paired_cluster_bootstrap.json').read_text());intervals=[]
    for ref,models in boot['results'].items():
        for name,conditions in models.items():
            for condition,met in conditions.items():
                for metric,d in met.items():intervals.append(dict(reference=ref,model=name,condition=condition,metric=metric,delta=d['delta'],lower=d['ci95'][0],upper=d['ci95'][1]))
    intervals=pd.DataFrame(intervals);intervals.to_csv(ROOT/'results/bootstrap_intervals.csv',index=False)
    fig,axes=plt.subplots(2,3,figsize=(13,8))
    for j,name in enumerate(['gated','dual_gate','late_fusion']):
        d=predictions[name];p=d['out'][0].mean(0);cm=confusion_matrix(d['c'],p[:,:3].argmax(-1),labels=[0,1,2]);axes[0,j].imshow(cm,cmap='Blues')
        for a in range(3):
            for b in range(3):axes[0,j].text(b,a,str(cm[a,b]),ha='center',va='center',color='white' if cm[a,b]>cm.max()/2 else 'black')
        axes[0,j].set(title=name,xlabel='Predicted class',ylabel='True class',xticks=[0,1,2],yticks=[0,1,2])
        axes[1,j].scatter(d['y'],p[:,3],s=7,alpha=.35);axes[1,j].plot([-3,3],[-3,3],'k--',lw=1)
        axes[1,j].set(xlabel='True intensity',ylabel='Predicted intensity',xlim=(-3,3),ylim=(-3,3))
    fig.tight_layout();fig.savefig(ROOT/'results/figures/confusion_scatter.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for name in ['old_gated','gated','dual_gate','late_fusion']:
        d=effects[(effects.model==name)&(effects.factor=='rate')];x=d.value.astype(float)
        axes[0].plot(x,d.macro_f1,'o-',label=name);axes[1].plot(x,d.mae,'o-',label=name)
    for a in axes:a.set_xlabel('Missing fraction of valid slots');a.legend(fontsize=8)
    axes[0].set_ylabel('Macro-F1');axes[1].set_ylabel('MAE');fig.tight_layout();fig.savefig(ROOT/'results/figures/missing_rate.png',dpi=150);plt.close(fig)
    conclusion=f'候选 {lock["selected"]} 通过原七项门槛，锁定为本轮最终模型。' if lock['accepted_improvement'] else '两种候选均未通过全部原验收条件，保留旧版 old_gated；本轮不追加模型或调参。'
    failed='\n'.join('- '+r['model']+'：'+('全部通过' if r['eligible'] else '未通过 '+', '.join(k for k,v in r['checks'].items() if not v)) for r in lock['rows'] if r['model'] in ['dual_gate','late_fusion'])
    if lock['accepted_improvement']:
        t=pd.read_csv(ROOT/f'results/test_{lock["selected"]}_metrics.csv');t=t[(t.seed=='ensemble')&(t.name=='clean')]
        final_stage='仅在选择锁定后进行了最终测试复核和附件3推理。测试集过去已被评价，不能称为完全未见测试；未据测试结果继续调参。\n\n'+table(t[METRICS])
    else:final_stage='没有候选通过，因此按计划没有运行新的测试集复核，没有生成重复附件3预测，没有制作新联合包。沿用 v2 中的旧版最终模型及已验证交付。'
    body=f'''# 问题二任务独立融合实验报告

{conclusion}

## 1. 范围与诊断结论

训练3395、验证728，沿用aligned_50输入、冻结DistilBERT、动态连续删除和真实强度0为中性。只修改任务融合方式，没有恢复v3，没有新增序数、蒸馏、表示分解或调参搜索。

诊断发现训练与验证存在差距；音视频都能记忆96条训练样本；抽查梯度大多同向，因此没有把任务冲突当成既定原因。真值oracle显示部分救错机会，但其数值不可部署，不能当成模型收益。详细证据见[瓶颈诊断报告](问题二瓶颈诊断报告.md)。

## 2. 两种候选

**A / dual_gate**：只增加一套与原门控复制初始化的回归门控；分类、回归分别加权三模态特征，再进入同一个共享MLP和各自预测头。共有参数初始化与基线一致，评估模式下初始预测完全一致。主干和MLP仍共享，因此不声称完全消除了任务冲突。训练时两个流各自经过dropout，不声称与原基线随机轨迹逐位相同。[任务独立门控研究](https://research.google/pubs/modeling-task-relationships-in-multi-task-learning-with-multi-gate-mixture-of-experts/)

**B / late_fusion**：冻结同种子文本、音频、视觉专家，每个任务仅训练9→16→3、GELU门控，输入仅为三模态质量特征。输出层权重为0，偏置log(0.8,0.1,0.1)。分类混合概率，回归混合连续预测，不重复施加tanh。只有422个可训练参数，但部署还包含三个专家；9次既有专家训练的轮数和时间单列在 `expert_pretraining_cost.csv`，不声称总算力更低。

B缓存绑定全部实际输入数组、观测掩码、专家权重哈希、编码器版本和标准化统计。专家保持eval且参数冻结；归一化排除缺席模态，全缺失回退训练先验和均值。回归是可用专家预测的凸组合，无法超出其数值范围。专家曾在同一训练集拟合，门控并非OOF堆叠，报告不掩盖这一泛化限制。

## 3. 训练与选择

两个候选各种子17/29/43，共6次正式拟合。继承v2遮挡、AdamW学习率0.001、batch64、weight decay0.001、梯度裁剪5、20/15轮上限、耐心5、两阶段重置优化器和余弦调度。损失为训练类别加权分类损失＋2×SmoothL1。每轮额外评价固定512条训练子集，但检查点仅按六场景验证J选择。以下为集成结果：

{table(comparison[['model','J','parameters','clean_f1','clean_mae','robust_f1','robust_mae','neutral_recall','strong_negative_mae','strong_positive_mae']])}

完整矩阵217次输入实现，其中112个主场景先平均随机重复再等权汇总。六场景早停J与完整矩阵缺失均值是不同统计量，不混淆。

原七项门槛相对历史old_gated保持不变：完整F1下降≤0.005、MAE增加≤0.01；缺失平均F1至少增加0.01且MAE不升；中性召回下降≤0.01；强正、强负MAE分别增加≤0.02。全部通过才按J选，差距<0.002优先部署总参数少者。

{failed}

`seed_summary.csv` 分开给出三种子指标均值、样本标准差和集成，避免把集成收益归因于结构。完整分类与回归指标如下：

{table(clean)}

## 4. 中性、强情感与错误案例

{table(groups[(groups.model.isin(['gated','dual_gate','late_fusion']))&groups.group.isin(['neutral','strong_negative','strong_positive'])][['model','group','n','accuracy','mae','true_mean','pred_mean','saturation_fraction']])}

![混淆矩阵与强度](results/figures/confusion_scatter.png)

类别0/1/2为负向/中性/正向。每个新候选的12条最大绝对误差案例见 `representative_errors.csv`；完整样本审阅见 `validation_prediction_review.csv`。原文只用于审阅，不用于补齐输入。组均值收缩或单个案例不能直接确定误差因果来源。

## 5. 缺失与门控

![缺失率](results/figures/missing_rate.png)

类型、位置、比例见 `missing_effects.csv`。多短段和错开实验按相同模态组合、比例、重复配对，差值为目标场景减单段同步场景：

{table(paired)}

完整输入下两套任务门控的三种子、全样本平均如下；全缺失样本的零权重也计入均值。它们只描述模型内部路由，不能当作问题三的因果解释。

{table(gate_summary)}

## 6. 配对统计与结论边界

按{boot['source_video_groups']}个源视频进行1000次配对bootstrap。F1差为正更好，MAE差为负更好。以下相对同条件gated；相对old_gated也已保存。区间跨0不宣称稳定优势。验证集被重复使用，区间未校正模型选择，不能视作独立泛化保证。

{table(intervals[intervals.reference=='gated'])}

{final_stage}

本轮的有效结论是对这两种明确参数化、固定训练预算下的候选进行比较，不能外推为所有任务独立融合或所有后融合方法均有效／无效。冻结文本编码器、仅使用质量特征路由、专家回归范围和数据规模都构成适用边界。

## 7. 交付与复现

交付本目录的模型、配置、诊断、6个候选检查点、训练记录、比较表、错误案例、审阅页和行为核验。外部仍为text_bert/audio/vision输入，标准预测字段sample_id,polarity,intensity；新增两套门控仅供内部审计。旧版、v2、问题三及根Python环境保持不变。完整复现入口见[代码说明](代码说明.md)。
'''
    (ROOT/'问题二任务独立融合实验报告.md').write_text(body)
    blocks=[('集成比较',comparison),('训练／验证诊断',diagnostic_tables[0]),('96条记忆检查',diagnostic_tables[2]),('梯度诊断',diagnostic_tables[1]),('完整输入互补性（oracle不可部署）',diagnostic_tables[3]),('任务门控均值',gate_summary),('相对同条件基线的区间',intervals[intervals.reference=='gated'])]
    sections=''.join('<h2>'+html.escape(title)+'</h2>'+df.to_html(index=False,float_format=lambda x:f'{x:.4f}') for title,df in blocks)
    view='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>问题二任务独立融合</title><style>body{font:16px/1.65 sans-serif;max-width:1450px;margin:35px auto;padding:0 24px;background:#f5f7fb;color:#172338}h1,h2{color:#164b76}table{border-collapse:collapse;background:white;font-size:13px;display:block;overflow:auto}td,th{padding:8px;border:1px solid #d8e0ec;text-align:right}img{max-width:100%}.notice{padding:18px;background:#e5eef9;border-left:5px solid #3272ad}a{color:#1766a3}</style><h1>问题二：诊断与任务独立融合</h1><p class="notice">'''+html.escape(conclusion)+'''</p><p>6次正式训练；每候选217个验证输入；诊断和真值oracle不作为部署性能。测试及附件3仅在候选通过后运行。</p>'''+sections+'''<h2>图表</h2><img src="figures/generalization.png" alt="训练验证差距"><img src="figures/memory_probe.png" alt="小样本记忆"><img src="figures/confusion_scatter.png" alt="混淆矩阵与强度"><img src="figures/missing_rate.png" alt="缺失率"><h2>材料</h2><ul><li><a href="../问题二瓶颈诊断报告.md">诊断报告</a></li><li><a href="../问题二任务独立融合实验报告.md">实验报告</a></li><li><a href="../代码说明.md">复现说明</a></li><li><a href="seed_summary.csv">种子均值、标准差和集成</a></li><li><a href="representative_errors.csv">错误案例</a></li></ul></html>'''
    (ROOT/'results/refine_review.html').write_text(view)
    print(conclusion,flush=True)


if __name__=='__main__':main()
