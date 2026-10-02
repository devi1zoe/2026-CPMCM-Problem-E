"""所有结论从已落盘实验结果生成；不训练、不改变选择结果。"""
from common import *
import html
import re
import zipfile
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from sklearn.metrics import confusion_matrix

LABELS={'majority_mean':'多数类/均值','text_only':'仅文本','audio_only':'仅音频','vision_only':'仅视觉',
        'pooled_clean':'统计池化','fixed_clean':'固定融合','fixed_aug':'固定融合+增强','mask_aug':'掩码+增强','gated_aug':'门控+掩码+增强'}
ORDER=list(LABELS)
MAIN=['fixed_clean','fixed_aug','mask_aug','gated_aug']
FIG=ROOT/'results/figures'


def md(df):
    def f(x):
        if isinstance(x,(float,np.floating)):return '—' if pd.isna(x) else f'{x:.4f}'
        return str(x).replace('|','/').replace('$','\\$').replace('\n','<br>')
    return '| '+' | '.join(map(str,df.columns))+' |\n| '+' | '.join(['---']*len(df.columns))+' |\n'+'\n'.join('| '+' | '.join(f(x) for x in row)+' |' for row in df.itertuples(index=False,name=None))


def savefig(name):
    plt.tight_layout();plt.savefig(FIG/(name+'.png'),dpi=170,bbox_inches='tight');plt.close()


def main():
    FIG.mkdir(exist_ok=True)
    font=Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    if font.exists():
        font_manager.fontManager.addfont(str(font));plt.rcParams['font.family']=font_manager.FontProperties(fname=str(font)).get_name()
    plt.rcParams.update({'axes.unicode_minus':False,'text.parse_math':False,'font.size':10})
    lock=json.loads((ROOT/'selection_lock.json').read_text());selected=lock['selected']
    test=pd.read_csv(ROOT/'results/test_metrics.csv');val=pd.read_csv(ROOT/'results/validation_metrics.csv')
    ens=test[test.seed=='ensemble'];clean=ens[ens.name=='clean'].set_index('model').reindex(ORDER)
    clean_table=clean[['accuracy','macro_f1','mae','pearson','neutral_recall']].copy()
    clean_table.insert(0,'模型',[LABELS[x] for x in clean_table.index]);clean_table=clean_table.reset_index(drop=True)
    clean_table.to_csv(ROOT/'results/clean_comparison.csv',index=False,encoding='utf-8-sig')
    vc=val[(val.seed=='ensemble')&(val.profile=='clean')].set_index('model')
    random=ens[(ens.position=='random')&(ens.pattern=='single')&(ens['sync']==True)]
    aggregate=random.groupby('model')[['accuracy','macro_f1','mae','pearson']].mean().reindex(ORDER)
    aggregate.to_csv(ROOT/'results/robustness_average.csv',encoding='utf-8-sig')
    summaries=[]
    for model in MAIN:
        for scenario in ['clean','TAV_r50_random']:
            a=test[(test.model==model)&(test.name==scenario)&(test.seed!='ensemble')]
            by_seed=a.groupby('seed')[['accuracy','macro_f1','mae','pearson']].mean()
            r=dict(model=model,scenario=scenario)
            for k in by_seed:
                r[k+'_mean']=by_seed[k].mean();r[k+'_seed_std']=by_seed[k].std(ddof=1)
            summaries.append(r)
    ab=pd.DataFrame(summaries);ab.to_csv(ROOT/'results/ablation_seed_summary.csv',index=False)
    x=np.arange(len(ORDER));w=.38
    fig,ax=plt.subplots(1,2,figsize=(13,4.5))
    ax[0].bar(x-w/2,clean.accuracy,w,label='Accuracy');ax[0].bar(x+w/2,clean.macro_f1,w,label='Macro-F1')
    ax[0].set_ylim(0,1);ax[0].legend();ax[0].set_title('完整输入：三分类表现（测试集）')
    ax[1].bar(x,clean.mae,color='#de8f44');ax[1].set_title('完整输入：强度MAE（越低越好）')
    for a in ax:a.set_xticks(x,[LABELS[k] for k in ORDER],rotation=35,ha='right');a.grid(axis='y',alpha=.2)
    savefig('01_clean_comparison')
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for m in MAIN:
        t=random[(random.model==m)&(random.subset=='TAV')].groupby('rate')[['macro_f1','mae']].agg(['mean','std'])
        for ax,k in zip(axes,['macro_f1','mae']):
            ax.errorbar(t.index,t[k]['mean'],yerr=t[k]['std'],marker='o',capsize=3,label=LABELS[m])
            ax.set_xlabel('三个模态同步缺失比例');ax.set_ylabel(k);ax.grid(alpha=.2)
    axes[0].legend(fontsize=8);axes[0].set_title('随机位置：3次遮挡均值±标准差');axes[1].set_title('每点均使用三个训练种子的集成')
    savefig('02_ablation_missing_rate')
    chosen=random[random.model==selected]
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for subset in ['T','A','V','TA','TV','AV','TAV']:
        t=chosen[chosen.subset==subset].groupby('rate')[['macro_f1','mae']].mean()
        for ax,k in zip(axes,['macro_f1','mae']):ax.plot(t.index,t[k],marker='o',label=subset);ax.set_xlabel('缺失比例');ax.set_ylabel(k);ax.grid(alpha=.2)
    axes[0].legend(ncol=4);axes[0].set_title('最终模型：缺失模态与长度');axes[1].set_title('T文本 / A音频 / V视觉')
    savefig('03_modality_and_duration')
    position=ens[(ens.model==selected)&(ens.subset=='TAV')&(ens.pattern=='single')&(ens['sync']==True)].groupby(['rate','position'])[['macro_f1','mae']].mean()
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for ax,k in zip(axes,['macro_f1','mae']):
        t=position[k].unstack().reindex(columns=['start','middle','end','random'])
        im=ax.imshow(t,cmap='YlGnBu' if k=='macro_f1' else 'YlOrRd',aspect='auto')
        ax.set_xticks(range(4),['开头','中间','末尾','随机']);ax.set_yticks(range(4),['10%','30%','50%','70%']);ax.set_title(k)
        for i in range(4):
            for j in range(4):ax.text(j,i,f'{t.iloc[i,j]:.3f}',ha='center',va='center',color='white' if im.norm(t.iloc[i,j])>.55 else 'black')
        fig.colorbar(im,ax=ax)
    savefig('04_position')
    shape=ens[(ens.model==selected)&(ens.subset=='TAV')&(ens.position=='random')&(ens.rate.isin([.3,.5]))].copy()
    shape['condition']=np.where(shape.pattern=='multi','多短段',np.where(shape['sync'],'单长段同步','单长段错开'))
    shape_table=shape.groupby(['rate','condition'])[['macro_f1','mae']].mean().reset_index()
    shape_table.to_csv(ROOT/'results/shape_sync_comparison.csv',index=False,encoding='utf-8-sig')
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for cond,g in shape_table.groupby('condition'):
        for ax,k in zip(axes,['macro_f1','mae']):ax.plot(g.rate,g[k],marker='o',label=cond);ax.set_ylabel(k);ax.set_xlabel('相同总缺失比例');ax.grid(alpha=.2)
    axes[0].legend();savefig('05_shape_and_sync')
    data=dict(np.load(ROOT/'results/predictions_clean.npz'));truth,pred=data['c'],data['prob'].argmax(1)
    cm=confusion_matrix(truth,pred,labels=[0,1,2])
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    im=axes[0].imshow(cm,cmap='Blues');axes[0].set_xticks(range(3),['负向','中性','正向']);axes[0].set_yticks(range(3),['负向','中性','正向'])
    axes[0].set_xlabel('预测');axes[0].set_ylabel('真实');axes[0].set_title('最终模型完整输入混淆矩阵')
    for i in range(3):
        for j in range(3):axes[0].text(j,i,str(cm[i,j]),ha='center',va='center',color='white' if cm[i,j]>.55*cm.max() else 'black')
    axes[1].scatter(data['y'],data['pred'],s=10,alpha=.35);axes[1].plot([-3,3],[-3,3],'k--',lw=1)
    axes[1].set(xlabel='真实强度',ylabel='预测强度',xlim=(-3,3),ylim=(-3,3),title='强度回归：完整输入')
    savefig('06_errors')
    records=pd.read_csv(ROOT/'results/test_prediction_review.csv')
    records['contrast_marker']=records.raw_text.str.contains(r'\b(?:but|however|although|though|yet)\b',case=False,regex=True)
    records['intensity_bin']=pd.cut(records.true_intensity,[-3.01,-1.5,-.00001,.00001,1.5,3.01],labels=['强负向','弱负向','中性','弱正向','强正向'])
    error=[]
    for scenario,g in records.groupby('scenario'):
        for category,col in [('强度','intensity_bin'),('转折词线索','contrast_marker')]:
            for group,h in g.groupby(col,observed=True):
                error.append(dict(scenario=scenario,category=category,group=str(group),n=len(h),accuracy=float((h.true_class==h.predicted_class).mean()),mae=float(h.abs_error.mean())))
    error=pd.DataFrame(error);error.to_csv(ROOT/'results/error_groups.csv',index=False,encoding='utf-8-sig')
    worst=records[records.scenario=='clean'].nlargest(8,'abs_error')
    worst.to_csv(ROOT/'results/representative_errors.csv',index=False,encoding='utf-8-sig')
    valid_data=dict(np.load(ROOT/'results/predictions_valid_clean.npz'))
    vcm=confusion_matrix(valid_data['c'],valid_data['prob'].argmax(1),labels=[0,1,2])
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    axes[0].imshow(vcm,cmap='Blues');axes[0].set_xticks(range(3),['负向','中性','正向']);axes[0].set_yticks(range(3),['负向','中性','正向'])
    axes[0].set(xlabel='预测',ylabel='真实',title='验证集：完整输入混淆矩阵')
    for i in range(3):
        for j in range(3):axes[0].text(j,i,str(vcm[i,j]),ha='center',va='center',color='white' if vcm[i,j]>.55*vcm.max() else 'black')
    axes[1].scatter(valid_data['y'],valid_data['pred'],s=10,alpha=.35);axes[1].plot([-3,3],[-3,3],'k--',lw=1)
    axes[1].set(xlabel='真实强度',ylabel='预测强度',xlim=(-3,3),ylim=(-3,3),title='验证集：强度误差')
    savefig('08_validation_errors')
    valid_records=pd.read_csv(ROOT/'results/validation_prediction_review.csv')
    valid_errors=[]
    for scenario,g in valid_records.groupby('scenario'):
        g=g.copy();g['intensity_bin']=pd.cut(g.true_intensity,[-3.01,-1.5,-.00001,.00001,1.5,3.01],labels=['强负向','弱负向','中性','弱正向','强正向'])
        for name,h in g.groupby('intensity_bin',observed=True):
                valid_errors.append(dict(scenario=scenario,group=str(name),n=len(h),accuracy=float((h.true_class==h.predicted_class).mean()),mae=float(h.abs_error.mean()),
                    mean_true=float(h.true_intensity.mean()),mean_pred=float(h.predicted_intensity.mean()),signed_bias=float((h.predicted_intensity-h.true_intensity).mean())))
    valid_errors=pd.DataFrame(valid_errors);valid_errors.to_csv(ROOT/'results/validation_error_groups.csv',index=False,encoding='utf-8-sig')
    valid_records[valid_records.scenario=='clean'].nlargest(8,'abs_error').to_csv(ROOT/'results/validation_representative_errors.csv',index=False,encoding='utf-8-sig')
    case_templates=[
        ('评分表达','out of five stars, I would give it a five','明确的满分评分仍被判错，提示数字评分表达是当前模型的失败类型；不能凭输出断言具体哪个模态导致错误。'),
        ('比较结构','more convenient than stupid boots','负面词修饰被比较的靴子，整体表达偏正向；预测错误与比较结构处理不足相符，但这只是文本诊断线索。'),
        ('强度低估','it should be rated junk','核对极性是否正确与强度幅度是否充分，是两个不同的问题；此例用于审查强负向幅度低估。')]
    cases=[]
    for kind,pattern,note in case_templates:
        match=valid_records[(valid_records.scenario=='clean')&valid_records.raw_text.str.contains(pattern,case=False,regex=False)]
        if len(match):
            r=match.iloc[0]
            cases.append(dict(类型=kind,样本=r.sample_id,文本=r.raw_text,真实极性=['负向','中性','正向'][int(r.true_class)],
                预测极性=['负向','中性','正向'][int(r.predicted_class)],真实强度=r.true_intensity,预测强度=r.predicted_intensity,诊断边界=note))
    case_table=pd.DataFrame(cases)
    case_table.to_csv(ROOT/'results/validation_case_studies.csv',index=False,encoding='utf-8-sig')
    vg=valid_errors[valid_errors.scenario=='clean'].set_index('group')
    hardest_valid=vg.mae.idxmax()
    neutral_to=int(np.argmax(np.where(np.arange(3)==1,-1,vcm[1])))
    compression='强负向平均预测向上偏移、强正向向下偏移，表现出强度幅度压缩。' if vg.loc['强负向','signed_bias']>0 and vg.loc['强正向','signed_bias']<0 else '不同强度分组的偏差方向并不一致，应分别检查。'
    # 绘制实际训练日志，不把某个最好epoch事后移动到别的测试最优点。
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for m in MAIN:
        h=json.loads((ROOT/f'results/history_{m}_17.json').read_text())
        axes[0].plot([r['epoch'] for r in h],[r['loss'] for r in h],label=LABELS[m])
        axes[1].plot([r['epoch'] for r in h],[r['score'] for r in h],label=LABELS[m])
    axes[0].set(xlabel='epoch',ylabel='训练损失',title='种子17：训练曲线');axes[1].set(xlabel='epoch',ylabel='验证选择目标（越低越好）',title='同一鲁棒性选择目标')
    axes[0].legend(fontsize=8);savefig('07_training')
    special=pd.read_csv(ROOT/'results/附件3_预测审阅.csv')
    special_audit=json.loads((ROOT/'results/special_inference_audit.json').read_text())
    boot=json.loads((ROOT/'results/paired_cluster_bootstrap.json').read_text())
    a=clean.loc[selected];stress=random[(random.model==selected)&(random.subset=='TAV')&(random.rate==.5)].mean(numeric_only=True)
    ref=random[(random.model=='fixed_clean')&(random.subset=='TAV')&(random.rate==.5)].mean(numeric_only=True)
    mainval=pd.DataFrame([dict(模型=LABELS[k],验证选择目标=v) for k,v in lock['scores'].items()]).sort_values('验证选择目标')
    valtable=vc[['accuracy','macro_f1','mae','pearson']].copy();valtable.insert(0,'模型',[LABELS[k] for k in vc.index]);valtable=valtable.reset_index(drop=True)
    modality=chosen[chosen.rate==.5].groupby('subset')[['macro_f1','mae']].mean().reindex(['T','A','V','TA','TV','AV','TAV'])
    modality.to_csv(ROOT/'results/modality_r50.csv',encoding='utf-8-sig')
    av_gain=[m for m in ['A','V','AV'] if modality.loc[m,'macro_f1']>a.macro_f1]
    av_note=('其中 '+ '、'.join(av_gain)+' 的50%删除均值Macro-F1略高于完整输入。这不证明删除信息普遍有益；小幅波动、噪声分支和融合变化都可能有关，不能把“模态越多越好”作为未验证前提。') if av_gain else '这些比较用于检验实际互补性，不预设“模态越多越好”。'
    # 不在报告里把期望规律当作实验发现。
    hardest=modality.macro_f1.idxmin();easiest=modality.macro_f1.idxmax()
    pos50=position.loc[.5];worstpos=pos50.macro_f1.idxmin();bestpos=pos50.macro_f1.idxmax()
    counts=special.polarity.value_counts().to_dict()
    ci=boot['results']['TAV_r50_random']
    disagreement=int(special.nonneutral_sign_disagreement.sum())
    duration=chosen[chosen.subset=='TAV'].groupby('rate')[['macro_f1','mae']].mean()
    trend='Macro-F1随这四档缺失比例单调下降。' if np.all(np.diff(duration.macro_f1)<=0) else 'Macro-F1并非每一档都严格单调，局部波动应与三次遮挡的变异一起判断。'
    shape50=shape_table[shape_table.rate==.5].set_index('condition')
    pos_gap=float(pos50.macro_f1.max()-pos50.macro_f1.min())
    report=f'''# 问题二实验报告

## 1. 结论与任务完成情况

按照原题的三分类极性与连续强度目标，完成了统一数据接口、8类学习模型、主线4种结构各3种子拟合、验证集选择、留出测试压力实验和附件3全部30条预测。最终选择 **{LABELS[selected]}（`{selected}`）的三个种子集成**。选择发生在测试评价之前，依据及权重哈希见 `selection_lock.json`。

完整输入测试集727条：Accuracy **{a.accuracy:.4f}**，Macro-F1 **{a.macro_f1:.4f}**，MAE **{a.mae:.4f}**，Pearson **{a.pearson:.4f}**。三个模态同步缺失50%、随机位置的三次遮挡平均：Macro-F1 **{stress.macro_f1:.4f}**、MAE **{stress.mae:.4f}**。对应干净训练固定融合为 {ref.macro_f1:.4f}、{ref.mae:.4f}。

专项结果已写入 [附件3_问题二预测.csv](results/附件3_问题二预测.csv)。附件3没有标签，不能据此声称专项准确率。本轮不进入问题三。

## 2. 数据、接口与约束

附件2使用对齐版，训练3395、验证728、测试727；类别0/1/2分别是负向/中性/正向。全部4850条与 `label.xlsx` 的强度及划分吻合，源视频在三个划分之间无交叉。`text_bert` 与冻结编码器重新分词的token ID全量一致。审计详见 [data_audit.json](results/data_audit.json)。

附件3缺少预计算 `text`，因此三个划分及专项输入均由同一个冻结DistilBERT编码。音频74维、视觉35维沿用附件数据；不赋予未公开维度凭空推测的物理意义。该通用模型名称、精确修订、哈希及官方来源记录在[代码说明](代码说明.md)。不使用外部情感标注集，也不将问题一生成的特征混入训练。

有效词槽 `P` 排除CLS、SEP、尾部padding；`O` 表示范围内的可用观测；人工删除mask单独保存。文本删除先作用于token和attention，再重算上下文，行为测试确认被删除词无法泄露。音视频全零行视为不可用，但不是已知人工缺失。训练集有效范围内音频可用率约99.92%、视觉约94.47%，不能把整个50槽张量的零行比例当作缺失率。

附件3出现 {special_audit['unknown_token_total']} 个UNK token；统一按词义不可恢复处理，不反查原文本，也不把UNK来源臆断为某种故障。专项有效范围不确定的样本数为 {len(special_audit['span_uncertain_sample_ids'])}。其可用率见 [专项审阅表](results/附件3_预测审阅.csv)。

缺失时长用有效词槽比例表示，不是秒。人工删除数为 `max(1,floor(L*r+0.5))`，短样本的实际比例与名义值可能不同；[corruption_audit.csv](results/corruption_audit.csv)记录实际比例。

## 3. 数学模型与求解

完整符号定义、损失推导、掩码规则与鲁棒性边界见 [模型定义.md](模型定义.md)。

对训练集有效、可用的各模态特征拟合均值与标准差：`x̃=(x−μ)/max(σ,10⁻⁴)`，截断到[-8,8]后将不可用位置置零。冻结编码器不更新参数，所有学习参数仅用训练集标签更新。

三个模态分支各投影到64维，再通过两个残差时序卷积层，卷积核3、膨胀率1和2。将时间加权池化与均值池化拼接，投影为64维片段表示 `h_m`。显式掩码版本仅在可观测位置池化，并输入可用比例、最长不可用连续段比例及有效长度比例。固定融合取 `z=(h_T+h_A+h_V)/3`；动态门控取 `β=softmax(g(h_m,q_m))`、`z=Σβ_m h_m`，整段不可用模态不提供内容表示。

融合向量经共享全连接、GELU和dropout得到 `v`，输出 `p=softmax(W_c v+b_c)`、`ŷ=3 tanh(w_rᵀv+b_r)`。最小化类别加权交叉熵与SmoothL1之和，权重按训练集频数的逆平方根计算。分类与回归是两个头，预测可能存在符号分歧；未利用测试标签修正输出。

训练使用AdamW、batch64、初始学习率0.001、weight decay0.001、dropout0.2、最多35轮、早停耐心7轮。增强训练从6个固定随机连续遮挡视图抽样，25%概率使用完整视图；覆盖T/A/V/TA/TV/AV/TAV、10%/30%/50%/70%、开头/中间/末尾/随机及同步/错开删除。该有限增强库控制了重编码成本，不能等价宣称见过所有故障。

单模态与统计池化各1种子；固定融合、增强固定融合、掩码增强、门控增强各3种子。三种子集成平均分类概率与回归输出，不采用测试集阈值。完整配置见 [protocol.json](protocol.json)。

## 4. 验证集选择

统一选择目标是完整输入的 `MAE+1−MacroF1` 权重0.5，T/A/V/TAV各30%缺失及TAV50%缺失五个场景各0.1。越小越好。每次训练先按此目标早停，再比较四组三种子集成。

{md(mainval)}

验证集完整输入结果：

{md(valtable)}

上述选择只说明在预先确定的候选与验证目标内最优，不证明全体可能模型中的最优性。动态门控是否胜出以表内实测为准。完整输入下的Macro-F1和中性召回没有全面优于其他模型；鲁棒性收益需要与这类代价一并报告，不能将验证综合目标获胜写成所有指标均最优。

验证集错误分组（完整输入）：

{md(vg.reset_index().drop(columns='scenario'))}

![验证集混淆与强度误差](results/figures/08_validation_errors.png)

验证集上平均强度绝对误差最大的分组为 **{hardest_valid}**（MAE={vg.loc[hardest_valid,'mae']:.4f}）；中性被误判时，更多落入 **{['负向','中性','正向'][neutral_to]}**（{vcm[1,neutral_to]}条）。{compression}连续强度回归与三分类分界并非同一优化目标，弱情感/中性边界混淆需要与强情感幅度误差分别检查。此处是对真实标签与预测的诊断，不断言存在可观测的“模态冲突”真值。验证集最差8条文本、真值、预测和误差见 [validation_representative_errors.csv](results/validation_representative_errors.csv)，全量完整/缺失对照见 [validation_prediction_review.csv](results/validation_prediction_review.csv)。

以下三个验证案例使错误诊断可以回查。评分语义、比较结构属于文本层面的可能原因，未通过模态归因实验验证因果关系；强度低估则可以直接由真值与预测数值确认。

{md(case_table)}

## 5. 留出测试与基线

{md(clean_table)}

![完整输入基线比较](results/figures/01_clean_comparison.png)

Accuracy按三类计算，F1主要采用Macro-F1，同时保存Weighted-F1、每类F1和中性召回。常数回归的Pearson无定义，结果留空。表中四种主模型为三种子集成，其余学习基线为单种子，因此结构消融另提供一致三种子结果，不能将集成收益全归于结构。

压力实验共 {len(pd.read_csv(ROOT/'results/corruption_audit.csv'))} 个输入场景（含完整输入）：7组合×4比例×4位置；随机位置3次、确定位置1次；另比较同总长度多短段以及多模态错开缺失。所有模型共用mask。原始指标见 [test_metrics.csv](results/test_metrics.csv)，按训练种子汇总的消融见 [ablation_seed_summary.csv](results/ablation_seed_summary.csv)。

![消融与缺失比例](results/figures/02_ablation_missing_rate.png)

## 6. 缺失类型、位置、长度与形状

50%随机单连续段缺失，各模态组合的平均表现：

{md(modality.reset_index())}

在该条件下，Macro-F1最低的是 **{hardest}**，最高的是 **{easiest}**。这描述本数据、本模型和本遮挡机制下的敏感性，不是模态的因果贡献。

{av_note}

TAV同步随机缺失由10%增至70%时，Macro-F1从 {duration.loc[.1,'macro_f1']:.4f} 变为 {duration.loc[.7,'macro_f1']:.4f}，MAE从 {duration.loc[.1,'mae']:.4f} 变为 {duration.loc[.7,'mae']:.4f}。{trend}这说明应同时报告完整输入精度与严重信息损失时的退化，不能把“鲁棒”理解为信息丢失后性能不变。

![缺失模态和长度](results/figures/03_modality_and_duration.png)

TAV同步缺失50%时，表现最差的位置为 `{worstpos}`，最好为 `{bestpos}`，Macro-F1相差 {pos_gap:.4f}；具体数值与其他比例见图。随机位置报告三次均值，固定位置只做一次确定删除。位置效应不应被外推为原视频某个固定秒数的重要性。

![位置影响](results/figures/04_position.png)

同总缺失比例、TAV场景的连续段形状/同步性比较：

{md(shape_table)}

50%时，多短段相对单长段同步缺失的Macro-F1差值为 {shape50.loc['多短段','macro_f1']-shape50.loc['单长段同步','macro_f1']:+.4f}，错开相对同步的差值为 {shape50.loc['单长段错开','macro_f1']-shape50.loc['单长段同步','macro_f1']:+.4f}。这些是实测均值差；本实验没有为所有形状对比做多重比较校正，不据微小差值宣称普遍优劣。

![形状与同步性](results/figures/05_shape_and_sync.png)

多短段通过三个互不相交分区内各删除一段构造；在短序列或高比例情况下，边界相邻的段可能连接。错开场景使用各模态独立起点，并不保证三个区间互不重叠。

## 7. 不确定性与误差归因边界

选中模型相对干净训练固定融合，在TAV50%随机场景的第一个固定遮挡实现上，按源视频分组进行500次配对bootstrap：Macro-F1差值 {ci['macro_f1']['delta']:.4f}，95%区间 [{ci['macro_f1']['ci95'][0]:.4f}, {ci['macro_f1']['ci95'][1]:.4f}]；MAE差值 {ci['mae']['delta']:.4f}，区间 [{ci['mae']['ci95'][0]:.4f}, {ci['mae']['ci95'][1]:.4f}]。F1差值为正、MAE差值为负才表示改善。区间涵盖0时不能据此断言有稳定优势。该区间不合并其他遮挡重复，也不覆盖所有分布外故障。

![混淆与回归误差](results/figures/06_errors.png)

完整输入的中性召回为 {a.neutral_recall:.4f}；具体混淆方向见图。[error_groups.csv](results/error_groups.csv)进一步按真实强度分组，并用but/however/although/though/yet等字符串线索检查转折类文本。该线索仅是可复查的启发式分组，不是人工“讽刺/模态冲突”真值。不能仅凭不透明特征维度或门控权重宣布模型识别了讽刺。

绝对强度误差最大的8条完整输入样本保留在 [representative_errors.csv](results/representative_errors.csv)，全量逐样本记录见 [test_prediction_review.csv](results/test_prediction_review.csv)。应结合类别混淆、强弱情感误差和遮挡前后的变化解释失败，不把单个案例当普遍规律。

![训练曲线](results/figures/07_training.png)

## 8. 附件3交付与复现

最终CSV恰好30行，列名 `sample_id,polarity,intensity`；polarity采用negative/neutral/positive。当前预测分布为 `{counts}`。详细审阅文件另外给出三类概率、三种子强度标准差、各模态可用比例与融合权重。专项非中性分类与回归符号分歧有 {disagreement} 条，保留原双头输出以便核查；中性类别头也可能输出非零连续强度。

概率未经外部校准；种子标准差不是置信区间；融合权重不是问题三要求的归因证据。附件3只用于最终推理，无标签、无专项调参、未从另一个特征版本补全输入。

可复现命令、边界条件、固定编码器哈希及环境说明见 [代码说明.md](代码说明.md)。训练模型权重、标准化统计、锁定文件、日志和实验CSV均已保存。缓存和通用预训练大权重不进入提交压缩包；本轮不生成问题三结果。
'''
    (ROOT/'问题二实验报告.md').write_text(report,encoding='utf-8')
    # 轻量离线审阅页，适合IDE直接打开；不需要外部CDN。
    rows=[]
    for r in special.to_dict('records'):
        rows.append(f'<tr data-polarity="{r["polarity"]}"><td>{html.escape(r["sample_id"])}</td><td>{r["polarity"]}</td><td>{r["intensity"]:.4f}</td><td>{r["intensity_seed_std"]:.4f}</td><td>{r["observed_fraction_T"]:.0%}</td><td>{r["observed_fraction_A"]:.0%}</td><td>{r["observed_fraction_V"]:.0%}</td></tr>')
    images=''.join(f'<figure><img loading="lazy" src="figures/{f.name}"><figcaption>{html.escape(f.stem)}</figcaption></figure>' for f in sorted(FIG.glob('*.png')))
    page=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>问题二实验审阅</title><style>
body{{font:16px/1.7 system-ui,sans-serif;margin:0;background:#f4f6f9;color:#17243a}}main{{max-width:1200px;margin:auto;padding:28px}}h1{{font-size:30px}}.cards{{display:flex;gap:16px;flex-wrap:wrap}}.card{{background:white;padding:18px 24px;border-radius:12px;flex:1;min-width:160px}}.card b{{display:block;font-size:28px;color:#236a80}}section,figure{{background:white;padding:24px;border-radius:12px;margin:20px 0}}img{{max-width:100%}}table{{border-collapse:collapse;width:100%;font-size:14px}}td,th{{padding:8px;border-bottom:1px solid #e4e8ef;text-align:left}}thead{{background:#edf2f6}}select,input{{font:inherit;padding:6px;margin-right:12px}}a{{color:#126179}}.muted{{color:#637087}}figcaption{{color:#637087;font-size:13px}}
</style><main><h1>问题二 · 局部连续缺失实验</h1><p>最终模型：<b>{LABELS[selected]}</b>，三个训练种子集成。仅验证集选模型；测试集727条；专项30条。</p><div class="cards"><div class="card">完整输入 Accuracy<b>{a.accuracy:.3f}</b></div><div class="card">完整输入 Macro-F1<b>{a.macro_f1:.3f}</b></div><div class="card">完整输入 MAE<b>{a.mae:.3f}</b></div><div class="card">TAV缺失50% Macro-F1<b>{stress.macro_f1:.3f}</b></div></div>
<section><h2>实验记录</h2><p><a href="../问题二实验报告.md">完整实验报告</a> · <a href="../模型定义.md">数学模型定义</a> · <a href="../代码说明.md">代码与复现说明</a> · <a href="test_metrics.csv">全量指标</a> · <a href="附件3_问题二预测.csv">下载30条预测</a></p><p class="muted">缺失比例相对于有效词槽，不是秒；附件3无标签。融合权重不是因果解释；模型概率和种子分歧未经置信度校准。</p>{clean_table.round(4).to_html(index=False,border=0)}</section>
<section><h2>附件3预测审阅</h2><select id="polarity"><option value="">全部极性</option><option>negative</option><option>neutral</option><option>positive</option></select><input id="search" placeholder="输入样本编号"><table id="special"><thead><tr><th>样本</th><th>极性</th><th>强度</th><th>种子标准差</th><th>文本可用</th><th>音频可用</th><th>视觉可用</th></tr></thead><tbody>{''.join(rows)}</tbody></table></section>{images}<p class="muted">本页由实际实验结果生成。本轮止于问题二。</p></main><script>function filter(){{let p=document.querySelector('#polarity').value,s=document.querySelector('#search').value.toLowerCase();document.querySelectorAll('#special tbody tr').forEach(r=>r.hidden=!!((p&&r.dataset.polarity!==p)||!r.cells[0].textContent.toLowerCase().includes(s)))}}document.querySelector('#polarity').onchange=filter;document.querySelector('#search').oninput=filter;</script></html>'''
    (ROOT/'results/problem2_review.html').write_text(page,encoding='utf-8')
    save_json(ROOT/'results/summary.json',dict(selected=selected,clean={k:float(a[k]) for k in ['accuracy','macro_f1','mae','pearson','neutral_recall']},
        tav_50_random={k:float(stress[k]) for k in ['accuracy','macro_f1','mae','pearson']},special_prediction_count=len(special),
        scenario_count=len(pd.read_csv(ROOT/'results/corruption_audit.csv')),training_fits=len(list((ROOT/'results').glob('completed_*.json')))))
    print('Report and offline dashboard generated.',flush=True)


if __name__=='__main__':
    main()
