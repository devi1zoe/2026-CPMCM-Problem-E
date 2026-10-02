# 本程序及代码是在人工智能工具辅助下完成的。

# 人工智能工具名称：GPT-6 Astra

# 版本/型号：gpt-6-astra

# 开发机构/公司：OpenAI

# 版本颁布日期：2026年9月3日

"""问题三报告、统计图与20张离线可回看解释卡，全部来源于实际结果。"""
from common import *
import html
from urllib.parse import quote
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from sklearn.metrics import confusion_matrix

MOD={'text':'文本','audio':'音频','vision':'视觉','mixed':'混合','weak':'作用弱'}
POL={'negative':'负向','neutral':'中性','positive':'正向'}
MODEL={'step2_fixed_clean':'完整训练固定融合','step2_fixed_aug':'连续缺失增强固定融合','step2_mask_aug':'缺失掩码增强','step2_gated_aug':'问题二门控增强','step3_coalition_aug':'问题三组合增强'}
FIG=ROOT/'results/figures'

def md(df):
    def f(x):
        if isinstance(x,(float,np.floating)):return '—' if pd.isna(x) else f'{x:.4f}'
        return str(x).replace('|','/').replace('$','\\$').replace('\n','<br>')
    return '| '+' | '.join(df.columns)+' |\n| '+' | '.join(['---']*len(df.columns))+' |\n'+'\n'.join('| '+' | '.join(f(x) for x in row)+' |' for row in df.itertuples(index=False,name=None))

def savefig(name):
    plt.tight_layout();plt.savefig(FIG/(name+'.png'),dpi=155,bbox_inches='tight');plt.close()

def span_label(e):
    if e['start_seconds'] is None:return f"槽[{e['slot_start']},{e['slot_end']})；时间待复核"
    return f"槽[{e['slot_start']},{e['slot_end']})；估计{e['start_seconds']:.2f}–{e['end_seconds']:.2f}秒"

def local_svg(rows):
    vals=np.array([r['class_probability_drop'] for r in rows]);xs=np.array([r['slot_start'] for r in rows])
    lo=min(-.005,float(vals.min()));hi=max(.005,float(vals.max()));pad=(hi-lo)*.1;lo-=pad;hi+=pad
    xx=35+(xs-xs.min())/max(1,xs.max()-xs.min())*550;yy=115-(vals-lo)/(hi-lo)*95
    zero=115-(0-lo)/(hi-lo)*95
    line=' '.join(f'{x:.2f},{y:.2f}' for x,y in zip(xx,yy))
    marks=[]
    for x,y,r in zip(xx,yy,rows):
        color='#c44e52' if r['is_selected_evidence'] else ('#277ca2' if r['is_ig_candidate'] else '#777')
        radius=5 if r['is_selected_evidence'] or r['is_ig_candidate'] else 2
        marks.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{radius}" fill="{color}"><title>槽[{r["slot_start"]},{r["slot_end"]})；概率差 {r["class_probability_drop"]:.4f}</title></circle>')
    return f'<svg viewBox="0 0 620 140" role="img" aria-label="同长度局部删除效应分布"><line x1="35" x2="585" y1="{zero}" y2="{zero}" stroke="#bbb"/><polyline points="{line}" fill="none" stroke="#53758d" stroke-width="2"/>{"".join(marks)}<text x="2" y="20" font-size="11">{hi:.2f}</text><text x="2" y="115" font-size="11">{lo:.2f}</text><text x="35" y="135" font-size="11">起始槽 {xs.min()}</text><text x="530" y="135" font-size="11">{xs.max()}</text></svg>'

def main():
    field_doc=quote(os.path.relpath(PACKAGE/"结果字段说明.md",ROOT),safe="/")
    run_doc=quote(os.path.relpath(PACKAGE/"复现说明.md",ROOT),safe="/")
    FIG.mkdir(exist_ok=True)
    font='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc';font_manager.fontManager.addfont(font)
    plt.rcParams.update({'font.family':font_manager.FontProperties(fname=font).get_name(),'font.size':10,'axes.unicode_minus':False,'text.parse_math':False})
    summary=json.loads((ROOT/'results/validation_explanation_summary.json').read_text())
    lock=json.loads((ROOT/'selection_lock.json').read_text());special=json.loads((ROOT/'results/special_explanations.json').read_text())
    audit=json.loads((ROOT/'results/special_inference_audit.json').read_text())
    scores=pd.read_csv(ROOT/'results/validation_prediction_metrics.csv');ens=scores[scores.seed=='ensemble'].copy()
    metrics=summary['prediction'];fid=pd.read_csv(ROOT/'results/validation_fidelity.csv',dtype={'sample_id':str})
    mod=pd.read_csv(ROOT/'results/validation_modality_contributions.csv',dtype={'sample_id':str})
    mapping=pd.read_csv(ROOT/'results/time_mapping_audit.csv',dtype={'sample_id':str})
    pred=pd.read_csv(ROOT/'results/validation_predictions.csv',dtype={'sample_id':str})
    raw=np.load(ROOT/'results/validation_probe_outputs.npz');cube=raw['outputs'];target=raw['target']
    fid['random_conditional_class_retained']=(cube[:,:,1:,1,:3].argmax(-1)==target[:,None,None]).mean(-1).reshape(-1)
    fid['random_unconditional_class_retained']=(cube[:,:,1:,2,:3].argmax(-1)==target[:,None,None]).mean(-1).reshape(-1)
    fid.to_csv(ROOT/'results/validation_fidelity_comparison.csv',index=False)
    primary=fid[fid.is_main]
    groupcols=['deletion_drop','random_deletion_drop','deletion_advantage','conditional_sufficiency_gap','random_conditional_gap',
        'unconditional_sufficiency_gap','random_unconditional_gap','conditional_class_retained','random_conditional_class_retained','unconditional_class_retained','random_unconditional_class_retained']
    bymod=fid.groupby('modality')[groupcols].mean().reindex(['text','audio','vision'])
    bymod.to_csv(ROOT/'results/fidelity_by_modality.csv',encoding='utf-8-sig')
    x=np.arange(len(ens));fig,ax=plt.subplots(1,2,figsize=(11,4))
    ax[0].bar(x-.18,ens.accuracy,.36,label='Accuracy');ax[0].bar(x+.18,ens.macro_f1,.36,label='Macro-F1');ax[0].legend();ax[0].set_ylim(0,1)
    ax[1].bar(x,ens.mae,color='#d48946');ax[1].set_ylabel('MAE（越低越好）')
    for a in ax:a.set_xticks(x,[MODEL[m] for m in ens.model],rotation=25,ha='right');a.grid(axis='y',alpha=.2)
    ax[0].set_title('完整验证集 · 三个种子集成');ax[1].set_title('问题三按完整输入目标选择')
    savefig('01_prediction_comparison')
    z=np.load(ROOT/'results/validation_predictions.npz');cm=confusion_matrix(z['c'],z['prob'].argmax(1),labels=[0,1,2])
    fig,ax=plt.subplots(1,2,figsize=(10,4));ax[0].imshow(cm,cmap='Blues')
    ax[0].set_xticks(range(3),['负向','中性','正向']);ax[0].set_yticks(range(3),['负向','中性','正向']);ax[0].set(xlabel='预测',ylabel='真实',title='验证集混淆矩阵')
    for i in range(3):
        for j in range(3):ax[0].text(j,i,str(cm[i,j]),ha='center',va='center',color='white' if cm[i,j]>.55*cm.max() else 'black')
    ax[1].scatter(z['y'],z['pred'],s=10,alpha=.35);ax[1].plot([-3,3],[-3,3],'k--',lw=1);ax[1].set(xlim=(-3,3),ylim=(-3,3),xlabel='真实强度',ylabel='预测强度',title='验证集强度误差')
    savefig('02_prediction_errors')
    pred['intensity_group']=pd.cut(pred.true_intensity,[-3.01,-1.5,-.00001,.00001,1.5,3.01],labels=['强负向','弱负向','中性','弱正向','强正向'])
    error=[]
    for name,g in pred.groupby('intensity_group',observed=True):
        error.append(dict(分组=str(name),样本数=len(g),Accuracy=float((g.true_class==g.predicted_class).mean()),MAE=float(g.abs_error.mean()),偏差=float((g.predicted_intensity-g.true_intensity).mean())))
    errors=pd.DataFrame(error);errors.to_csv(ROOT/'results/error_groups.csv',index=False,encoding='utf-8-sig')
    pred.nlargest(8,'abs_error').to_csv(ROOT/'results/representative_errors.csv',index=False,encoding='utf-8-sig')
    fig,ax=plt.subplots(1,2,figsize=(10,4))
    absmeans=mod.groupby('modality').absolute_share.mean().reindex(['text','audio','vision'])
    ax[0].bar(['文本','音频','视觉'],absmeans,color=['#4477aa','#dd8452','#55a868']);ax[0].set(ylabel='平均绝对贡献占比',ylim=(0,1),title='验证集三模态作用程度')
    ax[1].boxplot([mod[mod.modality==m].class_shapley for m in ['text','audio','vision']],tick_labels=['文本','音频','视觉'],showfliers=False)
    ax[1].axhline(0,color='#777',lw=1);ax[1].set(ylabel='固定预测类别的有符号贡献',title='正值支持 / 负值抑制')
    savefig('03_modality_contributions')
    ft=pd.DataFrame([
        dict(指标='删除后的概率下降',IG=primary.deletion_drop.mean(),随机=primary.random_deletion_drop.mean()),
        dict(指标='条件保留概率缺口',IG=primary.conditional_sufficiency_gap.mean(),随机=primary.random_conditional_gap.mean()),
        dict(指标='只保留证据概率缺口',IG=primary.unconditional_sufficiency_gap.mean(),随机=primary.random_unconditional_gap.mean())])
    fig,ax=plt.subplots(1,2,figsize=(11,4));x=np.arange(3)
    ax[0].bar(x-.18,ft.IG,.36,label='IG筛选');ax[0].bar(x+.18,ft['随机'],.36,label='5个等长随机窗口');ax[0].axhline(0,color='#666',lw=1);ax[0].legend(fontsize=9)
    ax[0].set_xticks(x,ft['指标'],rotation=15,ha='right');ax[0].set_title('主影响模态：独立原输入验证')
    ax[1].hist(primary.deletion_advantage,bins=35,color='#4477aa');ax[1].axvline(0,color='#cc4444');ax[1].set(xlabel='IG删除效应 − 随机删除效应',ylabel='样本数',title='保留负优势和失败样本')
    savefig('04_fidelity')
    wrong=pd.DataFrame([dict(预测正确=bool(correct),样本数=len(g),删除优势=g.deletion_advantage.mean(),正删除效应比例=(g.deletion_drop>0).mean()) for correct,g in primary.groupby('prediction_correct')])
    wrong.to_csv(ROOT/'results/fidelity_correct_vs_wrong.csv',index=False,encoding='utf-8-sig')
    fig,ax=plt.subplots(1,2,figsize=(10,4))
    agreement=[summary['modality_baseline_main_agreement'],summary['modality_seed_main_agreement'],summary['stability']['noise_main_agrees']]
    ax[0].bar(['换替代机制','跨训练种子','微小表示扰动'],agreement,color=['#4477aa','#dd8452','#55a868']);ax[0].set(ylim=(0,1.08),ylabel='模态一致程度',title='不同定义的一致性检验')
    ious=[summary['stability']['reference_ig_window_jaccard'],summary['stability']['seed_window_pairwise_jaccard'],summary['stability']['noise_window_jaccard']]
    ax[1].bar(['换IG参考','跨训练种子','微小表示扰动'],ious,color=['#4477aa','#dd8452','#55a868']);ax[1].set(ylim=(0,1.08),ylabel='窗口Jaccard',title='局部证据稳定性有明显边界')
    for a,vals in zip(ax,[agreement,ious]):
        for i,v in enumerate(vals):a.text(i,v+.02,f'{v:.3f}',ha='center')
    savefig('05_stability')
    ig=np.load(ROOT/'results/validation_ig.npz');fig,ax=plt.subplots(1,2,figsize=(10,4))
    ax[0].hist(np.log10(np.maximum(abs(ig['original_residual32']),1e-9)),bins=30,alpha=.6,label='固定32节点')
    ax[0].hist(np.log10(np.maximum(abs(ig['residual']),1e-9)),bins=30,alpha=.6,label='自适应加密');ax[0].axvline(-3,color='#b44',ls='--');ax[0].legend();ax[0].set(xlabel='log10(完备性绝对残差)',ylabel='样本数')
    nodes,counts=np.unique(ig['used_nodes'],return_counts=True);ax[1].bar([str(n) for n in nodes],counts);ax[1].set(xlabel='最终积分节点',ylabel='样本数',title='以恒等式控制数值误差')
    savefig('06_numerical_check')
    vals=np.array([[m['class_shapley'] for m in r['modalities']] for r in special]);lim=abs(vals).max()
    fig,ax=plt.subplots(figsize=(7,7));im=ax.imshow(vals,cmap='RdBu_r',vmin=-lim,vmax=lim,aspect='auto')
    ax.set_xticks(range(3),['文本','音频','视觉']);ax.set_yticks(range(20),[f"{r['sample_id']} · {POL[r['polarity']]}" for r in special]);ax.set_title('附件4：固定预测类别的有符号Shapley')
    for i in range(20):
        for j in range(3):ax.text(j,i,f'{vals[i,j]:+.3f}',ha='center',va='center',fontsize=8,color='white' if abs(vals[i,j])>.6*lim else 'black')
    fig.colorbar(im,ax=ax,label='概率分数贡献');savefig('07_special_modality_map')
    typical=[next(r for r in special if r['sample_id']==id_) for id_ in ['04','09','15']]
    fig,axes=plt.subplots(3,2,figsize=(12,8.5))
    for row,(r,ax) in enumerate(zip(typical,axes)):
        m=r['main_modality'];records=r['window_distributions'][m]
        xs=[v['slot_start'] for v in records];ys=[v['class_probability_drop'] for v in records]
        ax[0].plot(xs,ys,marker='.',color='#4477aa');ax[0].axhline(0,color='#999',lw=1)
        selected=next(v for v in records if v['is_selected_evidence']);igrow=next(v for v in records if v['is_ig_candidate'])
        ax[0].axvline(selected['slot_start'],color='#c44',label='确认窗口');ax[0].axvline(igrow['slot_start'],color='#298',ls='--',label='IG候选')
        ax[0].set(title=f"样本{r['sample_id']} / {POL[r['polarity']]} / 主要{MOD[m]}",xlabel='等长窗口起始槽',ylabel='删除后的目标概率下降');ax[0].legend(fontsize=8)
        ax[1].imshow(plt.imread(ROOT/'results'/r['keyframe']['image']));ax[1].axis('off')
        ax[1].set_title(('视频浏览帧（非定位证据）' if r['keyframe']['type'].startswith('context') else '视觉窗口内/附近的原始帧')+f" · PTS {r['keyframe']['pts_seconds']:.3f}s")
    savefig('08_typical_evidence')
    table=[]
    for r in special:
        e=next((e for e in r['evidence'] if e['modality']==r['main_modality']),r['evidence'][0])
        table.append(dict(样本=r['sample_id'],极性=POL[r['polarity']],强度=r['intensity'],主要模态=MOD[r['main_modality']],
            文本贡献=r['contribution_text'],音频贡献=r['contribution_audio'],视觉贡献=r['contribution_vision'],关键位置=span_label(e)))
    table=pd.DataFrame(table);table.to_csv(ROOT/'results/special_summary.csv',index=False,encoding='utf-8-sig')
    ptable=ens[['model','accuracy','macro_f1','mae','pearson','neutral_recall']].copy();ptable.model=ptable.model.map(MODEL)
    epochs={str(s):int(torch_epoch(s)) for s in [17,29,43]}
    baseline=ens[ens.model=='step2_gated_aug'].iloc[0]
    risk=mapping[mapping.requires_timing_review]
    context_count=sum(r['keyframe']['type'].startswith('context') for r in special)
    report=f'''# 问题三实验报告

## 1. 完成结果与主要结论

完成可解释性预测模型、728条验证样本的解释检验、附件4全部20条预测与解释，以及20张可回看解释卡。最终采用 **问题三模态组合增强的三个种子集成**；按完整验证集选择，与问题二的缺失鲁棒性目标分开。

验证集Accuracy **{metrics['accuracy']:.4f}**、Macro-F1 **{metrics['macro_f1']:.4f}**、MAE **{metrics['mae']:.4f}**、Pearson **{metrics['pearson']:.4f}**。这些是参与模型选择的验证集成绩，不冒充新的独立盲测结果。

固定原预测类别后，IG筛选的主影响模态窗口被删除，目标概率平均下降 **{100*primary.deletion_drop.mean():.2f}个百分点**；5个等长随机窗口平均下降 **{100*primary.random_deletion_drop.mean():.2f}个百分点**。优势为 **{100*primary.deletion_advantage.mean():.2f}个百分点**，源视频聚类bootstrap 95%区间为 [{100*summary['random_advantage_ci95'][0]:.2f}, {100*summary['random_advantage_ci95'][1]:.2f}] 个百分点。它是对模型解释忠实性的证据，不是情感预测准确率的提升。

局部解释并非普遍稳定：跨训练种子的窗口平均Jaccard仅 **{summary['stability']['seed_window_pairwise_jaccard']:.3f}**。因此交付同时保留贡献方向、替代基线、种子差异、反向影响和时间定位限制。

交付入口：[解释审阅页](results/explanation_review.html)、[全量预测与解释CSV](results/附件4_问题三预测与解释.csv)、[结果字段说明]({field_doc})、[复现说明]({run_doc})。

## 2. 数据与预测模型

使用附件2对齐版训练3395、验证728；三分类编码0/1/2为负向/中性/正向，真实中性强度严格为0。参数、标准化和解释参考统计只由训练集学习。沿用统一冻结文本编码器及64维时序门控网络，损失为类别加权交叉熵加SmoothL1。

从问题二三个训练种子的权重出发，以80%完整输入和20%随机真子模态组合继续训练。初始化本身作为epoch0候选保留，三个种子的最佳轮次为 `{epochs}`。本次43号种子保留初始参数，不能写成所有组件都选用了组合增强之后的权重。

微调学习率0.0003，最多20轮，batch64，weight decay0.001，dropout0.2，早停耐心5。完整验证目标 `MAE+1−MacroF1` 最低的候选获选；配置、权重及哈希见 `protocol.json` 与 `selection_lock.json`。

{md(ptable)}

相对问题二门控集成，MAE由 {baseline.mae:.4f} 变为 {metrics['mae']:.4f}；分类指标没有全面提高。不能把综合选择目标获胜表述为所有指标最优。附件2测试集已在问题二使用，本问没有将它重新宣称为未见盲测集；附件4无标签，不能计算专项准确率。

![预测比较](results/figures/01_prediction_comparison.png)

## 3. 解释的数学含义

分类解释目标固定为完整输入原预测类别的平均概率 `F_c(X')=mean(p_original_class(X'))`，遮挡后不改变类别目标；强度另解释平均回归输出。三模态8个组合精确枚举，按Shapley公式分配 `F(full)−F(empty)`。逐样本检查贡献总和；同时保存有符号贡献和绝对贡献占比，不把门控权重直接当解释。

主基线是模态不可用：特征为零、观测掩码为False、可用性统计重算。敏感性基线用训练集逐维特征中位数替换表示并保留原掩码，它是另一种干预机制。主导模态由最大绝对贡献确定；贡献极弱或分散时允许weak/mixed，不强行制造可信主导模态。

局部层先在连续表示上用IG排序，再对20%有效槽长度的连续窗口做原输入验证。文本必须先删除token和attention再重新编码；音视频删除的是题目提供的特征行，不等价于重新剪辑物理音视频并重新提取全部特征。积分路径的文本位置只用于候选排序，不将上下文向量直接称为原词贡献。

专项解释另外穷举同长度窗口并给出最大支持或强反向影响位置。该穷举结果用于定位与复算，不能用其按定义取得的最大删除分数循环证明忠实性；独立证据来自验证集预先IG排序后的随机对照。

公式与建模原理见论文正文；字段、替代规则与解释边界见 [结果字段说明]({field_doc})。

![模态作用程度](results/figures/03_modality_contributions.png)

## 4. 忠实性、充分性与反例

主影响模态上的结果如下。删除效应越大表示移除该片段越削弱原目标；保留缺口越小表示该片段在指定上下文中越能维持原目标。

{md(ft)}

条件保留（其他模态完整）的原类别保持率：IG **{primary.conditional_class_retained.mean():.2%}**、随机 **{primary.random_conditional_class_retained.mean():.2%}**。只保留一个模态的证据窗口、其他模态均移除时：IG **{primary.unconditional_class_retained.mean():.2%}**、随机 **{primary.random_unconditional_class_retained.mean():.2%}**。两种条件分开报告，不能把保留其他完整模态的试验称为“只靠该证据”。

无条件保留的平均概率缺口为负，说明在本模型与替代规则下，移除其他信息有时提高原类别概率。它可能反映模态冲突、输出先验或反事实分布变化；单凭负缺口不能证明该片段更真实或原样本多余。

主影响窗口的正删除效应比例为 **{summary['positive_deletion_fraction']:.2%}**，其余样本保留为失败/反向影响，不删去。预测正确与错误样本分别统计：

{md(wrong)}

![忠实性](results/figures/04_fidelity.png)

三模态分别的结果见 [fidelity_by_modality.csv](results/fidelity_by_modality.csv)，2184条逐样本逐模态记录见 [validation_fidelity_comparison.csv](results/validation_fidelity_comparison.csv)，全部39312个探测结果保存在 `validation_probe_outputs.npz`。主影响模态汇总使用绝对贡献最大的索引作受控对比；弱/混合样本的正式输出状态仍保留。

## 5. 稳定性、基线与数值精度

更换模态替代机制后，主要影响模态一致率 **{summary['modality_baseline_main_agreement']:.2%}**；跨种子平均主影响模态一致程度 **{summary['modality_seed_main_agreement']:.2%}**。局部窗口对不同训练种子的敏感性明显，平均Jaccard **{summary['stability']['seed_window_pairwise_jaccard']:.3f}**；更换IG参考表示时为 **{summary['stability']['reference_ig_window_jaccard']:.3f}**。

在96条按真实三类各取32条的验证样本上，0.02标准差的微小表示扰动下，主模态一致率和窗口Jaccard均为 {summary['stability']['noise_window_jaccard']:.3f}。这只描述该小扰动，不能覆盖真实遮挡、说话人变化或任意噪声。不同指标定义也不同，不把它们混成统一“解释准确率”。

![稳定性](results/figures/05_stability.png)

初始32节点IG最大完备性残差为 {summary['ig']['initial32_max_error']:.5f}。按恒等式自适应加密后，节点数使用分布为 `{summary['ig']['nodes_used']}`，残差95%分位为 {summary['ig']['completeness_p95']:.6f}，最大值为 {summary['ig']['completeness_max']:.6f}，超过0.001容差的样本数为 {summary['ig']['failed_tolerance']}。这是积分数值检查，不是预测精度评价。

![积分检查](results/figures/06_numerical_check.png)

## 6. 附件4全量预测与解释

20条预测分布为 `{audit['polarity_counts']}`；主要参考模态为 `{audit['primary_main_counts']}`。作用程度是本模型对固定预测目标的贡献，不是人的真实情绪成因。音频没有成为这20条中最大的绝对贡献模态，不能因此删去音频结果或声称它没有作用。

{md(table)}

![专项模态贡献](results/figures/07_special_modality_map.png)

20条CSV每行都包含三模态贡献及三个关键证据记录；`evidence` 为规范JSON，保存原文字符、特征槽、贡献方向、IG候选、确认删除效应、时间状态和关键帧。详细解释卡数据见 [special_explanations.json](results/special_explanations.json)。

## 7. 典型解释卡与时间映射

选取04号展示视觉为主且模态方向相反的情况，09号展示负向文本证据，15号展示时间映射风险。三者均无专项标签，因此只能讨论模型的预测依据，不能断言预测正确。

![典型证据与实际视频帧](results/figures/08_typical_evidence.png)

对20个视频均按实际PTS解码；原文不被诊断ASR替换，分词ID与所用词表全量一致。原题给出对齐槽对应关系，另对同词多子词的音视频特征重复进行了核查：共 {int(mapping.multi_piece_groups.sum())} 组，音频一致 {int(mapping.audio_repeated_groups.sum())} 组、视觉一致 {int(mapping.vision_repeated_groups.sum())} 组。这支持组织规则解释，但不能独立证明官方特征的物理提取时间。

CTC与原文offset用于估计时间，没有用位置比例乘视频时长。通过片段级保守质控的样本有 {int((~mapping.requires_timing_review).sum())} 条；最终选中证据可提供保守估计时间的数量分别为文本 {audit['localized_evidence_counts']['text']}、音频 {audit['localized_evidence_counts']['audio']}、视觉 {audit['localized_evidence_counts']['vision']}。其他局部窗口含低置信词或整体匹配风险，只保留候选时间、原文和槽位；不能把它们写成可靠定位。

需重点复核的原文件：

- **对齐版本/15.pkl 与 videos/15.mp4**：提供转写从“However, despite their poverty...”开始，但诊断识别的大部分内容是此前父亲去世及家庭经济困境，末尾只到“however despite their pover...”。CTC平均分数约0.006，不能将整段提供文本强制分配到这段视频上。保留文本/特征证据，不补造时间。
- **对齐版本/16.pkl 与 videos/16.mp4**：诊断识别与给定句子有相近内容，但整句WER为0.875并带有额外尾部，触发既定复核规则。没有因看起来相似而事后放宽阈值，也不宣称视频一定配错。

7号和18号原文超过模型50槽上限，截断之外的文本不参与预测。句子完整CTC对齐保留，但不能将模型未读到的尾部作为其证据。

全部20条都有精确的提供原文字符与特征槽对应记录。当前 {context_count} 张视觉图片由于选定区间不能可靠定位，只是**全片浏览帧，非已定位证据**。其他帧仍是依据估计词时间选取的原始帧，记录真实PTS而非生成画面。音视频时间映射方法、置信度和局部限制见 [time_mapping_audit.csv](results/time_mapping_audit.csv) 与 [token_time_mapping.csv](results/token_time_mapping.csv)。

## 8. 验证集错误诊断与交付

![预测错误](results/figures/02_prediction_errors.png)

{md(errors)}

中性召回为 {metrics['neutral_recall']:.4f}，弱情感和中性边界仍容易混淆；强情感幅度存在向中间收缩的偏差。最差8条及逐样本真值、预测见 [representative_errors.csv](results/representative_errors.csv)。对错误预测作出忠实解释，并不使预测变正确；解释检验和预测评价必须分别保留。

CSV、数学定义、模型权重、训练日志、环境、基线比较、全量探测记录和解释卡全部提供。脚本执行与恢复方式见 [复现说明]({run_doc})。提交包排除原始视频、冻结通用大模型和可重建缓存，保留精确版本及哈希。
'''
    (ROOT/'问题三实验报告.md').write_text(report,encoding='utf-8')
    build_html(special,metrics,summary)
    save_json(ROOT/'results/report_summary.json',dict(prediction=metrics,primary_fidelity=summary['primary_fidelity'],
        special_samples=20,validation_samples=728,probe_count=int(cube.shape[0]*cube.shape[1]*cube.shape[2]*cube.shape[3]),
        main_counts=audit['primary_main_counts'],timing_review_ids=audit['timing_review_sample_ids'],context_preview_frames=context_count,
        figures=8,selected=lock['selected'],best_epochs=epochs))
    print('Report, 8 figures and 20 interactive evidence cards generated.',flush=True)

def torch_epoch(seed):
    import torch
    return torch.load(ROOT/f'checkpoints/final_{seed}.pt',map_location='cpu',weights_only=True)['epoch']

def build_html(samples,metrics,summary):
    html_field_doc=quote(os.path.relpath(PACKAGE/'结果字段说明.md',ROOT/'results'),safe='/')
    html_run_doc=quote(os.path.relpath(PACKAGE/'复现说明.md',ROOT/'results'),safe='/')
    cards=[]
    for r in samples:
        id_=r['sample_id'];risk=r['requires_timing_review'] or any(e['time_status']!='estimated_CTC' for e in r['evidence'])
        path=DATA/r['source_video'];video=quote(os.path.relpath(path,ROOT/'results'),safe='/')
        modrows=''.join(f'<tr><td>{MOD[m["modality"]]}</td><td class="signed">{m["class_shapley"]:+.4f}</td><td>{m["absolute_share"]:.1%}</td><td>{m["intensity_shapley"]:+.4f}</td><td>{m["class_shapley_seed_std"]:.4f}</td></tr>' for m in r['modalities'])
        evidence=[]
        for e in r['evidence']:
            name=e['modality'];a=e['char_start'];b=e['char_end'];text=r['raw_text']
            marked=html.escape(text[:a])+'<mark>'+html.escape(text[a:b])+'</mark>'+html.escape(text[b:])
            when=e['start_seconds'] if e['start_seconds'] is not None else e.get('candidate_start_seconds')
            end=e['end_seconds'] if e['end_seconds'] is not None else e.get('candidate_end_seconds')
            btn=(f'<button onclick="seek(\'{id_}\',{when:.5f},{end:.5f})">'+('回看估计区间' if e['time_status']=='estimated_CTC' else '查看候选区间（待复核）')+'</button>') if when is not None else '<span>无可用时间区间</span>'
            evidence.append(f'<div class="evidence"><h4>{MOD[name]} · {"支持" if e["role"]=="supporting" else "抑制" if e["role"]=="inhibitory" else "影响弱"}：目标概率下降 {e["class_probability_drop"]:+.4f}</h4><p>{html.escape(span_label(e))} {btn}</p><p class="transcript">{marked}</p>{local_svg(r["window_distributions"][name])}<p class="muted">蓝色点：IG候选；红色点：按实际删除确认的窗口。这里的最大值不作为独立忠实性证明。</p></div>')
        frame_caption='全片浏览帧，非已定位证据' if r['keyframe']['type'].startswith('context') else '原视频帧，时间由CTC估计区间选取'
        cards.append(f'''<details class="card" data-sample="{id_}" data-polarity="{r['polarity']}" data-main="{r['main_modality']}" data-risk="{str(risk).lower()}"><summary><b>{id_} · {POL[r['polarity']]}</b><span>强度 {r['intensity']:+.3f}　主要{MOD[r['main_modality']]}　{'部分时间待复核' if risk else '有估计时间'}</span></summary><div class="inside"><div class="media"><div><video id="video-{id_}" src="{video}" controls preload="none" poster="{r['keyframe']['image']}"></video><p class="muted">{frame_caption} · 实际PTS {r['keyframe']['pts_seconds']:.3f}s</p></div><div><p>固定解释目标：原预测的<b>{POL[r['polarity']]}</b>概率；完整 {r['full_probability']:.4f}，空模态参考 {r['baseline_probability']:.4f}。</p><table><thead><tr><th>模态</th><th>有符号贡献</th><th>绝对占比</th><th>强度贡献</th><th>种子标准差</th></tr></thead><tbody>{modrows}</tbody></table><p class="muted">正贡献支持原类别，负贡献抑制。占比与融合权重不同；标准差不是置信区间。</p><p class="risk">{html.escape('; '.join(r['timing_review_reasons'])) if r['requires_timing_review'] else '时间来自估计对齐，不是官方逐帧真值。'}</p><p class="muted">{'原文尾部超过模型长度，未参与预测。' if r['model_truncated_text'] else ''}</p></div></div>{''.join(evidence)}<p><a href="cards/{id_}.json">完整样本JSON</a> · <a href="mapping/{id_}.json">时间映射与诊断记录</a></p></div></details>''')
    figs=''.join(f'<figure><img loading="lazy" src="figures/{p.name}" alt="{p.stem}"></figure>' for p in sorted(FIG.glob('*.png')))
    page=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>问题三 · 可解释预测审阅</title><style>
body{{margin:0;background:#f2f5f8;color:#172b40;font:16px/1.7 system-ui,sans-serif}}main{{max-width:1200px;margin:auto;padding:28px}}h1{{font-size:30px}}a{{color:#15647b}}.stats{{display:flex;gap:14px;flex-wrap:wrap}}.stat{{padding:16px 24px;background:white;border-radius:12px;flex:1;min-width:160px}}.stat b{{display:block;font-size:28px;color:#176b82}}section,figure{{background:white;border-radius:12px;padding:20px;margin:18px 0}}.card{{background:white;border-radius:12px;margin:12px 0;overflow:hidden}}summary{{padding:18px;cursor:pointer;display:flex;gap:25px;justify-content:space-between}}summary span{{font-size:14px;color:#637387}}.inside{{padding:0 22px 22px}}.media{{display:grid;grid-template-columns:1fr 1fr;gap:25px}}video,img{{max-width:100%;width:100%}}video{{background:#111;max-height:330px}}table{{border-collapse:collapse;width:100%;font-size:14px}}th,td{{padding:7px;border-bottom:1px solid #e1e7ed;text-align:left}}.muted{{font-size:13px;color:#6a7786}}.risk{{font-size:14px;color:#a45c28}}.evidence{{border-top:1px solid #ddd;margin-top:18px;padding-top:8px}}.transcript{{font-size:14px;line-height:1.9}}mark{{background:#ffe0ad;padding:2px}}select,input,button{{font:inherit;padding:7px;border:1px solid #c6d0db;border-radius:5px;background:white;margin:4px}}button{{cursor:pointer;color:#176b82}}svg{{max-width:650px;width:100%}}@media(max-width:760px){{.media{{grid-template-columns:1fr}}summary{{display:block}}summary span{{display:block}}main{{padding:12px}}}}
</style><main><h1>问题三 · 可解释预测与证据回看</h1><p>20条专项样本 · 三模态精确Shapley · 局部删除确认 · 固定原预测目标</p><div class="stats"><div class="stat">验证集Accuracy<b>{metrics['accuracy']:.1%}</b></div><div class="stat">验证集MAE<b>{metrics['mae']:.3f}</b></div><div class="stat">IG窗口删除概率下降<b>{100*summary['primary_fidelity']['deletion_drop']:.2f}pp</b></div><div class="stat">等长随机删除<b>{100*summary['primary_fidelity']['random_deletion_drop']:.2f}pp</b></div></div><section><p><a href="../问题三实验报告.md">实验报告</a> · <a href="{html_field_doc}">字段说明</a> · <a href="{html_run_doc}">复现说明</a> · <a href="附件4_问题三预测与解释.csv">下载20条预测与解释</a></p><p>贡献是模型相对替代基线的作用，不是情感因果真相。跨训练种子局部证据Jaccard约{summary['stability']['seed_window_pairwise_jaccard']:.3f}，不能只凭一张热图断言稳定。</p><p class="risk">15、16号触发片段级时间复核，其他个别局部窗口也可能低置信。未确认的候选时间与浏览帧均明确标记。附件4无标签，本页不提供专项准确率。</p></section><section><input id="search" placeholder="样本编号或转写关键词"><select id="pol"><option value="">全部极性</option><option value="positive">正向</option><option value="neutral">中性</option><option value="negative">负向</option></select><select id="mod"><option value="">全部主要模态</option><option value="text">文本</option><option value="audio">音频</option><option value="vision">视觉</option></select><label><input id="risk" type="checkbox">仅时间待复核</label><span id="count"></span></section>{''.join(cards)}<section><h2>验证与全量比较</h2>{figs}</section></main><script>
function filter(){{let q=document.querySelector('#search').value.toLowerCase(),p=document.querySelector('#pol').value,m=document.querySelector('#mod').value,r=document.querySelector('#risk').checked,n=0;document.querySelectorAll('.card').forEach(c=>{{let ok=(!q||c.textContent.toLowerCase().includes(q))&&(!p||c.dataset.polarity===p)&&(!m||c.dataset.main===m)&&(!r||c.dataset.risk==='true');c.hidden=!ok;if(ok)n++}});document.querySelector('#count').textContent=n+' / 20 条';}}
document.querySelectorAll('input,select').forEach(e=>e.addEventListener('input',filter));filter();
function seek(id,start,end){{let v=document.getElementById('video-'+id);v.pause();if(v._stop)v.removeEventListener('timeupdate',v._stop);let go=()=>{{v.currentTime=Math.max(0,start-.12);v._stop=()=>{{if(v.currentTime>=end+.18)v.pause()}};v.addEventListener('timeupdate',v._stop);v.play().catch(()=>{{}})}};if(v.readyState>=1)go();else{{v.addEventListener('loadedmetadata',go,{{once:true}});v.load()}}v.scrollIntoView({{behavior:'smooth',block:'center'}});}}
</script></html>'''
    (ROOT/'results/explanation_review.html').write_text(page,encoding='utf-8')

if __name__=='__main__':main()
