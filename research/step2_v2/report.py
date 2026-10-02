"""Generate figures, review page, and evidence-based Chinese report from completed artifacts."""
from common import *
from data import load_attachment2
from engine import metrics,checkpoint_paths
import torch
import html
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from sklearn.metrics import confusion_matrix

FIG=ROOT/'results/figures'
LABELS={'old_gated':'旧版门控','text':'文本','audio':'音频','vision':'视觉','fixed':'固定融合','gated':'动态遮挡门控',
    'aux':'+辅助监督','msd':'+共享/特有分解','cce':'+局部增强','imtd':'+可信蒸馏','emc':'+能量协调'}

def md(df):
    def f(v):
        if isinstance(v,(float,np.floating)):return '—' if pd.isna(v) else f'{v:.4f}'
        return str(v).replace('|','/').replace('\n',' ')
    return '| '+' | '.join(map(str,df.columns))+' |\n| '+' | '.join(['---']*len(df.columns))+' |\n'+'\n'.join('| '+' | '.join(f(v) for v in row)+' |' for row in df.itertuples(index=False,name=None))

def fig(name):
    plt.tight_layout();plt.savefig(FIG/f'{name}.png',dpi=150,bbox_inches='tight');plt.close()

def main():
    font=Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    if font.exists():font_manager.fontManager.addfont(str(font));plt.rcParams['font.family']=font_manager.FontProperties(fname=str(font)).get_name()
    plt.rcParams.update({'axes.unicode_minus':False,'text.parse_math':False,'font.size':10})
    lock=json.loads((ROOT/'selection_lock.json').read_text());selected=lock['selected'];candidate=lock['audit_candidate']
    comparison=pd.read_csv(ROOT/'results/selection_comparison.csv');names=comparison.model.tolist()
    data=pd.concat([pd.read_csv(ROOT/f'results/valid_{n}_metrics.csv') for n in names],ignore_index=True)
    data.to_csv(ROOT/'results/validation_metrics.csv',index=False)
    ens=data[data.seed=='ensemble'];clean=ens[ens.name=='clean'].set_index('model').loc[names]
    clean[['accuracy','macro_f1','weighted_f1','mae','pearson','neutral_recall']].to_csv(ROOT/'results/clean_comparison.csv')
    summaries=[]
    for name in names:
        df=data[(data.model==name)&(data.seed!='ensemble')]
        for condition in ['clean','continuous_average']:
            v=df[df.name=='clean'].set_index('seed') if condition=='clean' else df[df.primary==True].groupby(['seed','name'])[['accuracy','macro_f1','mae','pearson']].mean().groupby('seed').mean()
            row=dict(model=name,condition=condition)
            for k in ['accuracy','macro_f1','mae','pearson']:row[k+'_mean']=v[k].mean();row[k+'_seed_std']=v[k].std(ddof=1)
            summaries.append(row)
    pd.DataFrame(summaries).to_csv(ROOT/'results/ablation_seed_summary.csv',index=False)
    x=np.arange(len(names));fig_,axes=plt.subplots(1,2,figsize=(14,5))
    for ax,col in zip(axes,['macro_f1','mae']):
        ax.bar(x,clean[col]);ax.set_xticks(x,[LABELS.get(n,n) for n in names],rotation=55,ha='right');ax.set_title('验证集完整输入 '+col);ax.grid(axis='y',alpha=.2)
    fig('01_clean_comparison')
    fig_,axes=plt.subplots(1,2,figsize=(14,5))
    for ax,col in zip(axes,['robust_f1','robust_mae']):
        ax.bar(x,comparison[col]);ax.set_xticks(x,[LABELS.get(n,n) for n in names],rotation=55,ha='right');ax.set_title('112个连续缺失场景等权均值 '+col)
    fig('02_robust_comparison')
    show=list(dict.fromkeys(['old_gated','text',candidate,selected]));fig_,axes=plt.subplots(1,2,figsize=(11,4))
    for name in show:
        sub=ens[(ens.model==name)&(ens.subset=='TAV')&(ens.primary==True)].groupby(['rate','position'])[['macro_f1','mae']].mean().groupby('rate').mean()
        for ax,col in zip(axes,['macro_f1','mae']):ax.plot(sub.index,sub[col],marker='o',label=LABELS.get(name,name));ax.set_xlabel('TAV缺失比例');ax.set_ylabel(col);ax.grid(alpha=.2)
    axes[0].legend();fig('03_missing_rate')
    chosen=ens[ens.model==candidate];primary=chosen[chosen.primary==True]
    fig_,axes=plt.subplots(1,2,figsize=(11,4))
    for subset in ['T','A','V','TA','TV','AV','TAV']:
        v=primary[primary.subset==subset].groupby(['rate','position'])[['macro_f1','mae']].mean().groupby('rate').mean()
        for ax,col in zip(axes,['macro_f1','mae']):ax.plot(v.index,v[col],marker='o',label=subset);ax.set_xlabel('缺失比例');ax.set_ylabel(col)
    axes[0].legend(ncol=2);fig('04_modality_type')
    fig_,axes=plt.subplots(1,2,figsize=(11,4))
    for ax,col in zip(axes,['macro_f1','mae']):
        v=primary[primary.subset=='TAV'].pivot_table(index='rate',columns='position',values=col).reindex(columns=['start','middle','end','random'])
        im=ax.imshow(v,aspect='auto',cmap='YlGnBu');ax.set_xticks(range(4),['开头','中间','末尾','随机']);ax.set_yticks(range(4),['10%','30%','50%','70%']);ax.set_title(candidate+' '+col)
        for i in range(4):
            for j in range(4):ax.text(j,i,f'{v.iloc[i,j]:.3f}',ha='center',va='center')
        fig_.colorbar(im,ax=ax)
    fig('05_position')
    raw=load_attachment2()['valid'];c=raw['classification_labels'].astype(int);y=raw['regression_labels']
    prediction={n:np.load(ROOT/f'results/valid_{n}_predictions.npz')['out'].mean(1)[0] for n in show}
    fig_,axes=plt.subplots(1,2,figsize=(10,4));out=prediction[candidate]
    cm=confusion_matrix(c,out[:,:3].argmax(-1),labels=[0,1,2]);axes[0].imshow(cm,cmap='Blues')
    for i in range(3):
        for j in range(3):axes[0].text(j,i,str(cm[i,j]),ha='center',va='center')
    axes[0].set_xticks(range(3),['负向','中性','正向']);axes[0].set_yticks(range(3),['负向','中性','正向']);axes[0].set_xlabel('预测');axes[0].set_ylabel('真实');axes[0].set_title(candidate+' 验证集混淆矩阵')
    axes[1].scatter(y,out[:,3],s=10,alpha=.35);axes[1].plot([-3,3],[-3,3],'k--');axes[1].set_xlabel('真实强度');axes[1].set_ylabel('预测强度');fig('06_validation_errors')
    groups=[('强负向',y<=-1.5),('弱负向',(y<0)&(y>-1.5)),('中性',y==0),('弱正向',(y>0)&(y<1.5)),('强正向',y>=1.5)]
    group_rows=[]
    for n,pr in prediction.items():
        for label,mask in groups:
            group_rows.append(dict(model=n,group=label,n=int(mask.sum()),accuracy=float((pr[mask,:3].argmax(-1)==c[mask]).mean()),
                mae=float(np.abs(pr[mask,3]-y[mask]).mean()),mean_true=float(y[mask].mean()),mean_pred=float(pr[mask,3].mean())))
    gd=pd.DataFrame(group_rows);gd.to_csv(ROOT/'results/validation_error_groups.csv',index=False)
    fig_,ax=plt.subplots(figsize=(10,4));pos=np.arange(5);w=.8/len(show)
    for j,n in enumerate(show):ax.bar(pos-.4+w*(j+.5),gd[gd.model==n].mae,w,label=LABELS.get(n,n))
    ax.set_xticks(pos,[g[0] for g in groups]);ax.set_ylabel('MAE');ax.legend();fig('07_intensity_groups')
    review=[]
    for i,sid in enumerate(raw['id']):
        row=dict(sample_id=sid,raw_text=raw['raw_text'][i],true_class=int(c[i]),true_intensity=float(y[i]))
        for n,pr in prediction.items():row[n+'_class']=int(pr[i,:3].argmax());row[n+'_intensity']=float(pr[i,3]);row[n+'_abs_error']=float(abs(pr[i,3]-y[i]))
        review.append(row)
    rd=pd.DataFrame(review);rd.to_csv(ROOT/'results/validation_prediction_review.csv',index=False)
    worst=rd.sort_values(candidate+'_abs_error',ascending=False).head(12);worst.to_csv(ROOT/'results/validation_representative_errors.csv',index=False)
    shape=chosen[(chosen.subset=='TAV')&(chosen.rate.isin([.3,.5]))&(chosen.position=='random')].copy()
    shape['shape']=np.where(shape.pattern=='multi','multi',np.where(shape['sync'],'single_sync','staggered'))
    shape=shape.groupby(['rate','shape'])[['macro_f1','mae']].mean().reset_index();shape.to_csv(ROOT/'results/shape_sync_comparison.csv',index=False)
    fig_,axes=plt.subplots(1,2,figsize=(11,4))
    for ax,col in zip(axes,['macro_f1','mae']):shape.pivot(index='rate',columns='shape',values=col).plot.bar(ax=ax);ax.set_title(candidate+' '+col)
    fig('08_shape_sync')
    fig_,axes=plt.subplots(1,2,figsize=(11,4));gradient=[];aux_rows=[]
    plot_models=list(dict.fromkeys(['gated','aux','cce',candidate]))
    for n in [name for name in names if name!='old_gated']:
        for seed in [17,29,43]:
            h=json.loads((ROOT/f'results/history_{n}_{seed}.json').read_text())
            if n in plot_models:axes[0].plot([r['epoch'] for r in h],[r['score'] for r in h],label=f'{n}/{seed}',alpha=.75)
            for r in h:
                for m,value in r['branch_grad_norm'].items():gradient.append(dict(model=n,seed=seed,epoch=r['epoch'],modality=m,grad_norm=value))
                for m,v in r['aux_validation'].items():aux_rows.append(dict(model=n,seed=seed,epoch=r['epoch'],modality=m,**v))
    pd.DataFrame(gradient).to_csv(ROOT/'results/branch_gradient_diagnostics.csv',index=False)
    pd.DataFrame(aux_rows).to_csv(ROOT/'results/unimodal_aux_diagnostics.csv',index=False)
    gr=pd.DataFrame(gradient);gr=gr[gr.model.isin(plot_models)].groupby(['model','modality']).grad_norm.mean().unstack();gr.plot.bar(ax=axes[1]);axes[0].legend(fontsize=7,ncol=2);axes[0].set_ylabel('六场景 J');axes[0].set_xlabel('全局轮次');axes[1].set_ylabel('分支梯度范数（未按参数数归一）');fig('09_training_diagnostics')
    tests=pd.concat([pd.read_csv(ROOT/f'results/test_{n}_metrics.csv') for n in lock['final_test_models']],ignore_index=True)
    tests.to_csv(ROOT/'results/test_metrics.csv',index=False);tc=tests[(tests.seed=='ensemble')&(tests.name=='clean')]
    bootstrap=json.loads((ROOT/'results/paired_cluster_bootstrap.json').read_text())
    ci_rows=[]
    for condition,values in bootstrap['results'][candidate].items():
        for metric,entry in values.items():ci_rows.append(dict(condition=condition,metric=metric,delta=entry['delta'],lower=entry['ci95'][0],upper=entry['ci95'][1]))
    ci_table=pd.DataFrame(ci_rows);ci_table.to_csv(ROOT/'results/bootstrap_intervals.csv',index=False)
    result_text='新版通过全部预注册验收条件，采用新版模型。' if lock['accepted_improvement'] else '本轮新模型未通过全部预注册验收条件，最终保留旧版门控模型。新增实验作为研究结果保存，不将其宣称为已验证的整体性能提升。'
    short=comparison[['model','J','clean_f1','clean_mae','robust_f1','robust_mae','neutral_recall','eligible']]
    fails={r['model']:[k for k,v in r['checks'].items() if not v] for r in lock['rows'] if r['model']!='old_gated'}
    stage_rows=[]
    for name in names:
        for s,path in zip([17,29,43],checkpoint_paths(name)):
            ck=torch.load(path,map_location='cpu',weights_only=True)
            stage_rows.append(dict(model=name,seed=s,selected_stage=ck.get('stage','legacy'),epoch=ck['epoch'],J=ck['score']))
    pd.DataFrame(stage_rows).to_csv(ROOT/'results/checkpoint_stages.csv',index=False)
    stage_notes=[]
    for n in ['imtd','emc']:
        kept=sum(r['model']==n and r['selected_stage']==2 for r in stage_rows)
        stage_notes.append(f'{n}有{kept}/3个种子保留第二阶段检查点')
    stage_note='；'.join(stage_notes)+'。若均保留第一阶段，它们与CCE结果相同是早停选择的结果，不代表省略了第二阶段训练；完整损失和验证曲线保存在history文件中。'
    a=comparison.set_index('model').loc[candidate];b=comparison.set_index('model').loc['old_gated']
    diagnosis=f'候选相对旧版：完整输入Macro-F1变化 {a.clean_f1-b.clean_f1:+.4f}，MAE变化 {a.clean_mae-b.clean_mae:+.4f}；连续缺失场景平均Macro-F1变化 {a.robust_f1-b.robust_f1:+.4f}，MAE变化 {a.robust_mae-b.robust_mae:+.4f}。分类F1越高越好，MAE越低越好，两者须分别判断。'
    error_diagnosis=f'候选中性召回为 {a.neutral_recall:.4f}，旧版为 {b.neutral_recall:.4f}，变化 {a.neutral_recall-b.neutral_recall:+.4f}。'
    for label in ['强负向','强正向']:
        row=gd[(gd.model==candidate)&(gd.group==label)].iloc[0];reference=gd[(gd.model=='old_gated')&(gd.group==label)].iloc[0]
        error_diagnosis+=f' {label}的真实平均强度为 {row.mean_true:.4f}，候选平均预测 {row.mean_pred:.4f}；MAE相对旧版变化 {row.mae-reference.mae:+.4f}。'
    type_scores=primary.groupby(['subset','rate','position'])[['macro_f1','mae']].mean().groupby('subset').mean()
    type_diagnosis=f'在候选模型的本次验证中，按比例和位置等权平均，Macro-F1最低的缺失组合为 {type_scores.macro_f1.idxmin()}（{type_scores.macro_f1.min():.4f}），最高为 {type_scores.macro_f1.idxmax()}（{type_scores.macro_f1.max():.4f}）。该比较是特定遮挡机制下的敏感性结果，不等于模态因果贡献。'
    examples=worst[['sample_id','raw_text','true_class','true_intensity',candidate+'_class',candidate+'_intensity']].head(4)
    ablation_names=list(json.loads((ROOT/'ablation_configs.json').read_text()))
    ablation_note=('另完成候选逐项移除实验：'+', '.join(ablation_names)+'。') if ablation_names else '最佳新候选不含五个新增EBMC模块，逐项移除清单为空；五项累加对照全部完成，不将不适用的消融宣称为额外实验。'
    report=f'''# 问题二优化实验报告

## 1. 结论

{result_text}

最终提交模型：**{selected}**；重点审查的新模型：**{candidate}**。选择仅依据验证集，完整规则与权重哈希保存在 `selection_lock.json`。附件3共30条预测，无标签，不能计算专项准确率。

## 2. 原题约束与实施范围

统一使用附件2对齐版：训练3395、验证728、测试727。文本采用同一冻结 DistilBERT；音频74维、视觉35维。只使用赛题数据学习参数。文本删除发生在编码之前，不调用原文或其他版本补齐。有效位置、原始可用性、人工删除分别记录；时长单位是有效词槽比例。分类为负向/中性/正向，中性真实强度恰为0，保留分类与强度两个预测头。

本轮在独立目录实施，未重训或替换问题三。根环境未安装、升级或重装依赖。旧版比较接口也统一采用全模态无观测时的训练先验兜底；旧权重及原文件不变。验证集4条单词槽样本可触发此边界，因此本报告旧版鲁棒指标与历史原始实现可能有微小差异；测试集没有单词槽样本。

为降低完整压力测试的时间成本，推理可使用代数等价的FP32展开卷积。每个模型先通过float64公式校验与六个验证输入的数值比较：最大差≤1e-3、集成类别无变化才启用，否则保留原cuDNN。后端选择不依据真实标签，审计见 `backend_*.json`。训练及六场景J仍采用原执行方式；不能把这项工程优化称为模型精度提升，也不宣称不同后端位级一致。

## 3. 模型与受控实验

动态连续遮挡每样本50%完整、50%遮挡，由种子、轮次、样本编号确定。7种模态组合、4种比例、4种位置均匀抽样，多模态同步/独立起点各一半。同一训练种子下所有模型共用遮挡；两个阶段最多20/15轮，早停耐心5。回归损失权重在种子17的三组门控实验后锁定为 **{json.loads((ROOT/'regression_weight_lock.json').read_text())['selected']}**。

十个主配置均使用17、29、43三个种子。新增模块依次为单模态辅助监督、共享/特有分解、局部互补增强、可信分类蒸馏、能量协调。{ablation_note}具体公式、训练权重、特殊情况见《模型定义.md》。采用全程最佳验证检查点；若第一阶段检查点更好，允许最终保留第一阶段，不能声称第二阶段模块必然生效。

这是借鉴 [EBMC](https://arxiv.org/html/2604.12518v2) 的本题适配：使用连续区间缺失、冻结统一文本接口、三分类与回归双任务，以及64维时序网络。局部增强只增强可观测位置，不声称恢复真实缺失信号；可信蒸馏只作用于分类，不使用回归教师共识约束。EMC采用范数项梯度正则的简化实现，见[官方实现](https://github.com/kangverse/EBMC/blob/main/EBMC/modules/emc.py)。

## 4. 验证集结果与模型选择

{md(short)}

{diagnosis}

robust 为112个单连续区间场景等权平均：每场景先平均随机遮挡重复，再对场景平均。额外的多短段与错开实验单独分析，不混入主验收均值。完整评价矩阵含217次输入实现。

完整输入性能：

{md(clean[['accuracy','macro_f1','weighted_f1','mae','pearson','neutral_recall']].reset_index())}

![完整输入](results/figures/01_clean_comparison.png)
![鲁棒对照](results/figures/02_robust_comparison.png)

各模型未满足的验收项：

{md(pd.DataFrame([dict(model=k,failed=', '.join(v) or '无') for k,v in fails.items()]))}

全部单种子、集成和逐场景结果在 `validation_metrics.csv`；训练种子均值/标准差在 `ablation_seed_summary.csv`。三种子标准差不是置信区间。音视频利用改善须结合辅助头性能、消融与融合收益判断；梯度范数受分支参数量影响，不能直接证明梯度竞争或因果贡献。

## 5. 缺失类型、位置、长度及形状

以下缺失规律图主要展示候选 **{candidate}**，与最终采用模型明确区分。

{type_diagnosis}

![缺失比例](results/figures/03_missing_rate.png)
![缺失类型](results/figures/04_modality_type.png)
![缺失位置](results/figures/05_position.png)
![形状与同步性](results/figures/08_shape_sync.png)

{md(shape)}

缺失率按有效词槽计算，短序列取整可能使实际比例偏离名义比例；逐场景实际删除比例见 `valid_corruption_audit.csv`。错开缺失采用独立起点，不保证区间互不重叠；多短段在边界可能连接。不能由一个位置或形状的小幅均值差推断普遍规律。

## 6. 错误归因与强度压缩

{md(gd)}

{error_diagnosis}

![混淆与强度](results/figures/06_validation_errors.png)
![分组误差](results/figures/07_intensity_groups.png)
![训练与分支诊断](results/figures/09_training_diagnostics.png)

强负向定义为 y≤−1.5，强正向为 y≥1.5；其余非零强度为弱情感。分别核对中性召回、强情感幅度、分类与回归的符号分歧，不把强度收缩预先归因于 tanh。逐样本原文、真值及新旧预测见 `validation_prediction_review.csv`；最大误差12例见 `validation_representative_errors.csv`。文本内容只用于错误审阅，不用于填补模型缺失输入。个例分析不构成模态冲突或讽刺识别的因果证据。

候选误差最大的4个验证样本如下；0/1/2分别为负向/中性/正向。

{md(examples)}

各模型最终保留的训练阶段和轮次见 `checkpoint_stages.csv`，可核查蒸馏或能量协调是否真正进入最终检查点。

{stage_note}

## 7. 不确定性与锁定后的测试复核

按源视频分组进行1000次配对 bootstrap，分别比较完整输入及连续缺失场景均值。结果见 `paired_cluster_bootstrap.json`；区间跨0不宣称稳定优势。区间以已训练集成及重复使用的验证集为条件，没有校正模型选择偏差，不能充当外部泛化保证。

候选相对旧模型的配对差值及95%区间如下，Macro-F1差值为正、MAE差值为负表示改善：

{md(ci_table)}

锁定后的测试集完整输入结果：

{md(tc[['model','accuracy','macro_f1','mae','pearson','neutral_recall']])}

测试集在旧版实验中已经使用，本轮只在方案锁定后复核一次，不宣称为完全未见测试，不根据结果继续调参。模型选择锁定哈希：`{sha256(ROOT/'selection_lock.json')}`。

## 8. 附件3与复现

最终CSV为 `results/附件3_问题二预测.csv`，列为 `sample_id,polarity,intensity`，30个编号完整且唯一。审阅文件保留概率、种子分歧、可用比例和全模态缺失标记。概率及种子分歧未经校准，不能称为可靠置信区间。

推理权重移除训练专用辅助头，保存/重载后的验证预测逐项一致。必要检查见 `verification.json` 与 `special_inference_audit.json`。三问联合交付包大小、打包目录推理验证及旧文件保护核验见 `package_audit.json`。通用冻结编码器不在50MB包内，精确版本、来源和SHA256均记录，复现需提供该固定资产。
'''
    (ROOT/'问题二优化实验报告.md').write_text(report,encoding='utf-8')
    cards=''.join(f'<figure><img src="figures/{p.name}"><figcaption>{html.escape(p.stem)}</figcaption></figure>' for p in sorted(FIG.glob('*.png')))
    page=f'''<!doctype html><html lang="zh"><meta charset="utf-8"><title>问题二优化审阅</title><style>body{{max-width:1250px;margin:32px auto;font:16px/1.65 sans-serif;background:#f5f7fa;color:#17243b}}section,figure{{background:white;padding:22px;border-radius:12px;margin:20px 0}}table{{border-collapse:collapse;width:100%;font-size:13px}}td,th{{padding:8px;border-bottom:1px solid #ddd;text-align:left}}img{{max-width:100%}}.scroll{{overflow:auto}}a{{color:#1857a4}}</style><h1>问题二优化实验审阅</h1><section><strong>{html.escape(result_text)}</strong><p>最终采用：{selected}；研究候选：{candidate}。冻结文本编码器，三种子对照，验证集选择。</p><a href="附件3_问题二预测.csv">下载30条专项预测</a> · <a href="selection_comparison.csv">验收对照</a> · <a href="paired_cluster_bootstrap.json">视频分组置信区间</a></section><section class="scroll">{short.to_html(index=False,float_format=lambda x:f'{x:.4f}')}</section>{cards}<section class="scroll"><h2>验证集最大误差案例</h2>{worst.to_html(index=False,escape=True,float_format=lambda x:f'{x:.3f}')}</section><section><p>完整报告：<a href="../问题二优化实验报告.md">问题二优化实验报告</a>。候选未达标时保留旧模型；图中候选实验不代表最终替换。门控权重不是问题三的因果归因证据。</p></section></html>'''
    (ROOT/'results/problem2_review.html').write_text(page,encoding='utf-8')
    save_json(ROOT/'results/report_summary.json',dict(selected=selected,candidate=candidate,accepted=lock['accepted_improvement'],models=len(names),figures=9,validation_scenarios=217))
    print('Report, figures, diagnostics and review page generated.',flush=True)

if __name__=='__main__':main()
