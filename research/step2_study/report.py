"""Readable Chinese report from completed numerical evidence only."""
from common import *
from summarize import LABELS
import pandas as pd
import numpy as np
import html

def md_table(df):
    def cell(v):
        if isinstance(v,(float,np.floating)):return f'{v:.4f}'
        return str(v).replace('|','/').replace('\n',' ').replace('$','\\$')
    return '| '+' | '.join(map(str,df.columns))+' |\n| '+' | '.join(['---']*len(df.columns))+' |\n'+'\n'.join('| '+' | '.join(cell(v) for v in row)+' |' for row in df.itertuples(index=False,name=None))

def main():
    protocol=json.loads((ROOT/'protocol.json').read_text());lock=json.loads((ROOT/'selection_lock.json').read_text())
    combinations=json.loads((ROOT/'combination_lock.json').read_text());table=pd.read_csv(ROOT/'results/selection_comparison.csv')
    conditions=pd.read_csv(ROOT/'results/condition_summary.csv',dtype={'seed':str});pairs=pd.read_csv(ROOT/'results/paired_comparisons.csv')
    errors=pd.read_csv(ROOT/'results/error_types.csv');rescues=pd.read_csv(ROOT/'results/rescued_harmed.csv');cases=pd.read_csv(ROOT/'results/representative_cases.csv')
    rec=pd.read_csv(ROOT/'results/reconstruction_quality.csv')
    assert all((ROOT/f'results/evaluated_valid_{n}.json').exists() for n in protocol['configs'])
    title='问题二四组后续实验报告';selected=LABELS.get(lock['selected'],lock['selected'])
    conclusion=f'最终采用：{selected}。'+('本轮新方法没有同时通过原有七项要求，因此保留此前正式模型。' if lock['selected']=='old_gated' else '该方法通过原有七项要求，具体收益和代价如下。')
    text=[f'# {title}',conclusion,'## 本轮实际做了什么',
        '四组实验均已完成。每种设置分别训练三次；普通对照、文本调整、可靠性融合和缺失特征估计共十种设置，合计30次主要训练。可靠性融合另外完成每个种子三组留出训练及一组全训练集拟合，每组含文本、声音、视觉三个模型，共36次单模态模型拟合。中断后重跑的记录另存，不计为独立实验。',
        '训练集3395条、验证集728条。每个方法使用相同的253次评价输入：完整输入1次、原有单段缺失168次、多段缺失24次、错开缺失24次、不等长缺失36次。原有112个单段场景先平均随机重复，再等权平均；新增情况单独报告。',
        'Macro-F1综合衡量三类识别，越高越好；MAE是强度平均绝对误差，越低越好。下表为三次训练模型平均预测后的结果，三次单模型的均值和波动另见 seed_mean_std.csv，不能将二者混为一谈。']
    short=table[table.model!='expert_text'][['label','clean_f1','clean_mae','robust_f1','robust_mae','text_missing_f1']].rename(columns={
        'label':'方法','clean_f1':'完整F1','clean_mae':'完整MAE','robust_f1':'单段缺失F1','robust_mae':'单段缺失MAE','text_missing_f1':'含文本缺失F1'})
    text.append(md_table(short));sections=[];decisions=[]
    descriptions={
        '1':('缺失形状','只改变训练时删除内容的形状，模型和其他训练设置相同。单段与多段的删除总数相同；不等长版本让各模态独立抽取10%、30%、50%、70%的比例。','比较多段训练、不等长训练相对单段训练的变化。'),
        '2':('文本理解','在原有语言模型最后两层加入少量可学习参数，共24,576个；主体保持不变。分别比较纯文本和三模态情感预测。所有文本均先删除缺失词元，再计算表示。','分别查看文本本身是否改善，以及改善能否传递到多模态预测。'),
        '3':('可靠性融合','三个对照共享相同的文本、声音、视觉预测模型。先只看缺失程度，再加入预测倾向、把握及模态分歧，最后允许不超过1分的强度修正。融合规则只用训练内部留出样本的预测来学习，标准化也仅使用每次拟合的样本。','检查修正了多少文本错误，同时又破坏了多少文本正确判断。模型自报的把握并不预先视为可靠。'),
        '4':('缺失特征估计','用剩余内容估计缺失位置的64维特征。只在原本有观测、后被人工删除的位置学习重建；补出值始终与实际观测分开。没有任何观测时仍使用训练集先验。','重建误差降低并不足以保留方法，还必须改善情感类别和强度预测。')}
    for group,names in protocol['groups'].items():
        heading,method,question=descriptions[group];text.extend([f'## 第{group}组：{heading}',method,question])
        subset=short[short['方法'].isin([LABELS[n] for n in names])];text.append(md_table(subset))
        pp=pairs[(pairs.group.astype(str)==group)&(pairs.condition=='单段缺失')]
        decision_rows=[]
        main_pairs=[('shape_multi','shape_single'),('shape_unequal','shape_single')] if group=='1' else ([('text_lora','text_frozen'),('multi_lora','shape_single')] if group=='2' else ([('gate_reliability','gate_quality'),('gate_correction','gate_reliability')] if group=='3' else [('reconstruct','shape_single')]))
        for model,reference in main_pairs:
            pair=pp[(pp.model==model)&(pp.reference==reference)].set_index('metric');f=pair.loc['macro_f1'];m=pair.loc['mae']
            both=f['delta']>0 and m['delta']<0;clear=bool(f.clear_direction and m.clear_direction)
            decision='支持保留这一改动' if both and clear else ('有改善迹象，但证据不足' if both else '不支持作为整体改进保留')
            sentence=f'{LABELS[model]}相对{LABELS[reference]}：缺失F1变化{f["delta"]:+.4f}，95%区间[{f.lower:+.4f}, {f.upper:+.4f}]；MAE变化{m["delta"]:+.4f}，区间[{m.lower:+.4f}, {m.upper:+.4f}]。三次独立训练中，F1改善{int(f.seeds_improved)}次、MAE改善{int(m.seeds_improved)}次。判断：{decision}。'
            text.append(sentence);decisions.append(dict(group=group,model=model,reference=reference,decision=decision,point_estimates_improve_both=bool(both),clear_both=clear))
            decision_rows.append(sentence)
        for model,reference in main_pairs:
            grouped=errors[(errors.model==model)&(errors.name=='clean')&errors.group.isin(['中性','强负向','强正向'])]
            worst=grouped.loc[grouped.mae.idxmax()];neutral=grouped[grouped.group=='中性'].iloc[0]
            text.append(f'失败类型检查：{LABELS[model]}在中性、强负向、强正向三组中，强度误差最大的是{worst.group}（MAE={worst.mae:.4f}）；中性识别率为{neutral.accuracy:.4f}。评分、比较和转折表达的完整分组结果见 error_types.csv。')
        if group=='1':
            extra=conditions[(conditions.seed=='ensemble')&conditions.model.isin(names)&conditions.condition.isin(['多段缺失','错开缺失','不等长缺失'])][['label','condition','macro_f1','mae']]
            text.append(md_table(extra.rename(columns={'label':'方法','condition':'缺失情况','macro_f1':'F1','mae':'MAE'})))
        if group=='3':
            rescue=rescues[(rescues.model.isin(names))&(rescues.name=='clean')&(rescues.group=='全部')&(rescues.scope=='全部样本')][['model','text_wrong','rescued','text_correct','harmed']].copy()
            rescue['model']=rescue.model.map(LABELS);text.append(md_table(rescue.rename(columns={'model':'方法','text_wrong':'文本判断错误','rescued':'融合后改对','text_correct':'文本判断正确','harmed':'融合后改错'})))
            strength=table[table.model.isin(names)][['label','neutral_recall','strong_negative_mae','strong_positive_mae']]
            text.append(md_table(strength.rename(columns={'label':'方法','neutral_recall':'中性识别率','strong_negative_mae':'强负向MAE','strong_positive_mae':'强正向MAE'})))
        if group=='4':
            quality=rec.groupby(['modality','name'])[['reconstruction_mse','training_mean_mse']].mean().groupby('modality').mean().reset_index()
            text.append('下表比较缺失特征的估计误差与直接使用训练集平均特征的误差。每种模态先按同名场景平均重复和训练种子，再平均场景。数值越低越接近本实验定义的目标表示，并不等于恢复了真实语音或画面。')
            text.append(md_table(quality.rename(columns={'modality':'模态','reconstruction_mse':'估计特征误差','training_mean_mse':'训练均值误差'})))
        # Each group has concrete reviewable success and failure examples.
        model_names={m for m,r in main_pairs};sample=cases[(cases.model.isin(model_names))&(cases.scenario=='clean')]
        fail=sample[sample.kind.isin(['类别改错','强度误差增大'])].head(2);success=sample[sample.kind.isin(['类别改对','强度误差减小'])].head(2)
        chosen=pd.concat([success,fail]).copy()
        if len(chosen):
            polarity={0:'负向',1:'中性',2:'正向'}
            chosen['classes']=[f'{polarity[int(t)]} / {polarity[int(a)]} / {polarity[int(b)]}' for t,a,b in zip(chosen.true_class,chosen.reference_class,chosen.new_class)]
            chosen['model']=chosen.model.map(LABELS)
            text.extend(['下面列出实际变化案例。“极性”一列依次给出真实类别、对照预测和新方法预测。分类与强度由不同输出分别预测，因此强度符号不一定与预测类别一致。原文仅用于事后检查，没有用于补齐模型输入。',md_table(chosen[['model','sample_id','kind','classes','true_intensity','reference_intensity','new_intensity','original_text_for_review']].rename(columns={
                'model':'方法','sample_id':'样本','kind':'变化','classes':'极性（真实/对照/新）','true_intensity':'真实强度','reference_intensity':'对照强度','new_intensity':'新方法强度','original_text_for_review':'原文'}))])
        section='<section><h2>第'+group+'组：'+html.escape(heading)+'</h2><p>'+html.escape(method)+'</p>'+subset.to_html(index=False,float_format=lambda x:f'{x:.4f}')+''.join('<p>'+html.escape(s)+'</p>' for s in decision_rows)+'</section>'
        sections.append(section)
    text.extend(['## 是否组合、是否替换',
        ('两个单项在规定的缺失F1和MAE上均改善，因此按事先限定的范围补做了一组组合实验；其结果也列入总表。' if combinations['run'] else '没有一对预先限定的“缺失形状调整＋文本调整”同时满足单项F1和MAE均改善，因此没有继续叠加。'),
        '正式替换仍要求七项同时满足：完整F1下降≤0.005、完整MAE增加≤0.01、原有缺失平均F1提高≥0.01、缺失MAE不增加、中性识别下降≤0.01、强正与强负MAE分别增加≤0.02。没有因结果不理想而降低要求。',conclusion])
    for name in lock['final_test_models']:
        test=pd.read_csv(ROOT/f'results/test_{name}_metrics.csv',dtype={'seed':str});clean=test[(test.seed=='ensemble')&(test.name=='clean')].iloc[0]
        text.append(f'选定方案后的历史测试复核：{LABELS.get(name,name)}，完整输入F1={clean.macro_f1:.4f}，MAE={clean.mae:.4f}，准确率={clean.accuracy:.4f}，相关系数={clean.pearson:.4f}。')
    text.extend(['## 结论的适用范围','验证集已经被多轮研究使用；按源视频分组计算的区间没有消除模型选择带来的偏差。既有测试集也曾被前几轮查看，本轮复核不能当作全新的独立泛化证明。以上结论只适用于本轮具体设置，不证明某类方法在所有参数、数据规模和缺失机制下均无效。',
        '缺失比例是有效序列位置比例，不是秒。等长指选定的删除位置数量相同；其中可能包含原本就没有观测的位置，实际损失的观测数另外记录。多段在短样本上可能退回单段，相邻区间也可能连接。不等长和多段结果没有混入原有验收平均。中性真实强度严格为0，强负向≤−1.5，强正向≥1.5。比较、评分和转折仅按可复核文本线索分组，不能据此声称识别了讽刺或模态冲突。',
        '附件3的30条预测已保存。附件没有真实标签，不能给出该附件的准确率。现有四个实验目录及问题三材料均保持原样。',
        '## 可查阅材料','- results/selection_comparison.csv：正式验收及整体指标。\n- results/condition_summary.csv：完整、各类缺失及文本缺失的分别结果。\n- results/seed_mean_std.csv：三次训练的均值与波动。\n- results/paired_comparisons.csv：按源视频分组的配对区间。\n- results/rescued_harmed.csv：改对与改错数量。\n- results/error_types.csv、representative_cases.csv：失败类型及实例。\n- results/reconstruction_quality.csv：缺失特征估计误差。\n- results/附件3_问题二预测.csv：最终30条预测。\n- 代码说明.md：运行与数值修正记录。'])
    (ROOT/'实验报告.md').write_text('\n\n'.join(text),encoding='utf-8');save_json(ROOT/'results/group_decisions.json',decisions)
    make_figures(table,pairs,conditions)
    page='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>'+title+'</title><style>body{font:16px/1.7 sans-serif;max-width:1280px;margin:32px auto;padding:0 24px;background:#f4f6fa;color:#17283d}section{background:white;padding:22px;margin:22px 0;border-radius:10px}table{border-collapse:collapse;width:100%;font-size:13px}th,td{padding:8px;border-bottom:1px solid #ddd;text-align:left}img{max-width:100%}a{color:#185e9e}</style><h1>'+title+'</h1><section><b>'+html.escape(conclusion)+'</b><p>四组实验 · 三次独立训练 · 253次固定评价输入</p>'+short.to_html(index=False,float_format=lambda x:f'{x:.4f}')+'</section>'+''.join(sections)+'<section><h2>结果图</h2><img src="figures/overall.png"><img src="figures/paired_changes.png"><img src="figures/missing_conditions.png"></section><section><a href="../实验报告.md">完整报告</a> · <a href="representative_cases.csv">实际案例</a> · <a href="附件3_问题二预测.csv">30条最终预测</a><p>历史测试已在前几轮使用；本次区间不消除反复选模型造成的偏差。</p></section></html>'
    (ROOT/'results/实验审阅.html').write_text(page,encoding='utf-8');print('Chinese report and figures saved')

def make_figures(table,pairs,conditions):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    font=Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    if font.exists():font_manager.fontManager.addfont(str(font));plt.rcParams['font.family']=font_manager.FontProperties(fname=str(font)).get_name()
    plt.rcParams.update({'axes.unicode_minus':False,'font.size':10,'text.parse_math':False})
    out=ROOT/'results/figures';out.mkdir(exist_ok=True)
    def save(name):
        plt.tight_layout();plt.savefig(out/(name+'.png'),dpi=160,bbox_inches='tight');plt.savefig(out/(name+'.pdf'),bbox_inches='tight');plt.close()
    frame=table[table.model!='expert_text'];fig,axes=plt.subplots(1,2,figsize=(15,7));yy=np.arange(len(frame))
    axes[0].barh(yy-.18,frame.clean_f1,height=.34,label='完整输入');axes[0].barh(yy+.18,frame.robust_f1,height=.34,label='单段缺失')
    axes[1].barh(yy-.18,frame.clean_mae,height=.34,label='完整输入');axes[1].barh(yy+.18,frame.robust_mae,height=.34,label='单段缺失')
    for ax in axes:ax.set_yticks(yy,frame.label);ax.invert_yaxis();ax.legend();ax.grid(axis='x',alpha=.2)
    axes[0].set_xlabel('Macro-F1（越高越好）');axes[1].set_xlabel('MAE（越低越好）');save('overall')
    selected=pairs[pairs.condition=='单段缺失'];fig,axes=plt.subplots(1,2,figsize=(17,8))
    for ax,metric in zip(axes,['macro_f1','mae']):
        f=selected[selected.metric==metric];labels=[LABELS[m]+' − '+LABELS[r] for m,r in zip(f.model,f.reference)];y=np.arange(len(f))
        ax.hlines(y,f.lower,f.upper,color='#376d94');ax.scatter(f.delta,y,color='#b9532a',zorder=3);ax.axvline(0,color='gray',linestyle='--');ax.set_yticks(y,labels);ax.invert_yaxis();ax.grid(axis='x',alpha=.2)
        ax.set_xlabel('F1变化（正值较好）' if metric=='macro_f1' else 'MAE变化（负值较好）')
    fig.suptitle('按源视频分组重复抽样的95%区间；区间跨零时不能确认稳定优势');save('paired_changes')
    f=conditions[(conditions.seed=='ensemble')&conditions.model.isin(['shape_single','shape_multi','shape_unequal'])];fig,axes=plt.subplots(1,2,figsize=(14,5))
    for name,g in f.groupby('model',sort=False):
        axes[0].plot(g.condition,g.macro_f1,'o-',label=LABELS[name]);axes[1].plot(g.condition,g.mae,'o-',label=LABELS[name])
    for ax in axes:ax.tick_params(axis='x',rotation=25);ax.legend();ax.grid(alpha=.2)
    axes[0].set_ylabel('Macro-F1');axes[1].set_ylabel('MAE');save('missing_conditions')

if __name__=='__main__':main()
