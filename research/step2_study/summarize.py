"""Summaries of actual completed experiments; never invent unfinished outcomes."""
from common import *
import numpy as np
import pandas as pd

LABELS={'old_gated':'当前正式模型','shape_single':'单段缺失训练','shape_multi':'单段与多段混合训练','shape_unequal':'再加入不等长缺失',
    'text_frozen':'只用文本，不调整语言模型','text_lora':'只用文本，少量调整语言模型','multi_lora':'三模态，少量调整语言模型',
    'gate_quality':'只看缺失情况的融合','gate_reliability':'增加预测可靠性线索','gate_correction':'再增加强度修正','reconstruct':'估计缺失位置特征','expert_text':'融合实验中的文本预测模型'}
METRICS=['accuracy','macro_f1','mae','pearson','neutral_recall','strong_negative_mae','strong_positive_mae']

def summarize_frame(frame):
    rows=[]
    for seed,g in frame.groupby('seed',sort=False):
        for condition,subset in [('完整输入',g[g.name=='clean']),('单段缺失',g[g.primary==True]),
            ('含文本的单段缺失',g[(g.primary==True)&g.subset.fillna('').str.contains('T')]),
            ('多段缺失',g[g.pattern=='multi']),('错开缺失',g[g.name.str.endswith('_staggered')]),('不等长缺失',g[g.pattern=='unequal'])]:
            if subset.empty:continue
            values=subset.groupby('name')[METRICS].mean().mean()
            rows.append(dict(seed=str(seed),condition=condition,**{k:float(v) for k,v in values.items()}))
    return rows

def main():
    protocol=json.loads((ROOT/'protocol.json').read_text());all_rows=[];selection=[]
    for name in ['old_gated',*protocol['configs'],'expert_text','combined']:
        marker=ROOT/f'results/evaluated_valid_{name}.json'
        if not marker.exists():continue
        frame=pd.read_csv(ROOT/f'results/valid_{name}_metrics.csv',dtype={'seed':str})
        rows=summarize_frame(frame);all_rows.extend([dict(model=name,label=LABELS.get(name,name),**r) for r in rows])
        ensemble={r['condition']:r for r in rows if r['seed']=='ensemble'}
        if '完整输入' not in ensemble or '单段缺失' not in ensemble:continue
        clean=ensemble['完整输入'];robust=ensemble['单段缺失'];six=json.loads((ROOT/f'results/six_{name}.json').read_text())
        selection.append(dict(model=name,label=LABELS.get(name,name),J=six['score'],clean_f1=clean['macro_f1'],clean_mae=clean['mae'],
            robust_f1=robust['macro_f1'],robust_mae=robust['mae'],text_missing_f1=ensemble['含文本的单段缺失']['macro_f1'],
            neutral_recall=clean['neutral_recall'],strong_negative_mae=clean['strong_negative_mae'],strong_positive_mae=clean['strong_positive_mae']))
    if not all_rows:print('No completed full evaluation yet');return
    pd.DataFrame(all_rows).to_csv(ROOT/'results/condition_summary.csv',index=False)
    sdf=pd.DataFrame(selection);reference=sdf[sdf.model=='old_gated']
    if not reference.empty:
        r=reference.iloc[0];g=protocol['guardrails']
        sdf['pass_clean_f1']=sdf.clean_f1>=r.clean_f1-g['clean_f1_drop']
        sdf['pass_clean_mae']=sdf.clean_mae<=r.clean_mae+g['clean_mae_rise']
        sdf['pass_robust_f1']=sdf.robust_f1>=r.robust_f1+g['robust_f1_gain']
        sdf['pass_robust_mae']=sdf.robust_mae<=r.robust_mae+g['robust_mae_rise']
        sdf['pass_neutral']=sdf.neutral_recall>=r.neutral_recall-g['neutral_recall_drop']
        for group in ['strong_negative','strong_positive']:sdf['pass_'+group]=sdf[group+'_mae']<=r[group+'_mae']+g['strong_mae_rise']
        sdf['eligible']=sdf[[c for c in sdf if c.startswith('pass_')]].all(axis=1)
    sdf.to_csv(ROOT/'results/selection_comparison.csv',index=False)
    seed_frame=pd.DataFrame(all_rows);nonensemble=seed_frame[seed_frame.seed!='ensemble']
    stats=nonensemble.groupby(['model','condition'])[METRICS].agg(['mean','std']);stats.to_csv(ROOT/'results/seed_mean_std.csv')
    print(sdf[['model','clean_f1','clean_mae','robust_f1','robust_mae']].to_string(index=False))

if __name__=='__main__':main()
