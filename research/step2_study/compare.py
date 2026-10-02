"""Paired video-group uncertainty and preregistered independent comparisons."""
from common import *
import numpy as np
import pandas as pd
from summarize import LABELS

PAIRS=[('1','shape_multi','shape_single'),('1','shape_unequal','shape_single'),('1','shape_unequal','shape_multi'),
    ('2','text_lora','text_frozen'),('2','multi_lora','shape_single'),('2','multi_lora','text_lora'),
    ('3','gate_reliability','gate_quality'),('3','gate_correction','gate_reliability'),('3','gate_correction','gate_quality'),
    ('4','reconstruct','shape_single')]

def condition_weights(profiles):
    import collections
    masks={'完整输入':[p['name']=='clean' for p in profiles],'单段缺失':[p['primary'] for p in profiles],
        '含文本的单段缺失':[p['primary'] and 'T' in p['subset'] for p in profiles],
        '多段缺失':[p['pattern']=='multi' for p in profiles],'错开缺失':[p['name'].endswith('_staggered') for p in profiles],
        '不等长缺失':[p['pattern']=='unequal' for p in profiles]}
    weights={}
    for condition,mask in masks.items():
        names=[p['name'] for p,keep in zip(profiles,mask) if keep];counts=collections.Counter(names)
        weights[condition]=np.array([1/(len(counts)*counts[p['name']]) if keep else 0 for p,keep in zip(profiles,mask)])
    return weights

def bootstrap(name,profiles,weights,rng_weights=None):
    path=ROOT/f'results/valid_{name}_predictions.npz';data=dict(np.load(path));out=data['out'].mean(1);c=data['c'];y=data['y']
    groups,gi=np.unique([s.split('$_$')[0] for s in data['ids']],return_inverse=True);g=len(groups)
    protocol=json.loads((ROOT/'protocol.json').read_text())
    stamp=sha256(path);cache=ROOT/f'cache/bootstrap_{name}_{stamp[:16]}.npz'
    if cache.exists():return dict(np.load(cache))
    if rng_weights is None:rng_weights=np.random.default_rng(protocol['bootstrap_seed']).multinomial(g,np.ones(g)/g,size=protocol['bootstrap_repeats']).astype(np.float32)
    cls=out[:,:,:3].argmax(-1);counts=[];error=[]
    for k in range(len(out)):
        counts.append(np.bincount(gi*9+c*3+cls[k],minlength=g*9).reshape(g,9))
        error.append(np.bincount(gi,weights=np.abs(out[k,:,3]-y),minlength=g))
    counts=np.asarray(counts,np.float32);error=np.asarray(error,np.float32)
    conf=(rng_weights@counts.transpose(1,0,2).reshape(g,-1)).reshape(len(rng_weights),len(out),3,3)
    diag=np.diagonal(conf,axis1=2,axis2=3);denom=conf.sum(2)+conf.sum(3)
    f1=np.divide(2*diag,denom,out=np.zeros_like(diag),where=denom>0).mean(-1)
    sample_counts=np.bincount(gi,minlength=g);mae=(rng_weights@error.T)/(rng_weights@sample_counts)[:,None]
    result={}
    for condition,w in weights.items():result[condition+'_f1']=f1@w;result[condition+'_mae']=mae@w
    np.savez_compressed(cache,**result);return result

def main():
    protocol=json.loads((ROOT/'protocol.json').read_text());profiles=json.loads((ROOT/'results/valid_profiles.json').read_text());weights=condition_weights(profiles)
    summary=pd.read_csv(ROOT/'results/condition_summary.csv',dtype={'seed':str});selection=pd.read_csv(ROOT/'results/selection_comparison.csv')
    available=set(summary.model);samples={};rows=[]
    for group,model,reference in PAIRS:
        if model not in available or reference not in available:continue
        for name in [model,reference]:
            if name not in samples:samples[name]=bootstrap(name,profiles,weights)
        for condition in weights:
            m=summary[(summary.model==model)&(summary.condition==condition)].set_index('seed')
            r=summary[(summary.model==reference)&(summary.condition==condition)].set_index('seed')
            for metric,short in [('macro_f1','f1'),('mae','mae')]:
                delta=float(m.loc['ensemble',metric]-r.loc['ensemble',metric]);distribution=samples[model][condition+'_'+short]-samples[reference][condition+'_'+short]
                lo,hi=np.quantile(distribution,[.025,.975]);seed_delta=m.loc[['17','29','43'],metric]-r.loc[['17','29','43'],metric]
                wins=int(((seed_delta>0) if metric=='macro_f1' else (seed_delta<0)).sum())
                rows.append(dict(group=group,model=model,reference=reference,condition=condition,metric=metric,delta=delta,lower=float(lo),upper=float(hi),seeds_improved=wins,
                    clear_direction=bool(lo>0 if metric=='macro_f1' else hi<0)))
    pd.DataFrame(rows).to_csv(ROOT/'results/paired_comparisons.csv',index=False)
    required={'shape_single','shape_multi','shape_unequal','text_frozen','text_lora','multi_lora'}
    if required<=available:
        indexed=selection.set_index('model');rules=json.loads((ROOT/'selection_rules.json').read_text());qualifying=[]
        for shape,text in rules['combination_candidates']:
            ref=indexed.loc['shape_single'];a=indexed.loc[shape];b=indexed.loc[text]
            if a.robust_f1>ref.robust_f1 and a.robust_mae<ref.robust_mae and b.robust_f1>ref.robust_f1 and b.robust_mae<ref.robust_mae:qualifying.append(shape)
        chosen=min(qualifying,key=lambda n:indexed.loc[n,'J']) if qualifying else None
        decision=dict(run=chosen is not None,shape=chosen,config=dict(lora=True),components=[chosen,'multi_lora'] if chosen else [],
            reason='Both independent components improved F1 and MAE' if chosen else 'No preregistered pair improved both primary robust outcomes independently',
            protocol_sha256=sha256(ROOT/'protocol.json'),rules_sha256=sha256(ROOT/'selection_rules.json'),
            evidence={n:sha256(ROOT/f'results/valid_{n}_predictions.npz') for n in sorted(required)})
        path=ROOT/'combination_lock.json'
        if path.exists():assert json.loads(path.read_text())==decision
        else:save_json(path,decision)
        print('Combination decision:',decision['run'],decision['shape'],flush=True)
    print('Paired comparisons saved:',len(rows),flush=True)

if __name__=='__main__':main()
