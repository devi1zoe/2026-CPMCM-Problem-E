# 本程序及代码是在人工智能工具辅助下完成的。

# 人工智能工具名称：GPT-6 Astra

# 版本/型号：gpt-6-astra

# 开发机构/公司：OpenAI

# 版本颁布日期：2026年9月3日

"""锁定后的留出测试压力实验。不会训练、调参或更改 selection_lock.json。"""
from common import *
from data import *
from train import MAIN,load_model,predict,metrics,load_cache
import argparse
import pandas as pd
from sklearn.metrics import f1_score


def scenarios():
    yield dict(name='clean',subset='',rate=0.,position='none',pattern='single',sync=True,repeat=0)
    for subset in ['T','A','V','TA','TV','AV','TAV']:
        for rate in [.1,.3,.5,.7]:
            for pos in ['start','middle','end','random']:
                for rep in range(3) if pos=='random' else [0]:
                    yield dict(name=f'{subset}_r{int(rate*100)}_{pos}',subset=subset,rate=rate,position=pos,pattern='single',sync=True,repeat=rep)
    for subset in ['T','A','V','TAV']:
        for rate in [.3,.5]:
            for rep in range(3):
                yield dict(name=f'{subset}_r{int(rate*100)}_multi',subset=subset,rate=rate,position='random',pattern='multi',sync=True,repeat=rep)
    for subset in ['TA','TV','AV','TAV']:
        for rate in [.3,.5]:
            for rep in range(3):
                yield dict(name=f'{subset}_r{int(rate*100)}_staggered',subset=subset,rate=rate,position='random',pattern='single',sync=False,repeat=rep)


def bootstrap_compare(target,reference,ids,c,y,repeats=500):
    """源视频聚类配对 bootstrap；多个 clip 不冒充独立视频。"""
    groups={}
    for i,id_ in enumerate(ids):
        groups.setdefault(id_.split('$_$')[0],[]).append(i)
    groups=list(groups.values());rng=np.random.default_rng(20260925);draws=[]
    for _ in range(repeats):
        ii=np.concatenate([groups[g] for g in rng.integers(len(groups),size=len(groups))])
        a=metrics(c[ii],y[ii],target[0][ii],target[1][ii]);b=metrics(c[ii],y[ii],reference[0][ii],reference[1][ii])
        draws.append([a[k]-b[k] for k in ['accuracy','macro_f1','mae']])
    lo,hi=np.quantile(draws,[.025,.975],axis=0)
    a=metrics(c,y,*target);b=metrics(c,y,*reference)
    return {k:dict(delta=a[k]-b[k],ci95=[float(lo[j]),float(hi[j])]) for j,k in enumerate(['accuracy','macro_f1','mae'])}


def main():
    torch.set_num_threads(4)
    lock=json.loads((ROOT/'selection_lock.json').read_text())
    protocol=json.loads((ROOT/'protocol.json').read_text())
    assert lock['protocol_sha256']==sha256(ROOT/'protocol.json')
    raw=load_attachment2();d=raw['test'];b=adapt(d)
    c=d['classification_labels'].astype(int);y=d['regression_labels']
    stats=dict(np.load(ROOT/'checkpoints/normalization.npz'))
    encoder=TextEncoder();clean_text=encoder.encode(b,b['o'])
    models={}
    for variant in protocol['models']:
        models[variant]=[load_model(ROOT/f'checkpoints/{variant}_{s}.pt')[0] for s in (protocol['seeds'] if variant in MAIN else [17])]
    # 原题明确要求验证集可视化与错误归因，独立保存验证预测，不能拿测试图代替。
    valid_rows=[]
    vd=raw['valid']
    for j in [0,5]:
        vv=load_cache('valid',j)
        vp=[predict(model,vv) for model in models[lock['selected']]]
        vprob=np.mean([p[0] for p in vp],0);vpred=np.mean([p[1] for p in vp],0)
        name='clean' if j==0 else 'TAV_r50_random'
        if j==0:
            np.savez(ROOT/'results/predictions_valid_clean.npz',prob=vprob,pred=vpred,
                y=vd['regression_labels'],c=vd['classification_labels'].astype(int))
        for i,id_ in enumerate(vd['id']):
            valid_rows.append(dict(scenario=name,sample_id=id_,raw_text=vd['raw_text'][i],
                true_class=int(vd['classification_labels'][i]),true_intensity=float(vd['regression_labels'][i]),
                predicted_class=int(vprob[i].argmax()),predicted_intensity=float(vpred[i]),
                abs_error=float(abs(vpred[i]-vd['regression_labels'][i])),confidence=float(vprob[i].max())))
    pd.DataFrame(valid_rows).to_csv(ROOT/'results/validation_prediction_review.csv',index=False)
    majority=int(np.bincount(raw['train']['classification_labels'].astype(int)).argmax())
    null_prob=np.eye(3)[np.full(len(y),majority)]
    null_pred=np.full(len(y),raw['train']['regression_labels'].mean())
    rows=[];audit=[];ci={};all_profiles=list(scenarios())
    detailed=[]
    save_json(ROOT/'results/test_protocol.json',dict(selection_sha256=sha256(ROOT/'selection_lock.json'),scenarios=all_profiles,
        deterministic_positions_repeats=1,random_positions_repeats=3,uncertainty='seed variation and corruption repeat variation kept separate'))
    for si,profile in enumerate(all_profiles):
        seed=31000+profile['repeat']
        view=make_view(b,encoder,stats,profile,seed,clean_text)
        lengths=np.maximum(b['p'].sum(-1),1)
        audit.append(dict(**profile,**{f'actual_deleted_fraction_{m}':float(np.mean(view['removed'][:,j].sum(-1)/lengths)) for j,m in enumerate('TAV')}))
        tv=torch_view(view);predictions={}
        for variant,ensemble in models.items():
            ps=[predict(model,tv) for model in ensemble]
            for s,(prob,pred,_) in zip(protocol['seeds'] if variant in MAIN else [17],ps):
                rows.append(dict(model=variant,seed=str(s),**profile,**metrics(c,y,prob,pred)))
            prob,pred,gates=[np.mean([p[j] for p in ps],axis=0) for j in range(3)]
            predictions[variant]=(prob,pred)
            rows.append(dict(model=variant,seed='ensemble',**profile,**metrics(c,y,prob,pred),
                **{f'gate_{m}':float(gates[:,j].mean()) for j,m in enumerate('TAV')}))
        rows.append(dict(model='majority_mean',seed='ensemble',**profile,**metrics(c,y,null_prob,null_pred)))
        if profile['name']=='clean' or (profile['name']=='TAV_r50_random' and profile['repeat']==0):
            key=profile['name']
            chosen=predictions[lock['selected']]
            ci[key]=bootstrap_compare(chosen,predictions['fixed_clean'],d['id'],c,y)
            np.savez(ROOT/f'results/predictions_{key}.npz',prob=chosen[0],pred=chosen[1],reference_prob=predictions['fixed_clean'][0],reference_pred=predictions['fixed_clean'][1],y=y,c=c)
            for i,id_ in enumerate(d['id']):
                detailed.append(dict(scenario=key,sample_id=id_,raw_text=d['raw_text'][i],true_class=int(c[i]),true_intensity=float(y[i]),
                    predicted_class=int(chosen[0][i].argmax()),predicted_intensity=float(chosen[1][i]),
                    abs_error=float(abs(chosen[1][i]-y[i])),confidence=float(chosen[0][i].max()),
                    source_video=id_.split('$_$')[0]))
        print(f'test {si+1}/{len(all_profiles)} {profile["name"]} rep={profile["repeat"]}',flush=True)
        if si%20==0:
            pd.DataFrame(rows).to_csv(ROOT/'results/test_metrics.partial.csv',index=False)
        del tv,view
    pd.DataFrame(rows).to_csv(ROOT/'results/test_metrics.csv',index=False)
    pd.DataFrame(audit).to_csv(ROOT/'results/corruption_audit.csv',index=False)
    pd.DataFrame(detailed).to_csv(ROOT/'results/test_prediction_review.csv',index=False)
    save_json(ROOT/'results/paired_cluster_bootstrap.json',dict(reference='fixed_clean',selected=lock['selected'],repeats=500,results=ci))
    partial=ROOT/'results/test_metrics.partial.csv'
    if partial.exists():partial.unlink()
    print('Held-out evaluation finished. Selection lock unchanged.',flush=True)


if __name__=='__main__':
    main()
