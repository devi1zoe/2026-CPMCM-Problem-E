"""Video-cluster paired bootstrap, vectorized sufficient statistics; repeats averaged before scenarios."""
from common import *
import numpy as np

def statistics(out,c,y,group_index,ngroups):
    # out [scenario, sample, 4], average training-seed predictions before metrics.
    s,n=out.shape[:2];pred=out[:,:,:3].argmax(-1)
    cm=np.zeros((s,ngroups,3,3),np.float64);errors=np.zeros((s,ngroups),np.float64)
    for k in range(s):
        np.add.at(cm[k],(group_index,c,pred[k]),1)
        np.add.at(errors[k],group_index,np.abs(out[k,:,3]-y))
    return cm,errors

def evaluate_draws(cm,errors,counts,weights):
    # counts [draw,group]; groups are sampled with replacement.
    confusion=np.einsum('bg,sgij->bsij',counts,cm,optimize=True)
    tp=confusion.diagonal(axis1=-2,axis2=-1)
    den=confusion.sum(-2)+confusion.sum(-1)
    f1=np.divide(2*tp,den,out=np.zeros_like(tp),where=den>0).mean(-1)
    n=confusion.sum((-2,-1));mae=(counts@errors.T)/n
    return np.stack([f1@weights,mae@weights],-1)

def main():
    from engine import metrics
    rng_test=np.random.default_rng(71);tc=np.array([0,0,1,1,2,2,2]);ty=np.array([-2,-.5,0,0,.4,1.5,3]);tg=np.array([0,0,1,2,2,3,3])
    fixture=rng_test.normal(size=(3,7,4));fixture[:,:,:3]=np.exp(fixture[:,:,:3]);fixture[:,:,:3]/=fixture[:,:,:3].sum(-1,keepdims=True)
    counts_test=np.array([[2,0,1,1],[0,1,1,2],[1,1,1,1]]);weights_test=np.array([.25,.25,.5])
    quick=evaluate_draws(*statistics(fixture,tc,ty,tg,4),counts_test,weights_test);explicit=[]
    for cnt in counts_test:
        ii=np.concatenate([np.tile(np.flatnonzero(tg==j),n) for j,n in enumerate(cnt)])
        values=[metrics(tc[ii],ty[ii],p[ii,:3],p[ii,3]) for p in fixture]
        explicit.append(weights_test@np.array([[r['macro_f1'],r['mae']] for r in values]))
    assert np.allclose(quick,explicit,atol=1e-12)
    save_json(ROOT/'results/bootstrap_verification.json',dict(group_multiplicity_matches_explicit_resampling=True,
        scenario_weights_match_explicit_metrics=True,max_error=float(np.abs(quick-explicit).max())))
    lock=json.loads((ROOT/'selection_lock.json').read_text());profiles=json.loads((ROOT/'results/valid_profiles.json').read_text())
    names=list(dict.fromkeys([lock['selected'],lock['audit_candidate']]))
    reference=np.load(ROOT/'results/valid_old_gated_predictions.npz');c=reference['c'];y=reference['y'];ids=reference['ids']
    _,groups=np.unique([x.split('$_$')[0] for x in ids],return_inverse=True);g=groups.max()+1
    rng=np.random.default_rng(20260926);counts=rng.multinomial(g,np.full(g,1/g),size=1000)
    original=np.ones((1,g));refstat=statistics(reference['out'].mean(1),c,y,groups,g)
    result={}
    for name in names:
        target=np.load(ROOT/f'results/valid_{name}_predictions.npz');stat=statistics(target['out'].mean(1),c,y,groups,g);result[name]={}
        for condition in ['clean','continuous_average']:
            weights=np.zeros(len(profiles))
            if condition=='clean':weights[0]=1
            else:
                scenario_names={p['name'] for p in profiles if p['primary']}
                for i,p in enumerate(profiles):
                    if p['primary']:weights[i]=1/len(scenario_names)/(3 if p['position']=='random' else 1)
            assert abs(weights.sum()-1)<1e-10
            delta=evaluate_draws(*stat,counts,weights)-evaluate_draws(*refstat,counts,weights)
            actual=(evaluate_draws(*stat,original,weights)-evaluate_draws(*refstat,original,weights))[0]
            bounds=np.quantile(delta,[.025,.975],axis=0)
            result[name][condition]={key:dict(delta=float(actual[j]),ci95=bounds[:,j].tolist()) for j,key in enumerate(['macro_f1','mae'])}
    save_json(ROOT/'results/paired_cluster_bootstrap.json',dict(reference='old_gated',split='valid',repeats=1000,source_video_groups=int(g),
        seed=20260926,scenario_weighting='mean repeats within scenario, then equal mean over 112 primary scenarios',
        limitation='Conditional on fitted ensembles and reused validation data; not a selection-adjusted or external-generalization interval',results=result))
    print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':main()
