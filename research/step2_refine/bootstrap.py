from common import *
from bootstrap_core import statistics,evaluate_draws
from engine import metrics
import numpy as np


def main():
    rng=np.random.default_rng(71);c=np.array([0,0,1,1,2,2,2]);y=np.array([-2,-.5,0,0,.4,1.5,3]);g=np.array([0,0,1,2,2,3,3])
    fixture=rng.normal(size=(3,7,4));counts=np.array([[2,0,1,1],[0,1,1,2]]);w=np.array([.25,.25,.5])
    fast=evaluate_draws(*statistics(fixture,c,y,g,4),counts,w);explicit=[]
    for cnt in counts:
        ii=np.concatenate([np.tile(np.flatnonzero(g==j),n) for j,n in enumerate(cnt)])
        met=[metrics(c[ii],y[ii],p[ii,:3],p[ii,3]) for p in fixture];explicit.append(w@np.array([[m['macro_f1'],m['mae']] for m in met]))
    assert np.allclose(fast,explicit,atol=1e-12)
    profiles=json.loads((ROOT/'results/valid_profiles.json').read_text());d=np.load(ROOT/'results/valid_gated_predictions.npz')
    c=d['c'];y=d['y'];ids=d['ids'];_,groups=np.unique([s.split('$_$')[0] for s in ids],return_inverse=True);ng=int(groups.max()+1)
    rng=np.random.default_rng(20260926);draws=rng.multinomial(ng,np.full(ng,1/ng),size=1000);original=np.ones((1,ng));results={}
    weights={}
    for condition in ['clean','continuous_average']:
        w=np.zeros(len(profiles))
        if condition=='clean':w[0]=1
        else:
            names={p['name'] for p in profiles if p['primary']}
            for i,p in enumerate(profiles):
                if p['primary']:w[i]=1/len(names)/(3 if p['position']=='random' else 1)
        assert abs(w.sum()-1)<1e-10;weights[condition]=w
    for reference in ['old_gated','gated']:
        d=np.load(ROOT/f'results/valid_{reference}_predictions.npz');assert np.array_equal(d['ids'],ids);ref=statistics(d['out'].mean(1),c,y,groups,ng);results[reference]={}
        for name in ['dual_gate','late_fusion']:
            d=np.load(ROOT/f'results/valid_{name}_predictions.npz');assert np.array_equal(d['ids'],ids) and np.array_equal(d['c'],c) and np.array_equal(d['y'],y)
            target=statistics(d['out'].mean(1),c,y,groups,ng);results[reference][name]={}
            for condition,w in weights.items():
                delta=evaluate_draws(*target,draws,w)-evaluate_draws(*ref,draws,w)
                actual=(evaluate_draws(*target,original,w)-evaluate_draws(*ref,original,w))[0];bounds=np.quantile(delta,[.025,.975],axis=0)
                results[reference][name][condition]={k:dict(delta=float(actual[j]),ci95=bounds[:,j].tolist()) for j,k in enumerate(['macro_f1','mae'])}
    save_json(ROOT/'results/paired_cluster_bootstrap.json',dict(repeats=1000,source_video_groups=ng,seed=20260926,results=results,
        explicit_resampling_equivalence_verified=True,limitation='Conditional on fitted ensembles and reused validation; not selection-adjusted or independent generalization evidence'))
    print(json.dumps(results,indent=2),flush=True)


if __name__=='__main__':main()
