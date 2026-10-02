"""Fixed train/valid diagnostics. No test predictions, no updates to existing models."""
from common import *
from data import *
from dynamic import EpochStore
from engine import *
from base_model import Model as BaseModel
from support import grouped_metrics,train_subset_indices,state_hash
from torch.nn import functional as F
import pandas as pd,random


def generalization(raw,protocol):
    rows=[];stages=[]
    for name in ['text','audio','vision','gated']:
        for seed,path in zip([17,29,43],checkpoint_paths(name)):
            m,ck=load_model(path);before=state_hash(m)
            for split in ['train','valid']:
                view=dict(np.load(ROOT/f'cache/{split}_0.npz'));d=raw[split]
                pred=predict(m,view)
                np.savez_compressed(ROOT/f'results/diagnostic_{split}_{name}_{seed}.npz',out=pred,ids=np.asarray(d['id'],dtype=str))
                rows.extend(dict(model=name,seed=seed,split=split,**r) for r in grouped_metrics(d['classification_labels'].astype(int),d['regression_labels'],pred))
            assert state_hash(m)==before
            hist=json.loads((PROJECT/f'step2_v2/results/history_{name}_{seed}.json').read_text())
            stages.append(dict(model=name,seed=seed,best_epoch=ck['epoch'],stage=ck['stage'],epochs_run=len(hist),initial_J=hist[0]['score'],final_J=hist[-1]['score']))
            print('Generalization',name,seed,flush=True);del m
    pd.DataFrame(rows).to_csv(ROOT/'results/generalization_groups.csv',index=False)
    pd.DataFrame(stages).to_csv(ROOT/'results/historical_stages.csv',index=False)


def memory_probe(raw,protocol):
    base=adapt(raw['train']);c=raw['train']['classification_labels'].astype(int);y=raw['train']['regression_labels']
    eligible=base['o'][:,1].any(-1)&base['o'][:,2].any(-1);rng=np.random.default_rng(17)
    indices=np.concatenate([rng.choice(np.flatnonzero((c==k)&eligible),32,replace=False) for k in range(3)]);rng.shuffle(indices)
    save_json(ROOT/'results/memory_probe_samples.json',dict(seed=17,indices=indices.tolist(),ids=[str(raw['train']['id'][i]) for i in indices],classes=np.bincount(c[indices],minlength=3).tolist()))
    view={k:v[indices] for k,v in dict(np.load(ROOT/'cache/train_0.npz')).items() if k!='removed'};tv=torch_view(view)
    target=torch.tensor(c[indices],device='cuda');intensity=torch.tensor(y[indices],device='cuda');rows=[];summary=[]
    train_c=torch.tensor(c,device='cuda');cw=(len(c)/(3*torch.bincount(train_c,minlength=3))).sqrt();cw/=cw.mean()
    for name,idx in [('audio',1),('vision',2)]:
        random.seed(17);np.random.seed(17);torch.manual_seed(17);torch.cuda.manual_seed_all(17)
        m=BaseModel(dict(indices=[idx]),protocol['prior'],protocol['mean']).cuda().eval()
        optimizer=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=0.)
        passed=False
        for step in range(1,301):
            optimizer.zero_grad(set_to_none=True);r=m(tv)
            loss=F.cross_entropy(r['logits'],target,weight=cw)+2*F.smooth_l1_loss(r['pred'],intensity)
            assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),5,error_if_nonfinite=True);optimizer.step()
            if step%10==0 or step==1:
                p=predict(m,tv);met=metrics(c[indices],y[indices],p[:,:3],p[:,3]);rows.append(dict(model=name,step=step,loss=float(loss.detach()),**met))
                passed=met['accuracy']>=.95 and met['mae']<=.1
                if passed:break
        summary.append(dict(model=name,steps=step,passed=passed,**met));print('Memory probe',summary[-1],flush=True)
    pd.DataFrame(rows).to_csv(ROOT/'results/memory_probe_history.csv',index=False);save_json(ROOT/'results/memory_probe_summary.json',summary)


def complementarity(raw,protocol):
    profiles=list(scenarios());base=adapt(raw['valid']);preds={n:dict(np.load(ROOT/f'results/valid_{n}_predictions.npz')) for n in ['text','audio','vision','gated']}
    c=raw['valid']['classification_labels'].astype(int);y=raw['valid']['regression_labels'];rows=[]
    for data in preds.values():assert np.array_equal(data['ids'],np.asarray(raw['valid']['id'],dtype=str)) and np.array_equal(data['c'],c) and np.array_equal(data['y'],y)
    avg={n:d['out'].mean(1) for n,d in preds.items()}
    for si,profile in enumerate(profiles):
        if si==0 or profile['primary']:
            o,_=corrupt(base,profile,protocol['evaluation_seed']+profile['repeat']);present=o.any(-1)
            correct=np.stack([avg[n][si,:,:3].argmax(-1)==c for n in ['text','audio','vision']],1)
            rescued=(correct[:,1:]&present[:,1:]).any(-1);text_correct=correct[:,0]
            multimodal_correct=avg['gated'][si,:,:3].argmax(-1)==c
            error=np.stack([np.abs(avg[n][si,:,3]-y) for n in ['text','audio','vision']],1)
            oracle=np.where(present,error,np.inf).min(-1);oracle=np.where(present.any(-1),oracle,np.abs(protocol['mean']-y))
            for group,mask in [('all',np.ones(len(y),bool)),('neutral',y==0),('strong_negative',y<=-1.5),('strong_positive',y>=1.5)]:
                wrong=mask&~text_correct;right=mask&text_correct
                rows.append(dict(**profile,group=group,n=int(mask.sum()),text_wrong=int(wrong.sum()),text_wrong_rescued=int((wrong&rescued).sum()),
                    rescue_fraction=float(rescued[wrong].mean()) if wrong.any() else None,text_correct=int(right.sum()),
                    text_correct_harmed=int((right&~multimodal_correct).sum()),harm_fraction=float((~multimodal_correct[right]).mean()) if right.any() else None,
                    text_mae=float(error[mask,0].mean()),oracle_available_mae=float(oracle[mask].mean()),
                    oracle_accuracy=float((correct&present).any(-1)[mask].mean()),deployable=False))
    pd.DataFrame(rows).to_csv(ROOT/'results/complementarity.csv',index=False)
    print('Complementarity complete; oracle uses true labels and is not deployable.',flush=True)


def gradients(raw,protocol):
    ii=train_subset_indices();c=torch.tensor(raw['train']['classification_labels'].astype(int),device='cuda');y=torch.tensor(raw['train']['regression_labels'],device='cuda')
    cw=(len(c)/(3*torch.bincount(c,minlength=3))).sqrt();cw/=cw.mean()
    store=EpochStore(raw['train']);views={'clean':dict(np.load(ROOT/'cache/train_0.npz')),'dynamic':store.get(17,1)};rows=[]
    save_json(ROOT/'results/fixed_train_subset.json',dict(seed=17,indices=ii.tolist(),ids=[str(raw['train']['id'][i]) for i in ii]))
    for seed,path in zip([17,29,43],checkpoint_paths('gated')):
        m,_=load_model(path);before=state_hash(m)
        blocks={'branches':list(m.input.parameters())+list(m.branches.parameters()),'gate':list(m.gate.parameters())}
        params=[p for ps in blocks.values() for p in ps]
        for condition,view in views.items():
            for bi in range(8):
                indices=ii[bi*64:(bi+1)*64];b=torch_view({k:v[indices] for k,v in view.items() if k!='removed'})
                r=m(b);cls=F.cross_entropy(r['logits'],c[indices],weight=cw);reg=2*F.smooth_l1_loss(r['pred'],y[indices])
                gc=torch.autograd.grad(cls,params,retain_graph=True,allow_unused=True);gr=torch.autograd.grad(reg,params,allow_unused=True)
                start=0
                for block,ps in blocks.items():
                    v1=torch.cat([(g if g is not None else torch.zeros_like(p)).flatten() for p,g in zip(ps,gc[start:start+len(ps)])])
                    v2=torch.cat([(g if g is not None else torch.zeros_like(p)).flatten() for p,g in zip(ps,gr[start:start+len(ps)])]);start+=len(ps)
                    n1=float(v1.norm());n2=float(v2.norm());zero=n1<=1e-12 or n2<=1e-12
                    cosine=float(F.cosine_similarity(v1[None],v2[None])) if not zero else None
                    rows.append(dict(seed=seed,condition=condition,batch=bi,block=block,classification_norm=n1,weighted_regression_norm=n2,
                        cosine=cosine,norm_ratio=n1/n2 if n2>1e-12 else None,zero_gradient=zero,negative=bool(cosine<0) if cosine is not None else None))
        assert state_hash(m)==before;print('Gradient diagnosis',seed,flush=True)
    pd.DataFrame(rows).to_csv(ROOT/'results/task_gradients.csv',index=False)


def main():
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    assert all(json.loads((ROOT/'results/verification.json').read_text()).values())
    raw=load_attachment2();protocol=json.loads((ROOT/'protocol.json').read_text())
    for fn in [generalization,memory_probe,complementarity,gradients]:fn(raw,protocol)
    save_json(ROOT/'results/diagnostics_complete.json',dict(protocol_sha256=sha256(ROOT/'protocol.json'),safe_to_compare=True,test_used=False,
        diagnostics=['full_train_valid','96_sample_memory','complementarity','gradient_cosine','intensity_saturation']))


if __name__=='__main__':main()
