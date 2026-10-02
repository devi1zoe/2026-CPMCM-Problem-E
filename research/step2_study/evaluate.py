"""Shared complete corruption matrix; test access requires final immutable selection lock."""
from common import *
from data import *
from engine import *
from evaluation_data import EvaluationViews,profiles_all,get_masks
import argparse,pandas as pd,fcntl,contextlib

def audit_backend(name,models,protocol,c,y,views):
    """Backend choice uses unlabeled prediction agreement, never sentiment accuracy."""
    path=ROOT/f'results/backend_{name}.json';hashes={str(p.relative_to(PROJECT)):sha256(p) for p in checkpoint_paths(name)}
    if path.exists():
        proof=json.loads(path.read_text());assert proof['checkpoints']==hashes
        return proof['backend'],proof['six_reference']
    from inference_ops import window_conv
    for dilation in [1,2]:
        layer=torch.nn.Conv1d(5,7,3,padding=dilation,dilation=dilation).double();fixture=torch.randn(3,5,11,dtype=torch.float64)
        assert torch.allclose(layer(fixture),window_conv(layer,fixture),atol=1e-12,rtol=1e-12)
    discrepancies=[];six=[]
    for j,profile in enumerate(protocol['validation_profiles']):
        view=dict(np.load(ROOT/f'cache/valid_{j}.npz'))
        paired=[torch_view(views.for_model(m,view,h)) for m,h in zip(models,hashes.values())]
        reference=np.stack([predict(m,tv) for m,tv in zip(models,paired)])
        fast=np.stack([predict(m,tv,backend='linear') for m,tv in zip(models,paired)])
        a=reference.mean(0);b=fast.mean(0)
        discrepancies.append(dict(profile=profile['name'],max_abs=float(np.abs(reference-fast).max()),
            ensemble_class_changes=int((a[:,:3].argmax(-1)!=b[:,:3].argmax(-1)).sum()),
            per_seed_class_changes=int((reference[:,:,:3].argmax(-1)!=fast[:,:,:3].argmax(-1)).sum())))
        six.append(dict(**profile,**metrics(c,y,a[:,:3],a[:,3])))
    backend='linear' if all(r['max_abs']<=1e-3 and r['ensemble_class_changes']==0 for r in discrepancies) else 'cudnn'
    save_json(path,dict(backend=backend,checkpoints=hashes,checks=discrepancies,six_reference=six,
        criterion='Float64 algebraic equivalence; six validation inputs max_abs <= 1e-3 and no ensemble-class changes; otherwise original cuDNN',
        no_label_used_for_backend_choice=True,training_backend_unchanged=True))
    print('Inference backend',name,backend,flush=True)
    return backend,six

def cached_view(raw,split,profile,encoder,stats,protocol):
    stamp=hashlib.sha256(json.dumps(dict(split=split,profile=profile,source=protocol['source_sha256'],encoder=protocol['encoder'],
        stats=protocol['normalization_sha256'],seed=protocol['evaluation_seed']),sort_keys=True).encode()).hexdigest()[:20]
    path=ROOT/f'cache/eval_{split}_{stamp}.npz'
    with open(str(path)+'.lock','w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if not path.exists():
            b=adapt(raw)
            if profile['name']=='clean' and split=='valid':
                view=dict(np.load(ROOT/'cache/valid_0.npz'))
            else:view=make_view(b,encoder,stats,profile,protocol['evaluation_seed']+profile['repeat'])
            tmp=Path(str(path)+'.tmp')
            with tmp.open('wb') as f:np.savez(f,**view)
            tmp.replace(path)
    return dict(np.load(path))

def evaluate(names,split):
    protocol=json.loads((ROOT/'protocol.json').read_text());lock_hash=None
    for name in names:
        if name not in ['old_gated','expert_text']:
            for seed in protocol['seeds']:
                done=json.loads((ROOT/f'results/completed_{name}_{seed}.json').read_text())
                assert done['sha256']==sha256(ROOT/f'checkpoints/{name}_{seed}.pt')
    if split=='test':
        lock=json.loads((ROOT/'selection_lock.json').read_text());lock_hash=sha256(ROOT/'selection_lock.json')
        assert lock['protocol_sha256']==sha256(ROOT/'protocol.json')
        assert set(names)<=set(lock['final_test_models'])
    d=load_attachment2()[split];c=d['classification_labels'].astype(int);y=d['regression_labels']
    profiles=profiles_all();views=EvaluationViews(d,split,protocol)
    models={name:[load_model(p)[0] for p in checkpoint_paths(name)] for name in names}
    if split=='valid':
        backend_info={name:audit_backend(name,ensemble,protocol,c,y,views) for name,ensemble in models.items()}
    else:
        backend_info={name:(json.loads((ROOT/f'results/backend_{name}.json').read_text())['backend'],[]) for name in names}
    outputs={name:np.empty((len(profiles),3,len(y),4),np.float32) for name in names};rows={name:[] for name in names};audit=[]
    hashes={name:[sha256(p) for p in checkpoint_paths(name)] for name in names}
    for si,profile in enumerate(profiles):
        view=views.get(profile)
        expected_o,expected_removed=get_masks(views.base,profile,protocol['evaluation_seed']+profile['repeat'])
        assert np.array_equal(view['o'],expected_o) and np.array_equal(view['removed'],expected_removed)
        for name,ensemble in models.items():
            for j,(seed,m) in enumerate(zip(protocol['seeds'],ensemble)):
                tv=torch_view(views.for_model(m,view,hashes[name][j]))
                out=predict(m,tv,backend=backend_info[name][0]);outputs[name][si,j]=out
                rows[name].append(dict(model=name,seed=str(seed),**profile,**metrics(c,y,out[:,:3],out[:,3])))
            out=outputs[name][si].mean(0);rows[name].append(dict(model=name,seed='ensemble',**profile,**metrics(c,y,out[:,:3],out[:,3])))
        audit.append(dict(**profile,**{f'actual_deleted_fraction_{m}':float((view['removed'][:,j].sum(-1)/np.maximum(view['p'].sum(-1),1)).mean()) for j,m in enumerate('TAV')}))
        del tv,view
        if si%12==0:print(f'{split} {names} scenario {si+1}/{len(profiles)} {profile["name"]}',flush=True)
    for name in names:
        # Six-profile early-stopping objective is evaluated separately from the full matrix.
        six=backend_info[name][1]
        if split=='valid':
            save_json(ROOT/f'results/six_{name}.json',dict(score=score(six),rows=six,
                parameters=sum(p.numel() for k,p in models[name][0].named_parameters() if not (k.startswith('language.backbone.') and '.lora_' not in k)),
                shared_frozen_text_encoder_excluded_for_all_models=True))
        pd.DataFrame(rows[name]).to_csv(ROOT/f'results/{split}_{name}_metrics.csv',index=False)
        np.savez_compressed(ROOT/f'results/{split}_{name}_predictions.npz',out=outputs[name],c=c,y=y,ids=np.asarray(d['id'],dtype=str))
        save_json(ROOT/f'results/evaluated_{split}_{name}.json',dict(checkpoints={str(p.relative_to(PROJECT)):sha256(p) for p in checkpoint_paths(name)},
            profiles=len(profiles),selection_sha256=lock_hash,protocol_sha256=sha256(ROOT/'protocol.json'),all_missing_policy='train_prior',backend=backend_info[name][0]))
    pd.DataFrame(audit).to_csv(ROOT/f'results/{split}_corruption_audit.csv',index=False)
    save_json(ROOT/f'results/{split}_profiles.json',profiles)
    if split=='test':assert lock_hash==sha256(ROOT/'selection_lock.json')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--split',choices=['valid','test'],default='valid');ap.add_argument('--names',nargs='+',required=True);args=ap.parse_args()
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    with contextlib.ExitStack() as stack:
        for name in sorted(set(args.names)):
            handle=stack.enter_context(open(ROOT/f'cache/evaluation_{args.split}_{name}.lock','w'));fcntl.flock(handle,fcntl.LOCK_EX)
        names=[]
        for name in args.names:
            path=ROOT/f'results/evaluated_{args.split}_{name}.json'
            if path.exists():
                existing=json.loads(path.read_text());assert existing['protocol_sha256']==sha256(ROOT/'protocol.json')
                assert all(sha256(PROJECT/p)==h for p,h in existing['checkpoints'].items())
                print('Verified existing completed evaluation',name,flush=True)
            else:names.append(name)
        if names:evaluate(names,args.split)

if __name__=='__main__':main()
