from common import *
from data import *
from engine import predict,metrics,score,load_model
from fusion import FusionModel
from torch.nn import functional as F
import argparse,random,time,fcntl

def validation_experts(seed):
    paths=[ROOT/f'checkpoints/expert_{seed}_full_{m}.pt' for m in range(3)];hashes={str(p.relative_to(ROOT)):sha256(p) for p in paths}
    key=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()[:20];cache=ROOT/f'cache/fusion_validation_{key}.npz'
    views=[dict(np.load(ROOT/f'cache/valid_{j}.npz')) for j in range(6)]
    if cache.exists():outputs=np.load(cache)['out']
    else:
        models=[load_model(p)[0] for p in paths];outputs=np.stack([np.stack([predict(m,view) for m in models],1) for view in views])
        np.savez(cache,out=outputs)
    return [torch_view({**{k:v for k,v in view.items() if k in ['p','o','quality']},'experts':out}) for view,out in zip(views,outputs)],hashes

def fit(name,seed,data,valid,expert_hashes,protocol):
    path=ROOT/f'checkpoints/{name}_{seed}.pt';donepath=ROOT/f'results/completed_{name}_{seed}.json'
    if donepath.exists():
        done=json.loads(donepath.read_text());assert done['sha256']==sha256(path) and done['protocol_sha256']==sha256(ROOT/'protocol.json');return
    config=protocol['configs'][name];random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    model=FusionModel(config,protocol['prior'],protocol['mean']).cuda()
    n=len(data['c']);views={k:torch.as_tensor(data[k].reshape((-1,)+data[k].shape[2:]),device='cuda') for k in ['experts','p','o','quality']}
    c=torch.as_tensor(np.tile(data['c'],9),device='cuda');y=torch.as_tensor(np.tile(data['y'],9),device='cuda')
    cw=(len(c)/(3*torch.bincount(c,minlength=3))).sqrt();cw/=cw.mean();val_labels=dict(np.load(ROOT/'cache/valid_labels.npz'))
    optimizer=torch.optim.AdamW(model.parameters(),lr=protocol['fusion']['lr'],weight_decay=protocol['weight_decay'])
    best=float('inf');patience=0;history=[];start_time=time.time()
    # Each epoch draws one of eight paired dynamic views per sample.
    for epoch in range(1,protocol['fusion']['epochs']+1):
        model.train();rng=np.random.default_rng(seed*1000+epoch);order=rng.permutation(n)
        # Dynamic views already contain 50% complete samples, so choose among them;
        # clean view is retained for diagnostic evaluation, not extra overweighting.
        view_id=rng.integers(1,9,size=n);indices=view_id*n+np.arange(n);loss_sum=0.;batches=0
        for off in range(0,n,64):
            ii=torch.as_tensor(indices[order[off:off+64]],device='cuda');optimizer.zero_grad(set_to_none=True);z=model(slice_view(views,ii))
            loss=F.cross_entropy(z['logits'],c[ii],weight=cw)+2*F.smooth_l1_loss(z['pred'],y[ii]);assert torch.isfinite(loss)
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5,error_if_nonfinite=True);optimizer.step();loss_sum+=float(loss.detach());batches+=1
        rows=[]
        for view in valid:
            out=predict(model,view);rows.append(metrics(val_labels['c'],val_labels['y'],out[:,:3],out[:,3]))
        js=score(rows);history.append(dict(epoch=epoch,score=js,clean=rows[0],validation=rows,loss=loss_sum/batches,elapsed_seconds=time.time()-start_time))
        if js<best-1e-5:
            best=js;patience=0
            torch.save(dict(name=name,seed=seed,config=config,prior=protocol['prior'],mean=protocol['mean'],state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()},
                expert_hashes=expert_hashes,epoch=epoch,score=js,protocol_sha256=sha256(ROOT/'protocol.json'),oof_sha256=sha256(ROOT/f'cache/oof_{seed}.npz')),path)
        else:patience+=1
        save_json(ROOT/f'results/history_{name}_{seed}.json',history)
        print(f'{name} seed={seed} epoch={epoch} J={js:.5f} F1={rows[0]["macro_f1"]:.4f} MAE={rows[0]["mae"]:.4f}',flush=True)
        if patience>=protocol['fusion']['patience']:break
    save_json(donepath,dict(score=best,config=config,sha256=sha256(path),protocol_sha256=sha256(ROOT/'protocol.json'),epochs=len(history),seconds=time.time()-start_time,expert_hashes=expert_hashes))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int,required=True);args=ap.parse_args();torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    protocol=json.loads((ROOT/'protocol.json').read_text());done=json.loads((ROOT/f'results/oof_completed_{args.seed}.json').read_text())
    assert done['sha256']==sha256(ROOT/f'cache/oof_{args.seed}.npz');data=dict(np.load(ROOT/f'cache/oof_{args.seed}.npz'))
    valid,hashes=validation_experts(args.seed)
    for name in protocol['groups']['3']:
        with open(ROOT/f'checkpoints/{name}_{args.seed}.lock','w') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX);fit(name,args.seed,data,valid,hashes,protocol)

if __name__=='__main__':main()
