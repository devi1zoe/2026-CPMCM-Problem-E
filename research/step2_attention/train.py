from common import *
from data import *
from dynamic import EpochStore
from model import Model
from engine import predict,metrics,score
import argparse,random,time


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int,required=True);args=ap.parse_args();seed=args.seed
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False
    proto=json.loads((ROOT/'protocol.json').read_text());assert seed in proto['seeds']
    assert json.loads((ROOT/'results/verification.json').read_text())['passed']
    raw=load_attachment2()['train'];store=EpochStore(raw);del raw
    labels={s:dict(np.load(ROOT/f'cache/{s}_labels.npz')) for s in ['train','valid']}
    y=torch.tensor(labels['train']['y'],device='cuda');c=torch.tensor(labels['train']['c'],device='cuda')
    weights=(len(c)/(3*torch.bincount(c).float())).sqrt();weights/=weights.mean()
    vv=[torch_view(dict(np.load(ROOT/f'cache/valid_{j}.npz'))) for j in range(6)]
    for name in proto['models']:
        done=ROOT/f'results/completed_{name}_{seed}.json'
        if done.exists():
            d=json.loads(done.read_text());assert d['protocol_sha256']==sha256(ROOT/'protocol.json') and d['checkpoint_sha256']==sha256(ROOT/f'checkpoints/{name}_{seed}.pt');continue
        torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);random.seed(seed);rng=np.random.default_rng(seed)
        model=Model(name,proto['prior'],proto['mean'],proto['hidden'],proto['heads'],proto['dropout']).cuda()
        opt=torch.optim.AdamW(model.parameters(),lr=proto['lr'],weight_decay=proto['weight_decay'])
        sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,proto['epochs'],eta_min=3e-5)
        best=float('inf');bad=0;history=[];start=time.time()
        for epoch in range(1,proto['epochs']+1):
            v=torch_view(store.clean if name=='cross_clean' else store.get(seed,epoch));model.train();losses=[]
            for ii in np.array_split(rng.permutation(len(c)),int(np.ceil(len(c)/proto['batch']))):
                idx=torch.as_tensor(ii,device='cuda');r=model(slice_view(v,idx))
                loss=torch.nn.functional.cross_entropy(r['logits'],c[idx],weight=weights)+proto['reg_weight']*torch.nn.functional.smooth_l1_loss(r['pred'],y[idx])
                opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);opt.step();losses.append(float(loss.detach()))
            sched.step();rows=[metrics(labels['valid']['c'],labels['valid']['y'],predict(model,x)) for x in vv];j=score(rows)
            history.append(dict(epoch=epoch,loss=float(np.mean(losses)),J=j,clean=rows[0],corrupt_macro_f1=float(np.mean([r['macro_f1'] for r in rows[1:]])),corrupt_mae=float(np.mean([r['mae'] for r in rows[1:]]))))
            if j<best-1e-6:
                best=j;bad=0
                torch.save(dict(name=name,seed=seed,epoch=epoch,J=j,protocol_sha256=sha256(ROOT/'protocol.json'),model_sha256=sha256(ROOT/'model.py'),state_dict={k:t.detach().cpu() for k,t in model.state_dict().items()}),ROOT/f'checkpoints/{name}_{seed}.pt')
            else:bad+=1
            save_json(ROOT/f'results/history_{name}_{seed}.json',history)
            print(f'{name} seed={seed} epoch={epoch} J={j:.4f} F1={rows[0]["macro_f1"]:.4f} MAE={rows[0]["mae"]:.4f}',flush=True)
            del v
            if bad>=proto['patience']:break
        save_json(done,dict(name=name,seed=seed,epochs=len(history),seconds=time.time()-start,parameters=sum(p.numel() for p in model.parameters()),checkpoint_sha256=sha256(ROOT/f'checkpoints/{name}_{seed}.pt'),protocol_sha256=sha256(ROOT/'protocol.json')))
        print(f'FINISHED {name} seed={seed} best={best:.4f} seconds={time.time()-start:.1f}',flush=True)


if __name__=='__main__':main()
