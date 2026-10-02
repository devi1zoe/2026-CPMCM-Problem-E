"""Exactly six preregistered candidate fits; validation-only checkpoint selection."""
from common import *
from data import *
from dynamic import EpochStore
from model import Model,task_loss
from support import load_experts,state_hash,expert_view,train_subset_indices
from engine import metrics,predict,score
import random,time,argparse,fcntl


def fit(name,seed,store,labels,valid,protocol):
    path=ROOT/f'checkpoints/{name}_{seed}.pt';done=ROOT/f'results/completed_{name}_{seed}.json';config=protocol['configs'][name]
    with open(str(path)+'.lock','w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if done.exists():
            proof=json.loads(done.read_text());assert proof['sha256']==sha256(path) and proof['protocol_sha256']==sha256(ROOT/'protocol.json');return
        random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
        model=Model(config,protocol['prior'],protocol['mean']).cuda();expert_hashes={};frozen_before=None
        if name=='late_fusion':expert_hashes=load_experts(model,seed);frozen_before=state_hash(model.experts)
        fixed={k:v[train_subset_indices()] for k,v in store.clean.items() if k!='removed'}
        vv=valid
        if name=='late_fusion':
            vv=[expert_view(model,v,expert_hashes) for v in valid];fixed=expert_view(model,fixed,expert_hashes)
        fixed=torch_view(fixed);vv=[torch_view(v) for v in vv]
        c=torch.tensor(labels['train']['c'],device='cuda');y=torch.tensor(labels['train']['y'],device='cuda')
        cw=(len(c)/(3*torch.bincount(c,minlength=3))).sqrt();cw/=cw.mean()
        trainable=[p for p in model.parameters() if p.requires_grad];history=[];best_global=float('inf');stage1_state=None;start=time.time()
        for stage,max_epochs in enumerate(protocol['stage_epochs'],1):
            if stage==2:model.load_state_dict(stage1_state)
            optimizer=torch.optim.AdamW(trainable,lr=protocol['lr'],weight_decay=protocol['weight_decay'])
            scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=max_epochs,eta_min=1e-4)
            best_stage=float('inf');patience=0
            for ep in range(1,max_epochs+1):
                epoch=ep+(20 if stage==2 else 0);view=store.get(seed,epoch)
                if name=='late_fusion':view=expert_view(model,view,expert_hashes)
                tv=torch_view(view);del view;model.train();sums={};count=0
                order=np.random.default_rng(seed*1000+epoch).permutation(len(c))
                for start_batch in range(0,len(c),protocol['batch']):
                    ii=torch.tensor(order[start_batch:start_batch+protocol['batch']],device='cuda');b=slice_view(tv,ii)
                    optimizer.zero_grad(set_to_none=True);loss,parts=task_loss(model(b),c[ii],y[ii],cw)
                    if not torch.isfinite(loss):raise FloatingPointError(f'Nonfinite {name}/{seed}/{epoch}')
                    loss.backward();torch.nn.utils.clip_grad_norm_(trainable,5,error_if_nonfinite=True);optimizer.step()
                    for key,v in parts.items():sums[key]=sums.get(key,0.)+float(v.detach())
                    count+=1
                scheduler.step();del tv
                results=[]
                for v in vv:
                    out=predict(model,v);results.append(metrics(labels['valid']['c'],labels['valid']['y'],out[:,:3],out[:,3]))
                p=predict(model,fixed);ii=train_subset_indices()
                train_diag=metrics(labels['train']['c'][ii],labels['train']['y'][ii],p[:,:3],p[:,3]);j=score(results)
                history.append(dict(stage=stage,epoch=epoch,score=j,clean=results[0],fixed_train=train_diag,
                    loss={k:v/count for k,v in sums.items()},elapsed_seconds=time.time()-start))
                if j<best_stage-1e-5:
                    best_stage=j;patience=0
                    if stage==1:stage1_state={k:v.detach().clone() for k,v in model.state_dict().items()}
                else:patience+=1
                if j<best_global-1e-5:
                    best_global=j;torch.save(dict(name=name,seed=seed,config=config,prior=protocol['prior'],mean=protocol['mean'],stage=stage,epoch=epoch,score=j,reg_weight=2.,
                        expert_sources=expert_hashes,state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()},protocol_sha256=sha256(ROOT/'protocol.json')),path)
                save_json(ROOT/f'results/history_{name}_{seed}.json',history)
                print(f'{name} seed={seed} stage={stage} ep={epoch} J={j:.5f} F1={results[0]["macro_f1"]:.4f} MAE={results[0]["mae"]:.4f} trainF1={train_diag["macro_f1"]:.4f}',flush=True)
                if patience>=protocol['patience']:break
        if name=='late_fusion':assert state_hash(model.experts)==frozen_before
        save_json(done,dict(score=best_global,config=config,reg_weight=2.,sha256=sha256(path),epochs=len(history),seconds=time.time()-start,
            protocol_sha256=sha256(ROOT/'protocol.json'),trainable_parameters=sum(p.numel() for p in trainable),
            total_parameters=sum(p.numel() for p in model.parameters()),expert_sources=expert_hashes,frozen_experts_unchanged=name=='late_fusion'))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int,required=True,choices=[17,29,43]);ap.add_argument('--names',nargs='+',default=['dual_gate','late_fusion'],choices=['dual_gate','late_fusion']);args=ap.parse_args()
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
    protocol=json.loads((ROOT/'protocol.json').read_text());proof=json.loads((ROOT/'results/diagnostics_complete.json').read_text())
    assert proof['safe_to_compare'] and proof['protocol_sha256']==sha256(ROOT/'protocol.json')
    raw=load_attachment2();store=EpochStore(raw['train']);del raw
    labels={s:dict(np.load(ROOT/f'cache/{s}_labels.npz')) for s in ['train','valid']}
    valid=[dict(np.load(ROOT/f'cache/valid_{j}.npz')) for j in range(6)]
    for name in args.names:fit(name,args.seed,store,labels,valid,protocol)


if __name__=='__main__':main()
