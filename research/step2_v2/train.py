"""Train/validation only. All ten preregistered models and leave-one-out ablations."""
from common import *
from data import *
from dynamic import EpochStore
from model import Model,VARIANTS,objectives
from engine import predict,metrics,score,load_model
import argparse,copy,random,time,fcntl

def fit(name,config,seed,reg_weight,store,labels,valid,protocol):
    with open(ROOT/f'checkpoints/{name}_{seed}.lock','w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        return fit_locked(name,config,seed,reg_weight,store,labels,valid,protocol)

def fit_locked(name,config,seed,reg_weight,store,labels,valid,protocol):
    path=ROOT/f'checkpoints/{name}_{seed}.pt';done=ROOT/f'results/completed_{name}_{seed}.json'
    if done.exists():
        entry=json.loads(done.read_text());assert entry['sha256']==sha256(path)
        assert entry['reg_weight']==reg_weight and entry['config']==config
        return entry['score']
    # Names never affect initialization, ordering, masks, schedule, or objectives.
    # Reuse only an exactly equivalent completed controlled experiment.
    normalized=lambda c:{k:v for k,v in c.items() if v is not False}
    reuse=[]
    if name=='gated' and seed==17:reuse=['pilot_'+str(reg_weight).replace('.','p')]
    if '_without_' in name:reuse=list(VARIANTS)
    for previous in reuse:
        source_done=ROOT/f'results/completed_{previous}_{seed}.json'
        if not source_done.exists():continue
        entry=json.loads(source_done.read_text())
        if normalized(entry['config'])!=normalized(config) or entry['reg_weight']!=reg_weight:continue
        source=ROOT/f'checkpoints/{previous}_{seed}.pt';assert sha256(source)==entry['sha256']
        ck=torch.load(source,map_location='cpu',weights_only=True);ck['name']=name;ck['config']=config
        torch.save(ck,path)
        (ROOT/f'results/history_{name}_{seed}.json').write_bytes((ROOT/f'results/history_{previous}_{seed}.json').read_bytes())
        save_json(done,dict(**{k:v for k,v in entry.items() if k not in ['config','sha256']},config=config,sha256=sha256(path),
            reused_identical_experiment=previous))
        print(f'Reused mathematically identical experiment {previous}/{seed} as {name}',flush=True)
        return entry['score']
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    model=Model(config,protocol['prior'],protocol['mean']).cuda()
    c=torch.as_tensor(labels['train']['c'],device='cuda');y=torch.as_tensor(labels['train']['y'],device='cuda')
    cw=(len(c)/(3*torch.bincount(c,minlength=3))).sqrt();cw/=cw.mean()
    history=[];best_global=float('inf');stage1_state=None;teacher=None;start_time=time.time()
    for stage,max_epochs in enumerate(protocol['stage_epochs'],1):
        if stage==2:
            model.load_state_dict(stage1_state)
            if config.get('imtd') or config.get('emc'):
                teacher=copy.deepcopy(model).eval().requires_grad_(False)
        optimizer=torch.optim.AdamW(model.parameters(),lr=protocol['lr'],weight_decay=protocol['weight_decay'])
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=max_epochs,eta_min=1e-4)
        best_stage=float('inf');patience=0
        for ep in range(1,max_epochs+1):
            epoch=ep+(20 if stage==2 else 0)
            tv=torch_view(store.get(seed,epoch));model.train()
            order=np.random.default_rng(seed*1000+epoch).permutation(len(y));sums={};grads={str(m):[] for m in model.indices};batches=0
            for start in range(0,len(y),protocol['batch']):
                ii=torch.as_tensor(order[start:start+protocol['batch']],device='cuda');b=slice_view(tv,ii)
                optimizer.zero_grad(set_to_none=True)
                if teacher is not None:
                    with torch.no_grad():t=teacher(b)
                else:t=None
                out=model(b);loss,components,_=objectives(model,out,b,c[ii],y[ii],cw,reg_weight,t,stage)
                if not torch.isfinite(loss):raise FloatingPointError(f'{name} seed={seed} epoch={epoch}: nonfinite loss')
                loss.backward()
                for m in model.indices:
                    params=list(model.input[str(m)].parameters())+list(model.branches[str(m)].parameters())
                    gs=[p.grad.detach().square().sum() for p in params if p.grad is not None]
                    grads[str(m)].append(float(torch.stack(gs).sum().sqrt()) if gs else 0.)
                torch.nn.utils.clip_grad_norm_(model.parameters(),5,error_if_nonfinite=True);optimizer.step()
                for k,v in components.items():sums[k]=sums.get(k,0.)+float(v.detach())
                batches+=1
            scheduler.step();del tv
            results=[];aux_metrics={}
            for j,view in enumerate(valid):
                pred,a=predict(model,view,auxiliary=True)
                results.append(metrics(labels['valid']['c'],labels['valid']['y'],pred[:,:3],pred[:,3]))
                if j==0 and a is not None:
                    for m in model.indices:
                        keep=view['o'][:,m].any(-1).cpu().numpy();pa=torch.tensor(a[keep,m,:3]).softmax(-1).numpy()
                        aux_metrics[str(m)]=metrics(labels['valid']['c'][keep],labels['valid']['y'][keep],pa,a[keep,m,3])
            jscore=score(results)
            row=dict(stage=stage,epoch=epoch,score=jscore,clean=results[0],loss={k:v/batches for k,v in sums.items()},
                branch_grad_norm={k:float(np.mean(v)) for k,v in grads.items()},aux_validation=aux_metrics,elapsed_seconds=time.time()-start_time)
            history.append(row)
            if jscore<best_stage-1e-5:
                best_stage=jscore;patience=0
                if stage==1:stage1_state={k:v.detach().clone() for k,v in model.state_dict().items()}
            else:patience+=1
            if jscore<best_global-1e-5:
                best_global=jscore
                ck=dict(name=name,seed=seed,config=config,prior=protocol['prior'],mean=protocol['mean'],stage=stage,epoch=epoch,score=jscore,
                    reg_weight=reg_weight,state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()},protocol_sha256=sha256(ROOT/'protocol.json'))
                torch.save(ck,path)
            save_json(ROOT/f'results/history_{name}_{seed}.json',history)
            print(f'{name} s={seed} stage={stage} ep={epoch} J={jscore:.5f} F1={results[0]["macro_f1"]:.4f} MAE={results[0]["mae"]:.4f}',flush=True)
            if patience>=protocol['patience']:break
    save_json(done,dict(score=best_global,config=config,reg_weight=reg_weight,sha256=sha256(path),epochs=len(history),seconds=time.time()-start_time))
    return best_global

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--pilot',action='store_true');ap.add_argument('--pilot-weight',type=float);ap.add_argument('--seed',type=int,default=17);ap.add_argument('--names',nargs='*');args=ap.parse_args()
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
    protocol=json.loads((ROOT/'protocol.json').read_text())
    raw=load_attachment2();store=EpochStore(raw['train'])
    labels={s:dict(np.load(ROOT/f'cache/{s}_labels.npz')) for s in ['train','valid']}
    valid=[torch_view(dict(np.load(ROOT/f'cache/valid_{j}.npz'))) for j in range(6)]
    del raw
    if args.pilot_weight is not None:
        w=args.pilot_weight;assert w in protocol['reg_weight_candidates']
        fit('pilot_'+str(w).replace('.','p'),VARIANTS['gated'],17,w,store,labels,valid,protocol);return
    if args.pilot:
        scores={str(w):fit('pilot_'+str(w).replace('.','p'),VARIANTS['gated'],17,w,store,labels,valid,protocol) for w in protocol['reg_weight_candidates']}
        selected=min(protocol['reg_weight_candidates'],key=lambda w:(scores[str(w)],abs(w-1)))
        save_json(ROOT/'regression_weight_lock.json',dict(selected=selected,scores=scores,protocol_sha256=sha256(ROOT/'protocol.json')))
        print('Regression weight locked:',selected,flush=True);return
    lock=json.loads((ROOT/'regression_weight_lock.json').read_text());assert lock['protocol_sha256']==sha256(ROOT/'protocol.json')
    configs=dict(protocol['configs'])
    if (ROOT/'ablation_configs.json').exists():configs.update(json.loads((ROOT/'ablation_configs.json').read_text()))
    for name in args.names or list(VARIANTS):fit(name,configs[name],args.seed,lock['selected'],store,labels,valid,protocol)

if __name__=='__main__':main()
