"""Controlled train/validation experiments; no test or attachment3 access."""
from common import *
from data import *
from dynamic import EpochStore
from model import Model
from engine import predict,metrics,score
from torch.nn import functional as F
import argparse,random,time,fcntl

def fit(name,config,seed,store,labels,valid,protocol):
    with open(ROOT/f'checkpoints/{name}_{seed}.lock','w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        return fit_locked(name,config,seed,store,labels,valid,protocol)

def fit_locked(name,config,seed,store,labels,valid,protocol):
    path=ROOT/f'checkpoints/{name}_{seed}.pt';done=ROOT/f'results/completed_{name}_{seed}.json'
    if done.exists():
        entry=json.loads(done.read_text());assert entry['sha256']==sha256(path)
        assert entry['config']==config and entry['protocol_sha256']==sha256(ROOT/'protocol.json')
        return entry['score']
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    model=Model(config,protocol['prior'],protocol['mean']).cuda()
    c=torch.as_tensor(labels['train']['c'],device='cuda');y=torch.as_tensor(labels['train']['y'],device='cuda')
    cw=(len(c)/(3*torch.bincount(c,minlength=3))).sqrt();cw/=cw.mean()
    history=[];best_global=float('inf');stage1_state=None;start_time=time.time()
    for stage,max_epochs in enumerate(protocol['stage_epochs'],1):
        if stage==2:model.load_state_dict(stage1_state)
        optimizer=torch.optim.AdamW(model.parameters(),lr=protocol['lr'],weight_decay=protocol['weight_decay'])
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=max_epochs,eta_min=1e-4)
        best_stage=float('inf');patience=0
        for ep in range(1,max_epochs+1):
            epoch=ep+(protocol['stage_epochs'][0] if stage==2 else 0)
            tv=torch_view(store.get(seed,epoch));model.train()
            order=np.random.default_rng(seed*1000+epoch).permutation(len(y));loss_sum=0.;batches=0
            for start in range(0,len(y),protocol['batch']):
                ii=torch.as_tensor(order[start:start+protocol['batch']],device='cuda');b=slice_view(tv,ii)
                optimizer.zero_grad(set_to_none=True);out=model(b)
                loss=F.cross_entropy(out['logits'],c[ii],weight=cw)+protocol['reg_weight']*F.smooth_l1_loss(out['pred'],y[ii])
                if not torch.isfinite(loss):raise FloatingPointError(f'{name} {seed} {epoch}')
                loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),protocol['clip_grad'],error_if_nonfinite=True);optimizer.step()
                loss_sum+=float(loss.detach());batches+=1
            scheduler.step();del tv
            results=[]
            for view in valid:
                pred=predict(model,view)
                results.append(metrics(labels['valid']['c'],labels['valid']['y'],pred[:,:3],pred[:,3]))
            jscore=score(results)
            history.append(dict(stage=stage,epoch=epoch,score=jscore,clean=results[0],validation=results,loss=loss_sum/batches,elapsed_seconds=time.time()-start_time))
            if jscore<best_stage-1e-5:
                best_stage=jscore;patience=0
                if stage==1:stage1_state={k:v.detach().clone() for k,v in model.state_dict().items()}
            else:patience+=1
            if jscore<best_global-1e-5:
                best_global=jscore
                ck=dict(name=name,seed=seed,config=config,prior=protocol['prior'],mean=protocol['mean'],stage=stage,epoch=epoch,score=jscore,
                    reg_weight=protocol['reg_weight'],state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()},protocol_sha256=sha256(ROOT/'protocol.json'))
                torch.save(ck,path)
            save_json(ROOT/f'results/history_{name}_{seed}.json',history)
            print(f'{name} seed={seed} stage={stage} epoch={epoch} J={jscore:.5f} F1={results[0]["macro_f1"]:.4f} MAE={results[0]["mae"]:.4f} elapsed={time.time()-start_time:.0f}s',flush=True)
            if patience>=protocol['patience']:break
    save_json(done,dict(score=best_global,config=config,sha256=sha256(path),protocol_sha256=sha256(ROOT/'protocol.json'),epochs=len(history),seconds=time.time()-start_time))
    return best_global

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int,required=True);ap.add_argument('--names',nargs='+',required=True);args=ap.parse_args()
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
    protocol=json.loads((ROOT/'protocol.json').read_text());raw=load_attachment2()
    labels={s:dict(np.load(ROOT/f'cache/{s}_labels.npz')) for s in ['train','valid']}
    valid=[torch_view(dict(np.load(ROOT/f'cache/valid_{j}.npz'))) for j in range(6)]
    for name in args.names:
        shape={'shape_multi':'multi','shape_unequal':'unequal'}.get(name,'single')
        store=EpochStore(raw['train'],shape)
        fit(name,protocol['configs'][name],args.seed,store,labels,valid,protocol)
        del store

if __name__=='__main__':main()
