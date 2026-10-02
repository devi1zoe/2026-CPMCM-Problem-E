"""Group 4 uses train-only targets and the same masks/schedule as shape_single."""
from common import *
from data import *
from dynamic import EpochStore
from model import Model,study_state,load_study_state
from engine import predict,metrics,score
from torch.nn import functional as F
import argparse,random,time,fcntl

def fit(seed,store,labels,valid,protocol):
    name='reconstruct';path=ROOT/f'checkpoints/{name}_{seed}.pt';done=ROOT/f'results/completed_{name}_{seed}.json'
    if done.exists():
        entry=json.loads(done.read_text());assert entry['sha256']==sha256(path)
        assert entry['protocol_sha256']==sha256(ROOT/'protocol.json');return
    source=ROOT/f'checkpoints/shape_single_{seed}.pt';source_done=json.loads((ROOT/f'results/completed_shape_single_{seed}.json').read_text())
    assert sha256(source)==source_done['sha256']
    ck=torch.load(source,map_location='cpu',weights_only=True)
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    config=protocol['configs'][name];model=Model(config,protocol['prior'],protocol['mean']).cuda()
    target_state={k[len('input.'):]:v for k,v in ck['state_dict'].items() if k.startswith('input.')}
    model.target_input.load_state_dict(target_state);assert all(not p.requires_grad for p in model.target_input.parameters())
    c=torch.as_tensor(labels['train']['c'],device='cuda');y=torch.as_tensor(labels['train']['y'],device='cuda')
    cw=(len(c)/(3*torch.bincount(c,minlength=3))).sqrt();cw/=cw.mean()
    targets={f'target_{m}':torch.as_tensor(store.clean[m],device='cuda') for m in ['text','audio','vision']}
    history=[];best_global=float('inf');stage1=None;start_time=time.time()
    for stage,max_epochs in enumerate(protocol['stage_epochs'],1):
        if stage==2:load_study_state(model,stage1)
        optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=protocol['lr'],weight_decay=protocol['weight_decay'])
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=max_epochs,eta_min=1e-4)
        best_stage=float('inf');patience=0
        for ep in range(1,max_epochs+1):
            epoch=ep+(protocol['stage_epochs'][0] if stage==2 else 0);view=store.get(seed,epoch)
            tv=torch_view(view);tv.update(targets)
            tv['reconstruction_mask']=torch.as_tensor(view['removed']&store.base['o'],device='cuda')
            model.train();order=np.random.default_rng(seed*1000+epoch).permutation(len(y));sums=np.zeros(3);batches=0
            for start in range(0,len(y),protocol['batch']):
                ii=torch.as_tensor(order[start:start+protocol['batch']],device='cuda');optimizer.zero_grad(set_to_none=True)
                out=model(slice_view(tv,ii));task=F.cross_entropy(out['logits'],c[ii],weight=cw)+protocol['reg_weight']*F.smooth_l1_loss(out['pred'],y[ii])
                loss=task+protocol['reconstruction']['weight']*out['reconstruction_loss']
                if not torch.isfinite(loss):raise FloatingPointError(f'reconstruct {seed} {epoch}')
                loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),protocol['clip_grad'],error_if_nonfinite=True);optimizer.step()
                sums+=np.array([float(task.detach()),float(out['reconstruction_loss'].detach()),float(out['reconstruction_mse'].detach())]);batches+=1
            scheduler.step();del tv,view;results=[]
            for view in valid:
                pred=predict(model,view,batch=128);results.append(metrics(labels['valid']['c'],labels['valid']['y'],pred[:,:3],pred[:,3]))
            js=score(results);history.append(dict(stage=stage,epoch=epoch,score=js,clean=results[0],validation=results,
                task_loss=sums[0]/batches,reconstruction_loss=sums[1]/batches,reconstruction_mse=sums[2]/batches,elapsed_seconds=time.time()-start_time))
            if js<best_stage-1e-5:
                best_stage=js;patience=0
                if stage==1:stage1=study_state(model,cpu=False)
            else:patience+=1
            if js<best_global-1e-5:
                best_global=js
                torch.save(dict(name=name,seed=seed,config=config,prior=protocol['prior'],mean=protocol['mean'],stage=stage,epoch=epoch,score=js,
                    state_dict=study_state(model),protocol_sha256=sha256(ROOT/'protocol.json'),target_source_sha256=sha256(source)),path)
            save_json(ROOT/f'results/history_{name}_{seed}.json',history)
            print(f'reconstruct seed={seed} stage={stage} epoch={epoch} J={js:.5f} F1={results[0]["macro_f1"]:.4f} MAE={results[0]["mae"]:.4f} elapsed={time.time()-start_time:.0f}s',flush=True)
            if patience>=protocol['patience']:break
    save_json(done,dict(score=best_global,config=config,sha256=sha256(path),protocol_sha256=sha256(ROOT/'protocol.json'),target_source_sha256=sha256(source),epochs=len(history),seconds=time.time()-start_time))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int,required=True);args=ap.parse_args()
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
    protocol=json.loads((ROOT/'protocol.json').read_text());raw=load_attachment2();store=EpochStore(raw['train'])
    labels={s:dict(np.load(ROOT/f'cache/{s}_labels.npz')) for s in ['train','valid']}
    valid=[torch_view(dict(np.load(ROOT/f'cache/valid_{j}.npz'))) for j in range(6)]
    with open(ROOT/f'checkpoints/reconstruct_{args.seed}.lock','w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);fit(args.seed,store,labels,valid,protocol)

if __name__=='__main__':main()
