"""Group 2: online masked text encoding, train only small low-rank adapters."""
from common import *
from data import *
from dynamic import EpochStore
from model import Model,study_state,load_study_state
from engine import predict,metrics,score
from torch.nn import functional as F
import argparse,random,time,fcntl

def fit(name,seed,store,labels,valid,protocol):
    path=ROOT/f'checkpoints/{name}_{seed}.pt';done=ROOT/f'results/completed_{name}_{seed}.json'
    if done.exists():
        entry=json.loads(done.read_text());assert entry['sha256']==sha256(path)
        assert entry['protocol_sha256']==sha256(ROOT/'protocol.json');return
    config=protocol['configs'][name] if name!='combined' else json.loads((ROOT/'combination_lock.json').read_text())['config']
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    model=Model(config,protocol['prior'],protocol['mean']).cuda()
    c=torch.as_tensor(labels['train']['c'],device='cuda');y=torch.as_tensor(labels['train']['y'],device='cuda')
    cw=(len(c)/(3*torch.bincount(c,minlength=3))).sqrt();cw/=cw.mean()
    named=[(k,p) for k,p in model.named_parameters() if p.requires_grad]
    assert all('.lora_' in k for k,p in named if k.startswith('language.'))
    heads=[p for k,p in named if not k.startswith('language.')];adapters=[p for k,p in named if k.startswith('language.')]
    history=[];best_global=float('inf');stage1=None;start_time=time.time()
    for stage,max_epochs in enumerate(protocol['stage_epochs'],1):
        if stage==2:load_study_state(model,stage1)
        optimizer=torch.optim.AdamW([dict(params=heads,lr=protocol['lr']),dict(params=adapters,lr=protocol['lora']['lr'])],weight_decay=protocol['weight_decay'])
        # Both groups retain 10% of their initial learning rate at the end of a stage.
        scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda e:.1+.9*(1+np.cos(np.pi*e/max_epochs))/2)
        scaler=torch.amp.GradScaler('cuda',init_scale=1024);best_stage=float('inf');patience=0
        for ep in range(1,max_epochs+1):
            epoch=ep+(protocol['stage_epochs'][0] if stage==2 else 0)
            tv=torch_view(store.get(seed,epoch,raw_tokens=True));model.train()
            order=np.random.default_rng(seed*1000+epoch).permutation(len(y));loss_sum=0.;batches=0;overflow_batches=0
            for start in range(0,len(y),protocol['batch']):
                ii=torch.as_tensor(order[start:start+protocol['batch']],device='cuda');optimizer.zero_grad(set_to_none=True)
                denominator=cw[c[ii]].sum();batch_loss=0.
                for off in range(0,len(ii),protocol['lora']['micro_batch']):
                    jj=ii[off:off+protocol['lora']['micro_batch']];out=model(slice_view(tv,jj))
                    loss=F.cross_entropy(out['logits'],c[jj],weight=cw,reduction='sum')/denominator
                    loss+=protocol['reg_weight']*F.smooth_l1_loss(out['pred'],y[jj],reduction='sum')/len(ii)
                    if not torch.isfinite(loss):raise FloatingPointError(f'{name} {seed} {epoch}')
                    scaler.scale(loss).backward();batch_loss+=float(loss.detach())
                scaler.unscale_(optimizer)
                # GradScaler has already recorded non-finite gradients during unscale.
                # In that case skip the update and reduce scale, never divide by inf.
                finite=all(p.grad is None or torch.isfinite(p.grad).all() for _,p in named)
                if finite:torch.nn.utils.clip_grad_norm_([p for _,p in named],protocol['clip_grad'],error_if_nonfinite=True)
                else:overflow_batches+=1
                scaler.step(optimizer);scaler.update();loss_sum+=batch_loss;batches+=1
            scheduler.step();del tv
            results=[]
            for view in valid:
                pred=predict(model,view,batch=64)
                results.append(metrics(labels['valid']['c'],labels['valid']['y'],pred[:,:3],pred[:,3]))
            js=score(results)
            history.append(dict(stage=stage,epoch=epoch,score=js,clean=results[0],validation=results,loss=loss_sum/batches,
                skipped_overflow_batches=overflow_batches,gradient_scale=scaler.get_scale(),elapsed_seconds=time.time()-start_time))
            if js<best_stage-1e-5:
                best_stage=js;patience=0
                if stage==1:stage1=study_state(model,cpu=False)
            else:patience+=1
            if js<best_global-1e-5:
                best_global=js
                torch.save(dict(name=name,seed=seed,config=config,prior=protocol['prior'],mean=protocol['mean'],stage=stage,epoch=epoch,score=js,
                    state_dict=study_state(model),protocol_sha256=sha256(ROOT/'protocol.json'),encoder_sha256=protocol['encoder']['weights_sha256']),path)
            save_json(ROOT/f'results/history_{name}_{seed}.json',history)
            print(f'{name} seed={seed} stage={stage} epoch={epoch} J={js:.5f} F1={results[0]["macro_f1"]:.4f} MAE={results[0]["mae"]:.4f} elapsed={time.time()-start_time:.0f}s',flush=True)
            if patience>=protocol['patience']:break
    save_json(done,dict(score=best_global,config=config,sha256=sha256(path),protocol_sha256=sha256(ROOT/'protocol.json'),epochs=len(history),seconds=time.time()-start_time,
        trainable_parameters=sum(p.numel() for _,p in named),adapter_parameters=sum(p.numel() for p in adapters)))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int,required=True);ap.add_argument('--names',nargs='+',default=['text_lora','multi_lora']);args=ap.parse_args()
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
    protocol=json.loads((ROOT/'protocol.json').read_text());raw=load_attachment2();base=adapt(raw['valid'])
    labels={s:dict(np.load(ROOT/f'cache/{s}_labels.npz')) for s in ['train','valid']}
    valid=[]
    for j in range(6):
        view=dict(np.load(ROOT/f'cache/valid_{j}.npz'));view.pop('text')
        view.update(tokens=base['tokens'],attention=base['attention']);valid.append(torch_view(view))
    store=EpochStore(raw['train'])
    for name in args.names:
        if name=='combined':
            decision=json.loads((ROOT/'combination_lock.json').read_text());assert decision['run']
            store.shape={'shape_multi':'multi','shape_unequal':'unequal'}[decision['shape']]
        else:store.shape='single'
        with open(ROOT/f'checkpoints/{name}_{args.seed}.lock','w') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX);fit(name,args.seed,store,labels,valid,protocol)

if __name__=='__main__':main()
