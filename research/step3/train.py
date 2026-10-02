"""问题三完整输入验证目标；在既有训练权重基础上作受控的模态组合增强微调。"""
from common import *
from engine import *
import argparse,time,random,shutil
import pandas as pd
from sklearn.metrics import f1_score

def fit(seed,protocol):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    rng=np.random.default_rng(seed)
    train=load_view('train');valid=load_view('valid')
    train_t=torch_view(train);valid_t=torch_view(valid)
    labels={s:load_labels(s) for s in ['train','valid']}
    y=torch.as_tensor(labels['train']['y'],device='cuda');c=torch.as_tensor(labels['train']['c'],device='cuda')
    model,old=load_model(P2/f'checkpoints/gated_aug_{seed}.pt')
    weight=(len(c)/(3*torch.bincount(c,minlength=3))).sqrt();weight/=weight.mean()
    opt=torch.optim.AdamW(model.parameters(),lr=protocol['lr'],weight_decay=protocol['weight_decay'])
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=protocol['epochs'],eta_min=3e-5)
    best=float('inf');patience=0;history=[]
    for epoch in range(protocol['epochs']+1):
        losses=[]
        if epoch:
            model.train();order=rng.permutation(len(y))
            for start in range(0,len(y),protocol['batch_size']):
                idx=order[start:start+protocol['batch_size']]
                mask=np.full(len(idx),7);sel=rng.random(len(idx))<protocol['coalition_probability'];mask[sel]=rng.integers(0,7,size=sel.sum())
                keep=(mask[:,None] & (1<<np.arange(3)))!=0
                batch_np=coalitional_view({k:v[idx] for k,v in train.items()},keep)
                batch=torch_view(batch_np)
                logits,pred,_=model(batch);ii=torch.as_tensor(idx,device='cuda')
                loss=torch.nn.functional.cross_entropy(logits,c[ii],weight=weight)+torch.nn.functional.smooth_l1_loss(pred,y[ii])
                opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);opt.step();losses.append(float(loss.detach()))
            scheduler.step()
        out=predictions([model],valid_t);m=metrics(labels['valid']['c'],labels['valid']['y'],out[:,:3],out[:,3]);score=m['mae']+1-m['macro_f1']
        history.append(dict(epoch=epoch,train_loss=float(np.mean(losses)) if losses else None,score=score,**m))
        if score<best-1e-5:
            best=score;patience=0
            ck=dict(variant='gated_aug',hidden=64,dropout=.2,seed=seed,epoch=epoch,score=score,
                state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()},initialization_sha256=sha256(P2/f'checkpoints/gated_aug_{seed}.pt'))
            torch.save(ck,ROOT/f'checkpoints/coalition_aug_{seed}.pt')
        else:patience+=1
        save_json(ROOT/f'results/history_{seed}.json',history)
        print(f'seed={seed} epoch={epoch} score={score:.5f} Acc={m["accuracy"]:.4f} F1={m["macro_f1"]:.4f} MAE={m["mae"]:.4f}',flush=True)
        if patience>=protocol['patience']:break
    save_json(ROOT/f'results/completed_{seed}.json',dict(seed=seed,score=best))

def lock_selection(protocol):
    valid=load_view('valid');labels=load_labels('valid');rows=[];scores={};paths={}
    for candidate in protocol['prediction_candidates']:
        if candidate.startswith('step2_'):
            paths[candidate]=[P2/f'checkpoints/{candidate[6:]}_{s}.pt' for s in protocol['seeds']]
        else:paths[candidate]=[ROOT/f'checkpoints/coalition_aug_{s}.pt' for s in protocol['seeds']]
        models=[load_model(p)[0] for p in paths[candidate]]
        out=predictions(models,valid,per_seed=True)
        for j,seed in enumerate(protocol['seeds']):rows.append(dict(model=candidate,seed=str(seed),**metrics(labels['c'],labels['y'],out[j,:,:3],out[j,:,3])))
        mean=out.mean(0);m=metrics(labels['c'],labels['y'],mean[:,:3],mean[:,3]);scores[candidate]=m['mae']+1-m['macro_f1']
        rows.append(dict(model=candidate,seed='ensemble',**m));del models
    selected=min(scores,key=scores.get);checkpoints={}
    for s,p in zip(protocol['seeds'],paths[selected]):
        dest=ROOT/f'checkpoints/final_{s}.pt';shutil.copy2(p,dest);checkpoints[dest.name]=sha256(dest)
    pd.DataFrame(rows).to_csv(ROOT/'results/validation_prediction_metrics.csv',index=False)
    save_json(ROOT/'selection_lock.json',dict(selected=selected,scores=scores,seeds=protocol['seeds'],checkpoints=checkpoints,
        selected_source_paths=[p.relative_to(PROJECT).as_posix() for p in paths[selected]],
        protocol_sha256=sha256(ROOT/'protocol.json'),criterion=protocol['checkpoint_selection'],special_predictions_seen=False))
    print('LOCKED',selected,scores,flush=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int);args=ap.parse_args()
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    protocol=json.loads((ROOT/'protocol.json').read_text())
    for seed in [args.seed] if args.seed else protocol['seeds']:
        if not ((ROOT/f'results/completed_{seed}.json').exists() and (ROOT/f'checkpoints/coalition_aug_{seed}.pt').exists()):fit(seed,protocol)
    if all((ROOT/f'results/completed_{s}.json').exists() for s in protocol['seeds']):lock_selection(protocol)

if __name__=='__main__':main()
