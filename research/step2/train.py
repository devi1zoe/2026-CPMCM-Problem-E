"""训练集更新参数；固定验证集场景早停；锁定模型后才允许测试评估。"""
from common import *
from data import torch_view, slice_view
from model import SentimentModel
import argparse
import random
import time
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score,f1_score,mean_absolute_error,recall_score

MAIN = ['fixed_clean','fixed_aug','mask_aug','gated_aug']


def metrics(c,y,prob,pred):
    cls = prob.argmax(1)
    pearson = float(np.corrcoef(y,pred)[0,1]) if np.std(pred)>1e-8 and np.std(y)>1e-8 else None
    f1 = f1_score(c,cls,labels=[0,1,2],average=None,zero_division=0)
    directional = cls!=1
    return dict(accuracy=float(accuracy_score(c,cls)),macro_f1=float(f1.mean()),
        weighted_f1=float(f1_score(c,cls,average='weighted',zero_division=0)),
        f1_negative=float(f1[0]),f1_neutral=float(f1[1]),f1_positive=float(f1[2]),
        neutral_recall=float(recall_score(c,cls,labels=[1],average=None,zero_division=0)[0]),
        mae=float(mean_absolute_error(y,pred)),pearson=pearson,
        nonneutral_sign_disagreement=float(np.mean((np.sign(pred[directional])+1)!=cls[directional])) if directional.any() else None)


@torch.inference_mode()
def predict(model,view,batch=256):
    model.eval()
    probs,ys,gates = [],[],[]
    for start in range(0,len(view['p']),batch):
        logits,y,g = model(slice_view(view,slice(start,start+batch)))
        probs.append(logits.softmax(-1).cpu().numpy()); ys.append(y.cpu().numpy()); gates.append(g.cpu().numpy())
    return np.concatenate(probs),np.concatenate(ys),np.concatenate(gates)


def load_cache(split,num):
    with np.load(ROOT/f'cache/{split}_{num}.npz') as f:
        return torch_view({k:f[k] for k in f.files if k!='removed'})


def selection_score(rows):
    scores = [r['mae']+1-r['macro_f1'] for r in rows]
    return .5*scores[0]+.1*sum(scores[1:])


def train_one(variant,seed,train_views,valid_views,labels,protocol):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    rng = np.random.default_rng(seed)
    model = SentimentModel(variant,protocol['hidden'],protocol['dropout']).cuda()
    optimizer = torch.optim.AdamW(model.parameters(),lr=protocol['lr'],weight_decay=protocol['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=protocol['epochs'],eta_min=1e-4)
    y,c = [torch.as_tensor(labels['train'][k],device='cuda') for k in ['y','c']]
    counts = torch.bincount(c,minlength=3).float()
    weights = (len(c)/(3*counts)).sqrt();weights/=weights.mean()
    augmented = variant.endswith('_aug')
    best,patience,history = float('inf'),0,[]
    begin = time.time()
    for epoch in range(1,protocol['epochs']+1):
        model.train()
        order = rng.permutation(len(y))
        losses = []
        for start in range(0,len(order),protocol['batch_size']):
            ii = order[start:start+protocol['batch_size']]
            if augmented:
                vv = rng.integers(1,7,size=len(ii));vv[rng.random(len(ii))<.25]=0
            else:
                vv = np.zeros(len(ii),int)
            idx = torch.as_tensor(ii,device='cuda');v = torch.as_tensor(vv,device='cuda')
            batch = {k:value[idx,v] for k,value in train_views.items()}
            logits,pred,_ = model(batch)
            loss = torch.nn.functional.cross_entropy(logits,c[idx],weight=weights)+torch.nn.functional.smooth_l1_loss(pred,y[idx])
            optimizer.zero_grad(set_to_none=True);loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
            optimizer.step();losses.append(float(loss.detach()))
        scheduler.step()
        rows = []
        for view in valid_views:
            prob,pred,_ = predict(model,view)
            rows.append(metrics(labels['valid']['c'],labels['valid']['y'],prob,pred))
        score = selection_score(rows)
        row = dict(epoch=epoch,loss=float(np.mean(losses)),score=score,clean=rows[0],corrupt_mae=float(np.mean([r['mae'] for r in rows[1:]])),corrupt_f1=float(np.mean([r['macro_f1'] for r in rows[1:]])))
        history.append(row)
        if score < best-1e-5:
            best,patience=score,0
            torch.save(dict(variant=variant,seed=seed,epoch=epoch,score=score,state_dict=model.cpu().state_dict(),hidden=protocol['hidden'],dropout=protocol['dropout']),ROOT/f'checkpoints/{variant}_{seed}.pt')
            model.cuda()
        else:
            patience+=1
        print(f'{variant} seed={seed} epoch={epoch} score={score:.4f} clean_acc={rows[0]["accuracy"]:.4f} clean_mae={rows[0]["mae"]:.4f}',flush=True)
        save_json(ROOT/f'results/history_{variant}_{seed}.json',history)
        if patience>=protocol['early_stopping_patience']:
            break
    print(f'FINISHED {variant} seed={seed} best={best:.4f} seconds={time.time()-begin:.1f}',flush=True)


def load_model(path):
    ckpt = torch.load(path,map_location='cpu',weights_only=True)
    model = SentimentModel(ckpt['variant'],ckpt['hidden'],ckpt['dropout']).cuda()
    model.load_state_dict(ckpt['state_dict'])
    return model.eval(),ckpt


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--variants',nargs='*');args=parser.parse_args()
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark=False
    torch.use_deterministic_algorithms(True)
    protocol=json.loads((ROOT/'protocol.json').read_text())
    labels={s:dict(np.load(ROOT/f'cache/{s}_labels.npz')) for s in ['train','valid']}
    tv=[load_cache('train',i) for i in range(7)]
    train_views={k:torch.stack([v[k] for v in tv],1) for k in tv[0]};del tv
    valid_views=[load_cache('valid',i) for i in range(6)]
    variants=args.variants or protocol['models']
    for variant in variants:
        for seed in protocol['seeds'] if variant in MAIN else [17]:
            done=ROOT/f'results/completed_{variant}_{seed}.json'
            if done.exists():
                continue
            train_one(variant,seed,train_views,valid_views,labels,protocol)
            save_json(done,dict(checkpoint=f'{variant}_{seed}.pt'))
    # Only lock after every predeclared fit has completed.
    if not all((ROOT/f'results/completed_{v}_{s}.json').exists() for v in protocol['models'] for s in (protocol['seeds'] if v in MAIN else [17])):
        return
    rows=[];ensemble_scores={}
    for variant in protocol['models']:
        seeds=protocol['seeds'] if variant in MAIN else [17]
        all_pred=[]
        for seed in seeds:
            model,ckpt=load_model(ROOT/f'checkpoints/{variant}_{seed}.pt')
            predictions=[predict(model,v)[:2] for v in valid_views]
            all_pred.append(predictions)
            for j,(prob,pred) in enumerate(predictions):
                rows.append(dict(model=variant,seed=str(seed),profile=protocol['validation_profiles'][j]['name'],epoch=ckpt['epoch'],**metrics(labels['valid']['c'],labels['valid']['y'],prob,pred)))
            del model
        ens_rows=[]
        for j in range(6):
            prob=np.mean([r[j][0] for r in all_pred],0);pred=np.mean([r[j][1] for r in all_pred],0)
            result=metrics(labels['valid']['c'],labels['valid']['y'],prob,pred)
            ens_rows.append(result)
            rows.append(dict(model=variant,seed='ensemble',profile=protocol['validation_profiles'][j]['name'],epoch=-1,**result))
        if variant in MAIN:
            ensemble_scores[variant]=selection_score(ens_rows)
    pd.DataFrame(rows).to_csv(ROOT/'results/validation_metrics.csv',index=False)
    selected=min(ensemble_scores,key=ensemble_scores.get)
    save_json(ROOT/'selection_lock.json',dict(selected=selected,scores=ensemble_scores,seeds=protocol['seeds'],criterion=protocol['selection'],
        checkpoint_sha256={f'{selected}_{s}.pt':sha256(ROOT/f'checkpoints/{selected}_{s}.pt') for s in protocol['seeds']},
        protocol_sha256=sha256(ROOT/'protocol.json'),test_metrics_seen=False))
    print('LOCKED',selected,ensemble_scores,flush=True)


if __name__=='__main__':
    main()
