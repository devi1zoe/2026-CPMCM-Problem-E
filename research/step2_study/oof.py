"""Group-held-out predictions for learning fusion. No validation/test label fitting."""
from common import *
from data import *
from dynamic import epoch_masks,quality,context_key
from model import Model
from engine import predict
from sklearn.model_selection import StratifiedGroupKFold
from torch.nn import functional as F
import argparse,random,time,fcntl

class RawViews:
    def __init__(self,raw):
        self.base=adapt(raw);self.ids=raw['id'];self.encoder=None
        self.encoder_hash=json.loads((ROOT/'protocol.json').read_text())['encoder']['weights_sha256']
    def text(self,o):
        key=context_key(self.base,o,self.encoder_hash);path=ROOT/f'cache/raw_text_{key}.npy'
        with open(str(path)+'.lock','w') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            if not path.exists():
                if self.encoder is None:
                    with torch.random.fork_rng():self.encoder=TextEncoder()
                x=self.encoder.encode(self.base,o).astype(np.float16);tmp=Path(str(path)+'.tmp')
                with tmp.open('wb') as f:np.save(f,x)
                tmp.replace(path)
        return np.load(path)
    def get(self,seed,epoch):
        o,removed=epoch_masks(self.base,self.ids,seed,epoch) if epoch else (self.base['o'],np.zeros_like(self.base['o']))
        return dict(p=self.base['p'],o=o,removed=removed,quality=quality(self.base['p'],o),text=self.text(o),audio=self.base['audio'],vision=self.base['vision'])

def normalized(view,stats,idx):
    out={k:view[k][idx] for k in ['p','o','quality']}
    for m,name in enumerate(['text','audio','vision']):
        z=np.clip((view[name][idx].astype(np.float32)-stats[name+'_mean'])/stats[name+'_std'],-8,8)
        z[~out['o'][:,m]]=0;out[name]=z.astype(np.float16)
    return out

def partitions(raw):
    c=np.asarray(raw['classification_labels']).astype(int);groups=np.array([str(i).split('$_$')[0] for i in raw['id']])
    folds=list(StratifiedGroupKFold(3,shuffle=True,random_state=20260925).split(np.zeros(len(c)),c,groups))
    for tr,ho in folds:assert not set(groups[tr])&set(groups[ho])
    assert sorted(np.concatenate([ho for tr,ho in folds]).tolist())==list(range(len(c)))
    return folds,groups

def prepare(seed):
    protocol=json.loads((ROOT/'protocol.json').read_text());raw=load_attachment2()['train'];folds,groups=partitions(raw)
    c=np.asarray(raw['classification_labels']).astype(int);y=np.asarray(raw['regression_labels']);store=RawViews(raw)
    outpath=ROOT/f'cache/oof_{seed}.npz';donepath=ROOT/f'results/oof_completed_{seed}.json'
    if donepath.exists():
        done=json.loads(donepath.read_text());assert done['sha256']==sha256(outpath)
        assert all(sha256(ROOT/p)==h for p,h in done['expert_hashes'].items());return
    clean=store.get(seed,0);n=len(c);oof=np.full((9,n,3,4),np.nan,np.float32);fold_id=np.full(n,-1,int);hashes={};fold_rows=[]
    for fold,(tr,ho) in enumerate(folds+[(np.arange(n),np.empty(0,dtype=int))]):
        tag=str(fold) if fold<3 else 'full'
        if fold<3:
            stats=fit_stats({k:store.base[k][tr] for k in ['o','audio','vision']},clean['text'][tr])
            fold_id[ho]=fold
        else:stats=dict(np.load(ROOT/'checkpoints/normalization.npz'))
        statspath=ROOT/f'checkpoints/expert_normalization_{tag}.npz'
        with open(str(statspath)+'.lock','w') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            if statspath.exists():
                previous=dict(np.load(statspath));assert all(np.array_equal(stats[k],previous[k]) for k in stats)
            else:np.savez(statspath,**stats)
        counts=np.bincount(c[tr],minlength=3);prior=(counts/counts.sum()).tolist();mean=float(y[tr].mean())
        models=[];paths=[]
        for m in range(3):
            random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
            models.append(Model(dict(indices=[m]),prior,mean).cuda())
            paths.append(ROOT/f'checkpoints/expert_{seed}_{tag}_{m}.pt')
        marker=ROOT/f'results/experts_{seed}_{tag}.json'
        if marker.exists():
            entry=json.loads(marker.read_text());assert entry['train_ids_sha256']==hashlib.sha256('\n'.join(str(raw['id'][i]) for i in tr).encode()).hexdigest()
            for m,path in zip(models,paths):
                assert sha256(path)==entry['hashes'][path.name];m.load_state_dict(torch.load(path,map_location='cpu',weights_only=True)['state_dict'])
        else:
            opt=torch.optim.AdamW([p for m in models for p in m.parameters()],lr=protocol['lr'],weight_decay=protocol['weight_decay'])
            sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=8,eta_min=1e-4)
            tc=torch.as_tensor(c[tr],device='cuda');ty=torch.as_tensor(y[tr],device='cuda');cw=torch.as_tensor(np.sqrt(len(tr)/(3*counts)),dtype=torch.float32,device='cuda');cw/=cw.mean()
            start=time.time();history=[]
            for epoch in range(1,9):
                tv=torch_view(normalized(store.get(seed,epoch),stats,tr));order=np.random.default_rng(seed*1000+epoch).permutation(len(tr));total=0.;batches=0
                for model in models:model.train()
                for off in range(0,len(tr),64):
                    ii=torch.as_tensor(order[off:off+64],device='cuda');b=slice_view(tv,ii);opt.zero_grad(set_to_none=True);losses=[]
                    for model in models:
                        z=model(b);losses.append(F.cross_entropy(z['logits'],tc[ii],weight=cw)+2*F.smooth_l1_loss(z['pred'],ty[ii]))
                    loss=sum(losses);assert torch.isfinite(loss);loss.backward()
                    for model in models:torch.nn.utils.clip_grad_norm_(model.parameters(),5,error_if_nonfinite=True)
                    opt.step();total+=float(loss.detach());batches+=1
                sch.step();del tv;history.append(dict(epoch=epoch,loss=total/batches,elapsed_seconds=time.time()-start))
                print(f'OOF seed={seed} fold={tag} epoch={epoch}/8 loss={total/batches:.4f}',flush=True)
            for i,(model,path) in enumerate(zip(models,paths)):
                torch.save(dict(name='expert',seed=seed,config=dict(indices=[i]),prior=prior,mean=mean,state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()},
                    normalization_sha256=sha256(statspath),fold=tag,epochs=8),path)
            save_json(marker,dict(hashes={p.name:sha256(p) for p in paths},history=history,
                train_ids_sha256=hashlib.sha256('\n'.join(str(raw['id'][i]) for i in tr).encode()).hexdigest(),
                train_indices=tr.tolist(),heldout_indices=ho.tolist(),normalization_sha256=sha256(statspath),heldout_used_for_selection=False))
        hashes.update({str(p.relative_to(ROOT)):sha256(p) for p in paths})
        if fold<3:
            for epoch in range(9):
                tv=torch_view(normalized(clean if epoch==0 else store.get(seed,epoch),stats,ho))
                for m,model in enumerate(models):oof[epoch,ho,m]=predict(model,tv)
                del tv
        fold_rows.append(dict(fold=tag,train=len(tr),heldout=len(ho),source_overlap=len(set(groups[tr])&set(groups[ho])),normalization_sha256=sha256(statspath)))
        del models
    assert np.isfinite(oof).all() and np.all(fold_id>=0)
    qs=[];obs=[];ps=[]
    for epoch in range(9):
        view=clean if epoch==0 else store.get(seed,epoch);qs.append(view['quality']);obs.append(view['o']);ps.append(view['p'])
    np.savez_compressed(outpath,experts=oof,quality=np.stack(qs),o=np.stack(obs),p=np.stack(ps),c=c,y=y,fold_id=fold_id,ids=np.asarray(raw['id'],dtype=str))
    save_json(donepath,dict(sha256=sha256(outpath),expert_hashes=hashes,folds=fold_rows,training_views=9,
        clean_view_and_eight_dynamic_views=True,full_experts_same_fixed_8_epochs=True,protocol_sha256=sha256(ROOT/'protocol.json')))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int,required=True);args=ap.parse_args()
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
    with open(ROOT/f'cache/oof_{args.seed}.lock','w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);prepare(args.seed)

if __name__=='__main__':main()
