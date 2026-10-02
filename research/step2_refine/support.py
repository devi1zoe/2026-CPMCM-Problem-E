from common import *
from data import *
from engine import predict
from model import LateFusion
import fcntl


def expert_paths(seed):
    return [PROJECT/f'step2_v2/checkpoints/{n}_{seed}.pt' for n in ['text','audio','vision']]


def load_experts(model,seed):
    hashes={}
    for m,p in zip(model.experts,expert_paths(seed)):
        ck=torch.load(p,map_location='cpu',weights_only=True);m.load_state_dict(ck['state_dict']);m.eval().requires_grad_(False)
        hashes[str(p.relative_to(PROJECT))]=sha256(p)
    return hashes


def state_hash(model):
    h=hashlib.sha256()
    for k,v in sorted(model.state_dict().items()):h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def expert_view(model,view,hashes):
    """Unlabeled cache binds all actual normalized inputs, masks and asset versions."""
    protocol=json.loads((ROOT/'protocol.json').read_text())
    meta=dict(experts=hashes,encoder=protocol['encoder'],normalization=protocol['normalization_sha256'],source=protocol['source_sha256'],backend='cudnn')
    h=hashlib.sha256(json.dumps(meta,sort_keys=True).encode())
    for name in ['text','audio','vision','p','o','quality']:
        a=view[name].detach().cpu().numpy() if isinstance(view[name],torch.Tensor) else view[name]
        h.update(name.encode());h.update(str(a.dtype).encode());h.update(np.asarray(a.shape,np.int64).tobytes());h.update(np.ascontiguousarray(a).tobytes())
    key=h.hexdigest();path=ROOT/f'cache/experts_{key}.npz'
    with open(str(path)+'.lock','w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if not path.exists():
            e=np.stack([predict(m,view,backend='cudnn') for m in model.experts],1)
            with Path(str(path)+'.tmp').open('wb') as f:np.savez(f,out=e)
            Path(str(path)+'.tmp').replace(path);save_json(Path(str(path)+'.json'),dict(**meta,input_sha256=key))
    out=dict(o=view['o'],p=view['p'],quality=view['quality'],expert_outputs=np.load(path)['out'])
    return {k:v.detach().cpu().numpy() if isinstance(v,torch.Tensor) else v for k,v in out.items()}


def train_subset_indices():
    return np.random.default_rng(17).permutation(3395)[:512]


def grouped_metrics(c,y,out):
    from engine import metrics
    groups=[('all',np.ones(len(y),bool)),('strong_negative',y<=-1.5),('weak_negative',(y<0)&(y>-1.5)),('neutral',y==0),('weak_positive',(y>0)&(y<1.5)),('strong_positive',y>=1.5)]
    rows=[]
    for name,keep in groups:
        if not keep.any():continue
        rows.append(dict(group=name,n=int(keep.sum()),**metrics(c[keep],y[keep],out[keep,:3],out[keep,3]),
            true_mean=float(y[keep].mean()),pred_mean=float(out[keep,3].mean()),saturation_fraction=float((np.abs(out[keep,3])>=2.9).mean()),
            pred_q10=float(np.quantile(out[keep,3],.1)),pred_q50=float(np.quantile(out[keep,3],.5)),pred_q90=float(np.quantile(out[keep,3],.9))))
    return rows
