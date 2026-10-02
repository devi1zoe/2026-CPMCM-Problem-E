"""Deterministic ID/epoch masks; cached contexts keyed by actual tokens, attention, and encoder hash."""
from common import *
from data import *
import fcntl

def quality(p,o):
    lengths=np.maximum(p.sum(-1),1);run=np.zeros(o.shape[:2],np.int32);longest=run.copy()
    for t in range(p.shape[1]):
        run=(run+1)*(p[:,t,None]&~o[:,:,t]);longest=np.maximum(longest,run)
    return np.stack([o.sum(-1)/lengths[:,None],longest/lengths[:,None],np.broadcast_to(lengths[:,None]/48,(len(p),3))],-1).astype(np.float32)

def epoch_masks(base,ids,seed,epoch):
    removed=np.zeros_like(base['o'])
    for i,sid in enumerate(ids):
        digest=hashlib.sha256(f'{seed}/{epoch}/{sid}'.encode()).digest()
        rng=np.random.default_rng(int.from_bytes(digest[:8],'little'))
        if rng.random()<.5:continue
        subset=rng.choice(['T','A','V','TA','TV','AV','TAV']);rate=float(rng.choice([.1,.3,.5,.7]))
        position=rng.choice(['start','middle','end','random']);sync=bool(rng.integers(2))
        slots=np.flatnonzero(base['p'][i]);length=len(slots)
        count=min(length,max(1,int(np.floor(length*rate+.5)))) if length else 0
        idx=intervals(length,count,position,rng)
        for m,name in enumerate('TAV'):
            if name in subset:
                if not sync:idx=intervals(length,count,position,rng)
                removed[i,m,slots[idx]]=True
    return base['o']&~removed,removed

def context_key(base,o,encoder_hash):
    h=hashlib.sha256(encoder_hash.encode())
    for x in [base['tokens'],base['attention'],base['p'],o[:,0]]:h.update(np.ascontiguousarray(x).tobytes())
    return h.hexdigest()

class EpochStore:
    def __init__(self,raw,encoder=None):
        self.base=adapt(raw);self.ids=raw['id'];self.encoder=encoder
        self.stats=dict(np.load(ROOT/'checkpoints/normalization.npz'))
        self.clean=dict(np.load(ROOT/'cache/train_0.npz'))
        self.encoder_hash=json.loads((ROOT/'protocol.json').read_text())['encoder']['weights_sha256']

    def get(self,seed,epoch):
        o,removed=epoch_masks(self.base,self.ids,seed,epoch)
        key=context_key(self.base,o,self.encoder_hash)
        path=ROOT/f'cache/epoch_{seed}_{epoch}_{key[:16]}.npz'
        with open(str(path)+'.lock','w') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            if not path.exists():
                view={k:v.copy() for k,v in self.clean.items()}
                view.update(o=o,removed=removed,quality=quality(self.base['p'],o))
                changed=np.flatnonzero(np.any(o[:,0]!=self.base['o'][:,0],axis=1))
                if len(changed):
                    if self.encoder is None:self.encoder=TextEncoder()
                    b={k:self.base[k][changed] for k in ['tokens','attention','p']}
                    text=self.encoder.encode(b,o[changed])
                    z=np.clip((text-self.stats['text_mean'])/self.stats['text_std'],-8,8);z[~o[changed,0]]=0
                    view['text'][changed]=z.astype(np.float16)
                for m,name in enumerate(['audio','vision'],1):view[name][~o[:,m]]=0
                temp=Path(str(path)+'.tmp')
                with temp.open('wb') as f:np.savez(f,**view)
                temp.replace(path)
                save_json(Path(str(path)+'.json'),dict(seed=seed,epoch=epoch,context_key=key,encoder_hash=self.encoder_hash,
                    clean_samples=int((~removed.any((1,2))).sum()),changed_text_samples=len(changed),mask_sha256=hashlib.sha256(removed.tobytes()).hexdigest()))
                print(f'Encoded dynamic view seed={seed} epoch={epoch} text={len(changed)}',flush=True)
        with np.load(path) as f:return {k:f[k] for k in f.files if k!='removed'}
