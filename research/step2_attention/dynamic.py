"""Epoch-wise raw-token deletion. Cache at most two views in RAM, no expanding disk cache."""
from common import *
from data import *
from collections import OrderedDict


def quality(p,o):
    lengths=np.maximum(p.sum(-1),1);run=np.zeros(o.shape[:2],np.int32);longest=run.copy()
    for t in range(p.shape[1]):run=(run+1)*(p[:,t,None]&~o[:,:,t]);longest=np.maximum(longest,run)
    return np.stack([o.sum(-1)/lengths[:,None],longest/lengths[:,None],np.broadcast_to(lengths[:,None]/48,(len(p),3))],-1).astype(np.float32)


def epoch_masks(base,ids,seed,epoch):
    removed=np.zeros_like(base['o'])
    for i,sid in enumerate(ids):
        rng=np.random.default_rng(int.from_bytes(hashlib.sha256(f'{seed}/{epoch}/{sid}'.encode()).digest()[:8],'little'))
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


class EpochStore:
    def __init__(self,raw):
        self.base=adapt(raw);self.ids=list(raw['id']);self.stats=dict(np.load(ROOT/'checkpoints/normalization.npz'))
        self.clean=dict(np.load(ROOT/'cache/train_0.npz'));self.encoder=None;self.memory=OrderedDict()
        self.encoder_hash=json.loads((ROOT/'protocol.json').read_text())['encoder']['weights_sha256']

    def get(self,seed,epoch):
        if epoch in self.memory:return self.memory[epoch]
        o,removed=epoch_masks(self.base,self.ids,seed,epoch)
        h=hashlib.sha256(self.encoder_hash.encode())
        for x in [self.base['tokens'],self.base['attention'],self.base['p'],o[:,0]]:h.update(np.ascontiguousarray(x).tobytes())
        # Optional read-only reuse of exactly the same v2 input realization, never its model outputs.
        existing=PROJECT/f'step2_v2/cache/epoch_{seed}_{epoch}_{h.hexdigest()[:16]}.npz'
        prior_norm=PROJECT/'step2_v2/checkpoints/normalization.npz'
        compatible=prior_norm.is_file() and sha256(prior_norm)==sha256(ROOT/'checkpoints/normalization.npz')
        if existing.is_file() and compatible:
            view=dict(np.load(existing));assert np.array_equal(view['o'],o) and np.array_equal(view['removed'],removed)
            mode='read_only_verified_v2_view'
        else:
            view={k:v.copy() for k,v in self.clean.items()};view.update(o=o,removed=removed,quality=quality(self.base['p'],o))
            changed=np.flatnonzero(np.any(o[:,0]!=self.base['o'][:,0],axis=1))
            if len(changed):
                if self.encoder is None:self.encoder=TextEncoder()
                sub={k:self.base[k][changed] for k in ['tokens','attention','p']}
                text=self.encoder.encode(sub,o[changed]);z=np.clip((text-self.stats['text_mean'])/self.stats['text_std'],-8,8);z[~o[changed,0]]=0
                view['text'][changed]=z.astype(np.float16)
            for j,name in enumerate(['audio','vision'],1):view[name][~o[:,j]]=0
            mode='reencoded'
        save_json(ROOT/f'results/mask_{seed}_{epoch}.json',dict(seed=seed,epoch=epoch,context_hash=h.hexdigest(),mask_sha256=hashlib.sha256(removed.tobytes()).hexdigest(),mode=mode))
        view={k:v for k,v in view.items() if k!='removed'};self.memory[epoch]=view
        while len(self.memory)>2:self.memory.popitem(last=False)
        return view
