"""Fixed evaluation cases, including historical masks and separate unequal cases."""
from common import *
from data import *
from dynamic import quality,context_key
from engine import scenarios
import itertools,fcntl

def profiles_all():
    rows=list(scenarios())
    for subset in ['TA','TV','AV','TAV']:
        rates=list(itertools.permutations([.1,.3,.7])) if subset=='TAV' else [(.1,.7),(.7,.1)]
        for values in rates:
            mapping=dict(zip(subset,values))
            for rep in range(3):
                rows.append(dict(name=subset+'_unequal_'+'_'.join(str(int(v*100)) for v in values),subset=subset,rate=-1.,rates=mapping,
                    position='random',pattern='unequal',sync=False,repeat=rep,primary=False))
    assert len(rows)==253
    return rows

def get_masks(base,profile,seed):
    if 'rates' not in profile:return corrupt(base,profile,seed)
    rng=np.random.default_rng(seed);removed=np.zeros_like(base['o'])
    for i,p in enumerate(base['p']):
        slots=np.flatnonzero(p);length=len(slots)
        for m,name in enumerate('TAV'):
            if name not in profile['subset']:continue
            count=min(length,max(1,int(np.floor(length*profile['rates'][name]+.5)))) if length else 0
            idx=intervals(length,count,profile['position'],rng)
            removed[i,m,slots[idx]]=True
    return base['o']&~removed,removed

class EvaluationViews:
    def __init__(self,raw,split,protocol):
        self.base=adapt(raw);self.split=split;self.protocol=protocol;self.encoder=None
        self.stats=dict(np.load(ROOT/'checkpoints/normalization.npz'))
    def get(self,profile):
        p=self.protocol
        stamp=hashlib.sha256(json.dumps(dict(split=self.split,profile=profile,source=p['source_sha256'],encoder=p['encoder'],
            stats=p['normalization_sha256'],seed=p['evaluation_seed']),sort_keys=True).encode()).hexdigest()[:20]
        historical=PROJECT/f'step2_v2/cache/eval_{self.split}_{stamp}.npz'
        path=ROOT/f'cache/eval_{self.split}_{stamp}.npz'
        if historical.exists():
            with np.load(historical) as f:return {k:f[k] for k in f.files}
        with open(str(path)+'.lock','w') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            if not path.exists():
                if profile['name']=='clean' and self.split=='valid':view=dict(np.load(ROOT/'cache/valid_0.npz'))
                else:
                    if self.encoder is None:
                        with torch.random.fork_rng():self.encoder=TextEncoder()
                    o,removed=get_masks(self.base,profile,p['evaluation_seed']+profile['repeat'])
                    text=self.encoder.encode(self.base,o)
                    view=dict(p=self.base['p'],o=o,removed=removed,quality=quality(self.base['p'],o))
                    for m,(name,x) in enumerate(zip(['text','audio','vision'],[text,self.base['audio'],self.base['vision']])):
                        z=np.clip((x-self.stats[name+'_mean'])/self.stats[name+'_std'],-8,8);z[~o[:,m]]=0;view[name]=z.astype(np.float16)
                temp=Path(str(path)+'.tmp')
                with temp.open('wb') as f:np.savez(f,**view)
                temp.replace(path)
        with np.load(path) as f:return {k:f[k] for k in f.files}

    @torch.inference_mode()
    def for_model(self,model,view,checkpoint_hash):
        if not hasattr(model,'language'):return view
        key=context_key(self.base,view['o'],checkpoint_hash+self.protocol['normalization_sha256'])
        path=ROOT/f'cache/adapted_{self.split}_{key}.npy'
        if path.exists():z=np.load(path)
        else:
            model.eval();raw={k:self.base[k] for k in ['tokens','attention','p']};raw['o']=view['o'];chunks=[]
            for start in range(0,len(view['p']),64):
                b=torch_view({k:v[start:start+64] for k,v in raw.items()});chunks.append(model.language(b).cpu().numpy())
            z=np.concatenate(chunks);temp=Path(str(path)+'.tmp')
            with temp.open('wb') as f:np.save(f,z)
            temp.replace(path)
        assert z.shape==(len(view['p']),50,768) and np.all(z[~view['o'][:,0]]==0)
        return {**{k:v for k,v in view.items() if k!='text'},'adapted_text':z}
