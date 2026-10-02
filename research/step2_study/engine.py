from common import *
from data import torch_view,slice_view
from model import Model,load_study_state
from legacy_model import SentimentModel
import numpy as np
import torch
from sklearn.metrics import accuracy_score,f1_score,mean_absolute_error,recall_score

def metrics(c,y,prob,pred):
    cls=prob.argmax(-1)
    f1=f1_score(c,cls,labels=[0,1,2],average=None,zero_division=0)
    out=dict(accuracy=float(accuracy_score(c,cls)),macro_f1=float(f1.mean()),weighted_f1=float(f1_score(c,cls,average='weighted',zero_division=0)),
        mae=float(mean_absolute_error(y,pred)),pearson=float(np.corrcoef(y,pred)[0,1]) if np.std(pred)>1e-8 and np.std(y)>1e-8 else None,
        neutral_recall=float(recall_score(c,cls,labels=[1],average=None,zero_division=0)[0]))
    out.update({f'f1_{n}':float(v) for n,v in zip(['negative','neutral','positive'],f1)})
    for name,mask in [('strong_negative',y<=-1.5),('strong_positive',y>=1.5)]:
        out[name+'_mae']=float(np.abs(y[mask]-pred[mask]).mean()) if mask.any() else None
    return out

def score(rows):return .5*(rows[0]['mae']+1-rows[0]['macro_f1'])+.1*sum(r['mae']+1-r['macro_f1'] for r in rows[1:])

def load_model(path,device='cuda'):
    ck=torch.load(path,map_location='cpu',weights_only=True)
    if ck.get('config',{}).get('fusion'):
        from fusion import FusionModel,FusionPredictor
        gate=FusionModel(ck['config'],ck['prior'],ck['mean']);gate.load_state_dict(ck['state_dict'])
        experts=[]
        for p,h in ck['expert_hashes'].items():
            assert sha256(ROOT/p)==h
            experts.append(load_model(ROOT/p,device)[0].requires_grad_(False))
        return FusionPredictor(gate,experts).to(device).eval(),ck
    if 'config' in ck:
        model=Model(ck['config'],ck['prior'],ck['mean'])
    else:model=SentimentModel(ck['variant'],ck['hidden'],ck['dropout'])
    if 'config' in ck:load_study_state(model,ck['state_dict'])
    else:model.load_state_dict(ck['state_dict'])
    if 'config' not in ck:
        protocol=json.loads((ROOT/'protocol.json').read_text())
        model.register_buffer('fallback_prior',torch.tensor(protocol['prior'],dtype=torch.float32))
        model.register_buffer('fallback_mean',torch.tensor(protocol['mean'],dtype=torch.float32))
    return model.to(device).eval(),ck

@torch.inference_mode()
def _predict(model,view,batch=256,auxiliary=False):
    model.eval();outputs=[];aux=[]
    for start in range(0,len(view['p']),batch):
        b=slice_view(view,slice(start,start+batch))
        if isinstance(b['p'],np.ndarray):b=torch_view(b)
        r=model(b)
        if isinstance(r,dict):
            logits,pred=r['logits'],r['pred']
            if auxiliary and r['aux'] is not None:aux.append(r['aux'].cpu().numpy())
        else:
            logits,pred,_=r
            empty=~b['o'].any(dim=(1,2))
            logits=torch.where(empty[:,None],model.fallback_prior.log()[None],logits)
            pred=torch.where(empty,model.fallback_mean,pred)
        outputs.append(torch.cat([logits.softmax(-1),pred[:,None]],-1).cpu().numpy())
    out=np.concatenate(outputs)
    return (out,np.concatenate(aux) if aux else None) if auxiliary else out

def predict(model,view,batch=256,auxiliary=False,backend='cudnn'):
    if backend=='linear':
        from inference_ops import linear_convolutions
        with linear_convolutions(model):return _predict(model,view,batch,auxiliary)
    assert backend=='cudnn'
    return _predict(model,view,batch,auxiliary)

def scenarios():
    yield dict(name='clean',subset='',rate=0.,position='none',pattern='single',sync=True,repeat=0,primary=False)
    for subset in ['T','A','V','TA','TV','AV','TAV']:
        for rate in [.1,.3,.5,.7]:
            for pos in ['start','middle','end','random']:
                for rep in range(3) if pos=='random' else [0]:
                    yield dict(name=f'{subset}_r{int(rate*100)}_{pos}',subset=subset,rate=rate,position=pos,pattern='single',sync=True,repeat=rep,primary=True)
    for subset in ['T','A','V','TAV']:
        for rate in [.3,.5]:
            for rep in range(3):yield dict(name=f'{subset}_r{int(rate*100)}_multi',subset=subset,rate=rate,position='random',pattern='multi',sync=True,repeat=rep,primary=False)
    for subset in ['TA','TV','AV','TAV']:
        for rate in [.3,.5]:
            for rep in range(3):yield dict(name=f'{subset}_r{int(rate*100)}_staggered',subset=subset,rate=rate,position='random',pattern='single',sync=False,repeat=rep,primary=False)

def checkpoint_paths(name):
    if name=='expert_text':return [ROOT/'checkpoints'/f'expert_{s}_full_0.pt' for s in [17,29,43]]
    return [(PROJECT/'step2' if name=='old_gated' else ROOT)/'checkpoints'/f'{"gated_aug" if name=="old_gated" else name}_{s}.pt' for s in [17,29,43]]
