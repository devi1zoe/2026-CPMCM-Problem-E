from common import *
from data import *
from model import Model
from legacy_model import SentimentModel
from sklearn.metrics import f1_score,recall_score


def metrics(c,y,out):
    pred=out[:,3];cls=out[:,:3].argmax(-1)
    fs=f1_score(c,cls,labels=[0,1,2],average=None,zero_division=0)
    result=dict(accuracy=float((c==cls).mean()),macro_f1=float(fs.mean()),
        weighted_f1=float(f1_score(c,cls,average='weighted',zero_division=0)),mae=float(np.abs(y-pred).mean()),
        pearson=float(np.corrcoef(y,pred)[0,1]) if pred.std()>1e-8 else None,
        neutral_recall=float((cls[c==1]==1).mean()),**{f'f1_{s}':float(v) for s,v in zip(['negative','neutral','positive'],fs)})
    for s,mask in [('strong_negative',y<=-1.5),('strong_positive',y>=1.5)]:
        result[s+'_mae']=float(np.abs(pred[mask]-y[mask]).mean()) if mask.any() else None
    return result


def score(rows):
    return .5*(rows[0]['mae']+1-rows[0]['macro_f1'])+.1*sum(r['mae']+1-r['macro_f1'] for r in rows[1:])


def load_model(name,seed,device='cuda'):
    proto=json.loads((ROOT/'protocol.json').read_text())
    if name=='old_gated':
        path=ROOT/f'checkpoints/old_gated_{seed}.pt';ck=torch.load(path,map_location='cpu',weights_only=True)
        model=SentimentModel(ck['variant'],ck['hidden'],ck['dropout'])
    else:
        path=ROOT/f'checkpoints/{name}_{seed}.pt';ck=torch.load(path,map_location='cpu',weights_only=True)
        assert ck['protocol_sha256']==sha256(ROOT/'protocol.json')
        model=Model(name,proto['prior'],proto['mean'],proto['hidden'],proto['heads'],proto['dropout'])
    model.load_state_dict(ck['state_dict']);return model.to(device).eval(),ck


@torch.inference_mode()
def predict(model,view,batch=128):
    model.eval();out=[];proto=json.loads((ROOT/'protocol.json').read_text())
    for start in range(0,len(view['p']),batch):
        b=slice_view(view,slice(start,start+batch));r=model(b)
        if isinstance(r,dict):logits,pred=r['logits'],r['pred']
        else:
            logits,pred,_=r;empty=~b['o'].any((1,2))
            logits=torch.where(empty[:,None],torch.tensor(proto['prior'],device=pred.device).log()[None],logits)
            pred=torch.where(empty,torch.tensor(proto['mean'],device=pred.device),pred)
        out.append(torch.cat([logits.softmax(-1),pred[:,None]],-1).cpu().numpy())
    return np.concatenate(out)


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
