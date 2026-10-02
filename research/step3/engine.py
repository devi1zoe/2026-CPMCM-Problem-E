"""预测、缺失操作与三模态合作博弈。固定原预测类别作为解释目标。"""
from common import *
from data import torch_view,slice_view
from model import SentimentModel
import numpy as np
import torch
from sklearn.metrics import accuracy_score,f1_score,mean_absolute_error,recall_score

NAMES=['text','audio','vision']

def metrics(c,y,prob,pred):
    cls=prob.argmax(1);f=f1_score(c,cls,labels=[0,1,2],average=None,zero_division=0)
    return dict(accuracy=float(accuracy_score(c,cls)),macro_f1=float(f.mean()),mae=float(mean_absolute_error(y,pred)),
        pearson=float(np.corrcoef(y,pred)[0,1]) if np.std(pred)>1e-8 else None,
        f1_negative=float(f[0]),f1_neutral=float(f[1]),f1_positive=float(f[2]),
        neutral_recall=float(recall_score(c,cls,labels=[1],average=None,zero_division=0)[0]))

def load_model(path):
    ck=torch.load(path,map_location='cpu',weights_only=True)
    m=SentimentModel(ck['variant'],ck['hidden'],ck['dropout']).cuda().eval();m.load_state_dict(ck['state_dict'])
    return m,ck

def load_view(split):
    path=ROOT/f'cache/{split}_clean.npz'
    if not path.exists():path=P2/f'cache/{split}_0.npz'
    with np.load(path) as z:
        return {k:z[k] for k in ['text','audio','vision','p','o','quality']}

def load_labels(split):
    path=ROOT/f'cache/{split}_labels.npz'
    if not path.exists():path=P2/f'cache/{split}_labels.npz'
    return dict(np.load(path))

def quality(p,o):
    lengths=np.maximum(p.sum(-1),1);run=np.zeros(o.shape[:2],int);longest=run.copy()
    for t in range(50):run=(run+1)*(p[:,t,None]&~o[:,:,t]);longest=np.maximum(longest,run)
    return np.stack([o.sum(-1)/lengths[:,None],longest/lengths[:,None],np.broadcast_to(lengths[:,None]/48,o.shape[:2])],-1).astype(np.float32)

def coalitional_view(view,keep,reference=None):
    b={k:v.copy() for k,v in view.items()}
    for j,name in enumerate(NAMES):
        absent=~np.asarray(keep)[:,j]
        if reference is None:
            b[name][absent]=0;b['o'][absent,j]=False
        else:
            b[name][absent]=reference[name][None,None,:]*b['o'][absent,j,:,None]
    if reference is None:b['quality']=quality(b['p'],b['o'])
    return b

@torch.inference_mode()
def predictions(models,view,batch=256,per_seed=False):
    b=torch_view(view) if isinstance(view['p'],np.ndarray) else view
    seed_out=[]
    for m in models:
        m.eval();out=[]
        for start in range(0,len(b['p']),batch):
            logits,y,_=m(slice_view(b,slice(start,start+batch)))
            out.append(np.c_[logits.softmax(-1).cpu().numpy(),y.cpu().numpy()])
        seed_out.append(np.concatenate(out))
    seed_out=np.stack(seed_out)
    return seed_out if per_seed else seed_out.mean(0)

def coalition_values(models,view,reference=None):
    values=[]
    for mask in range(8):
        keep=np.broadcast_to([bool(mask&(1<<j)) for j in range(3)],(len(view['p']),3))
        values.append(predictions(models,coalitional_view(view,keep,reference),per_seed=True))
    return np.stack(values,axis=2) # seeds,N,8,(3 probabilities + intensity)

def shapley(values,target):
    # values: N,8,4; target是完整输入下固定的原预测类别。
    n=len(values);scores=values[np.arange(n)[:,None],np.arange(8)[None,:],target[:,None]]
    y=values[:,:,3];phi=np.zeros((n,3,2),np.float64)
    for j in range(3):
        for mask in range(8):
            if mask&(1<<j):continue
            size=mask.bit_count();weight=[1/3,1/6,1/3][size]
            phi[:,j,0]+=weight*(scores[:,mask|(1<<j)]-scores[:,mask])
            phi[:,j,1]+=weight*(y[:,mask|(1<<j)]-y[:,mask])
    residual=phi.sum(1)-np.c_[scores[:,7]-scores[:,0],y[:,7]-y[:,0]]
    assert np.max(np.abs(residual))<2e-6
    return phi,residual

def selected_models():
    lock=json.loads((ROOT/'selection_lock.json').read_text())
    out=[]
    for file,digest in lock['checkpoints'].items():
        path=ROOT/'checkpoints'/file;assert sha256(path)==digest
        out.append(load_model(path)[0])
    return out,lock
