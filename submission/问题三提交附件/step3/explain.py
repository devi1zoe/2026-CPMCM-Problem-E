# 本程序及代码是在人工智能工具辅助下完成的。

# 人工智能工具名称：GPT-6 Astra

# 版本/型号：gpt-6-astra

# 开发机构/公司：OpenAI

# 版本颁布日期：2026年9月3日

"""精确模态Shapley + 表示层IG筛选 + 原输入局部删除/保留验证。"""
from common import *
from engine import *
from data import TextEncoder,adapt

def dominant(phi):
    absolute=np.abs(phi[:,:,0]);total=absolute.sum(1)
    share=absolute/np.maximum(total[:,None],1e-12)
    index=share.argmax(1)
    status=np.where(total<.001,'weak',np.where(share.max(1)<.45,'mixed','dominant'))
    return index,share,status


def integrated_gradients(models,view,target,nodes=32,reference=None,batch=8):
    """Gauss-Legendre路径积分。O和quality固定，只解释连续表示，不冒充原词删除。"""
    pts,weights=np.polynomial.legendre.leggauss(nodes)
    pts=torch.tensor((pts+1)/2,device='cuda',dtype=torch.float32)
    weights=torch.tensor(weights/2,device='cuda',dtype=torch.float32)
    results=[];residuals=[]
    for start in range(0,len(target),batch):
        idx=np.arange(start,min(start+batch,len(target)))
        v=torch_view({k:x[idx] for k,x in view.items()})
        n=len(idx);b={k:x.repeat_interleave(nodes,dim=0) for k,x in v.items() if k not in NAMES}
        originals=[];baselines=[];inputs=[]
        for j,name in enumerate(NAMES):
            x=v[name].float()
            ref=torch.zeros_like(x) if reference is None else torch.as_tensor(reference[name],device='cuda')[None,None,:].expand_as(x)*v['o'][:,j,:,None]
            path=ref[:,None]+pts[None,:,None,None]*(x-ref)[:,None]
            path=path.reshape(n*nodes,50,x.shape[-1]).detach().requires_grad_(True)
            b[name]=path;inputs.append(path);originals.append(x);baselines.append(ref)
        targets=torch.as_tensor(np.repeat(target[idx],nodes),device='cuda')
        score=0
        for model in models:
            model.eval();logits,_,_=model(b)
            score=score+logits.softmax(-1).gather(1,targets[:,None]).squeeze(1)/len(models)
        gradients=torch.autograd.grad(score.sum(),inputs)
        attr=[]
        for grad,x,ref in zip(gradients,originals,baselines):
            integral=(grad.reshape(n,nodes,50,-1)*weights[None,:,None,None]).sum(1)
            attr.append(((x-ref)*integral).sum(-1).detach().cpu().numpy())
        attr=np.stack(attr,1)
        orig={k:x[idx] for k,x in view.items()}
        baseline={k:x.copy() for k,x in orig.items()}
        for j,name in enumerate(NAMES):baseline[name]=baselines[j].cpu().numpy()
        full=predictions(models,orig);zero=predictions(models,baseline)
        delta=full[np.arange(n),target[idx]]-zero[np.arange(n),target[idx]]
        residuals.append(attr.sum((1,2))-delta);results.append(attr)
    return np.concatenate(results),np.concatenate(residuals)


def windows(p,fraction=.2):
    slots=np.flatnonzero(p)
    if not len(slots):return [(0,0)]
    k=min(len(slots),max(1,int(np.floor(len(slots)*fraction+.5))))
    return [(int(slots[s]),int(slots[s+k-1])+1) for s in range(len(slots)-k+1)]


def rank_windows(ig,p,fraction=.2):
    chosen=[];bank=[]
    for i in range(len(p)):
        row=[];candidates=windows(p[i],fraction);bank.append(candidates)
        for j in range(3):
            scores=[float(ig[i,j,a:b].sum()) for a,b in candidates]
            row.append(candidates[int(np.argmax(scores))])
        chosen.append(row)
    return np.asarray(chosen),bank


def refine_integrals(models,view,target,ig,residual,reference=None):
    """按完备性恒等式检查数值精度，不使用真实标签或删除成绩调解释。"""
    attr=ig.copy();res=residual.copy();used=np.full(len(target),32,int)
    for nodes in [64,128,256]:
        idx=np.flatnonzero(abs(res)>.001)
        if not len(idx):break
        subset={k:v[idx] for k,v in view.items()}
        new,error=integrated_gradients(models,subset,target[idx],nodes=nodes,reference=reference,batch=4 if nodes>=128 else 8)
        attr[idx]=new;res[idx]=error;used[idx]=nodes
    return attr,res,used


class ProbeEngine:
    """需要删除文本时始终重算冻结编码，其他模态不借文本替代回填。"""
    def __init__(self,models,raw,clean,encoder=None):
        self.models=models;self.base=adapt(raw);self.clean=clean
        self.stats=dict(np.load(ROOT/'checkpoints/normalization.npz'))
        self.encoder=encoder or TextEncoder()

    def probe(self,specs,batch=128,per_seed=False):
        # spec=(sample index, modality index, start slot, end slot, 'drop'/'keep'/'only')
        results=[]
        for begin in range(0,len(specs),batch):
            part=specs[begin:begin+batch];idx=np.array([s[0] for s in part],int)
            v={k:x[idx].copy() for k,x in self.clean.items()}
            for r,(_,m,a,b,mode) in enumerate(part):
                if mode=='drop':v['o'][r,m,a:b]=False
                elif mode in ['keep','only']:
                    if mode=='only':
                        for other in range(3):
                            if other!=m:v['o'][r,other]=False
                    v['o'][r,m,:a]=False;v['o'][r,m,b:]=False
                else:raise ValueError(mode)
            changed=np.any(v['o'][:,0]!=self.base['o'][idx,0],axis=1)
            encode_idx=np.flatnonzero(changed & v['o'][:,0].any(1))
            if len(encode_idx):
                base={k:x[idx[encode_idx]] for k,x in self.base.items() if isinstance(x,np.ndarray)}
                text=self.encoder.encode(base,v['o'][encode_idx])
                norm=np.clip((text-self.stats['text_mean'])/self.stats['text_std'],-8,8)
                norm[~v['o'][encode_idx,0]]=0
                v['text'][encode_idx]=norm.astype(np.float16)
            for m,name in enumerate(NAMES):v[name][~v['o'][:,m]]=0
            v['quality']=quality(v['p'],v['o'])
            results.append(predictions(self.models,v,per_seed=per_seed))
        return np.concatenate(results,axis=1 if per_seed else 0)


def modality_records(ids,values,refvalues,targets):
    phi,res=shapley(values.mean(0),targets)
    phi_ref,_=shapley(refvalues.mean(0),targets)
    per=np.stack([shapley(v,targets)[0] for v in values])
    main,shares,status=dominant(phi);mainref,_,_=dominant(phi_ref)
    per_main=np.stack([dominant(p)[0] for p in per])
    rows=[]
    for i,id_ in enumerate(ids):
        for m,name in enumerate(NAMES):
            rows.append(dict(sample_id=str(id_),modality=name,target_class=int(targets[i]),
                class_shapley=float(phi[i,m,0]),intensity_shapley=float(phi[i,m,1]),absolute_share=float(shares[i,m]),
                class_shapley_seed_std=float(per[:,i,m,0].std(ddof=1)),intensity_shapley_seed_std=float(per[:,i,m,1].std(ddof=1)),
                reference_class_shapley=float(phi_ref[i,m,0]),reference_intensity_shapley=float(phi_ref[i,m,1]),
                main_modality=NAMES[main[i]] if status[i]=='dominant' else status[i],
                strongest_modality=NAMES[main[i]],main_status=status[i],
                main_seed_agreement=float((per_main[:,i]==main[i]).mean()),
                baseline_main_agrees=bool(main[i]==mainref[i]),efficiency_residual=float(res[i,0])))
    return rows,phi,main
