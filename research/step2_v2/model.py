"""Frozen-input lightweight EBMC-inspired models; training-only teachers/losses are external."""
import torch
from torch import nn
from torch.nn import functional as F

NAMES = ['text', 'audio', 'vision']
DIMS = [768, 74, 35]
VARIANTS = {
    'text': dict(indices=[0]), 'audio': dict(indices=[1]), 'vision': dict(indices=[2]),
    'fixed': dict(fixed=True), 'gated': {}, 'aux': dict(aux=True),
    'msd': dict(aux=True, msd=True), 'cce': dict(aux=True, msd=True, cce=True),
    'imtd': dict(aux=True, msd=True, cce=True, imtd=True),
    'emc': dict(aux=True, msd=True, cce=True, imtd=True, emc=True),
}


class Branch(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.ModuleList([nn.Conv1d(64,64,3,padding=d,dilation=d) for d in [1,2]])
        self.norm = nn.ModuleList([nn.LayerNorm(64) for _ in range(2)])
        self.drop = nn.Dropout(.2)
        self.score = nn.Linear(64,1)
        self.output = nn.Sequential(nn.Linear(128,64),nn.GELU(),nn.LayerNorm(64))

    def temporal(self,x,p):
        for conv,norm in zip(self.conv,self.norm):
            x = norm(x+self.drop(F.gelu(conv(x.transpose(1,2)).transpose(1,2)) ))*p[...,None]
        return x

    def pool(self,x,o):
        a=self.score(x).squeeze(-1).masked_fill(~o,-1e4).softmax(-1)*o
        a=a/a.sum(-1,keepdim=True).clamp_min(1e-6)
        mean=(x*o[...,None]).sum(1)/o.sum(1,keepdim=True).clamp_min(1)
        return self.output(torch.cat([(x*a[...,None]).sum(1),mean],-1))*o.any(1,keepdim=True)


class Model(nn.Module):
    def __init__(self, config, prior, mean):
        super().__init__()
        self.config=dict(config);self.indices=config.get('indices',[0,1,2])
        self.register_buffer('prior',torch.tensor(prior,dtype=torch.float32))
        self.register_buffer('mean',torch.tensor(float(mean)))
        self.input=nn.ModuleDict({str(m):nn.Sequential(nn.Linear(DIMS[m],64),nn.GELU(),nn.LayerNorm(64)) for m in self.indices})
        self.branches=nn.ModuleDict({str(m):Branch() for m in self.indices})
        self.decompose=bool(config.get('msd'))
        if self.decompose:
            self.shared=nn.ModuleDict({str(m):nn.Linear(64,32) for m in self.indices})
            self.specific=nn.ModuleDict({str(m):nn.Linear(64,32) for m in self.indices})
        if config.get('cce'):
            self.enhance=nn.ModuleList([nn.Sequential(nn.Linear(131,64),nn.GELU(),nn.Linear(64,64)) for _ in range(3)])
            self.local_gate=nn.ModuleList([nn.Linear(131,1) for _ in range(3)])
            for e in self.enhance:nn.init.zeros_(e[-1].weight);nn.init.zeros_(e[-1].bias)
        self.quality=nn.Linear(3,64)
        self.gate=nn.Sequential(nn.Linear(67,32),nn.Tanh(),nn.Linear(32,1))
        self.head=nn.Sequential(nn.Linear(64,64),nn.GELU(),nn.Dropout(.2))
        self.classifier=nn.Linear(64,3);self.regressor=nn.Linear(64,1)
        if config.get('aux') or config.get('imtd') or config.get('emc'):
            self.aux_head=nn.ModuleList([nn.Linear(32 if self.decompose else 64,4) for _ in range(3)])

    def forward(self,b):
        p,o=b['p'],b['o']; n=len(p); hs=[]; ss=[]; vs=[]; xs=[]
        for m in self.indices:
            k=str(m)
            x=self.branches[k].temporal(self.input[k](b[NAMES[m]].float())*o[:,m,:,None],p)
            xs.append(x)
            if self.decompose:ss.append(self.shared[k](x));vs.append(self.specific[k](x))
        preserve=xs[0].sum()*0
        if self.config.get('cce'):
            enhanced=[]
            for m,x in enumerate(xs):
                other=[j for j in range(3) if j!=m]
                obs=o[:,other].transpose(1,2).float();den=obs.sum(-1,keepdim=True).clamp_min(1)
                if self.decompose:
                    sc=sum(ss[j]*o[:,j,:,None] for j in other)/den
                    sp=sum(vs[j]*o[:,j,:,None] for j in other)/den
                    v=torch.cat([x,sc,sp,o.transpose(1,2).float()],-1)
                else:
                    # Without MSD, retain CCE using raw 64-d other-modality features.
                    complementary=sum(xs[j]*o[:,j,:,None] for j in other)/den
                    v=torch.cat([x,complementary,o.transpose(1,2).float()],-1)
                delta=self.enhance[m](v)*torch.sigmoid(self.local_gate[m](v))
                active=o[:,m]&o[:,other].any(1)
                delta=delta*active[...,None]
                preserve=preserve+delta.square().sum()/(active.sum().clamp_min(1)*64)
                enhanced.append(x+delta)
            pool_x=enhanced
        else:pool_x=xs
        shared=[];specific=[];aux=[]
        for j,m in enumerate(self.indices):
            h=self.branches[str(m)].pool(pool_x[j],o[:,m]);hs.append(h)
            if self.decompose:
                den=o[:,m].sum(1,keepdim=True).clamp_min(1)
                shared.append((ss[j]*o[:,m,:,None]).sum(1)/den)
                specific.append((vs[j]*o[:,m,:,None]).sum(1)/den)
            if hasattr(self,'aux_head'):
                a=self.aux_head[m](specific[-1] if self.decompose else h)
                aux.append(torch.cat([a[:,:3],3*torch.tanh(a[:,3:])],-1))
        h=torch.stack(hs,1);present=o[:,self.indices].any(-1)
        q=b['quality'][:,self.indices].float()
        content=h+self.quality(q)*present[...,None]
        if self.config.get('fixed'):
            weights=present.float()/present.sum(-1,keepdim=True).clamp_min(1)
        else:
            weights=self.gate(torch.cat([content,q],-1)).squeeze(-1).masked_fill(~present,-1e4).softmax(-1)*present
            weights=weights/weights.sum(-1,keepdim=True).clamp_min(1e-6)
        z=self.head((content*weights[...,None]).sum(1))
        logits=self.classifier(z);pred=3*torch.tanh(self.regressor(z).squeeze(-1))
        empty=~present.any(-1)
        logits=torch.where(empty[:,None],self.prior.clamp_min(1e-8).log()[None],logits)
        pred=torch.where(empty,self.mean,pred)
        return dict(logits=logits,pred=pred,weights=weights,empty=empty,h=h,
            aux=torch.stack(aux,1) if aux else None,
            shared=torch.stack(shared,1) if shared else None,
            specific=torch.stack(specific,1) if specific else None,
            present=present,preserve=preserve,energy_features=hs)


def objectives(model,out,b,c,y,class_weight,reg_weight,teacher=None,stage=1):
    zero=out['logits'].sum()*0
    task=F.cross_entropy(out['logits'],c,weight=class_weight)+reg_weight*F.smooth_l1_loss(out['pred'],y)
    aux=inv=orth=kd=emc=zero; tasks=[]
    if out['aux'] is not None:
        for j,m in enumerate(model.indices):
            keep=out['present'][:,j]
            if keep.any():
                a=out['aux'][keep,j]
                loss=F.cross_entropy(a[:,:3],c[keep],weight=class_weight)+reg_weight*F.smooth_l1_loss(a[:,3],y[keep])
                tasks.append((j,keep,loss))
        if tasks:aux=torch.stack([x[2] for x in tasks]).mean()
    if model.config.get('msd'):
        terms=[]
        for i in range(3):
            keep=out['present'][:,i]
            if keep.any():orth=orth+F.cosine_similarity(out['shared'][keep,i],out['specific'][keep,i]).square().mean()/3
            for j in range(i+1,3):
                valid=keep&out['present'][:,j]
                if valid.sum()>1:
                    u=F.normalize(out['shared'][valid,i],dim=-1);v=F.normalize(out['shared'][valid,j],dim=-1)
                    sim=u@v.T/.1;target=torch.arange(len(u),device=u.device)
                    terms.append((F.cross_entropy(sim,target)+F.cross_entropy(sim.T,target))/2)
        if terms:inv=torch.stack(terms).mean()
    trust=torch.zeros_like(out['present'],dtype=torch.float32)
    if stage==2 and teacher is not None:
        tp=(teacher['aux'][...,:3]/2).softmax(-1)
        ent=-(tp*tp.clamp_min(1e-8).log()).sum(-1)
        trust=b['quality'][:,model.indices,0].float()*(1-ent/torch.log(torch.tensor(3.,device=y.device))).clamp_min(0)
        trust=trust*out['present'];trust=trust/trust.sum(-1,keepdim=True).clamp_min(1e-8)
        if model.config.get('imtd'):
            divergence=F.kl_div(F.log_softmax(out['logits']/2,-1)[:,None].expand_as(tp),tp,reduction='none').sum(-1)
            kd=(trust*divergence).sum(-1).mean()*4
        if model.config.get('emc'):
            energies=[];penalties=[]
            for j,keep,loss in tasks:
                feature=out['energy_features'][j]
                norm=feature[keep].square().mean()
                energies.append(norm+loss+ent[keep,j].mean())
                grad=torch.autograd.grad(norm,feature,create_graph=True,retain_graph=True)[0]
                penalties.append(grad.square().mean())
            if len(energies)>1:
                emc=sum((energies[i]-energies[j]).square() for i in range(len(energies)) for j in range(i+1,len(energies)))+.01*sum(penalties)
    total=task+(.1*aux if model.config.get('aux') else zero)+.05*(inv+orth)+.05*out['preserve']+.1*kd+.01*emc
    return total,dict(task=task,aux=aux,alignment=inv,orthogonal=orth,preserve=out['preserve'],kd=kd,emc=emc),trust
