"""Mask-safe multi-head self/cross attention; no temporal convolution or scalar gate."""
import math
import torch
from torch import nn


class AttentionBlock(nn.Module):
    def __init__(self, hidden=64, heads=4, dropout=.2):
        super().__init__()
        self.heads=heads;self.dim=hidden//heads
        self.norm=nn.LayerNorm(hidden)
        self.qkv=nn.Linear(hidden,3*hidden)
        self.out=nn.Linear(hidden,hidden)
        self.ff_norm=nn.LayerNorm(hidden)
        self.ff=nn.Sequential(nn.Linear(hidden,2*hidden),nn.GELU(),nn.Dropout(dropout),nn.Linear(2*hidden,hidden))
        self.drop=nn.Dropout(dropout)

    def forward(self,x,valid,allowed,return_weights=False):
        b,l,d=x.shape
        q,k,v=self.qkv(self.norm(x)).reshape(b,l,3,self.heads,self.dim).permute(2,0,3,1,4).unbind(0)
        legal=allowed[None]&valid[:,None,:]&valid[:,:,None]
        score=(q@k.transpose(-1,-2))/math.sqrt(self.dim)
        # Finite sentinel + exact mask renormalization handles all-empty rows without NaN.
        score=score.masked_fill(~legal[:,None],torch.finfo(score.dtype).min)
        att=score.softmax(-1)*legal[:,None]
        att=att/att.sum(-1,keepdim=True).clamp_min(1e-9)
        mixed=(self.drop(att)@v).transpose(1,2).reshape(b,l,d)
        delta=self.out(mixed)*legal.any(-1,keepdim=True)
        x=(x+self.drop(delta))*valid[:,:,None]
        x=(x+self.drop(self.ff(self.ff_norm(x))))*valid[:,:,None]
        return x,att if return_weights else None


class Model(nn.Module):
    def __init__(self,kind,prior,mean,hidden=64,heads=4,dropout=.2):
        super().__init__();self.kind=kind
        self.projections=nn.ModuleList([nn.Sequential(nn.Linear(d,hidden),nn.GELU(),nn.LayerNorm(hidden)) for d in [768,74,35]])
        self.modality=nn.Parameter(torch.empty(3,hidden));nn.init.normal_(self.modality,std=.02)
        pos=torch.arange(50).float()[:,None];freq=torch.exp(torch.arange(0,hidden,2)*(-math.log(10000)/hidden))
        pe=torch.zeros(50,hidden);pe[:,0::2]=torch.sin(pos*freq);pe[:,1::2]=torch.cos(pos*freq)
        self.register_buffer('position',pe)
        if kind!='pool':
            self.self_blocks=nn.ModuleList([AttentionBlock(hidden,heads,dropout) for _ in range(3)])
        if kind in ['cross','cross_clean']:
            self.cross_block=AttentionBlock(hidden,heads,dropout)
        self.head=nn.Sequential(nn.Linear(hidden*3+9,hidden),nn.GELU(),nn.Dropout(dropout))
        self.classifier=nn.Linear(hidden,3);self.regressor=nn.Linear(hidden,1)
        self.register_buffer('prior',torch.as_tensor(prior,dtype=torch.float32))
        self.register_buffer('mean',torch.tensor(float(mean)))
        mods=torch.arange(3).repeat_interleave(50)
        self.register_buffer('cross_allowed',mods[:,None]!=mods[None,:])
        self.register_buffer('self_allowed',torch.ones(50,50,dtype=torch.bool))

    def forward(self,b,return_attention=False):
        valid=b['o']&b['p'][:,None,:];states=[];maps=[]
        for j,name in enumerate(['text','audio','vision']):
            # Mask before projecting as well as after every block.
            h=self.projections[j](b[name].float()*valid[:,j,:,None])
            h=(h+self.position[None]+self.modality[j])*valid[:,j,:,None]
            if self.kind!='pool':
                h,a=self.self_blocks[j](h,valid[:,j],self.self_allowed,return_attention);maps.append(a)
            states.append(h)
        h=torch.cat(states,1);flat=valid.flatten(1);cross=None
        if self.kind in ['cross','cross_clean']:
            h,cross=self.cross_block(h,flat,self.cross_allowed,return_attention)
        h=h.reshape(len(h),3,50,-1)
        pooled=(h*valid[:,:,:,None]).sum(2)/valid.sum(2,keepdim=True).clamp_min(1)
        quality=b['quality'].float().reshape(len(h),-1)
        v=self.head(torch.cat([pooled.flatten(1),quality],-1))
        logits=self.classifier(v);pred=3*torch.tanh(self.regressor(v).squeeze(-1))
        empty=~flat.any(1)
        logits=torch.where(empty[:,None],self.prior.log()[None],logits)
        pred=torch.where(empty,self.mean,pred)
        return dict(logits=logits,pred=pred,self_attention=maps,cross_attention=cross)
