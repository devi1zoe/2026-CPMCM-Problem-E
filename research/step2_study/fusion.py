"""Reliability gates learn solely from group-held-out expert predictions."""
from common import *
import torch
from torch import nn
from torch.nn import functional as F

class FusionModel(nn.Module):
    def __init__(self,config,prior,mean):
        super().__init__();self.config=config
        self.register_buffer('prior',torch.tensor(prior,dtype=torch.float32));self.register_buffer('mean',torch.tensor(float(mean)))
        self.class_gate=nn.Sequential(nn.Linear(30,16),nn.GELU(),nn.Linear(16,3))
        self.reg_gate=nn.Sequential(nn.Linear(30,16),nn.GELU(),nn.Linear(16,3))
        for gate in [self.class_gate,self.reg_gate]:
            nn.init.zeros_(gate[-1].weight)
            with torch.no_grad():gate[-1].bias.copy_(torch.tensor([.8,.1,.1]).log())
        if config['fusion']=='correction':
            with torch.random.fork_rng():
                self.correction=nn.Sequential(nn.Linear(30,16),nn.GELU(),nn.Linear(16,1))
                nn.init.zeros_(self.correction[-1].weight);nn.init.zeros_(self.correction[-1].bias)
    def features(self,b):
        present=b['o'].any(-1);experts=b['experts'].float();prob=experts[:,:,:3];intensity=experts[:,:,3]
        entropy=-(prob*prob.clamp_min(1e-8).log()).sum(-1)/1.0986122886681098
        pairs=[(0,1),(0,2),(1,2)];disagreement=[];distance=[]
        for a,z in pairs:
            keep=present[:,a]&present[:,z]
            disagreement.append((prob[:,a]-prob[:,z]).abs().sum(-1)/2*keep)
            distance.append((intensity[:,a]-intensity[:,z]).abs()/6*keep)
        x=torch.cat([b['quality'].float().flatten(1),(prob*present[...,None]).flatten(1),intensity*present/3,
            (1-entropy)*present,torch.stack(disagreement,-1),torch.stack(distance,-1)],-1)
        assert x.shape[1]==30
        if self.config['fusion']=='quality':x=torch.cat([x[:,:9],torch.zeros_like(x[:,9:])],-1)
        return x,present
    def forward(self,b):
        x,present=self.features(b);weights=[]
        for gate in [self.class_gate,self.reg_gate]:
            w=gate(x).masked_fill(~present,-1e4).softmax(-1)*present
            weights.append(w/w.sum(-1,keepdim=True).clamp_min(1e-8))
        experts=b['experts'].float();prob=(weights[0][...,None]*experts[:,:,:3]).sum(1)
        pred=(weights[1]*experts[:,:,3]).sum(-1)
        if hasattr(self,'correction'):pred=(pred+self.correction(x).squeeze(-1).tanh()).clamp(-3,3)
        empty=~present.any(-1);prob=torch.where(empty[:,None],self.prior[None],prob);pred=torch.where(empty,self.mean,pred)
        return dict(logits=prob.clamp_min(1e-8).log(),pred=pred,weights=torch.stack(weights,1),empty=empty,aux=None)

class FusionPredictor(nn.Module):
    def __init__(self,gate,experts):
        super().__init__();self.gate=gate;self.experts=nn.ModuleList(experts);self.config=gate.config
    def forward(self,b):
        if 'experts' in b:return self.gate(b)
        outputs=[]
        for expert in self.experts:
            z=expert(b);outputs.append(torch.cat([z['logits'].softmax(-1),z['pred'][:,None]],-1))
        return self.gate({**b,'experts':torch.stack(outputs,1)})
