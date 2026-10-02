"""Two preregistered task-specific fusion candidates; baseline code is immutable."""
import copy
import torch
from torch import nn
from torch.nn import functional as F
from base_model import Model as BaseModel, NAMES


def normalized_gate(logits,present):
    w=logits.masked_fill(~present,-1e4).softmax(-1)*present
    return w/w.sum(-1,keepdim=True).clamp_min(1e-6)


class DualGate(BaseModel):
    def __init__(self,config,prior,mean):
        super().__init__({},prior,mean)
        self.config=dict(config)
        self.regression_gate=copy.deepcopy(self.gate)

    def forward(self,b):
        p,o=b['p'],b['o'];hs=[]
        for m in self.indices:
            branch=self.branches[str(m)]
            x=branch.temporal(self.input[str(m)](b[NAMES[m]].float())*o[:,m,:,None],p)
            hs.append(branch.pool(x,o[:,m]))
        h=torch.stack(hs,1);present=o.any(-1);q=b['quality'].float()
        content=h+self.quality(q)*present[...,None]
        gate_input=torch.cat([content,q],-1)
        wc=normalized_gate(self.gate(gate_input).squeeze(-1),present)
        wr=normalized_gate(self.regression_gate(gate_input).squeeze(-1),present)
        zc=self.head((content*wc[...,None]).sum(1));zr=self.head((content*wr[...,None]).sum(1))
        logits=self.classifier(zc);pred=3*torch.tanh(self.regressor(zr).squeeze(-1));empty=~present.any(-1)
        return dict(logits=torch.where(empty[:,None],self.prior.log()[None],logits),pred=torch.where(empty,self.mean,pred),
            weights=wc,weights_classification=wc,weights_regression=wr,empty=empty,present=present,aux=None)


class LateFusion(nn.Module):
    def __init__(self,config,prior,mean):
        super().__init__();self.config=dict(config)
        self.register_buffer('prior',torch.tensor(prior,dtype=torch.float32));self.register_buffer('mean',torch.tensor(float(mean)))
        self.experts=nn.ModuleList([BaseModel(dict(indices=[i]),prior,mean).eval().requires_grad_(False) for i in range(3)])
        self.classification_gate=nn.Sequential(nn.Linear(9,16),nn.GELU(),nn.Linear(16,3))
        self.regression_gate=nn.Sequential(nn.Linear(9,16),nn.GELU(),nn.Linear(16,3))
        for gate in [self.classification_gate,self.regression_gate]:
            nn.init.zeros_(gate[-1].weight)
            with torch.no_grad():gate[-1].bias.copy_(torch.tensor([.8,.1,.1]).log())

    def train(self,mode=True):
        super().train(mode)
        for m in self.experts:m.eval()
        return self

    @torch.no_grad()
    def expert_outputs(self,b):
        out=[]
        for m in self.experts:
            r=m(b);out.append(torch.cat([r['logits'].softmax(-1),r['pred'][:,None]],-1))
        return torch.stack(out,1)

    def forward(self,b):
        e=b['expert_outputs'] if 'expert_outputs' in b else self.expert_outputs(b)
        present=b['o'].any(-1);q=b['quality'].float().reshape(-1,9)
        wc=normalized_gate(self.classification_gate(q),present);wr=normalized_gate(self.regression_gate(q),present)
        prob=(e[:,:,:3]*wc[:,:,None]).sum(1);pred=(e[:,:,3]*wr).sum(1);empty=~present.any(-1)
        prob=torch.where(empty[:,None],self.prior[None],prob)
        return dict(logits=prob.clamp_min(1e-8).log(),pred=torch.where(empty,self.mean,pred),
            weights=wc,weights_classification=wc,weights_regression=wr,empty=empty,present=present,aux=None)


def Model(config,prior,mean):
    if config.get('kind')=='dual_gate':return DualGate(config,prior,mean)
    if config.get('kind')=='late_fusion':return LateFusion(config,prior,mean)
    return BaseModel(config,prior,mean)


def task_loss(out,c,y,cw):
    classification=F.cross_entropy(out['logits'],c,weight=cw)
    regression=2*F.smooth_l1_loss(out['pred'],y)
    return classification+regression,dict(classification=classification,weighted_regression=regression)
