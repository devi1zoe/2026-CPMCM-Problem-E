"""Small task network and optional trainable low-rank text adaptation."""
from base_model import Model as BaseModel, NAMES, DIMS
from common import ROOT,ENCODER
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
import math
import copy

class LowRankLinear(nn.Module):
    def __init__(self,base,rank=4,alpha=8):
        super().__init__();self.base=base;self.scale=alpha/rank
        self.lora_a=nn.Parameter(torch.empty(rank,base.in_features));self.lora_b=nn.Parameter(torch.zeros(base.out_features,rank))
        nn.init.kaiming_uniform_(self.lora_a,a=math.sqrt(5))
    def forward(self,x):
        return self.base(x)+F.linear(F.linear(x,self.lora_a),self.lora_b)*self.scale

class AdaptedText(nn.Module):
    def __init__(self):
        super().__init__()
        from transformers import AutoModel
        # Encoder loading and adapter initialization must not advance the task-head RNG.
        with torch.random.fork_rng():
            self.backbone=AutoModel.from_pretrained(ENCODER,local_files_only=True).requires_grad_(False).eval()
            for i in [4,5]:
                attention=self.backbone.transformer.layer[i].attention
                for name in ['q_lin','v_lin']:setattr(attention,name,LowRankLinear(getattr(attention,name)))
        stats=dict(np.load(ROOT/'checkpoints/normalization.npz'))
        self.register_buffer('mean',torch.as_tensor(stats['text_mean']))
        self.register_buffer('std',torch.as_tensor(stats['text_std']))
    def train(self,mode=True):
        super().train(mode);self.backbone.eval();return self
    def forward(self,b):
        tok=b['tokens'].clone();att=b['attention'].clone();missing=b['p']&~b['o'][:,0]
        tok[missing]=0;att[missing]=False
        empty=~att.any(1);tok[empty,0]=101;att[empty,0]=True
        with torch.autocast(device_type='cuda',dtype=torch.float16,enabled=tok.is_cuda):
            h=self.backbone(input_ids=tok,attention_mask=att).last_hidden_state
        # Same normalization and float16 storage precision as frozen reference views.
        return ((h.float()-self.mean)/self.std).clamp(-8,8).masked_fill(~b['o'][:,0,:,None],0).half()

class Model(BaseModel):
    def __init__(self,config,prior,mean):
        super().__init__(config,prior,mean)
        if config.get('lora'):self.language=AdaptedText()
        if config.get('reconstruct'):
            self.target_input=copy.deepcopy(self.input).requires_grad_(False)
            self.missing_position=nn.Embedding(50,64);self.missing_modality=nn.Embedding(3,64)
            self.reconstruction_attention=nn.MultiheadAttention(64,4,batch_first=True,dropout=0.)
            self.reconstruction_output=nn.Sequential(nn.Linear(64,64),nn.GELU(),nn.Linear(64,64))
    def forward(self,b):
        if hasattr(self,'language'):
            b=dict(b);b['text']=b['adapted_text'] if 'adapted_text' in b else self.language(b)
        if self.config.get('reconstruct'):return self.forward_reconstruction(b)
        return super().forward(b)

    def forward_reconstruction(self,b):
        p,o=b['p'],b['o'];n=len(p);available=o.any((1,2))
        projected=torch.stack([self.input[str(m)](b[NAMES[m]].float())*o[:,m,:,None] for m in range(3)],1)
        position=self.missing_position(torch.arange(50,device=p.device))
        queries=(position[None,:,:]+self.missing_modality.weight[:,None,:]).reshape(1,150,64).expand(n,-1,-1)
        keys=(projected+queries.reshape(n,3,50,64))*o[...,None]
        key_mask=o.reshape(n,150).clone();key_mask[~available,0]=True
        decoded,_=self.reconstruction_attention(queries,keys.reshape(n,150,64),projected.reshape(n,150,64),
            key_padding_mask=~key_mask,need_weights=False)
        estimate=self.reconstruction_output(decoded).reshape(n,3,50,64)*available[:,None,None,None]
        missing=p[:,None,:]&~o&available[:,None,None]
        filled=projected+estimate*missing[...,None]
        hs=[]
        for m in range(3):
            branch=self.branches[str(m)];x=branch.temporal(filled[:,m],p)
            pool_weight=o[:,m].float()+.5*missing[:,m].float()
            alpha=branch.score(x).squeeze(-1)+pool_weight.clamp_min(1e-8).log()
            alpha=alpha.masked_fill(pool_weight==0,-1e4).softmax(-1)*(pool_weight>0)
            alpha=alpha/alpha.sum(-1,keepdim=True).clamp_min(1e-6)
            mean=(x*pool_weight[...,None]).sum(1)/pool_weight.sum(1,keepdim=True).clamp_min(1e-6)
            hs.append(branch.output(torch.cat([(x*alpha[...,None]).sum(1),mean],-1))*available[:,None])
        h=torch.stack(hs,1);present=(o|missing).any(-1);q=b['quality'].float()
        content=h+self.quality(q)*present[...,None]
        weights=self.gate(torch.cat([content,q],-1)).squeeze(-1).masked_fill(~present,-1e4).softmax(-1)*present
        weights=weights/weights.sum(-1,keepdim=True).clamp_min(1e-6)
        z=self.head((content*weights[...,None]).sum(1));logits=self.classifier(z);pred=3*torch.tanh(self.regressor(z).squeeze(-1))
        logits=torch.where(available[:,None],logits,self.prior.log()[None]);pred=torch.where(available,pred,self.mean)
        rec_loss=estimate.sum()*0;rec_mse=rec_loss
        if 'reconstruction_mask' in b:
            supervised=b['reconstruction_mask']&missing
            if supervised.any():
                with torch.no_grad():target=torch.stack([self.target_input[str(m)](b['target_'+NAMES[m]].float()) for m in range(3)],1)
                rec_loss=F.smooth_l1_loss(estimate[supervised],target[supervised])
                rec_mse=F.mse_loss(estimate[supervised],target[supervised])
        return dict(logits=logits,pred=pred,weights=weights,empty=~available,h=h,aux=None,
            reconstruction_loss=rec_loss,reconstruction_mse=rec_mse,reconstruction=estimate,recovered=missing)

def study_state(model,cpu=True):
    """Omit only reproducible frozen encoder weights, never learned adapters."""
    return {k:(v.detach().cpu().clone() if cpu else v.detach().clone()) for k,v in model.state_dict().items()
        if not (k.startswith('language.backbone.') and '.lora_' not in k)}

def load_study_state(model,state):
    missing,unexpected=model.load_state_dict(state,strict=False)
    allowed={k for k in model.state_dict() if k.startswith('language.backbone.') and '.lora_' not in k}
    assert set(missing)==allowed and not unexpected,(missing,unexpected)
