# 本程序及代码是在人工智能工具辅助下完成的。

# 人工智能工具名称：GPT-6 Astra

# 版本/型号：gpt-6-astra

# 开发机构/公司：OpenAI

# 版本颁布日期：2026年9月3日

"""轻量时序融合，三个分支 + 三分类/连续强度双任务。"""
import torch
from torch import nn

NAMES = ['text','audio','vision']
DIMS = [768,74,35]


class Branch(nn.Module):
    def __init__(self, dim, hidden=64, dropout=.2, pooled=False):
        super().__init__()
        self.pooled = pooled
        if pooled:
            self.stat = nn.Sequential(nn.Linear(dim*2,hidden),nn.GELU(),nn.Dropout(dropout),nn.LayerNorm(hidden))
        else:
            self.input = nn.Sequential(nn.Linear(dim,hidden),nn.GELU(),nn.LayerNorm(hidden))
            self.conv = nn.ModuleList([nn.Conv1d(hidden,hidden,3,padding=d,dilation=d) for d in [1,2]])
            self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(2)])
            self.drop = nn.Dropout(dropout)
            self.score = nn.Linear(hidden,1)
            self.output = nn.Sequential(nn.Linear(hidden*2,hidden),nn.GELU(),nn.LayerNorm(hidden))

    def forward(self,x,p,observed,explicit):
        x = x.float()
        valid = observed if explicit or self.pooled else p
        mask = valid.unsqueeze(-1)
        count = mask.sum(1).clamp_min(1)
        if self.pooled:
            mean = (x*mask).sum(1)/count
            std = (((x-mean[:,None])**2*mask).sum(1)/count+1e-6).sqrt()
            h = self.stat(torch.cat([mean,std],-1))
        else:
            h = self.input(x)*p.unsqueeze(-1)
            for conv,norm in zip(self.conv,self.norms):
                h = norm(h+self.drop(torch.nn.functional.gelu(conv(h.transpose(1,2)).transpose(1,2))))
                h = h*p.unsqueeze(-1)
            scores = self.score(h).squeeze(-1).masked_fill(~valid,-1e4)
            alpha = scores.softmax(1)*valid
            alpha = alpha/alpha.sum(1,keepdim=True).clamp_min(1e-6)
            attended = (h*alpha[:,:,None]).sum(1)
            mean = (h*mask).sum(1)/count
            h = self.output(torch.cat([attended,mean],-1))
        # 即使“无显式掩码”对照也保证整段全无观测不制造伪信号。
        return h*observed.any(1,keepdim=True)


class SentimentModel(nn.Module):
    def __init__(self, variant, hidden=64, dropout=.2):
        super().__init__()
        self.variant = variant
        self.explicit = variant in ['mask_aug','gated_aug'] or variant.endswith('_only')
        self.indices = {'text_only':[0],'audio_only':[1],'vision_only':[2]}.get(variant,[0,1,2])
        self.branches = nn.ModuleDict({str(j):Branch(DIMS[j],hidden,dropout,variant=='pooled_clean') for j in self.indices})
        self.quality = nn.Linear(3,hidden)
        self.gate = nn.Sequential(nn.Linear(hidden+3,32),nn.Tanh(),nn.Linear(32,1))
        self.head = nn.Sequential(nn.Linear(hidden,hidden),nn.GELU(),nn.Dropout(dropout))
        self.classifier = nn.Linear(hidden,3)
        self.regressor = nn.Linear(hidden,1)

    def forward(self,b):
        p,o = b['p'],b['o']
        q = b['quality'][:,self.indices].float()
        hs = []
        for j in self.indices:
            h = self.branches[str(j)](b[NAMES[j]],p,o[:,j],self.explicit)
            if self.explicit:
                h = h+self.quality(b['quality'][:,j].float())*o[:,j].any(1,keepdim=True)
            hs.append(h)
        h = torch.stack(hs,1)
        present = o[:,self.indices].any(-1)
        if self.variant=='gated_aug':
            scores = self.gate(torch.cat([h,q],-1)).squeeze(-1).masked_fill(~present,-1e4)
            weights = scores.softmax(1)
        else:
            weights = torch.ones_like(present,dtype=torch.float32)/len(self.indices)
        z = self.head((h*weights[:,:,None]).sum(1))
        logits = self.classifier(z)
        intensity = 3*torch.tanh(self.regressor(z).squeeze(-1))
        gates = torch.zeros((len(z),3),device=z.device)
        gates[:,self.indices] = weights
        return logits,intensity,gates
