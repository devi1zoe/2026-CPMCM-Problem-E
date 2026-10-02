# 本程序及代码是在人工智能工具辅助下完成的。

# 人工智能工具名称：GPT-6 Astra

# 版本/型号：gpt-6-astra

# 开发机构/公司：OpenAI

# 版本颁布日期：2026年9月3日

"""Algebraically identical stride-one Conv1d via explicit windows and FP32 linear algebra.

Used only by inference contexts; training convolutions, parameters, and RNG are untouched.
The cuDNN TF32 path can differ at ~1e-4; backend audits record predictions and class agreement.
"""
import contextlib,types
import torch
from torch import nn
from torch.nn import functional as F

def window_conv(module,x):
    assert module.groups==1 and module.stride==(1,) and module.padding_mode=='zeros'
    k=module.kernel_size[0];d=module.dilation[0];p=module.padding[0]
    padded=F.pad(x,(p,p));length=padded.shape[-1]-d*(k-1)
    windows=torch.stack([padded[:,:,i*d:i*d+length] for i in range(k)],dim=-1)
    # [B,C,L,K] -> [B,L,C*K], matching weight [out,C,K].
    flat=windows.permute(0,2,1,3).reshape(x.shape[0],length,-1)
    return F.linear(flat,module.weight.reshape(module.out_channels,-1),module.bias).transpose(1,2)

@contextlib.contextmanager
def linear_convolutions(model):
    originals=[]
    try:
        for m in model.modules():
            if isinstance(m,nn.Conv1d):
                originals.append((m,m.forward));m.forward=types.MethodType(window_conv,m)
        yield
    finally:
        for m,forward in originals:m.forward=forward
