# 本程序及代码是在人工智能工具辅助下完成的。

# 人工智能工具名称：GPT-6 Astra

# 版本/型号：gpt-6-astra

# 开发机构/公司：OpenAI

# 版本颁布日期：2026年9月3日

"""最终三种子集成及全模态不可观测时的训练先验回退。"""
from common import PACKAGE, sha256
from model import SentimentModel
from data import slice_view
from inference_ops import linear_convolutions
import torch
import numpy as np


def load_ensemble(config, device):
    models = []
    for name, digest in config['checkpoints'].items():
        path = PACKAGE / '模型' / name
        if sha256(path) != digest:
            raise ValueError(f'模型参数校验失败：{name}')
        ck = torch.load(path, map_location='cpu', weights_only=True)
        model = SentimentModel(ck['variant'], ck['hidden'], ck['dropout'])
        model.load_state_dict(ck['state_dict'])
        model.register_buffer('fallback_prior', torch.tensor(config['fallback_prior'], dtype=torch.float32))
        model.register_buffer('fallback_mean', torch.tensor(config['fallback_mean'], dtype=torch.float32))
        models.append(model.to(device).eval())
    return models


@torch.inference_mode()
def ensemble_predict(models, view, config, batch=256):
    all_outputs = []
    for model in models:
        outputs = []
        with linear_convolutions(model):
            for start in range(0, len(view['p']), batch):
                b = slice_view(view, slice(start, start + batch))
                logits, pred, _ = model(b)
                empty = ~b['o'].any(dim=(1, 2))
                logits = torch.where(empty[:, None], model.fallback_prior.log()[None], logits)
                pred = torch.where(empty, model.fallback_mean, pred)
                outputs.append(torch.cat([logits.softmax(-1), pred[:, None]], -1).cpu().numpy())
        all_outputs.append(np.concatenate(outputs))
    per_seed = np.stack(all_outputs)
    return per_seed.mean(0), per_seed[:, :, 3].std(0, ddof=1)
