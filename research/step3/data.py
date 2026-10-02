"""统一 aligned_50 / 附件3接口，区分有效范围、观测可用性、人工删除。

文本先删除 token，再经同一冻结编码器重新编码；不使用附件2预计算 text。
"""
from common import DATA, ROOT, ENCODER, save_json
import pickle
import numpy as np
import torch

MODALITIES = 'TAV'
VALID_PROFILES = [
    dict(name='clean', subset='', rate=0., position='random'),
    *[dict(name=f'{s}_r30_random', subset=s, rate=.3, position='random') for s in ['T','A','V','TAV']],
    dict(name='TAV_r50_random', subset='TAV', rate=.5, position='random'),
]


def load_attachment2():
    with open(DATA / '附件2-数据集特征文件/aligned_50.pkl', 'rb') as f:
        return pickle.load(f)


def adapt(d):
    tb = np.asarray(d['text_bert'])
    assert tb.ndim == 3 and tb.shape[1:] == (3, 50)
    assert np.isfinite(tb).all() and np.equal(tb, np.round(tb)).all()
    tok = tb[:, 0].astype(np.int64)
    att = tb[:, 1].astype(bool)
    assert tok.min() >= 0 and tok.max() < 30522
    assert np.isin(tb[:, 1], [0, 1]).all() and np.all(tb[:, 2] == 0)
    av = [np.asarray(d[k], dtype=np.float32) for k in ['audio', 'vision']]
    assert all(np.isfinite(x).all() for x in av)
    n = len(tok)
    p = np.zeros((n, 50), bool)
    uncertain = []
    for i in range(n):
        sep = np.flatnonzero((tok[i] == 102) & att[i])
        if len(sep):
            end = int(sep[-1])
        else:
            # 无 SEP 时，只能用剩余观测推断下界，不能恢复未知尾部。
            evidence = att[i] | np.any(av[0][i] != 0, -1) | np.any(av[1][i] != 0, -1)
            end = min(50, int(np.flatnonzero(evidence)[-1]) + 1) if evidence.any() else 1
            uncertain.append(i)
        p[i, 1:end] = True
    # UNK 只表示词义不可恢复，不能断言是题目人工缺失；统一作为不可用语义。
    ot = p & att & ~np.isin(tok, [0, 100, 101, 102, 103])
    o = np.stack([ot, *[p & np.any(x != 0, -1) for x in av]], axis=1)
    return dict(tokens=tok, attention=att, p=p, o=o, audio=av[0], vision=av[1], uncertain=uncertain)


def intervals(length, count, position, rng, pattern='single'):
    """返回词槽相对下标，保持整数删除数一致；多短段分配到三个互不相交分区。"""
    count = min(length, max(0, count))
    if not count:
        return np.empty(0, dtype=int)
    if pattern == 'multi' and length >= 6 and count >= 3:
        bins = np.array_split(np.arange(length), 3)
        quotas = np.array([count // 3 + (j < count % 3) for j in range(3)])
        return np.concatenate([b[intervals(len(b), int(q), position, rng)] for b, q in zip(bins, quotas)])
    start = {'start': 0, 'middle': (length - count)//2, 'end': length-count}.get(position)
    if start is None:
        start = int(rng.integers(length - count + 1))
    return np.arange(start, start + count)


def corrupt(base, profile, seed):
    rng = np.random.default_rng(seed)
    p, obs = base['p'], base['o']
    removed = np.zeros_like(obs)
    for i, valid in enumerate(p):
        slots = np.flatnonzero(valid)
        if profile.get('random_mix'):
            subset = rng.choice(['T','A','V','TA','TV','AV','TAV'])
            rate = float(rng.choice([.1,.3,.5,.7]))
            position = str(rng.choice(['start','middle','end','random']))
            pattern = 'single'
            sync = bool(rng.integers(2))
        else:
            subset, rate = profile['subset'], profile['rate']
            position, pattern = profile.get('position','random'), profile.get('pattern','single')
            sync = profile.get('sync', True)
        # half-up 整数化；正比例短样本至少遮挡一个词槽。
        count = max(1, int(np.floor(len(slots)*rate+.5))) if rate > 0 and len(slots) else 0
        idx = intervals(len(slots), count, position, rng, pattern)
        for m, name in enumerate(MODALITIES):
            if name in subset:
                if not sync:
                    idx = intervals(len(slots), count, position, rng, pattern)
                removed[i, m, slots[idx]] = True
    available = obs & ~removed
    return available, removed


class TextEncoder:
    def __init__(self, device='cuda'):
        from transformers import AutoModel
        self.device = device
        self.model = AutoModel.from_pretrained(ENCODER, local_files_only=True).to(device).eval()
        self.model.requires_grad_(False)

    @torch.inference_mode()
    def encode(self, base, available, batch_size=128):
        tok = base['tokens'].copy()
        # 无词义的 UNK 与人工删除一样不参与上下文；CLS/SEP仍可见。
        att = base['attention'].copy()
        masked = base['p'] & ~available[:, 0]
        tok[masked] = 0
        att[masked] = False
        # 避免极端全零输入的 attention 全屏蔽。
        bad = ~att.any(1)
        tok[bad,0], att[bad,0] = 101, True
        chunks = []
        for start in range(0, len(tok), batch_size):
            ids = torch.as_tensor(tok[start:start+batch_size], device=self.device)
            mask = torch.as_tensor(att[start:start+batch_size], device=self.device)
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=str(self.device).startswith('cuda')):
                h = self.model(input_ids=ids, attention_mask=mask).last_hidden_state
            chunks.append(h.float().cpu().numpy())
        out = np.concatenate(chunks)
        out[~available[:,0]] = 0
        return out


def fit_stats(base, text):
    out = {}
    for j, (name, x) in enumerate(zip(['text','audio','vision'], [text, base['audio'], base['vision']])):
        v = x[base['o'][:,j]].astype(np.float64)
        out[name+'_mean'] = v.mean(0).astype(np.float32)
        out[name+'_std'] = np.maximum(v.std(0), 1e-4).astype(np.float32)
    return out


def make_view(base, encoder, stats, profile, seed, clean_text=None):
    o, removed = corrupt(base, profile, seed)
    if np.array_equal(o[:,0], base['o'][:,0]) and clean_text is not None:
        text = clean_text
    else:
        text = encoder.encode(base, o)
    p = base['p']
    lengths = np.maximum(p.sum(-1), 1)
    runs = np.zeros((len(p),3), np.int32)
    longest = runs.copy()
    for t in range(50):
        runs = (runs+1) * (p[:,t,None] & ~o[:,:,t])
        longest = np.maximum(longest,runs)
    quality = np.stack([o.sum(-1)/lengths[:,None], longest/lengths[:,None],
                        np.broadcast_to(lengths[:,None]/48, (len(p),3))],axis=-1).astype(np.float32)
    out = dict(p=p, o=o, removed=removed, quality=quality)
    for m, (name,x) in enumerate(zip(['text','audio','vision'], [text,base['audio'],base['vision']])):
        z = np.clip((x-stats[name+'_mean'])/stats[name+'_std'], -8, 8)
        z[~o[:,m]] = 0
        out[name] = z.astype(np.float16)
    return out


def torch_view(view, device='cuda'):
    return {k:torch.as_tensor(v, device=device) for k,v in view.items() if k != 'removed'}


def slice_view(view, idx):
    return {k:v[idx] for k,v in view.items()}

