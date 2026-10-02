# 本程序及代码是在人工智能工具辅助下完成的。

# 人工智能工具名称：GPT-6 Astra

# 版本/型号：gpt-6-astra

# 开发机构/公司：OpenAI

# 版本颁布日期：2026年9月3日

"""用最终权重复算附件2验证/测试指标；不训练、不重新选择模型。"""
import argparse
import os
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data-root', type=Path, required=True)
    ap.add_argument('--encoder', type=Path, required=True)
    ap.add_argument('--split', choices=['valid', 'test'], default='valid')
    ap.add_argument('--scope', choices=['clean', 'six', 'stress'], default='clean')
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    args = ap.parse_args()
    if args.scope == 'six' and args.split != 'valid':
        ap.error('六场景选型指标只在验证集上计算')
    os.environ['P2_DATA_ROOT'] = str(args.data_root.resolve())
    os.environ['P2_ENCODER'] = str(args.encoder.resolve())
    from common import PACKAGE, DATA, sha256
    from data import load_attachment2, adapt, TextEncoder, make_view, torch_view, VALID_PROFILES
    from evaluate import scenarios
    from train import metrics
    from inference import load_ensemble, ensemble_predict
    import numpy as np
    import pandas as pd
    import torch
    import json

    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    cfg = json.loads((PACKAGE / '配置/inference.json').read_text())
    hashes = json.loads((PACKAGE / '配置/input_sha256.json').read_text())
    assert sha256(DATA / '附件2-数据集特征文件/aligned_50.pkl') == hashes['aligned_50']
    assert sha256(args.encoder / 'model.safetensors') == cfg['encoder']['weights_sha256']
    assert sha256(PACKAGE / '模型/normalization.npz') == cfg['normalization_sha256']
    raw = load_attachment2()[args.split]
    base = adapt(raw)
    encoder = TextEncoder(args.device)
    clean_text = encoder.encode(base, base['o'])
    with np.load(PACKAGE / '模型/normalization.npz') as z:
        stats = dict(z)
    models = load_ensemble(cfg, args.device)
    profiles = (VALID_PROFILES if args.scope == 'six' else
                list(scenarios()) if args.scope == 'stress' else list(scenarios())[:1])
    rows = []
    for j, profile in enumerate(profiles):
        seed = 20260924 if args.scope == 'six' else 41000 + profile.get('repeat', 0)
        view = make_view(base, encoder, stats, profile, seed, clean_text)
        out, _ = ensemble_predict(models, torch_view(view, args.device), cfg)
        m = metrics(raw['classification_labels'].astype(int), raw['regression_labels'], out[:,:3], out[:,3])
        rows.append(dict(model='old_gated', seed='ensemble', split=args.split, **profile, **m))
        print(f'{j+1}/{len(profiles)} {profile["name"]}: F1={m["macro_f1"]:.6f}, MAE={m["mae"]:.6f}', flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False)


if __name__ == '__main__':
    main()
