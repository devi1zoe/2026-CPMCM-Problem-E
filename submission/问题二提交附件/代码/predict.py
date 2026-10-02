# 本程序及代码是在人工智能工具辅助下完成的。

# 人工智能工具名称：GPT-6 Astra

# 版本/型号：gpt-6-astra

# 开发机构/公司：OpenAI

# 版本颁布日期：2026年9月3日

"""使用提交权重对附件3对齐特征推理；不读取标签或训练缓存。"""
import argparse
import os
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input-dir', type=Path, required=True, help='附件3的“对齐版本”目录')
    ap.add_argument('--encoder', type=Path, required=True, help='锁定版本的DistilBERT目录')
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--details', type=Path, help='可选：三类概率、种子分歧和观测比例')
    ap.add_argument('--device', default='cuda', choices=['cuda', 'cpu'])
    args = ap.parse_args()
    os.environ['P2_ENCODER'] = str(args.encoder.resolve())
    from common import PACKAGE, sha256, save_json
    from data import adapt, make_view, TextEncoder, torch_view
    from inference import load_ensemble, ensemble_predict
    import json
    import pickle
    import numpy as np
    import pandas as pd
    import torch

    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    config = json.loads((PACKAGE / '配置/inference.json').read_text())
    expected = json.loads((PACKAGE / '配置/input_sha256.json').read_text())['attachment3']
    files = sorted(args.input_dir.glob('*.pkl'))
    if [p.name for p in files] != sorted(expected):
        raise ValueError('输入必须为附件3对齐版本的30个原始pkl文件')
    if any(sha256(p) != expected[p.name] for p in files):
        raise ValueError('附件3输入哈希与赛题对齐版本不一致')
    if sha256(args.encoder / 'model.safetensors') != config['encoder']['weights_sha256']:
        raise ValueError('冻结文本编码器权重版本不一致')
    rows = []
    for p in files:
        with p.open('rb') as f:
            row = pickle.load(f)['test']
        for name, shape in [('text_bert', (1, 3, 50)), ('audio', (1, 50, 74)), ('vision', (1, 50, 35))]:
            if np.asarray(row[name]).shape != shape:
                raise ValueError(f'{p.name}: {name}形状不符')
        rows.append(row)
    inputs = {k: np.concatenate([r[k] for r in rows]) for k in ['text_bert', 'audio', 'vision']}
    base = adapt(inputs)
    norm = PACKAGE / '模型/normalization.npz'
    if sha256(norm) != config['normalization_sha256']:
        raise ValueError('训练集标准化参数校验失败')
    with np.load(norm) as z:
        stats = dict(z)
    view = make_view(base, TextEncoder(args.device), stats,
                     dict(subset='', rate=0., position='none'), 0)
    models = load_ensemble(config, args.device)
    out, std = ensemble_predict(models, torch_view(view, args.device), config)
    labels = config['classes']
    records = [dict(sample_id=p.stem, polarity=labels[int(out[i, :3].argmax())],
                    intensity=float(out[i, 3])) for i, p in enumerate(files)]
    if args.output.resolve() == (PACKAGE / '附件3_问题二预测.csv').resolve():
        raise ValueError('请另选复算输出路径，保留提交的参考预测文件')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(args.output, index=False)
    if args.details:
        detail = []
        for i, r in enumerate(records):
            detail.append(dict(r, **{f'prob_{k}': float(out[i,j]) for j,k in enumerate(labels)},
                               intensity_seed_std=float(std[i]),
                               all_modalities_missing=bool(not base['o'][i].any()),
                               **{f'observed_{k}': float(view['quality'][i,j,0]) for j,k in enumerate('TAV')}))
        args.details.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(detail).to_csv(args.details, index=False)
    ref = pd.read_csv(PACKAGE / '附件3_问题二预测.csv')
    delta = np.abs(out[:,3] - ref.intensity.to_numpy())
    save_json(args.output.with_suffix('.核验.json'), dict(
        n=len(records), device=args.device, backend=config['backend'],
        input_sha256_match=True, encoder_sha256_match=True,
        ids_match=[r['sample_id'] for r in records] == ref.sample_id.tolist(),
        classes_match=[r['polarity'] for r in records] == ref.polarity.tolist(),
        max_intensity_difference=float(delta.max())))
    print(f'完成{len(records)}条预测：{args.output}')


if __name__ == '__main__':
    main()
