# 本程序及代码是在人工智能工具辅助下完成的。

# 人工智能工具名称：GPT-6 Astra

# 版本/型号：gpt-6-astra

# 开发机构/公司：OpenAI

# 版本颁布日期：2026年9月3日

"""按固定标准化参数重建完整输入缓存，不改动模型或选型配置。"""
from common import *
from data import load_attachment2, adapt, TextEncoder, make_view
import argparse
import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--splits', nargs='+', choices=['train', 'valid'], default=['valid'])
    args = ap.parse_args()
    torch.set_num_threads(4)
    protocol = json.loads((ROOT / 'protocol.json').read_text())
    assert sha256(DATA / '附件2-数据集特征文件/aligned_50.pkl') == protocol['input_sha256']
    stats = dict(np.load(ROOT / 'checkpoints/normalization.npz'))
    encoder = TextEncoder()
    raw = load_attachment2()
    (ROOT / 'cache').mkdir(parents=True, exist_ok=True)
    for split in args.splits:
        view = make_view(adapt(raw[split]), encoder, stats, dict(subset='', rate=0., position='random'), 0)
        np.savez(ROOT / f'cache/{split}_clean.npz', **view)
        np.savez(ROOT / f'cache/{split}_labels.npz', y=raw[split]['regression_labels'].astype(np.float32),
                 c=raw[split]['classification_labels'].astype(np.int64))
        print(f'{split}缓存已重建：{len(view["p"])}条', flush=True)


if __name__ == '__main__':
    main()
