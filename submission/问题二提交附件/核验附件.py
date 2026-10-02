# 本程序及代码是在人工智能工具辅助下完成的。

# 人工智能工具名称：GPT-6 Astra

# 版本/型号：gpt-6-astra

# 开发机构/公司：OpenAI

# 版本颁布日期：2026年9月3日

"""校验提交文件完整性和30条预测；仅使用Python标准库。"""
from pathlib import Path
import argparse
import collections
import csv
import hashlib
import json
import math


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--prediction', type=Path, help='可选：与提交结果比较的复算CSV')
    ap.add_argument('--atol', type=float, default=1e-3)
    args = ap.parse_args()
    root = Path(__file__).resolve().parent
    manifest = json.loads((root / '文件校验清单.json').read_text())
    for name, digest in manifest['sha256'].items():
        p = root / name
        if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest() != digest:
            raise ValueError(f'文件缺失或发生变更：{name}')
    def read(p):
        with p.open(encoding='utf-8-sig', newline='') as f:
            reader = csv.DictReader(f)
            if reader.fieldnames != ['sample_id', 'polarity', 'intensity']:
                raise ValueError('预测CSV字段不符')
            rows = list(reader)
        assert [r['sample_id'] for r in rows] == [f'附件3_{i:02d}' for i in range(1,31)]
        for r in rows:
            v = float(r['intensity'])
            assert math.isfinite(v) and -3 <= v <= 3
            assert r['polarity'] in ['negative', 'neutral', 'positive']
        return rows
    ref = read(root / '附件3_问题二预测.csv')
    assert dict(collections.Counter(r['polarity'] for r in ref)) == dict(negative=6, neutral=6, positive=18)
    with (root / '实验记录/附件3_预测明细.csv').open(encoding='utf-8-sig', newline='') as f:
        details = list(csv.DictReader(f))
    assert len(details) == len(ref)
    labels = ['negative', 'neutral', 'positive']
    for r,d in zip(ref, details):
        assert r['sample_id'] == d['sample_id'] and r['polarity'] == d['polarity']
        assert abs(float(r['intensity']) - float(d['intensity'])) < 1e-7
        prob = [float(d['prob_'+k]) for k in labels]
        assert all(math.isfinite(x) and 0 <= x <= 1 for x in prob)
        assert abs(sum(prob) - 1) < 1e-5
        assert labels[max(range(3), key=prob.__getitem__)] == r['polarity']
    if args.prediction:
        new = read(args.prediction)
        assert all(a['polarity'] == b['polarity'] for a,b in zip(ref,new)), '复算类别不一致'
        error = max(abs(float(a['intensity'])-float(b['intensity'])) for a,b in zip(ref,new))
        assert error <= args.atol, f'强度最大差异{error}超过容差{args.atol}'
        print(f'复算30条类别一致；强度最大绝对差：{error:.9g}')
    print(f'文件哈希校验通过：{len(manifest["sha256"])}个文件；预测30条，无重复或缺失。')


if __name__ == '__main__':
    main()
