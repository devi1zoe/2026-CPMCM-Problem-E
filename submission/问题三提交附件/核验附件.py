# 本程序及代码是在人工智能工具辅助下完成的。

# 人工智能工具名称：GPT-6 Astra

# 版本/型号：gpt-6-astra

# 开发机构/公司：OpenAI

# 版本颁布日期：2026年9月3日

"""校验文件、20条预测、60条证据及可选的数值复算结果。"""
from pathlib import Path
import argparse
import collections
import csv
import hashlib
import json
import math


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--numerical', action='store_true', help='用NumPy复算Shapley及验证集删除指标')
    ap.add_argument('--recomputed', type=Path, help='另一次专项复算的results目录')
    args = ap.parse_args()
    root = Path(__file__).resolve().parent
    if (root / '文件校验清单.json').exists():
        spec = json.loads((root / '文件校验清单.json').read_text())['sha256']
        for name, digest in spec.items():
            p = root / name
            assert p.is_file() and hashlib.sha256(p.read_bytes()).hexdigest() == digest, name
        print(f'文件哈希通过：{len(spec)}个文件')
    else:
        raise FileNotFoundError('缺少文件校验清单.json')
    results = root / 'step3/results'
    def read(path):
        with path.open(encoding='utf-8-sig', newline='') as f:
            return list(csv.DictReader(f))
    rows = read(results / '附件4_问题三预测与解释.csv')
    cards = json.loads((results / 'special_explanations.json').read_text())
    ids = [f'{i:02d}' for i in range(1,21)]
    assert [r['sample_id'] for r in rows] == ids == [r['sample_id'] for r in cards]
    assert collections.Counter(r['polarity'] for r in rows) == dict(negative=6, neutral=4, positive=10)
    assert collections.Counter(r['main_modality'] for r in rows) == dict(text=17, vision=3)
    assert [r['sample_id'] for r in cards if r['requires_timing_review']] == ['15','16']
    timed = collections.Counter()
    labels = ['negative','neutral','positive']
    for row, card in zip(rows,cards):
        sid = row['sample_id']
        assert math.isfinite(float(row['intensity'])) and -3 <= float(row['intensity']) <= 3
        assert abs(float(row['intensity'])-card['intensity']) < 1e-7
        assert labels[card['class_id']] == row['polarity']
        assert labels[max(range(3), key=card['probabilities'].__getitem__)] == row['polarity']
        evidence = json.loads(row['evidence'])
        assert evidence == card['evidence'] and len(evidence) == 3
        assert json.loads((results / f'cards/{sid}.json').read_text()) == card
        mapping = json.loads((results / f'mapping/{sid}.json').read_text())
        assert card['raw_text'] == mapping['raw_text']
        for name, e in zip(['text','audio','vision'],evidence):
            assert e['modality'] == name and e['fixed_target_class'] == card['class_id']
            assert 1 <= e['slot_start'] < e['slot_end'] <= 49
            assert card['raw_text'][e['char_start']:e['char_end']] == e['text']
            if e['time_status'] == 'estimated_CTC':
                assert not card['requires_timing_review']
                assert 0 <= e['start_seconds'] < e['end_seconds'] <= card['duration'] + .001
                timed[name] += 1
            else:
                assert e['start_seconds'] is None and e['end_seconds'] is None
        assert (results / card['keyframe']['image']).is_file()
    assert timed == dict(text=17,audio=15,vision=16)
    print('20条预测、60条证据、字符范围和时间状态校验通过')

    if args.numerical:
        import numpy as np
        def shapley(values,target):
            n = len(values)
            x = values[np.arange(n)[:,None],np.arange(8)[None,:],target[:,None]]
            scores = np.stack([x,values[:,:,3]],-1)
            phi = np.zeros((n,3,2),dtype=np.float64)
            for j in range(3):
                for mask in range(8):
                    if not mask & (1<<j):
                        phi[:,j] += [1/3,1/6,1/3][mask.bit_count()] * (scores[:,mask|(1<<j)]-scores[:,mask])
            assert np.abs(phi.sum(1) - (scores[:,7]-scores[:,0])).max() < 2e-6
            return phi
        with np.load(results / 'special_explanation_arrays.npz') as a:
            phi = shapley(a['values'].mean(0),a['target'])
            assert np.allclose(phi,a['phi'],atol=1e-7)
            assert np.max(np.abs(a['ig_residual'])) <= .001
            for i,row in enumerate(rows):
                for j,name in enumerate(['text','audio','vision']):
                    assert abs(float(row['contribution_'+name])-phi[i,j,0]) < 1e-7
        with np.load(results / 'validation_coalitions.npz') as a:
            for key in ['values','reference_values']:
                assert a[key].shape == (3,728,8,4)
                for seed in a[key]:shapley(seed,a['target'])
            target = a['target'].copy()
        with np.load(results / 'validation_ig.npz') as a:
            assert len(a['ig']) == 728 and np.abs(a['residual']).max() <= .001
        with np.load(results / 'validation_predictions.npz') as a:
            full = a['prob'][np.arange(728),target]
        with np.load(results / 'validation_probe_outputs.npz') as a:
            cube = a['outputs']; assert cube.shape == (728,3,6,3,4)
            assert a['specs'].shape == (39312,5)
            table = read(results / 'validation_fidelity.csv');assert len(table) == 2184
            for i in range(728):
                for m in range(3):
                    r = table[i*3+m]
                    assert abs(full[i]-cube[i,m,0,0,target[i]]-float(r['deletion_drop'])) < 1e-7
                    assert abs(cube[i,m,1:,0,target[i]].mean()-cube[i,m,0,0,target[i]]-float(r['deletion_advantage'])) < 1e-7
        print('三模态Shapley完备性、728条IG积分残差及2184行删除指标复算通过')
    if args.recomputed:
        new = read(args.recomputed / '附件4_问题三预测与解释.csv')
        assert [r['sample_id'] for r in new] == ids
        differences = []
        for a,b in zip(rows,new):
            for k in ['polarity','main_modality']:assert a[k] == b[k]
            for k in ['intensity','contribution_text','contribution_audio','contribution_vision']:
                differences.append(abs(float(a[k])-float(b[k])))
            for x,y in zip(json.loads(a['evidence']),json.loads(b['evidence'])):
                for k in ['modality','slot_start','slot_end','char_start','char_end','text','time_status','start_seconds','end_seconds','role']:
                    assert x[k] == y[k], (a['sample_id'],k)
                differences.append(abs(x['class_probability_drop']-y['class_probability_drop']))
        assert max(differences) <= 1e-4, max(differences)
        print(f'专项重新推理与证据计算一致；最大数值差：{max(differences):.9g}')


if __name__ == '__main__':
    main()
