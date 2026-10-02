"""核对学术精简稿新增的规则贡献和非退化样本统计，不改变原特征。"""
from pathlib import Path
import csv
import json
import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / 'results'
report = json.loads((HERE / 'data_audit.json').read_text())
assert report['status'] == 'passed'
sets = {k: set() for k in ['mean_below_0.2', 'majority_below_0.1', 'wer_above_0.8', 'no_full_path']}
for directory in (DATA / 'samples').iterdir():
    if not (directory / 'metadata.json').exists():
        continue
    meta = json.loads((directory / 'metadata.json').read_text())
    with np.load(directory / 'features.npz') as z:
        conditions = [z['word_confidence'].mean() < .2, (z['word_confidence'] < .1).mean() > .5,
                      meta['diagnostic_wer'] > .8, meta['alignment_error'] is not None]
        for name, condition in zip(sets, conditions):
            if condition:
                sets[name].add(directory.name)
assert [len(v) for v in sets.values()] == [19, 20, 21, 0]
assert all(v <= sets['wer_above_0.8'] for v in sets.values())
with (HERE / 'audited_window_comparison.csv').open(encoding='utf-8-sig') as f:
    rows = list(csv.DictReader(f))
constant = {r['sample_id'] for r in report['constant_features'] if r['modality'] == 'audio'}
assert len(constant) == 2
means = {}
for width in [.2, .5]:
    selected = [r for r in rows if float(r['window_seconds']) == width]
    assert len(selected) == 100
    assert all(float(r['audio_normalized_reconstruction_mse']) == 0 for r in selected if r['sample_id'] in constant)
    nonconstant = [float(r['audio_normalized_reconstruction_mse']) for r in selected if r['sample_id'] not in constant]
    assert len(nonconstant) == 98
    means[str(width)] = {'samples': len(nonconstant), 'audio_mse': float(np.mean(nonconstant))}
relative = {}
for modality in ['audio', 'vision']:
    stats = report['window_statistics'][modality + '_normalized_reconstruction_mse']
    relative[modality] = (1 - stats['0.2']['mean'] / stats['0.5']['mean']) * 100
cross = report['window_statistics']['word_boundary_crossing_ratio']
output = {
    'status': 'passed', 'source': '逐样本CTC记录与独立复算的200条窗口指标',
    'rule_counts': {k: len(v) for k, v in sets.items()},
    'all_rule_sets_contained_in_wer_set': True,
    'rule_sample_ids': {k: sorted(v) for k, v in sets.items()},
    'constant_audio_sample_ids': sorted(constant), 'nonconstant_audio_statistics': means,
    'relative_mse_reduction_percent': relative,
    'crossing_increase_percentage_points': (cross['0.2']['mean'] - cross['0.5']['mean']) * 100,
    'three_modal_usable_percent': report['counts']['joint_usable'] / report['counts']['words'] * 100,
    'manual_word_boundary_validation_added': False,
}
(HERE / 'analysis_statistics.json').write_text(json.dumps(output, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({k: v for k, v in output.items() if k != 'rule_sample_ids'}, ensure_ascii=False, indent=2))
