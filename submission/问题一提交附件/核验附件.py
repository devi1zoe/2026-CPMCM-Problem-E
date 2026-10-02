# 本程序及代码是在人工智能工具辅助下完成的。

# 人工智能工具名称：GPT-6 Astra

# 版本/型号：gpt-6-astra

# 开发机构/公司：OpenAI

# 版本颁布日期：2026年9月3日

"""离线核验提交文件哈希、100条样本覆盖及论文关键统计；无需GPU或原视频。"""
from pathlib import Path
from collections import Counter
import csv
import hashlib
import json
import numpy as np

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'step1/results'


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def csv_rows(name):
    with (DATA / name).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def matches_recorded_code(path, expected):
    """保留原实验代码哈希；仅允许剥离本次逐字添加的AI声明。"""
    data = path.read_bytes()
    annotation = '# 本程序及代码是在人工智能工具辅助下完成的。\n\n# 人工智能工具名称：GPT-6 Astra\n\n# 版本/型号：gpt-6-astra\n\n# 开发机构/公司：OpenAI\n\n# 版本颁布日期：2026年9月3日\n\n'.encode('utf-8')
    return (hashlib.sha256(data).hexdigest() == expected or
            (data.startswith(annotation) and
             hashlib.sha256(data[len(annotation):]).hexdigest() == expected))


def main():
    manifest = json.loads((ROOT / '文件校验清单.json').read_text(encoding='utf-8'))
    for item in manifest['files']:
        path = ROOT / item['path']
        assert path.is_file(), item['path']
        assert path.stat().st_size == item['bytes'], item['path']
        assert sha256(path) == item['sha256'], item['path']
    inputs = json.loads((DATA / 'input_manifest.json').read_text(encoding='utf-8'))
    inputs = {row['sample_id']: row for row in inputs}
    directories = sorted((DATA / 'samples').iterdir())
    assert len(inputs) == len(directories) == 100
    assert set(inputs) == {p.name for p in directories}
    catalog = {row['样本编号']: row for row in csv_rows('全部样本特征交付清单.csv')}
    assert set(catalog) == set(inputs)
    counts = Counter()
    risk_ids = set()
    required = {'features.npz', 'metadata.json', 'quality.json', 'quality_masks.npz',
                'windows_0.2s.npz', 'windows_0.5s.npz'}
    for directory in directories:
        assert required <= {p.name for p in directory.iterdir()}
        meta = json.loads((directory / 'metadata.json').read_text(encoding='utf-8'))
        quality = json.loads((directory / 'quality.json').read_text(encoding='utf-8'))
        sid = directory.name
        for key in ['sample_id', 'video_id', 'clip_id', 'raw_text', 'label', 'annotation', 'source_path', 'source_sha256']:
            assert meta[key] == inputs[sid][key], (sid, key)
        assert meta['config_sha256'] == sha256(ROOT / 'step1/config.json')
        for name, checksum in meta['code_sha256'].items():
            assert matches_recorded_code(ROOT / 'step1' / name, checksum)
        assert meta['padding_count'] == meta['truncated_words'] == 0
        n = len(meta['words'])
        with np.load(directory / 'features.npz', allow_pickle=False) as z, np.load(directory / 'quality_masks.npz', allow_pickle=False) as q:
            for name in z.files:
                if z[name].dtype.kind in 'fc':
                    assert np.isfinite(z[name]).all(), (sid, name)
            for mod, dim in [('text', 768), ('audio', 25), ('vision', 52)]:
                assert z[mod].shape == (n, dim)
                assert z[mod + '_valid'].shape == (n,)
                np.testing.assert_array_equal(q[mod + '_aligned_usable'], q['timing_usable'] & z[mod + '_valid'])
            assert z['word_bounds'].shape == (n, 2)
            conf = z['word_confidence']
            risk = bool(conf.mean() < .2 or (conf < .1).mean() > .5 or meta['diagnostic_wer'] > .8 or meta['alignment_error'] is not None)
            assert risk == quality['requires_transcript_review']
            np.testing.assert_array_equal(q['timing_usable'], z['alignment_valid'] & (conf >= .1) & ~np.full(n, risk))
            if risk:
                risk_ids.add(sid)
            joint = q['text_aligned_usable'] & q['audio_aligned_usable'] & q['vision_aligned_usable']
            row = catalog[sid]
            for name, value in [('原文词数', n), ('视觉可观测词数', z['vision_valid'].sum()),
                                ('通过对齐筛选词数', q['timing_usable'].sum()), ('其中三种模态都可观测词数', joint.sum())]:
                assert int(row[name]) == value, (sid, name)
            np.testing.assert_allclose([float(row['片段时长（秒）']), float(row['音频时长（秒）'])], [meta['duration'], meta['audio_duration']])
            counts.update(samples=1, words=n, audio_records=len(z['audio_frames']),
                          sampled_frames=len(z['vision_pts']), valid_face_frames=int(z['vision_frame_valid'].sum()),
                          low_energy=int(z['audio_silence'].sum()), timing_usable=int(q['timing_usable'].sum()), joint_usable=int(joint.sum()))
            for width in [.2, .5]:
                with np.load(directory / f'windows_{width:.1f}s.npz', allow_pickle=False) as w:
                    assert w['text_word_weights'].shape == (len(w['bounds']), n)
                    assert w['audio'].shape == (len(w['bounds']), 25)
                    assert w['vision'].shape == (len(w['bounds']), 52)
    assert dict(counts) == dict(samples=100, words=1932, audio_records=77261, sampled_frames=3973,
                               valid_face_frames=3253, low_energy=9918, timing_usable=1545, joint_usable=1387)
    assert len(csv_rows('sample_summary.csv')) == 100
    assert len(csv_rows('modality_summary.csv')) == 300
    assert len(csv_rows('word_alignment.csv')) == 1932
    review = csv_rows('人工审核_21条样本.csv')
    assert len(review) == 21 and {row['sample_id'] for row in review} == risk_ids
    assert sum(int(row['words']) for row in review) == 317
    assert Counter(row['manual_category'] for row in review) == Counter({'音轨与文本不对应':11, '无对应朗读':5, '非英语语音':3, '分段内容不对应':1, '音频卡顿、局部对应':1})
    events = [json.loads(line) for line in (DATA / 'processing_log.jsonl').read_text().splitlines() if line.strip()]
    assert len(events) == 100 and {r['sample_id'] for r in events} == set(inputs)
    assert all(r['status'] == 'processed' for r in events)
    assert json.loads((DATA / 'failures.json').read_text()) == []
    print(json.dumps({'文件哈希': '全部通过', '样本与字段核验': '全部通过', '统计': dict(counts),
                      '说明': '检查文件完整性和记录一致性，不评价真实词界准确率。'}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
