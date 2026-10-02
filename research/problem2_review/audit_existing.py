"""Read existing artifacts and independently audit them; write only in this directory."""
from pathlib import Path
import hashlib
import json
import pickle
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent


def digest(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def metrics(c, y, out):
    cls = out[:, :3].argmax(-1)
    cm = np.bincount(c * 3 + cls, minlength=9).reshape(3, 3)
    den = cm.sum(0) + cm.sum(1)
    f1 = np.divide(2 * cm.diagonal(), den, out=np.zeros(3), where=den != 0)
    return {'accuracy': float((cls == c).mean()), 'macro_f1': float(f1.mean()),
            'mae': float(np.abs(y - out[:, 3]).mean()),
            'pearson': float(np.corrcoef(y, out[:, 3])[0, 1]),
            'neutral_recall': float(cm[1, 1] / cm[1].sum())}


def main():
    audit = {'scope': 'Read-only audit of existing experiments; no retraining or model selection changes.'}
    tables = []
    prediction_checks = []
    tasks = [('step2_v2', 'valid', n) for n in ['old_gated','text','audio','vision','fixed','gated','aux','msd','cce','imtd','emc']]
    tasks += [('step2_refine', 'valid', n) for n in ['dual_gate','late_fusion']]
    tasks += [('step2_v2', 'test', n) for n in ['old_gated','text','gated']]
    for directory, split, name in tasks:
        folder = PROJECT / directory
        with np.load(folder / f'results/{split}_{name}_predictions.npz') as f:
            out, c, y = f['out'], f['c'], f['y']
        profiles = json.loads((folder / f'results/{split}_profiles.json').read_text())
        saved = pd.read_csv(folder / f'results/{split}_{name}_metrics.csv')
        ens = saved[saved.seed == 'ensemble'].reset_index(drop=True)
        assert len(out) == len(profiles) == len(ens) == 217
        recalculated = [metrics(c, y, p.mean(0)) for p in out]
        max_errors = {k: float(np.max(np.abs(ens[k].to_numpy() - np.array([r[k] for r in recalculated])))) for k in recalculated[0]}
        assert max(max_errors.values()) < 1e-6, (name, max_errors)
        assert np.isfinite(out).all()
        prediction_checks.append(dict(directory=directory, split=split, model=name, inputs=len(out), max_errors=max_errors))
        clean = ens[ens.name == 'clean'].iloc[0]
        primary = ens[ens.primary]
        robust = primary.groupby('name')[['macro_f1','mae']].mean().mean()
        assert primary.name.nunique() == 112
        tables.append(dict(directory=directory, split=split, model=name,
                           **{k:float(clean[k]) for k in ['accuracy','macro_f1','mae','pearson','neutral_recall']},
                           robust_f1=float(robust.macro_f1), robust_mae=float(robust.mae)))
        if name == 'old_gated' and split == 'valid':
            primary.groupby(['subset','rate','position'])[['macro_f1','mae']].mean().reset_index().to_csv(ROOT/'final_model_valid_missing.csv',index=False)
            primary.groupby(['subset','rate','position'])[['macro_f1','mae']].mean().groupby('subset').mean().to_csv(ROOT/'final_model_valid_by_modality.csv')
    audit['prediction_metric_checks'] = prediction_checks
    pd.DataFrame(tables).to_csv(ROOT/'unified_comparison.csv',index=False)

    old = pd.read_csv(PROJECT/'step2/results/test_metrics.csv')
    rows=[]
    for name in ['fixed_clean','fixed_aug','mask_aug','gated_aug']:
        e=old[(old.seed=='ensemble')&(old.model==name)]
        clean=e[e.name=='clean'].iloc[0]
        miss=e[e.name=='TAV_r50_random'][['macro_f1','mae']].mean()
        rows.append(dict(model=name,clean_f1=clean.macro_f1,clean_mae=clean.mae,missing_f1=miss.macro_f1,missing_mae=miss.mae))
    pd.DataFrame(rows).to_csv(ROOT/'old_ablation_ensemble.csv',index=False)

    checks={}
    for directory in ['step2_v2','step2_refine']:
        folder=PROJECT/directory
        lock=json.loads((folder/'selection_lock.json').read_text())
        checks[directory+'_protocol']=digest(folder/'protocol.json')==lock['protocol_sha256']
    lock=json.loads((PROJECT/'step2_v2/selection_lock.json').read_text())
    for p,h in lock['checkpoints'].items():checks[p]=digest(PROJECT/p)==h
    manifest=json.loads((PROJECT/'step2_v2/deployment/manifest.json').read_text())
    for p,h in manifest['checkpoints'].items():checks['deployment/'+p]=digest(PROJECT/'step2_v2/deployment'/p)==h
    checks['selection_lock']=digest(PROJECT/'step2_v2/selection_lock.json')==manifest['selection_sha256']
    audit['code_manifest_checks']={}
    for directory in ['step2_v2','step2_refine']:
        records=json.loads((PROJECT/directory/'code_manifest.json').read_text())
        bad=[p for p,h in records.items() if digest(PROJECT/directory/p)!=h]
        audit['code_manifest_checks'][directory]=dict(files=len(records),mismatches=bad)
        assert not bad
    audit['hash_checks']=checks
    assert all(checks.values())

    d=pd.read_csv(PROJECT/'step2_v2/results/附件3_问题二预测.csv')
    d0=pd.read_csv(PROJECT/'step2/results/附件3_问题二预测.csv')
    assert list(d.columns)==['sample_id','polarity','intensity'] and len(d)==d.sample_id.nunique()==30
    assert d.intensity.between(-3,3).all() and set(d.polarity)<=set(['negative','neutral','positive'])
    audit['special_csv']=dict(n=len(d),classes=d.polarity.value_counts().to_dict(),
                              same_classes_as_original=bool((d.polarity==d0.polarity).all()),
                              max_intensity_change_from_original=float(np.abs(d.intensity-d0.intensity).max()))
    if (ROOT/'附件3_独立复算.csv').exists():
        new=pd.read_csv(ROOT/'附件3_独立复算.csv')
        audit['independent_special_inference']=dict(ids_match=bool((d.sample_id==new.sample_id).all()),
            classes_match=bool((d.polarity==new.polarity).all()),
            max_intensity_difference=float(np.abs(d.intensity-new.intensity).max()))
    audit['second_stage_active']={}
    for name,loss in [('imtd','kd'),('emc','emc')]:
        audit['second_stage_active'][name]={}
        for seed in [17,29,43]:
            history=json.loads((PROJECT/f'step2_v2/results/history_{name}_{seed}.json').read_text())
            stage=[r for r in history if r['stage']==2]
            audit['second_stage_active'][name][str(seed)]=dict(epochs=len(stage),max_loss=max(r['loss'][loss] for r in stage))
    # The second-stage variants should retain the identical first-stage CCE outputs.
    with np.load(PROJECT/'step2_v2/results/valid_cce_predictions.npz') as f:a=f['out']
    for name in ['imtd','emc']:
        with np.load(PROJECT/f'step2_v2/results/valid_{name}_predictions.npz') as f:b=f['out']
        audit[name+'_equal_cce_predictions']=bool(np.array_equal(a,b))

    source=PROJECT/'E题_材料阅读/E题数据/附件2-数据集特征文件/aligned_50.pkl'
    protocol=json.loads((PROJECT/'step2_v2/protocol.json').read_text())
    assert digest(source)==protocol['source_sha256']
    with source.open('rb') as f:raw=pickle.load(f)
    videos={}
    audit['source_splits']={}
    for split,d in raw.items():
        c=np.asarray(d['classification_labels']).astype(int);y=np.asarray(d['regression_labels'])
        assert np.array_equal(np.sign(y).astype(int)+1,c)
        assert len(set(d['id']))==len(c)
        videos[split]=set(s.split('$_$')[0] for s in d['id'])
        audit['source_splits'][split]=dict(n=len(c),classes=np.bincount(c).tolist(),videos=len(videos[split]))
    audit['source_video_disjoint']=all(not videos[a]&videos[b] for a,b in [('train','valid'),('train','test'),('valid','test')])
    assert audit['source_video_disjoint']
    (ROOT/'audit_results.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2))
    print(json.dumps({'metric_files_checked':len(prediction_checks),'input_implementations_checked':217*len(prediction_checks),'hashes_pass':all(checks.values()),'special_csv':audit['special_csv'],'source_splits':audit['source_splits']},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
