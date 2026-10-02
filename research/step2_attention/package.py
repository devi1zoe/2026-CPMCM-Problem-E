"""Check completed artifacts and package reproducible small deliverables."""
from common import *
from engine import scenarios, score
import numpy as np
import pandas as pd
import torch
import zipfile
import re


def read(path):
    return json.loads((ROOT/path).read_text())


def main():
    proto=read('protocol.json');ph=sha256(ROOT/'protocol.json');lock=read('selection_lock.json')
    assert lock['protocol_sha256']==ph
    assert sha256(DATA/'附件2-数据集特征文件/aligned_50.pkl')==proto['source_sha256']
    assert sha256(ROOT/'checkpoints/normalization.npz')==proto['normalization_sha256']
    assert read('results/verification.json')['passed']
    assert read('results/verification.json')['protocol_sha256']==ph
    fits=[]
    for name in proto['models']:
        for seed in proto['seeds']:
            stem=f'{name}_{seed}';done=read(f'results/completed_{stem}.json')
            path=ROOT/f'checkpoints/{stem}.pt';ck=torch.load(path,map_location='cpu',weights_only=True)
            assert done['checkpoint_sha256']==sha256(path)
            assert done['protocol_sha256']==ck['protocol_sha256']==ph
            assert ck['model_sha256']==sha256(ROOT/'model.py')
            history=read(f'results/history_{stem}.json')
            assert len(history)==done['epochs'] and 1<=len(history)<=proto['epochs']
            assert abs(ck['J']-min(h['J'] for h in history))<1.1e-6
            assert history[ck['epoch']-1]['J']==ck['J']
            fits.append(dict(model=name,seed=seed,best_epoch=ck['epoch'],epochs=done['epochs'],parameters=done['parameters'],J=ck['J']))
    for f,h in lock['weights'].items():assert sha256(ROOT/'checkpoints'/f)==h
    profiles=list(scenarios());assert len(profiles)==217
    assert len({p['name'] for p in profiles if p['primary']})==112
    counts={}
    for split,n,names in [('valid',728,['old_gated']+proto['models']),('test',727,['old_gated',lock['selected_attention']])]:
        done=read(f'results/completed_evaluation_{split}.json')
        assert done['protocol_sha256']==ph and done['models']==names
        assert read(f'results/{split}_profiles.json')==profiles
        for f,h in done['weights'].items():assert sha256(ROOT/'checkpoints'/f)==h
        df=pd.read_csv(ROOT/f'results/{split}_metrics.csv')
        assert len(df)==len(names)*4*217==done['rows']
        assert not df.duplicated(['model','seed','name','repeat']).any()
        assert np.isfinite(df[['accuracy','macro_f1','mae','pearson']].to_numpy()).all()
        for name in names:
            d=np.load(ROOT/f'results/{split}_{name}_predictions.npz')
            out=d['out'];assert out.shape==(217,3,n,4) and np.isfinite(out).all()
            assert np.allclose(out[...,:3].sum(-1),1,atol=1e-6)
            assert ((out[...,:3]>=0)&(out[...,:3]<=1)).all() and (abs(out[...,3])<=3).all()
            assert len(set(d['ids']))==n
        counts[split]=len(df)
    six=pd.read_csv(ROOT/'results/validation_six.csv');js={}
    for name in proto['attention_candidates']:
        rows=[six[(six.model==name)&(six.profile==p['name'])].iloc[0] for p in proto['validation_profiles']]
        js[name]=score(rows)
    assert min(js,key=js.get)==lock['selected_attention']
    special=pd.read_csv(ROOT/'results/附件3_注意力模型预测.csv')
    sources=list((DATA/'附件3-模态缺失特征样本/对齐版本').glob('*.pkl'))
    assert len(special)==30 and special.sample_id.is_unique
    assert set(special.sample_id)=={f.stem for f in sources}
    assert set(special.polarity)<={'negative','neutral','positive'}
    assert np.isfinite(special.intensity).all() and (abs(special.intensity)<=3).all()
    assert read('results/special_audit.json')['sources']=={f.name:sha256(f) for f in sources}
    for condition in ['clean','missing']:
        data=np.load(ROOT/f'results/attention_{condition}.npz');a=data['cross'][0]
        valid=(data['o']&data['p'][:,None]).reshape(150)
        modalities=np.arange(3).repeat(50)
        legal=(modalities[:,None]!=modalities[None,:])&valid[:,None]&valid[None,:]
        assert np.isfinite(a).all() and (a[~legal]==0).all()
        assert np.allclose(a.sum(-1),legal.any(-1).astype(float),atol=1e-6)
    page=(ROOT/'results/attention_review.html').read_text()
    for target in re.findall(r'(?:src|href)="([^"]+)"',page):
        assert (ROOT/'results'/target).is_file(),target
    assert len(list((ROOT/'results/figures').glob('*.png')))==8
    assert (ROOT/'注意力实验报告.md').is_file()
    for f,h in read('cache_manifest.json').items():
        assert sha256(ROOT/'cache'/f)==h
    audit=dict(passed=True,protocol_sha256=ph,selection_lock_sha256=sha256(ROOT/'selection_lock.json'),
        completed_fits=fits,metric_rows=counts,realizations=217,primary_scenarios=112,special_predictions=30,
        attention_legal_connections=True,html_local_links=True,selection_recomputed_from_validation=True,
        previous_outputs_replaced=False)
    save_json(ROOT/'results/completion_audit.json',audit)
    # Raw arrays remain in the workspace and can be regenerated from included code.
    # Avoid manifest/self-reference and logs that might still be written by this command.
    excluded={'cache','__pycache__','logs'}
    files=[p for p in sorted(ROOT.rglob('*')) if p.is_file() and not p.is_symlink()
        and not any(x in excluded for x in p.relative_to(ROOT).parts)
        and p.suffix not in ['.zip','.pyc'] and not p.name.endswith('_predictions.npz')
        and p.name not in ['delivery_manifest.json','package_size.json']]
    manifest={str(p.relative_to(ROOT)):dict(bytes=p.stat().st_size,sha256=sha256(p)) for p in files}
    save_json(ROOT/'delivery_manifest.json',manifest)
    output=ROOT/'注意力方案交付.zip'
    with zipfile.ZipFile(output,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for p in files+[ROOT/'delivery_manifest.json']:z.write(p,'step2_attention/'+str(p.relative_to(ROOT)))
    with zipfile.ZipFile(output) as z:
        assert z.testzip() is None
        for f,d in manifest.items():assert hashlib.sha256(z.read('step2_attention/'+f)).hexdigest()==d['sha256']
    comparisons={}
    for f in ['step1/问题一交付.zip','step3/问题三交付.zip']:
        p=PROJECT/f
        if p.is_file():comparisons[f]=p.stat().st_size
    combined=sum(comparisons.values())+output.stat().st_size
    save_json(ROOT/'results/package_size.json',dict(zip_bytes=output.stat().st_size,zip_sha256=sha256(output),
        neighboring_packages=comparisons,aggregate_bytes=combined,under_50_decimal_mb=combined<50_000_000,
        note='Size comparison only. This experimental variant does not replace the adopted Problem 2 package.'))
    print(json.dumps(dict(passed=True,fits=len(fits),metric_rows=counts,zip_bytes=output.stat().st_size,aggregate_bytes=combined),ensure_ascii=False),flush=True)


if __name__=='__main__':main()
