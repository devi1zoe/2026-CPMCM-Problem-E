"""Export inference-only weights and infer all 30 attachment-3 samples after selection lock."""
from common import *
from data import *
from engine import *
import pandas as pd

def export():
    lock=json.loads((ROOT/'selection_lock.json').read_text());outdir=ROOT/'deployment';outdir.mkdir(exist_ok=True)
    for s,path in zip([17,29,43],checkpoint_paths(lock['selected'])):
        assert sha256(path)==lock['checkpoints'][str(path.relative_to(PROJECT))]
        ck=torch.load(path,map_location='cpu',weights_only=True)
        if 'config' in ck:
            ck['training_config']=dict(ck['config']);ck['config']={**ck['config'],'aux':False,'imtd':False,'emc':False}
            ck['state_dict']={k:v for k,v in ck['state_dict'].items() if not k.startswith('aux_head.')}
        torch.save(ck,outdir/f'final_{s}.pt')
    return outdir

def main():
    torch.set_num_threads(4);lock_sha=sha256(ROOT/'selection_lock.json');lock=json.loads((ROOT/'selection_lock.json').read_text())
    assert lock['accepted_improvement'],'No accepted candidate: special inference disabled'
    backend=json.loads((ROOT/f'results/backend_{lock["selected"]}.json').read_text())['backend']
    dest=export();old=[load_model(p)[0] for p in checkpoint_paths(lock['selected'])];new=[load_model(dest/f'final_{s}.pt')[0] for s in [17,29,43]]
    tv=torch_view(dict(np.load(ROOT/'cache/valid_0.npz')))
    for a,b in zip(old,new):assert np.array_equal(predict(a,tv,backend=backend),predict(b,tv,backend=backend))
    empty={k:v[:1].clone() for k,v in tv.items()}
    empty['o'].zero_();empty['quality'][:,:,0]=0;empty['quality'][:,:,1]=1
    for name in ['text','audio','vision']:empty[name].zero_()
    protocol=json.loads((ROOT/'protocol.json').read_text())
    for m in new:
        zero_prediction=predict(m,empty,backend=backend)[0]
        assert np.allclose(zero_prediction[:3],protocol['prior'],atol=1e-7) and abs(zero_prediction[3]-protocol['mean'])<1e-7
    ids,d=load_special();base=adapt(d);encoder=TextEncoder();stats=dict(np.load(ROOT/'checkpoints/normalization.npz'))
    view=make_view(base,encoder,stats,dict(subset='',rate=0.,position='none'),0);tv=torch_view(view)
    ps=np.stack([predict(m,tv,backend=backend) for m in new]);mean=ps.mean(0);labels=['negative','neutral','positive']
    rows=[];review=[]
    for i,sid in enumerate(ids):
        rows.append(dict(sample_id=sid,polarity=labels[int(mean[i,:3].argmax())],intensity=float(mean[i,3])))
        review.append(dict(**rows[-1],**{f'prob_{label}':float(mean[i,j]) for j,label in enumerate(labels)},
            intensity_seed_std=float(ps[:,i,3].std(ddof=1)),all_modalities_missing=bool(~base['o'][i].any()),
            **{f'observed_{m}':float(view['quality'][i,j,0]) for j,m in enumerate('TAV')}))
    pd.DataFrame(rows).to_csv(ROOT/'results/附件3_问题二预测.csv',index=False)
    pd.DataFrame(review).to_csv(ROOT/'results/附件3_预测审阅.csv',index=False)
    save_json(ROOT/'deployment/manifest.json',dict(selected=lock['selected'],selection_sha256=lock_sha,backend=backend,
        checkpoints={p.name:sha256(p) for p in sorted(dest.glob('*.pt'))},encoder_sha256=sha256(ENCODER/'model.safetensors'),
        training_heads_removed=True,validation_predictions_identical=True))
    save_json(ROOT/'results/special_inference_audit.json',dict(n=len(ids),unique=len(set(ids)),selection_sha256=lock_sha,
        special_sha256={p.name:sha256(p) for p in sorted((DATA/'附件3-模态缺失特征样本/对齐版本').glob('*.pkl'))},
        deployment_roundtrip=True,encoder_unchanged=sha256(ENCODER/'model.safetensors')==json.loads((ROOT/'protocol.json').read_text())['encoder']['weights_sha256']))
    assert len(rows)==30 and len(set(ids))==30 and np.isfinite(mean).all() and np.max(np.abs(mean[:,3]))<=3
    assert sha256(ROOT/'selection_lock.json')==lock_sha
    print('Exported inference-only models and all 30 special predictions.',flush=True)

if __name__=='__main__':main()
