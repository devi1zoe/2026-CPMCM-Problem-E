"""Audit and register immutable protocol. No test evaluation or special inference."""
from common import *
from data import *
from model import VARIANTS
import shutil,importlib.metadata,platform

def main():
    for name in ['cache','checkpoints','results/figures','logs']:(ROOT/name).mkdir(parents=True,exist_ok=True)
    raw=load_attachment2();old=json.loads((PROJECT/'step2/protocol.json').read_text())
    source=sha256(DATA/'附件2-数据集特征文件/aligned_50.pkl')
    assert source==old['source_sha256']
    encoder_hash=sha256(ENCODER/'model.safetensors');assert encoder_hash==old['frozen_encoder']['weights_sha256']
    audit={}
    for split,d in raw.items():
        b=adapt(d);c=d['classification_labels'].astype(int);y=d['regression_labels']
        assert np.array_equal(np.sign(y).astype(int)+1,c)
        assert len(set(d['id']))==len(c)
        audit[split]=dict(n=len(c),classes=np.bincount(c,minlength=3).tolist(),uncertain_span=len(b['uncertain']))
    for s,t in [('train','valid'),('train','test'),('valid','test')]:
        assert not set(i.split('$_$')[0] for i in raw[s]['id'])&set(i.split('$_$')[0] for i in raw[t]['id'])
    prior=(np.bincount(raw['train']['classification_labels'].astype(int),minlength=3)/len(raw['train']['id'])).tolist()
    protocol=dict(version=2,configs=VARIANTS,seeds=[17,29,43],prior=prior,mean=float(raw['train']['regression_labels'].mean()),
        hidden=64,dropout=.2,batch=64,lr=.001,weight_decay=.001,stage_epochs=[20,15],patience=5,clip_grad=5,
        clean_probability=.5,reg_weight_candidates=[.5,1.,2.],validation_profiles=VALID_PROFILES,validation_corruption_seed=20260924,
        source_sha256=source,encoder=old['frozen_encoder'],normalization_sha256=sha256(ROOT/'checkpoints/normalization.npz'),
        loss_weights=dict(aux=.1,alignment=.05,orthogonal=.05,preserve=.05,kd=.1,emc=.01,emc_gradient=.01),
        training_mask='SHA256(seed,global_epoch,sample_id); independent of model and label; clean probability .5',
        evaluation_seed=41000,evaluation_repeats=3,selection='six-profile J; complete validation guardrails; ties within .002 prefer fewer parameters',
        guardrails=dict(clean_f1_drop=.005,clean_mae_rise=.01,robust_f1_gain=.01,robust_mae_rise=0.,neutral_recall_drop=.01,strong_mae_rise=.02),
        previously_evaluated_test=True,bootstrap_repeats=1000,bootstrap_seed=20260926,
        package_limit_bytes=50_000_000,text_frozen=True,regression_distillation=False)
    p=ROOT/'protocol.json'
    if p.exists():assert json.loads(p.read_text())==protocol,'Existing protocol differs: use a new run directory'
    else:save_json(p,protocol)
    save_json(ROOT/'results/data_audit.json',dict(splits=audit,source_sha256=source,source_video_disjoint=True,special_n=30))
    save_json(ROOT/'environment.json',dict(python=platform.python_version(),executable=sys.executable,
        packages={k:importlib.metadata.version(k) for k in ['torch','transformers','numpy','pandas','scikit-learn','matplotlib']},
        setup='/home/ubuntu/.bashrc_hyf read; reuse root Python and step1/.venv; no installation',gpus='Use idle physical GPU 2/3; leave 0/1 untouched'))
    protected={str(p.relative_to(PROJECT)):sha256(p) for sub in ['step2','step3'] for p in (PROJECT/sub).rglob('*') if p.is_file() and p.suffix in ['.py','.md','.pt','.json','.csv','.zip']}
    if not (ROOT/'protected_files.json').exists():save_json(ROOT/'protected_files.json',protected)
    # Reuse the exact six previously registered validation inputs and train-only statistics.
    encoder=None;stats=dict(np.load(ROOT/'checkpoints/normalization.npz'))
    for split in ['train','valid']:
        for j in ([0] if split=='train' else range(6)):
            src=PROJECT/f'step2/cache/{split}_{j}.npz';dst=ROOT/f'cache/{split}_{j}.npz'
            if not dst.exists():
                if src.exists():os.link(src,dst)
                else:
                    if encoder is None:encoder=TextEncoder()
                    view=make_view(adapt(raw[split]),encoder,stats,VALID_PROFILES[j],protocol['validation_corruption_seed'])
                    np.savez(dst,**view)
        src=PROJECT/f'step2/cache/{split}_labels.npz';dst=ROOT/f'cache/{split}_labels.npz'
        if not dst.exists():
            if src.exists():shutil.copy2(src,dst)
            else:np.savez(dst,c=raw[split]['classification_labels'].astype(np.int64),y=raw[split]['regression_labels'].astype(np.float32))
    print('Protocol audited and registered.',flush=True)

if __name__=='__main__':main()
