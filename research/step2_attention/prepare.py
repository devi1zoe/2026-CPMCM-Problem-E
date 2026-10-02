from common import *
from data import *
import shutil,importlib.metadata,platform
from transformers import AutoTokenizer


def main():
    for name in ['cache','results','logs','checkpoints']:(ROOT/name).mkdir(exist_ok=True)
    raw=load_attachment2();tok=AutoTokenizer.from_pretrained(ENCODER,local_files_only=True);audit={}
    for s,d in raw.items():
        b=adapt(d);n=len(d['id'])
        assert len(set(d['id']))==n
        assert np.array_equal(np.sign(d['regression_labels'])+1,d['classification_labels'])
        encoded=tok(list(d['raw_text']),max_length=50,padding='max_length',truncation=True,return_tensors='np')['input_ids']
        assert np.array_equal(encoded,b['tokens'])
        assert all(np.isfinite(d[k]).all() for k in ['text','audio','vision'])
        audit[s]=dict(n=n,classes=np.bincount(d['classification_labels'].astype(int),minlength=3).tolist(),token_matches=n,uncertain_span=len(b['uncertain']))
    for a,b in [('train','valid'),('train','test'),('valid','test')]:
        assert not ({x.split('$_$')[0] for x in raw[a]['id']}&{x.split('$_$')[0] for x in raw[b]['id']})
    counts=np.bincount(raw['train']['classification_labels'].astype(int));prior=counts/counts.sum()
    original=dict(source_sha256='66e867aa74bc70a844e806e5571e371c9abb4a35f9e2887ce9b4d97ff2cb8fcd',frozen_encoder=dict(
        repo='distilbert/distilbert-base-uncased',revision='12040accade4e8a0f71eabdb258fecc2e7e948be',
        weights_sha256='5e3f1108e3cb34ee048634875d8482665b65ac713291a7e32396fb18f6ff0063'))
    assert sha256(DATA/'附件2-数据集特征文件/aligned_50.pkl')==original['source_sha256']
    assert sha256(ENCODER/'model.safetensors')==original['frozen_encoder']['weights_sha256']
    for seed in [17,29,43]:
        if not (ROOT/f'checkpoints/old_gated_{seed}.pt').exists():shutil.copy2(PROJECT/f'step2/checkpoints/gated_aug_{seed}.pt',ROOT/f'checkpoints/old_gated_{seed}.pt')
    needed=['train_0.npz','train_labels.npz','valid_labels.npz']+[f'valid_{j}.npz' for j in range(6)]
    if not all((ROOT/'cache'/f).is_file() for f in needed):
        encoder=TextEncoder();bt=adapt(raw['train']);clean=encoder.encode(bt,bt['o'])
        normpath=ROOT/'checkpoints/normalization.npz'
        stats=dict(np.load(normpath)) if normpath.exists() else fit_stats(bt,clean)
        if not normpath.exists():np.savez(normpath,**stats)
        for split in ['train','valid']:
            base=bt if split=='train' else adapt(raw[split]);text=clean if split=='train' else encoder.encode(base,base['o'])
            profiles=[VALID_PROFILES[0]] if split=='train' else VALID_PROFILES
            for j,profile in enumerate(profiles):
                path=ROOT/f'cache/{split}_{j}.npz'
                if not path.exists():
                    if path.is_symlink():path.unlink()
                    np.savez(path,**make_view(base,encoder,stats,profile,20260924,text))
            path=ROOT/f'cache/{split}_labels.npz'
            if not path.exists():
                if path.is_symlink():path.unlink()
                np.savez(path,c=raw[split]['classification_labels'].astype(np.int64),y=raw[split]['regression_labels'].astype(np.float32))
    source_files=[DATA/'附件2-数据集特征文件/aligned_50.pkl',DATA/'附件2-数据集特征文件/label.xlsx']
    proto=dict(version='attention-1',seeds=[17,29,43],models=['pool','self','cross','cross_clean'],attention_candidates=['self','cross'],
        hidden=64,heads=4,dropout=.2,batch=64,lr=.0003,weight_decay=.001,epochs=30,patience=6,reg_weight=2.,
        clean_probability=.5,validation_profiles=VALID_PROFILES,validation_seed=20260924,evaluation_seed=41000,
        selection='min six-profile J among augmented attention models; clean weight .5 and five corrupt profiles .1 each',
        historical_guardrails='reported separately; not an instruction to replace the previously adopted model',
        prior=prior.tolist(),mean=float(raw['train']['regression_labels'].mean()),
        encoder=original['frozen_encoder'],source_sha256=original['source_sha256'],normalization_sha256=sha256(ROOT/'checkpoints/normalization.npz'),
        input_version='aligned_50',training_mask='SHA256(seed/epoch/sample_id), 50% clean; token deletion before encoder',
        test_previously_used=True,test_use='one locked post-selection comparison; not a new independent test set',
        references=['https://aclanthology.org/P19-1656/','https://huggingface.co/distilbert/distilbert-base-uncased'])
    save_json(ROOT/'protocol.json',proto);save_json(ROOT/'results/data_audit.json',audit)
    save_json(ROOT/'environment.json',dict(python=platform.python_version(),packages={k:importlib.metadata.version(k) for k in ['torch','transformers','numpy','pandas','scikit-learn','matplotlib']},root_modified=False,existing_venv=str(VENV)))
    # Copy read-only material fingerprints; later cache loads must match these.
    save_json(ROOT/'cache_manifest.json',{p.name:sha256(p) for p in (ROOT/'cache').glob('*.npz')})
    for file in ['train_labels.npz','valid_labels.npz']:
        d=np.load(ROOT/'cache'/file);s=file.split('_')[0]
        assert np.array_equal(d['c'],raw[s]['classification_labels'])
    print(json.dumps(audit,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
