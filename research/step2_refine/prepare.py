"""Reconstruct local caches and read-only baseline references without modifying v2."""
from common import *
from data import *
import shutil,platform,importlib.metadata


def main():
    for d in ['cache','checkpoints','results/figures','logs']:(ROOT/d).mkdir(parents=True,exist_ok=True)
    protocol=json.loads((ROOT/'protocol.json').read_text());source=PROJECT/'step2_v2'
    assert sha256(DATA/'附件2-数据集特征文件/aligned_50.pkl')==protocol['source_sha256']
    assert sha256(ENCODER/'model.safetensors')==protocol['encoder']['weights_sha256']
    norm=ROOT/'checkpoints/normalization.npz'
    if not norm.exists():shutil.copy2(source/'checkpoints/normalization.npz',norm)
    assert sha256(norm)==protocol['normalization_sha256']
    for p in (source/'cache').iterdir():
        q=ROOT/'cache'/p.name
        if p.is_file() and p.suffix in ['.npz','.json'] and not p.name.startswith(('test_','eval_test_')) and not q.exists():q.symlink_to(p.resolve())
    for name in ['old_gated','gated','text','audio','vision']:
        for prefix in ['valid_','six_','backend_']:
            for p in (source/'results').glob(prefix+name+'*'):
                q=ROOT/'results'/p.name
                if p.is_file() and not q.exists():shutil.copy2(p,q)
    profiles=ROOT/'results/valid_profiles.json'
    if not profiles.exists():shutil.copy2(source/'results/valid_profiles.json',profiles)
    raw=load_attachment2();encoder=None;stats=dict(np.load(norm))
    for split in ['train','valid']:
        labels=ROOT/f'cache/{split}_labels.npz'
        if not labels.exists():np.savez(labels,c=raw[split]['classification_labels'].astype(np.int64),y=raw[split]['regression_labels'].astype(np.float32))
        for j in ([0] if split=='train' else range(6)):
            p=ROOT/f'cache/{split}_{j}.npz'
            if not p.exists():
                if encoder is None:encoder=TextEncoder()
                np.savez(p,**make_view(adapt(raw[split]),encoder,stats,VALID_PROFILES[j],protocol['validation_corruption_seed']))
    save_json(ROOT/'environment.json',dict(python=platform.python_version(),executable=sys.executable,torch=torch.__version__,cuda=torch.version.cuda,
        packages={k:importlib.metadata.version(k) for k in ['torch','transformers','numpy','pandas','scikit-learn','matplotlib']},
        setup='Existing root Python plus step1/.venv dependencies; no installation or environment creation',
        gpu_use='Physical GPUs 2 and 3 were free at training start; after user interruption GPU3 became externally occupied; remaining validation only GPU2'))
    print('Local references ready; source/encoder/normalization hashes verified.')


if __name__=='__main__':main()
