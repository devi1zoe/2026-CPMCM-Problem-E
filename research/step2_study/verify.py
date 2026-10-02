from common import *
from data import *
from dynamic import epoch_masks,quality
import importlib.util

def main():
    raw=load_attachment2();base=adapt(raw['train']);ids=raw['train']['id']
    spec=importlib.util.spec_from_file_location('historical_dynamic',PROJECT/'step2_v2/dynamic.py')
    old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    rows=[]
    for seed in [17,29,43]:
        a,ra=epoch_masks(base,ids,seed,1,'single');b,rb=epoch_masks(base,ids,seed,1,'multi');c,rc=epoch_masks(base,ids,seed,1,'unequal')
        oa,ora=old.epoch_masks(base,ids,seed,1)
        assert np.array_equal(a,oa) and np.array_equal(ra,ora)
        assert np.array_equal(ra.sum(-1),rb.sum(-1))
        assert np.array_equal(ra.any(-1),rc.any(-1))
        assert np.array_equal(ra.any((1,2)),rb.any((1,2)))
        for o,r in [(a,ra),(b,rb),(c,rc)]:
            assert not np.any(r & ~base['p'][:,None])
            assert np.array_equal(o,base['o']&~r)
            assert np.isfinite(quality(base['p'],o)).all()
        rows.append(dict(seed=seed,clean=int((~ra.any((1,2))).sum()),multi_changed=int((ra!=rb).any((1,2)).sum()),unequal_changed=int((rb!=rc).any((1,2)).sum())))
    groups={k:{str(i).split('$_$')[0] for i in raw[k]['id']} for k in ['train','valid','test']}
    assert not groups['train']&groups['valid'] and not groups['train']&groups['test'] and not groups['valid']&groups['test']
    protocol=json.loads((ROOT/'protocol.json').read_text())
    assert sha256(ROOT/'checkpoints/normalization.npz')==protocol['normalization_sha256']
    torch.set_num_threads(2)
    with torch.random.fork_rng():encoder=TextEncoder()
    sub={k:base[k][:8].copy() for k in ['tokens','attention','p']};o,_=epoch_masks(base,ids,17,1)
    o=o[:8].copy();o[:,0,10:20]=False
    h=encoder.encode(sub,o)
    hidden=sub['p']&~o[:,0];modified={k:v.copy() for k,v in sub.items()};modified['tokens'][hidden]=999
    h2=encoder.encode(modified,o)
    assert np.array_equal(h,h2) and np.all(h[~o[:,0]]==0)
    save_json(ROOT/'results/pretraining_verification.json',dict(passed=True,mask_rows=rows,baseline_masks_identical=True,
        single_multi_equal_counts=True,deleted_tokens_cannot_influence_remaining_context=True,source_video_splits_disjoint=True,
        protocol_sha256=sha256(ROOT/'protocol.json')))
    print('Pretraining verification passed',rows)

if __name__=='__main__':main()
