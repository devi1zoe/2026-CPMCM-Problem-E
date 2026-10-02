"""验证会影响结论的关键性质：遮挡前重编码、不泄露、边界掩码及空输入。"""
from common import *
from data import *
from model import SentimentModel


def main():
    torch.set_num_threads(4)
    raw=load_attachment2()['train']
    d={k:v[:8] for k,v in raw.items()}
    base=adapt(d)
    profile=dict(subset='TAV',rate=.5,position='middle')
    o,removed=corrupt(base,profile,123)
    assert not np.any(removed & ~base['p'][:,None,:])
    assert np.array_equal(o,base['o'] & ~removed)
    again=corrupt(base,profile,123)
    assert np.array_equal(o,again[0])
    for i in range(8):
        expected=max(1,int(np.floor(base['p'][i].sum()*.5+.5)))
        assert np.all(removed[i].sum(-1)==expected)
    encoder=TextEncoder()
    first=encoder.encode(base,o)
    altered=dict(base,tokens=base['tokens'].copy())
    altered['tokens'][removed[:,0]]=2987
    second=encoder.encode(altered,o)
    assert np.array_equal(first,second), 'Deleted token leaked through the encoder'
    assert np.all(first[~o[:,0]]==0)
    stats=dict(np.load(ROOT/'checkpoints/normalization.npz'))
    view=make_view(base,encoder,stats,profile,123)
    for k in ['text','audio','vision']:
        assert np.isfinite(view[k]).all()
    tv=torch_view(view)
    for variant in ['text_only','audio_only','vision_only','pooled_clean','fixed_clean','fixed_aug','mask_aug','gated_aug']:
        model=SentimentModel(variant).cuda().eval()
        with torch.no_grad():
            a,y,g=model(tv)
            assert torch.isfinite(a).all() and torch.isfinite(y).all() and (y.abs()<=3).all()
            # 人工改变 padding 数值不能影响预测。
            padded={k:v.clone() for k,v in tv.items()}
            for k in ['text','audio','vision']:
                padded[k][~tv['p']]=7
            b,z,_=model(padded)
            assert torch.allclose(a,b,atol=1e-6) and torch.allclose(y,z,atol=1e-6)
            empty={k:torch.zeros_like(v) for k,v in tv.items()}
            a,y,g=model(empty)
            assert torch.isfinite(a).all() and torch.isfinite(y).all() and torch.isfinite(g).all()
    save_json(ROOT/'results/verification.json',dict(deleted_token_invariance=True,padding_invariance=True,
        deterministic_corruption=True,exact_deleted_slot_count=True,finite_empty_input_all_eight_models=True,
        text_masked_before_context_encoding=True))
    print('All behavioral checks passed.',flush=True)


if __name__=='__main__':
    main()
