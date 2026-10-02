from common import *
from data import *
from dynamic import EpochStore,epoch_masks
from model import Model


def main():
    torch.set_num_threads(1);torch.manual_seed(117)
    p=json.loads((ROOT/'protocol.json').read_text())
    view=dict(np.load(ROOT/'cache/train_0.npz'));v=torch_view({k:a[:5] for k,a in view.items()});v['o'][:,1,5:15]=False
    proofs={}
    for name in p['models']:
        m=Model(name,p['prior'],p['mean']).cuda().eval()
        with torch.no_grad():
            r=m(v,True);changed={k:t.clone() for k,t in v.items()}
            for j,k in enumerate(['text','audio','vision']):changed[k][~(v['o'][:,j]&v['p'])]=17
            other=m(changed)
            assert torch.equal(r['logits'],other['logits']) and torch.equal(r['pred'],other['pred'])
            empty={k:torch.zeros_like(t) for k,t in v.items()};e=m(empty,True)
            assert torch.isfinite(e['logits']).all() and torch.isfinite(e['pred']).all()
            assert torch.allclose(e['logits'].softmax(-1),m.prior.expand(5,-1)) and torch.allclose(e['pred'],m.mean.expand(5))
            if r['cross_attention'] is not None:
                att=r['cross_attention'];assert torch.all(att[:,:,~m.cross_allowed]==0)
                flat=(v['o']&v['p'][:,None]).flatten(1)
                assert torch.all(att.masked_select(~flat[:,None,None,:].expand_as(att))==0)
        grad={k:t.clone() for k,t in v.items()}
        for k in ['text','audio','vision']:grad[k]=grad[k].float().requires_grad_()
        rr=m(grad);(rr['logits'].square().mean()+rr['pred'].square().mean()).backward()
        for j,k in enumerate(['text','audio','vision']):
            assert torch.isfinite(grad[k].grad).all()
            assert torch.all(grad[k].grad[~(v['o'][:,j]&v['p'])]==0)
        assert all(torch.isfinite(t.grad).all() for t in m.parameters() if t.grad is not None)
        if name in ['cross','cross_clean']:assert m.cross_block.qkv.weight.grad.abs().sum()>0
        proofs[name]='padding/deletion invariance; finite forward/backward; train prior fallback; valid attention masks'
    raw=load_attachment2()['train'];base=adapt({k:v[:5] for k,v in raw.items()})
    o,d=corrupt(base,dict(subset='TAV',rate=.5,position='middle'),17)
    enc=TextEncoder();first=enc.encode(base,o);alter=dict(base,tokens=base['tokens'].copy());alter['tokens'][d[:,0]]=2021
    assert np.array_equal(first,enc.encode(alter,o))
    oa,da=epoch_masks(base,raw['id'][:5],17,1);ob,db=epoch_masks(base,raw['id'][:5],17,1)
    assert np.array_equal(da,db) and not (da&~base['p'][:,None]).any()
    store=EpochStore(raw);ev=store.get(17,1);changed=np.flatnonzero(np.any(ev['o'][:,0]!=store.base['o'][:,0],axis=1))[:5]
    b={k:store.base[k][changed] for k in ['tokens','attention','p']};text=enc.encode(b,ev['o'][changed])
    z=np.clip((text-store.stats['text_mean'])/store.stats['text_std'],-8,8);z[~ev['o'][changed,0]]=0
    delta=float(np.abs(z-ev['text'][changed]).max());assert delta<.025,delta
    save_json(ROOT/'results/verification.json',dict(passed=True,models=proofs,raw_token_deletion_before_encoding=True,deterministic_epoch_masks=True,cached_text_reencode_max_error=delta,protocol_sha256=sha256(ROOT/'protocol.json')))
    print('Behavior verification passed; cache reencode max error',delta,flush=True)


if __name__=='__main__':main()
