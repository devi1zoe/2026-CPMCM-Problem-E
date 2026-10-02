"""Input invariants and behavior tests must pass before diagnostic or fitting."""
from common import *
from data import *
from dynamic import epoch_masks,quality
from model import Model
from base_model import Model as BaseModel
from support import load_experts,state_hash
from engine import predict,load_model
import copy,tempfile


def main():
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    protocol=json.loads((ROOT/'protocol.json').read_text());raw=load_attachment2();checks={}
    assert sha256(DATA/'附件2-数据集特征文件/aligned_50.pkl')==protocol['source_sha256']
    assert sha256(ROOT/'checkpoints/normalization.npz')==protocol['normalization_sha256']
    for split,n in [('train',3395),('valid',728),('test',727)]:
        d=raw[split];assert len(d['id'])==n and len(set(d['id']))==n
        if split!='test':
            labels=np.load(ROOT/f'cache/{split}_labels.npz')
            assert np.array_equal(labels['c'],d['classification_labels']) and np.array_equal(labels['y'],d['regression_labels'])
            assert np.array_equal(np.sign(labels['y'])+1,labels['c'])
    for a,b in [('train','valid'),('train','test'),('valid','test')]:
        assert not {i.split('$_$')[0] for i in raw[a]['id']}&{i.split('$_$')[0] for i in raw[b]['id']}
    checks['source_labels_and_video_splits_verified']=True
    base=adapt(raw['train']);available,removed=epoch_masks(base,raw['train']['id'],17,1)
    assert np.array_equal(removed,epoch_masks(base,raw['train']['id'],17,1)[1]);assert not (removed&~base['p'][:,None]).any()
    checks['dynamic_mask_reproducible_and_padding_excluded']=True
    enc=TextEncoder();before=sha256(ENCODER/'model.safetensors')
    b={k:v[:4].copy() for k,v in base.items() if k!='uncertain'};o=b['o'].copy();o[:,0,2:5]=False
    changed=copy.deepcopy(b);changed['tokens'][:,2:5]=12345
    assert np.array_equal(enc.encode(b,o),enc.encode(changed,o))
    assert not enc.model.training and all(not p.requires_grad for p in enc.model.parameters())
    checks['deleted_tokens_invariant_and_encoder_frozen']=True
    view={k:v[:8].copy() for k,v in dict(np.load(ROOT/'cache/train_0.npz')).items() if k!='removed'}
    view['o'][0]=False
    for n in ['text','audio','vision']:view[n][0]=0
    view['o'][1,1:]=False;view['audio'][1]=0;view['vision'][1]=0
    view['p'][2]=False;view['p'][2,1]=True;view['o'][2,:,0]=False;view['o'][2,:,2:]=False
    for n in ['text','audio','vision']:view[n][2,0]=0;view[n][2,2:]=0
    view['quality']=quality(view['p'],view['o']);tv=torch_view(view)
    torch.manual_seed(17);base_model=BaseModel({},protocol['prior'],protocol['mean']).cuda().eval()
    torch.manual_seed(17);dual=Model(protocol['configs']['dual_gate'],protocol['prior'],protocol['mean']).cuda().eval()
    assert all(torch.equal(v,dual.state_dict()[k]) for k,v in base_model.state_dict().items())
    assert np.array_equal(predict(base_model,tv),predict(dual,tv))
    checks['dual_gate_initialization_matches_baseline_exactly']=True
    late=Model(protocol['configs']['late_fusion'],protocol['prior'],protocol['mean']).cuda();load_experts(late,17)
    frozen_before=state_hash(late.experts);late.train();assert all(not e.training for e in late.experts)
    for name,m in [('dual_gate',dual),('late_fusion',late)]:
        r=m(tv);p=predict(m,tv);assert np.isfinite(p).all() and np.max(np.abs(p[:,3]))<=3
        assert np.allclose(p[0,:3],protocol['prior'],atol=1e-7) and abs(p[0,3]-protocol['mean'])<1e-7
        for k in ['weights_classification','weights_regression']:
            assert (r[k][~r['present']]==0).all()
            assert torch.allclose(r[k].sum(-1),r['present'].any(-1).float())
        with tempfile.TemporaryDirectory(dir=ROOT/'cache') as tmp:
            path=Path(tmp)/'m.pt';torch.save(dict(config=protocol['configs'][name],prior=protocol['prior'],mean=protocol['mean'],state_dict=m.cpu().state_dict()),path)
            restored,_=load_model(path);assert np.array_equal(p,predict(restored,tv))
        m.cuda()
    with torch.no_grad():
        e=late.expert_outputs(tv);direct=late(tv);cached=late({**tv,'expert_outputs':e})
        assert torch.equal(direct['logits'],cached['logits']) and torch.equal(direct['pred'],cached['pred'])
        assert torch.allclose(direct['logits'][1].softmax(-1),e[1,0,:3],atol=1e-7) and direct['pred'][1]==e[1,0,3]
    labels=np.load(ROOT/'cache/train_labels.npz');c=torch.tensor(labels['c'][:8],device='cuda');y=torch.tensor(labels['y'][:8],device='cuda')
    from model import task_loss
    for m in [dual,late]:
        m.train();loss,_=task_loss(m(tv),c,y,torch.ones(3,device='cuda'));loss.backward()
        assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
    assert all(p.grad is None and not p.requires_grad for p in late.experts.parameters())
    assert state_hash(late.experts)==frozen_before
    assert before==sha256(ENCODER/'model.safetensors')==protocol['encoder']['weights_sha256']
    checks.update(absent_weights_zero=True,one_slot_and_all_missing_finite=True,checkpoint_roundtrip=True,
        single_expert_equals_expert=True,expert_cache_matches_direct=True,frozen_experts_no_grad_or_state_change=True,
        finite_backward=True,encoder_hash_unchanged=True)
    save_json(ROOT/'results/verification.json',checks);print(json.dumps(checks,indent=2),flush=True)


if __name__=='__main__':main()
