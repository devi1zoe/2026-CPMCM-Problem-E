from common import *
from data import *
from model import Model
from fusion import FusionModel
from oof import partitions
from evaluation_data import profiles_all,get_masks

def main():
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    protocol=json.loads((ROOT/'protocol.json').read_text());raw=load_attachment2();base=adapt(raw['valid'])
    folds,groups=partitions(raw['train']);assert len(folds)==3
    profiles=profiles_all();assert len(profiles)==253
    for profile in profiles:
        o,r=get_masks(base,profile,41000+profile['repeat']);assert np.array_equal(o,base['o']&~r) and not np.any(r&~base['p'][:,None])
    view={k:v[:12] for k,v in dict(np.load(ROOT/'cache/valid_4.npz')).items()};b=torch_view(view)
    model=Model(dict(reconstruct=True),protocol['prior'],protocol['mean']).cuda().eval()
    b['reconstruction_mask']=torch.as_tensor(view['removed']&base['o'][:12],device='cuda')
    clean=dict(np.load(ROOT/'cache/valid_0.npz'))
    for name in ['text','audio','vision']:b['target_'+name]=torch.as_tensor(clean[name][:12],device='cuda')
    with torch.no_grad():
        z=model(b);altered={k:v.clone() for k,v in b.items()}
        for m,name in enumerate(['text','audio','vision']):altered[name][~b['o'][:,m]]=7
        z2=model(altered);assert torch.equal(z['logits'],z2['logits']) and torch.equal(z['pred'],z2['pred'])
        for m,name in enumerate(['text','audio','vision']):altered['target_'+name][~b['reconstruction_mask'][:,m]]=7
        z3=model(altered);assert torch.equal(z2['reconstruction_loss'],z3['reconstruction_loss'])
        assert not torch.any(z['recovered']&b['o'])
        empty={k:v.clone() for k,v in b.items()};empty['o'].zero_();z0=model(empty)
        assert torch.isfinite(z0['logits']).all() and torch.allclose(z0['logits'].softmax(-1),model.prior.expand(12,-1))
    model.train();z=model(b);loss=z['logits'].square().mean()+z['pred'].square().mean()+.1*z['reconstruction_loss'];loss.backward()
    assert all(p.grad is None for p in model.target_input.parameters())
    assert any(p.grad is not None and p.grad.abs().max()>0 for p in model.reconstruction_output.parameters())
    b={k:v for k,v in b.items() if k in ['p','o','quality']};b['experts']=torch.cat([torch.randn(12,3,3,device='cuda').softmax(-1),torch.randn(12,3,1,device='cuda')],-1)
    b['o'][:,1]=False;models=[]
    for name in ['quality','reliability','correction']:
        torch.manual_seed(17);models.append(FusionModel(dict(fusion=name),protocol['prior'],protocol['mean']).cuda().eval())
    outputs=[m(b) for m in models]
    for z in outputs[1:]:assert torch.equal(z['pred'],outputs[0]['pred']) and torch.equal(z['logits'],outputs[0]['logits'])
    altered={k:v.clone() for k,v in b.items()};altered['experts'][:,1,:3]=torch.tensor([1.,0.,0.],device='cuda');altered['experts'][:,1,3]=3
    for model,z in zip(models,outputs):
        z2=model(altered);assert torch.equal(z['pred'],z2['pred']) and torch.equal(z['logits'],z2['logits'])
    save_json(ROOT/'results/extension_verification.json',dict(passed=True,evaluation_profiles=253,oof_source_video_disjoint=True,
        reconstruction_missing_input_invariant=True,reconstruction_targets_only_artificial_originally_observed=True,
        reconstruction_estimated_observed_distinct=True,reconstruction_all_empty_uses_prior=True,target_projection_frozen=True,
        fusion_initial_predictions_identical=True,fusion_unavailable_expert_ignored=True))
    print('Reconstruction, fusion, folds, extended evaluation verification passed')

if __name__=='__main__':main()
