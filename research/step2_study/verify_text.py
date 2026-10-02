from common import *
from data import *
from model import Model,study_state,load_study_state
from base_model import Model as BaseModel
from torch.nn import functional as F

def main():
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    protocol=json.loads((ROOT/'protocol.json').read_text());raw=load_attachment2();base=adapt(raw['valid'])
    view={k:v[:8] for k,v in dict(np.load(ROOT/'cache/valid_1.npz')).items()}
    view.update(tokens=base['tokens'][:8],attention=base['attention'][:8]);b=torch_view(view)
    torch.manual_seed(17);reference=BaseModel({},protocol['prior'],protocol['mean']).cuda().eval()
    torch.manual_seed(17);model=Model(dict(lora=True),protocol['prior'],protocol['mean']).cuda().eval()
    for k,v in reference.state_dict().items():assert torch.equal(v,model.state_dict()[k])
    with torch.no_grad():
        r=reference(b);z=model(b)
        maxdiff=max(float((r[k]-z[k]).abs().max()) for k in ['logits','pred'])
        assert maxdiff<.003,maxdiff
        changed={k:v.clone() for k,v in b.items()};missing=b['p']&~b['o'][:,0]
        changed['tokens'][missing]=999;z2=model(changed)
        assert torch.equal(z['logits'],z2['logits']) and torch.equal(z['pred'],z2['pred'])
    model.train();z=model(b);loss=z['logits'].square().mean()+z['pred'].square().mean();loss.backward()
    adapters=[(k,p) for k,p in model.named_parameters() if '.lora_' in k]
    assert sum(p.numel() for k,p in adapters)==24576
    assert any(p.grad is not None and p.grad.abs().max()>0 for k,p in adapters)
    for k,p in model.named_parameters():
        if k.startswith('language.backbone.') and '.lora_' not in k:assert not p.requires_grad and p.grad is None
        if p.grad is not None:assert torch.isfinite(p.grad).all()
    state=study_state(model);load_study_state(model,state)
    assert not any(k.startswith('language.backbone.') and '.lora_' not in k for k in state)
    save_json(ROOT/'results/text_adaptation_verification.json',dict(passed=True,initial_prediction_max_difference=maxdiff,
        frozen_backbone_unchanged=True,adapter_parameters=24576,deleted_tokens_no_effect=True,head_initialization_identical=True,
        checkpoint_excludes_frozen_backbone=True,normalization='fixed train-only reference statistics',micro_batch=16,effective_batch=64,
        note='Microbatching changes dropout draws relative to frozen reference; three seeds are reported, no bitwise training equivalence claimed.'))
    print('Text adaptation verification passed; initial difference',maxdiff)

if __name__=='__main__':main()
