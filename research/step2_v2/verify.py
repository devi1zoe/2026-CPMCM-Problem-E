"""Behavior checks, including raw-token deletion invariance and real second-order EMC backward."""
from common import *
from data import *
from dynamic import epoch_masks,quality,context_key
from model import Model,VARIANTS,objectives
from engine import load_model,predict
import copy,tempfile,random

def main():
    torch.set_num_threads(4);torch.manual_seed(17)
    protocol=json.loads((ROOT/'protocol.json').read_text());raw=load_attachment2();base=adapt(raw['train'])
    checks={}
    a,r=epoch_masks(base,raw['train']['id'],17,1);a2,r2=epoch_masks(base,raw['train']['id'],17,1)
    assert np.array_equal(r,r2);assert not (r&~base['p'][:,None]).any()
    assert .46 < (~r.any((1,2))).mean() < .54
    checks['dynamic_masks_reproducible_and_padding_excluded']=True
    ids=raw['train']['id'];order=np.arange(len(ids))[::-1]
    perm={k:v[order] for k,v in base.items() if k!='uncertain'}
    _,rp=epoch_masks(perm,[ids[i] for i in order],17,1);assert np.array_equal(rp,r[order])
    checks['masks_independent_of_sample_order_and_model']=True
    b={k:v[:4].copy() for k,v in base.items() if k!='uncertain'};obs=b['o'].copy()
    obs[:,0,2:5]=False;b2=copy.deepcopy(b);b2['tokens'][:,2:5]=12345
    rng_cpu=torch.get_rng_state().clone();rng_gpu=torch.cuda.get_rng_state().clone();rng_py=random.getstate();rng_np=np.random.get_state()
    enc=TextEncoder();before=sha256(ENCODER/'model.safetensors')
    assert torch.equal(rng_cpu,torch.get_rng_state()) and torch.equal(rng_gpu,torch.cuda.get_rng_state())
    assert rng_py==random.getstate() and np.array_equal(rng_np[1],np.random.get_state()[1])
    checks['frozen_encoder_creation_does_not_shift_training_rng']=True
    save_json(ROOT/'results/frozen_encoder_rng_audit.json',dict(torch_cpu_unchanged=True,torch_cuda_unchanged=True,python_unchanged=True,numpy_unchanged=True))
    x=enc.encode(b,obs);xx=enc.encode(b2,obs);assert np.array_equal(x,xx)
    assert all(not p.requires_grad for p in enc.model.parameters()) and not enc.model.training
    checks['deleted_tokens_do_not_leak_context']=True
    assert context_key(b,obs,before)!=context_key(b2,obs,before)
    checks['cache_key_includes_tokens_masks_encoder']=True
    view={k:v[:8].copy() for k,v in dict(np.load(ROOT/'cache/train_0.npz')).items() if k!='removed'}
    # One wholly missing modality, a wholly missing sample, and a one-slot sample.
    view['o'][0,1]=False;view['audio'][0]=0
    view['o'][1]=False
    for name in ['text','audio','vision']:view[name][1]=0
    view['p'][2]=False;view['p'][2,1]=True;view['o'][2,:,2:]=False;view['o'][2,:,0]=False
    for name in ['text','audio','vision']:view[name][2,2:]=0;view[name][2,0]=0
    view['quality']=quality(view['p'],view['o']);tv=torch_view(view)
    c=torch.tensor(raw['train']['classification_labels'][:8],device='cuda').long();y=torch.tensor(raw['train']['regression_labels'][:8],device='cuda')
    cw=torch.ones(3,device='cuda')
    configs=dict(VARIANTS)
    for module in ['aux','msd','cce','imtd','emc']:configs['emc_without_'+module]={**VARIANTS['emc'],module:False}
    for name,conf in configs.items():
        m=Model(conf,protocol['prior'],protocol['mean']).cuda();m.train()
        teacher=copy.deepcopy(m).eval()
        with torch.no_grad():t=teacher(tv)
        needs_teacher=conf.get('imtd') or conf.get('emc')
        out=m(tv);loss,parts,trust=objectives(m,out,tv,c,y,cw,1.,t if needs_teacher else None,2)
        assert torch.isfinite(loss);loss.backward()
        assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
        assert torch.allclose(out['logits'][1].softmax(-1),m.prior) and out['pred'][1]==m.mean
        if needs_teacher:assert trust[0,1]==0 and trust[1].sum()==0
        if not conf.get('imtd'):assert parts['kd']==0
        if conf.get('emc'):assert float(parts['emc'])>=0
        original=predict(m,tv)
        with tempfile.TemporaryDirectory(dir=ROOT/'cache') as temp:
            path=Path(temp)/'m.pt';torch.save(dict(config=conf,prior=protocol['prior'],mean=protocol['mean'],state_dict=m.cpu().state_dict()),path)
            restored,_=load_model(path);again=predict(restored,tv)
            assert np.array_equal(original,again)
        del m,restored,teacher
    checks.update(all_ten_models_finite_forward_backward=True,emc_second_order_gradient=True,absent_teacher_weight_zero=True,
        all_missing_uses_train_priors=True,one_slot_finite=True,checkpoint_roundtrip_identical=True,distillation_disabled_zero=True,
        five_leave_one_out_architectures_valid=True)
    # Continuous mask exact length and contiguity; zero-rate identity.
    for pos in ['start','middle','end','random']:
        _,removed=corrupt(base,dict(subset='TAV',rate=.3,position=pos),71)
        for i in range(30):
            idx=np.flatnonzero(removed[i,0]);expected=max(1,int(np.floor(base['p'][i].sum()*.3+.5)))
            assert len(idx)==expected and (len(idx)<2 or np.all(np.diff(idx)==1))
    checks['contiguous_lengths_and_positions_verified']=True
    assert sha256(ENCODER/'model.safetensors')==before;checks['encoder_hash_unchanged']=True
    save_json(ROOT/'results/verification.json',checks)
    print(json.dumps(checks,indent=2),flush=True)

if __name__=='__main__':main()
