"""Requirement-level completion audit, including the deliberate no-test fallback."""
from common import *
from engine import load_model
from support import state_hash,expert_paths
from selection import summary,guardrails
import numpy as np,pandas as pd,re


def main():
    protocol=json.loads((ROOT/'protocol.json').read_text());lock=json.loads((ROOT/'selection_lock.json').read_text());checks={}
    assert lock['protocol_sha256']==sha256(ROOT/'protocol.json')
    assert all(json.loads((ROOT/'results/verification.json').read_text()).values())
    diag=json.loads((ROOT/'results/diagnostics_complete.json').read_text());assert diag['safe_to_compare'] and not diag['test_used'] and diag['protocol_sha256']==sha256(ROOT/'protocol.json')
    general=pd.read_csv(ROOT/'results/generalization_groups.csv');g=general[general.group=='all']
    assert len(g)==24 and set(g.model)=={'text','audio','vision','gated'} and set(g.split)=={'train','valid'}
    assert all(r.n==(3395 if r.split=='train' else 728) for r in g.itertuples())
    probe=json.loads((ROOT/'results/memory_probe_samples.json').read_text());assert len(set(probe['indices']))==96 and probe['classes']==[32,32,32]
    probes=json.loads((ROOT/'results/memory_probe_summary.json').read_text());assert len(probes)==2 and all(r['steps']<=300 for r in probes)
    grad=pd.read_csv(ROOT/'results/task_gradients.csv');assert len(grad)==96
    assert set(grad.condition)=={'clean','dynamic'} and set(grad.block)=={'branches','gate'} and set(grad.seed)=={17,29,43}
    assert grad.loc[~grad.zero_gradient,'cosine'].between(-1.000001,1.000001).all()
    comp=pd.read_csv(ROOT/'results/complementarity.csv');assert len(comp)==676 and (comp.deployable==False).all()
    checks.update(input_behavior_checks=True,full_train_valid_12_checkpoints=True,two_fixed_memory_probes=True,all_96_gradient_records=True,complementarity_oracles_separate=True)
    runs=[]
    for name in ['dual_gate','late_fusion']:
        for seed in [17,29,43]:
            path=ROOT/f'checkpoints/{name}_{seed}.pt';record=json.loads((ROOT/f'results/completed_{name}_{seed}.json').read_text())
            assert record['sha256']==sha256(path) and record['protocol_sha256']==sha256(ROOT/'protocol.json') and record['config']==protocol['configs'][name]
            history=json.loads((ROOT/f'results/history_{name}_{seed}.json').read_text());assert len(history)==record['epochs'] and {r['stage'] for r in history}=={1,2}
            assert all(np.isfinite(r['score']) and 'fixed_train' in r and np.isfinite(r['fixed_train']['mae']) for r in history)
            model,ck=load_model(path,device='cpu');assert ck['protocol_sha256']==sha256(ROOT/'protocol.json')
            assert sum(p.numel() for p in model.parameters())==record['total_parameters']
            if name=='late_fusion':
                assert sum(p.numel() for p in model.parameters() if p.requires_grad)==422
                for expert,source in zip(model.experts,expert_paths(seed)):
                    original,_=load_model(source,device='cpu');assert state_hash(expert)==state_hash(original)
                    assert sha256(source)==record['expert_sources'][str(source.relative_to(PROJECT))]
                    assert all(not p.requires_grad for p in expert.parameters())
            runs.append(dict(model=name,seed=seed,epochs=len(history),sha256=record['sha256']))
        proof=json.loads((ROOT/f'results/evaluated_valid_{name}.json').read_text());assert proof['protocol_sha256']==sha256(ROOT/'protocol.json')
        assert all(sha256(PROJECT/p)==h for p,h in proof['checkpoints'].items())
        p=dict(np.load(ROOT/f'results/valid_{name}_predictions.npz'));out=p['out']
        assert out.shape==(217,3,728,4) and np.isfinite(out).all() and (out[...,:3]>=0).all()
        assert np.allclose(out[...,:3].sum(-1),1,atol=1e-6) and np.max(np.abs(out[...,3]))<=3
        assert guardrails(summary(name),summary('old_gated'))==next(r['checks'] for r in lock['rows'] if r['model']==name)
    checks.update(six_fits_completed=True,fixed_train_diagnostics_each_epoch=True,all_selected_experts_unchanged=True,full_validation_matrix=True,acceptance_gates_recomputed=True)
    boot=json.loads((ROOT/'results/paired_cluster_bootstrap.json').read_text());assert boot['repeats']==1000 and set(boot['results'])=={'old_gated','gated'} and boot['explicit_resampling_equivalence_verified']
    if not lock['accepted_improvement']:
        assert lock['selected']=='old_gated' and lock['final_test_models']==[] and not any(r['eligible'] for r in lock['rows'])
        assert not list((ROOT/'results').glob('test_*')) and not list(ROOT.glob('*.zip')) and not (ROOT/'results/附件3_问题二预测.csv').exists()
        checks['no_new_test_special_or_package_when_rejected']=True
    else:
        proof=json.loads((ROOT/f'results/evaluated_test_{lock["selected"]}.json').read_text());assert proof['selection_sha256']==sha256(ROOT/'selection_lock.json')
        package=json.loads((ROOT/'results/package_audit.json').read_text());assert package['bytes']<=50_000_000 and package['standalone_predictions_identical']
        checks['accepted_candidate_final_inference_and_package_verified']=True
    protected=json.loads((ROOT/'protected_files.json').read_text())
    for name,digest in protected.items():assert sha256(PROJECT/name)==digest,name
    assert not (PROJECT/'step2_v3').exists()
    assert sha256(ENCODER/'model.safetensors')==protocol['encoder']['weights_sha256']
    checks.update(old_artifacts_unchanged=True,v3_not_restored=True,encoder_unchanged=True,bootstrap_verified=True)
    required=['问题二瓶颈诊断报告.md','问题二任务独立融合实验报告.md','代码说明.md','results/refine_review.html','results/seed_summary.csv',
        'results/representative_errors.csv','results/clean_gate_summary.csv','results/expert_pretraining_cost.csv','environment.json']
    assert all((ROOT/p).is_file() and (ROOT/p).stat().st_size>0 for p in required)
    view=(ROOT/'results/refine_review.html').read_text();links=re.findall(r'(?:href|src)="([^"]+)"',view)
    for link in links:assert (ROOT/'results'/link).exists(),link
    save_json(ROOT/'code_manifest.json',{p.name:sha256(p) for p in sorted(ROOT.glob('*.py'))})
    save_json(ROOT/'results/completion_audit.json',dict(checks=checks,runs=runs,protected_files=len(protected),html_links=len(links),selected=lock['selected'],accepted_improvement=lock['accepted_improvement']))
    print(json.dumps(checks,indent=2),flush=True)


if __name__=='__main__':main()
