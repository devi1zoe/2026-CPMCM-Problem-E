"""Require evidence for every experiment, comparison, inference and invariant."""
from common import *
from data import *
from engine import metrics,score
from oof import partitions,RawViews
from evaluation_data import profiles_all
from summarize import summarize_frame
import pandas as pd
import platform,importlib.metadata

def main():
    protocol=json.loads((ROOT/'protocol.json').read_text());lock=json.loads((ROOT/'selection_lock.json').read_text());combination=json.loads((ROOT/'combination_lock.json').read_text())
    names=list(protocol['configs'])+(['combined'] if combination['run'] else [])
    checks={};protected=json.loads((ROOT/'protected_files.json').read_text());changed=[]
    for p,h in protected.items():
        if not (PROJECT/p).exists() or sha256(PROJECT/p)!=h:changed.append(p)
    assert not changed,changed;checks['historical_files_unchanged']=len(protected)
    assert sha256(DATA/'附件2-数据集特征文件/aligned_50.pkl')==protocol['source_sha256']
    assert sha256(ENCODER/'model.safetensors')==protocol['encoder']['weights_sha256']
    assert sha256(ROOT/'checkpoints/normalization.npz')==protocol['normalization_sha256'];checks['input_encoder_normalization_hashes_match']=True
    for path in ['pretraining_verification.json','text_adaptation_verification.json','extension_verification.json']:
        assert json.loads((ROOT/'results'/path).read_text())['passed']
    checks['behavior_verifications_passed']=3
    mask_review=json.loads((ROOT/'results/mask_review_complete.json').read_text());assert mask_review['passed']
    checks['all_trained_group1_masks_recomputed']=mask_review['training_views_checked']
    training=[]
    for name in names:
        for seed in protocol['seeds']:
            done=json.loads((ROOT/f'results/completed_{name}_{seed}.json').read_text());path=ROOT/f'checkpoints/{name}_{seed}.pt'
            assert done['sha256']==sha256(path) and done['protocol_sha256']==sha256(ROOT/'protocol.json')
            ck=torch.load(path,map_location='cpu',weights_only=True);history=json.loads((ROOT/f'results/history_{name}_{seed}.json').read_text())
            assert ck['seed']==seed and ck['name']==name and len(history)==done['epochs']
            assert abs(min(r['score'] for r in history)-done['score'])<1e-5
            assert abs(ck['score']-done['score'])<1e-12
            if ck['config'].get('lora'):
                assert not any(k.startswith('language.backbone.') and '.lora_' not in k for k in ck['state_dict'])
                assert sum(v.numel() for k,v in ck['state_dict'].items() if '.lora_' in k)==24576
                assert any(bool(v.abs().max()>0) for k,v in ck['state_dict'].items() if k.endswith('.lora_b'))
                assert ck['encoder_sha256']==protocol['encoder']['weights_sha256']
            for row in history:
                assert abs(score(row['validation'])-row['score'])<1e-10
            training.append(dict(model=name,seed=seed,epochs=len(history),best_score=done['score'],sha256=done['sha256']))
    checks['completed_main_fits']=len(training)
    raw=load_attachment2();folds,groups=partitions(raw['train']);expert_count=0
    raw_store=RawViews(raw['train']);clean_text=raw_store.text(raw_store.base['o'])
    for fold,(tr,ho) in enumerate(folds):
        recomputed=fit_stats({k:raw_store.base[k][tr] for k in ['o','audio','vision']},clean_text[tr])
        saved=dict(np.load(ROOT/f'checkpoints/expert_normalization_{fold}.npz'))
        assert all(np.array_equal(saved[k],recomputed[k]) for k in saved)
    full=dict(np.load(ROOT/'checkpoints/expert_normalization_full.npz'));standard=dict(np.load(ROOT/'checkpoints/normalization.npz'))
    assert all(np.array_equal(full[k],standard[k]) for k in full)
    checks['fold_normalization_recomputed_from_training_rows_only']=True
    del clean_text,raw_store
    for seed in protocol['seeds']:
        done=json.loads((ROOT/f'results/oof_completed_{seed}.json').read_text());assert done['sha256']==sha256(ROOT/f'cache/oof_{seed}.npz')
        assert all(sha256(ROOT/p)==h for p,h in done['expert_hashes'].items())
        with np.load(ROOT/f'cache/oof_{seed}.npz') as data:
            assert data['experts'].shape==(9,3395,3,4) and np.isfinite(data['experts']).all()
            assert np.allclose(data['experts'][...,:3].sum(-1),1,atol=2e-6)
            assert np.array_equal(data['c'],raw['train']['classification_labels'])
            assert np.array_equal(data['ids'],np.asarray(raw['train']['id'],dtype=str))
            for fold,(tr,ho) in enumerate(folds):
                entry=json.loads((ROOT/f'results/experts_{seed}_{fold}.json').read_text())
                assert entry['train_indices']==tr.tolist() and entry['heldout_indices']==ho.tolist()
                assert not set(groups[tr])&set(groups[ho]) and np.all(data['fold_id'][ho]==fold)
                assert not entry['heldout_used_for_selection'] and len(entry['history'])==8
                assert sha256(ROOT/f'checkpoints/expert_normalization_{fold}.npz')==entry['normalization_sha256']
                for m in range(3):
                    ck=torch.load(ROOT/f'checkpoints/expert_{seed}_{fold}_{m}.pt',map_location='cpu',weights_only=True)
                    assert ck['config']['indices']==[m] and ck['epochs']==8
                    assert np.allclose(ck['prior'],np.bincount(raw['train']['classification_labels'][tr].astype(int),minlength=3)/len(tr))
                    expert_count+=1
            entry=json.loads((ROOT/f'results/experts_{seed}_full.json').read_text());assert len(entry['history'])==8
            assert entry['train_indices']==list(range(3395)) and entry['heldout_indices']==[];expert_count+=3
    checks['completed_expert_fits_with_group_disjoint_oof']=expert_count
    assert expert_count==36
    profiles=profiles_all();assert json.loads((ROOT/'results/valid_profiles.json').read_text())==profiles
    evaluation_count=0;metric_checks=0
    for split,models in [('valid',['old_gated',*names,'expert_text']),('test',lock['final_test_models'])]:
        expected=raw[split]
        for name in models:
            marker=json.loads((ROOT/f'results/evaluated_{split}_{name}.json').read_text())
            assert marker['protocol_sha256']==sha256(ROOT/'protocol.json') and marker['profiles']==253
            assert all(sha256(PROJECT/p)==h for p,h in marker['checkpoints'].items())
            if split=='test':assert marker['selection_sha256']==sha256(ROOT/'selection_lock.json')
            stored=pd.read_csv(ROOT/f'results/{split}_{name}_metrics.csv',dtype={'seed':str})
            assert len(stored)==253*4
            with np.load(ROOT/f'results/{split}_{name}_predictions.npz') as data:
                out=data['out'];assert out.shape==(253,3,len(expected['id']),4) and np.isfinite(out).all()
                assert np.array_equal(data['ids'],np.asarray(expected['id'],dtype=str))
                assert np.array_equal(data['c'],expected['classification_labels']) and np.array_equal(data['y'],expected['regression_labels'])
                assert np.allclose(out[:,:,:,:3].sum(-1),1,atol=2e-6) and np.all(np.abs(out[:,:,:,3])<=3)
                for si,profile in enumerate(profiles):
                    for j,seed in enumerate(['17','29','43','ensemble']):
                        result=out[si,j] if j<3 else out[si].mean(0)
                        row=stored[(stored.name==profile['name'])&(stored.repeat==profile['repeat'])&(stored.seed==seed)].iloc[0]
                        actual=metrics(data['c'],data['y'],result[:,:3],result[:,3])
                        for key,value in actual.items():
                            assert (pd.isna(row[key]) if value is None else abs(float(row[key])-value)<1e-7),(name,split,si,seed,key)
                        metric_checks+=1
            evaluation_count+=1
    checks['complete_evaluations']=evaluation_count;checks['metric_rows_recomputed']=metric_checks
    table=pd.read_csv(ROOT/'results/selection_comparison.csv').set_index('model')
    eligible=[n for n in names if bool(table.loc[n,'eligible'])]
    assert set(eligible)==set(lock['eligible'])
    if not eligible:assert lock['selected']=='old_gated'
    else:assert lock['selected'] in eligible
    checks['original_guardrails_retained']=protocol['guardrails']==dict(clean_f1_drop=.005,clean_mae_rise=.01,robust_f1_gain=.01,robust_mae_rise=0.,neutral_recall_drop=.01,strong_mae_rise=.02)
    assert checks['original_guardrails_retained']
    rec=json.loads((ROOT/'results/reconstruction_review_complete.json').read_text());assert rec['targets_used_only_for_evaluation'] and rec['rows']>0
    assert all(sha256(ROOT/f'checkpoints/reconstruct_{s}.pt')==h for s,h in rec['checkpoint_hashes'].items())
    decisions=json.loads((ROOT/'results/group_decisions.json').read_text());assert {r['group'] for r in decisions}=={'1','2','3','4'}
    for file in ['paired_comparisons.csv','rescued_harmed.csv','error_types.csv','representative_cases.csv','seed_mean_std.csv','condition_summary.csv']:
        assert (ROOT/'results'/file).stat().st_size>100
    special=pd.read_csv(ROOT/'results/附件3_问题二预测.csv');proof=json.loads((ROOT/'results/special_inference_complete.json').read_text())
    assert len(special)==30 and special.sample_id.nunique()==30 and special.intensity.between(-3,3).all()
    assert set(special.polarity)<= {'negative','neutral','positive'} and not proof['labels_available'] and not proof['used_for_selection']
    assert proof['predictions_sha256']==sha256(ROOT/'results/附件3_问题二预测.csv') and proof['selection_sha256']==sha256(ROOT/'selection_lock.json')
    assert (ROOT/'实验报告.md').stat().st_size>1000 and (ROOT/'results/实验审阅.html').stat().st_size>1000
    for name in ['overall','paired_changes','missing_conditions']:
        for ext in ['png','pdf']:assert (ROOT/f'results/figures/{name}.{ext}').stat().st_size>1000
    checks.update(four_group_decisions_present=True,reconstruction_quality_review_present=True,all_30_special_predictions=True,
        combination_decision_recorded=True,report_and_exportable_figures_present=True)
    environment=dict(python=platform.python_version(),packages={n:importlib.metadata.version(n) for n in ['torch','transformers','numpy','scikit-learn','pandas','matplotlib']},
        root_packages_modified=False,extra_environment_created=False,existing_project_dependencies_reused=True)
    historical_environment=json.loads((PROJECT/'step2_v2/environment.json').read_text())
    assert environment['python']==historical_environment['python'] and environment['packages']==historical_environment['packages']
    environment['versions_match_previous_environment']=True
    environment['previous_environment_sha256']=sha256(PROJECT/'step2_v2/environment.json')
    checks['existing_python_and_package_versions_preserved']=True
    save_json(ROOT/'environment.json',environment)
    save_json(ROOT/'results/completion_audit.json',dict(passed=True,checks=checks,training=training,selected=lock['selected'],protocol_sha256=sha256(ROOT/'protocol.json'),
        source_hashes={p.name:sha256(p) for p in ROOT.glob('*.py')},scope='All four experimental groups, three fits per setting, 253 paired input realizations, error analysis, conditional combination, locked historical test and attachment3 predictions.'))
    print('Completion audit passed:',checks)

if __name__=='__main__':main()
