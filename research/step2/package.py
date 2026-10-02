"""交付前按问题二要求检查成果，并生成不含缓存/预训练大权重的完整结果包。"""
from common import *
import zipfile
import numpy as np
import pandas as pd
import torch
from evaluate import scenarios
from train import MAIN,metrics,selection_score


def main():
    protocol=json.loads((ROOT/'protocol.json').read_text())
    lock=json.loads((ROOT/'selection_lock.json').read_text())
    assert sha256(ROOT/'protocol.json')==lock['protocol_sha256']
    assert lock['selected']==min(lock['scores'],key=lock['scores'].get)
    assert sha256(DATA/'附件2-数据集特征文件/aligned_50.pkl')==protocol['source_sha256']
    data_audit=json.loads((ROOT/'results/data_audit.json').read_text())
    assert data_audit['label_xlsx_rows_matched']==4850
    assert all(v==dict(sample=0,video=0) for v in data_audit['split_overlap'].values())
    assert all(v['token_exact_matches']==v['n'] for v in data_audit['splits'].values())
    validation=pd.read_csv(ROOT/'results/validation_metrics.csv')
    for variant in MAIN:
        rows=[validation[(validation.model==variant)&(validation.seed=='ensemble')&(validation.profile==p['name'])].iloc[0].to_dict()
              for p in protocol['validation_profiles']]
        assert abs(selection_score(rows)-lock['scores'][variant])<1e-10
    assert len(list((ROOT/'results').glob('completed_*.json')))==16
    for variant in protocol['models']:
        for seed in protocol['seeds'] if variant in MAIN else [17]:
            name=f'{variant}_{seed}'
            ckpt=torch.load(ROOT/f'checkpoints/{name}.pt',map_location='cpu',weights_only=True)
            history=json.loads((ROOT/f'results/history_{name}.json').read_text())
            assert abs(ckpt['score']-min(r['score'] for r in history))<1e-5
            saved_epoch=next(r for r in history if r['epoch']==ckpt['epoch'])
            assert abs(saved_epoch['score']-ckpt['score'])<1e-10
            assert all(torch.isfinite(t).all() for t in ckpt['state_dict'].values())
    for name,digest in lock['checkpoint_sha256'].items():
        assert sha256(ROOT/'checkpoints'/name)==digest
    checks=json.loads((ROOT/'results/verification.json').read_text())
    assert all(checks.values())
    test=pd.read_csv(ROOT/'results/test_metrics.csv')
    profiles=list(scenarios())
    for p in profiles:
        rows=test[(test.name==p['name'])&(test.repeat==p['repeat'])]
        assert len(rows)==25,(p,len(rows))
        assert set(rows[rows.seed=='ensemble'].model)==set(protocol['models'])|{'majority_mean'}
    assert len(test)==25*len(profiles)
    assert test[['accuracy','macro_f1','mae']].notna().all().all()
    assert test[['accuracy','macro_f1']].ge(0).all().all() and test[['accuracy','macro_f1']].le(1).all().all()
    for scenario in ['clean','TAV_r50_random']:
        saved=dict(np.load(ROOT/f'results/predictions_{scenario}.npz'))
        assert np.isfinite(saved['pred']).all() and np.all(np.abs(saved['pred'])<=3)
        assert np.allclose(saved['prob'].sum(1),1,atol=1e-6)
        recalc=metrics(saved['c'],saved['y'],saved['prob'],saved['pred'])
        recorded=test[(test.model==lock['selected'])&(test.seed=='ensemble')&(test.name==scenario)&(test.repeat==0)].iloc[0]
        for key in ['accuracy','macro_f1','mae','pearson']:assert abs(recalc[key]-recorded[key])<1e-10
    pred=pd.read_csv(ROOT/'results/附件3_问题二预测.csv')
    assert list(pred.columns)==['sample_id','polarity','intensity']
    assert len(pred)==30 and pred.sample_id.is_unique
    expected={f.stem for f in (DATA/'附件3-模态缺失特征样本/对齐版本').glob('*.pkl')}
    assert set(pred.sample_id)==expected
    assert pred.polarity.isin(['negative','neutral','positive']).all() and pred.intensity.between(-3,3).all()
    special_audit=json.loads((ROOT/'results/special_inference_audit.json').read_text())
    for name,digest in special_audit['special_sha256'].items():
        assert sha256(DATA/'附件3-模态缺失特征样本/对齐版本'/name)==digest
    required=['问题二实验报告.md','模型定义.md','代码说明.md','environment.json','results/problem2_review.html','results/error_groups.csv',
        'results/shape_sync_comparison.csv','results/ablation_seed_summary.csv','results/validation_metrics.csv',
        'results/corruption_audit.csv','results/paired_cluster_bootstrap.json','results/附件3_预测审阅.csv',
        'results/validation_error_groups.csv','results/validation_prediction_review.csv','results/validation_representative_errors.csv']
    for name in required:assert (ROOT/name).is_file() and (ROOT/name).stat().st_size>0,name
    assert len(list((ROOT/'results/figures').glob('*.png')))==8
    html=(ROOT/'results/problem2_review.html').read_text()
    for f in (ROOT/'results/figures').glob('*.png'):assert f.name in html
    audit=dict(requirements=dict(train_valid_test_separated=True,unified_aligned_interface=True,
        text_deletion_before_encoding=checks['text_masked_before_context_encoding'],
        eight_model_families=True,sixteen_completed_fits=True,three_seeds_for_all_four_main_variants=True,
        locked_validation_selection=True,missing_type_position_duration=True,shape_and_synchronization=True,
        classification_and_regression_metrics=True,source_video_cluster_bootstrap=True,
        all_thirty_special_predictions=True,report_code_checkpoints_and_environment=True),
        test_scenarios=len(profiles),test_metric_rows=len(test),special_rows=len(pred),selected=lock['selected'])
    save_json(ROOT/'results/completion_audit.json',audit)
    # 全部实验权重也入包，便于不重训直接复算对照；缓存和通用编码器按哈希重建。
    files=[p for p in ROOT.rglob('*') if p.is_file() and not any(q in ['cache','__pycache__'] for q in p.relative_to(ROOT).parts)
           and p.suffix!='.zip' and p.name not in ['package_size.json','delivery_manifest.json','package.log']]
    manifest={p.relative_to(ROOT).as_posix():dict(bytes=p.stat().st_size,sha256=sha256(p)) for p in sorted(files)}
    save_json(ROOT/'delivery_manifest.json',manifest);files.append(ROOT/'delivery_manifest.json')
    output=ROOT/'问题二交付.zip'
    with zipfile.ZipFile(output,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(files):z.write(p,arcname='step2/'+p.relative_to(ROOT).as_posix())
    with zipfile.ZipFile(output) as z:assert z.testzip() is None
    first=PROJECT/'step1/问题一交付.zip'
    combined=output.stat().st_size+(first.stat().st_size if first.exists() else 0)
    assert combined<=50_000_000,(combined,'combined attachment budget exceeded')
    save_json(ROOT/'results/package_size.json',dict(step2_bytes=output.stat().st_size,
        step1_plus_step2_bytes=combined,remaining_to_50MB=50_000_000-combined,zip_sha256=sha256(output)))
    print(json.dumps(dict(audit=audit,step2_zip_bytes=output.stat().st_size,combined_bytes=combined),ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
