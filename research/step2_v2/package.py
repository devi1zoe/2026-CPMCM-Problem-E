"""Build a <=50,000,000-byte combined submission; preserve original directories and verify extracted inference."""
from common import *
import zipfile,tempfile,subprocess
import numpy as np,pandas as pd

def main():
    protected=json.loads((ROOT/'protected_files.json').read_text())
    for path,digest in protected.items():assert sha256(PROJECT/path)==digest,('Old file changed',path)
    protocol=json.loads((ROOT/'protocol.json').read_text());lock=json.loads((ROOT/'selection_lock.json').read_text())
    assert sha256(ROOT/'protocol.json')==lock['protocol_sha256']
    assert sha256(ENCODER/'model.safetensors')==protocol['encoder']['weights_sha256']
    checks=json.loads((ROOT/'results/verification.json').read_text());assert all(checks.values())
    assert json.loads((ROOT/'results/bootstrap_verification.json').read_text())['group_multiplicity_matches_explicit_resampling']
    assert all(json.loads((ROOT/'results/frozen_encoder_rng_audit.json').read_text()).values())
    expected=pd.read_csv(ROOT/'results/附件3_问题二预测.csv');assert len(expected)==30 and expected.sample_id.is_unique
    assert expected.intensity.between(-3,3).all() and expected.polarity.isin(['negative','neutral','positive']).all()
    for name in protocol['configs']:
        for s in protocol['seeds']:assert (ROOT/f'results/completed_{name}_{s}.json').exists()
    for name in json.loads((ROOT/'ablation_configs.json').read_text()):
        for s in protocol['seeds']:assert (ROOT/f'results/completed_{name}_{s}.json').exists()
    names=['old_gated',*protocol['configs'],*json.loads((ROOT/'ablation_configs.json').read_text())]
    for name in names:
        df=pd.read_csv(ROOT/f'results/valid_{name}_metrics.csv');assert len(df)==217*4
        assert len(df[(df.seed=='ensemble')&(df.primary==True)].groupby('name'))==112
    for name in lock['final_test_models']:
        proof=json.loads((ROOT/f'results/evaluated_test_{name}.json').read_text())
        assert proof['selection_sha256']==sha256(ROOT/'selection_lock.json')
    code_manifest={p.name:sha256(p) for p in sorted(ROOT.glob('*.py'))}
    save_json(ROOT/'code_manifest.json',code_manifest)
    output=ROOT/'三问联合交付_问题二优化版.zip'
    payload={}
    for k,zh in [(1,'一'),(3,'三')]:
        with zipfile.ZipFile(PROJECT/f'step{k}/问题{zh}交付.zip') as z:
            for info in z.infolist():
                if not info.is_dir():payload[info.filename]=z.read(info.filename)
    # Keep old Q2 code/configs and Q3's initialization dependencies, not every old ablation weight.
    old=PROJECT/'step2'
    needed=[*old.glob('*.py'),*old.glob('*.md'),old/'protocol.json',old/'selection_lock.json',old/'environment.json',old/'checkpoints/normalization.npz']
    needed += [old/f'checkpoints/gated_aug_{s}.pt' for s in [17,29,43]]
    for path in needed:payload[path.relative_to(PROJECT).as_posix()]=path.read_bytes()
    exclude={'protected_files.json'}
    for p in ROOT.rglob('*'):
        if not p.is_file():continue
        rel=p.relative_to(ROOT)
        if any(x in ['cache','logs','__pycache__'] for x in rel.parts):continue
        if p.suffix in ['.zip','.lock'] or p.name in exclude:continue
        if rel.parts[0]=='checkpoints' and p.name!='normalization.npz':continue
        if p.suffix=='.npz':
            if p.name!='normalization.npz':continue
        if p.name in ['package_audit.json','delivery_manifest.json','run_status.json']:continue
        payload['step2_v2/'+rel.as_posix()]=p.read_bytes()
    note='''联合交付说明\n\n本包保留问题一、问题三原成果，问题二最终结果见 step2_v2/results/附件3_问题二预测.csv。\n冻结通用DistilBERT及赛题原附件不含在本包，请按 step2_v2/代码说明.md 提供精确版本与原数据。\n问题三无需重训；其最终推理权重完整保留，旧问题二门控初始化权重也保留。\n旧问题二其余消融检查点可由保留的原代码/配置重训。大缓存及本轮非最终检查点保留在工作区，不计入联合包。\n请独立解压，不覆盖原实验目录。\n'''
    payload['联合交付说明.txt']=note.encode()
    manifest={name:dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest()) for name,data in sorted(payload.items())}
    payload['step2_v2/delivery_manifest.json']=json.dumps(manifest,ensure_ascii=False,indent=2).encode()
    with zipfile.ZipFile(output,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for name,data in sorted(payload.items()):z.writestr(name,data)
    assert output.stat().st_size<=50_000_000,(output.stat().st_size,'Reduce redundant artifacts, never omit required Q1 features or final weights')
    with zipfile.ZipFile(output) as z:assert z.testzip() is None
    # Independently import and run only from extracted sources and exported parameters.
    with tempfile.TemporaryDirectory(prefix='delivery_check_',dir=ROOT/'cache') as temp:
        temp=Path(temp)
        with zipfile.ZipFile(output) as z:z.extractall(temp)
        (temp/'E题_材料阅读').symlink_to(PROJECT/'E题_材料阅读',target_is_directory=True)
        (temp/'step1/.venv').symlink_to(PROJECT/'step1/.venv',target_is_directory=True)
        assets=temp/'step1/assets';assets.mkdir(exist_ok=True)
        (assets/'distilbert').symlink_to(ENCODER,target_is_directory=True)
        pred=temp/'recomputed.csv'
        result=subprocess.run([sys.executable,str(temp/'step2_v2/predict_package.py'),'--output',str(pred)],cwd=temp,capture_output=True,text=True,env=os.environ.copy())
        assert result.returncode==0,result.stdout+'\n'+result.stderr
        actual=pd.read_csv(pred);assert expected.sample_id.equals(actual.sample_id) and expected.polarity.equals(actual.polarity)
        assert np.allclose(expected.intensity,actual.intensity,rtol=0,atol=1e-7)
        for name,digest in json.loads((temp/'step3/selection_lock.json').read_text())['checkpoints'].items():
            assert sha256(temp/'step3/checkpoints'/name)==digest
    audit=dict(bytes=output.stat().st_size,limit=50_000_000,remaining=50_000_000-output.stat().st_size,
        sha256=sha256(output),old_files_verified=len(protected),old_directories_unchanged=True,
        extracted_independent_special_inference=True,all_30_predictions_match=True,step3_final_weights_verified=True,
        main_fits=30,validation_inputs_per_model=217,final_selected=lock['selected'],accepted_improvement=lock['accepted_improvement'])
    save_json(ROOT/'results/package_audit.json',audit)
    # Include verification result without a self-referential archive hash.
    with zipfile.ZipFile(output,'a',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        z.writestr('step2_v2/results/package_audit.json',json.dumps({**{k:v for k,v in audit.items() if k not in ['bytes','remaining','sha256']},
            'note':'Final compressed bytes and whole-archive SHA256 are in the workspace audit; omitted here to avoid self-reference.'},ensure_ascii=False,indent=2))
    assert output.stat().st_size<=50_000_000
    audit.update(bytes=output.stat().st_size,remaining=50_000_000-output.stat().st_size,sha256=sha256(output))
    save_json(ROOT/'results/package_audit.json',audit)
    print(json.dumps(audit,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':main()
