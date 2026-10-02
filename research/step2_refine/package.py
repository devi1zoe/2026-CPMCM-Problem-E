"""Only accepted candidates generate a new <=50MB joint package."""
from common import *
import zipfile,tempfile,subprocess
import numpy as np,pandas as pd


def main():
    lock=json.loads((ROOT/'selection_lock.json').read_text());assert lock['accepted_improvement']
    protected=json.loads((ROOT/'protected_files.json').read_text())
    for name,digest in protected.items():assert sha256(PROJECT/name)==digest
    output=ROOT/'三问联合交付_任务独立融合版.zip';previous=PROJECT/'step2_v2/三问联合交付_问题二优化版.zip'
    files=[]
    for p in ROOT.rglob('*'):
        if not p.is_file() or p.is_symlink():continue
        parts=p.relative_to(ROOT).parts
        if parts[0] in ['cache','logs','__pycache__']:continue
        if parts[0]=='checkpoints' and p.name!='normalization.npz':continue
        if p.suffix not in ['.py','.json','.md','.csv','.html','.png','.pt','.npz']:continue
        if p.suffix=='.npz' and p.name!='normalization.npz':continue
        if p.name in ['package_audit.json','status.json','completion_audit.json']:continue
        if parts[0]=='results' and p.name.startswith(('valid_old_gated_','valid_gated_','valid_text_','valid_audio_','valid_vision_')):continue
        files.append(p)
    with zipfile.ZipFile(previous) as src,zipfile.ZipFile(output,'w',zipfile.ZIP_DEFLATED,compresslevel=9) as dst:
        for info in src.infolist():dst.writestr(info,src.read(info.filename))
        for p in sorted(files):dst.write(p,str(p.relative_to(PROJECT)))
        dst.writestr('任务独立融合交付说明.txt','本次问题二最终模型见 step2_refine/selection_lock.json；优先阅读该目录报告和代码说明。旧版和问题三保持可复现；研究权重只保留在工作区。')
    assert output.stat().st_size<=50_000_000,'Joint package exceeds hard limit'
    with tempfile.TemporaryDirectory(prefix='refine_package_') as temp:
        dest=Path(temp)
        with zipfile.ZipFile(output) as z:assert len(z.namelist())==len(set(z.namelist()));z.extractall(dest)
        for rel in ['step1/assets','step1/.venv','E题_材料阅读']:
            p=dest/rel;assert not p.exists();p.parent.mkdir(parents=True,exist_ok=True);p.symlink_to(PROJECT/rel,target_is_directory=True)
        subprocess.run([sys.executable,str(dest/'step2_refine/predict_package.py'),'--output',str(dest/'prediction.csv')],cwd=dest,check=True)
        a=pd.read_csv(ROOT/'results/附件3_问题二预测.csv');b=pd.read_csv(dest/'prediction.csv')
        assert len(b)==30 and b.sample_id.is_unique and a[['sample_id','polarity']].equals(b[['sample_id','polarity']])
        assert np.array_equal(a.intensity.values,b.intensity.values)
    save_json(ROOT/'results/package_audit.json',dict(package=output.name,bytes=output.stat().st_size,sha256=sha256(output),limit=50_000_000,standalone_predictions_identical=True,protected_files_unchanged=True))


if __name__=='__main__':main()
