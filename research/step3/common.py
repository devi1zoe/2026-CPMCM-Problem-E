"""问题三路径与已有环境复用；不改动问题一/二及根Python环境。"""
from pathlib import Path
import os,sys,json,hashlib,pickle
ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parent
P2=PROJECT/'step2'
DATA=PROJECT/'E题_材料阅读/E题数据'
SPECIAL=DATA/'附件4-可解释专项视频样本与特征文件/附件4-可解释专项视频样本与特征文件/对齐版本'
ENCODER=PROJECT/'step1/assets/distilbert'
ASSETS=PROJECT/'step1/assets'
sys.path.insert(0,str(PROJECT/'step1/.venv/lib/python3.10/site-packages'))
os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')

def save_json(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')

def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

class CompatUnpickler(pickle.Unpickler):
    def find_class(self,module,name):
        # NumPy2写出的模块路径，对应现有NumPy1.26的同一数组重建函数。
        if module.startswith('numpy._core'):module=module.replace('numpy._core','numpy.core',1)
        return super().find_class(module,name)

def load_special():
    import numpy as np
    files=sorted(SPECIAL.glob('*.pkl'));assert len(files)==20
    rows=[]
    for p in files:
        with open(p,'rb') as f:row=CompatUnpickler(f).load()
        assert str(row['id'])==p.stem and (SPECIAL/'videos'/(p.stem+'.mp4')).is_file()
        rows.append(row)
    return dict(id=[p.stem for p in files],raw_text=np.array([r['raw_text'] for r in rows]),
        **{k:np.stack([r[k] for r in rows]) for k in ['text_bert','audio','vision']})
