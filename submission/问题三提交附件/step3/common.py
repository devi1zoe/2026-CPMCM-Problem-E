# 本程序及代码是在人工智能工具辅助下完成的。

# 人工智能工具名称：GPT-6 Astra

# 版本/型号：gpt-6-astra

# 开发机构/公司：OpenAI

# 版本颁布日期：2026年9月3日

"""问题三数据、模型资源与复现输出路径。"""
from pathlib import Path
import os,sys,json,hashlib,pickle
SOURCE_ROOT=Path(__file__).resolve().parent
PACKAGE=SOURCE_ROOT.parent
ROOT=Path(os.environ.get('P3_WORKDIR',SOURCE_ROOT)).expanduser().resolve()
PROJECT=PACKAGE
P2=Path(os.environ.get('P3_P2_ROOT',PACKAGE/'训练初始化/step2')).expanduser().resolve()
DATA=Path(os.environ.get('P3_DATA_ROOT',PACKAGE/'赛题数据')).expanduser().resolve()
SPECIAL=Path(os.environ.get('P3_SPECIAL',DATA/'附件4-可解释专项视频样本与特征文件/附件4-可解释专项视频样本与特征文件/对齐版本')).expanduser().resolve()
ASSETS=Path(os.environ.get('P3_ASSETS',PACKAGE/'assets')).expanduser().resolve()
ENCODER=ASSETS/'distilbert'
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
        if module.startswith('numpy._core'):module=module.replace('numpy._core','numpy.core',1)
        return super().find_class(module,name)

def load_special():
    import numpy as np
    files=sorted(SPECIAL.glob('*.pkl'));assert [p.stem for p in files]==[f'{i:02d}' for i in range(1,21)]
    rows=[]
    for p in files:
        with open(p,'rb') as f:row=CompatUnpickler(f).load()
        assert str(row['id'])==p.stem and (SPECIAL/'videos'/(p.stem+'.mp4')).is_file()
        rows.append(row)
    return dict(id=[p.stem for p in files],raw_text=np.array([r['raw_text'] for r in rows]),
        **{k:np.stack([r[k] for r in rows]) for k in ['text_bert','audio','vision']})
