"""问题二公共路径与可复现记录。复用现有项目依赖，不修改根环境。"""
from pathlib import Path
import hashlib
import json
import os
import sys

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
DATA = PROJECT / 'E题_材料阅读/E题数据'
ENCODER = PROJECT / 'step1/assets/distilbert'
VENV = PROJECT / 'step1/.venv/lib/python3.10/site-packages'
if VENV.is_dir():
    sys.path.insert(0, str(VENV))
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()
