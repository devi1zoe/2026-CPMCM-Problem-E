# 本程序及代码是在人工智能工具辅助下完成的。

# 人工智能工具名称：GPT-6 Astra

# 版本/型号：gpt-6-astra

# 开发机构/公司：OpenAI

# 版本颁布日期：2026年9月3日

"""问题二路径配置；输入目录和复现输出目录可独立指定。"""
from pathlib import Path
import hashlib
import json
import os
import sys

PACKAGE = Path(__file__).resolve().parents[1]
ROOT = Path(os.environ.get('P2_WORKDIR', PACKAGE / '复现输出')).expanduser().resolve()
DATA = Path(os.environ.get('P2_DATA_ROOT', PACKAGE / '赛题数据')).expanduser().resolve()
ENCODER = Path(os.environ.get('P2_ENCODER', PACKAGE / 'assets/distilbert')).expanduser().resolve()
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
for name in ['cache', 'results', 'checkpoints']:
    (ROOT / name).mkdir(parents=True, exist_ok=True)

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
