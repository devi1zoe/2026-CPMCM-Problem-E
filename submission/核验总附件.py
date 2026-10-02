# 本程序及代码是在人工智能工具辅助下完成的。

# 人工智能工具名称：GPT-6 Astra

# 版本/型号：gpt-6-astra

# 开发机构/公司：OpenAI

# 版本颁布日期：2026年9月3日

"""核验三问合并附件的文件完整性，并调用各问原有核验程序。"""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
import sys


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--files-only', action='store_true', help='仅用标准库检查总包文件大小与SHA256')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    manifest = json.loads((root / '总文件校验清单.json').read_text(encoding='utf-8'))
    for item in manifest['files']:
        path = root / item['path']
        if not path.is_file():
            raise FileNotFoundError(f'缺少文件：{item["path"]}')
        if path.stat().st_size != item['bytes'] or sha256(path) != item['sha256']:
            raise ValueError(f'文件内容不一致：{item["path"]}')
    print(f'总包完整性核验通过：{len(manifest["files"])}个文件。', flush=True)
    if args.files_only:
        return
    try:
        import numpy  # noqa: F401
    except ImportError:
        parser.exit(1, '完整核验需要NumPy；请使用已安装NumPy的Python解释器，或使用--files-only。\n')
    if sys.version_info < (3, 10):
        parser.exit(1, '完整核验需要Python 3.10及以上版本。\n')
    for folder, extra in [('问题一提交附件', []), ('问题二提交附件', []),
                          ('问题三提交附件', ['--numerical'])]:
        print(f'\n正在核验：{folder}', flush=True)
        subprocess.run([sys.executable, '-B', str(root / folder / '核验附件.py'), *extra],
                       cwd=root / folder, check=True)
    print('\n三问合并附件核验全部通过。', flush=True)


if __name__ == '__main__':
    main()
