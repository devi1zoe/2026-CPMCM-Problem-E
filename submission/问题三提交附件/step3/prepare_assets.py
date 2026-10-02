# 本程序及代码是在人工智能工具辅助下完成的。

# 人工智能工具名称：GPT-6 Astra

# 版本/型号：gpt-6-astra

# 开发机构/公司：OpenAI

# 版本颁布日期：2026年9月3日

"""下载固定版本的冻结编码器和时间对齐模型，逐文件核对哈希。"""
from common import *
import argparse


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--models', nargs='+', choices=['distilbert', 'wav2vec2'], default=['distilbert','wav2vec2'])
    args = ap.parse_args()
    from huggingface_hub import snapshot_download
    spec = json.loads((SOURCE_ROOT / 'assets_manifest.json').read_text())
    for item in spec['huggingface']:
        folder = item['directory']
        if folder not in args.models:
            continue
        selected = {k.split('/')[-1]: v for k,v in spec['sha256'].items() if k.startswith(f'assets/{folder}/')}
        destination = args.output / folder
        snapshot_download(repo_id=item['repo'], revision=item['revision'],
                          allow_patterns=list(selected), local_dir=destination)
        for name,digest in selected.items():
            assert sha256(destination / name) == digest, f'{folder}/{name}版本不符'
        print(f'{folder}获取及校验完成')


if __name__ == '__main__':
    main()
