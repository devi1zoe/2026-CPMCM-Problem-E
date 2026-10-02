# 本程序及代码是在人工智能工具辅助下完成的。

# 人工智能工具名称：GPT-6 Astra

# 版本/型号：gpt-6-astra

# 开发机构/公司：OpenAI

# 版本颁布日期：2026年9月3日

"""下载锁定版本的公开冻结编码器，并逐文件校验。"""
import argparse
import json
from common import PACKAGE, sha256
from huggingface_hub import snapshot_download


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', required=True)
    args = ap.parse_args()
    spec = json.loads((PACKAGE / '配置/encoder.json').read_text())
    from pathlib import Path
    out = Path(args.output)
    snapshot_download(repo_id=spec['repo'], revision=spec['revision'],
                      allow_patterns=list(spec['files']), local_dir=out)
    for name, digest in spec['files'].items():
        if sha256(out / name) != digest:
            raise ValueError(f'编码器文件校验失败：{name}')
    print('冻结编码器版本及全部文件校验通过')


if __name__ == '__main__':
    main()
