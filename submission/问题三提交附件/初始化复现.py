# 本程序及代码是在人工智能工具辅助下完成的。

# 人工智能工具名称：GPT-6 Astra

# 版本/型号：gpt-6-astra

# 开发机构/公司：OpenAI

# 版本颁布日期：2026年9月3日

"""在新目录准备问题三复现输入，保留提交结果和模型不变。"""
import argparse
import shutil
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--training', action='store_true', help='从训练初始化模型开始，暂不复制最终模型与选型锁')
    args = ap.parse_args()
    src = Path(__file__).resolve().parent / 'step3'
    out = args.output.expanduser().resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError('请指定新的或空的复现目录')
    for name in ['checkpoints', 'cache', 'results', 'results/cards', 'results/mapping', 'results/figures']:
        (out / name).mkdir(parents=True, exist_ok=True)
    for name in ['protocol.json', 'numerical_protocol.json', 'analysis_protocol.json']:
        shutil.copy2(src / name, out / name)
    if not args.training:
        shutil.copy2(src / 'selection_lock.json', out / 'selection_lock.json')
        for p in (src / 'checkpoints').iterdir():
            shutil.copy2(p, out / 'checkpoints' / p.name)
        for p in (src / 'results/mapping').glob('*.json'):
            shutil.copy2(p, out / 'results/mapping' / p.name)
    print(f'复现目录已准备：{out}')
    print('运行后续程序前，将环境变量P3_WORKDIR设为该目录。')


if __name__ == '__main__':
    main()
