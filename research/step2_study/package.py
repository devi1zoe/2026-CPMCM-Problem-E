"""Small reading/inference bundle; research caches and generic weights stay out."""
from common import *
import zipfile
import torch

def main():
    audit=json.loads((ROOT/'results/completion_audit.json').read_text());assert audit['passed']
    lock=json.loads((ROOT/'selection_lock.json').read_text());files=set()
    for pattern in ['*.py','*.md','protocol.json','selection_rules.json','selection_lock.json','combination_lock.json','environment.json']:
        files.update(ROOT.glob(pattern))
    files.add(ROOT/'checkpoints/normalization.npz')
    for p,h in lock['checkpoint_hashes'][lock['selected']].items():
        path=PROJECT/p;assert sha256(path)==h;files.add(path)
        ck=torch.load(path,map_location='cpu',weights_only=True)
        for ep,eh in ck.get('expert_hashes',{}).items():
            expert=ROOT/ep;assert sha256(expert)==eh;files.add(expert)
    for pattern in ['*.csv','*.json','*.html','figures/*.png','figures/*.pdf']:
        files.update((ROOT/'results').glob(pattern))
    files.discard(ROOT/'results/finalization_ready.json')
    readme=ROOT/'材料包说明.md'
    readme.write_text('''# 材料包说明

本包包含问题二四组实验的报告、完整指标表、实际案例、导出图、30条最终预测及最终模型所需的学习权重。先阅读 `step2_study/实验报告.md`，或打开 `step2_study/results/实验审阅.html`。

为控制体积，不包含训练视图缓存、所有研究候选权重、原始赛题数据及通用文本编码器。它们仍保存在当前项目中。完整实验记录和所有研究权重见工作目录 `step2_study`；运行训练命令需保留原项目及其准备好的数据。最终推理也需原赛题数据和现有 `step1/assets/distilbert`，确切编码器名称、修订与校验值见 protocol.json。

这是问题二的阅读与最终推理材料包，不代表三问合并后的完整竞赛提交包。历史四个问题二目录和问题三材料均未替换。
''',encoding='utf-8');files.add(readme)
    bundle=ROOT/'问题二四组实验_报告与最终预测.zip'
    with zipfile.ZipFile(bundle,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
        for path in sorted(files):archive.write(path,str(path.relative_to(PROJECT)))
    with zipfile.ZipFile(bundle) as archive:
        assert archive.testzip() is None
        assert len(archive.namelist())==len(files)
    assert bundle.stat().st_size<50000000
    save_json(ROOT/'delivery_manifest.json',dict(bundle=bundle.name,bytes=bundle.stat().st_size,sha256=sha256(bundle),files=len(files),
        scope='Question2 reading and final inference materials; not a combined three-question competition submission',
        entries={str(p.relative_to(PROJECT)):sha256(p) for p in sorted(files)}))
    print('Verified bundle:',bundle.name,bundle.stat().st_size,'bytes',len(files),'files')

if __name__=='__main__':main()
