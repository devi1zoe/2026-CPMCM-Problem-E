# 本程序及代码是在人工智能工具辅助下完成的。

# 人工智能工具名称：GPT-6 Astra

# 版本/型号：gpt-6-astra

# 开发机构/公司：OpenAI

# 版本颁布日期：2026年9月3日

"""获取已锁定版本的通用冻结模型；不会下载任何情感训练数据。"""
from common import ROOT, activate_project_dependencies, save_json, sha256
activate_project_dependencies()
import urllib.request
import argparse
from huggingface_hub import snapshot_download


def main():
    """仓库commit和下载后SHA256共同记录来源，模型权重不重复放进小型交付包。"""
    parser=argparse.ArgumentParser()
    parser.add_argument("--manifest-only",action="store_true")
    args=parser.parse_args()
    sources = [
        ("facebook/wav2vec2-base-960h", "22aad52d435eb6dbaf354bdad9b0da84ce7d6156", "wav2vec2",
         ["config.json", "preprocessor_config.json", "tokenizer_config.json", "vocab.json", "special_tokens_map.json", "model.safetensors"]),
        ("distilbert/distilbert-base-uncased", "12040accade4e8a0f71eabdb258fecc2e7e948be", "distilbert",
         ["config.json", "tokenizer_config.json", "tokenizer.json", "vocab.txt", "model.safetensors"])]
    if not args.manifest_only:
        for repo, revision, local, files in sources:
            snapshot_download(repo_id=repo, revision=revision, allow_patterns=files,
                              local_dir=ROOT/"assets"/local)
    url="https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
    target=ROOT/"assets/face_landmarker.task"
    if not target.exists() and not args.manifest_only:
        urllib.request.urlretrieve(url,target)
    manifest=dict(huggingface=[dict(repo=r,revision=v,directory=d) for r,v,d,_ in sources],
                  mediapipe_url=url,
                  sha256={p.relative_to(ROOT).as_posix():sha256(p) for p in (ROOT/"assets").rglob("*")
                          if p.is_file() and ".cache" not in p.parts})
    save_json(ROOT/"assets_manifest.json",manifest)


if __name__ == "__main__":
    main()
