"""打包问题一代码与真实结果，排除预训练模型、环境和临时试跑。"""
from common import ROOT, save_json, sha256
from zipfile import ZipFile, ZIP_DEFLATED


def main():
    """先生成文件清单与校验值，再核对ZIP可读性和50 MB大小上限。"""
    files=[]
    for p in ROOT.iterdir():
        if p.is_file() and p.suffix in (".py", ".md", ".json", ".txt") and p.name!="delivery_manifest.json":
            files.append(p)
    files += [p for p in (ROOT/"results").rglob("*") if p.is_file()]
    files += [p for p in (ROOT/"logs").glob("*.log") if p.name!="dependency_install.log"]
    manifest=dict(scope="仅问题一", excluded=["assets预训练权重", ".venv环境", ".pip-cache", "smoke/repro试跑副本"],
                  files=[dict(path=p.relative_to(ROOT).as_posix(),bytes=p.stat().st_size,sha256=sha256(p))
                         for p in sorted(files)])
    save_json(ROOT/"delivery_manifest.json",manifest)
    target=ROOT/"问题一交付.zip"
    with ZipFile(target,"w",compression=ZIP_DEFLATED,compresslevel=6) as archive:
        for p in sorted(files+[ROOT/"delivery_manifest.json"]):
            archive.write(p,"step1/"+p.relative_to(ROOT).as_posix())
    with ZipFile(target) as archive:
        assert archive.testzip() is None
    size=target.stat().st_size
    print(dict(zip=str(target),bytes=size,decimal_MB=size/1e6,MiB=size/1024**2,files=len(files)+1))
    if size>50_000_000:
        raise SystemExit("本问包超过50MB，不可声称满足附件大小要求")


if __name__=="__main__":
    main()
