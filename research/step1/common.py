"""问题一的路径、元数据和可复现性工具，不读取附件2～4。"""
from pathlib import Path
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
DATA = PROJECT / "E题_材料阅读/E题数据/附件1-数据集原始多模态样本/MOSEI数据集部分原始视频-100条"


def activate_project_dependencies():
    """沿用根解释器与原有 torch；仅从现有项目 venv 读取补充依赖。"""
    local = ROOT / ".venv/lib/python3.10/site-packages"
    if local.is_dir():
        sys.path.insert(0, str(local))


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_samples():
    """视频编号按文本读取，核验100条标签与100个原视频精确一一对应。"""
    import openpyxl
    wb = openpyxl.load_workbook(DATA / "label-100.xlsx", read_only=True, data_only=True)
    rows = list(wb["label"].values)
    samples = []
    for values in rows[1:]:
        row = dict(zip(rows[0], values))
        video_id, clip_id = str(row["video_id"]), str(row["clip_id"])
        path = DATA / video_id / (clip_id + ".mp4")
        assert path.is_file(), path
        samples.append(dict(sample_id=video_id + "$_$" + clip_id,
                            video_id=video_id, clip_id=clip_id, raw_text=str(row["text"]),
                            label=float(row["label"]), annotation=str(row["annotation"]),
                            source_path=path.relative_to(PROJECT).as_posix(),
                            source_sha256=sha256(path)))
    wb.close()
    assert len(samples) == len({x["sample_id"] for x in samples}) == 100
    assert {PROJECT / x["source_path"] for x in samples} == set(DATA.rglob("*.mp4"))
    return samples
