# 本程序及代码是在人工智能工具辅助下完成的。

# 人工智能工具名称：GPT-6 Astra

# 版本/型号：gpt-6-astra

# 开发机构/公司：OpenAI

# 版本颁布日期：2026年9月3日

"""比较独立试跑与正式提取中重复的两条样本，而非只检查文件存在。"""
from common import ROOT,save_json
import numpy as np


def main():
    """浮点逐元素容差核验、离散字段精确核验，并公开实际最大差。"""
    rows=[]
    for p in sorted((ROOT/"smoke/samples").glob("*/features.npz")):
        q=ROOT/"results/samples"/p.parent.name/"features.npz"
        with np.load(p) as a,np.load(q) as b:
            assert set(a.files)==set(b.files)
            differences={}
            for key in a.files:
                assert a[key].shape==b[key].shape
                if a[key].dtype.kind in "fc":
                    error=float(np.max(np.abs(a[key].astype(float)-b[key].astype(float)))) if a[key].size else 0.
                    differences[key]=error
                    np.testing.assert_allclose(a[key],b[key],rtol=1e-5,atol=1e-6)
                else:
                    np.testing.assert_array_equal(a[key],b[key])
            rows.append(dict(sample_id=p.parent.name,max_absolute_difference=max(differences.values()),
                             float_arrays=differences,passed=True))
    assert len(rows)==2,"需要先按说明独立试跑前两条样本"
    save_json(ROOT/"results/reproducibility.json",dict(scope="独立试跑与正式运行的2条样本全部NPZ字段对照",
              samples=rows,tolerance=dict(rtol=1e-5,atol=1e-6)))
    print([(r["sample_id"],r["max_absolute_difference"]) for r in rows])


if __name__=="__main__":
    main()
