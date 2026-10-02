# 词级强制对齐实例配图

## 正文推荐：简化原理图

- `figures/fig_5_4_ctc_principle.pdf`：以 HELLO 构造14个输出时刻，展示合法路径、合并连续重复、删除空白符及保留时间位置确定词区间。
- `ctc_principle_insert.tex`：建议放在介绍CTC折叠映射之后、最优路径公式之前。
- `build_ctc_principle.py`：矢量绘图脚本；图中字符和时间是教学构造，不是实测。可用根 Python 直接执行。

原实测图仍保留，可放入结果展示或附录。简化图展示折叠规则及时间定位，不展示动态规划的全部候选路径。

## 实测结果图

- `figures/fig_5_4_ctc_word_alignment.pdf`：论文配图，仅交付 PDF。
- `ctc_example_insert.tex`：建议插入词级置信度公式及解释之后、小节结束的过渡段之前。
- `build_ctc_example.py`：读取已有 CTC 概率、路径、词边界，解码原始音频绘图；不重新运行模型。
- `ctc_example_manifest.json`：样本来源、显示区间、逐词结果及图中路径位置。

图中取完整转写的既有最优路径的局部片段，不对节选文本另做对齐。热图每行是目标字符对应的模型概率；相同字符的行数值相同，不应将这些行解释为互斥位置类别。底部合并显示空白状态，灰点也包括词界符。图中词边界和置信度已根据保存的路径复算核对；没有人工真值时间标签。

复现命令：

```bash
/usr/local/python3.10/bin/python3 step1/paper_5_4/build_ctc_example.py
```

可使用 `--preview /tmp/ctc_word_alignment_preview.png` 生成内部检查用预览。未编译整篇 LaTeX 文档。
