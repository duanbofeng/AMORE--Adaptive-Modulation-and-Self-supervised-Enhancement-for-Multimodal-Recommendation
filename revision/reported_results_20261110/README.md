# AMORE 真实汇总结果与论文修改

首先用浏览器打开 **[report.html](report.html)**。报告含完整表格、六张图和可展开的原句替换建议，可离线浏览。

## 文件入口

- [逐句修改清单](manuscript_changes.md)：原文、替换英文、理由及插入位置；包括原 Word 与本次数据的数值差异。
- [审稿回复](reviewer_responses.md)：R1.1、R1.2/R3.4、R2.1、R3.2 的英文回复建议。
- [新增英文实验小节](factorial_section_en.md)：传播与融合的四组合及交互项分析。
- [英文图注](figure_captions.md)。
- `figures/`：六张图各有 PNG（300 dpi）、可编辑 SVG 和矢量 PDF。
- `tables/`：总体表现、模块消融、亲和力四组合的 LaTeX 表格；Word 可从浏览器复制表格。
- `data/`：输入值、提升/降幅计算、交互项与原稿数值差异 CSV。

## 本次输入

数值来自作者于 2026-11-10 提供的三张实验表，并保存在 [inputs.json](inputs.json)。总体表包含 12 个基线和 AMORE；普通消融为 Recall@20 / NDCG@20；图三已确认按 Recall@10 整理，列顺序为 Baby、Sports、Clothing。

报告只使用这三组数值，没有导入此前模拟数据，也没有补造运行记录。尚未提供种子数及逐次值，所以图中不添加误差线或显著性标记。报告与修改清单是建议稿，未直接改写 `papers/AMORE.docx`。

## 建议放进论文的材料

| 内容 | 推荐材料 | 对应意见 |
|---|---|---|
| 总体排名 | Table 2；可选图 `02_relative_improvement` | R1.1 |
| 整块模块贡献 | Table 3；图 `04_module_drop` | R1.2、R3.4 |
| 亲和力传播与融合 | 新四组合表；图 `05_affinity_factorial` | R2.1、R3.4 |
| 交互项和联合收益的区别 | 图 `06_affinity_effects`，适合回复信或补充材料 | R2.1 |
| 原始绝对值的图形表达 | `01_overall_comparison`、`03_module_ablation` | 可作为补充材料 |

不必把六张图全部放进主文。保持总体表、消融表、消融降幅图与亲和力交互图即可覆盖主要信息。

## 复现

在包含 numpy、pandas、matplotlib、lxml 的 Python 环境中，从仓库根目录运行：

```bash
python revision/reported_results_20261110/build.py
```

脚本重算图表并验证提升百分比、完整模型锚点、交互项和原文引用，结果见 `validation.json`。所有生成文件均写入本目录。它读取当前 `papers/AMORE.docx` 以定位原句；如果原稿已修改，应相应更新文本定位。

## 一处输入差异

Sports 的 MMGCN NDCG@20：文字表为 **0.0254**，截图看起来为 **0.0612**。当前采用文字表的 0.0254。除此之外，当前数值按本次输入保存，不用旧实验或模拟数据替代。
