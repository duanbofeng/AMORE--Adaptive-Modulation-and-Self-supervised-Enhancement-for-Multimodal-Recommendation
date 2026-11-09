# 图表与证据表达学习示例

直接打开 **[report.html](report.html)**。报告可离线浏览，图片内嵌；总体表、噪声和分组部分可以切换 Baby / Sports / Clothing 场景。

这是独立生成的模拟材料，不读取本仓库的真实实验结果。方法和数据集名称是场景标签，不代表其真实表现或合理成绩范围。图角落保留小字号来源说明，没有大水印。

## 内容

- 13 幅图，PNG 与可编辑 SVG：总体性能、分支 2×2、损失热图与 MACA 2×2、敏感性、三类噪声、用户/物品/零交互分组、效率、收敛、四类证据解释。
- 13 个 CSV：逐模拟种子记录、均值/SD/配对区间、分组、假设计时等；各表都有来源字段。
- [paper_style_examples.md](paper_style_examples.md)：中文提示、英文图注及带场景前提的讨论/回复表达练习。
- [manifest.json](manifest.json)：固定随机数、生成器指纹、依赖版本及图表清单。

## 场景设计

主场景给予 AMORE 有限收益，但 Clothing 可落后于参照；辅助目标有强弱之分；亲和力并非每类都有效；噪声与冷启动存在失败条件；较高精度伴随额外计算代价。

所有数据来自明确的情景参数和一次固定随机生成，没有读取或拟合历史结果，也不反复挑随机种子直到区间“显著”。这些情景并不是实际实验“应当得到”的结果。

统计图中的 SD 表示模拟运行间波动，95% CI 使用五个配对差值的 t 区间；它们说明统计表达方式，不是对实际方法的科学证据。数据是聚合层面的场景，没有虚构真实逐用户排名、硬件测量或样本数。

## 复现

仓库根目录下，用包含 numpy、pandas、scipy、matplotlib 的 Python 环境运行：

```bash
python learning_examples/revision_scenarios/generate.py
python learning_examples/revision_scenarios/build_report.py
python learning_examples/revision_scenarios/validate.py
```

`validation.json` 记录跨表一致性核对，包括零权重与移除项一致、零噪声与干净条件一致，以及用户分组加权回到总体分数。

所有生成文件只位于本目录，不修改论文原文、正式审稿回复、训练代码或真实结果目录。

`figures/` 中的 SVG 适合学习版式、修改字号和配色；`data/` 可用于练习重新绘图。单独复制 HTML 仍能看图片，SVG/CSV 下载链接则需要保留目录结构。
