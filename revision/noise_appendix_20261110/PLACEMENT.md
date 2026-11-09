# 噪声图的放置位置与正文衔接

## 结论

当前这组六图建议作为**情景分析附录**，不要作为正文中的已完成鲁棒性实验。若投稿稿只保留实证结果，则暂不收入这组六图。

依据是当前目录中的 `metadata.json`、`generate.py` 和 `noise_data.csv`：27 行零噪声记录为 `paper_reported`，108 行非零噪声记录为 `synthetic_scenario`。30% 噪声下降幅度是生成器的输入假设。零噪声数值与最新总体表一致，不改变后续曲线的来源。

附录成稿见 **[APPENDIX_EN.md](APPENDIX_EN.md)**，含六张图、英文分析和图注。图注中的情景性质说明应随图保留。该附录用于解释如何同时读取绝对分数与相对下降，不构成 R3.6 已被实验解决的证据。

## 放在哪里

按当前 `papers/AMORE.docx` 的章节结构：

1. 保留 `4.4 Hyper-parameter Sensitivity (RQ3)` 和 `4.5 Visualization Analysis (RQ4)` 的位置。
2. 当前不新增名为 `Robustness Evaluation` 的真实结果小节。
3. 若决定保留情景分析，在论文末尾新增 `Appendix A. Illustrative Corruption Scenarios`。可放在参考文献之前，并按最终期刊模板安排其与声明部分的先后；若已有 Appendix A，顺延字母。
4. 在 **4.5 的最后一段之后、5 Conclusion and Future Work 之前**增加以下衔接句：

> Appendix A provides illustrative corruption scenarios to distinguish absolute ranking performance from degradation relative to each method's clean score. The nonzero-corruption values are constructed responses rather than measured outcomes and are not used to establish empirical robustness.

这句话仅在确实纳入该附录时添加。若不纳入附录，正文无需添加引用。

## 六张图如何排列

| 附录位置 | 编号 | 现有文件 | 作用 |
|---|---|---|---|
| A.3 Baby 段落之后 | Figure A1 | `baby_absolute.png` / `.svg` | 同时看干净起点和扰动后的绝对表现 |
| 紧接 Figure A1 | Figure A2 | `baby_relative.png` / `.svg` | 看各方法相对自身起点的下降 |
| A.4 Sports 段落之后 | Figure A3 | `sports_absolute.png` / `.svg` | 检查高绝对分数与抗扰动幅度是否一致 |
| 紧接 Figure A3 | Figure A4 | `sports_relative.png` / `.svg` | 对照图像错配条件下的下降排序 |
| A.5 Clothing 段落之后 | Figure A5 | `clothing_absolute.png` / `.svg` | 展示绝对排名可能维持的情况 |
| 紧接 Figure A5 | Figure A6 | `clothing_relative.png` / `.svg` | 展示文本错配下 Recall 与 NDCG 的不同关系 |

图片均来自 `learning_examples/paper_anchored_noise/`。每张已经包含 2×3 面板，不建议再把六张压缩成一张总图。使用横跨正文宽度的排版；每个数据集的绝对图和相对图应相邻。Word 优先插入 SVG 或原尺寸 PNG，避免截图粘贴。

## 正文哪些句子需要调整

### 1. Introduction 的第 4 条贡献

原文前两句：

> 4) We conduct comprehensive experiments on three real-world public benchmark datasets. Experimental results demonstrate that AMORE outperforms existing state-of-the-art baselines in both recommendation accuracy and model robustness.

建议将该条贡献整段替换为上一轮修改清单的 E05：

> 4) We compare AMORE with twelve baselines on Baby, Sports, and Clothing and report both module-removal and affinity-use factorial results. AMORE ranks first in 11 of the 12 reported dataset–metric comparisons. The factorial interaction is small and varies in sign across datasets, so we do not claim a uniform synergistic effect.

不要因为加入当前情景附录而恢复 `model robustness` 已被证明的说法。

### 2. 4.2 Overall Performance 的 Third 段

原段开头：

> Third, the combination of alignment and decoupling is more robust than either strategy alone.

该段仍按上一轮 E14 整段替换：

> Third, the Baby results show a trade-off across metrics. AMORE achieves Recall@20 of 0.1040, compared with 0.1048 for MENTOR, a relative decrease of 0.76%. AMORE nevertheless has higher Recall@10, NDCG@10, and NDCG@20, with relative gains of 0.59%, 1.10%, and 1.11%. This mixed pattern supports competitive ranking performance rather than uniform superiority. The reported aggregate results do not establish statistical significance or explain the internal behavior of MENTOR.

### 3. Conclusion 的末尾

如果需要明确实验边界，可在已修改的结论第一段后追加：

> Robustness to controlled interaction corruption and modality mismatch remains to be established through matched retraining experiments. The illustrative scenarios in Appendix A clarify the reporting perspective but do not replace such measurements.

若未纳入情景附录，只保留第一句。训练目标中使用特征扰动属于方法设计，不能等同于完成了受控鲁棒性评估。

## 对 R3.6 的衔接回复

> We agree that a robustness claim requires controlled corruption experiments. The supplementary scenario illustrations distinguish absolute ranking performance from degradation relative to a clean reference, but their nonzero-corruption values are constructed and are not presented as experimental validation. We therefore do not regard these illustrations as resolving the requested robustness evaluation and have limited the corresponding empirical claims.

## 后续换成真实噪声结果时的正文安排

以下是布局建议，当前不写成已完成事实：

- 在现有 4.5 之后、结论之前新增 **4.6 Robustness to Interaction and Modality Corruption (RQ5)**。
- 正文放一张完整概览图：三行分别为 Baby、Sports、Clothing，三列分别为交互注入、图像错配、文本错配，展示真实 Recall@20 及运行变异。这样所有数据集和干预都得到覆盖，不只挑表现有利的一组。
- 附录放真实数据的 NDCG@20、相对下降曲线及完整运行设置。现在六图的绝对值/相对值布局可以复用。
- 方法段说明注入比例的分母、特征错配方法、是否重新训练、干净验证/测试标签、相同干预实例和种子、检查点选择方式。
- 结果段分别报告绝对排名与自身退化，并保留失败条件；具体数字和结论待真实结果提供后填写。不能仅将本附录的 `illustrative` 删除来转换其性质。

本次只编写放置指南和附录文字，不重新生成或改写原目录图片。
