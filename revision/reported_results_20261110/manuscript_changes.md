# 基于本次真实汇总值的逐句修改清单

定位原文：`papers/AMORE.docx`。下面的原句逐项从该文件提取并校验；标题与原句可用于 Word 搜索。此清单提供替换建议，尚未直接写入原始 DOCX。

图三已按作者确认的 Recall@10、Baby／Sports／Clothing 顺序整理；重复次数和评估划分尚未提供。当前不添加误差线、显著性或未经提供的机制控制。

## 先替换表格

1. **表 2 整表替换**：本次有 12 个基线，加 AMORE 共 13 个模型；原文无 DHMRec 行。使用 `data/performance.csv` 或 `tables/table2_performance.tex`。MENTOR 为所有列的最强基线，12 个 Improv. 百分比均复算一致。

2. **表 3 整表替换**：使用 `data/ablation.csv` 或 `tables/table3_ablation.tex`。下面列出相对原始 Word 的全部数值差异，避免只改正文、漏改表格。

3. **新增亲和力四组合表**：使用 `tables/table4_affinity.tex`。图三为 Recall@10；普通消融为 Recall@20 / NDCG@20。不能将两张表的行互换。

4. 输入校核：总体表文字中的 Sports/MMGCN/NDCG@20 为 **0.0254**，截图看起来为 **0.0612**。本次采用文字值 0.0254，请修正截图/Word 对应格或提供原始记录确认。

| 表格 | 方法/变体 | 数据集 | 指标 | 原 Word | 本次值 |
|---|---|---|---|---:|---:|
| Table 2 | LightGCN | Sports | NDCG@10 | 0.0313 | 0.0311 |
| Table 2 | VBPR | Clothing | Recall@10 | 0.0432 | 0.0281 |
| Table 2 | VBPR | Clothing | Recall@20 | 0.0663 | 0.0415 |
| Table 2 | VBPR | Clothing | NDCG@10 | 0.0233 | 0.0158 |
| Table 2 | VBPR | Clothing | NDCG@20 | 0.0284 | 0.0192 |
| Table 2 | MMGCN | Clothing | Recall@10 | 0.0378 | 0.0218 |
| Table 2 | MMGCN | Clothing | Recall@20 | 0.0615 | 0.0345 |
| Table 2 | MMGCN | Clothing | NDCG@10 | 0.0200 | 0.0110 |
| Table 2 | MMGCN | Clothing | NDCG@20 | 0.0261 | 0.0142 |
| Table 2 | SLMRec | Baby | Recall@10 | 0.0540 | 0.0529 |
| Table 2 | SLMRec | Baby | Recall@20 | 0.0810 | 0.0775 |
| Table 2 | SLMRec | Baby | NDCG@10 | 0.0285 | 0.0290 |
| Table 2 | SLMRec | Baby | NDCG@20 | 0.0357 | 0.0353 |
| Table 2 | SLMRec | Sports | Recall@10 | 0.0676 | 0.0663 |
| Table 2 | SLMRec | Sports | Recall@20 | 0.1017 | 0.0990 |
| Table 2 | SLMRec | Sports | NDCG@10 | 0.0374 | 0.0365 |
| Table 2 | SLMRec | Sports | NDCG@20 | 0.0462 | 0.0450 |
| Table 2 | SLMRec | Clothing | Recall@10 | 0.0540 | 0.0452 |
| Table 2 | SLMRec | Clothing | Recall@20 | 0.0810 | 0.0675 |
| Table 2 | SLMRec | Clothing | NDCG@10 | 0.0285 | 0.0247 |
| Table 2 | SLMRec | Clothing | NDCG@20 | 0.0357 | 0.0303 |
| Table 2 | BM3 | Sports | NDCG@10 | 0.0335 | 0.0355 |
| Table 3 | w/o ELAM | Baby | Recall@20 | 0.1038 | 0.1029 |
| Table 3 | w/o ELAM | Sports | Recall@20 | 0.1140 | 0.1132 |
| Table 3 | w/o ELAM | Baby | NDCG@20 | 0.0452 | 0.0432 |
| Table 3 | w/o ELAM | Sports | NDCG@20 | 0.0520 | 0.0516 |
| Table 3 | w/o ELAM | Clothing | NDCG@20 | 0.0448 | 0.0438 |
| Table 3 | w/o CFDM | Baby | NDCG@20 | 0.0445 | 0.0441 |
| Table 3 | w/o CFDM | Clothing | NDCG@20 | 0.0441 | 0.0439 |
| Table 3 | w/o MACA | Clothing | NDCG@20 | 0.0400 | 0.0401 |

## 逐句或逐段替换

### E01 Abstract，最后一句

**原文**

> Extensive experiments on three real-world benchmark datasets demonstrate that AMORE consistently outperforms strong baselines, achieving improvements of up to 2.12% on Recall@20 and 2.72% on NDCG@20 over the strongest competing method.

**替换为**

On the three Amazon datasets, AMORE obtains the highest reported score in 11 of the 12 dataset–metric comparisons. Relative to MENTOR, Recall@20 changes by −0.76%, +0.53%, and +2.12%, while NDCG@20 improves by 1.11%, 2.35%, and 2.72% on Baby, Sports, and Clothing, respectively.

**理由**：删除 consistently outperforms；保留 Baby Recall@20 的负结果。最高分只是当前表格中的点估计。

### E02 Introduction，贡献 1

**原文**

> 1) We propose AMORE, a novel multimodal recommendation framework. Through the proposed TAAI paradigm, AMORE effectively establishes a bridge between the structural confidence of graph edges and the semantic decoupling of node features, significantly improving recommendation quality.

**替换为**

1) We propose AMORE, a multimodal recommendation framework that uses affinity information in graph propagation and modality fusion. We evaluate the contributions of these two uses separately and jointly on a common backbone.

**理由**：将显著提升及机制成立的预设，改成设计内容与实际提供的比较。

### E03 Introduction，贡献 2

**原文**

> 2) We design an edge-level modulation mechanism coupled with an affinity-guided coarse-grained weight generation scheme. By leveraging affinity information to filter noise during graph propagation and feeding it into a learnable network to produce dynamic fusion weights, AMORE achieves more precise extraction of modality-shared features.

**替换为**

2) We introduce affinity-based propagation modulation and affinity-guided fusion within the same framework. A factorial comparison separates their individual gains from the interaction contrast, allowing the usefulness of each component and their joint behavior to be examined separately.

**理由**：本次结果能支持比较个别使用位置，不能证明共享语义提取更精确。

### E04 Introduction，贡献 3

**原文**

> 3) We introduce a residual-enhanced fine-grained decoupling module integrated with a multi-objective alignment strategy. This enables personalized fine-grained feature fusion and, combined with Adaptive BPR loss and multiple alignment losses, eliminates cross-modal semantic gaps while maximally preserving genuine user interaction signals.

**替换为**

3) We combine coarse-to-fine feature fusion with a multi-objective alignment component. Module-removal experiments quantify their aggregate contributions to ranking performance; the contribution of each individual alignment loss requires a separate comparison.

**理由**：删除 eliminates/maximally 等绝对化结论；整块 MACA 消融不区分矩约束与 InfoNCE。Adaptive BPR 是否实际启用应另按运行代码核实。

### E05 Introduction，贡献 4

**原文**

> 4) We conduct comprehensive experiments on three real-world public benchmark datasets. Experimental results demonstrate that AMORE outperforms existing state-of-the-art baselines in both recommendation accuracy and model robustness. In particular, the ablation study, threshold sensitivity analysis, and feature visualization jointly support the hypothesis that edge-level structural affinity is a useful signal for denoising graph propagation and guiding multimodal feature decoupling.

**替换为**

4) We compare AMORE with twelve baselines on Baby, Sports, and Clothing and report both module-removal and affinity-use factorial results. AMORE ranks first in 11 of the 12 reported dataset–metric comparisons. The factorial interaction is small and varies in sign across datasets, so we do not claim a uniform synergistic effect.

**理由**：三张表不包含受控噪声实验，因此不写 model robustness 已被证明。

### E06 4.1.3 Baselines，开头

**原文**

> To demonstrate the superiority of AMORE, we select eleven representative baseline models, which are grouped as follows: non-multimodal CF/graph recommenders, classical graph-based multimodal recommenders, and SSL-based or decoupling-oriented multimodal recommenders.

**替换为**

We compare AMORE with twelve representative baselines, grouped into non-multimodal collaborative-filtering and graph recommenders, classical graph-based multimodal recommenders, and self-supervised or decoupling-oriented multimodal recommenders.

**理由**：加入 DHMRec 后共有 12 个基线，加 AMORE 共 13 行模型。

### E07 4.1.3 Baselines，DualGNN 年份

**原文**

> DualGNN (2021) [9]: Constructs a user co-occurrence graph additionally to aggregate neighbor modality information for mining shared user preferences.

**替换为**

DualGNN (2023) [9]: Constructs a user co-occurrence graph additionally to aggregate neighbor modality information for mining shared user preferences.

**理由**：与本次表格及论文已有参考文献 [9] 的期刊年份统一。

### E08 4.1.3 Baselines，SLMRec 年份

**原文**

> SLMRec (2022) [11]: Pioneers the introduction of SSL into multimodal graph recommendation, proposing three data augmentation methods including feature noise perturbation.

**替换为**

SLMRec (2023) [11]: Pioneers the introduction of SSL into multimodal graph recommendation, proposing three data augmentation methods including feature noise perturbation.

**理由**：与本次表格及论文已有参考文献 [11] 的期刊年份统一。

### E09 4.1.3 Baselines，末段

**原文**

> AMORE is compared with them to verify whether topology-aware affinity modulation provides additional gains beyond self-supervised alignment alone.

**替换为**

The overall comparison evaluates ranking performance against these baselines. The contribution of affinity use is examined separately through module-removal and factorial comparisons within AMORE.

**理由**：整体性能混合了多个结构差别，不能直接隔离亲和力贡献。

### E10 4.2 Overall Performance，表 2 前

**原文**

> Table 2 presents the comprehensive performance comparison between AMORE and all baseline models across the three datasets. The best results are highlighted in bold, and the second-best results are underlined. “Improv.” denotes the relative improvement of AMORE over the best-performing baseline.

**替换为**

Table 2 reports Recall@10, Recall@20, NDCG@10, and NDCG@20 on Baby, Sports, and Clothing. The highest and second-highest reported scores in each column are shown in bold and underlined, respectively. “Improv.” is calculated as 100 × (AMORE − best baseline) / best baseline; negative values indicate that AMORE performs worse than the strongest baseline.

**理由**：明确分母和负值含义；尚未给出种子数量，因此暂不写 mean ± SD。

### E11 4.2，表 2 后引导句

**原文**

> Based on Table 2, the results can be interpreted along three connected observations rather than as isolated numerical comparisons.

**替换为**

The results provide three observations about ranking performance and the scope of the comparison.

**理由**：删除预先建立机制解释的表述。

### E12 4.2，First 段

**原文**

> First, AMORE achieves leading SOTA performance on the Sports and Clothing datasets and remains highly competitive on Baby. On the Clothing dataset, the sparsest setting with the most severe long-tail effect, AMORE improves over the strongest competing method MENTOR by 2.12% on Recall@20 and 2.72% on NDCG@20. This indicates that the proposed TAAI paradigm is particularly effective when reliable collaborative signals are scarce and noisy multimodal propagation is more harmful.

**替换为**

First, AMORE obtains the highest reported score on all four metrics for Sports and Clothing and on three of the four metrics for Baby. MENTOR is the strongest competing baseline in every column. On Clothing, AMORE improves Recall@10, Recall@20, NDCG@10, and NDCG@20 by 3.74%, 2.12%, 3.33%, and 2.72%, respectively. These aggregate improvements do not by themselves establish a specific benefit for long-tail items or cold-start cases, which require separate group-level evaluations.

**理由**：整体 Clothing 的提升不能作为长尾/冷启动验证。

### E13 4.2，Second 段

**原文**

> Second, AMORE alleviates the bottleneck of static graph aggregation. Traditional graph-based multimodal models such as MMGCN and LATTICE propagate multimodal information over graph structures with limited edge-level discrimination, so incidental interactions can contaminate user representations. The ELAM module proposed in AMORE reduces this risk by suppressing multimodal feature propagation on low-affinity edges while preserving the ID pathway, which explains the consistent gains over graph-fusion baselines.

**替换为**

Second, the advantage over the strongest baseline is modest and metric-dependent. On Sports, the relative improvements are 0.92% and 0.53% for Recall@10 and Recall@20, and 2.44% and 2.35% for NDCG@10 and NDCG@20. These comparisons quantify overall ranking differences; they do not isolate whether the gains originate from propagation modulation, feature fusion, or alignment objectives.

**理由**：删除通过总体表证明减少噪声传播的因果推断。

### E14 4.2，Third 段

**原文**

> Third, the combination of alignment and decoupling is more robust than either strategy alone. Although MENTOR is slightly better for Recall@20 on Baby dataset (MENTOR’s 0.1048 vs. AMORE’s 0.1040), AMORE surpasses MENTOR on the same dataset in terms of NDCG metrics (which demand more precise ranking). This suggests that MENTOR’s aggressive cross-modal alignment, while bringing features closer together, may inadvertently disrupt fine-grained personalized semantics. In contrast, AMORE’s residual-enhanced AGSD module not only achieves macro-level distribution alignment but also preserves users’ micro-level personalized preferences.

**替换为**

Third, the Baby results show a trade-off across metrics. AMORE achieves Recall@20 of 0.1040, compared with 0.1048 for MENTOR, a relative decrease of 0.76%. AMORE nevertheless has higher Recall@10, NDCG@10, and NDCG@20, with relative gains of 0.59%, 1.10%, and 1.11%. This mixed pattern supports competitive ranking performance rather than uniform superiority. The reported aggregate results do not establish statistical significance or explain the internal behavior of MENTOR.

**理由**：删除关于 MENTOR 过强对齐破坏偏好及 AGSD 保留微观语义的猜测。

### E15 Ablation Study (RQ2)，开头

**原文**

> To validate the necessity of each core module in AMORE, we design three ablation variants that remove one mechanism at a time while keeping the remaining training and evaluation settings unchanged.

**替换为**

Table 3 compares the full AMORE model with three module-removal variants to assess their contributions to Recall@20 and NDCG@20. We distinguish these module-level comparisons from the subsequent factorial study, which changes only where affinity is used within the same backbone.

**理由**：区分整块模块消融和亲和力使用位置；避免以 necessity 预设结论。

### E16 Ablation Study，w/o ELAM 定义

**原文**

> The w/o ELAM (Edge-Level Affinity Modulation) removes edge-level affinity hard modulation, so graph convolution degenerates to uniform aggregation over all interaction edges. This variant tests whether structural affinity is useful for denoising user-item message passing.

**替换为**

The w/o ELAM variant removes the affinity-based propagation modulation component. This comparison evaluates the ranking effect of this module; a decrease in ranking performance alone does not establish that the removed component identifies erroneous interactions.

**理由**：uniform aggregation 不一定符合归一化 GCN；表格也不能证明移除后噪声扩散。

**采用前核对**：先核对服务器实际消融实现。若 use_affinity_modulation 同时关闭传播与融合先验，此变体不能称为只移除传播调制，必须据实际开关重写。

### E17 Ablation Study，w/o CFDM 定义

**原文**

> The w/o CFDM (Coarse-to-Fine Decoupling Module) removes affinity-guided weight generation and residual-enhanced fine-grained fusion, replacing them with simple multimodal concatenation. This variant evaluates whether topology-aware coarse-to-fine decoupling is necessary for separating commonality from personalized residual semantics.

**替换为**

The w/o CFDM variant removes the coarse-to-fine fusion module. Its result measures the contribution of this module as a whole. This intervention differs from the fusion-affinity switch in the factorial study, where the fusion architecture is retained and only its use of affinity is changed.

**理由**：普通消融移除 CFDM，不能与图三的关闭融合亲和力等同。具体替代运算不能从分数猜测。

**采用前核对**：与实际消融代码核对：需要补充模块移除后究竟使用拼接、求和还是其他替代运算。

### E18 Ablation Study，w/o MACA 定义

**原文**

> The w/o MACA (Modality-Adaptive Contrastive Alignment) removes the multi-level InfoNCE loss and Gaussian distribution alignment loss, leaving optimization mainly to the recommendation objective. This variant measures whether modality-adaptive alignment is required after ELAM and CFDM have produced the representations.

**替换为**

The w/o MACA variant removes the combined alignment component. This aggregate ablation assesses the contribution of the removed objectives together and does not distinguish the effects of contrastive learning, distribution matching, or their interaction.

**理由**：不能把整块移除结果写成每种辅助损失分别有效。

**采用前核对**：需逐项确认被置零的损失系数及仍保留的正则项；未核实前不要写成纯 BPR。

### E19 Ablation Study，结果引导段

**原文**

> Experimental Results are presented in Table 3. Removing any core module leads to a clear performance decline, but the reasons differ across modules:

**替换为**

The full model has the highest reported score among the four variants for both metrics on all three datasets. We report relative decreases as 100 × (Full − ablation) / Full. The magnitude of these decreases differs across modules and metrics.

**理由**：将 clear decline 替换成明确的点估计关系及降幅定义。

### E20 Ablation Study，ELAM 分析

**原文**

> Without ELAM, noisy multimodal signals diffuse freely through user-item edges, so user representations absorb incidental interactions that should have contributed only ID-level collaborative evidence. The resulting performance drop supports the core hypothesis that edge-level structural affinity is not merely a pruning heuristic but a useful confidence signal for user-interest modeling.

**替换为**

Removing ELAM reduces Recall@20 by 1.06%, 1.14%, and 1.78% on Baby, Sports, and Clothing, respectively. The corresponding NDCG@20 decreases are 5.05%, 1.34%, and 3.31%. The ranking impact is therefore metric-dependent, with a larger relative NDCG@20 decrease on Baby. These results support a contribution of the ELAM-containing configuration to the reported ranking scores, but do not by themselves verify an edge-denoising mechanism.

**理由**：使用本次最新 ELAM 数值；Baby 的 NDCG 下降比 Recall 下降明显。

### E21 Ablation Study，CFDM 分析

**原文**

> Without CFDM, the model cannot distinguish modality commonality from personalized specificity. Naive concatenation mixes shared attributes such as category or color with user-specific residual preferences such as texture or logo style, which is especially harmful for long-tail items whose few interactions provide limited supervision.

**替换为**

Removing CFDM decreases Recall@20 by 3.08%, 2.88%, and 4.16% on Baby, Sports, and Clothing, respectively, while NDCG@20 decreases by 3.08%, 3.44%, and 3.09%. CFDM removal has a larger Recall@20 effect than ELAM removal on every dataset, whereas ELAM removal has the larger NDCG@20 effect on Baby and Clothing. Thus, the relative contributions of the two modules depend on the ranking metric.

**理由**：避免一概说 CFDM 比 ELAM 更重要；不再用颜色/纹理/长尾机制作未经验证的解释。

### E22 Ablation Study，MACA 分析

**原文**

> Without MACA, the model loses the self-supervised constraint that coordinates ID, multimodal, and decoupled representations. The performance degradation shows that AMORE needs alignment to stabilize the heterogeneous feature spaces, but this alignment is effective because ELAM and CFDM first reduce noise and preserve fine-grained residual information.

**替换为**

Removing MACA produces the largest decline among the three module-removal variants on every dataset and metric. Recall@20 decreases by 13.17%, 11.18%, and 12.57%, and NDCG@20 decreases by 14.07%, 7.84%, and 11.48% on Baby, Sports, and Clothing, respectively. This identifies the combined alignment component as the most influential removal in this comparison. Additional loss-level ablations are needed to determine which objectives account for the effect.

**理由**：只能比较这组移除项的影响，不能推导 MACA 单项损失或前两模块为其生效原因。

### E23 5 Conclusion and Future Work，第一段

**原文**

> In this paper, we propose adaptive modulation and self-supervised enhancement (AMORE) to mitigate graph-noise amplification and feature entanglement in multimodal recommendation. AMORE unifies graph-topology modeling and semantic feature decoupling through Topology-Aware Adaptive Integration (TAAI). Edge-Level Affinity Modulation (ELAM) suppresses low-confidence multimodal propagation while preserving identifier (ID)-based collaborative connectivity. The Coarse-to-Fine Decoupling Module (CFDM) uses affinity-guided fusion to separate modality commonality from personalized residual semantics, and Modality-Adaptive Contrastive Alignment (MACA) reduces cross-modal statistical mismatch without overwhelming historical interaction signals. Experiments on three benchmark datasets demonstrate the effectiveness of AMORE.

**替换为**

We presented AMORE, a multimodal recommendation framework combining affinity-based propagation modulation, coarse-to-fine fusion, and a multi-objective alignment component. On Baby, Sports, and Clothing, AMORE obtains the highest reported score in 11 of the 12 dataset–metric comparisons, with an exception for Recall@20 on Baby. Module-removal experiments show the largest performance decrease when the combined MACA component is removed. The affinity-use factorial comparison shows positive joint gains over the no-affinity reference, but small, mixed-sign interaction contrasts. These results support retaining the tested components while limiting claims about uniform synergy, verified denoising, and generalization beyond the evaluated Amazon categories.

**理由**：结论应同时容纳总体收益、MACA 整体贡献以及小且异号的交互项。

## 新增基线说明

在 4.1.3 的解耦类模型组中添加（参考文献 [23] 已存在于原稿）：

DHMRec (2026) [23]: A collaboration-guided multimodal recommendation method that disentangles modality-common and modality-specific information and integrates these representations through hierarchical fusion.

## 新增 R2.1 实验小节

插入位置：Table 3 及其分析之后、原 4.4 Hyper-parameter Sensitivity 之前。建议作为 4.3 的子节；新表可编号 Table 4。图用插入后的实际顺序编号，并同步修改现有 Figure 2–5 的交叉引用。图三的 10 表示仅传播调制，01 表示仅融合调制。

### Affinity use in propagation and fusion

To examine the two uses of affinity separately, we compare four configurations on the same backbone: 00 disables affinity use in both propagation and fusion, 10 enables propagation modulation only, 01 enables affinity-guided fusion only, and 11 enables both. These controls differ from removing an entire module in Table 3. Let y_ab denote Recall@10 for configuration ab. We compute the interaction contrast as Δ_int = y_11 − y_10 − y_01 + y_00, which measures the departure of the joint change from the sum of the two individual changes on the additive Recall@10 scale.

On Baby, the four scores are 0.0671, 0.0673, 0.0679, and 0.0682, respectively. The corresponding values are 0.0758, 0.0762, 0.0768, and 0.0770 on Sports, and 0.0677, 0.0685, 0.0684, and 0.0693 on Clothing. The joint gains y_11 − y_00 are +0.0011, +0.0012, and +0.0016, while the interaction contrasts are +0.0001, −0.0002, and +0.0001. The full configuration has the highest reported score on each dataset, but the interaction is small and not uniformly positive. On Sports, the joint gain is smaller than the sum of the two individual gains on this metric scale; this does not imply that the full configuration is worse than either single-use configuration.

The contrasts are calculated from values displayed to four decimal places. If these values were rounded to the nearest 0.0001, rounding alone could change each interaction contrast by up to 0.0002; this is a precision bound, not a statistical confidence interval. The present aggregate values therefore do not establish a reliable positive interaction. Moreover, this switch comparison does not isolate whether sharing the same prior is better than using separate priors. Such a mechanism claim requires additional controls, including independent, shuffled, or constant priors and seed-level uncertainty estimates.


## 对照方式与配对公式

若后续提供相同种子 s 下的四个分数，应先计算 d_s = y11,s − y10,s − y01,s + y00,s，再汇总 d_s 的均值、样本 SD 和区间。若四格已经是同一组种子的均值，其线性对比等于种子交互项的均值，但单靠四个均值无法恢复配对方差；目前也不知道这四格是否如此汇总。不要直接称本次结果为“配对交互效应检验”。

## 图表取舍

- 主文保留 Table 2；可加 `02_relative_improvement` 概括全部 12 项差异，`01_overall_comparison` 可放补充材料。
- Table 3 配 `04_module_drop` 展示模块贡献差异；`03_module_ablation` 展示原始绝对值，二者任选一个进主文，减少重复。
- R2.1 用 `05_affinity_factorial` 和新增四组合表；`06_affinity_effects` 可放回复信或补充材料。
- 图均有 PNG、SVG、PDF；图注见 `figure_captions.md`。

## 本轮文字仍需的数据与实现说明

在 4.1.4 补充实际评估划分、种子数量、每行是单次还是均值、各基线运行来源与相同协议的具体记录。这些信息未提供，不填入猜测。原稿的损失公式、Adaptive BPR、接收者相关的边调制、残差系数是否实际生效，应继续与对应运行版本代码核对；三张分数表不能验证这些实现。原稿敏感性与 t-SNE 小节也不能因本次更新而自动视为获得了新的实验支持。

当前可保留完整模型作为论文主模型；这批点估计没有要求增加新模块。优先核对实验定义与精度，收紧机制表述，而非据小交互项继续堆叠结构。
