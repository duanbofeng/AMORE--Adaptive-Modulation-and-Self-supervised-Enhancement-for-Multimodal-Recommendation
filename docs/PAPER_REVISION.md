# AMORE 论文修订工作稿

当前更新：用户停止 AMOREX 等方向，回到 old_model 保存的完整 AMORE。准确的版本对应、源码与论文不一致之处、修正版损失定义及执行顺序以 [ORIGINAL_AMORE.md](ORIGINAL_AMORE.md) 为准。以下旧草稿均不是本次确定的最终论文方法，不应混用。

2026-11-08 更新：当前正在检验独立框架 AMOREX，方法公式、负面初筛结果与对照协议见 [AMOREX_EXPERIMENTS.md](AMOREX_EXPERIMENTS.md)。下文第 3 节描述的是旧 `amore`，不能直接当作 `amorex` 的方法章节。新框架的多兴趣思路已有 MIND/ComiRec 等先例；尚未获得正式数据上的新结果，不能在摘要中宣称提升或首次提出多兴趣。

本稿用于组织下一轮方法与实验写作。当前还没有新模型的真实数据集结果，因此未改写原 `papers/AMORE.docx` 中的实验数值，也未将候选机制写成已经成立的最终模型。正式稿应在实验选择完成后统一修改题目、摘要、方法、算法和实验表，避免方法版本混用。

## 1 首先撤回或重新验证的表述

旧稿关于完整模型提升 2.12%/2.72%、所有模块都有效、残差增强、adaptive BPR 以及“亲和力代表拓扑可靠性”的主张，不能移植到新版本。它们需要与对应代码版本、配置、数据划分、seed 和原始日志逐一对齐。尤其旧 residual_alpha 的计算没有进入最终融合，不能用该开关的实验解释残差贡献。

“加入多种已有模块”本身不足以成为贡献。“去掉模块仍然涨点”是有价值的实验发现，但仍需解释何时发生、与强基线比较以及结果可复现性。

## 2 建议采用的研究问题

RQ1：在同一评估和调参预算下，GUME 的额外用户模态分支、分布对齐、粗细融合是否必要？

RQ2：内容一致性应施加到全局、物品还是用户历史边上？保持分数分布并打乱边对应关系后，效果如何变化？

RQ3：语义图的邻域一致性、行为共现与一次性用户初始化是否分别有用？它们与局部亲和力是互补还是替代关系？

RQ4：改动对稀疏用户、长尾物品、内容特征受损以及运行资源有什么影响？

这些问题不要求所有答案都是正面。若亲和力没有稳定收益，应从最终方法和题目中去掉它；可以在探索性实验中如实报告负面结果。

## 3 可作为方法章节起点的英文草稿

### Collaborative and semantic representations

We use a shared user embedding table and an item embedding table to model collaborative preferences. We propagate these embeddings on the training user–item graph using symmetric degree normalization and average the representations from the initial layer through the final layer. For each available content modality, an affine projection followed by a sigmoid produces a feature gate, which modulates the item ID embeddings. This behavior-guided feature construction follows the design line of MGCN and GUME. The gated item representations are propagated on a content-derived item graph and aggregated over each user's observed training history. The final representation adds a scaled semantic residual to the collaborative representation.

### Optional history-conditioned semantic weighting

We investigate whether a user's remaining interaction history can provide a useful signal for weighting an observed item's semantic contribution. For each training edge, we compare the item's normalized content features with the sum of the features of the user's other observed items. We exclude the current item from this history to avoid a direct self-similarity contribution. The resulting cosine agreement is transformed into a bounded attenuation weight, with shrinkage toward the unweighted model for short histories. We apply this weight only to the semantic aggregation path. We interpret it as a proxy for content consistency, rather than as a calibrated probability that an interaction is correct.

This hypothesis is evaluated against unweighted aggregation, a modality-wide constant, item-level averaging, and within-user permutation controls. These comparisons are necessary because a gain from attenuation alone would not establish the value of user-conditioned information.

### Optimization

The default model optimizes the standard BPR objective using the same inner-product score used for evaluation, with L2 regularization on the sampled initial ID embeddings. We evaluate an optional symmetric contrastive objective between collaborative and semantic representations as a separate ablation. Duplicate user and item IDs are removed within each contrastive batch so repeated observations are not treated as distinct negative identities.

以上段落描述当前可运行实现，但“optional”模块是否进入正式方法须由实验决定。实际论文只能保留最终被验证的路径；其它路径作为对照或补充实验。

## 4 与已有工作的关系

- 采用行为引导净化时引用 MGCN/GUME；使用粗细残差融合时不能宣称是首次提出。
- 图邻域一致性重加权引用 IIMRec。说明本实现只用空间内容图和训练共现图，归一化和门控与原文不同；不能借用其定理直接证明本适配版本。
- 初始化引用 SG-URInit。明确本实现是共享 ID 空间上的一次性适配，不是原文的完整模态初始化算法，也没有沿用其“语义保证”。
- 边级调制的背景引用 EgoGCN。区别应建立在实际消息作用位置、统计控制和新证据上。

如果最终只剩“GUME + 已有 NCER + 已有初始化”，更合适的贡献定位可能是有充分实验支持的精简与分析，而不是把组合包装成三个独立新模块。是否足以支撑目标期刊/会议，需要结合正式结果和更全面的相关工作判断。

## 5 主文与补充实验结构

| 部分 | 必需证据 | 输出来源 |
|---|---|---|
| 总体效果 | 匹配 seed 的均值±标准差，统一 validation 选择，独立调参强基线 | result.json、analysis_test_recall_20.json |
| 模块消融 | 单因素删减，累计删减单列；有用模块再做组合 | prune、mechanisms、factorial |
| 亲和力解释 | none/global/item/local/shuffle，语义尺度控制，权重分布与度数相关性 | core、semantic_scale、diagnostics |
| 稀疏与长尾 | 固定 train 度数分组、每组用户数、置信区间 | test_rankings.npz、cohort 分析 |
| 鲁棒性 | 相同训练保留比例或模态受损率，明确重新训练协议 | sparsity、corruption |
| 效率 | 可训练参数、总参数、训练 epoch 时间、预处理时间、GPU peak allocated | result.json |
| 收敛 | 同预算验证曲线、最佳 epoch；不要只比较 loss 的数值 | history.json、TensorBoard |
| 超参数 | 有限预先定义搜索范围，参数间交互按需分析 | requested.json、plan.json |

现有脚本没有复现官方 MGCN/IIMRec/SG-URInit 的完整系统。正式主表若包含它们，需要另行运行并对齐数据/评估协议；本实现的 inspired 变体不能冒充它们的官方基线。原文报告的数值只可作带协议差异说明的参考，不能和本地重跑数值混合做显著性检验。

## 6 结果驱动的写作分支

- local 稳定胜过 base/global/item/shuffle：可把“用户历史条件下的局部语义可靠性”作为待进一步验证的主线，仍需排除语义强度和图质量的解释。
- local 与 global 接近：主要收益可能是语义残差衰减，删掉过强的个性化机制叙事。
- shared/NCER 获益、local 无增益：围绕语义结构与行为一致性重写，降低亲和力地位。
- history 获益而 cluster 无增益：只保留一次性历史初始化，不增加聚类复杂度。
- 精简版和完整 GUME 相当：报告精度与资源之间的取舍，不强称准确率显著提升。
- 所有候选均无稳定优势：保留完整 GUME，记录负面结果，重新检查数据特征与研究问题。
