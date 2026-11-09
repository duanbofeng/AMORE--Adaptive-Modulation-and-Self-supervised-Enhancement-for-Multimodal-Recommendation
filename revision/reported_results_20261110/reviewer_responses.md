# 审稿回复建议

以下为完成对应正文替换、表格更新和图表插入后可使用的英文回复。它们不声称已补做多种子检验或共享先验控制。图三按作者确认的 Recall@10 整理。

## R1.1 总体提升较小且不一致

Thank you for pointing out that the gains are modest and not uniform. We have revised the performance discussion to report the exception explicitly. AMORE obtains the highest reported score in 11 of the 12 dataset–metric comparisons, but its Recall@20 on Baby is 0.1040 versus 0.1048 for MENTOR, a relative decrease of 0.76%. The relative Recall@20 changes on Sports and Clothing are +0.53% and +2.12%, respectively. We have replaced the claim of consistent superiority with a dataset- and metric-specific description. The current aggregate values do not establish statistical significance, so we make no such claim. Repeated-run uncertainty remains necessary to assess the reliability of these differences.

## R1.2 与 R3.4 模块消融的解释

We have updated the module-ablation table and report both absolute scores and relative decreases. Removing ELAM reduces Recall@20 by 1.06%, 1.14%, and 1.78%, and NDCG@20 by 5.05%, 1.34%, and 3.31% on Baby, Sports, and Clothing, respectively. MACA removal yields the largest decrease among the three module removals for every dataset and metric. We have narrowed the interpretation to module-level ranking contributions. These comparisons do not identify erroneous edges, establish a denoising mechanism, or isolate the individual objectives inside MACA. We also distinguish whole-module removal from disabling a single use of affinity in the factorial comparison.

## R2.1 传播与融合是否因共享亲和力而产生收益

Thank you for this observation. We distinguish the module-removal study from a 2 × 2 comparison of affinity use in propagation and fusion on the same backbone. Configuration 00 disables both uses, 10 enables propagation modulation only, 01 enables affinity-guided fusion only, and 11 enables both. We report the four Recall@10 scores and the interaction contrast Δ_int = y_11 − y_10 − y_01 + y_00 in addition to the ordinary ablations.

The joint gains over 00 are +0.0011, +0.0012, and +0.0016 on Baby, Sports, and Clothing. The corresponding interaction contrasts, computed from the displayed scores, are +0.0001, −0.0002, and +0.0001. Although configuration 11 has the highest reported Recall@10 in each dataset, the interaction contrasts are small and vary in sign. We therefore do not interpret the full model's higher score as evidence of consistently positive synergy.

We also acknowledge two limits of this evidence. First, the scores are displayed to four decimal places, and seed-level records are not included in the present summary. If rounded to the nearest 0.0001, these values imply an interaction rounding bound of up to 0.0002, so the small positive contrasts require higher-precision and repeated-run analysis. Second, toggling the two uses of affinity does not establish that sharing the same prior is better than using independent priors. Independent-prior, shuffled-prior, or constant-prior controls are not included in the supplied results. Accordingly, we have limited the claim to the observed contribution of affinity use and explicitly leave the shared-prior mechanism unconfirmed.

## R3.2 MACA 内各损失的独立贡献

The updated module ablation removes MACA as a combined component. We have clarified that this result supports the aggregate contribution of the removed objectives but cannot separate the contributions of contrastive learning and moment matching. We do not present this whole-component ablation as a substitute for a loss-level factorial comparison.

## 尚不能用这三张表解决的意见

- R1.1 / R3.3：未提供逐种子分数，不能补写 SD、置信区间、p 值或显著性星号。
- R1.3 / R2.3：新增 DHMRec 不等于补齐所有审稿人指定基线；统一划分、特征、候选集及调参预算尚需记录支持。
- R2.1：已有四组合点估计，但共享先验与独立先验、打乱先验、常数先验的机制对照没有提供。
- R3.2：移除 MACA 不是矩约束与 InfoNCE 的独立消融。
- R2.4 / R3.5 / R3.6 / R1.5 / R3.7：长尾/冷启动、敏感性、噪声、效率仍需各自真实记录。
- R1.4：这三类数据仍均来自 Amazon，不支持跨平台、视频或音乐泛化。
