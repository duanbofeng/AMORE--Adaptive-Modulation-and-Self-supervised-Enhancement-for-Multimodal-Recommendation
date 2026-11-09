# Figure captions

## 01_overall_comparison

Recall@20 and NDCG@20 for five selected multimodal methods on Baby, Sports, and Clothing. Points show the reported values. The complete comparison includes twelve baselines in Table 2. No uncertainty estimates are available from the supplied aggregate results.

## 02_relative_improvement

Relative change of AMORE against the strongest baseline for each dataset and metric: 100 × (AMORE − baseline) / baseline. MENTOR is the strongest baseline in all twelve comparisons. The negative Baby Recall@20 entry is retained.

## 03_module_ablation

Module-removal results on the three datasets. The full model has the highest reported Recall@20 and NDCG@20 among the four variants. MACA removal produces the largest decrease in each comparison; this aggregate removal does not isolate its individual loss terms.

## 04_module_drop

Relative performance decrease after removing each module, calculated as 100 × (Full − ablation) / Full. The common color scale shows effect magnitude, not statistical significance. ELAM and CFDM have metric-dependent contributions.

## 05_affinity_factorial

Factorial comparison of affinity use in propagation and fusion on the same backbone. The full configuration (11) has the highest reported Recall@10, while the interaction contrast y11 − y10 − y01 + y00 is +0.0001, −0.0002, and +0.0001 on Baby, Sports, and Clothing, respectively. These are contrasts of the displayed point estimates, without seed-level uncertainty.

## 06_affinity_effects

Propagation-only (y10 − y00), fusion-only (y01 − y00), joint (y11 − y00), and interaction (y11 − y10 − y01 + y00) contrasts. All differences are in Recall@10 units scaled by 10³. Positive joint gains coexist with a negative interaction on Sports. Bars do not represent uncertainty or significance.
