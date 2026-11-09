# 图注、讨论与审稿回复的学习示例

本文件是上述模拟场景的文字练习，数值来自本目录生成器。英文段落保留场景前提。

## 1. 总体性能

In this illustrative Baby scenario, the mean Recall@20 difference relative to the comparator is +0.000655 (+0.636% relative), with a paired 95% interval of [+0.000100, +0.001209]. The interval excludes zero, but the practical magnitude remains modest.

In this illustrative Sports scenario, the mean Recall@20 difference relative to the comparator is +0.000240 (+0.201% relative), with a paired 95% interval of [-0.000500, +0.000981]. This does not establish a reliable positive difference under this scenario.

In this illustrative Clothing scenario, the mean Recall@20 difference relative to the comparator is -0.000001 (-0.002% relative), with a paired 95% interval of [-0.000664, +0.000661]. This does not establish a reliable positive difference under this scenario.

## 2. 机制讨论写法

Within this simulated factorial example, the joint model performs favorably in one category, while the pattern weakens or reverses in other categories. A shared-signal interpretation would require the paired interaction contrast and informative-prior controls; the rank of the joint variant alone is insufficient.

## 3. 损失讨论写法

The illustrative objective ablations assign a stronger role to the aggregate contrastive objective than to moment matching. Some individual terms yield small or inconsistent effects. Such a pattern would support distinguishing essential and weakly supported components, rather than claiming that every term is necessary.

## 4. 噪声讨论写法

The simulated corruption curves show why absolute performance and degradation from the clean condition should be reported together. The text-mismatch scenario for Clothing intentionally includes a failure condition, preventing a universal robustness claim.

## 5. 回复审稿人的条件式写法

If a corresponding real experiment produced this pattern, a response could explain that matched-seed estimates, uncertainty intervals, and negative conditions have been added, and that claims have been narrowed to the settings supported by those measurements. The simulated examples themselves do not establish that any empirical reviewer concern has been resolved.

## 6. 英文图注

### 01_performance

Points show means and error bars show sample SD across five synthetic runs; small dots retain individual runs. Model and dataset names identify illustrative roles, not measured benchmark results.

### 02_branch_factorial

The same backbone is represented with graph and fusion switches. Error bars are sample SD. Stronger total performance does not by itself establish a positive interaction between branches.

### 03_loss_analysis

The heatmap reports relative changes after removing one loss. Blue means removal hurts performance; red means removal improves it. The factorial panel keeps both representation penalties fixed. Heatmap colors are effect sizes, not significance labels.

### 04_sensitivity

One coefficient changes at a time. Bands show sample SD. Zero and positive values are displayed at categorical spacing; the plotted high point is not a new configuration-selection recommendation.

### 05_robustness_baby

Lines summarize synthetic retraining scenarios under each corruption level; bands are sample SD. Lower degradation is preferable, but must be interpreted alongside the absolute clean score. Clothing deliberately includes a text-noise failure case.

### 05_robustness_sports

Lines summarize synthetic retraining scenarios under each corruption level; bands are sample SD. Lower degradation is preferable, but must be interpreted alongside the absolute clean score. Clothing deliberately includes a text-noise failure case.

### 05_robustness_clothing

Lines summarize synthetic retraining scenarios under each corruption level; bands are sample SD. Lower degradation is preferable, but must be interpreted alongside the absolute clean score. Clothing deliberately includes a text-noise failure case.

### 06_cohorts_baby

User activity groups form a weighted partition. Item groups use conditional group metrics and need not sum to the overall score. The cold-item panel is a separate zero-training-interaction scenario, not a synonym for warm long-tail items.

### 06_cohorts_sports

User activity groups form a weighted partition. Item groups use conditional group metrics and need not sum to the overall score. The cold-item panel is a separate zero-training-interaction scenario, not a synonym for warm long-tail items.

### 06_cohorts_clothing

User activity groups form a weighted partition. Item groups use conditional group metrics and need not sum to the overall score. The cold-item panel is a separate zero-training-interaction scenario, not a synonym for warm long-tail items.

### 07_efficiency

Timing and memory values are hypothetical assumptions, not hardware measurements. The proposed model is not designed to win both accuracy and efficiency in this example. Parameter counts are fixed across runs; timing variation is not parameter variation.

### 08_convergence

These are generated validation trajectories and objective traces, separate from the simulated holdout table. Bands show run SD; a lower training loss does not establish better generalization or faster convergence.

### 09_interpretation

Intervals are paired t intervals from five generated run differences. A positive mean with an interval crossing zero is inconclusive; lack of significance is not proof of equivalence. The scenario names describe intended generating settings, not guaranteed statistical outcomes.
