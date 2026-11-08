# 原 AMORE 第一轮结果：保留完整主干，逐项检查

本轮分析直接读取 `experiment_reports/31_original_baby.json` 的完整配置和 history。四个运行均为 Baby、seed=999、validation Recall@20 选 checkpoint，test 尚未评估；数据与模型源码哈希一致，也与当前本地源码哈希一致。

## 与之前的 GUME 比较

GUME 参照取最近一次融合实验的 0.101644 / 0.044344，来源 `experiment_reports/11_fusion_baby_reported.json`。该 GUME 只有终端表转录，缺少原始配置、数据哈希和曲线，以下是暂时对照，不能宣称已完成同版本严格复现或显著性检验。

| 版本 | Recall@20 | 相对旧 GUME | NDCG@20 | 相对旧 GUME | 秒/训练 epoch |
|---|---:|---:|---:|---:|---:|
| GUME（上一轮） | 0.101644 | — | 0.044344 | — | 1.697 |
| original | 0.102092 | +0.441% | 0.044234 | −0.248% | 2.270 |
| fixed | 0.101843 | +0.196% | 0.043973 | −0.836% | 2.219 |
| compact | 0.097151 | −4.420% | 0.042574 | −3.991% | 1.302 |
| compact_no_affinity | 0.098237 | −3.352% | 0.043029 | −2.965% | 1.199 |

完整 AMORE 恢复到接近 GUME 的水平，但目前是 Recall 略高、NDCG 略低，没有一致优势。单 seed 只能提供筛选线索。上一轮 GUME 的机器负载和完整运行时间未核实，不能据此给出可靠的总耗时倍率。

## 哪些结论有依据

1. **保留 fixed 作为后续修改起点。** 相比 original，Recall −0.243%、NDCG −0.590%，可训练参数从 36,080,641 降到 33,591,681，减少 6.90%。主要删除的是原本无梯度的用户表；性能变化不能归因于这些表，而应考虑损失去重、实际 batch 分母、噪声采样等改变。不能因一个 seed 的小幅下降就恢复实现问题。
2. **暂停采用 compact。** 相比 fixed，Recall −4.607%、NDCG −3.181%，却只减少 4,289 个参数（0.0128%）。它省掉扩展传播和对比计算，单 epoch 更快，但没有显著减少模型容量。不能为了参数精简承受这种效果下降。
3. **亲和力在 compact 上的证据偏负面。** 关闭后 Recall +1.118%、NDCG +1.068%。这仅适用于这个结构、配置和 seed，不能直接推广到完整 AMORE，也不足以证明所有亲和力机制无用。
4. **暂时没有延长训练上限的理由。** original/fixed/compact/no_affinity 最佳轮次为 67/58/43/49，实际训练到 87/78/63/69，均按 patience=20 停止。compact 的训练 BPR 从最佳轮次的 0.0862 继续降到 0.0548，Recall 却从 0.097151 降到 0.095234；更像泛化损失，不能简单解释成“1000 轮还不够”。
5. **查噪声项，但不根据 loss 大小直接判定它错误。** fixed 最佳轮次的加权 BPR≈0.089、UM cross≈0.514、UM noise≈1.049。不同目标尺度不同，数值大不等于梯度主导，也不等于无用。需要单独关闭噪声项，保留 cross/扩展 BM/扩展正则，才能判断它是否必要。

本轮 wall time：original 451.0 秒、fixed 396.7 秒、compact 262.7 秒、no_affinity 278.2 秒。它包含启动、训练、验证等开销，不能与每 epoch 的训练时间混用。例如 no_affinity 每 epoch 更快，但多训练 6 轮，所以整个运行反而比 compact 更久。

## 下一轮只增加四次训练

新套件 `original_focus` 包含五个版本，其中已有 fixed 复用：

| 运行 | 保留什么 / 改什么 | 与谁比较 |
|---|---|---|
| gume | 当前原有 GUME 基线 | original、fixed；确认同代码版本参照 |
| amore_fixed | 原有完整修正版，不更改配置 | 已完成，resume 跳过 |
| amore_fixed_no_coarse | 仅关闭粗粒度融合中的亲和力网络，保留完整扩展视图及损失 | fixed：第二套融合网络是否必要 |
| amore_fixed_no_affinity | 在 no_coarse 基础上再关闭所有 UI 分支的亲和力调制 | no_coarse：传播调制是否必要 |
| amore_fixed_no_noise | 仅将 noise_loss 从 0.1 改为 0；保留完整其它项 | fixed：噪声对比是否必要 |

`fixed_no_affinity` 相比 fixed 同时移除了亲和力融合和传播，所以必须通过 no_coarse 中间对照解释。它也不等于 GUME：两者其它结构、图处理和默认损失系数仍有区别。

这次只增加 runner 的配置组合和文档，没有修改 `src/` 模型、损失或配置。已确认源码哈希与服务器导出的这轮结果一致。按原数据路径、seed、epochs、patience 和输出目录运行，可以复用原来的 fixed；如果你另外修改了源码或数据，resume 会据新的哈希安排新运行。

同步本次改动后，服务器项目根目录执行：

```bash
python experiments/run_suite.py --suite original_focus --datasets baby --seeds 999 --gpu 0 --output results/31_original_baby --execute --resume
```

汇总和导出：

```bash
python experiments/analyze_results.py results/31_original_baby --baseline gume --split validation --metric recall@20
python experiments/analyze_results.py results/31_original_baby --baseline gume --split validation --metric ndcg@20
python experiments/export_results.py results/31_original_baby --output experiment_reports/31_original_baby.json
```

TensorBoard 沿用 `results/31_original_baby`，勾选所需 run 对比。暂不加 `--test`，不运行旧 AMOREX/rebuild/gume_fusion 套件。

## 下一轮之后如何决策

- no_coarse 不降或更好：可删第二套亲和力融合网络，保留主干与完整损失；目前不做结论。
- no_affinity 进一步改善：亲和力在完整主干上也缺乏正面证据；用多个 seed 确认后，考虑从最终方法及论文主张中移除。
- no_noise 不降或更好：删除两项噪声对比任务；如果下降，则保留，随后最多测试一次较小权重，不能因为数值大就判错。
- 各自删除都有收益：组合仍须单独验证，不能把两个独立增益直接相加。
- 筛出一个候选后，再与 GUME、fixed 使用匹配的 999/2024/2025 三个 seed 复核；有效后再增加第二个数据集。最终锁定方案前不使用 test 决策。

新增三个配置已在本地合成数据检查前向、各项损失、反向、排序，以及关闭项确实生效。没有新增模型机制，没有声称后续结果已知。
