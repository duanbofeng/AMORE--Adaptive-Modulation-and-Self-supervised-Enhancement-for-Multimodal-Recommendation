# Baby 第一轮结果后的判断与下一步

**后续更新：用户已停止这些方向，当前请看 [原 AMORE 复跑与精简流程](ORIGINAL_AMORE.md)。以下仅保留第一轮分析，不执行 recover/efficient/content 套件。**

现在暂停亲和力超参数搜索。先确认完整 GUME 的优势来自哪里，再决定保留哪些有效信号、如何用更小的结构实现。第二轮代码提供了独立内容评分路线，不要求最终模型沿用 AMORE 的旧框架。

## 已有证据与解释边界

用户提供的是 Baby、validation、单 seed 汇总；原始数值保存在 `experiment_reports/01_core_baby_reported.json`。该文件明确标注为终端表格转录，不包含尚未取得的真实 seed、best_epoch 或完整配置。

| 版本 | Recall@20 | NDCG@20 | 单 epoch 秒 |
|---|---:|---:|---:|
| GUME | 0.101545 | 0.044377 | 1.717 |
| GUME lean | 0.087995 | 0.038392 | 0.812 |
| AMORE base | 0.089952 | 0.039522 | 0.678 |
| global | 0.090201 | 0.039207 | 0.687 |
| item | 0.089807 | 0.039228 | 0.695 |
| local | 0.089194 | 0.039167 | 0.696 |
| shuffle | 0.089524 | 0.039143 | 0.703 |

相对 AMORE base，local 的 Recall/NDCG 分别变化 −0.84%/−0.90%；global 的 Recall +0.28%，但 NDCG −0.80%。local 的 Recall 低于 shuffle，而 NDCG 仅略高。这不支持“当前局部亲和力对应关系带来有效增益”，但一个 seed 无法证明统计等价，也无法否定所有形式的亲和力。

AMORE base 相对 GUME 的 Recall/NDCG 分别低 11.42%/10.94%。主要缺口比亲和力变动大得多。单 epoch 约快 2.53 倍，但完整训练用时还取决于停止轮次、评估和预处理，不能直接说总训练快 2.53 倍。

上一轮的不足：GUME lean 同时改变冻结特征、损失、图增强和融合；AMORE 另外改变 II 深度、KNN 自环、投影参数化和正则位置。它们适合粗筛，不能归因到单个模块。没有消融证据前不应把被删除部分叫作“冗余”。

## 为什么当前亲和力可能无效

以下是可以证伪的假设，不是已有实验结论：

1. **内容相似性与偏好不是同一件事。** 用户购买互补物品时，它们可能与历史均值不相似。把低相似度当成低价值，可能压低真实的新增兴趣。
2. **单一历史均值抹平多兴趣。** 许多用户并不围绕一个内容中心消费。局部边级权重仍然建立在这个中心上，不会自动解决多兴趣问题。
3. **当前权重只做衰减。** 协同图和物品侧表示没有对应调整；收益可能只是改变语义尺度。global 的微小 Recall 增益和 NDCG 下降尤其不支持强机制主张。
4. **内容在当前模型中可能只是弱调制。** `ID * sigmoid(project(features))` 在门控近似常数时接近缩放后的 ID；亲和力再调整一个弱分支，指标自然变化有限。
5. **上下文空间不匹配。** 亲和力来自静态原始内容特征，最终排序在训练后的表示空间进行。两种相似性未必一致。

第二轮增加 `diagnostics/gate_*_between_item_std`、`gate_*_mean`、`semantic_to_id_norm_user/item`、`semantic_id_cosine_user/item`。GUME 记录总体语义/ID 范数比例和余弦。范数比例反映量级，不直接证明信息价值；必须结合删减实验。记录随验证 epoch 写入 history 和 TensorBoard，result.json 的 diagnostics 对应最佳验证检查点。

## 当前只执行的下一轮

先同步第二轮代码，再运行同一 Baby 数据、同一 seed 的 7 个完整 GUME 单因素对照：

```bash
python -m unittest discover -s tests -v
python experiments/run_suite.py --suite recover --datasets baby --seeds 999 --gpu 0 --output results/02_recover_baby --execute --resume
python experiments/analyze_results.py results/02_recover_baby --split validation --metric recall@20
python experiments/analyze_results.py results/02_recover_baby --split validation --metric ndcg@20
```

这 7 个版本依次是 GUME、冻结特征、去 BM、去 UM 分支及其专属正则、去 VT、去增强 II 边、均值融合。默认不测试 test。重新包含 GUME 是为了得到同代码版本、同机器负载条件的参照；不必重跑旧亲和力四组。

决策规则：

- **冻结特征明显下降**：优先尝试低秩修正，不能假设冻结高维特征没有代价。
- **去 BM 明显下降**：说明协同与语义之间的训练信号有价值；先保留 BM，不继续默认把所有辅助损失清零。
- **去 UM 明显下降**：先测试用历史生成扩展用户表示，区分“需要扩展视图”和“必须额外存两套自由用户向量”。
- **去 VT/粗细融合/增强边几乎不下降**：可以作为精简候选；接近的差值要增加 seed 后再决定。
- **每个单因素删减都接近完整 GUME，但同时删掉大幅下降**：可能存在互补或替代关系，接下来仅对相关两个因素做 2×2，而不是再加一批模块。
- **最佳 epoch 接近 1000 上限**：先检查训练是否充分。训练 loss 下降不等于 validation 改善，不能仅靠一张最优值表判断欠拟合。

如需事先定义工程筛选标准，可将“Recall 与 NDCG 相对下降都不超过 1%、可训练参数至少减少 30%”作为个人可接受的压缩目标。该 1% 是决策容忍度，不是显著性阈值，也不是已经证明的结果。

## 预先实现的两个后续方向

### 保留有效信号的轻量化

`--suite efficient` 包含 GUME、冻结特征、rank=8/16 的特征修正、历史扩展用户、历史扩展用户＋冻结特征。先根据 recover 结果决定是否运行。

低秩修正位置为：

```
gate_m = sigmoid(W2_m (W1_m F_m + A_m B_m) + b2_m)
A_m: [items, rank]
B_m: [rank, embedding_dim]
```

原始 F_m 冻结；其余完整 GUME 信号保留。B 初始化为 0，因此初始输出与同 seed 的冻结 GUME 一致。每模态新增参数约 `items*rank + rank*embedding_dim`；原始特征存储和每批投影开销仍存在，不能宣称总显存按这个比例下降。修正值有 L2 正则，A/B 本身没有分别添加新目标。低秩修正是已知参数节省思路，这里是针对性能来源的工程假设，不是已经成立的新贡献。

历史扩展用户版本保留原 UM 和专属正则，把扩展分支起点由两套自由用户表换成 `R * explicit_item_m`；不 detach，因此梯度仍回到物品特征。减少 `2 * users * embedding_dim` 参数，但传播深度和辅助损失仍保留。它检验扩展视图的收益能否通过确定性历史表示保留；也可能因视图过于相似而失败。

### 独立内容评分

新增 `CONTENTCF` 是单独模型文件，作为结构探索对照：

```
C = LightGCN(train_UI, ID_embeddings)
z_i^m = normalize(W_m normalize(raw_feature_i^m))
h_u^m = sum_{j in train_history(u)} z_j^m / (degree(u) + gamma)
s(u,i) = <C_u,C_i> + alpha/M * sum_m <h_u^m,z_i^m>
```

内容原始特征冻结。没有物品语义图、ID 特征乘法门控、额外模态用户表、亲和力和对比损失；使用同一 BPR 排序目标及 ID/投影参数 L2。训练每个三元组时，从用户内容历史中排除正例，并在正负打分中使用同一个排除后的历史。单历史用户的内容训练信号为零；推理使用全部训练历史。这是内容历史分支的防自匹配处理，不改变协同 UI 图原来的传导式训练协议。

`contentcf_cross` 在相同基本分数上增加 `sqrt(alpha)/M * sum_m(<C_u,z_i^m> + <h_u^m,C_i>)`，检验协同与内容的交叉项有无帮助；不增加额外的模态间交叉项。默认 alpha=0.1，`contentcf_weak` 为 0.01；由于内容表示归一化，alpha 与旧 AMORE 的 semantic_weight 数值不能直接比较。

先比较 text、image、both，检验模态互补是否实际存在。独立内容评分的目的，是让内容提供可独立观察的排序证据；它本身不能当作全新推荐范式或已经有效的创新。

后续可选命令（现在无需全部执行）：

```bash
python experiments/run_suite.py --suite efficient --datasets baby --seeds 999 --gpu 0 --output results/03_efficient_baby
python experiments/run_suite.py --suite content --datasets baby --seeds 999 --gpu 0 --output results/04_content_baby
```

默认仅写计划；明确要运行时加 `--execute`。取得正面结果后，优先用 3 个匹配 seed 和第二个数据集复核，再扩展组合或调参。若 CONTENTCF 不如 GUME，应如实结束该路线，不再把它强行拼回 GUME。

## 文献启发与未实施方向

- [FREEDOM](https://arxiv.org/abs/2211.06924)支持检验固定语义图的价值。注意“冻结图结构”不等于“冻结原始内容嵌入”；它不能证明当前 frozen-features 方案应该无损。
- [BM3](https://arxiv.org/abs/2207.05969)说明多模态推荐可以探索不依赖额外辅助图的训练路线。本次 CONTENTCF 保留 BPR，不是 BM3 复现。
- [ECIR 2026 的模态贡献研究](https://arxiv.org/abs/2508.07399)在多个模型上发现文本单模态经常接近完整多模态。它为 text/image 对照提供动机，不是对当前 Baby 结果的保证。

若后续仍想研究“亲和力”，更有意义的问题可能是“候选物品匹配用户的哪一种兴趣”或“哪种内容证据能改善当前候选的排序”，而不是给所有观察到的正交互附加一个固定可靠性权重。可考虑多兴趣历史、训练内分割校准的图邻居质量、或仅在候选重排阶段使用任务学习出的内容匹配；当前均未实现，不应写成实验发现。

## 把真正的结果同步回来

检查发现当前提交 `242ef75 all_result_1` 只新增/更新了 10 个 `.pyc`，不含这一轮结果。`results/` 在 `.gitignore` 中，普通 `git add .` 不会上传它。

在服务器运行：

```bash
python experiments/export_results.py results/01_core_baby --output experiment_reports/01_core_baby.json
python experiments/export_results.py results/02_recover_baby --output experiment_reports/02_recover_baby.json
git add experiment_reports/01_core_baby.json experiment_reports/02_recover_baby.json
git commit -m "Export Baby core and recovery experiment reports"
git push origin branch_2
```

第二轮未结束时，只执行第一条导出，并只 add 该文件。导出包含所有已完成运行的完整配置、result.json 和 history.json，没有模型权重、原始特征或 TensorBoard 二进制。推送后本地 pull 即可进一步检查 best_epoch、学习曲线和实际参数。导出不会触发重新训练。

## 本地验证状态

16 项自动测试通过，覆盖目前全部 46 个候选的前向、反向和排序；新增验证包括零初始化 adapter 与冻结 GUME 的初始输出一致且有梯度、history-users 精确移除两套用户表、CONTENTCF 的训练/完整排序公式一致以及正例排除、轻量结果导出。另对 adapter、history-users、CONTENTCF 和 cross 版本完成 4 次合成数据端到端短跑，包含检查点恢复和最终评估。

本地没有正式数据，这些检查仅验证代码与控制条件，尚未证明第二轮候选能提高 Baby 指标。本地验证为 PyTorch 2.12 CPU；保留项目已有 PyTorch 1.12 checkpoint 兼容处理，新候选尚需服务器短跑确认。
