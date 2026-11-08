# 已停止方向：AMOREX 独立框架实验

用户要求停止本方向。当前请按 [原 AMORE 复跑与精简流程](ORIGINAL_AMORE.md) 执行；以下仅保留历史设计和命令，不再作为执行计划。

2026-11-08。目标是改变用户表示和排序方式，检验多兴趣内容是否能提供 GUME 之外的有效信号。`amorex` 是实验名称，尚不是确定的论文模型。先只跑本文的短跑和六组初筛，不同时启动旧 core、recover 或 gume_fusion 套件。

## 1. 这次结果说明什么

终端表格已转录到 `experiment_reports/11_fusion_baby_reported.json`，不是原始实验记录。当前本地仍未取得完整配置、best_epoch 和训练曲线；不能据此诊断训练不足、判断显著性或宣称统计等价。

| Baby validation | Recall@20 | 相对 GUME | NDCG@20 | 相对 GUME | epoch 耗时倍率 |
|---|---:|---:|---:|---:|---:|
| gume | 0.101644 | — | 0.044344 | — | 1.000 |
| gume_ego_hard | 0.101632 | −0.012% | 0.044340 | −0.009% | 1.948 |
| gume_ego_hard_sg | 0.101130 | −0.506% | 0.044438 | +0.212% | 1.982 |
| gume_ego_soft | 0.101114 | −0.521% | 0.044207 | −0.309% | 1.658 |
| gume_ego_soft_sg | 0.101503 | −0.139% | 0.044467 | +0.277% | 1.646 |
| gume_sg | 0.101452 | −0.189% | 0.044389 | +0.101% | 0.991 |

所有变体的 Recall 都没有超过 GUME；部分 NDCG 有微小提升，且 Ego 版本有明显训练成本。**当前不足以支持采用这些附加模块作为主方法。** 一个 seed 不足以否定整个技术方向，但足以支持暂停这条路线的大规模搜索。

NDCG 表应理解为同一验证选模规则下得到的指标。按当前默认代码，检查点由 Recall@20 选择；运行分析器时换成 NDCG 不会重新按 NDCG 选 checkpoint。需原始配置确认服务器实际规则。

## 2. 框架真正改变在哪里

| 方面 | 当前 GUME 路线 | AMOREX |
|---|---|---|
| 内容表示 | 原始内容参与 ID 乘法门控和模态图 | 冻结原始内容，投影到可训练内容空间 |
| 内容传播 | 物品 KNN 图及增强图 | 训练历史到多个兴趣记忆，无物品 KNN 图 |
| 用户内容表示 | 融合后进入一个用户向量 | 每模态 K 个兴趣向量 |
| 排序 | 最终用户、物品向量内积 | 协同内积＋候选依赖的兴趣匹配分数 |
| 优化 | BPR 与多种对齐目标 | 一个排序目标＋L2；分别检验 BPR / 多负例交叉熵 |
| 亲和力位置 | 历史边消息权重 | 候选物品与不同兴趣的匹配，不给正交互判定可靠性 |

新模型文件 `src/models/amorex.py` 独立于 GUME。仅协同部分保留 LightGCN；没有继承 GUME 主体、SG 初始化、Ego 消息融合、三组辅助对齐损失、原始特征逐项微调或额外模态用户参数表。

这是实质性的框架探索，但不能把“多兴趣”本身称为首次提出：[MIND](https://arxiv.org/abs/1904.08030) 和 [ComiRec](https://arxiv.org/abs/2005.09347) 已研究多兴趣表示和匹配。这里针对非序列多模态推荐，研究**内容兴趣划分、排除目标后的历史记忆和候选条件评分能否替代静态语义图融合**。它们是否构成有价值的论文贡献，需要结果、消融及更完整的相关工作比较。本实现不是 MIND/ComiRec 的复现，也不把其论文结果当作本模型的证据。协同分支引用 [LightGCN](https://arxiv.org/abs/2002.02126)。

### 2.1 从内容构造多兴趣记忆

以下 H 只含 train 交互，m 为模态，k 为兴趣槽：

```text
z_i^m = normalize(W_m normalize(raw_feature_i^m))
a_ik^m = softmax_k(<z_i^m, normalize(p_k^m)> / route_temperature)

N_uk^m = sum_{j in H_u} a_jk^m z_j^m
D_uk^m = sum_{j in H_u} a_jk^m
h_uk^m = N_uk^m / (D_uk^m + history_shrinkage/K)
pi_uk^m = (D_uk^m + eps) / (sum_l D_ul^m + K eps)
```

W 和 p 可训练；原始图文向量冻结。每个模态独立划分兴趣，不强迫图文兴趣编号对齐。K 默认 4，不根据 validation/test 内容或标签聚类。原型用正交初始化，没有叠加“多样性损失”。槽位坍缩要通过诊断暴露，不能假设初始化可以保证槽位有不同语义。

分母把一个固定总量的收缩项分配给 K 个槽。若所有物品路由均匀，多槽会精确退化到单个历史均值的评分，不会仅因 K 增大就减弱内容强度。K=1 与同权重、无 cross 项的 CONTENTCF 公式一致，已加入回归测试。

### 2.2 候选决定哪个兴趣更相关

```text
C = LightGCN(train_UI, ID_embeddings)
s(u,i) = <C_u,C_i>
         + content_weight/M * sum_m [tau * logsumexp_k(
             log(pi_uk^m) + <h_uk^m,z_i^m>/tau)]
```

tau 为 `match_temperature`。这个非线性聚合不能一般性地化成一个固定用户向量与物品的内积。候选 A 可以匹配育儿兴趣，候选 B 匹配户外兴趣，不必先把两种兴趣平均到一个中心。这里只是解释性例子，模型槽位是否对应可解释主题需要后续实证。

不同兴趣分数近似时得到平滑混合；某一兴趣明显匹配候选时其贡献增大。pi 代表该兴趣在用户训练历史中的权重占比，不是已校准的可信度概率。历史很少时内容分数自动收缩。无历史用户的内容贡献为零。

### 2.3 排除目标，改变训练竞争集合

对训练正例 (u,i)，从 N、D 中分别减去 `a_ik z_i`、`a_ik`，然后构造 h、pi。**同一份排除后的历史用于正例和所有负例。** 单历史用户排除后内容贡献为零；验证时使用全部 train 历史。

这种排除只针对内容记忆。LightGCN 的训练图仍包含训练正边，与当前基线的传导式协议相同；不能声称整个模型不存在目标边消息传播。train 之外的交互完全不进入图、记忆或负例过滤。

主候选使用多负例交叉熵：

```text
L_rank = -log exp(s(u,i)/T) / sum_{j in {i, n_u} union P_u} exp(s(u,j)/T)
L = L_rank + reg_weight * L2
```

每个 batch 均匀无放回抽取最多 64 个共享候选，再保留数据加载器提供的逐用户负例 n_u。P_u 排除该用户所有已知 train 正例和重复的 n_u；不查看 validation/test 标签。采样器使用单独的 seed 流，使内容版和纯协同版使用一致的共享候选序列。T 默认 1。L2 沿用批内初始 ID 及投影矩阵正则，没有额外对齐或自监督损失。

这是**采样候选集上的排序交叉熵**，没有做采样概率校正，不等同于精确全物品 softmax，也不宣称无偏估计。未知交互仍可能是假负例。BPR 对照使用完全相同的评分、排除规则和温度，只改变竞争集合；没有额外候选时两种损失数学等价，已测试。

## 3. 第一轮只跑六组

| variant | 内容兴趣 | 排序目标 | 对照目的 |
|---|---|---|---|
| gume | 原 GUME | 原损失 | 同代码版本强参照 |
| amorex_id_bpr | 无内容 | BPR | 纯协同参照 |
| amorex_id_set | 无内容 | 多负例 CE | 训练目标本身的作用 |
| amorex_single | K=1 | BPR | 独立内容评分的作用 |
| amorex_multi | K=4 | BPR | 多兴趣相对单兴趣的作用 |
| amorex | K=4 | 多负例 CE | 内容与新目标的组合 |

这六组包含“内容关闭/开启 × BPR/CE”的 2×2 对照及 K=1。不能把 `amorex` 与 GUME 的全部差值归因于多兴趣。

### 第一步：同步后短跑

在服务器项目根目录、原来的训练环境执行。新模型没有新增第三方依赖；不要为它升级 Torch/CUDA。

```bash
conda activate torch1.12_cuda11.6
python -m unittest discover -s tests -v
python experiments/run_suite.py --suite rebuild --datasets baby --seeds 999 --epochs 2 --patience 2 --gpu 0 --output results/20_rebuild_check --execute --resume
```

六组没有 FAILED 且生成 result.json 后进行下一步。这是工程检查，2 epochs 的指标不用于决定模型优劣。

### 第二步：同样六组完整初筛

```bash
python experiments/run_suite.py --suite rebuild --datasets baby --seeds 999 --gpu 0 --output results/21_rebuild_baby --execute --resume
```

每组默认至多 1000 epochs，validation 连续 20 次未改善早停；顺序执行，不并行抢 GPU。先保持统一预算；若最佳 epoch 接近上限，再统一检查训练预算。`--resume` 跳过配置、代码和数据一致的已完成运行，未完成的运行从头训练，不从中断 epoch 续训。

另一个服务器终端看 TensorBoard：

```bash
~/venvs/amore-tensorboard/bin/tensorboard --logdir results/21_rebuild_baby --port 6006 --load_fast=false
```

如果旧 TensorBoard 正占着 6006，先停止那个查看进程再启动。不要中断训练终端。沿用之前的 SSH 6006 转发。

### 第三步：汇总并导出完整记录

```bash
python experiments/analyze_results.py results/21_rebuild_baby --baseline gume --split validation --metric recall@20
python experiments/analyze_results.py results/21_rebuild_baby --baseline gume --split validation --metric ndcg@20
python experiments/export_results.py results/21_rebuild_baby --output experiment_reports/21_rebuild_baby.json
```

把两张表和导出的 JSON 同步回来。`results/` 被 Git 忽略；只执行普通 pull/push 不会同步那些原始记录。也可用同样的 export 命令导出旧 `results/11_fusion_baby`，补上本次表格缺失的配置和曲线。

初筛不加 `--test`。暂时不跑全部变体、不扫大量超参数，也不据此改写论文实验表。

## 4. 看哪些诊断，下一步怎么决定

TensorBoard 继续支持多个 run 的 validation/recall、ndcg 曲线对比。新模型还输出：

- `train/component_0`：排序项；`train/component_1`：L2。BPR 和 CE 数值尺度不同，不按 loss 高低比较模型质量。
- `diagnostics/{image,text}_route_entropy`：物品路由是否接近均匀；K=4 时最大约 log(4)。
- `diagnostics/{image,text}_effective_slots`：总体槽使用量的熵指数，范围约 1 到 K。均匀路由也能得到 K，不能单独证明分化。
- `diagnostics/{image,text}_slot_cosine`：同一用户兴趣之间的余弦；只取固定的前至多 1024 个活跃用户作探针，接近 1 提示兴趣重复，不能代表全体用户。
- `diagnostics/{image,text}_memory_norm`：相同用户探针的记忆强度。
- `diagnostics/probe_content_abs` 与 `probe_collaborative_abs`：固定前至多 1024 条 train 边、排除正例后的分数绝对值均值，观察内容是否被协同分数淹没；不用于宣称贡献大小或推荐质量。

所有诊断随 epoch 写入 TensorBoard/history；result.json 保存最佳验证 checkpoint 对应诊断。没有增加诊断用的辅助损失。

决策按下面顺序：

1. **amorex_id_set 已解释几乎全部收益**：主要是训练目标效果，不能宣传多模态多兴趣创新。
2. **single 优于 id_bpr，multi 又优于 single**：有内容及多兴趣的正面线索，先检验多个 seed。
3. **multi 和 single 相当，槽余弦接近 1**：检查兴趣是否坍缩，不机械继续加 K 或增加损失。
4. **新框架均明显落后于 GUME**：检查训练曲线、最佳 epoch、内容分数量级。最多做有依据的局部尺度实验；仍无效就停止该结构，而不是把它再拼回 GUME。
5. **有候选稳定接近或超过 GUME**：选它和解释收益所必需的控制组，使用 `--seeds 999 2024 2025` 在同代码版本复核，再加 Sports。先获得可复现信号，再投入全面实验。

可选后续均已由配置支持，但现在不必执行：`--sweep 'num_interests=[1,2,4,8]'`、`--sweep 'content_weight=[0.3,1.0,3.0]'`、`--sweep 'modality=["image","text"]'`、`--sweep 'n_ui_layers=[0,3]'`；一次只选一个有诊断依据的因素。图层为 0 检验纯矩阵分解协同分支。冻结的最终比较应给 GUME 和候选合理、可比的调参预算，并在 validation 定型后才评估 test。

## 5. 资源与复现边界

内容分支可训练参数约为 `sum_m raw_dim_m * D + M*K*D`；K=1 不创建原型参数。没有按用户新增 K 套自由嵌入，也没有按物品微调原始内容。但冻结特征仍占内存/显存，历史记忆还占 `O(M*U*K*D)`，不能仅凭可训练参数量宣称总显存更小。

每批重新计算内容投影和稀疏历史记忆；主要内容计算量包含 `O(sum_m I*raw_dim_m*D + M*E*K*D)`。全排序增加 K 个兴趣的匹配，约 `O(M*B*I*K*D)`，可能明显慢于单内积 GUME。按用户与物品分块并逐槽 logaddexp，不构造全量 `[用户,物品,K]` 张量；最终 `[batch,物品]` 分数仍由现有 evaluator 接口要求保留。默认 eval_batch_size=512；需要同时报告训练时间、总 wall time、评估时间和显存，不能只看 epoch 时间。

本地验证只使用合成数据及 CPU。正式 Baby 的效果、PyTorch 1.12 CUDA 吞吐和峰值显存必须以服务器短跑和完整实验为准。当前没有宣称新框架涨点，也没有更改原 Word 论文里的实验数值。
