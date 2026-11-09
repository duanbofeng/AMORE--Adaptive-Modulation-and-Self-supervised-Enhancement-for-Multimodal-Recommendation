# run_6 之后：发现、现成图表与下一轮实验

本报告基于 `3919ce1` 的汇总与 `8f662d9` 补齐的原始 JSON/history。当前分支 **branch_3**。只分析最初 AMORE 的原始/保守修复/修稿控制入口，不混入早期 slim AMORE、GUME-Ego/SG 或 AMOREX。

## 1. 目前可以确定什么

### 1.1 数据完整性

- Baby：anchor 4 次（各 seed 999），affinity 7 变体 × 5 种子 = 35 次。
- 35 次 affinity 使用相同源码哈希、数据文件哈希、主干超参数和训练预算；配置差异仅为预设开关及运行路径/种子。
- 以下均为**验证集**，按 Recall@20 选最佳 epoch；NDCG@20 是同一检查点的值。不是最终测试集结果。
- anchor 的重复 `rev_11/999` 不计入五种子样本。分析脚本遇到重复种子、混合配置或缺失配对会报错。

### 1.2 亲和力没有显示出稳定收益

| 变体 | Recall@20 均值 | NDCG@20 均值 |
|---|---:|---:|
| 两支关闭 rev_00 | **0.102628** | **0.044512** |
| 只开图传播 rev_10 | 0.102591 | 0.044423 |
| 只开融合 rev_01 | 0.102621 | 0.044483 |
| 两支开启 rev_11 | 0.102588 | 0.044448 |
| 融合先验打乱 | 0.102555 | 0.044418 |
| 融合先验常数 | 0.102573 | 0.044447 |
| 融合先验停止梯度 | 0.102554 | 0.044431 |

`rev_11 − rev_00`：

- Recall@20：−0.00003970，相对 −0.0387%；配对 95% t 区间 **[−0.00042827, +0.00034888]**。
- NDCG@20：−0.00006467，相对 −0.1453%；区间 **[−0.00036250, +0.00023315]**。
- 图传播 × 融合的交互效应：Recall +0.00000340、NDCG +0.00005362，区间也跨零。
- 每个指标内对八项预设比较做 Holm 校正，所有校正 p 值均为 1。完整种子差值、SD、区间和原始 p 值见 `revision_outputs/run6/paired_statistics.csv`。

**结论：当前 Baby/固定配置不支持共享亲和力及双分支协同带来稳定增益。** 尤其真实融合先验与常数/打乱结果接近，未支持“物品特异的共享先验是增益来源”。这不是证明所有亲和力方法都错误，也不是等效性检验；只有五个种子和一个数据集。

### 1.3 anchor 说明了什么

| 入口（单种子 999） | Recall@20 | NDCG@20 | 最佳 epoch |
|---|---:|---:|---:|
| 干净 GUME | 0.101548 | 0.044292 | 37 |
| 最初 AMORE | 0.102100 | 0.044199 | 68 |
| 保守修复 AMORE | 0.101895 | 0.043975 | 58 |
| 修稿 rev_11 | 0.101805 | 0.043991 | 59 |

最初 AMORE 的 Recall 较高、NDCG 较低，单种子不足以宣称优于 GUME。修复解决的是实现与实验协议问题，未显示出性能提升。**不能把修复版的结果直接写成投稿时原模型的结果。**

同一 `rev_11/999` 在 affinity 又得到 Recall 0.101908，与 anchor 相差 **0.00010285**，约为两支开/关五种子均值差绝对值的 2.6 倍。两次源码、配置、训练数据、GPU 型号均匹配（配置只差输出路径）。存在运行间数值差异，尚未定位具体来源；两次运行不足以估计其方差。不要把同种子当成逐位确定性，也不要挑更好的一次混入比较。

### 1.4 模块在运行，问题是有无有效贡献

`rev_11` 最佳检查点的记录：硬调制选边比例约 **49.9%–52.8%**，物品先验 SD 约 **0.0869–0.0887**。因此“亲和力全部变成常数／硬调制没有触发”与日志不符。

约 15–22 epoch 的先升后降出现在关闭亲和力的模型中，也出现在开启的模型中，时点有偏移。不能把这个现象单独归因于亲和力。最佳 epoch 多在 54–75，后续正常触发 20 轮 patience；现有证据不支持只把训练上限从 1000 增加就能解决问题。

第 60 epoch，`rev_11` 五种子加权损失均值约为：BPR 0.0846、UM cross 0.5137、UM noise 1.0478、矩约束 0.00169。**损失值大小不等于梯度影响大小**，不能据此直接宣布噪声损失有害；但优先检验这些辅助目标比继续堆模块更有依据。

两支开/关参数相差仅 4,289 / 约 3,359 万（约 0.013%）。`rev_00` 为保持控制协议仍计算未使用的先验；且其 epoch 计时在 1.93–3.15 秒波动。现有计时不能证明关闭亲和力更快或更慢，也不能据此宣称显著压缩模型。

## 2. 一个可验证的结构解释

当前实现令粗融合为 Z，两个模态表示为 V/T，行为门控为 b_v/b_t，则语义分支为：

```text
S = [Z + b_v * (V - Z) + b_t * (T - Z)] / 3
  = [b_v * V + b_t * T + (1 - b_v - b_t) * Z] / 3
```

在固定协同表示/门控时，Z 到输出的逐维系数是 `(1-b_v-b_t)/3`。如果门控之和接近 1，亲和力改变 Z 的影响会被衰减。这是**代码中的精确代数关系 + 尚待验证的门控假设**；目前 JSON 没有门控张量，不能声称已经证实。

另一个假设：用户内 softmax 在分数接近时约为 `1 / 用户度数`。绝对阈值 `>0.1` 可能部分编码用户度数。新诊断会将真实物品先验与 `mean_{u in N(i)}(1/degree(u))` 比较。高相关也不是先验无效的充分条件，必须结合排序敏感性及受控训练。

## 3. 已经能用于论文分析的图（无需新增训练）

| 文件（同时有 PDF） | 内容与用途 |
|---|---|
| `revision_outputs/run6/paired_effects.png` | 每种子差值、均值及配对 95% CI；适合机制有效性/反事实对照分析 |
| `revision_outputs/run6/factorial.png` | 图传播 × 融合 2×2，误差条是种子 SD；适合协同主张检查 |
| `revision_outputs/run6/training_mechanisms.png` | 收敛、同轮差值、各加权损失、先验/选边诊断；适合训练行为说明 |

推荐正文用配对图和训练诊断，2×2 图可放附录，避免内容重复。所有图必须标注 Baby validation、五种子；不要放入最终 test 主表。训练图只画所有种子都存在的 epoch，不填充早停后的值。

复现：

```bash
python experiments/analyze_run6.py --root results/revision --output revision_outputs/run6
```

可用英文描述（当前探索证据，不替代最终实验）：

> On the Baby validation split, the joint affinity variant did not yield a consistent improvement over the same backbone with both affinity branches disabled. The mean paired differences were −0.000040 in Recall@20 and −0.000065 in NDCG@20 across five seeds, with both 95% confidence intervals including zero. Shuffling or replacing the fusion prior with its mean produced similar aggregate performance. These observations do not establish equivalence, but they do not support attributing an improvement to item-specific shared affinity under the evaluated configuration.

回复审稿人的口径：说明新增独立开关、常数/打乱/停止梯度对照后，没有得到预期的稳定提升，因此收缩机制主张；不能把不显著写成“证明稳健”，不能选择性删除负结果。当前仅一个数据集的验证证据，回复信应明确后续确认是否完成。

## 4. 接下来从这里运行

服务器仓库根目录、现有 torch 环境：

```bash
git checkout branch_3
git pull --ff-only origin branch_3
```

本次没有改训练模型源码或默认损失，因此原有实验的源码指纹保持一致。新脚本兼容 Python 3.8 写法；本地验证为 Python 3.12 / PyTorch 2.12 CPU，服务器的 PyTorch 1.12 / CUDA 环境仍以实跑为准。

### A. 首先：已有检查点诊断，0 次新增训练

服务器保留 `results/revision/affinity/**/trial_000/best.pt` 即可，无需传到本地。

```bash
python experiments/probe_affinity.py --root results/revision/affinity --dataset baby --data-path data --output revision_outputs/affinity_probe
```

它读取五个 `rev_11` 最佳检查点，校验训练源码、数据哈希及训练边，测量：

1. 先验与度数基准的 Spearman 相关、RMSE。
2. alpha、混合系数、粗融合输出系数的绝对均值/P90。
3. **固定同一组权重**，将先验改为常数/打乱、关闭融合/传播/两支后，表示与全目录分数的相对 L2 变化、Top-20 集合重合率。
4. 重复原始前向的数值差异，作为上述干预的运行噪声参照。

固定随机抽取最多 1024 个验证用户，所有种子使用相同抽样，掩码为训练正例；不使用测试集，不更新参数。输出 `probe.json` 和 `counterfactual_sensitivity.png/pdf`。Top-K 重合率表示模型对干预的敏感性，**不是推荐准确率，也不能代替重新训练消融**。

若常数/打乱几乎不改分数，而关闭图传播影响明显：重点怀疑融合先验的使用路径。若系数很小，才有依据尝试移除抵消路径；若与度数高度相关，再考虑度数校正的相对亲和力控制。任何新架构都需要新名字、新源码批次及重新训练参照，不预先承诺提升。

### B. 必做训练：损失与亲和力交叉，24 次 / Baby

```bash
python experiments/revision_suite.py --stage loss-factorial --datasets baby --seeds 999 2024 2025 --output results/revision_followup --execute
python experiments/plot_followup.py --root results/revision_followup --output revision_outputs/followup
```

设计：`rev_00 / rev_11` × 全部目标 / 无矩约束 / 无 InfoNCE / 两者皆无 × 三种子。

矩约束 = `vt_loss`；InfoNCE = BM、extended BM、UM cross、UM noise 四项。**关闭 InfoNCE 仍保留两个表示 L2**，所以它不是“纯 BPR”，也不代表辅助传播已经删除。其余超参数固定、验证选择协议一致。

输出：`baby_loss_removal.png/pdf`（移除目标的效果）和 `baby_affinity_by_objective.png/pdf`（不同目标下的亲和力收益），以及包含“亲和力 × 目标移除”差中之差的统计表。置信区间为逐项探索区间，不能用于从众多候选中挑选显著结果后作确认性结论。

判读：

- 无矩约束持续持平/更好：删去“矩约束改善跨模态分布”的主张，候选模型可去掉该项。
- 无 InfoNCE 明显下降：辅助自监督有贡献，不能因为损失标量大就全部删除。
- 亲和力在所有目标下都没有稳定收益：优先将 `rev_00` 作为简化候选，不继续为亲和力调出好看的结果。
- 只有特定目标组合下亲和力改善：记录为新假设，另用固定五种子和另外两个类别确认。

### C. 有必要补的论文图：噪声敏感性曲线，新增 36 次 / Baby

在 B 完成后运行，干净参照会复用 B 的 `followup_full`，不重复训练：

```bash
python experiments/revision_suite.py --stage stress-followup --datasets baby --seeds 999 2024 2025 --output results/revision_followup --execute
python experiments/plot_followup.py --root results/revision_followup --output revision_outputs/followup
```

两种亲和力设置 × 三类干预（假交互、图像打乱、文本打乱）× 两个非零强度（0.1、0.2）× 三种子 = **36 次新增**。若单独先跑 C，连干净参照共 42 次。每个条件重新训练，使用固定干预随机数 314159；测试/验证标签保持不变。特征打乱报告的是选中比例，置换可能有固定点。假边构造排除已知 train/validation/test 正例，是离线压力实验，不作为可部署的数据处理方法。

输出 `baby_stress_recall_20.png/pdf`、`baby_stress_ndcg_20.png/pdf`，分别画三类干预的绝对性能与 `rev_11 − rev_00` 配对区间。检验“干净数据没收益，但是否抗噪”，不能只展示下降较少的单个种子或只展示一类噪声。

### D. 若还要精简体量：辅助视图剪枝，新增 12 次 / Baby

```bash
python experiments/revision_suite.py --stage prune-followup --datasets baby --seeds 999 2024 2025 --output results/revision_followup --execute
python experiments/plot_followup.py --root results/revision_followup --output revision_outputs/followup
```

共享 B 的完整参照，新增“无噪声对比”“无整个 extended 分支”各 2 亲和力设置 × 3 种子 = 12 次。后者同时关闭 extended BM、UM cross、UM noise、extended L2，使 `use_extended=False`，实际跳过两次辅助传播；它是**分支级剪枝**，不能冒充只删除一个损失的消融。

三个阶段放在同一 `--output`，相同源码/数据/配置才会 resume。B+C 共 60 次新增训练；B+C+D 共 72 次。默认仍是验证实验。先不做大范围敏感性扫描，也不推进统一基线调参。

### E. 何时扩展 Sports / Clothing

先做 A+B；C 用来检验抗噪主张；D 在决定精简辅助分支时执行。根据结果固定候选和主张后，再把对应命令扩展到 Sports/Clothing，并补齐五种子。不要仅凭 Baby 确定三个类别的结论。确认候选时可把相同命令 `--seeds` 改成 `999 2024 2025 2026 2027`，原三种子会复用。

## 5. 尚未填补但应明确记录的缺口

- 现阶段只有 Baby validation，不能填写三数据集最终 test 主表；统一基线调参按本轮要求暂缓。
- 没有按用户活跃度/物品长尾拆分的逐用户排名记录。当前 `result.json` 的总体指标无法还原这些分组；不能按先验大小随意分组后声称改善冷启动。后续需要保存对应 split 的完整排名/真实标签再做 cohort 分析。
- 暂未做真正零交互物品实验。训练稀疏/长尾不等于零交互冷启动；此前 `--stage cold` 仍可用，等核心候选确定再投入。
- 图本身不是贡献。若 A–D 仍不支持亲和力，应修改方法定位、摘要和创新列表，保留透明的负结果；不能把删掉失败模块包装成已经证明的新模型。

## 6. 回传什么

回传新 `results/revision_followup/**/result.json`、`trial_000/history.json`、日志和 `revision_outputs/affinity_probe`、`revision_outputs/followup`。模型权重留在服务器即可。实验沿用 TensorBoard 写入；可以启动：

```bash
tensorboard --logdir results/revision_followup --port 6006
```

TensorBoard 展示训练与验证曲线，新绘图脚本负责配对统计和适合论文导出的 PDF。不要把不同目标/不同种子直接平均成一条未标注的曲线。
