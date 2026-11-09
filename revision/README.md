# AMORE 审稿修订：固定 rev_11，按意见补证

**当前决定：完整 `rev_11` 是论文主模型，固定 run_6 的主干、超参数和训练协议，不再探索新结构。** `rev_00` 仅保留在已完成的亲和力消融中。rev_11 不是当前 Baby 五种子均值最高者，不能称为“已经证明最优”。GUME 只作为鲁棒性、分组与效率的一个固定参照，不展开多基线调参。

这份清单替代此前的大菜单。旧命令保留在 [FULL_EXPERIMENT_MENU.md](FULL_EXPERIMENT_MENU.md)；[AFTER_RUN6.md](AFTER_RUN6.md) 的结果分析仍有效，但其中探索性 A–D 运行建议本轮暂停。

## 0. 本次回传核查：并未全部完成（2026-11-09）

核查基于 `5682edd (run_all)` 中的文件，以及作者随后提供的服务器进度。**提交名 `run_all` 不代表五个阶段都完成。保持完整 rev_11 为主模型；当前只补已有计划的缺项，不增加结构探索。**

### 0.1 完成度：训练、评估出图、原始证据分开记

| 阶段 | 计划规模（Baby） | 服务器进度 | 本地已同步 | 现在要做什么 |
|---|---|---|---|---|
| 1 reference | 2 模型 × 5 种子 = 10 | 已有测试、分组、效率导出 | 10 行运行、440 行分组、汇总/图；缺原始 JSON/history/排名 | **不因同步缺失重训**；补回原始记录 |
| 2 losses | 10 设置 × 3 种子 = 30 | **11/30** 有训练 `result.json`；未有评估/图 | 无该阶段输出 | 接着执行步骤 2 原命令 |
| 3 sensitivity | 3 参数 × 4 点 × 3 种子 = 36 | vt_loss **12/12**；reg_weight_1 **3/12**；reg_weight_2 **0/12**；未出图 | 无该阶段输出 | 步骤 2 完成后，接着执行步骤 3 原命令 |
| 4 robustness | 3 噪声 × 3 比例 × 2 模型 × 3 种子 = 54 | 交互噪声 **15/18**；图像 **0/18**；文本 **0/18**；未评估/出图 | 无该阶段输出 | 步骤 3 完成后，接着执行步骤 4 原命令 |
| 5 cold | 2 模型 × 3 种子 = 6 | 已有测试、分组导出 | 6 行运行、264 行分组、汇总/图；缺原始 JSON/history/排名 | **不因同步缺失重训**；补回原始记录 |

服务器进度来自作者检查，尚未由本地原始文件复核。`result.json` 数量是计划项的训练结果数，可能包含复用，不等于实际新增训练次数，也不意味着测试评估已完成。当前 `results/reviewer_fixed/` 尚未同步到本地；Sports/Clothing 也没有本轮完整证据。

**汇总数据质量检查通过：** reference 的 999/2024/2025/2026/2027、cold 的 999/2024/2025 均为两模型配对种子；无重复设置/种子、无指标缺值或越界。16 行运行的汇总均值、样本 SD、t 分布 95% CI 可重算；11 个分组 × K=5/10/20/50 的 **704 行**齐全。按有效用户数加权的三类用户分组能还原总体 Recall/NDCG，最大绝对误差小于 `6e-16`；head/tail/cold 正例总量也吻合。

CSV 中记录的源码哈希及交互/模态特征哈希一致。**这只验证导出数据相互一致，不等于重新核验原始文件、检查点或排名。** 尚需原始 JSON 检查完整配置、最佳轮次选择、干预物品身份，history 检查收敛，NPZ 复算排名。效率 CSV 的 `test_evaluation_seconds` 为空，但独立的预热推理计时字段完整；不能把前者当作零耗时。

审计结果与复算方式（只读已有 CSV，不训练）：

```bash
python experiments/audit_reviewer_exports.py
```

输出：[export_audit.json](../revision_outputs/reviewer_fixed/export_audit.json)，包含输入 CSV 的 SHA256、逐指标配对差值/CI/p 值、分组均值/SD、效率统计。该脚本专门检查当前 Baby reference/cold 导出，不用于宣布全部五阶段完成。

### 0.2 总体测试结果：尚未证明稳定提升

下表为 **测试集 mean ± sample SD**；不要与第 2 节 run_6 的验证集数值混合。

| 条件 | 模型 | 种子数 | Recall@20 | NDCG@20 |
|---|---|---:|---:|---:|
| 正常训练 reference | GUME | 5 | 0.103748 ± 0.001014 | 0.045495 ± 0.000472 |
| 正常训练 reference | rev_11 | 5 | 0.103439 ± 0.000826 | 0.045493 ± 0.000382 |
| 移除 20% warm 物品全部训练边 cold | GUME | 3 | 0.092463 ± 0.000362 | 0.040918 ± 0.000317 |
| 同上 | rev_11 | 3 | 0.092453 ± 0.001066 | 0.040792 ± 0.000361 |

正常条件下 rev_11 − GUME 的 Recall@20 差值为 **−0.000309**，配对 95% CI **[−0.001849, 0.001232]**；NDCG@20 差值 **−0.000002**，CI **[−0.000714, 0.000710]**。每阶段分别对 Recall/NDCG@10/@20 四项配对 t 检验进行 Holm 校正，两阶段均未检出差异（校正 p 均为 1）。这是未发现差异，**不是等效检验，也不能写“显著优于 GUME”**；三/五种子的统计精度有限。

cold 行是干预后的**全体测试用户总体指标**，不能代替真正 cold_items 的表现。若计算从干净到干预的退化，必须在 reference 中也取相同三个种子；不能直接拿五种子均值与三种子均值作配对差。

### 0.3 稀疏、长尾和零交互：需要保留负结果

以下都是对应分组的 Recall@20，均值 ± 种子 SD；人数是每个运行的有效评估用户数，不乘种子数。

| 条件/分组 | 用户 / 测试正例 | GUME | rev_11 | 可支持的描述 |
|---|---:|---:|---:|---|
| reference / 稀疏用户 | 9,783 / 9,783 | 0.106900 ± 0.001031 | 0.106327 ± 0.000844 | 未出现明确稀疏用户优势 |
| reference / warm 长尾 | 8,285 / 8,713 | 0.017320 ± 0.002234 | 0.016584 ± 0.001056 | rev_11 均值低约 4.25% |
| reference / 自然零度物品 | 7 / 7 | 0 | 0 | 样本太少，不能单独验证冷启动能力 |
| cold / 剩余 warm 长尾 | 6,710 / 6,977 | 0.018803 ± 0.001964 | 0.021709 ± 0.001961 | rev_11 均值高约 15.46%，仅为描述性结果 |
| cold / 零训练交互物品 | 4,106 / 4,225 | 0.001705 ± 0.000244 | 0.001055 ± 0.000141 | rev_11 均值低约 38.10%，且两者绝对召回都很低 |

零交互组 NDCG@20 同样较低：GUME **0.000606**、rev_11 **0.000363**。因此不能将 warm 长尾增益写成“解决冷启动”。这些组是不同条件下按训练度数定义的群体，尤其两条件的 head/tail 成员会变化，不能跨条件直接相减。用户活跃度分组按原始干净训练度数划分，cold 条件的物品分组按移边后度数划分。用户分组彼此互斥；不同物品组的有效用户可能重叠，不能把人数简单相加。

现有分组图共用纵轴，cold 的小数值差异视觉上不明显，论文中应同时给出上表数值及有效样本量。这里没有对筛选出的分组声称显著增益，也不能归因于亲和力：要隔离亲和力，应使用已有 run_6 消融；GUME vs rev_11 是整模型比较。

### 0.4 效率：参数略少，不能笼统称为全面更快

记录的训练与推理设备均为 RTX 3090，PyTorch 1.12.1。下表为五种子均值 ± SD；推理是每个种子一次预热、五次重复取中位数，再跨种子统计，覆盖 19,445 个用户。

| 指标 | GUME | rev_11 | 解释 |
|---|---:|---:|---|
| 可训练参数 | 36,076,352 | 33,591,681 | 减少 6.89% |
| 原记录训练秒/epoch | 4.956 ± 2.324 | 2.244 ± 0.009 | 记录均值较低，但 GUME 波动大且负载未受控 |
| 原记录训练 CUDA 峰值 MiB | 885.345 | 878.518 | 仅减少约 0.77% |
| 新测推理 ms/全用户 pass | 14.974 ± 1.852 | 20.050 ± 3.452 | 本次记录均值增加约 33.90% |

训练指标来自旧记录，并非这次重新基准测试；推理不含指标统计和写盘，不能当作在线服务端到端延迟。编码/排名/总耗时分别取中位数，其分量中位数之和未必等于总中位数。论文可写参数规模较小及实际训练耗时记录，同时披露推理代价；不能以此声称全面更快，也不能将不受控的训练计时当作严格加速比。

### 0.5 哪些图现在可以使用？

所有路径相对仓库根目录；下列 PNG 均有同名 PDF。表格是现有导出，未改写原始结果。

| 用途 / 意见 | 已存在图 | 对应分析数据 |
|---|---|---|
| Baby 总体测试 / R3.3 | `revision_outputs/reviewer_fixed/reference/overall/baby_performance.png` | 同目录 `runs.csv`、`summary.csv`；配对统计见 `export_audit.json` |
| 参数、训练、显存、推理 / R1.5、R3.7 | `revision_outputs/reviewer_fixed/reference/baby_efficiency.png` | 同目录 `efficiency_runs.csv`、`efficiency_provenance.csv` |
| 稀疏用户与长尾 / R2.4 | `revision_outputs/reviewer_fixed/reference/cohorts/baby_cohorts_84df8915.png` | 同目录 `cohort_metrics.csv` |
| 零交互与剩余 warm 分组 / R2.4 | `revision_outputs/reviewer_fixed/cold/cohorts/baby_cohorts_a7728f3a.png` | 同目录 `cohort_metrics.csv` |
| 干预后总体测试 | `revision_outputs/reviewer_fixed/cold/overall/baby_performance.png` | 同目录 `runs.csv`、`summary.csv` |
| 收敛辅助展示 | 两条件 `overall/baby_convergence.png` | 原始逐轮 history 尚未同步，当前不能本地复算曲线 |
| 亲和力独立贡献 / R1.2、R2.1、R3.4 | 第 2 节的 `revision_outputs/run6/` 图 | 已有五种子验证集消融，不与本轮测试集混用 |

效率正文优先用 reference 根目录的**四面板** `baby_efficiency`，其 `overall/` 下同名图只是训练时间与性能的关系图。损失、敏感性和噪声图目前**不存在**，第 4 节对应路径仍是待生成产物。

### 0.6 接下来只补三步

先按第 3 节更新服务器代码，然后逐条执行；**不要同时启动三条命令**，步骤 3/4 可复用前面已完成结果，也避免多任务抢占 GPU。使用与首次运行相同的数据路径、输出目录和种子。

```bash
# A. 先补损失消融（当前仅 11/30 有训练结果）；成功出图后再执行 B
python experiments/reviewer_fixed.py --stage losses --datasets baby --seeds 999 2024 2025 --execute --finish

# B. 补三个参数的敏感性（当前 15/36 有训练结果）；成功出图后再执行 C
python experiments/reviewer_fixed.py --stage sensitivity --datasets baby --seeds 999 2024 2025 --execute --finish

# C. 补三类噪声（当前 15/54 有训练结果）
python experiments/reviewer_fixed.py --stage robustness --datasets baby --seeds 999 2024 2025 --execute --finish
```

脚本会跳过符合协议的已完成结果，并尝试复用相同配置；尚未写出完整 `result.json` 的中断训练可能从头开始，**不是逐 epoch 恢复训练状态**。不要删除已完成目录或旧 `best.pt`。每一步必须命令正常退出、完成规定运行数并生成该阶段图表/表格，才标记完成；失败时保留日志，不能把缺失运行当成零分或跳过种子。

完成这三步才能回答 R3.2/R3.4（损失）、R3.5、R3.6。当前数据已经能够补充 Baby 的效率、稀疏/长尾、零交互分析，但**尚不足以支持完整机制、鲁棒性及三类别的统一结论**。不因为结果未提升而增加模块搜索；按固定 rev_11 如实报告每项支持与不支持的证据。

## 1. 哪条审稿意见，对应哪个实验？

编号对应根目录 `IPM-D-26-07304-reviews.pdf`：R1 的 Weaknesses 1–5，R2 的条目 1–5，R3 的 Review Comments 1–8。

| 审稿意见 | 要回答的问题 | 当前状态 | 本轮具体处理 |
|---|---|---|---|
| **R1.2、R2.1、R3.4（分支）** | ELAM 有用吗？共享亲和力连接传播与融合是否有用？ | **Baby 五种子已完成** | 使用已有 35 次实验，不重跑、不再加结构，见第 2 节 |
| **R3.2** | 均值/方差约束比单独 InfoNCE 多带来什么？ | 训练 11/30；未评估/出图 | **步骤 2**：固定 rev_11，矩约束 × InfoNCE 2×2 |
| **R3.4（损失）** | 每个辅助损失的独立贡献？ | 训练 11/30；未评估/出图 | **步骤 2**：七个非 BPR 损失逐项置零；与上述 2×2 合并，共十个设置 |
| **R3.5** | 矩约束和 L2 权重是否敏感？ | 训练 15/36；未出图 | **步骤 3**：只扫 vt_loss、reg_weight_1、reg_weight_2 |
| **R3.6** | 可控交互/模态噪声下是否更稳健？ | 训练 15/54；未评估/出图 | **步骤 4**：rev_11 vs 固定 GUME，三类噪声，0/10%/20% |
| **R2.4** | 稀疏用户、长尾与真正零交互物品的效果？ | Baby 分组导出齐全；原始排名待同步 | **步骤 1**：稀疏/长尾分组；**步骤 5**：移除 20% 物品全部训练边 |
| **R1.5、R3.7** | 参数、训练/推理耗时、显存开销？ | Baby 五种子效率导出齐全；原始记录待同步 | **步骤 1**：恢复检查点补推理计时；训练指标如实来自原记录 |
| **R3.3** | 多种子、SD、显著性，排除随机波动？ | 亲和力已完成；reference/cold 已有测试汇总；2–4 未完成 | 各阶段配对种子及 SD/CI；步骤 1 五种子，补充阶段先三种子，定稿扩五种子 |
| **R1.1、R1.3、R2.3** | 与 MENTOR/缺失方法公平比较，证明相对增益？ | **尚未解决，本轮暂缓** | 一个 GUME 参照不能代替完整多基线比较 |
| **R1.4** | 非 Amazon 跨域泛化？ | 不增加数据集 | 明确局限；三个商品类别不能替代视频/音乐跨域证据 |
| **R2.2、R2.5** | 公式、伪代码、符号、基线数目不一致？ | 有审计与修订稿，定稿需复核 | 文字/公式工作，不额外训练；按完整 rev_11 描述 |
| **R3.1** | 近期相关工作缺失？ | 有相关工作修订，定稿需复核 | 文献与归属说明，不新增结构实验 |
| **R3.8** | 未验证的机制被写成结论？ | 已有负结果，文字须同步 | 不把不显著写成等效，不把亲和力解释成真实噪声标签 |

**不是每条意见都单独训练。** 步骤 2 同时回答 R3.2/R3.4；步骤 1 同时产出测试统计、长尾图和效率表；步骤 4 一次回答三种噪声。已有亲和力证据继续用于 R1.2/R2.1/R3.3/R3.4/R3.8。

## 2. 已经完成：数据、图表和结论

- `results/revision/anchor/`：Baby 四个入口各一次、seed 999，用于身份/协议检查，不是多种子主表。
- `results/revision/affinity/`：Baby 七种设置 × 五种子 = 35 次，含 `result.json` 与 `trial_000/history.json`。
- 五种子：999、2024、2025、2026、2027；固定训练源码 SHA256：`2e8303f646847ee71f10b79316f68537af1d339ebf6e8c5e17bd022ee33cc894`。

以下路径均相对仓库根目录：

| 现成图表 | 路径（图同时有 PDF） | 分析重点 |
|---|---|---|
| 亲和力配对差值、95% CI | `revision_outputs/run6/paired_effects.png` | 双支/单支、打乱、常数、停止梯度未显示稳定增益 |
| 图传播 × 融合交互 | `revision_outputs/run6/factorial.png` | 未发现稳定协同，误差条为种子 SD |
| 收敛、损失、先验诊断 | `revision_outputs/run6/training_mechanisms.png` | 模块触发不等于有效；损失值大小不等于梯度强弱 |
| 配对统计、Holm 校正 | `revision_outputs/run6/paired_statistics.csv` | 每指标八项比较的差值、SD、CI、p 值 |
| 每种子差值 | `revision_outputs/run6/paired_seed_differences.csv` | 保留失败种子 |
| 各运行与逐轮数据 | `revision_outputs/run6/audited_runs.csv`、`training_diagnostics.csv` | 可追溯原始 JSON |
| 数据/源码与重复运行审计 | `revision_outputs/run6/audit.json` | anchor 重复种子未混入五种子统计 |

完整 rev_11 的 Recall@20=**0.102588**、NDCG@20=**0.044448**；rev_00 为 0.102628、0.044512。两项配对差值的 95% 区间均跨零，真实/常数/打乱先验也接近。**这是 Baby 验证集证据，不是三数据集最终测试结果。** 固定 rev_11 是本轮作者的模型选择，不改变以上结论。

需要重画时才执行，不训练：

```bash
python experiments/analyze_run6.py --root results/revision --output revision_outputs/run6
```

## 3. 服务器准备

```bash
git checkout branch_3
git pull --ff-only origin branch_3
```

使用已有 PyTorch 1.12/CUDA 环境；绘图需要 matplotlib、pandas、scipy，训练需要 tensorboard。**保留服务器上的旧 `best.pt`**，补评估需要读取，权重不必传 GitHub。

下列命令均在仓库根目录执行；数据不在 `data/` 时，每一步加相同的 `--data-path /实际目录`。默认 GPU 0，可统一加 `--gpu 1`。

- 没有 `--execute`：只生成计划、估计复用数。
- `--execute --finish`：训练缺失项，恢复检查点评估，并自动出图。
- 重复同一命令可续跑；真实复用核对完整训练配置、源码及数据哈希。
- 训练源码/默认配置保持 run_6 版本。`--finish` 除敏感性阶段外评估测试集；这些结果用于报告，**不用于改主模型或选参数**。

## 4. 分步运行：先 Baby，一步结束再进入下一步

### 步骤 1：固定参照、长尾/稀疏用户、效率

**对应 R1.5、R2.4（分组）、R3.3、R3.7。当前已有 Baby 五种子图表，见第 0 节；下面是原运行命令，不需要因原始文件未同步而重训。**

```bash
python experiments/reviewer_fixed.py --stage reference --datasets baby --seeds 999 2024 2025 2026 2027 --execute --finish
```

rev_11/GUME 各五种子。复用 rev_11 五次训练、GUME seed 999，预计只新增 **4 次 GUME**。恢复验证集最佳检查点评估测试集、导出排名；推理一次预热、五次计时，分别记录编码和全目录排名。

**产出位置：**

- 原始测试/排名/推理结果：`results/reviewer_fixed/evaluated/reference/`；训练结果：`training/reference/`。
- 性能表：`revision_outputs/reviewer_fixed/reference/overall/summary.csv`，同目录 `baby_performance.png/pdf`。
- 效率图：`revision_outputs/reviewer_fixed/reference/baby_efficiency.png/pdf`；表 `efficiency_runs.csv`、`efficiency_provenance.csv`。
- 分组图：`revision_outputs/reviewer_fixed/reference/cohorts/baby_cohorts_<条件哈希>.png/pdf`；表 `cohort_metrics.csv` 含每种子 Recall/NDCG、eligible_users、positives。
- 显著性结果：`results/reviewer_fixed/evaluated/reference/analysis_test_recall_10.json`、`analysis_test_recall_20.json`、`analysis_test_ndcg_10.json`、`analysis_test_ndcg_20.json`。

**怎样分析：** 用户按训练活跃度分三组；warm 物品中度数最高 20% 为 head，其余为 tail，零度另列。比较低活跃/尾部绝对指标及与 GUME 的差异。正常数据没有零度正例时 cold 栏为空，需步骤 5。

训练时间/显存来自原训练记录，推理当前新测。运行时避免其他 GPU 重负载。原训练负载未严格控制，不能把微小计时差解释为确定加速；表中保留测量来源。此步骤只比较 GUME，不代表已解决全部基线意见。

### 步骤 2：MACA 理由与各损失贡献

**对应 R3.2、R3.4（损失）、R3.3。**

```bash
python experiments/reviewer_fixed.py --stage losses --datasets baby --seeds 999 2024 2025 --execute --finish
```

十种固定设置：完整；分别去掉矩约束、BM、extended BM、UM cross、UM noise、最终表示 L2、extended L2；再加无全部 InfoNCE、无矩约束且无 InfoNCE。完整参考复用，预计 **27 次新增训练**。这些是固定模型消融，不据此替换主模型。

矩约束 × InfoNCE 四组为：完整、无矩、无 InfoNCE、两者都无，两个 L2 保持。无 InfoNCE 不是纯 BPR，也不等于删除辅助传播。

**产出：** `revision_outputs/reviewer_fixed/losses/`

- `baby_loss_effects.png/pdf`：相对完整模型的配对变化及 CI。
- `baby_moments_infonce.png/pdf`：2×2 曲线、种子 SD。
- `paired_statistics.csv`：九个移除比较的每指标 Holm 校正，矩×InfoNCE 交互另列；`runs.csv` 保存各运行。
- 原始测试：`results/reviewer_fixed/evaluated/losses/`。

**怎样分析：** 删除后下降支持该项贡献；持平/升高应承认缺乏支持。不能预写“全部损失有用”。三种子统计精度有限，最终模型不按这些测试结果更改。

### 步骤 3：只补点名的三个权重敏感性

**对应 R3.5。**

```bash
python experiments/reviewer_fixed.py --stage sensitivity --datasets baby --seeds 999 2024 2025 --execute --finish
```

| 参数 | 主模型固定值 | 曲线取值 |
|---|---:|---|
| vt_loss | 0.1 | 0、0.01、0.1、1 |
| reg_weight_1 | 1e-5 | 0、1e-6、1e-5、1e-4 |
| reg_weight_2 | 0.1 | 0、0.01、0.1、1 |

一次只变一个参数。36 个计划项，复用完整参考及步骤 2 的零权重点后，预计 **18 次新增训练**。不扫阈值、不组合调参、不更新主模型参数。

**产出：** `revision_outputs/reviewer_fixed/sensitivity/baby_sensitivity.png/pdf`、`runs.csv`；原始记录：`results/reviewer_fixed/training/sensitivity_*`。

**怎样分析：** 这是验证集敏感性图，横轴明确使用分类间距以包含零。报告参数在哪些区间稳定、过大/过小如何退化，不挑最高点替换主配置。

### 步骤 4：可控噪声鲁棒性

**对应 R3.6，同时约束 R3.8 的去噪解释。**

```bash
python experiments/reviewer_fixed.py --stage robustness --datasets baby --seeds 999 2024 2025 --execute --finish
```

rev_11 vs 固定 GUME；假交互、图像打乱、文本打乱分别取 **0%、10%、20%**。每个条件重训，干预种子 314159、评估标签和候选掩码保持原协议。零噪声复用，预计 **36 次新增训练**。

**产出：** `revision_outputs/reviewer_fixed/robustness/`

- `baby_robustness_recall_20.png/pdf`、`baby_robustness_ndcg_20.png/pdf`。
- 上排绝对指标，下排相对各自干净条件的下降百分比；误差条为种子 SD。
- `runs.csv`、`degradation_by_seed.csv`；原始测试：`results/reviewer_fixed/evaluated/noise_interactions`、`noise_image`、`noise_text`。

**怎样分析：** 同时报告干净水平和退化量；干净分数较高不能独自证明更抗噪。未优于 GUME 就收缩去噪主张。置换比例是被选中的物品比例，可能存在固定点；假边排除已知 train/validation/test 正例，是离线人工压力测试，不是真实噪声标签或部署策略。

### 步骤 5：零训练交互物品

**对应 R2.4 的 cold-start 部分，长尾不能替代。当前已有 Baby 三种子图表，见第 0 节；下面保留原运行命令，无需因同步缺失重训。**

```bash
python experiments/reviewer_fixed.py --stage cold --datasets baby --seeds 999 2024 2025 --execute --finish
```

固定选择 20% 原 warm 物品，移除其全部训练边，保留目录/内容特征并重建图。rev_11/GUME 各三种子，共 **6 次新增训练**。沿用固定验证选择与测试协议。

**产出：** `revision_outputs/reviewer_fixed/cold/cohorts/cohort_metrics.csv`、`baby_cohorts_<条件哈希>.png/pdf`；总体表 `cold/overall/summary.csv`；原始测试 `results/reviewer_fixed/evaluated/cold/`。

**怎样分析：** 重点是 cold_items 行及 eligible_users/positives。此处 head/tail 按干预后物品度数分组，与步骤 1 不一定是同一组样本，不能直接相减。该实验是目录内零交互物品的传导式场景，不是未见物品归纳泛化。

## 5. 预算、暂停点与定稿范围

以下是最初从 run_6 开始、旧数据和权重齐全时的计划预算，**不是本次剩余训练数，也不是已经完成的数量**。当前进度以第 0 节为准：

| 步骤 | 新增训练 | 主要图表 |
|---|---:|---|
| 已有亲和力 | 0 | 配对、交互、训练诊断 |
| 1 固定参照/效率/长尾 | 4 | 性能表、效率图、用户/长尾分组 |
| 2 损失 | 27 | 单项损失、MACA 2×2 |
| 3 敏感性 | 18 | 三参数敏感性 |
| 4 噪声 | 36 | 三类噪声、两指标 |
| 5 零交互 | 6 | 冷物品分组 |
| 合计 | **91** | 无结构搜索，无多基线调参 |

**当前 reference/cold 已有导出，按第 0.6 节依次补步骤 2 → 3 → 4，每步结束回传。** 每步独立回答具体意见。若旧结果不匹配，实际训练数会增加；没有权重需先找回，不会用随机初始化补评估。

这是 Baby 先行补证，不能覆盖原稿三个类别的所有论断。定稿须将适用阶段扩展到 Sports/Clothing、补足五种子：保持命令和设置，只改 `--datasets`/`--seeds`，旧匹配运行复用。Sports/Clothing 尚无本轮已完成证据，不能标为完成。三种子阶段不承诺稳定显著优势，也不按测试结果选新主模型。

**本轮暂停：** 跨亲和力的 `loss-factorial`、`prune-followup` 剪枝、`probe_affinity.py` 启发式改结构、Ego/SG 新模块、全量基线调参、全量阈值扫描。旧计划没开始的不用运行。

## 6. 回复审稿人与回传

已有亲和力负结果及本次 reference/cold 的收益与退化如实报告；步骤 2–4 完成前，不能将对应机制、敏感性、鲁棒性结论写成已证实。R1.1/R1.3/R2.3 的完整基线比较仍暂缓，R1.4 的跨域实验不开展；上述五步不代表解决了全部拒稿理由。`reviewer_matrix.json` 的 `current_fixed_protocol` 字段记录本轮映射，旧 response 字段仍是待回填草稿，不要直接提交。

回传 `results/reviewer_fixed` 的 JSON/history/日志，以及 `revision_outputs/reviewer_fixed`。本次已调整 `.gitignore`，允许该目录的 `.json` 和 `.log` 纳入版本控制；权重、TensorBoard 事件和排名 NPZ 继续忽略。**先 pull 此次修改**，以后可以显式提交这两类目录，无需强制添加整个 results：

```bash
git add results/reviewer_fixed revision_outputs/reviewer_fixed
git diff --cached --stat
git commit -m "Sync completed reviewer evidence and figures"
git push origin branch_3
```

提交前确认列出的文件不含 `best.pt` 或数据集。分组复算还需要 `test_rankings.npz`；它们较大，建议通过既有文件同步方式单独传回。下面只打包排名，**不包含模型权重**：

```bash
python - <<'PYZIP'
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
files = sorted(Path('results/reviewer_fixed').rglob('test_rankings.npz'))
if not files:
    raise SystemExit('No test_rankings.npz found; check checkpoint evaluation first')
with ZipFile('/tmp/reviewer_fixed_rankings.zip', 'w', ZIP_DEFLATED) as archive:
    for path in files:
        archive.write(path, str(path))
print(len(files), 'ranking files -> /tmp/reviewer_fixed_rankings.zip')
PYZIP
```

将该 ZIP 复制到本地，在仓库根目录解压，保留 `results/reviewer_fixed/...` 相对目录。只需同步已完成阶段，未完成阶段的计划/日志也可以一并回传以定位失败。已有 `results/revision/affinity` 的原始结果继续保留。

```bash
tensorboard --logdir results/reviewer_fixed/training --port 6006
```

新训练正常写 TensorBoard；复用训练不重建事件，但保留 history 用于静态图。重新绘图不训练，例如：

```bash
python experiments/reviewer_figures.py --stage losses
```

本次本地已完成 reference/cold 真实服务器 CSV 的一致性与配对统计复算；未进行新的 GPU 训练。服务器已产出这两阶段图表，步骤 2–4 仍需续跑。原始证据尚未同步的限制见第 0 节，不能把待生成路径当成已经取得的结果。
