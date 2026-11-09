# AMORE 审稿修订：固定 rev_11，按意见补证

**当前决定：完整 `rev_11` 是论文主模型，固定 run_6 的主干、超参数和训练协议，不再探索新结构。** `rev_00` 仅保留在已完成的亲和力消融中。rev_11 不是当前 Baby 五种子均值最高者，不能称为“已经证明最优”。GUME 只作为鲁棒性、分组与效率的一个固定参照，不展开多基线调参。

这份清单替代此前的大菜单。旧命令保留在 [FULL_EXPERIMENT_MENU.md](FULL_EXPERIMENT_MENU.md)；[AFTER_RUN6.md](AFTER_RUN6.md) 的结果分析仍有效，但其中探索性 A–D 运行建议本轮暂停。

## 1. 哪条审稿意见，对应哪个实验？

编号对应根目录 `IPM-D-26-07304-reviews.pdf`：R1 的 Weaknesses 1–5，R2 的条目 1–5，R3 的 Review Comments 1–8。

| 审稿意见 | 要回答的问题 | 当前状态 | 本轮具体处理 |
|---|---|---|---|
| **R1.2、R2.1、R3.4（分支）** | ELAM 有用吗？共享亲和力连接传播与融合是否有用？ | **Baby 五种子已完成** | 使用已有 35 次实验，不重跑、不再加结构，见第 2 节 |
| **R3.2** | 均值/方差约束比单独 InfoNCE 多带来什么？ | 未完成 | **步骤 2**：固定 rev_11，矩约束 × InfoNCE 2×2 |
| **R3.4（损失）** | 每个辅助损失的独立贡献？ | 未完成 | **步骤 2**：七个非 BPR 损失逐项置零；与上述 2×2 合并，共十个设置 |
| **R3.5** | 矩约束和 L2 权重是否敏感？ | 未完成 | **步骤 3**：只扫 vt_loss、reg_weight_1、reg_weight_2 |
| **R3.6** | 可控交互/模态噪声下是否更稳健？ | 未完成 | **步骤 4**：rev_11 vs 固定 GUME，三类噪声，0/10%/20% |
| **R2.4** | 稀疏用户、长尾与真正零交互物品的效果？ | 未完成 | **步骤 1**：稀疏/长尾分组；**步骤 5**：移除 20% 物品全部训练边 |
| **R1.5、R3.7** | 参数、训练/推理耗时、显存开销？ | 有训练记录；缺完整推理测量 | **步骤 1**：恢复检查点补推理计时；训练指标如实来自原记录 |
| **R3.3** | 多种子、SD、显著性，排除随机波动？ | 亲和力已完成；其余未完成 | 各阶段配对种子及 SD/CI；步骤 1 五种子，补充阶段先三种子，定稿扩五种子 |
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

**对应 R1.5、R2.4（分组）、R3.3、R3.7。**

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

**对应 R2.4 的 cold-start 部分，长尾不能替代。**

```bash
python experiments/reviewer_fixed.py --stage cold --datasets baby --seeds 999 2024 2025 --execute --finish
```

固定选择 20% 原 warm 物品，移除其全部训练边，保留目录/内容特征并重建图。rev_11/GUME 各三种子，共 **6 次新增训练**。沿用固定验证选择与测试协议。

**产出：** `revision_outputs/reviewer_fixed/cold/cohorts/cohort_metrics.csv`、`baby_cohorts_<条件哈希>.png/pdf`；总体表 `cold/overall/summary.csv`；原始测试 `results/reviewer_fixed/evaluated/cold/`。

**怎样分析：** 重点是 cold_items 行及 eligible_users/positives。此处 head/tail 按干预后物品度数分组，与步骤 1 不一定是同一组样本，不能直接相减。该实验是目录内零交互物品的传导式场景，不是未见物品归纳泛化。

## 5. 预算、暂停点与定稿范围

按顺序、旧数据和权重齐全时，Baby 预计新增：

| 步骤 | 新增训练 | 主要图表 |
|---|---:|---|
| 已有亲和力 | 0 | 配对、交互、训练诊断 |
| 1 固定参照/效率/长尾 | 4 | 性能表、效率图、用户/长尾分组 |
| 2 损失 | 27 | 单项损失、MACA 2×2 |
| 3 敏感性 | 18 | 三参数敏感性 |
| 4 噪声 | 36 | 三类噪声、两指标 |
| 5 零交互 | 6 | 冷物品分组 |
| 合计 | **91** | 无结构搜索，无多基线调参 |

**先步骤 1–2，回传图表后再做 3–5，不要求一次性跑完。** 每步独立回答具体意见。若旧结果不匹配，实际训练数会增加；没有权重需先找回，不会用随机初始化补评估。

这是 Baby 先行补证，不能覆盖原稿三个类别的所有论断。定稿须将适用阶段扩展到 Sports/Clothing、补足五种子：保持命令和设置，只改 `--datasets`/`--seeds`，旧匹配运行复用。Sports/Clothing 尚无本轮已完成证据，不能标为完成。三种子阶段不承诺稳定显著优势，也不按测试结果选新主模型。

**本轮暂停：** 跨亲和力的 `loss-factorial`、`prune-followup` 剪枝、`probe_affinity.py` 启发式改结构、Ego/SG 新模块、全量基线调参、全量阈值扫描。旧计划没开始的不用运行。

## 6. 回复审稿人与回传

已有亲和力负结果如实报告；步骤 2–5 完成前，不能将对应结论写成已证实。R1.1/R1.3/R2.3 的完整基线比较仍暂缓，R1.4 的跨域实验不开展；上述五步不代表解决了全部拒稿理由。`reviewer_matrix.json` 的 `current_fixed_protocol` 字段记录本轮映射，旧 response 字段仍是待回填草稿，不要直接提交。

回传 `results/reviewer_fixed` 的 JSON/history/日志，以及 `revision_outputs/reviewer_fixed`。分组复算需要 `test_rankings.npz`；模型权重留服务器即可。

```bash
tensorboard --logdir results/reviewer_fixed/training --port 6006
```

新训练正常写 TensorBoard；复用训练不重建事件，但保留 history 用于静态图。重新绘图不训练，例如：

```bash
python experiments/reviewer_figures.py --stage losses
```

本地只验证代码与合成数据正确性，真实 GPU 补充实验尚未运行。本文的待生成路径不是已经取得的结果。
