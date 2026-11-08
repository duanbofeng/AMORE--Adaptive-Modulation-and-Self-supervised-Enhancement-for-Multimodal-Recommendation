# 当前实验入口：完整 GUME + EgoGCN 消息融合 + SG-URInit 初始化

## 1. 这轮要回答什么

第一轮精简 AMORE 在 Baby validation 上明显落后于完整 GUME。现有结果不足以否定所有亲和力机制，也不支持继续堆叠旧 AMORE 模块。这轮统一使用**完整 GUME**，分别检验：

1. 在训练 UI 边上进行跨模态消息融合，是否改善用户模态表示？
2. 用训练历史及语义聚类初始化模态用户表，是否改善优化？
3. 两者组合是否有互补增益？

这是已经实现、可运行的候选模型；尚未获得真实数据上的新实验结果。`gume_ego_soft_sg` 是组合候选，不能预先称为最佳模型。借鉴已有方法本身也不能直接作为论文创新声明。

### 论文提供的依据

本地 `papers/GUME.pdf` 第 7 页表 3 的 Baby 结果如下：

| 论文版本 | R@20 | N@20 |
|---|---:|---:|
| GUME | 0.1042 | 0.0460 |
| w/o GE | 0.1039 | 0.0460 |
| w/o AL | 0.0954 | 0.0417 |
| w/o UM | 0.1011 | 0.0449 |

因此应先保留对齐和用户模态增强，再测试新模块。表中是论文报告结果，**不能与前一轮 validation 分数直接拼成对比表**，也不能替代本仓库的重跑。`gume_paper` 套件提供对应设计的工程消融：GE 关图增强；AL 同时关闭 BM、VT；UM 同时关闭 UM 损失和扩展物品正则。这里是当前实现的对应消融，不保证与作者未公开的每项消融细节完全一致。

## 2. 实际改动与方法边界

### EgoGCN：从消息融合切入

参考用户提供的 `EgoGCN-main/main.py` → `Model_routing_21sp.py` → `GATConv_pre.py`。实际入口使用的是 `GATConv_pre`。其 soft 分支采用其他模态的通道门控；hard 分支按**接收节点邻居间 softmax 注意力**决定是否融合；`reattn` 再计算融合后的注意力。它并非一个全局静态亲和力分数。

本轮在 GUME 的物品模态图传播之后、粗细融合之前，替换一小部分 item→user 模态聚合：

```text
GUME 原始特征门控 × item ID → 各模态 II 图传播 → 原始用户模态聚合
                                                     ↓
                               Ego 边注意力 + 跨模态消息融合
                                                     ↓
                  用户模态表示 = 原聚合 + λ × (新聚合 - 原聚合)
                                                     ↓
                        GUME 粗细融合、ID 残差与原有损失

SG 初始化（训练开始前，仅执行一次）
训练交互 + 原始特征聚类 → 初始潜空间的局部/聚类聚合 → extended modality user tables
```

具体适配：

- 两个模态，默认只修改显式 item→user 消息。ID 图、II 图、扩展用户分支和所有损失仍按完整 GUME 配置训练。
- 每条边的 query 默认减去该物品在原始用户聚合中的直接贡献，再计算余弦相似度。它是**直接项排除**，不删除 UI/II 图中所有间接路径，不声称严格的全图 leave-one-out。
- 在一个用户的训练邻居内 softmax，默认温度 0.2。soft 消息为 `(本模态 + sigmoid(W·另一模态+b) × 另一模态)/2`；hard 消息在通过阈值时取两模态均值，否则使用本模态。
- hard 默认使用 `邻居数 × 注意力 > 1.5`。绝对阈值 0.4 的对照也保留。固定绝对阈值会受节点度数影响，因此必须观察实际触发率；不能只因 hard 无提升便判断跨模态信息无效。
- hard 决策不使用直通梯度估计；梯度通过被选中的消息和重新计算的注意力传播。
- 聚合保留 GUME 原始 R 的行权重总量；这不保证向量范数完全一致。默认残差强度 0.2，避免一步完全替换。`self_message` 是同参数量的门控缩放对照，用于排查收益是否仅来自缩放。
- 度数为 0 或 1 的用户保留原聚合。边分数按块计算，默认开启梯度检查点；聚合使用稀疏矩阵，无稠密 user×item 张量，无新增 PyG/torch_scatter 依赖。
- PyTorch 1.12 原生 COO 边权反向会先构造稠密梯度再取稀疏位置（[官方源码 mm_mat1_sparse_backward](https://github.com/pytorch/pytorch/blob/v1.12.1/torch/csrc/autograd/FunctionsManual.cpp#L923)）。这里使用逐边内积计算边权梯度，避免该 user×item 分配，并用数值梯度及稠密小矩阵对照验证。只需一阶优化，不支持高阶优化用途。
- soft 新增 `2×(d²+d)` 个参数，d=64 时为 **8320**；hard 不新增参数。边注意力每个训练批次重算，会增加训练时间；额外计算约 O(E·d + I·d²)，还需实测 GPU 显存和速度。

代码：`src/utils/ego_router.py`；接入点：`src/models/gume.py`。这是 EgoGCN 启发的两模态、单向、残差式适配，**不是原版 EgoGCN 的复现结果**。

### SG-URInit：初始化现有模态用户表

核对的上游版本：[Jinfeng-Xu/SG-URInit，c85cb6d](https://github.com/Jinfeng-Xu/SG-URInit/tree/c85cb6d75b7fec75a8329ef4d0d10197dd992c7d)，重点是 `preprocessing/preprocess_agg.py` 和 `src/models/mentor_lg.py`。

保留其思想：按 `1/训练物品度数` 聚合交互物品；原始模态特征 KMeans；混合局部与聚类信息。KMeans 固定 preprocessing seed=0、n_init=10，默认 K=100，混合权重 0.5。

针对 GUME 做了明确适配：

1. 聚类依据原始模态特征；每簇代表向量是 GUME **初始、经过 II 传播的潜空间物品向量**的均值，使初始化与 GUME 消息处于同一坐标系。不是直接把高维特征塞入 64 维用户表。
2. 初始化 `extended_image_user` 和 `extended_text_user`，不改 ID 用户表、物品表、特征投影。该分支经 GUME UM 损失影响主模型，并非直接添加一个新的推理分支。
3. 默认将聚合结果的全局 RMS 匹配原用户表，保留相对用户幅度；另有不匹配尺度的 `gume_sg_raw` 对照。因此应称为 **SG-URInit 的 GUME 适配版**，不是原论文原样复现。
4. 只使用训练 loader 的交互，初始化后用户表继续正常学习。可使用全目录物品特征进行无监督聚类，这是与现有特征图一致的 transductive 设置；不使用验证/测试交互。
5. 稀疏聚合避免上游稠密 user×item 矩阵。聚类缓存按实际特征内容、K、算法设置与 sklearn 版本区分；交互聚合每次由当前训练集重算。改变数据稀疏程度不会复用错误的用户初始化。

无新增可训练参数或训练损失；首次 KMeans 有启动成本。代码：`src/utils/semantic_init.py`。不需要运行上游预处理或复制其 `agg_results`。

## 3. 从哪里开始运行

**所有下列训练命令在服务器的项目根目录运行，不是在 `src/` 中。** 使用已有 `torch1.12_cuda11.6` 环境，GPU 0，数据位于 `data/baby/`。本地修改需先通过 Git 提交推送或文件上传到服务器；仅 `git pull` 不会取得尚未推送的本地修改。

### A. 环境与短跑

SG 聚类新增依赖 scikit-learn。旧 Python 3.8 环境可以用下列范围；已有环境不必为此次实验重装整份 requirements：

```bash
python -m pip install 'scikit-learn>=1.1,<1.4'
python -m unittest discover -s tests -v
python -c "import torch, sklearn; print(torch.__version__, sklearn.__version__, 'CUDA:', torch.cuda.is_available())"
python experiments/run_suite.py --suite gume_fusion --datasets baby --seeds 999 --epochs 2 --patience 2 --gpu 0 --output results/10_fusion_check --execute --resume
```

短跑用正式模型尺寸，检查 GPU、训练、验证、聚类和保存流程。看到 `failures.json` 为 `[]` 再继续；2 个 epoch 的分数不用于选模型。此处不要附加 `experiments/smoke.json`，那个文件会缩小模型，仅供本地合成数据测试。

### B. 第一轮：Baby 六组完整筛选

```bash
python experiments/run_suite.py --suite gume_fusion --datasets baby --seeds 999 --gpu 0 --output results/11_fusion_baby --execute --resume
python experiments/analyze_results.py results/11_fusion_baby --baseline gume --split validation --metric recall@20
python experiments/analyze_results.py results/11_fusion_baby --baseline gume --split validation --metric ndcg@20
```

| 变体 | 新增内容 | 关键比较 |
|---|---|---|
| `gume` | 无 | 当前代码、相同训练条件的强基线 |
| `gume_ego_soft` | 软跨模态消息融合 | vs GUME |
| `gume_ego_hard` | 硬跨模态消息融合 | vs GUME、soft |
| `gume_sg` | 局部＋聚类用户初始化 | vs GUME |
| `gume_ego_soft_sg` | soft＋SG 初始化 | vs soft、SG、GUME |
| `gume_ego_hard_sg` | hard＋SG 初始化 | vs hard、SG、GUME |

默认最多 1000 epochs，验证 Recall@20 选 epoch，patience=20。所有组使用相同数据、seed、训练预算和原始 GUME 超参数；第一轮不读 test。单 seed 筛选只提供方向，不支持显著性结论。

**先执行到这里即可。** 不必先完成旧 `recover` 或旧 AMORE 的全部套件。

### C. 看曲线与导出给本地分析

```bash
tensorboard --logdir results --host 127.0.0.1 --port 6006
```

本地另开终端转发：

```bash
ssh -N -L 6006:127.0.0.1:6006 -p <SSH端口> <用户名>@<服务器地址>
```

浏览器打开 `http://localhost:6006`，选择 `11_fusion_baby` 下多个 run，同时比较：

- `validation/recall@20`、`validation/ndcg@20`：峰值、收敛速度、是否过早退化；曲线平滑设为 0 复查实际峰值。
- `train/loss`、`train/epoch_seconds`：总损失和时间。不同消融的总损失不可直接当性能指标。
- `diagnostics/ego_*_attention_relative_std`：注意力是否几乎均匀。
- `diagnostics/ego_*_update_ratio`：实际改动量，接近 0 表示机制几乎未改变表示。
- `diagnostics/ego_*_hard_fraction`、`*_hard_degree_2_10`、`*_hard_degree_11_plus`：硬调制是否完全不触发、是否对不同度数用户失衡。
- `diagnostics/ego_*_cross_gate_mean`：软门控是否饱和；`semantic_to_id_norm`：语义残差是否过大或过小。

```bash
python experiments/export_results.py results/11_fusion_baby --output experiment_reports/11_fusion_baby.json
```

把导出的 JSON 同步回来即可，内含配置、最佳 epoch、参数量、结果与历史曲线。`results/` 被 Git 忽略，仅 pull 仓库不会带回服务器原始结果。`--resume` 只跳过同配置已完成的运行，不从中断 epoch 继续训练。

## 4. 第一轮之后如何决策

| 观察 | 下一步 |
|---|---|
| soft/hard 优于 GUME，组合更好 | 做机制消融，再多 seed 验证组合增益 |
| SG 有效，Ego 无效 | 优先保留 SG，检查 local 是否已经足够；不因有两个模块就强行组合 |
| Ego 有效，SG 无效 | 保留融合模块，停止堆加初始化 |
| 组合只优于 GUME，却不优于最强单模块 | 暂无互补证据，选择更简单的单模块 |
| Recall 略升但 NDCG 下降 | 作为权衡报告；不要写“全面提升” |
| 几组都低于 GUME | 检查基线、触发率与幅度，做有限小范围调参；仍无效就放弃这条实现路线 |

### D. 有信号后做消融

```bash
python experiments/run_suite.py --suite ego_controls --datasets baby --seeds 999 --gpu 0 --output results/12_ego_controls --execute --resume
python experiments/run_suite.py --suite sg_controls --datasets baby --seeds 999 --gpu 0 --output results/13_sg_controls --execute --resume
```

只运行与有效方向有关的套件：

| 对照 | 排查的问题 |
|---|---|
| `gume_ego_attention` | 只做邻居重加权是否足够，无需跨模态消息？ |
| `gume_ego_uniform` | 无条件均值跨模态聚合是否已经有收益？ |
| `gume_ego_shuffle` | 同一用户内打乱初次／二次注意力与边的对应关系后，收益是否消失？ |
| `gume_ego_self_message` | 同参数量、自模态门控缩放是否能解释收益？ |
| `gume_ego_no_reattn` | 消息融合后重新算注意力是否必要？ |
| `gume_ego_self_query` | 排除直接自项是否必要？ |
| `gume_ego_hard_absolute` | 相对阈值是否比原始风格绝对阈值更适合本数据？ |
| `gume_sg_local` | 仅训练历史初始化是否已经足够，无需聚类？ |
| `gume_sg_shuffle` | 打乱用户与初始化向量对应关系后是否仍有收益？ |
| `gume_sg_raw` | RMS 尺度匹配的影响有多大？ |

shuffle 是诊断对照，不是性能候选。如果真实映射与 shuffle 的表现无法区分，就没有证据把收益归因于对应关系所携带的语义。

需要检查 GUME 原模块时，单独运行 `--suite gume_paper`。新模型轻量化应采用配对设计，例如对 `gume` 与选定候选**同时**施加 `--sweep 'vt_loss=[0.0]'`，与二者原版构成 2×2 对比；不要只在新模型中删除损失，再把差异全部归因于新模块。

### E. 有限调参，避免全面组合搜索

先固定 SG 参数，只对有效 Ego 分支试强度：

```bash
python experiments/run_suite.py --variants gume_ego_soft --datasets baby --seeds 999 --sweep 'ego_strength=[0.1,0.2,0.4]' --gpu 0 --output results/14_strength --execute --resume
```

hard 可依次试 `ego_threshold=[1.0,1.5,2.0]`；温度可试 `[0.1,0.2,0.5]`。SG 先试 `sg_cluster_weight=[0.0,0.25,0.5,0.75]`，再对胜出权重试 `sg_clusters=[50,100,200]`。每轮仅改变一个因素，根据 validation 决定是否继续。不要把所有列表一次放入笛卡尔积。

若为候选调优 GUME 学习率，也必须给基线同等调优预算。模块参数搜索不可用 test 决策。

### F. 多 seed、跨数据集、最终 test

默认参数组合已有信号时，下面 6×3=18 次训练可以检验 Baby 上的稳定性和互补性：

```bash
python experiments/run_suite.py --suite gume_fusion --datasets baby --seeds 999 2024 2025 --gpu 0 --output results/15_fusion_multiseed --execute --resume
python experiments/analyze_results.py results/15_fusion_multiseed --baseline gume --split validation --metric recall@20
python experiments/analyze_results.py results/15_fusion_multiseed --baseline gume_sg --split validation --metric recall@20
python experiments/analyze_results.py results/15_fusion_multiseed --baseline gume_ego_soft --split validation --metric recall@20
```

不同 `--baseline` 会覆盖同名分析 JSON，需长期保存时重命名。重点看逐 seed 增量、均值、标准差和置信区间，3 seeds 的统计能力仍有限；最终论文建议 5 seeds。组合还应报告交互增量 `组合 − Ego单模块 − SG单模块 + GUME`，不能只看组合是否超过 GUME。

再对 Sports、Clothing 做同样验证；可以先选 GUME、最强单模块和组合三个候选，避免一开始跑全量。

**设计、参数与筛选准则固定后**才运行带 `--test` 的最终版本。例如若最终确认 soft＋SG 默认参数方案：

```bash
python experiments/run_suite.py --variants gume gume_ego_soft gume_sg gume_ego_soft_sg --datasets baby sports clothing --seeds 999 2024 2025 --gpu 0 --test --output results/16_fusion_final --execute --resume
python experiments/analyze_results.py results/16_fusion_final --baseline gume --split test --metric recall@20
python experiments/analyze_results.py results/16_fusion_final --baseline gume --split test --metric ndcg@20
python experiments/export_results.py results/16_fusion_final --output experiment_reports/16_fusion_final.json
```

这是按锁定配置重新训练、按 validation 选 epoch、最后一次 test 的流程。若选择了非默认参数，先将它们写入独立 overrides JSON，再对相应候选用 `--overrides 文件`；不要遗漏第一轮选择的配置，也不要把只适用于候选的修改意外施加给基线。

最终报告应包含 Recall/NDCG@10/20、平均与标准差、用户稀疏度分组、物品冷/尾/头分组、可训练参数、启动时间和每 epoch 时间。分析器现有 tail 定义是**正训练度数的底部 20% 分位**，与 GUME 论文按物品度数五等分、后四组称 tail 的口径不同，必须注明；不能混用两者的长尾结论。

## 5. 复现边界

- 关闭 `ego_mode` 与 `sg_init` 时保持当前仓库 GUME 默认路径；Ego strength=0 的输出、损失、共同参数梯度和初始化 RNG 有回归测试。
- 23 项本地测试通过，覆盖现有全部 62 个候选的前向、反向和排序；包括分块梯度、边权数值梯度、训练边支持、邻居 softmax、度数 0/1 回退、硬阈值、SG 稀疏权重、初始化范围、尺度与打乱对照，以及组合模型训练和 checkpoint/TensorBoard 诊断。另完成六组候选的两 epoch 合成数据子进程短跑。
- 本地没有正式数据和 CUDA 环境。合成数据测试只能证明工程路径可运行，不能证明提升，也不能代替服务器 PyTorch 1.12 / CUDA 11.6 的短跑。
- 原版 EgoGCN 与 SG-URInit 源码仅作参考；本项目运行不依赖这两个外部目录或联网下载。保留并引用 GUME、EgoGCN、SG-URInit 的方法来源。
