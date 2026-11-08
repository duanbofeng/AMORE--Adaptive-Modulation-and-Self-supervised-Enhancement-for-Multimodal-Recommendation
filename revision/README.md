# AMORE 审稿修订：从这里开始

本轮针对 **最初 AMORE**。代码来源是 `old_model/src/models/gume.py`；`old_model/src/models/amore.py` 只是入口包装。不要把新目录下的 `gume`、早期 `amore_base/global/local`、GUME-Ego/SG 或 AMOREX 当成原始 AMORE。

当前材料是可执行的修稿方案和待实验回填的英文稿。**没有把未运行的实验写成结果，也不保证新开关提升效果。** 仅使用 Baby、Sports、Clothing。跨域证据仍缺失；EgoGCN、EVEN、MMSSL、DiffMM 的完整统一复现仍待补齐，不能将其写成已经完成。

## 1. 服务器更新和环境检查

本次工作在仓库当前的 `branch_3` 完成，不要切回旧 `branch_2` 后运行。

```bash
git checkout branch_3
git pull --ff-only origin branch_3
python experiments/setup_baselines.py
python -m unittest discover -s tests -p 'test_*.py'
```

使用现有 torch 环境先短跑；根目录 `requirements.txt` 面向新环境，不要因此覆盖现有 PyTorch 1.12/CUDA 11.6。本地验证环境是 Python 3.12 / PyTorch 2.12 CPU，旧版 CUDA 环境需服务器验证。绘图需要 matplotlib、pandas、scipy；TensorBoard 需要 tensorboard。固定版本基线下载到忽略的 `external_baselines/MMRec`；首次需要 GitHub 网络。下载器校验 SHA256，并带入上游许可。原始参考文件不改写。

数据布局继续使用 `data/baby`、`data/sports`、`data/clothing`，每个目录包含已有 `.inter`、`image_feat.npy`、`text_feat.npy`。不重新随机切分，不重新提取特征。若数据在别处，下面所有命令加相同的 `--data-path /实际数据目录`。

## 2. 第一次先运行这两步

先做真实数据上的两轮技术检查，输出与正式结果隔离。检查是否 OOM、损失非有限、图构建错误、检查点恢复错误；这不是可用于论文的实验。

```bash
python experiments/revision_suite.py --stage anchor --datasets baby --seeds 999 --epochs 2 --patience 2 --output results/revision_smoke --execute
```

随后完整训练四个参照模型，仍然只看验证集：

```bash
python experiments/revision_suite.py --stage anchor --datasets baby --seeds 999 --execute
python experiments/analyze_results.py results/revision/anchor --split validation --metric recall@20
```

| 名称 | 用途 |
|---|---|
| `gume` | 当前干净 GUME 基线，Ego/SG 关闭 |
| `amore_original` | 原始 AMORE 前向与损失，适配统一验证选择协议 |
| `amore_fixed` | 原始 AMORE 的保守修复版，保留完整辅助分支 |
| `rev_11` | 与 `amore_fixed` 对齐的修稿实验入口，两个亲和力开关均开 |

`amore_original` 与修复版不要求数值相同：去重 InfoNCE、真实 batch 分母等明确改变了优化。`amore_fixed` 与 `rev_11` 同配置时应等价；已有前向和共有参数梯度测试。若参照异常，先回传日志，不继续投入全套预算。

## 3. 先回答亲和力是否值得保留

```bash
python experiments/revision_suite.py --stage affinity --datasets baby --execute
python experiments/analyze_results.py results/revision/affinity --split validation --metric recall@20 --baseline rev_00
python experiments/plot_revision.py results/revision/affinity --split validation --output revision_outputs/affinity
```

默认五个种子：999、2024、2025、2026、2027。7 变体 × 5 种子 = 35 次训练/数据集。

| 变体 | 图传播使用亲和力 | 融合使用亲和力 | 检验问题 |
|---|---|---|---|
| `rev_00` | 否 | 否 | 同一完整主干的无亲和力参照 |
| `rev_10` | 是 | 否 | 传播的独立贡献 |
| `rev_01` | 否 | 是 | 融合的独立贡献 |
| `rev_11` | 是 | 是 | 联合贡献 |
| `rev_prior_shuffle` | 是，真实值 | 是，物品间打乱 | 物品与先验的对应关系是否重要 |
| `rev_prior_constant` | 是，真实值 | 是，均值常数 | 样本差异是否重要 |
| `rev_prior_detach` | 是，真实值 | 是，停止先验梯度 | 共享信号与反向梯度的区别 |

这些比较使用相同固定主干和超参数。不要各自调参后再称为“只改变一个因素”。原始硬阈值不可导；detach 检验的是融合分支到先验的梯度，不是取消整张图的梯度。关闭融合会去掉专用小 MLP，常数对照则保留它，用于排查额外容量解释。

判定规则：报告每个种子差值、平均差、95% CI、Holm 校正 p 值及效应大小。若真实先验与打乱/常数无法区分，应收缩“共享亲和力是核心创新”的论断。若 rev_00 持平或更好，优先无亲和力版本；这将改变论文主张，不能只删实验。若只单支有效，保留该支并修改方法和标题/摘要。只有稳定的成对增益才支持保留完整两支。

Baby 筛查后，将上述命令的 `--datasets baby` 改为 `--datasets sports clothing`。不能用 Baby 的结果宣称三个类别均有效。

## 4. 统一基线与公平调参

已接入 BPR、LightGCN、BM3、MGCN、FREEDOM、LGMRec、MENTOR，加本地 GUME 与 rev_11，共 9 个模型。7 个外部基线使用固定源码，适配清单见 [BASELINES.md](BASELINES.md)。默认 YAML 仅用于筛查，不是已经调优的最强成绩。

```bash
python experiments/revision_suite.py --stage baselines --datasets baby --seeds 999 --execute
python experiments/tune_revision.py --datasets baby --budget 12 --seeds 999 2024 2025 --execute
```

第二条为每个模型 12 个候选 × 3 个相同种子，9 个模型共 **324 次训练/数据集**，三个类别共 972 次。先不加 `--execute` 查看计划。预算紧张可在运行前统一改成 `--budget 6`；不要只给 AMORE 更多预算。候选空间在 `tune_revision.py`，固定搜索随机数 314159，首个候选包含仓库默认配置。预算公平指候选数与训练上限一致，不代表覆盖所有方法最佳参数；论文必须公开范围和实际试验数。

若筛查决定采用 rev_00 或单支版本，先固定该决定，再运行例如 `python experiments/tune_revision.py --datasets baby --variants rev_00 --budget 12 --seeds 999 2024 2025 --execute`，提供相同预算；冻结时把它加入 variants。不要在测试集上挑选最终模型。本轮默认主表候选仍是原始修复完整版 rev_11，以便审查原论文主张。

冻结 Baby 的验证选择（输入只给 tuning/runs，不要把 anchor/baselines 的重复种子合并进去）：

```bash
python experiments/freeze_revision.py results/revision/tuning/runs --dataset baby --variants mm_bpr mm_lightgcn mm_bm3 mm_mgcn mm_freedom mm_lgmrec mm_mentor gume rev_11 --output revision_outputs/locks/baby.json
python experiments/revision_suite.py --stage final --datasets baby --lock revision_outputs/locks/baby.json --execute
```

final 才评估测试集，并做 5 次推理计时；最终五个种子一致。Sports、Clothing 分别调参、分别冻结、分别运行 final。冻结器拒绝测试集选择、合成数据、重复种子、不同源码/数据的混合；final 检查数据哈希、源码哈希、epochs/patience 一致。修改代码后必须重新形成同版本实验，不能假装复用旧结果。

## 5. 损失、敏感性、噪声与冷启动

```bash
python experiments/revision_suite.py --stage losses --datasets baby sports clothing --execute
python experiments/revision_suite.py --stage sensitivity --datasets baby sports clothing --execute
python experiments/revision_suite.py --stage robustness --datasets baby sports clothing --execute
python experiments/revision_suite.py --stage cold --datasets baby sports clothing --execute
```

| 阶段 | 每数据集默认训练数 | 回答什么 |
|---|---:|---|
| anchor | 20 | 原始/修复/新入口区别 |
| baselines | 45 | 默认配置的技术筛查，不能替代调参 |
| affinity | 35 | 2×2 因子 + 打乱/常数/停止梯度 |
| losses | 50 | 8 个损失项的单项移除及矩/InfoNCE 2×2 |
| sensitivity | 100 个计划项 | vt、两个 L2、阈值分别 5 点；重复完整设置 resume 后只运行一次 |
| robustness | 180 | 3 模型 × 3 干预类型 × 4 强度 × 5 种子 |
| cold | 15 | 3 模型 × 5 种子，20% 物品训练交互全部移除 |
| confirm | 35 | 原始/修复与四种亲和力配置的额外共同参照 |
| final | 45（9 模型） | 冻结后五种子主表测试与效率 |

以上不是必须一次性全跑的任务清单。先 anchor 和 Baby affinity，再确定是否值得支付调参与完整实验成本。脚本默认验证集探索；不要把探索阶段的验证值直接填入论文最终测试表。

噪声：只在训练集加入独立固定的错误交互（比例相对干净训练边数），排除 train/validation/test 已知正例。此处使用 held-out 正例排除仅为构造已知假边的离线压力测试，不是部署策略。图与模型随后重建。模态干预分别对图像/文本选定物品打乱特征；随机置换可能有少数固定点，报告的是选定比例。所有模型使用相同干预随机数；查看 `intervention`、特征扰动配置与数据哈希。

冷启动：选定训练中出现过的物品，移除其全部训练边，保留目录和模态特征，重建图。是“目录内零交互物品”测试，不是新用户/完全未见物品归纳推理。候选掩码固定为原始训练正例，噪声边和被保留测试正例不会改变候选全集；实际 eligible 用户/正例数随分组公开。

最终压力实验使用主表冻结配置，直接运行：

```bash
python experiments/revision_suite.py --stage stress-final --datasets baby --lock revision_outputs/locks/baby.json --execute
```

自动选取锁中存在的 GUME、MENTOR 和 rev_00/10/01/11，保持各自选定配置，执行 4 点交互噪声、4 点图像/文本扰动及 20% 冷物品实验。默认主表锁包含 GUME/MENTOR/rev_11，故为 3 × 13 × 5 = 195 次训练/数据集。输出分别在 `stress_final_interactions`、`stress_final_image`、`stress_final_text`、`stress_final_cold`；用对应目录绘图和分组分析。先查看不带 `--execute` 的计划。不得根据压力测试成绩重新调参。验证阶段 robustness/cold 是固定默认配置探索，不能替代这一步。

## 6. 统计、绘图、TensorBoard

```bash
python experiments/analyze_results.py results/revision/final --split test --metric recall@20 --baseline mm_mentor
python experiments/revision_data_summary.py --data-path data --output revision_outputs/datasets.json
python experiments/analyze_results.py results/revision/final --split test --metric ndcg@20 --baseline mm_mentor
python experiments/plot_revision.py results/revision/final --split test --output revision_outputs/main
python experiments/plot_revision.py results/revision/noise_interactions --axis interaction_noise_rate --output revision_outputs/noise_interactions
python experiments/plot_revision.py results/revision/noise_image --axis feature_corruption_rate --output revision_outputs/noise_image
python experiments/plot_revision.py results/revision/sensitivity_vt_loss --axis vt_loss --output revision_outputs/vt
python experiments/revision_cohorts.py results/revision/final --data-path data --output revision_outputs/cohorts
tensorboard --logdir results/revision --host 127.0.0.1 --port 6006
```

cohort 脚本读取每个 run 的原始数据路径和保存的 top-K。若结果移动到另一台机器，传入新的数据根目录，交互文件哈希必须匹配。

本地终端：`ssh -L 6006:127.0.0.1:6006 <你的服务器别名>`，浏览器访问 `http://localhost:6006`。TensorBoard 左侧可多选不同模型、种子、超参数 run，比较训练损失、验证曲线和诊断量。曲线选择是验证集；最终测试不用于早停。统计图输出独立 PNG/PDF 和 CSV，图注与解读见 [FIGURES.md](FIGURES.md)。未获得真实结果前不会生成论文性能图。

主文报告 Recall/NDCG@10/20、均值±样本标准差。主要确认指标预设 Recall@20，NDCG@20 为次要；分析器每次调用的 Holm 校正不覆盖跨数据集/跨指标的所有假设，若合并宣称总体显著需再定义并校正整个检验族。五种子 t 检验样本很少，CI/成对差值比单个 p 值更重要；固定数据切分上的种子不表示跨数据集泛化不确定性。

## 7. 实验结束后回传什么

保留 `result.json`、`history.json`、完整训练日志、事件文件、锁、计划、源码 commit 和运行环境。分组分析还需 `test_rankings.npz` 与对应数据；这不是只靠最终一行日志能重建的。将 `revision_outputs` 和 `experiment_reports` 中导出的摘要同步给本地，保留服务器原始 results 目录。

最终填入 Word 前逐项完成 [reviewer_matrix.json](reviewer_matrix.json) 的 E01–E08。四个缺失的相关基线与跨域意见仍应诚实解释，不能把“已经提供脚本”回复成“已经获得提升”。
