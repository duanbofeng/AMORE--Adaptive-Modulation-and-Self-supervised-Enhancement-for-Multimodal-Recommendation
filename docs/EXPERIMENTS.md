# 服务器实验运行手册

所有命令从仓库根目录执行。使用 Python 3.10+，建议先在独立环境安装依赖。本地验证环境为 Python 3.12、PyTorch 2.12 CPU；GPU 和正式数据集尚未在本地验证。

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

数据目录沿用原仓库：`data/baby/`、`data/sports/`、`data/clothing/`，每个目录有对应 `.inter`、`image_feat.npy`、`text_feat.npy`。交互文件已有 `x_label=0/1/2`，分别代表 train/validation/test。特征第 i 行必须对应 itemID=i，不在训练程序里重新映射 ID。

## 1 先验证运行环境

```bash
python tests/make_toy_data.py --output /tmp/amore_toy_data
python experiments/run_suite.py --variants gume amore_base amore_local amore_cluster_init amore_shared_refined --datasets baby --seeds 999 --data-path /tmp/amore_toy_data --overrides experiments/smoke.json --epochs 2 --patience 2 --output results/smoke --cpu --test --execute
python experiments/analyze_results.py results/smoke --metric recall@3 --split test
```

这是随机生成的合成数据，仅检查管线，不得写进论文效果表。生成器拒绝覆盖非空目录。要在 CUDA 上做 smoke test，去掉 `--cpu`，指定 `--gpu 0`。

## 2 第一轮筛选

```bash
# 默认只生成计划：3 datasets × 7 variants × 1 seed = 21 runs
python experiments/run_suite.py --suite core --output results/core

# 查看 results/core/plan.json 后开始；默认不测 test
python experiments/run_suite.py --suite core --output results/core --gpu 0 --execute

# 重启已中断实验；已完成且代码、参数、数据一致的运行会跳过
python experiments/run_suite.py --suite core --output results/core --gpu 0 --execute --resume

python experiments/analyze_results.py results/core --baseline amore_base --split validation
```

每个运行的 `console.log` 保存完整输出。`failures.json` 列出失败项；出现失败最终返回非零退出码。OOM 等失败不要填成 0 分，应修复后重跑或记录不能运行的约束。

运行目录名包含代码与参数指纹；代码变化会产生新目录。分析脚本不会把不同配置混为同一个均值，但建议每次明确实验阶段使用新的输出根目录。

## 3 删减和机制验证

```bash
python experiments/run_suite.py --suite prune --datasets baby sports clothing --output results/prune --execute
python experiments/run_suite.py --suite mechanisms --datasets baby sports clothing --output results/mechanisms --execute
python experiments/run_suite.py --suite factorial --datasets baby sports clothing --output results/factorial --execute
```

先用单 seed 筛掉无效方向，再给少量候选增加 seed。`all` 会包含全部候选及重复基准，不是推荐的首轮运行方式。详细设计与归因边界见 [AMORE_RESEARCH.md](AMORE_RESEARCH.md)。

## 4 调参并保持预算可比

`--sweep` 接受 JSON 列表，每个值是独立运行。多个 `--sweep` 做笛卡尔积，请先不加 `--execute` 检查总量。

```bash
# 首先比较相同语义强度下 base 与 local；beta=0 提供完全关闭控制
python experiments/run_suite.py --variants amore_base amore_local --datasets baby --sweep 'semantic_weight=[0.1,0.3333333333333333,0.5,1.0]' --output results/semantic_scale --execute
python experiments/run_suite.py --variants amore_local --datasets baby --sweep 'affinity_strength=[0.0,0.25,0.5,0.75]' --output results/affinity_strength --execute

# 给 GUME 与精简对照相同学习率搜索预算
python experiments/run_suite.py --variants gume gume_lean amore_base --datasets baby --sweep 'learning_rate=[0.0005,0.001,0.002]' --output results/learning_rate --execute
```

选择超参数只能看 validation，不能从 test 表回选。不同数据集可以选不同超参数，但搜索预算与规则应提前固定。分析会为每种配置单列 setting，避免把不同参数或受损比例当成多个 seed。

自定义组合用 JSON 文件，例如：

```json
{
  "affinity_mode": "local",
  "affinity_strength": 0.25,
  "graph_mode": "shared",
  "cooc_weight": 0.25,
  "ncer_strength": 1.0,
  "user_init": "history"
}
```

将它保存为 `experiments/selected_candidate.json`，再用 `--variants amore_base --overrides experiments/selected_candidate.json`。`overrides` 会覆盖选中 variant 的设置；不要把同一个候选 overrides 同时应用到想保留的原始基线。文件应随代码一起保存以记录最终选择。

## 5 稀疏性和模态受损

```bash
python experiments/run_suite.py --variants amore_base amore_local --datasets baby --seeds 999 2024 2025 --sweep 'train_fraction=[0.25,0.5,0.75,1.0]' --output results/sparsity --execute

python experiments/run_suite.py --variants amore_base amore_local --datasets baby --seeds 999 2024 2025 --sweep 'feature_corruption=["zero","shuffle"]' --sweep 'feature_corruption_rate=[0.1,0.3,0.5]' --sweep 'corruption_modality=["image","text"]' --output results/corruption
```

第二条先生成计划，不自动执行 72 次运行。根据筛选结果缩小范围后加 `--execute`。每种扰动与数据保留比例必须分别比较。默认训练 1000 epochs、patience 20；缩短预算的试跑不能直接替代同预算主表。

## 6 最终多 seed 验证

先锁定设计。下面命令用默认参数说明操作，并不意味着这些候选已被选中或调参最优：

```bash
python experiments/run_suite.py --variants gume gume_lean amore_base amore_local --datasets baby sports clothing --seeds 999 2024 2025 2026 2027 --output results/final --test --execute
python experiments/analyze_results.py results/final --split test --baseline amore_base --metric recall@20
python experiments/analyze_results.py results/final --split test --baseline amore_base --metric ndcg@20
```

各候选调参结果不同，应分别用对应 overrides 文件启动，保持相同 seed 和评估预算。第一次读取 test 后再改变设计，需要明确这已成为新一轮开发，不能继续把同一 test 当作从未使用的验证证据。

每个运行目录包含：

- `requested.json`：外部参数；`result.json`：实际生效完整配置、指标、版本和资源统计。
- `trial_000/best.pt`：validation 最佳参数、epoch 和验证指标；固定图不打包，恢复时需同一配置与数据重建。
- `trial_000/history.json`：逐 epoch 损失、时间和 validation 指标。
- `trial_000/test_rankings.npz`：最终 test 每位用户的 top-k，用于分组分析。
- TensorBoard events：训练与验证曲线。

```bash
tensorboard --logdir results/final --port 6006
```

分析输出 `analysis_validation_recall_20.json` 或 `analysis_test_recall_20.json`，包括多 seed 统计、匹配 seed 差值、95% t 区间、至少 3 个匹配 seed 时的配对检验和 Holm 校正，以及 test 上的稀疏用户/长尾物品分组。一次分析中同一数据集若有多个 baseline setting，会跳过配对推断，避免自动选最有利的 baseline。可以把已锁定的结果放在独立根目录分析。

用户级原始指标不作为独立 seed 做显著性检验。三个 seed 的区间可能很宽，应保留这一不确定性。测试分组输出里的 `users` 是该组纳入统计的用户数，不能把不同分母的 Recall 直接相加。

## 7 单独运行与同步

```bash
python src/main.py --model gume --dataset baby --no-test --output-dir results/manual_gume
python src/main.py --model amore --dataset baby --set affinity_mode=local --set affinity_strength=0.25 --set seed=999 --no-test --output-dir results/manual_local
```

服务器同步方式仍是 Git：本地审阅改动后提交到 `branch_2` 并 push，服务器在该分支执行 `git pull --ff-only origin branch_2`。本次代码修改未自动 commit/push。`data/`、`papers/`、`old_model/` 沿用原来的忽略设置，不会因本次改动自动上传。

KNN 缓存写入各数据集的 `graph_cache/`，指纹包含特征内容、k、自环策略。不会读取旧 `inter.json` 或旧 `image_adj_*.pt`。首次构图会比复用缓存慢；若比较预处理时间，应明确是冷缓存还是热缓存。训练中的图与语义路径每批重算；评估阶段每次只编码一次，然后按用户批次评分。
