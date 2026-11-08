# 第一轮执行记录

**Baby core 和 GUME 融合初筛已完成。当前请看 [AMOREX 独立框架实验流程](AMOREX_EXPERIMENTS.md)，按短跑 → 六组完整训练 → 汇总执行。**

以下保留第一轮命令供复查。

现在只需要完成：环境检查 → Baby 短跑 → Baby 核心对照 → 汇总结果。先得到这一轮的证据，再决定下一轮实验。`EXPERIMENTS.md` 提供完整选项，不需要从头到尾逐条执行。

所有命令都在服务器的项目根目录、已激活的训练环境中执行。前一步成功后再执行下一步。以下使用 GPU 0，正式数据应已放在 `data/baby/`，包含 `baby.inter`、`image_feat.npy`、`text_feat.npy`。

## 第一步 同步代码并检查环境

```bash
git switch branch_2
git pull --ff-only origin branch_2
python -m unittest discover -s tests -v
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

看到测试 `OK` 且 CUDA available 为 True 后继续。单元测试使用合成数据和 CPU；下一步检查实际数据与 GPU。

已有可训练的服务器环境先直接使用。`requirements.txt` 是较新环境的依赖要求，新建环境时才执行 `python -m pip install -r requirements.txt`；现有 PyTorch 1.12/CUDA 11.6 环境先保留。本次已适配旧版 `torch.load`：训练结束恢复最佳权重和最终 test 恢复均使用兼容加载器。该旧环境仍需完成短跑确认。

## 第二步 用 Baby 做两次短跑

```bash
python experiments/run_suite.py --variants gume amore_base --datasets baby --seeds 999 --epochs 2 --patience 2 --gpu 0 --output results/00_check --execute --resume
```

这条命令自动依次运行 GUME 和精简 AMORE，各 2 个 epoch。目的是确认数据、显存和训练流程正常，不能用这两次短跑判断效果。首次 KNN 构图仍可能花时间。

成功标志：命令正常结束，`results/00_check/failures.json` 为 `[]`。若出现 FAILED，查看提示路径中的 `console.log`，先处理错误再继续。

## 第三步 运行当前唯一的一轮正式筛选

```bash
python experiments/run_suite.py --suite core --datasets baby --seeds 999 --gpu 0 --output results/01_core_baby --execute --resume
```

只需执行这一个命令，脚本会依次完成以下 7 次训练：

| 顺序 | 版本 | 目的 |
|---|---|---|
| 1 | gume | GUME 对照 |
| 2 | gume_lean | 检查整体删减后的效果 |
| 3 | amore_base | 精简 AMORE，不用亲和力 |
| 4 | amore_global | 全局平均亲和力控制 |
| 5 | amore_item | 物品级亲和力 |
| 6 | amore_local | 局部用户历史边亲和力 |
| 7 | amore_shuffle | 用户内部打乱亲和力的控制 |

每次最多 1000 epochs，按验证集早停，达到停止条件就自动开始下一版本。不会同时启动 7 个训练，也不需要手动逐个切换模型。这一轮不加 `--test`。

`--resume` 会跳过代码、配置、数据一致且已完成的运行。中途未完成的单次训练会从头重跑，不是从中断 epoch 恢复。保持相同代码与参数，再执行同一命令即可。

训练会占用当前终端。长时间运行时使用你平时的服务器持久终端或作业提交方式；`--resume` 本身不会让进程在 SSH 断开后持续运行。

## 训练期间 查看曲线

在服务器另一个终端启动，不需要等第三步结束：

```bash
tensorboard --logdir results/01_core_baby --port 6006
```

本地访问方式沿用前面的 SSH 6006 端口转发或 VS Code 端口转发。主要勾选同一 seed 下各版本的 `validation/recall@20` 和 `validation/ndcg@20`。短跑结果在另一个目录，不会混入这张对比图。

训练终端停留在 `[1/7] baby gume seed=999` 本身是正常的：每轮输出写入该运行的 `console.log`，完成后才显示 `[2/7]`。在另外的服务器终端查看近期进度：

```bash
tail -n 20 results/01_core_baby/baby/gume/seed_999_*/console.log
```

若 TensorBoard 2.14 报 `MessageToJson() got an unexpected keyword argument 'including_default_value_fields'`，是 protobuf 接口不兼容。若还出现 `Failed to communicate with data server`，可关闭快速加载。只在 TensorBoard 窗口按 Ctrl+C，保留训练窗口运行，然后在服务器项目根目录执行：

```bash
python -m venv ~/venvs/amore-tensorboard
~/venvs/amore-tensorboard/bin/python -m pip install 'tensorboard==2.14.0' 'protobuf>=4.25,<5' 'numpy<2' six
~/venvs/amore-tensorboard/bin/tensorboard --logdir results/01_core_baby --port 6006 --load_fast=false
```

这是专门读取日志的独立环境，不需要安装 PyTorch 或 TensorFlow。之后仍通过原 6006 端口打开页面。首次安装成功后，日常只需执行第三条启动命令。

如果此前安装后启动报 `ModuleNotFoundError: No module named 'six'`，执行 `~/venvs/amore-tensorboard/bin/python -m pip install six`，然后重新执行启动命令。必须使用这个独立环境的 Python，不能只在训练环境中安装。

依据：TensorBoard 官方仓库的 [protobuf 接口兼容问题](https://github.com/tensorflow/tensorboard/issues/6808)及[数据服务连接失败处理](https://github.com/tensorflow/tensorboard/issues/5384)。这些 TensorBoard 页面错误本身不代表独立训练进程失败；训练状态以训练日志为准。

## 第四步 汇总并讨论下一轮

第三步全部完成后执行：

```bash
python experiments/analyze_results.py results/01_core_baby --split validation --metric recall@20
python experiments/analyze_results.py results/01_core_baby --split validation --metric ndcg@20
```

先把两张终端汇总表发回讨论。更完整的结果分别保存在：

- `results/01_core_baby/analysis_validation_recall_20.json`
- `results/01_core_baby/analysis_validation_ndcg_20.json`

当前到这里即可。不要根据一个数据集、一个 seed 的结果直接宣布哪个模块有效，也不需要立即启动 `prune`、`mechanisms`、`factorial`、全部参数网格或最终 test。

后续顺序是：依据这一轮选择待核查的方向 → 在 Sports/Clothing 及额外 seeds 上验证 → 对有希望的模块做删减、组合和有限调参 → 锁定方案 → 做最终多 seed test 与论文实验。若亲和力暂时没有收益，下一轮可以优先验证图质量与初始化；若表现出收益，则先补尺度与随机种子控制。具体候选数量由这一轮结果决定。
