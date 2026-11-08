# 当前入口：先复跑最初 AMORE，再修正和精简

按用户要求停止 AMOREX、多兴趣、GUME＋Ego/SG 等探索。旧文件留作实验记录，不继续安排这些方向。当前工作围绕 `old_model/src/models/gume.py` 中的完整 AMORE。

## 1. 先确定“原版”究竟指什么

旧 `models/amore.py` 只是继承旧 `models/gume.py`，没有独立网络。旧 `main.py` 默认 `--model gume`，但 `AMORE.yaml` 与 `GUME.yaml` 参数不同：

| 设置 | 显式 `-m amore` | 旧 main 默认 `-m gume` |
|---|---:|---:|
| affinity_threshold | >0.1 | >0.5 |
| um_loss | 0.1 | 0.01 |
| um_temp | 0.4 | 0.1 |
| residual_alpha（原代码无效） | 0.1 | 0.2 |

当前默认复跑显式 AMORE 配置，variant 为 `amore_original`，实际模型名为 `amorelegacy`。若要重现旧 main 的默认设置，使用 `amore_original_main`；这两个名字均运行旧完整 AMORE 的网络，**不是当前 `src/models/gume.py` 中的 GUME 基线**。

旧日志 `gume-baby-Oct-22-2025-21-41-10.log` 没有亲和力配置，且 bm_loss=0.01，与现在保存的旧 AMORE 配置不一致，因此不能用它断言哪一套配置对应论文旧成绩。这里复跑的是目前实际保存在 old_model 的实现，不声称已经找回更早、缺失的代码版本。

### 原始代码已保存到可同步目录

- `src/models/archive/amore_original.py`：逐字节保存的旧 `gume.py`；原始前向和损失直接从它继承。
- `src/models/archive/AMORE.yaml` / `GUME.yaml`：原始配置快照。
- `src/models/archive/manifest.json`：来源文件 SHA-256。
- `src/models/amorelegacy.py`：运行适配器。
- `src/models/amoreclean.py`：修正和精简版本。

`old_model/` 原本被 Git 忽略。新的运行入口仅依赖 `src/` 中的快照，不要求服务器额外拥有 old_model，也没有覆盖它。

### 复跑边界

原始前向、原始总损失及其系数保留，包括原版中有问题的行为。适配只做以下工程处理：

1. 原始 dense 全物品余弦矩阵改为分块精确 top-k。保留自环、负余弦和已选中的零权邻居，不使用新版 GUME 的负数截断图。
2. 图交集按真实行索引计算，k=10 时对应原程序新建图的语义；修正原来硬编码 10 对其它 k 的错误。
3. 图缓存按原始特征内容和 k 命名，不复用来源不明的旧 `inter.json` / `.pt`。零特征或非正有符号图度数明确报错，不悄悄改变数学定义。相似度并列时 top-k 集合可能受设备/计算精度影响。
4. 同用户 softmax 和物品平均调制改为等价张量运算，避免每用户 Python 循环；保留阈值判定、再次注意力和按物品平均的原始机制。使用 PyTorch 1.12 已支持的 `scatter_reduce_`，无需安装 torch_scatter。[PyTorch 1.12 发布说明](https://github.com/pytorch/pytorch/releases/tag/v1.12.0)
5. 使用现有验证选模、checkpoint、TensorBoard 和结果导出；最终推理缓存不改变分数。**没有恢复旧 quick_start 用 test 挑配置的做法。** 因此这不是旧实验报告整套评估流程的逐位重现。

本地已在 k=10 的合成数据上对照快照，验证参数初始化、图、完整前向、含噪声总损失和共同参数梯度的一致性。不同设备浮点误差及硬阈值附近的离散变化仍需注意。

## 2. 代码与论文核对出的具体问题

| 项目 | 原代码实际行为 | 此次处理 |
|---|---|---|
| 两张 extended modality user 表 | 构造后从未进入前向或损失，无梯度 | 原版保留；fixed/compact 删除，保持共同参数初始值一致 |
| PAMD 配置 | 没有 PAMD 模块或对应损失调用 | 不补造模块；原版记录，修正版不暴露这些开关 |
| residual_alpha / fine_grained_weight | 增强残差算出后没有用于输出 | 原版保留；修正版直接删除无效运算，不另加残差路径 |
| affinity_gate / multimodal_user | 计算后未用于输出 | 修正版删除 |
| affinity_decouple_weight | 没有被模型读取 | 不当作有效超参数 |
| Adaptive BPR | 默认关闭；打开时训练分数与推理内积不一致 | 修正版只允许与推理一致的标准 BPR |
| InfoNCE | 同 ID 重复出现时被视为其它负例；直接 exp 再 log | 修正版先按 ID 去重，使用交叉熵/logsumexp |
| 正则分母 | 用配置 batch_size，最后短 batch 缩小正则 | 修正版使用实际 B |
| VT | 全部元素的一个均值、一个方差差值 | 明确定义为标量矩匹配，不称为逐样本语义对齐或完整分布距离 |
| 噪声 | 均匀随机数，归一化后乘符号；不是高斯噪声 | fixed 保留其统计形式，compact 删除这项训练任务 |
| BM 扩展对齐 | 原始 BM 外又对扩展视图做一次 BM | fixed 保留并单列权重，compact 删除 |
| 亲和力 | 用户内 softmax 后阈值判定，消息最后平均回物品 | 原版/fixed 保留；compact 只用于 ID 分支，另设完全关闭对照 |

“两张未使用用户表”和“未进入输出的残差”等是可由执行路径证明的无效计算。其余辅助损失和图分支是否多余，只能由实验判断；compact 是一个删减组合候选，不能用一次组合对比断言其中每项都无用。

文献核对的结论：

- **GUME** 的图增强、显式与扩展模态视图、BM/UM 解释了旧主干和辅助目标的来源。保留这些主干作为原版，不再称它们都是新增创新。
- **MGCN** 的行为净化、共享/特有属性融合及行为—模态对齐，支持保留旧模型的编码与粗细融合主干，单独检验扩展视图是否必要。
- **GUME 论文的 VT 写均值与标准差，旧 AMORE 代码及 AMORE 稿写均值与方差**。此次明确采用 AMORE 的方差矩匹配，并在修正版使用总体方差（unbiased=False）；不声称精确复现 GUME 该公式。
- **EgoGCN** 的边调制不能直接证明此处“先算边、再平均回物品”的亲和力是可靠性估计。此次不再引入另一套 Ego 网络，保留原实现作为可检验假设。
- **IIMRec / SG-URInit** 分别涉及图处理与初始化；它们不能修复上述损失定义或无效代码，因此此次不增加它们的模块。

目前不改写 Word 论文中的效果表。应先跑出与明确代码/配置对应的原版结果，再决定最终方法和贡献表述。

## 3. 两种修正版保留哪些东西

共同保留：可训练原始图文嵌入、ID 引导特征净化、原始 signed KNN 图、自环、双模态交集增强 UI 图、LightGCN 传播、粗细属性融合，以及融合后同一个内积评分。图深度和主要默认超参数不重新搜索。

### amore_fixed：结构保持，损失实现修正

保留完整 AMORE 的有效前向和所有现有辅助项，只删除无效表/运算，修正数值计算、ID 去重及批大小。无论是否训练，最终排序嵌入与相同初始权重的原版一致；训练轨迹会因损失修正而改变。

### amore_compact：主干保留，取消额外扩展训练分支

- 保留原始亲和力硬调制，但只在协同 ID 分支执行一次。
- 粗粒度融合保留原来的基础注意力，移除第二套 affinity_guided_coarse 网络和混合系数。
- 删除只供辅助损失使用的 image/text 扩展 UI 传播、扩展 BM、UM cross、两项噪声对比及扩展表示正则。
- 保留基础 BM、VT 矩匹配及 BPR、表示 L2。

其有效 UI 传播分支从三个降至一个。原始高维内容仍可训练，仍然是参数和显存的大头；不能把它宣传成极小模型或提前承诺速度提升。`amore_compact_no_affinity` 只在 compact 基础上关闭亲和力，用于决定最终是否保留它。

## 4. 修正版损失的准确定义

设 B 为实际三元组批大小，E 为最终融合嵌入，S 为融合语义表示，C 为协同表示，X 为扩展模态表示。`U_B` 和 `I_B` 分别是批内去重用户、去重正例物品集合。

```text
s(u,i) = <E_u,E_i>                              # 训练和推理相同
L_BPR = mean_(u,i,j) softplus(s(u,j)-s(u,i))
R_E = [sum ||E_u||² + sum ||E_i||² + sum ||E_j||²] / (2B)
L_VT = |mean(V)-mean(T)| + |var_pop(V)-var_pop(T)|

NCE(A,B,Q;tau) = mean_q [-log exp(cos(A_q,B_q)/tau)
                                / sum_{r in Q} exp(cos(A_q,B_r)/tau)]
L_BM = NCE(S,C,U_B;tau_bm) + NCE(S,C,I_B;tau_bm)

L_compact = L_BPR + reg_weight_1*R_E + vt_loss*L_VT + bm_loss*L_BM
```

分母中的候选 B_r 随 r 改变，包含正确正例 B_q；使用 logsumexp 计算，只有一个独立 ID 时 NCE 为零。去重只针对对比目标，BPR 仍按实际训练三元组计算。该改变会去掉批内重复次数带来的额外对比权重，不能称为与旧 loss 完全等价。

这里的 R_E 是**最终表示正则**，沿用旧 AMORE 的正则位置，只修正实际 B；不是所有模型参数的全局 L2，也不是“初始 ID L2”。这样不会在修正过程中悄悄更换正则目标。

fixed 额外保留：

```text
L_BM_ext = NCE(X,C,U_B;tau_bm) + NCE(X,C,I_B;tau_bm)
L_cross = NCE(X,S,U_B;tau_um)
L_noise = NCE(noise_1(S),noise_2(S),U_B;tau_um)
          + NCE(noise_1(X),noise_2(X),U_B;tau_um)
R_X = sum_(positive batch items) ||X_i||² / (2B)

L_fixed = L_compact + bm_extended_loss*L_BM_ext
                       + um_loss*L_cross + noise_loss*L_noise + reg_weight_2*R_X
```

fixed 的默认附加权重分别为 0.005、0.1、0.1、0.1，与显式原 AMORE 的各项有效系数对应；不是新调出的最佳系数。噪声仅对去重后的批内用户生成，与原来全用户生成再取 batch 的采样顺序不同，但同一扰动分布。compact 这些项为零，相关传播也不执行。

TensorBoard 的 `train/bpr`、`embedding_l2`、`vt_moments`、`bm`、`bm_extended`、`um_cross`、`um_noise`、`extended_l2` 均记录**乘权重后的项**；八项相加就是总 loss。原版保持原函数返回一个总 loss，避免为拆分日志改变它的计算顺序。

## 5. 服务器只按这个顺序执行

本次尝试了用户提供的 SSH 地址，服务器可达但认证失败：`Permission denied (publickey,password)`。本地没有正式数据，所以尚未完成正式 Baby 训练。以下由用户同步代码后在原服务器项目根目录执行。

### ① 原版两轮工程检查

```bash
conda activate torch1.12_cuda11.6
python -m unittest discover -s tests -v
python experiments/run_suite.py --variants amore_original --datasets baby --seeds 999 --epochs 2 --patience 2 --gpu 0 --output results/30_original_check --execute --resume
```

### ② 先完整运行原版

```bash
python experiments/run_suite.py --variants amore_original --datasets baby --seeds 999 --gpu 0 --output results/31_original_baby --execute --resume
python experiments/export_results.py results/31_original_baby --output experiment_reports/31_original_baby.json
```

默认至多 1000 epochs，validation Recall@20 连续 20 次未改善早停。完整运行包含实际训练、验证选 checkpoint、恢复最优模型和保存结果。开发阶段不评估 test；若需要固定原版的最终 test 报告，另外使用独立输出目录并显式添加 `--test`，不要用 test 来决定删哪些模块。

如果你明确以前一直没有给旧 main 加 `-m amore`，把上面 variant 换成 `amore_original_main`，并使用不同输出目录；不要混淆两套旧配置。

### ③ 原版完成后，再跑修正/精简对照

```bash
python experiments/run_suite.py --suite original_refine --datasets baby --seeds 999 --gpu 0 --output results/31_original_baby --execute --resume
python experiments/analyze_results.py results/31_original_baby --baseline amore_original --split validation --metric recall@20
python experiments/analyze_results.py results/31_original_baby --baseline amore_original --split validation --metric ndcg@20
python experiments/export_results.py results/31_original_baby --output experiment_reports/31_original_baby.json
```

代码、数据、预算不变时会跳过完成的原版，只新增 fixed、compact、compact_no_affinity。该顺序先建立原版参照，不再一次运行十几组。若只想先看原版结果，做到②即可，把 JSON 同步回来后再安排③。

另一个终端查看曲线：

```bash
~/venvs/amore-tensorboard/bin/tensorboard --logdir results/31_original_baby --port 6006 --load_fast=false
```

沿用已有 SSH 6006 端口转发。`results/` 被 Git 忽略，真正要同步回来的是导出的 `experiment_reports/31_original_baby.json`，它包括配置和训练曲线。

## 6. 判断标准与验证状态

先看原版是否正常收敛，再看 fixed 对损失实现修正后的变化。compact 若明显下降，说明这个删减组合损害效果，不能继续宣称删除的是冗余；再按 full-fixed 为起点逐项定位，不继续另造框架。若 compact 接近或更好，才增加匹配 seeds 并比较参数、总训练时间和显存。亲和力是否保留由 compact 与 no_affinity 的重复对照决定。

本地 34 项测试通过，含全部 72 个候选的基础计算回归。新增验证包括原始快照的前向/损失/梯度等价、原始 signed KNN、极端值 grouped softmax、无效参数无梯度、fixed 的输出保持、实际短 batch 正则、InfoNCE 去重和稳定性、compact 真正跳过扩展传播、checkpoint 和命名 TensorBoard 损失。四个本轮候选各两轮合成子进程短跑成功。上述是工程验证，不是正式数据效果或服务器 PyTorch 1.12/CUDA 验证。
