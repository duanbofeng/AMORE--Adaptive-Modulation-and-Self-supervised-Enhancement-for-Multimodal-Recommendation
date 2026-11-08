# 本地实现验证记录

## 2026-11-08：回到原完整 AMORE

- 34 项测试通过，含全部 72 个登记候选的基础回归。本轮仅开发原 AMORE 的复跑和修正/精简版本。
- 原始代码已按 SHA-256 保存在 `src/models/archive/`；适配器与快照的 k=10 图、参数初始化、前向、含噪声总损失及梯度通过等价测试。
- fixed 保持有效原始前向；两张无梯度用户表精确删除。compact 只执行一次 UI 传播分支，全部保留参数有有限梯度。
- 验证实际短 batch 正则分母、InfoNCE 去重/单 ID/极端温度、signed 图及数值稳定的 grouped softmax、checkpoint 与命名 TensorBoard 损失。
- 四个本轮候选各两轮合成子进程运行成功。服务器 SSH 可达但身份验证失败，未执行正式数据训练；使用原有用户同步运行流程，命令见 [ORIGINAL_AMORE.md](ORIGINAL_AMORE.md)。

## 2026-11-08：AMOREX 框架更新

- Python 3.12 / PyTorch 2.12 CPU，28 项测试通过；全部 67 个候选的前向、反向和完整排序检查通过。
- 新增单兴趣与 CONTENTCF 等价、均匀多槽不改变内容尺度、排除正例后的数值与目标梯度、度数 0/1 回退、候选依赖评分、分块/逐对/全排序一致性、原型及投影梯度、训练正例负采样屏蔽、单负例 CE/BPR 等价测试。
- 新模型训练、验证选 checkpoint、恢复、最终 test 排序导出和 TensorBoard 诊断均经过合成数据测试。
- rebuild 六组各 2 epochs 的独立子进程合成数据短跑全部完成。正式数据、服务器 PyTorch 1.12 / CUDA 尚待验证；本地测试不是效果提升证据。
- 对照与服务器命令见 [AMOREX_EXPERIMENTS.md](AMOREX_EXPERIMENTS.md)。以下是之前的验证记录。

日期：2026-11-07。分支：branch_2。验证环境：Python 3.12，PyTorch 2.12 CPU，NumPy 1.26.4，SciPy 1.17.1，pandas 2.3.3。使用独立临时虚拟环境，没有修改仓库 requirements 或用户已有 Python 环境。

## 已完成

- `python -m unittest discover -s tests -v`：10 项测试通过。其中遍历 37 个候选，检查有限 loss、反向梯度、排序输出。
- 验证分块 KNN 与小型稠密计算相符；零向量不产生 NaN；特征变化使图缓存失效。
- 验证 affinity_strength=0 与默认 AMORE 输出一致；单交互用户无自相似确认；用户内部打乱保持权重分布。
- 验证协同图仅包含训练边；冻结特征无优化梯度；图不进入参数 checkpoint。
- 验证真实训练、验证最优检查点恢复、最终 test top-k 导出；用相反配置的模拟验证分数检查网格只选 validation 最优且仅执行一次 test。
- 验证 GUME 的训练子采样与 validation-only 路径。
- 独立子进程 smoke：gume、amore_base、amore_local、amore_cluster_init、amore_shared_refined，3 seeds，每次 2 epochs，共 15 次合成训练全部成功。
- 多 seed 汇总、配对差值/区间、test 分组统计可生成 JSON；resume 跳过上述全部 15 个已完成运行。
- all suite 可生成 111 次运行的计划（37 variants × 3 datasets × 1 seed），没有启动正式训练。
- `git diff --check` 通过。

## 尚未完成的效果验证

本地 `data/` 没有正式数据，未进行 Baby/Sports/Clothing 的真实训练，未验证 CUDA 速度、真实显存、最优超参数或推荐质量提升。合成数据被标记为 synthetic_fixture，数值不用于效果结论。

KNN 分块减少峰值存储，但仍是二次计算量。GPU、数据规模和特征维度会影响首次构图成本。论文中的正式 baseline 复现和调参预算仍需在服务器执行。

目前没有把新假设写入原 Word 论文的最终结论；`PAPER_REVISION.md` 是待真实实验支撑的修订工作稿。改动未自动提交或推送 GitHub。
