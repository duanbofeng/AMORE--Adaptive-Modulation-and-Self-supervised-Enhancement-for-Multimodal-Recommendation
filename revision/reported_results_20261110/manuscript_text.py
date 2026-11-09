"""Evidence-specific manuscript replacements and an offline browsing report."""
import html
import json


def write_guidance(root,blocks,diffs,gains,drops,effects,figures):
    changes=[]
    def add(block,location,replacement,reason,original=None,condition=None):
        changes.append(dict(id=f'E{len(changes)+1:02d}',body_block=block,location=location,
                            original=original or blocks[block],replacement=replacement,reason=reason,
                            condition=condition))

    add(12,'Abstract，最后一句',
        'On the three Amazon datasets, AMORE obtains the highest reported score in 11 of the 12 dataset–metric comparisons. Relative to MENTOR, Recall@20 changes by −0.76%, +0.53%, and +2.12%, while NDCG@20 improves by 1.11%, 2.35%, and 2.72% on Baby, Sports, and Clothing, respectively.',
        '删除 consistently outperforms；保留 Baby Recall@20 的负结果。最高分只是当前表格中的点估计。',
        original='Extensive experiments on three real-world benchmark datasets demonstrate that AMORE consistently outperforms strong baselines, achieving improvements of up to 2.12% on Recall@20 and 2.72% on NDCG@20 over the strongest competing method.')
    add(24,'Introduction，贡献 1',
        '1) We propose AMORE, a multimodal recommendation framework that uses affinity information in graph propagation and modality fusion. We evaluate the contributions of these two uses separately and jointly on a common backbone.',
        '将显著提升及机制成立的预设，改成设计内容与实际提供的比较。')
    add(25,'Introduction，贡献 2',
        '2) We introduce affinity-based propagation modulation and affinity-guided fusion within the same framework. A factorial comparison separates their individual gains from the interaction contrast, allowing the usefulness of each component and their joint behavior to be examined separately.',
        '本次结果能支持比较个别使用位置，不能证明共享语义提取更精确。')
    add(26,'Introduction，贡献 3',
        '3) We combine coarse-to-fine feature fusion with a multi-objective alignment component. Module-removal experiments quantify their aggregate contributions to ranking performance; the contribution of each individual alignment loss requires a separate comparison.',
        '删除 eliminates/maximally 等绝对化结论；整块 MACA 消融不区分矩约束与 InfoNCE。Adaptive BPR 是否实际启用应另按运行代码核实。')
    add(27,'Introduction，贡献 4',
        '4) We compare AMORE with twelve baselines on Baby, Sports, and Clothing and report both module-removal and affinity-use factorial results. AMORE ranks first in 11 of the 12 reported dataset–metric comparisons. The factorial interaction is small and varies in sign across datasets, so we do not claim a uniform synergistic effect.',
        '三张表不包含受控噪声实验，因此不写 model robustness 已被证明。')
    add(126,'4.1.3 Baselines，开头',
        'We compare AMORE with twelve representative baselines, grouped into non-multimodal collaborative-filtering and graph recommenders, classical graph-based multimodal recommenders, and self-supervised or decoupling-oriented multimodal recommenders.',
        '加入 DHMRec 后共有 12 个基线，加 AMORE 共 13 行模型。')
    add(134,'4.1.3 Baselines，DualGNN 年份',blocks[134].replace('DualGNN (2021)','DualGNN (2023)'),
        '与本次表格及论文已有参考文献 [9] 的期刊年份统一。')
    add(139,'4.1.3 Baselines，SLMRec 年份',blocks[139].replace('SLMRec (2022)','SLMRec (2023)'),
        '与本次表格及论文已有参考文献 [11] 的期刊年份统一。')
    add(142,'4.1.3 Baselines，末段',
        'The overall comparison evaluates ranking performance against these baselines. The contribution of affinity use is examined separately through module-removal and factorial comparisons within AMORE.',
        '整体性能混合了多个结构差别，不能直接隔离亲和力贡献。')
    add(146,'4.2 Overall Performance，表 2 前',
        'Table 2 reports Recall@10, Recall@20, NDCG@10, and NDCG@20 on Baby, Sports, and Clothing. The highest and second-highest reported scores in each column are shown in bold and underlined, respectively. “Improv.” is calculated as 100 × (AMORE − best baseline) / best baseline; negative values indicate that AMORE performs worse than the strongest baseline.',
        '明确分母和负值含义；尚未给出种子数量，因此暂不写 mean ± SD。')
    add(150,'4.2，表 2 后引导句',
        'The results provide three observations about ranking performance and the scope of the comparison.',
        '删除预先建立机制解释的表述。')
    add(151,'4.2，First 段',
        'First, AMORE obtains the highest reported score on all four metrics for Sports and Clothing and on three of the four metrics for Baby. MENTOR is the strongest competing baseline in every column. On Clothing, AMORE improves Recall@10, Recall@20, NDCG@10, and NDCG@20 by 3.74%, 2.12%, 3.33%, and 2.72%, respectively. These aggregate improvements do not by themselves establish a specific benefit for long-tail items or cold-start cases, which require separate group-level evaluations.',
        '整体 Clothing 的提升不能作为长尾/冷启动验证。')
    add(152,'4.2，Second 段',
        'Second, the advantage over the strongest baseline is modest and metric-dependent. On Sports, the relative improvements are 0.92% and 0.53% for Recall@10 and Recall@20, and 2.44% and 2.35% for NDCG@10 and NDCG@20. These comparisons quantify overall ranking differences; they do not isolate whether the gains originate from propagation modulation, feature fusion, or alignment objectives.',
        '删除通过总体表证明减少噪声传播的因果推断。')
    add(153,'4.2，Third 段',
        'Third, the Baby results show a trade-off across metrics. AMORE achieves Recall@20 of 0.1040, compared with 0.1048 for MENTOR, a relative decrease of 0.76%. AMORE nevertheless has higher Recall@10, NDCG@10, and NDCG@20, with relative gains of 0.59%, 1.10%, and 1.11%. This mixed pattern supports competitive ranking performance rather than uniform superiority. The reported aggregate results do not establish statistical significance or explain the internal behavior of MENTOR.',
        '删除关于 MENTOR 过强对齐破坏偏好及 AGSD 保留微观语义的猜测。')
    add(155,'Ablation Study (RQ2)，开头',
        'Table 3 compares the full AMORE model with three module-removal variants to assess their contributions to Recall@20 and NDCG@20. We distinguish these module-level comparisons from the subsequent factorial study, which changes only where affinity is used within the same backbone.',
        '区分整块模块消融和亲和力使用位置；避免以 necessity 预设结论。')
    add(156,'Ablation Study，w/o ELAM 定义',
        'The w/o ELAM variant removes the affinity-based propagation modulation component. This comparison evaluates the ranking effect of this module; a decrease in ranking performance alone does not establish that the removed component identifies erroneous interactions.',
        'uniform aggregation 不一定符合归一化 GCN；表格也不能证明移除后噪声扩散。',
        condition='先核对服务器实际消融实现。若 use_affinity_modulation 同时关闭传播与融合先验，此变体不能称为只移除传播调制，必须据实际开关重写。')
    add(157,'Ablation Study，w/o CFDM 定义',
        'The w/o CFDM variant removes the coarse-to-fine fusion module. Its result measures the contribution of this module as a whole. This intervention differs from the fusion-affinity switch in the factorial study, where the fusion architecture is retained and only its use of affinity is changed.',
        '普通消融移除 CFDM，不能与图三的关闭融合亲和力等同。具体替代运算不能从分数猜测。',
        condition='与实际消融代码核对：需要补充模块移除后究竟使用拼接、求和还是其他替代运算。')
    add(158,'Ablation Study，w/o MACA 定义',
        'The w/o MACA variant removes the combined alignment component. This aggregate ablation assesses the contribution of the removed objectives together and does not distinguish the effects of contrastive learning, distribution matching, or their interaction.',
        '不能把整块移除结果写成每种辅助损失分别有效。',
        condition='需逐项确认被置零的损失系数及仍保留的正则项；未核实前不要写成纯 BPR。')
    add(162,'Ablation Study，结果引导段',
        'The full model has the highest reported score among the four variants for both metrics on all three datasets. We report relative decreases as 100 × (Full − ablation) / Full. The magnitude of these decreases differs across modules and metrics.',
        '将 clear decline 替换成明确的点估计关系及降幅定义。')
    add(163,'Ablation Study，ELAM 分析',
        'Removing ELAM reduces Recall@20 by 1.06%, 1.14%, and 1.78% on Baby, Sports, and Clothing, respectively. The corresponding NDCG@20 decreases are 5.05%, 1.34%, and 3.31%. The ranking impact is therefore metric-dependent, with a larger relative NDCG@20 decrease on Baby. These results support a contribution of the ELAM-containing configuration to the reported ranking scores, but do not by themselves verify an edge-denoising mechanism.',
        '使用本次最新 ELAM 数值；Baby 的 NDCG 下降比 Recall 下降明显。')
    add(164,'Ablation Study，CFDM 分析',
        'Removing CFDM decreases Recall@20 by 3.08%, 2.88%, and 4.16% on Baby, Sports, and Clothing, respectively, while NDCG@20 decreases by 3.08%, 3.44%, and 3.09%. CFDM removal has a larger Recall@20 effect than ELAM removal on every dataset, whereas ELAM removal has the larger NDCG@20 effect on Baby and Clothing. Thus, the relative contributions of the two modules depend on the ranking metric.',
        '避免一概说 CFDM 比 ELAM 更重要；不再用颜色/纹理/长尾机制作未经验证的解释。')
    add(165,'Ablation Study，MACA 分析',
        'Removing MACA produces the largest decline among the three module-removal variants on every dataset and metric. Recall@20 decreases by 13.17%, 11.18%, and 12.57%, and NDCG@20 decreases by 14.07%, 7.84%, and 11.48% on Baby, Sports, and Clothing, respectively. This identifies the combined alignment component as the most influential removal in this comparison. Additional loss-level ablations are needed to determine which objectives account for the effect.',
        '只能比较这组移除项的影响，不能推导 MACA 单项损失或前两模块为其生效原因。')
    add(186,'5 Conclusion and Future Work，第一段',
        'We presented AMORE, a multimodal recommendation framework combining affinity-based propagation modulation, coarse-to-fine fusion, and a multi-objective alignment component. On Baby, Sports, and Clothing, AMORE obtains the highest reported score in 11 of the 12 dataset–metric comparisons, with an exception for Recall@20 on Baby. Module-removal experiments show the largest performance decrease when the combined MACA component is removed. The affinity-use factorial comparison shows positive joint gains over the no-affinity reference, but small, mixed-sign interaction contrasts. These results support retaining the tested components while limiting claims about uniform synergy, verified denoising, and generalization beyond the evaluated Amazon categories.',
        '结论应同时容纳总体收益、MACA 整体贡献以及小且异号的交互项。')
    (root/'sentence_replacements.json').write_text(json.dumps(changes,ensure_ascii=False,indent=2)+'\n')

    insertion='''### Affinity use in propagation and fusion

To examine the two uses of affinity separately, we compare four configurations on the same backbone: 00 disables affinity use in both propagation and fusion, 10 enables propagation modulation only, 01 enables affinity-guided fusion only, and 11 enables both. These controls differ from removing an entire module in Table 3. Let y_ab denote Recall@10 for configuration ab. We compute the interaction contrast as Δ_int = y_11 − y_10 − y_01 + y_00, which measures the departure of the joint change from the sum of the two individual changes on the additive Recall@10 scale.

On Baby, the four scores are 0.0671, 0.0673, 0.0679, and 0.0682, respectively. The corresponding values are 0.0758, 0.0762, 0.0768, and 0.0770 on Sports, and 0.0677, 0.0685, 0.0684, and 0.0693 on Clothing. The joint gains y_11 − y_00 are +0.0011, +0.0012, and +0.0016, while the interaction contrasts are +0.0001, −0.0002, and +0.0001. The full configuration has the highest reported score on each dataset, but the interaction is small and not uniformly positive. On Sports, the joint gain is smaller than the sum of the two individual gains on this metric scale; this does not imply that the full configuration is worse than either single-use configuration.

The contrasts are calculated from values displayed to four decimal places. If these values were rounded to the nearest 0.0001, rounding alone could change each interaction contrast by up to 0.0002; this is a precision bound, not a statistical confidence interval. The present aggregate values therefore do not establish a reliable positive interaction. Moreover, this switch comparison does not isolate whether sharing the same prior is better than using separate priors. Such a mechanism claim requires additional controls, including independent, shuffled, or constant priors and seed-level uncertainty estimates.
'''
    (root/'factorial_section_en.md').write_text(insertion)

    response='''# 审稿回复建议

以下为完成对应正文替换、表格更新和图表插入后可使用的英文回复。它们不声称已补做多种子检验或共享先验控制。图三按作者确认的 Recall@10 整理。

## R1.1 总体提升较小且不一致

Thank you for pointing out that the gains are modest and not uniform. We have revised the performance discussion to report the exception explicitly. AMORE obtains the highest reported score in 11 of the 12 dataset–metric comparisons, but its Recall@20 on Baby is 0.1040 versus 0.1048 for MENTOR, a relative decrease of 0.76%. The relative Recall@20 changes on Sports and Clothing are +0.53% and +2.12%, respectively. We have replaced the claim of consistent superiority with a dataset- and metric-specific description. The current aggregate values do not establish statistical significance, so we make no such claim. Repeated-run uncertainty remains necessary to assess the reliability of these differences.

## R1.2 与 R3.4 模块消融的解释

We have updated the module-ablation table and report both absolute scores and relative decreases. Removing ELAM reduces Recall@20 by 1.06%, 1.14%, and 1.78%, and NDCG@20 by 5.05%, 1.34%, and 3.31% on Baby, Sports, and Clothing, respectively. MACA removal yields the largest decrease among the three module removals for every dataset and metric. We have narrowed the interpretation to module-level ranking contributions. These comparisons do not identify erroneous edges, establish a denoising mechanism, or isolate the individual objectives inside MACA. We also distinguish whole-module removal from disabling a single use of affinity in the factorial comparison.

## R2.1 传播与融合是否因共享亲和力而产生收益

Thank you for this observation. We distinguish the module-removal study from a 2 × 2 comparison of affinity use in propagation and fusion on the same backbone. Configuration 00 disables both uses, 10 enables propagation modulation only, 01 enables affinity-guided fusion only, and 11 enables both. We report the four Recall@10 scores and the interaction contrast Δ_int = y_11 − y_10 − y_01 + y_00 in addition to the ordinary ablations.

The joint gains over 00 are +0.0011, +0.0012, and +0.0016 on Baby, Sports, and Clothing. The corresponding interaction contrasts, computed from the displayed scores, are +0.0001, −0.0002, and +0.0001. Although configuration 11 has the highest reported Recall@10 in each dataset, the interaction contrasts are small and vary in sign. We therefore do not interpret the full model's higher score as evidence of consistently positive synergy.

We also acknowledge two limits of this evidence. First, the scores are displayed to four decimal places, and seed-level records are not included in the present summary. If rounded to the nearest 0.0001, these values imply an interaction rounding bound of up to 0.0002, so the small positive contrasts require higher-precision and repeated-run analysis. Second, toggling the two uses of affinity does not establish that sharing the same prior is better than using independent priors. Independent-prior, shuffled-prior, or constant-prior controls are not included in the supplied results. Accordingly, we have limited the claim to the observed contribution of affinity use and explicitly leave the shared-prior mechanism unconfirmed.

## R3.2 MACA 内各损失的独立贡献

The updated module ablation removes MACA as a combined component. We have clarified that this result supports the aggregate contribution of the removed objectives but cannot separate the contributions of contrastive learning and moment matching. We do not present this whole-component ablation as a substitute for a loss-level factorial comparison.

## 尚不能用这三张表解决的意见

- R1.1 / R3.3：未提供逐种子分数，不能补写 SD、置信区间、p 值或显著性星号。
- R1.3 / R2.3：新增 DHMRec 不等于补齐所有审稿人指定基线；统一划分、特征、候选集及调参预算尚需记录支持。
- R2.1：已有四组合点估计，但共享先验与独立先验、打乱先验、常数先验的机制对照没有提供。
- R3.2：移除 MACA 不是矩约束与 InfoNCE 的独立消融。
- R2.4 / R3.5 / R3.6 / R1.5 / R3.7：长尾/冷启动、敏感性、噪声、效率仍需各自真实记录。
- R1.4：这三类数据仍均来自 Amazon，不支持跨平台、视频或音乐泛化。
'''
    (root/'reviewer_responses.md').write_text(response)

    lines=['# 基于本次真实汇总值的逐句修改清单','',
           '定位原文：`papers/AMORE.docx`。下面的原句逐项从该文件提取并校验；标题与原句可用于 Word 搜索。此清单提供替换建议，尚未直接写入原始 DOCX。','',
           '图三已按作者确认的 Recall@10、Baby／Sports／Clothing 顺序整理；重复次数和评估划分尚未提供。当前不添加误差线、显著性或未经提供的机制控制。','',
           '## 先替换表格','',
           '1. **表 2 整表替换**：本次有 12 个基线，加 AMORE 共 13 个模型；原文无 DHMRec 行。使用 `data/performance.csv` 或 `tables/table2_performance.tex`。MENTOR 为所有列的最强基线，12 个 Improv. 百分比均复算一致。','',
           '2. **表 3 整表替换**：使用 `data/ablation.csv` 或 `tables/table3_ablation.tex`。下面列出相对原始 Word 的全部数值差异，避免只改正文、漏改表格。','',
           '3. **新增亲和力四组合表**：使用 `tables/table4_affinity.tex`。图三为 Recall@10；普通消融为 Recall@20 / NDCG@20。不能将两张表的行互换。','',
           '4. 输入校核：总体表文字中的 Sports/MMGCN/NDCG@20 为 **0.0254**，截图看起来为 **0.0612**。本次采用文字值 0.0254，请修正截图/Word 对应格或提供原始记录确认。','',
           '| 表格 | 方法/变体 | 数据集 | 指标 | 原 Word | 本次值 |','|---|---|---|---|---:|---:|']
    lines += [f"| {r['table']} | {r['model']} | {r['dataset']} | {r['metric']} | {r['old']:.4f} | {r['new']:.4f} |" for r in diffs]
    lines += ['', '## 逐句或逐段替换','']
    for c in changes:
        lines += [f"### {c['id']} {c['location']}",'','**原文**','', '> '+c['original'],'','**替换为**','',c['replacement'],'','**理由**：'+c['reason'],'']
        if c['condition']: lines += ['**采用前核对**：'+c['condition'],'']
    lines += ['## 新增基线说明','',
              '在 4.1.3 的解耦类模型组中添加（参考文献 [23] 已存在于原稿）：','',
              'DHMRec (2026) [23]: A collaboration-guided multimodal recommendation method that disentangles modality-common and modality-specific information and integrates these representations through hierarchical fusion.','',
              '## 新增 R2.1 实验小节','',
              '插入位置：Table 3 及其分析之后、原 4.4 Hyper-parameter Sensitivity 之前。建议作为 4.3 的子节；新表可编号 Table 4。图用插入后的实际顺序编号，并同步修改现有 Figure 2–5 的交叉引用。图三的 10 表示仅传播调制，01 表示仅融合调制。','',insertion,'',
              '## 对照方式与配对公式','',
              '若后续提供相同种子 s 下的四个分数，应先计算 d_s = y11,s − y10,s − y01,s + y00,s，再汇总 d_s 的均值、样本 SD 和区间。若四格已经是同一组种子的均值，其线性对比等于种子交互项的均值，但单靠四个均值无法恢复配对方差；目前也不知道这四格是否如此汇总。不要直接称本次结果为“配对交互效应检验”。','',
              '## 图表取舍','',
              '- 主文保留 Table 2；可加 `02_relative_improvement` 概括全部 12 项差异，`01_overall_comparison` 可放补充材料。','- Table 3 配 `04_module_drop` 展示模块贡献差异；`03_module_ablation` 展示原始绝对值，二者任选一个进主文，减少重复。','- R2.1 用 `05_affinity_factorial` 和新增四组合表；`06_affinity_effects` 可放回复信或补充材料。','- 图均有 PNG、SVG、PDF；图注见 `figure_captions.md`。','',
              '## 本轮文字仍需的数据与实现说明','',
              '在 4.1.4 补充实际评估划分、种子数量、每行是单次还是均值、各基线运行来源与相同协议的具体记录。这些信息未提供，不填入猜测。原稿的损失公式、Adaptive BPR、接收者相关的边调制、残差系数是否实际生效，应继续与对应运行版本代码核对；三张分数表不能验证这些实现。原稿敏感性与 t-SNE 小节也不能因本次更新而自动视为获得了新的实验支持。','',
              '当前可保留完整模型作为论文主模型；这批点估计没有要求增加新模块。优先核对实验定义与精度，收紧机制表述，而非据小交互项继续堆叠结构。','']
    (root/'manuscript_changes.md').write_text('\n'.join(lines).rstrip()+'\n')
    caps=['# Figure captions','']
    for f in figures: caps += ['## '+f['name'],'',f['caption'],'']
    (root/'figure_captions.md').write_text('\n'.join(caps).rstrip()+'\n')
    return changes


def write_report(root,inputs,perf,ab,gains,drops,effects,figures,changes,table,image):
    datasets=inputs['datasets'];metrics=inputs['metrics']
    parts=['<header><p class="eyebrow">AMORE · 实验结果与论文修改</p><h1>总体表现、模块消融与亲和力交互</h1><p>使用本次提供的实验汇总数值，沿用此前图表风格。所有图表由同一份输入生成。</p><p class="meta">数据接收日期：2026-11-10。种子数量、汇总方式及评估划分尚未提供，因此仅展示点估计。图三按作者确认的 Recall@10 整理。</p></header>',
           '<nav><a href="#overall">总体表现</a><a href="#ablation">模块消融</a><a href="#factorial">R2.1 交互</a><a href="#edits">原句与替换</a><a href="#files">文件与说明</a></nav>',
           '<section id="overall"><h2>1 总体表现</h2><p>AMORE 在 12 个数据集与指标组合中取得 11 个最高报告值。Baby Recall@20 下降 0.76%；其余差异为正。MENTOR 是每一列的最强基线。</p>',image('02_relative_improvement')]
    parts.append('<label>数据集 <select id="dataset">'+''.join('<option>'+ds+'</option>' for ds in datasets)+'</select></label>')
    for ds in datasets:
        sub=perf[perf.dataset==ds];rows=[]
        for row in sub.to_dict('records'):
            cells=[html.escape(row['label'])]
            for metric in metrics:
                v=row[metric];ranked=sorted(sub[metric].unique(),reverse=True);s=f'{v:.4f}'
                if v==ranked[0]: s='<strong>'+s+'</strong>'
                elif v==ranked[1]: s='<u>'+s+'</u>'
                cells.append(s)
            rows.append(cells)
        rows.append(['Improv. (%)']+[f'{gains[(gains.dataset==ds)&(gains.metric==m)].iloc[0].relative_percent:+.2f}' for m in metrics])
        parts.append('<div data-dataset="'+ds+'" '+('hidden' if ds!='Baby' else '')+'>'+table(['Models']+metrics,rows)+'</div>')
    parts += ['<p class="meta">粗体为最高值，下划线为次高值。文字输入中的 Sports / MMGCN / NDCG@20 为 0.0254，截图看起来为 0.0612；当前采用文字值，请核对该格。</p>',image('01_overall_comparison'),'</section>']
    parts += ['<section id="ablation"><h2>2 模块消融</h2><p>MACA 整体移除带来最大下降。CFDM 对 Recall@20 的影响均大于 ELAM，但在 Baby 与 Clothing 的 NDCG@20 上，ELAM 移除降幅更大。不能将所有指标概括为同一种模块重要性排序。</p>',image('04_module_drop'),image('03_module_ablation')]
    rows=[]
    for variant in inputs['ablation']:
        for metric in inputs['ablation_metrics']:
            rows.append([variant,metric]+[f'{ab[(ab.variant==variant)&(ab.dataset==ds)].iloc[0][metric]:.4f}' for ds in datasets])
    parts += [table(['Architecture','Metric']+datasets,rows),'<p>普通消融是移除 ELAM、CFDM 或 MACA 整体。图三只切换亲和力在两个位置的使用；两种干预的定义不同，指标也不同。</p></section>']
    parts += ['<section id="factorial"><h2>3 R2.1 亲和力使用的交互</h2><p><code>Δint = y11 − y10 − y01 + y00</code>。完整组合表现最高，与“收益超过两个单独收益之和”是两个不同结论。</p>',image('05_affinity_factorial')]
    rows=[[r.dataset]+[f'{getattr(r,k):.4f}' for k in ['y00','y10','y01','y11']]+[f'{r.joint_gain:+.4f}',f'{r.interaction:+.4f}'] for r in effects.itertuples()]
    parts += [table(['数据集','00','10','01','11','11 − 00','交互项'],rows),image('06_affinity_effects'),
              '<p><strong>可支持的结论：</strong>三个数据集的完整组合都优于无亲和力和两个单独使用的配置；当前显示的交互项很小，Sports 为负，不能写成稳定的正向协同。</p>',
              '<p><strong>精度限制：</strong>若四格均按最近值四舍五入到四位小数，交互项仅由舍入就可能相差至多 0.0002。这是精度界限，不是置信区间。当前不能用这些数值做配对显著性检验。</p>',
              '<p><strong>机制范围：</strong>2×2 开关能区分使用位置，但没有比较共享先验与独立先验，也没有打乱或常数先验结果。R2.1 获得了部分证据，尚不能宣称共享先验机制已经证实。</p>',
              '<p>完整回复与新增小节：<a href="reviewer_responses.md">审稿回复</a> · <a href="factorial_section_en.md">英文实验小节</a></p></section>']
    parts += ['<section id="edits"><h2>4 原句改成什么</h2><p>以下原句均与本地 <code>papers/AMORE.docx</code> 核对。展开查看可复制的替换文本；完整表格差异与插入位置见 <a href="manuscript_changes.md">逐句修改清单</a>。</p>']
    for c in changes:
        parts += ['<details><summary>'+html.escape(c['id']+' '+c['location'])+'</summary><h3>原文</h3><p class="original">'+html.escape(c['original'])+'</p><h3>替换为</h3><p class="replacement">'+html.escape(c['replacement'])+'</p><p class="reason">'+html.escape(c['reason'])+'</p>']
        if c['condition']: parts.append('<p><strong>采用前核对：</strong>'+html.escape(c['condition'])+'</p>')
        parts.append('</details>')
    parts += ['</section><footer id="files"><h2>5 文件与使用</h2><p><a href="README.md">使用说明</a> · <a href="manuscript_changes.md">逐句修改清单</a> · <a href="reviewer_responses.md">审稿回复</a> · <a href="figure_captions.md">英文图注</a></p><p>图表下可下载 PNG、SVG 和 PDF。输入见 <a href="inputs.json">inputs.json</a>；可编辑表格源文件在 <code>tables/</code>，计算明细在 <code>data/</code>。</p><p>本报告覆盖本次提供的三组结果。损失独立消融、噪声、敏感性、冷启动与效率需要各自真实记录，尚未在这里生成对应实验图。</p></footer>']
    css='''*{box-sizing:border-box}body{margin:0;background:#f1f4f5;color:#253e49;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC",sans-serif;line-height:1.8}main{max-width:1200px;padding:24px;margin:auto}header{background:#203e4c;color:white;padding:30px 36px;border-radius:10px}h1{font-size:29px;margin:8px 0}h2{font-size:23px;margin:0 0 14px}h3{font-size:15px}.eyebrow{font-size:13px;color:#bdd2dc}.meta{font-size:12px;opacity:.8}nav{display:flex;gap:12px;flex-wrap:wrap;padding:20px 0}a{color:#267392;text-decoration:none}nav a{background:white;padding:5px 12px;border-radius:5px}section,footer{background:white;padding:30px;border:1px solid #dee5e8;border-radius:10px;margin-bottom:22px}img{display:block;width:100%;height:auto}figure{margin:22px 0}figcaption,.downloads{font-size:12px;color:#667b85}.scroll{overflow:auto}table{width:100%;border-collapse:collapse;font-size:13px;margin:18px 0}th,td{padding:10px 12px;white-space:nowrap;border-bottom:1px solid #e3e8eb;text-align:right}th{background:#eef3f5}td:first-child,th:first-child{text-align:left}select{padding:6px 18px;margin-left:10px;border:1px solid #bacbd3;border-radius:4px;background:white}details{border-top:1px solid #e0e6e9;padding:14px 0}summary{cursor:pointer;font-weight:600}.original{color:#74828a}.replacement{font-family:Georgia,"Times New Roman",serif;font-size:16px;color:#213b49}.reason{font-size:13px;color:#54717d}code{font-size:13px}section{scroll-margin-top:16px}[hidden]{display:none!important}@media(max-width:650px){main{padding:10px}header,section,footer{padding:20px}h1{font-size:24px}h2{font-size:21px}nav{gap:7px}nav a{font-size:13px}}@media print{body{background:white}nav,.downloads{display:none}header{background:white;color:#203e4c}section{border:none}}'''
    js="document.getElementById('dataset').addEventListener('change',e=>document.querySelectorAll('[data-dataset]').forEach(p=>p.hidden=p.dataset.dataset!==e.target.value));"
    (root/'report.html').write_text('<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>AMORE 实验结果与论文修改</title><style>'+css+'</style></head><body><main>'+''.join(parts)+'</main><script>'+js+'</script></body></html>')
