#!/usr/bin/env python3
"""Build an offline, self-contained learning report from this directory's examples."""
import base64
import html
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT=Path(__file__).resolve().parent
MODELS=['LightGCN','MGCN','GUME','MENTOR','AMORE']
DATASETS=['Baby','Sports','Clothing']
METRICS=['recall@10','recall@20','ndcg@10','ndcg@20']
MANIFEST=json.loads((ROOT/'manifest.json').read_text())
FIGURES={x['stem']:x for x in MANIFEST['figures']}


def image(stem):
    f=FIGURES[stem]
    data=base64.b64encode((ROOT/'figures'/(stem+'.png')).read_bytes()).decode()
    return '<figure><img loading="lazy" src="data:image/png;base64,'+data+'" alt="'+html.escape(f['title'])+'"><figcaption>'+html.escape(f['caption'])+'</figcaption><div class="downloads"><a href="figures/'+stem+'.png" download>PNG</a><a href="figures/'+stem+'.svg" download>SVG 矢量图</a></div></figure>'


def table(headers, rows):
    return '<div class="table-scroll"><table><thead><tr>'+''.join('<th>'+html.escape(str(x))+'</th>' for x in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+str(x)+'</td>' for x in row)+'</tr>' for row in rows)+'</tbody></table></div>'


def switch(panels):
    options=''.join('<option>'+ds+'</option>' for ds in DATASETS)
    return '<div data-switch><label class="selector">查看场景 <select>'+options+'</select></label>'+''.join('<div class="ds-panel" data-dataset="'+ds+'" '+('hidden' if ds!='Baby' else '')+'>'+panels[ds]+'</div>' for ds in DATASETS)+'</div>'


def main():
    perf=pd.read_csv(ROOT/'data/performance_runs.csv')
    paired=pd.read_csv(ROOT/'data/table_02_ablation_effects.csv')
    effects=[]
    panels={}
    for ds in DATASETS:
        group=perf[perf.dataset==ds]
        means=group.groupby('model')[METRICS].mean();sds=group.groupby('model')[METRICS].std()
        rows=[]
        for model in MODELS:
            row=[model]
            for metric in METRICS:
                s='{:.4f} ± {:.4f}'.format(means.loc[model,metric],sds.loc[model,metric])
                if means.loc[model,metric]==means[metric].max(): s='<strong>'+s+'</strong>'
                row.append(s)
            rows.append(row)
        panels[ds]=table(['方法标签','Recall@10','Recall@20','NDCG@10','NDCG@20'],rows)
        for metric in ['recall@20','ndcg@20']:
            a=group[group.model=='AMORE'].set_index('seed')[metric]
            b=group[group.model=='MENTOR'].set_index('seed')[metric]
            delta=a-b;half=stats.t.ppf(.975,4)*delta.std(ddof=1)/np.sqrt(5)
            effects.append(dict(provenance='synthetic_learning_example',dataset=ds,metric=metric,n=5,
                                paired_delta=delta.mean(),ci95_low=delta.mean()-half,ci95_high=delta.mean()+half,
                                relative_percent=100*(a.mean()/b.mean()-1)))
    effects=pd.DataFrame(effects);effects.to_csv(ROOT/'data/table_05_paired_comparison.csv',index=False,float_format='%.10g')
    effect_rows=[[r.dataset,r.metric,'{:+.3f}%'.format(r.relative_percent),
                  '{:+.6f}'.format(r.paired_delta),'[{:+.6f}, {:+.6f}]'.format(r.ci95_low,r.ci95_high)] for r in effects.itertuples()]
    efficiency=pd.read_csv(ROOT/'data/efficiency_runs.csv')
    erows=[]
    for model in MODELS:
        g=efficiency[efficiency.model==model]
        erows.append([model,'{:.2f}'.format(g.parameters_million.mean()),
                      '{:.3f} ± {:.3f}'.format(g.epoch_seconds.mean(),g.epoch_seconds.std()),
                      '{:.0f}'.format(g.peak_mib.mean()),
                      '{:.3f} ± {:.3f}'.format(g.inference_seconds.mean(),g.inference_seconds.std())])
    main_text=[]
    for ds in DATASETS:
        r=effects[(effects.dataset==ds)&(effects.metric=='recall@20')].iloc[0]
        main_text.append('<li><strong>'+ds+'</strong>：Recall@20 相对 MENTOR 为 {:+.3f}%，配对均值差 {:+.6f}，95% 区间 [{:+.6f}, {:+.6f}]。</li>'.format(r.relative_percent,r.paired_delta,r.ci95_low,r.ci95_high))
    ab_rows=[]
    for variant in ['No moments','No InfoNCE','No BM','No UM cross','No UM noise','No extended L2']:
        r=paired[(paired.dataset=='Baby')&(paired.metric=='recall@20')&(paired.variant==variant)].iloc[0]
        ab_rows.append([variant,'{:+.3f}%'.format(r.relative_percent),'[{:+.6f}, {:+.6f}]'.format(r.ci95_low,r.ci95_high)])
    reviewer_rows=[['R1.1 / R3.3','总体表、配对区间','幅度、波动与显著性是不同问题。'],
                   ['R1.2 / R2.1 / R3.4','分支 2×2、共享先验控制','总性能好不代表共享机制成立。'],
                   ['R3.2 / R3.4','矩约束 × InfoNCE、逐项损失','强贡献、弱贡献与冗余项可以共存。'],
                   ['R3.5','三个参数的敏感性','平坦区域比单一尖峰更支持稳定性。'],
                   ['R3.6','绝对指标与自身退化曲线','不能用干净条件的领先代替抗噪证据。'],
                   ['R2.4','用户、物品、冷启动分组','分组分母与实验场景需要分别定义。'],
                   ['R1.5 / R3.7','效率表、精度与代价','小增益可能需要额外资源。'],
                   ['R3.8','四种证据解释','结论强度必须与区间及控制实验对应。']]
    contents='''<header><div class="eyebrow">AMORE · 图表与证据表达练习</div><h1>小幅改进，应该怎样呈现？</h1><p class="lead">用一组有收益、有波动、也有失败条件的场景，练习把性能、机制与代价讲清楚。</p><p class="source">数据来源：独立生成的模拟学习示例；方法与数据集名称仅作为场景标签，不对应实际测量，也不预测真实训练结果。</p><div class="facts"><span>3 个场景类别</span><span>5 次模拟重复</span><span>9 类分析</span><span>13 幅图 · PNG / SVG</span></div></header>
<nav><a href="#overall">总体性能</a><a href="#branches">分支</a><a href="#losses">损失</a><a href="#sensitivity">敏感性</a><a href="#noise">噪声</a><a href="#cohorts">分组</a><a href="#efficiency">效率</a><a href="#training">收敛</a><a href="#language">分析措辞</a></nav>'''
    contents+='<section id="overall"><div class="kicker">01 / 总体表现</div><h2>让读者同时看到均值和波动</h2><p>示例中，AMORE 在 Baby 和 Sports 有限领先，在 Clothing 并未全面领先。不同指标不必给出同一个排名。</p>'+image('01_performance')+switch(panels)+'<p class="small">表中为均值 ± 样本 SD；粗体只表示最高均值。精确数值见 CSV，不用星号制造“显著”的视觉暗示。</p><h3>与较强参照的配对差值</h3>'+table(['类别','指标','相对变化','绝对均值差','配对 95% CI'],effect_rows)+'<ul>'+''.join(main_text)+'</ul><div class="note">合理的表述：“总体上有有限收益，但不同类别和指标存在差异。” 区间跨零时应补充“证据尚不足以确认差异”，不能直接写成稳定优越。</div></section>'
    contents+='<section id="branches"><div class="kicker">02 / 机制</div><h2>联合版本较好，不等于两个模块存在协同</h2>'+image('02_branch_factorial')+'<p>看两条线的斜率差，而不是只看右上角。正式分析应计算每个种子的交互项 <code>y11 − y10 − y01 + y00</code>。打乱、常数和停止梯度控制保存在消融数据中，可用于排查容量与先验信息的区别。</p><div class="note">这里刻意让 Baby 有较清晰的收益、Sports 接近无效、Clothing 存在负面变化。这比三个类别完全同向、同幅度更适合练习证据边界。</div></section>'
    contents+='<section id="losses"><div class="kicker">03 / 训练目标</div><h2>强约束、弱约束和冗余项可以同时存在</h2>'+image('03_loss_analysis')+table(['Baby 移除项','Recall@20 相对变化','配对 95% CI'],ab_rows)+'<p>示例中的 InfoNCE 整体贡献明显，矩约束较弱；个别损失在 Clothing 可出现轻微反作用。不能用所有损失相加后的效果，替代每个损失的证据。</p><p class="small">热图为效果大小，不表示统计显著性。去掉全部 InfoNCE 时仍保留表示正则，因此不能将其称为纯 BPR。</p></section>'
    contents+='<section id="sensitivity"><div class="kicker">04 / 参数</div><h2>平坦区域与极端退化，比漂亮的尖峰更有解释力</h2>'+image('04_sensitivity')+'<p>曲线采用较宽的平台、略有偏移的较好取值和过大权重下的退化。三个类别不需要共享完全相同的最优点。零值使用独立位置，不伪装成对数坐标。</p><div class="note">建议分析“在哪个区间变化不大、什么时候开始退化”，避免只强调最高点。误差带重叠较大时，最高点也未必是稳定最优。</div></section>'
    contents+='<section id="noise"><div class="kicker">05 / 噪声</div><h2>把起点优势与抗扰动能力分开</h2>'+switch({ds:image('05_robustness_'+ds.lower()) for ds in DATASETS})+'<p>上排是绝对性能，下排是相对各自干净条件的下降比例，二者回答不同问题。这个场景允许 AMORE 在大部分干预下较稳健，同时在 Clothing 的文本错配下失去优势。</p><p class="small">这些是人为指定的噪声响应，不是对真实数据中错误交互比例的估计。相对退化逐种子计算后再汇总，避免把均值的比值误当成配对结果。</p></section>'
    contents+='<section id="cohorts"><div class="kicker">06 / 人群与物品</div><h2>稀疏、长尾、零交互是三个不同问题</h2>'+switch({ds:image('06_cohorts_'+ds.lower()) for ds in DATASETS})+'<p>用户活动分组构成总体用户的划分，其加权均值与总体分数一致。物品分组指标按组内目标条件化，不能直接相加。最右侧零交互物品来自另一个场景，不是普通长尾的别名。</p><div class="note">这里仅生成聚合层面的学习数据，没有虚构逐用户排名或真实样本数。实际论文需要额外报告各组 eligible users、目标数及分组规则。</div></section>'
    contents+='<section id="efficiency"><div class="kicker">07 / 代价</div><h2>精度的小幅收益，可以伴随明确的额外成本</h2>'+image('07_efficiency')+table(['方法标签','参数量 / M','秒 / epoch','峰值 / MiB','秒 / 排名遍历'],erows)+'<p>AMORE 没有被设置成既最准又最快、同时最省显存。读者可以看到额外参数、训练和推理代价。这里的时间与显存是场景假设，没有对应实测硬件。</p></section>'
    contents+='<section id="training"><div class="kicker">08 / 训练过程</div><h2>训练损失下降，不保证验证指标一直上升</h2>'+image('08_convergence')+'<p>允许初期快速进展、中间小幅回落、后期平台和轻微退化。验证曲线与总体表的模拟 holdout 指标属于不同序列，不能拿最后一轮训练值替换验证选择后的评估值。</p></section>'
    contents+='<section id="language"><div class="kicker">09 / 证据解释</div><h2>相似的平均值，可能需要不同的结论</h2>'+image('09_interpretation')+table(['情形','建议解释'],[
        ['均值为正，区间完整位于零右侧','该设置提供正向差异证据；仍需看实际效应大小和适用范围。'],
        ['均值略正，区间跨零','观察到正向趋势，但不足以确认稳定提升。'],
        ['正负种子差异很大','可能存在运行敏感性；不应只展示最好的种子。'],
        ['均值为负，区间完整位于零左侧','该条件下出现退化，需保留负结果并限制主张。']])+ '<h3>这些图在审稿问题中各自回答什么？</h3>'+table(['意见主题','对应图表','分析边界'],reviewer_rows)+'<p>英文图注已附于每张图下。更多带有场景前提的英文讨论与回复写法，见 <a href="paper_style_examples.md">文字示例</a>。</p></section>'
    contents+='<footer><h2>数据与复现</h2><p>所有数据均在 <code>data/</code> 中，含来源字段；图表对应的生成参数与版本见 <a href="manifest.json">manifest.json</a>。</p><p><a href="README.md">使用说明</a> · <a href="data/table_01_performance.csv">性能汇总 CSV</a> · <a href="data/table_02_ablation_effects.csv">消融统计 CSV</a> · <a href="data/table_05_paired_comparison.csv">配对比较 CSV</a> · <a href="data/table_04_efficiency.csv">效率汇总 CSV</a></p><p class="small">这份报告可离线打开，图片已内嵌；SVG 与 CSV 下载链接需要保留同目录文件。</p></footer>'
    css='''*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;color:#263b46;background:#f2f4f4;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC","Microsoft YaHei",sans-serif;line-height:1.75}main{max-width:1180px;margin:auto;padding:24px}header{padding:34px 38px;background:#223e4b;color:#fff;border-radius:12px}.eyebrow,.kicker{font-size:12px;letter-spacing:.09em}.eyebrow{color:#b8d0da}h1{font-size:34px;font-weight:650;margin:12px 0}h2{font-size:24px;font-weight:650;margin:7px 0 16px}h3{font-size:18px;margin:26px 0 12px}.lead{font-size:17px;color:#dbe5e9}.source{font-size:12px;color:#b8cbd3;max-width:900px}.facts{display:flex;flex-wrap:wrap;gap:12px;margin-top:24px}.facts span{font-size:12px;border:1px solid #58717e;border-radius:5px;padding:4px 11px}nav{display:flex;flex-wrap:wrap;gap:10px;padding:18px 3px;margin-bottom:8px}nav a{padding:6px 12px;background:#fff;border:1px solid #dbe2e5;border-radius:5px;font-size:13px}a{color:#286782;text-decoration:none}a:hover{text-decoration:underline}section,footer{background:#fff;border:1px solid #e0e6e8;border-radius:10px;padding:30px;margin-bottom:22px;scroll-margin-top:16px}.kicker{color:#78909b}figure{margin:18px 0 24px}img{display:block;width:100%;height:auto}figcaption{font-size:12px;color:#637780;margin:10px 5px;line-height:1.65}.downloads{display:flex;gap:14px;font-size:12px;margin:8px 5px}.table-scroll{overflow:auto}table{border-collapse:collapse;width:100%;font-size:13px;margin:16px 0}th{text-align:left;background:#eef3f5;color:#355565;font-weight:600}th,td{padding:11px 13px;border-bottom:1px solid #e1e7ea;white-space:nowrap}td:first-child{font-weight:500}tbody tr:hover{background:#f8fafb}.note{border-left:3px solid #91b0bf;background:#f2f6f7;padding:14px 18px;margin-top:18px;font-size:14px}.small{font-size:12px;color:#6c7c84}.selector{font-size:13px;color:#506771;display:flex;gap:10px;align-items:center;margin:12px 0}select{padding:6px 28px 6px 12px;border:1px solid #c4d1d7;border-radius:5px;background:#fff;color:#294653}.ds-panel[hidden]{display:none}code{font-size:12px;background:#edf2f4;padding:2px 4px;border-radius:3px}@media(max-width:700px){main{padding:10px}header,section,footer{padding:20px}h1{font-size:27px}h2{font-size:21px}nav{gap:6px}th,td{padding:9px}p{font-size:14px}}@media print{body{background:white}nav,.downloads,.selector{display:none}main{max-width:none;padding:0}section{break-inside:avoid}header{color:#223e4b;background:white;border-bottom:2px solid #223e4b}.lead,.source,.eyebrow{color:#526c79}}'''
    script="document.querySelectorAll('[data-switch] select').forEach(s=>s.addEventListener('change',()=>{s.closest('[data-switch]').querySelectorAll('.ds-panel').forEach(p=>p.hidden=p.dataset.dataset!==s.value)}));"
    (ROOT/'report.html').write_text('<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>AMORE 图表与证据表达练习</title><style>'+css+'</style></head><body><main>'+contents+'</main><script>'+script+'</script></body></html>')
    text='# 图注、讨论与审稿回复的学习示例\n\n本文件是上述模拟场景的文字练习，数值来自本目录生成器。英文段落保留场景前提。\n\n## 1. 总体性能\n\n'
    for ds in DATASETS:
        r=effects[(effects.dataset==ds)&(effects.metric=='recall@20')].iloc[0]
        text+='In this illustrative {} scenario, the mean Recall@20 difference relative to the comparator is {:+.6f} ({:+.3f}% relative), with a paired 95% interval of [{:+.6f}, {:+.6f}]. '.format(ds,r.paired_delta,r.relative_percent,r.ci95_low,r.ci95_high)
        text+=('The interval excludes zero, but the practical magnitude remains modest.\n\n' if r.ci95_low>0 else 'This does not establish a reliable positive difference under this scenario.\n\n')
    text+='## 2. 机制讨论写法\n\nWithin this simulated factorial example, the joint model performs favorably in one category, while the pattern weakens or reverses in other categories. A shared-signal interpretation would require the paired interaction contrast and informative-prior controls; the rank of the joint variant alone is insufficient.\n\n'
    text+='## 3. 损失讨论写法\n\nThe illustrative objective ablations assign a stronger role to the aggregate contrastive objective than to moment matching. Some individual terms yield small or inconsistent effects. Such a pattern would support distinguishing essential and weakly supported components, rather than claiming that every term is necessary.\n\n'
    text+='## 4. 噪声讨论写法\n\nThe simulated corruption curves show why absolute performance and degradation from the clean condition should be reported together. The text-mismatch scenario for Clothing intentionally includes a failure condition, preventing a universal robustness claim.\n\n'
    text+='## 5. 回复审稿人的条件式写法\n\nIf a corresponding real experiment produced this pattern, a response could explain that matched-seed estimates, uncertainty intervals, and negative conditions have been added, and that claims have been narrowed to the settings supported by those measurements. The simulated examples themselves do not establish that any empirical reviewer concern has been resolved.\n\n'
    text+='## 6. 英文图注\n\n'
    for f in MANIFEST['figures']:
        text+='### '+f['stem']+'\n\n'+f['caption']+'\n\n'
    (ROOT/'paper_style_examples.md').write_text(text.rstrip()+'\n')
    print('Built offline report and narrative examples:',ROOT/'report.html')


if __name__=='__main__':
    main()
