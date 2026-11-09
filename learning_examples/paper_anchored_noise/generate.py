#!/usr/bin/env python3
"""Paper-reported clean anchors with explicitly assumed corruption responses."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
DATASETS = ['Baby', 'Sports', 'Clothing']
MODELS = ['AMORE', 'MENTOR', 'DHMRec']
METRICS = ['recall@10', 'recall@20', 'ndcg@10', 'ndcg@20']
KINDS = {'interaction': 'Interaction injection', 'image': 'Image mismatch', 'text': 'Text mismatch'}
ZH = {'interaction': '交互注入', 'image': '图像错配', 'text': '文本错配'}
RATES = [0, .05, .10, .20, .30]
COLORS = {'AMORE': '#c47749', 'MENTOR': '#387b9c', 'DHMRec': '#69a389'}
# Values transcribed from AMORE.docx Table 2 and DHMRec original Table 1.
ANCHORS = {
    'Baby': {'AMORE': [.0682, .1040, .0366, .0455],
             'MENTOR': [.0678, .1048, .0362, .0450],
             'DHMRec': [.0646, .1016, .0343, .0436]},
    'Sports': {'AMORE': [.0770, .1145, .0419, .0523],
               'MENTOR': [.0763, .1139, .0409, .0511],
               'DHMRec': [.0738, .1132, .0401, .0502]},
    'Clothing': {'AMORE': [.0693, .1010, .0372, .0453],
                 'MENTOR': [.0668, .0989, .0360, .0441],
                 'DHMRec': [.0629, .0931, .0340, .0417]}}
SOURCES = {
    'AMORE': 'papers/AMORE.docx, Table 2, AMORE (ours)',
    'MENTOR': 'papers/AMORE.docx, Table 2, MENTOR (2025)',
    'DHMRec': 'https://ojs.aaai.org/index.php/AAAI/article/download/38661/42623 ; Table 1, p.16249'}
# Assumed relative Recall@20 loss (%) at 30% corruption, NOT empirical estimates.
# Within each list the method order is AMORE, MENTOR, DHMRec.
ENDPOINTS = {
    'interaction': {'Baby': [15.7, 19.3, 17.6], 'Sports': [13.9, 16.8, 15.1], 'Clothing': [20.4, 24.7, 22.2]},
    'image': {'Baby': [9.6, 11.8, 8.3], 'Sports': [8.1, 10.2, 7.5], 'Clothing': [16.7, 18.4, 13.2]},
    'text': {'Baby': [16.1, 18.7, 17.4], 'Sports': [13.8, 16.2, 15.7], 'Clothing': [14.6, 13.8, 15.9]}}


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(261110)
    records = []
    for ds in DATASETS:
        for kind in KINDS:
            for mi, model in enumerate(MODELS):
                # Positive irregular interval increments; not repeated fixed decrements.
                increments = np.array([.13, .16, .32, .39])*rng.uniform(.79, 1.21, 4)
                fraction = np.r_[0, np.cumsum(increments)/increments.sum()]
                loss20 = fraction*ENDPOINTS[kind][ds][mi]/100
                losses = []
                for metric in METRICS:
                    if metric == 'recall@20':
                        loss = loss20.copy()
                    else:
                        multiplier = {'recall@10': 1.04, 'ndcg@10': 1.11, 'ndcg@20': 1.06}[metric]
                        loss = loss20*multiplier*rng.uniform(.96, 1.04, 5)
                        assert np.all(np.diff(loss) > 0)
                    losses.append(loss)
                for ri, rate in enumerate(RATES):
                    row = dict(dataset=ds, noise=kind, noise_rate=rate, model=model,
                               value_origin='paper_reported' if rate == 0 else 'synthetic_scenario',
                               clean_source=SOURCES[model])
                    for k, metric in enumerate(METRICS):
                        base = ANCHORS[ds][model][k]
                        value = round(base*(1-losses[k][ri]), 6)
                        row[metric] = value
                        row[metric+'_drop_percent'] = round(100*(1-value/base), 6)
                    records.append(row)
    with (ROOT/'noise_data.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]), lineterminator='\n');writer.writeheader();writer.writerows(records)
    # Validate every clean anchor and every saved degradation value.
    for row in records:
        for k, metric in enumerate(METRICS):
            anchor = ANCHORS[row['dataset']][row['model']][k]
            assert 0 < row[metric] <= anchor
            if row['noise_rate'] == 0:
                assert row[metric] == anchor and row[metric+'_drop_percent'] == 0
            assert abs(row[metric+'_drop_percent']-100*(1-row[metric]/anchor)) < 1e-6
        assert row['recall@10'] <= row['recall@20']
        assert row['ndcg@10'] <= row['ndcg@20']
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.grid': True, 'grid.alpha': .15, 'svg.fonttype': 'none'})
    for ds in DATASETS:
        for relative in [False, True]:
            fig, axes = plt.subplots(2, 3, figsize=(11.8, 6.6))
            for j, kind in enumerate(KINDS):
                for i, metric in enumerate(['recall@20', 'ndcg@20']):
                    ax = axes[i, j]
                    for model in MODELS:
                        rows = [r for r in records if r['dataset']==ds and r['noise']==kind and r['model']==model]
                        key = metric+'_drop_percent' if relative else metric
                        ax.plot([r['noise_rate']*100 for r in rows], [r[key] for r in rows],
                                marker='o', markersize=4, label=model, color=COLORS[model])
                    ax.set(xticks=[0,5,10,20,30], xlabel='Corruption rate (%)',
                           ylabel=metric+(' drop (%)' if relative else ''), title=KINDS[kind])
                    if relative: ax.set_ylim(bottom=0)
                    ax.legend(fontsize=8)
            fig.suptitle(ds+(' · Relative degradation' if relative else ' · Performance under corruption'), fontsize=13)
            fig.text(.99, .008, 'Paper-reported clean anchors; simulated noise responses', ha='right', fontsize=7, color='#666')
            fig.tight_layout(rect=(0,.04,1,.95))
            stem=ds.lower()+('_relative' if relative else '_absolute')
            for ext in ['png','svg']:
                path=ROOT/(stem+'.'+ext);fig.savefig(path,dpi=200)
                if ext=='svg': path.write_text('\n'.join(x.rstrip() for x in path.read_text().splitlines())+'\n')
            plt.close(fig)
    lines = ['# 以论文指标为起点的噪声情景数据', '',
             '0% 为来源论文报告值，其余为模拟估计点；没有附加虚构的实测标准差或显著性。', '',
             'AMORE、MENTOR 起点来自 [AMORE.docx](../../papers/AMORE.docx) 表 2；DHMRec 来自 [原论文](https://ojs.aaai.org/index.php/AAAI/article/download/38661/42623) 表 1（16249 页）。这些起点来自不同论文，未核验为同协议复现结果。', '',
             '## 情景定义', '',
             '- 交互注入：新增伪交互数 / 原训练交互数，保留原训练边。',
             '- 图像错配：选定比例的物品，其图像特征替换为其他物品的图像特征；文本保持原值。',
             '- 文本错配：选定比例的物品，其文本特征替换为其他物品的文本特征；图像保持原值。',
             '- 三种干预分别施加，设想每个强度重新训练，验证及测试交互固定。比例不是高斯噪声标准差。',
             '- 每个点是场景指定值，不代表五种子均值。30% 下的下降范围是人为假设，没有由论文零噪声值推导出的统计保证。', '',
             '## 完整数据', '',
             '[noise_data.csv](noise_data.csv) 包含 135 行，四项指标 R@10、R@20、N@10、N@20，以及各自的相对下降百分比。下表展示 R@20 / N@20。', '']
    for ds in DATASETS:
        lines += ['### '+ds, '']
        for kind in KINDS:
            lines += ['#### '+ZH[kind], '', '| 噪声比例 | AMORE R@20 | AMORE N@20 | MENTOR R@20 | MENTOR N@20 | DHMRec R@20 | DHMRec N@20 |',
                      '|---|---:|---:|---:|---:|---:|---:|']
            for rate in RATES:
                vals=[]
                for model in MODELS:
                    row=next(r for r in records if r['dataset']==ds and r['noise']==kind and r['model']==model and r['noise_rate']==rate)
                    vals += [f"{row['recall@20']:.6f}", f"{row['ndcg@20']:.6f}"]
                lines.append('| '+str(round(rate*100))+'% | '+' | '.join(vals)+' |')
            lines.append('')
        lines += ['![绝对指标]('+ds.lower()+'_absolute.png)', '', '![相对下降]('+ds.lower()+'_relative.png)', '']
    lines += ['## 如何解读', '',
              '- Baby 的零噪声 Recall@20 保留 MENTOR 高于 AMORE 的论文排序；不要把 AMORE 的起点改成第一。',
              '- 交互注入场景假定 AMORE 相对下降较小，图像错配场景则允许 DHMRec 更耐受。',
              '- Clothing 文本错配中，AMORE 的相对下降可以大于 MENTOR，尽管绝对指标仍可能更高。',
              '- “分数最高”和“下降最少”必须分别判断；曲线不能归因于某个模块，模块归因还需要消融。', '',
              '## 复现', '', '`python learning_examples/paper_anchored_noise/generate.py`', '',
              '仅依赖 numpy、matplotlib；CSV 使用 Python 标准库输出。相对下降按保存的六位小数值计算。']
    (ROOT/'README.md').write_text('\n'.join(lines)+'\n')
    metadata=dict(generator_seed=261110, anchors=ANCHORS, anchor_metric_order=METRICS,
                  clean_sources=SOURCES, assumed_recall20_drop_at_30_percent=ENDPOINTS,
                  source_note='Clean values are paper reports, not independently verified runs. All nonzero-noise points are synthetic.',
                  generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  row_count=len(records), verification='clean anchors, bounds, metric ordering and drop calculations passed')
    (ROOT/'metadata.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2)+'\n')
    print('Created',len(records),'rows and six PNG/SVG figures in',ROOT)


if __name__ == '__main__':
    main()
