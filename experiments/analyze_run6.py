#!/usr/bin/env python3
"""Paired affinity evidence and training diagnostics from real validation logs."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from plot_revision import read_runs, save

VARIANTS = ['rev_00', 'rev_10', 'rev_01', 'rev_11', 'rev_prior_shuffle',
            'rev_prior_constant', 'rev_prior_detach']
LABELS = ['Neither', 'Graph only', 'Fusion only', 'Both', 'Shuffle prior',
          'Constant prior', 'Detach prior']
LOSSES = ['BPR', 'Representation L2', 'Moment alignment', 'BM', 'Extended BM',
          'UM cross', 'UM noise', 'Extended L2']
CONTRASTS = [
    ('Both - neither', {'rev_11': 1, 'rev_00': -1}),
    ('Graph - neither', {'rev_10': 1, 'rev_00': -1}),
    ('Fusion - neither', {'rev_01': 1, 'rev_00': -1}),
    ('Both - graph', {'rev_11': 1, 'rev_10': -1}),
    ('Shuffle - both', {'rev_prior_shuffle': 1, 'rev_11': -1}),
    ('Constant - both', {'rev_prior_constant': 1, 'rev_11': -1}),
    ('Detach - both', {'rev_prior_detach': 1, 'rev_11': -1}),
    ('Graph x fusion interaction', {'rev_11': 1, 'rev_10': -1, 'rev_01': -1, 'rev_00': 1}),
]


def paired_stats(values):
    values = np.asarray(values, dtype=float)
    n = len(values)
    sd = values.std(ddof=1)
    ci = stats.t.ppf(.975, n - 1) * sd / np.sqrt(n)
    p = float(stats.ttest_1samp(values, 0).pvalue) if sd else float(not values.mean())
    return dict(n=n, mean=float(values.mean()), sd=float(sd),
                ci95_low=float(values.mean() - ci), ci95_high=float(values.mean() + ci),
                p_raw=p, positive_seeds=int((values > 0).sum()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default='results/revision')
    parser.add_argument('--output', default='revision_outputs/run6')
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    frame, histories = read_runs(Path(args.root) / 'affinity', 'validation')
    if set(frame.variant) != set(VARIANTS) or frame.dataset.nunique() != 1:
        raise ValueError('Provide one dataset with all seven affinity controls')
    if frame.source.nunique() != 1 or frame.data_hash.nunique() != 1:
        raise ValueError('Source/data mismatch')
    if frame.groupby('variant').setting.nunique().max() != 1:
        raise ValueError('Multiple settings per variant')
    if frame.duplicated(['variant', 'seed']).any():
        raise ValueError('Duplicate seed; do not pool anchor reruns')
    seeds = sorted(map(int, frame.seed.unique()))
    if len(seeds) < 3 or len(frame) != len(VARIANTS) * len(seeds):
        raise ValueError('Require complete matching seeds (at least three)')
    # Check all non-intervention training settings, not just hashes.
    ignored = {'seed', 'variant', 'output_dir', 'tensorboard_log_dir', 'revision_graph',
               'revision_fusion', 'revision_prior', 'coarse_affinity'}
    signatures = set()
    for path in frame.path:
        result = json.loads(Path(path).read_text())
        cfg = result['trials'][result['selected_trial']]['config']
        signatures.add(json.dumps({k: v for k, v in cfg.items() if k not in ignored}, sort_keys=True))
    if len(signatures) != 1:
        raise ValueError('Uncontrolled configuration differences')
    plt.rcParams.update({'font.size': 9, 'axes.spines.top': False,
                         'axes.spines.right': False, 'pdf.fonttype': 42})
    title = '{} validation; {} matched seeds'.format(frame.dataset.iloc[0].title(), len(seeds))
    all_stats, differences = [], []
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True)
    for ax, metric in zip(axes, ['recall@20', 'ndcg@20']):
        table = frame.pivot(index='seed', columns='variant', values=metric).loc[seeds]
        rows = []
        for y, (name, weights) in enumerate(CONTRASTS):
            delta = sum(table[v] * w for v, w in weights.items())
            row = dict(metric=metric, contrast=name, **paired_stats(delta))
            rows.append(row)
            differences.extend(dict(metric=metric, contrast=name, seed=int(s), difference=float(d))
                               for s, d in delta.items())
            ax.scatter(delta * 1000, y + np.linspace(-.13, .13, len(seeds)),
                       s=18, color='#9aafbe', zorder=2)
            ax.errorbar(row['mean'] * 1000, y,
                        xerr=(row['ci95_high'] - row['mean']) * 1000,
                        fmt='o', color='#174c6b', capsize=3, zorder=3)
        # Holm family: all eight predeclared contrasts within this metric.
        order = sorted(range(len(rows)), key=lambda i: rows[i]['p_raw'])
        previous = 0.
        for rank, i in enumerate(order):
            previous = min(1., max(previous, (len(rows) - rank) * rows[i]['p_raw']))
            rows[i]['p_holm'] = previous
        all_stats.extend(rows)
        ax.axvline(0, color='black', lw=.7, ls='--')
        ax.set(title=metric, xlabel='Paired difference (x 1,000)',
               yticks=range(len(CONTRASTS)), yticklabels=[c[0] for c in CONTRASTS])
    axes[0].invert_yaxis()
    fig.suptitle(title + '\nDark: mean and 95% paired t interval; light: individual seeds')
    save(fig, output, 'paired_effects')
    pd.DataFrame(all_stats).to_csv(output / 'paired_statistics.csv', index=False)
    pd.DataFrame(differences).to_csv(output / 'paired_seed_differences.csv', index=False)
    frame.to_csv(output / 'audited_runs.csv', index=False)

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    for ax, metric in zip(axes, ['recall@20', 'ndcg@20']):
        for graph, color in [(0, '#2878a2'), (1, '#cf7040')]:
            means, sds = [], []
            for fusion in [0, 1]:
                values = frame.loc[frame.variant == 'rev_{}{}'.format(graph, fusion), metric]
                means.append(values.mean()); sds.append(values.std(ddof=1))
            ax.errorbar([0, 1], means, yerr=sds, marker='o', capsize=4,
                        color=color, label='Graph ' + ('on' if graph else 'off'))
        ax.set(xticks=[0, 1], xticklabels=['Fusion off', 'Fusion on'], ylabel=metric)
        ax.legend()
    fig.suptitle(title + '; mean +/- sample SD')
    save(fig, output, 'factorial')

    # Retain actual epoch support; do not forward-fill early-stopped runs.
    records = []
    for row in frame.itertuples():
        if row.path not in histories:
            raise ValueError('Missing history: ' + row.path)
        for h in histories[row.path]:
            records.append(dict(variant=row.variant, seed=row.seed, epoch=h['epoch'],
                                recall=h['validation']['recall@20'],
                                **h.get('diagnostics', {}),
                                **dict(zip(LOSSES, h['loss_components']))))
    history = pd.DataFrame(records)
    history.to_csv(output / 'training_diagnostics.csv', index=False)
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.4))
    for v, label, color in zip(VARIANTS[:4], LABELS[:4], ['#2878a2', '#cf7040', '#509b60', '#9467bd']):
        grouped = history[history.variant == v].groupby('epoch').recall.agg(['mean', 'std', 'count'])
        grouped = grouped[grouped['count'] == len(seeds)]
        axes[0, 0].plot(grouped.index, grouped['mean'], label=label, color=color)
        axes[0, 0].fill_between(grouped.index, grouped['mean'] - grouped['std'],
                                grouped['mean'] + grouped['std'], color=color, alpha=.10)
    axes[0, 0].set(ylabel='Recall@20', title='Training trajectory (mean +/- SD)')
    axes[0, 0].legend(fontsize=8)
    for v, label in [('rev_11', 'Both'), ('rev_prior_constant', 'Constant'), ('rev_prior_shuffle', 'Shuffle')]:
        wide = history[history.variant.isin(['rev_00', v])].pivot(index=['seed', 'epoch'], columns='variant', values='recall').dropna()
        delta = (wide[v] - wide.rev_00).groupby('epoch').agg(['mean', 'std', 'count'])
        delta = delta[delta['count'] == len(seeds)]
        axes[0, 1].plot(delta.index, delta['mean'] * 1000, label=label)
    axes[0, 1].axhline(0, color='black', lw=.7, ls='--')
    axes[0, 1].set(ylabel='Recall@20 difference (x 1,000)', title='Matched-epoch difference vs neither')
    axes[0, 1].legend(fontsize=8)
    full = history[history.variant == 'rev_11'].groupby('epoch').mean(numeric_only=True)
    counts = history[history.variant == 'rev_11'].groupby('epoch').size()
    full = full[counts == len(seeds)]
    for name in LOSSES:
        axes[1, 0].plot(full.index, full[name], label=name)
    axes[1, 0].set(yscale='log', ylabel='Weighted objective value', title='Both: objective magnitudes (not gradients)')
    axes[1, 0].legend(fontsize=7, ncol=2)
    for key, label in [('last_modulation_selected_fraction', 'Selected edge fraction'),
                       ('prior_std', 'Item prior SD'), ('semantic_to_id_norm', 'Semantic / collaborative norm')]:
        axes[1, 1].plot(full.index, full[key], label=label)
    axes[1, 1].set(title='Both: active signal and representation', ylabel='Diagnostic value')
    axes[1, 1].legend(fontsize=7)
    for ax in axes.flat:
        ax.set_xlabel('Epoch (all seeds available)')
    fig.suptitle(title)
    save(fig, output, 'training_mechanisms')

    anchor, _ = read_runs(Path(args.root) / 'anchor', 'validation')
    anchor.to_csv(output / 'anchor_separate.csv', index=False)
    repeat = []
    for row in anchor.itertuples():
        match = frame[(frame.variant == row.variant) & (frame.seed == row.seed) &
                      (frame.setting == row.setting)]
        for _, other in match.iterrows():
            repeat.append(dict(variant=row.variant, seed=row.seed, anchor_path=row.path,
                               affinity_path=other.path))
            original = anchor[anchor.path == row.path].iloc[0]
            repeat[-1]['recall_difference'] = other['recall@20'] - original['recall@20']
            repeat[-1]['ndcg_difference'] = other['ndcg@20'] - original['ndcg@20']
    audit = dict(dataset=frame.dataset.iloc[0], split='validation', seeds=seeds,
                 affinity_runs=len(frame), anchor_runs=len(anchor), source=frame.source.iloc[0],
                 data_hash=json.loads(frame.data_hash.iloc[0]), repeated_runs=repeat,
                 note='Anchor repeats are excluded from paired inference. CI is not an equivalence test.')
    (output / 'audit.json').write_text(json.dumps(audit, indent=2))
    print(json.dumps(audit, indent=2))


if __name__ == '__main__':
    main()
