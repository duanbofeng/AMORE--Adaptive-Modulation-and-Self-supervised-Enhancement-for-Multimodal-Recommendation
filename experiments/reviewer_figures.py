#!/usr/bin/env python3
"""Fixed rev_11 reviewer figures, with explicit stage, split and paired references."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from plot_revision import read_runs, save
from analyze_run6 import paired_stats

ROOT = Path(__file__).resolve().parents[1]
METRICS = ['recall@20', 'ndcg@20']


def checked(root, split):
    frame, histories = read_runs(root, split)
    for _, group in frame.groupby('dataset'):
        if group.source.nunique() != 1 or group.data_hash.nunique() != 1:
            raise ValueError('Mixed source or data')
    return frame


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage', choices=['reference', 'losses', 'sensitivity', 'robustness', 'cold'], required=True)
    p.add_argument('--root', default='results/reviewer_fixed')
    p.add_argument('--output', default='revision_outputs/reviewer_fixed')
    p.add_argument('--data-path', default='data')
    args = p.parse_args()
    root = Path(args.root); output = Path(args.output) / args.stage; output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.size': 9, 'pdf.fonttype': 42, 'axes.spines.top': False, 'axes.spines.right': False})
    if args.stage in ('reference', 'cold'):
        source = root / 'evaluated' / args.stage
        subprocess.run([sys.executable, str(ROOT / 'experiments/plot_revision.py'), str(source),
                        '--split', 'test', '--output', str(output / 'overall')], check=True)
        subprocess.run([sys.executable, str(ROOT / 'experiments/revision_cohorts.py'), str(source),
                        '--data-path', args.data_path, '--output', str(output / 'cohorts')], check=True)
        frame = checked(source, 'test')
        for _, group in frame.groupby('dataset'):
            seed_sets = [tuple(sorted(g.seed)) for _, g in group.groupby('variant')]
            if set(group.variant) != {'rev_11', 'gume'} or len(set(seed_sets)) != 1 or len(seed_sets[0]) < 3:
                raise ValueError('Reference/cold evidence requires both models with >=3 identical seeds')
        if args.stage == 'reference':
            rows = []
            for row in frame.itertuples():
                result = json.loads(Path(row.path).read_text())
                meta = result['checkpoint_evaluation']
                rows.append(dict(dataset=row.dataset, variant=row.variant, seed=row.seed,
                                 training_gpu=result.get('gpu_name'), inference_device=meta['device_name'],
                                 inference_torch=meta['torch'], training_torch=result.get('torch'),
                                 training_measurements=meta['training_measurements']))
            frame.to_csv(output / 'efficiency_runs.csv', index=False)
            pd.DataFrame(rows).to_csv(output / 'efficiency_provenance.csv', index=False)
            for dataset, group in frame.groupby('dataset'):
                if group.groupby('variant').setting.nunique().max() != 1:
                    raise ValueError('Multiple reference configurations')
                devices = pd.DataFrame(rows).query('dataset == @dataset')
                if (devices.inference_device.nunique() != 1 or devices.inference_torch.nunique() != 1
                        or devices.training_gpu.nunique() != 1 or devices.training_torch.nunique() != 1):
                    raise ValueError('Efficiency comparisons require matching hardware and torch within each measurement phase')
                columns = [('parameters', 'Trainable parameters'), ('epoch_seconds', 'Recorded training sec/epoch'),
                           ('gpu_peak_mib', 'Recorded training peak CUDA MiB'),
                           ('inference_total_seconds_median', 'Warmed inference sec / full pass')]
                fig, axes = plt.subplots(1, 4, figsize=(12, 3.7))
                for ax, (column, title) in zip(axes, columns):
                    values = group.groupby('variant')[column].agg(['mean', 'std'])
                    ax.bar(values.index, values['mean'], yerr=values['std'].fillna(0), capsize=3)
                    ax.set_title(title, fontsize=9)
                fig.suptitle(dataset + '; training records reused, inference measured now; mean +/- SD')
                save(fig, output, dataset + '_efficiency')
        for metric in ['recall@10', 'recall@20', 'ndcg@10', 'ndcg@20']:
            subprocess.run([sys.executable, str(ROOT / 'experiments/analyze_results.py'), str(source),
                            '--split', 'test', '--metric', metric, '--baseline', 'gume'], check=True)
        return
    if args.stage == 'losses':
        source = root / 'evaluated/losses'
        frame = checked(source, 'test')
        frame.to_csv(output / 'runs.csv', index=False)
        rows = []
        variants = ['rev_no_moments', 'rev_no_infonce', 'rev_no_alignment', 'rev_no_bm',
                    'rev_no_extended_bm', 'rev_no_um', 'rev_no_noise', 'rev_no_repr_l2', 'rev_no_extended_l2']
        for dataset, group in frame.groupby('dataset'):
            if group.duplicated(['variant', 'seed']).any():
                raise ValueError('Multiple loss settings / duplicate seeds')
            fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True)
            for ax, metric in zip(axes, METRICS):
                table = group.pivot(index='seed', columns='variant', values=metric)
                if table.isna().any().any() or len(table) < 3 or set(table.columns) != set(variants + ['rev_11']):
                    raise ValueError('Complete loss cohort with >=3 matching seeds required')
                family = []
                for i, variant in enumerate(variants):
                    delta = table[variant] - table.rev_11
                    s = paired_stats(delta)
                    family.append(dict(dataset=dataset, metric=metric, variant=variant, **s))
                    ax.errorbar(s['mean']*1000, i, xerr=(s['ci95_high']-s['mean'])*1000, fmt='o', capsize=3)
                    ax.scatter(delta*1000, i+np.linspace(-.1,.1,len(delta)), color='grey', s=12, alpha=.5)
                previous = 0.
                for rank, item in enumerate(sorted(family, key=lambda x: x['p_raw'])):
                    previous = min(1., max(previous, (len(family)-rank)*item['p_raw']))
                    item['p_holm'] = previous
                rows.extend(family)
                interaction = table.rev_11 - table.rev_no_moments - table.rev_no_infonce + table.rev_no_alignment
                rows.append(dict(dataset=dataset, metric=metric, variant='moments_x_infonce', **paired_stats(interaction)))
                ax.axvline(0, color='black', lw=.7, ls='--')
                ax.set(yticks=range(len(variants)), yticklabels=variants, xlabel=metric + ': removal - full (x 1,000)')
            axes[0].invert_yaxis()
            fig.suptitle(dataset + ' test; paired mean / 95% t interval; dots = seeds')
            save(fig, output, dataset + '_loss_effects')
            fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
            for ax, metric in zip(axes, METRICS):
                for label, names in [('InfoNCE off', ['rev_no_alignment', 'rev_no_infonce']),
                                     ('InfoNCE on', ['rev_no_moments', 'rev_11'])]:
                    series = [group[group.variant == v][metric] for v in names]
                    ax.errorbar([0,1], [s.mean() for s in series], yerr=[s.std(ddof=1) for s in series],
                                marker='o', capsize=3, label=label)
                ax.set(xticks=[0,1], xticklabels=['Moments off', 'Moments on'], ylabel=metric)
                ax.legend()
            fig.suptitle(dataset + ' test: fixed rev_11; mean +/- seed SD; both L2 penalties retained')
            save(fig, output, dataset + '_moments_infonce')
        pd.DataFrame(rows).to_csv(output / 'paired_statistics.csv', index=False)
        return
    if args.stage == 'sensitivity':
        frames = []
        for key in ['vt_loss', 'reg_weight_1', 'reg_weight_2']:
            frame = checked(root / 'training' / ('sensitivity_' + key), 'validation')
            frame['parameter'] = key; frames.append(frame)
        all_runs = pd.concat(frames, ignore_index=True); all_runs.to_csv(output / 'runs.csv', index=False)
        for dataset, data in all_runs.groupby('dataset'):
            fig, axes = plt.subplots(2, 3, figsize=(12, 7))
            for col, key in enumerate(['vt_loss', 'reg_weight_1', 'reg_weight_2']):
                group = data[data.parameter == key]
                if set(group.variant) != {'rev_11'} or group[key].nunique() != 4 or group.seed.nunique() < 3:
                    raise ValueError('Sensitivity requires rev_11, all four points and at least three seeds')
                if group.duplicated(['seed', key]).any():
                    raise ValueError('Duplicate sensitivity setting')
                if len({tuple(sorted(g.seed)) for _, g in group.groupby(key)}) != 1:
                    raise ValueError('Incomplete sensitivity seeds')
                for ax, metric in zip(axes[:, col], METRICS):
                    summary = group.groupby(key)[metric].agg(['mean', 'std'])
                    ax.errorbar([str(x) for x in summary.index], summary['mean'], yerr=summary['std'].fillna(0), marker='o', capsize=3)
                    ax.set(xlabel=key + ' (categorical spacing)', ylabel=metric)
            fig.suptitle(dataset + ' validation sensitivity; mean +/- seed SD; main configuration remains fixed')
            save(fig, output, dataset + '_sensitivity')
        return
    frames = []
    for kind in ['interactions', 'image', 'text']:
        frame = checked(root / 'evaluated' / ('noise_' + kind), 'test'); frame['noise'] = kind; frames.append(frame)
    all_runs = pd.concat(frames, ignore_index=True); all_runs.to_csv(output / 'runs.csv', index=False)
    degradation_rows = []
    for dataset, data in all_runs.groupby('dataset'):
        for metric in METRICS:
            fig, axes = plt.subplots(2, 3, figsize=(12, 7))
            for col, kind in enumerate(['interactions', 'image', 'text']):
                key = 'interaction_noise_rate' if kind == 'interactions' else 'feature_corruption_rate'
                group = data[data.noise == kind]
                seed_sets = [tuple(sorted(g.seed.unique())) for _, g in group.groupby('variant')]
                if set(group.variant) != {'rev_11', 'gume'} or len(set(seed_sets)) != 1:
                    raise ValueError('Noise comparisons require identical seeds for both models')
                for variant, sub in group.groupby('variant'):
                    if sub.duplicated(['seed', key]).any():
                        raise ValueError('Duplicate robustness condition')
                    table = sub.pivot(index='seed', columns=key, values=metric).sort_index(axis=1)
                    if table.isna().any().any() or set(table.columns) != {0., .1, .2} or len(table) < 3:
                        raise ValueError('Complete clean/noisy matching seeds required')
                    if (table[0.] <= 0).any():
                        raise ValueError('Relative degradation requires a positive clean score')
                    decline = (1-table.div(table[0.], axis=0))*100
                    for ax, values in [(axes[0,col], table), (axes[1,col], decline)]:
                        ax.errorbar(values.columns, values.mean(), yerr=values.std(ddof=1), marker='o', capsize=3, label=variant)
                    for seed, row in decline.iterrows():
                        for rate, value in row.items():
                            degradation_rows.append(dict(dataset=dataset, variant=variant, noise=kind, metric=metric,
                                                         seed=seed, rate=rate, degradation_percent=value))
                axes[0,col].set(title=kind, ylabel=metric); axes[0,col].legend()
                axes[1,col].set(xlabel='Selected noise rate', ylabel='Drop from own clean score (%)')
            fig.suptitle(dataset + ' test robustness; fixed configurations; mean +/- seed SD')
            save(fig, output, dataset + '_robustness_' + metric.replace('@','_'))
    pd.DataFrame(degradation_rows).to_csv(output / 'degradation_by_seed.csv', index=False)


if __name__ == '__main__':
    main()
