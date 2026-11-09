#!/usr/bin/env python3
"""Loss pruning and paired corruption curves from the run_6 follow-up stages."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from plot_revision import read_runs, save
from analyze_run6 import paired_stats


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', default='results/revision_followup')
    p.add_argument('--output', default='revision_outputs/followup')
    args = p.parse_args()
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    frames = []
    for directory in sorted(Path(args.root).glob('followup_*')):
        if not list(directory.rglob('result.json')):
            continue
        frame, _ = read_runs(directory, 'validation')
        frame['condition'] = directory.name[len('followup_'):]
        frames.append(frame)
    if not frames:
        raise ValueError('No follow-up results')
    frame = pd.concat(frames, ignore_index=True)
    frame.to_csv(output / 'runs.csv', index=False)
    records = []
    plt.rcParams.update({'font.size': 9, 'pdf.fonttype': 42, 'axes.spines.top': False, 'axes.spines.right': False})
    for dataset, data in frame.groupby('dataset'):
        if data.source.nunique() != 1 or data.data_hash.nunique() != 1:
            raise ValueError('Mixed source or data')
        base = data[data.condition == 'full']
        if base.empty:
            raise ValueError('Need full-condition references')
        def series(group, metric):
            if group.duplicated('seed').any():
                raise ValueError('Multiple settings / repeated seeds in a condition')
            return group.set_index('seed')[metric].sort_index()
        def paired(a, b, metric):
            a, b = series(a, metric), series(b, metric)
            if not a.index.equals(b.index) or len(a) < 3:
                raise ValueError('Complete matching seeds (at least three) required; finish all planned runs first')
            return paired_stats(a - b)
        changes = [x for x in ['no_moments', 'no_infonce', 'no_alignment', 'no_noise', 'no_extended'] if x in set(data.condition)]
        if changes:
            fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), sharey=True)
            for ax, metric in zip(axes, ['recall@20', 'ndcg@20']):
                for j, variant in enumerate(['rev_00', 'rev_11']):
                    for i, condition in enumerate(changes):
                        s = paired(data[(data.condition == condition) & (data.variant == variant)],
                                   base[base.variant == variant], metric)
                        records.append(dict(dataset=dataset, variant=variant, condition=condition, metric=metric, **s))
                        ax.errorbar(s['mean'] * 1000, i + (j-.5)*.2,
                                    xerr=(s['ci95_high']-s['mean'])*1000, fmt='o', capsize=3,
                                    color=['#2878a2', '#cf7040'][j], label=variant if i == 0 else None)
                ax.axvline(0, color='black', lw=.7, ls='--')
                ax.set(yticks=range(len(changes)), yticklabels=changes, xlabel=metric + ' difference (x 1,000)')
                ax.legend()
            axes[0].invert_yaxis()
            fig.suptitle(dataset + ' validation: removal - full objective; paired mean and 95% t CI')
            save(fig, output, dataset + '_loss_removal')
            fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), sharey=True)
            conditions = ['full'] + changes
            for ax, metric in zip(axes, ['recall@20', 'ndcg@20']):
                reference_delta = (series(base[base.variant == 'rev_11'], metric) -
                                   series(base[base.variant == 'rev_00'], metric))
                for i, condition in enumerate(conditions):
                    group = data[data.condition == condition]
                    a, b = group[group.variant == 'rev_11'], group[group.variant == 'rev_00']
                    s = paired(a, b, metric)
                    records.append(dict(dataset=dataset, variant='rev_11 - rev_00', condition=condition, metric=metric, **s))
                    if condition != 'full':
                        interaction = series(a, metric) - series(b, metric) - reference_delta
                        if interaction.isna().any():
                            raise ValueError('Objective-by-affinity interaction requires common seeds')
                        records.append(dict(dataset=dataset, variant='affinity x objective removal', condition=condition,
                                            metric=metric, **paired_stats(interaction)))
                    ax.errorbar(s['mean'] * 1000, i, xerr=(s['ci95_high']-s['mean'])*1000,
                                fmt='o', capsize=3, color='#174c6b')
                ax.axvline(0, color='black', lw=.7, ls='--')
                ax.set(yticks=range(len(conditions)), yticklabels=conditions,
                       xlabel=metric + ': both - neither (x 1,000)')
            axes[0].invert_yaxis()
            fig.suptitle(dataset + ' validation: affinity effect under each objective; paired 95% t CI')
            save(fig, output, dataset + '_affinity_by_objective')
        noise = [x for x in ['noise_interactions', 'noise_image', 'noise_text'] if x in set(data.condition)]
        if noise:
            for metric in ['recall@20', 'ndcg@20']:
                fig, axes = plt.subplots(2, len(noise), figsize=(4*len(noise), 7), squeeze=False)
                for col, condition in enumerate(noise):
                    perturbed = data[data.condition == condition].copy()
                    key = 'interaction_noise_rate' if condition == 'noise_interactions' else 'feature_corruption_rate'
                    clean = base.copy(); clean[key] = 0.
                    points = pd.concat([clean, perturbed], ignore_index=True)
                    for j, variant in enumerate(['rev_00', 'rev_11']):
                        means, sds, rates = [], [], []
                        for rate, group in points[points.variant == variant].groupby(key):
                            values = series(group, metric)
                            if not values.index.equals(series(base[base.variant == variant], metric).index):
                                raise ValueError('Stress and clean seed sets differ')
                            rates.append(rate); means.append(values.mean()); sds.append(values.std(ddof=1))
                        axes[0, col].errorbar(rates, means, yerr=sds, marker='o', capsize=3,
                                             label=variant, color=['#2878a2', '#cf7040'][j])
                    for rate, group in points.groupby(key):
                        s = paired(group[group.variant == 'rev_11'], group[group.variant == 'rev_00'], metric)
                        records.append(dict(dataset=dataset, variant='rev_11 - rev_00', condition=condition,
                                            rate=rate, metric=metric, **s))
                        axes[1, col].errorbar(rate, s['mean']*1000, yerr=(s['ci95_high']-s['mean'])*1000,
                                             fmt='o', capsize=3, color='#174c6b')
                    axes[0, col].set(title=condition, ylabel=metric + ' (mean +/- SD)')
                    axes[0, col].legend()
                    axes[1, col].axhline(0, color='black', ls='--', lw=.7)
                    axes[1, col].set(xlabel='Selected perturbation rate', ylabel='Both - neither (x 1,000); 95% CI')
                fig.suptitle(dataset + ' validation; models retrained under each perturbation')
                save(fig, output, dataset + '_stress_' + metric.replace('@', '_'))
    pd.DataFrame(records).to_csv(output / 'paired_statistics.csv', index=False)
    print('Figures saved to', output)


if __name__ == '__main__':
    main()
