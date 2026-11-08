#!/usr/bin/env python3
"""Publication plots and tables from actual result.json files; no invented values."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def read_runs(root, split, allow_synthetic=False):
    rows, histories = [], {}
    for path in sorted(Path(root).rglob('result.json')):
        r = json.loads(path.read_text())
        if r.get('synthetic_fixture') and not allow_synthetic:
            raise ValueError('Synthetic fixture refused: ' + str(path))
        t = r['trials'][r['selected_trial']]
        cfg = t['config']
        if not r[split]:
            continue
        ignored = {'seed', 'variant', 'tensorboard_log_dir', 'output_dir', 'device', 'gpu_id'}
        signature = {k: v for k, v in cfg.items() if k not in ignored}
        signature['source_sha256'] = r.get('source_sha256')
        signature['data_sha256'] = r['data_sha256']
        setting = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()[:10]
        row = dict(dataset=r['dataset'], variant=cfg.get('variant', r['model']), seed=cfg['seed'], setting=setting,
                   source=r.get('source_sha256'), data_hash=json.dumps(r['data_sha256'], sort_keys=True),
                   split=split, epoch_seconds=t['mean_epoch_seconds'], setup_seconds=t['setup_seconds'],
                   parameters=t['parameters_trainable'], best_epoch=t['best_epoch'], epochs_run=t['epochs_run'],
                   gpu_peak_mib=(t.get('peak_cuda_bytes') / 2**20 if t.get('peak_cuda_bytes') is not None else np.nan),
                   test_evaluation_seconds=t.get('test_evaluation_seconds', np.nan),
                   path=str(path), **r[split])
        for key, value in t.get('inference_benchmark', {}).items():
            if isinstance(value, (int, float)):
                row['inference_' + key] = value
        for key in ['interaction_noise_rate', 'feature_corruption_rate', 'corruption_modality',
                    'cold_item_fraction', 'vt_loss', 'reg_weight_1', 'reg_weight_2', 'affinity_threshold']:
            row[key] = cfg.get(key)
        rows.append(row)
        history_path = path.parent / ('trial_{:03d}'.format(r['selected_trial'])) / 'history.json'
        if history_path.exists():
            histories[str(path)] = json.loads(history_path.read_text())
    if not rows:
        raise ValueError('No completed results for requested split')
    frame = pd.DataFrame(rows)
    if frame.duplicated(['dataset', 'variant', 'setting', 'seed']).any():
        raise ValueError('Duplicate seed for a setting; use a single completed run cohort')
    return frame, histories


def save(fig, output, stem):
    fig.tight_layout()
    for extension in ['pdf', 'png']:
        fig.savefig(output / (stem + '.' + extension), dpi=300, bbox_inches='tight')
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('root')
    p.add_argument('--output', required=True)
    p.add_argument('--split', choices=['validation', 'test'], default='validation')
    p.add_argument('--axis', choices=['interaction_noise_rate', 'feature_corruption_rate', 'vt_loss',
                                     'reg_weight_1', 'reg_weight_2', 'affinity_threshold'])
    p.add_argument('--allow-synthetic', action='store_true', help='QA only; figures prominently marked')
    args = p.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    frame, histories = read_runs(args.root, args.split, args.allow_synthetic)
    frame.to_csv(output / 'runs.csv', index=False)
    metrics = [m for m in ['recall@10', 'recall@20', 'ndcg@10', 'ndcg@20'] if m in frame]
    if not metrics:
        raise ValueError('Paper plots require K=10/20; do not use the reduced smoke configuration')
    summaries = []
    for key, group in frame.groupby(['dataset', 'variant', 'setting']):
        for metric in metrics:
            values = group[metric].to_numpy()
            n, mean = len(values), float(values.mean())
            sd = float(values.std(ddof=1)) if n > 1 else np.nan
            ci = stats.t.ppf(.975, n - 1) * sd / np.sqrt(n) if n > 1 else np.nan
            summaries.append(dict(dataset=key[0], variant=key[1], setting=key[2], metric=metric,
                                  n=n, mean=mean, sd=sd, ci95_low=mean-ci, ci95_high=mean+ci))
    pd.DataFrame(summaries).to_csv(output / 'summary.csv', index=False)
    plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False,
                         'pdf.fonttype': 42, 'ps.fonttype': 42})
    mark = 'SYNTHETIC QA ONLY — ' if args.allow_synthetic else ''
    for dataset, subset in frame.groupby('dataset'):
        if subset['source'].nunique() > 1 or subset['data_hash'].nunique() > 1:
            raise ValueError('Mixed source/data hashes: separate plots by experiment cohort')
        metric = 'recall@20' if 'recall@20' in subset else metrics[0]
        if args.axis:
            axis = args.axis
            # Distinct parameter settings at a single x must never be pooled.
            if subset.groupby(['variant', axis], dropna=False)['setting'].nunique().max() > 1:
                raise ValueError('Multiple settings per variant/x; plot each controlled sweep separately')
            fig, ax = plt.subplots(figsize=(6.4, 4))
            for variant, group in subset.groupby('variant'):
                summary = group.groupby(axis, dropna=False)[metric].agg(['mean', 'std']).sort_index()
                ax.errorbar(summary.index.astype(str), summary['mean'], yerr=summary['std'].fillna(0),
                            marker='o', capsize=3, label=variant)
            ax.set(xlabel=axis, ylabel=metric, title=mark + dataset + ' ' + args.split)
            ax.legend(fontsize=7)
            save(fig, output, dataset + '_sweep')
            continue
        if subset.groupby('variant')['setting'].nunique().max() > 1:
            raise ValueError('Multiple settings per variant; use a single cohort or --axis')
        fig, axes = plt.subplots(1, len(metrics), figsize=(max(8, len(metrics)*3.2), 4.6), squeeze=False)
        for metric, ax in zip(metrics, axes[0]):
            values = subset.groupby('variant')[metric].agg(['mean', 'std', 'count'])
            ax.bar(values.index, values['mean'], yerr=values['std'].fillna(0), capsize=3, color='#386b8d')
            ax.tick_params(axis='x', rotation=70)
            ax.set(title=metric, ylabel='Mean ± sample SD')
        fig.suptitle(mark + dataset + ' ' + args.split + ' — missing SD denotes one seed')
        save(fig, output, dataset + '_performance')
        fig, ax = plt.subplots(figsize=(8, 4.4))
        markers = ['o', 's', '^', 'D', 'v', 'P', 'X', '<', '>']
        for index, (variant, group) in enumerate(subset.groupby('variant')):
            x, y = group['epoch_seconds'].mean(), group['recall@20'].mean()
            ax.scatter(x, y, s=65, label=variant, marker=markers[index % len(markers)], alpha=.8)
        ax.set(xlabel='Training seconds per epoch', ylabel='Recall@20', title=mark + dataset + ' accuracy and cost')
        ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=8)
        save(fig, output, dataset + '_efficiency')
        # 2x2 interaction effect from matched optimization seeds, no unpaired pooling.
        variants = ['rev_00', 'rev_10', 'rev_01', 'rev_11']
        if set(variants) <= set(subset['variant']):
            pivot = subset[subset.variant.isin(variants)].pivot(index='seed', columns='variant', values='recall@20').dropna()
            delta = pivot.rev_11 - pivot.rev_10 - pivot.rev_01 + pivot.rev_00
            pd.DataFrame({'seed': delta.index, 'interaction_delta_recall20': delta.values}).to_csv(
                output / (dataset + '_interaction.csv'), index=False)
    # Weighted loss components are descriptive; they are not gradient magnitudes.
    for dataset, subset in frame.groupby('dataset'):
        fig, ax = plt.subplots(figsize=(6.5, 4.2))
        for _, row in subset.iterrows():
            history = histories.get(row['path'], [])
            points = [(v['epoch'], v['validation'].get('recall@20')) for v in history if 'validation' in v]
            points = [(e, y) for e, y in points if y is not None]
            if points:
                ax.plot(*zip(*points), alpha=.65, label=row['variant'] + '/' + str(row['seed']))
        ax.set(xlabel='Epoch', ylabel='Validation Recall@20', title=mark + dataset + ' convergence')
        if ax.lines:
            ax.legend(fontsize=6, ncol=2)
            save(fig, output, dataset + '_convergence')
        else:
            plt.close(fig)
    print(output)


if __name__ == '__main__':
    main()
