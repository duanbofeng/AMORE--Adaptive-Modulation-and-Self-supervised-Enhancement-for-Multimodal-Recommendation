#!/usr/bin/env python3
"""Aggregate matched-seed runs and train-degree cohorts without choosing on test scores."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


def cohort_metrics(result, trial, path):
    ranking_file = Path(trial['config']['tensorboard_log_dir']) / 'test_rankings.npz'
    if not ranking_file.exists():
        return {}
    cfg = trial['config']
    frame = pd.read_csv(Path(cfg['data_path']) / result['dataset'] / cfg['inter_file_name'], sep=cfg['field_separator'])
    uid, iid, split = cfg['USER_ID_FIELD'], cfg['ITEM_ID_FIELD'], cfg['inter_splitting_label']
    training, testing = frame[frame[split] == 0], frame[frame[split] == 2]
    user_degree = training.groupby(uid).size()
    item_degree = training.groupby(iid).size()
    user_thresholds = user_degree.quantile([1 / 3, 2 / 3]).values
    item_threshold = item_degree.quantile(0.2)
    positives = testing.groupby(uid)[iid].agg(set).to_dict()
    arrays = np.load(ranking_file)
    output = {}
    for k in cfg['topk']:
        groups = defaultdict(list)
        for user, predicted in zip(arrays['users'], arrays['topk'][:, :k]):
            truth = positives[user]
            degree = user_degree.get(user, 0)
            group = 'sparse' if degree <= user_thresholds[0] else 'medium' if degree <= user_thresholds[1] else 'dense'
            hits = np.array([int(item) in truth for item in predicted])
            recall = hits.sum() / len(truth)
            ndcg = (hits / np.log2(np.arange(k) + 2)).sum() / (1 / np.log2(np.arange(min(k, len(truth))) + 2)).sum()
            groups['user_' + group + '/recall'].append(recall)
            groups['user_' + group + '/ndcg'].append(ndcg)
            for label, predicate in [('cold', lambda d: d == 0), ('tail', lambda d: 0 < d <= item_threshold),
                                     ('head', lambda d: d > item_threshold)]:
                target = {i for i in truth if predicate(item_degree.get(i, 0))}
                if target:
                    groups['item_' + label + '/recall'].append(len(set(predicted) & target) / len(target))
        for name, values in groups.items():
            output[name + '@' + str(k)] = {'mean': float(np.mean(values)), 'users': len(values)}
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root')
    parser.add_argument('--baseline', default='amore_base')
    parser.add_argument('--metric', default='recall@20')
    parser.add_argument('--split', choices=['validation', 'test'], default='validation')
    args = parser.parse_args()
    root = Path(args.root)
    records = defaultdict(dict)
    cohorts = {}
    for path in sorted(root.rglob('result.json')):
        result = json.loads(path.read_text())
        trial = result['trials'][result['selected_trial']]
        cfg = trial['config']
        metrics = result[args.split]
        if args.metric not in metrics:
            continue
        variant = cfg.get('variant', result['model'])
        # Do not silently pool hyperparameters, stress levels, protocols or code versions.
        ignored = {'seed', 'tensorboard_log_dir', 'output_dir', 'gpu_id', 'device', 'variant'}
        comparison_cfg = {k: v for k, v in cfg.items() if k not in ignored}
        comparison_cfg['_source_sha256'] = result.get('source_sha256', cfg.get('execution_fingerprint'))
        comparison_cfg['_data_sha256'] = result['data_sha256']
        import hashlib
        setting = hashlib.sha256(json.dumps(comparison_cfg, sort_keys=True).encode()).hexdigest()[:10]
        key = (result['dataset'], variant, setting)
        seed = int(cfg['seed'])
        if seed in records[key]:
            raise ValueError('Duplicate seed for same setting: ' + str(path))
        records[key][seed] = {'score': metrics[args.metric], 'trial': trial,
                              'hashes': result['data_sha256'], 'protocol': result['protocol'],
                              'synthetic_fixture': result.get('synthetic_fixture', False),
                              'source_sha256': result.get('source_sha256', cfg.get('execution_fingerprint'))}
        if result['test']:
            cohorts[str(path)] = cohort_metrics(result, trial, path)
    rows, comparisons = [], []
    for (dataset, variant, setting), seeds in sorted(records.items()):
        scores = np.array([r['score'] for r in seeds.values()])
        n = len(scores)
        std = float(scores.std(ddof=1)) if n > 1 else None
        interval = float(stats.t.ppf(0.975, n - 1) * std / np.sqrt(n)) if n > 1 else None
        row = {'dataset': dataset, 'variant': variant, 'setting': setting, 'n_seeds': n,
               'mean': float(scores.mean()), 'sample_std': std,
               'synthetic_fixture': any(r['synthetic_fixture'] for r in seeds.values()),
               'mean_ci95': [float(scores.mean() - interval), float(scores.mean() + interval)] if interval is not None else None,
               'trainable_parameters': [r['trial']['parameters_trainable'] for r in seeds.values()],
               'mean_epoch_seconds': float(np.mean([r['trial']['mean_epoch_seconds'] for r in seeds.values()]))}
        rows.append(row)
        bases = [value for (d, v, _), value in records.items() if d == dataset and v == args.baseline]
        if variant == args.baseline or len(bases) != 1:
            continue
        baseline = bases[0]
        paired = sorted(set(seeds) & set(baseline))
        if len(paired) < 3:
            continue
        # Only compare identical source data and stress/budget conditions.
        match_keys = ['train_fraction', 'feature_corruption', 'feature_corruption_rate', 'corruption_modality',
                      'epochs', 'stopping_step', 'execution_fingerprint', 'topk', 'valid_metric']
        if any(seeds[s]['hashes'] != baseline[s]['hashes'] or seeds[s]['protocol'] != baseline[s]['protocol']
               or seeds[s]['source_sha256'] != baseline[s]['source_sha256']
               or any(seeds[s]['trial']['config'].get(k) != baseline[s]['trial']['config'].get(k) for k in match_keys)
               for s in paired):
            continue
        delta = np.array([seeds[s]['score'] - baseline[s]['score'] for s in paired])
        stderr = delta.std(ddof=1) / np.sqrt(len(delta))
        half = stats.t.ppf(0.975, len(delta) - 1) * stderr
        pvalue = float(stats.ttest_1samp(delta, 0).pvalue) if stderr > 0 else None
        comparisons.append({'dataset': dataset, 'variant': variant, 'setting': setting, 'baseline': args.baseline,
                            'paired_seeds': paired, 'mean_delta': float(delta.mean()),
                            'paired_ci95': [float(delta.mean() - half), float(delta.mean() + half)], 'p_raw': pvalue})
    # Holm correction across all finite comparisons in this invocation.
    ordered = sorted([c for c in comparisons if c['p_raw'] is not None and np.isfinite(c['p_raw'])], key=lambda c: c['p_raw'])
    previous = 0.
    for i, comparison in enumerate(ordered):
        previous = max(previous, min(1., (len(ordered) - i) * comparison['p_raw']))
        comparison['p_holm'] = previous
    output = {'split': args.split, 'metric': args.metric, 'summary': rows, 'paired_comparisons': comparisons,
              'test_cohorts': cohorts, 'notes': [
                  'Confidence intervals quantify variation across these seeds, not across datasets.',
                  'Multiple settings are not pooled; select settings using validation before final testing.',
                  'Cohort thresholds use original training degrees; item recall conditions on users with targets in that group.',
                  'All test runs used train-only masking, matching the repository protocol.',
                  'A favorable validation comparison is exploratory evidence, not a confirmed test improvement.']}
    destination = root / ('analysis_' + args.split + '_' + args.metric.replace('@', '_') + '.json')
    destination.write_text(json.dumps(output, indent=2, allow_nan=False))
    print('| Dataset | Variant | Setting | Seeds | Mean | Sample std | Epoch seconds |')
    print('|---|---|---|---:|---:|---:|---:|')
    for row in rows:
        std = '{:.6f}'.format(row['sample_std']) if row['sample_std'] is not None else 'NA'
        print('| {} | {} | {} | {} | {:.6f} | {} | {:.3f} |'.format(
            row['dataset'], row['variant'], row['setting'], row['n_seeds'], row['mean'], std, row['mean_epoch_seconds']))
    print('Analysis:', destination)


if __name__ == '__main__':
    main()
