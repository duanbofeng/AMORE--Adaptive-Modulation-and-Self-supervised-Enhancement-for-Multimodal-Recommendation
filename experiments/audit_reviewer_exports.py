#!/usr/bin/env python3
"""Audit the Baby reference/cold CSV exports without training or changing evidence.

This checks aggregate consistency, not checkpoint/ranking authenticity. Raw results
and rankings must be synchronized separately for a full reproduction audit.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


METRICS = ['recall@10', 'recall@20', 'ndcg@10', 'ndcg@20']
COHORTS = ['user_sparse', 'user_medium', 'user_dense', 'head_items',
           'tail_items', 'cold_items'] + ['item_quintile_' + str(i) for i in range(1, 6)]
SOURCE = '2e8303f646847ee71f10b79316f68537af1d339ebf6e8c5e17bd022ee33cc894'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def paired(frame, metric):
    table = frame.pivot(index='seed', columns='variant', values=metric).sort_index()
    require(not table.isna().any().any(), 'Missing matched seed: ' + metric)
    delta = table.rev_11 - table.gume
    mean, sd = float(delta.mean()), float(delta.std(ddof=1))
    radius = float(stats.t.ppf(.975, len(delta) - 1) * sd / np.sqrt(len(delta)))
    # Identically zero differences have no estimable t statistic; do not invent a p value.
    p = float(stats.ttest_rel(table.rev_11, table.gume).pvalue) if sd > 0 else None
    return dict(metric=metric, n=len(delta), mean_difference=mean, sd_difference=sd,
                ci95_low=mean-radius, ci95_high=mean+radius, p_raw=p,
                seed_differences={str(k): float(v) for k, v in delta.items()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('revision_outputs/reviewer_fixed'))
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    report = dict(scope='Baby reference/cold aggregate exports only; raw evidence not verified',
                  paired_method='rev_11 minus GUME; paired t CI; Holm across four overall metrics separately per stage',
                  stages={}, input_sha256={})
    frames = {}
    data_hash = None
    for stage, seeds in [('reference', [999, 2024, 2025, 2026, 2027]), ('cold', [999, 2024, 2025])]:
        folder = args.root / stage
        runs = pd.read_csv(folder / 'overall/runs.csv')
        summary = pd.read_csv(folder / 'overall/summary.csv')
        cohorts = pd.read_csv(folder / 'cohorts/cohort_metrics.csv')
        frames[stage] = runs
        for file in sorted(folder.rglob('*.csv')):
            report['input_sha256'][str(file)] = hashlib.sha256(file.read_bytes()).hexdigest()
        expected = {(v, s) for v in ['gume', 'rev_11'] for s in seeds}
        require(len(runs) == len(expected) and set(zip(runs.variant, runs.seed)) == expected, stage + ': run grid')
        require(set(runs.dataset) == {'baby'} and set(runs.split) == {'test'}, stage + ': dataset/split')
        require(set(runs.source) == {SOURCE} and set(cohorts.source) == {SOURCE}, stage + ': reported source')
        hashes = {json.dumps(json.loads(v), sort_keys=True) for v in list(runs.data_hash) + list(cohorts.data)}
        require(len(hashes) == 1, stage + ': reported data hashes differ')
        current_hash = next(iter(hashes))
        require(data_hash in (None, current_hash), 'Data hashes differ across stages')
        data_hash = current_hash
        expected_conditions = dict(cold_item_fraction=0. if stage == 'reference' else .2,
                                   interaction_noise_rate=0., feature_corruption_rate=0.)
        for key, value in expected_conditions.items():
            require((runs[key] == value).all(), stage + ': ' + key)
        require(all(json.loads(c) == expected_conditions for c in cohorts.condition), stage + ': cohort conditions')
        require((runs.best_epoch < runs.epochs_run).all(), stage + ': epoch bounds')
        require(runs.groupby('variant').setting.nunique().eq(1).all(), stage + ': mixed settings')
        numeric = runs[[m + '@' + str(k) for m in ['recall', 'ndcg', 'precision', 'map'] for k in [5, 10, 20, 50]]]
        require(np.isfinite(numeric).all().all() and ((numeric >= 0) & (numeric <= 1)).all().all(), stage + ': metrics')
        summary_keys = {(v, m) for v in ['gume', 'rev_11'] for m in METRICS}
        require(len(summary) == len(summary_keys) and set(zip(summary.variant, summary.metric)) == summary_keys,
                stage + ': summary grid')
        for row in summary.itertuples():
            sample = runs.loc[runs.variant == row.variant, row.metric]
            radius = stats.t.ppf(.975, len(sample)-1) * sample.std(ddof=1) / np.sqrt(len(sample))
            expected_values = [sample.mean(), sample.std(ddof=1), sample.mean()-radius, sample.mean()+radius]
            require(row.n == len(sample) and np.allclose([row.mean, row.sd, row.ci95_low, row.ci95_high],
                    expected_values, atol=1e-12, rtol=1e-10), stage + ': summary does not reproduce')
        cohort_keys = {(v, s, c, k) for v, s in expected for c in COHORTS for k in [5, 10, 20, 50]}
        require(len(cohorts) == len(cohort_keys) and set(zip(cohorts.variant, cohorts.seed, cohorts.cohort, cohorts.k)) == cohort_keys,
                stage + ': cohort grid')
        require(np.isfinite(cohorts[['recall', 'ndcg']]).all().all() and
                cohorts[['recall', 'ndcg']].ge(0).all().all() and cohorts[['recall', 'ndcg']].le(1).all().all(), stage + ': cohort metrics')
        require((cohorts.positives >= cohorts.eligible_users).all() and (cohorts.eligible_users > 0).all(), stage + ': counts')
        require(cohorts.groupby('cohort')[['eligible_users', 'positives']].nunique().eq(1).all().all(), stage + ': population differs')
        max_error = 0.
        for (variant, seed, k), group in cohorts.groupby(['variant', 'seed', 'k']):
            by_name = group.set_index('cohort')
            users = group[group.cohort.str.startswith('user_')]
            overall = runs[(runs.variant == variant) & (runs.seed == seed)].iloc[0]
            require(int(users.eligible_users.sum()) == 19445 and int(users.positives.sum()) == 21682, stage + ': user totals')
            for metric in ['recall', 'ndcg']:
                error = abs(float(np.average(users[metric], weights=users.eligible_users)) - overall[metric + '@' + str(k)])
                max_error = max(max_error, error)
                require(error < 1e-10, stage + ': user cohorts do not reconstruct overall metrics')
            require(np.allclose(by_name.loc['head_items', ['recall', 'ndcg']].astype(float),
                                by_name.loc['item_quintile_1', ['recall', 'ndcg']].astype(float)), stage + ': head != quintile 1')
            require(by_name.loc[['head_items', 'tail_items', 'cold_items'], 'positives'].sum() == users.positives.sum(), stage + ': item totals')
            require(by_name.loc[['item_quintile_' + str(i) for i in range(2, 6)], 'positives'].sum() == by_name.loc['tail_items', 'positives'], stage + ': tail totals')
        comparisons = [paired(runs, m) for m in METRICS]
        running = 0.
        for entry in comparisons:
            entry['p_holm_four_metrics'] = None
        for i, entry in enumerate(sorted([c for c in comparisons if c['p_raw'] is not None], key=lambda x: x['p_raw'])):
            running = max(running, min(1., (len(comparisons)-i)*entry['p_raw']))
            entry['p_holm_four_metrics'] = running
        cohort_summary = []
        for (variant, name), group in cohorts[cohorts.k == 20].groupby(['variant', 'cohort']):
            cohort_summary.append(dict(variant=variant, cohort=name, n=len(group),
                eligible_users=int(group.eligible_users.iloc[0]), positives=int(group.positives.iloc[0]),
                recall_mean=float(group.recall.mean()), recall_sd=float(group.recall.std(ddof=1)),
                ndcg_mean=float(group.ndcg.mean()), ndcg_sd=float(group.ndcg.std(ddof=1))))
        report['stages'][stage] = dict(aggregate_checks='passed', runs=len(runs), cohort_rows=len(cohorts),
            seeds=seeds, max_user_reconstruction_error=max_error, summary=summary.to_dict('records'),
            paired_overall=comparisons, cohort_at20=cohort_summary)
    efficiency = pd.read_csv(args.root / 'reference/efficiency_runs.csv')
    pd.testing.assert_frame_equal(efficiency, frames['reference'])
    provenance = pd.read_csv(args.root / 'reference/efficiency_provenance.csv')
    require(len(provenance) == 10 and set(zip(provenance.variant, provenance.seed)) ==
            set(zip(efficiency.variant, efficiency.seed)), 'Efficiency provenance coverage')
    require((efficiency.inference_repeats == 5).all() and (efficiency.inference_warmup_passes == 1).all()
            and (efficiency.inference_users == 19445).all(), 'Inference protocol')
    columns = ['parameters', 'epoch_seconds', 'gpu_peak_mib', 'inference_encoding_seconds_median',
               'inference_ranking_seconds_median', 'inference_total_seconds_median']
    require(np.isfinite(efficiency[columns]).all().all() and efficiency[columns].gt(0).all().all(), 'Efficiency values')
    report['efficiency'] = {variant: {col: dict(mean=float(group[col].mean()), sd=float(group[col].std(ddof=1)))
                           for col in columns} for variant, group in efficiency.groupby('variant')}
    report['raw_evidence_note'] = ('CSV-reported source/data hashes agree; input bytes, checkpoint selection, '
        'training histories and item identities cannot be reverified without raw JSON/rankings/data. '
        'test_evaluation_seconds is absent; inference benchmark has separate populated fields. '
        'Cohort means are descriptive; no cohort significance claim is made.')
    report['reported_data_hashes'] = json.loads(data_hash)
    target = args.output or args.root / 'export_audit.json'
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print('Aggregate consistency checks passed; raw-evidence audit remains separate: ' + str(target))


if __name__ == '__main__':
    main()
