#!/usr/bin/env python3
"""Check cross-table consistency of the learning scenarios."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent


def main():
    tables = {p.stem: pd.read_csv(p) for p in (ROOT / 'data').glob('*.csv')}
    checks = []
    for name, frame in tables.items():
        assert frame.provenance.eq('synthetic_learning_example').all(), name
        allowed_nulls = ['weight'] if name == 'cohort_runs' else []
        assert not frame.drop(columns=allowed_nulls).isna().any().any(), name
    checks.append('Source fields and missing values')
    perf = tables['performance_runs'].set_index(['dataset', 'model', 'seed']).sort_index()
    abl = tables['ablation_runs'].set_index(['dataset', 'variant', 'seed'])
    metrics = ['recall@10', 'recall@20', 'ndcg@10', 'ndcg@20']
    for (ds, variant, seed), row in abl.iterrows():
        if variant == 'Full':
            np.testing.assert_allclose(row[metrics].astype(float), perf.loc[(ds, 'AMORE', seed), metrics].astype(float), atol=1e-10, rtol=0)
    checks.append('Full ablation equals main model for each seed and metric')
    refs = {'Moment weight': (.1, 'No moments'), 'Representation L2': (1e-5, 'No representation L2'), 'Extended L2': (.1, 'No extended L2')}
    for row in tables['sensitivity_runs'].itertuples():
        reference, variant = refs[row.parameter]
        if row.weight not in (0, reference):
            continue
        expected = (abl.loc[(row.dataset, variant, row.seed)] if row.weight == 0
                    else perf.loc[(row.dataset, 'AMORE', row.seed)])
        np.testing.assert_allclose([row.recall, row.ndcg], expected[['recall@20', 'ndcg@20']].astype(float), atol=1e-10, rtol=0)
    checks.append('Sensitivity zero/reference points equal matching ablation/main configuration')
    for row in tables['robustness_runs'].itertuples():
        clean = perf.loc[(row.dataset, row.model, row.seed), row.metric]
        np.testing.assert_allclose(row.clean, clean, atol=1e-10, rtol=0)
        if row.rate == 0:
            np.testing.assert_allclose(row.value, clean, atol=1e-10, rtol=0)
        np.testing.assert_allclose(row.degradation_percent, 100*(1-row.value/clean), atol=2e-7, rtol=0)
    checks.append('Zero-noise anchors and paired degradation calculations')
    users = tables['cohort_runs'].query('family == "User activity"')
    for key, group in users.groupby(['dataset', 'model', 'seed']):
        np.testing.assert_allclose(group.weight.sum(), 1, atol=1e-12)
        for metric in ['recall', 'ndcg']:
            np.testing.assert_allclose((group.weight*group[metric]).sum(), perf.loc[key, metric+'@20'], atol=1e-10, rtol=0)
    checks.append('Weighted user groups reconcile to overall performance')
    for row in tables['table_05_paired_comparison'].itertuples():
        a = perf.loc[(row.dataset, 'AMORE'), row.metric]
        b = perf.loc[(row.dataset, 'MENTOR'), row.metric]
        delta = a-b
        half = stats.t.ppf(.975, len(delta)-1)*delta.std(ddof=1)/np.sqrt(len(delta))
        np.testing.assert_allclose([row.paired_delta, row.ci95_low, row.ci95_high],
                                   [delta.mean(), delta.mean()-half, delta.mean()+half], atol=1e-10, rtol=0)
    checks.append('Paired differences and t intervals recalculated from runs')
    assert tables['efficiency_runs'].groupby('model').parameters_million.nunique().eq(1).all()
    checks.append('Parameter counts invariant across seeds')
    manifest = json.loads((ROOT / 'manifest.json').read_text())
    for figure in manifest['figures']:
        for ext in ['png', 'svg']:
            assert (ROOT / 'figures' / (figure['stem']+'.'+ext)).stat().st_size > 0
    checks.append('Every listed figure has both PNG and SVG exports')
    report = dict(status='passed', checks=checks, csv_files=len(tables),
                  figures=len(manifest['figures']))
    (ROOT / 'validation.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
