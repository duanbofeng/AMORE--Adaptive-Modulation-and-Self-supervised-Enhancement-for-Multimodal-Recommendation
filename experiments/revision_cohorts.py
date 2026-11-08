#!/usr/bin/env python3
"""Full-ranking test metrics by train-degree cohorts; head=top 20% of warm items."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('root')
    p.add_argument('--output', required=True)
    p.add_argument('--data-path', help='Override data root when analyzing copied results')
    p.add_argument('--allow-synthetic', action='store_true')
    args = p.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in sorted(Path(args.root).rglob('result.json')):
        result = json.loads(path.read_text())
        if not result['test']:
            continue
        if result.get('synthetic_fixture') and not args.allow_synthetic:
            raise ValueError('Synthetic cohort refused')
        trial = result['trials'][result['selected_trial']]
        cfg = trial['config']
        rankings = np.load(path.parent / ('trial_{:03d}'.format(result['selected_trial'])) / 'test_rankings.npz')
        source = Path(args.data_path or cfg['data_path']) / result['dataset'] / cfg['inter_file_name']
        if hashlib.sha256(source.read_bytes()).hexdigest() != result['data_sha256']['inter_file_name']:
            raise ValueError('Interaction file does not match the saved ranking cohort')
        data = pd.read_csv(source, sep=cfg['field_separator'])
        uid, iid, label = cfg['USER_ID_FIELD'], cfg['ITEM_ID_FIELD'], cfg['inter_splitting_label']
        clean = data[data[label] == 0]
        user_degree = clean.groupby(uid).size()
        item_degree = clean.groupby(iid).size().to_dict()
        removed = set(trial.get('intervention', {}).get('cold_items', []))
        for item in removed:
            item_degree[item] = 0
        warm = sorted((i for i, degree in item_degree.items() if degree > 0), key=lambda i: (-item_degree[i], i))
        quintiles = {int(i): q + 1 for q, items in enumerate(np.array_split(warm, 5)) for i in items}
        cutoffs = user_degree.quantile([1/3, 2/3]).to_numpy()
        truth = data[data[label] == 2].groupby(uid)[iid].agg(set).to_dict()
        for k in cfg['topk']:
            groups = {}
            for user, ranked in zip(rankings['users'], rankings['topk'][:, :k]):
                positives = truth[int(user)]
                degree = user_degree.get(user, 0)
                bucket = 'sparse' if degree <= cutoffs[0] else ('medium' if degree <= cutoffs[1] else 'dense')
                targets = {'user_' + bucket: positives}
                for name, predicate in [('cold_items', lambda i: item_degree.get(i, 0) == 0),
                                        ('head_items', lambda i: quintiles.get(i) == 1),
                                        ('tail_items', lambda i: quintiles.get(i, 0) >= 2)]:
                    targets[name] = {i for i in positives if predicate(i)}
                for q in range(1, 6):
                    targets['item_quintile_' + str(q)] = {i for i in positives if quintiles.get(i) == q}
                for name, target in targets.items():
                    if not target:
                        continue
                    hits = np.array([int(item) in target for item in ranked])
                    recall = float(hits.sum() / len(target))
                    ndcg = float((hits / np.log2(np.arange(len(ranked)) + 2)).sum() /
                                 (1/np.log2(np.arange(min(k, len(target))) + 2)).sum())
                    groups.setdefault(name, []).append((recall, ndcg, len(target)))
            for name, values in groups.items():
                array = np.asarray(values)
                rows.append(dict(dataset=result['dataset'], variant=cfg.get('variant', result['model']),
                                 seed=cfg['seed'], cohort=name, k=k, recall=array[:, 0].mean(),
                                 ndcg=array[:, 1].mean(), eligible_users=len(values), positives=int(array[:, 2].sum()),
                                 condition=json.dumps({key: cfg.get(key) for key in ['cold_item_fraction', 'interaction_noise_rate',
                                                                                   'feature_corruption_rate']}, sort_keys=True),
                                 source=result.get('source_sha256'), data=json.dumps(result['data_sha256'], sort_keys=True)))
    if not rows:
        raise ValueError('No test rankings; run the locked final protocol with export_rankings=true')
    frame = pd.DataFrame(rows)
    frame.to_csv(output / 'cohort_metrics.csv', index=False)
    for (dataset, condition), group in frame[frame.k == 20].groupby(['dataset', 'condition']):
        if group.source.nunique() > 1 or group.data.nunique() > 1:
            raise ValueError('Mixed source/data cohort')
        if group.duplicated(['variant', 'seed', 'cohort']).any():
            raise ValueError('Multiple configurations per variant; use a locked cohort')
        fig, ax = plt.subplots(figsize=(8, 4.5))
        order = ['user_sparse', 'user_medium', 'user_dense', 'head_items', 'tail_items', 'cold_items']
        for variant, sub in group.groupby('variant'):
            summary = sub.groupby('cohort').recall.agg(['mean', 'std']).reindex(order)
            ax.errorbar(order, summary['mean'], yerr=summary['std'].fillna(0), marker='o', capsize=3, label=variant)
        ax.set(ylabel='Cohort Recall@20', title=('SYNTHETIC QA ONLY ' if args.allow_synthetic else '') + dataset)
        ax.tick_params(axis='x', rotation=25)
        ax.legend(fontsize=7)
        fig.tight_layout()
        suffix = hashlib.sha256(condition.encode()).hexdigest()[:8]
        for extension in ('png', 'pdf'):
            fig.savefig(output / (dataset + '_cohorts_' + suffix + '.' + extension), dpi=300)
        plt.close(fig)
    print(output)


if __name__ == '__main__':
    main()
