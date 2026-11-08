#!/usr/bin/env python3
"""Freeze per-dataset configurations chosen on validation; never read test scores."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('root')
    p.add_argument('--dataset', required=True)
    p.add_argument('--variants', nargs='+', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    ignored = {'seed', 'model', 'dataset', 'variant', 'output_dir', 'tensorboard_log_dir', 'device',
               'execution_fingerprint', 'hyper_parameters', 'evaluate_test', 'gpu_id', 'use_gpu', 'data_path'}
    groups = defaultdict(dict)
    for path in sorted(Path(args.root).rglob('result.json')):
        r = json.loads(path.read_text())
        cfg = r['trials'][r['selected_trial']]['config']
        variant = cfg.get('variant')
        if r['dataset'] != args.dataset or variant not in args.variants:
            continue
        if r.get('synthetic_fixture'):
            raise ValueError('Synthetic results cannot lock a paper configuration')
        if r.get('test'):
            raise ValueError('Selection input must be validation-only, not a previously tested cohort')
        if (cfg.get('interaction_noise_rate', 0) or cfg.get('cold_item_fraction', 0)
                or cfg.get('train_fraction', 1) != 1 or cfg.get('feature_corruption_rate', 0)):
            continue
        overrides = {k: v for k, v in cfg.items() if k not in ignored}
        evidence = (r['source_sha256'], json.dumps(r['data_sha256'], sort_keys=True))
        signature = hashlib.sha256(json.dumps([overrides, evidence], sort_keys=True).encode()).hexdigest()
        key = (variant, signature)
        if cfg['seed'] in groups[key]:
            raise ValueError('Duplicate run for the same configuration and seed')
        groups[key][cfg['seed']] = (r['validation']['recall@20'], overrides, str(path), evidence)
    if len({value[3] for group in groups.values() for value in group.values()}) > 1:
        raise ValueError('All selection candidates must share source and data hashes')
    counts = {variant: sum(v == variant for v, _ in groups) for variant in args.variants}
    if len(set(counts.values())) != 1:
        raise ValueError('Unequal completed candidate counts across models: ' + str(counts))
    models, selected_evidence, cohort = [], [], None
    for variant in args.variants:
        candidates = [g for (v, _), g in groups.items() if v == variant]
        if not candidates or any(len(g) < 3 for g in candidates):
            raise ValueError('Need at least three validation seeds per candidate: ' + variant)
        if len({tuple(sorted(g)) for g in candidates}) != 1:
            raise ValueError('All hyperparameter candidates must have the same seed set')
        selected = max(candidates, key=lambda g: sum(v[0] for v in g.values()) / len(g))
        first = next(iter(selected.values()))
        identity = (tuple(sorted(selected)), first[3])
        if cohort is not None and identity != cohort:
            raise ValueError('Models differ in selection seeds, source version or data hashes')
        cohort = identity
        models.append({'variant': variant, 'overrides': first[1]})
        selected_evidence.append({'variant': variant, 'runs': [v[2] for v in selected.values()],
                                  'mean_validation_recall20': sum(v[0] for v in selected.values()) / len(selected),
                                  'source_and_data': first[3]})
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps({'dataset': args.dataset, 'selection_metric': 'validation recall@20',
                                 'models': models, 'selection_evidence': selected_evidence}, indent=2))
    print(target)


if __name__ == '__main__':
    main()
