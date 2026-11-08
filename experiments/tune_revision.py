#!/usr/bin/env python3
"""Equal candidate-count validation search; plans only unless --execute."""
import argparse
from itertools import product
import json
from pathlib import Path
import random
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
# Explicit finite spaces. These are search ranges, never claims of optimality.
SPACES = {
    'mm_bpr': {'learning_rate': [0.0001, 0.001], 'reg_weight': [0.00001, 0.0001, 0.001, 0.01, 0.1, 1.]},
    'mm_lightgcn': {'learning_rate': [0.0001, 0.001], 'n_layers': [1, 2, 3, 4], 'reg_weight': [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]},
    'mm_bm3': {'learning_rate': [0.0001, 0.001], 'n_layers': [1, 2], 'dropout': [0.3, 0.5], 'reg_weight': [0.01, 0.1]},
    'mm_mgcn': {'learning_rate': [0.0001, 0.001], 'cl_loss': [0.001, 0.01, 0.1], 'reg_weight': [1e-5, 1e-4, 1e-3]},
    'mm_freedom': {'learning_rate': [0.0001, 0.001], 'dropout': [0.8, 0.9], 'reg_weight': [0., 1e-5, 1e-4, 1e-3]},
    'mm_lgmrec': {'learning_rate': [0.0001, 0.001], 'n_hyper_layer': [1, 2], 'hyper_num': [4, 64],
                  'keep_rate': [0.2, 0.4, 0.5], 'alpha': [0.2, 0.3, 0.6], 'cl_weight': [1e-5, 1e-4, 1e-3]},
    'mm_mentor': {'learning_rate': [0.0001, 0.001], 'mask_weight_g': [0.0001, 0.001], 'temp': [0.2, 0.4, 0.6]},
    'gume': {'learning_rate': [0.0001, 0.001], 'bm_loss': [0.001, 0.01], 'um_loss': [0.01, 0.1], 'vt_loss': [0.001, 0.01, 0.1]},
    'rev_11': {'learning_rate': [0.0001, 0.001], 'vt_loss': [0., 0.01, 0.1], 'reg_weight_1': [1e-6, 1e-5, 1e-4],
               'reg_weight_2': [0.001, 0.01, 0.1], 'affinity_threshold': ['>0.05', '>0.1', '>0.2']},
}
DEFAULT_VARIANTS = list(SPACES)
for name in ['rev_00', 'rev_10', 'rev_01']:
    SPACES[name] = dict(SPACES['rev_11'])


def candidates(variant, budget, dataset='baby'):
    keys = list(SPACES[variant])
    pool = [dict(zip(keys, values)) for values in product(*(SPACES[variant][k] for k in keys))]
    random.Random(314159).shuffle(pool)
    # Always include the published-repository screening configuration.
    # {} is a distinct candidate and never silently called an optimized setting.
    sys.path.insert(0, str(ROOT / 'src'))
    from utils.configurator import Config
    model_file = 'GUME' if variant == 'gume' else 'AMOREREVISION' if variant.startswith('rev_') else variant.upper()
    defaults = Config(model_file.lower(), dataset, {'use_gpu': False}).final_config_dict
    pool = [v for v in pool if any(defaults.get(k) != val for k, val in v.items())]
    if not 1 <= budget <= len(pool) + 1:
        raise ValueError('Budget outside candidate space for ' + variant)
    return [{}] + pool[:budget - 1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--datasets', nargs='+', choices=['baby', 'sports', 'clothing'], default=['baby'])
    p.add_argument('--variants', nargs='+', choices=list(SPACES), default=DEFAULT_VARIANTS)
    p.add_argument('--budget', type=int, default=12)
    p.add_argument('--seeds', type=int, nargs='+', default=[999, 2024, 2025])
    p.add_argument('--output', default='results/revision/tuning')
    p.add_argument('--data-path', default='data')
    p.add_argument('--epochs', type=int, default=1000)
    p.add_argument('--patience', type=int, default=20)
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--execute', action='store_true')
    args = p.parse_args()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    commands = []
    for dataset, variant in product(args.datasets, args.variants):
        for index, values in enumerate(candidates(variant, args.budget, dataset)):
            path = output / ('{}_{}_candidate_{:02d}.json'.format(dataset, variant, index))
            path.write_text(json.dumps(values, indent=2))
            cmd = [sys.executable, str(ROOT / 'experiments/run_suite.py'), '--variants', variant,
                   '--datasets', dataset, '--seeds', *map(str, args.seeds), '--epochs', str(args.epochs),
                   '--patience', str(args.patience), '--data-path', args.data_path, '--gpu', str(args.gpu),
                   '--overrides', str(path), '--output', str(output / 'runs'), '--resume']
            if args.execute:
                cmd.append('--execute')
            commands.append(cmd)
    (output / 'tuning_commands.json').write_text(json.dumps(commands, indent=2))
    for cmd in commands:
        subprocess.run(cmd, cwd=ROOT, check=True)


if __name__ == '__main__':
    main()
