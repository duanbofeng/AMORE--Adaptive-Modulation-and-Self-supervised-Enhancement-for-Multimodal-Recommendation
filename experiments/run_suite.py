#!/usr/bin/env python3
"""Plan/execute independent experiments. One subprocess and result directory per trial."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]

# Each change is relative to its model's declared default, unless explicitly combined.
VARIANTS = {
    'gume': ('gume', {}),
    'gume_no_vt': ('gume', {'vt_loss': 0.}),
    'gume_no_bm': ('gume', {'bm_loss': 0.}),
    'gume_no_um': ('gume', {'um_loss': 0., 'reg_weight_2': 0.}),
    'gume_no_enhance': ('gume', {'enhance_graph': False}),
    'gume_coarse': ('gume', {'fusion': 'coarse'}),
    'gume_mean': ('gume', {'fusion': 'mean'}),
    'gume_frozen': ('gume', {'freeze_features': True}),
    'gume_lean': ('gume', {'um_loss': 0., 'reg_weight_2': 0., 'vt_loss': 0., 'bm_loss': 0.,
                          'freeze_features': True, 'fusion': 'mean', 'enhance_graph': False}),
    'amore_base': ('amore', {}),
    'amore_id_only': ('amore', {'semantic_weight': 0.}),
    'amore_train_features': ('amore', {'freeze_features': False}),
    'amore_no_ii': ('amore', {'n_layers': 0}),
    'amore_attention': ('amore', {'fusion': 'attention'}),
    'amore_global': ('amore', {'affinity_mode': 'global'}),
    'amore_item': ('amore', {'affinity_mode': 'item'}),
    'amore_local': ('amore', {'affinity_mode': 'local'}),
    'amore_shuffle': ('amore', {'affinity_mode': 'shuffle'}),
    'amore_shared': ('amore', {'graph_mode': 'shared'}),
    'amore_ncer': ('amore', {'graph_mode': 'shared', 'ncer_strength': 1.}),
    'amore_cooc': ('amore', {'graph_mode': 'shared', 'cooc_weight': 0.25}),
    'amore_shared_refined': ('amore', {'graph_mode': 'shared', 'cooc_weight': 0.25, 'ncer_strength': 1.}),
    'amore_gate': ('amore', {'residual_gate': True}),
    'amore_half_scale': ('amore', {'semantic_weight': 1 / 6}),
    'amore_history_init': ('amore', {'user_init': 'history'}),
    'amore_cluster_init': ('amore', {'user_init': 'cluster'}),
    'amore_cl': ('amore', {'cl_weight': 0.01}),
    'amore_image': ('amore', {'modality': 'image'}),
    'amore_text': ('amore', {'modality': 'text'}),
}
for affinity, graph, initialization in itertools.product([False, True], repeat=3):
    VARIANTS['factor_{}{}{}'.format(int(affinity), int(graph), int(initialization))] = ('amore', {
        'affinity_mode': 'local' if affinity else 'none',
        'graph_mode': 'shared' if graph else 'separate',
        'ncer_strength': 1. if graph else 0., 'cooc_weight': 0.25 if graph else 0.,
        'user_init': 'history' if initialization else 'random'})

SUITES = {
    'core': ['gume', 'gume_lean', 'amore_base', 'amore_global', 'amore_item', 'amore_local', 'amore_shuffle'],
    'prune': [v for v in VARIANTS if v.startswith('gume')],
    'mechanisms': [v for v in VARIANTS if v.startswith('amore')],
    'factorial': [v for v in VARIANTS if v.startswith('factor')],
    'all': list(VARIANTS),
}


def source_fingerprint():
    digest = hashlib.sha256()
    files = list((ROOT / 'src').rglob('*.py')) + list((ROOT / 'src/configs').rglob('*.yaml'))
    for path in sorted(files):
        digest.update(str(path.relative_to(ROOT)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', choices=SUITES, default='core')
    parser.add_argument('--variants', nargs='+', choices=VARIANTS)
    parser.add_argument('--datasets', nargs='+', default=['baby', 'sports', 'clothing'])
    parser.add_argument('--seeds', nargs='+', type=int, default=[999])
    parser.add_argument('--output', default='results/screen')
    parser.add_argument('--data-path', default=str(ROOT / 'data'))
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--overrides', help='JSON object file applied to each variant')
    parser.add_argument('--sweep', action='append', default=[], help='KEY=JSON_LIST, Cartesian product')
    parser.add_argument('--test', action='store_true', help='Explicit final evaluation after validation-based design selection')
    parser.add_argument('--execute', action='store_true', help='Without this flag, only write the plan')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    variants = args.variants or SUITES[args.suite]
    overrides = json.loads(Path(args.overrides).read_text()) if args.overrides else {}
    if set(overrides) & {'model', 'dataset', 'seed', 'output_dir', 'hyper_parameters', 'evaluate_test'}:
        parser.error('Use dedicated CLI options for model/dataset/seed/output/test, not overrides')
    sweeps = {}
    for entry in args.sweep:
        key, values = entry.split('=', 1)
        sweeps[key] = json.loads(values)
        if not isinstance(sweeps[key], list) or not sweeps[key]:
            parser.error('Each sweep must be a nonempty JSON list')
    if set(sweeps) & {'model', 'dataset', 'seed', 'output_dir', 'hyper_parameters', 'evaluate_test'}:
        parser.error('Cannot sweep reserved run-control parameters')
    grid = [dict(zip(sweeps, values)) for values in itertools.product(*sweeps.values())]
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    plans = []
    fingerprint = source_fingerprint()
    for dataset, variant, seed, swept in itertools.product(args.datasets, variants, args.seeds, grid):
        model, changes = VARIANTS[variant]
        config = dict(changes, **overrides)
        config.update(swept)
        config.update(seed=seed, hyper_parameters=['seed'], epochs=args.epochs,
                      stopping_step=args.patience, gpu_id=args.gpu, use_gpu=not args.cpu,
                      data_path=str(Path(args.data_path).resolve()), evaluate_test=args.test,
                      variant=variant, execution_fingerprint=fingerprint)
        signature = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:12]
        directory = output / dataset / variant / ('seed_{}_{}'.format(seed, signature))
        directory.mkdir(parents=True, exist_ok=True)
        cfgfile = directory / 'requested.json'
        cfgfile.write_text(json.dumps(config, indent=2))
        command = [sys.executable, str(ROOT / 'src/main.py'), '--model', model, '--dataset', dataset,
                   '--config', str(cfgfile), '--output-dir', str(directory)]
        plans.append({'dataset': dataset, 'variant': variant, 'seed': seed,
                      'directory': str(directory), 'command': command})
    (output / 'plan.json').write_text(json.dumps(plans, indent=2))
    print('{} runs; plan: {}'.format(len(plans), output / 'plan.json'), flush=True)
    if not args.execute:
        return
    failures = []
    for index, plan in enumerate(plans, 1):
        directory = Path(plan['directory'])
        result = directory / 'result.json'
        if result.exists():
            if args.resume:
                # Also verify data content, not just code/config, before reusing a completed run.
                sys.path.insert(0, str(ROOT / 'src'))
                from utils.quick_start import file_hash
                previous = json.loads(result.read_text())
                cfg = previous['trials'][previous['selected_trial']]['config']
                same = all(file_hash(Path(cfg['data_path']) / cfg['dataset'] / cfg[key]) == digest
                           for key, digest in previous['data_sha256'].items())
                if same:
                    print('Resume: completed', directory, flush=True)
                    continue
                raise RuntimeError('Data changed; use a new output directory: ' + str(directory))
            raise FileExistsError('Completed result exists; use --resume or a new output: ' + str(directory))
        print('[{}/{}] {} {} seed={}'.format(index, len(plans), plan['dataset'], plan['variant'], plan['seed']), flush=True)
        with (directory / 'console.log').open('w') as stream:
            status = subprocess.run(plan['command'], cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
        if status.returncode:
            failures.append(plan)
            print('FAILED; inspect', directory / 'console.log', flush=True)
    (output / 'failures.json').write_text(json.dumps(failures, indent=2))
    if failures:
        raise SystemExit('{} runs failed'.format(len(failures)))


if __name__ == '__main__':
    main()
