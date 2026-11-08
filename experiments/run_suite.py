#!/usr/bin/env python3
"""Plan/execute independent experiments. One subprocess and result directory per trial."""
import argparse
from collections import deque
import hashlib
import itertools
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]

# Each change is relative to its model's declared default, unless explicitly combined.
VARIANTS = {
    'amore_original': ('amorelegacy', {}),
    # old main.py defaulted to -m gume, which selects different historical settings.
    'amore_original_main': ('amorelegacy', {'affinity_threshold': '>0.5', 'um_loss': 0.01,
                                           'um_temp': 0.1, 'residual_alpha': 0.2}),
    'amore_fixed': ('amoreclean', {'coarse_affinity': True, 'bm_extended_loss': 0.005,
                                  'um_loss': 0.1, 'noise_loss': 0.1, 'reg_weight_2': 0.1}),
    'amore_compact': ('amoreclean', {}),
    'amore_compact_no_affinity': ('amoreclean', {'use_affinity_modulation': False}),
    'gume': ('gume', {}),
    'gume_no_vt': ('gume', {'vt_loss': 0.}),
    'gume_no_bm': ('gume', {'bm_loss': 0.}),
    'gume_no_alignment': ('gume', {'bm_loss': 0., 'vt_loss': 0.}),
    'gume_no_um': ('gume', {'um_loss': 0., 'reg_weight_2': 0.}),
    'gume_no_enhance': ('gume', {'enhance_graph': False}),
    'gume_coarse': ('gume', {'fusion': 'coarse'}),
    'gume_mean': ('gume', {'fusion': 'mean'}),
    'gume_frozen': ('gume', {'freeze_features': True}),
    'gume_lean': ('gume', {'um_loss': 0., 'reg_weight_2': 0., 'vt_loss': 0., 'bm_loss': 0.,
                          'freeze_features': True, 'fusion': 'mean', 'enhance_graph': False}),
    'gume_adapter8': ('gume', {'freeze_features': True, 'feature_adapter_rank': 8}),
    'gume_adapter16': ('gume', {'freeze_features': True, 'feature_adapter_rank': 16}),
    'gume_history_users': ('gume', {'extended_user_source': 'history'}),
    'gume_history_frozen': ('gume', {'extended_user_source': 'history', 'freeze_features': True}),
    'gume_ego_soft': ('gume', {'ego_mode': 'soft'}),
    'gume_ego_hard': ('gume', {'ego_mode': 'hard'}),
    'gume_ego_attention': ('gume', {'ego_mode': 'attention'}),
    'gume_ego_uniform': ('gume', {'ego_mode': 'uniform'}),
    'gume_ego_shuffle': ('gume', {'ego_mode': 'soft', 'ego_shuffle': True}),
    'gume_ego_self_message': ('gume', {'ego_mode': 'soft', 'ego_cross_source': 'self'}),
    'gume_ego_no_reattn': ('gume', {'ego_mode': 'soft', 'ego_reattention': False}),
    'gume_ego_self_query': ('gume', {'ego_mode': 'soft', 'ego_leave_one_out': False}),
    'gume_ego_hard_absolute': ('gume', {'ego_mode': 'hard', 'ego_threshold_type': 'absolute', 'ego_threshold': 0.4}),
    'gume_sg_local': ('gume', {'sg_init': 'local'}),
    'gume_sg': ('gume', {'sg_init': 'cluster'}),
    'gume_sg_shuffle': ('gume', {'sg_init': 'shuffle'}),
    'gume_sg_raw': ('gume', {'sg_init': 'cluster', 'sg_scale': 'raw'}),
    'gume_ego_soft_sg': ('gume', {'ego_mode': 'soft', 'sg_init': 'cluster'}),
    'gume_ego_hard_sg': ('gume', {'ego_mode': 'hard', 'sg_init': 'cluster'}),
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
    'contentcf': ('contentcf', {}),
    'contentcf_weak': ('contentcf', {'content_weight': 0.01}),
    'contentcf_text': ('contentcf', {'modality': 'text'}),
    'contentcf_image': ('contentcf', {'modality': 'image'}),
    'contentcf_cross': ('contentcf', {'cross_weight': 1.0}),
    # Independent architecture screen: content x training objective, plus K=1.
    'amorex_id_bpr': ('amorex', {'content_weight': 0., 'objective': 'bpr'}),
    'amorex_id_set': ('amorex', {'content_weight': 0.}),
    'amorex_single': ('amorex', {'num_interests': 1, 'objective': 'bpr'}),
    'amorex_multi': ('amorex', {'objective': 'bpr'}),
    'amorex': ('amorex', {}),
}
# Focused pruning of the full repaired original AMORE. Keep the same backbone
# and coefficients; each comparison below has an explicit reference condition.
for name, changes in {
    'amore_fixed_no_coarse': {'coarse_affinity': False},
    'amore_fixed_no_affinity': {'coarse_affinity': False, 'use_affinity_modulation': False},
    'amore_fixed_no_noise': {'noise_loss': 0.},
}.items():
    VARIANTS[name] = ('amoreclean', dict(VARIANTS['amore_fixed'][1], **changes))
for affinity, graph, initialization in itertools.product([False, True], repeat=3):
    VARIANTS['factor_{}{}{}'.format(int(affinity), int(graph), int(initialization))] = ('amore', {
        'affinity_mode': 'local' if affinity else 'none',
        'graph_mode': 'shared' if graph else 'separate',
        'ncer_strength': 1. if graph else 0., 'cooc_weight': 0.25 if graph else 0.,
        'user_init': 'history' if initialization else 'random'})

SUITES = {
    'original': ['amore_original'],
    'original_refine': ['amore_original', 'amore_fixed', 'amore_compact', 'amore_compact_no_affinity'],
    'original_focus': ['gume', 'amore_fixed', 'amore_fixed_no_coarse',
                       'amore_fixed_no_affinity', 'amore_fixed_no_noise'],
    'rebuild': ['gume', 'amorex_id_bpr', 'amorex_id_set', 'amorex_single', 'amorex_multi', 'amorex'],
    # A full-backbone screen, followed by attribution controls only if warranted.
    'gume_fusion': ['gume', 'gume_ego_soft', 'gume_ego_hard', 'gume_sg',
                    'gume_ego_soft_sg', 'gume_ego_hard_sg'],
    'ego_controls': ['gume', 'gume_ego_soft', 'gume_ego_hard', 'gume_ego_attention',
                     'gume_ego_uniform', 'gume_ego_shuffle', 'gume_ego_no_reattn',
                     'gume_ego_self_query', 'gume_ego_hard_absolute', 'gume_ego_self_message'],
    'sg_controls': ['gume', 'gume_sg_local', 'gume_sg', 'gume_sg_shuffle', 'gume_sg_raw'],
    'gume_paper': ['gume', 'gume_no_enhance', 'gume_no_alignment', 'gume_no_um'],
    # Stage two: identify what GUME needs before choosing an alternative mechanism.
    'recover': ['gume', 'gume_frozen', 'gume_no_bm', 'gume_no_um', 'gume_no_vt', 'gume_no_enhance', 'gume_mean'],
    'efficient': ['gume', 'gume_frozen', 'gume_adapter8', 'gume_adapter16', 'gume_history_users', 'gume_history_frozen'],
    'content': ['gume', 'amore_base', 'amore_id_only', 'amore_text', 'amore_image',
                'contentcf', 'contentcf_weak', 'contentcf_text', 'contentcf_image', 'contentcf_cross'],
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


def log_tail(path, lines=60):
    """Read only the last lines for display; retain the complete log on disk."""
    with path.open(encoding='utf-8', errors='replace') as stream:
        return ''.join(deque(stream, maxlen=lines)).rstrip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', choices=SUITES, default='original')
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
    print('Python:', sys.executable, flush=True)
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
            logfile = directory / 'console.log'
            excerpt = log_tail(logfile)
            failures.append(dict(plan, returncode=status.returncode, log_tail=excerpt))
            print('FAILED (exit code {}); full log: {}'.format(status.returncode, logfile), flush=True)
            print('Last 60 log lines:\n{}'.format(excerpt or '(log is empty)'), flush=True)
    (output / 'failures.json').write_text(json.dumps(failures, indent=2))
    if failures:
        raise SystemExit('{} runs failed'.format(len(failures)))


if __name__ == '__main__':
    main()
