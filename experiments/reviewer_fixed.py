#!/usr/bin/env python3
"""Fixed rev_11 evidence, organized by reviewer concern. No architecture search."""
import argparse
import copy
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from run_suite import source_fingerprint
from utils.configurator import Config
from utils.quick_start import file_hash

IGNORED = {'variant', 'data_path', 'output_dir', 'tensorboard_log_dir', 'device', 'gpu_id', 'use_gpu'}
FIXED_SOURCE = '2e8303f646847ee71f10b79316f68537af1d339ebf6e8c5e17bd022ee33cc894'


def signature(cfg):
    return json.dumps({k: v for k, v in cfg.items() if k not in IGNORED}, sort_keys=True, default=str)


def specs(stage):
    if stage == 'reference':
        return [('reference', ['rev_11', 'gume'], {}, [])]
    if stage == 'losses':
        return [('losses', ['rev_11', 'rev_no_moments', 'rev_no_infonce', 'rev_no_alignment',
                           'rev_no_bm', 'rev_no_extended_bm', 'rev_no_um', 'rev_no_noise',
                           'rev_no_repr_l2', 'rev_no_extended_l2'], {}, [])]
    if stage == 'sensitivity':
        return [('sensitivity_' + key, ['rev_11'], {}, [key + '=' + json.dumps(values)])
                for key, values in {'vt_loss': [0., .01, .1, 1.],
                                    'reg_weight_1': [0., 1e-6, 1e-5, 1e-4],
                                    'reg_weight_2': [0., .01, .1, 1.]}.items()]
    if stage == 'robustness':
        return [('noise_interactions', ['rev_11', 'gume'], {}, ['interaction_noise_rate=[0.0,0.1,0.2]'])] + [
            ('noise_' + modality, ['rev_11', 'gume'],
             {'feature_corruption': 'shuffle', 'corruption_modality': modality, 'feature_corruption_seed': 314159},
             ['feature_corruption_rate=[0.0,0.1,0.2]']) for modality in ('image', 'text')]
    if stage == 'cold':
        return [('cold', ['rev_11', 'gume'], {'cold_item_fraction': .2}, [])]
    raise ValueError(stage)


def canonical(cfg):
    cfg = dict(cfg)
    # A zero-rate feature intervention has no effect; do not retrain its clean point.
    if cfg.get('feature_corruption_rate', 0.) == 0.:
        cfg.update(feature_corruption='none', corruption_modality='image', feature_corruption_seed=None)
    return signature(cfg)


def reuse(plans, sources, data_path, execute):
    index = {}
    current = source_fingerprint()
    # The caller puts affinity before anchor: prefer the designated five-seed cohort.
    for root in sources:
        if not root.exists():
            continue
        for path in sorted(root.rglob('result.json')):
            r = json.loads(path.read_text())
            if r.get('synthetic_fixture') or r.get('source_sha256') != current or r.get('test'):
                continue
            t = r['trials'][r['selected_trial']]
            index.setdefault(canonical(t['config']), (path, r))
    reused = 0
    hashes = {}
    for plan in plans:
        target = Path(plan['directory'])
        if (target / 'result.json').exists():
            continue
        request = json.loads((target / 'requested.json').read_text())
        cfg = Config(plan['command'][plan['command'].index('--model') + 1], plan['dataset'], dict(request)).final_config_dict
        found = index.get(canonical(cfg))
        if found is None:
            continue
        path, r = found
        for key, expected in r['data_sha256'].items():
            file = Path(data_path) / r['dataset'] / cfg[key]
            if not execute and not file.exists():
                continue  # Dry-run estimates; actual reuse always verifies bytes.
            if file not in hashes:
                hashes[file] = file_hash(file)
            if hashes[file] != expected:
                raise ValueError('Cannot reuse changed dataset: ' + str(file))
        reused += 1
        print(('REUSE ' if execute else 'CAN REUSE ') + str(path) + ' -> ' + str(target))
        if not execute:
            continue
        copied = copy.deepcopy(r)
        trial = copied['trials'][copied['selected_trial']]
        original_checkpoint = path.parent / ('trial_{:03d}'.format(copied['selected_trial'])) / 'best.pt'
        if original_checkpoint.exists():
            trial['checkpoint'] = str(original_checkpoint.resolve())
        for key in ['variant', 'data_path', 'output_dir', 'tensorboard_log_dir', 'feature_corruption',
                    'corruption_modality', 'feature_corruption_seed']:
            if key in cfg:
                trial['config'][key] = cfg[key]
        copied['reused_evidence'] = {'input_path': str(path.resolve()), 'sha256': file_hash(path),
                                     'note': 'Same resolved training configuration; zero-rate interventions canonicalized. No new training.'}
        history = path.parent / ('trial_{:03d}'.format(copied['selected_trial'])) / 'history.json'
        if history.exists():
            directory = target / history.parent.name; directory.mkdir(parents=True, exist_ok=True)
            shutil.copy2(history, directory / 'history.json')
        (target / 'result.json').write_text(json.dumps(copied, indent=2))
    return reused


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage', choices=['reference', 'losses', 'sensitivity', 'robustness', 'cold'], required=True)
    p.add_argument('--datasets', nargs='+', default=['baby'], choices=['baby', 'sports', 'clothing'])
    p.add_argument('--seeds', type=int, nargs='+', default=[999, 2024, 2025])
    p.add_argument('--data-path', default='data'); p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--output', default='results/reviewer_fixed'); p.add_argument('--cpu', action='store_true')
    p.add_argument('--execute', action='store_true')
    p.add_argument('--finish', action='store_true', help='After training, evaluate checkpoints and produce reviewer figures')
    p.add_argument('--figures-output', default='revision_outputs/reviewer_fixed')
    args = p.parse_args()
    if source_fingerprint() != FIXED_SOURCE:
        p.error('This protocol freezes the run_6 training source; review the protocol before changing models/configs')
    if args.finish and not args.execute:
        p.error('--finish requires --execute')
    root = Path(args.output).resolve(); root.mkdir(parents=True, exist_ok=True)
    commands, totals = [], {'planned': 0, 'existing': 0, 'reuse': 0}
    for label, variants, changes, sweeps in specs(args.stage):
        overrides = root / (label + '_overrides.json'); overrides.write_text(json.dumps(changes, indent=2))
        destination = root / 'training' / label
        command = [sys.executable, str(ROOT / 'experiments/run_suite.py'), '--variants', *variants,
                   '--datasets', *args.datasets, '--seeds', *map(str, args.seeds), '--data-path', args.data_path,
                   '--gpu', str(args.gpu), '--output', str(destination), '--overrides', str(overrides), '--resume']
        for sweep in sweeps:
            command += ['--sweep', sweep]
        if args.cpu:
            command += ['--cpu']
        subprocess.run(command, cwd=ROOT, check=True)
        plans = json.loads((destination / 'plan.json').read_text())
        totals['planned'] += len(plans)
        totals['existing'] += sum((Path(plan['directory']) / 'result.json').exists() for plan in plans)
        totals['reuse'] += reuse(plans, [ROOT / 'results/revision/affinity', ROOT / 'results/revision/anchor',
                                        root / 'training'], args.data_path, args.execute)
        commands.append(command)
        if args.execute:
            subprocess.run(command + ['--execute'], cwd=ROOT, check=True)
    totals['new_training'] = totals['planned'] - totals['existing'] - totals['reuse']
    (root / (args.stage + '_plan.json')).write_text(json.dumps({'commands': commands, 'counts': totals}, indent=2))
    print('Stage counts:', totals)
    if args.finish:
        if args.stage != 'sensitivity':
            for label, _, _, _ in specs(args.stage):
                command = [sys.executable, str(ROOT / 'experiments/evaluate_saved_revision.py'),
                           str(root / 'training' / label), '--output', str(root / 'evaluated' / label),
                           '--data-path', args.data_path, '--gpu', str(args.gpu)]
                if args.stage == 'reference':
                    command += ['--benchmark']
                if args.cpu:
                    command += ['--cpu']
                subprocess.run(command, cwd=ROOT, check=True)
        subprocess.run([sys.executable, str(ROOT / 'experiments/reviewer_figures.py'), '--stage', args.stage,
                        '--root', str(root), '--output', args.figures_output, '--data-path', args.data_path], cwd=ROOT, check=True)
    else:
        print('Training complete; add --finish to evaluate and plot.' if args.execute else 'Plan only. Reuse counts are provisional until data hashes are checked with --execute.')


if __name__ == '__main__':
    main()
