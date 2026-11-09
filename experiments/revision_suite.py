#!/usr/bin/env python3
"""Stage reviewer experiments on the original AMORE. Dry-run unless --execute."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
SEEDS = [999, 2024, 2025, 2026, 2027]


def stage_jobs(stage, datasets, seeds, lock=None):
    def job(variants=None, suite=None, overrides=None, sweep=None, label=None, test=False):
        return dict(variants=variants, suite=suite, overrides=overrides or {}, sweep=sweep or [],
                    label=label or stage, test=test)
    if stage == 'anchor':
        return [job(suite='revision_anchor')]
    if stage == 'baselines':
        return [job(suite='revision_baselines')]
    if stage == 'affinity':
        return [job(suite='revision_affinity')]
    if stage == 'losses':
        return [job(suite='revision_losses')]
    if stage in ('loss-factorial', 'prune-followup'):
        # Cross objectives with affinity; a loss-only study on rev_11 cannot
        # identify whether affinity depends on auxiliary supervision.
        conditions = {'full': {}}
        if stage == 'loss-factorial':
            no_cl = dict(bm_loss=0., bm_extended_loss=0., um_loss=0., noise_loss=0.)
            conditions.update(no_moments={'vt_loss': 0.}, no_infonce=no_cl,
                              no_alignment=dict(no_cl, vt_loss=0.))
        else:
            conditions.update(no_noise={'noise_loss': 0.},
                              no_extended=dict(bm_extended_loss=0., um_loss=0., noise_loss=0., reg_weight_2=0.))
        return [job(['rev_00', 'rev_11'], overrides=changes, label='followup_' + name)
                for name, changes in conditions.items()]
    if stage == 'stress-followup':
        # One shared clean reference, followed by six fixed perturbations.
        return [job(['rev_00', 'rev_11'], label='followup_full'),
                job(['rev_00', 'rev_11'], sweep=['interaction_noise_rate=[0.1,0.2]'],
                    label='followup_noise_interactions')] + [
            job(['rev_00', 'rev_11'], overrides={'feature_corruption': 'shuffle', 'corruption_modality': modality,
                                             'feature_corruption_seed': 314159},
                sweep=['feature_corruption_rate=[0.1,0.2]'], label='followup_noise_' + modality)
            for modality in ('image', 'text')]
    if stage == 'sensitivity':
        return [job(['rev_11'], sweep=[key + '=' + json.dumps(values)], label='sensitivity_' + key)
                for key, values in {
                    'vt_loss': [0., 0.001, 0.01, 0.1, 1.],
                    'reg_weight_1': [0., 1e-6, 1e-5, 1e-4, 1e-3],
                    'reg_weight_2': [0., 0.001, 0.01, 0.1, 1.],
                    'affinity_threshold': ['>0.01', '>0.05', '>0.1', '>0.2', '>0.5']}.items()]
    if stage == 'robustness':
        return [job(['gume', 'rev_00', 'rev_11'], sweep=['interaction_noise_rate=[0.0,0.05,0.1,0.2]'],
                    label='noise_interactions')] + [
            job(['gume', 'rev_00', 'rev_11'], overrides={'feature_corruption': 'shuffle', 'corruption_modality': modality,
                                                     'feature_corruption_seed': 314159},
                sweep=['feature_corruption_rate=[0.0,0.1,0.2,0.4]'], label='noise_' + modality)
            for modality in ('image', 'text')]
    if stage == 'cold':
        return [job(['gume', 'rev_00', 'rev_11'], overrides={'cold_item_fraction': 0.2})]
    if stage == 'confirm':
        return [job(['gume', 'amore_original', 'amore_fixed', 'rev_00', 'rev_10', 'rev_01', 'rev_11'])]
    if stage in ('final', 'stress-final'):
        if not lock or not lock.get('models') or not lock.get('selection_evidence'):
            raise ValueError('Final stage requires --lock created by freeze_revision.py from validation results')
        if stage == 'stress-final':
            jobs = []
            for entry in lock['models']:
                if entry['variant'] not in ('gume', 'mm_mentor', 'rev_00', 'rev_10', 'rev_01', 'rev_11'):
                    continue
                base = dict(entry['overrides'], benchmark_repeats=0)
                jobs.append(job([entry['variant']], overrides=base,
                                sweep=['interaction_noise_rate=[0.0,0.05,0.1,0.2]'], label='stress_final_interactions', test=True))
                jobs.append(job([entry['variant']], overrides=dict(base, cold_item_fraction=.2),
                                label='stress_final_cold', test=True))
                for modality in ('image', 'text'):
                    changed = dict(base, feature_corruption='shuffle', corruption_modality=modality,
                                   feature_corruption_seed=314159)
                    jobs.append(job([entry['variant']], overrides=changed,
                                    sweep=['feature_corruption_rate=[0.0,0.1,0.2,0.4]'],
                                    label='stress_final_' + modality, test=True))
            if not jobs:
                raise ValueError('Lock has no eligible stress comparators')
            return jobs
        return [job([entry['variant']], overrides=dict(entry['overrides'], benchmark_repeats=5), label='final', test=True)
                for entry in lock['models']]
    raise ValueError(stage)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage', required=True, choices=['anchor', 'baselines', 'affinity', 'losses', 'sensitivity', 'robustness', 'cold', 'confirm', 'final', 'stress-final', 'loss-factorial', 'prune-followup', 'stress-followup'])
    p.add_argument('--datasets', nargs='+', default=['baby'])
    p.add_argument('--seeds', type=int, nargs='+', default=SEEDS)
    p.add_argument('--data-path', default='data')
    p.add_argument('--output', default='results/revision')
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--epochs', type=int, default=1000)
    p.add_argument('--patience', type=int, default=20)
    p.add_argument('--lock')
    p.add_argument('--cpu', action='store_true')
    p.add_argument('--execute', action='store_true')
    args = p.parse_args()
    if any(d not in ('baby', 'sports', 'clothing') for d in args.datasets):
        p.error('This revision is restricted to the three requested Amazon datasets')
    lock = json.loads(Path(args.lock).read_text()) if args.lock else None
    if args.stage in ('final', 'stress-final') and lock:
        from run_suite import source_fingerprint
        if args.datasets != [lock['dataset']]:
            p.error('Run final separately for each dataset using its own validation lock')
        for entry in lock['models']:
            if (entry['overrides'].get('epochs') != args.epochs or
                    entry['overrides'].get('stopping_step') != args.patience):
                p.error('Final epochs and patience must match the frozen selection budget')
        for entry, evidence in zip(lock['models'], lock['selection_evidence']):
            expected_source, expected_data = evidence['source_and_data']
            if source_fingerprint() != expected_source:
                p.error('Source changed after validation selection; repeat selection on this source version')
            actual_data = {}
            for key in json.loads(expected_data):
                path = Path(args.data_path) / lock['dataset'] / entry['overrides'][key]
                digest = hashlib.sha256()
                with path.open('rb') as stream:
                    for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                        digest.update(chunk)
                actual_data[key] = digest.hexdigest()
            if actual_data != json.loads(expected_data):
                p.error('Dataset bytes changed after validation selection')
    jobs = stage_jobs(args.stage, args.datasets, args.seeds, lock)
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    commands = []
    for i, spec in enumerate(jobs):
        overrides = root / ('{}_overrides_{}.json'.format(args.stage, i))
        overrides.write_text(json.dumps(spec['overrides'], indent=2))
        command = [sys.executable, str(ROOT / 'experiments/run_suite.py'), '--datasets', *args.datasets,
                   '--seeds', *map(str, args.seeds), '--data-path', args.data_path, '--gpu', str(args.gpu),
                   '--epochs', str(args.epochs), '--patience', str(args.patience),
                   '--output', str(root / spec['label']), '--overrides', str(overrides), '--resume']
        command += ['--suite', spec['suite']] if spec['suite'] else ['--variants', *spec['variants']]
        for sweep in spec['sweep']:
            command += ['--sweep', sweep]
        if spec['test']:
            command.append('--test')
        if args.cpu:
            command.append('--cpu')
        if args.execute:
            command.append('--execute')
        commands.append(command)
    (root / (args.stage + '_commands.json')).write_text(json.dumps(commands, indent=2))
    for command in commands:
        subprocess.run(command, cwd=ROOT, check=True)


if __name__ == '__main__':
    main()
