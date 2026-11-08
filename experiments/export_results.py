#!/usr/bin/env python3
"""Export small, Git-trackable result/config/history snapshots; no torch dependency."""
import argparse
import json
from pathlib import Path


def export_results(root, output):
    root, output = Path(root), Path(output)
    records = []
    for path in sorted(root.rglob('result.json')):
        result = json.loads(path.read_text())
        histories = {}
        for trial in result['trials']:
            # Use the run-local path so exports also work after moving a result tree.
            history = path.parent / 'trial_{:03d}'.format(trial['trial']) / 'history.json'
            if history.exists():
                histories[str(trial['trial'])] = json.loads(history.read_text())
        records.append({'path': str(path.relative_to(root)), 'result': result, 'histories': histories})
    if not records:
        raise ValueError('No completed result.json found under ' + str(root))
    if output.resolve() == root.resolve() or output.name == 'result.json':
        raise ValueError('Choose a separate snapshot filename, not result.json')
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_name(output.name + '.tmp')
    temp.write_text(json.dumps({'snapshot_schema': 1, 'runs': records}, indent=2, ensure_ascii=False))
    temp.replace(output)
    return records


def describe(records):
    print('| Variant | Seed | Best epoch | Epochs | Trainable params | R@20 | N@20 |')
    print('|---|---:|---:|---:|---:|---:|---:|')
    for record in records:
        result = record['result']
        trial = result['trials'][result['selected_trial']]
        cfg = trial['config']
        print('| {} | {} | {} | {} | {} | {} | {} |'.format(
            cfg.get('variant', result['model']), cfg['seed'], trial['best_epoch'], trial['epochs_run'],
            trial['parameters_trainable'], result['validation'].get('recall@20', 'NA'),
            result['validation'].get('ndcg@20', 'NA')))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', help='Directory containing completed runs')
    parser.add_argument('--output', required=True, help='e.g. experiment_reports/01_core_baby.json')
    args = parser.parse_args()
    records = export_results(args.root, args.output)
    describe(records)
    print('Exported {} runs: {} ({} bytes)'.format(len(records), args.output, Path(args.output).stat().st_size))


if __name__ == '__main__':
    main()
