#!/usr/bin/env python3
"""Extract table-1 facts from actual split/feature files, with file hashes."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from utils.configurator import Config


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--datasets', nargs='+', default=['baby', 'sports', 'clothing'])
    p.add_argument('--data-path', default='data')
    p.add_argument('--output', default='revision_outputs/datasets.json')
    args = p.parse_args()
    rows = []
    for dataset in args.datasets:
        cfg = Config('amorerevision', dataset, {'data_path': str(Path(args.data_path).resolve()), 'use_gpu': False})
        folder = Path(args.data_path) / dataset
        if (folder / 'SYNTHETIC_DATA.txt').exists():
            raise ValueError('Synthetic fixtures cannot populate the paper data table')
        paths = {k: folder / cfg[k] for k in ['inter_file_name', 'vision_feature_file', 'text_feature_file']}
        data = pd.read_csv(paths['inter_file_name'], sep=cfg['field_separator'])
        uid, iid, label = cfg['USER_ID_FIELD'], cfg['ITEM_ID_FIELD'], cfg['inter_splitting_label']
        row = {'dataset': dataset, 'users': int(data[uid].nunique()), 'observed_items': int(data[iid].nunique()),
               'train': int((data[label] == 0).sum()), 'validation': int((data[label] == 1).sum()),
               'test': int((data[label] == 2).sum()), 'sha256': {k: digest(v) for k, v in paths.items()}}
        for key in ['vision_feature_file', 'text_feature_file']:
            row[key + '_shape'] = list(np.load(paths[key], mmap_mode='r').shape)
        rows.append(row)
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(rows, indent=2))
    print(target)


if __name__ == '__main__':
    main()
