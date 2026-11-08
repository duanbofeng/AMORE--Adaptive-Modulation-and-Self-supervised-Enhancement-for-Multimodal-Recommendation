#!/usr/bin/env python3
"""Download pinned MMRec references, including their upstream license; verify bytes."""
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]


def main():
    manifest = json.loads((ROOT / 'experiments/baseline_sources.json').read_text())
    target = ROOT / 'external_baselines/MMRec'
    target.mkdir(parents=True, exist_ok=True)
    for name, spec in manifest['files'].items():
        path = target / name
        if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() == spec['sha256']:
            continue
        url = spec.get('url') or 'https://raw.githubusercontent.com/enoche/MMRec/{}/{}'.format(manifest['commit'], spec['path'])
        data = urlopen(url, timeout=60).read()
        if hashlib.sha256(data).hexdigest() != spec['sha256']:
            raise ValueError('Upstream source hash mismatch: ' + name)
        path.write_bytes(data)
    print('Verified pinned MMRec sources:', target)


if __name__ == '__main__':
    main()
