import argparse
import json
from pathlib import Path
import yaml
from utils.quick_start import quick_start


def main():
    parser = argparse.ArgumentParser(description='Train GUME or AMORE research variants')
    parser.add_argument('--model', '-m', default='gume')
    parser.add_argument('--dataset', '-d', default='baby')
    parser.add_argument('--config', action='append', default=[], help='YAML/JSON override; repeat to merge')
    parser.add_argument('--set', action='append', default=[], metavar='KEY=VALUE', help='YAML scalar or list override')
    parser.add_argument('--output-dir')
    parser.add_argument('--no-test', action='store_true', help='Validation-only development run')
    parser.add_argument('--no-save', action='store_true')
    args = parser.parse_args()
    config = {}
    for filename in args.config:
        loaded = yaml.safe_load(Path(filename).read_text())
        if not isinstance(loaded, dict):
            parser.error('Config must be a mapping: ' + filename)
        config.update(loaded)
    for entry in args.set:
        if '=' not in entry:
            parser.error('--set expects KEY=VALUE')
        key, value = entry.split('=', 1)
        config[key] = yaml.safe_load(value)
    if args.output_dir:
        config['output_dir'] = args.output_dir
    if args.no_test:
        config['evaluate_test'] = False
    quick_start(args.model, args.dataset, config, save_model=not args.no_save)


if __name__ == '__main__':
    main()
