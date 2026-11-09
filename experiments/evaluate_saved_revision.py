#!/usr/bin/env python3
"""Evaluate frozen reviewer checkpoints without retraining or changing weights."""
import argparse
import copy
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import torch
from run_suite import source_fingerprint
from utils.configurator import Config
from utils.dataset import RecDataset
from utils.dataloader import TrainDataLoader, EvalDataLoader
from utils.quick_start import file_hash
from utils.revision_protocol import perturb_training
from utils.checkpoint import load_checkpoint
from utils.utils import init_seed, get_model
from common.trainer import Trainer
from utils.benchmark import benchmark_ranking


def evaluate(path, destination, data_path, gpu=0, cpu=False, benchmark=False, allow_synthetic=False):
    original = json.loads(path.read_text())
    if original.get('synthetic_fixture') and not allow_synthetic:
        raise ValueError('Synthetic evidence refused')
    if original['source_sha256'] != source_fingerprint():
        raise ValueError('Training source changed; cannot restore evidence with different code')
    trial = original['trials'][original['selected_trial']]
    cfg = dict(trial['config'])
    if cfg.get('train_fraction', 1.) != 1.:
        raise ValueError('This evaluator supports the full reviewer training protocol')
    for key, digest in original['data_sha256'].items():
        if file_hash(Path(data_path) / original['dataset'] / cfg[key]) != digest:
            raise ValueError('Data bytes changed: ' + key)
    provenance = file_hash(path)
    if (destination / 'result.json').exists():
        saved = json.loads((destination / 'result.json').read_text())
        meta = saved.get('checkpoint_evaluation', {})
        artifacts_ok = (destination / ('trial_{:03d}'.format(original['selected_trial'])) / 'test_rankings.npz').exists()
        if (meta.get('input_sha256') == provenance and meta.get('benchmark') == benchmark
                and meta.get('cpu') == cpu and meta.get('gpu') == gpu and artifacts_ok):
            print('Reuse checkpoint evaluation:', destination)
            return
        raise ValueError('Existing evaluation differs/incomplete; use a new evaluation output directory')
    checkpoint = path.parent / ('trial_{:03d}'.format(original['selected_trial'])) / 'best.pt'
    if not checkpoint.exists():
        checkpoint = Path(trial['checkpoint'])
    if not checkpoint.exists():
        raise FileNotFoundError('Keep best.pt on the server: ' + str(checkpoint))
    trial_dir = destination / ('trial_{:03d}'.format(original['selected_trial']))
    cfg.update(data_path=str(Path(data_path).resolve()), gpu_id=gpu, use_gpu=not cpu,
               export_rankings=True, save_recommended_topk=False, tensorboard_log_dir=str(trial_dir))
    config = Config(original['model'], original['dataset'], cfg)
    init_seed(config['seed'])
    clean, valid, test = RecDataset(config).split()
    train, intervention = perturb_training(clean, valid, test, config)
    if intervention['train_pairs_sha256'] != trial['train_pairs_sha256']:
        raise ValueError('Restored training edges differ from the recorded checkpoint')
    train_loader = TrainDataLoader(config, train, batch_size=config['train_batch_size'], shuffle=True)
    train_loader.pretrain_setup()
    test_loader = EvalDataLoader(config, test, additional_dataset=clean, batch_size=config['eval_batch_size'])
    model = get_model(original['model'])(config, train_loader, None).to(config['device'])
    saved = load_checkpoint(checkpoint, map_location=config['device'])
    if saved['epoch'] != trial['best_epoch'] or saved['validation'] != trial['validation']:
        raise ValueError('Checkpoint does not match the recorded validation selection')
    model.load_state_dict(saved['state_dict'])
    trainer = Trainer(config, model)
    result = copy.deepcopy(original)
    selected = result['trials'][result['selected_trial']]
    try:
        result['test'] = trainer.evaluate(test_loader, is_test=True)
        selected['test'] = result['test']
        if benchmark:
            selected['inference_benchmark'] = benchmark_ranking(model, test_loader, config['device'], repeats=5)
    finally:
        trainer.writer.close()
    result['checkpoint_evaluation'] = dict(input_path=str(path.resolve()), input_sha256=provenance,
                                         checkpoint=str(checkpoint.resolve()), benchmark=benchmark,
                                         cpu=cpu, gpu=gpu, torch=torch.__version__,
                                         device_name=torch.cuda.get_device_name(config['device']) if config['device'].type == 'cuda' else 'CPU',
                                         training_measurements='Copied from original training; not remeasured',
                                         selection='Original validation-selected checkpoint; no optimization')
    history = path.parent / ('trial_{:03d}'.format(original['selected_trial'])) / 'history.json'
    if history.exists():
        shutil.copy2(history, trial_dir / 'history.json')
    destination.mkdir(parents=True, exist_ok=True)
    temporary = destination / 'result.json.tmp'
    temporary.write_text(json.dumps(result, indent=2, default=str))
    temporary.replace(destination / 'result.json')
    print('Evaluated:', destination)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('root'); p.add_argument('--output', required=True)
    p.add_argument('--data-path', default='data'); p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--cpu', action='store_true'); p.add_argument('--benchmark', action='store_true')
    args = p.parse_args()
    root, output = Path(args.root), Path(args.output)
    if root.resolve() == output.resolve() or root.resolve() in output.resolve().parents:
        p.error('Evaluation output must be outside the input tree')
    paths = sorted(root.rglob('result.json'))
    if not paths:
        p.error('No result.json files')
    for path in paths:
        evaluate(path, output / path.parent.relative_to(root), args.data_path, args.gpu, args.cpu, args.benchmark)


if __name__ == '__main__':
    main()
