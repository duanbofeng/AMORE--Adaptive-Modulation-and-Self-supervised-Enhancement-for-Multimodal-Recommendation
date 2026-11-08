"""Reproducible training, validation selection and machine-readable results."""
import hashlib
import json
import os
import platform
import subprocess
import uuid
from itertools import product
from logging import getLogger
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from utils.dataset import RecDataset
from utils.dataloader import TrainDataLoader, EvalDataLoader
from utils.logger import init_logger
from utils.configurator import Config
from utils.checkpoint import load_checkpoint
from utils.utils import init_seed, get_model, get_trainer
from utils.revision_protocol import perturb_training


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def quick_start(model, dataset, config_dict=None, save_model=True):
    run_started = perf_counter()
    config = Config(model, dataset, dict(config_dict or {}))
    local_time = init_logger(config)
    logger = getLogger()
    base = dict(config.final_config_dict)
    keys = config['hyper_parameters']
    choices = [config[key] if isinstance(config[key], list) else [config[key]] for key in keys]
    combinations = list(product(*choices))
    if len(combinations) > 1 and config['train_fraction'] != 1:
        raise ValueError('Use separate runs for train_fraction experiments')
    if not combinations:
        raise ValueError('Empty hyperparameter grid')
    root = Path(config['output_dir'] or (Path(config.project_root) / 'results' / (model + '-' + dataset + '-' + local_time + '-' + uuid.uuid4().hex[:6])))
    root.mkdir(parents=True, exist_ok=True)
    dataset_object = RecDataset(config)
    train_original, valid_dataset, test_dataset = dataset_object.split()
    if any(len(d) == 0 for d in [train_original, valid_dataset, test_dataset]):
        raise ValueError('Train, validation and test splits must all be nonempty')
    # Reject duplicate pairs across splits: otherwise masking makes results invalid.
    pairs = [set(zip(d.df[d.uid_field], d.df[d.iid_field])) for d in [train_original, valid_dataset, test_dataset]]
    if pairs[0] & pairs[1] or pairs[0] & pairs[2] or pairs[1] & pairs[2]:
        raise ValueError('User-item pairs overlap across train/validation/test splits')
    source_dir = Path(config['data_path']) / dataset
    source_hash = hashlib.sha256()
    for path in sorted(list((config.project_root / 'src').rglob('*.py')) + list((config.project_root / 'src/configs').rglob('*.yaml')) + [config.project_root / 'experiments/baseline_sources.json']):
        source_hash.update(str(path.relative_to(config.project_root)).encode())
        source_hash.update(path.read_bytes())
    hashes = {name: file_hash(source_dir / config[name]) for name in
              ['inter_file_name', 'vision_feature_file', 'text_feature_file'] if (source_dir / config[name]).exists()}
    try:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=config.project_root, text=True).strip()
        dirty = bool(subprocess.check_output(['git', 'status', '--porcelain'], cwd=config.project_root, text=True).strip())
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, None
    trials = []
    best_index, best_score = None, -float('inf') if config['valid_metric_bigger'] else float('inf')
    for index, values in enumerate(combinations):
        config.final_config_dict = dict(base)
        config.final_config_dict.update(dict(zip(keys, values)))
        init_seed(config['seed'])
        train_dataset = train_original
        fraction = config['train_fraction']
        if not 0 < fraction <= 1:
            raise ValueError('train_fraction must lie in (0,1]')
        if fraction < 1:
            # Retain at least one interaction per observed user; keep validation/test fixed.
            sampled = train_original.df.groupby(train_original.uid_field, group_keys=False).sample(frac=fraction, random_state=config['seed'])
            anchors = train_original.df.groupby(train_original.uid_field, group_keys=False).sample(n=1, random_state=config['seed'])
            sampled = sampled.combine_first(anchors).sort_index()
            sampled = sampled.astype(train_original.df.dtypes.to_dict())
            train_dataset = train_original.copy(sampled)
        train_dataset, intervention = perturb_training(train_dataset, valid_dataset, test_dataset, config)
        logger.info('Training dataset: %s', train_dataset)
        logger.info('Validation dataset: %s', valid_dataset)
        logger.info('Test dataset: %s', test_dataset)
        train_data = TrainDataLoader(config, train_dataset, batch_size=config['train_batch_size'], shuffle=True)
        # Mask all known original training interactions even under training thinning.
        valid_data = EvalDataLoader(config, valid_dataset, additional_dataset=train_original, batch_size=config['eval_batch_size'])
        test_data = EvalDataLoader(config, test_dataset, additional_dataset=train_original, batch_size=config['eval_batch_size'])
        trial_dir = root / ('trial_{:03d}'.format(index))
        config['tensorboard_log_dir'] = str(trial_dir)
        logger.info('Configuration: %s', config)
        train_data.pretrain_setup()
        start = perf_counter()
        if config['device'].type == 'cuda':
            torch.cuda.reset_peak_memory_stats(config['device'])
        instance = get_model(model)(config, train_data, local_time).to(config['device'])
        if config['device'].type == 'cuda':
            torch.cuda.synchronize(config['device'])
        setup_seconds = perf_counter() - start
        trainer = get_trainer()(config, instance)
        score, valid, _ = trainer.fit(train_data, valid_data, test_data=None, saved=True)
        train_pairs = train_dataset.df[[train_dataset.uid_field, train_dataset.iid_field]].sort_values(
            [train_dataset.uid_field, train_dataset.iid_field]).to_numpy(dtype=np.int64)
        trial = {
            'trial': index, 'config': dict(config.final_config_dict), 'validation': valid,
            'best_epoch': trainer.best_epoch, 'epochs_run': len(trainer.history),
            'parameters_total': sum(p.numel() for p in instance.parameters()),
            'parameters_trainable': sum(p.numel() for p in instance.parameters() if p.requires_grad),
            'setup_seconds': setup_seconds, 'training_seconds': sum(trainer.epoch_seconds),
            'mean_epoch_seconds': float(np.mean(trainer.epoch_seconds)),
            'peak_cuda_bytes': torch.cuda.max_memory_allocated(config['device']) if config['device'].type == 'cuda' else None,
            'diagnostics': getattr(instance, 'diagnostics', {}), 'checkpoint': str(trainer.checkpoint),
            'train_interactions': len(train_dataset),
            'actual_train_fraction': len(train_dataset) / len(train_original),
            'train_pairs_sha256': hashlib.sha256(train_pairs.tobytes()).hexdigest(),
            'intervention': intervention,
            'original_train_user_degree': train_original.df.groupby(train_original.uid_field).size().to_dict(),
            'original_train_item_degree': train_original.df.groupby(train_original.iid_field).size().to_dict(),
        }
        trials.append(trial)
        improved = score > best_score if config['valid_metric_bigger'] else score < best_score
        if improved:
            best_score, best_index = score, index
        del trainer, instance
        if config['device'].type == 'cuda':
            torch.cuda.empty_cache()
    selected = trials[best_index]
    test_result = {}
    if base['evaluate_test']:
        config.final_config_dict = dict(selected['config'])
        init_seed(config['seed'])
        # All trials use the same training pairs; thinning grids were rejected above.
        instance = get_model(model)(config, train_data, local_time).to(config['device'])
        instance.load_state_dict(load_checkpoint(selected['checkpoint'], map_location=config['device'])['state_dict'])
        trainer = get_trainer()(config, instance)
        try:
            if config['device'].type == 'cuda':
                torch.cuda.synchronize(config['device'])
            start = perf_counter()
            test_result = trainer.evaluate(test_data, is_test=True)
            if config['device'].type == 'cuda':
                torch.cuda.synchronize(config['device'])
            selected['test_evaluation_seconds'] = perf_counter() - start
            selected['test_users'] = len(test_data.get_eval_users())
            if config['benchmark_repeats']:
                from utils.benchmark import benchmark_ranking
                selected['inference_benchmark'] = benchmark_ranking(
                    instance, test_data, config['device'], int(config['benchmark_repeats']))
        finally:
            trainer.writer.close()
        selected['test'] = test_result
    result = {'schema_version': 1, 'model': model, 'dataset': dataset, 'selected_trial': best_index,
              'selection_metric': config['valid_metric'].lower(), 'validation': selected['validation'],
              'test': test_result, 'trials': trials, 'data_sha256': hashes,
              'git_commit': commit, 'git_dirty': dirty, 'python': platform.python_version(),
              'source_sha256': source_hash.hexdigest(), 'run_wall_seconds': perf_counter() - run_started,
              'synthetic_fixture': (source_dir / 'SYNTHETIC_DATA.txt').exists(),
              'gpu_name': torch.cuda.get_device_name(config['device']) if config['device'].type == 'cuda' else None,
              'torch': torch.__version__, 'numpy': np.__version__, 'host': platform.node(),
              'protocol': 'validation selection; train-only masking; full-item ranking; test after selection'}
    if not save_model:
        for trial in trials:
            Path(trial['checkpoint']).unlink(missing_ok=True)
    target = root / 'result.json'
    temp = root / 'result.json.tmp'
    temp.write_text(json.dumps(result, indent=2, default=str))
    os.replace(temp, target)
    logger.info('Selected by validation: %s; test: %s; result: %s', selected['validation'], test_result, target)
    return result
