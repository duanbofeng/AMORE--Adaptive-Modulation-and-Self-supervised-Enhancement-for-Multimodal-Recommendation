#!/usr/bin/env python3
"""Frozen-checkpoint mechanism probes. No optimization or test-set selection."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import numpy as np
import torch
from scipy.stats import spearmanr
from run_suite import source_fingerprint
from utils.configurator import Config
from utils.dataset import RecDataset
from utils.dataloader import TrainDataLoader
from utils.checkpoint import load_checkpoint
from utils.utils import get_model, init_seed


def capture(model):
    """Capture the actual forward rather than maintain a second implementation."""
    seen, hooks = {}, []
    def save(name):
        def hook(module, inputs, output):
            seen.setdefault(name, []).append((inputs[0].detach(), output.detach()))
        return hook
    for name in ['separate_coarse', 'affinity_guided_coarse', 'image_behavior', 'text_behavior']:
        hooks.append(getattr(model, name).register_forward_hook(save(name)))
    try:
        with torch.no_grad():
            embedding = model.forward()
    finally:
        for hook in hooks:
            hook.remove()
    return embedding, seen


def mechanism(model):
    embedding, seen = capture(model)
    q = seen['affinity_guided_coarse'][0][0][:, -1]
    users, items = model.ui_edges
    degree = torch.bincount(users, minlength=model.n_users)
    uniform = torch.zeros_like(q).index_add_(0, items, 1. / degree[users].float())
    uniform = uniform / model.item_degree.clamp_min(1)
    active = model.item_degree > 0
    gi = seen['image_behavior'][0][1]
    gt = seen['text_behavior'][0][1]
    attenuation = (1 - gi - gt) / 3
    mix = (3 * q).sigmoid() * model.affinity_coarse_alpha.detach().clamp(0, 1)
    correlation = (float(spearmanr(q[active].cpu(), uniform[active].cpu())[0])
                   if q[active].std(unbiased=False) > 0 and uniform[active].std(unbiased=False) > 0 else None)
    report = dict(prior_std=float(q.std(unbiased=False)),
                  prior_degree_baseline_spearman=correlation,
                  prior_degree_baseline_rmse=float((q[active] - uniform[active]).square().mean().sqrt()),
                  alpha_raw=float(model.affinity_coarse_alpha.detach()), mixing_mean=float(mix.mean()),
                  coarse_output_coefficient_abs_mean=float(attenuation.abs().mean()),
                  coarse_output_coefficient_abs_p90=float(torch.quantile(attenuation.abs(), .9)),
                  selected_edge_fraction=model.diagnostics['last_modulation_selected_fraction'])
    return embedding, seen, report


def counterfactuals(model, users, histories, topk=20):
    """Sample users fixed independently of outcomes; full catalog, train masking."""
    model.eval()
    baseline, seen, report = mechanism(model)
    original = model.prior_control, model.graph_affinity, model.use_coarse_affinity
    def rank(output, ids):
        scores = output[ids] @ output[model.n_users:].T
        for i, u in enumerate(ids.tolist()):
            scores[i, sorted(histories.get(u, []))] = -torch.inf
        return scores, scores.topk(topk, dim=1).indices
    comparisons = []
    try:
        for name, prior, graph, fusion in [
            ('repeat_forward', 'shared', True, True),
            ('constant_prior', 'constant', True, True),
            ('shuffle_prior', 'shuffle', True, True),
            ('fusion_off', 'shared', True, False),
            ('graph_off', 'shared', False, True),
            ('both_off', 'shared', False, False),
        ]:
            model.prior_control, model.graph_affinity, model.use_coarse_affinity = prior, graph, fusion
            with torch.no_grad():
                changed = model.forward()
                overlap, error, norm, count = 0., 0., 0., 0
                for ids in users.split(128):
                    a, ka = rank(baseline, ids); b, kb = rank(changed, ids)
                    finite = torch.isfinite(a) & torch.isfinite(b)
                    error += float((a[finite] - b[finite]).square().sum())
                    norm += float(a[finite].square().sum())
                    overlap += float((ka[:, :, None] == kb[:, None, :]).any(2).float().mean(1).sum())
                    count += len(ids)
                item = dict(intervention=name, topk_overlap=overlap / count,
                            relative_score_l2=(error / max(norm, 1e-30)) ** .5,
                            relative_embedding_l2=float((changed-baseline).norm()/baseline.norm().clamp_min(1e-15)))
                comparisons.append(item)
    finally:
        model.prior_control, model.graph_affinity, model.use_coarse_affinity = original
    report['counterfactuals'] = comparisons
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', default='results/revision/affinity')
    p.add_argument('--data-path', default='data')
    p.add_argument('--dataset', default='baby', choices=['baby', 'sports', 'clothing'])
    p.add_argument('--output', default='revision_outputs/affinity_probe')
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--cpu', action='store_true')
    p.add_argument('--max-users', type=int, default=1024)
    p.add_argument('--seeds', type=int, nargs='+', default=[999, 2024, 2025, 2026, 2027])
    args = p.parse_args()
    if args.max_users < 1:
        p.error('--max-users must be positive')
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    reports = []
    for path in sorted(Path(args.root).rglob('result.json')):
        r = json.loads(path.read_text()); t = r['trials'][r['selected_trial']]
        cfg = dict(t['config'])
        if r['dataset'] != args.dataset or cfg.get('variant') != 'rev_11' or cfg['seed'] not in args.seeds:
            continue
        if r.get('synthetic_fixture') or r['source_sha256'] != source_fingerprint():
            raise ValueError('Requires real data and unchanged training source: ' + str(path))
        if cfg.get('train_fraction', 1) != 1 or any(cfg.get(k, 0) for k in
                ['interaction_noise_rate', 'feature_corruption_rate', 'cold_item_fraction']):
            raise ValueError('This probe expects the clean affinity cohort')
        if not cfg['revision_graph'] or not cfg['revision_fusion'] or cfg['revision_prior'] != 'shared':
            raise ValueError('Expected both branches with shared prior')
        for key, expected in r['data_sha256'].items():
            digest = hashlib.sha256()
            with (Path(args.data_path) / r['dataset'] / cfg[key]).open('rb') as f:
                for chunk in iter(lambda: f.read(1024*1024), b''):
                    digest.update(chunk)
            if digest.hexdigest() != expected:
                raise ValueError('Data hash mismatch: ' + key)
        checkpoint = path.parent / ('trial_{:03d}'.format(r['selected_trial'])) / 'best.pt'
        if not checkpoint.exists():
            checkpoint = Path(t['checkpoint'])
        cfg.update(data_path=str(Path(args.data_path).resolve()), use_gpu=not args.cpu, gpu_id=args.gpu)
        config = Config(r['model'], r['dataset'], cfg)
        train, valid, _ = RecDataset(config).split()
        pairs = train.df[[train.uid_field, train.iid_field]].sort_values([train.uid_field, train.iid_field]).to_numpy(dtype=np.int64)
        if hashlib.sha256(pairs.tobytes()).hexdigest() != t['train_pairs_sha256']:
            raise ValueError('Training graph mismatch')
        init_seed(cfg['seed'])
        loader = TrainDataLoader(config, train, batch_size=config['train_batch_size'])
        loader.pretrain_setup()
        model = get_model(r['model'])(config, loader, None).to(config['device'])
        saved = load_checkpoint(checkpoint, map_location=config['device'])
        if saved['epoch'] != t['best_epoch']:
            raise ValueError('Checkpoint epoch does not match selected result')
        model.load_state_dict(saved['state_dict'])
        users = np.sort(valid.df[valid.uid_field].unique())
        users = np.random.default_rng(314159).permutation(users)[:args.max_users]
        if len(users) == 0 or model.n_items < 20:
            raise ValueError('Need validation users and at least 20 catalog items')
        with torch.no_grad():
            report = counterfactuals(model, torch.tensor(users, device=config['device']), loader.history_items_per_u)
        report.update(dataset=r['dataset'], seed=cfg['seed'], result_path=str(path),
                      checkpoint=str(checkpoint), source=r['source_sha256'], selected_epoch=t['best_epoch'],
                      sampled_validation_users=len(users), user_sample_seed=314159,
                      sampled_user_ids_sha256=hashlib.sha256(users.astype(np.int64).tobytes()).hexdigest(),
                      note='Frozen-checkpoint intervention; top-20 overlap is sensitivity, not recommendation accuracy.')
        reports.append(report)
        del model, saved
        print('Probed', r['dataset'], cfg['seed'])
    if sorted(r['seed'] for r in reports) != sorted(set(args.seeds)):
        raise ValueError('Missing or duplicate requested rev_11 seeds')
    (output / 'probe.json').write_text(json.dumps(reports, indent=2, allow_nan=False))
    plot_reports(reports, output)


def plot_reports(reports, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from plot_revision import save
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    names = [x['intervention'] for x in reports[0]['counterfactuals']]
    for report in reports:
        for ax, metric in zip(axes, ['topk_overlap', 'relative_score_l2']):
            ax.plot(names, [r[metric] for r in report['counterfactuals']], marker='o', label=str(report['seed']))
            ax.set_ylabel(metric); ax.tick_params(axis='x', rotation=40)
    axes[0].set_ylim(0, 1.02); axes[1].set_yscale('symlog', linthresh=1e-7)
    axes[1].legend(title='Seed', fontsize=8)
    fig.suptitle('Frozen checkpoints: validation-user sample; no retraining')
    save(fig, output, 'counterfactual_sensitivity')


if __name__ == '__main__':
    main()
