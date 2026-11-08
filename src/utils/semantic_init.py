"""SG-URInit-inspired initialization in GUME's existing latent coordinates.

Reference: Jinfeng-Xu/SG-URInit, c85cb6d, preprocessing/preprocess_agg.py.
Use only TRAIN interactions, inverse item-degree weights, raw-feature KMeans,
and local/cluster interpolation. GUME adaptation: cluster representatives are
means of its initial latent item vectors; optional global RMS matching preserves
the original user-table scale. This is not an exact reproduction of SG-URInit.
"""
import hashlib
from logging import getLogger
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch


def cluster_labels(features, clusters, cache_dir):
    try:
        import sklearn
        from sklearn.cluster import KMeans
        from threadpoolctl import threadpool_limits
    except ImportError as exc:
        raise ImportError('SG initialization needs scikit-learn; on Python 3.8 use '
                          'pip install "scikit-learn>=1.1,<1.4"') from exc
    features = np.ascontiguousarray(features, dtype=np.float32)
    digest = hashlib.sha256()
    digest.update(memoryview(features).cast('B'))
    digest.update(repr((features.shape, clusters, 0, 10, 300, sklearn.__version__)).encode())
    cache = Path(cache_dir) / ('sg_clusters_' + digest.hexdigest() + '.npz')
    if cache.exists():
        with np.load(cache, allow_pickle=False) as saved:
            labels = saved['labels']
        if labels.shape == (len(features),) and (labels >= 0).all() and (labels < clusters).all():
            getLogger().info('SG initialization: reuse KMeans labels %s', cache.name)
            return labels
    cache.parent.mkdir(parents=True, exist_ok=True)
    # Fixed preprocessing seed; model seed remains an independent experimental factor.
    getLogger().info('SG initialization: KMeans shape=%s, clusters=%d, n_init=10; first run may take time',
                     features.shape, clusters)
    with threadpool_limits(limits=1):
        labels = KMeans(n_clusters=clusters, random_state=0, n_init=10,
                        max_iter=300, algorithm='lloyd').fit_predict(features)
    np.savez_compressed(cache, labels=labels)
    return labels


def aggregate_semantic_users(interactions, items, labels=None, cluster_weight=0.5):
    """Pure sparse computation; interactions must already be the training split."""
    interactions = interactions.tocsr().astype(np.float32).copy()
    interactions.data[:] = 1.
    degree = np.asarray(interactions.sum(axis=0)).ravel()
    weights = interactions @ sp.diags(1 / np.maximum(degree, 1))
    local = weights @ items
    if labels is None or cluster_weight == 0:
        return np.asarray(local)
    clusters = int(labels.max()) + 1
    centers = np.zeros((clusters, items.shape[1]), dtype=np.float32)
    np.add.at(centers, labels, items)
    centers /= np.maximum(np.bincount(labels, minlength=clusters), 1)[:, None]
    return np.asarray((1 - cluster_weight) * local + cluster_weight * (weights @ centers[labels]))


@torch.no_grad()
def initialize_modality_user(table, latent_items, raw_features, interactions, config, cache_dir):
    mode = config['sg_init']
    if mode not in ('local', 'cluster', 'shuffle'):
        raise ValueError('sg_init must be off, local, cluster, or shuffle')
    mixing = float(config['sg_cluster_weight'])
    clusters = config['sg_clusters']
    if not 0 <= mixing <= 1 or int(clusters) != clusters or clusters < 1:
        raise ValueError('sg_cluster_weight must be in [0,1]; sg_clusters must be a positive integer')
    if config['sg_scale'] not in ('match', 'raw'):
        raise ValueError('sg_scale must be match or raw')
    raw = raw_features.detach().cpu().numpy()
    labels = None if mode == 'local' or mixing == 0 else cluster_labels(raw, min(int(clusters), len(raw)), cache_dir)
    initialized = aggregate_semantic_users(interactions, latent_items.detach().cpu().numpy(), labels, mixing)
    initialized = torch.as_tensor(initialized, dtype=table.dtype, device=table.device)
    active = torch.as_tensor(np.asarray(interactions.sum(axis=1)).ravel() > 0, device=table.device)
    if active.any():
        if config['sg_scale'] == 'match':
            scale = table[active].square().mean().sqrt() / initialized[active].square().mean().sqrt().clamp_min(1e-8)
            initialized = initialized * scale
        if mode == 'shuffle':
            generator = torch.Generator().manual_seed(int(config['seed']) + 29173)
            permutation = torch.randperm(int(active.sum()), generator=generator).to(table.device)
            initialized[active] = initialized[active][permutation]
        table[active] = initialized[active]
    return {'initialized_users': int(active.sum()), 'clusters': int(labels.max() + 1) if labels is not None else 0}
