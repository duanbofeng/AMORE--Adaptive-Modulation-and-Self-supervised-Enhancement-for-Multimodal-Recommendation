"""Sparse graph preprocessing. All behavioral inputs must contain training edges only."""
import hashlib
import os
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F


def as_torch(matrix, device):
    matrix = matrix.tocoo().astype(np.float32)
    return torch.sparse_coo_tensor(
        np.vstack([matrix.row, matrix.col]), matrix.data, matrix.shape,
        device=device, dtype=torch.float32).coalesce()


def as_scipy(matrix):
    matrix = matrix.detach().coalesce().cpu()
    return sp.csr_matrix((matrix.values().numpy(), matrix.indices().numpy()), shape=matrix.shape)


def normalize(matrix, symmetric=True):
    degree = np.asarray(matrix.sum(axis=1)).ravel()
    inv = np.zeros_like(degree, dtype=np.float32)
    np.power(degree, -0.5 if symmetric else -1, out=inv, where=degree > 0)
    result = sp.diags(inv) @ matrix
    return (result @ sp.diags(inv) if symmetric else result).tocsr()


def topk_sparse(matrix, k):
    matrix = matrix.tocsr(copy=True)
    for row in range(matrix.shape[0]):
        start, end = matrix.indptr[row:row + 2]
        if end - start > k:
            drop = np.argpartition(matrix.data[start:end], -k)[:-k]
            matrix.data[start + drop] = 0
    matrix.eliminate_zeros()
    return matrix


@torch.no_grad()
def knn_graph(features, k, chunk_size=512, self_loops=False, cache_dir=None):
    """Exact cosine KNN with O(chunk_size * items) peak similarity storage.

    Cache key includes feature contents (including corruption), k and self-loop policy.
    Runtime is still quadratic; chunking bounds memory, not arithmetic.
    """
    if k < 1 or chunk_size < 1:
        raise ValueError('knn_k and graph_chunk_size must be positive')
    digest = hashlib.sha256(features.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
    cache = None
    if cache_dir:
        cache = Path(cache_dir) / ('knn_v1_{}_{}_{}.npz'.format(digest, k, int(self_loops)))
        if cache.exists():
            return sp.load_npz(cache)
    n = len(features)
    count = min(k, n if self_loops else n - 1)
    if count <= 0:
        return sp.csr_matrix((n, n), dtype=np.float32)
    normalized = F.normalize(features.detach(), dim=1)
    rows, cols, vals = [], [], []
    for start in range(0, n, chunk_size):
        stop = min(start + chunk_size, n)
        scores = normalized[start:stop] @ normalized.T
        if not self_loops:
            scores[torch.arange(stop - start, device=scores.device),
                   torch.arange(start, stop, device=scores.device)] = -torch.inf
        value, index = scores.topk(count, dim=1)
        rows.append(np.repeat(np.arange(start, stop), count))
        cols.append(index.cpu().numpy().ravel())
        vals.append(value.clamp_min(0).cpu().numpy().ravel())
    graph = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n))
    graph.eliminate_zeros()
    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        temp = cache.with_name(cache.stem + '.{}.npz'.format(os.getpid()))
        sp.save_npz(temp, graph)
        os.replace(temp, cache)
    return graph


def shared_graph(graphs, interactions, k, cooc_weight=0., consistency=0., chunk_size=512):
    """IIMRec-inspired spatial/co-occurrence fusion and NCER on bounded neighborhoods.

    No spectral channels, virtual UI edges or neighbor pseudo-positive losses.
    """
    graph = sum(normalize(g, symmetric=False) for g in graphs) / len(graphs)
    if cooc_weight:
        blocks = []
        for start in range(0, interactions.shape[1], chunk_size):
            block = (interactions[:, start:start + chunk_size].T @ interactions).tocsr()
            row = np.arange(block.shape[0])
            block[row, row + start] = 0
            block.eliminate_zeros()
            blocks.append(topk_sparse(block, k))
        cooc = normalize(sp.vstack(blocks).tocsr(), symmetric=False)
        graph = (1 - cooc_weight) * graph + cooc_weight * cooc
    graph = topk_sparse(graph, k)
    if consistency:
        neighbors = [set(graph.indices[graph.indptr[i]:graph.indptr[i + 1]]) for i in range(graph.shape[0])]
        for i in range(graph.shape[0]):
            for offset in range(graph.indptr[i], graph.indptr[i + 1]):
                j = graph.indices[offset]
                graph.data[offset] *= 1 + consistency * len(neighbors[i] & neighbors[j]) / k
    return normalize(graph, symmetric=False)


def ui_graph(interactions, item_graph=None):
    adjacency = sp.bmat([[None, interactions], [interactions.T, item_graph]], format='csr')
    return normalize(adjacency)


@torch.no_grad()
def edge_affinity(features, interactions, mode, strength, shrinkage, temperature, seed, chunk_size=2048):
    """Leave-one-out content agreement on observed training edges only.

    Singleton histories get weight 1. Item/global/shuffle are explicit controls.
    Weights remain in [1-strength, 1], preserving nonzero semantic messages.
    """
    interactions = interactions.tocsr(copy=True)
    interactions.sort_indices()
    coo = interactions.tocoo()
    # One-time CPU preprocessing avoids allocating users x raw_feature_dim on the GPU.
    device = torch.device('cpu')
    rows = torch.as_tensor(coo.row, device=device, dtype=torch.long)
    cols = torch.as_tensor(coo.col, device=device, dtype=torch.long)
    feat = F.normalize(features.detach().cpu(), dim=1)
    sums = torch.sparse.mm(as_torch(interactions, device), feat)
    degree = torch.as_tensor(np.asarray(interactions.sum(1)).ravel(), device=device)
    agreement = []
    for start in range(0, len(rows), chunk_size):
        u, i = rows[start:start + chunk_size], cols[start:start + chunk_size]
        history = sums[u] - feat[i]
        score = (F.normalize(history, dim=1) * feat[i]).sum(1)
        reliability = (degree[u] - 1).clamp_min(0) / ((degree[u] - 1).clamp_min(0) + shrinkage)
        reliability = reliability * (history.norm(dim=1) > 1e-8)
        agreement.append(1 - reliability * (1 - torch.sigmoid(score / temperature)))
    values = torch.cat(agreement).cpu().numpy()
    if mode == 'item':
        count = np.bincount(coo.col, minlength=interactions.shape[1])
        means = np.bincount(coo.col, weights=values, minlength=interactions.shape[1]) / np.maximum(count, 1)
        values = means[coo.col]
    elif mode == 'global':
        values[:] = values.mean()
    elif mode == 'shuffle':
        # Within-user shuffle keeps each user's weight distribution unchanged.
        rng = np.random.default_rng(seed)
        csr = interactions.tocsr()
        for a, b in zip(csr.indptr[:-1], csr.indptr[1:]):
            values[a:b] = rng.permutation(values[a:b])
    elif mode != 'local':
        raise ValueError('Unknown affinity mode: ' + mode)
    weights = 1 - strength + strength * values
    return sp.csr_matrix((weights, (coo.row, coo.col)), shape=interactions.shape)
