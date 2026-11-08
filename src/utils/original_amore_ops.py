"""Original AMORE graph semantics and numerically stable loss primitives."""
import hashlib
import os
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
from torch.nn import functional as F

from utils.graph_ops import as_torch, as_scipy


@torch.no_grad()
def original_knn(features, k, chunk_size, cache_dir=None):
    """Exact signed cosine top-k INCLUDING self, as in old_model.

    Unlike the newer GUME helper, negative cosines are NOT clipped. Keep selected
    zero-weight neighbors as structural entries for the cross-modality intersection.
    Legacy cache files are not reused because their inputs were not fingerprinted.
    """
    if not 1 <= k <= len(features) or chunk_size < 1:
        raise ValueError('Original KNN requires 1 <= k <= item count and positive chunk size')
    norm = features.norm(dim=1, keepdim=True)
    if not torch.isfinite(features).all() or (norm == 0).any():
        raise ValueError('Original cosine graph is undefined for zero/nonfinite feature rows')
    path = None
    if cache_dir:
        digest = hashlib.sha256(features.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
        path = Path(cache_dir) / ('signed_self_v1_{}_{}.npz'.format(digest, k))
        if path.exists():
            return as_torch(sp.load_npz(path), features.device)
    z = features.detach() / norm
    rows, cols, values = [], [], []
    for start in range(0, len(z), chunk_size):
        score, index = (z[start:start + chunk_size] @ z.T).topk(k, dim=1)
        rows.append(np.repeat(np.arange(start, start + len(score)), k))
        cols.append(index.cpu().numpy().ravel())
        values.append(score.cpu().numpy().ravel())
    row, col, value = np.concatenate(rows), np.concatenate(cols), np.concatenate(values)
    degree = np.bincount(row, weights=value, minlength=len(z)).astype(np.float32)
    if (degree <= 0).any():
        raise ValueError('Original signed KNN has nonpositive degree; cannot reproduce its normalization safely')
    value = value * degree[row] ** -0.5 * degree[col] ** -0.5
    graph = sp.csr_matrix((value, (row, col)), shape=(len(z), len(z)))
    if path:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.stem + '.{}.npz'.format(os.getpid()))
        sp.save_npz(temporary, graph)
        os.replace(temporary, path)
    return as_torch(graph, features.device)


def neighbor_intersection(image, text):
    a, b = as_scipy(image), as_scipy(text)
    a.data[:] = 1
    b.data[:] = 1
    common = a.multiply(b).tocsr()
    common.setdiag(0)
    common.eliminate_zeros()
    return {i: common.indices[common.indptr[i]:common.indptr[i + 1]].tolist() for i in range(common.shape[0])}


def grouped_softmax(scores, groups, group_count):
    # Detaching the numerical shift is exact: softmax is invariant to a row shift.
    with torch.no_grad():
        maximum = scores.new_full((group_count,), -torch.inf)
        maximum.scatter_reduce_(0, groups, scores, reduce='amax', include_self=True)
    values = (scores - maximum[groups]).exp()
    total = values.new_zeros(group_count).index_add(0, groups, values)
    return values / total[groups]


def unique_infonce(first, second, ids, temperature):
    """One directional InfoNCE over unique identities, with logsumexp stability."""
    if temperature <= 0:
        raise ValueError('InfoNCE temperature must be positive')
    ids = ids.unique()
    if len(ids) < 2:
        return (first[ids].sum() + second[ids].sum()) * 0
    a, b = F.normalize(first[ids], dim=1), F.normalize(second[ids], dim=1)
    logits = a @ b.T / temperature
    return F.cross_entropy(logits, torch.arange(len(ids), device=ids.device))
