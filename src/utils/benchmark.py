"""Separate full-graph encoding and full-ranking costs without metrics or file IO."""
from time import perf_counter
import numpy as np
import torch


@torch.no_grad()
def benchmark_ranking(model, loader, device, repeats=5):
    def sync():
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
    measurements = []
    model.eval()
    for repeat in range(repeats + 1):
        sync()
        start = perf_counter()
        model.prepare_evaluation()
        sync()
        encoded = perf_counter()
        users = 0
        try:
            for batch in loader:
                scores = model.full_sort_predict(batch)
                scores[batch[1][0], batch[1][1]] = -torch.inf
                scores.topk(min(20, scores.shape[1]), dim=1)
                users += len(batch[0])
            sync()
            end = perf_counter()
            if repeat:
                measurements.append([encoded-start, end-encoded, end-start])
        finally:
            model.clear_evaluation()
    values = np.asarray(measurements)
    return {'repeats': repeats, 'warmup_passes': 1, 'users': users,
            'encoding_seconds_median': float(np.median(values[:, 0])),
            'ranking_seconds_median': float(np.median(values[:, 1])),
            'total_seconds_median': float(np.median(values[:, 2])),
            'users_per_second_median': float(np.median(users / values[:, 2])),
            'includes': 'one graph encoding plus all user-item scores, train mask and top20; excludes metric calculation and IO'}
