"""Load checkpoints created by this training pipeline across PyTorch versions."""
import inspect

import torch


def load_checkpoint(path, map_location='cpu'):
    """Prefer weights-only loading when explicitly supported by torch.load.

    PyTorch 1.12 forwards unknown keywords to pickle's Unpickler, so passing
    weights_only there fails. Legacy loading is only for our own local checkpoints.
    Loading errors propagate; we do not retry a rejected weights-only load.
    """
    kwargs = {'map_location': map_location}
    if 'weights_only' in inspect.signature(torch.load).parameters:
        kwargs['weights_only'] = True
    return torch.load(path, **kwargs)
