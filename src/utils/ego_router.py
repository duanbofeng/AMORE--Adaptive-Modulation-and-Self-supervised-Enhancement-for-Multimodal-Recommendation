"""Two-modality adaptation of EgoGCN's GATConv_pre message modulation.

Only train UI edges enter this layer. No torch_geometric / torch_scatter needed.
Cosine logits are bounded; grouped softmax uses index_add (PyTorch 1.12).
"""
from functools import partial
import inspect

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


class EdgeAggregate(torch.autograd.Function):
    """Sparse weighted sum with edge-only gradients, including on PyTorch 1.12.

    Native 1.12 COO matmul backward constructs grad @ items.T before masking:
    that allocates U x I when edge weights require grad. Compute only the
    existing edge derivatives instead. This layer needs first-order gradients.
    """
    @staticmethod
    def forward(ctx, edges, weight, items, n_users, chunk_size):
        ctx.save_for_backward(edges, weight, items)
        ctx.n_users, ctx.chunk_size = n_users, chunk_size
        adj = torch.sparse_coo_tensor(edges, weight, (n_users, len(items)))
        return torch.sparse.mm(adj, items)

    @staticmethod
    def backward(ctx, grad_output):
        edges, weight, items = ctx.saved_tensors
        grad_weight = grad_items = None
        if ctx.needs_input_grad[1]:
            grad_weight = torch.empty_like(weight)
            for start in range(0, len(weight), ctx.chunk_size):
                stop = min(start + ctx.chunk_size, len(weight))
                users, indices = edges[:, start:stop]
                grad_weight[start:stop] = (grad_output[users] * items[indices]).sum(dim=1)
        if ctx.needs_input_grad[2]:
            transpose = torch.sparse_coo_tensor(edges.flip(0), weight, (len(items), ctx.n_users))
            grad_items = torch.sparse.mm(transpose, grad_output)
        return None, grad_weight, grad_items, None, None


class EgoRouter(nn.Module):
    def __init__(self, matrix, dim, config):
        super().__init__()
        self.mode = config['ego_mode']
        if self.mode not in ('soft', 'hard', 'attention', 'uniform'):
            raise ValueError('ego_mode must be off, soft, hard, attention, or uniform')
        self.strength = float(config['ego_strength'])
        self.temperature = float(config['ego_temperature'])
        self.threshold = float(config['ego_threshold'])
        self.threshold_type = config['ego_threshold_type']
        self.reattention = bool(config['ego_reattention'])
        self.loo = bool(config['ego_leave_one_out'])
        self.shuffle = bool(config['ego_shuffle'])
        self.cross_source = config['ego_cross_source']
        if self.cross_source not in ('other', 'self'):
            raise ValueError('ego_cross_source must be other or self')
        self.chunk = int(config['ego_chunk_size'])
        self.checkpoint_chunks = bool(config['ego_checkpoint'])
        if not 0 <= self.strength <= 1 or not 0.05 <= self.temperature <= 10:
            raise ValueError('ego_strength must be in [0,1]; ego_temperature in [0.05,10]')
        if self.chunk < 1 or self.threshold < 0 or self.threshold_type not in ('relative', 'absolute'):
            raise ValueError('Invalid Ego chunk size or hard threshold')
        matrix = matrix.coalesce()
        self.n_users, self.n_items = matrix.shape
        self.register_buffer('edges', matrix.indices(), persistent=False)
        self.register_buffer('base_weight', matrix.values(), persistent=False)
        users = self.edges[0]
        degree = torch.bincount(users, minlength=self.n_users)
        mass = matrix.values().new_zeros(self.n_users).index_add(0, users, matrix.values())
        self.register_buffer('degree', degree, persistent=False)
        self.register_buffer('mass', mass, persistent=False)
        # Local RNG keeps common GUME parameters and subsequent training RNG identical.
        with torch.random.fork_rng(devices=[]):
            # Seed CPU only: torch.manual_seed would also alter CUDA's training RNG.
            torch.default_generator.manual_seed(int(config['seed']) + 17011)
            if self.mode == 'soft':
                self.cross_gates = nn.ModuleList([nn.Linear(dim, dim), nn.Linear(dim, dim)])
                for gate in self.cross_gates:
                    nn.init.zeros_(gate.bias)
            generator = torch.Generator().manual_seed(int(config['seed']) + 17012)
            permutations, offset = [], 0
            for count in degree.cpu().tolist():
                permutations.append(torch.randperm(count, generator=generator) + offset)
                offset += count
            permutation = torch.cat(permutations) if permutations else torch.empty(0, dtype=torch.long)
        self.register_buffer('permutation', permutation.to(users.device), persistent=False)
        self.checkpoint_kwargs = ({'use_reentrant': False}
                                  if 'use_reentrant' in inspect.signature(checkpoint).parameters else {})
        self.diagnostics = {}

    def _scores_chunk(self, start, stop, own, message, history, alternate, selection):
        users, items = self.edges[:, start:stop]
        query = history[users]
        if self.loo:
            query = query - self.base_weight[start:stop, None] * own[items]
            query = torch.where((self.degree[users] > 1)[:, None], query, torch.zeros_like(query))
        sent = message[items]
        if selection is not None:
            sent = sent + selection[start:stop, None] * (alternate[items] - sent) / 2
        return F.cosine_similarity(query, sent, dim=1, eps=1e-8) / self.temperature

    def _scores(self, own, message, history, alternate=None, selection=None):
        chunks = []
        for start in range(0, self.edges.shape[1], self.chunk):
            fn = partial(self._scores_chunk, start, min(start + self.chunk, self.edges.shape[1]))
            args = (own, message, history, alternate, selection)
            if self.checkpoint_chunks and torch.is_grad_enabled() and own.requires_grad:
                chunks.append(checkpoint(fn, *args, **self.checkpoint_kwargs))
            else:
                chunks.append(fn(*args))
        scores = torch.cat(chunks) if chunks else own.new_empty(0)
        return scores[self.permutation] if self.shuffle else scores

    def _softmax(self, scores):
        # cos/T <= 1/T; shift is constant and cancels within every user's row.
        weight = (scores - 1 / self.temperature).exp()
        sums = weight.new_zeros(self.n_users).index_add(0, self.edges[0], weight)
        return weight / sums[self.edges[0]].clamp_min(torch.finfo(weight.dtype).tiny)

    def _aggregate(self, weight, items):
        return EdgeAggregate.apply(self.edges, weight, items, self.n_users, self.chunk)

    def _route(self, own, other, history, branch, collect):
        if self.cross_source == 'self':
            other = own  # Same-size control: gated rescaling without cross-modal information.
        users = self.edges[0]
        uniform = self.degree[users].to(own.dtype).reciprocal()
        if self.mode == 'uniform':
            first = uniform
        elif self.mode == 'soft' and self.reattention and not collect:
            # Soft channel gates do not use first-pass alpha; only the recomputed
            # alpha affects aggregation. Avoid an unused full edge pass in training.
            first = None
        else:
            first = self._softmax(self._scores(own, own, history))
        selection = None
        message = own
        if self.mode == 'soft':
            message = (own + torch.sigmoid(self.cross_gates[branch](other)) * other) / 2
        elif self.mode == 'uniform':
            message = (own + other) / 2
        elif self.mode == 'hard':
            cutoff = self.threshold * uniform if self.threshold_type == 'relative' else self.threshold
            # A true hard decision, as in EgoGCN; no unreported straight-through estimator.
            selection = (first > cutoff).to(own.dtype)
        weight = first
        if self.reattention and self.mode in ('soft', 'hard'):
            weight = self._softmax(self._scores(own, message, history, other, selection))
        scaled = weight * self.mass[users]  # Preserve GUME's row mass, not necessarily its vector norm.
        if selection is None:
            routed = self._aggregate(scaled, message)
        else:
            routed = (self._aggregate(scaled * (1 - selection / 2), own)
                      + self._aggregate(scaled * selection / 2, other))
        # A degree-one user's query has no other evidence; keep its original representation.
        routed = torch.where((self.degree > 1)[:, None], routed, history)
        delta = self.strength * (routed - history)
        if collect:
            prefix = 'ego_' + ('image' if branch == 0 else 'text')
            eligible = self.degree[users] > 1
            active = self.degree > 1
            relative = first / uniform
            self.diagnostics[prefix + '_attention_relative_std'] = float(relative.std(unbiased=False)) if len(first) else 0.
            self.diagnostics[prefix + '_update_ratio'] = float(
                (delta[active].norm(dim=1) / history[active].norm(dim=1).clamp_min(1e-8)).mean()) if active.any() else 0.
            if selection is not None:
                self.diagnostics[prefix + '_hard_fraction'] = float(selection[eligible].mean()) if eligible.any() else 0.
                for name, mask in [('degree_2_10', (self.degree[users] >= 2) & (self.degree[users] <= 10)),
                                   ('degree_11_plus', self.degree[users] > 10)]:
                    self.diagnostics[prefix + '_hard_' + name] = float(selection[mask].mean()) if mask.any() else 0.
            if self.mode == 'soft':
                self.diagnostics[prefix + '_cross_gate_mean'] = float(torch.sigmoid(self.cross_gates[branch](other)).mean())
        return history + delta

    def forward(self, image, text, image_users, text_users, collect=False):
        if self.strength == 0:
            return image_users, text_users
        return (self._route(image, text, image_users, 0, collect),
                self._route(text, image, text_users, 1, collect))
