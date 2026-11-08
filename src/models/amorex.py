"""Experimental candidate-dependent multimodal interest retrieval.

LightGCN supplies collaborative scores. Learned content prototypes partition TRAIN
histories into interest memories; each candidate is scored against all memories.
Training subtracts the target from every content memory used for its comparison.
No GUME inheritance, item KNN graph, ID feature gate, or auxiliary alignment loss.
See docs/AMOREX_EXPERIMENTS.md for equations, provenance and research limitations.
"""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from common.abstract_recommender import GeneralRecommender
from utils.graph_ops import as_torch, ui_graph


class AMOREX(GeneralRecommender):
    def __init__(self, config, dataset, local_time=None):
        super().__init__(config, dataset)
        self.config = config
        self.dim = config['embedding_size']
        self.k = config['num_interests']
        self.content_weight = float(config['content_weight'])
        self._validate()
        interactions = dataset.inter_matrix(form='csr').astype(np.float32)
        interactions.sum_duplicates()
        interactions.eliminate_zeros()
        interactions.data[:] = 1
        self.register_buffer('adj', as_torch(ui_graph(interactions), self.device), persistent=False)
        self.register_buffer('history', as_torch(interactions, self.device), persistent=False)
        edges = interactions.tocoo()
        keys = np.sort(edges.row.astype(np.int64) * self.n_items + edges.col)
        self.register_buffer('train_keys', torch.as_tensor(keys, device=self.device), persistent=False)
        self.user_embedding = nn.Embedding(self.n_users, self.dim)
        self.item_id_embedding = nn.Embedding(self.n_items, self.dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_id_embedding.weight)
        self.features = nn.ModuleList()
        self.projections = nn.ModuleList()
        self.prototypes = nn.ParameterList()
        self.modality_names = []
        # Keep sampled competitors matched across architectures with the same
        # seed, regardless of how many parameters initialization consumes.
        self._negative_generator = torch.Generator(device=self.device)
        self._negative_generator.manual_seed(int(config['seed']) + 104729)
        if self.content_weight:
            for name, feature in [('image', self.v_feat), ('text', self.t_feat)]:
                if feature is None or config['modality'] not in ('both', name):
                    continue
                if len(feature) != self.n_items or not torch.isfinite(feature).all():
                    raise ValueError('Feature rows must match item IDs and be finite')
                self.features.append(nn.Embedding.from_pretrained(F.normalize(feature, dim=1), freeze=True))
                self.projections.append(nn.Linear(feature.shape[1], self.dim, bias=False))
                self.modality_names.append(name)
            if not self.features:
                raise ValueError('No selected content features')
            # Create anchors AFTER all projections, so K=1 and K>1 start with the
            # same ID/projection weights under a matched seed.
            if self.k > 1:
                for _ in self.features:
                    anchors = nn.Parameter(torch.empty(self.k, self.dim))
                    nn.init.orthogonal_(anchors)
                    self.prototypes.append(anchors)
        self.v_feat, self.t_feat = None, None
        self.diagnostics = {}

    def _validate(self):
        c = self.config
        for key in ['embedding_size', 'num_interests', 'sampled_negatives', 'score_user_chunk', 'score_item_chunk']:
            if not isinstance(c[key], int) or isinstance(c[key], bool) or c[key] < 1:
                raise ValueError(key + ' must be a positive integer')
        for key in ['route_temperature', 'match_temperature', 'history_shrinkage', 'loss_temperature']:
            if not np.isfinite(c[key]) or c[key] <= 0:
                raise ValueError(key + ' must be finite and positive')
        for key in ['content_weight', 'reg_weight']:
            if not np.isfinite(c[key]) or c[key] < 0:
                raise ValueError(key + ' must be finite and nonnegative')
        if not isinstance(c['n_ui_layers'], int) or c['n_ui_layers'] < 0:
            raise ValueError('n_ui_layers must be a nonnegative integer')
        if c['objective'] not in ('bpr', 'sampled_ce') or c['modality'] not in ('both', 'image', 'text'):
            raise ValueError('Invalid objective or modality')

    def forward(self):
        initial = torch.cat([self.user_embedding.weight, self.item_id_embedding.weight])
        current, total = initial, initial
        for _ in range(self.config['n_ui_layers']):
            current = torch.sparse.mm(self.adj, current)
            total = total + current
        collaborative = total / (self.config['n_ui_layers'] + 1)
        memories = []
        for m, (feature, projection) in enumerate(zip(self.features, self.projections)):
            z = F.normalize(projection(feature.weight), dim=1)
            if self.k == 1:
                assignment = z.new_ones((self.n_items, 1))
            else:
                anchors = F.normalize(self.prototypes[m], dim=1)
                assignment = (z @ anchors.T / self.config['route_temperature']).softmax(1)
            # Sparse multiplication avoids materializing [training_edges, K, D].
            packed = torch.cat([z, z.new_ones((self.n_items, 1))], dim=1)
            weighted = assignment[:, :, None] * packed[:, None, :]
            sums = torch.sparse.mm(self.history, weighted.flatten(1)).view(self.n_users, self.k, self.dim + 1)
            memories.append((z, assignment, sums[:, :, :self.dim], sums[:, :, self.dim]))
        return collaborative, memories

    def profiles(self, users, state, excluded_positive=None):
        result = []
        for z, assignment, sums, mass in state[1]:
            numerator, counts = sums[users], mass[users]
            if excluded_positive is not None:
                weights = assignment[excluded_positive]
                numerator = numerator - weights[:, :, None] * z[excluded_positive, None, :]
                counts = counts - weights
            counts = counts.clamp_min(0)
            # Distribute a fixed total pseudocount over slots: uniform K slots
            # reproduce the K=1 profile rather than shrinking content K times.
            memory = numerator / (counts[:, :, None] + self.config['history_shrinkage'] / self.k)
            prior = (counts + 1e-8) / (counts.sum(1, keepdim=True) + self.k * 1e-8)
            result.append((memory, prior))
        return result

    def score_candidates(self, users, items, state, profiles, paired=False):
        """Shared items [J] or per-row items [B,J]; stream across interest slots."""
        collaborative, memories = state
        u = collaborative[users]
        item = collaborative[self.n_users + items]
        score = (u[:, None, :] * item).sum(-1) if paired else u @ item.T
        tau = self.config['match_temperature']
        for (z, _, _, _), (memory, prior) in zip(memories, profiles):
            candidates = z[items]
            pooled = None
            for k in range(self.k):
                similarity = ((memory[:, k, None, :] * candidates).sum(-1) if paired
                              else memory[:, k] @ candidates.T)
                term = similarity / tau + prior[:, k, None].log()
                pooled = term if pooled is None else torch.logaddexp(pooled, term)
            score = score + self.content_weight / len(memories) * tau * pooled
        return score

    def training_positive_mask(self, users, items):
        """Membership in TRAIN only. Never consult validation/test positives."""
        keys = users[:, None] * self.n_items + items
        if self.train_keys.numel() == 0:
            return torch.zeros_like(keys, dtype=torch.bool)
        locations = torch.searchsorted(self.train_keys, keys)
        return (locations < len(self.train_keys)) & (self.train_keys[locations.clamp_max(len(self.train_keys) - 1)] == keys)

    def sampled_logits(self, users, positive, negative, state, profiles, pool=None):
        if pool is None:
            pool = torch.randperm(self.n_items, device=users.device,
                                  generator=self._negative_generator)[:self.config['sampled_negatives']]
        # The loader's explicit negative guarantees at least one admissible competitor.
        pairs = torch.stack([positive, negative], dim=1)
        paired = self.score_candidates(users, pairs, state, profiles, paired=True)
        shared = self.score_candidates(users, pool, state, profiles)
        # Exclude known positives AND the explicit negative duplicated in the pool.
        invalid = self.training_positive_mask(users, pool[None, :]) | (pool[None, :] == negative[:, None])
        shared = shared.masked_fill(invalid, -torch.inf)
        return torch.cat([paired, shared], dim=1) / self.config['loss_temperature']

    def calculate_loss(self, interaction):
        users, positive, negative = interaction[:3]
        state = self.forward()
        profiles = self.profiles(users, state, excluded_positive=positive)
        if self.config['objective'] == 'bpr':
            scores = self.score_candidates(users, torch.stack([positive, negative], dim=1), state, profiles, paired=True)
            ranking = F.softplus((scores[:, 1] - scores[:, 0]) / self.config['loss_temperature']).mean()
        else:
            logits = self.sampled_logits(users, positive, negative, state, profiles)
            ranking = F.cross_entropy(logits, torch.zeros(len(users), dtype=torch.long, device=users.device))
        penalty = (self.user_embedding(users).square().sum() + self.item_id_embedding(positive).square().sum()
                   + self.item_id_embedding(negative).square().sum())
        penalty = penalty + sum(p.weight.square().sum() for p in self.projections)
        # Anchor directions are normalized; no extra diversity/alignment objective.
        return ranking, self.config['reg_weight'] * penalty / (2 * len(users))

    @torch.no_grad()
    def prepare_evaluation(self):
        self._eval_state = self.forward()
        active_users = self.history.coalesce().indices()[0].unique()[:1024]
        profiles = self.profiles(active_users, self._eval_state)
        for name, (_, assignment, _, _), (memory, prior) in zip(self.modality_names, self._eval_state[1], profiles):
            self.diagnostics[name + '_route_entropy'] = float(-(assignment * assignment.clamp_min(1e-8).log()).sum(1).mean())
            usage = assignment.mean(0)
            self.diagnostics[name + '_effective_slots'] = float(torch.exp(-(usage * usage.clamp_min(1e-8).log()).sum()))
            self.diagnostics[name + '_memory_norm'] = float(memory.norm(dim=-1).mean())
            if self.k > 1:
                unit = F.normalize(memory, dim=-1)
                cosine = unit @ unit.transpose(1, 2)
                mask = ~torch.eye(self.k, dtype=torch.bool, device=memory.device)
                self.diagnostics[name + '_slot_cosine'] = float(cosine[:, mask].mean())
        # Fixed TRAIN-edge probe, no held-out information. Monitor whether the
        # content score is actually visible relative to the collaborative score.
        probe = self.train_keys[:1024]
        if probe.numel():
            users, items = probe // self.n_items, probe % self.n_items
            profiles = self.profiles(users, self._eval_state, items)
            total = self.score_candidates(users, items[:, None], self._eval_state, profiles, paired=True).squeeze(1)
            cf = self._eval_state[0]
            base = (cf[users] * cf[self.n_users + items]).sum(1)
            self.diagnostics['probe_collaborative_abs'] = float(base.abs().mean())
            self.diagnostics['probe_content_abs'] = float((total - base).abs().mean())

    def clear_evaluation(self):
        self._eval_state = None

    def full_sort_predict(self, interaction):
        state = getattr(self, '_eval_state', None)
        if state is None:
            state = self.forward()
        users = interaction[0]
        scores = state[0].new_empty((len(users), self.n_items))
        for start in range(0, len(users), self.config['score_user_chunk']):
            selected = users[start:start + self.config['score_user_chunk']]
            profiles = self.profiles(selected, state)
            for offset in range(0, self.n_items, self.config['score_item_chunk']):
                items = torch.arange(offset, min(offset + self.config['score_item_chunk'], self.n_items), device=users.device)
                scores[start:start + len(selected), offset:offset + len(items)] = self.score_candidates(selected, items, state, profiles)
        return scores
