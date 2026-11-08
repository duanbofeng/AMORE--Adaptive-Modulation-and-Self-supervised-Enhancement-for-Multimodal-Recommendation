"""Independent content-score probe, not a claimed new SOTA model.

LightGCN collaborative score + modality-specific history/content scores.
No item graph, ID feature gate, affinity, modality-user table or auxiliary loss.
Training removes the positive item's content from BOTH sides of a BPR comparison.
"""
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from common.abstract_recommender import GeneralRecommender
from utils.graph_ops import as_torch, ui_graph


class CONTENTCF(GeneralRecommender):
    def __init__(self, config, dataset, local_time=None):
        super().__init__(config, dataset)
        self.config = config
        self.dim = config['embedding_size']
        self.content_weight = float(config['content_weight'])
        self.cross_weight = float(config['cross_weight'])
        if self.content_weight < 0 or self.cross_weight < 0 or config['history_shrinkage'] <= 0:
            raise ValueError('Content/cross weights must be nonnegative; history_shrinkage must be positive')
        if config['modality'] not in ('both', 'image', 'text') or config['n_ui_layers'] < 0:
            raise ValueError('Invalid modality or number of UI layers')
        self.user_embedding = nn.Embedding(self.n_users, self.dim)
        self.item_id_embedding = nn.Embedding(self.n_items, self.dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_id_embedding.weight)
        interactions = dataset.inter_matrix(form='csr').astype(np.float32)
        interactions.sum_duplicates()
        interactions.data[:] = 1
        self.register_buffer('adj', as_torch(ui_graph(interactions), self.device), persistent=False)
        self.register_buffer('history', as_torch(interactions, self.device), persistent=False)
        self.register_buffer('degree', torch.as_tensor(np.asarray(interactions.sum(1)),
                                                     dtype=torch.float32, device=self.device), persistent=False)
        self.features = nn.ModuleList()
        self.projections = nn.ModuleList()
        for modality, feature in [('image', self.v_feat), ('text', self.t_feat)]:
            if feature is None or config['modality'] not in ('both', modality):
                continue
            if len(feature) != self.n_items or not torch.isfinite(feature).all():
                raise ValueError('Feature rows must match item IDs and be finite')
            self.features.append(nn.Embedding.from_pretrained(F.normalize(feature, dim=1), freeze=True))
            self.projections.append(nn.Linear(feature.shape[1], self.dim, bias=False))
        if not self.features:
            raise ValueError('No selected content features')
        self.diagnostics = {}

    def forward(self):
        initial = torch.cat([self.user_embedding.weight, self.item_id_embedding.weight])
        current, total = initial, initial
        for _ in range(self.config['n_ui_layers']):
            current = torch.sparse.mm(self.adj, current)
            total = total + current
        collaborative = total / (self.config['n_ui_layers'] + 1)
        content = [F.normalize(projection(feature.weight), dim=1)
                   for feature, projection in zip(self.features, self.projections)]
        sums = [torch.sparse.mm(self.history, view) for view in content]
        return collaborative, content, sums

    def _profiles(self, users, content, sums, excluded_positive=None):
        denominator = self.degree[users] + self.config['history_shrinkage']
        if excluded_positive is None:
            return [total[users] / denominator for total in sums]
        denominator = denominator - 1
        return [(total[users] - view[excluded_positive]) / denominator
                for total, view in zip(sums, content)]

    def _paired_scores(self, users, items, views, profiles):
        collaborative, content, _ = views
        user_id = collaborative[users]
        item_id = collaborative[self.n_users + items]
        scores = (user_id * item_id).sum(1)
        count = len(content)
        for profile, view in zip(profiles, content):
            scores = scores + self.content_weight / count * (profile * view[items]).sum(1)
            if self.cross_weight:
                scores = scores + self.cross_weight * math.sqrt(self.content_weight) / count * (
                    (user_id * view[items]).sum(1) + (profile * item_id).sum(1))
        return scores

    def calculate_loss(self, interaction):
        users, positive, negative = interaction[:3]
        views = self.forward()
        # Same history for both scores; no asymmetric removal only from the positive score.
        profiles = self._profiles(users, views[1], views[2], positive)
        margin = self._paired_scores(users, positive, views, profiles) - self._paired_scores(users, negative, views, profiles)
        ranking = F.softplus(-margin).mean()
        penalty = (self.user_embedding(users).square().sum() + self.item_id_embedding(positive).square().sum()
                   + self.item_id_embedding(negative).square().sum())
        penalty = penalty + sum(projection.weight.square().sum() for projection in self.projections)
        return ranking, self.config['reg_weight'] * penalty / (2 * len(users))

    def prepare_evaluation(self):
        self._eval_embeddings = self.forward()

    def clear_evaluation(self):
        self._eval_embeddings = None

    def full_sort_predict(self, interaction):
        views = getattr(self, '_eval_embeddings', None)
        if views is None:
            views = self.forward()
        collaborative, content, sums = views
        users = interaction[0]
        user_id, item_id = collaborative[users], collaborative[self.n_users:]
        scores = user_id @ item_id.T
        profiles = self._profiles(users, content, sums)
        for profile, view in zip(profiles, content):
            scores = scores + self.content_weight / len(content) * (profile @ view.T)
            if self.cross_weight:
                scores = scores + self.cross_weight * math.sqrt(self.content_weight) / len(content) * (
                    user_id @ view.T + profile @ item_id.T)
        return scores
