"""AMORE research candidates; no performance or novelty claim is implied.

GUME-derived ID/purified-feature pathways, independently testable extensions.
See docs/AMORE_RESEARCH.md for equations, attribution and experimental protocol.
"""
import os
import numpy as np
import scipy.sparse as sp
import torch
from torch import nn
from torch.nn import functional as F

from common.abstract_recommender import GeneralRecommender
from utils.graph_ops import as_torch, knn_graph, normalize, shared_graph, ui_graph, edge_affinity


class AMORE(GeneralRecommender):
    def __init__(self, config, dataset, local_time=None):
        super().__init__(config, dataset)
        self.config = config
        self.dim = config['embedding_size']
        self.ui_layers = config['n_ui_layers']
        self.ii_layers = config['n_layers']
        self.scale = config['semantic_weight']
        self.cl_weight = config['cl_weight']
        self.temperature = config['cl_temperature']
        self._validate()
        interactions = dataset.inter_matrix(form='csr').astype(np.float32)
        interactions.sum_duplicates()
        interactions.data[:] = 1
        self.user_embedding = nn.Embedding(self.n_users, self.dim)
        self.item_id_embedding = nn.Embedding(self.n_items, self.dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_id_embedding.weight)
        self.register_buffer('adj', as_torch(ui_graph(interactions), self.device), persistent=False)
        self.features = nn.ModuleList()
        self.projections = nn.ModuleList()
        self.graph_count = 0
        self.diagnostics = {}
        if self.scale == 0 and self.cl_weight == 0:
            if config['user_init'] != 'random':
                raise ValueError('The ID-only control requires user_init=random')
            self.v_feat, self.t_feat = None, None
            self.to(self.device)
            return
        raw = []
        for modality, feat in [('image', self.v_feat), ('text', self.t_feat)]:
            if feat is None or (config['modality'] != 'both' and config['modality'] != modality):
                continue
            if feat.shape[0] != self.n_items or not torch.isfinite(feat).all():
                raise ValueError('Feature rows must match item IDs and be finite')
            raw.append(feat)
            self.features.append(nn.Embedding.from_pretrained(feat, freeze=config['freeze_features']))
            self.projections.append(nn.Linear(feat.shape[1], self.dim))
        if not raw:
            raise ValueError('No selected modality features available')
        cache = os.path.join(config['data_path'], config['dataset'], 'graph_cache')
        graphs = [knn_graph(f, config['knn_k'], config['graph_chunk_size'], cache_dir=cache) for f in raw]
        if config['graph_mode'] == 'shared':
            graph = shared_graph(graphs, interactions, config['knn_k'], config['cooc_weight'],
                                 config['ncer_strength'], config['graph_chunk_size'])
            graphs = [graph] * len(graphs)
        else:
            graphs = [normalize(g) for g in graphs]
        # Symmetric UI normalization from the unmodified training graph.
        du = np.maximum(np.asarray(interactions.sum(1)).ravel(), 1)
        di = np.maximum(np.asarray(interactions.sum(0)).ravel(), 1)
        for m, (feature, graph) in enumerate(zip(raw, graphs)):
            weights = interactions
            if config['affinity_mode'] != 'none':
                weights = edge_affinity(feature, interactions, config['affinity_mode'],
                                        config['affinity_strength'], config['affinity_shrinkage'],
                                        config['affinity_temperature'], config['seed'])
            # Do not renormalize gates away; preserve original graph degree scale.
            aggregation = sp.diags(du ** -0.5) @ weights @ sp.diags(di ** -0.5)
            self.register_buffer('graph_{}'.format(m), as_torch(graph, self.device), persistent=False)
            self.register_buffer('aggregation_{}'.format(m), as_torch(aggregation, self.device), persistent=False)
            self.diagnostics['affinity_{}_mean'.format(m)] = float(weights.data.mean())
            self.diagnostics['affinity_{}_std'.format(m)] = float(weights.data.std())
            self.diagnostics['affinity_{}_quantiles'.format(m)] = np.quantile(weights.data, [0, .1, .5, .9, 1]).tolist()
            edge_degree = du[weights.tocoo().row]
            correlation = np.corrcoef(weights.data, edge_degree)[0, 1] if weights.data.std() > 1e-8 and edge_degree.std() > 0 else None
            self.diagnostics['affinity_{}_degree_correlation'.format(m)] = float(correlation) if correlation is not None else None
            self.diagnostics['ii_{}_edges'.format(m)] = int(graph.nnz)
        self.graph_count = len(graphs)
        if config['fusion'] == 'attention':
            self.attention = nn.Linear(self.dim, 1, bias=False)
        if config['residual_gate']:
            self.residual_gate = nn.Linear(2 * self.dim, 1)
            nn.init.zeros_(self.residual_gate.weight)
            nn.init.zeros_(self.residual_gate.bias)
        self.to(self.device)
        if config['user_init'] != 'random':
            self._initialize_users(interactions)

    def _validate(self):
        c = self.config
        for key, choices in [('affinity_mode', ['none', 'global', 'item', 'local', 'shuffle']),
                             ('graph_mode', ['separate', 'shared']), ('fusion', ['mean', 'attention']),
                             ('user_init', ['random', 'history', 'cluster']),
                             ('modality', ['both', 'image', 'text'])]:
            if c[key] not in choices:
                raise ValueError('{} must be one of {}'.format(key, choices))
        for key in ['affinity_strength', 'cooc_weight', 'init_cluster_weight', 'init_blend']:
            if not 0 <= c[key] <= 1:
                raise ValueError(key + ' must lie in [0, 1]')
        for key in ['affinity_temperature', 'affinity_shrinkage', 'cl_temperature']:
            if c[key] <= 0:
                raise ValueError(key + ' must be positive')
        for key in ['cl_weight', 'reg_weight', 'semantic_weight', 'ncer_strength', 'n_ui_layers', 'n_layers']:
            if c[key] < 0:
                raise ValueError(key + ' must be nonnegative')
        if c['graph_mode'] != 'shared' and (c['cooc_weight'] or c['ncer_strength']):
            raise ValueError('co-occurrence and NCER require graph_mode=shared')

    @torch.no_grad()
    def _initialize_users(self, interactions):
        """SG-URInit-inspired adaptation to a shared ID embedding space.

        Use purified item representations, inverse-popularity aggregation and optional
        cluster centroids; match each user's original norm. No extra trainable tables.
        """
        items = torch.stack([self.item_id_embedding.weight * torch.sigmoid(p(f.weight))
                             for f, p in zip(self.features, self.projections)]).mean(0)
        source = items
        if self.config['user_init'] == 'cluster':
            count = min(self.config['init_clusters'], self.n_items)
            if count < 1:
                raise ValueError('init_clusters must be positive')
            centers = items[torch.randperm(self.n_items, device=items.device)[:count]].clone()
            for _ in range(self.config['init_kmeans_steps']):
                labels = torch.cat([torch.cdist(block, centers).argmin(1) for block in items.split(4096)])
                sums = torch.zeros_like(centers).index_add_(0, labels, items)
                counts = torch.bincount(labels, minlength=count).unsqueeze(1)
                centers = torch.where(counts > 0, sums / counts.clamp_min(1), centers)
            labels = torch.cat([torch.cdist(block, centers).argmin(1) for block in items.split(4096)])
            weight = self.config['init_cluster_weight']
            source = (1 - weight) * items + weight * centers[labels]
        popularity = np.maximum(np.asarray(interactions.sum(0)).ravel(), 1)
        history = as_torch(interactions @ sp.diags(1 / popularity), self.device)
        user = torch.sparse.mm(history, source)
        original = self.user_embedding.weight
        user = F.normalize(user, dim=1) * original.norm(dim=1, keepdim=True)
        blend = self.config['init_blend']
        valid = user.norm(dim=1, keepdim=True) > 0
        original.copy_(torch.where(valid, (1 - blend) * original + blend * user, original))

    def forward(self, return_views=False):
        initial = torch.cat([self.user_embedding.weight, self.item_id_embedding.weight])
        current, total = initial, initial
        for _ in range(self.ui_layers):
            current = torch.sparse.mm(self.adj, current)
            total = total + current
        collaborative = total / (self.ui_layers + 1)
        if self.graph_count == 0:
            return (collaborative, collaborative, torch.zeros_like(collaborative)) if return_views else collaborative
        modalities = []
        for m, (feature, projection) in enumerate(zip(self.features, self.projections)):
            item = self.item_id_embedding.weight * torch.sigmoid(projection(feature.weight))
            for _ in range(self.ii_layers):
                item = torch.sparse.mm(getattr(self, 'graph_{}'.format(m)), item)
            user = torch.sparse.mm(getattr(self, 'aggregation_{}'.format(m)), item)
            modalities.append(torch.cat([user, item]))
        views = torch.stack(modalities, dim=1)
        if self.config['fusion'] == 'attention':
            semantic = (self.attention(views).softmax(dim=1) * views).sum(1)
        else:
            semantic = views.mean(1)
        if self.config['residual_gate']:
            gate = torch.sigmoid(self.residual_gate(torch.cat([collaborative, semantic], dim=1)))
            semantic = semantic * gate
        output = collaborative + self.scale * semantic
        return (output, collaborative, semantic) if return_views else output

    def calculate_loss(self, interaction):
        user, positive, negative = interaction[:3]
        output, collaborative, semantic = self.forward(return_views=True)
        users, items = output.split([self.n_users, self.n_items])
        margin = (users[user] * (items[positive] - items[negative])).sum(1)
        ranking = F.softplus(-margin).mean()
        regularization = self.config['reg_weight'] * (
            self.user_embedding(user).square().sum() + self.item_id_embedding(positive).square().sum()
            + self.item_id_embedding(negative).square().sum()) / (2 * len(user))
        contrastive = ranking.new_zeros(())
        if self.cl_weight:
            # Deduplicate IDs so repeated positives are not treated as false negatives.
            for ids in [user.unique(), positive.unique() + self.n_users]:
                a, b = F.normalize(collaborative[ids], dim=1), F.normalize(semantic[ids], dim=1)
                logits = a @ b.T / self.temperature
                labels = torch.arange(len(ids), device=ids.device)
                contrastive = contrastive + (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2
            contrastive = self.cl_weight * contrastive
        return ranking, regularization, contrastive

    def prepare_evaluation(self):
        self._eval_embeddings = self.forward()

    def clear_evaluation(self):
        self._eval_embeddings = None

    def full_sort_predict(self, interaction):
        output = getattr(self, '_eval_embeddings', None)
        if output is None:
            output = self.forward()
        users, items = output.split([self.n_users, self.n_items])
        return users[interaction[0]] @ items.T
