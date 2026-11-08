"""Original full AMORE on the validation-only experiment runner.

Forward/loss are inherited verbatim from the archived user implementation.
Only graph construction, equivalent vectorized modulation and evaluation caching
are adapted. See docs/ORIGINAL_AMORE.md for provenance and exact boundaries.
"""
import os
import numpy as np
import torch
from torch import nn

from common.abstract_recommender import GeneralRecommender
from models.archive.amore_original import GUME as OriginalAMORE
from utils.original_amore_ops import original_knn, neighbor_intersection, grouped_softmax


class AMORELEGACY(OriginalAMORE):
    def __init__(self, config, dataset, local_time):
        GeneralRecommender.__init__(self, config, dataset)
        self.config = config
        self.diagnostics = {}
        self.sparse = True
        self.bm_loss = config['bm_loss']
        self.um_loss = config['um_loss']
        self.vt_loss = config['vt_loss']
        self.reg_weight_1 = config['reg_weight_1']
        self.reg_weight_2 = config['reg_weight_2']
        self.bm_temp = config['bm_temp']
        self.um_temp = config['um_temp']
        self.n_ui_layers = config['n_ui_layers']
        self.embedding_dim = config['embedding_size']
        self.knn_k = config['knn_k']
        self.n_layers = config['n_layers']
        
        # EgoGCN-style affinity threshold parameters
        self.use_affinity_modulation = config['use_affinity_modulation']
        self.affinity_threshold = config['affinity_threshold']  # Format: '>0.5' or '<0.3'
        self.affinity_reattn = config['affinity_reattn']  # Whether to re-compute attention after modulation
        
        # Adaptive optimization parameters
        self.use_adaptive_bpr = config['use_adaptive_bpr'] # 是否使用自适应BPR优化
        self.adaptive_weight_smooth = config['adaptive_weight_smooth']  # 自适应权重平滑系数
        
        # Residual enhancement parameters for decoupling
        self.residual_alpha = config['residual_alpha'] # 细粒度残差连接权重
        self.fine_grained_weight = config['fine_grained_weight'] # 细粒度特征融合权重

        # load dataset info
        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_id_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_id_embedding.weight)
        
        self.extended_image_user = nn.Embedding(self.n_users, self.embedding_dim)
        nn.init.xavier_uniform_(self.extended_image_user.weight)
        
        self.extended_text_user = nn.Embedding(self.n_users, self.embedding_dim)
        nn.init.xavier_uniform_(self.extended_text_user.weight)

        self.dataset_path = os.path.abspath(os.getcwd()+config['data_path'] + config['dataset'])
        # Fix path join to avoid duplicating cwd
        self.dataset_path = os.path.join(config['data_path'], config['dataset'])
        self.data_name = config['dataset']

        if self.v_feat is None or self.t_feat is None:
            raise ValueError('Original AMORE requires both image and text features')
        self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False)
        self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False)
        for name, feature in [('image', self.v_feat), ('text', self.t_feat)]:
            graph = original_knn(feature, self.knn_k, config['graph_chunk_size'],
                                 os.path.join(self.dataset_path, 'original_graph_cache'))
            self.register_buffer(name + '_original_adj', graph, persistent=False)

        #  Enhancing User-Item Graph
        self.inter = self.find_inter(self.image_original_adj, self.text_original_adj)
        self.ii_adj = self.add_edge(self.inter)
        self.norm_adj = self.get_adj_mat(self.ii_adj.tolil())
        self.R = self.sparse_mx_to_torch_sparse_tensor(self.R).float().to(self.device)
        self.norm_adj = self.sparse_mx_to_torch_sparse_tensor(self.norm_adj).float().to(self.device)
        
        
        self.image_reduce_dim = nn.Linear(self.v_feat.shape[1], self.embedding_dim)
        self.image_trans_dim = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.Sigmoid()
        )
        self.image_space_trans = nn.Sequential(
            self.image_reduce_dim,
            self.image_trans_dim
        )
        
        self.text_reduce_dim = nn.Linear(self.t_feat.shape[1], self.embedding_dim)
        self.text_trans_dim = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.Sigmoid()
        )
        self.text_space_trans = nn.Sequential(
            self.text_reduce_dim,
            self.text_trans_dim
        )
        
        self.separate_coarse = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.Tanh(),
            nn.Linear(self.embedding_dim, 1, bias=False)
        )
        
        # 亲和度引导的动态粗粒度生成网络
        # 输入：模态嵌入 + 亲和度信息，输出：亲和度调节后的权重
        self.affinity_guided_coarse = nn.Sequential(
            nn.Linear(self.embedding_dim + 1, self.embedding_dim),  # +1 for affinity
            nn.ReLU(),
            nn.Linear(self.embedding_dim, 1, bias=False)
        )
        
        # 亲和度强度控制参数（可学习）
        self.affinity_coarse_alpha = nn.Parameter(torch.tensor(0.5))  # 控制亲和度对粗粒度的影响强度
        
        self.softmax = nn.Softmax(dim=-1)
                
        self.image_behavior = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.Sigmoid()
        )
        self.text_behavior = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.Sigmoid()
        )

        self.tau = 0.5
        for name in ['R', 'norm_adj']:
            matrix = getattr(self, name)
            delattr(self, name)
            self.register_buffer(name, matrix.coalesce(), persistent=False)
        indices = self.norm_adj.indices()
        valid = (indices[0] < self.n_users) & (indices[1] >= self.n_users)
        self.register_buffer('ui_edges', torch.stack([indices[0, valid], indices[1, valid] - self.n_users]), persistent=False)
        self.register_buffer('item_degree', torch.bincount(self.ui_edges[1], minlength=self.n_items), persistent=False)
        if config['affinity_threshold'][0] not in ('>', '<'):
            raise ValueError('affinity_threshold must start with > or <')
        threshold = float(config['affinity_threshold'][1:])
        if not 0 <= threshold <= 1:
            raise ValueError('affinity threshold must lie in [0,1]')

    def find_inter(self, image_adj, text_adj):
        return neighbor_intersection(image_adj, text_adj)

    def _edge_dot(self, users, items):
        u, i = self.ui_edges
        chunk = self.config['edge_chunk_size']
        return torch.cat([(users[u[a:a + chunk]] * items[i[a:a + chunk]]).sum(1)
                          for a in range(0, len(u), chunk)])

    def apply_edge_level_hard_modulation(self, adj, user_embeds, item_embeds, item_multimodal_embeds):
        # Closed-form item mean is the original edge feature averaging operation:
        # mean_u[item + 0.5*selected_ui*(multimodal-item)]. No E x D messages.
        users, items = self.ui_edges
        if len(users) == 0:
            return item_embeds, item_embeds.new_zeros(self.n_items)
        scores = self._edge_dot(user_embeds, item_embeds)
        affinity = grouped_softmax(scores, users, self.n_users)
        threshold = float(self.affinity_threshold[1:])
        above = self.affinity_threshold[0] == '>'
        selected = affinity > threshold if above else affinity < threshold
        if self.affinity_reattn:
            alternative = self._edge_dot(user_embeds, item_multimodal_embeds)
            scores = scores + selected.to(scores.dtype) * (alternative - scores) / 2
            affinity = grouped_softmax(scores, users, self.n_users)
            selected = affinity > threshold if above else affinity < threshold
        count = self.item_degree.to(scores.dtype).clamp_min(1)
        ratio = scores.new_zeros(self.n_items).index_add(0, items, selected.to(scores.dtype)) / count
        item_affinity = scores.new_zeros(self.n_items).index_add(0, items, affinity) / count
        output = item_embeds + ratio[:, None] * (item_multimodal_embeds - item_embeds) / 2
        if not self.training:
            self.diagnostics['last_modulation_selected_fraction'] = float(selected.float().mean())
        return output, item_affinity

    @torch.no_grad()
    def prepare_evaluation(self):
        self._eval_embeddings = self.forward(self.norm_adj)

    def clear_evaluation(self):
        self._eval_embeddings = None

    def full_sort_predict(self, interaction):
        embeds = getattr(self, '_eval_embeddings', None)
        if embeds is None:
            embeds = self.forward(self.norm_adj)
        users, items = embeds.split([self.n_users, self.n_items])
        return users[interaction[0]] @ items.T
