import os
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

from common.abstract_recommender import GeneralRecommender
from utils.utils import build_sim, compute_normalized_laplacian, build_knn_neighbourhood, build_knn_normalized_graph
from collections import defaultdict
import math
from scipy.sparse import lil_matrix
import random
import json

class GUME(GeneralRecommender):
    def __init__(self, config, dataset, local_time):
        super(GUME, self).__init__(config, dataset)
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

        image_adj_file = os.path.join(self.dataset_path, 'image_adj_{}_{}.pt'.format(self.knn_k, self.sparse))
        text_adj_file = os.path.join(self.dataset_path, 'text_adj_{}_{}.pt'.format(self.knn_k, self.sparse))

        if self.v_feat is not None:
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False)
            if os.path.exists(image_adj_file):
                image_adj = torch.load(image_adj_file)
            else:
                image_adj = build_sim(self.image_embedding.weight.detach())
                image_adj = build_knn_normalized_graph(image_adj, topk=self.knn_k, is_sparse=self.sparse,norm_type='sym')
                torch.save(image_adj, image_adj_file)
            self.image_original_adj = image_adj.to(self.device)

        if self.t_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False)
            if os.path.exists(text_adj_file):
                text_adj = torch.load(text_adj_file)
            else:
                text_adj = build_sim(self.text_embedding.weight.detach())
                text_adj = build_knn_normalized_graph(text_adj, topk=self.knn_k, is_sparse=self.sparse, norm_type='sym')
                torch.save(text_adj, text_adj_file)
            self.text_original_adj = text_adj.to(self.device)

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
    
    def find_inter(self, image_adj, text_adj):
        inter_file = os.path.join(self.dataset_path, 'inter.json')
        if os.path.exists(inter_file):
            with open(inter_file) as f:
                inter = json.load(f)
        else:
            j = 0
            inter = defaultdict(list)
            img_sim = []
            txt_sim = []
            for i in range(0,len(image_adj._indices()[0])):
                img_id = image_adj._indices()[0][i]
                txt_id = text_adj._indices()[0][i]
                assert img_id == txt_id
                id = img_id.item()
                img_sim.append(image_adj._indices()[1][j].item())
                txt_sim.append(text_adj._indices()[1][j].item())
                
                if len(img_sim)==10 and len(txt_sim)==10:
                    it_inter = list(set(img_sim) & set(txt_sim))
                    inter[id] = [v for v in it_inter if v != id]
                    img_sim = []
                    txt_sim = []
                
                j += 1
            
            with open(inter_file, "w") as f:
                json.dump(inter, f)
        
        return inter

    def add_edge(self, inter):
        sim_rows = []
        sim_cols = []
        for id, vs in inter.items():
            if len(vs) == 0:
                continue
            for v in vs:
                sim_rows.append(int(id))
                sim_cols.append(v)
        
        sim_rows = torch.tensor(sim_rows)
        sim_cols = torch.tensor(sim_cols)
        sim_values = [1]*len(sim_rows)

        item_adj = sp.coo_matrix((sim_values, (sim_rows, sim_cols)), shape=(self.n_items,self.n_items), dtype=np.int64)
        return item_adj
    
    def pre_epoch_processing(self):
        pass

    def get_adj_mat(self, item_adj):
        adj_mat = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)
        adj_mat = adj_mat.tolil()

        R = self.interaction_matrix.tolil()
        adj_mat[:self.n_users, self.n_users:] = R
        adj_mat[self.n_users:, :self.n_users] = R.T

        adj_mat[self.n_users:, self.n_users:] = item_adj
        
        adj_mat = adj_mat.todok()

        def normalized_adj_single(adj):
            rowsum = np.array(adj.sum(1))

            d_inv = np.power(rowsum, -0.5).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)

            norm_adj = d_mat_inv.dot(adj_mat)
            norm_adj = norm_adj.dot(d_mat_inv)
            return norm_adj.tocoo()

        norm_adj_mat = normalized_adj_single(adj_mat)
        norm_adj_mat = norm_adj_mat.tolil()
        
        self.R = norm_adj_mat[:self.n_users, self.n_users:]
        
        return norm_adj_mat.tocsr()

    def sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
        """Convert a scipy sparse matrix to a torch sparse tensor."""
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse.FloatTensor(indices, values, shape)
    
    def apply_edge_level_hard_modulation(self, adj, user_embeds, item_embeds, item_multimodal_embeds):
        """
        Apply edge-level hard modulation following EgoGCN logic.
        Optimized version with vectorized operations to avoid Python loops.
        
        Args:
            adj: Sparse adjacency matrix [n_users+n_items, n_users+n_items]
            user_embeds: User embeddings [n_users, emb_dim]
            item_embeds: Item base embeddings (ID) [n_items, emb_dim]
            item_multimodal_embeds: Item multimodal embeddings [n_items, emb_dim]
        
        Returns:
            modulated_item_embeds: Edge-level modulated item embeddings [n_items, emb_dim]
        """
        # Extract user-item edges from adjacency matrix
        adj_coo = adj.coalesce()
        edge_indices = adj_coo.indices()  # [2, num_edges]
        
        # Separate user and item indices
        user_indices = edge_indices[0]  # [num_edges]
        item_indices = edge_indices[1] - self.n_users  # [num_edges], adjust for item offset
        
        # Filter to only user->item edges (source is user, target is item)
        user_item_mask = (user_indices < self.n_users) & (item_indices >= 0) & (item_indices < self.n_items)
        user_indices = user_indices[user_item_mask]
        item_indices = item_indices[user_item_mask]
        
        if len(user_indices) == 0:
            return item_embeds
        
        # Get embeddings for each edge
        user_emb_selected = user_embeds[user_indices]  # [num_edges, emb_dim]
        item_emb_selected = item_embeds[item_indices]  # [num_edges, emb_dim]
        item_multimodal_selected = item_multimodal_embeds[item_indices]  # [num_edges, emb_dim]
        
        # Step 1: Compute edge-level affinity for each edge (u, i): affinity = u · i
        edge_affinity = (user_emb_selected * item_emb_selected).sum(dim=-1)  # [num_edges]
        
        # Step 2: Normalize affinity per user using softmax (vectorized version)
        # Use scatter_softmax for efficient computation
        try:
            from torch_scatter import scatter_softmax
            normalized_affinity = scatter_softmax(edge_affinity, user_indices, dim=0)
        except ImportError:
            # Fallback: use manual softmax with groupby (more efficient than original loop)
            # Sort by user_indices to group edges by user
            sorted_indices = torch.argsort(user_indices)
            sorted_user_indices = user_indices[sorted_indices]
            sorted_edge_affinity = edge_affinity[sorted_indices]
            
            # Find where each user's edges start
            unique_users, counts = torch.unique_consecutive(sorted_user_indices, return_counts=True)
            num_users = len(unique_users)
            
            # Compute softmax for each user group using chunked processing
            normalized_affinity_sorted = torch.zeros_like(sorted_edge_affinity)
            start_idx = 0
            for i in range(num_users):
                end_idx = start_idx + counts[i].item()
                normalized_affinity_sorted[start_idx:end_idx] = F.softmax(
                    sorted_edge_affinity[start_idx:end_idx], dim=0
                )
                start_idx = end_idx
            
            # Map back to original order
            reverse_indices = torch.argsort(sorted_indices)
            normalized_affinity = normalized_affinity_sorted[reverse_indices]
        
        # Step 3: Extract threshold value and comparison operator
        threshold_value = float(self.affinity_threshold[1:])
        compare_op = self.affinity_threshold[0]
        
        # Step 4: Apply threshold for each edge (edge-level judgment)
        if compare_op == '>':
            high_affinity_mask = normalized_affinity > threshold_value
        else:  # '<'
            high_affinity_mask = normalized_affinity < threshold_value
        
        # Step 5: For each edge: high affinity -> fuse, low affinity -> keep original
        modulated_item_emb_selected = torch.where(
            high_affinity_mask.unsqueeze(-1),
            (item_emb_selected + item_multimodal_selected) / 2,  # Fuse for high affinity edges
            item_emb_selected  # Keep original ID for low affinity edges
        )
        
        # Step 6: Re-compute affinity with modulated features if reattn is enabled
        if self.affinity_reattn:
            new_edge_affinity = (user_emb_selected * modulated_item_emb_selected).sum(dim=-1)
            
            try:
                from torch_scatter import scatter_softmax
                normalized_affinity = scatter_softmax(new_edge_affinity, user_indices, dim=0)
            except ImportError:
                # Fallback: same optimized approach as above
                sorted_indices = torch.argsort(user_indices)
                sorted_user_indices = user_indices[sorted_indices]
                sorted_new_affinity = new_edge_affinity[sorted_indices]
                
                unique_users, counts = torch.unique_consecutive(sorted_user_indices, return_counts=True)
                num_users = len(unique_users)
                
                normalized_affinity_sorted = torch.zeros_like(sorted_new_affinity)
                start_idx = 0
                for i in range(num_users):
                    end_idx = start_idx + counts[i].item()
                    normalized_affinity_sorted[start_idx:end_idx] = F.softmax(
                        sorted_new_affinity[start_idx:end_idx], dim=0
                    )
                    start_idx = end_idx
                
                reverse_indices = torch.argsort(sorted_indices)
                normalized_affinity = normalized_affinity_sorted[reverse_indices]
            
            if compare_op == '>':
                high_affinity_mask = normalized_affinity > threshold_value
            else:
                high_affinity_mask = normalized_affinity < threshold_value
            
            modulated_item_emb_selected = torch.where(
                high_affinity_mask.unsqueeze(-1),
                (item_emb_selected + item_multimodal_selected) / 2,
                item_emb_selected
            )
        
        # Step 7: Aggregate modulated features per item (vectorized version)
        # Use scatter_mean for efficient aggregation
        try:
            from torch_scatter import scatter_mean
            modulated_item_embeds = item_embeds.clone()
            # Aggregate using scatter_mean: faster than Python loops
            aggregated_embeds = scatter_mean(
                modulated_item_emb_selected, 
                item_indices, 
                dim=0, 
                dim_size=self.n_items
            )
            # Only update items that have edges
            item_has_edge = torch.zeros(self.n_items, dtype=torch.bool, device=item_indices.device)
            item_has_edge[item_indices] = True
            modulated_item_embeds[item_has_edge] = aggregated_embeds[item_has_edge]

            # Also aggregate affinity weights per item
            item_affinity = scatter_mean(
                normalized_affinity.unsqueeze(-1),
                item_indices,
                dim=0,
                dim_size=self.n_items
            ).squeeze(-1)
        except ImportError:
            # Fallback: use index_add_ for efficient aggregation (much faster than loops)
            modulated_item_embeds = item_embeds.clone()
            
            # Get unique items and their inverse indices
            unique_items, item_inverse, item_counts = torch.unique(
                item_indices, return_inverse=True, return_counts=True
            )
            
            # Aggregate by summing using index_add_ (vectorized operation)
            emb_dim = modulated_item_emb_selected.shape[1]
            item_sums = torch.zeros((len(unique_items), emb_dim), 
                                   dtype=modulated_item_emb_selected.dtype,
                                   device=modulated_item_emb_selected.device)
            item_sums.index_add_(0, item_inverse, modulated_item_emb_selected)
            
            # Divide by counts to get mean (vectorized)
            item_means = item_sums / item_counts.unsqueeze(1).float()
            
            # Update the embeddings for items that have edges
            modulated_item_embeds[unique_items] = item_means

            # Affinity aggregation fallback
            affinity_sums = torch.zeros(len(unique_items), device=item_indices.device, dtype=normalized_affinity.dtype)
            affinity_sums.index_add_(0, item_inverse, normalized_affinity)
            item_affinity = torch.zeros(self.n_items, device=item_indices.device, dtype=normalized_affinity.dtype)
            item_affinity[unique_items] = affinity_sums / item_counts.float()
        
        return modulated_item_embeds, item_affinity
    
    def conv_ui(self, adj, user_embeds, item_embeds, multimodal_user=None, multimodal_item=None, return_affinity=False):
        """
        User-Item graph convolution with edge-level affinity-based hard modulation.
        
        Args:
            adj: Adjacency matrix [n_users+n_items, n_users+n_items]
            user_embeds: User embeddings [n_users, emb_dim]
            item_embeds: Item ID embeddings [n_items, emb_dim]
            multimodal_user: Optional multimodal user embeddings (not used in edge-level modulation)
            multimodal_item: Item multimodal embeddings for hard modulation [n_items, emb_dim]
        
        Returns:
            Convolved embeddings
        """
        # Apply edge-level hard modulation if enabled
        item_affinity = None
        if self.use_affinity_modulation and multimodal_item is not None:
            # Modulate item embeddings based on edge-level affinity
            modulated_item_embeds, item_affinity = self.apply_edge_level_hard_modulation(
                adj, user_embeds, item_embeds, multimodal_item
            )
            ego_embeddings = torch.cat([user_embeds, modulated_item_embeds], dim=0)
        else:
            ego_embeddings = torch.cat([user_embeds, item_embeds], dim=0)
        
        all_embeddings = [ego_embeddings]
        
        for i in range(self.n_ui_layers):
            side_embeddings = torch.sparse.mm(adj, ego_embeddings)
            ego_embeddings = side_embeddings
            all_embeddings += [ego_embeddings]
        
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = all_embeddings.mean(dim=1, keepdim=False)
        
        if return_affinity:
            return all_embeddings, item_affinity
        return all_embeddings

    def conv_ii(self, ii_adj, single_modal):
        for i in range(self.n_layers):
            single_modal = torch.sparse.mm(ii_adj, single_modal)
        return single_modal

    def forward(self, adj, train=False):
        #  Encoding Multiple Modalities

        image_item_embeds = torch.multiply(self.item_id_embedding.weight, self.image_space_trans(self.image_embedding.weight))
        text_item_embeds = torch.multiply(self.item_id_embedding.weight, self.text_space_trans(self.text_embedding.weight))

        item_embeds = self.item_id_embedding.weight
        user_embeds = self.user_embedding.weight

        # Apply affinity-based hard modulation for ID embeddings
        # Compute multimodal user/item embeddings for modulation
        explicit_image_item = self.conv_ii(self.image_original_adj, image_item_embeds)
        explicit_image_user = torch.sparse.mm(self.R, explicit_image_item)
        
        explicit_text_item = self.conv_ii(self.text_original_adj, text_item_embeds)
        explicit_text_user = torch.sparse.mm(self.R, explicit_text_item)
        
        # Combine image and text for multimodal representation
        multimodal_user = (explicit_image_user + explicit_text_user) / 2
        multimodal_item = (explicit_image_item + explicit_text_item) / 2
        
        # Apply edge-level hard modulation in conv_ui
        # For ID embeddings: modulate items based on user-item edge affinity
        extended_id_embeds, item_affinity = self.conv_ui(
            adj, user_embeds, item_embeds,
            multimodal_user=None,  # Not used in edge-level modulation
            multimodal_item=multimodal_item if self.use_affinity_modulation else None,
            return_affinity=True
        )
        # 构造物品级亲和度门控，用户部分设为 1，避免破坏用户侧表征
        if item_affinity is None:
            item_affinity = torch.ones(self.n_items, device=adj.device, dtype=user_embeds.dtype)
        affinity_gate = torch.ones(self.n_users + self.n_items, 1, device=adj.device, dtype=user_embeds.dtype)
        affinity_gate[self.n_users:, 0] = item_affinity.clamp(min=0.0, max=1.0)
        
        explicit_image_embeds = torch.cat([explicit_image_user, explicit_image_item], dim=0)
        
        # Apply edge-level hard modulation for image modality
        # The base item embedding is explicit_image_item, modulate with multimodal_item
        extended_image_embeds = self.conv_ui(
            adj, explicit_image_user, explicit_image_item,
            multimodal_user=None,  # Not used in edge-level modulation
            multimodal_item=multimodal_item if self.use_affinity_modulation else None
        )

        explicit_text_embeds = torch.cat([explicit_text_user, explicit_text_item], dim=0)
        
        # Apply edge-level hard modulation for text modality
        # The base item embedding is explicit_text_item, modulate with multimodal_item
        extended_text_embeds = self.conv_ui(
            adj, explicit_text_user, explicit_text_item,
            multimodal_user=None,  # Not used in edge-level modulation
            multimodal_item=multimodal_item if self.use_affinity_modulation else None
        )

        extended_it_embeds = (extended_image_embeds + extended_text_embeds) / 2
        
        # Attributes Separation for Better Integration

        # ===== 亲和度引导的动态粗粒度生成 =====
        # Step 1: 计算基础粗粒度权重（原有方法，基于模态特征）
        base_img_scores = self.separate_coarse(explicit_image_embeds)  # [n_users+n_items, 1]
        base_txt_scores = self.separate_coarse(explicit_text_embeds)  # [n_users+n_items, 1]
        base_weights = self.softmax(torch.cat([base_img_scores, base_txt_scores], dim=-1))
        base_image_weights, base_text_weights = torch.split(base_weights, 1, dim=-1)
        
        # Step 2: 亲和度引导的动态权重调整（仅作用于物品部分）
        if self.use_affinity_modulation and item_affinity is not None:
            # 提取物品部分的嵌入和亲和度
            img_item_embeds = explicit_image_embeds[self.n_users:, :]  # [n_items, dim]
            txt_item_embeds = explicit_text_embeds[self.n_users:, :]  # [n_items, dim]
            item_affinity_expanded = item_affinity.unsqueeze(-1)  # [n_items, 1]
            
            # 将亲和度信息融入到权重计算中
            # 高亲和度 -> 期望权重更平衡（两个模态都重要）
            # 低亲和度 -> 保持原始权重分配（可能一个模态主导）
            
            # 方法1: 亲和度作为额外特征，直接参与权重网络决策
            img_input_with_affinity = torch.cat([img_item_embeds, item_affinity_expanded], dim=-1)  # [n_items, dim+1]
            txt_input_with_affinity = torch.cat([txt_item_embeds, item_affinity_expanded], dim=-1)  # [n_items, dim+1]
            
            # 计算亲和度引导的权重调节项
            affinity_img_adjust = self.affinity_guided_coarse(img_input_with_affinity)  # [n_items, 1]
            affinity_txt_adjust = self.affinity_guided_coarse(txt_input_with_affinity)  # [n_items, 1]
            affinity_adjust_weights = self.softmax(torch.cat([affinity_img_adjust, affinity_txt_adjust], dim=-1))
            affinity_item_img_w, affinity_item_txt_w = torch.split(affinity_adjust_weights, 1, dim=-1)
            
            # 方法2: 基于亲和度动态融合基础权重和亲和度权重
            # 高亲和度(接近1) -> 更依赖亲和度权重（更平衡）
            # 低亲和度(接近0) -> 更依赖基础权重（保持原始分配）
            # 使用sigmoid将亲和度映射到[0,1]作为混合系数
            affinity_mixing_coef = torch.sigmoid(item_affinity_expanded * 3.0)  # 放大亲和度差异，[n_items, 1]
            affinity_mixing_coef = affinity_mixing_coef * torch.clamp(self.affinity_coarse_alpha, 0.0, 1.0)
            
            # 混合基础权重和亲和度引导权重
            # # 对 affinity_mixing_coef 进行阈值处理
            # # 小于0.6的变成0，大于等于0.6的变成原值-0.5
            # affinity_mixing_coef = torch.where(
            #     affinity_mixing_coef < 0.6,
            #     torch.zeros_like(affinity_mixing_coef),  # 小于0.6的值设为0
            #     affinity_mixing_coef - 0.5  # 大于等于0.6的值减去0.5
            # )

            # mixed_item_img_w=base_image_weights[self.n_users:, :]+affinity_item_img_w * affinity_mixing_coef
            # mixed_item_txt_w = base_text_weights[self.n_users:, :]+affinity_item_txt_w * affinity_mixing_coef

            mixed_item_img_w = base_image_weights[self.n_users:, :] * (1 - affinity_mixing_coef) + \
                              affinity_item_img_w * affinity_mixing_coef
            mixed_item_txt_w = base_text_weights[self.n_users:, :] * (1 - affinity_mixing_coef) + \
                               affinity_item_txt_w * affinity_mixing_coef
            
            # 归一化确保权重和为1
            mixed_item_weights_sum = mixed_item_img_w + mixed_item_txt_w
            mixed_item_img_w = mixed_item_img_w / (mixed_item_weights_sum + 1e-8)
            mixed_item_txt_w = mixed_item_txt_w / (mixed_item_weights_sum + 1e-8)
            
            # 拼接用户和物品权重（用户部分使用基础权重）
            image_weights = torch.cat([
                base_image_weights[:self.n_users, :],  # 用户部分保持基础权重
                mixed_item_img_w  # 物品部分使用亲和度引导的动态权重
            ], dim=0)
            text_weights = torch.cat([
                base_text_weights[:self.n_users, :],  # 用户部分保持基础权重
                mixed_item_txt_w  # 物品部分使用亲和度引导的动态权重
            ], dim=0)
        else:
            # 未启用亲和度调制时，使用基础权重
            image_weights = base_image_weights
            text_weights = base_text_weights

        
        # Step 3: 计算亲和度引导的动态粗粒度嵌入
        coarse_grained_embeds = image_weights * explicit_image_embeds + text_weights * explicit_text_embeds
                
        # ===== 残差增强的解耦合机制 =====
        # 1. 计算细粒度残差（模态与粗粒度的差异）
        img_residual_global = explicit_image_embeds - coarse_grained_embeds  # [n_users+n_items, dim]
        txt_residual_global = explicit_text_embeds - coarse_grained_embeds  # [n_users+n_items, dim]
        
        # 2. 应用行为网络提取细粒度特征（原有逻辑）
        fine_grained_image_base = torch.multiply(
            self.image_behavior(extended_id_embeds),
            img_residual_global  
        )
        fine_grained_text_base = torch.multiply(
            self.text_behavior(extended_id_embeds),
            txt_residual_global  
        )
        
        # 3. 残差增强：添加残差连接，使训练更稳定
        # 方法：细粒度特征 = 基础细粒度 + α * 原始残差
        # 这样即使行为网络学习失败，也能保留原始残差信息
        fine_grained_image = fine_grained_image_base + self.residual_alpha * img_residual_global
        fine_grained_text = fine_grained_text_base + self.residual_alpha * txt_residual_global
        
        # # 4. 多级残差融合：进一步稳定集成过程
        # # 第一级：细粒度特征融合
        # fine_grained_combined = (fine_grained_image + fine_grained_text) / 2
        
        # # 第二级：细粒度 + 粗粒度融合（带残差连接）
        # # 使用加权融合，粗粒度作为基础，细粒度作为增强
        # # 残差连接确保即使细粒度学习失败，也能保留粗粒度信息
        # integration_embeds = coarse_grained_embeds + self.fine_grained_weight * fine_grained_combined
        
        # # 5. 最终集成：ID嵌入 + 多模态集成嵌入（带残差连接）
        # # 添加残差连接，确保即使集成失败也能保留ID信息
        integration_embeds = (fine_grained_image_base + fine_grained_text_base + coarse_grained_embeds) / 3
        all_embeds = extended_id_embeds + integration_embeds

        if train:
            return all_embeds, (integration_embeds, extended_id_embeds, extended_it_embeds), (explicit_image_embeds, explicit_text_embeds), (extended_image_embeds, extended_text_embeds)

        return all_embeds


    def sq_sum(self, emb):
        return 1. / 2 * (emb ** 2).sum()
    
    def adaptive_optimization(self, user_e, pos_e, neg_e):
        """
        自适应优化：基于模态重要性动态调整不同特征维度的权重
        参考Cohesion的Adaptive Optimization方法，适配GUME的多模态场景
        
        Args:
            user_e: 用户嵌入 [batch_size, embedding_dim]
            pos_e: 正样本物品嵌入 [batch_size, embedding_dim]
            neg_e: 负样本物品嵌入 [batch_size, embedding_dim]
        
        Returns:
            adaptive_weight: 自适应权重 [batch_size, embedding_dim]
        """
        # 计算每个维度的得分差异
        pos_scores_per_dim = torch.mul(user_e, pos_e)  # [batch_size, embedding_dim]
        neg_scores_per_dim = torch.mul(user_e, neg_e)  # [batch_size, embedding_dim]
        score_diff = pos_scores_per_dim - neg_scores_per_dim  # [batch_size, embedding_dim]
        
        # 对每个样本的维度差异进行softmax归一化，得到维度重要性分布
        # 差异越大 -> softmax值越大 -> 该维度越重要
        dim_importance = F.softmax(score_diff, dim=-1)  # [batch_size, embedding_dim]
        
        # 自适应权重：重要性低的维度（得分差异小）需要更多关注
        # 使用 1 - importance 来强调被忽略的维度，同时保留重要维度
        adaptive_weight = 1.0 - dim_importance.detach()  # detach避免梯度传播
        
        # 添加平滑项，避免权重过小导致训练不稳定
        # 使用配置的平滑系数，映射到合理范围
        smooth_coef = self.adaptive_weight_smooth
        adaptive_weight = adaptive_weight * smooth_coef + (1.0 - smooth_coef)  # 映射到[1-smooth_coef, 1.0]范围
        
        return adaptive_weight
    
    def bpr_loss(self, users, pos_items, neg_items, use_adaptive=None):
        """
        BPR损失计算，支持自适应权重优化
        
        Args:
            users: 用户嵌入 [batch_size, embedding_dim]
            pos_items: 正样本物品嵌入 [batch_size, embedding_dim]
            neg_items: 负样本物品嵌入 [batch_size, embedding_dim]
            use_adaptive: 是否使用自适应权重优化
        """
        if use_adaptive:
            # 计算自适应权重
            adaptive_weight = self.adaptive_optimization(users, pos_items, neg_items)  # [batch_size, embedding_dim]
            
            # 使用自适应权重调整嵌入
            users_weighted = users * adaptive_weight
            pos_items_weighted = pos_items * adaptive_weight
            neg_items_weighted = neg_items * adaptive_weight
            
            # 计算加权后的得分
            pos_scores = torch.sum(torch.mul(users_weighted, pos_items_weighted), dim=1)
            neg_scores = torch.sum(torch.mul(users_weighted, neg_items_weighted), dim=1)
        else:
            # 原始BPR计算
            pos_scores = torch.sum(torch.mul(users, pos_items), dim=1)
            neg_scores = torch.sum(torch.mul(users, neg_items), dim=1)

        regularizer = (self.sq_sum(users) + self.sq_sum(pos_items) + self.sq_sum(neg_items)) / self.batch_size

        maxi = F.logsigmoid(pos_scores - neg_scores)
        mf_loss = -torch.mean(maxi)

        reg_loss = self.reg_weight_1 * regularizer

        return mf_loss, reg_loss

    def InfoNCE(self, view1, view2, temperature):
        view1, view2 = F.normalize(view1, dim=1), F.normalize(view2, dim=1)
        pos_score = (view1 * view2).sum(dim=-1)
        pos_score = torch.exp(pos_score / temperature)
        ttl_score = torch.matmul(view1, view2.transpose(0, 1))
        ttl_score = torch.exp(ttl_score / temperature).sum(dim=1)
        cl_loss = -torch.log(pos_score / ttl_score)
        
        return torch.mean(cl_loss)

    def calculate_loss(self, interaction):
        users = interaction[0]
        pos_items = interaction[1]
        neg_items = interaction[2]

        embeds_1, embeds_2, embeds_3,embeds_4= self.forward(self.norm_adj, train=True)
        users_embeddings, items_embeddings = torch.split(embeds_1, [self.n_users, self.n_items], dim=0)
        
        integration_embeds, extended_id_embeds, extended_it_embeds = embeds_2
        explicit_image_embeds, explicit_text_embeds = embeds_3
        extended_image_embeds, extended_text_embeds = embeds_4

        u_g_embeddings = users_embeddings[users]
        pos_i_g_embeddings = items_embeddings[pos_items]
        neg_i_g_embeddings = items_embeddings[neg_items]

        vt_loss = self.vt_loss * self.align_vt(explicit_image_embeds, explicit_text_embeds)

        #-----------------------------------------------------------
        # vt_loss= vt_loss+self.vt_loss*self.align_vt(extended_image_embeds,extended_text_embeds)
        #-----------------------------------------------------------

        integration_users, integration_items = torch.split(integration_embeds, [self.n_users, self.n_items], dim=0)
        extended_id_user, extended_id_items = torch.split(extended_id_embeds, [self.n_users, self.n_items], dim=0)
        extended_it_user, extended_it_items = torch.split(extended_it_embeds, [self.n_users, self.n_items], dim=0)

        # 使用自适应BPR优化（如果启用）
        bpr_loss, reg_loss_1 = self.bpr_loss(
            u_g_embeddings, 
            pos_i_g_embeddings, 
            neg_i_g_embeddings,
            use_adaptive=self.use_adaptive_bpr
        )
        
        # bm_loss = self.bm_loss * (self.InfoNCE(integration_users[users], extended_id_user[users], self.bm_temp) + self.InfoNCE(integration_items[pos_items], extended_id_items[pos_items], self.bm_temp))
        bm_loss_1 = (self.InfoNCE(integration_users[users], extended_id_user[users], self.bm_temp) + 
             self.InfoNCE(integration_items[pos_items], extended_id_items[pos_items], self.bm_temp))
        bm_loss_2 = (self.InfoNCE(extended_it_user[users], extended_id_user[users], self.bm_temp) +  # 先修正extended_it_embeds的索引（你原来写的extended_it_embeds[users]是错的，应该是extended_it_user[users]）
             self.InfoNCE(extended_it_items[pos_items], extended_id_items[pos_items], self.bm_temp))
        # # 核心调整：bm_loss_2乘以小权重，避免总强度翻倍
        bm_loss = self.bm_loss * (bm_loss_1 + bm_loss_2)
        # bm_loss=self.bm_loss*bm_loss_1
        al_loss = vt_loss + bm_loss
        
        # extended_it_user, extended_it_items = torch.split(extended_it_embeds, [self.n_users, self.n_items], dim=0)

        # Enhancing User Modality Representation
        c_loss = self.InfoNCE(extended_it_user[users], integration_users[users], self.um_temp)
        noise_loss_1 = self.cal_noise_loss(users, integration_users, self.um_temp)
        noise_loss_2 = self.cal_noise_loss(users, extended_it_user, self.um_temp)
        um_loss = self.um_loss * (c_loss + noise_loss_1 + noise_loss_2)
        
        reg_loss_2 = self.reg_weight_2 * self.sq_sum(extended_it_items[pos_items]) / self.batch_size
        reg_loss = reg_loss_1 + reg_loss_2
        
        return bpr_loss + al_loss + um_loss + reg_loss

    
    
    def cal_noise_loss(self, id, emb, temp):

        def add_perturbation(x):
            random_noise = torch.rand_like(x).to(self.device)
            x = x + torch.sign(x) * F.normalize(random_noise, dim=-1) * 0.1
            return x

        emb_view1 = add_perturbation(emb)
        emb_view2 = add_perturbation(emb)
        emb_loss = self.InfoNCE(emb_view1[id], emb_view2[id], temp)

        return emb_loss
    
    def align_vt(self,embed1, embed2):
        emb1_var, emb1_mean = torch.var(embed1), torch.mean(embed1)
        emb2_var, emb2_mean = torch.var(embed2), torch.mean(embed2)
        
        vt_loss = (torch.abs(emb1_var - emb2_var) + torch.abs(emb1_mean - emb2_mean)).mean()
        
        return vt_loss
    
    def full_sort_predict(self, interaction):
        user = interaction[0]

        all_embeds = self.forward(self.norm_adj)
        restore_user_e, restore_item_e = torch.split(all_embeds, [self.n_users, self.n_items], dim=0)
        u_embeddings = restore_user_e[user]

        scores = torch.matmul(u_embeddings, restore_item_e.transpose(0, 1))
        return scores


