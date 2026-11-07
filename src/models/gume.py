import os
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

from common.abstract_recommender import GeneralRecommender
from utils.utils import build_sim, compute_normalized_laplacian, build_knn_neighbourhood, build_knn_normalized_graph
from utils.graph_ops import knn_graph, normalize, as_torch, as_scipy
from collections import defaultdict
import math
from scipy.sparse import lil_matrix
import random
import json

class GUME(GeneralRecommender):
    def __init__(self, config, dataset, local_time):
        super(GUME, self).__init__(config, dataset)
        self.config = config
        self.sparse = True
        self.use_extended = bool(config['um_loss'] or config['reg_weight_2'])
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

        # load dataset info
        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_id_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_id_embedding.weight)
        
        if self.use_extended:
            self.extended_image_user = nn.Embedding(self.n_users, self.embedding_dim)
            nn.init.xavier_uniform_(self.extended_image_user.weight)
            self.extended_text_user = nn.Embedding(self.n_users, self.embedding_dim)
            nn.init.xavier_uniform_(self.extended_text_user.weight)

        self.dataset_path = os.path.join(config['data_path'], config['dataset'])
        self.data_name = config['dataset']

        if self.v_feat is None or self.t_feat is None:
            raise ValueError('GUME requires image and text features')
        self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=bool(config['freeze_features']))
        self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=bool(config['freeze_features']))
        cache = os.path.join(self.dataset_path, 'graph_cache')
        chunk = config['graph_chunk_size'] or 512
        for name, feature in [('image', self.v_feat), ('text', self.t_feat)]:
            graph = knn_graph(feature, self.knn_k, chunk, self_loops=True, cache_dir=cache)
            self.register_buffer(name + '_original_adj', as_torch(normalize(graph), self.device), persistent=False)

        #  Enhancing User-Item Graph
        self.inter = self.find_inter(self.image_original_adj, self.text_original_adj)
        self.ii_adj = self.add_edge(self.inter) if config['enhance_graph'] else sp.csr_matrix((self.n_items, self.n_items))
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
        
        self.softmax = nn.Softmax(dim=-1)
                
        self.image_behavior = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.Sigmoid()
        )
        self.text_behavior = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.Sigmoid()
        )

        if config['fusion'] not in ('gume', 'mean', 'coarse'):
            raise ValueError('Unknown GUME fusion mode')
        if config['fusion'] == 'mean':
            del self.separate_coarse
        if config['fusion'] in ('mean', 'coarse'):
            del self.image_behavior, self.text_behavior
        self.tau = 0.5
    
    def find_inter(self, image_adj, text_adj):
        # Derive the intersection from actual row indices; no hard-coded k or stale JSON cache.
        image, text = as_scipy(image_adj), as_scipy(text_adj)
        image.data[:] = 1
        text.data[:] = 1
        common = image.multiply(text).tocsr()
        common.setdiag(0)
        common.eliminate_zeros()
        return {i: common.indices[common.indptr[i]:common.indptr[i + 1]].tolist()
                for i in range(self.n_items)}

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

            d_inv = np.zeros_like(rowsum)
            np.power(rowsum, -0.5, out=d_inv, where=rowsum > 0)
            d_inv = d_inv.flatten()
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
        return torch.sparse_coo_tensor(indices, values, shape).coalesce()
    
    def conv_ui(self, adj, user_embeds, item_embeds):
        ego_embeddings = torch.cat([user_embeds, item_embeds], dim=0)
        all_embeddings = [ego_embeddings]
        
        for i in range(self.n_ui_layers):
            side_embeddings = torch.sparse.mm(adj, ego_embeddings)
            ego_embeddings = side_embeddings
            all_embeddings += [ego_embeddings]
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = all_embeddings.mean(dim=1, keepdim=False)
        
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

        extended_id_embeds = self.conv_ui(adj, user_embeds, item_embeds)
        
        explicit_image_item = self.conv_ii(self.image_original_adj, image_item_embeds)
        explicit_image_user = torch.sparse.mm(self.R, explicit_image_item)
        explicit_image_embeds = torch.cat([explicit_image_user, explicit_image_item], dim=0)
        
        extended_image_embeds = self.conv_ui(adj, self.extended_image_user.weight, explicit_image_item) if self.use_extended else None

        explicit_text_item = self.conv_ii(self.text_original_adj, text_item_embeds)
        explicit_text_user = torch.sparse.mm(self.R, explicit_text_item)
        explicit_text_embeds = torch.cat([explicit_text_user, explicit_text_item], dim=0)
        
        extended_text_embeds = self.conv_ui(adj, self.extended_text_user.weight, explicit_text_item) if self.use_extended else None

        extended_it_embeds = (extended_image_embeds + extended_text_embeds) / 2 if self.use_extended else None
        
        # Keep the same 1/3 semantic scale in all fusion ablations.
        if self.config['fusion'] == 'mean':
            integration_embeds = (explicit_image_embeds + explicit_text_embeds) / 6
        else:
            image_weights, text_weights = torch.split(self.softmax(torch.cat([
                self.separate_coarse(explicit_image_embeds), self.separate_coarse(explicit_text_embeds)
            ], dim=-1)), 1, dim=-1)
            coarse = image_weights * explicit_image_embeds + text_weights * explicit_text_embeds
            if self.config['fusion'] == 'coarse':
                integration_embeds = coarse / 3
            else:
                fine_image = self.image_behavior(extended_id_embeds) * (explicit_image_embeds - coarse)
                fine_text = self.text_behavior(extended_id_embeds) * (explicit_text_embeds - coarse)
                integration_embeds = (fine_image + fine_text + coarse) / 3

        all_embeds = extended_id_embeds + integration_embeds

        if train:
            return all_embeds, (integration_embeds, extended_id_embeds, extended_it_embeds), (explicit_image_embeds, explicit_text_embeds)

        return all_embeds

    def sq_sum(self, emb):
        return 1. / 2 * (emb ** 2).sum()
    
    def bpr_loss(self, users, pos_items, neg_items):
        pos_scores = torch.sum(torch.mul(users, pos_items), dim=1)
        neg_scores = torch.sum(torch.mul(users, neg_items), dim=1)

        regularizer = (self.sq_sum(users) + self.sq_sum(pos_items) + self.sq_sum(neg_items)) / self.batch_size

        maxi = F.logsigmoid(pos_scores - neg_scores)
        mf_loss = -torch.mean(maxi)

        reg_loss = self.reg_weight_1 * regularizer

        return mf_loss, reg_loss

    def InfoNCE(self, view1, view2, temperature):
        view1, view2 = F.normalize(view1, dim=1), F.normalize(view2, dim=1)
        logits = view1 @ view2.T / temperature
        return F.cross_entropy(logits, torch.arange(len(view1), device=view1.device))

    def calculate_loss(self, interaction):
        users = interaction[0]
        pos_items = interaction[1]
        neg_items = interaction[2]

        embeds_1, embeds_2, embeds_3 = self.forward(self.norm_adj, train=True)
        users_embeddings, items_embeddings = torch.split(embeds_1, [self.n_users, self.n_items], dim=0)
        
        integration_embeds, extended_id_embeds, extended_it_embeds = embeds_2
        explicit_image_embeds, explicit_text_embeds = embeds_3

        u_g_embeddings = users_embeddings[users]
        pos_i_g_embeddings = items_embeddings[pos_items]
        neg_i_g_embeddings = items_embeddings[neg_items]

        vt_loss = self.vt_loss * self.align_vt(explicit_image_embeds, explicit_text_embeds)
        
        integration_users, integration_items = torch.split(integration_embeds, [self.n_users, self.n_items], dim=0)
        extended_id_user, extended_id_items = torch.split(extended_id_embeds, [self.n_users, self.n_items], dim=0)
        bpr_loss, reg_loss_1 = self.bpr_loss(u_g_embeddings, pos_i_g_embeddings,neg_i_g_embeddings)
        
        bm_loss = self.bm_loss * (self.InfoNCE(integration_users[users], extended_id_user[users], self.bm_temp) + self.InfoNCE(integration_items[pos_items], extended_id_items[pos_items], self.bm_temp)) if self.bm_loss else bpr_loss.new_zeros(())
        
        al_loss = vt_loss + bm_loss
        
        um_loss = bpr_loss.new_zeros(())
        reg_loss_2 = bpr_loss.new_zeros(())
        if self.use_extended:
            extended_it_user, extended_it_items = torch.split(extended_it_embeds, [self.n_users, self.n_items], dim=0)
            if self.um_loss:
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
    
    def prepare_evaluation(self):
        self._eval_embeddings = self.forward(self.norm_adj)

    def clear_evaluation(self):
        self._eval_embeddings = None

    def full_sort_predict(self, interaction):
        user = interaction[0]

        all_embeds = getattr(self, '_eval_embeddings', None)
        if all_embeds is None:
            all_embeds = self.forward(self.norm_adj)
        restore_user_e, restore_item_e = torch.split(all_embeds, [self.n_users, self.n_items], dim=0)
        u_embeddings = restore_user_e[user]

        scores = torch.matmul(u_embeddings, restore_item_e.transpose(0, 1))
        return scores
