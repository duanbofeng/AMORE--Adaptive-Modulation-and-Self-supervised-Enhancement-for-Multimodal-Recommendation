"""Conservative repair/pruning of the original full AMORE, not AMOREX.

Retains original feature learning, signed self-KNN, enhanced UI graph, hard item
modulation and coarse/fine fusion. Auxiliary objectives have separate explicit
weights. Removing extended views skips their propagation entirely.
"""
import torch
from torch.nn import functional as F

from models.amorelegacy import AMORELEGACY
from utils.original_amore_ops import unique_infonce


class AMORECLEAN(AMORELEGACY):
    loss_names = ('bpr', 'embedding_l2', 'vt_moments', 'bm', 'bm_extended',
                  'um_cross', 'um_noise', 'extended_l2')

    def __init__(self, config, dataset, local_time=None):
        super().__init__(config, dataset, local_time)
        if self.use_adaptive_bpr:
            raise ValueError('Corrected AMORE uses the same unweighted score for BPR and inference')
        for name in ['reg_weight_1', 'reg_weight_2', 'bm_loss', 'bm_extended_loss',
                     'um_loss', 'noise_loss', 'vt_loss']:
            value = config[name]
            if value is None or not 0 <= float(value) < float('inf'):
                raise ValueError(name + ' must be finite and nonnegative')
        if self.bm_temp <= 0 or self.um_temp <= 0:
            raise ValueError('Contrastive temperatures must be positive')
        self.use_extended = bool(config['bm_extended_loss'] or self.um_loss or config['noise_loss'] or self.reg_weight_2)
        self.use_coarse_affinity = bool(config['coarse_affinity'] and self.use_affinity_modulation)
        # Original forward never references these two tables. Initialize then
        # discard to preserve all common parameters' initialization RNG.
        del self.extended_image_user, self.extended_text_user
        if not self.use_coarse_affinity:
            del self.affinity_guided_coarse, self.affinity_coarse_alpha

    def forward(self, adj=None, train=False):
        adj = self.norm_adj if adj is None else adj
        image_item = self.conv_ii(self.image_original_adj,
                                  self.item_id_embedding.weight * self.image_space_trans(self.image_embedding.weight))
        text_item = self.conv_ii(self.text_original_adj,
                                 self.item_id_embedding.weight * self.text_space_trans(self.text_embedding.weight))
        image_user = torch.sparse.mm(self.R, image_item)
        text_user = torch.sparse.mm(self.R, text_item)
        multimodal_item = (image_item + text_item) / 2 if self.use_affinity_modulation else None
        collaborative, affinity = self.conv_ui(adj, self.user_embedding.weight, self.item_id_embedding.weight,
                                                multimodal_item=multimodal_item, return_affinity=True)
        image = torch.cat([image_user, image_item])
        text = torch.cat([text_user, text_item])
        weights = torch.cat([self.separate_coarse(image), self.separate_coarse(text)], dim=1).softmax(1)
        if self.use_coarse_affinity:
            affinity = affinity[:, None]
            adjustment = torch.cat([
                self.affinity_guided_coarse(torch.cat([image_item, affinity], dim=1)),
                self.affinity_guided_coarse(torch.cat([text_item, affinity], dim=1))], dim=1).softmax(1)
            mixing = (3 * affinity).sigmoid() * self.affinity_coarse_alpha.clamp(0, 1)
            item_weights = weights[self.n_users:] * (1 - mixing) + adjustment * mixing
            item_weights = item_weights / (item_weights.sum(1, keepdim=True) + 1e-8)
            weights = torch.cat([weights[:self.n_users], item_weights])
        coarse = weights[:, :1] * image + weights[:, 1:] * text
        fine_image = self.image_behavior(collaborative) * (image - coarse)
        fine_text = self.text_behavior(collaborative) * (text - coarse)
        integration = (coarse + fine_image + fine_text) / 3
        output = collaborative + integration
        extended_image = extended_text = extended = None
        if train and self.use_extended:
            extended_image = self.conv_ui(adj, image_user, image_item, multimodal_item=multimodal_item)
            extended_text = self.conv_ui(adj, text_user, text_item, multimodal_item=multimodal_item)
            extended = (extended_image + extended_text) / 2
        if not self.training:
            self.diagnostics['semantic_to_id_norm'] = float((integration.norm(dim=1) / collaborative.norm(dim=1).clamp_min(1e-8)).mean())
        if train:
            return output, (integration, collaborative, extended), (image, text), (extended_image, extended_text)
        return output

    @staticmethod
    def moment_alignment(first, second):
        # Explicit scalar population-moment penalty, NOT a distribution distance
        # or sample-wise cross-modal alignment. No unsupported Gaussian claim.
        return ((first.mean() - second.mean()).abs()
                + (first.var(unbiased=False) - second.var(unbiased=False)).abs())

    def _noise_contrast(self, embeddings, users):
        ids = users.unique()
        view = embeddings[ids]
        first = view + view.sign() * F.normalize(torch.rand_like(view), dim=1) * 0.1
        second = view + view.sign() * F.normalize(torch.rand_like(view), dim=1) * 0.1
        return unique_infonce(first, second, torch.arange(len(ids), device=users.device), self.um_temp)

    def calculate_loss(self, interaction):
        users, positive, negative = interaction[:3]
        output, (integration, collaborative, extended), (image, text), _ = self.forward(train=True)
        user_vectors, item_vectors = output.split([self.n_users, self.n_items])
        u, p, n = user_vectors[users], item_vectors[positive], item_vectors[negative]
        bpr = F.softplus(-(u * (p - n)).sum(1)).mean()
        # Retain original representation regularization, with the ACTUAL batch
        # denominator. This is not advertised as L2 over all model parameters.
        regularizer = self.reg_weight_1 * (u.square().sum() + p.square().sum() + n.square().sum()) / (2 * len(users))
        zero = bpr.new_zeros(())
        vt = self.vt_loss * self.moment_alignment(image, text) if self.vt_loss else zero
        item_ids = positive + self.n_users
        bm = self.bm_loss * (unique_infonce(integration, collaborative, users, self.bm_temp)
                             + unique_infonce(integration, collaborative, item_ids, self.bm_temp)) if self.bm_loss else zero
        bm_extended = um = noise = extended_reg = zero
        if self.config['bm_extended_loss']:
            bm_extended = self.config['bm_extended_loss'] * (
                unique_infonce(extended, collaborative, users, self.bm_temp)
                + unique_infonce(extended, collaborative, item_ids, self.bm_temp))
        if self.um_loss:
            um = self.um_loss * unique_infonce(extended, integration, users, self.um_temp)
        if self.config['noise_loss']:
            noise = self.config['noise_loss'] * (self._noise_contrast(integration, users)
                                                + self._noise_contrast(extended, users))
        if self.reg_weight_2:
            extended_reg = self.reg_weight_2 * extended[item_ids].square().sum() / (2 * len(users))
        return bpr, regularizer, vt, bm, bm_extended, um, noise, extended_reg
