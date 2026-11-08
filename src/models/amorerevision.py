"""Reviewer experiments on the repaired ORIGINAL AMORE, not GUME-Ego/SG/AMOREX.

Reference forward is AMORECLEAN with full original auxiliary views. Independent
graph and fusion switches isolate reuse of the original item-pooled affinity.
"""
import torch
from models.amoreclean import AMORECLEAN


class AMOREREVISION(AMORECLEAN):
    def __init__(self, config, dataset, local_time=None):
        # Keep affinity computation available even when only the fusion branch uses it.
        config['use_affinity_modulation'] = True
        config['coarse_affinity'] = bool(config['revision_fusion'])
        super().__init__(config, dataset, local_time)
        self.graph_affinity = bool(config['revision_graph'])
        self.prior_control = config['revision_prior']
        if self.prior_control not in ('shared', 'shuffle', 'constant', 'detach'):
            raise ValueError('Unknown revision_prior')
        generator = torch.Generator().manual_seed(int(config['seed']) + 8191)
        self.register_buffer('prior_permutation', torch.randperm(self.n_items, generator=generator), persistent=False)

    def conv_ui(self, adj, user_embeds, item_embeds, multimodal_user=None,
                multimodal_item=None, return_affinity=False):
        affinity = None
        initial_items = item_embeds
        if multimodal_item is not None and (self.graph_affinity or return_affinity):
            modulated, affinity = self.apply_edge_level_hard_modulation(
                adj, user_embeds, item_embeds, multimodal_item)
            if self.graph_affinity:
                initial_items = modulated
        current = torch.cat([user_embeds, initial_items])
        layers = [current]
        for _ in range(self.n_ui_layers):
            current = torch.sparse.mm(adj, current)
            layers.append(current)
        output = torch.stack(layers, dim=1).mean(dim=1)
        if return_affinity:
            if self.prior_control == 'shuffle':
                affinity = affinity[self.prior_permutation]
            elif self.prior_control == 'constant':
                affinity = affinity.mean().expand_as(affinity)
            elif self.prior_control == 'detach':
                affinity = affinity.detach()
            if not self.training:
                self.diagnostics['prior_mean'] = float(affinity.mean())
                self.diagnostics['prior_std'] = float(affinity.std(unbiased=False))
            return output, affinity
        return output
