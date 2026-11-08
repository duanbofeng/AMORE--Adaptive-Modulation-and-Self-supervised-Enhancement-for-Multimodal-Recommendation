import hashlib
from pathlib import Path
import sys
import unittest
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import test_research as research
from utils.revision_protocol import perturb_training
from utils.dataset import RecDataset
from utils.configurator import Config
from utils.benchmark import benchmark_ranking
from utils.dataloader import EvalDataLoader


class RevisionTests(unittest.TestCase):
    def setUp(self):
        self.fixture = research.ResearchTests()
        self.fixture.setUp()

    def tearDown(self):
        self.fixture.tearDown()

    def test_original_archive_is_the_manifest_snapshot(self):
        import json
        root = Path(__file__).resolve().parents[1]
        manifest = json.loads((root/'src/models/archive/manifest.json').read_text())
        digest = hashlib.sha256((root/'src/models/archive/amore_original.py').read_bytes()).hexdigest()
        self.assertEqual(digest, manifest['old_model/src/models/gume.py'])

    def test_revision_11_matches_fixed_full_forward_loss_and_gradients(self):
        fixed, loader = self.fixture.make_model('amoreclean', {'coarse_affinity': True, 'bm_extended_loss': .005,
                                                            'um_loss': .1, 'noise_loss': .1, 'reg_weight_2': .1})
        revision, _ = self.fixture.make_model('amorerevision', {})
        batch = next(iter(loader))
        for model in (fixed, revision):
            torch.manual_seed(32)
            losses = model.calculate_loss(batch)
            sum(losses).backward()
        torch.testing.assert_close(fixed.forward(), revision.forward(), atol=0, rtol=0)
        other = dict(revision.named_parameters())
        for name, param in fixed.named_parameters():
            if param.grad is not None:
                torch.testing.assert_close(param.grad, other[name].grad, atol=1e-7, rtol=1e-5)

    def test_fusion_only_has_prior_but_unmodulated_propagation(self):
        m, _ = self.fixture.make_model('amorerevision', {'revision_graph': False, 'revision_fusion': True})
        u, i = m.user_embedding.weight, m.item_id_embedding.weight
        output, prior = m.conv_ui(m.norm_adj, u, i, multimodal_item=i * 2, return_affinity=True)
        current = torch.cat([u, i]); states = [current]
        for _ in range(m.n_ui_layers):
            current = torch.sparse.mm(m.norm_adj, current); states.append(current)
        torch.testing.assert_close(output, torch.stack(states, 1).mean(1))
        self.assertEqual(prior.shape, (m.n_items,))
        self.assertTrue(m.use_coarse_affinity)

    def test_prior_shuffle_preserves_graph_output_and_prior_distribution(self):
        real, _ = self.fixture.make_model('amorerevision', {})
        shuffled, _ = self.fixture.make_model('amorerevision', {'revision_prior': 'shuffle'})
        values = []
        for m in (real, shuffled):
            values.append(m.conv_ui(m.norm_adj, m.user_embedding.weight, m.item_id_embedding.weight,
                                    multimodal_item=m.item_id_embedding.weight * 2, return_affinity=True))
        torch.testing.assert_close(values[0][0], values[1][0], rtol=0, atol=0)
        torch.testing.assert_close(values[0][1].sort().values, values[1][1].sort().values)

    def test_noise_never_injects_heldout_or_duplicate_pairs_and_is_seed_matched(self):
        cfg = Config('amorerevision', 'baby', dict(self.fixture.base, interaction_noise_rate=.5))
        train, valid, test = RecDataset(cfg).split()
        first, a = perturb_training(train, valid, test, cfg)
        cfg['seed'] = 1234
        second, b = perturb_training(train, valid, test, cfg)
        self.assertEqual(a, b)
        self.assertEqual(len(first), len(train) + 6)
        pairs = set(zip(first.df[first.uid_field], first.df[first.iid_field]))
        self.assertEqual(len(pairs), len(first))
        for held in (valid, test):
            self.assertFalse(pairs & set(zip(held.df[held.uid_field], held.df[held.iid_field])))

    def test_cold_items_have_no_training_interactions(self):
        cfg = Config('amorerevision', 'baby', dict(self.fixture.base, cold_item_fraction=.5))
        train, valid, test = RecDataset(cfg).split()
        transformed, info = perturb_training(train, valid, test, cfg)
        self.assertTrue(info['cold_items'])
        self.assertFalse(set(transformed.df[transformed.iid_field]) & set(info['cold_items']))
        self.assertEqual(transformed.get_item_num(), train.get_item_num())

    def test_benchmark_reports_encoding_and_ranking_separately(self):
        model, _ = self.fixture.make_model('amorerevision', {})
        cfg = model.config
        train, valid, test = RecDataset(cfg).split()
        loader = EvalDataLoader(cfg, test, additional_dataset=train, batch_size=2)
        report = benchmark_ranking(model, loader, cfg['device'], repeats=2)
        self.assertEqual(report['repeats'], 2)
        self.assertGreater(report['users_per_second_median'], 0)

    def test_mentor_forward_does_not_register_checkpoint_aliases(self):
        model, loader = self.fixture.make_model('mm_mentor', {'embedding_size': 64, 'feat_embed_dim': 64})
        before = set(model.state_dict())
        model.pre_epoch_processing()
        model.calculate_loss(next(iter(loader))).backward()
        self.assertEqual(before, set(model.state_dict()))
        restored, _ = self.fixture.make_model('mm_mentor', {'embedding_size': 64, 'feat_embed_dim': 64})
        restored.load_state_dict(model.state_dict())

    def test_fixed_feature_corruption_is_independent_of_optimization_seed(self):
        args = {'feature_corruption': 'shuffle', 'feature_corruption_rate': 1.,
                'corruption_modality': 'image', 'feature_corruption_seed': 314159}
        first, _ = self.fixture.make_model('amorerevision', dict(args, seed=999))
        second, _ = self.fixture.make_model('amorerevision', dict(args, seed=2024))
        torch.testing.assert_close(first.v_feat, second.v_feat, rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
