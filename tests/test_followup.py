"""Checks for frozen probes and the objective-by-affinity experimental design."""
from pathlib import Path
import sys
import unittest
import torch
sys.path.insert(0, str(Path(__file__).parent))
import test_research as research
from probe_affinity import counterfactuals
from revision_suite import stage_jobs


class FollowupTests(unittest.TestCase):
    def setUp(self):
        self.fixture = research.ResearchTests(); self.fixture.setUp()

    def tearDown(self):
        self.fixture.tearDown()

    def test_probe_preserves_weights_and_flags_and_repeats_exactly_on_cpu(self):
        model, loader = self.fixture.make_model('amorerevision', {})
        model.eval()
        before = {k: v.clone() for k, v in model.state_dict().items()}
        with torch.no_grad():
            report = counterfactuals(model, torch.tensor([0, 1]), loader.history_items_per_u, topk=3)
        self.assertEqual(report['counterfactuals'][0]['relative_embedding_l2'], 0.)
        self.assertEqual(report['counterfactuals'][0]['topk_overlap'], 1.)
        self.assertEqual((model.prior_control, model.graph_affinity, model.use_coarse_affinity), ('shared', True, True))
        for k, value in model.state_dict().items():
            torch.testing.assert_close(value, before[k], rtol=0, atol=0)

    def test_zero_alpha_makes_prior_interventions_inert(self):
        model, loader = self.fixture.make_model('amorerevision', {})
        with torch.no_grad():
            model.affinity_coarse_alpha.zero_()
            report = counterfactuals(model, torch.tensor([0]), loader.history_items_per_u, topk=3)
        for row in report['counterfactuals'][:3]:
            self.assertEqual(row['relative_embedding_l2'], 0.)

    def test_design_crosses_losses_with_affinity_and_prunes_real_extended_view(self):
        jobs = stage_jobs('loss-factorial', ['baby'], [999, 2024, 2025])
        self.assertEqual(len(jobs), 4)
        self.assertTrue(all(j['variants'] == ['rev_00', 'rev_11'] and not j['test'] for j in jobs))
        prune = stage_jobs('prune-followup', ['baby'], [999])[-1]['overrides']
        model, _ = self.fixture.make_model('amorerevision', prune)
        self.assertFalse(model.use_extended)


if __name__ == '__main__':
    unittest.main()
