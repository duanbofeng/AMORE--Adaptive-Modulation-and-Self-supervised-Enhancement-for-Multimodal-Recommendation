"""Frozen-checkpoint restoration and safe reuse, not performance experiments."""
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).parent))
import test_research as research
from reviewer_fixed import canonical, specs, reuse
from run_suite import source_fingerprint
from evaluate_saved_revision import evaluate
from utils.quick_start import quick_start


class FixedReviewerTests(unittest.TestCase):
    def test_only_zero_rate_feature_conditions_are_canonicalized(self):
        base = dict(model='amorerevision', variant='rev_11', seed=999,
                    feature_corruption='none', feature_corruption_rate=0.,
                    corruption_modality='image', feature_corruption_seed=None, vt_loss=.1)
        zero = dict(base, feature_corruption='shuffle', corruption_modality='text', feature_corruption_seed=314159)
        self.assertEqual(canonical(base), canonical(zero))
        self.assertNotEqual(canonical(base), canonical(dict(zero, feature_corruption_rate=.1)))
        self.assertNotEqual(canonical(base), canonical(dict(base, vt_loss=0.)))
        self.assertNotEqual(canonical(base), canonical(dict(base, seed=2024)))

    def test_design_keeps_full_model_and_no_architecture_search(self):
        self.assertEqual(specs('reference')[0][1], ['rev_11', 'gume'])
        self.assertTrue(all(v == ['rev_11'] for _, v, _, _ in specs('sensitivity')))
        self.assertTrue(all('affinity_threshold' not in str(s) for s in specs('sensitivity')))
        self.assertEqual(len(specs('losses')[0][1]), 10)

    def test_checkpoint_evaluation_matches_original_without_training_or_input_mutation(self):
        f = research.ResearchTests(); f.setUp()
        try:
            result = quick_start('amorerevision', 'baby', dict(f.base, variant='rev_11', evaluate_test=True))
            path = f.root / 'out/result.json'; before = path.read_bytes()
            destination = f.root / 'evaluation'
            with patch('common.trainer.Trainer.fit', side_effect=AssertionError('No training allowed')):
                evaluate(path, destination, str(f.root), cpu=True, benchmark=True, allow_synthetic=True)
            restored = json.loads((destination/'result.json').read_text())
            self.assertEqual(restored['test'], result['test'])
            self.assertEqual(path.read_bytes(), before)
            self.assertTrue((destination/'trial_000/test_rankings.npz').exists())
            self.assertEqual(restored['trials'][0]['inference_benchmark']['repeats'], 5)
        finally:
            f.tearDown()

    def test_reuse_preserves_source_and_rejects_different_data(self):
        f = research.ResearchTests(); f.setUp()
        try:
            args = dict(f.base, variant='rev_11', evaluate_test=False, execution_fingerprint=source_fingerprint())
            quick_start('amorerevision', 'baby', args)
            original = f.root / 'out/result.json'; before = original.read_bytes()
            requested = f.root / 'copy'; requested.mkdir()
            (requested/'requested.json').write_text(json.dumps(args))
            plan = {'directory': str(requested), 'dataset': 'baby', 'command': ['python', 'main.py', '--model', 'amorerevision']}
            self.assertEqual(reuse([plan], [f.root/'out'], f.root, execute=True), 1)
            copied = json.loads((requested/'result.json').read_text())
            self.assertEqual(copied['validation'], json.loads(before)['validation'])
            self.assertEqual(original.read_bytes(), before)
            self.assertTrue((requested/'trial_000/history.json').exists())
            other = f.root/'different'; other.mkdir()
            (other/'requested.json').write_text(json.dumps(args))
            with (f.root/'baby/baby.inter').open('a') as stream:
                stream.write('\n')
            with self.assertRaisesRegex(ValueError, 'changed dataset'):
                reuse([dict(plan, directory=str(other))], [f.root/'out'], f.root, execute=True)
        finally:
            f.tearDown()


if __name__ == '__main__':
    unittest.main()
