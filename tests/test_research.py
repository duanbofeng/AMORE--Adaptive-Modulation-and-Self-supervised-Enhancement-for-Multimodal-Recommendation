"""Synthetic correctness tests. These do not measure recommendation improvement."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'experiments'))
from utils.configurator import Config
from utils.dataset import RecDataset
from utils.dataloader import TrainDataLoader
from utils.graph_ops import knn_graph, edge_affinity, shared_graph
from utils.quick_start import quick_start
from utils.utils import get_model, init_seed
from run_suite import VARIANTS

torch.set_num_threads(1)


class ResearchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        directory = self.root / 'baby'
        directory.mkdir()
        rows = []
        for user in range(4):
            for item in range(5):
                rows.append((user, (user * 3 + item) % 12, 0 if item < 3 else item - 2))
        pd.DataFrame(rows, columns=['userID', 'itemID', 'x_label']).to_csv(directory / 'baby.inter', sep='\t', index=False)
        rng = np.random.default_rng(44)
        np.save(directory / 'image_feat.npy', rng.normal(size=(12, 8)).astype('float32'))
        np.save(directory / 'text_feat.npy', rng.normal(size=(12, 6)).astype('float32'))
        self.base = dict(data_path=str(self.root), use_gpu=False, seed=11, embedding_size=4,
                         train_batch_size=4, eval_batch_size=2, topk=[3, 5], valid_metric='Recall@3',
                         epochs=2, stopping_step=2, graph_chunk_size=3, knn_k=3, n_ui_layers=1,
                         export_rankings=True, output_dir=str(self.root / 'out'))

    def tearDown(self):
        self.temp.cleanup()

    def make_model(self, model, overrides):
        cfg = Config(model, 'baby', dict(self.base, **overrides))
        data = RecDataset(cfg).split()[0]
        loader = TrainDataLoader(cfg, data, batch_size=4)
        init_seed(cfg['seed'])
        return get_model(model)(cfg, loader, None), loader

    def test_all_variants_backward_and_ranking(self):
        for variant, (model, overrides) in VARIANTS.items():
            with self.subTest(variant=variant):
                instance, loader = self.make_model(model, overrides)
                batch = next(iter(loader))
                loss = instance.calculate_loss(batch)
                total = sum(loss) if isinstance(loss, tuple) else loss
                self.assertTrue(torch.isfinite(total))
                total.backward()
                for name, parameter in instance.named_parameters():
                    if parameter.grad is not None:
                        self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                instance.eval()
                with torch.no_grad():
                    instance.prepare_evaluation()
                    prediction = instance.full_sort_predict([torch.tensor([0, 1])])
                    instance.clear_evaluation()
                self.assertEqual(prediction.shape, (2, 12))
                self.assertTrue(torch.isfinite(prediction).all())

    def test_chunked_graph_matches_dense(self):
        features = torch.tensor([[1., 0.], [0.8, 0.2], [0., 1.], [0., 0.]])
        graph = knn_graph(features, 2, chunk_size=1)
        normalized = torch.nn.functional.normalize(features, dim=1)
        scores = normalized @ normalized.T
        scores.fill_diagonal_(-torch.inf)
        values, indices = scores.topk(2, dim=1)
        expected = torch.zeros_like(scores).scatter_(1, indices, values.clamp_min(0))
        np.testing.assert_allclose(graph.toarray(), expected.numpy())
        self.assertTrue(np.isfinite(graph.data).all())
        self.assertTrue((graph.diagonal() == 0).all())

    def test_affinity_leave_one_out_and_strength(self):
        features = torch.tensor([[1., 0.], [0., 1.], [0.8, 0.2]])
        interactions = sp.csr_matrix(np.array([[1., 0., 0.], [1., 1., 1.]], dtype=np.float32))
        graph = edge_affinity(features, interactions, 'local', 0.5, 5., 0.2, 11)
        self.assertEqual(graph[0, 0], 1.)  # Single-item history is neutral, not self-confirming.
        self.assertGreaterEqual(graph.data.min(), 0.5)
        self.assertLessEqual(graph.data.max(), 1.)
        self.assertNotEqual(graph[1, 0], graph[1, 1])
        neutral = edge_affinity(features, interactions, 'local', 0., 5., 0.2, 11)
        np.testing.assert_array_equal(neutral.toarray(), interactions.toarray())
        shuffled = edge_affinity(features, interactions, 'shuffle', 0.5, 5., 0.2, 11)
        np.testing.assert_allclose(np.sort(graph.data), np.sort(shuffled.data))

    def test_zero_affinity_reproduces_base(self):
        base, _ = self.make_model('amore', {})
        neutral, _ = self.make_model('amore', {'affinity_mode': 'local', 'affinity_strength': 0.})
        torch.testing.assert_close(base.forward(), neutral.forward())

    def test_refined_graph_and_frozen_features(self):
        graph = sp.csr_matrix(np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=np.float32))
        interactions = sp.eye(3, format='csr', dtype=np.float32)
        refined = shared_graph([graph], interactions, 2, cooc_weight=0.25, consistency=1, chunk_size=1)
        np.testing.assert_allclose(np.asarray(refined.sum(1)).ravel(), 1.)
        instance, _ = self.make_model('amore', {})
        self.assertFalse(instance.features[0].weight.requires_grad)
        self.assertNotIn('adj', instance.state_dict())

    def test_corruption_and_feature_cache_invalidation(self):
        instance, _ = self.make_model('amore', {'feature_corruption': 'zero',
                                              'feature_corruption_rate': 1., 'corruption_modality': 'image'})
        self.assertEqual(instance.features[0].weight.count_nonzero(), 0)
        self.assertGreater(instance.features[1].weight.count_nonzero(), 0)
        cached = list((self.root / 'baby/graph_cache').glob('*.npz'))
        self.make_model('amore', {})
        self.assertGreater(len(list((self.root / 'baby/graph_cache').glob('*.npz'))), len(cached))

    def test_only_training_edges_enter_collaborative_graph(self):
        instance, loader = self.make_model('amore', {'affinity_mode': 'local', 'user_init': 'history'})
        graph = instance.adj.to_dense().numpy()
        expected = loader.inter_matrix(form='csr').toarray() > 0
        np.testing.assert_array_equal(graph[:4, 4:] > 0, expected)

    def test_end_to_end_checkpoint_and_test_export(self):
        result = quick_start('amore', 'baby', dict(self.base, affinity_mode='local'), save_model=True)
        self.assertTrue(result['test'])
        trial = result['trials'][0]
        saved = torch.load(trial['checkpoint'], weights_only=True)
        self.assertEqual(saved['epoch'], trial['best_epoch'])
        self.assertEqual(saved['validation'], result['validation'])
        self.assertTrue((self.root / 'out/trial_000/test_rankings.npz').exists())
        self.assertTrue((self.root / 'out/result.json').exists())
        history = json.loads((self.root / 'out/trial_000/history.json').read_text())
        self.assertTrue(all('test' not in row for row in history))

    def test_grid_selects_validation_and_tests_once(self):
        from common.trainer import Trainer
        test_calls = []
        original_evaluate = Trainer.evaluate
        def fake_fit(trainer, *args, **kwargs):
            score = 0.9 if trainer.config['semantic_weight'] == 0.1 else 0.1
            trainer.best_epoch = 1
            trainer.epoch_seconds = [1.]
            trainer.history = [{}]
            torch.save({'state_dict': trainer.model.state_dict()}, trainer.checkpoint)
            trainer.writer.close()
            return score, {'recall@3': score}, {}
        def capture_test(trainer, data, is_test=False, **kwargs):
            if is_test:
                test_calls.append(trainer.config['semantic_weight'])
            return original_evaluate(trainer, data, is_test=is_test, **kwargs)
        with patch.object(Trainer, 'fit', fake_fit), patch.object(Trainer, 'evaluate', capture_test):
            result = quick_start('amore', 'baby', dict(self.base, semantic_weight=[0.1, 0.9],
                                hyper_parameters=['semantic_weight']), save_model=False)
        self.assertEqual(result['selected_trial'], 0)
        self.assertEqual(test_calls, [0.1])

    def test_validation_only_and_thinning(self):
        result = quick_start('gume', 'baby', dict(self.base, evaluate_test=False, train_fraction=0.5), save_model=False)
        self.assertEqual(result['test'], {})
        self.assertFalse(Path(result['trials'][0]['checkpoint']).exists())


if __name__ == '__main__':
    unittest.main()
