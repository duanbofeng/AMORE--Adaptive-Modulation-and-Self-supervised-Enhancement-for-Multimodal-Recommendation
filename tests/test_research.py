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
from utils.checkpoint import load_checkpoint
from utils.dataset import RecDataset
from utils.dataloader import TrainDataLoader
from utils.graph_ops import knn_graph, edge_affinity, shared_graph
from utils.quick_start import quick_start
from utils.utils import get_model, init_seed
from run_suite import VARIANTS
from export_results import export_results

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
                if model == 'mm_mentor':
                    overrides = dict(overrides, embedding_size=64, feat_embed_dim=64)
                elif model.startswith('mm_'):
                    overrides = dict(overrides, feat_embed_dim=self.base['embedding_size'])
                instance, loader = self.make_model(model, overrides)
                instance.pre_epoch_processing()
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

    def test_zero_adapter_matches_frozen_gume_and_learns(self):
        frozen, _ = self.make_model('gume', {'freeze_features': True})
        adapted, loader = self.make_model('gume', {'freeze_features': True, 'feature_adapter_rank': 2})
        torch.testing.assert_close(frozen.forward(frozen.norm_adj), adapted.forward(adapted.norm_adj))
        adapted.calculate_loss(next(iter(loader))).backward()
        self.assertGreater(adapted.feature_adapters['image'][1].weight.grad.abs().sum(), 0)
        self.assertIsNone(adapted.image_embedding.weight.grad)

    def test_ego_zero_strength_preserves_gume_loss_gradients_and_rng(self):
        baseline, loader = self.make_model('gume', {})
        reference_rng = torch.get_rng_state()
        # Constructors may consume CPU RNG internally, but must never reseed CUDA.
        from utils.ego_router import EgoRouter
        cfg = baseline.config
        cfg['ego_mode'] = 'soft'
        with patch.object(torch.cuda, 'manual_seed_all') as cuda_seed:
            EgoRouter(baseline.R, baseline.embedding_dim, cfg)
        cuda_seed.assert_not_called()
        neutral, _ = self.make_model('gume', {'ego_mode': 'soft', 'ego_strength': 0.})
        self.assertTrue(torch.equal(reference_rng, torch.get_rng_state()))
        batch = next(iter(loader))
        torch.manual_seed(119)
        a = baseline.calculate_loss(batch)
        a.backward()
        torch.manual_seed(119)
        b = neutral.calculate_loss(batch)
        b.backward()
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        other = dict(neutral.named_parameters())
        for name, parameter in baseline.named_parameters():
            torch.testing.assert_close(parameter, other[name], rtol=0, atol=0)
            if parameter.grad is not None:
                torch.testing.assert_close(parameter.grad, other[name].grad, rtol=0, atol=0)

    def test_ego_chunks_preserve_gradients_and_train_edge_support(self):
        first, loader = self.make_model('gume', {'ego_mode': 'soft', 'ego_chunk_size': 2})
        second, _ = self.make_model('gume', {'ego_mode': 'soft', 'ego_checkpoint': False})
        router = first.ego_router
        pairs = set(map(tuple, router.edges.T.tolist()))
        train = loader.inter_matrix(form='coo')
        self.assertEqual(pairs, set(zip(train.row, train.col)))
        batch = next(iter(loader))
        for instance in (first, second):
            torch.manual_seed(87)
            instance.calculate_loss(batch).backward()
        for (name, a), (_, b) in zip(first.named_parameters(), second.named_parameters()):
            if a.grad is not None:
                torch.testing.assert_close(a.grad, b.grad, atol=1e-6, rtol=1e-5, msg=name)
        self.assertGreater(first.ego_router.cross_gates[0].weight.grad.abs().sum(), 0)

    def test_edge_aggregate_backward_matches_dense_and_numerical_gradient(self):
        from utils.ego_router import EdgeAggregate
        edges = torch.tensor([[0, 0, 1, 2], [0, 2, 1, 2]])
        weight = torch.randn(4, dtype=torch.double, requires_grad=True)
        items = torch.randn(3, 2, dtype=torch.double, requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(
            lambda w, x: EdgeAggregate.apply(edges, w, x, 4, 2), (weight, items)))
        dense = torch.sparse_coo_tensor(edges, weight, (4, 3)).to_dense() @ items
        routed = EdgeAggregate.apply(edges, weight, items, 4, 2)
        torch.testing.assert_close(routed, dense)
        probe = torch.randn_like(dense)
        expected = torch.autograd.grad((dense * probe).sum(), (weight, items))
        actual = torch.autograd.grad((routed * probe).sum(), (weight, items))
        for a, b in zip(actual, expected):
            torch.testing.assert_close(a, b)

    def test_ego_softmax_shuffle_and_degree_one_fallback(self):
        from utils.ego_router import EgoRouter
        model, _ = self.make_model('gume', {'ego_mode': 'hard'})
        cfg = model.config
        matrix = torch.tensor([[1., 0., 0.], [1., 1., 1.], [0., 0., 0.]]).to_sparse()
        router = EgoRouter(matrix, 4, cfg)
        logits = torch.tensor([2., 1., 3., -1.], requires_grad=True)
        alpha = router._softmax(logits)
        torch.testing.assert_close(alpha[0], torch.tensor(1.))
        torch.testing.assert_close(alpha[1:].sum(), torch.tensor(1.))
        shuffled = router._softmax(logits[router.permutation])
        torch.testing.assert_close(alpha[1:].sort().values, shuffled[1:].sort().values)
        image, text = torch.randn(3, 4), torch.randn(3, 4)
        vu, tu = matrix @ image, matrix @ text
        routed_v, routed_t = router(image, text, vu, tu)
        torch.testing.assert_close(routed_v[[0, 2]], vu[[0, 2]], atol=0, rtol=0)
        torch.testing.assert_close(routed_t[[0, 2]], tu[[0, 2]], atol=0, rtol=0)
        # Absolute threshold 0.4 rejects all uniform 3-neighbor edges; a relative
        # threshold below 1 selects them. Hard fusion must follow this decision.
        uniform_items = torch.ones(3, 4)
        other_items = uniform_items * 3
        cfg['ego_strength'] = 1.
        cfg['ego_threshold_type'] = 'absolute'
        cfg['ego_threshold'] = 0.4
        absolute = EgoRouter(matrix, 4, cfg)
        cfg['ego_threshold_type'] = 'relative'
        cfg['ego_threshold'] = 0.9
        relative = EgoRouter(matrix, 4, cfg)
        original = matrix @ uniform_items
        av = absolute(uniform_items, other_items, original, matrix @ other_items)[0]
        rv = relative(uniform_items, other_items, original, matrix @ other_items)[0]
        torch.testing.assert_close(av, original)
        torch.testing.assert_close(rv[1], original[1] * 2)

    def test_sg_sparse_aggregation_uses_inverse_item_degree(self):
        from utils.semantic_init import aggregate_semantic_users
        matrix = sp.csr_matrix([[1., 1., 0.], [1., 0., 1.]], dtype=np.float32)
        items = np.array([[2., 0.], [0., 4.], [2., 2.]], dtype=np.float32)
        local = aggregate_semantic_users(matrix, items)
        np.testing.assert_allclose(local, [[1., 4.], [3., 2.]])
        labels = np.array([0, 1, 0])
        mixed = aggregate_semantic_users(matrix, items, labels, 0.5)
        np.testing.assert_allclose(mixed, [[1., 4.25], [3., 1.75]])

    def test_sg_changes_only_extended_users_and_controls_preserve_scale(self):
        base, _ = self.make_model('gume', {})
        state = torch.get_rng_state()
        sg, _ = self.make_model('gume', {'sg_init': 'cluster', 'sg_clusters': 2})
        self.assertTrue(torch.equal(state, torch.get_rng_state()))
        shuffled, _ = self.make_model('gume', {'sg_init': 'shuffle', 'sg_clusters': 2})
        parameters = dict(sg.named_parameters())
        for name, parameter in base.named_parameters():
            if name in ('extended_image_user.weight', 'extended_text_user.weight'):
                self.assertFalse(torch.equal(parameter, parameters[name]))
                torch.testing.assert_close(parameter.square().mean(), parameters[name].square().mean())
            else:
                torch.testing.assert_close(parameter, parameters[name], rtol=0, atol=0)
        for name in ('extended_image_user', 'extended_text_user'):
            a, b = getattr(sg, name).weight, getattr(shuffled, name).weight
            torch.testing.assert_close(a.sort(dim=0).values, b.sort(dim=0).values)
        local, _ = self.make_model('gume', {'sg_init': 'local'})
        zero, _ = self.make_model('gume', {'sg_init': 'cluster', 'sg_cluster_weight': 0.})
        torch.testing.assert_close(local.extended_image_user.weight, zero.extended_image_user.weight)

    def test_combined_fusion_checkpoint_and_diagnostics(self):
        result = quick_start('gume', 'baby', dict(self.base, ego_mode='soft', sg_init='cluster',
                                               sg_clusters=2, ego_chunk_size=2), save_model=True)
        self.assertTrue(result['test'])
        saved = load_checkpoint(result['trials'][0]['checkpoint'])
        self.assertIn('ego_image_update_ratio', saved['diagnostics'])
        self.assertIn('sg_image_initialized_users', saved['diagnostics'])
        self.assertTrue(any('ego_router.cross_gates' in key for key in saved['state_dict']))

    def test_history_users_remove_parameters_and_keep_gradients(self):
        original, _ = self.make_model('gume', {})
        history, loader = self.make_model('gume', {'extended_user_source': 'history'})
        difference = sum(p.numel() for p in original.parameters()) - sum(p.numel() for p in history.parameters())
        self.assertEqual(difference, 2 * history.n_users * history.embedding_dim)
        history.calculate_loss(next(iter(loader))).backward()
        self.assertTrue(torch.isfinite(history.image_reduce_dim.weight.grad).all())

    def test_content_scores_match_full_sort_and_exclude_positive(self):
        for cross in (0., 1.):
            model, loader = self.make_model('contentcf', {'cross_weight': cross})
            batch = next(iter(loader))
            users, positive, negative = batch[:3]
            views = model.forward()
            profiles = model._profiles(users, views[1], views[2])
            paired = model._paired_scores(users, positive, views, profiles)
            ranking = model.full_sort_predict([users])
            torch.testing.assert_close(paired, ranking.gather(1, positive[:, None]).squeeze(1))
            without_positive = model._profiles(users, views[1], views[2], positive)
            raw_history = loader.inter_matrix(form='csr')
            for row, (user, pos) in enumerate(zip(users.tolist(), positive.tolist())):
                remaining = [i for i in raw_history[user].indices if i != pos]
                expected = views[1][0][remaining].sum(0) / (len(remaining) + model.config['history_shrinkage'])
                torch.testing.assert_close(without_positive[0][row], expected)

    def test_snapshot_contains_history_and_no_binary_artifacts(self):
        quick_start('contentcf', 'baby', dict(self.base, evaluate_test=False), save_model=True)
        snapshot = self.root / 'report.json'
        records = export_results(self.root / 'out', snapshot)
        self.assertEqual(len(records), 1)
        self.assertEqual(len(records[0]['histories']['0']), 2)
        self.assertTrue(snapshot.exists())
        self.assertNotIn('state_dict', snapshot.read_text())

    def test_original_amore_matches_archived_forward_loss_gradients(self):
        import hashlib
        from models.archive.amore_original import GUME as Archived
        from models.amorelegacy import AMORELEGACY
        archive = ROOT / 'src/models/archive'
        manifest = json.loads((archive / 'manifest.json').read_text())
        self.assertEqual(hashlib.sha256((archive / 'amore_original.py').read_bytes()).hexdigest(),
                         manifest['old_model/src/models/gume.py'])
        cfg = Config('amorelegacy', 'baby', dict(self.base, knn_k=10, affinity_threshold='>0.3333'))
        loader = TrainDataLoader(cfg, RecDataset(cfg).split()[0], batch_size=4)
        init_seed(cfg['seed'])
        reference = Archived(cfg, loader, None)
        init_seed(cfg['seed'])
        migrated = AMORELEGACY(cfg, loader, None)
        for name, value in reference.state_dict().items():
            torch.testing.assert_close(value, migrated.state_dict()[name], rtol=0, atol=0)
        torch.testing.assert_close(reference.image_original_adj.to_dense(), migrated.image_original_adj.to_dense())
        torch.testing.assert_close(reference.norm_adj.to_dense(), migrated.norm_adj.to_dense())
        torch.testing.assert_close(reference.forward(reference.norm_adj), migrated.forward(migrated.norm_adj))
        batch = next(iter(loader))
        torch.manual_seed(33)
        original_loss = reference.calculate_loss(batch)
        torch.manual_seed(33)
        actual_loss = migrated.calculate_loss(batch)
        torch.testing.assert_close(original_loss, actual_loss)
        original_loss.backward()
        actual_loss.backward()
        parameters = dict(migrated.named_parameters())
        for name, parameter in reference.named_parameters():
            if parameter.grad is None:
                self.assertIsNone(parameters[name].grad, name)
            else:
                torch.testing.assert_close(parameter.grad, parameters[name].grad, rtol=3e-4, atol=2e-6, msg=name)
        self.assertIsNone(reference.extended_image_user.weight.grad)
        self.assertIsNone(reference.extended_text_user.weight.grad)

    def test_original_graph_signed_neighbors_and_group_softmax(self):
        from utils.original_amore_ops import original_knn, grouped_softmax
        from utils.utils import build_sim, build_knn_normalized_graph
        feature = torch.tensor([[1., 0.], [.8, .2], [0., 1.], [-.1, 1.]])
        expected = build_knn_normalized_graph(build_sim(feature), 4, True, 'sym').coalesce()
        actual = original_knn(feature, 4, 2)
        torch.testing.assert_close(actual.to_dense(), expected.to_dense())
        self.assertTrue((actual.values() < 0).any())
        self.assertEqual(actual._nnz(), 16)  # selected zero cosines remain structural neighbors
        groups = torch.tensor([0, 0, 2, 2, 2])
        scores = torch.tensor([1000., 1001., -1000., -999., -998.], requires_grad=True)
        reference = torch.cat([scores[:2].softmax(0), scores[2:].softmax(0)])
        actual = grouped_softmax(scores, groups, 4)
        torch.testing.assert_close(reference, actual)
        torch.testing.assert_close(torch.autograd.grad(actual.square().sum(), scores, retain_graph=True)[0],
                                   torch.autograd.grad(reference.square().sum(), scores)[0])

    def test_clean_full_preserves_original_forward_and_removes_dead_tables(self):
        original, _ = self.make_model('amorelegacy', {})
        _, overrides = VARIANTS['amore_fixed']
        fixed, _ = self.make_model('amoreclean', overrides)
        torch.testing.assert_close(original.forward(original.norm_adj), fixed.forward())
        self.assertFalse(hasattr(fixed, 'extended_image_user'))
        self.assertEqual(sum(p.numel() for p in original.parameters()) - sum(p.numel() for p in fixed.parameters()),
                         2 * original.n_users * original.embedding_dim)
        changed, _ = self.make_model('amorelegacy', {'residual_alpha': 0.99, 'fine_grained_weight': 100.,
                                                    'use_pamd': False, 'pamd_loss_weight': 999.})
        torch.testing.assert_close(original.forward(original.norm_adj), changed.forward(changed.norm_adj))

    def test_corrected_losses_actual_batch_dedup_and_stability(self):
        from utils.original_amore_ops import unique_infonce
        model, loader = self.make_model('amoreclean', {})
        batch = next(iter(loader))[:, :2]
        losses = model.calculate_loss(batch)
        users, positive, negative = batch[:3]
        output = model.forward()
        u, p, n = output[users], output[model.n_users + positive], output[model.n_users + negative]
        expected_bpr = torch.nn.functional.softplus(-(u * (p - n)).sum(1)).mean()
        expected_reg = model.reg_weight_1 * (u.square().sum() + p.square().sum() + n.square().sum()) / 4
        torch.testing.assert_close(losses[0], expected_bpr)
        torch.testing.assert_close(losses[1], expected_reg)
        first, second = torch.randn(4, 5, requires_grad=True), torch.randn(4, 5, requires_grad=True)
        ids = torch.tensor([0, 1, 0, 2, 1])
        torch.testing.assert_close(unique_infonce(first, second, ids, .4),
                                   unique_infonce(first, second, ids.unique(), .4))
        stable = unique_infonce(first, second, ids, 1e-5)
        self.assertTrue(torch.isfinite(stable))
        stable.backward()
        self.assertTrue(torch.isfinite(first.grad).all())
        self.assertEqual(float(unique_infonce(first, second, torch.tensor([1, 1]), .4)), 0)
        self.assertEqual(float(model.moment_alignment(torch.ones(1), torch.ones(1))), 0)
        # Hand-computable two-identity contrastive problem: positive cosine=1,
        # negative cosine=0, so loss is log(1+exp(-1/tau)).
        identity = torch.eye(2)
        expected = torch.nn.functional.softplus(torch.tensor(-2.))
        torch.testing.assert_close(unique_infonce(identity, identity, torch.tensor([0, 1, 0]), .5), expected)

    def test_compact_skips_extended_branches_and_all_parameters_have_gradients(self):
        model, loader = self.make_model('amoreclean', {})
        self.assertFalse(model.use_extended)
        self.assertFalse(hasattr(model, 'affinity_guided_coarse'))
        with patch.object(model, 'conv_ui', wraps=model.conv_ui) as propagation:
            losses = model.calculate_loss(next(iter(loader)))
        self.assertEqual(propagation.call_count, 1)
        self.assertEqual([float(x) for x in losses[4:]], [0., 0., 0., 0.])
        sum(losses).backward()
        for name, p in model.named_parameters():
            self.assertIsNotNone(p.grad, name)
            self.assertTrue(torch.isfinite(p.grad).all(), name)

    def test_original_and_compact_checkpoint_roundtrip(self):
        for model_name in ['amorelegacy', 'amoreclean']:
            with self.subTest(model=model_name):
                output_dir = self.root / model_name
                config = dict(self.base, output_dir=str(output_dir))
                result = quick_start(model_name, 'baby', config, save_model=True)
                saved = load_checkpoint(result['trials'][0]['checkpoint'])
                model, _ = self.make_model(model_name, {})
                model.load_state_dict(saved['state_dict'])
                self.assertTrue(result['test'])
                self.assertEqual(saved['validation'], result['validation'])
                self.assertNotIn('norm_adj', saved['state_dict'])
                if model_name == 'amoreclean':
                    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
                    events = EventAccumulator(str(output_dir / 'trial_000')).Reload()
                    self.assertIn('train/bpr', events.Tags()['scalars'])
                    self.assertIn('train/um_noise', events.Tags()['scalars'])

    def test_amorex_single_matches_contentcf_and_uniform_slots(self):
        single, _ = self.make_model('amorex', {'num_interests': 1, 'objective': 'bpr'})
        control, loader = self.make_model('contentcf', {'content_weight': 1.})
        torch.testing.assert_close(single.full_sort_predict([torch.arange(4)]),
                                   control.full_sort_predict([torch.arange(4)]))
        batch = next(iter(loader))
        for actual, expected in zip(single.calculate_loss(batch), control.calculate_loss(batch)):
            torch.testing.assert_close(actual, expected)
        multi, _ = self.make_model('amorex', {'num_interests': 4, 'objective': 'bpr'})
        # Equal routes must not change content scale merely by increasing K.
        with torch.no_grad():
            for anchors in multi.prototypes:
                anchors.fill_(1.)
        torch.testing.assert_close(multi.full_sort_predict([torch.arange(4)]),
                                   single.full_sort_predict([torch.arange(4)]))

    def test_amorex_leave_one_out_values_and_target_gradient(self):
        model, loader = self.make_model('amorex', {})
        state = model.forward()
        users, positive, _ = next(iter(loader))[:3]
        users, positive = users[:1], positive[:1]
        profiles = model.profiles(users, state, positive)
        history = loader.inter_matrix(form='csr')
        np.testing.assert_array_equal(model.history.to_dense().numpy(), history.toarray())
        remaining = [i for i in history[int(users[0])].indices if i != int(positive[0])]
        for (z, a, _, _), (memory, prior) in zip(state[1], profiles):
            masses = a[remaining].sum(0)
            expected = (a[remaining, :, None] * z[remaining, None, :]).sum(0)
            expected = expected / (masses[:, None] + model.config['history_shrinkage'] / model.k)
            torch.testing.assert_close(memory[0], expected)
            gradient = torch.autograd.grad(memory.sum(), z, retain_graph=True)[0]
            torch.testing.assert_close(gradient[positive], torch.zeros_like(gradient[positive]), atol=1e-6, rtol=0)
            self.assertGreater(float(gradient[remaining].abs().sum()), 0)
            torch.testing.assert_close(prior.sum(1), torch.ones(1))
        # Explicit degree-zero and degree-one histories, including empty LOO.
        z, a, _, _ = state[1][0]
        sums = torch.stack([torch.zeros_like(a[0, :, None] * z[0]), a[0, :, None] * z[0]])
        mass = torch.stack([torch.zeros_like(a[0]), a[0]])
        fake = (state[0], [(z, a, sums, mass)])
        cold = model.profiles(torch.tensor([0]), fake)[0]
        singleton = model.profiles(torch.tensor([1]), fake, torch.tensor([0]))[0]
        for memory, prior in [cold, singleton]:
            torch.testing.assert_close(memory, torch.zeros_like(memory))
            torch.testing.assert_close(prior, torch.full_like(prior, 1 / model.k))

    def test_amorex_candidate_conditioning_and_chunked_scores(self):
        model, _ = self.make_model('amorex', {'num_interests': 2, 'score_user_chunk': 1, 'score_item_chunk': 3})
        users = torch.tensor([0, 2, 1])
        state = model.forward()
        profiles = model.profiles(users, state)
        dense = model.score_candidates(users, torch.arange(12), state, profiles)
        torch.testing.assert_close(model.full_sort_predict([users]), dense)
        choices = torch.tensor([[0, 2], [1, 4], [3, 7]])
        paired = model.score_candidates(users, choices, state, profiles, paired=True)
        torch.testing.assert_close(paired, dense.gather(1, choices))
        sum(model.calculate_loss([users, torch.tensor([0, 6, 3]), torch.tensor([8, 1, 10])])).backward()
        for projection, anchors in zip(model.projections, model.prototypes):
            self.assertGreater(float(projection.weight.grad.abs().sum()), 0)
            self.assertGreater(float(anchors.grad.abs().sum()), 0)
        # Opposing interests cancel under mean pooling, but matching either one
        # must score above an unrelated candidate with the nonlinear scorer.
        z = torch.tensor([[1., 0., 0., 0.], [0., 1., 0., 0.]])
        fake_state = (torch.zeros(model.n_users + model.n_items, 4), [(z, None, None, None)])
        fake_profiles = [(torch.tensor([[[1., 0., 0., 0.], [-1., 0., 0., 0.]]]), torch.tensor([[.5, .5]]))]
        score = model.score_candidates(torch.tensor([0]), torch.tensor([0, 1]), fake_state, fake_profiles)
        self.assertGreater(float(score[0, 0]), float(score[0, 1]))

    def test_amorex_negative_mask_and_one_negative_loss_equivalence(self):
        model, loader = self.make_model('amorex', {})
        id_only, _ = self.make_model('amorex', {'content_weight': 0.})
        self.assertTrue(torch.equal(model._negative_generator.get_state(), id_only._negative_generator.get_state()))
        users, positive, negative = next(iter(loader))[:3]
        expected = loader.inter_matrix(form='csr').toarray()[users.numpy()] > 0
        mask = model.training_positive_mask(users, torch.arange(12)[None, :])
        np.testing.assert_array_equal(mask.numpy(), expected)
        state = model.forward()
        profiles = model.profiles(users, state, positive)
        logits = model.sampled_logits(users, positive, negative, state, profiles, pool=torch.arange(12))
        invalid = mask | (torch.arange(12)[None, :] == negative[:, None])
        self.assertTrue(torch.equal(torch.isneginf(logits[:, 2:]), invalid))
        self.assertTrue(torch.isfinite(logits[:, :2]).all())
        # With no additional pool negatives, sampled CE is exactly pairwise BPR.
        logits = model.sampled_logits(users, positive, negative, state, profiles, pool=torch.empty(0, dtype=torch.long))
        ce = torch.nn.functional.cross_entropy(logits, torch.zeros(len(users), dtype=torch.long))
        bpr = torch.nn.functional.softplus(logits[:, 1] - logits[:, 0]).mean()
        torch.testing.assert_close(ce, bpr)

    def test_amorex_checkpoint_diagnostics_and_validation_protocol(self):
        result = quick_start('amorex', 'baby', dict(self.base), save_model=True)
        trial = result['trials'][0]
        saved = load_checkpoint(trial['checkpoint'])
        self.assertEqual(saved['validation'], result['validation'])
        self.assertTrue(result['test'])
        self.assertIn('image_effective_slots', saved['diagnostics'])
        self.assertIn('probe_content_abs', saved['diagnostics'])
        self.assertNotIn('train_keys', saved['state_dict'])
        model, _ = self.make_model('amorex', {})
        model.load_state_dict(saved['state_dict'])
        model.eval()
        with torch.no_grad():
            model.prepare_evaluation()
            cached = model.full_sort_predict([torch.arange(4)])
            model.clear_evaluation()
            torch.testing.assert_close(cached, model.full_sort_predict([torch.arange(4)]))
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        events = EventAccumulator(str(self.root / 'out/trial_000')).Reload()
        self.assertIn('diagnostics/image_effective_slots', events.Tags()['scalars'])
        self.assertEqual(len(events.Scalars('train/loss')), 2)

    def test_end_to_end_checkpoint_and_test_export(self):
        result = quick_start('amore', 'baby', dict(self.base, affinity_mode='local'), save_model=True)
        self.assertTrue(result['test'])
        trial = result['trials'][0]
        saved = load_checkpoint(trial['checkpoint'])
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

    def test_legacy_checkpoint_training_and_final_evaluation(self):
        original_load = torch.load
        calls = []
        # Emulate the 1.12 signature and reject keywords forwarded to pickle.
        def legacy_load(f, map_location=None, pickle_module=None, **pickle_load_args):
            self.assertNotIn('weights_only', pickle_load_args)
            calls.append(f)
            return original_load(f, map_location=map_location)
        with patch.object(torch, 'load', legacy_load):
            for model in ['gume', 'amore']:
                with self.subTest(model=model):
                    overrides = dict(self.base, output_dir=str(self.root / ('legacy_' + model)))
                    result = quick_start(model, 'baby', overrides, save_model=True)
                    self.assertTrue(result['test'])
                    self.assertTrue((self.root / ('legacy_' + model) / 'result.json').exists())
        self.assertEqual(len(calls), 4)  # Best checkpoint plus final test restore, per model.

    def test_modern_checkpoint_load_does_not_fall_back_on_error(self):
        calls = []
        def modern_load(f, map_location=None, *, weights_only=False, **kwargs):
            calls.append((map_location, weights_only))
            raise ValueError('checkpoint rejected')
        with patch.object(torch, 'load', modern_load):
            with self.assertRaisesRegex(ValueError, 'checkpoint rejected'):
                load_checkpoint('unused.pt', map_location='cpu')
        self.assertEqual(calls, [('cpu', True)])


if __name__ == '__main__':
    unittest.main()
