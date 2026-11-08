"""Explicit execution adapter for pinned external MMRec sources.

Original files and license remain unmodified outside src. Adaptations below:
SciPy DOK API, device selection, exact chunked KNN, no stale graph caches,
and a single full-graph embedding computation per evaluation pass.
"""
import hashlib
import json
from pathlib import Path
import re
import torch
import types

from utils.original_amore_ops import original_knn


class MentorGCN(torch.nn.Module):
    """Exact parameter-free add aggregation of the reference Base_gcn on UI edges."""
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x, edge_index, size=None):
        row, col = edge_index[:, edge_index[0] != edge_index[1]]
        degree = torch.bincount(row, minlength=len(x)).to(x.dtype).clamp_min(1)
        weight = (degree[row] * degree[col]).rsqrt()
        adjacency = torch.sparse_coo_tensor(torch.stack([col, row]), weight, (len(x), len(x)))
        return torch.sparse.mm(adjacency, x)


def baseline_class(name):
    root = Path(__file__).resolve().parents[2]
    short = name.removeprefix('mm_') if hasattr(name, 'removeprefix') else name[3:]
    manifest = json.loads((root / 'experiments/baseline_sources.json').read_text())
    path = root / 'external_baselines/MMRec' / (short + '.py')
    if not path.exists():
        raise FileNotFoundError('Run python experiments/setup_baselines.py before using MMRec baselines')
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != manifest['files'][path.name]['sha256']:
        raise ValueError('External baseline source changed; check the pinned manifest')
    source = raw.decode().replace('A._update(data_dict)',
                                  'A = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)\n        for pair, value in data_dict.items():\n            A[pair] = value')
    source = source.replace("os.path.abspath(config['data_path'] + config['dataset'])",
                            "os.path.join(config['data_path'], config['dataset'])")
    if short == 'mgcn':
        for modality, prefix in [('image', 'v'), ('text', 't')]:
            pattern = r'            if os.path.exists\(' + modality + r'_adj_file\):.*?            self\.' + modality + r'_original_adj = ' + modality + r'_adj.cuda\(\)'
            replacement = ('            self.' + modality + '_original_adj = original_knn(self.' + prefix +
                           '_feat, self.knn_k, config["graph_chunk_size"] or 512, '
                           'os.path.join(dataset_path, "mmrec_mgcn_graph_cache"))')
            source, count = re.subn(pattern, replacement, source, flags=re.S)
            if count != 1:
                raise ValueError('Pinned MGCN graph patch no longer matches')
    if short in ('freedom', 'mentor'):
        source = source.replace('if os.path.exists(mm_adj_file):', 'if False:')
        source = source.replace('torch.save(self.mm_adj, mm_adj_file)', 'pass  # Rebuilt from current features; no stale cache')
    if short == 'mentor':
        source = re.sub(r'^.*(?:from torch_geometric|import torch_geometric).*\n', '', source, flags=re.M)
        source = source.split('class Base_gcn(MessagePassing):')[0]
        source = source.replace('torch.rand_like(h).cuda()', 'torch.rand_like(h)')
        source = source.replace("np.load(os.path.join(dataset_path, config['user_graph_dict_file']),\n                                       allow_pickle=True).item()",
                                '{i: ([], []) for i in range(self.n_users)}')
        # Reference stores obsolete initial result_* Parameters on CPU; its forward
        # later overwrites these with computed tensors. They are not model inputs.
        source = re.sub(r'(self.result_embed\w* = )nn.Parameter\(\s*nn.init.xavier_normal_\(torch.tensor\(np.random.randn\(num_user \+ num_item, dim_x\)\)\)\)\)',
                        r'\1nn.init.xavier_normal_(torch.tensor(np.random.randn(num_user + num_item, dim_x)))', source)
        source = source.replace('pos_item_nodes += self.n_users', 'pos_item_nodes = pos_item_nodes + self.n_users')
        source = source.replace('neg_item_nodes += self.n_users', 'neg_item_nodes = neg_item_nodes + self.n_users')
        # GCN returns its existing Parameter. Preserve an alias without registering
        # a new top-level Parameter during forward (which breaks checkpoint reload).
        for prefix in ('v', 't', 'id'):
            pattern = r'(        self\.' + prefix + r'_rep), self\.' + prefix + r'_preference = (.*)'
            replacement = (r'\1, preference = \2\n        object.__setattr__(self, "' + prefix + '_preference", preference)')
            source, count = re.subn(pattern, replacement, source)
            if count != 1:
                raise ValueError('Pinned MENTOR preference alias patch no longer matches')
    module = types.ModuleType('mmrec_' + short)
    module.__dict__['original_knn'] = original_knn
    module.__dict__['Base_gcn'] = MentorGCN
    exec(compile(source, str(path), 'exec'), module.__dict__)
    class_name = {'lgmrec': 'LGMRec', 'lightgcn': 'LightGCN'}.get(short, short.upper())
    original = getattr(module, class_name)

    class Adapted(original):
        def __init__(self, config, dataset, local_time=None):
            self._reference_name = short
            self._chunk = config['graph_chunk_size'] or 512
            if short == 'mentor' and config['embedding_size'] != 64:
                raise ValueError('Official MENTOR has fixed 64-dimensional latent branches; use embedding_size=64')
            super().__init__(config, dataset)
            if short == 'mentor':
                for key in list(self._parameters):
                    if key.startswith('result_embed'):
                        value = self._parameters.pop(key)
                        setattr(self, key, value.detach())
            self.diagnostics = {}
            self._eval_pair = None

        def get_knn_adj_mat(self, features):
            # FREEDOM uses BINARY top-k edges (including self), unlike MGCN.
            z = features / features.norm(dim=1, keepdim=True).clamp_min(1e-12)
            columns = torch.cat([(z[a:a+self._chunk] @ z.T).topk(self.knn_k, dim=1).indices
                                 for a in range(0, len(z), self._chunk)])
            rows = torch.arange(len(z), device=z.device)[:, None].expand_as(columns)
            indices = torch.stack([rows.flatten(), columns.flatten()])
            return indices, self.compute_normalized_laplacian(indices, (len(z), len(z)))

        @torch.no_grad()
        def prepare_evaluation(self):
            if short == 'mentor':
                dummy = torch.zeros(3, 1, dtype=torch.long, device=self.device)
                self.forward(dummy)
                self._eval_pair = (self.result_embed[:self.n_users], self.result_embed[self.n_users:])
                return
            if short in ('mgcn', 'freedom'):
                pair = self.forward(self.norm_adj)
            else:
                pair = self.forward()
            user, item = pair[:2]
            if short == 'bm3':
                user, item = self.predictor(user), self.predictor(item)
            self._eval_pair = (user, item)

        def clear_evaluation(self):
            self._eval_pair = None

        def full_sort_predict(self, interaction):
            if self._eval_pair is None:
                self.prepare_evaluation()
            user, item = self._eval_pair
            return user[interaction[0]] @ item.T

    return Adapted
