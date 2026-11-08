# Baseline provenance and adaptation boundaries

Sources are pinned in `experiments/baseline_sources.json`. `setup_baselines.py` verifies exact bytes. MMRec commit: `c68ee7c9b0de1ca7c5150258322edb4994be29d9`. Official MENTOR commit: `1581d2743b46a3a25d313f3e5a727e4393490ea6`. The seven models are **adapted implementations**, not unmodified author executables.

| Model | Source | Adaptation |
|---|---|---|
| BPR, LightGCN, BM3, LGMRec | enoche/MMRec | Shared data and evaluator, current SciPy sparse API; cache one encoding per evaluation pass |
| MGCN | enoche/MMRec | Above, device portability, exact chunked signed cosine self-KNN with content-based cache |
| FREEDOM | enoche/MMRec | Exact chunked binary self-KNN; rebuild graph from current features; preserve upstream sampled-edge training |
| MENTOR | Jinfeng-Xu/MENTOR | Parameter-free PyG aggregation replaced by equivalent sparse normalized aggregation; device/sparse fixes; exact binary self-KNN; recompute embeddings after checkpoint restoration; remove obsolete result parameters and avoid registering aliases of existing GCN parameters during forward |

MENTOR's loaded user-neighbor dictionary is unused by its scoring forward; the adapter supplies empty lists for that unused preprocessing path. It retains the published 64-dimensional branches, so reduced-dimension smoke configs are rejected. Its original forward mutates item indices in place; the adapter uses new tensors. Initial obsolete result parameters consume their original initialization randomness before removal.

LGMRec samples Gumbel assignments even during evaluation. Caching one encoding produces one shared draw per evaluation pass rather than a fresh draw per user batch. Report this adaptation, retain repeated seeds, and avoid claiming byte-for-byte equality with its original evaluator. All models score all items with the same training-positive mask and use validation checkpoint selection. BM3 applies its upstream predictor at evaluation. Training objectives are kept model specific; common BPR is not imposed on methods with different original objectives.

Exact chunking avoids dense all-item similarity storage, but still has quadratic construction arithmetic. Setup time is recorded separately. Baseline graph-cache behavior differs across methods and must be reported when comparing costs; use a common cold/warm-cache policy. CUDA timings still require actual server measurements. CPU synthetic runs are functionality checks only.

## Missing requested comparisons

| Method | Reference implementation | Status / next requirement |
|---|---|---|
| EgoGCN | supplied `EgoGCN-main`; DOI 10.1145/3503161.3548399 | Original standalone method not yet integrated. GUME-Ego is a hybrid, not this baseline. Map shared split/features and validate its own optimizer/edge modulation before reporting. |
| EVEN | AAAI paper DOI 10.1609/aaai.v39i12.33358 | Verified paper; official runnable source not confirmed. Obtain author implementation or explicitly disclose absence. |
| MMSSL | https://github.com/HKUDS/MMSSL | Full adversarial training loop not integrated. A BPR-only class with this name is not acceptable. |
| DiffMM | https://github.com/HKUDS/DiffMM | Full diffusion training/sampling not integrated. A ranking-only surrogate is not acceptable. |

These four comparisons remain open reviewer requests. Quoted paper numbers may be in a separate contextual table with incompatible protocols disclosed; they cannot establish significance or replace matched reruns. Never use missing entries as zeros in plots.

The local tests cover finite losses, backward passes and ranking interfaces. End-to-end synthetic runs additionally exercise checkpoint restoration. They do not verify reproduction of published Amazon scores. For a reproduction claim, compare each adapter against its pinned native executable on identical data/settings and document numerical differences.
