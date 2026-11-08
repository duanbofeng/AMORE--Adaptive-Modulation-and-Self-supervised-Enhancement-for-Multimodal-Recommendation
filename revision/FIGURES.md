# Figure and table plan

All performance figures are generated from completed `result.json` files. Synthetic fixtures are rejected by default. Error bars are sample SD unless the caption explicitly says CI. Runs with different source/data/configuration hashes are not pooled.

The model schematic is supplied as `figures/model.pdf`, `figures/model.svg` and `figures/model.png`; regenerate it with `python experiments/draw_revision_model.py`. It depicts computation and controlled switches, not an empirical improvement.

| Artifact | Inputs | Suggested caption and interpretation |
|---|---|---|
| Table 1: data | `.inter` and feature arrays | Unique users/items, train/validation/test counts, modality dimensions and file hashes. Report actual files; do not copy unverifiable counts. |
| Table 2: primary performance | locked final runs | Full-ranking Recall/NDCG@10/20, five-seed mean±SD. Bold only actual best mean; significance markers require the specified paired/Holm test. |
| Figure 1: model | audited equations | Edge affinities are pooled into item statistics. Dashed switches separate propagation and fusion; unchanged topology does not imply unchanged ID embeddings. |
| Figure 2: affinity factorial | affinity runs | Four shared-backbone conditions and shuffle/constant/detach controls. Report Δ11−Δ10−Δ01+Δ00 per seed, not merely the rank of four means. |
| Figure 3: objective ablation | losses runs | Individual removals plus moments × InfoNCE 2×2. Removing all InfoNCE leaves ranking and representation regularization. Lack of a reliable moment effect requires narrowing MACA claims. |
| Figure 4: sensitivity | four sensitivity folders | One parameter changed at a time; show zero explicitly for weights, avoid log(0). Threshold is absolute per-user softmax probability and degree-dependent. |
| Figure 5: robustness | separate interaction/image/text folders | Fixed corruption seed and clean validation/test targets; report absolute performance and degradation from the matching zero-noise condition. A better clean score alone is not robustness evidence. |
| Figure 6: cohorts | final top-K + clean degrees | User activity tertiles; warm-item head=top 20% by degree, tail=remaining 80%; zero-degree items separate. Define denominator as eligible users with group positives. Show per-seed counts. |
| Figure 7: efficiency | final benchmark + training timing | Same GPU, batch size and candidate universe; parameters, peak allocated CUDA MiB, setup seconds, training sec/epoch, wall time to best epoch, warmed inference time. Do not combine CPU and GPU timings. |
| Figure 8: convergence | history.json | Validation Recall@20 versus epochs; also show elapsed training time if comparing convergence speed. Avoid smoothed curves that conceal variance. |

`plot_revision.py` exports performance, accuracy/cost scatter, convergence and sweep plots; factorial interaction values are exported as CSV. `revision_cohorts.py` exports cohort metrics/figures. Main statistical tables come from `analyze_results.py`. Cost and paired-difference tables can be assembled directly from exported rows without copying values by hand.

Do not use t-SNE separation as proof of semantic disentanglement, nor affinity histograms as proof that removed edges were noise. Neither has ground-truth semantics. If exploratory projections are added later, fix the item sample, projection settings and seed, and label them qualitative.

## Evidence slots in manuscript and response

E01 primary matched reruns/statistics; E02 branch-factorial and prior controls; E03 missing related baselines; E04 GPU efficiency; E05 zero-interaction and tail cohorts; E06 individual losses/moments rationale; E07 weight sensitivity; E08 controlled noise. A slot is closed only after the associated real output and its protocol are reviewed. No placeholder should remain in a submitted manuscript.
