# AMORE: Adaptive Modulation and Self-supervised Enhancement for Multimodal Recommendation

Bofeng Duan (a), Caihong Mu (a), Tanveer Nazir (b), Yi Liu (b, corresponding author)

(a) School of Artificial Intelligence, Xidian University, Xi'an 710071, China

(b) School of Electronic Engineering, Xidian University, Xi'an 710071, China

Correspondence: yiliuxd@foxmail.com (Y. Liu)

## Abstract

Multimodal recommendation combines interaction histories with item content, but a content signal useful for one interaction need not be useful for another. AMORE investigates whether an interaction-derived affinity can guide both representation propagation and multimodal fusion. Building on GUME, the model computes affinities on observed user-item links, pools them into item-level statistics, and uses these statistics to modulate item initialization and coarse-to-fine fusion. Its objective combines pairwise ranking, cross-view contrastive terms, a scalar moment penalty, and representation regularization. We distinguish the implemented item-pooled mechanism from recipient-specific edge filtering and treat the utility of sharing affinity as an empirical hypothesis. The evaluation protocol isolates the two affinity uses, includes shuffled and constant priors, separates individual auxiliary objectives, and evaluates controlled corruption, degree-based cohorts, and computational cost. Experiments are restricted to the Baby, Sports and Clothing categories from Amazon. [PENDING E01/E02: insert matched-protocol test results, uncertainty and the supported conclusion; do not reinstate a universal superiority claim.] The formulation and evaluation make explicit which mechanism claims are testable and which require additional evidence.

Keywords: multimodal recommendation; graph propagation; affinity-guided fusion; contrastive learning; controlled evaluation

## 1 Introduction

Implicit-feedback recommendation learns user preferences from observed interactions. Such records indicate exposure and behavior rather than verified relevance, and sparse histories can limit collaborative estimation. Item images and text provide additional signals, although their usefulness depends on the recommendation context. Combining these inputs requires decisions about propagation, fusion and optimization, rather than an assumption that more multimodal components always improve ranking.

LightGCN demonstrates a simple collaborative graph encoder [1]. Multimodal approaches extend the available structure or representations: GRCN refines interaction graphs [2], LATTICE constructs item relations from content [3], and FREEDOM separates frozen item structure from interaction-graph denoising [4]. MGCN uses behavioral signals in feature purification and fusion [5]. GUME augments graph structure and introduces user representations and multimodal enhancement objectives [6]. These developments supply the backbone and context for this work; their established components are not claimed as new contributions of AMORE.

AMORE asks a narrower question: can the same interaction-derived statistic be useful at both propagation and fusion? Its affinity is a within-user softmax of embedding dot products. Such a score is relative to a user's observed neighbors, depends on degree and embedding scale, and is not a calibrated probability that an interaction is reliable. The implementation pools scores and threshold decisions by item. Consequently, it changes an item's shared initial representation, rather than assigning a separate final message to every recipient. This distinction determines the mechanism that the experiments must test.

A second question concerns auxiliary optimization. Cosine contrastive losses constrain correspondence between views, while a scalar moment penalty constrains global first and second moments. These constraints have different invariances, but neither their coexistence nor a lower training loss demonstrates better recommendation. AMORE therefore exposes the objectives separately and evaluates their incremental contributions on a fixed backbone.

The study contributes an explicit specification of the affinity-guided extension to GUME, a decomposition of its propagation and fusion uses, and an evaluation design that distinguishes signal reuse from additional capacity and auxiliary regularization. Claims about ranking improvements, robustness and sparse-item benefit depend on the corresponding experiments. We do not claim that residual subtraction proves semantic disentanglement, that an affinity threshold identifies genuine noise, or that three Amazon categories establish generalization to other recommendation domains.

## 2 Related work

### 2.1 Graph construction, refinement and fusion

GRCN [2] and FREEDOM [4] address graph refinement through different structural choices. EgoGCN explicitly exchanges information across modality-specific graph channels through edge-wise modulation [7]. AMORE is motivated by this use of interaction-dependent modulation, but its supplied implementation pools decisions into an item representation before propagation. It must therefore be distinguished from EgoGCN's recipient-dependent exchange. GUME [6] is the direct backbone, including the enhanced graph and auxiliary representation views; AMORE is an extension of that model rather than a new general graph-learning paradigm.

LGMRec combines local and global graph information [8]. EVEN jointly considers graph-structure evaluation and denoising [9], while DiffMM introduces diffusion-based multimodal recommendation [10]. These approaches motivate comparisons with explicit graph refinement and generative mechanisms. An observed gain over a collaborative baseline alone would not establish an advantage over these more closely related methods.

### 2.2 Purification, self-supervision and alignment

MGCN [5] provides closely related behavior-aware purification and fusion components. BM3 bootstraps latent multimodal representations [11]; MMSSL uses multimodal self-supervision [12]; and MENTOR employs ID-guided, multi-level alignment [13]. Recent disentanglement and hierarchical fusion work includes DHMRec [14]. These studies motivate both strong baseline comparisons and scrutiny of the added objectives. We do not assume that another method destroys fine-grained semantics, or that global moment agreement is equivalent to semantic alignment. The incremental question is whether AMORE's particular affinity reuse and auxiliary penalties help under a shared evaluation protocol.

## 3 Method

### 3.1 Inputs, graphs and notation

Let U and I be the user and item sets, E the observed training interactions, and R their binary incidence matrix. Visual and textual feature matrices are Fv and Ft. User and item ID embeddings are Eu and Ei with dimension d. The implementation initializes trainable feature tables from the supplied features. The fixed modality graphs are constructed once from their initial cosine similarities; the features used in representation learning remain trainable. The evaluation does not assume an independently verified feature extractor: dimensions and file hashes are recorded from the actual input files.

For each modality m, retain its k largest cosine similarities per row, including the self-neighbor. Let Wm denote these signed weighted neighbors and Dm its row-sum degree matrix. Negative retained similarities are not clipped. The normalized modality operator is Am = Dm^(-1/2) Wm Dm^(-1/2). Rows with nonpositive weighted degree are rejected rather than silently changing this definition. Neighbor sets can be directed; normalization does not make an asymmetric kNN graph symmetric.

Let B be the binary intersection of the visual and textual neighbor sets with its diagonal removed. The enhanced graph has blocks [0, R; R-transpose, B]. Its row-degree-normalized operator is A = D^(-1/2) G D^(-1/2), with inverse degree zero for isolated nodes. Let Rbar denote the user-to-item block of A. These operators use training interactions only; the catalogue and content features are available for all retained items.

| Symbol | Meaning |
|---|---|
| Hm, Xm | modality m item embeddings and concatenated user/item embeddings |
| C | propagated collaborative representation for all nodes |
| a(ui), q(i), r(i) | normalized edge affinity, pooled item affinity, selected-edge fraction |
| Z, Pm, S | coarse component, modality residual, integrated multimodal representation |
| V, Ym | averaged and modality-specific extended representations |
| O | final all-node representation used for ranking |
| lambda1, lambda2 | final and extended representation regularization weights |
| lambdaVT, lambdaBM, lambdaBME, lambdaUM, lambdaN | moment and contrastive coefficients |

### 3.2 Modality representations and pooled affinity

Each feature table is mapped through a linear projection followed by a second linear layer and sigmoid. Its output gates the item ID embedding elementwise. Lm modality-graph propagation steps yield Hm; there is no layer-average readout for this item graph. User modality representations are Rbar Hm, and Xm concatenates them with Hm.

EQ: H_m = A_m^{L_m}\left[E_i\odot\sigma\left(W_{m2}(W_{m1}F_m+b_{m1})+b_{m2}\right)\right].

The equation uses row-wise application of the linear maps. Define M = (Hv + Ht)/2. For a given propagation call with input user vectors U0 and item vectors I0, edge scores and per-user affinities are

EQ: s_{ui}=\langle U_{0,u},I_{0,i}\rangle,\qquad a_{ui}=\frac{\exp(s_{ui})}{\sum_{j:(u,j)\in E}\exp(s_{uj})}.

For the default threshold t, b(ui) = 1[a(ui) > t]. If re-attention is enabled, the selected edge score is replaced by the average of its ID and multimodal scores. The within-user softmax and threshold are then recomputed once. The final scores and decisions define

EQ: q_i=\frac{1}{d_i}\sum_{u:(u,i)\in E}a_{ui},\qquad r_i=\frac{1}{d_i}\sum_{u:(u,i)\in E}b_{ui},\qquad \widetilde I_{0,i}=I_{0,i}+\frac{r_i}{2}(M_i-I_{0,i}).

Here di is the training interaction count for item i; q and r are zero when di is zero. The expression is equivalent to averaging the original hard-modulated edge representations by item. Every recipient subsequently sees the same modified item initializer. The enhanced graph's connectivity is unchanged, but the collaborative representation is numerically affected. In particular, preserving connectivity does not mean preserving an untouched ID embedding path.

Starting from the concatenation of U0 and the modulated item initializer, propagation uses A, and readout averages layer zero through L UI layers. The main call uses U0 = Eu and I0 = Ei and produces C. The scalar item prior q from this call is reused in fusion. Hard threshold decisions are nondifferentiable; gradients through the continuous prior remain available to the fusion objective.

EQ: C=\frac{1}{L+1}\sum_{\ell=0}^{L}A^{\ell}[E_u;\widetilde E_i].

### 3.3 Affinity-guided coarse-to-fine fusion

A shared linear-tanh-linear network g maps each modality node vector to a scalar. A softmax over modalities produces base weights w. For items, a second shared linear-ReLU-linear network h takes the concatenation of Hi,m and qi and produces alternative modality weights p through a softmax. With a learned scalar alpha clamped to [0,1], the blend coefficient is beta(i) = sigmoid(3qi) alpha.

EQ: w_i'=\frac{(1-\beta_i)w_i+\beta_i p_i}{\sum_m[(1-\beta_i)w_{im}+\beta_i p_{im}]+10^{-8}}.

User weights retain w without this affinity blend. Let w* denote the resulting user/item weights. The coarse component Z, residual components P and final representation O are

EQ: Z=w_v^*\odot X_v+w_t^*\odot X_t,\quad P_m=\sigma(B_m C+c_m)\odot(X_m-Z),\quad S=(Z+P_v+P_t)/3,\quad O=C+S.

The scalar modality weights broadcast across embedding dimensions. The residuals describe the implemented subtraction and behavior gate; they are not orthogonal components and do not ensure identifiable semantic factors. The supplied original code computes an additional residual-enhancement expression that does not enter its final output. That unused expression and its ineffective coefficient are excluded from this specification and from claimed contributions.

### 3.4 Auxiliary views and training objective

For each modality, extended representations Ym use the same enhanced-graph propagation and layer-average readout, initialized by its explicit user vectors Rbar Hm and item vectors Hm. Each extended propagation call computes its own hard modulation from those input representations and the same M. V = (Yv + Yt)/2. Extended views are training auxiliaries; final prediction uses O only.

For a training batch of triples (u,i,j), the ranking score is the dot product of the final user and item vectors. The revision uses ordinary BPR [15], matching the scoring function used at inference.

EQ: L_{BPR}=\frac{1}{|\mathcal B|}\sum_{(u,i,j)\in\mathcal B}\log\left(1+\exp[-\langle O_u,O_i-O_j\rangle]\right).

Define N(X,Y;J,tau) as one-directional InfoNCE using normalized vectors, with one positive per unique identity in J and all other unique identities in that batch as negatives. A set containing fewer than two identities has a zero contrastive term. User sets Ju and positive-item sets Ji are treated separately; item indices refer to the item block of the concatenated representations.

EQ: N(X,Y;J,\tau)=-\frac{1}{|J|}\sum_{j\in J}\log\frac{\exp(\langle\widehat X_j,\widehat Y_j\rangle/\tau)}{\sum_{k\in J}\exp(\langle\widehat X_j,\widehat Y_k\rangle/\tau)}.

The behavior-multimodal term is N(S,C;Ju,tauBM) + N(S,C;Ji,tauBM). Its extended counterpart replaces S with V. The cross-view user term is N(V,S;Ju,tauUM). The perturbation term creates two views of each unique selected user vector in S and V by adding 0.1 times its sign times an independently sampled, row-normalized uniform random vector. It sums the corresponding two InfoNCE terms. These perturbations are not Gaussian noise.

The moment penalty compares a scalar mean and population variance over all entries of Xv and Xt:

EQ: L_{VT}=|\mu(X_v)-\mu(X_t)|+|\operatorname{var}(X_v)-\operatorname{var}(X_t)|.

It is neither a Gaussian divergence nor a sample-wise semantic alignment objective. Positive global rescaling of a view leaves cosine InfoNCE unchanged but can change this moment penalty. Conversely, arbitrary permutation of entries preserves both scalar moments while potentially destroying correspondence. These observations explain a possible complementary role and an important limitation; they do not prove a recommendation gain.

The final-representation penalty R1 sums squared norms of sampled final users, positive items and negative items, divided by twice the actual batch size. R2 sums squared norms of extended positive-item vectors with the same denominator. These are not weight decay over every model parameter. The complete objective has eight separately reported terms:

EQ: L=L_{BPR}+\lambda_1 R_1+\lambda_{VT}L_{VT}+\lambda_{BM}L_{BM}+\lambda_{BME}L_{BME}+\lambda_{UM}L_{UM}+\lambda_N L_N+\lambda_2 R_2.

### 3.5 Algorithm and implementation boundary

Algorithm 1. Training and validation selection for the repaired original AMORE.

1. Read fixed training, validation and test splits; build Am, the neighbor intersection, A and Rbar from training edges and available catalogue features. Initialize embeddings and transformation networks with the specified seed.
2. For each training batch, compute Hm and Xm. Compute edge affinities, optional one-step re-attention, and item-pooled q and r.
3. If propagation modulation is enabled, modify item initialization using r; otherwise keep Ei. Apply A for L layers and average layer zero through L to obtain C.
4. Compute base fusion weights. If affinity-guided fusion is enabled, blend item weights with the q-conditioned network. Construct Z, Pm, S and O exactly as in Section 3.3.
5. Compute auxiliary views when their coefficients require them. Evaluate the eight terms in Section 3.4, sum them, and update trainable parameters.
6. Evaluate validation Recall@20 after each epoch. Save an improving checkpoint and stop after the specified patience. Never select a checkpoint or hyperparameter using test scores.
7. After configurations are frozen using validation runs, restore the selected checkpoint and evaluate full-catalogue test ranking and efficiency.

The original source is archived separately from this repaired implementation. Repairs remove unused extended-user tables, use actual batch denominators, and prevent repeated identities from becoming contrastive false negatives. They are reported through an original-versus-repaired anchor, rather than hidden as part of the affinity contribution. The graph/fusion control with both switches enabled matches the full repaired reference. Earlier compact, GUME-Ego/SG and AMOREX experiments are distinct models and are not single-component ablations of this AMORE.

### 3.6 Computational considerations

Exact kNN construction has quadratic item-pair arithmetic; chunking reduces its intermediate memory, not this arithmetic complexity. Sparse propagation scales with retained graph edges, embedding dimension and layer count. Computing edge affinities uses O(|E|d) work and item pooling avoids storing a separate dense message vector for every edge. The affinity-guided fusion network adds a shared small MLP. Extended views add propagation work during training, while unique-identity contrastive matrices have quadratic cost in the number of distinct batch identities. Empirical costs must include feature tables, graph construction and auxiliary objectives, rather than counting only added fusion parameters.

## 4 Experimental protocol

### 4.1 Scope, inputs and research questions

The evaluation is limited to Amazon Baby, Sports and Clothing. It does not test short-video, music or cross-platform transfer. The supplied interaction labels define train/validation/test splits; all compared methods use the same mappings and feature files. Table 1 reports counts and dimensions extracted from those files. No claim is made that unverified encoder descriptions reproduce the supplied features.

Table 1. Dataset specification. [PENDING E01: fill actual user/item counts, split sizes, feature dimensions and hashes for the three input directories.]

RQ1 asks whether the repaired original AMORE improves ranking under matched selection and tuning. RQ2 asks whether propagation and fusion each benefit from affinity and whether reusing the corresponding item prior matters. RQ3 asks which auxiliary objectives and coefficients contribute. RQ4 asks how models behave under controlled corruption and for sparse, tail and zero-interaction cohorts. RQ5 measures accuracy against computational cost.

### 4.2 Baselines, selection and uncertainty

Executable common-runner comparators include BPR, LightGCN, BM3, MGCN, FREEDOM, LGMRec, MENTOR and GUME. External source revisions and adapter changes are pinned. These are adapted implementations and should not be described as unmodified native executions. EgoGCN, EVEN, MMSSL and DiffMM require additional matched comparisons or a documented method-specific limitation; they are not counted as completed experiments. Previously quoted literature values are not mixed with common-protocol scores.

The planned primary search uses 12 candidate configurations and three matched selection seeds per method and dataset. Search spaces and the deterministic candidate generator are released; model-specific objectives are retained. Final configurations are frozen by mean validation Recall@20, then evaluated with five matched seeds: 999, 2024, 2025, 2026 and 2027. Training has a maximum of 1,000 epochs and patience 20. Any reduction of this budget is applied and documented before results are observed. The protocol prevents test results from selecting among candidates, but does not imply exhaustive optimization of every model.

Full-catalogue ranking masks training positives. Validation positives are not added to the test mask; the same policy is used for all methods. Recall and NDCG at 10 and 20 are reported as means and sample standard deviations. Recall@20 is the primary confirmation metric, and NDCG@20 is secondary. Paired seed differences, 95% intervals and Holm-adjusted comparisons accompany effect sizes. The correction family is stated for each analysis; separate script calls do not automatically correct across datasets and metrics. Five optimization seeds have limited power and characterize optimization variability on fixed data, not generalization over independently sampled datasets.

### 4.3 Mechanism and objective controls

The main mechanism study holds the full repaired backbone and coefficients fixed and independently enables graph modulation and affinity-guided fusion. This yields y00, y10, y01 and y11. In addition to direct paired differences, it reports the interaction contrast y11 − y10 − y01 + y00. A positive full-model score alone does not demonstrate beneficial coupling.

Three controls retain propagation and alter only the fusion prior: a fixed item permutation, a global constant mean, and stop-gradient. The first two retain the fusion network capacity. The stop-gradient condition isolates the continuous prior's backward path; it does not make the hard threshold differentiable. Because disabling affinity-guided fusion removes its dedicated parameters, the same-capacity prior controls are needed to separate informative conditioning from additional capacity.

Loss ablations remove each auxiliary or representation-regularization term independently. A separate 2 by 2 study crosses scalar moments with all InfoNCE terms: full objective, no moments, no InfoNCE, and neither. BPR and representation penalties remain in that factorial comparison. Sensitivity sweeps vary one coefficient at a time: lambdaVT in {0, 0.001, 0.01, 0.1, 1}; lambda1 in {0, 10^-6, 10^-5, 10^-4, 10^-3}; lambda2 in {0, 0.001, 0.01, 0.1, 1}; and absolute affinity thresholds in {0.01, 0.05, 0.1, 0.2, 0.5}. These are exploratory validation analyses; final mechanism tests require settings fixed in advance.

### 4.4 Controlled corruption and degree cohorts

Interaction corruption adds unique false pairs equal to 0%, 5%, 10% or 20% of the clean training count. Known training, validation and test positives are excluded from injection. This use of heldout positives defines an offline artificial-corruption experiment, not a deployable noise detector. Modality corruption separately shuffles the image or text rows of selected items at rates 0%, 10%, 20% and 40%. Selected rows may contain permutation fixed points; the stated rate is the selected fraction. The same intervention seed is used across models and optimization seeds. Graphs are rebuilt from perturbed training inputs, while heldout labels and the original training-positive candidate mask remain fixed.

User activity groups are defined by clean training-degree tertiles. Warm items are ordered by training degree, breaking ties by item ID, and divided into five equally sized groups. The highest group is head and the remaining groups form tail; zero-degree items are separate. For each item cohort, full-ranking Recall/NDCG is computed over users with at least one test positive in that cohort, with the cohort positives defining the denominator. Eligible-user and positive counts accompany the results.

A separate intervention removes every training interaction of a fixed 20% subset of previously warm items, retaining their catalogue entries and content. This measures zero-training-interaction items in a transductive setting. It does not establish inductive generalization to unseen catalogue items or new users. The overall test average and the affected-item cohort are reported separately, since aggregate improvements can conceal poor cold-item performance.

### 4.5 Efficiency

On the same hardware and precision, the comparison records total and trainable parameters, setup time, training time per epoch including epoch-specific preprocessing, total training time, and peak allocated CUDA memory. After checkpoint restoration, one warmup and five timed inference passes separate representation encoding from ranking scores, masking and top-20 retrieval. Metric computation and output writing are excluded from the inference benchmark. Batch sizes, candidate count, graph cache state and software versions are disclosed. Parameter counts alone do not capture training cost, and CPU functionality checks are not presented as GPU performance measurements.

## 5 Results and analysis

### 5.1 Main comparison

Table 2. Common-protocol ranking performance, mean ± sample SD over five seeds. [PENDING E01/E03: insert test results for completed comparators, paired effects and adjusted tests. Retain all entries where a baseline wins. Finalize the baseline count from the actual table.]

The interpretation must distinguish a consistent positive effect from a small or dataset-dependent difference. A non-significant result indicates insufficient evidence of a difference, not equivalence. No percentage improvement from the submitted single-run table is retained without matching it to a completed rerun.

### 5.2 Affinity and objective mechanisms

Figure 2 and Table 3. Propagation/fusion factorial and corresponding-prior controls. [PENDING E02: insert per-seed contrasts and intervals. If shuffled or constant priors match the real prior, do not attribute the result to item-specific affinity reuse.]

Figure 3 and Table 4. Individual objectives and moments-by-InfoNCE factorial. [PENDING E06: report both gains and losses; determine whether the scalar moment term contributes beyond the contrastive objectives.] Figure 4. Coefficient sensitivity. [PENDING E07: insert controlled curves with uncertainty and identify stable versus fragile settings.]

### 5.3 Robustness, cohorts and cost

Figure 5. Controlled interaction and modality corruption. [PENDING E08: report absolute scores and change from each model's matching clean condition. Artificial robustness does not establish identification of real erroneous interactions.]

Figure 6 and Table 5. Sparse users, warm head/tail items and zero-interaction items. [PENDING E05: insert eligible counts, subgroup metrics and the separate cold-item intervention. Avoid treating ordinary long-tail evaluation as cold-start evidence.]

Figure 7 and Table 6. Accuracy and computational cost. [PENDING E04: insert same-GPU measurements, including preprocessing and cache policy.] Figure 8. Validation convergence. [PENDING E01: report epochs and elapsed training time without using the test curve for selection.]

## 6 Discussion and limitations

The proposed affinity is relative, degree-dependent and pooled by item. It cannot distinguish the final message received by two users of the same item, and a hard threshold may select very different fractions across user degrees. These limitations motivate the branch and prior controls rather than an a priori claim of reliability estimation. If the controls do not support sharing the prior, the appropriate conclusion is to remove or narrow that contribution, even if the complete model remains competitive.

The moment penalty controls only two global statistics. It cannot identify semantic correspondences or prove disentanglement. Residual subtraction likewise does not guarantee unique, independent factors. Visual projections may illustrate geometry but cannot establish these mechanisms. Robustness to injected interactions or feature shuffling is specific to those interventions and does not certify performance under naturally occurring errors.

All datasets are product categories from one source. Catalogue-aware graph construction and the zero-interaction intervention are transductive. Cross-domain, temporal, new-user and fully inductive claims remain untested. The common evaluator improves comparability but adaptations to external implementations, finite tuning budgets, five seeds and remaining missing related-method comparisons must be considered when interpreting results.

## 7 Conclusion

AMORE specifies an interaction-derived, item-pooled affinity for propagation initialization and multimodal fusion on a GUME-based backbone. Its revised description makes the pooling, fusion and auxiliary objectives explicit and separates them into testable interventions. [PENDING E01–E08: state the demonstrated outcome, the supported mechanism and its scope after the experiments; if affinity has no reliable benefit, say so and revise the final model accordingly.] The current scope does not establish universal superiority or cross-domain generalization.

## Data and code availability

The implementation, pinned source manifest, configuration plans and evaluation scripts are available at https://github.com/duanbofeng/AMORE--Adaptive-Modulation-and-Self-supervised-Enhancement-for-Multimodal-Recommendation. The final experimental release must identify the exact commit and data-file hashes. Dataset redistribution remains subject to the original source terms; feature provenance and access instructions must accompany the completed release.

## Acknowledgments

This work was supported by the National Natural Science Foundation of China (Nos. 62077038, 61672405, 62176196 and 62271374).

## References

[1] Xiangnan He, Kuan Deng, Xiang Wang, Yan Li, Yongdong Zhang, and Meng Wang. 2020. LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation. In Proceedings of the 43rd International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR ’20). 639–648. https://doi.org/10.1145/3397271.3401063

[2] Yinwei Wei, Xiang Wang, Liqiang Nie, Xiangnan He, and Tat-Seng Chua. 2020. Graph-Refined Convolutional Network for Multimedia Recommendation with Implicit Feedback. In Proceedings of the 28th ACM International Conference on Multimedia (MM ’20). 3541–3549. https://doi.org/10.1145/3394171.3413556

[3] Jinghao Zhang, Yanqiao Zhu, Qiang Liu, Shu Wu, Shuhui Wang, and Liang Wang. 2021. Mining Latent Structures for Multimedia Recommendation. In Proceedings of the 29th ACM International Conference on Multimedia (MM ’21). 3872–3880. https://doi.org/10.1145/3474085.3475259

[4] Xin Zhou and Zhiqi Shen. 2023. A Tale of Two Graphs: Freezing and Denoising Graph Structures for Multimodal Recommendation. In Proceedings of the 31st ACM International Conference on Multimedia (MM ’23). 935–943. https://doi.org/10.1145/3581783.3611943

[5] Penghang Yu, Zhiyi Tan, Guanming Lu, and Bing-Kun Bao. 2023. Multi-View Graph Convolutional Network for Multimedia Recommendation. In Proceedings of the 31st ACM International Conference on Multimedia (MM ’23). 6576–6585. https://doi.org/10.1145/3581783.3613915

[6] Guojiao Lin, Zhen Meng, Dongjie Wang, Qingqing Long, Yuanchun Zhou, and Meng Xiao. 2024. GUME: Graphs and User Modalities Enhancement for Long-Tail Multimodal Recommendation. CIKM 2024. https://arxiv.org/abs/2407.12338

[7] Feiyu Chen, Junjie Wang, Yinwei Wei, Hai-Tao Zheng, and Jie Shao. 2022. Breaking Isolation: Multimodal Graph Fusion for Multimedia Recommendation by Edge-Wise Modulation. In Proceedings of the 30th ACM International Conference on Multimedia (MM ’22). 385–394. https://doi.org/10.1145/3503161.3548399

[8] Zhiqiang Guo, Jianjun Li, Guohui Li, Chaoyang Wang, Si Shi, and Bin Ruan. 2024. LGMRec: Local and Global Graph Learning for Multimodal Recommendation. In Proceedings of the Thirty-Eighth AAAI Conference on Artificial Intelligence (AAAI ’24). 8454–8462. https://doi.org/10.1609/aaai.v38i8.28688

[9] Yuxin Qi, Quan Zhang, Xi Lin, Xiu Su, Jiani Zhu, Jingyu Wang, and Jianhua Li. 2025. Seeing Beyond Noise: Joint Graph Structure Evaluation and Denoising for Multimodal Recommendation. In Proceedings of the Thirty-Ninth AAAI Conference on Artificial Intelligence (AAAI ’25). 12461–12469. https://doi.org/10.1609/aaai.v39i12.33358

[10] Yangqin Jiang, Lianghao Xia, Wei Wei, Da Luo, Kangyi Lin, and Chao Huang. 2024. DiffMM: Multi-Modal Diffusion Model for Recommendation. ACM MM 2024. https://arxiv.org/abs/2406.11781

[11] Xin Zhou, Hongyu Zhou, Yong Liu, Zhiwei Zeng, Chunyan Miao, Pengwei Wang, Yuan You, and Feijun Jiang. 2023. Bootstrap Latent Representations for Multi-Modal Recommendation. In Proceedings of the ACM Web Conference 2023 (WWW ’23). 845–854. https://doi.org/10.1145/3543507.3583251

[12] Wei Wei, Chao Huang, Lianghao Xia, and Chuxu Zhang. 2023. Multi-Modal Self-Supervised Learning for Recommendation. In Proceedings of the ACM Web Conference 2023 (WWW ’23). 790–800. https://doi.org/10.1145/3543507.3583206

[13] Jinfeng Xu, Zheyu Chen, Shuo Yang, Jinze Li, Hewei Wang, and Edith C. H. Ngai. 2025. MENTOR: Multi-Level Self-Supervised Learning for Multimodal Recommendation. In Proceedings of the Thirty-Ninth AAAI Conference on Artificial Intelligence (AAAI ’25). 12908–12917. https://doi.org/10.1609/aaai.v39i12.33408

[14] Xiaohan Zhan, Yuliang Shi, Jihu Wang, Shijun Liu, Fanyu Kong, and Zhiyong Chen. 2026. DHMRec: Collaboration-Guided Multimodal Disentanglement and Hierarchical Fusion for Recommendation. In Proceedings of the Fortieth AAAI Conference on Artificial Intelligence (AAAI ’26). 16244–16252. https://doi.org/10.1609/aaai.v40i19.38661

[15] Steffen Rendle, Christoph Freudenthaler, Zeno Gantner, and Lars Schmidt-Thieme. 2009. BPR: Bayesian Personalized Ranking from Implicit Feedback. In Proceedings of the Twenty-Fifth Conference on Uncertainty in Artificial Intelligence (UAI ’09). AUAI Press, Arlington, VA, 452–461.
