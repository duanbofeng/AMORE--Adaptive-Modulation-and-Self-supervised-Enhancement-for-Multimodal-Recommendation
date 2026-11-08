# Response to reviewers: IPM-D-26-07304

## R1.1: Marginal and inconsistent gains over MENTOR; no uncertainty or significance testing.

We agree that the submitted single-run differences do not establish a reliable advantage. We have removed the statement of consistent superiority and defined a common validation-based selection and full-ranking evaluation protocol. The revised comparison must report all five seed scores, means, sample standard deviations, paired differences, confidence intervals and multiplicity-adjusted tests. MENTOR is included as a direct executable comparator. [PENDING E01: insert the completed matched-protocol table and qualified interpretation, including metrics on which AMORE loses.]

**Location:** Sections 4.2 and 5.1

**Status:** text corrected; experiment pending

## R1.2: Removing ELAM changes Recall only marginally; the flagship module is not supported.

We agree that the reported changes of 0.0002 and 0.0005 are insufficient evidence without repeated runs. We have implemented separate propagation and fusion switches on the repaired original AMORE, preserving its full backbone and auxiliary objectives. This provides a 2 by 2 comparison and matched controls that shuffle or replace the pooled affinity signal. We will retain a mechanism claim only if its effect is reproducible relative to the relevant single-branch controls. [PENDING E02: insert effect sizes and uncertainty; remove the flagship claim if unsupported.]

**Location:** Sections 3.2, 4.3 and 5.2

**Status:** controls implemented; experiment pending

## R1.3: EgoGCN, LGMRec, EVEN, MMSSL and DiffMM are absent from comparisons.

We agree that discussing closely related methods without corresponding comparisons weakens the empirical case. LGMRec has been connected to the common runner, alongside MENTOR, MGCN, FREEDOM, BM3, LightGCN, BPR and GUME. EgoGCN, EVEN, MMSSL and DiffMM remain required comparison work or require a specific, documented exclusion rationale. We have not represented those missing reruns as completed or substituted quoted results for them. [PENDING E03: complete these comparisons and update the final baseline count, or provide method-specific availability and protocol limitations.]

**Location:** Sections 2 and 4.2

**Status:** partially implemented; four external comparisons open

## R1.4: All datasets come from Amazon; generalization to video or music is unknown.

We agree that three product categories from one source do not establish cross-domain generalization. The revision remains limited to Baby, Sports and Clothing; we do not add a video or music dataset. We have restricted the scope of the claims and stated this limitation explicitly in the experimental scope and conclusion. Consequently, this concern is acknowledged rather than claimed to be empirically resolved.

**Location:** Sections 4.1, 6 and 7

**Status:** scope narrowed; cross-domain request not experimentally addressed

## R1.5: Training, inference and memory efficiency are missing.

We have added reproducible efficiency instrumentation to the shared runner. It records trainable and total parameter counts, setup time, training epoch time including epoch-specific graph processing, and peak allocated GPU memory. Inference is measured after restoring the selected checkpoint, with one warmup and five timed passes, separating graph encoding from full-ranking scores, masking and top-20 retrieval. Metric computation and file writing are excluded from that inference benchmark. [PENDING E04: insert same-hardware measurements and disclose batch sizes and cache state.]

**Location:** Sections 3.6, 4.5 and 5.3

**Status:** instrumentation implemented; GPU measurements pending

## R2.1: The same affinity connecting propagation and fusion is never isolated.

We agree. Removing an entire module changes too many factors to validate reuse of a shared signal. The new controls independently enable graph modulation and affinity-guided fusion. The fusion-only variant still computes the same prior but does not modulate the propagated item representation. In the shuffled-prior variant the propagation output is preserved while the association between items and fusion priors is broken. We report the paired interaction effect y11 minus y10 minus y01 plus y00 in addition to ordinary ablations. [PENDING E02: insert results; a positive total model effect alone will not be described as evidence of coupling.]

**Location:** Sections 3.2–3.3 and 4.3

**Status:** controls implemented; experiment pending

## R2.2: Algorithm 1 and the equations are inconsistent; symbols are undefined.

We have audited the original network in old_model/src/models/gume.py and verified its archived source hash. The corrected description states that edge scores and binary decisions are pooled by item before propagation; it does not describe a recipient-specific edge message after pooling. The updated algorithm defines all inputs, uses the actual layer-average readout and fusion factor of one third, and specifies the exact training losses. The unused extra residual path and the inactive adaptive BPR claim have been removed. The revision reference is tested against the full repaired original model for forward outputs and common-parameter gradients.

**Location:** Section 3 and Algorithm 1

**Status:** text and executable checks completed

## R2.3: Most baseline scores are quoted rather than rerun under one protocol.

We agree that quoted numbers cannot support a controlled improvement claim. The revised primary table is reserved for runs with the same item/user mapping, split files, feature files, candidate set, validation criterion and seed set. Pinned reference sources are used for the implemented baselines, and each adaptation is documented. Model-specific objectives are retained rather than forcing every model to use BPR. Hyperparameter selection is restricted to validation data, and the final comparison records the search budget and selected configuration for every model. [PENDING E01 and E03: fill the rerun table only after the comparator cohort is complete.]

**Location:** Section 4.2

**Status:** common runner available; tuning and reruns pending

## R2.4: Cold-start and long-tail motivation has no corresponding evaluation.

We have separated the previously conflated settings. Warm items are divided into five equally sized groups using clean training degree, with the top group defined as head and the remaining groups as tail. Sparse-user groups are also defined from training degree. A separate controlled experiment removes every training interaction for a fixed subset of items while retaining their content features and the item catalogue. This tests zero-interaction items in a transductive setting, not unseen-item inductive generalization. [PENDING E05: insert group metrics, eligible-user and positive counts, and the cold-item comparison.]

**Location:** Sections 4.4 and 5.3

**Status:** protocol and cohort analysis implemented; experiment pending

## R2.5: Baseline counts, hyperparameters and terminology contain inconsistencies.

We have rewritten the research questions and method terminology, removed the duplicated sentence in the experimental introduction, and removed sensitivity claims for parameters that do not affect prediction in the supplied implementation. The final baseline count will be generated from the completed rerun table, rather than carried over from the inconsistent eleven/twelve descriptions. All hyperparameter names in the revised specification map to explicit configuration keys. [PENDING E01: finalize the count and table cross-references after inserting results.]

**Location:** Sections 3–5

**Status:** text corrected; final table check pending

## R3.1: Recent multimodal graph denoising literature is insufficiently covered.

We have reorganized related work by graph refinement, representation purification and alignment, and explicitly acknowledged the GUME backbone and the relation to MGCN and EgoGCN. The discussion now includes EVEN, DiffMM and recent disentanglement work such as DHMRec. We distinguish edge removal, cross-modal message exchange, moment regularization and graph-generation mechanisms, without presenting their components as original AMORE contributions. The comparison checklist separately records which methods have executable reruns and which remain open.

**Location:** Sections 1–2

**Status:** literature and attribution revised; comparison gaps disclosed

## R3.2: The value of mean/variance alignment beyond InfoNCE is unexplained.

We have corrected the interpretation of this term. The implementation penalizes differences in a global scalar mean and population variance; it does not estimate a Gaussian divergence or enforce sample-wise semantic correspondence. Cosine-based InfoNCE is invariant to positive rescaling of an entire view, whereas the moment term can respond to that rescaling. Conversely, matching these two scalar moments does not identify semantic alignment: permuting entries preserves them. The two constraints can therefore be complementary, but improvement is an empirical question. We have added the four combinations of InfoNCE and moment matching with representation regularization held fixed. [PENDING E06: insert the factorial loss results.]

**Location:** Sections 3.4 and 4.3

**Status:** interpretation corrected; factorial experiment pending

## R3.3: Multiple seeds, standard deviations and significance tests are missing.

We agree. The protocol specifies five matched optimization seeds, reports Recall and NDCG at 10 and 20, and uses paired seed-level comparisons rather than treating every user as an independent training replicate. We report confidence intervals and Holm-adjusted tests alongside effect sizes, and acknowledge the limited power of five seeds. A non-significant result will be reported as insufficient evidence of a difference, not proof of equivalence. [PENDING E01 and E02: insert the completed uncertainty analysis.]

**Location:** Sections 4.2 and 5.1

**Status:** analysis support available; repetitions pending

## R3.4: Branch-specific affinity use and individual auxiliary losses need ablation.

We have implemented separate propagation/fusion controls and explicit independent loss coefficients. The full repaired model is the common reference; each loss removal changes one coefficient, while the InfoNCE/moment factorial study is reported as a separate joint intervention. The compact model and the earlier GUME-Ego/SG or AMOREX explorations are not treated as single-module ablations of the submitted AMORE. [PENDING E02 and E06: insert matched-seed effects for each control.]

**Location:** Sections 4.3 and 5.2

**Status:** controls implemented; experiment pending

## R3.5: Sensitivity to moment-alignment and L2 coefficients is absent.

We have added separate sensitivity sweeps for the moment coefficient and both representation regularization coefficients, including zero-valued controls and logarithmically spaced positive values. The text now specifies that these penalties act on sampled final or extended representations, rather than on every learnable parameter. The inactive extra-residual coefficient is not included as an effective model hyperparameter. [PENDING E07: insert validation-selected sensitivity curves with variation across seeds.]

**Location:** Sections 4.3 and 5.2

**Status:** sweeps implemented; experiment pending

## R3.6: Controlled interaction and modality-noise robustness is missing.

We agree that a denoising claim requires controlled evidence. The revision provides training-only interaction injection and separate visual/textual feature-mismatch experiments. The same intervention is used for compared models; validation and test labels are unchanged, and injected pairs exclude known heldout positives. We distinguish these artificial corruptions from verified erroneous interactions in real data. We report both absolute performance and degradation from each model's clean condition. [PENDING E08: insert robustness curves and remove or qualify denoising claims if unsupported.]

**Location:** Sections 4.4 and 5.3

**Status:** interventions and plotting implemented; experiment pending

## R3.7: Parameter counts, epoch time and GPU memory are absent.

We have added the efficiency measurements described in our response to R1.5, including total and trainable parameters, graph/setup time, per-epoch training time, and peak allocated CUDA memory. The table will disclose hardware, numerical precision, batch sizes, graph-cache state and the cost of full-catalogue ranking. [PENDING E04: fill the measured table; no CPU smoke-test timings will be presented as GPU benchmark results.]

**Location:** Sections 4.5 and 5.3

**Status:** instrumentation implemented; GPU measurements pending

## R3.8: Mechanistic explanations are asserted without validation.

We agree. The revised text separates the executed computation, the hypothesis it motivates, and empirical support. In particular, an attention score is not described as calibrated interaction reliability; residual subtraction is not presented as proof of semantic disentanglement; and a two-dimensional visualization is not treated as causal evidence. The earlier claims that MENTOR damages fine-grained semantics and that AMORE reliably filters real interaction noise have been removed. Any replacement interpretation must be supported by the corresponding controlled experiment.

**Location:** Sections 1, 3 and 6–7

**Status:** unsupported interpretations removed; empirical conclusions pending
