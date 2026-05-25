# Accelerating LLM Inference with Staged Speculative Decoding

**ArXiv:** [2308.04623](https://arxiv.org/abs/2308.04623)

## 🎯 Pitch

This paper introduces 'staged speculative decoding,' a novel approach that restructures speculative decoding into a tree-shaped batch and adds a second stage of speculation, dramatically boosting the arithmetic intensity of small-batch large language model (LLM) inference. By reducing memory bandwidth requirements and parallelizing predictions across models, the method achieves up to 3.16× faster single-sequence decoding with no loss in output quality—making fast, private, and on-device LLM applications significantly more practical and accessible.

---

## 1. Executive Summary
This paper introduces staged speculative decoding, a two-part acceleration method for small‑batch large language model (LLM) inference. It restructures speculative decoding into a tree-shaped batch and adds a second speculative stage, achieving up to 3.16× faster single‑sequence decoding on a 762M‑parameter GPT‑2‑Large while preserving output distribution and quality (Abstract; Table 2).

## 2. Context and Motivation
- Problem addressed
  - Small‑batch, on‑device LLM inference is slow because decoding is memory‑bandwidth bound: each new token requires streaming all model weights with limited computation per byte moved. The paper quantifies the arithmetic intensity of batch‑1 decoding as ≈1 FLOP/byte and shows an RTX 4090 reaches only 150 tokens/s for GPT‑2‑Large with a compute utilization of 0.13% (Section 2.2; Figure 1).
  - Speculative decoding helps, but its gains saturate because a single draft sequence rarely matches the oracle for many consecutive tokens; the probability of long correct runs decays exponentially (Section 3.1).

- Why it matters
  - On‑device inference improves latency, personalization, and privacy, and broadens access to LLMs without cloud resources (Introduction).
  - The roofline analysis (Figure 1) demonstrates that to go faster under small‑batch conditions, one must increase arithmetic intensity—i.e., do more useful work per byte of memory traffic.

- Prior approaches and limitations
  - Quantization and FlashAttention improve efficiency but don’t resolve low arithmetic intensity during sequential decoding (Section 2.3).
  - Classic speculative decoding uses a smaller draft model to propose a linear sequence, then checks it with the large model (Section 2.3; Leviathan et al., 2022; Chen et al., 2023). This helps, but performance gains saturate as the proposed run length grows.

- Positioning of this work
  - The method targets the same bandwidth bottleneck as speculative decoding but improves both how the speculative batch is constructed (tree instead of a single path) and how it is produced (a second, even cheaper speculative stage), converting more sequential work into parallel work (Sections 3.1–3.2).

## 3. Technical Approach
The paper proposes two linked mechanisms: tree‑structured speculative batches and a second speculative stage.

Key terms
- `oracle model`: the large, accurate LLM whose outputs we must preserve.
- `draft model`: a smaller, faster model used to propose tokens for speculative checking by the oracle.
- `draft2 model`: an even cheaper model used to accelerate the draft itself.
- `KV cache`: stored key/value tensors that let a transformer reuse past computations during decoding.
- `arithmetic intensity`: ratio of floating‑point operations to bytes moved; higher ratios reduce memory‑bandwidth bottlenecks.
- `speculative decoding`: proposing multiple future tokens (from a draft) and verifying them in parallel with the oracle.

A. Why a tree instead of a single draft path (Section 3.1)
- Problem: A single linear proposal faces exponentially decreasing match probability across positions; most work at the tail is wasted when the oracle disagrees.
- Solution: Build a branching “tree” of candidate sequences that reallocates effort toward early positions where agreement probability is higher. This:
  - Increases expected accepted tokens per oracle batch by covering second/third-best options early.
  - Produces more leaf nodes (i.e., more candidate continuations) “for free” because the draft runs only at internal nodes; deeper leaves reuse shared prefixes.
  - Enables greater parallelism for the draft: nodes at the same depth can be processed in parallel.

B. How the tree‑structured batch is realized (Section 3.1)
- Attention decomposition: During speculative decoding, the oracle’s attention is partitioned into:
  - Cross‑attention to the existing KV cache (history up to the current position).
  - Self‑attention within the synthetic, tree‑shaped batch block (the speculative tokens).
- Two implementation levers maintain causal structure inside the batch:
  - Positional embeddings are assigned so each node aligns with the correct future position.
  - A custom causal mask enforces the tree’s partial order (children only attend to allowed ancestors/siblings).
- KV management:
  - Compute a “new” KV cache for the speculative batch separately.
  - After verification/sampling, append only the accepted slices from the batch cache to the main KV cache.

Concrete intuition
- Imagine generating the next 4 tokens. Instead of proposing just one 4‑token path (e.g., A→B→C→D), build a small tree at the early steps (A1/A2 at step 1; B1/B2 under each Aj at step 2). The oracle evaluates all leaves in one batched pass. If the oracle agrees with path A2→B1 for the first two tokens, you accept both at once. If it disagrees early, computation spent on deeper branches still yields candidates for other early choices.

C. Second speculative stage (“staged speculation,” Section 3.2)
- Motivation: With larger speculative batches, the draft model can dominate runtime; it too is memory‑bound at small batch size.
- Approach: Apply speculative decoding to the draft using an even cheaper `draft2` model (here, a Katz backoff trigram language model).
  - The `draft2` proposes a mini‑tree for the draft, which the draft verifies in parallel.
  - The draft’s verified tree then forms the input tree for the oracle.
- Net effect: Two levels of parallel verification transform more of the end‑to‑end process from sequential token generation into batched compute, increasing arithmetic intensity at both stages.

D. Quality preservation (Sections 2.3, 4)
- Deterministic (greedy) decoding: No change in output because the oracle always verifies and selects tokens; speculative tokens are accepted only when they match the oracle.
- Sampling (e.g., top‑k): A rejection‑sampling scheme ensures sampling from the original oracle distribution; speculative tokens are treated as proposals that can be accepted or rejected so the final distribution matches the oracle’s (Section 2.3).

E. Experimental setup details that matter (Section 4)
- Hardware: Single NVIDIA RTX 4090.
- Models:
  - Oracle: GPT‑2‑Large (762M) fine‑tuned on Python from The Stack.
  - Draft: 40M GPT‑2 trained on the same data.
  - Draft2: Katz backoff trigram model generated from 120M tokens sampled from the draft.
- Workload: The 164 HumanEval prompts; both deterministic decoding and top‑k sampling (k=50, T=1).

## 4. Key Insights and Innovations
1) Tree‑structured speculative batches (Section 3.1)
- What’s new: Instead of a single speculative path, construct a masked-attention batch that encodes a branching tree of early alternatives.
- Why it matters:
  - Higher expected acceptances per batch because early branches cover likely alternatives.
  - More leaf nodes relative to draft compute: the draft runs only at internal nodes, raising the ratio of oracle‑checked candidates to draft cost.
  - Draft-level parallelism: process nodes at the same depth together, relieving the draft’s own memory‑bound bottleneck.
- Difference from prior work: Classic speculative decoding forms a single candidate path; the tree restructures where speculative compute is spent.

2) Staging speculation (Section 3.2)
- What’s new: Apply speculative decoding again to the draft using a lightweight `draft2` (here, an n‑gram model).
- Why it matters:
  - In practice, large speculative batches make the draft a bottleneck; accelerating the draft restores overall gains.
  - This adds another conversion of sequential steps into parallel checks, increasing arithmetic intensity twice (draft and oracle).
- Difference from prior work: Prior methods used one draft stage; this work shows a practical two‑stage pipeline.

3) Attention/KV engineering for a speculative tree (Section 3.1)
- What’s new: A concrete method to encode a speculative tree into a single oracle forward pass by:
  - Splitting attention into cross‑attention to history plus masked self‑attention within the batch.
  - Controlling positions and masks to maintain causality across the tree.
  - Managing a separate speculative KV cache and merging accepted slices.
- Why it matters: It operationalizes tree speculation with standard transformer blocks.

4) Performance–entropy perspective (Results discussion, Section 4; Figure 3)
- Insight: Speedups are largest on low‑entropy content (e.g., whitespace/indentation in code) that smaller models or n‑grams can reliably predict, and smaller on dense, high‑entropy content.
- Evidence: Figure 3 visualizes token origins—easy tokens often come from the n‑gram (`draft2`) or small draft; complex decision points (e.g., after “if”) often require the oracle.

Overall, the tree structure is the fundamental innovation; staging is a strong incremental extension that amplifies the benefit.

## 5. Experimental Analysis
- Evaluation methodology (Section 4)
  - Dataset: HumanEval (164 programming prompts).
  - Decoding modes:
    - Deterministic (greedy).
    - Top‑k sampling (k=50, T=1).
  - Baselines:
    - Baseline sequential decoding with the oracle alone.
    - Standard speculative decoding (single draft, single path) per Leviathan et al. (2022) / Chen et al. (2023).
  - Metrics:
    - Throughput in tokens/second (Table 2).
    - Relative memory bandwidth consumption (Table 1).
    - Per‑prompt speedup distributions (Figure 2A for greedy; Figure 2B for top‑k).
    - Qualitative token‑origin visualization (Figure 3).
  - Implementation note: Reported 35% overhead from Python infrastructure, suggesting further speed is possible with optimized code (Section 4).

- Main quantitative results
  - Bandwidth savings (Table 1):
    - Deterministic: relative bandwidth 1.00 (baseline) → 0.31 (speculative) → 0.23 (staged).
    - Top‑k: 1.00 → 0.48 → 0.35.
    - Interpretation: staged speculation uses substantially less memory bandwidth, consistent with the roofline argument in Figure 1 that bandwidth dominates at small batch.
  - Throughput (Table 2):
    - Deterministic: 150 (baseline) → 350 (speculative) → 475 (staged) tokens/s.
      - That is 3.16× over baseline and 1.36× over single‑stage speculative.
    - Top‑k: 150 → 219 → 298 tokens/s.
      - 1.98× over baseline and again 1.36× over speculative.
  - Per‑prompt variability (Figure 2):
    - Speedup varies widely across prompts; staged remains consistently above speculative but with a broad spread in gains.
  - Token origin (Figure 3):
    - Easy tokens (e.g., whitespace) often originate from `draft2` (n‑gram); moderately challenging tokens from the 40M draft; critical tokens from the oracle.

- Do the experiments support the claims?
  - The reported numbers back three core claims:
    - Speedup: 3.16× on greedy decoding and clear gains under sampling (Table 2).
    - Bandwidth reduction: measurable and consistent (Table 1), directly addressing the memory‑bound nature shown in Figure 1.
    - Quality preservation: While there’s no accuracy metric reported (e.g., pass@k on HumanEval), the method theoretically preserves the oracle’s distribution via verification and rejection sampling (Sections 2.3, 4). In deterministic mode, outputs must match the oracle exactly because the oracle verifies each accepted token.
  - Caveats:
    - Evaluation focuses on throughput rather than task accuracy. The distribution‑preserving argument is standard for speculative decoding, but no explicit empirical quality checks are shown for sampling settings on HumanEval (Section 4).

- Ablations and robustness
  - There is no dedicated ablation isolating tree width/depth or the contribution of `draft2`, though the comparison “speculative vs staged speculative” indirectly measures the added stage’s benefit (Table 2).
  - The paper discusses variability and attributes it to content entropy density, providing qualitative reasoning (Section 4, Figure 3).

- Conditions and trade‑offs
  - Gains diminish as sampling randomness increases because more speculative tokens get rejected (Section 4; compare deterministic vs top‑k in Table 2).
  - Benefits are largest on low‑entropy stretches (whitespace/indentation) and smallest on dense, high‑entropy segments (Section 4).

## 6. Limitations and Trade-offs
- Assumptions and applicability
  - Most valuable in small‑batch, memory‑bandwidth‑bound regimes (Figure 1). In large‑batch server settings where arithmetic intensity is already high, relative gains may shrink.
  - Relies on availability of a well‑aligned draft; if the draft is too weak, tree candidates can still be largely rejected, limiting speedups.

- Sampling degradation
  - With stochastic decoding (e.g., top‑k), more speculative tokens are rejected, reducing acceleration (Table 2). This is inherent: randomness increases mismatch between proposals and oracle draws.

- Engineering complexity and overhead
  - Requires careful control of positional embeddings and custom causal masks to encode the tree, plus separate speculative KV cache management (Section 3.1). This adds implementation complexity and potential memory overhead for the extra KV storage during batched checks.
  - Reported 35% Python overhead (Section 4) implies results may depend on implementation efficiency; conversely, optimized kernels could further improve performance.

- Evaluation scope
  - Hardware: Only one GPU class (RTX 4090). Behavior on different memory hierarchies or accelerators (e.g., mobile NPUs) is not evaluated.
  - Models: Oracle is 762M; while the paper argues larger models should benefit more (Section 4), this is not empirically demonstrated here.
  - Quality metrics: No explicit end‑task accuracy or distributional fidelity checks are shown for sampling settings; correctness is argued from the mechanism (Sections 2.3, 4).

- Memory usage
  - Tree width and depth increase the in‑flight activations and speculative KV cache footprint; the paper does not quantify memory ceilings or memory–speed trade‑offs.

## 7. Implications and Future Directions
- How this changes the landscape
  - It reframes speculative decoding as a batching/parallelization problem across a structured search space (a tree), not just a linear guess. This is a general template for turning sequential decoding into batched compute at multiple levels (draft and oracle).
  - By explicitly targeting arithmetic intensity (Figure 1), it clarifies why some inference tricks help and where additional parallelism is needed in small‑batch regimes.

- Follow‑up research enabled or suggested
  - Speculation under stochastic sampling (Section 4, Future Work 1): Use the sampled multinomial CDF to guide which candidates to include in the tree (e.g., if the draw is at 0.99 quantile, skip top tokens and probe lower‑rank options), potentially reducing rejections at temperature > 0.
  - Larger staged stacks (Section 4, Future Work 2): With 8‑bit quantization, explore 20B → 1B → 50M → n‑gram pipelines on consumer GPUs to further amplify gains.
  - Better ultra‑fast `draft2` models (Section 4, Future Work 3): Replace n‑grams with sub‑millisecond neural predictors that outperform trigram models while still running in <10 μs.
  - Kernel‑level integration: Fuse tree‑structured attention and speculative KV management with libraries like FlashAttention and optimized sampling to reduce the current 35% software overhead (Section 4).
  - Adaptive tree shaping: Online policies that adjust tree width/depth based on estimated entropy of the next tokens (Figure 3’s insight) to concentrate computation where the oracle is most likely to agree.

- Practical applications
  - On‑device assistants where latency and privacy are critical (Introduction): code completion, mobile chat, and embedded controllers.
  - Personalization: Faster local inference makes per‑user fine‑tuning or adapters more viable without cloud resources.
  - Edge deployment: Devices constrained by bandwidth rather than compute (e.g., consumer GPUs, some NPUs) can benefit from staged speculative decoding.

Block‑quoted highlights from the paper
- Performance headline:
  > “Taken together, we reduce single-batch decoding latency by 3.16x with a 762M parameter GPT-2-L model while perfectly preserving output quality.” (Abstract)
- Tree rationale:
  > “We restructure the speculative batch as a tree of possible token sequences, so as to more quickly create larger and higher quality speculative batches.” (Section 3.1)
- Staging rationale:
  > “We correspondingly add speculative decoding to the draft model in our approach.” (Section 3.2)
- Bandwidth and throughput gains:
  > “Staged speculative decoding uses substantially less memory bandwidth…” (Table 1) and
  > “…provides an average performance boost of 3.16x over our reference implementation… and 1.36x over standard speculative sampling.” (Section 4; Table 2)
- Variability explained:
  > “Obvious tokens—like whitespace—are preferentially accelerated relative to difficult ones.” (Figure 3 caption and Section 4)

In sum, the paper operationalizes a principled way to trade extra, cheap computation for reduced memory traffic by batching future possibilities as a tree and by stacking speculative stages. The gains are strongest when the next tokens are predictable by smaller models and remain positive—even if reduced—under common sampling settings. Further engineering and broader evaluations could cement staged speculative decoding as a default for on‑device LLMs.
