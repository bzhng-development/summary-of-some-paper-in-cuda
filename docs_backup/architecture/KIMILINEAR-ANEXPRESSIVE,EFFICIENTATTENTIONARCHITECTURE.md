# KIMI LINEAR: AN EXPRESSIVE, EFFICIENT ATTENTION ARCHITECTURE

**ArXiv:** [2510.26692](https://arxiv.org/abs/2510.26692)

## 🎯 Pitch

Kimi Linear introduces Kimi Delta Attention (KDA), a novel hybrid linear attention mechanism that, for the first time, surpasses full attention on language modeling quality while being dramatically more efficient in both speed and memory usage. By combining a fine-grained gating mechanism with a delta-rule–based update, Kimi Linear enables drop-in replacement of quadratic attention for million-token contexts and RL scaling, delivering up to 6× faster decoding and 75% lower key-value cache usage—making efficient, long-context LLMs truly practical without sacrificing performance.

---

## 1. Executive Summary
Kimi Linear introduces a hybrid attention architecture that replaces most full attention layers with a new linear module, Kimi Delta Attention (`KDA`), and shows—for the first time under matched training recipes—that it can outperform full attention in accuracy while being substantially faster and more memory‑efficient across short context, long context (up to 1M tokens), and reinforcement learning (RL) settings. The core significance is that it provides a drop‑in, production‑ready alternative to quadratic attention that scales to million‑token contexts with up to 6× decoding throughput and up to 75% lower KV‑cache usage, without sacrificing quality (Figure 1; §4–§5–§6).

## 2. Context and Motivation
- Problem/gap
  - Full attention has quadratic time in sequence length and requires a key–value (`KV`) cache that grows linearly with the context, making long contexts (e.g., ≥128k) and decoding‑heavy workloads slow and memory‑hungry (§1). This bottleneck becomes acute for agentic LLMs and RL test‑time scaling that process long trajectories and tool-use traces.
  - Linear attention offers linear complexity but historically underperforms on language modeling and retrieval, especially for long or copy-heavy sequences (§1–§2.2). This quality gap has limited its adoption as a general replacement for full attention.
- Why it matters
  - Real-world impact: More efficient inference enables million‑token contexts, larger batch sizes, faster interactive agents, and lower serving cost for long-horizon tasks (§1, Figure 1b, Figure 7).
  - Theoretical significance: The work advances linear attention by combining the delta learning rule (a corrective, “fast-weight” update) with fine-grained memory control, narrowing or surpassing full attention performance while keeping RNN‑like constant‑state inference (§2.2–§3).
- Prior approaches and shortcomings
  - Gating/decay + delta rule improved linear attention: RetNet (fixed decay), Mamba2 (data-dependent scalar decay), DeltaNet (delta rule), Gated DeltaNet (`GDN`) (delta + scalar forget) (§2.2; Table 6).
  - However, finite-state memory still limits perfect retrieval; purely linear models struggle with exact copying and long-range recall (§1, §5.1).
  - Hybrid models (interleaving linear and full attention) existed but typically did not beat full attention across diverse, large-scale benchmarks, or lacked unified, efficient kernels and fair, matched training (§1, §7.2).
- Positioning
  - This work builds a stronger linear module (`KDA`) by making the forget gate channel‑wise (per feature dimension) instead of head‑wise, integrates a hardware‑efficient chunkwise algorithm, and combines it with periodic global NoPE attention in a simple 3:1 ratio. It then validates at scale with matched tokens and near-identical recipes, open-sourcing kernels and models (§3–§4–§5.4–§5.5, Figure 1).

## 3. Technical Approach
The system has two layers of design: a new linear attention operator (`KDA`) and a hybrid model architecture (“Kimi Linear”) that interleaves `KDA` with full attention.

- Kimi Delta Attention (`KDA`): a fine‑grained gated delta rule
  - Intuition
    - Think of a small matrix `S_t` as an associative memory that stores key→value mappings (“fast weights”). At each step, the model both forgets some of the memory and corrects it to better map the current key `k_t` to value `v_t`. The output is produced by querying this memory with `q_t` (§2.2).
  - Core update (Eq. 1)
    - Update memory: `S_t = (I − β_t k_t k_t^T) · Diag(α_t) · S_{t−1} + β_t k_t v_t^T`.
      - `β_t` is a learnable step size (Sigmoid‑bounded to [0,1]).
      - `Diag(α_t)` is a per‑channel (feature‑wise) forget gate where each channel has its own decay in [0,1].
      - The rank‑1 term `(I − β_t k_t k_t^T)` is the delta rule: a corrective, Householder‑like transformation that fixes the memory toward mapping `k_t → v_t` (§2.2, Eq. 1).
    - Output: `o_t = S_t^T q_t` (Eq. 1).
  - Why the fine‑grained (per‑channel) gate matters
    - Prior `GDN` used a single forget scalar per head; `KDA` makes this gate channel‑wise, giving each feature dimension its own time constant. This increases control over what is retained or forgotten, improving copying, recall, and stability (§3; Figure 4).
  - Relation to positional encoding
    - `KDA`’s decayed, corrective transitions can be written as a product of data‑dependent matrices between positions (Eq. 12). This acts like a learnable, multiplicative positional encoding, analogous to RoPE but without fixed rotations (§6.1, Eq. 11–12; Table 6). This motivates using NoPE in the full attention layers so `KDA` carries the positional bias (§4, “NoPE for MLA”; §5.2 “NoPE vs. RoPE”).

- Hardware‑efficient chunkwise algorithm for `KDA`
  - Goal: Parallelize over chunks of length `C` to fully exploit GPU matmuls while keeping the recurrent semantics (§3.1).
  - Key ingredients
    - Chunk re-indexing and partial unrolling (Eq. 2): Unroll `C` steps into a closed form that depends on the initial state of the chunk and a sum of transformed rank‑1 updates.
    - WY representation (Eq. 3–5): Packs products of rank‑1 updates into compact matrices without explicit inverses, reducing overhead and improving numeric stability (§3.1).
    - UT transform (Eq. 6–7): Replaces some scalar FLOPs with triangular solves and matmuls, lowering non‑matmul overhead to get better Tensor Core utilization (§3.1).
    - Chunkwise state/output formulas (Eq. 8–9): Provide a batched, GPU‑friendly way to update the state and compute outputs with an inter‑block recurrent / intra‑block parallel schedule, maximizing matmul throughput (§3.1).
  - Specialized DPLR variant for speed and stability
    - Background: A general Diagonal‑Plus‑Low‑Rank (`DPLR`) transition writes the state transform as `D − a_t b_t^T`. It is expressive but incurs extra matmuls and can require secondary chunking for numeric stability (§3.2, §6.2).
    - `KDA` ties both low-rank vectors to the key (`a=b=k`). This choice:
      - Avoids divisions by cumulative decays that cause instability in intra‑chunk ops.
      - Cuts the number of “second‑level” chunk matmuls from four to two and removes three additional matmuls (§3.2; Listing 8a vs 8b).
      - Yields roughly 2× operator speed vs a general DPLR kernel up to 64k tokens (Figure 2), and “~100% operator efficiency improvement” vs DPLR in analysis (§3.2, §6.2).

- Kimi Linear architecture (how `KDA` is used)
  - Layout
    - Stack blocks with a token‑mixing layer followed by a MoE channel‑mixing layer (Figure 3). Token mixing alternates 3 `KDA` layers with 1 full attention layer (`MLA`, Multi‑Head Latent Attention), i.e., a 3:1 ratio (§4).
  - Positional encoding: NoPE in the MLA layers
    - No positional encoding in global attention; `KDA` carries all positional/recency bias. Benefits: simpler long‑context training (no RoPE tuning), and MLA heads can turn into efficient MQA at inference (§4 “NoPE for MLA”, §5.2 “NoPE vs. RoPE”).
  - Per‑layer parameterization (Section 4)
    - `q,k,v` are produced via short depthwise convolutions + Swish, with `q,k` L2‑normalized for eigenvalue stability.
    - `α_t` (forget gate) uses a low‑rank projection and monotone map to [0,1] per channel.
    - `β_t` uses a Sigmoid.
    - An additional low‑rank output gate (Sigmoid) after head‑wise RMSNorm improves stability and avoids attention sink (Eq. 10; Table 1 ablations).
    - Head dimensions `d_k=d_v=128` in experiments.
  - Inference strategy and complexity
    - Prefill uses the chunked kernel; decoding uses the recurrent update (Eq. 2). The model maintains a fixed state per head (`d_k × d_v`), independent of sequence length, unlike KV caches (§6.3). FLOPs per head scale as `O(T d_h^2 + T C d_h + T C^2)` for `KDA` vs `O(T^2 d_h)` for full attention (Eq. 13–14).

- Implementation notes
  - Kernels are open-sourced with vLLM integration (links in Abstract). The chunked `KDA` pseudo‑code is given in Appendix C (Listing 1), showing the matmul‑heavy path needed for high GPU utilization.

## 4. Key Insights and Innovations
- Fine‑grained, channel‑wise gated delta rule (fundamental)
  - What’s new: `KDA` extends `GDN`’s per‑head decays to per‑channel decays (`Diag(α_t)`) and keeps the delta rule’s corrective update (§3; Eq. 1).
  - Why it matters: More precise memory control increases expressivity without growing state size, enabling better recall/copy behavior and faster convergence on synthetic tasks (Figure 4) and better long‑context performance (§5.1–§5.5).
- Bespoke chunkwise algorithm with specialized DPLR tying (fundamental + systems)
  - What’s new: A numerically stable, matmul‑centric chunked algorithm that combines WY + UT transforms and constrains DPLR to `a=b=k` (§3.1–§3.2; §6.2).
  - Why it matters: Substantially fewer matmuls and no second‑level chunking for divisions, yielding ~2× kernel speed vs general DPLR up to 64k tokens (Figure 2) and enabling throughput at million‑token scales (Figure 1b, Figure 7).
- Simple, effective hybrid recipe: 3 `KDA` : 1 full attention with NoPE (architectural)
  - What’s new: A layerwise interleaving that is infrastructure‑friendly, reduces KV cache up to 75%, and empirically gives the best perplexity among tested ratios (Table 1), while preserving global information flow (§4, §5.2).
  - Why it matters: This combination beats matched full attention baselines in quality across short and long contexts and under RL while substantially improving speed and memory (Figure 1a–b; Table 3–5; Figure 6–7).
- KDA as learnable positional encoding (conceptual)
  - What’s new: A unifying view showing the gated delta recurrence forms a data‑dependent multiplicative positional encoding (Eq. 12; Table 6).
  - Why it matters: Explains why pairing NoPE global attention with `KDA` yields robust long‑context extrapolation and reduces RoPE sensitivity (§6.1; §5.2 “NoPE vs. RoPE”).

## 5. Experimental Analysis
- Evaluation setup (fairness and breadth)
  - Models and training (§5.4)
    - 48B total parameters with MoE (8 of 256 experts active; ~3B activated params). Identical layer counts and heads across baselines.
    - Three matched models: full attention `MLA`, hybrid `GDN‑H` (Gated DeltaNet + MLA), and hybrid `Kimi Linear` (KDA + MLA). A RoPE variant, `Kimi Linear (RoPE)`, isolates positional design effects.
    - Pretraining budget: 1.4T tokens, context 4,096; same optimizer/schedule. SFT and RL use identical recipes across models.
  - Benchmarks (§5.4)
    - Short-context knowledge/reasoning: HellaSwag, ARC‑C, Winogrande, MMLU, MMLU‑Redux/Pro, GPQA‑Diamond, BBH.
    - Math & Code: GSM8K, MATH, AIME 2025, HMMT 2025, PolyMath‑en, LiveCodeBench v6, EvalPlus, CRUXEval.
    - Long-context: RULER (128k), MRCR, HELMET‑ICL, LongBench v2, Frames, RepoQA, Long Code Arena.
    - Chinese: C‑Eval, CMMLU.
    - All generation with temperature 1.0; some tasks evaluated via perplexity (listed in §5.4); GPQA averaged over 8 runs.

- Main quantitative results
  - Synthetic tasks (Figure 4)
    - On Palindrome, MQAR (multi‑query associative recall), and Stack tracking, `KDA` consistently achieves the highest accuracy as sequence length grows (256→2048) and converges faster at 1024‑token training than `GDN`; `Mamba2` fails in this configuration (§5.1).
  - Ablations (Table 1; §5.2)
    - Hybrid ratio: 3:1 `KDA:MLA` gives best train/valid perplexities among {1:1, 3:1, 7:1, 15:1, 0:1 full attention}. Too many linear layers hurt validation; too few increase inference cost.
    - Output gate: Removing it or using Swish hurts performance; Sigmoid is best (aligns with avoiding attention sink).
    - Short convolution: Removing it increases perplexity; local convolutions still help even in hybrid models.
  - Scaling law (Figure 5; §5.3)
    - Across 5 sizes (compute‑optimal training), the fitted loss–compute curve shows ~1.16× compute efficiency gain for Kimi Linear over full attention MLA at the same PFLOP/s‑days.
  - Short‑context pretrain results (Table 3; §5.5.1)
    - Kimi Linear tops most benchmarks at 1.4T tokens. Examples:
      - > MMLU: 73.8 (Kimi Linear) vs 72.2 (MLA) vs 71.6 (GDN‑H).
      - > MMLU‑Pro: 51.0 vs 47.2 vs 47.9.
      - > CRUXEval‑O (CoT): 62.0 vs 61.5 vs 58.1.
      - Small exceptions: EvalPlus slightly favors GDN‑H (63.1) over Kimi Linear (60.2).
  - Instruction‑tuned results (Table 4)
    - Kimi Linear leads broadly after the same SFT:
      - > MMLU‑Redux: 80.3 vs 79.2 (MLA) vs 78.7 (GDN‑H).
      - > GPQA‑Diamond Avg@8: 62.1 vs 57.1 vs 58.6.
      - > LiveCodeBench v6 Pass@1: 26.0 vs 25.1 vs 25.4.
      - Exceptions: EvalPlus (61.0) trails MLA/GDN‑H (~62.5) and MATH500 is slightly lower than GDN‑H (81.2 vs 83.0).
  - Long‑context results at 128k (Table 5)
    - Kimi Linear has the best average (54.5). Notable wins:
      - > RULER: 84.3 (Kimi Linear) vs 81.3 (MLA) vs 80.5 (GDN‑H).
      - > RepoQA: 68.5 vs 63.0 vs 63.0.
    - `Kimi Linear (RoPE)` underperforms Kimi Linear on long context despite similar short‑context scores, supporting the NoPE design (§5.2).
  - RL training on math (Figure 6)
    - Using identical RLVR settings and data, Kimi Linear shows faster and higher accuracy improvements than MLA on the training set and generalizes better on MATH500 and AIME 2025 test curves across training steps (§5.5.1).
  - Efficiency: Prefill and decoding (Figure 7; Figure 1b; §5.6; §6.3)
    - Batch size 1:
      - > Prefill latency at 1M tokens: ~2.9× faster than full attention; matches GDN‑H (Figure 7a).
      - > Decoding TPOT at 1M: ~1.8–2.2× faster than MLA; similar to GDN‑H (Figure 7b).
    - With larger batches made possible by the small, constant state (no large KV cache), decoding TPOT improves up to 6.3× at 1M (1.84 ms vs 11.48 ms; Figure 1b).
  - Extended training (Appendix D)
    - With 5.7T tokens, the released `Kimi‑Linear‑Instruct` reaches RULER 94.8 at 1M context and large gains on code and math over the Moonlight baseline (Table 9). These comparisons involve different total parameterizations (48B vs 16B) and are provided as a capability demonstration.

- Do the experiments support the claims?
  - Yes, on three fronts:
    - Quality: Kimi Linear consistently matches or beats the full attention baseline on a broad set of short‑context and long‑context tasks at the same training tokens (Table 3–5), and improves RL learning curves (Figure 6).
    - Efficiency: Measured kernel speedups vs DPLR (Figure 2), prefilling and decoding speedups vs MLA (Figure 7) and the large‑batch TPOT result (Figure 1b) substantiate the efficiency claims.
    - Design choices: Ablations on hybrid ratio, gates, and NoPE vs RoPE (Table 1; Table 5; §5.2) buttress the architectural decisions.

- Notable caveats
  - Some code evaluations (EvalPlus) are mixed; GDN‑H edges out Kimi Linear in a few cases (Table 3–4).
  - LongBench v2 and Frames do not show clear gains (Table 5), suggesting task‑dependent trade‑offs.

## 6. Limitations and Trade-offs
- Finite‑state constraint and retrieval
  - `KDA` maintains a fixed‑size state per head; exact retrieval/copying over extreme ranges remains challenging for purely linear layers (§1, §7.2). The interleaved full attention layers mitigate but do not eliminate this limitation.
- Design sensitivity
  - Performance depends on the hybrid ratio and gating choices; too many linear layers hurt generalization (Table 1). The NoPE/positional design is important—`Kimi Linear (RoPE)` weakens long‑context results (Table 5).
- Kernel specialization and numerical issues
  - The speedups rely on specialized kernels (WY + UT, tied DPLR). Portability across hardware backends or extreme precision regimes may require additional engineering (§3.1–§3.2; Appendix C).
- Benchmark coverage
  - While broad, some domains still show mixed results (e.g., EvalPlus, some long-context suites), indicating room for robustness improvements (Table 4–5).
- Training scope
  - The strongest “beats full attention” claims are at 1.4T token parity. The 5.7T token results are informative but differ in total parameterization when compared to Moonlight (Appendix D).

## 7. Implications and Future Directions
- How this changes the field
  - Demonstrates that a linear‑dominant hybrid can surpass full attention under matched training, reshaping the default assumption that full attention is necessary for peak quality. It unlocks million‑token contexts with practical throughput and memory footprints (Figure 1; §5.6), directly benefiting long‑horizon tool use, codebase‑level reasoning, and RL at inference time.
- Research directions
  - Combining `KDA` with sparse attention for finer retrieval while keeping a small state (§7.1 discussion).
  - State expansion or mixture‑of‑memories to further close copying/selectivity gaps while maintaining efficiency (§7.1).
  - Theory: deeper analysis of `KDA` as learnable positional encoding and its extrapolation behavior vs RoPE (Table 6; §6.1).
  - Auto‑hybridization: learning the layer ratio or scheduling linear/global layers by depth or domain (§7.2).
- Practical applications
  - Production LLM serving with million‑token contexts (chat assistants, retrieval‑augmented code understanding, repository‑level Q&A).
  - Agentic systems and RL test‑time scaling where decoding speed and memory footprint dominate cost.
  - On‑device or edge deployment where KV cache growth is prohibitive, but a constant‑state linear module is viable.

> Bottom line: With `KDA`’s fine‑grained forget + delta updates, a hardware‑aware chunkwise algorithm, and a simple 3:1 hybrid with NoPE global attention, Kimi Linear achieves a rare combination—better accuracy than full attention at matched training, much faster long‑context decoding (up to 6× with large batches), and drastically reduced memory use—making it a credible, drop‑in replacement for full attention in long‑horizon LLMs (Figure 1; Table 3–5; Figure 7; §6.3).
