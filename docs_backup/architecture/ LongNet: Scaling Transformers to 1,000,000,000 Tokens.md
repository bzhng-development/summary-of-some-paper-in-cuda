# LongNet: Scaling Transformers to 1,000,000,000 Tokens

**ArXiv:** [2307.02486](https://arxiv.org/abs/2307.02486)
**Authors:** Jiayu Ding, Shuming Ma, Li Dong, Xingxing Zhang, Shaohan Huang, Wenhui Wang, Nanning Zheng, Furu Wei
**Institutions:** Microsoft Research (likely affiliation for authors)

## 🎯 Pitch

LONGNET introduces the dilated attention mechanism, transforming Transformer models to handle sequences up to one billion tokens with nearly linear compute growth, vastly improving their ability to leverage long contexts without sacrificing short-context performance. This innovation enables more robust models for tasks with extensive data, paving the way for advancements in fields like AI-driven legal analysis, genomics, and long-form multi-modal interactions, where understanding large-scale information is crucial.

---

## 1. Executive Summary
LONGNET introduces a new attention module, called `dilated attention`, that lets Transformer models process sequences up to one billion tokens while keeping compute nearly linear in sequence length. It achieves this by combining several sparse, segment-based attention patterns whose coverage grows exponentially with distance, plus a distributed training scheme that parallelizes across the sequence dimension with almost constant communication cost.

## 2. Context and Motivation
- Problem addressed
  - Standard self-attention has quadratic runtime and memory in sequence length N, which makes very long contexts impractical. This limits language models’ ability to remember and reason over distant information (Section 1).
- Why it matters
  - Longer contexts mean:
    - richer memory and receptive field for interactive agents,
    - exposure to longer causal chains during training (potentially better generalization),
    - the possibility of many-shot in-context learning without catastrophic forgetting (Section 1).
- Where prior approaches fall short
  - RNN-style and state-space models: good for long range but either sequentially trained (hurts parallelism) or empirically weaker on standard-length language modeling (Section 1; references [GGR22, SWL23, FDS+23, FPB+23]).
  - Efficient Transformer variants:
    - Local windows/convolutions: near-linear, but forget the earliest tokens because they don’t connect distant positions (Section 1).
    - Sparse/low-rank/kernel approximations: better efficiency, but none scales to 1B tokens and often require custom kernels or trade off accuracy (Section 1; Table 1; references).
- Positioning relative to existing work
  - LONGNET keeps Transformer-style attention and optimization tooling, but changes the attention pattern to be exponentially expanding with distance. It targets both compute efficiency (linear in N) and global connectivity (logarithmic path length between any two tokens), enabling distributed sequence-parallel training up to 1B tokens (Sections 2–3; Figure 5).

## 3. Technical Approach
At a high level, LONGNET replaces standard attention with a mixture of sparse attention patterns that:
- compute exact local attention,
- progressively sample fewer positions as distance grows (exponentially),
- and combine them into one output using a theoretically grounded softmax merge.

Step-by-step:

1) Start from standard self-attention
- Inputs `Q, K, V ∈ R^{N×d}`; output is `O = softmax(QK^T) V` (Equation 1).

2) Sparse attention as a lens
- A generic sparse pattern can be represented as masking entries in the attention matrix, `O = softmax(QK^T ⊙ 1_S) V` (Equation 2), where `S` indicates which pairs are allowed to attend. LONGNET defines a new family of such patterns with strong coverage but low cost.

3) Dilated attention (new pattern family)
- Idea: split the sequence into equal-length segments of size `w`, and within each segment “sparsify” by keeping every `r`-th row/column. Here:
  - `w` is the `segment length`,
  - `r` is the `dilation rate` (keep tokens at interval `r`; larger `r` = sparser).
- Construction (Equations 3–5):
  - Within each segment i, pick `Q̃_i, K̃_i, Ṽ_i` by selecting rows/positions at strides of `r`.
- Compute attention per segment on the downsampled tensors, then scatter the resulting outputs back to the original positions (Equations 6–8).
- Systems detail: the operation reduces to a gather (select rows), a dense attention call (e.g., with FlashAttention), and a scatter (insert outputs), so existing, optimized attention kernels can be reused (Section 2.2).

4) Mixture of multiple dilations and segments
- To capture both local details and far-away context, use several `(w_i, r_i)` settings in parallel and mix their outputs:
  - `O = sum_i α_i O|_{r_i, w_i}` (Equation 9).
  - Weights `α_i` are computed from the softmax normalizers `s_i` (“denominator of the attention softmax”) so that mixing is equivalent to having gathered all keys across the patterns and doing one softmax (Equations 9–10; Section 2.2). This preserves probabilistic correctness and avoids tuning extra mixing parameters.
- Design choice: larger segments go with larger dilations—exact local attention and approximate global attention. Segment sizes `w` and dilations `r` increase geometrically (Equations 11–12), so the “attentive field” grows exponentially with distance.

5) Multi-head coverage
- Different attention heads sample different offsets: for head `j`, offset `s_j = j mod r` shifts which rows are kept (Equations 13–15; Figure 3). Across heads, this fills in gaps and increases coverage without adding compute.

6) Complexity and connectivity guarantees
- Single pattern flops: `FLOPs = 2N w d / r^2` (Equation 16) after sparsification from `w × w` to `(w/r) × (w/r)`.
- With k geometric patterns, total flops sum to `O(N d)` (Equation 18). This achieves linear complexity in sequence length.
- Connectivity: the maximum “path length” (number of hops needed for information to flow between tokens) scales as `O(log N)` because each pattern expands the reachable distance by a multiplicative factor `α`, and you need about `log_α N` layers/patterns to connect endpoints (Equations 19–20; Section 2.4). Intuition: exact local links plus exponentially expanding “express lanes” reach far tokens quickly.

7) Distributed sequence-parallel training
- Motivation: even linear-time attention is too big for million-to-billion token sequences on a single device; memory and compute must be spread across GPUs.
- Algorithm (Figure 4; Section 3.1):
  - Split the input along the sequence dimension across devices: `X = [X1, X2, ...]` (Equation 21).
  - Each device computes its local projections `Q_i, K_i, V_i` (Equation 22).
  - For patterns whose segment fits locally (`w_i ≤` local length `l`), compute attention locally.
  - For global patterns (`w_i > l`), first sparsify to `K̃_i, Ṽ_i`, then `all-gather` only these sparsified keys/values across devices (Equation 23). Importantly, the size of `K̃_i, Ṽ_i` is independent of total `N` because sparsification has already reduced them. In backward pass, the collective reduces to `reduce-scatter` (Section 3.1).
  - Compute cross-attention with local queries and gathered global keys/values, then concatenate outputs (Equations 24–25).
- Communication is almost constant with more devices because what is communicated is the already-sparsified `K̃, Ṽ`, not full-length tensors (Section 3.1).

8) Practical scaling
- Using FlashAttention kernels under the hood, the runtime grows slowly and remains near-constant when scaling sequence length from 8K to 1B tokens (Figure 5). This is a systems-level “feasibility” result demonstrating end-to-end forward throughput, not a fully trained 1B-token language model.

Analogy
- Think of attention as a road network. Local roads (exact attention) densely connect nearby neighborhoods. As you travel farther, you switch to expressways (dilated attention) that skip many intermediate exits. Multiple expressways with increasing speed limits (larger segments and dilations) let you cross the map in a few hops. Multi-head offsets ensure different roads cover different lanes so the city remains well connected.

## 4. Key Insights and Innovations
- Dilated attention as a drop-in, exponentially expanding sparse pattern
  - What’s new: attention is computed exactly locally but increasingly sparsified with distance using geometric segment/dilation schedules (Equations 11–12; Figures 2–3).
  - Why it matters: preserves the ability to reach any token with `O(log N)` hops (Equation 20) while keeping total compute linear in N (Equation 18).
- Softmax-consistent mixing of multiple attention patterns
  - What’s new: mix multiple dilated attentions via the softmax denominators `s_i` (Equations 9–10). This equals doing a single softmax over the union of keys from all patterns, but implemented efficiently with parallel kernels (Section 2.2).
  - Why it matters: removes the need for hand-tuned mixing weights and retains the probabilistic correctness of attention.
- Sequence-parallel distributed algorithm with constant communication
  - What’s new: split by sequence dimension, sparsify before communication, and all-gather only the compacted `K̃, Ṽ` (Figure 4; Section 3.1).
  - Why it matters: enables scaling to billion-token contexts without quadratic networking or memory blow-up (Figure 5).
- Engineering compatibility
  - What’s new: implemented as gather → dense attention (FlashAttention) → scatter; supports standard Transformer optimizations such as kernel fusion and quantization (Section 2.2).
  - Why it matters: lowers adoption barrier—no bespoke attention kernels are required beyond common ones.

Fundamental vs. incremental:
- Fundamental: the combination of exponential attentive field, linear compute, and log-depth connectivity; the sequence-parallel algorithm with constant communication.
- Incremental: multi-head offsetting and FlashAttention integration are effective engineering choices that improve coverage and practicality.

## 5. Experimental Analysis
Evaluation design
- Tasks and data
  - Autoregressive language modeling on The Stack, a large code dataset in 300+ languages (Section 4.1; [KLA+22]).
- Model backbone and training setup
  - Use MAGNETO base config with XPOS (relative positions): 12 layers, hidden 768, 12 heads, FFN 3072 (Section 4.1).
  - Train with 500K tokens per batch for 300K steps; other hyperparameters in Table 3 (Appendix A).
  - LONGNET replaces standard attention with dilated attention; FlashAttention kernels are used for speed/memory across all methods (Sections 4.1–4.2).
- Baselines
  - Vanilla Transformer (dense attention; limited to 32K due to cost).
  - Sparse Transformer with fixed local+strided patterns (as in [CGRS19]); its sparse ratio is tuned to match FLOPs with LONGNET for fairness (Section 4.2).
- Metrics
  - Perplexity (PPL) on test splits with inputs of 2K, 8K, and 32K tokens; when testing beyond a model’s training window, use blockwise causal attention (BCA) for extrapolation (Section 4.2; [SDP+22]).

Main quantitative results
- Perplexity vs. training context length (Table 2)
  - Models trained to 8K context:
    - Sparse Transformer: PPL 4.39 (2K), 3.35 (8K), 8.79 (32K).
    - LONGNET: PPL 4.23 (2K), 3.24 (8K), 3.36 (32K).
  - Models trained to 16K context:
    - Sparse Transformer: 4.85 (2K), 3.73 (8K), 19.77 (32K).
    - LONGNET: 4.27 (2K), 3.26 (8K), 3.31 (32K).
  - Models trained to 32K context:
    - Sparse Transformer: 5.15 (2K), 4.00 (8K), 3.64 (32K).
    - LONGNET: 4.37 (2K), 3.33 (8K), 3.01 (32K).
  - Dense Transformer trained to 2K:
    - 4.24 (2K), 5.07 (8K), 11.29 (32K).
  - Takeaways:
    - LONGNET consistently beats the sparse baseline at equal compute across all tested lengths.
    - LONGNET does not sacrifice short-context performance: at 2K it is close to the 2K-trained dense Transformer (4.23–4.37 vs. 4.24).
    - Extrapolating far beyond training length with BCA degrades PPL sharply for baselines (e.g., 8.79 and 19.77 at 32K for models trained to 8K and 16K), while LONGNET’s 32K PPL stays low (3.31–3.36), suggesting better long-range modeling.

- Compute–performance scaling with longer training contexts (Figure 6)
  - Figure 6 plots test PPL vs total FLOPs for different training context lengths from 2K to 32K.
  - Quote:
    > “LONGNET outperforms dense Transformers with a lower perplexity and a significantly smaller amount of computation.”
  - Interpretation: for the same or less compute, increasing context length yields larger gains for LONGNET than for dense Transformers; the curves suggest more compute-efficient scaling when training with long contexts.

- Model size scaling (Figure 7a)
  - Models from 125M to 2.7B parameters follow a near power-law decrease in test loss vs compute.
  - Training tokens: ~40B for 125M–760M; 300B for 2.7B (Section 4.4).
  - Quote:
    > “The scaling curve follows a similar law to the vanilla Transformers.”
  - Interpretation: LONGNET retains the favorable scaling behavior commonly observed in dense architectures.

- Long-context prompting (Figure 7b)
  - Fix the suffix to evaluate; increase prefix length from 1K to 32K used as a prompt.
  - Quote:
    > “A longer context window yields better language modeling.”
  - Interpretation: LONGNET effectively uses more context during inference; test loss decreases monotonically with prompt length up to 32K.

- Systems scaling to 1B tokens (Figure 5; Section 3.2)
  - Runtime of LONGNET with FlashAttention remains nearly flat as sequence length increases from 8K to 1B by distributing across GPUs and keeping communication small and constant-sized.
  - In contrast, dense attention’s runtime grows rapidly with N (quadratic trend).
  - This is a performance/feasibility demonstration of forward passes rather than end-to-end training at 1B sequence length.

Ablations and robustness
- Mixing strategy: Section 2.2 notes that using softmax denominators to weight each dilated attention yields better results than fixed learnable weights; while no separate ablation table is shown, the method is justified as equivalent to a single softmax over the union of keys.
- Extrapolation: Section 4.2 shows that simple blockwise causal attention during inference cannot compensate for insufficient training lengths; longer training contexts are more effective, especially under LONGNET (Figure 6).

Overall assessment
- The experiments support the central claims:
  - LONGNET maintains or improves short-context performance while dramatically improving long-context perplexity at the same compute (Table 2).
  - It scales more compute-efficiently with longer training contexts (Figure 6).
  - It exhibits standard scaling-law behavior with model size (Figure 7a).
  - It can utilize longer prompts (Figure 7b).
  - The systems result (Figure 5) demonstrates plausible path to billion-token contexts.

## 6. Limitations and Trade-offs
- Approximation of global attention
  - LONGNET computes exact local attention but approximates global interactions via sparse sampling. Tasks requiring precise token-to-token alignment at extreme distances may still prefer denser global patterns. The paper evaluates language modeling but not specialized long-range benchmarks (e.g., retrieval-heavy tasks) in this version.
- Training at 1B-token sequence length is not shown
  - Figure 5 demonstrates near-constant forward runtime up to 1B tokens via distribution and sparsification, but end-to-end training to convergence at that length is not reported. This is a feasibility-of-runtime result rather than a full training study (Section 3.2).
- Hyperparameter choices for patterns
  - Segment lengths `w` and dilation rates `r` are chosen as geometric sequences and paired heuristically (larger `w` with larger `r`) (Equations 11–12). Although well-motivated, the paper does not explore automated selection or sensitivity analysis across many tasks.
- Communication and kernel assumptions
  - The constant communication cost relies on sparsifying before collective ops and using efficient kernels (FlashAttention) (Sections 2.2, 3.1). Performance depends on high-quality kernel implementations and network fabric; real-world throughput may vary across hardware.
- Evaluation scope
  - Main empirical results focus on code modeling (The Stack) and context windows up to 32K for perplexity; broader domains (long documents, genomics, multimodal streams) are discussed as future directions, not extensively evaluated here (Section 5 Conclusion & Future Work).

## 7. Implications and Future Directions
- Field impact
  - LONGNET reframes “context length scaling” from quadratic to essentially linear compute with logarithmic dependency between any two tokens. This opens the door to treating very large collections—even an entire corpus or the Internet—as a single sequence for training or inference (Abstract; Sections 1–2).
- Follow-up research enabled
  - Pattern learning: learn or adapt `(w, r)` per layer, head, or token content instead of using fixed geometric schedules.
  - Hybrid models: combine dilated attention with retrieval, memory modules, or state-space layers to balance precision and scale.
  - Optimization: co-design with kernel and networking enhancements; explore quantization and sparsity-aware compilers for even larger contexts.
  - Theoretical analysis: tighter bounds on approximation error vs. dilation; conditions under which `O(log N)` path length suffices for exact tasks.
  - Benchmarks: comprehensive long-range evaluation (e.g., reasoning over books, long codebases, scientific articles), beyond perplexity.
- Practical applications
  - Long-form assistants that maintain session memory over months, code models that see entire repositories, legal or biomedical review over thousands of pages, continuous multimodal streams (speech, video) where context is hours long, and genomics where sequences are naturally millions of tokens (Conclusion; references to multimodal and BEiT extensions).

In short, LONGNET’s dilated attention and sequence-parallel training contribute a scalable, practical path to billion-token contexts, while preserving compatibility with existing Transformer toolchains and delivering strong empirical performance on language modeling.
