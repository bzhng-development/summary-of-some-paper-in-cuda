# KIMI LINEAR: AN EXPRESSIVE, EFFICIENT ATTENTION ARCHITECTURE

**ArXiv:** [2510.26692](https://arxiv.org/abs/2510.26692)

## 🎯 Pitch

Kimi Linear introduces a hybrid attention architecture that combines a novel, fine-grained linear attention module (Kimi Delta Attention, KDA) with periodic full-attention layers, achieving a 3:1 ratio. This approach not only matches or surpasses conventional full-attention models across short-context, long-context (up to 1M tokens), and RL tasks, but it does so with up to 75% less memory usage and up to 6× faster decoding. By dramatically improving both efficiency and performance, Kimi Linear enables the next generation of large language models to scale to longer contexts and higher throughput, unlocking new capabilities for real-world, agentic, and resource-constrained scenarios.

---

## 1. Executive Summary (2-3 sentences)
Kimi Linear introduces a hybrid attention architecture that interleaves a new linear-attention module called `Kimi Delta Attention (KDA)` with periodic full-attention layers at a 3:1 ratio. It consistently outperforms strong full-attention baselines at the same training compute and data across short-context, long-context (up to 1M tokens), and reinforcement‑learning (RL) settings, while cutting key–value (KV) cache memory by up to 75% and achieving up to 6× faster decoding at 1M context length (Figures 1 and 7).

## 2. Context and Motivation
- Problem addressed
  - Standard full attention scales quadratically in time and linearly in memory with sequence length, which becomes the bottleneck for long-horizon inference and RL test-time scaling where models must process long trajectories and produce long outputs (§1).
  - Prior “linear attention” methods reduce complexity but historically underperform on language modeling and long-range recall, mainly due to limited expressivity and the finite-state nature of their recurrent memory (§1, §2.2).
- Why it matters
  - Long-context processing is increasingly required for agentic LLMs, codebase-level reasoning, and RL-style test-time scaling (§1). Efficiency improvements directly unlock longer contexts, higher throughput, and larger batches in production systems.
- Prior approaches and gaps
  - Linear attention iterations:
    - Gating/decay mechanisms (e.g., RetNet, Mamba/Mamba2) introduce learnable forgetting but have limited precision/selectivity, often using coarse (per-head) decay (§1, §2.2).
    - Delta-rule–based methods (e.g., DeltaNet, Gated DeltaNet (GDN)) treat the state as a learnable associative memory updated by rank-1 corrections, improving stability and enabling chunkwise parallelism, but GDN uses only a scalar forget gate (§2.2).
    - DPLR (Diagonal-Plus-Low-Rank) structures are expressive but computationally heavy and less parallelizable (§6.2).
  - Hybrid models combining linear and full attention exist, but they have been limited in scale or breadth of evaluation; it remained unclear whether they can surpass full attention under matched training recipes (§1, §7.2).
- Positioning of this work
  - Kimi Linear provides a principled hybrid design that, for the first time under matched training (1.4T tokens), surpasses full attention across short/long-context and RL regimes. It hinges on a more expressive and hardware-efficient linear module (`KDA`) plus a simple, deployment-friendly interleaving strategy (3:1 linear-to-full attention) with NoPE (no positional encoding) in full-attention layers (§1, §4, §5).

## 3. Technical Approach
Kimi Linear combines a new linear attention module (`KDA`) with periodic full-attention layers (`MLA`, Multi-Head Latent Attention [19]) in a 3:1 stack (§4). The core ideas are (1) make the linear module more expressive and stable via fine-grained gating and a delta-rule update, and (2) implement a bespoke, chunkwise-parallel algorithm that is both mathematically sound and accelerator-friendly.

- What `KDA` computes (intuitively)
  - Maintain a compact, per-head, matrix-valued “fast-weight” state `S_t` that stores associations between keys and values.
  - At each token t, update `S_t` by (a) decaying the previous state feature-wise (“forget what’s irrelevant”) and (b) applying a rank‑1 corrective update using the current key/value pair (“learn what matters now”), then read out with the current query to get the output (§3, Eq. (1)).
  - Unlike previous GDN which has one forget scalar per head, `KDA` uses a diagonal, channel-wise forget gate `Diag(α_t)` (one learnable forget per feature dimension), yielding finer control over memory and implicit positional structure (§3, Eq. (1), §6.1).

- The update rule in notation (then unpacked)
  - Equation (1):
    - `S_t = (I − β_t k_t k_t^⊤) · Diag(α_t) · S_{t−1} + β_t k_t v_t^⊤`
    - `o_t = S_t^⊤ q_t`
  - Interpretation:
    - `Diag(α_t)`: per-feature forgetting (fine-grained decay).
    - `(I − β_t k_t k_t^⊤)`: a rank‑1 “delta” correction that stabilizes learning and focuses the state on mapping `k_t → v_t` (§2.2 “DeltaNet”).
    - `β_t`: a learned step size (Sigmoid output; §4).
    - The readout multiplies the updated state by the current query `q_t` to produce the token output.

- Chunkwise-parallel algorithm for speed and stability
  - Sequences are split into chunks of length `C`. Within a chunk, the sequence of rank‑1 updates can be “packed” into compact matrix forms using the `WY` representation (a standard trick for products of Householder-like updates) to avoid repeated matrix inversions (§3.1; Eqs. (2)–(5)).
  - The `UT transform` (a triangular transform; Eq. (6)) reduces non-matrix-multiply FLOPs so the algorithm better utilizes Tensor Cores (§3.1).
  - Equations (8)–(9) give the chunkwise state update and output computation. They combine:
    - Inter-chunk recurrence: the state that flows across chunks.
    - Intra-chunk parallelism: triangular operations within a chunk that allow batched, fused matmuls.
  - The implementation avoids numerically unstable divisions often needed for fine-grained decays by coupling certain variables (details below) (§3.2, §6.2).

- Why `KDA` is more efficient than general DPLR
  - General DPLR writes the transition as `S_t = (D − a_t b_t^⊤) S_{t−1} + k_t v_t^⊤`. It is expressive but expensive: it tends to require more “secondary chunking” and log-domain tricks for numerical stability, which reduce throughput (§3.2, §6.2).
  - KDA ties `a_t` and `b_t` to `k_t` (i.e., `a = b = k` after factoring out the shared decay), which:
    - Eliminates additional matrix multiplications at both inter-chunk update and output stages.
    - Cuts the number of second-level chunk computations roughly in half (§3.2; pseudocode comparison in Listings 8a vs. 8b).
  - Result: “roughly 100%” operator-level speedup versus a general DPLR kernel and ~2× faster kernels up to 64k tokens (Figure 2).

- Model architecture and parameterization (§4)
  - Token mixing:
    - For each head, queries/keys/values are constructed by a small `ShortConv` (captures local token patterns) + `Swish`, with `q,k` L2‑normalized for eigenvalue stability (as in [112]) and `v` unnormalized (§4, “Neural Parameterization”).
    - Forget gate `α_t` is generated by a low‑rank projection `W↓_α, W↑_α` with a nonlinearity `f(·)` similar to prior gated models; the step size `β_t` uses Sigmoid (§4).
    - Output uses head-wise `RMSNorm` and a low‑rank Sigmoid `output gate` (`W↓_g, W↑_g`), which improves stability and combats attention sink (Eq. (10), §5.2).
  - Hybrid stacking:
    - Interleave 3 KDA layers with 1 full-attention `MLA` layer (3:1). This preserves global communication while keeping most layers fast and memory-constant (§4).
    - Use `NoPE` (no positional encoding) in full-attention layers so `KDA` provides the positional bias. Benefits: simpler long-context scaling, easy conversion to efficient MQA at inference, and better extrapolation (§4 “NoPE for MLA Layers”, §6.1).
  - Mixture-of-Experts (MoE) backbone similar to Moonlight [62]; 8/256 experts active for the 48B total/3B active-parameter configuration used in large-scale experiments (§5.4).

- Complexity and inference strategy (§6.3)
  - Per-head FLOPs for `KDA` with chunk size `C=64`:
    > FLOPs_KDA(T; C, d_h) = 6 T d_h^2 + 3 T C d_h + T C^2  (Eq. (13))
  - Full attention per head:
    > FLOPs_Attn(T; d_h) = 2 T^2 d_h  (Eq. (14))
  - Inference:
    - Prefill (processing the input context) uses the chunked kernel; decoding (autoregressive generation) switches to the lighter recurrent kernel (§6.3).
    - KDA’s state per head is fixed-size (`d_k × d_v`, 128×128 in experiments), so memory does not grow with sequence length; this shrinks KV cache needs by up to 75% in the 3:1 hybrid (§1, §6.3).

- How KDA acts as a position encoder (§6.1)
  - The gated delta rule can be written as a product of transition matrices between positions (Eq. (12)), similar in form to how RoPE injects relative positions (Eq. (11)).
  - Because `KDA` uses data-dependent, non-orthogonal transitions with per-dimension decays (`Diag(α_t)`), it supplies a learnable, fine-grained positional bias—arguably more flexible than the fixed-frequency rotations of RoPE (§6.1, Table 6).

## 4. Key Insights and Innovations
- Fine-grained, channel-wise gating inside a delta-rule linear attention (`KDA`) is the main conceptual advance (§3).
  - What’s new: replace the scalar per-head forget gate in GDN with `Diag(α_t)`—a separate forget for each feature—and keep the stabilizing delta update `(I − β_t k_t k_t^⊤)`.
  - Why it matters: greater control over what to forget/retain in the fixed-size state, improving copying/recall behaviors and convergence on synthetic memory tasks (Figure 4) and lifting overall language modeling quality under matched training (§5.1, §5.5.1).
- A bespoke, chunkwise-parallel algorithm specialized for the `KDA` structure (§3.1–§3.2).
  - What’s new: pack many rank‑1 updates with `WY` representation and reduce scalar FLOPs via the `UT` transform; tie variables so log-domain and second-level chunking can be avoided or reduced (§3.1–§3.2; Listings 8a vs. 8b).
  - Why it matters: ~2× kernel speedup over general DPLR across lengths up to 64k (Figure 2), enabling the large observed system-level speedups (Figures 1 and 7).
- A simple, deployment-friendly hybrid: 3 KDA layers to 1 full `MLA` layer with `NoPE` in `MLA` (§4, §5.2).
  - What’s new: a fixed, layer-wise pattern that yields the best trade-off in ablations (Table 1) and makes full-attention layers easier to optimize and serve (e.g., convert to MQA).
  - Why it matters: achieves higher accuracy than full attention across benchmarks while reducing KV memory and improving speed at long contexts (Tables 3–5; Figures 1 and 7).
- Interpreting gated delta linear attention as a learnable positional encoding (§6.1).
  - What’s new: formal equivalence that places `KDA` alongside RoPE and other mechanisms in a unified view (Table 6; Eqs. (11)–(12)).
  - Why it matters: supports the design choice to use `NoPE` in full-attention layers and rely on `KDA` for position—improving long-context extrapolation and simplifying scaling (§4, §5.2 “NoPE vs. RoPE”).

## 5. Experimental Analysis
- Evaluation setup (§5.4)
  - Three models trained on the same 1.4T-token corpus and recipe with 4k context: `MLA` (full attention), `GDN-H` (hybrid using Gated DeltaNet), and `Kimi Linear` (hybrid using KDA). All share the same MoE backbone with 48B total and 3B activated parameters (§5.4).
  - Benchmarks span: general knowledge/reasoning (e.g., MMLU, MMLU-Pro, GPQA-Diamond, BBH), math & code (e.g., AIME 2025, MATH500, LiveCodeBench), long-context (RULER, MRCR, Frames, HELMET-ICL, RepoQA, Long Code Arena, LongBench v2), and Chinese tasks (C-Eval, CMMLU) (§5.4).
  - Evaluation settings: temperature 1.0; consistent harness; Avg@k where high variance; perplexity evaluation for some base-model tasks (§5.4).

- Synthetic tasks (Figure 4; §5.1)
  - Tasks: Palindrome (copying), Multi-Query Associative Recall (MQAR), and Stack (state tracking across 64 stacks).
  - Results:
    - `KDA` reaches the highest accuracy across sequence lengths 256–2,048 on all three tasks.
    - Convergence at 1,024 tokens is faster than `GDN`; `Mamba2` fails under these settings.
  - Takeaway: fine-grained gating improves the ability to manage finite memory while preserving crucial information.

- Pretraining results at 1.4T tokens (Table 3; §5.5.1)
  - General knowledge:
    > MMLU-Pro: 51.0 (`Kimi Linear`) vs. 47.9 (`GDN-H`) vs. 47.2 (`MLA`)  
    > MMLU: 73.8 vs. 72.2 vs. 71.6  
    > BBH: 72.9 vs. 70.6 vs. 71.6  
  - Math & code:
    > GSM8K: 83.9 (`Kimi Linear`) vs. 81.7 (`GDN-H`) vs. 83.7 (`MLA`)  
    > CRUXEval-O-cot: 62.0 vs. 58.1 vs. 61.5  
    > EvalPlus: 60.2 (`Kimi Linear`) slightly below `GDN-H` 63.1
  - Chinese:
    > CMMLU: 80.8 vs. 80.7 vs. 79.5; C‑Eval: 79.5 vs. 79.1 vs. 79.3
  - Conclusion: under identical training, the KDA-based hybrid consistently beats full attention (and generally GDN-H), with a small trade-off on EvalPlus.

- SFT (instruction tuning) results (Table 4; §5.5.1)
  - General:
    > MMLU: 77.0 (`Kimi Linear`) vs. 75.6 (`GDN-H`) vs. 75.7 (`MLA`)  
    > MMLU-Pro: 67.4 vs. 64.8 vs. 65.7  
    > GPQA-Diamond Avg@8: 62.1 vs. 58.6 vs. 57.1
  - Math & code:
    > AIME 2025: 21.3 (`Kimi Linear`) vs. 21.1 (`GDN-H`) vs. 20.6 (`MLA`)  
    > HMMT 2025: 12.5 vs. 11.3 vs. 11.3  
    > LiveCodeBench v6 Pass@1: 26.0 vs. 25.4 vs. 25.1  
    > MATH500: 81.2 (`Kimi Linear`) slightly below `GDN-H` 83.0  
    > EvalPlus: 61.0 (`Kimi Linear`) slightly below both.
  - Conclusion: the hybrid with KDA remains the best overall after SFT, with minor dips on specific code metrics.

- Long-context performance at 128k (Table 5; §5.5.1)
  - Average across long-context suites:
    > `Kimi Linear`: 54.5 (best) vs. `MLA`: 52.2 vs. `GDN-H`: 51.2; `Kimi Linear (RoPE)`: 51.8
  - Highlights:
    > RULER: 84.3 (`Kimi Linear`) vs. 81.3 (`MLA`)  
    > RepoQA: 68.5 (`Kimi Linear`) vs. 63.0 (`MLA`)  
    > MRCR: 29.6 (`Kimi Linear`) vs. 22.6 (`MLA`)
  - Observation: using `NoPE` in full-attention layers and delegating positional bias to KDA yields stronger long-context extrapolation.

- RL efficiency (Figure 6; §5.5.1)
  - Setup: RLVR on in-house math training set; evaluate on AIME 2025 and MATH500.
  - Result: `Kimi Linear` improves training accuracy faster and to higher levels than `MLA`; test curves show consistent gains.
  - Interpretation: under reasoning-heavy, long-form generation with RL, the KDA hybrid is more sample/compute efficient.

- Efficiency measurements (Figures 1 and 7; §5.6, §1)
  - Prefill (batch size 1):
    > At 1M tokens, `Kimi Linear` latency is 2.9× lower than `MLA`; at 512k, 2.3× (§5.6, Figure 7a).
  - Decoding (TPOT, batch size 1):
    > At 1M tokens, `Kimi Linear` achieves up to 6× lower TPOT than full attention (§1 figure caption; Figure 7b shows 2.2–1.8× vs. `MLA`/`GDN-H` at shown batch-1 TPOT; the 6.3× comes from enabling larger batches due to smaller KV cache).
  - Kernel-level:
    > KDA kernel is ~2× faster than DPLR up to 64k (Figure 2).

- Ablations (Table 1; §5.2)
  - Hybrid ratio: 3:1 delivers the best training and validation perplexities; 7:1 hurts generalization; 1:1 increases inference cost without accuracy gains.
  - Output gate: removing it or using `Swish` instead of `Sigmoid` degrades performance.
  - Convolution layer: removing ShortConv modestly hurts perplexity—local patterns are still useful.

- Scaling law study (Figure 5; Table 2; §5.3)
  - Five MoE sizes trained compute‑optimally. Fitted curves show:
    > ~1.16× better compute efficiency for Kimi Linear vs. MLA (Figure 5).
  - Suggests KDA hybrids retain advantages as models scale.

- Extended training (Appendix D; Table 8–9)
  - With 5.7T tokens, the released `Kimi-Linear-48B-A3B-Instruct` notably surpasses `Moonlight` on most tasks; long-context scores are very high (RULER@1M = 94.8).

Overall, the experiments are broad (short-context, long-context, RL), use controlled training recipes, and include ablations and kernel benchmarks—together they convincingly support the claims of both quality and efficiency.

## 6. Limitations and Trade-offs
- Finite-state memory remains: linear attention compresses history into a fixed-size state; exact copying and fine-grained retrieval remains theoretically challenging, motivating the inclusion of periodic full-attention layers (§1, §7.1, §7.2). Mixed results on some long-context subsets (e.g., LongBench V2 and Frames where `MLA` is competitive; Table 5).
- Expressivity constraints from specialization:
  - The DPLR simplification (tying `a = b = k`) is key for speed, but may limit the class of transitions compared to a fully general DPLR update (§6.2). The paper prioritizes efficiency over the most general parameterization.
- Positional encoding reliance:
  - Full-attention layers use `NoPE`, pushing all positional bias into `KDA`. If `KDA` is poorly trained or misconfigured, global layers lack positional guidance. The RoPE alternative improves short-context in some cases but weakens long-context extrapolation (Table 5, “NoPE vs. RoPE” discussion in §5.2).
- System complexity:
  - The chunkwise algorithm involves triangular operations and specialized kernels (UT transform, WY packing), which require careful engineering for stability and throughput (§3.1). Broad framework support may lag compared to standard attention (though vLLM and kernels are released).
- Evaluation scope:
  - RL comparisons center on mathematics datasets (§5.5.1). It remains to be seen how advantages translate to other RL task families (e.g., tool use, planning) under varying rollout/inference conditions.

## 7. Implications and Future Directions
- How this changes the landscape
  - Demonstrates that a linear–full hybrid can surpass full attention in accuracy while dramatically improving long-context efficiency under matched training. This reframes the default assumption that “full attention is needed for quality,” especially for million-token contexts and decoding-heavy workloads (§1, §5).
  - Provides a practical recipe—3:1 KDA:full attention with `NoPE`—that integrates into standard inference stacks (vLLM), enabling immediate memory and speed gains without changing serving interfaces (§1, §5.6).
- What follow-up research it enables
  - Architecture:
    - Explore other ratios or adaptive schedules (e.g., more full attention near the input, more KDA deeper; §7.2).
    - Combine KDA with sparse selection (NSA/MoBA/DSA) to marry constant-memory compression with targeted retrieval (§7.1).
    - State expansion or mixtures-of-memories for stronger recall without losing efficiency (§7.1 “state expansion” refs [23, 34, 117, 39]).
  - Theory:
    - Further formalize the positional-encoding view (Table 6) and characterize when KDA’s data-dependent transitions outperform fixed-frequency RoPE.
  - Systems:
    - Wider kernel and compiler support for KDA’s chunkwise operations; auto-tuning chunk sizes by hardware and sequence length.
- Practical applications
  - Long-document and repository-level code understanding, sustained tool-use sessions, streaming assistants with minimal latency growth over hours-long contexts, and RL-style inference loops that require fast, repeated decoding (§1, §5.6).
  - Deployment benefits include reduced GPU memory (smaller KV cache), higher batch sizes, and lower time-per-output-token at very long sequences (Figures 1 and 7).

> In short: Kimi Linear makes linear attention a drop‑in, higher‑quality, and faster alternative to full attention for long-context and decoding‑heavy LLM use, by upgrading the linear module’s expressivity (fine‑grained gated delta rule) and making it hardware‑efficient (chunkwise algorithm specialized to a DPLR variant), then interleaving it simply and effectively with a few full‑attention layers.
