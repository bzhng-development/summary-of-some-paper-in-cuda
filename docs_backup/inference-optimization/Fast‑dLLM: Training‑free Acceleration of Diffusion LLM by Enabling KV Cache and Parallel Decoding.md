# Fast‑dLLM: Training‑free Acceleration of Diffusion LLM by Enabling KV Cache and Parallel Decoding

**ArXiv:** [2505.22618](https://arxiv.org/abs/2505.22618)
**Authors:** Chengyue Wu, Hao Zhang, Shuchen Xue, Zhijian Liu, Shizhe Diao, Ligeng Zhu, Ping Luo, Song Han, Enze Xie
**Institutions:** The University of Hong Kong, NVIDIA, MIT

## 🎯 Pitch

Fast-dLLM revolutionizes diffusion-based LLMs by introducing an innovative KV cache and confidence-aware parallel decoding, achieving up to a 27.6× speedup with minimal accuracy loss. This breakthrough makes non-autoregressive models viable for real-world applications, significantly enhancing the performance of chatbots, code completion, and math solvers.

---

## 1. Executive Summary
Fast-dLLM introduces two inference-time techniques that make diffusion-based large language models (LLMs) fast enough for practical use: an approximate key–value (KV) cache tailored to bidirectional diffusion models and a confidence-aware parallel decoding scheme. Combined, these methods yield up to 27.6× end-to-end speedup with minimal accuracy loss across math and code benchmarks, closing much of the throughput gap with autoregressive LLMs (Figure 1c; Tables 1–2, 4–5).

## 2. Context and Motivation
- Problem and gap
  - Diffusion LLMs promise non-autoregressive (parallel) text generation and bidirectional attention, which should be fast. Yet open-source models (e.g., LLaDA, Dream) are slower than comparable autoregressive (AR) models in practice.
  - Two key blockers:
    1) They lack `KV cache`, the core inference optimization that lets AR models reuse attention states across steps.
    2) Quality drops when decoding many tokens in parallel because token dependencies get broken by independent sampling (Section 2.2, “Curse of Parallel Decoding”).
- Why it matters
  - Practical deployments (chat assistants, code completion, math solvers) need high throughput with stable quality. Without KV caching and safe multi-token decoding, diffusion LLMs underperform in latency and cost, despite their theoretical parallelism.
- Prior approaches and shortfalls
  - Diffusion LLMs such as LLaDA and Dream use masked diffusion (MDM) where tokens are iteratively unmasked. They typically do not support caching due to bidirectional attention (full attention across the sequence) and suffer from quality degradation when sampling multiple tokens independently (Section 2.2; Eq. 3).
  - Some works add auxiliary models to capture dependencies (e.g., copulas or energy-based decoders), but at the cost of complexity (Section 3.3 cites [16, 34]).
- Positioning
  - This work is training-free and modifies only the inference process. It proposes:
    - A block-wise, approximate KV cache usable with bidirectional diffusion attention (Section 3.2, Figure 2).
    - A confidence-aware parallel decoding policy with a new theoretical guarantee for when independent parallel decoding matches sequential decoding (Theorem 1, Section 3.3).
  - The methods are validated on multiple diffusion LLMs (LLaDA, Dream) and both text and multimodal tasks (LLaDA-V).

## 3. Technical Approach
This section explains how diffusion LLM decoding works, why caching is hard, and how Fast-dLLM enables caching and safe parallel decoding.

- Preliminaries: masked diffusion language modeling (MDM)
  - Forward noising: tokens are progressively replaced with a special `[MASK]` token; each position is independently masked with probability controlled by a diffusion time `t ∈ [0,1]` (Equation 1).
  - Generation (reverse process): instead of strictly inverting the forward chain one token at a time (slow), practical samplers use a “τ-leaping” approximation to unmask multiple tokens per step from noise level `t` to `s < t` (Equation 3). The model predicts, for each masked position, a distribution over the vocabulary `q0|t(x_i | x_t[, p])` and samples new tokens.

- Why standard KV caching does not work “as is”
  - In AR models with causal attention, the past does not change, so previous keys/values can be reused exactly.
  - In diffusion LLMs with full (bidirectional) attention, the entire sequence can influence every position and the token content changes between steps—so keys/values change, apparently invalidating cache reuse.

- Fast-dLLM’s block-wise approximate KV cache (Section 3.2; Figures 2–3)
  - Key idea: Within a “block” of positions, KV activations for the rest of the sequence change very little between adjacent steps. Therefore, caching them is a good approximation.
  - How it works (Figure 2a):
    1) Divide the output sequence into contiguous blocks of size `B`.
    2) Before decoding a block k, compute and store the KV cache for tokens outside the block (initially, the prompt/prefix).
    3) Decode multiple steps within the block while reusing that cache (“Cache Reuse” in Algorithm 1, line 6).
    4) After finishing the block, recompute and refresh caches for all tokens (Algorithm 1, line 19). This refresh can be fused with the last decoding step so there is negligible extra overhead.
  - Why it is reasonable:
    - Figure 3 (both panels) shows cosine similarity of KV activations across adjacent steps is very high near the diagonal (≈1.0), for both prompt tokens (Figure 3a) and tokens in the last block (Figure 3b), indicating minimal change between steps and justifying cache reuse.
  - DualCache (Figure 2b):
    - Extends caching to both directions: cache keys/values for the prefix and the masked suffix surrounding the current block. Because the suffix is all `[MASK]` until its block is decoded, its activations are also stable; this yields further speedup.

- Confidence-aware parallel decoding (Section 3.3; Figure 5; Algorithm 1)
  - Problem: If you independently sample multiple tokens in a step (product of marginals), dependencies are broken and nonsense combinations can occur (e.g., “high house” instead of “high card,” Section 2.2).
  - Mechanism:
    1) At each step, the model outputs a distribution for each masked position in the active block.
    2) Compute a per-position confidence score, e.g., max softmax probability (Algorithm 1, line 7).
    3) Threshold strategy: unmask only those positions with confidence above a global threshold τ; if none exceed τ, unmask the single most confident token to ensure progress (Algorithm 1, lines 8–14).
    4) Factor strategy: sort confidences and choose the largest `n` such that `(n + 1) × (1 − c_(n)) < f` (Algorithm 1, lines 11–13), where `c_(n)` is the `n`-th largest confidence. This ties `n` to a “safe parallelism” factor `f`.
  - Theoretical footing (Theorem 1):
    - In words: if every token to be decoded has high marginal confidence (> 1 − ε), then greedy decoding with the product of marginals (parallel) returns the same argmax as the true joint model (sequential) provided `(n + 1)ε ≤ 1` (Equation 4, Part 1).
    - The theorem also bounds how far the parallel product distribution can be from the true joint in Lp distance and forward KL (Part 2).
    - The bound is tight: if ε > 1/(n+1), mismatches can occur (tightness construction in Appendix A, “Step 3”).
    - Intuition: When each position is very certain, the most likely joint assignment is the vector of each position’s most likely token. As uncertainty grows or the number of positions grows, independence assumptions break more easily.

- Design choices and rationale
  - Block-wise caching is chosen over per-step full recomputation because similarity within a block enables reusing activations with negligible accuracy cost (Figure 3).
  - DualCache is used when the suffix remains mostly masked, enabling bidirectional reuse.
  - Confidence-aware selection avoids the quality loss of fixed “top-K per step” policies by decoding only high-confidence positions (Figure 5c).
  - Factor-based selection mirrors the theorem’s bound for safe parallelism and adapts parallel degree to uncertainty (Table 11; Figure 8).

## 4. Key Insights and Innovations
- Approximate KV cache for bidirectional diffusion decoding (Section 3.2; Figure 2a; Figure 3)
  - Novelty: KV caching is standard in AR models but considered incompatible with bidirectional diffusion. The paper shows a workable approximation by block-wise reuse plus periodic refresh, justified by measured activation stability (Figure 3).
  - Significance: 2–3.6× speedups when used alone (Tables 1–2), and essential to the 27.6× combined gains at long outputs (Figure 1c; Tables 4–5).

- DualCache: caching both prefix and masked suffix (Figure 2b; Table 5)
  - Novelty: Further expands cache reuse to the suffix, exploiting the stability of all-mask regions before they are decoded.
  - Significance: Adds substantial speedup on long generations and longer prefills; e.g., 27.6× at 1024 tokens, 8-shot (Table 5 and Figure 1c).

- Confidence-aware parallel decoding with theoretical guarantees (Section 3.3; Theorem 1; Figures 5 and 8)
  - Novelty: Moves from fixed-K parallel decoding toward thresholded or factor-based decoding grounded in a high-confidence equivalence theorem.
  - Significance: Up to 13.3× speedup from parallel decoding alone (Figure 1c) while preserving quality; consistently better accuracy–efficiency trade-offs than fixed-K baselines (Figure 5 and Figure 8).

- Training-free, generalizable acceleration across models and modalities
  - Novelty: No retraining; purely inference-time. Also applies to multimodal LLaDA-V with careful refresh settings (Section C.1; Table 3).
  - Significance: Widens practical viability of diffusion LLMs for real deployments.

## 5. Experimental Analysis
- Setup (Section 4.1)
  - Hardware: NVIDIA A100 80GB.
  - Models: `LLaDA`, `LLaDA-1.5`, `Dream` (text); `LLaDA-V` (vision-language).
  - Benchmarks: GSM8K, MATH, HumanEval, MBPP; multimodal MathVista and MathVerse (Table 3).
  - Metrics: accuracy; throughput measured as tokens per second end-to-end until `<eos>`.
  - Default hyperparameters: Prefix cache, block size 32; threshold τ = 0.9 unless stated.

- Main results (Figures 1a–c; Tables 1–2)
  - Throughput–accuracy trade-off (Figure 1a):
    - Combined caching + parallel decoding raises throughput with negligible accuracy change compared to baselines LLaDA/Dream.
  - Breakout of contributions (Figure 1b):
    - Tokens per step and throughput both increase substantially with caching + parallelism.
  - End-to-end speed (Figure 1c; 8-shot, 1024 tokens):
    - Baseline LLaDA: 266s/sample, 0.7 tok/s.
    - +Parallel: ~26s, 9.3 tok/s → 13.3×.
    - +PrefixCache: ~20s, 13.0 tok/s → 2.1× (relative to +Parallel) and big overall gain from baseline.
    - +DualCache: ~12s, 19.3 tok/s → total up to 27.6× over baseline.
  - LLaDA (Table 1): At 256 tokens on GSM8K (5-shot), throughput goes from 6.7 tok/s to 54.4 tok/s (8.1×) with accuracy ~79% → 78.5%. At 512 tokens, up to 35.3 tok/s (11.0×) with accuracy ~77% → 77.2%.
  - Dream (Table 2): On MBPP at 512 tokens, 9.4 → 73.6 tok/s (7.8×) with accuracy ~55.6% → 55.2%. On GSM8K at 512 tokens, 7.7 → 42.9 tok/s (5.6×) with accuracy 76.0% → 74.0%.

- Multimodal results: LLaDA-V (Table 3; Section C.1)
  - Using a refresh-based strategy (keep large block length, refresh every r steps), Fast-dLLM achieves:
    - MathVista: 2.84 → 28.2 tok/s (9.9×) with accuracy 59.2% → 56.6%.
    - MathVerse: 2.75 → 23.3 tok/s (8.5×) with no accuracy loss (28.5% → 28.6%).
  - Sensitivity to block size: too small blocks hurt accuracy (Table 9), hence the refresh strategy (Table 10).

- Ablations and analyses
  - Cache block size (Figure 4): Throughput grows with block size; accuracy starts to dip at very large blocks. Block size 32 balances both (≈3.3× faster than no cache in the highlighted setting).
  - Prefill and generation length (Tables 4–5): Longer prefills and outputs amplify speedups due to more cache reuse. DualCache excels at long sequences: up to 27.6× speedup at 1024 tokens with 8-shot (Table 5).
  - Threshold vs factor strategies (Table 11; Figure 8):
    - Factor-based policy yields higher throughput (e.g., GSM8K 256 tokens: 78.5 tok/s vs 54.4 tok/s; 11.7× vs 8.1×) with a small accuracy trade-off (≈1–3%).
  - Parallel token dynamics (Figure 7): The number of tokens decoded in parallel rises during mid-decoding (higher confidence) and drops toward the end.
  - Throughput vs batch size and AR comparison (Figure 9; Section C.5):
    - PrefixCache lifts LLaDA’s throughput substantially, particularly at small–medium batch sizes. AR LLaMA scales better at very large batches (AR becomes compute-bound and thus faster), highlighting a diffusion compute-bound limitation.

- Qualitative checks
  - Case studies show unchanged reasoning quality across cache modes and thresholds for simple arithmetic (Tables 6–8) and comparable multimodal captions with 10× faster decoding (Figure 6).

- Do the experiments support the claims?
  - Yes. Speedups are large and consistent across models, tasks, and lengths; accuracy drops are small and well-characterized. Analyses explain when/why each component contributes most (e.g., longer sequences, longer prefills, block size sensitivity, factor vs threshold trade-offs).

## 6. Limitations and Trade-offs
- Approximation in caching
  - The KV cache is approximate because diffusion decoding uses full attention; activations change between steps. While similarity is high within a block (Figure 3), this is not an exact cache as in AR. Very large blocks can reduce accuracy (Figure 4; Table 9).
- Assumptions behind the theory
  - Theorem 1 presumes a coherent joint distribution whose marginals come from the same model. Masked diffusion models approximate this but may not strictly satisfy it (Appendix A, “Remark 1”). The bound is worst-case and tight; when ε exceeds 1/(n+1), parallel decoding can disagree with the joint optimum.
- Sensitivity and hyperparameters
  - Performance depends on block size, refresh interval (for multimodal), and threshold/factor settings. Poor choices can hurt quality or squander speedups (Figure 4; Tables 9–11).
- Compute-bound behavior
  - Diffusion decoding remains more compute-bound than AR at scale. As batch size grows, AR models like LLaMA can surpass in throughput (Figure 9).
- Scope
  - Methods are designed for masked diffusion LLMs; generalization to other discrete diffusion families is plausible but not demonstrated. Safety, alignment, and domain robustness are not addressed here.

## 7. Implications and Future Directions
- Field impact
  - Fast-dLLM shows diffusion LLMs can be practical at inference time without retraining—bringing their parallelism and bidirectional context closer to AR efficiency. This lowers the barrier to deploying diffusion-based reasoning and coding models.
- Enabled research
  - Exact or tighter KV reuse for bidirectional attention (e.g., low-rank or residual delta caching).
  - Adaptive, theory-guided parallelism policies beyond max-probability confidence (e.g., calibration-aware or entropy-based criteria).
  - Learning-to-schedule decoding: predict block sizes, refresh intervals, and factors per input to optimize speed/quality.
  - Hybrid decoders: combine confidence-aware parallel steps with dependency-capturing post-hoc adjustments (e.g., small energy-based correction) only when needed.
  - Extending to other modalities and tasks (RAG, long-context summarization) and integrating with inference compilers/serving stacks.
- Applications
  - Faster math/code assistants (GSM8K, HumanEval, MBPP results in Tables 1–2).
  - Multimodal reasoning systems with near order-of-magnitude lower latency (Table 3; Figure 6).
  - Any setting where long outputs or multi-shot prompts magnify cache reuse benefits (Tables 4–5; Figure 1c).

> Representative outcome: “With 8-shot prefilling and 1024-token generation, combined confidence-aware parallel decoding plus DualCache achieves 19.3 tokens/s versus 0.7 tokens/s for vanilla LLaDA—27.6× end-to-end speedup with accuracy from 77.3% to 76.0% on GSM8K” (Figure 1c; Tables 4–5).
