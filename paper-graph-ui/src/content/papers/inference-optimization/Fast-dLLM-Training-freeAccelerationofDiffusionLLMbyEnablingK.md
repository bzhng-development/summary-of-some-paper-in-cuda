# Fast-dLLM: Training-free Acceleration of Diffusion LLM by Enabling KV Cache and Parallel Decoding

**ArXiv:** [2505.22618](https://arxiv.org/abs/2505.22618)

## 🎯 Pitch

Fast-dLLM dramatically accelerates diffusion-based large language models at inference time without retraining, introducing a novel block-wise approximate Key-Value cache for bidirectional attention and a confidence-aware parallel decoding strategy. This approach achieves up to 27.6× speedup with minimal accuracy loss, bridging the throughput gap with autoregressive LLMs and enabling practical, high-performance deployment of diffusion LLMs for both text and multimodal applications.

---

## 1. Executive Summary
Fast-dLLM is a training-free inference method that makes diffusion-based large language models (LLMs) much faster by: (1) enabling an approximate Key-Value (KV) cache for these bidirectional models via block-wise decoding, and (2) decoding multiple tokens in parallel only when the model is confident enough to avoid dependency errors. Across text and multimodal benchmarks, it achieves up to 27.6× end-to-end speedup with minimal accuracy loss (e.g., Figure 1c; Tables 1–5), substantially narrowing the practical throughput gap with autoregressive LLMs.

## 2. Context and Motivation
- Problem the paper tackles
  - Diffusion LLMs (non-autoregressive generators that iteratively denoise masked tokens) promise high throughput because, in principle, many tokens can be produced at once. Yet open-source systems (e.g., LLaDA, Dream) are slow in practice and often lag behind autoregressive models (Section 1).
  - Two bottlenecks identified:
    1) No KV cache: Unlike autoregressive LLMs, diffusion LLMs use bidirectional attention over the full sequence, so standard KV caching is not straightforward (Section 1).
    2) Parallel decoding hurts quality: Generating many tokens at once assumes those tokens are conditionally independent, which is not generally true (Section 2.2 “Curse of Parallel Decoding”).

- Why this matters
  - Practical deployment: Without caching and safe parallelization, diffusion LLMs cannot realize their theoretical speed advantages in real systems. Closing this gap impacts latency-sensitive applications and cost (Figure 1; Tables 1–2).
  - Theoretical clarity: The work explains when parallel decoding is justified and quantifies its error, guiding principled design (Theorem 1, Section 3.3; Appendix A).

- Prior approaches and their limits
  - Diffusion LLMs (e.g., LLaDA, Dream) demonstrated competitive accuracy but lacked AR-style caching and suffered from quality degradation when parallelizing decoding (Section 1; [21, 36]).
  - Simple “decode K tokens per step” strategies ignore token dependencies and can produce incoherent phrases (Section 2.2; the “high house” example).
  - Existing acceleration for diffusion language models did not provide both a workable KV cache for bidirectional attention and a theoretically grounded parallel decoding rule.

- How this paper positions itself
  - Introduces a block-wise approximate KV cache compatible with bidirectional diffusion decoding (Figure 2), justified by observed KV similarity across adjacent steps (Figure 3).
  - Proposes confidence-aware token selection—threshold- or factor-based—grounded in a high-confidence theorem (Theorem 1) that bounds the gap between independent and joint decoding.
  - Demonstrates state-of-the-art acceleration on text and multimodal tasks without retraining (Tables 1–3; Figure 1).

## 3. Technical Approach
Fast-dLLM is an inference-time pipeline for Masked Diffusion Models (MDMs), which progressively replace [MASK] tokens with actual tokens over a schedule.

- Brief MDM background (Section 2)
  - Forward process: Each token is independently turned into a special `[MASK]` with probability controlled by a time variable `t ∈ [0, 1]`. Equation (1) formalizes this: at `t=1` all tokens are masked; at `t=0` the original sequence is intact.
  - Reverse process: One can denoise from a higher `t` (more masking) to a lower `s` (less masking). A τ-leaping approximation (Equation (3)) lets multiple masked tokens be predicted at once by sampling from the model’s per-token predictive distributions `q0|t`.
  - Practical issue: Sampling many tokens independently breaks dependencies (Section 2.2).

- Pipeline overview (Algorithm 1; Section 3.1)
  1) Divide the answer sequence into blocks of size `B`. For block `k`, reuse cached attention states (KV) for tokens outside the block while iteratively decoding inside the block.
  2) Within a block, at each step compute a per-token confidence (max softmax probability). Decode only the tokens that are sufficiently confident (threshold-based) or as many tokens as justified by a confidence factor rule (factor-based; see theorem).
  3) After finishing a block, refresh the cache to keep it consistent for the next block.

- Block-wise approximate KV Cache (Section 3.2; Figure 2)
  - Why standard AR KV cache doesn’t directly apply: Bidirectional attention looks at both left and right contexts; the set of “past states” to reuse is not fixed as in left-to-right decoding.
  - Core idea: During decoding of a specific block, the KV representations of tokens outside that block (the “prefix” and, in DualCache, also the “suffix”) change very little across adjacent steps. Therefore, reuse them within the block to avoid recomputation; then refresh once the block is done (Figure 2a for prefix, Figure 2b for DualCache).
  - Evidence: Cosine similarity heatmaps of KV activations across steps (Figure 3a “Prompt block” and Figure 3b “Last block”) show values near 1.0 along the diagonal neighborhood, indicating adjacent-step stability within a block.
  - How it operates:
    - PrefixCache: Before decoding block `k`, compute and store KV for the prompt/prefix. Reuse this cache for multiple steps while unmasking tokens inside the block. After the block, recompute the full KV to stay exact for the next block (lines 2, 6, and 19 in Algorithm 1).
    - DualCache: Also cache the suffix (which is all `[MASK]` under block-wise decoding). Because these suffix KVs are even more stable, DualCache saves more recomputation (Figure 2b; Table 5).

- Confidence-aware parallel decoding (Section 3.3)
  - Goal: Decode many tokens at once only when it is safe. “Safe” means the approximate product-of-marginals distribution is close enough to the true joint so that greedy parallel choices match greedy sequential choices.
  - Threshold strategy (Algorithm 1, lines 8–14 with `strategy==threshold`):
    - Compute confidence `c_i = max_x pθ(x_i | ·)` for each masked token in the block.
    - Decode all tokens with `c_i ≥ τ`; if none meets the threshold, decode at least the single highest-confidence token to guarantee progress.
  - Factor strategy (Algorithm 1, lines 10–14 with `strategy==factor`):
    - Sort confidences decreasingly as `(c(1), c(2), …)`.
    - Choose the largest `n` such that `(n + 1)*(1 - c(n)) < f`, where `f` is a hyperparameter (“decoding factor”).
    - Decode the top-`n` tokens. This mirrors the theorem’s condition below and adapts the degree of parallelism to confidence.
  - Why this is principled (Theorem 1; Section 3.3; Appendix A):
    - Plain-language statement: If each of `n` tokens has a very confident marginal (probability > `1 - ε` for its top candidate) and `(n+1)ε ≤ 1`, then choosing tokens independently (product of marginals) yields the same greedy choice as if we had the true joint distribution. The theorem also bounds the divergence between the true joint and the product of marginals: total variation distance < `(3n - 1)/2 * ε`, and forward KL < `(n - 1) * (Hb(ε) + ε ln(|V| - 1))`. These formalize that in high-confidence regimes, independent selection is reliable.
    - Consequence: Only expand parallelism when confidences are high enough; otherwise, hold back.

- Design choices and rationale
  - Blocks enable approximate KV reuse because KV changes are locally smooth across steps (Figure 3). Recomputing once per block balances accuracy and speed (Figure 4 shows the accuracy–throughput trade-off as block size varies).
  - Confidence-based selection mitigates the “curse of parallel decoding” described in Section 2.2 (e.g., avoiding “high house” combinations when independent draws would mismatch dependencies).
  - Factor-based selection ties parallelism to a provable bound (Theorem 1), giving a knob (`f`) for accuracy vs. speed.

## 4. Key Insights and Innovations
- Block-wise KV caching for bidirectional diffusion models (Section 3.2; Figures 2–3)
  - Novelty: KV caching is standard in autoregressive LLMs but non-trivial for bidirectional diffusion generation. The paper introduces a block-wise approximation that reuses KV for tokens outside the current block and refreshes post-block.
  - Why it matters: It slashes redundant attention computation, delivering 2–3.6× speedups alone (Tables 1–2, compare `+Cache` to baselines) with negligible accuracy loss.

- DualCache: caching both prefix and suffix (Figure 2b; Tables 4–5)
  - Novelty: Goes beyond prefix caching by also caching the masked suffix, leveraging its extreme stability during block decoding (validated by Figure 3b).
  - Impact: Further accelerates long generations and long prompts; with 8-shot and 1024-token generation, DualCache contributes to 27.6× end-to-end speedup (Figure 1c; Tables 4–5).

- Confidence-aware parallel decoding with theory (Section 3.3; Theorem 1; Figure 5; Table 11; Figure 8)
  - Novelty: Parallelize only when confidences justify the independence approximation. The factor rule is explicitly tied to a tight high-confidence bound showing when greedy parallel equals greedy sequential decoding.
  - Impact: Parallel decoding alone achieves up to 13.3× speedup in some settings (Figure 1c), and combined with caching yields up to 8.1–11.0× throughput at moderate lengths (Table 1) and larger end-to-end gains at longer lengths (Tables 4–5).

- Practical, training-free recipe with broad applicability
  - No retraining required; integrates cleanly with LLaDA, Dream, and multimodal LLaDA-V (Tables 1–3), indicating generality across architectures and modalities.

## 5. Experimental Analysis
- Evaluation setup (Section 4.1)
  - Hardware: NVIDIA A100 80GB, single GPU.
  - Models: `LLaDA`, `LLaDA-1.5` (Appendix C.3), `Dream-Base`, and `LLaDA-V` for multimodal.
  - Benchmarks:
    - Text: GSM8K (math word problems), MATH (competition problems), HumanEval and MBPP (code generation).
    - Multimodal: MathVista and MathVerse (vision-language math).
  - Metrics: Accuracy and throughput (tokens/sec). Throughput is measured end-to-end until `<eos>`.
  - Defaults: PrefixCache, block size 32; threshold 0.9 unless noted.

- Main quantitative results
  - Text-only LLaDA (Table 1; Figure 1a–b):
    - Gen length 256, GSM8K 5-shot:
      - Baseline: 6.7 tok/s, 79.3% accuracy.
      - +Cache: 21.2 tok/s (3.2×), 79.5%.
      - +Parallel: 16.5 tok/s (2.5×), 79.2%.
      - +Cache+Parallel: 54.4 tok/s (8.1×), 78.5%.
    - Gen length 512, GSM8K 5-shot:
      - Baseline: 3.2 tok/s.
      - +Cache+Parallel: 35.3 tok/s (11.0×) with 77.2% accuracy (vs. 77.5% baseline).
  - Text-only Dream (Table 2):
    - MBPP, length 512:
      - Baseline: 9.4 tok/s, 55.6%.
      - +Cache+Parallel: 73.6 tok/s (7.8×), 55.2% (−0.4%).
    - MATH, length 512:
      - Baseline: 9.6 tok/s, 39.8%.
      - +Cache+Parallel: 63.3 tok/s (6.5×), 39.3% (−0.5%).
  - End-to-end long-context gains (Figure 1c; Tables 4–5):
    - With 8-shot prefill and 1024 generation on LLaDA:
      - Baseline latency 266s (0.7 tok/s) → Fast-dLLM latency 12s (19.3 tok/s): “27.6×” speedup with accuracy 76.0% vs. 77.3%.
    - Speedup rises with longer sequences and longer prefills (Table 5 and Table 4), consistent with more cache reuse opportunities.
  - Multimodal LLaDA-V (Table 3; Appendix C.1):
    - MathVista:
      - Full Steps: 2.84 tok/s, 59.2%.
      - Fast-dLLM: 28.2 tok/s (9.9×), 56.6% (−2.6%).
    - MathVerse:
      - Full Steps: 2.75 tok/s, 28.5%.
      - Fast-dLLM: 23.3 tok/s (8.5×), 28.6% (+0.1%).
    - Note: For LLaDA-V, small block sizes harm accuracy; the paper keeps full block length and refreshes caches every r steps (Appendix C.1; Table 10), trading a small accuracy drop for large speedups.

- Ablations and analyses
  - Block size trade-off (Figure 4): Larger blocks increase throughput but can slightly reduce accuracy due to cache staleness; block size 32 offers a good balance (≈3.3× speedup over no cache with minimal accuracy impact).
  - Threshold vs. fixed K tokens per step (Figure 5): Confidence-thresholding yields higher accuracy for the same average tokens/step and reduces required steps compared to fixed-K baselines.
  - Factor vs. threshold (Table 11; Figure 8):
    - Factor decoding often achieves 1.4–1.5× higher throughput than threshold with only 1–3% accuracy loss (e.g., GSM8K length 256: 11.7× vs. 8.1× speedup).
  - Parallelism over time (Figure 7): Average tokens-per-step rises mid-generation and decreases near the end; variance increases late, suggesting conservative decoding near completion is appropriate.
  - Throughput vs. batch size vs. AR models (Appendix C.5; Figure 9):
    - With PrefixCache, diffusion LLM throughput improves markedly and can rival AR models at small batch sizes, but AR models scale better at large batch sizes (diffusion remains more compute-bound).

- Do results support the claims?
  - Yes. Across two diffusion LLM families and four text benchmarks, both caching and confidence-aware parallel decoding independently yield substantial speedups, and together they deliver order-of-magnitude improvements with small accuracy changes (Tables 1–2).
  - Long-context experiments (Figure 1c; Tables 4–5) confirm that speedup increases with more prefill and longer generation, aligning with the method’s design.
  - Multimodal results show the approach transfers to vision-language settings with careful cache refresh scheduling (Table 3; Table 10).

## 6. Limitations and Trade-offs
- Assumptions behind the method
  - KV stability assumption: The cache relies on adjacent-step KV similarity within a block (Figure 3). If a task induces rapid non-local context changes, the approximation could degrade more.
  - High-confidence requirement: The theoretical guarantee for parallel decoding holds when per-token confidence is high and `(n+1)ε ≤ 1` (Theorem 1). In ambiguous or highly dependent spans, fewer tokens should be decoded in parallel, limiting speed gains.

- Scope and edge cases not fully addressed
  - Strong token interdependence (e.g., fixed phrases, code identifiers across lines) reduces safe parallelism. The paper mitigates this by thresholds/factors but does not model dependencies explicitly (Section 3.3).
  - LLaDA-V shows accuracy drops with small block sizes; the paper uses full blocks and periodic refresh instead (Appendix C.1). Other multimodal tasks may require task-specific cache policies.

- Computational and scalability considerations
  - Diffusion LLMs remain relatively compute-bound; at large batch sizes, AR models can become compute-bound and scale better (Appendix C.5; Figure 9).
  - Hyperparameters (block size, threshold, factor) require tuning per task/model to balance accuracy and throughput (Figure 4; Figures 5 and 8; Table 11).

- Residual accuracy deltas
  - Most settings show ≤1–2% accuracy change, but some longer settings (e.g., LLaDA length 512 in Table 1; LLaDA-V in Table 3) show modest drops. These are acceptable in many applications but not universally.

## 7. Implications and Future Directions
- How this changes the landscape
  - Demonstrates that diffusion LLMs can be inference-competitive with AR models when equipped with approximate KV caching and principled parallel decoding, without retraining (Figure 1; Tables 1–2).
  - Provides a theoretical lens (Theorem 1) to decide when parallel decoding is safe, likely influencing decoding strategies beyond diffusion models.

- Follow-up research enabled or suggested
  - Adaptive scheduling: Learn block partitioning, threshold/factor, and refresh intervals online to maximize speed under accuracy constraints.
  - Dependency-aware parallel decoding: Combine confidence thresholds with lightweight structures (e.g., constrained decoding graphs or small dependency predictors) to safely expand parallelism in dependent spans.
  - Better cache approximations: Explore low-rank or learned cache updates within a block to further reduce refresh costs while keeping accuracy (building on KV similarity patterns in Figure 3).
  - Training-time alignment: Fine-tune diffusion LLMs to increase per-token confidence calibration in high-dependency regions, directly boosting safe parallelism under the theorem’s regime.

- Practical applications
  - Latency-critical assistants in math and coding domains (GSM8K, MATH, HumanEval, MBPP) that prefer diffusion-style global reasoning but need fast responses.
  - Multimodal reasoning systems (e.g., LLaDA-V on MathVista/MathVerse) where high-throughput VLM decoding is valuable for interactive use.
  - Edge or cost-sensitive deployments: The training-free nature simplifies adoption—swap the decoding engine to Fast-dLLM to obtain large speedups.

> Representative headline results:
> - “Up to 27.6× end-to-end speedup” with long prefill and 1024-token generation while keeping accuracy within ≈1–2 points (Figure 1c; Tables 4–5).
> - At 256 tokens on GSM8K (LLaDA), throughput rises from 6.7 to 54.4 tok/s (8.1×) with accuracy 79.3% → 78.5% (Table 1).
> - Multimodal LLaDA-V on MathVista: 2.84 tok/s → 28.2 tok/s (9.9×) with accuracy 59.2% → 56.6% (Table 3).

Overall, Fast-dLLM offers a clear, mechanism-based path to accelerate diffusion LLMs: cache what barely changes, and parallelize only when confidence is high enough to be theoretically safe.
