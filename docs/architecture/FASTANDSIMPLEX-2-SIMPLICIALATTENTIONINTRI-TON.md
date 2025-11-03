# FAST AND SIMPLEX: 2-SIMPLICIAL ATTENTION IN TRI-TON

**ArXiv:** [2507.02754](https://arxiv.org/abs/2507.02754)

## 🎯 Pitch

This paper introduces 2-simplicial attention, a novel higher-order generalization of standard Transformer attention that allows each token to attend to pairs of other tokens via a trilinear function. By providing an efficient Triton-based implementation and rigorous scaling law analysis, the authors demonstrate that interleaving 2-simplicial layers enables models to achieve superior token efficiency and improved performance on reasoning, math, and coding tasks—crucially, with a steeper scaling law exponent under a fixed token budget. This breakthrough addresses the pressing bottleneck of limited high-quality training data in large language models, paving the way for more resource-efficient and capable AI systems.

---

## 1. Executive Summary
This paper introduces 2‑simplicial attention, a higher‑order generalization of standard attention that lets each token attend to pairs of other tokens via a trilinear scoring function. It provides a practical, high‑throughput Triton implementation and shows that, at a fixed token budget, models with interleaved 2‑simplicial layers achieve better token efficiency and a steeper scaling‑law exponent on reasoning, math, and coding benchmarks compared to dot‑product Transformers.

## 2. Context and Motivation
- Problem addressed
  - Modern large language models (LLMs) are increasingly limited by the availability of high‑quality training tokens rather than compute. Standard scaling laws recommend scaling parameters and tokens together, but this assumes infinite data. Section 1–3 emphasize the need for token‑efficient architectures that improve loss with fewer tokens.
  - Prior practice shows that most architectural or optimizer tweaks shift the loss offset but typically do not change the scaling exponent; see Section 1 with references to Kaplan et al. (2020), Hoffmann et al. (2022), Hestness et al. (2017), and the summary in Everett (2025).

- Why it matters
  - Token scarcity is now a practical bottleneck. If an architecture improves the exponent in the loss–vs–parameters power law at fixed tokens, one can get better quality without acquiring proportionally more data (Sections 1 and 3).

- Prior approaches and gaps
  - Linear‑time attention (e.g., kernelized attention, state‑space models like Mamba) improves complexity but often trails quality (Section 2).
  - Higher‑order attention ideas exist (2‑simplicial attention, triangular attention in Edge Transformer or AlphaFold), but a scalable, general‑purpose implementation and end‑to‑end scaling analysis for LLM pre‑training remained lacking (Section 2).

- Positioning
  - The paper revisits 2‑simplicial attention (Clift et al., 2019) and contributes:
    - An efficient sliding‑window design with a Triton kernel (Sections 6–7).
    - A rotation‑invariant trilinear form compatible with positional encodings and a simple expressivity theorem (Section 5, Theorem 5.1).
    - Empirical evidence that interleaving 2‑simplicial layers yields better token efficiency and a different (steeper) scaling exponent than standard attention (Sections 8 and Tables 2–3).

## 3. Technical Approach
At a high level, 2‑simplicial attention lets each query position i attend not to single keys j, but to pairs (j, k). Concretely:

- Baseline: standard attention (Section 4)
  - For a sequence `X ∈ R^{n×d}`, compute query/key/value: `Q = X W_Q`, `K = X W_K`, `V = X W_V`.
  - Logits are dot products (Equation 2): `A = Q K^T / sqrt(d)`, softmax along each row (Equation 3), then the output is a weighted sum of values (Equation 4).

- 2‑simplicial attention (Section 4)
  - Add a second key/value stream: `K′ = X W_K′`, `V′ = X W_V′`.
  - Use a trilinear score tying one query to a pair of keys (Equation 5):
    > A^{(2s)}_{i j k} = (1/√d) Σ_{l=1..d} Q_{i l} K_{j l} K′_{k l}
  - Apply a softmax jointly over the pair indices (Equation 6).
  - Aggregate values via element‑wise (Hadamard) product of the paired values (Equation 7):
    > ṽ^{(2s)}(i) = Σ_{j,k} S^{(2s)}_{i j k} · (v_j ◦ v′_k)
  - Intuition: the model can directly capture triangular or three‑way relations among tokens (i, j, k), which are hard to represent with only pairwise dot products in a single layer.

- Rotary encodings and rotation invariance (Section 5)
  - A stumbling block: the trilinear form in Equation 5 is not invariant to the same orthogonal rotation applied to all three vectors, which complicates using rotary positional embeddings (RoPE).
  - Solution: use a determinant‑based trilinear form that is rotation‑invariant. For 3‑D chunks of `q, k, k′`, define (Equations 8–9):
    > f̂3(a,b,c) = det([a; b; c])  
    > A^{(det)}_{i j1 j2} = Σ_{l=1..p} det([q_i^{(l)}, k_{j1}^{(l)}, k′_{j2}^{(l)}])
  - This preserves inner‑product‑like invariances under rotations, making it compatible with RoPE semantics.

- Expressivity result (Appendix A, Theorem 5.1)
  - With a single attention head of dimension `d = 7` using the determinant‑style logits (Equation 9), there is a construction whose output at position i is 1 iff there exists a pair (j1, j2) such that `(x_i + x_{j1} + x_{j2}) ≡ 0 (mod M)`. This “Match3”‑style capability formalizes a class of triple‑matching problems solvable in one layer.

- Making it practical: sliding‑window 2‑simplicial attention (Section 6)
  - Full 2‑simplicial attention is O(n^3). The paper constrains attention to a local rectangle `[w1 × w2]` in the two key axes around each query i (Figure 2, left).
  - Complexity comparison (Section 6):
    > Dot‑product causal attention: O(A) = 2 n^2  
    > 2‑simplicial (windowed): O(A^{(2s)}) = 6 n w1 w2
  - The implementation evaluates window configurations and chooses `(w1, w2) = (512, 32)` to balance quality and latency (Table 1).

- System and kernel design (Section 7)
  - Kernel core ideas:
    - Use “online softmax” à la FlashAttention for numerical stability and IO‑awareness.
    - Convert the 3‑tensor Q·K·K′ into a sequence of dense GEMMs by first element‑wise multiplying one pair (e.g., `Q·K1` or `V·V′`) and then applying a matrix multiply with the third factor (Figure 2, right).
    - Overlap CUDA‑core (element‑wise) and Tensor‑core (GEMM) work; Triton implementation reaches ~520 TFLOPS (Figure 3).
  - Backward pass (Equations 10–16) is decomposed into two kernels (Section 7) to avoid excessive atomics; for small `w2`, a two‑stage algorithm (Algorithm 2) computes `dQ` with `dK′, dV′` without atomics by alternating even/odd tiles.

- Model architecture choices (Sections 6 and 8)
  - Interleave sliding‑window 2‑simplicial layers: every fourth layer is 2‑simplicial to spread compute evenly across pipeline stages.
  - Use high Grouped Query Attention ratio (`GQA`=64) to tile heads efficiently and avoid expensive element‑wise masks (Section 6).
  - Experiments use Mixture‑of‑Experts (MoE) models with “active parameters” (parameters used per token during routing) substantially smaller than total parameters (Section 8).

## 4. Key Insights and Innovations
- Higher‑order attention that changes scaling exponents (Sections 3, 8; Tables 2–3)
  - Novelty: While many changes shift the loss intercept, the paper shows a changed exponent α in the token‑fixed scaling law `L(N) ≈ E' + A / N^α` (Equations 17–20).
  - Significance: A higher α means that increasing model size yields faster improvement at fixed token count—directly addressing token scarcity.

- Practical 2‑simplicial attention kernel (Section 7; Figure 3; Listings B–C)
  - Novelty: A Triton kernel that fuses 2‑simplicial online softmax with 2D tiling, reaching performance competitive with optimized FlashAttention v3 implementations at long sequence lengths.
  - Significance: Transforms a theoretically attractive but cubic‑cost mechanism into a usable building block by leveraging sliding windows and hardware‑aware tiling.

- Rotation‑invariant trilinear form and expressivity (Section 5; Appendix A)
  - Novelty: Determinant‑based trilinear logits (Equations 8–9) compatible with rotational invariances introduced by RoPE, alongside a simple expressivity theorem (Theorem 5.1).
  - Significance: Bridges a key gap between theory and practical positional encoding, and shows one‑layer solvability of triple‑matching patterns.

- Systems design for throughput (Sections 6–7; Table 1; Algorithm 2)
  - Novelty: Interleaving 2‑simplicial layers with standard layers, aggressive GQA (64), and a two‑kernel backward pass with a two‑stage, no‑atomics path for small `w2`.
  - Significance: Converts a complex n‑way aggregation into a pipeline‑friendly operator with predictable latency.

## 5. Experimental Analysis
- Setup (Section 8)
  - Models: Sparse MoE LLMs with three sizes, reported as active/total parameters: `1B/57B`, `2B/100B`, `3.5B/176B`.
  - Layering: Every fourth layer uses sliding‑window 2‑simplicial attention; other layers use standard global attention.
  - Training: AdamW, peak LR `4e-3`, weight decay `0.0125`, 4k warmup, cosine decay to `0.01×` peak (Section 8).
  - Token budget: Models in each pair are trained on the same number of tokens, enabling parameter‑only scaling analysis (Section 8; Equations 17–20).
  - Metrics: Negative log‑likelihood (NLL) on benchmarks that probe reasoning/math/coding quality in pre‑training:
    - GSM8K (5‑shot NLL), MMLU, MMLU‑pro, MBPP (Section 8).
  - Baseline: Identically sized dot‑product Transformers.

- Latency/throughput evidence (Sections 6–7)
  - Window search: Table 1 shows per‑sequence latency for combinations of `w1` and `w2`, e.g., `(w1=512, w2=32)` at ~55.1 ms for 16k context.
  - FLOPs and runtime vs FlashAttention v3: Figure 3 compares theoretical FLOPs and measured ms; the proposed kernel tracks FA v3 closely at longer sequences.

- Main quantitative results (Table 2)
  - NLL (lower is better) summary; 2‑simplicial vs Transformer:
    - 1B active: mixed results; slight degradations or ties:
      > GSM8K: 0.3302 vs 0.3277 (Δ +0.79%)  
      > MMLU: 0.6423 vs 0.6411 (Δ +0.19%)  
      > MMLU‑pro: 0.8718 vs 0.8718 (Δ −0.01%)  
      > MBPP: 0.2714 vs 0.2690 (Δ +0.88%)
    - 2B active: consistent improvements:
      > GSM8K: 0.2942 vs 0.2987 (Δ −1.51%)  
      > MMLU: 0.5862 vs 0.5932 (Δ −1.19%)  
      > MMLU‑pro: 0.8135 vs 0.8193 (Δ −0.71%)  
      > MBPP: 0.2411 vs 0.2435 (Δ −1.0%)
    - 3.5B active: larger gains, especially on reasoning‑heavy sets:
      > GSM8K: 0.2718 vs 0.2781 (Δ −2.27%)  
      > MMLU: 0.5484 vs 0.5543 (Δ −1.06%)  
      > MMLU‑pro: 0.7689 vs 0.7858 (Δ −2.15%)  
      > MBPP: 0.2193 vs 0.2203 (Δ −0.45%)

- Scaling analysis (Tables 3–4; Equations 17–20)
  - With tokens fixed, fit `L(N) ≈ E′ + A / N^α`. Rewriting as `−log L ≈ α log N + β` (Equation 20) yields slope α and intercept β.
  - Table 3 shows α increases for 2‑simplicial attention, e.g.:
    > GSM8K: α 0.1420 → 0.1683 (+18.5%)  
    > MMLU: α 0.1256 → 0.1364 (+8.5%)  
    > MMLU‑pro: α 0.0901 → 0.1083 (+20.2%)  
    > MBPP: α 0.1720 → 0.1837 (+6.8%)
  - Goodness‑of‑fit is strong for both models (Table 4), with R² mostly ≥ 0.997 and small residuals, indicating the three‑point fit is consistent.

- Do the experiments support the claims?
  - Evidence aligns with two main claims:
    - Token‑efficiency: At fixed tokens, larger active‑parameter models with interleaved 2‑simplicial layers yield consistently lower NLL than dot‑product baselines for 2B and 3.5B models (Table 2).
    - Changed exponent: The fitted α is higher across all four benchmarks (Table 3), implying more favorable parameter scaling at fixed tokens. This is exactly the desired property in data‑limited regimes.
  - Caveats:
    - Gains do not appear at the smallest 1B active scale (Table 2), suggesting a scale threshold for benefit.
    - Metrics are NLL rather than task accuracy; while NLL is a strong pre‑training indicator, it is not a full downstream evaluation.

- Ablations and robustness
  - Latency ablation across window choices is provided (Table 1).
  - No ablation on percentage of 2‑simplicial layers, window size vs quality, or comparison of determinant‑based logits vs simple trilinear logits in training (Section 5 notes the simpler Equation 5 is used for backpropagation derivations).

## 6. Limitations and Trade-offs
- Computational structure and locality (Sections 6–7)
  - Even with windows, 2‑simplicial cost is `O(n w1 w2)` and requires sophisticated tiling to be fast; windows reduce global receptive field for these layers. The model mitigates this by interleaving with global attention layers, but the optimal ratio is unexplored.
  - The kernel is in Triton and optimized for prototyping; the paper notes it is “still far away from being used in production” and would benefit from lower‑level implementations (Section 9).

- Theory–practice gap (Section 5; Appendix A)
  - The rotation‑invariant determinant form and expressivity theorem are not the exact form used in the main experiments (which use the simpler trilinear Equation 5 for derivations/implementation). The incremental quality gained by determinant logits vs simple trilinear logits under RoPE is not empirically isolated.

- Scope of evaluation (Section 8)
  - Only MoE models are reported; behavior for dense transformers is not shown.
  - Training data composition and total token counts are not detailed, limiting reproducibility and interpretation of absolute NLLs.
  - Reported metrics are NLL on few reasoning/coding sets; broader downstream evaluations (accuracy, robustness, calibration, long‑context tasks) are not included.

- Scaling‑law estimation
  - Power‑law fits use only three model sizes; while R² is high (Table 4), more points across a wider scale would solidify the exponent estimates.

## 7. Implications and Future Directions
- Field impact
  - Demonstrates that architectural changes can alter scaling exponents at fixed tokens for knowledge/reasoning tasks (Table 3), contradicting the prevailing view that most changes only shift the loss offset. This opens a path to token‑efficient scaling when data is scarce (Sections 1 and 3).

- Practical applications
  - Pre‑training regimes constrained by data budgets (e.g., domain‑specific corpora) may benefit from interleaving 2‑simplicial layers to reach better quality without proportionally more tokens.
  - Reasoning‑heavy domains (math, code, logic) appear to benefit most (Table 2 shows the largest relative NLL gains on GSM8K and MMLU‑pro at 3.5B active).

- Follow‑up research
  - Kernel and systems co‑design: Implement 2‑simplicial kernels in CUTLASS or vendor libraries; explore fused scheduling across attention types; extend to other accelerators (Section 9).
  - Architectural ablations: Vary window sizes, proportion/order of 2‑simplicial layers, and GQA ratios; test dense models and larger scales.
  - Positional encoding study: Empirically compare determinant‑based rotation‑invariant logits vs simple trilinear logits with RoPE, and test other relative position encodings.
  - Generalizations: Explore k‑simplicial attention (k>2) with sparse tiling patterns; combine with looped Transformers to trade depth for higher‑order interactions.
  - Evaluation: Move beyond NLL to accuracy and robustness on a wide suite (reasoning chains, long‑context tasks, code execution correctness).

> Core takeaways grounded in the paper’s evidence:  
> - Equation (1) motivates the search for better token efficiency via architectures that can change the exponent α.  
> - Equations (5)–(7) define a tractable 2‑simplicial operator; Sections 6–7 turn it into a high‑throughput kernel with `O(n w1 w2)` cost.  
> - Table 2 shows consistent NLL improvements at 2B and 3.5B active parameters on GSM8K, MMLU, MMLU‑pro, and MBPP.  
> - Table 3 shows α increases of 6.8%–20.2% across benchmarks, with strong fits (Table 4), supporting the claim that 2‑simplicial attention improves the scaling exponent under fixed tokens.
