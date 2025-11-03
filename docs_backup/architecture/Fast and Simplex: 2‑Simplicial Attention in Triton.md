# Fast and Simplex: 2‑Simplicial Attention in Triton

**ArXiv:** [2507.02754](https://arxiv.org/abs/2507.02754)
**Authors:** Aurko Roy, Timothy Chou, Sai Surya Duvvuri, Sijia Chen, Jiecao Yu, Xiaodong Wang, Manzil Zaheer, Rohan Anil
**Institutions:** 

## 🎯 Pitch

This paper presents 2-simplicial attention, an innovative mechanism that evaluates triplets of tokens rather than pairs, enhancing token efficiency and achieving improved parameter scaling for reasoning-intensive tasks. By addressing the data scarcity challenge, it provides a scalable solution that enables large language models to perform better with limited tokens, offering significant potential for advances in reasoning, math, and coding applications.

---

## 1. Executive Summary
This paper introduces 2‑simplicial attention—an attention mechanism that scores triplets of tokens rather than pairs—and makes it practical with an efficient Triton implementation and a sliding‑window design. In fixed‑token training regimes, similarly sized models with interleaved 2‑simplicial layers achieve lower negative log‑likelihoods on reasoning, math, and coding benchmarks and exhibit a steeper parameter scaling exponent than standard dot‑product attention (Tables 2–3), suggesting improved token efficiency.

## 2. Context and Motivation
- Problem addressed
  - Modern scaling laws model loss as a power law in both parameters and tokens: L(N,D) = E + A/N^α + B/D^β (Equation 1, Section 3). Compute‑optimal scaling suggests increasing both parameters and tokens together (Hoffmann/Chinchilla).
  - In practice, high‑quality data (tokens) is becoming the bottleneck. Many architectural tweaks mostly shift the loss offset E but do not change the exponent α or β, so they do not help when tokens are limited (Section 1, citing Kaplan, Shen, Hestness; summary discussion in Section 3).
- Why this matters
  - If an architecture can increase α (the parameter scaling exponent) for the same token budget, the model improves faster with scale under data scarcity—directly addressing a dominant practical constraint.
- Prior approaches and shortcomings
  - Linear‑time or sparse attentions reduce compute but often degrade quality (Section 2).
  - Higher‑order attentions (e.g., 2‑simplicial, triangle attention in proteins) exist but lacked scalable, general‑purpose implementations and evidence of better token efficiency for language tasks (Section 2).
- Positioning
  - This work: (1) re‑instantiates 2‑simplicial attention with a carefully optimized Triton kernel and a sliding‑window design to control cubic costs (Sections 6–7), (2) proposes a rotation‑invariant trilinear form to enable position encoding analogous to RoPE (Section 5), and (3) empirically shows improved token‑efficiency and larger α than standard attention for reasoning‑heavy tasks (Sections 8, Tables 2–4).

## 3. Technical Approach
At a high level, 2‑simplicial attention extends the pairwise interaction in dot‑product attention to a triple interaction. Instead of comparing every query `q_i` to each key `k_j`, a query is compared to pairs of keys `(k_j, k'_k)`, producing a 3D tensor of logits.

- Standard dot‑product attention refresher (Section 4)
  - Compute Q, K, V via linear projections.
  - Logits: A = QK^T / √d (Equation 2); weights via row‑wise softmax (Equation 3); output ṽ_i = ∑_j S_ij v_j (Equation 4).

- 2‑simplicial attention (Section 4)
  - Additional projections K′ and V′.
  - Trilinear logits:
    - A^(2s)_i j k = ⟨q_i, k_j, k′_k⟩ / √d = (1/√d) ∑_{l=1}^d Q_il K_jl K′_kl (Equation 5).
    - Softmax across both j and k axes (Equation 6).
  - Output combines values multiplicatively:
    - ṽ^(2s)(i) = ∑_{j,k} S^(2s)_i j k (v_j ∘ v′_k) (Equation 7), where ∘ is element‑wise product.
  - Intuition: The score for token i depends on a triangular relation among tokens (i, j, k). This can capture constraints that are inherently triplet‑based (e.g., transitive relations, simple logical/matching structures) that pairwise attention struggles with.

- Position encoding via a rotation‑invariant trilinear (Section 5)
  - Issue: the naive trilinear ⟨a,b,c⟩ is not invariant under a shared rotation, which breaks positional schemes like RoPE that rely on rotational invariance of the scoring function.
  - Solution: use a determinant‑based trilinear. Chunk each vector into 3‑dim blocks and sum 3×3 determinants:
    - A^(det)_{i j1 j2} = ∑_{l=1}^p det([q_i^(l), k_{j1}^(l), k′_{j2}^(l)]) (Equation 9).
  - Why it works: det([a,b,c]) is invariant to a shared rotation of a, b, c. By Sarrus’ rule, each determinant becomes a sum/difference of dot‑products (Equation 8), so it remains implementable with standard tensor ops.
  - Expressivity: Theorem 5.1 shows a single 7‑dim head with this determinant‑based attention can implement a modular “Match3” predicate (existence of j1, j2 such that x_i + x_{j1} + x_{j2} ≡ 0 mod M). The constructive proof (Appendix A) embeds inputs into sinusoidal features so that the summed determinants realize cos(θ_i + θ_j + θ_k), peaking exactly when the modular sum constraint holds.

- Making 2‑simplicial attention practical (Sections 6–7)
  - Complexity control with sliding windows (Section 6):
    - Global 2‑simplicial is O(n^3). The paper uses local windows of widths `w1` for K and `w2` for K′ so each query attends only to a rectangle of size w1×w2 (Figure 2, left).
    - Complexity becomes O(n w1 w2) with a constant 6 from the trilinear/einsum arithmetic (Section 6: O(A^(2s)) = 6 n w1 w2).
    - The paper explores several (w1,w2) pairs and picks `(512,32)` as a latency/quality compromise (Table 1).
  - Head sharing and tiling for throughput (Section 6–7):
    - Adopts high `GQA` (grouped‑query attention) ratio of 64 so many queries share K/K′/V/V′, allowing tiling along the head dimension and dense computation without expensive masks.
  - Triton kernel with online softmax (Section 7; Appendix B/C):
    - 2D tiling trick: pre‑multiply two inputs elementwise (e.g., Q∘K or V∘V′) so the remaining contraction is a matrix multiply; this lets the pipeline overlap CUDA‑core elementwise work with Tensor‑core matmuls (Figure 2, right).
    - Uses online softmax as in FlashAttention to keep memory traffic low.
    - Achieves up to ~520 TFLOPS in Triton, comparable to high‑end FlashAttention v3 Triton kernels; potential to gain more with CUTLASS (Section 7).
  - Backward pass without atomic bottlenecks (Section 7; Algorithm 2; Appendix C):
    - Splits grad computation into two kernels—one for (dK, dV) and another for (dK′, dV′, dQ)—to avoid excessive atomics across three reduction orders (Equations 10–16).
    - For small `w2`, uses a two‑stage “even/odd tile” sweep (Algorithm 2) to compute dQ jointly with dK′/dV′ without atomics.

## 4. Key Insights and Innovations
- 2‑simplicial attention with practical efficiency
  - Novelty: Local rectangular windows and a Triton kernel that fuses the trilinear contraction into matmul‑friendly tiles (Figure 2, Section 7). This turns a conceptually cubic operator into a near‑quadratic‑like cost at long contexts.
  - Significance: Enables routine inclusion of 2‑simplicial layers in large LMs rather than restricting them to niche tasks (Sections 6–7).
- Rotation‑invariant trilinear form enabling relative positions (Section 5)
  - Novelty: Uses a sum of 3×3 determinants over 3‑dim chunks (Equation 9). This preserves invariance under shared rotations, a property needed to generalize RoPE to trilinear attention.
  - Significance: Makes 2‑simplicial attention compatible with widely used positional schemes and provides a clean analytical object (determinant) with geometric meaning (signed volume).
- Theoretical expressivity result (Theorem 5.1; Appendix A)
  - Novelty: A single 7‑dim head can realize a triplet‑matching predicate modulo M through the determinant form, with a constructive sinusoidal embedding.
  - Significance: Shows that 2‑simplicial attention can solve classes of triplet constraints in one layer that would be awkward or deep for pairwise attention, aligning with prior theory that higher‑order attention broadens the representable function class.
- Empirical scaling‑law change, not just a constant shift (Sections 3 and 8; Tables 2–4)
  - Novelty: When trained with the same number of tokens, models with interleaved 2‑simplicial layers show larger parameter exponents α than dot‑product baselines (e.g., GSM8k α: 0.1683 vs 0.1420; +18.5%, Table 3).
  - Significance: If α is genuinely higher, one can increase parameters faster than tokens and still see returns—valuable in token‑scarce regimes.

## 5. Experimental Analysis
- Setup (Section 8)
  - Models: Mixture‑of‑Experts (MoE) LMs with “active parameters” (the part used per token) ranging 1B, 2B, 3.5B; total parameters 57B, 100B, 176B respectively. Every fourth layer is a 2‑simplicial layer; the rest use standard attention. This interleaving balances pipeline stage compute (Section 8).
  - Training: AdamW, peak LR 4e‑3, wd 0.0125, 4k warmup, cosine decay to 0.01× peak (Section 8).
  - Evaluation: Negative log‑likelihood (NLL) on GSM8k (5‑shot), MMLU, MMLU‑pro, MBPP (Section 8). NLL is a pretraining‑aligned metric; lower is better.
  - Baseline: Same MoE sizes trained with purely dot‑product attention; token budget is the same across conditions so the D term in Equation 1 can be treated as constant when fitting α (Section 8, Equations 17–20).
- Main quantitative results (Table 2)
  - 1B active params: 2‑simplicial is roughly neutral to slightly worse.
    - Example: GSM8k NLL 0.3302 vs 0.3277 (+0.79%).
  - 2B active params: consistent improvements.
    - GSM8k: 0.2942 vs 0.2987 (−1.51%); MMLU: 0.5862 vs 0.5932 (−1.19%).
  - 3.5B active params: bigger gains.
    - GSM8k: 0.2718 vs 0.2781 (−2.27%); MMLU‑pro: 0.7689 vs 0.7858 (−2.15%).
  - The improvements concentrate on reasoning‑heavy benchmarks (GSM8k, MMLU‑pro).
- Scaling‑law analysis (Section 8; Tables 3–4)
  - With D fixed, fit −log L(N) ≈ α log N + β (Equations 18–20).
  - Reported α gains:
    - GSM8k: 0.1683 vs 0.1420 (+18.5%).
    - MMLU‑pro: 0.1083 vs 0.0901 (+20.2%).
  - Goodness of fit is very high (R^2 ≈ 0.997–0.9999, Table 4), but note the fit uses just three model sizes.
- Kernel performance (Figure 3; Section 7)
  - 2‑simplicial forward achieves up to ~520 TFLOPS in Triton and is competitive with FlashAttention v3 (FAv3) on long sequence lengths; execution time grows similarly with sequence length.
- Latency vs window sizes (Table 1; Section 6)
  - Investigated several (w1, w2) settings; chose (512,32) as a balanced point. For example, at 16k context:
    - (512,32): ~55 ms; (128,128): ~59 ms; (1024,16): ~55.1 ms.
- Assessment of evidence
  - Strengths:
    - Same token budget across models isolates the effect on α (Section 8).
    - Consistent NLL gains at 2B and 3.5B across multiple benchmarks (Table 2).
    - A clear, reproducible recipe for making 2‑simplicial layers efficient (Sections 6–7; Appendices B–C).
  - Caveats:
    - Only three model sizes for the scaling fit make α estimates sensitive (Tables 3–4).
    - Metrics are NLL rather than task accuracy/pass@k; conversion to downstream accuracy is not reported.
    - Ablations are limited: e.g., the frequency of 2‑simplicial layers, choice of (w1,w2), or head dimensions are not systematically tied to quality outcomes (Table 1 reports latency only).

## 6. Limitations and Trade-offs
- Computational and memory costs
  - Even with windows, complexity is O(n w1 w2) (Section 6). Choosing large windows can approach quadratic‑to‑super‑quadratic costs and increases memory bandwidth needs.
  - Doubling key/value streams (K/K′ and V/V′) increases memory traffic and storage versus standard attention.
- Kernel/engineering maturity
  - The current Triton kernels are “efficient for prototyping” but “far away from being used in production” (Section 9). More low‑level optimization (e.g., CUTLASS‑based) may be required for peak deployment performance.
- Model‑size sensitivity
  - Gains appear at 2B and 3.5B active parameters; 1B shows no improvement and sometimes slight regressions (Table 2). This suggests a scale threshold before 2‑simplicial layers pay off.
- Evaluation scope
  - Pretraining‑style NLL is reported; no end‑task accuracy or generative metrics (e.g., chain‑of‑thought correctness, code execution pass@k).
  - No robustness checks reported (e.g., sensitivity to data distribution, long‑context extrapolation quality beyond latency plots).
- Theory–practice gap
  - The expressivity theorem (Theorem 5.1) is for the determinant form; experiments use the simpler trilinear form for backprop derivations (Section 5 end). While the two are argued to be comparably expressive, the empirical models do not directly use the determinant‑based logits.

## 7. Implications and Future Directions
- Impact on the field
  - Provides a practical template for higher‑order attention in LLMs and shows that architectural changes can affect the scaling exponent α for reasoning‑heavy tasks (Tables 2–3). This directly addresses token‑scarce scaling.
- Research directions
  - Systematic ablations:
    - Frequency and placement of 2‑simplicial layers; head dimension and number of heads; alternatives to the Hadamard combination of V and V′.
    - Window schedules (adaptive w1,w2 by layer or by token), and global‑plus‑local hybrids.
  - Positional schemes and invariances:
    - Deploy and evaluate the determinant‑based rotation‑invariant form at scale; compare to vanilla trilinear with RoPE variants.
  - Theory:
    - Formalize when and why α increases for certain task families; extend Match3‑style results to richer compositional reasoning tasks and to deeper networks.
  - Systems:
    - Hardware‑aware kernels (CUTLASS/CUDA custom kernels) and accelerator co‑design for triple contractions; memory‑saving layouts for K′/V′.
  - Applications:
    - Reasoning‑centric pretraining (math, code, logic) where token supply is precious; retrieval‑augmented setups where triplet relations (query, document, snippet) are natural; protein/graph domains where triangle structures are intrinsic.
- Practical takeaway
  - When data is the bottleneck, inserting a modest fraction of 2‑simplicial layers—implemented with sliding windows and efficient kernels—can yield better returns per token than scaling standard attention alone. As Section 10 concludes:
    > “2‑simplicial attention … changes the exponent corresponding to parameters in the scaling law … leading to more favorable scaling under token constraints,” with larger α gains on harder benchmarks like GSM8k and MMLU‑pro (Table 3).
