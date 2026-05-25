# Replacing softmax with ReLU in Vision Transformers

**ArXiv:** [2309.08586](https://arxiv.org/abs/2309.08586)

## 🎯 Pitch

This paper introduces a simple yet powerful modification to Vision Transformer (ViT) attention: replacing the cross-token softmax normalization with a pointwise ReLU activation divided by sequence length. Remarkably, this 'ReLU/seqlen' attention retains the scaling accuracy of softmax-based attention on ImageNet-21k while enabling much easier and more efficient parallelization, eliminating the costly reductions needed by softmax and potentially paving the way for faster and simpler transformer implementations.

---

## 1. Executive Summary (2-3 sentences)
This paper proposes a drop-in alternative to softmax in Transformer attention for vision: replace the softmax with a pointwise ReLU and divide by sequence length (`ϕ = L^{-1}·ReLU`). Trained on ImageNet-21k, Vision Transformers with this “ReLU/seqlen” attention match the scaling behavior of standard softmax attention in accuracy versus compute, while enabling simpler parallelization because it avoids the cross-token normalization in softmax (Figure 1).

## 2. Context and Motivation
- Problem addressed
  - In standard attention, the `softmax` over the token dimension converts similarity scores into a distribution that sums to 1. Computing the exponentials and, crucially, normalizing across the full sequence requires reductions (“gathers”) across tokens, which is costly and complicates parallelization across devices or sequence shards.
  - Prior attempts to remove softmax (e.g., substituting ReLU or squared ReLU) typically reduced accuracy (Section 2; refs [25, 14, 15]).

- Why it matters
  - Practical: Avoiding cross-token normalization makes attention easier to parallelize across the sequence dimension with fewer gather operations, which can reduce communication overhead on accelerators (Figure 1 caption).
  - Conceptual: If we can retain performance without softmax, we broaden the design space for attention mechanisms and potentially simplify implementations.

- Prior approaches and their shortcomings
  - Pointwise activations without normalization (ReLU, squared ReLU) were empirically weaker and lacked sequence-length-aware scaling (Section 2).
  - Softmax alternatives that still normalize across the sequence dimension preserve accuracy but also preserve the expensive cross-token reduction step [21].
  - Linear-attention variants remove nonlinearities to achieve O(L·d²) or O(L²·d)→O(L·d²) compute scaling by reordering multiplies (footnote 1; refs [16, 22, 18]), but the paper reports accuracy drops “in our experiments, removing the activation entirely reduced accuracy” (Section 2).

- Positioning
  - The paper focuses on standard O(L²) attention but replaces the softmax with a simple pointwise function scaled by the inverse sequence length. The key insight is that dividing by `L` (or roughly `L^α` with α≈1) restores the right magnitude of attention weights and mitigates the accuracy loss observed in earlier softmax-free attempts (Figures 1–2).

## 3. Technical Approach
- Setup: Standard attention with dot-product scores
  - For each query `q_i`, keys `k_j`, and values `v_j` (all in R^d for j=1..L), standard attention computes:
    - Scores: `s_ij = (q_i^T k_j) / sqrt(d)`
    - Weights: `α_ij = softmax_j(s_ij)`
    - Output: `o_i = Σ_j α_ij v_j`
  - This is summarized in Equation (1), where `ϕ` is the activation turning scores into weights; traditionally `ϕ = softmax`.

- Proposal: Scaled pointwise attention
  - Replace softmax with a pointwise activation `h` applied independently to each score, then scale by a function of sequence length:
    - `α_ij = L^{-α} · h( s_ij )`
    - The paper explores several `h` (ReLU, squared ReLU, GELU, softplus, identity, ReLU6, sigmoid) and a sweep over `α` (Figure 2).
  - The recommended choice is `h(x) = ReLU(x)` with `α = 1`, i.e., `ReLU/seqlen`.

- Why divide by sequence length (`L`)?
  - With softmax, weights for each query sum to 1 by construction, so the average weight per token is exactly `1/L`.
  - Without normalization, a pointwise function like ReLU does not constrain the sum of weights. The paper observes that scaling by `1/L` approximately preserves the initial scale of weights so that the expected α’s are O(1/L) at initialization (Sequence length scaling section).
  - Intuition: If elements of `q` and `k` are O(1), then `s_ij` is O(1) after the `1/sqrt(d)` factor (Sequence length scaling). A pointwise function like ReLU keeps magnitudes O(1) (with the exception of squared ReLU). Without dividing by `L`, the sum `Σ_j α_ij` would scale with `L`; dividing by `L` keeps it O(1).

- Design choices and variants studied
  - `α` sweep: The paper varies the exponent α in `L^{-α}` and finds α≈1 is typically best across datasets, model sizes, and activations (Figure 2).
  - Activation choice: No single pointwise function is universally best near α≈1; ReLU is chosen for speed (Figure 2 caption).
  - `qk-layernorm`: Applying LayerNorm to queries and keys before the dot product (qk-layernorm [8]) is not critical at the tested scales when using `ReLU/seqlen` (Figure 3).
  - Gated attention unit: A multiplicative gating variant (from [15]) still benefits from sequence-length scaling; gating is not a substitute for `L^{-α}` (Figure 4). Gating increases compute by ~9.3% for S/8 with ReLU (Figure 4 caption/discussion).

- What this does and does not change
  - Complexity: This approach keeps standard O(L²) attention (footnote 1 discusses linear-attention alternatives but is not the focus here).
  - Parallelization: Because `ReLU/seqlen` is pointwise and does not require normalizing across tokens, it can be parallelized across the sequence dimension with fewer gather operations than softmax attention (Figure 1 caption). In practice, softmax needs a reduction across all tokens to compute the denominator for each query, which can force cross-device synchronization; `ReLU/seqlen` avoids this.

## 4. Key Insights and Innovations
- Sequence-length-aware scaling restores performance
  - Novelty: Prior softmax-free attention commonly omitted scaling by `L` and suffered accuracy drops. Introducing `L^{-α}` (best near α≈1) provides a simple, architecture-agnostic way to match softmax’s effective scaling of weights (Figure 2).
  - Significance: It lets a purely pointwise activation stand in for softmax without requiring cross-token normalization.

- ReLU/seqlen achieves comparable scaling with compute
  - Evidence: In ImageNet-21k training for 30 epochs, accuracy versus TPU core-hours shows `relu/seqlen` “approaches or matches the scaling performance of traditional attention” across ViT sizes S/32 to L/16 (Figure 1, left). The same trend appears for average 10-shot transfer accuracy across 8 datasets (Figure 1, right).
  - Significance: Comparable scaling suggests the alternative has similar returns on additional compute.

- Simpler parallelization due to no sequence-wise normalization
  - Claim: “Attention with ReLU can be parallelized over the sequence length dimension with less gather operations than softmax attention” (Figure 1 caption).
  - Significance: On modern accelerators, fewer cross-token gathers can translate to efficiency gains, especially at longer sequences or with sharded sequence layouts.

- Robustness to architectural details
  - `qk-layernorm` ablation: Removing qk-layernorm barely changes accuracy for the tested models and datasets (Figure 3), implying the method does not critically rely on this stabilization trick at moderate scale.
  - Gating ablation: Adding a gating unit does not remove the need for sequence-length scaling; best accuracy still occurs near α≈1, with or without gating (Figure 4). This suggests `L^{-α}` addresses a core scaling issue with softmax-free attention rather than compensating for a specific architecture.

Overall, the most fundamental innovation is the explicit `L`-dependent scaling of pointwise attention that makes softmax-free attention viable in practice.

## 5. Experimental Analysis
- Evaluation methodology
  - Datasets and training
    - ImageNet-21k (i21k) and ImageNet-1k (i1k) using BigVision configurations without changing hyperparameters (Section 4).
    - i21k training: 30 epochs; i1k training: 300 epochs; both have roughly ~9e5 steps (Section 4).
  - Models
    - ViT variants including S/32, S/16, S/8, B/32, B/16, L/16, with qk-layernorm by default (Figure 1 and Section 4).
  - Metrics
    - i21k-pretrained models are evaluated on i1k by reporting the top-1 among classes shared with i1k, without fine-tuning (Figure 1 caption).
    - Transfer: average 10-shot linear probe accuracy over 8 datasets (CUB-200, Caltech-101, Cars, CIFAR-100, DTD, ColHist, Pets, UCMerced) averaged over 3 seeds (Section 4).
  - Baselines
    - Standard softmax attention.
    - Pointwise variants `h ∈ {relu, relu², gelu, softplus, identity, relu6, sigmoid}` with scaling exponent α swept (Figure 2).
    - Ablations: with/without qk-layernorm (Figure 3), with/without a gated attention unit (Figure 4).

- Main quantitative results
  - Scaling with compute
    - Quote: “Replacing softmax with relu/seqlen approaches or matches the scaling performance of traditional attention” (Figure 1 caption).
    - In both i1k top-1 (left panel) and avg. 10-shot transfer (right panel), the accuracy vs. TPU core-hours lines for softmax and `relu/seqlen` are close and often overlapping across sizes S/32 to L/16 (Figure 1). The y-axes show the ranges: roughly 68–82% for i1k top-1 and 72–84% for averaged transfer accuracy.
  - Effect of `L^{-α}` scaling and activation choice
    - Quote: “We typically observe the best results when α is close to 1. There is no clear best non-linearity at α ≈ 1, so we use ReLU in our main experiment for its speed.” (Figure 2 caption).
    - The sweep across S/32, S/16, S/8 on both i21k and i1k training settings shows clear peaks near α≈1 for multiple `h` choices (Figure 2).
  - qk-layernorm ablation
    - Figure 3 shows minimal differences between using vs. removing qk-layernorm for both ReLU and squared-ReLU variants across S/32, S/16, S/8 on i21k. The curves largely overlap, indicating the mechanism is not heavily dependent on qk-layernorm at these scales.
  - Gating ablation
    - Figure 4 shows that adding a gated attention unit still benefits from sequence-length scaling; best accuracy still occurs at α≈1. The paper notes a compute cost increase of about 9.3% for S/8 with ReLU when adding gating.
  - Linear attention note
    - Section 2 remarks that removing the activation altogether (to get linear attention) reduced accuracy in their experiments.

- Do the experiments support the claims?
  - The plots consistently show that `relu/seqlen` tracks softmax closely in accuracy across model scales and compute budgets (Figure 1). The α-sweeps (Figure 2) strongly support the need for `L`-dependent scaling and show that α≈1 is a robust choice. Ablations (Figures 3–4) indicate the effect is not an artifact of qk-layernorm or gating.
  - Caveat: Training i21k for only 30 epochs is shorter than common full-pretraining regimens; performance comparisons are therefore about relative scaling, not absolute SOTA.

- Failure cases or mixed results
  - Squared ReLU can inflate magnitude (because it increases score magnitude), which is why `L^{-1}` is especially important; the paper notes it is the one exception to the “preserves O(1)” heuristic (Sequence length scaling; footnote 2).
  - Without `L^{-α}`, non-softmax variants degrade as sequence length grows (Figure 2 shows worse accuracy at α ≈ 0).

## 6. Limitations and Trade-offs
- Assumptions and conditions
  - The theoretical motivation is heuristic: preserving the expected O(1/L) scale of attention weights at initialization, not a formal guarantee of optimization or generalization benefits (Sequence length scaling).
  - The method keeps O(L²) compute and memory; it does not deliver the algorithmic speedups of linear-attention methods (footnote 1). Its benefit is reduced cross-token synchronization, not lower complexity.

- Scope and generality
  - Experiments are on image classification with ViTs and ImageNet-21k/1k; applicability to very long sequences, dense detection/segmentation, or autoregressive language modeling remains to be demonstrated.
  - The largest vision model tested is ViT-L/16; the paper notes qk-layernorm “may change at scale,” implying stability/benefit at much larger scales is unverified (Figure 3 caption discussion).

- Computational and training design constraints
  - i21k pretraining uses 30 epochs (Section 4), shorter than common practice; absolute accuracy is not the focus. Gains are shown in scaling trends, not final SOTA.
  - The paper does not report end-to-end wall-clock speedups or communication metrics; it argues “fewer gather operations” (Figure 1 caption) but does not quantify system-level gains.

- Open questions
  - Quote: “We are unsure why the factor L^{-1} improves performance or if this term could be learned.” (Conclusion)
  - The best pointwise activation is not conclusively identified; ReLU is chosen for speed, not proven optimality (Figure 2).

## 7. Implications and Future Directions
- Impact on the field
  - Demonstrates that softmax is not indispensable for competitive accuracy in ViT attention, provided one rescales by sequence length. This relaxes a long-standing design constraint and can simplify parallel implementations.

- Practical applications
  - Drop-in replacement in ViT-based systems where parallel efficiency across tokens matters, such as large-batch pretraining on multi-accelerator clusters. The lack of cross-token normalization may reduce communication overhead, especially under sequence sharding.

- Follow-up research directions
  - Learnable or data-dependent scaling: Replace fixed `L^{-1}` with a learned per-head or per-layer scaling factor that maintains stable weight magnitudes but adapts during training (Conclusion).
  - Beyond vision: Test `ReLU/seqlen` in language models (with causal masks), audio, and multimodal Transformers to assess generality and impacts on calibration, perplexity, and long-range reasoning.
  - Better pointwise activations: Systematically search for activations that, combined with `L^{-α}`, improve accuracy or stability beyond ReLU (Figure 2 hints no clear winner yet).
  - Systems quantification: Measure wall-clock speedups, communication volume, and energy usage from eliminating sequence-wise normalizations; integrate with FlashAttention-like kernels to see net benefits (related to [7]).
  - Hybrid approaches: Combine `ReLU/seqlen` with linear-attention techniques or kernels for very long sequences, preserving accuracy while gaining algorithmic scaling.

- Broader takeaway
  - A simple, theoretically motivated scaling rule—match the expected O(1/L) scale of attention weights—can revive softmax-free attention. The method is conceptually minimal, empirically effective in ViTs (Figures 1–4), and opens a practical path to more parallel-friendly attention mechanisms.
