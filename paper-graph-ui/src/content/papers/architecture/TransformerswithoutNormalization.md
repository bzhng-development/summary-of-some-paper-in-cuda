# Transformers without Normalization

**ArXiv:** [2503.10622](https://arxiv.org/abs/2503.10622)

## 🎯 Pitch

This paper introduces Dynamic Tanh (DyT), a simple, element-wise alternative to traditional normalization layers in Transformers. By showing that DyT can match or exceed the performance of LayerNorm and RMSNorm across diverse vision, language, and generative tasks—without the need for costly per-token statistics—this work both streamlines model implementations and fundamentally challenges the long-held belief that normalization is indispensable in deep neural networks.

---

## 1. Executive Summary
This paper shows that standard normalization layers in Transformers (such as LayerNorm and RMSNorm) can be replaced by a simple element‑wise operation, Dynamic Tanh (`DyT`), with little or no loss in accuracy and often small gains. The key insight is that the input→output mappings of LayerNorm in trained Transformers look like an S‑shaped curve, which `tanh(αx)` replicates while avoiding the cost and complexity of computing per‑token statistics.

## 2. Context and Motivation
- Problem addressed
  - Modern Transformers almost always include normalization layers (LayerNorm/RMSNorm). These are widely believed to be essential for stable optimization and good generalization in deep, wide models.
  - The paper asks: Are normalization layers actually indispensable in Transformers, or can we achieve the same effects with something simpler?

- Why this matters
  - Practical: Normalization layers compute means/variances and require reduction operations. That adds kernel complexity and can be a bottleneck on some hardware. A drop‑in, element‑wise replacement would simplify implementations and may enable new fusions and optimizations (Appendix C).
  - Conceptual: If normalization is not strictly required, we gain a clearer understanding of what it really does in deep networks.

- Where prior approaches fall short
  - “No‑norm” methods existed but rely on carefully crafted initializations (Fixup, SkipInit) or weight reparameterizations/constraints (e.g., spectral reparametrization). These often need significant hyperparameter tuning and may underperform normalized baselines (Section 6.3; Table 9).
  - Some methods remove norms only after pretraining via fine‑tuning, rather than training from scratch without norms.

- How this paper positions itself
  - It proposes a tiny, drop‑in replacement called `Dynamic Tanh (DyT)` that:
    - Does not compute statistics.
    - Is element‑wise and therefore simple to implement and potentially easier to optimize.
    - Empirically matches or exceeds LayerNorm/RMSNorm across diverse tasks and scales, including large language models (Sections 5 and 7.2).

## 3. Technical Approach
Step-by-step overview:

- What normalization layers do (background and observation)
  - Standard formulation (Equation 1, Section 2): a normalization layer transforms input `x` by subtracting a mean `µ`, dividing by the standard deviation `σ`, then applying learnable per‑channel scale `γ` and shift `β`.
  - Empirical observation (Section 3, Figures 2–4):
    - When plotting the element‑wise input vs. output (before the learned affine `γ`/`β`) of LayerNorm in trained models (ViT, wav2vec 2.0, DiT), the mapping looks like an S‑curve—highly reminiscent of `tanh`.
    - Deeper LayerNorm layers show this effect most clearly (Figure 2). Earlier layers look more linear.
    - By coloring points by token (left panels of Figure 4), each token’s mapping is linear but with a different slope (because each token has different variance). Collectively these different lines form an S‑curve.
    - By coloring by channel (right panels of Figure 4), a few channels exhibit extreme input ranges; these are squashed the most by normalization.

  Plain-language interpretation:
  - LayerNorm isn’t globally linear over all elements. Across tokens with different statistics it collectively acts like a near‑linear mapping around zero, but it disproportionately squashes extreme values—just like a saturating nonlinearity.

- The proposed replacement: Dynamic Tanh (`DyT`)
  - Definition (Equation 2, Section 4): `DyT(x) = γ * tanh(α x) + β`
    - `α`: a single learnable scalar that rescales inputs so that `tanh` operates in the “right” part of its S‑curve.
    - `γ`, `β`: standard learnable per‑channel scale and shift, same shapes as in LayerNorm/RMSNorm.
  - Implementation (Algorithm 1, Section 4): a tiny module—apply `tanh(αx)`, then affine scale/shift.
  - Where it is used (Figure 1): replace each normalization layer in attention blocks, MLP/FFN blocks, and the final normalization.
  - What it does mechanistically:
    - Near zero, `tanh` is approximately linear, so most activations pass almost unchanged (Figure 3 shows different slopes via different `α`).
    - Large-magnitude activations are squashed into a bounded range (−1 to 1), reproducing the key “extreme‑value suppression” observed in LayerNorm (Figures 2–4).
    - The scalar `α` adapts over training and closely tracks the inverse activation scale: Section 6.2 and Figure 8 show `α` correlates with `1/std` both during and after training.

- Design choices and rationale
  - Why `tanh`? Section 6.1 and Figure 7 compare `tanh`, `hardtanh`, `sigmoid`, and an identity mapping:
    - Without squashing (identity), training diverges (Table 7).
    - With squashing, training is stable; `tanh` performs best among the tested functions (Table 7), likely due to smoothness and being zero-centered.
  - Why a single scalar `α` (instead of per-channel or per-token)?
    - Simplicity and stability. Empirically, a single `α` already learns to match global scale dynamics (Figure 8). Per-channel or per-token `α` is not explored here.

- Practicalities and initialization
  - Default initialization: `γ=1`, `β=0`, `α0=0.5` typically works without hyperparameter changes (Section 4; Section 7.1).
  - LLM exception: training large LLaMA models benefits from tuned `α0`, with different values in attention vs. other blocks, and smaller `α0` as width increases (Section 7.2; Table 10; Figure 11; Table 11).
  - LLM embedding scale: an extra learnable scalar right after the embedding, initialized to `√d`, is added so early activations aren’t too small (Appendix A, “Large Language Models”).

- How DyT differs from normalization in computation and behavior
  - No reduction: DyT is element‑wise; no means/variances are computed.
  - No per-token adaptation: LayerNorm normalizes each token separately; DyT uses a single global `α`. The nonlinearity of `tanh` supplies the extreme‑value squashing.
  - Affine re-scaling is retained via `γ`/`β`, preserving representational flexibility.

## 4. Key Insights and Innovations
- Empirical reinterpretation of LayerNorm’s role (fundamental insight)
  - Observation: LayerNorm’s aggregated input→output mapping across tokens is S‑shaped, strongly resembling `tanh` (Section 3; Figures 2–4).
  - Significance: Recasts normalization’s global effect not as “pure normalization,” but as “near‑linear around zero + outlier squashing,” clarifying why it stabilizes training.

- A minimalist, drop‑in alternative to normalization (`DyT`) (core contribution)
  - Element‑wise `tanh(αx)` plus standard affine parameters replaces LN/RMSNorm across Transformer blocks (Section 4; Figure 1; Equation 2).
  - No statistics, no reductions, simple kernel—yet comparable or better performance across many tasks (Section 5, Tables 1–6).

- Understanding and leveraging `α` as a learned scale controller (explanatory insight)
  - `α` tracks `1/std` of activations during training and correlates with it after training (Figure 8), showing that DyT learns an implicit “global normalization” scale.
  - Removing `α` hurts performance (Table 8), confirming its necessity.

- Practical training guidelines for LLMs (useful innovation)
  - Tuned `α0` improves LLM training; larger widths need smaller `α0`; attention blocks benefit from higher `α0` than MLP/final blocks (Section 7.2; Table 10; Figure 11; Table 11).
  - This yields stable 7B–70B LLaMA training matching RMSNorm in loss and zero‑shot accuracy (Table 4; Figure 6).

- Strong comparison to other “no‑norm” methods (evidence of significance)
  - DyT outperforms initialization‑based approaches (Fixup, SkipInit) and matches/exceeds σReparam in ViT/MAE settings (Table 9).

## 5. Experimental Analysis
- Evaluation setup (Section 5; Appendix A)
  - “Replace all LN/RMSNorm with DyT” and keep the rest of the architecture unchanged (Figure 1).
  - Hyperparameters: as close as possible to the original training recipes; in most vision/speech/DNA experiments no tuning is needed. For DiT, a small LR search is done on the LN baseline and reused for DyT (Appendix A). For LLMs, `α0` is tuned and an embedding scalar is added (Appendix A; Section 7.2).

- Datasets, tasks, metrics
  - Supervised Vision on ImageNet‑1K (top‑1 accuracy): ViT‑B/L and ConvNeXt‑B/L.
  - Self‑supervised Vision: MAE and DINO pretrain on ImageNet‑1K, then fine‑tune (top‑1 accuracy).
  - Diffusion Models (DiT) on ImageNet‑1K: Fréchet Inception Distance (FID; lower is better).
  - Large Language Models (LLaMA 7B/13B/34B/70B): trained on The Pile to 200B tokens; report pretraining loss and average zero‑shot score across 15 lm‑eval tasks (Table 4).
  - Speech (wav2vec 2.0 on LibriSpeech): validation loss.
  - DNA sequence modeling (HyenaDNA, Caduceus): average accuracy across GenomicBenchmarks datasets.

- Main quantitative results
  - Supervised Vision (Table 1):
    > ViT‑B: 82.3% (LN) → 82.5% (DyT); ViT‑L: 83.1% → 83.6%  
    > ConvNeXt‑B: 83.7% → 83.7%; ConvNeXt‑L: 84.3% → 84.4%
    - Training losses are nearly identical (Figure 5), suggesting similar learning dynamics.
  - Self‑supervised Vision (Table 2):
    > MAE ViT‑B: 83.2% → 83.2%; MAE ViT‑L: 85.5% → 85.4%  
    > DINO ViT‑B (p16): 83.2% → 83.4%; DINO ViT‑B (p8): 84.1% → 84.5%
  - Diffusion (Table 3):
    > DiT‑B FID: 64.9 → 63.9 (better); DiT‑L: 45.9 → 45.7 (better); DiT‑XL: 19.9 → 20.8 (worse)
    - Mostly comparable; one degradation at XL size.
  - LLMs (Table 4; Figure 6):
    > Zero‑shot average and final training loss match RMSNorm across 7B/13B/34B/70B, with at most ±0.01 difference in loss for smaller models.
  - Speech (Table 5):
    > Base: 1.95 → 1.95; Large: 1.92 → 1.91 (slightly better)
  - DNA (Table 6):
    > HyenaDNA: 85.2% → 85.2%; Caduceus: 86.9% → 86.9%

- Ablations, diagnostics, and analysis
  - Squashing is essential (Section 6.1; Table 7; Figure 7):
    > Replacing `tanh` with identity leads to divergence. Squashing with `hardtanh`/`sigmoid` trains, but underperforms `tanh`.
  - `α` is essential (Section 6.1; Table 8):
    > Removing `α` drops ViT‑B top‑1 from 82.5% to 81.1%.
  - `α` dynamics and interpretation (Section 6.2; Figure 8):
    > `α` tracks `1/std` during training; final `α` correlates with `1/std` across layers, supporting the “implicit scale normalization” view.
  - Comparison to other norm‑removal methods (Section 6.3; Table 9):
    > ViT‑B: Fixup 77.2%, SkipInit 74.1%, σReparam 82.5%, DyT 82.8% (LN is 82.3%).  
    > MAE ViT‑L: Fixup 74.1%, SkipInit 74.0%, σReparam 85.4%, DyT 85.8% (LN is 85.5%).
  - Sensitivity to `α0`
    - Non‑LLM tasks: broad plateau; α0 in [0.5, 1.2] usually works (Figure 9). Larger models or higher LRs need smaller `α0` to avoid instability; DyT with `α0=0.5` has stability similar to LN (Figure 10).
    - LLMs: best `α0` depends strongly on model width and block type (Section 7.2):
      > Optimal `α0` (attention / other):  
      > 7B: 0.8 / 0.2; 13B: 0.6 / 0.15; 34B: 0.2 / 0.05; 70B: 0.2 / 0.05 (Table 10)  
      > Width, not depth, primarily determines `α0` (Table 11).
  - Efficiency (Appendix C; Tables 14–15):
    > Without compilation, DyT speeds up the norm layers a lot and yields ≈8% end‑to‑end speedups on LLaMA‑7B. After `torch.compile`, DyT and RMSNorm have similar latency.
  - Failure case in ConvNets with BatchNorm (Appendix D; Table 16):
    > Replacing BN with DyT in ResNet‑50: 76.2% → 68.9%; VGG19: 72.7% → 71.0%. DyT is not a drop‑in replacement for BN.

- Do the experiments support the claims?
  - Breadth: The method is tested across recognition (supervised), self‑supervised, generation (diffusion), speech, DNA, and LLMs, using standard public codebases and recipes (Section 5; Appendix A).
  - Strength: On Transformers with LN/RMSNorm, DyT consistently matches or slightly betters baselines, including at large LLM scales (Table 4).
  - Caveats:
    - Some adjustments exist: DiT learning rate search on the baseline and non‑zero init differences for DyT (Appendix A), and an extra embedding‑scale parameter for LLMs (Appendix A). These are documented and reasonable, but they mean “no‑tuning” has exceptions.
    - Not universal: Fails to replace BatchNorm in classic ConvNets (Appendix D).

## 6. Limitations and Trade-offs
- Scope limitation
  - The positive results primarily cover Transformers using LayerNorm or RMSNorm. DyT is not shown to replace BatchNorm in ConvNets effectively (Appendix D).

- Granularity of normalization effect
  - DyT uses a single scalar `α` shared across channels/tokens. It cannot reproduce per‑token standardization like LN. The outlier suppression comes from the nonlinearity rather than per‑token variance control. This works well empirically but might be suboptimal in settings where token‑wise normalization is crucial.

- Initialization sensitivity in LLMs
  - Large, wide LLMs require careful `α0` selection, differing between attention and other blocks (Section 7.2). This adds a small but non‑negligible tuning burden compared to off‑the‑shelf RMSNorm.

- Potential saturation
  - Because DyT relies on `tanh`, overly large `α` or extreme activations could push many values into saturation, diminishing gradients. The paper mitigates this by learning `α` and shows empirically that `α` tracks `1/std` (Figure 8), but no theoretical guarantees are provided.

- Efficiency gains are situational
  - After compiler optimizations (`torch.compile`), DyT and RMSNorm have similar latency (Appendix C, Table 15). The hoped‑for speedup depends on hardware/kernels and is not guaranteed.

- Theoretical underpinnings
  - The paper provides compelling empirical evidence and a mechanistic interpretation but no formal proof that `tanh(αx)` and LayerNorm are equivalent in any sense; this remains an open theoretical question.

## 7. Implications and Future Directions
- How this changes the landscape
  - It reframes normalization in Transformers as largely “S‑curve squashing with scale adaptation,” not necessarily the computation of per‑token statistics. That opens a path to simpler, norm‑free architectures.
  - For practitioners, this provides a practical alternative when normalization computation is undesirable (e.g., custom accelerators where reductions are expensive) or when kernel simplicity helps deployment.

- Follow‑up research enabled/suggested
  - Theory: Formalize when and why S‑curve squashing plus a learned global scale can substitute for per‑token normalization; analyze gradient flow and sharpness effects under DyT.
  - Variants of DyT:
    - Per‑channel or per‑head `α`; gating `α` by layer depth; dynamic `α` conditioned on attention statistics.
    - Other smooth, zero‑centered squashing functions or learned S‑curves.
  - Beyond Transformers:
    - Investigate why DyT fails to replace BatchNorm in ConvNets (Appendix D) and whether architectural changes (e.g., fewer norm sites, different residual scaling) can make it viable there.
  - Optimization and systems:
    - Kernel fusion with adjacent matmuls; low‑precision and quantization friendliness; custom hardware implementations leveraging no‑reduction property.
  - Applications:
    - Test in reinforcement learning, multi‑modal models, retrieval‑augmented systems, and very long‑context models to probe limits of token‑wise vs. global scaling.

- Practical takeaways
  - If you use Transformers with LN/RMSNorm, you can likely swap in `DyT` (`γ * tanh(αx) + β`) with your existing hyperparameters. Use `α0 = 0.5` by default; for LLMs, start from Table 10/11 heuristics (higher `α0` in attention blocks, smaller with greater width) and add the post‑embedding scalar initialized to `√d` (Appendix A).
  - Expect accuracy parity and similar loss curves (Figures 5 and 6), with occasional small gains. Do not expect universal speedups after compilation (Appendix C).

Overall, this work offers a concise, empirically strong alternative to normalization in Transformers, backed by a compelling mechanistic insight—LayerNorm’s observed S‑curve behavior—which `DyT` emulates with a single learned scale and a saturating nonlinearity.
