# Transformers without Normalization

**ArXiv:** [2503.10622](https://arxiv.org/abs/2503.10622)
**Authors:** Jiachen Zhu, Xinlei Chen, Kaiming He, Yann LeCun, Zhuang Liu
**Institutions:** Meta (Fair), New York University, MIT, Princeton University

## 🎯 Pitch

The paper presents Dynamic Tanh (DyT), an innovative element-wise operation that replaces normalization layers in Transformers, maintaining or slightly enhancing model performance across diverse domains. By eliminating the need for computationally heavy normalization processes, DyT simplifies architectures and improves portability, particularly benefiting hardware where reduction operations are costly, thereby redefining the essential mechanisms behind training stability and model efficacy.

---

## 1. Executive Summary (2–3 sentences)
This paper introduces Dynamic Tanh (`DyT`), a single element‑wise operation `DyT(x) = γ · tanh(αx) + β` that can replace normalization layers (LayerNorm/RMSNorm) inside Transformers. Across vision, speech, diffusion, DNA sequence modeling, and large language models (LLMs), models with DyT match or slightly exceed the performance and stability of their normalized counterparts while requiring little or no hyperparameter tuning (Tables 1–6, Figures 5–6).

## 2. Context and Motivation
- Gap being addressed
  - Modern deep networks nearly always include normalization layers like LayerNorm (LN) and RMSNorm to stabilize training and improve generalization. The paper asks whether such normalization is truly indispensable in Transformers and, if not, what minimal mechanism is actually needed.
- Why it matters
  - Practical: Removing normalization removes reductions over dimensions (means/variances), potentially simplifying kernels and improving portability to hardware where reductions are costly. It also simplifies architectural design and may ease integration with fused kernels.
  - Scientific: Understanding what normalization layers “do” clarifies their role in optimization and representation. The paper empirically characterizes LN’s behavior and proposes a simpler primitive that appears to capture the essential effect.
- Prior approaches and shortcomings
  - Initialization-only strategies (e.g., Fixup; SkipInit) and weight reparameterization or spectral control (e.g., σReparam) can train without normalization but often require delicate learning-rate tuning and still underperform on strong baselines (Table 9 shows 4–10+ percentage‑point gaps on ImageNet-1K for ViT and MAE).
  - Other work removes or relocates normalization but keeps some form of normalization or requires extensive fine‑tuning.
- Positioning
  - The paper starts from an empirical analysis of LN’s input–output behavior (Figures 2–4), observes it is S‑shaped and tanh‑like, and introduces a drop‑in element‑wise layer that reproduces this behavior without computing statistics. The claim is not that normalization is unnecessary in principle, but that Transformers can achieve parity when the “squash extreme values while stay nearly linear near zero” behavior is provided by a much simpler mechanism.

## 3. Technical Approach
Step 1: Examine what normalization layers compute
- General normalization form (Equation 1): `normalize(x) = γ * (x - μ) / sqrt(σ^2 + ε) + β`. LN computes `μ, σ` per token; RMSNorm drops mean-centering but scales by the RMS of features. These operations are linear for a single token’s features but vary token‑by‑token.

Step 2: Empirical probe of LN’s input–output mapping
- Method: For trained ViT, wav2vec 2.0, and DiT models, collect tensors immediately before and after LN (pre‑affine) and plot output vs. input elementwise (Figure 2).
- Observation: Deeper LNs produce S‑shaped curves closely resembling a scaled `tanh` (Figure 3). Early layers look nearly linear; deeper layers show a clear “squash the extremes” effect.
- Why LN appears non‑linear overall despite being linear per token: When points are colored by token, each token’s mapping is a straight line with a different slope (due to different per‑token standard deviations). Overlaying all tokens yields an aggregate S‑shape (Figure 4, left). When colored by channel, a few channels have extreme ranges and get squashed most (Figure 4, right).

Step 3: Hypothesis distilled
- The “essential” effect of LN in Transformers is:
  - Keep small/typical activations roughly linear (center of the curve).
  - Disproportionately squash extreme activations so they do not dominate downstream computation.
- This is analogous to a smooth saturating nonlinearity with learnable scale.

Step 4: Replace LN with a direct element‑wise squasher
- Dynamic Tanh (Equation 2): `DyT(x) = γ · tanh(αx) + β`.
  - `tanh(·)` provides the saturating S‑curve.
  - `α` is a learnable scalar shared across the whole layer that adjusts how “wide” the linear regime is.
  - `γ, β` are per‑channel scale and shift (same shapes as in normalization layers). They preserve representational flexibility of the downstream layer.
- Implementation: Replace every LN/RMSNorm in attention blocks, feed‑forward blocks, and the final pre-output normalization with one DyT (Figure 1). Pseudocode is given in Algorithm 1; it is a few lines of element‑wise math.

Step 5: Initialization and minimal architectural additions
- Default initialization works broadly: `γ = 1`, `β = 0`, and `α0 = 0.5` for non‑LLM models (Section 4; Section 7.1 and Figure 9).
- For LLMs, training improves if `α0` is tuned and a single scalable scalar is inserted after the token embedding to set a reasonable activation scale at the start of training (Appendix A). Optimal `α0` tends to be higher in attention blocks than in FFN blocks (Table 10; Figure 11), and smaller for wider models (Table 11).

Why this design over alternatives
- Squashing function choice: ablations show the squasher is essential; replacing `tanh` with identity causes divergence; `hardtanh` and `sigmoid` train but underperform `tanh` (Table 7).
- Learnable `α` is needed: removing `α` degrades accuracy for all squashers (Table 8).
- Mechanistic support: During and after training, learned `α` tracks the inverse standard deviation of the pre‑DyT activations (Figure 8), approximating the scale‑setting role of normalization while using a single scalar rather than per‑token statistics.

## 4. Key Insights and Innovations
- Empirical decoding of LayerNorm’s behavior into a tanh‑like mapping
  - Novelty: Rather than assuming LN is “just linear standardization,” the paper shows its global element‑wise mapping across tokens is S‑shaped and closely matches `tanh` (Figures 2–4). This reframes normalization as a smooth “squash‑extremes” mechanism with a large linear center.
  - Significance: Provides an interpretable and testable target behavior for normalization substitutes.
- Dynamic Tanh (`DyT`) as a drop‑in, statistics‑free replacement
  - Novelty: `DyT(x) = γ · tanh(αx) + β` replaces the entire normalization computation (no means/variances, no reductions) with an element‑wise function plus learnable scale/shift (Equation 2, Algorithm 1).
  - Significance: In practice, DyT achieves parity or small gains across diverse Transformer applications and sizes (Tables 1–6), challenging the view that per‑token normalization statistics are essential.
- Mechanistic link between learned `α` and activation scale
  - Novelty: `α` learns to track `1 / std(preactivations)` throughout training (Figure 8).
  - Significance: Explains why DyT stabilizes training—`α` maintains activations in the wide linear region of `tanh` and lets the tails saturate, mimicking LN’s outlier control.
- Practical recipe for LLMs: role‑specific and width‑aware `α0`
  - Novelty: Tuning `α0` differently for attention vs. other blocks materially improves LLM pretraining, and optimal values shrink with model width (Tables 10–11; Figure 11).
  - Significance: Converts DyT from a general proof‑of‑concept into a practical norm‑free alternative for billion‑parameter language models.

## 5. Experimental Analysis
Evaluation setup
- Models and domains (Section 5)
  - Supervised vision: ViT‑B/L and ConvNeXt‑B/L on ImageNet‑1K.
  - Self‑supervised vision: MAE and DINO with ViT backbones, evaluated by fine‑tuning on ImageNet‑1K.
  - Diffusion: DiT‑B/L/XL on ImageNet‑1K; metric is FID (lower is better).
  - Speech: wav2vec 2.0 Base/Large pretraining on LibriSpeech; report validation loss.
  - DNA sequence modeling: HyenaDNA and Caduceus; accuracy on GenomicBenchmarks.
  - LLMs: LLaMA 7B/13B/34B/70B trained on The Pile for 200B tokens; report training loss and average zero‑shot accuracy on 15 lm‑eval tasks (Appendix A for dataset and protocol details).
- Baselines and hyperparameters
  - Wherever possible, use the exact training recipe of the normalized baseline; only replace LN/RMSNorm with DyT (Section 5). For DiT, a brief LR search was applied to both LN and DyT; for LLMs, a single learnable scalar is added after embeddings and `α0` is tuned per Section 7.

Main quantitative results
- Supervised vision (Table 1; Figure 5)
  - > ViT‑B: 82.3% (LN) → 82.5% (DyT); ViT‑L: 83.1% → 83.6%.
  - > ConvNeXt‑B: 83.7% → 83.7%; ConvNeXt‑L: 84.3% → 84.4%.
  - Training curves are nearly identical (Figure 5), suggesting comparable optimization dynamics.
- Self‑supervised vision (Table 2)
  - > MAE ViT‑B: 83.2% (LN) vs 83.2% (DyT); MAE ViT‑L: 85.5% vs 85.4%.
  - > DINO ViT‑B p16: 83.2% vs 83.4%; p8: 84.1% vs 84.5%.
- Diffusion (Table 3)
  - > FID↓: DiT‑B 64.9 (LN) → 63.9 (DyT); DiT‑L 45.9 → 45.7; DiT‑XL 19.9 → 20.8 (slightly worse).
- Speech pretraining (Table 5)
  - > wav2vec 2.0 Base: loss 1.95 (LN) vs 1.95 (DyT); Large: 1.92 vs 1.91.
- DNA sequence modeling (Table 6)
  - > HyenaDNA: 85.2% (LN) vs 85.2% (DyT); Caduceus: 86.9% vs 86.9%.
- LLMs (Table 4; Figure 6; Section 7.2)
  - > Zero‑shot average: LLaMA 7B/13B/34B/70B all identical to three decimals (0.513/0.529/0.536/0.549).
  - > Final losses within 0.01 of RMSNorm for 7B and 13B and identical for 34B/70B.
  - Loss curves overlap closely during pretraining (Figure 6).
- Comparison to other norm‑free training methods (Table 9)
  - > On ImageNet‑1K with ViT‑B/L and MAE ViT‑B/L, DyT consistently outperforms Fixup and SkipInit by large margins and slightly edges σReparam.

Ablations, diagnostics, and robustness
- Squasher necessity and choice (Table 7; Figure 7)
  - Identity in place of `tanh` causes divergence; `hardtanh`/`sigmoid` are stable but worse than `tanh`. This confirms the centrality of “bounded squashing.”
- Role of `α` (Table 8)
  - Removing `α` drops ViT‑B accuracy from 82.5% to ~81% even with squashing, showing scale control is critical.
- Dynamics and values of `α` (Figure 8)
  - `α` evolves in tandem with `1/std` of inputs during training and correlates with it at convergence, explaining stability and suggesting an approximate normalization effect without statistics.
- Sensitivity of `α0` (Section 7; Figures 9–11; Tables 10–11)
  - Non‑LLM tasks: broad plateau of good performance for `α0` in [0.5, 1.2] (Figure 9). Very large ViT‑L with high LR can diverge if `α0` is too big; reducing LR or `α0` restores stability (Figure 10).
  - LLMs: Best `α0` is higher in attention than in FFN/last layers and shrinks with model width—e.g., 7B: 0.8(attn)/0.2(other); 70B: 0.2/0.05 (Table 10). Heatmaps (Figure 11) visualize loss improvements with these settings. Width dictates `α0` more than depth (Table 11).

Do the experiments support the claims?
- Breadth: The study spans recognition and generation, supervised and self‑supervised learning, and multiple modalities—strong evidence of generality.
- Strength: Parity with RMSNorm on multi‑billion‑parameter LLMs (loss curves in Figure 6 and Table 4) is particularly convincing.
- Nuance: Slight regressions exist (e.g., DiT‑XL FID 20.8 vs 19.9), but overall parity holds. The added embedding scalar for LLMs is a small but real architectural tweak (Appendix A).

Efficiency observations (Appendix C)
- On uncompiled Hugging Face LLaMA‑7B, DyT layer time is lower than RMSNorm and modestly improves end‑to‑end runtime (Table 14), but with `torch.compile` the advantage disappears (Table 15). DyT is element‑wise (no reductions), which could be beneficial on reduction‑limited hardware, but benefits are not guaranteed on well‑optimized GPU stacks.

## 6. Limitations and Trade-offs
- Not a drop‑in for all normalization types
  - Replacing BatchNorm in ConvNets (ResNet‑50, VGG19) degrades accuracy notably (Table 16). DyT appears most suitable where LN/RMSNorm are standard (i.e., Transformers).
- LLM recipe is not “zero‑change”
  - Successful LLM training uses one extra learnable scalar after embeddings and tuned `α0` per block type and width (Appendix A; Section 7.2). This is still simple, but not a literal 1:1 swap.
- Performance parity, not consistent superiority
  - Most tasks show parity or small gains; some settings regress slightly (e.g., DiT‑XL FID). There is no universal win across all scales and domains.
- Efficiency is context‑dependent
  - Despite being element‑wise, DyT did not yield speedups once layers were compiled (Table 15). Gains may depend on hardware, kernel fusion, and compiler maturity.
- Statistical adaptivity vs. global scale
  - LN adapts per token and per sample. DyT uses a single `α` per layer, so it cannot rescale different tokens differently within a batch. The “per‑token linear, globally S‑shaped” effect emerges from tanh saturation rather than per‑token statistics; edge cases with highly heterogeneous token statistics might stress DyT.
- Assumptions
  - The central hypothesis is that “squash extremes + learned global scale” captures what matters about LN in Transformers. While well supported empirically, it remains a modeling assumption rather than a proof.

## 7. Implications and Future Directions
- Conceptual shift
  - Normalization’s indispensability is questioned: a simple, learnable saturating nonlinearity paired with per‑channel affine parameters appears sufficient for Transformer stability and performance. This reframes normalization as “robust activation shaping” rather than “per‑token standardization.”
- Practical applications
  - Model simplification: Element‑wise DyT can simplify kernels and may be easier to fuse with adjacent matrix multiplies. It could benefit accelerators where reductions are expensive.
  - Deployment: For inference‑only deployments, removing reductions may reduce latency variance and improve portability, especially in small‑batch or sequential regimes.
- Research directions
  - Extend beyond Transformers: Can DyT be augmented to handle BatchNorm‑style roles (Appendix D shows current shortcomings)? Hybrids that combine DyT with light-weight per‑token or per‑channel statistics are promising.
  - Token‑aware variants: Introduce a small number of learnable `α`s (e.g., per head, per group of channels, or conditional on token type) to reintroduce some adaptivity without full statistics.
  - Alternative squashers: While `tanh` wins among tested options (Table 7), other smooth bounded functions or learnable parametric families could provide better trade‑offs between linear range and saturation.
  - Theory: Formalize why the global S‑shape arises in LN (Figures 2–4) and when an element‑wise saturating nonlinearity suffices for stable gradient flow in deep residual Transformers.
  - Scaling laws for `α0`: The width‑driven schedule for `α0` (Tables 10–11) hints at predictable scaling rules; deriving and validating these across architectures could make DyT fully “plug‑and‑play.”

Overall, the work provides a clear mechanistic story, an extremely simple replacement layer, and broad empirical evidence that Transformers can train stably and perform competitively without normalization statistics. The strongest signals are the LLM results (Table 4, Figure 6) and the mechanistic tracking of `α` to activation scale (Figure 8), which together make a persuasive case that “squash extremes + learn a scale” captures the essential role of LayerNorm in these architectures.
