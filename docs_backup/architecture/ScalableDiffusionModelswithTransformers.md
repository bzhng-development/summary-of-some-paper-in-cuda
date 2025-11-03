# Scalable Diffusion Models with Transformers

**ArXiv:** [2212.09748](https://arxiv.org/abs/2212.09748)

## 🎯 Pitch

This paper introduces Diffusion Transformers (DiTs), a pure transformer-based backbone for diffusion models that replaces the ubiquitous U-Net architecture and operates efficiently in the latent space of a VAE. By scaling model Gflops through depth, width, or increased token count, DiTs achieve state-of-the-art image synthesis performance on ImageNet benchmarks with substantially lower computational cost. This breakthrough demonstrates that diffusion models can fully leverage transformer architectures, setting the stage for unified, scalable generative modeling across domains and unlocking new avenues for efficiency and cross-domain research.

---

## 1. Executive Summary
This paper replaces the standard U‑Net backbone in diffusion models with a pure transformer, called the Diffusion Transformer (`DiT`), and shows that it scales predictably and efficiently when trained in the latent space of a VAE. By increasing the transformer’s forward computation (Gflops) through depth/width or more tokens, `DiT` achieves state‑of‑the‑art FID on class‑conditional ImageNet at 256×256 (FID 2.27) and 512×512 (FID 3.04), while using far less compute than pixel‑space U‑Nets (Tables 2–3).

## 2. Context and Motivation
- Problem/gap:
  - Diffusion models have advanced image synthesis, but almost all high‑performing variants still use U‑Net backbones (Section 1). This limits cross‑domain architectural unification and inherits convolutional inductive biases that may not be necessary.
  - Prior architectural analyses often rely on parameter counts, which poorly capture complexity for image models where resolution and sequence length dominate compute (Section 2 “Architecture complexity”).
- Importance:
  - Transformers have shown superior scalability in language and vision (Section 2). If diffusion models could also use transformers effectively, they would benefit from shared training recipes, robustness, and clearer scaling laws across domains.
- Prior approaches and shortcomings:
  - Pixel‑space DDPMs based on U‑Nets (e.g., ADM, Section 2) are compute‑heavy at high resolution. Latent Diffusion Models (`LDMs`) reduce compute by running diffusion in a learned latent space, yet still rely on U‑Nets (Section 3.1 “Latent diffusion models”).
  - Conditioning in diffusion has leaned on cross‑attention or adaptive normalization in U‑Nets; it is unclear which conditioning mechanisms best suit transformers (Figure 3, Section 3.2).
- Positioning:
  - This work introduces a transformer‑only backbone (`DiT`) trained in VAE latent space, systematically explores conditioning mechanisms, and studies scaling through forward pass Gflops rather than only parameters. It provides a practical, compute‑efficient alternative to U‑Nets with strong empirical scaling laws (Figures 6–9).

## 3. Technical Approach
High‑level pipeline
- Two‑stage latent diffusion setup (Section 3.1 “Latent diffusion models”):
  1) A pretrained VAE encodes images `x` (e.g., 256×256×3) into latents `z = E(x)` of size 32×32×4 (downsample factor 8).  
  2) The diffusion model operates on latents `z`; after sampling a new latent, the VAE decodes it back to an image `x = D(z)`.
- Diffusion training objective (Section 3.1 “Diffusion formulation”):
  - Forward noising: sample time `t`, draw noise `ε ~ N(0, I)`, and produce `x_t = sqrt(ᾱ_t) x_0 + sqrt(1 − ᾱ_t) ε`.
  - Model learns the reverse process by predicting the noise `ε_θ(x_t, t, c)` (class label `c` is optional), trained with mean‑squared error `||ε_θ − ε||^2` plus a KL term to learn diagonal covariance `Σ_θ` (ADM parameterization).
- Classifier‑free guidance at sampling (Section 3.1 “Classifier‑free guidance”):
  - During training, randomly drop the condition `c` to learn an unconditional “null” embedding.
  - At sampling, compute a guided score `ε̂_θ = ε_θ(x_t, ∅) + s · (ε_θ(x_t, c) − ε_θ(x_t, ∅))`, where `s > 1` controls strength.

`DiT` architecture (Sections 3.2 and Figure 3)
- Inputs as tokens (“Patchify,” Figure 4):
  - Patchify the noised latent `x_t` of shape `I×I×C` into a sequence of `T = (I/p)^2` tokens with hidden size `d`.  
  - `p` is patch size; smaller `p` → more tokens → higher Gflops; parameters barely change.
  - Add sine‑cosine positional embeddings to the tokens.
- Transformer backbone (Table 1; model sizes S, B, L, XL):
  - Standard ViT‑style stack: `N` blocks with multi‑head self‑attention and MLP.
  - Conditioning enters via normalization layers (explained below).
- Conditioning mechanisms compared (Figure 3 right; Section 3.2 “DiT block design”):
  - In‑context tokens: append embeddings of time `t` and class `c` to the token sequence.
  - Cross‑attention: process `t` and `c` as a separate 2‑token memory, add a cross‑attention layer.
  - `adaLN` (adaptive LayerNorm): replace each block’s LayerNorm scale/shift (`γ, β`) with outputs of an MLP driven by the sum of `t` and `c` embeddings.
  - `adaLN‑Zero`: like `adaLN`, but also predict per‑residual scaling factors `α` that are initialized to zero; this initializes each block as the identity, improving stability.
- Why `adaLN‑Zero`?
  - It injects conditioning everywhere through normalization, adds negligible compute, and starts blocks as “do nothing,” which stabilizes learning at scale—an idea inspired by residual network initialization (Section 3.2; Figure 5 shows its empirical dominance).
- Output head (“Transformer decoder,” Section 3.2):
  - Apply a final (adaptive) LayerNorm, then a linear layer maps each token to `p×p×2C` values (for both noise and diagonal covariance). Reshape back to the `I×I×C` grid.

Training and implementation (Section 4; Table 4; Appendix A)
- Dataset: ImageNet, class‑conditional at 256×256 and 512×512.
- Optimizer/schedule: AdamW, constant learning rate 1e‑4, batch 256, no weight decay, horizontal flip augmentation, EMA 0.9999, identical hyperparameters across all DiT variants.
- Diffusion schedule: Linear variance with `t_max = 1000`, same as ADM.
- VAE: pretrained Stable Diffusion VAE (downsample 8; 84M parameters, excluded from DiT parameter counts per Table 4 note).
- Hardware: JAX on TPU‑v3 pods; `DiT‑XL/2` trains at ~5.7 it/s on a v3‑256 pod (Section 4 “Compute”).

Step‑by‑step generative process (sampling)
1) Sample latent noise `x_T ~ N(0, I)`.
2) For `t = T … 1`, run the transformer on the patchified `x_t` with `t` and (optionally) class `c` using `adaLN‑Zero`.
3) Obtain predicted `ε_θ` and `Σ_θ`, optionally apply classifier‑free guidance (`s`), and sample `x_{t−1}` from the learned Gaussian reverse process.
4) Decode final latent `z = x_0` to an image using the VAE decoder.

Why this approach over alternatives
- Replacing U‑Nets with a ViT‑style backbone makes diffusion compatible with the transformer scaling toolkit, including tokenized inputs and compute that grows with sequence length instead of spatial convolutions (Sections 1 and 3.2).
- Conditioning via `adaLN‑Zero` is more compute‑efficient than cross‑attention and empirically better than in‑context tokens (Figure 5).
- Training in latent space keeps compute manageable while preserving image quality (Figure 2 right; Table 6).

## 4. Key Insights and Innovations
1) A pure transformer backbone for diffusion that scales cleanly in latent space
   - What’s new: A ViT‑style stack over latent patches replaces U‑Nets without specialized convolutional structure (Figure 3 left).
   - Why it matters: It demonstrates that the U‑Net inductive bias is not essential for high‑quality diffusion; transformers can be competitive and easier to scale (Section 1; Figure 2 right).

2) `adaLN‑Zero`: a simple, global conditioning method that initializes blocks as identity
   - What’s new: Adaptive LayerNorm with an extra learnable residual scale `α` initialized to zero, so each transformer block starts as an identity mapping (Section 3.2).
   - Impact: It consistently yields the best FID during training while adding negligible compute, outperforming cross‑attention and in‑context conditioning (Figure 5).

3) Compute‑centric scaling law: forward Gflops strongly predict quality, more than parameter count
   - Evidence: Across 12 models (S/B/L/XL × patch sizes 8/4/2), FID improves monotonically with transformer Gflops; correlation −0.93 at 400K steps (Figure 8). Holding parameters roughly fixed and increasing tokens (smaller patch size) significantly improves FID (Figure 6 bottom).
   - Significance: It reframes architectural scaling for diffusion around forward compute, not just parameter counts.

4) State‑of‑the‑art ImageNet results with better compute efficiency than pixel‑space U‑Nets
   - Results:
     - 256×256: `DiT‑XL/2‑G (cfg=1.50)` achieves FID 2.27, surpassing `LDM‑4‑G` (FID 3.60) and StyleGAN‑XL (FID 2.30; Table 2).  
     - 512×512: `DiT‑XL/2‑G (cfg=1.50)` achieves FID 3.04, improving over ADM’s best 3.85 (Table 3).
   - Compute comparison: At 512×512, `DiT‑XL/2` uses 524.6 Gflops vs ADM 1983 Gflops and ADM‑U 2813 Gflops (Table 3; Section 5.1).

5) More sampling steps cannot compensate for insufficient model compute
   - Evidence: Even with 5× higher sampling compute, a smaller model (e.g., `L/2` with 1000 steps) trails a larger model (`XL/2` with 128 steps) in FID‑10K (25.9 vs 23.7; Figure 10).
   - Takeaway: Invest in model compute (Gflops) over just increasing sampling iterations.

## 5. Experimental Analysis
Evaluation setup
- Datasets:
  - ImageNet, class‑conditional, at 256×256 and 512×512 (Section 4).
- Metrics: FID‑50K (main), sFID, Inception Score (IS), Precision/Recall (Sections 4 and 5.1). FID computed with 250 DDPM steps via ADM’s TensorFlow evaluator to ensure comparability (Section 4).
- Baselines:
  - U‑Net diffusion: ADM, ADM‑U, ADM‑G; latent U‑Net: LDM‑4/8 (Tables 2–3).
  - GANs: BigGAN‑deep; StyleGAN‑XL (Tables 2–3).
- Model grid and compute:
  - 12 DiT variants: S/B/L/XL × patch sizes 8/4/2 (Figure 6; Table 4).  
  - Gflops range from 0.36 to 118.64 at 256×256; 524.6 at 512×512 (Table 4).
  - Training hyperparameters kept constant across all variants (Section 4).

Main quantitative results
- Scaling trends:
  - “Increasing transformer size” (depth/width) at fixed patch size uniformly reduces FID across training (Figure 6 top).  
  - “Decreasing patch size” (more tokens) at fixed model size also reduces FID (Figure 6 bottom).
  - Forward Gflops vs FID shows a strong negative correlation: −0.93 (Figure 8).
  - Larger models are more compute‑efficient when plotting FID vs training compute (Figure 9).
- State‑of‑the‑art on ImageNet:
  - 256×256 (Table 2):
    > `DiT‑XL/2‑G (cfg=1.50)`: FID 2.27; IS 278.24; Precision 0.83; Recall 0.57.  
    > `LDM‑4‑G (cfg=1.50)`: FID 3.60; IS 247.67; Precision 0.87; Recall 0.48.  
    `DiT` beats prior diffusion models and matches/approaches leading GANs on FID while offering higher recall than LDM variants at tested guidance scales.
  - 512×512 (Table 3):
    > `DiT‑XL/2‑G (cfg=1.50)`: FID 3.04; IS 240.82; Precision 0.84; Recall 0.54.  
    Improves upon ADM best FID 3.85 with far less compute (524.6 vs 1983–2813 Gflops; Table 3 and Section 5.1).
- Conditioning ablations (Figure 5):
  > `adaLN‑Zero` consistently outperforms in‑context tokens and cross‑attention across training at similar or lower compute.
- Additional metrics and loss curves:
  - The compute‑centric scaling trend extends to sFID, IS, Precision, and Recall (Figure 12).  
  - Larger models achieve lower training loss faster and settle at better optima (Figure 13).
- Sampling compute vs model compute (Figure 10):
  > Scaling sampling steps cannot close the FID gap to larger models; invest in model Gflops.
- VAE decoder ablation (Table 5):
  > Different pretrained decoders (original, ft‑MSE, ft‑EMA) lead to very similar results; final SOTA numbers use ft‑EMA.

Qualitative results
- Visual samples improve with more Gflops—via larger backbones or more tokens—holding everything else fixed (Figure 7).
- High‑quality, diverse class‑conditional samples at 256×256 and 512×512 (Figure 1; Figures 11, 14–33 show uncurated grids across guidance scales).

Do the experiments support the claims?
- The paper’s central claims—transformers can replace U‑Nets in diffusion, scale well via forward compute, and achieve SOTA with better compute efficiency—are supported by:
  - A systematic model/patch grid (12 variants) with consistent training settings (Figure 6; Table 4).
  - Clear compute analyses (Figures 8–10).
  - Strong benchmarks at two resolutions (Tables 2–3).
- Robustness and checks:
  - Conditioning ablations (Figure 5), multi‑metric scaling (Figure 12), training loss trends (Figure 13), and VAE decoder swaps (Table 5) all point in the same direction.

## 6. Limitations and Trade-offs
- Reliance on latent space:
  - Results depend on a pretrained VAE; artifacts or biases from compression can cap ultimate fidelity and affect semantic alignment. Pixel‑space DiT is not explored (Section 3.1, “could be applied to pixel space without modification” but untested).
- Scope of conditioning:
  - Experiments are class‑conditional. Text‑conditional setups (which often use cross‑attention) are only suggested as future work (Conclusion). The finding that `adaLN‑Zero` beats cross‑attention may not transfer unchanged to text prompts.
- Compute and hardware:
  - Although compute‑efficient relative to pixel‑space U‑Nets, top models are still expensive (e.g., `DiT‑XL/2` at 524.6 Gflops for 512×512; Table 4) and trained on TPU v3‑256 (Section 4). Memory footprint and wall‑clock cost remain high.
- Fair accounting:
  - Reported DiT parameter and Flop counts exclude the VAE (84M params; Table 4 note). End‑to‑end costs in practical deployments should include VAE encode/decode overhead.
- Limited resolutions and modalities:
  - Experiments cover 256×256 and 512×512 images on ImageNet. Generalization to higher resolutions, other datasets, video, audio, or 3D is not evaluated.
- Diversity vs fidelity trade‑offs:
  - As with classifier‑free guidance in general, higher guidance scales increase fidelity but can reduce diversity (Appendix B; Figures 14–33).

## 7. Implications and Future Directions
- Field impact:
  - Establishes transformers as a viable, scalable backbone for diffusion, aligning image generation with the broader transformer ecosystem. The compute‑centric scaling perspective (Gflops over parameters) provides a clearer design dial for future models (Figures 6–9, 12).
- Follow‑up research enabled:
  - Text‑to‑image DiT: Substitute labels with text embeddings; revisit conditioning (e.g., combine `adaLN‑Zero` with cross‑attention for language tokens). The Conclusion explicitly suggests `DiT` as a drop‑in for DALL·E 2 or Stable Diffusion.
  - Tokenization strategies: Explore dynamic patch sizes, learned tokenization, or hybrid early convolutions to reduce sequence length while retaining global modeling.
  - Pixel‑space DiT: Test whether the compute scaling law holds without a VAE, possibly with improved training tricks or multi‑grid tokenization.
  - Scaling laws: Extend the Gflops‑vs‑quality analysis to more datasets, resolutions, and modalities; build predictive models of training compute vs performance (Figure 9 hints at compute efficiency cross‑overs).
  - Efficient inference: Since more sampling steps cannot replace model compute (Figure 10), focus on architectural efficiency and distillation to speed sampling without quality loss.
- Practical applications:
  - High‑fidelity, class‑conditional generation under tighter compute budgets than pixel‑space U‑Nets (e.g., dataset augmentation, creative tools).
  - A unified architecture for multimodal generative modeling where transformers dominate, simplifying system integration and scaling strategies across tasks.

In short, `DiT` demonstrates that a pure transformer, equipped with a simple `adaLN‑Zero` conditioning mechanism and trained in latent space, can deliver state‑of‑the‑art diffusion performance with clear and favorable scaling behavior tied to forward compute. The work opens a practical path to transformer‑native, compute‑aware generative modeling across modalities.
