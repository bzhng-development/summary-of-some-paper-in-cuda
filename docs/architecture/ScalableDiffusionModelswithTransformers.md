# Scalable Diffusion Models with Transformers

**ArXiv:** [2212.09748](https://arxiv.org/abs/2212.09748)

## 🎯 Pitch

This paper introduces Diffusion Transformers (DiT), replacing the conventional U-Net backbone in image diffusion models with a pure Vision Transformer architecture. The authors demonstrate that DiT scales smoothly with compute and achieves state-of-the-art image generation quality, establishing a strong empirical link between model FLOPs and sample fidelity. This innovation not only challenges longstanding assumptions about convolutional inductive biases but also paves the way for unified, scalable architectures across vision and language, unlocking easier cross-domain research and more efficient training practices.

---

## 1. Executive Summary
This paper replaces the long‑standing U‑Net backbone used in diffusion image generators with a pure Vision Transformer and shows that this “Diffusion Transformer” (`DiT`) scales smoothly with compute and achieves state‑of‑the‑art results on class‑conditional ImageNet. A central finding is a strong, quantitative link between forward‑pass compute (measured in `Gflops`) and sample quality (measured by FID), plus a transformer‑specific conditioning/initialization trick (`adaLN‑Zero`) that makes training stable and efficient.

## 2. Context and Motivation
- Problem/gap:
  - Diffusion models dominate high‑quality image generation but almost universally rely on the convolutional U‑Net architecture introduced in early DDPM work. The field lacked a careful study of whether the U‑Net inductive bias is necessary and how a transformer backbone would scale for diffusion (Section 1).
  - Model comparison often uses parameter counts, which ignore factors like input resolution and sequence length. This can be misleading for generative architectures whose per‑token compute dominates runtime (Architecture complexity discussion, p.3).

- Why it matters:
  - If a transformer backbone works as well or better, diffusion models can inherit the transformer ecosystem’s favorable scaling, training practices, and cross‑domain unification, simplifying research and deployment across vision and language (Section 1).

- Prior approaches and shortcomings:
  - U‑Net based DDPMs and variants (e.g., ADM, LDM) achieve strong results but are tied to convolutional hierarchies and architectural conventions tuned over years (Related Work, p.2–3).
  - Prior transformer uses in diffusion largely targeted non‑spatial latents or autoregressive modeling, not a full diffusion backbone for images (Related Work, p.2).

- Positioning:
  - The paper introduces `Diffusion Transformers (DiT)`, a ViT‑style backbone that operates on latent patches rather than pixels to keep compute manageable, and studies scaling using `Gflops` (Sections 3 and 4). It then compares DiT to U‑Nets on ImageNet and analyzes scaling behavior across 12 model/patch configurations (Figures 2, 6, 7, 8).

## 3. Technical Approach
This section explains the full pipeline from data to generated image, and the core design choices.

- Data space: latent diffusion rather than pixels
  - Images are encoded by a pre‑trained variational autoencoder (`VAE`) from Stable Diffusion (Diffusion, Section 4, p.6). At 256×256 pixels the latent `z` has shape 32×32×4; at 512×512 it is 64×64×4. Diffusion is trained and sampled in this latent space (LDM, Section 3.1).

- Diffusion training objective (DDPM refresher; Section 3.1):
  - Forward noising: real image `x0` is gradually perturbed to `xt` by adding Gaussian noise: q(xt|x0) = Normal(√ᾱt x0, (1−ᾱt) I).
  - Reverse denoising: train a network to predict the noise `εθ(xt, c)` and (diagonal) covariance `Σθ(xt, c)` so that sampling `xt−1 ∼ pθ(xt−1|xt, c)` inverts the process.
  - Loss: main term is mean‑squared error between predicted and true noise (Equation “Lsimple”, p.3). The covariance is trained with the full KL term (Nichol & Dhariwal trick, p.3).

- Conditioning mechanism: classifier‑free guidance (CFG; Section 3.1)
  - During training, class labels `c` are randomly dropped and replaced by a learned “null” embedding. At sampling, combine unconditional and conditional predictions:
    ε̂θ(xt, c) = εθ(xt, ∅) + s · (εθ(xt, c) − εθ(xt, ∅)) with guidance scale `s > 1` (p.3–4).
  - This generally improves fidelity at the cost of diversity (used in final benchmarks; Tables 2–3).

- From latent grid to transformer tokens (“patchify”; Section 3.2; Figure 4):
  - The 2D latent `z` is split into non‑overlapping p×p patches; each patch is linearly embedded to a token. Sequence length is T = (I/p)^2 where `I` is the latent spatial size (e.g., I=32 for 256×256 images). Smaller `p` → more tokens → higher `Gflops`, with negligible change in parameters (p.4).
  - Standard sine‑cosine positional embeddings are added (p.4).

- Transformer backbone and conditioning (Figure 3; p.4–5):
  - A stack of `N` DiT blocks operates on tokens. Several conditioning strategies are compared:
    1) In‑context conditioning: append two tokens for time `t` and class `c` to the sequence; no architectural changes.
    2) Cross‑attention: add a multi‑head cross‑attention layer that queries the 2‑token condition sequence; ≈15% `Gflops` overhead (p.5).
    3) Adaptive LayerNorm (`adaLN`): replace standard LayerNorm with a version whose scale/shift `(γ, β)` are learned from the combined `t+c` embedding; minimal compute overhead (p.5).
    4) `adaLN‑Zero` (new): like `adaLN` but also predict per‑residual scaling gates `α` applied right before residual connections, and initialize those gates to zero so each transformer block starts as the identity function (p.5). This mirrors zero‑init tricks known to stabilize deep ResNets.

- Output head (Transformer decoder; p.5):
  - Apply a final LayerNorm and a linear layer to each token to produce p×p×2C values, where `C` is latent channels (4 here). The 2C channels are split into predicted noise and predicted diagonal covariance, then reshaped back to the latent grid.

- Model scale (Table 1; p.5) and design grid:
  - Four ViT‑style sizes: `DiT‑S`, `B`, `L`, `XL` with increasing depth/width. Combined with three patch sizes `p ∈ {8,4,2}` → 12 models total (e.g., `DiT‑XL/2` is XL with p=2).
  - Gflops span roughly 0.36 to 118.6 at 256×256; the 512×512 `XL/2` uses 524.6 Gflops (Table 4).

- Training setup (Section 4; p.6):
  - Common hyperparameters across all models: AdamW, constant LR 1e−4, batch 256, only horizontal flips, no LR warmup or regularization, EMA decay 0.9999. Quote:
    > “We use a constant learning rate of 1×10−4, no weight decay and a batch size of 256… training was highly stable across all model configs” (p.6).
  - Diffusion schedule: 1000 linear betas from 1e−4 to 2e−2, embeddings for time/labels, 250 sampling steps for FID evaluation (p.6). Implemented in JAX on TPU v3 pods (Compute, p.6).

- Compute modeling:
  - Architectural complexity tracked primarily via forward `Gflops` (p.3). For training compute, they estimate:
    training compute ≈ (model `Gflops`) × (batch size) × (steps) × 3
    where “3” accounts for forward + backward passes (Figure 9 caption).

## 4. Key Insights and Innovations
1) A pure ViT backbone can replace the U‑Net in DDPMs without loss—and with gains.
   - The `DiT` family consistently improves as either transformer size increases or patch size decreases (Figures 2 left and 6). Visual samples show clearly higher fidelity as `Gflops` rise (Figure 7).

2) `adaLN‑Zero`: transformer‑native conditioning that initializes each block as identity.
   - Mechanism: predict LayerNorm `(γ, β)` and residual gates `α` from the combined time+label embedding; initialize `α=0` so residual paths are off at the start (Figure 3, p.5).
   - Evidence: On the largest setup (`XL/2`) this conditioning beats cross‑attention and in‑context conditioning at every point in training (Figure 5). Quote:
     > “At 400K training iterations, the FID achieved with the adaLN‑Zero model is nearly half that of the in‑context model” (p.6).

3) Compute–quality scaling law using `Gflops`.
   - Across 12 models, FID strongly correlates with forward `Gflops` regardless of whether compute comes from depth/width or token count; correlation −0.93 (Figure 8). Holding parameters roughly fixed and increasing tokens still improves FID (Figure 6 bottom), showing that compute—not just parameter count—drives quality here.

4) Model compute cannot be replaced by more sampling steps.
   - Increasing the number of diffusion sampling steps (test‑time compute) helps but does not close the gap to larger backbones. Example (Figure 10):
     > `DiT‑L/2` with 1000 steps uses ~80.7 Tflops per sample and still has worse FID‑10K (25.9) than `DiT‑XL/2` with 128 steps (15.2 Tflops, FID‑10K 23.7).

5) State‑of‑the‑art ImageNet results with competitive compute.
   - On ImageNet 256×256, `DiT‑XL/2` with CFG=1.5 achieves FID‑50K 2.27, improving over prior diffusion and even over the best GAN baseline (StyleGAN‑XL, FID 2.30) (Table 2). At 512×512, `DiT‑XL/2` achieves FID 3.04, best among diffusion baselines and using far fewer `Gflops` than pixel‑space ADM variants (Table 3).

Together, these are more than incremental tweaks: they establish a transformer‑first design space for diffusion, a conditioning scheme that makes it work well, and a compute‑centric lens that predicts performance.

## 5. Experimental Analysis
- Evaluation setup
  - Dataset: ImageNet, class‑conditional at 256×256 and 512×512 (Section 4).
  - Metrics: FID‑50K with 250 DDPM steps as the main metric, plus Inception Score, sFID, and Precision/Recall (p.6). They use ADM’s TensorFlow evaluation suite to reduce implementation variance in FID (p.6). Note: FID measures distance between the feature distributions of generated vs real images under the Inception network; lower is better.
  - Baselines: ADM, ADM‑U/ADM‑G (pixel‑space U‑Net DDPM variants), LDM‑4/LDM‑8 (latent U‑Net diffusion), GANs (BigGAN‑deep, StyleGAN‑XL) (Tables 2–3).
  - Design grid and training regime:
    - 12 DiT models spanning four sizes (S/B/L/XL) and three patch sizes (p=8/4/2), trained for up to 400K steps for scaling analyses (Figure 6; Table 4). Final `XL/2` models are trained much longer: 7M steps for 256×256 and 3M for 512×512 (Table 4).
    - Identical optimization hyperparameters across all models (Section 4).

- Main quantitative results
  - Scaling behavior:
    - Increasing transformer size at fixed patch size reduces FID across training (Figure 6 top).
    - Decreasing patch size at fixed model size (i.e., more tokens) also reduces FID (Figure 6 bottom).
    - Strong global correlation between `Gflops` and FID at 400K steps (−0.93; Figure 8). Inception Score and Precision also rise with `Gflops` (Figure 12).
    - Larger models are more compute‑efficient: for the same total training compute, bigger DiTs achieve lower FID (Figure 9).
    - Visual confirmation: same noise seed and class produce higher‑quality images as either transformer size or token count grows (Figure 7).

  - Conditioning ablation:
    - `adaLN‑Zero` dominates cross‑attention and in‑context across training (Figure 5), with less `Gflops` than cross‑attention (≈15% overhead for cross‑attention, p.5).

  - State‑of‑the‑art benchmarks:
    - ImageNet 256×256 (Table 2):
      > `DiT‑XL/2‑G (cfg=1.50)` FID‑50K = 2.27, Inception Score = 278.24, Precision = 0.83, Recall = 0.57.
      - This improves over LDM‑4‑G at cfg=1.50 (FID 3.60) and over StyleGAN‑XL (FID 2.30). It also shows higher recall than LDM variants at the same guidance scales (p.9).
      - Without guidance, the 7M‑step model achieves FID‑50K 9.62 (Table 2; Table 4), showing CFG’s importance for fidelity.
    - ImageNet 512×512 (Table 3):
      > `DiT‑XL/2‑G (cfg=1.50)` FID‑50K = 3.04, beating prior diffusion models (ADM‑G,U best = 3.85) with far fewer `Gflops` (524.6 vs 1983–2813 for ADM variants; Table 3).
      - StyleGAN‑XL still has a lower FID (2.41) at this resolution; the contribution here is SOTA among diffusion models plus compute efficiency.

  - Sampling‑compute vs model‑compute (Figure 10):
    - Even extreme increases in sampling steps for a smaller model cannot match the FID of a larger model at moderate steps; e.g., `L/2` (1000 steps) vs `XL/2` (128 steps) example quoted above.

  - Additional ablations and checks:
    - VAE decoder choice: swapping between original LDM decoder and two fine‑tuned Stable Diffusion decoders yields similar scores (Table 5); the best (ft‑EMA) gives the headline results.
    - Classifier‑free guidance applied to only the first 3 of 4 latent channels performs comparably to guiding all channels after rescaling the guidance factor (Appendix A: “Classifier‑free guidance on a subset of channels,” p.12).
    - Training loss curves consistently improve with scale (Figure 13).

- Do the experiments support the claims?
  - The scaling law is supported by a broad sweep (12 architectures × multiple patch sizes), consistent trends across training, and correlation analyses (Figures 6, 8, 9, 12).
  - Conditioning conclusions are backed by direct ablations holding everything else fixed (Figure 5).
  - Benchmark claims rely on standardized FID evaluation and widely used baselines (Tables 2–3). The 256×256 result is state‑of‑the‑art across generative models; the 512×512 result is SOTA among diffusion methods but not against the best GAN.

## 6. Limitations and Trade-offs
- Reliance on a pre‑trained VAE:
  - All training and sampling occur in latent space, not pixels (Section 3.1; Diffusion). This assumes the VAE preserves information needed for high‑fidelity synthesis. While decoder ablations look favorable (Table 5), performance could still be bounded by the autoencoder’s capacity or bias.

- Class‑conditional only:
  - Experiments are on class‑conditional ImageNet. Extension to text‑to‑image or unconditioned settings is discussed as future work (Conclusion, p.9), but not demonstrated here.

- Compute demands:
  - Although compute‑efficient relative to pixel‑space U‑Nets, the best results come from very large models trained for millions of steps (e.g., `XL/2` at 7M steps). Training required TPU v3‑256 pods with ~5.7 it/s for the largest model (Compute, p.6), so practical training remains expensive.

- Evaluation dependence:
  - FID is known to be sensitive to implementation details (p.6). The paper mitigates this by using ADM’s evaluation code but still relies primarily on FID and Inception‑based metrics; no human evaluation or downstream tasks are reported.

- Transformer memory/latency:
  - Decreasing patch size increases sequence length quadratically (Figure 4), which can stress memory and slow training/sampling notwithstanding favorable `Gflops`.

- Scope of architectural exploration:
  - The study focuses on ViT‑like blocks and four conditioning variants. Other transformer designs (e.g., local attention, sparse attention, hybrid conv‑transformers) are not explored.

## 7. Implications and Future Directions
- Field impact:
  - Demonstrates that diffusion models need not be tied to U‑Nets; a ViT backbone works, scales predictably with compute, and can surpass prior state‑of‑the‑art. This unifies generative modeling architecture with modern vision/language stacks and opens the door to shared scaling laws and training recipes (Section 1; Figure 2).

- Research enabled or suggested:
  - Scaling: Figures 8–9 and 12 indicate “more tokens + bigger transformer” should continue to pay off. Expect further gains from longer training, larger models, or smaller patches (more tokens).
  - Conditioning: `adaLN‑Zero` is simple and effective; it may generalize to text condition, multi‑modal conditioning, or other iterative refinement models.
  - Efficiency: Explore attention variants (local/sparse), patch routing, or distillation to reduce the token‑length bottleneck introduced by small `p`.
  - Beyond class labels: Integrate DiT as a drop‑in backbone for text‑to‑image systems (e.g., Stable Diffusion) as proposed in the conclusion (p.9).

- Practical applications:
  - High‑fidelity class‑conditional generation for data augmentation, simulation, and content creation with better compute/quality trade‑offs than pixel‑space diffusion.
  - As a modular backbone, DiT may simplify integrating vision generation with transformer‑based perception or language models in multi‑modal systems.

Overall, this work replaces a widely assumed architectural choice (U‑Net) with a transformer design that is simple, compute‑predictable, and high‑performing. The detailed scaling study (Figures 6–10, 12) and the strong ImageNet results (Tables 2–3) make a convincing case that transformers are a first‑class backbone for diffusion image generation.
