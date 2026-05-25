# DIFFUSION TRANSFORMERS WITH REPRESENTATION AUTOENCODERS

**ArXiv:** [2510.11690](https://arxiv.org/abs/2510.11690)

## 🎯 Pitch

This paper introduces Representation Autoencoders (RAE), which replace the traditional VAE with frozen, pretrained semantic encoders (like DINOv2 or MAE) plus a lightweight decoder, creating a richer and higher-dimensional latent space for Diffusion Transformers. By addressing fundamental challenges—matching model width to token dimension, optimizing the noise schedule, and decoding from noisy latents—the authors unlock much faster training and dramatically better image generation quality, achieving state-of-the-art FID scores on ImageNet. This work bridges the worlds of representation learning and generative modeling, setting a new standard for efficient, high-performance diffusion models.

---

## 1. Executive Summary
This paper replaces the traditional VAE latent space used by Diffusion Transformers with a new “Representation Autoencoder” (RAE): a frozen, pretrained representation encoder (e.g., DINOv2, SigLIP2, MAE) plus a lightweight ViT decoder trained for reconstruction. It then “tames” diffusion training in these higher‑dimensional, semantic latents by: matching model width to token dimensionality, shifting the noise schedule by total latent dimensionality, and making the decoder robust to noisy latents; further, a shallow‑but‑wide diffusion head (DiTDH) boosts efficiency. The result is faster convergence and new state‑of‑the‑art FID on ImageNet (e.g., 1.51 at 256×256 without guidance; 1.13 with guidance at 256×256 and 512×512; Table 8, Table 7).

## 2. Context and Motivation
- Problem/gap:
  - Diffusion Transformers (`DiT`) typically operate in a VAE latent space (e.g., SD‑VAE) that is low‑dimensional, trained only for pixel reconstruction, and built on older conv backbones. This often yields latents that capture local appearance but weak global semantics, limiting generative quality and scaling (Introduction; Figure 2).
  - Meanwhile, modern representation encoders (DINOv2, MAE, SigLIP2) learn rich, semantically structured features, but diffusion models rarely use these features directly as their latent space.
- Importance:
  - Practical: Better latents can accelerate training and improve sample quality; compute and data budgets are major bottlenecks in scaling generative models.
  - Conceptual: It links representation learning (semantics) and generative modeling through a shared latent space rather than aligning them post hoc via auxiliary losses.
- Prior approaches and their limits:
  - VAE-based latent diffusion (LDM/DiT) compresses aggressively (e.g., 256×256 → 32×32×4), which harms capacity and semantics (Introduction; Section 3).
  - Alignment methods (REPA/REG/DDT; Section 2) try to bring DiT internal features closer to external encoders but still train in VAE latents and add extra losses/stages.
  - Common belief: semantic encoders are “too high‑level” for faithful reconstruction and high‑dimensional latents are “hard to diffuse” (Section 1).
- Positioning:
  - The paper replaces the VAE encoder entirely with a frozen pretrained representation encoder and trains only a decoder (RAE), then adapts the diffusion recipe to this higher‑dimensional semantic latent space. It argues this is simpler, more scalable, and more effective than aligning DiT to an external encoder while still using a VAE (Sections 3–5).

## 3. Technical Approach
The method has two main components: (A) building the Representation Autoencoder (RAE) and (B) adapting Diffusion Transformers to high‑dimensional semantic latents and scaling them efficiently.

A. Representation Autoencoder (RAE) (Section 3; Appendix C)
- What it is:
  - Encoder `E`: a frozen, pretrained representation model such as `DINOv2-B/14` (`pe=14`, hidden size `d=768`), `SigLIP2-B/16` (`pe=16`, `d=768`), or `MAE-B/16` (same `d=768`). It produces `N = HW / pe^2` tokens (256 tokens at 256×256) of dimension `d`.
  - Decoder `D`: a ViT that maps tokens back to pixels. It can use the same patch size as the encoder (`pd=pe`) for 1× reconstruction, or a larger patch size (e.g., `pd=2pe`) to upsample without adding tokens (Section 6.1).
- Training the decoder:
  - Loss: a standard reconstruction cocktail—`L1 + LPIPS + GAN` with adaptive weighting (Eq. in Section 3; Appendix C.2). Adversarial training begins after a short warm‑up (Appendix C.2, Table 12).
  - Practicalities: LayerNorm alignment, token handling, discriminators, and data augmentations are detailed in Appendix C. Figure 8 shows reconstructions.
- Why this matters:
  - It preserves the encoder’s semantics (linear probe accuracy for `DINOv2-B` is 84.5% vs ~8% for SD‑VAE; Table 1d), while producing reconstructions that match or beat SD‑VAE in rFID (e.g., `MAE-B`: 0.16 rFID vs 0.62 for SD‑VAE; Table 1a). It’s also much more efficient in FLOPs (Figure 2; Table 1b).

B. Taming Diffusion in High‑Dimensional RAE Latents (Section 4)
1) Match model width to token dimensionality (Section 4.1)
   - Observation: Standard DiT struggles in RAE latents (Table 2). Overfitting experiments on a single image show that when DiT width `d_model` is less than the token dimension `n` (e.g., 768 for `DINOv2-B`), the model cannot even overfit; when `d_model ≥ n`, it converges (Figure 3; Table 3).
   - Reason (Theorem 1; Section B.1): Because training injects Gaussian noise along all dimensions (`xt = (1−t)x + tε`), the support becomes full‑rank. Any model whose hidden width `d < n` suffers a provable lower bound on the training loss equal to the sum of the smallest `n−d` eigenvalues of `Cov(ε−x)`. Intuition: diffusion “diffuses” a low‑dimensional data manifold into the full ambient space during training, so the network needs enough width to represent functions over all dimensions.
   - Takeaway: Use a DiT whose width meets or exceeds the RAE token dimension (e.g., DiT‑XL for `DINOv2-B`).

2) Dimension‑dependent noise schedule shift (Section 4.2)
   - Problem: Standard “resolution‑based” schedule shifts (developed for images or low‑channel VAEs) under‑corrupt high‑dimensional tokens. The effective corruption should depend on the total latent dimensionality (`#tokens × channels`) rather than spatial resolution alone.
   - Solution: Reparameterize timesteps using the shift from Esser et al. (2024), with scaling `α = sqrt(m/n)`, where `n=4096` is the base dimension and `m` is the RAE’s effective dimension; use `t_m = α t_n / (1 + (α − 1) t_n)` (Section 4.2). This halves FID from 23.08 to 4.81 in the RAE setting (Table 4).

3) Noise‑augmented decoder training (Section 4.3)
   - Problem: VAEs train decoders on continuous latents (N(μ,σ²I)); RAE decoders see only “clean” discrete latents `{z_i = E(x_i)}`. Diffusion outputs can be slightly noisy and off‑manifold, causing the decoder to generalize poorly.
   - Solution: Train the decoder on a smoothed latent distribution `p_n(z) = ∫ p(z − n) N(0, σ² I)(n) dn`, with `σ` sampled from `|N(0, τ²)|` each step (Section 4.3).
   - Effect: Improves generative FID (e.g., 4.81 → 4.28) with a small hit to reconstruction rFID (0.49 → 0.57), as shown in Table 5; ablations on `τ` and encoders in Table 15c/15a/15b.

4) Scalable denoising via a shallow‑but‑wide diffusion head (DiTDH) (Section 5)
   - Architecture: Keep a standard DiT backbone `M`, then add a shallow, very wide transformer head `H` that conditions on both the noisy input `x_t` and the DiT features `z_t = M(x_t|t,y)` to produce the velocity `v_t = H(x_t|z_t,t)` (Figure 5).
   - Why: Increasing width everywhere is quadratic in FLOPs; a wide, shallow head adds width where needed without quadratic blowup.
   - Design: Best head is 2 layers with width ≈2048 (“G” width), rather than deeper or narrower heads (Table 16); larger RAE encoders benefit from even wider heads (Table 17).
   - Outcome: Dramatic FLOP efficiency and better FID than plain DiT across scales (Figure 6a–c; Table 6).

C. Diffusion objective, model, and training (Sections 4, D, J, I)
- Objective: Flow matching with linear interpolation `x_t = (1−t)x + tε` and target `v(x_t,t) = ε − x` (Section 4; Appendix J).
- Backbone: `LightningDiT` with patch size 1 for RAEs → sequence length 256 at 256×256 (Section 4). Compute is dominated by sequence length, so RAE vs VAE costs are comparable for the DiT itself (Section D).
- Sampling: 50‑step ODE (Euler) by default; guidance primarily via `AutoGuidance` (a small early‑epoch DiTDH‑S guides the large model), which yielded better quality and easier tuning than CFG with intervals (Appendix I).

## 4. Key Insights and Innovations
- RAE as a drop‑in replacement for VAE latents (Section 3; Figure 2; Table 1)
  - Novelty: Use frozen representation encoders directly as the latent space and train only a decoder; no compression and no auxiliary alignment losses.
  - Why it matters: Better reconstructions (e.g., rFID 0.16–0.53 vs 0.62 for SD‑VAE; Table 1a) and much stronger representations (linear probe 68–84.5% vs ~8%; Table 1d), with lower encoder/decoder FLOPs (Figure 2, Table 1b).
- The width‑vs‑dimension principle for diffusion on semantic latents (Section 4.1; Theorem 1, Figure 3)
  - Novelty: A clear theoretical lower bound and empirical evidence that DiT width must meet or exceed token dimensionality to “fit” diffusion training in high‑dimensional latents.
  - Significance: Explains why standard DiT fails “out of the box” on RAE latents (Table 2), and offers a principled design rule that enables stable training.
- Dimension‑aware noise schedule shift (Section 4.2; Table 4)
  - Novelty: Generalizes resolution-based schedule shifts to depend on total latent dimensionality, not just spatial resolution.
  - Impact: Large FID gains (23.08 → 4.81), enabling effective diffusion in high‑dimensional features.
- Noise‑augmented decoder training (Section 4.3; Tables 5, 15)
  - Novelty: Train decoders on a smoothed latent distribution to handle off‑manifold diffusion outputs.
  - Trade‑off: Improves generation (gFID) at small reconstruction cost (rFID), consistently across encoders and sizes.
- DiTDH: shallow‑wide head for efficient denoising (Section 5; Figure 6; Tables 16–17)
  - Novelty: Decoupled, very wide, shallow head that increases effective width without quadratic costs.
  - Effect: Faster convergence and better FID than DiT at lower FLOPs (Figure 6a–b) and across scales (Figure 6c); e.g., DiTDH‑XL reaches FID 2.16 at 80 epochs vs DiT‑XL’s 4.28 (Figure 6a).

## 5. Experimental Analysis
- Evaluation setup (Sections D, E, K)
  - Datasets: ImageNet‑1K at 256×256 and 512×512.
  - Metrics: FID‑50K, Inception Score (IS), precision/recall (Appendix K).
  - Sampling protocol: Emphasizes class‑balanced sampling (50 per class) for conditional FID; shows it yields ~0.1 lower FID than uniform random sampling and re‑evaluates several baselines under this protocol (Appendix E; Table 14).
  - Training: Most results at 80 epochs; best results up to 800 epochs; Euler ODE with 50 steps.
- Reconstruction and representation (Section 3; Table 1; Figure 2; Figure 8)
  - Reconstruction rFID (lower is better): `MAE‑B`: 0.16; `DINOv2‑B`: 0.49; `SigLIP2‑B`: 0.53; `SD‑VAE`: 0.62 (Table 1a).
  - Decoder scaling: ViT‑B rFID 0.58 → ViT‑XL 0.49, still ≤ 1/3 the FLOPs of SD‑VAE (Table 1b; Figure 2).
  - Representation quality via linear probe: `DINOv2‑B`: 84.5%; `SigLIP2‑B`: 79.1%; `MAE‑B`: 68.0%; `SD‑VAE`: 8.0% (Table 1d).
- Diffusion on RAE latents: base DiT vs tamed DiT (Section 4; Tables 2, 3, 4, 5; Figure 3; Figure 4)
  - “Out of the box” failure: DiT‑S gFID 215.8 on RAE vs 51.7 on SD‑VAE; DiT‑XL 23.08 vs 7.13 (Table 2).
  - Width–dimension overfitting tests: Convergence only when `d_model ≥ token_dim` (Figure 3; Table 3). Theorem‑based lower bound matches observed loss curve (left panel of Figure 3).
  - Schedule shift: gFID 23.08 → 4.81 (Table 4).
  - Noise‑augmented decoder: gFID 4.81 → 4.28 (Table 5).
  - Convergence speed: With the above pieces, DiT‑XL on RAE reaches gFID 4.28 at 80 epochs and 2.39 at 720 epochs—47× faster than SiT‑XL and 16× faster than REPA‑XL to reach comparable FIDs (Figure 4).
- DiTDH vs DiT and other baselines (Section 5; Figure 6; Tables 6–8)
  - FLOP efficiency: DiTDH‑B outperforms DiT‑XL at ~40% of the training FLOPs (Figure 6a).
  - Scaling: DiTDH improves steadily from S→XL; at 80 epochs, DiTDH‑XL hits FID 2.16 (Figure 6c).
  - Across encoders: DiTDH outperforms DiT for DINOv2‑S/B/L (Table 6).
  - State‑of‑the‑art ImageNet FIDs:
    - 256×256, no guidance: 
      > “DiTDH‑XL (DINOv2‑B, 800 epochs): gFID 1.51” (Table 8).
    - 256×256, with `AutoGuidance`:
      > “gFID 1.13” (Table 8; scale tuned to 1.42 with a 14‑epoch DiTDH‑S; Appendix I).
    - 512×512, with guidance:
      > “gFID 1.13” after 400 epochs (Table 7), surpassing EDM‑2 (1.25).
  - Qualitative samples show diverse, detailed images (Figure 7; Figures 10–15).
- Ablations and robustness (Sections G, 6.1–6.3)
  - Encoder choice: Although `MAE‑B` has best rFID, `DINOv2‑B` yields much better gFID; rFID is not sufficient to judge “generatability” (Table 15a).
  - Noise‑augmented decoder: Improves gFID across encoders and sizes; stronger gains for weaker encoders (Table 15a/b/c).
  - DDT head shape: 2‑layer, very wide performs best (Table 16); larger encoders benefit from wider heads (Table 17).
  - High‑res via decoder upsampling: Reuse 256‑token models to output 512×512 by setting `pd=2pe`; gFID 1.61 vs 1.13 with direct 512‑token training, but at ~4× lower token cost (Section 6.1; Table 9).
  - Where DiTDH helps: It helps with RAE (high‑dim latents), not with low‑dim VAEs (Table 10) and not enough to make pixel diffusion competitive (Table 11). This supports the claim that semantic structure plus dimensionality is key.
- Unconditional generation (Appendix L; Table 18)
  - With AutoGuidance, DiTDH‑XL + RAE achieves gFID 4.96 / IS 123.1, far better than DiT‑XL + VAE (gFID 30.68) and competitive with RCG (4.89), despite being a simpler one‑stage method.

Assessment: The experiments are thorough, include theory‑guided ablations (width, schedule, decoder noise), practical design sweeps (head width/depth), scaling curves, re‑evaluated baselines with a consistent sampling protocol (Appendix E), and both 256 and 512 resolutions. The evidence convincingly supports the key claims.

## 6. Limitations and Trade-offs
- Dependence on strong pretrained encoders:
  - RAE quality inherits from the frozen encoder. DINOv2 works best for generation; MAE excels at rFID but underperforms gFID (Table 15a). The approach assumes access to high‑quality encoders and may be sensitive to their training domain.
- Width and memory requirements:
  - The theoretical width requirement (`d_model ≥ token_dim`) pushes the backbone to be wide. DiTDH mitigates quadratic cost, but very wide heads still add parameters and memory (Section 5; Tables 16–17).
- Decoder trade‑off:
  - Noise‑augmented training improves gFID but reduces rFID slightly (Table 5; Table 15c). If perfect reconstructions are the goal, there is a small penalty.
- Guidance reliance for very best numbers:
  - The best 256×256 and 512×512 results use `AutoGuidance` (Tables 8 and 7; Appendix I). Though lighter‑weight than CFG, it adds an extra small model and tuning step.
- Evaluation subtleties:
  - FID is sensitive to sampling protocol; the paper highlights a consistent ~0.1 gain with class‑balanced sampling (Appendix E). While they re‑evaluate some baselines, not all prior works can be controlled fully.
- Scope:
  - The paper focuses on class‑conditional ImageNet. Extensions to text‑to‑image, video, or out‑of‑distribution data are not evaluated here.

## 7. Implications and Future Directions
- Field impact:
  - Recasting the latent space from “compressed pixels” to “semantic representations” shifts the default for diffusion transformers. Results and scaling curves (Figure 6) suggest that RAEs can be a new baseline for training DiTs efficiently and to higher quality.
- Research avenues enabled:
  - Token‑space design: Study which pretrained representations (self‑supervised vs multimodal) are most “diffusible,” and how to pretrain encoders specifically for both semantics and generative “diffusability.”
  - Theory: Extend the width‑vs‑dimension analysis to multi‑scale latents, non‑Gaussian training paths, and other flow objectives; characterize when rFID correlates with gFID.
  - Architecture: Explore alternative wide‑head designs, mixture‑of‑experts heads, or adapter‑based width inflation to further improve FLOP efficiency.
  - Decoders: Develop principled noise curricula or learned latent smoothing to balance reconstruction and generation; investigate token‑space upsampling for even higher resolutions and video.
- Practical applications:
  - Faster training for class‑conditional or text‑conditional image generation, especially in regimes where compute is constrained.
  - Plug‑and‑play use: Since RAEs hold strong representations (Table 1d), the same latent space can support downstream tasks (e.g., editing, retrieval) and generation within a single ecosystem.
  - Efficient scaling to 512+ resolutions without growing tokens (Section 6.1), useful for deployment where memory or latency is tight.

> Headline results: “RAE‑based DiTDH‑XL reaches 1.51 FID on ImageNet 256×256 without guidance and 1.13 with `AutoGuidance`; at 512×512 with guidance, it also achieves 1.13” (Table 8; Table 7). Combined with Figure 6’s scaling and Figure 4’s convergence speedups, the approach is both higher‑quality and more compute‑efficient than prior VAE‑based DiTs.
