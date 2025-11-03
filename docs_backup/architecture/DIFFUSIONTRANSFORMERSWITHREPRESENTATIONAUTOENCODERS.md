# DIFFUSION TRANSFORMERS WITH REPRESENTATION AUTOENCODERS

**ArXiv:** [2510.11690](https://arxiv.org/abs/2510.11690)

## 🎯 Pitch

This paper introduces Representation Autoencoders (RAEs), which swap out traditional VAEs for frozen, pretrained representation encoders (like DINOv2 or MAE) plus lightweight decoders, enabling Diffusion Transformers to operate directly within high-dimensional, semantically-rich latent spaces. With principled architectural changes—including a novel, wide diffusion head—the approach achieves state-of-the-art image generation on ImageNet, dramatically improving both sample quality and training efficiency. By bridging semantic understanding and generative modeling, RAEs redefine the foundation for latent diffusion, setting a powerful new default for diffusion transformer training and unlocking faster, higher-fidelity generative models.

---

## 1. Executive Summary
This paper replaces the usual variational autoencoder (VAE) used in latent diffusion with a frozen, pretrained representation encoder (e.g., DINOv2, SigLIP2, MAE) plus a lightweight decoder, forming a Representation Autoencoder (`RAE`). It then “tames” Diffusion Transformers (`DiT`) to work directly in these higher‑dimensional, semantically rich latents through three principled changes and a new shallow‑but‑wide diffusion head (`DiTDH`), achieving state‑of‑the‑art ImageNet FID with much faster convergence (e.g., 1.51 FID at 256×256 without guidance; 1.13 with guidance at 256×256 and 512×512; Tables 8 and 7).

## 2. Context and Motivation
- Problem addressed
  - Diffusion Transformers almost universally diffuse in VAE latents. These VAEs compress images into low‑dimensional codes optimized only for reconstruction, which:
    - limit information capacity and semantic structure, hurting generation quality and generalization (Intro; Sec. 1).
    - rely on legacy, compute‑heavy convolutional backbones (Fig. 2).
  - Two common beliefs block progress:
    1) semantic encoders (e.g., DINO/CLIP/MAE) are poor for reconstruction because they discard low‑level detail; and
    2) diffusion is unstable/inefficient in high‑dimensional latent spaces (Sec. 1).

- Why it matters
  - Latent diffusion dominates modern image/video generation because it trades pixel complexity for compact latent spaces. If the latent space can be made more semantically meaningful without extra compute, both sample quality and training efficiency can improve (Intro; Fig. 1).

- Prior approaches and their shortfalls
  - Improve VAE latents indirectly by aligning with external encoders during DiT training (REPA, REG, DDT; Sec. 2). These add auxiliary losses/stages and tuning complexity, and still inherit compressed, weak latents.
  - Enhanced VAEs (e.g., MAE‑style tokenizers) still compress latents and are trained for reconstruction, not semantics (Sec. 2).

- This paper’s positioning
  - It directly adopts frozen representation encoders as the latent space (no compression), demonstrates they reconstruct competitively or better than SD‑VAE (Table 1a), and develops theory and practice for training DiTs stably and efficiently on these higher‑dimensional, semantic tokens (Sec. 3–5).

## 3. Technical Approach
The method has two parts: building the `RAE`, then adapting diffusion training and architecture to operate effectively in high‑dimensional representation features.

1) Representation Autoencoder (RAE) (Sec. 3; Appx. C)
- What it is
  - Encoder `E`: a frozen, pretrained vision representation model (DINOv2, SigLIP2, MAE). It splits an image into patches and outputs one token per patch (N tokens). Each token has `d` channels (e.g., 768 for DINOv2‑B).
  - Decoder `D`: a lightweight Vision Transformer that maps tokens back to pixels. It is trained; the encoder is not.
- How it works
  - Given an image x, compute tokens z = E(x). Train D to reconstruct x̂ = D(z) using a composite loss (Sec. 3; Appx. C.2):
    - L1 (per‑pixel difference),
    - LPIPS (perceptual similarity),
    - adversarial loss with a frozen DINO‑S/8 discriminator (stabilizes details).
- Training choices and efficiency
  - No encoder compression: the number of tokens equals (H×W)/pe²; for 256×256 with pe=16, N=256 tokens, matching standard DiT sequence length.
  - Decoder losses and hyperparameters are detailed in Appx. C; the discriminator setup follows StyleGAN‑T, with some stabilizing tweaks (Appx. C.2).
  - Compute: Fig. 2 shows the SD‑VAE encoder/decoder needs ~135/310 GFLOPs per 256×256 image vs ~22/106 GFLOPs for RAE’s encoder/decoder—substantial savings.

2) Training diffusion in RAE latent space (Sec. 4)
- Base training objective and backbone
  - Use flow matching with linear interpolation: xt = (1−t)x + tε, where ε is Gaussian noise; train to predict the “velocity” v(xt,t)=ε−x (Sec. 4; Appx. J).
  - Backbone: LightningDiT (a DiT variant), sequence length = 256 (patch size 1 for latents), so DiT compute is comparable to VAE baselines (Sec. 4; Appx. D).
- Why standard DiT fails out of the box on RAE
  - Table 2: training DiT directly on RAE latents yields very poor FID (e.g., DiT‑S fails catastrophically; DiT‑XL far worse than with SD‑VAE).
- Three fixes, with mechanisms:
  a) Match model width to token dimensionality (Sec. 4.1)
     - Observation via “single‑image overfitting”: the diffusion model cannot even overfit unless its width d ≥ latent dimension n (Fig. 3, left; Table 3). Increasing depth does not help if d < n (Fig. 3, right).
     - Intuition: because training adds Gaussian noise (spreading support over the full space), the target is effectively full‑rank; capacity must scale with data dimensionality (Sec. 4.1).
     - Formalization: Theorem 1 (Sec. 4.1; Appx. B.1) lower‑bounds the loss when d<n; in a toy case with a single image, the lower bound equals (n−d)/n, matching the empirical curve in Fig. 3.
     - Practical rule: pick a DiT width at least the latent channel dimension (e.g., ≥768 for DINOv2‑B).
  b) Dimension‑dependent noise schedule shift (Sec. 4.2)
     - Prior “resolution‑based” schedule shifts only adjust for more spatial tokens; here the per‑token channel dimension is also large (e.g., 768), so effective data dimension = (#tokens)×(channels).
     - Use the shift from Esser et al. (2024): for base dim n and target dim m, rescale t as tm = αt / (1+(α−1)t), with α=√(m/n). Using n=4096 as base (as in prior work) and m as RAE’s effective dimension drastically improves FID from 23.08 to 4.81 (Table 4).
  c) Noise‑augmented decoder training (Sec. 4.3)
     - Mismatch: The RAE decoder is trained on a discrete set of clean latents {E(x)}, but diffusion generates slightly noisy latents. To make D generalize, inject Gaussian noise into latents during decoder training, i.e., train D on z̃ = z + n, n~N(0,σ²) with randomness in σ (Sec. 4.3).
     - Effect: better generation (gFID 4.81→4.28) at a small cost in reconstruction fidelity (rFID 0.49→0.57), Table 5. Ablations over τ (noise scale) and encoders in Appx. G.2 show consistent gains.

3) Scaling width efficiently: a shallow‑but‑wide diffusion head (`DiTDH`) (Sec. 5)
- Motivation: simply making the whole DiT wider is expensive (quadratic cost in width).
- Design (Fig. 5): keep a standard DiT `M` (normal width) but add a lightweight, wide transformer head `H` that takes both the noisy input xt and features zt=M(xt|t,y) to predict the velocity vt.
  - This increases denoising width where needed, avoiding quadratic blow‑up.
- What works best: a 2‑layer, very wide head (e.g., width 2048) outperforms deeper or narrower heads at similar compute (Table 16). Wider heads benefit larger RAE encoders more (Table 17).
- Empirical impact: with RAE latents, DiTDH converges faster and to better FID than DiT at the same or lower FLOPs (Fig. 6a–c; Table 6).

4) Efficient high‑resolution synthesis via the decoder (Sec. 6.1)
- To go from 256→512 without 4× tokens, keep the same latent tokens and only upsample in the decoder by using a larger patch size `pd=2·pe`. This “decoder upsampling” attains competitive 512‑FID (1.61 vs 1.13 trained directly at 512) while being ~4× cheaper (Table 9).

## 4. Key Insights and Innovations
- Turn frozen semantic encoders into practical autoencoders for generation (RAE)
  - Novelty: Defies the belief that semantic encoders cannot reconstruct faithfully. With a modest ViT decoder and standard reconstruction+adversarial losses, RAEs match or beat SD‑VAE reconstruction while being faster (Table 1a,b; Fig. 2).
  - Evidence:
    - Reconstruction FID (rFID) on ImageNet val set: MAE‑B/16 achieves 0.16 vs SD‑VAE’s 0.62; DINOv2‑B 0.49 (Table 1a).
    - Efficiency: RAE encoder/decoder ~22/106 GFLOPs vs SD‑VAE ~135/310 (Fig. 2).
    - Representation quality: linear probing top‑1 accuracy 84.5% (DINOv2‑B), 79.1% (SigLIP2‑B), 68.0% (MAE‑B) vs 8.0% for SD‑VAE (Table 1d).
- A principled capacity rule for diffusion on high‑dimensional latents (Sec. 4.1)
  - Insight: because the training interpolation injects Gaussian noise, the target becomes full‑rank; a DiT with width d<n (latent channels) cannot fit, even on one image.
  - Theorem 1 (Appx. B.1) quantifies a loss lower bound when d<n; the empirical single‑image experiments match the bound exactly (Fig. 3; Table 3).
- Dimension‑aware noise schedule shift (Sec. 4.2)
  - Extends “resolution‑aware” shifts to “effective‑dimension‑aware” shifts to handle many channels per token. This single change cuts gFID from 23.08 to 4.81 (Table 4).
- Noise‑augmented decoder to close the training–sampling gap (Sec. 4.3)
  - Simple but important: training the decoder on a smoothed latent distribution makes it robust to the slightly noisy latents produced by diffusion, improving gFID consistently (Table 5; Appx. G.2).
- `DiTDH`: decoupling denoising width from backbone width (Sec. 5)
  - Fundamental capability: scale width where it matters (denoising head) without quadratic cost. Yields large FID reductions and better compute–performance scaling (Fig. 6a–c; Tables 6, 16, 17).

## 5. Experimental Analysis
- Evaluation setup
  - Dataset: ImageNet‑1K at 256×256 and 512×512 (Sec. D).
  - Metrics: FID‑50k, Inception Score, precision/recall (Appx. K).
  - Sampling: ODE Euler sampler, typically 50 steps; Class‑conditional generation.
  - Important protocol detail: class‑balanced sampling (50 samples per class) vs uniform random over labels affects FID by ~0.1; this paper re‑evaluates several baselines with balanced sampling for fairness (Sec. 5.1; Appx. E; Table 14).

- Main results (state of the art)
  - ImageNet 256×256
    - Without guidance: DiTDH‑XL + DINOv2‑B achieves 1.51 FID after 800 epochs (Table 8).
    - With AutoGuidance: 1.13 FID (Table 8). Also strong IS and recall.
  - ImageNet 512×512
    - With guidance: 1.13 FID after 400 epochs (Table 7), surpassing prior diffusion best (EDM‑2 at 1.25).
  - Convergence and compute
    - Fig. 6b: DiTDH‑XL surpasses REPA‑XL, MDTv2‑XL and SiT‑XL at far less compute; reaches best FID with over 40× less training FLOPs than some baselines.
    - Fig. 4: Even before adding the head, DiT trained on RAE latents converges much faster than VAE‑based SiT or REPA (up to 47× and 16× speedups to comparable FID).

- Ablations and diagnostics
  - Why standard DiT fails initially on RAE:
    - Table 2 shows large FID gaps; single‑image overfit experiments (Fig. 3; Table 3) and Theorem 1 explain the need for width ≥ latent dimension.
  - Noise schedule shift:
    - Table 4: 23.08 → 4.81 FID by dimension‑aware shift (crucial).
  - Noise‑augmented decoder:
    - Table 5: gFID improves (4.81→4.28) while rFID drops slightly (0.49→0.57)—a deliberate trade‑off; robustness across encoders and τ scales in Appx. G.2 (Table 15).
  - `DiTDH` design:
    - Fig. 6a: better FLOP–FID scaling than DiT at all sizes.
    - Table 6: DiTDH outperforms DiT consistently across RAE sizes (e.g., DINOv2‑L gFID 2.73 vs 6.09).
    - Tables 16–17: the head should be wide and shallow; benefit grows with larger encoders.
  - What does not help:
    - DiTDH on SD‑VAE latents performs worse than DiT (Table 10)—the head mainly helps in high‑dimensional RAE spaces.
    - High dimensionality alone is not enough: pixel diffusion at the same dimensionality (768 per token) performs far worse than RAE (Table 11).
  - High‑resolution without extra tokens:
    - Decoder upsampling nearly matches direct 512 training (gFID 1.61 vs 1.13) at ~4× less compute (Table 9).

- Guidance method
  - AutoGuidance (a learned‑model‑guides‑model scheme) is the default because it outperforms classifier‑free guidance with interval here and is easier to tune (Appx. I).

- Do the experiments support the claims?
  - Yes: Multiple lines of evidence—reconstruction quality (Table 1), training dynamics (Fig. 3, Theorem 1), schedule/decoder ablations (Tables 4–5), wide head scaling (Fig. 6)—converge to a coherent story, culminating in SOTA FID at both 256 and 512 resolutions (Tables 8 and 7), with qualitative samples (Fig. 7; Appx. M).

## 6. Limitations and Trade-offs
- Reliance on strong pretrained encoders
  - RAE inherits strengths/weaknesses of the chosen representation encoder. Performance varies by encoder (Table 15a); DINOv2‑B generally works best for generation, although MAE has the best reconstruction rFID.
- Width requirement
  - Practical constraint: the diffusion model’s width must match or exceed the latent channel dimension (Sec. 4.1; Fig. 3; Theorem 1). This forces non‑trivial width, though DiTDH mitigates compute.
- Decoder robustness vs reconstruction fidelity
  - Noise‑augmented training helps generation but slightly harms reconstruction (Table 5). For use cases demanding exact reconstruction, this trade‑off must be tuned.
- Evaluation scope
  - Experiments focus on class‑conditional ImageNet; no text‑to‑image or cross‑domain tests. Generalization to other domains (medical, satellite) is untested.
- Guidance dependence and evaluation subtlety
  - Best 256/512 numbers use AutoGuidance; gains depend on a small guide model (Appx. I). Also, FID is sensitive to label sampling strategy (random vs balanced, Appx. E).
- High‑resolution pathway
  - Decoder upsampling is efficient but trails direct 512 training (gFID 1.61 vs 1.13; Table 9). There may be a ceiling to what upsampling alone can recover.

## 7. Implications and Future Directions
- Shift in default practice for DiTs
  - The results argue for replacing VAEs with RAEs when training diffusion transformers: better semantics, faster convergence, and SOTA quality without extra token lengths (Fig. 1; Tables 7–8). This reframes the autoencoder not as a compressor, but as a semantic representation foundation.
- Broader research avenues
  - Unified representation–generation pipelines: leveraging foundation encoders (self‑supervised or multimodal) as the latent space for generative training across images, video, and 3D.
  - Theory of capacity vs. dimensionality in diffusion: Theorem 1 and Fig. 3 suggest general design rules; extending this analysis to other noise processes or schedules could further systematize architecture choices.
  - Decoder training curricula: explore richer noise models, consistency regularization, or partial denoising tasks to further close the train–sample gap without losing reconstruction fidelity.
  - Multi‑modal generation: RAEs based on language‑supervised encoders (e.g., SigLIP2) could ease conditioning on text or other modalities without bespoke alignment losses.
  - Efficient scaling: the `DiTDH` pattern (shallow‑but‑wide denoising heads) may generalize to other high‑dimensional latent diffusion settings, enabling width scaling at modest cost.

> Headline result (Table 8): “DiTDH‑XL (RAE with DINOv2‑B) reaches 1.51 FID at 256×256 without guidance and 1.13 with AutoGuidance;”  
> Headline result (Table 7): “At 512×512 with guidance, 1.13 FID—surpassing the previous best diffusion result.”

Overall, this work demonstrates that semantically rich, high‑dimensional features are not an obstacle to diffusion training; with the right capacity rule, dimension‑aware noise scheduling, a robust decoder, and a strategically wide denoising head, they are a distinct advantage.
