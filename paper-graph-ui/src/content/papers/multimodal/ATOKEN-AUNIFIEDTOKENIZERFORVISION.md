# ATOKEN: A UNIFIED TOKENIZER FOR VISION

**ArXiv:** [2509.14476](https://arxiv.org/abs/2509.14476)

## 🎯 Pitch

ATOKEN introduces the first unified visual tokenizer capable of encoding images, videos, and 3D assets into a single shared 4D latent space, enabling both high-fidelity reconstruction and robust semantic understanding. By leveraging a pure transformer architecture with 4D positional embeddings and a stable, adversarial-free training regime, ATOKEN breaks the modality barrier and matches or exceeds specialized models on both generation and comprehension tasks. This innovation paves the way for a new generation of multimodal AI systems, unlocking language model–style scaling and transfer learning in vision, with sweeping implications for both generative and understanding applications.

---

## 1. Executive Summary (2–3 sentences)
ATOKEN is a single “visual tokenizer” that converts images, videos, and 3D assets into a shared set of tokens that work for both high‑fidelity reconstruction and semantic understanding. It achieves this by encoding all modalities into a sparse 4D latent space (time x, y, z) using a pure transformer with 4D positional geometry, and by training with a stable, GAN‑free perceptual objective plus a progressive curriculum (Sections 3–4). Across benchmarks, it delivers competitive generation quality while preserving strong alignment with text for understanding (Table 3).

## 2. Context and Motivation
- Problem/gap addressed:
  - In language, a common tokenizer (e.g., BPE) enables one model to generalize across many tasks. Vision lacks such a unified tokenizer because:
    - Reconstruction tokenizers (e.g., VAEs/VQ‑VAEs) preserve fine detail but do not produce semantic features suitable for understanding tasks.
    - Understanding encoders (e.g., CLIP/SigLIP2) map images to semantic spaces but cannot reconstruct pixels.
    - Tokenizers are fragmented by modality: image vs. video vs. 3D typically require separate systems (Section 2; Table 1).
  - Existing transformer tokenizers that aim for reconstruction often rely on adversarial (GAN) training, which is unstable at scale—especially for transformers and when extending to 3D (Section 2; Figure 4a).

- Why this matters:
  - A shared token space across modalities and tasks would enable “language model–style” scaling and transfer in vision: one encoder feeds many downstream generators and multimodal LLMs, reduces duplicated training, and simplifies system design (Introduction; Figure 1).

- Prior approaches and their shortcomings (Table 1; Section 2):
  - Reconstruction-only (e.g., SD‑VAE, Hunyuan, Wan, Trellis‑SLAT): excellent fidelity but no text alignment or understanding.
  - Understanding-only (e.g., SigLIP2, VideoPrism, PEcore): excellent semantics but no reconstruction.
  - Limited “unified” attempts (VILA‑U, UniTok) handle both tasks but only for images.
  - Architectural trade-offs: conv VAEs handle resolution flexibly but scale poorly in parameter efficiency; transformers scale better but training is unstable with GANs (Section 2).

- Positioning:
  - ATOKEN is the first to jointly unify tasks (reconstruction + understanding) and modalities (images, videos, 3D) in one transformer framework with both continuous and discrete tokens, while avoiding adversarial training (Abstract; Table 1; Sections 3–4).

## 3. Technical Approach
ATOKEN comprises a unified 4D latent representation, a pure transformer encoder–decoder that preserves sparsity, and a stable multi‑objective training scheme, all brought together by a progressive training curriculum.

1) Unified sparse 4D latent representation (Section 3.1; Eq. (1); Figure 2)
- Idea in plain terms:
  - Represent any visual input as a set of “patch tokens,” each with a feature vector and a 4D position p = [t, x, y, z]. Only the relevant axes are “active”:
    - Image: occupies a single 2D slice (x, y) with t = z = 0.
    - Video: occupies (t, x, y) with z = 0.
    - 3D asset: occupies (x, y, z) with t = 0 (surface voxels).
- Mechanism:
  - Space‑time patchification: split inputs into non‑overlapping blocks of size t × p × p (Section 3.2). For images, add temporal padding so shapes match video patches.
  - 3D assets: render multiple views on a sphere, patchify the RGB views, then aggregate view features into a 64^3 voxel grid by back‑projection and nearest‑view aggregation (Figure 3; Section 3.2).
- Output:
  - A sparse set of pairs {(z_i, p_i)} where z_i ∈ R^C is the token feature and p_i is the 4D coordinate (Eq. (1)).

2) Dual pathways for reconstruction and understanding (Sections 3.1–3.2; Figure 2)
- Reconstruction path:
  - Project each latent to a lower‑dim continuous space `z_r = W_r(z)` with KL regularization (to make the distribution well-behaved); optionally quantize `z_r` into discrete tokens via FSQ (Finite Scalar Quantization) (Section 3.1).
    - FSQ here splits the 48‑dim latent into 8 groups of 6 dims, quantized to 4 levels per dimension → each group is a 4096‑way code (4^6), producing 8 discrete tokens (Stage 4; Figure 5; Section 3.4).
  - A transformer decoder maps the set of structured latents back to outputs:
    - Image/video: decode directly to pixels (Eq. (2)).
    - 3D: decode to per‑voxel sets of Gaussian “splats” (position offset o, color c, scale s, opacity α, rotation r) used for fast rendering; offsets are constrained near the source voxel: x_k = p + tanh(o_k) (Eq. (3); Figure 3).
- Understanding path:
  - Use attention pooling over the latent tokens to produce a global representation z̄, then project it to a semantic vector `z_s = W_s(z̄)` for text alignment (Section 3.1; Figure 2).
  - This reuses the same encoded features, so one encoder supports both decoding (reconstruction) and pooled semantic alignment (understanding).

3) Transformer architecture with 4D geometry and sparsity (Section 3.2; Figure 2)
- Encoder:
  - Initialize from SigLIP2’s vision tower (a strong image‑text encoder).
  - Extend to 4D by:
    - Space‑time patch embedding (t × p × p) with zero‑initialized temporal weights so image performance is preserved initially.
    - 4D RoPE (Rotary Position Embeddings) in every attention layer, giving relative position awareness across (t, x, y, z) (Section 3.2). RoPE rotates query/key vectors based on positions; 4D RoPE generalizes this to 4 axes so tokens “know” where they are in time and 3D space.
- Decoder:
  - Same transformer style, trained from scratch for reconstruction.
- Sparse processing:
  - The model processes sets of (feature, 4D‑position) pairs rather than dense grids. This naturally supports arbitrary resolutions and sequence lengths without padding (Sections 3.2–3.4).

4) Training objectives: stable, adversarial‑free (Section 3.3; Figure 4)
- Global objective: L = λ_rec L_rec + λ_sem L_sem + λ_KL L_KL (Eq. (4)).
- Reconstruction loss:
  - Image: L1 (pixel), LPIPS (perceptual similarity), Gram matrix loss (matches second‑order feature statistics like texture/style), and CLIP perceptual loss (semantic consistency) (Eq. (6)).
  - Video/3D: L1 only for efficiency; detailed textures transfer from the image objective (Section 3.3).
  - Why Gram loss? Decomposing rFID into mean and covariance shows 86.6% of error comes from covariance (texture/style) rather than means (Figure 4b). Gram loss directly targets covariance and avoids GAN instability (Figure 4a). It trains stably and improves rFID consistently (Figure 4c).
- Semantic loss:
  - Images: distill SigLIP2 image‑text alignment by matching similarity distributions via KL divergence (Eq. (7)).
  - Videos/3D: use Sigmoid alignment loss (as in SigLIP) which is more stable for smaller batch sizes (Section 3.3).

5) Progressive curriculum and efficiency (Section 3.4; Figure 5; Figure 6; Table 2)
- Four stages:
  - Stage 1 (Image foundation): add reconstruction to SigLIP2 with 1×16×16 patches; train on 64–512 px images.
  - Stage 2 (Video dynamics): switch to 4×16×16 patches, enable temporal modeling; handle images up to 1024 px, videos up to 512 px. Use temporal tiling with KV‑caching to avoid redundant compute across tiles (Figure 6).
  - Stage 3 (3D geometry): add 64^3 3D voxel latents and Gaussian decoding; raise image to 2048 px and video to 1024 px.
  - Stage 4 (Discrete tokens): apply FSQ quantization (8×6D groups, 4 levels per dim → 4096 codes per group), fine‑tune all modalities end‑to‑end.
- Sampling ratios and resolution limits per stage are specified in Table 2.

6) Implementation (Section 3.5)
- Encoder and decoder: 27 transformer blocks each, hidden size 1152, 16 heads. Encoder initialized from SigLIP2‑SO400M (patch16). AdamW training; cosine schedule; EMA 0.9999.
- Compute: 256×H100 GPUs, global batch sizes tuned per task; full curriculum totals ~138k GPU‑hours (≈22 days on 256 GPUs).
- Data (progressively): DFN + Open Images + internal (images); WebVid + TextVR + Panda70M (videos); Objaverse + Cap3D (3D) (Section 3.5).

## 4. Key Insights and Innovations
- Unified sparse 4D token space across modalities and tasks (Sections 3.1–3.2; Figures 1–3)
  - Fundamental innovation: one encoder produces structured tokens that work for both per‑pixel decoding and pooled semantics, across images, videos, and 3D, without architectural forks.
  - What’s new vs. prior work: earlier “unified” tokenizers covered only images; video tokenizers didn’t handle 3D; 3D tokenizers didn’t leverage large‑scale image/video pretraining (Table 1).

- Pure‑transformer tokenizer with 4D RoPE and native resolution (Section 3.2)
  - Significance: maintains transformer scaling advantages while handling arbitrary spatial/temporal sizes natively and efficiently via sparse sets and KV‑caching (Figure 6).

- Adversarial‑free reconstruction objective centered on Gram loss (Section 3.3; Figure 4)
  - Innovation: replaces GANs with a principled, stable combination of L1 + LPIPS + Gram + CLIP perceptual for images, and L1 for video/3D—driven by an empirical analysis that covariance dominates rFID error (Figure 4b).
  - Impact: state‑of‑the‑art reconstruction quality without GAN instability (Figure 4c; Tables 3–4, 6, 8).

- Progressive curriculum that improves, rather than hurts, single‑modality performance (Section 3.4; Table 4; Figure 7)
  - Observation: image rFID improves from 0.258 → 0.246 → 0.209 as video and 3D are added (Table 4 “ATOKEN‑So/C Stage 1→2→3”).
  - Capacity finding: scaling study shows a small “Base” model degrades when adding modalities, while the larger “So400m” improves (Figure 7). This clarifies a capacity requirement for multimodal tokenizers.

- Dual continuous and discrete tokens from the same encoder (Stages 3–4; Tables 3, 11–12)
  - Continuous latents deliver top reconstruction and diffusion‑based generation; discrete FSQ enables autoregressive generation and drop‑in compatibility with discrete LLM‑style generators.

## 5. Experimental Analysis
- Evaluation setup (Sections 4–5):
  - Datasets and metrics:
    - Images: ImageNet 256×256 for reconstruction (PSNR, rFID, LPIPS) and zero‑shot classification; COCO for reconstruction generalization (Table 4–5).
    - Videos: DAVIS 1080p, TokenBench 720p for reconstruction (PSNR/SSIM/LPIPS/rFVD); MSR‑VTT/MSVD for retrieval (Table 6–7).
    - 3D: Toys4k for reconstruction (PSNR/SSIM/LPIPS) and zero‑shot classification (Table 8; Table 3).
  - Baselines span reconstruction‑only VAEs/VQ‑VAEs, understanding‑only encoders, and prior “unified” image‑only tokenizers (Table 3).

- Cross‑modality headline (Table 3):
  > ATOKEN‑So/C achieves “0.21 rFID with 82.2% ImageNet accuracy” for images, “3.01 rFVD with 40.2% MSRVTT R@1” for video, and “28.28 PSNR with 90.9% classification accuracy” for 3D, while also supporting discrete tokens (ATOKEN‑So/D).
  - Compared to unified image‑only baselines, ATOKEN improves both reconstruction (e.g., rFID 0.21 vs. UniTok 0.36) and understanding (82.2% vs. 78.6% ImageNet accuracy) and extends coverage to video and 3D.

- Image reconstruction and understanding (Tables 4–5):
  - Reconstruction:
    - Under a unified evaluation protocol, ATOKEN‑So/C (16×16 compression, 48 channels) improves across stages to rFID 0.209 on ImageNet and 2.026 on COCO (Table 4).
    - It outperforms many strong tokenizers at similar or higher compression; the curriculum notably helps (Stage 1→3: 0.258→0.209 rFID).
  - Understanding:
    - Zero‑shot ImageNet accuracy remains close to SigLIP2 across resolutions and stages (e.g., 82.2% vs. 83.4% at 256px; Table 5).
    - Retrieval on COCO/Flickr remains competitive (Table 5).

- Video reconstruction and retrieval (Tables 6–7; Figure 6):
  - Reconstruction:
    - ATOKEN‑So/C Stage 3: 33.11 PSNR on DAVIS; 36.07 PSNR with rFVD 3.01 on TokenBench, comparable to Wan2.1/2.2 and Hunyuan (Table 6).
    - Discrete ATOKEN‑So/D achieves 29.75 PSNR on DAVIS and 22.16 rFVD on TokenBench, outperforming OmniTokenizer’s discrete variant (Table 6).
    - Temporal tiling + KV‑cache accelerates decoding while keeping coherence (Figure 6).
  - Retrieval:
    - MSRVTT R@1 = 40.2%; MSVD R@1 ≈ 53.5% (Table 7)—reasonable but below specialized video encoders trained on larger video‑text corpora.

- 3D reconstruction and understanding (Table 8; Figure 11):
  - Reconstruction:
    - 28.28 PSNR and 0.951 SSIM, surpassing Trellis‑SLAT’s 26.97 PSNR (Table 8).
    - Qualitatively stronger color consistency (Figure 11).
  - Understanding:
    - 90.9% zero‑shot classification on Toys4k (Table 3).

- Scaling and representation ablations (Section 4.5; Figure 7–8):
  - Capacity ablation: larger “So400m” improves when adding modalities; smaller “Base” degrades (Figure 7).
  - Embedding visualization: dense features cluster cleanly by class; after 48‑dim projection (with KL), t‑SNE shows more mixing, yet performance remains strong (Figure 8).

- Downstream applications (Section 5; Tables 9–13; Figures 12–14):
  - Multimodal LLMs: swapping ATOKEN‑So/C into SlowFast‑LLaVA‑1.5 (frozen encoder) yields gains vs. Oryx‑ViT on several image QA benchmarks (e.g., RW‑QA and SQA) and competitive video QA, especially at smaller LLM scales (Tables 9–10).
  - Image generation (continuous tokens): with Lightning‑DiT, ATOKEN‑So/C Stage 3 reaches gFID 1.56 (Table 11), approaching specialized reconstruction tokenizers while being multimodal.
  - Image generation (discrete tokens): with TokenBridge‑L, gFID 2.23—competitive with prior discrete tokenizers and better than UniTok (Table 12).
  - Text‑to‑video: with an MMDiT‑style generator under limited compute, ATOKEN matches Hunyuan/Wan on VBench totals and surpasses Cosmos (Table 13).
  - Image‑to‑3D: generates plausible 3D assets but sometimes misses color/style faithfulness, likely due to higher latent dimensionality; authors suggest tuning diffusion schedules and conditioning (Section 5.5; Figure 14).

- Do the experiments support the claims?
  - Yes for unification and breadth: metrics across three modalities confirm both reconstruction and understanding with one tokenizer (Table 3).
  - Stability: GAN‑free training is empirically substantiated (Figure 4).
  - Capacity requirement: clearly supported (Figure 7).
  - Video and 3D understanding are solid but not state‑of‑the‑art; they reflect data/batch constraints (Tables 7, 10; Section 3.3).

## 6. Limitations and Trade-offs
- Compute and scale:
  - Training is resource‑intensive (≈138k GPU‑hours on 256 H100s; Section 3.5). Benefits rely on large capacity (Figure 7).
- Data dependence and reproducibility:
  - Uses internal datasets for images; reconstruction quality and semantics can reflect data curation choices (Section 3.5).
- Semantic alignment for video:
  - Retrieval lags specialized video encoders trained on massive video‑text data (Table 7). The model uses sigmoid loss with relatively small batch sizes for video/3D (Section 3.3); understanding performance improves with more/longer video data (Table 10 discussion).
- 3D representation scope:
  - 3D assets are integrated via multi‑view rendering and 64^3 voxel aggregation, decoded as Gaussian splats (Sections 3.1–3.2). This targets object‑level geometry; complex scenes, large environments, or very high‑frequency details may be limited by voxel resolution and rendering setup.
- Reconstruction objectives for video/3D:
  - Only L1 loss is used to save compute; while image Gram/LPIPS/CLIP losses transfer some detail, they may cap perceptual sharpness in video/3D (Section 3.3).
- Discrete tokens trade‑off:
  - FSQ discretization preserves semantics (Table 5) but reduces reconstruction quality vs. continuous latents (Table 3; Table 4), and autoregressive generation still trails highly optimized discrete pipelines (Table 12).
- Latent projection semantics:
  - t‑SNE suggests semantic clusters are less separable after 48‑dim projection (Figure 8), though performance is good; how to preserve clear clusters in very low dimensions remains open (Section 4.5).

## 7. Implications and Future Directions
- Field impact:
  - Demonstrates that a single transformer tokenizer—with a 4D sparse latent and stable, GAN‑free training—can unify understanding and generation across images, videos, and 3D (Table 1; Figure 1). This is a step toward “visual BPE”-style infrastructure for multimodal systems.
- What it enables:
  - One encoder for multimodal LLMs, diffusion and autoregressive generators, and cross‑modal tasks (e.g., image‑to‑video, image‑to‑3D) without modality‑specific tokenizers (Section 5; Tables 9–13).
  - Drop‑in continuous/discrete tokens from the same model broaden the set of compatible generators (diffusion, AR; Tables 11–12).
- Practical applications:
  - Text‑to‑video generation, video editing, image compression and restoration, 3D asset reconstruction and retrieval, and general VLM/MLLM tasks (Figures 10–14; Tables 9–13).
- Future work suggested by results and analyses:
  - Scale long‑video data and training to improve video semantics and long‑range temporal understanding (Table 10).
  - Enrich 3D with higher resolution grids, multi‑object scenes, or neural field backends; investigate end‑to‑end differentiable rendering with the 4D tokens (Sections 3.2, 4.4).
  - Explore semantic preservation in low‑dimensional latents (Figure 8) via post‑projection alignment losses or hybrid objectives.
  - Optimize discrete generation with frequency‑aware token ordering or mixed continuous‑discrete schemes to close the gap with specialized AR systems (Table 12).
  - Move toward an “omnimodel” trained end‑to‑end on joint downstream tasks to fully realize unified tokenization benefits (Section 7).
