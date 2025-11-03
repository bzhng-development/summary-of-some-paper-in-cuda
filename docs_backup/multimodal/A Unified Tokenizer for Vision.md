# A Unified Tokenizer for Vision

**ArXiv:** [2509.14476](https://arxiv.org/abs/2509.14476)
**Authors:** Jiasen Lu, Liangchen Song, Mingze Xu, Byeongjoo Ahn, Yanjun Wang, Chen Chen, Afshin Dehghan, Yinfei Yang
**Institutions:** 

## 🎯 Pitch

ATOKEN revolutionizes visual processing with a unified "visual tokenizer" that seamlessly integrates images, videos, and 3D assets into a shared 4D latent space, supporting both high-fidelity reconstruction and semantic understanding. By eliminating the need for modality-specific models and adversarial training, ATOKEN paves the way for more cohesive, scalable multimodal AI systems, transforming how unified visual intelligence is developed and applied.

---

## 1. Executive Summary (2–3 sentences)
ATOKEN introduces a single “visual tokenizer” that works across images, videos, and 3D assets while simultaneously supporting high‑fidelity reconstruction and semantic understanding. It does this by encoding all inputs into a shared, sparse 4D latent space using a pure transformer with 4D rotary positional embeddings and an adversarial‑free (non‑GAN) training objective, achieving competitive or state‑of‑the‑art results across modalities and enabling both continuous and discrete tokens (Sections 3–4; Tables 3–8).

## 2. Context and Motivation
- Problem addressed
  - Vision systems still split into two camps:
    - Reconstruction tokenizers (good for generation) that compress images/videos into latents but lack semantic alignment with text.
    - Understanding encoders (good for recognition/retrieval) that align with text but cannot reconstruct pixels.
  - They are also fragmented by modality: image‑only, video‑only, or 3D‑only models (Section 2; Table 1).
- Why it matters
  - Language modeling advanced dramatically once diverse text forms were mapped into a unified token space (BPE for code, documents, etc.). Vision lacks such a universal token space, which limits generalization, transfer, and unified system design (Introduction).
- Prior approaches and gaps
  - Reconstruction only: SD‑VAE, VQGAN, Hunyuan, Wan, Cosmos, etc., excel at compression and synthesis but do not produce text‑aligned features (Table 1).
  - Understanding only: CLIP/SigLIP2/VideoPrism/PE produce strong semantics but cannot reconstruct inputs (Table 1).
  - Early unifiers (e.g., VILA‑U, UniTok) remain image‑only; no method previously unified images, video, and 3D for both reconstruction and understanding, nor supported both continuous and discrete tokens in one model (Section 2; Table 1).
- Positioning
  - ATOKEN proposes the first unified tokenizer across images, videos, and 3D that:
    - Preserves pixel‑level detail for generation and supports semantic alignment for understanding,
    - Handles arbitrary resolutions and sequence lengths natively,
    - Avoids adversarial training instabilities via a Gram‑matrix‑based perceptual loss,
    - Produces both continuous latents and discrete tokens (Sections 3.1–3.4; Figures 1–5; Table 1).

## 3. Technical Approach
ATOKEN is built around a unified 4D representation and a sparse, pure‑transformer encoder–decoder that jointly serve reconstruction and understanding objectives.

- Core representation: sparse 4D latents (Section 3.1; Eq. 1)
  - Each input is patchified into features with explicit 4D coordinates p = [t, x, y, z] (time and 3 axes in space).
  - The model operates on a set of pairs z = {(z_i, p_i)} where z_i is a feature vector and p_i its 4D location (Eq. 1).
  - Modality mapping into 4D:
    - Image: a 2D slice at t = z = 0.
    - Video: a temporal stack at z = 0.
    - 3D: surface voxels at t = 0, aggregated from multi‑view images (Figure 3).
  - Why sparse: keeps only “active” positions per modality (e.g., no time dimension for images), making processing efficient and native to each sample’s resolution/length.

- Space–time patchification pipeline (Section 3.2)
  - Images/videos: non‑overlapping patches of size t × p × p (for images, temporal padding gives consistent shapes across modalities).
  - 3D assets: render multi‑view RGB images from sampled cameras; patchify them the same way; aggregate view features back into a 64³ voxel grid via back‑projection (Figure 3).
    - Unlike Trellis‑SLAT, ATOKEN directly tokenizes RGB patches, not DINO features, then aggregates to voxels (Section 3.2).

- Transformer architecture with 4D RoPE (Section 3.2; Figure 2)
  - Encoder
    - Starts from a pretrained SigLIP2 vision tower (semantic prior and native‑resolution robustness).
    - Generalizes patch embeddings from 2D to space–time blocks; initializes temporal weights at zero to preserve image behavior initially.
    - Adds 4D Rotary Position Embeddings (RoPE) in every attention layer to inject relative 4D awareness across (t, x, y, z).
    - Produces shared features z that serve both reconstruction and understanding.
  - Decoder
    - Architecturally similar transformer, trained from scratch for reconstruction.
    - Decoding heads by modality:
      - Image/video: map to pixels D_P: {(z_i, p_i)} → x ∈ R^{T×H×W×3} (Eq. 2).
      - 3D: decode to pixel‑space features then to Gaussian splatting parameters per active voxel D_GS (Eq. 3; Figure 3). To keep Gaussians tied to their voxels, positions are constrained via x_k_i = p_i + tanh(o_k_i).

- Outputs for both tasks from a single encoding (Section 3.1; Figure 2)
  - Reconstruction path
    - Project encoder features to a low‑dimensional latent z_r = W_r(z); apply KL regularization (as in VAE‑style latents).
    - Optional discrete tokens via FSQ quantization: z̃_r = FSQ(z_r). FSQ (finite scalar quantization) maps continuous values to a small set of levels per dimension, enabling LLM‑friendly discrete vocabularies without codebook learning instability.
    - The decoder reconstructs pixels or 3D Gaussians from these latents.
  - Understanding path
    - Aggregate structured latents with attention pooling into a global vector z̄; project to a semantic embedding z_s = W_s(z̄) for alignment with a frozen text encoder (Figure 2).

- Training objectives: adversarial‑free joint loss (Section 3.3; Eq. 4)
  - Overall: L = λ_rec L_rec + λ_sem L_sem + λ_KL L_KL (Eq. 4).
  - Reconstruction loss (image): a weighted sum (Eq. 6)
    - L1 pixel error (sharpness/photometric fidelity),
    - LPIPS perceptual distance,
    - Gram‑matrix loss L_Gram (captures second‑order statistics like texture/style by matching covariances of perceptual features; Eq. 5),
    - CLIP perceptual loss (semantic consistency).
    - For video and 3D, they use L1 for efficiency, relying on cross‑modal transfer from image losses for perceptual detail (Section 3.3).
  - Why no GAN: Figure 4(a) shows discriminator domination and instability for transformer tokenizers. Figure 4(b) decomposes rFID error, showing ≈86.6% comes from covariance differences (texture/style). Gram loss directly optimizes feature covariance (Eq. 5) and yields stable, superior rFID (Figure 4(c)).
  - Semantic alignment:
    - Images: distill SigLIP2 via KL divergence over temperature‑scaled vision–text similarity distributions (Eq. 7; Section 3.3).
    - Video/3D: use Sigmoid loss from SigLIP for robust contrastive learning with smaller batch sizes.

- Progressive curriculum (Section 3.4; Figure 5; Table 2)
  - Stage 1 (image foundation): add reconstruction to SigLIP2; patch size 4×16×16; 64–512px images.
  - Stage 2 (video dynamics): add temporal modeling; latent dims 32→48; image res up to 1024, video up to 512; temporal tiling and KV‑caching for efficient encoding/decoding (Figure 6).
    - KV‑caching: reuse attention key/value states across temporal tiles to avoid recomputing overlapping context, keeping temporal coherence while speeding up inference (Figure 6).
  - Stage 3 (3D geometry): integrate 64³ voxels; decode to Gaussians; image up to 2048px, video up to 1024px.
  - Stage 4 (discrete tokens): FSQ on 48‑D latents using 8 codebooks of 6 dims, 4 levels each (4096‑entry vocabulary); finetune encoder/decoder to work with discrete tokens.

- Implementation choices (Section 3.5)
  - Encoder/decoder: 27 transformer blocks, hidden 1152, 16 heads; encoder initialized from SigLIP‑So400M‑patch16‑naflex.
  - Optimization: AdamW, cosine LR; EMA 0.9999; reduced LR for encoder (0.1× base).
  - Compute: 256×H100 across stages (200k + 200k + 50k + 100k steps); ~138k GPU hours total.
  - Data (proportional sampling; Table 2 for task ratios): DFN, Open Images, internal images; WebVid, TextVR for video understanding and Panda70M for video reconstruction; Objaverse+Cap3D for 3D.

## 4. Key Insights and Innovations
- A single sparse 4D latent space across modalities (Section 3.1; Figure 1, Figure 2)
  - What’s new: Images, videos, and 3D are all represented as sparse points in a 4D grid (time + 3D space). Prior work unified only images (e.g., UniTok/VILA‑U). This design cleanly factors modality structure and lets one encoder operate natively at any resolution/length.
  - Why it matters: Enables a single model to serve both generation and understanding across modalities without padding/packing overhead and without separate modality‑specific towers.

- Pure transformer with 4D RoPE for vision tokenization (Section 3.2)
  - What’s new: Transformer‑only tokenizer/decoder (no conv backbones) with 4D rotary positional embeddings in all attention layers for relative position awareness over time and space.
  - Why it matters: Prior transformer tokenizers have struggled with adversarial training instability and/or limited scalability. The 4D RoPE makes one architecture scale across modalities and resolutions.

- Adversarial‑free reconstruction via Gram loss (Section 3.3; Figure 4)
  - What’s new: Diagnose why FID errors persist (covariance dominates), then directly optimize covariance with Gram matrices, plus LPIPS and CLIP. No GAN.
  - Why it matters: Stabilizes training for transformer tokenizers and 3D settings; still achieves top‑tier reconstruction quality (Table 3, Table 4; Figure 4(c)).

- Progressive curriculum that improves, not degrades, single‑modality performance (Section 3.4; Table 4 “Stage 1→3”, Figure 7)
  - What’s new: As the model learns video and 3D, image reconstruction improves (ImageNet rFID 0.258 → 0.246 → 0.209 in Table 4). Scaling analysis shows larger capacity avoids interference and benefits from cross‑modal learning (Figure 7).
  - Why it matters: Challenges a common belief that adding modalities dilutes single‑modality quality; suggests a path to unified multimodal foundations.

- Dual token output: continuous and discrete in a single framework (Section 3.1, 3.4 Stage 4; Table 3)
  - What’s new: Same encoder supports continuous latents for high‑fidelity generation and FSQ discrete tokens for compatibility with autoregressive/LLM‑style models.
  - Why it matters: Gives practitioners flexibility to plug into diffusion or AR pipelines without changing the tokenizer.

## 5. Experimental Analysis
- Evaluation design (Section 4; Tables 3–8, 11–13; Figures 9–11)
  - Datasets and tasks
    - Images: ImageNet (reconstruction metrics at 256²; zero‑shot classification for understanding), COCO (reconstruction).
    - Video: DAVIS 1080p, TokenBench 720p (reconstruction); MSRVTT/MSVD (retrieval for understanding).
    - 3D: Toys4k (multi‑view reconstruction and zero‑shot classification).
    - Downstream: multimodal LLMs (image/video QA benchmarks, Tables 9–10), image generation (Lightning‑DiT; Table 11), discrete image generation (TokenBridge; Table 12), text‑to‑video (MMDiT‑style; Table 13), image‑to‑3D (Figure 14).
  - Metrics (examples)
    - Reconstruction: PSNR/SSIM (pixel fidelity), LPIPS (perceptual), rFID (reference FID), rFVD (reference FVD for video).
    - Understanding: zero‑shot ImageNet accuracy; text–video retrieval R@k.
    - Generation: gFID, sFID, Inception Score, precision/recall; CLIPScore, PickScore, GenEval (T2I); VBench (T2V).

- Main unified tokenizer comparison (Table 3)
  - ATOKEN‑So/C (continuous 48‑D, comp. ratio (4,16,16)):
    - Images: 29.72 PSNR, 0.21 rFID with 82.2% zero‑shot ImageNet accuracy.
    - Video: 36.07 PSNR, 3.01 rFVD, 40.2% MSRVTT R@1.
    - 3D: 28.28 PSNR, 0.062 LPIPS, 90.9% classification accuracy.
  - ATOKEN‑So/D (discrete via FSQ):
    - Maintains competitive reconstruction (e.g., ImageNet rFID 0.38, TokenBench rFVD 22.16) and 82.2% zero‑shot ImageNet accuracy.
  - Takeaway: It is the only model in Table 3 that covers both reconstruction and understanding across images, video, and 3D, with both continuous and discrete tokens.

- Image reconstruction (Table 4; qualitative Figure 9)
  > Stage 1→2→3 rFID improves from 0.258 → 0.246 → 0.209 at 16×16 compression with 48‑D latents.  
  - Despite a transformer architecture and no GAN, ATOKEN’s rFID is close to strong image‑only tokenizers and shows better generalization from ImageNet (0.209 rFID) to COCO (2.026 rFID) than discrete baselines like UniTok (0.362 → 3.918).
  - Qualitatively, ATOKEN preserves high‑frequency details and text legibility better than several baselines at higher compression (Figure 9).

- Image understanding (Table 5)
  > Zero‑shot ImageNet accuracy 82.2% (256‑px) and stays stable across stages (82.7 → 82.3 → 82.2).  
  - Within ~1.2% of SigLIP2 (83.4%), while supporting reconstruction and extra modalities.
  - Retrieval on COCO/Flickr remains strong and stable across resolutions.

- Video reconstruction (Table 6; qualitative Figure 10)
  > DAVIS 1080p: 33.11 PSNR, rFVD 10.76; TokenBench 720p: 36.07 PSNR, rFVD 3.01.  
  - Comparable to specialized video tokenizers (Wan2.2: 36.39 PSNR, 3.19 rFVD on TokenBench). Stage 3 (with 3D) improves PSNR over Stage 2, suggesting beneficial cross‑modal transfer.

- Video understanding (Table 7)
  > MSRVTT R@1 ≈ 40.2%, MSVD R@1 ≈ 53.5% with simple frame averaging.  
  - Not SOTA (smaller video‑text data than dedicated encoders), but solid for a unified tokenizer primarily optimizing reconstruction.

- 3D reconstruction and understanding (Table 8; qualitative Figure 11)
  > 28.28 PSNR and 0.062 LPIPS; 90.9% zero‑shot classification accuracy on Toys4k.  
  - Matches or surpasses Trellis‑SLAT on reconstruction despite being unified and transformer‑based; qualitative results show better color consistency (Figure 11).

- Scaling and ablations (Section 4.5; Figure 7–8)
  - Model capacity matters: the smaller Base model degrades when adding modalities (ImageNet rFID 0.323 → 0.483), while So400m improves (0.258 → 0.209) and video PSNR rises (Figure 7).
  - Representation visualization: dense features cluster semantically, but low‑dimensional projected latents look more mixed in t‑SNE (Figure 8) yet still deliver strong performance—suggesting that strict cluster separation is not required with sufficient capacity.

- Downstream applications
  - Multimodal LLMs (Tables 9–10): Plugging ATOKEN into SlowFast‑LLaVA‑1.5 (frozen vision encoder) matches or improves over Oryx‑ViT on many image and video QA benchmarks, especially at smaller scales and for general video QA (e.g., VideoMME 64.5% at 7B).
  - Image generation with continuous tokens (Table 11): Lightning‑DiT on ATOKEN latents achieves gFID 1.56 (So400m, Stage 3), competitive with image‑specialized tokenizers.
  - Discrete image generation (Table 12): With TokenBridge‑L, ATOKEN discrete tokens reach gFID 2.23, outperforming UniTok (2.51) despite a larger vocabulary (4096 vs 8).
  - Text‑to‑video (Table 13): Under matched small‑compute settings, ATOKEN matches Hunyuan/Wan quality/semantic scores and beats Cosmos; T2I metrics also competitive.
  - Image‑to‑3D (Figure 14): Works “out of the box,” though fidelity can trail Trellis‑SLAT; likely needs tuning for higher‑dimensional (48‑ch) latents (Section 5.5).

- Do the experiments support the claims?
  - Yes for unification and competitive performance: Table 3 documents coverage and strong results across modalities and tasks; Table 4–8 provide per‑modality depth; Figures 4–8 justify the technical choices (Gram loss, capacity scaling).
  - Understanding performance is slightly below top understanding‑only models but impressively close given added reconstruction and multimodality burdens (Table 5, 7, 9–10).

## 6. Limitations and Trade-offs
- Compute and data requirements (Section 3.5)
  - Training requires 256 H100s for ~22 days (~138k GPU hours) with diverse large‑scale datasets (images, videos, 3D). This may limit reproducibility and iteration speed.
- Dependence on a strong image encoder prior
  - The encoder initializes from SigLIP2‑So400M; the semantic strength (and some resolution robustness) piggybacks on this prior. Results with weaker initial encoders degrade (Figure 7 vs Base).
- Low‑dimensional latents vs semantic separability (Figure 8)
  - Projected 48‑D latents look less cluster‑separable in t‑SNE. Although performance holds up, it suggests a trade‑off between extreme compression and explicit semantic structure.
- Long‑form video understanding
  - While general video QA is strong, long‑context benchmarks sometimes favor Oryx‑ViT, likely due to its design and training for very long videos (Table 10). ATOKEN’s training data has fewer long videos (Section 5.1 discussion).
- Discrete video quality
  - The discrete variant improves over prior discrete baselines but still lags continuous latents in video rFVD/LPIPS (Table 6). Discrete codes remain harder for high‑fidelity temporal consistency.
- 3D generative modeling
  - Generative image‑to‑3D with 48‑channel latents shows some color/style drift (Figure 14), implying diffusion hyperparameters and conditioning strategies need adaptation for higher‑D latents (Section 5.5).
- No adversarial loss
  - While Gram loss yields stable, high quality (Figure 4), GANs can sometimes add fine textures. The paper shows Gram covers most covariance‑driven FID error, but there may be edge cases where adversarial priors help.

## 7. Implications and Future Directions
- How this changes the landscape
  - Provides a concrete, working path to a universal visual token space—akin to BPE in language—spanning image, video, and 3D, and serving both generation and understanding. That unification removes a major engineering barrier to building single‑backbone multimodal AI systems (Figures 1–2; Table 3).
- Research enabled
  - Unified multimodal pretraining: use one tokenizer for joint scaling across web images, videos, and 3D assets.
  - Cross‑modal transfer at scale: Stage‑wise findings (Table 4; Figure 7) suggest that adding modalities can improve single‑modality quality when capacity suffices; invites studies on curricula, sampling, and architecture scaling laws for vision tokenization.
  - Token‑space modeling advances:
    - Continuous: better latent‑space diffusion/flow models that exploit ATOKEN’s 4D structure and KV‑cache‑friendly temporal tiling.
    - Discrete: codebook design and ordering strategies (e.g., FFT ordering as in TokenBridge) tailored to 4D tokens; hybrid discrete‑continuous schemes for video/3D.
  - 3D learning unification: ATOKEN’s 3D branch (voxelized SLAT with Gaussian decoding) can interoperate with 2D/temporal signals, encouraging unified 2D/3D generative and recognition systems.
- Practical applications
  - A single vision front‑end for:
    - Multimodal LLMs (Tables 9–10): plug‑and‑play with frozen vision encoder to improve or match existing baselines.
    - Image/video generation (Tables 11–13): train one generator head per modality on the same token space.
    - Image‑to‑3D content pipelines (Figure 14): shared tokens for asset creation, editing, and understanding in AR/VR and robotics.

In sum, ATOKEN demonstrates that a sparse, 4D, transformer‑based, adversarial‑free tokenizer can unify reconstruction and understanding across images, videos, and 3D while supporting both continuous and discrete representations. The empirical results across Tables 3–13, along with stability and scaling insights in Figures 4 and 7, make a compelling case for building next‑generation multimodal systems on top of a single visual tokenization substrate.
