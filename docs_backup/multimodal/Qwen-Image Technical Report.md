# Qwen-Image Technical Report

**ArXiv:** [2508.02324](https://arxiv.org/abs/2508.02324)
**Authors:** Chenfei Wu, Jiahao Li, Jingren Zhou, Junyang Lin, Kaiyuan Gao, Kun Yan, Sheng‑ming Yin, Shuai Bai, Xiao Xu, Yilei Chen, Yuxiang Chen, Zecheng Tang, Zekai Zhang, Zhengyi Wang, An Yang, Bowen Yu, Chen Cheng, Dayiheng Liu, Deqing Li, Hang Zhang, Hao Meng, Hu Wei, Jingyuan Ni, Kai Chen, Kuan Cao, Liang Peng, Lin Qu, Minggang Wu, Peng Wang, Shuting Yu, Tingkun Wen, Wensen Feng, Xiaoxiao Xu, Yi Wang, Yichang Zhang, Yongqiang Zhu, Yujia Wu, Yuxuan Cai, Zenan Liu
**Institutions:** Alibaba / Tongyi Qianwen team

## 🎯 Pitch

Qwen-Image introduces a groundbreaking approach to image generation and editing by enhancing complex text rendering, especially in Chinese, and enabling precise, instruction-following edits while preserving semantic and visual integrity. This model is pivotal for industries requiring accurate visual content creation, such as design and marketing, and sets a new standard for multilingual, layout-aware image generation. It addresses critical gaps in existing systems, significantly improving professional workflows where text precision and editing consistency are paramount.

---

## 1. Executive Summary (2–3 sentences)
Qwen-Image is an image generation and editing foundation model that tackles two longstanding pain points: rendering complex text (especially Chinese and multi-line layouts) and performing precise, instruction-following image edits that preserve both meaning and visual details. It combines a carefully engineered data pipeline, a dual-encoding architecture (semantic + reconstructive), a new multimodal positional encoding (`MSRoPE`), and progressive training (including RL) to achieve state-of-the-art or near–state-of-the-art results across generation, text rendering, and editing benchmarks (see Figures 1, 16 and Tables 3–14).

## 2. Context and Motivation
- Problem addressed
  - Text-to-image (T2I) models still struggle with complex text rendering: multi-line phrases, localized insertions, mixture of languages, and low-frequency characters in logographic scripts like Chinese (§1; Figures 1–2).
  - Instruction-based image editing faces two alignment challenges (§1): 
    - Visual consistency: modify only targeted regions while preserving other details.
    - Semantic coherence: keep global meaning and identity consistent during structural changes.
- Why it matters
  - Real use cases include posters, slides, UI mockups, packaging, signage, comics, and documents—often with dense, precise text (Figure 2). Editing reliability underpins professional workflows (e.g., style transfer, pose changes, adding/removing objects; Figure 4).
- Prior approaches and shortcomings
  - Diffusion-based T2I systems (e.g., SDXL, FLUX, Seedream) focus on photorealism and compositional prompts but exhibit failure modes in non-Latin text, multi-line layouts, and localized editing drift (§1; qualitative comparisons in Figures 18–23, 24–28).
  - Joint image-video VAEs often trade off image reconstruction fidelity (small text suffers), and unified multimodal transformers can blur modality roles (§2.3).
- Positioning
  - Qwen-Image positions itself as a general-purpose image foundation model that emphasizes text rendering and consistent editing. It contributes:
    - A text-centric, multi-stage data pipeline with targeted synthesis (§§3.1–3.4; Figures 9–13).
    - A dual-conditioning architecture that separately captures semantics (via Qwen2.5-VL) and pixel-level details (via VAE) in a double-stream diffusion transformer (Figure 6).
    - A new multimodal positional encoding, `MSRoPE`, to better align text with images across resolutions (§2.4; Figure 8).
    - Progressive training from non-text to text, and from simple to paragraph-level prompts (§4.1.3), plus SFT and RL (DPO + GRPO; §4.2; Eqs. 3–8).

## 3. Technical Approach
Step-by-step, from data to model to training and tasks.

- Data and curation (Section 3; Figures 9–13)
  - Four domains: Nature (~55%), Design (~27%), People (~13%), Synthetic (~5%) (Figure 9).
  - Seven-stage filtering and balancing pipeline (Figure 10):
    1) Pre-train curation (remove low-res, duplicates, NSFW) at 256p.
    2) Image quality (rotation, clarity, luma/saturation extremes, entropy/texture; examples in Figure 11).
    3) Image–text alignment: split captions into Raw, Recaption (Qwen-VL Captioner), and Fused; filter with Chinese CLIP and SigLIP 2, remove invalid/overlong captions.
    4) Text rendering enhancement: split by language (EN/ZH/Other/Non-Text); inject synthetic text data (see synthesis below); filter overly dense/small text.
    5) High-resolution refinement at 640p with aesthetic and artifact filters.
    6) Category rebalance + portrait augmentation with detailed synthesized captions.
    7) Balanced multi-scale training at 640p and 1328p with hierarchical taxonomy and resampling for text long-tail.
  - Annotation unifies detailed captions and structured metadata in one pass using an image captioner outputting JSON (Figure 12).
  - Text-aware synthesis (Section 3.4; Figure 13):
    - Pure Rendering: paragraphs on clean backgrounds; strict discard if any character fails to render (high character fidelity).
    - Compositional Rendering: render text on media (paper, boards), then composite into real scenes; generate contextual captions with Qwen-VL Captioner.
    - Complex Templates: programmatic replacement in slide/UI templates to teach layout-sensitive following (position, font, color).

- Architecture (Section 2; Figure 6; Table 1)
  - Three components:
    - Condition encoder: `Qwen2.5-VL` (a multimodal LLM) extracts semantic features from text-only or text+image inputs (§2.2; Figure 7 and Figure 15 show system prompts for T2I and TI2I).
    - Image tokenizer: a video-capable `VAE` with single encoder and dual decoders (Wan-2.1-VAE backbone), but only the image decoder is fine-tuned to boost small-text fidelity (§2.3). Training uses reconstruction + perceptual losses; adversarial loss was dropped since it did not help as recon quality improved.
    - Backbone diffusion model: `MMDiT` (Multimodal Diffusion Transformer; §2.4), a double-stream transformer jointly modeling text and image latents with flow matching.
  - Dual conditioning signal into MMDiT (Section 4.3; Figure 14):
    - Semantic stream: Qwen2.5-VL features guide instruction following and high-level semantics.
    - Image stream: concatenates the noised image latent with VAE-encoded latents of the input (for editing), preserving pixel-level fidelity and structure.
  - New positional encoding: `MSRoPE` (Multimodal Scalable RoPE; §2.4; Figure 8).
    - Problem: prior column-wise 2D RoPE for mixed text/image can create isomorphic rows where text tokens are indistinguishable from some image latents.
    - Solution: treat text as 2D tokens placed along the image grid diagonal; start image encoding from center. This keeps text equivalent to 1D RoPE while enabling resolution scaling for images.
  - Model scale (Table 1): VLM with 7B parameters; VAE encoder/decoder effectively 19M/25M for images (even though video-capable); MMDiT 20B parameters.

- Training objective and schedule (Section 4.1)
  - Flow matching with Rectified Flow (intuitive view): the model learns the “velocity” that transports pure noise to the target image latent along a continuous time variable t in [0,1]. 
  - Notation (Eqs. 1–2):
    - Encode image x0 → latent z via VAE encoder `E`.
    - Sample Gaussian noise x1 ~ N(0, I).
    - Interpolate latent at time t: `x_t = t x0 + (1 − t) x1`.
    - True velocity `v_t = d x_t / dt = x0 − x1`.
    - Train MMDiT to predict velocity `v_θ(x_t, t, h)` conditioned on guidance `h` from Qwen2.5-VL, using MSE to `v_t`.
  - Progressive curriculum (Section 4.1.3):
    - Resolution: 256 → 640 → 1328 with multi-aspect ratios.
    - Text content: non-text first, then simple text, then long/complex paragraphs and layouts.
    - Data quality: start broad, then increasingly refined/filtered.
    - Distribution: re-balance domains and resolutions to avoid overfitting.
    - Synthetic augmentation: fill long-tail and layout-heavy cases not present in real data.

- Post-training with human preferences (Section 4.2; Eqs. 3–8)
  - SFT: curated, photorealistic, detailed samples to sharpen realism (§4.2.1).
  - RL – two stages:
    - `DPO` (Direct Preference Optimization; Eq. 3): pairwise comparisons per prompt (best vs worst or gold vs generated). Reformulated on the flow-matching objective by contrasting squared velocity errors of chosen vs rejected generations (policy vs reference).
    - `GRPO` (Group Relative Policy Optimization; Eqs. 4–8): on-policy exploration by sampling G images per prompt, computing per-group standardized advantages from a reward model, and optimizing a PPO-style clipped objective. For diversity, deterministic ODE sampling is perturbed into an SDE (Eqs. 6–7); a closed-form KL regularizer tying policy to a reference is used (Eq. 8).

- Multi-task image editing and vision tasks (Section 4.3; Figures 14–15)
  - Input image is encoded twice: semantically (Qwen2.5-VL) and pixel-wise (VAE), then both are injected into MMDiT (Figure 14, left).
  - `MSRoPE` is extended with a `frame` dimension to tell “before” vs “after” images apart (Figure 14, right).
  - The same engine is reused for novel view synthesis, depth/canny estimation, and other “editing-like” tasks (Figure 5).

- System scale and optimization (Sections 4.1.1–4.1.2)
  - Producer–Consumer training framework: Producers filter, caption, and pre-encode with VAE and VLM; Consumers (GPU clusters) train MMDiT via Megatron with hybrid parallelism. Batches are pulled asynchronously via an HTTP transport with RPC semantics (§4.1.1).
  - Activation checkpointing was tested but disabled: it reduced per-GPU memory from 71 GB to 63 GB but slowed iterations 3.75× (from 2s to 7.5s), so the final setup relies on distributed optimizers, bfloat16 all-gathers and float32 reduce-scatter for stability (§4.1.2).

## 4. Key Insights and Innovations
- Dual-encoding for editing consistency (fundamental)
  - What’s new: pass both high-level semantics (Qwen2.5-VL) and low-level pixel latents (VAE) to MMDiT, then distinguish multiple images with an added `frame` axis in `MSRoPE` (Section 4.3; Figure 14).
  - Why it matters: balances semantic coherence (e.g., identity, global intent) with visual fidelity of untouched regions—critical for precise edits (Figures 24–26, 28); reflected in top editing scores on GEdit/ImgEdit (Tables 11–12).
- `MSRoPE` positional encoding (fundamental)
  - What’s new: a multimodal, resolution-scalable rotary encoding that places text along the diagonal in the image grid (Section 2.4; Figure 8).
  - Why it matters: avoids token indistinguishability seen in some previous 2D text-image encodings, while preserving 1D-RoPE equivalence for text. It improves text–image alignment and scales to higher image resolutions (§2.4).
- Text-centric data engineering + curriculum for rendering (fundamental)
  - What’s new: seven-stage pipeline with language-balanced splits, long-tail synthesis at three difficulty levels (pure/compositional/complex), and a progressively text-heavy curriculum (Sections 3.1–3.4; 4.1.3; Figures 10, 13).
  - Why it matters: substantially boosts native text rendering in both English and Chinese, especially for long, multi-line, and layout-sensitive prompts (Figures 18–21; Tables 8–10).
- Flow-matching RL alignment (incremental but important)
  - What’s new: preference optimization adapted to flow matching (Eq. 3), followed by GRPO with SDE sampling and closed-form KL (Eqs. 4–8) (Section 4.2.2).
  - Why it matters: improved controllability and compositionality on GenEval; the RL-enhanced model is the only one exceeding 0.9 overall (0.91; Table 4).

## 5. Experimental Analysis
- Evaluation setup
  - General T2I: DPG (dense prompts), GenEval (object-focused compositionality), OneIG-Bench (omni-dimensional English/Chinese), TIIF (instruction following) (§5.2.2; Tables 3–7).
  - Text rendering: CVTG-2K (English word accuracy), ChineseWord (new benchmark by this work), LongText-Bench (long English/Chinese) (Tables 8–10).
  - Editing: GEdit (semantic consistency + perceptual quality), ImgEdit (nine editing tasks) (§5.2.3; Tables 11–12).
  - Multi-task “editing-like”: Novel View Synthesis (GSO); depth estimation across five datasets (Tables 13–14).
  - VAE reconstruction: ImageNet-256 and a text-rich corpus (Table 2), to probe the upper bound of detail preservation.

- Main quantitative results (selected highlights)
  - General T2I
    - DPG (Table 3): overall 88.32, with strong sub-scores: Global 91.32, Entity 91.56, Attribute 92.02, Relation 94.31, Other 92.73. It edges out Seedream 3.0 (88.27) and is clearly ahead of GPT Image 1 [High] (85.15).
    - GenEval (Table 4): base model 0.87 overall; RL-enhanced reaches 0.91, surpassing FLUX.1 [Dev] (0.66), SD3.5 Large (0.71), GPT Image 1 [High] (0.84), Seedream 3.0 (0.84).
    - OneIG-Bench-EN (Table 5): overall 0.539, highest among compared models; especially strong on Alignment (0.882) and Text (0.891).
    - OneIG-Bench-ZH (Table 6): overall 0.548, leading; Text score 0.963.
    - TIIF (Table 7): instruction following—Qwen-Image ranks second overall (e.g., 86.14/86.83 short/long), behind GPT Image 1 [High] (89.15/88.29), but ahead of others.
  - Text rendering
    - English CVTG-2K (Table 8): average Word Accuracy 0.8288 (near GPT Image 1 [High] at 0.8569; much higher than Seedream 3.0 at 0.5924).
    - ChineseWord (Table 9): overall 58.30 vs GPT Image 1 [High] 36.14 and Seedream 3.0 33.05. By tier: Level-1 97.29, Level-2 40.53, Level-3 6.48 (rare characters remain hard).
    - LongText-Bench (Table 10): 0.943 (EN) and 0.946 (ZH). It is second-best on English (GPT Image 1 [High] 0.956) but far ahead on Chinese long text (GPT Image 1 [High] 0.619; Seedream 3.0 0.878).
  - Editing
    - GEdit (Table 11): top scores in both English and Chinese. 
      - EN: SQ 8.00, PQ 7.86, Overall 7.56 (vs GPT Image 1 [High] 7.53 overall).
      - CN: SQ 7.82, PQ 7.79, Overall 7.52 (GPT Image 1 [High] 7.30; FLUX.1 Kontext [Pro] collapses due to limited Chinese: overall 1.23).
    - ImgEdit (Table 12): highest “Overall” 4.27 vs GPT Image 1 [High] 4.20 and FLUX.1 Kontext [Pro] 4.00.
  - Multi-task generalization
    - Novel View Synthesis (GSO; Table 13): PSNR 15.11, SSIM 0.884, LPIPS 0.153. Close to specialized CRM (15.93/0.891/0.152) and better than general baselines like GPT Image 1 [High] (12.07/0.804/0.361).
    - Depth (Table 14): competitive among generalists; e.g., KITTI AbsRel 0.078 (δ1 0.951), NYUv2 0.055 (0.967), ScanNet 0.047 (0.974)—not reaching specialized Metric3D v2 on all datasets but strong for a generative model.
  - VAE reconstruction (Table 2):
    - On ImageNet-256: PSNR 33.42, SSIM 0.9159—state-of-the-art among compared VAEs with similar latent specs.
    - On text-rich data: PSNR 36.63, SSIM 0.9839; qualitative zoom-ins show clearer small text (Figure 17).
  - Human arena (Figure 16): in a large-scale Elo system with >10k pairwise votes per model (Chinese prompts excluded), Qwen-Image ranks third overall and outperforms GPT Image 1 [High] and FLUX.1 Kontext [Pro].

- Do the experiments support the claims?
  - Complex text rendering: Yes. Strong quantitative gains for Chinese and long texts (Tables 9–10) and robust English rendering (Table 8) match extensive qualitative demonstrations (Figures 18–21).
  - Consistent editing: Yes. Qwen-Image tops or matches SOTA on GEdit/ImgEdit (Tables 11–12) and shows visual consistency in style, pose manipulation, and novel views (Figures 24–28).
  - General T2I capability: Yes. It leads or is near the top on DPG, GenEval (after RL), and OneIG (Tables 3–6).
  - Robustness/ablations: While the paper offers rich comparisons, explicit ablations isolating `MSRoPE`, dual-encoding, or the data curriculum are not reported as separate studies; support is largely indirect via end-to-end gains.

## 6. Limitations and Trade-offs
- Rare-character ceiling remains
  - Despite large gains in Chinese, Level-3 accuracy is 6.48 (Table 9), indicating very rare glyphs remain challenging and would likely need more targeted synthesis or font coverage.
- Instruction following vs. absolute SOTA
  - On TIIF, Qwen-Image is second to GPT Image 1 [High] (Table 7). Some instruction nuances—especially long, abstract reasoning instructions—may still benefit from stronger language-side modeling or RL reward shaping.
- Depth and 3D vs specialized experts
  - Depth metrics trail the very best discriminative models on some datasets (e.g., Metric3D v2 in Table 14), though performance is strong for a unified generative model. Highly precise metric depth or full 3D reconstruction remains out of scope.
- Compute and engineering complexity
  - Training uses a 20B-parameter diffusion transformer with a 7B VLM and a video-capable VAE, plus an industrial Producer–Consumer pipeline (Sections 2, 4.1.1–4.1.2). This implies considerable infrastructure costs and engineering overhead. Activation checkpointing was too slow for their regime (§4.1.2).
- Limited explicit ablations
  - The individual effect sizes of `MSRoPE`, dual-encoding, and various data stages are not isolated in ablation tables. The evidence is primarily holistic (end-to-end benchmarks).
- Safety and content controls
  - The pipeline filters NSFW and abnormal elements (Stage 1, Stage 5) and excludes AI-generated images in synthetic data (§3.1), but broader safety topics (biases, watermark robustness, adversarial misuse) are not deeply audited.

## 7. Implications and Future Directions
- Field-level impact
  - Elevates “text rendering” from a niche evaluation to a core design axis for image foundation models. The new ChineseWord benchmark and strong LongText-Bench scores (Tables 9–10) push the community toward multilingual, layout-aware generation.
  - Shows that a generative editing framework with dual conditioning can unify many “vision tasks” as special cases of editing (Figure 5), narrowing the gap between understanding and generation.
- Research directions
  - Rare glyphs and fonts: extend synthesis (e.g., coverage-adaptive fonts, OCR-in-the-loop filtering) and multilingual scripts beyond Chinese/English.
  - Explicit ablations: quantify contributions of `MSRoPE`, dual-encoding, and curriculum stages; analyze failure cases for dense, tiny text; explore robustness to occlusion/lighting.
  - Reward learning: richer, multimodal rewards for editing precision (e.g., identity preservation metrics) and text adherence (OCR-based, layout correctness).
  - 3D/video generalization: leverage the video-capable VAE to scale to video generation and consistent multi-view scene synthesis; couple with 3D priors or NeRF-like supervision.
  - Efficiency: investigate activation recomputation alternatives (e.g., selective checkpointing), low-rank adapters for MMDiT, or student distillation for faster inference.
- Practical applications
  - Design and marketing: posters, slides, brand assets, packaging—where multilingual text accuracy and layout fidelity are crucial (Figure 2).
  - Photo and content editing: high-quality, localized edits for media, e-commerce, and entertainment (Figure 4; Tables 11–12).
  - Vision–Language UIs: generate visual explanations with text embedded in the image (conclusion, §6), aiding communication of complex ideas.
  - Proto-3D and geometry-aware tasks: quick novel views (Table 13) and usable monocular depth (Table 14) as building blocks for 3D pipelines.

> In sum, Qwen-Image contributes a complete stack—data, architecture, training, and evaluation—focused on the under-served but essential capabilities of complex text rendering and consistent editing. Evidence across diverse public benchmarks (Tables 3–14; Figures 18–28) indicates it advances the practical frontier of multilingual, layout-aware image generation while laying groundwork for unified multimodal generation–understanding systems.
