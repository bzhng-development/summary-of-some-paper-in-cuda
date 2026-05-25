# Qwen-Image Technical Report

**ArXiv:** [2508.02324](https://arxiv.org/abs/2508.02324)

## 🎯 Pitch

Qwen-Image introduces a new image foundation model that sets a new standard for high-fidelity text rendering—excelling in challenging tasks like multi-line, layout-sensitive, and Chinese text generation—and instruction-based image editing, where edits precisely target user-specified content while preserving semantic and visual fidelity. Its innovations—a progressive text-focused data pipeline, a dual-encoding architecture, and novel multimodal positional encoding—equip it to outperform leading models on both general image tasks and the hardest text rendering cases. This leap bridges critical gaps for real-world applications demanding accurate text-in-image and controlled editing, and paves the way for truly generalist, unified visual understanding and generation frameworks.

---

## 1. Executive Summary (2–3 sentences)
Qwen-Image is a large image generation and editing model that pushes two fronts simultaneously: high-fidelity text rendering (especially long, multi-line, and Chinese text) and precise, instruction-based image editing. It achieves these with a curriculum/data pipeline for text-in-image learning, a dual-encoding architecture that balances semantic consistency and visual fidelity, and a new multimodal positional encoding, delivering state-of-the-art results on compositional generation and Chinese text rendering while rivaling specialized models on tasks like novel view synthesis and depth estimation (see Figure 1, Tables 3–14).

## 2. Context and Motivation
- Problem addressed
  - Two persistent gaps in image generation/editing (Section 1):
    - Complex prompt alignment in text-to-image (T2I), especially rendering long, multi-line, layout-aware, and non-Latin (e.g., Chinese) text within images.
    - Editing alignment in text-and-image-to-image (TI2I): changing only what the instruction specifies while preserving the original image’s semantics (what things are) and appearance details (how they look).
- Why it matters
  - Real-world: posters, slides, UIs, signage, and documents require accurate text rendering and layout; practical editing tasks (style changes, object insertion/removal, pose manipulation) demand fine control (Figures 2, 4).
  - Scientific: bridging generation and understanding enables “generalist” models that can both synthesize and reason about visual content, unlocking unified workflows (Figure 5).
- Shortcomings of prior approaches
  - Leading models (e.g., GPT Image 1, FLUX, Seedream) perform well on general aesthetics but struggle with multi-line or localized text, Chinese characters, and complex layouts; they also often modify unintended regions during editing (Section 1; qualitative comparisons in Figures 18–23, 24–28).
- Positioning
  - Qwen-Image is a foundation model for both generation and editing that:
    - Builds a specialized data/annotation/synthesis pipeline for text rendering (Section 3; Figure 13).
    - Integrates a dual-encoding pathway (MLLM + VAE) and an improved MMDiT backbone with a new multimodal positional encoding (MSRoPE) (Section 2; Figures 6 and 8).
    - Trains across T2I, TI2I, and image reconstruction to align semantics and pixels (Sections 4.1, 4.3).

## 3. Technical Approach
This section unpacks the system end-to-end: data, model, objectives, training, and the editing extension.

- Overall architecture (Figure 6; Table 1)
  - Backbone: `MMDiT` (Multimodal Diffusion Transformer) processes text and image latents jointly.
  - Condition encoder: frozen `Qwen2.5-VL` (a multimodal LLM) extracts semantic features from prompts (and input images for editing).
  - Image tokenizer: a `VAE` (variational autoencoder) compresses images into latents for training and reconstructs them at inference.
  - New positional encoding: `MSRoPE` (Multimodal Scalable RoPE) encodes positions of both image and text tokens in a way that supports resolution scaling and clear modality separation (Figure 8).

- Inputs and representations
  - T2I: the textual prompt is formatted with a system template (Figure 7) and encoded by Qwen2.5-VL into a hidden representation `h` (Section 2.2).
  - TI2I (editing): in addition to the text instruction, the input image is encoded twice (Section 4.3; Figure 14):
    - Semantics via Qwen2.5-VL (captures content and context).
    - Reconstruction fidelity via the VAE encoder (preserves textures, colors, small details).
    - Both streams condition MMDiT; MSRoPE is extended with a “frame” dimension to distinguish original vs. edited images (right of Figure 14). The editing prompt template is shown in Figure 15.

- VAE design choices (Section 2.3; Table 2; Figure 17)
  - Single encoder, dual decoders: one shared encoder supports both images and videos (future-proofing for video), with a specialized image decoder fine-tuned on text-rich images.
  - Training objective excludes adversarial loss after observing that higher-quality reconstructions make the discriminator uninformative; the model balances reconstruction and perceptual losses with a dynamic ratio (Section 2.3).
  - Outcome: best-in-class reconstruction on ImageNet-256 and a text-rich corpus (Table 2), preserving small text that other VAEs blur (Figure 17).

- New multimodal positional encoding: `MSRoPE` (Section 2.4; Figure 8)
  - Problem: Naive concatenations or row-based 2D encodings can cause ambiguous overlaps in positional signals between text and image tokens, harming alignment and scaling (Figure 8A–B).
  - Mechanism: treat text as a 2D grid with identical position IDs on both axes and conceptually “place” it along the diagonal of the image grid. This maintains clear separation, supports resolution scaling, and is functionally equivalent to 1D RoPE for text (Figure 8C).

- Training objective: Rectified Flow with flow matching (Section 4.1; Equations 1–2)
  - Intuition: Instead of denoising step by step, learn a velocity field that moves a noisy latent `x1` to the clean latent `x0` along a straight path. The model predicts the instantaneous “velocity” `vt = x0 − x1` at a time `t ∈ [0,1]`.
  - Formulation:
    - Path: `xt = t x0 + (1−t) x1` and `vt = dxt/dt = x0 − x1` (Eq. 1).
    - Loss: mean-squared error between predicted and true velocities, conditioned on `h`: `L = E || vθ(xt, t, h) − vt ||²` (Eq. 2).

- Data pipeline and curriculum (Section 3; Figures 9–13)
  - Seven-stage filtering and rebalancing:
    - Stage 1–2: basic curation at 256p plus quality filters (rotation, luma, saturation, clarity, entropy, texture) to remove atypical or low-quality images (Figures 10–11).
    - Stage 3: image–text alignment improvements: splits by caption source; CLIP/SigLIP-2 filters; remove overly long or invalid captions (Section 3.2).
    - Stage 4: text rendering enhancement: split data by language (English / Chinese / other / non-text); inject synthetic text data; remove “intensive text” or tiny characters to ensure legibility (Section 3.2).
    - Stage 5: high-resolution refinement at 640p with aesthetic/watermark filters (Section 3.2).
    - Stage 6: category rebalance and portrait augmentation; targeted retrieval and synthesized captions emphasize faces, clothing, background, lighting (Section 3.2).
    - Stage 7: balanced multi-scale training at 640p and 1328p using a hierarchical taxonomy and resampling to address long-tail token frequency (Section 3.2).
  - Annotation: one-pass captioner (Qwen2.5-VL) emits both natural-language captions and structured metadata JSON (type, style, watermark list, abnormal elements), enabling scalable, rich supervision (Section 3.3; Figure 12).
  - Synthesis for text: three strategies (Section 3.4; Figure 13)
    - Pure rendering (characters on simple backgrounds) with strict failure rejection for any unrenderable character.
    - Compositional rendering (text written on objects in scenes) plus captions from a captioner.
    - Complex templates (slides/UIs) with rule-based template filling to preserve layout.

- System scaling and optimization (Sections 4.1.1–4.1.2)
  - Producer–Consumer pipeline: a Ray-like asynchronous setup where the Producer does data filtering, VAE encoding, and Qwen2.5-VL feature extraction, caching results by resolution; Consumers train MMDiT across GPUs and fetch batches via a zero-copy HTTP RPC layer (Section 4.1.1).
  - Parallelism/memory: hybrid data+tensor parallelism with Transformer-Engine; QK-Norm uses RMSNorm, others LayerNorm (Figure 6 caption); distributed optimizers; bfloat16 for all-gather and float32 for reduce-scatter for numerical stability (Section 4.1.2).
  - Trade-off study: activation checkpointing reduced memory by 11.3% (71GB→63GB/GPU) but slowed training 3.75× (2s→7.5s/iteration), so it is disabled (Section 4.1.2).

- Post-training alignment (Section 4.2)
  - SFT: human-curated, photorealistic, high-detail images to steer the model toward realism and fine detail (Section 4.2.1).
  - Preference-based RL adapted to flow models (Section 4.2.2):
    - `DPO` for flow matching (Eq. 3): for each prompt, gather a chosen vs. rejected image; compute preference differences in squared velocity prediction errors for policy vs. reference models; optimize a logistic objective favoring the chosen generations.
    - `GRPO` (Flow-GRPO; Eqs. 4–8): sample groups of G trajectories with an SDE-form of the flow to add stochasticity (Eqs. 6–7), compute within-group normalized advantages (Eq. 4), and optimize a PPO-style clipped objective with a closed-form KL regularizer between current and reference velocity fields (Eq. 8).
  - Purpose: DPO scales cheaply offline; GRPO provides fine-grained on-policy refinement. Together they improve compositionality and instruction following, as evidenced by GenEval (Table 4).

## 4. Key Insights and Innovations
- Curriculum + synthetic data targeted at text rendering (Sections 3.2–3.4; Figure 13)
  - What’s new: a carefully staged pipeline plus three synthesis modes that expose the model to rare characters (Chinese), multi-line paragraphs, and layout-rich templates.
  - Why it matters: it enables strong English and especially Chinese text rendering, including long text, which prior models struggled with.
  - Evidence: best overall on ChineseWord (58.30) with 97.29% on Level-1 characters and sizable gains on Level-2/3 (Table 9); near-top on English (CVTG-2K; Table 8); top or near-top on LongText-Bench for both languages (Table 10).

- Dual-encoding for editing (Section 4.3; Figure 14)
  - What’s new: feed semantic features from `Qwen2.5-VL` and reconstructive features from the `VAE`, and extend positional encoding with a “frame” dimension to disambiguate input vs. target images.
  - Why it matters: balances instruction adherence (semantics) with visual fidelity (appearance), reducing unintended changes.
  - Evidence: top overall on GEdit (G_O 7.56 EN / 7.52 CN; Table 11) and ImgEdit (Overall 4.27; Table 12), with qualitative consistency across pose changes and chained edits (Figures 24–27).

- `MSRoPE`: diagonal multimodal positional encoding (Section 2.4; Figure 8)
  - What’s new: a geometrically intuitive way to co-encode text and image positions that avoids positional collisions and supports resolution scaling without special-casing text rows/columns.
  - Why it matters: improves text–image alignment and scalable training across resolutions; complements the double-stream MMDiT.
  - Evidence: contributes to strong scores on compositional benchmarks DPG (Overall 88.32; Table 3) and GenEval (0.87 base → 0.91 with RL; Table 4), where position-sensitive instructions are critical.

- Flow-aware preference optimization (Section 4.2.2; Eqs. 3–8)
  - What’s new: formulate DPO and GRPO directly on the rectified-flow velocity objective, including an SDE sampler for exploration and a closed-form KL (Eq. 8).
  - Why it matters: aligns generations with human preferences while respecting the flow-matching dynamics, improving controllable generation without breaking training stability.
  - Evidence: GenEval improves from 0.87 to 0.91—the only foundation model above 0.9 in the table (Table 4).

- Scalable Producer–Consumer training (Section 4.1.1)
  - What’s new: push expensive preprocessing (VAE/MLLM encodings) off the training nodes and cache by resolution; zero-copy transfers to keep GPUs saturated.
  - Why it matters: practical scalability—critical for sustained multi-stage, multi-task training and RL.

## 5. Experimental Analysis
- Evaluation setup
  - Human ELO evaluation: “AI Arena” with ~5,000 diverse prompts and >200 evaluators; each model ≥10,000 pairwise comparisons; Chinese text prompts excluded to avoid bias against closed APIs (Section 5.1; Figure 16).
  - T2I (general compositionality): DPG, GenEval, OneIG-Bench EN/ZH, TIIF (Section 5.2.2; Tables 3–7).
  - Text rendering: CVTG-2K (English), ChineseWord (new), LongText-Bench EN/ZH (Section 5.2.2; Tables 8–10).
  - Editing (TI2I): GEdit EN/CN, ImgEdit; plus novel view synthesis (GSO) and depth estimation (five datasets), all unified as image-editing tasks (Section 5.2.3; Tables 11–14).
  - VAE quality: ImageNet-256 and a text-rich corpus (Section 5.2.1; Table 2; Figure 17).

- Headline quantitative results
  - Human ELO
    - > “Qwen-Image … ranks third … trailing Imagen 4 Ultra Preview 0606 by ~30 Elo points, but leading GPT Image 1 [High] and FLUX.1 Kontext [Pro] by >30” (Figure 16).
  - General compositionality
    - DPG: Overall 88.32, slightly above Seedream 3.0 (88.27) and well above GPT Image 1 [High] (85.15) and FLUX.1 [Dev] (83.84) (Table 3).
    - GenEval: Base 0.87 Overall; with RL 0.91—the best in the table (Table 4).
    - OneIG-Bench EN: Overall 0.539, slightly ahead of GPT Image 1 [High] (0.533) and Seedream 3.0 (0.530); top Text score 0.891 (Table 5).
    - OneIG-Bench ZH: Overall 0.548, ahead of Seedream 3.0 (0.528) and GPT Image 1 [High] (0.474); top Text score 0.963 (Table 6).
    - TIIF: second overall behind GPT Image 1 (Table 7). Qwen-Image maintains very high instruction-following, especially on long prompts.
  - Text rendering
    - English (CVTG-2K): `Word Accuracy` 0.8288 (vs. GPT Image 1’s 0.8569), `NED` 0.9116 (vs. 0.9478), `CLIPScore` 0.8017 (best among the listed models) (Table 8).
    - Chinese (ChineseWord): Overall 58.30 vs. GPT Image 1’s 36.14 and Seedream 3.0’s 33.05; especially strong on common (Level-1) characters at 97.29% (Table 9).
    - Long text (LongText-Bench): highest in Chinese (0.946), second-highest in English (0.943; GPT Image 1 is 0.956) (Table 10).
  - Editing and general vision
    - GEdit EN/CN: top Overall scores (7.56 EN; 7.52 CN), surpassing GPT Image 1 [High] (7.53 EN; 7.30 CN) and others (Table 11).
    - ImgEdit: best Overall 4.27 vs. GPT Image 1 [High] 4.20 and FLUX.1 Kontext [Pro] 4.00 (Table 12).
    - Novel view synthesis (GSO): PSNR 15.11 / SSIM 0.884 / LPIPS 0.153, close to specialized CRM (15.93 / 0.891 / 0.152) and ahead of general models like GPT Image 1 (12.07 / 0.804 / 0.361) (Table 13).
    - Depth estimation (zero-shot): competitive with diffusion-based and some classical baselines across KITTI/NYUv2/ScanNet/DIODE/ETH3D; not SOTA vs. specialized metric-depth models (Table 14).
  - VAE reconstruction
    - Top PSNR/SSIM on ImageNet-256 (33.42/0.9159) and especially on the text-rich corpus (36.63/0.9839), showing strong preservation of small text (Table 2; Figure 17).

- Qualitative support
  - Multi-line, multi-location text in both English and Chinese; layout-sensitive renders; object counting and relations; pose and style editing; chained edits; and consistent view rotations (Figures 18–28). Notable cases include paragraph-level English rendering (Figure 18), storefront/slide compositions (Figure 19), Chinese couplets and complex street scenes (Figures 20–21), and pose changes preserving details (Figure 26).

- Are the experiments convincing?
  - Strengths:
    - Broad, multi-benchmark coverage with both automatic and human evaluations.
    - Clear compositional and multilingual text rendering advantages; editing benchmarks include both EN and CN and cover fine-grained operations.
    - RL ablation via pre/post RL on GenEval shows concrete gains (Table 4).
  - Caveats:
    - ChineseWord is newly introduced by the authors; while well-motivated (long-tail characters), wider community adoption would further validate it.
    - TIIF and editing evaluations rely on GPT-4.1 scoring (Tables 11–12), which is common but not perfect; human side-by-side checks complement this but are separate (Figure 16).

## 6. Limitations and Trade-offs
- Data and supervision
  - Heavy reliance on large-scale curation, filtering, and synthetic pipelines (Section 3). While justified, the approach depends on the quality/coverage of templates, fonts, and captioners (Figure 13).
  - New benchmark (ChineseWord) is valuable but author-introduced; broader validation would strengthen claims (Table 9).
- Compute and scaling
  - Training requires substantial compute and system engineering; disabling activation checkpointing for throughput implies high memory usage (~71 GB/GPU reported; Section 4.1.2).
- Long-tail and rare cases
  - Despite big gains, very rare Chinese characters (Level-3) remain hard (6.48% accuracy; Table 9), indicating remaining long-tail gaps.
- Evaluation dependence
  - Some metrics depend on automated evaluators (e.g., GPT-4.1 for editing scores; CLIP-based metrics), which can encode their own biases (Tables 11–12).
- Task breadth vs. specialization
  - On depth, Qwen-Image is competitive but not SOTA versus specialized metric-depth models like Metric3D v2 and DepthAnything v2 (Table 14); this is expected for a generalist generator.

## 7. Implications and Future Directions
- How this work shifts the field
  - Demonstrates that a generalist image generator can excel at precise, multilingual text rendering and high-consistency editing—capabilities previously requiring task-specific systems. The dual-encoding design and MSRoPE offer practical blueprints for others (Sections 2.3–2.4, 4.3).
  - Shows that generation and “understanding” tasks (novel view synthesis, depth) can be cast as editing and benefit from a single model trained with rectified flow (Figure 5; Tables 13–14).
- What it enables next
  - Visual-Language User Interfaces (VLUIs): produce diagrams, slides, posters, and UI mockups from rich prompts, especially for Chinese/English bilingual contexts (Figures 2, 19–21).
  - Multimodal agents: an MLLM with Qwen-Image could “explain with pictures,” render document-like outputs, and iteratively edit them based on dialogue.
  - Toward video: the VAE’s shared encoder for image/video and Qwen-Image’s strong pose/view consistency suggest a natural path to video generation/editing with preserved identity and scene coherence (Section 6 discussion; Section 2.3).
- Concrete future research
  - Extend text rendering to more scripts (Arabic, Devanagari, Thai) using the same synthesis/curriculum recipe (Sections 3.2–3.4).
  - Public release and community validation of ChineseWord; expand LongText-Bench-style tests to multilingual, multi-layout scenarios (Tables 9–10).
  - Stronger safety and watermark handling via the annotation pipeline (Figure 12) and explicit training to avoid inadvertent logo/QR/barcode artifacts (Stage 5 in Section 3.2).
  - RL at scale for image editing: richer preference data for chained edits and global consistency, leveraging the flow-aware DPO/GRPO framework (Section 4.2.2).
  - Efficiency research: better memory/time trade-offs (e.g., selective activation checkpointing or low-rank adapters) without the 3.75× slowdown seen in Section 4.1.2.

> Bottom line: Qwen-Image combines a targeted curriculum for text, a principled dual-encoding architecture for editing, and flow-aware alignment to deliver top-tier compositional generation and editing, with standout performance on Chinese and long-text rendering. The system design and training recipes are broadly reusable and point toward unified multimodal generation-and-understanding models.
