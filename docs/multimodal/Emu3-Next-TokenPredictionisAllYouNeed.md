# Emu3: Next-Token Prediction is All You Need

**ArXiv:** [2409.18869](https://arxiv.org/abs/2409.18869)

## 🎯 Pitch

Emu3 introduces a unified multimodal Transformer that achieves state-of-the-art results in text, image, and video tasks using only next-token prediction, dispensing with complex diffusion pipelines and compositional architectures. By encoding all modalities into discrete tokens and training a single decoder from scratch, Emu3 not only matches or surpasses specialized systems like SDXL and LLaVA-1.6, but also dramatically simplifies model design, scaling, and deployment—paving the way for efficient, generalizable multimodal intelligence.

---

## 1. Executive Summary (2–3 sentences)
Emu3 is a single multimodal Transformer that learns to handle text, images, and videos using only next‑token prediction on a shared sequence of discrete tokens. By replacing diffusion pipelines and CLIP+LLM compositions with one decoder‑only model and a vision tokenizer, it achieves state‑of‑the‑art or competitive performance in text‑to‑image, text‑to‑video, and vision‑language understanding, while simplifying model design and scaling (see Figure 1 and Figure 2).

## 2. Context and Motivation
- Problem addressed
  - Multimodal generation and understanding are usually solved with specialized systems: diffusion models for image/video synthesis and CLIP+LLM combinations for vision‑language reasoning. This fragmentation complicates training, inference, and scaling.
  - The open question is whether the simple “predict the next token” paradigm that powers modern language models can also underpin strong multimodal models without diffusion or separate encoders.

- Why it matters
  - Unifying modalities under one objective would reduce engineering complexity, remove fragile cross‑model interfaces, and create a single scaling path for both perception and generation. This has practical value (simpler deployment/inference) and theoretical value (a common learning principle across modalities).

- Prior approaches and gaps
  - Diffusion models (e.g., SDXL) dominate image/video generation but need a separate text encoder and iterative sampling; they are not trained with next‑token prediction (Section 1; Figure 2).
  - CLIP+LLM systems (e.g., LLaVA‑1.6) dominate vision‑language tasks but are compositional; they depend on a pretrained vision encoder plus an LLM (Section 1; Figure 2).
  - Earlier unified next‑token attempts either coupled to diffusion (e.g., Emu) or lagged task‑specific SoTA (e.g., Chameleon) (Section 1).

- Positioning
  - Emu3 claims a clean, single‑decoder alternative: tokenize visual content into discrete codes and train one Transformer from scratch on mixed text‑image‑video sequences, achieving competitive or superior results versus flagship diffusion and CLIP+LLM systems (Section 1; Figure 2).

## 3. Technical Approach
Emu3 is an 8B‑parameter decoder‑only Transformer trained on mixed text, image, and video tokens using the same next‑token prediction loss.

- Data pipeline (Section 2.1)
  - Language: same high‑quality bilingual corpus as Aquila (Chinese+English).
  - Images
    - Curation: remove low resolution (<512×512), filter aesthetics with LAION predictor (>5.5), remove text‑heavy/monochrome images via OCR and color filters, and add special understanding‑oriented imagery (charts, tables, text‑rich) following DenseFusion (Section 2.1; page 4).
    - Captioning: build an image captioner by fine‑tuning Emu2‑17B on ~1M captions seeded by GPT‑4V prompts; inference accelerated with vLLM (Section 2.1).
  - Videos
    - Preprocess: scene split (PySceneDetect), text filtering (PaddleOCR), motion filtering with optical flow (exclude too static or too extreme), and aesthetic filtering (LAION) (Section 2.1).
    - Captioning: train a video captioner from the image captioner using GPT‑4V‑seeded labels; long videos are split into 10–20s clips; 12 evenly sampled frames for short clips (Section 2.1).

- Discrete vision tokenizer (Section 2.2; Table 1; Figure 3)
  - Architecture: SBER‑MoVQGAN backbone with added 3D temporal residual layers in encoder and decoder to support video (Table 1; Section 2.2).
  - Codebook size: 32,768; compression: 4× in time and 8×8 in space; a 512×512 image or a 4‑frame 512×512 video chunk becomes 4096 tokens.
  - Training: on LAION‑High‑Resolution images + InternVid videos with L2, LPIPS, GAN, and VQ commitment losses.
  - Reconstruction quality: on Pexels videos (5s, 8 frames at 12 FPS), LPIPS/PSNR/SSIM improve with higher resolution; e.g., at 720×720 LPIPS=0.110, PSNR=24.30, SSIM=0.771 (Table 2). Figure 3 shows visual reconstructions.

- Single Transformer architecture (Section 2.3; Table 3)
  - Llama‑style decoder with RMSNorm, grouped‑query attention (GQA), SwiGLU, rotary position embeddings (RoPE) with a large base (1,000,000) for very long contexts; 8B parameters; 32 layers; hidden size 4096; 32 heads; 131,072 context length to accommodate long video sequences (Table 3).
  - Vocabulary expanded to include vision tokens; multilingual text tokenization via `QwenTokenizer`.

- Multimodal serialization and training objective (Section 2.4)
  - Sequence format packs everything into one stream:
    - “[BOS] {caption text} [SOV] {meta text} [SOT] {vision tokens with EOL/EOF} [EOV] [EOS]”
    - `meta text` is plain text (e.g., resolution, fps, duration). Some samples move captions after `[EOV]` to train image understanding and Q&A settings.
  - Loss: standard next‑token cross‑entropy over the entire sequence; to avoid over‑fitting to visuals, vision‑token loss is down‑weighted by 0.5.
  - Training strategy: two stages
    - Stage‑1: text+image, context length 5120.
    - Stage‑2: add video, context length 131,072.
    - Both stages use the same base LR (5e‑5) with cosine annealing; training parallelized with tensor/context/data parallelism and careful packing to avoid splitting images (Section 2.4).

- Post‑training for generation (Section 2.5.1)
  - Quality Fine‑Tuning (QFT): continue next‑token training but supervise only vision tokens; select high‑quality data using averaged preference scores (HPSv2.1, MPS, LAION aesthetics); raise training resolution to 720p for images and apply resolution/flow filters for videos; linearly anneal LR to zero.
  - Direct Preference Optimization (DPO): adapt DPO to autoregressive image generation. For each prompt, generate 8–10 candidates, collect human votes on visual appeal and prompt alignment, form `(prompt, chosen, rejected)` triplets, and fine‑tune with DPO loss + next‑token loss. Token streams from candidates are stored to avoid re‑tokenization artifacts (Section 2.5.1; Figure 6).

- Post‑training for vision‑language understanding (Section 2.5.2)
  - Stage‑1: image‑to‑text with mixed image‑understanding and pure‑text data (vision‑token losses ignored for text‑only prediction).
  - Stage‑2: instruction tuning using a subset of QA pairs from LLaVA‑OneVision [44]; images resized within [512, 1024] while preserving aspect ratio.

- Inference & post‑processing
  - Video: natively generate 5s clips at 24 FPS; longer videos by autoregressively extending tokens (Section 3.2; Figure 7, Figure 8).
  - Appendix B.1 gives sampling details for image benchmarks (e.g., guidance scales, Top‑k=16384, Top‑p=1.0; Emu3 outputs 512×512, Emu3‑DPO outputs 720×720).
  - Appendix B.2 applies learned stabilization and 4× super‑resolution to videos before evaluation; VBench comparisons are run on these processed outputs.

Analogy: the system treats everything—words, pixels, and frames—as entries in one very long “sentence.” It learns to keep writing the sentence one token at a time, whether the next token is a word, a pixel‑code, or the next video frame code.

## 4. Key Insights and Innovations
- A single next‑token decoder for generation and perception (Figure 1; Section 1)
  - What’s new: no diffusion steps, no CLIP encoder, and no LLM+vision encoder composition. Visual inputs are discretized into tokens and fed directly to the same decoder used for text.
  - Why it matters: simplifies architecture and unifies training/inference; enables extremely long context windows for video reasoning and generation (131k tokens; Table 3).

- Open‑sourced video‑capable vision tokenizer (Section 2.2; Table 1; Figure 3; Table 2)
  - What’s new: MoVQGAN augmented with temporal residual layers; one tokenizer handles both images and videos, compressing time and space jointly.
  - Why it matters: previous open tokenizers were primarily image‑focused; high‑fidelity video tokenization is a bottleneck for autoregressive video generation and long‑context training.

- Document‑like multimodal serialization with meta text (Section 2.4)
  - What’s new: a simple but effective way to fold captions, visual metadata (resolution, fps), and vision tokens into one sequence, plus strategic placement of captions after visuals to train understanding.
  - Why it matters: allows the same objective to support both conditional generation (prompt before visuals) and perception/Q&A (prompt after visuals).

- DPO adapted to autoregressive visual generation (Section 2.5.1; Figure 6)
  - What’s new: standard DPO, widely used in language alignment, is applied directly to tokenized images to optimize human preference for visual quality and prompt adherence.
  - Why it matters: boosts human‑judged image quality and alignment (Figure 6) and provides a blueprint for aligning AR vision models without diffusion.

- Autoregressive video generation and future prediction (Section 3.2; 3.3; Figure 7; Figure 8)
  - What’s new: causal generation of videos by predicting the next visual token; videos can be extended arbitrarily by continuing the sequence.
  - Why it matters: demonstrates that the next‑token paradigm can model temporal dynamics competitively with diffusion on VBench (Table 5) and can extend a given video by predicting future frames (Figure 8).

## 5. Experimental Analysis
- Evaluation setup and metrics
  - Image generation
    - Benchmarks: MSCOCO‑30K (CLIP‑I, CLIP‑T, FID), GenEval (object/attribute/position tests), T2I‑CompBench (color/shape/texture binding), DPG‑Bench (long‑prompt following) (Section 3.1; Table 4, Table 7, Table 8).
    - Prompt rewriting: for GenEval and T2I‑CompBench, prompts are expanded with GPT‑4V to match the model’s strength in dense captions; both original and rewritten variants are reported (Table 4; Appendix B.1).
    - Human evaluation: 100 prompts, 3 raters focusing on “visual quality” and “prompt following” (Section 3.1.2; Figure 5).
  - Video generation
    - Benchmark: VBench (16 dimensions; report 11 plus total score) comparing 13 state‑of‑the‑art models; Emu3 is the only autoregressive model in the table (Section 3.2; Table 5).
    - Note: video outputs are stabilized and super‑resolved before evaluation (Appendix B.2).
  - Vision‑language understanding
    - 12+ datasets spanning OCR, VQA, charts, diagrams, and real‑world QA (Section 3.4; Table 6).

- Quantitative highlights
  - MSCOCO‑30K (Table 4)
    - Emu3: CLIP‑I 0.689 (slightly above SDXL’s 0.674); CLIP‑T 0.313 (≈ SDXL 0.310); FID 12.8 (worse is higher; better FIDs are achieved by diffusion baselines like PixArt‑alpha at 7.32). This shows stronger text‑image similarity and prompt relevance than SDXL by CLIP metrics, but weaker FID.
  - GenEval (Table 7)
    - With rewritten prompts, Emu3 overall 0.66 vs DALL‑E 3 at 0.67 and SDXL at 0.55; SD3 reaches 0.74. Emu3 is close to top diffusion systems without using any diffusion or external text encoder.
  - T2I‑CompBench (Table 7)
    - With rewriting, Emu3 reaches 0.79 (color), 0.58 (shape), 0.74 (texture), comparable to diffusion SoTA (e.g., DALL‑E 3: 0.81/0.68/0.81).
  - DPG‑Bench (Table 8)
    - Emu3‑DPO overall 81.6, exceeding SDXL (74.65) and PixArt‑alpha (71.11) and approaching proprietary SoTA like DALLE‑3 (83.50). This benchmark stresses long, detailed prompts.
  - Human evaluation (Figure 5)
    - Emu3‑DPO improves perceived visual quality and prompt following (Figure 6). Overall scores place it above SDXL and roughly on par with DALLE‑3/MJ‑v5.2 on the 100‑prompt study (Figure 5; bar heights indicate ~70 on English prompts and high‑60s on Chinese for Emu3‑DPO).
  - Video generation (Table 5)
    - Total VBench score: 80.96. Emu3 is competitive with diffusion models (e.g., Gen‑3 82.32, Kling 81.85, CogVideoX‑5B 81.61).
    - Standout dimensions: “Dynamic degree” 79.27 (best in table), strong “Subject”/“Background consistency” (~95–98). Weaker in “Aesthetic quality” (59.64) and “Human action” (77.71).
  - Vision‑language understanding (Table 6)
    - Emu3 (encoder‑free) surpasses many encoder‑based 7B systems on several datasets: SEEDBench‑Img 68.2 (vs LLaVA‑1.6’s 64.7), strong scores on TextVQA 68.6* and ChartQA 76.3*; lower on MMMU (31.6). Asterisks mark datasets whose images overlap with training, so those particular gains need caution.

- Do experiments support claims?
  - Generation: Yes, by multiple metrics and a human study. Emu3 outperforms SDXL in long‑prompt following (DPG‑Bench) and human preference, and approaches DALLE‑3 on GenEval with rewritten prompts (Table 7, Table 8, Figure 5). FID is not SoTA, but CLIP‑based alignment and human judgments favor Emu3 in several settings (Table 4).
  - Video: Competitive total VBench, best dynamics, but aesthetics lag top proprietary diffusion (Table 5). The use of stabilization and super‑resolution during evaluation (Appendix B.2) is disclosed and should be considered when comparing to baselines that may or may not apply similar post‑processing.
  - Understanding: Strong average performance among encoder‑free models and competitive with popular encoder‑based systems (Table 6), though some datasets have training exposure.

- Ablations / robustness
  - DPO: Improves human‑rated visual quality and alignment (Figure 6), but slightly reduces some automated scores (Table 4 notes a decline after DPO), likely because preference data emphasize global aesthetics versus metric‑specific alignment.
  - Prompt rewriting: Emu3 benefits from richer prompts, consistent with its training on dense captions (Table 4, Table 7, Appendix B.1).

## 6. Limitations and Trade-offs
- Tokenizer bottlenecks
  - Discrete tokenization inevitably loses some detail; despite good LPIPS/PSNR/SSIM (Table 2), FID is not SoTA on MSCOCO (Table 4). Improving codebook design or compression might raise both fidelity and speed.

- Aesthetics vs dynamics trade‑off in video
  - Emu3 excels at motion consistency (“Dynamic degree” 79.27) but trails diffusion in aesthetic quality (59.64) (Table 5). Autoregressive causality may prioritize temporal coherence over per‑frame polish without additional refinements.

- Dependence on prompt richness
  - Emu3’s performance notably improves when prompts are rewritten/expanded (Table 7), suggesting sensitivity to prompt detail due to training on dense synthetic captions.

- Evaluation caveats
  - Some understanding benchmarks include images seen during training (asterisk in Table 6), inflating those specific scores.
  - Video evaluation uses stabilized and super‑resolved outputs (Appendix B.2). This is reasonable for user‑facing quality but complicates strict like‑for‑like comparisons if baselines do not apply similar processing.

- Compute and memory
  - Very long context (131k tokens; Table 3) and joint training over modalities require heavy parallelism (TP, CP, DP; Section 2.4). Practical training cost and inference latency for long videos may be significant.

- Scope
  - No audio modality; video lengths beyond a few seconds rely on autoregressive extension and may accumulate drift. Human action realism and fine‑grained aesthetics lag the very best diffusion systems (Table 5).

## 7. Implications and Future Directions
- How it changes the landscape
  - Demonstrates that a single next‑token decoder can rival specialized diffusion and CLIP+LLM stacks across generation and perception (Figure 2; Tables 4–6). This simplifies system design and suggests scaling laws for multimodal models that mirror language LMs.

- Follow‑up research enabled
  - Tokenization
    - Explore larger or hierarchical codebooks, variable‑rate compression, or learned entropy models to improve fidelity and efficiency.
  - Training curriculum
    - Better balancing of text/vision losses, curriculum over sequence lengths, and multi‑scale training for aesthetics and dynamics.
  - Alignment
    - Richer preference datasets (diverse cultures/languages/domains), multi‑objective DPO for both alignment and metric‑based faithfulness.
  - Long‑horizon video
    - Memory‑efficient attention, chunked generation with cross‑chunk constraints, and error‑correction during autoregressive extension.
  - Broader modalities and tasks
    - Add audio tokens for AV generation/understanding; unify detection, segmentation, and OCR by predicting structured tokens; integrate retrieval‑augmented generation inside the same next‑token framework.

- Practical applications
  - Text‑to‑image/video creation for media, education, and marketing; video continuation and future prediction for storyboarding and simulation; vision‑language assistants for diagrams, charts, and OCR‑heavy documents (Table 6 categories). The open‑sourced tokenizer and model components lower the barrier to such deployments.

> In summary, Emu3 provides concrete evidence (Tables 4–6; Figures 5–8) that next‑token prediction alone, when backed by an effective vision tokenizer and long‑context Transformer, can support high‑quality multimodal generation and understanding. While aesthetics and some understanding benchmarks still leave headroom, the simplification, competitiveness, and extensibility make this a compelling path for general multimodal intelligence.
