# Emu3: Next‑Token Prediction is All You Need

**ArXiv:** [2409.18869](https://arxiv.org/abs/2409.18869)
**Authors:** Xinlong Wang, Xiaosong Zhang, Zhengxiong Luo, Quan Sun, Yufeng Cui, Jinsheng Wang, Fan Zhang, Yueze Wang, Zhen Li, Qiying Yu, Yingli Zhao, Yulong Ao, Xuebin Min, Tao Li, Boya Wu, Bo Zhao, Bowen Zhang, Liangdong Wang, Guang Liu, Zheqi He, Xi Yang, Jingjing Liu, Yonghua Lin, Tiejun Huang, Zhongyuan Wang
**Institutions:** Beijing Academy of Artificial Intelligence (BAAI)

## 🎯 Pitch

By leveraging a unified token-based autoregressive model, Emu3 achieves competitive performance across image and video generation as well as vision-language understanding, without the need for separate diffusion models or CLIP encoders. This approach not only simplifies deployment and maintenance but also paves the way for scalable multimodal systems, challenging existing paradigms in visual content creation and perception.

---

## 1. Executive Summary (2-3 sentences)
Emu3 is a single, decoder-only Transformer trained purely with next-token prediction over a shared sequence of text, image, and video tokens. By replacing diffusion and separate vision encoders with a discrete “vision tokenizer,” Emu3 achieves competitive-to-state-of-the-art performance in image generation, video generation, and vision-language understanding (see Figures 1–2 and Tables 4–6), suggesting that a unified, token-based autoregressive paradigm can scale beyond language.

## 2. Context and Motivation
- Problem/Gaps addressed
  - Modern multimodal systems are split: diffusion models dominate image/video generation; CLIP+LLMs dominate understanding and instruction following. A single next-token model has not matched these specialized systems in both generation and perception.
  - The paper asks whether one decoder trained only with next-token prediction can unify high-quality image/video generation and strong vision-language understanding without diffusion or compositional pipelines (Figure 1).

- Why this matters
  - Practical: One model simplifies engineering and deployment (no separate diffusion scheduler, no CLIP encoder), potentially lowering maintenance and enabling shared improvements across tasks.
  - Scientific: It probes whether the core paradigm that scaled language models (next-token prediction) can generalize to high-fidelity visual generation and multimodal reasoning.

- Prior approaches and shortcomings
  - Diffusion models (e.g., SDXL) excel at photorealistic generation but require iterative denoising, separate encoders, and task-specific adaptations.
  - VLMs like LLaVA link a pretrained vision encoder (e.g., CLIP) to an LLM; they are strong at perception but not native generators.
  - Earlier autoregressive multimodal models (Emu, Chameleon, CM3Leon) either coupled to diffusion or underperformed task-specific models.

- Positioning
  - Emu3 trains a single Transformer from scratch on mixed text, image, and video sequences tokenized into a discrete space (Figure 1). It aims to match or exceed flagship task-specific systems (Figure 2) while keeping the entire pipeline within next-token prediction.

## 3. Technical Approach
Step-by-step, Emu3 converts all modalities to one token stream and trains a single decoder to predict the next token.

- Core idea: next-token prediction over mixed-modality tokens
  - “Next-token prediction” is the standard causal modeling objective: given a sequence of tokens, predict the next one. Emu3 applies it to a sequence that can contain text tokens and discrete vision tokens (Figure 1).

- Vision tokenizer (Section 2.2; Table 1; Figure 3; Table 2)
  - What it is: A learned compressor that maps images or videos to discrete indices in a finite `codebook` (here, 32,768 entries). Each index is a “vision token.”
  - Built on `SBER-MoVQGAN` (a VQGAN variant). VQGANs quantize latent features into codebook entries so images/videos become sequences of discrete codes.
  - Capabilities
    - Encodes a 512×512 image (or 4×512×512 video clip with temporal dimension) into 4,096 tokens from a 32,768-size codebook.
    - Compression: 4× temporally and 8×8 spatially (Table 1).
    - Architecture tweaks: adds two temporal residual layers with 3D convolutions in both encoder/decoder to better capture motion (Section 2.2).
  - Training data/objective: LAION-High-Resolution (images) + InternVid (videos), optimized with L2, LPIPS (a perceptual distance), GAN loss, and VQ commitment loss.
  - Quality: Reconstruction metrics on Pexels videos show LPIPS ≈ 0.11–0.112, PSNR up to 24.3, SSIM up to 0.771 as resolution increases (Table 2); qualitative reconstructions in Figure 3.

- Model architecture (Section 2.3; Table 3)
  - A decoder-only Transformer with:
    - 8B parameters; 32 layers; hidden size 4096; intermediate size 14336; 32 attention heads with 8 KV heads (GQA).
    - `GQA` (grouped-query attention): reduces KV cache size and improves efficiency by sharing keys/values across groups of queries.
    - `RoPE` (rotary position embeddings) with a large base (1,000,000) to support very long context.
    - `RMSNorm`, `SwiGLU`; no biases in qkv/linear; dropout 0.1.
    - Vocabulary size 184,622 (text + vision tokens); context length up to 131,072 tokens for long videos.
  - Tokenizer for text: Qwen tokenizer (multilingual).

- Unified data format and packing (Section 2.4)
  - Each training example is a “document” made of text and vision tokens with special markers:
    - Format: `[BOS] caption [SOV] meta [SOT] vision_tokens [EOV] [EOS]`
    - `[SOV]`: start-of-vision input; `[SOT]`: start-of-vision tokens; `[EOV]`: end-of-vision; `[EOL]`: line breaks; `[EOF]`: frame breaks.
    - `meta` carries properties in plain text, e.g., image resolution; for video: resolution, frame rate, duration.
    - To train understanding, some data places the caption after `[EOV]` so the model must read vision tokens first, then generate text.
  - Loss: standard cross-entropy next-token loss; vision-token losses weighted by 0.5 so text learning is not overwhelmed by dense visual sequences.

- Training strategy (Section 2.4)
  - Two-stage pretraining:
    1) Stage 1 on text+images with context length 5,120, training from scratch.
    2) Stage 2 introduces videos and extends context to 131,072 tokens.
  - Optimization: LR 5e-5 with cosine decay to zero; heavy use of tensor, context, and data parallelism; careful packing to avoid splitting an image across sequences.

- Data curation and labeling (Sections 2.1, A.1)
  - Language: same corpus as Aquila—bilingual Chinese/English.
  - Images: web data + in-house + AI-generated, filtered by resolution (≥512×512), LAION aesthetics score (>5.5), OCR- and color-based filters to de-prioritize text-heavy or monochrome images; extra “text-rich” categories via DenseFusion pipeline.
  - Captions: bootstrap an Emu2-17B image captioner using about 1M GPT‑4V-labeled pairs; inference accelerated with vLLM.
  - Videos: segmented into scenes (PySceneDetect); OCR filter; motion filtering via RAFT optical flow; aesthetics filter; captions from a video captioner fine-tuned from the image captioner, seeded by GPT‑4V prompts; frame sampling strategy depends on clip length (Section 2.1).

- Post-training for generation (Section 2.5.1)
  - Quality fine-tuning (QFT): continue next-token training but supervise only vision tokens; curate high-aesthetic images/videos; raise resolution from 512 to 720 during QFT; linear LR anneal to zero at the end.
  - DPO alignment for generation:
    - `DPO` (Direct Preference Optimization) is a technique to align models with human preferences using pairwise comparisons without training an explicit reward model.
    - Process: for each prompt, sample 8–10 candidates; three human voters score visual appeal and prompt alignment; build triplets (prompt, chosen, rejected); store the exact generated tokens to avoid re-tokenization mismatch; optimize DPO loss plus next-token loss.

- Post-training for vision-language understanding (Section 2.5.2)
  - Stage 1: image-to-text training mixed with text-only data; do not backprop through vision tokens when predicting text-only.
  - Stage 2: instruction tuning on a subset of QA pairs from LLaVA-OneVision [44]; images are resized to fit within 512–1024 while preserving aspect ratio.

- Inference and post-processing (Appendix B)
  - Image generation: Top‑k=16,384, Top‑p=1.0; output 512×512 for Emu3, 720×720 for Emu3-DPO (B.1).
  - Video post-processing: to improve temporal stability and resolution for evaluation, they train two auxiliary models (B.2):
    - Stabilization: temporal VAE (from Stable Video Diffusion) trained with L1, LPIPS, GAN, KL losses on 16×256×256 clips.
    - Super-resolution: a spatiotemporal U-Net upsampling by 4× using BlurPool downsampling and sub-pixel upsampling; trained with L2, LPIPS, GAN losses.

## 4. Key Insights and Innovations
- A single next-token decoder can rival specialized systems across modalities
  - What’s new: No diffusion, no CLIP encoder, no multi-model composition—just one Transformer predicting the next token over a unified vocabulary (Figure 1).
  - Why it matters: It simplifies the stack while achieving competitive results in three domains at once: image generation, video generation, and V+L understanding (Figure 2; Tables 4–6).

- Practical, open vision tokenizer for images and videos
  - What’s new: A public tokenizer that discretizes both images and videos into one codebook (Table 1) with temporal modeling via 3D residual layers; consistent compression factors; demonstrated reconstructions (Figure 3) and metrics (Table 2).
  - Why it matters: Prior public tokenizers for video were either missing or limited; this is a key enabler for training token-based autoregressive models on video.

- Long-context unified data format with explicit “vision structure” tokens
  - What’s new: A document-style format with `[SOV]/[SOT]/[EOV]/[EOF]/[EOL]` plus plain-text meta that lets the same model handle captioning, understanding, and generation; enormously long context (131k) makes long video modeling feasible (Section 2.4).
  - Why it matters: It standardizes multimodal supervision under one objective, making future scaling straightforward.

- DPO adapted to autoregressive vision generation
  - What’s new: Apply DPO—commonly used in language alignment—to tune an AR vision generator using human preferences, storing the exact token sequences to avoid tokenization drift (Section 2.5.1).
  - Why it matters: Human-aligned visual quality and prompt following improve by human evaluation (Figure 6), addressing a classic gap between automatic metrics and user preference.

- Demonstrated future video prediction via pure next-token extension
  - What’s new: Given 2 seconds of video tokens as context, the model autoregressively predicts future frames, enabling iterative extension (Figure 8).
  - Why it matters: Shows temporal world modeling without diffusion or explicit physics priors; supports the claim that causal token modeling can simulate plausible dynamics.

## 5. Experimental Analysis
- Evaluation setup
  - Image generation
    - Datasets/benchmarks: MSCOCO‑30K (CLIP-I, CLIP-T, FID), GenEval (object/attribute alignment), T2I‑CompBench (binding tests), DPG‑Bench (long-prompt following).
    - Generation settings: Top‑k=16,384; Top‑p=1.0; 512×512 (Emu3), 720×720 (Emu3‑DPO). Prompt rewriting for GenEval/T2I‑CompBench using GPT‑4V to expand short prompts, reported as “+ Rewriter” (Appendix B.1).
  - Video generation: VBench across 16 criteria; the paper reports 11 key metrics plus the total score (Table 5). Post-processed videos (stabilization and SR) are evaluated (Appendix B.2).
  - Vision-language understanding: 12+ benchmarks including SEEDBench‑Img, OCRBench, MMVet, VQAv2, GQA, ScienceQA-Img, TextVQA, ChartQA, DocVQA, InfoVQA, AI2D, RealWorldQA, MMMU, MMBench (Table 6).

- Main quantitative results
  - Image generation (Table 4; Appendix B.1)
    - MSCOCO‑30K: Emu3 CLIP‑I 0.689 and CLIP‑T 0.313; FID 12.8. Emu3-DPO has slightly lower CLIP scores and worse FID (19.3).
      - Note: Diffusion baselines have lower FID (better), e.g., PixArt-alpha 7.32; Transfusion 6.78. So FID is not Emu3’s strength.
    - GenEval:
      - Emu3 “+ Rewriter” overall 0.66 vs SDXL 0.55; close to DALL‑E 3 at 0.67 (Table 4 and Table 7).
      - Without rewriting: 0.54 overall; rewriting especially boosts “Position” and “Color Attribute” (Table 7).
    - T2I‑CompBench:
      - With rewriting: Emu3 scores 0.791 (Color), 0.585 (Shape), 0.742 (Texture), competitive with strong diffusion models (Table 4, Table 7).
    - DPG‑Bench (long prompts): Emu3‑DPO overall 81.6, surpassing SDXL (74.65) and PixArt‑alpha (71.11), approaching DALL‑E 3 (83.5) (Table 4 and Table 8).
    - Human study: On 100 prompts with three voters each, Emu3 outperforms SDXL and is “on par” with DALL‑E 3 and Midjourney v5.2 (Figure 5). DPO improves visual quality and prompt following in human preference (Figure 6).
  - Video generation (Table 5; Figure 7)
    - VBench total score: Emu3 80.96, comparable to top diffusion models (e.g., OpenSora‑1.2 at 79.76) and near proprietary leaders like Kling (81.85) and Runway Gen‑3 (82.32).
    - Strong areas: motion smoothness 98.93, dynamic degree 79.27, subject/background consistency 95.32/97.69.
    - Weaker areas: human action 77.71 (lower than CogVideoX-5B’s 99.40) and scene/appearance style scores (~37–24).
  - Vision-language understanding (Table 6; Section 3.4)
    - Emu3, without a pretrained CLIP or LLM, exceeds or approaches encoder-based methods on several tasks. Examples:
      - ScienceQA‑Img: 89.2 (noted with “*” indicating overlap with training images).
      - ChartQA/TextVQA/DocVQA/InfoVQA: competitive to strong encoder-based VLMs.
    - Mixed results:
      - MMBench: 58.5 (below LLaVA‑1.6 at 67.4).
      - RealWorldQA: 31.6 (weaker than several encoder-based models).

- Do the experiments support the claims?
  - Generation:
    - Yes for alignment and preference: On GenEval (with rewrite), T2I‑CompBench, DPG‑Bench, and human evaluations, Emu3 is competitive with SDXL and approaches DALL‑E 3 (Tables 4, 7, 8; Figures 5–6).
    - Less so for FID: Emu3’s FID is worse than leading diffusion and autoregressive-diffusion methods (Table 4).
  - Video:
    - Yes: Emu3 is competitive by total VBench score and shows qualitatively coherent, high-fidelity clips (Figure 7). Post-processing contributes to final quality (Appendix B.2).
  - Vision-language understanding:
    - Largely supportive: Despite lacking a pretrained vision encoder or LLM, Emu3 achieves solid scores across a broad suite (Table 6). Some scores are flagged with “*” due to training data overlap, and it underperforms on MMBench and RealWorldQA.

- Ablations/robustness/notes
  - DPO trade-off: Automated metrics slightly drop after DPO (Table 4), while human preference improves (Figure 6). The paper attributes this to DPO data emphasizing aesthetics, which differs from automated scorers’ criteria.
  - Future prediction: Demonstrated qualitatively (Figure 8), no quantitative metric reported.
  - Prompt rewriting: For GenEval/T2I‑CompBench, Emu3 benefits notably from GPT‑4V-based prompt rewriting (Appendix B.1), complicating apples-to-apples comparisons if other methods are not similarly rewritten.

## 6. Limitations and Trade-offs
- Discrete token bottleneck
  - The 32,768-codebook tokenizer compresses 512×512 images into 4,096 tokens (Section 2.2). While efficient, quantization can lose fine detail, which may relate to weaker FID versus diffusion (Table 4).

- Sequence length and compute
  - Autoregressive decoding scales linearly with the number of tokens; long videos (131k context) and high-res images produce very long sequences, impacting inference latency and memory. The paper notes heavy use of TP/CP/DP to train (Section 2.4) but does not provide explicit compute cost or throughput numbers.

- Post-processing dependency for videos
  - Stabilization and 4× super-resolution are applied before evaluation (Appendix B.2). While practical, it means the raw generator’s output is not directly evaluated by VBench, and improvements partially come from auxiliary models.

- Data construction and potential bias
  - Extensive use of synthetic captions from GPT‑4V and an in-house captioner fine-tuned on them (Section 2.1) could bias the model toward verbose or stylized descriptions, which in turn motivates prompt rewriting in evaluation (Appendix B.1).
  - Some V+L benchmarks mark “*” indicating potential training image overlap (Table 6), which can inflate perceived generalization.

- Mixed performance on some perception tasks
  - Lower scores on MMBench and RealWorldQA (Table 6) indicate there is still a gap to leading encoder-based VLMs in certain reasoning or real-world robustness settings.

- Missing modalities and controls
  - No audio; limited discussion of controllability beyond text prompts (e.g., fine-grained pose/layout control).

## 7. Implications and Future Directions
- Field impact
  - Emu3 provides concrete evidence that one decoder optimized by next-token prediction can be competitive in vision generation and understanding. It challenges the assumption that diffusion or CLIP-based compositions are necessary for SOTA multimodal performance (Figures 1–2; Tables 4–6).
  - The open vision tokenizer and unified data format lower the barrier to building token-based multimodal LMs, likely spurring a wave of “everything-as-tokens” research for images and videos.

- Follow-up research enabled/suggested
  - Better tokenizers
    - Variable-bitrate or hierarchical tokenizers to reduce sequence length for static regions, capture fine detail where needed, and improve FID without excessive token counts (Table 2 hints room to improve perceptual scores).
  - Efficient long-context decoding
    - Memory- and compute-efficient inference for 100k+ token contexts; combining GQA with speculative decoding or caching across frames.
  - Alignment beyond aesthetics
    - Extend DPO to multi-dimensional rewards (faithfulness, factual OCR, composition, safety) and study trade-offs systematically (Figure 6 shows current focus on visual quality and prompt alignment).
  - Robustness and generalization
    - Quantify out-of-distribution generalization; disentangle training data overlap (Table 6 marks) through held-out test sets and rigorous decontamination.
  - Raw-generator evaluation for video
    - Report VBench both before and after stabilization/SR to attribute gains accurately (Appendix B.2).
  - Multimodal extensions
    - Add audio tokens for video-audio generation; integrate spatial layout or 3D tokens; explore planning/control tasks where future prediction (Figure 8) can be evaluated quantitatively.

- Practical applications
  - Unified creative tools: one model for drafting images, making short videos, and answering questions about visual content.
  - Video editing and extension: autoregressive continuation from context (Figure 8) for storyboarding, surveillance forecasting, or content completion.
  - Multimodal assistants: vision-language understanding (Table 6) plus native generation makes end-to-end assistants feasible without coordinating multiple subsystems.

> Core takeaway: By showing that careful tokenization, long-context training, and preference alignment let a next-token decoder compete with diffusion and encoder-based VLMs, Emu3 substantiates “tokens all the way down” as a viable path to general multimodal intelligence (Figures 1–2; Tables 4–6).
