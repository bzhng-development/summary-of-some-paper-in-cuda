# Qwen2-VL: Enhancing Vision-Language Model’s Perception of the World at Any Resolution

**ArXiv:** [2409.12191](https://arxiv.org/abs/2409.12191)

## 🎯 Pitch

Qwen2-VL breaks through the fixed-resolution barrier of previous vision-language models by introducing Naive Dynamic Resolution and Multimodal Rotary Position Embedding (M-RoPE), enabling seamless processing of images and videos at any size or aspect ratio. By unifying spatial and temporal understanding across modalities, Qwen2-VL achieves state-of-the-art results on tasks like OCR and video reasoning—empowering AI agents to perceive high-detail images and extended videos much like humans do. This advance sets a new benchmark for real-world multimodal applications, equipping both researchers and developers with open models that finally overcome critical limitations in perception, generalization, and cross-modal reasoning.

---

## 1. Executive Summary
Qwen2‑VL is a family of vision‑language models that removes the fixed‑resolution bottleneck in visual processing and unifies text, images, and videos with a single positional system. It introduces two core mechanisms—`Naive Dynamic Resolution` (variable visual tokenization at any image size) and `Multimodal Rotary Position Embedding (M‑RoPE)` (a 3‑axis positional scheme for time, height, width)—and scales the approach to 2B, 7B, and 72B parameters. The 72B model matches or surpasses leading closed models on many multimodal benchmarks (e.g., document OCR and video understanding), while revealing remaining gaps on highly challenging reasoning tests.

## 2. Context and Motivation
- Problem/Gaps addressed
  - Fixed‑resolution image handling in most LVLMs (large vision‑language models) either downsample or pad images to one size (e.g., 224×224). This loses small details in high‑resolution images (Section 1; Figure 2) and wastes compute on low‑detail images.
  - One‑dimensional positional embeddings in LLMs poorly represent spatiotemporal structure in images and videos, making long‑video reasoning and cross‑modal fusion harder (Introduction; Section 2.1; Figure 3).
  - Many LVLMs rely on a frozen CLIP‑style vision encoder, limiting adaptation to fine details and complex reasoning (Introduction).
  - The field has limited evidence on how scaling LVLMs and data changes capability (Introduction; Section 3.3.3).

- Why it matters
  - Real documents, charts, and UIs are often high‑resolution; losing details hurts OCR, grounding, and agent reliability.
  - Long videos (minutes to an hour) require clean temporal modeling and scalable token budgets.
  - Generalist agents need robust OCR, grounding, and function calling to operate devices and tools.

- What existed and where it fell short
  - Prior LVLMs downsample/upsample or scale‑then‑pad (cited in Introduction), locking models to a single token budget and aspect ratio.
  - Video was often treated as a separate modality with 1D positions, limiting temporal‑spatial fidelity (Introduction).
  - Some works fine‑tune the ViT to improve representations, but without a principled positional scheme for multimodality or a token‑efficient variable‑resolution pipeline (Introduction).

- This work’s position
  - Replaces fixed input sizing with `Naive Dynamic Resolution` and introduces `M‑RoPE`, a 3‑component (time, height, width) position system that is consistent across text, images, and videos (Section 2.1; Figures 2–3).
  - Unifies image/video processing (2 fps sampling, lightweight 3D convolutions) so the same stack handles both (Section 2.1, Unified Image and Video Understanding).
  - Scales to 72B parameters and 1.4T multimodal tokens with a clear training and infrastructure recipe (Sections 2.2, 2.3; Figure 6).

## 3. Technical Approach
Step‑by‑step, from inputs to outputs.

- Overall architecture (Section 2.1; Figure 2; Table 1)
  - A shared Vision Transformer (`ViT`, ~675M params) encodes images and videos.
  - An LLM backbone uses the Qwen2 series (2B/7B/72B variants).
  - A lightweight connector merges visual tokens into the LLM stream using special markers `<|vision_start|>` and `<|vision_end|>` (Section 2.2.1).

- Naive Dynamic Resolution (Section 2.1; Figure 2)
  - What it is: The vision stack accepts images at their native resolutions and converts them into a variable number of `visual tokens` (discrete embeddings fed to the LLM).
  - How it works:
    - Remove absolute position embeddings from the ViT and use `2D‑RoPE` (2D rotary embeddings) so positions are resolution‑agnostic.
    - For memory control, pack multiple images with different sizes into one sequence; cap the packed length to fit GPU memory (Section 2.1).
    - After the ViT, a small MLP compresses every 2×2 patch group into one token (quartering token count) before handing tokens to the LLM (Section 2.1).
    - Example token count: An image of 224×224 with patch size 14 becomes 66 visual tokens after 2×2 compression (Section 2.1).
  - Why this design: It preserves fine details when images are large, saves tokens when images are small, and avoids aspect‑ratio distortions from resizing (Introduction; Section 3.3.1 and Table 7).

- Multimodal Rotary Position Embedding, `M‑RoPE` (Section 2.1; Figure 3)
  - What it is: A positional system decomposed into three rotaries—temporal (t), height (h), width (w).
  - How it works:
    - Text: uses identical position IDs for t/h/w, reducing to standard 1D‑RoPE behavior (Figure 3).
    - Images: `t` is constant; `h` and `w` carry the 2D spatial layout (Figure 3).
    - Videos: `t` increases per frame; `h` and `w` mirror the image scheme (Figure 3).
    - Across multiple modalities in one prompt, positions for each new modality start from the previous modality’s max ID + 1 (Figure 3, caption; Section 2.1).
  - Why this design: Encodes space and time consistently across modalities and lowers absolute position IDs for images/videos, which improves length extrapolation (Section 2.1 and ablation in Section 3.3.2; Figure 5).

- Unified image and video handling (Section 2.1, “Unified Image and Video Understanding”)
  - Training mixes images and videos; every video is sampled at 2 fps.
  - Two‑layer `3D convolutions` (depth 2) process short temporal tubes, letting the model handle more frames without increasing the sequence length as much as pure 2D patching would (Section 2.1).
  - Each image is treated as two identical frames for consistency (Section 2.1).
  - Total tokens per video are capped at 16,384 during training; frame resolution is dynamically adjusted to respect this cap (Section 2.1).

- Training pipeline (Section 2.2)
  - Three stages:
    1) Pretrain the ViT on large image‑text corpora to align visual semantics.
    2) Unfreeze all parameters and continue multimodal pretraining.
    3) Freeze ViT and instruction‑tune the LLM with ChatML‑formatted multimodal dialogs.
  - Data scale: ~600B tokens (stage 1) + ~800B tokens (stage 2) = 1.4T total tokens; supervision is on text tokens only while being exposed to visual tokens (Section 2.2).
  - Initializations: LLM from Qwen2; ViT from DFN but replacing its fixed positional encoding with 2D‑RoPE (Section 2.2).
  - Instruction format and tools: ChatML; special tokens for images, boxes (`<|box_start|>…<|box_end|>`), and object references; agent tasks represented as sequences of tool calls with arguments and results (Section 2.2.1).

- Systems and scaling (Section 2.3)
  - 3D parallelism (data/tensor/pipeline), FlashAttention, fused ops; dynamic sequence lengths broadcast before pipeline execution; checkpointing on CPFS; vision data streamed from OSS (Section 2.3).
  - Practical detail: handled non‑deterministic conv ops in tensor parallelism via offline shared‑weight reduction (Section 2.3).

## 4. Key Insights and Innovations
- Dynamic tokenization for any image size (fundamental)
  - Different from prior fixed‑resolution pipelines: Qwen2‑VL converts images to variable token counts and compresses 2×2 token groups via an MLP (Section 2.1).
  - Why it’s significant: Better detail retention on large images and fewer wasted tokens on small ones, improving OCR, diagrams, and UI tasks. Ablation shows dynamic tokenization achieves “top‑tier performance while consuming fewer tokens on average” (Table 7).

- M‑RoPE: a position system that natively models time and 2D space (fundamental)
  - Unlike 1D positional schemes, M‑RoPE uses separate rotary components for `temporal`, `height`, and `width` (Section 2.1; Figure 3).
  - Significance:
    - Improves video benchmarks (Table 8).
    - Enables strong length extrapolation—from training at 16K tokens per video to inference at 80K tokens (Figure 5).

- A unified image–video recipe that scales to long videos (incremental but enabling)
  - Mixed training, 2 fps frame sampling, shallow 3D conv tubes, and token caps let one model handle images and 20+ minute videos (Section 2.1; Table 4).

- Systematized scaling of LVLMs (incremental but impactful)
  - Jointly scales model size (2B/7B/72B) and multimodal data (1.4T tokens). Performance curves show steady gains, especially in math‑reasoning tasks with bigger models (Figure 6).

## 5. Experimental Analysis
- Evaluation setup (Sections 3.1–3.2; Tables 2–6)
  - Broad coverage: general VQA (RealWorldQA, MMStar, MMVet, MME, MMBench, MMT‑Bench), document/diagram OCR (DocVQA, InfoVQA, ChartQA, TextVQA, OCRBench), multilingual OCR (MTVQA and internal OCR in Table 3), math‑vision (MathVista, MathVision), referring expression grounding (RefCOCO/+/g), video understanding (MVBench, PerceptionTest, EgoSchema, Video‑MME), and agent tasks (function calling, UI operations, robotics, navigation, card games).
  - Metrics vary by benchmark (accuracy, scores, success rate).

- Headline results (Table 2)
  - Strong wins in document OCR and visually indispensable tasks:
    - > “DocVQAtest: Qwen2‑VL‑72B = 96.5 vs prior SoTA 94.1; GPT‑4o = 92.8; Claude‑3.5 = 95.2.”
    - > “RealWorldQA: 77.8 vs GPT‑4o 75.4 and prior SoTA 72.2.”
    - > “MMVet: 74.0 vs GPT‑4o 69.1.”
    - > “MathVista test‑mini: 70.5 vs prior SoTA 69.0.”
  - Multilingual OCR (Table 3, internal): Qwen2‑VL‑72B surpasses GPT‑4o on 6 of 8 languages (e.g., Korean 94.5 vs 87.8; Japanese 93.4 vs 88.3) but trails on Arabic (70.7 vs 75.9).

- Video understanding (Table 4)
  - > “MVBench: 73.6 (72B).”
  - > “PerceptionTest: 68.0 (72B).”
  - > “EgoSchema: 77.9 (72B) vs GPT‑4o 72.2 and Gemini 1.5‑Pro 63.2.”
  - Video‑MME: 71.2/77.8 (without/with subtitles) for 72B; note the evaluation limited to 768 frames per video, which may dampen performance on hour‑long videos (Table 4; Section 3.2.6).

- Agents and tool use (Table 5)
  - Function calling:
    - > “Type Match (function selection): 93.1 (72B) vs GPT‑4o 90.2; Exact Match (arguments): 53.2 vs 50.0.”
  - UI operations (AITZ):
    - > “Type Match: 89.6 vs prior SoTA 83.0; Exact Match: 72.1 vs prior 47.7.”
  - Robotics (ALFRED valid‑unseen):
    - > “Success Rate: 67.8 (72B) ≈ specialized ThinkBot 67.7.”
  - Vision‑Language Navigation:
    - > “R2R SR: 51.7 (72B), behind specialized SoTA 79.0; REVERIE SR: 31.0 vs specialized 61.0.”

- Referring expression grounding (Table 6)
  - 72B achieves top‑tier generalist scores, e.g.,
    - > “RefCOCO test‑A: 95.3; test‑B: 90.7; RefCOCO+ test‑A: 93.8; test‑B: 85.6; RefCOCOg test: 90.4.”

- Ablations and robustness
  - Dynamic resolution vs fixed tokens (Table 7):
    - For 7B, dynamic tokens average 1,924 per image and deliver the best or tied results across InfoVQA (75.89), RealWorldQA (70.07), OCRBench (866), and MMMU (53.44), while using fewer tokens than some fixed‑high settings (e.g., 3,136 tokens).
    - Upscaling very small images helps perception tasks up to a point; too much upscaling harms OCRBench (Figure 4), likely due to distribution shift (Section 3.3.1).
  - M‑RoPE vs 1D‑RoPE (Table 8; Figure 5):
    - Image tasks: small but consistent gains (e.g., MathVista 43.4 vs 39.2; MMBench 60.6 vs 58.6).
    - Video tasks: clearer gains (PerceptionTest 47.4 vs 46.6; NextQA 46.0 vs 43.9; STAR 57.9 vs 55.5).
    - Length extrapolation: strong robustness up to 80K tokens at inference despite training at 16K (Figure 5).
  - Scaling (Figure 6):
    - Performance rises with parameters and training tokens; math improves most with scale, while OCR‑like tasks are strong even for smaller models.

- Are claims supported?
  - Evidence aligns with the core claims:
    - Dynamic resolution is both efficient and strong (Table 7).
    - M‑RoPE improves video and length generalization (Table 8; Figure 5).
    - The 72B model competes with closed models on many benchmarks and excels at document OCR and long‑video tasks (Tables 2 and 4).
  - Mixed/conditional results:
    - On MMMU (college‑level reasoning), 72B at 64.5 trails GPT‑4o (69.1) (Table 2), indicating remaining reasoning gaps.

## 6. Limitations and Trade-offs
- Token budget and compression
  - The 2×2 token compression (Section 2.1) reduces visual tokens by 4×, improving efficiency; it could also discard micro‑details in very dense images. Strong OCR numbers suggest careful tuning, but extreme microtext may still be vulnerable.

- Sequence length constraints
  - Training caps video tokens at 16,384 (Section 2.1). M‑RoPE extends inference length to 80K (Figure 5), yet the model has not been trained at that length, which could still affect reliability on the longest sequences.

- Video evaluation limits
  - Video‑MME evaluation extracts at most 768 frames per video (Section 3.2.6), which may under‑utilize information in hour‑long videos.

- Data and compute intensity
  - Training uses 1.4T tokens with substantial infrastructure (Section 2.3). Reproducing results requires large‑scale compute, storage, and engineering (parallelism, caching/decoding).

- Benchmark coverage and generalization
  - Some agent evaluations rely on internally constructed datasets (Section 3.2.7, Function Calling), which may limit external reproducibility.
  - Navigation (R2R/REVERIE) lags behind specialized models (Table 5), reflecting an open challenge in constructing and maintaining accurate 3D mental maps from sparse visual observations.

- Knowledge freshness and languages
  - Knowledge cutoff is June 2023 (Section 2.2). Internal multilingual OCR shows Arabic remains weaker than GPT‑4o (Table 3).

## 7. Implications and Future Directions
- How this changes the landscape
  - Demonstrates that variable‑token visual processing at any resolution is practical and beneficial. Future LVLMs can allocate compute where detail truly exists rather than forcing one size for all inputs.
  - Establishes a simple, general positional scheme (M‑RoPE) for text–image–video that scales to long sequences without architectural contortions.

- Follow‑up research enabled
  - Token‑budget scheduling: dynamic policies that allocate tokens adaptively across regions (e.g., more tokens on text or small objects; fewer on sky/blank areas).
  - Longer and streaming video: with M‑RoPE’s extrapolation, extend training to longer contexts, add memory modules, or retrieval‑augmented video summarization.
  - Better 3D reasoning for navigation: integrate explicit mapping modules or spatial memory to close the gap with specialized VLN systems (Table 5).
  - Fine‑grained OCR and diagram reasoning: combine dynamic resolution with content‑aware token compression (learned pooling or super‑resolution blocks) to push micro‑text fidelity further.
  - Tool‑use and agents: leverage strong OCR and grounding to build robust GUI agents, robotic perception pipelines, and code‑interpreter workflows (Section 2.2.1; Table 5; Appendix A.4).

- Practical applications
  - Document understanding at enterprise scale (contracts, invoices, forms) with multilingual OCR (Tables 2–3).
  - Long‑video Q&A, surveillance triage, lecture or meeting analysis (Table 4; Section 3.2.6).
  - Device automation and UI agents for accessibility, testing, and RPA (Table 5; Appendix A.4).
  - Robotics manipulation with visual grounding (Table 5, ALFRED).

In sum, Qwen2‑VL’s dynamic resolution and M‑RoPE are the pivotal mechanisms that make high‑fidelity, token‑efficient multimodal perception workable at scale. The experimental evidence shows strong payoffs in OCR, grounding, and video, while also clarifying where generalist LVLMs still trail specialized systems (e.g., 3D navigation and some expert‑level reasoning).
