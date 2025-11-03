# Qwen2.5-VL: Enhancing Vision‑Language Model's Perception of the World at Any Resolution

**ArXiv:** [2502.13923](https://arxiv.org/abs/2502.13923)
**Authors:** Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, Humen Zhong, Yuanzhi Zhu, Mingkun Yang, Zhaohai Li, Jianqiang Wan, Pengfei Wang, Wei Ding, Zheren Fu, Yiheng Xu, Jiabo Ye, Xi Zhang, Tianbao Xie, Zesen Cheng, Hang Zhang, Zhibo Yang, Haiyang Xu, Junyang Lin
**Institutions:** Qwen Team (Alibaba)

## 🎯 Pitch

Qwen2.5‑VL revolutionizes vision-language models by integrating windowed attention in a Vision Transformer, a compact vision-language merger, and a novel absolute-time positional encoding. This innovation not only enhances processing efficiency for native-resolution and long-video content but also significantly advances applications in object localization, document parsing, and video understanding, paving the way for improved UI automation, robotics, and surveillance systems.

---

## 1. Executive Summary
Qwen2.5‑VL is a family of vision‑language models that re-engineers the visual stack and temporal encoding to handle native‑resolution images and long videos efficiently, while adding precise object localization and strong document parsing. It introduces windowed attention in a newly trained Vision Transformer, a compact vision‑language merger, and a positional encoding that aligns tokens to absolute time, yielding state‑of‑the‑art or competitive results across grounding, OCR/doc parsing, and long‑video understanding (see Figure 1 and Tables 3, 5, 6, 7, 8, 9).

## 2. Context and Motivation
- Problem addressed
  - LVLMs struggle with fine‑grained perception (precise localization, counting), computational blow‑ups at native resolution, and inconsistent performance when sequence length or frame rate varies; temporal reasoning is often tied to frame counts rather than real time (Introduction; Sections 2, 2.1.1–2.1.3).
- Why it matters
  - Real applications demand accurate object grounding (e.g., UI automation, robotics), robust document parsing (invoices, forms, charts), and reliable long‑video understanding (surveillance, meeting analysis) with second‑level timestamping (Abstract; “sparkling characteristics” bullets).
- Prior approaches and gaps
  - Standard LVLM design = visual encoder + projector + LLM (Introduction). Many models normalize coordinates and/or downsample inputs, losing scale fidelity; temporal position is usually tied to frame index, which fails to capture absolute timing across variable FPS (Sections 2.1.2–2.1.3).
  - Fine‑grained perception has often relied on specialized detectors (e.g., Grounding DINO, SAM) or auxiliary heads; many LVLMs lack native point grounding and fine‑grained spatial outputs in absolute image coordinates (Section 2.2.1 “Grounding Data with Absolute Position Coordinates”).
- Positioning of this work
  - Qwen2.5‑VL proposes a native‑resolution ViT with window attention for linear‑scaling compute, a multimodal rotary positional embedding aligned to absolute time for stable temporal reasoning across variable FPS, and a light, trainable vision‑language merger to compress tokens before the LLM (Sections 2.1–2.1.3).
  - It scales training to ~4.1T tokens with curated multimodal data, including an HTML‑based omni‑document format that unifies layout, OCR, charts, formulas, and images (Sections 2.2–2.2.1; “Document Omni‑Parsing Data”).

## 3. Technical Approach
Step‑by‑step architecture and training pipeline (Figure 1; Table 1; Sections 2.1–2.3.4):

- High‑level architecture
  - Components: a redesigned `Vision Encoder` (ViT), an `MLP‑based Vision‑Language Merger`, and a `Qwen2.5` LLM with a multimodal rotary positional embedding aligned to absolute time (`MRoPE‑Time`) (Section 2.1).
- Vision Encoder (fast, native resolution; Section 2.1.1)
  - Input handling
    - Images are resized only so height/width become multiples of 28, preserving aspect ratio (avoids heavy normalization).
    - Split into 14×14 patches (stride 14) to produce patch tokens.
    - For videos, “3D patching”: two consecutive frames are grouped to reduce token count while retaining temporal continuity.
  - Positional encoding
    - Uses 2D‑RoPE for spatial relations; for videos, becomes 3D with temporal IDs (Section 2.1.1).
  - Windowed attention for efficiency
    - Most layers use local windows (max 112×112 pixels = 8×8 patches) so attention cost scales roughly linearly in number of patches; four layers keep full self‑attention to pass global information (full‑attention layers at indices {7, 15, 23, 31}, Table 1).
    - This design maintains native resolution without padding small regions (Section 2.1.1).
  - Alignment with LLM stack
    - Replaces typical ViT norms/activations with `RMSNorm` and `SwiGLU` to match LLM design choices and efficiency (Section 2.1.1).
  - Training
    - ViT is trained from scratch in stages (CLIP‑style pretraining, alignment, end‑to‑end finetuning) with dynamic sampling over native resolutions (Section 2.1.1).
- MLP‑based Vision‑Language Merger (Section 2.1)
  - Problem: Visual feature sequences can be long and costly for the LLM.
  - Solution: Group each 2×2 spatial neighborhood’s patch features; concatenate and pass through a 2‑layer MLP to project into the LLM’s embedding size (compresses tokens while preserving local detail).
  - Why an MLP: A simple trainable projection avoids heavy cross‑modal attention before the LLM and allows dynamic compression that scales with image resolution.
- LLM and multimodal positional encoding (Sections 2.1, 2.1.3)
  - Base LLM: The `Qwen2.5` family (e.g., 7B and 72B; Table 1), initialized from pre‑trained Qwen2.5.
  - `MRoPE‑Time` (Multimodal Rotary Position Embedding aligned to absolute time)
    - Background: MRoPE decomposes positions into temporal, height, and width components; text uses identical IDs (acts like 1D RoPE). For images: temporal ID is constant; height/width reflect spatial location. For videos: temporal ID increases with frames (Section 2.1.3).
    - Innovation: Instead of indexing by frame count, temporal IDs align directly to absolute timestamps. The time interval between IDs reflects real time, allowing the model to learn tempo and localize events at second resolution across variable FPS (Sections 2.1.2–2.1.3; Figure 1).
- Dynamic resolution and FPS (Section 2.1.2)
  - Spatial: Token sequence length directly follows native image size; spatial outputs (e.g., bounding boxes, points) are expressed in absolute pixel coordinates of the input image (no normalization), preserving scale awareness.
  - Temporal: Training uses dynamic FPS sampling, and temporal IDs are tied to absolute time, so videos with different frame rates map consistently in time.
- Training data and recipe (Sections 2.2–2.2.2; Table 2)
  - Scale and coverage: Scales pretraining tokens from ~1.2T to ~4.1T, mixing image captions, interleaved multimodal streams, OCR, grounding (boxes and points), document parsing, video captioning/grounding, agent interaction data, and pure text.
  - Data quality control
    - Interleaved data are cleaned with a 4‑stage scoring pipeline focusing on text quality, image–text relevance and complementarity, and information balance.
    - Grounding uses absolute pixel coordinates; datasets include >10k categories and synthetic “non‑existent” categories to stress open‑vocabulary detection (Section 2.2.1).
    - Omni‑document data: a unified `QwenVL HTML` format encodes layout boxes and modality content (tables, charts, formulas, music sheets, chemical formulas) with bounding boxes inside HTML tags (Section 2.2.1).
  - Three training phases (Table 2)
    - Phase 1 (1.5T tokens, seq len 8192): Train ViT alone on vision‑centric data (captioning, knowledge, OCR) to align with the LLM interface.
    - Phase 2 (2T tokens, 8192): Unfreeze all parameters; train on interleaved data, VQA, video, math, agent tasks, and pure text.
    - Phase 3 (0.6T tokens, 32768): Long‑context training with long videos, long agents, and long documents.
  - Efficiency techniques
    - Window attention reduces ViT costs; dynamic “packing” balances LLM sequence lengths per GPU to equalize load (Section 2.2.2).
- Post‑training alignment (Sections 2.3–2.3.4)
  - SFT (Supervised Fine‑Tuning): ~2M entries (50% text, 50% multimodal), using ChatML formatting for structured multimodal dialogue and careful domain coverage (e.g., OCR/Doc, Grounding, Video, Agent) with both single/multi‑turn and single/multi‑image contexts (Section 2.3.1).
  - Data filtering: Two‑stage pipeline—domain classification into 8 domains/30 subdomains, then domain‑tailored rule/model-based filtering to remove noise, truncation, or harmful/irrelevant entries; reward models score correctness, completeness, clarity, and visual grounding quality (Section 2.3.2).
  - Rejection sampling for reasoning: Build datasets with verified chain‑of‑thought outputs that match ground truth; further filter out code‑switching, over-long or repetitive outputs, and ensure visual evidence is used properly in intermediate steps (Section 2.3.3).
  - DPO (Direct Preference Optimization): Preference-based alignment with image‑text and pure text examples; ViT is frozen during both SFT and DPO (Section 2.3.4).

## 4. Key Insights and Innovations
- Windowed attention in a native‑resolution ViT (Section 2.1.1; Table 1)
  - What’s new: Most ViT layers attend within local windows (112×112 px), with only four global layers.
  - Why it matters: Reduces quadratic attention cost to near‑linear in tokens while preserving critical global routing through selected layers, enabling native‑resolution inference without aggressive downsampling.
  - Difference from prior work: Many LVLMs downsample or normalize inputs, losing scale cues; this design keeps native scale and controls compute.
- Absolute‑time `MRoPE‑Time` for videos (Sections 2.1.2–2.1.3; Figure 1)
  - What’s new: Temporal position IDs encode real time rather than frame count, so the same 3‑second event aligns regardless of FPS.
  - Why it matters: Enables robust timestamp grounding, tempo awareness, and second‑level localization in long videos with variable sampling rates, without extra heads or textual timestamps.
- Simple, effective `MLP‑based Vision‑Language Merger` (Section 2.1)
  - What’s new: A 2‑layer MLP compresses 2×2 patch neighborhoods into LLM‑sized embeddings before the LLM.
  - Why it matters: Cuts sequence length and LLM compute, while preserving local structure; avoids complex cross‑modal attention or heavy pooling schemes.
- Unified omni‑document representation and large‑scale grounding with absolute coordinates (Section 2.2.1 “Document Omni‑Parsing Data” and “Grounding Data…”)
  - What’s new: A standardized HTML format stores layout, tables, charts, formulas, images, with `data-bbox` attributes. Grounding data uses absolute pixel coordinates across multiple formats (JSON/XML/custom), and includes >10k categories plus synthetic “non‑existent” categories.
  - Why it matters: Trains a single model to parse diverse document types end‑to‑end and to ground objects precisely in absolute coordinates—key for UI agents and real‑world measurements.
- Dynamic FPS training and 3D patch grouping (Sections 2.1.1–2.1.2)
  - What’s new: During training, videos are sampled at varying FPS; two sequential frames are grouped at the patch level.
  - Why it matters: Improves robustness to frame‑rate variation and reduces token count without losing short‑range temporal signals—important for long video processing.

## 5. Experimental Analysis
- Evaluation setup and breadth
  - Benchmarks span general VQA, math, OCR/docs/charts, spatial grounding (boxes, points, counting), video understanding/grounding (short to hours), agents (mobile/desktop/web GUIs), and pure text tasks (Tables 3–9; Section 3).
  - Model sizes: `Qwen2.5‑VL‑3B`, `7B`, `72B` (Table 1). Results reported against strong baselines including GPT‑4o, Claude 3.5 Sonnet, Gemini 1.5/2.0, InternVL2.5.
- Headline results (selected)
  - General VQA (Table 3)
    - > “`MMBench‑EN` test: 88.6 (72B), slightly exceeding prior best 88.3; `MMStar`: 70.8 (72B)”—competitive at high‑level visual QA and multi‑image understanding (MuirBench: 70.7).
  - Math‑in‑vision (Table 3)
    - > “`MathVista`: 74.8 (72B), surpassing the previous open‑source SoTA 72.3; `MATH‑Vision`: 38.1; `MathVerse`: 57.6.”
  - OCR / Document / Charts (Table 5)
    - > “`TextVQA` val: 84.9 (72B); `DocVQA` test: 96.4 (72B); `OCRBench`: 885 (InternVL2.5) vs 864 (72B).”
    - > “`OCRBench_v2` (comprehensive): 61.5/63.7 en/zh (InternVL2.5) vs 56.3/57.2 (72B).” Results are strong overall; some mixed relative to best proprietary baselines depending on track.
    - > “`CC‑OCR`: 79.8 (72B), above GPT‑4o 66.9 and Claude 62.5.”
    - OmniDocBench edit distance (lower is better) shows competitive but not top numbers (e.g., `0.226/0.324` InternVL2.5 vs `0.275/0.324` GPT‑4o vs `0.308/0.398` 7B; Table 5).
  - Spatial grounding and counting (Tables 6–7)
    - > “RefCOCO/RefCOCO+/RefCOCOg: 72B reaches 92–95% on multiple splits; ODinW‑13 open‑vocab detection: 43.1 mAP (72B), surpassing most LVLMs, narrowing gap to specialist detectors.”
    - > “Point grounding: 67.5 (72B), near Molmo‑72B’s 69.2.”
    - > “`CountBench`: 93.6 (72B), higher than GPT‑4o 87.9 and Claude 89.7.”
  - Video understanding and grounding (Table 8)
    - > “`MVBench`: 70.4 (72B), above GPT‑4o 64.6; `LVBench` (long video): 47.3 (72B) vs GPT‑4o 30.8; `MLVU` M‑Avg: 74.6 (72B) vs GPT‑4o 64.6.”
    - > “`Charades‑STA` temporal localization: mIoU 50.9 (72B) vs GPT‑4o 35.7.” The setup caps frames at 768 and video tokens at 24,576 (Section 3.3.4).
    - Results indicate strong timestamp grounding and long‑video QA consistent with the absolute‑time MRoPE design.
  - Agents and GUI grounding (Table 9)
    - > “`ScreenSpot` (GUI element grounding): 87.1 (72B), competitive with Gemini 2.0’s 84.0; `ScreenSpot Pro`: 43.6 (72B), far above Aguvis‑72B 23.6 and `Qwen2‑VL‑72B` 1.6.”
    - > “Android Control HighEM/LowEM: 67.36/93.7 (72B), leading among reported baselines.”
    - > “Online: AndroidWorld SR 35% (72B) vs GPT‑4o 34.5% (SoM); MobileMiniWob++ 68% (72B) vs GPT‑4o 61%.” The model performs without Set‑of‑Mark (SoM) visual hints, while some baselines require SoM.
  - Pure text tasks (Table 4)
    - > “`LiveBench‑0831`: 57.0 (72B‑VL) vs Qwen2.5‑72B 52.3; `MMLU‑Pro`: 71.2; `HumanEval`: 87.8; `MultiPL‑E`: 79.5; `MATH`: 83.0.” The VL models retain strong language and coding capability.
- Do the experiments support the claims?
  - Fine‑grained perception: Yes—box/point grounding and counting show strong gains (Tables 6–7).
  - Document parsing: Strong across multiple OCR/doc tasks; not uniformly best on every benchmark, but the breadth (TextVQA, DocVQA, CC‑OCR, OCRBench variants, OmniDocBench) indicates robust generality (Table 5).
  - Long‑video and temporal grounding: Clear improvements where absolute time matters (e.g., Charades‑STA mIoU 50.9; LVBench and MLVU; Table 8).
  - Agentic functionality: Marked advances in GUI grounding and downstream device control (Table 9).
- Missing ablations or diagnostics
  - No explicit ablation quantifying the contribution of windowed attention vs. full attention, of the MLP merger vs. alternatives, or of absolute‑time MRoPE vs. frame‑indexed MRoPE.
  - Robustness to extreme resolutions and to window partition choices is not separately reported.
  - Training compute and wall‑clock efficiency gains are described conceptually, not benchmarked against strong open baselines.

## 6. Limitations and Trade-offs
- Compute and data scale
  - Training uses ~4.1T tokens and a redesigned ViT trained from scratch, which implies significant compute and data demands (Table 2). This limits reproducibility for smaller labs.
- Windowed attention trade‑offs
  - Local windows reduce cost but can restrict long‑range spatial interactions; four global layers mitigate but may not fully capture global patterns in edge cases (Section 2.1.1; Table 1).
- Token budget for videos
  - Despite absolute‑time encoding and dynamic FPS, experiments cap to ≤768 frames and ≤24,576 video tokens (Table 8), which may constrain truly “hours‑long” detailed analysis without careful frame sampling.
- Absolute coordinate dependence
  - Using absolute pixels improves scale fidelity but can be brittle when downstream systems rescale or crop inputs unpredictably; precision relies on consistent handling of native resolution (Section 2.1.2).
- Limited transparency on error modes
  - The report emphasizes wins; systematic failure analyses (e.g., complex diagrams that require multi‑step textual reasoning plus long‑range spatial context) are not detailed.
- Post‑training choices
  - ViT is frozen during SFT/DPO (Section 2.3.4). While efficient, this can limit last‑mile adaptation in domains where vision features need slight task‑specific adjustment.

## 7. Implications and Future Directions
- Landscape impact
  - Demonstrates that careful architectural surgery on the visual stack—windowed attention with selective global layers, absolute‑time positional encoding, and a minimal merger—can unlock native‑resolution perception and long‑video grounding in a generalist LVLM (Figure 1; Sections 2.1–2.1.3).
  - Establishes a practical path to unify doc parsing, grounding, video understanding, and agent control within one model family (Tables 5–9).
- Follow‑up research enabled
  - Ablations on:
    - Window size vs. number of global layers vs. accuracy/latency trade‑offs.
    - Alternative mergers (e.g., cross‑modal adapters, learned pooling, token pruning) vs. the 2‑layer MLP.
    - Absolute‑time MRoPE vs. hybrid timestamp tokens or learned time bases across diverse FPS distributions.
  - Scaling long‑video without a hard frame cap, possibly via hierarchical memory, retrieval‑augmented video tokens, or compressive streaming.
  - Adaptive coordinate systems that preserve absolute fidelity yet remain robust to unknown rescaling/cropping in deployment.
  - Extending the `QwenVL HTML` document format to native PDF/Office converters and round‑trip editing; richer chart/diagram semantics (e.g., program extraction).
- Practical applications
  - Enterprise document workflows (end‑to‑end conversion, verification, and extraction with layout grounding).
  - Autonomous UI assistants on desktop/mobile/web (reliable element grounding, step‑wise reasoning, and action planning—Table 9).
  - Video analytics for surveillance, sports, meetings, and industrial inspection with second‑level event localization and multi‑format timestamp outputs (Section 2.2.1 “Video Data”; Table 8).
  - Open‑vocabulary visual search and counting in retail, logistics, and quality control (Tables 6–7).

> Core takeaway: The combination of native‑resolution visual processing, absolute‑time positional encoding, and a compact vision‑language merger—backed by large‑scale, carefully curated multimodal data—yields a single LVLM that is competitive across fine‑grained grounding, document understanding, long‑video reasoning, and agentic interaction (Figure 1; Tables 3–9).
