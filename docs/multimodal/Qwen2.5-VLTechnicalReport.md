# Qwen2.5-VL Technical Report

**ArXiv:** [2502.13923](https://arxiv.org/abs/2502.13923)

## 🎯 Pitch

Qwen2.5-VL introduces a next-generation vision-language model series that pioneers fine-grained spatial perception, robust document parsing, and long-form video comprehension—delivering precise object localization, advanced document understanding, and second-level event localization in hours-long videos. By innovating with native dynamic resolution processing, efficient window attention, and absolute time-aligned multimodal embeddings, Qwen2.5-VL matches or outperforms leading closed-source models like GPT-4o and Claude 3.5 across key benchmarks. This work sets a new standard for real-world multimodal AI agents, enabling practical applications in business automation, device control, and video analytics at both large and resource-constrained scales.

---

## 1. Executive Summary (2-3 sentences)
Qwen2.5-VL is a large vision–language model family that focuses on fine-grained perception (document parsing, object grounding, counting) and long-form video understanding while keeping strong pure-text skills. It achieves this by redesigning the vision encoder for native dynamic resolution with efficient window attention, aligning multimodal positional embeddings to absolute time for videos, and training on a large, carefully curated corpus (4.1T tokens), yielding state-of-the-art results especially in document/OCR and temporal grounding (e.g., Charades-STA mIoU 50.9) with competitive or better performance than GPT-4o/Claude 3.5 on many benchmarks (Tables 3, 5, 8, 9).

## 2. Context and Motivation
- Problem addressed:
  - LVLMs (large vision–language models) struggle with:
    - Fine-grained spatial perception (precise bounding boxes/points, counting).
    - Robust document parsing across diverse layouts (tables, charts, formulas, handwriting).
    - Long and time-sensitive video understanding (hours-long videos, second-level event localization).
    - Efficiency when processing native-resolution images/videos (computational cost grows quickly if everything is globally attended).
  - Qwen2.5-VL targets all four: computation, fine-grained perception, temporal grounding, and consistency across sequence lengths (Section 1; Section 2).
- Why it matters:
  - Real-world impact:
    - Accurate document conversion and data extraction (invoices, forms, charts) is a ubiquitous business need (Section 2.2.1, “Document Omni-Parsing Data”).
    - Spatially grounded agents for computers/phones need precise object localization and UI element grounding (Section 2.2.1, “Agent Data”; Table 9).
    - Long video comprehension enables search, safety monitoring, sports analytics, and education (Section 2.1.2; Table 8).
  - Theoretical significance:
    - Native dynamic resolution and absolute-time positional encoding provide a more faithful representation of space and time than conventional normalized coordinates and frame-index-only time IDs (Section 2.1.2–2.1.3).
- Prior approaches and gaps:
  - Standard LVLM recipe (visual encoder + projector + LLM) is well known, but most systems either:
    - Downscale/normalize inputs (losing scale fidelity), or
    - Use global attention everywhere (quadratic cost), or
    - Treat time as frame indices rather than real timestamps (Section 1; 2.1.1–2.1.3).
  - Document parsing often relies on separate specialized models for layouts, charts, OCR, etc., instead of a unified representation (Section 2.2.1 “Document Omni-Parsing Data”).
- Positioning:
  - Qwen2.5-VL offers:
    - A redesigned ViT with window attention plus a small set of global-attention layers to keep compute manageable while preserving native resolution (Section 2.1.1; Table 1).
    - MRoPE (Multimodal Rotary Position Embedding) aligned to absolute time for videos, enabling FPS-agnostic temporal reasoning (Section 2.1.3; Figure 1).
    - A unified document HTML target format that encodes layout and coordinates, training a single model to parse everything (Section 2.2.1, “QwenVL HTML Format”).

## 3. Technical Approach
Step-by-step overview of the system (Figure 1; Section 2):

- Overall architecture (Section 2.1):
  - `Vision Encoder`: a redesigned Vision Transformer (ViT) trained from scratch to operate at native image/video resolution with efficiency.
  - `Vision–Language Merger`: a simple two-layer MLP that compresses spatial tokens before the LLM, aligning visual features to the LLM embedding space.
  - `Large Language Model (LLM)`: Qwen2.5 LLM (72B/7B/3B) with modified positional encoding (MRoPE aligned to absolute time).
  - Configurations (Table 1):
    - ViT: hidden size 1280, 32 layers, 16 heads, patch size 14, window size 112, full-attention layers at indices {7, 15, 23, 31}.
    - Merger output sizes match the LLM hidden sizes: 8192 (72B), 3584 (7B), 2048 (3B).

- Efficient vision encoder for native resolution (Section 2.1.1):
  - Problem: global self-attention on high-res images/videos is quadratic in token count; token count grows with resolution.
  - Solution: `window attention`—attention is computed locally within windows so cost scales approximately linearly with the number of patches. Only four layers use full self-attention (global context) to enable cross-window information flow.
    - Implementation details:
      - `Window size 112×112` pixels, corresponding to `8×8` patch tokens since patch size is 14 (Table 1; Section 2.1.1).
      - Regions smaller than 112×112 are processed without padding (preserve native size).
      - Position encoding: `2D-RoPE` for spatial XY in images; RMSNorm and SwiGLU used to better align with LLM conventions (Section 2.1.1).
  - Video token reduction:
    - `3D patching`: two consecutive frames are grouped to reduce token count while preserving temporal continuity (Section 2.1.1).

- Native dynamic resolution and frame rate (Section 2.1.2):
  - `Native dynamic resolution`: the model consumes images and videos at their actual sizes; token sequence lengths vary accordingly.
  - Spatial outputs use `absolute coordinates` (pixel units) rather than normalized [0,1] values. This lets the model “feel” real scale and location, useful for precise grounding and counting.
  - `Dynamic FPS training`: during video training, FPS is sampled dynamically to expose the model to varied temporal densities.

- Multimodal Rotary Position Embedding aligned to absolute time (MRoPE; Section 2.1.3, Figure 1):
  - Background: Rotary Position Embeddings (RoPE) rotate feature dimensions based on position. Traditional 1D RoPE doesn’t naturally encode 2D space or real time.
  - MRoPE decomposes positional IDs into three components: `time`, `height`, and `width`.
  - The new piece: the `time` component is aligned to `absolute time`, i.e., IDs correspond to real timestamps rather than frame indices. This makes the gap between IDs proportional to elapsed time, so the model can infer event speeds and align events across different FPS without extra heads or textual timestamps.
  - For text, MRoPE collapses to standard 1D RoPE (Section 2.1.3).

- Vision–Language Merger (Section 2.1):
  - To reduce LLM compute on long visual sequences, the model:
    - Groups every four spatially adjacent patch features.
    - Concatenates them and passes through a two-layer MLP to match LLM embedding dimension.
  - Effect: roughly 4× reduction in visual tokens entering the LLM while preserving local spatial context.

- Training data and recipe (Section 2.2; Table 2):
  - Scale and phases:
    - Total pretraining tokens: ~4.1T (expanded from Qwen2-VL’s 1.2T).
    - Three phases:
      - Visual pretraining: 1.5T tokens, seq len 8192, train ViT only.
      - Multimodal pretraining: 2.0T tokens, seq len 8192, train ViT+LLM.
      - Long-context pretraining: 0.6T tokens, seq len 32768, train ViT+LLM.
  - Data composition (Section 2.2.1):
    - Interleaved image–text; OCR; document omni-parsing; grounding with `absolute coordinates`; video (with dynamic FPS and time annotations in seconds and h:m:s:frame); agent interaction data for GUIs.
    - Interleaved data are scored in four dimensions: text quality, image–text relevance, complementarity, and information density balance.
  - Document omni-parsing supervision:
    - A unified `QwenVL HTML` format encodes paragraphs, tables, charts, formulas, images, music sheets, chemical formulas, etc., with bounding boxes via `data-bbox="x1 y1 x2 y2"` (Section 2.2.1 “Document Omni-Parsing Data”).
  - Grounding data:
    - Uses absolute coordinates; 10,000+ categories for open-vocabulary detection; synthetic and public data for bounding-box and point grounding (Section 2.2.1).
  - Training efficiency:
    - Dynamic packing balances LLM compute by grouping samples into similar sequence lengths (8,192 for phases 1–2; 32,768 for phase 3) to keep GPU load steady (Section 2.2.2).

- Post-training alignment (Section 2.3):
  - Two stages with ViT frozen:
    - `Supervised Fine-Tuning (SFT)` in ChatML format (structured dialogue turns with visual embeddings) on ~2M entries, 50% pure text and 50% multimodal (note: multimodal consumes more tokens).
    - `Direct Preference Optimization (DPO)` on image–text and text-only preference pairs (each sample seen once).
  - Data quality pipeline (Section 2.3.2–2.3.3):
    - Domain-specific categorization (Qwen2-VL-Instag), then domain-tailored rule-based and model-based filtering.
    - `Rejection sampling for reasoning`: generate chain-of-thought (CoT) with intermediate Qwen2.5-VL, keep only answers matching ground truth, and remove code-switching/repetitive/overlong responses; add checks that visual information is actually used in reasoning.

- Inference constraints for videos (Section 3.3.4):
  - Max frames per video: 768; max video tokens: 24,576.

Key implementation choices and why:
- Window attention with a few global layers: near-linear scaling at native resolution while retaining some global context (Section 2.1.1; Table 1).
- Absolute coordinates and absolute-time MRoPE: preserve true spatial scale and temporal dynamics; avoid task-specific heads or coordinate normalization that can hide global scale (Sections 2.1.2–2.1.3).
- MLP merger and frame grouping: reduce token load without discarding locality (Sections 2.1, 2.1.1).

Definitions of uncommon terms used above:
- `Window attention`: restrict attention computations to local windows instead of all tokens, lowering computational cost from quadratic to approximately linear in the number of tokens.
- `MRoPE`: Multimodal Rotary Position Embedding; extends RoPE to separate time/height/width and, here, aligns the time component to real timestamps.
- `Absolute coordinates`: use original pixel coordinates of the image (e.g., x1,y1,x2,y2 in pixels) rather than normalized [0,1] values relative to image width/height.

## 4. Key Insights and Innovations
- Absolute-time MRoPE for temporal grounding (Section 2.1.3; Figure 1)
  - What’s new: The temporal positional IDs correspond to real timestamps, not just frame indices, making intervals meaningful across different FPS settings.
  - Why it matters: Enables “second-level event localization” without extra heads or text timestamps, and underpins strong results on temporal tasks (e.g., Charades-STA mIoU 50.9; Table 8).

- Native dynamic resolution with absolute coordinates (Sections 2.1.2, 2.2.1)
  - What’s new: Train and infer at native image sizes and output absolute pixel coordinates; no forced normalization.
  - Why it matters: Captures true object scales and locations; improves grounding/counting fidelity and document parsing where layout scale is crucial. The paper backs this with strong grounding and counting results (Tables 6–7) and document/OCR results (Table 5).

- Efficient ViT with window attention + sparse global layers (Section 2.1.1; Table 1)
  - What’s new: Most layers use window attention; only 4 layers use global attention ({7, 15, 23, 31}); include LLM-style components (RMSNorm, SwiGLU).
  - Why it matters: Drastically reduces compute while preserving long-range communication via sparse global layers and maintaining native input resolution.

- Unified document supervision via HTML with layout boxes (Section 2.2.1, “QwenVL HTML Format”)
  - What’s new: A single target format encodes paragraphs, tables, charts, formulas, etc., with bounding boxes; integrates layout and content supervision for “omni-parsing.”
  - Why it matters: Avoids training many specialized models; supports end-to-end document parsing and transformation. Strong OCR/document results in Table 5 corroborate this design choice.

- Pragmatic token compression for the LLM (Section 2.1)
  - What’s new: Group 4 neighboring patches and project with a two-layer MLP to LLM embedding size.
  - Why it matters: Keeps LLM compute manageable with minimal architectural overhead, making native-resolution processing feasible.

These are more than incremental tweaks: absolute-time MRoPE and native dynamic resolution with absolute coordinates change how temporal and spatial information are represented; the ViT redesign makes such representations computationally viable at scale.

## 5. Experimental Analysis
- Evaluation setup and baselines:
  - Models compared: Qwen2.5-VL-72B (flagship), 7B, 3B vs GPT-4o, Claude 3.5 Sonnet, InternVL2.5 (78B), Gemini 1.5 Pro, Molmo (varies by task).
  - Domains: college-level reasoning, math, general VQA, OCR/document parsing, spatial grounding/counting, video understanding/grounding, GUI agents.
  - Constraints: For video, capped at 768 frames and 24,576 tokens (Section 3.3.4).

- Headline results (numbers cited directly from tables):
  - College-level problems (Table 3)
    - MMMU val: 70.2 (Qwen-72B), matching the previous open-source SOTA 70.1 and close to GPT-4o 69.1; MMMU-Pro overall: 51.1 vs GPT-4o 51.9.
  - Math (Table 3)
    - MathVista mini/full: 74.8 (Qwen-72B), beating prior open-source 72.3 and GPT-4o 63.8; MATH-Vision full: 38.1 (Qwen-72B) vs GPT-4o 30.4; MathVerse mini: 57.6 (Qwen-72B).
  - General VQA (Table 3)
    - MMBench-EN test: 88.6 (Qwen-72B), slightly above prior best 88.3; MMStar: 70.8 (Qwen-72B); MME-RealWorld (EN): 63.2 (Qwen-72B), state-of-the-art in that table row.
  - OCR/document/diagram (Table 5)
    - OCRBench: 885 (Qwen-72B) vs 854 (InternVL2.5) and 736 (GPT-4o).
    - OCRBench_v2 (en/zh): 61.5/63.7 (Qwen-72B), +9.6 (en) and +20.6 (zh) over Gemini 1.5 Pro’s 51.9/43.1.
    - DocVQA test: 96.4 (Qwen-72B), top among listed models.
    - SEED-Bench-2-Plus (text-rich scenarios): 73.0 (Qwen-72B), top among listed models.
    - ChartQA: 89.5 (Qwen-72B), competitive but below Claude’s 90.8.
    - AI2D (diagrams): 88.7 (Qwen-72B), close to InternVL2.5’s 89.1 and Gemini’s 88.4.
    - CharXiv (realistic chart QA): 49.7/87.4 (RQ/DQ) for Qwen-72B; here RQ trails Claude’s 60.2 while DQ is strong.
  - Spatial grounding & counting (Tables 6–7)
    - RefCOCO family: Qwen-72B is very competitive but slightly below InternVL2.5-78B on some splits (e.g., RefCOCO testA 94.6 vs 95.6).
    - Open-vocabulary detection ODinW-13: 43.1 mAP (Qwen-72B) vs 31.7 (InternVL2.5-78B).
    - Point grounding: 67.5 (Qwen-72B) vs 69.2 (Molmo-72B).
    - CountBench: 93.6 (Qwen-72B), best among listed models.
  - Video understanding & grounding (Table 8)
    - MVBench: 70.4 (Qwen-72B) vs 64.6 (GPT-4o).
    - LVBench (very long video QA): 47.3 (Qwen-72B) vs 30.8 (GPT-4o).
    - MLVU mean: 74.6 (Qwen-72B) vs 64.6 (GPT-4o).
    - TempCompass Avg: 74.8 (Qwen-72B) vs 73.8 (GPT-4o).
    - Charades-STA (temporal grounding): mIoU 50.9 (Qwen-72B) vs 35.7 (GPT-4o).
    - Mixed elsewhere: Video-MMMU 60.2 (Qwen-72B) vs 61.2 (GPT-4o); LongVideoBench val 60.7 (Qwen-72B) vs 66.7 (GPT-4o).
  - GUI Agents (Table 9)
    - Grounding: ScreenSpot Pro 43.6 (Qwen-72B), far above Aguvis-72B 23.6 and Qwen2-VL-72B 1.6.
    - Offline control: Android Control HighEM 67.36 and LowEM 93.7 (Qwen-72B), both top among listed models.
    - Online: AndroidWorld SR 35% (Qwen-72B) vs 34.5% (GPT-4o, with Set-of-Mark aid); MobileMiniWob++ SR 68% (Qwen-72B) vs 61% (GPT-4o). On OSWorld, Qwen-72B reaches 8.83, below Claude (14.90).

- Do the experiments support the claims?
  - Document/OCR: Strong evidence of leadership (OCRBench, OCRBench_v2, DocVQA, SEED-Bench-2-Plus; Table 5).
  - Temporal grounding: Clear wins on Charades-STA, LVBench, MLVU, TempCompass (Table 8) are consistent with the absolute-time MRoPE motivation.
  - Fine-grained grounding and counting: Strong on ODinW and CountBench; RefCOCO is competitive but not uniformly best (Tables 6–7).
  - General VQA and math: Generally strong with several wins; competitive with top proprietary models on many tasks (Table 3).
  - Agent capabilities: Very strong grounding and offline control; competitive online performance, but not uniformly best on OSWorld (Table 9).
  - Caveat: The paper does not provide ablation studies isolating the contribution of window attention, MLP merger, or absolute-time MRoPE (no explicit ablation tables in Sections 2–3). Thus, causality is inferred indirectly through outcome strengths that align with the proposed innovations.

- Robustness/failure cases:
  - Mixed results on some video understanding datasets (e.g., LongVideoBench), chart benchmarks (CharXiv RQ), and OSWorld suggest remaining gaps in certain reasoning styles, chart question types, or real-computer task generalization.

> Table 5 (OCRBench_v2 en/zh): Qwen2.5-VL-72B = 61.5/63.7, Gemini 1.5 Pro = 51.9/43.1.
>
> Table 8 (Charades-STA mIoU): Qwen2.5-VL-72B = 50.9, GPT-4o = 35.7.
>
> Table 9 (ScreenSpot Pro): Qwen2.5-VL-72B = 43.6 vs 23.6 (Aguvis-72B) and 1.6 (Qwen2-VL-72B).

## 6. Limitations and Trade-offs
- Compute and data intensity:
  - Pretraining on ~4.1T tokens and training a 72B LLM with long-context sequences (up to 32,768 tokens) demand substantial compute and memory (Table 2).
  - Video inference limits (768 frames, 24,576 tokens) can still be heavy for long videos (Section 3.3.4).

- Window attention trade-off:
  - Local windows reduce compute but can limit long-range spatial dependencies; only four layers use global attention to mitigate this (Table 1). Some tasks that require holistic global reasoning might still be constrained.

- Token compression trade-off:
  - Grouping four patches and projecting with an MLP reduces detail entering the LLM; while results are strong, the paper does not show ablations quantifying fidelity vs. compression.

- Absolute coordinates and native resolution:
  - Using absolute pixel coordinates and native resolution improves scale fidelity but may complicate batching and increase variability in sequence lengths; training uses dynamic packing to manage this (Section 2.2.2).

- Mixed coverage across domains:
  - Charts: CharXiv RQ lags behind Claude (Table 5), suggesting reasoning over realistic chart questions may still be challenging.
  - Video: Mixed performance (e.g., LongVideoBench lower than GPT-4o; Table 8).
  - Agents: OSWorld score (8.83) trails Claude (14.90), indicating room for improvement in open-ended PC tasks (Table 9).

- Limited ablations:
  - No explicit component-wise ablations for absolute-time MRoPE, window attention placement, or frame grouping leave open questions about the individual effect sizes of each design choice.

## 7. Implications and Future Directions
- How this work changes the landscape:
  - Demonstrates that native dynamic resolution plus absolute-time positional encoding are practical at scale and translate into concrete gains in document parsing and temporal grounding—two historically difficult areas for LVLMs.
  - Bridges perception and agency by combining precise grounding with decision-making, enabling more autonomous device operation (Table 9).

- Follow-up research enabled/suggested:
  - Component ablations and principled studies:
    - Quantify the contribution of absolute-time MRoPE vs. frame-index time.
    - Explore optimal placement/number of global-attention layers under compute budgets.
    - Study the fidelity–efficiency curve of the 4-patch MLP merger and alternative pooling schemes.
  - Extending temporal modeling:
    - Continuous-time encodings, variable-speed events, multi-camera synchronization.
    - Event-centric memory and retrieval that leverage absolute-time MRoPE.
  - Document intelligence:
    - Expand the HTML target to richer semantics (e.g., logical relations between tables/figures), multi-page long documents, and layout-aware editing.
  - Agents:
    - Tighten grounding→action chains with verification loops; incorporate explicit UI state models; evaluate on more realistic enterprise workflows and security-constrained environments.

- Practical applications:
  - Enterprise document automation: invoices, forms, compliance reports, scientific papers with formulas/diagrams (Section 2.2.1; Table 5).
  - Visual analytics: chart/diagram understanding, counting and open-vocabulary detection (Tables 5–7).
  - Video analytics: surveillance/event detection, sports highlights with precise timestamps, educational video indexing (Table 8).
  - Device agents: automated app workflows on phones/PCs, testing and RPA-style tasks (Table 9).

> Figure 1 illustrates the end-to-end pipeline with native-resolution ViT + MRoPE aligned to absolute time + LLM decoder; Table 1 details ViT and LLM configs; Table 2 outlines the three-stage training with sequence lengths; Tables 3, 5, 6–9 report broad SOTA-competitive results, especially in document/OCR and temporal grounding.
