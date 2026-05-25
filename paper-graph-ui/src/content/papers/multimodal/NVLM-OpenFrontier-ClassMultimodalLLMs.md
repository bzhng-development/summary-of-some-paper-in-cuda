# NVLM: Open Frontier-Class Multimodal LLMs

**ArXiv:** [2409.11402](https://arxiv.org/abs/2409.11402)

## 🎯 Pitch

NVLM 1.0 introduces a family of open, frontier-class multimodal large language models (LLMs) that achieve state-of-the-art results on vision-language tasks while preserving—or even improving—text-only capabilities, breaking the prevalent trade-off seen in prior open models. Through a unified architectural comparison (decoder-only, cross-attention, hybrid), a novel high-resolution 'tile-tagging' approach, and a carefully curated training recipe, NVLM demonstrates that open models can match proprietary leaders like GPT-4o and Gemini Pro 1.5 in both vision and text tasks. This work empowers the research community with a transparent, high-performing foundation for seamless, production-grade multimodality, setting new standards for open-access LLM development.

---

## 1. Executive Summary (2-3 sentences)
NVLM 1.0 introduces a family of three multimodal large language models—`NVLM-D` (decoder-only), `NVLM-X` (cross-attention), and `NVLM-H` (hybrid)—that deliver frontier-level vision-language performance while preserving or improving text-only capabilities. The work combines a carefully designed high‑resolution visual input pathway with a novel “tile tagging” mechanism and a curated training recipe; the flagship 72B models rival leading proprietary systems and, unusually for open models, improve math and coding accuracy over their text backbones after multimodal training (Table 7, Table 8).

## 2. Context and Motivation
- Problem addressed
  - Open multimodal LLMs often force a trade-off: strong vision-language results at the cost of substantially degraded text-only performance after multimodal finetuning (Table 8), unlike proprietary systems that perform well on both (Introduction; §6.4).
  - Architectural comparisons in the literature are confounded by differing backbones, data, and recipes; there is no apples-to-apples evaluation of decoder-only vs. cross-attention models (§1; §3.2).
  - Handling high-resolution images (crucial for OCR and documents) boosts OCR but can harm reasoning accuracy if naively implemented (§3.3; §4.1–4.2).

- Why this matters
  - Practical: “Production-grade multimodality” requires a single model that users can seamlessly apply to both text and multimodal tasks with strong performance on each (§1).
  - Scientific: Clarifies architectural trade-offs (compute, sequence length, reasoning) and data/recipe components that actually preserve text competence in multimodal training.

- Prior approaches and their gaps
  - Decoder-only (e.g., LLaVA, InternVL): simple and unified token processing, but long sequences for high-res images and substantial text performance degradation during multimodal SFT (§3.2; Table 8).
  - Cross-attention (e.g., Flamingo, Llama 3‑V): efficient with high-resolution inputs and often freeze the LLM to preserve text skills, but freezing can limit adaptation to new multimodal tasks (§3.2; §6.5).
  - Dynamic high-resolution (DHR) tiling: helps OCR but can reduce reasoning scores when tile tokens are simply concatenated without structure (§3.3; Table 1 “DHR + No tag”).

- How this paper positions itself
  - Provides three architectures built on the same backbones, data, and training procedure, enabling a fair comparison (§4; Figure 3).
  - Introduces 1‑D `tile tags` to inject tiling structure into the model’s sequence, improving OCR and reasoning simultaneously (Table 1, Table 2).
  - Presents a training blend that integrates high-quality text-only SFT into multimodal SFT, yielding improved text accuracy over the backbone (Table 8).

## 3. Technical Approach
Step-by-step overview of the system design and training recipe.

- Shared visual pathway with dynamic high resolution (§4.1; Figure 3; Figure 4)
  - Define “dynamic high-resolution (DHR)”: instead of resizing every image to a fixed small resolution, large images are divided into a small number of tiles chosen to match aspect ratio and resolution, so the fixed-resolution vision encoder can process detailed regions efficiently.
  - Implementation details:
    - Use a frozen `InternViT-6B-448px-V1-5` vision encoder that outputs 1,024 tokens per 448×448 tile (§4.1).
    - Include a global “thumbnail” tile (downscaled whole image) plus up to 6 regular tiles chosen from a set of predefined aspect ratios; each tile yields 1,024 tokens (§4.1; Figure 4).
    - Downsample each tile’s 1,024 tokens to 256 using a “pixel shuffle” operation that merges neighboring tokens (§4.1), reducing the load on the LLM while retaining spatial locality.

- Tile tagging to expose image layout to the LLM (§4.2; Table 1; §4.3; Table 2)
  - Define “tile tag”: a small text token (e.g., `<tile_1>`) inserted into the LLM input to mark the start and identity of each tile’s tokens.
  - Rationale: Without explicit tags, a decoder-only model receives a long flat sequence of image tokens and must infer tile boundaries; this harms reasoning and OCR. Tile tags explicitly encode structure.
  - Variants tested (Table 1): no tag, 1‑D tags (`<tile_k>`), 2‑D grid tags, and 2‑D bounding-box tags. The simple 1‑D tag generalizes best across benchmarks.

- Three model architectures (Figure 3; §4.2–§4.4)
  - `NVLM-D` (decoder-only; §4.2):
    - Project each tile’s 256 visual tokens into the LLM’s embedding space via a 2‑layer MLP (“projector”) and concatenate with text tokens. The LLM then processes everything with self-attention.
    - Use 1‑D tile tags in the text stream to delineate tiles, followed by the tile’s 256 tokens.
    - Training: Pretraining freezes the LLM and vision encoder while training the projector. In SFT, both the LLM and projector are trained; a curated text-only SFT blend is mixed in to preserve text skills (§4.5).
  - `NVLM-X` (cross-attention; §4.3):
    - Do not unroll image tokens into the decoder sequence. Instead, insert gated cross-attention (“X-attention”) layers every 6–8 LLM layers (10 total) that attend directly to image tokens (after a 1‑layer MLP to match dimensions) (§4.5).
    - Omit the Perceiver resampler used in Flamingo to avoid spatial mixing that hurts OCR; an overfitting experiment shows the resampler prevents fitting document OCR data (Appendix C; Figure 9).
    - Use text `tile tags` in the decoder, and configure cross-attention masks so each tag attends only to the corresponding tile’s tokens (§4.3).
    - SFT regime: unlike Flamingo/Llama 3‑V, unfreeze the LLM in SFT and mix in high-quality text-only SFT to maintain text performance (§4.3; §6.5; Table 9).
  - `NVLM-H` (hybrid; §4.4):
    - Split image processing across two paths: the global thumbnail’s tokens are injected into the decoder sequence (like `NVLM-D`) to support joint multimodal reasoning; the regular tile tokens are accessed via gated cross-attention (like `NVLM-X`).
    - Gains more reasoning capacity than pure cross-attention (thanks to the thumbnail in the decoder) with better efficiency than decoder-only (because high-res tile tokens aren’t unrolled into the sequence).

- Efficiency trade-offs (Table 3)
  - With identical backbones and DHR settings, `NVLM-X` (cross-attention) trains ~1.76× faster than `NVLM-D` because the decoder sequence excludes the large number of image tokens:
    > Table 3: 34B models on 128 H100s—`NVLM-X`: 50.6 samples/s; `NVLM-D`: 28.8; `NVLM-H`: 36.2.

- Data and training pipeline (§5; §4.5; Appendix B)
  - Two-stage training:
    - Pretraining: freeze LLM and vision encoder; train projector or X-attention on curated captioning, VQA, chart, document, OCR, and math datasets (Table 4).
      - Key finding: diverse, high-quality pretraining data beats larger but noisier web corpora even with a frozen LLM (Table 5).
    - SFT: unfreeze the LLM (for all three models), keep the vision encoder frozen, and train on a broad multimodal SFT mixture, plus a high-quality text-only SFT blend (Table 6; §5.3). This is the central ingredient that avoids text-skill degradation.
  - Backbones: `Qwen2-72B-Instruct` (72B) and `Nous-Hermes-2-Yi-34B` (34B) (§4.5).

## 4. Key Insights and Innovations
- 1) Tile-tagged dynamic high-resolution that improves both OCR and reasoning (Table 1; Table 2; §4.2–4.3)
  - What’s new: simple text tokens `<tile_k>` inserted before each tile’s tokens (or as anchors for cross-attention) to expose the tiling structure.
  - Why it matters: replaces the DHR trade-off (better OCR, worse reasoning) with improvements on both. For a decoder-only 34B model, `DHR + 1-D tag` improves MathVista to 53.8 (from 46.1 low-res) and OCRBench to 806 (from 622) while also improving MMMU val to 52.0 (Table 1). Similar gains hold for cross-attention (Table 2).

- 2) A hybrid architecture (`NVLM-H`) that combines joint reasoning and efficiency (Figure 3; §4.4; Table 3)
  - What’s new: thumbnail tokens go through the decoder (enabling unified text-image reasoning), while detailed tiles are read via cross-attention. This mixture achieves strong reasoning (best MathVista within NVLM) with better throughput than decoder-only.
  - Why it matters: shows a middle ground between the simplicity of decoder-only and the efficiency of cross-attention.

- 3) A training recipe that preserves/improves text-only capabilities in open models (Table 8; §5.3)
  - What’s new: mixing a curated text-only SFT dataset during multimodal SFT, plus substantial multimodal math data. Unlike prior open models, text accuracy increases over the backbone.
  - Why it matters: `NVLM-D 72B` improves the average of MMLU, GSM8K, MATH, HumanEval by +4.3 points over `Qwen2‑72B‑Instruct` (Table 8). This closes the “production-grade” gap where open models typically underperform on text after multimodal training.

- 4) Removing the Perceiver resampler to protect spatial fidelity for OCR (Appendix C; Figure 9)
  - What’s new: demonstrate that Flamingo’s Perceiver resampler can hinder dense OCR by mixing spatial information; `NVLM-X` processes image tokens directly with cross-attention.
  - Why it matters: leads to much stronger OCR/document performance in a cross-attention architecture (Table 7; OCRBench 828–831 for `NVLM-X/H` 72B).

## 5. Experimental Analysis
- Evaluation setup (§6.1–§6.3)
  - Vision-language: MMMU (reasoning), MathVista (math in visual context), VQAv2 (natural scenes), AI2D (diagram reasoning), TextVQA (scene text), ChartQA, DocVQA, RealWorldQA, OCRBench (§6.1).
    - Note: The SFT blend includes training splits of several of these datasets (e.g., ChartQA, DocVQA, VQAv2, TextVQA, AI2D), so reported scores are not zero‑shot on those (Section 5.2).
  - Text-only: MMLU, GSM8K, MATH, HumanEval (§6.1).
  - Baselines: strong proprietary systems (e.g., GPT‑4o, Claude 3.5, Gemini 1.5) and leading open models (e.g., InternVL 2, LLaVA-OneVision; §6.2).

- Main quantitative results (Table 7, Table 8)
  - Frontier-level VL performance:
    > Table 7: `NVLM-D 72B` reaches OCRBench 853 and VQAv2 85.4, the top scores among listed models; MMMU test/val 54.6/59.7; AI2D 85.2 (94.2 no_mask); TextVQA 82.1; ChartQA 86.0; DocVQA 92.6.
  - Hybrid and cross-attention trade-offs:
    > Table 7: `NVLM-H 72B` achieves the highest NVLM MathVista (66.6) and best open-access MMMU val among listed open models (60.2), while `NVLM-X 72B` offers similar overall strength with better training throughput (Table 3).
  - Preserved/improved text-only performance:
    > Table 8: `NVLM-D 72B` average across text benchmarks improves from 79.8 (backbone) to 84.1 (+4.3); `NVLM-X 72B` and `NVLM-H 72B` also improve (+2.5 and +2.7). In contrast, other open multimodal models show −6.3 to −6.9 average drops.

- Ablations and analyses that support claims
  - Tile-tag ablation (Table 1; decoder-only 34B):
    > `DHR + 1-D tag` outperforms `DHR + No tag` across all vision-language benchmarks, including MMMU (52.0 vs. 50.0), MathVista (53.8 vs. 51.7), and OCRBench (806 vs. 728).
  - Tile-tag ablation (Table 2; cross-attention 34B):
    > Adding 1‑D tags improves over `DHR + No tag` on MMMU (54.1 vs. 53.0), MathVista (59.6 vs. 57.6), and OCRBench (744 vs. 682).
  - Pretraining data quality (Table 5):
    > Using the curated pretraining blend (Table 4) instead of the standard LLaVA pretraining raises MathVista from 48.9 to 53.8 and OCRBench from 760 to 806 (decoder-only 34B), with gains across other tasks too.
  - Efficiency comparison (Table 3):
    > Cross-attention (`NVLM-X`) trains ~1.76× faster than decoder-only for the same 34B setup; the hybrid sits in between.
  - Freezing vs. unfreezing the LLM in cross-attention SFT (Table 9):
    > Freezing preserves text ability by design but leaves vision-language performance noticeably lower than the unfrozen variant at the same scale. For 72B, unfrozen `NVLM-X 1.0` improves AI2D (84.2 vs. 76.2), TextVQA (80.2 vs. 76.2), ChartQA (82.9 vs. 76.2), DocVQA (82.9 vs. 76.4), and OCRBench (828 vs. 722).

- Do the experiments support the claims?
  - Yes, across multiple angles:
    - The tile-tag mechanism systematically lifts both OCR and reasoning benchmarks (Table 1–2).
    - The hybrid architecture meaningfully shifts the reasoning/efficiency frontier (Table 3, Table 7).
    - The integrated text-only SFT strategy is validated by improved text benchmarks over the backbone (Table 8) and by the freezing/unfreezing study (Table 9).
  - Caveat: Some evaluated datasets were included in SFT (Section 5.2), so those numbers reflect finetuned—not zero-shot—performance.

## 6. Limitations and Trade-offs
- Computational efficiency vs. unified reasoning
  - Decoder-only (`NVLM-D`) offers unified multimodal reasoning in the decoder but suffers from long sequences when unrolling high-res image tokens; this lowers throughput and raises memory use (Table 3; §4.3 “Decoder-only vs. X-attention”).
  - Cross-attention (`NVLM-X`) is more efficient but does less joint reasoning within the decoder. The hybrid (`NVLM-H`) partially mitigates this by inserting the thumbnail into the decoder sequence.

- Scope and assumptions
  - Vision encoder is always frozen; while simplifying training and stability, it may limit improvements from end-to-end tuning on some domains (§4.1; §4.2).
  - DHR uses up to 6 tiles at training (§4.1); extremely large or complex pages/images may exceed this capacity unless the tiling or token budget is increased.

- Data and evaluation scope
  - Some vision-language benchmarks (e.g., ChartQA, DocVQA, VQAv2, TextVQA, AI2D) appear in the SFT mixture (Section 5.2); results on these reflect supervised exposure and are not purely zero-shot.
  - The curated text-only SFT relies on model-assisted refinement using GPT‑4o/4o‑mini (§5.3), which may embed stylistic biases into the responses.

- Modality coverage
  - The work focuses on images and text. Video and audio are not addressed, and extending tile-tagged DHR to long videos raises new design and efficiency questions.

## 7. Implications and Future Directions
- How this changes the field
  - Demonstrates that open multimodal models can reach frontier-level VL performance without sacrificing—indeed, improving—text-only skills by carefully structuring both architecture and data/recipes (Table 7–8). This narrows the “production-grade multimodality” gap in open access (§1; §6.4).
  - Provides a clear, controlled comparison between decoder-only, cross-attention, and a new hybrid approach, with quantified efficiency and accuracy trade-offs (Figure 3; Table 3; Table 7).

- Follow-up research enabled or suggested
  - Extending tile-tagging to 2‑D tags that generalize well (Table 1 shows 1‑D tags win here, but there may be smarter 2‑D encodings or learned positional schemas).
  - Learning to adapt the number of tiles or token budgets dynamically per task/query for better compute–accuracy Pareto fronts.
  - End-to-end adaptation of the vision encoder (perhaps with careful regularization) for domains like dense documents or charts while preserving text skills.
  - Unified multi-resolution strategies for video, with temporal “tile tags” or segment tags and memory-efficient attention.

- Practical applications
  - Document understanding, OCR-heavy workflows, and chart/table reasoning benefit from the DHR + tile-tag pipeline (Table 7: OCRBench 828–853; DocVQA up to 92.6).
  - General assistants that must switch seamlessly between text-only tasks (coding, math word problems) and image reasoning; the improved GSM8K/MATH/HumanEval scores (Table 8) are particularly relevant for developers and education tools.
  - On-device or latency-constrained scenarios can opt for `NVLM-X` for higher throughput, whereas offline high-accuracy pipelines might prefer `NVLM-D` or `NVLM-H` depending on task mix (Table 3; Table 7).

In sum, NVLM 1.0 contributes both a set of high-performing open models and concrete design principles—tile-tagged DHR, hybrid processing, and a text-preserving SFT recipe—that others can adopt and build upon. The released weights (`NVLM-D-72B`) and forthcoming code aim to accelerate community progress (Abstract; project links).
