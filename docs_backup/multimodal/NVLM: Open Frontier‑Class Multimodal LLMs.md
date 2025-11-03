# NVLM: Open Frontier‑Class Multimodal LLMs

**ArXiv:** [2409.11402](https://arxiv.org/abs/2409.11402)
**Authors:** Wenliang Dai, Nayeon Lee, Boxin Wang, Zhuolin Yang, Zihan Liu, Jon Barker, Tuomas Rintamaki, Mohammad Shoeybi, Bryan Catanzaro, Wei Ping
**Institutions:** NVIDIA (likely)

## 🎯 Pitch

NVLM 1.0 revolutionizes open multimodal language models by introducing three cutting-edge architectures and a novel 'tile-tagging' approach for handling high-resolution images. This innovation not only excels in vision-language tasks but also enhances text-only capabilities, bridging a critical gap between open and proprietary LLMs, and paving the way for advanced applications in OCR, document analysis, and STEM education.

---

## 1. Executive Summary (2-3 sentences)
This paper introduces NVLM 1.0, a family of open multimodal large language models (LLMs) that integrate images and text using three architectures—`NVLM-D` (decoder-only), `NVLM-X` (cross-attention), and `NVLM-H` (hybrid)—and a new “tile-tagging” mechanism for high‑resolution images. Trained with a carefully curated blend of multimodal and text-only data, NVLM 1.0 achieves frontier-level results on vision-language benchmarks while improving its text-only math and coding performance over its own backbone LLM (Table 7 and Table 8).

## 2. Context and Motivation
- Problem and gap
  - Open multimodal LLMs often excel at vision-language tasks but degrade on pure text benchmarks after multimodal training. In contrast, proprietary models (e.g., GPT‑4o) perform strongly on both modalities (“production-grade multimodality”). Section §1 and §3.2 detail this gap; Table 8 quantifies the degradation in open models and the lack of degradation in some proprietary ones.
  - Architectural comparisons in prior work are inconclusive because they vary backbones, vision encoders, and training data. This paper performs an apples-to-apples comparison using the same backbones, encoders, and data across decoder-only and cross-attention designs (§3.2, §4).
  - Handling high-resolution images is crucial for OCR-style tasks, but dynamic tiling methods that simply concatenate tile tokens can hurt reasoning tasks like MMMU (§3.3; Table 1 “DHR + No tag” lowers MMMU vs. low-resolution).
- Importance
  - Real-world impact: OCR, document understanding, charts, tables, and diagrams power enterprise workflows; strong OCR/diagram performance is necessary for practical deployment (Benchmarks in §6.1 include DocVQA, OCRBench, AI2D, ChartQA).
  - Theoretical and systems significance: understanding how architectural choices (decoder-only vs. cross-attention vs. hybrid) and data curation influence efficiency, scaling, and generalization.
- Prior approaches and their shortcomings
  - Decoder-only models (e.g., LLaVA, InternVL; §3.2) unify tokens across modalities but become inefficient with high-resolution images because all image tokens are unrolled into the LLM’s sequence.
  - Cross-attention models (e.g., Flamingo; §3.2) are more efficient, but components like the “perceiver resampler” can hurt OCR by mixing spatial information (Appendix C, Fig. 9). Freezing the LLM preserves text-only performance but can limit adaptation to new multimodal instructions (§3.2; Table 9).
  - Dynamic high-resolution tiling without structure cues can hurt reasoning (§3.3; Table 1).
- Positioning
  - NVLM 1.0 provides a controlled architectural comparison (§4), a high‑resolution input pipeline with “tile tags” (§4.2–§4.4; Fig. 3–4), and a training recipe that improves text-only performance after multimodal SFT by adding high-quality text-only data and multimodal math data (§5.2–§5.3; Table 8).

## 3. Technical Approach
Step-by-step summary of the system (Fig. 3; §4–§5):

1) Shared vision pathway and dynamic high-resolution (DHR) preprocessing (§4.1; Fig. 3–4)
- Vision encoder: a single, frozen `InternViT-6B-448px-V1-5` produces 1,024 tokens per 448×448 tile (§4.1).
- Dynamic tiling for high-res images: the input image is tiled to at most 6 tiles plus one global thumbnail tile, chosen to match aspect ratio; each tile is encoded independently (Fig. 4, §4.1).
- Downsampling via “pixel shuffle”: the 1,024 tokens per tile are reduced to 256 tokens by grouping neighboring tokens along channels (§4.1, Fig. 4). This reduces LLM load without changing spatial alignment semantics.
- Why DHR: OCR tasks benefit enormously from high-resolution details, but naive concatenation of all tile tokens can confuse the LLM about spatial grouping, hurting reasoning tasks (§3.3; Table 1 “DHR + No tag” lowers MMMU).

2) Tile tags to preserve structure in DHR (§4.2–§4.4; Table 1–2; Fig. 3)
- What they are: lightweight, text tokens like `<tile_1> ... <tile_k>` inserted before each tile’s image tokens to mark tile boundaries and positions (a “global” tag marks the thumbnail).
- Why they help: they give the LLM explicit structure so it does not have to infer the tiling purely from content. This consistently improves both OCR and reasoning accuracy (Table 1 and Table 2).
- Which tags work best: a simple 1‑D tag (`<tile_k>`) generally outperforms 2‑D grid or bounding-box tags across many tasks (Table 1: best overall; Table 2: wins on most benchmarks).

3) Three architectures (Fig. 3; §4.2–§4.4)
- `NVLM-D` (decoder-only; §4.2):
  - Modality alignment: a 2-layer MLP projects image tokens into the LLM’s embedding space; then the LLM processes text and image tokens uniformly via self-attention.
  - Training: two-stage. Pretraining freezes LLM and vision encoder while training only the MLP. SFT unfreezes the LLM and trains it jointly with the MLP (§4.2). Risk: text-only capability can degrade; mitigation: include a high-quality text-only SFT mixture (§5.3; Table 8 shows improvements, not degradation).
  - Pros: unified multimodal reasoning in the decoder; strong OCR (Table 7). Cons: long sequences and higher compute when many image tokens are unrolled (Table 3 shows lower throughput).
- `NVLM-X` (cross-attention; §4.3):
  - Image tokens are not unrolled into the LLM sequence. Instead, “gated cross-attention” layers read from the image tokens while the LLM processes text. Gates can modulate how much image context flows.
  - Design choice: remove the “perceiver resampler” found in Flamingo because it harms OCR by mixing spatial structure (Appendix C, Fig. 9).
  - Tile tags are inserted in the LLM sequence, and cross-attention masks ensure `<tile_k>` attends only to its own tile tokens (§4.3; Fig. 3).
  - Training: both cross-attention layers and the LLM are trained in SFT (not frozen), plus text-only SFT data is blended to preserve/improve text performance (§4.3; Table 9 contrasts frozen vs. unfrozen).
  - Pros: much faster training/inference with high-res images (Table 3). Cons: extra parameters for cross-attention; may trail decoder-only on OCR-heavy tasks (Table 7).
- `NVLM-H` (hybrid; §4.4):
  - Thumbnail tokens go into the LLM sequence (like decoder-only) to enable joint global-image + text reasoning; fine‑grained regular tiles are read via gated cross-attention (like cross-attention models). Both flows use modality-alignment MLPs (§4.4; Fig. 3).
  - Pros: balances reasoning quality and efficiency; higher throughput than decoder-only (Table 3) and strong reasoning scores (Table 7).

4) Training method (§4.5; Appendix B)
- Two-stage:
  - Pretraining: freeze LLM and vision encoder; train only modality-alignment modules on curated pretraining data (Table 4). Large global batch (2048) and cosine LR schedules (§4.5; Table 10).
  - SFT: freeze vision encoder; train LLM plus modality-alignment on a large, task-diverse multimodal SFT mixture (Table 6) blended with a refined, high-quality text-only SFT set (Table 6, “Text-only SFT”; §5.3; Table 11 for hyperparameters).
- Backbones: Qwen2‑72B‑Instruct for the 72B models; Nous‑Hermes‑2‑Yi‑34B for ablations (§4.5).
- Implementation detail: gated cross-attention layers are inserted every 6–8 transformer layers; 10 such layers total in `NVLM-X` and `NVLM-H` (§4.5).

5) Data pipeline and the “quality over scale” finding (§5.1–§5.3)
- Pretraining data: combines high-quality captioning corpora with large task-oriented datasets (OCR, documents, charts, math; Table 4). Attempts with much larger but noisier web data and interleaved corpora added little or hurt alignment (§5.1; Table 5 shows clear gains from curated pretraining).
- Multimodal SFT: diverse, task‑oriented datasets for natural images, OCR, charts/diagrams/tables, documents, science, and math reasoning (Table 6).
- Text-only SFT: high-quality instruction, math, and coding datasets refined using GPT‑4o/-mini to improve response quality; decontamination performed (§5.3).

## 4. Key Insights and Innovations
- A controlled, apples-to-apples architectural comparison (§4; Table 3, Table 7–8)
  - Distinguishes where decoder-only vs. cross-attention vs. hybrid truly differ when backbones, data, and encoders are held constant. This isolates effects on throughput, OCR quality, and reasoning.
- Hybrid architecture (`NVLM-H`) for joint reasoning + efficiency (§4.4; Table 3 and Table 7)
  - New design that routes global information (thumbnail) through self-attention while keeping fine-grained details via cross-attention. This achieves strong reasoning (MMMU Val 60.2) with better throughput than decoder-only.
- Tile-tagging for dynamic high resolution (§4.2–§4.4; Table 1–2; Fig. 3–4)
  - A simple, text-based 1‑D tag per tile consistently boosts OCR and reasoning across architectures compared to no tags or more complex tags. This resolves the typical MMMU drop seen with naive tiling.
- Data quality and task diversity > scale during pretraining (§5.1; Table 5)
  - Curated, task-oriented pretraining substantially improves math and OCR performance—even for decoder-only models—versus using large, noisy web pairs or interleaved corpora.
- Production-grade multimodality without text regression (§5.3; §6.4; Table 8)
  - Blending high-quality text-only SFT and multimodal math data during SFT not only prevents text capability loss but improves math and coding vs. the backbone LLM (+4.3 avg points for `NVLM-D 72B` across MMLU/GSM8K/MATH/HumanEval; Table 8).

## 5. Experimental Analysis
- Evaluation setup (§6.1–§6.2)
  - Vision-language (9 tasks): MMMU (Val/Test), MathVista, VQAv2, AI2D (two settings), TextVQA, ChartQA, DocVQA, RealWorldQA, OCRBench.
  - Text-only (4 tasks): MMLU, GSM8K, MATH, HumanEval.
  - Baselines: leading proprietary systems (e.g., GPT‑4o, Claude 3.5, Gemini 1.5) and strong open models (e.g., InternVL-2, LLaVA-OneVision, Llama 3‑V). Some baselines are not open-source (marked “*” in Table 7–8).
- Main quantitative results (Table 7; highlights)
  - `NVLM-D 1.0 72B`:
    - Top scores among listed models on OCRBench and VQAv2:
      > “VQAv2 85.4; OCRBench 853” (Table 7).
    - Competitive multimodal reasoning: MMMU Val 59.7, MathVista 65.2 (Table 7).
  - `NVLM-H 1.0 72B`:
    - Best MMMU Val among open-access models listed:
      > “MMMU (Val) 60.2; MathVista 66.6” (Table 7).
  - `NVLM-X 1.0 72B`:
    - Frontier-level accuracy while being far more efficient than decoder-only on high-res images:
      > “Throughput: 50.6 samples/s vs. 28.8 for decoder-only; same hardware and batch” (Table 3).
- Text-only performance (Table 8)
  - Many open models regress after multimodal training (e.g., `LLaVA‑OneVision 72B`: −6.3 avg points vs. backbone).
  - NVLM-1.0 improves vs. its own backbone (`Qwen2‑72B‑Instruct`):
    > “`NVLM-D 72B`: 84.1 avg vs. 79.8 backbone (+4.3). `NVLM-H 72B`: 82.5 (+2.7). `NVLM-X 72B`: 82.3 (+2.5).”  
    Gains are strongest on math and coding—e.g., `NVLM‑D 72B` reaches MATH 73.1 and HumanEval 88.4 (Table 8).
- Ablations and diagnostics
  - Tile tags (Table 1–2):
    - Decoder-only (`NVLM-D 34B`): adding 1‑D tile tags improves OCRBench 806 vs. 728 (no tag) and raises MathVista/MMMU (53.8/52.0 vs. 51.7/50.0). Without tags, DHR hurts MMMU vs. low-res (50.0 vs. 50.9), but tags reverse this (52.0) (Table 1).
    - Cross-attention (`NVLM-X 34B`): 1‑D tags beat no tags on all listed benchmarks (e.g., OCRBench 744 vs. 682; MMMU Val 54.1 vs. 53.0; Table 2).
  - Pretraining data quality (Table 5): replacing “LLaVA‑1.5 data” with the curated pretraining set boosts OCR and math substantially (e.g., OCRBench 806 vs. 760; MathVista 53.8 vs. 48.9) while also improving chart/document tasks.
  - Throughput (Table 3): cross-attention is fastest under high-res loads, hybrid is in between, decoder-only is slowest due to long sequences (e.g., 2,816 tokens vs. 1,024 text-only for `NVLM-D`).
  - Frozen vs. unfrozen LLM in cross-attention SFT (Table 9):
    - Freezing the LLM preserves text-only ability by design but produces lower vision-language accuracy than unfrozen training (e.g., `NVLM-X 72B` unfrozen achieves higher AI2D/TextVQA/ChartQA/DocVQA/OCRBench).
- Caveats in evaluation setup (§5.2)
  - The SFT blend includes training splits of some benchmarks (e.g., VQAv2, TextVQA, ChartQA, DocVQA, AI2D). Results on those tasks are therefore not zero-shot; this mirrors common open-source practice but differs from unknown proprietary training regimens.

Do the experiments support the claims?
- Yes, the numbers directly show:
  - Frontier-level VL performance (Table 7).
  - Efficiency trade-offs across architectures (Table 3).
  - Tile-tag benefits (Table 1–2).
  - Text-only improvements post multimodal SFT (Table 8).
  - Impact of data quality in pretraining (Table 5).
  - Impact of freezing/unfreezing the LLM (Table 9).

## 6. Limitations and Trade-offs
- Compute and memory
  - Training is resource-intensive (e.g., throughput figures use 128 H100 GPUs; Table 3). Decoder-only becomes particularly heavy as image tiles increase due to long sequences.
- Architectural trade-offs (Table 3; §4.3)
  - Decoder-only: best OCR and unified reasoning but slowest under DHR.
  - Cross-attention: fastest but adds parameters for cross-attention layers; can trail decoder-only on OCR-heavy tasks (Table 7).
  - Hybrid: balances both but still more complex than either extreme.
- DHR and tile tagging constraints (§4.1–§4.4)
  - The system is configured for up to 6 tiles plus one thumbnail; scenarios requiring many more tiles or extremely large documents may stress the current design.
  - The best-performing 1‑D tag encodes tile order, not full 2‑D geometry. While accurate empirically (Table 1–2), it may be suboptimal in edge cases requiring fine 2‑D spatial reasoning.
- Data and evaluation considerations (§5.1–§5.3)
  - Some evaluation tasks appear in SFT training splits (not zero-shot), making strict comparisons to proprietary models difficult (§5.2).
  - The text-only SFT refinement uses GPT‑4o/-mini (§5.3), which may raise licensing or reproducibility questions for some users.
  - Interleaved web-scale corpora did not help in this framework (§5.1); future filtering or recaptioning methods might change this conclusion.
- Scope
  - The work targets images + text; it does not include audio, video, or speech modalities (though the methods could extend conceptually).

## 7. Implications and Future Directions
- Field impact
  - Provides a clear recipe for “production-grade” open multimodal models: high-quality pretraining and SFT data, explicit tile structure for high resolution, and a training scheme that preserves and even improves text-only skills (Table 8).
  - Establishes a rigorous, controlled architectural comparison and introduces a practical hybrid alternative (Fig. 3; §4.4), guiding future system design under compute and latency constraints.
- Follow-up research
  - Better structured tags: explore richer but still robust tile encodings (learned 2‑D tags, relative position codes shared across tiles) without hurting generalization (Table 1 suggests simplicity wins today).
  - Adaptive tiling: learn to place and size tiles based on content, possibly conditioned by the LLM, and allow more tiles for very large documents.
  - End-to-end adaptation: carefully unfreeze portions of the vision encoder in later stages to boost domain transfer (the current study keeps it frozen; §4.1).
  - Data advances: improve filtering/recaptioning of interleaved web corpora (§5.1) to unlock further gains without overfitting; expand multilingual OCR and math datasets.
  - Broaden modalities: extend the hybrid architecture to video (thumbnail = global clip summary; tiles = salient frames) and to audio-text-image tri-modal setups.
- Practical applications
  - Document processing (OCR/IE), chart/table QA, enterprise search and analytics, digital assistants that handle screenshots and scanned documents, STEM education tools combining diagrams with math reasoning, and code/math assistants that can “see” visual context (Table 6 task coverage; Table 7 performance).

Overall, NVLM 1.0 contributes both a high-performing model family and a set of design and training principles—tile-tagged dynamic high resolution, high-quality task-diverse data, and a hybrid architecture—that together produce strong, balanced multimodal systems without sacrificing text-only competence. The released weights (`NVLM-D-72B`) and forthcoming training code make these findings actionable for the community (§1; model card link on Hugging Face).
