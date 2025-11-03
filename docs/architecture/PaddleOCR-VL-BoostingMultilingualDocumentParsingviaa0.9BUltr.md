# PaddleOCR-VL: Boosting Multilingual Document Parsing via a 0.9B Ultra-Compact Vision-Language Model

**ArXiv:** [2510.14528](https://arxiv.org/abs/2510.14528)

## 🎯 Pitch

PaddleOCR-VL presents a cutting-edge, resource-efficient document parsing framework that decouples layout analysis from content recognition, combining a lightweight detector (PP-DocLayoutV2) with the ultra-compact PaddleOCR-VL-0.9B vision-language model. This innovation achieves state-of-the-art accuracy on multilingual, structurally complex documents—including expert-level handling of text, tables, formulas, and charts—while delivering unprecedented speed and efficiency over both expert pipelines and heavyweight VLMs. Its robust, language-agnostic, and scalable design empowers real-world applications like search, analytics, and RAG, unlocking high-fidelity structured data extraction for over 100 languages on practical hardware.

---

## 1. Executive Summary (2–3 sentences)
PaddleOCR‑VL introduces a two‑stage, resource‑efficient system for parsing complex, multilingual documents: a lightweight layout analyzer (`PP-DocLayoutV2`) followed by a compact vision‑language model (`PaddleOCR‑VL‑0.9B`) specialized for element recognition (text, tables, formulas, charts). Across public and in‑house benchmarks, it achieves state‑of‑the‑art accuracy while running faster and with lower memory than competing systems; e.g., it tops OmniDocBench v1.5 with an overall score of 92.56 and is 15.8% faster in end‑to‑end throughput than the strongest baseline on A100 GPUs (Table 2 and Table 13).

## 2. Context and Motivation
- Problem addressed
  - Parsing real‑world documents into structured, machine‑usable formats requires: detecting layout regions (paragraphs, tables, formulas, charts), ordering them for correct reading flow, and recognizing content precisely in 100+ languages. Documents often have multi‑column layouts, mixed text/graphics, handwriting, and dense small text (Section 1).
- Why it matters
  - Reliable parsing is the substrate for search, analytics, and Retrieval‑Augmented Generation (RAG). Errors in structure (e.g., wrong reading order) or content (e.g., misread formulas) degrade downstream LLM applications (Section 1).
- Prior approaches and gaps
  - Pipeline systems with “expert” modules (e.g., `PP‑StructureV3`, Marker, OpenParse, MinerU pipeline) can be accurate but suffer from integration complexity and error propagation, and they struggle on very complex layouts (Section 1; Table 2 and Table 3 show lower scores on several metrics).
  - End‑to‑end VLMs (e.g., `GOT`, `olmOCR`, `MonkeyOCR`, `dots.ocr`, general VLMs like `Qwen2.5‑VL`, `GPT‑4o`) simplify pipelines but tend to hallucinate, misorder long texts, and are expensive for long sequences (Section 1).
- Positioning of this work
  - The paper deliberately decouples page layout from content recognition: a small, stable detector predicts regions and reading order; a compact VLM—optimized for dynamic high‑resolution vision—recognizes element content. This hybrid design targets both stability and efficiency (Section 2; Figure 2).

## 3. Technical Approach
The system has two stages (Figure 2): (1) layout analysis with reading order, and (2) element‑level recognition with a compact VLM. A lightweight post‑processor merges both into Markdown/JSON outputs (Section 2.1).

1) Stage 1: `PP‑DocLayoutV2` (Layout analysis + reading order)
- Detection backbone: `RT‑DETR` (a real‑time variant of DETR for object detection). It localizes and classifies elements (text blocks, tables, formulas, charts) (Section 2.1.1; Figure 3).
- Reading order with a `pointer network`:
  - A pointer network is a sequence model that outputs an ordering (a permutation) over input items; here, the “items” are the detected layout boxes.
  - Inputs to the ordering network:
    - Absolute 2D positional encodings of each box and class label embeddings.
    - A geometric bias in attention inspired by `Relation‑DETR` to model pairwise spatial relations (Section 2.1.1).
  - Pairwise relation head:
    - Projects element features into queries/keys and computes bilinear similarities, yielding an `N×N` matrix of “which comes before which” pairwise logits (Section 2.1.1).
  - Decoding:
    - A deterministic `win‑accumulation` algorithm converts the pairwise matrix into a globally consistent reading order (Section 2.1.1).
- Training (Section 2.2.1):
  - Two‑stage: train detection/classification first (initialized from `PP‑DocLayout_Plus‑L`), then freeze it and train the pointer network for ordering.
  - Loss for ordering: `Generalized Cross Entropy (GCE)` to be robust to noisy labels; optimizer `AdamW`, constant LR 2e‑4 for 200 epochs.
  - Detection is trained 100 epochs following RT‑DETR practice on >20k curated pages.

2) Stage 2: `PaddleOCR‑VL‑0.9B` (Element‑level recognition)
- Goal: Given each cropped element from Stage 1, output its content in a structured format (text, `OTSL` tokens for tables, LaTeX for formulas, Markdown tables for charts).
- Key components (Figure 4; Section 2.1.2):
  - Dynamic high‑resolution vision encoder (`NaViT`‑style): NaViT (“Patch‑n‑Pack”) accepts native image resolutions by packing flexible‑sized patches instead of forcibly resizing/tiling; this reduces distortions and hallucinations on text‑dense images.
  - Vision weights initialized from `Keye‑VL` (Section 2.1.2).
  - A 2‑layer MLP projector (with `GELU`) maps vision features to the language embedding space; a “merge size of 2” reduces visual token count to cut decoding cost (Section 2.1.2).
  - Language backbone: `ERNIE‑4.5‑0.3B` (a 0.3B‑parameter multilingual LLM) for fast autoregressive decoding, enhanced with `3D‑RoPE` positional encoding to better represent multi‑axis positions in multimodal sequences (Section 2.1.2).
- Training recipe (Section 2.2.2; Table 1):
  - Stage 1 (vision–language alignment): 29M image–text pairs, batch 128, seq length 16384, max image resolution up to “1280×28×28” (paper’s NaViT notation), LR max/min 5e‑5/5e‑6 for 1 epoch. All modules trainable.
  - Stage 2 (instruction fine‑tuning): 2.7M curated samples, batch 128, seq length 16384, higher max resolution “2048×28×28”, LR max/min 5e‑6/5e‑7 for 2 epochs. Trains four instruction families:
    - OCR text: lines, blocks, pages.
    - Table recognition: outputs `OTSL` (Optimized Table Tokenization) sequences for structure + content (Section 2.2.2; [28]).
    - Formula recognition: outputs LaTeX; distinguishes inline `\(...\)` vs display `\[...\]`.
    - Chart to table: converts charts to normalized Markdown tables.
- Data engine (Section 3; Appendix A):
  - Automatic annotation: use `PP‑StructureV3` to produce pseudo‑labels; refine with powerful multimodal LLMs (`ERNIE‑4.5‑VL`, `Qwen2.5‑VL`) through prompt engineering; filter hallucinations and invalid outputs (Section 3.2).
  - Hard‑case mining: build fine‑grained eval suites (23 text types, 20 table types, 4 formula types, 11 chart types). Identify weak spots by metric (EditDist/TEDS/RMS‑F1/BLEU), then synthesize similar hard cases using fonts, CSS, corpora, and renderers (XeLaTeX, browsers) (Section 3.3).
  - Scale and coverage:
    - Text: 20M image–text pairs, 109 languages, multi‑style (printed, handwritten, vertical) (Appendix A.1).
    - Table: >5.5M pairs from automatic labeling, arXiv HTML mining, and high‑speed synthesis with structural/style controls; uses `OTSL` as target format (Appendix A.2).
    - Formula: 34,816 manually curated eval samples; training gathers LaTeX from arXiv sources, public sets (UniMER‑1M, MathWriting), filtered by renderability and image similarity; targeted synthesis for long‑tail patterns (Appendix A.3).
    - Chart: ~0.8M bilingual pairs via cleaned public data, two‑stage LLM annotation of axes/data, persona‑based style diversification, and long‑tail augmentation (Appendix A.4).
- Inference system (Section 4.3)
  - Asynchronous three‑thread pipeline: PDF rendering → layout model → VLM, with queues between stages. Cross‑page batching for VLM triggers either by batch size or timeout to maintain GPU utilization.
  - Serves on high‑throughput backends (`vLLM`, `SGLang`), tuning `max-num-batched-tokens` and `gpu-memory-utilization`.

Terminology that may be unfamiliar (defined above when first used):
- `NaViT`: a vision transformer that packs variable‑sized image patches so models can ingest native resolutions without fixed resizing/tiling artifacts.
- `RT‑DETR`: a DETR‑style detector optimized for real‑time object detection.
- `Pointer network`: a neural module that outputs an order (permutation) over input items by “pointing” to them sequentially.
- `Relation‑DETR` geometric bias: an attention bias that explicitly encodes pairwise geometric relations among boxes.
- `3D‑RoPE`: a three‑dimensional rotary positional embedding extending standard RoPE to better encode multiple positional axes in multimodal sequences.
- `OTSL`: a compact tokenization of table structure that simplifies learning table layout compared to raw HTML.
- `TEDS`: Tree‑Edit Distance Similarity; higher is better; it compares predicted vs ground‑truth table trees.
- `CDM`: Character Detection Matching; evaluates formula recognition by matching rendered characters and their positions.
- `RMS‑F1`: Root‑mean‑square F1 score across multiple column/row aggregations of chart‑to‑table predictions.

## 4. Key Insights and Innovations
1) Decoupled, order‑aware layout analysis for stability and speed
- What’s new: Instead of relying on a single VLM to “think” about layout and content jointly, the paper introduces a small, specialized layout model with a pointer network that predicts a globally consistent reading order from pairwise relations (Section 2.1.1; Figure 3).
- Why it matters: It avoids long‑sequence decoding for layout, reduces hallucinations on complex multi‑column pages, and makes adding new layout categories easier (Section 2.1; Figure 2). The reading‑order metric is best on OmniDocBench v1.5 (Table 2).

2) Dynamic‑resolution vision + compact language model for element recognition
- What’s new: A `NaViT`‑style encoder handles native‑resolution crops with fewer distortions, coupled with a lightweight `ERNIE‑4.5‑0.3B` decoder augmented by `3D‑RoPE` (Section 2.1.2).
- Why it matters: Better text fidelity on tiny/dense regions while keeping decoding fast. It delivers state‑of‑the‑art element recognition at ~0.9B parameters, showing a strong accuracy/efficiency Pareto point (Tables 2, 8, 10, 11, 12; Section 4.3).

3) Large‑scale, quality‑controlled data engine with hard‑case mining
- What’s new: A pipeline that (a) seeds pseudo‑labels from specialist models, (b) upgrades them using strong VLMs via carefully designed prompts, (c) filters hallucinations, and (d) continually mines/synthesizes targeted hard cases guided by an internal eval engine (Sections 3.1–3.3; Appendix A).
- Why it matters: Supports 109 languages and improves robustness on long‑tail patterns (handwriting, vertical scripts, messy scans), which is reflected in multilingual and handwriting benchmarks (Table 6; Table 7).

4) Practical end‑to‑end throughput gains
- What’s new: A multithreaded, cross‑page batching inference design integrated with `vLLM`/`SGLang` (Section 4.3).
- Why it matters: Real deployments care about pages/s, tokens/s, and VRAM. The system attains the best measured throughput while using less GPU memory than some larger baselines (Table 13).

Overall, (2) is an architectural innovation; (1) and (4) are system‑level innovations; (3) is a data/engineering advance that materially impacts robustness.

## 5. Experimental Analysis
- Evaluation methodology
  - Page‑level document parsing: OmniDocBench v1.5 and v1.0 (Section 4.1; Tables 2, 3) and `olmOCR‑Bench` unit tests (Table 4). Metrics include normalized `Text‑Edit` (lower is better), `Formula‑CDM` (higher), `Table‑TEDS`/`TEDS‑S` (higher), and `Reading Order Edit` (lower).
  - Element‑level recognition:
    - OCR text: `OmniDocBench‑OCR‑block` (17,148 cropped text blocks; Table 5), large in‑house multilingual/typology sets (Table 6), and handwriting `Ocean‑OCR‑Bench` (Table 7).
    - Tables: `OmniDocBench‑Table‑block` (512 tables; Table 8) and in‑house tables (Table 9).
    - Formulas: `OmniDocBench‑Formula‑block` (1,050 formulas; Table 10) and in‑house (34,816; Table 11).
    - Charts: in‑house (1,801) scored by `RMS‑F1` (Table 12).
  - Inference throughput: end‑to‑end from PDF path to Markdown on OmniDocBench v1.0 with 512‑PDF batches, single A100, comparing `vLLM` baselines (Table 13).
- Main quantitative results (all numbers from the paper)
  - OmniDocBench v1.5 (Table 2):
    > Overall score 92.56, beating the next best 90.67 (`MinerU2.5‑1.2B`). Sub‑metrics: `Text‑Edit` 0.035 (lower is better), `Formula‑CDM` 91.43, `Table‑TEDS` 89.76, `Table‑TEDS‑S` 93.52, `Reading‑Order‑Edit` 0.043.
    - This outperforms general VLMs (e.g., `Qwen2.5‑VL‑72B` overall 87.02) and specialized VLMs (`MonkeyOCR‑pro‑3B` 88.85; `dots.ocr` 88.41).
  - OmniDocBench v1.0 (Table 3):
    > Average Overall Edit Distance 0.115 (lower is better). Strong on Chinese/English text edit distance (0.062 ZH; 0.041 EN) and formulas (0.241 EN, 0.316 ZH). Chinese `Table‑TEDS` 92.14 is best; English `Table‑TEDS` 88.0 is competitive but not the top.
  - olmOCR‑Bench (Table 4):
    > Overall unit‑test pass rate 80.0 ± 1.0, highest among compared methods; category highs include ArXiv 85.7 and Headers/Footers 97.0, and second‑best on Multi‑column (79.9) and Long Tiny Text (85.7).
  - Text recognition (Tables 5–7):
    - `OmniDocBench‑OCR‑block` (Table 5): lowest edit distance across 9 document genres (e.g., Academic Literature 0.021; Newspaper 0.034; Note 0.081).
    - In‑house multilingual (Table 6a): best on all shown scripts—Arabic 0.122, Korean 0.052, Tamil 0.043, Greek 0.135, Thai 0.081, Telugu 0.011, Devanagari 0.097, Cyrillic 0.109, Latin 0.013, Japanese 0.086.
    - Handwriting (Ocean‑OCR‑Bench, Table 7): best edit distance `EN 0.118`, `ZH 0.034`, with top F1/Precision/Recall/BLEU/METEOR.
  - Table recognition (Tables 8–9):
    - `OmniDocBench‑Table‑block` (Table 8): `TEDS` 0.9195, `Structural TEDS` 0.9543, `Overall Edit Dist` 0.0561—best among all listed.
    - In‑house Tables (Table 9): `TEDS` 0.8699 and `Structural TEDS` 0.9066—best.
  - Formula recognition (Tables 10–11):
    - `OmniDocBench‑Formula‑block` (Table 10): `CDM` 0.9453 overall (EN 0.9677; ZH 0.9228)—best.
    - In‑house (Table 11): `CDM` 0.9882 overall (EN 0.9914; ZH 0.9849)—best.
  - Chart recognition (Table 12):
    > `RMS‑F1` 0.844 overall (EN 0.822; ZH 0.855), surpassing `Qwen2.5‑VL‑72B` (0.730) and the specialist `PP‑StructureV3` (0.806).
  - Inference performance (Table 13):
    > With `vLLM`, total time 800.9s (pages/s 1.2241; tokens/s 1881.2) vs `MinerU2.5` 927.3s (pages/s 1.0574; tokens/s 1647.9). Uses 43.7 GB VRAM (less than `dots.ocr` at 78.5 GB).
- Do experiments support the claims?
  - Yes on public benchmarks: The model leads on OmniDocBench v1.5 overall and on most sub‑metrics (Table 2), performs strongly on v1.0 (Table 3), and tops olmOCR‑Bench unit tests (Table 4). Element‑level SOTAs on table/formula crops (Tables 8, 10) indicate that decoupling layout from recognition did not hamper element accuracy.
  - Breadth: The paper includes multilingual and handwriting evaluations (Table 6, Table 7) and charts (Table 12), matching the stated “109 languages and multiple element types.”
- Notable omissions/considerations
  - Ablation studies are limited: there is no quantitative isolation of `NaViT` vs fixed‑resolution encoders, nor of the pointer network vs alternative ordering methods. Robustness to severe detection errors is not systematically studied.
  - Public vs in‑house: Many strongest numbers for language coverage, tables, charts, and formulas come from in‑house benchmarks; while informative, generalization beyond those distributions should be validated further.

## 6. Limitations and Trade‑offs
- Dependence on Stage‑1 layout quality
  - Although decoupling improves stability, misdetections or mis‑classified regions (e.g., formulas detected as images) will propagate to Stage 2. The paper does not quantify sensitivity to detection thresholds or propose recovery strategies.
- Reading order assumptions
  - The pointer network assumes that pairwise order relations are consistent and recoverable by `win‑accumulation`. Highly irregular layouts (e.g., complex sidebars, marginalia, bidirectional scripts with nested figures) may break these assumptions; no specific evaluation on such edge cases is shown beyond benchmark distributions.
- Data and evaluation coverage
  - Support for 109 languages is reported (Appendix B), but public multilingual evaluations cover a subset. Some very low‑resource scripts get limited third‑party benchmarking; performance may vary outside the in‑house sets.
- Charts and formulas
  - Chart evaluation is only in‑house (Section 4.2.4; Table 12), and thus comparability to established chart QA/dataset standards is indirect (though many public sources were used in training; Appendix A.4).
- Efficiency vs memory
  - Despite being compact for a VLM, end‑to‑end processing with dynamic high resolution still needs substantial GPU memory at scale (e.g., ~44 GB average with `vLLM` on A100 in Table 13). Deployments on edge devices may need further distillation or quantization.
- Transparency/ablations
  - The method integrates several design choices (NaViT, 3D‑RoPE, token merge size 2, ERNIE‑0.3B). Without ablations, it is hard to attribute gains to specific components or to tune trade‑offs (speed vs accuracy) for new settings.

## 7. Implications and Future Directions
- How this changes the landscape
  - Demonstrates that a small, well‑engineered VLM—paired with a dedicated layout model—can beat much larger general VLMs and specialized document models on both accuracy and throughput. For industry, this validates decoupled, efficiency‑focused architectures for large‑scale document conversion.
- Follow‑up research enabled/suggested
  - System ablations: quantify the contribution of `NaViT`, `3D‑RoPE`, token merging, and the pointer network; explore alternative graph‑based ordering or learnable decoding from pairwise relations.
  - Error‑tolerant interfaces: design feedback loops where Stage 2 can flag likely mis‑detections to re‑query Stage 1 (closing the loop on error propagation).
  - Cross‑page and multi‑document context: extend reading order and coreference across pages/sections (e.g., linking table references, figure captions, and bibliographic citations).
  - Open multilingual benchmarks: establish broader public evaluations for under‑represented scripts (beyond Latin/CJK), charts (with standardized structure extraction metrics), and formula varieties (print vs handwritten, bilingual LaTeX).
  - Edge deployment: further compress/quantize `PaddleOCR‑VL‑0.9B` and study CPU‑only performance; investigate knowledge distillation into even smaller decoders.
- Practical applications
  - Enterprise document ingestion (reports, invoices, forms), digital archiving of historical texts (vertical/ancient scripts; Figure A22), academic/scientific literature conversion (with formulas and tables), and high‑quality RAG pipelines that require faithful reading order and structure (Section 1; Figures A5–A8).

Overall, the work’s main contribution is an engineering‑sound, accuracy‑and‑efficiency‑balanced architecture that proves a compact VLM plus a dedicated layout analyzer is a winning recipe for real‑world, multilingual document parsing. The extensive results on public benchmarks (Tables 2–4, 8, 10) substantiate this claim, while the in‑house analyses and data engine point to a robust path for continued improvement.
