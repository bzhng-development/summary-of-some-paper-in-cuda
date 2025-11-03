# PaddleOCR-VL: Boosting Multilingual Document Parsing via a 0.9B Ultra-Compact Vision-Language Model

**ArXiv:** [2510.14528](https://arxiv.org/abs/2510.14528)

## 🎯 Pitch

PaddleOCR-VL introduces a state-of-the-art, ultra-compact document parsing framework that couples a lightweight, dedicated layout analyzer with a high-efficiency 0.9B vision-language model to achieve exceptional accuracy in multilingual, page-level, and element-level recognition—including text, tables, formulas, and charts—across 109 languages. By decoupling layout detection from element recognition and leveraging an innovative NaViT-style visual encoder, PaddleOCR-VL surpasses prior solutions in both speed and accuracy while drastically reducing resource requirements, making robust document understanding possible on practical, resource-constrained deployments and enabling new capabilities for information extraction and retrieval in global, real-world scenarios.

---

## 1. Executive Summary
PaddleOCR-VL introduces a two‑stage, resource‑efficient system for multilingual document parsing that combines a lightweight layout analyzer with a compact 0.9B vision‑language model (`PaddleOCR‑VL‑0.9B`). It delivers state‑of‑the‑art accuracy on page‑level and element‑level tasks (text, tables, formulas, charts) across 109 languages while running faster and with less memory than competing systems (see Table 2, Table 4, Table 13).

## 2. Context and Motivation
- Problem addressed
  - Parsing real documents involves identifying layout regions (text blocks, tables, formulas, charts, images), ordering them for correct reading, and converting each element into structured content (e.g., Markdown, JSON) with high fidelity. Section 1 stresses that modern documents mix dense text, complex tables/charts, mathematical expressions, multiple languages, and handwriting—making accurate, scalable parsing difficult.
- Why it matters
  - Reliable parsing is a foundation for information retrieval and RAG systems and is critical for digitizing reports, contracts, academic papers, forms, and historical or multilingual materials (Section 1).
- Prior approaches and their limitations
  - Pipeline systems of “expert” models achieve strong accuracy but suffer from integration complexity and error compounding across stages (Section 1; e.g., PP‑StructureV3, MinerU‑pipeline; Table 2 and Table 3 include these baselines).
  - End‑to‑end vision‑language models (VLMs) simplify workflows but incur high latency/memory due to long autoregressive outputs and can hallucinate layout/reading order, especially on multi‑column pages (Section 1, Section 2.1.1).
- Positioning of this work
  - The work decouples layout analysis from recognition:
    - Stage 1: a dedicated, small layout model (`PP‑DocLayoutV2`) detects elements and predicts reading order (Figure 3).
    - Stage 2: a compact VLM (`PaddleOCR‑VL‑0.9B`) recognizes each cropped element with a dynamic‑resolution NaViT vision encoder and a small, efficient language model (ERNIE‑4.5‑0.3B) (Figure 4).
  - This design aims to keep layout stable and fast while achieving high recognition accuracy with minimal compute (Section 2.1).

## 3. Technical Approach
The system is a two‑stage pipeline (Figure 2), followed by light post‑processing to produce Markdown/JSON.

1) Stage 1: Layout analysis with `PP‑DocLayoutV2` (Section 2.1.1; Figure 3)
- Tasks: detect element boxes and classes; infer reading order.
- Detection and classification: extends `RT‑DETR` (a real‑time transformer detector) to localize and classify text blocks, tables, formulas, charts (Section 2.1.1).
- Reading order with a `pointer network` (six transformer layers):
  - A pointer network is a sequence model that outputs an ordering over a set of inputs by “pointing” to items in the input. Here, it orders detected layout regions.
  - Pre‑selection: apply per‑class thresholds to keep “foreground” boxes to be ordered.
  - Embeddings: each proposal is embedded with absolute 2D positional encodings and class‑label embeddings.
  - Relation modeling: the encoder attention adds a geometric bias from `Relation‑DETR` to capture pairwise spatial relations (near/far, left/right, above/below).
  - Pairwise relation head: linearly projects region features to queries/keys, computes bilinear similarities, and produces an N×N matrix of pairwise “who comes before whom” logits.
  - Decoding: a deterministic “win‑accumulation” algorithm turns pairwise preferences into a topologically consistent reading order (Section 2.1.1).
- Training (Section 2.2.1):
  - Two stages: (i) train `RT‑DETR` on 20k+ curated pages for 100 epochs (initialized from `PP‑DocLayout_Plus‑L`), then freeze; (ii) train the pointer network for 200 epochs with `Generalized Cross Entropy` loss (robust to noisy labels), constant LR 2e‑4, `AdamW`.

2) Stage 2: Element‑level recognition with `PaddleOCR‑VL‑0.9B` (Section 2.1.2; Figure 4)
- Inputs: crops of each detected element in reading order.
- Vision encoder: a `NaViT`‑style encoder initialized from Keye‑VL, which ingests images at their native resolution by “patch‑and‑pack,” avoiding forced resizing/tiling. This reduces distortions and hallucinations on dense text (Section 2.1.2).
- Projector: a 2‑layer MLP with GELU activation and merge size 2 bridges visual features to the language embedding space (Section 2.1.2).
- Language model: `ERNIE‑4.5‑0.3B`—a small LLM chosen to speed up autoregressive decoding; enriched with `3D‑RoPE` positional encoding. 3D‑RoPE extends rotary positional embeddings to capture multiple axes (e.g., sequence and modality), improving alignment of visual and textual token positions (Section 2.1.2).
- Outputs: for each element type, the model emits structured text:
  - OCR: characters/words/lines/blocks; simple page structure hints.
  - Tables: an `OTSL` representation. OTSL (Optimized Table Tokenization) is an efficient, token‑friendly table serialization format (Section 2.2.2, Table 1; [28]).
  - Formulas: LaTeX with distinctions between inline `\(...\)` and display `\[...\]`.
  - Charts: normalized Markdown tables (Section 2.2.2).

3) Training the VLM (Section 2.2.2; Table 1)
- Stage 1 (alignment pretraining): 29M image‑text pairs; 1 epoch; batch 128; sequence length 16384; max resolution 1280×28×28; data augmentation on; LR decays from 5e‑5 to 5e‑6. Goal: align vision features with language space across diverse content.
- Stage 2 (instruction fine‑tuning): 2.7M curated samples; 2 epochs; same batch and context length; higher max resolution (2048×28×28); smaller LR (5e‑6 → 5e‑7). Tasks explicitly cover OCR, tables (OTSL), formulas (LaTeX), charts (Markdown table).

4) Data construction (Section 3; Figure 5; Appendix A)
- Sources (Section 3.1):
  - Open datasets (e.g., CASIA‑HWDB for handwriting [29]; UniMER‑1M & MathWriting for formulas [30, 31]; wide chart corpora including ChartQA/PlotQA/DVQA/Unichart/etc. [32–40]).
  - Synthesized data to fix long‑tail gaps and balance distributions.
  - “Network accessible” web data (papers, newspapers, scans, slides, exams) to diversify style/quality.
  - In‑house datasets from prior OCR research.
- Automatic annotation (Section 3.2):
  - Use `PP‑StructureV3` expert models to produce pseudo labels, then craft prompts for strong VLMs (`ERNIE‑4.5‑VL`, `Qwen2.5‑VL`) to refine them; apply hallucination filtering and rule checks.
- Hard case mining (Section 3.3):
  - Build an evaluation engine with fine‑grained categories across text, tables, formulas, charts and measure with task‑specific metrics: Edit Distance (text), `TEDS` (tables), `CDM` (formulas), `RMS‑F1` (charts).
    - `TEDS` (Tree Edit Distance‑based Similarity) compares predicted vs. ground‑truth table trees.
    - `CDM` (Character Detection Matching) matches rendered character positions for formulas—robust to LaTeX surface variations (Section 4.2.3; [64]).
    - `RMS‑F1` summarizes table reconstruction from charts (Section 4.2.4; [42]).
  - Identify weaknesses and synthesize targeted “hard” cases using font/CSS libraries, XeLaTeX, and browsers.

5) Inference system (Section 4.3)
- Multi‑threaded asynchronous pipeline with three threads: data loading (PDF → images), layout model, VLM; queues connect stages to overlap computation. VLM batches are formed either by size or wait‑time, allowing cross‑page batching for higher GPU utilization.
- Deployed on high‑throughput backends (`vLLM`, `SGLang`), tuning batch tokens and memory utilization (Section 4.3; Table 13; Table A2).

## 4. Key Insights and Innovations
- Decoupled, geometry‑aware reading order (fundamental)
  - Instead of relying on a VLM to “write out” layout sequences, `PP‑DocLayoutV2` first detects regions and then orders them via a pointer network with geometric bias (Section 2.1.1; Figure 3).
  - Significance: removes long‑sequence layout generation, improving stability and speed on complex multi‑column pages and graphics‑text mixtures (Section 2.1.1). Table 2 shows top reading‑order accuracy (Edit 0.043 on OmniDocBench v1.5).
- Native dynamic‑resolution vision encoder in a compact VLM (fundamental)
  - A `NaViT`‑style encoder processes arbitrary‑resolution inputs without tiling; coupled with a small `ERNIE‑4.5‑0.3B` decoder for fast autoregressive generation (Section 2.1.2; Figure 4).
  - Significance: fewer hallucinations, better dense text performance, strong multilingual coverage (109 languages; Appendix B), with lower compute (Tables 2, 3, 12; and Table 13 for speed).
- LLM‑assisted, quality‑controlled data pipeline (incremental but impactful)
  - Combining expert pseudo labels, strong VLM refiners, rule‑based validation, and targeted hard‑case synthesis yields 30M+ high‑quality training pairs across elements (Section 3; Appendix A).
  - Significance: scales high‑fidelity labels while mitigating hallucinations; drives SOTA element‑level accuracy (Tables 5, 8, 10, 12).
- High‑throughput, cross‑page batching inference (incremental)
  - Asynchronous, multi‑threaded queues and micro‑batching across documents on `vLLM/SGLang` deliver best pages/s and tokens/s among compared systems on A100 (Table 13).

## 5. Experimental Analysis
Evaluation setup spans page‑level and element‑level tasks with public and in‑house sets (Section 4).

- Datasets and metrics
  - Page‑level: OmniDocBench v1.5 and v1.0 (weighted combinations of text Edit Distance, formula CDM, table TEDS; includes reading‑order Edit Distance), and olmOCR‑Bench unit tests (pass rates) (Sections 4.1; Tables 2–4).
  - Element‑level: text (OmniDocBench‑OCR‑block, In‑house‑OCR, Ocean‑OCR‑Bench), tables (OmniDocBench‑Table‑block, In‑house‑Table), formulas (OmniDocBench‑Formula‑block, In‑house‑Formula), charts (In‑house‑Chart). Metrics: Edit Distance, `TEDS`, `CDM`, `RMS‑F1` (Section 4.2; Tables 5–12).

- Main quantitative results
  - Page‑level
    - OmniDocBench v1.5 (Table 2): 
      > Overall score `92.56` (best), Text‑Edit `0.035` (lower is better), Formula‑CDM `91.43`, Table‑TEDS `89.76`, Table‑TEDS‑S `93.52`, Reading‑order Edit `0.043`.  
      Next best overall is MinerU2.5 at `90.67`.
    - OmniDocBench v1.0 (Table 3):
      > Avg overall edit `0.115` (lower is better). Text Edit: `0.041` EN, `0.062` ZH. Reading order: `0.045` EN (near‑SOTA), `0.063` ZH (best).  
      Table TEDS: `88.0` EN (slightly below SOTA; the paper attributes this to annotation typos in v1.0), `92.14` ZH (strong).
    - olmOCR‑Bench (Table 4):
      > Overall pass rate `80.0 ± 1.0` (best). Category highlights: ArXiv `85.7` (best), Headers&Footers `97.0` (best), Multi‑column text `79.9` (2nd), Long Tiny Text `85.7` (2nd).  
      Stronger than dots.ocr (`79.1`), MinerU2.5 (`77.5`), and MonkeyOCR‑pro‑3B (`75.8`).
  - Element‑level
    - Text
      - OmniDocBench‑OCR‑block (Table 5): best or tied‑best Edit Distance in all nine document types; e.g., PPT2PDF `0.049`, Academic literature `0.021`, Newspaper `0.034`.
      - In‑house‑OCR (Table 6): multilingual Edit Distance best in all reported scripts, e.g., Arabic `0.122` vs Qwen2.5‑VL‑72B `0.405`; Japanese `0.086`; Latin `0.013`. Across text types: Handwritten CN `0.089`, Printed EN `0.016`, Vertical text `0.005`, Rare characters `0.001`.
      - Ocean‑OCR‑Bench (Table 7): 
        > EN Edit `0.118` (best); ZH Edit `0.034` (best) with highest F1/Precision/Recall/BLEU/METEOR in both EN and ZH.
    - Tables
      - OmniDocBench‑Table‑block (Table 8): 
        > Overall `TEDS 0.9195` (best), Structural `TEDS 0.9543` (best), Overall Edit Dist `0.0561` (best).
      - In‑house‑Table (Table 9): 
        > Overall `TEDS 0.8699` and Structural `0.9066` (both best).
    - Formulas
      - OmniDocBench‑Formula‑block (Table 10): 
        > Overall `CDM 0.9453` (best), with EN `0.9677`, ZH `0.9228`.  
        Note: dots.ocr scores are low because cropped formulas are often treated as images (table note).
      - In‑house‑Formula (Table 11): 
        > Overall `CDM 0.9882` (best).
    - Charts
      - In‑house‑Chart (Table 12): 
        > Overall `RMS‑F1 0.844` (best), surpassing `PP‑StructureV3` (`0.806`) and Qwen2.5‑VL‑72B (`0.730`).
  - Inference efficiency (Table 13; Section 4.3)
    - On OmniDocBench v1.0 end‑to‑end, with vLLM backend:
      > `1.2241 pages/s` and `1881.2 tokens/s` (best), using `43.7 GB` VRAM.  
      MinerU2.5: `1.0574 pages/s`, `1647.9 tokens/s`, `41.9 GB`; dots.ocr: `0.3522 pages/s`, `78.5 GB`.
    - Cross‑hardware stability (Table A2) on A100, A10, RTX 3060/4090D shows consistent speed‑memory trade‑offs.

- Do the experiments support the claims?
  - Yes, across three independent page‑level benchmarks (Tables 2–4) and multiple element‑level tasks (Tables 5–12), the system consistently reaches SOTA or near‑SOTA with concrete gains. The speed study (Table 13) substantiates efficiency claims. The only caveat is that several element‑level benchmarks are in‑house (tables, formulas, charts), so external reproducibility depends on future releases.

- Ablations and robustness
  - The paper details training procedures and data construction but does not present ablations isolating the impact of NaViT vs. fixed‑resolution encoders, or pointer‑network design choices (no ablation tables reported). Robustness is indirectly evidenced by category‑wise results (e.g., handwriting, vertical text, multilingual; Tables 5–7).

## 6. Limitations and Trade-offs
- Dependence on accurate detection
  - The two‑stage pipeline hinges on Stage‑1 detection quality; missed or mis‑classified boxes propagate to recognition. Per‑class thresholding for proposal selection (Section 2.1.1) can trade recall for precision.
- Pairwise ordering complexity
  - The pointer network computes an N×N pairwise matrix; while page N is moderate, this is O(N²) and might stress very dense pages (Section 2.1.1).
- Specialized scope
  - The VLM is tailored for document parsing. It is not evaluated on general multimodal reasoning or open‑ended VQA; capabilities beyond text/tables/formulas/charts are out of scope (Sections 2.2.2, 4.2).
- Data and benchmarking constraints
  - Heavy reliance on automatic labels and synthesis (Section 3; Appendix A) can introduce biases from the teacher models and rendering engines. Several strong results are on in‑house datasets (Tables 9, 11, 12), limiting external verification until those sets or equivalents are public.
- Compute requirements, while efficient, are non‑trivial
  - Best throughput is reported on A100 with ~44 GB average VRAM (Table 13). Consumer GPUs (e.g., RTX 3060) work (Table A2) but at significantly lower throughput (~0.35 pages/s).
- Chart benchmarking
  - The chart evaluation is only in‑house due to issues with public test quality and imbalance (Section 4.2.4), so cross‑paper comparability is limited.

## 7. Implications and Future Directions
- Field impact
  - This work demonstrates that a compact, task‑specialized VLM with native dynamic‑resolution vision and a dedicated layout front‑end can outperform much larger general VLMs on document parsing while being faster (Tables 2–4, 13). It challenges the assumption that bigger end‑to‑end models are required for high‑fidelity document conversion.
- Practical applications
  - High‑throughput enterprise ingestion of PDFs into structured Markdown/JSON; multilingual digitization of archives and newspapers; ingestion of scientific literature and financial reports; robust OCR for complex layouts and handwriting; improved RAG pipelines thanks to accurate reading order and structural parsing (Section 1; Demos in Appendix D).
- Research directions
  - Public release of the in‑house datasets or creation of standardized chart/table/formula testbeds with reliable annotations to improve comparability (Section 4.2.4).
  - Ablations on NaViT vs. alternatives, projector designs, and pointer network components (geometric bias, decoding strategies) to quantify each contribution.
  - Joint training of layout and recognition with shared features while preserving the current pipeline’s stability—e.g., using the VLM to refine detection/ordering proposals from `PP‑DocLayoutV2`.
  - Further compression and distillation for edge deployment; mixed‑precision and sparse attention to reduce A100‑class dependence (Table A2 suggests feasibility).
  - Extending chart understanding from table reconstruction to code‑level generation (e.g., matplotlib/seaborn programs) and semantic captioning, leveraging the data synthesis pipeline (Appendix A.4).
  - Document‑level reasoning across pages (cross‑page reading order, footnote–reference linking, figure–caption alignment) by augmenting the pointer network and VLM context.

Overall, PaddleOCR‑VL offers a clear, well‑engineered path to accurate and efficient document parsing at scale: decouple layout from recognition, use a native‑resolution vision backbone, keep the decoder small for speed, and invest in strong data construction and inference engineering. The reported results and speed (Tables 2–4, 13) indicate that this approach currently sets the standard for practical, multilingual document conversion.
