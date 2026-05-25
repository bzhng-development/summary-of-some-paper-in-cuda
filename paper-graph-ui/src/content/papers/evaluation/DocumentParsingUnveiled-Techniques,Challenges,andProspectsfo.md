# Document Parsing Unveiled: Techniques, Challenges, and Prospects for Structured Data Extraction

**ArXiv:** [2410.21169](https://arxiv.org/abs/2410.21169)

## 🎯 Pitch

This comprehensive survey unifies the fragmented landscape of document parsing by systematically comparing modular pipeline systems and end-to-end vision-language models, spanning layout analysis, OCR, mathematical expression understanding, table and chart extraction, evaluation, and tools. By critically examining both methodologies and outlining their strengths, weaknesses, and future research frontiers, the paper delivers an essential and up-to-date roadmap for researchers and practitioners driving automation in knowledge extraction, search, and generative AI from complex real-world documents.

---

## 1. Executive Summary
This survey organizes the entire field of document parsing into two end‑to‑end strategies—modular pipelines and large vision‑language models (VLMs)—and then walks through every major component: layout analysis, OCR, math expression understanding, table and chart parsing, evaluation metrics, and tools. Its primary significance is a unifying, up‑to‑date map of methods, datasets, and metrics, plus a careful discussion of where pipeline systems and VLMs succeed or fail in complex, real‑world documents (Sections 2–9; Figures 1–7; Tables 1–10; Appendix 11).

## 2. Context and Motivation
- Problem/gap addressed
  - Extracting structured, machine‑readable data (e.g., Markdown, JSON, LaTeX) from unstructured or semi‑structured documents (PDFs, scans) is brittle in practice because documents include dense text, tables, formulas, charts, and complex reading orders. Section 1 frames this as “document parsing (DP),” which converts visual content into structured representations while preserving relationships among elements like text, equations, tables, images, and reading order.
  - Existing surveys are either dated or limited to subareas (layout, math, tables, charts). This survey aims to be holistic and current (Section 1: “limitations… high‑quality reviews often focus on specific sub‑technologies”).
- Why it matters
  - Real‑world impact: Parsed content fuels search and retrieval, RAG (retrieval‑augmented generation), knowledge base construction, and training/evaluation of multi‑modal models (Section 1; Section 2.1.2; Section 9).
  - Theoretical significance: Document parsing combines layout structure, language semantics, and visual signals—an ideal setting to study multi‑modal representation learning (Sections 3 and 7).
- Prior approaches and their shortfalls
  - Modular pipelines (layout → OCR → tables/math/charts → relation integration) are precise but fragile: error propagation, complex module orchestration, rule‑heavy reading‑order logic, and poor generalization to varied layouts (Section 9, “Challenges… Pipeline‑Based Systems”).
  - General‑purpose VLMs understand images but struggle with dense, high‑resolution, text‑heavy pages and long multi‑page documents (Section 7.1; Section 9).
- Positioning relative to existing work
  - The paper proposes a taxonomy (Figures 1–2) spanning:
    - Modular systems: layout analysis (physical and logical), OCR (detection/recognition/spotting), math detection/recognition, table detection/structure recognition, chart tasks, and relation integration (Sections 3–6; Figure 2).
    - End‑to‑end VLMs and specialized document parsers (Nougat, Donut, Fox, Vary, OmniParser, GOT) with their training strategies and trade‑offs (Section 7).
  - It consolidates datasets and metrics across all components (Appendix 11; Tables 2–10), and catalogs open‑source tools (Section 8; Table 1).

## 3. Technical Approach
This is a survey, so the “approach” is a structured decomposition of how document parsing systems work, not a single new algorithm. The paper’s framework (Figures 1–2) divides the space into modular pipelines and end‑to‑end VLMs, then drills into how each module or model class operates.

A) Modular pipeline systems (Figure 2; Sections 2.1, 3–6)
1) `Document Layout Analysis (DLA)` (Section 3; Figure 3)
   - Goal: Identify elements (paragraphs, images, tables, formulas, headers/footers) with coordinates and reading order; also distinguish “physical” layout (bounding boxes) from “logical” semantic roles (title, caption).
   - Approaches and mechanics:
     - CNN‑based detectors (Section 3.1.1): Adapt object detectors like R‑CNN/Mask R‑CNN/YOLO to page objects; FCNs for segmentation of regions. Example: `DocLayout‑YOLO` augments YOLOv10 with a global‑to‑local receptive module to catch elements at multiple scales.
     - Transformer‑based models (Section 3.1.2): `BEiT`/`DiT` split pages into patches and use self‑attention to capture global structure; strong features but computationally heavy.
     - Graph‑based models (Section 3.1.3): Build a graph where nodes are detected regions and edges encode spatial/semantic proximity; run GCNs to refine types and relationships (`Doc‑GCN`, `GLAM`).
     - Grid‑based models (Section 3.1.4): Rasterize the page into a 2D token grid (e.g., `BERTGrid`, `VGT`) to keep spatial structure explicit; useful but often large/slow.
     - Integrating semantics (Section 3.2): Pretrained multi‑modal Transformers like `LayoutLM/v2/v3` fuse text, 2D positions, and image features via masking and cross‑modal attention; `UniDoc` aligns ResNet visual features with Transformer text features via gated cross‑attention. These target “logical” layout (roles, hierarchy), not just boxes.

2) `Optical Character Recognition (OCR)` (Section 4; Figure 4)
   - Text detection (Section 4.1): Four families
     - Single‑stage regression: predict oriented boxes directly (e.g., `TextBoxes++`, `SegLink`, `DRRG`).
     - Two‑stage proposal: adapt Faster R‑CNN‑like pipelines to text proposals for arbitrarily oriented text.
     - Segmentation‑based: per‑pixel text masks (`CRAFT`, `PixelLink`, SPCNet).
     - Hybrids: combine regression with segmentation and improved NMS or attention (`EAST`, `CentripetalText`).
   - Text recognition (Section 4.2):
     - Vision‑only encoders (CNN or ViT) with either `CTC` decoding (align free) or `seq2seq` decoders with attention.
     - Handling irregular/curved text: input rectification (`STN`, `MORAN`, `ESIR`) or 2D attention decoders (`SATRN`, `ViTSTR`, `TrOCR`).
     - Injecting semantics (Section 4.2.2): 
       - Character‑level hints (e.g., counting auxiliary tasks in `RF‑L`, sorting/counting in `CDDP`).
       - Dedicated semantic modules (`SRN`’s global reasoning; `SEED` between encoder/decoder; `ABINet` iterative language refinement).
       - Pretraining with masked objectives (`VisionLAN`) or degradation robustness (`Text‑DIAE`) to teach context even when pixels are noisy.
   - Text spotting (Section 4.3): Joint detection+recognition
     - Two‑stage: share a backbone; detect regions, then ROI‑based recognition (Mask TextSpotter v1–v3, RoIRotate/BezierAlign variants, GLASS).
     - One‑stage: avoid ROIs; directly model characters/words end‑to‑end (CRAFTS; Transformer decoders like `PGNet`, `SPTS`, `TESTR`).

3) `Mathematical expressions` (Section 5; Figure 5)
   - Detection (Section 5.1): 
     - U‑Net‑style segmentation for inline vs display formulas; object detection variants (`DS‑YOLOv5`, SSD, Faster/Mask R‑CNN); or treat it as entity‑relation extraction (`FormulaDet`) to leverage context.
   - Recognition (Section 5.2):
     - Encoder‑decoder to produce LaTeX/MathML from images: CNN or Transformer encoders; RNN/Transformer decoders with attention (manage nesting and 2D structure).
     - Enhancements: multi‑scale encoders (DenseNet/ResNet), global dependency via Swin Transformer; character/length hints; stroke orders for online handwriting; heavy augmentation.

4) `Tables` (Section 6; Figure 6)
   - Detection (Section 6.1): Fine‑tune generic detectors (YOLO, Faster R‑CNN, Deformable ConvNets) with table‑specific anchors/features; handle sparsity with modified training (e.g., SparseR‑CNN variants).
   - Structure recognition (Section 6.2):
     - Row/column segmentation: use Transformers (`DETR`, `DQ‑DETR`) or Bi‑GRU scanning to find separators; then merge cells, sometimes with global attention to predict spanning relations.
     - Cell‑based (bottom‑up): detect individual cells or their keypoints; construct a graph and merge via GNNs; robust to irregular structures.
     - Image‑to‑sequence: encode the whole table image; decode to HTML/LaTeX/Markdown (dual decoders for structure+content; e.g., `MASTER`, `VAST`).

5) `Charts` (Section 6.3–6.5; Figure 7)
   - Tasks: classification of chart types; split multi‑panel composites; detect elements (axes, bars, legends, labels); link text to visual marks; extract data series; parse structures like flowcharts/org charts.
   - Methods:
     - Classification: CNNs, then stronger Vision Transformers; Swin Transformer fine‑tuning leads current accuracy (Section 6.4).
     - Detection and text‑element linking: Faster‑R‑CNN/YOLO for visual elements; OCR for text; transformer‑based methods for correlating labels to marks.
     - Structure extraction: DETR‑style models for nodes+edges (`FR‑DETR`) to parse diagrams with linking lines.

6) `Relation integration` (Section 2.1.3)
   - Combine outputs from all modules into a single document representation (Markdown/JSON/LaTeX), preserving spatial order and semantics. Reading order may use rules or specialized models. Figure 2 shows an explicit “Integrate All” step using bounding boxes and types to rebuild the page.

B) End‑to‑end, VLM‑driven systems (Section 2.2 and Section 7)
- Early general VLMs (`Qwen‑VL`, `InternVL`) handle image+text but miss fine‑grained OCR and complex structures (Section 7.1).
- Specialized models:
  - `Donut`/`Nougat` (Section 7.2): Swin encoder + seq2seq decoder trained to emit Markdown with embedded LaTeX; excels on scientific PDFs without modular OCR, but slower and weaker on non‑Latin scripts.
  - `Vary` (Section 7.2): Enlarged visual vocabulary and SAM‑style tokens to better handle charts and OCR on high‑res pages without splitting.
  - `Fox` (Section 7.3): Multi‑page understanding with multiple pretrained visual vocabularies (CLIP‑ViT + SAM‑ViT) for fine‑grained cross‑page tasks.
  - `Detect‑Order‑Construct` (Section 7.3): Tree‑construction for document hierarchy—detect page objects, assign roles, predict reading order, then construct the hierarchical tree.
  - Unified frameworks (Section 7.4): 
    - `OmniParser`: decouple OCR from structural decoding with a two‑stage decoder; handles text spotting, KIE, and tables in one system.
    - `GOT` (“General OCR Theory”): treat all text‑like entities (text, formulas, tables, scores) as objects and train end‑to‑end across scene and document OCR.

C) Evaluation foundations, benchmarks, and tools
- Datasets and metrics across all sub‑tasks are consolidated in Appendix 11 (Tables 2–10; Section 11.2).
- Open‑source tools summarized in Table 1 (Section 8), spanning OCR engines (Tesseract, PaddleOCR), PDF parsers (PyMuPDF, pdfplumber), conversion systems (MinerU, PDF‑Extract‑Kit), and model‑based parsers (OmniParser).

## 4. Key Insights and Innovations
- A unifying two‑track taxonomy with explicit module interlocks (Figures 1–2; Sections 2–6).
  - What’s new: A single map that spans physical layout and logical semantics, from detection all the way to reconstruction, while also covering VLM alternatives side‑by‑side.
  - Why it matters: It helps practitioners decide when to prefer a robust module versus an end‑to‑end model, and how to connect modules cleanly (e.g., reading order and relation integration in Figure 2).
- Clear separation of physical vs logical layout—and methods that bridge them (Section 3; Figure 3).
  - Different from many older surveys that stop at bounding boxes, this review centers methods that fuse text, vision, and positions (`LayoutLM` family, `UniDoc`, `VSR`), i.e., the “logical layout” layer needed for titles, captions, and hierarchical roles.
- End‑to‑end document parsers are not universal replacements yet (Section 7; Section 9).
  - The paper distills the practical limits of OCR‑free or VLM‑only approaches: dense text, formatting fidelity, high‑resolution images, and multi‑page flow still challenge general VLMs. This is a corrective to “one model fits all.”
- Comprehensive, cross‑task evaluation toolbox (Appendix 11).
  - Beyond collecting datasets, the survey foregrounds task‑specific metrics that practitioners often overlook:
    - `TEDS` and `CAR` for table structure (Appendix 11.2.4).
    - `CDM` for math recognition fairness across LaTeX variants (Appendix 11.2.3).
    - Chart‑specific scoring (strict/relaxed OKS, data‑series scores) (Appendix 11.2.5).
  - Significance: Better metrics change what models optimize for, especially in structure‑heavy tasks where token‑level accuracy alone is misleading.

## 5. Experimental Analysis
This survey does not run new experiments; instead it codifies how the community evaluates document parsing and what data is available. The most “quantitative” contributions are the dataset and metric inventories, with sizes and task coverage.

- Evaluation methodology (Appendix 11.2)
  - Layout: IoU, mAP, F1 for element localization and classification (Table 7).
  - OCR: character error rate (CER), edit distance, BLEU/METEOR/ROUGE for long text fidelity (Table 8).
  - Math: exact expression rate, image‑based similarity (SSIM/MSE), and `CDM` to avoid penalizing equivalent LaTeX renderings (Table 9).
  - Tables: detection purity/completeness; structure `CAR`, `TEDS`, simplified `S‑TEDS`, `A_all`, `F_β`, and `WAF` for adjacency across IoU thresholds (Table 10).
  - Charts: detection IoU/precision/recall; chart‑type classification; task‑specific series‑matching scores and keypoint similarity (Appendix 11.2.5).

- Datasets (Appendix 11.1)
  - Layout analysis (Table 2): from large synthetic corpora to modern diverse sets.
    - Example numbers: `DocLayNet` 80,863 docs with 11 classes; `M6Doc` 9,080 docs with 74 classes; `DocSynth‑300K` 300,000 synthetic pages.
  - OCR (Table 3): classic scene‑text to document‑centric sets.
    - Example numbers: `LOCR` lists 7,000,000 instances across TD/TR/TS; `ICDAR2019‑ReCTS` has 25,000 Chinese pages with detection/recognition/structure labels.
  - Math (Table 4): `ArxivFormula` 700,000 pages; `FormulaNet` 46,672 images with ~1,000,000 expressions; `UniMER‑1M` 1,061,791 printed/handwritten images for recognition.
  - Tables (Table 5): end‑to‑end structure benchmarks.
    - `PubTables‑1M` 1M scientific tables; `TableBank` 417,234; `Wired Table in the Wild` contains deformed and occluded images; `TableGraph‑350K` 358,767; `TabRecSet` 38,100.
  - Charts (Table 6): classification/extraction sets.
    - `UB‑PMC 2019/2020/2022` for classification and extraction; `LINEEX430K` for line charts; `ExcelChart400K` for pie/bar extraction.
  - Multi‑task document sets (Appendix 11.1.6):
    - `Readoc` (2,233 PDF–Markdown pairs) for end‑to‑end structure extraction.
    - `OmniDocBench`: “981 PDF pages and 100,000 annotations… 9 document types, 19 layout tags, and 14 attribute tags” to compare modular and VLM methods.
- Tools (Section 8; Table 1)
  - From OCR engines (Tesseract, PaddleOCR) to high‑level pipelines (`MinerU`, `PDF‑Extract‑Kit`, `OmniParser`), with brief capabilities summarized.

Do the curated evaluations support the claims?
- Yes, for two reasons:
  - The metrics are task‑appropriate and highlight nuances: e.g., using `TEDS`/`CAR` for tables (structure, not just detection), and `CDM` for math (Appendix 11.2.3–11.2.4).
  - The datasets are broad enough to expose known weaknesses: multi‑page, diverse layouts, borderless/nested tables, and both printed/handwritten math (Appendix 11.1).

What’s missing (and acknowledged in Section 9)?
- Cross‑model, like‑for‑like benchmarks on a single protocol are not run here; instead, the paper equips readers to perform them. Failure cases are discussed qualitatively (Section 9), not through ablation studies.

> Section 9 explicitly contrasts pipeline vs VLM failure modes: pipelines suffer from “modular coordination, standardization of outputs, and handling irregular reading orders,” while VLMs struggle with “high‑density text, intricate table structures,” and “repeated outputs or formatting errors” in long generations.

## 6. Limitations and Trade-offs
- Assumptions and boundaries of current practice (Section 9)
  - Pipelines assume module interfaces are stable and that reading order can be rule‑based or heuristically predicted; this breaks in multi‑column, nested, or heavily styled pages.
  - Many OCR modules assume moderate text density and common fonts; complex fonts (bold/italics) and tiny characters degrade accuracy.
  - Math recognition pipelines often assume clean, printed expressions; inline, noisy, screen‑captured, and handwritten expressions remain problematic.
  - Table structure methods often assume clear separators; borderless/nested/multi‑page tables and multi‑line cells remain hard.
  - Chart extraction lacks unified problem definitions and standard evaluation; many systems are partially manual or chart‑type‑specific.
- Computational and data constraints (Section 7 and Section 9)
  - VLMs:
    - High‑res documents require tiling or special encoders; naive downscaling erases small text.
    - Many systems freeze LLM backbones; this hampers fine‑grained OCR and leads to repetition/formatting drift on long outputs.
    - Resource heavy: training/inference on dense pages is computationally wasteful without architectural compression or sparse sampling.
  - Pipelines:
    - Error propagation: mis‑detections early on (e.g., missing a table region) corrupt downstream structure and final serialization.
    - Domain adaptation: each module may need re‑tuning for new document styles or languages.
- Scenarios not fully addressed (Sections 7, 9)
  - Long‑document, multi‑page coherence and cross‑page entity linking are still emerging (even in `Fox`, Section 7.3).
  - Multilingual and multi‑script robustness is uneven (e.g., `Nougat` struggles on non‑Latin scripts, Section 7.2).
  - Diagram‑rich documents (posters, manuals, newspapers) are underrepresented in training and evaluation (Section 9).

## 7. Implications and Future Directions
- How this work changes the landscape
  - Provides a complete “wiring diagram” for document parsing systems that connects low‑level vision modules, multi‑modal pretraining, and end‑to‑end VLMs (Figures 1–2). Practitioners can make principled build‑vs‑buy decisions and identify exactly where to insert semantics (Section 3.2) or switch to OCR‑free parsing (Section 7).
- Research enabled or suggested (Sections 7 and 9; Appendix 11)
  - High‑resolution, long‑document VLMs: compression strategies (e.g., DocOwl2‑style high‑res compression), sparse sampling (`WuKong` in refs), and multi‑page memory mechanisms.
  - Reading order learning: move beyond rules to learned sequence prediction that aligns visual, positional, and linguistic signals reliably in complex layouts (Section 2.1.3; Section 9).
  - Structure‑aware decoding: decoders that emit hierarchical representations (trees/graphs) with guaranteed well‑formedness, not just token streams—building on `Detect‑Order‑Construct` (Section 7.3) and table/diagram graph models (Sections 6.2.2 and 6.5.4).
  - Unified, fair metrics: wider adoption and refinement of `TEDS`, `CAR`, `CDM`, and chart OKS‑style measures; new math metrics that reflect semantic equivalence beyond LaTeX strings (Appendix 11.2).
  - Dataset diversification: beyond scientific papers/textbooks to manuals, posters, newspapers; multi‑script and low‑resource languages (Appendix 11.1; Section 9).
  - Hybrid systems: tight coupling of strong detectors/recognizers with VLM decoders that enforce global coherence and formatting constraints—e.g., `OmniParser`‑style decoupling or `GOT`’s object‑as‑token approach (Section 7.4).
- Practical applications
  - RAG pipelines that rely on faithful structure preservation for chunking and retrieval (Section 1; Section 9).
  - Enterprise ingestion: contracts, invoices, receipts, financial reports with table‑heavy and math‑heavy content.
  - Scientific workflows: Markdown/LaTeX export (`Nougat`), chart data extraction for meta‑analysis, and equation databases for math search.
  - Training data engines: automatic generation of structured corpora to train or fine‑tune domain‑specific VLMs (Section 1; Section 7).

In short, this survey is a field guide: it shows exactly how each piece of a document parser works, where VLMs do and don’t help, what data and metrics to use for each subtask, and what hard problems remain—especially reading order, dense text, structure fidelity, and long‑document reasoning.
