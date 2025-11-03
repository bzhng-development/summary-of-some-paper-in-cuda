# olmOCR: Unlocking Trillions of Tokens in PDFs with Vision Language Models

**ArXiv:** [2502.18443](https://arxiv.org/abs/2502.18443)

## 🎯 Pitch

olmOCR introduces a breakthrough open-source toolkit that leverages document-anchoring and a fine-tuned 7B vision-language model to convert diverse PDF documents—including those with complex layouts and poor scans—into accurate, linearized plain text. This innovation makes high-fidelity extraction of structured content (like tables, sections, and formulas) both affordable and scalable, outperforming top commercial and open models while cutting costs by over 30x. By unlocking the trillions of high-quality tokens trapped in global PDF archives, olmOCR empowers language model training and downstream AI applications with richer, higher-quality data at unprecedented scale.

---

## 1. Executive Summary
olmOCR is an open-source system that converts PDF pages into clean, linearized plain text by fine-tuning a 7B vision-language model (VLM) and guiding it with a technique called document-anchoring. It delivers state-of-the-art accuracy on a new, unit-test–style benchmark (olmOCR-Bench) while being dramatically cheaper than commercial VLM APIs; for example, it costs about $176 per million pages on commodity GPUs versus $12,480 for GPT‑4o (Table 6).

## 2. Context and Motivation
- Problem/gap
  - Large parts of the world’s high-quality text are locked in PDFs whose content is stored as positioned glyphs rather than logical text (Figure 2). This breaks reading order and structure (sections, tables, formulas), making extraction hard and error-prone.
  - Standard open-source parsers (e.g., GROBID, layout pipelines) often mis-handle order and structure; general VLMs can do better but are expensive or closed (Section 1; Table 6).
  - For language models, low-fidelity extraction harms both training stability and downstream performance; inference workflows (RAG, reading assistance) also depend on faithful linearization (Section 1).

- Importance
  - Unlocks “trillions of tokens” of high-quality training data residing in PDFs and scanned books (Abstract).
  - Low-cost, scalable extraction makes large-scale corpus creation feasible (Figure 1; Table 6).

- Prior approaches and limitations
  - Pipeline tools stitch together specialized models/heuristics (e.g., Grobid, MinerU, Marker; Section 1; Related Work §5). They often focus on extraction, not robust linearization (proper reading order, floating elements).
  - End-to-end OCR/VLM systems (e.g., Nougat, GOT Theory 2.0) work from page images but ignore that many PDFs are born-digital and contain anchorable internal text/metadata (Appendix A).

- Positioning
  - olmOCR combines the strengths of end-to-end VLMs with PDF-internal signals via document-anchoring, fine-tunes a 7B open model for the task, and releases a rigorous, unit-test–style benchmark to measure progress (§3; Figure 3).

## 3. Technical Approach
The system has four pillars: (1) data generation via document-anchoring, (2) fine-tuning a 7B VLM, (3) a scalable inference pipeline with robustness heuristics, and (4) a pass/fail benchmark.

1) Document-anchoring (what it is and how it works)
- Definition: document-anchoring injects structured hints from the PDF’s internals—text blocks and image coordinates—into the VLM prompt alongside the page image, grounding the model in what’s actually on the page (Appendix A; Figure 3).
- Mechanism
  - Parse page structure with `pypdf` to get text blocks and images plus their coordinates (Appendix A).
  - Prioritize blocks from the start/end of the page and sample them until a character limit is met (Appendix A).
  - Prompt the model with the rasterized page plus this “anchor” text and request a structured JSON response, including `natural_text` in reading order and flags like `is_table`, `is_diagram`, and rotation fields (Appendix E.1 schema).
- Why this design
  - Pure image OCR tends to hallucinate or “complete” text and struggle with dense layouts; anchoring reduces omissions and hallucinations by tying the generation to PDF internals (Appendix A).
  - It leverages a crucial advantage of born-digital PDFs while still working for scans (anchors become empty; Appendix A).

2) Data creation: olmOCR‑mix‑0225 (§2)
- Sources and scale
  - 102,825 PDFs, 258,641 pages after filtering; 96,929 web-crawled docs and 5,896 Internet Archive books (Table 1).
  - Document mix: academic (55.9%), brochures (11.2%), legal (10.2%), books (6.8%), tables (5.6%), diagrams (4.7%), slideshows (1.9%), other (3.7%), estimated via VLM classification (Table 2).
- Supervision
  - Generate “silver” linearized text with GPT‑4o prompted with document-anchoring and a strict JSON schema so outputs include `natural_text` plus page metadata (Section 2.2; Appendix E.1).
  - Anchored prompting measurably improves GPT‑4o quality on the benchmark (Table 4, “GPT‑4o (Anchored)” vs “No Anchor”).

3) Model training: fine-tune a 7B VLM (§2.3)
- Base model: `Qwen2‑VL‑7B‑Instruct`.
- Training setup
  - Effective batch size 4, LR 1e‑6, AdamW, cosine schedule, 10,000 steps (~1.2 epochs) on 8×H100 80GB (16 node-hours for a run; §2.3).
  - Images rendered to 1024 px long edge; prompts capped to ~6,000 characters anchor text; total input often ~3k tokens (image ~1k, anchors ~1.8k; §2.3).
  - Sequence length truncated to 8,192; loss masked to only the final response tokens; output format remains the structured JSON (Appendix E.2).
- Design choice: Full fine-tune vs LoRA
  - Full fine-tuning gives lower validation loss than LoRA on both web PDFs and Internet Archive scans (Appendix C, Figures 4–5).

4) Inference pipeline and robustness (§D)
- Serving stack
  - Efficient GPU serving with `SGLang` or `vLLM`, batching ~500-page work items and scaling from one to hundreds of GPUs via shared object storage (Appendix D.1).
- Robustness heuristics (Appendix D.2)
  - Prompt-length control: regenerate anchors with exponentially decreasing limits to fit 8,192 tokens.
  - JSON without forced decoding: rely on the model’s learned schema adherence; on parse failure, retry.
  - Rotation: read `is_rotation_valid` and `rotation_correction`; if needed, rotate the page and reprocess.
  - Degenerate repetitions: raise temperature (τ≈0.8) and permit up to N retries, falling back to plain text PDF extraction if failures persist. Measured 12% retry rate in practice (Table 6; Appendix D.2).

5) Benchmark: olmOCR‑Bench (§3)
- Philosophy: pass-or-fail “unit tests” against outputs from full PDFs, avoiding fuzzy references and LLM-judging (Section 3; Table 3).
- Test categories and counts (Table 3; §3.1–3.2)
  - Text presence, absence, and natural reading order for general pages.
  - Tables: check cell value relationships; supports Markdown and HTML, including row/colspan where possible.
  - Math formulas: render LaTeX with KaTeX, collect symbol boxes, and verify relative symbol placements in outputs (a robust, tokenization-agnostic check).
- Scale and coverage: 1,402 PDFs, 7,010 unit tests across categories including arXiv Math (AM), Old Scans (OS, OSM), Tables (TA), Headers/Footers (HF), Multi-Column (MC), Long Tiny Text (LTT) (Table 3; §3.2).
- Scoring: macro-average pass rates across sources to avoid dominance by any single source (Section 3.3).

## 4. Key Insights and Innovations
- Document-anchoring as a generalizable prompt-time scaffold (fundamental)
  - Different from image-only OCR, this approach fuses PDF internals with the page image. It reduces hallucination and improves linearization without bespoke layout modules (Appendix A; Figure 3).
  - Impact: measurable gains across models; e.g., GPT‑4o improves from 68.9% to 69.9% overall on the benchmark, and the fine-tuned model from 74.7% to 75.5% when anchoring is enabled (Table 4).

- A unit-test–style benchmark that is model- and tokenization-agnostic (fundamental)
  - Pass/fail tests for presence/absence/order, plus structural checks for tables and formula geometry (Section 3.1). This avoids pitfalls of edit-distance metrics and LLM judges, and enables fair comparisons across heterogeneous systems.

- Specializing a small open VLM beats larger general models on this task at far lower cost (substantial)
  - The fine-tuned 7B model outperforms GPT‑4o, Gemini Flash 2, Qwen‑2.5‑VL, and dedicated tools like Mistral OCR on overall pass rate (Table 4) while costing ~$176 per million pages on L40S GPUs versus $12,480 for GPT‑4o (Table 6; Figure 1).

- Downstream evidence that better linearization improves LM pretraining (substantial)
  - Replacing the same PDFs in peS2o with olmOCR outputs yields +1.3 average points across common LM benchmarks after 50B continued pretraining tokens (Table 5), suggesting practical gains in model quality.

## 5. Experimental Analysis
- Evaluation setup
  - Systems produce plain text or Markdown/HTML from each PDF; tests are deterministic pass/fail; macro-averaged over document sources (Section 3.3). In addition to the benchmark, there is a cost analysis (Appendix B, Table 6), alignment analysis with the teacher model (Appendix C.1, Tables 7–8), and a pairwise human ELO study (Appendix C.2; Figure 7; Table 9).

- Main quantitative results
  - Overall accuracy (macro pass rate; Table 4):
    > Ours (Anchored): 75.5% ±1.0  
    > Ours (No Anchor): 74.7% ±1.1  
    > Mistral OCR: 72.0% ±1.1  
    > Marker v1.7.5: 70.1% ±1.1  
    > GPT‑4o (Anchored): 69.9% ±1.1; (No Anchor): 68.9% ±1.1  
    > Gemini Flash 2 (Anchored): 63.8% ±1.2  
    > Qwen‑2.5‑VL: 65.5% ±1.2  
    > MinerU: 61.5% ±1.1

  - Category highlights (Table 4):
    - Math-heavy pages (AM, OSM): Ours reaches 74.9% (AM) and 71.2% (OSM) vs GPT‑4o Anchored 53.5% and 74.5%; the method is particularly strong on arXiv math and competitive on old scanned math.
    - Multi-column reading order (MC): 78.3% vs GPT‑4o Anchored 69.3%.
    - Headers/footers exclusion (HF): high across strong systems; Ours 94.5%.
    - Tables (TA): 71.0% for Ours; GPT‑4o Anchored 70.0%; Mistral OCR 60.6%.

  - Cost and throughput (Table 6):
    > “Cost per Million Pages” (log scale in Figure 1):  
    > olmOCR: $176 (L40S), $178 (H100)  
    > Gemini Flash 2 Batch: $249  
    > MinerU: $596  
    > Mistral OCR API: $1,000  
    > Marker (force_ocr on H100): $1,484  
    > GPT‑4o Batch: $6,240; non-batch: $12,480  
    Throughput measured at 906 tokens/s on L40S and 3,050 tokens/s on H100; observed 12% retry rate for olmOCR.

  - Downstream pretraining (Table 5):
    > Average across tasks improves from 53.9 → 55.2 on an OLMo‑2‑7B checkpoint trained for +50B tokens on the same PDFs processed with olmOCR vs prior Grobid+rulings.  
    > Notably, HellaSwag jumps 57.4 → 62.6; ARCC rises 75.0 → 76.4; some tasks are flat or slightly down (e.g., NQ 29.4 → 29.1).

  - Faithfulness to teacher model (Appendix C.1):
    > Alignment score (word-level alignment via Hirschberg): olmOCR 0.875 vs GPT‑4o self-alignment 0.954 and GPT‑4o mini 0.833 (Table 7). Most pages fall into medium/high alignment bins (Table 8), indicating stable imitation without heavy drift.

  - Human pairwise preference (Appendix C.2):
    > ELO ranking places olmOCR >1800, above Marker, MinerU, and GOTOCR (Figure 7). Pairwise wins: 61.3% vs Marker, 71.4% vs MinerU, 58.6% vs GOTOCR (Table 9).

- Do the experiments support the claims?
  - Yes, via three complementary angles:
    - A rigorous unit-test benchmark across diverse failure modes (math structure, tables, order, headers/footers).
    - Cost/throughput measurements with explicit pricing and hardware assumptions (Appendix B; Table 6).
    - Downstream pretraining gains on a widely used scientific corpus (Table 5).
  - Ablations/controls:
    - Anchored vs non-anchored prompts across several models (Table 4).
    - Full fine-tune vs LoRA validation curves (Appendix C).
    - Teacher alignment and temperature effects (Tables 7–8).
  - Failure/robustness:
    - Explicit handling of retries, rotation, and decoding collapse; reported 12% retries (Appendix D.2; Table 6).
    - Baseline tests detect degenerate outputs (Section 3.1, “Baseline” checks).

- Mixed or conditional findings
  - Anchoring helps most models but the gain size varies by model (Table 4).  
  - Downstream improvements are not uniform across all tasks (Table 5), though the average rises.

## 6. Limitations and Trade-offs
- Reliance on silver labels from GPT‑4o (§2.2)
  - Supervision is generated, not human-verified at scale. Despite structural prompts and anchoring, residual biases or subtle errors may exist; the alignment analysis shows high but not perfect fidelity (Table 7).

- Anchoring benefits depend on born-digital metadata (Appendix A)
  - When PDFs are pure scans, anchors vanish and the system reverts to image-only behavior. Results on old scans remain strong (e.g., OSM), but anchoring’s core advantage is reduced.

- Output format and structural fidelity
  - Table tests benefit from HTML with row/colspans; when only Markdown is produced, some structure may be lost (§3.1). The toolkit outputs plain text/Markdown/JSON; full semantic HTML for complex layouts is not guaranteed.

- Decoding stability and retries (Appendix D.2)
  - Without hard-constrained JSON decoding (intentionally avoided for quality), occasional parse failures or repetitions occur; mitigated by retries and temperature but at a throughput cost (12% retry rate measured).

- Language scope and domain coverage
  - Non-English documents are filtered during data creation (§2.1). Multilingual generalization is not evaluated.

- Benchmark scope
  - Pass/fail unit tests are precise but inevitably partial. They emphasize correctness on presence/order/tables/formulas; other aspects (e.g., figure captions vs text trade-offs) may need additional tests.

- Compute and infra
  - Fine-tuning uses 8×H100 for a short time (§2.3), which is modest for labs but non-trivial for small teams. Inference is efficient but still requires GPUs for peak throughput (Table 6).

## 7. Implications and Future Directions
- Field impact
  - Demonstrates that a small, task-specialized open VLM, when grounded by document-anchoring, can outperform larger general models on PDF linearization while being orders of magnitude cheaper (Figure 1; Table 4, Table 6). This rebalances the ecosystem away from closed APIs and toward reproducible, large-scale open pipelines.

- Practical applications
  - Massive corpus construction and cleaning for LM pretraining (e.g., reprocessing peS2o, as shown in §4.2; Table 5).  
  - Enterprise document processing at scale (legal, financial, government archives) where data cannot leave controlled infrastructure.  
  - Research tools for scientific PDFs, enabling better retrieval-augmented generation and reading assistance by providing faithful reading-order text.

- Follow-up research
  - Multilingual and domain expansion: include non-English PDFs and specialized domains (e.g., forms, code-heavy docs).  
  - Richer outputs: consistent HTML with reliable table row/colspans, figure references, cross-page footnotes, and improved math LaTeX fidelity.  
  - Better decoding safety: early detection/termination of repetitive generations (Appendix D.2 notes planned work).  
  - Teacher diversification: distill from ensembles (multiple VLMs) to reduce bias from a single teacher’s errors.  
  - Stronger benchmarks: broaden unit tests to cover cross-page references, figure-text alignment, and document-level tasks beyond single-page checks.

Overall, this work delivers a practical, open, and rigorously evaluated pathway to unlock high-quality text from PDFs at scale. The combination of document-anchoring, targeted fine-tuning, and unit-test evaluation provides both a methodological template and an operational toolchain that downstream LM researchers and practitioners can adopt immediately.
