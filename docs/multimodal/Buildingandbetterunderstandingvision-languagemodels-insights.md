# Building and better understanding vision-language models: insights and future directions

**ArXiv:** [2408.12637](https://arxiv.org/abs/2408.12637)

## 🎯 Pitch

This paper delivers both a comprehensive, evidence-based tutorial on building vision-language models (VLMs) and a practical demonstration by introducing Idefics3-8B, a powerful open VLM trained efficiently using only open datasets and a straightforward pipeline. By distilling architectural, data, training, and evaluation choices—and releasing the massive Docmatix dataset for document understanding—the authors chart a clear, transparent path for the community to build robust and high-performing VLMs, directly advancing accessible research and impactful real-world applications such as document processing and visual reasoning.

---

## 1. Executive Summary
This paper does two things: (1) it distills the current design space of vision–language models (VLMs)—covering architectures, data, training, and evaluation—into a practical, evidence-based tutorial; and (2) it demonstrates the guidance in action by building `Idefics3-8B`, an open VLM trained with a simple, efficient pipeline and a new large-scale document-understanding dataset, `Docmatix`. The result matters because it clarifies which choices actually move performance and shows that strong, open VLMs can be built with transparent methods and data, especially for high-value tasks like document understanding.

## 2. Context and Motivation
- Problem/gap addressed
  - The field lacks consensus on core VLM design choices—how to connect vision encoders to language models, what data to use when, how to train, and how to evaluate fairly (Section 1). Many papers do not ablate or justify choices, making it hard to know what truly improves performance or efficiency.
- Why this matters
  - VLMs power applications such as document and figure understanding, visual math, and converting website screenshots to code (Section 1). Solidifying best practices improves both research efficiency and practical system building.
- Prior approaches and shortcomings
  - Architectures: Two dominant families exist—cross-attention (e.g., Flamingo) and self-attention/fully-autoregressive (e.g., LLaVA). Head-to-head comparisons are sparse and sometimes confounded by different training strategies (Section 2.1.3).
  - Data: Massive web image–alt-text corpora are common but noisy; interleaved image–text documents (web pages) help few-shot and text-only abilities but are newer and less standardized (Section 3.1). Document images (PDFs) are under-served by open datasets, limiting progress on business-critical tasks (Section 5.1.2).
  - Training: Multi-stage pipelines exist but vary widely (freezing vs. unfreezing, LoRA usage, resolution schedules), and lack shared recipes (Figure 2; Section 3).
  - Evaluation: Open-ended benchmarks are format-sensitive; contamination and over-optimization are real; and pretraining-time results often mispredict post-finetuning behavior (Section 4).
- Positioning of this work
  - A tutorial-style mapping of the design space, with concrete recommendations and evidence, followed by a complete, open build of `Idefics3-8B` (Sections 5.1–5.2) that significantly improves document understanding via the new `Docmatix` dataset (Section 5.1.2; Table 2; Figure 5).

## 3. Technical Approach
This section explains (A) the architectural options and trade-offs, (B) the data and training pipeline, and (C) the concrete instantiation in `Idefics3`.

A) Architectural options and how they work (Section 2)
- Two ways to fuse vision with language:
  - `Cross-attention` (Section 2.1.1): Insert new cross-attention layers between frozen LLM blocks. Language tokens query visual features; keys/values come from vision. In Flamingo-style systems, one cross-attn block every four LLM blocks adds ~25% of LLM parameters as new modules. Because the LLM remains frozen, text-only skills are preserved while the new layers learn to inject vision.
  - `Self-attention` / `fully-autoregressive` (Section 2.1.2): Convert image features to “visual tokens,” optionally compress them, concatenate them with text tokens, and pass the combined sequence through the LLM. A `modality projection` maps the vision feature space into the LLM’s token space; optional pooling reduces sequence length (Figure 1).
- Key empirical insight on when each is better (Section 2.1.3):
  - With both backbones frozen, cross-attention wins (more expressive new layers).
  - Once you train parts of the backbones (e.g., with LoRA), self-attention catches up and surpasses cross-attention despite having fewer new parameters (Section 2.1.3; based on a controlled study with Mistral-7B + SigLIP-SO400M).
- Other architectural choices (Section 2.2):
  - Is a separate vision encoder necessary? `Fuyu` directly projects image patches into the LLM without a vision encoder (Section 2.2.1). This preserves raw details but performed worse than using a strong pre-trained vision encoder; training cost can be higher to recover the lost prior.
  - How to compress visual tokens efficiently:
    - Simple linear projection (no pooling) retains all encoder features but is long-sequence and slow.
    - Efficient compressors like the `perceiver resampler` (learnable latent queries cross-attend to image features) can compress 77× to ~64 tokens with small loss, except for OCR-heavy tasks (Section 2.2.2).
    - 2D-aware compressors (e.g., Conv-based H-Reducer, pixel shuffle) keep spatial structure and often help OCR (Section 2.2.2).
  - `Image-splitting` (tiling) (Section 2.2.3): Split large images into tiles processed at native encoder resolution; optionally add a downscaled full image to retain global context; insert layout hints (“row/col” markers) in the token stream. This flexibly trades computation for OCR fidelity at inference.

B) Data and training pipeline (Sections 3 and 5.1)
- Staged pretraining (Figure 2; Section 3):
  1) Freeze backbones, learn the connector at low image resolution for stability/efficiency.
  2) Gradually unfreeze or adapt backbones (often with LoRA/DoRA) and increase resolution.
  3) Use higher-quality and synthetic data that match downstream tasks.
- Data types (Section 3.1; Figure 3):
  - Image–text pairs from the web (e.g., LAION family); often re-captioned to reduce noise.
  - Interleaved image–text documents (e.g., `OBELICS`): boosts few-shot and text-only performance and teaches models to handle many images per context.
  - PDFs with OCR text (`PDFA`, `OCR-IDL`): crucial for document understanding; enables transcription tasks.
  - Synthetic datasets targeted at hard skills: dense image captioning, real-world VQA, OCR-in-the-wild, charts/tables, chain-of-thought, visual math, and screenshot-to-code (Section 3.1).
- Supervised fine-tuning (SFT) + optional alignment (Section 3.2):
  - Curated “academic” mixtures with unified Q/A formatting (e.g., `The Cauldron`) are effective but tend to encourage short answers; answer expansion can mitigate (Table 1 notes coverage and mixture weights).
  - Alignment (e.g., DPO) can reduce hallucinations and improve instruction-following, but `Idefics3` does not include a full alignment stage (Section 5.2.2, “Qualitative evaluation”).

C) `Idefics3-8B`: concrete design and training (Section 5)
- Backbones (Section 5.2.1):
  - Vision: `SigLIP-SO400M` (strong open encoder; good accuracy/parameter ratio).
  - Language: `Llama 3.1 Instruct` (replaces Mistral-7B used in Idefics2).
- Connector and tokenization of images (Section 5.2.1):
  - Replace the perceiver resampler (used in Idefics2) with a `pixel shuffle` compressor that reduces the vision feature map by 4× and yields 169 visual tokens per 364×364 tile.
  - Use `image-splitting`: split an image into 364×364 tiles; each tile produces 169 tokens. To preserve layout:
    - Prepend textual tags like `<row_x_col_y>` per tile.
    - Insert the newline token `\n` at tile-row boundaries.
    - Append a downscaled whole image (364×364) to maintain global context.
- Training schedule and data (Table 3):
  - Three pretraining stages and then SFT; sequence length 10k; batch size 1024.
  - Resolution increases from 364² up to 1820² by the end of pretraining.
  - Backbones: frozen in Stage 1, adapted with `DoRA` (a variant of LoRA) thereafter.
  - Data by stage:
    - Stage 1: OBELICS, LAION-COCO.
    - Stage 2: add `PDFA` (PDF images) as resolution grows.
    - Stage 3: large synthetic sets (`Docmatix`, `WebSight`, `LNQA`, `PixelProse`, `ChartGemma`).
    - SFT: the extended `Cauldron` (Table 1 lists datasets, sizes, and mixture shares).
  - SFT details: apply `NEFTune` input noise; compute loss only on answer tokens; cosine decay to zero LR in the final pretraining stage and SFT (Section 5.2.1).
  - Compute: total end-to-end training completed in 5 days on 32 H100 nodes (Section 5.2.1).
- `Docmatix` dataset construction (Section 5.1.2; Figure 4; Table 2):
  - Start from `PDFA` English transcriptions; generate Q/A pairs with `Phi-3-small` using 5 prompt templates; filter out low-quality pairs (regex-based code removal, “unanswerable” rejection), discarding about 15%.
  - Scale: 2.4M images and 9.5M Q/A pairs from 1.3M PDFs (Section 5.1.2).
  - Ablation (Table 2): training Florence-2 on a small `Docmatix` subset (20% images, 4% Q/A, 1 epoch) + 1 epoch on DocVQA yields 71.4 ANLS vs 60.1 when trained on DocVQA alone—a ~19% relative improvement.

## 4. Key Insights and Innovations
- A practical, evidence-backed map of the VLM design space
  - What’s new: a side-by-side analysis of cross-attention vs self-attention conditioned on training regime (freezing vs LoRA/DoRA tuning) (Section 2.1.3).
  - Why it matters: it explains conflicting reports in the literature—cross-attention shines with frozen backbones, but once you train the backbones, self-attention’s efficiency and parameter sharing win out.
- `Docmatix`: scalable document-understanding supervision (Section 5.1.2; Figure 4; Table 2)
  - What’s new: a simple, replicable pipeline to turn OCR’d PDFs into diverse Q/A pairs with LLMs at unprecedented open-source scale (2.4M images; 9.5M Q/A).
  - Why it matters: document understanding is a high-value but under-resourced domain. Table 2 shows that even a small slice of `Docmatix` provides a large boost on DocVQA.
- An OCR-friendly visual token strategy with minimal complexity (Section 5.2.1)
  - What’s new: replacing the perceiver resampler with `pixel shuffle` plus tiling and explicit 2D-position text tags (`<row_x_col_y>`), newline separators, and a downscaled full image.
  - Why it matters: it increases the number of visual tokens per image (169 per tile) to capture fine print needed for OCR-heavy tasks, while remaining simple and efficient.
- A transparent, reproducible, multi-stage training recipe (Table 3; Section 5.2.1)
  - What’s new: a concise schedule with resolution ramp-up, frozen-then-DoRA adaptation, dataset transitions from generic to synthetic, and SFT with NEFTune.
  - Why it matters: it is simple enough to reproduce, yet achieves meaningful improvements over Idefics2 with only open data (Figure 5).

## 5. Experimental Analysis
- Evaluation setup (Section 5.2.2; Appendix A.1.1)
  - Benchmarks and metrics:
    - `MMMU` (val; multiple-choice; “MMMU score”)
    - `MathVista` (testmini; numeric answers; MMMU score)
    - `MMStar` (val; multiple-choice accuracy)
    - `DocVQA` (test; ANLS—Average Normalized Levenshtein Similarity; higher is better)
    - `TextVQA` (val; VQA accuracy)
  - Image sizing during evaluation: longest side to 4×364 px for most tasks; 5×364 px for DocVQA to match training max resolution (Section 5.2.2).
  - Prompts: standardized multiple-choice format; Gemini-like concise-answer prompts for TextVQA and DocVQA (Appendix A.1.1).
- Main results (Figure 5)
  - `Idefics3-8B` vs `Idefics2-8B`:
    - MMMU: 46.6 vs 45.2 (+1.4)
    - MathVista: 58.4 vs 52.2 (+6.2)
    - MMStar: 55.9 vs 49.5 (+6.4)
    - DocVQA: 87.7 vs 74.0 (+13.7)
    - TextVQA: 74.9 vs 73.0 (+1.9)
  - `Idefics3-8B` vs `Idefics2-70B` (scale effects):
    - MMMU: 46.6 vs 58.0 (−11.4; knowledge scale matters)
    - MathVista: 58.4 vs 59.8 (close)
    - MMStar: 55.9 vs 58.1 (close)
    - DocVQA: 87.7 vs 84.1 (+3.6; `Idefics3` wins despite being smaller)
    - TextVQA: 74.9 vs 77.3 (slightly lower)
  - Takeaway: The new vision tokenization + synthetic doc data produce very large gains on document understanding; global reasoning/knowledge-heavy MMMU still benefits substantially from larger LMs.
- Category breakdown on MMMU (Table 4)
  - Strong areas: Literature 80.0, Art Theory 76.7, Design 73.3.
  - Weaker areas: Math 26.7, Physics 26.7, Materials 26.7, Mechanical Engineering 33.3.
  - Interpretation: OCR/document changes don’t directly address STEM reasoning depth; scale and targeted reasoning data likely needed.
- Ablations and diagnostics
  - `Docmatix` ablation with Florence-2 (Table 2) shows large relative gains on DocVQA with minimal additional training—evidence that large-scale, task-focused synthetic data is effective.
  - Training choices and compute: Table 3 documents that Stage 1–2 losses were not fully converged and Stage 3 used only fractions of available data—suggesting headroom (Section 5.2.1, “Opportunities for improvement”).
- Do the experiments support the claims?
  - Yes for document understanding: Figure 5 and Table 2 jointly show that the new data and tokenization strategy deliver outsized improvements on DocVQA.
  - Mixed for broad reasoning: Idefics3-8B narrows the gap to 70B models on MathVista/MMStar but still lags substantially on MMMU, supporting the claim that scale remains critical for that benchmark (Figure 5).
- Qualitative behavior (Figure 6)
  - The model can extract structured info from a CV, transform a screenshot to HTML (with Tailwind CSS), and summarize a paper from a screenshot. However, without an alignment stage and with SFT biased to short answers, instruction following can be brittle for complex prompts.

## 6. Limitations and Trade-offs
- Architectural and tokenization choices
  - `Image-splitting` + `pixel shuffle` increase token count and OCR fidelity but can still lose global context (Section 2.2.3). Adding a downscaled full image helps but is imperfect for fine global-local alignment.
  - Pixel shuffle’s fixed 4× reduction is simple but not task-adaptive; very dense text or complex layouts might still be under-tokenized.
- Training choices
  - Backbones were adapted with `DoRA` rather than fully unfrozen, primarily for efficiency; full unfreezing may improve results if stable (Section 5.2.1, “Opportunities for improvement”).
  - Early stages did not converge; Stage 3 used only a fraction of available synthetic data (Section 5.2.1). The reported numbers are thus conservative.
  - No formal alignment phase (e.g., DPO/RLHF-V), which likely limits instruction-following and safety/hallucination control (Section 3.2; Figure 6).
- Data assumptions
  - `Docmatix` relies on accurate OCR and LLM Q/A generation; while filtered, residual errors and style biases may remain (Section 5.1.2).
  - Potential benchmark contamination remains a field-wide concern; the paper urges careful exclusion during SFT (Section 4.3).
- Generalization limits
  - MMMU gap vs. 70B indicates that knowledge breadth and complex reasoning still scale with model size (Figure 5).
  - The paper does not address video or audio modalities, nor long-context memory beyond 10k tokens (Table 3).

## 7. Implications and Future Directions
- What changes in the field
  - A clearer, empirically grounded checklist for building open VLMs: start with strong backbones (LLM and vision encoder), choose self-attention if you will adapt backbones, compress visual tokens with 2D-aware methods, use tiling with layout tags for OCR, ramp up resolution, and introduce synthetic task-matched data late in pretraining—then SFT with a broad, formatted mixture (Sections 2–3; Table 3).
  - The value of large, targeted synthetic data is re-affirmed for vision-language tasks, not just pure text LLMs (Table 2; Section 3.1).
- Follow-on research enabled/suggested
  - Vision encoders that natively support arbitrarily large, variable-resolution images without tiling, possibly with Patch’n’Pack-like efficiency for long visual contexts (Section 2.2.3).
  - Stronger open vision encoders trained at scale (Section 2.1.4 notes today’s paucity relative to LLMs).
  - Task-adaptive visual token compressors that dynamically trade off tokens for fidelity per task/image.
  - Earlier introduction of instruction-style data in pretraining to reduce the pretrain–finetune discrepancy (Section 4.2).
  - Multimodal alignment and safety (DPO/RLAIF-V) to improve instruction-following and reduce hallucination (Section 3.2).
  - Robust evaluation: wider use of multiple-choice formats, LLM-as-judge metrics like `LAVE`, and rigorous decontamination (Sections 4.1–4.3).
- Practical applications
  - Document understanding at scale: contracts, invoices, forms, scientific articles, and reports—`Docmatix` demonstrates a path to large, open training corpora (Section 5.1.2; Figure 5’s DocVQA gains).
  - OCR-in-the-wild and chart/table analysis (Sections 3.1 and 5.1.1; Table 1 datasets).
  - UI and web understanding (screenshot-to-code via `WebSight`; Section 3.1 and Table 3) and potential grounding for GUI agents via bounding-box grounding datasets.

> Key quantitative highlight: “Idefics3-8B achieves 87.7 ANLS on DocVQA test, up from 74.0 for Idefics2-8B (+13.7), while remaining competitive on other benchmarks” (Figure 5).

Overall, this work offers both a practical guide for building VLMs and an open, high-performing model/data release that materially advances document understanding, while candidly exposing where scale and alignment still matter.
