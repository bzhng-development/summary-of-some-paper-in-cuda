# Building and better understanding vision‑language models: insights and future directions

**ArXiv:** [2408.12637](https://arxiv.org/abs/2408.12637)
**Authors:** Hugo Laurençon, Andrés Marafioti, Victor Sanh, Léo Tronchon
**Institutions:** 

## 🎯 Pitch

This paper offers a dual contribution: a comprehensive map of design choices for vision-language models (VLMs) and the creation of a high-performance model, `Idefics3-8B`, trained on open data. By using straightforward techniques like pixel shuffle and image tiling, coupled with a new large-scale dataset `Docmatix`, it significantly enhances document understanding capabilities, providing a scalable, open solution that can revolutionize enterprise applications in document automation and visual data analytics.

---

## 1. Executive Summary (2-3 sentences)
This paper has two intertwined contributions: (1) a tutorial-style, evidence-based map of design choices for vision-language models (VLMs) covering architectures, data, training, and evaluation; and (2) a practical instantiation, `Idefics3-8B`, trained only on open data with a simple pipeline that markedly improves document understanding. A new large-scale dataset, `Docmatix`, enables the model’s 13.7-point gain on DocVQA and offers a scalable recipe for synthetic multimodal data at enterprise-relevant scale (Figure 5, Table 2).

## 2. Context and Motivation
- Problem/gap addressed
  - VLMs—models that take images+text and generate text—have rapidly diverged in architecture, data, and training methods with little consensus or ablations clarifying trade-offs. The paper addresses this fragmentation by organizing the design space and demonstrating a strong open VLM built with those insights (Introduction; Sections 2–5).
- Why this matters
  - VLMs underpin real applications: document and figure understanding, visual math, and screenshot-to-code (Introduction). Organizations need reliable, open, and compute-efficient approaches that preserve text-only capabilities while scaling visual understanding.
- Prior approaches and shortcomings
  - Architectural split: `cross-attention` models (e.g., Flamingo, Llama 3‑V) vs. `self-attention` models that treat vision features as tokens and concatenate them with text (e.g., BLIP‑2, LLaVA). Section 2.1 discusses how these choices affect frozen vs. unfrozen training dynamics (Section 2.1.1–2.1.3; Figure 1).
  - Data: Abundant but noisy image–alt-text pairs; scarce instruction-tuned multimodal data; little scalable document QA data; limited systematic filtering and deduplication (Section 3.1; Figure 3).
  - Training: Multi-stage pipelines are common but under-specified; instability and compute limits force compromises; pretraining ablations rarely predict post‑SFT performance (Sections 3 and 4.2).
  - Evaluation: Open-ended benchmarks favor specific answer styles; risk of contamination in popular suites like MathVista; few-shot and judge-based evaluations are underused (Section 4.1–4.3).
- Positioning of this work
  - Provides a synthesis of the design space with concrete, ablated insights (e.g., architecture trade-offs in Section 2.1.3).
  - Introduces `Docmatix` for scalable document QA (Section 5.1.2; Figure 4; Table 2).
  - Builds `Idefics3-8B`, an open, efficient VLM centered on OCR-intensive tasks using simple components (pixel-shuffle connector and image tiling) plus a staged training recipe (Sections 5.2.1–5.2.2; Table 3; Figure 5).

## 3. Technical Approach
This section explains HOW the model and data pipeline work.

- Core components
  - Backbones
    - Vision encoder: `SigLIP-SO400M` (400M parameters), chosen for strong performance/parameter tradeoff (Section 2.1.4).
    - Language model: `Llama 3.1 Instruct` (Section 5.2.1), replacing Mistral‑7B used in Idefics2.
  - Architecture style: `self‑attention` (fully autoregressive) where visual tokens are concatenated with text tokens and fed into the LLM (Section 2.1.2; Figure 1).
    - Term: `modality projection` refers to mapping vision features into the LLM embedding space.

- Connector and visual tokenization
  - From Idefics2 to Idefics3
    - Idefics2 used a `perceiver resampler` (a cross-attention module that compresses vision tokens into a small set of learnable latent tokens) down to 64 tokens per image (Section 2.2.2).
    - Idefics3 replaces this with a `pixel shuffle` pooling strategy (as in InternVL‑1.5), which reduces the number of image hidden states by a factor of 4 and yields 169 visual tokens per 364×364 image (Section 5.2.1).
      - Definition: `pixel shuffle` reorganizes features across channels and spatial dimensions; here it acts like downsampling that preserves local structure while keeping more spatial detail than heavy token compression.

- Scaling to higher-resolution images
  - `Image-splitting strategy` (Section 2.2.3): Each input image is divided into tiles of 364×364; each tile is encoded separately by the shared vision encoder. This scales the number of visual tokens with the original resolution without changing the encoder’s native resolution.
  - Preserving 2D structure despite linear token sequences
    - Tiling metadata: prepend text tokens `"<row_x_col_y>"` to each tile, insert `"\n"` after each row, and append a single downscaled whole image (364×364) to preserve global context (Section 5.2.1).
    - Why: When visual tokens are concatenated, positional relationships can be lost; these textual hints restore approximate layout.

- Data pipeline and training curriculum
  - Multi-stage pretraining (Section 3; Figure 2; Table 3)
    - Stage 1 (frozen backbones): learn the connector on low-to-moderate resolution images; data includes `OBELICS` (interleaved web documents) and `LAION COCO` (recaptioned image–text pairs).
    - Stage 2 (DoRA/LoRA on backbones): progressively increase maximum resolution (364² → 1820²) and introduce `PDFA` (large document images with OCR text).
      - Definition: `LoRA` is low‑rank adaptation; `DoRA` decomposes weights for more stable and effective LoRA-style training (Table 3).
    - Stage 3 (DoRA): high-quality, task-oriented synthetic data focused on skills users actually want (e.g., document QA, image captioning, LNQA for real‑world VQA, ChartGemma, WebSight; Table 3).
  - Supervised fine-tuning (SFT)
    - Instruction mixture: `The Cauldron` expanded to 56 datasets spanning OCR, charts, tables, reasoning, and “screenshot→code,” plus text-only instruction sets to preserve LLM skills (Section 5.1.1; Table 1).
    - Regularization: `NEFTune` noise added to input embeddings during SFT to improve robustness; loss is computed only on answer tokens; learning rate linearly decayed to zero (Section 5.2.1).
      - Definition: `NEFTune` injects small noise in token embeddings during fine-tuning to reduce overfitting and increase instruction-following robustness.
  - Training logistics
    - Sequence length 10K; batch size 1024; training completed in ~5 days on 32 H100 nodes, including restarts (Table 3).

- Building the Docmatix dataset (Section 5.1.2; Figure 4; Table 2)
  - Problem: existing open document VQA datasets are tiny (e.g., DocVQA 10K images, 40K QAs).
  - Method
    - Start with `PDFA` text transcriptions (English).
    - Use `Phi‑3‑small` to generate diverse QAs from text, driven by five prompts.
    - Filter low-quality pairs: remove code-like outputs via regex; discard answers containing “unanswerable”; ~15% removed (Figure 4).
    - Render PDFs to images (DPI=150), producing aligned image–QA pairs.
  - Outcome: `Docmatix` includes 2.4M images and 9.5M QA pairs from 1.3M PDFs—roughly 240× larger than prior open alternatives (Section 5.1.2).

## 4. Key Insights and Innovations
- Scalable, open, OCR-centric VLM with simple components
  - What’s new: swapping the resampler for `pixel shuffle` + systematic `image tiling` + lightweight textual layout tags recovers much more per-image detail (169 tokens vs. previous 64), crucial for text-heavy documents (Section 5.2.1).
  - Significance: drives large DocVQA gains without complex modules; retains compute efficiency because the vision encoder is reused over tiles.

- Docmatix: a data recipe rather than just another dataset
  - What’s new: converts OCR’d PDF text to large-scale synthetic QA using small LLMs with simple filters (Section 5.1.2; Figure 4).
  - Significance: an open, replicable path to high-quality document QA at scale. Table 2 shows a 60.1 → 71.4 ANLS jump on DocVQA for a 700M‑parameter `Florence‑2` when trained on a small Docmatix subset plus DocVQA formatting pass.

- A staged training curriculum that mirrors how capabilities emerge
  - What’s new: explicit sequencing—frozen connector learning → low-rank adaptation with larger images → targeted synthetic pretraining → SFT—with clear data choices and resolutions (Table 3; Figure 2).
  - Significance: codifies a practical, reproducible pipeline that others can adapt, using only open datasets.

- Evidence-guided architectural guidance rather than dogma
  - What’s new: Section 2 compares `cross-attention` vs. `self-attention` in realistic regimes: cross‑attention excels when backbones are frozen; self‑attention can surpass it once some backbone layers are tuned with LoRA (Section 2.1.3).
  - Significance: clarifies when each architecture is appropriate, informing compute- and data-limited settings.

## 5. Experimental Analysis
- Evaluation setup (Section 5.2.2; Appendix A.1)
  - Benchmarks and metrics
    - `MMMU (val)`—college-level multimodal reasoning; metric: MMMU score.
    - `MathVista (testmini)`—visual math; metric: MMMU-style score.
    - `MMStar (val)`—general image understanding; metric: accuracy.
    - `DocVQA (test)`—document QA; metric: `ANLS` (Average Normalized Levenshtein Similarity; higher is better).
    - `TextVQA (val)`—reading text in natural images; metric: VQA accuracy.
  - Inference resolution
    - Idefics3: longest side resized to 4×364; DocVQA uses 5×364 (max seen during training). Idefics2‑70B uses 1960 pixels longest side (Section 5.2.2).

- Main results (Figure 5)
  - Idefics3‑8B vs Idefics2‑8B (absolute gains)
    - MMMU: 46.6 vs 45.2 (+1.4).
    - MathVista: 58.4 vs 52.2 (+6.2).
    - MMStar: 55.9 vs 49.5 (+6.4).
    - DocVQA: 87.7 vs 74.0 (+13.7).
    - TextVQA: 74.9 vs 73.0 (+1.9).
  - Idefics3‑8B vs Idefics2‑70B
    - Idefics3 surpasses 70B on DocVQA (87.7 vs 84.1) while trailing on MMMU (46.6 vs 58.0), MMStar (55.9 vs 58.1), and TextVQA (74.9 vs 77.3) (Figure 5).

- Category breakdown (Table 4, MMMU)
  - Strongest categories include Literature (80.0), Art Theory (76.7), and Design (73.3); weakest include Math (26.7), Physics (26.7), Materials (26.7), and Music (26.7). This pattern suggests the model benefits from linguistic context and visually grounded humanities but still lacks deep technical domain knowledge.

- Ablation: Docmatix efficacy (Table 2)
  - Quote the key comparison:
    > Florence‑2 (700M) trained on `DocVQA` achieves 60.1 ANLS.  
    > The same model trained on a small `Docmatix` subset (+ one epoch on DocVQA for evaluation formatting) reaches 71.4 ANLS.  
    > For reference, Idefics2‑8B with a general mixture achieves 74.0 ANLS.
  - Interpretation: high-quality synthetic document QA materially improves performance even for smaller models.

- Evaluation methodology and prompts (Appendix A.1)
  - The paper standardizes MCQ prompting (“Answer with the letter”) and uses Gemini-like concise prompts for TextVQA and DocVQA, controlling for stylistic biases that plague open-ended setups (Appendix A.1.1).

- Do the experiments support the claims?
  - Yes for document understanding: large, consistent improvements on DocVQA and competitive TextVQA gains line up with the design choice to increase visual tokens and employ tiling (Section 5.2.1; Figure 5).
  - Mixed for broad knowledge and reasoning: Idefics3‑8B closes some gap on MathVista and MMStar but remains behind a 70B model on MMMU (Figure 5), consistent with the paper’s note that scale is vital for knowledge-heavy tasks (Section 5.2.2).

- Robustness and evaluation caveats (Section 4)
  - Open-ended metrics can reward models that match expected answer styles rather than true understanding; multiple-choice reduces ambiguity but still has distributional quirks (Section 4.1).
  - Risk of contamination—e.g., MathVista contains items overlapping common SFT datasets—means scores can overstate generalization if training data isn’t carefully filtered (Section 4.3).
  - Pretraining ablations may mislead: benefits (e.g., more visual tokens) often surface only after instruction data is included or after SFT (Section 4.2).

## 6. Limitations and Trade-offs
- Architectural trade-offs
  - `Image tiling` encodes tiles independently, risking loss of global context. The paper mitigates this by appending a downscaled whole image and injecting row/column markers, but acknowledges this is imperfect for fine cross-tile relations (Section 2.2.3; 5.2.1).
  - `Pixel shuffle` increases token count but still compresses; tasks requiring extremely fine OCR or dense small text may need even more tokens or native variable‑resolution encoders (Sections 2.2.2–2.2.3).

- Training choices constrained by efficiency (Section 5.2.1, “Opportunities for improvement”)
  - Backbones are adapted via DoRA/LoRA rather than fully unfrozen; full unfreezing could yield better performance but at increased compute and instability risk.
  - Early stage transitions happen before loss convergence to save compute; Stage 3 uses only a fraction of available synthetic data.

- Data limitations
  - `Docmatix` uses automatic QA generation with simple heuristics to filter errors (~15% dropped), which can leave residual noise. No large-scale human auditing is reported (Section 5.1.2).
  - While the training mixture includes text-only instruction datasets (Table 1), the paper does not report text-only benchmark scores; subtle regressions in pure NLP tasks are possible without dedicated evaluation (Section 2.1.3 notes this as a general concern).

- Evaluation scope and fairness
  - Some benchmarks (e.g., VQAv2) are sensitive to short-answer formats; scores can be inflated/deflated by prompt style and exposure during SFT (Section 4.1).
  - The paper itself warns of contamination risk (MathVista) and encourages excluding benchmark items from SFT (Section 4.3), but end-to-end decontamination evidence is not presented.

- Usability considerations
  - The model was “mainly trained on short answers” and lacks a full preference-alignment stage; the paper reports occasional instruction-following issues for complex prompts (Figure 6 caption).

## 7. Implications and Future Directions
- How this work shifts the field
  - It demonstrates that an open, compute-conscious pipeline can reach and sometimes surpass much larger models in targeted domains (DocVQA), when the architecture and data are tuned to the task (Figure 5, Table 2).
  - It reframes synthetic data generation for multimodality as a practical, modular workflow: start from structured corpora (PDFs with OCR), use small LLMs for QA generation, and apply lightweight filtering (Section 5.1.2; Figure 4).

- Research directions enabled/suggested
  - Variable‑resolution, long‑context vision encoders that avoid tiling (Section 2.2.3 proposes Patch’n’Pack/NaViT-like approaches).
  - Smarter connectors that preserve 2D structure with fewer tokens (e.g., hybrid pooling with 2D positional priors or learned compression that’s OCR-aware).
  - Systematic data selection and deduplication for multimodal corpora, analogous to FineWeb‑Edu and SNIP/SemDeDup for text/images (Section 3.1).
  - Alignment methods specialized for VLMs (preference data, hallucination reduction like RLHF‑V/RLAIF‑V; Section 3.2), addressing the short-answer bias observed in SFT.
  - Cleaner evaluation: few-shot or judge‑based scoring (e.g., LAVE), improved decontamination protocols, and reporting both multimodal and text-only scores (Sections 4.1–4.3).

- Practical applications
  - Document automation: enterprise search, form extraction, invoice/contract Q&A—Docmatix + Idefics3 deliver state-of-the-art open performance for such tasks (Figure 5; Section 5.1.2).
  - Code generation from UI screenshots and UX analysis (WebSight data; Figure 6).
  - Analytics over charts, tables, and figures where mixed symbolic–visual reasoning is required (Table 1 coverage; ChartGemma, TabMWP, etc.).

> Bottom line: By pairing a transparent design-space tutorial with a concrete open model and a scalable dataset recipe, the paper provides both a blueprint and a benchmark for building practical VLMs. The strongest gain—document understanding—comes not from exotic modules but from careful token budgeting (pixel shuffle + tiling) and targeted synthetic data, a pattern future work can generalize to other modalities and tasks.
