# Molmo and PixMo: Open Weights and Open Data for State-of-the-Art Vision‑Language Models

**ArXiv:** [2409.17146](https://arxiv.org/abs/2409.17146)
**Authors:** Matt Deitke, Christopher Clark, Sangho Lee, Rohun Tripathi, Yue Yang, Jae Sung Park, Mohammadreza Salehi, Niklas Muennighoff, Kyle Lo, Luca Soldaini, Jiasen Lu, Taira Anderson, Erin Bransom, Kiana Ehsani, Huong Ngo, YenSung Chen, Ajay Patel, Mark Yatskar, Chris Callison‑Burch, Andrew Head, Rose Hendrix, Favyen Bastani, Eli VanderBilt, Nathan Lambert, Yvonne Chou, Arnavi Chheda, Jenna Sparks, Sam Skjonsberg, Michael Schmitz, Aaron Sarnat, Byron Bischoff, Pete Walsh, Chris Newell, Piper Wolters, Tanmay Gupta, Kuo‑Hao Zeng, Jon Borchardt, Dirk Groeneveld, Crystal Nam, Sophie Lebrecht, Caitlin Wittlif, Carissa Schoenick, Oscar Michel, Ranjay Krishna, Luca Weihs, Noah A. Smith, Hannaneh Hajishirzi, Ross Girshick, Ali Farhadi, Aniruddha Kembhavi
**Institutions:** Allen Institute for AI (AI2)

## 🎯 Pitch

Introducing Molmo, a groundbreaking suite of open vision-language models powered by the PixMo data suite, designed to rival top proprietary systems without reliance on distilled VLMs. Molmo's unmatched transparency and diverse multimodal capabilities set a new standard in reproducibility and accessibility, fostering advanced research and safety analysis in AI.

---

## 1. Executive Summary (2–3 sentences)
This paper introduces `Molmo`, a family of vision–language models (VLMs), and `PixMo`, a fully open, large-scale multimodal data suite designed to train them without distilling from proprietary VLMs. With a simple encoder–decoder architecture plus several targeted innovations in data, training, and inference (e.g., speech-driven dense captions, 2D pointing for grounding and counting, overlapping multi-crop encoding, and length-conditioned captioning), `Molmo-72B` achieves the highest average score across 11 academic benchmarks (81.2%) and ranks second in human preference just behind GPT-4o, while outperforming Claude 3.5 Sonnet and Gemini 1.5 Pro/Flash (Table 1).

## 2. Context and Motivation
- Problem/gap addressed
  - State-of-the-art VLMs remain proprietary, and many strong open-weight VLMs are trained using synthetic labels or captions created by those proprietary VLMs (i.e., effective distillation). This leaves the field without a fully open recipe for building high-performing VLMs from scratch and obscures which modeling and data choices matter most (Abstract; Introduction §1).
- Why it matters
  - Practically: Open, reproducible models and data enable research, auditing, safety analyses, and deployment without usage restrictions.
  - Scientifically: The community needs to know how to assemble effective training data (pretraining + instruction tuning) and which architectural/training choices improve real-world capabilities like OCR, counting, grounding, and fine-grained understanding (§§1, 3–6).
- Prior approaches and their limitations
  - Early open models like `LLaVA` released weights and data but lag in capability (§1).
  - Newer open-weight models (e.g., Qwen2-VL) are strong but often do not release full training data; or they rely heavily on synthetic data produced by closed VLMs (e.g., ShareGPT4V) (Introduction §1; Related Work §H).
  - Data collection for high-quality multimodal supervision is expensive and difficult to scale (§1).
- Positioning
  - This work offers an end-to-end open pipeline: released model weights, released training data, released code, and open evaluations (Table 1, bottom block; Figure 11 “VLM Openness Comparison”).
  - It also contributes a careful ablation analysis that isolates the impact of data and modeling choices (Tables 2–4, 15; Figure 9).

## 3. Technical Approach
This section explains the end-to-end system: data, model, training, and evaluation.

- High-level model design (Figure 2; §2)
  - Architecture: A standard “vision encoder + connector + decoder-only LLM” stack.
    - Preprocessor: Converts an input image to multi-scale, multi-crop inputs (one low-res global view + tiled high-res crops).
    - Vision encoder: A ViT (CLIP ViT-L/14 at 336 px by default; MetaCLIP and SigLIP also work) extracts per-patch features.
    - Connector: Concatenates features from two ViT layers (3rd-from-last and 10th-from-last), pools each 2×2 patch window via attention (query = mean of the window), then projects into the LLM embedding space with an MLP (§2 “Vision-language connector”; Table 2f).
    - LLM: A decoder-only language model; variants include `OLMo-7B`, `OLMoE-1B-7B` (Mixture-of-Experts, “1B-E”), `Qwen2 7B`, and `Qwen2 72B`.
  - Token arrangement: Vision tokens are ordered left-to-right, top-to-bottom, with special tokens marking image start/end and row ends (§2 “Arranging vision tokens”).

- Overlapping multi-crop image encoding (Figures 3 and 5; §2; Appendix A.1)
  - Problem: Fixed-resolution square ViTs lose context on crop borders.
  - Approach: Extract tiled crops with overlap; only non-overlapping central patches are passed to the LLM so the tiles exactly cover the image while border patches are always encoded with neighbor context (Figure 3).
  - Implementation: Resize the image to best fit a selected grid (minimizing up/down-scaling), pad to square for each crop, and add learned embeddings that indicate “real pixels vs padding” (§A.1).
  - Impact: Significant gains vs. no overlap (Table 2d).

- Training with multi-annotated images (Appendix A.3; §2 “Multi-annotated images”)
  - Many datasets contain multiple annotations per image (e.g., several QAs).
  - Mechanism: Concatenate all text annotations for the image in one sequence; use attention masks so each annotation attends to the image and itself but not other annotations.
  - Benefit: Avoids redundant image encoding, cutting image processing by ~2/3 and training time by over half, at the cost of ~25% longer sequences for their data mix (§2).

- Dropout strategy in pretraining (§2 “Dropout”; Table 2c)
  - Only apply residual dropout to text tokens (not to image tokens) during dense caption pretraining.
  - Intuition: Forces reliance on visual evidence rather than language priors.
  - Effect: Improves downstream captioning quality and broad benchmark average (Table 2c).

- PixMo data suite (Figure 1; §3; Appendix F, G)
  - Annotated datasets:
    - `PixMo-Cap`: 712k images with very long, detailed captions (avg ~196 words), collected via speech descriptions lasting 60–90 seconds; transcripts are cleaned/summarized by a text-only LLM. Also preserves audio “receipts” to ensure no VLM was used (Abstract; §3 “PixMo-Cap”).
    - `PixMo-AskModelAnything` (AMA): 162k QAs on 73k images; annotators collaborate with a text-only LLM guided by OCR output and `PixMo-Cap` captions to iteratively refine answers (§3).
    - `PixMo-Points`: 2.3M grounded annotations across 223k images, using 2D points (not boxes/masks) to mark entity instances and support counting; includes “not present” examples; also a “points as explanations” subset (79k annotations) (§3 “PixMo-Points”).
  - Synthetic datasets (no VLMs used):
    - `PixMo-CapQA`: 214k QAs from 165k images, generated from `PixMo-Cap` captions by a text-only LLM.
    - `PixMo-Docs`: 255k rendered documents/charts/tables/diagrams + 2.3M QAs, created via code generation in tools like Matplotlib, Plotly, LaTeX, HTML, Vega-Lite, Mermaid, Graphviz (Appendix F “PixMo-Docs”).
    - `PixMo-Clocks`: ~826k watch-face images with QA for time reading across ~50 bodies and ~160k faces (Figure 17; §3).
    - `PixMo-Count`: 36k training images created by running a standard detector, with manual validation sets; each has points (object centers) and a counting QA (Abstract; §3 “PixMo-Count”).
  - Design choice: Prioritize human-grounded, highly-detailed supervision (speech captions; 2D pointing) and targeted synthetic generators for skills missing in public data (clocks, charts, tables, diagrams).

- Pretraining and instruction tuning (§4; Figure 4; Table 7)
  - Pretrain all parameters on `PixMo-Cap` to generate either the cleaned “long caption” or a raw transcript; 90% of the time include a noisy “length hint” (integer 0–100) conditioning the desired output length (§4 “Pre-training”; §B.1 “Length conditioning”)—this improves both captioning and downstream accuracy (Table 2e).
  - No separate connector pretrain: Instead, use higher LR with short warmup for connector; keeps pipeline simple and matches/beat alternatives (Table 3b; §4).
  - Fine-tune on a mixed curriculum: combine `PixMo-*` datasets with many academic VQA/OCR/chart/table benchmarks (e.g., VQA v2, TextVQA, ChartQA, DocVQA, AI2D, etc.). Use “style tags” (e.g., prefix `vqa2:`) to constrain answer style where test sets require specific formatting (§4).
  - Up-weight pointing data (learns more slowly), down-weight very large synthetic datasets. Sampling roughly proportional to the square root of dataset size (Figure 4; Table 7).

- Pointing I/O format and chain-of-thought counting (§4 “For pointing”)
  - Representation: The model outputs normalized coordinates in plain text (0–100) using an HTML-like tag such as `<point x="..." y="..." alt="...">...</point>` (Appendix B.2).
  - Ordered points: For multiple instances, enforce top-down, left-to-right order; this makes counting a simple “last point index” (Table 4b).
  - Chain-of-thought: “Point-then-count” (first point to all instances, then output the count) improves counting vs. “count-only” or “count then point” (Table 4a).

- Implementation and compute (Appendix A.3; Table 8)
  - FSDP training in PyTorch with bfloat16 AMP, but keep weights and gradient reduction in float32 to stabilize loss (Figure 6).
  - Cautionary fix: Normalize gradients by the average number of loss tokens across devices (not per-device) to avoid bias toward short-answer examples (Appendix A.3).
  - Training time on H100s ranges from ~264 GPU-hours for `MolmoE-1B` pretrain to ~8.3k GPU-hours for `Molmo-72B` fine-tune (Table 8).

## 4. Key Insights and Innovations
- Fully open, high-detail multimodal data without VLM distillation (Figure 1; §3; Appendix F)
  - What’s new: Speech-based caption collection (`PixMo-Cap`) yields long, detailed descriptions quickly; 2D point grounding (`PixMo-Points`) scales cheaply; targeted synthetic generators (`PixMo-Docs`, `PixMo-Clocks`) fill known skill gaps.
  - Why it matters: Enables building strong VLMs without copying proprietary VLM outputs; ablations show `PixMo-Cap` scale correlates with downstream gains (Table 3a; Figure 9, ρ=0.82).

- Overlapping multi-crop encoding for detail and context (Figures 3 and 5; Table 2d)
  - What’s new: Preserve context around patch borders by overlapping crops but only transmit the non-overlapping centers to the LLM.
  - Why it matters: Large improvements over single-crop and non-overlap encodings, important for OCR and fine-grained captioning.

- Length-conditioned caption pretraining (Table 2e; §B.1, Figure 7)
  - What’s new: Provide a noisy “length hint” token during caption generation pretraining.
  - Why it matters: Improves both a caption F1 metric and average performance across 11 benchmarks; encourages controllable, dense descriptions that better align with later instruction tuning.

- Point-then-count with ordered 2D points (Tables 4a–4d; §3 “PixMo-Points”)
  - What’s new: Treat counting as a chain-of-thought sequence of points, ordered spatially; coordinates emitted in plain text (not special tokens).
  - Why it matters: Best-in-class counting accuracy across two datasets (Table 1, CountBenchQA and PixMo-Count), and ablations show the “point then count” recipe and plain-text coordinates outperform alternatives (Table 4).

- Efficient multi-annotation training (§2 “Multi-annotated images”)
  - What’s new: Pack all annotations for an image into one sequence with selective attention masks.
  - Why it matters: 2×–3× reduction in redundant vision encoding and >2× speedup without compromising learning (§2).

## 5. Experimental Analysis
- Evaluation setup (Section 5; Appendix C–E)
  - Academic benchmarks: 10 common datasets (AI2D, ChartQA, VQA v2.0, DocVQA, InfographicVQA, TextVQA, RealWorldQA, MMMU, MathVista, CountBenchQA) plus a new, harder counting set `PixMo-Count` (Table 1).
  - Human preference: 15k prompts × multiple images, ~870 annotators, >325k pairwise preferences; Elo scores via Bradley–Terry as in Chatbot Arena (Section 5).
  - Inference: 36 crops for most academic tasks; for counting and pointing the number of test crops must match training (12) or accuracy degrades—fixed by doing a brief high-res fine-tuning (Table 12).

- Main quantitative results (Table 1)
  - Overall average across 11 benchmarks:
    - `Molmo-72B`: 81.2% (best), Elo 1077 (rank 2).
    - GPT-4o (0513): 78.5% (Elo 1079, rank 1).
    - Gemini 1.5 Pro: 78.3% (Elo 1074, rank 3).
    - Claude 3.5 Sonnet: 76.7% (Elo 1069, rank 4).
  - Per-task highlights:
    - Natural image QA: State-of-the-art on VQA v2.0 (86.5% for `Molmo-72B`) and best or tied on RealWorldQA (75.2%) (Table 1).
    - OCR-heavy tasks: Very strong but slightly behind Qwen2-VL on some (e.g., TextVQA 83.1 vs Qwen2-VL-72B 85.5; ChartQA 87.3 vs 88.3) (Table 1).
    - Counting: Best across both CountBenchQA (91.2) and PixMo-Count (85.2) (Table 1), attributable to `PixMo-Points` and point-then-count (Tables 4, 11–12).
    - Reasoning: Trails on MMMU (54.1) and MathVista (58.6), suggesting less training on advanced reasoning (Section 5).

- Human preference studies (Section 5; Table 1 “Elo score”; Table 5; Figure 8; Table 9)
  - Global study: `Molmo-72B` ranks 2nd by Elo (1077), just behind GPT-4o (1079), and ahead of Gemini 1.5 Pro/Flash and Claude 3.5 Sonnet (Table 1).
  - Controlled ablation study: Human preference drops sharply if removing `PixMo-Cap` (win rate 35% vs. default `Molmo-7B-D`) or `PixMo-AMA` (40%), showing both datasets matter (Table 5; Figure 8).
  - Chatbot Arena (independent): `Molmo-72B` tops all open models but sits below several proprietary systems; likely reflects differences in question distributions (Table 9; Section 5 discussion).

- Skill-specific tests and ablations
  - Clock reading (Table 10): `Molmo-7B-D` achieves 68.2% vs single-digit to teens for many proprietary VLMs; specialized non-VLM still highest (78.9%). Synthetic `PixMo-Clocks` transfers surprisingly well to real images from COCO/OpenImages/Clock Movies.
  - Pointing evaluation (Table 11): `Molmo` models score F1 ≈ 74–75 with matched crop counts; mismatched crops at inference severely hurt pointing (F1 ~58), reinforcing the crop consistency constraint.
  - High-resolution fine-tuning (Table 12): A short fine-tune at 36 crops restores counting performance when evaluating at 36 crops (without hurting the 11-task average).
  - Vision encoder choice (Table 2a): MetaCLIP and SigLIP perform on par with OpenAI CLIP; self-supervised `DINOv2` is surprisingly competitive and fares well in human preference (Table 5).
  - Overlap cropping (Table 2d), attention pooling (Table 2f), text-only dropout (Table 2c), and length conditioning (Table 2e) all yield measurable gains.
  - Data mixture ablations (Table 3c): Adding `PixMo-Docs` improves document/chart tasks; removing pointing hurts the overall average; `PixMo` data brings gains beyond academic datasets alone (76.9 vs 72.5).
  - Pretraining data alternatives (Table 3b): A LAION connector stage gives no benefit; `ShareGPT4V/o` at similar scale underperforms; using GPT-4o to caption the same images as `PixMo-Cap` performs similarly to human captions—suggesting image set quality matters (also Table 3a scaling).

- Does the evidence support the claims?
  - Yes, on three fronts:
    1) Open, novel data collection correlates with performance (Figure 9; Tables 3a–3c).
    2) The targeted modeling choices drive reliable improvements (Tables 2, 4, 11–12).
    3) Head-to-head results against both open and proprietary systems demonstrate competitive or superior performance on most benchmarks and strong human preference (Table 1; Section 5).

## 6. Limitations and Trade-offs
- Assumptions and design constraints
  - Crop consistency: Counting and pointing degrade if the number of test crops differs from training (Tables 11–12). A short high-res fine-tune fixes this but adds complexity to deployment.
  - Answer styles: Reliance on “style tags” for some datasets could limit zero-shot robustness to unseen evaluation formats (§4).
- Scope not addressed
  - The work focuses on single-image understanding. Video, audio (beyond speech used for data collection), or multi-image reasoning are out of scope.
  - Advanced math and multi-step reasoning remain weaker (MMMU, MathVista; Table 1).
- Computational and data costs
  - Training large variants is compute-intensive (e.g., `Molmo-72B` fine-tune ~8.3k H100 GPU-hours; Table 8).
- Text-only capability regression
  - Multimodal fine-tuning reduces pure text performance vs the base LLM on several NLP benchmarks (Table 13). Adding text-only data (e.g., Tulu 3) mitigates this, but requires balancing to preserve multimodal gains.
- Evaluation dependencies and potential biases
  - The internal caption metric `cap F1` uses GPT-4o to extract/match “atomic statements,” which could introduce evaluator bias (Appendix C). While Figure 9 shows strong correlation to downstream performance (ρ=0.82), it is not causal.
- “Open” with caveats
  - The pipeline does not use VLMs to generate supervision, but it does employ closed text-only LLMs for transcript cleaning, AMA answer iteration, and code/QA synthesis for `PixMo-Docs` (§H “Bootstrapping from LLMs”). The paper argues this is replaceable by open LLMs as they improve.

## 7. Implications and Future Directions
- How this work changes the landscape
  - Demonstrates that strong VLMs can be trained with fully open data and code, without distilling from proprietary VLMs, narrowing the gap to top closed models (Table 1). This reduces barriers for research, safety auditing, and customized deployment.
  - Introduces “point-then-count” as a practical chain-of-thought for visual enumeration and shows the utility of 2D points as a lightweight grounding signal (Tables 4, 11).
- Follow-up research enabled or suggested
  - Replace closed text-only LLMs in the data pipeline with open LLMs as they mature; evaluate how this affects data quality and model performance (§H).
  - Strengthen reasoning: incorporate curated math/logic curricula and cross-modal chain-of-thought that goes beyond pointing, to improve MMMU/MathVista (Table 1) and text-only benchmarks (Table 13).
  - Explore cross-attention connectors or hybrid architectures to better preserve base LLM text skills while maintaining strong vision grounding.
  - Robustness to crop configurations: develop training schemes that generalize pointing/counting across resolutions and crop counts without post-hoc fine-tuning (Tables 11–12).
  - Richer multimodal actions: generalize from “pointing” to “dragging/gesturing” sequences and UI automation—promising for robotics and software agents (motivated in §3 “PixMo-Points” and Appendix D “AndroidControl” results).
- Practical applications
  - Document understanding and business analytics (charts/tables/diagrams via `PixMo-Docs`).
  - Assistive technologies and UI agents that act by pointing (Appendix D: AndroidControl results near reported SOTA).
  - Education and accessibility: dense, controllable image descriptions (length-conditioned captions; §B.1).
  - Domains requiring reliable counting or locating (e.g., retail inventory, quality control, remote sensing).

> Representative headline result: “`Molmo-72B` achieves 81.2 average across 11 academic benchmarks and ranks second by human preference (Elo 1077), outperforming Gemini 1.5 Pro/Flash and Claude 3.5 Sonnet, and trailing only GPT-4o” (Table 1).
