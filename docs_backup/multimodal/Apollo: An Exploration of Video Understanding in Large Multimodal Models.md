# Apollo: An Exploration of Video Understanding in Large Multimodal Models

**ArXiv:** [2412.10360](https://arxiv.org/abs/2412.10360)
**Authors:** Orr Zohar, Xiaohan Wang, Yann Dubois, Nikhil Mehta, Tong Xiao, Philippe Hansen‑Estruch, Licheng Yu, Xiaofang Wang, Felix Juefei‑Xu, Ning Zhang, Serena Yeung‑Levy, Xide Xia
**Institutions:** Meta AI, Stanford University

## 🎯 Pitch

Apollo revolutionizes video understanding in Large Multimodal Models by introducing 'Scaling Consistency,' a principle that ensures design choices effective on smaller models transfer reliably to larger ones. This breakthrough allows cost-efficient experimentation and lays a solid foundation for video comprehension, evidenced by Apollo's state-of-the-art performance on video benchmarks with relatively compact models.

---

## 1. Executive Summary
Apollo is a systematic exploration of what truly drives video understanding in Large Multimodal Models (LMMs), paired with a new benchmark (ApolloBench) and a family of trained models (Apollo-1.5B/3B/7B). The study introduces Scaling Consistency—showing that design choices validated on moderately sized models (≈2–4B parameters) reliably transfer to larger ones—then uses this to dissect video sampling, visual encoders, token resampling/integration, data mixtures, and training schedules, culminating in state-of-the-art performance on several video benchmarks (e.g., MLVU) at relatively small model sizes (Table 4).

## 2. Context and Motivation
- Problem/gap
  - Video-LMMs lag behind image-LMMs in both understanding and engineering clarity. Design choices—how to sample frames, which encoders to use, how to compress billions of visual tokens—are often ad hoc because training and evaluating large video systems is expensive and slow.
  - Existing video QA benchmarks are resource-heavy and frequently solvable without real video understanding (by text alone or a single frame), so they poorly diagnose video perception (Section 2; Fig. 2 left).

- Why it matters
  - Practical: Video is ubiquitous (education, sports, surveillance, creative tools). Efficient, grounded design guidance can cut compute costs and accelerate progress.
  - Scientific: Clarifying which architectural and training decisions actually improve temporal understanding builds a more principled foundation for video-LMMs.

- Prior approaches and shortcomings
  - Many works extend image-LMMs to video by sparsely sampling frames and reusing image encoders; or they add video-specific encoders without clear evidence of when they help (Section 1; Section 7).
  - Benchmarks proliferated (Video-MME, MLVU, LongVideoBench, TempCompass), but often correlate strongly with each other and are partly solvable by language priors or single images (Section 2.1–2.2; Fig. 2).
  - Training recipes vary (single- vs multi-stage, frozen vs unfrozen encoders), but without systematic comparisons tailored to video (Section 5.1–5.2).

- Positioning
  - This work provides a comprehensive, controlled study of key video-LMM design axes, introduces a principled notion—Scaling Consistency—for efficient experimentation (Section 3; Fig. 3), curates a perception-focused evaluation suite (ApolloBench, Section 2.3), and distills actionable, compute-aware recommendations that are validated by building the Apollo model family (Section 6; Table 4).

## 3. Technical Approach
This is primarily an empirical methodology plus a carefully designed model and evaluation pipeline.

A. Study protocol and Scaling Consistency (Section 3; Fig. 3; App. Sec. D)
- Core idea (term definition): Scaling Consistency is the observed phenomenon that the relative ranking of design choices measured on moderately sized LMMs (≈2–4B parameters) correlates strongly (R² > 0.9) with rankings on larger models (here up to 7B). This allows reliable decision-making with smaller, cheaper experiments.
- How they test it:
  - Train 84 LMM variants spanning 21 design choices (architecture, sampling, training, data mix), each paired with one of four LLM backbones: `Qwen2-0.5B`, `Qwen2-1.5B`, `Qwen1.5-4B`, `Qwen2-7B` (Section 3).
  - Measure correlations of model performances across sizes. Results: R²(4B,7B)=0.938 (Fig. 3 left; App. Fig. 15). Correlation improves roughly log-linearly with LLM size and stabilizes for datasets ≈500K samples (Fig. 3 right).

B. Evaluation re-grounding with ApolloBench (Section 2.3; Fig. 2)
- Problem: Many benchmarks can be solved with text-only or single frames; they are redundant and expensive (184 A100 GPU hours to evaluate a 3B model across popular suites; Section 2).
- Curation process (Fig. 11; App. Sec. B.3):
  1) Start with multiple-choice subsets of popular video QA benchmarks to avoid external LLM grading.
  2) Filter out questions that more than 50% of tested models can answer from text-only or a center frame—keeping items that require video perception (Fig. 2 left, ApolloBench column).
  3) Categorize remaining questions into five temporal perception categories: Temporal OCR, Egocentric, Spatial, Perception, Reasoning.
  4) Select 400 high-entropy items (maximally discriminative across models) and manually verify.
- Outcome: 41× faster evaluation than the full suite, while still highly correlated with them (Fig. 2 right).

C. Architectural and training ablations (Sections 4 and 5)
The study systematically varies:
- Video sampling strategies: `uniform` vs `fps` (“frames-per-second”) sampling (Section 4.1; Fig. 4).
  - Term: `fps sampling` samples frames at a fixed temporal rate (e.g., 2 frames/sec), preserving a consistent “playback speed.”
  - Term: `uniform sampling` picks N frames evenly spaced across the video, which implicitly changes playback speed across videos of different lengths.
- Visual representation: image encoders vs video encoders vs both (Section 4.2; Fig. 5; App. Tab. 9).
  - Key components include `SigLIP-SO400M` (image language-supervised), `InternVideo2` (video encoder), `VideoMAE`, `V-JEPA`, `DINOv2`, and `LanguageBind` variants.
- Token resampling (Section 4.3; Table 1):
  - Term: `token resampler` compresses the large set of per-frame visual tokens into a smaller set fed to the LLM. Compared: MLP+avg pooling, 2D conv+avg pooling, `Perceiver Resampler` (a cross-attention module that learns a small set of latent queries to summarize inputs).
- Token integration with text (Section 4.4; Table 2):
  - Compare direct concatenation vs insertion of textual timestamps vs learned separator tokens.
- Training schedules (Section 5.1; Table 3):
  - One-, two-, three-stage protocols; progressively unfreezing encoders and the LLM.
- When/how to train vision encoders (Section 5.2).
- Data composition (Section 5.3; Fig. 6; App. Tab. 13):
  - Vary proportions of `text`, `image`, `multi-image`, `video` samples during supervised fine-tuning (SFT).

D. The Apollo model family (Section 6; Fig. 8; Table 7)
- Architecture (Fig. 8; Section 6; App. Sec. C):
  - Dual encoders: `SigLIP-SO400M` (image) + `InternVideo2` (video).
  - For videos, frames are fed in clips of N frames (N depends on the video encoder; e.g., InternVideo2 encodes 4 frames per clip). For images, they replicate the image N times to reuse the same pipeline (App. Sec. C.1).
  - Encoder outputs are interpolated and channel-concatenated, then projected and downsampled by a `Perceiver Resampler` to a fixed budget (e.g., 32 tokens per frame).
  - Visual tokens are interleaved with textual timestamps: `clip from {MM:SS}-{MM:SS}: <vid_token>` (Table 2).
- Training (Table 7):
  - Stage 1 (Alignment): train connector only on 198K image+video captions.
  - Stage 2 (Vision pretraining): unfreeze vision encoders; train on 396K video-only captions.
  - Stage 3 (SFT): unfreeze LLM; train on 3.2M mixture of text, image, multi-image, and video with ≈10–14% text (Fig. 6).
  - Sampling during SFT: fps=2, tokens-per-second (tps)=32, tokens-per-frame (tpf)=16; max clips per video up to 150–200 depending on LLM size (Table 7).
  - Term: `tps` is the total visual tokens fed per second of video, and `tpf` is per-frame token budget after resampling.

## 4. Key Insights and Innovations
1) Scaling Consistency (Section 3; Fig. 3)
- What’s new: A practical, evidence-backed rule that design decisions validated on ≈2–4B models and ≈500K-sample datasets transfer to larger settings.
- Why it matters: It enables fast, low-compute exploration without sacrificing reliability.
- Evidence:
  > “R² between the 4B and 7B models is 0.938” (Fig. 3 left).
  > “Correlation to larger datasets starts to plateau at around 500K samples” (Fig. 3 right).

2) fps sampling beats uniform sampling, for principled reasons (Section 4.1; Fig. 4; App. Tab. 10–11)
- Mechanism: Uniformly sampling N frames from videos of different lengths alters the effective playback speed during training, confusing temporal reasoning. Fixed-fps sampling preserves time-scale consistency across videos.
- Evidence: Models trained with uniform sampling underperform consistently, and the gap is not explained by test-time frame counts (Fig. 4 left vs middle; App. Tab. 11, rows 1–4 vs 5–8).
- Further nuance: There’s a trade-off between tps and fps; optimal tpf is typically 8–32 (Fig. 4 right; App. Fig. 9–10).

3) Best vision representation: SigLIP (image) + InternVideo2 (video) with channel concatenation (Section 4.2; Fig. 5; App. Tab. 9)
- What’s new: A clear, large-scale comparison shows language-supervised image encoder `SigLIP-SO400M` is the strongest single encoder (Fig. 5 left), but the best overall is a dual-encoder combination (`InternVideo2 + SigLIP`) with channel-wise concatenation before resampling (Fig. 5 right).
- Evidence:
  > Single encoders: `SigLIP-SO400M` achieves the strongest overall (Fig. 5 left; App. Tab. 9 row 3, overall 52.7).
  > Dual encoders: `InternVideo2 + SigLIP-SO400M` is best (App. Tab. 9 row 19, overall 57.9), ≈+5 points over `SigLIP` alone.

4) Token resampling and integration matter (Sections 4.3–4.4; Tables 1–2)
- `Perceiver Resampler` yields superior compression vs pooling:
  > Table 1: Perceiver overall 55.5 vs MLP+avg 53.2 and Conv+avg 44.7.
- Simple textual timestamps between clips improve multimodal fusion:
  > Table 2: Inserting timestamps raises overall from 55.5 to 56.8 (+1.3), and adding either learned separators or timestamps yields consistent +2–3% gains across categories.

5) Training curriculum and data composition are decisive (Sections 5.1–5.3; Table 3; Fig. 6; App. Tab. 13)
- Progressive, three-stage unfreezing is best (connector → vision encoders → LLM):
  > Table 3: 3-stage overall 59.2 vs best 2-stage 57.8 vs 1-stage 48.7.
- Include ≈10–14% pure text in SFT to prevent catastrophic forgetting and improve reasoning:
  > Fig. 6 and App. Tab. 13 (e.g., 15% text with video-heavy mix: overall 59.0; 25% text: 54.1; 2% text: 48.7).
- Fine-tune vision encoders on video-only data when the LLM is frozen; avoid training them concurrently with the LLM on mixed modalities (Section 5.2; Finding 9).

These are fundamental innovations (new evaluation protocol, new efficiency principle, and system-level guidelines) rather than incremental tweaks.

## 5. Experimental Analysis
- Evaluation methodology
  - Benchmark audit (Section 2): Ten open-source LMMs are tested under three input regimes—full video, single center frame, and text-only—across Video-MME, MLVU, LongVideoBench, TempCompass, NExT-QA, PerceptionTest (Fig. 2 left; App. Tables 5–6).
  - Redundancy check: Correlations across benchmarks, durations, and question formats (Fig. 2 right; App. Fig. 12–14).
  - ApolloBench creation: Filtering out non-perception items; category balancing; human verification (Section 2.3; Fig. 11).

- Main quantitative findings
  - Benchmark quality:
    > “Some benchmarks can almost be entirely solved using a single frame… NExT-QA is solved using a single frame; Perception-Test behaves similarly” (Section 2.1; Fig. 2 left).
    > “As videos get longer, reliance on video perception decreases… compare Video-MME Short/Medium/Long” (Section 2.1; Fig. 2 left).
    > Strong cross-benchmark correlations (Fig. 2 right) and within-benchmark correlations across durations and question types (App. Fig. 12–14).
  - Scaling Consistency:
    > R²(4B,7B)=0.938; correlation increases with LLM size, plateaus around 500K samples (Fig. 3).
  - Video sampling (Section 4.1):
    > Uniform vs fps: Training with uniform sampling underperforms (Fig. 4 left), and the gap is not rescued by fps at test time (Fig. 4 middle). Example: uniform-64 train/test overall 55.1 (App. Tab. 11 row 4) vs fps-trained settings reaching ≈58–59 overall (App. Tab. 10 rows 9–20).
    > Trade-off: High tpf helps OCR/Spatial; fps helps temporal tasks; optimal tpf ≈ 8–32 (Fig. 4 right; App. Fig. 9–10).
  - Vision encoders (Section 4.2):
    > Single: `SigLIP-SO400M` best overall (Fig. 5 left; App. Tab. 9 row 3 overall 52.7).
    > Dual: `InternVideo2 + SigLIP-SO400M` best (App. Tab. 9 row 19 overall 57.9), with ≈+7% on ApolloBench vs single encoders reported in-text (Section 4.2).
  - Resampling and integration:
    > Table 1: Perceiver Resampler overall 55.5 vs MLP+avg 53.2.
    > Table 2: Timestamps or separators add ≈2–3% over naïve concatenation; timestamps overall 56.8.
  - Training schedules (Section 5.1; Table 3):
    > Three-stage curriculum overall 59.2; best two-stage 57.8; one-stage 48.7.
  - Data composition (Section 5.3; Fig. 6; App. Tab. 13):
    > ≈10–14% text helps; more than ≈14% or less than ≈7% degrades; slightly video-heavy mixes perform best.

- Apollo model results (Section 6; Table 4)
  - Apollo-3B:
    > MLVU: 68.7 vs Oryx-7B 67.5.
    > PerceptionTest: 65.0 (beats several 7B models).
    > Video-MME (w/o subs): 58.4 vs Oryx-7B 50.3; competitive with LLaVA-OV-7B (58.2).
    > ApolloBench overall: 62.7.
  - Apollo-7B:
    > MLVU: 70.9—on par with or better than some 30B+ models (Oryx-34B 70.8) while smaller.
    > Video-MME (w/o subs): 61.3 (competitive though not the absolute best 7B).
    > ApolloBench overall: 66.3 (top-tier among 7B models listed).
  - Takeaway: The system-level insights translate into strong across-the-board performance, especially notable at 3B (Table 4).

- Convincingness
  - The study’s claim set is supported by ablations that manipulate one factor at a time and by aggregated correlations ensuring choices generalize from smaller to larger models. The consistency across multiple benchmarks and the final Apollo results reinforce the practical value.

- Robustness/failure cases and conditions
  - Conditional results:
    - fps vs tps: Best settings depend on the mix of tasks (OCR vs temporal reasoning); see iso-accuracy contours (Fig. 4 right; App. Fig. 9).
    - Training encoders: Beneficial when trained on video-only, but harmful when co-trained with the LLM on mixed modalities (Section 5.2).
  - Not explored: Active frame/token selection and memory-based approaches (Section 4.1 and App. Sec. A, “Future work”).

## 6. Limitations and Trade-offs
- Assumptions and scope
  - Scaling Consistency is validated up to 7B and across Qwen-family LLMs; extrapolation to much larger models (e.g., 30–70B) is inferred but not directly tested (Section 3; App. Sec. D).
  - ApolloBench focuses on multiple-choice questions. While this avoids evaluator drift (no external LLM grader), it does not test open-ended conversational ability (App. Sec. A; Section 2.3).
  - The architecture standardizes image and video through replicated frames for images (App. Sec. C.1), which simplifies the pipeline but could underutilize image-specialized pathways in certain regimes (Table 8 shows unified ≈ split).

- Computational and data constraints
  - Training still uses substantial compute (e.g., 128 A100s, multi-million-sample SFT; App. Sec. C.4). The contribution is about making exploration cheaper (via Scaling Consistency and ApolloBench), not making training itself light-weight.
  - Data choices exclude non-permissive sources (e.g., ChatGPT-labeled sets), which may cap absolute scores relative to models trained on broader corpora (Section 6; App. Sec. C.3).

- Unaddressed scenarios
  - Retrieval/memory and interactive resampling across multi-turn dialogues are not evaluated; the paper hypothesizes these may not generalize well if resampling is conditioned only on the first question (Sections 4.3–4.4; App. Sec. A).
  - Very long videos are handled by spacing constant-fps clips across the timeline (App. Sec. C.1–C.2), which preserves local temporal fidelity but may miss subtle long-range dependencies unless clip count is high.

## 7. Implications and Future Directions
- How this changes the field
  - Establishes a compute-aware methodology: validate design choices on ~3B models using ≈500K samples and expect them to transfer (Fig. 3). This can meaningfully reduce iteration cycles for video-LMM research.
  - Provides clear, evidence-backed defaults for video-LMMs:
    - Train with fps sampling; target tpf in 8–32; manage tps/fps trade-off per task (Section 4.1).
    - Use dual encoders (`SigLIP-SO400M` + `InternVideo2`), concatenated before a `Perceiver Resampler` (Sections 4.2–4.3).
    - Interleave clips with textual timestamps (Section 4.4).
    - Adopt a 3-stage training curriculum with ≈10–14% text in SFT (Sections 5.1–5.3).
  - Re-ground evaluation with ApolloBench—faster yet still predictive of broader performance (Section 2.3; Fig. 2).

- Follow-up research enabled or suggested
  - Build better video encoders that close the gap with image encoders on spatial detail, while retaining temporal strengths; the dual-encoder gains (Fig. 5) indicate complementarity.
  - Explore memory/routing (e.g., retrieval banks, iterative clip selection), but re-evaluate them in truly conversational, multi-turn settings where resampling must update on each turn (App. Sec. A).
  - Develop conversation-centric benchmarks that avoid LLM-as-judge costs and variability while genuinely testing dialogue and reasoning over long videos (App. Sec. A).
  - Investigate unified vs split architectures in more depth with larger scales and diverse data, especially where images are dominant.

- Practical applications
  - Long-form video assistants (sports analysis, lecture summarization, compliance monitoring) where efficient clip sampling and strong temporal reasoning matter.
  - Video search and QA systems that must scale to hour-long content on limited compute budgets.
  - Industrial model development workflows that need reliable small-scale proxies for large-scale deployment decisions.

> Bottom line: By auditing benchmarks, codifying Scaling Consistency, and validating end-to-end design choices, Apollo offers a rigorous blueprint for building effective, efficient video-LMMs—and demonstrates the blueprint’s value with state-of-the-art results from relatively small models (Table 4).
