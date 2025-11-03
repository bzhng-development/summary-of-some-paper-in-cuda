# Eagle 2.5: Boosting Long-Context Post-Training for Frontier Vision-Language Models

**ArXiv:** [2504.15271](https://arxiv.org/abs/2504.15271)

## 🎯 Pitch

Eagle 2.5 introduces a family of vision-language models tailored for high-resolution images and extended video understanding, natively handling long sequences without relying on specialized compression modules. By pioneering information-first sampling techniques (Image Area Preservation and Automatic Degradation Sampling), adopting a progressive long-context post-training framework, and releasing the large-scale Eagle-Video-110K dataset, Eagle 2.5 achieves consistent, scalable performance—matching or surpassing much larger or commercial models on challenging long-video and image benchmarks. This positions Eagle 2.5 as a breakthrough foundation for real-world multimodal applications where contextual breadth, efficiency, and generality are paramount.

---

## 1. Executive Summary (2–3 sentences)
Eagle 2.5 is a long-context vision–language model (VLM) family that can natively handle high-resolution images and long videos without specialized compression modules. It introduces an information-first sampling scheme—Image Area Preservation (IAP) and Automatic Degradation Sampling (ADS)—plus progressive long-context post-training and a new long-video dataset (Eagle-Video-110K), enabling consistent performance gains as input length grows and achieving competitive results with far larger models on long-video benchmarks (e.g., 72.4% on Video-MME at 512 frames; Fig. 1, Table 2).

## 2. Context and Motivation
- Problem gap
  - Many VLMs excel on short-context tasks (few images, short clips) but struggle with extended visual inputs: long videos, multi-image documents, and high-resolution media (Sec. 1). This limits real-world applications like movie-length analysis, slide-deck QA, or surveillance understanding where information spans hundreds of frames or pages.
- Why it matters
  - Real-world content is long and high resolution. Effective long-context understanding improves tasks such as long-form content retrieval, temporal reasoning (who did what, when, and why), and precise document comprehension at scale.
- Prior approaches and shortcomings
  - Compression/selection modules (e.g., question-guided selection, token reduction) avoid extending model context but add compute or capacity bottlenecks and can clip useful context (Sec. 1; citations across Jin et al., Korbar et al., Shen et al., Weng et al.).
  - Extending LLM context directly for multimodal inputs (e.g., LongVILA, LongViTA) is promising but has struggled to:
    - Match proprietary models,
    - Scale performance consistently with more visual input,
    - Clarify robust training strategies and data recipes (Sec. 1–2).
- Positioning
  - Eagle 2.5 builds a generalist long-context VLM that:
    - Avoids bespoke compression modules (flexibility preserved; Sec. 3.1, Fig. 2),
    - Preserves information in both text and visuals via an information-first sampler (Sec. 3.2.1; Eq. (1), Eq. (2), Fig. 3),
    - Trains progressively to longer contexts (32K → 64K → 128K; Sec. 3.2.2, Table 8),
    - Introduces a dual-annotated long-video dataset (story-level + clip-level; Sec. 3.3.2, Figs. 4–5).

## 3. Technical Approach
Eagle 2.5 comprises a standard multimodal architecture plus three pillars: information-first sampling, progressive mixed post-training, and a long-video dataset.

- Architecture (Sec. 3.1; Fig. 2)
  - Vision encoder: `SigLIP-so400M` (a vision backbone).
  - Connector: an MLP that projects vision features into the language model space (LLaVA-style).
  - LLM: `Qwen2.5` series.
  - Any-resolution images handled with tiling: split large images into a grid of tiles so the model sees high-resolution content without downscaling away detail. Unlike prior fixed-grid tiling, Eagle 2.5’s tiling is governed by IAP (below).

- Information-first sampling (Sec. 3.2.1)
  - Goal: maximize the “useful information density” that fits into a fixed model context window `L_max` by:
    - Keeping the full text,
    - Allocating remaining tokens to visuals in a way that preserves area, aspect ratio, and temporal coverage.
  - Component A: Image Area Preservation (`IAP`; Fig. 3, Eq. (1))
    - Problem with prior tiling: fixed grids force downsampling or distort aspect ratios, reducing usable detail (Fig. 3a).
    - Idea: choose a tiling configuration `(r_w, r_h)` that:
      - Preserves at least 60% of the original image area, and
      - Aligns tiling aspect ratio with the image’s native aspect ratio.
    - Scoring (Eq. (1)): select the candidate tiling ratio that maximizes
      - an area term (penalizes <60% preserved area; saturates at 0.6 so it doesn’t over-reward),
      - times an aspect-ratio alignment term (1 at perfect match, decays symmetrically for deviations).
    - Effect: tiles that preserve more of the image and keep geometry faithful (Fig. 3b).
  - Component B: Automatic Degradation Sampling (`ADS`; Eq. (2))
    - Context budget split: compute text token length `L_text` first; fix it. Visual budget is `L_visual = L_max - L_text`.
    - For images: choose a max tiles-per-image `t` (up to 12) to maximize spatial information for `M` images.
    - For videos/documents (temporal content): choose a sampling count `n` to maximize temporal coverage.
    - Constrained optimization (Eq. (2)): maximize total visual tokens `sum_i L(t, I_i) + 256*n` subject to the visual budget. Temporal units (frame or page) cost 256 tokens each; images depend on `t` through `L(t, I_i)`.
    - Two-phase “degradation” to fit budget:
      1) Temporal first: set `t=1` (no tiling), aim for 2 FPS for videos and all images for multi-image docs; enforce a minimum frames per visual input. If the minimum cannot be met, discard the sample. Compute `n* = floor((L_visual - M)/256)`.
      2) Then tiling: from the set `T={12,8,6,4,2,1}`, pick the largest `t*` that still fits `sum_i L(t, I_i) ≤ (L_visual - 256·n*)`.
    - Intuition: always keep complete text; then maximize temporal coverage; then spend remaining budget on higher-resolution imagery via tiling. This prevents text truncation and retains fine details where possible.

- Progressive mixed post-training (Sec. 3.2.2; Table 8; Fig. 6)
  - Mixed post-training: use `ADS` to adaptively fit each sample to a target `L_max` while mixing short and long sequences with length-balanced packing (Stage-2+).
  - Progressive schedule: train sequentially at larger contexts—`32K → 64K → 128K` tokens—rather than jumping straight to maximum. This:
    - Enhances robustness across all input sizes,
    - Makes optimization easier (learn shorter contexts first), then scales up (Table 7, Fig. 6).
  - Practical settings (Table 8):
    - Stage 1: connector-only alignment,
    - Stage 1.5: full-model pretraining (short+long data),
    - Stages 2–4: full-model post-training at `32K, 64K, 128K` with short+long data.

- Data recipe (Sec. 3.3; Table 1)
  - “Diversity first, then quality”: assemble a broad pool of open data for videos, multi-page documents, and long text (Table 1), then add a focused long-video dataset to cover very long durations absent from public sets (Fig. 4).
  - Eagle-Video-110K (Sec. 3.3.2; Figs. 4–5)
    - Collection via diversity filtering: cut videos into 10s clips, embed with CLIP at 1 FPS, compute max similarity to current pool; keep clips whose max similarity is <0.5 to ensure novelty (Sec. 3.3.2).
    - Dual-level annotation (Fig. 5):
      - Story-level (top-down): use human-annotated chapters as segments (not shot boundaries), sample up to 2 FPS (max 50 frames/segment), caption with GPT-4o; aggregate to generate long-form QA with GPT-4 (Sec. 3.3.2).
      - Clip-level (bottom-up): for short clips, sample up to 2 FPS and generate diverse QA pairs with GPT-4o using a broad pool of question types; add time anchors and textual context anchors to safely “lift” clip QA to full-video QA without leaking the answer (Sec. 3.3.2; Appendix E has exact prompts and the 63-category type pool in Table 12).

- Efficiency and scaling (Appendix B)
  - Memory- and throughput-oriented engineering: fused Triton ops, CPU offloading, vLLM for inference, and a customized two-layer context-parallel communication pattern (“zigzag Llama3-style” with all-gather KV; B.1) to serve long contexts and sparse video frame sampling efficiently.

## 4. Key Insights and Innovations
- Information-first sampling that preserves both semantics and detail (Sec. 3.2.1; Eq. (1), Eq. (2), Fig. 3)
  - What’s new: instead of prioritizing visuals (risking text truncation) or applying rigid tiling, Eagle 2.5 fixes the entire text, then optimizes visual allocation across space (tiling) and time (sampling). IAP enforces area and aspect-ratio fidelity; ADS enforces a budget-aware allocation across temporal coverage and resolution.
  - Why it matters: yields higher information density in the same context window and produces consistent performance scaling as the number of frames increases (Fig. 1, Fig. 6). Ablations show both IAP and ADS matter (Table 6).

- Progressive mixed post-training for long contexts (Sec. 3.2.2; Table 7; Fig. 6)
  - What’s new: a mixed training curriculum that grows `L_max` stepwise, preserving short-context competence while expanding long-context ability.
  - Why it matters: improves performance over one-shot long-context training, particularly in frame-heavy regimes (Table 7), and shifts the performance-vs-frames curve upward as training progresses (Fig. 6).

- A dual-annotated long-video dataset with diversity-aware collection (Sec. 3.3.2; Figs. 4–5)
  - What’s new: Eagle-Video-110K explicitly targets long durations absent in existing open datasets (Fig. 4), and its annotations combine story-level (chapter-based) semantics with clip-level fine-grained temporal QA enhanced via anchors (Fig. 5, Appendix E).
  - Why it matters: empirically boosts long-video performance, particularly when many frames are presented (Table 7, Fig. 6, Q4 in Sec. 4.2).

- Generalist design without specialized compression modules (Sec. 3.1, Fig. 2)
  - What’s new: a simple LLaVA-style projection from `SigLIP` to `Qwen2.5` plus IAP/ADS sampling achieves parity with larger specialized systems on long-video tasks (Table 2, Fig. 1).
  - Why it matters: easier to adapt/extend across tasks and inputs, and avoids lock-in to task-specific compression engineering.

## 5. Experimental Analysis
- Evaluation methodology (Sec. 4; Tables 2–3; Fig. 1)
  - Video benchmarks: MVBench, Perception-Test, EgoSchema, MMB-Video, MLVU, LVBench, Video-MME (w/ and w/o subtitles), CG-Bench (multiple metrics), HourVideo, Charade-STA. Default sampling is 2 FPS, tiling off for videos, minimum 8 frames per video; Perception-Test enables tiling for high resolution (Table 2 note).
  - Image benchmarks: DocVQA, ChartQA, InfoVQA, TextVQA, OCRBench, MMStar, RWQA, AI2D, MMMU, MMB1.1, MMVet, HallB, MathVista (Table 3). Average score divides OCRBench by 10 to normalize scales.
  - Setup reflects the paper’s intended use: the same generalist model is evaluated across both long videos and high-resolution images.

- Main quantitative results
  - Long video capability and scaling with more frames
    - Performance rises as frames increase (Fig. 1 and Fig. 6). On Video-MME (no subtitles) Eagle 2.5-8B reaches:
      > “72.4% at 512 input frames,”
      closely matching GPT-4o and large open models like `Qwen2.5-VL-72B` and `InternVL2.5-78B` (Fig. 1; Table 2).
  - Broad video benchmark performance (Table 2)
    - Eagle2.5-8B achieves:
      - MVBench: 74.8,
      - Perception-Test: 82.0,
      - EgoSchema: 72.2,
      - MLVU: 77.6,
      - Video-MME (w/o subtitles): 72.4, (with subtitles): 75.7.
    - On CG-Bench, across metrics “Clue / Long / Open / mIoU”, Eagle 2.5-8B scores:
      > “55.8 / 46.6 / 45.6 / 13.4,”
      surpassing `Claude-3.5-Sonnet` and `Gemini-1.5-Pro` on several metrics (Table 2).
    - HourVideo (long-form understanding): Dev 44.5, Test 41.8—surpassing `Gemini-1.5-Pro` (Table 2).
    - Charade-STA (temporal grounding): substantial gains vs similar-sized public models (Table 2, last column).
  - High-resolution image/document capability (Table 3)
    - Strong document/chart/text QA: DocVQA 94.1, ChartQA 87.5, InfoVQA 80.4, TextVQA 83.7, OCRBench 869 (Table 3; OCRBench scaled by 10 in the average).
    - General multimodal reasoning: MMStar 66.2, RWQA 76.7, AI2D 84.5, MMB1.1 81.7, MMVet 62.9, HallB 54.7, MathVista 67.8; overall average 75.6 (Table 3).
  - Additional doc benchmarks: SlideVQA (Dev ANLS 73.8; Test 72.7) and MMLongBench-Doc (Overall F1 29.4) in Appendix C (Tables 9–10).

- Ablations and diagnostics (Sec. 4.2)
  - Impact of long-context training on image tasks (Table 4)
    - Training at longer `L_max` (progressive 32K→64K→128K) does not harm short-context image benchmarks; slight improvements are seen, with average moving from 74.8 (S2 baseline) to 75.7 at 128K (Table 4).
  - Image pretraining helps video benchmarks (Table 5)
    - Adding image data in Stage 1.5 improves MVBench and MLVU (e.g., MVBench from 72.9 to 73.1 and MLVU from 70.9 to 71.5); Video-MME shows modest gains (65.2 → 65.4) when limited to 32 frames at 2 FPS (Table 5).
  - IAP and ADS are both useful (Table 6)
    - Removing IAP significantly hurts high-res and fine-grained tasks: InfoVQA drops 77.6→76.2; Perception-Test 76.3→73.3 (Table 6).
    - Removing ADS can truncate supervision and degrade performance: e.g., MLVU 71.5→70.1; Video-MME 65.4→65.0 (Table 6).
  - Progressive vs direct long-context training; benefit of Eagle-Video-110K (Table 7; Fig. 6)
    - Progressive 32K→64K outperforms direct 64K on MVBench/MLVU/Video-MME (e.g., 73.0 vs 71.3 on MVBench; Table 7).
    - Adding Eagle-Video-110K further improves all three (to 73.9/75.1/68.8; Table 7) and especially boosts performance at ≥128 frames on Video-MME (Fig. 6).

- Do the experiments support the claims?
  - Yes—three lines of evidence are consistent:
    - Scaling with more frames (Figs. 1 and 6),
    - Broad benchmark improvements vs similar-sized open models (Table 2) and competitive performance against much larger models on long-video tasks,
    - Ablations linking gains to IAP/ADS, progressive training, and Eagle-Video-110K (Tables 6–7).

## 6. Limitations and Trade-offs
- Assumptions in sampling
  - ADS assigns a fixed 256-token cost per temporal unit (frame/page) and uses a fixed minimum frames requirement; samples are discarded if they can’t meet it within budget (Sec. 3.2.1). This may bias training away from ultra-dense or very text-heavy samples that need atypical allocations.
  - For temporal content, tiling is disabled (fixed 256 tokens per frame; Sec. 3.2.1). Very high-resolution video frames could still benefit from spatial tiling, which is currently not used for videos.
- Data dependencies and label quality
  - Eagle-Video-110K relies on GPT-4/4o for captioning and QA generation (Sec. 3.3.2; Appendix E). While cost-effective, auto-annotation can encode model biases or errors, especially for subtle temporal/causal queries.
  - Novelty filtering via CLIP similarity depends on the embedding and threshold (τ=0.5; Sec. 3.3.2). It may omit useful data with high semantic overlap but different fine-grained details.
- Compute and memory
  - Long-context training (32K–128K tokens) is compute- and memory-intensive (Appendix B). The work deploys fused kernels, CPU offloading, and custom parallelism to make it feasible; replicating this may be challenging for smaller labs.
- Generality vs specialization
  - Avoiding specialized compression/selection modules preserves flexibility but may leave some optimality on the table for niche tasks where learned selection could outperform heuristic ADS/IAP.
- Reporting gaps
  - While many benchmarks are covered, detailed failure case analyses are limited in the paper. For example, where the model still struggles (e.g., extremely long narratives with sparse cues, or cross-modal coreference over hours) is not deeply dissected.

## 7. Implications and Future Directions
- Field impact
  - Demonstrates that a generalist VLM, carefully trained with information-first sampling and progressive long-context curricula, can realize genuine performance gains as input length increases (Figs. 1, 6) without bespoke compression modules. This challenges the assumption that long-video understanding must rely on complex selection/compression stacks.
- Follow-up research enabled or suggested
  - Learned allocation policies: replace ADS’s hand-crafted budget rules with policy learning to decide per-sample allocations of tiles vs frames vs text.
  - Video tiling: extend IAP to temporal inputs (selective tiling for certain frames or segments).
  - Better temporal planning: integrate memory mechanisms (e.g., hierarchical summaries or recurrent memory bridges) with Eagle 2.5’s long-context backbone.
  - Label quality auditing: human-in-the-loop refinement of auto-generated QA pairs, especially for causal, counterfactual, and multi-episode reasoning (see type pool in Table 12).
  - Robustness and fairness: evaluate bias propagation from auto-annotations; investigate domain adaptation for specialized fields (medicine, industrial inspection).
- Practical applications
  - Long-form video analytics: sports broadcasting, movie/TV analytics, surveillance review, meeting/video lecture summarization and QA.
  - Enterprise document QA: slide decks, scanned PDFs, and multi-page reports (strong DocVQA/AI2D/MMMU/InfoVQA/TextVQA results in Table 3).
  - Multimodal assistants: systems that must reason across dozens/hundreds of images or frames with precise temporal grounding and minimal hallucination (HallB 54.7; Table 3), where IAP helps preserve fine details and ADS avoids truncating instructions.

In short, Eagle 2.5 contributes a principled, practical path to long-context multimodal learning: fix the text, allocate visual tokens where they matter (space and time), and grow the model’s context capacity progressively. The result is a compact 8B model that keeps up with or challenges much larger systems on long-video understanding (Table 2; Fig. 1), while remaining broadly capable on high-resolution image/document tasks (Table 3).
