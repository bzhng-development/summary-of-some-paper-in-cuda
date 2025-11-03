# Eagle 2.5: Boosting Long‑Context Post‑Training for Frontier Vision‑Language Models

**ArXiv:** [2504.15271](https://arxiv.org/abs/2504.15271)
**Authors:** Guo Chen, Zhiqi Li, Shihao Wang, Jindong Jiang, Yicheng Liu, Lidong Lu, De‑An Huang, Wonmin Byeon, Matthieu Le, Max Ehrlich, Tuomas Rintamaki, Tyler Poon, Tong Lu, Limin Wang, Bryan Catanzaro, Jan Kautz, Andrew Tao, Zhiding Yu, Guilin Liu
**Institutions:** NVIDIA (likely, including NVlabs), affiliated universities or labs (not specified explicitly in abstract)

## 🎯 Pitch

Eagle 2.5 revolutionizes vision-language models by embracing long-context inputs through its innovative "information-first" sampling strategy and progressive post-training approach. By achieving state-of-the-art results with the new Eagle-Video-110K dataset, this framework enhances understanding of extended video sequences and high-resolution images, unlocking potential in applications like surveillance and multi-page document analysis.

---

## 1. Executive Summary (2-3 sentences)
Eagle 2.5 is a long-context vision-language model (VLM) training framework that makes performance scale with more visual input rather than merely tolerate it. It introduces an “information-first” sampling strategy and a progressive post-training schedule, plus a new long-video dataset (Eagle-Video-110K), to achieve state-of-the-art results on long video and high-resolution image benchmarks with an 8B-parameter model (e.g., 72.4% on Video-MME with 512 frames; Table 2, Fig. 1).

## 2. Context and Motivation
- Problem addressed
  - Long-context multimodal understanding is underdeveloped: current VLMs often perform well on short clips or a few images but falter when asked to reason over long videos, many images, or high-resolution pages (Sec. 1).
  - Extended contexts can mean hundreds of video frames, multi-page documents, or very large images that must be split (“tiled”) to preserve fine details.

- Why this matters
  - Real-world applications (meeting analysis, movie understanding, surveillance, lectures/slides, long instructions) are inherently long-form. Effective long-context models can track events, entities, and dependencies over time and space more reliably than short-context systems.

- Prior approaches and shortcomings (Sec. 2)
  - Context compression/selection modules (e.g., question-guided selection, token reduction) reduce the amount of input the core model sees. These avoid extending LLM context but add components that can miss information or introduce overhead and capacity limits.
  - Directly extending LLM context for multimodal inputs (e.g., LongVILA, LongViTA) remains difficult: performance often lags proprietary systems, and accuracy does not consistently improve as more visual input is provided.
  - Training strategies and data “recipes” for native long-context VLMs have been unclear, especially how to balance text vs. visual tokens and how to schedule training across different sequence lengths.

- Positioning of this work
  - Eagle 2.5 aims to be a generalist long-context framework (no specialized compression modules) that:
    - Preserves both text and high-fidelity visual details through “information-first” sampling (Sec. 3.2.1).
    - Trains progressively on longer sequence lengths so accuracy scales with more frames/tiles/pages (Sec. 3.2.2).
    - Supplies long-form supervision via Eagle-Video-110K with story-level and clip-level annotations (Sec. 3.3.2).

## 3. Technical Approach
Eagle 2.5 consists of a standard VLM backbone with two training pivots—information-first sampling and progressive long-context post-training—plus a long-context data pipeline.

- Architecture (Sec. 3.1; Fig. 2)
  - Vision encoder: `SigLIP-so400M` extracts image/frame embeddings.
  - MLP connector: a small projection aligns vision embeddings into the LLM token space.
  - LLM: Qwen2.5 family serves as the language backbone.
  - Any-resolution images are handled via `tiling` (splitting a large image into smaller tiles that can be encoded at fixed resolution, then fed as a sequence).
  - Design choice: no extra compression/selection module. The model stays flexible across tasks and avoids the brittleness of specialized components.

- Information-first sampling (Sec. 3.2.1)
  - Goal: Preserve essential semantics by keeping all text intact and then filling remaining context with the most informative visual details.
  - Two mechanisms:

    1) Image Area Preservation (IAP) for tiling (Fig. 3; Eq. (1))
       - Problem: Common tiling grids (e.g., 3×4 tiles at fixed resolution) can distort aspect ratios or downsample away large portions of the original image, wasting high-resolution details.
       - Approach: For an input image of size `W × H`, choose a tile grid `(r_w, r_h)` (subject to a maximum number of tiles `N`) that:
         - Preserves at least 60% of the original area, and
         - Aligns the tiling aspect ratio `r_t = r_w / r_h` with the original aspect ratio `r_orig = W / H`.
       - Selection criterion (Eq. (1)): maximize
         - `min(A_new / A_orig, 0.6) × min(r_t / r_orig, r_orig / r_t)`
         - where `A_new = r_w × r_h × s^2` is the total tiled area at fixed tile size `s` (e.g., 448×448), and `A_orig = W × H`.
       - Intuition: reward keeping enough of the image (up to the 0.6 threshold) and matching aspect ratio; penalize grids that would crop or distort the scene. Fig. 3 shows this retains more of the original content than InternVL-style rigid aspect ratios.

    2) Automatic Degradation Sampling (ADS) for visual-text token budgeting (Eq. (2))
       - Problem: With a fixed maximum sequence length `L_max`, naively sampling many frames/tiles risks truncating text or wasting tokens on uninformative visual units.
       - Approach: Make text “first-class”—compute `L_text` for the prompt and keep it all. The remaining visual budget is `L_visual = L_max − L_text`.
       - Under that visual budget, optimize:
         - For images: the maximum tiles-per-image `t` (spatial coverage).
         - For videos/documents: the number of temporal/page samples `n` (temporal coverage).
       - Constrained packing problem (Eq. (2)):
         - Maximize the total visual coverage `Σ_i L(t, I_i) + 256·n` subject to `Σ_i L(t, I_i) + 256·n ≤ L_visual`,
           with `1 ≤ t ≤ 12`, and `1 ≤ n ≤ N_max` (`N_max` proportional to video duration or doc pages). For videos/docs, each frame/page consumes 256 tokens: `L(1,·) = 256`.
       - Dual-phase “degradation” strategy:
         - Temporal-first: set `t = 1`, aim for 2 FPS (videos) and include all images for multi-image docs; ensure each visual input has at least `N_min` frames. If not achievable within `L_visual`, drop the sample. Compute
           > `n* = floor((L_visual − M) / 256)`  
           where `M` is the number of images.
         - Tiling next: pick the largest `t ∈ {12, 8, 6, 4, 2, 1}` such that  
           > `Σ_i L(t, I_i) ≤ (L_visual − n*·256)`  
           This uses leftover capacity to increase spatial detail.
       - Intuition: Always keep full text; then maximize temporal coverage; finally spend any remaining budget on spatial details. This is “all-context-centric,” not “vision-only-centric,” ensuring stable supervision signals.

- Progressive post-training schedule (Sec. 3.2.2; Fig. 6; Tab. 7)
  - Mixed post-training: Length-balanced packing trains across varied input sizes so performance is consistent from short to long sequences.
  - Progressive mixing: Increase `L_max` in stages—32K → 64K → 128K tokens—rather than jumping to a very long context at once.
    - Why: A single long-context run spreads samples thin across the huge length space; shorter contexts don’t get enough focus; and some long samples are hard to learn without “warm-up” at shorter contexts (Sec. 4.2, Q3; Tab. 7).

- Data recipe (Sec. 3.3; Tab. 1; Figs. 4–5)
  - Open-Data: a curated mixture emphasizing diversity first, then quality, including human-annotated video/image-document datasets and synthetic video QA/captioning generated by strong models (Tab. 1).
  - Eagle-Video-110K: a new long-video dataset targeting durations underrepresented in open data (Fig. 4).
    - Diversity-driven collection: Use CLIP features (1 FPS) on 10s clips from candidate sources `A` and existing training set `B`. Select novel clips where the maximum cosine similarity with any clip in `B` is below `τ = 0.5` (Sec. 3.3.2).
    - Dual annotation strategy (Fig. 5):
      - Top-down story-level: Use human-annotated chapters (from ViTT, VidChapters) as segments; for each chapter, sample up to 2 FPS (≤ 50 frames) and ask GPT-4o for dense captions guided by chapter titles, then aggregate them to generate video-level QA with GPT-4.
      - Bottom-up clip-level: For short clips, sample up to 2 FPS and generate diverse QA pairs with GPT-4o across a question-type pool (Tab. 12). Then convert to video-level QA by adding time anchors and textual context anchors that hint at scene context without leaking the answer (Sec. 3.3.2).

- Training system engineering (Appendix B; Tab. 8)
  - GPU memory optimizations (Triton-fused ops, CPU offloading), distributed context parallelism (Ulysses + ring all-gather KV, “zigzag Llama3-style”), accelerated video decoding, and vLLM-based serving.
  - Progressive stages (Tab. 8): start from Eagle-2 Stage-1.5 weights; Stage-2 to Stage-4 progressively extend long-context training with mixed short+long data; growth in `L_max` from 32K → 64K → 128K.

## 4. Key Insights and Innovations
- Information-first sampling that protects text and smartly allocates visual tokens (Sec. 3.2.1; Eq. (2))
  - Novelty: The token budget is set “around all modalities”; full text is retained by design, and visual tokens are then optimized to maximize temporal and spatial coverage.
  - Significance: Prevents supervision loss from text truncation and raises information density compared to fixed-rate frame sampling or rigid tiling. Ablations (Tab. 6) show removing ADS or IAP degrades performance on both image and video tasks.

- Image Area Preservation (IAP) as an optimal-tiling objective (Fig. 3; Eq. (1))
  - Novelty: Instead of hard aspect-ratio rules with unavoidable area loss, IAP formulates tiling as an optimization that jointly preserves area (≥ 60%) and aspect-ratio fidelity.
  - Significance: Better retention of high-res details, which matter for OCR and fine-grained recognition. Ablations (Tab. 6) show InfoVQA and Perception-Test drop notably without IAP.

- Progressive mixed long-context training (Sec. 3.2.2; Tab. 7; Fig. 6)
  - Novelty: A curriculum over context lengths (32K→64K→128K) while keeping mixed short+long sequences in each stage.
  - Significance: Improves stability and yields consistent accuracy gains, especially at larger frame counts (Fig. 6). Outperforms single-stage 64K training on multiple video benchmarks (Tab. 7).

- Eagle-Video-110K with dual-level (story and clip) supervision (Sec. 3.3.2; Figs. 4–5)
  - Novelty: Diversity-first selection of long videos plus two complementary annotations—chapters for narrative structure and clips for local details—with time and context anchors to lift clip-level QA to full-video QA.
  - Significance: Boosts long-video understanding and improves scaling at high frame counts (Fig. 6: the gap widens beyond 128 frames when adding Eagle-Video-110K).

These are fundamental training/data innovations rather than incremental architecture tweaks.

## 5. Experimental Analysis
- Evaluation setup
  - Video (Table 2; Sec. 4.1):
    - Frame sampling: 2 FPS by default; minimum 8 frames per video; no tiling for most benchmarks; tiling enabled for Perception-Test to support high resolution.
    - Max frames: 512 on Video-MME; 256 on others.
    - Baselines: Open models across scales (e.g., Qwen2.5-VL-8B/72B, InternVL2.5-8B/78B, LLaVA-Video-8B/72B) and closed models (GPT-4o, Gemini-1.5 Pro, Claude 3.5 Sonnet).
  - Images (Table 3):
    - Tasks include document/diagram QA (DocVQA, AI2D), chart/table understanding (ChartQA), OCR-heavy QA (InfoVQA, TextVQA, OCRBench), multi-skill reasoning (MMstar, MM-Vet, MMMU), hallucination diagnosis (HallB), and math with visuals (MathVista).

- Main results
  - Long video understanding (Table 2; Fig. 1):
    - Consistent scaling with more frames (Fig. 1): Eagle 2.5’s curve rises as input frames increase to 512 on Video-MME.
    - Quantitatively:
      > “Eagle 2.5-8B achieves 72.4 on Video-MME (w/o subtitles) with 512 frames” (Table 2; also highlighted in Abstract and Fig. 1).  
      This is on par with large open/commercial systems and above similar-size baselines.
      > MVBench: 74.8; Perception-Test: 82.0; EgoSchema: 72.2; MLVU: 77.6; LongVideoBench (LVBench): 66.4 (Table 2).  
      On CG-Bench, four metrics are 55.8, 46.6, 45.6, 13.4, exceeding Gemini-1.5 Pro and Claude 3.5 Sonnet entries listed in Table 2.
    - Temporal grounding:
      > “Charades-STA mIoU Dev/Test: 44.5/41.8” (Table 2), showing strong temporal localization relative to baselines shown.

  - High-resolution images and OCR (Table 3):
    - Strong, balanced performance:
      > DocVQA: 94.1; ChartQA: 87.5; InfoVQA: 80.4; TextVQA: 83.7; OCRBench: 869; MMstar: 66.2; RWQA: 76.7; MMB1.1: 81.7; MM-Vet: 62.9; AI2D: 84.5; MMMU: 55.8; HallB: 54.7; MathVista: 67.8;  
      Average: 75.6 (Table 3).  
      These are competitive with or above 8B peers (e.g., matches Qwen2.5-VL-8B’s reported 75.6 average) while delivering the long-video gains above.

- Ablations and diagnostics (Sec. 4.2)
  - Information-first sampling (Tab. 6):
    > Removing IAP: InfoVQA 77.6→76.2, Perception-Test 76.3→73.3;  
    Removing ADS: MLVU 71.5→70.1, Video-MME 65.4→65.0.  
    This confirms both spatial retention (IAP) and token budgeting (ADS) matter—especially for high-res and fine-grained video tasks.
  - Progressive schedule and Eagle-Video-110K (Tab. 7; Fig. 6):
    > 32K→64K progressive training outperforms single 64K training on MVBench (73.0 vs. 71.3), MLVU (74.5 vs. 74.0), and Video-MME (68.1 vs. 67.9) with Open-Data alone.  
    Adding Eagle-Video-110K further lifts these to 73.9, 75.1, 68.8 respectively (Table 7).  
    Fig. 6 shows accuracy increases with 16→512 frames, and the curve with Eagle-Video-110K is notably higher for ≥128 frames.
  - Cross-influence of image/video data (Tabs. 4–5):
    > Adding long-context training and increasing `L_max` from 32K→64K→128K does not harm short-context image benchmarks; it slightly helps (Table 4, average rises to ~75.7 at 128K).  
    > Image-heavy pretraining significantly improves MVBench and MLVU, but gives smaller gains on the more challenging held-out Video-MME (Table 5).

- Do results support the claims?
  - Yes. Three pillars—scaling curves (Fig. 1, Fig. 6), SOTA or near-SOTA scores at 8B scale (Tables 2–3), and ablations (Table 6–7)—demonstrate that (a) performance improves with more frames/longer contexts, and (b) the specific training choices (IAP, ADS, progressive scheduling, Eagle-Video-110K) each contribute.

## 6. Limitations and Trade-offs
- Token accounting assumptions (Sec. 3.2.1)
  - ADS treats each video frame/page as a fixed 256-token unit and does not tile video frames. This prioritizes temporal coverage over per-frame spatial fidelity. Scenarios requiring both very high spatial resolution and long temporal spans might still be constrained.

- Sample dropping when budgets are too tight (Sec. 3.2.1)
  - If the minimum temporal requirement `N_min` can’t be met under `L_visual`, the training sample is discarded. This maintains signal quality but may reduce exposure to some challenging long examples.

- No specialized compression modules (Sec. 3.1)
  - The framework intentionally avoids compression/selection components. While this preserves generality and simplicity, it may underutilize cases where smart content selection could further boost signal-to-noise under strict token budgets.

- Compute and systems complexity (Appendix B)
  - Training relies on substantial systems engineering (fused kernels, context parallelism, video decoding). Reproducing full long-context training at 64K–128K requires significant hardware and software sophistication.

- Data and annotation biases (Sec. 3.3.2)
  - Eagle-Video-110K leverages GPT-4/4o for captions/QA. Benefits are clear, but annotations inherit biases of these models, and availability/licensing of some sources may affect reproducibility.

- Mixed strengths across benchmarks (Tables 2–3)
  - While overall strong, some domains (e.g., MMMU: 55.8) leave headroom compared to much larger models; hallucination scores (HallB 54.7) indicate remaining challenges in robustness.

## 7. Implications and Future Directions
- Field impact
  - Demonstrates that long-context VLMs can be trained to improve with more input, not merely accept it. The combination of token-aware sampling (ADS), spatially faithful tiling (IAP), and progressive schedules provides a practical recipe that others can adopt.

- Practical applications
  - Long-form video analytics (sports, lectures, movies), meeting/minute summarization, episodic memory over hours of video, interactive analysis of multi-page documents and slides, compliance review across long recordings, fine-grained QA over surveillance or instructional videos.

- Follow-up research
  - Smarter hybrid of selection and native long-context: integrate content-aware frame selection or dynamic tile/frame allocation learned end-to-end.
  - Adaptive visual tokenization: relax the fixed “256 tokens per frame” assumption; tile a subset of high-information frames when needed.
  - Multi-stage data curricula: expand the dual-level annotation paradigm to broader domains (e.g., egocentric lifelogging, instructional procedures) and study how anchors can systematically bridge clip↔video supervision.
  - Robustness and safety: targeted training against hallucinations and adversarial long-context pitfalls; improved evaluation suites beyond current benchmarks.

- Deployment considerations
  - The provided systems optimizations (Appendix B) and progressive schedule (Tab. 8) show a path to serving and training long-context VLMs efficiently on modern infrastructure. As hardware improves, these techniques can scale to even longer contexts and richer multimodal interleavings.

In sum, Eagle 2.5 contributes a concrete, reproducible training blueprint—information-first sampling (IAP+ADS), progressive long-context post-training, and a long-video dataset with complementary annotations—that together deliver measurable, scalable gains on long-context multimodal understanding (Figs. 1 & 6; Tables 2–7).
