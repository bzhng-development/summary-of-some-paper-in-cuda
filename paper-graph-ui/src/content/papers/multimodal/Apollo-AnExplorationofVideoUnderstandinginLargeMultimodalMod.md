# Apollo: An Exploration of Video Understanding in Large Multimodal Models

**ArXiv:** [2412.10360](https://arxiv.org/abs/2412.10360)

## 🎯 Pitch

Apollo presents the first systematic exploration of what actually enables effective video understanding in Large Multimodal Models, experimentally mapping the effect of key design and training choices across model and dataset scales. The paper introduces the concept of Scaling Consistency, showing that design decisions validated on mid-sized models and datasets reliably translate to larger, expensive settings—dramatically accelerating innovation and lowering resource barriers. Building on these insights, the authors provide ApolloBench for faster, more meaningful evaluation and launch Apollo, a new model family that sets a new state of the art for open video-LMMs under 30B parameters, empowering the whole community to drive progress in video AI.

---

## 1. Executive Summary (2–3 sentences)
This paper systematically maps the design space of video-capable Large Multimodal Models (video-LMMs), identifies what actually improves video understanding, and distills those findings into a compact, fast benchmark (ApolloBench) and a new model family (Apollo) that is state-of-the-art at 7B parameters. The central methodological advance is Scaling Consistency: design decisions validated on moderately sized models (≈2–4B) and mid-sized datasets (≈500K samples) reliably predict what will work at larger scales, drastically reducing compute needed for exploration (Sec. 3; Fig. 3).

## 2. Context and Motivation
- Problem addressed
  - Training and testing video-LMMs is expensive, and the community lacks principled guidance on critical design choices (e.g., how to sample frames, which visual encoders to use, how to compress video tokens, what data mix to train on). As a result, many choices are heuristics without clear justification (Sec. 1).
  - Evaluation is also inefficient and noisy: many popular video QA benchmarks can be “solved” without true video perception—by using text-only inputs or a single frame—and different benchmarks are highly redundant (Sec. 2; Fig. 2).

- Why it matters
  - Practically, video understanding underpins applications in assistive tech, education, robotics, sports, and long-form media. Efficient research cycles and reliable evaluation are essential.
  - Scientifically, the paper clarifies which architectural, training, and data decisions truly contribute to temporal understanding—as distinct from language-only pattern matching.

- Prior approaches and gaps
  - Most recent video-LMMs extend image-LMMs with minimal video-specific analysis (e.g., uniform frame sampling, average pooling, or image encoders only) (Sec. 1; Sec. 7).
  - Extensive “scaling laws” exist for training-from-scratch models, but LMMs are hybrids of pre-trained parts; it was unclear whether those laws apply to design choices for LMMs (Sec. 3).
  - Benchmarks often emphasize language modeling or single-image cues; some even show high text-only accuracy, obscuring real video perception progress (Sec. 2; Fig. 2).

- Positioning
  - This work: (1) audits popular benchmarks and curates a leaner, perception-focused alternative (ApolloBench) (Sec. 2.3); (2) introduces Scaling Consistency to make small-scale studies predictive for larger models (Sec. 3); (3) performs systematic ablations on video sampling, encoders, token resampling/integration, training schedules, and data mix (Secs. 4–5); and (4) packages the winning recipe into Apollo models that achieve state-of-the-art results for their size (Sec. 6; Table 4).

## 3. Technical Approach
This is an empirical, end-to-end investigation combining benchmark analysis, controlled design ablations, and final system building.

- Benchmark audit and curation (Sec. 2)
  - Method: Evaluate 10 open-source LMMs on multiple video QA benchmarks (Video-MME, TempCompass, LongVideoBench, MLVU, NExTQA, Perception-Test) under three settings: full video, single center frame, and text-only (Sec. 2.1).
  - Finding: Many questions are answerable by text alone or a single frame; long-video versions often increase reliance on language, not video (Fig. 2, left). Benchmarks are highly correlated (redundant) across datasets, durations, and question types (Fig. 2, right; App. Figs. 12–14).
  - Output: ApolloBench—400 multiple-choice questions filtered to require video perception (remove those where >50% of models succeed using text or image alone), categorized into five temporal perception types (Temporal OCR, Egocentric, Spatial, Perception, Reasoning), entropy-selected and manually verified (Sec. 2.3; App. Fig. 11). Evaluation is 41× faster yet remains highly correlated with larger suites (Fig. 2, right).

- Scaling Consistency (Sec. 3)
  - Concept: Instead of full scaling-law studies, test whether “which design choice is better” correlates between small and large LMMs. Train 21 design variants across four LLM sizes (0.5B, 1.5B, 4B, 7B) for 84 total models and compare per-variant scores across sizes (App. Fig. 15).
  - Result: Design rankings at 2–4B strongly predict 7B choices (R²=0.938 for 4B→7B). Correlation improves roughly log-linearly with model size. 0.5B is too small to be predictive (R²<0.8) (Fig. 3, left; Sec. 3).
  - Dataset size: With a fixed design, increasing data beyond ≈500K samples yields diminishing returns for correlation to 7B (Fig. 3, right).

- Design-space exploration (Secs. 4–5)
  - Setup: Use a 3B LLM (Qwen2.5-3B) and ≈750K samples as the “design lab” (justified by Scaling Consistency, Sec. 3). Unless stated, videos are encoded at 2 fps, resampled to 16 tokens/frame, using a Perceiver Resampler; vision towers are InternVideo2 + SigLIP-SO400M (Sec. 4).
  - Explored dimensions:
    - Video sampling: compare uniform vs fixed frames-per-second (fps) sampling; vary tokens-per-second (`tps`) and tokens-per-frame (`tpf`) (Sec. 4.1; Fig. 4; App. Figs. 9–10).
    - Video representation: single vs dual encoders; language-supervised (e.g., SigLIP) vs self-supervised (e.g., DINOv2, V-JEPA, VideoMAE), and image vs video encoders (Sec. 4.2; Fig. 5; App. Table 9).
    - Token resampling: MLP/Conv + average pooling vs Perceiver Resampler (Sec. 4.3; Table 1).
    - Token integration: how to interleave video and text tokens—plain concatenation, learned separation tokens, textual timestamps, or both (Sec. 4.4; Table 2).
    - Training schedules: one-, two-, and three-stage regimes, with progressively unfrozen modules and staged data mixtures (Sec. 5.1; Table 3).
    - What to train in each stage and on what data: whether/when to finetune encoders and whether to use video-only or mixed data when encoders are trainable (Sec. 5.2).
    - Data composition: how much text vs image vs multi-image vs video in SFT; test from 0–25% text with balanced vs video-heavy variants (Sec. 5.3; Fig. 6; App. Table 13).

- Final model family: Apollo (Sec. 6; Fig. 8; Table 7)
  - Architecture
    - Dual encoders: `SigLIP-SO400M` (image-language, strong spatial recognition) and `InternVideo2` (video foundation model capturing temporal cues). Each processes short clips; outputs are interpolated and channel-concatenated (Sec. 4.2; Fig. 8).
    - Connector + Perceiver Resampler: up-projects to LLM hidden size, then compresses visual tokens to a fixed budget (e.g., 32 tokens/frame used in Apollo) using the Perceiver (Sec. 4.3; Table 1).
    - Token integration: textual timestamps (“clip from {MM:SS}-{MM:SS}: <vid_token>”) between clip tokens to mark temporal boundaries (Sec. 4.4; Table 2).
  - Video sampling at training and inference
    - Fixed-fps sampling at the clip level; for long videos, keep per-clip fps constant and space out the clips rather than spacing individual frames (Sec. 4.1; C.1). This avoids changing the “effective playback speed” that uniform sampling introduces.
  - Training (3 stages; Table 7)
    - Stage 1 (Alignment): freeze LLM and encoders; train connector on mixed image/video captions.
    - Stage 2 (Vision pretraining): finetune only the encoders (on video-only captions) for better vision-language alignment (Sec. 5.2).
    - Stage 3 (Supervised finetuning): unfreeze LLM; train on a curated T+I+multi-image+video mix, with ≈10–14% text and slightly video-heavy imagery (Sec. 5.3; Fig. 6).

Definitions for uncommon terms used above:
- `fps sampling`: sampling frames at a fixed physical rate (e.g., 2 frames per second) regardless of video length.
- `uniform sampling`: picking a fixed number of frames uniformly across the entire video; the effective fps then varies with video length (leading to different “playback speeds” per example).
- `tokens per second (tps)` and `tokens per frame (tpf)`: how many visual tokens the model processes per second of video or per frame; controlled via the resampler. They trade off temporal vs spatial detail given a fixed token budget (Sec. 4.1; Fig. 4 right).
- `Perceiver Resampler`: a learned module (Jaegle et al., 2021) that compresses many image/video patch embeddings into a smaller set of latent tokens via cross-attention, preserving task-relevant information under a token budget (Sec. 4.3; Table 1).

## 4. Key Insights and Innovations
- Scaling Consistency (fundamental innovation; Sec. 3)
  - What’s new: A correlation-based principle for LMM design selection showing that design choices validated at 2–4B parameters on ≈500K samples transfer robustly to 7B (R² up to 0.938; Fig. 3 left/right; App. Fig. 15).
  - Why it matters: Researchers can iterate with small, cheaper models to pick architectures, sampling strategies, and data mixes, confident these choices will hold when scaling up—dramatically reducing exploration cost.

- Benchmark triage and ApolloBench (fundamental innovation; Sec. 2)
  - What’s new: A diagnostic showing that many widely used video QA benchmarks reward language-only or single-frame reasoning (Fig. 2 left), plus a highly correlated, 41× faster, perception-focused subset (ApolloBench) filtered to remove questions solvable by text or a single frame (Sec. 2.3; Fig. 2 right).
  - Why it matters: Enables inexpensive, high-signal evaluation tied to actual video perception (temporal OCR, egocentric cues, spatial/physical reasoning).

- Video sampling at constant fps and token budgeting (substantial innovation; Sec. 4.1)
  - What’s new: Demonstrates that training with fixed-fps clips (and spacing out clips for long videos) consistently outperforms uniform frame sampling, independent of test-time frame counts (Fig. 4 left/middle; App. Table 11).
  - Why it matters: Avoids teaching the model inconsistent “playback speeds,” leading to better temporal understanding. Also quantifies a usable `tps`–`tpf` trade-off with 8–32 tokens/frame as a sweet spot (Fig. 4 right; App. Figs. 9–10).

- Visual representation strategy (substantial innovation; Sec. 4.2)
  - What’s new: Finds `SigLIP-SO400M` is the single best encoder overall among tested vision models (image or video), while combining a language-supervised image encoder (SigLIP) with a video encoder (InternVideo2) yields the strongest performance (+~7% on ApolloBench vs single encoders; App. Table 9 row 19).
  - Why it matters: Clarifies that today’s video encoders contribute temporal cues but need an image encoder’s strong spatial semantics; language-supervised pretraining outperforms self-supervised options in this LMM setting (Fig. 5).

- Token resampling and integration (incremental but consistently beneficial; Secs. 4.3–4.4)
  - Perceiver Resampler outperforms average pooling approaches across metrics (Table 1; overall 55.5 vs 53.2/44.7).
  - Adding delimiters—learned tokens or textual timestamps—between clips improves performance by 2–3% (Table 2; best is textual timestamps with overall 56.8).

- Training schedule and data composition (substantial innovation; Sec. 5)
  - Progressive unfreezing across 3 stages is best (Table 3; overall 59.2 vs 57.8 for 2-stage and 48.7 for 1-stage).
  - Finetuning vision encoders on video-only data during the vision stage improves reasoning and egocentric tasks (Sec. 5.2).
  - Keeping ≈10–14% pure text during SFT prevents catastrophic forgetting and helps overall performance; a slight video-heavy mix is optimal (Fig. 6; App. Table 13).

## 5. Experimental Analysis
- Evaluation methodology
  - Benchmarks: TempCompass, MLVU, Perception-Test, Video-MME (short/medium/long; with/without subtitles), LongVideoBench, and ApolloBench (Sec. 2; Table 4).
  - Modalities for audit: video vs single frame vs text-only (Sec. 2.1; Fig. 2 left).
  - Design ablations: sampling, encoders, resamplers, integration, schedule, encoder training, data mix (Secs. 4–5; Tables 1–3; Fig. 4–6; App. Tables 9–13).
  - Final models: Apollo-1.5B/3B/7B built from the winning recipe (Sec. 6; Table 7 for training specifics).

- Main quantitative results (Table 4)
  - Apollo-7B is state-of-the-art among 7B models on major suites:
    - MLVU: 70.9 (beats Oryx-34B at 70.8; matches/exceeds 7B peers) and achieves 66.3 overall on ApolloBench.
    - Video-MME (w/o subtitles): 61.3/63.3 (w/ subs), competitive with larger open-weight models and ahead of most 7B baselines.
  - Apollo-3B rivals or beats recent 7B models:
    - MLVU: 68.7 (surpasses Oryx-7B at 67.5 and Video-XL-7B at 64.9).
    - Video-MME (w/o subs): 58.4 (vs Oryx-7B 50.3).
    - ApolloBench: 62.7 (close to LLaVA-OV-7B at 64.0; still beats several 7B models; Table 4).
  - Apollo-1.5B outperforms many models larger than itself (e.g., LongVU-3.2B) with MLVU 63.3 and ApolloBench 57.0.

- Do the ablations support the claims?
  - Yes. Each major design choice is tied to a clear, controlled gain:
    - Sampling: Uniform training underperforms even when tested with fps sampling (Fig. 4 left/middle; App. Table 11; best uniform overall ≈55.1 vs fps configurations reaching ≈58–59 on ApolloBench; App. Table 10 rows 17–19).
    - Encoders: `SigLIP` tops single encoders (Fig. 5 left), and `InternVideo2 + SigLIP` is the best dual combo (Fig. 5 right; App. Table 9 row 19).
    - Resampler: Perceiver > pooled MLP/Conv across all metrics (Table 1).
    - Integration: inserting timestamps or learned separators helps by ~2–3% (Table 2).
    - Schedule: 3-stage progressive unfreezing best (Table 3; 59.2 vs 57.8 vs 56.3 vs 48.7; multiple configs tested).
    - Data mix: ≈10–14% text necessary; slight video-heavy mix best (Fig. 6; App. Table 13).

- Robustness checks and diagnostics
  - Benchmark audit shows that many datasets reward language or single-frame shortcuts; ApolloBench mitigates this confound and remains predictive (Fig. 2).
  - Scaling Consistency repeated across two Qwen families (Qwen2 and Qwen1.5) and multiple dataset sizes; 0.5B models fail to correlate well, identifying a “critical size” (Fig. 3; App. Fig. 15).

- Mixed or conditional results
  - As videos get longer in Video-MME, the added challenge often stems from language context, not necessarily temporal vision—model scores for long videos correlate strongly with short/medium (R² 0.83–0.97) (App. Fig. 13). This motivates using benchmarks like ApolloBench that emphasize perception.

## 6. Limitations and Trade-offs
- Assumptions and scope
  - Scaling Consistency is correlation-based. It demonstrates strong transfer from 2–4B to 7B on the studied families and datasets (Fig. 3; App. Fig. 15), but it does not prove universality for all architectures or tasks, especially far beyond 7B or in drastically different model families.
  - The study focuses on multiple-choice evaluation to ensure consistency and cost control; open-ended conversational quality is not directly measured, and MCQ-optimized models may not be ideal assistants (Sec. 2.3; App. B.1 “Question types”).

- What is not addressed
  - Agentic workflows, retrieval/memory, or active frame/token selection are explicitly out of scope in the ablations (Sec. 4.1, 4.3).
  - The work uses a unified visual pipeline; potential benefits of split image/video pathways remain an open avenue (App. A; App. Table 8 shows only slight differences in a small test).

- Computational and data costs
  - Although exploration is made cheaper via Scaling Consistency, training final Apollo models remains compute-intensive (e.g., 128 A100 GPUs; App. C.4). Long-video capability still depends on careful token budgeting (fps, tps, tpf) and resampling.
  - Due to licensing, some widely used datasets (e.g., ChatGPT-generated) were excluded, which could limit peak performance compared to less constrained settings (Sec. 6; C.3).

- Open questions
  - How do these findings scale to much larger LLMs (e.g., 30–70B) with different backbones?
  - Can future video encoders close the gap so that a single video encoder rivals the dual-encoder setup?

## 7. Implications and Future Directions
- How this changes the field
  - Provides a roadmap of “what matters” for video-LMMs: constant-fps training, dual (language-supervised image + video) encoders, Perceiver resampling, timestamped token integration, progressive unfreezing, and a specific data mix (≈10–14% text) (Secs. 4–5).
  - Establishes a compute-savvy development protocol via Scaling Consistency: do ablations at 2–4B on ≈500K samples, then scale (Sec. 3).
  - Offers a practical, perception-focused evaluation (ApolloBench) that is fast and predictive, enabling frequent, low-cost testing (Sec. 2.3).

- Follow-up research enabled
  - Architectures: Explore split vs unified pipelines at larger scales; investigate better video encoders that integrate temporal cues without sacrificing spatial semantics (App. A; Sec. 4.2).
  - Memory and retrieval: Evaluate long-context memory, frame selection, and multi-turn conversational robustness—explicitly left for future work (App. A).
  - Evaluation: Develop a conversational benchmark that avoids the instability and cost of LLM-graded free-form answers (App. A).

- Practical applications
  - Long-form video analysis (lectures, meetings, sports, instructional content), temporal reasoning for robotics and egocentric assistants, video QA in education and accessibility tools, and enterprise media analytics—where constant-fps sampling, tight token budgets, and robust temporal cues are crucial.

> Representative results: “Apollo-7B attains 70.9 on MLVU and 61.3/63.3 on Video-MME (w/o/w subtitles), and 66.3 on ApolloBench” (Table 4). “Apollo-3B scores 68.7 on MLVU and 58.4 on Video-MME (w/o subs), outperforming many 7B models” (Table 4). “Perceiver Resampler achieves the best overall (55.5) versus MLP pooling (53.2) and Conv pooling (44.7)” (Table 1). “Textual timestamps between clips yield the strongest integration (overall 56.8)” (Table 2). “Three-stage progressive unfreezing reaches 59.2 overall, higher than 2-stage (57.8) and 1-stage (48.7)” (Table 3). “Including ≈10–14% text during SFT and a slightly video-heavy mix is optimal” (Fig. 6). “Design correlations: 4B→7B R²=0.938; dataset correlation plateaus at ≈500K samples” (Fig. 3).
