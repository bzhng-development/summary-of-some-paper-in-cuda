# Scaling RL to Long Videos

**ArXiv:** [2507.07966](https://arxiv.org/abs/2507.07966)

## 🎯 Pitch

This paper introduces a groundbreaking full-stack framework that enables vision-language models to perform sophisticated reasoning over long videos—spanning thousands of frames—by combining a massive new long-video reasoning dataset, a two-stage training process (chain-of-thought supervised fine-tuning plus reinforcement learning), and an efficient new system for reinforcement learning called Multi-modal Reinforcement Sequence Parallelism (MR-SP). The result, LongVILA-R1-7B, achieves state-of-the-art performance on major benchmarks and unlocks practical, scalable video reasoning for applications in robotics, analytics, education, and beyond—moving multimodal AI capabilities from mere recognition to true, temporally extended understanding.

---

## 1. Executive Summary
This paper introduces a full-stack framework for reasoning over long videos with vision–language models (VLMs). It contributes (i) a 104K-example long-video reasoning dataset with chain‑of‑thought, (ii) a two‑stage training recipe that combines chain‑of‑thought supervised fine‑tuning and reinforcement learning (RL), and (iii) a new training system, `MR‑SP` (Multi‑modal Reinforcement Sequence Parallelism), that makes RL on hundreds–thousands of video frames practical. The resulting 7B model, `LongVILA‑R1‑7B`, sets strong results on standard video benchmarks and processes up to 8,192 frames while training up to hour‑long videos on 8×A100 GPUs (Abstract; Fig. 2; Sec. 4–6).

## 2. Context and Motivation
- Problem addressed
  - Long videos require reasoning that spans time (events unfolding over minutes), space (tracking objects across views), goals/intent, and narrative (plots and causal arcs). The paper highlights four reasoning types—Temporal, Goal & Purpose, Spatial, and Plot & Narrative (Fig. 1; Sec. 1, 3).
  - Two blockers have slowed progress: a lack of high‑quality long‑video reasoning data and the difficulty/cost of RL training when inputs contain hundreds–thousands of frames (Sec. 1).

- Why it matters
  - Many real scenarios (sports analytics, robotics, games, vlogs, education) contain long temporal dependencies. Systems that only “recognize” visuals without reasoning over extended context miss crucial signals for decision‑making or question answering (Sec. 1; Fig. 1).

- Prior approaches and gaps
  - Reasoning‑focused multi‑modal works (e.g., LMM‑R1, Vision‑R1, Video‑R1) mainly target single images or short clips (often ≤16 frames) and do not address the engineering and algorithmic hurdles of long‑video RL (Sec. 2).
  - Long-context training methods (sequence parallelism variants, e.g., Ring Attention, Ulysses) exist for LLMs and some VLM SFT (e.g., LongVILA’s `MM‑SP`), but no RL framework had been tailored to the unique sampling and prefilling costs of long videos (Sec. 2).

- Positioning
  - The work offers an end‑to‑end solution: a new dataset focused on long‑video reasoning, a two‑stage reasoning‑oriented training pipeline, and a dedicated RL system (`MR‑SP`) that parallelizes the two most expensive phases for long videos—vision encoding and LLM prefilling—while reusing video embeddings across multiple rollouts (Sec. 3–5).

## 3. Technical Approach
The approach has three pillars: data, training recipe, and a new RL system.

- Data: `LongVideo‑Reason` (Sec. 3; Fig. 3–4, 9)
  - Source videos: 18K long videos from Shot2Story plus ~2K additional 4K videos (autonomous driving, games, home robotics, wildlife) (Sec. 3.1).
  - Clip captioning: Each long video is segmented into ~10s clips; captions are generated with `NVILA‑8B` (Fig. 4).
  - Spatial grounding: For spatial questions, `VILA‑HD` provides object bounding boxes to anchor questions in specific frames (Sec. 3.2).
  - Reasoning Q&A generation: A strong text‑reasoning LLM (`DeepSeek‑R1‑671B`) takes all clip captions from a video and produces Question–Reasoning–Answer triples targeting the four reasoning types (Temporal, Goal/Purpose, Spatial, Plot/Narrative). The chain‑of‑thought is then refined for conciseness and consistency (Sec. 3.2; Fig. 4, 9).
  - Scale and splits:
    - 104K long‑video QA pairs with reasoning across 18K+ videos (Fig. 3).
    - 36K high‑quality examples used for `CoT‑SFT` warm‑up (Sec. 3.1; 4.1).
    - 68K challenging examples for RL, plus 102K extra QAs from other datasets [53, 46, 31, 18, 44] to improve generalization during RL (Sec. 3.1; 4.2; Fig. 3).
  - GRPO‑aware filtering: To avoid useless RL gradients when all rollouts are uniformly correct/incorrect, the pipeline runs the base model `LongVILA` 10 times on each question. “Too easy” and “too hard” items are filtered, retaining samples that induce diverse predictions (Sec. 3.1).

- Training recipe (Sec. 4; Fig. 5)
  - Stage‑1: Long `CoT‑SFT` (chain‑of‑thought supervised fine‑tuning)
    - Purpose: Seed explicit reasoning and instruction following for long videos.
    - Data: 36K CoT examples formatted as `<think>…</think><answer>…</answer>` (Sec. 3.1; 4.1).
    - System: Uses `MM‑SP` (multi‑modal sequence parallelism from LongVILA) to fit hundreds of frames during SFT (Sec. 4.1).
  - Stage‑2: RL with `GRPO` (Group Relative Policy Optimization) (Sec. 4.2; Eq. (1), (2))
    - Core idea: For each question `q`, sample a group of `G` responses from the old policy `πθ_old`, compute rule‑based rewards (format correctness + answer accuracy), transform rewards into normalized advantages within the group, and optimize a clipped PPO‑style objective with a KL regularization to a reference model.
    - Rewards: “format” and “accuracy” signals; advantages are group‑normalized
      > Equation (2): `A_i = (r_i − mean(r_1…r_G)) / std(r_1…r_G)`
    - Objective:
      > Equation (1): maximize (over `θ`) the group average of `min(ratio*A_i, clip(ratio,1−ε,1+ε)*A_i) − β D_KL(πθ || π_ref)`
      where `ratio = πθ(o_i|q)/πθ_old(o_i|q)`, `G=8` in experiments, `ε` and `β` are hyperparameters.
    - Scaling to long videos: Uses the new `MR‑SP` system (next bullet) to tame the rollout and prefilling costs.

- RL system: `MR‑SP` (Multi‑modal Reinforcement Sequence Parallelism) (Sec. 5; Fig. 6–7)
  - Motivation: In RL, each training step needs multiple “rollouts” (sampled answers). With long videos, repeatedly encoding hundreds–thousands of frames and then prefilling long sequences for both policy and reference models dominate runtime and memory.
  - Stage‑1 (rollout): Paralleled vision encoding + embedding reuse (Sec. 5.1; Fig. 7 left)
    - The video’s frames are shard‑split across GPUs; each GPU has its own vision tower to encode its slice.
    - A global ‘all‑gather’ merges these per‑GPU video embeddings with text embeddings into a single sequence.
    - Crucially, these gathered video embeddings are cached and reused across the multiple rollouts for that sample (typically 8–16 rollouts per step), eliminating redundant re‑encoding.
  - Stage‑2 (prefilling): Sequence‑parallel LLM prefilling (Sec. 5.2; Fig. 7 right)
    - After gathering, the long input sequence (video + text) is padded to a uniform length and evenly partitioned by token positions across GPUs (sequence parallelism).
    - Each GPU computes the prefilling (key–value cache construction) for its token slice. This applies to both policy and reference models.
  - Engine: A tailored `vLLM`-based engine is used for high‑throughput rollout sampling with long multimodal sequences (Fig. 6; Sec. 5). `vLLM` is a high‑throughput LLM serving/runtime layer that implements memory‑efficient attention (PagedAttention).

- Why these design choices?
  - CoT‑SFT first: RL with rule‑based rewards tends to explore better when the model already “knows how to think” in a structured way. Sec. 6.2 shows that skipping CoT‑SFT hurts RL results (Table 5).
  - GRPO with filtered data: Group‑normalized advantages need within‑group reward variance; the data filtering step preserves variance by removing trivial or impossible items (Sec. 3.1; 4.2).
  - MR‑SP: Parallelizing the two actual bottlenecks (vision encoding and prefilling) and reusing video embeddings addresses both compute and memory head‑on (Sec. 5; Fig. 2, 7).

## 4. Key Insights and Innovations
- Large, reasoning‑centric long‑video dataset with explicit chains of thought (fundamental)
  - `LongVideo‑Reason` provides 104K QA pairs grounded in whole‑video content with detailed reasoning across four categories (Temporal, Goal/Purpose, Spatial, Narrative) (Sec. 3; Fig. 3–4, 9).
  - Different from prior synthetic or short‑clip datasets, it is built from long videos via caption‑driven prompting of a strong reasoning LLM and includes a dedicated 1K‑sample eval set (`LongVideo‑Reason‑eval`) (Sec. 3.2; Table 2).

- A two‑stage reasoning training pipeline that scales with frames (substantial)
  - Combining `Long CoT‑SFT` (Stage‑1) with `GRPO` (Stage‑2) yields consistent gains and better generalization than either alone (Sec. 6.2; Table 5).
  - The method explicitly targets long‑video reasoning; CoT warms up reasoning, and RL pushes exploration toward better strategies under rule‑based rewards (Sec. 4; Fig. 5).

- `MR‑SP`: a practical RL system for long videos (fundamental engineering contribution)
  - New combination of (i) shard‑parallel video encoding with cached embedding reuse and (ii) sequence‑parallel prefilling for both policy and reference models (Sec. 5.1–5.2; Fig. 7).
  - Achieves up to 2.1× speed‑up at 512 frames and avoids out‑of‑memory where the baseline fails (Fig. 2). This turns long‑video RL from “infeasible” to “trainable” on a single 8×A100 node.

- Demonstrated long‑context capability and frame‑scaling behavior (notable)
  - The model supports up to 8,192 frames at inference and trains hour‑long (≈3,600‑frame) videos on a single 8×A100 node (Abstract; Sec. 7 Conclusion).
  - Performance improves with more frames when reasoning is trained (Table 4), showing the method actually uses the longer context rather than merely tolerating it.

## 5. Experimental Analysis
- Evaluation setup (Sec. 6; Tables 1–5; Fig. 2, 8)
  - Benchmarks and metrics
    - ActivityNet‑QA (accuracy), LongVideoBench (accuracy), PerceptionTest (accuracy), NExT‑QA (multiple‑choice accuracy), VNBench (accuracy), VideoMME (accuracy with and without subtitles) (Table 1).
    - New `LongVideo‑Reason‑eval` (1,000 samples) with four reasoning categories; metric: accuracy (Table 2).
  - Main model
    - `LongVILA‑R1‑7B`, evaluated with 512 input frames on VideoMME (Table 3). Subtitles are treated as an additional modality for the “with subtitle” setting.
  - Systems/hardware details used in timing
    - 8×A100 (80GB) single node; SP degree = 4; batch size = 1 per GPU; rollouts = 5 for timing; times averaged after warm‑up (Sec. 6.2 “Training efficiency on MR‑SP”; Fig. 2).

- Main quantitative results
  - Across six standard video benchmarks (Table 1), `LongVILA‑R1‑7B` improves over `LongVILA‑7B` everywhere:
    > ActivityNet‑QA: 64.8 vs 59.5; LongVideoBench: 58.0 vs 57.1; PerceptionTest: 68.9 vs 58.1; NExT‑QA: 81.5 vs 80.7; VNBench: 75.5 vs 63.0; VideoMME w/o sub: 65.1 vs 60.1; w/ sub: 71.1 vs 65.1.
  - On VideoMME in detail (Table 3):
    > Overall (w/o subtitles): 65.1, with per‑length breakdown Short/Medium/Long = 76.8/63.2/55.2.  
    > With subtitles: Overall 71.1, with Short/Medium/Long = 79.2/69.7/64.3.  
    These are competitive/leading among similarly sized open models, surpassing `LongVILA‑7B` and others such as `LongVA‑7B`, `VITA‑1.5‑7B`, `Kangaroo‑8B`.
  - On the new `LongVideo‑Reason‑eval` (Table 2):
    > Overall accuracy: 72.0%, outperforming `Video‑R1‑7B` (68.1) and slightly above `Gemini‑1.5‑Pro` (69.3). Category breakdown: Temporal 68.1, Goal 85.7, Plot 70.6, Spatial 53.3.

- Efficiency and training dynamics
  - Speed improvements from `MR‑SP` (Fig. 2):
    > Up to 2.1× faster per‑step runtime at 512 frames vs. vanilla RL; avoids OOM beyond 512 frames where the baseline fails. Results shown for both `Qwen2.5‑VL‑7B` and `LongVILA‑R1‑7B`.
  - Reward curves (Fig. 8) show stable growth in overall, format, and accuracy rewards during RL.

- Ablations and scaling behavior (Sec. 6.2)
  - “Frames × Reasoning” (Table 4) on a 1.5B model variant:
    > Without reasoning training: plateaus/degrades at 256–512 frames (60.7 → 60.2).  
    > With reasoning training (CoT‑SFT + RL): steadily improves to 64.3 at 512 frames.
    This indicates the method leverages long context when trained to reason.
  - “Pipeline and datasets” (Table 5):
    > Best results when using both CoT‑SFT and RL with the new dataset; using only RL or replacing CoT‑SFT/RL data with other datasets reduces accuracy.

- Qualitative analyses
  - The appendix shows multi‑minute examples (football, poker, house tour, LEGO, StarCraft) where the model’s reasoning references cues spread across time and space (Fig. 10–16). These illustrate the kinds of extended‑context inferences targeted.

- Convincingness
  - The method is supported by improvements across diverse public benchmarks, a new targeted eval set, and system speedups that enable the training regime. The experiments explicitly test with/without subtitles, short/medium/long splits, and include ablations isolating the contributions of CoT‑SFT and RL.

## 6. Limitations and Trade-offs
- Compute and data costs
  - Data generation is expensive—about 80,000 H100 GPU hours to produce the reasoning annotations (Sec. 3.2). RL training, even with `MR‑SP`, still requires multiple high‑end GPUs (8×A100) (Fig. 2; Sec. 7).
  - While the system can train ≈3,600‑frame (hour‑level) videos on a single node, scaling to much longer sequences, adding modalities like audio, or using larger batches likely requires multi‑node distributed training (Conclusion; “Limitations” paragraph).

- Reward design and supervision quality
  - RL rewards are rule‑based for formatting and answer accuracy (Sec. 4.2). They may not capture nuanced reasoning quality beyond correctness and formatting. Furthermore, dataset construction relies on LLM‑generated CoT; despite refinement, such supervision can import biases or artifacts from the generator (Sec. 3.2; “Border Impacts” Sec. 8 discusses mitigation choices).

- Coverage and modality scope
  - The focus is video+text. Audio reasoning (e.g., dialogues, sound cues) is not included; adding audio would stress both data and system design (Conclusion; “Limitations”).

- Sensitivity of GRPO to sampling
  - The paper explicitly mentions GRPO’s sensitivity to batch sampling and mitigates it with data filtering (Sec. 3.1; 4.2). Residual sensitivity may remain, especially if deployment data distribution differs.

- Spatial reasoning still trails other categories
  - On `LongVideo‑Reason‑eval`, Spatial accuracy (53.3) lags behind other categories (Table 2), suggesting room to improve fine‑grained spatial tracking in long contexts.

## 7. Implications and Future Directions
- Field impact
  - By making long‑video RL viable on moderate hardware and demonstrating clear accuracy gains, `MR‑SP` plus the two‑stage reasoning pipeline provide a blueprint for scaling multi‑modal reasoning beyond short clips. This could shift the default evaluation and training regimes for video‑understanding models toward longer contexts.

- Follow‑up research enabled
  - Richer rewards: extend beyond answer correctness to trajectory faithfulness (evidence‑rewarded RL), human/preferences RLHF, or verifier‑guided rewards that check whether reasoning steps cite correct frames.
  - Multimodal expansion: incorporate audio (speech, ambient sounds), sensor streams (egocentric IMU), or event‑structured representations—all tractable with `MR‑SP`-style parallelism but requiring new engineering.
  - Better spatial/temporal grounding: integrate object/tracklet memories or differentiable video retrieval to close the Spatial gap seen in Table 2.
  - Data efficiency and quality: reduce reliance on massive LLM‑generated CoT via active learning, human verification on hard cases, or distillation from stronger proprietary models while preserving privacy (Sec. 8 outlines privacy‑aware choices).

- Practical applications
  - Embodied AI and robotics: long‑horizon task monitoring and planning; video‑conditioned decision support (Sec. 8).
  - Sports and game analytics: extended tactical reasoning and outcome prediction from full matches (Fig. 1; Appendix examples).
  - Education/healthcare/media: summarizing long instructional or procedural videos and querying complex narratives (Sec. 8).

Overall, the work delivers a coherent stack—data, algorithm, and system—that together make long‑video reasoning with RL feasible and beneficial, evidenced by broad benchmark gains (Tables 1–3) and substantial training speedups (Fig. 2).
