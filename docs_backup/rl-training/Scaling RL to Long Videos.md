# Scaling RL to Long Videos

**ArXiv:** [2507.07966](https://arxiv.org/abs/2507.07966)
**Authors:** Yukang Chen, Wei Huang, Baifeng Shi, Qinghao Hu, Hanrong Ye, Ligeng Zhu, Zhijian Liu, Pavlo Molchanov, Jan Kautz, Xiaojuan Qi, Sifei Liu, Hongxu Yin, Yao Lu, Song Han
**Institutions:** NVIDIA, MIT, UC Berkeley, UT Austin

## 🎯 Pitch

LongVILA-R1 revolutionizes vision-language models by implementing a pioneering framework for reinforcement learning on long videos, featuring a large reasoning dataset and a groundbreaking Multi-modal Reinforcement Sequence Parallelism (MR-SP) system. This advancement enables practical, efficient RL processing of hour-long videos, significantly improving video comprehension across applications in robotics, sports analytics, and healthcare by providing deep causal and narrative analysis where traditional methods fall short.

---

## 1. Executive Summary
This paper introduces LongVILA-R1, a full-stack framework that enables reinforcement learning (RL) on long videos for vision-language models (VLMs). It contributes a new 104K-question reasoning dataset for long videos, a two-stage training recipe (chain-of-thought supervised fine-tuning followed by RL), and a training system called Multi-modal Reinforcement Sequence Parallelism (MR-SP) that makes long-video RL practical and up to 2.1× faster (Figure 2). The result is a 7B model that handles thousands of frames (up to 8,192 at inference) and achieves state-of-the-art accuracy on several video benchmarks (Tables 1–3).

## 2. Context and Motivation
- Problem gap
  - Long video understanding requires reasoning over extended time, space, goals, and narrative—far beyond frame-level recognition. Figure 1 illustrates tasks that need temporal aggregation (soccer penalties), goal inference (poker), and spatial tracking (shell-game).
  - Two key obstacles persist:
    1) Lack of high-quality long-video reasoning data with explanations.
    2) RL training is prohibitively expensive for long sequences because both vision encoding and language prefilling scale with sequence length.

- Why this matters
  - Real applications—robotics, sports analytics, healthcare video review, AR/VR—operate on minutes to hours of video and require causal, goal, and plot reasoning, not just short-clip recognition (Section 1, “Introduction” and Section 8, “Border Impacts”).

- Prior approaches and limits
  - Recent multimodal reasoning efforts (e.g., Video-R1, Vision-R1, LMM-R1) either focus on single images or short videos; long-context solutions emphasize architecture/training for text or perception but not RL over long video sequences (Section 2).
  - Existing sequence-parallel systems help with long context but are not tailored to multi-modal RL, which needs repeated rollouts and heavy prefilling (Section 2).
  - RL frameworks (e.g., GRPO-based) improve reasoning but are compute-heavy; for long videos, rollout/prefill costs explode (Section 2).

- Positioning
  - This work tackles the full problem stack: it builds a large long-video reasoning dataset with chain-of-thought (CoT), proposes a curriculum (CoT-SFT then RL), and engineers MR-SP to parallelize video encoding and LLM prefilling with caching to make long-video RL tractable (Sections 3–5).

## 3. Technical Approach
The system consists of three pillars: a dataset (LongVideo-Reason), a two-stage training pipeline, and an MR-SP RL training system.

- Data: LongVideo-Reason (Section 3, Figures 3–4, 9)
  - What it is
    - 18K long videos and 104K Question–Reasoning–Answer pairs with high-quality reasoning traces (“Long-CoT,” i.e., chain-of-thought for long videos). Questions span four categories: Temporal, Goal & Purpose, Spatial, Plot & Narrative (Figure 3, middle).
    - Balanced, curated 1K-sample evaluation set, LongVideo-Reason-eval, covering the same four reasoning types (Table 2).
  - How it is built (Figure 4 and Figure 9)
    - Each long video is segmented into ~10s clips.
    - NVILA-8B produces captions for each clip; VILA-HD provides bounding boxes for spatial prompts.
    - DeepSeek-R1-671B generates diverse Q–Reasoning–Answer across the whole video using all clip captions; an LLM then refines reasoning for clarity and concision.
    - The pipeline emphasizes video-grounded reasoning verbs (“checking the video,” “analyzing the scene”) to keep the reasoning tied to visual evidence.
  - Data selection for training (Section 3.1)
    - A “test-scaling” filter uses LongVILA to infer each question 10 times:
      - Too-easy (always correct) and too-hard (always wrong) items are removed; medium-difficulty items remain to suit group-based RL (GRPO) which benefits from diverse rollouts.
    - Final splits:
      - 36K high-quality CoT samples → Stage 1 (CoT-SFT).
      - 68K filtered QAs → Stage 2 (RL).
      - Plus 102K additional video QAs from prior datasets to improve generalization [53, 46, 31, 18, 44] (Figure 3, right).

- Training recipe: two stages (Section 4, Figure 5)
  - Stage 1 — Long CoT-SFT
    - Purpose: warm up the model on instruction following and video reasoning using the 36K CoT subset.
    - System: MM-SP (multi-modal sequence parallelism from LongVILA) enables SFT over hundreds of frames efficiently (Section 4.1).
    - What is CoT-SFT? “Chain-of-thought” supervised fine-tuning trains the model to produce intermediate reasoning steps before the answer. Here, the format is `<think>...</think><answer>...</answer>`.
  - Stage 2 — RL with GRPO (Sections 4.2 and 5)
    - What is GRPO? Group Relative Policy Optimization samples G outputs per question from a frozen “old” policy, computes rule-based rewards (format/accuracy), normalizes them within the group to compute advantages, and updates a new policy using a clipped PPO-style objective with a KL penalty to a reference policy (Equation (1)).
      - Equation (2) shows advantage computation by z-scoring rewards across the G rollouts.
      - In experiments, G=8, and rewards are rule-based (format correctness and QA accuracy) (Figure 6).
    - Why GRPO here? Its group normalization is well-suited to medium-difficulty questions with varied outcomes, which the data filter ensures.

- MR-SP: Multi-modal Reinforcement Sequence Parallelism (Section 5, Figures 6–7)
  - Problem: RL for long videos repeatedly performs expensive video encoding and long-context prefilling for the policy and reference models across many rollouts.
  - Solution: MR-SP parallelizes both the visual and language sides and caches cross-rollout embeddings.
    - Stage 1 — Rollout with paralleled encoding (Section 5.1)
      - Video frames are sharded across multiple GPUs; each GPU runs a vision tower on its shard to produce partial embeddings.
      - An all-gather assembles global video embeddings, which are then cached and reused across all rollouts in that step (typically 8–16). This avoids re-encoding the same video multiple times (Figure 7, “Embeddings copy”).
    - Stage 2 — Prefilling with sequence parallelism (Section 5.2)
      - What is “prefilling”? The compute-intensive pass that consumes the entire prompt (here, text plus long video embeddings) to build the LLM’s key-value memory before token-by-token generation.
      - MR-SP pads the global input to a uniform length and shards it across GPUs so each device pre-fills only a chunk. This applies to both policy and reference models.
      - It integrates a vLLM-based generation engine tailored for LongVILA, leveraging PagedAttention and efficient memory management (Figure 6).
  - Outcome: MR-SP removes redundant vision encoding and splits long prefills across GPUs, addressing both the memory and time bottlenecks.

- Model capability and system limits
  - LongVILA-R1-7B supports up to 8,192 frames per video (inference) with configurable FPS (Abstract, Section 6.1).
  - On a single 8×A100 node, RL training can use hour-long videos (≈3,600 frames) (Abstract; Conclusion).

## 4. Key Insights and Innovations
- LongVideo-Reason: a large, reasoning-centered dataset for long videos (Sections 3.1–3.2; Figures 3–4, 9)
  - Novelty: pairs long videos with multi-step reasoning across four reasoning facets and includes both multiple-choice and open-ended formats (~50/50).
  - Significance: enables training/evaluating explicit reasoning rather than only recognition; provides a new 1K-sample evaluation benchmark (Table 2).

- Two-stage training for long-video reasoning (Section 4; Figure 5)
  - Novelty: combines CoT-SFT on curated medium-difficulty data with RL via GRPO; the filtering ensures diverse rollouts and non-vanishing advantages.
  - Significance: ablations show that CoT-SFT provides a strong warm-up and RL adds further gains; skipping the warm-up degrades RL (Table 5).

- MR-SP: efficient RL system for long video (Section 5; Figures 6–7; Figure 2)
  - Novelty: caches all-gathered video embeddings across rollouts and applies sequence parallelism to prefilling for both policy and reference models; integrates a vLLM engine.
  - Significance: up to 2.1× speedup at 512 frames and avoids OOM while scaling to 1,024 frames per step on 7B models (Figure 2). Enables hour-long RL on 8×A100.

- LongVideo-Reason-eval: targeted evaluation for long-video reasoning (Table 2)
  - Novelty: balanced coverage of temporal, goal, spatial, and narrative reasoning.
  - Significance: LongVILA-R1-7B reaches 72.0% average, exceeding Video-R1-7B (68.1%) and slightly surpassing Gemini-1.5-Pro (69.3%), highlighting strong generalized reasoning.

## 5. Experimental Analysis
- Evaluation setup (Section 6)
  - Benchmarks: ActivityNet-QA, LongVideoBench, PerceptionTest, NExT-QA, VNBench, and VideoMME (Table 1); plus the new LongVideo-Reason-eval (Table 2).
  - Metrics: Multiple-choice accuracy; VideoMME reported with and without subtitles and by video length (Table 3).
  - Baselines: Both closed (GPT-4o, Gemini-1.5-Pro) and open models (Video-LLaVA, VideoLLaMA2(.1), Kangaroo-8B, LLaVA-OV-7B, LongVILA-7B, etc.; Tables 1 and 3).

- Main quantitative results
  - Overall gains across benchmarks (Table 1):
    - LongVILA-R1-7B vs LongVILA-7B
      - ActivityNet-QA: 64.8 vs 59.5
      - LongVideoBench: 58.0 vs 57.1
      - PerceptionTest: 68.9 vs 58.1
      - NExT-QA (val mc): 81.5 vs 80.7
      - VNBench: 75.5 vs 63.0
      - VideoMME: 65.1 (no subs) / 71.1 (with subs) vs 60.1 / 65.1
  - VideoMME by length (Table 3):
    - Without subtitles: 65.1 overall; Short 76.8, Medium 63.2, Long 55.2
    - With subtitles: 71.1 overall; Short 79.2, Medium 69.7, Long 64.3
    - These surpass LongVILA-7B (60.1/65.1 overall).
  - LongVideo-Reason-eval (Table 2):
    - LongVILA-R1-7B: Temporal 68.1, Goal 85.7, Plot 70.6, Spatial 53.3, Overall 72.0
    - Compared to Video-R1-7B: 61.4 / 85.0 / 62.0 / 58.5 / 68.1
    - Compared to Gemini-1.5-Pro: 65.4 / 81.9 / 67.8 / 53.3 / 69.3
    - Strengths: Goal and Plot reasoning; Spatial lags relative to Goal (53.3 vs 85.7).
  - Efficiency (Figure 2):
    - At 512 frames on 8×A100: MR-SP achieves up to 2.1× speedup over a plain RL system and avoids OOM beyond 512 frames when combining both stages (rollout reuse + SP prefilling).
    - Reward curves show stable training (Figure 8).

- Ablations and scaling behavior
  - Frames vs reasoning (Table 4; 1.5B scale):
    - With RL “reasoning” (R.), accuracy steadily improves with more frames, reaching 64.3 at 512 frames; without R., performance plateaus then drops at 512 (60.2).
    - Interpretation: reinforced reasoning helps exploit additional temporal context instead of being overwhelmed by long inputs.
  - Pipeline/dataset choices (Table 5; 1.5B scale):
    - CoT-SFT only (✓, RL ✗): 60.2
    - RL only (✗, ✓): 52.4 (poor without warm-up)
    - Both with this paper’s data (✓, ✓): 61.9
    - Using other datasets (O) in either stage yields lower scores (59.1–59.4) than this dataset.

- Qualitative evidence
  - Figure 1 and Figures 10–16 show cases where increasing frames enables spatial tracking (e.g., success only after 128 frames), and where LongVILA-R1 provides deeper, video-grounded reasoning than baselines for sports, games, and narrative inference.

- Do the experiments support the claims?
  - Yes, on two fronts:
    - Capability: The model consistently improves over LongVILA-7B across six standard benchmarks (Table 1) and reaches strong performance on the new evaluation (Table 2).
    - Efficiency: MR-SP reduces per-step time and removes OOM barriers (Figure 2), with stable reward improvements (Figure 8).
  - Caveats: Spatial reasoning remains the weakest category (Table 2), and long-video “long” in some benchmarks still differs from hour-long training scenarios; however, the model demonstrates both scaling ability (Table 4) and practical throughput (Figure 2).

## 6. Limitations and Trade-offs
- Compute and scalability
  - Dataset creation is costly (~80,000 H100 GPU hours; Section 3.2).
  - While MR-SP enables 3,600-frame RL on 8×A100, scaling to even longer sequences, more modalities (e.g., adding audio), or larger batch sizes likely requires multi-node or more GPUs (Conclusion “Limitations”).
- Reward design
  - RL uses rule-based rewards for accuracy and output format (Figure 6), not human preference modeling. This may limit the nuance of reasoning alignment and risks reward hacking on multiple-choice tasks.
- Data quality and bias
  - Many labels and rationales are generated by LLMs (DeepSeek-R1-671B), which can introduce artifacts or hallucinations; although a refinement step exists (Section 3.2), residual noise may persist.
  - The paper’s “Border Impacts” section discusses privacy-conscious design (Section 8), but any long-video system could still be misused for surveillance-like analysis if repurposed.
- Coverage and generalization
  - Spatial reasoning remains comparatively weak (53.3 in Table 2); tasks requiring fine-grained 3D tracking or occlusion-heavy scenes may challenge the model.
  - Audio is not modeled; tasks needing speech, music, or sound cues are out of scope.
- Engineering specificity
  - MR-SP relies on a particular stack (vLLM integration, custom sharding, embedding caching). Porting to other serving/training stacks may require substantial engineering.

## 7. Implications and Future Directions
- Field impact
  - Demonstrates that RL can be scaled to genuinely long videos by rethinking both data and systems. MR-SP offers a general recipe for multi-modal long-context RL, not just for this model.
  - Provides a public training system that supports RL for videos, text, audio, and even image/video generation models, broadening community adoption (Abstract; Conclusion).

- Follow-up research enabled
  - Richer rewards: move beyond accuracy/format to human preference scores, temporal consistency checks, or causal grounding metrics.
  - Multi-modal fusion: incorporate audio tracks and ASR to improve temporal and plot reasoning; extend MR-SP to tri-modal inputs.
  - Memory and retrieval: couple MR-SP with retrieval over long videos (keyframe selection, event indexing) to further reduce compute while improving accuracy.
  - Spatial/3D reasoning: strengthen object permanence and 3D scene understanding; combine with geometric inductive biases or video tracking modules.
  - Safety and governance: build safeguards for long-video analytics; explore privacy-preserving training and auditing tools.

- Practical applications
  - Robotics and embodied AI: long-horizon task tracking, plan monitoring, and error recovery (Section 8).
  - Sports/education/media: game analytics, lecture summarization, narrative understanding over entire episodes or films.
  - Healthcare/operations: review of long procedural videos and anomaly detection—subject to strict privacy and compliance controls.

> Headline results to remember:
> - Speed: MR-SP yields up to 2.1× step-time speedup at 512 frames and avoids OOM at longer inputs (Figure 2).
> - Accuracy: LongVILA-R1-7B reaches 65.1% (no subtitles) and 71.1% (with subtitles) on VideoMME (Tables 1 and 3), and 72.0% on the new LongVideo-Reason-eval (Table 2).
> - Scale: Processes up to 8,192 frames per video at inference and supports RL on hour-long videos on a single 8×A100 node (Abstract; Conclusion).

Overall, this paper closes a critical gap by making long-video reasoning with RL feasible and effective, combining a reasoning-centric dataset, a tailored training curriculum, and an efficient parallel RL system.
