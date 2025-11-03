# GLM-4.5: Agentic, Reasoning, and Coding (ARC) Foundation Models

**ArXiv:** [2508.06471](https://arxiv.org/abs/2508.06471)

## 🎯 Pitch

GLM-4.5 introduces an open-source Mixture-of-Experts (MoE) large language model that unifies top-tier agentic tool use, complex reasoning, and real-world coding abilities—domains historically siloed in both research and deployment. Through a novel multi-stage training and reinforcement learning pipeline, GLM-4.5 achieves state-of-the-art results across 12 ARC benchmarks while remaining highly efficient, closing the gap between open and proprietary models and paving the way for more capable, unified AI agents in real-world applications.

---

## 1. Executive Summary (2–3 sentences)
GLM-4.5 introduces an open-source family of large language models that aims to unify three capabilities often evaluated separately: agentic tool use, complex reasoning, and real-world coding (ARC). It combines a deep Mixture-of-Experts (MoE) architecture with a staged pretrain→mid-train→post-train pipeline that includes specialized reinforcement learning (RL) for reasoning, agents, and instruction following, yielding top-tier performance across 12 benchmarks while activating only 32B of its 355B total parameters (Figure 1; Table 1).

## 2. Context and Motivation
- Problem/gap addressed
  - Most open-source models excel in only one or two of: complex reasoning (e.g., math), agentic tool use (function calling, web browsing), or software engineering tasks (e.g., SWE-bench). A single open model that is competitive across all three has remained elusive (Introduction, p. 1–2; Figure 1).
  - Prior models either specialize (e.g., reasoning via heavy RL) or remain general but underperform on real-world agentic tasks such as multi-step function calling or browsing.

- Why this matters
  - Real-world productivity increasingly requires LLMs that can reason through multi-step problems, interact with tools/APIs, and modify large codebases—all within long contexts (Introduction, p. 1–2). A unified model reduces engineering complexity and improves reliability in end-to-end systems (agents that must plan, call tools, and verify outcomes).

- Prior approaches and shortcomings
  - Proprietary systems (o1/o3, Claude Sonnet 4) show strong results in specific ARC niches but are closed (Introduction). Open-source MoE systems (DeepSeek-V3, Kimi K2, Qwen3) are strong but still show gaps in either agentic robustness, long-context reasoning, or large-scale code modification (Table 3–5 comparisons).
  - Existing RL recipes often target math alone; less attention has been devoted to code/science RL and to RL in fully agentic environments with verifiable outcomes (Section 3.2–3.3; Figure 7).

- How GLM-4.5 positions itself
  - Provides two open models—`GLM-4.5` (355B total/32B active) and `GLM-4.5-Air` (106B/12B active)—with a hybrid “thinking” and “direct response” mode realized through expert-model distillation and RL (Abstract; Section 3). It aims for Pareto efficiency (performance vs. parameters) and breadth across ARC (Figures 1–2).

## 3. Technical Approach
This section unpacks the end-to-end recipe: model design, data/long-context training, and multi-track RL culminating in a hybrid-reasoning generalist.

- Architecture (Section 2.1; Table 1)
  - Deep MoE Transformer with “loss-free balance routing” (a balancing strategy that updates gating biases during pretraining without auxiliary token-level losses) and sigmoid gates in MoE layers.
  - Design choices:
    - Deeper over wider: fewer-routed-expert width but more layers, observed to improve reasoning (Section 2.1).
    - Grouped-Query Attention with partial RoPE and 2.5× more attention heads than typical for the same hidden size (96 heads at hidden 5120). While this did not lower training loss, it consistently improved reasoning benchmarks (Section 2.1).
    - `QK-Norm` applied to stabilize attention logits (Table 1).
    - A dedicated MoE `MTP` (Multi-Token Prediction) layer supports speculative decoding (Section 2.1).
  - Scale (Table 1):
    - `GLM-4.5`: 355B total params, 32B active; 89 MoE layers + 3 dense layers; 160 experts total with 8 active per token; 128K max context after mid-training.
    - `GLM-4.5-Air`: 106B total, 12B active; 45 MoE layers + 1 dense layer; 128 experts total with 8 active per token.

- Pre-training data and stages (Sections 2.2–2.3; Figure 3)
  - Web corpus: bucketing by quality scores (more epochs on the highest-quality bucket), with semantic deduplication (`SemDedup`) to remove templated auto-generated pages that MinHash misses (Section 2.2).
  - Multilingual documents: quality classifier emphasizes educational utility; high-quality samples are up-sampled.
  - Code data:
    - Curated from GitHub and code hosting sites; three-tier quality filtering; low-quality dropped; high-quality up-sampled.
    - `Fill-In-the-Middle (FIM)` objective applied to source code (Section 2.2).
    - Code-related web pages identified via HTML tags or a FastText classifier; then re-parsed to preserve code formatting.
  - Math & science texts: scored by an LLM then distilled into a small classifier; high-scoring documents up-sampled (Section 2.2).
  - Mid-training (Figure 3):
    - Repo-level code: concatenated files, associated issues/PRs/commits with diff-style formatting to learn cross-file dependencies (Section 2.3). Context extended to 32K to fit repositories.
    - Synthetic reasoning corpora: math, science, coding competition problems; reasoning traces synthesized by a reasoning model (Section 2.3).
    - Long-context & agent data: up to 128K sequence length with up-sampled long documents and synthetic agent trajectories (Section 2.3). Best-fit packing is applied here to avoid truncating long reasoning or repository spans.

- Optimization and scaling details (Section 2.4)
  - Optimizer: `Muon` on all parameters except embeddings, bias, and RMSNorm weights; Newton-Schulz steps N=5, momentum μ=0.95, update RMS=0.2.
  - LR schedule: cosine decay; warm-up to 2.5e-4 then decay to 2.5e-5 through mid-training.
  - Batch-size warmup: 16M → 64M tokens over first 500B tokens; weight decay 0.1; no dropout.
  - Long-context support:
    - Pretraining max length 4,096; extended to 32,768 and 131,072 during mid-training (Figure 3).
    - RoPE base frequency adjusted from 10,000 to 1,000,000 when extending to 32K (Section 2.4).
  - MoE routing: bias update rate 0.001 for first 15T tokens then 0; auxiliary sequence-level balance loss weight 1e-4 to avoid per-sequence imbalance.
  - MTP loss weight λ from 0.3 (first 15T tokens) to 0.1 thereafter.

- Post-training: Expert iteration → unified hybrid model (Section 3)
  - Stage 1: Build expert models (Reasoning, Agent, General Chat) with Supervised Fine-Tuning (SFT) cold start followed by domain-specific RL (Sections 3.1–3.4).
  - Stage 2: Unified Training. Distill expert outputs into a single hybrid model that can either “think” (deliberative Chain-of-Thought) or answer directly when appropriate (Section 3.1 “Overall SFT”).

  - SFT specifics and data engineering (Section 3.1)
    - Function-calling template redesign: swaps JSON strings for XML-like tags to avoid excessive character escaping in code snippets (Figure 4). This reduces learning burden when arguments contain code while keeping tool-execution semantics intact.
    - Rejection sampling and filtering: remove short/duplicate/truncated samples; verify objective answers; use reward models for subjective prompts; ensure tool-call traces reach terminal states (Section 3.1).
    - Prompt selection and response scaling: drop bottom 50% prompts by response length (improves math/science by 2–4%); generate 4 responses for hard prompts for another 1–2% boost (Section 3.1).
    - Automatic agentic SFT data: collect frameworks/tools; synthesize tasks; generate tool-call trajectories with LLMs and user simulators; select via multiple judge agents (Section 3.1).

  - Reasoning RL (Section 3.2; Figures 5–7)
    - Curriculum over difficulty: two-stage schedule avoids all-1 or all-0 rewards; switch to extremely hard problems once moderate problems saturate (Figure 5).
    - Long-output RL: single-stage RL directly at 64K output length outperforms multi-stage length ramp-ups because shorter-length stages cause the model to “unlearn” long-output behavior (Figure 6).
    - Dynamic sampling temperature: increase temperature when rewards plateau, constrained by validation to avoid >1% performance drop (Section 3.2).
    - Code RL: use token-weighted mean loss (rather than sequence mean) for faster, more stable convergence and less length bias (Figure 7 left).
    - Science RL: train on small pools of expert-verified multiple-choice data for higher quality signals, outperforming mixed-quality data (Figure 7 right).
    - Base algorithm: GRPO variant without KL term (Section 3.2).

  - Agentic RL (Section 3.3; Figure 8)
    - Data pipelines for web-search agents (multi-hop Q/A requiring multiple sources) and SWE tasks (PRs/issues + unit tests; run in hardened sandbox) with verifiable outcomes.
    - Objective: group-wise policy optimization with reward as final answer correctness for search and test pass for coding; add process-format penalties for tool-call syntax violations (Section 3.3.2).
    - Iterative self-distillation: alternate shorter RL phases with SFT refreshes using improved trajectories to accelerate training (Section 3.3.2).
    - Test-time scaling: more interaction turns yields steady accuracy gains on BrowseComp (Figure 8).

  - General RL (Section 3.4; Figure 9)
    - Holistic RL: ~5,000 prompts spanning 7→33→139 categories; rewards from human preferences and model-based rubrics (RLAIF) to improve general quality while reducing reward hacking risk.
    - Instruction-Following RL: fine-grained taxonomy (7 major, 151 minor constraint types) with a hybrid feedback stack: deterministic checkers, reward model, and critique model. Reward and SysBench-ISR increase in lockstep without clear hacking within ~1,000 steps (Figure 9).
    - Function Calling RL:
      - Step-wise rule-based RL: per-step exact-match reward (=1 only if name/parameters/format all exact), which strongly enforces usable tool-call outputs.
      - End-to-end multi-turn RL: trajectory-level reward based on environment completion or LLM judge; supports complex single-turn multi-step and multi-turn multi-step tasks (explicit reward definitions in Section 3.4).
    - Pathology RL: targeted training on prompts that trigger rare but disruptive errors (language mixing, repetition, formatting).

- RL infrastructure (Section 3.5; Figure 10)
  - `Slime`: a flexible framework supporting colocated synchronous training (good for math/code RL with dynamic sampling and high GPU utilization) and disaggregated asynchronous pipelines (needed for long-horizon agentic rollouts).
  - High-throughput rollouts via FP8 mixed-precision inference (with online block-wise quantization per update) while keeping BF16 for training.
  - Agent-oriented design: high-concurrency Docker runtimes for isolation, decoupled rollout vs. training engines to avoid stragglers, and a unified HTTP endpoint plus centralized data pool for heterogeneous agent frameworks.

## 4. Key Insights and Innovations
- Hybrid expert-to-generalist training that preserves both “thinking” and “direct answer” modes
  - Instead of always forcing long Chain-of-Thought, Overall SFT explicitly balances thought-rich and thought-free data so the final model can adapt its reasoning depth to task needs (Section 3.1 “Overall SFT”). This is a pragmatic innovation for usability.

- Deep-but-narrow MoE and attention-head scaling that improve reasoning without lowering training loss
  - The model increases layers, keeps per-layer width moderate, and uses 96 heads at 5120 hidden—an unusual choice that did not reduce pretraining loss but consistently boosted reasoning benchmarks (Section 2.1). This suggests reasoning may benefit from architectural factors beyond loss minimization.

- RL recipes that target often-neglected domains and long outputs
  - Single-stage 64K-output RL avoids “unlearning” long reasoning (Figure 6); token-weighted loss accelerates code-RL convergence (Figure 7 left); using only expert-verified science data yields better GPQA performance (Figure 7 right). These are concrete, generalizable RL methodology advances (Section 3.2).

- Format-robust function calling via XML-like templates
  - Replacing JSON-within-strings arguments (which require escaping) with XML-like tags reduces training burden for code-heavy parameters without harming execution (Section 3.1; Figure 4). This is an engineering-centric but impactful usability contribution.

- Asynchronous, FP8-accelerated agentic RL infrastructure
  - The Slime framework decouples rollout/training and uses FP8 for rollout to overcome collection bottlenecks in long-horizon agent tasks (Section 3.5). This infrastructure matters for scaling future agent RL.

## 5. Experimental Analysis
- Evaluation setup and coverage
  - Benchmarks: 12 ARC tasks comprising agentic (TAU-bench Retail/Airline, BFCL v3, BrowseComp), reasoning (MMLU-Pro, AIME24, MATH-500, SciCode, GPQA, HLE, LCB 2407–2501), and coding (SWE-bench Verified, Terminal-Bench), plus general abilities (MMLU, SimpleQA, IFEval, SysBench, MultiChallenge) and SafetyBench (Sections 4.1–4.2; Tables 2–7; Figures 1–2).
  - For AIME and GPQA, they average over multiple samples to reduce variance: Avg@32 for AIME, Avg@8 for GPQA (Section 4.2.2). An open-source `glm-simple-evals` toolkit is provided for reproducibility (Section 4.2.2).

- Base model quality before post-training
  - `GLM-4.5-Base` (no instruction tuning) shows balanced strength across English, code, math, and Chinese (Table 2). Examples:
    - Code: EvalPlus Pass@1 = 78.1; LiveCodeBench-Base Pass@1 = 28.1.
    - English MMLU (EM) = 86.1; BBH = 86.2.
    - Chinese C-Eval (EM) = 86.9; CLUEWSC = 83.5.

- Agentic benchmarks (Table 3; Figure 1)
  - TAU-bench (Retail/Airline):
    - Retail: 79.7 (GLM-4.5), close to Claude Sonnet 4 (80.5) and Claude Opus 4 (81.4), higher than o4-mini (65.6) and GPT-4.1 (75.1).
    - Airline: 60.4, on par with Claude Sonnet 4 (60.0) and Opus 4 (59.6), higher than GPT-4.1 (48.8).
  - BFCL v3 (function calling): 77.8 (best among listed baselines).
  - BrowseComp (web browsing): 26.4—behind o3 (49.7) and o4-mini-high (28.3) but ahead of Claude Opus 4 (18.8) and Gemini 2.5 Pro (7.6).
  - Average agentic score in Table 3: 58.1 for GLM-4.5, ranking second overall in Figure 1.

- Reasoning benchmarks (Table 4)
  - AIME24 Avg@32 = 91.0; MMLU-Pro = 84.6; MATH-500 = 98.2; SciCode = 41.7; GPQA = 79.1; HLE = 14.4; LCB = 72.9.
  - Compared to strong baselines: exceeds Claude Opus 4 on average reasoning (AA-Index est. 67.7 vs. 64.4) and is close to DeepSeek-R1-0528 (68.3). GLM-4.5 beats o3 on AIME24 (91.0 vs. 90.3) and is close on SciCode (41.7 vs. 41.0).

- Coding benchmarks (Table 5; Figure 2)
  - SWE-bench Verified: 64.2—above GPT-4.1 (48.6) and Gemini 2.5 Pro (49.0), and close to Claude Sonnet 4 (70.4) and Kimi K2 (65.4).
  - Terminal-Bench: 37.5—better than o3 (30.2) and GPT-4.1 (30.3) but below Claude Opus 4 (43.2).
  - Overall coding average in Table 5: 50.9 for GLM-4.5, making it one of the top open options. Figure 2 shows GLM-4.5 and GLM-4.5-Air on the Pareto frontier (score vs. parameter count).

- General abilities (Table 6)
  - MMLU: 90.0 (comparable to GPT-4.1 at 90.2).
  - IFEval: 86.1; SysBench-ISR: 81.0 (better than GPT-4.1 at 80.6 and DeepSeek V3 at 79.8).
  - SimpleQA (short factual): 26.4, notably lower than Gemini 2.5 Pro (54.0). This highlights a trade-off with rote factual recall.

- Safety (Table 7)
  - SafetyBench average: 89.9—comparable to GPT-4.1 (89.7) and Kimi-K2 (90.5).
  - Strong in Physical Health (96.7) and Mental Health (94.7). Lower on Unfairness & Bias (77.4), identified as an area for improvement.

- Human and scenario-based evaluations (Section 4.3)
  - General chat human evals:
    - English prompts (Table 8): GLM-4.5 overall 8.66, narrowly above DeepSeek-R1-0528 (8.62). Best on Logic (9.25) and Math (8.72).
    - Chinese prompts (Table 9): GLM-4.5 overall 8.37, leading across Text Generation (9.00) and Logic (9.27).
    - Other languages (Table 10): GLM-4.5 overall 8.49, ahead of DeepSeek-R1-0528 (8.27).
  - Coding Agent (CC-Bench; Figures 12–13):
    - Head-to-head task outcomes: vs. Kimi K2 (53.9% win, 17.3% tie), vs. Qwen3-Coder (80.8% win), vs. Claude Sonnet 4 (40.4% win, 50.0% loss). 
    - Tool-calling reliability: highest success rate 90.6% vs. 89.5% (Claude Sonnet 4), 86.2% (Kimi K2), 77.1% (Qwen3-Coder). Token usage moderate (Figure 13 right).
  - Novel logical reasoning set (Table 11): GLM-4.5 at 62.0, close to DeepSeek-R1-0528 (62.1), below Gemini 2.5 Pro (65.8).
  - Translation (Table 12): GLM-4.5 average score 1.71 on 0–3 human scale, substantially above specialized translation models on challenging internet-culture cases.

- Do the experiments support the claims?
  - Breadth: Yes. The model is tested across agentic, reasoning, coding, general instruction following, safety, and human evaluations with detailed setups (Sections 4.1–4.4).
  - Depth: Several ablations demonstrate why specific RL and data choices matter (Figures 5–7, 9).
  - Caveats:
    - BrowseComp trails far behind o3 (Table 3).
    - HLE (14.4) is low across the board and highlights limitations on open-ended, adversarial “last exam” style questions (Table 4).
    - SimpleQA lag indicates weaker short-form factual recall (Table 6).

## 6. Limitations and Trade-offs
- Data and supervision assumptions
  - Heavy reliance on synthetic reasoning traces and agentic trajectories (Sections 2.3, 3.1, 3.3) means performance may be sensitive to synthesis biases and coverage.
  - Some human evaluations used single evaluators for consistency, which reduces inter-rater variance but introduces single-rater bias risk (Section 4.3.1).

- Compute and engineering complexity
  - Deep MoE plus multi-stage RL demands substantial infrastructure. Although only 32B parameters are active at inference, training involves 23T+ tokens, long contexts to 128K, and complex asynchronous rollouts (Figure 3; Section 3.5).
  - Test-time performance on agent tasks scales with number of interaction turns (Figure 8), implying higher latency/cost for best results.

- Uneven capability profile
  - Strong on ARC but not universally superior: behind o3 on BrowseComp; below Claude Opus 4 on Terminal-Bench; SimpleQA much lower than Gemini 2.5 Pro (Tables 3, 5, 6).
  - Safety “Unfairness & Bias” lags other categories (Table 7).

- Generalization and robustness
  - Despite robust filtering and verifiable rewards, agentic RL depends on specific environments and judging setups; transfer to different tool ecosystems may require additional distillation or RL (Section 3.3.2; 3.5).

## 7. Implications and Future Directions
- Field impact
  - Demonstrates that a unified, open-source model can be competitive across agentic, reasoning, and coding with a carefully engineered MoE design and targeted RL. The hybrid thinking/direct modes and the function-calling template address usability—important for real applications (Sections 3.1–3.4).
  - Provides a repeatable RL stack (curriculum, 64K single-stage RL, token-weighted code RL, expert-verified science RL) and an agentic RL infrastructure (Slime with FP8 rollouts) that others can adopt (Sections 3.2–3.5).

- Research avenues
  - BrowseComp gap: richer browsing policies, better planning/retrieval under noisy web, and improved process supervision for search sequences (Table 3; Figure 8).
  - Long-context reasoning: extend the single-stage RL idea beyond 64K; study interactions with retrieval and memory systems (Figure 6).
  - Bias and fairness: targeted safety RL similar to “Pathology RL,” but for fairness-sensitive prompts (Table 7).
  - Data efficiency: further study of curriculum + verified-small-pool strategies in science/math/code to reduce RL token budgets (Figure 7).
  - Better knowledge recall without sacrificing reasoning: reconcile SimpleQA underperformance with reasoning strengths via balanced SFT/RL mixtures or dual-memory mechanisms (Table 6).

- Applications
  - Production agents for software engineering (SWE-bench/CC-Bench results), enterprise automation with robust function calling (BFCL v3), and long-context analytical assistants (repo-level code, 128K contexts).
  - Multilingual assistants that handle internet culture and domain-specific idioms (Table 12), and instruction-following systems with strong schema compliance (SysBench; Section 3.4).

> Overall, the evidence across Figures 1–2 and Tables 3–5 shows `GLM-4.5` ranks 3rd overall among evaluated models and 2nd on agentic tasks with markedly fewer total parameters than some competitors, while delivering practical engineering advances (function calling format, FP8 rollout infra) that will matter in real deployments.
