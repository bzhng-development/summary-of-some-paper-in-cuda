# Hermes 4 Technical Report

**ArXiv:** [2508.18255](https://arxiv.org/abs/2508.18255)

## 🎯 Pitch

Hermes 4 introduces a new family of open-weight hybrid reasoning models that combine advanced, structured multi-turn reasoning with broad instruction-following capability. By innovating in scalable reasoning-focused data generation (DataForge and Atropos), precise reasoning-length control, and transparent, unified benchmarking across key domains, Hermes 4 offers models that achieve top-tier performance while remaining fully open and reproducible. This matters because it empowers the research community to build, evaluate, and deploy generalist models that not only reason deeply but also reliably follow diverse instructions—bridging key gaps between proprietary reasoners and general open-source LLMs.

---

## 1. Executive Summary
Hermes 4 is a family of open‑weight “reasoner” models that blend long, structured thinking with broad instruction following. The report contributes three things at once: a scalable way to synthesize and verify reasoning‑heavy training data, a training recipe that controls how long models think without hurting quality, and a transparent, reproducible evaluation stack that covers math, code, knowledge, alignment, and creative writing.

## 2. Context and Motivation
- Problem gap
  - Many powerful models can follow instructions but stumble on problems that require multi‑step reasoning or tool‑use; specialized reasoners can think deeply but are often closed, hard to reproduce, or over‑think until they run out of context. Hermes 4 targets a “hybrid” capability: sustained, structured reasoning plus general versatility, with open weights and rigorous, reproducible evaluation (Sections 1, 2, 4).

- Why it matters
  - Real‑world tasks mix skills: solving a math proof, fixing code with strict schemas, obeying output formats, or judging other models. Hermes 4 seeks to make these behaviors controllable and reliable at scale, while remaining open for research and deployment.

- Prior approaches and their limitations
  - Proprietary reasoners existed first; recent open releases improved access but left gaps in data generation pipelines, thinking‑length control, and unified evaluation (Section 1). Existing data often lacks verified trajectories; models also tend to over‑think, hitting context limits (Figure 3a; Section 3.1).

- Positioning
  - Hermes 4 offers:
    - A graph‑based synthetic data generator (`DataForge`) to produce diverse, high‑quality instruction/reasoning data with judges and verifiers (Section 2.1).
    - Rejection‑sampled trajectories from many verifiable environments via `Atropos`, an open RL/eval controller (Section 2.2).
    - A two‑stage training method including targeted “thinking‑length” control by supervising only a single stop token `</think>` (Sections 3 and 3.1).
    - A standardized, OpenAI‑compatible evaluation stack with shared inference for all benchmarks to improve reproducibility (Section 4.1).

## 3. Technical Approach
Hermes 4 has three pillars: data, training, and evaluation.

- Data: hybrid, large‑scale, verified
  - Scope
    - ~5M samples, ~19B tokens (Section 2). About 3.5M are reasoning‑focused and 1.6M are non‑reasoning. Reasoning samples are token‑heavy (average ~5× longer) and include thinking traces up to 16k tokens, later extended for length‑control training (Section 2).
  - `DataForge` (Section 2.1)
    - What it is: a graph‑based synthetic data generator. Each node is an operation with declared preconditions/postconditions following a PDDL‑style interface; edges are implied when postconditions of one node satisfy the preconditions of another. A random walk through this DAG produces one datapoint.
    - How it works (Figure 1a):
      1) Start from a pre‑training “seed” passage (cleaned and semantically deduplicated; Section 2.1.1).
      2) Transform it into a target artifact (e.g., turn a news article into a debate transcript or rap).
      3) Generate an instruction either contextual (the transformed passage is included) or standalone (inspired by it without referencing it).
      4) Use a specialized answer generator for that instruction type.
      5) Use an instruction‑specific LLM judge with a rubric (style, coherence, relevance, etc.) to accept/iterate/reject.
      6) Train not only on the final QA pair but also on all intermediate LLM calls—this gives the model practice in instruction generation and judging itself (Section 2.1.2).
    - “Higher‑order graphs”: because every graph has a single source/target, a finished graph itself implements the node interface and can be nested inside larger graphs (Figure 1b; Section 2.1.3). This supports scalable composition of complex pipelines.
  - Rejection‑sampled verified trajectories via `Atropos` (Section 2.2)
    - Definition (rejection sampling): generate many candidate solutions and keep only those that pass a programmatic verifier or strict rubric.
    - Environments (Sections 2.2.1–2.2.5):
      - `Answer Format Training`: enforces correct final‑answer formatting and the use of `<think>…</think>` delimiters, rewarding format validity only.
      - `Instruction Following`: uses verifiable constraints (from RLVR‑IFEval) like “every Nth word is in French,” sampling only successful trajectories.
      - `Internbootcamp`: ~70k verified, multi‑domain reasoning trajectories from ~1,000 tasks; multiple correct solution paths kept (Section 2.2.3).
      - `Schema Adherence`: JSON generation and editing against dynamic Pydantic schemas; binary reward if the object validates (Section 2.2.4).
      - `Tool Use`: trains the model to produce structured `<tool_call>` JSON matching ground truth (Section 2.2.5).
    - Multiple unique trajectories leading to the same verified result are kept (OpenThoughts recipe; Section 2.2).
  - Coverage strategies (Section 2.3)
    - `Taxonomies`: recursively partition a domain into subdomains until reaching leaf prompts (Section 2.3.1). Example: enumerating parseable output formats.
    - `PersonaHub`: synthesize realistic user tasks from personas (Appendix A), then produce reasoning traces with strong teachers (e.g., `DeepSeek‑R1‑0528`; Section 2.3.2).

- Training (Section 3)
  - Base checkpoints: `Llama 3.1` (405B, 70B) and `Qwen3 14B` for the 14B version (Section 3).
  - Infrastructure: Modified TorchTitan; 192 NVIDIA B200 GPUs with a mix of Distributed Data Parallel, Tensor Parallel, and Fully Sharded Data Parallel (Section 3).
  - Curriculum and efficiency
    - Context length: 16,384 tokens for SFT; packing heterogeneous sample lengths using First‑Fit Decreasing achieves >99.9% batch efficiency (Figure 3a; Section 3).
    - Attention isolation: `Flex Attention` restricts attention within each packed sample (Section 3).
    - Loss masking: only tokens by the assistant role contribute to cross‑entropy loss (Figure 2; Section 3).
    - Learning schedule: cosine LR with 300 warmup steps, 9,000 total steps, global batch size 384 (Section 3).
    - Training parameters (Table 1): each model size trained on ~56B tokens with size‑specific learning rates and B200‑GPU hours.
  - Thinking‑length control (Section 3.1)
    - Problem: The 14B model often exceeded a 40,960 token context at inference—e.g., reached max context 60% of the time on LiveCodeBench when “reasoning” (Section 3.1; Figure 3a).
    - Method: Second SFT stage teaches the model to end its chain‑of‑thought at a fixed budget by supervising only a single token: the closing `</think>` at 30k tokens (Figure 3b).
      - Data collection: sample prompts (mostly STEM/coding) and generate long reasoning traces; if a trace stops after `</think>`, allow finishing the answer; if it stops before, force `\n</think>` and generate the answer (Section 3.1.1).
      - Training trick: mask out all tokens except `</think>` (and the training framework’s necessarily unmasked `<eos>`), so gradients focus entirely on “when to stop,” not on the reasoning content itself (Section 3.1; 3.1.2).
      - Rationale: single‑step supervision avoids synthetic‑data collapse that can happen when training on full self‑generated reasoning (Section 3.1).
    - Ablation (Appendix B; Table 5): at a 20k budget, naive SFT on truncated traces increased overlong rates; supervising only `</think>` slashed overlong to ≤0.6% but initially harmed some scores—hence the final choice of a more permissive 30k budget (Section 3.1.3; Appendix B).

- Evaluation (Section 4)
  - A single, shared OpenAI‑compatible endpoint for all benchmarks minimizes confounds from different inference engines or backends (Section 4.1).
  - `lighteval` for most math/multiple‑choice tasks, plus custom integrations for MMLU, OpenBookQA, SimpleQA, and DROP (Section 4.2).
  - `Atropos` as an evaluation framework (Section 4.3):
    - Single‑file evaluations, detailed sample‑level logging (useful when parsers/LLM judges disagree), overlap of inference and scoring (critical for expensive verifiers like code tests), explicit error semantics, and CLI/YAML configs generated from dataclasses.
    - Ports of Arena‑Hard v1 and RewardBench into Atropos (Section 4.3.2).
    - LiveCodeBench (Section 4.3.3): sandboxed verification in Modal containers; inference and verification overlapped so the run remains inference‑compute‑bound.
  - Elastic inference cluster: sglang‑router manages preemptible workers that dynamically attach/detach; each replica sharded at TP8; Triton attention backend on B200 (Sections 4.4–4.5).
  - Evaluation conditions: long context (up to 163,840 for DeepSeek baselines; Hermes evaluated at 40,960 for reasoning/code and 32,768 otherwise), shared sampling settings unless provider‑recommended changes apply; multiple samples per problem for pass@1 estimates (Section 4.5).

Definitions of uncommon terms used above:
- `reasoner model`: a model that performs extended internal “thinking” (multi‑step reasoning traces) and adapts its compute at inference time.
- `rejection sampling`: keep only generated solutions that pass a verifier/judge.
- `pass@1`: the fraction of problems solved correctly by the single best (or first) sample.
- `RefusalBench`: an internal benchmark that measures how often a model refuses certain categories of requests, with three categories scored inversely to prefer refusals (Section 4.5.1).
- `Arena‑Hard v1`: a “vibe‑check” benchmark graded by an LLM judge using pairwise comparisons (Section 4.3.2).

## 4. Key Insights and Innovations
1) Graph‑based synthetic data that is composable and verifiable (Sections 2.1–2.1.3; Figure 1)
- What’s new: `DataForge` builds instruction‑answer data via declarative nodes with pre/postconditions, supports “graphs of graphs,” and trains on all intermediate LLM calls—not only final QA pairs.
- Why it matters: enables large, diverse, high‑quality instruction and reasoning datasets with built‑in quality control (judges and verifiers), and exposes models to the structure of task creation and judging, not just answering.

2) Verified trajectories at scale via many task‑specific environments (Section 2.2)
- What’s new: ~1,000 verifiers across environments covering format compliance, constraints, schema validation, tool calls, and multi‑domain reasoning (e.g., Internbootcamp).
- Why it matters: creates a large pool of grounded, correctness‑checked reasoning traces—critical for reliable reasoner behavior.

3) Single‑token “thinking‑length” supervision (Section 3.1; Figure 3b; Table 2; Appendix B, Table 5)
- What’s new: rather than fine‑tuning on the model’s own long reasoning (which can cause collapse), supervise only the `</think>` emission at a fixed budget. This teaches “when to stop” without altering the rest of the reasoning distribution.
- Why it matters: reduces sequences that run past context by ~99% with minimal accuracy loss; e.g., for the 14B model, LiveCodeBench overlong rate drops from 60.0% to 0.1% and pass@1 improves from 28.6 to 42.5 (+48.6% relative) after 30k‑budget tuning (Table 2).

4) Reproducible, efficient evaluation stack (Sections 4.1–4.4)
- What’s new: one inference endpoint for everything, detailed sample‑level logs, overlapped inference/verification for code benchmarks, and elastic, preemptible worker pools.
- Why it matters: more credible cross‑benchmark comparisons and scalable, cheaper code verification runs (essential for LiveCodeBench’s many test cases).

## 5. Experimental Analysis
- Evaluation design (Sections 4.2, 4.5)
  - Benchmarks include math/reasoning (MATH‑500, AIME’24/’25, GPQA Diamond, MuSR), code (LiveCodeBench v6 Aug‑2024+), knowledge (MMLU, MMLU‑Pro, OBQA, SimpleQA), instruction‑following (IFEval), judge‑ability (RewardBench), refusal tendencies (RefusalBench), and creative writing (EQBench3, CreativeWriting3).
  - Sampling: typically temperature 0.6, top‑p 0.95, top‑k 20; long context for reasoning/code; multiple samples per task per lighteval defaults (e.g., AIME estimated with 64, LiveCodeBench with 16).

- Main quantitative results
  - Hermes 4 405B vs strong open‑weight baselines (Table 3):
    - Math/reasoning:
      > “MATH‑500: 96.2; AIME’24: 81.9; AIME’25: 78.1; GPQA Diamond: 70.6”
      These are close to frontier reasoners (e.g., DeepSeek‑R1‑0528 shows 97.5/86.5/83.1/78.1 respectively) while outperforming several others on some tasks.
    - Code (LiveCodeBench v6, Aug2024+ subset):
      > “61.4” vs DeepSeek‑V3’s 49.2 and Qwen3‑235B’s 65.1 (Table 3).
    - Alignment/formatting/QA:
      > “Arena‑Hard v1: 93.7; RewardBench: 73.0; IFEval (Loose): 81.5”
      High Arena‑Hard suggests good instruction‑following “vibe” under LLM‑judge settings.
    - Refusals:
      > “RefusalBench: 57.1 (reasoning mode), 43.2 (non‑reasoning mode)” (Figure 4; Table 3).
      This indicates Hermes 4 responds more often (fewer refusals) than many peers, except on three safety‑critical categories where scores are inverted to prefer refusals (Section 4.5.1).
    - Creativity:
      > “EQBench3: 85.5; CreativeWriting3: 79.3”—strong creative performance for an open model (Table 3).

  - Hermes 4 70B and 14B (Table 4):
    - 70B:
      > “AIME’24: 73.5; AIME’25: 67.5; MATH‑500: 95.5; LiveCodeBench: 50.5; Arena‑Hard: 90.1; EQBench3: 84.7.”
    - 14B:
      > “AIME’24: 55.4; AIME’25: 46.8; LiveCodeBench: 42.5 after length‑tuning; EQBench3: 77.2.”
      The 14B model particularly benefits from thinking‑length control (Table 2).

- Length‑control ablations (Section 3.1.3; Appendix B, Table 5)
  - At a stricter 20k budget (early experiments), “`</think>`‑only” masking reduced overlong rates to ≤0.6% but hurt AIME’24 by ~20 points (Table 5b). Moving to a 30k budget regained most accuracy while preserving the huge reduction in overlong sequences (Table 2).
  - The “Control” run (no truncated‑trace data, same framework/settings) unexpectedly boosted LiveCodeBench by 18%—likely due to fixes and longer‑context exposure in Stage 2 (Appendix B discussion).

- Do the experiments support the claims?
  - Yes, for three central claims:
    1) Hybrid capability: Strong results across math, code, alignment, and creative writing (Tables 3–4), with qualitative probes showing persona control and reduced sycophancy under prompt engineering (Section 5; Appendix C).
    2) Thinking‑length control: Overlong rates drop by ~99% with minimal or acceptable accuracy trade‑offs at a 30k budget (Table 2).
    3) Reproducibility/engineering: A single inference stack, public logs, and detailed methodology for each benchmark (Sections 4.1–4.5) increase transparency.

- Notable caveats
  - Hermes 4 405B trails DeepSeek‑R1‑0528 on some top‑end reasoning tasks (e.g., AIME’25: 78.1 vs 83.1; GPQA: 70.6 vs 78.1; Table 3).
  - MMLU/MMLU‑Pro are competitive but not state‑of‑the‑art (e.g., 405B: 87.2/80.6 vs DeepSeek‑R1‑0528: 90.4/84.3; Table 3).

## 6. Limitations and Trade-offs
- Synthetic‑data dependencies
  - Heavy reliance on synthetic generation with LLM judges risks preference for distributions seen during synthesis and potential judge biases. The report mitigates this by using different weights for answer and judge models (Section 2.1.2) and by adding verified, programmatic environments, but residual bias is still possible.

- Length‑control trade‑offs
  - Tight budgets can hurt accuracy (Appendix B, Table 5). The final 30k setting balances practicality and performance but remains a compromise (Table 2).

- Coverage choices
  - Domain coverage sometimes relies on “vibe inspection” between taxonomy and persona‑driven generation (Section 2.3), which is principled but not entirely objective.

- Compute and complexity
  - Training requires substantial B200 GPU hours (Table 1), and the evaluation stack (Modal for code, elastic clusters, TP8 sharding) is sophisticated. Replication is feasible but not lightweight (Sections 3–4.4).

- Evaluation dependencies
  - Some benchmarks use LLM judges (e.g., Arena‑Hard, RefusalBench), which can disagree with parsers or humans. The report counters this by logging sample‑level details and noting a 7.3% parser/judge disagreement found elsewhere (Section 4.3.1), but subjectivity remains.

- Safety/Refusal behavior
  - RefusalBench optimizes for fewer refusals except in three inverted categories (self‑harm, exploitation/trafficking, minor harm). A higher aggregate score (Figure 4) means the model is more willing to answer; depending on deployment, this can be a feature or a risk and may require additional policy tuning (Section 4.5.1).

## 7. Implications and Future Directions
- Field impact
  - Data: `DataForge` plus verified Atropos environments show a scalable pathway to create reasoning‑heavy, quality‑checked corpora. The graph‑of‑graphs abstraction could generalize to tool‑use curricula, multi‑agent simulations, and task‑planning datasets.
  - Training: Supervising only `</think>` for termination is a simple, powerful control mechanism. This invites new research on “thinking budget schedulers” (dynamic N per task) and multi‑objective control (e.g., length vs. accuracy vs. latency).
  - Evaluation: Treating RL environments as evaluations (and vice versa) with single‑stack, reproducible inference reduces benchmark variance and helps the community compare models fairly.

- Follow‑up research
  - Adaptive budget policies that predict per‑task “how much to think,” possibly with a small controller network.
  - Multi‑signal termination (e.g., supervise both “stop thinking” and “emit tool call”), or curriculum schedules that reward shorter, sufficient chains (efficiency‑oriented reasoning; Section 3.1 references related work [33, 34]).
  - More robust judge diversity and cross‑checking (parser vs LLM vs human) to minimize judge‑bias in both data synthesis and evaluation (Section 2.1.2; 4.3.1).
  - Causal analysis of “reasoning prefixes” that induce runaway thinking (Appendix B discussion) and techniques to detect/avoid them.

- Practical applications
  - Coding assistants that must pass strict test suites and respect schemas (Schema Adherence; Tool Use; LiveCodeBench).
  - Data wrangling and agentic workflows requiring precise output formats (`Answer Format Training`) and multi‑step tool invocation.
  - Judging and curation (RewardBench performance suggests Hermes 4 can serve as a strong LLM‑as‑a‑judge component).
  - Creative and stylistic control at scale (high EQBench/CreativeWriting results and qualitative probes in Section 5; Appendix C).

> Overall: Figures 1–3, Tables 1–4 (plus Appendix B, Table 5) demonstrate that Hermes 4 achieves a practical balance—long, structured reasoning when needed, the ability to stop on time, and competitive performance across a broad evaluation suite—while keeping the entire process open and reproducible.
