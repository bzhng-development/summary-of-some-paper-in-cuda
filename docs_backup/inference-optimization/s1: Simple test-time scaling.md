# s1: Simple test-time scaling

**ArXiv:** [2501.19393](https://arxiv.org/abs/2501.19393)
**Authors:** Niklas Muennighoff, Zitong Yang, Weijia Shi, Xiang Lisa Li, Li Fei‑Fei, Hannaneh Hajishirzi, Luke Zettlemoyer, Percy Liang, Emmanuel Candès, Tatsunori Hashimoto
**Institutions:** Stanford University, University of Washington, Allen Institute for AI, Contextual AI

## 🎯 Pitch

This paper revolutionizes inference-time reasoning in language models by introducing 'budget forcing,' a method that enables scalable, controllable accuracy using a small curated dataset. This advancement not only empowers users to optimize accuracy versus computation cost but also provides a transparent framework that demystifies test-time scaling, setting a new benchmark for reproducibility and efficiency in language model research.

---

## 1. Executive Summary
This paper introduces a very simple recipe for making a language model reason better by spending more computation at test time (“test-time scaling”). The recipe has two parts: (1) a tiny, carefully selected dataset of 1,000 problems with reasoning traces (`s1K`), and (2) a decoding-time control method called `budget forcing` that reliably lengthens or shortens how long the model “thinks.” Finetuning Qwen2.5‑32B on `s1K` and applying budget forcing yields `s1‑32B`, which shows clear, controllable accuracy gains as more thinking tokens are allowed (Figure 1; §3–4), and reaches strong sample‑efficiency and competitiveness with closed models (Table 1; Figure 2).

## 2. Context and Motivation
- Problem gap
  - “Test-time scaling” means using more compute at inference to get better answers. While OpenAI’s o1 family demonstrated this effect, details were not public, prompting numerous replication attempts that often relied on reinforcement learning (RL) and large private datasets (§1; related work §6).
  - A missing piece has been a minimal, openly described method that both: (a) reliably exhibits test-time scaling curves and (b) reaches competitive reasoning performance without vast training data.

- Why it matters
  - Practically: gives users a knob to trade latency for accuracy. For example, the model can spend more tokens when a question is hard and fewer when it is easy (Figure 1).
  - Scientifically: provides clean, controllable baselines and metrics to study inference-time scaling (§3.2), decoupled from opaque RL pipelines.

- Prior approaches and limits
  - RL-based systems (e.g., DeepSeek‑R1) achieve high reasoning scores but use “millions of samples and multiple training stages” (§1, §6.1).
  - Multi-agent and tree-search approaches (e.g., MCTS, REBASE) can be powerful but add complexity and extra models (reward models) (§6.2).
  - Many replication attempts did not openly reproduce a clear, monotonic scaling curve with test-time compute (§1).

- Positioning of this work
  - The paper pursues the simplest feasible route: supervised finetuning (SFT) on just 1,000 high-quality reasoning traces, plus an inference-time control that needs no extra training or models. It then defines metrics to evaluate test-time scaling methods (§3.2), and shows the method’s controllability and gains across math and science benchmarks (§4–5).

## 3. Technical Approach
The pipeline has three pillars: data curation (to build `s1K`), a lightweight SFT, and an inference-time controller (`budget forcing`).

1) Reasoning data curation → `s1K` (§2)
- Start with 59,029 questions across 16 sources focused on quality, difficulty, and diversity (Table 7; §2.1). Examples include NuminaMATH, MATH, OmniMath, OlympicArena, AGIEval, and two new sets: Stanford PhD probability (`s1-prob`) and hard trading brain‑teasers (`s1-teasers`).
- For each question, obtain a reasoning trace and final answer by calling Gemini 2.0 Flash Thinking and extracting its hidden chain of thought and solution (§2.1).
- Clean-up and filtering:
  - Remove API/formatting issues (down to 51,581), deduplicate, and decontaminate against evaluation sets using 8‑gram overlaps (§2.1, §C.5).
  - Difficulty filter: discard questions that Qwen2.5‑7B or Qwen2.5‑32B can already solve (to keep challenging items) and use generated trace length as a proxy for difficulty (§2.2).
  - Diversity filter: label questions with Mathematics Subject Classification‑style domains using Claude 3.5, then sample problems across 50 domains with a length‑weighted sampler to favor longer traces (§2.2; Algorithm 1 in §C.4).
- Outcome: `s1K` has 1,000 diverse, hard questions with reasoning traces (Figure 2 left; Table 6). Notably, traces need not be always correct; the grader reports 53.6% correctness for `s1K` and 63.0% for a later `s1K‑1.1` update (§2.2; §A).

2) Supervised finetuning (`s1‑32B`) (§4.1; §D)
- Base model: `Qwen2.5‑32B‑Instruct`.
- Training data: `s1K` (1,000 triples: question, reasoning trace, answer).
- Formatting: the training target is the reasoning trace then the answer, separated by special delimiters `<|im_start|>think` and `<|im_start|>answer` (§D).
- Hyperparameters: 5 epochs, batch size 16, bfloat16, AdamW, lr 1e‑5 with 5% warm‑up and cosine decay; 26 minutes on 16 H100 GPUs (§4.1; Figure 9; §D).
- Sequence length ablation (Table 8; §D.1): using a long training context (32k) reduces test-time “thinking” length and improves accuracy versus a short context (4096).
  - Example on AIME24: 50.0% accuracy with 6984 thinking tokens vs 30.0% with 20721 tokens for the short‑context model (Table 8).

3) Test-time compute control → `budget forcing` (§3.1)
- Goal: deterministically set a maximum and/or minimum “thinking” budget without changing the model weights.
- Thinking/answer phases: because training teaches the model to “think first, then answer” using explicit delimiters, decoding can intercept the transition.
- Two control levers:
  - Enforce a maximum: when the running count of thinking tokens reaches a cap, force the transition to the answer by appending the end‑of‑thinking delimiter (and optionally “Final Answer:”) (§3.1).
  - Enforce a minimum/extend thinking: when the model tries to stop thinking, suppress the delimiter and “nudge” the chain to continue by appending a short string such as “Wait” (§3.1). This often triggers self‑checking and fixes earlier steps (Figure 3).
- Why this over alternatives? The paper compares budget forcing to (i) token/step/class‑conditional prompting and (ii) rejection sampling, and finds budget forcing provides perfect control and the best accuracy‑vs‑compute scaling (§5.2; Table 3; Figure 6).

4) How scaling is measured (§3.2)
- A method is evaluated at several compute points (different thinking budgets), producing a piecewise‑linear curve of accuracy vs tokens (see Figure 1 and Figure 4).
- Three metrics:
  - `Control` (Eq. 1): fraction of runs that meet the prescribed compute budget (100% is perfect).
  - `Scaling` (Eq. 2): average slope across all budget pairs; positive means accuracy rises as compute increases.
  - `Performance` (Eq. 3): best accuracy achieved over the tested budgets.

5) Parallel scaling for comparison (§4.2; Figure 4 right; Figure 7)
- Majority voting: run the base model many times and pick the most frequent answer.
- REBASE: a tree‑search guided by a separate process reward model; used here as a strong parallel‑scaling reference (Figure 7).

## 4. Key Insights and Innovations
1) Budget forcing is a minimal, effective, and controllable test-time scaler (§3, §5.2)
- What’s new: it exploits the trained “think→answer” delimiter to intervene at decode time. No extra models or RL.
- Why it matters:
  - Perfect compute control (100% `Control`, Table 3) and positive scaling (slope 15) with the best AIME24 `Performance` among the tested methods (56.7%).
  - Works both to cap compute and to extend it in small increments; the tiny “Wait” token often induces useful reflection (Figure 3; Table 4).

2) Only 1,000 carefully chosen examples can unlock reasoning and scaling (§2, §4)
- What’s new: instead of massive distillation/RL corpora, the paper shows that curating for `Quality + Difficulty + Diversity` is sufficient.
- Why it matters: `s1‑32B` substantially outperforms the base model with just 1k samples (Table 1), and sits on the sample‑efficiency frontier (Figure 2 right).

3) A clear evaluation framework for test-time scaling (§3.2, §5.2)
- What’s new: explicit definitions of `Control`, `Scaling`, and `Performance`, applied to multiple test-time methods (Table 3).
- Why it matters: distinguishes methods that increase tokens but not accuracy (e.g., rejection sampling shows negative slope, Figure 6) from methods that truly scale.

4) Sequential scaling can beat naive parallel scaling at comparable budgets (§4.2; Figure 4)
- Insight: After SFT on `s1K`, sequentially extending a single, coherent reasoning trace (with budget forcing) yields better curves than many independent samples plus majority vote from the base model (Figure 4 right). It supports the hypothesis that “later computations can build on intermediate results” (§3.1).

Incremental but useful: small prompting tweaks for extrapolation
- Observation: different extrapolation strings matter—“Wait” is best among tried variants (Table 4).

## 5. Experimental Analysis
- Evaluation setup (§4.1):
  - Benchmarks:
    - `AIME24` (30 competition math problems; integer answers) with figure inputs provided via Asymptote (§4.1).
    - `MATH500` (500 competition math problems; OpenAI’s subset) (§4.1).
    - `GPQA Diamond` (198 PhD-level science Qs; experts 69.7% per OpenAI) (§4.1).
  - Metric: accuracy (pass@1); default decoding temperature 0 unless noted (§4.1).
  - Infrastructure: lm‑evaluation‑harness; vLLM; notes on determinism issues and mitigations in Appendix B.

- Main quantitative results (Table 1; Figure 1; Figure 4)
  - `s1‑32B` vs base `Qwen2.5‑32B‑Instruct`:
    - AIME24: 56.7% vs 26.7% (+30.0 points).
    - MATH500: 93.0% vs 84.0% (+9.0 points).
    - GPQA Diamond: 59.6% vs 49.0% (+10.6 points).
  - Against `o1‑preview`:
    - AIME24: 56.7% vs 44.6%.
    - MATH500: 93.0% vs 85.5%.
    - GPQA: 59.6% vs 73.3% (here `o1‑preview` is stronger).
  - Test‑time scaling curve: On AIME24, extending thinking multiple times by suppressing the stop and appending “Wait” increases accuracy from ~50% (no extrapolation) to ~57% at higher budgets (Figure 1 middle; Figure 4 left). The curve “eventually flattens out,” and too many suppressions can cause loops (§4.2).
  - Parallel scaling comparisons:
    - Majority voting on the base model (up to 64 generations) fails to catch `s1‑32B` sequential scaling (Figure 4 right).
    - Adding REBASE on top of `s1‑32B` can scale further at very large budgets but requires an extra reward model pass per step (Figure 7).

- Test-time method ablations (Table 3; §5.2)
  - Budget forcing: `Control 100%`, `Scaling 15`, `Performance 56.7`.
  - Token‑conditional control: poor control (40%), negative slope (−24); adding budget forcing improves control to 100% but not performance (40.0%).
  - Step‑conditional control: medium control (60%); still weak scaling and performance (≤36.7%).
  - Class‑conditional (“short” vs “long” prompts): some scaling (slope 25) but low control (50%) and low performance (36.7%).
  - Rejection sampling: perfect control (by construction) but inverse scaling (−35), i.e., longer traces sampled this way tended to be worse (Figure 6 and §E.2 case study).

- Data ablations (Table 2; §5.1)
  - Random 1k (“Only Quality”): much worse than `s1K` on AIME24 (36.7% vs 50.0%).
  - “Only Diversity” (uniform over domains): 26.7% on AIME24.
  - “Only Difficulty” (longest 1k traces): strong on GPQA (59.6%) but still below `s1K` overall.
  - Full 59k training: 53.3% AIME24, 92.8% MATH500, 58.1% GPQA—close to `s1K` but requires ~394 H100 GPU hours vs ~7 for `s1K` (§5.1).

- Training ablation: sequence length (Table 8; §D.1)
  - Longer training context yields better accuracy and shorter thinking at inference.
  - The paper explains the mechanism: with longer sequences, the model more often sees complete examples where the answer follows the chain, which raises the likelihood of transitioning to the answer earlier (§D.1).

- Illustrative generations and self‑correction
  - Figure 3 shows a simple example where appending “Wait” after an early stop pushes the model to “re‑read” and fix a counting mistake.
  - Figure 5 shows correct end‑to‑end outputs on one item each from AIME24, MATH500, and GPQA.

- Update s1.1 (Appendix A, Table 5)
  - Re‑distilling the same 1,000 prompts with DeepSeek‑R1 traces improves performance (e.g., `s1.1 w/o BF` AIME24 56.7%; AIME2025 50.0%). Table 5 gives a fuller matrix, including OpenAI o3‑mini baselines.

- Do the experiments support the claims?
  - Yes for controllable test-time scaling: the `Control/Scaling/Performance` metrics (Table 3) and the curves (Figure 1, Figure 4) consistently show that budget forcing both enforces budgets and increases accuracy with more tokens, within limits.
  - Yes for sample efficiency: Figure 2 (right) and Table 1 place `s1‑32B` near the best open models trained with vastly more reasoning data; ablating the selection criteria (Table 2) shows the 1k set is carefully chosen rather than arbitrary.

- Notable caveats disclosed
  - vLLM determinism issues can cause run‑to‑run differences; using full precision mitigates this (Appendix B).
  - Gemini API “recitation error” complicated their own evaluation of Gemini; they manually evaluated AIME24 in the web UI, and left other cells N/A (Table 1; §4.1).

> “Finetuning took 26 minutes on 16 NVIDIA H100 GPUs” (§4.1; §D).

> “Suppressing the end‑of‑thinking token delimiter too often can lead the model into repetitive loops” (§4.2; Figure 4 left).

> Budget forcing ablation: `Control 100%`, `Scaling 15`, `Performance 56.7` (Table 3).

## 6. Limitations and Trade-offs
- Dependence on distillation quality
  - Reasoning traces come from proprietary models (Gemini; later R1 in `s1.1`). Some traces are incorrect (only ~54–63% correct, §2.2, §A). While the method still works, noise in traces may bound ceiling performance.
- Scaling limits and context windows
  - Sequential test-time scaling “eventually flattens out,” and excessive continuation induces loops (Figure 4 left). Long chains can exceed the model’s context window, hurting performance (Figure 7, where 12/30 AIME questions overflow at 512 steps).
- Benchmark scope
  - Focuses on math and science QA (AIME, MATH500, GPQA Diamond). Other domains (e.g., code generation, open‑ended multi‑turn tasks, multimodal reasoning) are not evaluated.
- Evaluation non‑determinism
  - vLLM batching, continuation, and tensor parallelism can change results (Appendix B). The paper addresses this but it remains a practical consideration.
- Compute at inference
  - Gains come from spending more thinking tokens. In production, this is a latency/cost trade‑off. The method helps control it, but does not remove it.
- Comparison to strongest closed systems
  - While `s1‑32B` beats `o1‑preview` on AIME24 and MATH500, it trails `o1` and `o3‑mini` on GPQA/MATH in Table 5. The focus here is simplicity and openness rather than state‑of‑the‑art peak scores.

## 7. Implications and Future Directions
- How this changes the landscape
  - Establishes that a small, well‑curated SFT dataset plus a simple decoder-time controller is enough to produce clear, monotonic test‑time scaling curves—no RL or giant corpora required. This lowers the barrier for researchers to study inference-time reasoning.
  - Provides practical control metrics (`Control/Scaling/Performance`) that other scaling methods can be judged against (§3.2; Table 3).

- Follow‑up research enabled
  - Better extrapolation without loops:
    - Explore dynamic strings instead of a fixed “Wait,” add frequency penalties or temperature schedules to avoid repetition (§6.2, “Limits to further test‑time scaling”).
    - Learn the extrapolation policy (small controller or reward‑free heuristics) while keeping the main model fixed.
  - Combine sequential and parallel scaling:
    - Use REBASE/tree-search selectively when approaching context limits (Figure 7).
    - Study hybrid schedules that interleave single‑trace revision (sequential) with branching exploration (parallel).
  - Data curation science:
    - Make the selection algorithm adaptive to target domains; test if the `Quality + Difficulty + Diversity` triad transfers to other modalities (vision‑language), coding, or dialog planning.
  - RL interplay:
    - Apply budget forcing to RL‑trained “reasoning” models; assess whether RL improves extrapolation (suggested in §6.2). Use the paper’s metrics to compare.

- Practical applications
  - Systems with user‑tunable accuracy/latency: assistants that “think more” when asked to double‑check; on‑device models that default to short thinking and escalate only for difficult items.
  - Education and scientific assistance: math/physics tutoring or problem‑solving tools that can allocate more budget on tricky steps, and terminate early on easier ones.
  - Evaluation infrastructure: the `Control/Scaling/Performance` triad can become part of standard reports for any new inference‑time algorithm.

In short, the paper’s core achievement is making test‑time scaling both simple and rigorous: a small, public dataset (`s1K`) plus a clean decoding trick (`budget forcing`) yields a controllable accuracy‑vs‑tokens curve and competitive results, while providing the community with reproducible metrics and ablations that clarify what works and why.
