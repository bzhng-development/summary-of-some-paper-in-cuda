# Large Language Models as Optimizers

**ArXiv:** [2309.03409](https://arxiv.org/abs/2309.03409)

## 🎯 Pitch

This paper introduces Optimization by PROmpting (OPRO), a novel framework where a large language model (LLM) itself serves as an optimizer by iteratively generating candidate solutions based on a natural language meta-prompt that records past solutions and their scores. By leveraging only black-box access and natural language descriptions, OPRO enables powerful, general-purpose optimization—dramatically improving prompt quality for challenging reasoning tasks (up to +8% accuracy on GSM8K, and up to +50% on Big-Bench Hard) and demonstrating that LLMs can effectively navigate complex, derivative-free optimization problems. This breakthrough highlights the potential to harness LLMs not just for traditional text generation, but as flexible optimizers with wide-reaching implications for model usability and alignment in diverse domains.

---

## 1. Executive Summary (2-3 sentences)
This paper introduces Optimization by PROmpting (`OPRO`), a general procedure that uses a large language model (LLM) itself as an optimizer. By describing an optimization task and a running list of prior solution–score pairs in natural language (a “meta-prompt”), the LLM proposes new candidates that are externally scored and fed back, yielding iterative improvement. The approach boosts prompt quality for reasoning tasks substantially—e.g., on GSM8K it finds prompts that raise PaLM 2-L test accuracy from 71.8% to 80.2% (Table 4) and delivers large gains across 23 Big-Bench Hard tasks (Figure 5, Table 7)—and also shows the LLM can navigate small-scale math optimization problems.

## 2. Context and Motivation
- Problem addressed
  - Many optimization problems are “derivative-free” (no gradients available). Separately, prompt engineering for LLMs is itself an optimization problem over a large, discrete space where small wording changes cause large performance shifts (Section 4, 5.2.3).
  - Existing prompt optimization methods either need model internals (soft prompts, gradients) or use narrow edit operations; most are hard to apply when you only have API access to a black-box model (Section 6, “Related Work”).
- Why it matters
  - Real-world use: Better prompts materially change LLM performance on reasoning benchmarks (GSM8K, BBH), which correlate with downstream analytical tasks.
  - Theoretical significance: Demonstrates that an LLM can act as a general-purpose optimizer guided solely by natural language descriptions and observed solution quality (Figure 2).
- Prior approaches and limitations
  - Soft/continuous prompts and gradient-based/discrete search require access to model internals or differentiable objectives (Section 6).
  - Edit-based and LLM-generated prompts (e.g., APE, APO) operate by mutating a single prompt with feedback; they do not leverage a full optimization trajectory and often require semantically similar variants (Section 6).
  - Evolutionary prompting systems (e.g., EvoPrompt) require carefully seeded prompts and pairwise crossover; they omit task exemplars and full trajectory context, limiting stability and gains (Section 5.5, Figure 12).
- Positioning
  - `OPRO` reframes optimization as in-context learning: the LLM reads a natural language description of the goal plus a sorted history of solutions with scores and proposes better candidates. It does not assume gradients, model internals, or fixed edit operators, and is easy to retarget to new tasks by changing the meta-prompt (Section 2, Figures 2–3).

## 3. Technical Approach
At a high level, `OPRO` runs an iterative loop with three roles and a shared text prompt (the “meta-prompt”).

- Core roles (Section 4.1)
  - `optimizer LLM`: the model that reads the meta-prompt and generates new candidate solutions.
  - `scorer LLM` (or any evaluator): the system that receives each candidate and produces an objective score; for prompt optimization this is an LLM whose accuracy on a small training set is measured.
  - “Solutions”: the objects being optimized. In prompt optimization, a “solution” is a natural-language instruction; in math optimization, it’s a vector like `(w, b)` or a TSP tour.

- The meta-prompt (Section 2.2, Figure 3; Appendix C for math examples)
  - Optimization problem description: in natural language, states the goal (“generate a new instruction that achieves a higher accuracy”) and any format constraints.
  - Optimization trajectory: a list of past solutions paired with their scores, sorted from worst to best. This shows the optimizer what has worked so far and by how much.
  - Task exemplars (for prompt optimization): a few input–output examples illustrating the task and where the instruction should be inserted (Figure 3).
  - Meta-instructions: brief guidance such as “be concise and generally applicable” and the required output format (Figure 3).

- Iterative procedure (Section 2; Section 5.1 for default hyperparameters)
  1. Construct the meta-prompt with the current top-K solution–score pairs (default K=20) and a small set of exemplars (default 3).
  2. Sample multiple new solutions from the optimizer LLM (default 8 per step) at a chosen temperature (default 1.0) to balance exploration and exploitation (Section 2.3).
  3. Evaluate and score each new solution (for prompt optimization, run the scorer LLM at temperature 0 on the training subset).
  4. Append new solution–score pairs, keep the top-K, and repeat until convergence or reaching a step budget.

- Design choices and their rationale (Section 2.1–2.3; ablations in Section 5.3, Figures 7–10)
  - Multiple candidates per step improve stability and reduce variance, analogous to mini-batching in SGD (Section 2.3; Figure 8).
  - Temperature controls exploration; 1.0 worked best across GSM8K and BBH; too low gets stuck, too high ignores the trajectory (Figure 10).
  - Show explicit scores (bucketized integers) and sort solutions ascending; both choices give better convergence than omitting scores or using random/highest-first order (Figures 7c–d, 7a–b).
  - Include a few exemplars (3 by default); many more inflate the prompt and can distract from the optimization trajectory (Figures 7e–f).

- Prompt optimization specifics (Section 4)
  - Where to insert the instruction:
    - `Q_begin`: before the question text.
    - `Q_end`: after the question text.
    - `A_begin`: at the beginning of the model’s answer (used for non-instruction-tuned scorers like PaLM 2-L). Examples in Appendix B, Figures 14–18.
  - Train/test design: GSM8K uses 3.5% of the training set to score candidates efficiently; BBH uses a 20/80 train–test split; evaluation is greedy decoding (temperature 0) for consistent scoring (Section 5.1).

- Mathematical optimization setups (Section 3; Appendix C.1)
  - Linear regression: optimizer proposes `(w, b)` pairs; the meta-prompt lists prior pairs and their objective values (Figure 19). The analytic form is hidden to keep it black-box (Section 3.1).
  - Traveling Salesman Problem (TSP): optimizer proposes tours as sequences over node indices (Figure 20); quality measured by optimality gap against a Gurobi oracle (Section 3.2).

Analogy: Think of `OPRO` as showing an LLM a scoreboard and a highlight reel of previous plays; the model then suggests a few new plays that should score higher, and the game continues.

## 4. Key Insights and Innovations
- Using an LLM as a general-purpose optimizer via natural language (Figure 2)
  - Novelty: transforms optimization into in-context learning over a trajectory of solution–score pairs instead of requiring gradients or handcrafted edit rules.
  - Significance: works across domains—math optimization and prompt search—by swapping the problem description and evaluator (Sections 3 and 4).
- Optimization trajectory as a first-class in-context signal (Section 2.2; ablations in Section 5.3)
  - Different from prior prompt optimizers (APE/APO/EvoPrompt) that edit one or two prompts, `OPRO` conditions on a ranked list of many prior solutions and their scores. Ablations show sorting order and explicit scores materially affect convergence (Figures 7a–d).
- Simple but effective exploration–exploitation control
  - Multi-sample per step and temperature tuning yield stable progress; 8 samples/step at temperature 1.0 was consistently strong (Figures 8 and 10).
- Practical insight: tiny wording changes cause huge accuracy swings (Section 5.2.3)
  - Example on GSM8K: “Let’s think step by step.” (71.8% test) vs “Let’s work together to solve this problem step by step.” (49.4%) (Section 5.2.3). `OPRO` systematically searches the discrete prompt space to navigate this sensitivity.
- Evidence that iterative, trajectory-aware generation beats single-shot prompt generation
  - A one-step baseline that generates 50 instructions without iteration failed to improve: best remained “Let’s solve the problem” (60.8% test on GSM8K), while `OPRO` reached 76.3% by step 5 with only 8 candidates per step (Section 5.3, “Comparison with one-step instruction generation”).

## 5. Experimental Analysis
- Evaluation methodology (Section 5.1)
  - Datasets: GSM8K (grade-school math word problems), BBH (23 challenging reasoning tasks), plus transfer to MultiArith and AQuA (Table 6).
  - Metrics: accuracy for prompt optimization; optimality gap and success rate for TSP (Table 3); speed measured in optimization steps or candidate evaluations.
  - Scorers and optimizers: combinations of PaLM 2-L (pretrained), PaLM 2-L-IT (instruction-tuned), text-bison, `gpt-3.5-turbo`, and `gpt-4` (Sections 5.1–5.2).
  - Implementation defaults: 8 candidates/step, keep top-20 in trajectory, 3 exemplars per step, optimizer temperature 1.0, scorer temperature 0 (Section 5.1).

- Main quantitative results
  - GSM8K (Table 4; Figures 1a, 4)
    - Best result with PaLM 2-L as scorer: an instruction found by PaLM 2-L-IT—“Take a deep breath and work on this problem step-by-step.”—achieves 80.2% test accuracy, +8.4 points over “Let’s think step by step.” (71.8%).
    - Other optimizers also improved: PaLM 2-L found “Break this down.” (79.9%); `gpt-3.5-turbo` found a longer instruction yielding 78.5%; `gpt-4` reached 74.5%.
    - With text-bison as scorer (instruction-tuned), optimized `Q_end` instruction “Let’s work through this problem step-by-step:” reached 68.5% vs 65.6% for a strong baseline prompt (Table 4).
    - Optimization curves show steady upward trends with variance shrinking over time (Figure 1a; Figure 4a-b).
  - BBH (Figure 5; Table 7; Figure 6 for two tasks)
    - With PaLM 2-L as scorer and PaLM 2-L-IT as optimizer, `OPRO` outperforms “Let’s think step by step.” by >5 points on 19/23 tasks; similar gains against the empty-string baseline on 20/23 tasks (Figure 5a–b).
    - Example top instructions and accuracies (Table 5):
      - `movie_recommendation`: 90.8% with a targeted instruction explaining the selection criteria (A_begin).
      - `ruin_names`: 88.0% with a succinct “Which is the funniest pun…” A_begin instruction.
      - `temporal_sequences` (with text-bison scorer): 80.4% with a precise rule-based instruction (Q_begin).
    - Per-task curves rise over steps, suggesting iterative trajectory conditioning helps (Figures 6, 23, 24).
  - Transfer across datasets (Table 6)
    - The best GSM8K instruction for PaLM 2-L generalizes: 95.3% on MultiArith (vs 85.7% for “Let’s think step by step.”) and 54.3% on AQuA (vs 44.9%).
    - Transfer is weaker for some scorer/instruction combinations (e.g., a verbose text-bison-specific instruction performs 96.8% on MultiArith but 37.8% on AQuA), highlighting model- and task-specificity.
  - Mathematical optimization (Section 3; Tables 2–3)
    - Linear regression (2 parameters): `gpt-4` reaches global optima with fewer unique evaluations and steps than text-bison and `gpt-3.5-turbo`; e.g., when truth is inside the starting region, `gpt-4` needs ~4–6 steps vs ~6–13 for others (Table 2).
    - TSP: `gpt-4` attains 0% optimality gap on all 10-node problems and small gaps on 15–20 nodes (e.g., 0.2% at n=20), but quality degrades for 50 nodes (11% gap), where farthest insertion heuristic is better (Table 3). Success rates and steps are reported; e.g., 5/5 successes at n=10 with 9.6 steps on average.

- Ablations, robustness, and diagnostics (Section 5.3)
  - Trajectory presentation matters: ascending order with explicit scores is best (Figures 7a–d).
  - Exemplars: using 3 is better than none, but 10 can hurt (Figures 7e–f).
  - Batch size: 8 candidates/step is a good trade-off given a fixed total evaluation budget (Figure 8).
  - Initialization: strong seeds help early progress; with PaLM 2-L scorer, starting from “Let’s think step by step.” dominates starting from empty (Figure 9b).
  - Temperature: 1.0 balances exploration and exploitation; too low stalls, too high becomes noisy (Figure 10).
  - Overfitting check: when a validation set is held out, validation curves generally track training curves (Figure 11), but final training accuracies often exceed test by 5–20 points (discussion in Section 5.4; Tables 7 and 10 list both).
  - Comparison to EvoPrompt: `OPRO` is more stable and effective, especially from generic seeds; EvoPrompt improves only when given task-specific seeds and still lags (Figure 12).

- Do the experiments support the claims?
  - For prompt optimization, yes: multiple combinations of scorer/optimizer models and two benchmarks show consistent, often large improvements (Table 4, Figure 5, Table 7).
  - For mathematical optimization, evidence is qualitative: `OPRO` can descend towards optima on small problems but is not competitive for larger scales (Table 3), aligning with the stated goal to demonstrate feasibility rather than surpass specialized solvers (Section 3, “Limitations”).

- Notable failure analyses (Appendix A)
  - Hallucinated arithmetic when the LLM tries to compute objective values without tool use.
  - Repeats previous solutions despite instructions not to (incomplete instruction-following).
  - Gets stuck where descent directions from history are misleading (e.g., conflicting gradients of `w` and `b`), mitigated by sampling multiple proposals per step.
  - Struggles with bumpy landscapes like Rosenbrock; can fall into the valley around (0,0) and fail to navigate to the optimum (Figure 13).

> Example GSM8K improvement (Table 4): “Take a deep breath and work on this problem step-by-step.” → 80.2% test vs 71.8% for “Let’s think step by step.”

> Example BBH improvement (Table 7): `word_sorting` overall accuracy rises from 4.0% (“Let’s think step by step.”) to 54.4% with the `OPRO`-found instruction (+50.4 points).

## 6. Limitations and Trade-offs
- Assumptions and dependencies
  - Requires an evaluator to score candidates. For prompt optimization, this is another LLM run over a (small) labeled set (Section 4.1); for math problems, an external oracle or calculator.
  - Assumes the LLM can infer useful improvement directions from a textual trajectory; effectiveness depends on the optimizer model’s in-context learning ability.
- Scalability constraints
  - Context window limits how many prior solutions and exemplars can be shown, capping problem size (e.g., TSP with many nodes or high-dimensional regression; Section 3, “Limitations”).
  - Computational cost grows with candidates per step × steps × evaluation set size.
- Quality and stability issues
  - Overfitting to the training subset can inflate training accuracy relative to test; early stopping or larger training subsets help (Section 5.4).
  - Sensitive to initialization; stronger seeds improve early progress (Figure 9b).
  - The optimizer may hallucinate numeric computations or fail to obey “new solution” constraints (Appendix A).
- Performance boundaries
  - Not competitive with specialized solvers for large-scale combinatorial optimization (Table 3).
  - Gains can be model- and task-specific; verbose instructions that help one scorer may transfer poorly (Table 6).

## 7. Implications and Future Directions
- How this work shifts the landscape
  - Establishes a simple, general recipe to “plug in” an LLM as a black-box optimizer controlled entirely by a natural-language meta-prompt (Figures 2–3). This lowers the barrier to optimize over discrete design spaces (prompts, system messages, tool-use instructions) without gradients or internal access.
  - Shows that in-context trajectories can guide nontrivial search, suggesting a broader paradigm of “trajectory-conditioned problem solving.”
- Follow-up research enabled
  - Tool-augmented `OPRO`: invoke calculators/solvers when arithmetic is needed (Appendix A notes this would fix hallucinated values), akin to Toolformer-style integration.
  - Richer feedback signals: go beyond aggregate accuracy to include error analyses, counterexamples, or per-instance rationales as part of the trajectory; Section 7 proposes summarizing distinguishing features between high- and low-quality prompts.
  - Memory and scaling: compress trajectories, use retrieval to select informative history, or use multi-round hierarchical `OPRO` for large problems.
  - Better exploration strategies: adaptive temperatures, Thompson sampling over instruction families, or novelty search to avoid local optima.
  - Hybrid optimization: combine `OPRO` with reinforcement learning or evolutionary operators while retaining trajectory conditioning (contrast to Section 5.5).
- Practical applications
  - Automated prompt engineering for production LLM systems (system prompts, safety prompts, tool-use scaffolds).
  - Configuration search for multi-agent LLM systems or chains-of-thought styles tailored to tasks.
  - Black-box tuning of non-differentiable policies (e.g., instructions for data labeling, heuristics in rule-based systems) by scoring outcomes on held-out sets.
  - Small-scale combinatorial design tasks where a textual trajectory is informative and exact solvers are unavailable.

In sum, `OPRO` offers a surprisingly capable, low-infrastructure way to make iterative progress on discrete optimization problems—especially prompt optimization—by leveraging the LLM’s ability to recognize patterns in a history of solutions and extrapolate improved candidates. The strongest evidence is on GSM8K and BBH where it consistently beats established prompting baselines (Tables 4–7), while mathematical tests clarify where trajectory-conditioned optimization begins to break down (Tables 2–3, Appendix A), pointing to rich opportunities for tool use and better exploration.
