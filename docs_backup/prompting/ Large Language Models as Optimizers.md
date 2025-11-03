# Large Language Models as Optimizers

**ArXiv:** [2309.03409](https://arxiv.org/abs/2309.03409)
**Authors:** Chengrun Yang, Xuezhi Wang, Yifeng Lu, Hanxiao Liu, Quoc V. Le, Denny Zhou, Xinyun Chen
**Institutions:** Google DeepMind

## 🎯 Pitch

OPRO transforms Large Language Models (LLMs) into potent derivative-free optimizers capable of outperforming human-engineered prompts in complex reasoning tasks. By treating optimization as a language task, OPRO allows for automated, iterative enhancement of prompts, delivering significant accuracy improvements—as seen with up to an 8% increase on GSM8K benchmarks—while providing a scalable solution to black-box optimization challenges without requiring gradient access or extensive model retraining.

---

## 1. Executive Summary
Optimization by PROmpting (OPRO) reframes large language models (LLMs) as general-purpose, derivative-free optimizers. Instead of coding an algorithm, a “meta‑prompt” describes the optimization problem in natural language and includes a running list of prior solutions with their scores; the LLM then proposes new solutions that are evaluated and fed back into the next step. Applied to prompt optimization for reasoning tasks, OPRO discovers instructions that outperform strong human-written prompts—up to +8% on GSM8K and sizable gains on many Big-Bench Hard (BBH) tasks (Abstract; Table 4; Figure 5).

## 2. Context and Motivation
- Problem addressed
  - Many real-world optimization problems lack gradients or easy-to-program objectives. Existing derivative-free methods often require problem-specific engineering. Prompt engineering for LLMs is itself an optimization problem: the best instruction format varies by task and model, and semantically similar prompts can perform very differently (Section 1; Section 4).
- Why it matters
  - A general way to optimize without gradients enables black-box optimization when you only have API access to a model or evaluator. For prompt engineering, automatic discovery of high-performing instructions can yield immediate accuracy gains on widely used reasoning benchmarks (GSM8K, BBH) with minimal manual effort (Sections 1, 4, 5.2).
- Prior approaches and gaps
  - Soft prompt tuning and gradient-based search require model internals or differentiable objectives (Related Work).
  - Edit-based methods and recent LLM-driven prompt generators typically modify a single prompt or keep semantics fixed, using limited feedback (Zhou et al., 2022b; Pryzant et al., 2023). They generally do not use the full history of attempts and scores as an optimization trajectory (Section 4.2 vs. 6).
- Positioning
  - OPRO leverages an LLM’s in-context pattern recognition: it sees a trajectory of solutions with their scores and learns which attributes correlate with better performance, then proposes new candidates. This turns optimization into iterative prompting, without training new models or accessing gradients (Figure 2; Section 2).

## 3. Technical Approach
OPRO is a closed-loop prompting-and-scoring system. Think of it as giving an LLM a scoreboard and asking it to propose the next move.

- Core loop (Figure 2; Section 2)
  1. Construct a `meta-prompt` that includes:
     - A natural-language task description and any constraints.
     - An `optimization trajectory`: previously tried solutions paired with their scores, usually sorted from worst to best.
     - `Meta-instructions`: explicit guidance such as “propose new solutions that are different from the above and aim for a higher score.”
  2. The `optimizer LLM` reads the meta-prompt and generates multiple new candidate solutions per step (e.g., 8).
  3. An external `evaluator` scores the candidates:
     - For mathematical case studies, the evaluator computes objective values.
     - For prompt optimization, the `scorer LLM` runs the task using the generated instruction and returns accuracy on a small training subset (Section 5.1).
  4. Append the new solution–score pairs to the meta-prompt (keeping only the top-K for length) and repeat until convergence or a step limit.

- Roles and terminology (Section 4.1)
  - `Optimizer LLM`: the model that proposes new solutions (PaLM 2-L-IT, PaLM 2-L, text-bison, gpt-3.5-turbo, gpt-4).
  - `Scorer LLM`: the model whose performance is optimized via the instruction (pre-trained PaLM 2-L or text-bison).
  - Instruction positions (Appendix B): where to insert the instruction relative to the question/answer format.
    - `Q_begin`: instruction before the question.
    - `Q_end`: instruction after the question.
    - `A_begin`: instruction at the beginning of the model’s answer (used when the scorer is a pre-trained, non-instruction-tuned model like PaLM 2-L).

- Meta-prompt design (Section 4.2; Figure 3; Appendix C.2)
  - Includes:
    - Several `exemplars` (few input–output pairs) to ground the task.
    - Top-N prior solutions and their `bucketized` accuracy scores (e.g., rounded to 0–100).
    - Explicit formatting constraints (e.g., “write your text in [square brackets]”).
  - Example for GSM8K (Figure 3): shows prior instructions with scores, a few math problems with correct answers, and asks the optimizer to propose a different instruction that will yield higher accuracy.

- Stabilizing optimization and exploration–exploitation (Section 2.3; Section 5.3)
  - Generate multiple candidates per step to reduce variance.
  - Tune sampling `temperature` for diversity. Empirically, temperature 1.0 balances exploration and exploitation best (Figure 10).
  - Sort prior solutions ascending (worst → best) to exploit recency bias: LLMs attend more to the end of the context (Figures 7a–b).
  - Show explicit scores (rather than only ordering) to help the LLM infer quality differences (Figures 7c–d).
  - Keep the meta-prompt concise: 3 exemplars work well; adding 10 can dilute the optimization signal (Figures 7e–f).
  - Batch size vs. steps: 8 candidates per step typically outperforms 1–4 or 16 for a fixed evaluation budget (Figure 8).

- Example loops beyond prompts
  - Linear regression and Traveling Salesman Problem (TSP) are used as motivating black-box optimization cases where the meta-prompt lists prior `(w, b)` pairs or routes and their objective values, and asks for a new point/route with a better value (Appendix C.1; Table 2; Table 3).

## 4. Key Insights and Innovations
- Treating LLMs as optimizers via meta-prompting (fundamental innovation)
  - Novelty: converts optimization into an in-context generation task that reasons over a history of solutions and scores. This differs from editing a single prompt or relying on hand-crafted mutation/crossover operators (Section 2; Related Work).
  - Significance: works with black-box evaluators and requires only API access—no gradients, training, or internal model access.

- Optimization trajectory as in-context signal (conceptual and practical)
  - The meta-prompt provides a sorted leaderboard with numeric scores; the LLM learns which features correlate with success and proposes candidates accordingly. Ablations show ordering and explicit scores materially help (Figures 7a–d).

- Strong prompt optimization with little data and simple infrastructure (practical advance)
  - With only a small training subset (3.5% of GSM8K; 20% of each BBH task), OPRO discovers instructions that outperform widely used baselines like “Let’s think step by step.” (Section 5.2; Table 4; Figure 5).

- Demonstration on classic optimization tasks (proof-of-concept breadth)
  - On small linear regression and TSP instances, LLMs using OPRO can find good solutions and sometimes match heuristic solvers (Table 2; Table 3), illustrating that the method is not confined to prompt tuning.

## 5. Experimental Analysis
- Evaluation methodology (Section 5.1)
  - Optimizer LLMs: PaLM 2-L-IT, PaLM 2-L, text-bison, gpt-3.5-turbo, gpt-4.
  - Scorer LLMs: PaLM 2-L (pre-trained) and text-bison.
  - Prompt optimization settings:
    - GSM8K: use 3.5% of the 7,473 training examples to compute training accuracy during optimization; evaluate best instructions on all 1,319 test samples at the end (Section 5.2.1).
    - BBH: per task, use 20% for training during optimization, 80% for test; instructions placed at `A_begin` for PaLM 2-L and `Q_begin` for text-bison (Section 5.2.2).
  - Meta-prompt defaults: keep top 20 prior instructions; sample 3 exemplars; generate 8 new instructions per step; temperature 1.0 (Section 5.1).

- Main quantitative results
  - GSM8K (Table 4; Figures 1a, 4)
    - With PaLM 2-L as scorer and PaLM 2-L-IT as optimizer, OPRO discovers:
      > “Take a deep breath and work on this problem step-by-step.” with 80.2% test accuracy (Table 4).
    - This outperforms the zero-shot “Let’s think step by step.” baseline at 71.8% by +8.4 points (Table 4) and approaches few-shot chain-of-thought levels for this scorer.
    - Optimization curves show steady improvement with occasional leaps, reflecting both discovery of better single prompts and a shift toward generating generally stronger prompts per step (Figure 1a; Section 5.2.1).
    - Even the pre-trained PaLM 2-L can optimize its own prompts when used as both scorer and optimizer; the loop transitions from weak starters (“The answer is”) to stronger prefixes (“Let’s do it:”) with rising training accuracy (Figure 4b).
  - BBH (Figure 5; Table 5; Appendix E)
    - Across 23 tasks, the OPRO-found instructions beat “Let’s think step by step.” on most tasks:
      > With PaLM 2-L as scorer, OPRO wins by >5% on 19/23 tasks (Figure 5a).
      > With text-bison as scorer, wins by >5% on 15/23 tasks (Figure 5c).
    - Example highs (Table 5):
      > `movie_recommendation`: 90.8% with a domain-specific A_begin instruction under PaLM 2-L scorer and PaLM 2-L-IT optimizer.
      > `ruin_names`: 83.6% with a Q_begin instruction under text-bison scorer and PaLM 2-L-IT optimizer.
    - Optimization curves for tasks like `ruin_names` and `temporal_sequences` show consistent upward training accuracy (Figure 6).
    - The paper documents that paraphrases with similar meaning can differ wildly in accuracy, underscoring the value of iterative search:
      > For GSM8K with PaLM 2-L scorer: “Let’s think step by step.” = 71.8%, “Let’s solve the problem together.” = 60.5%, “Let’s work together to solve this problem step by step.” = 49.4% (Section 5.2.3).
  - Transfer to other math datasets (Table 6)
    - Applying GSM8K-optimized instructions to MultiArith and AQuA yields:
      > 95.3% on MultiArith and 54.3% on AQuA for A_begin “Take a deep breath …” with PaLM 2-L scorer, exceeding baselines (Table 6).
  - Motivating optimization tasks (Table 2; Table 3)
    - Linear regression (2D, synthetic):
      > gpt-4 generally reaches optima in fewer steps and with fewer unique points than text-bison and gpt-3.5-turbo; all models explore far fewer points than exhaustive search, indicating they infer descent directions from the trajectory (Table 2).
    - TSP (n=10–50):
      > For n=10, all LLMs achieve 0% optimality gap on all 5 instances; gpt-4 finds optima ~4× faster than others (9.6 steps vs. 40–47; Table 3).
      > For n=20, gpt-4’s average gap is 1.4%, better than gpt-3.5-turbo (4.4%) and text-bison (30.4%), but worse than Farthest Insertion (0.2%). For n=50, all LLMs degrade substantially (Table 3).

- Ablations and robustness (Section 5.3)
  - Order prior solutions ascending (worst → best): converges faster and higher (Figures 7a–b).
  - Show numeric scores: improves over showing order alone (Figures 7c–d).
  - Use 3 exemplars rather than 10 or none (Figures 7e–f).
  - Batch 8 candidates per step: best for a fixed total evaluation budget (Figure 8).
  - Temperature 1.0 best; 0.0–0.5 under-explore; 1.5–2.0 over-explore and ignore trajectory (Figure 10).
  - One-step generation baseline that samples 50 instructions once performs much worse than iterative OPRO (Section “Comparison with one-step instruction generation”; concrete GSM8K and BBH numbers given there).
  - Starting-point sensitivity exists—better initial prompts accelerate improvement (Figure 9b).
  - Overfitting check: when holding out a same-sized validation set, validation curves roughly track training curves, suggesting ranking by training accuracy remains meaningful (Figure 11).

- Comparison to evolutionary prompting (EvoPrompt) (Section 5.5; Figure 12)
  - OPRO (trajectory + exemplars) consistently improves over steps; EvoPrompt variants that rely on crossover/mutation of two prompts and no exemplars often stagnate or degrade on GSM8K starting from generic prompts (Figure 12a). With task-specific starters on a BBH task, EvoPrompt (DE) improves but is less stable than OPRO (Figure 12b).

- Failure cases (Appendix A)
  - Hallucinated arithmetic in math optimization without tool use.
  - Occasionally repeats prior solutions despite instructions to avoid duplicates.
  - Can get stuck in bumpy landscapes (e.g., Rosenbrock function) or when history points mislead direction (Appendix A; Figure 13).

- Overall assessment
  - The experiments are broad (multiple LLMs, two domains, many tasks) and ablations carefully justify design choices. The strongest claims are about prompt optimization performance and stability; mathematical optimization demonstrations are explicitly scoped as small-scale proofs-of-concept (Limitations under Section 3.2 and “Limitations” boxes after Table 3).

## 6. Limitations and Trade-offs
- Dependence on context window and prompt budget
  - The meta-prompt must fit the optimization trajectory and exemplars. Large problems (e.g., high-dimensional regression, TSP with many nodes) don’t fit well and performance degrades (Table 3 discussion; “Limitations” under Section 3.2).
- Requires many evaluations
  - Each step evaluates multiple candidates; total cost scales with number of steps × batch size. Although sample-efficient relative to naive search (Table 2), it still can be expensive for API-based scorers.
- Needs a training set for scoring in prompt optimization
  - Overfitting can occur, though validation tracks training reasonably (Figure 11). Scenarios without labeled data cannot use accuracy as an objective unless alternative feedback is devised (Section 7 Conclusion’s discussion).
- Sensitivity to initialization and sampling temperature
  - Better starting prompts accelerate convergence; low temperature stalls, high temperature wanders (Figures 9–10).
- Objective function access and fidelity
  - In math tasks, the LLM sometimes hallucinates computations; without tool augmentation, it can misjudge descent directions (Appendix A).
- Instruction style vs. position mismatch
  - Some generated prompts are ill-suited for `A_begin` vs. `Q_begin` positions unless the starting style nudges them (Appendix E.2–E.3).
- Not a replacement for specialized solvers
  - The paper explicitly does not claim superiority to gradient-based optimizers or state-of-the-art combinatorial solvers (Limitations after Table 3). Performance on larger-scale TSP is behind classic heuristics.

## 7. Implications and Future Directions
- Broader impact on optimization and prompt engineering
  - OPRO reframes optimization as a language task. This suggests a general recipe for black-box optimization when one can describe the goal, constraints, and “leaderboard” in text. For practitioners, it turns prompt engineering into an iterative, automated process that can be run with a small labeled subset.
- Practical applications
  - Auto-tuning system prompts/instructions for:
    - Math and logical reasoning benchmarks (as demonstrated).
    - Domain-specific assistants (medical, legal, customer support) where tone and structure matter.
    - Multi-agent or tool-using systems that rely on high-quality instructions.
  - Small-scale combinatorial or parameter search tasks where human intuition benefits from a history of scored candidates (e.g., simple hyperparameter choices, few-step workflows).
- Research directions
  - Richer feedback: incorporate explicit natural-language critiques of failures (Section 7 Conclusion points to using “richer feedback about error cases”).
  - Tool augmentation: integrate calculators/verifiers to reduce hallucination in math optimization (Appendix A mentions triggerable tools; related work like Toolformer).
  - Trajectory learning: train or fine-tune models to internalize how to read and act on optimization trajectories, akin to OptFormer but without full training (Related Work).
  - Active selection of exemplars: prioritize “hard” cases where past prompts failed instead of random sampling; the paper notes early attempts yielded similar results, indicating room for better strategies (Conclusion).
  - Multi-objective and constrained optimization: extend meta-prompts to handle trade-offs (e.g., accuracy vs. length), enforce constraints, and support Pareto search.
  - Scalability: compress trajectories, retrieve most informative history, or use external memory to overcome context-length constraints.
  - Theory: formalize why ascending order and explicit scores help (recency bias is hypothesized in Section 5.3); model the exploration–exploitation dynamics of LLM sampling.

> Bottom line: OPRO provides a simple, general, and surprisingly strong recipe—“show the model the scoreboard and ask it to improve”—that reliably upgrades prompts and can tackle small-scale black-box optimization, with clear levers (ordering, scores, batch size, temperature) that the ablations validate (Figures 7–10).
