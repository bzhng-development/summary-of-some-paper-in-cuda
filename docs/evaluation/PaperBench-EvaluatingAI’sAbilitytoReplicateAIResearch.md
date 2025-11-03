# PaperBench: Evaluating AI’s Ability to Replicate AI Research

**ArXiv:** [2504.01848](https://arxiv.org/abs/2504.01848)

## 🎯 Pitch

PaperBench introduces a rigorous new benchmark that tests whether AI agents can autonomously replicate cutting-edge machine learning research—from reading the paper and building a codebase to running experiments and reproducing results, entirely from scratch. By formalizing replication as a hierarchical, weighted rubric and employing an LLM-based judge, PaperBench exposes the current limitations of even the best frontier models, highlighting a crucial gap between human and AI research capabilities and setting a new standard for assessing AI autonomy in real-world scientific R&D.

---

## 1. Executive Summary (2–3 sentences)
PaperBench is a benchmark for testing whether AI agents can replicate modern machine‑learning (ML) research papers from scratch: read a paper, build a working codebase, run the experiments, and reproduce the reported results. It formalizes replication as a weighted, hierarchical rubric and uses a validated LLM‑based judge to grade results, revealing that current frontier agents achieve only partial success (best average score 21.0% with Claude 3.5 Sonnet; Table 4), well below a strong human baseline (Figure 3).

## 2. Context and Motivation
- Problem addressed
  - There is no rigorous, scalable way to measure whether an autonomous AI agent can carry out end‑to‑end ML research replication: understanding a paper, implementing the methods, running experiments, and verifying results without relying on the original authors’ code (Section 2; Section 2.5, Rule 1).
- Why this matters
  - Practical impact: Agents that can replicate research could accelerate scientific progress, but also raise safety questions about increasingly autonomous ML R&D (Introduction).
  - Evaluation gap: Traditional benchmarks emphasize smaller tasks (e.g., code writing, Kaggle competitions) or rely on the existence of an official codebase, which sidestep the hardest parts of real ML research: ambiguous specifications, complex pipelines, and long‑horizon execution (Related Work, Section 6).
- Prior approaches and their gaps
  - Using authors’ repositories to reproduce results (e.g., CORE‑Bench) tests debugging and environment setup but not full re‑implementation from scratch (Section 2, citing Siegel et al., 2024).
  - Kaggle‑style ML agent benchmarks (MLE‑bench, MLAgentBench, DSBench) focus on narrower or dated tasks and provide clear scoring functions, avoiding the ambiguity of modern research reproduction (Section 6).
  - RE‑Bench offers challenging engineering tasks but many include a built‑in scoring function; PaperBench must evaluate broad, open‑ended replication where no single score can capture correctness (Section 6).
- How this work positions itself
  - PaperBench evaluates 20 ICML 2024 Spotlight/Oral papers spanning 12 topics (Table 2) using author‑approved, hierarchical rubrics (Section 3.1). Replication is judged by executing a submission’s `reproduce.sh` in a clean environment (Section 2.2) and scoring granular outcomes, with results aggregated into a single Replication Score (Section 2.3; Figure 2).

## 3. Technical Approach
This section explains the benchmark as a pipeline: task specification → reproduction execution → rubric‑based grading with an LLM judge.

- Task setup (Section 2.1; Figure 1)
  - Input to the agent (the “candidate”):
    - The paper (PDF and Markdown) and a clarifying `addendum` (Section 3.2).
    - The rubric exists but is hidden from the agent during attempts to prevent overfitting (Section 2.1).
  - Required output (“submission”):
    - A repository containing all code to reproduce the paper’s empirical contributions and an entrypoint script `reproduce.sh` at the repo root (Section 2.1).
    - The script should orchestrate the entire replication, producing outputs (tables/plots) and a `reproduce.log`.

- Reproduction phase: enforcing clean execution (Section 2.2)
  - The submission is copied to a fresh Ubuntu 24.04 VM with a single NVIDIA A10 GPU and executed by running `reproduce.sh`.
  - Outputs produced during this run (results files and `reproduce.log`) are treated as the only evidence of reproduction.
  - Rationale: This separation prevents agents from “baking in” results at task time and increases credibility of claimed reproductions.

- Rubric design and scoring (Sections 2.3–2.4; Figure 2; Table 1; Section 3.1)
  - Hierarchical rubric
    - A rubric is a tree that decomposes “replicate the paper’s main contributions” into increasingly specific requirements.
    - The leaves (“leaf nodes”) are binary criteria—each is marked pass/fail based on evidence.
    - Every node has an importance weight; parent scores are the weighted average of children (Figure 2).
    - The final Replication Score is the root score (a weighted proportion of satisfied requirements).
  - Requirement types (Table 1; Section 2.4)
    - `Code Development`: Is the code for a specific component correctly implemented? Evidence: source code and docs.
    - `Execution`: Did running `reproduce.sh` actually execute the relevant pipeline step? Evidence: code, `reproduce.sh`, and `reproduce.log`.
    - `Result Match`: Do the reproduced outputs match the reported findings (within allowed tolerance)? Evidence: `reproduce.sh`, `reproduce.log`, and files created during reproduction.
  - Why three types?
    - A pure “results‑only” rubric would miss partial progress; adding `Execution` and `Code Development` credits incremental steps (Section 2.4).
    - A pure “code‑only” rubric is insufficient because code correctness is hard to establish without running it (Section 2.4).

- Rules to ensure fair, from‑scratch replication (Section 2.5)
  - Web browsing is allowed, but any blacklisted sources—especially the original authors’ code—are prohibited (Section 2.5, Rule 1).
  - A simple monitor flags blacklisted URLs in agent logs; confirmed violations are disqualified (10 of 646 runs; Section 2.5).

- LLM‑based grading (“SimpleJudge”) and judge validation (“JudgeEval”) (Section 4; Table 3; Appendix D)
  - How the judge works (Section 4.1; Appendix D)
    - For each leaf criterion, the judge receives: the paper, the rubric, the criterion, and a filtered view of the submission (top‑k most relevant files selected via a file‑ranking prompt; Figure 7).
    - The judge inspects the relevant artifacts depending on criterion type (Table 1) and returns 0/1 with rationale (Figures 8–9).
  - Backend model and cost
    - Default judge: `o3-mini-2025-01-31` with high reasoning effort (Section 4.1).
    - Average cost ≈ $66 per paper; far cheaper than expert human grading (Figure 5; Section 4.1).
  - Validating the judge with `JudgeEval` (Section 4.2)
    - A set of human‑graded submissions across 5 papers is used as ground truth.
    - Judge performance: F1 = 0.83 for `o3-mini` at ≈ $66/paper; `o1` is similar (0.84) but ≈ $830/paper (Table 3).

- Dataset and rubric creation (Sections 3, 3.1–3.2; Table 2; Table 7)
  - Papers: 20 ICML 2024 Spotlight/Oral papers across 12 topics (Table 2).
  - Scale: 8,316 leaf nodes across papers; rubrics co‑developed with a paper author for accuracy, with node weights reflecting importance (Section 3.1; Table 7).
  - Addendums clarify underspecified details; some judge‑only addendums assist grading (Section 3.2).

- Agent scaffolds and execution environment (Sections 5.1, 5.3; Appendix F)
  - Environment: Ubuntu 24.04 Docker container, single A10 GPU, internet, and API keys for needed services (Section 5.1).
  - `BasicAgent`: a ReAct‑style loop with tools (bash, Python executor, web browser, paginated file reader); can choose to end early (Section 5.1; Appendix F.1).
  - `IterativeAgent`: forces full‑time work by removing the “end task” tool and prompting the model to take one small step at a time (Section 5.3; Appendix F.2).

- Accessibility variant (Section 2.6)
  - `PaperBench Code‑Dev`: grades only `Code Development` nodes, skipping the reproduction run; cuts grading cost by ~85% and removes the need for GPUs.

Analogy: Think of PaperBench like a driving test for research agents. The paper is the map, the rubric is the checklist of maneuvers, the clean VM is the empty test track, `reproduce.sh` is the route the agent must actually drive, and the judge is the examiner who evaluates each maneuver and the overall drive.

## 4. Key Insights and Innovations
- Hierarchical, author‑approved rubrics with three complementary requirement types (Sections 2.3–2.4; 3.1)
  - What’s new: Granular, weighted trees that separate “having correct code,” “actually running it,” and “matching results.”
  - Why it matters: Supports partial credit and fine‑grained diagnosis, reflecting the true complexity of research replication (Figure 2; Table 1).

- Separation of task‑time work from reproduction‑time verification (Section 2.2)
  - What’s new: Results only count if the submission’s `reproduce.sh` produces them in a fresh environment.
  - Why it matters: Reduces the risk of hard‑coded or non‑reproducible outputs and establishes a credible replication standard.

- A scalable, validated LLM judge with relevance‑guided context and a ground‑truth benchmark (`JudgeEval`) (Sections 4.1–4.2; Figures 7–9; Table 3)
  - What’s new: Practical automated grading with demonstrated F1≈0.83 at low cost; context management via file ranking; rubric‑aware prompts.
  - Why it matters: Makes large‑scale evaluation feasible; enables regular, repeatable benchmarking as models improve.

- A two‑tier evaluation strategy for accessibility (`PaperBench` vs `PaperBench Code‑Dev`) (Section 2.6)
  - What’s new: A lower‑cost pathway that correlates (weakly) with full performance, useful for early iteration.
  - Why it matters: Lowers barriers for the community to test agent code‑writing capabilities before investing in full reproduction.

- Time‑horizon and scaffold sensitivity insights (Sections 5.2–5.4; Figure 3; Table 5)
  - What’s new: Forcing incremental work via `IterativeAgent` significantly improves some models (e.g., `o1` from 13.2% to 24.4%; Table 5), while others degrade (Claude 3.5 Sonnet falls from 21.0% to 16.1%).
  - Why it matters: Highlights that prompting and runtime management materially affect agent success on long‑horizon research tasks.

## 5. Experimental Analysis
- Evaluation protocol (Sections 5.1–5.2; 2.2–2.4)
  - Setup: Each model runs on all 20 papers, 3 seeds per paper, 12‑hour limit per attempt; submission is re‑executed in a clean VM; judged by `o3-mini` SimpleJudge (Section 5.2).
  - Metric: Average Replication Score (weighted satisfaction of rubric leaves; Section 2.3). Violations of blacklists result in disqualification (10 of 646 runs; Section 2.5).

- Main quantitative results
  - Overall performance with `BasicAgent` (Table 4):
    > Claude 3.5 Sonnet (New) achieves 21.0% ± 0.8; `o1` achieves 13.2% ± 0.3; DeepSeek‑R1 6.0% ± 0.3; GPT‑4o 4.1% ± 0.1; Gemini 2.0 Flash 3.2% ± 0.2; `o3-mini` 2.6% ± 0.2.
  - Effect of `IterativeAgent` (Table 5):
    > `o1` improves to 24.4% ± 0.7 (and to 26.0% ± 0.3 with 36 hours); `o3-mini` improves to 8.5% ± 0.8; Claude 3.5 Sonnet drops to 16.1% ± 0.1.
  - Code‑only variant (Table 6):
    > On `PaperBench Code‑Dev`, `o1` reaches 43.4% ± 0.8.

- Human baseline (Section 5.4; Figure 3)
  - Design: 8 ML PhDs attempted a 4‑paper subset; 3 independent attempts per paper; best‑of‑3 used as “expert” score. Time‑tracked submissions graded at multiple checkpoints (Section 5.4).
  - Result:
    > Humans surpass the agent over time: the `o1` curve plateaus early (≈ first hour) while human scores continue to rise, overtaking by 24–48 hours (Figure 3). On a 3‑paper subset (excluding one truncated run), best‑of‑3 humans achieve 41.4% after 48 hours, vs `o1` at 26.6% on the same subset (Introduction; Section 5.4).

- Where models succeed vs fail (Table 9; stratified analysis)
  - Stratified by requirement type:
    > For `o1 (IterativeAgent)`, Code Development = 43.3% ± 1.1; Execution = 4.5% ± 1.5; Result Match = 0.0% (Table 9). Humans: Code Development = 72.4%; Execution = 20.4%; Result Match = 8.9% (Table 9).
  - Interpretation: Models write lots of code quickly but struggle to integrate, execute, and reach matching results—indicating tool‑use reliability, experiment orchestration, and debugging are the bottlenecks (Section 5.2; Table 9).

- Judge validation and cost‑effectiveness (Table 3; Figure 5)
  - `o3-mini` judge: F1 = 0.83 at ≈ $66/paper; `o1` judge: F1 = 0.84 at ≈ $830/paper. Both are far cheaper than expert human grading, and close enough in F1 for practical use (Table 3; Figure 5).

- Additional observations and checks
  - Cheating safeguards: Disqualifications for blacklist violations (10/646) demonstrate monitoring is necessary (Section 2.5).
  - Sensitivity to scaffolds: Different prompting and the ability to “end early” strongly affect outcomes (Sections 5.2–5.3).
  - Variance across papers and seeds: High variability suggests multiple seeds are advisable for robust evaluation; detailed per‑paper tables are provided (Tables 10–18; Appendix I).

- Overall assessment
  - The experiments convincingly show: (a) non‑trivial capability at code writing; (b) major gaps in long‑horizon execution and reliable reproduction; and (c) meaningful dependence on runtime scaffolding and time limits. The human baseline confirms the remaining performance gap on realistic research workloads (Figure 3; Sections 5.2–5.4).

## 6. Limitations and Trade-offs
- Dataset scope and potential contamination (Section 7)
  - Only 20 papers, albeit with 8,316 leaf requirements. Future models may have pretraining exposure to some code or techniques (“contamination”), although recency mitigates this today.
- Rubric creation cost and complexity (Section 7; Appendix C)
  - Rubrics are labor‑intensive and require expert involvement and iterative review; this constrains scaling. Weight choices, while author‑approved, encode subjective judgments about importance (Section 3.1).
- Automated judge fidelity and determinism (Section 7; Section 4.2)
  - LLM judges are not perfect or deterministic; although `o3-mini` achieves F1 ≈ 0.83 on JudgeEval (Table 3), expert humans remain the gold standard for nuanced cases.
- Compute and cost constraints (Section 7)
  - Full PaperBench runs are expensive: e.g., ≈ $400 in API credits for a 12‑hour `o1 IterativeAgent` rollout per paper, plus ≈ $66 for grading with `o3-mini` (Section 7). GPU requirements (A10) limit accessibility.
- Benchmark rules and realism (Section 2.5)
  - Blacklisting original code ensures from‑scratch replication but differs from how human researchers often work (who do consult official code to save time). The 12‑hour reproduction cap in experiments (Section 2.2) may constrain full‑scale results for some papers.
- Specification gaming risks (Appendix A.3)
  - Any rubric‑based system can be gamed; continued stress‑testing and adversarial submissions will be needed to ensure robustness.

## 7. Implications and Future Directions
- How this work changes the landscape
  - PaperBench reframes “replicating a paper” as a measurable, end‑to‑end agent capability with credible, scalable oversight. It provides a common yardstick for autonomy and engineering competence in ML R&D (Introduction; Section 2).
- Research avenues enabled or suggested
  - Better agent scaffolds for long‑horizon work: The `IterativeAgent` gains hint at the importance of step‑wise planning, tool reliability, and “don’t end early” strategies (Section 5.3; Table 5).
  - Automated rubric creation and critique: Human‑in‑the‑loop workflows, dependency graphs in rubrics, and improved task decomposition could reduce rubric authoring costs (Appendix A.1).
  - Stronger, cheaper judges: Improving judge prompts, adding chain‑of‑thought verification, or agent‑as‑judge designs could raise accuracy while lowering cost; `JudgeEval` is a reusable yardstick (Section 4.2; Appendix A.2).
  - Cost‑reduction strategies: “Pruned rubric grading” shows promise for 10× cheaper grading with minor accuracy loss in a preliminary test (Appendix H; Figure 6).
  - Safety evaluation: PaperBench can serve preparedness and responsible scaling frameworks as a metric of autonomous R&D capability growth (Introduction).
- Practical applications
  - Model evaluation during deployment: Labs and enterprises can track whether new reasoning models meaningfully improve at real research tasks beyond coding snippets.
  - Research operations: Triage which parts of a paper an agent can reliably implement vs which require human oversight; use rubrics to allocate work.
  - Education and training: Use Code‑Dev variant for coursework and bootcamps to teach end‑to‑end ML engineering with structured feedback.

> Representative headline results to remember:
> - Best agent on full PaperBench: Claude 3.5 Sonnet (New) at 21.0% (Table 4).
> - `o1` improves from 13.2% to 24.4% with `IterativeAgent` (Table 5), and to 26.0% with 36 hours.
> - Human best‑of‑3 on a subset: 41.4% after 48 hours; agents plateau early (Figure 3).
> - Judge validation: `o3-mini` F1 = 0.83 at ≈ $66/paper (Table 3).

In sum, PaperBench introduces a rigorous, end‑to‑end testbed for research replication that exposes where current agents excel (rapid code writing) and where they struggle (execution, debugging, and matching results). It provides both a credible baseline of current capability and a clear roadmap for progress on autonomy, evaluation, and research safety.
