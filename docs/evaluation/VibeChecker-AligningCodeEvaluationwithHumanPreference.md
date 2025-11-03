# Vibe Checker: Aligning Code Evaluation with Human Preference

**ArXiv:** [2510.07315](https://arxiv.org/abs/2510.07315)

## 🎯 Pitch

This paper introduces VeriCode, a taxonomy of 30 verifiable non-functional code instructions, and Vibe Checker, a testbed that augments existing code benchmarks to measure both functional correctness and real-world instruction following. By showing that a blend of functional and instruction-following metrics aligns much more closely with actual human preference than traditional pass@k scores, the authors provide a compelling new framework for evaluating and training code-generating models to meet the nuanced expectations of real users. This has major implications for the development and assessment of AI coding assistants, ensuring they produce code that not only works but truly 'feels right' to human collaborators.

---

## 1. Executive Summary
This paper introduces two artifacts that make code evaluation reflect human preference, not just test passing: (1) VeriCode — a taxonomy of 30 verifiable, non‑functional “code instructions” (e.g., style, documentation, error handling) each paired with a deterministic checker; and (2) Vibe Checker — a testbed that augments mainstream code benchmarks with these instructions to evaluate both functional correctness and instruction following (IF). Evaluating 31 leading LLMs, the paper shows that adding such instructions significantly reduces pass@1 and that a composite of functionality and IF correlates best with real human preference (Figure 5, §4.5).

## 2. Context and Motivation
- Problem/gap
  - Code LLMs are increasingly used in “vibe coding”: users iterate with an AI partner until the solution “feels right” (“vibe check”). A vibe check includes not only functional correctness but also adherence to non‑functional expectations (style, minimal edits, docstrings, library choices) that developers routinely enforce in real projects (Introduction, p. 1–2; Figure 1).
  - Current evaluation focuses on pass@k, which captures only functional correctness and misses non‑functional preferences (§1, p. 1–2).

- Why it matters
  - Practical impact: In tools like Copilot or Cursor, users routinely reject functionally correct answers that violate style, complexity, or documentation constraints. The Copilot Arena ranking’s weak/negative correlation with functional benchmarks underscores this mismatch (Introduction, p. 2).
  - Scientific impact: RL training in code often uses verifiable rewards tied to unit tests (pass@k). Optimizing only this signal yields models that score well on benchmarks but fail many real user vibe checks (Introduction, p. 2).

- Prior approaches and shortcomings
  - General instruction following has synthetic, verifiable checks (e.g., forced wording) or LLM-as-a-judge (§5). In coding, existing non‑functional evaluation either lacks verifiable signals or relies on subjective/hard‑to‑scale judgments (§5).
  - Static linters exist, but there was no curated, verifiable set of non‑functional code instructions with corresponding deterministic verifiers that can be combined with standard benchmarks and unit tests (§2).

- Positioning
  - The paper reframes “instruction following” as the missing measurable component of vibe checks in code, alongside functionality. It contributes: a verifiable taxonomy (VeriCode), an augmented testbed (Vibe Checker), and evidence that human preference aligns best with a mix of IF and functionality (§§2–4).

## 3. Technical Approach
The work has two pillars: VeriCode (a taxonomy + verifiers) and Vibe Checker (benchmark augmentation + protocol + metrics).

- Terminology used throughout
  - `vibe coding`: iterative code development with an AI partner.
  - `vibe check`: the user’s accept/reject decision based on overall “fit,” not only correctness (Figure 1).
  - `instruction following (IF)`: whether generated code satisfies explicit non‑functional constraints.
  - `pass@k`: standard metric for functional correctness; whether any of k attempts pass tests.
  - `linter`: a static analysis tool that detects style and certain structural issues. The paper mainly uses `Ruff`, a Python linter that aggregates rules from popular tools (footnote 3).
  - `AST` (Abstract Syntax Tree): a structural representation of code used for deterministic checks.

A. VeriCode: building a verifiable instruction taxonomy (§2)
- Design principles (§2.1)
  - Verifiability: each instruction has a deterministic pass/fail checker.
  - Practice grounding: instructions reflect real developer expectations (style guides, linter rules).
  - Comprehensive coverage: style, logic patterns, docs, error handling, and library/API constraints.
  - Difficulty: each instruction challenges recent advanced LLMs; trivial ones are filtered out.

- Construction pipeline (§2.2)
  1) Candidate sourcing: start with >800 Ruff rules; add response‑level documentation instructions that linters alone cannot cover.
  2) Scope/relevance filtering: consolidate overlapping rules; keep broadly applicable ones.
  3) Difficulty filtering: run Gemini 2.5 Flash on BigCodeBench‑Hard; remove instructions that are too easy (success >90% with no functional degradation).
  4) Expert review + verifier implementation: prefer linter‑backed checks; write AST/regex checkers where no rule exists. All verifiers return binary pass/fail and share a common interface (Appendix B.1 shows the Ruff helper in Figure 6).

- Resulting taxonomy (§2.3)
  - 30 instructions across five categories: Coding Style & Conventions (9), Logic & Code Patterns (9), Documentation & Commenting (6), Error Handling & Exception Management (4), and Library & API Constraints (2).
  - Each instruction has: category, description, distinct prompts for single‑turn vs multi‑turn use, parameters with recommended ranges, and verification code (schema in §2.3; examples in Table 1 and full cases in Figures 7–11).
  - Parameterization as a difficulty dial: e.g., `line_length`, `max_branches`, or docstring `convention` (Table 1). This makes the 30 “core” instructions expandable into hundreds of concrete, checkable variants.

B. Vibe Checker: augmenting benchmarks and defining the protocol (§3)
- Benchmarks (§3.1)
  - BigVibeBench: BigCodeBench augmented with VeriCode instructions (real‑world programming).
  - LiveVibeBench: LiveCodeBench augmented similarly (algorithmic/contest tasks).
  - Category distributions show Logic/Style/Docs dominate; LiveVibeBench uses more Logic constraints; BigVibeBench has more Error and Library constraints (Appendix Figure 12).

- Augmentation pipeline (§3.1)
  1) Instruction selection: for each base problem, permute the 30 taxonomy instructions; an LLM selector scans the list and keeps only those that (a) are relevant to the task and (b) do not conflict with already chosen instructions. The kept instructions, in the scanned order, form the constraint set.
  2) Parameter selection + validation: an LLM proposes parameters for each selected instruction, guided by the instruction’s supported keys/ranges and the problem context; a rule‑based validator drops unsupported keys and reverts invalid values to defaults.
  3) Selector choice: Gemini 2.5 Pro and Claude 4 Opus produce similar category distributions; the final benchmark uses Claude 4 Opus due to a lower invalid‑parameter rate (0.96% vs 2.47%).

- Evaluation protocol (§3.2; Figure 2)
  - Settings
    - Single‑Turn Generation: all instructions appear once after the original query; the model returns a single implementation.
    - Multi‑Turn Editing: first generate a base solution; then reveal instructions one‑by‑one across turns; the model edits the code each round; the final code is evaluated.
  - Metrics
    - Functionality: pass@1 against unit tests; report functional regression `FR_k = (S0 – S_k)/S0`, where `S0` is base pass@1 and `S_k` is pass@1 with `k` instructions (equation in §3.2).
    - Instruction following:
      - Instruction‑level: average fraction of passed instruction verifiers.
      - Task‑level: all instructions must pass for a score of 1; otherwise 0 (equations in §3.2).

C. Experimental setup (§4.1)
- 31 LLMs across 10 families (Appendix Table 4 lists the exact models and their LMArena Elo).
- Data: 1,140 BigCodeBench and 1,055 LiveCodeBench tasks; each is augmented with five instructions, yielding >10K instruction‑level evaluations.
- Inference details: temperatures follow underlying benchmarks; thinking‑mode enabled where supported; max context 32,768 tokens; API providers Vertex AI and OpenRouter (§4.1).
- Some models with >10% response failures on LiveVibeBench are excluded from those analyses (Appendix D.1).

## 4. Key Insights and Innovations
1) A verifiable, parameterized taxonomy of non‑functional code instructions (VeriCode) (§2)
- What’s new: distills hundreds of linter/style rules into 30 broadly applicable, automatically checkable instructions with deterministic verifiers, many backed by Ruff. Includes response‑level doc checks that linters alone miss (Table 1; Figures 7–11).
- Why it matters: enables scalable, objective measurement (and potential training rewards) for non‑functional aspects that drive human preference but were previously hard to evaluate at scale.

2) A unified testbed (Vibe Checker) that couples unit tests with instruction verifiers (§3)
- What’s new: augments mainstream benchmarks with relevant, non‑conflicting instruction sets and evaluates models in single‑turn and multi‑turn modes (Figure 2).
- Why it matters: reveals trade‑offs that standard pass@k obscures (e.g., higher IF but lower functionality in multi‑turn), and supports realistic interaction patterns.

3) Empirical evidence that adding non‑functional instructions hurts functional correctness (§4.2)
- Novel observation: even though instructions do not target functionality, pass@1 regresses consistently. This is quantified across many models and tasks (Table 2; Figure 3a).

4) Instruction following is the primary differentiator among strong models and correlates with human preference (§4.5)
- Core finding: a weighted combination of IF and functionality best matches LMArena coding Elo; IF gets substantial weight, especially for real‑world tasks (Figure 5). This reframes evaluation/training priorities.

5) Behavioral analyses: position bias and single‑ vs multi‑turn trade‑offs (§4.3–§4.4)
- New diagnostics: lost‑in‑the‑middle pattern at instruction positions (Figure 4) and systematic trade‑off where single‑turn preserves functionality better, while multi‑turn yields higher IF (Figure 3b).

Overall, items (1)–(2) are infrastructure contributions; (3)–(5) are substantive empirical insights about LLM coding behavior.

## 5. Experimental Analysis
- Evaluation methodology
  - Datasets/benchmarks: BigVibeBench and LiveVibeBench — augmented versions of BigCodeBench (real‑world programming) and LiveCodeBench (algorithmic/contest) (§3.1).
  - Metrics: pass@1, `FR_k`, instruction‑level IF, and task‑level IF (§3.2).
  - Setup: 31 LLMs, five instructions per task, both single‑turn and multi‑turn settings (§4.1).

- Main quantitative results (selected highlights)
  - Functional regression (Table 2; Figure 3a)
    - Trend: Regression increases with the number of instructions and is worse in multi‑turn.
    - Example (BigVibeBench, multi‑turn, 5 instr.): most models incur >5% regression; e.g., `o4 mini` +8.05%, `Kimi K2` +6.12% (Table 2).
    - Example (LiveVibeBench, single‑turn, 5 instr.): strong regressions such as `o4 mini` +12.29% and `Kimi K2` +16.36% (Table 2).
    - Aggregate: “average pass@1 drops by 5.85% and 6.61% under five instructions” on BigVibeBench and LiveVibeBench respectively (summary bullets in §4).
    - Single‑turn vs multi‑turn: Single‑turn preserves functionality better; the gap grows with more constraints (Figure 3a).

  - Instruction following (Table 3; Figure 3b)
    - Task‑level IF (all constraints must pass) drops rapidly as constraints increase. With 5 instructions (single‑turn):
      - BigVibeBench: best model reaches only 46.75% (`Claude 4 Opus`), and many strong models are in the 30–41% range (Table 3).
      - LiveVibeBench: best among listed leaders is 40.95% (`GPT 5`), with several models below 30% (Table 3).
      - Quote: > “Even the best performing model reaches only 46.75% and 40.95% success rate under five instructions” (Table 3, §4.3).
    - Multi‑turn vs single‑turn: Multi‑turn improves task‑level IF by ~3–4.5% on BigVibeBench and ~8% on LiveVibeBench (Figure 3b), but at a functionality cost (Figure 3a).

  - Position bias (§4.4; Figure 4)
    - BigVibeBench shows a U‑shape across positions: mid‑list instructions are followed less reliably (lost‑in‑the‑middle). Single‑turn favors the first instruction (primacy bias), multi‑turn favors the last (recency bias).
    - LiveVibeBench lacks a strong U‑shape but keeps the primacy/recency asymmetry.

  - Human preference correlation (§4.5; Figure 5)
    - Composite score `α·IF + (1–α)·Func` correlates best with LMArena coding Elo.
      - BigVibeBench: Pearson best at α=0.4; Spearman best at α=0.7.
      - LiveVibeBench: Pearson best at α=0.4; Spearman best at α=0.6.
      - Quote: > “The peak correlation … is achieved with a mixture of the two metrics” (Figure 5).
    - Single‑metric comparison: For real‑world programming, pure IF correlates substantially better than pure functionality on rank correlation; for algorithmic tasks, functionality alone fares better than IF (Figure 5b discussion).

- Supporting analyses and implementation detail
  - Instruction‑level IF scores for all models/settings (Appendix Tables 7–10); per‑position IF (Appendix Tables 11–12).
  - System/evaluation prompts used (Appendix Figures 13–14).
  - Verifier implementation detail: Ruff helper (Appendix Figure 6) and representative instructions with code (Figures 7–11).

- Do the experiments support the claims?
  - Yes, on three counts:
    - Measurability: deterministic verifiers provide objective IF signals across >10K checks.
    - Behavioral regularities: monotonic regression with more constraints; multi‑turn IF gains vs functionality losses; positional biases — all shown across many models and two benchmarks (Figure 3; Figure 4; Tables 2–3).
    - Preference alignment: composite metric consistently outperforms either IF or functionality alone (Figure 5, Appendix E.2).

- Failure modes/robustness
  - Some models exhibit high error rates on LiveVibeBench (Appendix D.1), prompting exclusions.
  - Certain instruction categories are more frequent than others (Figure 12), which may influence aggregate difficulty.
  - The paper notes parameter validity checks and selector comparisons (Claude vs Gemini) to mitigate augmentation artifacts (§3.1).

- Conditions and trade‑offs
  - As the number of constraints rises, task‑level IF decays multiplicatively (satisfy-all requirement), leading to steep drops (Table 3, §4.3).
  - Single‑turn prioritizes global correctness; multi‑turn enables targeted edits but risks regressions (§4.3).

## 6. Limitations and Trade-offs
- Assumptions and scope
  - Language focus: current VeriCode and verifiers target Python, “the dominant language in code evaluation” (§2.3). Although the framework is language‑agnostic in principle, results may not immediately transfer to other ecosystems without equivalent lint/AST infrastructure.
  - Instruction source: heavy reliance on Ruff rules; this favors checks that static analysis can capture. Deep semantic or project‑specific conventions may require custom verifiers and could be underrepresented.

- Potential gaps
  - Verifier coverage: some non‑functional preferences (e.g., “minimal edits,” intent preservation across refactors) are difficult to verify deterministically and may be only partially captured (Introduction; §3.2 uses binary pass/fail per instruction).
  - Benchmark context: tasks are single‑file or function‑level; repository‑level constraints, cross‑file style consistency, and CI‑like pipelines are out of scope.
  - Selection bias: an LLM chooses “relevant and non‑conflicting” instructions (§3.1). While validated, selector biases might shape the distribution and difficulty of constraints (Appendix Figure 12).

- Computational/engineering trade‑offs
  - Multi‑turn editing incurs more tokens and turns, raising latency/cost and increasing regression risk (Figure 3a).
  - Strict task‑level IF (all‑pass) yields interpretable scores but can mask partial adherence improvements.

- Open questions
  - How to design verifiers for performance, memory use, or security properties that require dynamic analysis?
  - How to reward “minimal diff” edits or preservation of prior intent across long interaction histories?

## 7. Implications and Future Directions
- How this changes the field
  - Establishes instruction following — beyond test passing — as a measurable, primary dimension of code quality that aligns closely with user preference (Figure 5).
  - Provides infrastructure (VeriCode + Vibe Checker) to evaluate and train models on non‑functional constraints at scale.

- Research opportunities
  - Training: integrate IF verifiers as reward signals in SFT/RLVR, combining them with unit tests to optimize a composite objective. The paper’s correlation analysis (Figure 5) suggests non‑trivial weights on IF (up to 0.7 for rank correlation on real‑world tasks).
  - Coverage expansion: extend taxonomy to other languages (TypeScript, Java, Rust) by mapping to language‑specific linters and adding response‑level verifiers where needed (§2.3).
  - Richer constraints: add verifiable rules for performance, security, maintainability, and “minimal edits,” possibly combining static, dynamic, and differential analyses.
  - Interaction design: mitigate position bias (Figure 4) via prompt structuring, chunking, or curriculum‑style instruction revealing; explore planner/editor roles in agents for higher IF with less regression.

- Practical applications
  - IDE assistants that enforce organization or team style guides, docstring conventions, or API constraints out‑of‑the‑box.
  - Continuous evaluation dashboards for model providers that report both functionality and IF, better predicting end‑user satisfaction.
  - RL‑trained coding agents that manage multi‑turn editing while preserving functionality, guided by verifiable non‑functional rewards.

> Overall takeaway: “Human preference reflects a mixture of functional correctness and instruction following,” with IF “emerging as the key differentiator among advanced models on real‑world programming tasks” (Figure 5; §4.5). The paper provides both the measurement tools (VeriCode, Vibe Checker) and the empirical evidence to center instruction following in future code LLM evaluation and training.
