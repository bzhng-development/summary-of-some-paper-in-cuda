# How Is ChatGPT’s Behavior Changing over Time?

**ArXiv:** [2307.09009](https://arxiv.org/abs/2307.09009)

## 🎯 Pitch

This paper provides the first in-depth, longitudinal analysis of how GPT-3.5 and GPT-4's behaviors shift over time by systematically evaluating the March and June 2023 API releases across a suite of diverse tasks and a specialized instruction-following benchmark. The study uncovers substantial, sometimes adverse, behavioral drift—most notably a pronounced decline in GPT-4's ability to follow basic instructions, which often leads to decreased task performance. These findings expose critical reproducibility and reliability challenges for real-world applications that depend on LLM services, underscoring the urgent need for continuous monitoring as model updates can silently introduce regressions or unexpected risks.

---

## 1. Executive Summary (2-3 sentences)
This paper systematically measures how two widely used “LLM services,” `GPT‑3.5` and `GPT‑4`, changed between their March 2023 and June 2023 API releases. Across eight practical tasks and a new instruction-following benchmark, it finds substantial and sometimes adverse “behavior drift,” with the clearest pattern being a marked drop in `GPT‑4`’s ability to follow simple instructions—often correlating with performance declines (Figure 1; Sections 3–4).

## 2. Context and Motivation
- Problem addressed
  - Cloud-hosted LLMs are frequently updated, but update timing and content are opaque. The same model name (e.g., `gpt‑4`) can behave differently week to week.
  - This creates two risks:
    - Reproducibility and reliability: downstream pipelines can break when output formats or answers change.
    - Safety and governance: changes in refusal behavior, jailbreak robustness, or opinion answering can alter risk profiles (Sections 1–2).
- Why this matters
  - Real-world systems increasingly depend on LLM services for code generation, knowledge retrieval, and decision support. Even small formatting changes (e.g., extra Markdown fences) can make code non-executable and silently break automation (Figure 9b).
  - Scientific tracking of whether “model updates” constitute improvements, regressions, or trade-offs has been limited (Related Work, p. 2).
- Prior approaches and gaps
  - Benchmarks often compare different models at one time point; few measure the same service longitudinally. Some works find small temporal shifts on standard benchmarks, but largely for classification APIs, not generative LLMs (Related Work, p. 2).
- How this paper positions itself
  - It conducts a controlled, two-snapshot longitudinal study (March vs. June 2023) of `GPT‑4` and `GPT‑3.5` under the same API setup (default system prompt, `temperature=0.1`) across diverse tasks. It also probes instruction fidelity with a purpose-built suite of task-agnostic instructions (Sections 2, 4).
  - It releases prompts, responses, and code to catalyze continuous monitoring (p. 2).

## 3. Technical Approach
This is an empirical monitoring study—two dated snapshots of each service are evaluated on diverse, automatically or manually graded tasks, plus a focused instruction-following suite.

- Services and setup (Section 2)
  - Services: `GPT‑4` and `GPT‑3.5` (March 2023 and June 2023 API versions).
  - Querying: user prompt only (default system prompt), `temperature=0.1` to reduce randomness.
- Tasks and why they were chosen (Figure 1; Section 2)
  - Eight tasks spanning reasoning, safety, opinions, knowledge-intensive multi-hop QA, code generation, medical exams, and abstract visual reasoning.
  - Chosen for practical relevance and objective evaluation.
- Evaluation metrics (Section 2)
  - Task-specific primary metrics:
    - Accuracy (math, USMLE), Exact Match/EM (HotpotQA agent, ARC), Directly Executable (code).
    - Response rate for sensitive/opinion questions (whether the model directly answers).
  - Cross-task auxiliary metrics:
    - `verbosity`: number of generated characters (format stability proxy).
    - `mismatch`: for the same prompt, whether March vs. June final answers differ (1) or not (0), averaged across the dataset. This isolates functional differences from surface text variations.
- Datasets and prompts (Figure 2; Sections 3.1–3.8)
  - Math I: prime vs. composite (1,000 numbers; 500 primes, 500 composites from 1,000–20,000). Uses Chain-of-Thought (`CoT`) prompting—an instruction like “think step by step” to elicit intermediate reasoning (Section 3.1).
  - Math II: count “happy numbers” within small intervals (500 queries). A happy number repeatedly summing squares of digits eventually reaches 1 (Section 3.2).
  - SensitiveQA: 100 sensitive questions that should not be directly answered; manual labels for “direct answer” vs. refusal; also a jailbreak test (`AIM` prompt—an “always intelligent and Machiavellian” roleplay jailbreak, footnote p. 9; Table 3).
  - OpinionQA: 1,506 public opinion poll questions in multiple-choice format; metric is “response rate” (Section 3.4).
  - LangChain HotpotQA Agent: a `ReAct`-style agent that reasons-and-acts via Wikipedia search; expects a rigid “`[action]+text`” format; metric is exact match to ground truth (Section 3.6; Figure 10).
  - Code generation: 50 latest “Easy” LeetCode problems (as of Dec 2022), with Python templates; metric is “Directly Executable” (DE) by the LeetCode judge without post-processing; a secondary analysis strips non-code wrappers to see latent correctness (Section 3.5; Table 4).
  - USMLE: 340 multiple-choice medical exam questions; models are instructed to produce “The answer is (X)”, optionally with `CoT` (Figure 11).
  - Visual reasoning: 467 ARC tasks (input/output colored grids serialized as 2D arrays); metric is exact match (Section 3.8; Figure 12).
- Instruction-following benchmark (Section 4; Figures 13–14)
  - Single instructions:
    - `Extract Answer`: e.g., “Answer yes/no in [brackets]”.
    - `Stop Apologizing`: style constraint to avoid phrases like “sorry” or “as an AI model”.
    - `Writing Constraint`: generate text using words starting/ending with a given character.
    - `Format Text`: e.g., add brackets around each word’s first letter.
  - Composite instructions:
    - Combinations of `add comma`, `capitalize`, `no quotation` applied to sentences from arXiv abstracts.
  - Purpose: isolate instruction fidelity from domain knowledge or reasoning.

Design choices emphasize:
- Objective, automatically checkable metrics wherever possible.
- Step-by-step prompting when reasoning is required.
- A “mismatch” lens to quantify change irrespective of absolute accuracy.

## 4. Key Insights and Innovations
- Quantifying LLM service “drift” as a first-class phenomenon
  - Novelty: Treats hosted LLMs as evolving services and measures how their functional outputs change over time, using a simple but informative `mismatch` metric (Section 2).
  - Significance: Reveals large drifts over just three months, highlighting risks for reproducibility and pipeline stability (Figure 1).
- Linking performance drift to instruction-following drift (Section 4; Figures 13–14)
  - Finding: `GPT‑4`’s instruction fidelity collapses on simple directives from March to June.
    - > “Extract Answer” followed 99.5% → 0.5% (Figure 13a).
    - > “Stop Apologizing” followed 74.0% → 19.0% (Figure 13a).
    - Composite instructions show even larger drops (e.g., `no quotation` + `add comma`: −24.0%, Figure 14a).
  - Impact: This single factor helps explain diverse downstream degradations (e.g., failing CoT prompts in math; adding forbidden text around code).
- Revealing sensitivity to prompt formatting in real pipelines
  - LangChain ReAct agent failures due to format non-compliance (“could not parse LLM Output”) even when the underlying content was correct (Figure 10b). This underscores that small format shifts can nullify task performance in agentic systems.
- Demonstrating the fragility of “code only” generation
  - The “Directly Executable” rate plummets when models add Markdown code fences or comments despite instructions to output code only (Figure 9a–b). Yet, removing non-code text rescues latent correctness (Table 4), showing how format drift—not algorithmic competence—can drive perceived regressions.

These constitute fundamental insights about LLM-as-a-service reliability and evaluation, beyond incremental benchmark gains.

## 5. Experimental Analysis
Evaluation design is consistent across services and timepoints (Section 2), with task-specific metrics and two cross-task drift indicators (`verbosity`, `mismatch`). Below are the headline results and their interpretations.

- Math I: prime vs. composite with `CoT` (Section 3.1; Figure 3; Table 1; Figure 4)
  - Accuracy:
    - `GPT‑4`: 84.0% → 51.1% (Figure 3a).
    - `GPT‑3.5`: 49.6% → 76.2% (Figure 3a).
  - `CoT` efficacy:
    - `GPT‑4`: +24.4% (59.6 → 84.0) in March vs. +0.1% (51.0 → 51.1) in June (Table 1).
    - `GPT‑3.5`: −0.9% in March vs. +15.8% in June (Table 1).
  - Behavior evidence:
    - `GPT‑4` March follows step-by-step reasoning and is verbose (avg 638.3 chars) and correct on example 17077; June ignores the “think step by step” instruction and replies simply “[No]” (Figure 3b).
    - Confusion matrices show `GPT‑4` June predicts “composite” almost always (49.9% + 48.8% = 99.7%, Figure 4c).
  - Takeaway: The same `CoT` prompt can swing from highly beneficial to irrelevant depending on service version.

- Math II: counting happy numbers with `CoT` (Section 3.2; Figure 5; Table 2; Figure 6)
  - Accuracy:
    - `GPT‑4`: 83.6% → 35.2% (Figure 5a).
    - `GPT‑3.5`: 30.6% → 48.2% (Figure 5a).
  - `CoT` efficacy:
    - `GPT‑4`: +56.6% (March) vs. +3.2% (June) (Table 2).
    - `GPT‑3.5`: −1.6% (March) vs. +20.6% (June) (Table 2).
  - Bias patterns:
    - `GPT‑4` June tends to answer 0 or 1 happy numbers for nearly all intervals (Figure 6c).
    - `GPT‑3.5` June overestimates (answers >4 even when 4 is the upper bound; Figure 6b).
  - Drift magnitude: Answers differ March→June on 67.6% (`GPT‑4`) and 77.2% (`GPT‑3.5`) of queries (Figure 5a, “Mismatch”).

- Sensitive questions and jailbreaking (Section 3.3; Figure 7; Table 3)
  - Plain-text response rate (direct answers):
    - `GPT‑4`: 21.0% → 5.0% (Figure 7a) with much shorter refusals (verbosity 652.4 → 141.4).
    - `GPT‑3.5`: 2.0% → 8.0% (Figure 7a).
  - Jailbreak (`AIM`) response rate:
    - `GPT‑4`: 78.0% → 31.0% (Table 3), indicating stronger defense in June.
    - `GPT‑3.5`: 100.0% → 96.0% (Table 3), remaining highly vulnerable.
  - Qualitative shift: Refusals lose rationale/explanation; June often outputs “Sorry, but I can’t assist with that” (Figure 7b).

- OpinionQA (Section 3.4; Figure 8)
  - Response rate:
    - `GPT‑4`: 97.6% → 22.1% (Figure 8a), frequently refusing on the grounds that questions are “subjective” (Figure 8b).
    - `GPT‑3.5`: 94.3% → 96.7% (Figure 8a).
  - Opinion drift: For `GPT‑3.5`, 27.5% of answers differ across versions (Figure 8a, “Mismatch”).

- Code generation (Section 3.5; Figure 9; Table 4)
  - Directly Executable (no post-processing):
    - `GPT‑4`: 52.0% → 10.0% (Figure 9a).
    - `GPT‑3.5`: 22.0% → 2.0% (Figure 9a).
  - Root cause: extra non-code text (e.g., ```python fences) violates “code only” instruction (Figure 9b).
  - Latent correctness (after stripping non-code):
    - `GPT‑4`: 52.0% → 70.0% (+60.0 points from raw) (Table 4).
    - `GPT‑3.5`: 46.0% (March) and 48.0% (June) after cleaning (Table 4).
  - Interpretation: Apparent performance regressions largely stem from formatting drift/instruction non-compliance.

- LangChain HotpotQA Agent (Section 3.6; Figure 10)
  - Exact Match:
    - `GPT‑4`: 1.2% → 37.8% (Figure 10a).
    - `GPT‑3.5`: 22.8% → 14.0% (Figure 10a).
  - Format sensitivity: March `GPT‑4` produced correct content but not the exact `[action]+text` format the agent expects, so the agent “could not parse LLM Output” (Figure 10b).
  - Drift magnitude: >80% of final answers change across versions for both models (Figure 10a, “Mismatch”).

- USMLE medical exam (Section 3.7; Figure 11)
  - Accuracy:
    - `GPT‑4`: 86.6% → 82.1% (Figure 11a).
    - `GPT‑3.5`: ~54.3% → ~54.7% (Figure 11a), roughly flat.
  - Drift magnitude: `GPT‑4` answers differ on 12.2% of questions across timepoints; `GPT‑3.5` differs on 27.9% (Figure 11a, “Mismatch”).
  - Qualitative: `GPT‑3.5` June often uses longer reasoning yet can land on incorrect options (Figure 11b).

- Visual reasoning (ARC) (Section 3.8; Figure 12)
  - Exact Match:
    - `GPT‑4`: 24.6% → 27.2% (Figure 12a).
    - `GPT‑3.5`: 10.9% → 14.3% (Figure 12a).
  - Stability vs. reported mismatch:
    - The narrative states “more than 90%” of generations are identical across versions, yet the “Mismatch” bars in Figure 12a are high (64.5% `GPT‑4`, 77.1% `GPT‑3.5`). This discrepancy suggests caution in interpreting stability on ARC and warrants replication.
  - Case example: a problem solved in March but failed in June (Figure 12b), showing that improvements are not monotonic per-instance.

- Instruction following (Section 4; Figures 13–14)
  - Single-instruction fidelity drops sharply for `GPT‑4`:
    - > `Extract Answer`: 99.5% → 0.5% (Figure 13a).
    - > `Stop Apologizing`: 74.0% → 19.0% (Figure 13a).
    - Qualitative errors: capitalization in brackets when asked not to, continuing to say “sorry” despite constraints, missing required brackets (Figure 13b).
  - Composite-instruction fidelity degrades further:
    - Example: `add comma` + `capitalize` drops −9.2% from March to June; `no quotation` + `add comma` drops −24.0% (Figure 14a).
    - Qualitative failure: adding commas to every character rather than every word (Figure 14b, June example).

Overall assessment
- The experiments convincingly demonstrate substantial behavior drift within short intervals for hosted LLMs, with strong, triangulated evidence that loss of instruction fidelity is a common driver of regressions. The breadth of tasks and the use of objective, automatable metrics strengthen the case. Two caveats: ARC stability reporting appears inconsistent (Section 3.8), and causality (why the services changed) cannot be established from black-box observations.

## 6. Limitations and Trade-offs
- Scope and causality
  - Only two snapshots (March vs. June 2023). Results capture a slice in time and cannot isolate the internal cause of drift (data changes, alignment tweaks, safety layers, decoding defaults).
- Black-box dependence
  - The study uses the default system prompt and `temperature=0.1`. Small, undocumented provider-side changes (e.g., prompt templates, safety middleware) could affect behavior independently of the base model.
- Evaluation choices
  - `mismatch` counts answer changes regardless of correctness; useful for drift size but not for judging improvement.
  - Manual labeling for SensitiveQA “direct answer” introduces human judgment (mitigated by simple labeling criterion; Section 3.3).
  - Agent experiments depend on specific LangChain prompt formats; results may shift with alternative agent designs or stricter output validation.
  - ARC representation (2D arrays) and EM may under-credit partially correct visual transformations.
- Generalizability
  - Tasks, while diverse, are still a subset; complex real-world workloads may show different drift profiles.
- Potential confounders
  - The striking code regressions demonstrate that format instruction-following—not algorithmic code quality—drove much of the DE drop. This is both a strength (diagnosis) and a cautionary tale about interpreting raw pass rates without format checks.

## 7. Implications and Future Directions
- For practitioners and platform integrators
  - Treat LLMs as evolving services. Implement continuous evaluations, “canary” tests, and contract tests that verify both content and format before deploying an updated model.
  - Harden interfaces:
    - Enforce strict output schemas (e.g., JSON with validators).
    - Use programmatic post-processing to strip or normalize formatting (Table 4 shows large gains after removing non-code text).
    - Add self-check prompts that verify instruction compliance before finalizing an answer.
  - Be cautious with `CoT`: its benefit varies by version (Tables 1–2). Consider fallback strategies (with/without `CoT`) and meta-prompts that adapt when the model ignores reasoning instructions.
- For safety and governance
  - Expect shifts in refusal behavior and jailbreak robustness. The June `GPT‑4` is stricter on sensitive content and more robust to the AIM jailbreak (Table 3), while `GPT‑3.5` remains permissive. Periodically reassess risk profiles.
- For research
  - Standardize longitudinal benchmarks and telemetry for LLM services, including instruction-following batteries and schema adherence metrics.
  - Study mechanisms behind instruction fidelity degradation and ways to stabilize it (e.g., constrained decoding, function calling, structured outputs).
  - Develop agent frameworks resilient to format variance (e.g., robust parsing, autorepair strategies).
- For model providers
  - Provide explicit versioning and changelogs; expose alignment/safety layers as configurable policies separate from the base model; offer “pinned” models for production reproducibility.
- Broader takeaway
  - This work reframes progress from “a better single score” to “reliable, predictable behavior over time.” The evidence—spanning Figures 3–12 and Tables 1–4—shows that without continuous monitoring and clearer service guarantees, even strong models can regress on critical behaviors like instruction following and formatting, with outsized impact on real systems.
