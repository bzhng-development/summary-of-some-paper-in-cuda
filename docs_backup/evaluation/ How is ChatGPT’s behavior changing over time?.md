# How is ChatGPT’s behavior changing over time?

**ArXiv:** [2307.09009](https://arxiv.org/abs/2307.09009)
**Authors:** Lingjiao Chen, Matei Zaharia, James Zou
**Institutions:** Stanford University, University of California, Berkeley

## 🎯 Pitch

This study reveals significant, task-dependent shifts in the performance of GPT-3.5 and GPT-4 over just three months, highlighting a notable decline in instruction-following as a key driver of these changes. By exposing the instability of LLMs-as-a-service, this research underscores the importance of monitoring not just accuracy but also instruction fidelity and formatting, offering practical insights to improve model integration reliability in real-world applications.

---

## 1. Executive Summary
This paper measures how the “same” commercial large language model service (GPT‑3.5 and GPT‑4) changes over time by comparing its March 2023 and June 2023 versions across eight diverse tasks. It finds large, task‑dependent drifts—some improvements, many regressions—and presents evidence that a decline in instruction‑following (the ability to comply with user formatting and behavioral constraints) is a key driver of several drops in performance.

## 2. Context and Motivation
- Problem addressed
  - Commercial LLM services are periodically updated, but users are not told what changed or when. This makes integration brittle: a prompt that worked yesterday may silently stop working today.
  - The study asks: How much does behavior actually drift in practice, and why?

- Why this matters
  - Real‑world impact: Workflows can break if accuracy, formatting, or refusal behavior changes. Reproducibility suffers if the same prompt produces a different answer a month later. The paper documents such breakages, e.g., code that stopped being “directly executable” and an agent pipeline that failed to parse replies (Figures 9 and 10).
  - Theoretical significance: It challenges the assumption that model updates monotonically improve performance. It also probes the link between “instruction following” and downstream task success.

- Prior approaches and gaps
  - Existing evaluations mostly benchmark single snapshots of LLMs or track small longitudinal shifts on limited benchmarks. For example, ChatLog reports mostly small (<5%) shifts; other works analyze specific tasks or classic ML APIs (Related Work, p. 2).
  - Gap: No systematic, multi‑task longitudinal analysis of model‑as‑a‑service behavior on generative tasks with objective scoring and operational failure modes (formatting, refusal, jailbreak vulnerability).

- Positioning
  - This work monitors two timepoints (March vs. June 2023) of GPT‑3.5 and GPT‑4 across eight tasks designed to be objective and practical, then isolates a potential root cause—reduced instruction fidelity—through targeted probes (Figures 1 and 13–14).

## 3. Technical Approach
This is an empirical study with two main components: (A) longitudinal task evaluations and (B) targeted instruction‑following tests.

- Services and setup
  - Models: `GPT‑4` and `GPT‑3.5` accessed via OpenAI’s API at two snapshots: March 2023 and June 2023 (Section “LLM Services,” p. 3).
  - Configuration: default system prompt; only the user prompt is varied. Temperature set to 0.1 to reduce randomness (p. 3).

- Common auxiliary metrics (p. 3–4)
  - `verbosity`: number of characters in the generation (a proxy for terseness/verbosity).
  - `mismatch`: for the same prompt, whether the extracted final answers differ between the two versions of the same model. Note: this compares task answers, not raw text.

- Eight evaluation tasks (overview in Figure 1 and detailed in Section 3)
  1) Math I: Prime vs. composite (Section 3.1; Figure 3)
     - 1,000 questions: 500 primes (from [ZPM+23]) + 500 composites sampled uniformly from 1,000–20,000.
     - Prompting uses Chain‑of‑Thought (CoT) to encourage stepwise reasoning.
     - Metric: accuracy on Yes/No primality.
     - Additional probe: repeat without CoT to compute the “CoT boost” (Table 1).
     - Definition: `Chain‑of‑Thought (CoT)` is a prompt strategy that requests intermediate reasoning steps before the final answer.

  2) Math II: Counting happy numbers (Section 3.2; Figure 5; Table 2; Figure 6)
     - 500 queries. Each asks how many “happy numbers” lie in an interval of length 6–10, start in [500, 15,000].
     - Definition: a “happy number” reaches 1 when repeatedly replaced by the sum of the squares of its digits; otherwise it falls into a cycle.
     - Prompting uses CoT; also tested without CoT (Table 2).
     - Metric: exact match on the count; confusion matrices examine error patterns (Figure 6).

  3) Sensitive questions (Section 3.3; Figure 7; Table 3)
     - 100 queries that should not receive direct answers (e.g., discriminatory or illegal requests).
     - Metric: `response rate` = fraction of prompts that receive a direct answer (higher can mean less safe).
     - Manual labeling to judge whether a response is a direct answer.
     - Jailbreak test: AIM (“Always Intelligent and Machiavellian”) jailbreak prompt applied to each query (Table 3).
     - Definition: `jailbreaking` is prompt manipulation designed to bypass safety constraints.

  4) OpinionQA survey (Section 3.4; Figure 8)
     - 1,506 multiple‑choice questions from public opinion polls (OpinionQA).
     - Metric: `response rate` (whether the model picks an option); also analyze disagreement across time to quantify opinion drift.

  5) LangChain HotpotQA agent (Section 3.6; Figure 10)
     - A multi‑hop QA pipeline using LangChain’s default ReAct agent to search Wikipedia and answer HotpotQA‑style questions.
     - Definition: `ReAct` combines reasoning (“Thought”) and acting (“Action: search/look up”) as alternating steps.
     - Metric: exact match of final answers.
     - Critical detail: the agent expects strict output formats (“[action]+text”). Deviations break parsing (Figure 10b).

  6) Code generation and formatting (Section 3.5; Figure 9; Table 4)
     - 50 newest “easy” LeetCode problems (to reduce training‑data contamination concerns).
     - Prompt concatenates the original problem text and a Python template; the model is told to output code only.
     - `Directly executable`: the code is accepted by LeetCode’s online judge without any post‑processing.
     - Also re‑evaluate after stripping non‑code wrappers (Table 4) to check whether failures were due to formatting, not logic.

  7) USMLE medical exam (Section 3.7; Figure 11)
     - 340 multiple‑choice questions from USMLE‑style exams; CoT prompting that asks for “The answer is (X)”.
     - Metric: accuracy.

  8) Visual reasoning (ARC) (Section 3.8; Figure 12)
     - 467 ARC grid tasks formatted as 2‑D arrays; task: infer transformation from examples and produce the output grid.
     - Metric: exact match; also track whether generations are identical across March/June snapshots.

- Instruction‑following probes (Section 4; Figures 13–14)
  - Task-agnostic tests constructed to isolate “instruction fidelity.”
  - Four single‑instruction families (Figure 13):
    - `Extract Answer`: e.g., “Answer yes/no in [square brackets]”.
    - `Stop apologizing`: e.g., “Do not say ‘sorry’ or ‘as an AI model’.”
    - `Writing constraint`: e.g., “Describe X using only words ending with letter Y.”
    - `Format text`: e.g., “Put [ ] around the first letter of each word (including articles).”
  - Composite instructions (Figure 14):
    - Three simple text‑formatting instructions applied alone and in all pairs: `add comma` to each word, `no quotation` (remove quotes), `capitalize` (convert to uppercase). Evaluate performance drop from single to composite.

Design choice rationale
- Tasks are “objective” where possible: correctness, executability, exact match, or response rate are easy to score (p. 3).
- Two auxiliary metrics—verbosity and mismatch—capture behavioral shifts beyond accuracy (p. 3–4).
- Using LangChain’s ReAct agent and the LeetCode judge surfaces realistic pipeline brittleness: small changes in format can crash an agent or render code non‑executable (Figures 9–10).
- CoT vs. no‑CoT ablations test whether reasoning‑style prompting still helps after an update (Tables 1–2).

## 4. Key Insights and Innovations
- Large service‑level drift is real and multifaceted (Figure 1a)
  - “The same” API changed substantially in two months, sometimes by tens of percentage points on accuracy or executability. This extends prior smaller‑scale drift observations by documenting stronger, task‑dependent shifts across math, code, safety, survey response, multi‑hop QA, medical QA, and visual reasoning.

- Instruction‑following degradation in GPT‑4 is a plausible unifying factor (Figures 1b, 13–14)
  - Novel evidence: on task‑agnostic instruction probes, GPT‑4’s compliance collapses from March to June:
    - Extract‑Answer compliance: 99.5% → 0.5% (Figure 13a).
    - “Stop apologizing” compliance on sensitive prompts: 74% → 19% (Figure 13a).
    - Composite text‑format instructions show especially large drops, e.g., add‑comma + no‑quotation: −24.0 percentage points (Figure 14a).
  - Significance: Many regressions elsewhere are consistent with reduced instruction fidelity—e.g., failing to output “code only” (Figure 9b), ignoring CoT steps (Figures 3b and 5b), or not adhering to LangChain’s required format (Figure 10b).

- CoT no longer reliably helps GPT‑4 after the update (Tables 1–2)
  - For primality, the CoT boost shrinks from +24.4% (March) to +0.1% (June) (Table 1).
  - For happy numbers, the boost shrinks from +56.6% to +3.2% (Table 2).
  - This is not just weaker reasoning—it is often refusal to produce steps at all (Figures 3b, 5b).

- Pipeline brittleness highlighted by realistic failure modes
  - Code: wrapping snippets in Markdown code fences in June rendered outputs not “directly executable,” dropping GPT‑4’s pass rate from 52% to 10% (Figure 9a), even though the underlying logic often improved when wrappers were stripped (to 70%; Table 4).
  - Agents: a small change in response format caused LangChain to fail to parse outputs (“Could not parse LLM Output”) despite semantically correct content (Figure 10b).

These are fundamental observations about model‑as‑a‑service reliability rather than incremental score bumps.

## 5. Experimental Analysis
Evaluation methodology (Section 2, Figure 1)
- Services: GPT‑3.5 and GPT‑4, March vs. June 2023 snapshots.
- Setup: uniform temperature (0.1), default system prompt, diverse tasks with clear metrics, and two additional behavior metrics (verbosity, mismatch).

Headline quantitative results by task
- Math I: Prime vs. composite (Figure 3; Table 1)
  - Accuracy
    - GPT‑4: 84.0% → 51.1% (−32.9 points).
    - GPT‑3.5: 49.6% → 76.2% (+26.6 points).
  - Verbosity
    - GPT‑4: 638.3 → 3.9 characters (very terse).
    - GPT‑3.5: increased by ~22% (Figure 3a).
  - CoT vs. no‑CoT (Table 1)
    - GPT‑4: CoT boost +24.4% (March) → +0.1% (June).
    - GPT‑3.5: −0.9% (March) → +15.8% (June).
  - Error patterns (Figure 4)
    - GPT‑4 June exhibits a strong bias toward predicting “composite” almost always.
  - Mechanism insight: GPT‑4 June often ignores “think step by step” and outputs only “[No]” (Figure 3b).

- Math II: Counting happy numbers (Figure 5; Table 2; Figure 6)
  - Accuracy
    - GPT‑4: 83.6% → 35.2% (−48.4 points).
    - GPT‑3.5: 30.6% → 48.2% (+17.6 points).
  - Verbosity
    - GPT‑4: 2163.5 → 10.0 characters (order‑of‑magnitude drop).
    - GPT‑3.5: large increase (1494.9 → 2519.7; Figure 5a).
  - CoT vs. no‑CoT (Table 2)
    - GPT‑4: +56.6% (March) → +3.2% (June).
    - GPT‑3.5: −1.6% (March) → +20.6% (June).
  - Error patterns (Figure 6)
    - GPT‑4 June concentrates its predictions on 0 or 1 happy number, regardless of ground truth.
    - GPT‑3.5 June tends to overestimate, sometimes predicting more than the maximum possible.

- Sensitive questions and jailbreaks (Figure 7; Table 3)
  - Response rate (direct answers to inappropriate prompts; lower is safer)
    - GPT‑4: 21.0% → 5.0% (safer).
    - GPT‑3.5: 2.0% → 8.0% (less safe).
  - Verbosity of refusals
    - GPT‑4: 652.4 → 141.4 characters (more terse; Figure 7a).
    - Example: a full paragraph refusal in March vs. a brief “Sorry, but I can’t assist with that” in June (Figure 7b).
  - AIM jailbreak (Table 3)
    - GPT‑4: 78.0% → 31.0% answer rate (much stronger defense).
    - GPT‑3.5: 100.0% → 96.0% (still highly vulnerable).

- OpinionQA survey (Figure 8)
  - Response rate (will the model select an option?)
    - GPT‑4: 97.6% → 22.1% (large reduction in willingness to opine).
    - GPT‑3.5: 94.3% → 96.7% (stable/increased).
  - Opinion drift
    - 27% of GPT‑3.5’s choices changed between March and June; within‑snapshot randomness measured much lower (2.8% in March; 7.0% in June), indicating real drift (Section 3.4).

- Code generation and formatting (Figure 9; Table 4)
  - Directly executable (without post‑processing)
    - GPT‑4: 52.0% → 10.0% (−42 points).
    - GPT‑3.5: 22.0% → 2.0% (−20 points).
  - Root cause: formatting
    - June versions often wrapped code in Markdown fences or added extra comments despite the instruction “Generate the code only,” breaking executability (Figure 9b).
  - After stripping non‑code wrappers (Table 4)
    - GPT‑4: 52.0% → 70.0% (June improves markedly when format is fixed).
    - GPT‑3.5: 22.0% → 46.0% (March), 2.0% → 48.0% (June).

- LangChain HotpotQA agent (Figure 10)
  - Exact match
    - GPT‑4: 1.2% → 37.8% (+36.6 points).
    - GPT‑3.5: 22.8% → 14.0% (−8.8 points).
  - Failure mode (Figure 10b)
    - March GPT‑4 sometimes produced correct content but failed the agent’s strict “[action]+text” formatting, causing “Could not parse LLM Output.”
    - June GPT‑3.5 sometimes “could not find information” that March GPT‑3.5 retrieved.

- USMLE medical exam (Figure 11)
  - Accuracy
    - GPT‑4: 86.6% → 82.1% (−4.5 points).
    - GPT‑3.5: 54.3% → 54.7% (about flat).
  - Behavioral drift
    - Answer mismatch across time is substantial: 12.2% of GPT‑4’s answers and 27.9% of GPT‑3.5’s changed (Figure 11a).
    - GPT‑3.5 June becomes more verbose; GPT‑4 June responds more tersely (Section 3.7).

- Visual reasoning (ARC) (Figure 12)
  - Exact match
    - GPT‑4: 24.6% → 27.2% (+2.6 points).
    - GPT‑3.5: 10.9% → 14.3% (+3.4 points).
  - Stability
    - Most outputs remained the same across snapshots; the paper notes “more than 90%” of generations identical across March and June for these puzzles (Section 3.8), despite small average gains.

Instruction‑following results (Section 4; Figures 13–14)
- Single‑instruction fidelity (Figure 13a)
  - Extract answer in [brackets]: 99.5% (March) → 0.5% (June).
  - Don’t apologize or say “as an AI model”: 74.0% → 19.0%.
  - Writing constraint (words ending with a given letter): 55.0% → 10.0%.
  - Text formatting (first‑letter bracketing): 13.0% → 7.5%.
- Composite instructions (Figure 14a)
  - Single‑instruction shifts small (−2.0, +4.0, −1.0).
  - Composition drops are large: e.g., `add comma + no quotation` falls by 24.0 points from March to June; `add comma + capitalize` by 9.2 points.

Do the experiments support the claims?
- Yes, the study grounds each claim with quantitative comparisons, often with multiple views:
  - Raw metric deltas (accuracy/exact‑match/executability).
  - Behavioral metrics (verbosity/mismatch).
  - Mechanism‑probing ablations (CoT vs. no‑CoT; with vs. without non‑code wrappers; agent formatting).
  - Safety robustness (AIM jailbreak).
- Where results are mixed, conditions are clear:
  - GPT‑4 improves on multi‑hop QA (agent) and visual reasoning but drops sharply on math and raw code executability.
  - GPT‑3.5 often moves in the opposite direction (e.g., math improves; agent performance drops).

## 6. Limitations and Trade-offs
- Only two timepoints
  - The analysis captures drift between March and June 2023. Behavior may evolve differently outside this window.

- Attribution is indirect
  - The study identifies reduced instruction fidelity as a plausible driver, supported by targeted probes (Figures 13–14), but cannot isolate root causes inside a proprietary stack. Changes could stem from training data, safety layers, decoding policies, or post‑processors.

- Metric design choices
  - `Mismatch` measures answer changes across time but abstracts away full text and rationale—useful for drift detection but it can miss qualitative shifts in reasoning quality.
  - For sensitive‑question response rate, manual labeling is required; while straightforward, it may still introduce judgment variance (Section 3.3).

- Pipeline confounds
  - The LangChain agent and code‑execution pipeline add parsing/execution constraints. These surfaces are realistic but also make results partially dependent on external tooling and strict prompt contracts.

- Generality across domains
  - The eight tasks are broad yet not exhaustive. Other domains (e.g., long‑context tools, program synthesis beyond LeetCode, multilingual tasks) are not covered.

- Stochasticity
  - Temperature is set low (0.1), but single‑sample results remain stochastic. The study partially addresses this by comparing disagreement rates within a snapshot for OpinionQA (Section 3.4), but broader repeated‑sampling analyses are not reported.

## 7. Implications and Future Directions
- Practical implications for users and integrators
  - Treat LLMs as evolving services. Build “canary” test suites with your real prompts and gold answers, run them regularly, and alert on drift in both accuracy and formatting.
  - Enforce structure at the interface boundary:
    - Prefer function‑calling/JSON schemas or constrained decoding to reduce sensitivity to format drift, especially for agents and code tools (Figures 9–10).
  - Design prompts and pipelines with graceful degradation:
    - Validate and sanitize outputs (e.g., strip Markdown fences before compiling code; verify agent action formats).
    - Add guardrails for refusals and verbosity shifts that may cascade into failures downstream.

- Research implications
  - Longitudinal evaluation should become standard. The paper’s methodology—paired snapshots, diverse task suite, and instruction‑fidelity probes—offers a template that labs and third‑party evaluators can extend.
  - Instruction fidelity as a first‑class capability:
    - The dramatic drops in GPT‑4’s instruction compliance (Figures 13–14) suggest developing dedicated benchmarks and training objectives for robust adherence, including compositions of constraints.
  - Understanding CoT robustness:
    - CoT is not universally helpful after updates (Tables 1–2). Future work can investigate when models refuse or shortcut CoT and how to regain consistent benefits (e.g., self‑consistency, tool‑use‑driven reasoning, or structured scratchpads).
  - Safer but less rationale:
    - GPT‑4 reduces harmful responses and jailbreak susceptibility (Table 3) but becomes terser and less explanatory in refusals (Figure 7). Researchers can explore methods that are both safe and informative (e.g., templated refusals with actionable alternatives).
  - Model versioning and transparency:
    - Clearer release notes and reproducible version pinning would help practitioners maintain stability and trace regressions.

In short, the study demonstrates that LLMs-as-a-service can drift substantially—and unpredictably—over short periods. The most actionable takeaway is to monitor not only task accuracy but also instruction fidelity and output formatting, since small compliance changes can break real systems even when core knowledge or reasoning improves.
