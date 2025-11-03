# Why Do Multi‑Agent LLM Systems Fail?

**ArXiv:** [2503.13657](https://arxiv.org/abs/2503.13657)
**Authors:** Mert Cemri, Melissa Z. Pan, Shuyi Yang, Lakshya A. Agrawal, Bhavya Chopra, Rishabh Tiwari, Kurt Keutzer, Aditya Parameswaran, Dan Klein, Kannan Ramchandran, Matei Zaharia, Joseph E. Gonzalez, Ion Stoica
**Institutions:** University of California, Berkeley (Sky Computing Lab), UC Berkeley, Stanford University, University of Illinois Urbana–Champaign

## 🎯 Pitch

This paper introduces the MAST framework, the first comprehensive taxonomy and dataset for diagnosing failure modes in multi-agent systems constructed from large language models. By elucidating 14 specific failure types and providing automated labeling tools, it enables precise, scalable analysis critical for improving system reliability and efficiency, addressing failures rooted in design, coordination, and verification rather than relying merely on stronger base models. This work paves the way for targeted debugging and principled system design, crucial for advancing real-world MAS applications.

---

## 1. Executive Summary (2–3 sentences)
This paper builds the first empirically grounded framework for understanding why multi‑agent systems built from large language models (LLMs) fail. It contributes (a) MAST, a taxonomy of 14 concrete failure modes grouped into three categories, (b) MAST‑Data, 1,642 execution traces from seven popular multi‑agent systems labeled with those failures, and (c) an LLM‑as‑a‑Judge pipeline that scales failure annotation with strong agreement to human experts. Together, these tools show that most failures stem from system design, inter‑agent misalignment, and inadequate verification—problems that require architectural fixes beyond simply using stronger base models.

## 2. Context and Motivation
- Problem the paper addresses
  - Multi‑agent LLM systems (MAS)—sets of specialized LLM “agents” that collaborate via messages and tools—are increasingly used in software engineering, web/task agents, and science assistants. Yet their performance gains are inconsistent and often marginal.
  - The paper identifies a central gap: there is no principled, fine‑grained, cross‑system account of why MAS fail. Without shared definitions and datasets, debugging and improving these systems is ad hoc.

- Why this matters
  - Real‑world impact: MAS are deployed for complex tasks such as software development (e.g., ChatDev, MetaGPT) and web workflows (e.g., AppWorld, Magentic‑One). Failures can waste compute and human time, and can ship incorrect code or unsafe actions.
  - Evidence of the problem: On six open‑source MAS, failure rates range from “41% to 86.7%” (Figure 5; Appendix B), i.e., many runs do not achieve the intended objective.
  - Engineering significance: Reliability demands more than aggregate success/failure rates; we need to know what fails, when, and why (root causes and failure dynamics).

- Prior approaches and their gaps
  - Benchmarks evaluate overall success but not granular failure causes (Section 2.1; citations [27–32]).
  - Design checklists and single‑agent principles exist, but do not systematize multi‑agent failure patterns (Section 2.2).
  - There was no large, labeled corpus of MAS traces, and no validated taxonomy tailored to multi‑agent coordination problems.

- How this work positions itself
  - Provides a bottom‑up, data‑driven taxonomy—MAST—derived via qualitative analysis of real agent traces (Grounded Theory).
  - Publishes a large cross‑framework dataset—MAST‑Data—annotated using that taxonomy.
  - Supplies an automated annotator (LLM‑as‑a‑Judge) to scale labeling and enable quantitative analysis across systems, tasks, and models.

## 3. Technical Approach
This section explains how the study builds MAST‑Data, defines the MAST taxonomy, and calibrates the LLM‑as‑a‑Judge annotator.

- What is a “trace” and what counts as “failure”?
  - A “trace” is the full conversation and tool‑use log across agents for a task. A “failure” is when the MAS does not achieve the intended task objective (Section 3).

- Step 1 — Discover failure patterns with Grounded Theory (Section 3.1)
  - Grounded Theory is a qualitative method where categories emerge from data rather than from predefined hypotheses.
  - Procedure:
    - Collect an initial 150 traces across five MAS and two task types (programming and math): HyperAgent, AppWorld, AG2 (MathChat), ChatDev, and MetaGPT (Section 3.1).
    - Use “open coding” to label observed failure behaviors; iteratively compare cases and memo findings until theoretical saturation (no new failure types appear).
    - Output: a candidate set of failure modes and draft definitions.

- Step 2 — Turn the patterns into a precise taxonomy and validate human agreement (Section 3.2; Appendix A)
  - Inter‑Annotator Agreement (IAA): Three expert annotators independently label batches of traces with the draft taxonomy; disagreements are discussed and definitions refined.
  - After three IAA rounds, the finalized taxonomy achieves “κ = 0.88” Cohen’s Kappa (strong agreement), demonstrating unambiguous definitions (Section 3.2).
  - The paper also releases a small, triple‑labeled subset “MAST‑Data‑human” for transparency and future calibration.

- Step 3 — The MAST taxonomy (Figure 1; Appendix A)
  - 14 failure modes grouped into three categories and aligned to stages of an MAS run (Pre‑Execution, Execution, Post‑Execution):
    - FC1 System Design Issues (e.g., `FM‑1.1 Disobey Task Specification`, `FM‑1.3 Step Repetition`, `FM‑1.5 Unaware of Termination Conditions`).
    - FC2 Inter‑Agent Misalignment (e.g., `FM‑2.2 Fail to Ask for Clarification`, `FM‑2.4 Information Withholding`, `FM‑2.6 Reasoning–Action Mismatch`).
    - FC3 Task Verification (e.g., `FM‑3.1 Premature Termination`, `FM‑3.2 No or Incomplete Verification`, `FM‑3.3 Incorrect Verification`).
  - Figure 1 also gives prevalence across “1642 MAS execution traces,” e.g., `FM‑1.3 Step Repetition` at 15.7% and `FM‑2.6 Reasoning–Action Mismatch` at 13.2%, with overall category shares of 44.2% (FC1), 32.3% (FC2), and 23.5% (FC3).

- Step 4 — Build an LLM‑as‑a‑Judge annotator for scalable labeling (Section 3.3; Table 2; Appendix N)
  - Setup:
    - Input: a full trace, MAST definitions, and few‑shot examples from human‑labeled traces.
    - Model: OpenAI’s `o1`. The annotator outputs which failure modes occurred and textual justifications.
  - Calibration and performance:
    - Without few‑shot: accuracy 0.89, κ = 0.58.
    - With few‑shot: “accuracy 0.94 … κ = 0.77” (Table 2), showing substantial agreement with experts.
  - Generalization:
    - On two unseen MAS and benchmarks—OpenManus and Magentic‑One; MMLU and GAIA—it achieves “κ = 0.79” (Section 3.4).

- Step 5 — Construct MAST‑Data and analysis tools (Section 3.4; Table 1; Figure 2)
  - Data composition:
    - 1,642 annotated traces spanning seven MAS frameworks and multiple benchmarks (Table 1): ChatDev, MetaGPT, HyperAgent, AppWorld, AG2 (MathChat), Magentic‑One, OpenManus.
    - Models include `GPT‑4`, `GPT‑4o`, `Claude‑3.7‑Sonnet`, and open‑source `Qwen2.5‑Coder‑32B‑Instruct` and `CodeLlama‑7B‑Instruct` (Table 1; Appendix I).
  - Tooling:
    - A Python package `agentdash` exposes the annotator and taxonomy for developers (Appendix C gives a usage example).

- How the pieces work together
  - MAST gives precise labels; the LLM‑as‑a‑Judge scales those labels to thousands of traces; MAST‑Data then supports cross‑system, cross‑model, and per‑benchmark analyses and interventions.

## 4. Key Insights and Innovations
- A validated, fine‑grained failure taxonomy for MAS
  - Novelty: Prior work discussed challenges qualitatively; MAST provides 14 well‑defined modes with stage alignment and prevalence numbers (Figure 1; Appendix A).
  - Significance: Enables apples‑to‑apples diagnosis across frameworks and tasks—critical for engineering reliable multi‑agent systems.

- A large, publicly released dataset of labeled MAS traces
  - 1,642 traces with failure annotations plus a human‑labeled subset for calibration (Table 1; Section 3.4).
  - Significance: Establishes a common empirical basis for MAS reliability research.

- A calibrated LLM‑as‑a‑Judge for failure labeling
  - With few‑shot prompting, the judge reaches strong agreement with human experts (accuracy 0.94; κ = 0.77; Table 2).
  - Significance: Dramatically reduces the cost/time of large‑scale failure analysis while preserving definitional fidelity.

- Three design insights grounded in the taxonomy (Section 4)
  - Insight 1 (System Design): Many failures arise from architecture and prompt specifications, not just base‑model limits. Example: improving agent role specifications in ChatDev yields “+9.4%” success gains without changing the LLM (Section 4 and Appendix H).
  - Insight 2 (Inter‑Agent Misalignment): Communication problems often reflect weak “theory of mind” between agents (e.g., `FM‑2.4 Information Withholding`, Figure 3), suggesting the need for deeper communicative intelligence, not just better message formats (Section 4, FC2).
  - Insight 3 (Verification): Single‑stage or shallow verifiers miss real errors; multi‑level verification is needed. Adding a high‑level objective check to ChatDev improves success by “+15.6%” on ProgramDev (Section 4 and Appendix H).

## 5. Experimental Analysis
- Evaluation methodology
  - Systems and tasks: Seven MAS frameworks, spanning software engineering (ProgramDev, SWE‑Bench Lite), math (GSM‑Plus, OlympiadBench), general agent tasks (GAIA), and knowledge tests (MMLU) (Table 1; Appendices B, D).
  - Labels: Failure modes assigned by human experts for calibration and by the LLM‑as‑a‑Judge for scale (Sections 3.2–3.4).
  - Agreement checks: Human–human κ = 0.88; LLM‑judge vs human κ = 0.77 on held‑out traces; generalizes to unseen systems/benchmarks with κ = 0.79 (Sections 3.2–3.4).

- How often do systems fail?
  - Figure 5 reports failure rates by system/benchmark (not meant for direct cross‑system comparison). The failures range broadly—e.g., AppWorld (Test‑C) shows “86.7% failure,” while AG2 (OlympiadBench) shows “41.0% failure.”
  - The study uses these rates illustratively and emphasizes per‑system failure profiles rather than leaderboard‑style comparisons.

- What fails and where in the pipeline?
  - Global distribution (all 1,642 traces): System Design Issues ≈ 44.2%, Inter‑Agent Misalignment ≈ 32.3%, Task Verification ≈ 23.5% (Figure 1).
  - Per‑system profiles (first 30 traces/system; n=210): System Design 41.8%, Inter‑Agent 36.9%, Verification 21.3% (Figure 4). The mix differs by framework (e.g., AppWorld is prone to `FM‑3.1 Premature Termination`; HyperAgent shows `FM‑1.3 Step Repetition` and `FM‑3.3 Incorrect Verification`).

- Are the failure categories distinct?
  - A correlation analysis shows low cross‑category correlations (0.17–0.32; Figure 6), suggesting the taxonomy captures distinct aspects of failure.
  - Some modes with similar symptoms correlate moderately (max 0.63; Figure 7), warning that automated annotators can conflate causes if not carefully prompted.

- Model choice vs. architecture: Which matters more?
  - Effect of LLMs within the same architecture (MetaGPT): With GPT‑4o vs Claude‑3.7‑Sonnet on ProgramDev‑v2, GPT‑4o shows substantially fewer System Design and Inter‑Agent failures; both still suffer many Verification failures (Figure 8).
  - Effect of MAS architecture with the same LLM (GPT‑4o): MetaGPT vs ChatDev on ProgramDev‑v2—MetaGPT has far fewer System Design and Inter‑Agent failures but more Verification failures than ChatDev (Figure 9). Architectural choices shift which problems dominate.

- Open‑source models
  - With `Qwen2.5‑Coder‑32B‑Instruct` vs `CodeLlama‑7B‑Instruct` on ChatDev and MetaGPT, Qwen is markedly more robust (fewer failures), but both lag behind GPT‑4o/Claude (Appendix I; Table 6).
  - Example counts (Table 6): on ChatDev, `FM‑1.3 Step Repetition` occurs 96 times with Qwen vs 97 with CodeLlama; more stark gaps appear in `FM‑1.5 Unaware of Termination Conditions` (Qwen 94 vs CodeLlama 97) and misalignment modes (`FM‑2.5` 2 vs 42).

- Do failure modes predict success?
  - Table 7 separates traces by final outcome and shows that some modes are “fatal” (e.g., `FM‑1.5 Unaware of Termination Conditions` and `FM‑2.4 Information Withholding` appear almost only in failed runs), while verification failures (`FM‑3.2`, `FM‑3.3`) often appear even in successful runs—indicating systemic verifier weakness that doesn’t always flip the final outcome.

- Benchmark difficulty matters
  - With AG2 (GPT‑4o) across benchmarks (Table 8): failure rates per trace rise with task difficulty—e.g., OlympiadBench shows higher System Design (1.19) and Verification (0.67) failure rates than GSM (0.53; 0.37).

- Case studies: Can MAST guide improvements?
  - AG2 MathChat (Section H.1; Table 5):
    - Baseline vs Prompt improvement vs New topology, repeated 6× runs.
    - GPT‑4: 84.75% → 89.75% (prompt) → 85.50% (topology).
    - GPT‑4o: 84.25% → 89.00% (prompt) → 88.83% (topology).
    - Statistical note: on GPT‑4, only the prompt change gives significant gains; on GPT‑4o, both prompt and topology yield significant improvements (Wilcoxon p = 0.03 vs baseline).
  - ChatDev (Section H.2; Table 5):
    - ProgramDev‑v0: 25.0% → 34.4% (prompt) → 40.6% (topology).
    - HumanEval: 89.6% → 90.3% (prompt) → 91.5% (topology).
    - A separate workflow tweak (“CEO final say”) earlier yields “+9.4%” (Section 1 and Section 4).
  - MAST detects how interventions change failure profiles, not just success rates (Appendix H.3; Figures 10–11).

- Cost of automated annotation
  - Average $1.80 per trace; costs vary with trace length (Table 9), e.g., OpenManus ≈ $4.14/trace; MetaGPT ≈ $2.45/trace.

- Representative failures
  - The paper provides concrete trace snippets for each mode (Appendix N). Example: Figure 3 shows `FM‑2.4 Information Withholding`—a Phone Agent fails to tell the Supervisor that the API expects a phone number, causing repeated login failures.

- Do the experiments support the claims?
  - The taxonomy’s reliability is supported by high human agreement (κ = 0.88) and strong judge agreement (κ = 0.77) on held‑out traces and across new systems (Sections 3.2–3.4).
  - Cross‑system analyses repeatedly show that:
    - Many failures cluster in System Design and Inter‑Agent categories (Figures 1 and 4).
    - Verification is a persistent weak link across LLMs and frameworks (Figures 8–9).
  - Intervention studies demonstrate that MAST‑guided changes measurably improve outcomes (Section H; Table 5), though not universally or completely—suggesting deeper, structural fixes are needed (Section 5.3; Appendix G).

## 6. Limitations and Trade‑offs
- Scope of taxonomy
  - The taxonomy is comprehensive but not exhaustive: “we do not claim MAST covers every potential failure pattern” (Section 4). New domains (e.g., robotics) may surface additional modes.
- Root‑cause certainty
  - Labels are derived from traces; some modes have similar surface symptoms (Appendix E), and the judge’s recall is 0.77 (Table 2), so subtle cases can be misclassified.
- Dataset breadth and comparability
  - Benchmarks and tasks vary per system (Table 1; Figure 4 caption), so performance numbers are illustrative, not head‑to‑head comparisons.
  - Closed‑source systems (e.g., Manus) could not be included in failure analyses due to missing full traces (Appendix B.3).
- Generalization beyond studied settings
  - Most traces are programming/math/web tasks. Other domains (embodied agents, safety‑critical control) may display different failure dynamics.
- Verification ground truth
  - For some tasks, final success/failure requires human evaluation (Table 1 “HE”). Root‑cause verification can be subjective without formal specs or unit tests.
- Cost/compute
  - Automated annotation has non‑trivial cost (Table 9) and depends on access to a high‑end model (`o1`). Reproducibility may be limited by API/price changes.

## 7. Implications and Future Directions
- How this changes the landscape
  - Provides a shared language and dataset for MAS reliability. Researchers and practitioners can now measure not only whether a system fails but how and where it fails—enabling targeted debugging and principled system design.
  - The taxonomy reveals that many issues are organizational: agent roles, workflows, and verification pipelines—echoing insights from reliability engineering (Section 5.3).

- Practical recommendations (Appendix G; Section 4)
  - Prioritize structural fixes:
    - Multi‑level verification (unit tests, runtime checks, high‑level objective validation) rather than only final “does it compile?” checks.
    - Standardized, structured inter‑agent communication (beyond free‑form chat) and protocols that surface assumptions and uncertainties.
    - Memory/state management for long‑horizon coordination; avoid conversation resets and context loss (`FM‑1.4`, `FM‑2.1`).
    - Incorporate uncertainty/confidence thresholds to trigger clarification (`FM‑2.2`) and avoid premature termination (`FM‑3.1`).
  - Tactical measures help but are inconsistent:
    - Better prompts and clearer role specs can reduce `FM‑1.x` failures; majority‑vote/resampling helps only if backed by real verifiers (Appendix G.1, G.2; [67, 68]).

- Research directions enabled by MAST‑Data
  - Train “socially aware” agents: datasets for modeling other agents’ information needs and improving `FM‑2.x` cases (Section 4, FC2).
  - Learn verifiers: domain‑adaptive, multi‑granular verifiers that combine symbolic checks and test generation; move toward provable guarantees in constrained domains (Section 4, FC3; Appendix G.2).
  - Automated failure attribution and repair: integrate MAST‑style labels with causal debugging tools and agent‑level credit assignment (Appendix H; related to [41, 42]).
  - Taxonomy expansion: incorporate new domains (embodied agents, safety/security) and refine ambiguous boundary cases identified by the correlation analysis (Appendix E).

- Downstream applications
  - Engineering dashboards that track failure modes over time during development and after deployment.
  - Model selection and system design: choose LLMs and architectures based on expected failure profiles (Figures 8–9), not just average accuracy.
  - Continuous integration for agents: enforce verification gates tied to `FM‑3.x`, regression tests for `FM‑1.x` design regressions, and communication linting for `FM‑2.x`.

> In summary, the paper provides the field with a vocabulary (MAST), an evidence base (MAST‑Data), and a scalable instrument (LLM‑as‑a‑Judge) to diagnose and reduce MAS failures. The main takeaway is actionable: most problems trace to design, coordination, and verification; fixing them requires architectural rigor and better verifiers—not merely stronger base models.
