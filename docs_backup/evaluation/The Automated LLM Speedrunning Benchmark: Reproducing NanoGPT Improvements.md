# The Automated LLM Speedrunning Benchmark: Reproducing NanoGPT Improvements

**ArXiv:** [2506.22419](https://arxiv.org/abs/2506.22419)
**Authors:** Bingchen Zhao, Despoina Magka, Minqi Jiang, Xian Li, Roberta Raileanu, Tatiana Shavrina, Jean‑Christophe Gagnon‑Audet, Kelvin Niu, Shagun Sodhani, Michael Shvartsman, Andrei Lupu, Alisia Lupidi, Edan Toledo, Karen Hambardzumyan, Martin Josifoski, Thomas Foster, Lucia Cipolina‑Kun, Abhishek Charnalia, Derek Dunfield, Alexander H. Miller, Oisin Mac Aodha, Jakob Foerster
**Institutions:** Not specified in available sources

## 🎯 Pitch

The paper presents the Automated LLM Speedrunning Benchmark, a novel framework focused on evaluating AI's ability to replicate known GPT-2 training enhancements and achieve similar speedups. This benchmark is crucial for advancing AI's role in automated reproducibility and innovation, highlighting current limitations and paving the way for more capable and reliable AI research agents.

---

## 1. Executive Summary (2-3 sentences)
This paper introduces the Automated LLM Speedrunning Benchmark, a suite of 19 step-by-step tasks that measure how well an AI “research agent” can reimplement known improvements to GPT‑2 training and recover the corresponding wall‑clock speedups. On fixed hardware and a common target loss, the benchmark evaluates agents with optional hints and a standardized search scaffold, revealing that current frontier reasoning models struggle to reliably reproduce already-known code innovations (e.g., best mean fraction of speedup recovered ≈40–46% with strong hints; Figure 4, Table 3).

## 2. Context and Motivation
- Problem addressed
  - The work targets automated reproducibility: can an AI agent read a description of an experiment and produce an implementation that reproduces the reported outcomes? Here, outcomes are “speedrun records” that reduce the time to train GPT‑2 to a fixed target loss (Section 1).
  - The benchmark transforms a community “speedrun” (successive human-record improvements to nanoGPT training) into reproducibility tasks with clear ground truth code and timing for each step (Figure 2, Section 3.1).

- Why this matters
  - Reproducibility is a bedrock of science and a prerequisite for trustworthy automation in research workflows (Introduction; citations therein).
  - For AI research agents, reliably converting textual descriptions into functioning, performant code is necessary before attempting autonomous innovation. The tasks are small and quick by design (minutes per run in human records), making them accessible and practical (Section 1).

- Prior approaches and gaps
  - Existing benchmarks test code execution or paper-to-code translation across diverse topics, but typically lack:
    - A single overarching scientific arc with cumulative steps and a unified metric.
    - Code-level ground truth “next step” diffs and exact timing targets on fixed hardware.
  - Table 1 contrasts this benchmark with CORE-bench, PaperBench, and others: this work is “Reproducibility: Yes; Sequential: Yes; LLM research: Yes; Agent scaffold: Yes.”

- Positioning
  - The benchmark is built directly on the NanoGPT Speedrun, which progressed GPT‑2 training from 45 minutes to under 3 minutes on a single 8×H100 node (Section 1; Table E.1). It evaluates agents’ ability to replicate each record-to-record change, with optional hints at three levels of detail (Section 3.1).

## 3. Technical Approach
- Overall task formulation (Section 3.1; Figure 2)
  - Records: Let `R_i` be the i‑th “speedrun record,” each a self-contained training script and its wall time `t_i` to reach a fixed validation loss (3.28) on FineWeb, using one 8×H100 node (Section 1).
  - Tasks: For each transition `R_i → R_{i+1}` (excluding `i = 6→7`, a PyTorch version upgrade), the agent receives:
    - The starting script `R_i`.
    - Optional hints about what changed in `R_{i+1}`: `Δ1` pseudocode, `Δ2` text description, `Δ3` mini-paper (Section 3.1; Appendix F for examples).
  - Outputs: The agent edits code to produce `R'_{i+1}`, which is executed on fixed hardware until the target loss is reached, or timeouts.

- Evaluation metric (Equations (1) and (2), Section 3.1)
  - Fraction of Speedup Recovered (`FSR`): “How much of the human speedup did the agent reproduce?”
  - Definition per step:
    - `FSR_i = (t_i − t'_{i+1}) / (t_i − t_{i+1})`, where `t'_{i+1}` is the agent’s achieved wall time; `t_i` and `t_{i+1}` are human-measured on the same cluster (Appendix A).
  - Overall score is the mean across records in the benchmark (Equation (2)).

- Two task modes (Section 3.1)
  - Record reproduction: hints provided (1/2/3 or combinations). The agent tries to match the next record.
  - Record optimization: no hints (`m = {0}`). The agent can try any improvement starting from `R_i`.

- Hints and their provenance
  - Three levels: pseudocode (L1), natural-language description (L2), paper-like “mini-paper” (L3). Drafted with LLM assistance then manually verified and edited for correctness (Section 3.1; Appendix D, F).

- Search scaffold (Section 3.2; Figure 3; Table 2)
  - Concept: A general “search over code versions” where each node contains a full runnable solution (e.g., `train_gpt2.py`) and an execution summary (`results.json`).
  - Workflow per iteration:
    1. Implementation: an LLM “coder” edits code using Aider (a diff-based tool) given the task and optional hints/history.
    2. Execution: run on an 8×H100 node; capture stdout/stderr and performance.
    3. Analysis: summarize logs and extract metrics (Appendix D shows the prompts/templates).
  - Branching strategies (all with the same total budget `M` nodes):
    - `Flat (Best-of-M)`: generate M independent candidates.
    - `Tree`, `Forest`: different initial/branching factors.
    - `AIDE`, `Multi-AIDE`: include explicit debugging of buggy nodes with probability `p_debug` up to depth `D_max` (Table 2).
  - Default parameters in experiments: `N0=3`, `N=3`, `p_debug=0.5`, `D_max=5`, `M=20` nodes per run (Section 4.1).

- Models evaluated (Section 4.1)
  - Four LLMs as the agent “brain”: `DeepSeek‑R1`, `o3‑mini`, `Gemini‑2.5‑Pro`, `Claude‑3.7‑Sonnet`.

- Code similarity measures (Section 4.6)
  - Embedding-based similarity: compute code embeddings with SFR-Embedding-Code 2B; define “L2 distance recovered” as the fraction of the human-to-human change that the agent’s code closes (Section 4.6; Figure 7).
  - LLM-as-judge: prompt an LLM to score how much of the human diff was reproduced (Appendix C; judge prompt shown).

- Special probe on missing knowledge (Section 4.7)
  - For a hard step (`R12`, which introduces `FlexAttention`), the authors insert a small “documentation snippet” as extra context to see if performance improves (Table 4).

- Cumulative reproduction experiment (Section 4.8; Figure 9)
  - Harder setting: each step starts from the agent’s own previous code `R'_{i}` instead of the human `R_i`, to test whether agents can “chain” their reproductions across multiple steps.

## 4. Key Insights and Innovations
- A sequential, code-grounded reproducibility benchmark with a single metric
  - Novelty: Each task has a unique, code-level ground-truth target and the same success criterion (time to target loss on fixed hardware). This enables measuring an agent’s ability to reproduce an entire “research arc,” not just isolated results (Section 3.1; Figure 2; Table E.1).
  - Significance: Avoids apples-to-oranges comparisons across papers/datasets and elevates reproducibility from a one-off exercise to a cumulative capability.

- Multi-level hints that consciously vary abstraction
  - Contribution: Three hint formats—from pseudocode to paper-like narrative—allow controlled tests of how background information impacts reproduction (Section 3.1; Appendix F).
  - Insight: Pseudocode is most helpful for strong models, while long-form hints sometimes degrade performance (Table 3), revealing model- and context-length sensitivities (Section 4.3).

- A general, extensible search scaffold for code improvement
  - Design: Unified framework that supports best-of-M, tree/forest, and AIDE-style debugging, with consistent budgets (Section 3.2; Table 2).
  - Insight: Surprisingly, simple “flat” search often matches or beats iterative scaffolds without hints; with richer hints, `Multi-AIDE` tends to win in robust IQM comparisons (Figure 5), suggesting limited value of explicit debugging unless models can leverage it.

- Diagnostic lenses beyond speedup: similarity and search-tree forensics
  - The paper relates `FSR` to code similarity and search-tree composition:
    - Moderate positive correlation between “L2 distance recovered” and `FSR`, stronger for richer hints (Figure 7).
    - Search-tree breakdowns explain discrepancies in averaged `FSR` vs. robust IQM (e.g., Claude yields many buggy nodes; Figure 8), guiding where scaffolds and prompts fail.

- Finding: Current reasoning LLMs struggle with faithful scientific reproduction even with strong guidance
  - Quantitatively, the best setting recovers less than half of human speedups on average (Figure 4; Table 3), and chaining reproductions rapidly collapses (Figure 9). This pinpoints automated reproducibility as a core bottleneck to “AI Scientist” aspirations.

## 5. Experimental Analysis
- Evaluation setup
  - Tasks: 19 record transitions (excluding the PyTorch-upgrade record) from the NanoGPT speedrun (Section 3.1; Table E.1).
  - Hardware and runtime: Each candidate runs for up to 60 minutes; 6,840 agent runs total (19 records × 6 hint regimes × 5 scaffolds × 4 models × 3 seeds), averaging ≈10 hours per full agent run across the search budget (Section 4.1). Human record times were re-measured on the same cluster to ensure fairness (Appendix A, Figure A.1).
  - Metric: `FSR` per Equation (1) and mean `FSR` across records per Equation (2). Robustness summaries use interquartile mean (IQM) with bootstrapped 95% CIs (Figure 5; Section 4.4).

- Main quantitative findings
  - Hints are necessary. Without hints, all agents recover under ≈20% of speedups on average (Section 4.2; Figure 4).
  - Best-performing model and hint:
    - `o3‑mini` with pseudocode (L1) consistently strong: mean `FSR ≈ 0.40–0.43` across scaffolds (Table 3, rows “L1 (pseudocode) o3‑mini”).
    - With all hints combined, performance can either improve or drop depending on the scaffold; the best cell in Table 3 for `o3‑mini` is `Multi‑AIDE` with `L1+L2+L3` at `0.46 ± 0.04`.
  - Mixed effects of combining hints (Section 4.3; Table 3)
    - For `o3‑mini`, adding long-form hints to pseudocode often hurts (e.g., `Flat` L1+L2 vs. L1 drops from `0.40` to `0.27`; Table 3).
    - For `DeepSeek‑R1`, combining hints helps substantially (e.g., `Forest` L1+L2+L3 reaches `0.40 ± 0.04`, much higher than any single hint row for R1).
  - Model-level IQM (robust) comparison (Figure 5)
    - Best model overall is still `o3‑mini`. `Gemini‑2.5‑Pro` and `Claude‑3.7‑Sonnet` perform close to zero `FSR` in IQM despite some favorable averages elsewhere, highlighting instability/outliers and frequent buggy code (Section 4.4 and 4.5).
  - Scaffold-level insights (Figures 4–5; Section 4.2, 4.4, 4.5)
    - Simple `Flat` (best-of-M) is surprisingly competitive and often matches or beats iterative scaffolds for individual hint levels (Section 4.2).
    - In IQM aggregation, `Multi‑AIDE` is best among scaffolds (Figure 5), but only when models can benefit from debugging.
    - Search tree analysis (Figure 8) shows branching lowers the fraction of buggy nodes vs. flat; `Claude‑3.7‑Sonnet` produces many buggy nodes that grow over time, explaining its poor IQM despite some high averages.
  - Difficulty increases over records (Figure 6)
    - For `o3‑mini` with text hints, later records show lower recovered L2 similarity and lower `FSR`, indicating that earlier steps are easier to reproduce faithfully.

- Similarity, ablations, and robustness checks
  - Code similarity correlates with speedup (Figure 7): R² rises for richer hints, suggesting that actually implementing the intended change matters and the embedding metric captures this.
  - LLM-as-judge (Appendix C) also correlates with `FSR` for `o3‑mini` (Figure C.2) and highlights inherently hard records (e.g., Record 10 shows uniformly low reproducibility scores across methods; Figure C.1).
  - External knowledge stress test (Section 4.7; Table 4):
    - Adding `FlexAttention` documentation to the prompt for `R12` hurts performance:
      > Table 4: `o3‑mini` `FSR` drops from `0.10±0.01` to `0.06±0.01`; `DeepSeek‑R1` drops from `0.09±0.01` to `0.07±0.01`.
    - Interpretation: models struggle to correctly operationalize new technical docs within a complex codebase edit.
  - Cumulative reproduction (Section 4.8; Figure 9)
    - Starting from the agent’s own prior code, speedup recovery collapses after a few steps:
      > Figure 9: `o3‑mini` + `Multi‑AIDE` + all hints recovers ≈60% for `R2` from `R1'`, ≈20% for `R3` from `R2'`, and ≈0% by `R4`.

- Do experiments support claims?
  - The results are extensive (6,840 runs), use a single hardware platform, and replicate human baseline times (Appendix A). Multiple scaffolds, four models, and six hint regimes are compared with robust statistics (Figure 5). The observed difficulties—low `FSR` without hints, modest ceilings even with detailed hints, and degradation in cumulative reproduction—consistently support the claim that automated reproducibility remains a hard, unsolved capability.

## 6. Limitations and Trade-offs
- Assumptions and scope (Section 5; “Limitations and future directions”)
  - Hints are small and hand-curated to fit context windows; agents are not given open-ended retrieval or persistent memory (Section 5: “Scaling up external knowledge”).
  - The domain is narrow: GPT‑2 speedrun with a single metric (time to target loss) on one node type. This is realistic for LLM training but excludes other ML goals (e.g., accuracy trade-offs at scale, multi-node training) (Section 5: “From LLM speedrun to ML speedrun”).

- Potential memorization (Section 5)
  - Many records predate the LLMs’ cutoffs, so in-weight exposure cannot be ruled out. The paper reruns all human records on their cluster and still finds large performance gaps, but disentangling memorization vs. generalization remains open.

- Compute cost and reproducibility of the benchmark
  - Each agent run requires substantial GPU time (6,840 × 8×H100 executions), which may hinder widespread replication of all experiments (Section 4.1).

- Metric focus and proxy risks
  - `FSR` captures wall-time speedups only. It does not directly measure code quality, maintainability, or generalization of techniques, though the similarity analyses partially address this (Section 4.6).

- External knowledge integration
  - A small “docs injection” hurt performance on `R12` (Table 4), indicating current models struggle with integrating unfamiliar, technical documents into nontrivial code edits. The benchmark, as instantiated, does not yet include a full retrieval-and-memory agent loop (Section 5).

## 7. Implications and Future Directions
- How this changes the landscape
  - The benchmark provides a precise, cumulative, code-level yardstick for automated reproducibility in AI/ML. It highlights that “reading, reasoning, editing, and executing” end-to-end remains far from solved even on small, fast tasks (Section 6, Conclusions).

- Research enabled or suggested
  - Agent architecture:
    - Richer retrieval and tool-use with persistent memory and scratchpads (Section 5: “Scaling up external knowledge”).
    - Stronger debugging and verification steps that reduce buggy nodes (Figure 8) without sacrificing exploration.
  - Prompting and hint design:
    - Pseudocode is especially effective; long-form hints can overload some models (Table 3). Adaptive hinting or automatic synthesis of concise “diff-intent” could help.
  - Robust evaluation:
    - Move beyond code embeddings to semantic diff summaries and automated commit-like messages to categorize successes and failure modes (Section 5: “Semantic diffs”).
  - Generalization and contamination checks:
    - Explicit protocols to measure and control for memorization as stronger models approach saturation (Section 5).

- Practical applications
  - As a testbed for:
    - Assessing the reliability of AI research agents before deploying them on higher-stakes or larger-scale ML R&D.
    - Comparing scaffolds and orchestration strategies under identical constraints.
    - Training and curriculum design for agents: start with early records and gradually add later, harder ones (Figure 6), or attempt cumulative chains (Figure 9) to evaluate long-horizon reliability.

- Longer-term vision
  - If future agents begin to match or exceed human speedrun records across the chain—and do so cumulatively—this would indicate progress toward automated scientific discovery. Until then, the benchmark isolates automated reproducibility as a keystone capability that must be strengthened (Section 6; Appendix G on broader impacts).

> Key result to remember: even with detailed hints, the best setting (`o3‑mini` + `Multi‑AIDE`) recovers only about 46% of human speedups on average (Table 3), and cumulative reproduction degrades rapidly (Figure 9).
