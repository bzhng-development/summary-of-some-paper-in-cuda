# The Automated LLM Speedrunning Benchmark: Reproducing NanoGPT Improvements

**ArXiv:** [2506.22419](https://arxiv.org/abs/2506.22419)

## 🎯 Pitch

This paper introduces the Automated LLM Speedrunning Benchmark, a groundbreaking suite evaluating AI agents on their ability to faithfully reproduce 19 sequential, real-world innovations from the NanoGPT speedrun—each demanding code-level replication of state-of-the-art LLM training improvements and measurement of actual speedup. The results reveal that even the best current LLM agents struggle, recovering less than half the human-achieved speedups, demonstrating that reliable automated scientific reproducibility remains a core obstacle for autonomous AI research—highlighting a vital, unsolved challenge on the road to trustworthy automated science.

---

## 1. Executive Summary
This paper introduces the Automated LLM Speedrunning Benchmark, a controlled suite of 19 sequential “reproduction” tasks built from the community NanoGPT speedrun records. Each task asks an AI research agent to re-implement the next record’s code-level improvement and recover its wall‑clock training speedup on a fixed setup. Across 6,840 agent runs, recent reasoning LLMs with state-of-the-art scaffolds recover only a fraction of human speedups; the best setting (o3‑mini with a branching search scaffold and detailed hints) recovers about 46% on average (Table 3), revealing automated reproducibility as a current bottleneck.

## 2. Context and Motivation
- Problem addressed
  - Automated science requires more than proposing ideas; it demands turning textual descriptions into faithful, runnable experiments that reproduce published outcomes. The paper targets this “automated reproducibility” step for LLM training (Section 1).
  - Prior reproducibility agent benchmarks use heterogeneous papers and metrics, making cross-task comparisons difficult. This work offers a unified, sequential benchmark with a single metric: time to a target loss (Sections 1, 3.1).

- Why it matters
  - Reproducibility underpins trustworthy science. If agents cannot reliably reconstruct known improvements, they are not ready to autonomously scale research (Section 1).
  - The benchmark focuses on practically important LLM training improvements (e.g., optimizers, attention kernels), not toy tasks. The NanoGPT speedrun has reduced GPT‑2 training from 45 minutes to under 3 minutes on 8×H100 since 2024 (Section 1; Table E.1).

- Prior approaches and gaps
  - Agent benchmarks like CORE-Bench, PaperBench, and SciReplicate test installing or reproducing disparate papers/code bases (Section 2; Table 1). They lack a single continuous research arc with consistent metrics.
  - Coding/agent benchmarks evaluate software tasks but not faithful reproduction of scientific results with hardware-constrained runtime targets (Section 2).

- Positioning
  - This benchmark builds tasks from consecutive speedrun records of the same codebase, each with known code diffs and measured speedups (Sections 1, 3.1; Figure 2). It isolates whether an agent can implement already-known, diverse improvements (optimizers, attention variants, precision modes, etc., Table E.1) and recover their runtime benefits on identical hardware.

## 3. Technical Approach
- Core task construction (Section 3.1; Figure 2)
  - A “record” is a single speedrun script (`train_gpt2.py`) that reaches a fixed validation loss (3.28 on FineWeb) as fast as possible on one 8×H100 node (Section 1).
  - For each adjacent pair of records Ri−1 → Ri (19 transitions after excluding a pure framework upgrade), a benchmark task provides:
    - Starting code: the previous record’s `train_gpt2.py` (Ri−1).
    - Measured target time: `ti` (seconds) for Ri on the benchmark’s hardware (Appendix A confirms near-exact reproduction).
    - Optional hints about the change that produced Ri:
      - Level 1 (`Δ1`): pseudocode of the change.
      - Level 2 (`Δ2`): natural-language description.
      - Level 3 (`Δ3`): a mini-paper-like summary including rationale (all manually verified; Appendix D/F).
  - Two evaluation modes (Section 3.1):
    - Record reproduction (with hints): reproduce Ri using Ri−1 and hints.
    - Record optimization (no hints): improve Ri−1 however possible.

- Metric: Fraction of Speedup Recovered (FSR) (Eq. 1–2, Section 3.1)
  - Intuition: How much of the human-record speedup did the agent’s code recover, relative to the previous record?
  - Formula for a task i (from Ri to Ri+1):
    - FSRi = (ti − t′i+1) / (ti − ti+1), where t′i+1 is the agent’s time.
    - Aggregate score: mean FSR across all records I.

- Agent scaffold and search (Section 3.2; Figure 3; Table 2)
  - Each search node is a self-contained solution directory with:
    - Edited `train_gpt2.py`.
    - A results file (wall time, loss).
    - A natural-language summary of the run.
  - Three-stage loop per node:
    - Coder: uses Aider (diff-based edits) guided by prompts and optional hints (Appendix D, Figures D.4–D.5).
    - Executor: runs on 8×H100 with a 60‑minute per‑solution cap (Section 4.1).
    - Analyzer: extracts metrics from logs and summarizes failures (Appendix D, Figures D.6–D.7).
  - Search variants (Table 2), all given the same budget M=20 nodes (Section 4.1):
    - Flat (Best‑of‑M): generate M independent solutions, pick best.
    - Tree/Forest: branching without explicit debugging.
    - AIDE/Multi‑AIDE: iterative branching with debugging of buggy leaves (`pdebug=0.5`, `Dmax=5`), N0=3 initial roots, branch factor N=3 where applicable.

- Similarity analyses (Section 4.6)
  - Code-embedding distance using `SFR-Embedding-Code 2B` (normalize distance between agent code and target human code by the distance between consecutive human records). The “L2 distance recovered” is 1 − normalized L2.
  - LLM-as-judge: R1 reads diffs and scores fraction of human changes reproduced in agent code (Appendix C).

- Experimental workload (Section 4.1)
  - 4 models: `DeepSeek‑R1`, `o3‑mini`, `Gemini‑2.5‑Pro`, `Claude‑3.7‑Sonnet`.
  - 5 scaffolds × 6 hint regimes × 19 records × 3 seeds = 6,840 runs; ≈10 hours per agent run on average; 60‑minute cap per attempted solution.

- Extra probes
  - Adding missing background knowledge: injecting the FlexAttention blog into context for the FlexAttention record (R12) to test whether external docs help (Section 4.7; Table 4).
  - Cumulative reproduction: making each task build on the agent’s previous solution (`R′i−1 → R′i`) rather than resetting to the human code each time (Section 4.8; Figure 9).

Notes on uncommon terms used here:
- `Speedrun record`: the best‑known script achieving the target loss fastest at a point in time.
- `FlexAttention`: a PyTorch programming model for fast, custom attention kernels (Section 4.7).
- `Muon`: an optimizer introduced in the speedrun that later generalized to larger models (Section 1; Table E.1).
- `IQM` (interquartile mean): mean computed after removing the lowest and highest quartiles; more robust for small samples (Section 4.4).

## 4. Key Insights and Innovations
1) A sequential, code-level reproducibility benchmark for LLM training
- Novelty
  - Converts 19 consecutive improvements from a single real project (NanoGPT speedrun) into standardized tasks with a single metric and ground-truth code diffs (Sections 1, 3.1; Table E.1; Figure 2).
- Why it matters
  - Enables measuring whether agents can walk an entire research arc rather than a one-off replication. This is not offered by prior benchmarks (Table 1).

2) A unified evaluation metric tied to wall-clock gains
- Novelty
  - `FSR` (Eq. 1–2) compares the agent’s improvement against the exact speedup achieved by the next human record on identical hardware (Section 3.1).
- Significance
  - Makes gains directly meaningful for LLM training operations, not just proxy metrics.

3) Systematic study of search scaffolds under equal budgets
- Novelty
  - Five search schemes (Table 2) run under the same node budget M=20, separating search design from compute advantage (Sections 3.2, 4.1).
- Significance
  - Yields actionable findings on the relative benefits of flat vs. iterative/debugging search for research agents (Section 4.2; Figure 4; Figure 8).

4) Multi-form hints with controlled difficulty
- Novelty
  - Three hint levels—from pseudocode to mini‑paper—plus their combinations allow controlled ablations on how information format and length affect agent performance (Section 3.1; Figure 4; Table 3).
- Significance
  - Reveals surprising context-length/format sensitivities in frontier models (Section 4.3; Table 3).

These are fundamental contributions (benchmark design, metric, scaffold comparison, and hinting study). Empirical observations listed below are insights produced by the benchmark rather than structural innovations.

## 5. Experimental Analysis
- Evaluation setup (Sections 4.1, 3.1, Appendix A)
  - Task: edit `train_gpt2.py` starting from Ri−1 to reach val_loss 3.28 faster. Dataset: FineWeb (Section 1). Hardware: single 8×H100 node. Records: 19 (excluding a pure PyTorch version change). Budget: M=20 nodes/search; 60‑minute per-solution cap.

- Main quantitative results
  - Without hints, all agents recover <20% of human speedups on average (Figure 4; Section 4.2).
  - With hints, the best mean FSR comes from o3‑mini:
    - Pseudocode alone: “0.43±0.02” with Multi‑AIDE (Table 3).
    - All hints combined: “0.46±0.04” with Multi‑AIDE (Table 3).
  - IQM aggregation across all runs shows:
    - Best performance when all three hints are available; Multi‑AIDE outperforms other scaffolds; `Gemini‑2.5‑Pro` and `Claude‑3.7‑Sonnet` lag close to 0 FSR (Figure 5; Section 4.4).
  - Hint-format interactions:
    - For `o3‑mini`, adding text/mini-paper to pseudocode often lowers performance for some scaffolds (e.g., Flat 0.40→0.24 with L1→L1+L2+L3; Table 3), suggesting information overload.
    - For `DeepSeek‑R1`, the opposite: combined hints substantially improve FSR (e.g., Multi‑AIDE 0.16 [L1] → 0.41 [L1+L2+L3]; Table 3).

- Search scaffolds and debugging (Sections 4.2, 4.5)
  - Flat search surprisingly matches or outperforms iterative scaffolds for single-hint levels (levels 1–3), while branching search reduces the fraction of buggy nodes (Figure 8).
  - Explicit debugging (AIDE, Multi‑AIDE) does not assure better results than non‑debugging branching (tree/forest) under the equal budget (Section 4.2).
  - Model-specific behavior:
    - `Claude‑3.7‑Sonnet` generates many buggy nodes that grow over time (Figure 8), explaining why IQM penalizes it despite occasional strong FSR (Section 4.5).
    - `Gemini‑2.5‑Pro` yields fewer buggy nodes but low FSR, implying conservative, robust edits that miss the targeted improvements (Section 4.5).

- Difficulty increases with later records
  - Recovered speedups and code-embedding similarity drop as the record index advances (Figure 6), matching intuition that later gains are more complex.

- Code similarity vs. performance (Section 4.6)
  - “Modest correlation” between L2 distance recovered and FSR, stronger with richer hints (Figure 7). An independent R1 “judge” that reads diffs also correlates with FSR (Appendix C; Figure C.2).

- External knowledge injection can hurt
  - For the FlexAttention record R12, adding a FlexAttention blog summary into the prompt worsens FSR for both `R1` and `o3‑mini` (Table 4: 0.09→0.07 for R1; 0.10→0.06 for o3‑mini), indicating difficulty in operationalizing new APIs from documentation within this complex setting (Section 4.7).

- Cumulative reproduction is fragile
  - When each task must build on the agent’s previous code (`R′i−1 → R′i`), `o3‑mini` with Multi‑AIDE and all hints recovers ~60% of the speedup for the first step (R2) but drops to ~20% on the next and fails by R4 (Figure 9; Section 4.8). Small deviations early compound and derail later steps.

- Do the experiments support the claims?
  - The study spans 6,840 runs with controlled hardware, fixed budgets, and multiple models/scaffolds, with human record times re-verified on the same cluster (Appendix A). Results are consistently triangulated via FSR, IQM, bug-rate analysis, and code-similarity metrics (Figures 4–8; Appendix C). The evidence robustly supports the central claim that automated reproduction of real LLM training improvements remains difficult.

## 6. Limitations and Trade-offs
- Scope and assumptions (Section 5; Limitations)
  - Tasks are single-file (`train_gpt2.py`) reproductions on a fixed 8×H100 node; this excludes multi-file refactors, multi-node distributed training, and broader engineering challenges typical in real LLM systems.
  - Success is a single outcome metric: time to reach a target validation loss of 3.28 on FineWeb for GPT‑2 scale (124M). It does not assess downstream task quality, stability under different datasets, or larger model scales (Sections 1, 3.1).

- Hinting and knowledge access
  - Hints are manually curated and concise to fit within model context; the agent does not fetch or manage large external corpora or long-term memory (Section 5 “Scaling up external knowledge”). The negative R12 doc-injection result (Table 4) suggests current agents struggle to ground long or unfamiliar documentation into correct code.

- Potential data memorization
  - Some human records predate the models’ training cutoffs; though performance is far from saturated, future models may memorize solutions, complicating interpretation (Section 5 “Memorization or generalization?”).

- Compute budget and search depth
  - The equal budget of M=20 nodes per run keeps comparisons fair but may be tight for complex changes, perhaps understating benefits of iterative debugging in deeper searches (Sections 3.2, 4.1).

- Benchmark coverage
  - Focuses on reproducing known results, not discovering new ones. It is a necessary but not sufficient condition for autonomous research (Section 6).

## 7. Implications and Future Directions
- Field impact
  - The benchmark establishes automated reproducibility—turning descriptions into correct, performant code—as a measurable capability distinct from code generation or reasoning alone. Results in Figures 4–5 set a clear, non‑saturated baseline for the community.

- Follow-up research opportunities
  - Retrieval and memory: Equip agents with tools for retrieval, scratchpads, and long-term memory to go beyond compact hints (Section 5). Tree-organized retrieval methods (e.g., `RAPTOR`) are relevant references cited.
  - Robust planning and verification: Integrate static/dynamic analyzers and unit tests tailored to performance goals, not just correctness; improve error triage so debugging steps meaningfully reduce bug rates (Figure 8 shows room for improvement).
  - Semantic diffs and reasoning over code changes: Move from numeric similarity to automatic, structured “commit message” style explanations to identify missed micro‑edits that block speedups (Section 5 “Semantic diffs”).
  - Larger and more realistic settings: Extend to multi-file codebases, multi-node distributed training, alternative objectives (memory footprint, quality vs. speed), and agent-defined intermediate metrics (Section 5 “From LLM speedrun to ML speedrun”).
  - Training agents with the benchmark: Use the tasks as curricula for RL or tool-augmented training (search over code, execution feedback, and log parsing prompts in Appendix D).

- Practical applications
  - Reproducibility assistants for ML labs: Given a changelog or ablation description, propose and validate code diffs, summarize logs, and quantify speed gains.
  - CI for research code: Automated regression checks on “time‑to‑target‑loss” under fixed hardware to guard against performance regressions as code evolves.
  - Education and onboarding: Teach engineers how concrete code changes (e.g., `Muon`, FlexAttention, precision modes) translate to measurable speedups with consistent scaffolds.

> Representative results to remember:
> - “All agents fail to recover more than 20% of the speed-up … without hints” (Figure 4; Section 4.2).
> - Best setting: `o3‑mini + Multi‑AIDE + (L1+L2+L3)` ≈ “0.46±0.04 FSR” (Table 3).
> - Branching search reduces buggy nodes, but `Claude‑3.7‑Sonnet`’s bug rate still dominates its trees (Figure 8).
> - Code similarity correlates with FSR when richer hints are present (Figure 7; Appendix C).
> - Injecting FlexAttention docs into prompts worsens reproduction of that record (Table 4).
> - Building cumulatively on one’s own prior code causes rapid deterioration by the third record (Figure 9).

Overall, the benchmark crystallizes a critical capability gap: today’s best LLM agents can describe improvements but struggle to implement them faithfully and robustly enough to recover real, hardware-measured training speedups across an entire research trajectory.
