# SWE‑Perf: Can Language Models Optimize Code Performance on Real‑World Repositories?

**ArXiv:** [2507.12415](https://arxiv.org/abs/2507.12415)
**Authors:** Xinyi He, Qian Liu, Mingzhe Du, Lin Yan, Zhijie Fan, Yiming Huang, Zejian Yuan, Zejun Ma
**Institutions:** Xi’an Jiaotong University, TikTok, National University of Singapore, University of California San Diego

## 🎯 Pitch

SWE-Perf pioneers a benchmark for evaluating whether language models can enhance the runtime efficiency of multi-file software repositories, rather than merely fixing bugs or optimizing toy functions. By revealing a significant performance gap between models and expert patches, this benchmark illuminates the challenges and opportunities for advancing automated performance optimization in software, ultimately setting the stage for substantial system-wide benefits in production environments.

---

## 1. Executive Summary
SWE-Perf introduces the first benchmark that tests whether language models can speed up real, multi-file software repositories—not just fix correctness bugs or optimize toy functions. Built from 140 performance-improving pull requests across 9 widely used Python repositories, it standardizes runtime measurement, defines clear performance metrics, and evaluates both file-level and full-repository agent systems. Results show a large gap between current models and expert patches: the best agent setup achieves 2.26% average speedup vs. 10.85% for experts (Table 2), highlighting a new, open research frontier.

## 2. Context and Motivation
- What gap is addressed
  - Repository-level performance optimization is under-evaluated. Existing benchmarks either:
    - Test repository-level correctness (e.g., bug fixing) but not efficiency; or
    - Test function-level efficiency in isolation, missing cross-file and cross-module interactions.
  - SWE-Perf fills this by measuring whether models can make meaningful runtime improvements to real repositories (Introduction; Related Work §§1–2).

- Why this matters
  - In production systems, shaving runtime often yields system-wide benefits (lower latency, cost, and energy) beyond mere correctness (Introduction).
  - Performance optimization is traditionally expert-driven and hard to automate; it depends on deep understanding of code paths, data flow, and testing infrastructure (Introduction).

- Where prior approaches fall short
  - Repository-level benchmarks (SWE-Bench, SWE-Gym, SWE-Dev) focus on correctness-constrained tasks (bug fixing, issue resolution), not open-ended speedups (Related Work).
  - Efficiency benchmarks (Mercury, EFFIBENCH, EvalPerf, KernelBench) mostly target single functions or GPU kernels, not multi-file, test-driven repos (Related Work).
  - Identifying “true” optimizations is nontrivial without reproducible environments and human reference patches; many benchmarks lack both. A concurrent dataset (GSO) mines optimization commits using an LLM judge, but does not pair each with executable environments and unit-test-based runtime evidence (Related Work).

- Positioning
  - SWE-Perf provides:
    - A rigorous data pipeline that pairs pre- and post-optimization codebases with executable Docker environments and performance tests (Figure 2; §3.2).
    - Two evaluation settings—`Oracle` (file-level) and `Realistic` (repo-level)—to test both local and end-to-end optimization capabilities (§3.1, Phase 5).
    - A conservative, statistics-based performance metric that quantifies “minimum significant speedup” (Algorithm 1; §3.2 Phase 4; §4).

## 3. Technical Approach
SWE-Perf is both a dataset and an evaluation protocol. It defines what the model sees, how patches are applied and tested, and how speedup is computed.

- Task formulation (Figure 1; §3.1)
  - Input:
    - A real open-source Python codebase.
    - A set of `target functions`—the APIs whose performance matters for evaluation.
  - Output:
    - A `patch` that modifies the codebase to run faster while preserving correctness on supplied tests.
  - Two settings:
    - `Oracle (File-Level)`: The model receives the exact files edited by the human patch and the oracle target functions. This isolates “can the model write faster code here?” from repository navigation. Targets come from the human patch via AST parsing + diff matching (§3.2 Phase 5).
    - `Realistic (Repo-Level)`: The agent receives the repository and functions measured during tests. It must navigate, find bottlenecks (which may be in callees), and edit any file. Functions are identified by dynamically profiling the tests with `yappi` and mapping to directly invoked functions via AST parsing (§3.2 Phase 5).

- Data collection pipeline (Figure 2; §3.2)
  1) Collect PRs
     - Start from popular, permissively licensed repositories used in prior SWE benchmarks (e.g., scikit-learn, xarray, sympy; Figure 3).
     - Crawl 102,241 PRs; keep PRs that resolve an issue (drop the “must contribute tests” filter to not bias performance changes; §3.2 Phase 1).

  2) Measure codebase performance
     - For each PR, obtain both “original” and “modified” codebases (pre- and post-PR).
     - Build standardized Docker images; restrict containers to one CPU core and 16 GB RAM (Phase 2; §3.2). To ensure comparability, each codebase is run three times.
     - Run all unit tests with `pytest` and record per-test runtimes (§3.2 Phase 2). This is expensive: e.g., xarray averages 220k tests and “testing a single codebase may take over one hour” on a single core (Phase 2; §3.2). Table 4 reports average per-codebase test time: 58.1 minutes for xarray and 83.9 minutes for scikit-learn.

  3) Identify performance-optimizing PRs
     - Keep unit tests that:
       - Pass on both versions (correctness).
       - Show an `optimized ratio` better than a fixed threshold (ratio ≥ 0.3 improvement). The ratio per test is:
         - Ratio = (R_original − R_modified) / R_original (Phase 3; §3.2).
     - Ensure the speedup is attributable to the patch by dynamically checking tests exercise patched code and do not rely on modified tests (§3.2 Phase 3).

  4) Verify stable improvements (statistical safeguards)
     - Add warm-up runs before timing to avoid initialization artifacts.
     - Repeat each performance test 20 times.
     - Remove timing outliers using Interquartile Range (IQR) filtering with k=1; values outside [Q1 − IQR, Q3 + IQR] are dropped (Eq. (1); §3.2 Phase 4).
     - Compute a conservative “minimum significant speedup” per test, `δ`, with Algorithm 1:
       - Gradually “weaken” the observed improvement and test whether the modified distribution remains faster than the original using a one-sided Mann–Whitney U test (p < 0.1). The largest `δ` that remains significant is kept (Algorithm 1; §3.2 Phase 4).
       - Keep only tests with δ ≥ 0.05.
     - This yields 140 high-confidence instances out of 1,696 preliminary ones (Table 3).

  5) Extract optimization targets (Phase 5; §3.2)
     - `Oracle`: Trace target functions from human patch via AST + diff; provide entire files as context.
     - `Realistic`: Trace directly called functions in the performance tests via `yappi` profiling + AST, and present those to the agent (not the tests themselves, to avoid test overfitting or “functional pruning”).

- Dataset characteristics (Figure 3; Table 1; §3.3)
  - 140 instances across 9 repos; xarray (54) and scikit-learn (32) dominate (Figure 3).
  - Average non-test code per repo: 447 files and 170k lines.
  - Expert patches are nontrivial: 131 lines edited on average across 4.3 files and 7.6 functions.
  - Each instance has 8.1 related performance tests on average.
  - Average “original” test runtime is 0.28 s per test (max 25.2 s), but per-codebase test suites are large (Table 4).
  - Final mean performance ratio of expert patches across tests: 10.9% (Table 1).

- Evaluation protocol (Section 4; Eq. (2))
  - After applying a model’s patch to the original codebase, run the performance-related tests in the same Docker environment and collect:
    - `Apply`: fraction of patches that apply cleanly.
    - `Correctness`: fraction of instances where all performance tests pass post-patch.
    - `Performance`: average of per-test conservative speedups `pi,j` (δ from Algorithm 1) across tests and instances:
      - Performance = (1/N_total) Σ_i P_i, where P_i = (1/n_i) Σ_j p_i,j (Eq. (2)).
  - To neutralize environment drift, the original codebase is re-measured during evaluation even if data collection timings exist (Section 4).

- Baselines and settings (Section 5; Appendix C)
  - `Oracle` direct-edit prompting evaluates 10 popular models (e.g., `Claude-4-sonnet`, `GPT-4o`, `OpenAI-o3`, `Gemini-2.5-Pro`) with a structured Search/Replace patch format (Figure 14).
  - `Agentless` (pipeline-based) and `OpenHands` (agent-based) run in the `Realistic` setting, both with `Claude-3.7-sonnet` as the backend model; OpenHands allows up to 50 iterations (Section 5.1; Appendix C).

## 4. Key Insights and Innovations
- A repository-scale, runtime-grounded benchmark for performance
  - Unlike correctness-only or function-only datasets, SWE-Perf pairs pre/post repositories with executable Docker environments and unit-test workloads, then verifies speedups with multiple safeguards (Figure 2; §3.2). This makes “optimization” measurable and comparable across repos.

- Conservative, statistically sound speedup metric (`δ`)
  - Algorithm 1 embeds a “minimum significant improvement” search using the Mann–Whitney U test, ensuring the reported gain would withstand conservative down-adjustments (Algorithm 1; §3.2 Phase 4). This reduces false positives from noisy timings—an advance over simple mean-difference metrics.

- Dual evaluation settings that separate skills
  - `Oracle` probes whether models can write faster code when pointed to the right files; `Realistic` probes repository navigation, profiling, and multi-step planning (§3.1, Phase 5). This disambiguates “can’t find bottleneck” from “can’t optimize code.”

- Target-function extraction without leaking test specifics
  - In the `Realistic` setting, target functions are derived via dynamic profiling (`yappi`) and AST parsing rather than exposing tests directly (§3.2 Phase 5). This mitigates “functional pruning,” where a model speeds up only the exercised subset of behavior.

- Reproducibility at scale
  - Standardized Docker, resource limits (one CPU core and 16 GB RAM during Phase 2; five cores later for Phase 4/evaluation), warm-ups, 20 repeats, and IQR filtering collectively create a careful timing environment (Phase 2–4; §3.2). Few prior SWE datasets tie this many controls together for performance.

## 5. Experimental Analysis
- Evaluation setup (Section 4; Section 5)
  - Metrics: `Apply`, `Correctness`, `Performance` (Eq. (2)).
  - Settings:
    - `Oracle`: 10 models generate file-level patches with a fixed prompt format (Figure 14).
    - `Realistic`: `OpenHands` (agent-based) and `Agentless` (pipeline-based), both with `Claude-3.7-sonnet`.

- Main quantitative results (Table 2; Figure 4)
  - Overall ceiling:
    - Quote: “Expert 100.00% Apply, 100.00% Correctness, 10.85% Performance” (Table 2).
  - Oracle setting (file-level):
    - Best `Correctness` among evaluated models: `Gemini-2.5-Pro` at 83.57% (Apply 95.00%).
    - Best `Performance` among Oracle models: `Claude-4-sonnet` at 1.76%; others range roughly 0.41%–1.48% (Table 2).
    - This shows that even when given the relevant files, current models’ average speedups are an order of magnitude below expert patches.
  - Realistic setting (repo-level):
    - `OpenHands (Claude-3.7-sonnet)` achieves 87.86% Apply, 77.86% Correctness, and 2.26% Performance.
    - `Agentless (Claude-3.7-sonnet)` reaches 88.57% Apply, 70.71% Correctness, but only 0.41% Performance (Table 2).
    - Agent-based iteration and tool use (OpenHands) translate to notably higher speedups than a pipeline approach.
  - Per-repository differences:
    - Figure 4 shows heterogeneity by repo. Notably, “on sklearn, OpenHands outperforms the Expert by 0.4%,” demonstrating that agents can sometimes exceed the human patch (Figure 4; §5.2).

- Controlling for correctness (Figures 5–6)
  - To isolate “optimization skill” from “failure to compile or pass tests,” the study recomputes Performance only on examples where the model’s patch passes all tests.
    - Quote (Figure 5): “Performance on correct examples: Oracle ≈ 2.0%, Agentless ≈ 0.6%, OpenHands ≈ 2.9%; corresponding Expert ceilings on those subsets ≈ 8.3%, 8.8%, 11.4%.”
    - Figure 6 repeats this per Oracle model; even the strongest Oracle runs are far below the expert ceilings for the same subsets (e.g., model ≈ 2.5% vs. expert ≈ 12.2% for one configuration).

- What makes some instances harder (Figures 7–9)
  - More target functions → harder:
    - Quote (Figure 8): “As the number of `Realistic` target functions increases, expert performance declines (lower ceiling), and the gap between OpenHands and Expert widens.” This suggests models struggle with multi-function, cross-module optimization.
  - Longer baseline runtimes → higher potential, but models plateau:
    - Quote (Figure 9): “Expert performance climbs with runtime, whereas models flatten,” implying experts exploit longer-running workloads better (e.g., algorithmic changes) while models may rely on smaller, local tweaks.

- Qualitative patch analysis (Figures 10–13)
  - Word clouds of added lines:
    - OpenHands patches emphasize low-level structure/infra terms (“children”, “identifier”, “attributes”, “time”) (Figure 10).
    - Expert patches emphasize type/dtype, values, labels—suggesting higher-level refactoring and data semantics awareness (Figure 11).
    - This aligns with the quantitative gap: models often perform micro-optimizations; humans more often restructure abstractions that unlock larger gains.

- Dataset scale and rigor (Tables 3–5)
  - The pipeline prunes from 102,241 PRs to 140 final instances through multiple runtime-based filters (Table 3).
  - Runtime costs are substantial: average test runtimes per codebase reach 58–84 minutes for xarray and scikit-learn (Table 4), underlining the need for careful experimental design.

- Do the experiments support the claims?
  - Yes. The multi-metric design (Apply, Correctness, Performance), statistical speedup algorithm, per-repo analysis, correctness-controlled plots, and function/runtime stratifications together substantiate:
    - There is a large performance gap to experts (Table 2).
    - Agent-based methods better translate model capability into realized speedup (Table 2; Figure 5).
    - Difficulty increases with target-function count and with the kind of long-runtime cases where experts excel (Figures 8–9).

## 6. Limitations and Trade-offs
- Dataset scope and diversity
  - The final benchmark covers 9 popular Python repos and 140 instances (Figure 3; §3.3). This is strong for reproducibility but may underrepresent other languages, domains (e.g., C++/Rust), or GPU-heavy workloads (Appendix A.1).

- Upper bound uncertainty
  - “Expert patches” are drawn from real PRs and “may not represent the optimal achievable performance” (Appendix A.1). The 10.85% average ceiling could be below the true optimum.

- Test selection and evaluation scope
  - Performance is measured only on selected “performance-related” tests; other repository behaviors and untested paths might regress unnoticed (§3.1). This is mitigated by correctness checks on those tests but not a full test suite.

- Target guidance vs. full autonomy
  - Even the `Realistic` setting supplies the list of target functions (derived via profiling), not a fully unconstrained repo-wide objective (§3.1; §3.2 Phase 5). Full autonomy remains future work.

- Statistical choices
  - The Mann–Whitney U test uses p < 0.1 and `δ` ≥ 0.05 thresholds with a 0.01 step size (Algorithm 1; §3.2 Phase 4). Different thresholds could shift inclusion/exclusion of instances.

- Compute and reproducibility cost
  - Building images and running tens of thousands of tests repeatedly is expensive (Table 4), limiting the pace of iteration and making large-scale training with this protocol challenging.

- CPU-only environment
  - Timing is CPU-constrained (one core and 16 GB in Phase 2; five cores later), which is appropriate for many Python repos but does not represent GPU/accelerator contexts (§3.2).

- Avoiding test leakage vs. realism
  - Not exposing tests to prevent “functional pruning” is sound (§3.2 Phase 5), but also means models don’t interact with the exact performance harness developers typically use.

## 7. Implications and Future Directions
- Impact on the field
  - SWE-Perf reframes “SWE agents” from correctness-only to performance-aware development. It establishes a practical way to measure repository-scale speedups with statistical rigor, enabling apples-to-apples comparisons.

- Research directions enabled
  - Agent design:
    - Integrate dynamic profilers and cost models directly into agent loops for targeted optimization under the `Realistic` setting.
    - Explore hierarchical plans that move beyond local edits toward cross-module refactoring (Figures 10–11 suggest this gap).
  - Learning from experts:
    - Mine patterns from expert patches (types/dtypes, label semantics, algorithm changes) to train models that prefer high-leverage changes.
  - Multi-function and long-runtime cases:
    - Develop planning, retrieval, and verification strategies that scale with the number of target functions (Figure 8) and exploit high-runtime opportunities (Figure 9).
  - Robust performance metrics:
    - Extend `δ` with effect-size measures or Bayesian models; adapt thresholds to balance inclusion with confidence.
  - Broader coverage:
    - Port the pipeline to other languages, GPU/accelerator workloads (e.g., blending with KernelBench ideas), and system-level performance (I/O, memory, parallelism).

- Practical applications
  - Bench-testing AI coding assistants for speed, not just correctness.
  - Continuous integration hooks that propose performance patches and verify gains statistically.
  - Education and training: curated instances to teach performance-aware refactoring patterns.

Quote highlights grounding key results and mechanisms:
- “Expert 100.00% Apply, 100.00% Correctness, 10.85% Performance” (Table 2).
- “OpenHands … 77.86% Correctness, 2.26% Performance” vs. “Agentless … 70.71% Correctness, 0.41% Performance” (Table 2).
- “As the number of Realistic target functions increases … the gap … widens” (Figure 8).
- “Expert performance climbs with runtime, whereas models flatten” (Figure 9).
- Conservative speedup via Algorithm 1: Mann–Whitney U (p < 0.1) with δ search (Algorithm 1; §3.2 Phase 4).

Bottom line: SWE-Perf provides the first rigorous, repository-scale proving ground for performance optimization by language models. Current agents show promise—occasionally matching or beating humans on certain repos (Figure 4)—but average speedups remain far from expert-level, especially in multi-function and long-runtime scenarios. The benchmark’s design makes it possible to measure and close that gap systematically.
