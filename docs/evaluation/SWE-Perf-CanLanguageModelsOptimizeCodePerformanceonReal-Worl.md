# SWE-Perf: Can Language Models Optimize Code Performance on Real-World Repositories?

**ArXiv:** [2507.12415](https://arxiv.org/abs/2507.12415)

## 🎯 Pitch

SWE-Perf introduces the first benchmark to systematically evaluate language models on their ability to optimize code performance at the scale of full, real-world software repositories—moving beyond existing tests of code correctness or toy function tuning. By curating 140 authentic, performance-improving GitHub pull requests with reproducible execution environments and expert-written speedup patches, SWE-Perf reveals a significant gap between current LLM capabilities and human expertise, emphasizing both the challenge and critical importance of automated efficiency improvements for production software. This benchmark sets a new standard for measuring and advancing AI-driven performance optimization at the system level.

---

## 1. Executive Summary (2–3 sentences)
SWE-Perf introduces the first benchmark that evaluates whether language models can speed up real code in whole repositories, not just fix bugs or optimize toy functions. Built from 140 real performance-improving GitHub pull requests across 9 widely used projects, it provides executable Docker environments, target functions, and expert patches, and measures statistically significant runtime gains; results show a large gap between current LLM systems (best ≈2.26% gain) and expert edits (≈10.85% gain) (Table 2).

## 2. Context and Motivation
- Problem addressed
  - Most LLM benchmarks for software engineering check correctness (do tests pass?) rather than efficiency (how fast does the code run?). Repository-level performance optimization requires cross-file reasoning, API awareness, and safe refactoring—capabilities that current datasets and systems do not directly assess (§1, §2).
- Why this matters
  - In production systems, faster code reduces compute cost, latency, and energy use. Prior work shows efficiency can deliver system-wide benefits (e.g., energy/process efficiency studies) that correctness alone cannot unlock (§1, citations).
- Prior approaches and shortcomings
  - Function-level efficiency datasets (e.g., Mercury, EFFIBENCH, EvalPerf, KernelBench) focus on isolated algorithmic snippets, missing inter-module dependencies and repository-scale trade-offs (§2 “Code Efficiency”).
  - Repository-level SWE benchmarks (SWE-Bench, SWE-Gym, SWE-Dev, etc.) target bug fixing or test-oriented objectives with crisp ground truth, not the open-ended task of improving runtime performance (§2 “Repository-Level SWE Tasks”).
  - Without human reference implementations, it is hard to know whether further optimization is possible; existing resources lack reproducible environments and clear tests of statistically significant speedups (§1).
- How this work positions itself
  - SWE-Perf bridges the gap by (1) extracting real performance-improving PRs, (2) recreating runnable environments (Docker), (3) supplying expert patches as a reference “ceiling,” (4) offering both file-level (“oracle”) and repo-level (“realistic”) targets, and (5) using rigorous statistics to validate repeatable speedups (Figures 1–2; §3–§4).

## 3. Technical Approach
SWE-Perf is a dataset and evaluation framework built in five phases (Figure 2), plus a task setup with two evaluation settings and three metrics.

- Key terms defined
  - Pull request (PR): a proposed set of code changes submitted to a repository for review and merge.
  - Unit test: a small automated test that verifies a specific function or behavior.
  - Docker: a container system that packages an application and its dependencies into a reproducible environment.
  - Oracle vs. Realistic settings: “Oracle” gives the model the exact files/functions humans edited; “Realistic” gives only functions executed by tests and lets an agent explore/modify the whole repository (§3.2 Phase 5).

Step-by-step data pipeline (Figure 2; §3.2)
1) Collect PRs from high-star repos
   - Start from the 12 SWE-Bench repositories; crawl PRs and keep those that “resolve an issue.” Unlike SWE-Bench, do not require PRs to contribute tests, because the goal is speed, not changing correctness (§3.2 Phase 1).
   - Scale: 102,241 PRs collected; 19,797 remain after initial filtering (Appendix Table 3).

2) Measure performance of both codebases per PR
   - For each PR, there are two codebases: original (pre-PR) and modified (post-PR).
   - Build Docker images/containers and run all unit tests with pytest. Constrain containers to 1 CPU core and 16 GB RAM for comparability (§3.2 Phase 2; note: Phase 4/evaluation sometimes use 5 cores—see Limitations).
   - Record runtimes per test, with three replicates to reduce noise (§3.2 Phase 2).
   - Scale: 34,397 codebases built; runtimes for 19,499 codebases captured (Appendix Table 3). Per-repo timing examples show substantial variability (Appendix Table 4), e.g., scikit-learn tests take on average 83.9 minutes per codebase.

3) Identify performance-optimizing PRs (§3.2 Phase 3)
   - Filter for unit tests that:
     - Pass in both original and modified codebases (“Correctness” filter).
     - Show a substantial improvement ratio. The text gives a formula as an improvement fraction Ratio = (Roriginal − Rmodified)/Roriginal and mentions a threshold “below 0.3.” This appears inconsistent with the goal of “substantial improvement” and may be a typographical mix-up with a “modified/original” ratio; see Limitations for discussion.
   - Dynamic execution check: Keep only tests that actually execute the patched code and exclude tests that themselves were modified in the PR (§3.2 Phase 3, Step 2).

4) Verify stable performance improvements (§3.2 Phase 4)
   - Warm-up: Run 3 preliminary invocations to stabilize caches/startup effects.
   - Repeat: Run each test 20 times to obtain a distribution of runtimes.
   - Outlier removal: Interquartile Range (IQR) filtering with k=1 removes outlier runs; a run ri is an outlier if ri < Q1 − IQR or ri > Q3 + IQR (Equation (1)).
   - Statistical gain δ: Compute a conservative minimum significant speedup for each test (Algorithm 1). Mechanism:
     - Intuition: Gradually reduce the observed speedup by scaling the post-patch runtimes upward by (1 − x). Find the largest x for which the speedup remains statistically significant using a Mann–Whitney U test (non-parametric test of distribution shift) with one-sided alternative (“greater”) and p < 0.1.
     - Output: δ is that largest x; if δ > 0.05, the improvement is considered stable/significant and the instance is kept.
   - Result: 140 final instances across 9 repositories (Figure 3; Appendix Table 3).

5) Extract optimization targets (Oracle vs. Realistic) (§3.2 Phase 5)
   - Oracle (file-level): Identify the exact functions humans edited in the PR using unified-diff + AST matching. Provide those functions and their entire files to the model so it can optimize “in place.”
   - Realistic (repo-level): Identify functions actually invoked by the performance tests via dynamic profiling with `yappi`, combined with test AST parsing to find direct callees. Do not use the test code itself as a target to avoid “functional pruning” (removing functionality not covered by tests).
   - Scale characteristics: Averages per instance—Oracle targets ≈7.6 functions; Realistic targets ≈30.1 functions (Table 1).

Task setup and evaluation (§4)
- Inputs per instance: original codebase, target functions, performance-related tests, and Docker environment.
- Outputs: a patch to apply to the original repository.
- Metrics (computed by re-running tests in the provided Docker container to remove environment drift):
  - Apply: was the patch applied cleanly? (percentage of instances).
  - Correctness: do all performance-related tests pass after the patch?
  - Performance: average conservative minimum gain across tests and instances:
    - For each test j in instance i, compute pi,j using Algorithm 1 after warm-up, 20 repetitions, and outlier filtering.
    - Average per instance Pi = (1/Ni) Σj pi,j, then average over all instances (Equation (2)).

Baselines and systems evaluated (§5.1; Appendix C)
- Oracle (direct model) setup: Chain-of-thought style prompt; outputs code edits in a required “SEARCH/REPLACE” patch format (Appendix Figure 14). Restrictions: cannot edit tests; only standard Python libraries.
- Agentless (pipeline-based): A staged workflow (e.g., localization → repair → selection) built for bug fixing, repurposed here for performance (Xia et al., 2024).
- OpenHands (agent-based): An autonomous coding agent platform with iterative reasoning and repository navigation (Wang et al., 2024). Max 50 iterations in experiments.
- Models in Oracle setting: Claude-3.7/-4 sonnet/opus, GPT-4o, OpenAI o1/o3 variants, DeepSeek V3/R1, Gemini-2.5-Pro, Qwen3-235B (Table 2).
- For Agentless/OpenHands, the backend LLM is `Claude-3.7-sonnet` as recommended by OpenHands (§5.1).

## 4. Key Insights and Innovations
- A repository-level performance benchmark with executable environments (fundamental)
  - What’s new: 140 instances drawn from real PRs across 9 major repos, each with full codebase, Docker, target functions, tests, runtime logs, and the expert patch (Figure 1; §3.3, Table 1).
  - Why it matters: It enables reproducible, end-to-end evaluation of performance optimization in realistic settings—beyond toy kernels or isolated functions.

- A rigorous pipeline to identify and verify true runtime gains (fundamental)
  - What’s new: Systematic PR filtering, dynamic coverage to ensure tests hit patched code, 20-run repetitions, IQR outlier filtering, and the δ statistic via Mann–Whitney U to quantify a conservative significant speedup (Figure 2; §3.2; Algorithm 1; Equation (1)).
  - Why it matters: Guards against noise, flakiness, and spurious “optimizations,” anchoring results in statistically supported improvements.

- Two complementary evaluation regimes: Oracle vs. Realistic (capability-scoped)
  - What’s new: File-level Oracle isolates code-quality optimization ability, while repo-level Realistic tests retrieval, navigation, planning, and multi-file refactoring (Phase 5).
  - Why it matters: Separates “can you write faster code when shown the right place?” from “can you find the right place and coordinate changes across the repo?”

- An interpretable three-tier metric suite (incremental but important)
  - Apply, Correctness, Performance together reveal where systems fail: patch application, regression introduction, or lack of speedup (§4).
  - This encourages method development that addresses all stages, not only runtime.

## 5. Experimental Analysis
- Evaluation design
  - Dataset: 140 instances from 9 repos; on average 447 non-test files and 170k lines per repo; expert patches touch ≈131 lines across ≈4.3 files and ≈7.6 functions (Table 1).
  - Metrics: Apply, Correctness, and Performance (Equation (2)).
  - Systems: Oracle with 10 LLMs; Realistic with Agentless and OpenHands (both backed by `Claude-3.7-sonnet`) (§5.1).
  - Environment: Dockerized single-core CPU/16GB RAM for Phase 2; warm-up and 20 repetitions with IQR filtering for performance measurement (§3.2; §4).

- Main quantitative results (Table 2; Figure 4)
  - Reference expert performance:
    - “Expert” (human patch) yields Apply 100%, Correctness 100%, Performance 10.85%.
  - Oracle (file-level, direct patching):
    - Best Performances are small: `Claude-4-sonnet` 1.76%, `Gemini-2.5-Pro` 1.48%, `OpenAI o3` 1.37%; Apply 54–95%, Correctness 43–84%.
    - Many strong models still achieve <1% average gain (e.g., GPT-4o 0.60%, DeepSeek-V3 0.54%).
  - Realistic (repo-level agents):
    - `OpenHands` achieves the strongest overall Performance among LLM systems: 2.26% with Apply 87.86% and Correctness 77.86%.
    - `Agentless` is substantially lower on Performance: 0.41% (Apply 88.57%, Correctness 70.71%).
  - Per-repository behavior (Figure 4):
    - Performance varies widely by repo. Notably, on scikit-learn, OpenHands slightly surpasses the expert by 0.4%, showing isolated cases where agents can exceed human patches.

- Does the evidence support the claims?
  - The large gap between expert (10.85%) and best model (2.26%) convincingly shows current limitations for repo-scale performance optimization (Table 2).
  - The methodology builds a strong case that measured gains are real: re-run original and post-patch under the same container, warm-up, 20× repeats, IQR filtering, and non-parametric significance testing with δ (Algorithm 1; §4).

- De-coupling performance from correctness (Figures 5–6; §5.3.1)
  - When measuring only on instances where patches keep tests passing (i.e., “correct” cases), OpenHands still outperforms other methods by roughly 3 percentage points, indicating its iterative agent loop better converts opportunities into speedups near the attainable ceiling.

- Difficulty grows with the number of functions (Figures 7–8; §5.3.2)
  - As the number of target functions rises, both expert and model performance drop, but the model’s drop is steeper. In multi-function scenarios, the performance gap widens, implying challenges in planning and dependency management across call graphs.

- Runtime-length matters (Figure 9; §5.3.3)
  - Longer original runtimes present higher potential ceilings, and experts exploit this with steadily increasing gains. Model performance, however, plateaus for long-running tests, suggesting current LLM systems fail to uncover deeper algorithmic or architectural wins under heavier workloads.

- Qualitative patch analysis (Figures 10–11; Appendix Figures 12–13; §5.3.4)
  - Word clouds of added lines show models gravitating to lower-level infrastructure tweaks (e.g., attributes, identifiers, environment/tooling hints), while experts bias toward higher-level abstractions (types, dtypes, labels) that affect algorithmic structure and data flow. This hints that models often “tune around” code rather than redesigning critical paths.

- Additional statistics and scale (Appendix Tables 3–5)
  - The pipeline is compute-heavy: e.g., average single-codebase pytest runtime ≈84 minutes for scikit-learn; xarray ≈58 minutes (Appendix Table 4).
  - The data funnel is steep—102k PRs to 140 instances—underscoring how rare, measurable, and stable performance improvements are in the wild (Appendix Table 3).

## 6. Limitations and Trade-offs
- Dataset scope and representativeness
  - The benchmark currently covers 9 repositories and 140 instances (§3.3; Appendix A.1). While carefully curated, it may not represent all domains (e.g., systems programming, GPU code, mobile apps).
- Human patch as “ceiling”
  - Expert patches show feasibility but are not guaranteed to be optimal; the true attainable speedup could be higher (Appendix A.1).
- CPU-only, runtime-only metric
  - Evaluations constrain CPU and memory; they do not measure energy, memory usage, or GPU performance (Table 4; §4). Gains that trade CPU for memory or vice versa are not captured.
- Statistical thresholds and environment details
  - δ uses a relatively lenient significance level (p < 0.1) and δ > 0.05 cutoff (Algorithm 1; §3.2 Phase 4). This is reasonable for noisy real systems but still a design choice.
  - The paper mentions 1 CPU core for comparability in Phase 2, but also “5 CPU cores in Phase 4 and evaluation” (§3.2 Phase 2). This discrepancy could affect cross-phase comparability; clarification is desirable.
  - An apparent inconsistency in Phase 3’s improvement ratio criterion (text says ratio must be “below 0.3” while the given formula is an improvement fraction) could confuse replication (§3.2 Phase 3, Step 1).
- Restrictions in Oracle prompting
  - Oracle prohibits editing tests and using non-standard libraries (Appendix Figure 14). This avoids test leakage and dependency drift but prevents some realistic optimization strategies (e.g., switching to a faster library).
- Agent systems and budget
  - OpenHands is capped at 50 iterations (Appendix C). Larger search budgets or richer tool use might yield higher gains but at greater compute cost.

## 7. Implications and Future Directions
- How the landscape changes
  - SWE-Perf shifts evaluation from “does it pass tests?” to “can it make production code faster reproducibly?” with statistically credible measurements. It creates a common ground for comparing agent systems, direct prompting, and future methods on realistic performance objectives (Figures 1–2; §4–§5).
- Research opportunities enabled
  - Multi-function, cross-module planning: Figure 8 highlights the growing difficulty as targets multiply; research on call-graph construction, profile-guided planning, and hierarchical refactoring agents is a natural next step.
  - Longer-runtime optimization strategies: Figure 9 shows a plateau for models on heavy workloads; hybrid approaches combining static analysis, profilers, and algorithmic reasoning could close this gap.
  - Better search and verification loops: Integrate performance profilers, memoized benchmarks, and statistical decision procedures (like δ) directly into agent inner loops to guide edits.
  - Richer objectives: Extend metrics beyond CPU runtime to memory, I/O, energy, and cost; add regression guards for numerical stability and precision under optimization.
  - Broader corpora and modalities: Include systems-level repos, GPU kernels, data engineering pipelines, and language ecosystems beyond Python; enable multi-language optimization scenarios.
- Practical applications
  - CI/CD performance gating: Use SWE-Perf-style δ checks to block performance regressions and to validate automated optimization PRs in real projects.
  - Developer copilots for optimization: Train/evaluate agents that propose safe, profiled patches with confidence estimates; triage hotspots, suggest vectorization or batching, and refactor data layouts.
  - Educational tooling: Provide “before/after” expert patches and measured gains as case studies for teaching performance engineering at the repository scale.

Quotes of key results/claims
- “Expert” reference performance: 
  > Table 2: Apply 100.00%, Correctness 100.00%, Performance 10.85%.
- Best LLM system (repo-level):
  > Table 2: OpenHands (Claude-3.7-sonnet) — Apply 87.86%, Correctness 77.86%, Performance 2.26%.
- Oracle model ceiling (file-level):
  > Table 2: Best Oracle Performance 1.76% (Claude-4-sonnet), with Apply 73.57% and Correctness 70.00%.
- Per-repo standout:
  > Figure 4: On scikit-learn, OpenHands slightly outperforms Expert by 0.4%.
- Difficulty with many targets:
  > Figures 7–8: Performance declines as the number of target functions increases; the model’s gap to Expert widens.
- Heavy-run potential:
  > Figure 9: Expert performance rises with increasing original runtime; model performance plateaus.

Overall, SWE-Perf establishes both a dataset and a methodology to study an under-explored but crucial ability: making real repositories faster. The benchmark’s careful construction and sobering results clarify where LLMs and agents fall short today, and they point to concrete avenues—profiling-guided planning, multi-function reasoning, and deeper algorithmic refactoring—for tomorrow’s systems to close the expert gap.
