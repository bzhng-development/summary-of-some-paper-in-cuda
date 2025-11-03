# RPG: A Repository Planning Graph for Unified and Scalable Codebase Generation

**ArXiv:** [2509.16198](https://arxiv.org/abs/2509.16198)
**Authors:** Jane Luo, Xin Zhang, Steven Liu, Jie Wu, Yiming Huang, Yangyu Huang, Chengyu Yin, Ying Xin, Jianfeng Liu, Yuefeng Zhan, Hao Sun, Qi Chen, Scarlett Li, Mao Yang
**Institutions:** (not specified in arXiv metadata)

## 🎯 Pitch

This paper presents the Repository Planning Graph (RPG), a novel structured graph that replaces ambiguous natural language plans with a persistent, executable blueprint for generating entire software repositories. By operationalizing RPG through the ZeroRepo system, the study achieves significantly higher functionality coverage and accuracy, offering a breakthrough in automated software engineering. The impact lies in making AI-driven, end-to-end software development more coherent and scalable, which could revolutionize research, prototyping, and industry processes.

---

## 1. Executive Summary (2-3 sentences)
The paper introduces the Repository Planning Graph (`RPG`), a structured graph that replaces free-form natural-language plans with an explicit, evolving blueprint for generating entire software repositories. Built on `RPG`, the `ZeroRepo` system plans, implements, and tests codebases from high-level goals, achieving substantially higher functionality coverage, correctness, and scale than strong agentic coding baselines on a new benchmark, `RepoCraft` (Table 2, §5–6).

## 2. Context and Motivation
- Problem addressed
  - Repository-level code generation from scratch remains unreliable: current agent systems are good at writing single functions or files, but they struggle to transform vague, high-level intent into a coherent, multi-file repository with consistent interfaces and dependencies (§1).
  - Existing systems plan in natural language; over long horizons this leads to ambiguity, drifting specifications, misaligned modules, and brittle dependencies (§1, §2).

- Why it matters
  - Practical impact: automated construction of coherent software systems from high-level specs would unlock end-to-end “AI software engineer” workflows for research, prototyping, and industry (§1).
  - Theoretical significance: it tests whether long-horizon planning can be made stable and extensible for software, an archetypal structured reasoning problem (§1, §7.1).

- Prior approaches and their shortcomings
  - Multi-agent role systems (e.g., `MetaGPT`, `ChatDev`), workflow pipelines (e.g., `Paper2Code`, `AutoP2C`), and terminal agents (`OpenHands`, `Claude Code`, `Gemini CLI`) externalize plans as markdown/text (§2).
  - Limitations of NL planning documented in §1–§2: ambiguity between intent and constraints, difficulty tracking dependencies without an explicit hierarchy, and compounding drift over long iterations, causing incomplete/overlapping features and inconsistent data flows.

- Positioning of this work
  - The paper replaces free-form plans with a persistent, executable graph—`RPG`—that unifies proposal-level planning (“what to build”) with implementation-level planning (“how to build it”) by encoding capabilities, files, classes, functions, and typed data flows in one structure (§3.1, Fig. 2).
  - On top of `RPG`, `ZeroRepo` operationalizes three graph-centric stages: proposal construction, implementation refinement, and graph-guided generation with test validation (Fig. 1).

## 3. Technical Approach
This section explains the end-to-end pipeline, from high-level goal to running repository.

- Core representation: Repository Planning Graph (`RPG`) (§3.1; Fig. 2)
  - Nodes are dual-purpose:
    - Functional hierarchy: high-level modules (e.g., “algorithms”, “evaluation”) down to concrete features.
    - Structural mapping: root nodes align to folders/regions; intermediate nodes to files; leaf nodes to functions/classes.
  - Edges encode execution semantics:
    - Inter-module data-flow edges (black arrows in Fig. 2) specify typed inputs/outputs (e.g., “Data Loading → Algorithms → Evaluation”).
    - Intra-module order edges (dashed gray, Fig. 2) impose file-level sequencing.
  - The resulting DAG induces a topological order that simultaneously respects functional dependencies and code organization.

- Stage A: Proposal-level construction (Fig. 1A; §3.2; Appendix A)
  - Stabilizing knowledge base: EpiCoder Feature Tree (1.5M+ capabilities, 7 levels; Table 5, Appendix A.2). Each feature node is embedded and indexed in a vector database with path metadata to preserve hierarchy (§3.2).
  - Explore–Exploit subtree building:
    - Exploit: retrieve top‑k feature paths aligned to the user’s goal; allow keyword augmentation via LLM (§3.2).
    - Explore: sample unvisited ontology regions to capture less obvious but relevant capabilities (§3.2).
    - Diversity-aware rejection sampling (Algorithm 1) accepts candidate trees with low overlap to expand breadth; otherwise picks the least-overlapping candidate after Tmax retries (Appendix A.1).
    - LLM filters, ranks, and proposes truly missing features; batch self-check then integrates accepted paths (Algorithm 2, Appendix A.1).
  - Refactoring into a repository-aligned functionality graph:
    - The selected subtree is reorganized by cohesion/coupling into coherent modules (e.g., clustering metrics under “evaluation”, not “clustering”) to form the initial, modular functionality graph (§3.2).

- Stage B: Implementation-level construction (Fig. 1B; §3.3; Appendix B)
  - File structure encoding (§3.3.1):
    - Folder-level: assign each top-level module a directory namespace; descendants inherit a consistent namespace (e.g., `algorithms/`, `evaluation/`).
    - File-level: map intermediate nodes to specific files (e.g., `preprocess.py`, `linear_models.py`) to reduce cross-file coupling and anchor code generation.
  - Data-flow and function/interface encoding (§3.3.2):
    - Encode typed inter- and intra-module flows as input–output constraints to finalize the DAG and drive interface design (Fig. 2).
    - Abstract global interfaces: factor common patterns into shared data structures or base classes (e.g., `BaseEstimator`) to maintain consistency across modules.
    - Map leaf features to executable interfaces:
      - Independent features → standalone functions.
      - Interdependent features → methods of a shared class (e.g., `DataLoader` with `load_csv` and `load_json`).
    - The result is the full `RPG`: a file-augmented, function-augmented graph with explicit flows and contracts.

- Stage C: Graph-guided code generation (Fig. 1C; §4; Appendix C)
  - Topological traversal: generate and validate nodes in dependency order so upstream contracts exist before downstream use (§4).
  - Test-driven development (TDD):
    - For each leaf node, derive a unit test from the task/specification or docstring; implement code until tests pass; only then commit code (§4 “Graph-Guided Code Generation” and “Graph-Guided Test Validation”).
    - Regression and integration tests: modified components trigger regression tests; completed subgraphs undergo integration tests to confirm flows and contracts (§4; Algorithm 3 and Algorithm 4 in Appendix C.4).
    - Majority-vote diagnosis attributes failures to implementation vs test/environment, auto-remediating when it’s the latter (§4).
  - Graph-guided localization and editing:
    - Tools support search-by-functionality over the `RPG`, full code retrieval, and dependency tracing (Appendix C.1).
    - Once localized, targeted edits are applied with structured edit tools (function/method/class/import edits; Appendix C.2).
    - Example trajectory shows step-by-step localization and patching for a symbolic calculus task (Appendix C.3).

- Why this approach?
  - Replacing natural-language plans with a persistent graph enforces explicit structure and dependencies, preventing drift and misalignment over long horizons (§1–§3).
  - Grounding planning in a large, hierarchical ontology plus explore–exploit ensures both coverage and diversity (Appendix A.2; Fig. 7–11).
  - Topological TDD plus regression/integration tests ensures incremental stability (§4; Algorithms 3–4).

## 4. Key Insights and Innovations
- `RPG`: a unified, persistent planning substrate
  - Novelty: integrates proposal-level (“what”) and implementation-level (“how”) planning in one DAG that ties functionality to folders/files/classes/functions with typed flows (Fig. 2; §3.1).
  - Significance: enables consistent, long-horizon planning with explicit dependencies and ordering—something prior NL-only plans could not guarantee (§1–§2). This is a fundamental representational shift.

- Graph-driven repository synthesis (`ZeroRepo`)
  - Novelty: three-stage pipeline that constructs an `RPG`, then traverses it with TDD, regression, and integration testing (Fig. 1; §3–§4).
  - Significance: empirically yields much higher coverage and correctness than strong baselines while scaling to larger codebases (Table 2; §6; Fig. 5–6). This is more than an incremental tweak; it re-architects the planning medium and the generation loop.

- Ontology-grounded planning with explore–exploit search
  - Novelty: uses a 1.5M+ node Feature Tree (Table 5) as a structured prior plus diversity-aware rejection sampling and LLM self-checks to stabilize capability selection (Algorithms 1–2; §3.2; Appendix A).
  - Significance: avoids LLM enumeration myopia; supports near-linear growth in planned features across iterations (Fig. 5) and healthier functional distributions than the long-tailed global ontology (Appendix A.2, Fig. 8–11).

- `RepoCraft`: benchmark and evaluation pipeline for repo-level generation
  - Novelty: six real-world Python projects with 1,052 tasks derived from curated test suites; metrics for coverage, novelty, and correctness; and a localization–validation–execution pipeline calibrated via “Gold Projects” (§5.1–§5.3; Table 1–2; Fig. 3; Appendix D).
  - Significance: enables end-to-end, from-scratch evaluation of repository planning/generation—moving beyond single-file metrics. The “Gold Projects” calibration indicates the automated evaluation’s ceiling (81% pass / 92% voting; Table 2).

## 5. Experimental Analysis
- Evaluation methodology (§5; Appendix D)
  - Benchmark: `RepoCraft` with six anonymized, paraphrased repositories—`scikit-learn` (MLKit-Py), `pandas` (TableKit), `sympy` (SymbolicMath), `statsmodels` (StatModeler), `requests` (HttpEasy), `django` (PyWebEngine). Table 1 lists category counts, file counts, LOC, tokens, and task counts.
  - Metrics (Appendix D.3.1):
    - Functionality Coverage: fraction of reference categories represented at least once.
    - Functionality Novelty: fraction of generated functionalities outside the reference taxonomy (assigned to a “new features” category).
    - Functionality Accuracy: two views—“Pass Rate” (execution success on adapted tests) and “Voting Rate” (majority-vote semantic validation that the intended algorithm exists).
    - Code-level statistics: files, normalized LOC, token count (tests/examples excluded).
  - Tasks: sampled and filtered from original test trees; each task provides a natural-language description, ground-truth test, and auxiliary materials (Fig. 3; Appendix D.2). Evaluation proceeds via localization → semantic validation (5 votes) → test adaptation and execution (§5.1.3).

- Baselines and setup (§5.2–§5.3)
  - Baselines span multi-agent (MetaGPT, ChatDev), workflow (Paper2Code), and terminal agents (Codex CLI, Claude Code CLI, Gemini CLI, OpenHands), some with two backbone models (`o3-mini` and `Qwen3-Coder`). All runs for 30 iterations; baselines allowed web search; per-iteration prompting to propose or implement features.
  - `ZeroRepo` config: 30 feature-selection iterations; up to 8 debugging iterations per function; 20 localization attempts per iteration; majority-vote diagnosis for test/environment errors with up to 20 remediation attempts (§5.3).

- Main results (Table 2; §6)
  - Overall performance:
    - Coverage: `ZeroRepo (o3-mini)` reaches 81.5% vs `Claude Code` at 54.2% (+27.3 points). `ZeroRepo (Qwen3-Coder)`: 75.1%.
    - Pass/Vote (accuracy): `ZeroRepo (o3-mini)` 69.7% / 75.0% vs `Claude Code` 33.9% / 52.5% (+35.8 points pass). `ZeroRepo (Qwen3-Coder)` 57.3% / 68.0%.
    - Scale: `ZeroRepo (Qwen3-Coder)` produces ~389 files, 36,941 LOC, 445,512 tokens—about 3.9× Claude Code in tokens and 68× other baselines (Table 2 “Tokens ↑”).
    - Novelty: `ZeroRepo (o3-mini)` 13.6% (151.5/1114.2 features), suggesting coherent extension beyond reference taxonomies.
  - Calibration with “Gold Projects”:
    - The automated pipeline reaches 81.0% pass / 92.0% voting on human-developed projects, indicating a high—but not perfect—ceiling (Table 2 “Gold Projects”).

- Scaling analyses (§7.1; Fig. 5–6; Table 3)
  - Feature growth: Fig. 5 shows near-linear increase to 1,100+ leaf features for `ZeroRepo (o3-mini)` across 30 iterations, while strong baselines plateau early (Codex stops after 4–5 iterations).
  - Code growth: Fig. 6 shows near-linear LOC growth surpassing 30K at 30 iterations; baselines plateau around 1–4K LOC.
  - Coverage/novelty over iterations: Table 3 (MLKit-Py) shows coverage rising from 70.2% (iter 5) to 95.7% (iter 30) while novelty remains steady (≈4–8%).

- Structural fidelity and dependency modeling
  - Fig. 4 visualizes a generated MLKit-Py repository: clear folder hierarchy, inter-module data flows (e.g., `data_lifecycle → clustering → models → evaluation`), and class-level inheritance/invocation links.

- Localization ablation (§7.3; Table 4)
  - Graph guidance reduces localization steps by 30–50% across integration testing, incremental development, and debugging.
  - Example: Integration Testing steps drop from 13.3±11.1 (no graph) to 6.2±2.1 (with graph).

- Additional observations (Appendix C.5; Table 9; Fig. 12)
  - Code success vs test coverage: Table 9 shows relatively high code success rates for `o3-mini` (often 75–90%), but moderate coverage of generated tests (≈60–70%). Fig. 12 indicates test coverage declines as code length grows, particularly at function-level.

- Do the experiments support the claims?
  - The combination of strong aggregate improvements (Table 2), near-linear scaling (Fig. 5–6), and localization efficiency (Table 4) directly supports that an explicit graph representation stabilizes long-horizon planning and scales better than NL-based plans.
  - Use of “Gold Projects” makes the automated evaluation credible by showing an attainable ceiling. Still, some judgments rely on LLM voting (validation), which is strong but not infallible (Appendix D.3).

## 6. Limitations and Trade-offs
- Dependence on a large external ontology
  - The Feature Tree (1.5M+ nodes; Table 5) strongly guides planning. This stabilizes coverage but may bias which functionalities are discoverable; domains poorly represented in the ontology could be underexplored (Appendix A.2).
  - LLM filtering and “missing-feature” proposals mitigate but do not eliminate ontology bias (Algorithm 2).

- Evaluation scope and modality
  - Benchmark breadth: six Python projects (Table 1) is substantial but limited—results may not generalize to other languages, build systems, or domains like frontend/UI, embedded, or distributed systems (§5.1.1).
  - Coverage metric counts representation, not correctness; novelty counts out-of-taxonomy features, which may not always be useful (Appendix D.3.1).

- Reliance on LLM judgments
  - Localization validation uses majority-vote LLM checks; while “Gold Projects” calibration is reassuring, LLM-based judgments can err, especially on nuanced semantic equivalence (Appendix D.3).

- Computational cost and complexity
  - The pipeline involves many iterations, searches, and tests: 30 planning iterations, up to 8 debugging loops per function, 20 localization attempts per iteration, and multi-round majority voting (§5.3). This increases compute/time budgets.

- Graph rigidity vs creativity
  - While `RPG` enforces coherence, overly strict flows or premature abstraction (e.g., shared base classes) could overconstrain implementations, limiting alternative designs or non-DAG workflows (implied by DAG assumption in §3.3.2 “Acyclic Structure” in prompts; Appendix B.1).

- Testing gaps
  - Fig. 12 and Table 9 suggest limited test coverage for longer code, indicating that test generation/adaptation remains a bottleneck. Integration tests exist, but thorough property- or fuzz-based testing is not reported (§4; Appendix C.4).

## 7. Implications and Future Directions
- How this changes the field
  - `RPG` reframes repository generation as graph construction plus graph execution, moving beyond ad hoc NL planning. The strong gains in coverage, correctness, and scale (Table 2) indicate that structured, persistent planning media may be essential for long-horizon software generation.

- Follow-up research enabled/suggested
  - Learning the graph: Train models to propose `RPG`s end-to-end from specs, perhaps with supervised signals from human repos or RL from pass/fail outcomes (building on Algorithms 1–2).
  - Richer analysis: Integrate static analysis, type inference, and formal contracts into `RPG` edges to catch interface mismatches before code generation (§3.3.2).
  - Stronger testing: Couple `RPG` with property-based, fuzz, and mutation testing to address the coverage drop with code length (Fig. 12).
  - Cross-language, multi-environment support: Extend to Java/C++/TypeScript and to complex build/dependency systems; incorporate container orchestration in data-flow nodes.
  - Continual development: Use `RPG` as a living architecture map for maintenance and refactoring—support PR-level edits, impact analysis via dependency traversal (Appendix C.1 tools show a starting point).

- Practical applications
  - Rapid bootstrapping of research libraries and internal platforms from high-level charters.
  - Enterprise codebase modernization: encode legacy architecture into an `RPG`, then plan systematic refactors with graph-guided localization.
  - Educational tooling: teach software architecture by visualizing how modules, files, and data flows fit together; auto-generate scaffolding and tests.

> Key quantitative takeaway: “On RepoCraft, ZeroRepo attains 81.5% functional coverage and a 69.7% pass rate… while producing repositories with 36K LOC and 445K tokens, about 3.9× larger than the strongest baseline” (Table 2, §6).  
> Key qualitative takeaway: `RPG`’s explicit nodes/edges align functional intent with file layout and typed data flows (Fig. 2), enabling coherent topological generation (Fig. 1C) and faster localization (Table 4).

Together, these results make a compelling case that a graph-first planning medium is a powerful foundation for scalable, repository-level code generation.
