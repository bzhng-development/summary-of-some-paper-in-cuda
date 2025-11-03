# RPG: A REPOSITORY PLANNING GRAPH FOR UNIFIED AND SCALABLE CODEBASE GENERATION

**ArXiv:** [2509.16198](https://arxiv.org/abs/2509.16198)

## 🎯 Pitch

This paper introduces the Repository Planning Graph (RPG), a unified, graph-based approach for planning and generating full-scale software repositories—from high-level specifications to file structure, data flows, and function interfaces. By using RPG as an explicit, persistent blueprint, the ZeroRepo framework produces Codebases that are vastly larger, more coherent, and more correct than prior natural language–driven or agentic baselines. This breakthrough enables consistent long-horizon software synthesis, demonstrating a leap toward true automated codebase generation and providing a scalable foundation for next-generation AI software agents.

---

## 1. Executive Summary
This paper introduces the Repository Planning Graph (`RPG`), a structured “blueprint” for planning and generating entire software repositories. Built on `RPG`, the `ZeroRepo` framework turns a short, high-level request (e.g., “build an ML library”) into a coherent multi-file codebase by planning features, file layout, data flows, and function interfaces before writing code. On the new `RepoCraft` benchmark of six real projects and 1,052 tasks, `ZeroRepo` substantially outperforms strong agentic coding baselines in both scale and correctness (Table 2).

## 2. Context and Motivation
- Problem addressed
  - Modern language models can write functions or single files, but they struggle to generate an entire repository from scratch that is internally consistent, modularized, and testable (Abstract; §1).
  - Existing systems plan in natural language (NL)—bulleted lists or markdown notes—which becomes ambiguous, drifts over long runs, and fails to encode dependencies precisely (§1; §2).

- Why it matters
  - Real-world software is multi-module and long-lived; generating a full repo from a brief specification would enable end-to-end software agents and accelerate prototyping and maintenance (Abstract; §1).

- Prior approaches and shortcomings (§2)
  - Multi-agent workflows (e.g., MetaGPT, ChatDev) role-play “architect/engineer,” yet still coordinate via NL text, leading to drift and inconsistency.
  - Workflow systems (e.g., Paper2Code, AutoP2C) stage architectures then fill details, but their intermediate plans are still NL artifacts.
  - Terminal agents (e.g., Claude Code, Gemini CLI, OpenHands) externalize plans as markdown/checklists and iterate, but plans remain unstructured.
  - Across paradigms, the missing piece is a persistent, structured representation that encodes hierarchy, data flow, and interfaces.

- Positioning of this work
  - The paper replaces NL planning with a single, evolving graph—the Repository Planning Graph (`RPG`)—that unifies “what to build” (features/modules) with “how to build it” (files, flows, classes, functions) (§3.1–§3.3). `ZeroRepo` is the agent that constructs and executes this graph to synthesize repositories (§4).

## 3. Technical Approach
At a high level, `ZeroRepo` turns a user request into code in three stages (Figure 1):
1) Proposal-level planning → 2) Implementation-level refinement → 3) Graph-guided code generation and testing.

Key terms (paper-specific):
- `Repository Planning Graph (RPG)`: a graph whose nodes represent capabilities mapped to code artifacts (folders, files, classes, functions) and whose edges encode data flows and ordering constraints (§3.1; Figure 2).
- `Feature Tree`: a large ontology (1.5M+ capability nodes) used as a knowledge base to anchor feature selection (§3.2; Appendix A.2, Table 5, Figure 8).
- `ZeroRepo`: the agent that builds and traverses `RPG` to generate the repository (§4).

Step-by-step:

A) Proposal-level construction (§3.2; Figure 1A)
- Goal: translate a vague user goal (e.g., “Make a machine learning repository”) into a coherent functionality graph.
- How it works:
  1) Retrieve candidates from a global `Feature Tree` (EpiCoder; 1.5M nodes, seven levels; Appendix A.2, Table 5). Each node is vector-embedded with path metadata for scalable retrieval (§3.2).
  2) Explore–Exploit Subtree Selection: 
     - Exploit: retrieve top-k feature paths semantically aligned with the goal.
     - Explore: sample unvisited regions to avoid tunnel vision (§3.2; Algorithm 2).
     - Diversity-aware Rejection Sampling encourages coverage of new areas by rejecting candidate subtrees that overlap too much with already selected ones (Algorithm 1).
  3) Refactor for goal alignment: reorganize the retrieved subtree into cohesive modules (e.g., move `silhouette_score` under an `evaluation` module rather than `clustering`) to create a clean functionality graph (§3.2).

Why this design:
- NL lists drift and don’t encode hierarchy/relations. Here, a large, structured prior (Feature Tree) plus diversity control produces broad, stable coverage, and subsequent refactoring enforces high cohesion/low coupling (§3.2; Appendix A.1–A.2).

B) Implementation-level construction (§3.3; Figure 1B)
- Goal: turn the abstract functionality graph into an implementation-ready plan (the full `RPG`).
- Two sub-steps:

  B1) File structure encoding (§3.3.1)
  - Map modules to folders and features to files, producing a file-augmented graph.
  - Example: `algorithms/`, `evaluation/` as root folders; `linear_models.py`, `preprocess.py` as files (Figure 2; Appendix B.2 “Repository Skeleton”).
  - Design motivation: preserve semantic cohesion and reduce cross-file coupling by aligning functionality and file boundaries.

  B2) Data flow and function/interface encoding (§3.3.2)
  - Add inter-module data-flow edges (e.g., outputs of `data_loading` feed `algorithms`, then `evaluation`) and intra-module orderings (e.g., `load_data.py` → `preprocess.py`) to impose a topological build/execution order (Figure 2).
  - Abstract shared interfaces as base classes or common types (e.g., `BaseEstimator` for consistent `fit/predict` contracts) to unify interactions across modules (§3.3.2 “Abstracting Global Interfaces”).
  - Adaptively map leaf features to functions or to classes with multiple methods when features are interdependent (e.g., `load_json` and `load_csv` become a `DataLoader` class; §3.3.2; Figure 2).
  - Result: the complete `RPG`, which is a dependency- and interface-aware plan spanning folders→files→classes→functions.

C) Graph-guided code generation (§4; Figure 1C)
- Traversal and TDD:
  - Traverse the `RPG` in topological order so that dependencies are implemented before dependents (§4).
  - For each leaf (function/class), use test-driven development: derive a unit test from the node’s docstring, implement, then run tests. Failures trigger iterative fixes; only passing code is committed (§4; Appendix C.4–C.5).
- Localization and editing:
  - When a task refers to “the SVR algorithm,” “RPG-Guided search” maps that intent to candidate nodes; “repository code view” and “dependency exploration” let the agent inspect and patch the relevant code (§4 “Graph-Guided Localization and Editing”; Appendix C.1–C.3).
- Graph-guided test validation:
  - Unit tests for nodes; regression tests when nodes are modified; integration tests for subgraphs to verify cross-module data contracts. A lightweight majority-vote judge separates true implementation bugs from test/environment errors (§4; Algorithms 3–4).

Analogy:
- Treat `RPG` like an architectural blueprint. Instead of a prose “plan,” the blueprint specifies rooms (folders), walls (files), plumbing/electrical lines (data-flow edges), and appliances (functions/classes) with ports (interfaces). The builder (`ZeroRepo`) follows the blueprint room-by-room in a valid order and tests each appliance before moving on.

## 4. Key Insights and Innovations
1) A single, persistent, and executable planning representation (`RPG`) (§3.1–§3.3)
   - What’s new: the graph simultaneously encodes functionality, file layout, data flows, and interfaces. Prior systems kept these in free-form NL, spreading details across prompts and notes (§1–§2).
   - Why it matters: the topological order binds planning to implementation, reducing drift across long horizons and enabling near-linear scaling of features and code (Figures 5–6; §7.1).

2) Feature-tree–grounded planning with diversity-aware selection (§3.2; Appendix A.1–A.2)
   - What’s new: `ZeroRepo` anchors planning in a 1.5M-node Feature Tree and balances exploit vs. explore via retrieval plus overlap-controlled sampling (Algorithms 1–2).
   - Why it matters: stabilizes coverage (less randomness/bias), while exploration avoids myopic plans. Empirically, the number of planned features grows almost linearly over 30 rounds (Figure 5), unlike baselines that quickly plateau.

3) Graph-guided generation and debugging with staged tests (§4; Algorithms 3–4)
   - What’s new: unit→regression→integration tests are aligned to the graph; failures trigger graph-aware localization/editing (Appendix C.1–C.3).
   - Why it matters: increases correctness and editing efficiency. Removing graph guidance roughly doubles the steps needed for localization across tasks (Table 4).

4) A benchmark for repository-from-scratch generation (`RepoCraft`) (§5.1; Figure 3)
   - What’s new: six real Python projects (pandas, scikit-learn, SymPy, statsmodels, requests, Django) with paraphrased names, and 1,052 evaluation tasks derived from official tests (§5.1.1–§5.1.3; Table 1).
   - Why it matters: evaluates both breadth (coverage, novelty) and correctness (pass rate, voting rate) at repository scale (Table 2), moving beyond single-file/problem benchmarks.

These are fundamental innovations (representation and evaluation), not just incremental tuning.

## 5. Experimental Analysis
- Evaluation setup
  - Benchmark: `RepoCraft` with six repositories and 1,052 tasks (§5.1; Table 1), built by parsing and sampling official tests (Figure 3; Appendix D.2).
  - Metrics (§5.1.2; Appendix D.3.1):
    - `Functionality Coverage`: fraction of reference categories covered.
    - `Functionality Novelty`: fraction of generated features outside the reference taxonomy.
    - `Voting Rate`: majority-vote semantic validation that the located implementation matches the target algorithm.
    - `Pass Rate`: fraction of tasks whose adapted tests pass.
    - Scale stats: file count, normalized LOC, token count (non-core code excluded).
  - Baselines: multi-agent (MetaGPT, ChatDev), workflow (Paper2Code), and terminal agents (Codex CLI, Claude Code CLI, Gemini CLI, OpenHands). Each runs up to 30 iterations; terminal agents allowed web search (Table 2; §5.2).
  - Implementation details: proposal-level feature selection runs for 30 iterations; code generation allows up to 8 debug iterations per function, with 20 localization attempts per iteration (§5.3).

- Main results (Table 2)
  - Coverage and correctness:
    > `ZeroRepo (o3-mini)` reaches 81.5% coverage and 69.7% pass rate, versus `Claude Code` at 54.2% and 33.9%.
    > `ZeroRepo (Qwen3-Coder)` reaches 75.1% coverage and 57.3% pass rate.
  - Scale:
    > `ZeroRepo (Qwen3-Coder)` generates 36,941 LOC and 445,512 code tokens on average, ~3.9× larger than `Claude Code` and ~68× larger than most others (Table 2).
  - Novelty:
    > Novelty rates of 9–14% with over 100 additional features on average, exceeding most baselines (Table 2, “Nov.” column).

- Per-repository examples
  - MLKit-Py (scikit-learn-like): `ZeroRepo (o3-mini)` hits 97.9% coverage and 73.5%/78.7% pass/vote; generates 31,596 LOC (Table 11).
  - HttpEasy (requests-like): `ZeroRepo (o3-mini)` achieves 100% coverage and 64%/72% pass/vote (Table 12).
  - PyWebEngine (Django-like): `ZeroRepo (o3-mini)` hits 79.2% coverage and 74.1%/84.4% pass/vote, producing 27,647 LOC (Table 13).

- Scalability studies (§7.1; Figures 5–6; Table 3)
  - Feature scaling: feature counts increase near-linearly over 30 iterations for `ZeroRepo`, while baselines plateau (Figure 5).
  - Code scaling: LOC grows near-linearly to >30K LOC within 30 iterations for `ZeroRepo`, versus 3–4K plateau for Claude/Gemini and <1K for Codex (Figure 6).
  - Coverage/novelty vs. iteration on MLKit-Py: coverage rises from 70.2% (round 5) to 95.7% (round 30); novelty climbs to 7.9% with ~100 new features (Table 3).

- Graph-guided localization ablation (§7.3; Table 4)
  - With graph guidance, steps to localize/resolve tasks drop by 30–50% across integration testing, incremental development, and debugging.
    > Example: Integration testing falls from 13.3±11.1 steps “w/o Graph” to 6.2±2.1 with graph guidance (Table 4).

- Additional diagnostics
  - Complex dependencies are realized in practice: Figure 4 visualizes the generated MLKit-Py repository’s folder skeleton, inter-module data flows, and function/class dependencies, illustrating layered planning realized as code (§6).
  - Test generation vs. code success: while code success rates for `o3-mini` often exceed 75–85%, the coverage achieved by generated tests themselves is moderate (typically 60–70%), suggesting test generation remains a bottleneck (Table 9; Figure 12).

- Do the experiments support the claims?
  - The systematic performance gap on coverage and pass rate across six realistic projects (Table 2) supports the central claim that a structured planning medium improves long-horizon repository synthesis.
  - Scaling plots (Figures 5–6) directly connect the representational choice (graph vs. NL) to sustained growth in features and LOC.
  - The localization ablation (Table 4) shows the graph is not just a reporting artifact; it concretely guides search and editing.

- Where results are mixed or conditional
  - Models differ: `o3-mini` yields higher pass rates, while `Qwen3-Coder` often builds larger repos (Table 2, §7 and Appendix B.3).
  - Test coverage declines with longer code units (Figure 12), meaning correctness verification weakens as implementations get more complex.

## 6. Limitations and Trade-offs
- Reliance on a large external ontology
  - The approach assumes access to a comprehensive `Feature Tree` (EpiCoder; 1.5M nodes) to ground proposal-level planning (§3.2; Appendix A.2). Biases in this ontology (e.g., over-representation of data processing; Figure 8) could skew planned functionality unless carefully refactored.

- Automatic evaluation and potential biases
  - Task presence and correctness use LLM-based localization and majority-vote semantic validation before executing adapted tests (§5.1.3; Appendix D.3). Although validated on human “Gold Projects” (81.0% pass, 92.0% vote; Table 2 “Gold Projects”), LLM judgments can still introduce noise or bias.

- Test generation limitations
  - Unit tests are derived from docstrings at generation time (§4 “Graph-Guided Test Validation”), and overall test coverage is moderate (Table 9, Figure 12). Incorrect or shallow docstrings can yield weak tests.

- Compute and iteration budget
  - Building large repos requires multiple planning and debugging rounds: 30 planning iterations, up to 8 debugging iterations per function, 20 localization attempts per iteration, and 5-round voting (§5.3). This is compute-intensive compared to simpler baselines.

- Scope of codebases
  - The benchmark and examples are Python-centric (§5.1; Table 1), and the approach is demonstrated on library-style repositories. Non-Python ecosystems, multi-language repos, or systems with heavy build tooling (e.g., C++/Rust) may require additional static analysis or build-system modeling.

- Dynamic evolution and maintenance
  - While `RPG` evolves during construction, the paper does not deeply analyze long-term maintenance under changing requirements (e.g., large refactors, cross-cutting concerns) beyond the presented edit tools (Appendix C.2–C.3).

## 7. Implications and Future Directions
- How this changes the landscape
  - Moving from NL plans to a persistent, executable graph (`RPG`) shifts repository generation from ad hoc iteration to blueprint-guided construction, enabling near-linear scaling in both features and code size (Figures 5–6). This provides a foundation for more reliable, long-horizon software agents.

- Practical applications
  - Rapid bootstrapping of domain libraries (ML, stats, data tools).
  - Internal tooling scaffolds with consistent interfaces and tests.
  - Educational codebases where functionality maps cleanly to curriculum modules.
  - Automated maintenance: graph-guided localization/editing could aid large-scale refactors and bug-fixing.

- Research directions
  - Generalize beyond Python: enrich `RPG` with language- and build-system–aware nodes/edges (e.g., Bazel/CMake rules, cross-language bindings).
  - Learn the ontology: replace or complement the handcrafted `Feature Tree` with embeddings learned from large corpora, while keeping the graph structure to avoid NL drift.
  - Stronger verification: integrate property-based testing, fuzzing, and static/dynamic analysis to raise test coverage beyond docstring-derived unit tests (Figure 12; Table 9).
  - Human-in-the-loop planning: expose `RPG` for interactive editing so developers can steer modules, interfaces, and flows directly.
  - Continuous maintenance: study how `RPG` supports incremental evolution (API changes, deprecations, performance optimizations) and how graph diffs correlate with regression risk.

Overall, the paper’s central move—replacing natural-language plans with an explicit, evolving graph that encodes hierarchy and data flow—proves crucial for building larger, more accurate repositories from scratch. The `RepoCraft` benchmark and the empirical analyses (Tables 2–4; Figures 4–6) indicate that this representational shift is a durable path forward for agentic code generation.
