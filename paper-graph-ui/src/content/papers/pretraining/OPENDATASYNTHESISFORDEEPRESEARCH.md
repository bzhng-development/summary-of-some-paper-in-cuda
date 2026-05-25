# OPEN DATA SYNTHESIS FOR DEEP RESEARCH

**ArXiv:** [2509.00375](https://arxiv.org/abs/2509.00375)

## 🎯 Pitch

This paper introduces InfoSeek, the first fully open-source framework for synthesizing large-scale 'Deep Research' tasks by transforming web and Wikipedia content into hierarchical, multi-step reasoning problems with verifiable and unique answers. By formalizing Deep Research as Hierarchical Constraint Satisfaction Problems and supporting scalable dataset generation with rich intermediate evidence, InfoSeek enables the training of compact language models (like the 3B-parameter InfoSeeker) that significantly outperform larger models on complex, real-world research tasks. This innovation paves the way for more robust, transparent, and accessible AI agents capable of tackling challenging, evidence-synthesizing queries crucial for domains such as science and policy.

---

## 1. Executive Summary (2-3 sentences)
This paper introduces InfoSeek, a fully open-source framework that synthesizes large-scale, verifiable “Deep Research” questions by turning web/Wikipedia content into hierarchical reasoning problems, and InfoSeeker, a 3B-parameter agent trained on this data with a search-and-refine workflow. The approach improves small-model deep research performance: on BrowseComp-Plus, the 3B InfoSeeker model reaches 16.5% accuracy using BM25 retrieval, outperforming several larger open baselines and approaching some commercial APIs (Table 4).

## 2. Context and Motivation
- Problem addressed:
  - Deep Research tasks require decomposing complex questions into sub-problems, running multi-step searches, and synthesizing evidence across sources. Existing datasets (e.g., Natural Questions, HotpotQA) lack this hierarchical, interdependent structure and often admit shortcut reasoning or ambiguous answers (Abstract; Figure 2; Section 1).
  - There is no large, open dataset that cleanly captures hierarchical dependency structures with verifiable, unique answers (Table 1; Section 1).

- Why it matters:
  - Real-world tasks like scientific discovery and policy analysis demand layered evidence synthesis and tool use, not just factual lookup (Section 1).
  - Training and evaluating such capabilities requires data with controllable structural complexity and traceable intermediate steps (Sections 1–3).

- Prior approaches and gaps:
  - Traditional QA datasets: largely single-hop or flat multi-hop constraints (Table 1; Section 2.1).
  - Recent synthetic or agentic-benchmark efforts: often lack public datasets/workflows, introduce shortcuts, or don’t model hierarchical dependencies explicitly (Table 1; Sections 1, Related Works §6).
  - Some agentic search frameworks optimize policies but rely on simpler data or closed workflows (Sections 1, 5.1–5.2, Related Works §6).

- Positioning:
  - Formalizes Deep Research as a Hierarchical Constraint Satisfaction Problem (HCSP) with unique, verifiable answers (Sections 2.2–2.4; Figure 2).
  - Introduces InfoSeek to synthesize HCSPs from the web/Wikipedia via a dual-agent tree-building process with explicit evidence traces (Section 3; Figure 2 right).
  - Trains an agent (InfoSeeker) using a multi-query search-and-refine workflow with rejection-sampled supervised trajectories and lightweight reinforcement learning (Section 4; Figure 3).

## 3. Technical Approach
This section explains, step by step, the paper’s formalization, data synthesis pipeline, and training workflow.

- Formalizing Deep Research as HCSPs (Sections 2.1–2.3; Figure 2; Eqs. 1–7)
  - Concepts:
    - `Constraint Satisfaction Problem (CSP)`: Find the unique answer A that satisfies all independent constraints. Plainly: list conditions, collect candidates matching each, intersect them. Equation (1) defines A as the intersection of sets S(ci) for each constraint ci, with the uniqueness condition |A| = 1.
    - `Multi-hop Problem (MHP)`: A sequential chain where later steps depend on earlier outputs. Equation (2) models the answer as k chained reasoning steps S^(k)(c).
    - `Hierarchical CSP (HCSP)`: A nested combination where constraints are organized in a hierarchy that mixes parallel constraints and embedded sub-questions; higher-level validity depends on satisfying all lower-level constraints. Equation (3) defines a decomposition H(x) that intersects direct constraints with recursively solved sub-questions H(yj).
  - ‘Research tree’ representation:
    - A tree `T = (V, E)` whose nodes are entities/facts and edges are relations from source pages (Section 2.3.1). The base case is a root node (Eq. 4); expansion adds a node/edge (Eq. 5).
    - From tree to HCSP:
      - At a node v: convert leaf edges to constraints and internal children to sub-questions (Eqs. 6–7). The overall question is obtained at the root (Section 2.3.2).

- InfoSeek data synthesis (Section 3; Figure 2 right)
  - Dual-agent roles:
    - `Planner`: Maintains the global view of the partially built tree; chooses where to expand and when to stop to meet target complexity (Section 3).
    - `Browser`: Executes on pages by extracting hyperlinks (to grow depth) and atomic claims (to add constraints) and validates relevance with evidence traces (Section 3).
  - Four actions (Sections 3.1–3.4):
    1) Initialization: Sample a valid entity from Wikipedia/web as the final answer (root r); add one related child via a validated hyperlink (Section 3.1).
    2) Blur parent with constraints: Add multiple factual claims from the current entity’s page as children so that, taken together, they uniquely determine the parent. To prevent “overdetermined” shortcuts, they ensure mutual exclusivity among candidate sets (Section 3.2; Figure 2 right, “Blur Parent Node”).
       - Definition: “Blurring” means replacing a named parent entity with a set of constraints that describe it indirectly, forcing solvers to combine multiple pieces of evidence rather than guessing from a single telltale feature.
    3) Extend the tree: Add a child via a dependency link (e.g., “X was discovered by Y”) to increase depth and require additional reasoning hops (Section 3.3).
    4) Terminate and generate question: Stop when the tree meets targeted depth/branching and all nodes have sufficient constraints; convert the tree into a natural language question that requires traversing the hierarchy (Section 3.4).
  - Quality assurance (Section 3.5):
    - Difficulty filter: Remove samples answerable from parametric memory alone by testing `Qwen2.5-32B-Inst`; only 2% were correctly answered; those are removed.
      > “the model was able to correctly answer only 2% of the questions. We remove these samples” (Section 3.5).
    - Verifiability filter: Provide the ground-truth web pages (plus distractors) to `Gemini 2.5 Flash` and require solving from those pages; discard ambiguous/unsolved items (Section 3.5).
  - Dataset statistics (Section 3.6; Table 2):
    - 52,138 samples; most require 4–6 reasoning vertices; cost $571.8; mean question length 53.4 tokens.
    - Difficulty proxy: `Qwen2.5-72B` with CoT fails on 92.7% overall; failure rate grows with vertices (e.g., 88.1% at 3 vertices to 94.1% at ≥7).
      > “high overall failure rate of 92.7%… increasing from 88.1% for 3-vertex problems to 94.1% for [≥7]” (Section 3.6; Table 2).

- InfoSeeker agent workflow and training (Section 4; Figure 3; Appendix A.1)
  - Inference-time workflow (Section 4.1; Figure 3):
    - Think step: The model explicitly plans what to search next using `<think>…</think>` to reduce aimless retrieval.
    - Parallel multi-query search: Generates multiple queries at once (`<search>` tags) to cover different angles, improving recall without many turns.
    - Refiner Agent: A separate lightweight model (Qwen2.5-7B-Inst) summarizes top-k retrieved results per query into concise `<information>` snippets and suggests next steps, preventing context bloat.
    - Finalization: Output the final answer under `<answer>`.
  - Supervised fine-tuning via rejection sampling (Section 4.2; Appendix A.1):
    - Build a trajectory dataset by executing tasks with a strong teacher (Qwen2.5-72B) and a preview InfoSeeker; keep only trajectories that reach correct answers; also screen out search/reasoning shortcuts with Gemini 2.5 Flash (Section 4.2).
    - Round 1: Distill 24k valid trajectories (from 50k InfoSeek + 5k NQ/HQA; teacher accuracy ~21.8%), SFT Qwen2.5-3B-Inst, then RL (Appendix A.1).
    - Round 2: Use rejection sampling on the Round 1 model to build 3,450 high-quality trajectories (from 16,494 generated), SFT again, then a short RL phase on harder examples the model still fails (Appendix A.1).
  - Reinforcement learning recipe (Sections 4.3–4.3.2; Appendix A.1):
    - Algorithm: Group Relative Policy Optimization (GRPO), a PPO-style method that avoids a value model by normalizing group rewards (Section 4.3.1). The objective mixes a clipped advantage term with a KL penalty to a reference policy (equation in Section 4.3.1).
    - Reward: Binary—1 only if the output has correct format and the extracted answer matches; 0 otherwise (Section 4.3.2).

## 4. Key Insights and Innovations
- Formalizing Deep Research as HCSP with a research-tree construction (Sections 2.2–2.3; Eqs. 3, 6, 7; Figure 2).
  - Novelty: Moves beyond flat constraints (CSP) and simple sequential chains (MHP) by encoding nested constraints and sub-questions in a tree. This formalization clarifies what structure “Deep Research” should have and how to generate it.
  - Significance: Enables complexity control (depth/branching) and principled conversion to natural language questions, supporting verifiable, unique answers.

- “Blurred parent node” technique to prevent shortcuts and enforce uniqueness (Section 3.2; Figure 2 right).
  - Novelty: Rather than name a target entity directly, replace it with multiple jointly sufficient constraints and enforce mutual exclusivity to avoid single-clue solutions.
  - Significance: Raises difficulty while avoiding underdetermined or overdetermined cases (Section 2.4), strengthening training signal for hierarchical reasoning.

- Dual-agent data synthesis with explicit evidence traces and two-pronged QA (Sections 3.1–3.5).
  - Novelty: A Planner–Browser loop grows both depth and breadth via hyperlink and claim extraction, logs evidence, and uses automated difficulty and verifiability filters (Qwen2.5-32B screening; Gemini 2.5 Flash with distractors).
  - Significance: Scales to 52k examples at low cost ($571.8; Table 2) with high failure rates for strong LLMs (92.7%), indicating genuine difficulty (Section 3.6).

- Efficient agent workflow: parallel multi-query search plus Refiner Agent (Section 4.1; Figure 3).
  - Novelty: Summarizes per-query retrievals before feeding them back, keeping the context lean while maintaining high recall.
  - Significance: Avoids the typical context-bloat failure mode of ReAct-like agents in multi-turn search (Section 4.1).

- Compact-model training pipeline (rejection-sampled SFT + light RL) that transfers deep research abilities to 3B models (Section 4; Appendix A.1; Tables 3–5).
  - Novelty: Careful trajectory curation and two-round training yield a competitive 3B agent that can match or surpass larger open baselines on difficult browsing tasks.
  - Significance: Demonstrates that capable deep-research agents do not require massive proprietary models if trained on the right data and workflow.

## 5. Experimental Analysis
- Evaluation setup (Section 5.1; Appendix A.2):
  - Benchmarks:
    - Single-hop QA: NQ, TriviaQA, PopQA.
    - Multi-hop QA: HotpotQA, 2Wiki, Musique, Bamboogle.
    - Deep research: BrowseComp-Plus (830 problems; fixed 100K web page corpus).
  - Metrics: Exact Match (EM) for QA; BrowseComp-Plus uses LLM-judged accuracy as in the official setting.
  - Retrieval: Wikipedia-25 with BGE-M3 embedding retriever (top-5) for QA; BM25 for BrowseComp-Plus (Appendix A.2).
  - Baselines: RAG variants (RAG, IRCoT, RQRAG, Self-RAG) and agentic search models (Search-o1, Search-R1, ZeroSearch, AutoRefine, InForage) plus commercial APIs and open models on BrowseComp-Plus (Section 5.1).

- Main quantitative results:
  - Overall QA (Table 3):
    - InfoSeeker-3B achieves the best average across seven QA sets (43.5), outperforming all listed RAG and agentic-search baselines.
    - Notable per-dataset numbers: NQ 42.7, TQA 57.1, PopQA 48.0, HQA 44.6, 2Wiki 52.0, MSQ 20.5, Bamboogle 39.8.
    - Quote:
      > “InfoSeeker-3B … Avg 43.5,” versus best baseline averages: AutoRefine-3B 39.6 and InForage-3B 40.6 (Table 3).
  - Deep research (BrowseComp-Plus; Table 4):
    - InfoSeeker-3B reaches 16.5% accuracy with BM25 retriever, using 8.24 search calls per problem on average.
    - Comparison:
      - Beats Qwen3-32B (3.5%), SearchR1-32B (3.9%), and some commercial APIs: Gemini 2.5 Flash (15.5%) and Sonnet 4 (14.3%).
      - Trails Gemini 2.5 Pro (19.0%) and GPT-5 (55.9%).
    - Quote:
      > “InfoSeeker-3B … 16.5%,” vs “Gemini 2.5 Flash 15.5%,” “Gemini 2.5 Pro 19.0%,” “GPT-5 55.9%” (Table 4).
  - Data effectiveness (Table 5):
    - Training on InfoSeek vs NQ+HQA for RL (same 3B backbone): BrowseComp-Plus accuracy 16.5% vs 3.0%; InfoSeek-trained agents also make more search calls (8.24 vs 1.39), suggesting better tool use.
    - Quote:
      > “NQ+HQA: 3.0% … InfoSeeker: 16.5%” (Table 5).

- Dataset validation results:
  - Difficulty: Only 2% of InfoSeek problems are solved by Qwen2.5-32B without context; these are removed (Section 3.5).
  - Hardness scaling: Failure rate of Qwen2.5-72B (CoT) increases with more vertices (Section 3.6; Table 2).
    > “overall failure rate of 92.7%,” “88.1% (3 vertices) → 94.1% (≥7)” (Table 2).

- Assessment:
  - The breadth of baselines and benchmarks is strong. Results consistently show that the InfoSeek dataset plus the proposed workflow and training pipeline yield gains in both standard QA and complex browsing tasks (Tables 3–5).
  - The BrowseComp-Plus result for a 3B model is notable; while far from GPT-5, it surpasses several larger or proprietary baselines using a transparent setup (Table 4).
  - Ablations and robustness:
    - The paper does not provide a component ablation for the workflow (e.g., without Refiner Agent, without parallel queries) or for the blurring/verification procedures in data synthesis.
    - The data effectiveness ablation (Table 5) is clear: training on InfoSeek is far better than NQ+HQA for deep research.

## 6. Limitations and Trade-offs
- Assumptions about knowledge sources and structure:
  - The synthesis relies on Wikipedia and selected webpages with hyperlink structures and extractable claims (Sections 3.1–3.3). Domains with sparse hyperlinks or poor factual structure may be harder to encode as research trees.
  - Unique-answer requirement (Eqs. 1, 3, 6–7) is enforced; genuinely open-ended research questions or those with multiple valid answers are excluded by design.

- Data-generation choices:
  - Question naturalness depends on prompting a powerful LLM to verbalize blurred nodes into questions (Section 3, end of §3). Style and distribution may reflect LLM biases, though verifiability filters mitigate factual issues (Section 3.5).
  - Automated filtering uses specific models (Qwen2.5-32B-Inst for difficulty, Gemini 2.5 Flash for verifiability). Different validators might accept/reject different items.

- Evaluation breadth:
  - The strongest deep-research competitor (GPT-5) remains far ahead on BrowseComp-Plus (55.9% vs 16.5%; Table 4), leaving a sizable gap.
  - Limited ablation studies: no quantitative breakdown of how much each component (blurring, parallel search, Refiner Agent, RL) contributes.

- Compute and infrastructure:
  - While total dataset curation cost is low ($571.8; Table 2), training still uses multi-round SFT and RL with teacher rollouts (Appendix A.1). Replicators need search infrastructure, a Refiner Agent, and validators.

- Generalization scope:
  - Experiments focus on English Wikipedia/web data and the BrowseComp-Plus corpus (Appendix A.2). Multi-lingual or domain-specific generalization is not evaluated.

## 7. Implications and Future Directions
- Field impact:
  - Establishes a precise, reproducible definition of Deep Research as HCSP, with executable synthesis rules and verifiable outputs (Sections 2–3). This can standardize how the community builds and assesses complex, multi-layer reasoning tasks.
  - Demonstrates that a carefully designed dataset and workflow can lift small models to competitive deep-research performance, potentially democratizing access to capable research agents (Tables 3–5; Figure 1).

- Practical applications:
  - Training compact assistants that plan multi-step searches, verify evidence, and synthesize answers for investigative tasks in journalism, competitive intelligence, scientific literature review, and due diligence.
  - Incorporating the Refiner Agent pattern into enterprise search pipelines to control context size while maximizing recall (Section 4.1).

- Research directions enabled:
  - Reward design using preserved meta-information: InfoSeek records intermediate steps and retrieval labels, enabling compound rewards (e.g., per-step correctness, retrieval precision/recall) and trajectory-level exploration (Abstract; Section 1).
  - Component ablations and improvements:
    - Test single- vs multi-query search; compare different Refiner architectures; analyze the impact of blurring constraints and exclusivity checks.
    - Explore stronger verifiability signals (e.g., citation grounding at each step) and robustness to distractors.
  - Broader domains and modalities:
    - Extend synthesis to domains with fewer hyperlinks (legal documents, clinical guidelines) or multi-modal evidence (figures, tables), while keeping uniqueness and verifiability.
  - From unique-answers to synthesis tasks:
    - Expand beyond unique-answer HCSPs toward structured reports where intermediate claims are evaluated for consistency—leveraging the same tree scaffolding.

> Code and data are released under Apache 2.0 (Appendix A.4), including 50k+ QA pairs and 16.5k reasoning trajectories, making the framework reproducible and extensible.
