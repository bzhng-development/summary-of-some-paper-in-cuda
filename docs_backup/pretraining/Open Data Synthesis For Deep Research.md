# Open Data Synthesis For Deep Research

**ArXiv:** [2509.00375](https://arxiv.org/abs/2509.00375)
**Authors:** Ziyi Xia, Kun Luo, Hongjin Qian, Zheng Liu
**Institutions:** Beijing Academy of Artificial Intelligence (BAAI)

## 🎯 Pitch

InfoSeek and InfoSeeker represent a groundbreaking advance in AI-assisted research by formalizing complex questions as Hierarchical Constraint Satisfaction Problems (HCSPs) and providing an open dataset with 50k+ problems. By enabling small models to perform multi-step reasoning with verifiable outputs, this work drives improvements in tasks vital for scientific discovery and policy analysis, offering a scalable path to practical, efficient research assistants.

---

## 1. Executive Summary (2-3 sentences)
This paper introduces InfoSeek, an open, scalable framework that synthesizes complex, verifiable “Deep Research” questions by turning real web content into hierarchical reasoning problems, and InfoSeeker, a 3B-parameter search agent trained on this data. By formalizing Deep Research as Hierarchical Constraint Satisfaction Problems (HCSPs) and releasing 50k+ problems with 16.5k trajectories, the work enables small models to coordinate multi-step search and reasoning, achieving competitive performance on hard web-browsing benchmarks (e.g., BrowseComp-Plus).

## 2. Context and Motivation
- Problem addressed:
  - Modern LLMs increasingly need to perform “Deep Research”: decompose a complex question into sub-questions, search across heterogeneous sources, and synthesize evidence. Existing tasks (e.g., Natural Questions, HotpotQA) underrepresent the depth, hierarchy, and interdependence of constraints in real investigations (Introduction; Table 1).
  - Recent synthetic datasets often contain shortcut reasoning or knowledge leakage, and lack principled control over structure and difficulty (Abstract; Introduction).
- Importance:
  - Real-world tasks such as scientific discovery, policy analysis, and due diligence require layered reasoning with verifiable, unique answers (Sec. 2.2).
  - Training agents for this setting depends on high-quality data with hierarchical structure and explicit intermediate steps to support supervision and reinforcement learning (Abstract; Conclusion).
- Prior approaches and their limitations:
  - Traditional QA datasets: single-hop or flat multi-hop (Table 1; Sec. 2.1).
  - Recent multi-hop datasets or workflows: often closed-source, overfit to narrow domains, or do not expose the full generation pipeline for reproducibility (Introduction; Table 1).
  - Agentic search frameworks improve inference-time behavior but still rely on easier datasets for training (Sec. 6 Related Works).
- Positioning:
  - This work formalizes Deep Research with verifiable answers as HCSPs (Sec. 2.2–2.3; Fig. 2), proposes a dual-agent data synthesis pipeline that constructs HCSPs from web/Wikipedia content (Sec. 3), and shows that training on the resulting data improves agentic search performance, especially on BrowseComp-Plus (Sec. 5; Table 4–5).

## 3. Technical Approach
The paper contains two main technical parts: (A) a formal/algorithmic framework and pipeline to generate hierarchical, verifiable research questions (InfoSeek), and (B) a search-and-reason agent and training scheme (InfoSeeker).

A) Formalization and data synthesis (Sec. 2–3)
- Core definitions (Sec. 2; Fig. 2):
  - `Constraint Satisfaction Problem (CSP)`: Find the unique entity set `A` that satisfies all conditions of a question. Plainly: collect entities matching each condition, then intersect them. Equation (1) formalizes this as `A = ⋂ S(ci)` with `|A| = 1`.
  - `Multi-hop Problem (MHP)`: Solve through a chain of dependent steps. Equation (2) models k-step reasoning as repeated application of a mapping `S`.
  - `Hierarchical Constraint Satisfaction Problem (HCSP)`: Extends CSPs by nesting sub-questions within a hierarchy; the final answer emerges only when all constraints across levels are satisfied. Equation (3) recursively combines direct constraints `S(ci)` with sub-questions `H(yj)` into `H(x)`.
- Research Tree representation (Sec. 2.3; Fig. 2):
  - A `Research Tree` is an acyclic graph whose nodes are entities or facts; edges encode relations. Leaves correspond to atomic constraints; internal nodes are sub-problems; the root represents the final answer.
  - Mapping to HCSP:
    - Base case (height 1): turn edges from a node `v` to its leaf children into constraints and form a CSP for `v` (Eq. 6).
    - Recursive case: for node `v` with internal children, combine its leaf constraints with sub-questions generated from those internal children (Eq. 7).
- Potential pitfalls and how they are handled (Sec. 2.4; Sec. 3.2–3.5):
  - Underdetermined problems (multiple valid answers) and overdetermined ones (a single constraint already identifies the answer).
  - Mitigation via “blurring parent” and verification:
    - `Blurring parent with constraints` (Action 2 in Sec. 3.2; Fig. 2 right): the Browser agent adds multiple, mutually exclusive constraints to an internal node so that only one entity remains valid. This raises difficulty and enforces uniqueness, reducing both under- and overdetermination.
    - Verifiability checks (Sec. 3.5): a model (Gemini 2.5 Flash) is given the ground-truth documents plus distractors and must derive the correct answer. Samples that are ambiguous, incorrect, or unsolvable are filtered out.

- Dual-agent synthesis pipeline (Sec. 3; Fig. 2 right panels):
  - `Planner`: keeps global view, targets nodes to expand, and decides when to add constraints vs. extend depth.
  - `Browser`: executes actions by visiting web/Wikipedia pages, extracting candidate entities, links, and atomic claims.
  - Four actions (Sec. 3.1–3.4):
    1) Initialization: pick a root entity (final answer) and add a related child node (Sec. 3.1).
    2) Blur parent with constraints: add claims that together uniquely identify a parent (Sec. 3.2).
    3) Extend the tree vertically: follow a hyperlink relation to create deeper dependencies (Sec. 3.3).
    4) Terminate and generate question: once structural goals (breadth/depth) are met and all vertices have sufficient constraints, produce the natural-language question that forces traversal of the entire tree (Sec. 3.4).
- Quality assurance and difficulty control (Sec. 3.5–3.6; Table 2):
  - Difficulty: remove “easy” samples solvable by a strong model without browsing—Qwen2.5-32B answered only ~2% directly, and those were removed (Sec. 3.5).
  - Verifiability: solve with provided ground-truth pages plus distractors using Gemini 2.5 Flash; discard questions with wrong answers or multiple answers (Sec. 3.5).
  - Dataset statistics (Table 2): 52,138 problems; majority have 4–6 vertices. Failure rates of a strong baseline (Qwen2.5-72B with CoT) increase with more vertices, from 88.1% (3 vertices) to 94.1% (≥7), confirming controllable complexity.

B) Agent workflow and training (Sec. 4; Fig. 3; Appendix A.1–A.2)
- Inference workflow (Sec. 4.1; Fig. 3):
  - `Think Before Action`: the agent writes out a plan in `<think>...</think>` to decide what to search next.
  - `Parallelized multi-query search`: generate several queries at once in `<search>...</search>` to broaden recall without many slow, sequential attempts.
  - `Refiner Agent`: a lightweight model (Qwen2.5-7B) receives the top-k web results for each query, extracts salient evidence, and produces a concise summary in `<information>...</information>`. This keeps the agent’s working context focused while preserving recall.
  - `Answer` phase: the agent outputs the final result between `<answer>...</answer>`.
  - Design rationale: simply increasing top-k snippets bloats the context and dilutes focus; the multi-query + refiner setup maintains high recall with compact context (Sec. 4.1).
- Training pipeline (Sec. 4.2–4.3; Appendix A.1):
  - Stage 1—SFT via rejection sampling (Sec. 4.2; A.1):
    - Generate reasoning/browsing trajectories with a teacher model (Qwen2.5-72B) and a preview InfoSeeker; keep only successful trajectories that yield the correct answer and pass a shortcut check by Gemini 2.5 Flash.
    - Data scale and quality (A.1): from 50k InfoSeek + 5k NQ/HQA tasks, two rollouts each, 24k verified trajectories remain after filtering. Round-1 SFT on Qwen2.5-3B-Inst for 2 epochs (context length 16,384; 8×H100; ~2 hours).
  - Stage 2—Reinforcement Learning (Sec. 4.3; 4.3.1–4.3.2; A.1):
    - Algorithm: `Group Relative Policy Optimization (GRPO)`—a PPO-style method that avoids a value model by normalizing rewards within a group of rollouts (Sec. 4.3.1).
    - Reward: binary—1 only if both format and extracted answer are correct; 0 otherwise (Sec. 4.3.2).
    - Two-round scheme (A.1): RL after Round-1 SFT (200 steps; group size 5; temperature 0.8; top-5 retrieval). Then perform rejection sampling again, re-SFT with 3,450 high-quality multi-turn trajectories, and a second RL phase (100 steps without KL loss) on 14k harder samples the model previously failed on.

## 4. Key Insights and Innovations
- Formalization of Deep Research as `HCSP` (Sec. 2.2–2.3; Fig. 2; Eq. 3)
  - What’s new: A principled, mathematical framework that subsumes both CSP and multi-hop problems, capturing hierarchical dependencies of constraints and sub-questions.
  - Why it matters: It clarifies target capabilities and directly motivates a tree-based synthesis method that controls depth and breadth.
- `Research Tree` to question generation with “blurred parent” constraints (Sec. 2.3–2.4; 3.2; Fig. 2 right)
  - What’s new: Explicitly enrich internal nodes with multiple constraints that together uniquely identify the parent while making each constraint individually insufficient. This avoids shortcuts and enforces traversal of the hierarchy.
  - Why it matters: It significantly reduces under/overdetermination and encourages honest multi-step reasoning grounded in evidence.
- Dual-agent, verifiable, scalable data synthesis (Sec. 3.1–3.5; Table 2)
  - What’s new: A Planner/Browser loop with four actions, explicit evidence traces, difficulty screening with a strong LLM, and verifiability checks with ground-truth pages plus distractors.
  - Why it matters: Produces 50k+ high-quality problems with controllable structure at very low cost ($571.8 total; Table 2), and includes meta-information (intermediate steps, retrieval labels) for advanced training signals.
- Multi-query search + `Refiner Agent` workflow (Sec. 4.1; Fig. 3)
  - What’s new: Parallel query generation within a single step plus per-query summarization to keep contexts compact without sacrificing recall.
  - Why it matters: Improves search efficiency and relevance during agentic rollouts, avoiding context bloat from naive high top-k retrieval.
- Two-stage training with rejection-sampled SFT and GRPO RL (Sec. 4.2–4.3; A.1)
  - What’s new: A practical, reproducible pipeline that starts from distilled trajectories, then uses a simple, outcome-based reward to refine reasoning and querying behavior.
  - Why it matters: Enables a compact `3B` model to perform competitively on difficult web-browsing benchmarks (Table 4–5) and to generalize across single-/multi-hop tasks (Table 3).

## 5. Experimental Analysis
- Evaluation methodology (Sec. 5.1; A.2):
  - Datasets:
    - Single-hop: `NQ`, `TriviaQA (TQA)`, `PopQA`.
    - Multi-hop: `HotpotQA (HQA)`, `2WikiMultihopQA (2Wiki)`, `Musique (MSQ)`, `Bamboogle (Bamb)`.
    - Deep research browsing: `BrowseComp-Plus` (830 problems; fixed 100k web corpus).
  - Metrics: Exact Match (EM) for QA datasets; LLM judge (per official setting) for BrowseComp-Plus.
  - Retrieval/corpora:
    - Wikipedia-25 chunked to 512 tokens with `BGE-M3` for QA tasks (A.2).
    - `BM25` over BrowseComp-Plus’s provided corpus (A.2; Table 4).
  - Baselines (Sec. 5.1): Traditional RAG (RAG, IRCoT, RQRAG, Self-RAG) and agentic search methods (Search-o1, Search-R1, ZeroSearch, AutoRefine, InForage). For BrowseComp-Plus: commercial APIs (Gemini 2.5 Flash/Pro, Sonnet 4, GPT-4.1, GPT-5), and open-source Qwen3-32B, SearchR1-32B.
- Main quantitative results:
  - Overall QA performance (Table 3):
    - InfoSeeker-3B achieves the highest average across seven datasets: 
      - Quote: “InfoSeeker-3B … Avg. 43.5” versus InForage-3B (40.6), AutoRefine-3B (39.6), ZeroSearch-3B (34.0).
    - Notable per-dataset numbers:
      - `PopQA`: 48.0 (best among listed).
      - `2Wiki`: 52.0 (best among listed).
      - `HQA`: 44.6, competitive with AutoRefine-3B (40.4) and InForage-3B (40.9).
      - `NQ`: 42.7 (slightly below AutoRefine-3B’s 43.6).
      - `TQA`: 57.1 (below ZeroSearch-3B’s 61.5 and AutoRefine-3B’s 59.7).
      - `MSQ`: 20.5 (improves over others).
      - `Bamboogle`: 39.8 (best among listed).
    - Takeaway: While not the top on every dataset, the average gain suggests broad generalization across different QA types.
  - BrowseComp-Plus (Table 4):
    - Quote: “InfoSeeker-3B … Accuracy 16.5% with 8.24 search calls”
    - Comparison:
      - Beats open-source baselines of much larger size: Qwen3-32B (3.5%) and SearchR1-32B (3.9%).
      - Close to Gemini 2.5 Flash (15.5%) and within range of Gemini 2.5 Pro (19.0%); well below GPT-5 (55.9%).
    - Interpretation: For a 3B model, performance is strong; the approach effectively transfers to deep research browsing.
  - Data effectiveness ablation (Table 5):
    - Quote: “Training Set NQ+HQA → 3.0%; InfoSeek → 16.5%”
    - Interpretation: Purpose-built InfoSeek data is essential for deep research capabilities; generic QA data is insufficient.
  - Dataset difficulty and cost (Table 2):
    - Quote: “Total 52,138; failure rate (Qwen2.5-72B, CoT) 91.6% overall; cost $571.8”
    - The monotonic rise in failure rate with more vertices (from 88.1% at 3 vertices to 94.1% at ≥7) empirically validates structure-controlled difficulty.
- Do the experiments support the claims?
  - Yes, on three fronts:
    - Structure-controlled difficulty is demonstrated in Table 2.
    - Training on InfoSeek improves broad QA performance (Table 3) and deep research browsing (Table 4–5).
    - A compact model can rival or surpass larger or proprietary systems on difficult browsing tasks (Table 4).
- Observed trade-offs and mixed results:
  - On single-hop `TQA` and `NQ`, some baselines edge out InfoSeeker-3B (Table 3). Gains are strongest on multi-hop and compositional datasets and in deep research browsing, which aligns with the dataset’s design intent.

## 6. Limitations and Trade-offs
- Scope of tasks:
  - Focuses on questions with unique, verifiable answers; does not cover open-ended outputs like long-form reports or creative synthesis (Sec. 2.2; Conclusion).
  - The reasoning structure is constrained to trees (acyclic); real investigations can require graph structures with cycles and revisiting prior nodes (Sec. 2.3).
- Dataset and knowledge source bias:
  - Built primarily from Wikipedia and sampled webpages; topics underrepresented online or requiring domain-specific sources may be less well covered (Sec. 3.1).
- Quality checks rely on strong LLMs:
  - Difficulty and verifiability filters use Qwen2.5-32B and Gemini 2.5 Flash (Sec. 3.5), which may introduce biases. Although processes are described, replicating results without similar APIs could be challenging.
- Training signals:
  - RL reward is a simple binary success metric (Sec. 4.3.2). This yields a clear objective but weak credit assignment for intermediate decisions; stronger, trajectory-level or retrieval-quality rewards are suggested but not implemented (Conclusion).
- Efficiency/cost at inference:
  - Multi-query search with a separate Refiner Agent increases tool calls (BrowseComp-Plus shows 8.24 search calls for InfoSeeker-3B; Table 4). Although more efficient than naive large-context approaches, this still incurs latency/cost in production settings.
- Evaluation:
  - BrowseComp-Plus relies on LLM-as-judge (Sec. 5.1), which can introduce evaluation noise. The paper follows official settings, but this is a general limitation of such benchmarks.

## 7. Implications and Future Directions
- Impact on the field:
  - Provides a rigorous target—`HCSP`—for Deep Research and an open, scalable pipeline to produce such data. This scaffolds research on hierarchical reasoning, retrieval-aware RL, and agent orchestration with reproducible resources (Table 1; Conclusion).
  - Demonstrates that small models (3B) can perform competitively on complex browsing tasks when trained on the right data and workflows (Table 4–5), suggesting a path to cost-effective, deployable research assistants.
- Enabled follow-up research:
  - Reward design: leverage the preserved meta-information (intermediate steps, retrieval labels) to build compound rewards that score step correctness, retrieval precision/recall, and trajectory efficiency (Conclusion).
  - Beyond trees: generalize from trees to directed acyclic graphs or even cyclic reasoning, enabling re-checking and hypothesis revision.
  - Robustness and safety: analyze susceptibility to misinformation, adversarial webpages, and retrieval noise; integrate fact-checking agents.
  - Multilingual and domain-specialized variants: extend synthesis to non-English sources and specialized corpora (e.g., biomedical, legal) with tailored constraint taxonomies.
- Practical applications:
  - Enterprise research assistants that verify answers by citing internal documents and external sources.
  - Academic literature review tools that decompose complex queries into verifiable sub-claims.
  - Policy and market analysis systems that integrate multi-source evidence while providing auditable reasoning trails.

> In short, this work reframes Deep Research as a hierarchical, verifiable reasoning task (HCSP), supplies the first large-scale open dataset tailored to that framing (50k+ questions; Table 2), and offers a practical agent and training pipeline that demonstrably improves performance on difficult browsing evaluations (Table 4–5).
