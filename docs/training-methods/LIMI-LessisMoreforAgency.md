# LIMI: Less is More for Agency

**ArXiv:** [2509.17567](https://arxiv.org/abs/2509.17567)

## 🎯 Pitch

LIMI ('Less Is More for Agency') fundamentally challenges the prevailing belief that large-scale data is required to cultivate sophisticated autonomous AI agents. By strategically curating just 78 high-quality agentic demonstrations, LIMI enables a large language model to outperform state-of-the-art baselines trained on datasets up to 128 times larger, achieving remarkable agentic intelligence on real-world collaborative tasks. This breakthrough reveals that the essence and quality of demonstrations, not sheer data volume, are the key to developing practical, autonomous 'working AI' for complex environments—a finding poised to reshape the principles of building truly agentic AI systems.

---

## 1. Executive Summary (2-3 sentences)
This paper introduces LIMI (“Less Is More for Intelligent Agency”), a training paradigm showing that strong autonomous-agent behaviors can be learned from a very small number of carefully curated demonstrations. With only 78 long, high-quality interaction trajectories focused on collaborative coding and research workflows, LIMI fine-tunes a large model to 73.5% on AgencyBench, outperforming much larger state-of-the-art models and even models trained on 7.6k–10k agent datasets (Figure 1; Table 2).

## 2. Context and Motivation
- Problem/gap
  - The work targets “agentic” capability: the ability of an AI system to act autonomously over multiple steps, use tools, and interact with an environment, rather than only answer questions. The paper defines Agency as an emergent capacity to discover problems, form hypotheses, and execute solutions via self-directed interaction with tools and environments (Section 1).
  - A common assumption is that better agency requires ever-larger training datasets (extending language-model scaling laws to agent training). This has not been rigorously tested.

- Why this matters
  - Practical impact: Industry needs “working AI” that can complete tasks end-to-end (software development, research assistance), not just “thinking AI” that outputs answers (abstract, Section 1).
  - Scientific significance: If agentic abilities depend more on the quality and structure of demonstrations than on data volume, this reshapes how the field should build autonomous systems (Abstract; Sections 1, 6).

- Prior approaches and their limits
  - Large-scale data synthesis and agentic RL have dominated recent systems (Section 5.1), but they are compute- and data-intensive, and whether scale alone is the key driver of agency is unclear.
  - Evidence from adjacent areas—alignment (LIMA) and math reasoning (LIMO)—suggests small, curated sets can outperform large datasets (Section 1, citing Zhou et al., 2023; Ye et al., 2025). Whether the same holds for agency was open.

- Positioning
  - LIMI focuses on two “knowledge-work” domains that need full agentic behavior: “vibe coding” (collaborative software development) and “research workflows” (literature/data/analysis tasks). The paper builds a data construction and training pipeline around these domains and evaluates comprehensively on AgencyBench plus generalization benchmarks (Sections 2.2, 3, 4).

## 3. Technical Approach
The approach is primarily empirical and data-centric: instead of scaling dataset size, it designs and curates a small number of dense, complete trajectories that encode the full agentic process end-to-end.

- Core definitions (Section 3.1)
  - `query qi`: a natural-language task request (e.g., “build a Gomoku app with replay and AI”).
  - `trajectory τi = {ai,1, …, ai,ni}`: the complete multi-turn interaction to solve qi. Each action `ai,j` is one of:
    - `model reasoning (mi,j)`: plans, analyses, decisions the model outputs.
    - `model tool calling (ti,j)`: structured invocations of tools (e.g., a CLI command).
    - `environment observation (oi,j)`: outputs/feedback from tools or the user.
  - Why this matters: learning from full trajectories (not only final answers) exposes state tracking, error recovery, planning, tool orchestration, and collaboration patterns that define agency (Sections 2.1, 3.1).

- Domain focus and task design (Sections 2.2, 3.2; Figure 2; Table 1)
  - Two domains:
    - `Vibe coding`: collaborative coding in real repositories and dev environments.
    - `Research workflows`: dataset search, analysis, experiment design/metrics, reporting.
  - The paper argues long-horizon tasks in these domains provide “dense learning signals” across planning, execution, and collaboration. Figure 2 exemplifies one query (Gomoku) decomposed into five escalating subtasks (UI, rules, state, heuristic AI, search-based AI).

- Query pool construction: 78 total (Section 3.2; Figure 3, right; Figure 4, right)
  - `60 real-world queries`: gathered from professional developers/researchers; several research tasks stem from real academic projects (Section 3.2).
  - `18 synthesized queries from GitHub PRs`: using a PR-to-query pipeline with GPT-5 to preserve semantic fidelity to real code changes. Steps include:
    - Repository selection: 100 repos with >10k stars.
    - Domain diversification: front-end, back-end, deployment, debugging, optimization.
    - Complexity filtering: PR patch size <1,200 tokens; exclude Markdown-only changes.
    - Sampling: 100 PRs per repo; sample 100/pr for query synthesis.
    - Expert QA: four PhD annotators validate that each query aligns with its PR (Section 3.2).
  - The final 18 are chosen to best match the two target domains. The rest are planned for future release.

- Trajectory collection environment and protocol (Section 3.3; Figure 3, right; Figure 4, left)
  - `SII CLI`: a command-line ecosystem with integrated tools for both domains and detailed logging to capture complete trajectories (Section 3.3; Section 3.4 “Execution Environment Selection”).
  - Human–AI collaboration: four PhD annotators work with GPT-5 through SII CLI to complete each query. Trajectories are collected iteratively “until successful completion,” so they include retries, refinements, tool outputs, and corrections (Section 3.3).
  - Scale and richness: trajectories average 42.4k tokens and can be very long (max 152k), indicating high interaction complexity (Figure 4, left).

- Training setup (Section 4.1)
  - Base models: `GLM-4.5 (355B)` and `GLM-4.5-Air (106B)`.
  - Fine-tuning framework: `slime` for supervised fine-tuning (SFT), with matched training configurations for fair comparisons (Section 4.1).
  - Variants:
    - `LIMI`: GLM-4.5 fine-tuned on the 78 LIMI trajectories.
    - `LIMI-Air`: GLM-4.5-Air fine-tuned on the same 78 trajectories.
    - Comparison fine-tunes on alternative datasets:
      - `GLM-4.5-CC`: 260 samples from CC-Bench-trajectories.
      - `GLM-4.5-Web`: 7,610 samples from AFM-WebAgent-SFT.
      - `GLM-4.5-Code`: 10,000 samples from AFM-CodeAgent-SFT.

- Evaluation methodology (Sections 3.4, 4.1; Table 1; Tables 2–4)
  - `Primary`: AgencyBench (10 multi-subtask tasks across coding and research; Table 1).
  - Metrics on AgencyBench (Section 3.4):
    - `FTFC (First-Turn Functional Completeness)`: fraction of requirements implemented in the first response.
    - `SR@R (Success Rate within R rounds)`: solved within at most R interaction rounds; R=3 here.
    - `RC@R (Remaining Chances)`: average unused rounds when success occurs (measures efficiency).
  - `Generalization benchmarks` (Table 3):
    - Tool use: `tau2-bench-airline`, `tau2-bench-retail` (Pass^4, i.e., success in any of 4 independent runs).
    - Code: `EvalPlus-HumanEval`, `EvalPlus-MBPP`, `DS-1000`.
    - Scientific computing: `SciCode` (Main Problem and Sub Problem).
  - With- and without-environment comparisons: generalization benchmarks are also run without SII CLI to isolate tool-free reasoning effects (Table 4).

- Why these design choices?
  - The pipeline prioritizes “learning from success” in realistic long-horizon collaborations, hypothesized to encode agentic patterns better than large quantities of short or synthetic traces (Sections 1–2; Figure 2; Figure 4).
  - Focusing on two high-utility domains gives coverage of the most common knowledge-work agent tasks while remaining tractable for careful curation (Section 2.2; Figure 4, right).

## 4. Key Insights and Innovations
- Agency Efficiency Principle (fundamental)
  - Claim: For agency, curation quality of full trajectories beats dataset scale. Evidence is strong: with only 78 samples, LIMI achieves 73.5% on AgencyBench vs. 47.8% for a model trained on 10,000 samples (Table 2). Figure 1 visualizes the gap and “53.7% improvement.”
  - Why it matters: This challenges the default “more data is better” assumption for agent training and reframes the goal as “collect the right interactions, not just more of them.”

- Complete trajectory formalization and capture (important, method-level)
  - The `query–trajectory` tuple with three action types (reasoning, tool calls, environment observations) ensures models see the full loop of plan–act–observe–revise (Section 3.1). Figure 4 shows trajectories are long and varied, implying rich state, planning, and error-recovery signals.
  - Significance: It operationalizes what “agentic supervision” should include and provides a reusable template for future datasets.

- Real-world plus PR-synthesized queries with expert QA (incremental but practical)
  - Mixing 60 real-world tasks with 18 high-fidelity PR-derived tasks balances authenticity and coverage (Section 3.2). The PR pipeline includes repository selection, complexity filtering, sampling, and human quality checks.
  - Significance: A concrete and replicable way to synthesize high-quality agent tasks directly from real code change contexts.

- Environment-aware evaluation and tool-free checks (methodological rigor)
  - The paper separates improvements from tool access vs. intrinsic reasoning by evaluating without SII CLI (Table 4), showing LIMI still improves average performance (50.0% vs. 48.7% for GLM-4.5). This helps attribute gains to the training data rather than only to environment coupling (Sections 4.4–4.5).

## 5. Experimental Analysis
- Setup recap (Sections 4.1–4.2; Tables 1–4)
  - Models compared: very strong baselines (Kimi-K2-Instruct 1T, DeepSeek-V3.1 671B, Qwen3-235B-A22B-Instruct 235B, GLM-4.5 355B), plus GLM-4.5 variants trained on 260/7.6k/10k agent datasets.
  - Metrics: AgencyBench (FTFC, RC@3, SR@3); generalization across tau2-bench, EvalPlus, DS-1000, SciCode.

- Main AgencyBench results (Table 2; Figure 1)
  - Quote:
    > Table 2: LIMI (355B, 78 samples) achieves FTFC 71.7, RC@3 74.2, SR@3 74.6, AVG 73.5.  
    > GLM-4.5 baseline: AVG 45.1 (FTFC 37.8, SR@3 47.4).  
    > Kimi-K2-Instruct: 24.1; DeepSeek-V3.1: 11.9; Qwen3-235B-A22B-Instruct: 27.5.
  - Interpretation:
    - A 28.4-point absolute gain over the strong GLM-4.5 baseline on AVG.
    - Particularly large jump in first-turn completeness (FTFC +33.9 points), suggesting LIMI learns to “do the right thing early,” which is a hallmark of strong planning and specification adherence.

- Data efficiency comparisons (Table 2, “Data Efficiency” block; Figure 1 right)
  - Quote:
    > GLM-4.5-Code (10,000 samples): 47.8 AVG; GLM-4.5-Web (7,610): 36.7; GLM-4.5-CC (260): 29.2.  
    > LIMI (78 samples): 73.5 AVG.
  - Interpretation:
    - Despite 128× fewer samples than AFM-CodeAgent (10k), LIMI improves AgencyBench AVG by +25.7 points (relative ≈54%).
    - This strongly supports the central claim that “less but better” data drives agency more effectively than sheer volume.

- Generalization benchmarks (Table 3)
  - Quote:
    > LIMI reaches EvalPlus-HumanEval 92.1 and EvalPlus-MBPP 82.3; tau2-bench-airline 34.0, tau2-bench-retail 45.6; DS-1000 36.6; SciCode-MP 3.1; SciCode-SP 25.3.  
    > Average reported is 57.2 (Table 3 notes the average includes AgencyBench performance for a comprehensive view).
  - Interpretation:
    - Coding performance is very strong (EvalPlus HE/MBPP). Tool-use outcomes are competitive and generally better than baselines. SciCode remains low across all models, reflecting difficulty in scientific coding benchmarks.
  - Against GLM-4.5 baseline:
    - LIMI outperforms GLM-4.5 across the board on the averaged metric (57.2 vs. 43.0; Table 3), with notable gains on tau2-bench-retail (+8.8 points) and coding tasks.

- Tool-free evaluation (Table 4)
  - Quote:
    > Without SII CLI access, LIMI averages 50.0; GLM-4.5 averages 48.7. External baselines are lower (Kimi-K2 40.3; DeepSeek 36.5; Qwen3 37.3).
  - Interpretation:
    - Gains persist without tools, indicating the data curation improved intrinsic reasoning and planning, not only tool orchestration.

- Cross-scale effect (Table 2, “Generalization” block)
  - Quote:
    > LIMI-Air (106B) rises from 17.0 to 34.3 on AgencyBench compared to its base; LIMI (355B) from 45.1 to 73.5.
  - Interpretation:
    - The approach scales across model sizes, suggesting the curated trajectories capture general patterns of agentic behavior.

- Qualitative evidence (Appendix C)
  - Case study narratives show LIMI solves multi-step coding tasks more reliably, needs fewer hints, and produces better search/analysis outcomes on research/dataset-discovery tasks (Appendix C.1–C.2).

- Do the experiments support the claims?
  - Strengths:
    - Strong, consistent quantitative gains on the primary agentic benchmark and several generalization tasks (Tables 2–4).
    - Careful controls against alternative large training sets on the same base model (GLM-4.5), isolating the effect of LIMI’s data design.
    - Tool-free evaluations to separate environment effects.
  - Caveats:
    - The “78 samples” headline masks high token counts per trajectory (avg 42.4k; Figure 4). A token-normalized comparison against the 7.6k–10k datasets is not provided, which could clarify whether “less is more” in tokens, not only in sample count.
    - Details such as hyperparameters, compute budget, and exact SFT recipe are light; while the slime framework is mentioned, reproducibility might need more granularity (Section 4.1).
    - AgencyBench is closely aligned with the training domains; broader domains (e.g., robotics, enterprise tools) remain to be tested.

## 6. Limitations and Trade-offs
- Assumptions and scope
  - The training data targets two domains (“vibe coding” and “research workflows”), assuming these cover “the majority of knowledge work” (Section 2.2). Tasks far from these domains may not benefit similarly.
  - Success-first trajectory collection (“until successful completion”) may bias the data toward best-case procedures and reduce exposure to persistent failure or adversarial environments (Section 3.3).

- Potential confounds and missing controls
  - No token- or time-normalized efficiency comparison vs. large-scale datasets; counting samples alone may understate the effective data volume encoded in long trajectories (Figure 4).
  - Using GPT-5 to synthesize PR-based queries (Section 3.2) introduces a risk of stylistic coupling to GPT-5-generated tasks; although expert QA mitigates this, cross-origin validation would help.

- Environment coupling and evaluation breadth
  - The SII CLI is both the trajectory source and part of evaluation; although tool-free tests help, independent environments would further reduce coupling concerns (Sections 3.3–3.4, 4.5).
  - Some benchmarks (e.g., SciCode) still show low absolute scores across models, highlighting room for improvement in scientific computing agents (Table 3).

- Practical constraints
  - Data collection requires expert annotators and a sophisticated CLI with logging, which may limit replication until the tooling and data are broadly released (Sections 3.2–3.3).

## 7. Implications and Future Directions
- How this changes the landscape
  - If agentic capabilities scale with quality and completeness of trajectories rather than dataset size, the field should redirect effort from mass data synthesis to strategic curation and environment design. This reframes the “agent scaling law” from volume-centric to information- and structure-centric (Abstract; Sections 1, 6).

- Follow-up research enabled/suggested
  - Token- and step-normalized scaling studies: quantify “less is more” as a function of tokens, states visited, and unique tool interactions to formalize an “agency information law.”
  - Cross-domain replication: apply LIMI to other agent domains (e.g., data engineering, bioinformatics, robotics control) and to different base model families.
  - Ablations within LIMI data: vary the number of queries, trajectory length, success vs. failure traces, and degree of human intervention to identify the most causally important ingredients.
  - Beyond SFT: combine curated trajectories with preference optimization or offline RL tailored to long-horizon POMDP settings (Section 5.1 discussion), and test whether LIMI-style curation complements RL.

- Practical applications
  - Enterprise coding assistants that can own tickets end-to-end (triage → implement → test → PR) using curated development traces.
  - Research copilots that execute full workflows (dataset search, evaluation, statistical analysis, report generation) as in Tasks 5–7 and Task 6’s metric suite (Table 1; pages 20–22).
  - Education and training: use LIMI’s trajectory structure to teach planning and tool use in safe sandboxes before deployment.

> Most central quantitative takeaway (Table 2; Figure 1): “LIMI reaches 73.5% on AgencyBench with only 78 curated samples,” outperforming strong baselines and models trained on 7.6k–10k-agent datasets. The curated, long-horizon trajectories—capturing reasoning, tool calls, and environment feedback—appear to be the key driver of agentic competence.
