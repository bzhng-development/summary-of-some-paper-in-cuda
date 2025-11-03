# LIMI: Less Is More for Agency

**ArXiv:** [2509.17567](https://arxiv.org/abs/2509.17567)
**Authors:** Yang Xiao, Mohan Jiang, Jie Sun, Keyu Li, Jifan Lin, Yumin Zhuang, Ji Zeng, Shijie Xia, Qishuo Hua, Xuefeng Li, Xiaojie Cai, Tongyu Wang, Yue Zhang, Liming Liu, Xia Wu, Jinlong Hou, Yuan Cheng, Wenjie Li, Xiang Wang, Dequan Wang, Pengfei Liu
**Institutions:** 

## 🎯 Pitch

Introducing LIMI, a transformative training paradigm demonstrating that high-quality, curated demonstrations can achieve superior autonomous-agent performance with significantly less data. By surpassing models trained on vastly larger datasets, LIMI offers a sustainable, efficient approach to developing capable AI agents, meeting industries' real-world demands while challenging data-heavy conventions.

---

## 1. Executive Summary (2-3 sentences)
This paper introduces LIMI (“Less Is More for Intelligent Agency”), a training recipe showing that strong autonomous-agent capabilities can emerge from a very small number of carefully curated demonstrations. With only 78 multi-turn “agentic” trajectories focused on software development (“vibe coding”) and scientific research workflows, LIMI fine-tuned models achieve state-of-the-art results on AgencyBench (73.5% average), outperforming much larger models and models trained on 7,600–10,000 agent datasets (see Table 2 and Figure 1).

## 2. Context and Motivation
- Problem addressed
  - Modern large language models (LLMs) “think” well but often fail to “work” as autonomous agents that can decompose tasks, use tools, and complete multi-step goals in realistic environments. The paper formalizes this as `Agency`: “the emergent capacity of AI systems to function as autonomous agents—actively discovering problems, formulating hypotheses, and executing solutions through self-directed engagement with environments and tools” (Abstract; Section 1).
  - A prevailing assumption is that agency improves mainly by scaling data, mirroring language modeling scaling laws (Section 1).

- Why it matters
  - Real-world demand: “industries demand autonomous agents that can execute tasks, operate tools, and drive real-world outcomes” (Section 1). Data- and compute-heavy training pipelines make this prohibitively expensive and environmentally costly.
  - Theoretical importance: if agency follows different scaling laws than language modeling, training paradigms should change. The paper proposes an “Agency Efficiency Principle,” i.e., autonomy emerges from strategically curated high-quality agentic demonstrations, not from sheer data abundance (Abstract; Section 1).

- Prior approaches and their gaps
  - Tool use and agent frameworks (Toolformer, ReAct, AutoGPT; Section 5.1) demonstrate acting/reasoning, but typically rely on large synthetic corpora and heavy compute (Section 5.1).
  - Data-efficiency precedents exist for alignment/reasoning—LIMA (~1,000 curated pairs) and LIMO (817 math samples) (Section 1; Section 5.2)—but not for “agentic” multi-turn, tool-mediated workflows.

- Positioning
  - LIMI focuses on quality-over-quantity, curating 78 long-horizon, tool-using trajectories from two high-yield domains—vibe coding and research workflows (Sections 2.2, 3.2)—and shows these suffice to train strong, general agent behavior.

## 3. Technical Approach
This section explains both the data and the training/evaluation protocol. The core idea is to collect complete, high-signal trajectories that show how successful agents behave end-to-end in realistic environments.

- Data objects and interaction formalization (Section 3.1)
  - Each training example is a tuple `(q_i, τ_i)`:
    - `q_i`: a natural-language user request initiating a realistic, multi-step task (e.g., implementing a Gomoku AI or running a research analysis).
    - `τ_i`: a full multi-turn trajectory `τ_i = {a_{i,1}, …, a_{i,n_i}}` composed of:
      - `m_{i,j}` (model reasoning): explicit thoughts like analysis, plans, and decisions.
      - `t_{i,j}` (tool calling): structured invocations of external tools (e.g., shell, code runner, web).
      - `o_{i,j}` (environment observation): tool outputs and human feedback; these guide the next step.
  - Purpose: Teach models not only target outputs but also how to orchestrate tools, recover from errors, and iterate plans under feedback.

- Query pool construction (Section 3.2)
  - Real-world queries: 60 tasks from professional developer and researcher scenarios; research tasks come from real papers to ensure authenticity (Section 3.2).
  - GitHub PR-based synthesis: systematically generate additional realistic queries from pull requests using GPT-5, with careful filtering and review (Section 3.2):
    - Repository filter: 100 repos with >10,000 stars.
    - Domain coverage: frontend, backend, infra, debugging, optimization.
    - Complexity filter: unified diff patch <1,200 tokens; exclude Markdown-only PRs.
    - Sampling: from 1,000 PRs per repo, sample 100 per repo to remain representative; expert annotators (four CS PhDs) validate semantic alignment; from the large synthetic pool, select 18 that best match the two target domains.
  - Final training set: `Q = {q_1, …, q_78}` (60 real + 18 PR-derived) across vibe coding and research workflows; Figure 4 (right) visualizes domain coverage.

- Trajectory collection in a realistic environment (Section 3.3)
  - Execution environment: SII CLI (selected over Claude Code and Gemini CLI for best tool integration, logging, and collaboration support).
  - Procedure: four PhD annotators collaborate with GPT-5 in SII CLI to complete each query end-to-end, repeatedly collecting full trajectories until success. This captures authentic “back-and-forth” collaboration, error recovery, and tool use.
  - Interaction length and richness: trajectories are long (min 13k tokens, max 152k, average 42.4k; Figure 4 left), emphasizing high signal per example.
  - Why this matters: Instead of shallow, single-turn data, each trajectory teaches complete multi-step workflows—planning, tool orchestration, iterative refinement—so fewer examples can still “cover” many agentic behaviors.

- Target domains (Section 2.2)
  - `Vibe coding`: collaborative software development in realistic codebases and toolchains (planning, navigating repos, coding, debugging).
  - `Research workflows`: literature/data analysis, experiment design, model evaluation, and scientific reporting.

- Model training and variants (Section 4.1)
  - Base models: `GLM-4.5` (355B) and `GLM-4.5-Air` (106B).
  - Fine-tuning framework: `slime` SFT framework for consistent, reproducible training across datasets (Section 4.1).
  - Model variants:
    - `LIMI`: GLM-4.5 fine-tuned on the 78 LIMI trajectories.
    - `LIMI-Air`: GLM-4.5-Air fine-tuned on the same 78 trajectories.
    - Comparators with the same base model but trained on alternative datasets:
      - `GLM-4.5-CC` on CC-Bench trajectories (260 samples).
      - `GLM-4.5-Web` on AFM-WebAgent-SFT (7,610 samples).
      - `GLM-4.5-Code` on AFM-CodeAgent-SFT (10,000 samples).

- Evaluation methodology (Sections 3.4, 4.1)
  - Primary benchmark: `AgencyBench` (Table 1 lists 10 tasks across coding and research).
    - Metrics (Section 3.4):
      - `FTFC` (First-Turn Functional Completeness): fraction of requirements already correct after the first response.
      - `SR@R` (Success Rate within R rounds): fraction of tasks completed within the interaction budget; the paper uses `R=3`.
      - `RC@R` (Remaining Chances at R): number of unused rounds when success occurs (higher = more efficient).
  - Generalization benchmarks (Section 3.4; Table 3):
    - Tool-use: `tau2-bench-airline` and `tau2-bench-retail` with `Pass^4` (success rate across 4 independent runs).
    - Coding: EvalPlus-HumanEval and EvalPlus-MBPP (accuracy).
    - Data science: DS-1000 (accuracy).
    - Scientific computing: SciCode (Main Problem and Sub Problem metrics, accuracy).
  - Two evaluation conditions (Section 4.1):
    - With SII CLI tool access (full agent mode).
    - Without tool access (to measure intrinsic reasoning gains).

## 4. Key Insights and Innovations
1) Agency Efficiency Principle (fundamental innovation)
- What’s new: Demonstrates that “machine autonomy emerges not from data abundance but from strategic curation of high-quality agentic demonstrations” (Abstract).
- Why it matters: LIMI achieves 73.5% average on AgencyBench with 78 examples, outperforming a GLM-4.5 model trained on 10,000 agent SFT samples (47.8%)—a “53.7% gain” in Figure 1 (right) and a +25.7 absolute point increase in Table 2.
- Difference from prior work: Extends the “less is more” theme (LIMA/LIMO) from alignment/reasoning to multi-turn, tool-mediated agency—a harder setting involving planning, tool use, and error recovery.

2) Full-trajectory, high-signal demonstrations (methodological innovation)
- What’s new: Each example includes explicit reasoning, tool invocations, and environment feedback across long horizons (Section 3.1; Figure 4 left).
- Why it matters: The model learns procedural knowledge—how to plan, call tools, and adapt to feedback—rather than only end outputs. This richness allows strong learning from few samples.

3) Targeted domain focus on vibe coding and research workflows (design innovation)
- What’s new: Curating dense, high-yield tasks that “span the majority of knowledge work scenarios” (Section 2.2) and naturally require planning and tool orchestration.
- Why it matters: Concentrating on representative, high-complexity work reduces sample needs while covering broad agentic competencies.

4) PR-based query synthesis with quality control (data innovation)
- What’s new: Generate realistic agent tasks from GitHub PRs using a pipeline with repository filtering, complexity thresholds, and expert verification (Section 3.2).
- Why it matters: Maintains ecological validity; avoids synthetic drift and ensures tasks reflect real development challenges.

## 5. Experimental Analysis
- Evaluation setup recap
  - Primary: AgencyBench (10 tasks; Table 1) with FTFC, RC@3, SR@3; interaction rounds R=3 (Section 3.4).
  - Generalization: tau2-bench, EvalPlus HE/MBPP, DS-1000, SciCode (Table 3).
  - With-vs-without tool access (Tables 3 and 4).

- Main quantitative results
  - Headline (AgencyBench; Table 2; also Figure 1 left):
    > LIMI (78 samples, GLM‑4.5 base) achieves 71.7 FTFC, 74.2 RC@3, 74.6 SR@3, averaging 73.5.
    - Best baseline GLM-4.5 (no fine-tune) averages 45.1; Kimi-K2 24.1; Qwen3-235B-A22B 27.5; DeepSeek-V3.1 11.9.
    - Absolute gains vs strongest baseline: +28.4 points (73.5 vs 45.1).
    - First-turn advantage: FTFC 71.7 vs 37.8 for GLM-4.5 (+33.9 points), showing stronger initial plans.

  - Data efficiency comparisons (Table 2, lower block):
    > GLM‑4.5‑Code (10,000 samples): 47.8; GLM‑4.5‑Web (7,610 samples): 36.7; GLM‑4.5‑CC (260 samples): 29.2; LIMI (78 samples): 73.5.
    - Fewer samples but far higher performance—illustrating the “Less-Is-More” claim with concrete numbers.

  - Generalization with tools (Table 3; average includes AgencyBench per the table note):
    > LIMI achieves 92.1 on EvalPlus‑HumanEval, 82.3 on EvalPlus‑MBPP, 36.6 on DS‑1000, and Pass^4 of 34.0/45.6 on tau2‑bench airline/retail; average 57.2.
    - Outperforms GLM-4.5 baseline average (43.0) and other baselines (Kimi-K2: 37.3; Qwen3: 36.7; DeepSeek-V3.1: 29.7).

  - Tool-free evaluation (Table 4):
    > Without CLI access, LIMI averages 50.0 vs GLM‑4.5’s 48.7 and outperforms all external baselines.
    - The tool-free gain (about +1.3 points over GLM‑4.5) is smaller than with tools, but still positive—evidence that LIMI improves intrinsic reasoning, not only tool orchestration.

  - Cross-scale generalization (Table 2, middle block):
    > LIMI‑Air (106B) improves GLM‑4.5‑Air from 17.0 to 34.3 on AgencyBench (+17.3 points).
    - Suggests the approach transfers across model sizes.

- Evidence coverage and convincingness
  - The combination of:
    - Strong SOTA-like AgencyBench scores on a wide set of tasks (Table 2; Table 1),
    - Comparisons against three large agent SFT datasets (Table 2; Table 3),
    - Tool-free evaluations (Table 4),
    - And cross-scale replication (Table 2),
    provides a convincing empirical case that high-quality, full-trajectory, low-count data can be more valuable than large-scale SFT for building agentic capabilities.

- Ablations, failure cases, robustness
  - Dataset ablation-by-proxy: training on CC/Web/Code vs LIMI shows clear performance separation (Tables 2–3).
  - The paper includes qualitative case studies (Appendix C) indicating where base GLM-4.5 struggled (e.g., timeout errors, failure to implement AI difficulty in Gomoku) while LIMI succeeded; these are illustrative but anecdotal.
  - Mixed outcomes:
    - Scientific computing remains low across all models (SciCode MP ~3.1 for LIMI; Table 3), indicating headroom.
    - Tool-free gains are modest (Table 4), suggesting much of the lift manifests when tools are available.

- Notable configuration choices
  - AgencyBench interaction budget R=3 (Section 3.4) stresses early planning quality, which aligns with LIMI’s FTFC advantage.

## 6. Limitations and Trade-offs
- Assumptions about data quality and domains
  - The paradigm assumes it’s possible to design a small number of very high-signal tasks that “span” agentic behavior. LIMI targets two domains (vibe coding, research workflows; Section 2.2); transfer to other domains (e.g., robotics, enterprise workflows with unique tools) is untested here.

- Potential dataset/test coupling
  - AgencyBench (Table 1) is from the same broader research ecosystem (Section 3.4 references Li et al., 2025b). While the paper emphasizes distinct training queries, the proximity of training domain focus and benchmark domains could inflate gains. A clearer statement on potential overlap or leakage would strengthen the case.

- Collection cost vs. sample count
  - Although sample count is small (78), each trajectory is long (average 42.4k tokens; max 152k; Figure 4 left) and curated through PhD–model collaboration (Section 3.3). This is “data efficient,” but not necessarily “cheap” to collect.

- Reproducibility details
  - The paper specifies a common SFT framework (`slime`) and base models but does not detail hyperparameters, compute budget, or training durations (Section 4.1 only provides framework-level assurances). Reproduction may require additional disclosures.
  - Many additional PR-derived queries are not yet released; only the 78 LIMI samples drive the reported results (Section 3.2 mentions future release plans), limiting immediate community validation.

- Evaluation scope
  - Tool-free results improve modestly (Table 4), suggesting much of LIMI’s strength is realized when an execution environment is available; real-world deployments without rich tools may see smaller gains.
  - Some generalization tasks (SciCode) remain challenging across models (Table 3), indicating incomplete coverage of scientific computing skills.

## 7. Implications and Future Directions
- How this changes the landscape
  - The results challenge the default “more data is better” mindset for training agentic LLMs. The Agency Efficiency Principle suggests investing in curating whole-process demonstrations (planning → tool use → feedback → recovery → completion) may be a more scalable path to capable agents than simply synthesizing more SFT data.

- Practical applications
  - Targeted agent training for enterprise workflows: software maintenance, analytics, and research assistants that must navigate internal tools and policies.
  - Platform design: prioritize environments (like SII CLI) that record reasoning, tool calls, and observations to continuously “harvest” high-signal trajectories from real usage.

- Follow-up research opportunities
  - Automated curation: develop algorithms to select the “next most informative” trajectory to collect or fine-tune on (active data selection for agency).
  - Broader domains: replicate LIMI in other long-horizon settings (e.g., data engineering, MLOps, biomedical curation, robotics simulators).
  - Learning algorithms: combine LIMI-style SFT with reinforcement learning or credit assignment over trajectories; explore how far few-shot agentic reinforcement can go when seeded with high-quality demonstrations.
  - Generalization stress tests: evaluate on benchmarks from independent groups and in zero-shot tool ecosystems to quantify domain transfer and robustness.
  - Open science: release more of the synthesized PR-derived tasks and detailed training settings to enable controlled replication and meta-analysis.

> Key take-away (Figure 1; Tables 2–4): with only 78 curated, long-horizon trajectories, LIMI-trained models attain 73.5% on AgencyBench, surpass models trained on 7,600–10,000 agent SFT samples, improve coding/tool-use generalization, and retain modest gains even without tools. The evidence supports a strategic shift toward quality, completeness, and realism in agent training data.
