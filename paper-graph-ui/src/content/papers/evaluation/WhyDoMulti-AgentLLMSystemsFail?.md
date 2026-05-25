# Why Do Multi-Agent LLM Systems Fail?

**ArXiv:** [2503.13657](https://arxiv.org/abs/2503.13657)

## 🎯 Pitch

This paper delivers the first systematic investigation into why multi-agent large language model (LLM) systems frequently underperform, introducing MAST—a 14-mode, empirically grounded taxonomy of failure types—and accompanying it with MAST-Data, a publicly available dataset of 1,642 annotated execution traces. By releasing both the taxonomy and dataset alongside a scalable LLM-as-a-judge annotation pipeline, the authors provide the entire community with practical tools for diagnosing, comparing, and mitigating MAS failures, paving the way for more robust multi-agent AI systems and accelerating research progress in collaborative intelligence.

---

## 1. Executive Summary (2-3 sentences)
This work explains why Multi‑Agent Large Language Model Systems (MAS) so often fail and provides concrete tools to diagnose and reduce those failures. It introduces (i) MAST, a 14‑mode taxonomy of MAS failures mapped to conversation stages (Figure 1), (ii) MAST‑Data, a public corpus of 1,642 annotated MAS execution traces across 7 frameworks and multiple tasks (Table 1), and (iii) an LLM‑as‑a‑Judge annotator that scales failure labeling with high agreement to human experts (Table 2).

## 2. Context and Motivation
- Problem/gap addressed
  - MAS—systems where multiple LLM “agents” interact to solve tasks—are increasingly popular for coding, math, web/desktop tasks, and general assistants. Yet their gains over single‑agent systems are often marginal, and failure rates remain high. The paper documents failures between 41% and 86.7% across six popular systems (Figure 5).
  - There is no shared, fine‑grained framework for what “failure” means in MAS or how to diagnose root causes. Without common definitions, teams cannot compare systems, reproduce findings, or direct engineering effort.

- Why it matters
  - Practical: MAS are being deployed in software engineering, web automation, and research assistance. Failures create wasted compute, unreliable behavior, and user risk (Sections 1, 2.1).
  - Scientific: Understanding failure patterns in multi‑agent coordination, memory, and verification reveals where architectural or training changes are most needed (Sections 1, 4).

- Prior approaches and their limits
  - Single‑system design advice (e.g., modularity, keep frameworks simple) offers principles but not a cross‑system error taxonomy (Related Work 2.2).
  - Benchmarks target overall performance (e.g., SWE‑Bench, GAIA, AppWorld) but do not explain why failures happen (Related Work 2.1, 2.3).
  - Debugging tools exist (e.g., interactive agent debuggers) but lack a standardized failure vocabulary to aggregate insights across systems (Related Work 2.3).

- Positioning
  - This work takes a bottom‑up, empirical approach: derive a general failure taxonomy from real execution traces (Grounded Theory in §3.1), validate it through inter‑annotator agreement (IAA) (§3.2), then scale annotation with an LLM‑based judge (§3.3) to build a large, public dataset (§3.4). The taxonomy is then used to analyze failure distributions across models, frameworks, and tasks (Figures 4, 8, 9) and to guide concrete system interventions (Appendix H, Table 5).

## 3. Technical Approach
At a high level: derive a taxonomy from data, validate it with humans, scale with an LLM annotator, and apply it across many MAS to analyze and improve systems. Below is the step‑by‑step pipeline (also depicted in Figure 2).

- Core concepts (defined for clarity)
  - `Agent`: an LLM‑powered component with a role prompt (initial state), a conversation history (state), and the ability to act (e.g., tool calls); see Introduction, p.1.
  - `MAS` (Multi‑Agent System): a set of agents interacting via an orchestration/workflow to solve tasks (p.1).
  - `Trace`: the full, ordered record of agent messages, tool calls, outputs, and orchestrator decisions for one task run.
  - `Failure mode`: a recurring pattern describing how/why a run fails its task objective.
  - `Grounded Theory`: a qualitative method that codes raw data to surface recurring phenomena without pre‑set labels; here: open coding, constant comparison, memoing, theorizing (§3.1).
  - `Inter‑Annotator Agreement (IAA)` and `Cohen’s κ`: a statistic (−1 to 1) that measures agreement beyond chance between annotators; κ≈0.8–1.0 is considered strong.

Step 1 — Derive a failure taxonomy from traces (Grounded Theory; §3.1, §4)
- Data: 150 traces from 5 MAS (HyperAgent, AppWorld, AG2, ChatDev, MetaGPT) spanning programming and math tasks; each trace averages >15,000 lines of text (§3.1).
- Method: six experts independently performed open coding; through constant comparison and theorizing, they converged on recurring failure patterns; analysis proceeded until “theoretical saturation,” i.e., no new failure patterns emerged with new traces (§3.1).
- Outcome: MAST (Multi‑Agent System Failure Taxonomy), 14 failure modes grouped into 3 categories and mapped to conversation stages (Figure 1; details in Appendix A).

Step 2 — Standardize definitions and validate human agreement (§3.2)
- Three annotators labeled 15 traces in three rounds, refining definitions each round through discussion to resolve disagreements; final IAA achieved κ = 0.88 (§3.2).
- A visual example shows how a snippet is labeled, e.g., “Information Withholding” when a Phone Agent fails to share required API username format, leading to repeated failed logins (Figure 3).

Step 3 — Build an LLM‑as‑a‑Judge annotator to scale labels (§3.3)
- Approach: prompt OpenAI `o1` with (a) a trace, (b) MAST definitions, and (c) few‑shot examples from the human‑labeled set to predict which failure modes occurred (Appendix N describes examples).
- Calibration: on held‑out human‑labeled traces, the few‑shot `o1` annotator achieves:
  > Accuracy 0.94, Recall 0.77, Precision 0.833, F1 0.80, Cohen’s κ 0.77 (Table 2).
- Generalization: on two unseen MAS (OpenManus, Magentic‑One) and two new benchmarks (MMLU, GAIA), human IAA with the finalized MAST reached κ = 0.79 (§3.4), indicating definitions transfer.

Step 4 — Construct the dataset (§3.4; Table 1)
- `MAST‑Data`: 1,642 annotated traces from 7 MAS frameworks across coding, math, and general tasks (ChatDev, MetaGPT, HyperAgent, AppWorld‑derived multi‑agent, AG2 MathChat, Magentic‑One, OpenManus).
- Tasks/Models: e.g., ProgramDev/ProgramDev‑v2 (coding), SWE‑Bench Lite (code maintenance), GSM‑Plus & OlympiadBench (math), MMLU (knowledge), GAIA (general agents) with models GPT‑4/4o, Claude 3.7 Sonnet, Qwen2.5‑Coder‑32B, and CodeLlama‑7B (Table 1).
- `MAST‑Data‑human`: a smaller subset with triple human annotations and rationales (used for IAA).
- Cost to annotate with LLM‑as‑Judge averages $1.8 per trace; per‑framework costs in Table 9.

Step 5 — Use MAST to analyze failures and guide improvements
- Failure categories and modes (Figure 1; Appendix A):
  - FC1 System Design Issues (44.2% of failures in Figure 1): task/role non‑adherence (FM‑1.1, FM‑1.2), step repetition (FM‑1.3), history loss (FM‑1.4), unaware of termination (FM‑1.5).
  - FC2 Inter‑Agent Misalignment (32.3%): conversation reset (FM‑2.1), failure to ask clarification (FM‑2.2), derailment (FM‑2.3), information withholding (FM‑2.4), ignoring others (FM‑2.5), reasoning‑action mismatch (FM‑2.6).
  - FC3 Task Verification (23.5%): premature termination (FM‑3.1), no/incomplete verification (FM‑3.2), incorrect verification (FM‑3.3).
- The taxonomy maps each mode to conversation stages (pre‑execution, execution, post‑execution) to indicate where detection/mitigation fits (Figure 1).

Step 6 — Release a developer tool
- A Python library `agentdash` exposes the annotator and taxonomy for practitioners (Appendix C shows usage).

Design choices and why they matter
- Grounded, data‑first taxonomy: avoids importing pre‑conceived categories; ensures modes reflect real system behavior.
- Stage mapping (Figure 1): helps engineering teams place checks where they can be most effective (e.g., pre‑execution role prompts vs. post‑execution verifiers).
- LLM‑as‑Judge: manual labeling at scale is infeasible; the calibrated annotator maintains high agreement with humans (Table 2) and keeps costs reasonable (Table 9).

## 4. Key Insights and Innovations
1) A unified, fine‑grained taxonomy for MAS failures (MAST)
- Novelty: First empirically derived taxonomy specific to multi‑agent LLM systems, with 14 distinct modes organized by system stage (Figure 1; Appendix A).
- Why it matters: Distinguishes visually similar symptoms with different roots (e.g., “missing information” can be FM‑2.4 withholding, FM‑2.5 ignoring, or FM‑1.4 history loss). This precision enables targeted fixes rather than generic “make prompts better.”

2) A large, publicly annotated dataset of real multi‑agent executions (MAST‑Data)
- Novelty: 1,642 traces from 7 frameworks and diverse tasks/models (Table 1).
- Why it matters: Enables comparative analysis across systems and benchmarks (Figures 4, 8, 9), correlation studies (Figures 6–7), and failure‑success relationships (Table 7). Prior work lacked such breadth and standardized labels.

3) Scalable annotation via an LLM‑as‑a‑Judge with strong human agreement
- Novelty: A practical pipeline using `o1` with MAST definitions and few‑shot examples achieves κ = 0.77 (Table 2) and generalizes to unseen systems/benchmarks with κ = 0.79 (§3.4).
- Why it matters: Makes large‑scale, fine‑grained failure analysis feasible for industry teams that cannot afford extensive human coding.

4) Three actionable system‑design insights grounded in data (§4)
- FC1: Failures are often architectural/specification issues, not just “LLM can’t follow instructions.” Example: in ChatDev, adjusting hierarchy so the CEO has final say raised success by +9.4% (§4, Appendix H).
- FC2: Communication protocol alone is insufficient; agents often lack “theory‑of‑mind”‑like modeling of others’ information needs (discussion near Figure 3). This calls for structural message content changes and/or model‑level training for social reasoning.
- FC3: Verification must be multi‑level and task‑aware. Superficial checks (e.g., code compiles) miss deeper errors; adding high‑level objective verification improved ChatDev by +15.6% on ProgramDev (Appendix H).

These are fundamental innovations (taxonomy, dataset, scalable annotator) plus concrete, validated design principles.

## 5. Experimental Analysis
Evaluation setup
- Datasets/benchmarks and systems: Table 1 lists the MAS–benchmark–LLM combinations that contribute to MAST‑Data, covering coding (ProgramDev, SWE‑Bench Lite), math (GSM‑Plus, OlympiadBench), general agent tasks (GAIA), and knowledge (MMLU).
- Metrics:
  - For annotation quality: Accuracy/F1 and Cohen’s κ vs. human experts (Table 2).
  - For system behavior: (a) task success rates (human‑evaluated for several configs in Table 1 and Appendix B), (b) failure counts per mode/category (Figures 1, 4, 8, 9; Tables 6–8), (c) correlations across modes/categories (Figures 6–7).
- Experimental design highlights:
  - Human IAA to validate taxonomy (κ = 0.88; §3.2).
  - Calibrated LLM annotator, including few‑shot prompting; accuracy 0.94 and κ = 0.77 (Table 2).
  - Generalization checks on new systems/benchmarks (§3.4).
  - Case‑study interventions (Appendix H; Table 5; Figures 10–11).

Main quantitative results
- High failure rates across MAS: for six systems run on their respective benchmarks, failures ranged from 41.0% to 86.7% (Figure 5). This motivates the need for failure‑aware design.
- Distinct failure profiles by system: Figure 4 (210 traces) shows, for example, AppWorld often ends too early (FM‑3.1 premature termination), OpenManus repeats steps (FM‑1.3), and HyperAgent shows both step repetition (FM‑1.3) and incorrect verification (FM‑3.3).
- Category distinctiveness: correlations among the three categories are low (0.17–0.32; Figure 6), suggesting they capture different aspects (design vs. coordination vs. verification). Some mode‑level correlations exist but remain moderate (max 0.63; Figure 7).
- Model choice matters (Figure 8):
  - Within MetaGPT on ProgramDev‑v2, GPT‑4o exhibits substantially fewer FC1/FC2 failures than Claude 3.7 Sonnet (the text notes a 39% reduction in FC1; §5.1/Figure 8), though FC3 remains high for both—verification remains a bottleneck.
- MAS architecture matters (Figure 9):
  - With GPT‑4o held fixed, MetaGPT shows far fewer FC1/FC2 failures than ChatDev but considerably more FC3 failures. Interpretation in §F: MetaGPT’s strong role SOPs reduce specification/coordination errors but ChatDev’s explicit testing/review phases catch more verification issues.
- Case studies show targeted gains (Appendix H; Table 5):
  - AG2 MathChat on GSM‑Plus: “Improved prompt (with verification)” increased accuracy from 84.75%±1.94 to 89.75%±1.44 with GPT‑4; with GPT‑4o, from 84.25%±1.86 to 89.00%±1.38. A topology redesign yielded smaller or model‑dependent gains.
  - ChatDev: On ProgramDev‑v0, success improved from 25.0% to 34.4% (prompt changes) and to 40.6% (topology changes). On HumanEval, the gains were modest (89.6%→91.5% with topology).
  - Failure‑mode reductions from these interventions are visualized in Figures 10–11, showing broad decreases, with topology generally outperforming prompt tweaks.
- Open‑source model comparison (Table 6):
  - Qwen2.5‑Coder‑32B shows substantially fewer failures than CodeLlama‑7B in both ChatDev and MetaGPT. Both lag GPT‑4o/Claude overall, indicating a remaining performance gap for open models in MAS settings.
- Failure–success relationships (Table 7):
  - Some modes are “fatal”: FM‑1.5 (Unaware of termination conditions) and FM‑2.4 (Information withholding) appear almost exclusively in failed runs.
  - Verification failures (FM‑3.2 No/Incomplete, FM‑3.3 Incorrect) also appear in successful runs, indicating superficial checks can pass with latent defects—a critical engineering warning.
- Benchmark difficulty shifts distributions (Table 8):
  - With AG2+GPT‑4o fixed, harder benchmarks (e.g., Olympiad) show higher overall failure rates and different mixes across categories than GSM.

Do the experiments support the claims?
- The taxonomy’s reliability is supported by strong human IAA (κ=0.88) and annotator agreement (κ=0.77) with generalization to new systems (κ=0.79) (Sections 3.2–3.4; Table 2).
- The analysis across systems/models/tasks (Figures 4, 8, 9; Tables 6–8) substantiates that (a) failures are common, (b) failure profiles depend on architecture and model, and (c) verification is a persistent bottleneck—directly supporting the three design insights in §4.
- Interventions yield measurable, mode‑specific improvements (Table 5; Figures 10–11), demonstrating practical value and that MAST can guide debugging.

Caveats the paper notes
- Cross‑system bar charts (e.g., Figure 4) are diagnostic rather than leaderboard comparisons because tasks differ (§5 caption).
- LLM‑as‑Judge remains an automated proxy for human annotations; however, agreement is high and costs are transparent (Table 9).

## 6. Limitations and Trade-offs
- Assumptions and scope
  - The taxonomy emphasizes failures amenable to system design and verification improvements. It does not attempt to catalog all underlying model limitations (e.g., factual hallucination) except where they manifest as MAS failures (§4 note).
  - Failure “stage” mapping (pre/execution/post) is a heuristic; some modes span stages (Figure 1).

- Annotation dependencies
  - The scalable annotator depends on a specific closed‑source model (`o1`) and few‑shot prompts (Table 2; Appendix N). Replicating κ may require similar models; domain shifts could change agreement.

- Comparability constraints
  - Many results are across different tasks/benchmarks per system (Figure 4), so not strict apples‑to‑apples comparisons. The paper uses them for profile discovery, not ranking.

- Generality and exhaustiveness
  - MAST is empirical and may not cover rare or domain‑specific failures (explicitly acknowledged in §4). New domains (robotics, safety‑critical control) could introduce additional modes.

- Ground truth for “root cause”
  - Even with fine‑grained labels, causal chains in MAS can be intertwined (e.g., a verifier miss might be precipitated by earlier misalignment). The taxonomy captures observed modes, not formal causal proofs.

- Compute/cost trade‑offs
  - LLM‑based annotation adds cost (avg $1.8/trace; Table 9) and latency. Stronger few‑shot calibrations can raise accuracy but also context length and price.

## 7. Implications and Future Directions
- How this changes the landscape
  - Provides a shared language (MAST) and public data (MAST‑Data) to move MAS development from ad‑hoc debugging to systematic diagnosis. Teams can now measure not only “how often” systems fail but “how” they fail, and test whether interventions reduce the right modes (Figures 10–11).

- Research enabled
  - Learning for coordination: Train agents to model other agents’ information needs (addressing FM‑2.2/2.4/2.5), possibly via supervised traces from MAST‑Data or multi‑agent RL.
  - Verification research: Develop multi‑level, domain‑aware verifiers that combine static analysis, test generation, external knowledge, and symbolic checks (addresses FC3; §4 and Appendix G).
  - Organizational design for agents: Explore workflows/topologies that reduce FC1 and FC2 (e.g., role hierarchies, turn‑taking protocols, “final authority” agents), supported by ablations like those in Appendix H.
  - Memory/state management: Reduce FM‑1.4 (history loss) with structured memory (e.g., workflow memory, OS‑style context management) and robust state machines (Appendix G).
  - Standardized communication protocols: Formal message schemas to reduce misalignment (Appendix G) and enable automated coherence checks.

- Practical applications
  - MAS engineering dashboards: Integrate `agentdash` to track failure modes over time, identify regressions, and target mitigations.
  - Model/architecture selection: Use failure profiles (Figures 8–9) to pick LLMs and workflows suited to a domain’s risk tolerance (e.g., choose architectures strong on FC3 for safety‑critical coding).
  - Guardrail design: Detect “fatal” modes early (Table 7 shows FM‑1.5, FM‑2.4 correlate with failures) and trigger fallbacks: ask for clarification, escalate to a verifier, or hand off to a human.

Key takeaway
- MAS reliability is not primarily a matter of “more agents” or “more samples.” The data show that design choices (roles, workflows, memory) and multi‑level verification dominate outcomes, and that different architectures and models fail in systematically different ways. MAST and MAST‑Data give the community the vocabulary, data, and tooling to turn those insights into durable engineering practices.
