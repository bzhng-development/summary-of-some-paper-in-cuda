# PaperBench: Evaluating AI’s Ability to Replicate AI Research

**ArXiv:** [2504.01848](https://arxiv.org/abs/2504.01848)
**Authors:** Giulio Starace, Oliver Jaffe, Dane Sherburn, James Aung, Jun Shern Chan, Leon Maksin, Rachel Dias, Evan Mays, Benjamin Kinsella, Wyatt Thompson, Johannes Heidecke, Amelia Glaese, Tejal Patwardhan
**Institutions:** 

## 🎯 Pitch

PaperBench introduces a pioneering benchmark that evaluates AI agents' ability to replicate state-of-the-art ML research papers from scratch, highlighting significant gaps with current models only achieving partial success. This work is crucial for advancing autonomous R&D capabilities, providing a structured, scalable metric for assessing AI's potential to accelerate scientific progress or pose regulatory challenges.

---

## 1. Executive Summary (2-3 sentences)
PaperBench is a benchmark that tests whether AI agents can replicate modern ML research papers from scratch—reading the paper, writing a fresh codebase, running the experiments, and matching results—using a rigorously defined, author-approved rubric and an automated LLM judge (Sections 2–4; Figure 1). Across 20 ICML 2024 Spotlight/Oral papers (8,316 fine-grained requirements), current frontier agents achieve only partial replication (best: 21.0%), revealing substantial gaps in long-horizon ML engineering and research capabilities (Table 4).

## 2. Context and Motivation
- Problem addressed
  - Replicating cutting-edge ML papers is hard: it requires understanding nuanced methods, building complex codebases, executing experiments reliably, and verifying results. PaperBench targets whether autonomous agents can do all of this from scratch, not just run or edit existing research code (Section 2.1, 2.5).
- Why it matters
  - Practical impact: If agents could faithfully replicate ML research, they could accelerate science, improve reproducibility, and reduce engineering bottlenecks. Conversely, reliably measuring this capability is critical for safety and governance, since autonomous R&D could rapidly amplify capabilities (Abstract; Impact Statement).
  - Evaluation need: Human grading takes tens of hours per paper; scalable, trustworthy evaluation infrastructure is a prerequisite to track progress and risks (Section 4; Abstract).
- Prior approaches and gaps
  - Reproduction using existing repos: CORE-Bench evaluates on reproducing results given authors’ codebases; this measures different skills than building from scratch (Section 6, Related Work).
  - Kaggle-style tasks (MLE-bench, MLAgentBench, DSBench) emphasize standard ML problems that are simpler and less representative of frontier research workflows (Section 6).
  - RE-Bench proposes open-ended research engineering tasks but with narrower scope or built-in scoring functions that don’t capture the full breadth of ML paper replication (Section 6).
- Paper’s position
  - PaperBench focuses on end-to-end, from-scratch replication of SOTA research, with detailed, author-vetted rubrics; a strict “no authors’ code” rule enforced via blacklists and monitoring; an independent reproduction step; and an LLM judge evaluated on a separate human-labeled benchmark (Sections 2.1–2.5, 3.1, 4.2; Appendix E).

## 3. Technical Approach
PaperBench is a pipeline with four stages: task setup, agent attempt, independent reproduction, and rubric-based grading by an LLM judge.

1) Task and inputs (Section 2.1; Figure 1)
- Each sample provides:
  - The full paper (PDF + Markdown) and an addendum with clarifications (Section 3.2).
  - The agent must create a fresh repository that reproduces the paper’s empirical results via a top-level script `reproduce.sh`.
- Critical constraints:
  - The agent is not shown the rubric during its attempt (to avoid overfitting; Section 2.1).
  - The agent may browse the web but cannot use blacklisted sources (including the authors’ repos and online replications); violations are disqualified (Section 2.5; Appendix E).
  - No restrictions on runtime/compute during the attempt are mandated by the benchmark, but the paper’s experiments use a 12-hour agent time limit and standard hardware (Section 5.2).

2) Independent reproduction step (Section 2.2)
- After the agent finishes, its repo is copied to a fresh VM (Ubuntu 24.04, A10 GPU).
- The grader runs `reproduce.sh` end-to-end to:
  - Generate outputs (results, tables, plots, artifacts).
  - Produce an execution log `reproduce.log`.
- Rationale: This separates generation from evaluation, prevents hard-coded results, and ensures outputs are reproducible from a clean start.
  - Quote: “We execute the submission’s reproduction script… and also produces a reproduce.log file as a side-effect” (Section 2.2).

3) Rubrics and scoring (Sections 2.3–2.4; Figure 2; Table 1; Section 3.1)
- Rubric structure:
  - Hierarchical tree where each leaf is a binary requirement (“pass” or “fail”); parents aggregate children by weighted average; the root score is the final Replication Score (Figure 2).
  - Weights reflect importance (not implementation difficulty), emphasizing core contributions (Section 3.1).
  - Scope: 20 papers, 8,316 leaf nodes in total; per-paper totals in Table 2; type breakdown in Table 7.
- Three leaf types (Table 1; Section 2.4):
  - `Code Development`: Is the implementation present and plausibly correct? Judge inspects source and docs.
  - `Execution`: Did running `reproduce.sh` execute the relevant parts successfully? Judge examines code + `reproduce.log`.
  - `Result Match`: Do generated outputs match the paper’s reported trends or results? Judge inspects `reproduce.log` and generated outputs.
  - If `reproduce.sh` is missing, all `Execution` and `Result Match` leaves score 0 (Section 2.4).
- Granularity and design:
  - Leaves are decomposed until an expert could grade each in <15 minutes (Section 3.1), enabling partial credit aligned to importance.

4) LLM judge and JudgeEval (Section 4; Table 3; Appendix D, G)
- Judge scaffolding (“SimpleJudge”):
  - Inputs per leaf: paper Markdown, rubric JSON, the specific leaf requirement, and a filtered subset of the submission. Because repos can be large, the judge first ranks files by relevance and loads only the “top-K” (Appendix D; Figure 7 shows the file-ranking prompt).
  - Different inputs shown depending on leaf type (Table 1).
  - Outputs: a binary score with explanation; scores then propagate up the rubric tree (Figure 9).
  - Backend: `o3-mini-2025-01-31` with high reasoning, selected for best cost–performance trade-off (Table 3).
- JudgeEval: a mini-benchmark of human-graded submissions used to evaluate automated judges. Models are compared by accuracy, precision, recall, F1, and cost (Section 4.2).
  - Result: `o3-mini` reaches macro F1 0.83 at about $66 per paper; `o1` is similar (0.84) but more expensive (~$830/paper) (Table 3; Fig. 5). Stratified F1 shows strongest reliability on `Result Match` (0.94 for `o3-mini`) and weakest on `Code Development` (0.72) (Table 8).

5) Agents and scaffolding (Section 5.1; Appendix F)
- Execution environment: Docker on Ubuntu 24.04 with an A10 GPU, internet access, tools for shell, Python, web browsing, and paginated file reading (Section 5.1).
- Baseline agent (“BasicAgent”): a ReAct-style loop—plan, call tools, observe, repeat—until time expires or the model explicitly ends (Appendix F.1; Figure 10).
  - Observed failure: many models stop early or produce plans instead of doing tool-based work (Appendix F.1; Section 5.2).
- Iterative scaffolding (“IterativeAgent”): removes the ability to end early and prompts the model to take one small, concrete next step at a time (Appendix F.2; Figure 11–12). This substantially changes outcomes (Table 5).

6) “PaperBench Code-Dev” variant (Section 2.6)
- A lighter version that skips the reproduction step and grades only `Code Development` leaves (no GPU required; ~85% cheaper to grade).
- Correlates weakly with the full benchmark but useful as a fast proxy (Section 2.6; footnote 5).

Analogy for clarity: Think of the rubric as a graded “project spec” with required milestones. The agent builds a full project (repo + `reproduce.sh`). A TA (the LLM judge), having access to the spec, the paper, and your repo, checks code, whether your script runs the right parts, and whether the generated results match the paper’s claims.

## 4. Key Insights and Innovations
- Author-approved, fine-grained rubrics for frontier ML (Section 3.1; Table 2; Table 7)
  - Novelty: The rubric trees translate vague replication goals into hundreds of concrete, weighted, binary checks co-developed with original paper authors, enabling precise partial credit and comparability. This is a qualitative leap over coarse, pass/fail reproduction or single-metric judging.
- Independent reproduction before grading (Section 2.2)
  - Significance: Running `reproduce.sh` in a clean environment and grading only the fresh outputs makes it substantially harder to fake success (e.g., by hard-coding results), improving credibility relative to prior in-place evaluations.
- An LLM judge validated on a separate human-graded benchmark (Section 4.2; Table 3; Table 8)
  - Novelty: Evaluating the evaluator (“JudgeEval”) quantifies judge reliability and cost, allowing the community to iterate on judges and track progress. The chosen judge (`o3-mini`) offers near-`o1` accuracy at ~10× lower cost (Table 3; Figure 5).
- From-scratch constraint with blacklist enforcement (Sections 2.1, 2.5; Appendix E)
  - Significance: This tests real engineering and research capability rather than code reuse skills. Automatic monitoring plus manual review guard against contamination.
- Evidence about agent weaknesses on long-horizon work and the role of scaffolding (Section 5.2–5.3; Figure 3; Table 4–5)
  - Insight: Prompting and control loops matter. Forcing agents to work iteratively and to not end early boosts performance substantially for some models (e.g., `o1`; Table 5), and time-series analysis shows early plateaus without better scaffolding and strategy (Figure 3).

## 5. Experimental Analysis
- Evaluation setup (Sections 5.1–5.2)
  - 20 ICML 2024 Spotlight/Oral papers spanning 12 topics (Table 2).
  - Hardware: single A10 GPU; Ubuntu 24.04; internet access; API keys for relevant services (Section 5.1).
  - Each model–paper pair: 3 independent runs; 12-hour time limit per run (Section 5.2).
  - Blacklist monitoring flagged 10 violations across 646 runs; these were set to 0 (Section 2.5).
- Models and scaffolds (Section 5.2; Table 4–6)
  - Models: `GPT-4o`, `o1`, `o3-mini`, `DeepSeek-R1`, `Claude 3.5 Sonnet (New)`, `Gemini 2.0 Flash`.
  - Scaffolds: BasicAgent (main); IterativeAgent (forces continued work; Section 5.3).
- Main results with BasicAgent (Table 4)
  > Table 4 (mean Replication Score over 20 papers):  
  > `Claude 3.5 Sonnet`: 21.0% ± 0.8  
  > `o1-high`: 13.2% ± 0.3  
  > `DeepSeek-R1`: 6.0% ± 0.3  
  > `GPT-4o`: 4.1% ± 0.1  
  > `Gemini 2.0 Flash`: 3.2% ± 0.2  
  > `o3-mini-high`: 2.6% ± 0.2
  - Interpretation: Even strong models achieve only partial replication; most struggle to sustain multi-hour engineering workflows (Section 5.2).
- Effect of IterativeAgent (Table 5)
  > Table 5: With IterativeAgent,  
  > `o1-high` rises to 24.4% ± 0.7 (36-hour run: 26.0% ± 0.3),  
  > `o3-mini-high` rises to 8.5% ± 0.8,  
  > `Claude 3.5 Sonnet` drops to 16.1% ± 0.1.
  - Interpretation: Preventing early termination and enforcing stepwise progress helps some models substantially (notably `o1`), but the same prompting can hinder others (Claude), highlighting model–scaffold interactions (Section 5.3).
- Code-Dev variant (Table 6)
  > Table 6: On PaperBench Code-Dev (only implementation graded), `o1-high` achieves 43.4% ± 0.8.
  - Interpretation: Models can write a fair amount of plausible code, but turning that code into successful executions and matching results is the main bottleneck (Table 9 below).
- Requirement-type breakdown (Appendix I.1; Table 9)
  > Table 9 (BasicAgent vs IterativeAgent; averages across 20 papers):  
  > `o1 (BasicAgent)`: Code Dev 19.5% ± 1.2; Execution 5.7% ± 0.9; Result Match 0.0%.  
  > `o1 (IterativeAgent)`: Code Dev 43.3% ± 1.1; Execution 4.5% ± 1.5; Result Match 0.0% (36 hours: 1.4% ± 0.1).  
  > `Claude 3.5 Sonnet (BasicAgent)`: Code Dev 35.4% ± 0.8; Execution 1.8% ± 0.7; Result Match 0.7% ± 0.3.
  - Interpretation: The stark drop from Code Dev to Execution and Result Match shows where agents falter—integrating, running, and obtaining correct outcomes.
- Human baseline (Section 5.4; Figure 3)
  > Figure 3 (4-paper subset, time-series): o1 initially leads but plateaus after ~1 hour; after 24 hours, human attempts surpass and continue improving. The paper also reports a best-of-3 human baseline of 41.4% after 48 hours vs. o1’s 26.6% on a 3-paper subset (Abstract; Section 5.4).
  - Interpretation: Humans start slowly (time spent understanding), then outpace the agent at longer horizons, underscoring deficiencies in agent planning, debugging, and persistence.
- Judge reliability and cost (Section 4.2; Table 3; Appendix G)
  > Table 3 (JudgeEval): `o3-mini` macro F1 0.83 at ~$66/paper; `o1` macro F1 0.84 at ~$830/paper; `GPT-4o` macro F1 0.73 at ~$120/paper. Stratified F1 for `o3-mini`: Code Dev 0.72; Execution 0.82; Result Match 0.94 (Table 8).
  - Interpretation: The chosen judge is a reasonable, cost-effective proxy for expert grading, with highest confidence on result verification.
- Exploratory cost reduction: rubric pruning (Appendix H; Figure 6)
  > Figure 6: For one JudgeEval submission, pruning rubric depth to 3 reduced cost ~10× with only slight score drift relative to the unpruned case; human grade shown for reference.
  - Interpretation: Coarser-grained judging may further reduce cost as judges improve, though this remains experimental.

Do the experiments support the claims?
- Yes on measurement and diagnosis: The multi-model, multi-run results across 20 papers, requirement-type breakdowns, and human comparisons convincingly show that (a) agents can implement parts of papers, (b) they largely fail to execute and match results, and (c) scaffolding and time control meaningfully affect performance (Sections 5.2–5.4; Tables 4–6; Table 9; Figure 3).
- The automated judge’s reliability is quantified and appears strong enough for aggregate benchmarking (Table 3; Table 8), though not yet at expert parity in all cases.

## 6. Limitations and Trade-offs
- Dataset scope and recency (Section 7)
  - 20 papers is sizable in depth (8,316 leaves) but small in breadth relative to all ML research; broader coverage would improve generality.
- Potential pretraining contamination (Section 7)
  - Future models may have memorized parts of recent papers or code; while the blacklist and from-scratch rule reduce direct leakage, pretraining effects could inflate performance.
- Rubric creation cost and difficulty (Section 7; Appendix C)
  - Rubrics are labor-intensive to design and validate with authors; this constrains scaling and reproducibility of the benchmark construction process.
- Judge accuracy and determinism (Section 7; Section 4.2; Appendix G)
  - The LLM judge, while strong on JudgeEval (F1 0.83), is not perfect or deterministic; edge cases and adversarial submissions remain open risks.
- Cost and compute (Section 7)
  - Full PaperBench runs are expensive (e.g., ~$8k for a 20-paper o1 IterativeAgent pass at 12 hours; judge adds ~$66/paper). PaperBench Code-Dev helps but measures only implementation quality.
- Assumptions embedded in task design (Sections 2.1–2.5; Appendix B)
  - Requires `reproduce.sh` and a single-machine setup; excludes papers needing multi-node training or closed-source model dependencies; addenda reduce but cannot eliminate underspecification.

## 7. Implications and Future Directions
- How this changes the field
  - PaperBench establishes a rigorous, scalable, and transparent way to measure autonomous ML R&D capability. By decomposing “replicate a paper” into verified, weighted sub-outcomes and validating an LLM judge, it offers a yardstick for long-horizon AI competence and an instrument for safety-preparedness tracking (Abstract; Sections 2–4).
- What research it unlocks
  - Agent scaffolding and control: Results show large gains from preventing early termination and enforcing stepwise progress (Table 5). This invites work on planning, self-debugging, recovery from failures, and better tool-use loops.
  - Judging and oversight: JudgeEval provides a platform to improve automated judges, including agent-as-a-judge methods, and to study adversarial behavior or specification gaming (Appendix A.3; Section 4.2).
  - Rubric generation: Automating rubric drafting/critique could reduce construction cost; dependency-aware graphs and subtree scoring could make rubrics both clearer and cheaper to grade (Appendix A.1, H).
- Practical applications
  - Capability tracking for labs and safety teams: Use the benchmark to audit the real-world ML engineering competence of agents, including readiness to run complex experiments safely.
  - Training and evaluation loops: PaperBench Code-Dev can serve as a fast proxy to iterate on agents; the full benchmark can be run periodically to measure progress and detect regressions.
  - Tooling and infra: The reproduce-first, grade-later pattern and the file-ranking judge interface are reusable components for other complex evaluation settings.

In sum, PaperBench translates “can an AI replicate frontier ML research from scratch?” into a measurable, reproducible evaluation. The current answer—partial success on implementation but not on end-to-end execution and result fidelity—highlights where research should focus next: sustained long-horizon reasoning, robust tool use, and reliable self-correction.
