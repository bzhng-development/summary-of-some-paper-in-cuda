# A Survey on Evaluation of Large Language Models

**ArXiv:** [2307.03109](https://arxiv.org/abs/2307.03109)
**Authors:** Yupeng Chang, Xu Wang, Jindong Wang, Yuan Wu, Kaijie Zhu, Hao Chen, Linyi Yang, Xiaoyuan Yi, Cunxiang Wang, Yidong Wang, Wei Ye, Yue Zhang, Yi Chang, Philip S. Yu, Qiang Yang, Xing Xie
**Institutions:** Jilin University, Microsoft Research, Institute of Automation, CAS, Carnegie Mellon University, Westlake University, Peking University, University of Illinois, Hong Kong University of Science and Technology

## 🎯 Pitch

This paper introduces a groundbreaking framework for evaluating large language models (LLMs) by organizing assessments into three dimensions: tasks and capabilities, datasets and benchmarks, and evaluation methods. By framing evaluation as a standalone discipline, it provides a comprehensive blueprint for ensuring LLMs are safe, capable, and trustworthy, significantly impacting applications in high-stakes environments like medicine and finance.

---

## 1. Executive Summary (2-3 sentences)
This survey systematizes how to evaluate large language models (LLMs) by organizing the space into three dimensions: what to evaluate (tasks and capabilities), where to evaluate (datasets and benchmarks), and how to evaluate (automatic and human methods). It compiles successes, failure modes, and 46+ benchmarks, and proposes “grand challenges” for future evaluation, arguing that evaluation must be treated as its own discipline to ensure safe, capable, and trustworthy LLMs (Fig. 1; Sec. 1, 3–7).

## 2. Context and Motivation
- Problem addressed
  - LLMs are used in sensitive settings (e.g., education, medicine, finance) and exhibit emergent abilities and new risks; existing evaluation recipes designed for earlier NLP or vision systems are no longer sufficient (Sec. 1–2). Static test suites (e.g., GLUE, SuperGLUE) do not capture evolving capabilities, adversarial vulnerabilities, or societal impacts (Sec. 2.2; Fig. 3).
- Why it matters
  - Real-world impact: Miscalibration, hallucinations, social biases, and adversarial brittleness can harm users when LLMs are deployed in search, medical advice, law, or autonomous agents (Sec. 3.1.5, 3.2, 3.5, 3.6).
  - Theoretical significance: Stronger models with poorer interpretability (Table 1) demand principled, multi-dimensional measurements (Sec. 2.2).
- Prior approaches and gaps
  - Traditional ML evaluation relies on fixed train/test splits and task-specific metrics (Sec. 2.2). For LLMs, these miss:
    - Robustness to adversarial prompts and out-of-distribution (OOD) inputs (Sec. 3.2.1, PromptBench [264]; GLUE-X [234]).
    - Safety, bias, and value alignment at a societal level (Sec. 3.2.2; Table 3).
    - Open-ended generation quality where reference-based metrics fail (Sec. 5.2).
- Positioning
  - The survey integrates the fragmented literature into a cohesive 3D framework (what/where/how), catalogs benchmarks (Table 7), codifies evaluation criteria (Tables 8–10), and distills cross-task success and failure cases (Sec. 6.1), culminating in a research agenda of seven “grand challenges” (Sec. 7).

## 3. Technical Approach
The survey’s “methodology” is a structured framework rather than an algorithm: it maps evaluation to three interacting axes and then deep-dives each.

1) What to evaluate (Sec. 3; Fig. 1)
- Capabilities and application domains are grouped into:
  - Natural language processing (NLP) tasks: understanding (sentiment, NLI), reasoning (mathematical, logical, commonsense), generation (summarization, dialogue, translation, QA), multilingual, and factuality (Sec. 3.1; Table 2).
  - Robustness/ethics/bias/trustworthiness (Sec. 3.2; Table 3).
  - Social science, natural science and engineering, medicine, agents, and other applications (Sec. 3.3–3.7; Tables 4–6).
- Mechanism: Within each area, the survey summarizes how researchers probe a capability, typical prompts or setups, and the observed strengths/weaknesses.

2) Where to evaluate (Sec. 4)
- Benchmarks are categorized into:
  - General-task benchmarks (e.g., HELM, MMLU, BIG-bench, MT-Bench, Chatbot Arena) that span many tasks or simulate real-world dialogue (Table 7; Sec. 4.1).
  - Specific downstream tasks (e.g., MATH, APPS for coding, C-Eval/CMMLU for Chinese knowledge, TRUSTGPT for ethics, API-Bank and ToolBench for tool use) (Sec. 4.2; Table 7).
  - Multi-modal benchmarks (e.g., MME, MMBench, SEED-Bench, MM-Vet, LAMM, LVLM-eHub) for vision-language and video understanding (Sec. 4.3; Table 7).
- Design choice: Prefer breadth plus depth—use general suites for coverage and specialized suites for targeted diagnosis (Table 7 shows focus, domain, and criteria per benchmark).

3) How to evaluate (Sec. 5)
- Automatic evaluation (Sec. 5.1)
  - Metrics are grouped into four families (Table 9) and defined with mechanisms:
    - Accuracy: exact match, F1, ROUGE—compute string or token overlaps for classification or generation (Table 9).
    - Calibration: how well model confidence matches correctness; e.g., `Expected Calibration Error (ECE)` bins predictions by confidence and compares bin-wise accuracy (Sec. 5.1).
    - Fairness: parity across demographic groups; e.g., `Demographic Parity Difference (DPD)` measures gaps in positive prediction rates between groups; `Equalized Odds Difference (EOD)` measures error-rate gaps conditional on the true label (Sec. 5.1).
    - Robustness: adversarial success (`Attack Success Rate`) and relative performance degradation (`Performance Drop Rate`) after prompt attacks (Sec. 5.1; PromptBench’s PDR formula).
  - Evaluator models: Automated “LLM-as-judge” approaches (e.g., PandaLM, MT-Bench) train or prompt a judge model to compare outputs, increasing reproducibility (Sec. 5.1).
- Human evaluation (Sec. 5.2)
  - When references or metrics are insufficient (open-ended dialogue, long-form reasoning), the survey prescribes human-in-the-loop judgments using rubrics aligned with the extended “3H” rule (Sec. 5.2; Table 10):
    - Accuracy, relevance, fluency, transparency (explainability of the model’s reasoning), safety (avoidance of harmful content), and human alignment (value-consistency).
  - Implementation details: ensure adequate raters and expertise; account for variance (culture- and rater-dependent variability is a known issue, Sec. 5.2).

4) Supporting processes
- The evaluation pipeline for AI models (Fig. 3) clarifies where “what/where/how” sit in relation to models and data. Table 1 contrasts traditional ML, deep learning, and LLMs to motivate different evaluation needs (very large data, high hardware demands, poor interpretability).

Definitions used where uncommon:
- `RLHF` (Reinforcement Learning from Human Feedback): fine-tuning a model to prefer human-rated outputs by training a reward model and optimizing via reinforcement learning (Sec. 2.1).
- `OOD` (Out-of-Distribution): inputs that differ from the training distribution; stress-tests generalization (Sec. 3.2.1).
- `Adversarial prompt`: an input deliberately crafted to trigger failure or policy jailbreak (Sec. 3.2.1; PromptBench).
- `Calibration`: agreement between predicted confidence and actual correctness (Sec. 5.1).
- `Elo rating` (Chatbot Arena): a pairwise comparison system from chess that yields a single ranking score based on wins/losses (Sec. 4.1).

## 4. Key Insights and Innovations
- A unifying 3D evaluation framework (What/Where/How)
  - Novelty: Shifts from task-by-task checklists to a lifecycle view that spans capability selection, benchmark choice, and evaluation protocol. This helps practitioners assemble end-to-end, fit-for-purpose evaluations (Fig. 1; Sec. 3–5).
  - Significance: Prevents over-reliance on single metrics or static test sets, a known failure mode for LLMs (Sec. 1, 2.2).
- Consolidated benchmark atlas with scope labels
  - Content: 46 benchmarks are cataloged with their focus, domain, and criteria (Table 7), including general, specific, and multimodal suites (Sec. 4).
  - Value: Makes it straightforward to select the right benchmark for capabilities like advanced reasoning (ARB), safety (SafetyBench; TRUSTGPT), or tool use (API-Bank; ToolBench).
- Codified evaluation criteria across accuracy, calibration, fairness, and robustness
  - Mechanistic detail: Provides equations and operational definitions (Table 9), not just metric names. This facilitates reproducible setups and cross-paper comparisons (Sec. 5.1).
- Cross-domain synthesis of success and failure
  - The survey distills consistent patterns across many studies into actionable summaries (Sec. 6.1), e.g., strong arithmetic/logical reasoning trends vs. weaknesses in NLI, abstract reasoning, non-Latin multilingual tasks, adversarial prompts, and hallucination.
- Research agenda (“Grand challenges”)
  - Seven concrete directions—from AGI benchmarking and behavioral tests to dynamic/evolving and trustworthy evaluation systems (Sec. 7). This goes beyond a literature summary to chart a path for the field.

## 5. Experimental Analysis
Because this is a survey, the “experiments” are the compiled findings across domains. The survey grounds these with figures and tables and reports representative numbers.

- Evaluation methodology (what/where/how)
  - Capability coverage: NLP, reasoning, generation, multilingual, factuality; robustness/ethics; social/natural sciences; medicine; agents; education, search/recs, personality (Sec. 3; Tables 2–6).
  - Benchmarks used: General (HELM, MMLU, BIG-bench, Chatbot Arena, MT-Bench, AlpacaEval), domain-specific (e.g., MATH, APPS, C-Eval/CMMLU, M3Exam, SafetyBench, TRUSTGPT), multi-modal (MME, MMBench, SEED-Bench, MM-Vet) (Sec. 4; Table 7).
  - Metrics: Accuracy, EM/F1/ROUGE, calibration (ECE, AUC), fairness (DPD, EOD), robustness (ASR, PDR), plus human criteria (Table 9–10).

- Representative quantitative results the survey compiles
  - Reasoning and math
    - GPT-4 outperforms earlier models on math; gains include “~+10 percentage points accuracy and ~50% reduction in relative error on arithmetic tasks,” driven by better division/trigonometry and consistent step-by-step calculations (Sec. 3.4.1; [221]).
    - On difficult high-school competition problems, GPT-4 reaches 60% on half the categories, but only ~20% on intermediate algebra/precalculus categories (Sec. 3.4.1; [225]).
    - Abstract reasoning remains limited: “existing LLMs have very limited ability” (Sec. 3.1.2; [56]).
  - Factuality and open QA
    > “GPT-4 and BingChat answer more than 80% of open questions correctly on Natural Questions and TriviaQA; there remains a >15% gap to 100%” (Sec. 3.1.5; [204]).
    - Fact evaluation methods favor NLI- and QG-based approaches (Sec. 3.1.5; [74]); FActScore decomposes outputs into “atomic facts” for precision (Sec. 3.1.5; [138]).
  - Translation and multilingual
    - ChatGPT/GPT-4 match or exceed commercial MT in human evaluation and sacreBLEU for X→English, but struggle for English→X and non-Latin scripts (Sec. 3.1.3–3.1.4; [208], [6], [2], [100]).
  - Dialogue
    - Crowd-sourced evaluation: Chatbot Arena uses pairwise voting with Elo to rank models in realistic chat (Sec. 4.1). MT-Bench uses multi-turn dialogue questions with LLM-judge evaluation (Sec. 4.1).
  - Safety, robustness, and bias
    - PromptBench shows notable prompt-level adversarial vulnerability; PDR quantifies relative performance drops (Sec. 3.2.1, 5.1; [264]).
    - DecodingTrust finds GPT-4 generally more robust than GPT-3.5 in standard settings yet “more susceptible to certain attacks,” and analyzes toxicity, stereotype bias, adversarial/OOD robustness, privacy, and fairness (Sec. 3.2.3; [201]).
  - Medicine
    - On the USMLE exams, performance “achieves or approaches the passing threshold” without tailored training (Sec. 3.5.2; [97]); across question sets, performance varies (Sec. 3.5.2; [57]).
    - Surgical knowledge: GPT-3.5 ~46.8% vs GPT-4 ~76.4% (Sec. 3.5.3; [143]).
    - Plain-language radiology report translation is feasible and improves with GPT-4 (Sec. 3.5.3; [131]).
  - Education and testing
    - On a high-school comprehension exam, ChatGPT averages ~71.8% (near student mean); GPT-4 reaches 8.33/10 with bootstrapping aiding error diagnosis (Sec. 3.7.1; [32]).
  - Social science
    - For computational social science tasks, LLMs are strong on misinformation/stance/emotion classification but below 40% on event argument extraction and implicit hate (Sec. 3.3; [269]).
- Ablations/failure analyses and robustness checks
  - Success/failure synthesis (Sec. 6.1) highlights consistent weaknesses: NLI and human disagreement (Sec. 3.1.1; [105]), abstract reasoning and multi-hop reasoning (Sec. 3.1.2; [56], [148]), non-Latin multilingual tasks and low-resource languages (Sec. 3.1.4; [2], [6], [100], [250]), adversarial prompt sensitivity (Sec. 3.2.1; [264]), hallucinations (Sec. 3.2.3; [163], [253]), and vulnerability to visual adversarial transfer (Sec. 3.2.1; [258]).
- Do the results support the survey’s claims?
  - Yes. The survey anchors claims in tables that aggregate literature (Tables 2–6 for domains; Table 7 for benchmarks) and in metric definitions and protocols (Tables 8–10), and it consistently ties numeric findings to cited studies within each subsection (Sec. 3–5).
- Mixed/conditional results and trade-offs
  - Zero-shot generality vs. task-specialized fine-tuning: instruction-tuned/fine-tuned models often outperform general LLMs on targeted tasks (Sec. 3.1.3, QA, dialogue; Table 2).
  - Improved safety alignment can reduce some risks but does not eliminate jailbreak susceptibility (Sec. 3.2.3; [201]).

## 6. Limitations and Trade-offs
- Assumptions and scope
  - The synthesis reflects the state through 2023; online models like GPT-4 evolve, so specific numbers may change (Disclaimer).
  - Findings are often dataset-specific; Section 6.1 cautions that success/failure cases depend on chosen benchmarks and data.
- Coverage constraints
  - While Table 7 lists 46 benchmarks, the authors note the space is fast-moving, so some may be missing or superseded (Sec. 4 note; Disclaimer).
- Evaluation variance
  - Human evaluations can be high-variance and culturally sensitive (Sec. 5.2), which complicates cross-study comparison.
- Metric fidelity
  - Reference-overlap metrics (ROUGE, EM) can misjudge quality for open-ended generation; LLM-as-judge methods raise their own bias concerns (Sec. 5.1–5.2).
- Computational and data requirements
  - Multi-metric, multi-benchmark evaluation is compute- and cost-intensive—especially for large multi-modal suites (Table 1 on hardware; Sec. 4.3).
- Open problems
  - Designing truly OOD, dynamic, and contamination-resistant evaluations; ensuring the trustworthiness of the evaluation system itself (Sec. 7.4–7.5).

## 7. Implications and Future Directions
- How this work changes the landscape
  - It reframes evaluation as a first-class discipline, giving practitioners a blueprint to construct holistic assessments rather than rely on any single leaderboard. The three-axis framework plus the metric catalog (Tables 9–10) provide a playbook for building rigorous test plans.
- Follow-up research enabled or suggested (Sec. 7)
  - Designing AGI benchmarks that reflect integrated, cross-domain intelligence beyond task proxies (Sec. 7.1).
  - Behavioral evaluations in open environments (e.g., LLM-controlled robots, multi-modal agents) to complement standard suites (Sec. 7.2).
  - Robustness expansions: broader adversarial spaces, more realistic OOD shifts, evolving definitions as threats change (Sec. 7.3).
  - Dynamic and evolving testbeds to avoid memorization and to keep pace with model updates (Sec. 7.4).
  - Trustworthy evaluation systems: verifiable generation of OOD examples, auditability of LLM-as-judge, statistical reliability (Sec. 7.5).
  - Unified evaluators that support alignment, safety, verification, and fine-tuning workflows (e.g., PandaLM-like judges for more tasks; Sec. 7.6).
  - From evaluation to enhancement: use diagnostic signals (e.g., PromptBench attention analyses) to guide prompt design, data curation, and model training (Sec. 7.7).
- Practical applications
  - Responsible deployment pipelines for enterprises: select domain-appropriate benchmarks from Table 7; adopt multi-metric automatic checks (Table 9) plus human evaluation rubrics (Table 10); add adversarial prompt and OOD tests before release.
  - Tool-using agents: validate API planning and execution with API-Bank/ToolBench before integrating into production (Sec. 4.2).
  - Multimodal products (e.g., vision-language apps): require MMBench/MME/MM-Vet coverage for perception, cognition, and integrated reasoning (Sec. 4.3).

In short, this survey provides both a map and a method: a way to pick the right capabilities to test, the right places to test them, and the right measures to trust—along with a candid accounting of where LLMs work well and where they do not, and a roadmap to build the next generation of evaluations.
