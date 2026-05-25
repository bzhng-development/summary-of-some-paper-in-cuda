# A Survey on Evaluation of Large Language Models

**ArXiv:** [2307.03109](https://arxiv.org/abs/2307.03109)

## 🎯 Pitch

This comprehensive survey introduces a unified 'what–where–how' framework for evaluating large language models, systematizing task coverage, benchmark selection, and evaluation methodologies across 40+ datasets and domains. By mapping current successes and failures, formalizing metrics such as robustness and fairness, and articulating future grand challenges, it elevates evaluation itself as a foundational discipline—critical for ensuring safe, reliable, and verifiably intelligent LLM deployment in real-world, high-stakes applications.

---

## 1. Executive Summary
This survey systematizes how to evaluate large language models (LLMs) by introducing a three-part framework: what to evaluate (tasks and capabilities), where to evaluate (benchmarks and datasets), and how to evaluate (automatic and human methods). It compiles success/failure cases across domains, catalogs 40+ benchmarks, formalizes evaluation metrics (including robustness and fairness), and outlines seven grand challenges to treat evaluation itself as a first-class discipline (Fig. 1; Secs. 3–7).

## 2. Context and Motivation
- Problem/gap addressed
  - LLMs are rapidly deployed in high-stakes settings (e.g., healthcare, finance) but existing evaluations are fragmented, static, and often narrow (Sec. 1). The field lacks a unified view that simultaneously covers task coverage (“what”), benchmark selection (“where”), and method/metrics (“how”), and it lacks principled ways to assess emerging risks (robustness, bias, hallucination).
- Why it matters
  - Real-world impact: Unsafe or brittle systems can cause harm in medicine, law, finance (Sec. 1; 3.5).
  - Theoretical significance: Claims of “AGI sparks” hinge on human-crafted evaluations that are hard to reproduce; rigorous, multi-faceted evaluation is needed to separate genuine generalization from prompt/benchmark artifacts (Sec. 1).
- Prior approaches and limitations
  - Static leaderboards (e.g., GLUE/SuperGLUE in NLP; ImageNet in vision) helped earlier paradigms but can be memorized by LLMs, miss evolving capabilities, and under-represent safety/robustness (Sec. 2.2).
  - Single-task evaluations (e.g., QA only) ignore cross-task trade-offs and societal risks (Sec. 3).
  - Human-crafted demos for “intelligence” lack coverage, repeatability, and principled scoring (Sec. 1).
- Positioning
  - The paper organizes the field around a comprehensive schema—“What–Where–How”—then synthesizes evidence on strengths/weaknesses (Sec. 6) and articulates concrete research directions (Sec. 7). It also curates benchmarks (Table 7) and evaluation metrics (Table 9) and highlights the rise of community evaluations (Chatbot Arena; MT-Bench) (Sec. 4.1).

## 3. Technical Approach
The survey is a structured synthesis rather than a new algorithm. Its “method” is the analytical framework and the way evidence is organized.

Step 1 — Establish background and the evaluation pipeline
- Sec. 2.1 explains LLM building blocks (Transformer, in-context learning, RLHF). RLHF (Reinforcement Learning from Human Feedback) fine-tunes a model to prefer human-approved outputs.
- Fig. 3 abstracts the evaluation pipeline around four moving parts: “What” (task), “Where” (data/benchmarks), “How” (process/metrics), and “Model”. This clarifies that evaluation is not just metrics but also careful task and dataset choice.

Step 2 — Define “What to evaluate” (Sec. 3)
- Natural Language Processing capabilities (Sec. 3.1; Table 2):
  - NLU (sentiment, text classification, NLI, semantics, social knowledge).
  - Reasoning (mathematical, commonsense, logical, multi-step).
  - NLG (summarization, dialogue, translation, QA, style transfer).
  - Multilingual performance.
  - Factuality and hallucination.
- Safety and Reliability (Sec. 3.2; Table 3):
  - Robustness (adversarial prompts, out-of-distribution/OOD generalization). OOD means test inputs come from a different distribution than training/validation.
  - Ethics/bias (toxicity, stereotypes).
  - Trustworthiness (privacy, consistency, calibrated confidence, hallucination).
- Applied domains
  - Social science (Sec. 3.3), natural science and engineering (Sec. 3.4; Table 4), medical uses (Sec. 3.5; Table 5), tool-using agents (Sec. 3.6), and other applications (education, search/recommendation, personality testing; Sec. 3.7; Table 6).

Step 3 — Define “Where to evaluate” (Sec. 4; Table 7)
- General-purpose benchmarks (e.g., HELM, MMLU, BIG-bench, AGIEval, MT-Bench, Chatbot Arena).
- Task-specific benchmarks (e.g., APPS for coding, MATH for math, CEval/CMMLU for Chinese academic tasks, SafetyBench/CVALUES for safety).
- Multi-modal benchmarks (e.g., MME, MMBench, SEED-Bench, MM-Vet, LAMM, LVLM-eHub).

Step 4 — Define “How to evaluate” (Sec. 5)
- Automatic evaluation (Sec. 5.1; Table 9):
  - Accuracy-style metrics (e.g., Exact Match, F1, ROUGE).
  - Calibration metrics (e.g., Expected Calibration Error/ECE; Area Under the selective accuracy–coverage Curve/AUC).
  - Fairness metrics (e.g., Demographic Parity Difference/`DPD`, Equalized Odds Difference/`EOD`), which measure whether predictions differ across sensitive groups.
  - Robustness metrics:
    - Attack Success Rate/`ASR`: fraction of previously correct predictions flipped by adversarial inputs. Formula (Sec. 5.1): `ASR = sum I[f(A(x)) ≠ y] / sum I[f(x) = y]`.
    - Performance Drop Rate/`PDR`: relative degradation after adversarial prompt attack (Sec. 5.1).
- Human evaluation (Sec. 5.2; Table 10):
  - When automatic metrics fail (e.g., open-ended generation), use human raters.
  - Key rubrics: accuracy, relevance, fluency, transparency, safety, and human alignment.
  - Design guidance includes rater count for statistical power and domain expertise requirements.

Design choices and why they matter
- Separating “What–Where–How” prevents common evaluation pitfalls: choosing a metric without ensuring task coverage (“What”), or using a dataset that does not reflect deployment (“Where”), or assessing only accuracy when robustness/bias drive real risk (“How”) (Fig. 3; Secs. 3–5).
- Including robustness/fairness/hallucination turns evaluation from leaderboards into risk assessment (Sec. 3.2, 5.1).

## 4. Key Insights and Innovations
- A unifying evaluation schema for LLMs (fundamental innovation)
  - The “What–Where–How” decomposition (Fig. 3; Secs. 3–5) is a clean abstraction that integrates capability testing, benchmark choice, and methodology. It generalizes earlier single-axis views (e.g., task-only leaderboards) and explicitly includes safety and social impact.
- Curated, cross-domain evidence of strengths and weaknesses (synthesis with actionable takeaways)
  - Sec. 6.1 distills success/failure patterns across many studies (e.g., strong at QA and arithmetic reasoning; weak at NLI, abstract reasoning, multi-hop, non-Latin multilingual, and robust performance). This turns heterogeneous literature (Tables 2–6) into decision-ready guidance.
- Formalization of evaluation beyond accuracy (scope broadening)
  - Table 9 codifies calibration, fairness (`DPD`, `EOD`), and robustness (`ASR`, `PDR`), expanding default evaluation toolkits. Including both mathematical definitions and procedural advice improves reproducibility (Sec. 5.1).
- Concrete roadmap of seven grand challenges (agenda-setting)
  - Sec. 7 proposes: AGI benchmarks; complete behavioral evaluation; robustness evaluation; dynamic/evolving evaluation; principled evaluation; unified evaluation spanning all LLM tasks; and “beyond evaluation” guidance to improve models. This reframes evaluation as an essential research discipline rather than an afterthought.

## 5. Experimental Analysis
Because this is a survey, “experiments” are aggregated evidence from many works. The paper anchors conclusions to specific sections, figures, and tables.

Evaluation methodology (What/Where/How)
- Datasets/benchmarks (“Where”)
  - General: HELM, MMLU, BIG-bench, AGIEval, MT-Bench, Chatbot Arena, AlpacaEval (Table 7; Sec. 4.1).
  - Task-specific: APPS (code), MATH (math), CEval/CMMLU (Chinese academic), SafetyBench/CVALUES (safety), FRESHQA (dynamic QA), Dialogue-CoT (in-depth dialogue), API-Bank/ToolBench (tool use) (Sec. 4.2).
  - Multi-modal: MME, MMBench, SEED-Bench, MM-Vet, LAMM, LVLM-eHub (Sec. 4.3).
- Metrics (“How”)
  - Automated: accuracy/EM/F1/ROUGE, calibration (ECE), fairness (`DPD`, `EOD`), robustness (`ASR`, `PDR`) (Table 9; Sec. 5.1).
  - Human: multi-dimensional rubrics (accuracy, relevance, fluency, transparency, safety, human alignment), with rater expertise and sample size guidance (Table 10; Sec. 5.2).
- Baselines and setups
  - Many results compare ChatGPT, GPT‑3.5/4, Claude, LLaMA-family, PaLM, and task-specific fine-tuned systems (summarized per task in Sec. 3; Tables 2–6).

Main quantitative and qualitative findings (with citations)
- NLU and classification (Sec. 3.1.1; Table 2)
  - Credibility classification of news outlets achieves “acceptable accuracy” in the binary setting with AUC=0.89 (Yang & Menczer; Sec. 3.1.1).
  - GLM‑130B tops a broad text classification batch at 85.8% (HELM-style evaluation; Sec. 3.1.1).
  - NLI remains weak: models struggle to model human disagreement and perform poorly on some NLI suites (Sec. 3.1.1).
- Reasoning (Sec. 3.1.2; Table 2)
  - Arithmetic/logical: GPT‑4 and ChatGPT generally outperform GPT‑3.5 on arithmetic and logical reasoning benchmarks, but still falter on out-of-distribution and multi-step reasoning (Sec. 3.1.2). On abstract reasoning, capability remains limited (Sec. 3.1.2).
  - Multi-step: LLaMA‑65B is the most robust open-source baseline, but still below code-davinci-002, while PaLM and Claude2 approach GPT-family performance yet remain worse overall (Sec. 3.1.2).
  - Conditional strengths: temporal reasoning tends to be stronger than spatial; multi-hop reasoning and counterfactual variants expose limitations (Sec. 3.1.2).
- Generation (Sec. 3.1.3; Table 2)
  - Summarization: large models such as TNLG v2 (530B) and OPT (175B) lead; zero-shot ChatGPT trails fine-tuned summarizers and GPT‑3.5 (Sec. 3.1.3).
  - Dialogue: ChatGPT/Claude score strongly on multi-dimensional dialogue quality; task-oriented, fully finetuned systems can still win in narrow domains (Sec. 3.1.3).
  - Translation: strong on X→English, weaker on English→X and non-Latin scripts; GPT‑4 often better at discourse-aware explanations but may still choose wrong candidates (Sec. 3.1.3).
  - QA: instruction-tuned models (e.g., InstructGPT) excel; ChatGPT surpasses GPT‑3.5 in most domains but is slightly weaker on some commonsense tasks due to cautious refusals (Sec. 3.1.3).
- Multilingual (Sec. 3.1.4)
  - Generative LLMs degrade substantially in non-Latin and low-resource languages—even after translating prompts to English (Sec. 3.1.4).
- Factuality and hallucination (Sec. 3.1.5)
  - > “GPT‑4 and BingChat can provide correct answers for more than 80% of the questions” on open QA (Natural Questions, TriviaQA), yet a “remaining gap of over 15% to achieve complete accuracy” persists (Sec. 3.1.5).
  - Evaluation methods include converting consistency checks to binary NLI-style judgments, decomposing outputs into atomic facts (FActScore), and zero-resource hallucination detectors (SelfCheckGPT) (Sec. 3.1.5).
- Robustness, ethics, trustworthiness (Sec. 3.2; Table 3)
  - Adversarial prompts are effective across character→semantic levels; PromptBench formalizes `PDR` to quantify drop (Sec. 3.2.1; 5.1).
  - OOD robustness is limited (GLUE‑X, AdvGLUE++), and multimodal models are vulnerable to visual perturbations with transferable attacks (Sec. 3.2.1).
  - Toxicity and social bias persist and can be amplified by role-playing personas; moral and political leanings are measurable; system prompts can be jailbroken (Sec. 3.2.2).
  - Trust issues include vacillation under negations or misleading cues, poor calibration, and widespread hallucinations in multimodal settings (Sec. 3.2.3).
- Social science, science & engineering, and tools (Secs. 3.3–3.6; Tables 4–6)
  - Social/computational social science: best on misinformation/stance/emotion; weakest on event-argument extraction, implicit hate, empathy (<40% accuracy) (Sec. 3.3).
  - Math: strong on basic arithmetic with GPT‑4 > ChatGPT; poor on complex competition problems and algebraic manipulation (Sec. 3.4.1).
  - Engineering: code generation is competent (GPT‑4 excels in comprehension/reasoning about code), but automated planning and commonsense planning remain weak (Secs. 3.4.3).
  - Tool use: frameworks like API‑Bank, Toolformer, ToolBench, MRKL, and HuggingGPT formalize when/how models should call external tools (Sec. 3.6; Sec. 4.2).
- Medical (Sec. 3.5; Table 5)
  - > ChatGPT “achieves or approaches the passing threshold” on the USMLE without task-specific training (Sec. 3.5.2).
  - > On surgery-related clinical questions, GPT‑4 scores 76.4% vs 46.8% for GPT‑3.5 (Sec. 3.5.3).
  - Quality limitations: unreliable citations and occasional fabricated content restrict clinical utility (Sec. 3.5.1).
- Community and dynamic evaluations (Sec. 4.1)
  - Chatbot Arena aggregates >1M human votes to produce Elo-style ratings; MT-Bench targets multi-turn dialogue with GPT‑4-as-judge (Sec. 4.1).

Ablations, failure analyses, robustness checks
- The survey highlights adversarial prompt studies (PromptBench) and OOD stress tests (GLUE‑X, BOSS) (Secs. 3.2.1, 4.1–4.2).
- Human-in-the-loop and crowd-sourcing approaches (AdaVision, AdaTest, DynaBench) explore interactive error discovery (Sec. 6.2; Table 8).

Do the results support the claims?
- Yes, the cross-validated pattern—strong general language abilities but persistent weaknesses in NLI, abstract/multi-hop reasoning, non-Latin multilinguality, and robustness—recurs across multiple independent benchmarks and domains (Sec. 6.1; Tables 2–7). The paper is careful to state dataset-conditional conclusions (Sec. 6.1).

## 6. Limitations and Trade-offs
- Assumptions and scope
  - Many summarized results are dataset-specific; performance can change with prompt formats, sampling temperature, or model versions (Sec. 6, Disclaimer).
  - Static public benchmarks risk contamination or memorization by web-scale LLM training (Sec. 7.4).
- Coverage gaps
  - Not every new benchmark can be included (fast-moving field; Table 7 note).
  - Behavioral evaluation in open-world settings (robots, multi-agent systems) is identified as necessary but remains underdeveloped (Sec. 7.2).
- Methodological trade-offs
  - Adversarial evaluation (e.g., AdaFilter) can create “unfair” tests if not carefully controlled (Sec. 6.2).
  - Human evaluation has rater variance and cultural/contextual sensitivity (Sec. 5.2).
- Practical constraints
  - Computation and cost limit broad human evaluations; closed APIs hinder full metric access (e.g., token-level probabilities for calibration; Sec. 5.1).
  - Multi-modal safety evaluation is harder due to attack transferability across modalities (Sec. 3.2.1).

## 7. Implications and Future Directions
- Field-level shifts this survey catalyzes
  - Treat evaluation as an independent discipline with its own methods and theory, not just leaderboard reporting (Sec. 7).
  - Normalize inclusion of safety, robustness, and human-alignment metrics alongside accuracy (Sec. 5.1–5.2; Table 9–10).
- Research directions (Sec. 7)
  - AGI benchmarks: Design truly diagnostic, cross-domain, contamination-resistant tests that go beyond human-crafted puzzles (Sec. 7.1).
  - Complete behavioral evaluation: Evaluate LLM-agents in open environments (e.g., robotics), with multi-modal inputs, long-horizon tasks, and tool use (Sec. 7.2).
  - Robustness: Expand adversarial and OOD stress testing; standardize `ASR`/`PDR` reporting; consider prompt distribution shifts (Sec. 7.3).
  - Dynamic/evolving evaluation: Continual test-set refresh (e.g., FRESHQA), leakage control, and time-aware benchmarks (Sec. 7.4).
  - Principled/trustworthy evaluation: Meta-evaluation of the evaluators; measurement theory for reliability/validity; proof-of-OOD sampling; judge-model audits (Sec. 7.5).
  - Unified evaluation for all tasks: One framework spanning instruction-tuning, safety, verification, and multi-modal tasks—akin to HELM but broader (Sec. 7.6).
  - Beyond evaluation: Turn findings into actionable improvements (e.g., prompt robustness guidance from PromptBench; bias mitigation informed by CVALUES/SafetyBench) (Sec. 7.7).
- Practical applications
  - Procurement and compliance checklists for enterprises can adopt the paper’s “What–Where–How” and Table 9–10 metrics as default due diligence templates.
  - Product teams can prioritize mitigations where failures are most consistent: multi-hop/abstract reasoning, non-Latin multilingual support, adversarial prompt defenses, citation verification.
  - Regulators and standard bodies can use the curated safety/ethics benchmarks (SafetyBench, CVALUES, TRUSTGPT) as starting points for policy-aligned test suites (Sec. 4.2).

> Bottom line: Fig. 3’s “What–Where–How” gives teams a practical blueprint to evaluate LLMs holistically, Table 7 maps the benchmark landscape, Tables 9–10 operationalize metrics and human review, and Sec. 7 sets a research agenda that upgrades evaluation from leaderboard chasing to trustworthy, dynamic, risk-aware assessment.
