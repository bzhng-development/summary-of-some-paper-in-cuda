# The Prompt Report: A Systematic Survey of Prompt Engineering Techniques

**ArXiv:** [2406.06608](https://arxiv.org/abs/2406.06608)

## 🎯 Pitch

This paper delivers the most comprehensive, evidence-based synthesis of prompt engineering to date by conducting a PRISMA-guided systematic review that maps 58 text-based and 40 multimodal/multilingual prompting techniques into an actionable taxonomy, paired with a unified vocabulary. By benchmarking prompting strategies and providing empirical case studies—including high-stakes domains like suicidality detection—it empowers both practitioners and researchers to navigate and standardize best practices in a field whose inconsistent terminology and scattered methods have previously impeded reliable, secure, and effective AI deployment.

---

## 1. Executive Summary (2–3 sentences)
This paper systematizes the rapidly expanding area of prompt engineering by (a) performing a PRISMA‑guided systematic literature review that yields a taxonomy of 58 text‑based prompting techniques (plus 40 multimodal/multilingual variants) and a consolidated vocabulary of 33 terms, and (b) validating practice through two empirical studies: a technique benchmark on MMLU and a real‑world prompt‑engineering case study for detecting suicidal “entrapment.” It matters because prompting is the primary interface to modern generative models; inconsistent terminology, scattered techniques, and unclear best practices hinder reliable deployment, evaluation, and safety.

## 2. Context and Motivation
- Problem/gap addressed
  - Prompt engineering has grown ad hoc with conflicting terminology and overlapping techniques, which makes it hard to know what works, when, and why (Section 1; Figure 1.3 for terminology; Figure 2.2 for technique map).
  - There is no consolidated, evidence‑based guide that spans text, multilingual, and multimodal prompting, while also covering safety (prompt hacking) and evaluation (Sections 3–5).
  - Practical guidance has lacked empirical case studies demonstrating how experts iterate prompts on real tasks (Section 6).

- Why this is important
  - Prompts are the operational handle on LLM behavior in consumer, enterprise, and research settings (Section 1). Better prompts consistently improve performance across tasks (Section 1 citing Wei et al., 2022b; Liu et al., 2023b).
  - Security and safety failures (e.g., prompt injection and jailbreaking) create real business risk (Section 5.1); poor calibration, bias, and ambiguity hurt reliability (Section 5.2).

- Prior approaches and where they fall short
  - Earlier surveys covered prompting broadly (e.g., including cloze or soft prompts) or specific subareas (reasoning, multimodal, etc.). They did not provide an up‑to‑date, PRISMA‑grounded synthesis focused on modern hard, prefix prompts, nor did they pair a field map with actionable empirical case studies (Section 7).

- Positioning
  - Scope is deliberately focused on standard, deployable prompting: hard (discrete) `prefix` prompts, not `cloze` or soft prompts, and task‑agnostic techniques (Section “Scope of Study”). The work unifies vocabulary, organizes techniques, quantifies usage, and demonstrates practice through benchmarking and a detailed real‑world prompt engineering process.

## 3. Technical Approach
This paper has three pillars: a systematic review to build a taxonomy, an empirical benchmark, and a real‑world case study.

- Systematic review (PRISMA pipeline)
  - Data sources: arXiv, Semantic Scholar, ACL; 44 prompting‑related keywords (Appendix A.4).
  - Process (Figure 2.1; Section 2.1.1):
    - Start with 4,247 unique records after deduplication.
    - Human review on 1,661 arXiv titles/abstracts with 92% inter‑annotator agreement; criteria focused on novel prompting techniques using hard, prefix prompts; fine‑tuning papers excluded.
    - LLM‑assisted screening of remaining records using a GPT‑4 prompt (Appendix A.5); validated at 89% precision and 75% recall (F1 = 81%).
    - Final corpus: “1,565 quantitative records included in analysis” (Figure 2.1).
  - Output:
    - A terminology chart capturing components of prompts and related artifacts (Figure 1.3).
    - A taxonomy of text‑based prompting techniques grouped into six families (Figure 2.2).
    - Extensions for multilingual and multimodal prompting (Figures 3.1 and 3.2).
    - A structured treatment of security, safety, and evaluation (Sections 4–5).

- Terminology and prompt components
  - A `prompt` is any input (text, image, audio, etc.) used to guide a generative model’s output; prompts often come from `prompt templates`, which are functions with variables that render into concrete prompts (Section 1.1; Figure 1.2).
  - Typical components (Section 1.2.1):
    - `Directive` (the task, e.g., “Classify…”).
    - `Exemplars` (a.k.a. “shots”).
    - `Output formatting` and `style` constraints.
    - `Role` (a persona to influence style or behavior).
    - `Additional Information` (domain facts, constraints).

- Taxonomy of text techniques (Figure 2.2; Sections 2.2.1–2.2.5)
  - `In‑Context Learning (ICL)`: learning from exemplars or instructions inside the prompt; includes `few‑shot` prompts and `zero‑shot` instruction prompts (Figures 2.4–2.6).
  - `Thought generation`: inducing explicit reasoning, e.g., `Chain‑of‑Thought (CoT)` and its zero‑shot, few‑shot, and table/analogical variants (Section 2.2.2; Figure 2.8).
  - `Decomposition`: dividing complex problems into sub‑problems (`Least‑to‑Most`, `Tree‑of‑Thought`, `Program‑of‑Thoughts`) (Section 2.2.3).
  - `Ensembling`: produce multiple answers using prompt variations or sampling, then aggregate (`Self‑Consistency`, `DENSE`, `USP`) (Section 2.2.4).
  - `Self‑criticism`: generate, critique, and revise answers (`Self‑Refine`, `COVE`, `Self‑Verification`) (Section 2.2.5).

- Formalization and answer engineering
  - Prompts are treated as conditioning mechanisms on a language model `p_LM`, optionally via a template `T(x)` and few‑shot exemplars `X` (Appendix A.8, Eqs A.1–A.5).
  - Prompt optimization is maximizing a score function `S` over a dataset, sometimes jointly with an `answer extractor` `E` that parses LLM outputs into canonical answers (Appendix A.8, Eqs A.6–A.8; Section 2.5).
  - `Answer engineering` decisions: answer `shape` (token/span), `space` (allowed values), and `extractor` (regex, verbalizer, or a separate LLM) (Section 2.5; Figure 2.13).

- Empirical benchmark (Section 6.1)
  - Task: MMLU subset (2,800 questions; 20% per category; sensitive “human_sexuality” excluded).
  - Model: `gpt-3.5-turbo`.
  - Prompting conditions (Figure 6.2): `Zero‑Shot`; `Zero‑Shot CoT` with three thought inducers; `Zero‑Shot CoT` with `Self‑Consistency` (3 samples); `Few‑Shot`; `Few‑Shot CoT`; and `Few‑Shot CoT` with `Self‑Consistency`.
  - Two question formats (Figures 6.3 and 6.4).
  - Decoding: temperature 0.5 for `Self‑Consistency`, 0.0 otherwise (Section 6.1.3).
  - Answer parsing: pattern‑based extraction of choices; variations tested (Section 6.1.4).

- Real‑world case study: suicidal “entrapment” detection (Section 6.2)
  - Data: 221 Reddit r/SuicideWatch posts labeled by two trained coders for presence/absence of “entrapment” (Krippendorff’s α = 0.72) (Section 6.2.2).
    - `Entrapment` is defined as “a desire to escape from an unbearable situation, tied with the perception that all escape routes are blocked” with concrete phrasing cues (Figure 6.7).
  - Goal: prompt‑engineer a binary classifier using only prompting (no fine‑tuning).
  - Process (~20 hours, 47 steps) included:
    - Guardrail conflicts: initial models responded with crisis advice instead of labels; switching to `GPT‑4‑32K` allowed one‑word labels (Section 6.2.3.2).
    - Iterative technique exploration: zero‑shot with definition, few‑shot, `CoT`, `contrastive CoT`, targeted answer extraction, ensembling, and a new method, `Automatic Directed CoT (AutoDiCoT)` (Figures 6.12–6.16).
  - `AutoDiCoT` (Figure 6.12):
    - For each training item, elicit a reasoning chain `r_i` that either justifies a correct label or explains why the earlier label was wrong; store triplets `(q_i, r_i, a_i)`.
    - Use selected triplets as exemplars (including an “incorrect reasoning” one) to steer reasoning on new inputs (Figures 6.13, 6.16).
  - Observations:
    - Seemingly innocuous context duplication (pasting the same context email twice) improved performance, and de‑duplication hurt it (Section 6.2.3.3: “Full Context Only,” “De‑Duplicating Email”).
    - Over‑restricting to “explicit” entrapment raised precision but crashed recall—misaligned with the clinical goal of minimizing false negatives (Section 6.2.3.3).

- Evaluation frameworks and safety (Sections 4.2 and 5)
  - LLM‑as‑evaluator designs: prompt roles, CoT, model‑generated guidelines, output formats (binary, Likert, JSON/XML), and frameworks like `LLM‑EVAL`, `G‑EVAL`, and `ChatEval` (Section 4.2.1–4.2.3).
  - Security taxonomy: `prompt hacking` ⊇ `prompt injection` (override developer instructions) and `jailbreaking` (coax unsafe actions with no developer instruction present), plus risks (data leakage, package hallucination, chatbot liability), and defenses (detectors, guardrails) (Section 5.1; Figure 5.1).

## 4. Key Insights and Innovations
- A field‑wide, PRISMA‑grounded taxonomy and vocabulary
  - Novelty/significance: integrates 58 text‑based techniques into six coherent families (Figure 2.2) and provides a consistent terminology (Figure 1.3), reducing confusion across papers that use overlapping or conflicting names (Section 1.2).
  - Difference from prior: earlier reviews were broader or less structured; here the focus on deployable hard, prefix prompts plus a formal prompt definition and answer engineering pipeline (Appendix A.8; Section 2.5) is practice‑oriented.

- A practical, formal view of `answer engineering`
  - What’s new: elevates output parsing as a first‑class design space—answer `shape`, `space`, and `extractor` (Section 2.5; Figure 2.13)—and unifies it with prompt optimization via Eq. A.8 (Appendix A.8).
  - Why it matters: many failures in real systems come from brittle post‑processing rather than modeling; formalizing this makes evaluations repeatable and prompts more robust.

- `Automatic Directed CoT (AutoDiCoT)` for reasoning control
  - What it is: a simple algorithm to build CoT exemplars that explicitly steer reasoning away from observed error modes by including “what not to do” alongside correct rationales (Figure 6.12; One‑Shot contrastive example in Figure 6.13).
  - Impact: on the entrapment task, a `10‑Shot AutoDiCoT` prompt reached the best development F1 (0.53) with recall 0.86 and precision 0.38 (Section 6.2.3.3; Figure 6.16 and Figure 6.6). This demonstrates targeted reasoning control without fine‑tuning.

- Empirical clarity on technique performance trade‑offs
  - Benchmark result (MMLU; `gpt‑3.5‑turbo`): `Few‑Shot CoT` outperforms `Zero‑Shot` and `Zero‑Shot CoT`, while `Self‑Consistency` helps zero‑shot but not few‑shot in this setup (Section 6.1.5; Figure 6.1).
  - Quote of main numbers:
    > Zero‑Shot 0.627; Zero‑Shot CoT 0.547; Zero‑Shot CoT + Self‑Consistency 0.574; Few‑Shot 0.652; Few‑Shot CoT 0.692; Few‑Shot CoT + Self‑Consistency 0.691 (Figure 6.1).

- A consolidated treatment of prompt security and alignment
  - Security: clear distinctions between `prompt injection` and `jailbreaking`, concrete risks (e.g., training‑data leakage, package hallucination), and layered defenses (detectors, guardrails) (Section 5.1; Figure 5.1).
  - Alignment: practical prompt‑level mitigations for prompt sensitivity, miscalibration, sycophancy, bias, and ambiguity (Section 5.2; Figure 5.2).

## 5. Experimental Analysis
- Evaluation methodology
  - Benchmark (Section 6.1):
    - Dataset: MMLU subset (2,800 items; 20% per category).
    - Model: `gpt-3.5-turbo`.
    - Prompt settings: 6 technique families with 2 question formats; temperature 0.5 for `Self‑Consistency`, else 0.0.
    - Parsing: pattern‑based extraction rules (Section 6.1.4).
  - Case Study (Section 6.2):
    - Dataset: 221 Reddit posts; 121 for development, 100 for test.
    - Metric: F1, plus precision and recall reported throughout.
    - Iterative exploration across ~20 prompting variants (Figures 6.5, 6.6).

- Main quantitative results
  - Benchmark headline (Figure 6.1):
    > Accuracy: `Few‑Shot CoT` 0.692 ≈ `Few‑Shot CoT + Self‑Consistency` 0.691, both better than `Few‑Shot` 0.652 and `Zero‑Shot` 0.627; `Zero‑Shot CoT` alone drops to 0.547; `Zero‑Shot CoT + Self‑Consistency` recovers to 0.574.
  - Case study progression (Figures 6.5–6.6):
    - Starting point: zero‑shot with definition (Section 6.2.3.3 “Zero‑Shot + Context”) achieved F1 0.40 with recall 1.00 and precision 0.25.
    - Best development result: `10‑Shot AutoDiCoT` (with duplicated context) reached F1 0.53, recall 0.86, precision 0.38 (Figure 6.16; Figure 6.6).
    - Sensitivity: removing duplicated context (“De‑Duplicating Email”) reduced F1 to 0.45 (recall 0.74; precision 0.33).
    - Test set via automated prompt optimization (`DSPy`, BootstrapFewShotWithRandomSearch): 
      > “F1 0.548 (precision 0.385, recall 0.952)” without using the email or the “explicit entrapment” constraint (Figure 6.19).
  
- Do the experiments support the claims?
  - The MMLU benchmark systematically compares common techniques under a controlled setup and reveals a non‑obvious outcome: `Zero‑Shot CoT` can hurt performance unless stabilized by `Self‑Consistency` (Figure 6.1). This supports the paper’s caution that technique effectiveness is context‑dependent (Section 6.1.5).
  - The entrapment case study convincingly demonstrates the realities of prompt engineering: sensitivity to context, the value of exemplar selection and reasoning control (`AutoDiCoT`), and the importance of aligning prompt objectives to domain goals (high recall in clinical screening) (Section 6.2.4).

- Ablations, failures, robustness
  - Answer‑extraction choices mattered; parsing only the “first characters” improved F1 over “exact match” during early steps (Section 6.2.3.3).
  - Ensembling (`10‑Shot AutoDiCoT Ensemble + Extraction`) unexpectedly degraded performance due to unstructured outputs requiring additional extraction (Section 6.2.3.3).
  - Overly strict instruction (“only explicit entrapment”) improved precision but sharply reduced recall, revealing value misalignment for the clinical use case (Section 6.2.3.3).
  - The paper quantifies model/dataset citation frequency to characterize technique adoption (Figures 2.9–2.11). For example:
    > Figure 2.11 shows `Chain‑of‑Thought` and `Few‑Shot` methods among the most cited in the corpus.

- Conditionality and trade‑offs
  - `Self‑Consistency` improves zero‑shot CoT but has little added value for few‑shot CoT in this benchmark—an interaction effect worth checking per task/model (Figure 6.1).
  - Reasoning control (`AutoDiCoT`) increased recall at some cost in precision—beneficial for high‑risk screening, but perhaps not for precision‑critical tasks.

## 6. Limitations and Trade-offs
- Scope constraints
  - Focused on `hard` (discrete) `prefix` prompts; excludes soft prompts and gradient‑based prompt tuning (Section “Scope of Study”).
  - Task‑agnostic techniques only; domain‑specific prompting is out of scope (Section “Scope of Study”).

- Empirical limits
  - Benchmark uses a single model (`gpt‑3.5‑turbo`), one dataset (MMLU subset), and limited variants (Section 6.1), so generalization across models/tasks is not guaranteed.
  - Parsing‑based evaluation can misjudge answers if the model’s formatting drifts (Section 6.1.4; Section 2.5).
  - Citation counts as a proxy for usage (Figures 2.9–2.11) reflect research discourse, not necessarily industry adoption.

- Prompt sensitivity and brittleness
  - Small format variations (e.g., duplicated context email) significantly affected results (Section 6.2.3.3), echoing broader sensitivity findings (Section 5.2.1).
  - Over‑constraint or poorly aligned instructions can degrade metrics most valued by stakeholders (e.g., recall in clinical screening) (Section 6.2.3.3).

- Compute, cost, and latency
  - Techniques like `Self‑Consistency`, ensembling, `Tree‑of‑Thought`, or agentic tool‑use increase API calls, latency, and cost (Sections 2.2.4, 4.1), which may limit real‑time or large‑scale use.

- Security defenses remain partial
  - Prompt‑based defenses can mitigate but not eliminate prompt hacking; detectors/guardrails reduce risk but are not foolproof (Section 5.1.3).

## 7. Implications and Future Directions
- How this work changes the landscape
  - Provides a shared map and vocabulary for the field (Figures 1.3, 2.2, 3.1, 3.2), making it easier to reason about design choices, compare techniques, and teach best practices.
  - Bridges prompting research with safety and evaluation, promoting end‑to‑end thinking: prompts, answer extraction, evaluation pipelines, and defenses (Sections 2.5, 4.2, 5).

- Follow‑up research enabled
  - Multi‑model, multi‑dataset replications of the benchmark to test interaction effects (e.g., when `Zero‑Shot CoT` helps vs. hurts).
  - Programmatic methods for `answer engineering` (learned extractors, structured decoding) that reduce formatting brittleness (Section 2.5).
  - Generalized `Directed CoT` methods: algorithmic selection of “what not to do” exemplars, with uncertainty‑aware sampling or RL.
  - Robust prompting under adversarial or noisy inputs; formal safety metrics for guards/detectors (Section 5.1.3).

- Practical applications and downstream use cases
  - Enterprise prompt design playbooks: exemplar selection (Figure 2.3), role/style instructions (Section 2.2.1.3), ensembling with cost control (Section 2.2.4), and built‑in answer extraction (Section 2.5).
  - Safety‑critical screening workflows (healthcare, trust & safety): prioritize high‑recall prompts, add calibration prompts (Section 5.2.2), and include human‑in‑the‑loop confirmation.
  - Agentic systems: tool routing and retrieval‑augmented reasoning patterns (`MRKL`, `ReAct`, `IRCoT`) for tasks needing factuality and planning (Section 4.1; Figure 4.1).
  - LLM‑as‑evaluator: adopt `G‑EVAL`/`LLM‑EVAL` styles with CoT and model‑generated guidelines for consistent, auditable assessments (Section 4.2).

> Overall, the work offers a coherent blueprint for designing, evaluating, and securing prompt‑driven systems—from vocabulary and taxonomy (Figures 1.3, 2.2) through empirical guidance (Figure 6.1) and real‑world procedure (Figures 6.12–6.16)—while candidly surfacing sensitivity and safety pitfalls that practitioners must manage.
