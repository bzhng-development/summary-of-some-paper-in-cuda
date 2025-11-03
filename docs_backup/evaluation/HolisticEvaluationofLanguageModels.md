# Holistic Evaluation of Language Models

**ArXiv:** [2211.09110](https://arxiv.org/abs/2211.09110)

## 🎯 Pitch

This paper introduces HELM, a comprehensive framework for evaluating language models across a uniquely broad taxonomy of 42 scenarios and 7 key metric families—including accuracy, robustness, fairness, toxicity, and efficiency—enabling standardized, head-to-head comparison of 30 state-of-the-art models. By systematically surfacing both strengths and shortcomings across diverse use cases and metrics, HELM reveals critical trade-offs and sets a new standard for transparency and rigor in language model assessment, informing responsible development and deployment for real-world impact.

---

## 1. Executive Summary (2–3 sentences)
HELM (“Holistic Evaluation of Language Models”) introduces a taxonomy, benchmark, and tooling to evaluate language models comprehensively across 42 scenarios and 7 metric families, and reports a dense, standardized comparison of 30 prominent models. It shows how capabilities, risks, and efficiency vary by scenario and adaptation method, and surfaces consistent patterns (e.g., instruction-tuned models lead on accuracy and robustness; accuracy correlates strongly with robustness and fairness), enabling more transparent, head‑to‑head evaluations and clearer trade‑offs (see §1.1, §1.2; Fig. 3, Fig. 4, Fig. 24–26).

## 2. Context and Motivation
- Problem/gap addressed
  - Evaluation of language models (LMs) is fragmented and narrow:
    - Most benchmarks emphasize a single metric (accuracy) and a small set of tasks, leaving important desiderata (e.g., robustness, fairness, toxicity, efficiency) under-measured or siloed into separate bespoke datasets (Fig. 3; §1.1 “Multi-metric measurement”).
    - Head‑to‑head comparisons have been hard because models are adapted differently (e.g., fine-tuning vs. 0/5‑shot prompting), evaluated on disjoint scenario sets, and sometimes behind proprietary APIs (§1.1 “Standardization”; §6).
  - Before HELM, prominent models often had no overlapping test sets; on average they were evaluated on only 17.9% of HELM’s core scenarios even after aggregating across multiple papers (Fig. 4, top; §1.1).
- Why this matters
  - Real-world deployments hinge not only on accuracy but on reliability (calibration and robustness), equity (fairness and bias), safety (toxicity), and usability (efficiency); failing to evaluate these in the same contexts obscures trade‑offs critical for responsible use (§1, §4).
- Prior approaches and shortcomings
  - Task suites like GLUE/SuperGLUE, BIG-bench, or LM harnesses advanced breadth, but typically:
    - Center accuracy, rarely report multiple non‑accuracy metrics per use case (Fig. 3, left).
    - Lack standardization of adaptation (prompting vs. fine‑tuning) that affects outcomes (§7, §8.2).
    - Provide limited coverage across domains, dialects, and targeted risks (e.g., memorization, disinformation) (§5; §10).
- This paper’s positioning
  - HELM supplies: (i) a top‑down taxonomy for scenarios (by task, domain, language) and metrics; (ii) a concrete benchmark implementation emphasizing coverage and multi‑metric density (98 of 112 core scenario–metric pairs, Table 4); and (iii) standardized evaluation of 30 models under unified conditions, improving overlap to 96% across core scenarios (Fig. 4, bottom; §1.1, §3–4, §6–7).

## 3. Technical Approach
HELM’s methodology has three pillars—taxonomy-driven coverage, multi‑metric measurement, and standardized adaptation—and a large‑scale execution across models.

- Abstraction and primitives (§2; Fig. 5–7)
  - `Scenario`: a use case defined as a list of instances, each with an `input` and reference `outputs` (with labels or properties). Scenarios are structured by task, domain, and language (Fig. 8).
  - `Adaptation`: a method to turn a general LM into a solver for a scenario (e.g., 5‑shot prompting; §7).
  - `Metric`: a quantitative function over model outputs (and probabilities) to assess performance.
- Scenario taxonomy and selection (§3; Fig. 8)
  - Taxonomy dimensions:
    - Task (e.g., question answering, information retrieval, summarization, sentiment analysis, toxicity detection, miscellaneous classification).
    - Domain decomposed as “what” (genre), “who” (speaker demographics), and “when” (time) (Fig. 8).
    - Language (focus here is English and English varieties; §3.1–3.2; Fig. 10).
  - Core set: 16 user‑facing scenarios across six task families; plus 26 targeted scenarios for language, knowledge, reasoning, memorization/copyright, disinformation, bias, toxicity (§3, §5; Table 4).
- Metric taxonomy and selection (§4)
  - Seven metric families span system desiderata; HELM instantiates those that can be measured with black‑box model access:
    - `Accuracy` (task‑specific, e.g., Exact Match, F1, ROUGE‑2, RR@10, NDCG@10; §4.3; Appx C.1).
    - `Calibration` via ECE (expected calibration error) and `selective classification` (accuracy at a given confidence coverage) (§4.4; Fig. 17; Appx C.2).
      - ECE compares predicted confidence with empirical accuracy across probability bins.
    - `Robustness` as worst‑case accuracy over semantic-preserving perturbations (invariance) and over human-crafted contrast sets (equivariance) (§4.5; Fig. 18; Appx D.1).
    - `Fairness` via counterfactual perturbations (dialect, gender, race term substitutions; Fig. 19; Appx D.2) and performance disparities when demographic metadata exists (§4.6).
    - `Bias` in generations measured as demographic representation skew and stereotypical association distance from a uniform reference across groups (§4.7; Fig. 20; Appx C.5 describes word lists and formulas).
    - `Toxicity` in generations via Perspective API, reporting rate of toxic completions (§4.8; Fig. 21).
    - `Efficiency` covering training energy/emissions and two inference metrics:
      - `Denoised runtime`: the provider’s stack with queueing noise factored out.
      - `Idealized runtime`: unified optimized hardware/software (A100 + Megatron) for apples‑to‑apples LM comparison (§4.9; Fig. 22; Appx C.7).
  - Multi‑metric density: 98/112 core scenario–metric pairs (87.5%) measured (Table 4).
- Standardized adaptation (§7; Fig. 23; Table 7)
  - Models are treated as black‑box text‑to‑text APIs (no training data or internals required). All models are adapted with few‑shot prompting (by default 5 in‑context examples) using the same prompt templates and decoding settings where applicable:
    - In‑context examples are fixed across test instances (to reflect real few‑shot use) and sampled to cover label classes; experiments are repeated with 3 different example sets to estimate variance (§7; §8.2; Fig. 31).
    - Multiple‑choice scenarios are adapted in three ways and compared (§8.2; Fig. 33):
      - `Joint`: present all choices and predict the label token.
      - `Separate`: score each choice with the prompt and pick the highest.
      - `Separate‑calibrated`: normalize scores by choice priors.
    - Decoding: temperature 0 for short, deterministic tasks; higher temperature for generation; unified stopping conditions (§7; Appx J.3–J.4).
- Models evaluated (§6; Table 5)
  - 30 models spanning open, limited‑access APIs, and closed deployment (12 organizations). Where possible, HELM also estimates training energy/emissions and measures inference efficiency on common hardware (Appx C.7; §4.9).

## 4. Key Insights and Innovations
- A. A top‑down taxonomy + dense multi‑metric benchmark (fundamental innovation)
  - What’s new: HELM frames evaluation as a matrix of scenarios × metrics, chosen from explicit taxonomies (Fig. 2, Fig. 8; §3–4) rather than an ad‑hoc list of datasets. It then implements a dense subset: 16 core scenarios × 7 metrics (98/112 measured), plus 26 targeted evaluations (Table 4; §5).
  - Why it matters: Measuring multiple desiderata in the same context exposes trade‑offs (Fig. 24–25) and prevents safety/equity metrics from being sidelined (Fig. 3).
- B. Standardized, head‑to‑head evaluation of 30 models (fundamental innovation)
  - What’s new: All models are evaluated under unified adaptation (5‑shot prompting with identical templates), prompting variants are analyzed (§8.2), and scenario overlap is raised from 17.9% to 96.0% across core scenarios (Fig. 4).
  - Why it matters: Fair comparisons are possible; sensitivity to adaptation becomes explicit (e.g., multiple‑choice methods drastically change accuracy, Fig. 33).
- C. New efficiency metrics enabling fairer runtime comparison (incremental but impactful)
  - What’s new: `Denoised` and `idealized` inference runtime separate provider stack effects from model‑intrinsic speed (Fig. 22; §4.9.2), plus training energy/CO₂ estimates for models with enough transparency (Appx C.7).
  - Why it matters: Users and researchers can evaluate capability–efficiency trade‑offs (Fig. 24, bottom right).
- D. Targeted evaluations of risks and primitives (incremental breadth)
  - What’s new: Dedicated suites for linguistic phenomena (BLiMP, ICE; §5.1), knowledge (WikiFact; §5.2), reasoning (Dyck, GSM8K, MATH, LSAT, bAbI, HumanEval/APPS; §5.3), memorization/copyright (books, Linux code; §5.4), disinformation with human evaluation (§5.5, §8.5), and bias/toxicity beyond core (§5.6–5.7).
  - Why it matters: The benchmark reveals capability/risk profiles that core tasks alone would miss (e.g., memorization risk correlates with model capability; §5.4).

## 5. Experimental Analysis
- Evaluation methodology
  - Scenarios and datasets: 16 core user‑facing scenarios (e.g., NaturalQuestions, MS MARCO ranking, CNN/DM, XSUM, IMDB, CivilComments, RAFT) plus 26 targeted across language, knowledge, reasoning, copyright, disinformation, bias, toxicity (Table 4; §3, §5).
  - Metrics: Accuracy variants (EM/F1/ROUGE/RR/NDCG), Calibration (ECE; selective accuracy), Robustness (invariance and contrast sets), Fairness (counterfactuals; disparities), Bias (representation/associations), Toxicity (Perspective API), Efficiency (training and inference) (§4; Table 4).
  - Adaptation: Unified 5‑shot prompting; multiple‑choice variants compared (§7; §8.2; Fig. 33).
- Main quantitative results and comparisons
  - Overall head‑to‑head performance (Fig. 26):
    - > “text‑davinci‑002” wins the most head‑to‑head accuracy comparisons (>90% win rate), and also leads on robustness and fairness; TNLG v2 (530B) is second on accuracy and fairness; Anthropic‑LM v4‑s3 (52B) is consistently top‑3 on accuracy, robustness, and fairness (§1.2; Fig. 26).
  - Model accessibility and accuracy (Fig. 28):
    - > Limited‑access models (e.g., “text‑davinci‑002”) generally outperform open models across core scenarios; open models are sometimes competitive but lag on knowledge‑heavy QA (e.g., MMLU, closed‑book NQ) and IR (§1.2 finding 2; Fig. 28).
  - Accuracy–robustness–fairness correlation (Fig. 24–25):
    - > Across scenarios, accuracy correlates strongly with robustness and fairness; however, top models can still suffer larger drops on some tasks (e.g., NarrativeQA robustness drop: TNLG v2 530B from 72.6% to 38.9%; §1.2 finding 4).
  - Calibration is scenario‑dependent (Fig. 24–25):
    - > On HellaSwag, improving accuracy worsens calibration (higher ECE); on OpenBookQA, accuracy improvements align with better calibration (§1.2 finding 3; §8.1).
  - Sensitivity to adaptation (Fig. 33; §8.2):
    - > For HellaSwag, `separate` > `separate‑calibrated` > `joint`; OPT‑175B shifts from 79.1% EM (separate 0‑shot) to 30.2% EM (joint 5‑shot) (Fig. 33; §8.2). Anthropic‑LM v4‑s3 (52B) reverses this pattern on some tasks (joint works best), underscoring that a single “standard” format can advantage some models over others.
  - Information retrieval (MS MARCO; §3.4; §8.3 IR):
    - > On “regular”, best models reach 39.8% RR@10 (boosted) and ~22.5% RR@10 (vanilla) vs BM25 19.0%; on “TREC”, best models reach 65.3% NDCG@10 (boosted) and 61.0% (vanilla) vs BM25 50.6% (§1.2 finding 9; §8.3 IR).
  - Summarization (CNN/DM, XSUM; §8.3):
    - > Metrics often fail to discriminate quality; TNLG v2 (530B) tops XSUM ROUGE‑2 at 17.9 versus 15.6 for OPT‑175B (Fig. 34; §1.2 finding 10).
  - Sentiment and misc. classification (§8.3):
    - > IMDB: many models >90% EM; best GLM (130B) at 95.5%; calibration varies—BLOOM (176B) ECE≈0.35 (§1.2 finding 11).
    - > RAFT: GLM (130B) reaches 85.8% overall; performance varies widely across its 11 sub‑tasks (§1.2 finding 13).
  - Toxicity detection (CivilComments; §8.3):
    - > Most models near chance; best “text‑davinci‑002” ~66.8% EM; large robustness/fairness drops (OPT‑175B Black split from 51.3% to 8.8% under robustness perturbations; White split 50.8%→24.3%) (§1.2 finding 12).
  - Linguistic evaluations (§5.1; §8.4):
    - > Language modeling BPB: Pile‑trained models (e.g., GPT‑NeoX, OPT) are strongest on The Pile, TwitterAAE, and ICE; BLiMP scores are similar across models, with largest spread on irregular morphology where some top downstream models underperform (Fig. 36; §1.2 finding 14).
    - > Dialect disparities: On TwitterAAE, all models have higher BPB (worse) on AAE vs White English (e.g., OPT‑175B: 2.114 vs 1.506 BPB; §1.2 finding 5).
  - Knowledge and reasoning (§5.2–5.3; §8.4):
    - > Knowledge: “text‑davinci‑002” leads TruthfulQA by a wide margin (62.0% vs 36.2% for Anthropic‑LM v4‑s3) and MMLU (57.0% vs 49.8%); TNLG v2 (530B) excels on closed‑book NQ and WikiFact (§1.2 finding 15; Fig. 37).
    - > Reasoning: code models dominate; “code‑davinci‑002” achieves 52.1% on GSM8K (vs 35.0% for text‑davinci‑002, others ≤16%) and leads on synthetic reasoning (Fig. 38; §1.2 finding 16).
  - Memorization/copyright (§5.4; §8.4; Fig. 39):
    - > Verbatim regurgitation is rare but noticeable for popular books and correlates with accuracy (e.g., text‑davinci‑002, davinci, Anthropic‑LM v4‑s3 show highest regurgitation; §1.2 finding 17).
  - Disinformation (human evaluation; §5.5; §8.5; Table 8):
    - > Models can generate stylistically plausible headlines and divisive messages; “text‑davinci‑002” and Anthropic‑LM v4‑s3 score highest on style/quality for reiteration; wedging results are mixed, with limited accurate audience‑targeting (§1.2 finding 18; §8.5).
  - Generative harms in core tasks (Fig. 24 bottom row; §8.3):
    - > Average bias and toxicity in core scenario generations are low and largely constant across models, but targeted prompts (RealToxicityPrompts) elicit substantially higher toxicity (§1.2 findings 6, 20).
- Ablations and robustness checks
  - Prompt sensitivity: number/choice of in‑context examples and prompt formatting shift results non‑trivially (Fig. 31–32; §8.2).
  - Multiple‑choice adaptation strongly affects accuracy and calibration and the ranking of models (Fig. 33; §8.2).
- Do results support claims?
  - Yes, quantitatively and at scale. The reported head‑to‑head charts (Fig. 26), cross‑metric correlations (Fig. 24–25), and scenario‑specific result pages (linked throughout §8.1–8.4) provide converging evidence for the major claims (instruction‑tuning advantage; accessibility gap; accuracy–robustness–fairness coupling; calibration trade‑offs; code‑model advantage on reasoning).

## 6. Limitations and Trade-offs
- Assumptions and scope limits
  - English‑centric evaluation with targeted but limited coverage of dialects/varieties; multilingual and multimodal tasks are largely out of scope (§3.1–3.2; §10).
  - Only black‑box access is assumed, which prevents metrics that require internals (e.g., interpretability based on activations) (§4.1–4.2).
  - Models are adapted via few‑shot prompting only; fine‑tuning or instruction‑optimization is not explored here, and prompting details substantially affect outcomes (§7; §8.2).
- Measurement constraints
  - Fairness and robustness rely on perturbations; while scalable, they approximate rather than perfectly instantiate social/linguistic variation (Appx D.1–D.2; §4.6 “Discussion”).
  - Toxicity uses Perspective API; known limitations and cultural biases apply (§4.8).
  - Some metrics (e.g., training emissions) require model‑provider disclosures; estimates are approximate (Appx C.7).
- Dataset/model contamination and validity
  - Training–test contamination cannot be ruled out for many models due to incomplete transparency; known evidence is cataloged (Appx G; Table 13). Validity of some standard datasets (e.g., summarization) is debated (§3.5; §11.2).
- Computational cost and scalability
  - Running HELM comprehensively is expensive (12.2B tokens; 17.4M queries; ~$38k API cost plus ~19.5k GPU hours; §1.2) and requires prioritization (Appx H).
- Trade‑offs observed
  - Accuracy vs calibration can conflict (Fig. 24–25); accuracy correlates with robustness/fairness, but top models still suffer large drops on certain tasks (§8.3 QA).
  - Efficiency–capability trade‑offs exist but are model‑family specific; no universal Pareto dominates across all scenarios (Fig. 24 bottom‑right; §1.2 finding 7).

## 7. Implications and Future Directions
- How this work changes the landscape
  - HELM establishes a blueprint for holistic, standardized LM evaluation with explicit taxonomies, dense multi‑metric measurement, and reproducible adaptation. It provides a shared “score matrix” rather than a single scalar score, making real‑world trade‑offs visible and comparable (Fig. 3–4; Table 4).
- Follow‑up research it enables
  - Extension of taxonomies and coverage to:
    - Multilingual and multimodal evaluation; richer domain “who/when/why” coverage (e.g., biomedical, finance, education, and non‑US demographic categories) (§10.1).
    - Deeper fairness/robustness measures, including human‑in‑the‑loop red‑teaming, causal fairness analyses, and improved toxicity detectors (§10.2–10.3).
  - Exploration of adaptation axes (fine‑tuning, parameter‑efficient tuning, prompt optimization, retrieval‑augmentation) under standardized multi‑metric evaluation (§10.5).
  - Better efficiency metrics (end‑to‑end energy per request) and emissions reporting standards; principled capability–efficiency Pareto analyses (§4.9; §10.2).
- Practical applications and downstream use
  - Model selection for deployment can now weigh accuracy against calibration, robustness, fairness, and efficiency within the same scenario contexts (Fig. 24–26).
  - Policy and governance: evidence for responsible disclosure (e.g., contamination, emissions), model access standards, and best practices for evaluation of APIs/closed models (§6; Appx G; §10.4).
  - Risk assessment: targeted evaluations (e.g., disinformation, memorization) provide templates for auditing domain‑specific harms (§5.4–5.5; §8.4–8.5).

> “By both evaluating these models on the same scenarios and by conducting the evaluation under standardized conditions … we facilitate direct head‑to‑head comparisons” (Fig. 4; §1.1 “Standardization”).

> “Of the 112 possible (core scenario, metric) pairs, we measure 98 (87.5%)” (Table 4).

> “text‑davinci‑002 achieves an accuracy of 62.0% [TruthfulQA] … the next best is 36.2%” (Fig. 37; §1.2 finding 15).

> “OPT (175B) is 79.1% on HellaSwag with separate 0‑shot but 30.2% with joint 5‑shot” (Fig. 33; §8.2).

> “On MS MARCO (TREC), best models reach 65.3% NDCG@10 (boosted) vs BM25 50.6%” (§8.3 IR).

> “All models show higher BPB (worse modeling) on AAE vs. White English [TwitterAAE] (e.g., 2.114 vs 1.506 for OPT‑175B)” (§8.4; §1.2 finding 5).

> “CivilComments: most models near chance; best ~66.8%, with large robustness/fairness drops on Black split” (§8.3).

---

If you plan to use HELM’s results or extend the suite, prioritize (a) aligning scenario–metric pairs with your deployment context, and (b) stress‑testing sensitivity to adaptation choices (prompting formats and multiple‑choice strategies) to avoid misleading model rankings.
