# Holistic Evaluation of Language Models

**ArXiv:** [2211.09110](https://arxiv.org/abs/2211.09110)

## 🎯 Pitch

HELM introduces the first comprehensive, standardized benchmark and analysis framework that evaluates 30 leading language models across 42 scenarios and seven key metrics—including not just accuracy but also calibration, robustness, fairness, bias, toxicity, and efficiency. By making head-to-head, multi-metric comparisons possible under uniform conditions, HELM exposes critical trade-offs and overlooked failure modes that one-dimensional benchmarks miss—paving the way for more responsible, transparent, and effective deployment of language models in real-world applications.

---

## 1. Executive Summary (2–3 sentences)
Holistic Evaluation of Language Models (HELM) is a comprehensive framework and benchmark that evaluates large language models across a broad taxonomy of real-world scenarios and a plural set of metrics beyond accuracy (e.g., calibration, robustness, fairness, bias, toxicity, and efficiency). The paper matters because it replaces fragmented, apples-to-oranges reporting with standardized, multi-metric, and broad-coverage comparisons of 30 prominent models across 42 scenarios, revealing critical trade-offs and failure modes that single-metric leaderboards obscure (Table 4, Figure 4, §1.1–§1.2).

## 2. Context and Motivation
- Problem addressed
  - Prior evaluations of language models are narrow (few tasks, one metric—usually accuracy), inconsistent in setup, and sparse in coverage; research rarely compares the same models under the same conditions (§1.1, Figure 4 top). Before HELM, on average each model had been evaluated on only 17.9% of HELM’s core scenarios and many pairs of top models shared no common benchmarks (§1.1; Figure 4 top; Appendix F).
- Why it matters
  - Language models underpin web search, assistants, moderation, and more. When deployed, they must satisfy multiple desiderata at once—accuracy, calibration, robustness to real-world noise, fairness across groups, low toxicity, and efficiency (cost/latency/carbon). Optimizing for accuracy alone can harm other properties (§1, §4; Figure 24–25).
- Prior approaches and gaps
  - Benchmarks like GLUE/SuperGLUE focus mainly on accuracy. Multi-task suites (e.g., EleutherAI LM Harness, BIG-bench) broaden tasks but still center accuracy and lack standardized adaptation across models (§1.1; Figure 2–3; Table 2–3). Calibration, fairness, and robustness are often relegated to separate bespoke datasets, masking trade-offs.
- How HELM positions itself
  - HELM formalizes a top-down taxonomy over scenarios (task × domain × language) and metrics, then implements a broad but tractable subset with an explicit account of what is missing (§3, §4, §10). It adopts a unified adaptation procedure (few-shot prompting) to enable standardized, head‑to‑head model comparisons (§7; Figure 4 bottom), and evaluates 30 models under identical conditions across 16 “core” scenarios with seven metric categories (Table 4), plus 26 targeted scenarios (§5).

## 3. Technical Approach
HELM is both a framework (taxonomy + methodology) and a concrete benchmark + toolchain.

- The evaluation unit: (scenario, adaptation, metric)
  - A `scenario` is a list of input instances with references (ground truth or acceptable outputs), drawn from a specific task, domain, and language (Figure 8; §2.1, §3).
  - `Adaptation` is how an LM is converted into a system for the scenario; HELM uses uniform few-shot prompting where possible (§2.2, §7). Inputs + in-context examples form a `prompt`, the model emits a `completion`. The evaluation is black-box: only text I/O and (if available) token log-probabilities are used (§2.2).
  - `Metrics` operationalize desiderata beyond accuracy: calibration, robustness, fairness, bias, toxicity, and efficiency (§4).

- Scenario selection and coverage
  - HELM taxonomizes tasks by mapping ACL tracks to canonical tasks (Table 1) and focuses on user‑facing tasks (question answering, information retrieval, summarization, sentiment analysis, toxicity detection, and broad text classification via RAFT) (§3.2–§3.8).
  - Domains and languages are diversified by design (e.g., news/books/dialogue/web; and English varieties like African American English (AAE) and regional Englishes via ICE) (§3.1, §5.1).
  - Core set: 16 scenarios × 7 metric categories with 98/112 possible (scenario, metric) cells filled (87.5%) (Table 4).
  - Targeted set: 26 scenarios probe language, knowledge, reasoning, memorization/copyright, disinformation, bias, and toxicity (§5).

- Unified adaptation (HOW HELM runs models)
  - Few-shot prompting with 5 fixed in‑context examples per scenario (or fewer if context doesn’t fit). Examples are shared across all test instances to reflect realistic few-shot use; this increases variance but avoids “oracle” selection for each instance (§7; §8.2; Figure 31–32).
  - Multiple-choice adaptation is carefully controlled: three methods are studied—`joint` (predict A/B/C from one prompt), `separate` (score each choice separately), and `separate‑calibrated` (separate + calibration by choice prior); the chosen default varies by scenario (§7; §8.2; Figure 33).
  - All models get the same prompts for head-to-head fairness (§7; Figure 4 bottom).

- Metrics (definitions and mechanics)
  - `Accuracy`: task-specific (EM/F1 for QA, ROUGE-2 for summarization, RR@10/NDCG@10 for IR, bits-per-byte for language modeling). See §4.3 and Appendix C.1 for precise formulas.
  - `Calibration`: compares predicted probability to correctness frequency using Expected Calibration Error (ECE; 10-bin by default) and selective classification (accuracy at top-10% confidence; area under coverage-accuracy curve) (§4.4; Figure 17; Appendix C.2).
  - `Robustness`: worst-case performance under natural, semantics-preserving perturbations (e.g., misspellings, case changes, contractions, synonyms) and, when available, equivariance checks via human-authored `contrast sets` that flip the gold answer (§4.5; Figure 18; Appendix C.3, D.1).
  - `Fairness`: (i) `counterfactual fairness` using perturbations that switch demographic markers (e.g., gender terms, AAE/SAE dialect mapping) while keeping content otherwise the same (Figure 19; Appendix D.2); (ii) `performance disparities` on datasets with demographic labels (e.g., CivilComments group splits, ICE regions) (§4.6; Appendix C.4).
  - `Bias` in generation: distributional properties—demographic representation skew and stereotypical associations measured by co-occurrence counts vs. a uniform reference (lower is better). Word lists follow prior literature (Appendix C.5; Figure 20).
  - `Toxicity`: rate of generations labeled toxic by Perspective API at a 0.5 threshold (§4.8; Figure 21; Appendix C.6).
  - `Efficiency`:
    - Training: estimated energy (kWh) and CO2 emissions from reported/estimated hardware-hours and datacenter characteristics (§4.9.1; Appendix C.7).
    - Inference: `denoised runtime` (best-case API time with queuing noise removed) and `idealized runtime` (standardized A100/Megatron stack, apples-to-apples across open models). Captures prompt encoding cost and per-token generation (`F(num_prompt_tokens) + g * num_output_tokens`; Figure 22; §4.9.2).

- Experimental design
  - 30 models from 12 organizations: open (e.g., OPT, BLOOM), limited-access APIs (e.g., `text-davinci-002`), and closed models run by collaborators (e.g., TNLG v2 530B, Anthropic-LM 52B) (Table 5; §6).
  - 4,939 runs; 12.17B tokens; 17.43M queries; ~$38K API cost; ~19,500 GPU-hours for open models (§1.2).
  - Standardization: same scenarios, same prompts, same decoding for all models to enable head‑to‑head comparisons (Figure 4 bottom; §7).

## 4. Key Insights and Innovations
- A. A taxonomy-driven, multi-metric, standardized evaluation protocol (fundamental innovation)
  - HELM first sets a taxonomy (scenarios × metrics), then selects a balanced subset and reports what’s missing (§1.1; Figure 2; §10). This is different from “dataset collections” because coverage and incompleteness are made explicit, trade-offs are exposed (Table 4), and results are comparable across models (Figure 4 bottom).
  - Significance: moves the field from single-number leaderboards to profile-oriented evaluation; reveals interactions such as accuracy–fairness–robustness correlations and calibration trade-offs (Figure 24–25).

- B. Multi-metric dense coverage and standardized adaptation (incremental but impactful)
  - 98/112 core cells evaluated; 96% model–scenario coverage post-HELM (Figure 4), vs. 17.9% pre-HELM. A uniform 5‑shot prompting protocol with controlled multiple-choice adaptations quantifies how prompt design affects conclusions (Table 4; §7; §8.2; Figure 33).
  - Significance: shows that “which prompting template?” can change model rankings (Figure 33), emphasizing the need for interoperable prompting.

- C. Efficiency measures that separate “deployment noise” from model-intrinsic costs (new capability)
  - Introduction of `denoised` vs. `idealized` inference time (Figure 22; §4.9.2), plus estimated training energy/CO2 (Appendix C.7), allows capability–efficiency trade-off analysis beyond raw throughput numbers.
  - Significance: reveals that no simple accuracy–efficiency frontier exists across families (Figure 24 bottom-right), so efficiency cannot be inferred from accuracy alone.

- D. Targeted probes of societal risks and scientific capabilities (novel combinations and breadth)
  - Copyright/memorization with long-span regurgitation measures (LCS and edit similarity) on Books and Linux kernel (Figure 39; §5.4).
  - Disinformation evaluated with human studies on narrative reiteration and wedging (Table 8; §5.5, §8.5.1).
  - Reasoning suite spanning abstract symbol matching, Dyck languages, math word problems, code tasks, and legal reasoning (Figure 38; §5.3).
  - Significance: connects broad “core” results to deeper analyses that explain why errors happen and where models are potentially harmful or beneficial.

## 5. Experimental Analysis
- Evaluation methodology
  - Core tasks and datasets:
    - QA: NaturalQuestions (open/closed-book), NarrativeQA, QuAC, BoolQ, HellaSwag, OpenBookQA, TruthfulQA, MMLU (§3.3).
    - IR: MS MARCO (regular, TREC); ranking via pointwise Yes/No scoring and NDCG/RR metrics (§3.4; Figure 12; Appendix C.1.2–C.1.3).
    - Summarization: CNN/DailyMail, XSUM; quality (ROUGE-2), faithfulness (SummaC, QAFactEval), extractiveness (coverage/density) (§3.5).
    - Sentiment: IMDB (with contrast sets for robustness) (§3.6).
    - Toxicity detection: CivilComments with group splits from WILDS for disparities (§3.7).
    - RAFT: 11 real-world classification tasks (banking, legal, clinical, etc.) (§3.8).
  - Metrics as defined in §4 (Table 4 shows complete matrix).

- Main quantitative results
  - Head-to-head winners (Figure 26):
    - Accuracy: `text-davinci-002` wins >90% of pairwise comparisons across core scenarios; `TNLG v2 (530B)` next; `Anthropic-LM v4-s3 (52B)` competitive despite being 10× smaller than TNLG v2.
    - Robustness and Fairness: same top tier (`text-davinci-002` ≈ `Anthropic-LM` ≈ `TNLG v2`).
    - Calibration: smaller `text-ada-001` and older OpenAI variants do better on ECE than the most accurate models—showing a notable accuracy–calibration tension (Figure 24 top-left; Figure 26).
    - Bias and Toxicity in generation: differences are smaller and sometimes inverted—e.g., `T0++ (11B)` is among the least gender-biased but most toxic; `davinci (175B)` more biased but less toxic (Figure 26 bias/toxicity panes).
  - Trends over time and by access (Figures 27–28):
    - SOTA accuracy jumps with GPT-3 and improves again with Anthropic LM and instruction tuning; overall improvements vary by scenario (Figure 27).
    - Access gap: on many core QA tasks, the best limited-access model outperforms the best open model (bars in Figure 28). On MMLU and closed-book QA, non-open models have a clearer edge.
  - Metric inter-relationships (Figures 24–25):
    - Strong positive correlations among accuracy, robustness, and fairness across scenarios; the most accurate models tend to also be more robust/fair. Calibration often conflicts: moving to more accurate systems increases ECE on some datasets (HellaSwag) but not others (OpenBookQA) (§1.2 bullet 3; Figure 24–25).
  - Sensitivity to prompting (ablation studies)
    - Seeds/in-context examples: even with fixed 5-shot examples, swapping examples changes scores; NaturalQuestions (open-book) shows median performance range ~0.17 F1 across seeds (Figure 31).
    - Number of shots: big gains from 0→1 shot, smaller beyond that; exceptions exist (OPT‑175B keeps improving monotonically; CNN/DM often worse at 1-shot than 0-shot) (Figure 32).
    - Multiple choice adaptation: scenario-dependent. For HellaSwag, `separate` > `separate‑calibrated` > `joint`. For OpenBookQA, TruthfulQA, MMLU, `separate‑calibrated` usually best—except `Anthropic-LM` prefers `joint`, flipping rankings (Figure 33). This shows that “one adaptation to rule them all” is untenable and can change conclusions.
  - Task-specific highlights
    - QA: `text-davinci-002` tops all nine QA scenarios; gaps vary—from +26.6 points on TruthfulQA (62.0% vs. 35.4% next best; §1.2 bullet 15) to near ties on NQ closed-book (38.9% vs. 38.5%; §8.3).
    - IR: `text-davinci-002` achieves 39.8% RR@10 boosted on MS MARCO (regular) and 65.3% NDCG@10 boosted on TREC—better than BM25 and comparable to older neural rankers, but behind specialized SOTA retrievers; re‑ranking costs scale poorly unless parallelized (§8.3; MS MARCO section).
    - Summarization: automatic metrics struggle to discriminate model quality on CNN/DM and XSUM; faithfulness/length control is challenging; toxicity rates remain very low (§8.3).
    - Sentiment (IMDB): many models >90% EM; GLM (130B) highest at 95.5%; contrast sets reveal larger robustness drops (e.g., -8% for GLM) than simple perturbations (Figure 35; §8.3).
    - Toxicity detection (CivilComments): most models barely above chance; `text-davinci-002` 66.8% best; robustness/fairness perturbations slash accuracy (e.g., TNLG v2 drops to 40.9% under robustness; OPT 175B drops to 8.8% robust accuracy on Black split vs. 24.3% on White) (§8.3).
    - RAFT: heterogeneous; `GLM (130B)` top overall (85.8%), while `text-davinci-002` underperforms on some subsets (40.8% on SystematicReview) but is consistently strong elsewhere (§8.3).
  - Targeted evaluations
    - Language modeling: “Pile‑trained” models (GPT‑J/NeoX/OPT/BLOOM) do best on The Pile and transfer better to ICE/TwitterAAE BPB than instruction‑tuned models; all models perform worse on AAE vs. White English across TwitterAAE (e.g., OPT‑175B BPB 2.114 AAE vs. 1.506 White; §8.4; Figure 36).
    - Knowledge: `text-davinci-002` leads across knowledge QA; larger TNLG v2 (530B) shines on knowledge-heavy closed-book QA and WikiFact (e.g., 38.5% vs. 34.3%; §8.4; Figure 37).
    - Reasoning: code models dominate; `code‑davinci‑002` scores 52.1% on GSM8K vs. next best `text‑davinci‑002` 35.0%; also best on HumanEval/APPS and Dyck (80.2%) (Figure 38; §8.4).
    - Copyright/memorization: rare but significant long‑span verbatim regurgitation for popular books; higher accuracy models and code‑specialized models exhibit more regurgitation risk on respective corpora (Figure 39; §8.4).
    - Disinformation (human eval): for narrative reiteration, `text‑davinci‑002` and `Anthropic‑LM` produce headlines rated both high‑quality and style‑faithful (Table 8). For wedging, quality is mixed; style largely acceptable except for GLM; hostility mostly covert/subtle rather than overt (Table 8; §8.5.1).
    - Bias (BBQ): `text‑davinci‑002` hits 89.5% EM—far above others—but also shows the strongest bias in ambiguous contexts, while many lower-accuracy models show reversed bias (Figure 40; §8.4).

- Do the experiments support the claims?
  - Yes: the breadth of scenarios + multi‑metric reporting + ablations substantiate claims about trade-offs (accuracy vs. calibration), the importance of adaptation, accessibility gaps, and the correlation between accuracy, robustness, and fairness (Figures 24–26, 28, 31–33).
  - Caveat: where live APIs change over time or training data are undisclosed, repeatability and contamination remain concerns (§6; Appendix G).

## 6. Limitations and Trade-offs
- Assumptions and scope
  - Black‑box evaluation: HELM does not require internal activations or training data access (§2.2), which makes it widely applicable but prevents deeper interpretability or retraining-based fixes.
  - Single adaptation by default: 5-shot prompting as the primary adaptation method can disadvantage models not optimized for prompting (e.g., T5) (§6, §7).
- Coverage gaps
  - Language coverage is largely English, with only targeted coverage of English varieties; many important domains (biomedical, finance, education, customer service) and interaction paradigms (dialogue, tool use) are out of scope for now (§10.1).
  - Metric gaps: user experience, interpretability, provenance, privacy, and safety in multi-modal/robotic contexts are not comprehensively measured (§10.2).
- Measurement limitations
  - Robustness/fairness via perturbations may not capture all real-world shifts; contrast sets are only available for some datasets (§4.5–§4.6).
  - Bias/toxicity detection relies on Perspective API; known biases can affect both false positives and false negatives (§4.8; §5.7).
  - Summarization metrics (ROUGE, reference-free faithfulness) are imperfect proxies and sometimes fail to reflect quality differences (§3.5; §8.3).
- Computational and access constraints
  - Some API models are “live systems” without versioned checkpoints; models changed during the evaluation window may confound comparisons (§6).
  - Efficiency estimates rely on reported or approximated hardware usage; CO2 and PUE estimates are approximate (§4.9.1; Appendix C.7).
- Contamination and generalizability
  - Training data are often undisclosed; known contamination exists (e.g., The Pile for many models; Brown et al. datasets for OpenAI variants), and few-shot evaluation magnifies contamination risks (Appendix G; Table 13).
  - Prompting sensitivity (Figures 31–33) shows that small template changes can significantly alter outcomes; rankings are not absolute.

## 7. Implications and Future Directions
- How this work changes the landscape
  - HELM establishes a new norm: models should be reported as multi-metric profiles across standardized scenarios, not single-number ranks. It exposes hidden trade-offs (accuracy–calibration), confirms some intuitions (accuracy–fairness–robustness correlations), and cautions that evaluation conditions (prompting and adaptation) can flip conclusions (Figure 33).
- Follow-up research enabled/suggested
  - Adaptation research: robust, interoperable prompting and better per-family adaptation recipes; programmatic “prompt contracts” for fair cross-model use (§8.2).
  - Metric advances: culturally-aware toxicity/bias detectors, better summarization faithfulness measures, user-centered UX metrics, and privacy/memorization tests with formal guarantees (§10.2–§10.3).
  - Scenario expansion: more non-English languages and dialects, domain-heavy tasks (biomed/finance/law), interactive and tool-augmented settings, and evaluation of time-awareness and updating (§10.1).
  - Efficiency reporting standards: unified, versioned disclosures for training/inference footprints; standardized “idealized runtime” reporting akin to FLOPs in model training (§4.9; Appendix C.7).
- Practical applications and downstream use
  - Procurement and risk assessment: organizations can use HELM profiles to select models that meet multi-criterion thresholds (e.g., high fairness + adequate accuracy + low toxicity).
  - Product tuning: insight that instruction-tuned models (`text-davinci-002`, `Anthropic-LM`) excel on many user-facing tasks (§1.2 bullets 1, 8, 15–16) and that code-tuned models excel at reasoning (§1.2 bullet 16) can guide model choice.
  - Safety and governance: disinformation, memorization, and demographic disparities analyses ground red-teaming and compliance audits (Table 8; Figure 39–40; §5.4–§5.6).

> “We improve [model coverage] to 96.0%… now all 30 models have been densely benchmarked on a set of core scenarios and metrics under standardized conditions.” (Figure 4, §1.1)

> “Of the 112 [core scenario, metric] pairs, we measure 98 (87.5%).” (Table 4, §1.1)

> “text‑davinci‑002 achieves 62.0% on TruthfulQA vs. 36.2% next best.” (§1.2 bullet 15; Figure 37)

> “On NarrativeQA, TNLG v2 (530B) drops from 72.6% to 38.9% under robustness perturbations.” (§1.2 bullet 4)

> “OPT (175B): 51.3% → 8.8% robust accuracy on CivilComments (Black split).” (§1.2 bullet 12)

> “code‑davinci‑002: 52.1% on GSM8K; next best 35.0%.” (§1.2 bullet 16; Figure 38)

> “HellaSwag accuracy for OPT‑175B: 79.1% (separate) vs. 30.2% (joint).” (§1.2 bullet 23; Figure 33)

> “TwitterAAE BPB: OPT‑175B 1.506 (White) vs. 2.114 (AAE). Lower is better.” (§1.2 bullet 5; Figure 36)

> “Narrative reiteration: text‑davinci‑002 and Anthropic‑LM produce high-quality, headline-like generations.” (Table 8; §8.5.1)

Overall, HELM delivers a practical, extensible blueprint for evaluating language models “in their totality,” providing both a toolkit and a set of findings that can guide model development, selection, and governance.
