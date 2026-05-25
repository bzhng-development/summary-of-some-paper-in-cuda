# DEMOCRATIZING OPEN AND COMPLIANT LLMS FOR GLOBAL LANGUAGE ENVIRONMENTS APERTUS V1 TECHNICAL REPORT

**ArXiv:** [2509.14233](https://arxiv.org/abs/2509.14233)

## 🎯 Pitch

Apertus sets a new standard for transparency and accountability in large language models by releasing a fully open, end-to-end LLM suite—at both 8B and 70B scales—that combines rigorous data compliance (including retroactive robots.txt enforcement and PII/toxicity filtering), robust memorization mitigation, and truly global multilingual coverage (1,811 languages). By making not only weights but also all training data recipes, filtering code, and evaluation tools open and auditable, Apertus empowers researchers and developers worldwide to build LLMs that are both lawful and inclusive—addressing regulatory demands and opening access to high-quality AI for diverse language communities.

---

## 1. Executive Summary (2–3 sentences)
Apertus is a fully open suite of large language models (LLMs) released at 8B and 70B parameters that tackles two persistent gaps in the open-model ecosystem: lawful, auditable training data and strong multilingual coverage. It combines a compliance-first data pipeline (retroactive robots.txt opt‑outs, PII and toxicity filtering), a memorization‑mitigating pretraining objective (“Goldfish loss”), and broad multilingual pretraining/post‑training (1,811 languages in pretraining; 149 in post‑training), while releasing all artifacts—weights, data preparation code, checkpoints, and evaluation suites—for full reproducibility (Section 1; Section 3; Section 4).

## 2. Context and Motivation
- Problem/gap addressed
  - “Open” models often mean open weights only, without transparent data pipelines; many include material that content owners forbid for AI training, and few models prioritize non‑English users (Section 1).
  - LLMs can memorize and regurgitate training text, carrying privacy and copyright risks; most demonstrations of mitigation are small‑scale (Section 1; Section 5.4).
- Why this matters
  - Regulatory compliance (e.g., EU AI Act–style provisions) requires traceable, lawful data use and provable risk mitigation (Section 1, “Data Compliance”).
  - Many communities operate in low‑ or mid‑resource languages; models that underperform outside English exclude these users (Section 1, “Multilinguality”).
- Shortcomings of prior approaches
  - Open‑weight releases typically do not publish data recipes or legal filtering, making audits impossible (Section 1).
  - Memorization defenses are often post‑hoc (e.g., safety tuning, constrained decoding) and reversible via fine‑tuning or prompt attacks (Appendix F, “Limitations of post‑hoc…”).
  - Multilingual efforts exist but usually cover far fewer languages and/or devote a small fraction of tokens to non‑English (Section 1; footnote 2).
- Positioning
  - Apertus frames itself as a “fully open” alternative: it releases weights plus scripts, checkpoints, and evaluation harnesses; it enforces retroactive consent and targeted filtering; and it expands multilingual coverage and post‑training alignment to 149 languages (Sections 1, 3, 4).
  - The 70B model is trained on 15T tokens at production scale while remaining fully auditable, which is rare among fully open efforts (Section 1, “Scale”; Figure 11; Section 6).

## 3. Technical Approach
This section explains how Apertus is built and trained, from architecture to data, training objectives, and alignment.

- Model architecture (Section 2.1; Table 1)
  - Dense decoder‑only Transformers at two scales: `Apertus‑8B` (32 layers) and `Apertus‑70B` (80 layers), both using grouped‑query attention (GQA) for inference efficiency and rotary positional embeddings (RoPE) with NTK‑aware scaling for long‑context extension.
  - Two stabilizing components:
    - `QK‑Norm`: normalizes queries and keys in attention to control logit magnitude spikes (Section 2.1).
    - `xIELU` activation: a modified activation with trainable positive/negative branches; on the positive side it behaves like a smooth square‑root growth, on the negative side like a corrected ELU (Equation in Section 2.1). This reduces outliers while retaining expressivity.
  - Input/output embeddings are untied; documents are bracketed with begin/end tokens and attention across document boundaries is masked (Section 2.1).

- Tokenizer selection (Section 2.2; Figure 1)
  - Uses the `Mistral‑Nemo v3 tekken` byte‑BPE (131k vocab), chosen via four intrinsic metrics on FLORES+ (55 languages): fertility, compression ratio, vocabulary utilization, and a cross‑language fairness measure (Gini coefficient). It offers competitive compression and lower inequity across languages than the alternatives compared (Figure 1).

- Training recipe (Section 2.3; Table 2)
  - Objective: `Goldfish loss` (define). Instead of training on every token, a small, deterministic subset of tokens per sequence (2%) is used to compute the loss, with a hash computed over the preceding 50 tokens to ensure the mask is reproducible but input‑dependent (Algorithm 1; Section 2.3; Appendix F). Intuition: by breaking the tight coupling between every context‑token pair and its next token, the model is less likely to recall long exact spans verbatim while still learning general patterns.
  - Optimizer: `AdEMAMix` (define). An Adam‑style method that keeps an extra long‑term exponential moving average (“slow momentum”) to better leverage old gradients during long training; warm‑ups are used for the additional terms (Section 2.3; Appendix C).
  - Learning rate schedule: `Warmup‑Stable‑Decay (WSD)` with a 1‑sqrt cooldown tail, enabling continued training without re‑warming and safer late‑stage convergence (Section 2.3).
  - Batch size doubled midway without changing LR, using the WSD plateau (Table 2; Figure 3), to improve hardware efficiency late in training.

- Data pipeline with compliance (Section 3)
  - Retroactive robots.txt (“with hindsight”): crawl permissions as of Jan‑2025 are applied to all historical snapshots; if a site blocks major AI bots, its content is removed from 2013–2024 data (Appendix B; Table B.1 shows token reductions ≈8% in English, ≈4% multilingual; Tables B.2–B.3 list blocked bots and volumes).
  - PII removal via regex for emails, IPs, and IBAN; multilingual toxicity filtering with XLM‑R encoders + language‑specific MLPs, removing the top 5% toxic documents per language for nine languages (Section 3.1.2–3.1.3; Figure 4 shows score distributions and thresholds).
  - Pretraining mixture:
    - `FineWeb‑2` across 1,811 languages as the base multilingual source; English high‑quality slices from `FineWeb‑HQ` and `FineWeb‑Edu`; code from `StarCoderData`; math from `FineMath` and `MegaMath`; and parallel corpora (`EuroParl`, `ParaDocs`) for translation (Section 3.2; Figure 5).
    - Curriculum in five stages to gradually raise quality and increase math/code proportions (Section 3.3; Table 6). Stage choices were validated via cooldown experiments on smaller checkpoints (Table 7).
  - Long‑context extension to 65,536 tokens:
    - RoPE base θ increased across 8k→16k→32k→64k phases; context parallelism used for memory scaling; data mixture enriched with “FineWeb‑Long” (documents >4k) and `Institutional Books 1.0` (post‑1900, OCR‑cleaned) (Section 2.5; Section 3.4; Table 5; Table 8).

- Post‑training for instruction following and alignment (Section 4)
  - Supervised fine‑tuning (SFT): ≈4.18M examples across general instructions, math, code, and multilingual/conversational data (149 languages), after license filtering and decontamination (Section 4.1; Table 12). Romansh—six idioms—receives dedicated coverage (Appendix J.1).
  - Preference alignment via `QRPO` (define). A direct‑alignment algorithm that optimizes absolute rewards using quantile ranks of completions sampled from a reference model (Section 4.3). Rewards come from (1) a pretrained reward model (`Skywork‑Reward‑V2`, Section 4.3.1) for standard topics and (2) an LLM‑as‑judge that scores adherence to the “Swiss AI Charter” for ideologically sensitive prompts (Section 4.3.2; Appendix O).
    - Length‑normalized QRPO is used (divide the KL regularizer by completion length), improving stability (Section 4.3).

- Infrastructure & engineering (Section 6)
  - Trained on up to 4,096 NVIDIA GH200 GPUs at CSCS with a vCluster setup enabling container‑first ML workloads and robust node vetting (Sections 6.1–6.3).
  - Throughput and stability were improved through systems fixes (driver/kernel patches, storage and NCCL/libfabric alignment), checkpointing strategy (Young/Daly‑guided), and distributed training tweaks (Figure 12; Section 6.3).
  - Estimated 6.74×10^24 FLOPs to train the 70B on 15T tokens; ≈6M GPU‑hours consumed (Section 6.2; Appendix E code for FLOPs).

## 4. Key Insights and Innovations
1) Compliance you can audit end‑to‑end (Section 3; Appendix B)
- What’s new: retroactive application of robots.txt opt‑outs (“with hindsight”) to all historical snapshots—data from sites blocking AI crawlers in Jan‑2025 is removed across the entire 2013–2024 period (Appendix B).
- Why it matters: enables legal and ethical reuse for downstream models and provides a clear audit trail; token impact is quantified (Table B.1).
- Distinct from prior work: most open‑weight releases neither document nor implement such retroactive consent enforcement.

2) Memorization mitigation at scale with `Goldfish loss` (Sections 2.3, 5.4; Appendix F)
- What’s new: a 70B model trained on 15T tokens with a loss that masks ~2% of tokens deterministically based on recent context, suppressing verbatim recall even after up to 128 exposures (Figure 8; Table 25).
- Why it matters: reduces copyright/privacy risks without sacrificing performance (Appendix F, Table F.5 shows downstream parity; Section 5.4 shows low Rouge‑L/LCCS memory signals).
- Caveat: failure modes on ubiquitous texts that exist as many near‑duplicates (e.g., Shakespeare, US Constitution) due to hash fragility to formatting/tokenization changes (Section 5.4.2; Figure 9).

3) Multilingual breadth and targeted post‑training (Sections 3.2, 4.1)
- What’s new: pretraining spans 1,811 languages; post‑training covers 149 languages, including Swiss Romansh idioms (Appendix J.1) and extensive conversational data (Table 12).
- Why it matters: Apertus performs strongly on multilingual cultural/knowledge benchmarks relative to fully open peers (Tables 15, 20) and achieves better Romansh↔German translation than `Llama‑3.3‑70B‑Instruct` (Table 24).

4) QRPO + Constitutional judging for sensitive topics (Section 4.3.2; Appendix O)
- What’s new: instead of a single global reward model, the alignment uses an LLM judge prompted with the “Swiss AI Charter”—11 articles distilled from Swiss constitutional and civic values. A public survey shows high approval of these principles (Table 13).
- Why it matters: makes value‑laden alignment explicit, inspectable, and adaptable to a cultural context; integrates naturally with QRPO’s absolute‑reward training.

5) Transparent, reproducible scaling recipe (Sections 2.4–2.6; 6)
- What’s new: ablations show `xIELU + AdEMAMix + QK‑Norm + WSD + Goldfish` lowers loss and gradient volatility; a re‑run over OLMo2 data achieves similar loss with 30–46% fewer tokens in the first 20k steps (Table 4). Full training logs and checkpoints are released.
- Why it matters: provides a tested, efficient training blueprint for future fully open models.

## 5. Experimental Analysis
- Evaluation setup
  - Pretraining evaluation uses the lm‑evaluation‑harness in probabilistic mode (log‑likelihood) for sensitivity during early training (Section 5.1), covering general understanding (ARC, HellaSwag, WinoGrande, XNLI, PIQA/XCOPA) and factual knowledge (MMLU, Global‑MMLU, INCLUDE v1/v2, CulturalBench, BLEnD, SwitzerlandQA) (Tables 14–15).
  - Post‑training evaluation uses open generation with the same harness, spanning knowledge (MMLU, Global‑MMLU, TruthfulQA), instruction following (IFEval, Multi‑IFEval), reasoning (BBH, DROP, ACPBench, GPQA, MLogiQA, MGSM), coding (HumanEval, MBPP), math (GSM8K, GSM8K‑Platinum, Hendrycks’ Math, MathQA), cultural knowledge, and long‑context (RULER) (Section 5.2; Tables 17–21, 23).
  - Memorization measured by Rouge‑L and normalized longest common contiguous substring (LCCS) on injected Gutenberg probes across exposure frequencies and offsets; Type–Token Ratio (TTR) used as a degeneracy and filtering signal (Section 5.4; Figures 8–10; Table 25).
  - Safety assessed with BBQ (bias), HarmBench (harmful behavior elicitation), RealToxicityPrompts (subsamped with Llama‑Guard‑3 classifier), and ToxiGen (implicit toxicity detection) (Section 5.5; Table 26); multilingual safety examined with LinguaSafe (Tables 27–28).

- Main quantitative results
  - Pretraining capability (Tables 14–15; Figure 7)
    > `Apertus‑70B` achieves 67.5% macro on general language understanding (Table 14), leading fully open models and matching or surpassing several open‑weight peers at comparable scale on some tasks (e.g., XCOPA 45.3%).  
    > On factual knowledge, `Apertus‑70B` scores 58.9% macro; it is strong on INCLUDE (57.0% v1; 38.5% CulturalBench) and SwitzerlandQA (60.2%), outperforming fully open baselines like EuroLLM‑9B (58.1% SwitzerlandQA) and OLMo2‑7B (52.5%) (Table 15).
  - Post‑training (Tables 17–21)
    - Knowledge & commonsense: `Apertus‑70B‑Instruct` achieves 63.4% macro across knowledge tasks, with 69.6% on MMLU and 78.1% on HellaSwag; this trails top open‑weight models (`Llama‑3.3‑70B‑Instruct` 68.4% macro, MMLU 87.5%; Table 17) but is competitive with fully open baselines.
    - Coding & math: Results are mixed. `Apertus‑70B‑Instruct` scores 73.0% pass@10 on HumanEval and 77.6% on GSM8K, but its Hendrycks’ Math score (30.8%) lags models that likely used RL with verifiers (Table 18).
    - Reasoning & instruction following: `Apertus‑70B‑Instruct` reaches 61.8% macro across BBH/DROP/ACP/IFEval, solid but behind the best open‑weight systems (`Qwen3‑32B` 80.8% macro) (Table 19).
    - Cultural knowledge: Both Apertus‑Instruct models are strong among fully open models, with `Apertus‑70B‑Instruct` scoring 61.5% macro; SwitzerlandQA 67.2% (Table 20).
    - Held‑out tests: `Apertus‑70B‑Instruct` achieves 51.4% macro across AGIeval, ARC‑Challenge Chat/Multilingual, GPQA, GSM8K‑Platinum, and MLogiQA; `OLMo‑2‑32B‑Instruct` is higher at 58.3% (Table 21).
  - Long context (Table 23)
    > `Apertus‑70B‑Instruct` scores 94.8/89.9/85.7/81.9 on RULER at 4k/8k/16k/32k contexts; evaluation at 64k was runtime‑limited. Scores are competitive but below `Llama‑3.3‑70B‑Instruct` (95.2/94.7/94.8/93.7).
  - Low‑resource translation (Table 24)
    > On WMT24++ Romansh↔German, `Apertus‑70B‑Instruct` beats `Llama‑3.3‑70B‑Instruct` in all six Romansh variants in both directions (e.g., Rumantsch Grischun DE→RM: 27.8 vs 21.6 BLEU).
  - Memorization (Section 5.4; Figures 8–10; Table 25)
    - Across 1–128 exposures and 50–5,000‑token prefixes, Rouge‑L stays ≈0.17–0.19 (baseline level), showing no scalable verbatim recall under greedy or nucleus sampling; TTR remains high under nucleus sampling (≈0.50), confirming mitigation is not an artifact of degeneration (Table 25; Figure 8).
    - Failure mode: near‑duplicate canonical texts across the web can escape masking alignment and show higher recall (Figure 9); low‑diversity templates (tables, lists) yield high Rouge‑L without privacy/copyright risk (Figure 10).
  - Safety (Table 26; Tables 27–28)
    - RealToxicityPrompts (Llama‑Guard‑3 subsample): very low average toxicity score (0.2), competitive with open‑weight models.
    - HarmBench: higher harm rates than the very best models, especially under human jailbreaks (e.g., 36.2 for `Apertus‑70B‑Instruct` vs 10.1 for `Qwen2.5‑72B‑Instruct`), indicating room for stronger guardrails.
    - BBQ/ToxiGen: mid‑tier performance; multilingual safety (LinguaSafe) shows non‑trivial harm scores, highlighting inherent difficulty across languages.

- Ablations and robustness (Section 2.4; Table 3; Figure 2)
  - On a 1.5B/3B setting, each design element improves stability or loss: `AdEMAMix` and `xIELU` provide the largest single‑changes; the combined recipe matches baseline loss with 30–40% fewer tokens (Figure 2; Table 3).
  - Replicating OLMo2’s early training with identical data, the Apertus recipe achieves similar loss with 30–46% fewer tokens (Table 4).

- Do results support the claims?
  - Yes on the core claims: demonstrable compliance pipeline; memorization mitigation at 70B scale; strong multilingual outcomes and Romansh translation; full transparency of artifacts.
  - Performance is competitive but not state‑of‑the‑art on math/reasoning/coding compared to top open‑weight models that apply heavier reinforcement learning and verifier pipelines (Tables 18–19), which the paper explicitly lists as future work (Section 7).

## 6. Limitations and Trade‑offs
- Compliance scope and coverage
  - Robots.txt retroactivity enforces consent, but legality can involve more than crawler directives (licenses, database rights); toxicity filtering covers only nine languages during pretraining (Section 3.1.3).
  - Post‑training license filtering and decontamination measurably reduce benchmark scores in some settings (e.g., MMLU CoT: 0.513→0.253 when license‑filtering Tulu3; Table 10), illustrating a real compliance‑vs‑capability trade‑off.
- Memorization defense boundaries
  - Goldfish loss can miss near‑duplicates because hash decisions differ with minor formatting/tokenization changes (Section 5.4.2). High‑frequency canonical texts remain a risk area.
- Capability trade‑offs
  - Math and coding lag behind leaders that applied RL with verifiers (Table 18); instruction‑following and reasoning are solid but not best‑in‑class (Tables 17–19).
- Computational cost and practicality
  - ≈6M GPU‑hours; ≈5 GWh estimated energy on 4,096 GH200s over ~90 days for a full run (Section 6.2). While Alps is hydro‑powered, not every lab can replicate this.
- Safety guardrails
  - HarmBench shows notable vulnerability to jailbreaks (Table 26); multilingual safety remains unconquered (Tables 27–28). Paper acknowledges that jailbreak resistance cannot be guaranteed for open weights and should be handled in deployment (Section 5.5.1).

## 7. Implications and Future Directions
- Field impact
  - Sets a new bar for “fully open” releases: transparent, auditable data/process; memorization risk quantified and reduced at 70B scale; and serious multilingual commitment. This makes Apertus a practical baseline for regulatory‑grade AI development and multilingual research.
- Enabled research
  - Data governance science: with released scripts and filters, the community can measure how specific data slices and compliance steps affect capability, fairness, and memorization (Section 7, “Data‑to‑performance mapping”).
  - Memorization & privacy: the Goldfish framework plus FM‑Probes (Appendix F; Section 3.2.4) invites deeper tests, especially on near‑duplicate robustness and alternative masking designs.
  - Alignment methods: QRPO with constitutional judges opens a pathway to culturally scoped alignment objectives and human‑in‑the‑loop evaluations (Section 4.3.2; Appendix O).
- Practical applications
  - Public‑sector and enterprise deployments requiring data provenance and auditability; multilingual assistants in government, education, and healthcare; localization for under‑served languages (Romansh results, Table 24).
- Near‑term directions proposed in the paper (Section 7)
  - Scaling (larger or longer‑context models), distillation for deployment efficiency, RL with verifiers for math/code (`RLVR`), adaptive test‑time compute, multimodal extensions with the same compliance standards, broader preference elicitation in Swiss/multilingual populations, and field evaluations with professionals.

> “Apertus‑70B is trained on 15T tokens with retroactive robots.txt opt‑outs, toxicity/PII filtering, and Goldfish loss; performance is strong on multilingual knowledge and competitive overall, while memorization remains at baseline across exposures.” (Sections 1–5; Tables 14–15, 20; Figures 8–10)

> “All artifacts—training code, data preparation, checkpoints, evaluation harness—are released for audit and extension.” (Section 1, “Transparency”)
