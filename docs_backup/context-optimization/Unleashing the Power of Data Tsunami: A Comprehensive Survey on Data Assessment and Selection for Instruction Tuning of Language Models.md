# Unleashing the Power of Data Tsunami: A Comprehensive Survey on Data Assessment and Selection for Instruction Tuning of Language Models

**ArXiv:** [2408.02085](https://arxiv.org/abs/2408.02085)
**Authors:** Yulei Qin, Yuncheng Yang, Pengcheng Guo, Gang Li, Hang Shao, Yuchen Shi, Zihan Xu, Yun Gu, Ke Li, Xing Sun
**Institutions:** 

## 🎯 Pitch

This comprehensive survey introduces a unified framework for optimizing data used in instruction tuning for large language models, focusing on the triad of quality, diversity, and importance. By operationalizing data selection through detailed algorithms and comparative analyses, it enables more efficient tuning pipelines that reduce computational costs while maintaining or enhancing model performance, offering a practical guide for practitioners seeking to align models with human preferences cost-effectively.

---

## 1. Executive Summary (2-3 sentences)
This survey organizes the exploding literature on “which data to use” for instruction tuning large language models (LLMs) into a unified, actionable framework. It reframes data assessment and selection around three complementary dimensions—quality, diversity, and importance—and shows how to operationalize each with concrete scoring functions, sampling mechanisms, and empirical evidence (Figures 1–2, Sections 3–5).

## 2. Context and Motivation
- Problem addressed
  - Instruction tuning (fine-tuning LLMs on instruction–response pairs) is essential to align models with human preferences, but simply training on “everything available” is costly and often suboptimal. The paper asks: how do we assess and select the most useful subset of instruction data?
  - There is no consistent definition or measurement of “good data” for instruction tuning, and prior surveys either focus on pretraining pipelines or list datasets without offering method-level guidance tailored to instruction tuning (Section 1.1).

- Why it matters
  - Real-world: Smaller, better-curated instruction sets can deliver equal or better performance with less compute, lower cost, and fewer risks from noise or duplication (Abstract; Sections 1.2 and 6).
  - Theoretical: It clarifies how evaluation functions and selection processes relate to the probabilistic view of learning—data distributions determine model behavior (Section 1).

- What existed before and where they fall short
  - General coreset or data selection surveys classify methods broadly (e.g., geometry-, uncertainty-, error-, gradient- or bilevel-based) but don’t explain how to adapt them to instruction tuning’s specifics (Section 1.1; Guo et al. 2022).
  - LLM-focused data surveys list corpora and stats but give limited guidance on building tuned subsets for downstream alignment (Section 1.1; Liu et al. 2024d).
  - There is no unified vocabulary or taxonomy for instruction-tuning-specific “quality” vs “diversity” vs “importance” (Section 1.2).

- How this paper positions itself
  - It provides:
    - A unified mathematical lens for assessment and selection under a budget (Section 2: Eqs. (2)–(4)).
    - A fine-grained taxonomy mapping existing techniques into three pillars—quality (Section 3), diversity (Section 4), and importance (Section 5)—with detailed mechanisms, formulas, and algorithms (Figures 1–2).
    - Comparative evidence compiled from reported results (Tables 2–4), plus an analysis of open challenges (Section 7).

## 3. Technical Approach
The paper builds a step-by-step framework that turns raw instruction data into a scored, selected subset tailored for instruction tuning. It formalizes both the data processing and the selection logic.

- Data preparation and loss (Section 2; Figure 3; Eq. (1))
  - A raw item contains an instruction, optional input, and a reference response.
  - “Template wrapping” inserts special tokens to form a chat-like prompt; “tokenization” converts text to token IDs (Figure 3).
  - The token sequence `x_i` is split at index `t` into instruction context `x_i(<t)` and response `x_i(≥t)`. Supervised tuning minimizes the standard autoregressive cross-entropy on the response tokens.

- The selection problem (Section 2; Eqs. (2)–(4))
  - Goal: choose a subset `S_b ⊂ S` of size ≤ budget `b`.
  - Two key components:
    - An evaluation function `q(x_i)` that assigns each datapoint a scalar utility.
    - A selection mechanism `π` that uses `q` to pick items, e.g. greedy (maximize total score; Eq. (3)) or probabilistic sampling (normalize scores to probabilities; Eq. (4)).

- Unifying the three pillars with explicit formulations
  - Quality (Section 3; Eq. (5)):
    - Decomposes into instruction quality `q_I` (clarity, accuracy, explicitness) and response quality `q_R` (correctness, coherence, pertinence), then aggregates with `f_q`.
  - Diversity (Section 4; Eq. (23)):
    - Combines lexical (`q_L`, e.g., n-gram variety) and semantic (`q_S`, e.g., embedding-space dispersion) measures with `f_d`.
  - Importance (Section 5; Eq. (46)):
    - Mixes data complexity (`q_C`), contribution to performance (`q_P`, via losses/errors), and gradient influence (`q_G`), aggregated by `f_i`.

- How each pillar is operationalized (Sections 3–5)
  - Quality (Section 3)
    - Hand-crafted indicators (Section 3.1): Linguistic/readability features (e.g., Type–Token Ratio; error/duplication heuristics), percentile or threshold filtering (Eqs. (7)–(8)).
    - Model-based indicators (Section 3.2):
      - Perplexity (Eq. (10)); EL2N error norm (Eq. (13)); memorization rank (Eq. (14)).
      - Reward scores (Eq. (12)).
      - “Instruction-Following Difficulty” IFD (Eq. (15)): relative loss with vs without instruction context; identifies mismatched or unhelpful pairs.
      - AFLite predictability (Eq. (16)); uncertainty-based scores (Eqs. (17)–(20)).
    - GPT-as-a-judge (Section 3.3; Figure 4): Prompted grading (0–5) per dimension (Eq. (21)) with formatted outputs for automation.
    - Human evaluation (Section 3.4; Figure 5): Multi-aspect Likert scoring with explicit guidelines.
  - Diversity (Section 4)
    - Hand-crafted indicators (Section 4.1):
      - Lexical diversity: TTR (Eq. (24)), vocd-D (Eqs. (25)–(27)), MTLD (Eq. (28)), HD-D (Eq. (29)).
      - Semantic uniqueness: kNN distance in sentence-embedding space (Eq. (30)); per-sample PCA variance (Eq. (31)); dataset-level dispersion and cluster metrics (Eqs. (32)–(35)).
    - Model-based indicators (Section 4.2):
      - Entropy and Rényi entropy (Eqs. (36)–(37)).
      - Simpson’s Index variant (Eq. (38)).
      - Vendi Score (Eq. (39)) over kernel eigenvalues; can be quality-weighted.
      - Task2Vec/Fisher-based diversity coefficients (Eqs. (40)–(41)).
      - Tag-based coverage using LLM tagging and greedy tag expansion (Algorithm 1).
    - Geometry-based coreset sampling (Section 4.3):
      - k-center greedy (facility location; Eq. (42); Algorithm 2): iteratively add the farthest point from current core.
      - Herding (Algorithm 3): minimize distance between subset and full-set centers.
      - Hybrid heuristics: cluster-first sampling with distance and complexity constraints (Eq. (43); Algorithm 4); quality–diversity joint objective (facility-location + GPT score; Eq. (44); Algorithm 5); quality-first then representativeness filter (Algorithm 6); dedup + prototypicality (Algorithm 7).
      - Clustering families: k-means, DBSCAN, spectral; sampling from large diverse clusters, de-emphasizing near-duplicates.
    - Bilevel optimization (Section 4.4; Eq. (45)):
      - Optimize selection (outer loop) and model training (inner loop) jointly or via relaxations; examples include GREEDY/first-order approximations, proxy models, and validation-loss-driven objectives.
  - Importance (Section 5)
    - Hand-crafted difficulty (Section 5.1): Readability indices and domain “level” tags (e.g., math grade levels) to identify challenging samples.
    - Model-based indicators (Section 5.2):
      - Prompt uncertainty from perturbed instructions (Eq. (47)) or calibrated ensembles for multiple-choice (CAPE).
      - “Necessity” via reward of a model’s own generation (Eq. (48))—low reward implies need for tuning.
      - Datamodels: predict loss impact of including a sample using a linear TRaK-style regressor (Eqs. (49)–(50)), or simulators of training dynamics (Eq. (51)).
      - Distributional resemblance DSIR via hashed n-gram importance weights (Eq. (52)).
    - Loss/error-based importance (Section 5.3):
      - Forgetting events (Eq. (53)); memorization and influence via leave-one-out probabilities (Eqs. (54)–(55)).
      - Practical approximations via batching and proxy models.
    - Gradient-based importance (Section 5.4):
      - Gradient matching: selected-set gradients approximate full-set or validation-set gradients (Eq. (56)).
      - Influence functions: upweight a sample and estimate parameter and loss changes using Hessian-vector approximations (Eqs. (57)–(58)).
      - GraNd score (expected gradient norm; Eq. (59)); often approximated by EL2N.

- Design choices and why they matter
  - The three-pillar decomposition makes hidden trade-offs explicit: quality removes noise/mismatch; diversity avoids redundancy and broadens coverage; importance targets samples that most change model behavior (Figures 1–2).
  - The unified selection abstraction (`q`, `π`, budget `b`) lets practitioners swap scoring functions and samplers without changing the overall pipeline (Section 2).

## 4. Key Insights and Innovations
- A unified, fine-grained taxonomy tailored to instruction tuning (Figures 1–2; Sections 3–5)
  - Novelty: Goes beyond generic data selection to articulate instruction-specific notions of quality (instruction vs response), diversity (lexical vs semantic), and importance (complexity, loss contribution, gradient influence), each with concrete scoring recipes and algorithms.
  - Significance: Clarifies terminology and provides a menu of interoperable components.

- A single mathematical lens to connect assessment and selection (Section 2; Eqs. (2)–(5), (23), (46))
  - Novelty: Expresses varied methods—perplexity pruning, facility-location coresets, datamodels, influence functions—under the same “score + sampler + budget” framework.
  - Significance: Enables principled comparisons and hybridization (e.g., quality-diversity joint objectives in Algorithm 5).

- Mechanism-level walkthroughs with actionable algorithms and formulas
  - The paper does not stop at listing methods; it explains how to compute each indicator (e.g., IFD Eq. (15); Vendi Eq. (39); forgetting Eq. (53); gradient matching Eq. (56)) and how to implement selection (Algorithms 1–7).

- Evidence synthesis across dozens of works (Tables 2–4)
  - Novelty: Curates reported results into side-by-side comparisons—quality-, diversity-, and importance-driven selection—on widely used models and benchmarks.
  - Significance: Provides empirical priors about when each pillar helps (and when it doesn’t), guiding practitioners.

## 5. Experimental Analysis
Because this is a survey, the paper compiles results reported by each method rather than running new experiments. Still, the cross-paper synthesis is informative.

- Evaluation methodology landscape (Tables 2–4)
  - Datasets: Alpaca (52K), WizardLM, UltraChat, FLAN v2 subsets, OpenOrca/Dolly, OpenWebMath, The Pile/Dolma, etc. (Tables 2–4; Table 1 lists training-set sizes).
  - Benchmarks/metrics: ARC, HellaSwag, MMLU, TruthfulQA, BBH, HumanEval, SuperGLUE, LAMBADA, PIQA, etc. Scores are accuracy unless specified.
  - Models: LLaMA/LLaMA2 (7B/13B), Mistral-7B, StarCoder-15B, MPT-1B, Pythia (410M–1B), GPT-Neo-3B.

- Main findings with specific numbers
  - Quality-driven selection can beat full-data fine-tuning at small budgets.
    - IFD (Eq. (15)): On LLaMA2-7B with WizardLM data, using only 5% data improves MMLU from 0.541 (full) to 0.557, ARC from 0.576 to 0.624, and HellaSwag from 0.820 to 0.840; TruthfulQA improves from 0.415 to 0.428 (Table 2).
    - Alpagasus (GPT scoring, Figure 4): On LLaMA2-13B with Alpaca, 9K curated samples slightly outperform or match full 52K on BBH (0.344 vs 0.338) and HumanEval (0.159 vs 0.157) while being close elsewhere (Table 2).
    - Perplexity-based pruning (Eq. (10)): Selecting mid/high-perplexity halves of The Pile/Dolma yields equal or better scores on multiple composite categories (WK/CR/LU/SPS/RC) versus training on all data with a 1B model (Table 2).
  - Diversity-focused coresets reduce redundancy and generalize better.
    - DEITA (quality-first, diversity-aware filter): With 10K examples on LLaMA-13B, MMLU rises from 0.474 (random) to 0.606 and ARC from 0.558 to 0.595 (Table 3). On Mistral-7B with only 6K, DEITA boosts MMLU from 0.587 to 0.619 and TruthfulQA from 0.536 to 0.598.
    - QDIT (Algorithm 5): On LLaMA-7B with UltraChat, 10K data improves ARC from 0.583 (random) to 0.607; MMLU also increases (0.321→0.361) while HellaSwag is roughly similar (Table 3). Effects vary by source (UltraChat, LMSYS, Alpaca, etc.) and task (ARC, BBH, DROP).
    - ClusterClip: Balanced cluster sampling slightly but consistently edges out random/uniform on SuperGLUE and MT-Bench in instruction data; similar trends in math corpora (Table 3).
  - Importance-driven methods target what most moves the needle.
    - DsDm (datamodels; Eqs. (49)–(50)): On a Chinchilla-optimal 1.3B model, selected subsets improve BoolQ (0.549→0.580), COQA (0.188→0.255), and TriviaQA (0.037→0.071), while sometimes trading off on others (e.g., Hellaswag 0.449→0.423) (Table 4).
    - MATES (online datamodel partner): Across Pythia-410M and 1B, 20% curated data lifts multiple benchmarks consistently (e.g., OBQA 0.294→0.308 and Winogrande 0.506→0.527 at 410M) (Table 4).
    - DSIR (distributional resemblance, Eq. (52)): On RoBERTa-base, selected 51.2M examples beat random on GLUE tasks, notably RTE (0.674→0.751) and CoLA (0.494→0.540) (Table 4).
    - LESS (gradient-similarity selection): With only 5% data, LLaMA2-13B matches full-data BBH (0.506 vs 0.508) and slightly improves MMLU (0.540) over random (0.534); gains are larger on Mistral-7B (e.g., MMLU 0.618 vs full 0.604) (Table 4).

- Do the results support the claims?
  - Yes, but conditionally:
    - Quality-only selection often matches or beats full-data baselines at small fractions, yet not universally across all tasks (Table 2).
    - Diversity helps when redundancy is high or domain coverage is uneven (Table 3).
    - Importance is particularly effective when tuned to target evaluation distributions (Table 4), but may trade off on unrelated tasks (e.g., DsDm’s Hellaswag drop).
  - Robustness checks and ablations are method-specific; the survey highlights techniques like percentile thresholds (Eqs. (7)–(8)), clustering sensitivity, and calibration of uncertainty (Section 3.2 and 5.2), but consistency across all methods isn’t uniformly established.

- Notable caveats and mixed outcomes
  - Quality-based approaches using GPT-as-judge are strong but can be costly; pairwise judgments are often more stable than absolute scores (Section 3.3).
  - Uncertainty-based sampling sometimes underperforms random selection in instruction tuning (Section 3.2 citing Wu et al. 2023).
  - Importance-driven selection aligned to a specific evaluation set can reduce performance on unrelated benchmarks (Table 4).

## 6. Limitations and Trade-offs
- Assumptions and scope (Section 7)
  - Loss or gradient proxies aren’t universally predictive of downstream accuracy across tasks/models; the linkage is task-specific (Section 7.1).
  - “Good data” is not universally defined; quality, diversity, and importance overlap and their optimal weighting is task- and model-dependent (Section 7.2).

- Scenarios not fully addressed
  - Fairness and bias: Most selection pipelines don’t explicitly measure or optimize for demographic fairness or harm reduction beyond generic quality filtering (Section 1.3; Section 7.5).
  - Test contamination: Pretraining or annealing phases may already include benchmark-like data; few pipelines systematically check for leakage (Section 7.1).

- Computational and scalability constraints
  - Many indicators demand model training/inference (perplexity, reward, gradients, influence/Hessian), which grows expensive with model size (Section 7.4).
  - Clustering or k-NN over massive corpora requires efficient embeddings, indexing, and deduplication strategies (Section 7.3).

- Open weaknesses
  - Lack of standardized reporting across studies makes apples-to-apples comparison hard; budget sizes, evaluation suites, and scoring definitions differ (Tables 2–4).
  - Hybrid methods often hard-code step order (e.g., quality filter before diversity), which can prematurely discard important data (Section 6 “Hybrid Selection” discussion).

## 7. Implications and Future Directions
- How this work changes the landscape
  - Provides a practical blueprint to build instruction-tuning data pipelines: define `q(x_i)` using one or more pillars; choose a sampler `π` (greedy, probabilistic, coreset, bilevel); constrain by budget `b` (Section 2). This turns disparate ideas into swappable components.
  - Encourages hybrid, multi-objective selection that balances noise removal, coverage, and impact—moving beyond one-dimensional filtering (Figures 1–2; Algorithms 1–7).

- Follow-up research it suggests (Section 7)
  - Benchmarking and meta-evaluation:
    - Standardize how selection success is measured beyond loss proxies (Section 7.1).
    - Build “data selection report cards” that log quality/diversity/importance stats of chosen subsets.
    - Systematic contamination checks and data portraits for instruction data.
  - Adaptive, task-aware weighting:
    - Learn dynamic weights over quality/diversity/importance to match desired behaviors (creative generation vs factual QA) rather than fixed heuristics (Section 7.2).
  - Scaling and efficiency:
    - Develop robust proxy models for perplexity, reward, and influence that correlate with large LLMs (Section 7.4).
    - Hierarchical and approximate selection (random projections, PCA, hashed features) for billion-sample regimes (Section 7.3).
  - Fairness-aware selection:
    - Integrate embedding-, probability-, and generation-based bias measures into `q(x_i)` and evaluate fairness changes post-selection (Section 7.5).

- Practical applications
  - Low-budget instruction tuning for startups or teams with limited compute: e.g., use IFD or perplexity to filter, then k-center greedy to diversify, optionally add importance via DSIR or LESS for a target benchmark (Sections 3–5; Tables 2–4).
  - Domain specialization (math/code/biomed): combine domain-appropriate difficulty tags, semantic deduplication, and importance against target evals.
  - Continual alignment: periodically reweight or refresh data using online datamodels (MATES) to track changing model needs (Section 5.2).

> Overall takeaway (Figure 2, Sections 3–5): robust instruction-tuning data pipelines benefit from explicitly separating and then recombining three concerns—“Is this sample correct and well-formed?” (quality), “Does it add non-redundant coverage?” (diversity), and “Will it change my model the most for my goals?” (importance)—implemented through concrete, swappable scoring functions and selection mechanisms under a fixed budget.
