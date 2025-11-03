# CLIMB: CLustering‑based Iterative Data Mixture Bootstrapping for Language Model Pre‑training

**ArXiv:** [2504.13161](https://arxiv.org/abs/2504.13161)
**Authors:** Shizhe Diao, Yu Yang, Yonggan Fu, Xin Dong, Dan Su, Markus Kliegl, Zijia Chen, Peter Belcak, Yoshi Suhara, Hongxu Yin, Mostofa Patwary, Yingyan (Celine) Lin, Jan Kautz, Pavlo Molchanov
**Institutions:** NVIDIA, Georgia Institute of Technology

## 🎯 Pitch

CLIMB revolutionizes language model pre-training by automating the optimization of data mixtures using unsupervised clustering and iterative mixture search. This groundbreaking framework outperforms existing methods by significantly enhancing reasoning performance while reducing manual curation costs, paving the way for more efficient and adaptable language models.

---

## 1. Executive Summary (2-3 sentences)
CLIMB is an automated framework that discovers and refines optimal pre‑training data mixtures for language models by clustering unlabeled web-scale data and iteratively searching mixture weights with a small proxy model plus a learned predictor. Across extensive experiments, CLIMB improves downstream reasoning performance at fixed token budgets and produces two public corpora (ClimbLab, ClimbMix); for example, a ~1B model trained 400B tokens on the CLIMB mixture outperforms Llama‑3.2‑1B by 2.0% average across 12 benchmarks (Table 2).

## 2. Context and Motivation
- Problem addressed
  - Pre‑training at web scale uses heterogeneous sources (e.g., Common Crawl) that lack domain labels. Deciding “how much of what” to train on—the data mixture—is crucial yet underexplored and typically hand‑crafted (Introduction; Figure 2).
  - The relationship between mixture composition and model performance is highly nonlinear; the best mixture for, say, coding may also need math and reasoning content (p. 2).
- Importance
  - Mid‑training (final stage of pre‑training) gains hinge on targeted, high‑quality data; curated mixtures improve reasoning, math, and coding (Introduction, citing [1], OLMo‑2 [5]).
  - Automating mixture optimization promises better performance per token and lowers manual curation costs.
- Prior approaches and gaps
  - Heuristic filtering (e.g., perplexity, “educational value”) misses domain-specific signal (p. 1).
  - Methods like DoReMi [16] and RegMix [36] assume pre‑defined domains or perform single‑shot mixture regression; they do not discover latent domains or optimize mixtures iteratively (Related Work; Section 3.2).
- Positioning
  - CLIMB replaces manual domains with unsupervised clusters (Section 3.1), then performs iterative mixture search with a weak predictor and proxy LMs (Section 3.2; Figure 4), enabling domain‑agnostic, compute‑aware optimization.

## 3. Technical Approach
CLIMB comprises two phases—data preprocessing (discover latent “domains”) and iterative mixture bootstrapping (search for sampling weights).

1) Data preprocessing: discover clusters that act as mixture units (Section 3.1; Figure 4a)
- Text embedding
  - Every document is mapped into an embedding using `textstella_en_400M_v5` (Section 4.1, “Text embedding”).
- Embedding clustering
  - FAISS k‑means groups embeddings into fine‑grained clusters with `K_init = 1000` (Section 4.1).
  - Rationale: start fine‑grained to capture semantic distinctions for later merging.
- Cluster pruning and merging
  - Quality screening: train fastText classifiers on 1M GPT‑annotated examples to score each document for four dimensions—quality, ad level, informational value, educational value (Section 4.1; Appendix A.8 for the prompt rubric).
  - Keep clusters whose average score ≥ 3.0, yielding ~240 clusters; merge by centroid distance threshold 1.5 to reduce to `K_enhanced = 16` super‑clusters (Section 4.1).
  - Final source set has 21 clusters by adding five curated clusters (e.g., Cosmopedia, FineWeb‑Edu, Python‑Edu, plus academic clusters) for ~800B tokens (Appendix A.1.2; Table 4 lists topics).

2) Iterative mixture bootstrapping: search over sampling weights (Section 3.2; Figure 4b–c)
- Goal in plain language
  - Choose a probability vector `α` over clusters (nonnegative, sums to 1) so that a model trained on data sampled according to `α` maximizes validation performance `P` on target tasks (e.g., PIQA, ARC_E, HellaSwag).
- Bi‑level formulation (Equation 1)
  - Inner problem: given `α`, train model weights `ω*(α)` to minimize training loss on data sampled by `α`.
  - Outer problem: choose `α` that minimizes (or maximizes) validation loss (equivalently maximizes accuracy) when the model is trained with `ω*(α)`.
- Approximating the objective with a predictor (Equation 2)
  - Full evaluation for every `α` is infeasible; instead, fit a regression predictor `f_θ(α)` from a small set `S` of observed (mixture, performance) pairs—obtained by training proxy LMs on sampled mixtures.
  - The search minimizes `f_θ(α)` over `α`, while `f_θ` itself is learned from pairs in `S`.
- Coordinate‑descent style iterative search (Equations 3–4; Figure 3)
  - Iteration k:
    1) Score all unseen mixtures using current predictor; keep top‑N; sample M candidates from them to balance exploitation and exploration.
    2) Train proxy LMs on these M mixtures; add their (mixture, performance) pairs to `S`; refit `f_θ`.
  - After K iterations, select the `α` with best predicted performance for the target training.
- Practical choices (Section 4.1)
  - Mixture initialization: sample from a Dirichlet distribution whose concentration reflects cluster token counts; encourages plausible sparsity and coverage.
  - Proxy models: primarily 350M parameters (also 62M, 132M in ablations), trained for mixture evaluation on 40B tokens of continuous pre‑training using a warmup‑stable‑decay LR schedule (Section 4.1; A.1.3).
  - Predictor: LightGBM regression with L1/L2 regularization, depth≤4, early stopping, separate validation set (Section 4.1).
  - Iterations: three rounds with 64, 32, 16 trained mixtures respectively (total 112 proxy trainings under the 100% budget).
  - Optimization targets: validation sets of PIQA, ARC_E, HellaSwag; evaluation on multiple held‑out test benchmarks via LM‑Eval harness; MMLU is 5‑shot, others 0‑shot (Section 4; A.1.2).
- Interpreting the procedure by analogy
  - Think of the clusters as ingredients in a recipe. A small chef model tastes many quick mini‑recipes. A critic learns to predict which ingredient mixes will taste best. The search gradually narrows to promising flavor regions, avoiding expensive blind exploration.

3) From CLIMB to public datasets (Section 7; Figure 6)
- ClimbLab: a 1.2‑trillion‑token, 20‑cluster corpus constructed by CLIMB’s clustering pipeline as a research playground.
- ClimbMix: a 400B‑token mixture selected by CLIMB; used to train models from scratch; mixture is more balanced than in continuous pre‑training (Figure 6), because from‑scratch models need broader coverage.

## 4. Key Insights and Innovations
- Unsupervised domain discovery at web scale
  - Instead of relying on human domain labels, CLIMB clusters document embeddings and prunes by learned quality signals (Section 3.1; 4.1; Table 4 for qualitative topics). This removes the dependency on existing taxonomies, addressing a core bottleneck in prior work like DoReMi, which assumes domain groups.
- Iterative mixture search with a weak predictor
  - A coordinate‑descent process alternates between sampling promising mixtures (based on the predictor) and updating the predictor with new proxy training results (Equations 3–4; Figure 3). RegMix performs only a single regression pass; CLIMB’s iterations focus compute on high‑value regions and empirically yield better mixtures (Table 1).
- Compute‑aware evaluation using small proxy LMs
  - The framework treats mixture evaluation as a multi‑fidelity process: use smaller, cheaper proxy LMs to estimate downstream performance, then fit a fast predictor (LightGBM). The predictor achieves high rank correlation with truth (Spearman 0.94; Appendix A.7, Figure 9).
- Mixture generalizes beyond the optimization targets
  - Despite optimizing only on PIQA/ARC_E/HellaSwag validation, CLIMB’s mixtures improve performance on diverse held‑out tasks (Table 1 and Table 2), implying the discovered clusters capture transferable capabilities.
- Public corpora enabling reproducible data‑centric research
  - ClimbLab and ClimbMix provide large, structured data assets for further mixture studies and efficient pre‑training (Section 7; Figure 1 for scaling comparison).

## 5. Experimental Analysis
- Evaluation setup
  - Source data: Nemotron‑CC high‑quality bucket plus curated sets; clustered into 21 clusters (~800B tokens) for the main continuous pre‑training experiments (Appendix A.1.2).
  - Models: decoder‑only Transformers at 62M, 350M, 1B; all received a 10T‑token phase‑1 pre‑training foundation; mixture search then continues training for 40B tokens (Section 4; A.1.3).
  - Targets and metrics: PIQA (accuracy_norm), ARC_C/E (accuracy), HellaSwag (accuracy_norm), WinoGrande and SIQA (accuracy); wiki and lambda perplexities are also reported (Table 1). Broader suite in Table 2 includes MMLU, OBQA, BoolQ, RACE, TruthfulQA.
  - Baselines: Random (uniform cluster weights), DoReMi (GroupDRO domain reweighting), RegMix (single‑round regression over sampled configurations) (A.1.1).
- Main quantitative results (continuous pre‑training, 40B tokens; Table 1)
  - 350M target:
    > CLIMB average = 54.83 vs RegMix 53.78, DoReMi 53.38, Random 52.17 (Table 1).
  - 1B target:
    > CLIMB average = 60.41 vs RegMix 59.37, DoReMi 59.16, Random 57.93 (Table 1).
  - Improvements are broad, not just on the optimized tasks; e.g., WinoGrande and SIQA also rise (Table 1).
- Scaling and SOTA comparison (from scratch, 400B tokens; Table 2)
  - Sub‑1.2B regime:
    > “CLIMB (Ours) 950M” average = 53.54, beating Llama‑3.2‑1.2B (51.56), TinyLlama‑1.1B (48.42), AMD‑OLMo‑1.2B (49.93). Per‑task examples: ARC_C 40.96, ARC_E 73.57, HellaSwag 66.90 (Table 2).
  - Sub‑500M regime:
    > “CLIMB (Ours) 350M” average = 48.93, outperforming Qwen2.5‑490M (48.14) and SmolLM‑360M (47.78) (Table 2).
- Domain‑specific optimization (MMLU domains; Figure 5)
  - CLIMB iteratively improves over Random and also over a strong baseline that searches directly with the target model on random configs (`CLIMB‑Best@N`):
    > For 1B on Social Sciences: 41.79% (CLIMB‑iter3) vs 40.66% (CLIMB‑Best@N) vs 36.69% (Random) (Figure 5c).
- Robustness and ablation studies (Table 3; Appendix A.6–A.7; Figures 6–8)
  - Search compute budget:
    > Increasing budget from 100% to 200% lifts 1B average from 60.41 to 61.12 (Table 3, Abl.comp).
  - Compute allocation across iterations:
    > 4:2:1 (64/32/16 searches) performs best vs 6:1 or 2:2:1:1 (Table 3, Abl.allo).
  - Proxy size:
    > Larger proxies help modestly: 62M→350M improves 1B average from 60.11 to 60.41 (Table 3, Abl.proxy).
  - Number of clusters:
    > Performance is relatively insensitive within a broad range; e.g., `K_init=1000, K_enhanced=21` yields 60.41; nearby settings vary ±0.5 points (Table 3, Abl.clus).
  - Initialization:
    > Dirichlet beats Random slightly (60.41 vs 60.21; Table 3, Abl.init).
  - Predictor quality:
    > Spearman rank correlation 0.94 between predicted and true accuracies (A.7, Figure 9).
  - Mixture analysis:
    > CLIMB finds sparse, high‑impact clusters (e.g., C8, C9, C18, C19) for general reasoning; weights shift over iterations as the search “homes in” (Figure 8a; A.4). Similarities to task embeddings explain some—but not all—of the weight patterns (Figure 7; A.3).
  - From‑scratch mixtures are more balanced than continuous mixtures, reflecting broader coverage needs (Section 7; Figure 6).
- Do the experiments support the claims?
  - Yes, in three ways:
    1) Against strong baselines at fixed tokens (Table 1), CLIMB improves averages across multiple tasks.
    2) When scaled up and compared to public SOTA models (Table 2), CLIMB-trained models score higher on a diverse suite, not just on the optimization targets.
    3) Ablations show the mechanism is robust to design choices and benefits from additional search compute (Table 3), while the predictor reliably ranks candidates (Figure 9).
  - The domain‑specific MMLU experiments (Figure 5) demonstrate adaptability beyond general reasoning.

## 6. Limitations and Trade-offs
- Upfront data engineering cost and bias
  - Quality scoring requires 1M GPT‑style annotations to train fastText classifiers (Section 4.1; Appendix A.8), which introduces cost and potential rubric bias (the rubric emphasizes “educational” and “advertising” aspects; Figure A.8).
  - Clustering relies on a particular embedding model; semantic partitioning quality depends on that choice (Section 4.1).
- Proxy mismatch and compute overhead
  - Although larger proxies help, gains are modest (Table 3), implying residual mismatch between proxy and target models. The search still requires 112–224 proxy trainings, which is substantial even if cheaper than brute force (Section 4.1; Table 3).
- Objective narrowness
  - Mixtures are optimized on a small set of validation tasks (PIQA, ARC_E, HellaSwag). While generalization is strong (Tables 1–2), other objectives (e.g., safety, factuality, multilinguality) are not explicitly optimized.
- Static mixture per training run
  - The method chooses one mixture before the 40B token continuation; dynamically changing mixtures during training is not explored experimentally (though discussed conceptually as “dynamic refinement,” Section 3.2). 
- Scope
  - Experiments focus on English web and curated data (Nemotron‑CC + SmolLM‑corpus). Cross‑lingual or multimodal generalization is not studied.
- Reproducibility at full scale
  - Recreating the exact clusters and filters requires the same embeddings, thresholds, and rubric; slight deviations may produce different partitions.

## 7. Implications and Future Directions
- How it changes the landscape
  - CLIMB operationalizes “data as a first‑class citizen” for LLM pre‑training: the mixture is engineered and optimized with the same rigor as model architecture or compute budget. The release of ClimbLab/ClimbMix enables reproducible, data‑centric research at scale (Section 7; Figure 1).
- Follow‑up research enabled or suggested
  - Multi‑objective mixture optimization (e.g., jointly target reasoning, safety, multilinguality), turning Equation (1) into a vector‑valued objective with Pareto search.
  - Dynamic curriculum schedules that adapt `α` over training time, potentially handled as a sequential decision process rather than a single `α`.
  - Better predictors: multi‑fidelity modeling that combines learning curves, loss profiles, or gradient signals; Bayesian optimization atop CLIMB’s clustering space.
  - Alternative clustering and pruning: contrastive or topic‑aware embeddings; human‑in‑the‑loop audits to mitigate rubric bias.
  - Cross‑domain specialization: run CLIMB for law, medicine, and code; compare discovered mixtures and cross‑domain transfer (Figure 5 shows a template via MMLU domains).
- Practical applications
  - Building small‑to‑mid‑scale specialist LMs with tight token budgets (e.g., a Social Sciences specialist improved by ~5% vs random sampling; Section 5.2, Figure 5).
  - Refreshing production pre‑training pipelines as the web shifts: re‑cluster, re‑search mixtures with small proxies to track domain drift.
  - Data governance: the clustering/mixture records provide transparent, auditable data composition knobs for compliance and safety reviews.

Overall, CLIMB provides a concrete, scalable recipe for turning unlabeled web data into purpose‑built pre‑training mixtures. The iterative search with proxy models and a predictor is the core mechanism (Section 3.2; Equations 3–4), and the empirical results (Tables 1–2, Figure 5) show that mixture engineering can unlock significant performance at fixed token budgets.
