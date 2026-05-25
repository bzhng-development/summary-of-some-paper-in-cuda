# CLIMB: CLustering-based Iterative Data Mixture Bootstrapping for Language Model Pre-training

**ArXiv:** [2504.13161](https://arxiv.org/abs/2504.13161)

## 🎯 Pitch

CLIMB introduces the first fully automated framework to optimize pre-training data mixtures for language models by discovering, clustering, and iteratively refining web-scale datasets—without the need for manual domain labels. This breakthrough matters because it enables the efficient harnessing of vast, unlabeled web corpora to systematically boost domain-specific and general language capabilities, outperforming state-of-the-art data mixing baselines and laying the foundation for robust, efficient, and scalable future language model training.

---

## 1. Executive Summary
CLIMB is a fully automated framework that discovers and refines the “right mix” of pre‑training data for language models by clustering web-scale corpora and iteratively searching mixture weights with small proxy models and a performance predictor. It matters because large web datasets lack reliable domain labels, and manually curating mixes is costly; CLIMB consistently yields stronger downstream performance than state-of-the-art mixing baselines and enables new, released corpora (ClimbLab and ClimbMix) that scale well (Fig. 1, Tables 1–2).

## 2. Context and Motivation
- Problem/gap:
  - Most web data (e.g., Common Crawl) has no domain labels; curated datasets (e.g., The Pile) are expensive to build and still leave open the question of how to proportion domains for a given goal (Abstract; Sec. 1).
  - The “best” pre-training mixture depends on target capabilities (general reasoning vs. STEM vs. coding) and is a non-linear function of composition (Sec. 1, para. starting “Despite the success…”).
- Why it matters:
  - Targeted data during the last stage of pre-training (“mid-training”) boosts key capabilities like math, reasoning, and code (Sec. 1 citing [1], OLMo 2 [5]). Efficiently allocating limited training tokens to high-value data can reduce compute and improve quality.
- Prior approaches and shortcomings:
  - Heuristic filtering (perplexity, “educational value”) misses domain-relevant content (Sec. 1).
  - Domain‑weight optimization methods like `DoReMi` and `RegMix` typically assume clear domain partitions (The Pile) and often do a single-shot selection/sweep (Sec. 2). They do not directly solve: “discover the domains, then optimize the mix” for unlabeled, messy web data.
  - Other selection strategies (e.g., training-dynamics, embedding de-dup, domain classifiers) either require labels, rely on heuristics, or target fine-tuning/continued pre-training at smaller scales (Sec. 2).
- Positioning:
  - CLIMB unifies unsupervised data discovery (clustering) and an iterative mixture search using proxy LMs plus a learned predictor, enabling domain-aware mixes without manual labels and with compute reuse across iterations (Sec. 3; Fig. 4).

## 3. Technical Approach
CLIMB has two phases: (a) Data preprocessing to create semantically coherent clusters; (b) Iterative search to learn mixture weights that maximize downstream validation performance.

A. Data preprocessing (Sec. 3.1; Fig. 4a)
- Goal: create a small set of meaningful “clusters” from unlabeled web-scale text, forming the search space over which to mix.
- Steps:
  1) Text embedding:
     - Encode each document with an embedding model `Me` (`textstella_en_400M_v5`) to get vectors `E={E1,…,En}` (Sec. 3.1; Sec. 4.1 “Text embedding”).
     - Rationale: clustering in an embedding space groups by semantics rather than surface words (Sec. 3.1).
  2) Initial clustering:
     - Run K‑means (FAISS implementation) with a large `Kinit=1000` to get fine-grained groups (Sec. 4.1).
  3) Quality pruning and merging:
     - Train fastText classifiers on LLM‑annotated scores (quality, educational value, informational value, ad score; 1–5) for 1M texts labeled by Nemotron‑340B; prune clusters below a loose threshold (3.0), leaving `Kpruned=240` (Sec. 4.1).
     - Merge similar clusters by centroid distance (threshold 1.5), yielding `Kenhanced=16` super-clusters (Sec. 4.1).
     - Source data ultimately spans 21 clusters by adding five curated/high‑quality groups (Cosmopedia, FineWeb‑Edu, Python‑Edu, and two academic clusters; Appendix A.1.2).

B. Iterative mixture-weight search (Sec. 3.2; Fig. 4b–c)
- Problem formulation:
  - We want mixture weights `α` over clusters `D={D1,…,Dk}` that maximize downstream performance `P` of a model trained with that mixture.
  - Bi-level view (Eq. 1): for any `α`, the model is trained to get weights `ω*(α)`, and we evaluate validation loss/score `ℓval(α,ω*(α))`. We seek the `α*` minimizing validation loss (maximizing performance).
- Approximation with a predictor:
  - Training a model for every `α` is infeasible. Fit a predictor `fθ(α)` on a set `S` of tried mixtures and their measured scores to approximate `ℓ(α, ω)` (Eq. 2).
  - Use `LightGBM` regression because it works well with few samples, supports regularization, and is fast (Sec. 4.1).
- Coordinate-descent style iteration (Eqs. 3–4; Fig. 3):
  - Iteration `k`:
    - Sampling: score all unseen `α` via current predictor `fk`; pick `M` new candidates from the Top‑`N` predicted set to balance exploration/exploitation; add them to `S` (Eq. 3).
    - Proxy training: train proxy LMs on those mixtures; measure their validation performance; append `(α, score)` to `S`.
    - Predictor update: refit `f` on expanded `S` (Eq. 4).
  - Initialization: sample `α` from a Dirichlet distribution biased by cluster token counts (Sec. 4.1 “Iterative bootstrapping”; Ablation “Abl.init” in Table 3).
  - Practical schedule: three iterations with 64, 32, and 16 candidate mixtures, respectively—112 total proxy runs at the default 100% search budget (Sec. 4.1).
  - Rationale: iterative pruning avoids wasting compute on poor configurations and concentrates resolution (Fig. 3 shows the search space narrowing over iterations).

C. Proxy and target training pipeline (Sec. 4)
- Pre-training context (“phase‑1”):
  - Base Transformer decoder models of 62M, 350M, 1B are trained on 10T tokens with a warmup‑stable‑decay schedule (WSD) to provide a common foundation for continuous pre-training studies (Sec. 4 “Model”).
- Proxy models:
  - 62M and 350M proxies are used during search to estimate mixture quality efficiently (Sec. 4 “Model”).
- Target models and evaluation:
  - After selecting the best mixture, target models (62M/350M/1B) are further trained (“continuous pre-training”) on 40B tokens with that mixture (Sec. 4 “Model”).
  - For scaling comparisons, a ~1B model is also trained on 400B tokens (Table 2; Fig. 1).

D. How the pieces fit together (analogy)
- Imagine the raw web as a massive, unlabeled library. CLIMB:
  - Groups similar books into shelves (embedding + clustering),
  - Tosses out low-quality or spammy shelves (pruning),
  - Combines shelves in different proportions (mixtures),
  - Uses a quick reviewer (proxy LM) plus a trained critic (predictor) to decide which combinations are promising,
  - Iteratively refines choices, spending more time on promising combos and discarding poor ones.

## 4. Key Insights and Innovations
1) Unsupervised domain discovery + iterative mixture search
- What’s new: Instead of assuming labeled domains, CLIMB creates them via embedding-based clustering and then optimizes mixture weights iteratively (Sec. 3; Fig. 4).
- Why it matters: Enables domain-aware mixing on unlabeled, web-scale data—something prior methods (e.g., `DoReMi`, `RegMix`) do not fully address when labels are absent (Sec. 2).

2) Bi-level optimization solved by proxy models and a learned predictor
- What’s new: Casting mixture selection as bi-level optimization (Eq. 1) and approximating the objective with a learned regressor (Eq. 2), updated in a coordinate-descent loop (Eqs. 3–4).
- Why it matters: Avoids brute-force sweeps; concentrates compute on high-value regions of the mixture space (Fig. 3). Yields better results within the same or less search compute (Table 3 “Abl.comp”, “Abl.allo”).

3) Practical, scalable clustering pipeline with quality-aware pruning
- What’s new: A two-stage K-means (large `Kinit`, then quality filtering and merging) that converts terascale text into a tractable number of high-quality clusters (Sec. 4.1).
- Why it matters: Reduces noise and makes the search space compact and meaningful. The fastText-based quality pruning ties cluster construction to empirical data quality signals.

4) Released research corpora and mixtures that scale well
- What’s new: `ClimbLab` (1.2T tokens, 20 clusters) as a research playground and `ClimbMix` (400B tokens) as a compact, optimized corpus (Sec. 7). Figure 1 shows better scaling than several strong baselines when training a 1B model.
- Why it matters: Makes the approach actionable and testable by others; demonstrates that the discovered mixtures can outperform manual or heuristic datasets under the same token budget (Fig. 1; Table 2).

## 5. Experimental Analysis
A. Evaluation setup (Sec. 4; Tables 1–2)
- Datasets:
  - Search and training data come from Nemotron‑CC and smolLM‑corpus; CLIMB produces 21 clusters (Appendix A.1.2). For released corpora, CLIMB organizes into 20 clusters (Sec. 7).
- Benchmarks and metrics:
  - General reasoning tasks: PIQA, ARC‑C/E, HellaSwag, WinoGrande, SIQA, TruthfulQA, MMLU, OBQA, BoolQ, RACE (LM‑Evaluation Harness; 0‑shot except MMLU 5‑shot; Sec. 4 “Data”).
  - Optimization set during search: PIQA, ARC_E, HellaSwag validations; evaluation on their test splits plus broader benchmarks (Sec. 4 “Data”).
- Baselines:
  - `Random` (uniform cluster weights), `DoReMi`, `RegMix` (single-round regression-based mixture) (Sec. 4 “Baselines”; Appendix A.1.1).
- Models:
  - Proxy: 62M/350M; Target: 62M/350M/1B trained for 40B tokens continuous pre-training with the selected mixture (Sec. 4 “Model”).
  - Larger-scale: ~1B model trained for 400B tokens with the optimized mixture for SOTA comparison (Table 2).

B. Main results
- Versus data-mixing baselines (Table 1; 40B tokens continuous pre-training):
  - 350M target: average accuracy 54.83% for CLIMB vs 53.78% (`RegMix`), 53.38% (`DoReMi`), 52.17% (`Random`).
  - 1B target: average 60.41% for CLIMB vs 59.37% (`RegMix`), 59.16% (`DoReMi`), 57.93% (`Random`).
  - Transfer beyond optimized tasks: Gains hold across benchmarks even though the predictor was trained only on PIQA/ARC_E/HellaSwag validations (Table 1, last paragraph of Sec. 5.1).
  - Quote:
    > Table 1 (1B target): CLIMB avg. 60.41 vs RegMix 59.37; DoReMi 59.16; Random 57.93.
- Against strong LMs under a fixed 400B-token budget (Table 2):
  - ~1B CLIMB model achieves the best average across 12 reasoning tasks: 53.54 vs Llama‑3.2‑1.2B’s 51.56 (+2.0 points).
  - Sub‑500M regime: CLIMB‑350M average 48.93, exceeding Qwen2.5‑490M (48.14) and SmolLM‑360M (47.78).
  - Quote:
    > Table 2: CLIMB (~1B) average 53.54; Llama‑3.2‑1.2B 51.56; AMD‑OLMo‑1.2B 49.93; TinyLlama‑1.1B 48.42.

C. Domain‑specific optimization (Fig. 5; Appendix A.6, Table 5)
- When the objective is a specific MMLU domain (STEM, Humanities, Social Sciences), CLIMB steadily improves over iterations and often surpasses an oracle-style baseline `CLIMB‑Best@N` that searches with a same-size target proxy (Sec. 6 “Optimizing towards Specific Domains”).
  - Example: 1B target on MMLU‑Social‑Sciences improves from 40.03% (Iter1) to 41.79% (Iter3), beating Best@N by +1.13% (Fig. 5c).

D. Ablations and robustness (Table 3; Figs. 6–8; Fig. 9)
- Search compute budget:
  - More candidate evaluations (150%, 200% of default 112 runs) continue to yield gains (avg 60.72 and 61.12 vs 60.41; Table 3 “Abl.comp”).
- Compute allocation across iterations:
  - The 4:2:1 split (64:32:16) outperforms 6:1 and 2:2:1:1, indicating a balance between exploration depth and breadth is beneficial (Table 3 “Abl.allo”).
- Proxy size:
  - Larger proxies slightly help (62M→350M improves avg 60.11→60.41; Table 3 “Abl.proxy”). A 62M proxy can still drive meaningful gains (Appendix A.6).
- Number of clusters:
  - Performance is fairly stable across `Kinit` and merging choices; overly fine-grained (`Kinit=2000`) or too many super-clusters can hurt or raise search cost (Table 3 “Abl.clus”).
- Initialization:
  - Dirichlet init is modestly better than random (60.41 vs 60.21; Table 3 “Abl.init”).
- Predictor quality:
  - The LightGBM predictor exhibits high rank correlation with true proxy scores (Spearman 0.94; Fig. 9).
- Weight dynamics and interpretability:
  - A few clusters dominate final weights; importance shifts across iterations (e.g., C8 and C9 gain while C19 and C21 shrink; Fig. 8a).
  - Topic inspection (Appendix A.2) and similarity analyses (Appendix A.3; Fig. 7) show that both in-domain similarity and complementary diversity matter.

E. Scaling behavior of released mixtures (Fig. 1; Sec. 7)
- With pre-training from scratch on `ClimbMix` (400B tokens), a 1B model shows better scaling than several dataset baselines across 32–400B tokens; average performance climbs from 46.36 to 52.43 (Fig. 1).  
  > Fig. 1: CLIMB’s mixture curve sits above Nemotron‑CC‑HQ, SmolLM, DCLM‑baseline, FineWeb‑Edu across token budgets.

F. Do results support claims?
- Yes, within scope:
  - Repeated improvements vs. strong baselines (`DoReMi`, `RegMix`) at matched 40B token budgets for continuous pre-training (Table 1).
  - SOTA‑competitive averages under a 400B token budget for ~1B models (Table 2).
  - Robustness shown by ablations on compute, proxy size, clustering granularity, initialization (Table 3), and by high predictor fidelity (Fig. 9).
- Caveats:
  - Absolute fairness across different labs’ 400B training recipes is always hard to guarantee; nevertheless, the token-budget framing, broad metrics, and released corpora make the case strong and reproducible (Sec. 7).

## 6. Limitations and Trade-offs
- Assumptions in clustering and pruning:
  - Quality pruning relies on scores predicted by fastText models trained on LLM annotations (Nemotron‑340B), which may encode annotator model biases and prompt design artifacts (Sec. 4.1; Appendix A.8). If the scoring scheme shifts, cluster retention could change.
- Dependence on embedding model:
  - The initial grouping quality depends on the embedding model (`textstella_en_400M_v5`); domain‑shifted or multilingual content might be imperfectly captured without dedicated embeddings (Sec. 4.1).
- Compute requirements for search:
  - Even with predictors, CLIMB’s default search uses 112 proxy runs (64/32/16). While cheaper than exhaustive sweeps, it is still nontrivial, especially if proxies are large or long‑trained (Sec. 4.1; Table 3).
- Task‑objective coupling:
  - The mixture is optimized for specific validation tasks (PIQA, ARC_E, HellaSwag). Although transfer appears robust (Table 1), extreme domain shifts (e.g., code generation, multilingual tasks) may need re-optimization (Sec. 5.1).
- Static clusters during search:
  - Clusters are fixed after preprocessing; there is no online reclustering as the search progresses. Mis-clustered or heterogeneous groups cannot be split adaptively mid‑search (Sec. 3.1–3.2).
- Scaling law deviations:
  - The base models are pre-trained on 10T tokens, acknowledged as over-training relative to standard scaling-law guidance (Appendix A.1.3). This helps standardize the starting point but complicates cross‑paper efficiency comparisons.
- Transparency of per‑candidate proxy training budget:
  - While iteration counts are clear, the exact per-candidate proxy training tokens/steps are not detailed in the main text, which would help readers estimate practical compute (Sec. 4.1 describes counts and predictor training, not the proxy token budget).

## 7. Implications and Future Directions
- Field impact:
  - CLIMB demonstrates that automated, iterative mixture optimization can outperform manual/heuristic curation even at web scale and without domain labels. This reframes data curation as an active learning/search problem with measurable gains (Sec. 5; Tables 1–2; Fig. 1).
- Practical applications:
  - Building domain-specialist pre-training runs (e.g., STEM, humanities, social sciences) with evidence of >5% gains vs. random sampling in the target domain (Sec. 5.2; Fig. 5).
  - Efficient continual pre-training for organizations with fixed token budgets: use proxies + predictors to pick the best mixture before committing large training runs.
  - Deploying released corpora: `ClimbLab` enables community research on mixture search; `ClimbMix` offers a strong 400B-token starting point (Sec. 7; Fig. 6 shows its final weights).
- Follow-up research:
  - Adaptive reclustering and mixture evolution: allow clusters to split/merge during search based on learned importance (extending Sec. 3.1–3.2).
  - Multi-objective mixture search: jointly optimize for, say, general reasoning and safety/harmlessness, or for multilingual + code, by extending Eq. (2) to vector-valued targets.
  - Causal and interaction-aware predictors: model cross-cluster synergies beyond additive features; explore GPs or neural surrogates with uncertainty to guide exploration.
  - Lower-cost proxies and early-stopping curricula: formalize when a small proxy or a few-billion-token “mini-run” suffices, leveraging the strong 0.94 rank correlation (Fig. 9).
  - Label-efficient quality pruning: replace LLM scoring with self-supervised or contrastive signals to reduce reliance on expensive annotations (Sec. 4.1).

> Bottom line: CLIMB turns pre-training data selection into a principled, iterative search over discovered semantic clusters, delivering consistent gains over strong baselines and producing openly released corpora that scale well. Its main trade-off is the added orchestration and proxy training compute, which future work can further reduce with adaptive clustering and uncertainty-aware predictors.
