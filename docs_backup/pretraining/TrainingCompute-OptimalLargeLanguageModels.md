# Training Compute-Optimal Large Language Models

**ArXiv:** [2203.15556](https://arxiv.org/abs/2203.15556)

## 🎯 Pitch

This paper fundamentally rethinks how to allocate compute in training large language models: instead of prioritizing ever-larger models with fixed data, the authors demonstrate that scaling model size and training data equally (about 20 tokens per parameter) yields far better results under a fixed compute budget. By validating this rule through both exhaustive empirical analysis (400+ models) and by training the new Chinchilla model, which outperforms much larger LLMs, the work shows that current practices leave significant accuracy and compute-efficiency on the table—marking a paradigm shift in how future LLMs should be built and optimized.

---

## 1. Executive Summary (2-3 sentences)
This paper asks: with a fixed training compute budget, how should one split that compute between model size (`N`, number of parameters) and data size (`D`, number of training tokens) to get the best language model? By fitting three complementary scaling analyses over 400+ training runs and then validating the prediction with a 70B-parameter model (“Chinchilla”) trained on 1.4T tokens, the work finds a simple rule: for compute‑optimal training, scale parameters and tokens in roughly equal proportion (≈“20 tokens per parameter”), which overturns prior guidance that favored ever larger models trained on roughly fixed data.

## 2. Context and Motivation
- Problem addressed
  - Large language model (LLM) training is constrained by a fixed compute budget (`C`, typically measured in FLOPs—floating point operations). Given `C`, the critical design choice is the trade‑off between:
    - `N` (model size, parameters) and
    - `D` (number of training tokens—the unit of text the model sees during training).
  - The paper formalizes the goal as minimizing final pretraining loss `L(N, D)` subject to `FLOPs(N, D) = C` (Equation (1), Section 3).

- Why this matters
  - Real‑world: Training and inference of LLMs consume substantial compute and energy. Choosing a non‑optimal `N` vs `D` wastes compute and yields worse models for the same budget (Introduction; Table 1).
  - Theoretical: Scaling laws guide the field’s investments and architecture choices. Prior scaling work (Kaplan et al., 2020) suggested increasing parameters much faster than data; many recent LLMs followed this recipe and trained roughly 300B tokens regardless of size (Table 1).

- Shortcomings of prior approaches
  - Fixed data schedule: Kaplan et al. used a fixed learning‑rate schedule calibrated to 130B tokens for all models. For models actually trained on fewer tokens, intermediate losses are overestimates because the schedule is too long (Appendix B, Figure A1). This biases conclusions toward “more parameters, less data.”
  - Limited size range and curvature: Most Kaplan runs were small (<100M parameters). Here, many runs are >500M parameters up to 16B, revealing curvature in the efficient frontier (Appendix E) that matters at higher budgets.

- Positioning relative to existing work
  - Builds on scaling law methodology but models loss as a function of both `N` and `D` and tunes the learning‑rate schedule to the actual number of tokens (Section 3).
  - Validates predictions by training a new model (Chinchilla, 70B) that re‑allocates Gopher’s compute (280B, 300B tokens) into fewer parameters and more data (Section 4).

## 3. Technical Approach
The paper’s central object is the constrained optimization:
- Goal: choose `N` and `D` to minimize final training loss `L(N, D)` under a fixed compute budget `C`:
  - Minimize `L(N, D)` subject to `FLOPs(N, D) = C` (Equation (1), Section 3).
- Compute model: As in prior work, training FLOPs are well‑approximated by `C ≈ 6 N D` (Appendix F), and the paper also derives a more detailed FLOP accounting that closely matches this approximation (Table A4).

Three complementary estimation strategies are used to infer the compute‑optimal allocation functions `N_opt(C)` and `D_opt(C)`.

1) Approach 1 — “Minimum over training curves” (Section 3.1; Figure 2)
- What was run
  - For each model size `N` (70M to >10B parameters), train the model across four different training‑horizon settings (i.e., four numbers of tokens `D`) using a cosine learning‑rate schedule that decays 10× over the intended number of tokens.
  - The cosine schedule length is matched to the training horizon; overestimating schedule length by more than ≈25% degrades performance (Appendix B; Figure A1).
- How the estimate is constructed
  - For each run, smooth and interpolate the loss as a function of FLOPs to get a continuous curve.
  - Across all runs, form the “envelope” of minimal loss at each FLOP budget—i.e., the best point available among all (`N`, `D`) choices for that compute (Figure 2, left).
  - Fit power laws relating the envelope’s optimal `N` and `D` to compute `C`: `N_opt ∝ C^a`, `D_opt ∝ C^b` (Figure 2, center and right).
- Result
  - `a = 0.50`, `b = 0.50` (Table 2): scale parameters and tokens equally with compute.

2) Approach 2 — IsoFLOP profiles (Section 3.2; Figure 3)
- Definition (uncommon term): An “IsoFLOP profile” is the curve of final losses obtained by training many model sizes under the same total training compute `C`. To keep compute fixed when changing `N`, the number of tokens `D` is adjusted so that `6 N D ≈ C`.
- What was run
  - For nine FLOP budgets ranging from 6e18 to 3e21, vary `N` up to 16B parameters, set `D` accordingly, and train to completion with a cosine schedule matched to `D`.
- How the estimate is constructed
  - For each budget, plot final loss vs `N`. The curve exhibits a U‑shaped valley with a clear minimum, revealing the optimal `N` for that compute (Figure 3, left).
  - Fit a parabola to the valley to estimate the minimizing `N`.
  - Repeat across budgets; fit power laws `N_opt ∝ C^a`, `D_opt ∝ C^b` (Figure 3, center and right).
- Result
  - `a = 0.49`, `b = 0.51` (Table 2): again, nearly equal scaling.

3) Approach 3 — Parametric loss model with closed‑form optimum (Section 3.3; Figure 4)
- Loss decomposition (define uncommon terms)
  - Postulate that loss can be decomposed into:
    - `E`: irreducible error (entropy of text; Bayes risk),
    - `A / N^α`: function-approximation error (even with infinite data, a finite model underperforms the ideal model),
    - `B / D^β`: optimization/data error (with finite training steps over finite data, you do not fully optimize).
  - Together: `L̂(N, D) = E + A/N^α + B/D^β` (Equation (2); derivation in Appendix D.2).
- Fitting procedure
  - Fit `(A, B, E, α, β)` by minimizing a robust Huber loss between `log L̂(N, D)` and observed `log L` over all runs, using L‑BFGS with many initializations to avoid local minima (Equation (3), Section 3.3).
  - Robustness is needed because small‑compute points are noisier; Huber loss down‑weights outliers.
- Closed‑form compute‑optimal solution
  - Under `C ≈ 6 N D`, minimizing `L̂` yields:
    - `N_opt(C) = G (C/6)^a` and `D_opt(C) = G^{-1} (C/6)^b` with
      - `a = β/(α + β)`, `b = α/(α + β)`, and `G = (α A / (β B))^{1/(α + β)}` (Equation (4)).
  - Fitted exponents: `α ≈ 0.34`, `β ≈ 0.28` (Appendix D.2, Equation (10)), which imply `a ≈ 0.46`, `b ≈ 0.54` (Table 2).
- Interpretation
  - Once `α, β` are learned from data, the rule `a ≈ b ≈ 0.5` drops out analytically.

Practical rule of thumb implied by all three approaches
- Equal scaling with compute implies a nearly constant tokens‑per‑parameter ratio along the optimal frontier. Table 3 shows that across many sizes, the optimal `D / N` is ≈20–22 tokens per parameter (e.g., 205B tokens for 10B parameters; 1.5T tokens for 67B).

Validation experiment (“Chinchilla”)
- Prediction: With the same compute budget used for Gopher (≈5.76e23 FLOPs; Figure 2, green markers), the compute‑optimal model should be far smaller (≈40–70B parameters) but trained on ≈4× more tokens (Section 3.4; Figures 1 and 3).
- Test: Train `Chinchilla` with 70B parameters on 1.4T tokens—same total compute as Gopher’s 280B trained on 300B tokens (Section 4).
- Implementation details (Table 4; Section 4.1)
  - Same architecture family and dataset (MassiveText) as Gopher, differences include:
    - Optimizer: AdamW instead of Adam; empirically better late‑training loss (Appendix G; Figure A7).
    - Tokenizer: SentencePiece variant without NFKC normalization; helps math/chemistry tokens.
    - Precision: store FP32 master copy of weights in the sharded optimizer state.
  - Learning‑rate schedule: cosine decay matched to total tokens; decay by 10× (Appendix B).

## 4. Key Insights and Innovations
- Equal‑proportions scaling law for compute‑optimal training
  - What’s new: Across three independent methods, the optimal exponents are `a ≈ b ≈ 0.5` (Table 2; Figures 2–4).
  - Why it matters: Prior guidance (Kaplan et al.) suggested `a = 0.73`, `b = 0.27` (Table 2), encouraging much bigger models with relatively little extra data. The new result implies most recent large models are under‑trained for their size (Figure 1; Table 3) and that data scaling must keep up with parameter scaling.

- Practical “20 tokens per parameter” rule of thumb
  - Grounding: Table 3’s optimal tokens for a given parameter count imply ≈20–22 tokens per parameter along the efficient frontier (e.g., 67B → 1.5T tokens; 175B → 3.7T tokens).
  - Significance: Gives practitioners a simple target to decide dataset size for a planned model and compute budget.

- Loss decomposition with closed‑form optimum (Section 3.3; Equation (4))
  - Novelty: A simple parametric `L̂(N, D)` with interpretable terms—irreducible entropy, approximation error, and optimization/data error—leads to closed‑form compute‑optimal allocations once exponents are fitted.
  - Benefit: Offers a conceptual explanation for the equal‑scaling result and a way to extrapolate beyond observed runs (Figure 4).

- Learning‑rate schedule must match training horizon (Appendix B; Figure A1)
  - Observation: Using a cosine schedule longer than the actual number of steps by >25% materially degrades final loss.
  - Implication: Intermediate checkpoints from long schedules (as in some prior analyses) can bias conclusions; training runs must be planned with schedule length matched to `D`.

- Cross‑dataset consistency (Appendix C; Figure A2; Table A2)
  - Finding: The equal‑proportion scaling repeats on C4 and a GitHub code dataset, suggesting the rule is not an artifact of MassiveText and holds in the “infinite‑data” regime (i.e., fewer than one epoch; Introduction footnote 2 and Section 3.4).

## 5. Experimental Analysis
- Evaluation methodology (Section 4.2; Table 5)
  - Tasks include language modeling (Wikitext‑103; The Pile subsets), reading comprehension (RACE‑h/m, LAMBADA), general knowledge and reasoning (MMLU, BIG‑bench), closed‑book QA (Natural Questions, TriviaQA), common sense (HellaSwag, PIQA, Winogrande, SIQA, BoolQ), and safety/bias probes (Winogender, toxicity).
  - Metrics: bits‑per‑byte (language modeling), perplexity, accuracy (0/5/10/64‑shot), toxicity scores, and bias resolution accuracy.

- Core quantitative results
  - Language modeling (Figure 5; Table A5)
    - On every Pile subset, Chinchilla reduces bits‑per‑byte compared to Gopher; e.g., `arXiv`: 0.627 vs 0.662; `github`: 0.337 vs 0.377. Wikitext‑103 perplexity improves to 7.16 vs 7.75.
    - Caveat acknowledged: more training data increases potential for train/test overlap; hence downstream tasks weigh more heavily in judging generalization (Section 4.2.1).

  - MMLU, a 57‑task exam‑like benchmark (Table 6; Figure 6)
    - 5‑shot average accuracy: “Chinchilla 67.6%” vs “Gopher 60.0%.”
    - Chinchilla outperforms on 51/57 tasks, ties on 2, slightly worse on 4 (Figure 6).
    - Notable >90% accuracies on individual tasks (e.g., `international_law`, `sociology`, `us_foreign_policy`, `high_school_gov_and_politics`; Section 4.2.2; Table A6).

  - BIG‑bench subset (Figure 7; Table A7)
    - Average accuracy across 62 tasks: 65.1% vs Gopher’s 54.4% (+10.7% absolute).
    - Underperforms Gopher on only 4 tasks (`crash_blossom`, `dark_humor_detection`, `mathematical_induction`, `logical_args`) out of 62 (Figure 7).

  - Reading comprehension (Table 7)
    - LAMBADA zero‑shot: 77.4 vs 74.5 (Gopher) and 76.6 (MT‑NLG‑530B).
    - RACE‑m: 86.8 vs 75.1 (Gopher); RACE‑h: 82.3 vs 71.6. (Note: GPT‑3 and MT‑NLG prompts are not directly comparable for RACE; Section 4.2.3).

  - Closed‑book QA (Table 9)
    - Natural Questions (dev): 5‑shot 31.5 vs 24.5 (Gopher); 64‑shot 35.5 vs 28.2; also better than GPT‑3’s 29.9 (64‑shot).
    - TriviaQA (unfiltered, test): 0‑shot 67.0 (vs Gopher 52.8, GPT‑3 64.3); 64‑shot 72.3 (vs 61.3 and 71.2). On the filtered set, approaches open‑book SOTA (FiD) within 7.9 points at 64‑shot.

  - Common sense (Table 8)
    - Beats Gopher and GPT‑3 on all reported tasks (HellaSwag, PIQA, Winogrande, SIQA, BoolQ), and MT‑NLG‑530B on 4/5.

  - Safety and bias (Table 10; Section 4.2.7)
    - Winogender coreference: overall 78.3% vs 71.4% (Gopher). Bigger gains for female and neutral pronouns (e.g., female “gotcha” +10% absolute).
    - Toxicity (unprompted generation measured by PerspectiveAPI): distributions are very similar to Gopher; 95th percentile toxicity 0.238 (Chinchilla) vs 0.230 (Gopher)—i.e., better models are not necessarily more toxic (Section 4.2.7).

- Supporting ablations and checks
  - Optimizer and precision changes: AdamW and FP32 master weights help, especially late in training (Appendix G; Figures A6–A7).
  - Learning‑rate schedule sensitivity: matching schedule length to tokens is important (Appendix B; Figure A1).
  - Cross‑dataset scaling checks: equal‑scaling holds on C4 and GitHub (Appendix C; Figure A2; Table A2).
  - Small‑scale head‑to‑head vs Kaplan’s predicted model size: at 1e21 FLOPs, the size predicted by this paper outperforms Kaplan’s prediction (Appendix D.4; Figure A4).
  - FLOPs accounting: detailed formula aligns closely with `6 N D` (Appendix F; Table A4).

- Do the experiments support the claims?
  - Yes on two fronts:
    - Scaling law inference: Three independent estimation methods agree on near‑equal scaling (Table 2; Figures 2–4), with consistent cross‑dataset evidence (Appendix C).
    - Practical validation: Reallocating compute from parameters to data (Chinchilla vs Gopher) yields broad, uniform improvements across diverse downstream tasks (Figures 5–7; Tables 6–9).

## 6. Limitations and Trade-offs
- Assumptions
  - Infinite‑data regime: Analyses assume fewer than one epoch over the corpus; i.e., no re‑use of data (Introduction footnote 2; Section 3.4). Multi‑epoch behavior is not addressed.
  - Power‑law frontier: Efficient frontier is modeled as power laws of `C` in `N` and `D`, although curvature is observed at higher budgets (Appendix E). This may make large‑scale extrapolations conservative on the “smaller is better” side.
  - Compute approximation: Uses `C ≈ 6 N D`; while checked (Appendix F; Table A4), architectural details can shift constants.

- External validity and confounds
  - Only two large‑scale full‑budget runs (Gopher vs Chinchilla) validate the prediction; intermediate scales are inferred from smaller runs (Discussion, Section 5).
  - Data quality and leakage: Using much more data may increase risk of overlap with evaluation sets; the paper explicitly cautions on language modeling metrics and emphasizes downstream tasks (Section 4.2.1).

- Practical constraints
  - Data acquisition: The rule implies multi‑trillion‑token datasets for 100B+ models (Table 3). Gathering high‑quality, diverse, responsibly sourced data at that scale is challenging.
  - Training time and system engineering: Even with fewer parameters, training on 4× tokens stresses throughput, storage, and data pipelines (not deeply analyzed here).
  - Inference vs training trade‑off: While inference becomes cheaper with smaller `N`, training time may increase due to larger `D`; the balance depends on hardware and parallelism.

- Methodological choices
  - Robust fitting (Huber) down‑weights low‑compute points, potentially biasing toward trends seen at higher budgets (Section 3.4).
  - Optimizer and tokenizer differences between Chinchilla and Gopher are shown to help but are not the dominant cause of gains; nevertheless, they are a mild confound (Appendix G).

## 7. Implications and Future Directions
- How this changes the field
  - Re‑centers data scale as a first‑class lever: For compute‑optimal training, data must scale with parameters, not trail far behind. This challenges “bigger‑is‑better” parameter‑centric scaling and suggests many current LLMs are under‑trained (Figure 1; Table 3).
  - Practical recipe: When planning a model for a target compute, aim for roughly equal log‑scaling in `N` and `D`, or approximately 20 tokens per parameter along the efficient frontier.

- Follow‑up research enabled/suggested
  - Multi‑epoch regime: Extend the analysis when datasets are finite and repeated passes are unavoidable; characterize how the `a, b` exponents change.
  - Frontier curvature: Model the observed concavity (Appendix E) and test at intermediate large scales to refine extrapolations.
  - Better loss decompositions: Improve `α, β` via architectures/optimizers that raise data/parameter efficiency; the fitted values `α≈0.34`, `β≈0.28` (Appendix D.2) are below the classical 0.5 rates.
  - Modality generalization: Apply the methodology to other domains (code, speech, vision) and to sparse/MoE models where the FLOPs‑to‑parameter mapping differs.
  - Data quality and governance: Build large, high‑quality, responsibly sourced corpora; study deduplication, contamination, and privacy at trillion‑token scales.

- Practical applications and downstream use
  - More capable models for the same training budget: Chinchilla shows broad task improvements with fewer parameters—useful for organizations with fixed compute.
  - Lower inference and fine‑tuning cost: Smaller models are cheaper to deploy and adapt (Section 4).
  - Training‑planning tools: The parametric fit (Equation (4)) plus Table 3 provide concrete targets for `N`, `D`, and `C` planning.

Quote highlights grounded in the paper
- Scaling law exponents:
  > “All three methods predict that current large models should be substantially smaller and therefore trained much longer…” (Figure 1; Table 2 with `a ≈ b ≈ 0.5`).
- Validation via Chinchilla:
  > “Chinchilla uniformly and significantly outperforms Gopher (280B)… on a large range of downstream evaluation tasks.” (Abstract; Section 4.2 with Tables 6–9, Figures 5–7).
- Practical planning numbers:
  > “A 175 billion parameter model should be trained… on over 3.7 trillion tokens” (Table 3), implying ≈21 tokens per parameter.

Definitions (selective, for uncommon terms)
- `Token`: a basic unit of text (e.g., a subword) used by the model; total tokens `D` measure training data size.
- `FLOPs`: count of floating‑point operations; proxy for compute budget.
- `IsoFLOP profile`: the curve of model performance vs size when total training FLOPs are held fixed by trading off parameters and data.
- `Cosine schedule`: a learning‑rate schedule that decays following a cosine curve from a maximum to a lower bound; here the decay is 10× over the intended number of training tokens (Appendix B).
- `Huber loss`: a robust loss function that behaves like squared error near zero and absolute error for larger residuals; used to reduce the influence of outliers when fitting the loss model.
- `Bayes risk`: the minimum achievable expected loss given the data distribution; here it corresponds to the entropy of natural text (Appendix D.2).

Overall, the paper provides both a new empirical law—equal scaling of parameters and data for compute‑optimal training—and a compelling validation via a stronger, smaller model trained on more data with the same compute.
