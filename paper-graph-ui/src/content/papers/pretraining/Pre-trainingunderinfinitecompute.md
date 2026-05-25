# Pre-training under infinite compute

**ArXiv:** [2509.14786](https://arxiv.org/abs/2509.14786)

## 🎯 Pitch

This paper tackles the regime where training compute is plentiful but high-quality pre-training data is limited, a setting increasingly relevant as model compute scales outpace web data growth. The authors show that properly tuned, much stronger regularization—especially weight decay—enables predictable gains from larger models, and that scaling ensembles of independently trained models yields even greater data efficiency than scaling a single model. By composing these strategies and using distillation, they achieve dramatic, robust improvements in data efficiency that transfer to real downstream tasks and continued pre-training—offering new recipes for getting much more out of limited data in a compute-rich future.

---

## 1. Executive Summary
This paper studies how to pre-train language models when training compute is effectively unlimited but high‑quality training data is fixed and scarce. It shows that (a) tuning strong regularization (especially much larger weight decay) yields predictable, monotonic scaling with model size, and (b) scaling ensembles of independently trained models beats scaling a single model in this data‑constrained regime. Composing both and then distilling recovers most of the gains at small inference cost, delivering large and robust data‑efficiency improvements that also transfer to downstream tasks and continued pre‑training.

## 2. Context and Motivation
- Problem and gap
  - Compute devoted to pre‑training grows 4× per year while available web text grows ~1.03× per year (Introduction; citing [Sevilla & Roldán 2024; Villalobos et al. 2024]). The field is entering a regime where compute vastly exceeds data.
  - Standard scaling recipes assume fresh data can grow with compute. Under fixed data, common practices are to repeat (epoch) the dataset and to increase model size. Both eventually overfit—validation loss rises—even if one spends more compute (Section 2.1; Figure 2).
  - Open question: What training algorithms give the best possible performance when data is fixed and compute is unconstrained?

- Why it matters
  - Practical: Labs and developers increasingly face de‑duplicated or domain‑limited corpora. Getting more from the same tokens lowers costs, unlocks domain models, and reduces data‑collection risks.
  - Scientific: Understanding generalization under severe over‑parameterization and repeated exposure to the same data clarifies how scaling laws behave once “more data” is no longer the cure.

- Prior approaches and their limits
  - Chinchilla‑style compute‑optimal scaling couples data and parameters; it does not apply when data cannot grow (Section 2).
  - Data repetition/epoching shows monotone improvements in some reports but in practice often overfits for language modeling at scale (Section 2.1; Figure 2 left; Appendix D.1 discusses tuning).
  - Simply making models larger at fixed data also overfits (Section 2.1; Figure 2 right; echoes Kaplan et al. Figure 9 for single‑pass fixed‑data behavior).

- Position relative to existing work
  - The paper reframes evaluation: with data fixed and compute unconstrained, compare training recipes by the asymptote of their scaling laws (the predicted loss as model size or ensemble size tends to infinity), not by performance at a fixed compute budget (Section 3, “asymptote ED” in the fit L̂D,N = AD/N^αD + ED).
  - It combines classic techniques—regularization, ensembling, distillation—in a compute‑rich/data‑limited setting and quantifies them via scaling‑law asymptotes and cross‑data‑scale fits (Sections 3–6).

## 3. Technical Approach
This section reconstructs the paper’s methods step by step.

- Formalization of the objective (Section 2)
  - Define a training routine `A(D, H)` that takes a token budget `D` and hyperparameters `H` (including model size `N`, epochs `E`, LR, weight decay, etc.) and outputs a model `M`.
  - Quality is measured by validation loss `L(M)` on a held‑out i.i.d. split from the same corpus.
  - With data fixed at `D`, the goal is to minimize `L(A(D, H))`, unconstrained by compute (Section 2; “Problem setting” in Appendix A).
  - Evaluation reframing: since compute is unconstrained, the relevant comparison for monotone recipes is the asymptote (limit loss as we scale the knob—parameters or ensemble members—to infinity), not a point on a compute–performance curve (Section 3).

- Experimental environment (Sections 2, A)
  - Data: DCLM web corpus; default focus on D = 200M tokens, plus 400M/800M/1.6B to study data scaling (Section 2; Figure 7).
  - Architecture: Llama‑style decoder‑only models with context length 4096, bf16 compute, AdamW, cosine LR schedule with 1% warmup, gradient norm clipping 1.0 (Appendix A).
  - Model sizes: ~150M/300M/600M/1.4B parameters (Appendix A Table 2); note the 1.4B preset is relatively wide/few layers—a nonstandard scaling acknowledged later as a caveat (Appendix B.5).
  - Validation: fixed set of 1024 sequences (~4M tokens) across all experiments (Appendix A).
  - Metrics: validation loss (per‑token negative log‑likelihood) and downstream accuracy on ARC‑Easy, PIQA, SciQ to check correlation (Section 7; Figure 10; Table 5).

- Baseline: “standard recipe” under fixed data (Section 2.1; Figure 2)
  - Two knobs: increase epoch count `E` (more passes over the same tokens) and increase model size `N`.
  - Tuning: LR and `E` tuned per `N` (Appendix B.1).
  - Observation: Increasing `E` initially lowers loss, but too many epochs overfit (loss rises, Figure 2 left). Increasing `N` past a point also increases validation loss (Figure 2 right), even after tuning LR and `E`. Training loss keeps dropping in both cases—clear overfitting (Appendix B.5; Figure 15).

- Regularized parameter scaling (Section 3)
  - Key idea: Stronger regularization—especially much larger weight decay than the de facto 0.1—can prevent overfitting and restore monotone scaling with `N`.
  - Hyperparameter search: Coordinate‑descent‑style search for “locally optimal” `H` per (`D`, `N`) over a discretized grid of LR, `E`, and weight decay; a point is “locally optimal” if none of its 1‑step neighbors improves validation loss (Appendix B.1). Ablations show this joint tuning is necessary; naive transfer of best `E` or decay across `N` breaks monotonicity (Appendix B.2; Figure 11).
  - Result: Optimal weight decay grows with `N` and can be ~30× higher than standard practice (e.g., up to 3.2 vs 0.1; Figure 3 table; Appendix B.3 Figure 12). With this tuning, loss decreases monotonically with `N` and follows a power law:
    - L̂D,N = AD / N^αD + ED, fit over four `N` values (Section 3).
    - For D = 200M, the fit is L̂200M,N = 0.05 / N^1.02 + 3.43 (Figure 3). The exponent ~1 is steep compared to Chinchilla’s parameter exponent ~0.34.

- Ensembling: scaling number of independently trained members `K` (Section 4)
  - Mechanism: Train `K` independent copies of the same architecture and hyperparameters (different random seeds affect data order and initialization; Appendix C.1), and average their pre‑softmax outputs (“logits”) at inference. This “logit averaging” produces the ensemble prediction (Section 4.1).
    - Define EA(D, N, K, H) = LogitAvg({A(D, N, Zi, H)}i∈[K]).
  - Compute model: Inference cost and total parameters scale linearly with `K`; total “parameters used” is `N*K` (Section 4.1).
  - Observation: For fixed `N`, increasing `K` reduces loss roughly as 1/K and, crucially, approaches a lower asymptote than scaling `N` alone (Figure 4).
  - Hyperparameters for the `K → ∞` limit differ from single‑model optima: more epochs and less weight decay per member give a better ensemble asymptote (Section 4.2; Figure 5).
    - Heuristic that works broadly: “double the epochs, halve the weight decay” vs the single‑model optimum, with the same LR (Appendix C.2; Figure 17).

- Joint scaling: compose parameter scaling and ensemble scaling (Section 4.3)
  - Goal: Estimate the best possible loss when `K → ∞` and `N → ∞`. They take the limits in the order limN→∞ limK→∞ minH L(EA(D, N, K, H)) and argue the value is order‑independent under monotonicity (Appendix C.4).
  - Procedure (Figure 6):
    1) For each `N` at fixed `D`, fit a power law over `K` and extract its `K → ∞` asymptote (left panel).
    2) Fit a second power law over those asymptotes as `N` increases and read off the `N → ∞` asymptote (right panel).
  - Hyperparameters for ensembles use the above “double‑epoch/half‑decay” heuristic, validated across scales with one exception in the most over‑parameterized corner (Appendix C.2).

- Data‑scaling analysis (Section 5)
  - For each recipe—standard, regularized single model, and the joint N+K ensemble—estimate the best achievable loss at D ∈ {200M, 400M, 800M, 1.6B}.
    - Standard recipe: grid search over `N`, `E`, LR (weight decay fixed 0.1) and take the best model per `D` (Section 5.1; Appendix D.1; Figure 7 right, red points).
    - Regularized single model: for each `D`, fit the `N` power law and take its asymptote `ED` (Figure 7 left; purple points migrate to right panel).
    - Joint scaling: for each `D`, take `K → ∞` asymptotes per `N`, then `N → ∞` asymptotes to get one value per `D` (Figure 8 left→middle→right).
  - Then fit data‑scaling laws of the form L̂D = A / D^α + E for each recipe (Figures 7 right, 8 right).

- Distillation to reduce inference compute (Section 6; Figure 9; Appendix E)
  - Sequence‑level knowledge distillation: generate a large pool of synthetic tokens from a stronger “teacher” (here, ensembles of 300M models) and train a “student” (same 300M architecture) on a mixture of real tokens and synthetic tokens (Appendix E.1–E.3).
  - Key knobs: mixing ratio (batches of real : synthetic), epochs, LR, weight decay. Optimal distillation uses much smaller weight decay (0.1) than regularized pre‑training (Appendix E.2 Table 3). Teacher data is generated unconditionally at temperature 1 with a high‑throughput engine (Appendix E.1).
  - Self‑distillation: teacher and student have the same size; mixing real data with synthetic avoids model collapse and yields a student that surpasses the teacher (Figure 9 and Appendix E.3 Table 4).

- Downstream evaluation and continued pre‑training (Section 7)
  - Downstream benchmarks: ARC‑Easy, PIQA, SciQ using lm‑evaluation‑harness, mostly 200M‑token models/ensembles (Section 7.1; Figure 10 right; Table 5).
  - Continued pre‑training (CPT) case study: Llama 3.2 3B base on MegaMath‑Web‑Pro; restrict to 4B seed tokens (out of 73B) and apply small batch size + epoching + ensembling (Section 7.2; Table 1). Also compare ensembling vs weight‑averaged “model soups” here (Appendix G.2; Table 7).

## 4. Key Insights and Innovations
- Asymptote‑based evaluation for data‑constrained, compute‑unconstrained pre‑training
  - Innovation: Evaluate training “recipes” by the asymptote of their scaling law as the scalable knob (parameters or ensemble size) tends to infinity at fixed data (Section 3). This directly targets “best achievable with infinite compute,” not “compute‑optimal today.”
  - Why it matters: It ranks algorithms by their ceiling performance in the regime the field is heading into.

- Heavy regularization unlocks monotone parameter scaling far beyond Chinchilla ratios
  - Finding: With locally optimal tuning, weight decay needs to be ~30× larger than common defaults to prevent overfitting when tokens are scarce and models are very large (Figure 3 table; Appendix B.3).
  - Significance: Loss decreases smoothly with `N` following L̂200M,N = 0.05/N^1.02 + 3.43 up to parameter‑to‑token ratios ~140× Chinchilla (Figure 3). This is both a practical recipe and a conceptual bridge to theory showing optimal regularization mitigates double descent.

- Ensembles beat single‑model scaling under fixed data—and need different member hyperparameters
  - Finding: For `N = 300M`, scaling `K` gives an asymptote ≈3.34 vs 3.43 for `N → ∞` single‑model scaling (Figure 4). After re‑tuning member hyperparameters for the `K → ∞` regime (more epochs, less weight decay), the asymptote improves further to ≈3.27 (Figure 5).
  - Significance: Under fixed data, allocating compute to train multiple smaller models that specialize (and then average) outperforms training one huge model. This contrasts with some classic scaling intuitions and is central for a compute‑rich future.

- Joint scaling (parameters + ensemble members) achieves the best ceiling
  - Result: The double‑limit asymptote at 200M tokens is ≈3.17 (Figure 6 right), improving upon the regularized single‑model asymptote 3.43 and the unregularized recipe ≈3.75.

- Distillation compresses most of the ensembling gain into a single small model
  - Result: Distilling an 8‑member 300M ensemble into a single 300M student attains loss 3.36—retaining 83% of the ensemble improvement and even beating the regularized single‑model asymptote (Figure 9; Appendix E.2).
  - Self‑distillation also works if synthetic data is mixed with real: at matched tokens, self‑distill (1:1 mixing) reaches 3.437 vs teacher 3.710; omitting real data collapses to 4.069 (Appendix E.3 Table 4). This links distillation to an “implicit ensembling” view and gives a compute‑heavy path to better small models without training larger models first.

## 5. Experimental Analysis
- Evaluation methodology
  - Datasets and splits
    - Pre‑training: DCLM web data; main seed token counts 200M, plus 400M/800M/1.6B; fixed 4M‑token validation set (Section 2; Appendix A).
    - Downstream: ARC‑Easy, PIQA, SciQ, using standard harness (Section 7.1; Table 5).
    - Continued pre‑training: MegaMath‑Web‑Pro; base model Llama 3.2 3B; restrict to 4B tokens to simulate data scarcity (Section 7.2; Table 1).
  - Metrics
    - Validation loss for scaling‑law analysis; accuracy for downstream and math benchmarks (Sections 5, 7).
  - Baselines/recipes compared
    - Standard (epoching + model scaling, decay=0.1).
    - Regularized model scaling (joint LR/epoch/decay tuning).
    - Ensemble scaling (logit averaging) and joint scaling.
    - Distillation (ensemble→student; self‑distill).
    - For CPT: default hyperparameters vs small batch size + epoching + ensembling (Section 7.2; Table 1).

- Main quantitative results
  - Overfitting of standard recipe under fixed data
    - “Too many epochs” increases validation loss (Figure 2 left), even though train loss declines (Appendix B.5 Figure 15 left).
    - Larger single models at fixed data start to hurt at 1.4B vs 600M in tuned runs (Figure 2 right); train loss keeps falling (Appendix B.5 Figure 15 right), confirming overfitting.
  - Regularized single‑model scaling
    - With tuned large weight decay per `N`, loss falls monotonically with `N` and fits L̂200M,N = 0.05 / N^1.02 + 3.43 (Figure 3).
    - Hyperparameters evolve systematically with scale: as `N` grows, optimal LR and epochs decrease, weight decay increases (Appendix B.3 Figure 12).
  - Ensemble scaling
    - For fixed `N = 300M` with single‑model‑optimal hyperparameters, fit suggests asymptote ≈3.34 (Figure 4).
    - Tuning member hyperparameters for the `K → ∞` regime improves the asymptote to ≈3.27 and flips the hyperparameter ranking relative to single‑model optima (Figure 5). “Double epochs, half weight decay” is the robust rule (Appendix C.2; Figure 17).
    - Either data order or initialization randomness alone suffices to harvest most ensemble gains (Appendix C.1; Figure 16).
  - Joint scaling
    - Double‑limit at 200M tokens: ≈3.17 (Figure 6 right), best among all.
  - Data‑efficiency comparisons across D
    - Data‑scaling fits (Figures 7 right and 8 right) have similar exponents across recipes (~0.23–0.24) and similar infinite‑data asymptotes (~1.89–1.96), implying a roughly constant data‑efficiency multiplier at all scales (Section 5.4).
    - At D = 200M:
      > “Regularized recipe is 2.29× more data‑efficient than the standard recipe; joint scaling is 5.17×” (Sections 5.2–5.3; Figures 7–8).
      - Without extrapolating asymptotes, a 5×1.4B ensemble already delivers 3.75× data efficiency over the standard recipe at 200M (Section 5.3).
  - Distillation
    - Ensemble→student (300M): teacher 8‑ensemble ≈3.32; student 3.36; best regularized single 300M ≈3.57; student surpasses single‑model asymptote (Figure 9; Appendix E).
    - Self‑distill ablation:
      > “Teacher 3.7103; self‑distill with 1:1 real:synthetic 3.4373; self‑distill without real data 4.0693” (Appendix E.3 Table 4).
  - Downstream benchmarks
    - Validation loss improvements translate to accuracy gains (Figure 10: left vs right). Example summary (Table 5, averages):
      > Best unregularized 300M: 58.47 avg; best regularized 1.4B: 60.73; 300M K=5 ensemble: 63.00; 1.4B K=5 ensemble: 64.39. Distilled 300M (ensemble teacher): 62.19; self‑distilled 300M: 60.54.
    - Model soups (weight averaging) perform poorly in pre‑training context (Table 5, ~35% avg), supporting the view that independent pre‑training runs land in different basins (Appendix C.3.2).
  - Continued pre‑training (math)
    - With only 4B tokens:
      > Default CPT: 30.59 avg; small batch only: 34.48; add epoching (K=1): 35.82; K=8 ensemble: 40.58 (Table 1).
      - This exceeds a baseline that used all 73B tokens (39.23), i.e., a 17.5× data‑efficiency gain.
    - In CPT, weight‑averaged soups slightly outperform ensembles as K grows (Appendix G.2; Table 7).

- Ablations, robustness, and diagnostics
  - Batch size: Smaller is better for fixed tokens (e.g., batch 64 best; Figure 13 left), aligning with generalization literature.
  - Weight decay dynamics: High decay slows early improvement but wins decisively by end of training (Figure 14).
  - Sensitivity of power‑law fits: Re‑fitting across seeds for model scaling and sub‑sampling K‑points for ensembles shows small asymptote variance (Appendix H.1; Figure 20), though the authors caution these are rough estimates.

- Do the experiments support the claims?
  - The paper triangulates its core claims with multiple, complementary analyses: overfitting diagnostics (train vs val), controlled hyperparameter searches, clear monotone scaling fits with small residuals, cross‑data‑scale checks, downstream transfer, and a second, independent setting (CPT) where the techniques again help. The asymptote estimation remains extrapolative, but sensitivity checks and many direct points (e.g., K=3 already beating single‑model asymptote; Figure 4) make the qualitative conclusions convincing.

## 6. Limitations and Trade-offs
- Assumptions and scope
  - Fixed‑data, compute‑unconstrained framing: valuable for the emerging regime but not a replacement for compute‑optimal trade‑off studies. Asymptote estimates depend on functional forms and fit ranges (Appendix H).
  - Validation data is i.i.d. from the same distribution; real deployments may care about domain shift or alignment objectives not probed here (Section 7.1 focuses on three standard small‑model benchmarks).
- Algorithmic choices not fully explored
  - Regularization space is vast. The study varies weight decay, epochs, LR, batch size, but not, e.g., dropout, data augmentation beyond distillation, or alternative objectives (Appendix B.4 notes dropout wasn’t tuned; Section 8 suggests many avenues).
  - Alternatives to ensembles:
    - Mixture‑of‑Experts is discussed conceptually and in small internal tests but not deeply benchmarked; intuition is that MoE’s shared learning trajectory reduces the “multi‑view” benefit (Appendix C.3.1).
    - Model soups underperform for pre‑training but worked well for CPT (Appendix C.3.2; G.2), indicating context‑specific behavior that deserves deeper study.
- Architectural caveat
  - The 1.4B configuration is non‑standard (wider, fewer layers; Appendix A Table 2), which might affect relative scaling. The authors argue heavy decay compensates (Appendix B.5), but strict comparability to canonical depth‑scaled models is a caveat.
- Compute and inference costs
  - Ensembles linearly increase inference cost with `K`. Distillation recovers most gains at fixed inference cost, but requires additional training compute and a generation pipeline for synthetic data (Appendix E.1 notes 10B synthetic tokens generated; some teacher data was epoched due to limits).
- Statistical uncertainty
  - Asymptote extrapolations are based on four `N` values (and five `K` values per `N`), and the fits are sensitive in principle to run noise; sensitivity checks are promising but limited (Appendix H.1).

## 7. Implications and Future Directions
- How this changes the landscape
  - It reframes “scaling” for the data‑scarce era: rather than chasing compute‑optimal single models, use regularization to unlock stable scaling, invest compute into ensembles to reach a lower loss ceiling, then distill to deployable size. This pipeline is explicitly designed for the trajectory where compute outpaces data.
  - The asymptote‑centric view gives a principled way to compare recipes when compute is not the bottleneck and encourages designing methods that improve the limiting performance, not just the point‑wise trade‑off.

- Follow‑up research enabled or suggested
  - Regularization: systematic studies of dropout, data augmentation, noise injection, and alternative optimizers under this fixed‑data regime; characterize how the optimal decay scales with parameter‑to‑token ratio (Appendix B.3 suggests ~0.8 when that ratio is fixed).
  - Structured ensembles: beyond independent seeds—diversity‑encouraging training (e.g., negative correlation learning), co‑training, or sub‑architectures to improve “multi‑view” coverage while easing inference via smarter distillation.
  - Asymptote estimation: more rigorous uncertainty quantification for multi‑tier scaling fits (N, K, D), potentially Bayesian fits; evaluating whether equal infinite‑data asymptotes across recipes (Section 5.4) empirically converge to the entropy bound.
  - Alternatives to ensembling: revisit MoE with diversity‑oriented training or dropout; study why CPT soups outperform ensembles and whether that can transfer back to pre‑training.
  - Self‑generated data curricula: The self‑distillation result (Figure 9; Appendix E.3) suggests broader synthetic‑data training that carefully mixes real data to avoid collapse; explore task‑aware sampling and temperature schedules.

- Practical applications
  - Training small and mid‑sized foundation models in data‑poor domains (healthcare, legal, enterprise internal text) by combining strong regularization, ensembles during training time, and distillation for deployment.
  - Continued pre‑training on mid‑training domains (e.g., math, code, safety) with scarce curated tokens: the CPT case shows 17.5× data‑efficiency gains over a 73B‑token baseline using only 4B tokens (Table 1).
  - Budget planning: when compute is cheap but data acquisition is costly or capped, these recipes offer a compute‑centric roadmap to higher quality without new data.

> Representative headline results to carry forward:
> - Regularized single‑model asymptote at 200M tokens: 3.43 (Figure 3).
> - Ensemble member re‑tuning reduces the `K → ∞` asymptote to 3.27 (Figure 5).
> - Joint scaling `N, K → ∞` at 200M tokens: 3.17 (Figure 6 right).
> - Data‑efficiency at 200M: 2.29× (regularized) and 5.17× (joint scaling) over the standard recipe; similar multipliers across higher `D` (Figures 7–8; Section 5.4).
> - Distill 8×300M ensemble → 300M student: loss 3.36, retaining 83% of the ensembling gain (Figure 9; Appendix E).
> - CPT on math with 4B tokens and ensembles beats a 73B‑token baseline (Table 1), a 17.5× data efficiency improvement.

Overall, the work provides a concrete, well‑tested blueprint for the compute‑rich, data‑poor future: regularize hard, ensemble broadly, measure by asymptotes, and distill to deploy.
