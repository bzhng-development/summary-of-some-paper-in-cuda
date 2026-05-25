## 5. Experimental Analysis

### Evaluation Methodology

- **Dataset.** All experiments use the MATH benchmark (Hendrycks et al., 2021), a collection of high-school competition‑level math problems that demand multi‑step logical inference rather than novel factual recall. The authors adopt the specific split released by Lightman et al. (2022): 12,000 training questions and 500 test questions, and answers are graded with the official grading function from Lightman et al. (2022) (Appendix G).  
- **Base model(s).** The core test‑bed is PaLM 2‑S* (Codey) (Anil et al., 2023). The authors argue this model is representative of contemporary LLMs — it attains a non‑trivial but unsaturating pass@1 on MATH (~10–19% depending on prompting and sampling configuration), making it sensitive to test‑time compute interventions. For the FLOPs‑matched comparison, a second model with approximately 14× more parameters (the same family, scaled along the parameter axis only) is used as the pretraining‑scaled baseline.  
- **Metrics.** The primary metric throughout is **MATH test accuracy (%)**: the fraction of the 500 test questions whose final selected answer exactly matches the ground truth, as determined by the grading function. When studying difficulty‑dependent behaviour, accuracy is reported separately within each of the five difficulty quintiles.  
- **Baselines.** The following baselines are used across the two axes (search, revisions):
  - **Majority voting:** select the most frequent final answer among `$N$` sampled solutions, without any learned verifier.
  - **ORM best‑of‑N weighted:** score `$N$` solutions with an outcome reward model and apply best‑of‑N weighted selection (Cobbe et al., 2021; Li et al., 2023).
  - **PRM best‑of‑N weighted:** score `$N$` solutions with a process reward model and apply best‑of‑N weighted selection.
  - **Parallel sampling (revisions setting):** generate `$N$` independent answers from the revision model and select via verifier or majority.
  - **Sequential revisions (revisions setting):** generate a chain of `$N$` revisions from the revision model and select the best answer within the chain.
- **Generation budget / compute accounting.** The universal unit of test‑time compute is a **generation** — one complete sampled answer from the base LLM. For best‑of‑N and beam search the budget equals the number of beams or samples `$N$`. For lookahead search with `$k$` lookahead steps, the budget is `$N \times (k+1)$` to account for the extra rollout tokens. Budgets are swept in powers of two, typically from `$2^0$` to `$2^9$` (1 to 512 generations); a cap of 256 is used for most search sweeps.  
- **Cross‑validation / statistical protocol.** To avoid contaminating strategy selection with test‑set performance, a two‑fold cross‑validation is performed **within each difficulty bin** on the 500‑question test set. For each bin and budget, the best strategy is selected on one fold and evaluated on the other, then folds are swapped and the results averaged. This yields an unbiased estimate of the compute‑optimal policy’s accuracy while still using the test set for both selection and evaluation.

### Main Quantitative Results

#### PRM Search Results

The central finding for verifier‑guided search is that **beam search significantly outperforms best‑of‑N at low generation budgets, but its advantage diminishes or reverses at high budgets due to PRM over‑optimisation**, and the relative efficacy depends on problem difficulty. The compute‑optimal policy that selects the best search algorithm per difficulty bin yields up to **4× efficiency gains** over standard PRM best‑of‑N.

**Aggregate search method comparison (Figure 3, left).** Across all 500 test questions with budgets up to 256 generations:

- At 2–8 generations, beam search with a fixed beam width of `$M = 4$` reaches roughly 27% accuracy compared to roughly 16% for PRM best‑of‑N weighted — a wide gap. Beam search with `$M = \sqrt{N}$` performs similarly at low budgets but trails the fixed‑width variant.
- As the budget increases, beam search performance flattens: at 256 generations, beam search (`$M = 4$`) reaches approximately 34% while PRM best‑of‑N weighted climbs to approximately 37% and continues to improve at higher budgets (reaching ~38% at 512). Beam search with `$M = \sqrt{N}$` plateaus even earlier.
- Lookahead search (both `$k = 1$` and `$k = 3$`) underperforms all other methods at the same generation budget; at 256 generations, 3‑step lookahead with `$M = 4$` scores roughly 32–33%, below both best‑of‑N and plain beam search. The extra rollout cost prevents lookahead from exploring enough beams to compensate for more accurate per‑step scoring.
- Majority voting (no verifier) trails all verifier‑based methods substantially, reaching only about 29% at 512 generations.

**Difficulty‑bin analysis for search (Figure 3, right).** The paper breaks down beam search (`$M = 4$`) vs. PRM best‑of‑N weighted across five difficulty bins, each showing four bars per bin for budgets of 4, 16, 64, and 256 generations. This is where the most diagnostic pattern emerges:

- **Bin 1 (easiest):** Beam search accuracy *decreases* from roughly 78% to 77% as the budget grows from 4 to 256 generations, while best‑of‑N weighted improves from 68% to 88%. This is direct evidence of **PRM over‑optimisation** — the aggressive search amplifies spurious verifier signals on problems where the base model already samples correct answers frequently.
- **Bin 2:** Beam search improves from ~14% to ~32% but best‑of‑N weighted accelerates faster (~14% → ~60%), opening a large gap at high budgets.
- **Bin 3 (medium difficulty):** Beam search consistently outperforms best‑of‑N across all budgets, reaching ~34% vs. ~23% at 256 generations.
- **Bin 4:** Beam search shows the strongest relative advantage, reaching ~17% vs. ~10% for best‑of‑N at 256 generations — search genuinely helps when the correct answer is in the base model’s distribution but rare.
- **Bin 5 (hardest):** Both methods remain near 1–3% accuracy regardless of budget. No test‑time search method helps; the base model essentially never generates a correct solution.

**Compute‑optimal search (Figure 4).** When the best search strategy (best‑of‑N weighted, beam search with either beam‑width setting, lookahead variants) is selected per difficulty bin for each budget:

- At 16 generations, compute‑optimal oracle (difficulty bins derived from ground‑truth correctness) achieves ~27% accuracy, nearly matching PRM best‑of‑N weighted at 64 generations (~27–28%) — a **4× reduction in required generations**.
- At 256 generations, compute‑optimal oracle reaches ~39.5%, surpassing PRM best‑of‑N weighted at the same budget (~37%). The gap continues to widen.
- Compute‑optimal with **predicted** difficulty bins (PRM final‑answer score averaged over 2048 samples) tracks the oracle curve closely, especially at low budgets. At 256 generations, predicted bins reach ~37%, only slightly below the oracle. The overlap between oracle and predicted curves demonstrates that ground‑truth answers are not needed for effective adaptation.
- Both compute‑optimal variants consistently dominate the weaker baselines: ORM best‑of‑N weighted (~34% at 512) and majority voting (~29% at 512).

#### Revision Model Results

The headline for the proposal‑distribution axis is that **sequential revisions outperform parallel sampling in aggregate, but the optimal sequential‑to‑parallel ratio depends on difficulty: easy questions do best with purely sequential revisions, while hard questions need a mix**. The compute‑optimal policy that adapts this ratio per difficulty bin yields up to **4× efficiency gains** over parallel best‑of‑N.

**Revision model pass@1 trajectory (Figure 6, left).** Starting from ~18.2% pass@1 at step 1, the revision model’s per‑step accuracy improves to ~24–25% by steps 15–20 and remains in the 23–25% range out to 64 steps. The model generalises beyond the 4‑step training horizon, showing that it has learned a transferable revision skill.

**Sequential vs. parallel at equal budgets (Figure 6, right).** At 64 generations:
- Sequential + best‑of‑N weighted (revision‑specific ORM): ~41.5%
- Parallel + best‑of‑N weighted: ~39%
- Sequential + majority voting: ~38%
- Parallel + majority voting: ~35%

Sequential revisions modestly but consistently outperform parallel sampling under both selection mechanisms (verifier and majority).

**Sequential‑to‑parallel ratio sweep (Figure 7, left).** For a fixed generation budget, varying the ratio of sequential to parallel compute reveals:
- At 256 generations, the optimal ratio lies between `$2^1$` and `$2^3$` (~2:1 to 8:1 sequential‑to‑parallel), achieving ~43–44% accuracy. Fully parallel (leftmost) yields ~40%; fully sequential (rightmost) yields ~42%.
- At lower budgets (8–32), the curves are monotonically increasing with the sequential‑to‑parallel ratio — fully sequential is optimal when the total compute is small.

**Difficulty‑dependent ratio (Figure 7, right).** At 128 generations with verifier selection, the breakdown by difficulty bin shows:
- **Bin 1:** Performance is nearly flat across all ratios at ~90–92%. Easy questions are insensitive.
- **Bin 2:** A slight advantage for higher sequential ratios: ~63% at fully sequential vs. ~58% at fully parallel.
- **Bin 3:** A clear optimum at moderate sequential‑to‑parallel ratios (~`$2^1$`–`$2^3$`), reaching ~42% vs. ~35% at the extremes.
- **Bin 4:** Similar pattern with a peak at ~18% vs. ~14% at the extremes.
- **Bin 5:** All ratios produce roughly 2–3% accuracy.

**Compute‑optimal revisions (Figure 8).** Selecting the best sequential‑to‑parallel ratio per difficulty bin:
- At 64 generations, compute‑optimal oracle achieves ~40%, matching parallel best‑of‑N weighted at 256 generations — a **4× reduction**.
- At 256 generations, compute‑optimal oracle reaches ~44%, compared to ~41% for parallel best‑of‑N weighted and ~37% for a parallel‑only baseline. Notably, the parallel baseline appears to plateau at high budgets while compute‑optimal scaling continues to improve.
- Predicted difficulty bins (PRM‑based) perform slightly below oracle at high budgets (~41% at 256) but still substantially outperform the parallel baseline across the full range.

#### FLOPs‑Matched Comparison: Test‑Time vs. Pretraining Compute

The core question of Section 7 is: given a fixed total FLOPs budget, should we scale pretraining or inference compute? The experiments compare PaLM 2‑S* with compute‑optimal test‑time scaling against a ~14× larger model (greedy decoding, no extra test‑time compute), parameterised by `$R = D_{\text{inference}} / D_{\text{pretrain}}$`.

**Revisions axis (Figure 9, left; Figure 1, top‑right bar chart):**

| Difficulty grouping | `$R \ll 1$` (0.16) | `$R \approx 1$` (0.79) | `$R \gg 1$` (22) |
|---|---|---|---|
| Easy (bins 1–2) | +11.8% | +3.5% | −11.9% |
| Medium (bins 3) | +27.8% | +16.7% | +5.4% |
| Hard (bins 4–5) | +21.6% | (implied negative) | −37.2% |

At `$R \ll 1$`, test‑time compute **outperforms the 14× larger model across all difficulty levels**. At `$R \gg 1$`, test‑time compute remains preferable only on easy‑to‑medium questions; on hard ones it shows a −37.2% relative deficit, meaning pretraining is a better investment.

**PRM search axis (Figure 9, right; Figure 1, bottom‑right bar chart):**

| Difficulty grouping | `$R \ll 1$` (0.16) | `$R \approx 1$` (0.79) | `$R \gg 1$` (22) |
|---|---|---|---|
| Easy | +19.1% | +2.2% | +2.0% |
| Medium | 0.0% | −35.3% | −30.8% |
| Hard | −3.6% | −35.3% | −52.9% |

PRM search provides weaker benefits than revisions in the FLOPs‑matched comparison. At `$R \ll 1$`, it still wins on easy questions but is essentially flat on medium and slightly negative on hard. At higher `$R$`, the larger model dominates across all but the easiest questions.

**Line‑plot interpretation (Figure 9).** The larger model’s greedy‑decoding performance (stars) is placed at three x‑axis positions corresponding to the three `$R$` values, representing the FLOPs‑equivalent inference budget the smaller model can use. If the scaling line for the smaller model lies above the star, test‑time compute wins. On bin 1 (purple, topmost line for revisions), the scaling line is above all three stars; on bin 5 (blue, bottommost), it is essentially flat near 0–5% and always below the stars. This visual makes the conditional nature of the tradeoff immediately clear.

### Ablation Studies and Robustness Checks

- **PRM step‑wise aggregation strategy (Appendix E, Figure 13):** Comparing “min” (minimum score across steps), “prod” (product of step probabilities), and “last” (final‑step prediction) on 256 samples: “last” achieves ~37%, “min” ~35%, “prod” ~27%. Using “last” is effectively equivalent to an ORM at aggregation time, yet the PRM outperforms a separately trained ORM (~34%), implying that the step‑level training serves as a form of **representation learning** even when intermediate predictions are not directly exploited for final scoring.
- **PRM vs. ORM best‑of‑N weighted (Appendix F, Figure 14):** At 2048 samples, PRM best‑of‑N weighted reaches ~40% vs. ORM’s ~35%. The gap widens with more samples, confirming the PRM’s superior scaling behaviour and justifying its use throughout the search experiments.
- **Revision model verifier: base‑LM PRM vs. revision‑specific ORM (Appendix J, Figure 15a):** The PRM trained on PaLM 2‑S* base model outputs underperforms when applied to the revision model’s outputs — at 64 generations, sequential + base‑LM PRM achieves ~40% vs. ~42% for sequential + revision‑specific ORM. This confirms that **distribution shift** is a genuine concern and motivates training a separate verifier for the revision setting.
- **Revision history in the verifier’s context (Appendix J, Figure 15b):** Including previous revisions in the ORM’s input yields a small gain (~1–2 percentage points at 64 generations) over an ORM that sees only the current answer. However, even the no‑history ORM combined with sequential revisions outperforms the parallel baseline, demonstrating that the benefit of sequential sampling is not solely due to the verifier having more context.
- **Oracle vs. predicted difficulty bins (Figures 4 and 8, and Appendices C, 11–12):** In both the search and revision settings, predicted bins (PRM average final‑answer score) produce difficulty‑dependent trends that are qualitatively identical to oracle bins and nearly overlap quantitatively at low‑to‑medium budgets. In the search setting (Figure 4), the two curves are almost indistinguishable; in the revisions setting (Figure 8), predicted bins show slightly lower performance at high budgets (~41% vs. ~44% at 256 generations), but still yield substantial gains over the parallel baseline. This is the critical robustness check: the compute‑optimal policy works without ground‑truth labels.
- **Majority voting for revisions (Appendix B, Figure 10):** When majority voting replaces the verifier for answer selection in the revision setting, the sequential‑to‑parallel trends persist: there exists an optimal ratio per difficulty bin, and fully sequential outperforms fully parallel in aggregate. This shows that the revision model’s benefit from sequential sampling is robust to the choice of selection mechanism.
- **ReST$^{\text{EM}}$‑optimised revision model (Appendix K, Figure 16):** An attempt to further improve the revision model by applying the ReST$^{\text{EM}}$ RL‑style training procedure backfires dramatically. With this model, additional sequential revisions **degrade** performance: at 256 generations, fully sequential drops to ~33.5% compared to ~38.5% at the optimal ratio. The authors hypothesise that on‑policy data collection in ReST$^{\text{EM}}$ exacerbates spurious correlations in revision trajectories, causing the model to fail to learn a meaningful revision skill. This negative result highlights the sensitivity of revision training to data generation methodology and underscores that the positive results depend on the offline, edit‑distance‑based procedure described in Section 6.1.

### Critical Assessment

The experiments largely support the paper’s central claims, but with important boundary conditions and some methodological limitations that temper the strength of the conclusions.

**Claim: compute‑optimal allocation yields >4× efficiency gains over best‑of‑N.** The evidence for this claim is strong but **bounded to the specific budget regimes tested**. In the PRM search setting, Figure 4 shows that compute‑optimal scaling at 16 generations (~27%) nearly matches best‑of‑N at 64 generations (~27–28%), a factor of 4. In the revision setting, Figure 8 shows compute‑optimal at 64 generations (~40%) matching parallel best‑of‑N at 256 generations (~40–41%), again a factor of 4. However, the 4× figure is most reliable at low‑to‑moderate budgets; at higher budgets (256–512), the relative advantage narrows when using predicted difficulty bins (Figure 8: predicted bins at 256 generations reach ~41% vs. ~44% for oracle, while best‑of‑N at 1024 is not shown). Moreover, the cost of estimating difficulty — generating 2048 samples per question to bin it — is **not amortised into any budget calculation**. The paper explicitly acknowledges this omission (Section 3.2). If that estimation cost were included, the actual efficiency gain over a baseline that does not pre‑bin problems would be substantially smaller, and for low‑generation‑budget regimes the pre‑binning cost could even dominate. Thus the 4× number should be understood as an **upper bound** on achievable gains once a cheap difficulty estimator is available, not as a realised deployment improvement.

**Claim: test‑time compute with a smaller model can outperform a ~14× larger pretrained model.** The FLOPs‑matched comparison backs this claim **conditionally**. On easy‑to‑medium questions at `$R \ll 1$`, test‑time compute clearly wins (+11.8% to +27.8% relative improvement for revisions). On hard questions, the claim **fails** — the smaller model with test‑time compute is **systematically worse**, often by large margins (−37.2% for revisions, −52.9% for PRM search at `$R \gg 1$`). The paper is transparent about these boundaries, which strengthens credibility. However, two experimental design choices make the comparison **somewhat skewed in favour of test‑time compute**: (1) The ~14× larger model scales parameters only, not data (the LLaMA paradigm), which is known to be suboptimal compared to Chinchilla‑optimal scaling where data and parameters are grown together. A properly compute‑optimal larger model would likely be a stronger baseline. (2) The larger model uses **greedy decoding** with no test‑time compute of its own — no majority voting, no best‑of‑N, no search. A fairer match would give the larger model at least a modest test‑time compute allowance (e.g., best‑of‑8). The current comparison therefore tends to overstate the advantage of test‑time compute relative to a more realistic deployment of a larger model.

**Claim: the effectiveness of different test‑time strategies depends critically on prompt difficulty.** This is **the most robustly supported claim in the paper**, replicated across both search methods (Figure 3, right) and revision strategies (Figure 7, right), and observed with both oracle and predicted difficulty bins (Figures 4, 8, 11, 12). The patterns are striking and consistent: beam search hurts easy problems, helps medium problems; sequential revisions excel on easy problems, mixed strategies on hard ones; no method helps on the hardest problems. The difficulty‑bin concept is validated as a sufficient statistic for strategy selection. The main weakness is the **coarseness** of five bins — performance heterogeneity within a bin is not measured, and a fine‑grained continuous policy might extract additional gains.

**Claim: verifier over‑optimisation is the primary bottleneck preventing further scaling of test‑time compute via search.** The evidence for this claim is **suggestive rather than exhaustive**. Figure 3 (right) clearly shows beam search degrading on easy problems (bins 1–2) as the budget increases — a signature of over‑optimisation. Lookahead search, the most powerful method, underperforms overall (Figure 3, left), and qualitative examples show degenerate outputs (repetitive steps, overly short solutions) that score highly under the PRM (Appendix M). While these observations are consistent with over‑optimisation, the paper does **not** run a specific experiment to decouple verifier quality from search breadth (e.g., training PRMs of varying strength and measuring the point of degradation). The claim that “verifier robustness is the primary bottleneck” is thus an inference drawn from behavioural patterns rather than a controlled causal demonstration. Future work that ablates PRM capacity, calibration, or adversarial training would strengthen this claim.

**Additional methodological weaknesses.** (1) **Single benchmark, single model family:** all experiments are on MATH with PaLM 2‑S*. While the authors argue this is “representative,” there is no replication on other models (e.g., GPT‑class, open‑source models) or other reasoning domains (code, logical reasoning). The observed difficulty‑dependent patterns may be sensitive to the base model’s calibration and the dataset’s structure. (2) **Small test set:** 500 questions split into five difficulty quintiles of ~100 each, further split by two‑fold cross‑validation, means the compute‑optimal strategy is selected based on ~50 questions per fold per bin. No confidence intervals are reported for the compute‑optimal scaling curves, making it unclear whether observed differences between oracle and predicted bins at high budgets are statistically reliable. (3) **No combination of PRM tree‑search with revisions:** the two axes are studied independently except for a limited combination where a separate ORM is used to select among revision chains. True integration — applying PRM‑guided beam search on top of the revision model’s output — is not explored, so the reported results are a lower bound on what a fully combined system might achieve. (4) **The revision model’s 38% correct‑to‑incorrect reversion rate** is a fundamental limitation that the current mitigation (cross‑chain selection) only partially addresses; a model trained to stabilise on correct answers was not explored. (5) **Latency and serialisation are not considered:** sequential revisions impose a wall‑clock delay that is absent from fully parallel sampling, making the reported efficiency gains potentially less attractive for interactive, latency‑sensitive applications.