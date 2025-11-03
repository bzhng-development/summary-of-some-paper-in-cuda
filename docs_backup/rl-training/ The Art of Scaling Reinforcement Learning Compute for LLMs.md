# The Art of Scaling Reinforcement Learning Compute for LLMs

**ArXiv:** [2510.13786](https://arxiv.org/abs/2510.13786)
**Authors:** Devvrit Khatri, Lovish Madaan, Rishabh Tiwari, Rachit Bansal, Sai Surya Duvvuri, Manzil Zaheer, Inderjit S. Dhillon, David Brandfonbrener, Rishabh Agarwal
**Institutions:** Meta, University of Texas at Austin, University College London (UCL), University of California, Berkeley, Harvard University, Periodic Labs

## 1. Executive Summary (2–3 sentences)

Reasoning: The paper tackles a concrete gap—predictively scaling RL compute for LLMs—by building both a model of how performance responds to compute and a practical training recipe, then stress-testing them at very large scales. The core contribution is a framework that separates “how high a method can go” from “how fast it gets there,” enabling early-stage forecasts of large-scale behavior.

This paper defines a predictive, sigmoidal compute–performance relationship for RL on LLMs (Equation (1)) and uses it to analyze which training choices raise the ultimate performance ceiling versus which mainly improve efficiency. It consolidates the most robust choices into `ScaleRL`, then validates both predictability and stability in extended runs up to 100,000 GPU-hours (Figure 1), achieving higher asymptotic performance than popular RL recipes (Figure 2).

## 2. Context and Motivation

Reasoning: RL post-training has exploded in compute but lacks the kind of “scaling laws” that guide pre-training. Without a predictable methodology, algorithmic progress depends on expensive trial-and-error. The paper seeks to replace the “art” of RL scaling with a scientific framework that can be applied early, cheaply, and across recipes.

- Problem and gap addressed:
  - RL is now central to unlocking advanced LLM capabilities (test-time reasoning, agentic behavior), yet there is no principled way to predict how an RL method scales with compute or to compare methods without running very large experiments.
  - The paper explicitly frames this gap: 
    > “We present the first large-scale systematic study… that defines a principled framework for analyzing and predicting RL scaling in LLMs.” (Abstract; also Figure 1)
- Importance:
  - Real-world impact: RL compute is skyrocketing—e.g., DeepSeek-R1-Zero reportedly used 100,000 H800 GPU-hours on RL alone (Introduction, p.1).
  - Theoretical significance: Pre-training has stable scaling laws (Kaplan et al., Hoffmann et al.), but RL lacks analogous predictability (Introduction, p.2).
- Prior approaches and limitations:
  - Algorithm-specific reports (e.g., `GRPO`, `DAPO`, `VAPO`, “ProRL”, “Magistral”, “Minimax M1”) provide techniques and tricks, but no generalized scaling methodology (Introduction, §Related Work; Appendix A.1, A.16).
  - Practical pain point: instability at higher compute (Appendix A.6, A.15) and lack of early indicators of which method will win at scale (“Bitter Lesson” observation, Figure 2, §3).
- Positioning relative to existing work:
  - Introduces a bounded, sigmoidal fit tailored to accuracy/pass-rate metrics (Equation (1), §2.1; Appendix A.4) and shows robust extrapolation from mid-scale compute to extreme scales (Figure 1).
  - Distinguishes ceiling-shifting decisions (`A`) from efficiency-shifting ones (`B`, `Cmid`) and provides an RL recipe (`ScaleRL`) that is both predictable and high-performing (Figure 2; §4, §5).

## 3. Technical Approach

Reasoning: The authors build the framework step-by-step: define a measurable setting with held-out validation, explain the RL training setup, propose the sigmoidal scaling law, and test design knobs through ablations that are scaled up only when stable. Mechanistically, the approach separates the data pipeline, the off-policy method, the loss formulation, and control of generation length, then studies how each affects scaling parameters.

- Problem setting and data:
  - Domain: RL for reasoning with thinking traces (`<think> ... </think>`) and a final solution (§2).
  - Validation: 1,000 held-out prompts from the Polaris-53k math dataset; pass-rate measured as mean correctness with 16 generations per prompt every 100 training steps (§2.1, “Scaling curve on held-out validation”).
- Training pipeline:
  - Generator–trainer split across GPUs: generators produce rollouts at high throughput; trainers update parameters using FSDP (distributed training backend) (§2).
  - Policies: `πθ_gen` (generator side) and `πθ_train` (training side). Off-policy learning uses stale rollouts from the old `πθold_gen` (§2).
- Asynchronous off-policy algorithms:
  - `PPO-off-policy-k`: process each batch of size `B` with `k = B / B̂` gradient updates of mini-batches `B̂ = 48` (16 generations/prompt) (§3.1).
  - `PipelineRL-k`: streaming generation where trainers push updated weights back immediately; generators continue with a stale KV cache but updated weights. Trainers wait if they get `k` steps ahead (§3.1).
  - Empirical effect: `PipelineRL-k` is significantly more compute-efficient (`B`) and slightly improves asymptotic performance (`A`) over `PPO-off-policy-k` (Figure 4a). Best `k` found is 8 (Figure 4b; Appendix A.11).
- Base RL objective and losses:
  - Start from a `GRPO`-like baseline without KL control; use asymmetric `DAPO` clipping to avoid entropy collapse (§2, Eq. (2), (3)).
  - Token importance sampling ratio `ρi,t(θ) = πθ_train(yi,t)/πθold_gen(yi,t)` with asymmetric clipping `clipasym(ρ, ϵ−, ϵ+)` (Eq. (2)). Advantages are group-centered and optionally normalized (Eq. (2); §2).
  - Baseline surrogate objective averages token-level clipped terms at the sample level (Eq. (3)).
- Controlling generation length:
  - Forced interruptions: insert an end-of-thinking phrase to stop overly long generations (e.g., “</think>”), improving stability and efficiency (§2; Appendix A.10).
  - Compared against length penalty that subtracts reward for long correct generations (Appendix A.10). At scale, interruptions were preferred.
- Predictive scaling formulation:
  - Sigmoidal law for pass-rate versus compute (`C`):
    - `RC − R0 = (A − R0) × 1/(1 + (Cmid/C)^B)` (Equation (1), §2.1; Figure 3).
    - Parameters:
      - `A`: asymptotic pass-rate ceiling.
      - `B`: scaling exponent controlling steepness/efficiency.
      - `Cmid`: compute at half total gain (midpoint). Figure 3 explains each.
  - Fitting regime: skip early low-compute (∼1.5k GPU-hours) for stability; grid search over `A` and `Cmid`, then fit `B` (Appendix A.5). Fits are robust across regimes for stable recipes (Appendix A.7).
  - Error margins: three independent runs yield ±0.02 error on `A` (Figure 8a), used to assess meaningful differences (§4).
  - Relation to power-laws: sigmoid approximates a power-law in the high-compute regime (Appendix A.4).
- Experimental protocol:
  - Three stages (§3): (1) ablate design choices at 3.5–4k GPU-hours to screen for stability; (2) combine best choices into `ScaleRL` and run leave-one-out (LOO) at 16k GPU-hours (§4); (3) scale axes—batch, context length, model size, multi-task—and validate extrapolation (§5).

## 4. Key Insights and Innovations

Reasoning: The most novel elements are the predictive scaling framework, how it decomposes performance and efficiency, and the empirical finding that seemingly important tricks often only move efficiency—not the ceiling. The practical innovation is `ScaleRL`, a recipe that exhibits predictable behavior across large scales and axes.

- A predictive, bounded compute–performance law for RL:
  - Novelty: Use a sigmoidal fit tailored to bounded metrics (pass-rate), more stable than power-laws in RL settings (Equation (1); Appendix A.4).
  - Why it matters: Enables extrapolation from small/mid-scale runs to extreme compute, reducing cost and risk.
  - Evidence: Extended 100k GPU-hour runs closely match extrapolated curves fitted up to 50k GPU-hours (Figure 1a). Downstream AIME-24 trends also follow the fitted trajectory (Figure 1b).
- Asymptote versus efficiency: disentangling `A` from `B` and `Cmid`
  - Insight:
    > “Details such as loss aggregation, normalization, curriculum, and off-policy algorithm primarily modulate compute efficiency without materially shifting the asymptote.” (Abstract bullets; elaborated in Figure 7 and Appendix A.8/A.9)
  - Why it matters: Choose ceiling-raising changes first (loss type, precision fixes, longer generation length, larger model/batch), then optimize for efficiency. Avoid being misled by early faster gains in less scalable recipes (Figure 13b).
- `PipelineRL` for streaming off-policy updates:
  - Innovation: Near-on-policy update cadence (updated weights pushed immediately; stale KV cache only for partial traces).
  - Impact: Substantial improvement in compute efficiency `B` and slight lift in `A` (Figure 4a). This design reduces idle time and mismatch between generator and trainer distributions (Appendix A.11).
- Robust loss and precision choices:
  - `CISPO` loss: truncated importance sampling with REINFORCE-style gradient and stop-gradient (Eq. (4)).
    - Improves `A` vs `DAPO` and shows prolonged near-linear gains (Figure 5a).
    - Robust to clipping hyperparameters (Appendix A.17.2), unlike `DAPO` where `ϵmax` changes the asymptote itself (Figure 19a).
  - FP32 logits at LM head:
    - Mechanism: Prevents generator–trainer numerical mismatches that corrupt importance ratios.
    - Impact: Large jump in asymptotic `A` (from 0.52 to 0.61; Figure 5b).
- `ScaleRL`: a predictable, stable recipe, not a new algorithm
  - Composition: `PipelineRL-8`, forced interruptions, `CISPO`, prompt-level loss averaging, batch-level advantage normalization, FP32 logits, zero-variance filtering, and `No-Positive-Resampling` (§4).
  - Validation: LOO experiments show `ScaleRL` is consistently the most efficient while keeping similar `A` across variants (Figure 7), and extrapolated fits match extended training points.

## 5. Experimental Analysis

Reasoning: To evaluate claims, we need to see (1) the measurement setup, (2) rigorous baselines, (3) quantified comparisons, and (4) stress-tests across axes. The paper provides all four, along with ablations and error margins, making its conclusions credible within the chosen domain.

- Evaluation methodology:
  - Dataset: Polaris-53K math for training; 1,000 prompts held out for validation (§2.1).
  - Metrics: Mean pass-rate (mean@16 generations), measured every 100 steps (§2.1).
  - Compute accounting: Curve fits begin after ∼1.5k GPU-hours to avoid noisy early regime (§2.1; Appendix A.6/A.7).
  - Verification: Automated checkers (SymPy or Math-Verify) for math; custom execution for code (Appendix A.3).
- Baselines compared:
  - `GRPO` (DeepSeek-like), `DAPO` (Qwen2.5-like), `Magistral` (DAPO + `PipelineRL`), `MiniMax-M1` (`CISPO`) (Figure 2; Appendix A.16 for recipe details).
- Main quantitative results:
  - Predictability at extreme compute:
    - Fitting up to 50k GPU-hours and extrapolating to 100k GPU-hours: extrapolated curve tracks actual training (Figure 1a). Scout MoE (17B×16) shows similar behavior up to 50k (Figure 1a).
    - Downstream AIME-24 generalization mirrors in-distribution scaling (Figure 1b).
  - Off-policy setup:
    - `PipelineRL-k` outperforms `PPO-off-policy-k` in efficiency (`B`) and slightly in asymptote (`A`) (Figure 4a). Optimal off-policyness `k=8` (Figure 4b).
  - Loss choice:
    - `CISPO` and `GSPO` lift `A` over `DAPO` (Figure 5a). `CISPO` chosen for robustness and later-stage gains. `GSPO` shows instability in some runs; `CISPO` is more stable (Appendix A.17.4).
  - FP32 precision at LM head:
    - Increases asymptotic `A` from 0.52 to 0.61 (Figure 5b).
  - Aggregation, normalization, and filtering:
    - Prompt-level loss averaging yields best `A` (Appendix A.9, Figure 14a).
    - Batch-level advantage normalization is marginally better than alternatives, but differences are small (Appendix A.9, Figure 14b).
    - Zero-variance filtering (drop prompts with identical rewards) improves `A` (Figure 6a).
    - `No-Positive-Resampling` (drop prompts whose pass-rate ≥ 0.9) improves scalability and asymptote (Figure 6b).
  - `ScaleRL` LOO ablations:
    - When reverting each component one-by-one, most variants reach similar `A` (within ±0.02 error, Figure 7), but `ScaleRL` achieves highest efficiency `B` overall. The slope visualization via `F(Rc)` transform makes differences in `B` clear (Figure 7 caption).
  - Multi-axis scaling:
    - Context length (generation budget): 32k tokens slows early progress (lower `B`, higher `Cmid`), but raises asymptote (`A`) and eventually wins (Figure 9a,b). Table 1 reports `ScaleRL-32k`: `A=0.645`, `B=1.89`, `Cmid=11272`.
    - Batch size: Larger batches start slower but end higher (inverse trend early); they avoid downstream stagnation seen in smaller batches (Figure 10a,b). Table 1: `bs2048` `A=0.645` vs `bs512` `A=0.605`.
    - Model size: Scout 17B×16 MoE has much higher `A=0.710` while remaining predictable/stable (Figure 1a; Table 1).
    - Generations per prompt (fixed total batch): Largely second-order within moderate ranges (Appendix A.13; Table 1 shows `A≈0.585–0.595` across 8/24/32 gens).
    - Multi-task (math + code): Clean, parallel fits for math and code. Extended training aligns with extrapolations (Figure 11). Table 1 lists separate math/code curves.
- Do the experiments support the claims?
  - Convincing evidence for predictability: Extrapolated curves consistently match extended training across axes and scales (Figure 1, Figure 9, Figure 10, Figure 11).
  - Strong comparative baselines and ablations: Multiple popular recipes re-implemented (Appendix A.16); leave-one-out isolates contributions (Figure 7).
  - Error bars: Explicit variance analysis (Figure 8a) provides a decision criterion for meaningful `A` differences.
- Robustness checks and failure cases:
  - Early low-compute regime excluded for fitting, justified in Appendix A.7; sigmoids are more robust than power-laws in this setting (Appendix A.4).
  - Instabilities tied to truncations: destabilization when truncations rise to 10–15%; `ScaleRL` keeps truncations low and stable (Appendix A.15).
  - Loss stability: `GSPO` sometimes diverges mid-training; `CISPO` is stable and less sensitive (Appendix A.17.4).
  - Entropy: Not a reliable predictor of downstream performance; both small and large batch runs show similar entropy trajectories despite performance differences (Appendix A.12, Figure 16).

## 6. Limitations and Trade-offs

Reasoning: The framework is strong, but it is also specific—bounded metrics, held-out in-distribution validation, particular domains and reward schemes. The predictability claim is convincing within those constraints; generalization beyond them is plausible but not proven.

- Assumptions baked into the framework:
  - Bounded metric (pass-rate/accuracy) suits a sigmoid; unbounded rewards or multi-turn interactive tasks may need adaptations (Appendix A.4).
  - Held-out validation prompts are in-distribution; downstream generalization is observed qualitatively but not the primary target (§7; Figures 1b, 9b, 10b, 18).
  - Excludes early low-compute regime to stabilize fits (∼1.5k GPU-hours), which assumes access to modest compute (§2.1; Appendix A.7).
- Scope limitations:
  - Domain focus: verifiable math (Polaris-53K) with some code multi-task runs; other RL settings (dialogue, agentic tasks, multi-turn) are not studied here (§7 Discussion; Appendix A.1).
  - Reward design: binary ±1 rewards for math correctness; structured/dense rewards are suggested as future axes (§7 Future Work).
  - Loss family coverage: `CISPO`, `GSPO`, `DAPO` studied; value-based methods (e.g., `VAPO`) are referenced but not deeply analyzed under this scaling framework.
- Computational constraints:
  - Large-scale experiments (400k GPU-hours total; single runs up to 100k GPU-hours) are still expensive; smaller labs may need to adopt the early-fit extrapolation practice but cannot fully reproduce extreme-scale validations (§1 Figure 1; §3 Methodology).
  - FP32 logits introduce compute overhead; trade off between numerical stability and speed (Figure 5b).
- Training trade-offs:
  - Longer generation length raises `A` but lowers early efficiency (`B`) and increases `Cmid`; requires sufficient compute (Figure 9).
  - Larger batch sizes similarly slow early steps but lift `A` and improve downstream consistency (Figure 10; Appendix A.12 on entropy).
  - `No-Positive-Resampling` excludes prompts that become too easy; improves asymptote but reduces data diversity (Figure 6b).
  - Interruptions vs length penalty: interruptions worked better at scale here (Appendix A.10), but this could be task/model dependent.

## 7. Implications and Future Directions

Reasoning: The paper changes how we evaluate RL recipes: rather than betting big on expensive full-scale runs, we can measure a few thousand GPU-hours, fit a curve, and forecast which method will likely win at 100k GPU-hours. It also clarifies which knobs truly lift ceilings, guiding compute allocation.

- Field-level impact:
  - Provides a practical, predictive methodology for RL scaling, analogous to pre-training scaling laws:
    > “Stable, scalable recipes follow predictable scaling trajectories, enabling extrapolation from smaller-scale runs.” (Abstract; Figure 1)
  - Separates ceiling-raising (`A`) decisions from efficiency (`B`, `Cmid`) decisions—an actionable perspective for researchers and engineers (Figure 3; Figure 7; Appendix A.8).
- What this enables next:
  - Early-phase decision making: Fit curves after a small fraction of the planned budget and forecast returns—decide whether to double batch size or extend context length based on which raises `A` more.
  - More principled algorithm ablations: Use the ±0.02 `A` margin (Figure 8a) to judge if a change truly lifts the ceiling or only improves efficiency.
  - Standardized reporting: Encourage RL system cards to include sigmoidal fit parameters (`A`, `B`, `Cmid`) alongside downstream scores.
- Practical applications:
  - Building stable, scalable RL pipelines for LLMs in production: Adopt `ScaleRL` defaults (PipelineRL streaming, `CISPO`, FP32 logits, prompt-averaged loss, batch-level normalization, zero-variance filtering, interruptions, `No-Positive-Resampling`) (§4).
  - Compute budgeting: Use fitted `Cmid` to anticipate when half of the attainable gain will be reached and decide whether more compute is justified (Figure 3; Table 1).
- Future research:
  - Cross-axis scaling laws: Jointly model pre-training compute, RL compute, model size, and RL data size/quality (§7 Future Work).
  - Broader tasks and rewards: Study multi-turn RL, agentic interaction, and long-form reasoning; incorporate structured/dense rewards and generative verifiers (Setlur et al., Zhang et al.; §7 Future Work).
  - Methodological extensions: Test whether the sigmoidal framework generalizes to other post-training regimes (e.g., RLHF variants, multi-task mixtures beyond math/code), and whether alternative bounded fits improve robustness in different domains (Appendix A.4).

Overall, the paper offers both a scientifically grounded scaling framework and a robust RL recipe. It shows that you can predict—and thus plan—RL training at scales that would otherwise be prohibitively risky to explore blindly. The accompanying curve-fitting code (Appendix A.5) lowers the barrier to adopting this methodology.
