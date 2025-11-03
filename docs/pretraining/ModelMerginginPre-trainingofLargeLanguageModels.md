# Model Merging in Pre-training of Large Language Models

**ArXiv:** [2505.12082](https://arxiv.org/abs/2505.12082)

## 🎯 Pitch

This paper introduces Pre-trained Model Averaging (PMA), a novel framework for merging multiple checkpoints during the pre-training of large language models—spanning from millions to over 100 billion parameters. By applying PMA, the authors show that one can achieve significant, consistent gains in model performance while also enabling accurate prediction of annealing outcomes, reducing both computational cost and development time. This breakthrough offers the community practical guidelines for efficient LLM pre-training, unlocking faster model validation, improved stability, and democratizing state-of-the-art model development.

---

## 1. Executive Summary
The paper introduces Pre-trained Model Averaging (`PMA`), a simple but systematic way to merge multiple pre-training checkpoints of the same large language model to obtain a stronger model and to accurately anticipate the benefits of later learning-rate annealing. Across dense and mixture-of-experts (MoE) models from hundreds of millions to over 100B parameters, merging checkpoints from the long constant-learning-rate (“stable”) phase consistently improves downstream performance and can stand in for cosine annealing—saving time and compute while also stabilizing later training stages.

## 2. Context and Motivation
- Problem addressed
  - Pre-training LLMs is expensive and slow. Practitioners typically run a warmup → long stable phase at a constant learning rate → cosine decay (“annealing”). It is hard to know how a model trained at the current step will perform after annealing without actually running the long annealing phase.
  - While model merging (combining weights from different models) is popular post-training, it is largely unexplored during pre-training at scale. There is little public detail on how teams like DeepSeek and LLaMA-3 employ it (Section 1).
- Importance
  - Practical: Faster, cheaper model development cycles. If merging during the stable phase can approximate post-annealing quality, teams can validate designs sooner and shorten/skip annealing (Figures 2–3).
  - Theoretical: Understanding when and why weight averaging across checkpoints improves a single training trajectory (Section 4.6).
- Prior approaches and gaps
  - Post-training merging: Task Arithmetic, TIES-Merging, Fisher merging, DARE, etc., combine separate task-specialized models (Section 2).
  - Pre-training merging: LAWA and related work show benefits but at smaller scales and without guidance for modern LLM schedules or MoE architectures (Section 2).
- Positioning
  - This work focuses on merging along a single pre-training trajectory (“checkpoint merging”) and scales it to very large dense and MoE models. It provides practical recipes (how many checkpoints to merge, spacing between them, which weighting scheme to use) and analyzes mechanisms (Sections 3–4, 4.6, and Appendix A–C).

Definitions (paper-specific or uncommon):
- `Checkpoint`: a saved model state at a specific training step.
- `Model merging` (here): computing a weighted average of parameters from several checkpoints of the same training run to form a single model (Equations (1)–(5)).
- `Warmup–Stable–Decay (WSD)`: a learning-rate schedule with a brief warmup, a long constant-LR phase, then cosine decay (Section 3).
- `Annealing`: the cosine-decay phase of the learning rate.
- `PMA-init`: using a merged model as the initialization for later stages (Continual Training `CT` or Supervised Fine-Tuning `SFT`) to stabilize training (Section 4.4–4.5).
- `GradNorm`: the magnitude of the gradients; spikes often indicate instability (Section 4.5).

## 3. Technical Approach
Step-by-step method (Section 3):
1. Training setup
   - Train dense and MoE LLMs with a WSD schedule on a large internal corpus (trillions of tokens). Models span from small (e.g., 0.7B/7B MoE) to very large (e.g., 20B/200B MoE; “activated/total” parameters) (Section 1; Section 3).
   - Periodically save checkpoints during the stable and decay phases.
2. Define which checkpoints to merge
   - Select `N` checkpoints `{M1, M2, …, MN}` along the same training trajectory.
   - Ensure they are evenly spaced in consumed tokens: if `Ti` is the token count of checkpoint `i`, the interval is `V = Ti+1 − Ti` (Equation (2)).
3. Choose a weighting scheme for averaging (Equations (1)–(5))
   - `SMA` (Simple Moving Average): uniform weights (`wi = 1/N`).
   - `WMA` (Weighted Moving Average): linearly larger weights for later checkpoints (e.g., `wi = i`, normalized).
   - `EMA` (Exponential Moving Average): a recursive weighted average emphasizing recent checkpoints; smoothing factor `α` controls how much more weight recent checkpoints receive.
   - Resulting merged model: `Mavg = Σ wi Mi`.
4. Where and when to merge
   - Merge during the long stable phase to obtain immediate quality gains and to predict how the model would look after annealing (Figures 1–3).
   - Merge in early annealing to obtain a model comparable to, or sometimes better than, the final annealed endpoint (Figure 2).
5. Practical hyperparameters (Section 4.3)
   - Interval `V` (spacing between checkpoints) and number of checkpoints `N` matter:
     - Larger models typically prefer larger `V`; smaller models prefer smaller `V` (upper panel of Figure 5).
     - Merging more checkpoints helps at the end of training but can hurt if training is still unstable (lower panel of Figure 5). The paper often uses `N = 10` as a good compute/performance trade-off.
6. Using merged weights to initialize later stages (`PMA-init`)
   - Replace the usual “latest checkpoint” initialization for `CT` or `SFT` with the merged model (Section 4.4).
   - Observed effects: smoother GradNorm and fewer loss spikes; early metrics sometimes improve without harming final results (Figure 6; Figure 7-left).
7. Recovery from instability
   - If training collapses (loss spike), average the few checkpoints before the spike and resume from the merged state to stabilize and rejoin the original trajectory (Figure 7-right).

Why these choices?
- Emphasis on simplicity and robustness: while `WMA` or `EMA` can slightly outperform early on, differences vanish in later phases; `SMA` is used by default for stability and ease (Section 4.2).
- Even spacing in tokens (`V`) aligns with training dynamics and batch sizes; larger models use larger batches, so their weights evolve more smoothly over larger token spans (Section 4.3).

Mechanism in plain language (Section 4.6):
- Checkpoints are like slightly different “snapshots” of a model exploring a valley in the loss landscape. Averaging those snapshots can cancel idiosyncratic deviations if their errors point in different directions, landing the average closer to the bottom of the valley.
- The paper makes this precise with a second-order Taylor approximation: averaging helps if cross-terms `δi^T H δj` (deviation vectors with the curvature `H`) are mostly negative, which means the deviations are complementary (Equations (6)–(15)). A 2D visualization shows the merged point lying near higher MMLU contours than individual checkpoints (Figure 8).

## 4. Key Insights and Innovations
- Reliable “early annealing simulator” during pre-training
  - Insight: Merging checkpoints from the stable phase can match the benefit of later cosine annealing. In practice, continuing with a constant LR and applying PMA at intervals closely tracks or even anticipates the performance after annealing (Figure 3 across Humaneval, BBH, MMLU, GSM8K).
  - Significance: Enables faster validation and potentially skipping long annealing runs, reducing compute costs and time-to-signal (Section 4.1).
- Scalable, architecture-agnostic gains
  - Insight: PMA improves MoE models from 0.7B/7B to 20B/200B (Figure 1) and dense models up to 70B (Appendix A, Figure 9).
  - Significance: The approach generalizes across sizes and architectures, making it broadly applicable in LLM pre-training pipelines.
- Practical recipes for when/how to merge
  - Insight: The optimal checkpoint interval `V` scales with model size (e.g., ~4B tokens for 0.7B/7B, ~8B for 1.3B/13B, ~80B for 10B/100B; Section 4.3 and Figure 5). Using more checkpoints helps at convergence but can hurt early if weights are still volatile (lower panel of Figure 5).
  - Significance: Converts a general idea (“average checkpoints”) into concrete, easily deployable guidance.
- `PMA-init` for stability and recovery
  - Insight: Initializing CT/SFT with merged weights yields smoother gradient norms and fewer loss spikes (Figure 7-left) and can rescue training after collapse by averaging the few pre-spike checkpoints (Figure 7-right).
  - Significance: This is a new, low-cost tool for stabilizing large-scale training without sacrificing final performance (Sections 4.4–4.5).
- Mechanistic explanation of why merging works
  - Insight: Second-order analysis shows checkpoint deviations can cancel through averaging when they are complementary with respect to the Hessian (Equations (6)–(15)); visualization confirms the merged point often sits in a higher-scoring region (Figure 8).
  - Significance: Moves beyond empirical heuristics to a principled explanation, strengthening confidence in the approach.

## 5. Experimental Analysis
- Evaluation setup (Section 3)
  - Models: Multiple MoE lines (e.g., 0.7B/7B to 20B/200B, with active parameters ≪ total) and dense lines (411M to 70B; Appendix A).
  - Training schedule: WSD with long constant-LR (“stable”) followed by cosine decay (“annealing”).
  - Data: Internal corpus with trillions of tokens (exact composition not disclosed).
  - Metrics: Weighted average over many standard benchmarks (ARC-C, BBH, DROP, WinoGrande, HellaSwag, MMLU, C-Eval, TriviaQA, Ape210K, GSM8K, MATH, MBPP, HumanEval, AGIEval, GPQA, MMLU-Pro; Section 3).
- Main quantitative results
  - Stable phase merging improves performance (MoE)
    - Figure 1 shows consistent gains after merging across Humaneval, BBH, MMLU, GSM8K for Seed-MoE models from 1.3B/13B to 20B/200B.
    - Examples reported in Section 4.1:
      > Seed-MoE-1.3B/13B on HumanEval: 31.1 → 36.6 after PMA.  
      > Seed-MoE-10B/100B on HumanEval: 54.3 → 61.6 after PMA.
  - Early annealing merging suffices
    - Figure 2: During cosine decay, PMA at early annealing achieves performance comparable to end-of-annealing, sometimes surpassing it for larger models.
  - Constant LR + PMA vs actual annealing
    - Figure 3: Forked runs from 1.4T tokens. The constant-LR+PMA path initially outperforms both constant-LR and annealed paths and later remains comparable to annealed across tasks (Humaneval, BBH, MMLU, GSM8K).
  - Which merging method?
    - Figure 4: At 204B tokens, `WMA` > `EMA` (α=0.2 > α=0.1) > `SMA`, reflecting that late checkpoints are more informative early in training. Differences vanish as training stabilizes; the paper defaults to `SMA` for simplicity (Section 4.2).
  - How many checkpoints and how far apart?
    - Interval `V` ablation (upper Figure 5): too-large `V` early (e.g., 16B–32B on 1.3B/13B) underperforms by blending unstable early weights. As training matures, performance across `V` converges.
    - Number `N` ablation (lower Figure 5): more checkpoints hurts early but helps at convergence; `N=15` beats `N=3` by almost 1 point in the final aggregate metric; the paper often uses `N=10` as a good trade-off (Section 4.3).
  - Downstream stages (`PMA-init`)
    - CT: Loss curves start lower and MMLU rises faster with PMA-init; end performance roughly matches baselines across multiple LR schedules (Figure 6).
    - SFT: Mixed but non-degrading; one large model (15B/150B) shows gains on many metrics when using PMA-init with the same LR schedule (Appendix B, Table 1). For example:
      > Table 1: With identical schedule 2e−5→2e−6, MMLU 86.8 → 87.1; LiveBench 50.5 → 52.0; AMC-2023 61.0 → 64.0; OOD 32.6 → 34.7; Reasoning 32.1 → 34.0; Instruction Following 36.3 → 38.8.
    - Stability: PMA-init reduces GradNorm spikes in SFT (Figure 7-left). In a small-model stress test with very high LR (6e−3), PMA-init from pre-spike checkpoints recovers a diverged run (Figure 7-right).
  - Dense models
    - Appendix A (Figure 9): PMA improves dense models too, including a 70B dense model:
      > HumanEval 50.6 → 57.9; GSM8K 85.9 → 91.3 after merging.
- Do the experiments support the claims?
  - Breadth: Results are shown across sizes, architectures (MoE and dense), training phases (stable and annealing), and stages (CT, SFT), with ablations over methods (`SMA/EMA/WMA`) and hyperparameters (`V`, `N`). This breadth supports generality.
  - Caveats: The pre-training corpus and some architecture details are not disclosed, and the overall metric is a weighted average over many benchmarks; reproducibility is limited (Section 3; Appendix C).
- Robustness and failure modes
  - Early merging with too-large `V` or too-large `N` can hurt because it blends unstable weights (Figure 5).
  - SFT gains are not guaranteed across all models and settings; however, PMA-init rarely harms performance (Section 4.4 and Appendix B).

## 6. Limitations and Trade-offs
- Assumptions and scope
  - The approach assumes a single training trajectory and regularly spaced checkpoints by tokens (Equation (2)). It is not a method for merging different models trained on different data/tasks (Section 3).
  - Theoretical argument assumes a local quadratic approximation around an optimum and a positive definite Hessian; real LLM loss landscapes can deviate from these assumptions (Section 4.6).
- Scenarios not addressed
  - Post-training reinforcement learning (RL) merging is not explored, even though modern RL phases can be long and may offer many adjacent checkpoints (Appendix C).
  - Cross-run or cross-domain merging (different initializations or data slices) is outside scope.
- Computational considerations
  - Storing many full checkpoints and loading them to compute an average incurs I/O and memory overhead. The paper does not quantify wall-clock savings versus full annealing, though Figure 3 suggests validation can happen earlier (Section 4.1).
- Hyperparameter sensitivity
  - Early in training, choosing `V` too large or `N` too big can degrade performance (Figure 5). Practical tuning remains necessary, though the paper offers rules of thumb (Section 4.3).
- Reproducibility and generality
  - Training data and some architecture details are not public; precise gains may vary by corpus, tokenizer, and batch sizes (Section 3; Appendix C).
- Open questions
  - How does the optimal `V` scale beyond the reported ranges, or under non-WSD schedules?
  - Can the theoretical negative cross-term condition `δi^T H δj < 0` be measured or encouraged during training?

## 7. Implications and Future Directions
- Impact on the field
  - PMA reframes “checkpoint averaging” from a post-training trick into a core pre-training tool. It enables teams to approximate annealed performance without actually annealing, accelerating iteration and potentially reducing compute budgets (Figures 2–3).
  - The stability benefits of `PMA-init` provide a simple operational safeguard for large-scale training pipelines (Figure 7).
- Practical applications
  - Faster architecture, data, and LR-schedule exploration by validating with constant-LR+PMA “simulated annealing.”
  - A “checkpoint-merging monitor” that periodically computes a PMA model to project end-of-run quality and decide whether to continue, branch, or stop (Section 1 and Conclusion).
  - Training reliability: When loss spikes occur, use PMA over the last few healthy checkpoints to recover without full restarts (Figure 7-right).
- Research directions
  - Theory: Formalize when negative cross-terms are likely, how they evolve across layers/blocks, and whether merging can be optimized layer-wise or with curvature-aware weights (extending Equations (6)–(15)).
  - Methods: Adaptive selection of `V` and `N` based on online stability estimates (e.g., GradNorm variance), or hybrid schemes that switch from `WMA/EMA` early to `SMA` late (Section 4.2–4.3).
  - Beyond pre-training: Systematic study of merging in RL and multi-stage post-training; investigate merging across different runs or domains with alignment constraints (Section 2; Appendix C).
  - Systems: Implement streaming/online PMA to avoid checkpoint I/O bottlenecks and integrate with distributed training.

In short, the paper shows that averaging the right set of checkpoints during pre-training is not just a small trick—it can reliably lift quality, forecast annealing gains, and stabilize later stages, with concrete recipes (Figures 1–5) and a plausible mechanism (Section 4.6) to guide deployment.
