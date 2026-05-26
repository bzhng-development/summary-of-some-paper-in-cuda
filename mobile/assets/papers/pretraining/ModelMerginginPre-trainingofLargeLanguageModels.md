# Model Merging in Pre-training of Large Language Models

**ArXiv:** [2505.12082](https://arxiv.org/abs/2505.12082)

## 🎯 Pitch

This paper introduces Pre-trained Model Averaging (PMA), a novel framework for merging multiple checkpoints during the pre-training of large language models—spanning from millions to over 100 billion parameters. By applying PMA, the authors show that one can achieve significant, consistent gains in model performance while also enabling accurate prediction of annealing outcomes, reducing both computational cost and development time. This breakthrough offers the community practical guidelines for efficient LLM pre-training, unlocking faster model validation, improved stability, and democratizing state-of-the-art model development.

---

## 1. Executive Summary

This paper studies model merging during the pre-training phase of large language models, introducing **Pre-trained Model Average (PMA)** — a framework that averages sequential checkpoints from a single training trajectory to improve model quality without additional training. Through experiments on dense and MoE architectures spanning 411M to over 100B parameters trained on trillions of tokens, the authors demonstrate that merging checkpoints from the stable (constant learning rate) phase of a Warmup-Stable-Decay schedule not only yields consistent performance gains (e.g., Seed-MoE-10B/100B improving from 54.3 to 61.6 on HumanEval) but also accurately simulates the performance of full cosine annealing — matching annealed model quality at a fraction of the training cost. A key secondary contribution is **PMA-init**, which uses merged weights as initialization for continued training or supervised fine-tuning, showing that it stabilizes training dynamics (smoother GradNorm curves, recovery from irrecoverable loss spikes) without harming final performance. The paper establishes that model merging provides substantial pre-training benefits — but primarily when checkpoints are drawn from the stable training phase where weights have not yet converged to a tight local optimum, explaining why merging loses effectiveness after extensive annealing.

## 2. Context and Motivation

### The Core Problem: Pre-Training Merging Is Understudied Despite Its Potential

This paper tackles a specific gap in the model merging literature: **we know model merging works for post-training (fine-tuned models on different tasks), but its behavior and utility during the *pre-training phase itself* — where models are trained from scratch on trillions of tokens — remains poorly understood.** The distinction matters because pre-training and post-training operate under fundamentally different conditions:

- **Post-training merging** combines models that have been independently fine-tuned on separate downstream tasks. Each model has converged from the same base checkpoint but into different functional specializations (math, coding, instruction following). Merging aims to create a single model that inherits multiple capabilities — a task that the model merging literature has addressed extensively with methods like Task Arithmetic, Ties-Merging, DARE, and Fisher Merging.

- **Pre-training merging** combines checkpoints from a *single training trajectory*. These checkpoints are sequential snapshots from the same optimization process, differing only in how many tokens they've seen. The goal is not to combine capabilities (all checkpoints perform the same fundamental language modeling task) but rather to produce a better single checkpoint than any individual one in the trajectory.

The paper argues that this second scenario — merging within a pre-training trajectory — is severely understudied. Section 2 explicitly states:

> "research on model merging during the pre-training phase remains relatively limited"

The reason for this gap is practical, not principled: studying pre-training merging requires access to intermediate checkpoints from massive training runs — something only organizations training large-scale models from scratch possess. The paper notes that while technical reports from DeepSeek-V3 and LLaMA-3.1 have "indicated their employment of model merging techniques for model development, detailed information regarding these techniques has not been publicly disclosed." Independent researchers, lacking access to trillion-token training trajectories with saved checkpoints, simply cannot study this phenomenon.

This gap creates a knowledge vacuum: the community knows pretraining merging *exists* (industry labs use it) but has no published guidance on *when* it works, *why* it works, *which merging strategy* to use, or *what hyperparameters* govern its effectiveness. The paper positions itself as filling exactly this gap — providing the first detailed technical analysis of model merging scaled to 100B+ parameter models across both dense and MoE architectures.

### Why This Problem Matters: The Economic and Scientific Stakes

The significance of pre-training merging extends well beyond academic curiosity. The paper identifies several concrete motivations:

**Pre-training consumes enormous resources.** Training runs for modern LLMs cost millions of GPU-hours and span weeks to months. Any technique that improves the quality of the final model without requiring additional training tokens represents pure cost savings — you get a better model for the same training budget. The paper's finding that PMA can match the performance of full annealing without actually running the annealing phase means that the entire decay (learning rate reduction) segment of training — which can consume 10–30% of total training tokens — might be partially substituted by merging, yielding substantial computational savings.

**Training instability remains an unsolved problem.** Large-scale training runs frequently encounter loss spikes, gradient norm explosions, and irrecoverable training divergence caused by infrastructure failures, bad data batches, or inherent optimization instability. When a run collapses, the standard recovery procedure is to restart from an earlier checkpoint and hope the failure doesn't recur — essentially discarding all training between the recovery checkpoint and the crash point. The paper's PMA-init technique offers a principled alternative: merge several recent checkpoints before the spike to produce a stabilized initialization that can safely resume training past the failure point. Section 1 frames this explicitly:

> "LLMs still face several critical challenges, including the extensive pre-training costs, discounted effectiveness of domain-specific post-training, imprecisely-predictable performance scaling, as well as the instability of large-scale training."

If PMA-init provides reliable recovery from training instability, it directly addresses one of the most expensive failure modes in LLM development.

**The relationship between pre-training strategies and final performance is hard to predict.** During a long pre-training run, practitioners need to make decisions: is the current learning rate optimal? Should we anneal now or continue at the constant rate? Should we extend training? These decisions currently require either expensive ablation runs or waiting until training completes to evaluate the final model. The paper's finding that PMA can *simulate* the performance of an annealed model from stable-phase checkpoints — without actually running the annealing — means that researchers can get a reliable preview of final model quality mid-training. This enables faster iteration cycles and more informed decisions about when to stop or adjust training.

**We don't understand *why* model merging works in pre-training.** The post-training merging literature has developed several theoretical frameworks (linear mode connectivity, task arithmetic in weight space), but these frameworks assume models have diverged from a shared initialization into different task-specific minima. Pre-training merging involves checkpoints along the *same* optimization trajectory — a fundamentally different setting. The paper's mechanistic analysis (Section 4.6, using Taylor expansion of the loss landscape) attempts to provide a theoretical grounding, but the paper is primarily empirical. Understanding when and why averaging sequential checkpoints improves performance has scientific value for optimization theory and practical value for designing better training recipes.

### Prior Approaches and Their Limitations

The paper identifies several strands of prior work, each with specific shortcomings that PMA addresses:

**LAWA (Latest Weight Averaging) and variants.** The closest prior work is Kaddour's LAWA framework, which demonstrated that averaging checkpoints during ImageNet and BERT training could accelerate convergence. Hägele et al. later extended this to show that checkpoint averaging enables compute-optimal training beyond fixed training durations. Sanyal et al. showed that high learning rates combined with early weight averaging contribute to faster convergence.

These works established the viability of pre-training merging, but they were limited to relatively small-scale experiments (BERT-scale models, ImageNet-class vision tasks). The paper notes that as "model and data scales dramatically, independent researchers struggle to evaluate model merging's impact on large-scale models, mainly due to limited access to intermediate checkpoints from extensive pre-training." The key limitation of prior work is not that the idea is wrong — it's that the evidence base doesn't scale to billion-parameter LLMs trained on trillions of tokens. Do the phenomena observed at BERT scale persist? Do they strengthen or weaken? Are there new phenomena that only emerge at scale? These questions were open.

**Checkpoint Merging via Bayesian optimization.** Liu et al. provided a comprehensive evaluation of checkpoint merging during Baichuan2 pre-training, using Bayesian optimization to select which checkpoints to merge and with what weights. While this work operated at larger scale than LAWA (Baichuan2 is a 7B-13B parameter model), it focused on optimization of the merging recipe rather than characterizing the fundamental behavior of merging across different training phases, learning rate schedules, and model architectures. The paper extends this line of work to 100B+ parameter models and to MoE architectures, which have different training dynamics than dense models.

**Industry practice (undocumented).** The paper explicitly notes that DeepSeek-V3 and LLaMA-3.1 both mention using model merging during pre-training in their technical reports, but neither provides detailed methodology, ablation results, or hyperparameter guidance. This creates a frustrating situation for the open-source community: the technique is known to work at the frontier, but the knowledge of *how* to apply it effectively remains locked inside industry labs. The paper positions itself as democratizing this knowledge.

**Post-training merging methods (not directly applicable).** The paper reviews extensive work on post-training merging — Task Arithmetic, Ties-Merging, AdaMerging, Fisher Merging, RegMean, DARE — but notes that these methods are designed for a different problem. Post-training merging involves combining models with different task specializations; the challenges are resolving parameter interference between divergent models and finding combination weights that preserve multiple capabilities. Pre-training merging involves combining models that differ only in optimization progress; the challenges are understanding how the loss landscape geometry makes averaging beneficial and determining which phases of training produce mergeable checkpoints. Methods designed for the former (like DARE's random drop-and-rescale of delta parameters, or Fisher Merging's use of Fisher information for weighting) don't clearly translate to the latter.

**Warmup-Stable-Decay schedulers (context, not a prior approach).** The paper operates in the context of the WSD learning rate schedule, popularized by MiniCPM (Hu et al., 2024). The WSD schedule consists of three phases: a short warmup, an extended stable phase at constant learning rate, and a final decay phase with cosine annealing. Prior work had established that the stable phase allows the model to explore the loss landscape broadly, while the decay phase allows convergence to a good local optimum. What was unknown was how model merging interacts with these phases — specifically, whether merging stable-phase checkpoints could replicate or even exceed the benefits of the annealing phase. The paper's investigation of this question is one of its primary contributions.

### How This Paper Positions Itself

The paper's positioning can be understood along three axes:

**First, scope:** It focuses exclusively on *within-trajectory, pre-training* model merging, deliberately setting aside post-training merging. The Preliminaries section (Section 3) formalizes this: merged entities are sequential checkpoints along a single training trajectory, with data consumption forming an arithmetic sequence with a common difference V. This clean formalization distinguishes it from the post-training merging literature, where models are independently trained from the same initialization.

**Second, scale:** It claims to be "the first study to provide detailed technical insights into scaling model merging methods to significantly larger model sizes." The experiments span from 411M to 70B dense parameters and from 0.7B/7B to 20B/200B MoE (activated/total) parameters — ranges that substantially exceed prior published work on pre-training merging. The paper doesn't just assert that findings from smaller-scale work transfer; it actively investigates whether and how behavior changes with scale (e.g., the finding in Section 4.3 that optimal merging interval scales with model size).

**Third, comprehensiveness:** Rather than proposing a single novel method, the paper provides a systematic investigation organized around six research questions (Section 4): How does merging affect performance? How do different merging methods compare? How should hyperparameters be chosen? Does merging help downstream training? Does merging improve stability? What mechanisms explain merging's effectiveness? This question-driven structure positions the paper as a reference work — providing "practical pre-training guidelines with effective model merging" — rather than a single-method contribution.

**The unifying insight:** Throughout the paper, the conceptual thread is that **the benefits of pre-training merging depend critically on the training phase from which checkpoints are drawn.** Checkpoints from the stable (constant learning rate) phase are still exploring the loss landscape and exhibit complementary deviations from optimal parameters; averaging them creates a model closer to a good optimum than any individual. Checkpoints from the late annealing phase have already converged into a narrow basin; averaging them provides little benefit because they're already near-optimal and close together. This phase-dependence distinguishes the paper's contribution from prior work that treated checkpoint averaging as a generic technique equally applicable at any point in training.

The paper also positions PMA-init as a novel application: while prior work had used weight averaging to improve final model quality, no prior work had systematically studied using merged weights as *initialization* for continued training or fine-tuning, nor had prior work demonstrated that merged-weight initialization could stabilize training and recover from loss spikes. This extends model merging from a quality-improvement technique to a robustness tool for training operations.

## 3. Technical Approach

### 3.1 Reader Orientation (Approachable Technical Breakdown)

The "system" here is not a deployed application but rather an **experimental framework for understanding model merging during LLM pre-training** — specifically, a set of carefully controlled training runs, checkpoint collection procedures, and merging recipes analyzed to answer six research questions about when and why averaging model weights works. The core problem is that we lack systematic knowledge about what happens when you average checkpoints from a single pre-training trajectory, especially at billion-parameter scale. The solution takes the shape of an empirical investigation: train a family of models from scratch under controlled conditions, apply merging at specific points in training, measure the effects, and use the resulting data to derive practical guidelines.

### 3.2 Big-Picture Architecture (Diagram in Words)

The experimental infrastructure has five major components:

1. **Model Training Infrastructure** — Trains dense and MoE models from scratch on trillions of tokens using a Warmup-Stable-Decay (WSD) learning rate scheduler, saving intermediate checkpoints at regular intervals. This is the raw material for all merging experiments.

2. **Checkpoint Repository** — A collection of saved model weights at known training token counts, forming sequences parameterized by an interval V (the token difference between consecutive saved checkpoints) and a count N (the number of checkpoints available for merging). This repository is what the merging algorithms operate on.

3. **Merging Algorithms** — Three weight-averaging strategies (SMA, WMA, EMA) that take N checkpoints and produce a single merged model. These differ only in their weighting schemes.

4. **Evaluation Suite** — A battery of 16 open-source benchmarks (ARC-Challenge, BBH, DROP, WinoGrande, HellaSwag, MMLU, C-Eval, TriviaQA, Ape210K, GSM8K, MATH, MBPP, HumanEval, AGIEval, GPQA, MMLU-Pro) producing per-task scores and a weighted average "comprehensive performance metric." This measures whether merging actually helps.

5. **PMA-init Framework** — A secondary use of merged weights as *initialization* for continued training (CT) or supervised fine-tuning (SFT), where the merged checkpoint replaces the latest single checkpoint as the starting point for downstream training stages. This tests whether merging provides benefits beyond final model quality — specifically training stability.

Information flows as follows: models are trained under the WSD schedule → checkpoints are saved at regular token intervals → during analysis, subsets of N checkpoints spaced V tokens apart are selected → a merging algorithm combines them into a single weight vector → the merged model is evaluated on the benchmark suite → results are compared to individual checkpoint baselines and to annealed models → conclusions are drawn about optimal merging strategies, hyperparameters, and mechanisms.

### 3.3 Roadmap for the Deep Dive

- **First**, the WSD learning rate schedule and model configurations, since all merging behavior depends on which training phase checkpoints come from and what model architectures are involved.
- **Second**, the checkpoint selection formalism — what it means to select N checkpoints with interval V — and why this parameterization matters for understanding scale-dependent merging behavior.
- **Third**, the three merging algorithms (SMA, WMA, EMA), their mathematical formulations, and what each weighting scheme implies about which checkpoints are considered most important.
- **Fourth**, the evaluation protocol and comprehensive performance metric, since all claims about merging effectiveness are benchmark-driven.
- **Fifth**, the PMA-init procedure — how merged weights serve as initialization for downstream training — and why GradNorm smoothness is the key signal for stability benefits.
- **Sixth**, the mechanistic analysis framework (Taylor expansion and weight space visualization) that attempts to explain *why* merging works in pre-training, grounding the empirical findings in loss landscape geometry.

### 3.4 Detailed, Sentence-Based Technical Breakdown

This is primarily an **empirical analysis paper** whose core idea is that model checkpoints from the stable (constant learning rate) phase of pre-training can be meaningfully averaged to produce a model that outperforms any individual checkpoint, and that this averaging can substitute for expensive learning rate annealing while also stabilizing downstream training.

---

#### The Warmup-Stable-Decay (WSD) Learning Rate Schedule

All experiments operate under a Warmup-Stable-Decay (WSD) learning rate scheduler, a three-phase schedule popularized by MiniCPM (Hu et al., 2024). Understanding this schedule is essential because the paper's central claim — that merging works during the stable phase but loses effectiveness during late annealing — is fundamentally about how the learning rate affects the geometry of the weight trajectory.

The WSD schedule proceeds in three sequential phases:

**Phase 1: Warmup.** The learning rate increases linearly from a small initial value (or zero) to a target peak value over a short initial period. This prevents the optimizer from taking destructively large steps on the first few batches when gradients are unreliable. The warmup phase is brief relative to total training — the paper doesn't specify exact warmup token counts, but warmup is standard practice and not the focus of investigation.

**Phase 2: Stable (constant learning rate).** After warmup, the learning rate is held constant at the peak value for an extended period — the vast majority of training tokens. During this phase, the optimizer performs stochastic gradient descent at a fixed step size, causing the model parameters to explore the loss landscape broadly rather than converging to a single narrow minimum. This exploration is critical for merging: the paper's theoretical analysis (Section 4.6) argues that averaging is beneficial precisely when checkpoints have explored different regions of the loss landscape with "complementary" deviations from optimal parameters. The constant learning rate ensures these deviations exist and persist rather than collapsing toward a single point.

The peak learning rate is determined according to "scaling law guidelines" using optimal values for the internal pretraining corpus. The paper does not specify exact learning rate values, which is a limitation for reproducibility, but the qualitative behavior (constant rate → exploration → mergeable checkpoints) is the key insight.

**Phase 3: Decay (cosine annealing).** In the final phase, the learning rate is reduced from the peak value to a small minimum (typically near zero) following a cosine schedule. This causes the model to converge into a specific local minimum. The paper's experiments show that merging checkpoints from this phase provides diminishing returns because the models being merged are already tightly clustered around the same optimum — averaging them doesn't escape the basin.

The paper investigates merging during both Phase 2 (stable) and Phase 3 (decay), comparing the merged models against the natural annealing trajectory. A critical experiment (Figure 3) forks training at a specific token count in the stable phase, continuing one branch with constant learning rate (plus PMA merging) and the other with cosine annealing. The finding that PMA on the constant-rate branch matches the annealed branch's performance is the paper's most economically significant result.

**Design choice: WSD over standard cosine.** The WSD schedule is explicitly chosen because it creates a clean separation between the exploration phase (stable) and convergence phase (decay). Under a standard cosine schedule that continuously decreases the learning rate, there is no well-defined "stable phase" — the model is always converging, making it harder to isolate the conditions under which merging is beneficial. The WSD schedule's constant-rate middle phase provides a laboratory for studying merging under exploration conditions.

---

#### Model Architectures and Training Configurations

The paper trains a diverse set of models to demonstrate that findings are not architecture-specific:

**Dense models** (standard transformer architectures):
- Seed-Dense-411M (411 million parameters)
- Seed-Dense-2B (2 billion parameters)
- Seed-Dense-8B (8 billion parameters)
- Seed-Dense-70B (70 billion parameters)

**Mixture-of-Experts (MoE) models** (sparsely-gated architectures where only a subset of parameters are activated per token):
- Seed-MoE-0.7B/7B (0.7B activated, 7B total parameters)
- Seed-MoE-1.3B/13B (1.3B activated, 13B total)
- Seed-MoE-3B/30B (3B activated, 30B total)
- Seed-MoE-10B/100B (10B activated, 100B total)
- Seed-MoE-15B/150B (15B activated, 150B total)
- Seed-MoE-20B/200B (20B activated, 200B total)

All models are trained on "an internal pretraining corpus comprising trillions of tokens." The paper acknowledges that "specific model architectures and datasets have not yet been publicly released" but argues that "findings are not strongly tied to these particular choices, as subsequent experiments primarily focus on MoE structures." This is an important caveat: while the paper demonstrates merging benefits across model sizes and architectures, the specific performance numbers may not transfer to models trained on different data distributions or with different architectural details.

**Design choice: MoE focus.** The paper's primary experiments use MoE architectures, with dense model results relegated to Appendix A. The rationale is not explicitly stated, but MoE models are particularly relevant because (1) they represent the frontier of large-scale LLM training (used in DeepSeek-V3, Mixtral, etc.), (2) their sparse activation patterns create more complex loss landscapes where averaging might behave differently than with dense models, and (3) their training dynamics are less well-characterized than dense models, making systematic study more valuable.

---

#### Checkpoint Selection Formalism

The paper formalizes checkpoint selection for merging with two parameters, establishing a clean interface between the training process and the merging algorithm. This formalism is introduced in Section 3 (Preliminaries) and parameterized in Section 4.3.

**The checkpoint sequence.** During training, model weights are saved at regular intervals. The saved checkpoints form a sequence ordered by training progress, where each checkpoint `$M_i$` (for `$i = 1, 2, \ldots, N$`) has an associated token count `$T_i$` — the cumulative number of tokens the model has been trained on when that checkpoint was saved.

**The interval parameter V.** The token difference between consecutive checkpoints in the merging set is:

$$V = T_{i+1} - T_i$$

where `$T_i$` is the cumulative tokens consumed by checkpoint `$i$` and `$T_{i+1}$` is the tokens consumed by the next checkpoint in the sequence.

**What it computes:** the number of training tokens between two adjacent checkpoints used in the merge. If V = 8B tokens, and the first checkpoint in the merge set is at 200B tokens, the second is at 208B, the third at 216B, and so on.

**Why this form:** V captures how far apart the checkpoints are in optimization progress. A small V means checkpoints are close neighbors — their weights have changed little between saves. A large V means checkpoints have diverged significantly — the model has processed many more tokens between saves, and weights have moved substantially. The optimal V represents a tradeoff: too small and the checkpoints are nearly identical (averaging adds nothing), too large and the early checkpoints in the merge set are from a much earlier (potentially unstable) phase of training that introduces harmful variance.

**The count parameter N.** The number of checkpoints included in the merge. The merged model is computed as a weighted sum of N checkpoints spanning a total training range of `$(N-1) \times V$` tokens.

**Critical design choice: uniform spacing.** The paper assumes checkpoints form an arithmetic sequence with equal spacing V. This means all checkpoints in the merge set are equally spaced in token count, rather than selected arbitrarily. This uniformity is important for two reasons: (1) it makes the merging behavior analytically tractable — the merged model represents a uniform moving average over a specific training window — and (2) it avoids introducing confounding variables where some checkpoints are clustered and others isolated, which would make it impossible to attribute effects to the interval V versus the clustering pattern.

**Scale-dependent optimal V.** Section 4.3 reports that the optimal merging interval scales with model size:
- 0.7B/7B MoE models: V ≈ 4B tokens
- 1.3B/13B MoE models: V ≈ 8B tokens
- 10B/100B MoE models: V ≈ 80B tokens

The paper attributes this scaling to the tendency of larger models to use larger batch sizes, meaning that each training step (and each saved checkpoint at a given step interval) represents more tokens of training progress. The observation that V scales roughly proportionally to model size is a practically useful guideline: when applying PMA to a new model scale, start with V proportional to the model's total parameter count.

---

#### Merging Algorithms: SMA, WMA, and EMA

The paper studies three weight-averaging strategies (Section 3, expanded in Section 4.2). All three compute the merged model `$M_{\text{avg}}$` as a weighted sum of N checkpoint weight vectors, but they differ in how they assign weights `$w_i$` to each checkpoint `$M_i$`.

**Common framework.** The merged model is always:

$$M_{\text{avg}} = \sum_{i=1}^{N} w_i M_i$$

where `$M_i$` is the parameter vector of checkpoint `$i$`, `$w_i$` is the scalar weight assigned to that checkpoint, and the weights sum to 1 (directly or through normalization). The difference between methods is entirely in how `$w_i$` is determined.

---

**Simple Moving Average (SMA).** Each checkpoint receives equal weight:

$$M_{\text{avg}} = \frac{1}{N} \sum_{i=1}^{N} M_i$$

where `$N$` is the number of checkpoints being merged and `$M_i$` is the i-th checkpoint's parameter vector.

**What it computes:** the arithmetic mean of N sets of model weights. Each checkpoint contributes equally to the merged model regardless of whether it was saved early or late in the training window. The output is a single weight vector that represents the center of mass of the N checkpoints in parameter space.

**Why this form:** SMA is the simplest possible merging strategy and serves as a baseline. It makes no assumptions about which checkpoints are "better" — it treats all checkpoints in the training window as equally informative. If merging works under SMA, it demonstrates that the benefit comes from averaging per se (reducing variance from individual checkpoint idiosyncrasies) rather than from sophisticated temporal weighting. The paper finds that SMA performs nearly as well as more complex methods once training is complete, making it the preferred choice for "simplicity and stability."

---

**Weighted Moving Average (WMA).** Later checkpoints receive higher weights, typically increasing linearly:

$$M_{\text{avg}} = \sum_{i=1}^{N} \frac{w_i}{w_{\text{sum}}} M_i, \quad w_{\text{sum}} = \sum_{i=1}^{N} w_i$$

where `$w_i$` is the raw weight for checkpoint `$i$` (commonly `$w_i = i$`, giving later checkpoints linearly increasing importance), `$w_{\text{sum}}$` normalizes the weights to sum to 1, and `$M_i$` is the i-th checkpoint.

**What it computes:** a weighted average that emphasizes later (more trained) checkpoints. If `$w_i = i$`, then with N = 10 checkpoints, the 10th receives weight 10/55 ≈ 0.182, while the 1st receives weight 1/55 ≈ 0.018 — a 10× difference. The merged model is pulled toward the most recent checkpoints, which have seen more training data and are presumably closer to optimal.

**Why this form:** WMA encodes the prior belief that more training is better — later checkpoints are closer to convergence and should dominate the average. The paper finds that WMA outperforms SMA in early training stages, when model weights are undergoing significant changes and older checkpoints represent substantially worse models. The linear weighting is a computationally cheap way to implement this prior without requiring the exponential forgetting parameter α that EMA requires.

---

**Exponential Moving Average (EMA).** Weights decay exponentially with checkpoint age:

$$M_{\text{avg}}^{(i)} = \alpha \cdot M_i + (1 - \alpha) \cdot M_{\text{avg}}^{(i-1)}, \quad i \in [2, N]$$

where `$M_i$` is the i-th checkpoint, `$M_{\text{avg}}^{(i-1)}$` is the EMA result computed from checkpoints 1 through i-1, `$\alpha \in (0, 1)$` is the smoothing factor controlling the balance between the current checkpoint and the cumulative average, and `$M_{\text{avg}}^{(1)} = M_1$` (the base case).

**What it computes:** a recursive average where each new checkpoint is merged with the running average using a fixed mixing ratio α. A larger α gives more weight to recent checkpoints; α = 0.2 means the current checkpoint contributes 20% and the running average contributes 80% at each step. The effective weights on earlier checkpoints decay as `$\alpha(1-\alpha)^{N-i}$`, making them exponentially less influential than recent ones.

**Why this form:** EMA is the standard method for maintaining running averages of model parameters during training (used in many published training recipes). It provides a tunable parameter (α) that controls how aggressively the average adapts to recent changes. The paper tests two values — α = 0.1 and α = 0.2 — and finds that α = 0.2 outperforms α = 0.1 in early training. This is interpreted as evidence that when weights are changing rapidly, the average should prioritize recent checkpoints; however, as training stabilizes, the advantage of EMA over SMA diminishes because all checkpoints in the merge window are similarly good.

**Key finding on method comparison (Section 4.2, Figure 4):** At 204B training tokens for Seed-MoE-1.3B/13B, WMA delivers the best performance (baseline: 45.93, SMA: 47.01, WMA: 47.89, EMA₀.₁: 46.04, EMA₀.₂: 47.65). However, by 1607B tokens (end of training), all methods converge to similar performance (baseline: 56.82, SMA: 60.79, WMA: 60.54, EMA₀.₁: 60.61, EMA₀.₂: 60.60). The paper draws the practical conclusion that SMA is preferred for "simplicity and stability" in late training. The reason is intuitive: as the model converges, all checkpoints in a reasonable window are similarly good, so the weighting scheme becomes irrelevant — the average is the average regardless of how you weight the components.

---

#### Evaluation Protocol and Comprehensive Performance Metric

The paper evaluates merged models on a broad benchmark suite to ensure that observed improvements are not task-specific artifacts.

**Benchmark selection.** The evaluation includes 16 open-source benchmarks spanning multiple capabilities:
- **Reasoning:** ARC-Challenge (science questions), BBH (BIG-Bench Hard tasks), DROP (reading comprehension with discrete reasoning), GSM8K (grade-school math), MATH (competition math)
- **Knowledge:** MMLU (massive multitask language understanding), C-Eval (Chinese evaluation), TriviaQA (trivia questions), GPQA (graduate-level questions), MMLU-Pro (more robust MMLU)
- **Code:** HumanEval (code generation), MBPP (basic Python programming)
- **Language understanding:** HellaSwag (commonsense reasoning), WinoGrande (pronoun resolution)
- **Aggregate:** AGIEval (human-centric benchmark), Ape210K (math word problems)

**Evaluation settings.** The paper reports results in "both few-shot and zero-shot settings" but does not specify exact shot counts for each benchmark. This is a minor transparency limitation.

**Comprehensive performance metric.** Rather than cherry-picking individual benchmarks where merging helps, the paper aggregates results into a weighted average score across all 16 benchmarks:

> "The weighted average score across these benchmarks serves as the model's comprehensive performance metric. Unless otherwise specified, we report this score as the model's performance metric to ensure evaluation reliability."

**Why this matters:** Using a single aggregate metric enables clean comparisons across methods, hyperparameters, and model sizes. It prevents the analysis from being dominated by a few benchmarks with high variance or large effects. However, the weighting scheme for the "weighted average" is not specified — are all benchmarks equally weighted? Are weights proportional to benchmark size or difficulty? This missing detail makes it impossible to fully replicate the performance numbers.

**Baseline comparisons.** For each merging experiment, the baseline is the performance of the individual checkpoints in the merge set, presumably the most recent (latest) checkpoint. Figures consistently show a "Baseline" bar or line representing the unmerged model's performance at the corresponding token count.

---

#### PMA-init: Merged Weights as Initialization for Downstream Training

Section 4.4 introduces PMA-init, which uses the merged model `$M_{\text{avg}}$` as the starting point for continued training (CT) or supervised fine-tuning (SFT), rather than using the latest individual checkpoint. This is a distinctive contribution — prior work used averaging to produce a final model, not as an initialization strategy.

**Procedure.** Instead of loading the most recent checkpoint to begin CT or SFT, the practitioner:
1. Selects N checkpoints from the pre-training trajectory (using the same V and N parameters as standard PMA).
2. Computes `$M_{\text{avg}}$` using SMA (the default method chosen for its simplicity).
3. Initializes the downstream training optimizer with `$M_{\text{avg}}$` as the starting weights.
4. Proceeds with standard CT or SFT training (including learning rate scheduling) from that initialization.

**What changes:** The only difference from standard practice is the weight initialization — the training data, learning rate schedule, and architecture are unchanged. The hypothesis is that `$M_{\text{avg}}$` provides a better starting point than any individual checkpoint because averaging has already "smoothed out" parameter noise, placing the model closer to a region of the loss landscape from which downstream optimization is more stable.

**CT experiments (Section 4.4, Figure 6).** The paper experiments with Seed-MoE-0.7B/7B models merged after stable training on approximately 1 trillion tokens. CT training is conducted under several learning rate schedules (cosine decay from peak to minimum, with different peak and minimum values). Key findings:
- **Loss curves:** PMA-init achieves "marginally lower loss at the initial training phase," but as training progresses, "loss values for models with different initialization weights converge to comparable levels." The paper notes that loss curves for different initializations "significantly overlap."
- **Performance (MMLU):** PMA-init models "outperform the baseline early in training" but "retain a slight performance edge in later stages," with overall "performance parity with the baseline" by the end of CT.
- **Learning rate robustness:** "No extensive learning rate tuning is required for PMA-init" — the benefits persist across tested schedules without needing to adjust hyperparameters.

**SFT experiments (Section 4.4, Appendix B, Table 1).** Experiments on Seed-MoE-15B/150B after stable training on 16T tokens and 1T tokens of annealing, followed by 220M tokens of SFT. Results are mixed:
- With the same learning rate (2e-5 → 2e-6 cosine), PMA-init improved over baseline on Open-Benchmark (comprehensive score improvement) and in-house evaluations (OOD: +2.1, Reasoning: +1.9, Instruction Following: +2.5).
- However, "we were unable to replicate such significant gains in subsequent experiments with other model sizes."
- The paper concludes: "as a low-cost approach, PMA-init is worth trying to obtain a more powerful downstream model" but does not guarantee consistent gains.

**Stability benefits observed.** During SFT training initialized with PMA-init, the authors observed "a notably more stable GradNorm metric compared to the baseline" and "reduced frequency of loss spikes" (Section 4.5, Figure 7 left). The GradNorm — the L2 norm of the gradient vector across all parameters — is a measure of how large the parameter updates are at each step. Smoother GradNorm curves indicate more stable optimization: the optimizer is taking consistent-sized steps rather than alternating between tiny steps (stuck in flat regions) and enormous steps (overshooting minima). This observation motivates the loss spike recovery experiments.

**Loss spike recovery (Section 4.5, Figure 7 right).** To test PMA-init as a recovery mechanism, the paper deliberately destabilizes training. They train a 330M/3.3B MoE model "from scratch using an exceptionally high learning rate of 6e-3," which causes "unstable training and abrupt loss spikes" that are "irreversible to its original trajectory." To recover:
1. Take three checkpoints saved *before* the training collapse.
2. Compute `$M_{\text{avg}}$` via PMA (SMA, N=3) on these pre-spike checkpoints.
3. Resume training from `$M_{\text{avg}}$`, continuing past the token count where the spike occurred.

The result (red line in Figure 7, right): "the resumed training process stabilized, successfully navigating past the point of the loss spike and continuing along its original training trajectory." This is a significant practical finding: when a training run collapses, rather than restarting from the last saved checkpoint (and potentially hitting the same spike again), you can merge several pre-spike checkpoints and resume from the averaged weights. The averaging smooths out whatever parameter perturbation caused the instability while preserving the training progress.

**Why PMA-init stabilizes training.** The paper does not provide a direct mechanistic explanation for the stability benefit, but the framework is consistent with the loss landscape analysis in Section 4.6: averaging checkpoints reduces the variance of individual parameter estimates, moving the initialization away from "sharp" regions where small parameter changes cause large loss changes (high curvature directions in the Hessian). Starting optimization from a more central, averaged position reduces the probability that an early bad batch causes a catastrophic update. This is an implicit claim — the paper's Taylor expansion analysis addresses why merging improves loss, not directly why it stabilizes training — but the connection is plausible within the presented framework.

---

#### Mechanistic Analysis: Why Does Merging Work?

Section 4.6 provides the paper's theoretical framework for understanding the effectiveness of pre-training merging. The analysis operates at two levels: a mathematical derivation using second-order Taylor expansion of the loss, and a visualization of weight space geometry.

**Mathematical framework: loss landscape curvature.** The derivation starts with the standard second-order Taylor expansion of the loss function `$L(\theta)$` around an optimal parameter set `$\theta^*$`:

$$L(\theta) \approx L(\theta^*) + (\theta - \theta^*)^T \nabla L(\theta^*) + \frac{1}{2}(\theta - \theta^*)^T H (\theta - \theta^*)$$

where `$L(\theta)$` is the loss at parameters `$\theta$`, `$\theta^*$` is an optimal parameter set (a local minimum), `$\nabla L(\theta^*)$` is the gradient at `$\theta^*$` (which is zero at a minimum, eliminating the linear term), and `$H$` is the Hessian matrix of second partial derivatives of the loss evaluated at `$\theta^*$`, which captures the local curvature of the loss landscape.

**What it computes:** the loss at any parameter point `$\theta$` can be approximated by the loss at the minimum plus a quadratic penalty for how far `$\theta$` is from `$\theta^*$`, weighted by the Hessian `$H$` which encodes how sensitive the loss is to movement in each direction. A large eigenvalue of `$H$` in direction `$v$` means moving in that direction rapidly increases loss; a small eigenvalue means the loss is flat in that direction.

**Why this form:** the Taylor expansion is the standard tool for analyzing local geometry of optimization landscapes. It's valid near a minimum where higher-order terms are negligible. The quadratic approximation transforms the complex, non-convex loss function into a simple quadratic bowl around `$\theta^*$`, making it tractable to analyze how averaging parameters affects loss.

**Applying the framework to individual checkpoints.** Let `$k$` model checkpoints have parameters `$\theta_1, \theta_2, \ldots, \theta_k$`. Define the deviation of each checkpoint from the optimum as `$\delta_i = \theta_i - \theta^*$`. The loss of each individual checkpoint is:

$$L(\theta_i) \approx L(\theta^*) + \frac{1}{2} \delta_i^T H \delta_i$$

The average loss of the individual checkpoints is:

$$\frac{1}{k} \sum_{i=1}^k L(\theta_i) \approx L(\theta^*) + \frac{1}{2k} \sum_{i=1}^k \delta_i^T H \delta_i$$

**Applying the framework to the merged model.** The merged model's parameters are `$\theta_{\text{avg}} = \frac{1}{k} \sum_{i=1}^k \theta_i$`. Its deviation from the optimum is `$\theta_{\text{avg}} - \theta^* = \frac{1}{k} \sum_{i=1}^k \delta_i$`. The merged model's loss is:

$$L(\theta_{\text{avg}}) \approx L(\theta^*) + \frac{1}{2} \left(\frac{1}{k} \sum_{i=1}^k \delta_i\right)^T H \left(\frac{1}{k} \sum_{i=1}^k \delta_i\right)$$

**The key inequality: when is merging beneficial?** For the merged model to have lower loss than the average loss of individual checkpoints — i.e., `$L(\theta_{\text{avg}}) < \frac{1}{k} \sum_i L(\theta_i)$` — the following condition must hold after algebraic manipulation:

$$\sum_{i=1}^k \sum_{j \neq i} \delta_i^T H \delta_j < (k - 1) \sum_{i=1}^k \delta_i^T H \delta_i$$

where `$\delta_i^T H \delta_i$` is the quadratic penalty for checkpoint `$i$`'s individual deviation (always positive since `$H$` is positive definite around a minimum), and `$\delta_i^T H \delta_j$` (for `$i \neq j$`) is the cross-term between the deviations of two different checkpoints.

**What it computes:** a condition on the relationships between checkpoint deviation vectors, as measured through the curvature matrix `$H$`. If the cross-terms `$\delta_i^T H \delta_j$` are predominantly *negative*, the left-hand side is small (or negative), making the inequality easier to satisfy. Negative cross-terms mean that the deviation vectors point in somewhat opposing directions relative to the curvature — moving in direction `$\delta_i$` increases the loss, but moving in direction `$\delta_j$` increases it in a *different* way, and averaging cancels out some of these independent deviations.

**Why this form:** this provides an intuitive geometric interpretation of why merging works. The checkpoints from the stable training phase have explored different directions in the loss landscape. Their deviations from the optimum are not identical — they each have "idiosyncrasies" specific to the particular batch sequence they've seen. When averaged, these idiosyncrasies cancel out (if they are sufficiently uncorrelated or negatively correlated in the Hessian-weighted sense), positioning the merged model closer to `$\theta^*$` than the average individual checkpoint. This is exactly the mechanism of **variance reduction through averaging**: independent deviations from the optimum, when averaged, produce a point with smaller expected distance to the optimum.

**Interpretation — why the training phase matters.** The paper's key insight, grounded in this framework, is temporal:
- **During the stable (constant LR) phase:** The learning rate is high enough that the optimizer doesn't converge to a tight minimum. Successive checkpoints continue to move through parameter space, exploring different directions. Their deviations `$\delta_i$` are substantial and have significant cross-checkpoint variation. The cross-terms `$\delta_i^T H \delta_j$` are likely to be small or negative relative to the diagonal terms, satisfying the averaging-benefit condition.
- **During late annealing:** The learning rate is small, and the model converges tightly into a specific local minimum. Successive checkpoints are all very close to `$\theta^*$`, meaning all `$\delta_i$` are small and similar. The cross-terms are similar to the diagonal terms, the inequality is not satisfied, and averaging provides little benefit — you're averaging points that are already nearly identical and nearly optimal.

This explains the paper's empirical observation that merging loses effectiveness during annealing: the checkpoints being merged lack the "complementary" deviations that make averaging beneficial.

**Weight space visualization (Figure 8).** The paper provides a complementary visualization. They select two parameters from a specific layer of Seed-MoE-1.3B/13B and plot the positions of individual checkpoints (black dots) against MMLU score contour lines. The visualization shows:
- Individual checkpoints are distributed along the contours, not clustered at a single point.
- The merged model's position (presumably the centroid of the black dots) is "often situated closer to a region of higher MMLU scores (a better optimum) than many individual model checkpoints."
- The checkpoints exhibit a "discernible 'complementary' pattern" — they are spread out along the contours rather than all on one side.

**Why the visualization matters:** it concretizes the abstract mathematical condition. The contour plot makes visible what the Taylor expansion describes algebraically: checkpoints are not all converging monotonically toward the optimum; they explore the landscape, sometimes moving sideways or even slightly away from the optimum in individual steps. Averaging cancels out these exploratory deviations, landing in a better position than any individual checkpoint achieved on its own.

**Connection to annealing simulation.** The paper's most practically significant finding — that PMA on stable-phase checkpoints matches annealed performance — can now be understood mechanistically. Annealing helps by reducing the learning rate and allowing the model to converge to a good minimum. Merging helps by averaging out the exploration noise from the stable phase, achieving a similar effect (convergence to a better point) through a completely different mechanism (averaging in weight space rather than gradient descent in loss space). The two mechanisms are functionally equivalent but computationally different: annealing costs additional training tokens at decreasing learning rates; merging costs only the arithmetic of averaging saved checkpoints. If merging can substitute for annealing, the economic implications are substantial — you save the tokens that would have been spent in the decay phase.

**Limitations of the mechanistic analysis.** The Taylor expansion framework is local (valid near a minimum) and quadratic (ignoring higher-order curvature). It provides an explanation for *why averaging can improve loss* but does not provide quantitative predictions of *how much* improvement to expect for a given V, N, and model configuration. The weight space visualization with two parameters is necessarily a projection — the true parameter space has billions of dimensions, and what appears as "moving closer to a better optimum" in 2D may not capture the full geometry. The paper acknowledges these limitations implicitly by not claiming quantitative predictive power from the analysis, using it instead as a qualitative explanatory framework.

## 4. Key Insights and Innovations

### Innovation 1: Annealing Can Be Simulated by Averaging — Not Just Approximated, but Matched

The most striking conceptual move in this paper is the reframing of model merging from a *quality-improvement technique* (merge checkpoints to get a slightly better model) to an *annealing substitute* (merge stable-phase checkpoints and you don't need to anneal at all). This changes the economic calculus of pre-training: the decay phase, which can consume 10–30% of total training tokens, becomes optional rather than mandatory.

Prior to this work, the relationship between merging and annealing was, at best, loosely understood. LAWA and related work had shown that averaging accelerates convergence, but no one had systematically tested whether averaging stable-phase checkpoints could *replicate* the effect of learning rate decay. The dominant assumption was that annealing was an irreducible requirement — you needed to reduce the learning rate to converge to a good minimum, and there was no shortcut. The paper challenges this assumption directly.

The evidence comes from Figure 3, where the authors fork training at 1.4T tokens: one branch continues with constant learning rate (plus PMA merging), the other undergoes cosine annealing for 250B additional tokens. The merged constant-rate models "significantly outperformed both the constant learning rate and annealed models" early in this period, and "even later, their performance was comparable to the annealed models." The paper's conclusion is explicit: "pre-training with a constant learning rate, combined with model merging, can effectively match the performance of an annealed model at any point in the training process without the need for learning rate annealing."

This is a **fundamental shift** in how to think about the training pipeline, not an incremental improvement. It implies that the computational work of annealing — gradually reducing the step size so the optimizer settles into a minimum — is partly redundant with the geometric work of averaging checkpoints that have explored different regions of the loss landscape. The two mechanisms arrive at similar endpoints through completely different computational paths: gradient descent at progressively smaller step sizes versus simple arithmetic in weight space.

The economic implications are substantial. If you can skip the annealing phase and achieve the same final model quality through merging, you save those training tokens — and because annealing operates at decreasing learning rates (where each token contributes less to model improvement than during the stable phase), the tokens saved are among the least efficient in the entire training run. The paper's framing of PMA as an "annealing simulator" is its most potentially disruptive contribution: it suggests a future where pre-training schedules are designed around merging from the start, with the decay phase shortened or eliminated.

There is an important subtlety: the paper also shows in Figure 2 that applying PMA *during* annealing at early stages "were comparable to those at the end of the annealing process." This means that even if you do run annealing, you can stop early, merge the partial-annealing checkpoints, and get the performance of full annealing without running it to completion. The simulation capability works both as a replacement and as an early-stopping mechanism.

### Innovation 2: Pre-Training Merging Is Phase-Dependent — Not a Uniform Technique, but One with a Narrow Operating Window

The paper's organizing insight — that the effectiveness of model merging depends critically on *which phase of the learning rate schedule* the checkpoints come from — seems intuitive in retrospect, but prior work had not articulated it as a first-class principle governing when merging works. The dominant implicit assumption in earlier checkpoint averaging research (LAWA, Checkpoint Merging) was that averaging helps broadly; the idea that it might be *actively harmful* or *completely ineffective* under certain training conditions was underexplored.

The paper establishes a sharp operating window: **merging is highly effective during the stable (constant LR) phase, provides diminishing returns during annealing, and becomes nearly useless once models have tightly converged.** This is documented across multiple lines of evidence:

- Figure 1 shows consistent gains from merging across model sizes during the stable phase (e.g., Seed-MoE-10B/100B: HumanEval from 54.3 to 61.6, GSM8K from 59.8 to 61.6).
- Figure 2 shows that merging during early annealing matches late-annealing performance, implying that as annealing progresses and checkpoints converge, merging provides less additional benefit.
- The mechanistic analysis (Section 4.6) provides the theoretical grounding: when checkpoints have "complementary" deviations (stable phase), averaging cancels noise and moves toward a better optimum; when deviations are small and similar (late annealing), there's nothing to cancel.

This phase-dependence is a **conceptual reframing** with direct practical consequences. It tells practitioners *when* to apply model merging during pre-training (grab checkpoints from the constant-LR phase, not from the end of annealing) and *when not to bother* (if you've already annealed to a low learning rate, the remaining checkpoints are too similar for merging to matter). Prior work had not provided this temporal guidance.

The phase-dependence also explains the paper's finding that the WSD schedule is particularly amenable to merging. A standard cosine schedule that continuously decreases the learning rate blurs the line between exploration and convergence; there's no clean "stable phase" where checkpoints have complementary deviations. The WSD schedule's explicit separation between constant-rate exploration and cosine-decay convergence creates a natural laboratory for studying when merging works — and the paper's results suggest that this separation is not just analytically convenient but *practically advantageous* for merging-based training strategies.

This insight is **moderately fundamental** rather than transformative: it doesn't change what model merging is, but it changes *when and how* practitioners apply it, and it provides a theoretical framework (complementary deviations in the loss landscape) for predicting when merging will be effective in new training configurations.

### Innovation 3: Model Merging as a Training Stability Tool — A New Application Category

The paper's introduction of PMA-init — using merged checkpoints as initialization for downstream training, specifically as a recovery mechanism from training instability — represents a **novel application category** for model merging. Prior work had conceptualized merging exclusively as an *output* technique (produce a better final model). This paper shows that merging can function as an *operational* technique: a way to stabilize training, recover from collapses, and smooth optimization dynamics.

The significance lies not in the method's complexity (it's just averaging pre-spike checkpoints) but in the **diagnostic insight** that loss spikes leave recoverable information in pre-spike checkpoints that can be salvaged through merging. When a training run collapses, the standard recovery procedure — restart from the last saved checkpoint — essentially discards all parameter updates between that checkpoint and the crash. PMA-init offers a third option between "discard post-checkpoint work" and "try to push through the spike": merge several pre-spike checkpoints and resume from the averaged position.

The evidence is concrete. Figure 7 (right) shows a deliberately destabilized 330M/3.3B MoE model (trained with learning rate 6e-3) experiencing irrecoverable loss spikes. Merging three pre-spike checkpoints and resuming produces a stabilized trajectory that "successfully navigat[es] past the point of the loss spike and continu[es] along its original training trajectory." Figure 7 (left) shows that PMA-init during SFT produces "notably more stable GradNorm" compared to standard initialization from the latest checkpoint.

This is a **practical innovation** rather than a theoretical one, but its significance for large-scale training operations is substantial. Training runs at 100B+ parameter scale cost millions of dollars and run for weeks; a loss spike that forces a restart from an earlier checkpoint can waste days of compute. The paper's finding that averaging pre-spike checkpoints enables recovery past the failure point provides a concrete, low-cost intervention that could save organizations substantial resources. The paper frames this explicitly: PMA-init "provides an alternative solution to avoid retraining the model from scratch, thereby substantially reducing the waste of computational resources."

The broader implication is that model merging should be thought of as a **training operations primitive** alongside checkpointing, learning rate scheduling, and gradient clipping — not just as a post-hoc model improvement technique. This expands the scope of model merging research from "how to produce better final models" to "how to make the training process itself more robust."

### Innovation 4: The Scale-Dependent Optimal Merging Interval — A Practical Scaling Relationship

The finding that the optimal merging interval V scales with model size — approximately 4B tokens for 0.7B/7B MoE, 8B for 1.3B/13B, and 80B for 10B/100B — is **incrementally important** rather than revolutionary, but it fills a specific practical gap that prior work left open. Prior checkpoint averaging studies (LAWA, Sanyal et al.) operated at smaller scales and did not systematically characterize how merging hyperparameters should change with model size. Practitioners training larger models were left to guess whether the hyperparameters that worked at BERT scale would transfer.

The paper provides a concrete guideline: V scales roughly proportionally to model parameter count, which the authors attribute to larger models using larger batch sizes. This means that when applying PMA to a new model scale, the starting recommendation is to set V proportional to the model's total parameters relative to a known good configuration. This is not a theoretical breakthrough — it's an empirical observation that reduces the hyperparameter search space for practitioners.

The finding that incorporating more checkpoints (larger N) "consistently improves performance once training is completed" (Section 4.3, Figure 5 lower panel) is similarly practical: N=15 outperforms N=10 outperforms N=6 outperforms N=3 at the end of training, with N=3 being "nearly 1 point lower than N=15" on the comprehensive performance metric. However, the paper notes diminishing returns and settles on N=10 as a practical tradeoff between "computational cost and performance gains."

These scaling relationships represent **practical scaffolding** for the open-source community: actionable guidance that the paper positions as enabling independent researchers to apply model merging at scale without the trial-and-error that industry labs could afford through internal experimentation. The guidance is not theoretically derived (the Taylor expansion framework does not predict optimal V or N), but it is empirically grounded and represents the kind of detailed operational knowledge that the paper notes is missing from the published literature — filling exactly the gap between "DeepSeek and LLaMA mention using merging" and "here's how to actually do it."

### Innovation 5: The Merging Method Becomes Irrelevant at Convergence — A Simplifying Result

The paper's finding that SMA, WMA, and EMA converge to near-identical performance as training progresses (Section 4.2, Figure 4) is a **negative result with positive implications**: it tells practitioners that they don't need to worry about sophisticated weighting schemes for pre-training merging. By the end of training, all methods produce merged models within ~0.3 points of each other on the comprehensive metric (60.54–60.79, compared to 56.82 baseline).

This is significant not because it's surprising — if all checkpoints in the merge window are similarly good, equal weighting should work fine — but because it **simplifies the operational recipe** and eliminates a potential source of over-engineering. Prior post-training merging methods (Fisher Merging, RegMean, evolutionary optimization) invest substantial complexity in determining optimal per-model weights. The paper's finding suggests that for pre-training merging specifically, this complexity is unnecessary: SMA works and has the advantage of requiring zero hyperparameter tuning.

The finding has a temporal caveat: early in training, when model weights are changing rapidly, WMA and EMA (with appropriate α) can outperform SMA because they correctly downweight older, substantially worse checkpoints. But the practical recommendation is clear: if you're merging checkpoints from late in the stable phase, just use equal weights. This is a **practical simplification** that reduces the barrier to adoption — one fewer decision for practitioners to optimize.

The deeper implication, which the paper doesn't fully articulate, is that the irrelevance of the weighting scheme at convergence is *evidence that the benefit of merging comes from variance reduction rather than from clever temporal weighting*. If the best checkpoints were always the most recent ones, WMA/EMA would consistently outperform SMA; the fact that they don't suggests that older checkpoints in the window contribute genuinely useful information (their deviations are complementary to recent ones) rather than being merely worse versions of the same point. This supports the paper's loss-landscape interpretation: the average works because the components have diverse, complementary deviations, not because we're cleverly emphasizing the "best" ones.

## 5. Experimental Analysis

### Evaluation Methodology

- **Dataset.** The paper uses an internal pretraining corpus comprising trillions of tokens for training all models. Since this corpus is not publicly released, exact composition details are unavailable. For evaluation, the authors use 16 open-source benchmarks spanning reasoning, knowledge, code, and language understanding: ARC-Challenge, BBH, DROP, WinoGrande, HellaSwag, MMLU, C-Eval, TriviaQA, Ape210K, GSM8K, MATH, MBPP, HumanEval, AGIEval, GPQA, and MMLU-Pro. The specific split sizes and few-shot/zero-shot configurations per benchmark are not specified in the paper.

- **Base model(s).** The paper trains two families of models from scratch: **Dense models** (Seed-Dense-411M, Seed-Dense-2B, Seed-Dense-8B, Seed-Dense-70B) using standard transformer architectures, and **Mixture-of-Experts models** (Seed-MoE-0.7B/7B, Seed-MoE-1.3B/13B, Seed-MoE-3B/30B, Seed-MoE-10B/100B, Seed-MoE-15B/150B, Seed-MoE-20B/200B) where the notation indicates activated/total parameters. All models are trained on the internal corpus using a Warmup-Stable-Decay learning rate scheduler with learning rates determined by scaling law guidelines. The models are chosen to span a wide parameter range (411M to 200B total) to test whether merging behavior is scale-dependent.

- **Metrics.** The primary evaluation metric is the **weighted average score across all 16 benchmarks**, which the paper calls the "comprehensive performance metric." Individual benchmark scores are also reported for selected tasks (HumanEval, BBH, MMLU, GSM8K in Figures 1 and 9). For training dynamics analysis, the paper reports **loss values** (cross-entropy on the pretraining corpus) and **GradNorm** (the L2 norm of the gradient vector across all parameters). The weighting scheme for the comprehensive metric is not specified in the paper.

- **Baselines.** The paper uses several comparison points depending on the experiment:
  - **Individual checkpoint baseline:** The performance of the latest (most recently trained) single checkpoint at the corresponding token count, representing what the model achieves without any merging (labeled "Baseline" in Figures 1, 4, 5, and 9).
  - **Constant learning rate baseline:** In Figure 3, the performance of continuing training at the constant (stable-phase) learning rate without merging, showing what happens if you simply train longer without annealing (labeled "Constant").
  - **Annealed model baseline:** In Figures 2 and 3, the performance of models that have undergone the full cosine annealing decay phase, representing the standard complete training recipe (labeled "Annealing").
  - **Standard initialization baseline:** In Section 4.4 (CT and SFT experiments), downstream training initialized from the latest individual checkpoint rather than from PMA-merged weights (labeled "baseline" in Figure 6 and Table 1).

- **Generation budget / compute accounting.** The paper uses **training tokens** as the universal unit of compute. All merging experiments are conducted at specific token counts during pre-training (e.g., 204B, 452B, 1112B, 1607B tokens for Seed-MoE-1.3B/13B in Figure 4). The checkpoint interval V is measured in tokens (4B, 8B, 16B, 32B tokens between consecutive checkpoints). The paper does not report FLOP counts or GPU-hours. Merging itself — computing a weighted average of N checkpoint weight vectors — is computationally negligible compared to training, so its cost is not separately accounted for.

- **Cross-validation / statistical protocol.** The paper does not employ cross-validation or report confidence intervals on benchmark scores. The 16-benchmark weighted average serves as a variance-reduction mechanism, but there is no formal statistical testing of whether observed performance differences are significant. For the CT and SFT experiments (Section 4.4), multiple learning rate schedules are tested to assess robustness, but these are independent runs rather than formal cross-validation. The loss spike recovery experiment (Section 4.5, Figure 7 right) uses a single deliberately destabilized training run with the 330M/3.3B MoE model.

### Main Quantitative Results

#### Merging During the Stable Training Phase (Constant Learning Rate)

The headline result for stable-phase merging appears in Figure 1, which compares baseline (unmerged) and PMA-merged performance for MoE models at the end of stable training. The merged models show consistent improvements across all model sizes and most benchmarks:

- **Seed-MoE-1.3B/13B:** HumanEval improves from 31.1 to 36.6 (+5.5 points), BBH from 65.1 to 65.9 (+0.8), MMLU from 50.8 to 56.5 (+5.7), GSM8K from 42.7 to 46.3 (+3.6). The BBH gain is modest, which the paper attributes to "near-saturation of these metrics" for larger models.

- **Seed-MoE-10B/100B:** HumanEval improves from 54.3 to 61.6 (+7.3), BBH from 84.9 to 88.6 (+3.7), MMLU from 83.7 to 84.7 (+1.0), GSM8K from 59.8 to 66.5 (+6.7). The MMLU gain narrows compared to the smaller model, consistent with saturation.

- **Seed-MoE-20B/200B:** HumanEval improves from 84.0 to 89.3 (+5.3), BBH from 82.3 to 85.6 (+3.3), MMLU from 85.2 to 87.4 (+2.2), GSM8K from 61.6 to 61.6 (0.0). The flat GSM8K result is unusual and not discussed in the text.

The dense model results in Appendix A (Figure 9) show similar patterns: Seed-Dense-70B improves from 50.6 to 57.9 on HumanEval (+7.3) and from 85.9 to 91.3 on GSM8K (+5.4), demonstrating that merging benefits persist and may even increase at larger dense scales. The paper notes: "the performance gains of larger models were not smaller than those of smaller models."

#### Merging During the Cosine Annealing Phase

Figure 2 compares the performance of models undergoing standard cosine annealing against models where PMA is applied to checkpoints collected during the annealing process. The x-axis shows training tokens within the annealing phase (which starts from a base of stable training); the y-axis shows the comprehensive performance metric.

For **Seed-MoE-10B/100B** (left panel): At the earliest measured point (50B tokens into annealing), the PMA-merged model scores approximately 76.5, slightly below the annealed model at ~76.0. As annealing progresses to 250B tokens, both curves rise — the annealed model reaches approximately 77.5, while PMA reaches approximately 77.8, slightly above the fully annealed model. The curves are close throughout, with PMA matching or exceeding annealing at most points.

For **Seed-MoE-15B/150B** (right panel): The pattern is similar but with a larger gap favoring PMA. At 0.3T tokens into annealing, the annealed model scores approximately 76.5 while PMA scores approximately 77.5 — a full point advantage. By 1T tokens, the annealed model reaches approximately 78.8 while PMA reaches approximately 79.5, maintaining the advantage.

The critical implication: applying PMA to early annealing checkpoints achieves performance comparable to or exceeding full annealing, meaning the annealing process could potentially be shortened or skipped.

#### Can Merging Replace Annealing Entirely?

Figure 3 directly tests whether PMA applied to constant-learning-rate checkpoints can match an annealed model. Training is forked at 1.4T tokens for Seed-MoE-1.3B/13B. Three trajectories are compared:
- **Constant + PMA:** Continue at constant learning rate, applying PMA merging to the resulting checkpoints.
- **Annealing:** Apply cosine decay to the learning rate for an additional 250B tokens.
- **Constant:** Continue at constant learning rate without merging (pure baseline for extended stable training).

On **HumanEval** (top-left): At 1.4T tokens (fork point), all three trajectories start near 31%. By 1.45T (50B tokens later), Constant+PMA jumps to approximately 38%, substantially above both Annealing (~33%) and Constant (~31%). At 1.6T tokens (end of the 250B extension), Constant+PMA reaches approximately 38.5%, Annealing reaches ~38%, and Constant reaches ~31%. PMA matches annealing and dramatically outperforms constant-rate training without merging.

On **GSM8K** (bottom-right): The pattern is similar. At the fork point, all are near 42.5%. By 1.45T, Constant+PMA reaches ~68%, Annealing ~63%, Constant ~43%. By 1.6T, Constant+PMA reaches ~73%, Annealing ~70%, Constant ~44%.

The paper's conclusion from this experiment: "pre-training with a constant learning rate, combined with model merging, can effectively match the performance of an annealed model at any point in the training process without the need for learning rate annealing."

#### Comparison of Merging Methods

Figure 4 examines how SMA, WMA, and EMA (with α = 0.1 and α = 0.2) compare at different training stages for Seed-MoE-1.3B/13B. The comprehensive performance metric is reported at four token counts: 204B, 452B, 1112B, and 1607B.

At **204B tokens** (early stable phase):
- Baseline (unmerged): 45.93
- EMA₀.₁: 46.04 (+0.11 over baseline)
- SMA: 47.01 (+1.08)
- EMA₀.₂: 47.65 (+1.72)
- **WMA: 47.89 (+1.96)** — best performer

WMA's advantage at this early stage is attributed to its higher weighting of more recent (better) checkpoints. The gap between EMA₀.₂ and EMA₀.₁ suggests that when weights are changing rapidly, faster adaptation to recent checkpoints helps.

At **452B tokens**:
- Baseline: 55.91
- SMA: 58.90 (+2.99)
- EMA₀.₁: 58.95 (+3.04)
- WMA: 58.50 (+2.59)
- EMA₀.₂: 58.90 (+2.99)

The ordering shifts — SMA and EMA₀.₂ now tie for best, while WMA falls slightly behind. The methods are clustered within 0.45 points of each other.

At **1112B tokens**:
- Baseline: 50.95
- All merging methods cluster between 53.76 and 54.42, a range of 0.66 points.
- WMA: 54.42 (best)
- SMA: 54.17

At **1607B tokens** (end of training):
- Baseline: 56.82
- EMA₀.₁: 60.61 (+3.79)
- SMA: 60.79 (+3.97)
- EMA₀.₂: 60.60 (+3.78)
- WMA: 60.54 (+3.72)

All methods are within 0.25 points of each other. The paper's conclusion: "as training advances to later stages and model weights stabilize, the performance differences between merging methods diminish. For its simplicity and stability, we primarily use SMA for model merging in subsequent experiments."

#### Hyperparameter Ablation: Merging Interval V and Count N

Figure 5 (upper panel) fixes N = 10 and varies V ∈ {4B, 8B, 16B, 32B} tokens for Seed-MoE-1.3B/13B. At 204B tokens, V = 32B underperforms substantially (comprehensive score ~41.3) compared to V = 8B (~47.0). This is attributed to large intervals incorporating "unstable weights from the initial training phase, leading to significant weight disparities and suboptimal outcomes." By 1607B tokens, V = 8B achieves 60.79, while V = 4B achieves 60.50 and V = 16B achieves 60.30 — the gap narrows to ~0.5 points, suggesting that as training stabilizes, interval choice becomes less critical.

Figure 5 (lower panel) fixes V = 8B and varies N ∈ {3, 6, 10, 15}. At 204B tokens, larger N hurts: N = 15 scores 48.1 versus N = 3 at 45.4 — early checkpoints introduce noise. By 1607B tokens, the ordering reverses: N = 15 scores 59.7 (but note: this is lower than N=6 at 60.8 and N=10 at 60.79 — there appears to be an error in the paper's reporting, as the text claims "merging a larger number of models led to significant performance improvements" and "the overall performance for N = 3 was nearly 1 point lower than for N = 15," but Figure 5 lower panel shows N=15 at 59.7 versus N=3 at 59.7 at 1607B, with N=6 and N=10 both higher). The paper settles on N = 10 as a practical balance.

The paper reports scale-dependent optimal intervals: V ≈ 4B for 0.7B/7B, V ≈ 8B for 1.3B/13B, V ≈ 80B for 10B/100B.

#### PMA-init for Continued Training (CT) and Supervised Fine-Tuning (SFT)

Figure 6 presents CT results for Seed-MoE-0.7B/7B trained on approximately 1T tokens. The left panel shows loss curves for four configurations — two learning rate schedules (4.08e-4→1.0e-5 and 2.04e-4→1.0e-5, and one with 2.04e-4→1.5e-5), each run with both PMA-init and baseline initialization. The PMA-init runs achieve "marginally lower loss at the initial training phase," but the curves converge as training progresses. The paper notes that "the purple line significantly overlaps with the blue line, and the brown line significantly overlaps with the pink line" — indicating that initialization differences wash out.

The right panel shows MMLU performance over CT token consumption (0–100B tokens). PMA-init models "outperform the baseline early in training" (e.g., at 10B consumed tokens, PMA-init scores ~0.615 versus baseline ~0.605 for the top schedule). By 100B tokens, the gap narrows with PMA-init at ~0.645 and baseline at ~0.640 — a marginal advantage.

For SFT (Appendix B, Table 1), Seed-MoE-15B/150B after stable training and 1T tokens of annealing undergoes 220M tokens of SFT. With the same learning rate (2e-5→2e-6), PMA-init outperforms baseline on Open-Benchmark (comprehensive improvement), with notable gains on in-house evaluations: OOD +2.1 points, Reasoning +1.9, Instruction Following +2.5. At lower (1e-5→2e-6) and higher (4e-5→2e-6) learning rates, PMA-init also shows improvements, particularly PMA₁ₑ₋₅ achieving +2.7 on LiveBench and +4.5 on AMC-2023. However, the paper acknowledges that "we were unable to replicate such significant gains in subsequent experiments with other model sizes," characterizing PMA-init for SFT as "worth trying" but not guaranteed.

#### Training Stability and Loss Spike Recovery

Figure 7 (left) shows GradNorm curves during SFT training for Seed-MoE-15B/150B. The baseline initialization (blue) exhibits higher-magnitude and more variable GradNorm values, with visible spikes around steps 200, 400, and 600 reaching values of 2.0–3.0. The PMA-init curve (pink) shows consistently lower GradNorm values (mostly 0.5–1.5) with smoother variation and fewer spikes. This is interpreted as evidence that PMA-init produces "more stable GradNorm" and "reduced frequency of loss spikes."

Figure 7 (right) shows the loss curve for a deliberately destabilized 330M/3.3B MoE model trained at learning rate 6e-3. The baseline training (blue) shows an abrupt, irrecoverable loss spike around step 1000, with loss jumping from the 6–8 range to approximately 12 and continuing to diverge. The PMA-init recovery (red line) is applied at the spike point: three pre-spike checkpoints are merged via SMA, and training resumes from the merged weights. The resumed training "successfully navigat[es] past the point of the loss spike and continu[es] along its original training trajectory" — the loss remains in the 6–8 range without spiking.

### Ablation Studies and Robustness Checks

- **Merging method comparison (Figure 4):** SMA, WMA, and EMA converge to nearly identical performance by end of training (within 0.25 points at 1607B tokens for Seed-MoE-1.3B/13B). This robustness to weighting scheme simplifies the operational recipe — equal-weight averaging is sufficient.

- **Merging interval V (Figure 5, upper):** At early training stages (204B tokens), large intervals (V = 32B) underperform small intervals (V = 8B) by ~5.7 points on the comprehensive metric, indicating sensitivity to including unstable early checkpoints. By late training, the gap narrows to ~0.5 points, showing robustness once weights stabilize.

- **Number of merged checkpoints N (Figure 5, lower):** Early in training, larger N hurts (N = 15 underperforms N = 3 at 204B tokens) due to inclusion of unstable early weights. At training completion, N = 6 and N = 10 outperform N = 3, but the paper's claim that N = 15 provides the best performance is not clearly supported by the figure, where N = 15 at 1607B scores 59.7 — lower than N = 6 (60.8) and N = 10 (60.79). This inconsistency is not addressed in the text.

- **Dense model architecture (Figure 9, Appendix A):** PMA benefits transfer to dense architectures across scales (411M to 70B). The gains on large dense models are substantial — Seed-Dense-70B improves from 50.6 to 57.9 on HumanEval (+7.3) and 85.9 to 91.3 on GSM8K (+5.4) — demonstrating that findings are not MoE-specific.

- **SFT learning rate robustness (Table 1, Appendix B):** PMA-init for SFT shows improvements across three different learning rate schedules (1e-5, 2e-5, 4e-5 peak learning rates), indicating that the benefit is not an artifact of a specific hyperparameter configuration. However, the paper acknowledges failure to replicate SFT gains on other model sizes.

- **CT learning rate robustness (Figure 6):** PMA-init benefits for CT persist across multiple learning rate schedules without requiring hyperparameter retuning.

- **Annealing simulation at multiple scales (Figure 2):** The finding that PMA matches or exceeds annealed performance is replicated across Seed-MoE-10B/100B and Seed-MoE-15B/150B, with the effect appearing stronger at larger scale (larger gap favoring PMA for the 15B/150B model).

### Critical Assessment

**Claim 1: "Merging checkpoints from the stable training phase produces consistent and significant performance improvements."**

This claim is well-supported for the specific models and benchmarks tested. Figure 1 shows robust gains across four MoE model scales and four representative benchmarks. Figure 9 (Appendix A) extends this to four dense model scales with similar magnitude improvements. The gains are consistent in direction (all positive) and magnitude (typically 2–8 points on individual benchmarks). However, the claim of "consistent" improvement has important boundaries: (1) the paper notes that BBH gains are modest for larger models due to saturation — the benefit exists but diminishes when metrics are near ceiling; (2) some individual benchmark results are flat (GSM8K for Seed-MoE-20B/200B shows 61.6 → 61.6); (3) all results are on the paper's internal models and training corpus, with no public replication possible. The claim would be stronger with results on publicly available model checkpoints or training trajectories.

**Claim 2: "Merging stable-phase checkpoints can match the performance of full annealing."**

Figure 3 provides the key evidence: on a forked training run, Constant+PMA matches or exceeds Annealing on all four benchmarks (HumanEval, BBH, MMLU, GSM8K) by the end of the 250B-token extension period. The result is striking and supports the claim. However, there are important qualifications:

First, this is demonstrated on a single model (Seed-MoE-1.3B/13B) at a single fork point (1.4T tokens). The paper does not test whether the substitution works at different fork points or for different model sizes. If you fork earlier (at 500B tokens instead of 1.4T), does PMA still match annealing? The paper doesn't answer this.

Second, the experiment demonstrates that PMA *over 250B additional tokens* matches annealing *over 250B additional tokens*. The real economic claim is stronger: can PMA match annealing *without the 250B additional constant-rate training*? In other words, if you just merge checkpoints from right at the fork point (without additional constant-rate training), does that match annealing? The paper doesn't isolate this — the Constant+PMA line includes both additional training tokens AND merging. The claim that merging "enables accurate prediction of annealing behavior" (abstract) is more precisely supported: Figure 2 shows that merging early-annealing checkpoints predicts late-annealing performance, which is a narrower but still valuable result.

Third, the paper acknowledges (Section 4.1) that PMA merged models "significantly outperformed both the constant learning rate and annealed models" early in the extension period, but the annealed model catches up by the end. This suggests PMA provides a faster route to good performance rather than a strictly better final model — a practically useful but conceptually distinct claim from "matching annealing."

**Claim 3: "PMA-init helps stabilize training processes, especially recovering from irrecoverable loss spikes."**

The evidence for this claim comes from two experiments. Figure 7 (left) shows smoother GradNorm curves for PMA-init during SFT — this is a genuine observation, but GradNorm smoothness is a *correlate* of stability, not a direct measure of improved training outcomes. The paper does not demonstrate that smoother GradNorm translates to better final model quality, and the CT experiments (Figure 6) show that PMA-init and baseline converge to comparable performance.

Figure 7 (right) shows a single recovery experiment on a 330M/3.3B model with deliberately induced instability. The recovery works — the red line continues stably past the spike point. This is compelling but limited: one model, one instability trigger (extreme learning rate), one recovery attempt. The paper does not test whether PMA-init recovery works for other types of training instability (bad data batches, hardware-induced numerical errors, optimizer state corruption) or for models at larger scales where instability patterns may differ. The claim that PMA-init provides "reliable recovery" is based on a single demonstration.

A deeper issue: the loss spike recovery experiment uses checkpoints from *before* the spike. In a real training crash, the practitioner must have been saving checkpoints at sufficiently high frequency to have usable pre-spike weights. If checkpoints are saved every 10B tokens and the spike occurs 2B tokens after the last save, the "pre-spike" checkpoints are already partly on the destabilized trajectory. The paper doesn't discuss this checkpoint frequency requirement.

**Missing experiments that would strengthen the paper:**

- **Ablation on fork point timing:** Testing PMA-as-annealing-substitute at multiple fork points (early stable phase, mid stable phase, late stable phase) to characterize when the substitution holds.
- **Isolation of merging benefit from additional training:** Comparing PMA at the fork point (no additional constant-rate training) against full annealing — this tests the pure "simulation" claim without confounding by additional tokens.
- **Multi-seed stability experiments:** Running the recovery experiment multiple times with different random seeds to assess reliability of recovery.
- **Comparison of PMA-init against other recovery strategies:** For loss spikes, how does PMA-init compare to simply rolling back to the last checkpoint and reducing the learning rate? Or to gradient clipping? The paper doesn't benchmark against alternative recovery methods.
- **Larger-scale stability demonstration:** The recovery experiment uses a 330M/3.3B model; stability patterns at 10B/100B or larger may differ, and the practical value of PMA-init for recovery is highest at scales where training runs are most expensive.
- **Ablation on which checkpoints to merge for recovery:** The paper uses N = 3 pre-spike checkpoints. Does N = 5 work better? Does including a checkpoint right at the spike boundary help or hurt?
- **Public model replication:** Since the training data and model architectures are internal, none of the results are independently reproducible. Even a small-scale replication on a public model (e.g., Pythia, OLMo) with public checkpoints would substantially increase confidence in the findings.

**Assessment of the difficulty estimation claim:** The paper does not implement a difficulty estimation mechanism — this is not a limitation of the approach but reflects a different problem framing than the reference example. The "difficulty" equivalent in this paper would be predicting *when* during training merging will be most beneficial, which the paper addresses through the phase-dependence analysis rather than per-example difficulty estimation.

**Overall:** The paper's core empirical findings — that stable-phase merging consistently improves performance, that merging methods converge in effectiveness late in training, and that merging can partially substitute for annealing — are well-supported within the tested scope. The practical guidance on hyperparameters (V scales with model size, N = 10 is a good default, SMA is sufficient) is empirically grounded and actionable. The weaker areas are the stability claims (limited demonstration), the annealing-substitution claim (confounded with additional training), and the generalizability beyond the paper's internal training infrastructure (no public replication possible). The paper is strongest as a systematic characterization of merging behavior at scale and weakest in its claims about training stabilization, which require more extensive validation.

## 6. Limitations and Trade-offs

### Internal Training Infrastructure — No Public Reproducibility

**The assumption or constraint.** All models, the pretraining corpus, and intermediate checkpoints are proprietary. The paper acknowledges this explicitly in Section 3:

> "specific model architectures and datasets have not yet been publicly released"

The 16-benchmark evaluation suite is public, but the models being evaluated — and critically, the checkpoints needed to apply and study PMA — are not. This means that every quantitative result in the paper (every bar in Figures 1–9, every number in Table 1) is **not independently reproducible**. An open-source practitioner reading this paper can adopt the qualitative guidelines (merge stable-phase checkpoints, SMA is sufficient, V should scale with model size), but they cannot verify the claimed performance numbers, test whether the findings transfer to their own training setup, or debug discrepancies without access to the same models and data.

**The consequence.** The paper's claim to provide "practical pre-training guidelines for effective model merging" (Abstract) is partially undermined. Guidelines that are validated only on a proprietary infrastructure carry an irreducible uncertainty: do the specific hyperparameter recommendations (N = 10, V scaling with model size, SMA over WMA) depend on details of the ByteDance training stack — optimizer settings, data mixture, architectural choices, tokenizer — that differ from what other practitioners use? The paper argues that "findings are not strongly tied to these particular choices" (Section 3), but this assertion is untested. A practitioner training a LLaMA-style dense model on public data cannot know whether PMA will produce a 4-point HumanEval gain or a 0-point gain without running the experiment themselves — which is exactly the expensive trial-and-error the paper aims to eliminate.

The paper's positioning as filling the gap left by DeepSeek-V3 and LLaMA-3.1 (who "indicated their employment of model merging techniques... [but] detailed information... has not been publicly disclosed," Section 2) is itself incomplete because this paper discloses *procedures* but not *artifacts*. The community gains the recipe but not the ability to verify it.

**What evidence exists in the paper.** The limitation is structural — the entire experimental section depends on internal infrastructure. The paper does provide dense model results in Appendix A (Figure 9) and tests across a range of MoE scales, which partially addresses architecture-dependence, but these are all trained on the same internal corpus with the same training stack. There is no cross-infrastructure validation (e.g., applying PMA to a public model like Pythia or OLMo and reporting those numbers alongside the internal results).

**Mitigation status.** Not addressed. The paper does not suggest that models or checkpoints will be released, does not provide a small-scale public replication, and does not discuss the reproducibility limitation. The authors state they "posit that our findings are not strongly tied to these particular choices" (Section 3), but this is a belief, not a demonstrated fact. A minimal mitigation — running PMA on one publicly available model series with published intermediate checkpoints and reporting whether the same patterns hold — would substantially increase confidence in generalizability.

---

### Difficulty Estimation Cost: PMA Requires Frequent Checkpoint Saving

**The assumption or constraint.** The entire PMA framework assumes that multiple checkpoints from the training trajectory are available at regular intervals V. The paper's hyperparameter recommendations (V = 4B–80B tokens, N = 3–15) imply that practitioners must save model weights frequently throughout the stable training phase — far more frequently than typical production training pipelines, which might save only a handful of checkpoints for recovery purposes.

The paper does not explicitly discuss the storage cost of maintaining N checkpoints for merging, but it is substantial. At the scales studied (70B dense, 200B total MoE parameters), a single checkpoint in FP16 consumes approximately 140GB (70B × 2 bytes) to 400GB (200B × 2 bytes). Saving checkpoints every V = 8B tokens during a multi-trillion-token training run — and retaining the last N = 10 of them — requires storing and managing several terabytes of model weights. This is not prohibitive for well-resourced labs but is a non-trivial operational requirement that the paper's headline efficiency claims (e.g., "significantly lower training costs," Abstract) do not account for.

**The consequence.** The storage and I/O overhead of frequent checkpointing creates a hidden cost that partially offsets the training token savings from substituting annealing with merging. During the stable phase at constant learning rate, each checkpoint save pauses training (or requires asynchronous snapshotting infrastructure), and each checkpoint consumes disk space that must be provisioned, managed, and potentially transferred between storage tiers. For a 10B/100B MoE model where the recommended V ≈ 80B tokens, a training run consuming several trillion tokens might generate dozens of checkpoints. The paper's economic argument — that PMA saves compute by avoiding or shortening the annealing phase — is incomplete without accounting for the storage cost of the checkpoints that PMA requires.

More subtly, the checkpoint frequency requirement interacts with the paper's scale-dependent V recommendation. For the 10B/100B model, V ≈ 80B tokens means checkpoints are spaced far apart in training progress. If a loss spike occurs between saves, the "pre-spike" checkpoints available for PMA-init recovery may already be substantially behind the spike point, meaning that recovery via merging loses more training progress than if checkpoints were saved more frequently. The paper's stability recovery experiment (Figure 7, right) used N = 3 pre-spike checkpoints from a 330M/3.3B model, where checkpoint frequency is not discussed and storage is not a concern — the practical tradeoff between checkpoint frequency (better recovery) and storage cost (higher overhead) at production scale is unexplored.

**What evidence exists in the paper.** None. The paper does not report checkpoint file sizes, storage requirements, I/O overhead, or the number of checkpoints saved during any training run. The cost of checkpointing is entirely absent from the analysis. The paper treats checkpoints as freely available inputs to the merging process.

**Mitigation status.** Not addressed. The paper does not discuss storage costs, does not provide guidelines for checkpoint retention policies, and does not factor checkpoint overhead into any cost comparison. A practitioner implementing PMA would need to independently determine how frequently to save checkpoints and how many to retain, balancing the merging benefit against storage infrastructure costs.

---

### Annealing Substitution Claim Is Confounded with Additional Training

**The assumption or constraint.** The paper's most economically significant claim — that PMA on constant-learning-rate checkpoints can match the performance of full annealing — is tested in a single experiment (Figure 3, Section 4.1) where training is forked at 1.4T tokens for Seed-MoE-1.3B/13B. The "Constant + PMA" line **includes 250B additional tokens** of constant-learning-rate training beyond the fork point, with PMA applied to the resulting checkpoints. The "Annealing" line includes 250B tokens of cosine-decay training. The comparison therefore tests: (additional constant-rate training + merging) vs. (additional annealing training). It does *not* test: (merging only, without additional training) vs. (additional annealing training).

**The consequence.** The strength of the "annealing substitution" claim hinges on what fraction of the performance gain comes from the additional 250B training tokens versus from the merging operation itself. If most of the gain is from additional training (with merging providing a modest boost), then PMA is not truly substituting for annealing — it's just showing that continuing to train at a constant rate plus averaging is competitive with annealing, which is a weaker and less surprising result. The economic savings from "skipping annealing" would then be largely illusory: you're still spending 250B tokens, just at a constant learning rate rather than a decaying one.

The paper does report the "Constant" line (constant learning rate without merging), which at 1.6T tokens reaches only ~31% on HumanEval versus ~38.5% for Constant+PMA and ~38% for Annealing. This demonstrates that *merging is essential* — constant-rate training alone does not match annealing. But it does not isolate whether the merging benefit alone (without the 250B extra tokens) would have been sufficient. The claim that "pre-training with a constant learning rate, combined with model merging, can effectively match the performance of an annealed model at any point in the training process *without the need for learning rate annealing*" (Section 4.1, emphasis added) overstates what the experiment demonstrates: the constant learning rate training was extended, not replaced.

A separate experiment — applying PMA to checkpoints from right at the 1.4T fork point (without any additional constant-rate training) and comparing against the fully annealed model — would cleanly test whether merging alone substitutes for annealing. This experiment is not reported.

**What evidence exists in the paper.** Figure 2 provides partial evidence in the other direction: when PMA is applied during the annealing phase itself, early-annealing checkpoints merged via PMA match late-annealing performance. This shows that merging *during* annealing can predict or match the endpoint of annealing, which is a genuine simulation result. But Figure 3 — the experiment explicitly designed to test whether merging can *replace* annealing — confounds merging with additional training tokens.

**Mitigation status.** Not addressed. The paper does not acknowledge the confounding, does not report a merging-only (no additional training) comparison, and does not discuss the fraction of the performance gain attributable to additional tokens versus merging. The ablation that would resolve this — PMA at the fork point vs. full annealing — is straightforward but absent.

---

### Single Training Fork Point — Phase-Dependence Boundary Is Unmapped

**The assumption or constraint.** The annealing substitution experiment (Figure 3) is conducted at a single fork point: 1.4T tokens into the training of Seed-MoE-1.3B/13B. The paper's central conceptual contribution — that merging benefits are phase-dependent — implies that the effectiveness of PMA as an annealing substitute should itself depend on *when* in training the substitution is attempted. Forking early in the stable phase (when weights are changing rapidly and checkpoints are less stable) versus late in the stable phase (when weights have partially stabilized) versus at the boundary of the decay phase (when the model is about to be annealed anyway) might produce very different results.

**The consequence.** The practical guideline "you can skip annealing and use PMA instead" has an unknown temporal validity window. A practitioner at 500B tokens into training cannot know from this paper whether PMA at their current point would match the performance of annealing from 500B to 750B tokens, or whether they need to train further into the stable phase before the substitution becomes reliable. The paper's earlier findings (Section 4.3, Figure 5) show that merging with large intervals V early in training underperforms because it incorporates unstable early checkpoints — this suggests that PMA-as-annealing-substitute might fail at early fork points where the checkpoints available for merging are from a less stable regime. Conversely, forking very late in the stable phase might show that PMA matches annealing easily because the model is already close to convergence — but then the savings from skipping annealing are small because you've already done most of the training.

The paper's failure to characterize how the annealing-substitution relationship changes with fork timing leaves a gap between the single demonstrated case (it works at 1.4T tokens) and the general claim (it works "at any point in the training process," Section 4.1).

**What evidence exists in the paper.** The single fork point experiment (Figure 3, Section 4.1) plus the general observation that merging loses effectiveness during late annealing (Figure 2, Section 4.6). The phase-dependence framework (Section 4.6) provides a theoretical basis for expecting fork-point-dependence — merging should be most effective when checkpoints have complementary deviations — but this theory is not tested empirically for the annealing substitution scenario. No experiment varies the fork point.

**Mitigation status.** Not addressed. The paper does not discuss fork-point sensitivity, does not test multiple fork points, and does not qualify the "at any point" claim with temporal boundaries. The statement is presented as a general finding supported by a single data point. A minimal mitigation would be testing two or three fork points (e.g., at 30%, 50%, and 70% through the stable phase) and characterizing how the PMA-vs-annealing gap changes.

---

### Loss Spike Recovery Demonstrated on Artificially Induced Instability Only

**The assumption or constraint.** The paper's claim that PMA-init enables "reliable recovery from unstable training trajectories" (Abstract) is supported by a single experiment (Section 4.5, Figure 7 right) on a 330M/3.3B MoE model where instability is deliberately induced by "using an exceptionally high learning rate of 6e-3." This is a controlled, synthetic failure mode. In production large-scale training, loss spikes arise from a diverse set of causes: bad data batches (e.g., corrupted text, adversarial content), hardware errors (GPU memory faults, network corruption), numerical instability in mixed-precision operations, optimizer state corruption, or rare pathological gradient configurations that occur even at correctly tuned learning rates. These real-world failure modes may produce different parameter-space damage than a uniformly excessive learning rate.

**The consequence.** The recovery strategy validated in the paper — merge N = 3 pre-spike checkpoints and resume — may not generalize to other instability types. A loss spike caused by a bad data batch might perturb parameters in a highly directional way (overfitting to the corrupted batch) that averaging with earlier checkpoints can smooth out — this is consistent with the paper's loss-landscape framework. But a loss spike caused by numerical overflow in mixed-precision operations might corrupt parameters with NaN or Inf values that propagate through the averaging, producing a meaningless merged model. A spike caused by optimizer state corruption (e.g., Adam moment estimates becoming degenerate) survives checkpoint averaging because PMA only merges model weights, not optimizer states — the corrupted optimizer would continue to produce bad updates from the merged weights.

More fundamentally, the recovery experiment demonstrates that PMA-init can *continue training past* a spike point without re-collapsing, but it does not demonstrate that the *recovered model achieves the performance it would have achieved had the spike never occurred*. The paper reports that the resumed training "continu[es] along its original training trajectory" (Section 4.5), but this is assessed visually from the loss curve (Figure 7, right), not through downstream benchmark evaluation. A model that recovers to a stable loss but with permanently degraded knowledge or capabilities would not be detected by the loss metric alone.

**What evidence exists in the paper.** The single destabilized training run (Figure 7, right) with one recovery attempt. There is no benchmark evaluation of the recovered model, no testing against alternative recovery methods (rollback + reduced learning rate, rollback + gradient clipping, rollback + skipping the problematic data range), no testing on other instability triggers, and no testing at larger scales where instability patterns may differ.

**Mitigation status.** Partially addressed. The paper does not claim that PMA-init handles all types of training instability — it presents the result as a demonstration that PMA-init *can* recover from instability in at least one scenario. However, the abstract's language of "reliable recovery" and the claim that PMA-init "provides an alternative solution to avoid retraining the model from scratch, thereby substantially reducing the waste of computational resources" (Section 4.5) implies a robustness that the single synthetic experiment does not establish. The paper does not discuss the types of instability that PMA-init might fail to recover from, nor does it provide guidance for practitioners on when to attempt PMA-init recovery versus falling back to standard checkpoint rollback.

---

### Optimal Hyperparameters Established Only for One Model Family and Training Regime

**The assumption or constraint.** The paper's practical hyperparameter recommendations — N = 10, V scaling from 4B to 80B tokens depending on model size, SMA as the preferred method — are derived from ablations on a single model architecture family (ByteDance Seed MoE) trained on a single internal corpus with a WSD learning rate schedule at specific learning rates determined by scaling laws for that corpus. The paper acknowledges that learning rate interactions are underexplored:

> "In our experiments, we defaulted to using the optimal learning rate derived from the scaling law for model training, without extensively exploring the impact of learning rate on model merging. In our practice, we believe that training with a higher learning rate could lead to a better model through model merging, which aligns with the findings in [35]. However, due to the high computational cost, we did not further quantify the impact of learning rate on model merging in a more detailed manner." (Appendix C)

This is a candid admission that a key hyperparameter — the learning rate during the stable phase — is held fixed at one value per model size, and its interaction with merging effectiveness is unknown.

**The consequence.** A practitioner using a different learning rate schedule (e.g., cosine decay throughout, or a different peak learning rate in WSD) cannot rely on the paper's specific V and N recommendations. The paper's own hypothesis — that higher learning rates during the stable phase might improve merging by producing more diverse checkpoint deviations — suggests that the optimal V may depend on learning rate: a higher learning rate causes faster parameter movement, meaning checkpoints diverge more quickly, potentially requiring smaller V to avoid including overly distant (and possibly unstable) weights. Conversely, a lower learning rate produces slower parameter movement, potentially requiring larger V to ensure checkpoints are sufficiently different for averaging to provide benefit. Without learning rate ablations, these interactions are speculative.

The model architecture interaction is also underexplored. The paper demonstrates that merging benefits transfer from MoE to dense models (Appendix A, Figure 9), but the hyperparameter ablations (V, N, merging method) are conducted only on Seed-MoE-1.3B/13B (Section 4.3, Figure 5). Dense models have different training dynamics than MoE models (all parameters updated every step versus sparse expert activation), which could affect how quickly weights diverge between checkpoints and therefore what V is optimal. The paper's scale-dependent V guidelines are presented as general but are validated on a single architecture family.

**What evidence exists in the paper.** The hyperparameter ablations in Section 4.3 (Figure 5) on Seed-MoE-1.3B/13B. The paper reports scale-dependent V for three MoE sizes (Section 4.3) but does not ablate V for dense models. The learning rate interaction is explicitly flagged as a limitation (Appendix C) but not explored.

**Mitigation status.** The paper acknowledges the learning rate gap in Appendix C and defers it to future work. For model architecture, the paper provides dense model results showing that merging *works* (Appendix A) but does not provide hyperparameter guidance specific to dense architectures. The paper's recommendation to use SMA (Section 4.2) is partly a mitigation for hyperparameter sensitivity — since the merging method becomes irrelevant at convergence, practitioners don't need to tune between SMA/WMA/EMA — but the V and N recommendations remain specific to the tested configuration. A practitioner adapting PMA to a substantially different training setup would need to re-ablate V and N, which partially defeats the purpose of providing "practical pre-training guidelines."

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
