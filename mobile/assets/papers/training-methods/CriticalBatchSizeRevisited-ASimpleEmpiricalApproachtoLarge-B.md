# Critical Batch Size Revisited: A Simple Empirical Approach to Large-Batch Language Model Training

**ArXiv:** [2505.23971](https://arxiv.org/abs/2505.23971)

## 🎯 Pitch

This paper introduces a direct, empirical method for measuring the critical batch size (CBS) during language model training, sidestepping the strong assumptions required by previous proxies like the gradient noise scale. By using these CBS measurements to design a 'batch size warmup' schedule, the authors demonstrate that language models can be trained with significantly fewer gradient steps—improving efficiency and scalability—without compromising, and even slightly improving, final model performance. This advance provides both a practical tool for faster, more cost-effective large-batch training runs and a clearer understanding of optimization dynamics in modern large language models.

---

## 1. Executive Summary

This paper introduces a simple empirical method to directly measure the **critical batch size (CBS)** — the largest batch size that does not degrade the loss trajectory — via branched training from intermediate checkpoints, avoiding the strong assumptions (SGD optimizer, well-conditioned Hessian) required by the gradient noise scale proxy from McCandlish et al. (2018). Applying this method to OLMo 1B and 7B models, the authors find that the CBS starts near 0, increases rapidly in early training, and then plateaus around 4096 documents, with the trend largely independent of model size. This local CBS knowledge motivates **batch size warmup**: starting training with a small batch size and doubling it whenever the CBS has grown sufficiently, which enables training OLMo 1B to slightly better loss than the original small-batch run while using 43% fewer gradient steps — establishing that large-batch training can be made reliable without compromising token efficiency, provided the batch size never exceeds the CBS as it evolves over training.

## 2. Context and Motivation

### The Core Problem: We Don't Know How Large a Batch Size We Can Safely Use

Training large language models is an exercise in managing compute constraints. The fundamental tension this paper addresses is deceptively simple: **larger batch sizes enable faster training through increased data parallelism, but batch sizes that are too large degrade the token efficiency of training** — that is, you get worse loss for the same number of tokens processed. This matters because, in practice, the bottleneck in large-scale pretraining is often the number of sequential gradient steps rather than the total FLOPs. Each gradient step requires synchronization across accelerators, and reducing the number of steps (by processing more data per step) directly translates to wall-clock speedups. But reducing steps blindly, by naively cranking up the batch size, risks wasting expensive pretraining compute on a suboptimal loss trajectory.

The **Critical Batch Size (CBS) hypothesis**, formalized by McCandlish et al. (2018), provides a conceptual resolution to this tension:

> "There is some critical batch size $B^*$ up to which increasing the batch size (and appropriately modifying the learning rate) approximately preserves the loss trajectory as a function of tokens trained, but, above which, the loss trajectory degrades."

If this CBS $B^*$ exists and can be measured, it represents the sweet spot: the largest batch size you can use before diminishing returns in gradient estimation quality begin to hurt. Training at $B^*$ maximizes throughput without sacrificing model quality. The practical payoff is enormous — if you know $B^*$, you can train with fewer gradient steps (lower communication overhead, less synchronization) while matching or exceeding the loss of a smaller-batch run.

However, the paper identifies a critical gap: **we lack a reliable, assumption-light method to measure the CBS in realistic language model pretraining settings.** Without such a method, practitioners are forced to either (a) guess the batch size, risking degraded loss if they guess too high, or (b) remain conservative with small batch sizes, leaving potential throughput gains on the table.

### Why This Problem Matters

The importance of reliable CBS measurement extends beyond a single training run. The paper's context makes clear that the stakes are both practical and scientific:

**Practical: Throughput directly determines what models can be trained.** Large-scale pretraining runs (GPT-3, PaLM, OLMo, LLaMA) operate at the frontier of available compute. A reliable method to increase batch size without degrading loss translates immediately to: faster iteration cycles for researchers, lower total cost for equivalent model quality, and the ability to train larger models within fixed time budgets. The paper's demonstration of 43% fewer gradient steps with no loss degradation (Section 4.3) is precisely the kind of practical gain that changes how training runs are configured.

**Scientific: The CBS encodes fundamental information about the optimization landscape.** The CBS is not an arbitrary engineering parameter — it reflects properties of the loss surface, specifically the ratio of gradient variance to gradient norm (as McCandlish et al. showed under idealized assumptions). Understanding how the CBS evolves during training provides a window into how the optimization problem itself changes: early training has a small CBS (gradients are high-variance relative to their magnitude, so small batches suffice), while later training has a larger CBS (gradients are lower-variance, so larger batches are needed to make efficient progress). Measuring this evolution empirically, without relying on simplified theoretical models, is a contribution to the science of neural network optimization.

**Economic: The pretraining-vs-inference tradeoff depends on training efficiency.** While not the paper's focus, the ability to train efficiently at larger batch sizes feeds into broader questions about how to allocate compute between training larger models versus deploying smaller models with more inference-time compute — a question that the example paper in this prompt addresses directly.

### Prior Approaches and Where They Fall Short

The paper identifies two families of prior work on CBS measurement, each with significant limitations that motivate the authors' new approach.

#### Approach 1: Full Training Runs at Multiple Batch Sizes

The most direct way to measure the CBS is to train many models from initialization, each with a different batch size, to the same target loss, and observe at which batch size the token efficiency begins to degrade. This is the approach taken by Zhang et al. (2019) and Zhang et al. (2024). While conceptually straightforward, this method is **prohibitively expensive**: each training run consumes the full pretraining budget. For a model like OLMo 7B trained on trillions of tokens, launching even a handful of such runs to sweep batch sizes would multiply the total compute cost by the number of batch sizes tested. This makes the approach impractical for large-scale settings and impossible for exploratory analysis (e.g., measuring how the CBS changes at different points during training).

A second limitation of this approach is that it produces a **single, global CBS estimate** for the entire training run. It cannot reveal how the CBS evolves *during* training — whether it's small at initialization and grows, or remains constant, or oscillates. Without this local information, practitioners cannot design batch size schedules that adapt to the changing optimization landscape.

#### Approach 2: The Gradient Noise Scale Proxy (McCandlish et al., 2018)

McCandlish et al. (2018) proposed a much cheaper alternative: estimate the CBS indirectly from the **gradient noise scale**, defined as the ratio of the trace of the gradient covariance matrix to the squared norm of the true gradient:

$$B_{\text{simple}} = \frac{\text{tr}(\Sigma)}{\|G\|^2}$$

where $\Sigma$ is the covariance matrix of per-example gradients and $G$ is the true (population) gradient. This quantity can be estimated efficiently from gradient norms computed at two different batch sizes, without launching full training runs. The method gained significant traction — it was cited as influencing batch size selection in the GPT-3 technical report (Brown et al., 2020) and inspired follow-up work on improved noise scale estimators (Gray et al., 2023, 2024).

The problem, as the paper argues in Section 2, is that **the link between $B_{\text{simple}}$ and the true CBS relies on two strong assumptions that are violated in standard LM pretraining**:

**Assumption 1: SGD Optimizer.** McCandlish et al.'s derivation assumes the optimizer is plain stochastic gradient descent. But modern language models are universally trained with Adam (Kingma and Ba, 2017). This is not a minor technicality — Malladi et al. (2022) showed theoretically that the linear scaling rule between batch size and learning rate (Equation 2 in the paper) is appropriate for SGD, but a **square-root scaling rule** is more principled for Adam. Similarly, Li et al. (2024) provided theoretical support for square-root scaling with adaptive optimizers. If the learning rate scaling rule changes, the relationship between noise scale and CBS changes as well — $B_{\text{simple}}$ may no longer estimate the CBS correctly.

**Assumption 2: Well-Conditioned Hessian.** Even under SGD, the noise scale estimate that properly accounts for curvature involves the Hessian $H$:

$$B_{\text{noise}} = \frac{\text{tr}(\Sigma H)}{G^\top H G}$$

To simplify this to $B_{\text{simple}} = \text{tr}(\Sigma) / \|G\|^2$, McCandlish et al. must assume the Hessian is a multiple of the identity matrix — i.e., the optimization landscape is perfectly isotropic. This is a strong assumption unlikely to hold in practice. McCandlish et al. suggest informally that $B_{\text{simple}}$ might still be *correlated* with $B_{\text{noise}}$ even when the assumption is violated, but they provide no theoretical justification for why this correlation should exist, nor any guidance on what proportionality constant would relate the two. As the paper notes:

> "Even if this is true, it still poses a real problem for the noise scale methodology, since the goal of the method is to produce an absolute measure of $B^*$. It is unclear for practitioners what coefficient should be used to translate $B_{\text{simple}}$ to $B^*$ — and, more fundamentally, whether it is even valid to assume that such a coefficient exists."

This is not a hypothetical concern. The paper's own experiments (Figure 2 vs. Figure 3) show that the gradient noise scale **underestimates the CBS by several orders of magnitude** for OLMo models, and the qualitative trend does not reliably match the empirically measured CBS (especially for OLMo 7B, where the noise scale trends differently from the CBS). This empirical mismatch validates the authors' skepticism: the noise scale cannot be trusted as a CBS proxy in realistic LM training settings.

#### The Gap: No Reliable, Cheap, Local CBS Measurement

The synthesis of these limitations defines the gap the paper fills:

- **Full training runs** are too expensive and provide only a global CBS.
- **Gradient noise scale** is cheap but relies on assumptions violated by Adam and realistic loss landscapes, and the paper shows it gives quantitatively and sometimes qualitatively wrong answers.
- **Neither method** provides local CBS measurements at arbitrary points during training, which are needed to design adaptive batch size schedules.

### How This Paper Positions Itself

The paper positions itself not as a theoretical contribution to the analysis of the CBS, but as a **practical measurement methodology** that fills the gap described above. The authors are explicit about this framing in Section 1:

> "In this paper, we introduce a simple, empirical approach to directly measure the CBS and show how the CBS evolves over training."

The key words are "simple" and "empirical." Rather than deriving new theoretical relationships between batch size, gradient statistics, and loss, the paper proposes a **direct measurement via branched training**: take a pretrained checkpoint, launch several short training runs from it with different batch sizes (and appropriately scaled learning rates), and observe which batch sizes recover to the same loss after a fixed token budget. The CBS is the largest batch size that doesn't degrade loss. This approach is:

- **Direct**: It measures what we actually care about — the effect of batch size on loss — rather than a proxy (gradient noise) whose relationship to loss depends on unverified assumptions.
- **Local**: By applying it at different checkpoints, the method can track how the CBS evolves over training, enabling the design of dynamic batch size schedules.
- **Relatively cheap**: The branched training runs only need to be long enough to observe loss recovery (2B tokens in the paper's experiments), which is orders of magnitude cheaper than full training runs (hundreds of billions of tokens).

The paper's positioning is strengthened by its methodological transparency. The authors explicitly state the one assumption their method requires — the **Local Recovery Assumption** (Assumption 3): that if a batch size recovers in loss within the branched training window $\Delta$, it will continue to match smaller batch sizes for the remainder of training. This is a much weaker condition than assuming SGD optimization and an isotropic Hessian. Moreover, the assumption is empirically verifiable (by occasionally running longer branches) rather than being a structural claim about the optimization landscape.

The paper also positions its work relative to Zhang et al. (2024), who measured CBS by training from initialization with different batch sizes. The authors note that their method can be seen as a **dual formulation**: rather than fixing the target loss and measuring required steps (Zhang et al.'s approach), they fix the step budget $\Delta$ and measure achieved loss. This reformulation has the practical advantage that the training budget can be fixed in advance, making experimentation more predictable. More importantly, because the method branches from intermediate checkpoints rather than initializing from scratch each time, it can provide **local CBS estimates** throughout training — something Zhang et al.'s global approach cannot do.

Finally, the paper positions the downstream application — **batch size warmup** — as a natural consequence of the empirical CBS measurements, not as a separate methodological contribution. The finding that CBS starts near 0 and grows rapidly (Section 3.3, Figure 2) directly motivates starting training with a small batch size and increasing it as the CBS grows. This connects the measurement methodology to a concrete training recipe that practitioners can adopt, and the validation experiment (training OLMo 1B with 43% fewer steps and no loss degradation) demonstrates that the approach works end-to-end.

## 3. Technical Approach

### 3.1 Reader Orientation

This paper introduces a measurement-and-scheduling system for language model pretraining that determines, at any point during training, the largest batch size a model can use before its loss trajectory degrades, and then uses that knowledge to adaptively increase the batch size over the course of a training run. The core problem it solves is that practitioners currently lack a reliable, assumption-light method to know how aggressively they can scale up batch size (and thus training throughput) without silently damaging final model quality, and the shape of the solution is to directly measure loss sensitivity to batch size via short branched training runs rather than trusting theoretically-motivated but empirically unreliable proxies.

### 3.2 Big-Picture Architecture (Diagram in Words)

The system has three major components that operate sequentially across the lifecycle of a pretraining run:

1. **Critical Batch Size (CBS) Measurement via Branched Training** — given a model checkpoint and a base batch size, this component launches multiple short training branches from that checkpoint, each using a different batch size multiplier $k$ (with learning rate scaled by the square-root rule $f(k) = \sqrt{k}$ for Adam), trains each branch for a fixed token window $\Delta$, and identifies the largest multiplier $k^*$ whose final smoothed loss does not exceed the loss of all smaller batch sizes by more than a tolerance $\epsilon$. The output is the CBS $B^* = k^* \cdot B$, a local estimate of the largest batch size that preserves the loss trajectory at that point in training.

2. **CBS Evolution Tracking Across Checkpoints** — the branched training procedure is repeated at many pretraining checkpoints spanning the full training duration, producing a curve of $B^*$ as a function of tokens trained. This component reveals the qualitative pattern: CBS starts near 0 at initialization, grows rapidly in early training, and then plateaus. It also enables comparison across model sizes (1B vs. 7B) to assess whether CBS measurements from smaller models can inform larger-scale training.

3. **Batch Size Warmup Scheduler** — during a new pretraining run, this component initializes the batch size to a small value $B_0$ and periodically consults the pre-measured CBS curve (or an online estimator) to determine whether the CBS has grown sufficiently to support a larger batch size. Whenever $B^*_t > 2B_t$, it doubles the current batch size and scales the base learning rate by $\sqrt{2}$, following the square-root rule for Adam. This ensures the batch size never exceeds the CBS while allowing it to grow as the optimization landscape permits, reducing total gradient steps without degrading loss.

Information flows as follows: pretraining produces checkpoints → branched training from each checkpoint measures local CBS → CBS-vs-tokens curve is constructed → batch size warmup schedule is derived from the curve (manual thresholds or automated doubling rules) → new training run executes the schedule, starting small and increasing batch size at predetermined token counts.

### 3.3 Roadmap for the Deep Dive

- **First**, the formal definition of the CBS measurement problem and the branched training procedure, since this is the methodological core that everything else depends on.
- **Second**, the key implementation parameters ($\Delta$, $\epsilon$, loss smoothing, learning rate scaling rule) and the rationale behind each choice, since these determine the practical reliability of the measurements.
- **Third**, the formal connection between local CBS measurements and the global CBS scaling laws from prior work (Appendix D), since this connects the method to existing theoretical frameworks.
- **Fourth**, the batch size warmup algorithm and its operationalization, since this is the downstream application that validates the measurement methodology.
- **Fifth**, the experimental setup for the validation run (OLMo 1B with batch size warmup vs. small-batch and large-batch controls), including the specific doubling thresholds selected from the CBS curve.

### 3.4 Detailed, Sentence-Based Technical Breakdown

This is primarily a **measurement methodology and training recipe paper** whose core idea is that we can directly measure the critical batch size at any point in training by launching cheap branched training runs and observing which batch sizes recover in loss, and that these local measurements naturally motivate a batch size warmup schedule that increases training throughput without degrading final model quality.

---

#### The Branched Training CBS Measurement Procedure

The paper's central methodological contribution is a procedure for directly measuring the CBS from a model checkpoint without requiring full training runs or theoretical proxies. The procedure is defined in Section 3.1 as a sequence of steps that transform a checkpoint and a base configuration into a CBS estimate.

**Step 1: Choose a base configuration.** The method takes as input a training checkpoint that was produced with some original batch size $B$ and learning rate schedule $\eta$. This checkpoint represents a specific point in a training run — the CBS measured from it is local to that point. The base configuration defines the "small batch size" baseline against which larger batch sizes are compared.

**Step 2: Define the learning rate scaling rule.** The paper uses a function $f(k)$ that maps a batch size multiplier $k$ to a learning rate multiplier. The choice of $f$ depends on the optimizer:

- For SGD: $f(k) = k$ (linear scaling, as in McCandlish et al., 2018)
- For Adam: $f(k) = \sqrt{k}$ (square-root scaling)

The square-root scaling rule for Adam is justified by theoretical work from Malladi et al. (2022), who analyzed adaptive gradient algorithms through the lens of stochastic differential equations and showed that the linear scaling rule appropriate for SGD is not principled for Adam. Li et al. (2024, Equation 4) provided additional theoretical support for square-root scaling. The paper adopts this rule as a default, noting that "a linear scaling rule could also be used" in principle but is not recommended.

**What the scaling rule does operationally:** When we test a batch size that is $k$ times larger than the original, we multiply the base learning rate by $f(k)$. This scaling is critical because simply changing the batch size without adjusting the learning rate would change the effective step size, making it impossible to attribute changes in loss to the batch size change rather than to the modified optimization dynamics. The scaling rule attempts to keep the "effective learning dynamics" comparable across batch sizes, isolating the effect of batch size itself.

**Step 3: Launch branched training runs.** For a set of batch size multipliers $k \in \{k_1, k_2, ..., k_n\}$, the method creates independent training branches from the checkpoint. Each branch:

- Uses batch size $k \cdot B$ (the original batch size multiplied by $k$)
- Uses learning rate $f(k) \cdot \eta$ (the original learning rate multiplied by the scaling rule)
- Trains for a fixed token budget $\Delta$ (the "window size")
- Produces a final loss $L_k$ after $\Delta$ tokens, measured as the smoothed loss

The multipliers $k$ must span values both below and above the suspected CBS. In the paper's experiments, $k$ ranges from fractional values (0.0625, 0.125, 0.25) near initialization to integer values (1 through 8) at later checkpoints, with the specific ranges documented in Appendices A and shown in the loss-vs-batch-size plots in Figures 5 and 6.

**What "branched training" means physically:** Each branch starts from the exact same model parameters (the checkpoint), the exact same optimizer state (momentum and second-moment estimates in Adam), and the exact same data distribution. The only differences are the batch size and the learning rate. Each branch is trained independently — they do not share gradients or parameter updates. After $\Delta$ tokens of training, each branch produces a model with a potentially different loss. The key assumption (Assumption 3, discussed below) is that the relative ordering of these losses after $\Delta$ tokens predicts the relative ordering after the full training budget.

**Step 4: Apply loss smoothing.** Pretraining loss is noisy at the individual batch level — consecutive batches can show substantial variance due to data heterogeneity. To extract a reliable signal from the branched training runs, the paper applies exponential moving average (EMA) smoothing with parameter $\alpha = 0.5$:

$$\text{smoothed\_loss}_t = \alpha \cdot \text{raw\_loss}_t + (1 - \alpha) \cdot \text{smoothed\_loss}_{t-1}$$

where $\alpha = 0.5$ is the smoothing parameter that controls the tradeoff between responsiveness to recent changes and noise reduction. A value of 0.5 means each new raw loss observation contributes 50% to the smoothed value, which provides substantial noise reduction while still responding to genuine trends within a few steps.

**Step 5: Identify the critical multiplier $k^*$.** The CBS is defined operationally as the largest batch size that does not degrade loss relative to smaller batch sizes. Formally, the method identifies $k^*$ as:

> "the maximum $k$ such that, for all $k < k^*$, $L_{k^*} \leq L_k + \epsilon$"

where $\epsilon$ is a tolerance parameter for "similar" losses. In words: start from the smallest batch size multiplier and walk upward. As long as each larger batch size achieves loss no worse than all previously seen smaller batch sizes (within tolerance $\epsilon$), keep going. The first $k$ that violates this condition marks the point where batch size has become too large. The CBS is then $k^*$ — the last multiplier before degradation.

**Step 6: Define the CBS and scaled learning rate.** The final outputs are:

$$B^* = k^* \cdot B$$

$$\eta^* = f(k^*) \cdot \eta$$

where $B^*$ is the estimated critical batch size and $\eta^*$ is the appropriately scaled learning rate for training at $B^*$.

**What Figure 1 visualizes:** The three panels in Figure 1 show the loss-vs-batch-size curves for three representative checkpoints (20B, 419B, 629B tokens). The x-axis shows batch size (in documents), the y-axis shows smoothed loss after 2B tokens of branched training, and each point is one branched run. The red dotted line marks the identified $B^*$. At 20B tokens, loss is flat until roughly 3000-4000 documents, then rises — $B^*$ is approximately 3072. At 419B tokens, the flat region extends further right, and $B^*$ is approximately 4096. At 629B tokens, the pattern is similar. These plots demonstrate the core empirical phenomenon: there is indeed a region of batch sizes where loss is approximately flat (the "safe" region), followed by degradation at sufficiently large batch sizes.

---

#### The Local Recovery Assumption (Assumption 3)

The branched training method rests on one assumption, which the paper makes explicit:

> **Assumption 3: Local Recovery**
> "If the loss achieved by batch size $B^*$ recovers to match the loss with batch size $B < B^*$ after training for $\Delta$ tokens, the loss trajectories will remain the same beyond $\Delta$ as well."

**What this means in concrete terms:** When we launch a branch with batch size $k \cdot B$, the optimizer state (Adam's momentum buffers) was accumulated under the original batch size $B$. Switching to a different batch size causes an immediate "bump" in the loss as the optimizer adapts to the new gradient statistics. The branched training window $\Delta$ must be long enough for this transient to subside and for the loss to stabilize. If, after $\Delta$ tokens, the larger-batch branch has recovered to match the smaller-batch branches, we assume that this recovery is permanent — the larger batch size will continue to track the smaller batch size's loss trajectory for the remainder of training.

**Why this assumption is weak (in the paper's framing):** Unlike the assumptions required by the noise scale method (SGD optimizer, isotropic Hessian), Assumption 3 is not a claim about the structure of the loss landscape. It is an empirical claim about the predictive validity of a short training run. And, crucially, it is **testable** — one could occasionally run a longer branch to verify that the recovery observed at $\Delta$ tokens persists to larger token counts. The paper does not perform this verification explicitly, but the success of the batch size warmup experiment (where the CBS measurements from branched training successfully guide a full training run to good loss) provides indirect validation.

**What could violate this assumption:** If the loss landscape changes character after $\Delta$ tokens — for example, if a new "phase" of training begins where gradient statistics are qualitatively different — then the CBS measured locally might not predict behavior at longer horizons. The paper mitigates this risk by measuring the CBS at many checkpoints throughout training, so that any such phase transitions would be captured as changes in the CBS curve.

---

#### Key Implementation Parameters and Their Rationale

The branched training method has three user-specified parameters. The paper discusses each and justifies the chosen values:

**Window size $\Delta$.** Set to 2B tokens in the paper's experiments. The choice reflects a tension:

- *Too small:* The transient bump from changing batch size may not have subsided, leading to falsely low CBS estimates (the method would conclude a batch size is too large when it merely needs more time to recover).
- *Too large:* The branched training becomes expensive, defeating the purpose of a cheap measurement method.

The paper describes 2B tokens as "a small, conservative window size relative to our overall pretraining budget of 600B tokens." Since 2B is approximately 0.33% of the total budget, the branched training cost is negligible relative to a full run. The paper acknowledges that "the CBS measurements could, in principle, depend on $\Delta$, with larger values of $\Delta$ potentially producing larger CBS estimates" — this is listed as a direction for future work in Section 5.

**Loss tolerance $\epsilon$.** Set to 0.01. This parameter operationalizes "similar" losses: a larger batch size is considered to have "matched" smaller batch sizes if its smoothed loss is within 0.01 of the best loss among smaller batch sizes. The choice of 0.01 is described as "arbitrary," and the paper notes that "in principle, tolerance could be set in a more principled way using a statistical test in future work."

**What 0.01 means in context:** The raw loss values in Figure 1 range from roughly 2.65 to 3.50, so $\epsilon = 0.01$ represents approximately 0.3-0.4% relative tolerance. This is a fairly tight threshold — it means that even small degradations in loss will be flagged as exceeding the CBS. A looser tolerance would produce larger CBS estimates (more batch sizes would be considered "similar enough"), while a tighter tolerance would produce smaller estimates. The paper's choice errs on the conservative side (avoiding false claims that a large batch size is safe when it actually degrades loss).

**Smoothing parameter $\alpha$.** Set to 0.5 for the exponential moving average. This means the smoothed loss after each batch is an equally-weighted average of the current raw loss and the previous smoothed loss. With $\alpha = 0.5$, the effective window of the moving average is short (a few batches), which means the smoothed loss responds quickly to genuine changes while still filtering out batch-level noise.

**Why EMA rather than a simple moving average:** EMA gives more weight to recent observations, which is appropriate when we want to detect convergence (or lack thereof) within the $\Delta$ window. If loss is still decreasing at the end of $\Delta$, an EMA will reflect the most recent (lower) loss values, which is conservative — it makes it harder for a larger batch size to "match" smaller batch sizes if it's still recovering. This biases the CBS estimate downward, which is the safe direction.

---

#### Experimental Design for CBS Measurement

**Checkpoint selection.** The paper measures CBS at multiple checkpoints throughout OLMo 1B and 7B pretraining to track the CBS evolution. The specific checkpoints (documented in Appendix A) are:

- **OLMo 1B:** Step 0 (initialization), steps 10K-50K at 10K intervals, steps 100K-450K at 50K intervals — this covers the range from 0 to approximately 943B tokens.
- **OLMo 7B:** Step 0, steps 1K-3K at 1K intervals, steps 10K-30K at 10K intervals, and steps 72K, 150K, 200K, 239K, 300K, 350K, 400K, 477K — this covers the range from 0 to approximately 2000B tokens.

The denser sampling at the beginning of training (every 1K or 10K steps) reflects the paper's interest in capturing the rapid growth phase of the CBS.

**Batch size multiplier ranges.** The ranges of $k$ tested vary by checkpoint to keep the total number of branches manageable while covering the region of interest. Near initialization (step 0), $k$ starts at 0.0625 and goes up to 0.25 — this is because the CBS is very small early in training, and batch sizes even modestly larger than the base size may already exceed the CBS. Later in training, $k$ ranges from 1 to 8, reflecting the larger CBS. The specific ranges are documented in Appendix A and visualized in the loss-vs-batch-size plots in Figures 5 and 6.

**Representing CBS estimates.** For each checkpoint, the method produces an **interval** rather than a point estimate of the CBS:

- **Lower bound:** $B^* = k^* \cdot B$ (the largest batch size that did not degrade loss)
- **Upper bound:** The next larger batch size tested, $k_{\text{next}} \cdot B$, which did degrade loss

The plotted CBS value in Figure 2 is the **geometric mean** of these interval endpoints. Using the geometric mean (rather than the arithmetic mean) is appropriate because batch sizes are swept multiplicatively (powers of 2), so the geometric mean represents the midpoint on a log scale.

**Units.** The CBS is measured in "documents," where each document contains 4096 tokens (the pretraining sequence length used by OLMo). So a CBS of 4096 means a batch of 4096 × 4096 = approximately 16.8M tokens per gradient step.

---

#### Learning Rate Scaling Rule Selection

The paper's choice of square-root scaling for Adam is a substantive design decision that requires unpacking. The issue arises because the CBS hypothesis (McCandlish et al., 2018) was formulated in terms of SGD, where linear learning rate scaling ($\eta \propto B$) is theoretically justified: doubling the batch size halves the gradient noise variance, so doubling the learning rate keeps the effective step size in parameter space approximately constant.

For Adam, the situation is different. Adam normalizes gradients by a running estimate of their second moment, which means the effective step size is already scaled in a data-dependent way. Malladi et al. (2022) analyzed this through stochastic differential equations (SDEs) and showed that for adaptive optimizers, a square-root scaling rule ($\eta \propto \sqrt{B}$) is more principled than linear scaling. The intuition is that Adam's preconditioner already provides some invariance to gradient scale, so the learning rate should scale more slowly with batch size to avoid taking steps that are too large.

**Why this matters for CBS measurement:** If the wrong scaling rule is used, the branched training runs at different batch sizes are not "comparable" in their optimization dynamics — the larger batch size might underperform simply because its learning rate is set incorrectly, not because it inherently exceeds the CBS. The square-root rule attempts to isolate the effect of batch size on gradient estimation quality by keeping the effective optimization dynamics as similar as possible across branches.

**The paper's handling of this is pragmatic rather than prescriptive:** The authors note that "in principle a linear scaling rule could also be used" and present the square-root rule as a well-motivated default for Adam. They do not perform an ablation comparing linear vs. square-root scaling, which would be needed to definitively establish which is correct. However, the success of the batch size warmup experiment (which uses square-root scaling and achieves good results) provides indirect support.

---

#### Relating Local CBS to Global CBS (Appendix D)

Appendix D of the paper attempts to connect the local CBS measurements (how $B^*$ varies during training) to the "global" or "aggregate" CBS scaling laws studied in prior work (Zhang et al., 2024; Bergsma et al., 2025). This is not an empirical contribution but a theoretical bridge that shows the local measurements are consistent with existing scaling law findings.

**The question being asked:** Prior work measures a single CBS for an entire training run, defined as the largest fixed batch size that can be used throughout training without degrading the final loss. How does this global CBS relate to the local CBS curve $f(t)$ that varies with training progress $t$?

**The modeling assumption:** The paper assumes that the ideal fixed batch size $B$ is the one that minimizes the $L_2$ distance to the local CBS curve over the course of training. That is, we want to find a fixed $B$ such that the integrated squared difference between $B$ and $f(t)$ is minimized:

$$(R^2)^2 = \int_0^T (B - f(t))^2 dt$$

where $T$ is the total training duration in tokens, $f(t)$ is the local CBS at time $t$, and $B$ is the fixed batch size we are trying to choose.

**Proposition 1** shows that under this $L_2$ criterion (and assuming $f(0) = 0$, which the empirical measurements support), the optimal fixed batch size is simply the **average** of the local CBS over training:

$$B^* = \frac{1}{T} \int_0^T f(t) dt$$

**What this means intuitively:** If the CBS starts at 0 and grows to some final value $B^*_T$, the fixed batch size that best approximates the CBS curve (in the $L_2$ sense) is somewhere between 0 and $B^*_T$ — specifically, the time-average. This makes sense: using $B^*_T$ as a fixed batch size would be too large early in training (when CBS is small), while using 0 would be too small late in training (leaving throughput on the table). The average balances these considerations under squared-error loss. The paper notes a limitation of this view: "$L_2$ residuals may not be the right way to measure closeness to the CBS. In particular, it may be worse to overestimate the CBS compared to underestimate, as training above the CBS (with a scaled up learning rate) can be unstable."

**Propositions 2 and 3** then plug in specific functional forms for $f(t)$ that are consistent with the empirical observations:

- **Power-law CBS:** If $f(t) = t^c$ for some $c > 0$, then $B^* = \frac{T^c}{c+1}$. For $c = 1/2$ (square root growth), this gives $B^* = \frac{2}{3}\sqrt{T}$, which recovers the $\propto \sqrt{T}$ scaling law proposed by Zhang et al. (2024). This is significant because it means the local CBS measurements are mathematically consistent with the global scaling laws found by completely different experimental methods — the two approaches converge on the same functional relationship.

- **Logarithmic CBS:** If $f(t) = \log(t + 1)$, then $B^* = \frac{T}{T+1}\log(T+1) - 1$, which asymptotically scales as $\log T$.

The key insight from this analysis is that the local CBS growth pattern ($f(t)$ increasing over training) **predicts** that the global CBS should increase with total training budget $T$. This is because as $T$ grows, the average of $f(t)$ over a longer interval increases, since $f(t)$ is a monotonically increasing function. The paper does not claim to derive the exact scaling coefficient from the local measurements — that would require more precise functional form fitting — but shows that the qualitative relationship is consistent.

---

#### Batch Size Warmup: From CBS Measurements to Training Schedule

Section 4.1 operationalizes the CBS measurements into a concrete training algorithm. The core idea is straightforward but the implementation details matter.

**The algorithm** is defined as an iterative procedure over the course of a training run:

1. **Initialization:** At $t = 0$, set the batch size to $B_0 = B$ (the original small batch size) and the base learning rate to $\eta_0 = \eta$ (the original learning rate).

2. **Periodic CBS check:** After training for $t$ tokens, check whether the CBS at that point has grown sufficiently to support a larger batch size. Specifically, check whether $B^*_t > 2B_t$, where $B^*_t$ is the CBS measured at checkpoint $t$ and $B_t$ is the current batch size.

3. **Doubling step:** If $B^*_t > 2B_t$, update the batch size and base learning rate:
   $$B_{t+1} = 2B_t$$
   $$\eta_{t+1} = \sqrt{2} \eta_t$$

The factor of 2 in the doubling criterion ($B^*_t > 2B_t$) ensures that the new batch size (after doubling) will still be less than or equal to the CBS — the method doubles only when there is enough "headroom" that the doubled batch size remains safe.

**Why doubling and square-root scaling together maintain comparable optimization dynamics:** When we double the batch size, we multiply the learning rate by $\sqrt{2}$ (the square-root scaling rule). This means the learning rate grows more slowly than the batch size. The consequence is that if the batch size was at the CBS before doubling, and the CBS has exactly doubled, then both the batch size and the learning rate remain at their "optimal" values for the new CBS. The square-root rule ensures that the effective optimization dynamics don't change discontinuously at the doubling point.

**Operationalization in the paper's experiments:** The authors do not implement an automated CBS checking procedure. Instead, they pre-compute a batch size schedule based on manual inspection of the CBS curve in Figure 2:

> "Based on a manual reading of the CBS measurements in Section 3, we determine that the CBS reaches 2048 by 168B tokens and 4096 by 503B tokens."

This means the batch size warmup schedule is: start with $B = 1024$, double to $B = 2048$ at 168B tokens, double to $B = 4096$ at 503B tokens. After the second doubling, the batch size remains at 4096 for the remainder of training (and mid-training annealing).

**A footnote acknowledges an important methodological detail:** The thresholds (168B and 503B tokens) were determined from "preliminary CBS measurements at an earlier stage of the project" and changed slightly after those measurements were refined. The authors "deemed that it was not worth it to restart the expensive training runs because we do not think the results should be sensitive to a precise choice of threshold." This is a pragmatic concession — the exact doubling points are not critical as long as they occur after the CBS has genuinely grown past the doubled batch size. However, it also highlights that the method currently involves manual judgment rather than an automated, principled procedure.

**Connection to existing learning rate schedules.** The paper explicitly notes that batch size warmup "is compatible to overlay with existing learning rate schedules (in practice, the models we use follow a cosine schedule)." This is important: batch size warmup modifies the **base learning rate** — the peak of the learning rate schedule — but the schedule shape (cosine decay, linear warmup, etc.) operates on top of this base. At each doubling point, the base learning rate increases by $\sqrt{2}$, and the cosine schedule continues from the new base.

**Why powers of two?** The paper justifies this as a practical constraint: "the OLMo codebase requires the number of GPUs $g$ divides the batch size. Thus, doubling is convenient because it easily guarantees the new batch size remains a multiple of $g$." But the paper also acknowledges that in principle, one could update the batch size more frequently, e.g., "by setting updating the batch size the largest multiple of $g \leq B^*_t$." The power-of-two constraint is an implementation simplification, not a fundamental property of the method.

**Comparison to prior dynamic batch size work.** The paper distinguishes batch size warmup from two related but distinct ideas:

1. **McCandlish et al. (2018, Appendix D):** They experimented with dynamic batch sizes on the SVHN dataset but used the noise scale method to set the batch size, which the current paper argues is unreliable. They also only considered small-scale image classification, not large-scale language model pretraining.

2. **Smith et al. (2018):** They proposed replacing learning rate decay with increasing batch size — as the learning rate decreases over training, simultaneously increase the batch size to maintain optimization dynamics. This is "conceptually related" to batch size warmup but differs in two key ways: (a) it operates on top of or instead of learning rate decay, whereas batch size warmup overlays on existing schedules; (b) it uses linear scaling (appropriate for SGD) rather than square-root scaling (appropriate for Adam); and (c), most crucially, it does not ensure that the batch size never exceeds the CBS — it increases batch size on a fixed schedule regardless of whether the optimization landscape can support it.

The paper frames batch size warmup as a method that ensures the batch size "never exceeds the critical batch size" (Section 4.1), which is the key safety property that distinguishes it from prior dynamic batch size proposals.

---

#### Experimental Validation Design

The validation experiment (Section 4.2) compares three training runs, all using OLMo 1B with default pretraining hyperparameters except for the batch size and learning rate modifications:

**Batch Size Warmup (the proposed method):**
- Initial batch size: $B = 1024$
- Initial base learning rate: $\eta = \sqrt{2} \cdot 0.0004$
- Doubling schedule: double to 2048 at 168B tokens, double to 4096 at 503B tokens
- Learning rate scaling: $\sqrt{2}$ multiplier at each doubling
- Total gradient steps: 43% fewer than the small-batch control

**Small-Batch Control:**
- Fixed batch size: $B = 1024$
- Fixed base learning rate: $\eta = \sqrt{2} \cdot 0.0004$
- This represents the "safe" baseline — a batch size known to be below the CBS throughout training, since the CBS measurements show $B^* \ll 1024$ only at the very beginning of training
- The learning rate is scaled up from the default OLMo 1B value (0.0004) by $\sqrt{2}$ to account for the batch size being doubled from the default 512 to 1024, following the square-root rule

**Large-Batch Control:**
- Fixed batch size: $B = 4096$
- Fixed base learning rate: $\eta = 2\sqrt{2} \cdot 0.0004$
- This represents a naive attempt at large-batch training — using a large batch size from initialization, with learning rate scaled up accordingly
- The large batch size exceeds the CBS for the early part of training (since CBS starts near 0), so we expect degraded final loss

**Why 1024 as the base batch size rather than 512:** The default OLMo 1B configuration uses a batch size of 512 with learning rate 0.0004. The authors chose to start their experiments at 1024 (double the default) to have room for two doublings (to 2048 and then 4096) while staying within the range of batch sizes they had measured. The learning rate was scaled up accordingly by $\sqrt{2}$ to maintain comparable optimization dynamics.

**Training duration and evaluation protocol.** All three runs are trained for 608B tokens (41% of the full 4T token OLMo 1B schedule — the authors note they did not have resources to replicate the full training run). Evaluation occurs at two points:

- **After pretraining:** The loss at the end of the 608B token pretraining phase, averaged over the last 10B tokens to reduce noise.
- **After mid-training:** Starting from the final pretraining checkpoint, the learning rate is linearly annealed to 0 over 50B tokens with the batch size fixed at its final value. This mid-training (or annealing) phase is a standard part of the OLMo pipeline (OLMo et al., 2025), and the loss after annealing is taken as the primary metric since it more closely reflects the model quality that would be obtained in a complete training run.

The mid-training phase deserves emphasis because OLMo et al. (2025) showed that annealing can induce significant loss improvements for partial training runs. A model that is slightly behind after pretraining might catch up during annealing, or a model that is slightly ahead might maintain its advantage. The mid-training loss therefore provides a more robust comparison than the pretraining loss alone.

**Downstream evaluation.** Beyond training loss, the paper evaluates three types of out-of-distribution performance (Table 2):

- **Cross-entropy loss on C4** (Dodge et al., 2021): a standard web-text validation set.
- **Cross-entropy loss on The Pile** (Gao et al., 2020): another common validation set drawn from diverse sources.
- **Bits-per-byte (BPB) on downstream QA tasks:** Following Bhagia et al. (2024), the method computes the loss (in BPB) on the correct answers of multiple multiple-choice and generation datasets, including ARC-Easy, ARC-Challenge, MMLU, HellaSwag, PIQA, and others (full list in Appendix E). BPB normalizes loss by the byte length of the target answers, making it comparable across tasks with different answer lengths.

**Why BPB rather than accuracy:** The paper follows Bhagia et al. (2024) in using BPB because it provides a more fine-grained signal than accuracy for comparing pretraining runs. Accuracy on many of these benchmarks saturates or is noisy for models at the 1B scale, whereas BPB provides a continuous measure of how well the model assigns probability to correct answers, which is more sensitive to small differences in pretraining quality.

---

#### Summary of Design Choices and Their Justifications

The paper's methodology is characterized by a series of pragmatic choices, each with a stated rationale:

- **Direct loss measurement over theoretical proxy:** The noise scale method requires assumptions (SGD, isotropic Hessian) that are violated in practice, and the paper's own experiments show it underestimates the CBS by orders of magnitude. By directly measuring what we care about (loss) rather than a proxy (gradient statistics), the method avoids dependence on unverified theoretical models.

- **Square-root learning rate scaling for Adam:** The linear scaling rule from McCandlish et al. (2018) is theoretically justified only for SGD. For Adam, square-root scaling is more principled (Malladi et al., 2022; Li et al., 2024). This choice ensures that batch size comparisons are fair — differences in loss across branch sizes reflect genuine batch size effects rather than suboptimal learning rate settings.

- **Fixed token window $\Delta$ rather than fixed target loss:** By measuring loss after a fixed budget rather than measuring steps to reach a target loss, the method allows all branched runs to have the same cost and duration, making experimentation planning straightforward. This is described as a "dual" formulation to the approach of Zhang et al. (2019, 2024), who fixed the target loss and measured required steps.

- **Conservative tolerance $\epsilon = 0.01$:** The tight tolerance means the method errs on the side of reporting smaller CBS values, which is the safe direction — it's better to slightly underestimate the CBS and leave some throughput on the table than to overestimate it and silently degrade loss.

- **Exponential moving average smoothing with $\alpha = 0.5$:** EMA provides noise reduction while remaining responsive to genuine trends. The relatively high $\alpha$ (0.5, giving equal weight to current and past) means the smoothed loss adapts quickly, which is appropriate for detecting convergence within the $\Delta$ window.

- **Measurement at multiple checkpoints rather than only at initialization:** This captures the evolution of the CBS over training, which is essential for designing adaptive batch size schedules. Prior work (Zhang et al., 2024) measured only from initialization, producing a single global CBS estimate.

- **Geometric mean for CBS point estimates:** Since batch size multipliers are swept multiplicatively, the geometric mean of the lower and upper bounds represents the midpoint on a log scale, which is the natural scale for multiplicative quantities.

- **Powers of two for batch size increases:** Purely a practical constraint from the OLMo codebase (batch size must be divisible by the number of GPUs), not a fundamental limitation of the method. The paper explicitly notes that more frequent updates would be possible in principle.

- **Manual threshold selection for the warmup schedule:** The doubling thresholds were chosen by inspecting the CBS curve rather than through an automated procedure. The paper acknowledges this as a limitation and suggests "systematic methodology for choosing batch size warmup thresholds given the CBS measurements" as future work.

- **Mid-training annealing evaluation:** Following OLMo et al. (2025), the paper reports both pretraining and post-annealing loss because annealing can substantially change the relative performance of different training configurations. The post-annealing loss is taken as the primary metric.

## 4. Key Insights and Innovations

### Innovation 1: Direct Empirical CBS Measurement as a Methodological Break from Proxy-Based Estimation

The paper's most fundamental intellectual contribution is not any specific finding about the CBS, but rather the **methodological reframing** of how the CBS should be measured in the first place. Prior work (McCandlish et al., 2018; Gray et al., 2023, 2024) approached CBS measurement as an inference problem: derive a theoretical relationship between the CBS and some more easily computable quantity (the gradient noise scale), then estimate that quantity from gradient statistics. This is the standard scientific playbook — replace a hard-to-measure target with a proxy whose relationship to the target is justified by a model of the underlying system.

The paper argues that this playbook breaks down for language model pretraining because the model of the system — specifically, the assumptions of SGD optimization and a well-conditioned Hessian — is **demonstrably false** in the relevant regime. The theoretical derivation that connects gradient noise to CBS assumes an optimizer (SGD) that practitioners do not use, and a loss landscape geometry (isotropic curvature) that is unlikely to hold in deep networks. The paper's own experiments (Figure 3 vs. Figure 2) provide the empirical nail in the coffin: the gradient noise scale underestimates the measured CBS by orders of magnitude and shows qualitatively different trends for the 7B model, meaning the proxy fails both in absolute value and in relative behavior.

The conceptual move the paper makes in response is subtle but significant: **instead of trying to fix the proxy (by deriving better theoretical relationships or more sophisticated estimators), the paper abandons proxy-based estimation entirely and measures the CBS directly — by actually training at different batch sizes and observing what happens to the loss.** This is a shift from an inferential paradigm (estimate an unobservable quantity from observable statistics) to an experimental paradigm (perturb the system and measure the outcome of interest). The branched training procedure introduced in Section 3.1 is the operationalization of this shift.

What makes this more than just "throwing compute at the problem" is the **dual reformulation** that makes direct measurement affordable. Prior work that also measured CBS directly (Zhang et al., 2019, 2024) did so by launching full training runs from initialization at each candidate batch size — a cost that scales linearly with the number of batch sizes tested and the total training budget. The paper's key maneuver is to fix the token budget $\Delta$ (a small window) and measure achieved loss, rather than fixing a target loss and measuring required tokens. This dual formulation means that branched training runs only need to be long enough to observe loss convergence (2B tokens in the paper's experiments, versus hundreds of billions for full runs), making the measurement cost **independent of the total training budget**. A CBS measurement that would require trillions of tokens of total training under the global approach can be done with ~2B tokens per branch under the local approach — a reduction of two to three orders of magnitude.

This methodological contribution is **fundamental rather than incremental** because it changes what kinds of CBS questions can feasibly be asked. The global approach can only answer "what single batch size should I use for my entire training run?" The local approach can answer "how does the CBS evolve at each point in training?" and "does the CBS depend on model size?" and "can I use CBS measurements from a small model to inform a large model run?" — all questions that would be computationally prohibitive under the global paradigm. The paper's ability to produce Figure 2 (CBS curves over the full training trajectory for two model sizes) is a direct consequence of this methodological shift, not just an application of existing methods to a new setting.

### Innovation 2: CBS as a Dynamic Rather Than Static Property of Training

The paper's second conceptual contribution is the empirical demonstration — made possible by Innovation 1 — that the CBS is **not a fixed property of a model and dataset, but a function that evolves systematically over the course of training.** Figure 2 shows a consistent pattern for both OLMo 1B and 7B: the CBS starts near 0 at initialization, grows rapidly in the first ~50B tokens, and then plateaus around 4096 documents for the remainder of training. This finding reshapes how the CBS should be understood.

Prior work, both theoretical and empirical, treated the CBS as a **global constant** — a single number that characterizes the entire training run. McCandlish et al. (2018) derived the CBS in terms of the gradient noise scale, which is a property of the data distribution and model architecture computed at initialization. Zhang et al. (2024) measured "the" CBS by training from initialization with different fixed batch sizes and observing which batch sizes achieved equivalent final loss. Both approaches implicitly assume that the batch size constraints that apply at the beginning of training are the same constraints that apply at the end.

The paper's finding that the CBS grows over training fundamentally challenges this view, and in doing so, **explains a subtle tension in the literature**. If the CBS is small at initialization and grows over time, then a fixed batch size that is appropriate for the later stages of training will be **too large for the early stages**, potentially destabilizing optimization or degrading the final loss in ways that are not recoverable. This provides a mechanistic explanation for why naive large-batch training often fails: it is not that the batch size is too large in an absolute sense, but that it is too large *relative to the CBS at the beginning of training*. The large-batch control in the paper's own experiments (Section 4.3) demonstrates this: training with $B = 4096$ from initialization degrades final loss compared to the small-batch control, even though 4096 is below the CBS for most of training — the early phase where the CBS is small is what causes the degradation.

This insight matters beyond the specific CBS context because it exemplifies a more general point about **optimization in deep learning**: the difficulty of the optimization problem changes over time, and methods that treat it as static (fixed batch sizes, fixed learning rates, fixed regularization) may be leaving performance on the table. The CBS is not unique in this regard — learning rate schedules (warmup, cosine decay) already reflect the understanding that optimization dynamics evolve. The paper's contribution is to show that batch size belongs in the same category of "things that should adapt over training," and to provide an empirical method for determining *how* it should adapt.

The connection to the $\sqrt{T}$ scaling law from prior work (Zhang et al., 2024; Bergsma et al., 2025) is also significant at the conceptual level. Appendix D shows that if the local CBS grows as $\sqrt{t}$ during training, then the global CBS (the fixed batch size that best approximates the local CBS over the full run) should scale as $\propto \sqrt{T}$, where $T$ is the total training budget. This means the paper's local measurements are **mathematically consistent** with the global scaling laws found by completely different experimental methods — the two approaches converge on the same functional relationship, but the local approach provides the mechanistic explanation (the CBS grows during training, so longer training runs average over more large-CBS time) for *why* the global CBS scales with $T$. This is a case where a new measurement methodology doesn't just produce new findings but also **retrospectively explains** findings that were previously empirical regularities without clear mechanistic underpinning.

The paper is careful not to overclaim the precision of this connection, noting in Appendix D that the $L_2$ distance minimization used to derive the relationship may not be the right objective (since exceeding the CBS is likely worse than undershooting it). But the qualitative convergence — local CBS growth predicts global CBS scaling — is a form of validation that the method is measuring something real rather than an artifact of the $\Delta$ window or the $\epsilon$ tolerance.

### Innovation 3: Batch Size Warmup as a Principled Rather Than Heuristic Training Strategy

The paper's third contribution is to transform batch size warmup from a heuristic trick into a **principled training strategy grounded in empirical CBS measurements**. This is a shift in kind, not just degree.

Dynamic batch sizes have been explored before. McCandlish et al. (2018, Appendix D) experimented with increasing batch size during training on SVHN, using their noise scale method to set the schedule. Smith et al. (2018) proposed replacing learning rate decay with batch size increases, motivated by the linear scaling rule between batch size and learning rate in SGD. These are both examples of *heuristic* dynamic batch size methods: they follow a plausible intuition (the CBS might grow, so increase the batch size; the learning rate decays, so increase the batch size to compensate), but they lack a mechanism for determining *when* and *by how much* the batch size should increase for a specific model and dataset.

The paper's batch size warmup method (Section 4.1) is different in kind because it is **measurement-driven rather than schedule-driven**. The decision to double the batch size is not based on a predetermined token count or a fixed schedule — it is based on an explicit check: "has the CBS, measured empirically at this point in training, grown sufficiently to support a larger batch size?" The paper's operationalization (manually reading doubling thresholds from the CBS curve in Figure 2) is admittedly simplified, but the *principle* is what matters: the batch size schedule should be a function of the measured CBS curve, not a generic recipe applied identically to all training runs.

The significance of this shift becomes clear when considering the failure mode of heuristic approaches. If you increase the batch size on a fixed schedule (e.g., double every 100B tokens), you might increase it *before* the CBS has grown enough to support the new batch size, leading to degraded loss that cannot be recovered. Or you might increase it *after* the CBS has grown, leaving throughput on the table for some period. The measurement-driven approach guarantees (subject to measurement accuracy) that the batch size never exceeds the CBS, which is precisely the safety property that the CBS hypothesis promises: training at or below the CBS preserves the loss trajectory.

The empirical validation (Section 4.3) is notable not just for the 43% gradient step reduction, but for the **direction** of the result: batch size warmup achieves *slightly better* loss than the small-batch control, not just equivalent loss with fewer steps. The small-batch control trains at $B = 1024$ for the entire run, which is below the CBS for most of training — meaning its batch size is smaller than it could safely be, and it's making less progress per gradient step than it could. Batch size warmup corrects this by increasing the batch size as the CBS grows, effectively "catching up" to the CBS and using the full safe batch size for the later stages of training. The slight improvement in final loss (0.0053 after mid-training; Table 1) is consistent with this interpretation: batch size warmup doesn't just match the small-batch run with fewer steps, it actually makes slightly better use of the same token budget by keeping the batch size closer to the CBS throughout.

The large-batch control result provides the negative control that makes this interpretation coherent: training at $B = 4096$ from initialization (exceeding the early CBS) degrades final loss, even though 4096 is safe for most of training. The degradation from the early phase is not recovered by the later safe phase — early optimization mistakes have lasting consequences. This is consistent with the broader finding in the optimization literature that the early phase of training is disproportionately important for final model quality, and it provides a concrete example of why global CBS estimates (which would suggest 4096 is safe because it's below the average CBS) can be misleading.

The batch size warmup contribution is **incremental as a technique** (dynamic batch sizes existed before) but **fundamental as a principle** (the batch size schedule should be derived from empirical CBS measurements, not from a generic recipe). The paper provides both the measurement methodology to make this principle operational and the validation experiment to show it works end-to-end, but the core insight — that batch size adaptivity should be measurement-driven — is the lasting conceptual contribution.

### Innovation 4: The Gradient Noise Scale Is Empirically Invalid as a CBS Proxy

While the paper's primary contributions are constructive (a new measurement method, new findings about CBS evolution, a new training strategy), its **negative result** regarding the gradient noise scale is of comparable intellectual significance. The paper provides clear empirical evidence that the noise scale method from McCandlish et al. (2018) — which has been cited as influencing batch size selection in GPT-3 (Brown et al., 2020) and has spawned follow-up work on improved estimators (Gray et al., 2023, 2024) — does not produce valid CBS estimates in realistic language model pretraining settings.

The strength of this negative result comes from the **direct comparison** enabled by the paper's measurement methodology. Prior work on noise scale estimation evaluated the method by its internal consistency (do different estimators agree?) or by its theoretical properties (does the estimator converge under certain assumptions?). But without an independent, assumption-light measurement of the CBS, it was impossible to assess whether the noise scale was actually measuring the right thing. The paper's branched training method provides that independent measurement, and the comparison is stark: Figure 2 shows the CBS reaching ~4096 by mid-training, while Figure 3 shows the noise scale hovering around 10–50 — a discrepancy of roughly two orders of magnitude.

The qualitative mismatch is equally significant. For OLMo 7B, the noise scale trends downward over training while the measured CBS trends upward — opposite directions. For OLMo 1B, the qualitative trends are more similar (both increase), but the quantitative mismatch remains large. The authors are appropriately cautious in their conclusion: "since this similarity is not found for both models, we conclude that, in general, the noise scale cannot be used reliably as a proxy for the CBS."

This negative result matters beyond the specific method it critiques because it illustrates a broader lesson about **the gap between theoretical models and empirical reality in deep learning optimization**. The McCandlish et al. derivation is mathematically correct under its assumptions. The problem is not with the derivation but with the assumptions' applicability to modern training setups (Adam, deep transformers, large-scale data). This is a recurring pattern in the optimization literature — theoretically elegant results derived for convex optimization or simple architectures often fail to transfer to the non-convex, high-dimensional, adaptive-optimizer regime of modern deep learning. The paper's contribution is to provide a clean, empirically grounded example of this failure mode for a method that had gained significant adoption.

The negative result also has **practical implications for the research agenda around test-time and training-time compute optimization**. If the noise scale is not a reliable CBS estimator, then the body of work building on it — improved estimators, theoretical analyses linking noise scale to optimal batch size, practical recommendations for large-scale training — rests on a shaky foundation. The paper's findings suggest that these efforts, while well-motivated, may be optimizing a proxy that doesn't correspond to the quantity of actual interest. This redirects research attention toward direct measurement approaches (like branched training) rather than increasingly sophisticated proxy estimators.

The paper handles this critique constructively. Rather than simply dismissing the noise scale method, the authors note in Section 5 that "it would be interesting to further investigate the conditions under which noise scale might be a meaningful CBS proxy and used for batch size warmup." This leaves open the possibility that the noise scale works under some conditions (smaller models? different optimizers? different stages of training?), while establishing that it cannot be trusted as a general-purpose method for the LLM pretraining regime studied in the paper. This is a precise, evidence-backed negative result rather than a blanket dismissal — the kind of contribution that clarifies the boundaries of existing methods and guides future research toward more productive directions.

## 5. Experimental Analysis

### Evaluation Methodology

- **Dataset.** All CBS measurement experiments and training runs use the OLMo pretraining data (Dolma; Soldaini et al., 2024), consisting of approximately 3 trillion tokens of openly available web text, academic papers, code, and books. For out-of-distribution evaluation, the paper uses C4 (Dodge et al., 2021) and The Pile (Gao et al., 2020) as held-out validation sets, plus a suite of downstream QA datasets for bits-per-byte evaluation following Bhagia et al. (2024): ARC-Easy, ARC-Challenge, CommonsenseQA, HellaSwag, MMLU, PIQA, Social IQa, WinoGrande, GSM8K, Minerva, Humaneval, MBPP, and Copycolors (full list in Appendix E). The specific test/train splits used for CBS measurement are the standard training data stream that OLMo 1B and 7B were originally trained on.

- **Base model(s).** All experiments use the OLMo 1B and OLMo 7B models (OLMo et al., 2025; Groeneveld et al., 2024), chosen because their pretraining data and model weights are openly available — a prerequisite for the branched training method, which requires access to intermediate pretraining checkpoints and the ability to continue training from them on the original data distribution. The OLMo 1B model is used for the main validation experiments (batch size warmup vs. controls), while both 1B and 7B are used for the CBS evolution measurements. The authors state that these models are studied "due to their open weights and data" (Section 1), and the choice is pragmatic rather than based on any claim about their representativeness.

- **Metrics.** The primary metric throughout is **smoothed pretraining loss** — cross-entropy loss on the training data, smoothed via exponential moving average (EMA) with α = 0.5 to reduce batch-level noise. For the batch size warmup validation, the metric reported is the loss averaged over the last 10B tokens of both the pretraining phase and the mid-training (annealing) phase. Downstream evaluation uses three additional metrics: **cross-entropy loss on C4**, **cross-entropy loss on The Pile**, and **bits-per-byte (BPB) on downstream tasks**, computed as the loss (in bits per byte) on the correct answers of multiple QA and generation datasets. BPB normalizes by the byte length of the target answer, enabling aggregation across tasks with varying answer formats. For CBS measurement, the metric that determines whether a batch size has "recovered" is the smoothed loss at the end of the ∆ token window, compared across batch-size branches with tolerance ε = 0.01.

- **Baselines.** The batch size warmup experiment (Section 4) compares three training configurations:
  - **Small-Batch Control**: Fixed batch size B = 1024, base learning rate η = √2 · 0.0004, representing the conservative approach of training at a batch size known to be below the CBS throughout training. The learning rate is scaled up from the default OLMo 1B value (0.0004 at B = 512) using the square-root rule.
  - **Large-Batch Control**: Fixed batch size B = 4096, base learning rate η = 2√2 · 0.0004, representing the naive approach of training at a large batch size from initialization without regard for the CBS. This configuration exceeds the CBS during early training, so it is expected to degrade final loss.
  - **Batch Size Warmup (Proposed)**: Starts at B = 1024 and doubles to 2048 at 168B tokens, then to 4096 at 503B tokens, with learning rate scaled by √2 at each doubling. This represents the measurement-driven approach where batch size increases only after the CBS has grown sufficiently.
  
  The CBS measurement experiments do not have explicit baselines in the traditional sense — instead, the gradient noise scale method (McCandlish et al., 2018) serves as a point of comparison for whether CBS measurements are consistent with prior proxy-based approaches. The paper computes the noise scale at each checkpoint using the estimator from McCandlish et al. (2018, Appendix A) with large batch size B_big = 64 and small batch size B_small = 1, averaging gradient statistics over 4096 batches.

- **Generation budget / compute accounting.** The central unit of compute in the CBS measurement experiments is the **branched training window ∆**, set to 2B tokens per branch. For the OLMo 1B CBS curve (Figure 2, left), the paper tested 13 checkpoints with varying numbers of batch size multipliers per checkpoint (ranging from 3 multipliers at initialization to 8 multipliers at later checkpoints, as documented in Appendix A), yielding a total branched training cost of approximately 13 checkpoints × ~5–8 branches × 2B tokens ≈ 150–200B tokens of additional training — roughly 25–33% of a single full training run (608B tokens). The OLMo 7B CBS curve (Figure 2, right) covers 14 checkpoints with similar branch counts. For the batch size warmup validation, all three training runs (warmup, small-batch control, large-batch control) are trained for 608B tokens of pretraining plus 50B tokens of mid-training, making the compute cost identical in tokens across configurations; the efficiency gain is measured in **gradient steps saved** (43% fewer for batch size warmup vs. small-batch control), since each gradient step requires synchronization across GPUs and fewer steps directly translate to wall-clock speedup. The paper documents (Appendix A) that 1B branched runs used a single node of H100 GPUs and 7B runs used 8 nodes.

- **Cross-validation / statistical protocol.** The branched training CBS measurement does not use cross-validation — each CBS estimate is a single measurement at a specific checkpoint, bounded by an interval whose endpoints are the largest batch size that didn't degrade loss and the next-larger tested batch size that did (Section 3.2). The CBS values plotted in Figure 2 represent the geometric mean of these interval endpoints. For the gradient noise scale (Figure 3), the paper reports 95% confidence intervals constructed by first estimating confidence intervals for the numerator S (assumed to follow an exponential distribution) and the denominator ‖G‖² (assumed to follow a normal distribution, based on manual inspection of their histograms; Figure 6), then computing the ratio interval as [aS / b‖G‖², bS / a‖G‖²] where a and b denote the lower and upper bounds respectively (Appendix B). The batch size warmup validation runs are single training runs (not averaged over random seeds), and the loss values in Table 1 represent averages over the last 10B tokens within each run — the paper does not report standard deviations or confidence intervals for these values, which is a limitation for assessing whether the observed differences (e.g., 0.0053 loss improvement after mid-training) are statistically meaningful or within run-to-run noise.

### Main Quantitative Results

#### CBS Evolution During Training and Model Size Dependence

The headline finding from the branched training experiments is that the CBS evolves in a consistent pattern across both model sizes. Figure 2 (left, OLMo 1B; right, OLMo 7B) reveals a three-phase trajectory:

**Phase 1: Near-zero at initialization.** At step 0 (before any training), the CBS is extremely small — the paper reports that for OLMo 1B, the CBS at step 0 is approximately 50–100 documents (200K–400K tokens). This means that even modest batch sizes would exceed the CBS at initialization, and the optimization can only effectively use very small batches.

**Phase 2: Rapid growth in early training.** The CBS grows quickly within the first ~50B tokens (approximately 50K steps for OLMo 1B, 12K steps for OLMo 7B), rising from near 0 to roughly 2048 documents for both model sizes. The growth during this phase is approximately exponential or power-law in character — the plotted CBS values rise steeply on the log-scale y-axis in Figure 2.

**Phase 3: Plateau at late training.** After the rapid growth phase, the CBS plateaus around 4096 documents (approximately 16.8M tokens) for both OLMo 1B and 7B, with only minor fluctuations for the remainder of training (out to 943B tokens for 1B and 2000B tokens for 7B). The plateau value is roughly consistent across model sizes: the OLMo 1B CBS stabilizes at approximately 4096 documents by 200–400B tokens, and the OLMo 7B CBS reaches a similar plateau at approximately 4096 documents by 300–600B tokens.

**Model size independence.** The qualitative similarity of the CBS curves for 1B and 7B models (Figure 2) is one of the paper's key findings: the CBS does not appear to depend strongly on model size. Both models start near 0, grow rapidly, and plateau around 4096. The paper notes this is "in line with past findings using different methodology (Zhang et al., 2024)" (Section 1, contribution 2). This finding has practical significance because it suggests that CBS measurements from smaller, cheaper training runs can inform batch size decisions for larger, more expensive runs — the CBS is a property of the optimization dynamics at a given stage of training rather than a property of the model scale.

The CBS interval endpoints (documented in the geometric mean plots in Figure 2) show that the uncertainty in CBS estimates is reasonable relative to the overall trend. At early checkpoints where the CBS is small, the lower and upper bounds are close (since the tested batch size multipliers are closely spaced in that regime; Appendix A documents k ranging from 0.0625 to 0.25 at step 0). At later checkpoints, the intervals widen (since the multipliers are integer-valued and span larger gaps, e.g., k = 1 through 8), but the central trend is clear.

#### Gradient Noise Scale vs. Measured CBS

Figure 3 (left, OLMo 1B; right, OLMo 7B) compares the gradient noise scale computed via the McCandlish et al. (2018) estimator to the empirically measured CBS from Figure 2. The findings demonstrate a clear failure of the noise scale as a CBS proxy:

**Quantitative mismatch.** The gradient noise scale underestimates the CBS by roughly two orders of magnitude. For OLMo 1B, Figure 2 shows the CBS reaching ~2048–4096 by mid-training, while Figure 3 shows the noise scale hovering in the range of roughly 10–50 (with wide confidence intervals) throughout training. For OLMo 7B, the mismatch is even larger: the CBS reaches ~4096, while the noise scale is approximately 5–20. The noise scale estimate never approaches the measured CBS at any point in training for either model.

**Qualitative mismatch for 7B.** For OLMo 7B (Figure 3, right), the noise scale shows a downward trend over the course of training (decreasing from approximately 20–30 at early steps to approximately 5–10 at later steps), while the measured CBS shows the opposite trend (increasing from near 0 to ~4096). This directionality mismatch means that not only does the noise scale get the absolute value wrong, but it also gets the qualitative behavior wrong for the larger model — it would predict that the CBS decreases or remains flat over training, which is the opposite of what direct measurement reveals.

**Partial qualitative match for 1B.** For OLMo 1B (Figure 3, left), the noise scale does increase somewhat over training (from approximately 5–10 at early steps to approximately 30–50 at later steps), which is qualitatively similar to the CBS trend. However, the confidence intervals are wide and the quantitative values are off by roughly two orders of magnitude. The paper treats this partial match cautiously: "since this similarity is not found for both models, we conclude that, in general, the noise scale cannot be used reliably as a proxy for the CBS" (Section 3.3).

The confidence intervals reported in Figure 3 are instructive. Even at their upper bounds, the noise scale estimates for OLMo 7B never exceed approximately 20–30, still more than two orders of magnitude below the measured CBS of ~4096. This suggests the discrepancy is not an artifact of estimator variance but a genuine structural mismatch between what the noise scale measures and what the CBS represents.

#### Batch Size Warmup: Training Loss

Table 1 reports the core quantitative comparison between the three training configurations. The results are split into pretraining loss (after 608B tokens) and mid-training loss (after an additional 50B tokens of learning rate annealing), both averaged over the final 10B tokens of each phase.

**Pretraining loss.** The batch size warmup configuration achieves a pretraining loss of **2.5891**, which is lower (better) than both the small-batch control (**2.6057**) and the large-batch control (**2.5962**). The improvement over the small-batch control is 0.0166 — a margin that may seem small but represents a meaningful difference in the context of language model pretraining, where loss differences of 0.01–0.02 are often considered practically significant. The large-batch control, despite using a batch size (4096) that the CBS measurements suggest is safe for the later portion of training, underperforms the small-batch control by approximately 0.0095 — this degradation is attributed to the early phase of training where the batch size exceeded the CBS.

**Mid-training (post-annealing) loss.** After annealing the learning rate to 0 over 50B tokens, the ordering changes slightly. The batch size warmup configuration achieves a mid-training loss of **2.5433**, still outperforming both controls. The small-batch control achieves **2.5486**, and the large-batch control achieves **2.5506**. The margin between batch size warmup and small-batch control narrows to 0.0053 after annealing — this is expected since annealing tends to reduce differences between training configurations (OLMo et al., 2025). Crucially, batch size warmup maintains its advantage rather than the small-batch control catching up, indicating that the improvement from dynamic batch sizing is genuine and not an artifact of the pretraining phase.

**Gradient steps saved.** The key practical metric: batch size warmup uses **43% fewer gradient steps** than the small-batch control while achieving slightly better loss. This is computed by comparing the total number of gradient steps across the full training run (608B tokens of pretraining + 50B tokens of mid-training), accounting for the different batch sizes in effect at different stages. The large-batch control uses 75% fewer steps but degrades final loss — this quantifies the cost of exceeding the CBS.

**Interpreting the loss improvement.** The fact that batch size warmup achieves *better* loss than the small-batch control, not just equivalent loss, is consistent with the CBS framework. The small-batch control trains at B = 1024 for the entire run, which is below the CBS for most of training (once the CBS has grown past 1024 in early training). Training below the CBS is "safe" in that it doesn't degrade loss relative to even smaller batch sizes, but it means the model is making less progress per gradient step than it could if it used a larger batch size. Batch size warmup corrects this inefficiency by scaling up the batch size as the CBS grows, effectively spending the later portion of training at the CBS (or close to it) rather than below it. The result is a model that makes slightly better use of the same token budget.

#### Batch Size Warmup: Downstream Performance

Table 2 reports three categories of downstream evaluation: bits-per-byte (BPB) on a suite of QA tasks, and cross-entropy loss on C4 and The Pile, each evaluated both after pretraining and after mid-training.

**Downstream task BPB.** Batch size warmup achieves **1.0316 BPB** after pretraining and **1.0076 BPB** after mid-training. Compared to the small-batch control (1.0112 after pretraining, 0.9999 after mid-training), batch size warmup is slightly worse after pretraining (by 0.0204) but slightly better after mid-training (by 0.0077). The large-batch control performs worst in all conditions (1.0571 after pretraining, 1.01927 after mid-training). The pattern here is mixed — no single configuration is uniformly best across both evaluation points — but batch size warmup is competitive with or better than the small-batch control in three out of four conditions (pretraining and mid-training for two of the three metric categories in Table 2: C4 and Pile loss).

**C4 validation loss.** Batch size warmup achieves **2.8049** after pretraining and **2.7597** after mid-training, versus the small-batch control at 2.8196 and 2.7622 respectively. Batch size warmup is better in both conditions, with margins of 0.0147 (pretraining) and 0.0025 (mid-training).

**Pile validation loss.** Batch size warmup achieves **2.1916** after pretraining and **2.1521** after mid-training, versus the small-batch control at 2.2073 and 2.1471. Batch size warmup is better after pretraining (by 0.0157) but slightly worse after mid-training (by 0.0050).

**Overall assessment.** The downstream evaluation shows that batch size warmup does not systematically degrade out-of-distribution performance relative to small-batch training, and in several conditions it provides modest improvements. The paper appropriately tempers the interpretation: "With the caveat that precisely measuring the downstream impact of pretraining decisions is difficult, this suggests that, just as batch-size warmup does not degrade final loss, it should not degrade downstream measures of performance either" (Section 4.3). The mixed pattern — sometimes better, sometimes slightly worse, never dramatically different — is consistent with the training loss results, where the differences between configurations are small in absolute terms.

### Ablation Studies and Robustness Checks

This paper does not contain traditional ablation studies in the style of removing or varying components of a proposed method. The CBS measurement method has tuneable parameters (∆, ε, α), but the paper does not systematically sweep them to assess sensitivity. Instead, the "ablations" are implicit in the experimental design choices and the comparisons across configurations:

**Model size invariance of CBS (Figure 2):** The comparison of OLMo 1B and 7B CBS curves serves as an implicit robustness check on model size dependence. The finding that both models follow the same qualitative trajectory (start near 0, grow rapidly, plateau around 4096) suggests that the CBS is not sensitive to a 7× increase in parameter count. This is consistent with prior work (Zhang et al., 2024) but demonstrated here using a different methodology (branched training vs. full-run training), providing converging evidence from independent experimental paradigms.

**Oracle CBS thresholds vs. manual reading (Section 4.2 footnote):** The paper discloses that the doubling thresholds used in the batch size warmup experiment (168B and 503B tokens) were based on "preliminary CBS measurements at an earlier stage of the project" and that the measurements "changed slightly after this point." The authors deemed the changes insufficient to warrant restarting the expensive training runs, and the success of the warmup run (achieving better loss than controls) suggests the method is not highly sensitive to the precise threshold values — as long as doubling occurs *after* the CBS has genuinely grown past the new batch size, the exact timing may not be critical. However, this is a pragmatic observation rather than a controlled ablation.

**Square-root vs. linear learning rate scaling:** The paper uses square-root scaling throughout but does not run an ablation comparing it to linear scaling. The theoretical justification is drawn from Malladi et al. (2022) and Li et al. (2024), but the sensitivity of the CBS measurements to this choice is unexplored. If linear scaling were used instead, the branched training runs at larger batch sizes would use proportionally larger learning rates, potentially causing the CBS to appear smaller (since larger learning rates could destabilize training and inflate loss, even if the batch size itself is safe). This is a notable gap — the square-root rule is well-motivated, but the empirical consequences of getting it wrong are not quantified.

**Tolerance ε and window ∆ sensitivity:** The paper explicitly acknowledges these as limitations and future work directions (Section 5): "it would be interesting to carry out a more systematic analysis of the impact of the hyperparameter ∆ on CBS measurements: how sensitive is the method to ∆ and do different values for ∆ potentially bias the measurement?" This is an honest admission that the method's sensitivity to its key parameters has not been characterized.

**Checkpoint selection density:** The CBS curves in Figure 2 use different checkpoint densities at different stages of training (e.g., every 10K steps early, every 50K steps later for 1B). This reflects a deliberate choice to sample more densely during the rapid growth phase, but the paper does not discuss whether the conclusions would change with different checkpoint spacing.

### Critical Assessment

The paper's central claims, as articulated in the abstract and Section 1, are:

**(1) The gradient noise scale is an unreliable proxy for the CBS in language model pretraining.**

This claim is **strongly supported** by the direct comparison in Figures 2 and 3. The noise scale underestimates the measured CBS by roughly two orders of magnitude, and for OLMo 7B, the qualitative trend runs in the opposite direction (noise scale decreases while CBS increases). The confidence intervals on the noise scale (Figure 3) do not overlap with the measured CBS values at any point. The paper provides a clear theoretical explanation for why this mismatch occurs (Assumptions 1 and 2 are violated), and the empirical evidence is unambiguous.

However, the claim's scope should be noted: the noise scale was tested only for the OLMo model family with the Adam optimizer on this specific data distribution. The paper is careful to state that "in general, the noise scale cannot be used reliably as a proxy" — the evidence supports this for the tested regime, but the possibility that noise scale works under different conditions (SGD, smaller models, different data) is not ruled out. The failure is demonstrated, but the boundary conditions of the failure (when *does* noise scale become reliable?) are not characterized.

**(2) The CBS starts near 0, increases rapidly, and then plateaus, with this trend largely independent of model size.**

This claim is **supported** by Figure 2, which shows consistent qualitative behavior for both model sizes across the full training trajectory. The near-0 CBS at initialization and the plateau at ~4096 documents are consistent across 1B and 7B models. The model-size independence is particularly well-supported because the two models differ by a factor of 7 in parameters but show nearly identical CBS curves.

However, several caveats apply. First, the CBS measurement has inherent uncertainty (represented by the interval endpoints), and the plateau value of ~4096 is partly an artifact of the tested batch size multipliers — the method can only identify the CBS up to the resolution of the tested k values, and the plateau at 4096 might reflect that 4096 was the largest tested batch size that didn't degrade loss at many checkpoints. The true CBS could be higher than 4096 but not detectable because larger batch sizes were not tested, or the true CBS could be declining slightly at later checkpoints but the coarse multiplier spacing masks this. The loss-vs-batch-size curves in Appendix A (Figures 5 and 6) help address this concern — they show that for later checkpoints, loss begins to increase noticeably at batch sizes above 4096, suggesting the plateau is genuine.

Second, the model-size independence is demonstrated only for two models from the same family (OLMo) with the same architecture and training data. The paper frames this as "in line with past findings" from Zhang et al. (2024) and Bergsma et al. (2025), which used different methodologies and different model families. The converging evidence from independent work strengthens the claim, but the paper's own experiments are limited to OLMo.

Third, the CBS at initialization is measured as "near 0" but the exact value is coarse because only a few multipliers (0.0625, 0.125, 0.25) were tested at step 0. The claim that CBS "starts near 0" is qualitative and supported, but the exact initial CBS value is not precisely determined.

**(3) Batch size warmup, guided by empirical CBS measurements, enables training with 43% fewer gradient steps without degrading final loss (and slightly improving it).**

This claim is the paper's most important practical contribution, and the evidence is **mixed in strength**.

**What the experiments demonstrate:** For OLMo 1B trained on 608B tokens, a batch size schedule that doubles from 1024 to 2048 at 168B tokens and to 4096 at 503B tokens achieves slightly better loss (0.0053 after annealing) than a fixed batch size of 1024, while using 43% fewer gradient steps. The large-batch control (fixed 4096) degrades loss, confirming that the warmup schedule is essential — simply training at 4096 throughout is harmful. Downstream metrics (Table 2) are competitive, with no systematic degradation.

**What the experiments do NOT demonstrate:** Several important dimensions are not tested.

- **Statistical reliability.** Each training configuration is run once. Without multiple random seeds or error bars, it is impossible to determine whether the 0.0053 loss improvement is statistically significant or within the noise of training run variance. The paper reports loss averaged over the last 10B tokens, which reduces within-run noise, but does not account for between-run variance from different data orderings or initialization. In large-scale pretraining, differences of 0.0053 in final loss may or may not be meaningful — the paper provides no context (e.g., what is the typical run-to-run standard deviation for OLMo 1B at this scale?) for interpreting this magnitude.

- **Scaling to full training duration.** The experiments train for 608B tokens of a planned 4T token run (15% of the full schedule), followed by annealing. The paper argues that annealing makes the mid-training loss "a proxy for how these runs would compare if we were to fully train them for the full learning rate schedule," citing OLMo et al. (2025). This is a reasonable argument but not a substitute for demonstrating that batch size warmup works for full-length training. It is possible that the advantage of batch size warmup diminishes or reverses over longer training — perhaps the small-batch control would eventually catch up, or the larger batch sizes used in warmup would encounter optimization difficulties not visible in the shorter run.

- **Only one doubling schedule is tested.** The paper tests exactly one batch size warmup schedule (two doublings at two specific token counts). The claim is that "batch size warmup" as a strategy works, but the evidence demonstrates only that *this specific schedule* works. An ablation testing different numbers of doublings (e.g., one doubling vs. three doublings) or different threshold placements would strengthen the claim that the CBS-measurement-driven approach is robust to the specific schedule chosen. The footnote acknowledging that preliminary CBS measurements differed slightly is reassuring but not systematic.

- **Only OLMo 1B is tested.** The batch size warmup experiment is run only for the 1B model. The 7B CBS curve is measured, but no 7B warmup training run is conducted. The claim that the method "can be applied to reliably train language models at larger batch sizes" is demonstrated only at the 1B scale. The paper's finding that CBS is model-size-independent suggests transferability, but the actual training run at 7B is missing.

- **No comparison to alternative dynamic batch size methods.** The paper compares against fixed batch sizes (small and large), but not against other dynamic batch size strategies, such as the noise-scale-based approach from McCandlish et al. (2018) or the learning-rate-decay-replacement approach from Smith et al. (2018). A comparison against a noise-scale-guided warmup would directly test whether the paper's CBS measurement method produces better schedules than the proxy-based method it critiques.

- **The difficulty estimation cost is not amortized.** The batch size warmup experiment uses CBS thresholds derived from the full CBS measurement curve (Figure 2), which required approximately 150–200B tokens of branched training across 13 checkpoints. This measurement cost is not included in the comparison — the small-batch control didn't incur this cost, while the batch size warmup configuration benefited from it. In a real deployment, the CBS measurement cost would need to be amortized across multiple training runs or scaled down (e.g., by measuring only a few checkpoints or using smaller ∆). The paper does not address this amortization question.

**(4) CBS measurements from small training runs can inform larger-scale training runs.**

This claim is **supported conditionally** — conditional on the model family and data distribution being the same. The finding that OLMo 1B and 7B CBS curves are similar (Figure 2) provides evidence at a 7× scale factor, but this is a single data point. The paper does not test whether CBS measured on a 1B model accurately predicts CBS for a 70B model, or whether CBS measured on one dataset (Dolma) transfers to another dataset. The claim is further supported by consistency with Zhang et al. (2024), who found similar model-size independence using different methodology, but the paper's own evidence is modest.

**Missing experiments that would strengthen the paper:**

1. **Multiple seeds for the batch size warmup validation run**, to establish whether the 0.0053 loss improvement is statistically reliable.

2. **Ablation on ∆**, the branched training window size. Does the measured CBS depend on whether ∆ = 1B, 2B, or 4B tokens? If the CBS estimate is sensitive to ∆, the method's reliability is limited.

3. **Ablation on ε**, the loss tolerance. How does the CBS curve change if ε = 0.005 vs. 0.02? A more principled tolerance (e.g., based on noise level in the smoothed loss) would be preferable to the arbitrary 0.01.

4. **Comparison of square-root vs. linear learning rate scaling** in the branched training procedure, to quantify how much the CBS estimate depends on this choice.

5. **Batch size warmup at 7B scale**, to validate that the method transfers from measurement to training at larger model sizes.

6. **Comparison against noise-scale-guided warmup**, to test whether the paper's direct measurement approach produces empirically better training schedules than the proxy-based approach it critiques.

7. **Full-length training to 4T tokens**, to validate that the benefits of batch size warmup persist (or grow) over the complete training budget rather than being an artifact of the truncated 608B token run.

8. **Online CBS estimation**, where CBS is measured during the warmup run itself (not from a separate measurement run), to address the amortization concern and demonstrate that the method can be applied without a separate measurement phase.

**Overall assessment.** The paper's primary contributions are the branched training methodology and the empirical finding that CBS evolves systematically during training with a consistent pattern. These are well-supported by the experiments. The batch size warmup validation is a promising demonstration but is not a fully rigorous proof that the method reliably produces optimal batch size schedules — more extensive testing (multiple seeds, larger scale, full training duration) would be needed to make that claim with high confidence. The paper's honesty about limitations (discussed in Section 5 and throughout) is a strength, and the contributions should be evaluated as a methodological framework with promising initial validation rather than as a turnkey solution for all large-scale training scenarios.

## 6. Limitations and Trade-offs

### The CBS Measurement Cost Is Not Amortized and Rivals the Training Budget It Is Meant to Optimize

The branched training method for measuring the CBS requires launching multiple training branches at each checkpoint, each consuming a window of ∆ = 2B tokens. The paper acknowledges this cost but does not account for it in the headline efficiency gains. The consequence is that the reported 43% gradient step reduction for batch size warmup is computed *after* the CBS curve has already been measured, without including the cost of that measurement. A practitioner considering whether to adopt this method needs to know the total compute required, not just the savings after measurement.

**Scale of the unaccounted cost.** For the OLMo 1B CBS curve in Figure 2, the paper tested approximately 13 checkpoints, each with roughly 5–8 batch size multipliers (the exact counts vary by checkpoint; see Appendix A), and each branch trained for 2B tokens. This conservatively totals 13 × 6 × 2B ≈ 156B tokens of branched training — approximately 26% of the 608B token pretraining budget used in the validation experiment. For the OLMo 7B curve, covering 14 checkpoints with similar branch counts, the total is proportionally larger given the 7× increase in model size. The paper documents (Appendix A) that "1B runs were launched on a single node of H100 GPUs, and 7B runs were launched on 8 nodes," but does not convert this to a total FLOP or dollar cost. In practical terms, measuring the CBS curve for a 7B model costs roughly the equivalent of training it for hundreds of billions of tokens — a non-trivial fraction of a full pretraining budget.

**What the paper says.** Section 3.2 states that the branched training method measures the CBS "with only a small amount of additional training (controlled by ∆)," describing ∆ = 2B tokens as "a small, conservative window size relative to our overall pretraining budget of 600B tokens." This framing treats ∆ per branch as small, which it is — but it omits that ∆ must be multiplied by the number of branches per checkpoint times the number of checkpoints. The paper's description of the branched training cost does not mention this multiplicative factor.

**Amortization across runs.** If the CBS curve is measured once and then reused for many subsequent training runs (e.g., for the same model architecture on the same data distribution), the measurement cost could be amortized. The paper alludes to this scenario in Section 3.1: "CBS measurements with small models could be used to inform large-scale training runs," supported by the finding that CBS curves are similar for 1B and 7B models (Figure 2). However, this amortization argument assumes (a) the CBS curve transfers across model sizes, which is demonstrated only up to 7× scaling for one model family, and (b) the measurement is done proactively as a research investment, not as a per-run cost. For a practitioner training a single large model, the measurement cost is a direct overhead added to the training budget. The paper does not discuss this distinction or provide guidance on when amortization is feasible.

**Mitigation status.** The paper partially acknowledges the cost concern. Section 1 lists as a research question "How can we measure the CBS cheaply with minimal assumptions *before* launching a pretraining run?" (emphasis added), and Section 5 suggests "estimating the CBS in an online fashion" as future work — that is, measuring the CBS during the target training run itself rather than in a separate measurement phase. Section 5 also suggests training a model to "directly predict difficulty of a question" (in the context of difficulty estimation), which could be analogized to predicting CBS from checkpoint statistics without branched training. Neither approach is developed in the paper. Currently, the method requires a separate measurement phase whose cost is not folded into the efficiency claims.

### Only a Single Model Family and Dataset Are Tested, with No Evidence of Transfer

All CBS measurements and the batch size warmup validation are conducted exclusively on OLMo models (1B and 7B) trained on the Dolma dataset. The paper makes no measurements on any other model architecture, any other dataset, or any optimizer other than Adam. The consequence is that the paper's central finding — that CBS starts near 0, grows rapidly, and plateaus — is demonstrated for exactly one point in the space of possible training configurations, and the batch size warmup recipe is validated for exactly one model at one scale.

**Why this matters.** The CBS is a property of the interaction between model architecture, data distribution, optimizer, and training stage. There is no theoretical guarantee that the observed CBS evolution pattern (near-zero at initialization, rapid growth, plateau) generalizes beyond OLMo. Transformer architectures with different depths, widths, normalization schemes, or activation functions could exhibit different CBS dynamics. Datasets with different token distributions (code-heavy vs. web-text-heavy, multilingual vs. English-only) could produce different gradient statistics at each training stage, shifting the CBS curve. Even within the same model family, the plateau value of ~4096 documents may not hold at larger scales — the paper's evidence for model-size independence covers only a 7× range, and the finding that the CBS "does not seem to depend on model size" is described as being "in line with past findings" (Zhang et al., 2024) rather than proven by the paper's own experiments across a wide range of scales.

**What the paper says.** Section 1 states that the authors "focus our investigation on the OLMo models... due to their open weights and data." This is a pragmatic choice, not a claim of generality. Section 5 acknowledges the limitation implicitly by listing future work directions that would extend the method to other settings, but does not explicitly state that the findings may not transfer.

**Existing cross-validation.** The finding of model-size independence (1B vs. 7B) within OLMo provides some evidence that the CBS pattern is not an artifact of a specific model scale, but this is a within-family comparison and does not address transfer to different architectures, data distributions, or optimizer configurations. The paper cites Zhang et al. (2024) and Bergsma et al. (2025) for converging evidence, but those works used entirely different methodologies (full training runs from initialization) and different model families — the convergence is conceptual (both find that CBS scales with data size, both find model-size independence) but does not validate the specific CBS evolution pattern measured by branched training.

**Mitigation status.** Not addressed. The paper does not claim generality beyond OLMo, but it also does not explicitly bound the scope of its conclusions. A practitioner using a non-OLMo architecture or dataset would need to re-measure the CBS curve from scratch, incurring the measurement cost discussed above, without knowing whether the OLMo pattern is a useful prior.

### The Batch Size Warmup Validation Lacks Statistical Rigor and Tests Only a Single Schedule

The batch size warmup experiment (Section 4.2–4.3) trains exactly one model with the proposed schedule, one small-batch control, and one large-batch control. There are no repeated runs with different random seeds, no error bars on the reported losses, and only one specific doubling schedule is tested. The consequence is that the headline result — batch size warmup achieves "slightly better loss than the original training run with 43% fewer gradient steps" — cannot be distinguished from run-to-run variance, and the claim that the method "can be applied to reliably train language models at larger batch sizes" (abstract) is supported by a single positive example rather than systematic evidence.

**The magnitude of the claimed improvement relative to plausible noise.** The mid-training loss improvement of batch size warmup over the small-batch control is 0.0053 (2.5433 vs. 2.5486; Table 1). The paper provides no estimate of run-to-run standard deviation for OLMo 1B trained on 608B tokens. In large-scale pretraining, loss differences of this magnitude can arise from data ordering, initialization noise, or hardware non-determinism alone. Without error bars, a practitioner cannot assess whether the observed improvement is a genuine effect of batch size warmup or a fluctuation that would disappear or reverse in a replicate run.

**What the paper does report.** The loss values are "averaged over the past 10B tokens" (Table 1), which reduces within-run noise from batch-level loss fluctuations. But this averaging only addresses noise *within* a single run — it does not address between-run variance. The pretraining loss differences are larger (0.0166 for warmup vs. small-batch control; Table 1), but the paper explicitly frames the mid-training (post-annealing) loss as the primary metric because annealing "can induce significant gains in loss for partial pretraining runs" (Section 4.2) and "more closely reflects how these checkpoints are used in language model training." The mid-training differences are the smaller values, making the statistical uncertainty more consequential.

**Single schedule limitation.** The paper tests exactly one batch size warmup schedule: two doublings at 168B and 503B tokens. These thresholds were chosen "based on a manual reading of the CBS measurements from Section 3" (Section 4.2) — and, as a footnote reveals, were actually based on *preliminary* CBS measurements that "changed slightly after this point." The success of this specific schedule is consistent with the method working, but a single positive example does not demonstrate robustness. A practitioner following the paper's methodology on their own model would need to determine doubling thresholds from their own CBS measurements, and the paper provides no systematic procedure for this — only a manual reading, which introduces human judgment as a source of variance.

**Downstream evaluation is similarly single-run.** Table 2 reports BPB and validation losses from the same single training runs. The mixed pattern (warmup better in some conditions, small-batch better in others) could reflect genuine differences or statistical noise — without error bars, the two are indistinguishable.

**What the paper says.** The paper does not directly acknowledge the lack of statistical rigor. Sections 3 and 4 describe the experimental designs without mentioning that each configuration is run only once. Section 5 lists future work on "systematic methodology for choosing batch size warmup thresholds," which partially addresses the single-schedule limitation, but does not mention the need for multiple random seeds or statistical testing.

**Mitigation status.** Not addressed. Retraining large language models with multiple seeds is expensive, and the paper's choice to run each configuration once is understandable given compute constraints. But the claims should be appropriately tempered — the evidence supports the conclusion that batch size warmup *does not obviously degrade* loss while reducing gradient steps, not that it *reliably improves* loss. The distinction matters for a practitioner deciding whether to adopt the method.

### The Training Run Is Truncated to 15% of the Full Schedule, with Mid-Training Used as a Proxy for Full Convergence

The batch size warmup validation trains for 608B tokens of a planned 4T token run (OLMo et al., 2025), then applies 50B tokens of learning rate annealing (mid-training). The paper states that "we take the loss after mid-training to represent the loss that would be achieved by these training runs in a practical context" and that OLMo et al. (2025) "suggest that this kind of learning rate annealing can induce significant gains in loss for partial pretraining runs" (Section 4.2). The consequence is that the paper's central practical claim — that batch size warmup enables training with 43% fewer gradient steps without degrading final loss — is demonstrated for a truncated run and extrapolated to the full training duration via a proxy (mid-training loss) whose validity as a predictor of full-training performance has not been systematically established for the batch size warmup setting.

**Why truncation matters.** The CBS plateaus around 4096 documents by roughly 200–400B tokens (Figure 2), meaning that for the later ~3.4T tokens of a full training run, the batch size would remain at its maximum value (4096) under the warmup schedule. The small-batch control, which trains at B = 1024 throughout, would spend those same 3.4T tokens taking 4× more gradient steps at a batch size below the CBS. The paper's experiment captures only the first ~15% of this regime — the portion from 503B to 608B tokens where warmup is at B = 4096 and the small-batch control is at B = 1024. The majority of the potential advantage (or disadvantage) of larger-batch training lies in the unobserved 608B–4T token range.

**What could change in the unobserved regime.** Several dynamics could alter the relative performance of warmup vs. small-batch training over a full run:
- **Diminishing returns to larger batches.** If the CBS advantage (the benefit of training closer to the CBS vs. below it) diminishes as loss decreases, the warmup advantage observed in the truncated run might not grow proportionally with additional training — the small-batch control could asymptotically catch up.
- **Optimization difficulties at scale.** Training at B = 4096 for ~3.4T tokens might encounter stability issues (loss spikes, gradient divergence) not visible in the ~100B token window where warmup operates at that batch size in the truncated experiment.
- **Mid-training as an imperfect proxy.** OLMo et al. (2025) showed that annealing improves loss for partial runs, but they did not (to the paper's knowledge) establish that the *relative ordering* of different training configurations after partial-run annealing is preserved after full-run training. A configuration that looks better after 608B + annealing could look worse after 4T + annealing if the training dynamics differ in the later stages.

**What the paper says.** Section 4.2 is explicit about the truncation: "The original training run for OLMo 1B ran for 4T tokens, so we do not have the resources to replicate this full training run. Instead, we pre-train for 608B tokens using the original learning rate schedule for the longer run." The paper frames the mid-training loss as a proxy "for how these runs would compare if we were to fully train them for the full learning rate schedule" (Section 4.2). This is a reasonable argument given compute constraints, and it is standard practice in the scaling laws literature to extrapolate from shorter runs. But it is an extrapolation, not a direct demonstration, and the paper's claims should be read accordingly.

**Mitigation status.** Partially addressed. The paper acknowledges the truncation and provides a rationale for using mid-training loss as a proxy. Section 5 lists future work on "more systematic analysis" of the method, but does not specifically call for full-length training validation. The batch size warmup result, while encouraging, should be treated as evidence that the method works at the ~600B token scale, with the extrapolation to full training duration being a hypothesis rather than an established fact.

### The Square-Root Learning Rate Scaling Rule for Adam Is Justified by Theory but Not Empirically Validated in This Work

The branched training CBS measurement procedure and the batch size warmup algorithm both rely on the square-root learning rate scaling rule: when the batch size is multiplied by k, the base learning rate is multiplied by √k for Adam. The paper justifies this choice by citing theoretical work from Malladi et al. (2022) and Li et al. (2024), but does not run any experiments comparing square-root scaling to linear scaling or to any other scaling rule. The consequence is that the CBS measurements — and by extension the batch size warmup thresholds derived from them — are conditional on the correctness of the square-root rule for the OLMo training configuration.

**Why this matters for CBS measurement.** The branched training procedure compares loss across branches with different batch sizes. If the learning rate scaling rule is wrong, the larger-batch branches are not being given a fair chance to match the smaller-batch branches — they might underperform not because the batch size exceeds the CBS, but because their learning rate is suboptimal. This would bias the CBS estimate downward (the method would conclude a batch size is too large when the real problem is the learning rate). The direction of bias depends on whether the true optimal scaling is larger or smaller than √k — if linear scaling (k) is actually correct for Adam in practice, the paper's use of √k would systematically under-scale the learning rate at larger batches, making those branches underperform and producing CBS estimates that are too low.

**What the theory says (and its limits).** Malladi et al. (2022) analyzed Adam through the lens of stochastic differential equations and argued that square-root scaling is appropriate for adaptive optimizers because Adam's preconditioner provides partial invariance to gradient scale. Li et al. (2024, Equation 4) provided additional theoretical support. However, both analyses make simplifying assumptions (continuous-time limit, specific noise models) that may not hold in the discrete, finite-step regime of actual language model pretraining. The empirical validation in those papers focused on smaller-scale experiments (image classification, smaller language models), not on billion-parameter transformers trained on web-scale data. Whether square-root scaling holds exactly, approximately, or not at all for OLMo-scale training is an open empirical question that the paper does not address.

**What the paper says.** Section 3.1 defines the learning rate scaling rule as f(k) = √k for Adam, citing Malladi et al. (2022). Section 3.1 notes that "in principle a linear scaling rule could also be used," but does not explore this. Section 2 critiques McCandlish et al. (2018) for assuming linear scaling, arguing that "when training with Adam, it seems that the linear scaling rule assumed by McCandlish et al. (2018) should not apply." The paper thus positions square-root scaling as a correction to prior work, but provides no empirical evidence within its own experiments that this correction is quantitatively correct.

**Mitigation status.** Not addressed. The paper does not run a scaling rule ablation, does not measure the sensitivity of CBS estimates to the choice of scaling rule, and does not validate that √k is optimal for the specific training configuration used. The batch size warmup experiment's success provides indirect evidence that √k is not grossly wrong (since the warmup run works), but it does not distinguish between √k being optimal and √k being "close enough" that the CBS measurement and warmup procedure are robust to moderate misspecification. A practitioner using a different optimizer (e.g., AdamW with different β parameters, or a different adaptive optimizer like Lion or Sophia) would have no empirical guidance from this paper on what scaling rule to use.

### The Batch Size Warmup Thresholds Are Selected Manually, with No Automated or Principled Procedure

The batch size warmup experiment uses two doubling thresholds — double to 2048 at 168B tokens, double to 4096 at 503B tokens — that were determined by manually inspecting the CBS curve in Figure 2. The paper does not provide an algorithm, a decision rule, or a statistical procedure for translating a measured CBS curve into a batch size warmup schedule. The consequence is that the method, as presented, is not fully reproducible without human judgment, and the thresholds used in the paper's own experiment were based on preliminary CBS measurements that the authors acknowledge changed slightly after the training runs were launched.

**The operational gap.** The paper's batch size warmup algorithm (Section 4.1) states: "After training for t tokens, if we determine that the CBS exceeds the current batch size (B*_t > 2B_t), we double the current batch size." This requires implementing the condition "if we determine," which the paper does by manually reading the CBS curve. A practitioner following the paper would need to: (1) measure the CBS curve using branched training, (2) decide at which token counts the CBS has grown enough to support a doubling, (3) choose how many doublings to perform, and (4) decide how to handle the fact that the CBS curve has measurement uncertainty (the interval between lower and upper bounds shown in Figure 2) rather than exact values. None of these steps are algorithmic.

**The preliminary measurement issue.** The footnote in Section 4.2 states: "These thresholds were determined from preliminary CBS measurements at an earlier stage of the project. Our measurements changed slightly after this point, but we deemed that it was not worth it to restart the expensive training runs because we do not think the results should be sensitive to a precise choice of threshold." This is transparent but reveals a tension: the thresholds used in the paper's headline experiment were not derived from the final CBS measurements shown in Figure 2, and the authors' judgment that the run didn't need restarting is itself a manual intervention. A practitioner seeking to replicate the method would face the same judgment call — which version of the CBS measurements to trust, and how much deviation from the thresholds is acceptable — without a systematic framework for making it.

**Connection to the cost limitation.** The manual threshold selection is partly a consequence of the CBS measurement cost: the CBS curve is measured at discrete checkpoints with finite resolution (the multiplicative spacing of batch size multipliers k), and the exact point at which CBS exceeds 2B_t may fall between two measured checkpoints. An automated procedure would need to interpolate the CBS curve and make decisions under uncertainty, which the paper does not develop.

**What the paper says.** Section 4.1 acknowledges the manual nature: "In practice, we do this heuristically based on the measurements in Figure 2: since we only double the batch size twice over training, this involves just picking two thresholds." Section 5 lists as future work: "Going forward, it would be useful to establish a systematic way to set threshold for increasing the batch size given CBS measurements." The paper is honest about this being a manual step, but the headline claim that "our framework can be applied to reliably train language models at larger batch sizes" (Section 5) overstates the current level of automation.

**Mitigation status.** Not addressed in the current paper. The suggestion of "automated" and "online" CBS estimation in Section 5 points toward future work, but provides no concrete method. The paper's contribution is the measurement framework and the proof-of-concept that batch size warmup works; the operationalization of converting measurements into schedules remains a manual, judgment-dependent step.

## 7. Implications and Future Directions
- How this changes practice
  - Provides a practical, optimizer-aware way to choose and adapt batch size during LLM pretraining without relying on questionable proxies. Practitioners can:
    - Run short branched probes early and mid-training to map `B*` over time.
    - Use batch size warmup that never exceeds the measured CBS, preserving token efficiency while scaling data parallelism.

- Theoretical and empirical follow-ups
  - Formalize and test the local recovery assumption, including longer `Δ` windows, different optimizers (e.g., AdamW variants), and schedules.
  - Automate online CBS estimation and triggering (e.g., adaptive `Δ`, statistical hypothesis tests instead of a fixed `ε`, and curve-fitting of CBS vs. tokens).
  - Extend Appendix D’s connection between local CBS curves and global CBS scaling laws; explore asymmetric penalties for training above vs. below CBS.

- Broader applications
  - Training infrastructure: Cluster schedulers could automatically ramp batch size as CBS grows, minimizing idle compute and communication overhead.
  - Hyperparameter transfer: Small-scale pilot runs can estimate CBS growth to plan large-scale runs (Section 3.3 shows similar trends across 1B and 7B).
  - Methodology beyond LMs: The branched-recovery idea can be adapted to other domains where adaptive optimizers are standard (vision transformers, diffusion models), provided loss-vs-tokens (or steps) comparability is meaningful.

- Open questions
  - Can we predict CBS from model and data properties without any branching?
  - How does CBS interact with other axes: sequence length scaling, curriculum learning, data-mixing strategies, or optimizer hyperparameters like β1/β2 and weight decay?
  - What is the safest learning-rate scaling when also changing other training knobs (e.g., gradient clipping thresholds, normalization layers)?

Overall, this work supplies a concrete, low-assumption toolkit for large-batch LLM training: measure CBS locally with short, branched probes; then warm up the batch size only when it is safe. The empirical results (Figures 1–4; Tables 1–2) show this approach can preserve or slightly improve loss while materially reducing the number of gradient steps.
