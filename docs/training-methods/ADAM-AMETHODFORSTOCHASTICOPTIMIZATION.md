# ADAM: A METHOD FOR STOCHASTIC OPTIMIZATION

**ArXiv:** [1412.6980](https://arxiv.org/abs/1412.6980)

## 🎯 Pitch

Adam introduces a novel optimization algorithm that adapts the learning rate for each parameter by maintaining moving averages of both the gradients and their squared values—automatically adjusting to the noise and sparsity common in practical machine learning. By unifying the benefits of AdaGrad (good with sparse data) and RMSProp (stable in non-stationary settings), while adding crucial bias correction and strong theoretical convergence guarantees, Adam has become a robust, easy-to-use default for training deep neural networks and other large-scale models with stochastic objectives.

---

## 1. Executive Summary
Adam is an optimization algorithm for training models with stochastic (noisy) gradients that adapts the learning rate separately for every parameter by maintaining moving averages of first and second moments of the gradients. It matters because it combines the strengths of two influential methods (AdaGrad’s handling of sparsity and RMSProp’s stability on non‑stationary problems) while adding crucial bias‑correction and a convergence guarantee, making it a robust default optimizer for large, noisy, high‑dimensional problems.

## 2. Context and Motivation
- Problem addressed
  - Many learning problems optimize an objective that is observed through noisy estimates (e.g., mini‑batches, dropout). Choosing effective learning rates is hard, especially when gradients vary across parameters or over time.
  - Existing first‑order methods either require hand‑tuned schedules or handle only part of the problem (sparsity vs. non‑stationarity).

- Why important
  - Practical: determines whether large neural networks and other models train quickly and stably on GPUs/large datasets.
  - Theoretical: interest in provable guarantees for online convex optimization with adaptive methods.

- Prior approaches and gaps
  - SGD with momentum: strong baseline but sensitive to global learning‑rate choice; no per‑parameter adaptivity.
  - AdaGrad (Duchi et al., 2011): sums past squared gradients to set per‑parameter step sizes; excels with sparse gradients but its learning rate decays monotonically, which can be too aggressive.
  - RMSProp (Tieleman & Hinton, 2012): uses an exponential moving average (EMA) of squared gradients to adapt step sizes for non‑stationary problems but lacks bias‑correction; can be unstable when the EMA uses slow decay (Section 5 and Figure 4).
  - Quasi‑Newton/SFO: stronger curvature use but heavy memory/time costs and difficulty with stochastic regularization (Section 6.2).

- Positioning
  - Adam is designed to “combine the advantages” of AdaGrad and RMSProp (Section 1), add initialization bias‑correction (Sections 2–3), provide bounded effective step sizes with invariance to gradient rescaling (Section 2.1), and offer an online‑learning regret bound (Section 4). It also introduces a simple infinity‑norm variant, AdaMax (Section 7.1).

## 3. Technical Approach
Adam performs per‑parameter adaptive updates using two exponentially decayed moving averages; it then corrects their initialization bias and uses them to scale the step.

- Core update (Algorithm 1, Section 2)
  1. Initialize `m0 = 0`, `v0 = 0`, timestep `t = 0`. Hyperparameters: learning rate `α`, decay rates `β1` and `β2` in `[0,1)`, and a small `ε` for numerical stability. Defaults reported: `α = 0.001, β1 = 0.9, β2 = 0.999, ε = 1e-8`.
  2. At each step `t`:
     - Compute stochastic gradient `g_t = ∇θ f_t(θ_{t-1})`.
     - Update first‑moment EMA (an EMA is a weighted average that gives exponentially less weight to older values): `m_t = β1 m_{t-1} + (1 - β1) g_t`.
     - Update second raw moment EMA (uncentered variance): `v_t = β2 v_{t-1} + (1 - β2) g_t^2` (element‑wise square).
     - Bias‑correct both EMAs to remove small‑t underestimation caused by zero initialization (Section 3): `m̂_t = m_t / (1 - β1^t)`, `v̂_t = v_t / (1 - β2^t)`.
     - Parameter update: `θ_t = θ_{t-1} - α · m̂_t / (sqrt(v̂_t) + ε)`.

- Why bias‑correction?
  - With EMAs starting at zero, early estimates are biased low. Section 3 derives the expected value of `v_t`:
    - `E[v_t] = (1 - β2^t) E[g_t^2] + ζ` (Equations 1–4), where `ζ` is small if the second moment is slowly varying. Dividing by `(1 - β2^t)` removes the main bias term. An analogous correction applies to `m_t`.

- Effective step size properties (Section 2.1)
  - Ignoring `ε`, the effective step is `Δ_t = α · m̂_t / sqrt(v̂_t)`.
  - Bounded magnitude: 
    - If `(1 - β1) ≤ sqrt(1 - β2)`, then `|Δ_t| ≤ α` for each coordinate (upper bound); otherwise `|Δ_t| ≤ α · (1 - β1)/sqrt(1 - β2)`.
  - Invariance to gradient scaling: scaling all gradients by `c` scales `m̂_t` by `c` and `v̂_t` by `c^2`, which cancel in `m̂_t / sqrt(v̂_t)`.
  - “Signal‑to‑noise ratio” (their terminology, Section 2.1): the ratio `m̂_t / sqrt(v̂_t)` shrinks near optima (low gradient magnitude relative to variance), naturally annealing steps.

- Theoretical guarantee (Section 4)
  - Framework: online convex optimization with regret `R(T) = ∑_{t=1}^T [f_t(θ_t) - f_t(θ*)]`, where `θ*` is the best fixed point in hindsight.
  - Under assumptions of bounded gradients and bounded parameter diameter, with a decaying learning rate `α_t = α / sqrt(t)` and exponentially decaying momentum coefficient `β1,t = β1 λ^{t-1}`, Theorem 4.1 proves
    - `R(T) = O(√T)` and Corollary 4.2 gives average regret `R(T)/T = O(1/√T)`.
  - Proof elements rely on Lemma 10.3 (bounding sums of `|g_t|/√t`) and Lemma 10.4 (bounding terms involving bias‑corrected moments) in the Appendix.

- Relation to prior methods (Section 5)
  - If `β1 = 0` and `β2 → 1` with annealed `α_t = α / √t`, Adam reduces to AdaGrad with per‑coordinate step `α · g_t / √(∑_{i=1}^t g_i^2)`—but only when using bias‑correction; otherwise the limit is ill‑behaved.
  - RMSProp with momentum resembles Adam but lacks bias‑correction and applies momentum to rescaled gradients rather than maintaining a separate first‑moment EMA.

- AdaMax variant (Section 7.1; Algorithm 2)
  - Generalizes Adam’s `L2`‑based scaling to the `Lp` norm and takes `p → ∞`. The limiting update uses the exponentially weighted infinity norm:
    - `u_t = max(β2 · u_{t-1}, |g_t|)` and the update `θ_t = θ_{t-1} - [α/(1 - β1^t)] · m_t / u_t`.
  - No bias‑correction is needed for `u_t`. The parameter step has a simple bound: `|Δ_t| ≤ α`.

- Temporal averaging (Section 7.2)
  - Optional EMA over parameters: `\bar{θ}_t = β2 · \bar{θ}_{t-1} + (1 - β2) θ_t`, with bias‑corrected `\hat{θ}_t = \bar{θ}_t / (1 - β2^t)`; a standard technique to reduce the variance of the final iterate.

## 4. Key Insights and Innovations
- Bias‑corrected adaptive moments for stability and correctness (Sections 2–3)
  - Novelty: explicitly corrects the initialization bias of both first and second moment EMAs via division by `(1 - β^t)`. RMSProp and many momentum variants omit this step.
  - Significance: enables using very slow decay (`β2 ≈ 0.999`) needed for sparse gradients without exploding steps. Figure 4 shows that with `β2` close to 1, removing bias‑correction leads to instabilities early in training, while bias‑correction stabilizes and improves final loss.

- Bounded, scale‑invariant effective step sizes (Section 2.1)
  - Novelty: a clean analytical bound on each coordinate’s step magnitude (≤ `α` in common settings) and invariance to gradient rescaling.
  - Significance: easier hyperparameter selection and robust behavior across models with different gradient scales; the SNR view explains automatic annealing near optima.

- Unified view connecting AdaGrad and RMSProp (Section 5)
  - Novelty: shows Adam reduces to AdaGrad under a limit while behaving like a bias‑corrected RMSProp with a separate first‑moment track.
  - Significance: clarifies when and why Adam inherits strengths: sparse‑feature efficiency from AdaGrad and non‑stationary handling from RMSProp.

- Online‑learning convergence with adaptive moments (Section 4; Appendix)
  - Novelty: an `O(√T)` regret bound for Adam with decaying `α_t` and `β1,t`, under convexity and boundedness assumptions.
  - Significance: places Adam among methods with competitive theoretical guarantees while remaining practical.

- AdaMax: a simple, numerically stable infinity‑norm variant (Section 7.1)
  - Novelty: derivation by taking the `Lp` limit; update uses `u_t = max(β2 u_{t-1}, |g_t|)`.
  - Significance: fewer numerical issues and a very simple bound on steps; reported default `α = 0.002` works well in tested problems.

## 5. Experimental Analysis
- Evaluation methodology
  - Datasets and tasks:
    - Logistic regression on MNIST digits and IMDB bag‑of‑words sentiment (Section 6.1). IMDB features are highly sparse; dropout (50%) applied to input features.
    - Multilayer perceptrons (two hidden layers with 1000 ReLU units) on MNIST; both deterministic loss with `L2` weight decay and with dropout regularization (Section 6.2). Minibatch size 128.
    - Convolutional neural network on CIFAR‑10 with architecture `c64–c64–c128–1000` (three conv layers with 5×5 filters and 3×3 max‑pooling, then a 1000‑unit ReLU layer), with whitening and dropout on input and the fully connected layer (Section 6.3).
  - Baselines:
    - AdaGrad, RMSProp (for IMDB logistic regression), SGD with Nesterov momentum, AdaDelta, and SFO (a quasi‑Newton optimizer) for the deterministic MLP.
  - Schedules/hyperparameters:
    - Logistic regression uses `α_t = α / √t` to match the analysis in Section 4 (Section 6.1).
    - Minibatch size uniformly 128, consistent across comparisons (Sections 6.1–6.3).

- Main results
  - Logistic regression (Figure 1):
    - MNIST: Adam converges at roughly the same rate as SGD with Nesterov momentum and both are noticeably faster than AdaGrad. The left panel shows Adam and Nesterov curves dropping steeply in the first few full‑dataset iterations, reaching substantially lower training cost than AdaGrad by 45 iterations.
    - IMDB (sparse features with dropout): AdaGrad outperforms momentum SGD by a wide margin; Adam matches AdaGrad’s fast convergence. The right panel shows Adam+dropout and AdaGrad+dropout attaining the lowest training cost across 160 iterations.
  - Multilayer neural networks (Figure 2):
    - With dropout: Adam achieves the lowest training cost across ~200 passes, outperforming AdaGrad, RMSProp, Nesterov SGD, and AdaDelta (Figure 2a).
    - Deterministic objective vs. SFO: Adam reduces cost faster per iteration and per wall‑clock time. SFO requires curvature updates that make each iteration “5–10× slower” and its memory scales with the number of minibatches (Section 6.2).
    - SFO fails to converge with stochastic regularization (dropout), whereas Adam remains stable (Section 6.2).
  - Convolutional neural networks (Figure 3):
    - Early phase (first 3 epochs): Adam and AdaGrad drop the training cost fastest (left panel).
    - Later phase (45 epochs): Adam and SGD with momentum converge substantially faster than AdaGrad, whose progress slows markedly (right panel). Section 6.3 explains that Adam’s second‑moment estimate `v̂_t` “vanishes to zeros after a few epochs and is dominated by ε,” making geometry estimation less helpful; first‑moment variance reduction dominates, favoring Adam/SGD over AdaGrad.
  - Bias‑correction ablation (Figure 4):
    - With `β2` close to 1 (e.g., `0.999` and `0.9999`), removing bias‑correction produces instability “especially at first few epochs,” while bias‑correction yields lower loss both after 10 and 100 epochs when training a VAE (Section 6.4). The red (corrected) curves are consistently better than green (uncorrected) across stepsizes.

- Do the experiments support the claims?
  - Yes, across convex and non‑convex tasks, Adam shows:
    - At least parity with the best baseline in each setting, and clear wins under dropout/stochastic regularization (Figures 2 and 4).
    - Robustness to sparse features (IMDB; Figure 1 right) and non‑stationarity (dropout; Figure 2a).
    - Practical efficiency relative to quasi‑Newton SFO (Section 6.2).
  - Ablations: The bias‑correction study (Figure 4) directly tests the necessity of Adam’s correction and finds it critical when `β2` is large (a common practical choice).

- Numbers explicitly reported in the paper
  - Default hyperparameters used successfully: “α = 0.001, β1 = 0.9, β2 = 0.999, ε = 10^{-8}” (Algorithm 1 caption).
  - SFO per‑iteration cost: “5–10× slower per iteration compared to Adam” (Section 6.2).
  - Theoretical rates: regret `O(√T)` and average regret `O(1/√T)` (Theorem 4.1 and Corollary 4.2).

## 6. Limitations and Trade-offs
- Theoretical assumptions vs. practice (Section 4)
  - Convergence proof requires convex losses, bounded gradients/parameter distances, and decays `α_t = α/√t` and `β1,t = β1 λ^{t-1}`. Many practical deep‑learning uses keep `β1` constant and optimize highly non‑convex objectives; the proof does not apply there.
- Sensitivity to `β2` without correction (Section 6.4)
  - Large `β2` (slowly decaying second moment), which is desirable for sparsity, needs bias‑correction; otherwise training may diverge early (Figure 4).
- Second‑moment usefulness can diminish (Section 6.3)
  - In the presented CNN experiment, `v̂_t` “vanishes to zeros after a few epochs” and `ε` dominates, reducing curvature adaptivity; performance then resembles momentum‑based methods.
- Computational/storage profile
  - Adam stores two extra vectors (`m_t`, `v_t`), doubling memory over vanilla SGD. This is relatively small, but non‑negligible for extremely large models.
- Hyperparameter defaults are strong but not universally optimal
  - The paper provides defaults, but optimal `α` can still vary; Section 2.1 gives bounds that help, but tuning may still be required.

## 7. Implications and Future Directions
- Field impact
  - By offering an adaptive, stable, and largely tuning‑free optimizer with theoretical grounding and strong empirical results, Adam provides a robust default for training a broad range of models, especially when gradients are noisy, sparse, or non‑stationary (Sections 1, 2, 6).
  - The bias‑correction mechanism clarifies how to safely use slow EMA decays—a design element now standard in many optimizers.

- Follow‑up research enabled or suggested by this work
  - Extending theoretical guarantees to non‑convex settings and to constant `β1` schedules common in practice (Section 4 conditions).
  - Understanding when second‑moment information becomes uninformative (as in Section 6.3) and designing variants that adaptively down‑weight or replace it.
  - Exploring richer preconditioners beyond diagonal `v̂_t` while keeping memory affordable (Section 5 notes links to natural gradient and quasi‑Newton ideas).
  - Investigating temporal parameter averaging (Section 7.2) in combination with Adam across tasks for improved generalization.

- Practical applications
  - Training large neural networks under heavy stochastic regularization (dropout), NLP models with sparse features (IMDB BoW experiment), and general deep architectures where per‑parameter adaptivity and simple tuning are valued.
  - The AdaMax variant (Algorithm 2) offers a numerically stable alternative when `L∞`‑based scaling or tighter step bounds (`|Δ_t| ≤ α`) are preferred.

> Key takeaways grounded in the paper:
> - “Good default settings … α = 0.001, β1 = 0.9, β2 = 0.999 and ε = 10^{-8}” (Algorithm 1 caption).
> - Effective steps are approximately bounded by `α` and invariant to gradient rescaling (Section 2.1).
> - With decaying schedules, “Adam achieves … O(√T) regret” (Theorem 4.1) and average regret `O(1/√T)` (Corollary 4.2).
> - Bias‑correction is crucial at high `β2`; without it, training shows “instabilities … especially at first few epochs” (Section 6.4; Figure 4).
