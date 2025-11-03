# ADAM: A METHOD FOR STOCHASTIC OPTIMIZATION

**ArXiv:** [1412.6980](https://arxiv.org/abs/1412.6980)

## 🎯 Pitch

Adam introduces a groundbreaking optimization method that adaptively computes individual learning rates for each parameter using estimates of first and second moments of the gradients, combined with bias correction. This approach makes training large, complex models more efficient, robust, and less sensitive to hyperparameter tuning—particularly in settings with noisy, sparse, or non-stationary gradients—making Adam pivotal for advancing deep learning and large-scale machine learning.

---

## 1. Executive Summary
Adam is an optimization algorithm for training models with noisy or minibatch gradients. It adaptively tunes a separate learning rate for every parameter by tracking two running statistics of the gradient (its mean and uncentered variance) and corrects their early-iteration bias. This makes optimization fast, memory‑light, robust to non‑stationarity and sparse gradients, and theoretically sound with an online convex optimization regret bound of order O(√T) (Section 4; Theorem 4.1).

## 2. Context and Motivation
- Problem addressed
  - Training modern models requires optimizing stochastic objectives where only noisy gradients are available (Section 1). Conventional first‑order methods like SGD can be sensitive to the learning rate, struggle with sparse gradients, and require manual schedule tuning.
- Why it matters
  - Practical: Faster, stabler training on large datasets and large parameter spaces with minimal tuning is crucial for deep learning and other high‑dimensional problems.
  - Theoretical: A method that adapts per‑parameter step sizes while keeping convergence guarantees helps bridge practice and theory.
- Prior approaches and their gaps
  - SGD with momentum: effective but needs careful global learning rate schedules and often different rates per layer or feature frequency.
  - AdaGrad (Duchi et al., 2011): accumulates squared gradients to adapt per‑parameter rates and excels with sparse features, but its learning rate can decay too aggressively (Section 5).
  - RMSProp (Tieleman & Hinton, 2012): maintains an exponential moving average of squared gradients to cope with non‑stationarity, but (i) lacks the bias correction needed when decay is slow and (ii) applies momentum to the rescaled gradient instead of using an explicit estimate of the first moment (Section 5).
- Positioning
  - Adam combines the strengths of AdaGrad and RMSProp: it uses exponentially decayed moving averages (good for non‑stationarity) and explicit bias‑corrected estimates of both the first and second moments (good for stability and sparse gradients). It also provides convergence analysis and an infinity‑norm variant (AdaMax) with a simpler bound (Section 7.1).

## 3. Technical Approach
Adam estimates two statistics of the gradient at each timestep t and uses them to scale parameter updates.

Core idea in plain language
- Keep two running “memories”:
  - m_t: an exponentially decayed average of recent gradients (an estimated mean).
  - v_t: an exponentially decayed average of recent squared gradients (an estimated uncentered variance).
- Because both are initialized at zero, early values are biased low. Adam fixes this by dividing each by a known factor so they become unbiased.
- Update each parameter by stepping in the direction of the estimated mean, scaled by the square root of the estimated variance. Intuition: trust directions that are consistently pointing the same way (large mean) and discount directions that fluctuate a lot (large variance).

Step‑by‑step (Algorithm 1; defaults α=0.001, β1=0.9, β2=0.999, ε=1e−8)
1. Compute stochastic gradient g_t = ∇_θ f_t(θ_{t−1}).
2. Update first‑moment (mean) estimate:
   - m_t = β1 · m_{t−1} + (1−β1) · g_t.
3. Update second raw moment (uncentered variance) estimate:
   - v_t = β2 · v_{t−1} + (1−β2) · g_t^2 (element‑wise square).
4. Bias correction (Section 3):
   - m̂_t = m_t / (1−β1^t),
   - v̂_t = v_t / (1−β2^t).
   - Why needed: with zero initialization, early averages are biased toward zero; Section 3 derives the correction by taking expectations of the exponential average (Equations (1)–(4)).
5. Parameter update (element‑wise):
   - θ_t = θ_{t−1} − α · m̂_t / (sqrt(v̂_t) + ε).

Design choices and their rationale
- Exponential moving averages: give more weight to recent gradients, handling non‑stationarity better than AdaGrad’s cumulative sum (Section 2).
- Bias correction: crucial when β2 is close to 1 (slow decay), as otherwise v_t is severely underestimated and steps can explode (Section 3 and Section 6.4; Figure 4).
- Per‑parameter scaling: divides by sqrt(v̂_t), shrinking updates for parameters with high gradient variance and expanding them for stable parameters (Sections 2–2.1).
- ε: prevents division by zero and bounds the denominator when v̂_t becomes tiny; in practice set to 1e−8 (Algorithm 1).

How the update behaves (Section 2.1)
- Effective step per coordinate: Δ_t = α · m̂_t / sqrt(v̂_t) (assuming ε≈0).
- Bounded step sizes: Section 2.1 shows |Δ_t| is approximately bounded by α (exact upper bound depends on β1, β2), which acts like a trust region: the algorithm rarely makes steps much larger than the base learning rate.
- Scale‑invariance: scaling all gradients by a constant c scales m̂_t by c and v̂_t by c^2, which cancels in Δ_t; thus updates are invariant to gradient rescaling (Section 2.1).
- Automatic annealing: the “signal‑to‑noise ratio” SNR ≡ m̂_t / sqrt(v̂_t) shrinks near optima (mean decreases faster than variance), automatically reducing step sizes without an explicit schedule (Section 2.1).

Computational and memory cost
- Memory: stores m_t and v_t for each parameter (two extra arrays the size of θ; Section 1).
- Compute: constant overhead per parameter update relative to SGD (Algorithm 1).

Theoretical framework (Section 4)
- Setting: online convex optimization. At each step t, a convex loss f_t is revealed after choosing θ_t.
- Goal: small cumulative regret R(T) = Σ_t [f_t(θ_t) − f_t(θ*)], where θ* is the best fixed comparator in hindsight.
- Result: With decaying learning rate α_t = α / √t and decaying momentum β1,t = β1 λ^{t−1}, Adam achieves R(T) = O(√T) (Theorem 4.1; Corollary 4.2).
- Intuition of the proof: combine the convexity inequality (Lemma 10.2) with bounds on sums involving the bias‑corrected moments (Lemmas 10.3 and 10.4).

Connections and variants
- Relation to AdaGrad and RMSProp (Section 5): Adam reduces to AdaGrad when β1=0 and β2→1 with bias correction and α_t=α/√t; it differs from RMSProp by using explicit bias‑corrected first/second moments rather than momentum on rescaled gradients and by avoiding the instability seen when β2 is close to 1 without correction.
- AdaMax (Algorithm 2; Section 7.1): a p‑norm generalization where p→∞ leads to u_t = max(β2·u_{t−1}, |g_t|). Update becomes θ_t = θ_{t−1} − (α/(1−β1^t)) · m_t / u_t. No bias correction for u_t is required, and the update magnitude is bounded by α.

## 4. Key Insights and Innovations
- Bias‑corrected moment estimates
  - What: Divide m_t and v_t by (1−β1^t) and (1−β2^t), respectively (Algorithm 1; Section 3).
  - Why it’s novel/important: Corrects the initialization bias intrinsic to exponential moving averages started at zero. Section 3 derives the correction (Equations (1)–(4)). Empirically critical when β2 is close to 1 (sparse/noisy settings); Figure 4 shows training instability without correction and stable convergence with it.
- Per‑coordinate, variance‑normalized updates with bounded magnitude
  - What: Use m̂_t / sqrt(v̂_t) to scale each coordinate; Section 2.1 proves step size bounds roughly by α and invariance to gradient rescaling.
  - Why it matters: Reduces the burden of tuning α and improves robustness across layers and parameter scales; supports consistent progress even with non‑stationary or heteroskedastic gradients.
- Theoretical regret bound that leverages adaptivity under sparsity
  - What: Theorem 4.1 shows O(√T) regret with constants that depend on coordinate‑wise accumulators. Under sparse gradients, the sums Σ_i ||g_{1:T,i}||_2 and Σ_i √T v̂_{T,i}^0.5 can be much smaller than d·G∞√T, yielding tighter guarantees similar to AdaGrad’s improvements (Section 4, paragraph after Theorem 4.1).
  - Why it matters: Provides formal backing for observed gains on sparse features (e.g., IMDB BoW; Figure 1 right).
- AdaMax: an infinity‑norm variant with simple state and bound
  - What: Replace sqrt(v̂_t) with u_t = max(β2 u_{t−1}, |g_t|); no bias correction needed for u_t and |Δ_t| ≤ α (Section 7.1; Algorithm 2).
  - Why it matters: Numerically stable alternative with minimal bookkeeping and a clear update bound.

## 5. Experimental Analysis
Evaluation setup (Section 6)
- Datasets and models
  - Logistic regression on MNIST images (Figure 1 left) and on IMDB reviews represented as 10,000‑dimensional sparse bag‑of‑words (Figure 1 right).
  - Multilayer perceptron (MLP): two hidden layers, 1000 ReLU units each, minibatch 128; with and without dropout; comparison to the Sum‑of‑Functions Optimizer (SFO) for deterministic cost (Figure 2).
  - Convolutional neural networks (CNNs) on CIFAR‑10 with architecture c64‑c64‑c128‑1000: three 5×5 conv layers, 3×3 max‑pool (stride 2), fully connected layer with 1000 ReLUs; input whitening; dropout on input and fully connected layers; minibatch 128 (Figure 3).
- Baselines and hyperparameters
  - AdaGrad, RMSProp, SGD with Nesterov momentum, AdaDelta, and SFO (Figures 1–3). Learning rates and momenta are tuned on a grid; Adam uses defaults unless otherwise stated (Section 6).
  - For theoretical comparisons in logistic regression, α_t = α/√t to match the analysis (Section 6.1).

Main results
- Logistic regression, MNIST (Figure 1 left)
  - Observation: Adam converges at least as fast as SGD with Nesterov momentum and faster than AdaGrad in training negative log‑likelihood over 45 passes.
  - Takeaway: For dense features, Adam retains the speed of well‑tuned momentum methods without per‑problem scheduling.
- Logistic regression, IMDB BoW with dropout (Figure 1 right)
  - Observation: Adagrad and Adam significantly outperform SGD with Nesterov momentum. The plotted training cost shows a marked gap where AdaGrad and Adam rapidly drop to ≈0.25–0.3 while SGD remains higher across 160 passes.
  - Takeaway: In sparse feature regimes, adaptivity by per‑coordinate scaling is crucial; Adam matches AdaGrad while handling noise via exponential averaging.
- MLPs on MNIST (Figure 2)
  - With dropout (Figure 2a): Adam achieves the lowest training cost across iterations among first‑order methods (AdaGrad, RMSProp, AdaDelta, SGD+Nesterov).
  - Without dropout, comparison to SFO (Figure 2b): Adam reduces cost faster both per iteration and wall‑clock time; SFO requires 5–10× more time per iteration due to curvature updates and has memory linear in the number of minibatches (Section 6.2).
  - Additional note: SFO fails to converge when the objective includes stochastic regularization such as dropout (Section 6.2).
- CNNs on CIFAR‑10 (Figure 3)
  - Early epochs (left): Adam and AdaGrad both lower cost rapidly in the first three epochs.
  - Full training (right): Adam and SGD with momentum eventually converge considerably faster than AdaGrad. Section 6.3 reports that v̂_t “vanishes to zeros after a few epochs and is dominated by ε,” making the second‑moment estimate a poor geometry proxy for this CNN, while the first‑moment term (variance reduction) is more important.
  - Takeaway: On CNNs, Adam provides marginal improvement over SGD with momentum and removes the need for layer‑specific learning rates, but AdaGrad’s cumulative second‑moment can over‑attenuate updates.
- Bias‑correction ablation (Figure 4)
  - Setup: Train a variational autoencoder (single hidden layer of 500 softplus units; 50‑dim Gaussian latent) across grids of β1∈{0,0.9}, β2∈{0.99,0.999,0.9999}, log10(α)∈[−5,−1].
  - Result: Without bias correction, training is unstable for β2 close to 1, especially in early epochs; with correction, the loss curves are stable and generally better after 100 epochs.
  - Conclusion: Bias correction is not optional when slow decay is needed (e.g., sparse gradients).

Do the experiments support the claims?
- Yes, across three regimes:
  - Dense convex (MNIST logistic regression): fast convergence similar to well‑tuned momentum.
  - Sparse features (IMDB): clear advantage over momentum; matches AdaGrad.
  - Deep non‑convex (MLP/CNN): strong performance and robustness with dropout; competitive vs SGD and better than AdaGrad; SFO limitations with stochastic objectives are highlighted.
- Robustness checks
  - Bias‑correction ablation (Figure 4) demonstrates the necessity of Adam’s correction mechanism.
  - Multiple baselines and hyperparameter searches are used; however, quantitative numbers are shown primarily as curves rather than tables, so exact numeric margins are not reported.

## 6. Limitations and Trade-offs
- Theoretical assumptions (Section 4)
  - Convergence analysis requires convex losses, bounded gradients (||∇f_t(θ)||_2 ≤ G), and bounded parameter distances (||θ_m−θ_n||_2 ≤ D), with specific decay schedules α_t=α/√t and β1,t=β1·λ^{t−1}. These do not hold in general for deep non‑convex problems.
- Behavior under certain architectures (Section 6.3)
  - In the CNN setting, the second‑moment estimate v̂_t can become so small that ε dominates, weakening geometry adaptation and limiting the benefit over SGD with momentum.
- Hyperparameter coupling
  - While default (α, β1, β2, ε) works broadly (Algorithm 1), performance can depend on decay choices; the proof also relies on decaying β1,t, which is not always used in practice.
- No second‑order curvature
  - Adam uses only first‑order information. On smooth deterministic problems with reliable curvature estimates, quasi‑Newton methods may converge in fewer function evaluations (though at higher per‑iteration cost).
- Evaluation scope
  - Results are reported on standard vision/NLP benchmarks with training cost curves; broader tasks, test accuracy comparisons, or large‑scale industrial systems are not detailed in this paper.

## 7. Implications and Future Directions
- Impact on the field
  - Adam becomes a standard default optimizer because it combines fast initial progress, resilience to noise and sparsity, minimal tuning, and a principled foundation. The bias‑corrected moments and bounded, scale‑invariant steps generalize well across tasks.
- Follow‑up research enabled or suggested
  - Alternative decay schedules and adaptive β1,t strategies grounded in the theory of Section 4.
  - Diagnostics for when v̂_t collapses (as seen in Section 6.3) and mechanisms to prevent ε‑domination, potentially by mixing AdaMax‑style norms (Section 7.1) or adaptive ε.
  - Richer averaging schemes (Section 7.2) and their effect on generalization with Adam‑style updates.
  - Extensions that decouple regularization from adaptive scaling or combine Adam with curvature sketches while keeping O(d) memory.
- Practical applications
  - Training deep networks with dropout or other stochastic regularizers (Section 6.2), logistic models with sparse features (IMDB BoW; Section 6.1), and large‑parameter models where per‑parameter adaptation and low memory footprint matter.
  - The AdaMax variant (Algorithm 2) offers a simple, stable alternative in scenarios with heavy‑tailed gradients or when a hard bound |Δ_t| ≤ α is desirable.

Key takeaways grounded in the paper
- Adam’s update rule (Algorithm 1) is simple to implement and computationally efficient.
- Bias correction (Section 3) is essential for stability, particularly with β2 ≈ 1; Figure 4 empirically validates this.
- Step sizes are effectively bounded and invariant to gradient scaling (Section 2.1), improving robustness and ease of use.
- Convergence with O(√T) regret in online convex settings (Theorem 4.1) aligns with the best known rates while benefiting from adaptivity, especially under sparsity.
- Empirical results across MNIST, IMDB, and CIFAR‑10 (Figures 1–3) show strong practical performance and clarify when Adam excels (sparse, noisy, non‑stationary) and when gains are modest (some CNN regimes).
