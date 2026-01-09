# Random Search for Hyper-Parameter Optimization

**DOI:** [10.5555/1953048.1953056](https://doi.org/10.5555/1953048.1953056)

## Pitch

This paper shows that simple random search (i.i.d. sampling of hyper-parameter configurations) is more efficient than grid search for hyper-parameter optimization in many practical settings, because the hyper-parameter response surface typically has low effective dimensionality—only a few hyper-parameters matter. Empirically and theoretically, random search finds as-good-or-better models with far fewer trials (and with practical advantages for parallel and fault-tolerant execution), making it a strong, reproducible baseline for future adaptive optimization methods.

---

## 1. Executive Summary (2-3 sentences)
This paper argues that `random search` (i.i.d. sampling of hyper-parameter configurations) is a more computationally efficient baseline than `grid search` for hyper-parameter optimization, especially as the number of hyper-parameters grows. The central reason is that for many learning problems the hyper-parameter response function $\Psi(\lambda)$ has *low effective dimensionality*—only a few hyper-parameters strongly affect validation performance—so grid search wastes most trials on unimportant dimensions while random search continues to explore new values along the important ones (Figure 1, Section 3). Empirically, on neural networks random search matches or beats a prior grid search using far fewer trials (Figures 5–6), and on deep belief networks it is competitive with a sophisticated manual+grid procedure on several datasets (Figure 9).

## 2. Context and Motivation
- **Problem / gap.** Practical machine learning requires choosing hyper-parameters $\lambda$ (e.g., learning rate, regularization strength, number of hidden units), but we typically lack efficient algorithms to minimize the *outer-loop* objective:
  $$
  \lambda^{(*)}=\arg\min_{\lambda\in\Lambda}\ \mathbb{E}_{x\sim G_x}\Big[L\big(x;A_\lambda(X^{(\text{train})})\big)\Big]
  \tag{1}
  $$
  where $A_\lambda$ is the learning algorithm instantiated with hyper-parameters $\lambda$ (Section 1).
- **Why it matters.**
  - Hyper-parameter optimization directly governs generalization error, and model families are increasingly “large and hierarchical,” which increases the number of tunable knobs and makes naive search increasingly expensive (Abstract; Section 1).
  - Reproducibility matters: manual tuning is hard to reproduce, while grid search is reproducible but can be inefficient (Section 1).
- **What practitioners do.**
  - Replace the unknown expectation over $G_x$ using `cross-validation` / validation sets:
    $$
    \Psi(\lambda)\equiv \text{mean}_{x\in X^{(\text{valid})}} L\big(x;A_\lambda(X^{(\text{train})})\big)
    \tag{2–3}
    $$
  - Evaluate $\Psi(\lambda)$ on a finite set of trial configurations $\{\lambda^{(1)},\ldots,\lambda^{(S)}\}$ and pick the best observed one (Equation (4), Section 1).
  - Choose the trial set via grid search and/or manual search (Section 1).
- **Where grid search falls short (paper’s position).**
  - Grid search scales exponentially with the number of hyper-parameters (curse of dimensionality) because the number of configurations is the product of per-dimension choices (Section 1).
  - When only a few hyper-parameters matter for a given dataset, grid search “wastes” trials by repeatedly trying the same values along important dimensions while varying irrelevant ones (Figure 1).
- **Paper’s stance relative to prior work.**
  - It does not propose a new adaptive optimizer; instead it positions `random search` as a *simple, parallel, reproducible baseline* that is often strictly better than grid search and should be the baseline for judging more sophisticated sequential/adaptive methods (Abstract; Sections 1, 6, 7).

## 3. Technical Approach

### 3.1 Reader orientation (approachable technical breakdown)
- The “system” here is a methodology for selecting hyper-parameters: sample candidate configurations, train models, evaluate validation performance, and report what performance you can expect as you increase the number of trials.
- It solves hyper-parameter optimization as a black-box search problem, emphasizing *how to choose trial points* (random vs grid) and *how to evaluate experiments* fairly when many trials have similar validation scores.

### 3.2 Big-picture architecture (diagram in words)
- **(1) Define search space $\Lambda$** by specifying distributions/ranges for each hyper-parameter (Sections 2.4 and 5).
- **(2) Generate trials**:
  - `Grid search`: Cartesian product of fixed per-dimension sets (Section 1).
  - `Random search`: i.i.d. samples from the defined distributions (Section 1).
- **(3) For each trial $\lambda^{(s)}$**:
  - Train $A_{\lambda^{(s)}}$ on $X^{(\text{train})}$, possibly with early stopping (Section 2.4).
  - Compute validation loss/accuracy $\Psi^{(\text{valid})}(\lambda^{(s)})$ and test score $\Psi^{(\text{test})}(\lambda^{(s)})$ (Section 2.1).
- **(4) Aggregate results across trials** using:
  - A probabilistic “which trial is truly best?” weighting scheme (Equations (5)–(6), Section 2.1).
  - A `random experiment efficiency curve` that estimates the distribution of best-achieved performance vs number of trials (Section 2.2; Figure 2).
- **(5) Analyze why random works** via Gaussian process regression + `ARD`-style length scales to measure per-hyper-parameter sensitivity (Section 3; Figure 7).
- **(6) Additional comparison**: random vs low-discrepancy (Quasi Monte Carlo) point sets and vs expert-guided manual+grid tuning (Sections 4 and 5; Figures 8 and 9).

### 3.3 Roadmap for the deep dive
- Explain the formal hyper-parameter objective and how it is approximated in practice (Equations (1)–(4)).
- Describe the paper’s evaluation method for “best model from many trials,” including uncertainty from validation selection (Section 2.1; Equations (5)–(6)).
- Define and interpret random experiment efficiency curves (Section 2.2; Figure 2).
- Detail the actual sampled hyper-parameter spaces for (a) single-layer neural nets and (b) deep belief networks (Sections 2.4 and 5).
- Show the mechanism-level explanation for random search’s advantage via low effective dimensionality and ARD (Section 3; Figure 7).
- Summarize the simulation comparing grid/random/QMC sequences and what it implies for practice (Section 4; Figure 8).

### 3.4 Detailed, sentence-based technical breakdown
This is an **empirical + methodological** paper whose core idea is that *in high-dimensional hyper-parameter spaces, uniform random sampling is more efficient than grids because the objective $\Psi(\lambda)$ effectively depends on only a few coordinates* (Figure 1; Sections 1 and 3).

**Framing the optimization problem (outer loop).**
- A learning algorithm with hyper-parameters $\lambda$ maps training data to a predictor $f = A_\lambda(X^{(\text{train})})$ (Section 1).
- The goal is to minimize expected loss on the unknown data-generating distribution $G_x$ (Equation (1)), but since $G_x$ is unknown, the paper uses a validation set estimate:
  $$
  \Psi(\lambda)=\text{mean}_{x\in X^{(\text{valid})}}L\big(x;A_\lambda(X^{(\text{train})})\big)
  \tag{2–3}
  $$
- Because $\Psi(\lambda)$ is expensive (each evaluation means training a model), practice evaluates only $S$ trial points and picks the best (Equation (4), Section 1).

**How the paper reports “best model” performance when validation is noisy (Section 2.1).**
- The paper distinguishes validation and test estimates:
  - $\Psi^{(\text{valid})}(\lambda)$ is mean loss on $X^{(\text{valid})}$.
  - $\Psi^{(\text{test})}(\lambda)$ is mean loss on $X^{(\text{test})}$.
- It also estimates the *variance of the mean* on validation/test sets. For zero-one loss, it uses a Bernoulli variance estimate (Section 2.1):
  $$
  V^{(\text{valid})}(\lambda)=\frac{\Psi^{(\text{valid})}(\lambda)\big(1-\Psi^{(\text{valid})}(\lambda)\big)}{|X^{(\text{valid})}|-1}
  $$
  and analogously for $V^{(\text{test})}(\lambda)$.
- Instead of reporting the test score of the single best-validation trial (which can be unstable when several trials have similar validation means), it models each trial’s validation score as a Gaussian random variable:
  $$
  Z^{(i)}\sim \mathcal{N}\Big(\Psi^{(\text{valid})}(\lambda^{(i)}),\ V^{(\text{valid})}(\lambda^{(i)})\Big)
  $$
- It assigns each trial a weight
  $$
  w_s = P\Big(Z^{(s)} < Z^{(s')}\ \forall s'\neq s\Big),
  $$
  i.e., the probability that trial $s$ is truly the best given validation uncertainty (Section 2.1).
- It then reports the “best-of-experiment” test performance as a **mixture-weighted** mean and variance (Equations (5)–(6)):
  $$
  \mu_z=\sum_{s=1}^S w_s\,\mu_s,\quad \text{where } \mu_s=\Psi^{(\text{test})}(\lambda^{(s)})
  \tag{5}
  $$
  $$
  \sigma_z^2=\sum_{s=1}^S w_s\big(\mu_s^2+\sigma_s^2\big)-\mu_z^2,\quad \text{where } \sigma_s^2=V^{(\text{test})}(\lambda^{(s)})
  \tag{6}
  $$
- It estimates the weights $w_s$ by Monte Carlo simulation: repeatedly sample hypothetical validation scores from the Gaussians and count which trial wins (Section 2.1).

**Random experiment efficiency curve (Section 2.2; Figure 2).**
- Because random trials are i.i.d., a single random experiment with $S$ trials can be “reused” to estimate what would happen with smaller budgets.
- Concretely, if $S=256$ and you want the distribution of outcomes for experiments of size $s=8$, you can partition the 256 trials into $256/8=32$ independent 8-trial “sub-experiments” and compute best-of-sub-experiment performance using the weighted estimator above (Section 2.2; Figure 2).
- The resulting curve shows quantiles (boxplots) of achieved test accuracy as a function of $s$ (Figure 2), giving a reproducibility-oriented view: what performance range should you expect if you only can afford $s$ random trials?

**Neural network case study: what exactly is being randomized (Section 2.4).**
- The paper compares against a prior grid search study (Larochelle et al., 2007) by sampling trials from a distribution intended to cover “roughly the same domain with the same density,” except that random search also optionally includes preprocessing variants (Section 2.4).
- For single-hidden-layer neural networks, it samples (among others) the following hyper-parameters (Section 2.4):
  - **Preprocessing choice** (optional space expansion): none vs normalize vs PCA (each with probability $1/3$), and for PCA, keep a fraction of variance uniformly in $[0.5,1.0]$.
  - **Weight initialization distribution**: uniform on $(-1,1)$ vs unit normal, plus one of two scaling heuristics (LeCun-style fan-in scaling with a sampled multiplier, or a $\sqrt{6}/\sqrt{\text{fan-in}+\text{fan-out}}$-style scaling), and one of three RNG seeds.
  - **Hidden layer size**: “drawn geometrically” from 18 to 1024 (defined in footnote 3 as log-uniform sampling followed by rounding).
  - **Activation**: sigmoid vs tanh (50/50).
  - **Mini-batch size**: 20 vs 100 (50/50).
  - **Optimizer**: stochastic gradient descent with initial learning rate $\varepsilon_0$ log-uniform (“geometrically”) from 0.001 to 10.0.
  - **Learning rate annealing**: optional via an anneal time $t_0$ drawn geometrically from 300 to 30000, using:
    $$
    \varepsilon_t = \frac{t_0\varepsilon_0}{\max(t,t_0)}
    \tag{7}
    $$
  - **Training duration / early stopping**: 100–1000 iterations over training data, stopping if at iteration $t$ the best validation was observed before $t/2$ (Section 2.4).
  - **$\ell_2$ regularization**: applied with probability 0.5; if applied, strength sampled log-uniformly from $3.1\times 10^{-7}$ to $3.1\times 10^{-5}$.
- For each dataset, the paper runs **$S=256$ random trials** and then builds efficiency curves (Section 2.4; Figures 5 and 6).

**Datasets and sizes (Section 2.3; Figures 3–4).**
- Image size is **28×28 grayscale** across the MNIST-derived and synthetic tasks (Section 2.3).
- MNIST variants use **10,000 train / 2,000 validation / 50,000 test** examples (mnist basic, background images, background random, rotated, rotated background images).
- `rectangles` uses **1,000 train / 200 validation / 50,000 test** (binary classification: tall vs wide rectangle).
- `rectangles images` and `convex` use **10,000 train / 2,000 validation / 50,000 test** (Section 2.3).
- The tasks intentionally include “many factors of variation” (Figures 3–4) to stress model selection (Section 2.3).

**Mechanism for why random beats grid: low effective dimensionality (Section 3; Figure 7).**
- The paper formalizes the intuition with *effective dimensionality*: if $\Psi(\lambda)$ is mainly sensitive to a small subset of hyper-parameters, then spending trials on the others yields little gain (Section 1; Figure 1).
- It uses Gaussian process regression with squared exponential (RBF) kernels to model $\Psi(\lambda)$ and to estimate per-dimension length scales $l_k$ (Section 3).
  - Similarity in a single scalar hyper-parameter is modeled as $\exp(-((a-b)/l)^2)$ (Section 3).
  - Kernels across hyper-parameters are combined by multiplication (a product kernel).
  - Hyper-parameters are rescaled into $[0,1]$ to compare length scales, and for log-scaled parameters (learning rate, hidden units) the kernel uses the log of the effective value (Section 3).
- It estimates length scales by maximizing GP marginal likelihood; because this is non-convex, it repeats fitting **50 times** per dataset with:
  - random 80% subsamples of observations, and
  - random initialization of length scales in $[0.1,2]$ (Section 3).
- It interprets “relevance” as $1/l$ (Figure 7): small length scale $\Rightarrow$ high sensitivity $\Rightarrow$ high relevance.
- Empirical finding (Figure 7; Section 3):
  - For any one dataset, **only a small number of hyper-parameters dominate**.
  - **Which hyper-parameters matter changes across datasets** (e.g., learning rate is always important, but other dimensions like $\ell_2$ penalty, number of hidden units, or annealing differ by dataset).
- This directly undermines grid design for “new datasets”: even if you could afford a fine grid, you don’t know which dimensions deserve resolution ahead of time (Section 3).

**Comparison to quasi-random / low-discrepancy sequences (Section 4; Figure 8).**
- The paper asks whether better-than-random deterministic point sets (Sobol, Halton, Niederreiter, Latin hypercube) can outperform random sampling in the small-budget regime relevant to hyper-parameter tuning (Section 4).
- It evaluates via a synthetic search task: find a randomly placed target region occupying **1% of the unit hypercube volume**, in:
  - 3D cube target,
  - 3D hyper-rectangle target (elongated, hence lower effective dimension),
  - 5D cube target,
  - 5D hyper-rectangle target (Section 4; Figure 8).
- It evaluates experiment sizes from 1 up to **512 trials** (Section 4).
- Key outcomes (Figure 8; Section 4):
  - Many grids perform poorly, especially for elongated rectangular targets (the analogue of low effective dimensionality).
  - Latin hypercube is “no more efficient than” random in these simulations.
  - Sobol is “consistently best by a few percentage points,” especially around 100–300 trials (Section 4).

**Deep Belief Network (DBN) case study: random vs expert-guided manual+grid (Section 5; Figure 9).**
- DBNs have far more hyper-parameters because there are per-layer pretraining hyper-parameters plus finetuning hyper-parameters (Section 5).
- The paper defines a random search distribution for DBNs including:
  - number of layers: 1, 2, or 3 (uniform),
  - per-layer hidden units: log-uniform in [128, 4000],
  - per-layer initialization choices and scaling switches,
  - per-layer contrastive divergence iterations: log-uniform in [1, 10000],
  - per-layer pretraining learning rate: log-uniform in [0.0001, 1.0],
  - per-layer anneal time: log-uniform in [10, 10000],
  - preprocessing: raw vs ZCA; if ZCA, keep variance uniformly in [0.5,1.0],
  - RNG seed choice (2, 3, or 4),
  - finetuning learning rate: log-uniform in [0.001, 10],
  - finetuning anneal time: log-uniform in [100, 10000],
  - finetuning $\ell_2$: 0 w.p. 0.5 else log-uniform in [$10^{-7},10^{-4}$] (Section 5).
- This yields **8 global + 8 per-layer hyper-parameters**, i.e. up to **32 hyper-parameters** for 3-layer models (Section 5).
- The baseline manual+grid approach from Larochelle et al. (2007) is described as alternating architecture choices with coordinate-descent-like tuning of optimization hyper-parameters, using variable numbers of trials per dataset (e.g., 13 to 102; average 41 for DBN-3) (Section 5).
- The random search results are summarized qualitatively in Figure 9 and text (Section 5):
  - Versus the 3-layer DBN manual+grid results: random search finds **better on 1 dataset (convex)**, **statistically equal on 4**, and **worse on 3**.
  - Versus 1-layer DBN results: random search over the larger space finds “at least as good a model in all cases” (Section 5).

## 4. Key Insights and Innovations
- **(1) Random search is an unexpectedly strong baseline in high dimensions.**
  - Novelty is not the idea of randomness itself, but the paper’s argument—supported by experiments—that *purely random i.i.d. trials* can match or beat grid search at a fraction of the computation in realistic neural net tuning (Abstract; Section 2.4; Figures 5–6).
  - Significance: it reframes “default practice” for hyper-parameter optimization, especially in research workflows that rely on clusters and parallel evaluation.
- **(2) Low effective dimensionality explains grid search failures.**
  - The key conceptual contribution is connecting hyper-parameter tuning to the idea that $\Psi(\lambda)$ depends strongly on only a few coordinates, and that these coordinates differ across datasets (Sections 1 and 3; Figure 7).
  - This is more than “grid is exponential”: it explains *why* the wasted trials happen in practice (Figure 1).
- **(3) A practical method to report “best-of-search” performance with uncertainty.**
  - The weighted mixture approach (Section 2.1; Equations (5)–(6)) explicitly accounts for the fact that many configurations may be indistinguishable on a finite validation set.
  - This is a methodological contribution to empirical reporting: it separates uncertainty from (i) estimating a model’s test score and (ii) selecting which model is best based on validation noise.
- **(4) Clarifying the role of quasi-random sequences.**
  - The simulation (Section 4; Figure 8) suggests low-discrepancy sequences can be slightly better than pseudo-random points in some regimes, but also emphasizes that grid search is particularly ill-suited to elongated/low-effective-dimension targets.
  - The contribution is a grounded “don’t overcomplicate it” message: improvements over random are modest in the small-trial regime (Section 4).

## 5. Experimental Analysis
**Evaluation methodology (what is compared and how).**
- **Primary empirical comparison**: random search vs previously reported grid/manual tuning results from Larochelle et al. (2007) on multiple datasets (Sections 2 and 5).
- **Metrics**: the figures plot **test-set accuracy** as a function of experiment size (Figures 2, 5, 6, 9). The paper defines losses in general but uses accuracy plots in these experiments.
- **Experimental protocol**:
  - For neural nets, the paper runs **$S=256$ random trials per dataset** (Section 2.4) and then constructs random experiment efficiency curves (Section 2.2).
  - It compares those curves to a **dashed blue line** indicating grid search accuracy from Larochelle et al. (2007), described as “grids averaging 100 trials” (Figure 5 caption; Figure 6 caption).
  - For DBNs, it samples from a 32-dimensional space (for 3-layer models) and compares to Larochelle et al. (2007)’s manual+grid tuning using an average of **41 trials** for DBN-3, but varying by dataset (Section 5; Figure 9 caption/text).

**Main results (with the paper’s level of specificity).**
- **Neural networks (7 hyper-parameters, no preprocessing):**
  - Figure 5 and its caption state that **random searches of 8 trials match or outperform grid searches of ~100 trials** across the shown datasets when considering only “no preprocessing” trials (Section 2.4; Figure 5 caption).
- **Neural networks (9 hyper-parameters, with preprocessing options):**
  - Expanding the search space (allowing normalization and PCA) makes the space “less promising,” so more trials are needed to reliably beat the baseline, but **64 random trials are enough to find better models** in that larger space (Section 2.4; Figure 6 caption).
  - The paper notes a concrete grid-cost implication: even adding “just four PCA variance levels” to a grid would multiply trials substantially (Figure 6 caption mentions “5 times as many … average 500 trials per data set”).
- **Gaussian process/ARD analysis (why results look like this):**
  - Figure 7 shows that per dataset, a small subset of hyper-parameters dominates, and which ones dominate varies by dataset (Section 3; Figure 7).
  - The paper also links datasets that plateau quickly in the efficiency curves (Figure 5) with lower effective dimension in ARD, and harder datasets with higher effective dimension (Section 3 discussion).
- **QMC vs random vs grid (simulation):**
  - In the “find a 1% target region” simulation, **most grids are worst**, Sobol is best by a small margin, and Latin hypercube is roughly equivalent to random (Section 4; Figure 8).
- **DBNs (up to 32 hyper-parameters):**
  - Random search is **not uniformly superior** to an expert-guided sequential process (Section 5; Figure 9).
  - The paper’s summary: compared with manual+grid DBN-3 results, random search is *better on 1 dataset (convex), equal on 4, worse on 3* (Section 5).

**Do experiments support the claims?**
- For the claim “random > grid in high-dimensional hyper-parameter tuning,” the neural net experiments provide direct evidence in a realistic setting (Section 2.4; Figures 5–6) and the GP analysis provides a plausible explanatory mechanism (Section 3; Figure 7).
- For the stronger implicit claim “random is competitive with expert tuning,” the DBN results are mixed: they show competitiveness on several datasets but not consistent superiority (Section 5; Figure 9). This nuance is explicitly acknowledged in the text.

**Ablations / robustness checks present in the paper.**
- The GP length-scale estimation is repeated with resampling and random restarts (50 fits, 80% subsampling, random initial length scales), which functions as a robustness check on which hyper-parameters appear relevant (Section 3; Figure 7 caption).
- The paper also compares multiple point-set strategies in simulation, not only random vs grid (Section 4; Figure 8).

## 6. Limitations and Trade-offs
- **Random search can struggle when good regions are small (“peaked” $\Psi$).**
  - The paper notes that for some datasets the efficiency curves show substantial variability even at 16–32 trials (Section 2.4 discussion), suggesting small high-performing regions that i.i.d. sampling may miss.
  - The DBN case reinforces this: even with larger trial counts, variability remains and random does not reliably find the best regions on some datasets (Section 5).
- **Non-adaptive by design.**
  - Random search (and grid, and low-discrepancy non-adaptive sets) does not use results from completed trials to focus future sampling, which is a potential inefficiency when compute budgets are limited and the objective has exploitable structure (Sections 1 and 6).
- **Dependence on parameterization / sampling distribution choices.**
  - The method’s effectiveness depends on how you choose distributions (e.g., log-uniform vs uniform ranges for learning rates, hidden units, regularization) (Sections 2.4 and 5).
  - The paper explicitly flags that “work remains” to establish good parameterizations for reliable i.i.d. search per model family (Section 6).
- **Weighted performance estimator may underestimate (paper’s own caveat).**
  - In Section 5, the paper notes its score-averaging technique is “slightly biased toward underestimation” compared to the earlier study’s reporting, complicating apples-to-apples comparisons for DBNs.
- **Low-discrepancy sets are not i.i.d.**
  - The paper highlights a practical analysis limitation: the random experiment efficiency curve analysis relies on i.i.d. trials, which does not hold for Sobol/Halton-type sequences (Section 7).

## 7. Implications and Future Directions
- **How this changes practice.**
  - For many ML workflows, the paper suggests replacing grid search with random search as the default non-adaptive strategy because it preserves simplicity and parallelism while allocating trials more effectively in high-dimensional spaces (Sections 1 and 7).
  - It also reframes “high throughput” success: searching many hyper-parameters can work because most of them do not matter much for a given dataset (Abstract; Section 3).
- **Follow-up research directions suggested by the paper.**
  - Develop and evaluate **adaptive/sequential** hyper-parameter optimization methods (including Bayesian optimization / sequential model-based optimization), with random search as the baseline comparator (Section 6; Section 7).
  - Explore methods that identify important dimensions early and allocate sampling density accordingly (Section 6 mentions dimension screening ideas).
- **Practical applications / downstream use cases.**
  - Any setting with many hyper-parameters and easy parallel evaluation (cluster-based training) benefits: neural networks, deep models, and more generally algorithm configuration problems (Sections 1, 6, 7).
- **Repro/Integration Guidance (when to prefer what).**
  - Prefer `random search` over `grid search` when:
    - the number of hyper-parameters is moderate-to-large (the paper’s motivating regime), and/or
    - you suspect only a few hyper-parameters truly matter but you don’t know which ahead of time (Sections 1 and 3; Figure 7).
  - Grid search remains reasonable when:
    - the space is truly low-dimensional (the paper explicitly notes grid reliability in 1–2 dimensions in Section 1), and you can afford enough resolution.
  - Consider `Sobol`/low-discrepancy sequences if:
    - you want a fixed, non-adaptive design and can carefully execute all planned trials, and
    - you are in a regime where modest improvements over random might matter (the paper sees small gains around 100–300 trials in simulation; Section 4; Figure 8).
  - Move to adaptive methods if:
    - the objective is expensive, the good region is small/peaked (as suggested by the DBN variability), and you can afford the engineering complexity of a controller/master process that schedules sequential trials (Section 6).


