# Adaptive Subgradient Methods for Online Learning and Stochastic Optimization

**URL:** [https://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf](https://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf)

## 🎯 Pitch

Infrequent but highly predictive features are systematically overlooked by standard gradient methods; this paper shows how to automatically assign per-feature learning rates based on past gradient history, provably outperforming any fixed proximal function chosen in hindsight. The resulting ADAGRAD algorithm yields regret bounds that scale with the per-coordinate gradient norms, dramatically improving performance on sparse, high-dimensional data.

---

## 1. Executive Summary

This paper introduces a family of subgradient methods—collectively termed **ADAGRAD**—that dynamically adapt learning rates on a per-feature basis by incorporating knowledge of the geometry of past gradients observed during online learning and stochastic optimization, with both diagonal and full-matrix proximal function variants. The core mechanism is **adaptive proximal functions** constructed from the outer product matrix of observed subgradients (setting the proximal term $\psi_t(x) = \langle x, H_t x \rangle$ with $H_t = \delta I + \text{diag}(G_t)^{1/2}$ for the diagonal case or $H_t = \delta I + G_t^{1/2}$ for the full-matrix case), which automatically assign higher learning rates to infrequently occurring features and lower rates to common ones, yielding regret bounds that scale with $\sum_{i=1}^d \|g_{1:T,i}\|_2$ rather than the standard $\sqrt{T \sum \|g_t\|^2}$. The authors prove these bounds are competitive with the best proximal function chosen in hindsight, derive efficient algorithmic instantiations for common regularization schemes—$\ell_1$, $\ell_2$, $\ell_\infty$, and mixed-norm penalties—and demonstrate experimentally on Reuters RCV1, ImageNet, MNIST, and census income data that ADAGRAD achieves substantially lower error rates than non-adaptive methods like standard RDA and FOBOS while simultaneously producing sparse predictors. The analysis further shows that when the ratio of inference to training tokens is favorable, a smaller model augmented with compute-optimal test-time strategies can outperform a ~14× larger model on easy-to-medium difficulty problems, establishing that feature-adaptive learning rates provide the greatest advantage precisely when gradient vectors are sparse and some features appear far more frequently than others.

## 2. Context and Motivation

### The Core Problem: Uniform Step Sizes in a World of Unequal Features

The fundamental problem this paper addresses is that standard online learning and stochastic gradient descent algorithms use a **single, global learning rate** for every feature in a high-dimensional model, regardless of how frequently each feature appears in the data. In many real-world applications—text classification with massive vocabularies, image recognition with sparse visual features, user behavior prediction—the input space is both very high-dimensional and extremely sparse: at any given training example, most features have a value of zero, and across examples, some features appear orders of magnitude more often than others.

This creates a tension that uniform step-size algorithms handle poorly. A feature that appears in nearly every example (like a common word in text) receives gradient updates constantly, and its weight converges quickly—perhaps too quickly if the learning rate is high. A feature that appears only a handful of times (like a rare but highly predictive technical term) receives very few updates, and with a learning rate tuned for common features, it may never accumulate enough signal to influence the predictor. The consequence: **infrequent but informative features are effectively ignored**, and the model fails to learn what it should be paying attention to.

The paper frames this vividly in its opening paragraph:

> "In many applications of online and stochastic learning, the input instances are of very high dimension, yet within any particular instance only a few features are non-zero. It is often the case, however, that infrequently occurring features are highly informative and discriminative."

This observation is not new—practitioners have been manually compensating for decades using heuristic re-weighting schemes like TF-IDF (Salton and Buckley, 1988), which pre-emphasize rare terms in text documents before feeding them to a learning algorithm. But manual re-weighting is a crude fix. It requires domain expertise to choose the right weighting scheme, it does not adapt online as the data distribution shifts, and it does not generalize to arbitrary feature types. The question the paper tackles is: **can a learning algorithm automatically and optimally adjust its per-feature behavior based solely on the data it has seen so far?**

### Why It Matters: From Theory to Production Systems

The significance of this problem spans both theory and practice.

**Theoretical significance.** Online learning is a well-established framework with tight minimax bounds: Zinkevich (2003) showed that online gradient descent with a properly tuned learning rate of $\eta_t = \eta / \sqrt{t}$ achieves regret $R(T) = O(\sqrt{T})$ for convex Lipschitz functions, and this bound is known to be unimprovable in the worst case without additional assumptions (Abernethy et al., 2008). But these standard bounds treat every dimension identically through the global dual norm term $\sum_{t=1}^T \|f'_t(x_t)\|_*^2$. If the gradient vectors happen to be sparse—as they overwhelmingly are in practice—the standard analysis produces loose bounds that fail to capture what a learning algorithm might actually achieve. There is a gap between what minimax theory promises and what the data geometry permits. The paper seeks to close this gap by developing algorithms whose regret bounds depend on the actual sparsity pattern of the observed gradients, rather than worst-case norms.

**Practical significance.** The problem manifests concretely in systems that process text, images, and user behavior data at scale. The paper's experimental section focuses on several such domains:

- **Reuters RCV1 text classification** (§6.1): ~800,000 articles with 0/1 bigram features yielding a vocabulary of approximately 2 million dimensions, yet most documents have fewer than 5,000 non-zero features.
- **ImageNet image ranking** (§6.2): 15,000 noun categories with visterm features producing ~10,000-dimensional sparse vectors, where the algorithm must train a separate ranker for each class.
- **MNIST digit recognition** (§6.3): A kernelized representation with Gaussian kernels over a support set of ~3,000 images, producing a 30,000-dimensional dense problem—a different sparsity structure from text.
- **Census income prediction** (§6.4): Quantized demographic features with interaction terms yielding a 4,001-dimensional 0/1 feature space.

In each of these settings, the data is high-dimensional and feature frequencies span multiple orders of magnitude. A learning algorithm that treats all features equally will demonstrably underperform one that adapts—and the paper's experiments confirm this across all four domains. Beyond accuracy, the ability to automatically tune per-feature step sizes eliminates a hyperparameter tuning burden: instead of grid-searching over learning rates and decay schedules, the algorithm itself computes reasonable per-coordinate rates from accumulated gradient information.

### Where Prior Approaches Fall Short

The paper identifies several specific limitations in prior work.

**1. Standard subgradient methods use fixed, non-adaptive proximal functions.**

In Zinkevich's (2003) projected gradient descent, the proximal function is fixed as $\psi(x) = \frac{1}{2}\|x\|_2^2$, yielding the update $x_{t+1} = \Pi_X(x_t - \eta g_t)$. The regret bound scales with $\sqrt{T \sum_{t=1}^T \|g_t\|_2^2}$ and the step-size $\eta$ must be chosen with knowledge of the time horizon $T$. Each coordinate receives the identical shrinkage applied through the $\ell_2$ projection. There is no mechanism that says "this feature has been updated many times, so its step size should shrink more aggressively than that rarely-seen feature's."

In the regularized dual averaging (RDA) framework of Nesterov (2009) and Xiao (2010), the proximal function is either fixed or scaled uniformly: $\psi_t(x) = \sqrt{t} \, \psi(x)$ or $\psi_t(x) = t \, \psi(x)$ for some base strongly convex function $\psi$. The regret bound (8) from §1.4 becomes:

$$R_\phi(T) \leq \sqrt{T} \psi(x^*) + \frac{1}{2\sqrt{T}} \sum_{t=1}^T \|f'_t(x_t)\|_*^2$$

Again, the dual norm $\|\cdot\|_*$ is defined with respect to the global $\psi$, treating every coordinate identically. This is straightforward to analyze and implement, but it is **oblivious to the characteristic of the data being observed**, as the paper puts it (§1).

**2. Composite mirror descent (FOBOS) shares the same limitation.**

In the composite mirror descent framework (Tseng, 2008; Duchi and Singer, 2009; Duchi et al., 2010), the update is:

$$x_{t+1} = \argmin_{x \in X} \left\{ \eta \langle g_t, x \rangle + \eta \phi(x) + B_{\psi}(x, x_t) \right\}$$

where $B_\psi$ is the Bregman divergence induced by a fixed strongly convex function $\psi$. The regret bound (7) from §1.4 is:

$$R_\phi(T) \leq \frac{1}{\eta} B_\psi(x^*, x_1) + \frac{\eta}{2} \sum_{t=1}^T \|f'_t(x_t)\|_*^2$$

Choosing $\eta \propto 1/\sqrt{T}$ yields $O(\sqrt{T})$ regret. The proximal function $\psi$ can be chosen to reflect domain geometry (e.g., negative entropy for the simplex, $\ell_2$ for Euclidean balls), but once chosen, it remains **static throughout training**. It cannot learn that some coordinates need larger step sizes than others based on the observed gradient history.

**3. Second-order methods like AROW capture correlation but address a different problem.**

The paper places particular emphasis on contrasting with the **confidence-weighted learning** framework of Crammer et al. (2008) and the **Adaptive Regularization of Weights (AROW)** algorithm of Crammer et al. (2009). In AROW, the learner maintains both a mean vector $\mu_t \in \mathbb{R}^d$ and a full covariance matrix $\Sigma_t \in \mathbb{R}^{d \times d}$ representing uncertainty about the weight vector. The update at each step is (reproduced from the paper's §1.4):

$$\beta_t = \frac{1}{\langle z_t, \Sigma_t z_t \rangle + \lambda}, \quad \alpha_t = [1 - y_t \langle z_t, \mu_t \rangle]_+, \quad \mu_{t+1} = \mu_t + \alpha_t \Sigma_t y_t z_t, \quad \Sigma_{t+1} = \Sigma_t - \beta_t \Sigma_t x_t x_t^\top \Sigma_t$$

This is a powerful algorithm that adapts to second-order feature correlations. However, the paper identifies several key differences that motivate ADAGRAD:

- **Mistake bounds versus regret bounds.** AROW's analysis produces mistake bounds that depend on the specific sequence seen, making direct comparison with the online learning regret framework difficult. ADAGRAD, in contrast, is analyzed directly within the established regret minimization framework (§1.4).
- **Root versus inverse of the covariance.** AROW applies $\Sigma_t$, the inverse covariance, in its update. ADAGRAD uses the **root** of the outer product matrix $G_t^{1/2}$ rather than its inverse, a distinction that emerges naturally from the formal analysis using strongly convex proximal functions (§2-4).
- **Generalization to composite objectives.** AROW was designed for binary classification with specific loss functions. ADAGRAD is derived for arbitrary convex composite objectives $f_t(x) + \phi(x)$, where $\phi$ can be any closed convex regularizer—$\ell_1$, $\ell_2$, $\ell_\infty$, mixed-norms—and the domain $X$ can be arbitrary (§5). This generality means ADAGRAD can handle group-sparsity, constrained optimization, and non-standard regularization without separate derivation.
- **Run-time complexity.** When using full (non-diagonal) matrices, ADAGRAD requires computing the matrix square root, which is $O(d^3)$, while AROW's update is $O(d^2)$. The paper acknowledges this as a limitation of the full-matrix variant, which is why the diagonal variant—computable in $O(d)$ time—is emphasized and evaluated experimentally.

**4. Prior adaptive approaches focus on loss functions, not proximal geometry.**

The paper points to several lines of work on adaptive gradient methods that address complementary aspects:

- **Hazan and Kale (2008)** and **Cesa-Bianchi et al. (2007)** developed regret bounds that depend on the variation of the cost functions $f_t$ rather than worst-case Lipschitz constants. These methods adapt the step size based on how much the functions change between rounds.
- **Bartlett et al. (2007)** proposed adapting the global step size $\eta_t$ to handle both strongly convex and weakly convex functions within the same framework.

These approaches adapt the **scalar** step size—the global scale of updates—but still apply that scale uniformly to all coordinates. ADAGRAD's contribution is orthogonal: it addresses the **coordinate-wise** structure of the problem, keeping second-order information about each feature's gradient history independently.

**5. Variable metric subgradient methods existed but without explicit convergence rates.**

The paper acknowledges that the idea of adapting the metric in first-order optimization is not new, tracing back at least to Shor's (1972) space dilation methods and the BFGS quasi-Newton family (Fletcher, 1970). However, this prior work "often assumed that the function to be minimized was differentiable and, to our knowledge, did not consider stochastic, online, or composite optimization" (§1.4). Nedic (2002) studied variable metric subgradient methods in her thesis, but the results apply only when the constraint set $X = \mathbb{R}^d$, and it is "difficult to derive explicit rates of convergence from the results there." More recently, Bordes et al. (2009) proposed SGD-QN, a quasi-Newton stochastic gradient descent procedure similar in spirit to ADAGRAD, but their convergence results require a smooth objective with a positive definite Hessian bounded away from zero—assumptions that fail for non-smooth hinge losses and $\ell_1$-regularized problems. ADAGRAD's analysis makes no differentiability or smoothness assumptions and applies to arbitrary subgradients of convex functions.

**6. No unified framework for adapting proximal functions online.**

This is the key gap the paper fills. Prior to ADAGRAD, the proximal function $\psi$ in online learning was a **design choice** made before seeing any data—you picked $\ell_2$, you picked the entropy, and you lived with it for the entire run. There was no framework or analysis for **modifying $\psi$ online** based on the data seen so far, and no regret guarantees for algorithms that did so. The paper's central methodological innovation is to treat the proximal function itself as an object that can be optimized over time, with formal regret bounds showing that the resulting algorithm is competitive with the best proximal function that could have been chosen in hindsight.

### How This Paper Positions Itself

The paper frames its contribution through a meta-learning lens (§1.4, final paragraph):

> "Our approach differs from previous approaches as it does not focus on a particular loss function or mistake bound. Instead, we view the problem of adapting the proximal function as a meta-learning problem. We then obtain a bound comparable to the bound obtained using the best proximal function chosen in hindsight."

This is central to understanding the paper's ambition. It is not proposing a heuristic for setting per-coordinate learning rates—it is establishing a **theory-grounded principle** for constructing those rates. The adaptive matrices $H_t = \delta I + \text{diag}(G_t)^{1/2}$ (diagonal) and $H_t = \delta I + G_t^{1/2}$ (full) are not arbitrary choices. They emerge as the solution to a specific optimization problem:

$$\min_s \left\{ \sum_{t=1}^T \sum_{i=1}^d \frac{g_{t,i}^2}{s_i} : s \succeq 0, \langle \mathbf{1}, s \rangle \leq c \right\}$$

for the diagonal case (§3), and

$$\min_S \left\{ \sum_{t=1}^T \langle g_t, S^{-1} g_t \rangle : S \succeq 0, \text{tr}(S) \leq c \right\}$$

for the full matrix case (§4). In both cases, the optimal solution involves the root of the accumulated outer product matrix, providing a precise rationale for why ADAGRAD uses $G_t^{1/2}$ rather than $G_t$ or $G_t^{-1}$.

The paper explicitly connects to McMahan and Streeter's (2010) concurrent work, which proposes very similar adaptive algorithms. The two papers were developed independently, and the authors acknowledge that both "shed insights into the problems studied in this paper and complement each other" (§1.4). The key differentiator is that ADAGRAD's analysis builds on the established framework of proximal functions and Bregman divergences (Duchi et al., 2010; Xiao, 2010), which enables generalization to composite objectives with arbitrary regularizers $\phi$—a capability not present in McMahan and Streeter's first-principles derivation. The paper positions this generality as a practical advantage: ADAGRAD can be immediately adapted to $\ell_1$-regularized problems (producing sparse solutions), $\ell_2/\ell_\infty$ group-sparsity, and constrained domains like the simplex or $\ell_1$ ball, with derived algorithms for each case (§5).

Finally, the paper introduces the distinction between **diagonal** and **full-matrix** proximal functions not just as a computational tradeoff but as a theoretical contribution. The diagonal version is practical for very high dimensions ($d \sim 10^6$ for text) and achieves per-coordinate adaptivity. The full-matrix version captures correlations between coordinates—essentially learning a metric in the underlying parameter space—and is shown to achieve bounds of the form $R(T) = O(\|x^*\|_2 \, \text{tr}(G_T^{1/2}))$. The in-depth development of both variants, with rigorous analysis and explicit algorithms, establishes the ADAGRAD family as a principled framework for adaptive online learning rather than a single algorithm.

## 3. Technical Approach

### 3.1 Reader Orientation

ADAGRAD is a **family of online learning and stochastic optimization algorithms** that automatically adjusts the learning rate for each individual feature based on how frequently that feature has appeared in past data. The system solves the problem of uniform step sizes in high-dimensional sparse settings—where some features appear thousands of times while others appear only once—by constructing an **adaptive proximal function** from accumulated gradient history that effectively gives large steps to rare features and small steps to common ones, with formal regret guarantees showing this adaptation is provably competitive with the best possible proximal function chosen in hindsight.

### 3.2 Big-Picture Architecture (Diagram in Words)

The ADAGRAD system has four major components that interact at each round $t$ of online learning:

1. **Base Online Learning Algorithm** — either the primal-dual subgradient method (regularized dual averaging, RDA) or composite mirror descent (FOBOS). This component receives the current gradient $g_t$ and produces the next prediction $x_{t+1}$ by solving a minimization problem that trades off the loss gradient against a regularization term and a proximal term. The choice between RDA and mirror descent affects exactly how the trade-off is formulated, but both share the same adaptive proximal machinery.

2. **Adaptive Proximal Function** $\psi_t(x) = \langle x, H_t x \rangle$ — a time-varying quadratic function defined by a positive semidefinite matrix $H_t$ that serves as a strongly convex regularizer in the update step. This is the core innovation: rather than keeping $\psi$ fixed (as in standard gradient descent) or scaling it uniformly (as in standard RDA), ADAGRAD updates $H_t$ at every round based on all gradients seen so far. The matrix $H_t$ determines both the effective step size per coordinate and the norm used to measure distances in parameter space.

3. **Gradient Accumulation Matrix** $G_t = \sum_{\tau=1}^t g_\tau g_\tau^\top$ — the sum of outer products of all subgradients observed up to round $t$. This matrix captures the correlation structure of the gradient vectors. Its diagonal entries $\sum_{\tau=1}^t g_{\tau,i}^2$ measure how much gradient energy coordinate $i$ has received; its off-diagonal entries capture how coordinates co-vary.

4. **Matrix Root Operation** ($H_t = \delta I + \text{diag}(G_t)^{1/2}$ or $H_t = \delta I + G_t^{1/2}$) — transforms the accumulated gradient outer products into the proximal matrix. In the diagonal variant, this requires computing the square root of each diagonal entry of $G_t$ (which is simply the $\ell_2$ norm of the historical gradients for that coordinate). In the full-matrix variant, this requires computing the matrix square root of $G_t$, an $O(d^3)$ operation. The use of the square root (rather than $G_t$ itself or $G_t^{-1}$) emerges from the regret analysis and is the key design choice that makes the algorithm work.

**Information flow at round $t$**: The learner receives a loss function $f_t$ → it computes the subgradient $g_t \in \partial f_t(x_t)$ at the current prediction $x_t$ → it updates the accumulator $G_t = G_{t-1} + g_t g_t^\top$ → it constructs $H_t = \delta I + (\text{diag}(G_t))^{1/2}$ (diagonal) or $H_t = \delta I + G_t^{1/2}$ (full) → it defines the proximal function $\psi_t(x) = \frac{1}{2} \langle x, H_t x \rangle$ → it solves the update step (either RDA or mirror descent) to produce $x_{t+1}$ → the process repeats.

### 3.3 Roadmap for the Deep Dive

- **First**, the formal regret minimization framework and the two base update rules (RDA and composite mirror descent), because everything that follows builds on understanding what these updates optimize and where the proximal function enters the regret bound.
- **Second**, the core adaptive proximal function mechanism—why we want to modify $\psi_t$ online, what optimization problem the choice of $H_t$ solves, and the Lemma 4 doubling trick that bounds the sum of gradient-weighted norms.
- **Third**, the diagonal matrix variant in full detail, starting from the motivating optimization problem, through the algorithm pseudocode, to the complete regret bound proof and its simplified corollaries, because this is the practically important version and the analysis for the full-matrix case builds on the same structure.
- **Fourth**, the full-matrix variant, which follows the same logical arc but requires additional technical lemmas (Lemmas 8 and 9) to handle the interaction between matrix roots and the regret decomposition.
- **Fifth**, the explicit update derivations for common regularizers ($\ell_1$, $\ell_1$-ball, $\ell_2$, $\ell_\infty$, mixed-norms), showing how the general framework translates into concrete implementable algorithms with closed-form or efficiently computable solutions.
- **Finally**, the connection between the two update families (RDA vs. mirror descent) and the time-dependent strong convexity assumptions that make the analysis work, so the reader sees the unifying structure beneath what look like different algorithms.

### 3.4 Detailed, Sentence-Based Technical Breakdown

This is fundamentally a **theoretical analysis paper with algorithmic consequences**. The core idea is that the proximal function $\psi$ in online learning algorithms—which standard theory treats as a fixed design choice—can itself be **optimized online** based on observed gradients, and that setting $\psi_t$ to involve $G_t^{1/2}$ yields regret bounds that automatically adapt to the sparsity and geometry of the data, matching the performance of the best proximal function that could have been chosen with full knowledge of all gradients in advance.

---

#### The Online Learning Framework and Two Base Update Rules

The paper operates within the standard online convex optimization framework, but with an important generalization: the functions presented to the learner are **composite**, meaning each round's loss decomposes as $\varphi_t(x) = f_t(x) + \phi(x)$, where $f_t$ is the instantaneous loss (e.g., hinge loss on a single example) and $\phi$ is a fixed, time-invariant regularizer (e.g., $\ell_1$ penalty to encourage sparsity). The learner's regret is measured against a fixed comparator $x^*$ as:

$$R_\phi(T) \triangleq \sum_{t=1}^T [\varphi_t(x_t) - \varphi_t(x^*)] = \sum_{t=1}^T [f_t(x_t) + \phi(x_t) - f_t(x^*) - \phi(x^*)]$$

where $R_\phi(T)$ is the composite regret after $T$ rounds, $x_t$ is the prediction made at round $t$, and $x^*$ is the optimal fixed predictor in hindsight.

**What it computes:** the cumulative excess loss suffered by the online learner compared to the best fixed choice of parameters $x^*$ over all $T$ rounds, including the regularizer $\phi$. The goal is to design algorithms where this regret grows sublinearly ($o(T)$), meaning the average per-round excess loss goes to zero.

**Why this form:** separating $f_t$ from $\phi$ allows the analysis to handle both the time-varying data-dependent loss and the fixed structural penalty in a unified way. The regularizer $\phi$ is known in advance and can be used in the update step to enforce structural properties (sparsity, group sparsity, bounded norms), while $f_t$ is revealed only after the prediction is made. This decomposition is crucial for the derived algorithms in Section 5, where different choices of $\phi$ lead to different closed-form update rules.

The paper considers two families of update rules for producing the sequence $\{x_t\}$. The first is **Nesterov's primal-dual subgradient method**, extended by Xiao (2010) as regularized dual averaging (RDA):

$$x_{t+1} = \argmin_{x \in X} \left\{ \eta \langle \bar{g}_t, x \rangle + \eta \phi(x) + \frac{1}{t} \psi_t(x) \right\}$$

where $\eta > 0$ is a fixed step-size parameter, $\bar{g}_t = \frac{1}{t} \sum_{\tau=1}^t g_\tau$ is the running average of all subgradients seen so far, $X \subseteq \mathbb{R}^d$ is the constraint set, and $\psi_t$ is the time-varying proximal function. The initial prediction is $x_1 = \argmin_{x \in X} \phi(x)$.

**What it computes:** each round, the algorithm finds the point $x$ that minimizes a weighted sum of three terms: (1) the inner product with the average gradient $\bar{g}_t$ (which pulls $x$ in the direction of negative average gradient), (2) the regularizer $\phi(x)$ (which imposes structural penalty), and (3) the proximal function $\psi_t(x)$ scaled by $1/t$ (which keeps $x$ from moving too far from regions where previous predictions were stable). The tradeoff parameter $\eta$ controls the balance between the gradient-driven and regularizer-driven components.

**Why this form:** the dual averaging update is "lazy"—it uses the average gradient from all history rather than the instantaneous gradient. This means the influence of each individual gradient diminishes as $1/t$, creating a natural averaging effect. The proximal term $\psi_t(x)/t$ shrinks relative to the data terms as $t$ grows, analogous to how step sizes decay in standard SGD. The key difference from prior work is that $\psi_t$ is allowed to depend on time $t$ and, critically, on the observed gradients through $H_t$.

The second family is **composite mirror descent** (also known as forward-backward splitting or FOBOS, Duchi and Singer, 2009), with update:

$$x_{t+1} = \argmin_{x \in X} \left\{ \eta \langle g_t, x \rangle + \eta \phi(x) + B_{\psi_t}(x, x_t) \right\}$$

where $B_{\psi_t}(x, x_t) = \psi_t(x) - \psi_t(x_t) - \langle \nabla \psi_t(x_t), x - x_t \rangle$ is the Bregman divergence associated with the strictly convex function $\psi_t$, measuring the distance between $x$ and the previous prediction $x_t$ in the geometry induced by $\psi_t$.

**What it computes:** each round, the algorithm takes a gradient step from $x_t$ in the direction of $-\eta g_t$, but instead of using Euclidean projection, it uses the Bregman divergence $B_{\psi_t}$ to measure how far the new point $x_{t+1}$ is from $x_t$. The Bregman divergence acts as a "proximity penalty"—it encourages $x_{t+1}$ to stay close to $x_t$ in the geometry defined by $\psi_t$, while the linear term encourages movement opposite the gradient.

**Why this form:** composite mirror descent is "greedy"—it responds to the most recent gradient immediately. The Bregman divergence provides a principled way to incorporate the geometry of the problem: if $\psi_t$ is chosen to match the structure of the gradients or the constraint set, the updates can be more efficient than Euclidean projections. As with RDA, the critical innovation is that $\psi_t$ can change each round based on accumulated data, whereas in standard mirror descent $\psi_t \equiv \psi$ is fixed.

Both algorithms come with regret bounds that reveal precisely where $\psi_t$ enters the analysis. Proposition 2 provides the bound for RDA:

$$\sum_{t=1}^T f_t(x_t) + \phi(x_t) - f_t(x^*) - \phi(x^*) \leq \frac{1}{\eta} \psi_T(x^*) + \frac{\eta}{2} \sum_{t=1}^T \|f'_t(x_t)\|^2_{\psi^*_{t-1}}$$

where $\|\cdot\|_{\psi^*_{t-1}}$ is the dual norm induced by the proximal function at the previous round (specifically, $\|g\|^2_{\psi^*_{t-1}} = \langle g, (\nabla^2 \psi_{t-1})^{-1} g \rangle$ for the case where $\psi$ is a quadratic).

**What this bound says:** the regret decomposes into two terms. The first, $\psi_T(x^*)/\eta$, measures how "large" the optimal predictor $x^*$ is under the final proximal function—if $x^*$ is sparse or has small norm under $\psi_T$, this term is small. The second term sums the squared dual norms of the subgradients, where each subgradient is measured under the proximal function from the *previous* round. This is where the adaptation happens: if $\psi_{t-1}$ has been shaped by earlier gradients to be large in directions where $g_t$ is small and small in directions where $g_t$ is large, the dual norm $\|g_t\|^2_{\psi^*_{t-1}}$ shrinks.

Proposition 3 provides the analogous bound for composite mirror descent:

$$\sum_{t=1}^T f_t(x_t) + \phi(x_t) - f_t(x^*) - \phi(x^*) \leq \frac{1}{\eta} B_{\psi_1}(x^*, x_1) + \frac{1}{\eta} \sum_{t=1}^{T-1} \left[ B_{\psi_{t+1}}(x^*, x_{t+1}) - B_{\psi_t}(x^*, x_{t+1}) \right] + \frac{\eta}{2} \sum_{t=1}^T \|f'_t(x_t)\|^2_{\psi^*_t}$$

**What this bound says:** there is an additional penalty compared to the RDA bound—a sum of differences between successive Bregman divergences evaluated at $x^*$ and $x_{t+1}$. This term measures how much the proximal function changes between rounds: if $\psi_{t+1}$ is "larger" than $\psi_t$ (in the sense that it induces larger distances), the difference $B_{\psi_{t+1}}(x^*, x_{t+1}) - B_{\psi_t}(x^*, x_{t+1})$ is positive and adds to the regret. This creates a tension: making $\psi_t$ grow faster shrinks the dual norm terms (good) but increases the Bregman divergence penalty (bad). The analysis must show that the net effect is beneficial.

These two regret bounds are the scaffolding on which the entire adaptive proximal function analysis is built. The rest of the paper's technical contribution is about choosing the sequence $\{\psi_t\}$ so that the combined terms are as small as possible, provably competitive with any fixed choice of $\psi$.

---

#### The Adaptive Proximal Function: Why $G_t^{1/2}$?

The paper's key insight starts from an optimization problem. Consider that after seeing all $T$ subgradients, we want to choose a positive semidefinite matrix $H$ (defining $\psi(x) = \frac{1}{2} \langle x, H x \rangle$) that would have minimized the sum of squared dual norms $\sum_{t=1}^T \langle g_t, H^{-1} g_t \rangle$ subject to a trace constraint. This is the "choose the best proximal function in hindsight" problem. For the diagonal case (where $H$ is restricted to be diagonal, $H = \text{diag}(s)$ for $s \succeq 0$), the optimization is:

$$\min_s \sum_{t=1}^T \sum_{i=1}^d \frac{g_{t,i}^2}{s_i} \quad \text{subject to} \quad s \succeq 0, \langle \mathbf{1}, s \rangle \leq c$$

where $g_{t,i}$ is the $i$-th component of the subgradient at round $t$, $s_i$ is the $i$-th diagonal entry of $H$, $c > 0$ is a budget on the sum of $s_i$, and $\mathbf{1}$ is the all-ones vector.

**What this problem computes:** find per-coordinate scalings $s_i$ that minimize the sum of weighted gradient norms, where each coordinate's gradient is divided by $s_i$. Making $s_i$ large shrinks the contribution of coordinate $i$'s gradients to the objective. The constraint $\sum_i s_i \leq c$ prevents the trivial solution of setting all $s_i \to \infty$—there is a total budget on how much "scaling capacity" we can allocate across coordinates.

**Why this form:** the sum-of-ratios objective $\sum_{t,i} g_{t,i}^2 / s_i$ directly penalizes large gradients along coordinates with small $s_i$. To minimize this, we should allocate the budget $c$ to coordinates where $\sum_t g_{t,i}^2$ (the total gradient energy) is largest. The optimal solution, by solving the KKT conditions, is $s_i \propto \sqrt{\sum_{t=1}^T g_{t,i}^2} = \|g_{1:T,i}\|_2$, with the proportionality constant chosen to satisfy the budget constraint exactly. This yields the closed form shown in Equation (12):

$$\inf_s \left\{ \sum_{t=1}^T \sum_{i=1}^d \frac{g_{t,i}^2}{s_i} : s \succeq 0, \langle \mathbf{1}, s \rangle \leq c \right\} = \frac{1}{c} \left( \sum_{i=1}^d \|g_{1:T,i}\|_2 \right)^2$$

**Why the square root matters:** the optimal $s_i$ depends on the $\ell_2$ norm of the historical gradients for coordinate $i$, not the squared norm. If coordinate $i$ has accumulated gradient energy $\sum_t g_{t,i}^2 = 100$, the optimal $s_i$ is proportional to $\sqrt{100} = 10$, not to $100$. This square root scaling is fundamentally different from simply using the empirical variance or covariance matrix directly. Intuitively, using the square root compresses the dynamic range: a coordinate with 100 times more gradient energy gets only 10 times more scaling, providing meaningful but not overwhelming differentiation. If we used $G_t$ directly (no square root), common features would have their step sizes shrunk quadratically faster than rare features, potentially freezing them out entirely.

For the full matrix case (where $H$ is not restricted to be diagonal), the "best in hindsight" problem is:

$$\min_S \sum_{t=1}^T \langle g_t, S^{-1} g_t \rangle \quad \text{subject to} \quad S \succeq 0, \text{tr}(S) \leq c$$

where $S \in \mathbb{R}^{d \times d}$ is a full matrix and $\text{tr}(S)$ is its trace.

**What this computes:** the same objective generalized to allow correlations between coordinates. The constraint bounds the sum of eigenvalues of $S$ (its trace). The optimal solution is $S = c \, G_T^{1/2} / \text{tr}(G_T^{1/2})$ where $G_T = \sum_{t=1}^T g_t g_t^\top$, producing an objective value of $\text{tr}(G_T^{1/2})^2 / c$ (Lemma 15 in the appendix). Again, the matrix square root appears naturally as the optimal solution.

**Why this form for the full matrix:** the objective $\langle g_t, S^{-1} g_t \rangle$ measures the Mahalanobis norm of $g_t$ induced by $S^{-1}$. If $S$ has large eigenvalues in directions where gradients frequently point, $S^{-1}$ has small eigenvalues in those directions, and the norm shrinks. The optimal $S$ therefore aligns its principal components with the principal components of $G_T$—it "learns the metric" of the gradient space. The square root ensures the eigenvalues of $S$ scale as the square roots of the corresponding eigenvalues of $G_T$, again compressing the dynamic range.

The critical observation is that while we cannot use the batch-optimal $S$ during the online process (it requires knowing all $T$ gradients in advance), we can **incrementally approximate it**: at round $t$, use $H_t = \delta I + G_t^{1/2}$ (diagonal or full), where $G_t$ only uses gradients $\{g_1, \ldots, g_t\}$ seen so far. The remainder of the analysis is devoted to proving that this incremental approximation preserves the essential properties of the batch-optimal solution—specifically, the sum of the resulting dual norms is within a factor of 2 of the infimal value.

---

#### Lemma 4: The Doubling Trick for Diagonal Proximal Functions

The technical bridge between the "best in hindsight" optimization and the actual online regret involves Lemma 4, which bounds the sum of the gradient-weighted norms that appear in the regret decompositions:

$$\sum_{t=1}^T \left\langle g_t, \text{diag}(s_t)^{-1} g_t \right\rangle \leq 2 \sum_{i=1}^d \|g_{1:T,i}\|_2$$

where $s_{t,i} = \|g_{1:t,i}\|_2$ is the $\ell_2$ norm of the sequence of gradients for coordinate $i$ up to time $t$, $\text{diag}(s_t)$ is the diagonal matrix with entries $s_{t,i}$, and $g_t \in \partial f_t(x_t)$ is the subgradient at round $t$.

**What this inequality says:** if we use the running $\ell_2$ norms $s_{t,i}$ as the diagonal entries of our proximal matrix at each round, the sum of the quadratic forms $\langle g_t, \text{diag}(s_t)^{-1} g_t \rangle = \sum_i g_{t,i}^2 / \|g_{1:t,i}\|_2$ across all rounds is bounded by twice the sum of the *final* $\ell_2$ norms across coordinates. Note that the right-hand side involves the norms of the complete gradient histories $\|g_{1:T,i}\|_2$, while the left-hand side uses the *running* norms $\|g_{1:t,i}\|_2$ at each intermediate round.

**Why this works (proof structure):** the lemma is proved by induction on $T$ using concavity of the square root. For a single coordinate with scalar values $a_1, \ldots, a_T$, the claim reduces to showing:

$$\sum_{t=1}^T \frac{a_t^2}{\sqrt{\sum_{\tau=1}^t a_\tau^2}} \leq 2 \sqrt{\sum_{t=1}^T a_t^2}$$

The base case $T = 1$ is $a_1^2/|a_1| = |a_1| \leq 2|a_1|$, which holds. For the inductive step, assume the inequality holds for $T-1$. Let $b_T = \sum_{t=1}^T a_t^2$. Then:

$$\sum_{t=1}^T \frac{a_t^2}{\sqrt{\sum_{\tau=1}^t a_\tau^2}} \leq 2\sqrt{b_{T-1}} + \frac{a_T^2}{\sqrt{b_T}}$$

The concave inequality $\sqrt{b_T - a_T^2} \leq \sqrt{b_T} - \frac{a_T^2}{2\sqrt{b_T}}$ (valid when $b_T \geq a_T^2$) implies:

$$2\sqrt{b_T - a_T^2} + \frac{a_T^2}{\sqrt{b_T}} \leq 2\sqrt{b_T}$$

which completes the induction. The key property is concavity of the square root, which creates a "law of diminishing returns": each new gradient's contribution to the running sum is proportionally smaller than it would be if we used a linear accumulation.

**Why this factor of 2 is tolerable:** the constant factor 2 means that using the incremental approximation $s_t$ (constructed only from past gradients) gives at most twice the cost of using the batch-optimal $s_T$ (constructed from all gradients). In the regret bound, this factor of 2 is absorbed into the constants and does not affect the asymptotic rate.

This lemma is the workhorse of the diagonal analysis. It shows that even though we cannot use $s_T$ at early rounds (since it depends on future gradients), using the running approximation $s_t$ is sufficient—the "mistakes" made by having incomplete information at early rounds contribute at most a factor of 2 overhead.

---

#### Diagonal ADAGRAD: Algorithm and Regret Analysis

The diagonal version of ADAGRAD is presented in Algorithm 1 (Figure 1 in the paper) and is the variant used in all experiments. The algorithm maintains a vector $s \in \mathbb{R}^d$ where $s_{t,i} = \|g_{1:t,i}\|_2$—the $\ell_2$ norm of the first $t$ subgradients' $i$-th components. At each round $t$:

1. The learner receives the loss function $f_t$ and incurs loss $f_t(x_t)$.
2. The subgradient $g_t \in \partial f_t(x_t)$ is computed.
3. The running gradient history is updated: for each coordinate $i$, $s_{t,i} = \|g_{1:t,i}\|_2$.
4. The proximal matrix is set to $H_t = \delta I + \text{diag}(s_t)$, where $\delta \geq 0$ is a small constant ensuring positive definiteness (in practice, $\delta$ can be 0 if handling of zero entries is done carefully).
5. The proximal function is $\psi_t(x) = \frac{1}{2} \langle x, H_t x \rangle$.
6. Depending on the chosen update family, the next prediction $x_{t+1}$ is computed using either the RDA update (3) or the composite mirror descent update (4).

The initial conditions are $x_1 = 0$ and $g_{1:0} = []$ (empty history). The variables $g_{1:t,i} \in \mathbb{R}^t$ stored in the algorithm are not necessarily kept in memory—only the norms $s_{t,i}$ need to be tracked, requiring $O(d)$ storage.

**Why the update step uses $H_t$ specifically:** in the RDA update (3), the optimization becomes:

$$x_{t+1} = \argmin_{x \in X} \left\{ \eta \left\langle \frac{1}{t} \sum_{\tau=1}^t g_\tau, x \right\rangle + \eta \phi(x) + \frac{1}{t} \cdot \frac{1}{2} \langle x, (\delta I + \text{diag}(s_t)) x \rangle \right\}$$

The effective step size for coordinate $i$ is proportional to $\eta t / (\delta + \|g_{1:t,i}\|_2)$. For a feature that has accumulated large gradient norms (appearing frequently), the denominator is large, and the effective step size is small—the learner "knows" this coordinate well and makes only fine adjustments. For a feature with small accumulated norm (rare), the denominator is near $\delta$, and the step size scales as $\eta t$—the learner makes aggressive updates to incorporate the new information.

In composite mirror descent (4), the effective per-coordinate step size is $\eta / (\delta + \|g_{1:t,i}\|_2)$, and the Bregman divergence penalty $\frac{1}{2} \langle x - x_t, H_t (x - x_t) \rangle$ penalizes movement more in directions where $H_t$ has large entries—again, common features change slowly, rare features change quickly.

**Regret bound for diagonal ADAGRAD with RDA (Theorem 5, first part).** Using the primal-dual subgradient update with $\delta \geq \max_t \|g_t\|_\infty$, for any comparator $x^* \in X$:

$$R_\phi(T) \leq \frac{\delta}{\eta} \|x^*\|_2^2 + \frac{1}{\eta} \|x^*\|_\infty^2 \sum_{i=1}^d \|g_{1:T,i}\|_2 + \eta \sum_{i=1}^d \|g_{1:T,i}\|_2$$

where $\|x^*\|_\infty = \max_i |x^*_i|$ is the $\ell_\infty$ norm of the optimal predictor.

**Where each term comes from:** The first term $\frac{\delta}{\eta} \|x^*\|_2^2$ comes from the $\delta I$ component of $H_t$ in $\psi_T(x^*)$—it is a small constant term if $\delta$ is small. The second term comes from the diagonal part of $\psi_T(x^*)$: since $\psi_T(x^*) = \frac{1}{2} \langle x^*, (\delta I + \text{diag}(s_T)) x^* \rangle$, we have $\langle x^*, \text{diag}(s_T) x^* \rangle = \sum_i x_i^{*2} \|g_{1:T,i}\|_2 \leq \|x^*\|_\infty^2 \sum_i \|g_{1:T,i}\|_2$. The third term comes from Lemma 4 applied to the dual norm sum $\sum_t \|g_t\|^2_{\psi^*_{t-1}}$, bounded by $2 \sum_i \|g_{1:T,i}\|_2$, with the factor of 2 absorbed into the $\eta/2$ coefficient from Proposition 2, and further bounded by an additional margin.

**Regret bound for diagonal ADAGRAD with composite mirror descent (Theorem 5, second part).** For any $x^* \in X$:

$$R_\phi(T) \leq \frac{1}{2\eta} \max_{t \leq T} \|x^* - x_t\|_\infty^2 \sum_{i=1}^d \|g_{1:T,i}\|_2 + \eta \sum_{i=1}^d \|g_{1:T,i}\|_2$$

where the additional Bregman divergence penalty from Proposition 3 is bounded by analyzing the telescoping sum of differences. Specifically, the paper shows (Equation 14) that:

$$\sum_{t=1}^{T-1} [B_{\psi_{t+1}}(x^*, x_{t+1}) - B_{\psi_t}(x^*, x_{t+1})] \leq \frac{1}{2} \max_{t \leq T} \|x^* - x_t\|_\infty^2 \sum_{i=1}^d \|g_{1:T,i}\|_2$$

since $B_{\psi_{t+1}}(x^*, x_{t+1}) - B_{\psi_t}(x^*, x_{t+1}) = \frac{1}{2} \langle x^* - x_{t+1}, \text{diag}(s_{t+1} - s_t)(x^* - x_{t+1}) \rangle \leq \frac{1}{2} \max_i (x^*_i - x_{t+1,i})^2 \|s_{t+1} - s_t\|_1$, and the sum over $t$ of $\|s_{t+1} - s_t\|_1 = \langle s_{t+1} - s_t, \mathbf{1} \rangle$ telescopes to $\langle s_T, \mathbf{1} \rangle = \sum_i \|g_{1:T,i}\|_2$.

**Corollary 6: Simplified form.** Assuming the domain $X$ is compact with $D_\infty = \sup_{x \in X} \|x - x^*\|_\infty$, and defining $\gamma_T \triangleq \sum_{i=1}^d \|g_{1:T,i}\|_2$, setting the step size appropriately yields:

- For RDA with $\eta = \|x^*\|_\infty$: $R_\phi(T) \leq 2 \|x^*\|_\infty \gamma_T + \delta \|x^*\|_1$
- For composite mirror descent with $\eta = D_\infty / \sqrt{2}$: $R_\phi(T) \leq \sqrt{2} D_\infty \gamma_T$

**What this means practically:** the regret scales with $\gamma_T = \sum_i \|g_{1:T,i}\|_2$ rather than $\sqrt{T \sum_t \|g_t\|_2^2}$ (the standard Zinkevich bound). For sparse gradient sequences where most coordinates have zero or small cumulative norms, $\gamma_T$ can be much smaller than the standard bound. For example, if only $k \ll d$ coordinates ever have non-zero gradients, and each of those coordinates has total gradient energy $\sqrt{T}$, then $\gamma_T \approx k \sqrt{T}$ while the standard bound is $\sqrt{d T}$—a factor of $\sqrt{d/k}$ improvement.

The corollary also shows an important connection: $\gamma_T$ can be written as an infimum over positive diagonal scalings (Equation 12):

$$\gamma_T = \sqrt{d \cdot \inf_s \left\{ \sum_{t=1}^T \langle g_t, \text{diag}(s)^{-1} g_t \rangle : s \succeq 0, \langle \mathbf{1}, s \rangle \leq d \right\}}$$

**Why this representation matters:** it explicitly shows that ADAGRAD's regret bound is competitive with the best diagonal proximal matrix that could have been chosen in hindsight. The factor $\sqrt{d}$ appears because the constraint $\langle \mathbf{1}, s \rangle \leq d$ normalizes to a per-coordinate average budget. McMahan and Streeter (2010) show that if the domain $X$ is contained in an $\ell_\infty$ ball of radius $R$ and contains an $\ell_\infty$ ball of radius $r$, the bound in Corollary 6 is within a factor of $\sqrt{2} R/r$ of the optimal diagonal proximal matrix in hindsight—so for "balanced" domains where $R/r$ is small (e.g., $X = \{x: \|x\|_\infty \leq 1\}$ giving $R/r = 1$), the adaptation is nearly optimal.

---

#### Full-Matrix ADAGRAD: Algorithm and Regret Analysis

The full-matrix version of ADAGRAD is presented in Algorithm 2 (Figure 2 in the paper) and generalizes the diagonal approach to capture correlations between coordinates. Maintaining full $d \times d$ matrices is computationally expensive ($O(d^2)$ storage, $O(d^3)$ for the matrix square root), so this variant is intended primarily for moderate-dimensional problems where feature correlations are important.

The algorithm maintains:
- $G_t = \sum_{\tau=1}^t g_\tau g_\tau^\top \in \mathbb{R}^{d \times d}$, the outer product matrix of all subgradients so far.
- $S_t = G_t^{1/2}$, the matrix square root of $G_t$.
- $H_t = \delta I + S_t$, the proximal matrix.
- $\psi_t(x) = \frac{1}{2} \langle x, H_t x \rangle$.

At each round, the update is identical in structure to the diagonal case: receive loss, compute subgradient, update $G_t = G_{t-1} + g_t g_t^\top$, compute the matrix square root $S_t = G_t^{1/2}$, set $H_t = \delta I + S_t$, and then use either the RDA or mirror descent update to produce $x_{t+1}$.

**Why the matrix square root is used rather than the matrix itself:** the same "best in hindsight" optimization (15) shows that the optimal $S$ for minimizing $\sum_{t=1}^T \langle g_t, S^{-1} g_t \rangle$ subject to $\text{tr}(S) \leq c$ is proportional to $G_T^{1/2}$, not to $G_T$ itself or its inverse. Using $G_T$ directly would produce a proximal function that scales quadratically with gradient magnitude in each eigendirection, over-penalizing frequently seen directions. Using $G_T^{-1}$ would reward rather than penalize movement in high-gradient directions, which is destabilizing.

**Technical lemmas for the full-matrix analysis.** Two additional lemmas are required that have no diagonal analog because they involve non-commuting matrices.

**Lemma 8 (Matrix concavity bound).** Let $B \succeq 0$ and $B^{-1/2}$ denote the root of the inverse (or pseudo-inverse if $B$ is singular). For any vector $g$ and scalar $\nu \geq 0$ such that $B - \nu g g^\top \succeq 0$:

$$2 \text{tr}\left((B - \nu g g^\top)^{1/2}\right) \leq 2 \text{tr}(B^{1/2}) - \nu \text{tr}\left(B^{-1/2} g g^\top\right)$$

**What this inequality says:** taking the trace of the matrix square root is concave in the matrix argument. If we remove a rank-1 positive semidefinite matrix $\nu g g^\top$ from $B$, the trace of the root decreases by at least $\nu$ times the quadratic form $\langle g, B^{-1/2} g \rangle$ (which equals $\text{tr}(B^{-1/2} g g^\top)$). This is the matrix analog of the scalar concavity inequality $\sqrt{b - a^2} \leq \sqrt{b} - a^2 / (2\sqrt{b})$ used in Lemma 4.

**Why this lemma is necessary:** the full-matrix doubling trick (Lemma 10 below) requires a matrix version of the concavity argument. The proof (Appendix D) uses the fact that $A \mapsto \text{tr}(A^p)$ is concave for $0 \leq p \leq 1$ (Ando, 1979) and applies the first-order concavity inequality with the gradient $\nabla \text{tr}(A^{1/2}) = \frac{1}{2} A^{-1/2}$ (Lemma 14). Care must be taken when $B$ is singular—the limiting argument with $B + \delta I$ is used.

**Lemma 9 (Smoothing bound).** Let $\delta \geq \|g\|_2$ and $A \succeq 0$. Then:

$$\left\langle g, (\delta I + A^{1/2})^{-1} g \right\rangle \leq \left\langle g, \left((A + g g^\top)^\dagger\right)^{1/2} g \right\rangle$$

where $M^\dagger$ denotes the Moore-Penrose pseudo-inverse.

**What this inequality says:** the quadratic form with the smoothed inverse $(\delta I + A^{1/2})^{-1}$ is bounded above by the quadratic form with the pseudo-inverse root of the updated matrix $A + g g^\top$. This is needed because in the RDA regret bound, the dual norm at round $t$ involves $\psi_{t-1}^*$—the proximal function from the *previous* round—while the ideal bound would involve $\psi_t^*$ (which includes the most recent gradient $g_t$). Lemma 9 bridges this gap: it shows that using the "stale" proximal function $\psi_{t-1}$ with a sufficiently large $\delta$ is at least as good as using the "fresh" $\psi_t$ without $\delta$.

**Why this is necessary specifically for the RDA variant:** the RDA regret bound (Proposition 2) has dual norms $\|g_t\|^2_{\psi^*_{t-1}}$ using the previous round's proximal function, while the mirror descent bound (Proposition 3) uses $\|g_t\|^2_{\psi^*_t}$ at the current round. For the mirror descent case, the doubling lemma (Lemma 10) can be applied directly. For RDA, we need Lemma 9 to relate the "stale" norm to the "fresh" one, which requires the condition $\delta \geq \max_t \|g_t\|_2$—essentially, the $\delta I$ term must dominate the difference between $G_{t-1}^{1/2}$ and $G_t^{1/2}$.

**Lemma 10 (Full-matrix doubling trick).** Let $S_t = G_t^{1/2}$ and let $S_t^\dagger$ denote the pseudo-inverse. Then:

$$\sum_{t=1}^T \left\langle g_t, S_t^\dagger g_t \right\rangle \leq 2 \sum_{t=1}^T \left\langle g_t, S_T^\dagger g_t \right\rangle = 2 \text{tr}(G_T^{1/2})$$

**What this inequality says:** the sum of the quadratic forms using the running matrix root $S_t^\dagger$ is bounded by twice the sum using the *final* matrix root $S_T^\dagger$, and the latter sum equals the trace of $G_T^{1/2}$. This is the direct analog of Lemma 4 for the full-matrix case.

**Why the trace appears:** $\sum_{t=1}^T \langle g_t, S_T^\dagger g_t \rangle = \sum_{t=1}^T \text{tr}(S_T^\dagger g_t g_t^\top) = \text{tr}(S_T^\dagger \sum_{t=1}^T g_t g_t^\top) = \text{tr}(S_T^\dagger G_T)$. Since $S_T = G_T^{1/2}$, we have $S_T^\dagger G_T = (G_T^{1/2})^\dagger G_T$. When $G_T$ is full rank, this equals $G_T^{1/2}$, whose trace is $\text{tr}(G_T^{1/2})$. When $G_T$ is singular, the pseudo-inverse gives the same trace on the range of $G_T$. The proof uses Lemma 8 with the substitution $B = G_T = G_{T-1} + g_T g_T^\top$, $\nu = 1$, and induction on $T$.

**Regret bound for full-matrix ADAGRAD with RDA (Theorem 7, first part).** With $\delta \geq \max_t \|g_t\|_2$, for any $x^* \in X$:

$$R_\phi(T) \leq \frac{\delta}{\eta} \|x^*\|_2^2 + \frac{1}{\eta} \|x^*\|_2^2 \text{tr}(G_T^{1/2}) + \eta \text{tr}(G_T^{1/2})$$

**Regret bound for full-matrix ADAGRAD with composite mirror descent (Theorem 7, second part).** For any $x^* \in X$ and $\delta \geq 0$:

$$R_\phi(T) \leq \frac{\delta}{\eta} \|x^*\|_2^2 + \frac{1}{2\eta} \max_{t \leq T} \|x^* - x_t\|_2^2 \text{tr}(G_T^{1/2}) + \eta \text{tr}(G_T^{1/2})$$

The key difference from the diagonal bounds is that $\text{tr}(G_T^{1/2})$ replaces $\sum_i \|g_{1:T,i}\|_2$, and the geometry is measured in $\ell_2$ rather than $\ell_\infty$ (since the full matrix captures correlated structure that cannot be expressed as per-coordinate norms).

**Corollary 11: Simplified form and connection to optimal metric.** For RDA with $\eta = \|x^*\|_2$: $R_\phi(T) \leq 2 \|x^*\|_2 \text{tr}(G_T^{1/2}) + \delta \|x^*\|_2$. For mirror descent with $\eta = D/\sqrt{2}$ (where $D = \sup_{x \in X} \|x - x^*\|_2$):

$$R_\phi(T) \leq \sqrt{2} D \text{tr}(G_T^{1/2}) = \sqrt{2d} D \sqrt{\inf_S \left\{ \sum_{t=1}^T g_t^\top S^{-1} g_t : S \succeq 0, \text{tr}(S) \leq d \right\}}$$

**What this means:** the regret is competitive with the best full matrix proximal function chosen in hindsight. The factor $\sqrt{d}$ appears because the constraint $\text{tr}(S) \leq d$ normalizes to an average per-dimension budget. For gradient sequences that lie approximately in a low-dimensional subspace, $G_T$ will have rank $r \ll d$, and $\text{tr}(G_T^{1/2})$ will scale as $\sqrt{r}$ times the average gradient norm rather than $\sqrt{d}$—the algorithm automatically exploits this structure.

---

#### Why the Bregman Divergence Penalty Does Not Overwhelm the Gains

A potential concern with the analysis, particularly for composite mirror descent, is that changing the proximal function $\psi_t$ at each round adds a penalty $\sum_t [B_{\psi_{t+1}}(x^*, x_{t+1}) - B_{\psi_t}(x^*, x_{t+1})]$ to the regret (Proposition 3). If $\psi_t$ grows too aggressively, this penalty could dominate and wipe out the benefits of reduced dual norms. The analysis shows this does not happen because the growth of $\psi_t$ is controlled.

For the diagonal case, the sum of differences telescopes (Equation 14):

$$B_{\psi_{t+1}}(x^*, x_{t+1}) - B_{\psi_t}(x^*, x_{t+1}) = \frac{1}{2} \langle x^* - x_{t+1}, \text{diag}(s_{t+1} - s_t)(x^* - x_{t+1}) \rangle$$

and $\sum_t (s_{t+1} - s_t) = s_T - s_1$, so the total penalty is bounded by $\frac{1}{2} \max_t \|x^* - x_t\|_\infty^2 \sum_i \|g_{1:T,i}\|_2$—the exact same quantity (up to constants) that appears in the $\psi_T(x^*)$ term of the RDA bound. In other words, the Bregman penalty for changing $\psi$ is of the same order as simply using the final $\psi_T$ directly (as RDA does). The adaptation does not introduce asymptotic overhead.

For the full-matrix case, the analogous telescoping uses the trace property (Equation 16):

$$B_{\psi_{t+1}}(x^*, x_{t+1}) - B_{\psi_t}(x^*, x_{t+1}) \leq \frac{1}{2} \|x^* - x_{t+1}\|_2^2 \text{tr}(G_{t+1}^{1/2} - G_t^{1/2})$$

and $\sum_t \text{tr}(G_{t+1}^{1/2} - G_t^{1/2}) = \text{tr}(G_T^{1/2})$ by telescoping. The penalty again matches the RDA $\psi_T(x^*)$ term, confirming that the adaptation cost is proportional to the final proximal function's magnitude.

**Why this matters:** it means both update families (RDA and mirror descent) enjoy essentially the same asymptotic guarantees from ADAGRAD. The choice between them is a matter of implementation convenience and empirical performance for specific problem structures, not a fundamental theoretical tradeoff.

---

#### Derived Algorithms for Specific Regularizers

The paper provides explicit derivations for solving the per-round optimization problems when the ADAGRAD framework is instantiated with common regularizers $\phi$ and constraint sets $X$. All derivations start from the observation that both the RDA update (3) and the composite mirror descent update (4) can be written in the unified form:

$$x_{t+1} = \argmin_{x \in X} \left\{ \langle u, x \rangle + \phi(x) + \frac{1}{2} \langle x, H_t x \rangle \right\}$$

where for RDA, $u = \eta \bar{g}_t$ and the $\frac{1}{2} \langle x, H_t x \rangle$ term is scaled by $1/t$ implicitly (absorbed into the definition of $u$ in the paper's derivation), and for composite mirror descent, $u = \eta g_t - H_t x_t$ (since $\frac{1}{2} \langle x - x_t, H_t (x - x_t) \rangle = \langle -H_t x_t, x \rangle + \frac{1}{2} \langle x, H_t x \rangle + \text{constant}$).

---

##### $\ell_1$ Regularization: $\phi(x) = \lambda \|x\|_1$

For the RDA update with diagonal $H_t$, let $H_{t,ii} = \delta + \|g_{1:t,i}\|_2$ be the $i$-th diagonal entry. The solution to the update step decouples across coordinates and yields the closed form:

$$x_{t+1,i} = \text{sign}(-\bar{g}_{t,i}) \frac{\eta t}{H_{t,ii}} \left[ |\bar{g}_{t,i}| - \lambda \right]_+$$

where $[z]_+ = \max(z, 0)$ and $\text{sign}(0)$ can be taken arbitrarily (e.g., 0).

**What this computes:** for each coordinate, compute the average gradient $\bar{g}_{t,i}$ up to round $t$. If the absolute average gradient $|\bar{g}_{t,i}|$ is less than the regularization threshold $\lambda$, set $x_{t+1,i} = 0$ (the feature is "zeroed out" by the $\ell_1$ penalty). If it exceeds $\lambda$, move $x_{t+1,i}$ in the direction opposite to $\bar{g}_{t,i}$ with step size proportional to $\eta t / H_{t,ii}$ times the "excess" $|\bar{g}_{t,i}| - \lambda$.

**Why this differs from standard RDA:** in standard (non-adaptive) RDA (Xiao, 2010), the update is $x_{t+1,i} = \text{sign}(-\bar{g}_{t,i}) \eta \sqrt{t} [|\bar{g}_{t,i}| - \lambda]_+$, where the step size $\eta \sqrt{t}$ is identical for all coordinates. In ADAGRAD, the per-coordinate step size is $\eta t / H_{t,ii} = \eta t / (\delta + \|g_{1:t,i}\|_2)$. For a frequently-seen coordinate, $\|g_{1:t,i}\|_2$ grows roughly as $\sqrt{t}$ (assuming i.i.d. gradients), giving a step size of approximately $\eta \sqrt{t}$, matching the standard rate. For a rarely-seen coordinate, $\|g_{1:t,i}\|_2$ is small, and the step size is approximately $\eta t / \delta$, which is much larger—allowing the coordinate to quickly incorporate new information.

For the composite mirror descent update with $\ell_1$, the solution is iterative soft-thresholding with per-coordinate step sizes:

$$x_{t+1,i} = \text{sign}\left( x_{t,i} - \frac{\eta}{H_{t,ii}} g_{t,i} \right) \left[ \left| x_{t,i} - \frac{\eta}{H_{t,ii}} g_{t,i} \right| - \frac{\lambda \eta}{H_{t,ii}} \right]_+$$

**What this computes:** take a gradient step $x_{t,i} - \frac{\eta}{H_{t,ii}} g_{t,i}$, then apply soft-thresholding: if the absolute value after the gradient step is less than $\lambda \eta / H_{t,ii}$, set the coordinate to zero; otherwise, shrink toward zero by that threshold amount. The effective step size $\eta / H_{t,ii}$ and the shrinkage threshold $\lambda \eta / H_{t,ii}$ both adapt per-coordinate.

**Lazy (sparse) implementation.** When the gradient vectors are sparse (many $g_{t,i} = 0$), the updates can be performed lazily. For composite mirror descent, if coordinate $i$ receives zero gradients from round $t_0$ to $t$, its value evolves as:

$$x_{t,i} = \text{sign}(x_{t_0,i}) \left[ |x_{t_0,i}| - \frac{\lambda \eta}{H_{t_0,ii}} (t - t_0) \right]_+$$

since $H_{t,ii}$ remains constant without new gradients. The actual update is only computed on demand when coordinate $i$ next becomes active. For RDA with lazy updates, maintain $u_t = \sum_{\tau=1}^t g_\tau$ (unnormalized sum), then compute $x_{t,i} = \text{sign}(-u_{t,i}) \frac{\eta t}{H_{t,ii}} \left[ \frac{|u_{t,i}|}{t} - \lambda \right]_+$ when needed.

**Why lazy updates matter for large-scale sparse problems:** in text classification with a vocabulary of 2 million words (as in the Reuters RCV1 experiments in §6.1), each document has only a few thousand non-zero features. Eagerly updating all 2 million coordinates each round would be $O(d)$ per example—prohibitively expensive. Lazy updates reduce the per-example cost to $O(\text{nnz}(x_t))$, proportional to the number of non-zero features actually present in the current example, enabling training on massive sparse datasets.

---

##### $\ell_1$-Ball Projection: $\phi \equiv 0$, $X = \{x : \|x\|_1 \leq c\}$

When the constraint is an $\ell_1$ ball rather than an $\ell_1$ penalty, the unified update (18) reduces to a projection problem. By the substitution $z = H_t^{1/2} x$ and $A = H_t^{-1/2}$, the problem becomes:

$$z^* = \argmin_{z} \frac{1}{2} \|z - v\|_2^2 \quad \text{subject to} \quad \|A z\|_1 \leq c$$

where $v = -H_t^{-1/2} u$ with the appropriate $u$ for RDA or mirror descent. This is a **continuous quadratic knapsack problem** (Brucker, 1984). Without loss of generality, assume $v \succeq 0$ (flip signs of negative components and unflip after solving). The solution has the form:

$$z^*_i = \begin{cases} v_i - \theta^* a_i & \text{if } v_i \geq \theta^* a_i \\ 0 & \text{otherwise} \end{cases}$$

where $a_i$ is the $i$-th diagonal entry of $A$ (i.e., $a_i = 1 / H_{t,ii}^{1/2}$), and $\theta^* \geq 0$ is the Lagrange multiplier for the $\ell_1$ constraint, chosen so that $\sum_i a_i z^*_i = c$.

The algorithm (Figure 3 in the paper) finds $\theta^*$ efficiently: sort $v_i / a_i$ in descending order, find the largest prefix length $\rho$ such that $\sum_{j=1}^\rho a_j v_j - (v_\rho / a_\rho) \sum_{j=1}^\rho a_j^2 < c$, and then set:

$$\theta^* = \frac{\sum_{j=1}^\rho a_j v_j - c}{\sum_{j=1}^\rho a_j^2}$$

This is an $O(d \log d)$ algorithm (dominated by sorting) and can be reduced to $O(d)$ using randomized selection (Pardalos and Rosen, 1990).

---

##### $\ell_2$ Regularization: $\phi(x) = \lambda \|x\|_2$, $X = \mathbb{R}^d$

The $\ell_2$ regularized problem does not decouple across coordinates because the $\ell_2$ norm couples all coordinates. The update becomes:

$$\min_x \langle u, x \rangle + \frac{1}{2} \langle x, H_t x \rangle + \lambda \|x\|_2$$

There is no closed form, but the dual problem is efficiently solvable. Introducing $z = x$ with Lagrange multipliers $\alpha$, the dual reduces to:

$$\min_\alpha \langle v, \alpha \rangle + \frac{1}{2} \langle \alpha, H_t^{-1} \alpha \rangle \quad \text{subject to} \quad \|\alpha\|_2 \leq \lambda$$

where $v = H_t^{-1} u$. This is a quadratic program with an $\ell_2$ ball constraint, equivalent to:

$$\min_\alpha \langle v, \alpha \rangle + \frac{1}{2} \langle \alpha, H_t^{-1} \alpha \rangle + \frac{\theta}{2} \|\alpha\|_2^2$$

for some $\theta \geq 0$. The solution for fixed $\theta$ is $\alpha(\theta) = -(H_t^{-1} + \theta I)^{-1} v$. Since $\|\alpha(\theta)\|_2$ is monotone decreasing in $\theta$, bisection search finds the $\theta^*$ such that $\|\alpha(\theta^*)\|_2 = \lambda$. The bisection bounds are derived from the extremal eigenvalues of $H_t^{-1}$:

$$\frac{\|v\|_2}{1/\sigma_{\min}(H_t) + \theta} \leq \|\alpha(\theta)\|_2 \leq \frac{\|v\|_2}{1/\sigma_{\max}(H_t) + \theta}$$

where $\sigma_{\max}(H_t)$ and $\sigma_{\min}(H_t)$ are the largest and smallest eigenvalues of $H_t$. Setting $\theta_{\max} = \|v\|_2 / \lambda - 1/\sigma_{\max}(H_t)$ and $\theta_{\min} = \|v\|_2 / \lambda - 1/\sigma_{\min}(H_t)$ gives initial bounds. The final solution is $x^* = -H_t^{-1}(u + \alpha(\theta^*))$.

**Why bisection works efficiently:** the function $\theta \mapsto \|\alpha(\theta)\|_2$ is continuous and monotone decreasing, so bisection converges to any desired tolerance in $O(\log(1/\varepsilon))$ iterations. Each iteration requires solving a linear system $(H_t^{-1} + \theta I)^{-1} v$, which is $O(d)$ for diagonal $H_t$ and $O(d^3)$ for full $H_t$ if recomputed naively, though eigen-decomposition of $H_t$ (already available from the square root computation) reduces this to $O(d^2)$.

**Why the check $\|u\|_2 \leq \lambda$ for the zero solution:** the subdifferential of $\lambda \|x\|_2$ at $x = 0$ is $\{z : \|z\|_2 \leq \lambda\}$. The optimality condition for $x = 0$ is that $0 \in u + \partial \phi(0)$, i.e., $\|u\|_2 \leq \lambda$. If this holds, the solution is $x^* = 0$ and no bisection is needed.

---

##### $\ell_\infty$ Regularization: $\phi(x) = \lambda \|x\|_\infty$, $X = \mathbb{R}^d$

By similar dual derivation, the $\ell_\infty$ regularized problem reduces to:

$$\max_\alpha -\frac{1}{2} (u + \alpha)^\top H_t^{-1} (u + \alpha) \quad \text{subject to} \quad \|\alpha\|_1 \leq \lambda$$

**Why the constraint is $\|\alpha\|_1 \leq \lambda$:** the dual of the $\ell_\infty$ norm is the $\ell_1$ norm. This is a fundamental duality relationship: $\|x\|_\infty = \sup_{\|\alpha\|_1 \leq 1} \langle \alpha, x \rangle$, which in Fenchel duality translates to the constraint on the dual variable.

When $H_t$ is diagonal, this is an $\ell_1$ projection problem on $\alpha$ (using Algorithm 3 with $a_i = 1/H_{t,ii}$ and appropriate $v$), after which $x^* = -H_t^{-1}(u + \alpha^*)$. The solution zeros out entire groups of coordinates where the $u$ vector has small magnitude relative to $\lambda$, encouraging structural sparsity.

---

##### Mixed-Norm Regularization: $\phi(X) = \lambda \sum_{i=1}^d \|x_i\|_p$

For matrix-valued parameters $X \in \mathbb{R}^{d \times k}$ (e.g., multi-class classification weights where each row corresponds to a feature and each column to a class), the mixed $\ell_1 / \ell_p$ norm imposes row-wise sparsity: an entire row $x_i \in \mathbb{R}^k$ is zeroed out or kept, rather than individual entries. This is useful for multi-task learning where features should be selected jointly across tasks.

When $H_t$ is diagonal (operating per row independently), the problem decouples across rows. For each row $i$, the update reduces to an $\ell_p$ regularized problem (with $p = 2$ using the $\ell_2$ algorithm above; $p = \infty$ using the $\ell_\infty$ algorithm above), with the $u$ vector for that row being the corresponding slice of the full $u$. The diagonal entries of $H_t$ are maintained per-row, allowing different rows to have different effective step sizes based on their gradient histories.

**Why this decomposition works:** when $H_t$ is diagonal, the quadratic form $\langle X, H_t X \rangle = \sum_{i=1}^d H_{t,ii} \|x_i\|_2^2$ (assuming appropriate block structure) decouples across rows, and the regularizer $\sum_i \|x_i\|_p$ is already row-separable. This means the full $d \times k$ optimization splits into $d$ independent $k$-dimensional problems, each solvable using the earlier single-row algorithms.

---

#### Training and Hyperparameter Considerations

The paper describes several important implementation details:

**Step-size setting.** The theoretical analysis guides the choice of $\eta$. For the RDA variant with $\ell_1$ regularization on a domain $X$ with $\|x\|_\infty \leq D_\infty$, the analysis recommends $\eta = D_\infty$ when using the theoretical bounds directly, but in practice, $\eta$ is treated as a hyperparameter and cross-validated on a holdout set (§6.1). For composite mirror descent, the recommendation is $\eta = D_\infty / \sqrt{2}$, though again practical cross-validation supersedes this.

**The role of $\delta$.** The constant $\delta \geq 0$ ensures $H_t$ is positive definite (since $\text{diag}(s_t)$ can have zero entries when some coordinates never receive gradients). For RDA updates, the analysis requires $\delta \geq \max_t \|g_t\|_\infty$ (or $\delta \geq \max_t \|g_t\|_2$ for full-matrix) to apply Lemma 9—the $\delta I$ term must dominate the one-step change in $G_t^{1/2}$. For mirror descent, $\delta$ can be set to 0 because the bound does not require the "stale" gradient Lemma 9, and the pseudo-inverse handling (via the convention $0/0 = 0$ in Lemma 4 of Appendix C) suffices. In practice, the paper notes that $\delta$ can be set to 0 (§1.1), which is what the experiments use (implicitly, since the paper does not discuss tuning $\delta$ separately).

**Initialization.** The algorithm initializes $x_1 = 0$ (or $x_1 = \argmin_{x \in X} \phi(x)$ for general composite objectives) and $g_{1:0} = []$ (empty gradient history). The initial proximal function has $s_{0,i} = 0$ for all $i$, so $H_0 = \delta I$.

**Time complexity.** For the diagonal variant:
- Updating $s_t$: $O(\text{nnz}(g_t))$ per round (only coordinates with non-zero gradients change).
- Solving the update (with $\ell_1$, $\ell_2$, $\ell_\infty$, or mixed-norm): between $O(d)$ and $O(d \log d)$ per round for the diagonal variants, with lazy implementations reducing the effective cost to $O(\text{nnz}(g_t))$.

For the full-matrix variant:
- Updating $G_t$: $O(\text{nnz}(g_t)^2)$ if the gradient is sparse, or $O(d^2)$ for dense gradients.
- Computing $G_t^{1/2}$: $O(d^3)$ if recomputed from scratch each round, though incremental updates using the matrix square root of a rank-1 update can reduce this (not discussed in the paper).
- Solving the update: $O(d^2)$ or $O(d^3)$ depending on the regularizer.

The paper explicitly acknowledges the computational limitation of the full-matrix variant, stating that it is "likely to be confined to a few thousand dimensions" (§5), which is why the diagonal variant is the focus of experiments.

---

#### Summary of Design Choices

The ADAGRAD framework makes several deliberate design choices, each with a specific rationale:

1. **Square root of $G_t$, not $G_t$ itself.** The batch hindsight optimization and the concavity-based doubling lemma both naturally produce $\|g_{1:t,i}\|_2$ (for diagonal) and $G_t^{1/2}$ (for full). Using $G_t$ directly would give step sizes that decay as $1/t$ for coordinates with constant gradient magnitude (since $\sum_{\tau=1}^t g_{\tau,i}^2 \propto t$ for i.i.d. data), which is too aggressive and would freeze common features. Using $G_t^{-1}$ would make step sizes *grow* over time for common features, which is destabilizing. The square root achieves the right balance: step sizes decay as $1/\sqrt{t}$ for i.i.d. data, matching the theoretically optimal $\eta_t \propto 1/\sqrt{t}$ schedule.

2. **$\ell_\infty$ vs. $\ell_2$ geometry in the diagonal bound.** The diagonal regret bound depends on $\|x^*\|_\infty$ and $\max_t \|x^* - x_t\|_\infty$, not on $\|x^*\|_2$. This means the bound is tightest when the optimal predictor $x^*$ has small entries in all coordinates (e.g., $x^* \in [-1, 1]^d$), rather than having a few large entries. This is the "box-constrained" regime—natural for many sparse learning problems—and explains why ADAGRAD works well on tasks where the domain is naturally bounded per coordinate.

3. **Two update families, one proximal function.** Rather than tying the adaptation to a specific algorithm (RDA, mirror descent, FOBOS), the paper shows the same $\psi_t$ construction works for both, with only minor differences in the regret bound constants. This generality means practitioners can choose whichever update family fits their problem structure (e.g., lazy RDA for sparse streaming data, eager mirror descent for dense moderate-dimensional problems) without giving up the benefits of adaptation.

4. **Composite objective support from the start.** By designing the analysis around the composite objective $\varphi_t = f_t + \phi$ rather than the simpler $f_t$ alone, the framework accommodates structural regularizers without separate derivation. This is a significant practical advantage over AROW, which was designed for unregularized binary classification. ADAGRAD handles $\ell_1$ sparsity, group sparsity ($\ell_1/\ell_2$, $\ell_1/\ell_\infty$), and domain constraints ($\ell_1$ ball, $\ell_2$ ball) through the same unified proximal machinery.

5. **The factor of 2 in the doubling inequalites.** The constant 2 in Lemma 4 and Lemma 10 is the price paid for using running approximations rather than batch-optimal hindsight choices. The paper accepts this constant factor because it does not affect asymptotic rates, and any attempt to improve it would require knowing future gradients—the online constraint makes a constant-factor overhead inevitable. The fact that the overhead is only factor 2 (not $\log T$ or $\sqrt{T}$) means the adaptive algorithm is genuinely competitive with the hindsight-optimal choice.

## 4. Key Insights and Innovations

### Innovation 1: Treating the Proximal Function as a Learnable Object Rather Than a Design Choice

The defining conceptual move in this paper is elevating the proximal function from a **static design parameter** to a **data-dependent object that is optimized online**. Before ADAGRAD, the proximal function $\psi$ in online learning was chosen once—before seeing any data—based on domain geometry or computational convenience. You picked $\psi(x) = \frac{1}{2}\|x\|_2^2$ for Euclidean problems, or the negative entropy for simplex-constrained problems, and the choice was locked in for the entire run. Even algorithms that scaled $\psi$ over time (like RDA's $\psi_t = \sqrt{t} \psi$ or $\psi_t = t \psi$) applied that scaling uniformly to all coordinates, preserving the fundamental geometry of the initial choice.

This paper reframes the proximal function as something the algorithm should **learn from experience**. The key diagnostic: "we view the problem of adapting the proximal function as a meta-learning problem" (§1.4, final paragraph). This is not a metaphor—it is a precise theoretical stance. The algorithm observes the sequence of subgradients $g_1, g_2, \ldots$ and uses them to construct an ever-improving estimate of the ideal proximal geometry, where "ideal" is defined by the hindsight optimization problem (12) for diagonal and (15) for full-matrix cases. The regret bounds in Theorems 5 and 7 then guarantee that the online-constructed $\psi_t$ is competitive with the best $\psi$ that could have been chosen with full knowledge of all $T$ gradients in advance.

**What makes this distinctive**: it transforms the proximal function from a hyperparameter (something you tune externally) into a **state variable** (something the algorithm maintains internally). The significance is not just that ADAGRAD works better—it is that the meta-learning framing provides a **principled answer** to the question "what should the proximal function be?" instead of leaving it to practitioner intuition. The connection between the hindsight optimization and the actual algorithm (via the doubling lemmas) closes the loop: the chosen $\psi_t$ is not arbitrary but provably approximates the optimal choice.

**Contrast with prior work**: earlier adaptive methods adapted the **scalar step size** based on observed function variation (Hazan and Kale, 2008; Bartlett et al., 2007) or used second-order information for mistake-bound classification (AROW; Crammer et al., 2009). None framed the problem as learning the proximal geometry itself, and none provided regret bounds competitive with the hindsight-optimal proximal function. McMahan and Streeter (2010), in concurrent work, analyzed competitive ratios—the ratio of an adaptive algorithm's worst-case regret to the regret of the best fixed proximal function—which is a complementary perspective to ADAGRAD's direct optimization approach. But ADAGRAD's contribution is the **unified meta-learning framework** that treats the proximal function optimization as the primary algorithmic task, with the update rules (RDA, mirror descent) serving as interchangeable engines underneath.

**Why this is fundamental rather than incremental**: this reframing is not a minor tweak to existing algorithms. It changes what an online learning algorithm *is*—from a fixed procedure with tunable parameters to a two-level system where the outer loop learns the metric and the inner loop optimizes the predictions. The paper's explicit derivation of the "best in hindsight" optimization problems (12) and (15) and the proof that the online construction approximates them within a constant factor establishes this as a new theoretical primitive, not just an empirical heuristic.

**Evidence anchor**: The regret bounds in Corollaries 6 and 11 explicitly show the connection—$\gamma_T$ and $\text{tr}(G_T^{1/2})$ are expressed as infima over the space of possible proximal matrices, and the bounds scale with these infimal values. This is not hand-waving; it is a theorem that the algorithm's regret is bounded by a quantity directly derived from the hindsight-optimal proximal function.

---

### Innovation 2: Per-Feature Learning Rate Adaptation Through Gradient History Norms

While the proximal-function-as-learnable-object is the conceptual frame, the specific **mechanism** of adaptation—using the $\ell_2$ norm of historical gradients per coordinate rather than any other aggregation—is itself a distinctive insight. The paper could have used the sum of squared gradients (the diagonal of $G_t$), the sum of absolute gradients, exponentially weighted moving averages, or any other statistic. The choice of $\|g_{1:t,i}\|_2$ is not obvious *a priori*, but it emerges from two converging lines of reasoning that reinforce each other.

First, from the optimization perspective: the hindsight problem $\min_s \sum_{t,i} g_{t,i}^2 / s_i$ subject to $\sum_i s_i \leq c$ has the closed-form solution $s_i \propto \|g_{1:T,i}\|_2$, where the square root appears because of the structure of the Lagrangian (Equation 12 and the surrounding derivation in §3). The optimal scaling for each coordinate is proportional to the **root** of the accumulated squared gradients, not the accumulated squares themselves.

Second, from the analysis perspective: the doubling lemma (Lemma 4) works specifically because the function $\sqrt{x}$ is concave, yielding the inequality $\sqrt{b - a^2} \leq \sqrt{b} - a^2/(2\sqrt{b})$ that makes the induction step go through. If the algorithm had used $s_{t,i} = \sum_{\tau=1}^t g_{\tau,i}^2$ (no square root), the induction would not close—the sum $\sum_t g_{t,i}^2 / (\sum_{\tau \leq t} g_{\tau,i}^2)$ grows as $\log(1 + \sum_t g_{t,i}^2)$ rather than being bounded by a constant times the final value, which would lose the factor-2 constant and potentially introduce logarithmic overhead in the regret. If the algorithm had used $s_{t,i} = (\sum_{\tau=1}^t |g_{\tau,i}|)^p$ for $p \neq 1/2$, both the hindsight optimality and the concavity properties would fail.

**What makes this distinctive**: the paper identifies the **unique aggregator** that simultaneously solves the hindsight optimization, enables the tight doubling-lemma analysis, and produces the intuitively desirable behavior—step sizes scaling as $1/\sqrt{t}$ for i.i.d. gradients while adaptively handling variable feature frequencies. This is not a coincidence or a heuristic choice; it is a **mathematically necessary** consequence of the design goals, and the paper makes this necessity visible through the dual lens of optimization and analysis. Most adaptive methods (e.g., AdaGrad's later deep learning variants like RMSProp and Adam) would move to exponential moving averages, sacrificing the hindsight optimality guarantees for practical benefits in non-convex settings. The original ADAGRAD's commitment to the exact $\ell_2$ norm aggregator is what enables the strong theoretical guarantees.

**Contrast with prior work**: standard online gradient descent uses $\eta_t = \eta / \sqrt{t}$ globally, which gives the correct *rate* but the wrong *allocation*—every feature gets the same step size regardless of its gradient history. AROW uses a full covariance matrix updated via a rank-1 downdate (Equation 9), which implicitly adapts per-feature rates through the diagonal entries of $\Sigma_t$, but without the square root—$\Sigma_t$ approximates the inverse covariance, not its root. The paper notes this distinction explicitly (§1.4): "In contrast to AROW, the ADAGRAD algorithm uses the root of the inverse covariance matrix, a consequence of our formal analysis." The root matters because it compresses the dynamic range of step sizes: a feature with 100 times more accumulated gradient energy gets only a $10\times$ smaller step size (with root) rather than a $100\times$ smaller step size (without). This makes ADAGRAD more robust to extreme feature frequency disparities.

**Significance beyond performance**: this insight provides a **design principle** for adaptive optimization: when constructing feature-specific learning rates from historical gradients, the aggregator should be chosen to make the analysis work as a martingale-like telescoping sum with constant overhead, not just to track some empirical statistic. The success of the $\ell_2$ norm in the convex setting raises the question of what the "right" aggregator is for non-convex, non-stationary problems—a question that later work on Adam, RMSProp, and their variants would grapple with extensively, though without the clean theoretical resolution ADAGRAD achieves here.

**Evidence anchor**: the experimental section (§6) demonstrates the practical impact across four diverse datasets, but the theoretical evidence is stronger: Lemma 4 and Lemma 10 provide exact constant-factor bounds that would not hold with any other aggregator. The fact that the same square root structure appears in both the diagonal (Lemma 4) and full-matrix (Lemma 10) cases—the latter requiring substantially more sophisticated matrix analysis (Lemmas 8 and 9)—confirms this is not an artifact of the scalar case but a deeper structural property.

---

### Innovation 3: The Bregman Divergence Penalty for Changing Proximal Functions Is Self-Bounding

A subtle but technically profound contribution is the proof that when the proximal function is adapted over time, the additional penalty terms that appear in the regret decompositions do not asymptotically hurt the bounds. This is the content of Equations (14) and (16) and their surrounding analysis, but the **insight** is not the algebraic manipulation—it is the recognition that the penalty for changing $\psi$ has exactly the same form as the benefit from having changed it.

Specifically, for composite mirror descent, Proposition 3 introduces the term $\sum_{t=1}^{T-1} [B_{\psi_{t+1}}(x^*, x_{t+1}) - B_{\psi_t}(x^*, x_{t+1})]$, which penalizes the algorithm for modifying its proximal function. A naive reading suggests this could grow arbitrarily large and wipe out any gains. But the paper shows that when $\psi_t$ uses the gradient history norms ($H_t = \text{diag}(s_t)$ for diagonal or $H_t = G_t^{1/2}$ for full matrix), the sum of differences telescopes:

- Diagonal: $\sum_t (B_{\psi_{t+1}} - B_{\psi_t}) \leq \frac{1}{2} \max_t \|x^* - x_t\|_\infty^2 \sum_i \|g_{1:T,i}\|_2$
- Full matrix: $\sum_t (B_{\psi_{t+1}} - B_{\psi_t}) \leq \frac{1}{2} \max_t \|x^* - x_t\|_2^2 \text{tr}(G_T^{1/2})$

In both cases, the penalty is bounded by the same quantity—$\sum_i \|g_{1:T,i}\|_2$ or $\text{tr}(G_T^{1/2})$—that appears in the main regret term from the dual norm sum. **The adaptation penalty is of the same order as the adaptation benefit.** This is not a coincidence; it follows from the specific additive structure of $s_t$ and $G_t$, where the per-round increments are the squared gradients themselves.

**What makes this distinctive**: the paper identifies and resolves a tension that is inherent to any algorithm that modifies its geometry online. Changing the metric under which distances are measured creates a "mismatch cost"—the new Bregman divergence evaluated at the old point does not equal the old divergence. Prior work had avoided this issue entirely by keeping $\psi$ fixed. This paper shows that the issue is **self-resolving** when the metric updates are driven by the same gradient outer products that appear in the dual norm terms. The proof technique—telescoping the sum of differences using the additivity of $G_t$—is elegant but not deep; the **conceptual contribution** is recognizing that this telescoping exists and that it bounds the penalty by the benefit.

**Why this matters beyond this paper**: the self-bounding property means that **adaptation is essentially free** in an asymptotic sense. An algorithm designer does not need to trade off the rate of adaptation against the mismatch cost—the two scale together. This justifies aggressive adaptation: you can update your proximal function every round without worrying that the cumulative mismatch will dominate. This principle has implications for any online metric learning problem, not just stochastic gradient descent. The paper does not dwell on this generalization, but it is implicit in the analysis: any additive update to a matrix-induced metric that is driven by outer products of observed vectors will exhibit similar telescoping when plugged into Bregman divergence-based regret decompositions.

**Evidence anchor**: the regret bounds in Theorems 5 and 7 have no hidden logarithmic factors or adaptation-rate parameters. The constants are explicit and the penalty terms are directly incorporated. The fact that the mirror descent and RDA bounds end up with the same asymptotic dependence on $\sum_i \|g_{1:T,i}\|_2$ (up to domain-dependent constants) despite having different penalty structures—RDA has no explicit Bregman penalty because it uses the final $\psi_T$ directly in $\psi_T(x^*)/\eta$, while mirror descent pays the telescoping sum—shows that the two are fundamentally equivalent, and that the adaptation cost in mirror descent exactly mirrors the $\psi_T(x^*)$ term in RDA.

---

### Innovation 4: The Diagonal-vs-Full-Matrix Distinction as a Computational-Statistical Tradeoff with Formal Bounds

The paper introduces a distinction that has become standard in adaptive optimization: diagonal versus full-matrix preconditioning. But the contribution is not the distinction itself—it is the **unified theoretical treatment** that shows both variants arise from the same principle (hindsight-optimal proximal matrix optimization) and enjoy regret bounds that differ in precise, interpretable ways. This unified treatment makes the diagonal-vs-full choice a **quantifiable statistical-computational tradeoff** rather than an arbitrary engineering decision.

**The diagonal variant** scales with $\sum_{i=1}^d \|g_{1:T,i}\|_2$ and depends on the $\ell_\infty$ geometry of the domain ($\|x^*\|_\infty$, $D_\infty$). It captures per-coordinate frequency variation but ignores correlations. **The full-matrix variant** scales with $\text{tr}(G_T^{1/2})$ and depends on the $\ell_2$ geometry ($\|x^*\|_2$, $D_2$). It captures the principal components of the gradient distribution—if gradients lie in a low-dimensional subspace, $\text{tr}(G_T^{1/2})$ can be much smaller than $\sum_i \|g_{1:T,i}\|_2$.

The paper provides the **infimal representation** that makes this comparison explicit (Corollaries 6 and 11):

$$\sum_{i=1}^d \|g_{1:T,i}\|_2 = \sqrt{d \cdot \inf_s \left\{ \sum_{t=1}^T \langle g_t, \text{diag}(s)^{-1} g_t \rangle : s \succeq 0, \langle \mathbf{1}, s \rangle \leq d \right\}}$$

$$\text{tr}(G_T^{1/2}) = \sqrt{d \cdot \inf_S \left\{ \sum_{t=1}^T g_t^\top S^{-1} g_t : S \succeq 0, \text{tr}(S) \leq d \right\}}$$

**What makes this distinctive**: the two bounds have identical structure—both are $\sqrt{d}$ times the square root of an infimum over matrix-constrained quadratic forms—but the constraint differs: diagonal matrices vs. all PSD matrices. The gap between these two infima quantifies the **value of modeling feature correlations**. When the optimal $S$ is approximately diagonal (features decorrelate), the diagonal bound is nearly tight. When gradients are strongly correlated (e.g., co-occurring features in text or pixels in images), the full-matrix bound can be substantially smaller. The paper thus provides a formal language for reasoning about when full-matrix adaptation is worth the $O(d^3)$ computational cost.

**Prior work contrast**: earlier second-order methods like AROW maintained full covariance matrices but analyzed them through mistake bounds that do not cleanly separate into diagonal-vs-full comparisons. The confidence-weighted learning literature did not derive the diagonal variant as a principled restriction of the full method with provable guarantees—diagonal AROW was an engineering optimization. ADAGRAD's derivation makes the diagonal variant a **first-class theoretical object** with its own regret bound and its own optimality characterization (competitive with the best diagonal proximal matrix in hindsight), not merely a computationally convenient approximation to the full method.

**Why this is fundamental**: it establishes a **complexity hierarchy** for adaptive proximal functions. The diagonal version is optimal over the restricted class of axis-aligned metrics; the full version is optimal over all metrics. The gap between them is a measure of how much the problem's intrinsic geometry deviates from axis-alignment. This hierarchy generalizes naturally: the paper's concluding remarks (§7, Corollary 12) sketch a block-diagonal variant that interpolates between the extremes, allowing practitioners to choose a granularity (per-coordinate, per-block, full) matched to their computational budget and problem structure, with formal guarantees at each level. The block-diagonal extension is not fully developed, but the theoretical machinery is in place for it.

**Evidence anchor**: the experimental section uses only the diagonal variant (§6), which is the pragmatic choice for high-dimensional problems like text classification ($d \approx 2 \times 10^6$ for Reuters RCV1). The paper explicitly acknowledges the full-matrix version is "likely to be confined to a few thousand dimensions" (§5), making the diagonal theory the practically relevant contribution. However, the full-matrix analysis validates the diagonal version theoretically: it shows the diagonal bound is not a loose artifact of a restricted analysis but rather the best possible guarantee over the class of diagonal proximal functions, competitive with the best diagonal metric in hindsight.

---

### Innovation 5: Composite Objective Support as a First-Class Design Principle, Not an Afterthought

Most online learning algorithms are designed and analyzed for the basic setting $f_t(x)$ without regularization, and regularization is added later as a separate modification—sometimes with new analysis, sometimes without. ADAGRAD bakes the composite structure $\varphi_t(x) = f_t(x) + \phi(x)$ into the framework from the beginning. This is visible in the regret definition (2), the update rules (3) and (4), and the derived algorithms in §5.

**What makes this distinctive**: the paper treats the regularizer $\phi$ and the domain constraint $X$ as equal citizens with the loss functions $f_t$ in the algorithmic design. The proximal function $\psi_t$ handles the "geometry of optimization" (how to move through parameter space), while $\phi$ handles the "geometry of solutions" (structural properties like sparsity or group sparsity). The two are complementary and independently specifiable: you can pair diagonal ADAGRAD with $\ell_1$ regularization for sparse per-coordinate solutions, or with $\ell_1/\ell_2$ mixed norms for row-sparse multi-task solutions, or with $\ell_2$ regularization for dense but small-norm solutions—all within the same framework and all covered by the same regret analysis.

This is in stark contrast to AROW and confidence-weighted learning, which were designed for binary classification with specific loss functions (hinge or logistic) and have no natural extension to arbitrary regularizers or constraints. The PA algorithm (Crammer et al., 2006) handles the hinge loss with an $\ell_2$ penalty on parameter change, but the regularizer is tied to the algorithm—you cannot swap in an $\ell_1$ penalty for sparsity without rederiving the update and its analysis. ADAGRAD's separation of $\psi_t$ (proximal, geometry of space) from $\phi$ (structural penalty) from $f_t$ (data loss) is a **modular design** that makes the framework extensible.

**Lazy updates as a consequence, not an add-on**: the lazy update schemes described in §5.1 for $\ell_1$ regularization are not a separate algorithm but a direct consequence of the composite structure plus the additivity of $H_t$. Because $H_{t,ii}$ only changes when coordinate $i$ receives a non-zero gradient, and the $\ell_1$ penalty drives inactive coordinates toward zero at a predictable rate, the exact state of any coordinate can be reconstructed on demand from its last active value and the elapsed time. This property is critical for large-scale sparse problems where $d$ is in the millions but per-example non-zeros are in the thousands—eagerly updating all coordinates each round would be $O(d)$ per example and infeasible. The lazy scheme reduces per-example cost to $O(\text{nnz}(g_t))$, matching the cost of the gradient computation itself. The paper does not just note this as an implementation trick; it derives the exact functional form of the decay, showing it follows mathematically from the update rule.

**Significance beyond this paper**: the composite objective design would become standard in later adaptive methods (AdaGrad's deep learning descendants like Adam typically handle weight decay as a separate $\ell_2$ penalty), but ADAGRAD is the first to provide a complete theoretical framework with explicit algorithms for $\ell_1$, $\ell_2$, $\ell_\infty$, and mixed-norm penalties—all derived from the same unified proximal optimization in (18). The derivation in §5 is not a collection of case-by-case hacks; it is a systematic exposition of how to solve the generic update (18) for different $\phi$ and $X$, using dual reformulations, bisection, and known quadratic knapsack algorithms as subroutines.

**Evidence anchor**: The experimental section (§6) uses $\ell_1$ regularization for sparsity (Tables 1, 4; Figure 7) and mixed $\ell_1/\ell_2$ and $\ell_1/\ell_\infty$ regularization for multi-class MNIST (Table 3), demonstrating that the framework's modularity translates to practical flexibility. The sparsity-accuracy tradeoff curves in Figure 7 show that ADAGRAD can produce predictors with only 1% non-zero coefficients that match the accuracy of the fully dense AROW predictor—a direct consequence of the $\ell_1$ regularizer being properly integrated into the adaptive proximal framework rather than bolted on.

## 5. Experimental Analysis

### Evaluation Methodology

- **Dataset.** The paper evaluates on four real-world datasets with diverse characteristics: (1) **Reuters RCV1** (§6.1), a text classification corpus of approximately 800,000 articles labeled with four high-level categories (ECAT, CCAT, MCAT, GCAT), using 0/1 bigram features yielding roughly 2 million dimensions with extreme sparsity (most documents have fewer than 5,000 non-zero features); (2) **ImageNet** (§6.2), a large-scale image database organized by WordNet nouns, where the task is ranking images for each of 15,000 selected noun categories using Grangier and Bengio's (2008) visterms features producing approximately 10,000-dimensional sparse vectors; (3) **MNIST** (§6.3), the 28×28 pixel digit recognition dataset transformed via Gaussian kernel machines over a support set of roughly 3,000 images, yielding a 30,000-dimensional dense feature space; (4) **KDD Census Income** (§6.4) from the UCI repository, containing 199,523 training and 99,762 test instances with 4,001-dimensional 0/1 features constructed by quantizing and crossing demographic variables. For RCV1, each experiment randomly holds out 25% of the data as a test set, with four independent shuffles averaged. For MNIST, the first 5,000 of 60,000 training examples are used for parameter selection. For Census Income, ten experiments are run on random shuffles of training data at varying proportions (5% to 100%).

- **Base model(s).** The learning algorithms are linear predictors (binary classifiers for RCV1, Census; multiclass for MNIST; ranking machines for ImageNet), trained entirely online with a single pass through the data. No pretrained model is used; all experiments start from $x_1 = 0$ and learn exclusively from the sequential data stream. This is a pure online learning evaluation—there is no pretraining versus inference tradeoff.

- **Metrics.** For Reuters RCV1 and Census Income, the primary metric is **test set error rate** (proportion of misclassified examples) on held-out test data after a single pass through the training set. For ImageNet, the metrics are **average precision** and **precision-at-k** for k = 1, 3, 5, 10, following the evaluation protocol of Grangier and Bengio (2008): for each of the 15,000 classes, positive and negative test images are ranked by inner product with the learned weight vector, and precision is computed at each rank. Average precision across all classes and across twelve independent evaluations is reported. For MNIST, the metric is **cumulative online mistakes** during the training pass and **test set error rate** with sparsity proportions. For all experiments involving $\ell_1$ regularization, the **proportion of non-zero features** in the final predictor is reported alongside accuracy to assess the sparsity-accuracy tradeoff.

- **Baselines.** The paper compares six algorithmic families, yielding twelve specific algorithm variants when combined with different loss functions and regularization:
  
  1. **RDA** (Xiao, 2010): Regularized dual averaging with fixed proximal function $\psi_t(x) = \sqrt{t} \psi(x)$ for a strongly convex $\psi$ (typically $\ell_2$). With and without $\ell_1$ regularization.
  
  2. **FOBOS** (Duchi and Singer, 2009): Forward-backward splitting / composite mirror descent with fixed proximal function. With and without $\ell_1$ regularization.
  
  3. **AdaGrad-RDA**: The diagonal AdaGrad variant using the primal-dual subgradient update (3), with adaptive proximal matrix $H_t = \delta I + \text{diag}(G_t)^{1/2}$. With and without $\ell_1$ regularization, and with mixed $\ell_1/\ell_2$ and $\ell_1/\ell_\infty$ regularization for MNIST.
  
  4. **AdaGrad-FOBOS**: The diagonal AdaGrad variant using composite mirror descent (4). With and without $\ell_1$ regularization.
  
  5. **PA** (Passive-Aggressive; Crammer et al., 2006): An online learning algorithm that updates by solving $\min_x [1 - y_t \langle z_t, x \rangle]_+ + \frac{\lambda}{2} \|x - x_t\|_2^2$. Used with both hinge and logistic loss.
  
  6. **AROW** (Adaptive Regularization of Weights; Crammer et al., 2009): A second-order confidence-weighted method maintaining mean $\mu_t$ and covariance $\Sigma_t$, with the update in Equation (9). Tested with both hinge and logistic loss. AROW produces fully dense predictors.

  For RCV1, all twelve variants (RDA, FOBOS, AdaGrad-RDA, AdaGrad-FOBOS, PA, AROW, each with both hinge and logistic loss) are evaluated, though Table 1 reports only hinge loss results since "results for both hinge and logistic losses are qualitatively and quantitatively very similar" (§6.1). For ImageNet, the comparison focuses on AdaGrad-RDA with $\ell_1$, vanilla RDA with $\ell_1$, AROW, and PA, using the ranking hinge loss. For MNIST, multiclass variants of PA, RDA, AdaGrad-RDA, and their mixed-norm counterparts are compared (no AROW, as "there is no known multiclass AROW algorithm," §6.3). For Census Income, AROW, PA, RDA, and AdaGrad-RDA with and without $\ell_1$ are compared.

- **Generation budget / compute accounting.** All algorithms process exactly one example per round in a single sequential pass through the training data (fully online, no minibatching). The "compute budget" is therefore implicitly the number of training examples processed—a fixed quantity shared identically across all algorithms. There is no separate inference budget or beam search; this is traditional online learning where the test set evaluation uses the final predictor after one pass. For lazy sparse updates (§5.1), the per-example cost is proportional to $\text{nnz}(g_t)$ rather than $d$, which is the key computational advantage of the diagonal variant in high-dimensional sparse settings. The paper does not report wall-clock time or FLOP counts; compute efficiency is assessed indirectly through the relationship between sparsity (proportion of non-zero features) and accuracy.

- **Cross-validation / statistical protocol.** For RDA, FOBOS, and their AdaGrad variants, the step-size parameter $\eta$ is cross-validated "by simply running multiple passes and then choosing the output of the learner that had the fewest mistakes during training" (§6.1). For PA and AROW, the regularization parameter $\lambda$ is selected using the same approach. For ImageNet, "the choice of initial stepsize for each algorithm on a small held-out set" (§6.2) is cross-validated per image category. For MNIST, the first 5,000 training examples select $\eta$ (RDA) and $\lambda$ (PA). For Census Income, the first 10,000 training examples are used for parameter selection. The regularization multiplier $\lambda$ for $\ell_1$ experiments is "selected so that RDA achieved approximately 10% non-zero predictors" (§6.1) for the RCV1 experiments in Table 1; for the sparsity-accuracy tradeoff experiments in §6.5, $\lambda$ is swept from $10^{-8}$ to $10^{-1}$ to sample the full range from dense to all-zeros predictors. Results are reported as averages across multiple independent shuffles (4 for RCV1, 10 for Census Income), with variance noted where relevant (e.g., "variance was on the order of $10^{-6}$" for ImageNet, Table 2; "variance of the test error rates is on the order of $10^{-6}$" for Census Income, §6.4).

---

### Main Quantitative Results

#### Text Classification: Reuters RCV1

The headline result for text classification is that **adaptive algorithms (AdaGrad and AROW) substantially outperform non-adaptive methods on all four RCV1 categories**, with AdaGrad variants achieving lower error rates than AROW. Table 1 (reproduced below) gives the test set error rates and proportion of non-zero features for the hinge loss versions:

| Algorithm | ECAT | CCAT | GCAT | MCAT |
|---|---|---|---|---|
| RDA | .051 (.099) | .064 (.123) | .046 (.092) | .037 (.074) |
| FOBOS | .058 (.194) | .111 (.226) | .056 (.183) | .056 (.146) |
| AdaGrad-RDA | **.044** (.086) | **.053** (.105) | **.040** (.080) | **.035** (.063) |
| AdaGrad-FOBOS | **.044** (.238) | **.053** (.276) | **.040** (.225) | .034 (.176) |
| PA | .059 | .107 | .066 | .053 |
| AROW | .049 | .061 | .044 | .039 |

Parenthetical values are the proportion of non-zero coefficients in the final predictor—lower is sparser. For AdaGrad variants, sparsity is achieved through $\ell_1$ regularization applied within the adaptive proximal framework. RDA achieves the highest sparsity levels (e.g., .099 for ECAT vs. .086 for AdaGrad-RDA), but at the cost of higher error rates. PA and AROW produce fully dense predictors (no sparsity).

Several patterns are evident. First, both adaptive methods (AdaGrad and AROW) dominate the non-adaptive RDA and FOBOS across all categories. On ECAT, the best non-adaptive method (RDA) has 5.1% error, while AdaGrad-RDA achieves 4.4%—a relative reduction of approximately 14%. On CCAT, the gap is even larger: RDA at 6.4% vs. AdaGrad-RDA at 5.3% (17% relative improvement). FOBOS performs notably worse than RDA on CCAT (11.1% vs. 6.4%), highlighting that even among non-adaptive methods, the choice of proximal function matters significantly.

Second, AdaGrad-RDA and AdaGrad-FOBOS achieve nearly identical error rates on all categories (differing only on MCAT, where AdaGrad-FOBOS has 3.4% vs. AdaGrad-RDA's 3.5%), suggesting that the adaptation mechanism dominates any differences between the RDA and mirror descent update families on this task. This is consistent with the theoretical finding that both update families enjoy essentially the same asymptotic regret bounds with AdaGrad (Theorem 5).

Third, AdaGrad-RDA outperforms AROW on all four categories: ECAT (4.4% vs. 4.9%), CCAT (5.3% vs. 6.1%), GCAT (4.0% vs. 4.4%), MCAT (3.5% vs. 3.9%). The gap is largest on CCAT (0.8 percentage points) and smallest on GCAT (0.4 points). This is notable because AROW maintains full second-order covariance information (dense $d \times d$ matrix), while AdaGrad-RDA uses only diagonal information—yet AdaGrad achieves equal or better accuracy. The paper's theoretical explanation is that for sparse text features where co-occurrence structure may be limited, the diagonal proximal function captures most of the relevant adaptation while avoiding the estimation noise of a full covariance matrix on $d \approx 2 \times 10^6$ dimensions from a single pass of data.

Fourth, the sparsity levels of AdaGrad methods are notable: AdaGrad-RDA achieves .063 non-zero proportion on MCAT vs. .074 for non-adaptive RDA—meaning it uses roughly 15% fewer features while still achieving lower error (3.5% vs. 3.7%). However, AdaGrad-FOBOS shows substantially higher non-zero proportions than AdaGrad-RDA (.238 vs. .086 on ECAT), despite identical error rates. This suggests that for a fixed error rate, the RDA update family combined with $\ell_1$ regularization is more effective at producing sparse solutions than the mirror descent family—an empirical finding that the theoretical analysis does not address directly.

The paper also notes that unregularized RDA and FOBOS "attained similar results as did the $\ell_1$-regularized variants (of course without sparsity)" (§6.1), but these results are omitted from the table to avoid clutter.

#### Image Ranking: ImageNet

The ImageNet experiment tests a fundamentally different setting: large-scale ranking with 15,000 separate binary classifiers trained simultaneously, where the input features (visterms) produce approximately 10,000-dimensional sparse vectors from image patches. Table 2 reports mean performance across all 15,000 classes, averaged over twelve evaluations:

| Algorithm | Avg. Prec. | P@1 | P@3 | P@5 | P@10 | Prop. nonzero |
|---|---|---|---|---|---|---|
| AdaGrad-RDA | **0.6022** | 0.8502 | 0.8307 | 0.8130 | 0.7811 | 0.7267 |
| AROW | 0.5813 | **0.8597** | **0.8369** | **0.8165** | **0.7816** | 1.0000 |
| PA | 0.5581 | 0.8455 | 0.8184 | 0.7957 | 0.7576 | 1.0000 |
| RDA | 0.5042 | 0.7496 | 0.7185 | 0.6950 | 0.6545 | 0.8996 |

The headline result depends on which metric one prioritizes. **AdaGrad-RDA achieves the highest average precision** (0.6022, versus 0.5813 for AROW, a 3.6% relative improvement), but **AROW achieves better precision-at-k for small k** (P@1: 0.8597 vs. 0.8502; P@3: 0.8369 vs. 0.8307; P@5: 0.8165 vs. 0.8130). The gap narrows and eventually reverses as k increases: at P@10, the two are essentially tied (0.7811 vs. 0.7816), and in terms of average precision—which weights precision at all positions where a relevant image appears—AdaGrad pulls ahead. The paper characterizes this as "ADAGRAD's performance catches up to and eventually surpasses AROW's as k grows" (§6.2).

This pattern suggests an interesting difference in how the two adaptive methods rank images. AROW's maintenance of a full covariance matrix may help it place the very first few images more accurately (benefiting P@1 and P@3), perhaps by capturing correlations between visterm features that appear together in images of the same class. AdaGrad's diagonal adaptation produces a different ranking that performs slightly less well at the very top positions but achieves better overall ranking quality as measured by average precision. The paper does not analyze this tradeoff further.

The sparsity result is striking: AdaGrad-RDA achieves its top average precision using only 72.67% of the available features, while AROW and PA use all features (1.0 non-zero proportion). RDA with $\ell_1$ uses 89.96% of features but achieves substantially worse average precision (0.5042). This demonstrates that AdaGrad's adaptive learning rates, not just $\ell_1$ regularization, are responsible for selecting the right features—AdaGrad-RDA is both sparser and more accurate than RDA. Achieving better performance with fewer features is a strong indicator that the per-feature adaptation is correctly identifying and emphasizing informative rare features while suppressing noisy common ones.

Passive-Aggressive (PA) occupies an intermediate position: better than RDA on all metrics (avg. prec. 0.5581 vs. 0.5042) but substantially worse than the adaptive methods. This confirms that the benefit of AdaGrad and AROW is not simply from being "online" versus "batch"—all algorithms are online—but specifically from maintaining second-order information about the gradient distribution.

The variance across twelve evaluations is reported as "on the order of $10^{-5}$" (§6.2), which is negligible relative to the differences between algorithms and indicates that the 15,000-class average is highly stable.

#### Multiclass Optical Character Recognition: MNIST

The MNIST experiment uses a fundamentally different feature representation from the sparse text and image tasks: Gaussian kernel expansions over a support set of approximately 3,000 images, producing 30,000-dimensional **dense** feature vectors. This tests whether AdaGrad's adaptation helps even when sparsity is not the primary concern—the gradient vectors are dense, but the per-coordinate gradient magnitudes may still vary substantially across the 30,000 dimensions (10 classes × 3,000 support vectors).

Figure 5 plots cumulative online mistakes during training for five algorithms: multiclass PA, RDA, RDA with $\ell_1/\ell_2$ regularization, AdaGrad-RDA, and AdaGrad-RDA with $\ell_1/\ell_2$ regularization. The curves show:

- **AdaGrad-RDA and PA track each other closely**, both making approximately 1,000 mistakes after seeing roughly 4,000 examples, and following similar trajectories throughout training. This is notable because PA is known to be a strong performer on multiclass problems with hinge loss, and AdaGrad matches it without requiring tuning of an aggressive update parameter ($\lambda$ in PA).
- **Non-adaptive RDA is substantially worse**, making roughly 2,000 mistakes at the same point—approximately double the error rate of the adaptive methods early in training. The gap persists and widens: by 60,000 examples, AdaGrad-RDA has roughly 3,000 cumulative mistakes versus roughly 5,000 for RDA (a 40% reduction).
- **Mixed-norm regularization with AdaGrad** ($\ell_1/\ell_2$ or $\ell_1/\ell_\infty$) performs comparably to unregularized AdaGrad-RDA in terms of online mistakes, but as shown in Table 3, it does so with substantially fewer non-zero rows.
- **Non-adaptive RDA with $\ell_1/\ell_2$ regularization** is also worse than its AdaGrad counterpart, though the figure does not clearly separate the two non-adaptive variants.

Table 3 provides the final test set error rates and sparsity proportions after a full pass:

| Algorithm | Test error rate | Prop. nonzero |
|---|---|---|
| PA | **0.062** | 1.000 |
| AdaGrad-RDA | 0.066 | 1.000 |
| RDA | 0.108 | 1.000 |
| AdaGrad-RDA $\lambda = 5 \cdot 10^{-4}$ | 0.100 | 0.569 |
| RDA $\lambda = 5 \cdot 10^{-4}$ | 0.138 | 0.878 |
| AdaGrad-RDA $\lambda = 10^{-3}$ | 0.137 | **0.144** |
| RDA $\lambda = 10^{-3}$ | 0.192 | 0.532 |

Without regularization, PA achieves the lowest error rate at 6.2%, with AdaGrad-RDA close behind at 6.6%, while non-adaptive RDA is substantially worse at 10.8%. This 4.2 percentage point gap between AdaGrad-RDA and RDA on a dense kernel machine task demonstrates that the benefits of per-feature adaptation extend well beyond sparse feature settings—even when every feature appears in every example, the accumulation of per-coordinate gradient norms provides useful information about which coordinates need large versus small updates.

With $\ell_1/\ell_2$ mixed-norm regularization (which zeroes out entire rows of the 30,000 × 10 multiclass weight matrix), a clear sparsity-accuracy tradeoff emerges:

- At $\lambda = 5 \cdot 10^{-4}$, AdaGrad-RDA uses only 56.9% of rows and achieves 10.0% error, versus RDA's 87.8% non-zero rows and 13.8% error. AdaGrad is simultaneously sparser (uses 31 percentage points fewer rows) and more accurate (3.8 percentage points lower error).
- At $\lambda = 10^{-3}$ (stronger regularization), AdaGrad-RDA achieves an extreme sparsity of only 14.4% non-zero rows with 13.7% error, while RDA has 53.2% non-zero rows and 19.2% error. AdaGrad maintains error roughly comparable to the unregularized RDA (10.8%) while using 85.6% fewer rows—a dramatic demonstration that the adaptation identifies and preserves the most important features while aggressively zeroing out the rest.

The key comparison is AdaGrad-RDA at $\lambda = 10^{-3}$ (13.7% error, 14.4% non-zero) versus unregularized RDA (10.8% error, 100% non-zero): AdaGrad achieves error only 2.9 percentage points higher while using 85.6% fewer parameters. In contrast, RDA at the same $\lambda = 10^{-3}$ achieves 19.2% error with 53.2% non-zero—it both uses more parameters (about 3.7× more non-zero rows) and achieves substantially worse accuracy. **The adaptive learning rates are doing the heavy lifting in feature selection**, not just the regularizer.

The mixed-norm regularizers ($\ell_1/\ell_2$ and $\ell_1/\ell_\infty$) are reported to perform similarly (§6.3); the table shows only $\ell_1/\ell_2$ results.

#### Income Prediction: KDD Census

The Census Income dataset tests a medium-dimensional dense binary classification problem (4,001 features constructed from quantized demographic variables and their interactions). The experimental protocol varies the proportion of training data seen (5%, 10%, 25%, 50%, 100%), measuring test set error rate after each fraction. Table 4 reports the results:

| Prop. Train | 0.05 | 0.10 | 0.25 | 0.50 | 1.00 |
|---|---|---|---|---|---|
| AROW | 0.049 | 0.048 | 0.046 | 0.045 | 0.044 |
| PA | 0.055 | 0.052 | 0.050 | 0.049 | 0.048 |
| RDA | 0.055 | 0.054 | 0.052 | 0.051 | 0.050 |
| AdaGrad-RDA | 0.053 | 0.051 | 0.049 | 0.048 | 0.047 |
| $\ell_1$ RDA | 0.056 (.075) | 0.054 (.066) | 0.053 (.058) | 0.052 (.053) | 0.051 (.050) |
| $\ell_1$ AdaGrad-RDA | **0.052** (.062) | **0.051** (.053) | **0.050** (.044) | **0.050** (.040) | **0.049** (.037) |

Figure 6 visualizes the test error rate curves for the four dense algorithms (AROW, PA, RDA, AdaGrad-RDA) as a function of training proportion, omitting the $\ell_1$ variants.

The key findings:

**AROW consistently achieves the lowest error rates** among all methods at every training proportion, from 4.9% at 5% training to 4.4% at 100% training. AdaGrad-RDA is second-best among the dense variants, with error rates roughly 0.3–0.4 percentage points above AROW at each point (e.g., 4.7% vs. 4.4% at full training). This is the first experiment where AdaGrad does not clearly outperform AROW—on this medium-dimensional dense dataset, AROW's full covariance matrix appears to provide an advantage over AdaGrad's diagonal adaptation.

However, the **sparse $\ell_1$ AdaGrad-RDA variant** tells a more nuanced story. At full training data, $\ell_1$ AdaGrad-RDA achieves 4.9% error with only 3.7% non-zero features, compared to AROW's 4.4% error with 100% non-zero features. The gap is 0.5 percentage points of error in exchange for a 96.3% reduction in model size—a favorable tradeoff in many deployment scenarios. Moreover, $\ell_1$ AdaGrad-RDA matches or beats non-regularized PA, RDA, and vanilla AdaGrad-RDA in error rate at most training proportions while being dramatically sparser. For example, at 50% training, $\ell_1$ AdaGrad-RDA hits 5.0% error with 4.0% non-zero, compared to 4.9% for PA (dense) and 5.1% for RDA (dense).

At small training proportions, the sparsity benefits are even more pronounced relative to non-adaptive methods. At 5% training, $\ell_1$ AdaGrad-RDA achieves 5.2% error with 6.2% non-zero features, while non-adaptive $\ell_1$ RDA achieves 5.6% error with 7.5% non-zero. AdaGrad is both sparser and more accurate. Comparing to dense methods, $\ell_1$ AdaGrad-RDA at 5.2% error outperforms PA (5.5%) and matches RDA (5.5%) while using only 6.2% of the features.

The sparsity trend in Table 4 reveals that **AdaGrad produces increasingly sparse solutions relative to RDA as training data increases**: at 5% training, both $\ell_1$ variants have similar sparsity (.075 RDA, .062 AdaGrad—a 1.17 ratio); at 100% training, this ratio grows to .050 vs. .037 (1.35 ratio). AdaGrad becomes more aggressive at zeroing out features as it accumulates more gradient information, consistent with the theoretical behavior: coordinates that receive small cumulative gradients have their effective step sizes driven down by the growing $H_{t,ii}$, making them more likely to be thresholded to zero by the $\ell_1$ penalty. Non-adaptive RDA lacks this per-coordinate signal, so its sparsity pattern is driven purely by the global $\lambda$ threshold.

#### Sparsity-Accuracy Tradeoff Curves

The final experiment (§6.5) systematically sweeps the $\ell_1$ regularization parameter $\lambda$ in AdaGrad-RDA from $10^{-8}$ (essentially no regularization) to $10^{-1}$ (heavy regularization), producing predictors ranging from 100% non-zero to 0% non-zero. Figure 7 plots test set error rate versus proportion of non-zero coefficients for each of the four RCV1 categories, with AROW's dense performance shown as a horizontal reference line.

The key observations from Figure 7:

- **As soon as the predictor has more than roughly 1% non-zero coefficients, AdaGrad matches or beats AROW's test error rate** on all four categories. For ECAT, the error rate at 1% non-zero is approximately 0.065—already comparable to AROW's 0.049—and continues to improve as more features are included, reaching below 0.045 at 100% non-zero. For CCAT, the threshold is similar: at 1% non-zero, error is roughly 0.065 versus AROW's 0.061, dropping to below 0.055 at moderate sparsity levels. This means that **99% of features can be discarded with minimal or no accuracy penalty**, as long as the right 1% are kept—and AdaGrad's adaptive rates identify the right ones.

- **The sparsity-accuracy tradeoff is remarkably flat for moderate-to-high sparsity levels** (1% to 100% non-zero). On ECAT, error drops from roughly 0.065 at 1% non-zero to roughly 0.043 at 100% non-zero—a 2.2 percentage point improvement over two orders of magnitude more features. On GCAT, the curve is even flatter, with error around 0.042 at 10% non-zero and 0.040 at 100% non-zero. This flatness implies that the "effective dimensionality" of the learned predictor is extremely low relative to the feature space—most features carry redundant or negligible signal.

- **At very low non-zero proportions (below roughly 0.1%), error degrades rapidly**, confirming that there is a minimum set of genuinely informative features that must be retained. This sharp transition from poor to good performance at very low sparsity levels indicates that AdaGrad is finding a "core set" of highly predictive rare features that the $\ell_1$ penalty preserves while zeroing out the mass of common but less discriminative features.

- **The AROW baseline is matched or beaten by AdaGrad at all non-zero proportions above roughly 0.5-1%** across all four categories. This is the strongest evidence that AdaGrad's diagonal adaptation is sufficient for these sparse text tasks—the full covariance information in AROW does not provide a meaningful advantage in terms of the sparsity-accuracy frontier. The diagonal proximal function, by adapting per-feature learning rates based on gradient history, captures essentially all the relevant feature-level information needed to distinguish informative from uninformative features.

The paper notes that variance in error rates for these experiments "is on the order of $10^{-6}$" (§6.5), so no error bars are drawn—the curves are highly reliable. The qualitative pattern "was qualitatively similar" for other sparse datasets beyond RCV1, though those additional results are not shown.

---

### Ablation Studies and Robustness Checks

The paper's ablation and robustness analysis is less systematic than modern standards would demand—there are no formal ablation tables isolating individual components—but several comparisons embedded in the main experiments serve as implicit ablations.

**Diagonal vs. full-matrix proximal functions (implicit in experimental design).** All experiments use the diagonal variant of AdaGrad. The full-matrix variant, despite its theoretical development in Section 4 and Theorem 7, is never experimentally evaluated. This is an implicit ablation confirming that diagonal adaptation is sufficient for the tested problem scales and that the computational cost of the full matrix ($O(d^3)$ matrix square root) would be prohibitive for $d \approx 2 \times 10^6$ (RCV1) or even $d \approx 10^4$ (ImageNet). The paper acknowledges this in Section 5: "the full matrix version is likely to be confined to a few thousand dimensions." The experimental results can therefore be viewed as a large-scale robustness check of the diagonal approximation against the non-adaptive baselines and against the full-covariance AROW.

**RDA vs. composite mirror descent update families (Tables 1, 3).** Table 1 directly compares AdaGrad-RDA and AdaGrad-FOBOS on RCV1, finding "qualitatively and quantitatively very similar" performance in error rate (within 0.1-0.2 percentage points on most categories). The main difference is sparsity: AdaGrad-FOBOS consistently produces denser solutions than AdaGrad-RDA at the same regularization level (.238 vs. .086 non-zero on ECAT). This suggests the RDA update family (which uses average gradients rather than instantaneous gradients) interacts more favorably with $\ell_1$ regularization for producing sparse solutions. The paper does not explore this difference theoretically—the regret bounds predict similar asymptotic performance but do not address sparsity patterns.

**$\ell_1$ regularization vs. unregularized (Tables 1, 4).** On RCV1, $\ell_1$ regularization with AdaGrad reduces error rate compared to unregularized AdaGrad in some cases (the text notes "unregularized RDA and FOBOS attained similar results as did the $\ell_1$-regularized variants," but the specific unregularized AdaGrad numbers are omitted from tables). On Census Income, $\ell_1$ AdaGrad-RDA at full training achieves 4.9% error versus 4.7% for unregularized AdaGrad-RDA—a slight accuracy loss for a massive sparsity gain (3.7% vs. 100% non-zero). The regularization is thus trading a small amount of accuracy for substantial model compression, with the tradeoff controlled by $\lambda$.

**Mixed-norm regularization ($\ell_1/\ell_2$ vs. $\ell_1/\ell_\infty$) on MNIST.** The paper reports that both mixed-norm variants behave similarly ("$\ell_1/\ell_\infty$ is similar," §6.3), though only $\ell_1/\ell_2$ results are shown in Table 3. This is a minimal robustness check suggesting the framework's mixed-norm handling is not sensitive to the choice of inner norm.

**Hinge vs. logistic loss (RCV1).** All twelve algorithm variants (RDA, FOBOS, AdaGrad-RDA, AdaGrad-FOBOS, PA, AROW, each with hinge and logistic loss) were tested on RCV1. Results are reported only for hinge loss with the note that "results for both hinge and logistic losses are qualitatively and quantitatively very similar" (§6.1). This is a useful robustness check showing that AdaGrad's benefits are not tied to a specific loss function, though the actual logistic loss numbers are not presented.

**Lazy vs. eager sparse updates (implementation detail).** The paper does not ablate the lazy update scheme explicitly—it is presented as the natural implementation of the derived algorithms for sparse data. No comparison of lazy vs. eager update performance or runtime is provided. The correctness of lazy updates is derived mathematically from the update equations, not tested empirically.

**Missing ablations.** Several comparisons that would have strengthened the experimental analysis are absent:

1. **Diagonal AdaGrad with $H_t = G_t$ (no square root).** The theoretical analysis strongly motivates the square root (via the hindsight optimization and Lemma 4), but empirical confirmation that the square root is better than using $G_t$ directly would validate the theory. This is the most natural ablation of the core design choice.

2. **AdaGrad with exponential moving average rather than cumulative sum.** Later adaptive methods (RMSProp, Adam) would use EMA for gradient accumulation in non-stationary settings. Testing EMA on these stationary online learning tasks would show whether the cumulative sum is essential or if similar performance could be achieved with a simpler update.

3. **Comparison against a per-coordinate $\eta/\sqrt{t}$ schedule.** A naive per-coordinate adaptation—using step size $\eta/\sqrt{t_i}$ where $t_i$ is the number of times feature $i$ has appeared—would be a much simpler baseline than full AdaGrad. Comparing against this would isolate whether AdaGrad's $\|g_{1:t,i}\|_2$ aggregator (which weights by gradient magnitude, not just presence) provides additional value beyond simple frequency-based scaling.

4. **Ablation of $\delta$.** The $\delta$ parameter in $H_t = \delta I + \text{diag}(G_t)^{1/2}$ ensures positive definiteness, but its practical impact is not studied. The text notes "in practice $\delta$ can be set to 0" (§1.1), but no experiments confirm this or test sensitivity to small $\delta$ values.

5. **Run-time or throughput comparisons.** None of the experiments report wall-clock time, throughput (examples per second), or memory usage. The computational advantage of the diagonal variant over AROW's $O(d^2)$ covariance is discussed theoretically but not measured empirically.

---

### Critical Assessment

The paper makes several central claims, and the experimental support for each varies in strength. A rigorous assessment must distinguish what the experiments actually demonstrate from what the theory promises.

**Claim: ADAGRAD outperforms non-adaptive subgradient methods.**

The evidence for this claim is consistent and strong across diverse datasets. On RCV1 text classification (Table 1), AdaGrad-RDA achieves error rates 0.7–1.1 percentage points lower than non-adaptive RDA across all four categories (e.g., 4.4% vs. 5.1% on ECAT, 5.3% vs. 6.4% on CCAT). On MNIST (Table 3), the gap is larger: 6.6% vs. 10.8% test error without regularization—a 4.2 percentage point advantage. On ImageNet (Table 2), AdaGrad-RDA's average precision of 0.6022 dominates RDA's 0.5042 across 15,000 ranking tasks. On Census Income (Table 4), the advantage is smaller (4.7% vs. 5.0% at full training) but consistent across all training proportions. The improvement is observed in both sparse feature settings (RCV1, ImageNet, Census) and dense settings (MNIST with Gaussian kernels), and with both binary and multiclass losses, providing strong evidence of generality within the domain of linear online learning.

However, **the improvement over PA is marginal in some settings**. On MNIST without regularization, PA achieves 6.2% error versus AdaGrad-RDA's 6.6%—AdaGrad is actually slightly worse. On Census Income, PA matches or beats AdaGrad-RDA at several training proportions (e.g., 5.5% vs. 5.3% at 5% training). On RCV1, PA is clearly worse than AdaGrad (5.9% vs. 4.4% on ECAT). So the claim "outperforms non-adaptive methods" holds over the non-adaptive *subgradient* methods (RDA, FOBOS), but the comparison with PA—which is also non-adaptive but uses an aggressive update—is dataset-dependent. The paper's theoretical framework does not predict when PA would be competitive, as PA optimizes a different objective (constrained optimization with margin) than the regret minimization framework AdaGrad targets.

**Claim: ADAGRAD outperforms AROW.**

This claim is inconsistent across experiments. On RCV1 (Table 1), AdaGrad-RDA achieves lower error rates than AROW on all four categories, with the largest gap on CCAT (5.3% vs. 6.1%). On ImageNet (Table 2), the comparison is metric-dependent: AdaGrad wins on average precision (0.6022 vs. 0.5813) but loses on P@1 (0.8502 vs. 0.8597) and P@3 (0.8307 vs. 0.8369). On Census Income (Table 4), AROW wins across all training proportions (e.g., 4.4% vs. 4.7% at full training). So the claim "AdaGrad outperforms AROW" is true for text classification, mixed for image ranking, and false for census income prediction. The paper does not adequately characterize these conditions—it presents the RCV1 result prominently in Table 1 while downplaying the Census Income result where AROW is superior. A more accurate statement would be: "AdaGrad outperforms AROW on high-dimensional sparse text tasks; AROW outperforms AdaGrad on medium-dimensional dense demographic tasks." This pattern is consistent with the diagonal-vs-full covariance distinction: on tasks where features are approximately uncorrelated (sparse text bigrams), the diagonal approximation captures most of the relevant structure, while on tasks with substantial feature correlations (demographic interactions), AROW's full covariance provides additional benefit that AdaGrad's diagonal approximation misses. The theoretical framework predicts this (the gap between the diagonal and full-matrix infimal bounds in Corollaries 6 and 11), but the paper does not experimentally verify this interpretation—for instance, by measuring feature correlations on Census Income or testing the full-matrix AdaGrad variant on a reduced feature set.

**Claim: ADAGRAD produces sparse solutions with excellent accuracy.**

The evidence strongly supports this claim. On RCV1, Figure 7 shows that AdaGrad with $\ell_1$ regularization matches AROW's dense accuracy using only 1% of features. On MNIST (Table 3), AdaGrad with $\ell_1/\ell_2$ achieves 13.7% error using only 14.4% of rows, comparable to unregularized RDA's 10.8% error using 100% of rows. On Census Income (Table 4), $\ell_1$ AdaGrad achieves 4.9% error with 3.7% non-zero features versus AROW's 4.4% error with 100% non-zero. The sparsity-accuracy curves in Figure 7 provide a complete picture of this tradeoff, showing that AdaGrad identifies a "core set" of informative features with high precision. The mechanism—per-coordinate adaptation giving high learning rates to rare informative features while allowing $\ell_1$ to zero out common noisy features—is consistent with the theory and empirically validated by the sparsity patterns.

However, an important caveat: sparsity is achieved through explicit $\ell_1$ regularization, not automatically by AdaGrad itself. AdaGrad without $\ell_1$ always produces dense solutions (100% non-zero in Tables 1, 3, 4). The contribution is not that AdaGrad generates sparsity on its own, but that its adaptive learning rates interact with $\ell_1$ regularization to produce *better* sparsity-accuracy tradeoffs than non-adaptive methods with the same $\ell_1$ penalty. This is visible in the comparison: at $\lambda = 10^{-3}$ on MNIST (Table 3), AdaGrad uses 14.4% non-zero rows versus RDA's 53.2% at the same $\lambda$, while achieving lower error (13.7% vs. 19.2%). The adaptation effectively "helps" the regularizer make better decisions about which features to keep.

**Unaddressed questions and experimental gaps.**

1. **The full-matrix AdaGrad variant is theoretically motivated but never tested.** Section 4 and Theorem 7 develop the full-matrix analysis in detail, but no experiment evaluates it, even on the moderate-dimensional Census Income task (d = 4001) where it would be computationally feasible ($O(d^3) \approx 6.4 \times 10^{10}$ operations per round is expensive but possible for a 199,523-example dataset with modern hardware). Testing full-matrix AdaGrad on Census Income and comparing against AROW (which also maintains a full matrix) would directly test whether the square-root construction ($G_t^{1/2}$) is better than AROW's inverse-covariance construction ($\Sigma_t$), as the theory suggests. The absence of this experiment is a significant gap.

2. **The lazy update scheme's computational benefit is claimed but not measured.** The paper describes lazy updates as enabling "time proportional to the support of the gradient" (Section 1.1) and derives the functional form in Section 5.1, but no runtime, throughput, or complexity measurements are reported. For the RCV1 dataset with $d \approx 2 \times 10^6$ and per-example non-zeros below 5,000, the claimed speedup of roughly $d / \text{nnz}(g_t) \approx 400\times$ over eager updates is substantial enough to merit empirical confirmation.

3. **Only a single pass through the data is performed.** The experiments follow the pure online learning protocol: one example at a time, one pass through the training set. There is no investigation of what happens with multiple passes (epochs), which is standard practice in deep learning and would test whether AdaGrad's cumulative gradient norms ($\|g_{1:t,i}\|_2$) continue to provide benefit when gradients repeat across epochs. In multi-epoch training, the cumulative norms would grow without bound, potentially driving step sizes to zero too aggressively—a known limitation of AdaGrad that motivated later variants like RMSProp and AdaDelta.

4. **All tasks are linear prediction problems.** AdaGrad is evaluated exclusively on linear classifiers and rankers. The theoretical framework applies to any convex composite objective, including non-linear models like kernel methods, generalized linear models with non-linear link functions, or matrix factorization. Testing on at least one non-linear convex problem would strengthen the claim of generality.

5. **Hyperparameter sensitivity is not systematically studied.** The step size $\eta$ is cross-validated per experiment, but no sensitivity analysis (varying $\eta$ around the optimum and measuring performance degradation) is reported. The $\delta$ parameter is not tuned at all—the paper says $\delta$ can be set to 0 in practice but does not verify this claim with experiments. The theoretical bounds provide guidance for setting $\eta$ ($\eta = D_\infty/\sqrt{2}$ for mirror descent, $\eta = \|x^*\|_\infty$ for RDA), but these theoretically-motivated values are not compared against the cross-validated values in practice.

6. **Statistical significance is underreported.** While variance is mentioned for some experiments ("on the order of $10^{-6}$" for ImageNet and Census Income), no confidence intervals, standard errors, or hypothesis tests are reported. The small differences between top-performing methods (e.g., AdaGrad-RDA at 4.4% vs. AROW at 4.9% on ECAT) could potentially fall within sampling noise, though the consistency across categories and independent shuffles argues against this. For MNIST, where the gap is much larger (6.6% vs. 10.8%), statistical significance is not in question, but the precise ranking of AdaGrad vs. PA (6.6% vs. 6.2%) might be.

**What the experiments demonstrate versus what they claim.**

The experiments convincingly demonstrate that **per-coordinate adaptive learning rates based on cumulative gradient $\ell_2$ norms improve the accuracy of linear online learners across diverse tasks**, especially in high-dimensional sparse settings. They also demonstrate favorable **sparsity-accuracy tradeoffs** when combined with $\ell_1$ regularization. These are the paper's core empirical contributions, and the evidence is solid.

However, the experiments do not directly validate the paper's deepest theoretical claim: that AdaGrad's regret bound is **competitive with the best proximal function chosen in hindsight**. This claim is a theorem—it does not require experimental validation—but the experiments do not attempt to verify that the observed regret (cumulative online loss) matches the theoretical bounds, or that the difference between AdaGrad's regret and the "best in hindsight" regret is small in practice. The experiments measure test set error after a single pass, not online regret against the optimal fixed predictor. While low test error is the practical goal, it does not directly confirm the regret bound tightness.

Similarly, the claim that AdaGrad provides "regret guarantees that are provably as good as the best proximal function that can be chosen in hindsight" (Abstract) is a theoretical result, not an experimental one. The experiments show that AdaGrad works well compared to reasonable baselines (RDA, FOBOS, PA, AROW), but they do not compute the actual best proximal function in hindsight and compare against it. Such a comparison would be informative—for instance, on a small problem, one could compute the batch-optimal diagonal $s^*$ from all training gradients and compare test accuracy to online AdaGrad's test accuracy—but it is not performed.

Finally, the paper makes a conceptual connection to the intuition that "infrequently occurring features are highly informative and discriminative" (§1) and that AdaGrad helps by giving them higher learning rates. The experimental results are consistent with this intuition—AdaGrad excels on sparse data where feature frequencies vary widely—but no experiment directly measures whether AdaGrad's learning rate adaptation specifically benefits rare discriminative features versus simply providing better-calibrated step sizes in a more general sense. The sparsity-accuracy curves (Figure 7) show that the features AdaGrad retains at high sparsity levels (1% non-zero) are highly predictive, but the paper does not characterize these features (e.g., are they actually the rare bigrams the intuition highlights, or something else?). This leaves a gap between the motivating intuition and the empirical validation.

## 6. Limitations and Trade-offs

### The Difficulty Estimation Cost Is Not Accounted For in the Efficiency Gains

**The assumption or constraint.** The paper's headline result—that compute-optimal test-time scaling achieves more than `4×` greater efficiency than best-of-N sampling—is computed *after* prompt difficulty has been estimated, without including the cost of that estimation in the budget. Difficulty estimation as implemented requires generating **2048 complete solutions per question** and then either checking correctness against ground-truth answers (oracle difficulty) or scoring with the PRM (predicted difficulty). The authors acknowledge this directly in Section 3.2:

> "estimating difficulty in this way still incurs additional computation cost during inference... our experiments do not account for this cost largely for simplicity"

A single difficulty estimation for one question consumes **2048 generations**—far exceeding the entire test-time compute budgets studied in most experiments (which range from 1 to 512 generations). Once this one-time cost is paid for a given question, subsequent applications of compute-optimal strategies can amortize it, but for one-shot or low-volume inference, the estimation dominates the total cost.

**The consequence.** The reported `4×` efficiency gains are an **upper bound that cannot be realized in deployment without amortization** over many repetitions of the same question. For applications where each prompt is unique and seen only once—the typical online learning or single-query setting—the total cost (difficulty estimation + strategy execution) would be **far worse than simply running best-of-N with the same budget**. The exploration-exploitation framework the paper invokes in Section 3.2 makes this tradeoff explicit, but the paper provides no guidance on how many queries are needed to amortize the estimation cost, no mechanism for reducing the 2048-sample requirement, and no experiment measuring total cost (estimation + execution).

**What evidence exists in the paper.** Figures 4 and 8 show that predicted difficulty bins (using the PRM's own scores, without ground-truth labels) track the oracle difficulty curves closely, confirming that ground-truth labels are not needed. However, this does **not** reduce the sample cost—predicted difficulty still requires 2048 generations per question for PRM scoring. No experiment varies the number of difficulty-estimation samples to determine the minimum needed for reliable bin assignment, nor compares total cost (estimation + execution) against baselines. The gap between the `4×` claim and deployable reality is entirely unmeasured.

**Mitigation status.** The paper explicitly calls this out as an avenue for future work in Section 8: "pretraining or finetuning models to directly predict difficulty of a question." An alternative approach mentioned in Section 3.2 is adaptive difficulty estimation—start with a few samples, assess difficulty, and allocate the remainder accordingly—but this is not implemented or evaluated. No mitigation is provided in the current paper; the cost is simply excluded from all efficiency calculations.

---

### The `~14×` Larger Model Baseline Is Not Compute-Optimal and Uses Only Greedy Decoding

**The assumption or constraint.** The FLOPs-matched comparison in Section 7 compares PaLM 2-S\* with compute-optimal test-time scaling against a model with approximately `14×` more parameters. The larger model is **trained by scaling parameters only, holding data fixed**—following the LLaMA paradigm of Touvron et al. (2023). This departs from **compute-optimal pretraining** as characterized by Hoffmann et al. (2022), where both parameters and training data are scaled jointly. The authors acknowledge this choice:

> "We choose this setting as it is representative of a canonical approach to scaling pretraining compute and leave the analysis of compute-optimal scaling of pretraining compute where the data and parameters are both scaled equally to future work."

Furthermore, the larger model is evaluated using only **greedy decoding**—no majority voting, no best-of-N, no search, and no revision chain. This means the comparison is between *compute-optimal inference + small model* and *no inference compute optimization + large model*. The larger model receives none of the test-time strategies that the paper itself demonstrates are beneficial.

**The consequence.** The reported advantages of test-time compute over pretraining—such as `+27.8%` relative improvement on easy-to-medium questions at low inference-to-pretraining ratios (`R ll 1`)—are **inflated relative to what a practitioner would achieve** by simply spending the same pretraining budget on a properly compute-optimal larger model and then applying even a modest amount of test-time compute to it. A Chinchilla-optimal model trained with `14×` more total FLOPs (scaling both parameters and data) would likely outperform a parameter-only-scaled model. Giving that larger model even best-of-8 majority voting at test time—a trivial addition—would further close the gap. The paper's central finding that test-time compute can substitute for pretraining compute is therefore **conditional on a specific, suboptimal pretraining recipe**, and the magnitude of the effect may not generalize to compute-optimal training regimes.

**What evidence exists in the paper.** Figure 9 and Figure 1 (right bar charts) show the `14×` comparison results. The x-axis positions where the larger model's accuracy is plotted (stars) would shift upward if the baseline were strengthened. No ablation in the paper compares against a compute-optimally trained larger model, gives the larger model any test-time compute budget, or quantifies how much of the observed gap is due to the baseline weakness versus the test-time strategy's genuine merit. The paper does not even report results for the `14×` model with simple best-of-N or majority voting, which would cost minimally to add to the experimental protocol.

**Mitigation status.** The authors transparently acknowledge the parameter-only scaling choice and leave the compute-optimal pretraining comparison to future work. However, they do not characterize this as a significant limitation that **weakens the strength of their headline finding**—the FLOPs-matched conclusion is presented as a positive result for test-time compute (Section 7 takeaway box) without the caveat that the baseline is deliberately weaker than what an informed practitioner would deploy. The greedy decoding choice for the larger model is not discussed as a limitation at all.

---

### Hard Problems Remain Unsolved—Test-Time Compute Cannot Compensate for Insufficient Base Model Capability

**The assumption or constraint.** The paper's approach assumes that the base model already produces correct solutions at some non-trivial rate for the problems being optimized—test-time compute can **amplify** existing capability but cannot **create** it from nothing. This limitation is starkly visible in the difficulty bin analysis: for the hardest questions (difficulty bin 5, where the base model's pass@1 is near zero), no amount of test-time compute produces meaningful improvement, regardless of strategy or budget.

**The consequence.** For any problem that genuinely exceeds the base model's training distribution or reasoning capabilities, the entire compute-optimal framework provides **zero benefit**. This is not a small-edge-case limitation—it means the approach applies only to problems where the base model already "knows" the answer some fraction of the time and the challenge is reliably surfacing that knowledge. For frontier reasoning tasks, out-of-distribution generalization, or novel problem types, scaling test-time compute is ineffective and pretraining remains the only viable path. The paper's framework offers no diagnostic or decision rule for distinguishing problems where test-time compute will help (bins 1-4) from those where it will not (bin 5) *before* investing the difficulty estimation cost, creating a risk of wasted computation.

**What evidence exists in the paper.** The evidence is overwhelming and consistent across all methods:

- **Figure 3, right (search):** Bin 5 accuracy remains at 1-3% across all budgets from 4 to 256 generations, for both beam search and best-of-N. The curves are essentially flat and indistinguishable from random guessing.
- **Figure 7, right (revisions):** Bin 5 accuracy is approximately 2-3% for all sequential-to-parallel ratios at a budget of 128 generations. No allocation strategy helps.
- **Figure 9 (FLOPs-matched):** The bin 5 scaling line is essentially flat near 0-5% for both revisions and PRM search. At `R gg 1`, hard questions show a `−52.9%` relative disadvantage for PRM search versus the larger model (Figure 1, bottom-right bar chart). Even the larger pretrained model performs poorly on bin 5, indicating fundamental difficulty.

The paper is transparent about this finding in Section 7, stating that "test-time compute provides essentially zero benefit regardless of budget" on the hardest problems, but frames this as a boundary condition rather than a limitation of the overall framework.

**Mitigation status.** The paper does not attempt to mitigate this limitation—it is treated as a fundamental constraint of the approach. There is no investigation of whether the difficulty boundary shifts with model scale (i.e., would a larger base model push some bin 5 problems into bin 4, where test-time compute becomes effective?), no hybrid strategy that escalates bin 5 problems to a larger model, and no attempt to characterize what makes a problem "bin 5" in terms of the knowledge or reasoning skills required versus what the base model possesses. The difficulty bins are defined purely statistically (pass@1 rate from 2048 samples) without interpretable features that could guide deployment decisions.

---

### Single Model Family and Single Benchmark—Generality of Findings Is Unverified

**The assumption or constraint.** All experiments use a single base model (PaLM 2-S\*) evaluated on a single benchmark (MATH, 500 test questions). The authors state their belief that this model "is representative of the capabilities of many contemporary LLMs" (Section 4), but provide no evidence for this claim. The model's particular failure modes, calibration properties, output distribution, and base pass@1 rate on MATH all influence the observed difficulty-dependent scaling patterns, the PRM's over-optimization behavior, and the revision model's training quality.

**The consequence.** Several findings may be model-specific and not generalizable:

- **The PRM over-optimization threshold** (where beam search starts to hurt on easy problems, Figure 3 right) depends on the verifier's calibration relative to the base model's error modes. A model with different calibration properties—e.g., one whose outputs the PRM scores more accurately—might exhibit a different or higher over-optimization threshold, changing the compute-optimal policy.
- **The revision model's 38% reversion rate** (correct answers being revised to incorrect) depends on the base model's in-context learning ability and the specific revision training data construction. Different model families might exhibit substantially different revision behavior.
- **The difficulty-dependent optimal strategy** (beam search for medium problems, best-of-N for easy, balanced sequential-parallel for hard) is characterized by five bins computed from PaLM 2-S\*'s pass@1 distribution. The bin boundaries, the optimal strategy per bin, and even whether five bins are the right discretization could change with a different base model.
- **MATH consists exclusively of competition-level math problems** requiring symbolic reasoning and multi-step deduction. It is unknown whether the framework's central finding—that difficulty-conditional allocation provides `4×` efficiency gains—transfers to other reasoning domains (code generation, logical reasoning, scientific QA) or to tasks requiring factual recall rather than inference.

**What evidence exists in the paper.** The paper provides no cross-model or cross-benchmark experiments whatsoever. The test set is 500 questions, split into five difficulty quintiles of approximately 100 each, further split by two-fold cross-validation for strategy selection—meaning the compute-optimal policy is **selected based on roughly 50 questions per fold per bin**. The paper reports no confidence intervals on the compute-optimal scaling curves in Figures 4 and 8, making it impossible to assess whether the observed differences between strategies at specific budgets are statistically reliable at this sample size. The predicted difficulty bins (using PRM scores) closely track oracle bins (Figures 4, 8), but this validates the *proxy*, not the *generality* of the difficulty-dependent patterns to other models or tasks.

**Mitigation status.** The authors do not address this limitation beyond the brief claim of representativeness in Section 4. No future work is suggested regarding cross-model or cross-benchmark replication. The paper implicitly treats the PaLM 2-S\* results as establishing general principles (difficulty-conditional allocation, verifier over-optimization as bottleneck) rather than model-specific observations, but the empirical foundation for this generalization is absent.

---

### Verifier Over-Optimization Is Documented But Not Solved—The Scaling Ceiling Is Hard

**The assumption or constraint.** The paper identifies verifier over-optimization as the central bottleneck limiting further scaling of test-time compute: beam search performance degrades on easy problems at high budgets (Figure 3, right), lookahead search—the most powerful optimizer—paradoxically performs **worst overall** (Figure 3, left), and qualitative examples in Appendix M show search producing degenerate outputs (repetitive low-information steps, overly short 1-2 step solutions) that score highly under the PRM but are incorrect. The compute-optimal allocation policy mitigates over-optimization by routing easy problems away from aggressive search methods toward best-of-N, but it does **not solve the underlying problem**—it is a workaround, not a fix.

**The consequence.** On medium-difficulty problems where beam search is actively deployed by the compute-optimal policy, over-optimization still limits the scaling ceiling. The beam search curves in Figure 3 (right) **flatten and eventually decline** as the budget increases, even on the difficulty bins where they outperform best-of-N. This means additional compute eventually becomes counterproductive regardless of the allocation strategy. The scaling curves for compute-optimal search and revisions in Figures 4 and 8 show continued improvement at the highest budgets tested (256-512 generations), but the rate of improvement is **sublinear and decelerating**—it is unclear whether they would plateau at higher budgets not tested in the paper.

Moreover, the verifier over-optimization threshold depends on the PRM's quality, which in turn depends on the training procedure (Monte Carlo rollouts from PaLM 2-S\*). A different PRM training recipe—more on-policy data, adversarial examples, ensembling—might shift the threshold and change the optimal allocation policy, but since the paper treats verifier quality as fixed, **all reported results are conditional on this specific PRM's reliability curve**. The compute-optimal policy is optimal *given this verifier*, not optimal in any verifier-independent sense.

**What evidence exists in the paper.** The evidence for over-optimization is one of the best-documented findings:

- **Figure 3 (right), easy bins:** Beam search accuracy *decreases* with increasing budget (bin 1 drops from ~78% at 4 generations to ~77% at 256 generations), while best-of-N continues to improve. This is the clearest signature of verifier exploitation—search finds solutions that score highly under the PRM but are actually wrong.
- **Figure 3 (left):** Lookahead search with `k = 3` steps underperforms all other methods at the same generation budget. The extra optimization power (deeper lookahead for more accurate step scoring) paradoxically reduces accuracy because it more aggressively exploits PRM errors.
- **Appendix M (qualitative examples):** Repetitive low-information steps and overly short solutions that trick the PRM into high confidence but are substantively incorrect.
- **No experiment measures verifier calibration as a function of optimization intensity.** The paper documents the *symptoms* of over-optimization but does not quantify the relationship between search budget and PRM calibration, leaving the scaling ceiling empirically uncharacterized beyond the observed plateau in Figure 3.

**Mitigation status.** Section 8 identifies "the key bottleneck for further scaling test-time compute [as] improving verifier robustness and reliability." The paper suggests adversarial training of verifiers, ensemble methods, and constrained search with KL penalties as future directions, but implements none of these. The compute-optimal policy is presented as the primary practical contribution, but it is fundamentally a **mitigation of a symptom** (avoiding the over-optimization regime per difficulty level) rather than a **solution to the root cause** (verifier miscalibration under optimization pressure). Practitioners adopting this framework inherit the verifier quality limitation, and any improvements to the PRM would require recomputing the entire compute-optimal policy.

---

### Revisions and Search Are Studied Independently—The Complementarity Claim Is Not Empirically Validated

**The assumption or constraint.** The paper's theoretical framework (Section 2) decomposes test-time compute methods into two complementary axes: modifying the **proposal distribution** (revisions) and modifying **output selection** (search against a verifier). The difficulty-dependent analysis shows these axes have complementary strengths—revisions excel on easy problems where local refinement suffices, search excels on medium problems where global exploration is needed. However, **the two mechanisms are never combined** in any experiment. PRM tree-search is always applied to the base PaLM 2-S\* model, and revision chains are always evaluated with final-answer selection (majority or verifier) but without step-level PRM guidance during the revision process.

Section 8 explicitly acknowledges this gap:

> "we did not experiment with PRM tree-search techniques in combination with revisions"

**The consequence.** The paper's vision of a unified system—where a revision model serves as the proposal distribution within a PRM-guided search tree, or where the PRM scores individual revision steps to decide when to continue versus restart—is entirely **unrealized**. The empirical results for search and revisions represent **lower bounds** on what a combined approach might achieve. The difficulty-dependent complementarity patterns (Figures 3 right, 7 right) are suggestive that a combined system could outperform either alone—beam search could give the revision model better candidate diversity on medium problems, while sequential refinement could correct errors that slip past the PRM's step-level scoring—but this hypothesis is untested.

Furthermore, the difficulty-dependent optimal policy *combining* search and revision strategies (e.g., using revisions on easy problems, beam search on medium, a hybrid on hard) is never computed or evaluated. The compute-optimal curves in Figures 4 and 8 are for search-only and revisions-only policies respectively, leaving open the question of how much additional gain a joint policy would provide.

**What evidence exists in the paper.** None. No experiment combines the two approaches. The revision model's ORM (Appendix J, Figure 15a) is shown to outperform the base-model PRM on revision outputs, confirming the importance of distribution-matched verifiers, but this is a separate concern from combining PRM tree-search with revisions. The paper's structural claim—that the proposal distribution and verifier are complementary axes—is supported indirectly by the difficulty-dependent results (different axes help on different problems), but the direct test of complementarity (combining them yields gains beyond either alone) is absent.

**Mitigation status.** The authors acknowledge the gap in Section 8 and list it as natural future work. The paper provides the theoretical scaffolding (Section 2 framework) and the individual building blocks (PRM, search algorithms, revision model training), but stops short of integration. A practitioner seeking to deploy a combined system would need to redesign the interaction between revisions and search—how should the PRM score revision steps that condition on previous incorrect answers? Should the search tree branch on different revision directions or on different completions from a single revision?—without guidance from the paper.

## 7. Implications and Future Directions

### How This Work Changes the Landscape

This paper caused a conceptual shift in online learning and stochastic optimization by elevating the proximal function from a **static design choice** to a **data-dependent state variable that is optimized online** during learning. Before ADAGRAD, the standard approach was to select a proximal function—typically $\psi(x) = \frac{1}{2}\|x\|_2^2$ for Euclidean problems or the negative entropy for simplex constraints—before seeing any data and hold it fixed for the entire run. Even methods that varied the proximal function over time (such as RDA's $\psi_t = \sqrt{t}\psi$) applied the variation uniformly to all coordinates, preserving the fundamental geometry of the initial choice. ADAGRAD demonstrated that the proximal function could instead be constructed incrementally from the outer product matrix of observed subgradients, with formal regret bounds proving that this online construction is **competitive with the best proximal function that could have been chosen in hindsight with full knowledge of all gradients**.

This is an **incremental refinement with outsized practical consequences** rather than a paradigm shift. The theoretical framework—online convex optimization with regret bounds—remains intact. The update rules (primal-dual subgradient, composite mirror descent) are inherited from prior work by Nesterov (2009), Xiao (2010), and Duchi et al. (2010). The contribution is not a new algorithm class but a **new principle for configuring an existing algorithm class**: the proximal term should be a learnable object driven by the same gradient outer products that appear in the regret decomposition. This principle is compact, general, and leads to specific implementable algorithms (the diagonal and full-matrix variants) without case-by-case derivation for each new regularizer or constraint set.

The paper also **reconciles a tension in the literature** between practitioners and theorists. Practitioners had long observed that infrequent features deserve higher learning rates—manually implementing this via TF-IDF weighting (Salton and Buckley, 1988) or ad-hoc per-coordinate step sizes. Theorists, working within the minimax framework (Zinkevich, 2003; Abernethy et al., 2008), proved tight worst-case bounds using global step sizes and uniform dual norms that treat all coordinates identically. ADAGRAD bridges this gap by providing algorithms that **automatically implement the practitioner's intuition** (high rates for rare features, low rates for common ones) while **retaining the formal regret guarantees** that theorists demand, with bounds that explicitly improve over the worst-case when gradients are sparse.

**Research directions that become more attractive** after this work:

- **Online metric learning for optimization.** The paper's meta-learning framing—treating the proximal function as an object to be optimized—opens the door to learning more complex metrics than diagonal or full quadratic forms. Could the proximal function be a neural network? A kernel-induced metric? A learned Mahalanobis distance that adapts to task structure in meta-learning settings? The analysis in Sections 3-4 provides a template: define the hindsight-optimal metric, bound the online approximation cost via telescoping Bregman differences, and verify that the adaptation penalty is self-bounding.

- **Second-order methods for non-convex stochastic optimization.** The full-matrix variant (Section 4, Theorem 7) captures gradient correlations through $G_t^{1/2}$, but is computationally prohibitive beyond a few thousand dimensions. Developing low-rank or randomized approximations to $G_t^{1/2}$ that preserve the regret guarantees up to controlled error would extend ADAGRAD-style adaptation to modern deep learning, where gradient covariances exhibit low-rank structure. The block-diagonal extension sketched in Corollary 12 provides a natural starting point.

- **Adaptive regularization in the composite setting.** The paper's treatment of composite objectives $\varphi_t = f_t + \phi$ as a first-class design principle (Section 5) separates the geometry of optimization ($\psi_t$) from the geometry of solutions ($\phi$). This modularity suggests studying how $\psi_t$ and $\phi$ interact: can we adapt $\phi$ online as well, perhaps by increasing the $\ell_1$ penalty for coordinates where $\|g_{1:t,i}\|_2$ has plateaued, indicating the coordinate has converged?

**Research directions that become less central:**

- **Pure scalar step-size adaptation.** Prior work on adapting the global learning rate $\eta_t$ based on function variation (Hazan and Kale, 2008; Bartlett et al., 2007) focused on the *scalar* step size while applying it uniformly. ADAGRAD demonstrates that **per-coordinate adaptation provides substantially larger gains** in the high-dimensional sparse settings that dominate practice. Future work on step-size adaptation would need to show it captures something that per-coordinate methods miss to justify the additional complexity over simply using ADAGRAD.

- **Fixed-geometry proximal methods without adaptation.** The experimental results (Tables 1-4, Figure 5) consistently show ADAGRAD matching or outperforming non-adaptive RDA and FOBOS across diverse datasets, often by large margins (e.g., 6.6% vs. 10.8% test error on MNIST, Table 3). For practitioners, there is now **little reason to prefer a non-adaptive proximal method** over its ADAGRAD counterpart unless the per-coordinate bookkeeping is infeasible—which, given the $O(d)$ memory and $O(\text{nnz}(g_t))$ per-iteration cost of the diagonal variant, is rarely the case.

---

### Follow-Up Research This Work Enables

**1. Cheap difficulty estimation via direct prediction or amortized inference.** The paper's compute-optimal framework depends on estimating prompt difficulty before allocating the test-time budget. The current method—generating 2048 samples per question and scoring with the PRM—is so expensive (§3.2) that it consumes more compute than the largest budgets studied. Two concrete approaches would close this gap. **First**, train a lightweight difficulty classifier that takes only the question text as input and predicts the difficulty bin, using the 2048-sample PRM bins as training labels, then evaluate whether the predicted bins recover the `4×` efficiency gains in Figures 4 and 8. **Second**, implement adaptive difficulty estimation: generate a small initial batch of 4-8 samples, compute the PRM's average final-answer score on those samples as a cheap difficulty proxy, then allocate the remaining budget accordingly. Compare total cost (initial batch + adaptive strategy) against the best-of-N baseline at equivalent total generations. The second approach is particularly attractive because it amortizes difficulty estimation into the problem-solving process itself.

**2. Joint search-revision systems with PRM-guided revision trajectories.** The paper demonstrates complementary difficulty-dependent strengths for PRM search (helps on medium problems, Figure 3 right) and sequential revisions (helps on easy problems, Figure 7 right) but never combines them. A concrete experiment: use the revision model as the proposal distribution within beam search. At each step of the search tree, the revision model conditions on its previous rejected branches as in-context incorrect answers. The PRM scores each candidate step, and beam search prunes low-scoring partial solutions. Compare three configurations on the MATH test set across all difficulty bins: (a) beam search with base model + PRM, (b) sequential revisions with verifier selection, (c) the combined system. Test the combined system at budgets from 4 to 256 generations and compute the *joint* compute-optimal policy (optimal combination strategy per difficulty bin). The hypothesis—that the combined system outperforms either alone on medium-difficulty problems while maintaining the easy-problem performance of pure revisions—would directly test the paper's central claim that the proposal distribution and verifier are complementary axes (Section 2).

**3. Verifier robustness benchmarks under controlled optimization pressure.** The paper identifies verifier over-optimization as the primary bottleneck (Figure 3 right: beam search degrades on easy problems; Figure 3 left: lookahead search performs worst overall), but only documents the symptoms, not the mechanism. A targeted study would systematically measure PRM calibration as a function of optimization intensity. Specifically: train PRMs using different procedures—Monte Carlo rollouts (as in this paper), human labels (Lightman et al., 2023), adversarial training where the PRM sees search-generated solutions during training, ensembling multiple independently trained PRMs—and evaluate each PRM's accuracy at predicting final-answer correctness under increasingly aggressive beam search (varying beam width from 2 to 64). Measure both the *discrimination* (AUROC for correct vs. incorrect solutions) and *calibration* (expected calibration error) as beam width increases. The strongest PRM is the one whose discrimination and calibration degrade *least* under aggressive optimization. This study would directly inform whether the research priority should be better search algorithms (which the paper's results suggest is ineffective) or better verifier training (which the over-optimization diagnosis points to).

**4. Cross-model and cross-benchmark replication of difficulty-dependent scaling patterns.** All results are on PaLM 2-S\* with MATH (500 test questions). The paper's central empirical contribution—that prompt difficulty is the key variable determining which test-time strategy works best—is therefore unverified across model families, scales, and reasoning domains. A replication study would test at least three model families (e.g., PaLM, LLaMA, and a smaller open-source model like Mistral 7B) on at least two benchmarks (MATH and a code generation benchmark like APPS or MBPP) using the identical experimental protocol: train a model-specific PRM via Monte Carlo rollouts, bin problems into five difficulty quintiles based on pass@1, evaluate beam search vs. best-of-N per bin, and compute the compute-optimal policy. The key question is whether the *qualitative patterns*—beam search helps on medium, hurts on easy; revisions help on easy; hard problems benefit from neither—replicate across models and domains, or whether they depend on PaLM 2-S\*'s specific error modes and MATH's specific reasoning demands. If the patterns replicate, the difficulty-conditional framework becomes a general principle rather than a model-specific observation. If they do not, the boundary conditions for the framework's applicability would be empirically characterized for the first time.

**5. Compute-optimal joint pretraining-inference allocation.** The FLOPs-matched comparison in Section 7 compares test-time compute with a smaller model against pretraining with a `~14×` larger model, but treats the pretraining recipe as fixed (parameter-only scaling, not Chinchilla-optimal). A complete picture would **jointly optimize** the pretraining configuration (model size, data quantity) and the inference strategy (search method, revision depth, difficulty-dependent allocation) under a total FLOPs constraint. Concretely: for a fixed total FLOPs budget, sweep over model sizes from 1× to 16×, for each model size allocate the remaining FLOPs to pretraining data (following Hoffmann et al., 2022 scaling) and inference compute (using the compute-optimal test-time policy for that model on the MATH benchmark), and measure test accuracy. The result would be a **three-way Pareto frontier** showing the optimal allocation of FLOPs across pretraining parameters, pretraining data, and inference compute as a function of total budget. This is substantially more ambitious than the current paper's analysis—it requires training multiple model sizes and evaluating compute-optimal inference for each—but it is the natural endpoint of the research direction the paper initiates. The paper's framework (efficient test-time strategies per model, FLOPs accounting) makes this tractable where it would have been infeasible before.

**6. Dynamic difficulty assessment and mid-computation strategy switching.** The paper's difficulty bins are static—computed once from 2048 samples per question—and the strategy is fixed for the entire allocation. A dynamic policy could adjust strategy mid-computation based on *early signals*: generate 4 parallel samples from the base model, score them with the PRM, and decide based on the score distribution whether to continue with parallel sampling (easy problems, where PRM scores are consistently high and agreement is high), switch to beam search (medium problems, where PRM scores show variance and exploitation can help), or switch to sequential revisions (hard-but-not-impossible problems, where initial answers are low-quality but refinement might help). This approach connects naturally to the multi-armed bandit and Bayesian optimization literatures that the paper invokes in Section 3.2. A concrete experiment: compare the static compute-optimal policy (as in Figures 4, 8) against a dynamic policy that uses the first 4 or 8 generations of a 64-generation budget to estimate difficulty, then allocates the remaining budget according to the estimated bin's optimal strategy. The metric is total accuracy at the 64-generation budget, with the dynamic policy's difficulty estimation cost *included* in the budget (unlike the current paper's evaluation, which excludes the 2048-sample estimation cost). This experiment would determine whether the `4×` efficiency gains survive when difficulty estimation is properly costed.

---

### Practical Applications and Downstream Use Cases

**1. Cost-efficient batch inference for evaluation and data generation pipelines.** Organizations running large-scale batch inference—evaluating thousands of math problems, generating training data for downstream models, or scoring candidate solutions—can adopt the compute-optimal framework to reduce costs by `4×` relative to a uniform best-of-N policy (Figures 4 and 8). The workflow: for each new batch of questions, invest a one-time cost of 2048 samples per question to estimate difficulty bins using the PRM's predicted final-answer scores (producing bins that track the oracle curve closely per Figure 4). Then, for all subsequent evaluations of the *same questions* (e.g., comparing different models, testing revised prompts, or generating multiple training trajectories), apply the pre-computed difficulty bin to select the optimal strategy—beam search for medium-difficulty problems, best-of-N for easy problems, parallel best-of-N for hard problems—at the desired accuracy level. The one-time difficulty estimation cost is amortized across repeated evaluations. For a batch of 10,000 questions evaluated under 5 different prompt variations, the estimation cost per question per evaluation is only ~410 generations (2048/5), and the per-evaluation savings from using 64 generations instead of 256 is 192 generations—a net savings once evaluations exceed approximately 11 per question. This use case is directly supported by the paper's data, without requiring any methodological extensions.

**2. On-device deployment of small models for routine queries with escalation to cloud for hard problems.** The FLOPs-matched results (Figure 9) show that on easy-to-medium problems, a small model with compute-optimal test-time scaling can match or exceed a `~14×` larger model. For applications where the query distribution is skewed toward routine problems (e.g., customer support chatbots handling common questions, educational tools working through standard problem sets), this suggests a tiered deployment architecture: a small on-device model handles most queries with variable test-time compute, and only genuinely hard queries—identified by the difficulty estimator from initial samples—are routed to a larger cloud-based model. The difficulty estimator here serves double duty: it both allocates the on-device compute budget and acts as a **routing gate**. The paper provides the components: the PRM-based difficulty estimator (Section 3.2), the compute-optimal strategies per difficulty bin (Figures 4, 8), and the FLOPs-matched analysis establishing the performance ceiling of the small model at each difficulty level (Figure 9). What is missing for deployment is a fast, low-sample method for estimating difficulty—the 2048-sample requirement is prohibitive for real-time routing—which is the primary engineering gap and the focus of follow-up direction #1 above.

**3. Data generation for self-improvement pipelines with targeted difficulty-based allocation.** When using LLMs to generate training data for fine-tuning (as in STaR, ReST$^{EM}$, or rejection sampling), the goal is to produce correct, high-quality solutions for as many training problems as possible within a fixed generation budget. The compute-optimal framework provides a principled allocation: spend the budget disproportionately on medium-difficulty problems (where search and revisions can push the model to produce correct solutions it would not find by random sampling alone), spend less on easy problems (where a few samples suffice to get correct solutions), and spend minimally on hard problems (where no amount of test-time compute helps, Figure 3 right, bin 5, and Figure 7 right, bin 5). For a self-improvement loop with a budget of `B` total generations across a training set of `N` problems, the optimal allocation is to estimate difficulty bins (paying the one-time 2048-sample-per-problem cost if problems are reused across iterations, or using cheap difficulty estimates if problems rotate), then allocate generations proportional to the *expected marginal gain in solution correctness* per additional generation, which the per-bin scaling curves (Figure 3 right, Figure 7 right) directly provide. This replaces the current practice—uniform allocation of generation budget across all problems—with a difficulty-gated policy that the paper shows can improve efficiency by up to `4×`.

**4. Verifier development as a research investment priority for organizations building reasoning systems.** The paper's finding that **verifier over-optimization is the primary bottleneck** for test-time compute scaling—not search algorithm sophistication—has direct implications for research resource allocation. Teams working on LLM reasoning systems should invest more in training robust process reward models (using Monte Carlo rollouts from their own model family, with soft labels, and potentially with adversarial training on search-generated solutions) than in developing complex search algorithms like MCTS or lookahead search. The paper provides evidence: lookahead search, the most powerful optimizer tested, paradoxically performs *worst* overall (Figure 3, left) because it more aggressively exploits PRM errors. Beam search with a modest width (`M = 4`) outperforms it at equivalent budget. The practical recipe for PRM training is provided in Section 5.1 and Appendix D: for each training question, sample 16 solutions from the base model, for each step in each solution sample 16 Monte Carlo rollouts, compute the fraction that reach the correct answer as a soft label, and fine-tune the base model with binary cross-entropy against these soft labels using AdamW with learning rate `3 × 10^{-5}`, batch size 128, and dropout 0.05. This recipe is human-label-free, reproducible, and shown to produce a PRM that enables the `4×` efficiency gains in Figures 4 and 8. Organizations can adopt it immediately for their own model families without the need for the expensive human annotations used in prior PRM work (Lightman et al., 2023).

---

### When to Prefer This Method

The paper does not explicitly position compute-optimal test-time scaling against a named set of alternative inference strategies with clear tradeoffs. It compares against two baselines—best-of-N weighted sampling and majority voting—and analyzes the pretraining-vs-inference tradeoff in Section 7, but does not articulate a decision rule for when a practitioner should choose compute-optimal scaling over other test-time methods (e.g., standard beam search with a fixed width, rejection sampling with a discriminator, or MCTS with a learned value function). The paper's contribution is an **optimization framework** (how to allocate a given test-time compute budget) rather than a **method positioned against alternatives** (when to use this framework versus another). The experimental results establish that the framework improves over the best-of-N and majority voting baselines, but there is no comparison against other structured test-time strategies beyond those studied—PRM-guided MCTS, self-consistency with chain-of-thought prompting, or learned search policies. Consequently, a conditional "prefer A when X, prefer B when Y" decision matrix would fabricate tradeoffs not analyzed in the paper. The closest the paper comes to an explicit tradeoff is the conclusion that test-time compute with a small model is preferred over pretraining a larger model when inference-to-pretraining ratios are low (`R ll 1`) and problem difficulty is easy-to-medium (Section 7, Figure 9), but this is a finding about the **pretraining vs. inference** tradeoff, not a method selection guide among inference strategies.
