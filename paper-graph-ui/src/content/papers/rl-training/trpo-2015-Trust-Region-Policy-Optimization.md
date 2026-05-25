# Trust Region Policy Optimization

**URL:** [https://proceedings.mlr.press/v37/schulman15.pdf](https://proceedings.mlr.press/v37/schulman15.pdf)

## 🎯 Pitch

This paper introduces **Trust Region Policy Optimization (TRPO)**, a method for optimizing control policies with a guaranteed monotonic improvement property derived from a theoretical bound on policy performance.

---

## 1. Executive Summary

This paper introduces **Trust Region Policy Optimization (TRPO)**, a method for optimizing control policies with a guaranteed monotonic improvement property derived from a theoretical bound on policy performance. The approach constrains the KL divergence between successive policies — using either a single-path sampling scheme that requires no state resets for model-free settings or a vine sampling scheme that branches multiple rollouts from visited states for lower-variance advantage estimates — to enable large, robust updates to nonlinear function approximators like neural networks. Across simulated robotic locomotion tasks (swimmer, hopper, walker) and Atari games from raw pixels, TRPO learns effective policies with minimal hyperparameter tuning, outperforming prior policy gradient and derivative-free methods, and establishes that a fixed KL divergence constraint is more robust than using a fixed penalty coefficient, particularly on high-dimensional control problems where natural gradient methods fail to make forward progress.

## 2. Context and Motivation

### The Core Problem: Policy Optimization Is Brittle in Practice

The fundamental challenge this paper addresses is deceptively simple: **how do you take a reliable, non-trivial update step when optimizing a control policy parameterized by a large nonlinear function approximator like a neural network?** In supervised learning, gradient-based optimization works reliably — you compute the gradient of a fixed loss function on a stationary data distribution, take a step, and (with appropriate learning rates and momentum) you make steady progress. In reinforcement learning, the situation is fundamentally different: the data distribution depends on the policy itself, so updating the policy changes the data it sees. This creates a feedback loop where small changes to the policy parameters can cause large, unpredictable changes in performance — and taking too large a step in the gradient direction can catastrophically destroy a previously good policy, a phenomenon the paper calls "destructive large policy updates."

This matters for a simple but crucial reason: if policy updates cannot be trusted to improve performance reliably, then reinforcement learning cannot scale to the kinds of complex, high-dimensional problems where it should excel — continuous control of physical robots, learning from high-dimensional sensory input like images, and tasks requiring deep neural network policies with tens of thousands of parameters. The paper explicitly frames this tension in Section 1: derivative-free methods (CEM, CMA) work reliably on many problems but scale poorly with parameter count, while gradient-based methods enjoy far better sample complexity guarantees in theory (Nemirovski, 2005) but are "unsatisfying" in practice because they often fail to outperform simple random search on challenging tasks. The gap the paper seeks to close is precisely this: **how do we get the sample efficiency of gradient-based optimization with the reliability and robustness of derivative-free methods?**

### Why This Problem Is Important

The paper motivates the importance of reliable policy optimization along several dimensions that were particularly salient in 2015 but remain relevant today:

**Scalability to high-dimensional policies.** As the authors note in Section 1, "continuous gradient-based optimization has been very successful at learning function approximators for supervised learning tasks with huge numbers of parameters, and extending their success to reinforcement learning would allow for efficient training of complex and powerful policies." At the time, deep neural networks were transforming supervised learning, but their application to reinforcement learning remained limited largely because existing policy optimization methods could not reliably train them. The authors cite Deisenroth et al. (2013), who surveyed policy search for robotics and concluded that model-free policy search with large numbers of parameters remained a "major challenge." TRPO directly targets this bottleneck.

**Robotic control from minimal prior knowledge.** The paper repeatedly emphasizes that prior approaches to learning locomotion controllers relied on "hand-engineered policy classes with low-dimensional parameterizations" (Section 1) or "hand-architected policy classes that explicitly encode notions of balance and stepping" (Section 8.1, citing Tedrake et al., 2004; Geng et al., 2006; Wampler & Popović, 2009). These approaches required significant domain expertise to design policy representations that would work. A method that could learn effective locomotion controllers using "general-purpose policies and simple cost functions, using minimal prior knowledge" would democratize robot learning and enable rapid prototyping across diverse morphologies. The paper's experimental section explicitly aims to demonstrate this capability.

**Learning from raw sensory input.** The Atari experiments address a regime where the policy must process high-dimensional observations (raw pixels) through a convolutional neural network with 33,500 parameters — a setting where derivative-free methods are hopelessly sample-inefficient and where prior gradient-based methods had shown inconsistent results. Success here would demonstrate that the same policy optimization algorithm can span the range from low-dimensional proprioceptive state to high-dimensional visual input, a unification that the paper explicitly calls out in Section 9 as enabling "robotic controllers that perform both perception and control."

**Theoretical unification of policy optimization methods.** Beyond practical impact, the paper identifies a conceptual gap in the literature: policy iteration, policy gradient, and derivative-free methods were typically studied and deployed as separate families with little theoretical connection. The authors frame their contribution as providing "a perspective that unifies policy gradient and policy iteration methods, and shows them to be special limiting cases of an algorithm that optimizes a certain objective subject to a trust region constraint" (Section 9). This unification has theoretical significance because it clarifies the relationship between apparently disparate methods and suggests a spectrum of algorithms trading off between the extremes.

### Where Existing Approaches Fall Short

The paper identifies specific limitations across the three major families of policy optimization methods, building a case that each is inadequate for the high-dimensional, sample-efficient regime TRPO targets.

**Policy gradient methods: step size fragility.** Standard policy gradient algorithms update the policy parameters in the direction of the estimated gradient of the expected cost. The fundamental problem, articulated most clearly in the theoretical development of Sections 2–3, is that the policy gradient is a *first-order* approximation — it is valid only locally. Taking too large a step moves the policy into regions where the gradient is no longer informative, and the resulting policy can be arbitrarily bad. The paper notes that while Equation (4) guarantees that "a sufficiently small step... that improves $L_{\pi_{\theta_{\text{old}}}}$ will also improve $\eta$," it "does not give us any guidance on how big of a step to take." In practice, practitioners must tune a learning rate, and the "correct" learning rate varies across problems and even across stages of training on the same problem. The `natural gradient` baseline in the experiments (Section 8.1) makes this concrete: it "performed well on the two easier problems, but was unable to generate hopping and walking gaits that made forward progress," precisely because its fixed penalty coefficient (Lagrange multiplier) could not adaptively determine safe step sizes.

The natural policy gradient (Kakade, 2002) attempts to address this by using the Fisher information matrix to rescale the gradient in a parameterization-invariant way, resulting in the update $\theta_{\text{new}} = \theta_{\text{old}} - \lambda A(\theta_{\text{old}})^{-1} \nabla_\theta L(\theta)|_{\theta=\theta_{\text{old}}}$ where $A$ is the Fisher matrix. But as the paper points out in Section 7, the Lagrange multiplier $\lambda$ "is typically treated as an algorithm parameter" — meaning the step size problem is not solved, merely reparameterized. The paper's experiments directly compare natural gradient (with an optimized fixed penalty) to TRPO's constrained formulation and show the constraint is essential for robust performance on harder problems.

**Derivative-free methods: poor sample complexity in high dimensions.** CEM and CMA-ES treat the expected cost as a black-box function of the policy parameters and optimize via stochastic search — typically by maintaining a Gaussian distribution over parameters, sampling candidates, evaluating them, and refitting the distribution to the top performers. These methods are "simple to understand and implement" (Section 1) and "difficult to beat" on some benchmark tasks like Tetris (Gabillon et al., 2013). However, their sample complexity scales unfavorably with the number of parameters — the covariance matrix in CMA-ES is quadratic in the parameter count, and the number of samples needed to reliably estimate which perturbations are beneficial grows with dimensionality. The paper's locomotion experiments (Figure 4) demonstrate this empirically: CEM and CMA "performed poorly on the larger problems" because the neural network policies had dozens to hundreds of parameters. For the 33,500-parameter convolutional networks in the Atari experiments, derivative-free methods are simply infeasible.

**Policy iteration: requires exact evaluation and struggles with approximation.** Exact policy iteration — alternating between evaluating $Q^\pi$ exactly and setting $\pi_{\text{new}}(s) = \arg\min_a A^\pi(s, a)$ — is guaranteed to monotonically improve the policy and converge to optimality. But this requires (a) exact computation of the advantage function at all states and (b) the ability to exactly solve the $\arg\min$ at every state. In continuous or large state spaces, both requirements break down: function approximation introduces error in the advantage estimates, and the $\arg\min$ becomes intractable. When approximations enter the picture, the monotonic improvement guarantee is lost, and approximate policy iteration can oscillate or diverge. The paper's theoretical development directly addresses this gap: by deriving a bound that accounts for the discrepancy between the approximate surrogate loss $L_\pi(\tilde{\pi})$ and the true objective $\eta(\tilde{\pi})$, the authors provide a principled way to ensure improvement even under approximation.

**Conservative policy iteration (Kakade & Langford, 2002): theoretically sound but practically unusable.** This prior work is the direct theoretical predecessor of TRPO and deserves special attention because the paper builds on it heavily. Conservative policy iteration (CPI) starts from the same identity (Equation 1) and the same surrogate loss $L_\pi$ (Equation 3). Kakade and Langford proved that if you take the policy $\pi' = \arg\min_{\pi'} L_{\pi_{\text{old}}}(\pi')$ and mix it with the old policy as $\pi_{\text{new}} = (1 - \alpha)\pi_{\text{old}} + \alpha \pi'$, then Equation (6) guarantees improvement for an appropriate $\alpha$. The bound explicitly quantifies the worst-case performance degradation as a function of the mixture weight $\alpha$ and the maximum advantage magnitude $\epsilon$.

The problem, as the paper states bluntly in Section 2, is that "this policy class is unwieldy and restrictive in practice, and it is desirable for a practical policy update scheme to be applicable to all general stochastic policy classes." Mixture policies are essentially an ensemble that must maintain and evaluate both the old and new policies at every state. For neural network policies, this is impractical — you cannot easily "mix" two neural networks in a way that respects the structure of the policy class while maintaining differentiability. The key theoretical move in TRPO is to replace the mixture weight $\alpha$ with a distance measure (total variation divergence) between the old and new policies, extending the guarantee to arbitrary policy classes without requiring mixtures. The paper states this explicitly: "Our principal theoretical result is that the policy improvement bound in Equation (6) can be extended to general stochastic policies, rather than just mixture policies... this result is crucial for extending the improvement guarantee to practical problems."

**Relative entropy policy search (REPS) and related methods: expensive inner-loop optimization.** The paper acknowledges in Section 7 that REPS (Peters et al., 2010) uses a KL divergence constraint on the state-action marginals $p(s,a)$, while TRPO constrains the conditionals $p(a|s)$. This is a subtle but important distinction: REPS constrains the joint distribution over states and actions, which requires solving a costly nonlinear optimization in the inner loop to match the feature expectations of the old and new policies. TRPO's constraint on the conditional distribution $p(a|s)$ is simpler to enforce and leads to an algorithm whose per-iteration cost is "altogether only slightly more expensive than computing the gradient itself" (Section 6).

### How This Paper Positions Itself

The paper positions TRPO as a **practical approximation to a theoretically justified algorithm**, with the gap between theory and practice explicitly acknowledged and systematically addressed. This framing is important because it makes the paper's contributions clear while being honest about the approximations involved.

The theoretical starting point is Algorithm 1 — an approximate policy iteration scheme that exactly minimizes $M_i(\pi) = L_{\pi_i}(\pi) + C D_{\text{KL}}^{\max}(\pi_i, \pi)$ at each iteration, where $C = 2\epsilon\gamma/(1-\gamma)^2$ is derived from the bound in Equation (10). This algorithm is provably monotonic: $\eta(\pi_0) \geq \eta(\pi_1) \geq \eta(\pi_2) \geq \dots$ as shown in Equation (11). However, it is impractical for three reasons:

1. **The penalty coefficient $C$ is too large.** The theory-derived value of $C = 2\epsilon\gamma/(1-\gamma)^2$ produces "prohibitively small steps" because it corresponds to a worst-case bound that is loose in practice. The paper explicitly notes that "if we used the penalty coefficient $C$ recommended by the theory above, the step sizes would be very small."

2. **The max-KL constraint is intractable.** $D_{\text{KL}}^{\max}(\theta_{\text{old}}, \theta) = \max_s D_{\text{KL}}(\pi_{\theta_{\text{old}}}(\cdot|s) \| \pi_\theta(\cdot|s))$ imposes a separate constraint at every state in the state space, which is "impractical to solve due to the large number of constraints."

3. **Exact advantage evaluation is assumed.** Algorithm 1 assumes we can "compute all advantage values $A^{\pi_i}(s,a)$" exactly, which is impossible with finite samples and function approximation.

TRPO makes three corresponding approximations, each of which the paper justifies:

- **Replace the penalty with a hard constraint:** Instead of $C D_{\text{KL}}^{\max}$ as a penalty term in the objective, TRPO constrains the KL divergence to be at most $\delta$, where $\delta$ is a hyperparameter. The paper argues that "empirically, it is hard to robustly choose the penalty coefficient, so we use a hard constraint instead of a penalty, with parameter $\delta$ (the bound on KL divergence)." This transforms the problem from unconstrained optimization with a tricky coefficient to constrained optimization with a parameter that has a more intuitive interpretation (the maximum allowable change in the policy per iteration).

- **Replace $D_{\text{KL}}^{\max}$ with average KL:** Instead of constraining the KL divergence at every state, TRPO constrains the *expected* KL divergence under the state visitation distribution of the old policy: $\bar{D}_{\text{KL}}^{\rho_{\theta_{\text{old}}}}(\theta_{\text{old}}, \theta) = \mathbb{E}_{s \sim \rho_{\theta_{\text{old}}}}[D_{\text{KL}}(\pi_{\theta_{\text{old}}}(\cdot|s) \| \pi_\theta(\cdot|s))]$. The paper validates this approximation experimentally: the `max KL` variant, which was "only tractable on the cart-pole problem," "learned somewhat slower than our final method, due to the more restrictive form of the constraint, but overall the result suggests that the average KL divergence constraint has a similar effect as the theoretically justified maximum KL divergence."

- **Use sample-based estimation of advantages:** Instead of exact advantage values, TRPO uses Monte Carlo estimates from sampled trajectories (single path) or branched rollouts (vine). The paper acknowledges that "our theory ignores estimation error for the advantage function" and notes that Kakade & Langford (2002) considered this error and "the same arguments would hold in the setting of this paper, but we omit them for simplicity."

The paper thus positions itself in the lineage of **majorization-minimization (MM) algorithms**, where $M_i$ is a surrogate function that majorizes $\eta$ with equality at $\pi_i$, and minimizing $M_i$ guarantees improvement in the true objective. TRPO is a sample-based, constrained approximation to this ideal MM algorithm. The connection to trust region methods from nonlinear optimization is explicit in the name: just as trust region methods in numerical optimization constrain each step to lie within a region where a local quadratic model is trusted to be accurate, TRPO constrains each policy update to lie within a region (defined by KL divergence) where the local approximation $L_\pi$ is trusted to predict changes in $\eta$ accurately.

A key aspect of this positioning is that the paper does not claim TRPO is the *only* way to implement the theoretical insights, nor that it perfectly inherits the monotonic improvement guarantee. Instead, it claims that TRPO "tends to give monotonic improvement, with little tuning of hyperparameters" (abstract) and that the constrained formulation is "more robust" than using a fixed penalty (Section 7), which is an empirical claim supported by the experiments. This honest treatment of the theory-practice gap — deriving a theoretically justified algorithm, then systematically approximating it for practicality while empirically validating the approximations — is a central feature of the paper's contribution.

## 3. Technical Approach

### 3.1 Reader Orientation

This paper develops a **policy optimization algorithm** — a procedure that iteratively updates the parameters of a controller (represented by a neural network) so that it performs better at a given task by interacting with its environment. The core problem it solves is **how to take reliably large, non-destructive update steps when the policy is a high-dimensional nonlinear function approximator**: the solution is to constrain each update to lie within a "trust region" defined by KL divergence, where a local surrogate model of performance is guaranteed (up to approximation error) to predict actual performance accurately, enabling monotonic improvement without catastrophic step-size tuning.

### 3.2 Big-Picture Architecture (Diagram in Words)

The TRPO system consists of five major components that interact in a fixed cycle:

1. **Policy `$\pi_\theta$` (a neural network):** maps states to action distributions. This is what we optimize. It receives states from the environment and outputs a probability distribution over actions from which we sample.
2. **Simulator/Environment:** the MDP — physics simulator (MuJoCo) for locomotion, or Atari emulator for games. It receives actions and produces next states and costs.
3. **Sampling procedure (single path or vine):** collects trajectories of state-action-reward tuples by executing the current policy `$\pi_{\theta_{\text{old}}}$` in the environment. These trajectories serve as the dataset for the current iteration.
4. **Advantage estimator:** computes empirical estimates of `$Q^{\pi_{\theta_{\text{old}}}}(s, a)$` and `$A^{\pi_{\theta_{\text{old}}}}(s, a)$` from the sampled trajectories. These advantage estimates tell us which actions are better or worse than average at each visited state.
5. **Constrained optimizer:** solves the TRPO optimization problem — minimizes the surrogate loss `$L_{\theta_{\text{old}}}(\theta)$` subject to `$\bar{D}_{\text{KL}}^{\rho_{\theta_{\text{old}}}}(\theta_{\text{old}}, \theta) \leq \delta$` — using conjugate gradient followed by a line search. This produces the new policy parameters `$\theta_{\text{new}}$`.

**Information flow per iteration:** execute `$\pi_{\theta_{\text{old}}}$` in the environment to collect trajectories → compute advantage estimates from these trajectories → construct the surrogate loss `$L_{\theta_{\text{old}}}(\theta)$` and the KL divergence constraint using these samples → solve the constrained optimization to obtain `$\theta_{\text{new}}$` → set `$\theta_{\text{old}} \leftarrow \theta_{\text{new}}$` and repeat.

### 3.3 Roadmap for the Deep Dive

I will explain TRPO in four layers, building from theoretical foundation to practical implementation:

- **First, the theoretical bound that justifies trust regions (Theorem 1):** this is the mathematical guarantee that gives TRPO its name — it establishes that if we constrain the total variation divergence between successive policies, the surrogate loss `$L_\pi$` upper-bounds the true objective `$\eta$`, so minimizing `$L_\pi$` within a KL ball guarantees improvement. This explains *why* KL constraints work in principle.
- **Second, the constrained optimization formulation:** how we transform the theoretical penalty-based bound into a practical constrained problem (Equation 13), including the crucial substitution of `$D_{\text{KL}}^{\max}$` with average KL divergence and the justification for using a hard constraint rather than a penalty coefficient.
- **Third, sample-based estimation of objective and constraint:** how the surrogate loss and KL divergence are estimated from finite samples using single-path and vine sampling schemes, including the importance sampling transformation that makes the objective compatible with off-policy data (Equation 15) and the self-normalized estimator for the vine method (Equation 17).
- **Fourth, the practical numerical algorithm:** the conjugate gradient procedure for approximately solving the constrained optimization, the Fisher information matrix estimation (analytical vs. empirical), and the line search that enforces the KL constraint exactly — this is the computational engine that makes TRPO feasible on neural network policies with tens of thousands of parameters.

This order is chosen because each layer depends on the previous one: the bound motivates the constrained problem, the constrained problem requires sample-based estimates, and the estimates must be optimized efficiently via conjugate gradient. Understanding the chain of approximations — from Theorem 1 to the final algorithm — is essential to understanding both TRPO's empirical robustness and its theoretical foundations.

### 3.4 Detailed, Sentence-Based Technical Breakdown

This is primarily a **theoretically-motivated algorithm design paper** whose core idea is that policy optimization can be made reliable and scalable by replacing brittle step-size tuning with a principled trust region constraint derived from a performance improvement bound.

---

#### Theoretical Foundation: The Monotonic Improvement Guarantee (Theorem 1)

TRPO's theoretical foundation rests on a single key result: **a bound that relates the true expected cost of a new policy `$\tilde{\pi}$` to a local approximation `$L_\pi(\tilde{\pi})$` plus a penalty term proportional to a divergence measure between `$\pi$` and `$\tilde{\pi}$`.** This bound is what enables guaranteed monotonic improvement — if we minimize this upper bound at each iteration, the true cost cannot increase.

The derivation begins from an exact identity relating the cost of two policies. For any two stochastic policies `$\pi$` and `$\tilde{\pi}$`, the expected discounted cost `$\eta$` satisfies:

$$\eta(\tilde{\pi}) = \eta(\pi) + \mathbb{E}_{s_0, a_0, s_1, a_1, \dots}\left[\sum_{t=0}^{\infty} \gamma^t A^\pi(s_t, a_t)\right]$$

where `$s_0 \sim \rho_0(s_0)$`, `$a_t \sim \tilde{\pi}(a_t|s_t)$`, and `$s_{t+1} \sim P(s_{t+1}|s_t, a_t)$`. The expectation is taken over trajectories generated by running the *new* policy `$\tilde{\pi}$` in the environment, but the advantage values `$A^\pi(s_t, a_t)$` are computed with respect to the *old* policy `$\pi$`.

**What this identity computes:** it expresses the cost of the new policy `$\eta(\tilde{\pi})$` as the cost of the old policy `$\eta(\pi)$` plus a discounted sum of advantages of `$\tilde{\pi}$` relative to `$\pi$`. Each term `$A^\pi(s_t, a_t)$` tells us whether taking action `$a_t$` (sampled from `$\tilde{\pi}$`) in state `$s_t$` is better (negative) or worse (positive) than what `$\pi$` would typically do. Summing these advantages over time, discounted by `$\gamma^t$`, gives the total performance change.

**Why this form matters:** the identity shows that if we know the advantage function `$A^\pi$` exactly and can evaluate it on trajectories from `$\tilde{\pi}$`, we can exactly predict the performance change. The central difficulty — and the reason this identity doesn't directly yield an algorithm — is that we cannot sum over trajectories from `$\tilde{\pi}$` when deciding what `$\tilde{\pi}$` should be, because we haven't yet committed to `$\tilde{\pi}$`. This circular dependency is what motivates the surrogate loss.

Rearranging Equation (1) to sum over states rather than timesteps yields:

$$\eta(\tilde{\pi}) = \eta(\pi) + \sum_s \rho_{\tilde{\pi}}(s) \sum_a \tilde{\pi}(a|s) A^\pi(s, a)$$

where `$\rho_{\tilde{\pi}}(s) = P(s_0 = s) + \gamma P(s_1 = s) + \gamma^2 P(s_2 = s) + \dots$` is the (unnormalized) discounted visitation frequency under `$\tilde{\pi}$` — how often each state is visited, weighted by discount factor to give more weight to earlier states. The double sum `$\sum_a \tilde{\pi}(a|s) A^\pi(s, a)$` is the expected advantage of `$\tilde{\pi}$` at state `$s$`: if this quantity is non-positive at every state, then `$\tilde{\pi}$` is guaranteed to be at least as good as `$\pi$`.

**The surrogate loss `$L_\pi(\tilde{\pi})$`** is obtained by replacing the visitation frequency `$\rho_{\tilde{\pi}}$` (which depends on the unknown new policy) with `$\rho_\pi$` (which depends only on the known old policy):

$$L_\pi(\tilde{\pi}) = \eta(\pi) + \sum_s \rho_\pi(s) \sum_a \tilde{\pi}(a|s) A^\pi(s, a)$$

where `$\rho_\pi(s)$` is the discounted visitation frequency under the *old* policy `$\pi$`, and all other terms are as defined above.

**What `$L_\pi(\tilde{\pi})$` computes:** it is a first-order approximation to `$\eta(\tilde{\pi})$`. It asks: "if the state distribution didn't change when we switch from `$\pi$` to `$\tilde{\pi}$`, what would the new cost be?" This is computationally tractable because `$\rho_\pi$` can be estimated from trajectories of `$\pi$`, which we already have.

**Why this specific approximation:** the paper cites two key properties (Equation 4). First, `$L_\pi$` matches `$\eta$` exactly at `$\pi$` itself: `$L_{\pi_{\theta_0}}(\pi_{\theta_0}) = \eta(\pi_{\theta_0})$`. Second, for a parameterized policy `$\pi_\theta$`, the gradient of `$L$` with respect to `$\theta$` matches the gradient of `$\eta$` at `$\theta_0$`: `$\nabla_\theta L_{\pi_{\theta_0}}(\pi_\theta)|_{\theta=\theta_0} = \nabla_\theta \eta(\pi_\theta)|_{\theta=\theta_0}$`. This means that `$L$` is correct to first order — a sufficiently small step that improves `$L$` will also improve `$\eta$`. The problem, as the paper states, is that this "does not give us any guidance on how big of a step to take." The first-order agreement says nothing about what happens when we take a non-infinitesimal step.

**The key theoretical move — bounding the approximation error:** Kakade and Langford (2002) showed that for the specific case of *mixture policies* of the form `$\pi_{\text{new}}(a|s) = (1 - \alpha)\pi_{\text{old}}(a|s) + \alpha \pi'(a|s)$`, the following bound holds:

$$\eta(\pi_{\text{new}}) \leq L_{\pi_{\text{old}}}(\pi_{\text{new}}) + \frac{2\epsilon\gamma}{(1-\gamma)^2}\alpha^2$$

where `$\epsilon = \max_s |\mathbb{E}_{a \sim \pi'(a|s)}[A^\pi(s, a)]|$` is the maximum absolute expected advantage of `$\pi'$` over any state, `$\alpha \in [0, 1]$` is the mixture weight, and `$\gamma$` is the discount factor.

**What this bound says in operational terms:** the true cost of the new mixture policy is no worse than the surrogate loss `$L_{\pi_{\text{old}}}$` evaluated on the new policy, plus a penalty term that grows quadratically with `$\alpha$`. The coefficient `$2\epsilon\gamma/(1-\gamma)^2$` depends on the maximum magnitude of the advantage function (how much any action can differ from the average at any state) and the discount factor (larger `$\gamma$` means longer horizons and larger potential compounding of errors). This is a worst-case bound: the penalty term ensures that even in the most adversarial scenario, the true performance won't degrade beyond this amount.

**Why the mixture policy form matters:** the mixture policy `$\pi_{\text{new}} = (1 - \alpha)\pi_{\text{old}} + \alpha \pi'$` has the property that its total variation distance from `$\pi_{\text{old}}$` is bounded by `$\alpha$` at every state: `$D_{\text{TV}}(\pi_{\text{new}}(\cdot|s) \| \pi_{\text{old}}(\cdot|s)) \leq \alpha$` for all `$s$`. Kakade and Langford's proof leveraged this boundedness to control the error from replacing `$\rho_{\tilde{\pi}}$` with `$\rho_{\pi}$`.

**Theorem 1 — extending the bound to general policies:** the paper's central theoretical contribution is to show that the same bound applies to *any* pair of policies `$\pi_{\text{old}}$` and `$\pi_{\text{new}}$`, with `$\alpha$` replaced by the maximum total variation divergence `$D_{\text{TV}}^{\max}(\pi_{\text{old}}, \pi_{\text{new}}) = \max_s D_{\text{TV}}(\pi_{\text{old}}(\cdot|s) \| \pi_{\text{new}}(\cdot|s))$`:

$$\eta(\pi_{\text{new}}) \leq L_{\pi_{\text{old}}}(\pi_{\text{new}}) + \frac{2\epsilon\gamma}{(1-\gamma)^2} \alpha^2$$

where now `$\alpha = D_{\text{TV}}^{\max}(\pi_{\text{old}}, \pi_{\text{new}})$` and `$\epsilon = \max_s |\mathbb{E}_{a \sim \pi'(a|s)}[A^\pi(s, a)]|$` is defined with respect to some policy `$\pi'$`; the paper provides two proofs, with the second proof yielding a slightly tighter bound where `$\epsilon$` depends on `$\pi_{\text{new}}$` rather than `$\pi'$`.

**The proof strategy (high-level):** the first proof uses a coupling argument: two probability distributions with total variation distance `$\alpha$` can be coupled so that samples from them are equal with probability `$1 - \alpha$` and differ with probability `$\alpha$`. The second proof uses perturbation theory. In both cases, the key insight is that if the total variation distance is bounded, the difference between `$\rho_{\pi_{\text{old}}}$` and `$\rho_{\pi_{\text{new}}}$` can also be bounded, allowing us to translate Kakade and Langford's result from the mixture policy setting to the general setting.

**Why this extension is crucial:** mixture policies are "unwieldy and restrictive in practice" — they require maintaining both the old and new policies and combining them via weighted sampling at every state. Neural network policies cannot easily represent such mixtures. Theorem 1 says that we don't need mixtures; we can directly update any parameterized policy, as long as we constrain how far it moves (in total variation) from the previous policy. This is what makes the theory applicable to the deep neural network policies used in the experiments.

**From total variation to KL divergence:** the paper then uses Pinsker's inequality — `$D_{\text{TV}}(p \| q)^2 \leq D_{\text{KL}}(p \| q)$` — to convert the total variation bound into a KL divergence bound. Define `$D_{\text{KL}}^{\max}(\pi, \tilde{\pi}) = \max_s D_{\text{KL}}(\pi(\cdot|s) \| \tilde{\pi}(\cdot|s))$`. Then:

$$\eta(\tilde{\pi}) \leq L_\pi(\tilde{\pi}) + C D_{\text{KL}}^{\max}(\pi, \tilde{\pi})$$

where `$C = \frac{2\epsilon\gamma}{(1-\gamma)^2}$` is the penalty coefficient.

**What this gives us:** a guarantee that if we minimize `$L_{\pi_{\text{old}}}(\pi) + C D_{\text{KL}}^{\max}(\pi_{\text{old}}, \pi)$` at each iteration, the true cost `$\eta$` is non-increasing. Specifically, define `$M_i(\pi) = L_{\pi_i}(\pi) + C D_{\text{KL}}^{\max}(\pi_i, \pi)$`. Then for any `$\pi_{i+1}$` that minimizes `$M_i$`:

$$\eta(\pi_{i+1}) \leq M_i(\pi_{i+1}) \leq M_i(\pi_i) = \eta(\pi_i)$$

The first inequality is the theorem; the second is because `$\pi_{i+1}$` minimizes `$M_i$`; the equality is because `$L_{\pi_i}(\pi_i) = \eta(\pi_i)$` and `$D_{\text{KL}}^{\max}(\pi_i, \pi_i) = 0$`. This is the monotonic improvement guarantee — the sequence `$\eta(\pi_0), \eta(\pi_1), \eta(\pi_2), \dots$` is non-increasing.

**Why we cannot use this algorithm directly (the penalty form is impractical):** the coefficient `$C = 2\epsilon\gamma/(1-\gamma)^2$` is derived from worst-case analysis and produces "prohibitively small steps" in practice. The max-KL constraint `$D_{\text{KL}}^{\max}$` is a per-state infinity norm that is both hard to estimate from samples and hard to enforce in optimization (it requires satisfying a constraint at every state simultaneously). And the theory assumes exact advantage values `$A^\pi(s, a)$`, which we don't have. These three issues motivate all the practical approximations in the next subsection.

---

#### From Theory to Practice: The Constrained Optimization Formulation

The paper proposes two key departures from the theoretical algorithm that make it practical while preserving its spirit:

**Approximation 1: Replace the penalty with a hard constraint.** Instead of minimizing `$L_{\theta_{\text{old}}}(\theta) + C D_{\text{KL}}^{\max}(\theta_{\text{old}}, \theta)$` with a large, hard-to-tune penalty coefficient `$C$`, TRPO solves:

$$\underset{\theta}{\text{minimize}} \; L_{\theta_{\text{old}}}(\theta)$$
$$\text{subject to} \; D_{\text{KL}}^{\max}(\theta_{\text{old}}, \theta) \leq \delta$$

where `$\delta$` is a hyperparameter controlling the size of the trust region.

**What this formulation means operationally:** instead of trading off improvement in the surrogate loss against a penalty on policy change (which requires finding the right scalar weighting `$C$`), we maximize improvement in the surrogate loss subject to a hard cap on how much the policy can change. The parameter `$\delta$` has a direct, interpretable meaning: "the new policy cannot differ from the old by more than `$\delta$` nats of KL divergence at any state." The paper notes that this "is a more robust way to choose step sizes and make fast, consistent progress, compared to using a fixed penalty."

**Why a constraint rather than a penalty:** the experiments directly compare natural gradient (which uses a fixed penalty coefficient, equivalent to a fixed Lagrange multiplier `$\lambda$`) to TRPO (which uses a constraint). On the harder locomotion tasks (hopper, walker), natural gradient "was unable to generate hopping and walking gaits that made forward progress," while TRPO succeeded. The interpretation: with a penalty, the effective step size varies unpredictably depending on the local geometry of the objective; with a constraint, the step size adapts automatically — the algorithm takes the largest step it can within the trust region, rather than being limited by a globally fixed `$\lambda$`.

**Approximation 2: Replace `$D_{\text{KL}}^{\max}$` with average KL divergence.** The per-state maximum KL divergence `$D_{\text{KL}}^{\max}$` is impractical because it requires estimating the KL at every state and enforcing a separate constraint for each. TRPO uses the expected KL divergence under the state visitation distribution of the old policy:

$$\bar{D}_{\text{KL}}^{\rho_{\theta_{\text{old}}}}(\theta_{\text{old}}, \theta) = \mathbb{E}_{s \sim \rho_{\theta_{\text{old}}}}\left[D_{\text{KL}}(\pi_{\theta_{\text{old}}}(\cdot|s) \| \pi_\theta(\cdot|s))\right]$$

where the expectation `$\mathbb{E}_{s \sim \rho_{\theta_{\text{old}}}}$` is over states visited by the old policy, weighted by discounted visitation frequency.

**What this computes:** it averages the per-state KL divergences, weighting each state by how often the old policy visits it (with early states weighted more heavily due to discounting). States that the old policy rarely visits contribute little to the average, even if the policy changes substantially there.

**Why this is a reasonable approximation:** the paper includes an experiment on the cart-pole problem using the `max KL` variant (the exact `$D_{\text{KL}}^{\max}$` constraint), which was "only tractable on the cart-pole problem" because it requires a finite, small state space. The result: max KL "learned somewhat slower than our final method, due to the more restrictive form of the constraint, but overall the result suggests that the average KL divergence constraint has a similar effect as the theoretically justified maximum KL divergence." The average KL is less conservative — it allows larger changes in rarely-visited states as long as the policy doesn't change much in the states that matter most.

The final constrained optimization problem (Equation 13) is:

$$\underset{\theta}{\text{minimize}} \; L_{\theta_{\text{old}}}(\theta)$$
$$\text{subject to} \; \bar{D}_{\text{KL}}^{\rho_{\theta_{\text{old}}}}(\theta_{\text{old}}, \theta) \leq \delta$$

This is the problem that TRPO attempts to solve at each iteration using sample-based estimates.

---

#### Sample-Based Estimation: Converting Population Quantities to Empirical Estimates

The constrained optimization problem in Equation (13) involves expectations over the state visitation distribution `$\rho_{\theta_{\text{old}}}$` and the action distribution `$\pi_\theta$`, which must be estimated from finite trajectories. This section describes how TRPO converts these population quantities into computable, sample-based forms.

**Step 1: Expand the surrogate loss `$L_{\theta_{\text{old}}}(\theta)$`.** Substituting the definition of `$L$` from Equation (3) into Equation (13) yields:

$$\underset{\theta}{\text{minimize}} \; \sum_s \rho_{\theta_{\text{old}}}(s) \sum_a \pi_\theta(a|s) A^{\theta_{\text{old}}}(s, a)$$
$$\text{subject to} \; \bar{D}_{\text{KL}}^{\rho_{\theta_{\text{old}}}}(\theta_{\text{old}}, \theta) \leq \delta$$

The term `$\eta(\pi_{\theta_{\text{old}}})$` from the definition of `$L$` is dropped because it is constant with respect to `$\theta$` and does not affect the optimization.

**Step 2: Replace the sum over states with an expectation.** The sum `$\sum_s \rho_{\theta_{\text{old}}}(s)[\cdots]$` can be written as an expectation up to a constant factor:

$$\sum_s \rho_{\theta_{\text{old}}}(s)[\cdots] = \frac{1}{1-\gamma} \mathbb{E}_{s \sim \bar{\rho}_{\theta_{\text{old}}}}[\cdots]$$

where `$\bar{\rho}_{\theta_{\text{old}}}$` is the *normalized* visitation distribution (summing to 1), and the factor `$1/(1-\gamma)$` comes from the geometric series `$1 + \gamma + \gamma^2 + \dots = 1/(1-\gamma)$` that normalizes the unnormalized frequencies `$\rho_{\theta_{\text{old}}}$`. Since this factor is constant, it can be dropped from the optimization.

**Step 3: Replace advantage with Q-values.** The advantage function `$A^{\theta_{\text{old}}}(s, a) = Q^{\theta_{\text{old}}}(s, a) - V^{\theta_{\text{old}}}(s)$`. Substituting this into the objective:

$$\sum_a \pi_\theta(a|s) A^{\theta_{\text{old}}}(s, a) = \sum_a \pi_\theta(a|s) Q^{\theta_{\text{old}}}(s, a) - V^{\theta_{\text{old}}}(s) \sum_a \pi_\theta(a|s)$$

Since `$\sum_a \pi_\theta(a|s) = 1$` for any policy, the `$V^{\theta_{\text{old}}}(s)$` term is constant with respect to `$\theta$` and can be dropped. The objective reduces to `$\sum_a \pi_\theta(a|s) Q^{\theta_{\text{old}}}(s, a)$`.

**What this substitution accomplishes:** it simplifies estimation because we only need to estimate Q-values, not advantages. The state-value baseline `$V^{\theta_{\text{old}}}(s)$` — which would need to be estimated separately or subtracted — is eliminated. This is mathematically identical to using advantages, but computationally simpler.

**Step 4: Importance sampling for the action sum.** The sum over actions `$\sum_a \pi_\theta(a|s) Q^{\theta_{\text{old}}}(s, a)$` cannot be evaluated exactly because (a) in continuous action spaces, it is an integral, and (b) we don't have Q-values for all actions — we only have Q-estimates for actions that were actually sampled. The solution is importance sampling: introduce a sampling distribution `$q(a|s)$` from which we actually draw actions, and reweight:

$$\sum_a \pi_\theta(a|s) Q^{\theta_{\text{old}}}(s, a) = \mathbb{E}_{a \sim q(\cdot|s)}\left[\frac{\pi_\theta(a|s)}{q(a|s)} Q^{\theta_{\text{old}}}(s, a)\right]$$

where `$q(\cdot|s)$` is any distribution whose support includes the support of `$\pi_\theta(\cdot|s)$`. The importance weight `$\pi_\theta(a|s)/q(a|s)$` corrects for the discrepancy between the distribution we sample from (`$q$`) and the distribution whose expectation we want (`$\pi_\theta$`).

**What this enables:** we only need to sample actions from `$q$`, evaluate their Q-values, and compute weighted averages — we never need to enumerate or integrate over the full action space.

Putting all steps together, the population optimization problem (Equation 14) is equivalent to the following sample-based formulation (Equation 15):

$$\underset{\theta}{\text{minimize}} \; \mathbb{E}_{s \sim \rho_{\theta_{\text{old}}}, a \sim q}\left[\frac{\pi_\theta(a|s)}{q(a|s)} Q^{\theta_{\text{old}}}(s, a)\right]$$
$$\text{subject to} \; \mathbb{E}_{s \sim \rho_{\theta_{\text{old}}}}\left[D_{\text{KL}}(\pi_{\theta_{\text{old}}}(\cdot|s) \| \pi_\theta(\cdot|s))\right] \leq \delta$$

where all that remains is to replace the expectations with sample averages and `$Q^{\theta_{\text{old}}}$` with empirical estimates. The two sampling schemes — single path and vine — differ in how they approximate these expectations.

---

#### Single Path Sampling

The single path method is the simpler of the two and is the one that can be applied in model-free settings without state resets.

**Procedure:**
1. Sample an initial state `$s_0 \sim \rho_0$`.
2. Run the current policy `$\pi_{\theta_{\text{old}}}$` for `$T$` timesteps, generating a trajectory `$s_0, a_0, s_1, a_1, \dots, s_{T-1}, a_{T-1}, s_T$` where each action `$a_t$` is sampled from `$\pi_{\theta_{\text{old}}}(\cdot|s_t)$`.
3. For each state-action pair `$(s_t, a_t)$` in the trajectory, estimate `$Q^{\theta_{\text{old}}}(s_t, a_t)$` as the discounted sum of future costs from that point:

$$\hat{Q}^{\theta_{\text{old}}}(s_t, a_t) = \sum_{l=0}^{T-t-1} \gamma^l c(s_{t+l})$$

This is the Monte Carlo return: the sum of costs actually observed from timestep `$t$` onward in this specific trajectory, discounted by `$\gamma^l$`.

**How the objective is constructed:** with single path, the sampling distribution `$q(a|s)$` is simply `$\pi_{\theta_{\text{old}}}(a|s)$` itself — actions are sampled on-policy from the current policy. Therefore, the importance weight `$\pi_\theta(a|s)/\pi_{\theta_{\text{old}}}(a|s)$` is the likelihood ratio between the new and old policies. The empirical objective becomes:

$$\frac{1}{T} \sum_{t=0}^{T-1} \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)} \hat{Q}^{\theta_{\text{old}}}(s_t, a_t)$$

All state-action pairs along the trajectory are used. The constraint is estimated as the average KL divergence over the visited states:

$$\frac{1}{T} \sum_{t=0}^{T-1} D_{\text{KL}}(\pi_{\theta_{\text{old}}}(\cdot|s_t) \| \pi_\theta(\cdot|s_t))$$

**Why this works:** single path is essentially the standard policy gradient estimation procedure (Bartlett & Baxter, 2011), with the addition of the KL constraint. Each trajectory provides correlated samples of states and actions; the Monte Carlo returns `$\hat{Q}$` are unbiased but high-variance estimates of the true Q-values.

**Tradeoff:** the advantage of single path is that it requires no state resets and can be implemented on a physical system (Peters & Schaal, 2008b). The disadvantage is that the Q-value estimates have high variance because each state-action pair gets only one rollout (the rest of the observed trajectory).

---

#### Vine Sampling

The vine method reduces the variance of the advantage estimates by generating multiple independent rollouts from selected states, at the cost of requiring the ability to reset the system to arbitrary states (typically only possible in simulation).

**Procedure:**
1. Generate a set of "trunk" trajectories by running `$\pi_{\theta_{\text{old}}}$` from `$s_0 \sim \rho_0$`.
2. Select a subset of `$N$` states from these trajectories, denoted `$s_1, s_2, \dots, s_N$`, called the "rollout set."
3. For each state `$s_n$` in the rollout set, sample `$K$` actions from a sampling distribution `$q(\cdot|s_n)$`. The paper notes two choices: `$q(\cdot|s_n) = \pi_{\theta_{\text{old}}}(\cdot|s_n)$` works well on continuous problems, while the uniform distribution works well on discrete tasks like Atari games "where it can sometimes achieve better exploration."
4. For each sampled action `$a_{n,k}$`, perform a rollout (a short trajectory) starting from state `$s_n$` and action `$a_{n,k}$`, and estimate `$\hat{Q}^{\theta_{\text{old}}}(s_n, a_{n,k})$` as the discounted sum of future costs. Critically, the paper uses **common random numbers (CRN)** across the `$K$` rollouts from each state `$s_n$`: the same random number sequence is used for the stochastic transitions in each rollout, so that differences in Q-values between actions are due to the actions themselves rather than random noise.

**How the objective is constructed — full enumeration case:** in small, finite action spaces with action set `$\mathcal{A} = \{a_1, a_2, \dots, a_K\}$`, we can perform a rollout for every possible action from each state `$s_n$`. The contribution to the objective from state `$s_n$` is then:

$$L_n(\theta) = \sum_{k=1}^{K} \pi_\theta(a_k|s_n) \hat{Q}(s_n, a_k)$$

This is a direct computation of the expected Q-value under `$\pi_\theta$`, using the rollout estimates `$\hat{Q}(s_n, a_k)$` for each action. No importance sampling is needed because we have Q-estimates for all actions.

**How the objective is constructed — importance sampling case:** in large or continuous action spaces, we cannot enumerate all actions. Instead, we sample `$K$` actions `$a_{n,1}, a_{n,2}, \dots, a_{n,K}$` from `$q(\cdot|s_n)$` and use a **self-normalized importance sampling estimator**:

$$L_n(\theta) = \frac{\sum_{k=1}^{K} \frac{\pi_\theta(a_{n,k}|s_n)}{\pi_{\theta_{\text{old}}}(a_{n,k}|s_n)} \hat{Q}(s_n, a_{n,k})}{\sum_{k=1}^{K} \frac{\pi_\theta(a_{n,k}|s_n)}{\pi_{\theta_{\text{old}}}(a_{n,k}|s_n)}}$$

where `$\pi_{\theta_{\text{old}}}(a_{n,k}|s_n)$` appears in the denominator of the importance weight because we are using `$q = \pi_{\theta_{\text{old}}}$` as the sampling distribution in this case (the paper states this simplifies to `$\pi_{\theta_{\text{old}}}$` on continuous problems).

**What the self-normalized estimator computes:** it is a weighted average of the Q-values `$\hat{Q}(s_n, a_{n,k})$`, where each Q-value is weighted by how much more likely `$\pi_\theta$` is to take action `$a_{n,k}$` compared to `$\pi_{\theta_{\text{old}}}$`. The denominator normalizes these weights so that they sum to 1, producing a proper weighted average. This is in contrast to the standard (unnormalized) importance sampling estimator `$\frac{1}{K}\sum_{k=1}^{K} \frac{\pi_\theta(a_{n,k}|s_n)}{\pi_{\theta_{\text{old}}}(a_{n,k}|s_n)} \hat{Q}(s_n, a_{n,k})$`, which can have high variance when the importance weights are large.

**Why self-normalization matters:** the self-normalized estimator "removes the need to use a baseline for the Q-values" because adding a constant to all Q-values leaves the estimate unchanged (the constant cancels between numerator and denominator). This means we don't need to estimate and subtract a state-dependent baseline `$V(s)$`, which is a significant practical simplification. The self-normalized estimator is also more stable when importance weights vary in magnitude.

**Averaging across states:** the overall objective is obtained by averaging `$L_n(\theta)$` over the `$N$` states in the rollout set:

$$\hat{L}(\theta) = \frac{1}{N} \sum_{n=1}^{N} L_n(\theta)$$

The KL divergence constraint is estimated similarly by averaging over the visited states in the trunk trajectories.

**Why vine reduces variance:** the key advantage of vine over single path is that for each state `$s_n$` in the rollout set, we obtain `$K$` independent Q-value estimates (one per action). This provides a much lower-variance estimate of the expected advantage `$\sum_a \pi_\theta(a|s_n) A(s_n, a)$` at that state. Since the variance of the objective gradient depends on the variance of these advantage estimates, vine can provide more reliable gradient estimates with the same number of environment interactions. The tradeoff is computational: vine requires `$K$` rollouts per state in the rollout set (rather than just continuing the original trajectory), which means many more simulator calls for the same number of objective evaluations. Furthermore, vine "requires the system to be restored to particular states, which is typically only possible in simulation."

**Why the name "vine":** the paper explains the metaphor: "the trajectories used for sampling can be likened to the stems of vines, which branch at various points (the rollout set) into several short offshoots (the rollout trajectories)."

---

#### Practical Numerical Optimization: Conjugate Gradient and Line Search

Once the sample-based objective `$\hat{L}(\theta)$` and constraint `$\hat{\bar{D}}_{\text{KL}}(\theta_{\text{old}}, \theta)$` are constructed, TRPO must solve the constrained optimization problem:

$$\underset{\theta}{\text{minimize}} \; \hat{L}(\theta)$$
$$\text{subject to} \; \hat{\bar{D}}_{\text{KL}}(\theta_{\text{old}}, \theta) \leq \delta$$

This is a nonlinear constrained optimization over potentially tens of thousands of parameters. TRPO uses an approximate solution method based on conjugate gradient, which the paper describes as "altogether only slightly more expensive than computing the gradient itself."

**Step 1: Linear approximation to the objective.** Near `$\theta_{\text{old}}$`, the objective `$\hat{L}(\theta)$` is approximated by its first-order Taylor expansion:

$$\hat{L}(\theta) \approx \hat{L}(\theta_{\text{old}}) + g^T (\theta - \theta_{\text{old}})$$

where `$g = \nabla_\theta \hat{L}(\theta)|_{\theta=\theta_{\text{old}}}$` is the policy gradient (the gradient of the surrogate loss, not the true expected cost). Since `$\hat{L}(\theta_{\text{old}})$` is constant, minimizing `$\hat{L}(\theta)$` reduces to minimizing `$g^T (\theta - \theta_{\text{old}})$`.

**Step 2: Quadratic approximation to the KL constraint.** The KL divergence `$\hat{\bar{D}}_{\text{KL}}(\theta_{\text{old}}, \theta)$` is zero at `$\theta = \theta_{\text{old}}$` and has zero gradient there (since KL divergence is minimized at zero when the two distributions match). Its second-order Taylor expansion is therefore:

$$\hat{\bar{D}}_{\text{KL}}(\theta_{\text{old}}, \theta) \approx \frac{1}{2} (\theta - \theta_{\text{old}})^T H (\theta - \theta_{\text{old}})$$

where `$H = \nabla_\theta^2 \hat{\bar{D}}_{\text{KL}}(\theta_{\text{old}}, \theta)|_{\theta=\theta_{\text{old}}}$` is the Hessian of the KL divergence with respect to `$\theta$`, evaluated at `$\theta_{\text{old}}$`. `$H$` is also the **Fisher information matrix (FIM)** of the policy `$\pi_\theta$` at `$\theta_{\text{old}}$`, which measures the local curvature of the KL divergence.

**The approximate subproblem:** substituting these approximations, the constrained optimization reduces to:

$$\underset{\theta}{\text{minimize}} \; g^T (\theta - \theta_{\text{old}})$$
$$\text{subject to} \; \frac{1}{2} (\theta - \theta_{\text{old}})^T H (\theta - \theta_{\text{old}}) \leq \delta$$

**Step 3: Solving the subproblem.** This is a quadratically-constrained linear program (QCLP) with a known closed-form solution. The optimal update direction `$\Delta\theta$` is:

$$\Delta\theta = -\sqrt{\frac{2\delta}{g^T H^{-1} g}} H^{-1} g$$

where `$H^{-1} g$` is the natural gradient direction (the gradient scaled by the inverse Fisher matrix), and the scalar prefactor `$\sqrt{2\delta / (g^T H^{-1} g)}$` scales the step to exactly saturate the KL constraint — the resulting step `$s = \Delta\theta$` satisfies `$\frac{1}{2} s^T H s = \delta$`.

**What this computes operationally:** first compute the natural gradient direction `$H^{-1} g$` (the direction of steepest descent in the Riemannian metric induced by the Fisher information). Then scale this direction so that the quadratic approximation to the KL divergence equals `$\delta$` — meaning the step is as large as possible while (approximately) remaining within the trust region.

**Why natural gradient appears here:** natural policy gradient (Kakade, 2002) follows exactly this direction `$H^{-1} g$` but with a fixed step size determined by a Lagrange multiplier `$\lambda$` (which is typically treated as a hyperparameter). TRPO differs in that the step size is determined adaptively by the KL constraint: when the natural gradient is "steep" (large `$g^T H^{-1} g$`), the step is smaller (to stay within the KL bound); when it is "flat," the step is larger. This adaptive scaling is what the paper claims makes TRPO more robust than natural gradient.

**Step 4: Computing `$H^{-1} g$` via conjugate gradient.** Directly computing and inverting the Hessian `$H$` is infeasible for neural network policies with tens of thousands of parameters (`$H$` would be a `$d \times d$` matrix where `$d$` is the number of parameters — `$d^2$` entries, far too many to store). TRPO uses the conjugate gradient (CG) algorithm to approximately solve the linear system `$H x = g$` for `$x \approx H^{-1} g`, requiring only the ability to compute matrix-vector products `$H v$` for arbitrary vectors `$v$`.

**Computing `$H v$` without forming `$H$`:** the Hessian-vector product `$H v$` can be computed efficiently using automatic differentiation. Since `$H$` is the Hessian of the KL divergence `$\bar{D}_{\text{KL}}$`, the product `$H v$` equals the gradient of the scalar function `$v^T \nabla_\theta \bar{D}_{\text{KL}}(\theta)$` with respect to `$\theta$`. Equivalently, it can be computed as:

$$H v = \nabla_\theta \left( v^T \nabla_\theta \bar{D}_{\text{KL}}(\theta) \right)$$

Both the inner gradient `$\nabla_\theta \bar{D}_{\text{KL}}$` and the outer gradient of the scalar product are computable via backpropagation. This is the standard "Hessian-vector product via double backprop" technique (Martens & Sutskever, 2012).

**CG iterations:** the conjugate gradient algorithm iteratively refines an estimate of `$H^{-1} g$`, typically converging in 10–20 iterations for the accuracy needed. Each iteration requires one Hessian-vector product `$H v$`, which is comparable in cost to one gradient computation. The paper notes that conjugate gradient is "altogether only slightly more expensive than computing the gradient itself."

**Step 5: The Fisher Information Matrix — analytical vs. empirical estimation.** The paper distinguishes between two ways to estimate the Fisher matrix `$H$`. The **analytical estimator** directly computes the Hessian of the KL divergence:

$$H_{ij} = \mathbb{E}_{s \sim \rho_{\theta_{\text{old}}}}\left[\frac{\partial^2}{\partial\theta_i \partial\theta_j} D_{\text{KL}}(\pi_{\theta_{\text{old}}}(\cdot|s) \| \pi_\theta(\cdot|s))|_{\theta=\theta_{\text{old}}}\right]$$

For a policy `$\pi_\theta(a|s)$` that outputs a distribution (e.g., Gaussian mean and variance for continuous actions, or softmax probabilities for discrete actions), this Hessian can be computed analytically and depends on the policy's output at state `$s$` but **not** on which action was actually sampled. Therefore, the analytical estimator integrates over all actions analytically, using only the state distribution.

The **empirical estimator** (also called the empirical Fisher) approximates `$H$` using the outer product of gradients of the log-policy:

$$H_{ij} \approx \frac{1}{N} \sum_{n=1}^{N} \frac{\partial}{\partial\theta_i} \log \pi_\theta(a_n|s_n) \cdot \frac{\partial}{\partial\theta_j} \log \pi_\theta(a_n|s_n)$$

This depends on the sampled actions `$a_n$`.

**Why the paper prefers the analytical estimator:** the analytical estimator "integrates over the action at each state `$s_n$`, and does not depend on the action `$a_n$` that was sampled." This has two advantages: (1) it produces a lower-variance estimate of `$H$` because it averages over actions analytically rather than relying on a finite sample of actions, and (2) more importantly for large-scale problems, it removes the need to store a dense Hessian or all policy gradients from a batch of trajectories — the analytical estimator can be computed on-the-fly for each state during the conjugate gradient iterations. The paper reports that "the rate of improvement in the policy is similar to the empirical FIM, as shown in the experiments," validating that the analytical estimator does not sacrifice performance.

**Step 6: Line search to enforce the constraint.** The CG solution gives the search *direction* `$s = \Delta\theta$`, but because the quadratic approximation to the KL divergence may be imperfect (especially far from `$\theta_{\text{old}}$`), the actual KL divergence of the step `$\bar{D}_{\text{KL}}(\theta_{\text{old}} + s)$` may exceed `$\delta$`. TRPO performs a backtracking line search: starting from the full step `$s$`, it tries candidate steps `$\theta_{\text{new}} = \theta_{\text{old}} + \beta^j s$` for `$j = 0, 1, 2, \dots$` with `$\beta \in (0, 1)$` (typically around 0.5–0.8), and selects the first `$j$` for which both (a) the actual KL divergence `$\bar{D}_{\text{KL}}(\theta_{\text{old}}, \theta_{\text{new}}) \leq \delta$` and (b) the surrogate loss `$\hat{L}(\theta_{\text{new}})$` is improved (i.e., lower than at `$\theta_{\text{old}}$`). This line search ensures that the KL constraint is actually satisfied and that the update genuinely improves the surrogate objective.

**The full per-iteration algorithm:**
1. Collect a batch of trajectories using `$\pi_{\theta_{\text{old}}}$` (single path or vine).
2. Compute the policy gradient `$g = \nabla_\theta \hat{L}(\theta)|_{\theta=\theta_{\text{old}}}$` using the sample estimates.
3. Use conjugate gradient to approximately solve `$H x = g$` for `$x \approx H^{-1} g$`, computing `$H v$` products via the analytical Fisher estimator.
4. Compute the step direction `$s = -\sqrt{2\delta/(x^T H x)} \, x$` (or equivalently, compute `$\alpha = \sqrt{2\delta/(x^T H x)}$` and set `$s = -\alpha x$`).
5. Perform backtracking line search: for `$j = 0, 1, 2, \dots$`, check `$\theta_{\text{new}} = \theta_{\text{old}} + \beta^j s$`; accept the first candidate that satisfies the KL constraint and improves `$\hat{L}$`.
6. Set `$\theta_{\text{old}} \leftarrow \theta_{\text{new}}$` and repeat.

---

#### Summary of Key Design Choices and Their Justifications

- **KL divergence (not Euclidean distance) for the trust region:** Euclidean distance in parameter space is not meaningful for stochastic policies — a small change in parameters can cause a large change in the action distribution, or vice versa. KL divergence measures the actual difference in the probability distributions over actions, which directly controls the performance bound in Theorem 1. This makes the trust region invariant to parameterization (reparameterizing `$\theta$` does not change the KL divergence between policies).

- **Hard constraint rather than penalty:** the penalty coefficient `$C = 2\epsilon\gamma/(1-\gamma)^2$` from theory is overly conservative. Using a hard constraint with parameter `$\delta$` is more robust because the effective step size adapts automatically — the algorithm takes the largest step that stays within the trust region, rather than being limited by a globally fixed penalty weight that may be too small for some parts of the parameter space and too large for others.

- **Average KL rather than max KL:** the per-state maximum KL `$D_{\text{KL}}^{\max}$` is both computationally intractable (requires constraints at every state) and overly conservative (it prevents large policy changes even in states that are rarely visited). The average KL under `$\rho_{\theta_{\text{old}}}$` is a natural relaxation: it allows more flexibility in rarely-visited states while still controlling the policy change where it matters most. The cart-pole experiment validates that this approximation does not significantly harm performance.

- **Conjugate gradient rather than direct matrix inversion:** the Fisher matrix `$H$` for neural network policies has `$d^2$` entries where `$d$` can be tens of thousands — storing it would require gigabytes of memory. Conjugate gradient computes `$H^{-1}g$` using only `$H v$` products, which can be computed on-the-fly via automatic differentiation, avoiding any explicit storage of `$H$`. This is the key computational innovation that makes TRPO scalable to large neural network policies.

- **Analytical Fisher estimator rather than empirical Fisher:** the analytical estimator computes the Hessian of the KL divergence by integrating over actions analytically, producing a lower-variance estimate that "removes the need to store a dense Hessian or all policy gradients from a batch of trajectories." For large batches, the empirical Fisher would require storing per-sample gradients for all state-action pairs, which is memory-intensive. The analytical estimator computes `$H v$` directly from the policy's distributional parameters.

- **Self-normalized importance sampling (vine method):** the self-normalized estimator `$\frac{\sum_k w_k \hat{Q}_k}{\sum_k w_k}$` where `$w_k = \pi_\theta(a_k|s)/\pi_{\theta_{\text{old}}}(a_k|s)$` automatically handles the fact that Q-values estimated from rollouts are shifted by an unknown baseline. Since adding a constant to all `$\hat{Q}_k$` cancels out, the estimator does not require explicit baseline subtraction, simplifying the implementation and reducing variance compared to unnormalized importance sampling.

- **Common random numbers (vine method):** using the same random seed for all `$K$` rollouts from a given state ensures that differences in Q-values reflect genuine differences in action quality rather than random variation in environment transitions. This makes the Q-value comparisons more reliable and reduces the variance of the advantage estimates, which in turn reduces gradient variance.

- **`$\delta = 0.01$` across all locomotion experiments:** the paper uses a single value of `$\delta$` (the KL bound) for all three locomotion tasks (swimmer, hopper, walker) without per-task tuning, demonstrating that the parameter is robust and transferable. The fact that the same `$\delta$` works across tasks with different state dimensions, action dimensions, and dynamics is evidence that the KL divergence provides a task-agnostic measure of policy change.

## 4. Key Insights and Innovations

### Innovation 1: Replacing Mixture Policies with a Divergence Constraint Frees Policy Improvement Bounds from an Impractical Policy Class

Before TRPO, the only policy improvement guarantee with non-trivial step sizes came from Kakade and Langford's (2002) conservative policy iteration (CPI), which proved that mixing a new policy `π'` with the old policy as `π_new = (1 - α)π_old + απ'` yields a bounded performance degradation proportional to `α²`. The catch — which the paper identifies as the critical barrier to practical use — is that mixture policies are "unwieldy and restrictive in practice": they require maintaining both policies and sampling from their weighted combination at every state, which is fundamentally incompatible with neural network parameterizations. A practitioner wanting to use the CPI guarantee would have to abandon the very function approximators (deep networks) that make policy optimization powerful.

The paper's central theoretical insight is that the mixture weight `α` in CPI's bound does not need to be a mixture weight at all — it can be *any* measure of how far the new policy diverges from the old one. By proving that Equation (8) holds with `α = D_TV^max(π_old, π_new)` (Theorem 1), the authors decouple the performance guarantee from the mixture policy class entirely. The bound now reads: **any pair of policies** whose total variation distance is bounded by `α` enjoys the same worst-case improvement guarantee, regardless of how those policies are parameterized. This is a conceptual move from "you must use this restrictive policy class to get guarantees" to "you can use any policy class, as long as you constrain how much it changes per iteration."

The significance of this move goes beyond enabling neural network policies. It reframes policy optimization as a **constrained optimization problem in policy space** rather than a recipe for constructing policies: the guarantee follows not from *how* you build `π_new` (mixing) but from *how far* you allow it to be from `π_old` (a divergence budget). This separates the performance improvement theory from the policy representation, making the theory modular — any policy class, any parameterization, as long as you can measure and constrain divergence. The experimental validation that `max KL` (the theoretically justified constraint) and average KL (the practical approximation) produce similar behavior on cart-pole confirms that the relaxation from total variation to KL divergence — via Pinsker's inequality — does not break the conceptual framework even though it loosens the bound.

This is a **fundamental theoretical advance**, not an incremental refinement. CPI's guarantee was effectively unusable for deep RL because mixture policies cannot represent the kind of expressive, differentiable function approximators that make deep RL work. TRPO's reformulation makes the same guarantee *actionable* by changing what the bound constrains. The experiments in Figure 4 bear this out indirectly: natural gradient, which has access to the same Fisher information geometry that TRPO uses, fails on hopper and walker because it cannot adaptively determine step sizes. The trust region constraint — and the theory that justifies it — is what makes the difference.

---

### Innovation 2: The Hard KL Constraint Is a Robust, Adaptive Alternative to Penalty-Based Step Size Selection

The natural policy gradient (Kakade, 2002) and related methods (Bagnell & Schneider, 2003; Peters & Schaal, 2008b) all face the same problem: the update direction `H^{-1} g` is well-defined, but how large a step should you take along it? The standard answer — multiply by a scalar learning rate or Lagrange multiplier `λ` — treats the step size as a hyperparameter to be tuned. The paper's diagnosis, supported by the experiments, is that a fixed `λ` cannot be simultaneously appropriate across all stages of training and across all problems: "empirically, it is hard to robustly choose the penalty coefficient."

TRPO replaces the penalty formulation `L(θ) + λ D_KL(θ_old, θ)` with a hard constraint `D_KL(θ_old, θ) ≤ δ`. This seems like a minor reformulation — after all, for any `δ` there exists some `λ` such that the constrained and penalized problems have the same solution (they are related by Lagrange duality). But the paper identifies a crucial *practical* difference: with the constraint, the effective step size adapts automatically to the local geometry of the objective. When the natural gradient direction is "steep" (large `g^T H^{-1} g`), the step `s = -√(2δ/(g^T H^{-1} g)) H^{-1} g` is automatically scaled down to stay within the KL budget. When it is "flat," the step is scaled up. A fixed penalty coefficient `λ` cannot replicate this because it doesn't "know" the local curvature — the same `λ` might produce steps that are too small in flat regions and too large in steep ones.

The experimental evidence in Figure 4 makes this concrete. Natural gradient, with its best `λ` chosen by sweeping across factors of three and selecting the best final performance, "performed well on the two easier problems, but was unable to generate hopping and walking gaits that made forward progress." TRPO, using `δ = 0.01` across all three locomotion tasks without per-task tuning, succeeded on all of them. The interpretation is not that natural gradient is a bad algorithm — it uses the exact same gradient direction as TRPO — but that **fixed step sizes are fundamentally brittle for policy optimization on difficult problems**, and the KL constraint provides a principled, adaptive alternative.

This is a **methodological insight with practical consequences** rather than a theoretical breakthrough. The theory (Theorem 1) justifies using KL divergence to bound performance change, but it doesn't dictate whether to enforce that bound via a penalty or a constraint. The paper's empirical finding — that the constraint is substantially more robust — changed how subsequent algorithms (including PPO, which built directly on TRPO) handle step sizes. The innovation lies in recognizing that the *form* of the optimization problem (constrained vs. penalized) matters independently of the *content* of the divergence measure, and that practitioners should prefer constraints when the penalty coefficient would need to vary across problems and across training.

---

### Innovation 3: Unifying Policy Gradient, Natural Gradient, and Policy Iteration as Limiting Cases of a Trust Region Framework

The paper does not merely present a new algorithm — it provides a **unifying theoretical perspective** that reveals how apparently disparate policy optimization methods relate to each other through a single optimization problem. The general formulation is Equation (13): minimize `L_{θ_old}(θ)` subject to `D_KL(θ_old, θ) ≤ δ`. The paper shows in Section 7 that three well-known algorithms emerge as special cases of this problem with different approximations:

- **Policy gradient** (Equation 19): linearize `L`, use `L2` penalty instead of KL. This is the crudest approximation — it ignores the Fisher geometry entirely, treating all parameter directions as equivalent.
- **Natural policy gradient** (Equation 18): linearize `L`, use quadratic approximation to KL. This captures the Fisher geometry but uses a fixed Lagrange multiplier `λ` instead of enforcing the constraint.
- **Policy iteration**: solve `min_π L_{π_old}(π)` exactly with no constraint at all — equivalent to taking `δ → ∞` in the trust region formulation, which is theoretically sound only if `L` perfectly predicts `η` (which it does for exact policy iteration on tabular MDPs, but not with function approximation).

This unification is significant because it provides a **spectrum of algorithms** with clear tradeoffs. At one extreme, policy iteration takes the largest possible step but relies on exact evaluation and cannot be approximated safely. At the other, policy gradient takes small, safe steps but is parameterization-dependent and inefficient. Natural gradient sits in the middle — it rescales steps by the Fisher geometry but doesn't adapt step sizes. TRPO occupies the sweet spot: it uses the Fisher geometry *and* enforces the constraint, capturing the benefits of natural gradient's invariance while adding adaptive step size selection.

This framing is more than taxonomy. It explains *why* prior methods fail on hard problems: policy gradient cannot handle high-dimensional parameterizations (because Euclidean distance is meaningless in policy space), natural gradient cannot handle challenging landscapes (because fixed step sizes are brittle), and approximate policy iteration cannot handle function approximation error (because it takes steps that are too large for `L` to remain predictive of `η`). TRPO's success on locomotion and Atari — domains where these priors individually fail — validates the claim that the constrained trust region formulation is the right intermediate point on this spectrum.

The innovation here is **conceptual, not algorithmic**. The paper doesn't just propose a new update rule; it provides a lens through which to understand the entire family of policy optimization methods as different approximations to the same underlying problem. This makes TRPO more than a one-off algorithm — it is a framework that explains both the successes and failures of prior work, and suggests how future algorithms might navigate the tradeoffs differently.

---

### Innovation 4: Verifier-Free Importance Sampling with Self-Normalization Simplifies Off-Policy Advantage Estimation

The vine sampling scheme introduces a technique that, while not the headline contribution, represents a **practical innovation with implications beyond TRPO**: the self-normalized importance sampling estimator for expected advantages (Equation 17). In standard importance sampling, estimating `E_{a ~ π_θ}[Q(s,a)]` from actions sampled from `q` requires computing `(1/K) Σ_k (π_θ(a_k|s)/q(a_k|s)) Q(s,a_k)`, which suffers from two well-known problems: high variance when importance weights are large, and sensitivity to the absolute scale of Q-values (adding a constant to all Q-values changes the estimate). The standard fix — estimating and subtracting a state-dependent baseline `V(s)` — requires a separate value function approximator and adds complexity.

The self-normalized estimator `Σ_k w_k Q_k / Σ_k w_k` where `w_k = π_θ(a_k|s)/π_old(a_k|s)` elegantly solves both problems. The denominator normalizes the weights, bounding the estimator's variance. And because the estimator computes a weighted average, adding a constant to all `Q_k` cancels out — the estimator is **automatically baseline-free**. The paper notes this explicitly: the self-normalized estimator "removes the need to use a baseline for the Q-values."

This is an **incremental but clever practical refinement**. Self-normalized importance sampling was known in the Monte Carlo literature (Owen, 2013, Chapter 8), but its application to advantage estimation in policy optimization — and the recognition that it eliminates the need for explicit baseline subtraction — was novel in the RL context. The benefit is most pronounced in the vine setting, where multiple Q-value estimates from a single state need to be compared: without self-normalization, differences in the absolute scale of Q-estimates across states would introduce noise that a baseline would need to correct. With self-normalization, the estimator is invariant to state-dependent offsets, making the per-state advantage estimates directly comparable without additional machinery.

The broader implication is that **importance sampling in policy optimization need not require value function baselines** if the estimator is properly normalized. This insight influenced subsequent work on off-policy policy gradient methods, where self-normalization (or related techniques like weighted importance sampling) became standard practice.

## 5. Experimental Analysis

### Evaluation Methodology

- **Dataset.** The experiments use two distinct domains: (1) simulated robotic locomotion tasks implemented in the MuJoCo physics simulator (Todorov et al., 2012) — specifically swimmer, hopper, and walker environments — plus the classic cart-pole balancing problem from Barto et al. (1983); and (2) seven Atari 2600 games (B. Rider, Breakout, Enduro, Pong, Q*bert, Seaquest, Space Invaders) from the Arcade Learning Environment, following the same game selection and preprocessing protocol as Mnih et al. (2013). The locomotion tasks provide continuous state and action spaces with known dynamics and clearly defined cost functions, testing TRPO's ability to learn physical control from minimal prior knowledge; the Atari games provide high-dimensional visual observations (raw pixels) and discrete action spaces, testing scalability to policies with tens of thousands of parameters.

- **Base model(s).** For locomotion tasks, the policy is a neural network with a fully-connected hidden layer of 30 units (architecture shown in Figure 3, top), outputting the mean parameters of a Gaussian action distribution with separately parameterized standard deviations. For Atari games, the policy is a convolutional neural network with two convolutional layers (16 filters each, 4×4 kernels, stride 2), followed by one fully-connected layer with 20 units, outputting softmax probabilities over the discrete action space — totaling 33,500 parameters (Figure 3, bottom). All policies are trained from scratch with random initialization. The paper does not use pretrained models; the policy architectures are purposefully simple and general to demonstrate that TRPO works without hand-engineered policy classes — in contrast to prior locomotion work that used "hand-architected policy classes that explicitly encode notions of balance and stepping" (Section 8.1, citing Tedrake et al., 2004; Geng et al., 2006; Wampler & Popović, 2009).

- **Metrics.** The primary metric is the **average cost** (equivalently, negative reward) per episode, with lower values indicating better performance. For locomotion tasks, the cost function includes a reward term for forward progress and a small quadratic penalty on joint torques (`cost(x,u) = -v_x + 10^{-5}||u||^2`), with the hopper receiving an additional bonus of +1 for non-terminal states and the walker receiving a penalty for strong foot-ground impacts. For Atari games, the metric is the total game score per episode, following the standard Arcade Learning Environment evaluation. Learning curves in Figure 4 show cost averaged across five runs with random initializations. For Atari results (Table 1), only single-run scores are reported due to computational constraints, with the paper explicitly noting that "performance varies substantially from run to run (with different random initializations of the policy), but we could not obtain error statistics due to time constraints."

- **Baselines.** The paper compares TRPO against six prior methods and two ablated variants of TRPO itself:
  - **Reward-Weighted Regression (RWR)** (Peters & Schaal, 2007): an EM-like policy search method that weights state-action pairs by their reward and fits the policy via weighted maximum likelihood.
  - **Relative Entropy Policy Search (REPS)** (Peters et al., 2010): constrains the state-action marginals `p(s,a)` using KL divergence, requiring a costly nonlinear inner-loop optimization.
  - **Cross-Entropy Method (CEM)** (Szita & Lörincz, 2006): a gradient-free method that maintains a Gaussian distribution over policy parameters, samples candidates, evaluates them, and refits the distribution to the top performers.
  - **Covariance Matrix Adaptation (CMA)** (Hansen & Ostermeier, 1996): another gradient-free method that adapts a full covariance matrix of the parameter distribution.
  - **Natural Gradient** (Kakade, 2002): the classic natural policy gradient algorithm using a fixed penalty coefficient (Lagrange multiplier) instead of the KL divergence constraint; the penalty coefficient was swept in factors of three and the best value according to final performance was selected.
  - **Deep Q-Learning** (Mnih et al., 2013): a value-based method using experience replay and a target network; results are reported directly from their paper for the Atari games.
  - **UCC-I** (Guo et al., 2014): a combination of Monte-Carlo Tree Search with supervised training on Atari games; results are reported directly from their paper.
  - **Max KL (TRPO ablation):** a variant of TRPO that enforces the maximum per-state KL divergence constraint `D_KL^max` from Equation (12) rather than the average KL; only tractable on the low-dimensional cart-pole problem.
  - **Empirical FIM (TRPO ablation):** a variant identical to single-path TRPO except that the Fisher information matrix is estimated using the outer product of gradients (the "empirical Fisher") rather than the analytical Hessian of the KL divergence.

- **Generation budget / compute accounting.** For locomotion tasks, the x-axes of learning curves (Figure 4) show the number of **simulator steps** (environment interactions), which serves as a unified measure of sample complexity across all methods. For derivative-free methods (CEM, CMA), each candidate evaluation requires one full episode, so sample complexity equals `episodes × episode_length`. For gradient-based methods (TRPO, natural gradient, RWR, REPS), samples are trajectory timesteps. For the Atari experiments, the paper reports that 500 iterations of TRPO took "about 30 hours (with slight variation between games) on a 16-core computer," but does not report total environment frames or provide a compute-normalized comparison against deep Q-learning. This is a notable omission: the deep Q-learning method in Mnih et al. (2013) is known to require millions of frames, and without frame-normalized comparisons, it is unclear whether TRPO's competitive scores in Table 1 come from better sample efficiency or simply from more environment interactions.

- **Cross-validation / statistical protocol.** For locomotion tasks, all algorithms are run with five different random initializations, and learning curves in Figure 4 show the cost averaged across these five runs, providing a measure of reproducibility and variance. For the natural gradient baseline on locomotion, the penalty coefficient was individually tuned by sweeping in factors of three and selecting the best according to final performance, giving natural gradient an advantage in the comparison. For the cart-pole task (where max KL is evaluated), results are shown as learning curves comparing max KL against single-path TRPO. For Atari games, results are from a single run per game due to computational constraints, and the paper explicitly acknowledges the lack of error statistics. The TRPO hyperparameter `δ = 0.01` was fixed across all locomotion tasks without per-task tuning, testing robustness to this key parameter.

---

### Main Quantitative Results

#### Simulated Robotic Locomotion

The central empirical claim for locomotion is that **TRPO succeeds at learning complex locomotion controllers from scratch using general-purpose neural network policies, while prior methods either fail or produce substantially worse results.** Figure 4 shows learning curves for the swimmer, hopper, walker, and cart-pole tasks, with cost on the y-axis (lower is better) and simulator steps on the x-axis.

**Swimmer (10-dimensional state, 2-dimensional action).** The swimmer is the easiest locomotion task. Both single-path TRPO and vine TRPO converge to costs of approximately −0.8 to −1.0 — essentially solving the task and achieving efficient forward swimming. Natural gradient reaches comparable asymptotic performance, achieving this level after roughly 5× more simulator steps than TRPO. RWR also solves the task but learns more slowly than TRPO. CEM and CMA, despite being gradient-free, are competitive on this low-dimensional problem, reaching similar asymptotic costs but with substantially worse sample efficiency (they require more simulator steps to converge).

**Hopper (12-dimensional state, 3-dimensional action).** The hopper represents a significant jump in difficulty due to underactuation and contact discontinuities. Single-path TRPO and vine TRPO both converge to costs near −1.0 (the theoretical minimum achievable without forward velocity, representing balanced standing), and then progressively improve beyond this to achieve forward hopping. All other methods fail to make meaningful forward progress: natural gradient, RWR, REPS, CEM, and CMA all remain at approximately −1.0 — they learn to balance the hopper upright but never discover a hopping gait that produces forward velocity. This is the key result that demonstrates TRPO's advantage: "Natural gradient performed well on the two easier problems, but was unable to generate hopping and walking gaits that made forward progress" (Section 8.1). The fact that natural gradient can balance but cannot hop suggests that the failure is not in gradient estimation quality but in step-size selection — the fixed penalty coefficient prevents natural gradient from taking the larger updates needed to cross the performance plateau from standing to hopping.

**Walker (18-dimensional state, 6-dimensional action).** The walker is the most challenging locomotion task. Single-path and vine TRPO converge to costs below −1.0 and continue improving to approximately −2.0, indicating a stable walking gait with forward velocity and smooth foot-ground contacts. All other methods — natural gradient, RWR, REPS, CEM, CMA — fail to achieve costs below −1.0, never learning to walk. The paper does not provide specific numerical cost values beyond what can be read approximately from Figure 4, but the separation between TRPO and all baselines is visually unambiguous on this task.

**Cart-pole (6-dimensional linear policy).** This is a simple baseline task with a linear policy of only six parameters, included to evaluate whether the approximations TRPO makes (specifically, using average KL rather than max KL) are reasonable when both variants are computationally tractable. Both single-path TRPO and max KL converge rapidly to near-optimal performance. The max KL variant "learned somewhat slower than our final method, due to the more restrictive form of the constraint" (Section 8.1), but the asymptotic performance is similar. This result validates the paper's claim that the average KL divergence constraint has a similar effect to the theoretically justified maximum KL divergence, supporting the practical substitution.

**Key comparisons across tasks:**

- **TRPO vs. natural gradient:** On the two harder tasks (hopper, walker), natural gradient fails completely. This is the central empirical evidence that the hard KL constraint is more robust than a fixed penalty coefficient. Natural gradient uses the exact same gradient direction as TRPO — the difference is only in step-size selection — yet it cannot solve these tasks even with its penalty coefficient individually optimized. This is not merely a hyperparameter tuning issue; it suggests that the optimal penalty coefficient varies substantially across the course of training in a way that a fixed value cannot accommodate.

- **TRPO vs. derivative-free methods:** CEM and CMA are competitive on the swimmer (10-dimensional policy) but fail on hopper and walker (with neural network policies having hundreds of parameters). The paper attributes this to the poor scaling of derivative-free methods with parameter count: "CEM and CMA are derivative-free algorithms, hence their sample complexity scales unfavorably with the number of parameters, and they performed poorly on the larger problems" (Section 8.1). This validates the paper's motivation that gradient-based methods are necessary for scaling to high-dimensional policies.

- **Single-path vs. vine TRPO:** On all three locomotion tasks, single-path and vine TRPO produce nearly identical learning curves (the paper describes them as yielding "the best solutions" and appears to plot them together or in close overlap in Figure 4). This is significant because it shows that the simpler single-path method — which requires no state resets and can be deployed on physical systems — performs comparably to the lower-variance vine method when both are combined with the trust region constraint.

- **Empirical FIM vs. analytical FIM:** The paper reports that the empirical FIM variant (which estimates the Fisher matrix from the outer product of gradients) shows "the rate of improvement in the policy is similar to the empirical FIM, as shown in the experiments" (Section 6). However, Figure 4 does not explicitly separate this ablation, making it difficult to assess the precise difference from the provided plots. The paper's claim is based on computational advantages of the analytical estimator (no need to store dense Hessians or per-sample gradients), with similar empirical performance.

#### Atari Games from Images

**Table 1 reports raw game scores for TRPO (single-path and vine) compared to a random agent, a human expert (from Mnih et al., 2013), deep Q-learning (DQN from Mnih et al., 2013), and UCC-I (Guo et al., 2014) across seven games.**

The headline result is that TRPO achieves "reasonable scores" on all games, matching or approaching the performance of prior specialized methods on some games while falling substantially short on others. The paper frames this not as a claim of superiority but as a demonstration of generality: "Unlike the prior methods, our approach was not designed specifically for this task. The ability to apply the same policy search method to methods as diverse as robotic locomotion and image-based game playing demonstrates the generality of TRPO" (Section 8.2).

**Game-by-game results from Table 1 (TRPO single-path / TRPO vine / DQN / UCC-I / Human):**

- **B. Rider:** 1425.2 / 859.5 / 4092 / 5702 / 7456. TRPO substantially underperforms DQN and UCC-I on this game, though both TRPO variants exceed the random baseline (354).
- **Breakout:** 10.8 / 34.2 / 168.0 / 380 / 31.0. The vine variant more than triples the score of the single-path variant, but both trail DQN (168.0) and UCC-I (380). Notably, both TRPO variants fall short of human performance (31.0 — though the paper notes this is the human score from Mnih et al., 2013, which appears anomalously low compared to typical Breakout scores and may reflect a different evaluation protocol).
- **Enduro:** 534.6 / 430.8 / 470 / 741 / 368. TRPO single-path surpasses DQN (534.6 vs. 470) and approaches UCC-I (741). Both TRPO variants exceed human performance (368, from Mnih et al., 2013).
- **Pong:** 20.9 / 20.9 / 20.0 / 21 / −3.0. Both TRPO variants match DQN (20.0) and UCC-I (21), essentially achieving human-parity performance on this game.
- **Q*bert:** 1973.5 / 7732.5 / 1952 / 20025 / 18900. The vine variant achieves a substantial improvement over single-path (7732.5 vs. 1973.5), surpassing DQN (1952) but still far below UCC-I (20025) and human performance (18900). The factor-of-4 improvement from vine over single-path on this particular game suggests that the vine method's lower-variance advantage estimates are especially beneficial when the Q-value landscape is complex.
- **Seaquest:** 1908.6 / 788.4 / 1705 / 2995 / 28010. Single-path TRPO slightly exceeds DQN (1908.6 vs. 1705) but trails UCC-I (2995). Human performance on Seaquest (28010) is an order of magnitude higher than any learning-based method.
- **Space Invaders:** 568.4 / 450.2 / 581 / 692 / 3690. TRPO single-path (568.4) approximately matches DQN (581) and trails UCC-I (692).

**Key observations from the Atari results:**

- **Single-path vs. vine inconsistency across games:** On Breakout, vine triples single-path (34.2 vs. 10.8); on Q*bert, vine nearly quadruples single-path (7732.5 vs. 1973.5); but on Seaquest, single-path more than doubles vine (1908.6 vs. 788.4), and on Space Invaders, single-path outperforms vine (568.4 vs. 450.2). The paper does not provide an explanation for this inconsistency. The vine method should theoretically produce lower-variance estimates, which should translate to better or equal performance, but the empirical results show no consistent advantage. This may reflect the different exploration properties of the vine method's sampling distribution (which can use uniform sampling for discrete actions) or sensitivity to the rollout set selection.

- **No frame-normalized comparison:** The paper reports that 500 TRPO iterations took approximately 30 hours on a 16-core machine, but does not report how many environment frames this corresponds to per game. DQN (Mnih et al., 2013) was trained for 10 million frames on most games and 50 million on some. Without frame counts, it is impossible to determine whether TRPO is more or less sample-efficient than DQN. The competitive scores in Table 1 may reflect TRPO receiving more environment interactions, or they may reflect genuinely better sample efficiency — the paper provides no basis for distinguishing these possibilities.

- **UCC-I's strong performance:** UCC-I (Guo et al., 2014), which combines Monte-Carlo Tree Search with supervised training on offline data, outperforms TRPO on all games except Enduro (where single-path TRPO scores 534.6 vs. UCC-I's 741) and Pong (where methods differ by 0.1 points). This is expected, as UCC-I leverages offline planning and demonstration data, while TRPO learns purely from online interaction. The paper's comparison to UCC-I serves primarily to show TRPO's performance relative to the state of the art, not to claim superiority.

- **No learning curves for Atari:** Unlike the locomotion experiments (Figure 4), the Atari results are only final scores after 500 iterations. The paper does not show how performance evolved over training, making it impossible to assess whether TRPO was still improving at 500 iterations, whether it suffered from instability, or whether it exhibited monotonic improvement as the theory would suggest.

---

### Ablation Studies and Robustness Checks

The paper's ablation studies are relatively limited compared to modern standards, but several key design choices are empirically evaluated:

- **Max KL vs. average KL divergence constraint:** The max KL variant — which enforces the theoretically justified per-state maximum KL divergence `D_KL^max` rather than the average KL — was "only tractable on the cart-pole problem" due to the computational difficulty of enforcing per-state constraints in continuous or large state spaces. On cart-pole (with a 6-parameter linear policy), max KL "learned somewhat slower than our final method, due to the more restrictive form of the constraint, but overall the result suggests that the average KL divergence constraint has a similar effect as the theoretically justified maximum KL divergence" (Section 8.1). This is the paper's only direct empirical validation of its central approximation (replacing max KL with average KL). The fact that max KL is slower is consistent with it being more conservative (constraining policy change at rarely-visited states as much as at frequently-visited ones), but the asymptotic performance being similar supports the paper's claim that the average KL is a reasonable practical substitute.

- **Analytical Fisher vs. empirical Fisher information matrix:** The paper reports that using the analytical estimator (Hessian of the KL divergence) produces a "rate of improvement in the policy [that] is similar to the empirical FIM" (the outer product of gradients). The paper does not provide a separate figure isolating this comparison, but states that the analytical estimator was used in all reported experiments and that it has computational benefits — specifically, it "removes the need to store a dense Hessian or all policy gradients from a batch of trajectories" (Section 6). The analytical estimator integrates over actions analytically at each state, producing a lower-variance estimate of the Fisher matrix without requiring per-action gradient storage.

- **Single-path vs. vine sampling procedure:** This is tested implicitly across all experiments, with both variants run on locomotion and Atari tasks. On locomotion, the two variants produce nearly identical learning curves (Figure 4), suggesting that the lower variance of the vine estimator does not translate to meaningfully faster learning in practice — possibly because the variance reduction is offset by the computational overhead, or because the single-path method's variance is already sufficiently low when using the trust region constraint. On Atari games, results are mixed: vine substantially outperforms single-path on Breakout (34.2 vs. 10.8) and Q*bert (7732.5 vs. 1973.5), but underperforms on Seaquest (788.4 vs. 1908.6) and Space Invaders (450.2 vs. 568.4). The paper does not analyze this game-dependent variation, but notes that the vine method's choice of sampling distribution `q(·|s_n)` — which can be uniform (as used on Atari for exploration) or on-policy — affects results.

- **Fixed `δ = 0.01` across all locomotion tasks:** The KL divergence bound `δ` is the single most important hyperparameter in TRPO, controlling the size of the trust region. The paper uses `δ = 0.01` for all three locomotion tasks (swimmer, hopper, walker) without per-task tuning. The fact that a single value works across tasks with different state dimensions (10, 12, 18), action dimensions (2, 3, 6), and dynamics is evidence that `δ` is a relatively robust parameter whose meaning transfers across problems. However, the paper does not perform a sensitivity analysis by varying `δ` and showing how performance changes, which would quantify the parameter's robustness.

- **Policy architecture:** All locomotion tasks use the same neural network architecture (one hidden layer with 30 units, Gaussian output) and the same cost function structure (linear forward progress reward + quadratic control penalty). This demonstrates that TRPO can work with a fixed architecture across tasks, but it also means the paper does not explore how TRPO's performance scales with network size or architecture. The Atari architecture (two convolutional layers + one fully-connected layer) is held fixed across all seven games, with no architecture search.

---

### Critical Assessment

**Claim 1: TRPO gives monotonic improvement with little tuning of hyperparameters.** The paper's abstract states that "despite its approximations that deviate from the theory, TRPO tends to give monotonic improvement, with little tuning of hyperparameters." The experiments partially support this claim, but with important caveats.

The locomotion learning curves in Figure 4 show that TRPO's cost curves decrease smoothly and consistently across all three tasks, without the oscillations or instability that might characterize methods without the KL constraint. This is consistent with monotonic improvement, but the paper does not actually report per-iteration cost values to verify that each individual update strictly improves performance — the learning curves are averaged across runs and smoothed, which could mask occasional regressions. The claim of "little tuning" is supported by the use of `δ = 0.01` across all locomotion tasks, but the paper does not report results for other `δ` values, so it is unclear whether `δ = 0.01` was chosen after trying other values that failed, or whether it was selected a priori and simply reported as working.

For Atari games, the paper provides only final scores after 500 iterations; there are no learning curves, so monotonicity cannot be assessed. The single-run results also mean that any claims about reliability or robustness in the Atari domain are based on a single trial per game, which is insufficient to establish statistical reliability given the known high variance of Atari training runs.

**Claim 2: The KL constraint is more robust than a fixed penalty coefficient.** This claim is directly tested by the comparison between TRPO and natural gradient on the locomotion tasks. The evidence is strong: natural gradient (with its penalty coefficient individually optimized via grid search) fails to learn forward locomotion on hopper and walker, while TRPO succeeds. This is the paper's most compelling empirical result — it demonstrates not just that TRPO is better, but that the difference in step-size selection (constraint vs. penalty) is the *causal factor* behind the difference in performance, since both methods use the same gradient direction.

However, the claim should be qualified in two ways. First, the natural gradient baseline uses a fixed penalty coefficient across all iterations (though optimized for final performance). More sophisticated penalty schedules — annealing the penalty over time, or adapting it based on observed improvement — might close the gap. The paper's claim is specifically about *fixed* penalty coefficients, not about all possible penalty-based methods. Second, the experiments do not isolate whether the improvement comes from the constraint itself or from the *line search* that TRPO uses to enforce it. The line search (backtracking from the full step until the KL constraint is satisfied *and* the surrogate loss improves) provides an additional safety mechanism beyond the constraint formulation alone. A natural gradient method with an equivalent line search on the KL divergence might perform better than the fixed-penalty version tested.

**Claim 3: TRPO can optimize high-dimensional neural network policies for challenging control tasks.** Supported. The Atari experiments successfully train a 33,500-parameter convolutional neural network from raw pixels — a policy scale that is orders of magnitude beyond what derivative-free methods can handle. The locomotion experiments train neural network policies with dozens to hundreds of parameters for tasks that prior work had only solved with hand-engineered policy classes. The fact that TRPO learns swimming, hopping, and walking "with general-purpose policies and simple cost functions, using minimal prior knowledge" (Section 8.1) is a genuine advance over the state of the art at the time of publication.

However, the Atari results reveal a gap between "can optimize" and "produces competitive final performance." TRPO's Atari scores (Table 1) are reasonable but not state-of-the-art — UCC-I outperforms it on nearly all games, and DQN matches or exceeds it on several. The paper does not claim superiority on Atari; it claims demonstrative value. But without frame-normalized comparisons, we cannot tell whether TRPO is sample-efficient, and without learning curves, we cannot tell whether 500 iterations was sufficient for convergence or whether performance was still improving. These omissions limit the strength of the Atari results as evidence for scalability.

**Claim 4: The vine method provides lower-variance advantage estimates.** The paper claims that "the vine method gives much better estimates of the advantage values" (Section 5.2), but this is a theoretical claim about the estimator's variance, not an empirical claim about its downstream effect on policy optimization. The actual learning curves (Figure 4) show that vine and single-path produce nearly identical performance on locomotion — meaning the variance reduction does not translate to faster or better learning in practice. On Atari, vine sometimes helps (Breakout, Q*bert) and sometimes hurts (Seaquest, Space Invaders), with no consistent pattern. The paper does not report the actual variance of the advantage estimates under either method, so the claim about lower variance remains an untested theoretical assertion in the context of the reported experiments.

**Missing experiments that would have strengthened the paper:**

- **Sensitivity analysis for `δ`:** Varying `δ` across orders of magnitude on at least one locomotion task and showing how performance varies would quantify the parameter's robustness and provide guidance for practitioners. The paper's claim of "little tuning" would be much stronger if accompanied by evidence that performance is relatively flat for `δ` in a broad range (e.g., 0.001 to 0.1).

- **Frame-normalized Atari comparison:** Reporting the total number of environment frames used by TRPO and comparing against DQN's frame budget would allow assessing sample efficiency. If TRPO achieves competitive scores with fewer frames, that strengthens the scalability claim. If it uses comparable or more frames, the contribution is more about reliability (monotonic improvement) than about sample efficiency.

- **Per-iteration monotonicity check:** Reporting the percentage of iterations in which TRPO actually improved the policy (vs. iterations where the line search failed and the policy was not updated) would directly test the paper's theoretical claim of monotonic improvement. The averaged learning curves in Figure 4 cannot reveal occasional regressions.

- **Ablation of the line search:** Running TRPO without the line search (i.e., always taking the full CG step) and comparing performance would isolate whether the constraint formulation alone or the constraint-plus-line-search combination is responsible for the robustness over natural gradient.

- **Scaling with network size:** Training policies with varying numbers of hidden units on a single locomotion task and showing how TRPO's learning speed and final performance scale would test the claim that TRPO is specifically suited for large nonlinear policies. The experiments use a fixed architecture (30 hidden units for locomotion, 33,500 parameters for Atari) and do not explore scaling behavior.

- **Direct variance comparison of vine vs. single-path:** Computing the empirical variance of the advantage estimates under both methods from the same set of environment interactions would directly test the paper's theoretical claim about vine's lower-variance estimates. The mixed Atari results suggest that whatever theoretical advantage vine has may be offset by other factors (choice of sampling distribution, overhead of resetting to rollout states, reduced exploration diversity).

**Weaknesses in evaluation:**

- **Small number of seeds for Atari:** Single-run results with no error bars make it impossible to assess whether differences between methods are statistically significant or reflect random variation. The paper acknowledges this, but the limitation is severe for a domain where training variance is known to be high.

- **No held-out test set:** All reported results are on the training environments. For locomotion, the stochasticity of the physics simulator makes overfitting less of a concern, but for Atari, standard practice involves evaluating on separate test episodes with exploratory noise disabled. The paper does not report whether final Atari scores are from training or evaluation conditions.

- **Natural gradient baseline may not be optimally tuned:** While the paper sweeps the penalty coefficient, it does not report trying different learning rate schedules, momentum, or other optimizer configurations that might improve natural gradient's performance. The failure of natural gradient on hopper and walker is the paper's most important comparative result, and any residual doubt about the quality of the natural gradient implementation weakens this conclusion.

- **No comparison to trust-region-free policy gradient methods with modern optimizers (Adam, RMSProp):** At the time of TRPO's publication (2015), adaptive stochastic optimization methods were becoming standard in deep learning but were not yet widely applied in RL. Comparing TRPO against a policy gradient method using Adam with a well-tuned learning rate would help isolate the benefit of the trust region constraint from the benefit of modern optimization. The paper's baseline comparisons (CEM, CMA, natural gradient, RWR, REPS) are all from the pre-deep-RL era and may not represent the strongest possible gradient-based baseline.

**Summary of evidence quality:** The locomotion experiments provide strong, well-replicated evidence that TRPO's constrained optimization approach succeeds where prior policy optimization methods fail on challenging continuous control tasks. The Atari experiments demonstrate scalability to high-dimensional visual inputs but provide weaker evidence about sample efficiency or statistical reliability. The comparison between TRPO and natural gradient is the paper's most important empirical contribution — it isolates step-size selection as a causal factor and shows that the trust region constraint provides a meaningful improvement over fixed penalties. The paper's ablation of max KL on cart-pole provides some validation of the average KL approximation, but this is limited to a single low-dimensional task and leaves open how well the approximation holds in higher dimensions.

## 6. Limitations and Trade-offs

### Computational Cost of the Vine Method Limits Applicability to Simulation-Only Settings

**The assumption or constraint.** The vine sampling procedure, which the paper claims "gives much better estimates of the advantage values" (Section 5.2), achieves its variance reduction by performing multiple independent rollouts from selected states in the rollout set. This fundamentally requires the ability to reset the environment to arbitrary previously-visited states — a capability the paper explicitly acknowledges is "typically only possible in simulation" (abstract, Section 5.2). The paper contrasts this with single-path TRPO, which "requires no state resets and can be directly implemented on a physical system (Peters & Schaal, 2008b)."

**The consequence.** The vine method — which provides the theoretically lower-variance advantage estimates that should make policy updates more reliable — is categorically unavailable for real-world robotic systems, online learning from human interaction, or any setting where the environment cannot be checkpointed and replayed. This means that for the most impactful potential application of TRPO (learning control policies on physical robots), practitioners are restricted to the single-path variant. Given that real-world robotics is one of the primary domains the paper motivates TRPO for — the abstract highlights "learning simulated robotic swimming, hopping, and walking gaits" and Section 9 envisions "learning robotic control policies that use vision and raw sensory data as input" — this is a significant scope limitation.

Furthermore, the vine method incurs a substantial computational overhead even in simulation: for each state in the rollout set, the method performs K rollouts (one per sampled action) rather than simply continuing the original trajectory. The paper does not quantify this overhead in terms of additional simulator calls per iteration, but the structure of the algorithm implies a multiplicative increase in environment interactions proportional to the size of the rollout set times the number of actions sampled per state. A practitioner weighing whether to use vine or single-path must trade off this additional simulation budget against the variance reduction, and the paper provides no guidance on when this tradeoff is worthwhile.

**What evidence exists in the paper.** The locomotion learning curves (Figure 4) show that single-path and vine TRPO produce nearly identical final performance and learning speed on all three tasks (swimmer, hopper, walker). This is direct evidence that the theoretical variance reduction of the vine method does not translate to meaningfully better policy optimization in practice — at least for these continuous control tasks with modest policy sizes. On Atari games (Table 1), results are inconsistent: vine substantially outperforms single-path on Breakout (34.2 vs. 10.8) and Q*bert (7732.5 vs. 1973.5), but underperforms on Seaquest (788.4 vs. 1908.6) and Space Invaders (450.2 vs. 568.4). The paper provides no explanation for this game-dependent reversal and no direct measurement of the variance of the advantage estimates to confirm that vine actually achieves lower variance as claimed.

**Mitigation status.** The paper does not attempt to mitigate this limitation. It presents vine as an alternative sampling scheme with different tradeoffs (lower variance vs. requirement for state resets) and leaves the choice to the practitioner. The observation that single-path performs comparably to vine on locomotion suggests that the practical benefit of vine may be smaller than the theoretical analysis implies, but this is not explored or explained. The paper does not propose hybrid approaches (e.g., using vine only for a subset of states, or adaptively choosing between single-path and vine based on estimated advantage variance) that might capture vine's benefits with lower overhead.

---

### The Difficulty Estimation Cost Is Not Accounted for in the Headline Efficiency Claims

**The assumption or constraint.** TRPO's per-iteration update requires computing the natural gradient direction `H^{-1} g`, which involves solving a linear system via conjugate gradient. Each CG iteration requires computing a Hessian-vector product `H v`, which costs approximately one backpropagation pass through the policy network. The paper states that the CG procedure is "altogether only slightly more expensive than computing the gradient itself" (Section 6), but this claim is made without quantification — the paper never reports the number of CG iterations used in practice, the wall-clock time per TRPO update compared to a standard policy gradient update, or how the computational cost scales with policy size and batch size.

**The consequence.** A practitioner choosing between TRPO and a simpler method (e.g., standard policy gradient with a modern adaptive optimizer like Adam, or derivative-free methods on smaller problems) needs to understand the computational premium they pay for TRPO's trust region constraint. If the CG procedure takes 20 iterations and each iteration costs as much as computing the policy gradient, then one TRPO update costs roughly 20× the computation of one policy gradient step — yet the paper only compares methods by simulator steps (sample complexity), not by wall-clock time or total FLOPs. For problems where simulator calls are cheap but policy optimization is the bottleneck (e.g., simple physics simulators, game environments with fast forward models), the computational cost of the CG procedure could dominate, making TRPO less attractive than its sample-efficiency curves suggest.

Additionally, the analytical Fisher estimator — which the paper prefers for its computational benefits — requires computing the Hessian of the KL divergence analytically for each state. For complex policy architectures (recurrent networks, policies with sophisticated output distributions), this analytical Hessian may not be available or may be substantially more expensive to compute than the empirical Fisher (outer product of gradients). The paper's experiments use relatively simple architectures (one hidden layer with 30 units for locomotion; two convolutional layers plus one fully-connected layer for Atari), and the computational claims may not generalize to more complex policy classes.

**What evidence exists in the paper.** The paper provides almost no direct evidence about computational overhead. The Atari experiments report that "500 iterations of our algorithm took about 30 hours (with slight variation between games) on a 16-core computer" (Section 8.2), but this single number is insufficient for comparison — the paper does not report how long 500 iterations of a simpler policy gradient method would take on the same hardware, nor how many environment frames were processed. The locomotion experiments are evaluated purely in terms of simulator steps (Figure 4), with no timing data. The paper does not report the average number of CG iterations per TRPO update, the fraction of computation spent on CG vs. sampling vs. line search, or how these costs scale with network size.

**Mitigation status.** The paper does not address this limitation, nor does it acknowledge it as one. The claim that CG is "slightly more expensive" than computing the gradient is presented without supporting evidence. The paper's recommendation of the analytical Fisher estimator is motivated by memory efficiency (avoiding storage of dense Hessians or per-sample gradients) rather than computational efficiency, and no benchmarks are provided comparing wall-clock time of TRPO against baselines.

---

### The Theoretical Monotonic Improvement Guarantee Relies on Assumptions That Are Violated in Practice

**The assumption or constraint.** The paper's central theoretical contribution — Theorem 1 and the resulting Algorithm 1 — guarantees monotonic improvement under three assumptions: (1) exact evaluation of the advantage function `A^{\pi}(s, a)` at all states and actions, (2) the use of the maximum KL divergence `D_{\text{KL}}^{\max}` rather than the average KL, and (3) the penalty coefficient `C = 2\epsilon\gamma/(1-\gamma)^2` derived from the worst-case bound in Equation (10). The practical TRPO algorithm violates all three: it uses sample-based Monte Carlo estimates of Q-values (Section 5), replaces `D_{\text{KL}}^{\max}` with the average KL divergence `\bar{D}_{\text{KL}}` (Section 4), and replaces the theory-derived penalty `C` with a hard constraint on `\bar{D}_{\text{KL}}` with a user-chosen parameter `\delta = 0.01` (Section 6) — a value that is orders of magnitude larger than what the theory would prescribe (since `C` would produce "prohibitively small steps").

The paper is transparent about these deviations, listing them explicitly in Section 6: "The theory justifies optimizing a surrogate loss with a penalty on KL divergence. However, the large penalty coefficient... leads to prohibitively small steps, so we would like to decrease this coefficient" and "The constraint on `D_{\text{KL}}^{\max}(\theta_{\text{old}}, \theta)` is hard for numerical optimization and estimation, so instead we constrain `\bar{D}_{\text{KL}}(\theta_{\text{old}}, \theta)`" and "Our theory ignores estimation error for the advantage function."

**The consequence.** The practical TRPO algorithm described in Section 6 has no formal guarantee of monotonic improvement. The theoretical guarantee applies to Algorithm 1, which TRPO approximates, but the approximations are sufficiently coarse that the guarantee does not carry over. A practitioner deploying TRPO on a new problem cannot rely on the theory to ensure that performance will never degrade — they are instead relying on the empirical observation that TRPO "tends to give monotonic improvement" (abstract). This is a weaker claim than what the paper's title ("Trust Region Policy Optimization") and theoretical development might suggest to a casual reader.

More specifically, the replacement of `D_{\text{KL}}^{\max}` with `\bar{D}_{\text{KL}}` means that the new policy can differ arbitrarily from the old policy in states with low visitation probability under `\rho_{\theta_{\text{old}}}`, even though these states might be visited frequently under the *new* policy `\pi_{\theta_{\text{new}}}`. The bound in Theorem 1 requires constraining divergence at *all* states precisely because the new policy's visitation distribution `\rho_{\tilde{\pi}}` can shift to states that were rarely visited under the old policy. By constraining only the average KL under the old distribution, TRPO leaves open the possibility that the policy changes dramatically in states that become important after the update — exactly the kind of distribution shift that the original bound was designed to prevent. The empirical success of TRPO suggests that this failure mode is rare in practice, but the theoretical foundation does not explain why.

Furthermore, the use of `\delta = 0.01` as a fixed hyperparameter — while shown to work across the three locomotion tasks — is not theoretically justified. The theory would give `\delta \approx 2\epsilon\gamma/(1-\gamma)^2` if the penalty form were preserved and `\epsilon` were estimated, which would typically yield a value much smaller than 0.01 (since `\epsilon` can be large and `\gamma` is close to 1). The fact that a `\delta` three orders of magnitude larger than the theory prescribes works well suggests that the bound in Equation (10) is extremely loose, and that the practical relationship between KL divergence and policy performance is far more forgiving than the worst-case analysis indicates. This is good news for practitioners but means the theory provides no guidance on how to set `\delta` — it must be tuned empirically.

**What evidence exists in the paper.** The paper does not directly measure whether TRPO achieves monotonic improvement on a per-iteration basis. The locomotion learning curves (Figure 4) show smoothly decreasing costs averaged across five runs, but averaging can mask occasional regressions in individual runs, and the curves are plotted against simulator steps (not iterations), so per-iteration cost values are not visible. The Atari experiments (Table 1) report only final scores after 500 iterations, providing no evidence about the trajectory of improvement. The paper never reports the percentage of iterations where the line search successfully found an improving step vs. iterations where the policy was not updated.

The only direct empirical test of a theoretical approximation is the `max KL` experiment on cart-pole, which shows that max KL "learned somewhat slower than our final method, due to the more restrictive form of the constraint, but overall the result suggests that the average KL divergence constraint has a similar effect as the theoretically justified maximum KL divergence" (Section 8.1). This is limited to a single low-dimensional task (6-parameter linear policy) and does not validate the approximation for the higher-dimensional, nonlinear policies that are TRPO's primary target.

**Mitigation status.** The paper does not attempt to close the gap between theory and practice. The violations of the theoretical assumptions are acknowledged but treated as necessary practical compromises rather than problems to be solved. The line search (Section 6, Appendix C) provides a partial safeguard: by backtracking until the KL constraint is satisfied *and* the surrogate loss improves, TRPO ensures that each accepted update is locally improving in the surrogate objective while respecting the approximate KL constraint. However, this does not guarantee improvement in the true objective `\eta`, since the surrogate `\hat{L}` is itself a noisy estimate of `L`, which is only a first-order approximation to `\eta`. The paper does not discuss whether future work could derive weaker but still meaningful guarantees that account for these approximations.

---

### Single Benchmark Domain (MATH) and Single Model Family Constrain the Generality of the Findings

**The assumption or constraint.** All of TRPO's empirical validation is conducted on two benchmark domains: simulated robotic locomotion in MuJoCo (swimmer, hopper, walker, plus cart-pole) and seven Atari 2600 games. While diverse within each domain, these tasks share important structural properties: all are episodic, all have well-defined cost/reward functions, all involve control from proprioceptive or visual state, and none involve long-horizon credit assignment across hundreds or thousands of timesteps. The paper does not test TRPO on domains with fundamentally different characteristics: sparse-reward environments where meaningful learning signals occur only at episode termination (e.g., maze navigation, Montezuma's Revenge), partially observable settings where the state must be inferred from history (beyond the frame-stacking used in Atari), multi-agent scenarios, or tasks requiring exploration over very long horizons.

Furthermore, all experiments use the same neural network architectures within each domain (30-unit hidden layer for locomotion; two convolutional layers with 16 filters each plus 20-unit fully-connected layer for Atari). The paper makes no attempt to vary architecture depth, width, or activation functions to test whether TRPO's performance is sensitive to these choices.

**The consequence.** A practitioner considering TRPO for a new domain — say, natural language processing, recommendation systems, or real-world robotics with sparse rewards — cannot infer from the paper's evidence whether TRPO will work, what `\delta` value to use, or how the trust region constraint interacts with the credit assignment challenges of their domain. The finding that `\delta = 0.01` works across all three locomotion tasks is encouraging but provides only a single data point: the paper does not explore whether this value transfers to domains with different action dimensionalities, different reward scales, or different episode lengths. In a domain where a "small" KL divergence corresponds to a much larger or smaller change in behavior (due to differences in the policy's output distribution sensitivity), `\delta = 0.01` might be either too conservative (stalling progress) or too aggressive (allowing destructive updates).

The single-architecture design also limits what can be inferred about TRPO's scalability claims. The paper claims TRPO is "effective for optimizing large nonlinear policies" and "scalable" to "tens of thousands of parameters" (Section 1, abstract), but the evidence for scaling is binary: TRPO works on 6-parameter linear policies (cart-pole), on ~100-parameter neural networks (locomotion), and on 33,500-parameter convolutional networks (Atari). There is no systematic scaling study showing how TRPO's sample efficiency, wall-clock time, or final performance vary as the number of parameters increases. A practitioner wanting to use TRPO with a much larger policy (e.g., a ResNet with millions of parameters on a complex visual task) has no evidence that the CG procedure or the KL constraint remain well-behaved at that scale.

**What evidence exists in the paper.** The paper's evidence is entirely from locomotion (MuJoCo) and Atari. Table 2 in the appendix lists the experimental parameters used, confirming that a single set of hyperparameters (`\delta = 0.01`, batch size, CG damping coefficient, etc.) was used across tasks within each domain. The paper does not discuss cross-domain transfer of hyperparameters. The Atari results (Table 1) provide evidence of domain generality — the same algorithm works on locomotion (continuous control) and Atari (discrete control from pixels) — but the limited battery of seven games and the lack of learning curves or multiple seeds weakens these conclusions.

**Mitigation status.** The paper acknowledges the scope of its experiments implicitly through its choice of domains and explicit reporting of architecture details. It does not claim generality beyond what is demonstrated. However, it also does not discuss the limitations of the domains it chose, nor does it suggest what kinds of tasks might be challenging for TRPO. The discussion (Section 9) focuses on future extensions (recurrent policies, model-based variants, perception-and-control unification) rather than on characterizing the boundaries of the current method.

---

### The Advantage Estimation Procedure Has No Protection Against Systematic Bias from Function Approximation or Off-Policy Data

**The assumption or constraint.** TRPO's objective and gradient depend on estimates of `Q^{\theta_{\text{old}}}(s, a)` — the state-action value function of the old policy. In the single-path method, these are Monte Carlo returns computed from actual trajectory outcomes: `\hat{Q}(s_t, a_t) = \sum_{l=0}^{T-t-1} \gamma^l c(s_{t+l})`. In the vine method, they are computed from short rollout trajectories starting from `(s_n, a_{n,k})`. Both estimators are unbiased but high-variance. The paper's theory (Algorithm 1) assumes exact advantage values; the practical algorithm (Section 6) acknowledges that "our theory ignores estimation error for the advantage function" and notes that Kakade & Langford (2002) considered this error and "the same arguments would hold in the setting of this paper, but we omit them for simplicity."

**The consequence.** Monte Carlo returns, while unbiased, have variance that grows with the horizon. For problems with long episodes (hundreds or thousands of timesteps) or high environment stochasticity, the Q-value estimates used to construct the surrogate loss `\hat{L}(\theta)` can be extremely noisy. The trust region constraint bounds how much the policy can change per iteration, but it does nothing to address the quality of the gradient estimate within that region. If the advantage estimates are dominated by noise, the natural gradient direction `H^{-1} g` points in an essentially random direction, and the trust region constraint simply ensures that the random step is not too large — it cannot salvage the signal from the noise.

This limitation is particularly acute for domains with sparse rewards, where most Monte Carlo returns are zero or constant and informative signals appear only at rare states. In such domains, the variance of the advantage estimator can be so high relative to its mean that no practical number of trajectory samples can produce a reliable gradient. The paper's experiments use dense cost functions: the locomotion tasks have a per-timestep reward for forward velocity and control penalty, and Atari games provide frequent score changes. The paper provides no evidence about TRPO's behavior in sparse-reward settings, which constitute a significant fraction of practical RL problems.

Additionally, the importance sampling weights `\pi_\theta(a|s)/\pi_{\theta_{\text{old}}}(a|s)` in the single-path objective can become large if the new policy places high probability on actions that were unlikely under the old policy. This inflates the variance of the objective estimate and can cause the surrogate loss `\hat{L}(\theta)` to be a poor approximation of `L(\theta)`, even before considering the gap between `L` and `\eta`. The trust region constraint partially mitigates this by preventing `\pi_\theta` from diverging too far from `\pi_{\theta_{\text{old}}}`, but within the allowed KL budget, importance weights can still vary substantially — especially for policies with concentrated action distributions (e.g., Gaussian policies with small variance).

**What evidence exists in the paper.** The paper provides no direct measurements of advantage estimation quality: no variance estimates for the Q-values, no comparison between the Monte Carlo returns and true Q-values (which could be approximated via many rollouts), and no analysis of how importance weight variance affects the objective estimate. The learning curves (Figure 4) show that TRPO makes steady progress on all locomotion tasks, suggesting that the advantage estimates are sufficiently informative for these environments, but this is an existence proof rather than a characterization of when the estimation procedure works. The vine method is claimed to provide "much better estimates" (Section 5.2) but this claim is never empirically verified — the downstream performance of vine vs. single-path is indistinguishable on locomotion, which could mean vine's estimates are not actually better, or that the advantage estimation quality is not the bottleneck.

**Mitigation status.** The paper does not address advantage estimation error as a limitation. The vine method is presented as a way to reduce variance, but its effectiveness is not measured, and it does not address bias. The paper does not explore alternative advantage estimators: generalized advantage estimation (GAE), which would later become standard practice in policy gradient methods, uses a λ-weighted combination of Monte Carlo returns and bootstrapped value estimates to trade off bias and variance — a technique that would directly address the high-variance limitation of pure Monte Carlo returns. The paper also does not consider using a learned value function `V_\phi(s)` as a baseline to reduce variance in the advantage estimates, which was already standard practice in actor-critic methods at the time of TRPO's publication.

---

### The `max KL` Approximation to Average KL Is Validated Only on a Single Trivial Task

**The assumption or constraint.** The paper's central practical approximation — replacing the theoretically justified `D_{\text{KL}}^{\max}(\theta_{\text{old}}, \theta) = \max_s D_{\text{KL}}(\pi_{\theta_{\text{old}}}(\cdot|s) \| \pi_\theta(\cdot|s))` with the average KL divergence `\bar{D}_{\text{KL}}^{\rho_{\theta_{\text{old}}}}(\theta_{\text{old}}, \theta) = \mathbb{E}_{s \sim \rho_{\theta_{\text{old}}}}[D_{\text{KL}}(\pi_{\theta_{\text{old}}}(\cdot|s) \| \pi_\theta(\cdot|s))]` — is justified on the grounds that the max constraint is "impractical to solve due to the large number of constraints" (Section 4) and that "empirically, it is hard to robustly choose the penalty coefficient, so we use a hard constraint instead of a penalty" (Section 6). The only empirical validation of this approximation is the `max KL` experiment on cart-pole, described in a single sentence in Section 8.1.

**The consequence.** The average KL constraint is strictly weaker than the max KL constraint — it allows the policy to change arbitrarily in states that have low visitation probability under `\rho_{\theta_{\text{old}}}`, as long as the average (weighted by visitation frequency) remains below `\delta`. This is a potential failure mode: if the new policy `\pi_{\theta_{\text{new}}}` visits states that were rarely visited under `\pi_{\theta_{\text{old}}}`, and the policy changed significantly in those states (because they contributed little to the average KL constraint), then the performance bound in Theorem 1 does not apply and the update could degrade performance. The average KL approximation assumes that the state distribution shift between `\pi_{\theta_{\text{old}}}` and `\pi_{\theta_{\text{new}}}` is small enough that states with low `\rho_{\theta_{\text{old}}}` visitation remain low `\rho_{\theta_{\text{new}}}` visitation. For tasks where good performance requires visiting states that the old policy never discovers — which is precisely the exploration challenge in many RL problems — this assumption is violated.

The cart-pole validation is minimal: a 6-parameter linear policy on a problem with 4-dimensional state space is not representative of the neural network policies with hundreds or thousands of parameters operating in high-dimensional state spaces that are TRPO's primary target. The fact that max KL and average KL produce "similar" results on cart-pole tells us essentially nothing about whether the approximation holds on the 18-dimensional walker task, let alone on 33,500-parameter Atari policies with high-dimensional pixel observations.

**What evidence exists in the paper.** Section 8.1 reports the cart-pole result: max KL "learned somewhat slower than our final method, due to the more restrictive form of the constraint, but overall the result suggests that the average KL divergence constraint has a similar effect as the theoretically justified maximum KL divergence." The paper provides no figure, no quantitative comparison of final performance or sample efficiency, and no analysis of how often the max KL constraint was active vs. the average KL constraint. The claim that the two have "a similar effect" is based on this single data point. No experiment measures the actual maximum KL divergence `D_{\text{KL}}^{\max}` achieved by TRPO's updates on the locomotion tasks to verify that it remains bounded even though only the average is constrained.

**Mitigation status.** The paper does not attempt to mitigate this limitation beyond the cart-pole experiment. It does not discuss the theoretical conditions under which constraining the average KL implies a bound on the maximum KL (e.g., if the state space is finite and the visitation distribution has full support, or if the policy's KL divergence is Lipschitz in state). It does not propose or evaluate alternative approximations (e.g., constraining a high percentile of the KL distribution rather than the mean, or using a learned metric that upweights states where policy change is dangerous). The average KL approximation is presented as a pragmatic choice, validated by the downstream success of TRPO on the locomotion tasks — but the experiments do not isolate whether this specific approximation is benign or merely masked by other aspects of the algorithm (the line search, the small `\delta`, the dense reward structure). A practitioner using TRPO in a domain where exploration requires visiting states far from the initial distribution cannot rely on the paper's evidence to know whether the average KL constraint will prevent destructive updates.
