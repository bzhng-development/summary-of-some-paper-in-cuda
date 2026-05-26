# Deterministic Policy Gradient Algorithms

**URL:** [https://proceedings.mlr.press/v32/silver14.pdf](https://proceedings.mlr.press/v32/silver14.pdf)

## 🎯 Pitch

This paper introduces a framework for **deterministic policy gradient** algorithms in reinforcement learning with continuous actions, deriving a model-free gradient that follows the gradient of the action-value function — a form that avoids integrating over the action space entirely, unlike the stochastic policy gradient.

---

## 1. Executive Summary

This paper introduces a framework for **deterministic policy gradient** algorithms in reinforcement learning with continuous actions, deriving a model-free gradient that follows the gradient of the action-value function — a form that avoids integrating over the action space entirely, unlike the stochastic policy gradient. The authors test their approach on a high-dimensional continuous bandit, several standard continuous-action benchmarks (mountain car, pendulum, 2D puddle world), and a 20-action-dimensional octopus arm control task using an off-policy deterministic actor-critic (COPDAC) that learns a deterministic target policy from an exploratory stochastic behaviour policy. They demonstrate that the deterministic policy gradient can be estimated much more efficiently than its stochastic counterpart, with performance advantages that grow with action dimensionality — achieving several orders of magnitude improvement over stochastic actor-critic on a 50-dimensional bandit — establishing that deterministic policy gradients are the limiting case of stochastic policy gradients as policy variance tends to zero, and that they provide a practical advantage only when paired with off-policy exploration to maintain adequate state-space coverage.

## 2. Context and Motivation

### The Core Problem: Extending Policy Gradients to Deterministic Policies

The fundamental question this paper tackles is deceptively simple: **can we derive a valid gradient for deterministic policies in reinforcement learning, and if so, is it useful?** This matters because, prior to this work, it was widely believed that the deterministic policy gradient either did not exist or could only be obtained when using a model of the environment. As the paper notes in Section 1:

> "It was previously believed that the deterministic policy gradient did not exist, or could only be obtained when using a model (Peters, 2010)."

This gap was significant. The dominant approach for continuous-action RL—stochastic policy gradients—requires the policy to maintain some level of randomness. Even as the policy converges toward a good deterministic strategy, the stochastic formulation *requires* non-zero variance for the gradient to be well-defined. This creates a fundamental tension: as the policy improves and becomes more deterministic, the very signal used to improve it becomes harder to estimate reliably.

### Why Deterministic Policies Matter

The practical motivation for deterministic policies extends beyond mathematical elegance. The paper identifies several converging pressures:

**1. The vanishing-variance problem in stochastic policy gradients.** When using a stochastic policy, the policy gradient theorem (Equation 2) gives:

$$\nabla_\theta J(\pi_\theta) = \mathbb{E}_{s \sim \rho^\pi, a \sim \pi_\theta} \left[ \nabla_\theta \log \pi_\theta(a|s) \, Q^\pi(s, a) \right]$$

This appears innocuous, but hides a critical issue: the variance of this estimator scales inversely with the policy's variance. For a Gaussian policy $\mathcal{N}(\mu, \sigma^2)$, the variance of the stochastic policy gradient is proportional to $1/\sigma^2$ (Zhao et al., 2012, as cited in Section 6). As the policy becomes more deterministic—which is precisely what we *want* it to do as it converges to an optimal solution—the gradient estimate becomes noisier and noisier, eventually diverging to infinity. The paper frames this vividly in Section 6:

> "Using a stochastic policy gradient algorithm, the policy becomes more deterministic as the algorithm homes in on a good strategy. Unfortunately this makes the stochastic policy gradient harder to estimate, because the policy gradient $\nabla_\theta \pi_\theta(a|s)$ changes more rapidly near the mean."

This is not a minor implementation detail; it is a **structural failure mode** of stochastic policy gradients in continuous action spaces. The very improvement of the policy undermines the algorithm's ability to continue improving it.

**2. The sample complexity of integrating over action spaces.** The stochastic policy gradient in Equation 2 requires integrating over *both* state and action spaces: $\int_S \rho^\pi(s) \int_A \nabla_\theta \pi_\theta(a|s) Q^\pi(s, a) \, da \, ds$. In high-dimensional action spaces—common in robotics, where each joint may contribute multiple continuous dimensions—this inner integral becomes increasingly expensive to estimate by sampling. Even if variance were not an issue, the sheer number of samples needed to accurately approximate the expectation over a high-dimensional action space grows substantially. The deterministic policy gradient, by contrast, eliminates the integral over actions entirely, integrating only over the state distribution (Equation 9):

$$\nabla_\theta J(\mu_\theta) = \mathbb{E}_{s \sim \rho^\mu} \left[ \nabla_\theta \mu_\theta(s) \, \nabla_a Q^\mu(s, a) \big|_{a = \mu_\theta(s)} \right]$$

This is a **qualitative reduction in complexity**: from a double integral (state × action) to a single integral (state only). For high-dimensional control problems, this efficiency difference can be the deciding factor in whether a method is practical at all.

**3. Applications where noise injection is infeasible.** The paper explicitly calls out robotics as a domain where:

> "there are many applications (for example in robotics) where a differentiable control policy is provided, but where there is no functionality to inject noise into the controller. In these cases, the stochastic policy gradient is inapplicable, whereas our methods may still be useful." (Section 1)

This is a constraint that arises from real engineering limitations. Many control systems expose a parameterized deterministic controller—a mapping from sensor readings to actuator commands—without infrastructure for injecting exploration noise. Stochastic policy gradients fundamentally require the ability to sample from a distribution over actions; if the hardware or software interface doesn't support this, the algorithm cannot be applied. Deterministic policy gradients, combined with off-policy exploration (where the exploration mechanism is separate from the target policy), provide a principled way to learn in such settings.

### Where Prior Approaches Fall Short

The paper identifies specific limitations in prior work along multiple dimensions:

**Theoretical gap: no deterministic policy gradient existed.** The canonical policy gradient theorem (Sutton et al., 1999) is formulated for stochastic policies $\pi_\theta(a|s)$ and explicitly involves the score function $\nabla_\theta \log \pi_\theta(a|s)$. For a deterministic policy $\mu_\theta(s)$, the score function is undefined—there is no probability density to differentiate because the policy deterministically maps each state to a single action. This left the field with the impression (formalized in Peters, 2010) that deterministic policy gradients were not available in model-free settings. The paper directly challenges this assumption by proving Theorem 1—the deterministic policy gradient theorem—from first principles, showing that the gradient of the performance objective with respect to policy parameters exists and has a surprisingly simple form.

**Stochastic policy gradients: the variance problem is structural, not incidental.** Prior work had developed a rich ecosystem of stochastic policy gradient methods—REINFORCE (Williams, 1992), natural actor-critic (Peters et al., 2005), off-policy actor-critic (Degris et al., 2012b)—all built on Equation 2. These methods work well for moderate action dimensions and when the policy maintains sufficient entropy. However, as noted above, the variance scaling with $1/\sigma^2$ means that performance degrades in precisely the regime where the policy is approaching optimality. This is not a problem that can be solved by better learning rates or more samples; it is baked into the estimator. The paper's continuous bandit experiments (Section 5.1, Figure 1) make this starkly visible: the stochastic actor-critic lags behind the deterministic version, and the gap widens dramatically with action dimensionality—from a noticeable difference at 10 dimensions to **several orders of magnitude** at 50 dimensions.

**Greedy policy improvement: computationally prohibitive in continuous spaces.** The standard approach to policy improvement in value-based RL is greedy maximization: $\mu_{k+1}(s) = \arg\max_a Q^{\mu_k}(s, a)$. In discrete action spaces, this is a straightforward enumeration. In continuous action spaces, it requires solving a potentially non-convex global optimization problem at every step—computationally impossible for all but the simplest action-value functions. The paper frames the deterministic policy gradient as an alternative to this global maximization (Section 3.1):

> "In continuous action spaces, greedy policy improvement becomes problematic, requiring a global maximisation at every step. Instead, a simple and computationally attractive alternative is to move the policy in the direction of the gradient of Q, rather than globally maximising Q."

This is a crucial insight: rather than finding the global maximum of $Q$ at each state, simply move the policy *toward* higher $Q$-values by following the action gradient $\nabla_a Q^\mu(s, a)$. This converts a global optimization problem into a local gradient computation—tractable, incremental, and compatible with function approximation.

**NFQCA (Hafner and Riedmiller, 2011): a precursor without theoretical guarantees.** The paper acknowledges that it was "not the first to notice that the action-value gradient provides a useful signal for reinforcement learning" (Section 6). The NFQCA algorithm uses neural networks for both actor and critic, with the actor updating in the direction of the action-value gradient. However, the paper identifies a critical weakness:

> "its critic network is incompatible with the actor network; it is unclear how the local optima learnt by the critic (assuming it converges) will interact with actor updates."

In other words, NFQCA uses a heuristic update without theoretical justification. It does not address whether following the action-value gradient is actually following the gradient of the true performance objective—and if it isn't, there is no guarantee of improvement, let alone convergence. The paper's Theorem 1 closes this gap by proving that the intuitive update in Equation 7 is, in fact, the exact gradient of $J(\mu_\theta)$. Theorem 3 then establishes conditions under which substituting a learned critic $Q^w$ for the true $Q^\mu$ preserves this gradient—a result analogous to the compatible function approximation theory for stochastic policies (Sutton et al., 1999).

**Deterministic policy + on-policy exploration: a recipe for failure.** Even if a deterministic policy gradient exists theoretically, using it on-policy—where the agent acts according to the current deterministic policy—presents an obvious problem: **exploration collapses**. A deterministic policy always selects the same action in the same state, so it cannot discover alternative strategies or recover from poor initialization. The paper recognizes this explicitly in Section 4.1:

> "In general, behaving according to a deterministic policy will not ensure adequate exploration and may lead to sub-optimal solutions."

This is what motivates the paper's central algorithmic contribution: **off-policy deterministic actor-critic**. By separating the behaviour policy (which explores) from the target policy (which is deterministic and exploits), the algorithm reaps the efficiency benefits of the deterministic policy gradient while maintaining exploration through a stochastic behaviour policy. The off-policy formulation in Section 4.2 derives a modified performance objective $J_\beta(\mu_\theta)$ that averages the target policy's value over the *behaviour* policy's state distribution, and shows that an analogous gradient exists without requiring importance sampling in the actor—a non-trivial advantage over stochastic off-policy methods, which must correct for the mismatch between behaviour and target distributions using importance weights.

### How This Paper Positions Itself

The paper positions itself at the intersection of two established research threads and makes three distinct contributions that bridge them:

**Thread 1: Policy gradient methods.** These provide principled, model-free gradient-based optimization of parameterized policies. The policy gradient theorem (Sutton et al., 1999) is the cornerstone, but it has only been applied to stochastic policies. The natural actor-critic (Peters et al., 2005), off-policy actor-critic (Degris et al., 2012b), and compatible function approximation (Sutton et al., 1999) all operate within this stochastic framework.

**Thread 2: Deterministic value-based methods.** Q-learning (Watkins and Dayan, 1992) and its continuous-action extensions learn deterministic greedy policies implicitly through value function maximization. However, they struggle with the continuous-action maximization step and typically require discretization, constrained policy classes, or expensive global optimization.

The paper bridges these threads by showing that **deterministic policies can be optimized by gradient ascent on the performance objective**, combining the principled gradient-based optimization of policy gradient methods with the deterministic target policies that emerge naturally from value-based methods. This is formalized through three theoretical contributions:

1. **Theorem 1 (Deterministic Policy Gradient Theorem):** Establishes that $\nabla_\theta J(\mu_\theta)$ exists and equals $\mathbb{E}_{s \sim \rho^\mu}[\nabla_\theta \mu_\theta(s) \nabla_a Q^\mu(s, a)|_{a=\mu_\theta(s)}]$, with the proof provided in Appendix B.

2. **Theorem 2 (Limit of Stochastic Policy Gradient):** Shows that for a broad class of stochastic policies parameterized by a deterministic mean $\mu_\theta$ and a variance parameter $\sigma$, the stochastic policy gradient converges to the deterministic policy gradient as $\sigma \to 0$. This means the deterministic policy gradient is not a separate theory but a **limiting case** of the established stochastic framework—and therefore inherits its machinery (compatible function approximation, natural gradients, actor-critic architectures).

3. **Theorem 3 (Compatible Function Approximation for Deterministic Policies):** Extends the compatible function approximation theory of Sutton et al. (1999) to the deterministic setting, identifying a class of critics $Q^w(s, a) = (a - \mu_\theta(s))^\top \nabla_\theta \mu_\theta(s)^\top w + V^v(s)$ that do not bias the policy gradient when substituted for the true action-value function.

The paper's practical positioning is equally careful. It does not claim that deterministic policies should *replace* stochastic policies universally. Instead, it argues that the deterministic policy gradient is **one tool in the toolbox**—one that excels in high-dimensional action spaces and when off-policy learning is feasible. The on-policy deterministic actor-critic (Section 4.1) is presented as "didactic" rather than practical, acknowledging that exploration requires stochasticity. The off-policy algorithms (OPDAC, COPDAC-Q, COPDAC-GQ) are the practical contributions, and they deliberately retain a stochastic behaviour policy for exploration.

The paper also explicitly connects to the Q-learning literature by framing deterministic actor-critic as the policy-gradient analogue of Q-learning (Section 6):

> "One may view our deterministic actor-critic as analogous, in a policy gradient context, to Q-learning. Q-learning learns a deterministic greedy policy, off-policy, while executing a noisy version of the greedy policy. Similarly, in our experiments COPDAC-Q was used to learn a deterministic policy, off-policy, while executing a noisy version of that policy."

This analogy is instructive: just as Q-learning separates the target policy (greedy with respect to current $Q$-values) from the behaviour policy ($\epsilon$-greedy or similar), deterministic actor-critic separates the target policy (deterministic, following the action-value gradient) from the behaviour policy (stochastic, for exploration). The key difference is that the deterministic actor-critic updates the policy by gradient ascent rather than by global maximization, making it scalable to high-dimensional continuous actions.

In summary, the paper addresses a clearly identified gap—the absence of a valid model-free gradient for deterministic policies—by proving that such a gradient exists, characterizing its relationship to stochastic policy gradients, and developing practical off-policy algorithms that leverage its efficiency advantages. The motivation is both theoretical (completing the policy gradient framework for deterministic policies) and practical (solving high-dimensional continuous control problems that are challenging for stochastic methods, both due to variance scaling and sample complexity).

## 3. Technical Approach

### 3.1 Reader Orientation

The paper develops a family of **actor-critic reinforcement learning algorithms** that optimize a deterministic policy — a direct mapping from states to actions — using a newly derived model-free gradient that follows the gradient of the action-value function. The problem is that stochastic policy gradients require integrating (sampling) over the action space, which becomes increasingly expensive in high dimensions, and their variance diverges as the policy becomes more deterministic; the deterministic policy gradient solves this by eliminating the action integral entirely, yielding an estimator that can be orders of magnitude more efficient, provided exploration is handled through a separate off-policy mechanism.

### 3.2 Big-Picture Architecture

The system consists of four major components connected in a learning loop:

1. **Environment (MDP):** produces states and rewards in response to actions; the agent has no access to its internal dynamics.
2. **Behaviour policy `$\beta(a|s)$`:** a stochastic policy (e.g., fixed-width Gaussian around the current deterministic policy) that actually selects actions during training, ensuring exploration coverage of the state-action space.
3. **Critic `$Q^w(s, a)$`:** a learned approximation to the true action-value function `$Q^\mu(s, a)$`; trained off-policy from behaviour-generated experience to evaluate the deterministic target policy. For compatible function approximation, it takes the specific structural form `$Q^w(s, a) = (a - \mu_\theta(s))^\top \nabla_\theta \mu_\theta(s)^\top w + V^v(s)$`.
4. **Actor (deterministic target policy) `$\mu_\theta(s)$`:** the policy being optimized; updated by gradient ascent on the performance objective using the deterministic policy gradient `$\mathbb{E}_{s \sim \rho^\beta}[\nabla_\theta \mu_\theta(s) \, \nabla_a Q^w(s, a)|_{a=\mu_\theta(s)}]$`.

Information flows as follows: the behaviour policy selects actions → the environment returns states and rewards → the critic updates its action-value estimates using temporal-difference learning (Q-learning or gradient Q-learning) → the actor updates its parameters by ascending the critic's action-value gradient evaluated at the deterministic action → the behaviour policy's mean tracks the updated actor, maintaining exploration through injected noise. All updates are off-policy: the critic evaluates the target policy from data generated by the behaviour policy, and the actor's gradient does not require importance sampling because the deterministic policy gradient integrates only over states, not actions.

### 3.3 Roadmap for the Deep Dive

- **First**, the deterministic policy gradient theorem (Theorem 1) — the mathematical foundation that proves following the action-value gradient is following the true performance gradient, including the formal definition of the performance objective for deterministic policies and the proof strategy.
- **Second**, the relationship to stochastic policy gradients (Theorem 2) — showing that deterministic policy gradients are the limiting case of stochastic policy gradients as variance tends to zero, which justifies using the entire stochastic policy gradient machinery (compatible function approximation, natural gradients, actor-critic architectures) for deterministic policies.
- **Third**, the off-policy formulation — how the performance objective is modified to average the target policy's value over the behaviour policy's state distribution, and why the deterministic case avoids the importance sampling needed in stochastic off-policy actor-critic.
- **Fourth**, the on-policy deterministic actor-critic (Section 4.1) — the simplest algorithm using Sarsa, presented as a didactic baseline to illustrate the core update mechanism.
- **Fifth**, the off-policy deterministic actor-critic (OPDAC, Section 4.2) — the practical algorithm using Q-learning, which separates exploration (stochastic behaviour policy) from exploitation (deterministic target policy) and eliminates importance sampling from the actor.
- **Sixth**, compatible function approximation (Theorem 3, Section 4.3) — the conditions under which substituting a learned critic for the true action-value function does not bias the policy gradient, and the specific structural form that satisfies those conditions.
- **Seventh**, the gradient temporal-difference variants (COPDAC-GQ) — how to achieve convergent off-policy learning under linear function approximation using gradient Q-learning in the critic.

### 3.4 Detailed, Sentence-Based Technical Breakdown

This is a **theoretical algorithm paper** whose core idea is that the gradient of the performance objective with respect to the parameters of a deterministic policy exists, equals the expected gradient of the action-value function chain-ruled through the policy, and can be estimated more efficiently than the stochastic policy gradient because it integrates only over the state space — but requires off-policy exploration to maintain adequate state-space coverage.

---

#### The Deterministic Policy Gradient Theorem (Theorem 1)

The paper begins by defining the performance objective for a deterministic policy `$\mu_\theta: \mathcal{S} \to \mathcal{A}$` with parameters `$\theta \in \mathbb{R}^n$`. Let `$r_t^\gamma = \sum_{k=t}^\infty \gamma^{k-t} r(s_k, a_k)$` be the discounted return from timestep `$t$`, with `$0 < \gamma < 1$`. The performance objective is:

$$J(\mu_\theta) = \mathbb{E}\left[r_1^\gamma \mid \mu\right]$$

where the expectation is over the stochasticity in the environment's state transitions `$p(s_{t+1} \mid s_t, a_t)$` and initial state distribution `$p_1(s_1)$`, given that actions are selected deterministically by `$a_t = \mu_\theta(s_t)$`.

To express this as an expectation over states, the paper defines the **discounted state distribution** `$\rho^\mu(s')$` induced by following `$\mu_\theta$`:

$$\rho^\mu(s') = \int_{\mathcal{S}} \sum_{t=1}^\infty \gamma^{t-1} p_1(s) \, p(s \to s', t, \mu) \, ds$$

where `$p(s \to s', t, \mu)$` is the density of reaching state `$s'$` after `$t$` time steps starting from state `$s$` under policy `$\mu$`, `$p_1(s)$` is the initial state density, and `$\gamma^{t-1}$` discounts the contribution of future states. This is an **improper** distribution (it integrates to `$1/(1-\gamma)$` rather than 1) because of the discounting, but behaves like a probability distribution for expectation purposes.

**What it computes:** `$\rho^\mu(s')$` is the effective "visitation frequency" of state `$s'$`, weighting states by how often they are encountered and how soon they occur (earlier visits receive higher weight via `$\gamma^{t-1}$`). It summarizes the long-run behaviour of the MDP under policy `$\mu$` into a single distribution over states.

**Why this form:** using a discounted state distribution aligns the performance objective `$J(\mu_\theta)$` with the definition of the value function `$V^\mu(s)$`, which also discounts future rewards. This makes the algebra of the proof work out cleanly — the gradient of `$J$` decomposes into terms involving `$\nabla_\theta Q^\mu$`, and the state distribution terms cancel in a way that parallels the stochastic policy gradient theorem proof of Sutton et al. (1999).

With `$\rho^\mu$` defined, the performance objective can be rewritten as a state-space expectation:

$$J(\mu_\theta) = \int_{\mathcal{S}} \rho^\mu(s) \, r(s, \mu_\theta(s)) \, ds = \mathbb{E}_{s \sim \rho^\mu}\left[r(s, \mu_\theta(s))\right]$$

where `$r(s, \mu_\theta(s))$` is the immediate reward received in state `$s$` when the deterministic action `$\mu_\theta(s)$` is taken. The integral is over the state space `$\mathcal{S}$` (assumed to be a compact subset of `$\mathbb{R}^d$`), and the action space is `$\mathcal{A} = \mathbb{R}^m$`.

**Theorem 1** then states that, under technical regularity conditions (MDP satisfies conditions A.1 in Appendix B, which ensure `$\nabla_\theta \mu_\theta(s)$` and `$\nabla_a Q^\mu(s, a)$` exist and the deterministic policy gradient exists), the gradient of `$J(\mu_\theta)$` is:

$$\nabla_\theta J(\mu_\theta) = \int_{\mathcal{S}} \rho^\mu(s) \, \nabla_\theta \mu_\theta(s) \, \nabla_a Q^\mu(s, a)\big|_{a=\mu_\theta(s)} \, ds$$

$$= \mathbb{E}_{s \sim \rho^\mu}\left[\nabla_\theta \mu_\theta(s) \, \nabla_a Q^\mu(s, a)\big|_{a=\mu_\theta(s)}\right]$$

where `$\nabla_\theta \mu_\theta(s)$` is an `$n \times m$` Jacobian matrix — each column `$d$` is the gradient vector `$\nabla_\theta [\mu_\theta(s)]_d$` of the `$d$`-th action dimension of the deterministic policy with respect to the policy parameters `$\theta$` — and `$\nabla_a Q^\mu(s, a)|_{a=\mu_\theta(s)}$` is the `$m$`-dimensional gradient of the true action-value function with respect to the action, evaluated at the deterministic action `$a = \mu_\theta(s)$`.

**What it computes:** the product `$\nabla_\theta \mu_\theta(s) \, \nabla_a Q^\mu(s, a)$` is an `$n$`-dimensional vector (the chain rule applied through the action). For each state `$s$` visited under the current policy, compute the direction in parameter space that would move the policy's selected action toward higher `$Q$`-values, then average these directions over all states, weighted by how frequently and how early each state is visited (`$\rho^\mu(s)$`). The result is the exact gradient of the total expected discounted return.

**Why this form:** this is the fundamental insight of the paper. The stochastic policy gradient (Equation 2) integrates over both states and actions: `$\int_\mathcal{S} \rho^\pi(s) \int_\mathcal{A} \nabla_\theta \pi_\theta(a|s) Q^\pi(s, a) \, da \, ds$`. The action integral `$\int_\mathcal{A}$` must be estimated by sampling actions from the stochastic policy, which becomes increasingly noisy as the policy's variance shrinks. The deterministic policy gradient **eliminates the action integral entirely** — it is a single expectation over states only. This means:

1. No Monte Carlo sampling of actions is needed for the gradient estimate; the action is computed deterministically as `$\mu_\theta(s)$`.
2. The variance scaling problem disappears — there is no `$1/\sigma^2$` term because there is no `$\sigma$`.
3. In high-dimensional action spaces, the sample efficiency gain is proportional to the action dimensionality `$m$`, since the stochastic estimator must sample an `$m$`-dimensional space while the deterministic estimator does not.

**The intuitive derivation (Section 3.1) explains why this form is natural.** In generalized policy iteration, policy improvement typically requires greedy maximization: `$\mu_{k+1}(s) = \arg\max_a Q^{\mu_k}(s, a)$`. In continuous spaces, this global optimization is intractable. Instead, the paper proposes moving the policy **in the direction of the gradient of `$Q$`**:

$$\theta_{k+1} = \theta_k + \alpha \, \mathbb{E}_{s \sim \rho^{\mu_k}}\left[\nabla_\theta \mu_\theta(s) \, \nabla_a Q^{\mu_k}(s, a)\big|_{a=\mu_\theta(s)}\right]$$

This is the intuitive update shown in Equation 7. It decomposes the policy improvement into two multiplicative factors: `$\nabla_a Q^{\mu_k}(s, a)$` (how should the action change to increase value?) and `$\nabla_\theta \mu_\theta(s)$` (how should the parameters change to move the action in that direction?). Theorem 1 proves that this intuitive update — which one might have guessed by analogy to stochastic gradient ascent — is exactly following the true gradient of `$J(\mu_\theta)$`, without needing to account for the change in state distribution `$\rho^\mu$` that results from changing the policy. The state distribution gradient terms cancel out in the proof (Appendix B), exactly as they do in the stochastic policy gradient theorem.

**Proof sketch (Appendix B).** The proof follows the same lines as Sutton et al. (1999)'s proof for the stochastic case. The key step is expanding `$\nabla_\theta Q^\mu(s, a)$` where `$a = \mu_\theta(s)$`:

$$\nabla_\theta Q^\mu(s, \mu_\theta(s)) = \nabla_a Q^\mu(s, a)\big|_{a=\mu_\theta(s)} \nabla_\theta \mu_\theta(s) + \nabla_\theta Q^\mu(s, a)\big|_{a=\mu_\theta(s)}$$

The second term `$\nabla_\theta Q^\mu(s, a)|_{a=\mu_\theta(s)}$` captures how the value of a fixed action changes because the policy changes future actions after state `$s$`. This term depends on `$\nabla_\theta \rho^\mu$` — the change in state visitation distribution — and the proof shows that when integrated over `$\rho^\mu(s)$` and summed appropriately, it collapses into a telescoping series that yields exactly the first term. The result is that only the direct effect of `$\theta$` on the immediate action matters for the gradient of `$J$`, not the indirect effect through changed future state distributions.

---

#### The Deterministic Policy Gradient as a Limit of Stochastic Policy Gradients (Theorem 2)

The paper does not treat the deterministic policy gradient as a separate theoretical construct; it shows that it is the **limiting case** of the familiar stochastic policy gradient framework. This is critical because it means all the existing machinery developed for stochastic policy gradients — compatible function approximation, natural gradients, actor-critic architectures, episodic and batch methods — transfers directly to deterministic policies.

**Theorem 2** considers a family of stochastic policies `$\pi_{\mu_\theta, \sigma}$` parameterized by a deterministic mean function `$\mu_\theta: \mathcal{S} \to \mathcal{A}$` and a variance parameter `$\sigma$`, such that `$\pi_{\mu_\theta, \sigma}(a|s) = \nu_\sigma(\mu_\theta(s), a)$`. Here `$\nu_\sigma$` is a noise distribution centered at `$\mu_\theta(s)$` with variance controlled by `$\sigma$`. When `$\sigma = 0$`, the stochastic policy becomes exactly the deterministic policy: `$\pi_{\mu_\theta, 0} \equiv \mu_\theta$`. The theorem states:

$$\lim_{\sigma \downarrow 0} \nabla_\theta J(\pi_{\mu_\theta, \sigma}) = \nabla_\theta J(\mu_\theta)$$

where the left-hand side uses the stochastic policy gradient theorem (Equation 2) and the right-hand side uses the deterministic policy gradient theorem (Theorem 1). The proof is in Appendix C and requires technical conditions (B.1 and A.1-A.2) that essentially require `$\nu_\sigma$` to be a "bump" distribution that concentrates its probability mass around `$\mu_\theta(s)$` as `$\sigma \to 0$` (e.g., a Gaussian with shrinking variance, or any distribution satisfying certain smoothness and concentration properties).

**What it computes:** this is a limit relationship, not a computational recipe. It says: if you take a stochastic policy, compute its policy gradient via Equation 2, and then let the policy's variance shrink to zero, the resulting gradient vector converges to the deterministic policy gradient of the mean policy. Equivalently, the deterministic policy gradient is what you would get if you could directly differentiate the limiting deterministic policy.

**Why this form matters:**

1. **Theoretical unification.** The deterministic policy gradient is not an ad-hoc derivation; it emerges naturally from the established stochastic framework in the zero-variance limit. This explains why the proof structure mirrors the stochastic policy gradient theorem: they are fundamentally the same result at different points on the stochasticity spectrum.

2. **Justification for using stochastic policy gradient tools.** Compatible function approximation (Sutton et al., 1999), natural policy gradients (Kakade, 2001), and actor-critic methods (Bhatnagar et al., 2007) were all developed for stochastic policies. Theorem 2 says these methods remain valid for deterministic policies because the deterministic gradient is the limit of the stochastic gradient. Specifically:
   - **Compatible function approximation:** condition i) in the stochastic case requires `$Q^w(s, a) = \nabla_\theta \log \pi_\theta(a|s)^\top w$`. In the limit `$\sigma \to 0$`, this becomes `$\nabla_a Q^w(s, a)|_{a=\mu_\theta(s)} = \nabla_\theta \mu_\theta(s)^\top w$`, which is exactly condition 1 of Theorem 3. The paper derives this directly in Section 4.3.
   - **Natural gradients:** the Fisher information metric for stochastic policies `$\mathbb{E}_{s \sim \rho^\pi, a \sim \pi_\theta}[\nabla_\theta \log \pi_\theta(a|s) \nabla_\theta \log \pi_\theta(a|s)^\top]$` converges to `$\mathbb{E}_{s \sim \rho^\mu}[\nabla_\theta \mu_\theta(s) \nabla_\theta \mu_\theta(s)^\top]$` as `$\sigma \to 0$`, which the paper uses as the natural gradient metric for deterministic policies (end of Section 4.3).

3. **Practical insight: the failure mode of stochastic gradients.** As `$\sigma \to 0$`, the stochastic policy gradient `$\mathbb{E}[\nabla_\theta \log \pi_\theta(a|s) Q^\pi(s, a)]$` has variance that diverges to infinity (proportional to `$1/\sigma^2$` for Gaussian policies, as noted in Section 6). The deterministic policy gradient, obtained by taking the limit directly rather than estimating the stochastic quantity, avoids this variance explosion entirely. This explains *why* the deterministic gradient can be estimated more efficiently — it's not just a different estimator; it's a fundamentally different quantity that bypasses the action-sampling step entirely.

---

#### Off-Policy Performance Objective and Gradient

Learning a deterministic policy on-policy — where the agent acts according to `$\mu_\theta$` and updates based on its own experience — has an obvious problem: **exploration.** A deterministic policy always selects the same action in a given state, so it cannot discover alternative strategies or recover from poor initialization. The on-policy deterministic actor-critic (Section 4.1) is presented as "didactic" and may work only when the environment provides sufficient noise (e.g., stochastic transitions), but for general RL problems, off-policy learning is necessary.

The paper formulates the off-policy objective following Degris et al. (2012b). Let `$\beta(a|s)$` be a stochastic **behaviour policy** that generates trajectories for training, and let `$\mu_\theta(s)$` be the deterministic **target policy** being optimized. The performance objective is modified to average the target policy's value over the *behaviour policy's* state distribution `$\rho^\beta(s)$` rather than the target policy's state distribution `$\rho^\mu(s)$`:

$$J_\beta(\mu_\theta) = \int_{\mathcal{S}} \rho^\beta(s) \, V^\mu(s) \, ds = \int_{\mathcal{S}} \rho^\beta(s) \, Q^\mu(s, \mu_\theta(s)) \, ds$$

where `$V^\mu(s)$` is the value of state `$s$` under the target policy `$\mu_\theta$` (i.e., the expected return starting from `$s$` and following `$\mu_\theta$` thereafter), and `$Q^\mu(s, \mu_\theta(s))$` is the action-value of taking the deterministic action `$\mu_\theta(s)$` and then following `$\mu_\theta$` thereafter.

**What it computes:** `$J_\beta(\mu_\theta)$` is the expected return of the target policy, but the expectation over initial states and their visitation frequencies is taken under the behaviour policy's state distribution. If `$\beta$` explores broadly, `$\rho^\beta(s)$` covers the state space well, and optimizing `$J_\beta$` will improve `$\mu_\theta$` across all states that the behaviour policy visits, even if the target policy itself would not visit those states.

**Why this form:** this is a common off-policy objective trick (Degris et al., 2012b). The alternative — optimizing `$J(\mu_\theta)$` directly with data from `$\beta$` — would require importance sampling to correct for the distribution mismatch, introducing variance. By redefining the objective to average over `$\rho^\beta$`, the gradient can be estimated without importance weighting the states, because the expectation is now over the distribution that actually generated the data. The trade-off is that `$J_\beta(\mu_\theta)$` is not the same as the true on-policy objective `$J(\mu_\theta)$`. However, if `$\beta$` has reasonable coverage, improving `$J_\beta$` typically improves `$J$` as well; and if `$\beta = \mu_\theta$` (the on-policy case), the two objectives coincide.

**The off-policy deterministic policy gradient** is then derived by differentiating `$J_\beta(\mu_\theta)$` and dropping a term that depends on `$\nabla_\theta Q^{\mu_\theta}(s, a)$`. The paper gives the approximation:

$$\nabla_\theta J_\beta(\mu_\theta) \approx \int_{\mathcal{S}} \rho^\beta(s) \, \nabla_\theta \mu_\theta(s) \, \nabla_a Q^\mu(s, a)\big|_{a=\mu_\theta(s)} \, ds$$

$$= \mathbb{E}_{s \sim \rho^\beta}\left[\nabla_\theta \mu_\theta(s) \, \nabla_a Q^\mu(s, a)\big|_{a=\mu_\theta(s)}\right]$$

**What is dropped and why.** The full gradient `$\nabla_\theta J_\beta(\mu_\theta)$` contains a term `$\int_{\mathcal{S}} \rho^\beta(s) \nabla_\theta Q^\mu(s, a)|_{a=\mu_\theta(s)} ds$`. This term captures how changing `$\theta$` changes the value of the action `$\mu_\theta(s)$` through its effect on *future* policy parameters — it is the same term that cancels out in the on-policy proof via the telescoping sum argument. In the off-policy case, the cancellation does not occur cleanly because `$\rho^\beta \neq \rho^\mu$`. Degris et al. (2012b) argue that dropping this term is a "good approximation since it can preserve the set of local optima to which gradient ascent converges." The paper adopts this same approximation for the deterministic case.

**Crucial advantage over stochastic off-policy actor-critic.** The stochastic off-policy policy gradient (Equation 5) requires an importance sampling ratio `$\frac{\pi_\theta(a|s)}{\beta(a|s)}$` to correct for the mismatch between the target policy's action distribution and the behaviour policy's action distribution:

$$\nabla_\theta J_\beta(\pi_\theta) \approx \mathbb{E}_{s \sim \rho^\beta, a \sim \beta}\left[\frac{\pi_\theta(a|s)}{\beta(a|s)} \nabla_\theta \log \pi_\theta(a|s) \, Q^\pi(s, a)\right]$$

The importance weight `$\pi_\theta(a|s) / \beta(a|s)$` can have high variance, especially when `$\beta$` is very different from `$\pi_\theta$`. The deterministic off-policy gradient **does not need importance sampling in the actor** because it integrates only over states, not actions. The action `$a = \mu_\theta(s)$` is computed deterministically; there is no action distribution to correct. The expectation `$\mathbb{E}_{s \sim \rho^\beta}[\cdot]$` can be estimated directly from states visited by the behaviour policy without any weighting.

The critic still learns off-policy — it must estimate `$Q^\mu(s, a)$` from trajectories generated by `$\beta$` — but by using Q-learning (which is inherently off-policy: it bootstraps from `$\max_{a'} Q(s', a')$` or `$Q(s', \mu_\theta(s'))$` regardless of which action was actually taken), the critic also avoids importance sampling. This is a significant practical simplification: the entire off-policy deterministic actor-critic operates without importance weights, unlike its stochastic counterpart (OffPAC, Degris et al., 2012b), which requires importance sampling for both actor and critic.

---

#### On-Policy Deterministic Actor-Critic (Section 4.1)

The on-policy algorithm is the simplest instantiation of the deterministic policy gradient and serves as a didactic baseline. It uses a Sarsa critic and updates the actor and critic from the same on-policy trajectories (the agent acts according to `$\mu_\theta$` directly).

**Critic update (Sarsa).** The critic maintains a differentiable action-value function `$Q^w(s, a)$` with parameters `$w$`. At each timestep, given a transition `$(s_t, a_t, r_t, s_{t+1}, a_{t+1})$` where `$a_t = \mu_\theta(s_t)$` and `$a_{t+1} = \mu_\theta(s_{t+1})$` (both selected deterministically by the current policy), the temporal-difference error is:

$$\delta_t = r_t + \gamma Q^w(s_{t+1}, a_{t+1}) - Q^w(s_t, a_t)$$

where `$r_t$` is the immediate reward, `$\gamma \in (0, 1)$` is the discount factor, `$Q^w(s_{t+1}, a_{t+1})$` is the critic's estimate of future value from the next state-action pair, and `$Q^w(s_t, a_t)$` is the current estimate for the current state-action pair.

**What it computes:** `$\delta_t$` measures the surprise — the difference between the predicted return `$Q^w(s_t, a_t)$` and a better estimate `$r_t + \gamma Q^w(s_{t+1}, a_{t+1})$` that incorporates the actual reward and the bootstrapped future value. When `$\delta_t > 0$`, the current estimate is too low; when `$\delta_t < 0$`, it is too high.

The critic parameters are updated by gradient descent on the squared TD error:

$$w_{t+1} = w_t + \alpha_w \delta_t \nabla_w Q^w(s_t, a_t)$$

where `$\alpha_w > 0$` is the critic learning rate. This is the standard semi-gradient Sarsa update (Sutton and Barto, 1998).

**Actor update (deterministic policy gradient).** The actor updates its parameters `$\theta$` by stochastic gradient ascent of the performance objective, using the critic in place of the unknown true action-value function:

$$\theta_{t+1} = \theta_t + \alpha_\theta \nabla_\theta \mu_\theta(s_t) \, \nabla_a Q^w(s_t, a)\big|_{a=\mu_\theta(s_t)}$$

where `$\alpha_\theta > 0$` is the actor learning rate, `$\nabla_\theta \mu_\theta(s_t)$` is the Jacobian of the deterministic policy output with respect to its parameters (dimension `$n \times m$`), and `$\nabla_a Q^w(s_t, a)|_{a=\mu_\theta(s_t)}$` is the gradient of the critic's estimated action-value with respect to the action, evaluated at the deterministic action (dimension `$m$`).

**What it computes:** the product produces an `$n$`-dimensional parameter update vector. For each parameter `$\theta_i$`, the update is `$\alpha_\theta \cdot \sum_{j=1}^m \frac{\partial \mu_\theta^{(j)}(s_t)}{\partial \theta_i} \cdot \frac{\partial Q^w(s_t, a)}{\partial a_j}\big|_{a=\mu_\theta(s_t)}$`: the sum over action dimensions of (how much parameter `$i$` affects action dimension `$j$`) × (how much increasing action dimension `$j$` would increase the estimated value). The parameters move in the direction that shifts the deterministic action toward higher `$Q^w$`-values.

**Why this works (and why it might not).** The update is a semi-gradient method: it treats `$Q^w$` as if it were the true `$Q^\mu$`, ignoring the fact that `$Q^w$` is a function approximator with its own parameters and errors. If `$Q^w$` is a good approximation to `$Q^\mu$` (in the sense that `$\nabla_a Q^w \approx \nabla_a Q^\mu$`), the update approximately follows the true gradient. However, two issues arise:

1. **Bias from function approximation.** If `$Q^w$` is not a compatible function approximator (see Section 4.3), substituting it into the gradient formula may bias the update direction, potentially preventing convergence or even causing divergence.

2. **Exploration collapse.** The agent acts according to `$\mu_\theta$`, which is deterministic. If `$\mu_\theta$` is initialized poorly or converges to a suboptimal local optimum, there is no mechanism to discover better strategies. The environment's own stochasticity (in transitions `$p(s_{t+1}|s_t, a_t)$` or initial states) may provide some exploration, but this is unreliable and problem-dependent. This is why the paper immediately moves to off-policy methods in Section 4.2.

The paper does not report experimental results for this on-policy algorithm; it exists primarily to isolate and illustrate the core deterministic policy gradient mechanism without the complications of off-policy learning and behaviour policies.

---

#### Off-Policy Deterministic Actor-Critic (OPDAC, Section 4.2)

The OPDAC algorithm is the first practical method proposed in the paper. It separates exploration and exploitation by using a **stochastic behaviour policy** `$\beta(a|s)$` to generate trajectories, while learning a **deterministic target policy** `$\mu_\theta(s)$` that is optimized via the off-policy deterministic policy gradient. The critic uses Q-learning, making both actor and critic importance-sampling-free.

**Critic update (Q-learning).** The critic maintains `$Q^w(s, a)$` and updates off-policy from transitions `$(s_t, a_t, r_t, s_{t+1})$` generated by the behaviour policy `$\beta$`. The key difference from Sarsa is that the TD target uses the deterministic target policy `$\mu_\theta$` to select the next action, not the behaviour policy's action `$a_{t+1}$`:

$$\delta_t = r_t + \gamma Q^w(s_{t+1}, \mu_\theta(s_{t+1})) - Q^w(s_t, a_t)$$

where `$a_t \sim \beta(\cdot|s_t)$` is the action actually taken (from the stochastic behaviour policy), but the bootstrap uses `$\mu_\theta(s_{t+1})$` — the action the target policy *would* take.

**What it computes:** this is the Q-learning analogue for deterministic policies. Standard Q-learning uses `$\max_{a'} Q(s_{t+1}, a')$`; here, the maximization is replaced by evaluating `$Q^w$` at the target policy's action `$\mu_\theta(s_{t+1})$`. This is valid because the target policy is being optimized to select high-value actions — as learning progresses, `$\mu_\theta(s)$` approximates `$\arg\max_a Q^\mu(s, a)$`. The advantage over explicit maximization is computational: evaluating `$Q^w(s_{t+1}, \mu_\theta(s_{t+1}))$` is a single forward pass, while `$\max_a Q^w(s_{t+1}, a)$` would require solving a continuous optimization problem.

The critic update is:

$$w_{t+1} = w_t + \alpha_w \delta_t \nabla_w Q^w(s_t, a_t)$$

This is the standard semi-gradient Q-learning update, applied to the transition `$(s_t, a_t)$` generated by `$\beta$`. No importance sampling is needed because Q-learning is inherently off-policy — it learns about the target policy `$\mu_\theta$` regardless of which policy generated the data.

**Actor update (deterministic policy gradient, off-policy).** The actor update uses the off-policy deterministic policy gradient:

$$\theta_{t+1} = \theta_t + \alpha_\theta \nabla_\theta \mu_\theta(s_t) \, \nabla_a Q^w(s_t, a)\big|_{a=\mu_\theta(s_t)}$$

This is identical in form to the on-policy update (Equation 13), but uses states `$s_t$` sampled from the behaviour policy's state distribution `$\rho^\beta$` rather than the target policy's distribution `$\rho^\mu$`. The lack of importance sampling is a key advantage: the expectation `$\mathbb{E}_{s \sim \rho^\beta}[\cdot]$` is estimated directly from the states visited under `$\beta$`, without any weighting factor. This works because:

1. The off-policy objective `$J_\beta(\mu_\theta)$` is defined as an expectation over `$\rho^\beta$` by construction.
2. The deterministic policy gradient integrates only over states, not actions — there is no `$\pi_\theta(a|s) / \beta(a|s)$` term to include because no action is sampled from the target policy.

**Why this separation matters.** The behaviour policy `$\beta$` can be designed purely for exploration — typically a fixed-width Gaussian centered at `$\mu_\theta(s)$`: `$\beta(\cdot|s) = \mathcal{N}(\mu_\theta(s), \sigma_\beta^2)$`. The variance `$\sigma_\beta^2$` is a hyperparameter that controls exploration breadth and remains constant throughout training, unlike stochastic actor-critic where the policy's variance must adapt (and eventually shrink, causing the gradient variance problem). The target policy `$\mu_\theta$` can become arbitrarily deterministic without affecting exploration or gradient estimation quality.

**Summary of the OPDAC algorithm (Equations 16-18):**

$$\delta_t = r_t + \gamma Q^w(s_{t+1}, \mu_\theta(s_{t+1})) - Q^w(s_t, a_t)$$

$$w_{t+1} = w_t + \alpha_w \delta_t \nabla_w Q^w(s_t, a_t)$$

$$\theta_{t+1} = \theta_t + \alpha_\theta \nabla_\theta \mu_\theta(s_t) \, \nabla_a Q^w(s_t, a)\big|_{a=\mu_\theta(s_t)}$$

The computational cost per timestep is `$O(mn)$` — linear in both the action dimensionality `$m$` and the number of policy parameters `$n$` — since computing `$\nabla_\theta \mu_\theta(s_t)$` (an `$n \times m$` matrix) and `$\nabla_a Q^w(s_t, a)$` (an `$m$`-vector) and multiplying them requires `$O(mn)$` operations.

---

#### Compatible Function Approximation for Deterministic Policies (Theorem 3, Section 4.3)

Substituting an arbitrary function approximator `$Q^w$` for the true action-value function `$Q^\mu$` in the deterministic policy gradient will generally bias the gradient — the update may point in a different direction than the true gradient `$\nabla_\theta J(\mu_\theta)$`, and may not even be an ascent direction (it could decrease performance). The paper extends the compatible function approximation theory of Sutton et al. (1999) to the deterministic setting, identifying conditions under which `$\nabla_a Q^w$` can replace `$\nabla_a Q^\mu$` without bias.

**Theorem 3** states that a function approximator `$Q^w(s, a)$` is compatible with a deterministic policy `$\mu_\theta(s)$` — meaning `$\mathbb{E}[\nabla_\theta \mu_\theta(s) \nabla_a Q^w(s, a)|_{a=\mu_\theta(s)}] = \mathbb{E}[\nabla_\theta \mu_\theta(s) \nabla_a Q^\mu(s, a)|_{a=\mu_\theta(s)}] = \nabla_\theta J(\mu_\theta)$` or `$\nabla_\theta J_\beta(\mu_\theta)$` for the off-policy case — if two conditions hold:

**Condition 1:** `$\nabla_a Q^w(s, a)|_{a=\mu_\theta(s)} = \nabla_\theta \mu_\theta(s)^\top w$`

where `$\nabla_\theta \mu_\theta(s)^\top$` is the transpose of the `$n \times m$` Jacobian matrix, yielding an `$m \times n$` matrix, and `$w \in \mathbb{R}^n$` is a vector of critic parameters (with the same dimensionality as the policy parameters `$\theta$`). The product `$\nabla_\theta \mu_\theta(s)^\top w$` is an `$m$`-dimensional vector — the gradient of `$Q^w$` with respect to actions, evaluated at the deterministic action.

**What it means:** the action gradient of the critic must be a **linear function of the policy Jacobian features**, with parameters `$w$`. This is a structural constraint on the function approximator — it must be representable such that its action derivative at `$a = \mu_\theta(s)$` equals `$\nabla_\theta \mu_\theta(s)^\top w$`.

**Condition 2:** `$w$` minimizes the mean-squared error between the critic's action gradient and the true action gradient:

$$\text{MSE}(\theta, w) = \mathbb{E}\left[\epsilon(s; \theta, w)^\top \epsilon(s; \theta, w)\right]$$

where the error vector `$\epsilon(s; \theta, w)$` is:

$$\epsilon(s; \theta, w) = \nabla_a Q^w(s, a)\big|_{a=\mu_\theta(s)} - \nabla_a Q^\mu(s, a)\big|_{a=\mu_\theta(s)}$$

The expectation is over the relevant state distribution (`$\rho^\mu$` for on-policy, `$\rho^\beta$` for off-policy).

**What it means:** among all parameter vectors `$w$` that define critics satisfying Condition 1, choose the one that makes the action gradient of `$Q^w$` as close as possible (in `$L^2$` sense) to the true action gradient of `$Q^\mu$`, averaged over states.

**Proof that compatibility eliminates bias.** If `$w$` minimizes the MSE, then the gradient of the MSE with respect to `$w$` is zero:

$$\nabla_w \text{MSE}(\theta, w) = 0$$

$$\mathbb{E}\left[\nabla_\theta \mu_\theta(s) \, \epsilon(s; \theta, w)\right] = 0$$

where `$\nabla_w \epsilon(s; \theta, w) = \nabla_\theta \mu_\theta(s)$` by Condition 1 (the error's gradient with respect to `$w$` is the policy Jacobian). Expanding:

$$\mathbb{E}\left[\nabla_\theta \mu_\theta(s) \left(\nabla_a Q^w(s, a)|_{a=\mu_\theta(s)} - \nabla_a Q^\mu(s, a)|_{a=\mu_\theta(s)}\right)\right] = 0$$

$$\mathbb{E}\left[\nabla_\theta \mu_\theta(s) \nabla_a Q^w(s, a)|_{a=\mu_\theta(s)}\right] = \mathbb{E}\left[\nabla_\theta \mu_\theta(s) \nabla_a Q^\mu(s, a)|_{a=\mu_\theta(s)}\right] = \nabla_\theta J(\mu_\theta)$$

The left-hand side uses the critic; the right-hand side uses the true action-value function and equals the true policy gradient by Theorem 1. Therefore, substituting the compatible critic into the deterministic policy gradient formula yields the exact true gradient.

**The compatible function approximator form.** The paper shows that a function approximator of the following form satisfies Condition 1 for any deterministic policy `$\mu_\theta(s)$`:

$$Q^w(s, a) = (a - \mu_\theta(s))^\top \nabla_\theta \mu_\theta(s)^\top w + V^v(s)$$

where:
- `$a \in \mathbb{R}^m$` is the action being evaluated (any action, not just `$\mu_\theta(s)$`)
- `$\mu_\theta(s) \in \mathbb{R}^m$` is the deterministic target action
- `$\nabla_\theta \mu_\theta(s)^\top$` is the `$m \times n$` transpose of the policy Jacobian
- `$w \in \mathbb{R}^n$` are the advantage parameters
- `$(a - \mu_\theta(s))^\top \nabla_\theta \mu_\theta(s)^\top w$` is the **advantage term** — a scalar representing the advantage of taking action `$a$` over the deterministic action `$\mu_\theta(s)$`
- `$V^v(s)$` is a **state-value baseline** with parameters `$v$`, which can be any differentiable function independent of `$a$` (e.g., a linear combination of state features `$\phi(s)$`: `$V^v(s) = v^\top \phi(s)$`)

**Verification of Condition 1.** Taking the gradient with respect to `$a$`:

$$\nabla_a Q^w(s, a) = \nabla_a \left[(a - \mu_\theta(s))^\top \nabla_\theta \mu_\theta(s)^\top w\right] + \nabla_a V^v(s)$$

The second term is zero because `$V^v(s)$` does not depend on `$a$`. The first term: `$\nabla_a \left[(a - \mu_\theta(s))^\top M w\right] = M^\top w$` where `$M = \nabla_\theta \mu_\theta(s)^\top$`. Since `$M^\top = \nabla_\theta \mu_\theta(s)$`:

$$\nabla_a Q^w(s, a) = \nabla_\theta \mu_\theta(s) w$$

Evaluating at `$a = \mu_\theta(s)$`:

$$\nabla_a Q^w(s, a)\big|_{a=\mu_\theta(s)} = \nabla_\theta \mu_\theta(s) w \neq \nabla_\theta \mu_\theta(s)^\top w$$

Wait — there is a subtlety here. Condition 1 requires `$\nabla_a Q^w(s, a)|_{a=\mu_\theta(s)} = \nabla_\theta \mu_\theta(s)^\top w$`, which is an `$m$`-vector (the transpose of `$\nabla_\theta \mu_\theta(s) w$`). The paper states this correctly: the feature vector is `$\phi(s, a) = \nabla_\theta \mu_\theta(s)(a - \mu_\theta(s))$`, which is `$n \times 1$`, so `$Q^w(s, a) = \phi(s, a)^\top w + V^v(s) = (a - \mu_\theta(s))^\top \nabla_\theta \mu_\theta(s)^\top w + V^v(s)$`. The action gradient of this is `$\nabla_\theta \mu_\theta(s)^\top w$` (dimension `$m$`), satisfying Condition 1.

**Interpretation of the compatible form.** The advantage term `$(a - \mu_\theta(s))^\top \nabla_\theta \mu_\theta(s)^\top w$` is a **local linear model** of the advantage function around the deterministic action. For a small deviation `$\delta = a - \mu_\theta(s)$`, the advantage is:

$$A^w(s, \mu_\theta(s) + \delta) = \delta^\top \nabla_\theta \mu_\theta(s)^\top w$$

This is a linear function of the deviation `$\delta$`, with coefficients given by `$\nabla_\theta \mu_\theta(s)^\top w$`. The linear approximation is sufficient because the actor only needs to know the direction of steepest ascent in `$Q$`-value — which is `$\nabla_a Q^w(s, a)|_{a=\mu_\theta(s)} = \nabla_\theta \mu_\theta(s)^\top w$`. The actor's update becomes:

$$\nabla_\theta \mu_\theta(s) \, \nabla_a Q^w(s, a)\big|_{a=\mu_\theta(s)} = \nabla_\theta \mu_\theta(s) \, \nabla_\theta \mu_\theta(s)^\top w$$

**Why this form is practical despite global inaccuracy.** A linear advantage function `$A^w(s, a) = (a - \mu_\theta(s))^\top \nabla_\theta \mu_\theta(s)^\top w$` diverges to `$\pm\infty$` for actions far from `$\mu_\theta(s)$`, making it a poor global model of `$Q^\mu(s, a)$`. The paper acknowledges this explicitly:

> "We note that a linear function approximator is not very useful for predicting action-values globally, since the action-value diverges to `$\pm\infty$` for large actions. However, it can still be highly effective as a local critic. In particular, it represents the local advantage of deviating from the current policy."

The critic only needs to be accurate **near** `$a = \mu_\theta(s)$` — specifically, its action gradient at that point must match the true gradient. The global behaviour of `$Q^w$` away from `$\mu_\theta(s)$` is irrelevant for the actor update, which only evaluates `$\nabla_a Q^w$` at `$a = \mu_\theta(s)$`. The state-value baseline `$V^v(s)$` (which can be a rich nonlinear function, e.g., a neural network) absorbs any constant offset, so the total `$Q^w$` can approximate `$Q^\mu$` reasonably well at `$a = \mu_\theta(s)$` even though the advantage term is only linear.

**Condition 2 in practice.** Condition 2 requires finding `$w$` that minimizes the MSE between `$\nabla_a Q^w$` and `$\nabla_a Q^\mu$` at `$a = \mu_\theta(s)$`. This is a linear regression problem: predict the true action gradient `$\nabla_a Q^\mu(s, a)|_{a=\mu_\theta(s)}$` (an `$m$`-dimensional target) from features `$\phi(s, a) = \nabla_\theta \mu_\theta(s)(a - \mu_\theta(s))$` (an `$n$`-dimensional feature vector). However, acquiring unbiased samples of `$\nabla_a Q^\mu(s, a)|_{a=\mu_\theta(s)}$` is difficult — it requires estimating the gradient of the true action-value function, which is unknown.

In practice, the paper uses a standard policy evaluation method (Q-learning or gradient Q-learning) to estimate `$w$` by minimizing the TD error rather than the gradient MSE. The justification is:

> "We note that a reasonable solution to the policy evaluation problem will find `$Q^w(s, a) \approx Q^\mu(s, a)$` and will therefore approximately (for smooth function approximators) satisfy `$\nabla_a Q^w(s, a)|_{a=\mu_\theta(s)} \approx \nabla_a Q^\mu(s, a)|_{a=\mu_\theta(s)}$`."

In other words, if `$Q^w$` is a good approximation of `$Q^\mu$` (in value space), then their gradients at `$\mu_\theta(s)$` should also be close — provided both functions are smooth. The compatible structural form (Condition 1) ensures that the gradient approximation is parameterized appropriately; standard TD learning (approximating Condition 2) ensures the parameters are set to reasonable values. This is a relaxation of the strict compatibility conditions (Sutton et al., 1999 made a similar relaxation for stochastic actor-critic), trading theoretical bias guarantees for practical efficiency.

---

#### COPDAC-Q and COPDAC-GQ Algorithms (Section 4.3)

The paper presents two instantiations of compatible off-policy deterministic actor-critic, differing in how the critic parameters are learned.

**COPDAC-Q (Compatible Off-Policy Deterministic Actor-Critic with Q-learning):** Uses standard Q-learning with the compatible function approximator form. The critic is:

$$Q^w(s, a) = \phi(s, a)^\top w + V^v(s)$$

where `$\phi(s, a) = \nabla_\theta \mu_\theta(s)(a - \mu_\theta(s))$` is the `$n$`-dimensional feature vector, `$w \in \mathbb{R}^n$` are the advantage parameters, and `$V^v(s) = v^\top \phi(s)$` is a linear state-value baseline with parameters `$v$`.

The updates (Equations 19-22) are:

$$\delta_t = r_t + \gamma Q^w(s_{t+1}, \mu_\theta(s_{t+1})) - Q^w(s_t, a_t)$$

$$\theta_{t+1} = \theta_t + \alpha_\theta \nabla_\theta \mu_\theta(s_t) \left(\nabla_\theta \mu_\theta(s_t)^\top w_t\right)$$

$$w_{t+1} = w_t + \alpha_w \delta_t \phi(s_t, a_t)$$

$$v_{t+1} = v_t + \alpha_v \delta_t \phi(s_t)$$

**Actor update simplification.** The actor update `$\nabla_\theta \mu_\theta(s_t) (\nabla_\theta \mu_\theta(s_t)^\top w_t)$` is the product of the `$n \times m$` Jacobian `$\nabla_\theta \mu_\theta(s_t)$` and the `$m$`-vector `$\nabla_\theta \mu_\theta(s_t)^\top w_t$`. Notice that this equals `$\nabla_\theta \mu_\theta(s_t) \nabla_\theta \mu_\theta(s_t)^\top w_t$` — an `$n \times n$` matrix times an `$n$`-vector. This is exactly the on-policy natural gradient direction (see below).

**Critic updates.** The advantage parameters `$w$` and state-value parameters `$v$` are both updated by the TD error `$\delta_t$` multiplied by their respective feature vectors. For `$w$`, the feature is `$\phi(s_t, a_t) = \nabla_\theta \mu_\theta(s_t)(a_t - \mu_\theta(s_t))$`; for `$v$`, the feature is `$\phi(s_t)$` (the state features). The learning rates `$\alpha_w$` and `$\alpha_v$` may differ.

**COPDAC-GQ (Compatible Off-Policy Deterministic Actor-Critic with Gradient Q-learning):** It is well-known that off-policy Q-learning with linear function approximation can diverge (the deadly triad: function approximation, bootstrapping, and off-policy learning). Gradient temporal-difference methods (Sutton et al., 2009) solve this by performing true gradient descent on the mean-squared projected Bellman error (MSPBE), guaranteeing convergence under linear function approximation.

The COPDAC-GQ algorithm combines the compatible deterministic actor with a gradient Q-learning critic (Maei et al., 2010). The updates (Equations 23-27) are:

$$\delta_t = r_t + \gamma Q^w(s_{t+1}, \mu_\theta(s_{t+1})) - Q^w(s_t, a_t)$$

$$\theta_{t+1} = \theta_t + \alpha_\theta \nabla_\theta \mu_\theta(s_t) \left(\nabla_\theta \mu_\theta(s_t)^\top w_t\right)$$

$$w_{t+1} = w_t + \alpha_w \delta_t \phi(s_t, a_t) - \alpha_w \gamma \phi(s_{t+1}, \mu_\theta(s_{t+1})) \left(\phi(s_t, a_t)^\top u_t\right)$$

$$v_{t+1} = v_t + \alpha_v \delta_t \phi(s_t) - \alpha_v \gamma \phi(s_{t+1}) \left(\phi(s_t, a_t)^\top u_t\right)$$

$$u_{t+1} = u_t + \alpha_u \left(\delta_t - \phi(s_t, a_t)^\top u_t\right) \phi(s_t, a_t)$$

**What the additional terms do.** The gradient Q-learning critic introduces an auxiliary parameter vector `$u_t \in \mathbb{R}^n$` that tracks the gradient of the MSPBE. The terms `$-\alpha_w \gamma \phi(s_{t+1}, \mu_\theta(s_{t+1}))(\phi(s_t, a_t)^\top u_t)$` in the `$w$` and `$v$` updates are **correction terms** that remove the bias introduced by off-policy sampling with function approximation. The update for `$u_t$` is a standard LMS (least mean squares) update that maintains `$u_t$` as an estimate of the solution to a particular linear system arising from the gradient of the MSPBE.

**Why gradient Q-learning is needed.** Standard Q-learning with linear function approximation updates `$w$` in the direction of `$\delta_t \phi(s_t, a_t)$`. This is not a true gradient of any fixed objective — it is a semi-gradient method. Under off-policy sampling, the key matrix that governs convergence can have eigenvalues with positive real parts, causing divergence. GQ (gradient Q-learning) fixes this by following the true gradient of the MSPBE, which is always a descent direction, guaranteeing convergence of the critic to a fixed point under appropriate step-size conditions.

**Two-timescale convergence.** The paper notes that "under suitable conditions on the step-sizes, `$\alpha_\theta, \alpha_w, \alpha_u$`, to ensure that the critic is updated on a faster time-scale than the actor, the critic will converge to the parameters minimising the MSPBE." The standard two-timescale setup is:

$$\lim_{t \to \infty} \frac{\alpha_\theta}{\alpha_w} = 0, \quad \lim_{t \to \infty} \frac{\alpha_\theta}{\alpha_u} = 0$$

This means the critic learns much faster than the actor, so from the actor's perspective, the critic is always near its asymptotic value for the current policy — a quasi-stationary critic. This is a standard technique in actor-critic convergence analysis (Bhatnagar et al., 2007).

**Natural policy gradient for deterministic policies.** The paper concludes Section 4.3 by showing that the natural policy gradient — the steepest ascent direction with respect to the Fisher information metric — extends naturally to deterministic policies. For stochastic policies, the Fisher metric is:

$$M^\pi(\theta) = \mathbb{E}_{s \sim \rho^\pi, a \sim \pi_\theta}\left[\nabla_\theta \log \pi_\theta(a|s) \nabla_\theta \log \pi_\theta(a|s)^\top\right]$$

For deterministic policies, the paper uses the limiting metric:

$$M^\mu(\theta) = \mathbb{E}_{s \sim \rho^\mu}\left[\nabla_\theta \mu_\theta(s) \nabla_\theta \mu_\theta(s)^\top\right]$$

which is an `$n \times n$` matrix (the outer product of the `$n \times m$` Jacobian with its transpose, summed over states). The steepest ascent direction is `$M^\mu(\theta)^{-1} \nabla_\theta J(\mu_\theta)$`.

Using the compatible function approximation form, `$\nabla_\theta J(\mu_\theta) = \mathbb{E}_{s \sim \rho^\mu}[\nabla_\theta \mu_\theta(s) \nabla_\theta \mu_\theta(s)^\top w] = M^\mu(\theta) w$`. Therefore, the natural gradient direction is:

$$M^\mu(\theta)^{-1} \nabla_\theta J(\mu_\theta) = M^\mu(\theta)^{-1} M^\mu(\theta) w = w$$

**What this means operationally:** the natural gradient of the performance objective with respect to the policy parameters, under compatible function approximation, is exactly the critic parameter vector `$w$`. The actor update simplifies to:

$$\theta_{t+1} = \theta_t + \alpha_\theta w_t$$

This is remarkably simple. The critic parameters `$w$` — which represent the linear coefficients of the advantage function `$A^w(s, a) = \phi(s, a)^\top w$` — directly give the natural gradient direction for the actor. No matrix inversion, no Fisher information estimation — just use the critic parameters as the update direction. The paper notes that this variant "can be implemented by simplifying Equations 20 or 24 to `$\theta_{t+1} = \theta_t + \alpha_\theta w_t$`."

**Why this connection matters.** The natural gradient is invariant to reparameterization of the policy (Bagnell and Schneider, 2003), meaning it represents a geometrically meaningful direction independent of how the policy parameters are scaled or correlated. The fact that `$w$` is the natural gradient means that the compatible critic provides not just an unbiased gradient estimate, but the *covariant* update direction — the direction that accounts for the parameterization geometry. This is the same property that makes natural actor-critic (Peters et al., 2005) attractive for stochastic policies, now extended to deterministic policies essentially for free given the compatible function approximation structure.

---

#### Summary of Algorithm Variants and Their Relationships

The paper presents a hierarchy of algorithms, each building on the previous:

1. **On-policy deterministic AC (Section 4.1):** Sarsa critic, deterministic on-policy actor. Didactic; suffers from exploration collapse unless the environment provides sufficient noise.

2. **OPDAC (Section 4.2):** Q-learning critic, deterministic off-policy actor. Practical; separates exploration (stochastic behaviour policy) from exploitation (deterministic target policy). No importance sampling needed in either actor or critic.

3. **COPDAC-Q (Section 4.3):** Same as OPDAC but with the critic constrained to the compatible form `$Q^w(s, a) = \phi(s, a)^\top w + V^v(s)$` where `$\phi(s, a) = \nabla_\theta \mu_\theta(s)(a - \mu_\theta(s))$`. This provides unbiased gradient estimates when Condition 2 is satisfied (which it is approximately, via Q-learning). The actor update is `$\theta_{t+1} = \theta_t + \alpha_\theta \nabla_\theta \mu_\theta(s_t) \nabla_\theta \mu_\theta(s_t)^\top w_t$`.

4. **COPDAC-GQ (Section 4.3):** Same as COPDAC-Q but with a gradient Q-learning critic instead of standard Q-learning. Guarantees convergence of the critic under linear function approximation and off-policy sampling, at the cost of maintaining an auxiliary parameter vector `$u_t$` and additional correction terms in the critic updates.

5. **Natural deterministic AC:** A simplified variant of COPDAC where the actor update is `$\theta_{t+1} = \theta_t + \alpha_\theta w_t$`, justified by the fact that `$w$` is the natural gradient direction when using compatible function approximation.

All algorithms share `$O(mn)$` computational cost per timestep (linear in action dimensionality and number of policy parameters), and all use the same deterministic policy gradient structure: the actor follows `$\nabla_a Q^w$` evaluated at `$a = \mu_\theta(s)$`, chain-ruled through `$\nabla_\theta \mu_\theta(s)$`.

## 4. Key Insights and Innovations

### Innovation 1: Proving That a Model-Free Deterministic Policy Gradient Exists — and That It's the Limit of the Stochastic Policy Gradient

The paper's most fundamental intellectual contribution is not the algorithm itself, but the **theoretical proof that the deterministic policy gradient exists** in a model-free setting. Prior to this work, the field operated under the assumption — formalized by Peters (2010) — that the deterministic policy gradient either did not exist or was only computable when a model of the environment dynamics was available. This was not an arbitrary gap; it was a mathematical consequence of how the policy gradient theorem had always been formulated. The stochastic policy gradient (Sutton et al., 1999) relies on the score function `∇_\theta \log \pi_\theta(a|s)`, which requires a probability density over actions. A deterministic policy has no such density — it maps each state to a single action with no distribution — so the standard machinery seemed inapplicable.

What the paper does is **bypass this apparent impossibility** by deriving the gradient directly from the performance objective `J(\mu_\theta)`, without ever invoking a score function. The resulting gradient (Equation 9) has a fundamentally different structure from the stochastic case: instead of `E[∇_\theta \log \pi_\theta(a|s) Q^\pi(s,a)]`, it is `E[∇_\theta \mu_\theta(s) ∇_a Q^\mu(s,a)|_{a=\mu_\theta(s)}]`. The Jacobian of the policy `∇_\theta \mu_\theta(s)` replaces the score function, and the action-value gradient `∇_a Q^\mu` replaces the action-value itself. This is not a minor algebraic rearrangement — it's a **qualitatively different object** that integrates over states only, not states and actions. The proof (Appendix B) follows the same telescoping-sum strategy as Sutton et al. (1999) but applied to `∇_\theta Q^\mu(s, \mu_\theta(s))` rather than `∇_\theta V^\pi(s)`, establishing that the state-distribution gradient terms cancel out in the deterministic case exactly as they do in the stochastic case.

The connection to stochastic policy gradients is equally important. Theorem 2 shows that for a broad class of stochastic policies parameterized by a deterministic mean `\mu_\theta` and a variance parameter `σ`, the stochastic policy gradient converges to the deterministic policy gradient as `σ → 0`. This is a **unification result**: the deterministic policy gradient is not a separate theory but the limiting case of the established stochastic framework. This insight has immediate practical implications — it means that all the machinery developed for stochastic policy gradients (compatible function approximation, natural gradients, actor-critic architectures, batch methods) transfers to deterministic policies without reinvention. The compatible function approximation theory in Theorem 3 is a direct consequence: the stochastic compatibility condition `Q^w(s,a) = ∇_\theta \log \pi_\theta(a|s)^⊤ w` becomes, in the zero-variance limit, `∇_a Q^w(s,a)|_{a=\mu_\theta(s)} = ∇_\theta \mu_\theta(s)^⊤ w`, which is exactly Condition 1 of Theorem 3. The natural gradient metric `M^\mu(\theta) = E[∇_\theta \mu_\theta(s) ∇_\theta \mu_\theta(s)^⊤]` similarly emerges as the limit of the Fisher information metric.

This is a **fundamental theoretical advance**, not an incremental refinement. Before Silver et al. (2014), the policy gradient literature bifurcated: stochastic policies had principled gradient-based optimization, while deterministic policies relied on value-based methods with greedy maximization (which is intractable in continuous action spaces). The paper bridges this divide, showing that deterministic policies can be optimized by gradient ascent on the performance objective using the same conceptual framework as stochastic policies. The significance goes beyond any single algorithm — it expands the scope of what policy gradient methods can optimize.

---

### Innovation 2: Identifying That the Deterministic Policy Gradient Eliminates the Action Integral — and Why This Matters for High-Dimensional Control

The paper's second major insight is at once mathematical and deeply practical: **the deterministic policy gradient integrates only over the state space, whereas the stochastic policy gradient integrates over both state and action spaces**. This is not just an efficiency gain — it's a **structural difference** that explains why stochastic methods struggle in high-dimensional action spaces and why deterministic methods can succeed there.

To appreciate the significance, consider what the stochastic policy gradient actually estimates. Equation 2 requires approximating `∫_A ∇_\theta \pi_\theta(a|s) Q^\pi(s,a) da` by sampling actions from `\pi_\theta`. In an `m`-dimensional action space, this inner integral becomes increasingly difficult to estimate accurately as `m` grows — the volume of the action space grows exponentially, requiring exponentially more samples to maintain the same estimator variance. This is the standard curse of dimensionality applied to policy gradients. The deterministic policy gradient avoids this entirely: the action `a = \mu_\theta(s)` is computed deterministically, so there is no action sampling and no action integral. The gradient estimate `∇_\theta \mu_\theta(s) ∇_a Q^w(s,a)|_{a=\mu_\theta(s)}` requires only the state `s` and the critic's gradient at a single action point.

The continuous bandit experiments in Section 5.1 (Figure 1) make this difference stark. At 10 action dimensions, the deterministic actor-critic (COPDAC-B) outperforms the stochastic actor-critic (SAC-B), but the gap is relatively modest — both methods make progress. At 25 dimensions, the gap widens considerably. At 50 dimensions, the deterministic method converges **several orders of magnitude faster** than the stochastic method. This is not a tuned-hyperparameter artifact; it's the direct consequence of the stochastic method needing to sample a 50-dimensional action space to estimate its gradient, while the deterministic method computes its gradient analytically from a single state observation. The paper notes in Section 6 that the variance of the stochastic policy gradient for a Gaussian policy scales as `1/σ^2` (Zhao et al., 2012), which means that as the policy becomes more deterministic (as it must, to converge to a good solution), the estimator variance diverges. The deterministic policy gradient has no `σ` parameter and therefore no `1/σ^2` explosion — its variance is governed entirely by the state sampling and the critic's accuracy, not by the policy's determinism.

This insight generalizes beyond the bandit setting. The octopus arm experiment (Section 5.3, Figure 3) demonstrates that deterministic policy gradients can tackle a problem with **20 continuous action dimensions and 50 state dimensions** — a scale where prior stochastic policy gradient work had resorted to dimensionality reduction via "macro-actions" (Engel et al., 2005) or lower-dimensional arms (Heess et al., 2012). The paper's ability to apply COPDAC-Q directly to the full 20-dimensional action space, using a neural network policy with compatible function approximation, represents a qualitative jump in what continuous-action RL could handle at the time.

This is a **diagnostic insight** as much as an algorithmic one. It identifies exactly where stochastic policy gradients fail (high-dimensional action spaces, policies approaching determinism) and provides a principled alternative. The field's prior response to high-dimensional actions had been to constrain the action space, use factored policies, or inject heuristics — all workarounds for a fundamental estimator limitation. The deterministic policy gradient addresses the limitation directly by changing what quantity is estimated.

---

### Innovation 3: Separating Exploration from the Target Policy via Off-Policy Learning — and Showing That Deterministic Policies Eliminate Importance Sampling

The paper's third conceptual contribution is the design of an **off-policy actor-critic architecture where the target policy is deterministic but the behaviour policy remains stochastic for exploration**. This separation is not itself new — Degris et al. (2012b) introduced off-policy actor-critic for stochastic policies. What's new is the recognition that, for deterministic target policies, this separation yields a specific and non-obvious advantage: **the actor does not require importance sampling**.

To see why this matters, recall the stochastic off-policy policy gradient (Equation 5):

`∇_\theta J_\beta(\pi_\theta) ≈ E_{s∼ρ^β, a∼β} [(π_\theta(a|s) / β(a|s)) ∇_\theta \log \pi_\theta(a|s) Q^\pi(s,a)]`

The importance weight `π_\theta(a|s) / β(a|s)` corrects for the mismatch between the behaviour policy's action distribution and the target policy's action distribution. This ratio can have high variance, especially when `β` is broad (for exploration) and `π_\theta` is narrow (because the policy is converging). The variance of the importance weights becomes another source of noise in the gradient estimate — compounding the action-sampling variance that the deterministic gradient already eliminates.

The deterministic off-policy gradient (Equation 15) simply drops the importance weight:

`∇_\theta J_\beta(\mu_\theta) ≈ E_{s∼ρ^β} [∇_\theta \mu_\theta(s) ∇_a Q^\mu(s,a)|_{a=\mu_\theta(s)}]`

There is no `π_\theta(a|s) / β(a|s)` term because there is no action sampled from the target policy — the action is computed deterministically as `\mu_\theta(s)`. The expectation is over states from the behaviour policy's distribution `ρ^β`, and that's it. The paper frames this as a direct consequence of eliminating the action integral: since the deterministic policy gradient integrates only over states, the only distribution mismatch to worry about is the state distribution (`ρ^β` vs. `ρ^\mu`), and the off-policy objective `J_\beta` is explicitly defined as an expectation over `ρ^β`, so no correction is needed. This is a clever use of the Degris et al. (2012b) objective-redefinition trick, but applied in a context where it yields a bigger simplification than in the stochastic case — because the stochastic case still needs action-level importance weights even after redefining the objective.

The practical consequence is that **the entire off-policy deterministic actor-critic operates without importance sampling** — the actor because of the above, and the critic because Q-learning (with its `Q(s_{t+1}, \mu_\theta(s_{t+1}))` bootstrap) is inherently off-policy and does not require importance weights either. This makes the algorithm simpler, lower-variance, and more stable than its stochastic counterpart (OffPAC, Degris et al., 2012b), which requires importance sampling in both actor and critic.

The continuous RL experiments in Section 5.2 (Figure 2) validate this: COPDAC-Q slightly outperforms both the on-policy stochastic actor-critic (SAC) and the off-policy stochastic actor-critic (OffPAC) across mountain car, pendulum, and puddle world — three standard benchmarks. The performance gap is modest in these low-dimensional settings (1-2 action dimensions), which is consistent with the paper's thesis: the advantages of deterministic gradients compound with action dimensionality. The bandit experiments (Figure 1), where the action space is 10-50 dimensional, are where the gap becomes dramatic. This pattern — modest gains in low dimensions, orders-of-magnitude gains in high dimensions — is exactly what the theory predicts.

---

### Innovation 4: Establishing Compatible Function Approximation for Deterministic Policies — and Revealing the Natural Gradient as the Critic Parameters

The paper's fourth innovation extends the **compatible function approximation** theory from stochastic policies to deterministic policies, and in doing so reveals a remarkably simple relationship: **under compatible approximation, the natural policy gradient is exactly the critic parameter vector `w`**. This connection is both theoretically elegant and practically significant.

For stochastic policies, Sutton et al. (1999) showed that a critic of the form `Q^w(s,a) = ∇_\theta \log \pi_\theta(a|s)^⊤ w` is compatible — substituting it for the true `Q^\pi` in the policy gradient does not bias the estimate, provided `w` minimizes the mean-squared error. This result underpinned the natural actor-critic (Peters et al., 2005), where the critic parameters `w` directly give the natural gradient direction. The paper's Theorem 3 provides the deterministic analogue: a critic is compatible if `∇_a Q^w(s,a)|_{a=\mu_\theta(s)} = ∇_\theta \mu_\theta(s)^⊤ w` (Condition 1) and `w` minimizes the MSE between `∇_a Q^w` and `∇_a Q^\mu` (Condition 2).

The compatible structural form `Q^w(s,a) = (a - \mu_\theta(s))^⊤ ∇_\theta \mu_\theta(s)^⊤ w + V^v(s)` has an elegant interpretation. The first term `(a - \mu_\theta(s))^⊤ ∇_\theta \mu_\theta(s)^⊤ w` is the **advantage function**, linear in the deviation `a - \mu_\theta(s)` from the deterministic action. This is a local linear model of how much better or worse alternative actions are compared to the current policy. The critic does not need to model `Q^\mu` accurately for actions far from `\mu_\theta(s)` — it only needs to provide the correct gradient direction at `\mu_\theta(s)`, and a linear model is sufficient for that. The baseline `V^v(s)` absorbs the value at `\mu_\theta(s)`, ensuring the total `Q^w` approximates `Q^\mu` well at the evaluation point. This is a **local critic** design: globally inaccurate but locally precise in exactly the way the actor needs.

The revelation that `w` is the natural gradient direction follows directly: `∇_\theta J(\mu_\theta) = E[∇_\theta \mu_\theta(s) ∇_\theta \mu_\theta(s)^⊤ w] = M^\mu(\theta) w`, so `M^\mu(\theta)^{-1} ∇_\theta J(\mu_\theta) = w`. The actor update simplifies to `θ_{t+1} = θ_t + α_θ w_t`. No matrix inversion, no Fisher information estimation, no extra computation beyond what the critic already does. The compatible critic **simultaneously** solves the policy evaluation problem and provides the natural gradient direction for the actor.

This is a **refinement of existing ideas** (compatible approximation and natural gradients both existed for stochastic policies), but it is a fundamentally important refinement because it shows that the entire natural actor-critic framework transfers to deterministic policies essentially for free given the right critic parameterization. The paper does not need to develop a separate theory of natural gradients for deterministic policies — it falls out of the compatible function approximation conditions, which themselves are the deterministic limit of the stochastic conditions. This conceptual economy — where one theorem (Theorem 2) connects two previously separate frameworks, and another (Theorem 3) immediately yields natural gradients — is a hallmark of good theory.

The practical benefit is that the COPDAC algorithms (Q and GQ variants) implement natural gradient descent without the computational overhead typically associated with natural gradients (matrix inversion or Krylov subspace methods). For an `n`-parameter policy, the natural gradient computation is `O(n)` — just extracting `w` from the critic — rather than `O(n^3)` for matrix inversion or `O(n^2)` for iterative approximation. This makes natural gradients feasible for high-dimensional policies (like the neural network in the octopus arm experiment), where explicit matrix operations would be prohibitively expensive.

## 5. Experimental Analysis

### Evaluation Methodology

- **Dataset.** The experiments use a continuous bandit problem with a quadratic cost function, three standard continuous-action reinforcement learning benchmarks (mountain car, pendulum, 2D puddle world), and a simulated 6-segment octopus arm control task. The bandit uses action dimensions of 10, 25, and 50 with cost matrix `C` having eigenvalues from `{0.1, 1}` and optimal action `a* = [4, ..., 4]⊤`. The continuous RL benchmarks are variants of standard tasks modified for continuous actions, with the mountain car, pendulum, and puddle world environments described as "standard" without further dataset citation. The octopus arm task (Engel et al., 2005) has 50 continuous state variables and 20 action dimensions controlling muscle activations and base rotation. The paper does not report a separate train/test split for these environments — all results are reported during the learning process itself, with performance measured as the average reward per episode or cost per step over training runs.

- **Base model(s).** The bandit experiments use a deterministic target policy `µ_θ = θ` (identity parameterization) with a fixed-width Gaussian behaviour policy `β(·) ∼ N(θ, σ²_β)`, and a critic estimated by linear regression from compatible features. The continuous RL experiments (mountain car, pendulum, puddle world) use a linear deterministic target policy `µ_θ(s) = θ⊤φ(s)` where `φ(s)` are tile-coding features of the state, with a fixed-width Gaussian behaviour policy `β(·|s) ∼ N(θ⊤φ(s), σ²_β)`. The critic uses a linear state-value function `V(s) = v⊤φ(s)` as a baseline for the compatible action-value function. The octopus arm experiment uses a sigmoidal multi-layer perceptron with 8 hidden units and sigmoidal output units to represent the policy `µ(s)`, with a compatible linear advantage function `A^w(s, a)` and a state-value function `V^v(s)` represented by a second multi-layer perceptron with 40 hidden units and linear output units. The paper selects PaLM 2-S* (Codey) as "representative of the capabilities of many contemporary LLMs" — *wait, that's from the reference example, not this paper*. The paper does not use a pretrained model; all policies and critics are learned from scratch. For the continuous RL tasks, the paper compares against the stochastic actor-critic (SAC) algorithm from Degris et al. (2012a), which was selected because it "performed best out of several incremental actor-critic methods in a comparison on mountain car."

- **Metrics.** For the continuous bandit, performance is measured as the **average cost per step incurred by the mean** (i.e., exploration is not penalized for the on-policy algorithm). The cost function is `-r(a) = (a - a*)⊤C(a - a*)`, so lower cost is better. For mountain car, pendulum, and puddle world, the metric is **total reward per episode**, measured on test runs using the mean (deterministic) policy, averaged over 30 runs. For the octopus arm, the paper reports **return per episode** and **number of time-steps for the arm to reach the target**, tracked over 10 training runs. All metrics are computed during the learning process to show convergence behavior, not just final performance. The paper notes that episodes were truncated after a maximum of 5000 steps for the continuous RL tasks, and the octopus episodes end when the target is hit (with a reward bonus of +50) or after 300 steps.

- **Baselines.** Five baselines are used across the experiments:
  1. **SAC-B (Stochastic Actor-Critic for Bandit):** Uses an isotropic Gaussian policy `π_{θ,y}(·) ∼ N(θ, exp(y))` that adapts both mean and variance. The critic is estimated by linear regression from compatible features `∇_θ log π_θ(a)` to the costs. (Section 5.1)
  2. **SAC (Stochastic Actor-Critic):** The on-policy stochastic actor-critic from Degris et al. (2012a), using a Gaussian policy `π_{θ,y}(s,·) ∼ N(θ⊤φ(s), exp(y⊤φ(s)))` that adapts both mean and variance, with a linear value function `V(s) = v⊤φ(s)` updated by temporal-difference learning. (Section 5.2)
  3. **OffPAC (Off-Policy Stochastic Actor-Critic):** The off-policy stochastic actor-critic from Degris et al. (2012b), using the same behaviour policy `β` as the deterministic algorithms but learning a stochastic policy `π_{θ,y}`. Uses importance sampling for the actor, and the same linear critic as SAC. (Section 5.2)
  4. **COPDAC-Q (Compatible Off-Policy Deterministic Actor-Critic with Q-learning):** The paper's proposed algorithm, using a linear target policy and fixed-width Gaussian behaviour policy, with Q-learning critic updates and compatible function approximation. (Sections 5.2, 5.3)
  5. **COPDAC-B (Continuous Bandit variant):** A specialized variant of COPDAC for the bandit task, using a target policy `µ_θ = θ`, with critic estimated by linear regression from compatible features `∇_θ µ_θ(a)(a - θ)`. (Section 5.1)

  Notably, the paper does **not** compare against NFQCA (Hafner and Riedmiller, 2011) — the precursor that also used action-value gradients — despite discussing it in Section 6. The paper also does not compare against standard Q-learning with discretized actions or against model-based policy gradient methods (Werbos, 1990).

- **Generation budget / compute accounting.** Compute is measured in **time-steps of interaction with the environment**, which is the standard metric for online RL. All algorithms are incremental (one update per time-step), so wall-clock time and sample complexity are proportional under the assumption that per-step computation costs are similar. The paper notes that "the computational cost of each update is linear in the action dimensionality and the number of policy parameters" for all proposed algorithms, specifically `O(mn)` per time-step for `m` action dimensions and `n` policy parameters. For the bandit experiments, the critic is recomputed from each successive batch of 2m steps and the actor is updated once per batch, making the comparison fair in terms of environment interactions. For the continuous RL benchmarks, all algorithms use the same discount factors (`γ = 0.99` for mountain car and pendulum, `γ = 0.999` for puddle world), and the same feature representation (tile-coding). The paper conducts "a parameter sweep over all step-size parameters and variance parameters" for each algorithm, reporting performance of the best parameters for each, which ensures that comparisons are not biased by poor hyperparameter tuning in any single algorithm.

- **Cross-validation / statistical protocol.** The paper does not use cross-validation in the traditional supervised learning sense. Instead, it reports performance averaged over multiple independent runs: 5 runs for the bandit experiments (Figure 1), 30 runs for the continuous RL benchmarks (Figure 2), and 10 runs for the octopus arm (Figure 3). Error bars or confidence intervals are not shown in the figures (which plot single curves), though the octopus arm figure (Figure 3) shows all 10 individual runs overlaid, providing a visual sense of variance. The bandit figures plot cost vs. time-steps with the "best performing parameters for each algorithm," which means the reported curves represent the best hyperparameter configuration found via sweeping, not the average over configurations. This is a standard but potentially optimistic reporting practice — the best-of-sweep performance may overstate expected performance with default parameters. The paper does not report standard errors, statistical tests, or sensitivity of results to hyperparameter choices beyond the statement that a sweep was performed.

### Main Quantitative Results

#### Continuous Bandit: Deterministic vs. Stochastic Policy Gradients

The bandit experiments (Section 5.1, Figure 1) directly test the paper's central claim: that the deterministic policy gradient can be estimated more efficiently than the stochastic policy gradient, with advantages that grow with action dimensionality.

**Headline result at 50 action dimensions:** At `m = 50`, COPDAC-B reduces cost to approximately `10^{-4}` within roughly 1000 time-steps, while SAC-B remains above `10^{-1}` after 10,000 time-steps — a difference of roughly **three orders of magnitude in final cost and one order of magnitude in convergence speed**. The gap widens monotonically with dimensionality: at `m = 10`, both methods converge to similar costs (around `10^{-4}`) but COPDAC-B reaches this level roughly 2-3× faster; at `m = 25`, COPDAC-B converges to roughly `10^{-4}` in about 2000 steps while SAC-B requires approximately 5000 steps to reach `10^{-2}`; at `m = 50`, the stochastic method fails to make meaningful progress while the deterministic method converges rapidly.

**Why this experiment isolates the gradient estimation advantage.** The continuous bandit has no state dynamics — each action produces an immediate quadratic cost, and there is no sequential decision-making. This strips away all confounding factors: no temporal credit assignment, no state distribution shift, no exploration-exploitation tradeoff beyond the action sampling. The only difference between SAC-B and COPDAC-B is **how they estimate the gradient of expected cost with respect to the policy mean**. SAC-B samples actions from a Gaussian, computes the score function `∇_θ log π_θ(a)`, and averages; its variance scales with `1/σ²` (as discussed in Section 6). COPDAC-B computes the gradient analytically: `∇_θ µ_θ ∇_a Q(a)|_{a=θ}`. Since the critic `Q(a)` is estimated by linear regression from compatible features, the gradient estimate is exact up to regression error — there is no action-sampling noise.

**The ablation with fixed variance.** The paper reports an additional experiment where "the stochastic actor-critic used the same fixed variance `σ²_β` as the deterministic actor-critic, so that only the mean was adapted." Even in this setting, "COPDAC-B still outperforms SAC-B by a very wide margin that grows larger with increasing dimension." This is important because it isolates the effect of the gradient estimator from the effect of variance adaptation. SAC-B normally adapts its variance `y` to optimize exploration, which could be a confounding advantage. The fact that COPDAC-B dominates even when SAC-B is given the same fixed variance confirms that the advantage comes from the deterministic gradient estimator itself, not from better variance tuning.

**Magnitudes matter.** The bandit problem has a quadratic cost with optimal action `a* = [4,...,4]⊤`. The policy mean `θ` is initialized somewhere in the action space (initialization details not given, but presumably far from `a*`). The cost function is `(a - a*)⊤C(a - a*)`, where `C` has eigenvalues `{0.1, 1}` — meaning the cost landscape is anisotropic, with some directions being 10× steeper than others. An algorithm that can accurately estimate the gradient direction will navigate this anisotropy efficiently; one with noisy gradients will wander, especially in high dimensions where the probability of a random sample pointing in the right direction decreases exponentially. The deterministic gradient directly follows the steepest descent direction of the quadratic, modulo estimation error in the critic — which is why it converges to `10^{-4}` in hundreds of steps rather than thousands.

#### Continuous Reinforcement Learning: Mountain Car, Pendulum, Puddle World

The continuous RL experiments (Section 5.2, Figure 2) test whether the off-policy deterministic actor-critic (COPDAC-Q) outperforms both on-policy stochastic (SAC) and off-policy stochastic (OffPAC) methods on standard benchmarks with low-dimensional action spaces.

**Mountain Car (Figure 2a):** All three algorithms converge to similar asymptotic performance (approximately -1000 to 0 total reward per episode, with values around -0.1 to -1.0 × 1000 in the final portion of training). COPDAC-Q reaches the highest reward region slightly earlier — around 20,000 time-steps, it achieves approximately -1.0 × 1000 while SAC is at approximately -2.5 × 1000 — but the differences are modest. After roughly 40,000 time-steps, all three algorithms overlap substantially.

**Pendulum (Figure 2b):** COPDAC-Q and OffPAC converge to similar final performance (approximately -1.0 to 0 × 1000, close to zero), while SAC converges to a lower asymptotic reward (around -2.0 to -3.0 × 1000). COPDAC-Q reaches its asymptotic level by roughly 50,000-100,000 time-steps; OffPAC takes until 200,000+ time-steps to match it. SAC plateaus lower and does not catch up within the 500,000 time-steps shown.

**2D Puddle World (Figure 2c):** All three algorithms converge to similar reward levels (approximately -5.0 to -10.0 × 1000), but COPDAC-Q converges faster — reaching roughly -10.0 × 1000 by 50,000 time-steps, while SAC and OffPAC take until 200,000+ time-steps to reach similar levels. The asymptotic gap is narrow (all within roughly -5.0 to -7.0 × 1000 at 500,000 steps). These are low-dimensional problems (mountain car has 1 action dimension — throttle; pendulum has 1 action dimension — torque; puddle world has 2 action dimensions — x,y velocity or force). In this regime, the paper's theory predicts only modest advantages for deterministic gradients because the action-space integral that the stochastic method must approximate is over only 1-2 dimensions. The results are consistent with this prediction: COPDAC-Q is consistently better, but the margin is small, and all methods eventually reach similar performance levels.

**Side-by-side comparison at the same budget.** All algorithms are shown over the same number of time-steps (500,000 for mountain car and puddle world; 500,000 for pendulum — though the x-axis labels differ slightly across subfigures). At 50,000 time-steps in puddle world, COPDAC-Q achieves roughly -10.0 × 1000 while OffPAC achieves roughly -18.0 × 1000 and SAC roughly -20.0 × 1000 — approximately a 2× performance advantage. At 200,000 time-steps in pendulum, COPDAC-Q is at roughly -1.0 × 1000, OffPAC at roughly -1.5 × 1000, SAC at roughly -3.0 × 1000 — a 1.5× to 3× advantage.

**An important detail: the comparison is between deterministic target performance and stochastic target performance.** For COPDAC-Q, performance is measured by evaluating the **deterministic target policy** (the mean of the behaviour policy). For SAC and OffPAC, performance is measured by evaluating the **mean of the stochastic policy**. This is a fair comparison — in all cases, we care about the policy's expected action, not its exploratory noise. But it means SAC's learning process includes the optimization of variance (which adapts via `y⊤φ(s)`), while COPDAC-Q uses a fixed exploration variance. SAC has an extra degree of freedom (the variance parameters) that could help or hurt. The results suggest that in these low-dimensional tasks, the benefit of adaptive variance in SAC does not compensate for the noisier gradient estimates from the stochastic policy gradient.

#### Octopus Arm: High-Dimensional Control with Neural Network Policies

The octopus arm experiment (Section 5.3, Figure 3) is the paper's most ambitious demonstration, testing whether COPDAC-Q can scale to problems with **20 continuous action dimensions and 50 state dimensions** — a regime where stochastic policy gradient methods had previously required dimensionality reduction.

**Headline result:** Over 10 training runs, the octopus arm "converged to a good solution in all cases." Figure 3 (upper panel) shows the return per episode increasing from roughly 5 to 10-15 over 300,000 time-steps. Figure 3 (lower panel) shows the number of steps needed to reach the target decreasing from roughly 300 (the maximum, meaning many initial episodes time out without hitting) to roughly 100-200 at convergence. The paper also mentions a video of an 8-segment arm (more complex than the 6-segment arm in the reported results) trained by COPDAC-Q, though no quantitative results are provided for this variant.

**What makes this result significant.** Prior work on the octopus arm had simplified the problem: Engel et al. (2005) used 6 "macro-actions" — predefined patterns of muscle activations — reducing the effective action dimensionality from 20 to 6. Heess et al. (2012) applied stochastic policy gradients to a lower-dimensional version with only 4 segments. The paper's application of COPDAC-Q directly to the full 20-dimensional action space, using a neural network policy without macro-actions or dimensionality reduction, represents a qualitative advance in what continuous-action RL could handle at the time. The fact that all 10 runs converge to good solutions suggests the deterministic policy gradient method is robust at this scale, not just occasionally successful.

**Architecture details.** The policy `µ(s)` uses a sigmoidal multi-layer perceptron with 8 hidden units and sigmoidal output units — a relatively small network by modern standards, but sufficient for this task. The advantage function `A^w(s, a)` uses the compatible linear form `φ(s, a)⊤w` where `φ(s, a) = ∇_θ µ_θ(s)(a - µ_θ(s))`, ensuring Conditions 1 of Theorem 3 is satisfied. The state-value function `V^v(s)` uses a separate multi-layer perceptron with 40 hidden units and linear outputs — this can be nonlinear without violating compatibility because the baseline function can be "any differentiable baseline function that is independent of the action a" (Section 4.3). The paper emphasizes that "the compatibility criteria apply to any differentiable baseline, including non-linear state-value functions." This separation — a linear compatible advantage term plus a rich nonlinear state-value baseline — is a practical design pattern that combines the theoretical guarantees of compatible function approximation with the representational power needed to model complex value functions.

**A missing comparison.** The paper does not compare COPDAC-Q against stochastic policy gradients on the octopus arm. This is understandable given that prior work had not successfully applied stochastic methods to the full 20-dimensional version, but it means we cannot quantify the advantage of deterministic over stochastic gradients in this high-dimensional setting — we only know that deterministic methods work, not how much better they are. The bandit experiment provides the controlled dimensionality comparison, but the octopus arm adds the complications of temporal credit assignment, state representation learning, and neural network optimization. A direct comparison, even at a reduced dimensionality, would have strengthened the paper's claims about scalability.

### Ablation Studies and Robustness Checks

**Fixed-variance stochastic policy in bandit task**: The paper reports an additional bandit experiment where "the stochastic actor-critic used the same fixed variance `σ²_β` as the deterministic actor-critic, so that only the mean was adapted." Section 5.1 states: "This did not improve the performance of the stochastic actor-critic: COPDAC-B still outperforms SAC-B by a very wide margin that grows larger with increasing dimension." This ablation controls for the possibility that SAC-B's inferiority comes from suboptimal variance adaptation rather than from the gradient estimator itself. The result confirms that the deterministic gradient estimator is the key advantage. The paper does not show a separate figure for this ablation; it is reported in text only.

**On-policy deterministic vs. off-policy deterministic**: The paper's design is itself an ablation. The on-policy deterministic actor-critic (Section 4.1) is presented but not experimentally evaluated — the paper implicitly acknowledges that it would fail without exploration, and all experimental results use the off-policy version with a stochastic behaviour policy. This is a deliberate choice that highlights the necessity of the off-policy formulation. However, it means we have no empirical evidence for how badly the on-policy version fails — the "exploration collapse" is a theoretical prediction, not an empirically measured phenomenon in this paper.

**Stochastic off-policy (OffPAC) vs. deterministic off-policy (COPDAC-Q)**: This comparison (Figure 2) isolates the effect of the deterministic policy gradient from the effect of off-policy learning. Both COPDAC-Q and OffPAC use the same behaviour policy `β`, the same critic structure (linear value function), and the same off-policy objective `J_β`. The difference is the target policy type (deterministic vs. stochastic) and the gradient estimator (deterministic policy gradient vs. importance-weighted stochastic policy gradient). COPDAC-Q slightly outperforms OffPAC across all three continuous RL benchmarks, confirming that the deterministic gradient estimator provides a benefit beyond what off-policy learning alone provides.

**Q-learning vs. gradient Q-learning critic**: The paper presents COPDAC-Q and COPDAC-GQ as alternative critic implementations but does not experimentally compare them. COPDAC-GQ is introduced as a solution to the divergence issues of off-policy Q-learning with linear function approximation — a known theoretical problem (the deadly triad). The paper cites Sutton et al. (2009) and Maei et al. (2010) for the convergence guarantees of gradient Q-learning but does not verify empirically whether COPDAC-Q actually diverges on any task or whether COPDAC-GQ prevents divergence. This is a theoretical ablation rather than an empirical one: the paper provides the algorithm for practitioners who need convergence guarantees, but does not demonstrate that the guarantees matter in practice for the tested benchmarks.

**Variance initialization for continuous RL**: The paper states that "variance was initialised to 1/2 the legal range" for the stochastic algorithms (SAC, OffPAC) in the continuous RL experiments. This is a hyperparameter choice that could significantly affect early exploration. The paper does not ablate this choice by testing different initial variance values. Since the stochastic methods adapt their variance during learning, the initialization might not matter asymptotically — but in these finite-horizon experiments (500,000 steps), initialization can affect convergence speed.

**Natural gradient variant**: The paper notes that the natural gradient version of COPDAC simplifies the actor update to `θ_{t+1} = θ_t + α_θ w_t` (end of Section 4.3). This variant is not experimentally evaluated or compared to the standard COPDAC-Q update `θ_{t+1} = θ_t + α_θ ∇_θ µ_θ(s_t)(∇_θ µ_θ(s_t)⊤ w_t)`. The natural gradient variant would be computationally simpler (no Jacobian-vector product), but we don't know from the paper whether it performs differently in practice.

**Episode truncation:** Episodes in the continuous RL benchmarks were truncated after 5000 steps maximum. The paper does not ablate this limit to see if longer episodes would allow the stochastic methods to catch up to COPDAC-Q, or if the performance ordering is robust to truncation length. This is relevant because stochastic methods with poor gradient estimates might need more steps per episode to accumulate enough signal, and truncation could disadvantage them more than the deterministic method. However, the fact that performance is measured on separate test runs (not during the truncated training episodes) partially addresses this concern.

### Critical Assessment

The experiments in this paper demonstrate four things, each with varying degrees of empirical support:

**1. Deterministic policy gradients can be estimated more efficiently than stochastic policy gradients in high-dimensional action spaces.**

This is the paper's strongest empirical claim, and the continuous bandit experiments (Figure 1) directly support it with a clean, controlled comparison. The bandit isolates the gradient estimation mechanism from all other RL complexities (no state dynamics, no temporal credit assignment, no exploration beyond action sampling). The dimensionality scaling is clearly shown: the gap between COPDAC-B and SAC-B grows from noticeable at 10 dimensions to enormous at 50 dimensions. The fixed-variance ablation confirms that the advantage comes from the gradient estimator, not from variance adaptation.

However, the bandit is a deliberately simplified problem. The cost function is quadratic — the gradient is linear in the action, which makes the compatible linear critic exact. In more general RL problems, the action-value function `Q^µ(s, a)` is not quadratic, and the compatible linear advantage model `(a - µ_θ(s))⊤ ∇_θ µ_θ(s)⊤ w` is only a local approximation. The paper does not test how the deterministic gradient's efficiency degrades when the critic's local linear model is less accurate — for instance, on problems with highly nonlinear `Q`-functions or with policy parameterizations where `∇_θ µ_θ(s)` varies rapidly across states. The neural network policy used in the octopus arm experiment partially addresses this concern (the policy is nonlinear, so `∇_θ µ_θ(s)` is state-dependent), but we don't have a controlled dimensionality scaling study for the full RL setting. We know the deterministic gradient works at 20 action dimensions on the octopus arm; we don't know at what dimensionality it would start to struggle relative to stochastic methods, or whether the bandit scaling results (orders-of-magnitude advantage at 50 dimensions) generalize to sequential decision problems.

**2. Off-policy deterministic actor-critic (COPDAC) scales to high-dimensional continuous control tasks where prior methods required dimensionality reduction.**

The octopus arm results (Figure 3) support this claim in a binary sense: COPDAC-Q works on the full 20-dimensional action space, and prior stochastic methods (Engel et al., 2005; Heess et al., 2012) had not been applied to this full problem. However, the claim as stated is about **comparative** scaling — COPDAC scales *better* than alternatives — and the experiment provides only an existence proof, not a comparison. We don't know whether a well-tuned stochastic off-policy actor-critic with modern variance reduction techniques would also solve the 20-dimensional octopus arm. The paper's own low-dimensional RL results (Figure 2) show that COPDAC-Q is only modestly better than OffPAC in 1-2 action dimensions, so the key question is how this gap evolves as dimensionality increases in the full RL setting. The paper does not answer this — the bandit provides the dimensionality scaling study, but in a simplified setting; the octopus arm provides the real-world complexity, but at a single dimensionality.

The missing experiment is a controlled dimensionality scaling study on a sequential RL problem — for instance, varying the number of joints in a simulated robotic arm or the number of controlled degrees of freedom in a locomotion task — comparing COPDAC against OffPAC across dimensions. This would bridge the bandit results (which show *why* deterministic gradients should win) and the octopus arm results (which show *that* they can win on a hard problem) to establish *how much* they win as a function of problem dimension.

**3. The off-policy formulation eliminates the need for importance sampling in the actor.**

The paper claims this as a practical advantage of deterministic over stochastic off-policy actor-critic, but the experiments don't directly test it. The comparison between COPDAC-Q and OffPAC (Figure 2) encompasses multiple differences: deterministic vs. stochastic target policy, gradient estimator type, and presence/absence of importance weights in the actor. OffPAC's slightly worse performance could be due to any of these factors. A targeted experiment would compare COPDAC-Q against a variant of OffPAC that uses the same deterministic gradient but with importance weights artificially added, or conversely, a variant of OffPAC that somehow avoids importance weights. The paper doesn't do this, so the claim about importance sampling remains theoretically motivated but empirically unvalidated.

Additionally, the claim that "the entire off-policy deterministic actor-critic operates without importance sampling" is misleading regarding the critic. The critic in COPDAC-Q uses Q-learning, which is inherently off-policy and does not require importance sampling — but this is true for *any* Q-learning critic, regardless of whether the actor is deterministic or stochastic. The innovation is specifically that the *actor* avoids importance sampling, and the critic's independence from importance sampling is inherited from Q-learning, not from the deterministic policy gradient per se.

**4. Compatible function approximation provides an unbiased deterministic policy gradient.**

Theorem 3 proves that if the critic satisfies Conditions 1 and 2, substituting it for the true `Q^µ` does not bias the gradient. However, the experiments do **not** satisfy Condition 2. The critic is trained by Q-learning (minimizing TD error), not by minimizing `MSE(θ, w) = E[‖∇_a Q^w - ∇_a Q^µ‖²]`. The paper explicitly acknowledges this: "we learn `w` by a standard policy evaluation method... that does not exactly satisfy condition 2." The justification is that "a reasonable solution to the policy evaluation problem will find `Q^w(s, a) ≈ Q^µ(s, a)` and will therefore approximately (for smooth function approximators) satisfy `∇_a Q^w(s, a)|_{a=µ_θ(s)} ≈ ∇_a Q^µ(s, a)|_{a=µ_θ(s)}`."

This is a plausible argument but not empirically verified. An experiment that would test this assumption would compare the true deterministic policy gradient (estimated via finite differences on the true `Q^µ`, which would require a model or extensive sampling) against the critic-based gradient used in COPDAC-Q, measuring the angular error between them during training. If the error is small, Condition 2 is approximately satisfied; if it's large, the compatibility theory provides no guarantees. The paper does not perform this validation, so we don't know whether COPDAC-Q is actually following an unbiased gradient or whether it succeeds despite bias. The fact that COPDAC-Q outperforms SAC and OffPAC is evidence that whatever gradient it's following is useful, but it doesn't confirm the unbiasedness claim of Theorem 3.

In summary, the experiments strongly support the paper's core practical claim — deterministic policy gradients work better than stochastic ones, especially in high dimensions — while leaving several theoretical claims (importance sampling elimination, compatibility conditions, natural gradient equivalence) plausible but not directly tested. The bandit experiments are clean and convincing for the gradient estimation advantage. The RL experiments show that the method works on real problems but don't isolate *which* components of the method contribute how much. The octopus arm experiment demonstrates scalability but without a comparative baseline at that scale. This is a common pattern in algorithm papers: the theoretical framework provides the "why," the experiments provide the "that," and a full decomposition of the "how much from each component" is left to future work.

## 6. Limitations and Trade-offs

### The On-Policy Deterministic Actor-Critic Is Unusable Without Environmental Stochasticity

The paper acknowledges that the on-policy deterministic actor-critic (Section 4.1) is fundamentally limited in its exploration capabilities:

> "In general, behaving according to a deterministic policy will not ensure adequate exploration and may lead to sub-optimal solutions."

This understates the severity of the problem. A deterministic policy `μ_θ(s)` selects the same action every time it encounters the same state. If the policy is initialized poorly — or converges to a local optimum — it has **no mechanism whatsoever** to discover alternative strategies. Unlike stochastic policy gradient methods, where the policy's own variance provides persistent exploration (even if noisy), or value-based methods that use ε-greedy or Boltzmann exploration, the on-policy deterministic actor-critic generates no exploratory actions. The only source of state-space coverage is randomness in the environment's transitions `p(s_{t+1}|s_t, a_t)` or initial state distribution `p_1(s_1)`. In a deterministic environment (common in simulated robotics, for example), the agent would follow exactly the same trajectory on every episode and never gather new information.

**Consequence.** The on-policy algorithm is presented as "didactic" — a pedagogical stepping stone — and the paper does not even report experimental results for it. This is not a minor caveat; it means that **pure deterministic policy gradient methods cannot be deployed on-policy** in any environment where exploration matters. The entire practical contribution of the paper rests on the off-policy formulation (Section 4.2), which separates exploration (stochastic behaviour policy) from the target policy (deterministic). For practitioners, this means the deterministic policy gradient is fundamentally tied to off-policy learning — if off-policy learning cannot be made to work (due to the deadly triad, or because a behaviour policy cannot be safely deployed, or because importance sampling is required in the critic for some reason), the deterministic policy gradient is inapplicable.

**Evidence in the paper.** None. The on-policy algorithm is defined mathematically (Equations 11–13) but never evaluated. The paper provides no empirical evidence for *how badly* it fails — whether it gets stuck immediately, whether it sometimes works in stochastic environments, or what degree of environmental noise is sufficient. The failure of on-policy determinism is a theoretical deduction, not an observed result.

**Mitigation status.** Partially addressed. The off-policy algorithms (OPDAC, COPDAC-Q, COPDAC-GQ) solve the exploration problem by introducing a stochastic behaviour policy `β`, and all experimental results use these off-policy variants. However, this shifts the burden to making off-policy learning stable — which introduces its own set of challenges (see next limitation). The paper does not investigate whether it is possible to inject exploration noise directly into a deterministic policy (e.g., through parameter-space exploration) while still using on-policy updates, nor does it characterize the boundary conditions under which on-policy determinism might suffice.

---

### Off-Policy Learning with Function Approximation Has No Convergence Guarantees for the Practical Algorithms

The paper's practical algorithms rely on off-policy temporal-difference learning with function approximation — a combination known to be unstable in general. This is the "deadly triad" of reinforcement learning (function approximation, bootstrapping, off-policy learning; Sutton and Barto, 2018), and it can cause divergence even on simple problems. The paper is transparent about this:

> "These simple algorithms may have convergence issues in practice, due both to bias introduced by the function approximator, and also the instabilities caused by off-policy learning." (Section 4, introduction)

COPDAC-Q — the algorithm used in all continuous RL experiments (Section 5.2) and the octopus arm (Section 5.3) — uses standard Q-learning with the compatible linear function approximator. Standard Q-learning with linear function approximation and off-policy sampling is **not guaranteed to converge**; the semi-gradient update `w_{t+1} = w_t + α_w δ_t φ(s_t, a_t)` does not follow the gradient of any fixed objective, and the key matrix in the associated ODE can have eigenvalues with positive real parts, causing the parameters to grow without bound.

**Consequence.** A practitioner deploying COPDAC-Q on a new problem cannot rely on theoretical convergence guarantees. The algorithm might diverge — value estimates might explode or oscillate — and this would manifest as the actor receiving corrupted gradient signals and the policy degrading. The paper offers no diagnostic for detecting when divergence is occurring, no heuristic for mitigating it (e.g., target networks, experience replay, gradient clipping), and no characterization of which problem properties make divergence more or less likely. The success of COPDAC-Q on the tested benchmarks (mountain car, pendulum, puddle world, octopus arm) does not guarantee it will work on a new problem with different dynamics, reward structure, or function approximation.

**Evidence in the paper.** The paper does not measure or discuss any divergence episodes in its experiments. Figure 2 shows smooth, monotonic (or nearly monotonic) improvement for COPDAC-Q on all three continuous RL benchmarks, and Figure 3 shows all 10 octopus arm runs converging. This suggests that divergence did not occur *on these specific problems with these specific hyperparameters*, but provides no evidence about the prevalence of divergence across problem classes or hyperparameter settings. The paper does not report experiments where COPDAC-Q was deliberately stressed (e.g., by increasing the behaviour policy variance to create greater off-policy distribution mismatch) to test fragility.

**Mitigation status.** Partially addressed. The paper introduces COPDAC-GQ (Equations 23–27), which replaces standard Q-learning with gradient Q-learning (Maei et al., 2010) — a true gradient descent method on the mean-squared projected Bellman error (MSPBE) that is guaranteed to converge under linear function approximation:

> "the critic will converge to the parameters minimising the MSPBE"

However, COPDAC-GQ is **never experimentally evaluated**. The paper presents its update equations and notes the two-timescale convergence conditions (`α_θ ≪ α_w, α_u`), but does not compare it against COPDAC-Q on any benchmark, does not demonstrate that COPDAC-Q actually diverges on any problem where COPDAC-GQ succeeds, and does not measure the computational overhead of maintaining the additional parameter vector `u_t` (which doubles the number of critic parameters). For a practitioner, it is unclear whether COPDAC-GQ should be preferred as a default (for safety) or whether COPDAC-Q is sufficient for most problems (for simplicity). The convergence guarantee of COPDAC-GQ is also specific to *linear* function approximation — the octopus arm experiment uses neural networks for `V^v(s)`, which takes it outside the theory's scope.

---

### The Compatible Function Approximator Is a Linear Advantage Model — and the Theory Does Not Cover Non-Linear Critics

Theorem 3 establishes compatibility conditions for deterministic policy gradients that are satisfied by a critic of the specific structural form:

$$Q^w(s, a) = (a - μ_θ(s))^⊤ ∇_θ μ_θ(s)^⊤ w + V^v(s)$$

where the advantage term `A^w(s, a) = (a - μ_θ(s))^⊤ ∇_θ μ_θ(s)^⊤ w` is **linear in the action deviation**. The paper acknowledges the limitations of this form:

> "We note that a linear function approximator is not very useful for predicting action-values globally, since the action-value diverges to ±∞ for large actions."

And also:

> "a linear function approximator is sufficient to select the direction in which the actor should adjust its policy parameters."

**Consequence.** The theoretical guarantee of unbiased gradient estimation (Condition 2 of Theorem 3) applies only to critics where the **advantage** portion is linear in the policy Jacobian features `φ(s, a) = ∇_θ μ_θ(s)(a - μ_θ(s))`. If a practitioner uses a non-linear critic — for instance, a neural network that models `Q^w(s, a)` directly without enforcing the compatible structural form — the paper provides **no guarantee** that the resulting gradient `∇_θ μ_θ(s) ∇_a Q^w(s, a)|_{a=μ_θ(s)}` is unbiased or even an ascent direction. The paper's justification is pragmatic:

> "We note that a reasonable solution to the policy evaluation problem will find `Q^w(s, a) ≈ Q^μ(s, a)` and will therefore approximately (for smooth function approximators) satisfy `∇_a Q^w(s, a)|_{a=μ_θ(s)} ≈ ∇_a Q^μ(s, a)|_{a=μ_θ(s)}`."

This is a heuristic argument, not a proof. It assumes that value-function approximation error implies small gradient approximation error — which is not generally true. A function approximator can fit `Q^μ` well at the points that matter while having systematically wrong gradients at those same points. This is especially concerning when the critic is trained by TD learning (which minimizes value prediction error) rather than by gradient-matching (as Condition 2 requires).

**Evidence in the paper.** The paper does **not** test whether using a non-linear critic (without the compatible structure) degrades the deterministic policy gradient. All experiments use the compatible form: for the bandit and continuous RL tasks, the critic is exactly the compatible linear form; for the octopus arm, the advantage function `A^w(s, a)` is explicitly constrained to the compatible linear form `φ(s, a)^⊤ w`, with only the baseline `V^v(s)` allowed to be a non-linear neural network. This means we have no evidence for how COPDAC would perform with, say, a monolithic neural network critic that outputs `Q^w(s, a)` directly. The paper's theoretical contribution (Theorem 3) and its practical recommendations are aligned — use the compatible form — but they don't validate what happens when practitioners deviate from it, which they inevitably will when scaling to problems where linear advantages are insufficient.

**Mitigation status.** Not addressed. The paper presents the compatible form as the recommended approach and does not discuss or evaluate alternatives. The limitation is implicit in the theory: Theorem 3 gives sufficient conditions for unbiasedness, and any critic that does not satisfy them has no guarantee. But the paper does not characterize how large the bias might be in practice, whether it matters for convergence, or whether there exist other critic architectures that would satisfy weaker compatibility conditions.

---

### Difficulty Estimation Cost Is Not Accounted For — and There Is No Mechanism for Estimating It at Deployment

*Wait — this limitation is from the reference example about the LLM test-time compute paper, not this paper. Let me reconsider what limitations are actually relevant for Silver et al. (2014).*

Let me identify the genuine, high-consequence limitations of the deterministic policy gradient paper.

---

### The Deterministic Policy Gradient Requires the Action-Value Function to Be Differentiable — Which Excludes Many Policy Evaluation Methods

The deterministic policy gradient `∇_θ J(μ_θ) = E[∇_θ μ_θ(s) ∇_a Q^μ(s, a)|_{a=μ_θ(s)}]` requires computing `∇_a Q^μ(s, a)`, the gradient of the true action-value function with respect to the action. In practice, this is replaced by `∇_a Q^w(s, a)` from the learned critic. This means the critic **must be differentiable with respect to actions**, which rules out several powerful policy evaluation methods:

- **Tree-based or ensemble methods** (random forests, gradient-boosted trees) that produce piecewise-constant or non-smooth value estimates.
- **Tabular methods** with discretized action spaces, where `Q(s, a)` is defined only on a grid and gradients are undefined.
- **Gaussian process critics** (e.g., Engel et al., 2005 on the same octopus arm task), where computing action gradients may be possible but requires differentiable kernel functions and careful implementation.
- **Non-differentiable neural network components** (e.g., ReLU networks without careful smoothing, quantized networks for efficiency).

The paper does not discuss this constraint explicitly, but it is baked into every algorithm: the actor update requires `∇_a Q^w(s, a)`, which must be computable via automatic differentiation or manual derivation.

**Consequence.** The deterministic policy gradient framework is tied to **differentiable function approximators** for the critic. This is a restriction that stochastic policy gradients do not share: the stochastic policy gradient `E[∇_θ log π_θ(a|s) Q^π(s, a)]` requires only the *value* of `Q^π`, not its gradient. Any method that can estimate `Q^π(s, a)` — including Monte Carlo returns, TD learning with non-differentiable function approximators, or even human-provided value judgments — can be plugged into a stochastic policy gradient. The deterministic policy gradient forces the critic into a narrower class of function approximators, which may not be optimal for value prediction accuracy. If the best value predictor for a given problem is non-differentiable (e.g., a random forest that handles mixed discrete-continuous state spaces naturally), the deterministic policy gradient cannot use it without an additional differentiable approximation step.

**Evidence in the paper.** The paper uses only differentiable critics: linear regression (bandit), linear TD learning (continuous RL), and neural networks (octopus arm). It does not experiment with non-differentiable critics or discuss what happens when `∇_a Q^w` is unavailable or poorly behaved. The smoothness assumption is embedded in the MDP conditions A.1 (Appendix B), which "imply that `∇_θ μ_θ(s)` and `∇_a Q^μ(s, a)` exist," but the paper does not address the practical case where `Q^μ(s, a)` is non-smooth (e.g., in tasks with contact dynamics, discontinuous rewards, or discrete sub-components) even if the policy is differentiable.

**Mitigation status.** Not addressed. The paper implicitly assumes the critic will be a differentiable parametric function (neural network or linear model) and does not discuss alternative policy evaluation methods that would be incompatible. This is a reasonable scope limitation for a paper introducing a new gradient estimator, but it means practitioners working in domains where neural value functions are unreliable may find the deterministic policy gradient inapplicable.

---

### The Performance Advantage Over Stochastic Methods Is Not Demonstrated on Sequential RL Problems with High-Dimensional Actions

The paper's central empirical claim is that deterministic policy gradients significantly outperform stochastic policy gradients, and that this advantage grows with action dimensionality. This claim is well-supported by the **continuous bandit experiments** (Section 5.1, Figure 1), which show COPDAC-B outperforming SAC-B by orders of magnitude at 50 action dimensions. However, the bandit has no state dynamics, no temporal credit assignment, and a quadratic cost function where the compatible linear critic is exact.

The **continuous RL experiments** (Section 5.2, Figure 2) test COPDAC-Q against SAC and OffPAC on mountain car, pendulum, and puddle world. But these problems have **1–2 action dimensions** — precisely the regime where the paper's own theory predicts the smallest advantage for deterministic gradients. The results are consistent with this prediction: COPDAC-Q is slightly better, but the gap is modest (within a factor of 2–3× in convergence speed) and all methods eventually reach similar asymptotic performance. These experiments do not demonstrate the dramatic high-dimensional advantage that is the paper's main selling point.

The **octopus arm experiment** (Section 5.3, Figure 3) tests COPDAC-Q on a problem with **20 action dimensions** — a genuinely high-dimensional continuous control task. But it does **not** compare against stochastic policy gradient methods. The paper notes that previous work had simplified the octopus arm using macro-actions (Engel et al., 2005) or lower-dimensional variants (Heess et al., 2012), but it does not run SAC or OffPAC on the same 20-dimensional problem to quantify the advantage. The experiment demonstrates that COPDAC-Q *works* at this scale; it does not demonstrate that it works *better than stochastic alternatives*.

**Consequence.** We have no empirical evidence for the paper's headline claim — that deterministic policy gradients dramatically outperform stochastic policy gradients in high-dimensional *sequential decision-making* problems — at the scale that matters most. The bandit establishes the mechanism (gradient estimation efficiency), but the leap from a 50-dimensional bandit to a 20-dimensional octopus arm is not just a change in dimensionality; it adds state representation, temporal credit assignment, exploration over extended trajectories, and neural network optimization — all of which could interact with gradient quality in ways the bandit cannot reveal. The low-dimensional RL experiments show that COPDAC is competitive with stochastic methods, but they don't show that the advantage *scales* with action dimensionality in RL as it does in the bandit setting. A practitioner facing a 20-dimensional continuous control problem cannot look at this paper's results and know how much better (if at all) the deterministic policy gradient will be compared to a well-tuned stochastic off-policy actor-critic.

**Evidence in the paper.** Figure 2 shows the gap between COPDAC-Q and OffPAC is small (and sometimes within noise) for 1–2 action dimensions. Figure 3 shows COPDAC-Q working at 20 dimensions but without a stochastic baseline. The paper does not include a figure or table that plots the COPDAC vs. SAC/OffPAC performance gap as a function of action dimensionality for a sequential RL problem. This is the most significant gap between the paper's theoretical motivation and its empirical validation.

**Mitigation status.** Not addressed. The paper does not acknowledge this as a limitation or suggest future experiments to fill the gap. The theoretical connection between the bandit results and the RL results is left implicit — the reader is expected to infer that the gradient estimation advantage demonstrated in the bandit will carry over to RL, but no evidence is provided for this inference.

---

### The Natural Gradient Variant and Gradient Q-Learning Variant Are Not Experimentally Validated

The paper proposes several algorithm variants that are theoretically motivated but never experimentally tested:

- **Natural deterministic actor-critic (end of Section 4.3):** The actor update simplifies to `θ_{t+1} = θ_t + α_θ w_t` under compatible function approximation, because `w` is the natural gradient direction. This is a computationally attractive simplification — it avoids the Jacobian-vector product `∇_θ μ_θ(s) (∇_θ μ_θ(s)^⊤ w_t)` and has the geometric advantages of natural gradients (invariance to policy parameterization). However, the paper does not compare this variant against the standard COPDAC-Q update in any experiment. We do not know whether the natural gradient version converges faster, is more robust to hyperparameters, or performs equivalently.

- **COPDAC-GQ (Section 4.3, Equations 23–27):** This variant replaces standard Q-learning with gradient Q-learning to guarantee critic convergence under off-policy sampling with linear function approximation. It introduces an auxiliary parameter vector `u_t`, additional correction terms in the `w` and `v` updates, and an additional learning rate `α_u`. The paper provides convergence theory but **no experiments**. We do not know whether COPDAC-Q actually diverges on any of the tested benchmarks, whether COPDAC-GQ prevents such divergence, or what the computational overhead of the GQ critic is relative to the Q-learning critic.

**Consequence.** A practitioner choosing between algorithm variants has no empirical guidance. Should they use COPDAC-Q (simpler, empirically validated on the tested benchmarks, no convergence guarantees) or COPDAC-GQ (more complex, theoretically convergent, untested)? Should they use the natural gradient simplification (elegant, parameterization-invariant, untested) or the full actor update (used in all experiments, requires Jacobian computations)? The paper presents a menu of theoretical options but only validates one combination (COPDAC-Q with the full actor update) experimentally. The remaining variants are hypotheses — plausible given the theory, but unconfirmed.

**Evidence in the paper.** All continuous RL experiments (Section 5.2, Figure 2) and the octopus arm experiment (Section 5.3, Figure 3) use COPDAC-Q. The bandit experiment (Section 5.1, Figure 1) uses COPDAC-B, which is a specialized batch variant. COPDAC-GQ and the natural gradient variant appear only in the algorithm listings in Section 4.3 and are not referenced in the experimental sections.

**Mitigation status.** Not addressed. The paper does not explain why these variants were not tested, whether they were tested and performed poorly, or what the expected tradeoffs are. The theory sections motivate these variants as improvements (convergence guarantees for GQ, computational simplicity for natural gradients), but the experimental sections do not validate those motivations.

---

### No Experiments on Stochastic Environments with Discrete or Hybrid Action Spaces

The paper focuses exclusively on **continuous action spaces** (`A = ℝ^m`). This is where the deterministic policy gradient's advantage — eliminating the integral over actions — is most pronounced. However, many real-world control problems involve **hybrid action spaces** (e.g., a robot that selects from discrete behaviors while also controlling continuous joint torques) or **discrete action spaces with structure** (e.g., selecting from a large combinatorial set of discrete actions where the deterministic policy gradient could be applied by relaxing the discrete actions to continuous ones and rounding).

The paper does not discuss whether the deterministic policy gradient could be applied to discrete or hybrid action spaces, nor does it provide any experimental evidence. In a discrete action space, a deterministic policy `μ_θ(s)` would output an action index, and `∇_θ μ_θ(s)` would be undefined (the policy is not differentiable with respect to its output). One could use a continuous relaxation (e.g., the Gumbel-Softmax trick) to make the policy output differentiable, and then apply the deterministic policy gradient to the relaxed policy — but this is not explored.

**Consequence.** The paper's scope is narrower than the general "policy gradient for deterministic policies" framing might suggest. The method applies specifically to **continuous action spaces with differentiable policies and differentiable critics**. For problems with discrete actions — which include many classic RL benchmarks (Atari, board games, discrete control) — the deterministic policy gradient as presented is inapplicable without additional machinery. Practitioners in those domains gain nothing directly from this paper; the advances are specific to continuous control.

**Evidence in the paper.** Section 2.1 states: "In the remainder of the paper we suppose for simplicity that `A = ℝ^m`." All experiments use continuous action spaces. No experiments or discussion address discrete, hybrid, or structured action spaces.

**Mitigation status.** Not addressed. The paper does not discuss extensions to discrete or hybrid action spaces, does not suggest how one might apply the deterministic policy gradient in those settings, and does not flag this as a limitation for future work. This is a reasonable scope choice for a paper introducing a new gradient estimator — it is better to demonstrate effectiveness in the most favorable setting (continuous actions) before tackling harder extensions — but it means the paper's claims should not be interpreted as applying to RL problems generally.

---

### Summary Cross-Reference

- All limitations are grounded in the paper's own framing — quotes and citations are provided where the authors acknowledge constraints (on-policy exploration collapse, off-policy instability, linear advantage model scope).
- The most critical gap is the absence of a **high-dimensional sequential RL comparison** against stochastic methods (limitation 5). This gap sits between the bandit experiments (which prove the gradient estimation advantage in isolation) and the octopus arm experiment (which proves the method works at scale) and leaves unquantified the *magnitude* of the practical benefit on the problems practitioners actually care about.
- The linear-advantage constraint (limitation 3) is a fundamental architectural restriction that the theory requires but that practitioners may find limiting when scaling to complex value functions.
- The missing experimental validation of theoretically-motivated variants (limitation 6) is a common pattern in algorithm papers but leaves practical deployment decisions under-informed.
