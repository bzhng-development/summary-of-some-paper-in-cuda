# Trust Region Policy Optimization

**ArXiv:** [1502.05477](https://arxiv.org/abs/1502.05477)

## 🎯 Pitch

Trust Region Policy Optimization (TRPO) introduces a practical and theoretically-grounded algorithm for optimizing reinforcement learning policies by constraining each update to stay within a 'trust region,' measured by KL divergence. This innovation delivers robust, monotonic policy improvement even with complex, high-dimensional policies like deep neural networks, addressing stability and scaling challenges that hindered previous approaches. By unifying policy iteration and policy gradient perspectives, TRPO enables reliable training of sophisticated agents for tasks ranging from robotic locomotion to Atari games, marking a significant step toward stable and scalable deep reinforcement learning.

---

## 1. Executive Summary
Trust Region Policy Optimization (TRPO) introduces a principled way to update reinforcement learning policies that yields reliable, near-monotonic improvement without delicate step-size tuning. It does this by maximizing a local surrogate of the true return while constraining how much the policy can change—measured by Kullback–Leibler (KL) divergence—at each iteration, and provides a theoretical improvement bound that applies to general stochastic policies.

## 2. Context and Motivation
- Problem addressed
  - How to optimize policies in reinforcement learning (RL) so that performance consistently improves rather than oscillates or collapses, especially when using large, nonlinear function approximators (e.g., neural networks).
  - Prior robust theoretical guarantees (e.g., conservative policy iteration) did not directly apply to common, general policy classes and imposed impractical mixture-policy updates.

- Importance
  - Practical: Stable learning is essential for high-dimensional control (robotic locomotion) and perception-control tasks (Atari from pixels).
  - Theoretical: Provides a general improvement guarantee beyond special-case policy classes; unifies perspectives across policy iteration and policy gradient methods.

- Prior approaches and shortcomings (Section 1)
  - Derivative-free methods (CEM, CMA): often work well but scale poorly with parameters and use samples inefficiently.
  - Policy gradient and natural policy gradient: sensitive to step-size/penalty choices, risk making steps that degrade performance.
  - Conservative policy iteration (CPI): has a monotonic improvement bound but only for mixture updates (Eq. 5), which are impractical with modern parameterizations.

- Positioning
  - TRPO generalizes CPI’s guarantee from mixture policies to any stochastic policy by bounding total variation / KL divergence between old and new policies (Section 3; Theorem 1, Eq. 8; KL version Eq. 9).
  - Builds a practical algorithm that enforces an average KL trust region (Eq. 12), two sampling schemes for estimating the surrogate objective (Section 5; Fig. 1), and an efficient solver using conjugate gradient and Fisher-vector products (Appendix C).

## 3. Technical Approach
This section explains what TRPO optimizes, how it approximates the true objective, how it enforces safe step sizes, and how it is implemented.

- Core RL quantities (Section 2)
  - `policy` π(a|s): probability of action a in state s.
  - `expected discounted return` η(π): expected sum of discounted rewards.
  - `Qπ(s,a)`, `Vπ(s)`, `Aπ(s,a) = Qπ(s,a) − Vπ(s)`: advantage is the action’s benefit over the state’s average under π.
  - `discounted visitation frequency` ρπ(s): how often states are visited under π, weighted by γ^t.

- A key identity that reframes policy improvement (Eq. 1–2)
  - The performance difference between a new policy π̃ and the current policy π can be written in terms of the current policy’s advantage:
    > η(π̃) = η(π) + Σ_s ρ_{π̃}(s) Σ_a π̃(a|s) A_π(s,a)  (Eq. 2)
  - This is exact but hard to optimize because ρ_{π̃} depends on the new policy.

- Surrogate objective: optimize a local proxy that is first-order accurate (Section 2)
  - Define:
    > L_π(π̃) = η(π) + Σ_s ρ_π(s) Σ_a π̃(a|s) A_π(s,a)  (Eq. 3)
  - Key property:
    > L_π matches η at the current parameters and has the same gradient (Eq. 4).
  - Intuition: use old state visitation frequencies ρ_π to sidestep how π̃ impacts future state distribution, making optimization feasible.

- From conservative policy iteration to a general bound (Section 3)
  - CPI updates via a mixture policy and proves an improvement bound (Eq. 5–6).
  - TRPO’s theoretical advance: a performance lower bound that applies to any stochastic policy pair by using a divergence constraint.
    - Using total variation (TV) distance:
      > η(π_new) ≥ L_π_old(π_new) − (4 ε γ / (1−γ)^2) α^2, where α = max_s TV(π_old(·|s), π_new(·|s)) and ε = max_{s,a} |A_π(s,a)|  (Theorem 1, Eq. 8).
    - Using KL via TV–KL relation:
      > η(π̃) ≥ L_π(π̃) − C · D^max_KL(π, π̃), with C = 4 ε γ / (1−γ)^2  (Eq. 9).

- Surrogate maximization with a trust region (Sections 3–4)
  - Idealized scheme (Algorithm 1): at each iteration i, compute all advantages A_{π_i} exactly and maximize L_{π_i}(π) − C·max-state KL. This MM-style procedure ensures η(π_i) is non-decreasing (Eq. 10).
  - Practical TRPO replaces the penalty with a constraint and the max-state KL with an average KL for tractability:
    > maximize_θ L_{θ_old}(θ) subject to E_{s∼ρ_{θ_old}}[KL(π_{θ_old}(·|s) || π_θ(·|s))] ≤ δ  (Eq. 12).
  - Why constraint over penalty: fixed penalties make steps either too small or too large; a hard trust region δ yields robust progress without manual tuning (Section 4).

- Estimating the objective and constraint from samples (Section 5; Fig. 1)
  - Rewrite the objective using importance sampling:
    > maximize_θ E_{s∼ρ_{θ_old}, a∼q} [ (π_θ(a|s)/q(a|s)) · Q_{θ_old}(s,a) ]   (Eq. 14).
  - Two sampling schemes:
    - Single path (Section 5.1; Fig. 1 left): roll out π_{θ_old}, compute Monte Carlo returns from each state-action along trajectories; set q = π_{θ_old}.
      - Pros: works in real systems; no need to reset to arbitrary states.
      - Cons: higher variance estimates of advantages.
    - Vine (Section 5.2; Fig. 1 right): collect “trunk” trajectories, pick a “rollout set” of states, then from each state sample several actions and perform short rollouts to estimate Q; use common random numbers (CRN) to reduce variance; optionally use a self-normalized estimator (Eq. 16) in large action spaces.
      - Pros: lower-variance advantage estimates.
      - Cons: needs the ability to reset the simulator to stored states.

- Efficient constrained optimizer (Section 6; Appendix C)
  - Search direction: solve A x = g approximately with conjugate gradient, where g is the gradient of the surrogate and A is the Fisher Information Matrix (FIM) of the policy, computed as the Hessian of KL w.r.t. parameters (Appendix C).
    - Using the analytical FIM (Hessian of KL) integrates over actions, avoids dependence on sampled actions, and reduces memory footprint (Section 6).
  - Step size: scale the direction to satisfy the quadratic model of the KL trust region (β = sqrt(2δ / s^T A s)), then do a backtracking line search to ensure both surrogate improvement and KL ≤ δ with the true (nonlinear) functions.
  - Fisher-vector products (Appendix C.1): compute J^T M J y efficiently without forming matrices, where J is the Jacobian of the policy’s distribution parameters and M is the KL’s local metric for those parameters; subsample states for the FIM to reduce cost.

- Relation to existing updates (Section 7)
  - Natural policy gradient arises by linearizing L and quadratically approximating the KL constraint (Eq. 17).
  - Standard policy gradient with L2 trust region uses an L2 constraint instead of KL (Eq. 18).
  - Policy iteration corresponds to unconstrained maximization of L (Section 7).
  - TRPO enforces the constraint explicitly each iteration, which empirically improves stability on large problems.

## 4. Key Insights and Innovations
- Monotonic improvement bound for general stochastic policies
  - What’s new: Theorem 1 (Eq. 8) bounds η(π_new) − η(π_old) using total variation (and hence KL) between policies, not limited to mixture policies like CPI.
  - Why it matters: Justifies safe, non-trivial step sizes for any policy class; underpins the trust region idea.

- Trust-region formulation with an average KL constraint
  - What’s new: Replace a global penalty or max-state constraint with an expected KL constraint (Eq. 12), which is practical to estimate and optimize.
  - Why it matters: Delivers robust, largely step-size–free learning across disparate tasks with a single δ (Section 8 uses δ=0.01 everywhere).

- Two practical sampling regimes with variance control
  - What’s new: The “vine” method (Section 5.2; Fig. 1 right) constructs low-variance advantage estimates with CRN; the “single path” method (Section 5.1) keeps real-world feasibility.
  - Why it matters: Offers a controllable variance–cost trade-off depending on whether state resets are possible.

- Efficient second-order-like optimization via Fisher-vector products
  - What’s new: Use the analytical FIM (KL Hessian) and conjugate gradient to compute large, stable parameter updates without forming Hessians (Appendix C, C.1).
  - Why it matters: Enables scaling TRPO to tens of thousands of parameters (Section 8) while keeping each update only slightly more expensive than a gradient.

- Unifying perspective on policy optimization
  - What’s new: Shows natural gradient, L2-regularized gradient steps, and policy iteration are special cases/limits of the same trust-region view (Section 7; Eqs. 17–18).
  - Why it matters: Clarifies how to choose constraints (KL vs L2) and step computations, and why explicit constraints outperform penalties in practice.

## 5. Experimental Analysis
- Evaluation setup
  - Continuous control (Section 8.1; Fig. 2)
    - Environments: Swimmer, Hopper, Walker in MuJoCo; state/action dimensions and rewards described in Section 8.1.
    - Policies: small fully connected networks (Fig. 3 top).
    - Hyperparameters: δ=0.01 throughout; sampling budgets and network sizes in Table 2.
    - Baselines: TRPO (single path and vine), Natural Gradient (Eq. 17), CEM, CMA, Empirical FIM variant of TRPO, and Max KL (statewise max KL constraint) on Cartpole (Section 8.1).
  - Atari from pixels (Section 8.2; Fig. 3 bottom)
    - Games: 7 titles matching ALE benchmarks (Bellemare et al., 2013).
    - Policy: 2 conv layers (16 filters, stride 2) + 1 FC layer (20 units), ~33.5k parameters (Fig. 3 bottom).
    - Budgets: 500 policy iterations; single path 100k steps/iter; vine 400k steps/iter; ≈30 hours on 16 cores (Table 3).
    - Comparisons: Random, Human, Deep Q-learning (DQN), and UCC-I (offline MCTS + supervised learning) via Table 1.

- Main results
  - Continuous control (Fig. 4)
    - TRPO (both single path and vine) “solved all problems” and achieved the best final performance (Section 8.1).
    - Natural Gradient did well on simpler tasks but failed to produce forward-progress gaits on Hopper and Walker (Fig. 4 bottom panels).
    - Empirical FIM variant was similar to analytic FIM (Section 6 notes “rate of improvement … similar”); Max KL learned but slower due to stricter constraint (Cartpole; Fig. 4 top-left).
    - Derivative-free CEM/CMA underperformed as problem size grew, consistent with poor sample scaling (Section 8.1).
    - Qualitative: Learned robust gaits with generic networks and simple rewards, without hand-engineered policy structures (Section 8.1; video link referenced there).
  - Atari (Table 1; Fig. 5)
    - TRPO obtains reasonable, competitive scores without task-specific design; sometimes close to DQN/UCC-I, sometimes lower.
    - Examples from Table 1:
      > Beam Rider: TRPO-single 1425; DQN 4092; UCC-I 5702  
      > Breakout: TRPO-vine 34.2; DQN 168; UCC-I 380  
      > Enduro: TRPO-single 534.6; DQN 470; UCC-I 741  
      > Pong: TRPO 20.9 (both), near DQN 20.0  
      > Q*bert: TRPO-vine 7732.5; DQN 1952; UCC-I 20025  
      > Seaquest: TRPO-single 1908.6; DQN 1705; UCC-I 2995  
      > Space Invaders: TRPO-single 568.4; DQN 581; UCC-I 692
    - Learning curves (Fig. 5) show steady cost reduction (cost = −reward) over iterations for both single path and vine.

- Do the experiments support the claims?
  - Stability and monotonic tendencies: Learning curves for locomotion and Atari trend upward (or cost downward) with few regressions, consistent with the trust region’s stabilizing effect (Figs. 4–5). The paper explicitly notes monotonic improvement is not guaranteed by the practical approximations but is typically observed (Abstract; Section 6 summary).
  - Robustness across domains: The same δ and update procedure work on both control (states) and vision (pixels), indicating low sensitivity to tuning (Sections 8.1–8.2).
  - Ablations / variants:
    - Natural Gradient vs TRPO: fixed penalty vs fixed constraint—TRPO wins on hard control tasks (Fig. 4), underscoring the importance of enforcing a KL trust region (Section 7 discussion).
    - Empirical vs analytic FIM: similar improvement rate (Section 6) though analytic FIM has computational advantages.
    - Max KL (per-state) vs average KL: Max KL learns slightly slower (Cartpole), suggesting the average KL used by TRPO is a good practical proxy (Section 8.1).
  - Limitations visible in results:
    - TRPO does not surpass specialist methods everywhere on Atari (Table 1), but it is consistently “reasonable” without task-specific tricks (Section 8.2).
    - Vine often improves sample estimates but costs more wall-clock time (Tables 2–3).

## 6. Limitations and Trade-offs
- Theoretical–practical gap
  - The strict improvement guarantee (Algorithm 1; Eq. 9–10) requires exact advantages and a max-state KL penalty; the practical TRPO uses sampled estimates and an average KL constraint (Eq. 12). The resulting near-monotonic improvements are empirical, not formal guarantees (Abstract; Section 6).

- Sample efficiency and on-policy nature
  - Uses fresh on-policy trajectories each iteration with large budgets (e.g., Hopper/Walker: 1M simulator steps per iteration for 200 iterations; Table 2). This is costly compared to off-policy or replay-based methods.

- Vine method practicality
  - Requires the ability to reset to arbitrary states for branching rollouts, typically only possible in simulators (Section 5.2; Fig. 1), limiting direct use on physical robots.

- Hyperparameter reliance on δ
  - While δ is robustly set to 0.01 across tasks (Section 8), performance still depends on an appropriate trust region size; too small slows progress, too large risks constraint violations if line search fails.

- Advantage estimation variance
  - Single-path Monte Carlo returns can be high variance; while CRN and vine reduce variance (Section 5.2), there is no learned value baseline or generalized advantage estimator in this paper, which later works show can improve sample efficiency.

- Computational overhead
  - Each update requires several Fisher-vector products and a line search (Appendix C), which, while efficient relative to full Hessians, is still costlier than first-order methods.

- Scope of tasks
  - Experiments focus on continuous control in simulation and Atari with feedforward policies; recurrent policies and severe partial observability are not directly explored (Section 9 suggests this as future work).

## 7. Implications and Future Directions
- Field impact
  - TRPO establishes trust regions with KL constraints as a practical and theoretically grounded way to stabilize policy optimization. This reframing influences subsequent algorithms that simplify or extend the core idea (e.g., later proximal methods using clipped objectives).

- Research directions suggested in the paper (Section 9)
  - Integrate model learning to reduce sample complexity, making the approach more viable for real-world robotics where data are expensive.
  - Extend to richer policies, including recurrent architectures, to unify state estimation and control in partially observed settings.
  - Apply to vision-based robotics to jointly learn perception and control, leveraging TRPO’s stability with large neural policies.

- Practical applications
  - Robotics: learning locomotion and manipulation policies with minimal hand-engineering (Section 8.1).
  - Games and simulation: training policies directly from pixels (Section 8.2).
  - Any domain where safe, consistent improvement of a policy is needed and large neural networks are used.

- Takeaway for practitioners
  - Use a surrogate objective with an average KL trust region.
  - Estimate advantages carefully (consider vine if state resets are available; otherwise single path).
  - Compute the step with conjugate gradient and Fisher-vector products; enforce the KL constraint with a line search.
  - Start with δ≈0.01 as a robust default (Sections 8.1–8.2).

> In summary, TRPO contributes both a general policy improvement bound (Theorem 1; Eqs. 8–9) and a practical, scalable algorithm (Eq. 12; Appendix C) that delivers robust learning across diverse tasks with large policies. The experiments (Figs. 4–5; Tables 1–3) demonstrate strong performance on challenging control problems and competitive results on Atari, with a single hyperparameter controlling the trust region across settings.
