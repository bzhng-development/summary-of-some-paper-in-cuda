# Proximal Policy Optimization Algorithms

**ArXiv:** [1707.06347](https://arxiv.org/abs/1707.06347)

## 🎯 Pitch

This paper introduces Proximal Policy Optimization (PPO), a novel family of reinforcement learning algorithms that stabilize policy gradient methods using a simple, first-order optimization technique. By enabling multiple epochs of minibatch updates with a clipped surrogate objective, PPO combines the reliability and sample efficiency of trust-region methods like TRPO with far greater simplicity and broad applicability—making cutting-edge reinforcement learning both easier to implement and more effective across challenging benchmarks.

---

## 1. Executive Summary
This paper introduces Proximal Policy Optimization (PPO), a family of reinforcement learning (RL) algorithms that make on-policy policy-gradient training both stable and simple. PPO provides most of the reliability of trust-region methods while using only standard first-order optimization, enabling multiple minibatch epochs on the same collected data and improving sample efficiency.

## 2. Context and Motivation
- Problem/gap addressed
  - On-policy policy-gradient methods are attractive because they directly optimize a stochastic policy, but they often become unstable when performing multiple gradient steps on the same batch of data; single-step updates are sample-inefficient. Section 2.1 explains that repeatedly optimizing the vanilla policy-gradient loss L_PG (Equation (2)) with the same trajectories “often leads to destructively large policy updates.”
  - Trust Region Policy Optimization (TRPO) stabilizes updates by constraining the Kullback–Leibler (KL) divergence between the old and new policies (Equations (3)–(4)), but it requires a more complex second-order optimization procedure and is awkward with shared architectures or noisy layers (Introduction).
- Why this matters
  - Reliable, sample-efficient on-policy optimization is crucial for domains where replay buffers are difficult or where on-policy exploration matters (e.g., robotics, simulated control, and games), and where wall-clock speed and code simplicity affect adoption.
- Prior approaches and shortcomings
  - Deep Q-learning: strong in discrete-action games but “fails on many simple problems” in continuous control (footnote 1 in Section 1).
  - Vanilla policy gradient (A2C/A3C-style): “poor data efficiency and robustness” (Section 1).
  - TRPO: reliable but “relatively complicated” and “not compatible” with some architectures (Section 1).
- Positioning
  - PPO aims to emulate TRPO’s reliable, monotonic-like improvements using only first-order methods. It proposes two variants:
    - A clipped surrogate objective (Section 3, Equation (7)).
    - An adaptive KL-penalized objective (Section 4, Equation (8)).
  - The clipped variant consistently performs best in the paper’s ablations (Table 1).

## 3. Technical Approach
The goal is to stabilize policy-gradient updates so we can do several minibatch epochs per batch of collected trajectories without destructive policy shifts.

- Preliminaries and notation
  - A stochastic policy `π_θ(a | s)` outputs a distribution over actions given state `s`.
  - The “advantage” `Â_t` (defined in Section 5) estimates how much better an action is than the policy’s average at state `s_t`. PPO typically uses Generalized Advantage Estimation (GAE; Section 5, Equations (11)–(12)).
  - The standard policy-gradient objective L_PG (Equation (2)) is the empirical expectation of `log π_θ(a_t|s_t) * Â_t`.

- Why naive multi-epoch optimization is unstable
  - Optimizing L_PG multiple times on the same data drifts the policy far from the one that generated those data. The gradient then becomes a poor estimator, and performance can collapse (Section 2.1).

- TRPO’s stabilization mechanism (background for contrast)
  - TRPO maximizes a “conservative policy iteration” surrogate (Equation (6)) but restricts the average KL divergence between old and new policies (Equation (4)). This restricts over-large updates but requires solving a constrained optimization using conjugate gradients and Fisher-vector products (Section 2.2).

- PPO’s clipped surrogate objective (key mechanism)
  - Define the probability ratio `r_t(θ) = π_θ(a_t|s_t) / π_{θ_old}(a_t|s_t)`. If the new policy increases the probability of the sampled action, `r_t > 1`; if it decreases, `r_t < 1`.
  - PPO’s main loss (Equation (7)):
    - `L_CLIP(θ) = E_t [ min( r_t(θ) * Â_t, clip(r_t(θ), 1-ε, 1+ε) * Â_t ) ]`
    - Intuition:
      - When an action has positive advantage (`Â_t > 0`), we want to increase its probability, but only up to a factor of `1+ε`. Beyond that, the clipped term flattens the objective (no extra reward for pushing probability further), preventing overly large updates.
      - When `Â_t < 0`, we want to reduce its probability, but only down to `1-ε`.
      - Taking the minimum with the unclipped term makes this a pessimistic bound—updates that would over-improve the surrogate are capped; updates that would worsen it are not forgiven. Figure 1 visualizes these two cases; Figure 2 shows that `L_CLIP` is a lower bound on the standard conservative policy iteration objective and peaks near a moderate KL divergence (~0.02 in the example).
      - `L_CLIP` equals the usual surrogate to first order near `r=1` (no update), so small updates behave like standard policy gradient; the difference appears only with larger steps.
  - Why this works: it approximates a trust region without an explicit constraint. The clipping keeps changes “proximal” to the data-generating policy.

- PPO with adaptive KL penalty (alternative)
  - Replace clipping by adding an average KL penalty to the surrogate (Equation (8)). After each update, measure the realized KL and adapt β (the penalty weight) to push the next update toward a target `d_targ` (Section 4). Empirically this works but is inferior to clipping in the paper’s benchmarks (Table 1).

- Full training objective with value function and entropy
  - Many policy-gradient methods train a value function `V_θ(s)` to compute advantages and reduce variance. PPO combines:
    - Policy loss: `L_CLIP_t(θ)`
    - Value loss: squared error `(V_θ(s_t) − V_target_t)^2`
    - Entropy bonus: `S[π_θ](s_t)` to encourage exploration
  - The combined objective (Equation (9)):
    - `E_t[ L_CLIP_t(θ) − c1 * L_VF_t(θ) + c2 * S[π_θ](s_t) ]`
    - `c1` and `c2` are coefficients.

- Advantage estimation and data collection
  - Data collection style: run each of `N` actors for `T` steps, then perform `K` epochs of minibatch SGD/Adam on the pooled `N*T` samples (Algorithm 1, Section 5).
  - Advantage estimator: truncated GAE (Equations (11)–(12)):
    - Temporal-difference residuals `δ_t = r_t + γ V(s_{t+1}) − V(s_t)`
    - `Â_t = Σ_{l=0}^{T−t−1} (γλ)^l δ_{t+l}`
    - `λ` controls bias–variance trade-off; `γ` is the discount factor.

- Implementation choices highlighted in experiments
  - Continuous control: Gaussian policy parameterized by a 2-layer MLP (64 units each, tanh), outputting means and learnable log-stds; no policy–value parameter sharing and no entropy bonus in that ablation (Section 6.1).
  - Optimization: Adam with multiple epochs per batch (Tables 3–5 list typical hyperparameters).
  - PPO is “first-order”: standard SGD/Adam without second-order methods or constrained solvers.

## 4. Key Insights and Innovations
- Clipped probability-ratio surrogate (fundamental)
  - What’s new: The `min(·,·)` with `clip` on the probability ratio (Equation (7)) acts like a soft trust region in objective space, not parameter space. This is different from TRPO’s hard KL constraint.
  - Why it matters: You can safely do multiple epochs/minibatch updates per batch of data with just standard optimizers, increasing sample efficiency and simplifying code. Figure 2 shows it behaves as a lower bound on the unconstrained surrogate and peaks near moderate KL.
- Pessimistic bounding to prevent “bad improvement” (conceptual)
  - The objective keeps the more conservative of the unclipped and clipped terms, ensuring we only ignore the probability-ratio change when it would artificially inflate the objective. This directional asymmetry is central to PPO’s robustness (Section 3).
- Simple alternative with adaptive KL penalty (incremental)
  - PPO also offers a penalty-based variant (Equation (8)) with an automatic β-tuning scheme (Section 4). While not as strong as clipping in the paper’s experiments (Table 1), it shows the design space and provides a baseline.
- Practical, scalable training loop (incremental but impactful)
  - The multi-epoch, minibatch training over on-policy data (Algorithm 1) plus GAE (Equations (11)–(12)) strikes a balance between stability and sample efficiency. This pipeline is straightforward to implement with automatic differentiation.

## 5. Experimental Analysis
- Evaluation methodology
  - Ablation on surrogate variants (Section 6.1):
    - 7 MuJoCo continuous control tasks: HalfCheetah, Hopper, InvertedDoublePendulum, InvertedPendulum, Reacher, Swimmer, Walker2d (footnote 2).
    - One million timesteps per task; 3 random seeds; architecture: 2×64 tanh MLP Gaussian policy; no policy–value sharing; no entropy bonus.
    - Scoring: normalize per environment so random policy = 0 and best across settings = 1; then average over 21 runs (Table 1).
  - Continuous control comparisons (Section 6.2, Figure 3):
    - Baselines: TRPO, CEM, vanilla PG with adaptive stepsize, A2C, A2C+Trust Region.
    - PPO uses the clipped objective with ε=0.2, and generic hyperparameters (Table 3).
    - Each environment trained for 1 million timesteps.
  - Humanoid showcase (Section 6.3, Figures 4–5; Table 4):
    - RoboschoolHumanoid, Flagrun, FlagrunHarder; up to 100M timesteps; 32–128 actors; linear annealing of action log std.
  - Atari benchmark (Section 6.4, Table 2; Appendix B Figure 6 and Table 6):
    - 49 games with the standard convolutional policy used in A3C/A2C.
    - Metrics: (1) Average episode reward over entire training; (2) Average reward over last 100 episodes.
    - Baselines: A2C and ACER with tuned hyperparameters.
    - PPO hyperparameters summarized in Table 5 (e.g., T=128, 3 epochs, minibatch size 32×8, entropy coeff 0.01).

- Key quantitative results
  - Ablation of surrogate objectives (Table 1):
    - “No clipping or penalty” averages −0.39 normalized score (catastrophic on HalfCheetah), showing instability without a constraint.
    - Clipping performs best; ε=0.2 yields 0.82 normalized average, higher than ε=0.1 (0.76) and ε=0.3 (0.70).
    - Adaptive KL penalty is weaker: best dtarg=0.01 yields 0.74; fixed penalties range 0.62–0.72.
    - Quote: > “Note that the score is negative for the setting without clipping or penalties… worse than the initial random policy.” (Section 6.1)
  - Continuous control comparisons (Figure 3):
    - PPO is at or near the top across seven MuJoCo tasks for the same 1M steps. For example, on Hopper-v1 and Walker2d-v1 PPO reaches higher final rewards faster than TRPO and other baselines (visual inspection of Figure 3).
  - Humanoid showcase (Figure 4):
    - PPO learns challenging 3D behaviors across 50–100M timesteps, including turning and recovering from disturbances (Figure 5 shows frames).
  - Atari results (Table 2, Appendix Table 6):
    - Wins by metric (Table 2):
      - Average over entire training: PPO “wins” 30 games vs ACER’s 18 and A2C’s 1.
      - Last-100 episodes: ACER 28 wins, PPO 19, A2C 1, 1 tie—ACER often edges out final performance but is more complex.
    - Concrete examples from Table 6 (final 100 episodes):
      - Assault: PPO 4971.9 vs A2C 1562.9; ACER 4653.8.
      - Atlantis: PPO 2,311,815 vs A2C 729,265; ACER 1,841,376.
      - Boxing: PPO 94.6 vs A2C 17.7; ACER 98.9 (close).
      - Some cases favor ACER, e.g., UpNDown: ACER 145,051 vs PPO 95,445.

- Do the experiments support the claims?
  - Stability and sample efficiency: The ablation (Table 1) isolates clipping as the essential ingredient; without it performance collapses. Multiple epochs per batch become safe with clipping.
  - Simplicity with strong performance: PPO matches or exceeds TRPO on many continuous-control tasks (Figure 3) while using only first-order optimization. On Atari, PPO is competitive with ACER and substantially better than A2C in sample efficiency (Table 2).
  - Generality: Demonstrations span continuous control and Atari with a single principle and minor hyperparameter changes (Tables 3–5).
- Ablations and robustness
  - Explicit ablation of objective variants (Section 6.1) is a strength. They also swept ε and KL targets/penalties, showing ε≈0.2 as a good default.
  - Adaptive KL penalty variant works but is consistently worse than clipping, supporting the design choice.
- Mixed results and trade-offs
  - On Atari final performance (last 100 episodes), ACER often beats PPO (Table 2), while PPO dominates in average-over-training (faster learning). This suggests a speed–asymptote trade-off depending on domain and algorithm complexity.

## 6. Limitations and Trade-offs
- On-policy requirement and sample cost
  - PPO is an on-policy method—each update requires fresh interaction data generated by the current policy (Algorithm 1). This limits raw sample efficiency versus off-policy replay-based methods in some settings.
- No formal monotonic improvement guarantee
  - Unlike TRPO’s theoretical lower-bound motivation for a KL-constrained update, the clipped surrogate is a heuristic lower bound on the CPI surrogate (Figure 2 commentary), not a strict performance guarantee. Stability is empirical rather than proven.
- Hyperparameter sensitivity (moderate)
  - While ε=0.2 works well in Table 1, performance degrades for ε=0.3 and slightly for ε=0.1. Some tuning is still needed across tasks (Tables 3–5 provide starting points).
- Value function and entropy co-training
  - The approach relies on a well-tuned value function and sometimes an entropy bonus (Equation (9)), which add their own hyperparameters (`c1`, `c2`) and can affect stability.
- Final-performance trade-off vs ACER on Atari
  - PPO tends to learn faster but sometimes attains slightly worse asymptotic performance than the more complex ACER (Table 2 and Table 6), indicating that clipping’s conservative updates may occasionally limit late-stage improvements.
- Architectural and domain boundaries
  - The paper evaluates standard continuous control (MuJoCo/Roboschool) and Atari. It does not explore partial observability requiring recurrent policies in depth, multi-agent interactions, or large-scale real-world robotics deployments.

## 7. Implications and Future Directions
- Impact on the field
  - PPO provides a robust, easy-to-implement default for on-policy RL. By replacing TRPO’s second-order machinery with a simple clipped objective (Equation (7)), it makes stable policy optimization accessible and widely usable. This balance of simplicity and performance contributed to PPO becoming a standard baseline in RL.
- Practical applications
  - Simulated and real robotics (locomotion, manipulation) where reliable on-policy updates and safe multiple epochs matter.
  - Game-playing and interactive environments where policy stability is critical and code simplicity accelerates iteration.
- Suggested follow-up research
  - Theory:
    - Tighter performance bounds for clipped objectives; conditions for monotonic improvement.
    - Adaptive or state-dependent clipping thresholds tied to measured KL (combining insights from Section 4 and Section 3).
  - Algorithms:
    - Hybrid on-/off-policy PPO variants that leverage replay without destabilizing the ratio clipping.
    - Trust-region-aware early stopping based on measured KL per minibatch/epoch.
    - Better advantage/value estimation to reduce variance and bias in Equations (11)–(12).
  - Systems:
    - Scaling to large actor counts and distributed settings; automatic hyperparameter tuning (e.g., for ε, `c1`, `c2`, and learning rates) based on diagnostics like the realized KL and entropy.
  - Domains:
    - Robustness under partial observability with recurrent networks; multi-task and meta-RL; safety-constrained RL where conservative updates are desirable.

In short, the paper delivers a simple but powerful modification—the clipped probability-ratio surrogate—that enables safe multi-epoch on-policy training. The ablations (Table 1), control benchmarks (Figure 3), humanoid showcases (Figures 4–5), and Atari results (Table 2; Table 6) collectively demonstrate that PPO achieves a strong balance of stability, sample efficiency, and ease of implementation.
