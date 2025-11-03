# Proximal Policy Optimization Algorithms

**ArXiv:** [1707.06347](https://arxiv.org/abs/1707.06347)

## 🎯 Pitch

This paper introduces Proximal Policy Optimization (PPO), a novel family of reinforcement learning algorithms that combine the stability of trust-region methods with the simplicity and efficiency of standard policy gradients. By using a carefully designed 'clipped' surrogate objective, PPO enables robust and data-efficient policy improvement with multiple gradient steps per batch—without the complexity of second-order optimization. PPO’s ease of implementation, scalability, and strong empirical performance make it a transformative advance for deep reinforcement learning in both continuous and discrete control problems.

---

## 1. Executive Summary
This paper introduces Proximal Policy Optimization (PPO), a family of reinforcement-learning (RL) algorithms that make policy-gradient updates stable while remaining simple and scalable. The core idea is a “clipped” surrogate objective that safely reuses the same batch of experience for multiple optimization epochs, matching the robustness of trust-region methods with first‑order optimization and minimal code changes.

## 2. Context and Motivation
- Problem addressed
  - Policy-gradient RL methods improve a policy (a probability distribution over actions) using sampled trajectories. Standard implementations take only one gradient step per batch because additional steps can change the policy too much relative to the data that was collected, causing divergence or “destructive” updates (Section 2.1; see Eq. (2) and the note right after it).
  - Trust Region Policy Optimization (TRPO) stabilizes learning by enforcing a small step measured by Kullback–Leibler (KL) divergence, but it is algorithmically complex (requires second‑order optimization via conjugate gradient), and it’s less flexible with modern architectures that share parameters or include stochastic layers (Section 1).
- Why it matters
  - Stable, data‑efficient policy optimization is central to RL in continuous control (robotics) and large-scale discrete control (Atari). Instability leads to wasted experience and brittle training; complexity hinders adoption and adaptation to new architectures.
- Shortcomings of prior approaches (Section 1)
  - Deep Q‑learning struggles on continuous control benchmarks.
  - Vanilla policy gradients are data‑inefficient and fragile when reusing data for multiple epochs.
  - TRPO is robust but complex and less compatible with common neural‑network practices (e.g., dropout, parameter sharing).
- Positioning of this work
  - PPO aims to keep TRPO’s reliability while using only first‑order (stochastic gradient) updates. It does this by replacing TRPO’s hard constraint with a simple, differentiable objective that implicitly discourages too‑large policy updates (Sections 3–5).

## 3. Technical Approach
At a high level, PPO alternates between:
1) collecting trajectories with the current policy, and
2) optimizing a specially designed objective on that fixed batch for multiple epochs using minibatches.

Key pieces, step by step:

- Standard policy-gradient setup (Section 2.1)
  - Let `πθ(a|s)` be the stochastic policy. The goal is to maximize expected return using gradient ascent.
  - A classic gradient estimator (Eq. (1)) is
    - `ĝ = Ê_t[ ∇θ log πθ(a_t|s_t) · Â_t ]`,
    - where `Â_t` is an estimator of the advantage function (how much action `a_t` was better than average at state `s_t`).
  - Implementations construct a loss whose gradient equals this estimator (Eq. (2)):
    - `L_PG(θ) = Ê_t[ log πθ(a_t|s_t) · Â_t ]`.
  - Problem: performing many optimization steps on `L_PG` using the same data often causes “destructively large” updates (Section 2.1).

- Trust-region idea (Section 2.2)
  - TRPO maximizes a surrogate objective while constraining the average KL divergence from the old policy `πθ_old` to the new `πθ` (Eqs. (3)–(4)).
  - Theory also motivates a penalized form (Eq. (5)), but choosing a fixed penalty coefficient `β` that works throughout training is hard.

- PPO’s clipped surrogate objective (Section 3)
  - Define the probability ratio for the sampled action: `r_t(θ) = πθ(a_t|s_t) / πθ_old(a_t|s_t)`.
  - Conservative policy iteration objective (Eq. (6)): `L_CPI(θ) = Ê_t[r_t(θ) · Â_t]`—this improves expected return if the step is small.
  - PPO modifies this with a clip and a min (Eq. (7)):
    - `L_CLIP(θ) = Ê_t[ min( r_t(θ) · Â_t, clip(r_t(θ), 1−ε, 1+ε) · Â_t ) ]`, with `ε` typically 0.2.
  - Intuition:
    - When `Â_t > 0` (the action was better than expected), increasing `r_t` above `1+ε` would over‑encourage that action; the clip flattens the incentive beyond `1+ε`.
    - When `Â_t < 0`, decreasing `r_t` below `1−ε` would overly penalize the action; clipping limits that too.
    - Taking the min between the unclipped and clipped terms makes the objective a pessimistic lower bound on the unclipped one: it only “pays attention” to the clipping when the update would otherwise improve the objective too much (keeping steps conservative).
  - Figure 1 visualizes a single-term behavior: the `L_CLIP` curve flattens at `1−ε` or `1+ε` depending on the sign of the advantage.

  - Empirical geometry (Figure 2)
    - Along the actual PPO update direction on Hopper‑v1, `L_CLIP` sits below `L_CPI`, peaking at a KL of ≈0.02 from the old policy. This shows how clipping acts like a soft trust region in practice.

- Alternative (or complement): KL penalty with adaptive coefficient (Section 4)
  - Penalized objective (Eq. (8)):
    - `L_KLPEN(θ) = Ê_t[ r_t(θ) · Â_t − β · KL(πθ_old(·|s_t), πθ(·|s_t)) ]`.
  - After each policy update, compute the realized mean KL `d`. Adjust `β`:
    - If `d < d_targ/1.5`, halve `β`; if `d > 1.5·d_targ`, double `β`.
  - This maintains a target KL per update. In experiments, this variant underperforms clipping, but it is a meaningful baseline (Section 4).

- Full training objective (policy + value + entropy; Section 5)
  - The algorithm usually learns a value function `Vθ(s)` to reduce variance in `Â_t` and may add an entropy bonus to encourage exploration.
  - Combined loss (Eq. (9)):
    - `L_total(θ) = Ê_t[ L_CLIP_t(θ) − c1 · L_VF_t(θ) + c2 · S[πθ](s_t) ]`,
    - Where `L_VF_t(θ) = (Vθ(s_t) − V_targ_t)^2`, and `S[πθ]` is policy entropy. Coefficients `c1, c2` weight value loss and entropy bonus.

- Advantage estimation on fixed-length segments (Section 5)
  - For segments of length `T`, PPO uses truncated Generalized Advantage Estimation (GAE): `Â_t = δ_t + (γλ) δ_{t+1} + … + (γλ)^{T−t+1} δ_{T−1}` (Eq. (11)),
    - With temporal-difference residuals `δ_t = r_t + γ V(s_{t+1}) − V(s_t)` (Eq. (12)).
  - Special case `λ=1` yields the finite-horizon estimator in Eq. (10).

- Training loop (Algorithm 1, Section 5)
  - Parallel experience collection: `N` actors each run the current policy for `T` steps.
  - Compute `Â_t` for all `NT` transitions.
  - Optimize `L_total` (or `L_KLPEN`) with minibatch SGD/Adam for `K` epochs on this fixed batch.
  - Replace `θ_old ← θ` and repeat.

- Design choice rationale
  - Clipping vs. explicit constraint: clipping is easy to implement with standard autodiff, needs no conjugate-gradient solver, and works with any architecture (Section 1, Sections 3–5).
  - Multiple epochs on the same data are safe because the objective explicitly discourages policy drift away from the data-generating policy beyond a small, controlled range (Eq. (7), Figure 1).

## 4. Key Insights and Innovations
- Clipped probability-ratio surrogate (fundamental innovation)
  - What’s new: the min of the unclipped importance ratio term and a clipped version anchored at 1 (Eq. (7)).
  - Why it matters: it creates a simple, differentiable lower bound on performance improvement that stabilizes multi‑epoch updates. This mimics a trust region without second‑order optimization (Figures 1–2), enabling both robustness and simplicity.

- Reliable multi‑epoch minibatch updates from a single batch (practical breakthrough)
  - Prior vanilla policy gradients avoid multiple epochs because the policy drifts from the data (Section 2.1). The clipped loss makes repeated passes not only possible but effective, improving sample efficiency.

- Flexible, first‑order implementation
  - Unlike TRPO’s hard constraint (Eqs. (3)–(4)), PPO uses standard SGD/Adam and works with parameter sharing and stochastic modules (Section 1). This lowers the barrier to use across architectures and toolchains.

- Empirical validation across domains with ablations (incremental but impactful)
  - The paper compares clipping, fixed/adaptive KL penalties, and no regularization (Section 6.1; Table 1), demonstrating the advantage of the clipped objective—an important empirical insight guiding practical use.

## 5. Experimental Analysis
- Evaluation setup
  - Ablations on continuous control (Section 6.1)
    - 7 MuJoCo Gym tasks: HalfCheetah, Hopper, InvertedDoublePendulum, InvertedPendulum, Reacher, Swimmer, Walker2d.
    - One million timesteps per task; 3 random seeds each.
    - Policy: MLP with two 64‑unit tanh layers; Gaussian actions with learned std; no policy–value parameter sharing; no entropy bonus (Section 6.1).
    - PPO hyperparameters listed in Table 3 (e.g., horizon `T=2048`, 10 epochs, minibatch size 64, Adam 3e−4, `γ=0.99`, `λ=0.95`).
    - Scoring: last 100‑episode average reward, normalized per environment so random = 0 and best achieved = 1; then averaged across 21 runs.
  - Baseline comparisons in continuous control (Section 6.2; Figure 3)
    - Algorithms: TRPO, Cross‑Entropy Method (CEM), vanilla policy gradient with adaptive stepsize, A2C, and A2C with a trust region.
    - Same 7 tasks; one million timesteps; PPO uses `ε=0.2`.
  - High‑dimensional humanoid showcase (Section 6.3; Figures 4–5, Table 4)
    - RoboschoolHumanoid, RoboschoolHumanoidFlagrun, and FlagrunHarder (with disturbances and get‑up).
    - Long training (50M–100M timesteps); many actors (32–128).
  - Atari benchmark (Section 6.4; Figure 6, Table 2, Table 5, Table 6)
    - 49 games, same convolutional policy as A3C/A2C; 40M frames (10M timesteps).
    - Baselines: A2C and ACER (both tuned). PPO hyperparams in Table 5 (e.g., `T=128`, 3 epochs, minibatch size `32×8`, entropy coeff 0.01, clipping `ε=0.1×α` with `α` annealed from 1 to 0).

- Ablation results: What works and what does not (Table 1)
  - Quoted result:
    > Clipping, ε = 0.2 achieves the best average normalized score 0.82, outperforming ε=0.1 (0.76) and ε=0.3 (0.70). Adaptive KL with `d_targ` in {0.003, 0.01, 0.03} yields 0.68–0.74; fixed KL penalties β in {0.3, 1, 3, 10} yield 0.62–0.72. “No clipping or penalty” performs poorly (-0.39).
  - Interpretation:
    - The “no regularization” objective collapses on at least one task (HalfCheetah), confirming the need for trust-region‑like control.
    - Clipping at ε=0.2 strikes the best stability/learning balance on average across tasks.

- Continuous-control comparisons (Figure 3)
  - PPO (Clip) typically learns faster and to higher returns across the seven tasks within 1M steps. For example:
    - Hopper‑v1 and Walker2d‑v1: PPO reaches high returns quickly and stably; other methods show slower progress or plateau lower.
    - Reacher‑v1 (sparse/low magnitude rewards): PPO learns reliably while some baselines degrade or oscillate.
  - While exact numbers are visual in Figure 3, the qualitative pattern shows PPO dominating or tying the best on most tasks within the same sample budget.

- Humanoid showcase (Figures 4–5; Table 4)
  - PPO learns robust gaits and steering under disturbances, with learning curves steadily improving to high scores:
    > RoboschoolHumanoid-v0 reaches ~4000 reward by 50M timesteps; Flagrun and FlagrunHarder continue to improve up to 100M timesteps (Figure 4).
  - Still frames (Figure 5) illustrate target chasing and quick direction changes.

- Atari benchmark (Section 6.4; Table 2, Table 6, Figure 6)
  - Two metrics across 49 games:
    - Average reward per episode over all of training (sample efficiency): PPO “wins” 30 games, ACER 18, A2C 1 (Table 2).
    - Final performance (last 100 episodes): ACER wins 28, PPO 19, A2C 1; one tie (Table 2).
  - Game-level examples from Table 6 (last‑100 episode means):
    - Large PPO wins: Atlantis (2,311,815 vs. ACER 1,841,376; A2C 729,265), Kangaroo (9,928 vs. ACER 50), Jamesbond (561 vs. ACER 262; A2C 52).
    - Cases where ACER excels: UpNDown (145,051 vs. PPO 95,445), DemonAttack (38,808 vs. PPO 11,378).
    - PPO and ACER both solve classic control like Pong (~20), with PPO showing strong sample efficiency in Figure 6.

- Do the experiments support the claims?
  - Yes, for the central claims of stability, simplicity, and overall strong sample efficiency:
    - The ablation (Table 1) directly justifies the clipped objective.
    - Cross-domain performance shows PPO is competitive or superior on continuous control (Figure 3) and strongly sample‑efficient on Atari (Table 2, Figure 6), while final scores are competitive with ACER.
    - Hyperparameters are documented (Tables 3–5), and algorithmic details (Algorithm 1) are sufficient for reproduction.

## 6. Limitations and Trade-offs
- On‑policy nature and sample reuse limits
  - PPO reuses each batch for several epochs, but all data are still on‑policy with respect to `πθ_old`. It does not leverage large off‑policy replay buffers the way Q‑learning or ACER do; hence, absolute sample efficiency can lag off‑policy methods in some settings (e.g., several Atari games where ACER’s final scores are higher; Table 6).
- Hyperparameter sensitivity
  - Performance depends on clipping parameter `ε`, number of epochs `K`, and batch sizes. The ablation shows that `ε` affects outcomes noticeably (Table 1). While robust across a range, poor choices can hurt learning.
- No formal monotonic improvement guarantee
  - While `L_CLIP` is a pessimistic surrogate and empirically constrains updates (Figures 1–2), there is no hard guarantee like TRPO’s constrained formulation (Eqs. (3)–(4)). Section 4’s adaptive KL penalty reduces this gap but still relies on heuristic coefficient updates.
- Occasional large KL steps
  - With the adaptive KL scheme, the paper notes “occasional” updates with KL diverging from the target, though β quickly adjusts (Section 4).
- Compute considerations
  - PPO’s attractiveness comes from multiple epochs per batch, which increases optimization work per sample. Though wall‑time is competitive (Abstract), practitioners must balance epochs vs. throughput.
- Domain coverage
  - The paper evaluates standard continuous-control and Atari tasks. It doesn’t cover partially observable tasks with heavy recurrence at scale, multi‑agent settings, or long‑horizon sparse‑reward tasks beyond RoboschoolFlagrunHarder’s disturbances.

## 7. Implications and Future Directions
- Field impact
  - PPO provides a practical, reliable default for policy-gradient RL: simple to implement, stable under multi‑epoch updates, and effective across discrete and continuous domains. This lowers the barrier to applying RL in robotics and game environments and quickly became a standard baseline.
- Practical applications
  - Robotics control (locomotion, manipulation) where continuous actions and stable learning are critical (Sections 6.2–6.3).
  - Game AI and simulation-based optimization where scalability and ease of implementation matter (Atari, Section 6.4).
  - Architectures with parameter sharing or stochastic components (e.g., joint policy/value networks, auxiliary tasks), since PPO uses only first‑order optimization (Section 1).
- Research directions enabled/suggested
  - Theoretical analysis: characterize when `L_CLIP` guarantees improvement and how clip thresholds relate to implicit trust-region sizes.
  - Adaptive mechanisms: learn or schedule `ε`, epochs, or minibatch sizes based on observed KL, improvement, or variance—bridging Sections 3 and 4.
  - Off‑policy extensions: combine PPO’s stability with replay buffers and importance sampling corrections for broader sample reuse.
  - Exploration strategies: integrate stronger intrinsic motivation or entropy schedules within the `L_total` framework (Eq. (9)).
  - Large‑scale/distributed training: extend Algorithm 1 with efficient actor–learner architectures and heterogenous hardware while maintaining the KL‑aware control implied by the clipped loss.

Overall, the paper’s central idea—clipping the likelihood ratio inside a pessimistic surrogate (Eq. (7))—is both elegant and pragmatic. The ablations (Table 1) show why it matters, and the cross‑domain results (Figures 3–6; Tables 2 and 6) demonstrate that this simple change delivers robust, high‑performing policy optimization without the complexity of classical trust‑region solvers.
