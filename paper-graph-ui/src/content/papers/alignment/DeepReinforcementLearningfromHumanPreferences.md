# Deep Reinforcement Learning from Human Preferences

**ArXiv:** [1706.03741](https://arxiv.org/abs/1706.03741)

## 🎯 Pitch

This paper introduces a scalable method for training deep reinforcement learning agents using a learned reward model derived from human preferences over short trajectory segments, rather than relying on hand-crafted reward functions or human demonstrations. By requiring feedback on less than 1% of the agent's interactions, the method enables state-of-the-art RL agents to learn complex tasks in challenging domains—even creating novel behaviors—making human-aligned AI practical and lowering the barrier to deploying RL in real-world, value-sensitive applications.

---

## 1. Executive Summary
This paper introduces a practical way to train reinforcement learning (RL) agents when no programmatic reward is available: learn a reward model from human preferences over short video clips and then optimize that learned reward. Using less than 1% as many human-labeled interactions as total environment steps, the method solves many standard MuJoCo and Atari tasks and can teach novel behaviors (e.g., a hopper doing backflips) with about an hour of human time (Sections 3.1–3.2; Figures 2–4).

## 2. Context and Motivation
- Problem addressed
  - Many real-world tasks lack a clean, programmable reward function that correctly captures what “good behavior” means (Introduction, p. 1–2). Hard-coded proxies often lead to agents gaming the reward rather than doing what people want.
  - Directly using human feedback as a reward at every step is too expensive for modern deep RL, which needs millions of interactions.

- Why it matters
  - Bridging the gap between human intent and agent objectives is central to deploying RL in the real world (e.g., robotics, safety-critical systems). The work also touches core concerns about value alignment (Introduction).

- Prior approaches and shortcomings
  - Inverse RL and imitation learning require demonstrations. They struggle when humans cannot demonstrate the task (non-human morphologies, many degrees of freedom).
  - Earlier preference-based RL used hand-engineered features or small domains, and typically did not scale to modern deep RL (Section 1.1).
  - Systems that learn only when a human is providing feedback are infeasible for tasks needing thousands of hours of experience (Section 1.1).

- Positioning relative to existing work
  - The paper scales preference-based reward learning to deep RL, removes reliance on hand-crafted features, and shows it works with non-expert labelers in challenging domains (Atari, MuJoCo) using modest human time (Sections 3.1–3.2).

Key terms
- `policy (π)`: mapping from observations to actions.
- `reward function (r)`: assigns a numeric value to state–action pairs; the agent aims to accumulate high reward.
- `trajectory segment (σ)`: a short sequence of observation–action pairs; here, a 1–2 second clip (Section 2.2.2).
- `preference`: a human judgement that one segment is better than another for the task goal.

## 3. Technical Approach
At a high level (Figure 1, Section 2.2): learn a reward model from pairwise human preferences over short trajectory segments, and simultaneously train a policy to maximize predicted reward.

Step-by-step pipeline (three asynchronous processes)
1. Generate experience (Section 2.2.1)
   - The current policy `π` interacts with the environment to produce trajectories.
   - Use an RL algorithm to update `π` to maximize the current predicted reward `r̂(ot, at)`.
   - Algorithms:
     - Atari: `A2C` (synchronous A3C) (Mnih et al., 2016; Section 2.2.1, Appendix A.2).
     - MuJoCo: `TRPO` with an entropy bonus for exploration under changing rewards (Section 2.2.1, Appendix A.1).
   - Rewards from `r̂` are normalized to zero mean and fixed variance because absolute scale is underdetermined (Section 2.2.1).

2. Elicit human preferences (Section 2.2.2)
   - Select pairs of short video clips (1–2 seconds; Atari uses 25 time steps; MuJoCo uses ~1.5 seconds).
   - A human compares the pair and chooses: left better, right better, tie, or “can’t tell.”
   - Store each labeled pair `(σ1, σ2, μ)` in a database `D`, where `μ` is the label distribution over {1,2}. Ties are encoded as a uniform choice; “can’t tell” is discarded.

3. Fit the reward model (Section 2.2.3)
   - Model the human’s choice probability with a Bradley–Terry style preference model over summed predicted rewards:
     - In plain language: a segment is judged better with probability increasing in the total predicted reward across its frames.
     - Equation 1 (Section 2.2.3) formalizes this:
       - `P̂(σ1 ≻ σ2) = exp(Σ_t r̂(o1_t, a1_t)) / [exp(Σ_t r̂(o1_t, a1_t)) + exp(Σ_t r̂(o2_t, a2_t))]`.
   - Learn `r̂` by minimizing cross-entropy between predicted and actual human preferences.
   - Practical refinements (Section 2.2.3):
     - Ensemble: train multiple reward predictors on bootstrap samples; average their normalized outputs. This supports uncertainty estimation and regularizes learning.
     - Regularization: hold out ~1/e of data for validation; tune `L2` regularization (and dropout in some domains) to keep validation loss within 1.1–1.5× training loss.
     - Human lapse modeling: add a 10% chance that the label is random to reflect non-zero human error, even for obvious differences.
     - Reward normalization: independently normalize each ensemble member’s output before averaging.

Active query selection (Section 2.2.4)
- Approximate uncertainty by the ensemble’s disagreement: sample many candidate clip pairs, have each reward model predict the preferred segment, compute variance across the ensemble, and ask humans to label the most disagreed-upon pairs.
- This is a crude proxy for information gain. Ablations later show it helps sometimes, hurts in others (Section 3.3; Figures 5–6).

Design choices and how they address challenges
- Comparisons over absolute scores (Section 2.2.2, 3.3): Humans are more consistent at pairwise comparisons, especially in continuous control; comparisons also remove issues from varying reward scales.
- Short clips not single frames (Ablation “no segments,” Figure 5; discussion in Section 3.3): Short temporal context helps humans judge outcomes and helps the model infer reward-relevant dynamics.
- Online preference collection (Ablation “no online queries,” Figures 5–6): Continually updating `r̂` avoids agents exploiting early, partial reward models; offline-only labels led to pathological policies (e.g., endless pong volleys).
- Asynchrony (Section 2.2): Keeps the system data-efficient—RL generates experiences; labeling proceeds in parallel; reward models update continuously.

Experimental implementation (Appendix A)
- Environments: Gym MuJoCo and Atari ALE via OpenAI Gym.
- To avoid leaking goal information, the authors remove environment termination signals and scoreboard displays that would otherwise encode reward (Appendix A).
- Labeling:
  - Human contractors receive brief, task-specific instructions (Appendix B).
  - Average 3–5 seconds per query; 15–300 minutes total human time depending on task.
  - Label rate is annealed during training to emphasize early shaping and later adaptation (Appendix A).

## 4. Key Insights and Innovations
- Learning from pairwise human preferences at deep RL scale
  - Novelty: Prior preference-based RL relied on hand-crafted features or small domains. Here, a learned reward model directly from raw observations scales to MuJoCo and Atari (Sections 1.1, 3).
  - Significance: Enables RL when no reward function is available or is hard to specify.

- Online, asynchronous reward modeling intertwined with RL
  - Novelty: Reward learning is concurrent with policy learning and continuously informed by new on-policy clips (Section 2.2).
  - Significance: Prevents reward hacking on a fixed, partial model; ablations show offline-only labels cause pathological behaviors (Section 3.3; Figures 5–6).

- Short trajectory segments as the unit of feedback
  - Novelty: Instead of whole trajectories (hard to compare) or single frames (too little context), use 1–2 second clips (Section 2.2.2).
  - Significance: Maximizes information per human-second. Ablations show segments outperform single frames in continuous control (Figure 5, “no segments”).

- Simple statistical preference model with practical tweaks
  - Novelty: A Bradley–Terry model over summed rewards with explicit human lapse rate, ensemble normalization, and regularization (Section 2.2.3).
  - Significance: Robust to noisy, inconsistent human labels; provides uncertainty estimates for active query selection.

- Minimizing hidden supervision from the environment
  - Novelty: Remove variable-length episode endings and scoreboard displays to ensure the agent learns only from human preferences, not leaked rewards (Appendix A).
  - Significance: Validates that the method truly replaces explicit reward with human preferences.

These are fundamental innovations (new capability and training paradigm), not just incremental improvements.

## 5. Experimental Analysis
Evaluation methodology
- Domains
  - MuJoCo continuous control: Hopper, Walker2d, Swimmer, HalfCheetah, Ant, Reacher, DoublePendulum, Pendulum (Section 3.1.1; Appendix A.1).
  - Atari: BeamRider, Breakout, Pong, Q*bert, Seaquest, SpaceInvaders, Enduro (Section 3.1.2; Appendix A.2).
- Human feedback
  - Queries are clip-pair comparisons; 3–5 seconds per query; total 15 minutes to 5 hours per task (Section 3.1).
  - For some runs, labels come from a synthetic oracle that compares clips using the true reward function—this isolates the sample-efficiency of the preference-learning pipeline from human noise (Section 3.1).
- Baselines
  - Standard RL on true reward (TRPO for MuJoCo; A2C/A3C for Atari).
  - The proposed method with varying numbers of synthetic labels.
  - The proposed method with real human labels.
- Metrics
  - Report true task reward to compare against standard RL (even though true reward is hidden from the agent during training).
  - Ablations test components: query selection, ensemble, online queries, regularization, using single frames vs clips, and predicting comparisons vs regressing to target returns (Section 3.3; Figures 5–6).

Main quantitative findings

MuJoCo (Figure 2)
- With 700 human labels (purple curves), the method “nearly matches” standard RL across tasks such as Hopper, Walker, Swimmer, HalfCheetah, Ant, Reacher, DoublePendulum, and Pendulum.
- With 1,400 synthetic labels, the approach slightly exceeds RL on average in several tasks. Quote:
  > “By 1400 labels our algorithm performs slightly better than if it had simply been given the true reward” (p. 7), attributed to better-shaped learned rewards.
- Human vs synthetic labels: humans are typically only slightly less efficient than synthetic labels; on Ant, human labels outperform synthetic labels due to implicit reward shaping (“standing upright” priority), while the RL reward’s hand-crafted upright bonus was less effective (Section 3.1.1).

Atari (Figure 3)
- Using 5,500 labels:
  - BeamRider, Pong: synthetic labels match or closely approach RL performance with as few as 3,300 synthetic labels.
  - Seaquest, Q*bert: synthetic labels eventually reach near-RL scores but learn more slowly.
  - SpaceInvaders, Breakout: synthetic labels do not match RL but still achieve substantial learning; e.g., reach scores ~20 (and up to ~50 with more labels) in Breakout (p. 7–8).
  - Enduro: human labelers outperform A3C because they reward progress toward passing cars, effectively shaping the reward; performance comparable to DQN (Section 3.1.2).
- Human vs synthetic labels: human runs are slightly worse than synthetic runs with the same number of labels and are roughly comparable to synthetic runs with ~40% fewer labels (Section 3.1.2). Potential reasons include label noise, inconsistent raters, and uneven temporal coverage.

Novel behaviors (Section 3.2; Figure 4)
- Hopper backflips: learns repeated backflips landing upright, trained with 900 queries in <1 hour.
- HalfCheetah on one leg: learns to move forward on one leg with 800 queries in <1 hour.
- Enduro “even mode”: learns to keep pace with traffic with ~1,300 queries and ~4M frames.

Ablations and robustness checks (Section 3.3; Figures 5–6)
- No online queries (offline-only labels): large failures; agents exploit partial reward models. Example: in Pong, agents learn “don’t lose” but not “score,” creating infinite volleys that repeat the same sequence (p. 9–10; videos referenced).
- No segments (use single frames): big drop in continuous control (Figure 5).
- Comparisons vs targets (regressing to true segment returns):
  - Continuous control: comparisons work much better (scale invariance and robustness).
  - Atari: clipped rewards reduce scale issues; both are mixed with neither dominating (Section 3.3).
- No ensemble and random queries: ensemble uncertainty helps sometimes; in some domains random sampling is similar or better, indicating room to improve query selection (Figures 5–6).
- No regularization: degrades performance, underscoring the need to prevent overfitting the reward model to limited labels.

Do the experiments support the claims?
- Yes, for the core claims:
  - Data efficiency: 700 labels (MuJoCo) and 5,500 labels (Atari) suffice to get close to or sometimes surpass RL trained on true reward (Figures 2–3).
  - Learning complex novel behaviors from human time on the order of an hour (Section 3.2).
  - Necessity of online interaction between RL and reward learning to avoid exploitation (Figures 5–6; Section 3.3).

Implementation details that matter
- To prevent leakage of reward:
  - Atari: blank out on-screen score regions; treat episodes as continuous, not signaling life-loss or end-of-episode to the agent (Appendix A.2).
  - MuJoCo: remove termination conditions that encode falling; instead add penalties the agent must learn (Appendix A.1).
- Label annealing: decay label rate with timesteps to balance early shaping with later adaptation (Appendix A).

Cost considerations
- The paper notes diminishing returns on further sample-efficiency gains because compute cost (~$25 per run) approaches non-expert label cost (~$36 for 5k labels at US minimum wage) (footnote 6; p. 10).

## 6. Limitations and Trade-offs
- Assumptions about human feedback
  - Humans can reliably compare short clips and their preferences correspond (approximately) to a latent additive reward over the clip (Equation 1). If preferences depend on long-term consequences unseen in the clip, the model may misgeneralize.
  - Clip comparisons begin from different states (Section 2.1 notes this complicates interpretation), so preferences can be confounded by context differences.

- Reward model expressiveness and stability
  - Non-stationary target: as the policy changes, the state distribution shifts, and `r̂` must continually adapt. This can cause instability; hence the reliance on policy-gradient methods and entropy bonuses (Section 2.2.1).

- Query selection suboptimality
  - The ensemble-variance heuristic is a crude uncertainty measure and sometimes hurts performance (Section 2.2.4; Section 3.3). Optimal information-theoretic acquisition is left for future work.

- Scalability of human time
  - While efficient, the approach still needs thousands of labels for harder games (e.g., 5,500 for Atari; Section 3.1.2). Tasks demanding nuanced, long-horizon judgments may require more labels or richer feedback.

- Reward hacking if feedback is not interleaved
  - Offline-only reward learning leads to exploitation of model weaknesses (Section 3.3). Safe deployment requires tight loops between labeling and training.

- Domain coverage and generality
  - Demonstrated on simulated control and Atari. Real-world robotics introduces perception noise, safety constraints, and longer horizons, which may stress the clip-based preference model.

## 7. Implications and Future Directions
- How it changes the landscape
  - Establishes a scalable recipe for “RL without rewards” using human preferences, broadening the set of tasks amenable to deep RL. It also shows that reward modeling plus RL can sometimes outperform hand-coded rewards via better reward shaping (Figure 2, Ant).

- Follow-up research enabled or suggested
  - Smarter query strategies: move from ensemble variance to expected value-of-information or Bayesian experimental design (Section 2.2.4).
  - Richer feedback: beyond pairwise comparisons to ordinal rankings, natural-language critiques, or counterfactual queries.
  - Longer-horizon reasoning: incorporate recurrent reward models or variable-length segments; infer or learn discounting (footnote in Section 2.2.3).
  - Robustness and safety: detect and penalize reward model exploitation; uncertainty-aware RL that seeks clarifying human input before drifting off-policy.
  - Better architectures: task-adaptive reward networks, representation learning that aligns human-relevant features, ensembles with calibrated uncertainty.

- Practical applications
  - Robotics tasks where reward is hard to specify (e.g., household chores like “tidy the table,” where success is defined by human judgment).
  - Interactive training of agents in domains with sparse or misleading rewards (e.g., driving simulators, complex games).
  - Rapid prototyping of novel behaviors by non-experts (Section 3.2 shows backflips, one-leg running, driving even with traffic).

Quote capturing the overarching impact (Section 4):
> “By learning a separate reward model using supervised learning, it is possible to reduce the interaction complexity by roughly 3 orders of magnitude.”  
In the long run, the goal is to make learning from human preferences “no more difficult than learning it from a programmatic reward signal,” enabling powerful RL systems to optimize for complex human goals rather than simplistic proxies.
