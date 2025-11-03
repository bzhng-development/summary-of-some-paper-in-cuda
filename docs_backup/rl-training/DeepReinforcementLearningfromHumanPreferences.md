# Deep Reinforcement Learning from Human Preferences

**ArXiv:** [1706.03741](https://arxiv.org/abs/1706.03741)

## 🎯 Pitch

This paper pioneers a scalable method for training deep reinforcement learning agents by learning from human preferences between pairs of trajectory segments, rather than relying on hand-designed reward functions. By efficiently eliciting and modeling sparse human feedback, the approach enables agents to master complex tasks—including those with goals that are hard to specify programmatically—while using orders of magnitude less human input than prior systems. This breakthrough dramatically broadens the range of problems RL can tackle, making it possible to align machine behavior with nuanced human intentions in both research and real-world settings.

---

## 1. Executive Summary (2-3 sentences)
This paper introduces a practical way to train reinforcement learning (RL) agents using pairwise human preferences over short video clips instead of a programmatic reward function. It learns a reward model from these preferences and optimizes a policy against that model, achieving near-RL performance on MuJoCo robotics tasks with a few hundred to ~1,400 labels and substantial learning on Atari with ~5,500 labels, while also enabling novel behaviors that lack obvious reward functions.

## 2. Context and Motivation
- Problem addressed
  - Many real-world tasks do not have a clean, hand-coded reward function; specifying one can be brittle or misaligned with what people actually want (Introduction).
  - Simply using human feedback as the reward is too expensive for modern deep RL, which needs millions of environment steps (Abstract; Section 1).
- Importance
  - Practical: empowers RL to tackle tasks where goals are complex, implicit, or hard to formalize (e.g., “do a backflip,” “drive with the flow of traffic”).
  - Safety/alignment: addresses concerns that agents may exploit flawed reward proxies by instead learning human preferences (Introduction; (Amodei et al., 2016) cited in Section 1.1).
- Prior approaches and limitations (Section 1.1)
  - Inverse RL and imitation learning require demonstrations, which are often unavailable or infeasible for non-human morphologies.
  - Preference-based RL existed but assumed linear rewards over hand-coded features and small state spaces (e.g., Akrour et al., 2012/2014; Wilson et al., 2012), limiting scalability.
  - Online human feedback methods (e.g., TAMER) typically require too much human interaction for deep RL time scales.
- Positioning
  - Scales preference learning to deep RL: no hand-engineered features, works in high-dimensional settings (Atari; MuJoCo), and uses very few human comparisons (often <1% of agent interactions; Abstract).
  - Provides a complete system: asynchronous dataflow, active query selection, ensemble-based uncertainty, and online preference collection to avoid reward misspecification problems.

## 3. Technical Approach
High-level idea: maintain two learnable components—(1) a policy `π` that acts in the environment and (2) a reward model `r̂` that predicts human preferences—trained in a loop where human comparisons continuously refine `r̂` and the policy optimizes against the current `r̂` (Figure 1; Section 2.2).

Key terms
- `Trajectory segment` (Section 2.1): a short sequence of observation–action pairs, σ = ((o0,a0), …, (ok−1, ak−1)). The paper uses 1–2 second clips (Section 2.2.2).
- `Preference query`: a comparison of two segments, asking a human which is better (or tie / cannot tell).
- `Bradley–Terry model`: a probabilistic model for pairwise comparisons where the probability that A beats B depends on their latent “scores.”
- `Label annealing`: gradually reducing the feedback rate over time so the reward model is accurate early yet adapts as the policy changes (Appendix A).
- `Occupancy distribution`: the distribution over states/segments induced by the current policy. It shifts as the policy improves, making online feedback important.

Step-by-step pipeline
1) Optimize the policy with the predicted reward (Section 2.2.1)
- Treat `r̂` as the reward signal and run standard RL.
- Use algorithms robust to changing rewards: A2C for Atari and TRPO for MuJoCo (Section 2.2.1).
- Normalize reward outputs (zero mean, fixed variance) to stabilize learning (Section 2.2.1).
- Encourage exploration under non-stationary rewards via entropy bonuses, especially for TRPO (Section 2.2.1; Appendix A.1).

2) Elicit human preferences on short clips (Section 2.2.2)
- Show two 1–2 second video segments of recent agent behavior and ask which is better, or tie/uncomparable.
- Store as triples `(σ1, σ2, μ)` where `μ` is a distribution over {1, 2} representing the chosen preference (ties become uniform over the two options).

3) Fit the reward model to predict preferences (Section 2.2.3)
- Model the probability that σ1 is preferred over σ2 using a Bradley–Terry softmax over the sum of per-timestep rewards:
  - Equation (1): P(σ1 ≻ σ2) = exp(Σt r̂(o1_t, a1_t)) / [exp(Σt r̂(o1_t, a1_t)) + exp(Σt r̂(o2_t, a2_t))].
- Train `r̂` to minimize cross-entropy loss with the observed labels (Section 2.2.3).
- Practical refinements that stabilize learning and model human error (Section 2.2.3):
  - Ensemble of predictors with bagging; average normalized outputs.
  - Holdout validation; tune L2 regularization; use dropout in some domains.
  - Explicit noise model: assume a 10% chance the human answers uniformly at random.
  - Normalize the output scale of each predictor before averaging.

4) Select which pairs to query (Section 2.2.4)
- Active querying by uncertainty: sample many candidate pairs and choose those with highest variance across the reward-model ensemble’s predictions (“disagreement”).
- Note: This simple uncertainty heuristic sometimes hurts performance (Section 3.3; Figures 5–6), but generally reduces label waste.

Asynchronous training loop (Figure 1; Section 2.2)
- The policy continuously generates new segments; a small subset is sent for human comparison; the reward model is updated asynchronously; its parameters are periodically pushed back to the policy optimizer.

Implementation and domain-specific details
- Avoid leaking task information through environment signals (Appendix A):
  - Remove variable-length terminations that implicitly reveal task success/failure (e.g., no life-loss signals in Atari, replace with penalties known only to the oracle when creating synthetic labels).
  - Mask on-screen scores in Atari to prevent trivial reward inference (Appendix A.2).
- Reward-model architectures
  - MuJoCo: 2-layer MLP (64 units each) with leaky ReLU; 1.5-second clips; entropy bonus 0.01 (0.001 for Swimmer); normalization of rewards (Appendix A.1).
  - Atari: 4-layer conv net on 84×84×4 frames with batch norm and dropout; 25-timestep clips; synchronous A2C with standard hyperparameters; keep only the last 3,000 labels in a buffer to emphasize recency (Appendix A.2).
- Pretraining and label schedule
  - Pretrain the Atari reward model for 200 epochs on initial random-policy data to avoid early policy collapse (Appendix A).
  - Anneal label rate over time; e.g., MuJoCo uses rate ∝ 2e6 / (T + 2e6); Atari decays every 5e6 frames (Appendix A).

Why these design choices?
- Pairwise preferences instead of absolute scores: easier and more consistent for humans; avoids reward-scale issues in regression (Section 3.3).
- Online feedback (interleaved with policy learning): prevents the agent from exploiting a static, partially learned reward model (“reward hacking”), which produced pathological behaviors when trained offline (Section 3.3; Figure 6 discussion on Pong).
- Short clips: capture meaningful, contextual behavior while keeping labeling time low; single frames underperform (Section 3.3).
- Ensembles and regularization: reduce overfitting to sparse, noisy labels and provide uncertainty estimates for active querying (Section 2.2.3; Section 2.2.4).

## 4. Key Insights and Innovations
- Learning complex behaviors from very sparse human feedback at deep-RL scale
  - Novelty: demonstrates that preference learning can supervise modern deep RL across high-dimensional domains (Atari; MuJoCo) with label budgets in the hundreds to thousands rather than millions (Abstract; Sections 3.1–3.2).
  - Significance: makes human-in-the-loop RL practical and cost-effective, enabling tasks without explicit reward functions.
- Online preference learning prevents reward exploitation
  - Insight: training the reward model offline on a fixed dataset leads to agents gaming the learned reward (e.g., Pong agents create endless volleys without scoring), whereas online feedback corrects such failures (Section 3.3; Figure 6 discussion).
  - Significance: connects preference learning with AI safety concerns and shows a concrete mitigation using interleaved updates.
- Preference modeling over trajectory segments using a Bradley–Terry formulation
  - Different from prior small-scale settings: works with non-linear neural reward models and segments starting from different states (no resets to arbitrary states), which complicates preference interpretation (Sections 2.1–2.2.3).
  - Significance: generalizes preference-based RL beyond linear-feature regimes and hand-crafted feature spaces.
- Efficient querying via ensembles and uncertainty
  - Contribution: use of an ensemble to approximate epistemic uncertainty and query high-variance comparisons (Section 2.2.4).
  - Nuance: ablations show mixed utility—helpful in some tasks, harmful in others—highlighting the need for better query-theoretic methods (Section 3.3; Figures 5–6).
- New capabilities: learning behaviors without obvious reward functions
  - Examples: Hopper backflips (900 queries), Cheetah running on one leg (800), staying even with traffic in Enduro (~1,300) (Section 3.2; Figure 4).

## 5. Experimental Analysis
Evaluation setup
- Environments and data
  - MuJoCo robotics (8 tasks including Hopper, Walker, Swimmer, Cheetah, Ant, Reacher, Pendulum, Double-Pendulum). True rewards are known but hidden from the learner; results are reported using the true rewards (Section 3.1.1; Figure 2).
  - Atari (7 games: BeamRider, Breakout, Pong, Q*bert, Seaquest, SpaceInvaders, Enduro). Scores masked; no life-loss signals to the agent; evaluation uses true game scores (Section 3.1.2; Figure 3; Appendix A.2).
- Baselines
  - RL with true reward (orange curves).
  - Synthetic oracle preferences that exactly reflect the hidden true reward (“synthetic labels”; blues).
  - Real human preferences (purple).
- Label budgets
  - MuJoCo: typically 700 human labels (with synthetic label runs at 350, 700, 1,400; Figure 2).
  - Atari: typically 5,500 human labels (with synthetic label runs at 3,300, 5,500, 10,000; Figure 3).
- Training details and metrics
  - For all settings, plots show average true reward vs. timesteps (MuJoCo averages over 5 runs except human runs; Atari over 3 runs except human runs; Figures 2–3).
  - Ablations systematically remove components: random queries, no ensemble, no online queries, no regularization, no segments, and a “target” variant that regresses to oracle total rewards instead of preferences (Section 3.3; Figures 5–6).

Main results
- MuJoCo robotics (Figure 2; Section 3.1.1)
  - With 700 human labels, performance nearly matches training directly on the true reward across tasks.
  - With 1,400 synthetic labels, learned-reward agents can slightly outperform true-reward RL:
    - Quote: “by 1400 labels our algorithm performs slightly better than if it had simply been given the true reward… [the learned reward is] slightly better shaped” (Section 3.1.1).
  - Human vs. synthetic labels:
    - Human labels are typically only modestly less efficient than synthetic; on Ant, human feedback is better due to learned “uprightness” shaping (Section 3.1.1).
- Atari (Figure 3; Section 3.1.2)
  - With 3,300 synthetic labels, BeamRider and Pong approach or match true-reward RL.
  - Seaquest and Q*bert learn more slowly but eventually near RL performance.
  - SpaceInvaders and Breakout improve substantially but do not reach RL; still, the agent “often passes the first level in SpaceInvaders” and reaches “score of 20 on Breakout, or 50 with enough labels” (Section 3.1.2).
  - Human labels are similar to synthetic but slightly worse at equal counts; notable exceptions:
    - Q*bert: fails with real human feedback, likely because short clips are hard to evaluate (Section 3.1.2).
    - Enduro: human-shaping rewards outperform A3C with true reward because humans reward progress toward passing cars, effectively shaping the objective (Section 3.1.2).
- Novel behaviors without explicit rewards (Section 3.2; Figure 4)
  - Hopper backflips: consistent, repeated backflips with safe landings (900 queries, <1 hour).
  - Cheetah on one leg: forward locomotion on one leg (800 queries, <1 hour).
  - Enduro “even driving”: stays level with other cars for long stretches (~1,300 queries, 4M frames).

Ablations and robustness (Section 3.3; Figures 5–6)
- No online queries (offline reward learning) performs poorly and can be catastrophically misaligned:
  - Quote: “on Pong offline training sometimes leads our agent to avoid losing points but not to score points; this can result in extremely long volleys” (Section 3.3).
- Comparisons vs. absolute targets:
  - MuJoCo: comparisons outperform target regression due to reward-scale issues; comparisons avoid tricky calibration (Section 3.3; Figure 5).
  - Atari: with clipped rewards, the gap is mixed—neither dominates across all games (Section 3.3; Figure 6).
- No segments (single frames) hurts considerably in continuous control; longer clips give more context per label (Section 3.3; Figure 5).
- Query selection by disagreement is not uniformly helpful; random queries sometimes do as well or better (Section 3.3; Figures 5–6).
- Regularization and ensembling matter; removing them can degrade performance (Figures 5–6).

Do the experiments support the claims?
- Yes, for the central claim that complex RL tasks can be learned from sparse preferences:
  - Robotics: near-parity with true-reward RL at 700 labels; sometimes surpassing at 1,400 labels (Figure 2).
  - Atari: substantial learning with thousands of labels; sometimes matching true-reward RL (Figure 3).
- The safety/robustness claim that online feedback prevents exploitation is strongly supported by the “no online queries” ablation and detailed failure case in Pong (Section 3.3; Figure 6 discussion).

## 6. Limitations and Trade-offs
- Assumptions in the preference model
  - Additivity: preferences are modeled as the sum of per-timestep rewards over clips (Equation (1)), with no discounting (Section 2.2.3). Tasks requiring long-horizon credit not visible in short clips can violate this.
  - Noise: a fixed 10% random-response rate approximates human error (Section 2.2.3), which may not capture more complex human biases or context-dependence.
- Human labeling constraints
  - Quality and consistency: real human labels can be noisier than synthetic ones, and contractors may be inconsistent or uneven over time (Section 3.1.2).
  - Hard-to-evaluate clips: Q*bert failed with real feedback, likely due to confusing, context-dependent clips (Section 3.1.2).
  - Label budgets are small but not negligible (tens of minutes to a few hours of human time).
- Non-stationarity and distribution shift
  - As the policy improves, the occupancy distribution changes; without online updates, the reward model can be exploited (Section 3.3).
- Query selection and uncertainty
  - The heuristic ensemble-disagreement strategy is crude and sometimes harmful (Section 2.2.4; Figures 5–6). It does not directly optimize expected information gain.
- Computational and system complexity
  - Asynchronous training, ensembles, and RL loops require stable engineering and compute; however, the paper estimates compute cost per Atari run at ~$25 (footnote in Discussion), which is practical but still non-trivial for large-scale studies.
- Environment modifications
  - Removing termination signals and masking scores (Appendix A) ensures clean evaluation of the method but diverges from standard benchmark settings; results may differ under unmodified environments.

## 7. Implications and Future Directions
- How this changes the field
  - Establishes preference-based RL as a practical alternative when reward design is hard, scaling to deep RL on challenging benchmarks (Abstract; Sections 3.1–3.2).
  - Bridges performance and alignment: online preference learning directly addresses reward misspecification and exploitation (Section 3.3).
- Research opportunities
  - Better active preference querying: move from ensemble variance to principled expected-value-of-information methods (Section 2.2.4 suggests this; see references Akrour et al., 2012; Krueger et al., 2016).
  - Richer preference models: incorporate discounting, context windows, or hierarchical rewards; consider recurrent reward models for partially observable or long-horizon tasks (Section 2.2.1 footnote).
  - Uncertainty-aware RL: integrate reward-model uncertainty into exploration or risk-sensitive policy optimization.
  - Multi-signal supervision: combine demonstrations, sparse rewards, and preferences; bootstrap from imitation then refine via preferences.
  - Human-in-the-loop tooling: improved labeling interfaces, rater training, and quality control to reduce noise and improve coverage.
- Practical applications
  - Robotics (home assistance, manipulation) where specifying rewards is hard but video-based preferences are easy.
  - Autonomous driving behaviors and human-centered policies (e.g., “drive politely,” “stay with traffic”) as in the Enduro “even driving” task.
  - Game AI and content creation, user interface optimization, and any domain where subjective quality judgments are central.

Selected supporting quotes and pointers
- “less than 1% of our agent’s interactions with the environment” and “train complex novel behaviors with about an hour of human time” (Abstract).
- Method overview and dataflow: Figure 1; Section 2.2.
- Preference model and equation: Section 2.2.3, Equation (1).
- Query selection: Section 2.2.4.
- Robotics results: Figure 2; Section 3.1.1.
- Atari results and exceptions: Figure 3; Section 3.1.2.
- Novel behaviors: Section 3.2; Figure 4.
- Ablations and failure modes: Section 3.3; Figures 5–6; Pong failure case description.
- Environment safeguards to avoid reward leakage: Appendix A; Appendix A.2 (Atari details).

Overall, the paper delivers a working recipe for training deep RL agents from human preferences with minimal supervision, demonstrates when and why it works, and surfaces important pitfalls—especially the necessity of online preference collection—to inform both practical deployment and future research.
