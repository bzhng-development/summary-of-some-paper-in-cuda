# Deep Reinforcement Learning from Human Preferences

**ArXiv:** [1706.03741](https://arxiv.org/abs/1706.03741)

## 🎯 Pitch

This paper introduces a scalable framework for training deep reinforcement learning agents using sparse human feedback, specifically by asking non-expert humans to express preferences between pairs of short video clips instead of engineering explicit reward functions. By learning a reward model from these preferences and using it to guide RL agents, the approach matches or surpasses standard RL performance in challenging robotics and Atari tasks while requiring less than 1% of environment interactions to be labeled. This breakthrough makes it practical to align RL agents with complex, real-world goals that are hard to formalize, radically lowering the barriers to safe, value-aligned AI deployment.

---

## 1. Executive Summary
This paper shows how to train deep reinforcement learning (RL) agents using only brief human preferences between pairs of short video clips, instead of a hand-coded reward function. It learns a reward model from those comparisons and then optimizes that learned reward, achieving near–state-of-the-art performance on MuJoCo robotics control tasks and substantial learning on Atari games while requiring feedback on under 1% of the agent’s interactions.

## 2. Context and Motivation
- Problem addressed
  - Many RL successes assume a precisely specified reward function, but many real tasks have goals that are hard to encode (e.g., “clean a table” or “scramble an egg”). Mis-specified rewards can produce unintended behavior (“reward hacking”). Section 1 motivates replacing hard-coded reward functions with human guidance that expresses what is actually desired.

- Why this matters
  - Practical impact: Enables RL on tasks where writing a reward is impractical or unsafe.
  - Safety/Alignment: Communicating complex goals reduces misalignment between the agent’s objective and human values (Section 1, Abstract).

- Prior approaches and gaps
  - Imitation learning and inverse RL require demonstrations; many desired behaviors are hard for humans to demonstrate (non-human morphologies, high degrees of freedom).
  - Prior preference-based RL (e.g., Akrour et al. 2012/2014; Wilson et al. 2012) assumed small or linear settings, hand-coded features, or synthetic feedback, and did not scale to modern deep RL tasks (Section 1.1).
  - Direct use of human feedback as a dense reward is prohibitively expensive because deep RL needs massive experience (Section 1).

- Positioning relative to existing work
  - The paper scales preference-based reward learning to deep RL with high-dimensional observations and complex control, using:
    - Pairwise comparisons of short trajectory segments instead of absolute scores (Section 2.2.2).
    - A learned reward model trained asynchronously from these preferences (Figure 1; Section 2.2).
    - Active query selection and ensembles to manage uncertainty (Sections 2.2.3–2.2.4).
  - Demonstrates novel behaviors from scratch where no natural reward exists (backflips, one-legged gait, driving even with traffic) with about an hour of human time (Section 3.2; Figure 4).

## 3. Technical Approach
At a high level, the system alternates between collecting experience, asking a human to compare pairs of short clips from that experience, fitting a reward model to those comparisons, and then training a policy to maximize the learned reward (Figure 1; Section 2.2). Key terms:
- `Trajectory segment` (σ): a short sequence of observation–action pairs (e.g., 1–2 seconds of video; Section 2.2.2).
- `Preference` (σ1 ≻ σ2): human indicates which of two segments is better relative to a goal (Section 2.1).

Step-by-step algorithm (Sections 2.2.1–2.2.4)
1. Initialize
   - Policy π (maps observation `o` to action `a`).
   - Reward model r̂(o, a) (a neural network that predicts instantaneous reward).
   - Collect some early clips from a randomly initialized π to seed the comparison dataset D (Appendix A).

2. Gather experience and propose queries
   - π interacts with the environment, producing trajectories {τ1, …}.
   - From these, sample many candidate pairs of short segments (k steps long; 1–2 seconds) (Section 2.2.2).

3. Select which pairs to show the human (active learning)
   - Train an ensemble of reward models on the current preference dataset D (Section 2.2.3).
   - For each candidate pair, predict which segment is better using each ensemble member; pick pairs with highest ensemble disagreement (variance) to query first (Section 2.2.4). This approximates “query the most uncertain” pairs.

4. Human comparison
   - Human watches the two clips and chooses: left better, right better, tie, or can’t tell (Section 2.2.2).
   - Store as a triple (σ1, σ2, μ) in D, where μ is a probability distribution over {1, 2} reflecting the label (e.g., one-hot for a strict preference, uniform for ties).

5. Train the reward model r̂
   - Treat r̂ as a latent scoring function for pairwise preferences using a Bradley–Terry/Luce–Shepard model (Section 2.2.3, Equation 1):
     - Plain-language: if a segment’s total predicted reward (sum of r̂ over the segment’s time steps) is higher, it should be preferred with higher probability.
     - Notation (Eq. 1): P̂(σ1 ≻ σ2) = exp(∑t r̂(o1t, a1t)) / [exp(∑t r̂(o1t, a1t)) + exp(∑t r̂(o2t, a2t))].
   - Train r̂ by minimizing cross-entropy loss between P̂ and the human labels μ over the dataset D (Section 2.2.3).
   - Practical choices to improve robustness (Section 2.2.3):
     - Ensemble: train multiple predictors on bootstrap samples; normalize and average outputs.
     - Regularization: hold out 1/e of D as validation; use L2 and (in some domains) dropout; tune L2 so validation loss is 1.1–1.5× training loss.
     - Human error model: assume a 10% chance the human labels uniformly at random (adds noise floor in the likelihood).
     - Reward normalization: set mean to zero and control standard deviation to stabilize RL (Sections 2.2.1, A.2).

6. Train the policy π on the learned reward
   - Replace the environment’s reward with r̂(o, a) and run standard RL:
     - MuJoCo control: TRPO with slight entropy bonuses for exploration under non-stationary rewards (Section 2.2.1; Appendix A.1).
     - Atari: A2C (synchronous A3C) with standard hyperparameters (Appendix A.2).
   - Because r̂ changes over time (as new labels arrive), policy-gradient methods are used for robustness to non-stationary reward signals (Section 2.2.1).

7. Repeat steps 2–6 asynchronously
   - Trajectories flow from RL to the query selector, preferences flow to reward learning, and the updated r̂ flows back to RL (Figure 1; Section 2.2). Label rate is annealed over time to emphasize early shaping while adapting to new states later (Appendix A).

Design decisions and why
- Preferences over short clips vs absolute scores
  - Humans provide more consistent, faster judgments over comparisons; comparisons avoid scale issues in regression and were empirically more effective in continuous control (Section 3.3).
- Short clips (1–2 seconds)
  - Offer enough temporal context for meaningful choices without overwhelming evaluators; per-clip information density is high (Sections 2.2.2, 3.3).
- Online, interleaved labeling
  - Prevents the agent from exploiting a static, partially learned reward function and supports adaptation as the policy explores new states (Section 3.3).
- Ensemble-based uncertainty sampling
  - Simple approximation of epistemic uncertainty to prioritize informative queries (Section 2.2.4). Ablations show mixed benefit depending on domain (Figures 5–6).

Implementation details that avoid hidden supervision
- Remove termination signals and score displays that could leak task information beyond human preferences (Appendix A).
  - MuJoCo: replace termination with penalties the agent must learn (Appendix A.1).
  - Atari: hide score areas and convert to a single continuous episode; replace end-of-episode with penalties for synthetic labels (Appendix A.2).

## 4. Key Insights and Innovations
- Learning a reward function from human preferences scales to deep RL
  - Novelty: Prior preference-based RL worked on small or linearized domains; here it trains on high-dimensional vision (Atari) and complex control (MuJoCo) using deep networks (Sections 1.1, 3.1).
  - Significance: Enables training with no programmatic reward, using only ~0.1–1% labeled interactions (Abstract; Sections 3.1–3.2).

- Pairwise comparisons over short trajectory segments are an effective supervision primitive
  - Different from absolute scoring: avoids calibration/scale problems and yields better learning in continuous control (Section 3.3).
  - Why it matters: Human raters provide consistent feedback quickly (3–5 seconds per query; Section 3.1), enabling practical oversight budgets (30 minutes to 5 hours).

- Online, asynchronous learning of the reward prevents reward exploitation
  - Insight: Training the reward model only once (offline) leads to pathological behaviors when the agent maximizes an imperfect reward (e.g., Pong “endless volleys”; Section 3.3).
  - Contribution: Show that interleaving labeling with RL is necessary for stability and alignment with human intent (Figures 5–6; Section 3.3).

- Practical recipe for robust preference learning at scale
  - Components: Bradley–Terry likelihood over clip sums (Eq. 1), ensemble predictors with normalization, regularization with held-out set, human noise model, and active query selection (Section 2.2.3–2.2.4).
  - Significance: These engineering choices collectively make the method work in challenging settings (ablation in Figures 5–6).

- Demonstration of novel, hard-to-specify behaviors
  - Examples: Hopper performing repeated backflips (Figure 4), Half-Cheetah running on one leg, and Enduro “keep even with traffic” behavior (Section 3.2).
  - Importance: Clear evidence the approach can express goals that are awkward to encode as reward functions.

## 5. Experimental Analysis
Evaluation setup
- Domains
  - MuJoCo control tasks (8): Hopper, Walker, Swimmer, Cheetah, Ant, Reacher, Double-Pendulum, Pendulum (Section 3.1.1; Figure 2).
  - Atari games (7): BeamRider, Breakout, Pong, Q*bert, SeaQuest, Space Invaders, Enduro (Section 3.1.2; Figure 3).
- Baselines
  - RL with true reward (TRPO for MuJoCo, A2C for Atari).
  - Synthetic “oracle” feedback: comparisons are generated automatically by comparing true rewards of the two clips (Section 3.1). This isolates the effect of scarcity/noise in human feedback.
- Feedback budgets
  - MuJoCo: 700 human queries; compare to 350/700/1400 synthetic queries (Section 3.1.1; Figure 2).
  - Atari: 5,500 human queries; compare to 3,300/5,500/10,000 synthetic queries (Figure 3).
- Human labeling
  - Non-expert contractors with 1–2 sentence task instructions; 3–5 seconds per query; 15 minutes to 5 hours total per task (Section 3.1).

Main quantitative results
- MuJoCo (Figure 2)
  - With 700 human labels, performance nearly matches training on true reward across tasks. Curves show similar asymptotic returns though with somewhat higher variance.
  - With 1,400 synthetic labels, performance slightly exceeds true-reward RL on several tasks, likely due to better “reward shaping” learned from preferences (Section 3.1.1).
  - Notable case: On Ant, human feedback outperforms synthetic because humans prefer “standing upright,” providing helpful shaping beyond the simple hand-engineered upright bonus in the original reward (Section 3.1.1).

- Atari (Figure 3)
  - Mixed but substantial learning:
    - BeamRider and Pong: with 3,300–5,500 synthetic labels, performance matches or approaches true-reward RL; human labels are slightly worse but comparable (Section 3.1.2).
    - SeaQuest and Q*bert: synthetic labels eventually approach RL but learn more slowly; human labels underperform synthetic, and on Q*bert the agent fails to beat level 1 with human labels—clips are hard for raters to evaluate (Section 3.1.2).
    - Space Invaders and Breakout: agents improve substantially (e.g., often pass level 1 in Space Invaders and reach scores ≈20 with 5.5k labels or ≈50 with 10k labels in Breakout) but do not match RL (Figure 3 text).
    - Enduro: human labeling outperforms A2C with true reward due to reward shaping—humans reward “progress toward passing cars” whereas A2C struggles to discover passing through exploration (Section 3.1.2).

Ablation studies (Sections 3.3; Figures 5–6)
- No online queries (offline reward learning)
  - Large degradation; produces unintended behaviors (e.g., in Pong the agent avoids losing without learning to score, leading to long, repetitive volleys). This shows the necessity of interleaving labeling with RL to prevent exploitation of reward-model blind spots.
- Comparisons vs “target” regression to known returns
  - In MuJoCo, comparisons outperform regression to target returns due to reward-scale issues in regression (Section 3.3; Figure 5).
  - In Atari (where rewards are clipped to ±1), the two are mixed; neither consistently dominates (Figure 6).
- Clips vs single frames (MuJoCo)
  - Single-frame preferences perform much worse; clips are significantly more informative per judgment, though more time-consuming per frame (Section 3.3; Figure 5, “no segments”).
- Ensemble/uncertainty sampling
  - Removing ensembles and querying randomly sometimes hurts and sometimes has minor effect; disagreement-based sampling can even hurt in some tasks (Figure 5–6; Section 2.2.4).
- Regularization
  - Removing regularization harms performance, indicating overfitting is a real risk in the reward model (Figures 5–6).

Robustness and setup details that strengthen the evidence
- To ensure preferences are the only supervision, the experiments remove task-encoding signals like variable-length episode terminations and on-screen scores (Appendix A).
- Reward model and policy are trained asynchronously, with label-rate annealing and a rolling label buffer to emphasize recent, distribution-shifted data (Appendix A.2).

Overall assessment
- The MuJoCo results convincingly show that preference-based learned rewards can train competitive policies with modest human time budgets.
- Atari results demonstrate feasibility but also reveal challenges in credit assignment from short clips and label quality; yet they still achieve meaningful control without access to the score.

## 6. Limitations and Trade-offs
Assumptions and modeling choices
- Additive reward model over time
  - Equation 1 assumes human preferences arise from summing an (unknown) per-timestep reward; real human judgments may depend on non-additive temporal patterns (Section 2.1; footnote 1).
- Human noise model
  - A fixed 10% random-label assumption may not match varying rater reliability or systematic biases (Section 2.2.3).
- Observability
  - The reward model uses observations (and sometimes stacks of frames) as input; if crucial state is unobserved, preferences may be ambiguous.

Scenarios not addressed
- Physical robots and real-world deployment: experiments are in simulation; transfer to real hardware introduces delays, safety and interpretability demands.
- Long-horizon credit assignment from short clips: some tasks (e.g., Q*bert) show that short clips can be confusing and insufficient for evaluation (Section 3.1.2).
- Complex preference structures: only binary preferences (with tie/can’t tell), not richer feedback like ordinal scales or partial ordering constraints.

Computational and data constraints
- Human time is still a bottleneck for very large tasks; while reduced, 3–5 seconds per query and thousands of queries can still be costly.
- Non-stationary reward adds instability; policy-gradient methods and entropy bonuses mitigate but do not eliminate sensitivity (Sections 2.2.1, 3.1.1).

Failure modes and open questions
- Offline reward fitting leads to “reward hacking” behaviors (Section 3.3).
- Active query selection via ensemble disagreement is crude and inconsistently helpful; expected value of information (EVoI) is not implemented (Section 2.2.4).
- Generalization of r̂ outside labeled regions remains risky; the paper uses a rolling buffer and regularization but offers no formal guarantees.

## 7. Implications and Future Directions
How this changes the landscape
- Establishes a practical pipeline for aligning deep RL agents to human intent without programmatic rewards, using minimal feedback. It bridges preference learning and modern deep RL at scale (Sections 3–4).

Follow-up research opportunities
- Better query selection
  - Move from ensemble disagreement to EVoI-based selection (Section 2.2.4), model rater expertise, and diversify queried states to reduce over-concentration.
- Richer feedback signals
  - Beyond pairwise comparisons: include absolute ratings, natural-language rationales, or structured constraints, combined with uncertainty-aware training.
- Temporal modeling of human preferences
  - Relax additivity; use sequence models (e.g., recurrent reward models) to capture long-horizon or event-based preferences (footnote 1 in Section 2.1).
- Robustness and safety
  - Detect and penalize exploitation of r̂; integrate conservatism or adversarial training where the policy seeks r̂’s blind spots.
- Real-world deployment
  - Apply to physical robots and interactive systems where reward specification is the main obstacle (Section 4 discussion of long-run goals).

Practical applications
- Robotics: teaching non-trivial maneuvers (e.g., backflips) or task-specific styles without writing reward functions (Section 3.2; Figure 4).
- User-aligned behavior in complex environments (e.g., driving style like “keep with traffic,” Section 3.2).
- Interactive personalization: non-expert users can shape agent behavior with brief, intuitive choices rather than code (Section 2.1 goals).

Quotes and figure references that ground the above
- “Provide feedback on less than 1% of our agent’s interactions” (Abstract).
- Preference likelihood and clip-based learning: Equation 1; Sections 2.2.2–2.2.3.
- MuJoCo results nearing true-reward RL with 700 labels; slight surpass with 1,400 synthetic (Figure 2; Section 3.1.1).
- Atari mixed outcomes with concrete examples (Figure 3; Section 3.1.2).
- Ablations showing necessity of online queries and advantages of comparisons and clips (Figures 5–6; Section 3.3).
- Novel behaviors learned in about an hour of human time (Section 3.2; Figure 4).

In sum, the paper provides a clear, implementable framework—learn a reward from human preferences over short clips and optimize it—that substantially reduces the need for explicit reward design and opens the door to aligning deep RL with nuanced human goals.
