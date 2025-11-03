# Sharing is Caring: Efficient LM Post-Training with Collective RL Experience Sharing

**ArXiv:** [2509.08721](https://arxiv.org/abs/2509.08721)

## 🎯 Pitch

This paper presents SAPO, a decentralized and asynchronous reinforcement learning algorithm that enables independent language model nodes to efficiently post-train by sharing textual experiences (rollouts) rather than synchronizing weights or relying on centralized infrastructure. By allowing heterogeneous nodes—regardless of hardware or model architecture—to collaboratively propagate successful strategies through shared rollouts, SAPO nearly doubles cumulative reward in complex reasoning tasks compared to standard RL approaches. This innovation matters because it democratizes large language model training, slashes infrastructure costs, and accelerates collective learning, making powerful post-training accessible across diverse and unreliable compute environments.

---

## 1. Executive Summary
- This paper introduces SAPO (Swarm sAmpling Policy Optimization), a decentralized, asynchronous reinforcement learning (RL) post‑training method where many independent nodes fine‑tune their own models while sharing their text rollouts with one another instead of synchronizing weights (§3.2, Algorithm 1).
- In controlled tests on procedurally generated reasoning tasks, balanced sharing (4 local and 4 external rollouts per update) nearly doubles cumulative reward over standard single‑agent RL (+94% vs. no sharing; §5, Fig. 1–2), and a large open demo shows statistically significant gains for small models after ~175 rounds (Fig. 3).

## 2. Context and Motivation
- Problem addressed
  - RL post‑training can improve reasoning in language models without curated supervised data (e.g., RLHF, RLVR), but current distributed RL pipelines rely on centralized clusters that keep model weights synchronized. This introduces latency, communication bottlenecks, fragility, and high cost (§1–2).
- Why it matters
  - Making RL post‑training feasible on heterogeneous, unreliable, or volunteer hardware (e.g., laptops) would broaden access, reduce costs, and enable faster collective learning where “Aha moments” discovered by one model propagate to others (§1).
- Shortcomings of prior approaches
  - Centralized/distributed systems (e.g., PPO/GRPO at cluster scale) require tight coordination and weight synchronization across nodes (Refs. in §2). Multi‑agent techniques often orchestrate roles (e.g., generator/verifier) and communication patterns, adding complexity (§2).
- Positioning
  - SAPO bridges single‑agent RL and multi‑agent cooperation: each node trains independently but samples some training data from others’ shared rollouts. No assumptions about synchronous updates, identical models, or shared hardware are required, and nodes can act in isolation if they choose (§1–§3).

## 3. Technical Approach
Step‑by‑step overview (see §3 and Algorithm 1):

1) Swarm setup and data
- A “swarm” is a decentralized network of N nodes. Each node n has:
  - A set of verifiable tasks `Q_n` with ground-truth answers `y_q` and verification metadata `M_n` (how to check correctness) (§3.1).
  - A local policy model `π_n` (e.g., a small LM) and a local reward mechanism `ρ_n`. In this paper, rewards are programmatic verifiers from ReasoningGYM (§4.3).
- Verifiable task means correctness can be checked automatically (e.g., math answer matches solution); this enables fully decentralized training without human labels (§3.1).

2) Local rollout generation
- At each round t, node n samples a batch of tasks `B_n ⊆ Q_n`.
- For each question `q ∈ B_n`, the node produces `L_n` completions (the “rollout” for q): `R_n(q) = {a^n_1(q), …, a^n_{L_n}(q)}` (§3.1).
- Definition (paper‑specific): a “rollout” here is the set of multiple candidate answers an LM generates for a given question at one training step.

3) Sharing decoded rollouts
- Each node broadcasts a subset `S_n ⊆ B_n` as decoded text, packaged as `C_n(q) := (q, y_q, R_n(q), M_n)` (§3.2).
- “Decoded” means plain text, not token IDs from the original generator. Any node can re‑encode text with its tokenizer to compute token‑level log probabilities under its own policy. This is crucial: it lets a node treat others’ rollouts as if it had generated them—enabling policy‑gradient updates without needing the other node’s model weights (§3.2).

4) Constructing a training set with local and external data
- Node n builds its training minibatch `T_n` by combining:
  - `I_n` samples from its own rollouts in `B_n` (“self-rollouts”), and
  - `J_n` samples drawn from the union of shared items from other nodes (“external rollouts”), optionally using a local sampling/filtering strategy (§3.2).
- Paper’s filtering rule in experiments: discard rollouts with zero advantage, then uniformly sample from the rest (§5).
  - Definition (selective): “advantage” is the improvement signal used by policy gradients. In GRPO (the update rule used here), the advantage of a completion is computed relative to a group baseline over multiple completions for the same prompt. Zero advantage means the sample is not informative for improving the policy.

5) Rewarding and updating
- The node uses its local verifier `ρ_n` to award rewards to items in `T_n`. In ReasoningGYM, rewards are usually 1 for a correct answer (sometimes partial credit) and 0 otherwise (§4.3).
- The policy is updated using a policy‑gradient algorithm; the paper uses GRPO (§4.2) with:
  - No KL penalty (weight = 0), which the authors found more efficient for their setting.
  - Asymmetric clipping with lower/upper ratio bounds `ε_low = 0.2`, `ε_high = 0.28`.
  - Adam with default hyperparameters; 2000 training rounds (§4.2).

6) Why off-policy external data works without coordination
- In PPO/GRPO, the loss uses the probability ratio `r = π_new(a|q) / π_old(a|q)` of the current vs. previous local policy. For external rollouts, node n re‑encodes the shared text under its own `π_old` to compute probabilities; it does not need the remote node’s policy. Clipping the ratio keeps updates stable even if the external action was unlikely under the local model (§3.2, Algorithm 1).
- Result: no weight synchronization, no dependency on who generated the text; any node can learn from any other node’s experiences.

7) Why this design
- Minimal coupling: sharing small text blobs plus metadata is cheaper and more robust than exchanging weights or gradients, and it tolerates heterogeneous models/hardware (§1–§3).
- Built‑in quality control: nodes decide how many external samples to consume and how to filter them (§3.2). This autonomy helps adapt to variable swarm quality.

Implementation for experiments (§4):
- Eight `Qwen2.5-0.5B` models run as agents in Docker, orchestrated with Docker Compose. One GPU per agent, PyTorch distributed/NCCL used for message passing; training logic uses PyTorch (§4, §4.4).
- Data from ReasoningGYM: on-demand, procedurally generated, verifiable reasoning problems across 9 specialties (e.g., `base_conversion`, `propositional_logic`, `binary_matrix`; §4.1).

## 4. Key Insights and Innovations
- Decentralized experience sharing via decoded rollouts
  - What’s new: models share only text completions plus verification info; no weights, gradients, or architectures need to match (§3.2).
  - Why it matters: avoids synchronization bottlenecks, enables participation by heterogeneous and unreliable nodes, and lets learning propagate across the network like a multi‑agent system but with near‑zero orchestration (§1–§2).
- Re-encoding external text to make standard policy gradients work
  - What’s new: each node re‑encodes others’ text to compute token log‑probs under its own model, plugging them into PPO/GRPO with ratio clipping. This sidesteps the usual requirement that data come from the same policy (§3.2).
  - Significance: opens a practical route to off‑policy learning for LMs in decentralized settings without importance weights based on the generator’s policy.
- Balanced sharing is better than maximal sharing
  - Finding: in controlled tests, mixing equal parts local and external rollouts (4/4) gives the highest cumulative reward and strongest smoothed performance, while over‑reliance on external data (2/6) causes oscillations and forgetting (§5, Fig. 1–2).
  - Significance: identifies a practical operating point where benefits of cross‑pollination exceed the downsides of ingesting lower‑quality or mismatched samples.
- “Aha moment” propagation without explicit roles
  - Observation: once any node discovers useful behaviors (e.g., correct output formatting), sharing enables rapid spread across the swarm even without a special verifier agent or role design (§4.3, §5).
  - Significance: delivers some of the gains of multi‑agent systems (debate/specialization) with far less engineering (§2–§3).

## 5. Experimental Analysis
Evaluation design (§4–§6)
- Datasets and rewards
  - ReasoningGYM provides procedurally generated reasoning tasks with programmatic verifiers; 9 specialties selected (e.g., arithmetic, logic, abstract pattern reasoning; §4.1).
  - Reward: mostly binary 0/1 correctness; some verifiers allow partial credit (§4.3).
- Models and training
  - Eight agents each run `Qwen2.5-0.5B`. Per round, each agent samples several specialties and gets one question per chosen specialty; 8 completions per question (`L_n = 8`) (§4.1).
  - Update rule GRPO with no KL penalty; asymmetric clipping 0.2/0.28; Adam; 2000 rounds (§4.2).
- Sharing configurations compared (§5)
  - Baseline: 8 local / 0 external (no sharing).
  - SAPO variants: 6/2, 4/4, 2/6.
  - Each agent shares all its rollouts; when sampling external data, agents discard zero‑advantage items and uniformly sample from the remainder (§5).

Quantitative results
- Aggregate performance across agents and rounds (§5, Fig. 1):
  - Cumulative reward: 4/4 = 1093.31; 2/6 = 945.87; 6/2 = 854.43; baseline 8/0 = 561.79.
  - The 4/4 setting yields the best total reward and a 94% improvement over the baseline.
  - Quote from §5/Fig. 1 caption:
    > “Increasing the number of external rollouts raises peak rewards, but the highest overall reward accumulation occurs in the 4 local / 4 external setup, yielding a 94% improvement over the baseline.”
- Smoothed average performance across rounds (moving average window 100; §5, Fig. 2):
  - 4/4 consistently outperforms baseline and usually surpasses 6/2; it also beats 2/6 over most of training, though by a smaller margin.
  - Oscillations grow with more external data, especially in 2/6, indicating learning‑forgetting cycles (§5).
- Large-scale heterogeneous demo (§6, Fig. 3):
  - Nodes from thousands of community participants trained varied small models on diverse hardware. After each round, a “judge” sampled a ReasoningGYM problem; nodes submitted one answer (pass@1) scored by the verifier (§6).
  - For `Qwen2.5-0.5B`, the swarm variant’s per‑round performance becomes significantly better than the isolated baseline after roughly 175 normalized rounds (red regions mark p>0.05; blue/orange curves show cumulative and per‑round results):
    > “After a certain number of rounds, in this case approximately 175, the performance per round of the models in the swarm significantly exceeds that of the model trained in isolation.” (Fig. 3 caption)
  - For stronger `Qwen3-0.6B`, swarm vs. isolated showed similar performance (no consistent gain), plausibly because uniform random sampling admitted many low‑value external rollouts; better filtering might help (§6).

Do the experiments support the claims?
- The controlled 8‑agent study is carefully matched across settings (same total samples per round) and includes multiple sharing ratios, showing a clear optimum at 4/4 with both aggregate and smoothed metrics (§5, Fig. 1–2). This supports the claim that balanced sharing improves sample efficiency and peak reward.
- The open demo adds external validity: heterogeneous hardware and models in an unscripted environment. The significance analysis for `Qwen2.5-0.5B` (Fig. 3) supports SAPO’s practical utility, while mixed results for a larger small model (`Qwen3-0.6B`) highlight that benefits depend on sampling policy and model capacity (§6).

Ablations, robustness, and qualitative insights
- Ablation on local/external ratio serves as a robustness probe of sharing intensity (§5).
- Qualitative mechanism checks:
  - Removal of a formatting reward did not hurt because correct formatting propagated via sharing (§4.3) — evidence of “Aha moment” diffusion.
  - Oscillations under heavy external sampling are explained by two network effects (§5): (i) high performers being pulled down by low‑quality external data; (ii) “tragedy of the commons” when many consume but too few contribute, reducing pool quality.

## 6. Limitations and Trade-offs
- Dependence on verifiable rewards
  - Tasks must be programmatically checkable (RLVR‑style) for fully decentralized training (§3.1). Human feedback or nuanced, non‑verifiable objectives are outside the experimental scope.
- Stability vs. sharing intensity
  - Heavy reliance on external rollouts induces oscillations and forgetting (particularly 2/6; §5, Fig. 2). Choosing the local/external ratio is nontrivial and likely task‑ and swarm‑dependent.
- Sampling and adversarial quality
  - In the demo, uniform random sampling let in many low‑value rollouts (§6). Without robust filtering, high‑quality nodes may be harmed by noisy or adversarial contributions; trust and incentive mechanisms are not solved here.
- Model scale and diversity
  - Controlled experiments focus on small LMs (`<10B` parameters; here 0.5B) (§1, §4). Behavior for very large LMs or highly heterogeneous architectures remains an open question.
- Communication and compute overhead
  - Sharing decoded text and re‑encoding adds bandwidth and compute overhead, though the paper argues the gains in sample efficiency outweigh these costs (§2–§3, §5).
- Algorithmic details left implicit
  - While PPO/GRPO make external rollouts usable through ratio clipping, formal analysis of off‑policy bias and convergence in this decentralized setting is not provided; importance weighting or trust‑region theory for cross‑policy text data remains an open area.

## 7. Implications and Future Directions
- Field impact
  - SAPO offers a practical blueprint for collaborative RL post‑training without centralized orchestration. This can democratize RL‑based reasoning improvements by leveraging heterogeneous edge devices and volunteer compute (§1, §6).
- Practical applications
  - Organizations without large clusters can post‑train small LMs for domains with verifiable checks (math, code, data cleaning, constrained generation). Multi‑modal extensions could enable shared learning of style or “taste” when rewards encode aesthetics (§7).
- Research avenues
  - Adaptive sharing policies: learn to balance `I_n` vs. `J_n` per node and per phase; design filters based on advantage, diversity, or learned verifiers (§7).
  - Robustness and trust: reputation systems, cryptographic attestations, or incentive mechanisms to ensure high‑quality contributions in open swarms (§7).
  - Hybrid objectives: combine SAPO with RLHF, generative verifiers, or curriculum mechanisms to stabilize learning and expand beyond strictly verifiable tasks (§7; refs. to generative verifiers).
  - Heterogeneity at scale: systematic studies with mixed base models, specialized agents, and role emergence; explore whether structured roles emerge from unstructured sharing (§2, §7).
  - Theoretical analysis: characterize convergence/stability when using others’ decoded trajectories under PPO/GRPO clipping; derive principled rules for sampling and clipping thresholds in decentralized settings.

Overall, SAPO reframes distributed RL for LMs around exchanging lightweight experience rather than heavyweight weights. The controlled experiments and a real‑world demo support its core promise: collective sharing can accelerate and stabilize learning—up to a point—provided the mix of local and external data is balanced and sampling quality is maintained.
