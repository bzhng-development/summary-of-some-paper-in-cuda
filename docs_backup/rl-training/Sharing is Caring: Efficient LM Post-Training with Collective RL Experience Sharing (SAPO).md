# Sharing is Caring: Efficient LM Post-Training with Collective RL Experience Sharing (SAPO)

**ArXiv:** [2509.08721](https://arxiv.org/abs/2509.08721)
**Authors:** Jeffrey Amico, Gabriel Passamani Andrade, John Donaghy, Ben Fielding, Tristin Forbus, Harry Grieve, Semih Kara, Jari Kolehmainen, Yihua Lou, Christopher Nies, Edward Phillip Flores Nuño, Diogo Ortega, Shikhar Rastogi, Austin Virts, Matthew J. Wright
**Institutions:** 

## 🎯 Pitch

SAPO introduces a decentralized, asynchronous RL method that significantly enhances language model training by enabling nodes to share and learn from decoded experiences without central synchronization. This innovation democratizes training across heterogeneous hardware, boosting reasoning capabilities and collective intelligence while reducing traditional infrastructure costs and bottlenecks, marking a pivotal step in scalable and effective multi-agent reinforcement learning.

---

## 1. Executive Summary
This paper introduces SAPO (Swarm sAmpling Policy Optimization), a decentralized, asynchronous reinforcement-learning (RL) post‑training method where many independent nodes train their own language models but “share” experience by exchanging decoded rollouts. In controlled studies on verifiable reasoning tasks, balanced experience sharing (4 local / 4 external rollouts per update) nearly doubles cumulative reward versus standard single‑agent RL, and a large open demo shows swarm training overtakes isolated training for mid‑sized models after sufficient rounds.

## 2. Context and Motivation
- The problem: RL post‑training can substantially improve reasoning without extra labeled data, but scaling RL for language models typically requires centralized clusters that keep a single policy synchronized across many GPUs. This brings latency, memory and communication bottlenecks, fragility, and high cost (§1–§2; references to distributed RL systems in §2).
- Why it matters:
  - Practical: Many potential contributors only have heterogeneous, edge‑class hardware (e.g., laptops). A way to get RL benefits without centralized orchestration would democratize post‑training.
  - Scientific: Multi‑agent diversity can drive exploration and “Aha moments” that single policies might miss; enabling those effects during training could raise sample efficiency (§1–§2).
- Prior approaches and gaps:
  - Centralized distributed RL for LMs (e.g., weight‑synchronized PPO/GRPO in large clusters) is effective but costly and communication‑heavy (§2).
  - Multi‑agent methods often require designed roles or orchestration (debate, verifier/generator roles, etc.; §2), adding engineering complexity.
- Positioning:
  - SAPO is a bridge between single‑agent RL fine‑tuning and multi‑agent systems: it keeps each node independent yet lets nodes learn from others’ experience by sampling shared rollouts. It assumes no synchronized weights, no model homogeneity, and no timing guarantees (§1, §3).

Key terms (paper‑specific or uncommon):
- `swarm`: a decentralized network of N nodes, each training its own policy and sharing experience (§3.1).
- `rollout`: a set of completions/answers that a policy generates for one question/task (§3.1).
- `verifiable task`: a task whose answer can be checked programmatically by a `verifier`, giving a deterministic reward signal (used here via ReasoningGYM; §3.1, §4.1).
- `advantage`: in policy-gradient RL, how much better a sampled action performed than a baseline for the same context. SAPO uses it to filter out uninformative samples (zero-advantage rollouts) before training (§3.2, §5).
- `GRPO`: Group Relative Policy Optimization, a PPO‑style objective that uses a group baseline across multiple responses to the same prompt. Used here for policy updates (§4.2).

## 3. Technical Approach
At a high level, each node trains locally but augments its training data with other nodes’ experiences. Crucially, rollouts are shared as decoded text so any node can re‑encode them to compute token‑level likelihoods under its own policy—enabling policy-gradient updates without sharing weights.

Step‑by‑step (see §3.1–§3.2 and Algorithm 1):
1. Swarm setting and data
   - Each node n has:
     - A local set of verifiable tasks `D_n = {(q, y_q)}` with metadata `M_n` describing how to verify answers (§3.1).
     - Its own policy `π_n` (the LM) and a local reward function `ρ_n` (a verifier; §3.2).
   - Compatibility: rollouts must have compatible modalities (e.g., text‑only across nodes). Nodes can ignore incompatible items (§3.1).

2. Local rollout generation
   - Per training round, node n samples a batch of tasks `B_n` from its local tasks `Q_n` (§3.2).
   - For each question `q ∈ B_n`, it generates `L_n` completions `R_n(q) = {a^n_1(q), …, a^n_{L_n}(q)}`—this is the local rollout (§3.1–§3.2).

3. Sharing decoded experience
   - The node broadcasts a subset `S_n ⊆ B_n` along with tuples
     - `C_n(q) := (q, y_q, R_n(q), M_n)` (§3.2),
     - i.e., the question, ground‑truth, the decoded completions, and the metadata for verification.
   - Why decoded text matters: any other node can re‑encode these completions with its own tokenizer/model, compute log‑probabilities, and apply its policy-gradient algorithm “as if” it had generated those tokens—even if they were unlikely under its current policy (§3.2).

4. Constructing the per‑node training set
   - Each node n chooses how many items to use from its own rollouts (`I_n`) and how many to sample from the swarm (`J_n`) to form a training set `T_n` (§3.2):
     - `T_n = SampleSelf({C_n(q) | q ∈ B_n}, I_n) ∪ SampleExternal(⋃_{m≠n}{C_m(q) | q ∈ S_m}, J_n)`.
   - Sampling is fully local and customizable. In the controlled experiments (§5), nodes:
     - Drop rollouts with zero advantage,
     - Then uniformly sample `J_n` items from the remaining shared pool (§5).

5. Rewarding and updating
   - Node n computes rewards for items in `T_n` using its local verifier `ρ_n` (§3.2).
   - It updates its policy with a policy‑gradient method (here, GRPO; §4.2). Setting `J_n = 0` recovers standard single‑agent RL (§3.2).

Design choices and why:
- Share decoded rollouts instead of weights or gradients:
  - Avoids synchronization and network bottlenecks; works across heterogeneous models and hardware (§1–§3).
- Require verifiable tasks:
  - Enables decentralized, rule‑based reward computation without sharing a reward model or human‑in‑the‑loop labels (§3.1, §4.1).
- Local control over sampling/filtering:
  - Lets nodes tailor how much they rely on others, mitigate low‑quality or off‑distribution samples, and adapt to network conditions (§3.2, §5).

Concrete example (mirroring §4–§5):
- In one round, a node draws 8 ReasoningGYM tasks, generates 8 answers per task, shares those, then forms `T_n` by keeping 4 of its own tasks and 4 tasks sampled from the swarm (dropping any with zero advantage). It computes rewards with the corresponding verifiers and applies a GRPO update.

Training specifics used in the controlled experiments (§4):
- Models: eight `Qwen2.5-0.5B` SLMs, each in its own Docker container; multi‑GPU orchestration with PyTorch/NCCL, 1 GPU per agent (§4).
- Tasks: nine specialties from ReasoningGYM (e.g., base conversion, binary matrix reasoning; §4.1).
- Policy updates: GRPO with no KL penalty (following observations in DAPO), asymmetric clipping thresholds `ε_low = 0.2`, `ε_high = 0.28`, Adam lr=0.001, 2000 rounds (§4.2).
- Rewards: 1 if the verifier parses a correct answer, else 0, with rare partial credit exceptions (§4.3). A separate “formatting reward” was tested then removed because correct formatting propagated through sharing (§4.3).
- Platform: GenRL, the backend for Gensyn’s RLSwarm, which integrates ReasoningGYM and supports peer‑to‑peer coordination (§4.4).

## 4. Key Insights and Innovations
1. Decentralized, asynchronous RL via decoded experience sharing
   - Novelty: Nodes never synchronize weights or share gradients; they only exchange decoded rollouts plus minimal metadata (§3.2).
   - Why it matters: Removes central bottlenecks and cost, tolerates heterogeneity (different models/hardware), and functions under arbitrary latency (§1–§3).

2. Re‑encoding foreign rollouts for on‑policy style updates
   - Mechanism: A node re‑encodes others’ completions to compute its own token‑level log‑probs and advantages, then applies GRPO/PPO locally (§3.2).
   - Significance: Enables learning from “unlikely” but successful trajectories discovered by other policies—so “Aha moments” can propagate (§1, §5).

3. Balanced sampling of local vs. external experience is critical
   - Empirical finding: Too much reliance on external data increases oscillations and forgetting; a 50/50 split (4 local / 4 external) maximizes cumulative reward (§5; Figures 1–2).
   - Importance: Offers a principled knob for stability‑performance trade‑offs in collective RL.

4. Verifier‑driven rewards make collective RL practical at the edge
   - Using programmatic verifiers (ReasoningGYM) allows each node to score both local and external rollouts without shared reward models or humans (§4.1, §4.3).
   - Observation: Formatting conventions spread through the swarm without an explicit formatting reward, reducing reward‑engineering burden (§4.3).

Incremental vs. fundamental:
- Fundamental: The “share decoded rollouts + re‑encode for policy gradients” paradigm and the fully decentralized swarm structure.
- Incremental: Specific GRPO settings (no KL, clipping thresholds) and the particular ReasoningGYM task mix.

## 5. Experimental Analysis
Evaluation design
- Controlled study (§4–§5):
  - Setup: 8 agents (Qwen2.5‑0.5B), each round draws 8 tasks (from 9 ReasoningGYM specialties), generates 8 completions per task (§4.1–§4.2).
  - Sharing configurations (I local / J external with I + J = 8): baseline 8/0; and 6/2, 4/4, 2/6 (§5).
  - Sampling rule: discard zero‑advantage shared rollouts; uniformly sample the rest (§5).
  - Metric: per‑round rewards and cumulative rewards across agents; also moving‑average “expected reward” with window=100 (§5, Figure 2).
- Large open demo (§6):
  - Thousands of heterogeneous nodes; a “judge” issues random ReasoningGYM questions after each round and scores pass@1 via verifiers (§6).
  - Metric: cumulative and per‑round performance vs. an isolated baseline, aligned by normalized rounds to handle intermittent participation (§6, Figure 3).

Main quantitative results
- Controlled experiments (Figure 1–2; §5):
  - Cumulative reward across agents and rounds:
    - 4 local / 4 external: 1093.31
    - 2 local / 6 external: 945.87
    - 6 local / 2 external: 854.43
    - Baseline 8 / 0: 561.79
  - The 4/4 configuration “yields a 94% improvement over the baseline” (Figure 1 caption; §5).
  - Expected average reward over training (moving average, Figure 2): 4/4 consistently beats baseline and usually beats 6/2; it also surpasses 2/6 for most rounds, though by a smaller margin (§5).
  - Dynamics: Heavier external reliance (2/6) shows stronger oscillations, interpreted as (i) high‑performers polluted by weaker agents and (ii) overall pool quality dropping when many sample but few contribute (§5).
- Large demo (Figure 3; §6):
  - For `Qwen2.5-0.5B`, swarm training significantly outperforms isolated training after about 175 normalized rounds:
    > “After a certain number of rounds, in this case approximately 175, the performance per round of the models in the swarm significantly exceeds that of the model trained in isolation.” (Figure 3 caption; §6)
  - For stronger `Qwen3-0.6B`, performance with and without the swarm is similar (§6).
  - Note: In this demo, nodes sampled shared rollouts uniformly without filtering, likely over‑representing low‑value samples (§6).

Do the experiments support the claims?
- The controlled study convincingly shows that sharing improves both peak and cumulative reward, with a clear optimum around balanced sharing (Figures 1–2). The moving‑average analysis in Figure 2 supports that these gains are not mere noise.
- The large demo indicates the effect persists at scale and in heterogeneous conditions for mid‑sized models, though benefits appear model‑capacity dependent and sensitive to sampling strategy (§6).

Ablations, failure modes, robustness
- Explicit ablations:
  - Sharing ratio ablation (8/0, 6/2, 4/4, 2/6) and the zero‑advantage filtering rule (§5).
- Observed failure/instability:
  - Heavy external reliance causes oscillations and forgetting (§5, Figure 2).
- Not present:
  - No systematic ablation of alternative sampling/weighting strategies for shared rollouts, or comparisons of RL algorithms beyond GRPO (noted in §4.2 and §7 as future work).
  - No communication‑overhead measurements, though the method communicates text rather than weights (§2 acknowledges overhead but argues fewer rounds offset it).

## 6. Limitations and Trade-offs
Assumptions
- Verifiable tasks: SAPO’s experiments require tasks with programmatic verifiers to compute rewards locally (§3.1, §4.1). Pure preference‑based or hard‑to‑verify objectives would need different mechanisms (e.g., reward models or generative verifiers).
- Compatible modalities: Nodes must share rollouts in modalities other nodes can process; otherwise items are ignored (§3.1).

Scenarios not fully addressed
- Human‑preference alignment (RLHF) and open‑ended tasks without deterministic checks are not evaluated; integration is suggested as future work (§7).
- Security/trust in large swarms: The method assumes nodes can filter low‑quality samples, but robustness to malicious or systematically biased contributions is not studied (§7 hints at trust-aware sampling).

Computational and system trade‑offs
- Communication and re‑encoding overhead:
  - The system sends decoded rollouts and metadata; each node re‑encodes them locally (§2). While lighter than weight synchronization, it still adds overhead that grows with swarm size.
- Stability vs. sharing ratio:
  - More external data improves exploration but can destabilize learning (oscillations/forgetting) if it overwhelms local experience (Figures 1–2; §5).
- Model‑capacity dependence:
  - In the open demo, a stronger SLM (`Qwen3-0.6B`) did not benefit over isolation under naive uniform sampling (§6), suggesting benefits may depend on model capacity and sampling quality.

Open questions
- How to optimally balance local vs. external data online?
- How to design robust, trust‑aware sampling/weighting in open swarms?
- How does SAPO perform on larger LMs, other domains (e.g., code), or non‑verifiable objectives?

## 7. Implications and Future Directions
How this changes the landscape
- Makes RL post‑training accessible without centralized infrastructure: sharing decoded experience rather than weights lets heterogeneous, edge devices collaborate (§1–§3).
- Provides a practical path to capture multi‑agent exploration benefits without orchestrating roles or synchrony. Balanced sharing emerges as a simple, effective recipe (Figures 1–2).

What it enables next
- Smarter sampling from the swarm:
  - Reward‑guided filters, trust scores, per‑peer reliability, or curriculum‑style selection to avoid the oscillations seen with heavy external reliance (§5). The demo’s uniform sampling likely underestimates SAPO’s potential (§6).
- Hybrid training:
  - Combine SAPO with RLHF or generative verifiers to cover non‑verifiable objectives and richer reward signals (§7; refs. to generative verifiers).
- Adaptive controllers:
  - Meta‑policies that tune `I_n` vs. `J_n` per node based on performance, variance, or peer quality (§7).
- Heterogeneous/multi‑modal swarms:
  - Specialize nodes by domain or modality (text, code, images) and study cross‑modal influence; the paper notes intriguing effects when “taste” or aesthetics become rewards (§7).
- Human‑in‑the‑loop participation:
  - Since any “policy” can contribute rollouts, humans or other non‑LM policies could seed high‑value examples, provided incentives and verification exist (§3.1 note; §7).

Practical applications
- Reasoning‑centric SLMs on consumer hardware (e.g., educational tools, local assistants) that improve collaboratively.
- Federated‑style enterprise training where data cannot be centralized but verifiable tasks exist.
- Open research platforms (like Gensyn’s RLSwarm/GenRL) to study collective learning dynamics at scale (§4.4, §6).

Quoted highlights grounded in the paper:
- “Balanced experience sharing (4 local / 4 external) … yields a 94% improvement over the baseline.” (Figure 1 and §5)
- “The 4 local / 4 external configuration consistently achieves higher expected average reward than the baseline” (Figure 2; §5).
- “After approximately 175 normalized rounds, the performance per round of the models in the swarm significantly exceeds that of the model trained in isolation.” (Figure 3; §6)

In sum, SAPO’s core mechanism—learning from others’ decoded experience by re‑encoding and applying local policy gradients—is both simple and powerful. The controlled and at‑scale results show clear gains when sharing is balanced and filtered, and the framework opens a wide space for principled sampling, robustness, and hybrid reward research.
