# DeepSearch: Overcome the Bottleneck of Reinforcement Learning with Verifiable Rewards via Monte Carlo Tree Search

**ArXiv:** [2509.25454](https://arxiv.org/abs/2509.25454)
**Authors:** Fang Wu, Weihao Xuan, Heli Qi, Ximing Lu, Aaron Tu, Li Erran Li, Yejin Choi
**Institutions:** Stanford University, University of Tokyo, RIKEN AIP, University of Washington, UC Berkeley, Amazon AWS, Columbia University

## 🎯 Pitch

DeepSearch revolutionizes reinforcement learning for reasoning models by integrating Monte Carlo Tree Search into training, facilitating comprehensive exploration and overcoming sparse data challenges. This innovation achieves state-of-the-art accuracy across multiple math benchmarks while significantly reducing computational requirements, marking a transformative step in efficient model training and reasoning tasks.

---

## 1. Executive Summary (2-3 sentences)
DeepSearch integrates Monte Carlo Tree Search (MCTS) directly into reinforcement learning with verifiable rewards (RLVR) to fix a core bottleneck: sparse and myopic exploration during training. By learning from structured search rather than only from final outcomes or inference-time search, it achieves state-of-the-art accuracy for 1.5B-parameter reasoning models (62.95% average across six math benchmarks; Table 1) while using 5.7× fewer GPU hours than prolonged training (Table 2).

## 2. Context and Motivation
- Problem addressed:
  - RLVR trains models using a reward that can be automatically checked (a “verifier”), e.g., whether a math answer is correct. Despite its promise for reasoning, RLVR often plateaus after thousands of steps—performance gains diminish even as training compute grows (Introduction; Section 1). The paper identifies the cause as sparse, unsystematic exploration during training: limited rollouts miss important reasoning paths and provide weak supervision.
- Why it matters:
  - Reasoning tasks (math, science, planning) benefit from exploring many intermediate steps before the final answer. If training doesn’t explore, models overfit to shallow patterns and fail to generalize—even if we spend more compute. Improving training-time exploration unlocks better reasoning at lower cost.
- Prior approaches and shortcomings:
  - Search at inference only: Techniques like Tree-of-Thoughts and MCTS variants are commonly used at test time to improve accuracy by exploring multiple reasoning paths (Related Works A; cited works in Section 1). But they do not shape the policy during training—so the model doesn’t learn to search.
  - RLVR with prolonged steps: Simply running more RL steps yields diminishing returns and plateaus (Section 1; Section 4.2, Table 2).
  - Process supervision without search: Process-level rewards help, but still rely on limited rollouts and don’t systematically cover the solution space.
- Positioning:
  - DeepSearch moves search from inference to training. It couples MCTS with verifiable rewards during training so the policy learns from the exploration process itself (Section 2, Figure 1). It emphasizes “scaling training breadth” (better exploration) rather than “scaling training depth” (more steps).

## 3. Technical Approach
At a high level, for each problem `x`, DeepSearch constructs a search tree over reasoning steps using the current policy `π_θ`. It evaluates terminal nodes with a verifier `V` and uses the results to update node scores (`q-values`) and to select the next nodes to expand. The trajectories extracted from the tree supervise the policy via a PPO-style objective tailored to tree structures (“Tree-GRPO”). An adaptive training loop focuses MCTS compute on hard problems and caches solutions to avoid redundant search (Algorithm 1; Figure 1).

Step-by-step (Sections 2–3; Algorithm 1):
1) Build a reasoning tree with MCTS
- Nodes represent partial solutions (intermediate reasoning steps). The root is the question `x`. A path from root to a terminal node is a trajectory `t = x ⊕ s1 ⊕ … ⊕ s_end` (Section 2; notation before Eq. 1).
- Expansion: given the current state (the concatenation of steps so far), the policy `π_θ` proposes `n` next-step candidates `{s_{i,j}}` (Section 2.1), and expansion proceeds until terminal nodes `S_end` are reached, either because an answer is produced or the maximum depth `d_T` is hit.

2) Verify terminal nodes and select supervision examples
- Verification partitions terminal nodes into correct and incorrect subsets:
  - Eq. (1): `S_correct^(k) = {s ∈ S_end^(k) | V(s) = 1}`, `S_incorrect^(k) = {s ∈ S_end^(k) | V(s) = 0}`.
- If any correct solution exists, the correct trajectories are extracted for training (Algorithm 1, lines 26–29).
- If none is correct, choose the “most confident wrong” trajectory using entropy:
  - Eq. (2)–(3): pick the terminal node whose root-to-leaf path has the lowest average entropy of the policy distribution. This targets confidently wrong reasoning for targeted correction (“entropy-based guidance,” Section 2.1).

3) Back up scores (q-values) along the selected trajectory
- Terminal rewards: +1 for correct, −1 for incorrect or incomplete (Eq. 6).
- Heuristic backup with temporal decay:
  - Eq. (4)–(5): update a node’s `q` by adding a decayed portion of the terminal reward, weighting nodes nearer to the end more heavily.
- Constrained update to preserve positive credit on correct paths:
  - Eq. (7) ensures nodes on correct trajectories remain non-negative while penalizing nodes on incorrect/incomplete paths (Section 2.2). This yields fine-grained credit assignment across steps.

4) Choose the next node to expand with a hybrid selection strategy
- Local (sibling) selection: when comparing siblings under one parent, use UCT to balance exploitation and exploration:
  - Eq. (8): `UCT(s) = Q(s) + λ sqrt(ln N_parent(s) / N(s))`.
- Global frontier selection: instead of repeatedly traversing from the root, compare all frontier leaves `F` across the entire tree and select the single best node to expand next (Eq. 9–10):
  - Frontier score (Eq. 10): `F(s) = λ1 · tanh(Q_parent(s)) + λ2 · H(π_θ(s|o)) + λ3 · D(d(s))`
    - Quality potential: encourages nodes whose parents already look promising (`tanh(Q_parent)`).
    - Uncertainty bonus: uses policy entropy to steer exploration toward high- or low-uncertainty regions depending on the sign of `λ2`.
    - Depth bonus: e.g., `D(d(s)) = sqrt(d(s)/d_T)` encourages exploring deeper parts of the tree without overcommitting (Section 2.3).

5) Build the training set and optimize the policy (“Tree-GRPO”)
- Soft clipping prevents q-value explosion for intermediate nodes while preserving gradients:
  - Eq. (16): `q(s) = tanh(q^(kmax)(s)/ε_q) · q_max`.
- PPO-style objective over tokens along tree trajectories, with “Clip-Higher” bounds and no KL term (Section 3.3; Eq. 17). Advantages are “mean-only” normalized:
  - Eq. (18): `Â_{j,k} = q(s_j) − μ_t`, where `μ_t` is the mean terminal-node reward in the same tree. This combats length growth and miscalibration without variance scaling (Table 4 notes; Section 3.3).

6) Adaptive training loop with progressive filtering and replay buffer (Section 3; Algorithm 1)
- Progressive filtering: focus MCTS on hard problems only.
  - Build `D_hard^(0)` by evaluating `Pass1@K(x, π_θ^(0))` on the full training set and keeping items below a threshold (Eq. 11; typically δ = 25%). Here `Pass1@K` means whether at least 1 out of K samples solves the problem (K=4).
  - After each policy update, re-evaluate and shrink to the next `D_hard^(i+1)` (Eq. 12).
- Replay buffer with cached solutions: store correct trajectories found by MCTS for unsolved-hard items (Eq. 13). On future passes, if a cached solution exists, bypass MCTS and use:
  - Eq. (14)–(15): `tcached` plus a few direct rollouts from the current policy; otherwise run full MCTS. This preserves solved knowledge and saves compute for future iterations.
- Practical safeguards: remove garbled/infinite-repetition samples among incorrect outputs to avoid training collapse (Section 3.2).

Implementation details (Appendix B):
- MCTS depth up to 64; per-node token budget 256; expansion width 8; UCT exploration λ=2.0 (B.2).
- Frontier depth bonus uses `sqrt(d(s)/d_T)` with `λ3=0.01` (Table 3 and B.2).
- Optimizer uses AdamW; low learning rate (`1e-6`), clip range asymmetry (“Clip-Higher”), no KL penalty; maximum context length 18,432 tokens; overlong buffer penalty at 4,096 tokens (B.3–B.4).

Intuition:
- Think of training as teaching the model to conduct its own search. The tree lets it explore multiple partial solutions; the verifier tells which leaves are correct; the constrained backup and Tree-GRPO translate that into graded feedback for all steps along promising or confidently-wrong paths. Global frontier selection allocates search budget to the best next leaf anywhere in the tree, rather than re-traversing from the root.

## 4. Key Insights and Innovations
- Training-time search, not just inference-time search (fundamental)
  - Most prior “search” methods use test-time compute to boost accuracy. DeepSearch directly embeds MCTS into RLVR training (Figure 1; Algorithm 1), so the policy learns from exploration itself. This expands training coverage and improves credit assignment—addressing the plateau caused by sparse exploration.
- Global frontier selection across the whole tree (fundamental)
  - Instead of UCT’s repeated root-to-leaf traversals, DeepSearch scores all frontier leaves and expands the best one globally (Eq. 9–10). Table 3 shows fewer iterations (209.6 → 187.7) and better trajectory rewards (−0.82 → −0.65) than vanilla UCT while keeping depth/entropy similar.
- Entropy-guided selection of “most confident negatives” (incremental but impactful)
  - When no correct leaf exists, use the trajectory with the lowest average entropy for supervision (Eq. 2–3). This zeroes in on confidently wrong reasoning—high-value targets for correction (Section 2.1).
- Constrained backup rule that keeps correct-path nodes non-negative (incremental)
  - The update rule (Eq. 7) enforces that nodes on correct paths accumulate positive credit, while nodes leading to incorrect solutions can become negative. This creates clear step-level signals (Section 2.2).
- Adaptive training with replay buffer and solution caching (incremental but practical)
  - Progressive filtering (Eq. 11–12) focuses compute on hard problems; cached correct trajectories avoid repeated MCTS for previously solved items (Eq. 13–15). This is central to DeepSearch’s 5.7× compute savings (Table 2).
- Tree-GRPO with q-value soft clipping and mean-only normalization (incremental)
  - Soft clipping (Eq. 16) controls q-value explosion while keeping gradients; mean-only normalization (Eq. 18) stabilizes advantages without variance scaling, which the ablation identifies as effective for reasoning stability (Table 4).

## 5. Experimental Analysis
- Evaluation setup:
  - Base model: `Nemotron-Research-Reasoning-Qwen-1.5B v2` (Section 4.1; Appendix B).
  - Training data: DeepMath-103K, a challenging, decontaminated math dataset with verifiable answers (Section 4.1).
  - Benchmarks: Six math reasoning sets—AIME 2024, AIME 2025, AMC 2023, MATH500, Minerva, Olympiad (Table 1).
  - Metric: Pass@1 accuracy using 32 samples per problem; evaluations on a 128×H100 (96GB) cluster (Table 1 footnote).
  - Baselines: modern 1.5B math reasoners, including RL-based (DeepScaleR, Nemotron variants), search-based (Qwen2.5-Math-Oat-Zero), and strong distilled or instruction-tuned baselines (Section 4.1; Table 1).
- Main results (Table 1):
  - DeepSearch-1.5B achieves the best average Pass@1 across six benchmarks: 
    > “DeepSearch-1.5B … Avg 62.95”  
    outperforming the prior best `Nemotron-Research-Reasoning-Qwen-1.5B v2` at 61.70.
  - Notable per-dataset improvements:
    - AIME 2024: 53.65 vs 51.77 (+1.88)
    - AMC 2023: 90.39 vs 88.83 (+1.56)
    - Gains are consistent across all six datasets (Section 4.1).
- Efficiency and scaling (Table 2; Figure 2):
  - Prolonged DAPO training shows diminishing returns:
    > “+1875 steps … 62.02 … 1883.2 GPU hours”  
    while a smaller extension  
    > “+325 steps … 61.78 … 326.4 GPU hours”.
  - DeepSearch reaches higher accuracy with far fewer extra steps:
    > “Tree-GRPO +50 … 62.95 … 330 GPU hours.”  
    That is 5.7× fewer GPU hours than the 1,883.2-hour prolonged baseline, yet better accuracy (Section 4.2).
  - Training dynamics over 20 hours after 3K RLVR steps (Figure 2): DeepSearch’s trend line rises faster than DAPO’s, supporting the claim that exploration quality, not just time, is the bottleneck.
- Search strategy ablations (Table 3):
  - Global frontier selection vs UCT: fewer iterations (−10.4%), improved reward quality (−0.65 vs −0.82), similar depth/entropy.
  - Depth bonus forms:
    - Linear `d(s)` explores deepest but hurts quality (reward −0.76).
    - `sqrt(d(s)/d_T)` balances efficiency and quality best (adopted default).
  - Adding uncertainty bonus (λ2=0.4) increases exploration diversity (entropy 1.23 → 1.31) but with higher variance in iterations.
- Algorithm evolution ablation (Table 4):
  - A careful build-up is necessary. A “vanilla” DeepSearch variant underperforms the Nemotron v2 baseline (60.27 vs 61.70). Gains arrive after:
    - Switching to the constrained q-update and node-level advantages (61.85),
    - Mean-only normalization (62.32),
    - Global frontier selection (final 62.95).
  - This shows the final performance relies on the full stack: modern backup, fine-grained token-level credit, stable normalization, and global selection.
- Do the experiments support the claims?
  - Yes—on both accuracy and compute efficiency. The SOTA result (Table 1) is modest but consistent across datasets; the compute savings (Table 2) are substantial. Ablations isolate the contribution of each component (Tables 3–4), and training dynamics (Figure 2) match the exploration hypothesis.

## 6. Limitations and Trade-offs
- Dependence on verifiable rewards:
  - RLVR needs a reliable `V` to decide correctness (Section 2.1). DeepSearch is demonstrated on math where verification is objective. Extending to subjective tasks requires new verifiers (Limitations and Future Work).
- Compute profile and complexity:
  - Although more efficient than prolonged training, DeepSearch still requires MCTS during training on a multi-GPU setup (Appendix B: 16×H100 for training; 128×H100 for eval). Implementing and tuning MCTS (depth, width, entropy estimates, bonus weights) adds system complexity.
- Hyperparameter sensitivity:
  - Frontier weights (`λ1, λ2, λ3`), depth bonus form, decay `γ`, clipping scales, and filter thresholds (δ, typically 25%) are crucial (Sections 2–3; Table 3). Portability to new domains may require careful retuning.
- Reward shaping assumptions:
  - The constrained backup assumes that nodes on correct paths should always have non-negative q-values (Eq. 7). In tasks with partial credit or multi-solution ambiguity, this assumption might be too rigid.
- KL-free optimization and overconfidence:
  - The training removes KL regularization to “naturally diverge” (Section 3.3), which can increase overconfidence. The paper mitigates instability via mean-only normalization and clipping (Eq. 16–18; Table 4), but the generalization of this recipe to broader domains remains an open question.
- Scope:
  - Results focus on a 1.5B model and mathematical reasoning. How performance and efficiency scale to larger models or non-math domains is not yet established.

## 7. Implications and Future Directions
- Field impact:
  - DeepSearch reframes “scaling reasoning” from longer RL training to richer training-time exploration. This suggests a new baseline: train models to search, not just search at inference. If widely adopted, RLVR research may invest more in training-time exploration algorithms (global selection, uncertainty-aware supervision) than in pushing more steps or parameters.
- Follow-up research enabled:
  - Approximate or learned verifiers for less objective domains (ethics, rubrics, preference modeling), possibly combining “rubrics as rewards” with MCTS.
  - Process-aware reward models integrated with MCTS: combining outcome verification with step-level evaluators could further refine credit assignment.
  - Adaptive test-time compute that leverages training-time search signals (e.g., learned heuristics from `Q_parent`, entropy, and depth features).
  - Transfer beyond math: program synthesis, scientific problem solving, robotics planning—where intermediate reasoning can be structured as trees and partially verifiable.
- Practical applications:
  - More efficient tutoring systems and auto-graders for math and STEM subjects.
  - Systems that maintain higher accuracy under fixed compute budgets by investing compute where it matters (hard problems) during both training and deployment.
  - Model distillation: use DeepSearch-trained teachers to produce high-quality, diverse reasoning traces for smaller students.

Quotes and pointers for quick reference:
- Accuracy: 
  > “DeepSearch-1.5B … Avg 62.95” across AIME 2024/2025, AMC 2023, MATH500, Minerva, Olympiad (Table 1).
- Efficiency:
  > “Tree-GRPO +50 … 62.95 … 330 GPU hours” vs  
  > “+1875 … 62.02 … 1883.2 GPU hours” (Table 2).
- Global frontier selection score:
  > `F(s) = λ1·tanh(Q_parent(s)) + λ2·H(π_θ(s|o)) + λ3·D(d(s))` (Eq. 10).
- Entropy-guided negative selection:
  > pick `s*_neg = argmin_s H̄(t(s))` with average trajectory entropy `H̄(t(s))` (Eq. 2–3).
- Constrained backup:
  > Terminal rewards ±1; intermediate updates with decay and sign constraints (Eq. 4–7).
- Tree-GRPO with soft clipping and mean-only normalization:
  > `q(s)=tanh(q/ε_q)·q_max`; objective in Eq. (17) with advantages `Â=q(s)−μ_t` (Eq. 16–18).

Overall, DeepSearch shows that smarter exploration during training—implemented via MCTS with global frontier selection, entropy-guided supervision, and efficient replay—can unlock better reasoning with less compute than simply training longer.
