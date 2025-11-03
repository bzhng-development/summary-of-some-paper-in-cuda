# DAPO: An Open-Source LLM Reinforcement Learning System at Scale

**ArXiv:** [2503.14476](https://arxiv.org/abs/2503.14476)

## 🎯 Pitch

DAPO introduces a fully open-source, scalable reinforcement learning recipe for training large language models to excel at complex reasoning—specifically, long chain-of-thought generation. By combining a novel PPO-inspired objective with four tailored training techniques, this system achieves state-of-the-art results (50/100 on AIME 2024) with half the compute of prior open baselines and provides the open code, data, and logs needed for true reproducibility, lowering the barrier for researchers and the industry to develop next-generation reasoning LLMs.

---

## 1. Executive Summary
- The paper introduces DAPO, a practical reinforcement learning (RL) recipe for training reasoning-focused large language models (LLMs) at scale. It couples a new PPO-style objective with four training techniques tailored to long chain-of-thought (CoT) generation.
- Using a 32B-parameter base model (`Qwen2.5-32B`), DAPO reaches 50/100 on AIME 2024 with half the training steps of the prior open baseline and provides open-source code, data, and logs for reproducibility (Figure 1; Table 1).

## 2. Context and Motivation
- Problem/gap:
  - State-of-the-art “thinking” models (e.g., OpenAI o1, DeepSeek R1) rely on RL and test-time scaling (longer CoT), but crucial training details remain undisclosed. As a result, community attempts using standard algorithms like GRPO or PPO often underperform or collapse.
  - In the authors’ own baseline, naïve GRPO on `Qwen2.5-32B` yields only 30 AIME 2024 points—far below DeepSeek’s reported 47 (Section 1; Table 1).
- Why it matters:
  - Practical: Reasoning LLMs power math and coding assistants. Reliable, reproducible training pipelines lower the barrier to building strong, domain-specialized models.
  - Scientific: Clarifies which RL ingredients actually drive long-CoT success (exploration vs. exploitation balance, credit assignment over long sequences, reward design for truncation).
- Prior approaches and their shortcomings:
  - PPO (Section 2.1; Eq. (1)) uses a clipped objective and a value function; GRPO (Section 2.2; Eqs. (4–6)) removes the value network and normalizes rewards within a sampled group.
  - Issues observed in long-CoT training with these baselines:
    - Entropy collapse (policy quickly becomes overconfident, exploring less; Figure 2b).
    - Gradient vanishing when all responses in a group are equally rewarded (accuracy 0 or 1; Section 3.2; Figure 3b).
    - Misweighted updates for long responses due to sample-level loss reduction (Section 3.3; Figure 4).
    - Reward noise from how truncated, overlong samples are penalized (Section 3.4; Figure 5).
- Positioning:
  - DAPO is not a new general RL paradigm; it is a domain-specific refinement of GRPO-style policy gradient for long-CoT. The novelty is a coherent suite of mechanisms that, together, stabilize exploration, fix gradient pathologies, and improve reward shaping—and all components are implemented and released (Algorithm 1; Section 4.1 for reproducible hyperparameters).

## 3. Technical Approach
At a high level, DAPO keeps the GRPO flavor (group sampling, group-relative advantages) but reworks four parts of the training loop. The core loop (Algorithm 1) is:

- Sample a batch of prompts `q` (with ground-truth final answers `a`).
- Freeze the current model as `π_old`. For each `q`, sample a group of `G` candidate responses `{o_i}` from `π_old`.
- Compute rewards for each `o_i` using a rule-based, verifiable critic: `+1` if the final answer matches, `-1` otherwise (Section 2.4; Eq. (7)).
- Filter samples and form an effective training set (Dynamic Sampling; Section 3.2).
- Compute token-level advantages using group-wise z-scores (Eq. (9), from Eq. (4)).
- Update the policy `π_θ` by maximizing the DAPO objective (Eq. (8)/(10)/(12)) with decoupled clipping thresholds.

Key components with mechanisms:

1) Objective and importance ratio
- For each token `t` in each sampled response `o_i`, compute an importance ratio
  - `r_{i,t}(θ) = π_θ(o_{i,t} | q, o_{i,<t}) / π_old(o_{i,t} | q, o_{i,<t})` (Eq. (9)).
- Compute a group-relative advantage per token (z-scored within the `G` samples for the same prompt):
  - `Â_{i,t} = (R_i - mean({R_j})) / std({R_j})` (Eq. (9); reward `R_i` is the sample-level outcome reward).
- Optimize a PPO-style clipped objective at the token level:
  - `min(r_{i,t} Â_{i,t}, clip(r_{i,t}, 1-ε_low, 1+ε_high) Â_{i,t})` summed over all tokens, then normalized by total tokens in the batch (Eq. (8)/(10)/(12)).
- KL penalty is removed (Section 2.3) to avoid constraining the policy to stay near the base model when long-CoT reasoning requires substantial distribution shift.

2) Clip-Higher (Section 3.1; Eq. (10))
- Problem: Standard symmetric clipping (ε up=ε down) constrains increases in probability for low-probability “exploration” tokens much more than it constrains increases for already high-probability “exploitation” tokens, biasing the policy toward premature determinism.
- Mechanism: Decouple the clipping range into asymmetric `ε_low` (for decreases) and a larger `ε_high` (for increases): `clip(r, 1-ε_low, 1+ε_high)`.
  - Intuition: Raise the ceiling on how much the policy can upweight unlikely but potentially useful tokens while keeping the floor (downward clip) conservative to avoid probability mass collapsing to zero.
- Evidence:
  - With `ε_low=0.2` and `ε_high=0.28` (Section 4.1), training maintains higher entropy and better accuracy (Figure 2a/b).
  - The “mean up-clipped probability” is low (<0.2), meaning clipping often affects low-probability tokens (Figure 3a), consistent with the exploration rationale.

3) Dynamic Sampling (Section 3.2; Eq. (11); Algorithm 1 lines 6–8)
- Problem: In GRPO, if all `G` responses for a prompt have identical rewards (all correct or all wrong), their normalized advantages become zeros, yielding zero gradients for that prompt. As training improves, the fraction of “all-correct” prompts grows (Figure 3b), so an increasing share of the batch contributes no learning signal.
- Mechanism: Over-sample and filter out prompts whose groups have accuracy exactly 0 or 1, so that each included prompt has a mix of correct and incorrect responses (`0 < #correct < G`), guaranteeing non-zero gradients. A dynamic sampling buffer accumulates valid samples before an update (Algorithm 1 line 6–9).
- Practical note: Although this increases generation attempts, the authors argue generation time is dominated by long-tail samples anyway, and training converges faster in steps with Dynamic Sampling (Figure 6).

4) Token-Level Policy Gradient Loss (Section 3.3; Eq. (12))
- Problem: GRPO reduces loss at the sample level (average tokens within a sample, then average across samples). Long responses thus dilute token-level credit and can silently amplify bad patterns (gibberish/repetition).
- Mechanism: Aggregate loss over tokens directly (normalize by total tokens across the batch, not per-sample first). This gives long responses proportional influence and penalizes low-quality long continuations appropriately, while fairly rewarding genuinely useful long reasoning.
- Evidence: Token-level loss keeps entropy growth and response length in a healthier range (Figure 4a/b) and adds modest but real performance/stability (Table 1: +1 point).

5) Overlong Reward Shaping (Section 3.4; Eq. (13); Figure 5)
- Problem: In long-CoT settings, some generations hit the max length and are truncated. Naively punishing all truncated samples injects noise, because an otherwise good solution might simply exceed the limit.
- Two-step fix:
  - Overlong Filtering: Mask losses for truncated samples (no gradient) to stabilize training. Improves AIME accuracy and avoids entropy spikes (Figure 5a/b).
  - Soft Overlong Punishment: Add a length-aware penalty near the length cap:
    - `R_length(y) = 0` if `|y| ≤ L_max − L_cache`
    - `R_length(y) = ((L_max − L_cache) − |y|) / L_cache` if `L_max − L_cache < |y| ≤ L_max` (a gentle, more negative penalty as one approaches the cap)
    - `R_length(y) = −1` if `|y| > L_max` (Eq. (13))
  - This penalty is added to the correctness reward, nudging the model away from excessively long outputs without indiscriminately punishing all long-but-correct reasoning.

6) Rewards and data formatting (Sections 2.4 and 3.5; Appendix A)
- Reward model: Rule-based, verifiable final-answer check for math: `+1` if `is_equivalent(ŷ, y)` else `−1` (Eq. (7)); avoids reward-model overoptimization.
- Data: To make verification robust, answers are transformed to integers (e.g., reformulate a problem so the target is `a+b+c` if the original answer is `(a + √b)/c`). The curated `DAPO‑Math‑17K` set contains 17k integer-answer prompts (Section 3.5; Appendix A shows an example transformation pipeline with “thinking scaffolds”).

Implementation snapshot (Section 4.1; Algorithm 1):
- Base model: `Qwen2.5-32B`.
- Framework: `verl`.
- LR: 1e−6 with warm-up over 20 rollout steps. Prompt batch size 512. `G=16` samples per prompt.
- Token budget: `L_max = 16,384` with `L_cache = 4,096` (total generation cap 20,480 tokens).
- Clipping: `ε_low = 0.2`, `ε_high = 0.28`.
- Evaluation: AIME 2024 with 32 repeated runs, report `avg@32`. Decoding: temperature 1.0, top‑p 0.7.

## 4. Key Insights and Innovations
- Clip-Higher: asymmetrically relax the upward clip bound (Eq. (10))
  - Why novel/significant: Instead of the standard symmetric clip, DAPO explicitly raises the “ceiling” for increasing probabilities, which directly targets entropy collapse in long-CoT RL. Figure 2 shows higher, sustained entropy and better AIME accuracy; Figure 3a confirms clipping mostly affects low-probability tokens, so it’s truly encouraging exploration. This is a conceptually simple but impactful tweak.

- Dynamic Sampling: guarantee mixed-reward groups for gradient signal (Eq. (11), Algorithm 1)
  - Why novel/significant: Addresses a GRPO-specific blind spot—once many prompts are “too easy,” their gradients vanish. By filtering groups with all-correct or all-wrong outcomes, every update carries information. Figure 6 shows faster training progress; Table 1 shows the largest single jump (+8 points when added last).

- Token-level policy gradient reduction (Eq. (12))
  - Why it matters: Long-CoT updates require fair token-level credit assignment. This change curbs low-quality length inflation (Figure 4) and adds stability and a measurable gain (+1, Table 1). It’s an incremental but thoughtful correction to a widely used reduction scheme.

- Overlong reward shaping (masking + soft penalties; Eq. (13))
  - Why significant: Truncation is ubiquitous in long-CoT tasks but often mishandled. The two-stage approach removes noisy gradients (Figure 5b) while still discouraging pathological verbosity, yielding consistent performance gains (+6 total from Overlong Filtering and +3 more from Soft Punishment in Table 1).

- Open, end-to-end system
  - The released training code, metrics, and `DAPO‑Math‑17K` dataset (Section 1; Appendix A) fill a reproducibility gap, enabling others to inspect and reuse the exact RL recipe.

## 5. Experimental Analysis
- Evaluation setup (Section 4.1):
  - Task/domain: Math reasoning, verified by exact final answers.
  - Dataset: Training on curated `DAPO‑Math‑17K` (integer answers); evaluation on AIME 2024.
  - Metric: AIME accuracy averaged over 32 sampling runs (`avg@32`).
  - Baselines: Naïve GRPO; external reference line DeepSeek‑R1‑Zero‑Qwen‑32B (47 points).
- Main quantitative results:
  - Overall progress curve (Figure 1):
    > Using `Qwen2.5-32B` base, DAPO reaches 50 AIME points while requiring only ~50% of the training steps used by DeepSeek‑R1‑Zero‑Qwen‑32B.
  - Ablations (Table 1, incremental):
    > Naïve GRPO: 30 → +Overlong Filtering: 36 → +Clip-Higher: 38 → +Soft Overlong Punishment: 41 → +Token-level Loss: 42 → +Dynamic Sampling (full DAPO): 50.
    - Interpretation: Every component contributes; the largest single boost comes from Dynamic Sampling (+8 when added last). Overlong handling accounts for +9 combined.
  - Mechanism-level evidence:
    - Clip-Higher improves accuracy and prevents entropy collapse (Figure 2a/b).
    - Up-clipped tokens tend to be low-probability (exploratory) ones (Figure 3a).
    - Prompts with group accuracy 1 increase over time (Figure 3b), motivating Dynamic Sampling.
    - Token-level loss restrains unhealthy entropy/length growth (Figure 4a/b).
    - Overlong Filtering stabilizes both performance and entropy (Figure 5a/b).
    - Dynamic Sampling speeds convergence in steps (Figure 6).
  - Training dynamics (Section 4.3; Figure 7):
    - Mean response length increases but with plateaus/dips (Figure 7a); reward steadily rises yet may not correlate tightly with validation accuracy (overfitting risk; Figure 7b discussion).
    - Entropy shows a controlled, slow upward trend after Clip-Higher (Figure 7c).
- Do the experiments support the claims?
  - For the targeted domain (math with verifiable answers), yes: there is a clear, multi-figure diagnostic story linking each technique to concrete pathologies and improvements, plus a top-line benchmark gain on AIME 2024.
  - The ablation table (Table 1) is informative and ordered to expose marginal contributions.
  - Case studies indicate emergent reflective behaviors during training (Section 4.4; Table 2 and Table 3), aligning with the goal of eliciting complex reasoning patterns.
- Caveats in evidence:
  - Breadth: Results focus on a single public benchmark (AIME 2024) and a single base model (Qwen2.5‑32B).
  - Variance: While evaluation repeats 32 times for stability, run-to-run training variance and sensitivity to hyperparameters are not deeply quantified.
  - Cost: Training-time and hardware budgets are not reported in detail; Dynamic Sampling increases generation attempts.

## 6. Limitations and Trade-offs
- Domain dependence of reward:
  - The rule-based reward (Eq. (7)) works when answers are cheaply verifiable (math, programming). Generalization to open-ended tasks (e.g., long-form QA) requires reliable reward models—precisely where reward hacking can reappear.
- Data transformation:
  - Converting answers to integers (Section 3.5; Appendix A) avoids parsing ambiguity but narrows the target format. This could bias training toward certain reasoning styles and might not capture all complexities of original tasks.
- Compute and latency:
  - Long generation budgets (up to 20,480 tokens; Section 4.1) and over-sampling for Dynamic Sampling increase wall-clock cost, even if step-wise convergence improves (Figure 6).
- No KL regularization:
  - Removing the KL penalty (Section 2.3) allows larger policy drift. While beneficial for long-CoT exploration, it may risk undesirable behaviors or instability in other domains unless counterbalanced by other constraints.
- Group-based normalization assumptions:
  - GRPO-style z-scoring assumes useful within-group variance. Dynamic Sampling enforces this by construction, but that also discards “too easy” or “too hard” prompts, which could introduce selection bias in training distribution over time.
- Generalization and robustness:
  - Results are strong on AIME 2024; extrapolation to coding, science QA, or multilingual settings remains to be demonstrated.
- Credit assignment scope:
  - Token-level credit is a step forward, but there is no explicit mechanism for attributing credit to “reasoning steps” or sections of the CoT beyond tokens; more structured credit assignment might help further.

## 7. Implications and Future Directions
- Field impact:
  - DAPO clarifies which levers matter in long-CoT RL: asymmetric clipping for exploration, gradient-preserving sampling, token-level reduction, and better handling of truncation. With full code/data release, it sets a practical baseline for reproducing “thinking” models.
- Follow-up research it enables:
  - Extending Dynamic Sampling to actively target “borderline” prompts (curriculum) or uncertainty-aware selection rather than simple all-correct/all-wrong filtering.
  - Combining token-level loss with process-level rewards (e.g., verifying intermediate steps) to improve credit assignment beyond z-scored outcomes.
  - Studying when and how to reintroduce adaptive KL terms or other constraints to prevent pathological drift in non-math domains.
  - Exploring variance reduction and off-policy corrections tailored to group-normalized advantages.
- Practical applications:
  - Training domain-specialized reasoners for STEM tutoring, competitive programming, or theorem proving where final answers are verifiable.
  - Serving as a drop-in RL recipe for organizations that already have a capable base model but struggle with stability and exploration in long-CoT fine-tuning.
  - Dataset transformation templates (Appendix A) for building robust, verifiable training sets in other symbolic or semi-symbolic domains.

Block-cited highlights
- > Figure 1: DAPO on `Qwen2.5‑32B` reaches 50 AIME 2024 points, outperforming DeepSeek‑R1‑Zero‑Qwen‑32B (47) with 50% of the training steps.
- > Table 1: Stepwise gains from Overlong Filtering (+6), Clip-Higher (+2), Soft Overlong Punishment (+3), Token-level Loss (+1), Dynamic Sampling (+8) culminating in 50.
- > Figure 2: Clip-Higher prevents entropy collapse and raises AIME accuracy.
- > Figure 5: Overlong Filtering and soft length penalties stabilize entropy and improve accuracy.
- > Equation (10): Asymmetric clipping `clip(r, 1−ε_low, 1+ε_high)`—the core of Clip-Higher.
- > Equation (13): Length-aware penalty shaping near the generation cap.
