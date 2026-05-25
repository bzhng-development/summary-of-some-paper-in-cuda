# Group Sequence Policy Optimization

**ArXiv:** [2507.18071](https://arxiv.org/abs/2507.18071)

## 🎯 Pitch

Group Sequence Policy Optimization (GSPO) introduces a novel reinforcement learning algorithm that shifts from unstable token-level updates to robust sequence-level updates for tuning large language models. By aligning importance sampling to the actual unit of reward—entire sequences—GSPO dramatically stabilizes training, especially for Mixture-of-Experts models, and delivers improved training efficiency and performance over prior methods. This breakthrough resolves longstanding scalability bottlenecks in RL for LLMs, paving the way for reliably training much larger, more capable language models and accelerating advancement in complex reasoning tasks.

---

## 1. Executive Summary
Group Sequence Policy Optimization (GSPO) is a reinforcement learning (RL) algorithm for fine‑tuning large language models that replaces token‑level updates with sequence‑level updates. By defining importance weights on entire responses and clipping at the sequence level, GSPO stabilizes training—especially for Mixture‑of‑Experts (MoE) models—while improving sample efficiency and benchmark performance over GRPO (Group Relative Policy Optimization) (see §4.1–4.2, §5.1, Fig. 1).

## 2. Context and Motivation
- Problem addressed
  - Scaling RL for large language models (LLMs) requires stable, robust training when responses are long and models are huge or sparse (MoE) (§1). Existing strong baselines, notably GRPO, frequently become unstable and can catastrophically collapse (§1, §3).
  - Collapse here means model quality degrades sharply and cannot be restored by resuming from checkpoints or retuning hyperparameters. As §3 emphasizes: 
    > “We have empirically observed that this can lead to model collapse that is often irreversible.”
- Why it matters
  - RL is a key path to growing reasoning capabilities (competition‑level math, programming) by encouraging long, deep chains of thought (§1). If training is unstable, it blocks scaling these benefits to larger models and tasks.
- Prior approaches and their limitations
  - PPO (Proximal Policy Optimization): Uses a separate value network to estimate per‑token advantages and clips token‑level importance ratios (Eq. 1). In practice, the value model doubles memory/compute and is hard to make reliable for long responses (§2).
  - GRPO: Avoids a value model by using group‑relative advantages (normalize rewards within a set of G responses for one prompt) but still optimizes per‑token using token‑level importance ratios (Eqs. 2–3). §3 argues this misapplies importance sampling—using a single next‑token sample per time step as if it corrects distribution mismatch—creating high‑variance gradients that accumulate over long sequences and are amplified by clipping.
- Positioning
  - GSPO reframes the optimization unit to match the reward unit: sequences. It defines importance weights on sequence likelihood and performs sequence‑level clipping (Eq. 5–7). This aligns with the core principle of importance sampling (Eq. 4) and stabilizes MoE RL without extra tricks (§4.1–4.2, §5.3).

## 3. Technical Approach
At a high level: for each prompt `x`, generate a group of `G` responses `y₁,…,y_G` using the old policy `π_θ_old`. Compute a scalar reward `r(x,y)` for each response using a verifier (in [0,1]). Normalize these rewards within the group to form advantages. Then, update the new policy `π_θ` using a sequence‑level importance ratio with clipping.

Step‑by‑step:
1. Grouped rollouts and rewards (§2, §4.1)
   - For each query `x`, sample `G` responses from the current data‑collection policy `π_θ_old`.
   - Score each response with a reward model/verifier `r(x,y) ∈ [0,1]`.
   - Compute group‑relative advantage (Eq. 6):
     - Subtract the mean reward of the G responses and divide by their standard deviation.
     - Intuition: this focuses optimization on which responses are better within the group, avoiding a separate value estimator (as in GRPO).

2. Sequence‑level importance ratio with length normalization (§4.1; Eqs. 5–7)
   - Define sequence likelihood as the joint probability of all tokens in a response: `π_θ(y|x) = ∏_t π_θ(y_t | x, y_<t)`.
   - Define the sequence‑level importance ratio for response `y_i`:
     - `s_i(θ) = (π_θ(y_i|x) / π_θ_old(y_i|x))^(1/|y_i|)`.
     - The exponent `1/|y_i|` is length normalization. Without it, a few token changes can cause large swings in the ratio, and different lengths would need different clipping ranges (§4.1).
   - Objective with sequence‑level clipping (Eq. 5):
     - For each `i`, use `min(s_i(θ)*A_i, clip(s_i(θ), 1-ε, 1+ε)*A_i)` and average over the group.
     - This clips entire responses, not tokens.

3. Why sequence‑level? The importance sampling principle (Eq. 4; §3)
   - Importance sampling estimates expectations under a target distribution by reweighting samples from a behavior distribution.
   - In language generation, the natural unit is the whole sequence, because the reward is given per sequence.
   - Token‑level weighting in GRPO uses one sample per next‑token distribution—too few for the ratio to reliably correct the mismatch—injecting variance that accumulates across tokens (§3).

4. Gradient behavior and stability (§4.2)
   - GSPO gradient (Eq. 10): every token’s log‑prob gradient in a response gets the same weight `s_i(θ)*A_i/|y_i|`. This removes intra‑sequence token‑level weighting noise.
   - GRPO gradient (Eq. 12): token `t` is weighted by its own token‑level importance ratio `w_{i,t}` which varies across tokens, leading to unequal weights that can accumulate unpredictably.
   - Consequence: GSPO’s equal weighting per response reduces variance and avoids unstable training dynamics (§4.2).

5. Optional token‑granular advantages: GSPO‑token (§4.3; Eqs. 13–17)
   - When finer credit assignment is needed (e.g., multi‑turn RL), GSPO‑token allows per‑token advantages `A_{i,t}` but keeps the sequence‑level importance ratio by “stopping the gradient” through token‑level probabilities:
     - `s_{i,t}(θ) = sg[s_i(θ)] * π_θ(y_{i,t}|…)/sg[π_θ(y_{i,t}|…)]`.
     - Numerically, `s_{i,t}(θ)` equals `s_i(θ)` for all tokens, so clipping/weights remain sequence‑level; gradients distribute across tokens according to `A_{i,t}` (Eq. 17).
   - If all `A_{i,t}` are equal to `A_i`, GSPO‑token is identical to GSPO in value, clipping, and gradient (§4.3).

6. Practical training setup (§5.1)
   - Large rollout batches are split into mini‑batches for efficiency, creating an off‑policy gap between `π_θ_old` (generator) and `π_θ` (optimizer), hence the need for clipping (§3).
   - Example hyperparameters (for the head‑to‑head with GRPO):
     - GSPO clipping range: left 3e‑4, right 4e‑4 (Eq. 5).
     - GRPO clipping range: left 0.2, right 0.27 (Eq. 2).
     - Each rollout batch is split into four mini‑batches (§5.1).
   - Note the magnitude difference in clipping ranges: a by‑product of how ratios are defined (sequence‑ vs token‑level), not a simple retuning (§4.1, §5.1).

7. Why GSPO helps MoE models (§5.3)
   - MoE instability under GRPO: after each gradient step, which experts the model routes tokens to can shift. With Qwen3‑30B‑A3B‑Base (48 layers), about 10% of experts change for the same sample across updates (§5.3). This makes token‑level ratios `w_{i,t}` fluctuate dramatically.
   - Prior workaround: Routing Replay—cache expert choices from `π_θ_old` and force `π_θ` to reuse them when computing ratios—adds memory/communication overhead and limits capacity (§5.3).
   - GSPO focuses on sequence likelihood, which is much less sensitive to per‑token routing flips; it converges without Routing Replay (Fig. 1; §5.3).

8. Infrastructure simplification (§5.4)
   - Because GSPO uses sequence‑level likelihoods, it is more tolerant to numerical precision differences between training and inference engines. §5.4 notes it may be possible to use likelihoods returned by the inference engine directly, avoiding recomputation.

## 4. Key Insights and Innovations
- Importance ratios must match the reward unit (§3, §4.1)
  - Novelty: Define and clip importance ratios at the sequence level to align with sequence‑level rewards (Eqs. 5–7). This embodies the core importance sampling principle (Eq. 4) within LLM RL.
  - Significance: Removes a major source of variance and instability in GRPO’s token‑level weighting, especially for long sequences and MoE routing volatility (§4.2, §5.3).
- Length‑normalized sequence ratios (§4.1)
  - Novelty: Raise the ratio to the power `1/|y|` to normalize for response length.
  - Significance: Prevents a few tokens from causing outsized ratio fluctuations and allows a single clipping range to work across lengths, lowering variance and operational complexity (§4.1).
- Sequence‑level clipping of entire responses (§4.1)
  - Novelty: Clip complete response updates instead of per‑token updates.
  - Significance: Excludes overly off‑policy sequences cleanly and consistently with how rewards are assigned, improving sample exploitation and stability (Fig. 2; §5.2).
- GSPO‑token for fine‑grained credit assignment with sequence‑level stability (§4.3)
  - Novelty: A stop‑gradient construction that retains sequence‑level ratios while allowing token‑wise advantages; provably reduces to GSPO when per‑token advantages are uniform (Eqs. 13–17).
  - Significance: Extends GSPO to settings like multi‑turn RL without sacrificing the core stability benefit.
- Stabilizing MoE RL without Routing Replay (§5.3)
  - Fundamental change in practice: 
    > “GSPO eliminates the dependency on Routing Replay and is fully capable of computing the importance ratios s_i(θ) conventionally, converging normally, and optimizing stably.” (§5.3; Fig. 1)
  - Significance: Simplifies infrastructure and removes capacity‑limiting workarounds in large MoE training.

## 5. Experimental Analysis
- Setup (§5.1)
  - Model: Cold‑start fine‑tune from `Qwen3‑30B‑A3B‑Base`.
  - Training: Each rollout batch split into 4 mini‑batches; GRPO requires Routing Replay to converge on MoE; GSPO does not.
  - Metrics and benchmarks:
    - Training reward (verifier score).
    - AIME’24: average Pass@1 over 32 samples.
    - LiveCodeBench (202410–202502): average Pass@1 over 8 samples.
    - CodeForces: Elo Rating.
  - Clipping ranges: GSPO 3e‑4/4e‑4 (left/right), GRPO 0.2/0.27 (§5.1).
- Main results (Fig. 1)
  - Training stability: GSPO shows smooth, steady improvement throughout training.
  - Efficiency: At similar compute and query budgets, GSPO achieves higher training reward and higher scores on AIME’24, LiveCodeBench, and CodeForces than GRPO (which is run with Routing Replay).
  - Scaling: GSPO continues to improve with more compute, periodic query refresh, and longer generations.
- Clipping behavior (Fig. 2; §5.2)
  - Empirical observation:
    > “We observe a difference of two orders of magnitude in the fractions of clipped tokens between GSPO and GRPO…”
  - Reported averages: GRPO ≈ 0.0013 clipped fraction vs GSPO ≈ 0.15.
  - Interpretation: Despite clipping far more tokens (because whole responses are clipped), GSPO trains more efficiently. This suggests GRPO’s token‑level gradients are noisy/inefficient, while GSPO’s sequence‑level signal is cleaner (§5.2).
- MoE stability (Fig. 3; §5.3)
  - Without Routing Replay, GRPO fails to converge on MoE; with Routing Replay it converges but adds overhead and constrains capacity.
  - GSPO converges without Routing Replay and avoids the expert‑activation volatility issue because it does not rely on per‑token likelihood stability (§5.3).
- Broader deployment signal (§5.1, §6)
  - The method has been applied to train recent Qwen3 models, indicating practical readiness; however, detailed external benchmarks for those models are not enumerated here.
- Are the experiments convincing?
  - Strengths:
    - Head‑to‑head training curves across multiple benchmarks (Fig. 1) and infrastructure studies (Fig. 2–3) directly target the claimed failure modes: instability, sample efficiency, and MoE routing volatility.
    - Concrete hyperparameters for clipping show that GSPO’s ratio scale differs materially from GRPO (§5.1), which aligns with the length‑normalized sequence definition (§4.1).
  - Gaps:
    - Numerical summaries beyond plots are limited; exact gains are not tabulated.
    - Sensitivity to group size `G`, advantage normalization choice, and ε ranges is not ablated.
    - Comparisons are primarily to GRPO; PPO or other RLHF/RLAIF baselines are not included in this paper’s experiments.
    - Details on the verifier(s) used for rewards are abstracted (only the [0,1] range is specified).

## 6. Limitations and Trade-offs
- Sequence‑level focus may blunt token‑level credit assignment
  - GSPO addresses instability by equalizing token weights within a response (Eq. 10). This is ideal for sequence‑level rewards, but tasks that truly need precise temporal credit assignment could benefit from GSPO‑token (§4.3). The paper does not present empirical results for GSPO‑token.
- Heavy clipping of entire sequences
  - Fig. 2 shows GSPO discards a large fraction of tokens via sequence clipping (~0.15). While training remains efficient, this could translate to wasted generation compute; the paper argues the net effect is positive but does not quantify compute‑efficiency trade‑offs.
- Limited ablations
  - No reported sensitivity studies on:
    - Group size `G` and the statistics used for advantage normalization.
    - Length normalization choice and its exponent.
    - KL regularization strength (omitted from equations “for brevity” §2, but often important in practice).
- Scope of evaluation
  - Experiments focus on a single MoE base model and three reasoning/coding benchmarks. Broader tasks (dialogue, safety RL, preference RL) are not covered in this paper.
- Theoretical coverage
  - While the gradient comparison is clear (§4.2), formal variance or convergence analyses are not provided; claims are primarily empirical.

## 7. Implications and Future Directions
- Impact on the field
  - GSPO reframes the de facto RL objective for LLMs from token‑ to sequence‑level—all the way through importance weighting, clipping, and gradient aggregation. This is a fundamental shift that aligns the optimization unit with the reward unit (§3, §4.1).
  - For MoE models, removing Routing Replay simplifies training pipelines and unlocks true capacity usage (§5.3).
- Follow‑up research enabled
  - Algorithmic analysis:
    - Variance/bias characterization of length‑normalized sequence ratios and sequence‑level clipping.
    - Adaptive clipping policies driven by measured off‑policy distance.
  - Practical RL design:
    - Systematic studies of group size `G`, reward normalization strategies, and verifier design.
    - Extensions to multi‑turn RL and tool use with GSPO‑token, including per‑turn advantages and partial rollouts.
  - Infrastructure:
    - Training–inference disaggregation: directly using inference‑engine likelihoods (§5.4) to reduce recomputation and memory.
    - Efficient handling of high clipping fractions (e.g., early‑reject during generation when ratios exceed bounds).
- Applications
  - Stable large‑scale RL for math/coding assistants and long‑reasoning agents.
  - Training large MoE models without routing constraints or custom replay mechanisms.
  - Industrial RL pipelines where inference engines differ numerically from training engines and recomputation is costly (§5.4).

> Overall, GSPO’s central idea—sequence‑level importance weighting and clipping that matches how rewards are assigned—addresses a core instability in state‑of‑the‑art LLM RL. The experiments (Figs. 1–3; §5.1–5.3) show stability and efficiency gains, particularly for MoE, and point to simpler, more scalable RL infrastructure.
