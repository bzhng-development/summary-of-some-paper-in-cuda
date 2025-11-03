# DAPO: An Open-Source LLM Reinforcement Learning System at Scale

**ArXiv:** [2503.14476](https://arxiv.org/abs/2503.14476)

## 🎯 Pitch

DAPO breaks new ground by fully open-sourcing a state-of-the-art reinforcement learning framework for training large language models on complex reasoning tasks, including all algorithmic details, code, and data. By replacing fragile elements in prior RL pipelines with four robust innovations—Clip-Higher, Dynamic Sampling, Token-level loss, and Overlong reward shaping—DAPO enables reproducible, efficient, and industry-level performance, achieving 50/100 accuracy on AIME 2024 with just half the training steps of the previous open benchmark. This democratizes access to high-performing reasoning LLMs, accelerates research progress, and empowers the community to build on transparent and reproducible RL foundations.

---

## 1. Executive Summary
DAPO introduces a practical, large‑scale reinforcement learning (RL) recipe for training reasoning‑capable large language models (LLMs) and fully open‑sources the algorithm, code, and data. It replaces fragile ingredients in prior RL pipelines with four concrete techniques—Clip‑Higher, Dynamic Sampling, Token‑level loss, and Overlong reward shaping—enabling a Qwen2.5‑32B base model to reach 50/100 on AIME 2024 while using roughly half the training steps of the prior open benchmark (Figure 1).

## 2. Context and Motivation
- Problem addressed
  - Long chain‑of‑thought (CoT) reasoning with LLMs benefits from RL “test‑time scaling,” but reproducing strong results has proven difficult because key training details are opaque. The paper documents common failure modes: entropy collapse (the model becomes over‑confident and stops exploring), reward noise from truncated generations, and unstable gradients when prompts become too easy (Section 1; Figures 2, 5).
- Why this matters
  - Practical impact: Competitive math/coding tasks (AIME, Codeforces) require deep, multi‑step reasoning. A reproducible RL recipe lets the community build such capabilities without proprietary tooling.
  - Research impact: Clear ablations and open code/data reduce guesswork and enable systematic progress on long‑CoT RL.
- Prior approaches and gaps
  - PPO and GRPO (Sections 2.1–2.2) are the main baselines. GRPO normalizes rewards within a group of samples to compute advantages without a learned value function (Equation 4), but:
    - It uses a symmetric clip range (±ε) that limits probability increases for low‑probability “exploration” tokens (Section 3.1).
    - It averages loss per sample (not per token), which over‑weights short responses and under‑penalizes junk in long responses (Section 3.3).
    - It suffers gradient collapse when all G samples for a prompt are either all correct or all wrong (dynamic accuracy = 1 or 0), making the normalized advantage zero (Section 3.2).
  - Typical RLHF includes a KL penalty to keep the policy near a reference model, but for long‑CoT reasoning the target distribution can deviate substantially; a fixed KL constraint can be counter‑productive (Section 2.3).
- Positioning
  - DAPO is a drop‑in, open recipe that directly tackles the above issues with four simple, measurable interventions, plus a dataset transformation that enables low‑noise, rule‑based rewards (Sections 3 and Appendix A).

## 3. Technical Approach
DAPO (Decoupled Clip and Dynamic sAmpling Policy Optimization) keeps GRPO’s idea of group‑relative advantages but changes how gradients are computed, how samples are selected, how clipping is applied, and how rewards are shaped for overlong outputs.

High‑level loop (Algorithm 1):
1. Sample a batch of prompts `Db`.
2. Freeze the current policy as `π_old` and generate `G` responses per prompt.
3. Compute rewards for each response using a deterministic, rule‑based correctness function (Equation 7) and apply length shaping (Section 3.4).
4. Dynamic sampling: keep only prompts whose group outcomes are mixed (neither all correct nor all wrong), ensuring non‑zero, informative gradients; continue sampling until a buffer of size `N` is filled (Equation 11; Algorithm 1 lines 6–8).
5. Compute group‑normalized advantages `Â_i,t = (R_i − mean(R))/std(R)` per token (Equation 9).
6. Update the policy by maximizing a token‑level clipped objective with asymmetric clip bounds (Equation 8/10), for `µ` optimizer steps over the buffer (Algorithm 1 lines 9–11).

Key components in detail:
- Rule‑based reward with verifiable answers (Section 2.4; Equation 7)
  - For math tasks, the reward is +1 if the predicted final answer is equivalent to the ground truth, −1 otherwise. To reduce parsing errors, answers are transformed to integers (Appendix A), yielding the 17K‑problem `DAPO‑Math‑17K` dataset.
- Group‑relative advantage (GRPO‑style, Section 2.2; Equation 4)
  - For each prompt, `G` responses are sampled. Their scalar rewards are standardized within the group; the standardized score serves as the advantage for every token in that response, avoiding a learned value function and leakage from reward model bias.
- Clip‑Higher: asymmetric clipping (Section 3.1; Equation 10)
  - PPO‑style clipping uses the importance ratio `r_i,t(θ) = π_θ(o_i,t | q, o_i,<t) / π_old(o_i,t | q, o_i,<t)` and clamps it to `[1−ε_low, 1+ε_high]`. Increasing `ε_high` while keeping `ε_low` standard (0.2) raises the ceiling for probability increases on tokens with positive advantage, improving exploration and arresting entropy collapse (Figures 2a–b).
  - Intuition (Section 3.1): symmetric clipping disproportionately restricts low‑prob tokens; even a small absolute increase gets clipped. Asymmetric clipping lets such tokens rise more freely when they help reward.
  - Evidence: the mean probability of “up‑clipped” tokens is low (<0.2; Figure 3a), confirming that the upper clip bound had been throttling exploration.
- Dynamic Sampling (Section 3.2; Equation 11)
  - Problem: When a prompt becomes too easy, all `G` samples share the same reward → zero variance → standardized advantage 0 → zero gradient. As training progresses, the share of such prompts grows (Figure 3b).
  - Solution: Oversample and filter to keep only prompts with mixed outcomes `0 < #correct < G`. This maintains signal in every batch. Despite greater sampling, wall‑clock convergence can be faster because wasted updates on zero‑gradient data are eliminated (Figure 6).
- Token‑level policy gradient loss (Section 3.3; Equation 12)
  - GRPO’s per‑sample averaging assigns equal weight to each sample regardless of length, diluting the impact of long, informative sequences and insufficiently penalizing long, low‑quality outputs (repetition/gibberish).
  - DAPO sums losses across all tokens (normalized by total tokens in the batch), so long sequences contribute proportionally, improving stability: entropy grows steadily rather than exploding, and response length trends become healthier (Figures 4a–b).
- Overlong reward shaping (Section 3.4; Figure 5; Equation 13)
  - Issue: Truncated responses (due to max length) get blanket negative reward, injecting noise—good reasoning can be cut off and mislabeled as “bad.”
  - Two‑step fix:
    - Overlong filtering: mask the loss of truncated samples so they do not backpropagate (stabilizes training and improves AIME accuracy; Figure 5a; reduces entropy spikes; Figure 5b).
    - Soft overlong punishment: add a length‑aware penalty `R_length(y)` that smoothly increases as length approaches/exceeds the limit (`L_max − L_cache` to `L_max`, then −1 beyond; Equation 13). This warns the model away from excessive verbosity without mislabeling partially good reasoning.
- Removing the KL penalty (Section 2.3; contrast with GRPO’s KL term in Equation 5)
  - Standard RLHF regularizes divergence from a reference model via a KL term. For long‑CoT reasoning, useful behaviors may deviate substantially from the base distribution, so a fixed KL penalty can inhibit learning. DAPO drops this term.

Implementation snapshot (Section 4.1)
- Framework: `verl` (HybridFlow).
- Model: `Qwen2.5‑32B` base.
- Optimization: AdamW, LR=1e‑6 with 20 rollout‑step warmup; per‑rollout: 512 prompts, `G=16` samples/prompt; 16 gradient updates per rollout step (mini‑batch 512).
- Lengths: expected max 16,384 tokens; soft cache 4,096; hard cap 20,480.
- Clip‑Higher: `ε_low=0.2`, `ε_high=0.28`.
- Evaluation: AIME 2024, repeated 32 times (`avg@32`), temperature 1.0, top‑p 0.7.

## 4. Key Insights and Innovations
- Clip‑Higher (asymmetric clipping) is a simple but pivotal fix for exploration in long‑CoT RL.
  - Difference: replaces a symmetric clip with separated `ε_low` and `ε_high` (Equation 10).
  - Why it matters: prevents entropy collapse, maintains diversity (Figures 2a–b), and contributes measurable gains (+2 points in the ablation when added after overlong filtering; Table 1).
- Dynamic Sampling keeps gradients informative throughout training.
  - Difference: filters out groups with all‑correct or all‑wrong outcomes and oversamples until batches contain only prompts with mixed rewards (Equation 11; Algorithm 1 lines 6–8).
  - Why it matters: avoids vanishing gradients as tasks become easy and improves training efficiency; overall accuracy jumps from 42 to 50 when added last (Table 1) and training progresses faster (Figure 6).
- Token‑level policy gradient loss rebalances learning toward reasoning‑rich sequences.
  - Difference: aggregate loss by tokens rather than averaging per sample (Equation 12).
  - Why it matters: suppresses degenerate long outputs and supports stable entropy and length growth (Figures 4a–b); adds stability and modest accuracy gains (+1 point in Table 1).
- Overlong reward shaping removes a major source of reward noise.
  - Difference: masks gradients from truncated outputs and adds a smooth length penalty near the cap (Equation 13).
  - Why it matters: stabilizes training and improves accuracy (+6 points cumulatively across filtering and soft punishment; Table 1; Figure 5).

These are fundamental design changes to the RL signal and optimization dynamics rather than mere hyperparameter tweaks.

## 5. Experimental Analysis
- Evaluation setup (Section 4)
  - Task: mathematics; primary benchmark AIME 2024.
  - Data: `DAPO‑Math‑17K` (Appendix A), created by transforming diverse math answers into integers to enable exact or rule‑based equivalence checks (Equation 7 and Appendix A example).
  - Metrics: `avg@32` accuracy—repeat test prompts 32 times to reduce sampling variance.
  - Baseline: naive GRPO trained from the same `Qwen2.5‑32B` base with group normalization (Section 4.1). For external comparison, DeepSeek‑R1‑Zero‑Qwen‑32B (Figure 1).
- Main results
  - Overall performance:
    - Figure 1: DAPO reaches 50% on AIME 2024. The plot shows higher accuracy than DeepSeek‑R1‑Zero‑Qwen‑32B (47%) while using about 50% of the gradient update steps.
    - Quote: “achieves 50 points on AIME 2024 … using 50% training steps” (Abstract; Figure 1 caption).
  - Ablations (Table 1):
    - Naive GRPO: 30.
    - + Overlong Filtering: 36.
    - + Clip‑Higher: 38.
    - + Soft Overlong Punishment: 41.
    - + Token‑level Loss: 42.
    - + Dynamic Sampling (full DAPO): 50.
    - Takeaway: Each component contributes; Dynamic Sampling yields the largest final jump (+8).
- Training dynamics and diagnostics (Section 4.3; Figure 7)
  - Mean response length rises as the model explores longer rationales (Figure 7a), but can plateau or dip; length alone is not monotonic.
  - Reward on the training set increases steadily (Figure 7b), yet this can decouple from validation accuracy, signaling overfitting risk.
  - Entropy trends are informative: too low → under‑exploration; too high → gibberish. With Clip‑Higher, entropy follows a slow, healthy upward trend (Figure 7c), while mean probability exhibits a complementary pattern (Figure 7d).
- Mechanism‑level evidence
  - Clip‑Higher boosts entropy and AIME accuracy (Figures 2a–b).
  - Dynamic Sampling reduces the proportion of zero‑gradient batches (Figure 3b) and accelerates progress (Figure 6).
  - Token‑level loss tames entropy and length growth (Figures 4a–b).
  - Overlong filtering and soft punishment stabilize training and avert entropy spikes (Figure 5).
- Qualitative evidence
  - Emergence of reflective/backtracking behaviors appears later in training (Section 4.4; Table 2), suggesting RL can elicit higher‑order reasoning routines, not only longer text.

Overall, the experiments are convincing because they (i) isolate each design choice via ablations (Table 1), (ii) show mechanism‑aligned metrics (Figures 2–6), and (iii) couple quantitative gains with qualitative behavior changes (Table 2).

## 6. Limitations and Trade-offs
- Reward availability and task scope
  - The method relies on rule‑based, verifiable rewards (Equation 7). It is demonstrated on math and uses integer‑answer reformulation to avoid parsing noise (Appendix A). Tasks without objective, easily verifiable end states (e.g., open‑ended dialogue) would require a different reward design or a robust reward model.
- Potential distribution drift (KL removal)
  - Removing the KL penalty (Section 2.3) enables exploration but also allows the policy to drift far from the base model. This can degrade general alignment or stylistic control in settings where staying close to the base is important.
- Data and sampling bias
  - Dynamic Sampling filters out all‑correct and all‑wrong prompts (Equation 11). While beneficial for gradient signal, it effectively reweighs the training distribution toward “borderline” prompts, which could bias learning if not managed (the paper does not report negative effects but the risk is conceptual).
- Compute and memory footprint
  - Long sequences (up to 20,480 tokens), group sampling (`G=16`), and large batch sizes (512 prompts) imply substantial GPU memory and runtime requirements (Section 4.1). Although Dynamic Sampling improves efficiency (Figure 6), the overall system remains compute‑intensive.
- Generalization beyond math
  - The study focuses on AIME‑style math. While techniques are general, their efficacy in other domains (coding, science QA) with different failure modes and reward structures remains to be validated.
- Overfitting signals
  - Section 4.3 observes that training reward increases can decouple from validation gains, indicating overfitting potential; no explicit early‑stopping or regularization beyond the DAPO mechanisms is discussed.

## 7. Implications and Future Directions
- Field impact
  - By surfacing concrete, reproducible fixes to well‑known RL instabilities and open‑sourcing the full stack (algorithm, code, data), DAPO lowers the barrier to training reasoning LLMs at scale. It shifts attention from opaque “secret sauces” to transparent design choices with measurable effects.
- Immediate applications
  - Any domain with verifiable outcomes can benefit: competitive programming (unit tests as rewards), theorem proving (proof checkers), math competitions, data cleaning and constraint satisfaction tasks, and tool‑use scenarios where executors yield binary success/failure.
- Follow‑up research
  - Reward design: extend the rule‑based approach with partial credit and process‑based rewards to reduce sparsity while avoiding reward hacking.
  - Safety/alignment: study principled alternatives to a fixed KL term—e.g., adaptive trust regions or constraint‑based methods—to balance exploration with alignment.
  - Credit assignment: combine token‑level loss with more sophisticated per‑token advantage estimation (e.g., token‑supervised value models) to better localize which reasoning steps drive success.
  - Data curriculum: formalize dynamic sampling into a curriculum that schedules difficulty and diversity, not merely filters for mixed outcomes.
  - Cross‑domain validation: port DAPO to coding and scientific reasoning benchmarks; test robustness under different inference temperatures and sampling strategies.
  - Systems optimization: pipeline rollouts and updates, asynchronous sampling, and mixed‑precision/truncation‑aware kernels to further reduce the cost of long‑CoT RL.

Quoted results and references to figures/tables/equations:
- “achieves 50 points on AIME 2024 … using 50% training steps” (Abstract; Figure 1).
- PPO/GRPO objectives and group‑relative advantage: Equations (1), (4), (5).
- DAPO objective with asymmetric clipping and token‑level reduction: Equations (8), (10), (12); Algorithm 1.
- Rule‑based reward and length shaping: Equations (7), (13); Section 3.4; Figure 5.
- Mechanism‑level evidence: Figures 2–6; Table 1 ablations.
- Dataset transformation: Appendix A and Section 3.5.
