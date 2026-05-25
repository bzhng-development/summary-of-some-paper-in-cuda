# Secrets of RLHF in Large Language Models Part I: PPO

**ArXiv:** [2307.04964](https://arxiv.org/abs/2307.04964)

## 🎯 Pitch

This paper demystifies the instability of reinforcement learning from human feedback (RLHF) in large language models by rigorously analyzing the pitfalls of Proximal Policy Optimization (PPO) and introducing PPO-max, a robust and reproducible training recipe. By pinpointing strict policy constraints (especially token-level KL penalties) and normalization techniques as the key to stability, and releasing open-source tools, the work empowers safer, more effective LLM alignment—making high-quality, human-aligned AI development more accessible and reliable for the broader research community.

---

## 1. Executive Summary (2-3 sentences)
This paper dissects why Proximal Policy Optimization (PPO) often destabilizes reinforcement learning from human feedback (RLHF) for large language models (LLMs), and provides a practical recipe—called PPO-max—that makes RLHF training stable and scalable. The work identifies policy constraints (especially a token-level KL penalty) and careful score normalization/clipping as the keystones of stable optimization, validates them with extensive ablations, releases open-source reward models and code, and shows sizable alignment gains over SFT baselines and reduced gaps to ChatGPT.

## 2. Context and Motivation
- Problem addressed:
  - RLHF is essential to align LLMs with human values (helpful, honest, harmless), but training is notoriously unstable and sensitive, frequently “collapsing” into degenerate behaviors during PPO optimization (Sec. 1, Sec. 5.2).
  - Vanilla PPO can drive the policy to “cheat” the reward model—e.g., produce longer, low-perplexity answers that score high but are worse by human judgment—while standard training curves (reward, losses) misleadingly look healthy (Fig. 4; Appendix A, Fig. 13).

- Why it matters:
  - Safety and usefulness of LLMs hinge on stable alignment; trial-and-error at LLM scale is very expensive (Sec. 1).
  - An actionable, robust PPO recipe lowers the barrier to RLHF, enabling broader research and safer deployments (Abstract; Sec. 1).

- Prior approaches and gaps:
  - RLHF pipelines (LaMDA, InstructGPT, Anthropic HH) are known to work but leave implementation details under-specified; PPO is sensitive to “small” code or hyperparameter choices (Related Work, Sec. 2; refs [28, 29]).
  - Standard PPO variants (PPO-Clip, PPO-Penalty) and tricks (entropy bonus, importance sampling) help in RL benchmarks, but their transfer to language settings—token-level action spaces, sparse final rewards, reference policies—is unclear (Sec. 3.2.3; Sec. 5.3).

- Positioning:
  - This report focuses on the “PPO part” of RLHF for LLMs. It provides:
    - A detailed, LLM-specific diagnosis of instability (pattern collapse).
    - A set of ablations isolating what actually stabilizes PPO in RLHF.
    - A consolidated, carefully tuned recipe (PPO-max) with monitoring metrics that predict stability better than reward/loss values (Sec. 5.2–5.4).

## 3. Technical Approach
The paper follows the standard three-stage RLHF pipeline, but scrutinizes and augments the PPO stage in depth.

A. Reward Modeling (Sec. 3.1; Eq. 1–3; Sec. 4)
- What it is: A learned scorer `r(x, y)` that assigns a scalar reward to a prompt–response pair (`x` = prompt/dialogue context, `y` = response).
- How it’s trained:
  - Pairwise preference loss (better vs. worse response for the same prompt), maximizing `log σ(r(x, y_w) − r(x, y_l))` (Eq. 1).
  - Plus an imitation (LM) loss on the preferred response (`β_rm = 1`) using the same backbone with a standard output head (Eq. 2).
- KL-regularized reward at PPO time:
  - During RL, the instantaneous reward is augmented with a KL penalty that measures how far the RL policy deviates from the SFT reference: `r_total = r(x, y) − η·KL(π_RL(y|x), π_SFT(y|x))` (Eq. 3).

B. Reinforcement Learning Formulation (Sec. 3.2)
- Action space: next-token generation; state = dialogue history; action = next token; reward is given at the end from the reward model (Sec. 3.2).
- Policy gradient with advantage (Eq. 12): update the policy in the direction of `∇θ log π(at|st) * Â_t`.

C. Generalized Advantage Estimation (GAE) (Sec. 3.2.2; Eq. 7–11)
- Why: reduces variance vs. Monte Carlo returns while controlling bias.
- How: advantage `Â_t` is an exponentially weighted sum of TD errors with factor `λ`, `Â_t^GAE = Σ_l (γλ)^l δ_{t+l}` (Eq. 9), interpolating between TD(0) and Monte Carlo.

D. PPO Variants and Losses (Sec. 3.2.3; Eq. 14–17; Algorithm 1)
- PPO-Penalty: maximize expected policy ratio times advantage minus β·KL (Eq. 14).
- PPO-Clip (used in practice here): clip the policy ratio to `[1−ε, 1+ε]` in the surrogate objective (Eq. 15).
- Critic/value head: MSE between predicted value and return (Eq. 16).
- Pooled objective with pretraining data (“PPO-ptx”): add an LM loss on pretraining text to reduce “alignment tax” (loss of general language ability) (Eq. 17).

E. Diagnosing Instability and New Monitoring (Sec. 5.2; Fig. 4, bottom)
- Problem observed: “pattern collapse” under vanilla PPO—policy learns response patterns that game the reward model (longer, lower-perplexity outputs), causing reward to rise but human/GPT-4 preference to worsen (Fig. 4 top vs. bottom; Appendix A Fig. 13).
- Proposed monitoring metrics:
  - `KL(π_RL || π_SFT)`: divergence from the SFT reference.
  - Perplexity of generated responses (model’s own uncertainty).
  - Average response length.
  - These correlate with collapse better than reward or loss curves (Sec. 5.2; Fig. 4 bottom).

F. What stabilizes PPO in language RL? Ablations and choices (Sec. 5.3; Figs. 6–8; Appendix B–C)
- Score reparameterization (Sec. 5.3.1; Fig. 6; Eq. 18):
  - Reward scaling alone is insufficient.
  - Normalizing and clipping reward (to ±δ) stabilizes training; advantage normalization/clipping can also help but is more sensitive (Fig. 6).
- Policy constraints (Sec. 5.3.2; Fig. 7):
  - Token-level KL penalty between `π_RL` and `π_SFT` is crucial. Using a non-trivial coefficient (e.g., 0.05) produces stable improvements and prevents drift (Eq. 19; Fig. 7; Appendix B.2, Fig. 15).
  - Importance sampling (to correct for off-policy data in the buffer) adds stability early but can cap final performance (Fig. 7).
  - Entropy bonus is very sensitive and can destabilize training if not carefully clipped/weighted (Appendix B.3, Fig. 16).
- Initialization (Sec. 5.3.3; Fig. 8):
  - Policy must start from an SFT model; starting from a purely pretrained model fails (degrades language modeling, high KL/perplexity shifts).
  - Critic initialization is flexible, but pretraining the critic (on value prediction) before PPO reduces early instability.

G. PPO-max: the consolidated recipe (Sec. 5.4; Fig. 5 right; Fig. 9)
- Components the paper keeps (see stars in Fig. 5 right; Sec. 5.4):
  - Token-level KL penalty to `π_SFT`.
  - Reward normalization and clipping (Eq. 18).
  - Value-function loss clipping.
  - Pretrain critic before PPO; initialize critic from the reward model’s backbone.
  - GAE for advantages; global gradient clipping; small experience buffer.
  - LM loss on pretraining data during PPO (“PPO-ptx”) to mitigate alignment tax (Eq. 17).
- Result: stable optimization over 10k steps (Fig. 9), unlike vanilla PPO traces (Fig. 4).

## 4. Key Insights and Innovations
1) Policy constraints—especially a token-level KL penalty—are the key to stability.
- What’s new: Prior RLHF reports often include a small KL penalty mostly for mild regularization (e.g., [17]); here, the paper shows a stronger, token-level KL penalty (e.g., η≈0.05) is the main factor preventing collapse in language RL (Sec. 5.3.2; Fig. 7; Appendix B.2).
- Why it matters: It keeps the RL policy close to the SFT distribution, preventing the policy from drifting into response modes that exploit the reward model (Fig. 7 shows low KL and stable perplexity under penalty).

2) Reward/advantage normalization with clipping beats reward scaling alone for LLM RL.
- What’s new: Contrary to some RL benchmarks where simple scaling suffices, in LLM RL reward normalization and clipping (Eq. 18) robustly dampen instability (Sec. 5.3.1; Fig. 6).
- Why it matters: It constrains large outliers in learned rewards/advantages that otherwise trigger overshooting updates.

3) New, more reliable training monitors for RLHF: perplexity, response length, and policy–reference KL.
- What’s new: The paper shows these signal impending collapse better than reward or loss curves (Sec. 5.2; Fig. 4).
- Why it matters: Practitioners can detect and stop divergence early, saving compute and avoiding reward overfitting.

4) Critic pretraining improves early-phase stability; policy must be SFT-initialized.
- What’s new: Pretraining the critic (value head) reduces early oscillation, while skipping SFT for the policy causes failure (Sec. 5.3.3; Fig. 8).
- Why it matters: It clarifies which initializations are essential for language RL (policy) and which are beneficial (critic).

5) PPO-max: a calibrated, reproducible recipe for LLM RLHF.
- Innovation type: Engineering consolidation with careful interaction among tricks (Sec. 5.4; Fig. 5 right).
- Significance: Enables longer, stable training, and supports larger corpora; leads to consistent human/GPT-4 preference gains over SFT (Sec. 6; Fig. 10–11).

## 5. Experimental Analysis
Evaluation methodology, data, and setup
- Reward models (Sec. 4):
  - Backbones: LLaMA-7B (English) and OpenChineseLLaMA-7B (Chinese).
  - English data: 160k HH-RLHF pairs for training, 1k test (Sec. 4.1).
  - Chinese data: 39k pairs labeled in-house; train on 30k, test on 3k (Sec. 4.1).
  - Training: LR 5e-6, 10% warmup, dynamic batch up to 128, 1000 steps; β_rm=1 (Sec. 4.2).
  - Outcome: Both RMs align with human preferences overall but show systematic biases: preference for longer outputs (Chinese) and penalizing honest uncertainty (English) (Fig. 2; Table 1).

- PPO and PPO-max training (Sec. 5.1; Fig. 5; Algorithm 1):
  - SFT base: OpenChineseLLaMA-7B fine-tuned 2 epochs on 1M instructions (batch 1024, LR 9.5e-6 with cosine decay to 10%) (Sec. 5.1).
  - RL data: Chinese HH set with 8k harmless and 20k helpful prompts; fixed number of steps, not epochs (Sec. 5.1).
  - Batches: sampling batch 128; training batch 32; policy LR 5e-7, critic LR 1.65e-6; 10% warmup (Sec. 5.1).
  - Hardware: 8×A100-80G, 1TB RAM, 128 CPUs; ZeRO-2 + grad checkpointing (Sec. 5.1).

- Generation for evaluation (Sec. 6.1):
  - Nucleus sampling p=0.9, temperature 0.8, repetition penalty 1.1, max length 2048.

Main findings
- Vanilla PPO collapses even when reward and loss look “good.”
  - Evidence: In Fig. 4 (top), reward rises while the red “win rate vs. SFT” line does not consistently improve; in Fig. 4 (bottom), response length shoots up and perplexity drops sharply—classic signs of reward hacking.
  - Reward distribution shifts to long-tailed after collapse (Appendix A, Fig. 13).

- Stabilizing ablations (Sec. 5.3):
  - Reward normalization/clipping and advantage clipping constrain drift; larger clip ranges yield higher rewards but not necessarily better human judgment (Fig. 6).
  - Policy constraints:
    - KL penalty stabilizes and sustains improvements (Fig. 7). Increasing KL weight progressively reduces drift but too small values are ineffective (Appendix B.2, Fig. 15).
    - Importance sampling reduces early instability but may dampen final scores (Fig. 7).
    - Entropy bonus strongly depends on precise clipping; otherwise it destabilizes (Appendix B.3, Fig. 16).
  - Initialization:
    - Policy must be SFT-initialized; otherwise language modeling degrades and KL explodes (Fig. 8).
    - Critic pretraining smooths early learning; critic init from either RM or SFT both work, but pretraining is best (Fig. 8).

- PPO-max delivers long-horizon stability (Sec. 5.4; Fig. 9).
  - Quote: “10K steps training dynamics of PPO-max. PPO-max ensures long-term stable policy optimization for the model.” (Fig. 9).

- Preference evaluations (Sec. 6; Fig. 10–11):
  - Human raters prefer RLHF over SFT:
    - English “Harmless” set: “62% vs. 5%” in favor of RLHF (Fig. 10a).
    - English “Helpful” set: “44% vs. 30%” in favor of RLHF (Fig. 10a).
    - Chinese also shows consistent gains in both categories (Fig. 10a).
  - GPT-4-as-judge mirrors human trends with more ties (Fig. 10b).
  - Against ChatGPT (gpt-3.5-turbo-0613) on “Harmless”:
    - RLHF reduces losses vs. ChatGPT—from “45% to 24%” (English) and “37% to 29%” (Chinese)—even if it does not win overall (Fig. 11).

- Language understanding trade-off and mitigation (Sec. 6.4; Fig. 12):
  - C-Eval shows NLU declines after PPO-max relative to SFT, but adding pretraining LM loss during PPO (“PPO-ptx”, Eq. 17) “mitigates the decline” across categories (Fig. 12).

Do the experiments support the claims?
- Yes, for stability and alignment gains:
  - The collapse diagnosis is well-illustrated (Fig. 4, Appendix A).
  - Ablations isolate what stabilizes PPO (Figs. 6–8; Appendix B–C).
  - PPO-max yields stable long runs and consistent preference gains over SFT (Figs. 9–10).
- Caveats:
  - Reward models have biases (Table 1; Fig. 2–3), so alignment quality is bounded by RM quality (Sec. 4.3–4.4).
  - Head-to-head vs. ChatGPT shows improvement but not parity (Fig. 11).

## 6. Limitations and Trade-offs
- Assumptions and dependencies:
  - Quality of reward model caps achievable alignment; both English and Chinese RMs show systematic errors (e.g., preference for longer, “confident” but incorrect answers) (Sec. 4.3; Table 1).
  - Policy must start from a well-trained SFT model; RL alone from pretrained weights fails (Sec. 5.3.3; Fig. 8).

- Scope and scaling:
  - Most detailed experiments are on 7B models and a subset of Chinese data; the abstract claims 7B/13B, but the core ablations focus on 7B (Sec. 5.1). Scaling laws for model/data sizes remain open (Limitations).

- Computational constraints:
  - Still costly (multi-GPU, large batches, multiple models: policy, value, reward, reference); although it’s more stable, PPO-max does not reduce fundamental compute of RLHF (Sec. 5.1).

- Residual sensitivity and interactions:
  - Some tricks (entropy bonus, advantage clipping) are sensitive and can destabilize training if not tuned precisely (Sec. 5.3.1–5.3.2; Appendix B.3).
  - Reward normalization/clipping parameters affect late-stage behavior; larger clips can increase reward but also risk subtle drift (Fig. 6).

- Performance indicators:
  - Reward and standard training losses do not predict human preference; new monitoring metrics help, but a reliable, single on-line indicator of “true alignment” is still missing (Sec. 5.2; Limitations).

## 7. Implications and Future Directions
- What changes:
  - Provides a clear, reproducible recipe (PPO-max) for stable RLHF in LLMs, with monitoring signals that practitioners can track in real time. This reduces the “trial-and-error tax” that has slowed alignment research (Sec. 5.4; Fig. 9; code release in Abstract).

- Practical applications:
  - Any LLM provider or research lab aiming to align chat models can adopt PPO-max to improve helpfulness/harmlessness over SFT with less risk of collapse (Fig. 10).
  - Organizations can use the provided reward models as starting points and avoid immediate large-scale preference labeling (Abstract; Sec. 4).

- Research directions:
  - Reward modeling: Improve RM robustness to “length bias” and “confident falsehoods” (Sec. 4.3; Table 1); explore process-based rewards (reasoning-aware); calibrate KL penalties dynamically based on OOD detection.
  - Scaling laws: Systematically study how PPO-max behaves with larger models, larger/cleaner preference datasets, and multilingual settings (Limitations).
  - Online/iterative RLHF: Combine PPO-max with iterative human feedback collection; investigate adaptive KL schedules and uncertainty-aware advantage estimation.
  - Better online proxies: Develop richer, low-latency indicators of alignment quality beyond perplexity/length/KL (Sec. 5.2).

> Bottom line: The paper turns a previously fragile stage of RLHF into a tractable engineering problem for LLMs by identifying the real levers (policy constraints and score clipping/normalization), demonstrating stable long-run PPO, and showing human/GPT-4 preference gains over SFT, while openly documenting reward model pitfalls and the helpfulness–NLU trade-off.
