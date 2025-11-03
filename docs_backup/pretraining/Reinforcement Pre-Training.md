# Reinforcement Pre-Training

**ArXiv:** [2506.08007](https://arxiv.org/abs/2506.08007)
**Authors:** Qingxiu Dong, Li Dong, Yao Tang, Tianzhu Ye, Yutao Sun, Zhifang Sui, Furu Wei
**Institutions:** (not explicitly stated on arXiv metadata)

## 🎯 Pitch

Reinforcement Pre-Training (RPT) revolutionizes language model training by transforming next-token prediction into a scalable reinforcement learning task with verifiable rewards, eliminating the need for human labels. This innovation significantly enhances reasoning capabilities and improves zero-shot accuracy, paving the way for robust, label-free pretraining that bridges self-supervised and reinforcement learning for large-scale applications.

---

## 1. Executive Summary (2-3 sentences)
This paper introduces Reinforcement Pre-Training (`RPT`), a way to treat standard next-token prediction as a reinforcement learning (RL) problem with verifiable, automatic rewards. By asking a model to “think” (generate a short reasoning trace) and then guess the next token, and rewarding it only when the guess exactly matches the ground truth at a valid token boundary, the method scales RL to the entire pretraining corpus without human labels (Sections 1, 3.1–3.2). Experiments show improved next-token accuracy, favorable scaling with compute, better starting points for later RL fine-tuning, and stronger zero-shot results on general benchmarks (Tables 1–3, Figure 5).

## 2. Context and Motivation
- Problem addressed
  - Modern large language models (LLMs) are mainly trained with next-token prediction (NTP), which is scalable and label-free but does not explicitly teach multi-step reasoning before predicting (Section 1).
  - RL has helped in post-training (e.g., preference alignment and reasoning), but it typically needs human feedback or curated QA data, which limits scale and risks “reward hacking” when rewards come from learned models rather than verifiable signals (Section 1; RLHF [OWJ+22]; RL with verifiable rewards, RLVR [LMP+25]).

- Why it matters
  - A scalable, general-purpose RL objective that uses the same massive corpora as NTP could merge the strengths of self-supervised pretraining and RL: richer reasoning behaviors learned at scale, less dependence on expensive human labels, and more robust, verifiable feedback (Section 1).

- Prior approaches and shortcomings
  - RLHF scales poorly due to costly human annotation and susceptibility to reward hacking via learned reward models (Section 1).
  - RLVR uses rule-based verifiers but relies on scarce QA pairs or domain-specific tasks, so it hasn’t served as a universal pretraining objective (Section 1).
  - “Teach yourself to think” methods that reward helpful rationales can be gamed by superficial tokens (e.g., repeating the target token in the rationale), risking degraded modeling (Related Work; [ZHS+24]).

- Positioning of this work
  - `RPT` reframes NTP itself as an RL task with an automatically verifiable, rule-based reward: the prediction is correct if and only if it exactly matches the ground-truth next token (with a precise byte-level check and token-boundary constraint). This removes dependence on external annotations and mitigates reward hacking (Sections 1, 3.2; Eq. (3)).

## 3. Technical Approach
At a high level, `RPT` keeps the usual pretraining data but changes how the model uses it: for every prefix in the data, the model must reason and then make a single next-token prediction; the environment replies with a verifiable 0/1 reward.

- Task reformulation: next-token reasoning (Section 3.1; Figure 2)
  - For each sequence `x0 … xT` from the corpus and each position `t`, the prefix `x<t` is the “state,” and the ground-truth next token is `xt`.
  - The model `πθ` first emits a “thinking” sequence `ct` (chain of thought) and then a short “prediction sequence” `yt` meant to equal the true next token. Output is `ot = (ct, yt)`.
  - This differs from standard NTP (Eq. (1)), where the model directly outputs a distribution over the next token without an explicit reasoning trace.

- Reward design: prefix matching with token-boundary checks (Section 3.2; Eq. (3); Figure 3)
  - Define the ground-truth continuation’s byte sequence as `x≥t` and the model’s predicted bytes as `yit` with length `l`.
  - Reward `rit = 1` if and only if:
    - `yit` exactly equals the first `l` bytes of `x≥t` (exact prefix), and
    - `l` is a valid boundary of the tokenization for `x≥t` (so rewards align to real tokens).
  - Otherwise `rit = 0`.
  - Why bytes and boundaries? Byte comparison handles punctuation, spaces, and out-of-vocabulary forms; boundary checks ensure the prediction corresponds to some valid tokenization, preventing trivial partial matches.

- RL objective and sampling (Section 3.2; Eq. (4))
  - For each `x<t`, sample `G` on-policy trajectories `o1..G` (each is a thinking trace and a prediction), compute rewards for each, and maximize the expected reward over the dataset:
    - `JRPT(θ) = E[(rit)]` with `oi ~ πθ(·|x<t)`.
  - The implementation uses GRPO [GYZ+25], a PPO-style on-policy method where multiple responses per prompt compete; high-reward trajectories increase probability, low-reward ones decrease it (Appendix B for hyperparameters).

- Data and difficulty-driven sampling (Section 3.3)
  - Pretraining source: OmniMATH, 4,428 competition-level math problems and solutions (diverse math text with LaTeX) (Section 3.3).
  - Not all tokens need reasoning. To focus on challenging spots, a small proxy model computes the entropy of the top-16 next tokens at each position; low-entropy (easy) positions are downsampled/filtered, and RL focuses on higher-entropy positions (Section 3.3).

- Training setup highlights (Sections 3.3–3.4; Appendix B; Figure 3)
  - Base model: `DeepSeek-R1-Distill-Qwen-14B` (a 14B reasoning-capable model).
  - Rollouts: `G = 8` responses per context with temperature 0.8; maximum training length 8k tokens; batch size 256; learning rate 1e-6; zero KL penalty; 1,000 main training steps.
  - Output parsing: the final next-token guess is the text inside the last `\boxed{…}` following a `</think>` delimiter.
  - Dynamic sampling from step 500 improves throughput (Section 3.3).

- Why these choices?
  - Verifiable, binary reward eliminates learned reward models (minimizing reward hacking) and needs no additional labels (Sections 1, 3.2).
  - Entropy filtering concentrates RL compute on tokens where “thinking” can change outcomes (Section 3.3).
  - Byte-prefix + boundary reward robustly handles tokens that might start with spaces or special symbols (Eq. (3), footnote 2).
  - GRPO’s group-based advantages fit the “many rollouts per prefix” setting (Appendix B).

- Comparison with standard objectives (Sections 2–3)
  - Standard NTP maximizes log-likelihood of the ground-truth token (Eq. (1)).
  - RLVR fine-tunes on (question, answer) pairs with an external verifier (Eq. (2)).
  - `RPT` turns each token prediction into an RL step using the corpus itself as a verifier, avoiding the scarcity of QA pairs and moving RL from “post-training” to “pretraining” (Sections 1, 3.1–3.2).

- Optional rewards explored (Appendix A)
  - Variants include first-token-only matching and dense rewards proportional to model probabilities for incorrect predictions. Results were comparable to the main prefix-matching reward, suggesting robustness to reasonable reward tweaks.

## 4. Key Insights and Innovations
- Scaling RL to web-text via intrinsic, verifiable rewards (Fundamental)
  - Novelty: Converts every next-token event into a verifiable RL step using only the corpus (Sections 1, 3.1–3.2). No human labels, no learned reward model.
  - Significance: Removes the bottlenecks that kept RL limited to small, curated datasets; aligns pretraining with the same RL machinery used later for preference or reasoning optimization.

- Prefix-matching reward on bytes with token-boundary validation (Technical)
  - Novelty: A simple, exact checker that supports tokens with leading spaces/symbols and multi-token predictions while preventing partial-byte hacks (Eq. (3)).
  - Significance: Maintains the integrity of the reward signal and reduces reward gaming.

- Difficulty-aware token selection using entropy (Practical)
  - Novelty: Filters positions using a proxy model’s top-16 next-token entropy, so RL emphasizes hard tokens likely to benefit from reasoning (Section 3.3).
  - Significance: Improves sample efficiency, as shown by strong gains on “medium” and “hard” token splits (Table 1).

- Compute scaling behavior for RL pretraining (Conceptual + Empirical)
  - Novelty: Establishes a power-law-style fit between RL compute and next-token accuracy (Eq. (5)), with high R² across easy/medium/hard splits (Figure 5).
  - Significance: Indicates `RPT` scales predictably with more compute, similar to NTP scaling laws but now for an RL-based pretraining objective.

- Better starting point for later RL fine-tuning (Applied)
  - Observation: Models pre-trained with `RPT` achieve higher performance before and after RLVR fine-tuning compared to standard or NTP-continued baselines (Table 2).

## 5. Experimental Analysis
- Evaluation methodology
  - Datasets
    - Pretraining and language modeling validation: OmniMATH (Section 3.3); 200 held-out samples for validation; tokens split by difficulty using entropy thresholds: easy (≥0.5), medium (≥1.0), hard (≥1.5) as computed by `R1-Distill-Qwen-14B` (Section 4.1).
    - RL fine-tuning: Skywork-OR1 subset with 256 training, 200 test questions with verifiable answers (Section 4.3).
    - Zero-shot general tasks: SuperGPQA and MMLU-Pro, both evaluated in multiple-choice format (Section 4.4).
  - Metrics
    - Next-token prediction accuracy (rather than perplexity) for language modeling (Table 1, Figure 4).
    - Multiple-choice accuracy for SuperGPQA and MMLU-Pro (Table 3).
  - Baselines
    - `Qwen2.5-14B` and `R1-Distill-Qwen-14B` in standard NTP mode; `R1-Distill-Qwen-14B` in a “reasoning mode” that emits a chain-of-thought before predicting; and the larger `R1-Distill-Qwen-32B` (Sections 4.1, 4.4).

- Main quantitative results
  - Next-token prediction accuracy (Table 1)
    - On OmniMATH validation:
      - Standard NTP: `R1-Distill-Qwen-14B` scores 41.60 (easy), 29.46 (medium), 20.43 (hard).
      - `RPT-14B`: 45.11, 33.56, 23.75.
      - “Reasoning mode” without `RPT` performs poorly (e.g., 3.31 on easy), underscoring that simply adding chain-of-thought doesn’t help unless trained for this objective.
    - Relation to larger model (Figure 4)
      - `RPT-14B` reaches the next-token accuracy of `R1-Distill-Qwen-32B` despite having fewer parameters.
  - Scaling with compute (Figure 5; Eq. (5))
    - Accuracy increases consistently with RL training compute across all difficulty splits, with high fit quality:
      - Easy: R²=0.995
      - Medium: R²=0.997
      - Hard: R²=0.989
    - This supports the claim that `RPT` is a scalable pretraining objective.
  - RL fine-tuning after pretraining (Table 2)
    - Before RLVR: `RPT-14B` = 56.3 vs `R1-Distill-Qwen-14B` = 51.2.
    - After RLVR: `RPT-14B` = 58.3 vs `R1-Distill-Qwen-14B` = 52.7.
    - Continuing NTP on the same corpus harms reasoning (10.7 before RL, 13.0 after RL), indicating objective mismatch; `RPT` avoids that gap (Section 4.3).
  - Zero-shot general-domain tasks (Table 3)
    - SuperGPQA accuracy:
      - `RPT-14B` (reasoning mode) = 39.0
      - `R1-Distill-Qwen-14B` (reasoning) = 36.1
      - `R1-Distill-Qwen-32B` (standard NTP) = 37.2
    - MMLU-Pro accuracy:
      - `RPT-14B` (reasoning mode) = 71.1
      - `R1-Distill-Qwen-14B` (reasoning) = 68.9
      - `R1-Distill-Qwen-32B` (standard NTP) = 56.5
    - Per-category details are in Appendix C (Tables 6–7).
  - Reasoning-pattern analysis (Figure 6; Table 4; Appendix F)
    - Compared to problem solving with `R1-Distill-Qwen-14B`, `RPT-14B` uses more “hypothesis” (+161.8%) and “deduction” (+26.2%) patterns and less “breakdown,” suggesting different internal strategies during next-token reasoning (Section 4.5). Examples illustrate deliberation and self-critique before committing to a token (Table 4; Appendix F, Table 11).

- Ablations and robustness checks
  - Reward variants in Appendix A performed comparably, indicating reward robustness.
  - Prompt templates affect initial correctness; clearer prompts improved pass rates (Appendix D; Table 8), though main experiments used a conservative template (v0).
  - Hyperparameters are listed in Appendix B (Table 5) to aid reproducibility.

- Overall assessment
  - The experiments directly measure what `RPT` optimizes (next-token accuracy) and show consistent improvements, including compute scaling and transfer to later RL fine-tuning (Tables 1–2, Figure 5).
  - Generalization to non-math domains is partly supported by strong MMLU-Pro improvements (Table 3), though the pretraining corpus is math-heavy, so broader pretraining tests remain to be done (Section 6).

## 6. Limitations and Trade-offs
- Data domain and size
  - Pretraining uses OmniMATH, a math-centric corpus (Section 3.3). While zero-shot general benchmarks improve (Table 3), it remains unclear how `RPT` performs when pretraining on diverse web-scale text (acknowledged in Section 6).

- Initialization and fairness of comparison
  - Training starts from a reasoning-capable model (`R1-Distill-Qwen-14B`), not a plain base LM (Section 3.3). The paper notes evaluating `RPT` from a non-reasoning base as future work (Section 6).

- Metric choice and evaluation scope
  - The primary LM metric is next-token accuracy on 200 validation samples with entropy-based splits (Section 4.1). Unlike standard perplexity, this metric can be sensitive to tokenization and byte-boundary details.
  - The “reasoning mode” baseline for `R1-Distill-Qwen-14B` performs very poorly on token accuracy (Table 1), which strengthens `RPT`’s case but also reflects the need for careful training when adding thinking tokens.

- Computational cost
  - On-policy RL with `G=8` long rollouts per prefix, 8k token budgets, and 14B parameters is compute-intensive (Section 3.3; Appendix B). Although results scale with compute (Figure 5), cost may be substantial at web scale.

- Reward granularity
  - The main reward is binary (Eq. (3)). While Appendix A explores denser variants, the core approach does not use partial credit, which might limit learning signals when the model is “close” but incorrect.

- Entropy-filter design
  - Difficulty filtering uses a specific proxy model and top-16 entropy. While practical, this introduces dependence on proxy behavior and a hand-chosen entropy thresholding scheme (Section 3.3).

## 7. Implications and Future Directions
- How this changes the landscape
  - `RPT` is a credible path to make RL a first-class pretraining objective, not just a post-training tool. It offers a scalable, verifiable reward that exists everywhere next tokens exist—i.e., across the entire web-text corpus (Sections 1, 3.2, Figure 1).
  - The compute scaling curve (Figure 5; Eq. (5)) suggests predictable gains with more compute, akin to classical NTP scaling laws but now for RL-style pretraining.

- Follow-up research directions
  - Web-scale `RPT`: Train on general Internet text to test domain breadth and establish scaling laws rigorously (Section 6).
  - From-scratch studies: Start `RPT` from non-reasoning base models to quantify gains attributable purely to the `RPT` objective (Section 6).
  - Reward shaping: Explore structured partial-credit rewards (Appendix A hints at dense variants) to accelerate learning for near-miss predictions.
  - Adaptive thinking: Integrate “hybrid/adaptive reasoning” that triggers thinking only when necessary (Section 6 references hybrid thinking [JWH+25]).
  - Efficiency: Improve sampling policies, curriculum over difficulty, and off-policy or replay methods compatible with verifiable rewards to reduce compute.

- Practical applications
  - Pretraining foundations: Organizations can produce stronger “RL-ready” checkpoints that fine-tune faster and higher with RLVR or preference optimization, as illustrated by Table 2.
  - Reasoning-intensive domains: Mathematics, law, and scientific writing where correctness is verifiable at fine granularity may benefit most early; Table 3 indicates broader gains on general academic QA benchmarks.
  - Safer RL objectives: Rule-based verifiers (exact-match next token) reduce reward hacking compared to learned rewards, potentially improving reliability in large-scale training pipelines.

> Bottom line: By converting ubiquitous next-token prediction into a verifiable RL problem, `RPT` provides a scalable bridge between self-supervised pretraining and reinforcement learning. The initial results—better token prediction, compute-scaling behavior, stronger starting points for RL fine-tuning, and improved zero-shot reasoning—make `RPT` a promising foundation for future, web-scale RL-based pretraining.
