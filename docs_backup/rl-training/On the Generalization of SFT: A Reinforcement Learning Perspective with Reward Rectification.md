# On the Generalization of SFT: A Reinforcement Learning Perspective with Reward Rectification

**ArXiv:** [2508.05629](https://arxiv.org/abs/2508.05629)
**Authors:** Yongliang Wu, Yizhou Zhou, Ziheng Zhou, Yingzhe Peng, Xinyu Ye, Xinting Hu, Wenbo Zhu, Lu Qi, Ming‑Hsuan Yang, Xu Yang
**Institutions:** 

## 🎯 Pitch

This paper introduces Dynamic Fine-Tuning (DFT), a simple yet powerful one-line adjustment to standard supervised fine-tuning (SFT) that enhances generalization across challenging reasoning tasks by reweighting per-token losses. By removing a hidden inverse-probability effect, DFT stabilizes updates and achieves performance gains comparable to reinforcement learning without the latter's computational complexities, making it a versatile solution when reinforcement learning is impractical.

---

## 1. Executive Summary (2-3 sentences)
This paper explains why standard supervised fine-tuning (`SFT`) of large language models often generalizes poorly compared to reinforcement learning (`RL`), and introduces a one-line change—Dynamic Fine-Tuning (`DFT`)—that fixes the underlying cause. By reweighting the per-token loss with the model’s own predicted probability, `DFT` removes a hidden inverse-probability effect in `SFT`’s gradient (Eq. 6), stabilizes updates, and yields strong gains across math reasoning, code generation, and multimodal benchmarks (Tables 1–4), even rivaling online RL methods (Table 2).

## 2. Context and Motivation
- Problem addressed
  - `SFT` is the default method to adapt LLMs to curated demonstrations (Eq. 1–2), but it often overfits and generalizes worse than RL on hard reasoning tasks (Intro; Section 2).
  - RL typically generalizes better by using explicit rewards or verification signals, but is compute-intensive, finicky to tune, and requires reward models or on-policy interaction (Intro; Section 2).

- Why this matters
  - Practical: Many real datasets contain only positive demonstrations (no negatives or preferences). Improving `SFT` directly is crucial when RL is too expensive or infeasible (Intro).
  - Theoretical: Clarifying how `SFT` relates to RL can reveal why `SFT` overfits and how to fix it systematically (Sections 3.1–3.3).

- Prior approaches and gaps
  - Hybrid pipelines combine `SFT` + RL or preference learning (e.g., PPO/GRPO, DPO, RAFT) but still need rewards, preference pairs, or extra models (Section 2).
  - Theory papers note connections between `SFT` and RL via weighting, but do not pin down the exact gradient equivalence that exposes `SFT`’s instability (Section 2).

- Positioning of this work
  - Section 3.2 proves that the `SFT` gradient equals an on-policy policy gradient with a sparse reward and a harmful importance weight `1/πθ(y|x)` (Eq. 5–6).
  - Section 3.3 proposes `DFT`, which multiplies the `SFT` loss by the model’s probability of the target token (with stop-gradient), canceling the inverse-probability term. This keeps the setting purely “SFT-style” (no reward model, no reference model, no online sampling).

## 3. Technical Approach
Step-by-step, the paper moves from diagnosis to a minimal fix.

- Preliminaries: what `SFT` and RL optimize
  - `SFT` minimizes negative log-likelihood of the expert response `y*` conditioned on `x` (Eq. 1): `LSFT(θ) = E[-log πθ(y*|x)]`. Its gradient is Eq. 2.
  - RL maximizes expected reward for sampled responses `y ~ πθ(·|x)` (Eq. 3), with the policy gradient `E[∇θ log πθ(y|x) r(x,y)]` (Eq. 4).

- Key derivation: rewriting `SFT` as a policy gradient (Section 3.2; Appendix A.3)
  - Convert the dataset expectation into an on-policy expectation by importance sampling:
    - Insert the model distribution and an indicator for the expert answer (Eq. 5):
      - `Ex Ey~πθ [ 1[y=y*] / πθ(y|x) * ( -∇θ log πθ(y|x) ) ]`.
    - This matches the policy gradient form (Eq. 6) with:
      - Reward `r(x,y) = 1[y = y*]` (sparse: nonzero only if the model exactly matches the demonstration).
      - Importance weight `w(y|x) = 1 / πθ(y|x)` (large when the model assigns low probability to the expert output).
  - Why this is problematic
    - When `πθ(y*|x)` is small, the weighting `1/πθ` explodes, making gradients huge and unstable, especially because the reward is sparse (Section 3.2).
    - This biases training toward rare exact matches, promoting overfitting rather than robust generalization.

- The fix: Dynamic Fine-Tuning (`DFT`) as reward rectification (Section 3.3)
  - Multiply the `SFT` gradient by the model’s own probability of the expert response, but stop gradients through that factor (Eq. 7):
    - `∇θ LDFT = ∇θ LSFT * sg(πθ(y*|x))`.
  - Equivalent loss (sequence-level, Eq. 8): `LDFT = E[ sg(πθ(y*|x)) * (-log πθ(y*|x)) ]`.
  - Practical form (token-level, Eq. 9): weight each token’s negative log-probability by its predicted probability, with stop-gradient to avoid feedback loops:
    - `LDFT = E[ -Σ_t sg(πθ(y*_t | y*_<t, x)) * log πθ(y*_t | y*_<t, x) ]`.
  - What this does mathematically (Appendix A.4)
    - The gradient becomes `-∇θ πθ(y*|x)` (Eq. 13), i.e., it directly increases the probability, rather than scaling by `1/πθ` as in cross-entropy.
    - Intuition: Cross-entropy amplifies updates on unlikely tokens by `1/π`; `DFT` removes this amplification, preventing outlier tokens from dominating updates.
  - Why token-level weighting?
    - Sequence-level probabilities can be extremely small (product of many token probabilities), causing numerical issues; PPO-like token-level treatment is standard (Section 3.3).

- Design choice rationale
  - `DFT` cancels the harmful inverse-probability weight that emerges when `SFT` is seen as RL (Eq. 6), turning the sparse reward “match-only” signal into a uniform 1 for all expert tokens.
  - This is the opposite philosophy of focal loss (Section 2): instead of upweighting “hard” tokens, `DFT` downweights them to reduce overfitting in the LLM regime where memorization is a bigger threat than underfitting.

- A simple implementation mental model
  - Replace the usual per-token loss term `-log p` with `-p * log p` and stop gradients through `p`. That is the “one-line change” (Section 3.3).

## 4. Key Insights and Innovations
- Precise gradient equivalence between `SFT` and an on-policy policy gradient with hidden importance weighting (Section 3.2; Eq. 5–6)
  - Novelty: Identifies the exact `1/πθ` term that makes `SFT`’s effective reward ill-posed and explains its poor generalization on sparse, exact-match supervision.

- `DFT`: a principled, minimal fix that rectifies the reward by neutralizing the `1/πθ` term (Section 3.3; Eq. 7–9)
  - Significance: Stabilizes optimization (removes gradient blow-ups), improves generalization, needs no reward model, no reference model, no preference pairs, and no on-policy sampling.

- Empirical breadth and strength with minimal complexity (Tables 1–4; Figure 1)
  - Gains across math reasoning, code generation, and multimodal reasoning, with faster convergence and better sample efficiency (Figure 1).

- Behavioral diagnosis at the token level (Figure 2)
  - Observation: `SFT` uniformly pushes probabilities higher (especially on low-prob tokens), whereas `DFT` polarizes—boosting some tokens and suppressing others; the suppressed set often contains glue words and punctuation, suggesting beneficial regularization (Figure 2; Section 4.5).

- Competitive in an offline RL setting without online rollouts (Table 2)
  - With rejection-sampling–based supervision (math verification), `DFT` surpasses offline methods like DPO/RAFT and even online PPO/GRPO on the 1.5B math model, offering an efficient alternative when interaction is costly.

## 5. Experimental Analysis
- Evaluation setup
  - Main `SFT`-style experiments (Section 4.1)
    - Training data: 100k examples from NuminaMath-CoT.
    - Models: `Qwen2.5-Math-1.5B/7B`, `LLaMA-3.2-3B`, `LLaMA-3.1-8B`, `DeepSeekMath-7B`.
    - Metrics: Accuracy@16 (average across 16 decoding runs; temperature 1.0; max length 4096).
    - Benchmarks: Math500, Minerva Math, OlympiadBench, AIME 2024, AMC 2023 (Table 1).
    - Optimization: AdamW, cosine decay, batch 256, warmup 0.1 (Section 4.1.1).

  - Offline RL-style experiments (Section 4.2)
    - Data creation: For 100k math prompts, generate 4 responses each with the base model; use a math verifier to keep correct responses (~140k). For DPO, build 100k positive–negative pairs (Section 4.2.1).
    - Baselines: DPO (offline), RAFT/RFT (offline), PPO (online), GRPO (online). `DFT` is applied offline.

  - Code generation (Section 4.3)
    - Training on 10k high-scoring UltraFeedback prompts; Benchmarks: HumanEval, HumanEval+, MultiPL-E (Table 3).

  - Multimodal reasoning (Section 4.4)
    - Train on WeThink; evaluate on MathVerse, MathVision, WeMath (Table 4).

- Main quantitative results
  - Math reasoning (Table 1)
    - `Qwen2.5-Math-1.5B` average accuracy: base 15.92 → `SFT` 18.01 → `DFT` 31.58.
      - On OlympiadBench: 15.88 → 12.63 (SFT drops) → 27.08 (DFT large gain).
      - On AMC23: 19.38 → 18.75 (SFT slightly drops) → 38.13 (DFT large gain).
    - `Qwen2.5-Math-7B` average: 21.25 → 23.62 → 37.15.
    - Similar patterns on `LLaMA-3.2-3B`, `LLaMA-3.1-8B`, `DeepSeekMath-7B`. Gains from `DFT` are consistently larger than `SFT`’s gains.
    - Learning dynamics (Figure 1): `DFT` reaches higher accuracy in fewer steps, achieves strong early-stage performance, and converges faster across all math benchmarks.

  - Offline RL setting (Table 2)
    - Average scores: `DPO` 23.20, `RFT` 23.97, `PPO` 28.66, `GRPO` 32.00, `DFT` 35.43.
    - Highlights:
      - Math500: `DFT` 64.71 vs `GRPO` 62.86 and `RFT` 48.23.
      - AMC23: `DFT` 48.44 vs `GRPO` 41.25 and `RFT` 30.78.
      - Minerva: `DFT` 25.16 vs `GRPO` 18.93 and `PPO` 15.41.

  - Code generation (Table 3)
    - `Qwen2.5-Coder-7B`: HumanEval 62.2 (base) → 54.9 (SFT) → 67.7 (`DFT`); MultiPL-E average 57.76 → 57.62 → 62.30.
    - `Qwen2.5-Coder-3B`: HumanEval 52.4 → 51.8 → 56.7; HumanEval+ 42.7 → 43.9 → 50.0.

  - Multimodal reasoning (Table 4)
    - `Qwen2.5-VL-3B` on MathVerse overall: 33.83 (base) → 35.66 (SFT) → 37.54 (`DFT`).
    - MathVision: 21.25 → 21.02 → 22.30.
    - WeMath: 4.10 → 23.33 → 23.71 (large jump from training on WeThink; `DFT` slightly ahead of `SFT`).

  - Token distribution analysis (Figure 2; Section 4.5)
    - Quote:
      > “SFT tends to uniformly increase token probabilities… DFT exhibits a polarizing effect… with more tokens in both the highest and lowest probability bins.”
    - Suppressed tokens in `DFT` include conjunctions/punctuation (“the”, “let”, “,”, “.”), suggesting useful de-emphasis of non-semantic tokens.

  - Additional studies and robustness
    - Comparison to concurrent iw-SFT (Appendix A.5; Table 5–6): `DFT` generally achieves higher averages across model families; also stronger in the offline setting (35.43 vs 31.86).
    - Higher-quality math data (OpenR1-Math-220k; Appendix A.6; Table 7): `Qwen2.5-Math-1.5B` average: 15.92 → 29.16 (SFT) → 38.19 (`DFT`).
    - PEFT/LoRA setting (Appendix A.7; Table 8): `DFT` maintains gains even with adapter tuning.
    - Hyperparameter ablations (Appendix A.8; Figure 3): `DFT` > `SFT` across learning rates and batch sizes; best performance near 1e-4 to 5e-5; batch size largely neutral.

- Do the experiments support the claims?
  - Yes, across multiple model sizes, tasks, and training regimes, `DFT` outperforms `SFT`, often by a wide margin on hard math tasks (Table 1) and even bests strong RL baselines in the offline setting (Table 2). Convergence and token-distribution analyses (Figures 1–2) align with the theoretical diagnosis and the intended stabilization.

- Caveats
  - Comparisons to online RL are on one base model and specific hyperparameters; RL could do better with additional tuning or larger compute.
  - The offline RL setup uses verifier-generated positives; the exact supervision signal differs across methods (Section 4.2.1), which could influence relative performance.

## 6. Limitations and Trade-offs
- Assumptions and scope
  - The fix targets `SFT` with exact-match demonstrations; benefits are largest when rewards are sparse and low-probability tokens would otherwise dominate gradients (Sections 3.2–3.3).
  - The paper does not claim universal superiority of `DFT`; Section A.2 notes domains focused on factual memorization may still prefer `SFT`.

- Scenarios not addressed
  - No experiments on very large frontier models or instruction-tuned datasets beyond math/code/vision-math; broader generalization remains to be tested (A.2).
  - No exploration of mixtures between `DFT` and `SFT` (e.g., adjustable weighting exponents).

- Computational and practical considerations
  - `DFT` is lightweight (no extra models, no online rollouts), but computing per-token probabilities is standard in `SFT` anyway; the “one-line change” is cheap.
  - While `DFT` stabilizes gradients, it also downweights hard, low-probability tokens. In tasks where rare tokens carry essential content, this could slow learning.

- Open questions
  - How does `DFT` interact with curriculum learning, data quality filters, or self-consistency decoding?
  - What is the best way to combine `DFT` with preference-based methods (DPO/RAFT) or with online RL when available?

## 7. Implications and Future Directions
- How this changes the landscape
  - Conceptual: Recasting `SFT` as RL (Eq. 6) isolates a precise failure mode—an implicit inverse-probability reward—that explains overfitting and instability. This reframing enables principled objectives beyond ad-hoc heuristics.
  - Practical: `DFT` is a drop-in replacement for `SFT` that improves generalization without added infrastructure. It is especially appealing when only positive demonstrations exist or when RL is too costly.

- Follow-up research enabled
  - Variants of the `DFT` weight: try `p^α log p` with `α ∈ (0,1]` to interpolate between `SFT` (`α=0`) and `DFT` (`α=1`).
  - Hybrid pipelines: warm-start with `DFT`, then apply RL or preference optimization; analyze whether `DFT` improves stability or sample efficiency downstream.
  - Theoretical guarantees: characterize generalization/calibration properties of the `-p log p` objective and its interaction with sequence length, label noise, and data diversity.

- Practical applications
  - Math and program synthesis, where rewards are naturally sparse and verifiable: `DFT` already shows strong gains (Tables 1–3).
  - Multimodal reasoning with structured outputs, where overfitting surface form tokens is harmful: `DFT` selectively emphasizes semantically informative tokens (Figure 2).
  - Low-resource or on-device fine-tuning (Appendix A.7): `DFT` works under LoRA, making it useful when full-parameter updates are infeasible.

> In short, Section 3 reveals an instability in `SFT`’s implicit reward structure (Eq. 6), and Sections 3.3–4 show that eliminating it through a per-token `p * log p` loss (Eq. 9) delivers consistent, cross-domain improvements (Tables 1–4), competitive offline-RL performance (Table 2), faster convergence (Figure 1), and more selective token fitting (Figure 2)—all with a one-line change.
