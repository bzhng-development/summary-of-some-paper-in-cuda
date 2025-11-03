# Reinforcement Pre-Training

**ArXiv:** [2506.08007](https://arxiv.org/abs/2506.08007)

## 🎯 Pitch

This paper introduces Reinforcement Pre-Training (RPT), a paradigm that transforms next-token prediction for large language models into a scalable reinforcement learning (RL) objective—rewarding the model for reasoning and accurately predicting each token, using vast unannotated text as verifiable RL feedback. By unifying the scalability of self-supervised pre-training with the deeper reasoning benefits of RL, RPT achieves superior next-token accuracy, enhances zero-shot reasoning, and provides a robust, general foundation for further RL fine-tuning—significantly advancing how we build more powerful and general AI systems.

---

## 1. Executive Summary
This paper proposes Reinforcement Pre-Training (RPT), which turns standard next-token prediction into a verifiable reinforcement learning (RL) problem: the model “thinks,” predicts the next token, and receives a rule-based reward of 1 if the prediction matches the ground-truth next token and 0 otherwise (Figure 3; Eq. 3–4). RPT improves next-token accuracy and zero-shot reasoning and scales predictably with training compute (Table 1, Figure 4–5), positioning RL as a first-class, scalable pre-training objective rather than only a post-training technique.

## 2. Context and Motivation
- Gap addressed
  - Pre-training for large language models (LLMs) traditionally minimizes next-token negative log-likelihood (Eq. 1), which scales well but does not explicitly cultivate multi-step reasoning per prediction.
  - RL for LLMs has been powerful post-training (e.g., RL from human feedback and verifiable rewards), but it faces scalability issues: human preference data is costly and reward models can be exploited (“reward hacking”), while RL with verifiable rewards (RLVR) needs scarce annotated QA pairs (Section 1; “However…” paragraph).
- Why it matters
  - Unifying the scalability of self-supervised pre-training with the benefits of RL could let models allocate more computation per prediction during training, encouraging deeper reasoning and more robust representations (Section 1).
  - Turning web-scale text into a verifiable RL environment would allow RL to scale beyond small curated QA datasets (Figure 1; Section 1).
- Prior approaches and limitations
  - Standard next-token prediction (NTP) trains via maximum likelihood over tokens (Eq. 1).
  - RLHF improves alignment but needs costly human labels and introduces reward-model brittleness (Section 1).
  - RLVR uses objective, rule-based verifiers but depends on domain-specific, annotated answers, limiting scale (Section 1; Eq. 2).
- Positioning of this work
  - RPT reframes next-token prediction as “next-token reasoning”: for any prefix in a corpus, the model thinks, predicts the next token, and gets a verifiable reward using the ground-truth continuation (Section 3.1–3.2; Figure 2–3). This converts unannotated text into massive, objective RL supervision.

## 3. Technical Approach
RPT replaces the conventional likelihood objective with an on-policy RL objective built around verifiable next-token correctness, while explicitly eliciting multi-step reasoning before each prediction.

- Problem setup
  - Given a sequence `x0 … xT` from a corpus, for each position `t`, `x<t` is the context and `xt` is the ground-truth next token (Section 3.1).
  - The policy `πθ` must first generate a chain-of-thought `ct` and then a prediction `yt` for the next token: the full output is `ot = (ct, yt) ~ πθ(· | x<t)` (Section 3.1; Figure 2 shows the “think-then-predict” format).
    - Chain-of-thought is the model’s intermediate, free-form reasoning text; here it appears between `<think> … </think>`, and the final token prediction is extracted from the last `\boxed{…}` after `</think>` (Section 3.3).

- Reward design (how correctness is verified)
  - The paper uses a prefix-matching reward to support multi-token predictions and out-of-vocabulary tokens at the byte level (Section 3.2; Eq. 3).
  - Let `x≥t` be the ground-truth completion and `yit` the model’s predicted byte sequence of length `l`. Define `Lgt` as the set of valid cumulative byte lengths at the ground-truth token boundaries.
  - Reward for rollout `i` at position `t`:
    > r_it = 1 if the predicted bytes exactly match the first `l` bytes of `x≥t` and `l ∈ Lgt`; otherwise 0 (Eq. 3).
  - Intuition: reward is 1 only when the prediction is an exact next token (or exact multi-token unit when tokenization splits the symbol), preventing hacks like partial matches or crossing token boundaries.

- Optimization objective
  - Maximize expected reward over contexts and on-policy rollouts:
    > JRPT(θ) = E_(x<t,x≥t)∼D, {o_i_t}∼πθ(·|x<t) [ r_i_t ] (Eq. 4).
  - This replaces the classic next-token log-likelihood objective (Eq. 1) with an RL objective using verifiable signals from the corpus itself (Section 3.2).

- Training algorithm and implementation details
  - On-policy RL with GRPO (Group Relative Policy Optimization), a PPO-style algorithm with group-relative baselines (Section 3.3). Specific hyperparameters appear in Appendix B (Table 5).
  - Data
    - Corpus: OmniMATH (4,428 competition-level math problems and solutions) (Section 3.3).
    - Hard-token mining: a small proxy model (`Deepseek-R1-Distill-Qwen-1.5B`) computes top-16 entropy for each position; low-entropy (easy) positions are filtered out so training focuses on challenging tokens (Section 3.3).
  - Base model: `Deepseek-R1-Distill-Qwen-14B` (Section 3.3).
  - Key settings (Section 3.3; Appendix B):
    - Training length 8k, LR 1e-6, zero KL penalty, batch size 256 contexts.
    - For each context, sample `G = 8` on-policy rollouts at temperature 0.8.
    - Extract the model’s final prediction from the last `\boxed{…}` after `</think>`.
    - Dynamic sampling after 500 steps to improve efficiency (Section 3.3).
    - Main experiment: 1,000 training steps (Section 3.3).
  - Prompting the pretext task (Appendix D):
    - The model is prompted to “Complete the given text … by predicting the next token … reason step by step … wrap it in \boxed{}” (v0 template, Table 10). Variants of this template affect initial performance (Table 8).

- Alternative reward variants (robustness)
  - Appendix A explores (i) first-token-only matching, (ii) dense reward using model probability for incorrect predictions, and (iii) conditional dense reward per group of rollouts. Reported performance is “comparable” to the prefix reward, suggesting robustness to these reward choices (Appendix A).

- How this differs from standard NTP and prior RLVR
  - Unlike likelihood training (Eq. 1), RPT leverages an explicit reasoning phase and an RL objective tied to correctness.
  - Unlike RLVR on curated QA, RPT needs no external labels—reward comes from the known ground-truth next token in the corpus (Section 3.1–3.2; Figure 1).

A simple analogy: imagine each token position as a mini-quiz. The model can use scratch paper (the chain-of-thought) to reason. It then submits a one-token answer. The grader checks if it exactly matches the true next token (with byte-precise rules to avoid accidental partial credit). Points are awarded only on exact correctness. The policy is trained to produce thinking trajectories that more often lead to correct one-token answers.

## 4. Key Insights and Innovations
- Reframing next-token prediction as an RL problem with verifiable, intrinsic rewards
  - Novelty: The reward comes directly from the pre-training corpus—no learned reward model and no human labels (Section 3.1–3.2).
  - Significance: Makes RL scalable to web-text (Figure 1). Reduces reward hacking risk because the verifier is a simple, deterministic match (Section 1; Eq. 3).

- Next-token reasoning as a required intermediate computation
  - Novelty: The model must “think” before emitting the next token (Figure 2), operationalized via the `<think> … </think>` region and extracting the final answer from `\boxed{…}` (Section 3.3).
  - Significance: Encourages allocation of more compute per prediction during training ("apply inference-time scaling at training time for each token”; Section 1), improving accuracy especially on hard positions (Table 1).

- Prefix-matching reward at byte level with token-boundary validation
  - Novelty: The reward checks byte-level prefixes while enforcing valid token boundaries (`Lgt`), allowing multi-token symbols and out-of-vocab cases to be correctly verified (Section 3.2; Eq. 3).
  - Significance: Avoids ambiguous partial matches and supports realistic tokenizer quirks.

- Hard-position mining via entropy filtering
  - Novelty: A proxy model identifies high-entropy (uncertain) positions, and training emphasizes these challenging spots (Section 3.3).
  - Significance: Focuses RL compute where reasoning is most needed, which likely improves sample efficiency.

- Pretrain-then-finetune objective alignment for RL
  - Novelty: Because pre-training itself uses RL, there is less objective mismatch when later doing RLVR on downstream tasks (Section 3.4; 4.3).
  - Significance: Empirically yields better and faster gains during downstream RL (Table 2).

## 5. Experimental Analysis
- Evaluation methodology
  - Language modeling accuracy on a 200-sample held-out set from OmniMATH (Section 4.1). Token positions are split into easy/medium/hard by entropy thresholds 0.5/1.0/1.5 computed with `R1-Distill-Qwen-14B` (Section 4.1).
  - Scaling with compute: measure accuracy as training steps increase (100–1200 steps) and fit a compute-accuracy power law (Eq. 5) separately for easy/medium/hard splits (Section 4.2; Figure 5).
  - RL finetuning: continue training with verifiable rewards on Skywork-OR1 subset (256 train, 200 test), using `R1-Distill-Qwen-32B` for difficulty filtering and PPO-style settings (Section 4.3).
  - Zero-shot end tasks: SuperGPQA and MMLU-Pro, multiple-choice format, reasoning mode allows up to 12,288 tokens with T=0.8 (Section 4.4).
  - Reasoning-pattern analysis: keyword-based statistics over 200 sampled responses to compare patterns between problem solving and next-token reasoning (Figure 6; Section 4.5; Appendix E).

- Main quantitative results
  - Next-token prediction accuracy (Table 1)
    > Easy: RPT-14B 45.11% vs. R1-14B Standard 41.60%;  
    > Medium: 33.56% vs. 29.46%;  
    > Hard: 23.75% vs. 20.43%.
    - A “next-token reasoning” evaluation of the baseline `R1-14B` (without RPT training) performs very poorly (3.31/1.66/1.41%), showing that naive “think-then-predict” without RPT does not help (Table 1).
    - Figure 4 reports that RPT-14B matches the next-token accuracy of the larger `R1-Distill-Qwen-32B`, indicating strong efficiency gains.
  - Scaling behavior (Figure 5; Eq. 5)
    > Accuracy increases consistently with RL compute across all difficulty splits, with R² ≈ 0.995 (easy), 0.997 (medium), 0.989 (hard).
    - This supports the claim that RPT has favorable, predictable scaling with compute during pre-training.
  - Downstream RL finetuning (Table 2)
    > Before RL: RPT-14B = 56.3; R1-14B = 51.2; R1-14B + Continued NTP = 10.7.  
    > After RL: RPT-14B = 58.3; R1-14B = 52.7; R1-14B + Continued NTP = 13.0.
    - Continuing NTP on the same corpus severely degrades the model’s reasoning (10.7), and later RL recovers only slightly (13.0). In contrast, RPT both starts higher and ends higher, consistent with better objective alignment (Section 4.3).
  - Zero-shot end tasks (Table 3)
    > SuperGPQA: RPT-14B (reasoning) 39.0 vs. R1-14B (standard) 32.0 and R1-32B (standard) 37.2;  
    > MMLU-Pro: RPT-14B (reasoning) 71.1 vs. R1-14B (standard) 48.4 and R1-32B (standard) 56.5;  
    > R1-14B (reasoning) sits at 36.1 (SuperGPQA) and 68.9 (MMLU-Pro).
    - RPT-14B not only outperforms same-size baselines but surpasses the larger R1-32B in the standard evaluation mode.
  - Reasoning-style differences (Figure 6; Table 4)
    > RPT-14B shows a 161.8% higher use of “hypothesis” and 26.2% higher “deduction” patterns compared to problem solving with R1-14B, while R1-14B uses more “breakdown” (Section 4.5).
    - Qualitative examples (Table 4; Appendix F) illustrate deliberate exploration of alternatives and structural cues when deciding a single next token.

- Ablations and robustness checks
  - Reward variants (Appendix A): first-token-only, dense rewards, and conditional dense rewards perform “comparable” to the main prefix-matching reward, suggesting reward robustness.
  - Prompt template sensitivity (Appendix D; Table 8): some templates substantially improve initial correctness—best variant v6 achieves 6.0% Random@1 and 19.0% Pass@8 vs. the v0 template’s 3.0% and 8.5%. The main experiments use v0, implying headroom from prompt improvements.

- Do the experiments support the claims?
  - The gains on next-token accuracy across difficulty splits (Table 1), efficiency versus larger models (Figure 4), consistent scaling with compute (Figure 5), stronger starting point for RLVR (Table 2), and zero-shot improvements on challenging benchmarks (Table 3) collectively support the core claims that (i) RPT improves language modeling via reasoning and (ii) is a scalable RL-based pre-training paradigm.

## 6. Limitations and Trade-offs
- Data and domain scope
  - Pre-training is conducted on a math-centric corpus (OmniMATH, 4,428 problems; Section 3.3), so generality to broader domains, styles, and languages remains to be validated (explicitly noted in Conclusion).
- Initialization and fairness of comparison
  - RPT starts from a reasoning-distilled base model (`R1-Distill-Qwen-14B`; Section 3.3). It remains an open question how RPT behaves starting from a pure base language model (Conclusion).
- Compute and sampling overhead
  - On-policy RL with `G = 8` rollouts per context and long “thinking” traces (max response length 8192; Appendix B) raises training-time compute and memory costs relative to standard NTP.
- Reward sparsity and exactness
  - The 0/1 reward requires exact next-token matches (Eq. 3), which is inherently sparse. Appendix A explores denser alternatives but reports similar outcomes; detailed trade-offs across domains are not quantified.
- Sensitivity to prompting and extraction protocol
  - The method relies on special formatting (`<think> … </think>` and `\boxed{…}`) for separating reasoning from the final token (Section 3.3). Table 8 shows nontrivial sensitivity to template phrasing, suggesting that prompt design can influence training stability/efficacy.
- Data selection bias
  - Hard-token mining via a proxy model’s entropy (Section 3.3) focuses updates on uncertain positions; this may bias learning toward specific token types/patterns or reduce coverage of easy but frequent tokens.

## 7. Implications and Future Directions
- Field-level impact
  - RPT turns ubiquitous web text into a giant verifiable RL environment, making RL a scalable pre-training objective—not just a post-training add-on. This blurs the boundary between self-supervision and RL and opens a path to compute-scalable “reasoning-at-train-time” (Figure 1; Section 1; Figure 5).
- Practical applications
  - Pretraining models that are better at “thinking per token” may improve robustness on tasks requiring careful continuation (code, math, legal/technical text), and may provide stronger starting points for downstream RL with rule-based verifiers (Table 2).
- Follow-up research
  - Scale studies: establish formal scaling laws for RPT across model sizes, data sizes, and domains (Section 6 “Future Work”).
  - Broader corpora: apply RPT to general web-text beyond math to test generality and discover domain-specific reward variants (Conclusion).
  - Initialization studies: start from non-reasoning base models to isolate the pure contribution of RPT (Conclusion).
  - Adaptive thinking: integrate “hybrid thinking” to trigger reasoning only when needed (Section 6; reference [JWH+25]).
  - Reward shaping: explore richer verifiable rewards (e.g., byte-level edit distances with boundary constraints) or curriculum schedules that balance sparse correctness signals and dense guidance (Appendix A indicates robustness, but more systematic analyses could improve sample efficiency).
  - Efficiency: reduce rollouts or reuse off-policy data while maintaining on-policy benefits (Appendix B shows zero entropy regularization; investigating baselines/variance reduction, e.g., [HDW+25], could further stabilize training).

In short, the core conceptual leap is to treat each next-token prediction as a miniature RL problem with a deterministic, scalable verifier. Empirically, this yields better next-token accuracy, predictable scaling with compute, stronger foundations for downstream RL, and improved zero-shot reasoning—all while avoiding the annotation bottlenecks of conventional RL pipelines.
