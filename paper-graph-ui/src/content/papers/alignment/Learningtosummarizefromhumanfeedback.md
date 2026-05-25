# Learning to summarize from human feedback

**ArXiv:** [2009.01325](https://arxiv.org/abs/2009.01325)

## 🎯 Pitch

This paper pioneers a method for training summarization models by directly optimizing for human preferences instead of relying on indirect metrics like ROUGE. By collecting human comparisons of summaries, learning a reward model to predict preferred outputs, and using reinforcement learning to fine-tune language models against this reward, the authors achieve significant improvements in summary quality and cross-domain generalization. This approach demonstrates that aligning training objectives with actual human judgments can produce AI systems that better reflect what people truly value—showing a promising path forward for more human-aligned and reliable natural language generation.

---

## 1. Executive Summary (2-3 sentences)
This paper shows how to train summarization models to directly optimize human preferences rather than proxy metrics like ROUGE. It builds a pipeline that collects human comparisons of summaries, trains a learned “reward model” to predict which summary a human prefers, and then fine-tunes a language model with reinforcement learning to maximize that reward—substantially improving summary quality and cross-domain transfer.

## 2. Context and Motivation
- Problem/gap:
  - Summarization models are typically fine-tuned to imitate human reference summaries and evaluated with automatic overlap metrics (e.g., ROUGE). These are only loose proxies for what people actually want—faithful, clear, useful summaries.
  - Issues with the usual objective (Section 1):
    - Maximum likelihood training treats all mistakes equally (fabricating facts versus choosing a weak synonym).
    - It encourages imitating low-quality demonstrations when those exist in the data.
    - During generation, distribution shift can degrade output quality; inference tricks like beam search can induce repetition and other artifacts.
  - ROUGE correlates poorly with human judgments for abstractive summarization (Section 1 and Related Work).

- Importance:
  - Practical: Better alignment with human judgments yields more helpful, less misleading summaries (low-risk domain for testing alignment strategies).
  - Scientific: Demonstrates that optimizing a learned human-preference signal can outperform optimizing a standard proxy metric and even the human-written references.

- Prior approaches and their shortcomings:
  - Supervised fine-tuning on references + ROUGE evaluation: brittle correlation with quality; incentivizes copying (Section 4.4; Appendix G.5).
  - RL tuned to automatic metrics (e.g., optimizing ROUGE): leads to gaming of the metric and degraded human-perceived quality (Figure 7; Appendix G.3).
  - Prior human-in-the-loop RL for summarization had labeler–researcher misalignment and extractive outputs (Related Work; comparisons to [73] throughout).

- Positioning:
  - Uses large GPT-style decoder models (1.3B–6.7B parameters; Section 3.4; Table 3).
  - Moves to a batch/offline human-feedback collection with rigorous quality control and labeler training (Section 3.3; Appendix C.1–C.2).
  - Separates the reward/value networks from the policy and adds a KL penalty to keep the policy within the reward model’s comfort zone (Section 3.4; Appendix G.1).

## 3. Technical Approach
The method has three iterative steps (Figure 2). Below, key terms are defined when first needed.

1) Collect pairwise human preferences
- Data: A filtered subset of the Reddit TL;DR dataset (123,169 posts with human-written TL;DRs) emphasizing general-population subreddits and constraining reference summary length to 24–48 tokens (Section 3.2; Appendix A).
- Process (Section 3.3; Appendix C.1–C.5):
  - For a given post, the system samples multiple candidate summaries from sources including the current policy, the supervised baseline, reference summaries, and baselines (Step 1 in Figure 2).
  - Labelers see pairs and choose which is the better summary of the post. They provide confidence on a 9-point scale and also write “naive interpretations” of each summary before seeing the post to surface ambiguity.
  - Labeler training and quality control:
    - Hands-on onboarding, shared chat, calibration tasks, and ongoing feedback.
    - Agreement rates: labelers agree with researchers 77% ± 2%, and researchers with each other 73% ± 4% on a representative subset (Section 3.3 and Appendix C.2).

2) Train a reward model (RM) from these comparisons
- Goal: Learn a scalar function `rθ(x, y)` that scores a summary `y` for a post `x` higher when humans prefer it.
- Architecture and loss (Section 3.4):
  - Start from the supervised baseline model; add a linear head that outputs a scalar reward.
  - Train with a Bradley–Terry–style objective: maximize the log-probability that the preferred summary has a higher reward than the other. Formally, minimize:
    - `loss = - E[ log σ(rθ(x, y_i) - rθ(x, y_{1-i})) ]`
  - Normalize the RM so that the dataset reference summaries have mean score 0.
- Practical notes (Appendix B.1):
  - Hyperparameters chosen by validation; batch size 64; one epoch over current data.
  - Reward model size matches the policy (1.3B or 6.7B parameters).

3) Optimize the policy with reinforcement learning (RL) against the RM
- RL objective (Section 3.4):
  - Treat the RM output as the reward for the whole generated summary (episode reward).
  - Use PPO (Proximal Policy Optimization) to update the policy, with time steps at the token level and discount `γ = 1` (no time preference).
- KL penalty (Section 3.4):
  - Add a regularization term to discourage the RL policy `π_RL` from drifting too far from the supervised baseline `π_SFT`.
  - Full reward: `R(x, y) = rθ(x, y) - β log[π_RL(y|x) / π_SFT(y|x)]`
  - Intuition: prevents reward hacking outside the RM’s training distribution and maintains diversity (entropy bonus-like effect).
- Value function separation (Appendix G.1):
  - Use a separate Transformer (initialized from the RM) to estimate returns/advantages.
  - This avoids destabilizing the policy early in training and empirically improves learning (Figure 11).
- Training regime (Appendix B.1):
  - PPO with 4 epochs per batch, λ = 0.95 for advantage estimation, KL coefficient β ~ 0.05 for the main runs, 1M episodes.

Alternative “training-free” optimizer for comparison
- Best-of-N (BoN) sampling (Appendix C.6; Table 10; Appendix G.3):
  - Sample N summaries from the supervised model at T = 0.7, score with RM, and pick the best.
  - Useful to study how optimizing a metric (RM or ROUGE) changes human-perceived quality without RL.

Evaluation safeguards and confound control
- Length control: Because longer summaries often score higher on “coverage,” the TL;DR dataset constrains references to 24–48 tokens. Further, the paper post-hoc controls for length via logistic/linear regression when comparing policies (Appendix F; Figure 10).
- Cross-checks: labeler–researcher agreement monitoring (Appendix C.2); validation sets for RM generalization and sensitivity to semantic edits (Section 4.3; Tables 17–18).

## 4. Key Insights and Innovations
1) Train summarizers to optimize a learned human preference signal rather than a proxy metric
- What’s new: A scalable, offline pipeline that (a) collects many pairwise human preferences, (b) learns a reward function, and (c) optimizes a large LM with RL against that reward (Figure 2; Sections 3.1–3.4).
- Why it matters: It significantly improves human-judged quality compared to strong supervised baselines, even surpassing the human-written references on TL;DR (Figure 1, Section 4.1).

2) Careful data and labeler management yields reliable preference signals
- What’s new: A robust process for onboarded, calibrated labelers with demonstrated high agreement to researcher judgments (Section 3.3; Appendix C.1–C.2).
- Why it matters: Prior work reported misalignment between labelers and researchers leading to poor outcomes; here, agreement hits ~77% labeler–researcher on a key subset, approaching researcher–researcher agreement (Appendix C.2).

3) RL with a KL penalty and separate value network stabilizes optimization
- What’s new: A KL-regularized PPO objective with a separate value net initialized from the reward model improves optimization stability and prevents mode collapse (Section 3.4; Appendix G.1).
- Why it matters: It mitigates reward over-optimization outside the RM’s training distribution while still improving quality (Figure 5 shows limits of over-optimization).

4) Reward models generalize across domains and outperform standard automatic metrics
- What’s new: A reward model trained on Reddit TL;DR preferences predicts human preferences on news summaries (CNN/DM) with 62–67% agreement, approaching inter-labeler agreement (Appendix G.7; Table 23).
- Why it matters: ROUGE’s agreement with humans degrades once models are optimized against human preference; the RM maintains predictive power (Section 4.4; Figure 7; Tables 20–23).

## 5. Experimental Analysis
Evaluation setup
- Datasets:
  - TL;DR (Reddit): 123,169 posts after filtering; reference TL;DRs restricted to 24–48 tokens; training/validation splits (Section 3.2; Appendix A).
  - CNN/DailyMail (CNN/DM) for transfer: zero fine-tuning on news for the main HF models (Section 4.2).
- Metrics:
  - Primary: human pairwise preference versus the dataset’s reference summaries (Figure 1).
  - Multi-axis Likert (1–7): coverage, accuracy, coherence, overall (Section 4.1; Figure 3; Appendix C.5).
  - Automatic: ROUGE; length; copying (longest common subsequence of bigrams); log-probability under supervised baselines; reward model scores (Section 4.4; Appendix G.4–G.7).
- Baselines:
  - “Pretrain only” few-shot prompts (GPT-style).
  - Supervised fine-tuning (SFT) on TL;DR.
  - T5 fine-tuned on CNN/DM (encoder–decoder; Appendix D).
  - Extractive baselines for CNN/DM: lead-3; reference highlights (Appendix E).
- Model sizes and settings:
  - 1.3B and 6.7B parameter GPT-style decoders for main HF experiments (Section 3.4; Table 3).
  - Sampling with T=0 for final model comparisons (Appendix B.1).

Main quantitative results
- TL;DR—preference vs reference summaries:
  > Figure 1 and Section 4.1: A 1.3B human-feedback (HF) model’s summaries are preferred to the reference 61% of the time; a 6.7B HF model is preferred roughly 70% of the time. A 6.7B supervised model lags (≈43% against reference noted for a 10× larger supervised model versus the 1.3B HF model). Length control reduces HF’s advantage by ~5 percentage points but the 6.7B HF model still wins ≈65% (Appendix F; Figure 10a).

- TL;DR—quality axes:
  > Figure 3: HF models improve across coverage, accuracy, coherence, and overall, with the largest gains in coverage. The 6.7B PPO model earns a perfect overall score (7/7) on 45% of examples, versus 20% (6.7B supervised) and 23% (reference).

- Transfer to CNN/DM (no news-specific fine-tuning):
  > Figure 4a: HF models trained only on TL;DR nearly match a 6.7B model fine-tuned on CNN/DM references on overall Likert quality; both beat pretrain-only and TL;DR-supervised baselines.
  > Figure 4b: At matched lengths, HF transfer models’ overall quality approaches the supervised news model; longer outputs would likely further increase scores (length–quality slope quantified in Table 14).

- On automatic metrics and their optimization:
  > Figure 7: Best-of-N optimization using ROUGE peaks early and at a lower human preference rate than optimizing any of the reward models. ROUGE’s correlation with human preferences drops toward chance once models are optimized for human feedback (Section 4.4 and Tables 20–22).

- Reward model behavior and robustness:
  > Figure 5: As PPO optimization strength increases (lower effective KL to the supervised baseline), the RM’s predicted score continues to rise but true human preference eventually decreases, showing over-optimization (“reward hacking”). Light-to-moderate optimization improves quality; heavy optimization harms it.
  > Figure 6: Reward model accuracy scales with both model size and data (>64k comparisons). Doubling data gives ~1.1% absolute gain; doubling model size gives ~1.8%.
  > Tables 17–18 (Section 4.3): RM prefers human-edited “improvements” almost as reliably as human judges (e.g., 6.7B RM: 83.7% vs humans 85.6%) and catches role-reversal errors (97.4%). It is biased toward longer outputs (prefers edits that shorten summaries only 66.0% vs humans 76.2%).

- Agreement matrices (Appendix G.7; Tables 20–23):
  > Reward models (especially 6.7B) agree with labelers about as well as non-expert labelers agree with each other, and sometimes better than ROUGE, length, copying, or supervised log-probability. On CNN/DM, the 6.7B RM reaches 66.5% agreement with labelers (Table 23).

- Additional checks:
  - Value-function ablation (Appendix G.1): Separate value and policy networks improve learning (Figure 11).
  - Copying behavior (Appendix G.5; Table 16): Supervised and HF models copy less from the source than pretrain-only; CNN/DM models generally copy more than TL;DR models.
  - Lead-3 vs reference on CNN/DM (Appendix E): Labelers often prefer lead-3 to the official reference highlights; after length-normalization lead-3 still slightly outperforms references, questioning the suitability of the reference highlights for “gold” evaluation.

Assessment
- The experiments are extensive and multi-angled: direct preference tests, multi-axis Likert ratings, ablations (value-function separation), optimization analyses (over-optimization, best-of-N), cross-domain transfer, and metric–human agreement studies.
- The body of evidence strongly supports the claims that:
  - Optimizing a learned human-preference signal improves quality over supervised imitation and ROUGE-optimized policies.
  - The learned reward generalizes beyond its training domain better than ROUGE.
- Limits are transparent: over-optimization risks (Figure 5), length bias (Tables 17–18), and compute/data costs (Discussion).

## 6. Limitations and Trade-offs
- Assumptions and scope:
  - Task definition is “a good summary for a reader who only sees the summary” (Section 3.2; Appendix C.5). This privileges self-contained clarity and may penalize concise summaries that rely on context (Appendix E discusses implications for CNN/DM).
  - Human preferences are treated as a single scalar objective; diverse user preferences or multi-objective trade-offs (e.g., brevity vs coverage) are only partially addressed via length control (Appendix F).

- Over-optimization and reward hacking:
  - Figure 5 shows that pushing the policy too far from the supervised baseline (low KL) leads to anti-correlation with human preferences. The KL penalty helps but does not eliminate the risk.

- Biases in the reward model:
  - Length bias: prefers longer summaries more than humans do (Tables 17–18).
  - Distributional bias: trained on a Reddit subset dominated by relationship-related posts (Appendix A, Table 2), though transfer experiments mitigate, not eliminate, the concern.

- Data and compute costs:
  - Human labeling: 64,832 pairwise comparisons; extensive labeler onboarding and monitoring (Abstract; Appendix C.6).
  - Compute: RL tuning of the 6.7B model took ~320 GPU-days (Discussion).

- Evaluation limitations:
  - CNN/DM reference highlights may be a noisy “gold” target (Appendix E), complicating comparison to prior work that optimizes those references.
  - ROUGE is shown to be unreliable for this setting (Section 4.4); human evaluation is required, which limits reproducibility and scalability across studies without shared human-labeling resources.

- Unaddressed scenarios:
  - Very long document summarization, languages other than English, and domains with strong factuality constraints were not directly studied.
  - Real-time or interactive labeling (online RLHF) is not explored here; the paper uses batch/offline collection.

## 7. Implications and Future Directions
- Field impact:
  - Establishes a practical recipe for “reinforcement learning from human feedback” (RLHF) that scales to billion-parameter language models and demonstrably beats both supervised imitation and metric-optimized systems (Figures 1, 4, 7).
  - Shifts evaluation norms: learned preference models can be better target signals than ROUGE, and they generalize better across domains (Section 4.4; Tables 20–23).

- Follow-up research directions:
  - Reward modeling
    - Reduce length bias and capture multi-objective preferences (coverage, brevity, factuality) via structured rewards or multi-task RMs.
    - Data efficiency: share reward models across tasks, or train multi-domain RMs (Discussion; Future directions).
    - Robustness to over-optimization via uncertainty-aware rewards or adversarial training against reward hacking (Figure 5 suggests need).
  - Policy optimization
    - Explore alternative regularizers to the KL penalty; adaptive KL or trust-region methods beyond PPO.
    - Token-level or span-level credit assignment by learning intermediate rewards rather than only end-of-sequence rewards.
  - Human-in-the-loop workflow
    - Mixed feedback types: edits, rationales, graded scales, and demonstrations (Discussion and Appendix G.6–G.7 show promise for edits).
    - Better tooling for labeler calibration and disagreement resolution; study demographic impacts (Appendix C.3).

- Practical applications:
  - Higher-quality summarization for consumer products (news, email/meeting notes, policy briefs) with better alignment to user preferences.
  - Template for applying RLHF to other generative tasks where automatic metrics are weak: dialogue, translation, code comments, review or story generation (Discussion).
  - Safety and alignment: Summarization is a low-risk testbed; the same approach can steer powerful models toward human-valued behavior on higher-stakes tasks (Introduction; Broader Impacts), provided harms (e.g., persuasive manipulation) are actively mitigated.

> Overall, the paper demonstrates that when the goal is “what people prefer,” learning that goal directly from human comparisons—and optimizing a model against that learned reward under careful regularization—produces better, more generalizable summarizers than optimizing proxy metrics like ROUGE. The method is clear, well-analyzed, and now foundational for RLHF-based language model alignment.
