# Direct Preference Optimization: Your Language Model is Secretly a Reward Model

**ArXiv:** [2305.18290](https://arxiv.org/abs/2305.18290)

## 🎯 Pitch

This paper introduces Direct Preference Optimization (DPO), a novel approach that reframes reinforcement learning from human feedback (RLHF) as a simple supervised classification task, sidestepping the need for explicit reward modeling and reinforcement learning altogether. By analytically connecting language model policies and reward functions, DPO achieves or surpasses RLHF performance on tasks like summarization and dialogue with greater stability, efficiency, and ease of implementation—significantly lowering the computational and practical barriers to building aligned and controllable AI systems.

---

## 1. Executive Summary (2–3 sentences)
This paper introduces Direct Preference Optimization (DPO), a method that turns the standard “reinforcement learning from human feedback” (RLHF) objective into a simple supervised classification loss. By reparameterizing the reward in terms of the policy itself and a reference model, DPO eliminates reinforcement learning and on-policy sampling while matching or exceeding PPO-based RLHF across sentiment control, summarization, and dialogue (see Figures 2–3).

## 2. Context and Motivation
- Problem/gap:
  - Modern language models learn broad capabilities from large-scale pretraining but are hard to steer precisely toward desirable behaviors. The de-facto approach—RLHF—requires fitting a separate reward model and then doing RL to optimize a KL-constrained reward objective (Section 3; Eq. 3). This pipeline is complex, unstable, and compute-heavy because it:
    - Trains multiple models (SFT model, reward model, RL policy).
    - Samples from the model in the loop during training.
    - Requires careful reward normalization and hyperparameter tuning.
- Importance:
  - Practically: simpler, stable, and scalable alignment methods reduce the barrier to deploying safer, more helpful systems.
  - Theoretically: shows that, under standard preference models (Bradley–Terry/Plackett–Luce), the optimal KL-regularized policy can be fit directly without RL by exploiting a closed-form mapping between reward and policy (Sections 4–5).
- Prior approaches and shortcomings:
  - Supervised fine-tuning (SFT) on demonstrations improves instruction following but collecting demonstrations is costly and does not use relative preference signals (Related Work, Section 2).
  - RLHF (PPO variants) learns a reward from pairwise preferences (via Bradley–Terry) and then optimizes the policy with a KL penalty (Sections 3 and 5.2). Issues: instability, reward normalization, need for a value baseline, extensive sampling, and sensitivity to hyperparameters.
- Positioning:
  - DPO keeps the same preference modeling assumption as prior RLHF (Bradley–Terry; Eq. 1) but replaces RL with a closed-form, supervised objective over the policy that is equivalent to maximizing the same KL-regularized reward (Section 4; Eq. 7). It thus preserves the benefits of preference-based alignment while removing an entire stage of the pipeline.

## 3. Technical Approach
At a high level, DPO starts from the standard RLHF objective and analytically eliminates the need for reinforcement learning.

Step 0: The standard RLHF objective
- RLHF typically maximizes expected reward while penalizing divergence from a reference policy `π_ref` (usually the SFT model), per input `x`:
  - Eq. 3: maximize over `πθ`
    - E_{x∼D, y∼πθ(·|x)}[ r_ϕ(x, y) ] − β KL( πθ(·|x) || π_ref(·|x) )
  - `β` controls how far the new policy may deviate from the reference; the KL term stabilizes optimization and keeps generations in the region where the reward model is accurate (Section 3).

Step 1: The optimal solution form under the KL constraint
- For any reward function `r(x,y)`, the optimal solution of Eq. 3 has the Boltzmann form (Eq. 4):
  - π_r(y|x) ∝ π_ref(y|x) · exp( r(x,y)/β ), with normalization `Z(x) = Σ_y π_ref(y|x) exp(r(x,y)/β)`.
  - This shows a direct mapping reward → optimal policy, but computing the partition function Z(x) is expensive in practice (Section 4; Appendix A.1).

Step 2: Invert the mapping (policy → reward up to a constant)
- Take logs of Eq. 4 to express the reward in terms of the optimal policy π_r and the reference:
  - Eq. 5: r(x,y) = β log [ π_r(y|x) / π_ref(y|x) ] + β log Z(x).
- Key observation: common preference models (Bradley–Terry, Plackett–Luce) depend only on reward differences for the same prompt `x`, so the unknown `β log Z(x)` cancels (Section 4).

Step 3: Plug this into the preference likelihood
- Under Bradley–Terry (pairwise preferences), the probability that completion y₁ is preferred to y₂ depends on the reward difference (Eq. 1). Replacing `r` with Eq. 5 yields (Eq. 6):
  - p(y₁ ≻ y₂ | x) = σ( β log [π*(y₁|x)/π_ref(y₁|x)] − β log [π*(y₂|x)/π_ref(y₂|x)] ),
  - where `π*` is the (unknown) optimal policy and `σ` is the logistic sigmoid.

Step 4: Turn preference fitting into supervised learning over the policy
- Parameterize the policy directly as `πθ` and maximize the likelihood of observed human preferences D = {(x, y_w, y_l)}:
  - Eq. 7 (DPO loss): minimize
    - − E_{(x, y_w, y_l) ∼ D} log σ( β[log πθ(y_w|x) − log π_ref(y_w|x)] − β[log πθ(y_l|x) − log π_ref(y_l|x)] ).
- This is just binary cross-entropy on a log-odds margin between preferred and dispreferred responses, adjusted by their log-probability under the reference model. No RL loop or on-policy sampling is needed.

How the update behaves (mechanics)
- The gradient (Section 4; Appendix A.4) is:
  - Increase log-probability of `y_w` and decrease that of `y_l`.
  - Weight the step by how incorrectly the current policy’s implicit reward `r̂θ(x,y) = β log[πθ(y|x)/π_ref(y|x)]` ranks the pair (a large weight if the model currently prefers the wrong response). The weighting comes from σ( r̂θ(x, y_l) − r̂θ(x, y_w) ) (Section 4, “What does the DPO update do?”).
- Why this matters: a naïve objective that only increases the ratio πθ(y_w)/πθ(y_l) can cause degeneration; DPO’s per-example weighting stabilizes learning and preserves diversity (Section 4; Appendix Table 3).

Data pipeline and implementation
- Pipeline (Section 4, “DPO outline”):
  1) Collect preference pairs for prompts by sampling two completions from a reference `π_ref` (often the SFT model) and labeling the preferred one.
  2) Train `πθ` to minimize Eq. 7. If an SFT `π_ref` is unavailable, estimate it by maximum likelihood on preferred completions to reduce distribution shift.
- A minimal PyTorch implementation fits in ~10 lines (Appendix B). Default hyperparameters include β=0.1 or 0.5 (for summarization), batch size 64, RMSprop, LR=1e-6 with warmup (Appendix B).

Theoretical underpinning: “Your language model is secretly a reward model”
- Under Bradley–Terry/Plackett–Luce, rewards are identifiable only up to an additive function of the prompt (`f(x)`); such reward “equivalence classes” induce the same preferences (Lemma 1) and the same optimal KL-regularized policy (Lemma 2).
- Theorem 1 (Section 5; Appendix A.6) shows any such reward class can be represented as:
  - r(x,y) = β log [π(y|x)/π_ref(y|x)] for some policy π.
- Intuition: applying the “projection” f(r; π_ref, β) subtracts the log partition term so the induced optimal policy is properly normalized (Eq. 9). Thus optimizing the policy via Eq. 7 is equivalent to fitting a reward in the correct class and then extracting its optimal KL-regularized policy.

Diagnosing PPO instability (Section 5.2)
- Re-expressing the PPO target as KL to the optimal policy (Eq. 10) exposes a normalization term (a soft value function baseline) that must be estimated. Missing or poorly estimated baselines create high variance and instability. DPO’s reparameterized reward does not require such a baseline, avoiding this source of instability.

## 4. Key Insights and Innovations
- Closed-form elimination of RL in RLHF:
  - Innovation: transform the KL-regularized RL objective into a supervised preference-classification loss over policies (Eq. 7), using the analytical mapping between reward and policy (Eq. 4–6).
  - Significance: removes on-policy sampling, value baselines, and PPO-specific hyperparameter tuning, yielding a simpler and more stable pipeline (Sections 4–5).
- Implicit reward via policy/reference log-odds:
  - Innovation: define an implicit reward r̂θ(x,y)=β log[πθ(y|x)/π_ref(y|x)] and directly fit it by maximizing preference likelihood (Section 4; gradient discussion).
  - Significance: unifies “reward modeling” and “policy optimization”—the policy itself serves as the reward model’s representative within the equivalence class (Theorem 1).
- Stability through weighted updates:
  - Innovation: the per-example weight σ( r̂θ(x,y_l) − r̂θ(x,y_w) ) focuses learning on pairs where the model is most wrong while preventing runaway updates (Section 4). A naïve ratio objective leads to degeneration (Appendix Table 3).
  - Significance: preserves diversity and avoids mode collapse without explicit KL penalties in the loss (the KL control is baked into the reparameterization).
- Practical wins with less complexity:
  - Empirical finding: on sentiment control, DPO strictly dominates PPO and other baselines on the reward–KL frontier (Figure 2 left). On summarization and dialogue, DPO matches or exceeds PPO and is far more robust to sampling temperature (Figure 2 right; Figure 3 left).

## 5. Experimental Analysis
Evaluation methodology
- Tasks (Section 6):
  - Controlled sentiment generation (IMDb; Section 6.1): prompts are short prefixes; rewards from a pretrained sentiment classifier (ground-truth reward).
  - Summarization (Reddit TL;DR; Section 6.2): compare generated summaries to reference human-written summaries; win rate judged by GPT-4; SFT model from TRLX is used as `π_ref`.
  - Single-turn dialogue (Anthropic Helpful–Harmless; Section 6.2): responses evaluated by GPT-4 against the human-chosen reference response.
- Baselines (Section 6):
  - SFT; Preferred-FT (supervised on chosen responses); Unlikelihood; PPO with learned reward; PPO-GT (oracle reward in sentiment); Best of N (sample N responses, pick highest under learned reward); zero-/few-shot prompting (GPT-J, Pythia-2.8B).
- Metrics:
  - Sentiment: reward–KL frontier, where reward is the true classifier reward and KL is sequence-level KL to `π_ref` (Figure 2 left; footnote 3).
  - Summarization & dialogue: win rate vs references, judged by GPT-4. Human validation confirms GPT-4 correlates well with human judgments (Section 6.4; Table 2).

Main quantitative results
- Sentiment control (Figure 2 left):
  - DPO attains the highest reward at any fixed KL to the reference across 22 runs (varying β/targets), strictly dominating PPO (with learned reward) and PPO-GT (with true reward).
  - Interpretation: Given that DPO and PPO optimize the same KL-regularized objective, the more efficient frontier indicates DPO’s optimization is both more stable and more effective.
- Summarization (Figure 2 right):
  - DPO achieves a ~61% win rate vs reference at temperature 0.0; PPO peaks around ~57% at temperature 0.0.
  - DPO remains robust across temperatures, whereas PPO’s performance degrades toward the base model at higher temperatures.
  - Best-of-128 is outperformed by DPO while being computationally impractical at inference (requires 128 samples per prompt).
- Dialogue (Figure 3 left and right):
  - DPO is the only computationally feasible method that consistently beats the human-chosen responses in Anthropic HH one-turn dialogue (win rate > 0.5 across temperatures).
  - Best-of-128 reaches similar performance but is far more expensive; DPO reaches strong performance quickly during training (Figure 3 right).
- Out-of-distribution generalization (Table 1):
  - Apply the Reddit-trained summarizers to CNN/DailyMail articles; judge vs ground-truth summaries:
    - DPO wins 0.36 (temp 0), 0.31 (temp 0.25); PPO wins 0.26, 0.23.
    - Both are below 0.5 vs ground-truth (as expected), but DPO retains a clear advantage.
- Human study validating GPT-4 judgments (Section 6.4; Table 2):
  - For a high-quality matchup (DPO vs PPO-0), humans prefer DPO 58% of the time.
  - Agreement between humans and GPT-4 is comparable to inter-human agreement (e.g., 70–86% human–GPT-4 agreement; 65–87% human–human agreement, depending on prompt variant and matchup).
  - A “conciseness-aware” GPT-4 prompt (GPT-4 (C)) improves alignment with humans relative to a simpler prompt (GPT-4 (S)).

Ablations, robustness, and failure modes
- Unlikelihood training degenerates on complex tasks (Appendix Table 3 shows incoherent outputs), so it’s excluded from summarization/dialogue results (Appendix C.3).
- Best-of-N improves with N but plateaus around 64–128 (Appendix Figure 4), highlighting diminishing returns and cost.
- DPO’s performance is relatively stable across sampling temperatures (Figure 2 right; Figure 3 left).
- Training curves show early convergence for DPO (Figure 3 right).

Assessment of evidence
- The experiments directly test:
  - Optimization quality under a known reward (sentiment): DPO’s frontier dominance is strong evidence of optimization effectiveness.
  - Real-world tasks with proxy evaluators (GPT-4) and human validation: DPO consistently matches or exceeds PPO and other baselines, and human studies corroborate the evaluator.
- Caveat: results are on models up to ~6B parameters; large-scale replication on frontier models is left for future work (Discussion).

## 6. Limitations and Trade-offs
Assumptions and modeling choices
- Preference model assumption:
  - DPO relies on Bradley–Terry/Plackett–Luce models where only reward differences matter (Section 4; Appendix A.2–A.3). If real annotator behavior deviates from these assumptions, the likelihood used in Eq. 7 may be misspecified.
- Reference policy dependence:
  - The reparameterization requires a reference `π_ref` with nonzero support over completions (Theorem 1 assumptions). If `π_ref` is unavailable, DPO fits a proxy by MLE on preferred completions (Section 4, “DPO outline”), which may introduce mismatch.
- Hyperparameter β:
  - β controls the implicit KL strength via the log-odds margin. While authors report minimal tuning (β=0.1 or 0.5), different domains may require tuning, and no automated selection procedure is provided (Appendix B).
Scope and scenarios not addressed
- Multi-turn dialogue and long-horizon credit assignment are not directly studied; experiments focus on single-turn dialogue and sequence-level preferences.
- Safety/generalization:
  - Initial OOD results (Table 1) are promising but limited in scope. Robustness to distribution shift, adversarial prompts, and reward hacking is not deeply explored.
Computational aspects
- While DPO eliminates RL sampling, it still computes sequence log-probabilities under both `πθ` and `π_ref`. This is far cheaper than PPO but still nontrivial for very long generations.
- Best-of-N remains stronger in some contexts but is impractical at inference; DPO targets efficiency–quality trade-offs rather than brute-force selection.

Open questions
- How does over-optimization manifest in DPO (Section 7, “Limitations & Future Work” references a slight performance dip late in training; Figure 3 right)?
- How well does DPO scale to frontier model sizes and multi-attribute alignment (helpfulness, harmlessness, style) simultaneously?

## 7. Implications and Future Directions
How this work changes the field
- Conceptual shift: shows that the canonical RLHF objective can be optimized exactly with supervised learning under standard preference models. This reframes preference alignment as classification over policy/reference log-odds rather than RL.
- Practical impact: dramatically simplifies preference optimization (no reward model training, no RL loop, no on-policy sampling), lowering the barrier to training aligned models and enabling broader experimentation.

Follow-up research enabled/suggested
- Scaling and scope:
  - Apply DPO to larger models and multi-turn interactions; test compositional attributes (e.g., simultaneously controlling helpfulness, harmlessness, style).
- Data efficiency and self-training:
  - Combine DPO with synthetic preference generation (e.g., constitutional/self-play settings) and active selection of pairs; explore iterative self-labeling using the implicit reward r̂θ.
- Beyond Bradley–Terry:
  - Extend to richer preference structures (lists/rankings, partial orders, or continuous feedback) via the Plackett–Luce generalization (Appendix A.3) or alternative choice models.
- Objective variants and regularization:
  - Explore different divergences (f-divergences; see Related Work [15]) or dynamic β schedules; investigate calibration and uncertainty of the implicit reward.
- Safety and robustness:
  - Study failure cases, reward hacking under DPO, and OOD generalization at scale; integrate safety constraints into the loss.

Practical applications
- Any setting currently using RLHF: instruction following, summarization, coding assistants, dialog agents.
- Other modalities with pairwise preferences: image, audio, or multimodal generation where preference data is easier to collect than scalar rewards.

> Bottom line: DPO keeps the objective that makes RLHF work (maximize reward subject to a KL constraint; Eq. 3) but solves it with a closed-form reparameterization that turns the problem into supervised learning over preferences (Eq. 7). Empirically, it is simpler, more stable, and often better than PPO-based RLHF (Figures 2–3), with human validations supporting automated evaluations (Table 2).
