# Direct Preference Optimization: Your Language Model is Secretly a Reward Model

**ArXiv:** [2305.18290](https://arxiv.org/abs/2305.18290)

## 🎯 Pitch

This paper introduces Direct Preference Optimization (DPO), a novel, reinforcement learning–free approach for aligning language models with human preferences. By showing that the RLHF objective can be solved exactly with a simple cross-entropy loss—directly optimizing the language model policy rather than relying on complex, unstable reinforcement learning pipelines—DPO achieves equal or better alignment, controllability, and sample efficiency compared to state-of-the-art RLHF methods like PPO. This breakthrough dramatically reduces the complexity and resource requirements for training high-quality, human-aligned language models, making this alignment process far more accessible and robust.

---

## 1. Executive Summary (2-3 sentences)
The paper introduces Direct Preference Optimization (DPO), a simple, reinforcement-learning–free method for aligning language models with human preferences. It replaces the standard RLHF pipeline with a single cross-entropy objective that directly tunes a model’s policy to satisfy preference data, while implicitly representing the reward function inside the model itself; empirically, it matches or exceeds PPO-based RLHF across sentiment control, summarization, and single-turn dialogue (Figures 2–3).

## 2. Context and Motivation
- Problem/gap:
  - Modern language models are powerful but hard to steer precisely using pretraining alone. Typical alignment uses RL from human feedback (RLHF): learn a reward model from human comparisons and then fine-tune the policy to maximize that reward under a KL constraint (Section 3, Eq. 3).
  - RLHF is complex, unstable, and compute-hungry: it trains multiple models, samples from the policy in the loop, and needs careful tuning (Introduction; Figure 1 left).
- Importance:
  - Steering models toward helpful, safe, and high-quality responses without reproducing undesirable patterns is central to controllability and trustworthiness (Introduction).
  - Practically, simpler and more stable methods lower the barrier to deploying aligned models.
- Prior approaches and their limits:
  - Supervised fine-tuning (SFT) on demonstrations improves instruction following but requires expensive expert data and often underperforms preference-based methods (Related Work).
  - RLHF pipelines (e.g., PPO-based) deliver strong results but: require a separate reward model; need on-policy sampling and variance reduction; are sensitive to KL targets and other hyperparameters; and can be unstable (Sections 3 and 5.2).
- How this work positions itself:
  - DPO shows that the same RLHF objective (maximize reward subject to a KL constraint) can be solved exactly with a closed-form optimal policy and optimized via a simple binary classification loss over preference pairs (Sections 4–5; Figure 1 right). This removes the RL loop and the explicit reward model while preserving the standard preference modeling assumptions (Bradley–Terry/Plackett–Luce).

## 3. Technical Approach
Step-by-step overview (from RLHF to DPO):

1) The RLHF objective being “solved”
- Standard RLHF optimization (Section 3, Eq. 3):
  - Maximize expected reward under the learned reward model `rϕ(x,y)` while staying close to a reference policy `πref` (typically the SFT model), enforced by a KL penalty weighted by `β`:
  - maximize over `πθ`: E_x,y~πθ [ rϕ(x,y) ] − β KL( πθ(y|x) || πref(y|x) )
- Intuition: achieve high reward but avoid drifting too far from where the reward model is accurate and where generations remain diverse.

2) Closed-form optimal policy for the KL-constrained problem
- Using standard results (Appendix A.1), the optimal solution for any reward function `r` has this form (Section 4, Eq. 4):
  - π_r(y|x) ∝ πref(y|x) · exp( r(x,y) / β ), with normalization Z(x).
- Problem: Z(x) is expensive to compute; also we usually do not know the true `r`.

3) Reparameterize the (unknown) reward in terms of the policy
- Rearranging Eq. 4 yields (Section 4, Eq. 5):
  - r(x,y) = β · log[ π_r(y|x) / πref(y|x) ] + β · log Z(x)
- Key: many preference models (Bradley–Terry, Plackett–Luce) depend only on reward differences between two responses. Differences cancel the `log Z(x)` term (Section 4, Eq. 6; Appendix A.2–A.3).

4) From reward modeling to policy modeling on preferences
- Bradley–Terry preference model (define): a widely used model of pairwise preferences where the probability that response y₁ beats y₂ is σ( r(x,y₁) − r(x,y₂) ), with σ the logistic function (Section 3, Eq. 1).
- Substitute the reward reparameterization (Eq. 5) into the Bradley–Terry likelihood; the normalization cancels, yielding a preference probability purely in terms of the policy vs. reference ratios (Section 4, Eq. 6):
  - p(y_w ≻ y_l | x) = σ{ β [ log π*(y_w|x) − log πref(y_w|x) − (log π*(y_l|x) − log πref(y_l|x)) ] }
- Replace the unknown optimal policy π* with a parametric policy πθ and maximize the log-likelihood of observed preference pairs (Section 4, Eq. 7):
  - Minimize binary cross-entropy:
  - LDPO(πθ; πref) = − E_(x,y_w,y_l)~D log σ( β[ log πθ(y_w|x) − log πref(y_w|x) − (log πθ(y_l|x) − log πref(y_l|x)) ] )

5) What the update is doing
- Gradient (Appendix A.4; Section 4):
  - ∇θ LDPO = −β E[ σ( r̂θ(x,y_l) − r̂θ(x,y_w) ) · ( ∇θ log πθ(y_w|x) − ∇θ log πθ(y_l|x) ) ]
  - with the implicit reward r̂θ(x,y) = β · log( πθ(y|x) / πref(y|x) ).
- Interpretation:
  - Increase probability of the preferred response and decrease that of the dispreferred one.
  - Weight each pair by how “wrong” the current model’s implicit reward ordering is (the σ term). This dynamic weighting stabilizes learning; a naive “always push the ratio” update can degenerate (Section 4; see also the qualitative failures of the “Unlikelihood” variant in Appendix Table 3).

6) Practical pipeline
- Data: an offline dataset of preference pairs D = { (x, y_w, y_l) } from human or AI annotators (Section 4; Figure 1).
- Reference policy `πref`:
  - If available, use the SFT model (Section 4).
  - If not, fit a reference by MLE on preferred completions: πref = argmax_π E[log π(y_w|x)] (Section 4).
- Optimization: compute token-level log-probs for `y_w` and `y_l` under both πθ and πref; apply the DPO loss (Appendix B includes a ~10-line PyTorch function).
- Hyperparameters: default β ≈ 0.1 (β = 0.5 for TL;DR), RMSprop with lr 1e-6 and warmup (Appendix B).

7) Why this works theoretically (“your LM is secretly a reward model”)
- Preference models (Bradley–Terry / Plackett–Luce) only identify rewards up to an additive function of the prompt f(x). Rewards that differ by f(x) define the same preference distribution and the same optimal policy under the KL constraint (Section 5.1; Lemmas 1–2; Appendix A.5).
- Theorem 1 (Section 5.1; Appendix A.6): Every equivalence class of rewards (under those preference models) has a unique representative of the form r(x,y) = β log[ π(y|x) / πref(y|x) ] for some policy π. In other words, the policy defines an implicit reward—hence “the language model is a reward model.”
- Stability insight (Section 5.2; Eq. 10): actor–critic optimization of the original RLHF objective needs a soft value normalization term; omitting or poorly estimating it causes high variance and instability. DPO’s reparameterization bakes normalization into the implicit reward, avoiding value baselines and improving stability.

Analogy
- Think of each preference pair as a tug-of-war between two responses. DPO shifts the model’s relative log-probabilities in favor of the winner, but the strength of the shift depends on how confidently the current model gets the pair wrong—large corrections where needed, small nudges where it already agrees.

## 4. Key Insights and Innovations
- Change-of-variables from reward to policy (fundamental):
  - Novelty: transforms the KL-constrained reward maximization (Eq. 3) into a binary classification over preferences that depends only on the policy-to-reference log-probability ratios (Eqs. 5–7).
  - Significance: eliminates the separate reward model and the RL loop, while exactly optimizing the same underlying objective assumed in RLHF.
- “Implicit reward” inside the LM (theoretical advance):
  - Theorem 1 shows any reward consistent with Bradley–Terry/Plackett–Luce can be represented as r(x,y) = β log[π/πref], up to an equivalence class (Section 5.1). This clarifies identifiability and links preference data directly to the policy.
- Stability via dynamically weighted updates (practical innovation):
  - The gradient’s σ-weight emphasizes pairs the model orders incorrectly (Section 4; Appendix A.4). This guards against degenerate updates seen in naive ratio or “unlikelihood” objectives (Appendix Table 3, with failure examples in Appendix Table 3 and D.2).
- RL-free, sampling-free training loop (engineering simplification):
  - No on-policy sampling, no value baselines, no separate reward network. A short loss function (Appendix B) suffices; this reduces compute and tuning burden while delivering competitive or better results (Figures 2–3).

## 5. Experimental Analysis
Evaluation setup (Section 6; Appendix C)
- Tasks:
  - Controlled sentiment generation on IMDb: prompts are review prefixes; goal is positive sentiment. “Ground-truth” reward provided by a pretrained sentiment classifier—enables computing reward–KL frontiers (Section 6.1).
  - Reddit TL;DR summarization: align generations to human preferences collected in prior work (Section 6.2). Reference model is an SFT GPT-J summarizer; evaluation is GPT-4 “win rate” vs human-written references across sampling temperatures (Figure 2 right; Appendix C.2 for GPT-4 prompts).
  - Single-turn dialogue (Anthropic HH): align to helpful & harmless responses (Section 6.2). Evaluate GPT-4 win rate vs the “chosen” (preferred) response in the dataset (Figure 3 left).
- Baselines:
  - SFT; Preferred-FT (supervised on chosen responses); Unlikelihood training; PPO with learned reward; PPO-GT (oracle using ground-truth reward in sentiment); Best-of-N (sample N responses and choose the best under a learned reward) (Sections 6.1–6.2).
- Metrics:
  - Sentiment: expected reward vs sequence-level KL to reference (Figure 2 left).
  - Summarization & dialogue: GPT-4 win rate vs a fixed baseline; human validation of GPT-4 judgments (Section 6.4; Table 2).
- Key results
  - Controlled sentiment (Figure 2 left):
    - Quote: “DPO provides the highest expected reward for all KL values,” strictly dominating PPO and even PPO-GT across KL ∈ [~0–20].
    - Interpretation: for any allowed deviation from the reference, DPO achieves better reward, indicating more efficient optimization of the RLHF objective.
  - TL;DR summarization (Figure 2 right):
    - At temperature 0.0, DPO wins ≈61% vs reference summaries, outperforming PPO’s best case ≈57% at the same temperature.
    - DPO remains robust across temperatures (0.0–1.0), whereas PPO degrades toward the base model as temperature increases.
    - DPO meets or beats Best-of-128 without the heavy test-time sampling cost.
    - Human study (Table 2): with a concise GPT-4 rubric, humans prefer DPO over PPO-0 in 58% of cases; GPT-4 and humans agree at rates comparable to inter-human agreement (≈67–86% vs ≈65–87%).
  - Single-turn dialogue (Figure 3):
    - DPO is “the only method that improves over chosen summaries in the Anthropic-HH test set” (Figure 3 left caption). It matches or exceeds Best-of-128 across temperatures.
    - Training dynamics (Figure 3 right): win rate gains are attained quickly and remain stable over training.
  - Out-of-distribution generalization (Table 1):
    - Evaluating TL;DR-trained policies on CNN/DailyMail news: DPO wins 0.36 (temp 0) vs PPO’s 0.26; 0.31 vs 0.23 (temp 0.25). DPO maintains a margin under distribution shift.

- Ablations/robustness/failures:
  - “Unlikelihood” often collapses on complex tasks (Appendix Table 3 shows degenerate repetitive outputs).
  - The paper remarks that naive probability-ratio objectives without the σ-weighting can degenerate (Section 4); DPO’s weighting mitigates this.
  - Best-of-N saturates around N∈[64,128] (Appendix Figure 4), still trailing DPO in efficiency since it requires many samples per query.

- Do the experiments support the claims?
  - The reward–KL frontier (Figure 2 left) is a strong, diagnostic test that directly targets the RLHF objective; DPO’s frontier strictly dominates PPO/PPO-GT.
  - On real preference datasets, DPO consistently equals or improves upon PPO and strong sampling baselines, with additional evidence of robustness to sampling temperature and out-of-distribution inputs (Figures 2–3; Table 1).
  - Human validation (Table 2) increases confidence in GPT-4 win-rate conclusions.

## 6. Limitations and Trade-offs
- Assumptions about preference modeling:
  - Relies on Bradley–Terry / Plackett–Luce models where only reward differences matter (Sections 3–4). If real human preferences deviate from these models (e.g., context-dependent in ways not captured by pairwise differences), the mapping may be misspecified.
- Dependence on a good reference:
  - The KL is to `πref`, typically the SFT model used to generate preference data. If `πref` is missing or mismatched, the paper proposes fitting one to preferred completions (Section 4), but this is a heuristic; quality depends on data coverage.
- Offline setting:
  - DPO, as evaluated, learns from a fixed preference dataset (Section 4). It does not leverage unlabeled prompts via on-policy exploration, which PPO-based pipelines can (potentially) exploit.
- Hyperparameter β:
  - Controls the strength of the KL constraint. Although DPO appears robust (e.g., TL;DR used a single β=0.5; Appendix B), tuning β still trades off conservativeness vs. reward.
- Evaluation proxies:
  - GPT-4 win rates are used as a proxy for human judgments. While validated (Table 2), the choice of evaluation prompt matters (Section 6.4 notes GPT-4 prompt sensitivity).
- Scale and scope:
  - Experiments go up to ≈6B-parameter models and three tasks. Behavior at frontier scales and across broader, multi-turn interactions remains to be studied (Discussion).

## 7. Implications and Future Directions
- Field impact:
  - DPO shows that preference learning for LMs can be reframed as supervised classification over policy/reference ratios—removing the RL loop while retaining the core KL-constrained objective. This simplification may shift the default alignment practice away from PPO-style RLHF for many use cases.
- Practical applications:
  - Faster, cheaper alignment for product-facing text models: summarization, customer support, code assistants (one can turn preference datasets directly into tuned policies with a short loss).
  - Safety tuning and style conditioning via targeted preference data without training a separate reward model or running PPO.
- Research directions:
  - Beyond pairwise: use the Plackett–Luce generalization for ranked lists (Appendix A.3), and explore listwise or multi-attribute preferences (safety, helpfulness, harmlessness).
  - Semi- and self-supervised preference learning: can DPO be combined with self-labeling (RLAIF) and unlabeled prompts while retaining stability?
  - OOD generalization and reward overoptimization: characterize when implicit-reward training does or does not overfit preference datasets (Section 7 notes slight late-training dips in Figure 3 right).
  - Scaling studies: apply DPO at 10s–100s of billions of parameters; test multi-turn dialogue and tool-use settings.
  - Other modalities: apply the same change-of-variables idea to images, audio, and multimodal generation (Discussion).

> Core take-away: By recognizing that, under standard preference models and a KL constraint, “the language model is secretly a reward model,” DPO turns RLHF into a stable, supervised optimization over policy/reference ratios (Eqs. 5–7). Figures 2–3 demonstrate that this simplicity does not sacrifice performance—in several cases, it improves it.
