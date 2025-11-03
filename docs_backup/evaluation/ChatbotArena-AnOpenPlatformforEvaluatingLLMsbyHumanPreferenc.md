# Chatbot Arena: An Open Platform for Evaluating LLMs by Human Preference

**ArXiv:** [2403.04132](https://arxiv.org/abs/2403.04132)

## 🎯 Pitch

Chatbot Arena introduces the first large-scale, open, and live evaluation platform where large language models compete in anonymous, crowdsourced 'battles' judged by direct human preference. By using statistically robust pairwise comparison methods and active sampling, this platform overcomes shortcomings of static, ground-truth benchmarks—capturing real-world usage diversity and human alignment in a credible, scalable, and transparent leaderboard now widely referenced by both researchers and industry. Chatbot Arena fundamentally shifts LLM assessment toward human-centric, evolving benchmarks, enabling more accurate and dynamic evaluation of AI capabilities as they advance.

---

## 1. Executive Summary (2-3 sentences)
This paper introduces Chatbot Arena, a live, open, crowdsourced platform that evaluates large language models (LLMs) by direct human preference through anonymous, head-to-head “battles.” It contributes a statistically principled ranking system (based on Bradley–Terry modeling with uncertainty-aware intervals and an active sampling rule) and shows that crowdsourced votes are diverse, discriminative, and largely consistent with expert judgments (Section 6; Tables 3–4), establishing a credible, scalable alternative to static, ground-truth benchmarks.

## 2. Context and Motivation
- Problem/gap addressed
  - Modern LLMs are used for open-ended tasks where “ground truth” is ambiguous or non-existent. Static benchmarks (e.g., MMLU, HellaSwag, GSM-8K) are closed-ended, can saturate, and become contaminated over time (Section 1; “Risks of Static Benchmarks,” Section 2).
  - There is no open, continuous, large-scale platform that evaluates models on live, real-world prompts using human preferences and produces statistically reliable rankings.

- Why it matters
  - Real-world deployments hinge on how well models align with human preferences, not just correctness on static datasets. A live preference-based platform mirrors actual user scenarios, mitigates contamination, and captures evolving capabilities (Figure 1 positions Chatbot Arena in the “live + human preference” quadrant).

- Prior approaches and shortcomings
  - Static, ground-truth benchmarks are reproducible and cheap, but poorly capture open-ended helpfulness and can be gamed (Section 1; Yang et al., 2023).
  - Some open-ended datasets solicit human ratings (e.g., OpenAssistant, Anthropic HH; Section 2, Table 1) but are not live platforms for model-vs-model evaluation at scale.
  - Prior leaderboards sometimes used Elo ratings for pairwise comparisons, but Elo offers weaker statistical estimation guarantees for this setting; this paper moves to Bradley–Terry with robust uncertainty quantification (Section 4).

- Positioning relative to existing work
  - Chatbot Arena is presented as the first open, large-scale, crowdsourced, live platform for LLM preference evaluation (Section 1, “We build the first large-scale crowd-sourced live LLM evaluation platform…”). It combines: free-form prompts, double-blind model battles, principled ranking, active sampling, prompt-topic analysis, and expert validation.

## 3. Technical Approach
The system has three layers: data collection via anonymous pairwise battles, statistical ranking with uncertainty and active sampling, and data quality analyses (topic coverage, discriminative power, agreement with experts).

1) Data collection interface (Section 3.1)
- A user asks any free-form question once; two anonymous LLMs (randomly chosen) both respond in a side-by-side “battle.”
- The user votes for the preferred answer or selects “tie”/“both are bad.” Model identities are revealed only after voting (to reduce bias).
- Content safeguards: conversations mentioning model/organization names are filtered to preserve anonymity; OpenAI’s moderation API flags unsafe content (~3% flagged; Section 3.2).
- Scale and diversity (Section 3.2; Table 1; Figures 9–10)
  - As of Jan 2024: ~240K votes from ~90K users across 100+ languages (77% English, 5% Chinese; Section 3.2).
  - >50 models evaluated (proprietary and open). Average ~8K votes per model (Figure 10).
  - Vote volume stabilized at 1–2K/day with spikes for new models/releases (Figure 9).
  - Non-uniform pair sampling focuses voting on uncertain pairs (Section 3.2), later formalized as an active sampling rule (Section 5).

2) From pairwise votes to statistically sound rankings (Sections 4–5)
- Notation and goals
  - Let `A` be the set of all unordered model pairs. At time `t`, the platform serves a pair `A_t ∈ A` and observes a human response `H_t ∈ {0,1}` (1 = second model preferred; Section 4). Ties can be incorporated but the exposition focuses on binary outcomes.
  - Win matrix `θ*(a) = E[H_t | A_t = a]`: the probability that model `a2` beats `a1` (Section 4; Figure 2 left visualizes empirical win rates).
- Estimating the win matrix (Section 5; Equations 4–6)
  - Use inverse-probability weighting to correct for non-uniform pair sampling:
    - For pair `a`, define `X_t(a) = H_t 1{A_t=a} / P_t(a)` where `P_t(a)` is the probability of sampling `a` at `t`.
    - Unbiased estimator: `θ̂_T = (1/T) Σ_t X_t` (Eq. 4).
    - Empirical covariance: `Σ̂_T = (1/T) Σ_t (X_t − θ̂_T)(X_t − θ̂_T)^T` (Eq. 5).
    - Under mild conditions, `√T Σ̂^{-1/2} (θ̂ − θ*) → N(0, I)` for confidence intervals (Eq. 6).
- Converting pairwise outcomes to model scores with Bradley–Terry (Section 4; Equations 2–3, 7)
  - Bradley–Terry (BT) score assigns each model `m` a coefficient `ξ_m` so that
    - Probability that `m` beats `m'`: `P(H_t=1) = 1 / (1 + e^{ξ_{m'} − ξ_m})` (Eq. 2).
  - Estimate `ξ` by minimizing the reweighted logistic cross-entropy over observed comparisons (Eq. 7), with inverse weights `1/P(A_t)` so the learned score corresponds to a uniform pair distribution.
  - Uncertainty quantification
    - Two options explored: pivot bootstrap and robust “sandwich” standard errors (Huber/White). Simulation/replay studies (Appendix A; Figure 13–14) show the sandwich intervals are stable and, in large samples, smaller, so they are adopted.
  - Approximate ranking with uniform confidence (Section 5)
    - Build an M-dimensional confidence set `C` for the BT vector ensuring `P(s(P) ∈ C) ≥ 1−α` (Eq. 8).
    - Derive an “approximate rank” by comparing interval bounds; e.g., no model is understated with probability ≥ `1−α` when ordering by interval extrema (Section 5).
    - Multiplicity correction uses a chi-square CLT region so all models’ intervals are jointly valid (Figure 5 shows with/without correction).
- Active sampling to accelerate convergence (Section 5; Equation 9; Section 7.1)
  - Intuition: sample pairs where additional votes most reduce uncertainty.
  - Rule: choose pair `a` with probability proportional to the reduction in its interval width,
    - `P_t(a) ∝ sqrt(Σ̂_{t,a,a})/n_a − sqrt(Σ̂_{t,a,a})/(n_a+1)` where `n_a` = number of times `a` has been sampled (Eq. 9).
  - Empirically, this reduces the number of votes needed to reach target precision (Figure 7).
- Detecting anomalous users (Section 5.1; Equation 10; Table 5)
  - For each new vote, compute a per-action p-value by comparing the user’s response to the historical distribution for that pair:
    - `p_i = (1/(|H_{A'_i}|+1))(1 + Σ_{h∈H_{A'_i}} 1{h ≥ H'_i})` (Eq. 10).
  - Combine a sequence of p-values with Fisher’s method at 5 random checkpoints (to reduce gaming) and flag users when `M_j ≥ χ^2_{2j, 1−α/5}`.
  - In a hand-labeled test of 25 anomalous and 25 normal users, the heuristic showed promising detection rates (Table 5).

3) Data analyses to assess coverage and discriminative power (Section 6)
- Topic modeling pipeline (Section 6.1; Figure 3; Appendix Figures 11–12)
  - `BERTopic` clusters prompts using OpenAI embeddings (`text-embedding-3-small`), dimensionality reduction with `UMAP`, and density-based clustering with `HDBSCAN` (min cluster size 32). Labels are generated by GPT‑4‑Turbo from samples.
  - 600 clusters found; the largest cluster is only ~1%, indicating a long-tail of diverse real-world prompts (Figure 3).
- Can prompts distinguish models? (Section 6.2; Table 2)
  - Using “LLM-as-judge,” GPT‑4‑Turbo evaluates GPT‑4 vs. Llama‑2‑70B on 30 prompts per topic. Results vary by topic:
    - “Python Game Programming Challenge”: GPT-4 wins 96.7%.
    - “C/C++ Multi-Threading”: 86.7%.
    - “Movie Recommendations & Ratings”: 53.3% (almost parity).
  - Implication: Arena prompts capture domains where strong models separate (coding/reasoning) and domains where models are comparable (recommender-like tasks).
- Building a challenging benchmark from Arena prompts (Figure 4; Appendix D.2–D.3)
  - “Arena Bench” selects high-quality, diverse prompts across clusters and evaluates models vs a baseline using GPT‑4 as judge. Compared to MT‑Bench, Arena Bench magnifies the performance gap between proprietary and leading open models (Figure 4).

## 4. Key Insights and Innovations
- Live, crowdsourced preference evaluation at scale
  - Innovation: a public site where users create fresh, free-form prompts and compare two anonymous models, yielding over 240K votes from ~90K users (Section 3.2; Table 1).
  - Why it matters: avoids contamination/saturation of static datasets and reflects evolving, real-world usage (Figure 1).

- Statistically principled ranking for pairwise LLM comparisons
  - Innovation: end-to-end pipeline from inverse-probability estimation of the win matrix (Eq. 4–6) to BT coefficients with uncertainty (Eq. 7, Section 5), with multiplicity-corrected confidence regions (Figure 5).
  - Significance: users get rankings with error bars; researchers get reliable comparisons across many models.

- Active sampling that reduces label complexity
  - Innovation: a simple, variance-aware rule (Eq. 9) that focuses on pairs likely to reduce uncertainty fastest.
  - Impact: reduces sample needs by 54% for a given precision target on the win matrix and by 5% on BT scores (Figure 7).

- Independent evidence of data quality and discriminative power
  - Prompt diversity and long-tail structure (Section 6.1; Figure 3) show broad coverage of use cases.
  - Crowd vs expert agreement between 72–83% (Table 3) supports credibility; GPT-4‑Turbo’s win rates are consistent across labeling sources (Table 4).

- Public dataset and reproducible analysis resources
  - The platform commits to releasing >100K pairwise votes (Section 1, contributions) and provides methodological details enabling replication (Sections 4–5; Appendices).

These go beyond incremental tweaks: the combination of an open platform, principled statistics, and active sampling is a new capability for community-driven LLM evaluation.

## 5. Experimental Analysis
- Evaluation methodology
  - Data: 243,329 conversations/ votes; 50 models; ~149 languages (Table 1).
  - Metrics:
    - Pairwise win rates per model pair (Figure 2 left).
    - Bradley–Terry coefficients with uncertainty intervals (Figure 5).
    - Topic coverage via clustering similarity matrices (Figure 3; Appendix Figures 11–12).
    - Agreement metrics between crowd, experts, and a judge model (Tables 3–4).
  - Experimental setups:
    - Replay analysis on 213,576 historical votes to compute BT intervals (Section 7.1; Figure 5).
    - Synthetic simulations test interval coverage and width as sample size and number of models vary (Figure 6; Appendix Figure 14).
    - Active vs random sampling efficiency on a holdout BT parameterization (Figure 7).
    - Human-vs-LLM-as-judge agreement studies with expert re-labeling (Section 6.3; Tables 3–4).
    - Anomaly detection evaluation with hand-labeled users (Section 7.2; Table 5).

- Main quantitative results
  - Diversity: 600 prompt clusters; the largest ~1% of data; low inter-cluster similarity, evidencing a long tail (Section 6.1; Figure 3).
  - Discriminative power across topics (Section 6.2; Table 2):
    - GPT‑4 vs Llama‑2‑70B: 96.7% win in Python game programming; 66.7% in poetry; 53.3% in movie recommendations.
  - Crowd vs expert agreement (Section 6.3; Table 3):
    - GPT‑4‑Turbo vs Llama‑2‑13B: crowd agrees with Expert 1 at 72.8% and Expert 2 at 77.8%; experts agree 89.8%.
    - GPT‑4‑Turbo vs GPT‑3.5‑Turbo‑0613: crowd agrees with Expert 1 at 73.8% and Expert 2 at 83.1%; experts agree 79.4%.
  - Consistency of win rates (Table 4):
    - GPT‑4‑Turbo win rate vs Llama‑2‑13B: crowd 81.2%, Expert 1 89.4%, Expert 2 86.9%, GPT‑4 judge 78.8%.
    - GPT‑4‑Turbo vs GPT‑3.5‑Turbo‑0613: crowd 76.3%, Expert 1 82.5%, Expert 2 89.4%, GPT‑4 judge 79.4%.
  - Uncertainty intervals and multiplicity (Figure 5):
    - Shows BT coefficient intervals for many models; multiplicity correction widens intervals but is required for joint validity.
  - Interval coverage and width (Figure 6):
    - In simulation, uncorrected interval coverage hovers near the nominal level regardless of number of models; more models give wider intervals.
  - Sample efficiency of active sampling (Figure 7):
    - To estimate `θ*` with average width 0.2: random needs 6,800 samples vs adaptive 4,400 (54% more for random).
    - For BT score average width 0.3: random 17,200 vs adaptive 16,400 (5% more for random).
  - Anomaly detection (Table 5):
    - At α=0.1: 13/14 true positives, 24/36 true negatives; at α=0.3: 21/29 true positives, 17/21 true negatives.

- Do the experiments support the claims?
  - Credibility of crowdsourced votes is supported by substantial agreement with experts (Table 3) and consistent win-rate patterns (Table 4).
  - Statistical validity is supported by CLT-based intervals, simulation coverage (Figure 6), and replay analysis (Figure 5; Appendix Figure 13).
  - Efficiency gains from active sampling are clear and practically meaningful for the win matrix, with smaller but consistent gains for BT scores (Figure 7).
  - Topic modeling and cluster-level performance differences evidence that Arena collects varied, discriminative prompts (Figure 3; Table 2).
  - Robustness checks include bootstrap vs sandwich comparisons (Appendix A), multiplicity corrections (Figure 5), and anomaly detection trials (Table 5).

- Mixed/conditional findings
  - LLM-as-judge vs human votes align broadly but not perfectly (Table 4); this is expected for subjective tasks.
  - Active sampling shows strong gains for estimating pairwise win rates; gains for BT scores are smaller (Figure 7), reflecting aggregation effects.

## 6. Limitations and Trade-offs
- Assumptions and modeling choices
  - BT logistic form may be misspecified for real human preferences; the approach relies on robust (“sandwich”) errors and also introduces a nonparametric BT alternative in Appendix B that remains valid under non-transitivity or graded/tie feedback.
  - Asymptotic normality (Eq. 6) assumes regularity and that each pair keeps non-negligible sampling probability.

- Data and user-base biases
  - The user population tends to be hobbyists/researchers (Section 8, “Limitations”), which may skew prompt distribution and preference criteria relative to enterprise or specialized domains.

- Scope
  - The platform focuses on helpfulness and preference, not safety evaluation; moderation flags only 3% of content and does not constitute a comprehensive safety audit (Section 8).

- Practical constraints
  - Non-uniform sampling complicates naïve aggregation; the estimator corrects with inverse weights (Eq. 4, 7), but requires careful logging of sampling probabilities.
  - Multiplicity-corrected intervals widen uncertainty ranges (Figure 5), making rankings more conservative—useful for validity but less decisive.

- Anomaly detection
  - The sequential p-value approach is heuristic and evaluated on a small labeled set; power and false discovery properties in adversarial settings remain open (Section 8).

## 7. Implications and Future Directions
- How this changes the landscape
  - It establishes a credible, open, continuously updating human-preference leaderboard that complements or replaces static, ground-truth benchmarks for open-ended tasks (Figure 1).
  - It normalizes reporting rankings with uncertainty, discouraging overclaiming small differences between models.

- Follow-up research enabled/suggested
  - Topic leaderboards: per-domain rankings using the established clustering pipeline (Section 8, “Future Directions”).
  - Expanded modalities and tasks: multimodal inputs and agentic evaluations in “gamified settings” (Section 8).
  - Stronger sequential testing: replacing heuristic anomaly detection with anytime-valid E-value methods to handle dependence and adaptivity (Section 8 references to Howard et al., Vovk & Wang).
  - Nonparametric scoring: leveraging the Appendix B nonparametric BT score for ties/graded feedback and non-transitive preferences at scale.

- Practical applications
  - Model selection by developers and enterprises using live human preference signals with uncertainty bounds.
  - RLHF and fine-tuning: training data of high-quality, real-world prompts and pairwise preferences (>100K pairwise votes to be released; Section 1).
  - Rapid A/B testing and monitoring: the active sampling rule focuses effort where it matters most, accelerating comparisons of new model variants.

> Representative, cited highlights
> - Data scale and diversity: “~240K votes from ~90K users… >100 languages” (Section 3.2; Table 1).
> - Agreement with experts: “72–83% crowd–expert agreement” (Table 3); GPT‑4‑Turbo win rates consistent across labelers (Table 4).
> - Active sampling gains: “54% fewer samples for same win-matrix precision target; 5% fewer for BT score” (Figure 7).
> - Prompt diversity: “600 clusters; largest ~1%” with low inter-cluster similarity (Section 6.1; Figure 3).

Overall, Chatbot Arena operationalizes a community-scale, statistically rigorous approach to measuring what matters for LLMs: how well real users prefer their responses on fresh, open-ended tasks—and it does so with transparent uncertainty, adaptive efficiency, and evidence of vote quality.
