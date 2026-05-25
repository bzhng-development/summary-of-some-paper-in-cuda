# Chatbot Arena: An Open Platform for Evaluating LLMs by Human Preference

**ArXiv:** [2403.04132](https://arxiv.org/abs/2403.04132)

## 🎯 Pitch

Chatbot Arena introduces the first large-scale, open, crowdsourced platform for evaluating large language models based on live human preference through anonymous model 'battles' and robust, statistically principled ranking. By leveraging diverse real-world user inputs and votes, along with adaptive sampling and uncertainty quantification, it provides a trustworthy and dynamic alternative to static, ground-truth benchmarks—providing model developers and the wider community with a transparent leaderboard that better reflects genuine user needs and preferences across languages and domains.

---

## 1. Executive Summary (2-3 sentences)
This work introduces Chatbot Arena, a free, open, live platform that evaluates large language models (LLMs) by human preference through anonymous pairwise “battles.” It contributes (1) a principled, statistically grounded ranking system based on Bradley–Terry (BT) coefficients with uncertainty quantification and active sampling, and (2) an analysis showing that crowdsourced prompts and votes are diverse, discriminative, and broadly consistent with expert judgments (Sections 3–7; Figures 2–7; Tables 1–5).

## 2. Context and Motivation
- Problem addressed
  - Static, ground-truth benchmarks (e.g., MMLU, HellaSwag, GSM-8K) struggle to evaluate open-ended tasks and “alignment with human preferences,” and they suffer from contamination and saturation over time (Figure 1; Related Work: “Risks of Static Benchmarks”).
- Why it matters
  - Modern LLMs are used interactively across diverse tasks and languages; assessing which models people prefer in real use is essential for deployment and further alignment research.
- Prior approaches and gaps
  - Static datasets with ground truth are reproducible and cheap but miss open-ended, subjective aspects and get contaminated (Section 1; Figure 1).
  - Live evaluations exist (e.g., weekly competitions) and preference datasets exist (HH-RLHF, OASST), but an open, large-scale, continuously updated human-preference leaderboard across many LLMs was absent (Sections 2–3).
- Positioning
  - Chatbot Arena is an open, crowdsourced, continuous evaluation platform that collects live prompts and votes, then uses well-established statistical tools to produce reliable model rankings with uncertainty, and validates data quality and sampling strategies (Sections 3–7).

## 3. Technical Approach
Step-by-step pipeline and core concepts:

1) Data collection and interface (Section 3.1–3.2)
- Two anonymous models are sampled for each “battle.” A user enters any prompt; both models answer side-by-side; the user votes for the preferred answer. Options “tie” or “both are bad” exist. Model identities are revealed only after voting (Figure 8).
- No preset prompts: encourages diversity and real-world usage. Moderation flags unsafe content (3% of requests). Identity terms are filtered to preserve anonymity (Section 3.2).
- Scale as of Jan 2024: ~240K votes from ~90K users, >50 models, >100 languages; 77% English and 5% Chinese (Table 1; Section 3.2; Figure 9).

Key terms
- pairwise comparison: the user chooses which of two model responses is better for a given prompt.
- win matrix: for each ordered pair of models `(m, m')`, the probability that `m'` wins over `m` when shown together (Section 4; Figure 2).

2) From pairwise votes to model rankings (Section 4)
- Sequential setup: at time `t`, a pair `A_t = (m, m')` is shown; user feedback `H_t` is observed (1 if `m'` preferred over `m`, 0 otherwise).
- Estimate the win matrix `θ*` by an unbiased estimator (Equation (4)) and compute its covariance (Equation (5)); a central limit theorem (CLT) yields asymptotic normality (Equation (6)).
- Ranking score via Bradley–Terry (BT) model (Equations (2)–(3)):
  - BT assumes a latent “skill” `ξ_m` per model. The probability `m` beats `m'` follows a logistic function: `P(H=1) = 1 / (1 + e^{ξ_{m'} - ξ_m})` (Eq. 2).
  - Estimate `ξ` by minimizing reweighted logistic loss (Eq. 7). Reweighting by `1/P(A_t)` targets a uniform distribution over pairs, avoiding bias from non-uniform sampling.
  - Construct confidence intervals (CIs) for `ξ` using robust “sandwich” standard errors (preferred over pivot bootstrap after simulation; Appendix A; Section 5).

Why BT with robust inference?
- BT yields an interpretable “strength” per model from pairwise data and is well-studied.
- Even when the BT form is misspecified, asymptotic normality holds with robust (“sandwich”) variance (Section 4; Huber/White results).
- The paper also provides a nonparametric BT-style score that remains valid with non-binary feedback or non-transitive preferences (Appendix B), but the main ranking deploys standard BT with robust CIs.

3) Approximate ranking with uncertainty (Section 5)
- Build a joint confidence set `C` for all `ξ` such that `P(s(P) ∈ C) ≥ 1 − α` (Eq. 8), using a chi-square CLT with robust covariance.
- Define an “approximate rank” by comparing the intervals of scores: if the lower bound of `m'` exceeds the upper bound of `m`, then `m'` outranks `m`. This ensures with probability ≥ `1 − α` that no model is understated: “P(∃m : R_m > rank(P)_m) ≤ α” (Section 5).

4) Active sampling to accelerate convergence (Section 5; Eq. 9)
- At each step, choose the next model pair with probability proportional to the reduction in its CI upon one more sample:
  - `P_t(a) ∝ sqrt(Σ̂_{t,a,a}) / (count(a)) − sqrt(Σ̂_{t,a,a}) / (count(a)+1)` (Eq. 9).
- Intuition: spend more battles on pairs where uncertainty is still high (often similarly matched models), improving sample efficiency.

5) Detecting anomalous users (Section 5.1)
- For a user’s sequence of votes, compute a per-vote p-value by comparing their choice to the historical distribution for the same pair (Eq. 10 is a valid p-value under exchangeability; proof in Appendix C).
- Aggregate evidence sequentially using Fisher’s combination with a Bonferroni-like safeguard: flag users when the statistic exceeds a chi-square threshold at a few randomly chosen checkpoints (Section 5.1).

6) Prompt diversity and discriminativeness (Section 6)
- Topic modeling with BERTopic: embed prompts (OpenAI `text-embedding-3-small`), reduce dimensionality via UMAP, cluster with HDBSCAN (min cluster size 32), then label topics using GPT-4-Turbo (Section 6.1).
- 600 clusters discovered; top-16 are small and dissimilar, indicating long-tail diversity (Figure 3; Figures 11–12 for 64-cluster structure).
- Discriminative power: sample 30 prompts per topic and run “LLM-as-judge” comparisons to test if prompts separate strong from weaker models (Section 6.2; Table 2).

7) “Arena Bench”: a benchmark distilled from crowdsourced prompts (Section 6.2; Appendix D.2–D.3)
- Build a curated set from topic clusters with a standardized LLM-as-judge protocol to compare with MT-Bench; shows larger gaps between proprietary and open models (Figure 4). Details include dual judgments to avoid positional bias and specific scoring rules (Appendix D.3).

Design choices and why
- Pairwise votes over absolute ratings: simpler for inconsistent crowds, reduces rubric burden (Section 3.1).
- Anonymous, randomized models: avoids brand bias (Section 3.1).
- Robust CIs and multiplicity correction: rankings reflect uncertainty and avoid overclaiming (Section 5; Figure 5).
- Active sampling: concentrates effort where it improves rankings most (Section 5).

## 4. Key Insights and Innovations
- Live, open, crowdsourced preference evaluation at scale
  - Significance: gathers diverse, non-contaminated, real-user prompts—something static test sets cannot provide.
  - Evidence: ~240K votes from ~90K users; >100 languages; >50 models (Table 1; Section 3.2; Figure 9). This scale and openness are unusual and impactful for the community.

- Statistically principled ranking with uncertainty and simultaneity
  - Novelty: converts pairwise votes into BT coefficients with robust “sandwich” CIs and a simultaneous confidence set used to produce conservative “approximate ranks” (Section 5; Figure 5).
  - Why it matters: a leaderboard that quantifies uncertainty and controls false ordering across many models improves trust and decision-making. Simulation verifies near-nominal coverage (Figure 6).

- Active sampling policy for pair selection
  - Novelty: a simple, CI-reduction–based rule (Eq. 9) that provably focuses data collection where it reduces uncertainty most.
  - Impact: improves sample efficiency in both win-matrix and score estimation; e.g., to reach a win-matrix precision of 0.2, random needs ~6,800 samples versus ~4,400 with adaptive sampling (Figure 7).

- Data quality validation: diversity and expert agreement
  - Prompt diversity: 600 topics; small, low-similarity clusters (Figure 3).
  - Discriminative power: prompts in coding/reasoning clusters separate GPT‑4 from Llama‑2‑70B with very high win rates (e.g., 96.7% in “Python Game Programming Challenge”), while lifestyle/recommendation clusters yield closer results (e.g., 53.3% in “Movie Recommendations & Ratings”)—Table 2.
  - Vote quality: crowd–expert agreement 72–83%, comparable to expert–expert agreement (79–90%)—Table 3; GPT‑4 win rates consistent across judges (Table 4).

- First step toward anomaly detection in crowdsourced preference data
  - Method: per-pair p-values with Fisher’s combination, evaluated at random checkpoints (Section 5.1).
  - Result: achieves “~90% true positive and 60–70% true negative rate” on curated examples (Table 5).

## 5. Experimental Analysis
Evaluation methodology
- Data and setup
  - Historical replay of T = 213,576 votes to compute BT coefficients and CIs (Section 7.1; Figure 5).
  - Simulation studies to assess interval coverage and width under controlled conditions (Figure 6; Appendix A/Figure 14).
  - Active sampling evaluation by simulating from the best-fit BT model and comparing random vs. adaptive policies (Section 7.1; Figure 7).
  - Prompt analyses via BERTopic/UMAP/HDBSCAN; cluster similarity matrices and hierarchies (Figure 3, 11, 12).
  - Vote quality via expert relabeling on randomly selected battles: GPT‑4‑Turbo vs. Llama‑2‑13B and GPT‑3.5‑Turbo‑0613 (Section 6.3; Tables 3–4).

Key quantitative results
- Ranking intervals
  - Figure 5 shows BT coefficient intervals with and without multiplicity correction. The corrected (simultaneous) intervals are wider, but necessary for valid ranking inference across all models at once.
- Interval calibration and width
  - Simulation (Figure 6) shows uncorrected intervals have empirical coverage close to nominal (≈ 1 − α) and widened with more models; average interval width decreases with more samples.
  - Appendix A (Figure 13) compares sandwich vs. bootstrap on replayed data: sandwich intervals are more stable and become smaller in large samples.
- Active sampling improves sample efficiency
  - Win matrix precision 0.2: random ≈ 6,800 vs. adaptive ≈ 4,400 samples (≈54% more data needed for random).
  - BT score precision 0.3: random ≈ 17,200 vs. adaptive ≈ 16,400 samples (≈5% more data needed for random). Across the full horizontal range, adaptive is consistently better (Figure 7).
- Prompt discriminativeness (Table 2)
  - GPT‑4‑0613 beats Llama‑2‑70B on coding-heavy clusters with large margins:
    - “Python Game Programming Challenge”: 96.7% win rate.
    - “C/C++ Process Multi-Threading”: 86.7%.
    - “SQL Query Database Assistance”: 73.3%.
  - On lighter reasoning or preference tasks:
    - “Poetry Writing Prompts”: 66.7%.
    - “Movie Recommendations & Ratings”: 53.3%.
- Crowd vs. expert agreement and model win rates (Tables 3–4)
  - Agreement:
    - GPT‑4‑Turbo vs. Llama‑2‑13B: Crowd with Expert 1 = 72.8%, with Expert 2 = 77.8%; Expert 1 with Expert 2 = 89.8%.
    - GPT‑4‑Turbo vs. GPT‑3.5‑Turbo‑0613: Crowd with Expert 1 = 73.8%, with Expert 2 = 83.1%; Expert 1 with Expert 2 = 79.4%.
  - GPT‑4‑Turbo win rates:
    - vs. Llama‑2‑13B: Crowd 81.2%, Expert 1 89.4%, Expert 2 86.9%, GPT‑4-as-judge 78.8%.
    - vs. GPT‑3.5‑Turbo‑0613: Crowd 76.3%, Expert 1 82.5%, Expert 2 89.4%, GPT‑4-as-judge 79.4%.
- Arena Bench vs. MT-Bench (Figure 4)
  - Using GPT‑4 as judge, Arena Bench produces a larger separation between top proprietary and strong open models than MT‑Bench, indicating that curated prompts drawn from live usage can reveal performance gaps more clearly.

Do the experiments support the claims?
- The statistical pieces (interval coverage/width; simulation plus replay) support the reliability of the ranking method (Figures 5–6; Appendix A).
- The adaptive sampling evaluation demonstrates consistent efficiency gains (Figure 7).
- Data quality claims are supported by multi-angle evidence: prompt diversity (Figure 3), discriminativeness (Table 2; Appendix D.1 examples), and vote agreement with experts (Tables 3–4).
- Anomaly detection is preliminary but shows promising accuracy on curated cases (Table 5).

Robustness, ablations, and caveats
- Bootstrap vs. sandwich: experiments in Appendix A show both work; sandwich is deployed due to stability and smaller intervals at larger n (Figure 13–14).
- Multiplicity correction increases CI width (Figure 5); the paper reports both corrected and uncorrected.
- LLM-as-judge is used in specific analyses (e.g., Table 2; Figure 4) to factor out crowd noise; this assumes judge reliability, which is not the main focus here but is a known limitation of such evaluations.

## 6. Limitations and Trade-offs
- User/population bias
  - The platform likely attracts LLM hobbyists and researchers, not a representative cross-section of all end users (Section 8).
- Task/source bias
  - Data comes from the chat interface; production or domain-specific enterprise use may be underrepresented (Section 8).
- Safety not evaluated
  - The focus is helpfulness/preference; safety assessments are out of scope (Section 8). Only basic moderation (3% flagged) is mentioned (Section 3.2).
- Statistical assumptions and practical trade-offs
  - Rankings rely on BT modeling with robust variance; if true preferences are highly non-transitive or multi-modal, score interpretations can be subtle (Appendix B gives a nonparametric alternative).
  - Multiplicity-corrected CIs can be conservative, potentially delaying decisive rankings (Figure 5).
- Sampling and coverage
  - Non-uniform pair sampling (on purpose) means raw win rates depend on pair availability; reweighting and inference address this, but some pairs may have fewer direct observations (Figure 2 right).
- Anomaly detection is preliminary
  - The detection approach uses exchangeability assumptions and heuristic checkpoints; formal anytime-valid guarantees via E-values are proposed as future work (Section 8).

## 7. Implications and Future Directions
- How this changes the landscape
  - Establishes a widely-referenced, open, preference-based leaderboard with rigorous statistics—bridging the gap between static, ground-truth evaluations and real-world, open-ended usage (Sections 1, 3, 7). It encourages model developers to optimize for human preference in natural interactions.
- Follow-up research enabled
  - Topic-specific leaderboards and skill diagnostics across languages and domains (suggested in Section 8).
  - Deeper statistical tooling: anytime-valid inference and stronger outlier detection using E-values and nonnegative supermartingales (Section 8).
  - Methods to mitigate selection biases and to calibrate LLM-as-judge pipelines against expert panels.
  - Exploration of the nonparametric BT formulation (Appendix B) for settings with ties, graded preferences, or non-transitivity.
- Practical applications
  - Model selection and A/B testing for organizations deploying LLMs, with uncertainty-aware comparisons.
  - Training data curation: prompts and preferences feed back into RLHF or preference optimization.
  - Benchmark construction: Arena Bench demonstrates how to distill challenging, contemporary prompts from live usage (Section 6.2; Appendix D.2–D.3).
  - Monitoring and governance: anomaly detection offers a foundation for maintaining data integrity at scale (Section 5.1; Table 5).

> Overall, the paper provides an end-to-end, open ecosystem—from data collection (Section 3), to statistically principled ranking (Sections 4–5), to empirical validation (Sections 6–7)—demonstrating that large-scale crowdsourced preference evaluation can be both informative and scientifically rigorous.
