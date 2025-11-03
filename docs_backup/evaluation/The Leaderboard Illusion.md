# The Leaderboard Illusion

**ArXiv:** [2504.20879](https://arxiv.org/abs/2504.20879)
**Authors:** Shivalika Singh, Yiyang Nan, Alex Wang, Daniel D'Souza, Sayash Kapoor, Ahmet Üstün, Sanmi Koyejo, Yuntian Deng, Shayne Longpre, Noah Smith, Beyza Ermis, Marzieh Fadaee, Sara Hooker
**Institutions:** Cohere Labs, Princeton University, MIT, Stanford University, Various (as per author affiliations)

## 🎯 Pitch

This paper exposes critical flaws in the Chatbot Arena leaderboard, revealing how selective score retraction, unequal sampling, and opaque deprecations distort model rankings, promoting gameable strategies over true quality. By auditing these practices, the study highlights the need for reforms to ensure leaderboards accurately reflect model performance, crucial for guiding industry decisions, research priorities, and public perception of AI capabilities.

---

## 1. Executive Summary (2-3 sentences)
This paper audits the most influential live LLM leaderboard, Chatbot Arena, and shows that its current practices systematically distort rankings. Using 2M battles across 243 models, targeted scraping, simulations, and real-world tests, it demonstrates how undisclosed private testing with selective score retraction, unequal sampling, and opaque deprecations create data-access asymmetries, inflate a few providers’ scores, and promote overfitting to Arena-specific behaviors rather than general model quality.

## 2. Context and Motivation
- Problem addressed
  - Chatbot Arena has become the de facto public leaderboard for comparing large language models (LLMs) via head-to-head human votes. The paper examines whether its rankings reliably reflect model quality, or whether the evaluation process has drifted into Goodhart’s Law territory—when a metric becomes the target, it ceases to be a good measure (Section 1; Figure 1).
- Why it matters
  - The Arena influences media narratives, industry decisions, and academic research directions (Section 1). Misleading rankings can shape funding, research priorities, and public perception, and can incentivize gaming rather than true progress.
- Prior approaches and their gaps
  - Static benchmarks (e.g., GLUE, MMLU) are known to be vulnerable to contamination and overfitting as datasets become public and reused (Section 1; related work in Section 8). Live human-voting platforms like Chatbot Arena were intended to mitigate these risks by being dynamic, but their integrity depends on sound processes for sampling, transparency, and handling of private tests and deprecations.
- How this paper positions itself
  - It offers a comprehensive audit of Chatbot Arena’s processes and outcomes, combining:
    - Multi-source data (Table 1): public and private battle histories, scraped random samples, provider API logs, and leaderboard snapshots.
    - Formal analysis of the Arena’s scoring model (Bradley–Terry) and conditions under which its assumptions are broken (Sections 3, 5; Appendices B, C).
    - Real-world interventions (launching private model variants) to quantify practical impacts (Section 3.3; Figure 9).
    - Prescriptive recommendations to restore trust (Section 6; “Critical Recommendations” on p. 7).

## 3. Technical Approach
Definitions (selective, paper-specific):
- `Chatbot Arena`: A live platform where users submit prompts and vote on which of two anonymous model responses they prefer. Votes feed a `Bradley–Terry` (BT) model to produce an `Arena Score` per model (Section 2.1).
- `Arena Score`: A normalized score derived from the BT model, which estimates each model’s “skill” from pairwise comparisons (Section 2.1; Appendix B).
- `Private variant/testing`: Provider-only, anonymous model submissions evaluated on the Arena before public release. Historically allowed to be retracted (not published), enabling selective disclosure (Sections 3.1–3.3; Figure 6).
- `Best-of-N`: Submitting N private variants, then publicly releasing only the highest-scoring one (Sections 3.2–3.3; Figure 7).
- `Sampling rate`: The fraction of daily Arena battles in which a model appears (Section 4.1; Figure 5).
- `Deprecation`: Removing or nearly removing a model from active sampling. `Silent deprecation` refers to cutting a model’s sampling to near zero without listing it as deprecated (Section 5; Figures 18–19).
- `ArenaHard`: A curated test set derived from the Arena distribution, with high correlation to Arena rankings (Section 4.2; Li et al., 2024b/c).

Step-by-step methodology
1) Data assembly (Table 1; Appendix D)
   - `Historical-battles`: ~2M battles from April 2023–Jan 2025, combining public releases and 43,729 provider-side battles with full conversations (Appendix D.1).
   - `Leaderboard-stats`: 14.3K leaderboard records covering 243 models and 42 providers (Jan 2024–Apr 2025).
   - `Scraped-random-sample`: 5,864 live battles (Jan–Mar 2025) collected via a de-anonymizing prompt to identify private variants (Appendix E.1–E.4).
   - `API prompts`: 197,217 single-turn prompts sent to the authors’ own models via the Arena (Nov 2024–Apr 2025) to quantify prompt duplication (Section 4.2; Figure 12; Appendix H).
   - Models grouped into `Proprietary`, `Open-weights`, and `Open-source` licenses (Appendix F, Table 6).

2) Detecting private testing and its effects
   - Identify private variants by scraping live battles and asking “Who are you?”; the Arena discards such votes, so scraping does not affect rankings (Appendix E.1). Map aliases to providers via self-identification (Appendix E.4; Table 4).
   - Quantify private variant counts per provider (Figure 6; Section 3.1), including additional private vision-variant counts (Appendix E.2).
   - Model the statistical effect of `best-of-N` selection:
     - Show formally that selecting the maximum of N noisy performance estimates yields an upward bias (Appendix C, Eq. (1); order-statistics argument).
     - Simulate how increasing N lifts the expected top score even when variants are comparable (Figure 7; Appendix I). Also show a weaker model family can outrank a stronger one by exploiting best-of-N (Figure 8; Section 3.2).
   - Real-world test: submit controlled private variants
     - Two identical checkpoints of `Aya-Vision-8B` produce meaningfully different Arena Scores (1069 vs 1052; Figure 9 left).
     - Two high-quality but different `Aya-Vision-32B` variants yield a 38-point spread (1097 vs 1059; Figure 9 right).

3) Quantifying sampling and data-access asymmetries
   - Compute maximum daily sampling rates per provider using the scraped sample (Appendix E.5; Figure 5). Observe up to 34% for OpenAI/Google vs as low as 3.3% for Reka.
   - Estimate provider-level access to Arena data using share of all battles (Figures 3–4; Section 4.1). For example, OpenAI and Google together account for roughly 39.6% of all Arena prompts.

4) Measuring overfitting potential to Arena
   - Show measurable temporal drift in prompt languages (Figure 11; Section 4.2) and high within- and cross-month prompt duplication (Figure 12; Appendix H).
   - Fine-tune a 7B model with varying amounts of Arena-style data under a fixed budget (1.3K steps, batch 128) and test on `ArenaHard` using an LLM judge (`gpt-4o-2024-11-20`) (Section 4.2; Figure 10).
   - Check out-of-distribution generalization on `MMLU` (Table 9).

5) Testing the impact of deprecation on ranking reliability
   - Count `silent deprecations` by inspecting activity between Mar 3–Apr 23, 2025. If a “public” model has ≤10 battles on average, it is functionally inactive (Figure 18).
   - Compare deprecation rates by license class (Figure 13) and split official vs silent (Figure 19).
   - Analyze how deprecations can violate core Bradley–Terry assumptions:
     - `Transitivity under stable conditions`: violated when conditions shift but some models stop being evaluated (Section 5.1; Figure 14).
     - `Connected comparison graph`: violated when heavy deprecations and skewed sampling fragment the battle graph (Section 5.2; Figure 15; discussion referencing Ford Jr 1957).

6) Recommendations (Section 6; “Critical Recommendations” on p. 7)
   - Prohibit score retraction, cap private variants per provider, stratify deprecations, implement active sampling (Chiang et al., 2024 eq. in Section 6), and publish full transparency logs.

Why these design choices
- Using both simulation and real-world tests triangulates causal mechanisms (selection bias, sampling skew, deprecations) rather than relying on correlations.
- Focusing on BT assumptions ties each observed practice (retraction, sampling, deprecations) to a principled reason rankings become unreliable.

## 4. Key Insights and Innovations
1) Private best-of-N testing with retraction inflates Arena Scores
   - What’s new: The paper uncovers a de facto policy allowing some providers to test many private variants and retract poor scores, creating selective disclosure.
   - Why it’s significant:
     - Mathematically inevitable bias: Selecting the maximum of N noisy estimates makes the expected published score strictly larger than a single unbiased estimate (Appendix C, Eq. (1)).
     - Simulated magnitude: Testing just 20 variants can raise the expected “best” score by ~50 Arena points (Figure 7; Appendix I). Figure 8 shows a weaker “Family A” surpassing a stronger “Family B” by exploiting this.
     - Real-world confirmation: Identical checkpoints still yield a 17-point spread (1069 vs 1052), and two plausible “final candidates” differ by 38 points (Figure 9). This shows that even small N can confer material leaderboard gains.

2) Data-access asymmetries driven by sampling, private testing volume, and deprecations
   - What’s new: A quantitative breakdown of who gets how much Arena traffic and feedback.
   - Why it’s significant:
     - Sampling favors a few providers: Maximum daily sampling rates reach 34% for OpenAI/Google but ~3.3% for some others (Figure 5; Appendix E.5).
     - Concentrated data: Proprietary models receive 54–70% of monthly battles (Figure 3). Cumulatively, OpenAI and Google alone are estimated to receive ~20.4% and ~19.2% of all Arena prompts (Figure 4; Section 4.1).
     - Open models removed more: 87.8% of open-weights and 89.0% of open-source models are deprecated or inactive, vs 80.0% for proprietary (Figure 13), with the vast majority of open models silently deprecated (Figure 19).

3) Arena data strongly boosts Arena performance but not generalization
   - What’s new: A controlled fine-tuning study isolates the effect of Arena-style data within a fixed training budget.
   - Why it’s significant:
     - On-distribution gains: Increasing Arena data from 0% → 30% → 70% raises win-rate on `ArenaHard` to 42.7% and 49.9% vs `Llama-3.1-8B`, from 23.5% at 0% (Figure 10). Relative gain up to 112% (Section 4.2).
     - Off-distribution stagnation/decline: `MMLU` accuracy does not improve (66.5% → 64.4% → 65.9%; Table 9), indicating specialization to Arena rather than broad capability gains.

4) Deprecations under shifting distributions break Bradley–Terry reliability
   - What’s new: Two targeted simulations connecting Arena practices to BT model assumptions.
   - Why it’s significant:
     - Changing tasks without continuous evaluation leads to rank flips when a model is removed midstream (Figure 14; Section 5.1).
     - Sparse/disconnected comparison graphs produce incorrect global rankings (Figure 15; Section 5.2), invalidating maximum-likelihood estimation conditions (Ford Jr, 1957).

Overall, the novelty is not a new algorithm, but a systems-level, mechanism-centric audit that quantifies and experimentally verifies how specific policies and practices systematically bias a live leaderboard.

## 5. Experimental Analysis
Evaluation methodology
- Datasets and coverage
  - 2M+ battles across 243 public models from 42 providers (Table 1; Sections 2, D). Additional 5.8K live battles scraped (Appendix E), and 197K API prompts to quantify duplication (Figure 12; Appendix H).
- Metrics
  - `Arena Score` from Bradley–Terry (Section 2.1; Appendix B).
  - Win-rates on `ArenaHard` using LM-as-a-judge (Section 4.2).
  - Sampling rate (max daily share of battles; Figure 5).
  - Deprecation activity (≤10 battles during Mar 3–Apr 23, 2025 interpreted as silent deprecation; Figures 18–19).

Main quantitative results and where to find them
- Prevalence of private testing
  - Quote: “Meta…27 private models…Google 10…Amazon 7” (Figure 6; Section 3.1). Additional private vision variants raise Meta’s total to 43 (Appendix E.2, Table 2).
- Best-of-N inflation
  - Theoretical: E[max] > E[single] (Appendix C, Eq. (1)).
  - Simulated: ~+50 Arena points at N=20 (Figure 7; Appendix I).
  - Real-world: identical 8B checkpoints differ by 17 points (1069 vs 1052), and two 32B variants differ by 38 points (Figure 9).
- Sampling disparities and data concentration
  - Max daily sampling up to 34% for OpenAI/Google vs 3.3% for Reka (Figure 5; Appendix E.5).
  - Proprietary models consistently receive the majority of monthly battles (54.3%–70.1%; Figure 3).
  - Cumulative data share estimates: OpenAI 20.4%, Google 19.2% (Figure 4; Section 4.1).
- Overfitting potential
  - Temporal drift in languages (English drops from ~80% to ~50%; Figure 11), but high prompt duplication within and across months (Figure 12; Appendix H).
  - Fine-tuning with Arena data: win-rates on `ArenaHard` climb from 23.5% (0%) → 42.7% (30%) → 49.9% (70%) vs `Llama-3.1-8B` (Figure 10), while `MMLU` does not improve (Table 9).
- Deprecations and ranking reliability
  - Silent deprecations: 205 of 243 public models inactive vs only 47 officially deprecated (Figure 18; Section K).
  - Open models are more affected: 86.6% (open-weights) and 87.8% (open-source) silently deprecated (Figure 19).
  - BT reliability failures:
    - Rank shifts when a model is deprecated mid-distribution change (Figure 14).
    - Disconnected graphs distort rankings (Figure 15; Eq. (2) shows the logistic match outcome model used in the simulation).

Ablations, robustness, and credibility
- Private testing: both simulation and field experiments (identical and non-identical checkpoints) support the bias mechanism (Figures 7–9).
- Overfitting: cross-checked by showing on-distribution improvements (ArenaHard) but not on MMLU (Table 9). This triangulates that gains are highly specific to Arena.
- Graph connectivity and distribution shift: separate simulations for each failure mode (Figures 14–15).

Assessment
- The evidence convincingly ties observed practices to specific statistical failure modes:
  - Best-of-N selection bias is mathematically guaranteed (Appendix C) and visible in practice (Figure 9).
  - Sampling skew and deprecations systematically alter the comparison graph and training incentives (Figures 3–5, 13, 18–19).
  - Arena data predictably boosts Arena-style performance (Figure 10), not necessarily general capabilities (Table 9).
- The main caveat is data access: some analyses rely on scraped samples over a limited time window (Appendix E; Limitations in Section 7 below), which the paper openly discusses.

## 6. Limitations and Trade-offs
Assumptions and scope
- Restricted data visibility
  - The authors do not have the Arena’s raw, full battle logs. Public releases are preprocessed (deduplicated; anti-bot filtering), and provider-side data only covers their own models. Private-model battles are removed from public datasets (Section 7; Appendix D.1).
- Scraping window and volume
  - The private-variant counts (Figure 6) come from a 5.8K-battle scrape (Jan–Mar 2025). This may undercount providers that launched fewer models in that window (Section 7; Appendix E).
- Provider attribution of private variants
  - Based on model self-identification (Appendix E.4), which can be noisy or inconsistent. The paper lists identities and codenames to enable external checking.
- Overfitting estimates likely conservative
  - Training used only a fraction of Arena-style data accessible to large providers, with no hyperparameter sweeps; true on-Arena gains could be larger (Section 7).
- Not a full adversarial security audit
  - The study does not quantify bot voting or coordinated brigading, though it references recent work showing vulnerabilities (Section 7; Related Work Section 8.2).

Trade-offs and open questions
- Some operational choices (e.g., letting providers test privately) have valid motivations (confidential launches), but need strict, transparent limits to avoid gaming.
- Using an LLM-as-judge for ArenaHard approximates human preferences but is not a perfect proxy; however, it is a standard and practical method used in prior work (Section 4.2).
- The exact impact of sampling policy changes over time cannot be fully audited without complete internal logs.

## 7. Implications and Future Directions
How this work changes the landscape
- It reframes public LLM leaderboards as complex socio-technical systems whose rules materially shape measured “progress.” The study shows concrete, fixable mechanisms by which rankings become unreliable: best-of-N with retraction, skewed sampling, and opaque deprecations under a shifting task distribution.
- By tying these to the Bradley–Terry model’s assumptions, it moves the debate from opinions to testable, model-based reliability criteria (Sections 3, 5; Appendix B).

Actionable reforms (Section 6; “Critical Recommendations,” p. 7)
- Prohibit score retraction: publish all tested results (including private variants).
- Cap concurrent private variants per provider (e.g., at 3) and disclose counts.
- Stratified, auditable deprecation (e.g., retire bottom 30th percentile within each of Proprietary/Open-weights/Open-source once rankings converge).
- Implement active, uncertainty-reducing sampling (Chiang et al., 2024; Section 6 formula for `Pt(a)`), instead of favoring large proprietary providers.
- Full transparency: quarterly logs of tested models, sampling rates, and all deprecations (including “silent” ones).

Research directions enabled
- Robust live-eval design: adaptive sampling that minimizes uncertainty and adversarial leverage, with audit-friendly metadata.
- Anti-gaming analytics: online detectors for best-of-N patterns, selective disclosure, and vote manipulation.
- Generalization-sensitive metrics: combine on-distribution Arena-style performance with held-out distributions (e.g., category shifts, language shifts) to discourage narrow overfitting.
- Data governance: fair access protocols for community-generated prompts and outcomes, balancing privacy, competition, and scientific reproducibility.

Practical applications
- Benchmark governance bodies (academic and industry) can adopt the paper’s auditing templates—private-variant tracking, sampling-rate dashboards, connectivity diagnostics—to maintain trustworthy leaderboards.
- Model developers can calibrate expectations: strong Arena gains from small amounts of in-distribution data (Figure 10) may not transfer to broader capabilities (Table 9), informing product claims and evaluation planning.

> Representative findings to remember
> - “Testing just 20 private variants can raise the expected top score by ~50 Arena points” (Figure 7; Appendix I).
> - “OpenAI and Google together account for ~39.6% of Arena data; proprietary models receive 54–70% of monthly battles” (Figures 3–4).
> - “Identical checkpoints can land 17 Arena points apart; two plausible final variants can differ by 38 points” (Figure 9).
> - “205 of 243 public models are silently deprecated; open(-weights/source) models are disproportionately affected” (Figures 18–19).
> - “70% Arena-style data more than doubles ArenaHard win-rate vs Llama-3.1-8B (23.5% → 49.9%) but does not improve MMLU” (Figure 10; Table 9).

In sum, this paper shows, with mechanisms and measurements, how a few fixable policies can nudge an influential leaderboard from a measure of broad progress to a target of narrow optimization—and how to reverse course.
