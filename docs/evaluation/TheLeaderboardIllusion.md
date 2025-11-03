# The Leaderboard Illusion

**ArXiv:** [2504.20879](https://arxiv.org/abs/2504.20879)

## 🎯 Pitch

This paper delivers a rigorous, data-driven audit of Chatbot Arena—the leading human-preference leaderboard for large language models—revealing how hidden private variant testing, selective score reporting, unequal data access, and opaque model deprecation systematically bias leaderboard rankings. By combining analysis of 2 million model comparisons with real experiments and simulations, the authors show that these practices distort ratings, entrench advantages for a handful of providers, and promote overfitting to leaderboard quirks rather than genuine model improvement. Their findings underscore the urgent need for transparent and fair evaluation policies to maintain scientific integrity and trust in the benchmarks that shape AI research and industry direction.

---

## 1. Executive Summary (2-3 sentences)
This paper audits the widely used Chatbot Arena leaderboard and shows that three systemic mechanisms—private “best-of-N” testing with selective score disclosure, unequal exposure to user data, and opaque model deprecations—distort rankings away from true model quality. Using 2M battles plus targeted scraping, simulations, and real deployments, it demonstrates how these practices bias Bradley–Terry ratings, concentrate valuable evaluation data among a few providers, and enable overfitting to Arena-specific behavior, then proposes concrete policy fixes.

## 2. Context and Motivation
- Problem addressed
  - How reliable is Chatbot Arena as a live human-preference leaderboard for large language models (LLMs)? The paper investigates whether policies and platform dynamics bias rankings in ways unrelated to genuine model capability.
- Why it matters
  - Chatbot Arena has become a de facto public benchmark used by media, industry, and researchers to compare frontier systems (Section 1). Leaderboards shape investment, research direction, and deployment decisions. If the metric is gamed (Goodhart’s Law), resources can be misallocated and scientific progress misread.
- What existed before and where it falls short
  - Static task leaderboards (e.g., MMLU, GLUE) face contamination and poor alignment with real use (Section 1). Chatbot Arena’s live, open-ended comparisons were meant to remedy this. But the paper shows new failure modes specific to live leaderboards (private pre-release testing, uneven sampling, silent deprecations).
- Positioning
  - The work is an empirical audit plus theory-and-simulation analysis:
    - Audits data access, sampling, and (silent) deprecations across 243 models/42 providers (Table 1; Figures 2–6, 18–19).
    - Theorizes and simulates how private “best-of-N” testing violates Bradley–Terry assumptions and inflates ratings (Section 3.2; Appendix C; Figure 7).
    - Runs real-world Arena deployments to measure the advantage from multiple variants (Section 3.3; Figure 9).
    - Demonstrates how access to Arena-style data boosts in-distribution performance (Figure 10) but not out-of-distribution (Table 9).
    - Shows how deprecations under a shifting task mix and sparse comparison graphs undermine Bradley–Terry reliability (Section 5; Figures 14–15).

## 3. Technical Approach
The study follows a transparent multi-pronged methodology designed to isolate specific mechanisms that can distort a live ranking system.

- Data and instrumentation (Table 1; Appendix D)
  - `leaderboard-stats` (public): Versioned snapshots of Arena ratings, model counts, and battles from Jan 2024–Apr 2025 (14.3K rows across 243 models).
  - `historical-battles` (public+provider-share): ~2M battles (mostly without prompt text) from Apr 2023–Jan 2025; includes 43,729 proprietary “battles involving Cohere models,” with conversations, provided under Arena’s 20% data-sharing policy (Appendix D.1).
  - `scraped-random-sample`: 5,864 contemporary battles scraped (Jan–Mar 2025) to see private testing and sampling rates that are not visible in the public exports (Section 3.1; Appendix E). To avoid influencing scores, scrapes ask identity prompts that invalidate the votes (Appendix E.1).
  - `API prompts`: 197,217 single-turn conversations collected via Cohere’s API (Nov 2024–Apr 2025) to study prompt duplication and drift (Section 4.2; Figure 12; Appendix H).

- Key concepts (defined as used in the paper)
  - `battle`: a pairwise comparison where a user submits a prompt and votes for one of two anonymous model responses (or tie).
  - `sampling rate`: percent of daily battles involving a given model (Section 4.1). High sampling increases both visibility and access to user prompts.
  - `deprecation`: ceasing to sample a model (either officially listed or “silent,” where sampling rate is reduced to near zero without public labeling; Section 4.1 and Section 5).
  - `private variant`: a model tested anonymously on the Arena before public release; results can be kept private or retracted (Section 3.1).

- How rankings are computed and why that matters
  - Arena uses a Bradley–Terry (BT) model to estimate latent “skill” from pairwise win/loss outcomes. Probability model: the chance that model `i` beats model `j` is θ_i / (θ_i + θ_j). Ratings are a scaled form of the BT log-odds parameter (Appendix B). BT relies on key assumptions:
    1) unbiased sampling of comparisons,
    2) transitive inference over a connected comparison graph,
    3) stable evaluation conditions (Section 2.1; Section 5).
  - The paper shows how private “best-of-N” selection, sampling skew, and deprecations violate these assumptions.

- Analyses and experiments
  1) Private testing audit: Identify private variants by de-anonymization prompts (Appendix E.1) and count per provider (Figure 6; Table 2).  
  2) Best-of-N theory and simulation: Prove and simulate the upward bias from picking the max of N noisy estimates (Appendix C; Section 3.2; Figure 7; Appendix I).  
  3) Real Arena deployments: Submit multiple variants of the same model (identical and slightly different checkpoints) to measure observed score spread (Section 3.3; Figure 9).  
  4) Data access asymmetry: Quantify sampling rates (Figure 5; Appendix E.5), total battles by license type (Figure 3), and estimated provider data volume (Figure 4).  
  5) Overfitting risk: Measure within- and cross-month prompt duplication (Figure 12; Appendix H) and language drift (Figure 11). Then fine-tune models on increasing fractions of Arena-style data and evaluate on an in-distribution set (ArenaHard) and an out-of-distribution set (MMLU) (Section 4.2; Figure 10; Table 9).  
  6) Deprecation effects on BT reliability: Simulate a shifting task mix plus mid-way deprecation (Figure 14), and a sparse/disconnected comparison graph (Figure 15), to show how rankings become unreliable (Section 5).

- Why these design choices
  - Scraping was necessary because private battles and sampling weights are not fully disclosed in public datasets (Section 3.1; Appendix E).  
  - Using both theory (order statistics for max-of-N) and live deployments isolates the “selection bias” mechanism from confounders (Section 3.2–3.3).  
  - Fine-tuning on Arena-like data tests whether unequal data access plausibly translates into on-Arena gains (Section 4.2).  
  - Simulation of BT under dynamic conditions reveals assumption failures not visible from static exports alone (Section 5).

## 4. Key Insights and Innovations
1) Documenting undisclosed, asymmetric private testing at scale
- What’s new: A small set of providers privately test many variants concurrently and can retract or withhold scores; e.g., Meta with 27 private variants on the overall leaderboard in March 2025 (43 when including the vision leaderboard), Google with 10, Amazon with 7, Cohere with 4 (Figure 6; Table 2). No academic labs observed running private variants during the scrape period (Section 3.1).
- Why it matters: This enables “best-of-N” cherry-picking on the live distribution, violating BT’s unbiased sampling assumption before public release.

2) Formalizing and empirically validating best-of-N selection bias
- Novelty: The paper proves that choosing the maximum over N noisy performance estimates strictly increases the expected observed score (Appendix C, Theorem 1). Simulations show the expected maximum Arena Score rises with N; testing 20 variants increases the best observed score by ~50 points in one setup (Figure 7; Appendix I).  
- Live confirmation: Two identical `Aya-Vision-8B` checkpoints produced different Arena Scores (1069 vs. 1052) purely from measurement variability; two plausible `Aya-Vision-32B` candidates differed by 38 points (1097 vs. 1059) (Figure 9).  
- Significance: Even identical models can be “lifted” if multiple tries are allowed and the best is disclosed.

3) Quantifying data access inequities and their practical effect
- What’s new: The study ties sampling rate, private testing volume, and deprecations to aggregate data capture by provider. Estimated share of user prompts: OpenAI ~20.4%, Google ~19.2%, Anthropic ~12.2%, Meta ~11.0%; combined open-source models just ~8.9% (Figure 4). Proprietary models receive 54–70% of battles across quarters (Figure 3). Maximum observed daily sampling rates peak at 34% for OpenAI and Google vs. 3.3% for Reka (Figure 5; Appendix E.5).  
- Practical effect: Training with more Arena-style data yields large in-distribution gains: a model fine-tuned with 70% Arena-mix nearly doubles win-rate on ArenaHard versus the 0% Arena-mix variant (23.5% → 49.9%; Figure 10). The same models show minimal/no improvement on MMLU (Table 9).

4) Showing how deprecations under a shifting distribution undermine BT rankings
- What’s new: The paper demonstrates two BT failure modes.
  - Distribution shift + deprecation: When phase 2 prompts differ from phase 1 and one model is deprecated after phase 1, final rankings invert relative to the “no deprecation” scenario (Figure 14).  
  - Graph sparsity: If model removals fragment the comparison graph, rankings deviate from ground truth, and MLE uniqueness can fail (Figure 15; referencing Ford 1957).  
- The trigger: Silent deprecations are widespread—205 of 243 public models have effectively near-zero sampling over Mar 3–Apr 23, 2025, vs. only 47 being officially listed as deprecated (Figure 18). Silent deprecations disproportionately affect open-weight and open-source models (Figure 19).

These go beyond incremental measurement: they expose structural mechanisms that can make a live leaderboard systematically prefer certain providers and encourage overfitting.

## 5. Experimental Analysis
- Evaluation methodology
  - Private testing prevalence: Count private variants by provider via de-anonymizing scraped battles (Section 3.1; Appendix E.1–E.4).  
  - Selection bias:
    - Theory: Order statistics proof that E[max of N] exceeds the mean (Appendix C).  
    - Simulation: Vary number of private variants (0–50), estimate expected best Arena Score (Figure 7; Appendix I).  
    - Real experiments: Submit identical and near-identical checkpoints to Arena and compare their final Arena Scores (Figure 9).  
  - Data asymmetry: Compute sampling rates (Figure 5; Appendix E.5), share of battles by license type across time (Figure 3), and estimated provider-level data access (Figure 4).  
  - Overfitting potential: Measure prompt duplication within and across months (Figure 12; Appendix H) and prompt language drift (Figure 11). Fine-tune with different `arena-mix` proportions, evaluate on ArenaHard (Arena-style, with LLM-as-judge proxy) and MMLU (Section 4.2; Figure 10; Table 9).  
  - Deprecation reliability:
    - Dynamic distribution + deprecation simulation (Figure 14; Appendix L).  
    - Dense vs. disconnected comparison graphs (Figure 15).

- Main quantitative results
  - Private testing counts:
    > Meta (27 overall; 43 including vision), Google (10), Amazon (7); none observed from academic labs during the scrape window (Figure 6; Table 2).
  - Best-of-N inflation:
    > “Testing just 20 variants yields a notable increase of approximately 50 points in the maximum score identified” (Figure 7).  
    > Identical `Aya-Vision-8B` checkpoints: 1069 vs. 1052; Two `Aya-Vision-32B` finalists: 1097 vs. 1059 (Figure 9).
  - Data access inequity:
    > Proprietary models receive 54.3%–70.1% of battles across quarters (Figure 3).  
    > Estimated total data access: OpenAI ~1.24M samples (~20.4%), Google ~1.17M (~19.2%), Anthropic ~741K, Meta ~671K; all open-source models combined ~8.9% (Figure 4).  
    > Max daily sampling rates: OpenAI 34.0%, Google 34.2%, xAI 22.0%, Meta 17.9%; Reka 3.3% (Figure 5; Appendix E.5).
  - Overfitting signal and gains:
    > Within-month duplicates or near-duplicates often 16–33% (Figure 12).  
    > Cross-month exact duplicates: e.g., 7.3% of Dec 2024 prompts reappear exactly in Jan 2025; 9% at high semantic similarity (Appendix H).  
    > Language drift: non-English prompts rise from 23.9% (Apr 2023) to 43.5% (Jan 2025); Russian grows to 15.7% (Dec 2024) (Figure 11).  
    > Arena-mix fine-tuning: win-rate vs. `Llama-3.1-8B-Instruct` rises from 23.5% (0% Arena-mix) to 42.7% (30%) to 49.9% (70%) on ArenaHard (Figure 10). MMLU accuracy slightly decreases (66.5% → 64.4%/65.9%) (Table 9).
  - Deprecations and BT reliability:
    > 205 of 243 public models effectively silent-deprecated during Mar–Apr 2025 vs. 47 officially marked (Figure 18).  
    > Silent deprecation disproportionately hits open-weight (86.6%) and open-source (87.8%) models (Figure 19).  
    > Simulations show ranking reversals under shifting tasks when a model is deprecated midstream (Figure 14), and misrankings when comparison graphs are sparse/disconnected (Figure 15).

- Do the experiments support the claims?
  - The combination of theory (Appendix C), simulation (Figures 7–8, 14–15), and live deployments (Figure 9) convincingly demonstrates that best-of-N selection plus retraction inflates ratings and that deprecations can undermine BT’s statistical guarantees.
  - The sampling and data-access inequity is well supported by scrape-derived sampling rates, public leaderboard statistics, and the provider-level estimates (Figures 3–5).
  - The overfitting claim is appropriately bounded: Arena-specific gains are large (Figure 10), while out-of-distribution gains do not materialize (Table 9). Duplication and drift provide a mechanism (Figures 11–12).

- Ablations, failure cases, robustness
  - The paper transparently reports limitations (Section 7): restricted scraping window (Jan–Mar 2025), reliance on model self-identification for attributing private variants (Appendix E.4), and that proprietary logs/public exports omit private battles.  
  - The fine-tuning experiment uses a fixed small budget and one judge model (gpt-4o) to simulate human preferences; this is a standard but imperfect proxy (Section 4.2).

## 6. Limitations and Trade-offs
- Assumptions and data constraints
  - Private testing inference relies on de-anonymization prompts; some attributions could be noisy (Appendix E.4).  
  - The scrape window may undercount providers that launched fewer variants in Jan–Mar 2025 (Section 7).  
  - Public and provider-shared datasets exclude private battles and apply de-duplication/filters; true underlying vote streams are not available (Section 7).  
  - Estimating provider data access (Figure 4) assumes that “battles ≈ API calls” and counts prompts twice per battle (two models), which is reasonable but approximate.

- Methodological trade-offs
  - The fine-tuning study is deliberately conservative (only three mixes, no hyperparameter sweeps), so reported overfitting gains are likely lower bounds (Section 7).  
  - Using LLM-as-judge for ArenaHard correlates with human preferences but is still a proxy.

- Scope limitations
  - The work does not deeply analyze adversarial or collusive voting (e.g., vote brigading), which other studies suggest can manipulate leaderboards (Section 7; Related Work).  
  - Energy/compute costs and the environmental footprint of additional variant testing are not examined.

- Open questions
  - How large are the benefits if providers also leverage private, non-disclosed Arena-style data (e.g., logs) at full scale?  
  - How sensitive are outcomes to alternative sampling or rating models (e.g., dynamic BT variants, explicit variance-aware rankings)?

## 7. Implications and Future Directions
- How this changes the landscape
  - The paper reframes live leaderboards as socio-technical systems whose design choices (private testing policies, sampling, and deprecations) can materially steer outcomes—sometimes more than model innovations themselves. It argues that today’s Arena rankings partially reflect optimization to the platform’s idiosyncrasies rather than broad model quality.

- Practical recommendations (Section “Critical Recommendations…”; Section 6)
  - Prohibit retraction: publish all private-variant results permanently to remove best-of-N bias.  
  - Cap private variants per provider (e.g., ≤3 concurrently), and disclose counts.  
  - Fair, transparent deprecations: stratify by license type (e.g., retire bottom 30% within proprietary/open-weight/open-source separately) to preserve graph connectivity and reduce provider-type bias.  
  - Implement active, variance-aware sampling (as proposed by the Arena team in prior work) rather than ad hoc or provider-weighted sampling (Section 6; cites Chiang et al., 2024 equation).  
  - Public transparency: quarterly logs summarizing models tested (including private), sampling rates, and deprecations.

- Follow-up research
  - Design and evaluate rating systems robust to best-of-N gaming (e.g., per-provider variance caps, hierarchical models that penalize selection bias).  
  - Explore graph-aware sampling that maximizes connectivity and reduces uncertainty while maintaining equitable provider exposure.  
  - Systematically study adversarial voting and de-anonymization defenses in live settings.  
  - Develop benchmark health dashboards (duplication, drift, concentration metrics) to monitor overfitting risk in real time.

- Downstream applications
  - More trustworthy public rankings for labs, enterprises, and regulators.  
  - Fairer allocation of community-provided data and annotator effort.  
  - Better model selection practices within organizations—reducing dependence on leaderboard spikes driven by selection bias.

> Bottom line: The paper shows—in theory, simulation, and practice—that current Arena mechanics allow a small group of providers to steer rankings via private best-of-N testing, high sampling exposure, and survivorship through silent deprecations. It provides concrete, actionable fixes that, if adopted, would make live leaderboards far more reliable indicators of true model quality.
