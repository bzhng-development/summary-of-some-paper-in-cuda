# Can LLMs Generate Novel Research Ideas? A Large‑Scale Human Study with 100+ NLP Researchers

**ArXiv:** [2409.04109](https://arxiv.org/abs/2409.04109)
**Authors:** Chenglei Si, Diyi Yang, Tatsunori Hashimoto
**Institutions:** Stanford University

## 🎯 Pitch

This study establishes a groundbreaking framework for assessing whether large language models (LLMs) can generate expert-level novel research ideas by conducting the first large-scale, controlled evaluation with NLP experts. It reveals that AI-generated ideas are significantly more novel than human ideas, highlighting their potential as valuable ideation partners in scientific research, while underscoring the need for improved diversity and evaluation mechanisms to enhance their feasibility and execution.

---

## 1. Executive Summary
This study builds the first large-scale, controlled, head-to-head evaluation of whether large language models (LLMs) can generate novel, expert-level research ideas. Using blinded reviews from 79 qualified NLP researchers on 147 standardized proposals (49 human-written, 49 AI-generated, 49 AI-generated with human reranking), the study finds that AI ideas are rated significantly more novel than human ideas while being slightly weaker on feasibility (Figure 2; Tables 7–9). The work also identifies core obstacles to autonomous “research agents,” including limited idea diversity at scale and unreliable LLM self-evaluation (Figure 4; Table 11).

## 2. Context and Motivation
- Problem addressed
  - Can current LLMs generate research ideas that are truly novel at an expert level, not just plausible extensions of known work? Prior “AI research agent” papers proposed pipelines but lacked large-scale, expert comparisons against human researchers and often used weaker evaluators such as LLM-as-a-judge or small reviewer pools (Section 1; Related Work, Section 9).
- Why it matters
  - Research ideation is the first and most creative step in scientific discovery. If LLMs can produce genuinely novel ideas, they could amplify or partially automate parts of the research process, but only if their outputs stand up to expert assessment and rigorous controls (Section 1).
- Limitations of prior approaches
  - Earlier agent papers evaluated with few reviewers, constrained idea length, or used LLMs as judges (Section 1; Related Work, Section 9), which can inflate perceived performance and do not establish statistically significant, expert-level comparisons.
- Positioning of this work
  - This study focuses on rigorous evaluation rather than proposing a complex new agent. It constructs:
    - A matched topic design across seven NLP sub-areas to prevent topic-preference confounds (Appendix A).
    - A standardized proposal template and LLM-based style normalization to remove writing-style cues (Appendices B–D).
    - A large expert pool for blinded review with clearly defined metrics (Appendix E).
  - It also implements a minimalist but strong ideation agent to make the human–AI comparison meaningful (Section 3).

## 3. Technical Approach
This is an evaluation-centric paper with two coordinated pipelines: (a) how ideas are produced and standardized, and (b) how they are reviewed.

A. Research idea space and standardization
- Scope: “Prompting-based NLP research” across seven topics—Bias, Coding, Safety, Multilingual, Factuality, Math, and Uncertainty (Appendix A). This domain balances creativity with executability on modest resources.
- Proposal template: All ideas—human or AI—must fill a detailed template with Title, Problem, Motivation, Proposed Method, Step-by-Step Experiment Plan, Test Case Examples, and a Fallback Plan (Appendix B). This forces comparable specificity and makes feasibility reviewable.
- Style normalization: To remove stylistic signals that could reveal origin, an LLM rewrites each idea to a common style while preserving content; the first author manually verified content preservation (Appendix D). A small pilot showed experts then guessed AI vs. human at chance (~50%), indicating style cues were largely removed (Section 2, “Idea Writeup”).

B. Three experimental conditions
- Condition 1: 49 human-written ideas (Section 4.1), each author choosing one of the seven topics; average time spent 5.5 hours; 902 words long (Table 3).
- Condition 2: 49 AI-generated ideas: top-ranked outputs from the ideation agent (Section 3).
- Condition 3: 49 AI-generated + human rerank: a human expert selected top ideas from the same agent generations to approximate an upper bound when humans filter AI outputs (Section 2, Review and Evaluation; Section 3.3). Overlap with Condition 2 is only 17/49, revealing discrepancies between AI-ranking and expert judgment (Appendix I; Table 12).

C. The ideation agent (Section 3)
- Retrieval-augmented generation (RAG)
  - The agent queries the Semantic Scholar API through three structured actions—`KeywordQuery`, `PaperQuery`, and `GetReferences`—until up to 120 relevant papers are retrieved (Section 3.1).
  - The LLM then scores and re-ranks retrieved papers by relevance, empiricism, and inspirational potential (1–10 scale) to ground idea generation in recent literature (Section 3.1).
- Idea generation at scale
  - For each topic, the agent over-generates 4,000 “seed ideas,” each prompted with 6 demo exemplars and titles/abstracts from 10 randomly selected retrieved papers (Section 3.2; Appendix F–G).
  - Deduplication: seed ideas are embedded with Sentence-Transformers (`all-MiniLM-L6-v2`); pairs with cosine similarity ≥0.8 are considered duplicates. Only about 5% survive as unique ideas (Section 3.2), revealing a sharp diversity bottleneck later quantified in Figure 4.
  - Surviving ideas are expanded into full proposals using the standard template (Appendix B).
- Ranking
  - Pairwise LLM ranker calibrated on 1,200 ICLR 2024 LLM-related submissions. Rather than predicting absolute scores (which the LLM handled poorly), the system performs pairwise comparisons with a Swiss-system tournament across N=5 rounds (Section 3.3; Table 1).
  - Sanity check: top-10 vs. bottom-10 ranked ICLR submissions have large review-score gaps (e.g., gap 1.73 in round N=5; Table 1), indicating the ranker picks stronger papers on average.

D. Human participant pools and review protocol (Section 4)
- Experts:
  - 49 idea writers from 26 institutions; 79 reviewers from 32 institutions; 298 total reviews (Section 4.1–4.2; Tables 15–16).
  - Reviewers average 15 papers and 635 citations on Google Scholar; 72/79 have reviewed major conferences (Table 2).
- Assignment and review form:
  - Reviewers pick topics and loads; each idea receives 2–4 reviews; no one reviews their own idea or from the same institution (Section 4.4).
  - Ratings (1–10) on Novelty, Excitement, Feasibility, Expected Effectiveness, plus Overall; each with calibrated definitions (Appendix E). Reviewers also report Familiarity and Confidence (Table 6).
  - Quality indicators: reviews average 232 words and ~32 minutes, comparable in length and confidence to ICLR 2024 reviews (Table 6).

E. Statistical testing (Section 5)
- Three complementary analyses to address dependence and reviewer-bias confounds:
  1) Test 1: each review as a datapoint (Table 7; Figure 2).
  2) Test 2: idea-level averages (N=49 per condition; Table 8).
  3) Test 3: reviewer-level differences (within-reviewer contrasts; Table 9).
- Mixed-effects models add a fourth lens with Topic, Reviewer, and Idea as random effects (Appendix N; Table 17).

## 4. Key Insights and Innovations
1) A controlled, scalable protocol for expert-level ideation evaluation
- What’s new: Matched topics, standardized templates, style normalization, and multi-metric blind review with 79 experts. This resolves common confounds—topic drift, verbosity bias, and writing-style signals.
- Why it matters: It provides a reusable blueprint for future human–AI ideation studies beyond NLP (Sections 2, 4; Appendices A–E).

2) Strong evidence that AI ideas are more novel than expert ideas
- Evidence: Across all three statistical tests—review-level (Table 7), idea-level (Table 8), and reviewer-level within-subject differences (Table 9)—AI ideas score higher on Novelty with p<0.05. For example, in Test 2: Human 4.86±1.26 vs. AI 5.62±1.39 (p=0.03) and AI+Human Rerank 5.78±1.07 (p=0.00) (Table 8).
- Significance: Moves the debate from anecdotes to a statistically powered conclusion that current LLMs can generate ideas experts deem novel.

3) Inference-time scaling has a diversity ceiling
- Finding: When generating 4,000 seeds per topic, the share of non-duplicate ideas rapidly decays; the number of unique ideas plateaus at ~200 (about 5% of seeds), despite continued sampling (Figure 4).
- Why it matters: Simply sampling more from the same model does not keep producing fresh ideas; future agents need explicit diversity mechanisms.

4) LLMs are unreliable judges of idea quality
- Finding: On a balanced top/bottom-25% ranking task derived from human scores, LLM judges underperform human agreement: GPT‑4o direct 50.0%, GPT‑4o pairwise 45.0%, Claude‑3.5 direct 51.7%, Claude‑3.5 pairwise 53.3%, “AI Scientist” reviewer 43.3% vs. human–human 56.1% (Table 11).
- Implication: Fully automated “AI-as-reviewer” remains brittle; human reranking of AI ideas tends to improve quality (AI+Rerank beats AI on several metrics; Figure 2, Tables 7–8; Appendix I).

5) How experts weigh criteria
- Correlations show Overall score aligns most with Novelty (r=0.725) and Excitement (r=0.854), but hardly with Feasibility (r=0.097), suggesting reviewers prioritize “new and exciting” over “easy to execute” (Table 10). This contextualizes why AI ideas—often bolder but less grounded—fare well on Overall in some tests.

## 5. Experimental Analysis
A. Evaluation setup
- Datasets/tasks: 147 proposals across seven topics (Appendix A), standardized by the same template (Appendix B).
- Metrics: 1–10 scales for Novelty, Excitement, Feasibility, Expected Effectiveness, and Overall (Appendix E).
- Conditions: Human (N=49 ideas), AI (N=49), AI+Human Rerank (N=49). Total reviews: 298; average 3.8 per reviewer (Tables 5–6).

B. Main quantitative results
- Novelty (robust finding across three tests)
  - Test 1 (review-level): Human 4.84±1.79 vs. AI 5.64±1.76 (p<0.01) and AI+Rerank 5.81±1.66 (p<0.001) (Table 7; Figure 2).
  - Test 2 (idea-level): Human 4.86±1.26 vs. AI 5.62±1.39 (p=0.03) and AI+Rerank 5.78±1.07 (p=0.00) (Table 8).
  - Test 3 (within-reviewer): mean differences AI–Human +0.94 (p<0.01) and AI+Rerank–Human +0.86 (p<0.01) (Table 9).
  - Mixed-effects model confirms positive coefficients for AI conditions on Novelty: +0.756 (p=0.023) and +0.902 (p=0.003) (Appendix N; Table 17).
- Excitement and Overall
  - Test 1: AI and AI+Rerank have higher Excitement than Human (p<0.05 and p<0.01, respectively); AI+Rerank also higher on Overall (p<0.05) (Table 7; Figure 2).
  - Test 2: AI+Rerank higher on Excitement (p<0.01); Overall is numerically higher but only marginally significant (p=0.06) (Table 8).
  - Test 3: reviewer-level differences show significant gains in Excitement for both AI conditions and in Overall for AI+Rerank (Table 9).
- Feasibility and Expected Effectiveness
  - Feasibility is similar or slightly worse for AI ideas; differences are small and not significant in Tables 7–8; reviewer-level differences are negative but non-significant (Table 9).
  - Expected Effectiveness differences are small and not consistently significant (Tables 7–9).

C. Do the experiments support the claims?
- Yes for novelty: the finding is replicated across three analysis frames plus a mixed-effects model and remains after multiple-hypothesis corrections (Section 5; Appendix N).
- Excitement/Overall: positive but somewhat less consistent; strongest when humans rerank AI ideas, indicating value in human curation (Figure 2; Tables 7–8; Appendix I).
- Feasibility: reviewers perceive AI ideas as slightly less executable, matching qualitative critiques (Section 8.1).

D. Robustness and diagnostics
- Inter-reviewer agreement: 56.1% on the balanced top/bottom-25% split—above random but lower than NeurIPS’21 (66.0%) and ICLR’24 (71.9%), reflecting high subjectivity in idea-only evaluation (Table 11).
- Reviewer focus: Overall correlates strongly with Novelty and Excitement but not with Feasibility (Table 10).
- Topic breakdown: Trends generally persist but small per-topic Ns limit significance (Appendix O; Figure 5).
- Human idea quality: Writers reported their submitted ideas were on average in the top 43% of their typical ideas; 37/49 created ideas during the task rather than submitting long-held plans (Section 6.1). Average effort ~5.5 hours (Table 3).

E. Failure analyses and ablations
- Diversity limit: non-duplicate ideas plateau at ~200 out of 4,000 seeds per topic (Figure 4).
- LLM-as-judge underperforms human agreement (Table 11); even the paper’s best LLM ranker (Claude-3.5 pairwise) reaches only 53.3% on the balanced ranking task.
- Qualitative failure modes for AI ideas include vague implementation steps, dataset misuse, missing baselines, unrealistic assumptions, resource-heavy designs, weak motivation, and ignoring best practices (Section 8.1). Human ideas skew more grounded but sometimes incremental (Section 8.1).

## 6. Limitations and Trade-offs
- Study scope and generality
  - Domain-limited: ideas focus on prompting-based NLP topics (Appendix A). Effects may differ in other fields (Discussion, Question 3).
  - Idea-level evaluation only: no end-to-end project execution in this phase; feasibility and impact judgments may shift once projects are built (Discussion, Question 2).
- Human baseline considerations
  - Human ideas may reflect “good-but-not-best” brainstorming under a 10-day constraint; self-reported median quality (~top 43%) suggests some headroom (Section 6.1).
- Subjectivity and reviewer variance
  - Inter-reviewer agreement is modest (56.1%; Table 11), indicating substantial subjectivity, especially when judging ideas without results.
- Agent-specific constraints
  - Idea diversity saturates quickly despite heavy sampling (Figure 4), limiting benefits of inference-time scaling.
  - The LLM pairwise ranker, while somewhat useful, is far from reliable and diverges notably from human reranking (Table 11; Appendix I).
- Implementation details
  - Style normalization uses an LLM; although the first author manually ensured content preservation, any automated rewriting carries a risk of subtle content shifts (Section 2, “Idea Writeup”; Appendix D).
- Resource assumptions
  - Feasibility judgments assume “abundant API access but limited GPU compute” and 1–2 months of student time (Appendix E), which may not match all labs.

## 7. Implications and Future Directions
- What changes in the field?
  - LLMs can be credible ideation partners: across topics, their proposals are judged more novel than experts’ baseline ideas when evaluated under strong controls (Figures 1–2; Tables 7–9).
  - Human–AI collaboration beats either alone: human reranking of AI ideas improves Excitement and sometimes Overall (Figure 2; Tables 7–8), suggesting a practical workflow—over-generate with an LLM, then curate with experts.
  - Fully automated research agents remain out of reach: idea diversity caps and weak AI self-evaluation (Figure 4; Table 11) highlight the need for improved diversity mechanisms and robust evaluation that does not rely on LLM judges.

- Near-term applications
  - Brainstorming assistants in research labs: use the provided template (Appendix B), retrieval grounding, and large-batch generation, but route selection through human experts.
  - Program committees and funding panels: the review form (Appendix E) and calibration scales offer a structured way to assess “ideas-only” submissions if needed.

- Follow-up research suggested by the study
  - Execute the ideas: the team is launching an end-to-end study to implement ideas and evaluate whether novelty/feasibility judgments predict real research outcomes (Section 1; Discussion, Question 2).
  - Benchmark against accepted papers: cached AI ideas will be compared to accepted EMNLP 2024 papers to analyze overlap and quality (Discussion, Question 1; preregistration link).
  - Increase idea diversity: investigate sampling strategies, alternative model families, retrieval diversification, or explicit novelty-seeking objectives to break the plateau shown in Figure 4.
  - Replace or augment LLM-as-judge: develop hybrid evaluation pipelines intertwining expert raters, calibrated pairwise protocols, and transparent statistical models; avoid over-reliance on LLM judges (Table 11).
  - Ethical and sociotechnical guardrails (Section 11): mitigate flood risks of low-quality AI-generated submissions, clarify intellectual credit attribution, assess misuse potential, and monitor homogenization effects on research agendas.

> Core quantitative takeaway: “AI-generated ideas are judged as significantly more novel than human ideas across three tests (Tables 7–9), with human reranking further improving some metrics (Figure 2), while feasibility tends to be comparable or slightly lower and AI self-evaluation remains unreliable (Table 11).”

Overall, this work provides a careful, statistically grounded answer to a high-stakes question—LLMs can originate ideas experts find more novel than those from a time-constrained human baseline—while clarifying the conditions, caveats, and systems gaps that must be addressed for reliable research agents.
