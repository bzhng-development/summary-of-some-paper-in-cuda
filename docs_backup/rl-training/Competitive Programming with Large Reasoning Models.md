# Competitive Programming with Large Reasoning Models

**ArXiv:** [2502.06807](https://arxiv.org/abs/2502.06807)
**Authors:** OpenAI, Ahmed El‑Kishky, Alexander Wei, Andre Saraiva, Borys Minaiev, Daniel Selsam, David Dohan, Francis Song, Hunter Lightman, Ignasi Clavera, Jakub Pachocki, Jerry Tworek, Lorenz Kuhn, Lukasz Kaiser, Mark Chen, Max Schwarzer, Mostafa Rohaninejad, Nat McAleese, o3 contributors, Oleg Mürk, Rhythm Garg, Rui Shu
**Institutions:** OpenAI

## 🎯 Pitch

This paper demonstrates how scaling reinforcement learning in general-purpose reasoning models allows them to surpass domain-specific pipelines in competitive programming without human-crafted heuristics, exemplified by the `o3` model achieving gold-medal performance. This advancement signals a paradigm shift towards using generalized models for complex problem-solving across various fields, offering a promising pathway for the development of autonomous reasoning systems in software engineering and beyond.

---

## 1. Executive Summary
This paper investigates how to make large language models excel at competitive programming and real‑world software tasks by scaling reinforcement learning and letting models “think” at test time. It compares three systems — the general‑purpose reasoning models `o1` and `o3`, and a domain‑specialized pipeline `o1-ioi` — and shows that the scaled, general model `o3` achieves gold‑medal IOI performance and elite Codeforces ratings without hand‑engineered contest heuristics (Figures 5 and 7).

## 2. Context and Motivation
- Problem addressed
  - Can a general‑purpose “reasoning” language model, trained with reinforcement learning (RL) to generate extended chains of thought, match or surpass domain‑specialized pipelines on competitive programming? The paper contrasts:
    - General models: `o1` and a later checkpoint of `o3`.
    - A specialized system: `o1-ioi`, tailored for IOI 2024 with hand‑crafted test‑time strategies (Section 3).
- Why it matters
  - Competitive programming is a rigorous, objectively gradable testbed for complex reasoning and algorithm design (Section 1). Strong performance signals broader progress in mathematical and coding reasoning that could translate to software engineering, science, and other fields.
- Prior approaches and gaps
  - Earlier code LLMs (e.g., Codex) improved program synthesis; AlphaCode/AlphaCode2 solved contest problems by sampling up to a million candidates and then applying hand‑engineered selection pipelines at inference time [7, 6]. These pipelines were powerful but bespoke and compute‑hungry.
  - Emerging “large reasoning models” (LRMs) trained with RL (e.g., `o1`, DeepSeek‑R1, Kimi k1.5) improved chain‑of‑thought for math and code, but a key open question remained: Are domain‑specific, human‑designed inference strategies still necessary, or can general RL training learn test‑time strategies on its own (Section 1)?
- Positioning
  - The paper offers a clean comparison: start from a general reasoning model (`o1`), specialize and add inference heuristics (`o1-ioi`), then scale general RL further (`o3`) and test if the bespoke heuristics are still needed. It expands evaluation beyond contests to software engineering benchmarks (Section 5).

Definitions used throughout:
- `Large reasoning model (LRM)`: an LLM trained with RL to produce longer chains of thought and to self‑correct by exploring solution paths before committing to an answer (Section 2).
- `Test‑time compute`: the amount of model “thinking” and code execution performed per sample at inference (Figures 2 and 6).
- `Subtask` (IOI): IOI problems are partitioned into graded subtasks with differing constraints; contestants earn points per subtask (Section 3.2).
- `Codeforces rating`: a skill rating used by the Codeforces platform. The paper estimates model ratings by simulating contests and fitting to the rating model described by Codeforces (Appendix B.4).

## 3. Technical Approach
The paper examines three systems and how they are trained and used at test time.

- `o1`: a general LRM trained with RL
  - Chain‑of‑thought: `o1` generates an internal multi‑step reasoning trace before answering, allowing it to decompose problems and backtrack (Section 2).
  - Tool use: `o1` writes and executes code in a secure environment, runs public tests, and refines its solutions iteratively within a single sample (Section 2; footnote to the code execution tool).
  - Why this helps: code execution provides a verifiable feedback loop — compile, run on tests, fix errors — that improves robustness.

- `o1-ioi`: specialization + engineered contest pipeline
  - Additional coding‑focused RL fine‑tuning
    - Resume RL from `o1`; emphasize hard programming tasks; improve C++ generation and IOI submission formatting (Section 3.1).
  - Hand‑crafted test‑time strategy (Section 3.2)
    - Step 1: Subtask decomposition. Treat each IOI subtask as a separate target because scoring is per subtask.
    - Step 2: Massive sampling. For each subtask, sample 10,000 candidate solutions.
    - Step 3: Model‑generated test cases. Prompt the model to write C++ test‑input generators and validators; generate 256 random test inputs and keep ones passing ≥75% of validators.
    - Step 4: Clustering by behavior. Cluster programs that produce identical outputs on the generated test inputs.
    - Step 5: Reranking. Score solutions using:
      - a learned scoring function,
      - errors on the model‑generated tests,
      - public test failures, and
      - a per‑cluster penalty for repeated failed submissions.
      - Weights are tuned by random search on prior IOI tasks via simulated submissions (Section 3.2).
    - Step 6: Submission policy. Submit up to the IOI limit of 50 total submissions per problem in round‑robin order over subtasks (hardest first), picking the top solution from the top cluster. Once an easier subtask is solved, filter future candidates for harder subtasks by enforcing agreement with the solved subtask’s behavior on the generated tests — a powerful pruning mechanism (Section 3.2).
  - Why this approach: It mirrors AlphaCode’s “generate many, select a few” paradigm but integrates model‑written tests and a learned ranker.

- `o3`: scaled general RL without domain heuristics
  - Training: further RL scaling beyond `o1`, yielding more deliberate chains of thought and improved reliability (Section 4).
  - Emergent test‑time strategies: `o3` often writes a simple brute‑force reference solution and cross‑checks it against an optimized one to detect logic errors (Figure 6).
  - Simpler selection for IOI evaluation: sample ~1K solutions per problem from a single prompt (no subtask decomposition) and pick the top 50 with highest test‑time compute — no clustering/validators/reranking pipeline (Section 4.2).

- Evaluation methodology
  - Codeforces simulation (Appendix B)
    - Data: late‑2023 and 2024 Division 1 contests after model cut‑offs; contamination checked via embedding search (Appendix B.1).
    - Grading: run the full tests locally; allow up to 10 independent submissions per problem and mark solved if any pass, following AlphaCode (Appendix B.2).
    - Thinking‑time correction: to reduce the parallelism advantage of models, estimate each solved problem’s score as the median human score among contestants with the same number of failed attempts; use these scores to fit a model rating via the Codeforces probability function (Appendices B.3–B.4).
    - Reporting: also provide per‑problem pass@k and failure counts (Table 1).
  - IOI evaluations
    - Live competition for `o1-ioi`: strict IOI rules; 50 submissions per problem; 10 hours; 6 problems (Section 3.4).
    - Retrospective benchmarking for `o3`: same tasks and 50‑submission rule, but `o3` is a later checkpoint with fresher training data; the IOI 2024 tasks are after the training cut‑off and were checked for contamination (Section 4.2).
  - Software engineering tasks (Section 5)
    - HackerRank Astra: 65 realistic multi‑file coding challenges; no public tests, so hand‑crafted selection tactics offer less benefit (Section 5.1).
    - SWE‑bench Verified: 500 curated tasks that fix issues in the original SWE‑bench to ensure reliable grading (Section 5.2).

## 4. Key Insights and Innovations
- Scaling RL and test‑time compute pays off
  - Figure 2 shows accuracy steadily rising as both RL training compute and test‑time compute increase on a competitive mathematics proxy. This motivates the `o1-ioi` pipeline (more RL and more inference compute) and later the “RL scaling only” of `o3`.
- A specialized pipeline helps — but scaling a general model helps more
  - `o1-ioi` lifts Codeforces rating from `o1`’s 1673 (89th percentile) to 2214 (98th) when using the full test‑time strategy (Figure 3).
  - `o3` — without any hand‑engineered contest heuristics — jumps to 2724 (99.8th percentile) on the same benchmark (Figure 5), surpassing the specialized pipeline.
- Emergent verification behavior reduces reliance on human heuristics
  - Figure 6 shows `o3` writing a brute‑force solver to validate an optimized algorithm — a behavior the `o1-ioi` pipeline had to impose externally via clustering and validators (Section 3.2). This is a fundamental innovation: the model internalizes test‑time strategy through RL, simplifying deployment and improving generality.
- From contests to software engineering
  - Gains extend beyond contests. On SWE‑bench Verified, `o3` reaches 71.7% task success vs. `o1` at 48.9% and GPT‑4o at 33.2% (Figure 9). On HackerRank Astra, `o1` improves pass@1 to 63.92% (up from 50.91% for GPT‑4o), with `o1-preview` already showing a substantial bump (Figure 8). This suggests the reasoning improvements are not contest‑specific.

## 5. Experimental Analysis
- Setup and metrics
  - Codeforces benchmark
    - Contests: late‑2023/2024 Division 1 after training cut‑offs; full tests; up to 10 submissions per problem; rating fit by likelihood over contest rankings (Appendix B.1–B.4).
    - Metrics: estimated Codeforces rating and percentile; per‑problem pass@1, pass@10, and failure counts (Table 1).
  - IOI 2024
    - Live run for `o1-ioi` (strict rules, 50 submissions) and retrospective evaluation for `o3` (also 50 submissions).
    - Metrics: total points out of 600; gold threshold ~360 (Figure 7).
  - Software engineering
    - HackerRank Astra: pass@1 and average score (Figure 8).
    - SWE‑bench Verified: percent of issues correctly fixed (Figure 9).

- Main quantitative results
  - General competitive programming ability
    - “Reasoning” is a big step up from non‑reasoning: GPT‑4o scores 808 (11th percentile), `o1-preview` 1258 (62nd), and `o1` 1673 (89th) on Codeforces (Figure 1).
    - Specialization with engineered strategy helps: `o1-ioi` reaches 1807 (93rd). Simple “public test filtering” boosts to 2092 (96th). The full clustering/reranking/selection pipeline reaches 2214 (98th) (Figure 3).
    - Scaling general RL wins: `o3` achieves 2724 (99.8th percentile) (Figure 5). Figure 10 places `o3` among the top ~200 active Codeforces participants by solve rate and rating, while acknowledging the very best humans still outperform it.
  - Live IOI 2024 and retrospective IOI analysis
    - `o1-ioi` at IOI 2024 (50 submission limit):
      > Score 213 points; 49th percentile (Figure 4).
      > If 50 submissions were chosen at random, the mean score would be 156 — the test‑time selection strategy adds ~57 points (Figure 4).
      > With relaxed 10,000‑submission limit and no selection pipeline, score rises to 362.14 (gold threshold) (Figure 4; Appendix C lists samples).
    - `o3` on IOI 2024 (retrospective, 50 submissions, simple selection):
      > Score 395.64 — above the gold threshold without subtask prompts or hand‑crafted heuristics (Figure 7).
  - Software engineering
    - HackerRank Astra:
      > pass@1: GPT‑4o 50.91%, `o1-preview` 60.89%, `o1` 63.92%; average score: 69.52 → 75.55 → 75.80 (Figure 8).
    - SWE‑bench Verified:
      > GPT‑4o 33.2% → `o1-preview` 41.3% → `o1` 48.9% → `o3` 71.7% (Figure 9).

- Do the experiments support the claims?
  - The staged improvements (Figures 1, 3, 5) strongly support that (a) RL‑trained reasoning boosts coding, (b) domain‑specific pipelines add further gains, and (c) scaling RL in a general model surpasses the specialized pipeline.
  - The IOI comparison is particularly persuasive: `o1-ioi` needs heavy sampling plus a sophisticated selection pipeline and still lands at 213 under official limits; `o3` crosses gold with only 50 submissions and no human‑designed heuristics (Figure 7).
  - Robustness and fairness steps include contamination checks (Appendix B.1), alignment to human scoring via thinking‑time correction (Appendix B.3), and a per‑problem breakdown (Table 1) showing both successes and unsolved hardest problems (e.g., many 3200–3500‑rated tasks remain unsolved).

- Ablations and failure modes
  - Compute scaling curves (Figure 2) act as ablations on RL compute and test‑time compute.
  - Submission strategy ablations at IOI 2024 (Figure 4) quantify the value of the engineered selection pipeline versus random choice and relaxed limits.
  - Table 1 shows remaining weaknesses on the hardest Codeforces problems (e.g., many 3200–3500 problems show pass@10 near zero), indicating that top‑tier human champions still have an edge (Figure 10).

## 6. Limitations and Trade-offs
- Compute intensity
  - `o1-ioi` relies on sampling 10,000 solutions per subtask, model‑generated tests, clustering, and reranking (Section 3.2) — a major inference‑time cost.
  - Even `o3` depends on substantial test‑time compute (e.g., running brute force to validate optimized code; Figure 6) and hundreds to thousands of samples per problem (Section 4.2).
- Fairness and comparability to human contests
  - Models can think in parallel; the paper corrects this by scoring with median human times (Appendix B.3), but residual advantages may remain, and rating estimation is inherently indirect (Appendix B.4).
- Scope of problems
  - Interactive problems were excluded from the test set for convenience (Appendix B.1).
  - The per‑problem breakdown (Table 1) shows difficulty spikes at very high‑rated tasks; generalization to the absolute frontier of contest difficulty remains open.
- Data and contamination
  - The paper conducts embedding‑based checks and ensures contests are post‑cut‑off (Appendix B.1), but any closed‑source system faces reproducibility scrutiny.
- Differences in evaluation windows
  - The `o3` checkpoint used for IOI 2024 retrospective evaluation is later than the one used for Codeforces and includes fresher training data (Section 4.2). While IOI problems are verified post‑cut‑off, this timing difference complicates head‑to‑head comparisons with `o1-ioi`.

## 7. Implications and Future Directions
- Field impact
  - The central finding — that scaling general RL yields emergent test‑time strategies (Figure 6) and surpasses carefully engineered domain pipelines (Figures 5 and 7) — shifts the balance toward general reasoning models for complex domains. This mirrors trends in other areas where learned strategies replace hand‑tuned heuristics.
- Research directions
  - Reduce test‑time compute without sacrificing reliability: distill the emergent strategies, cache/reuse internal checks, or train models to learn when brute‑force cross‑checks are necessary.
  - Extend to interactive and adversarial settings: many contests include interactive problems; similar RL‑driven reasoning may generalize with the right tooling.
  - Interpretability of chains‑of‑thought: Figure 6 hints at structured internal workflows; understanding and making these strategies controllable could improve trust and efficiency.
  - Better evaluation protocols: standardized thinking‑time accounting and public leaderboards for LRM agents could improve comparability with humans.
- Practical applications
  - Developer tooling: models that automatically generate tests, create reference implementations, and cross‑validate optimized code can act as high‑reliability coding assistants.
  - Education and training: step‑by‑step reasoning with self‑verification is well suited for tutoring in algorithms and software engineering.
  - Broader reasoning domains: the “sample, verify, refine” pattern should transfer to mathematics, formal methods, scientific computing, and data analysis.

In short, the paper provides a detailed, head‑to‑head comparison showing that general RL‑trained reasoning models can learn test‑time behaviors (like self‑verification) that previously required bespoke pipelines — and that this shift delivers state‑of‑the‑art results in both competitive programming and practical software benchmarks.
