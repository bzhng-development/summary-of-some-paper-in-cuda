# Competitive Programming with Large Reasoning Models

**ArXiv:** [2502.06807](https://arxiv.org/abs/2502.06807)

## 🎯 Pitch

This paper demonstrates that scaling reinforcement learning in large language models enables them to autonomously develop sophisticated reasoning abilities, achieving gold-medal performance at the International Olympiad in Informatics and outcompeting nearly all human programmers on CodeForces—without any hand-crafted, domain-specific strategies. By rigorously benchmarking three OpenAI models (o1, o1-ioi, o3), the authors show that end-to-end RL alone is sufficient to yield state-of-the-art results on complex coding tasks, marking a pivotal shift from engineered solutions to scalable and general-purpose AI reasoning. This finding signals that learning-based approaches can unlock robust, broadly applicable AI systems for high-stakes reasoning domains beyond programming, including math and software engineering.

---

## 1. Executive Summary (2-3 sentences)
This paper investigates how reinforcement learning (RL) applied to large language models can turn them into large reasoning models (`LRMs`) that solve complex coding tasks competitively. It compares three systems—`o1` (general reasoning with tool use), `o1-ioi` (o1 fine‑tuned plus hand‑engineered test‑time strategies for the International Olympiad in Informatics), and an early `o3` checkpoint (scaled general-purpose RL)—and shows that scaling general-purpose RL yields gold‑medal IOI performance and a CodeForces rating near top human competitors without domain‑specific heuristics (Figures 5 and 7).

## 2. Context and Motivation
- Gap addressed:
  - Competitive programming demands precise algorithmic reasoning under strict time and memory limits. Previous LLM systems that performed well (e.g., AlphaCode/AlphaCode2) relied on massive sampling (up to 1M candidates per problem) and hand‑crafted selection heuristics at inference, which are expensive and domain‑specific (Section 1).
  - The open question: are domain‑specific, hand‑designed pipelines necessary, or can end‑to‑end RL training on general reasoning tasks make models discover equally effective test‑time strategies on their own (Section 1)?

- Why it matters:
  - Real‑world impact: Programmers and organizations need reliable AI that can synthesize correct, efficient code, not just plausible snippets. Competitive programming offers objective, rigorous grading at scale (Section 1).
  - Theoretical significance: Understanding whether reasoning strategies can be learned (rather than engineered) informs how to scale AI reasoning ability in broader domains like math and software engineering.

- Prior approaches and limitations:
  - Program synthesis with LLMs shows accuracy grows with model size and fine‑tuning (Section 1; [1], [2]).
  - AlphaCode/AlphaCode2 improved substantially via large‑scale sampling and selection heuristics, but still depended on hand‑engineered, coding‑specific inference logic (Section 1; [6], [7]).
  - Early chain‑of‑thought (CoT)–style LRMs (e.g., `o1`, independent work like DeepSeek‑R1, Kimi k1.5) improved reasoning but left unclear whether domain‑specific tricks remained necessary at test time (Section 1; [3], [12], [15]).

- Positioning:
  - `o1`: general LRM trained with RL that reasons using extended internal chains of thought and can call tools for code execution and testing (Section 2).
  - `o1-ioi`: a domain‑tailored system that adds coding‑focused RL fine‑tuning plus a sophisticated, hand‑designed test‑time pipeline for IOI (Section 3).
  - `o3`: a later, scaled general-purpose LRM trained with more RL compute that learns its own test‑time behaviors without domain‑specific heuristics, outperforming `o1-ioi` (Section 4).

## 3. Technical Approach
This section unpacks what the three systems do and how the evaluations are run.

- Definitions (paper‑specific or uncommon):
  - `Large reasoning model (LRM)`: a language model trained via RL to produce explicit, extended “thinking” (chain of thought) before final answers, and to use tools (e.g., code execution) effectively (Section 2).
  - `Test-time compute`: extra computation the model uses while answering a single prompt (e.g., producing long chains of thought, running code, trying multiple variants). Figures 2, 3 show performance increases from more test-time compute.
  - `IOI subtask`: IOI problems are split into subtasks, each with separate constraints and points; scores are the sum over subtasks. Competitors can submit multiple programs; the best score on each subtask counts (Section 3.2).
  - `Public tests`: sample tests provided in a problem statement or portal; `full tests` are hidden evaluator tests. Humans see pretests during contests; models here are evaluated against full tests offline (Appendix B.2).
  - `pass@k`: probability of solving with up to `k` independent attempts (here used in analysis like “pass@10” for CodeForces simulation; Appendix B.2).

- System 1 — `o1` (general-purpose LRM with tools; Section 2):
  - RL training produces longer internal reasoning that helps break down complex problems, explore alternatives, and self‑correct.
  - Tool use: runs generated code in a sandbox to compile, execute, and verify against tests, then iteratively refines the solution in a single sample (Section 2).

- System 2 — `o1-ioi` (domain‑specific fine‑tune + engineered inference; Section 3):
  - Coding‑focused RL fine‑tuning (Section 3.1):
    - Resume RL from an `o1` checkpoint.
    - Emphasize harder programming problems and C++ code generation, including runtime checks.
    - Produce outputs in IOI submission format.
  - Hand‑engineered test‑time pipeline (Section 3.2):
    - Problem decomposition: create one prompt per IOI subtask.
    - Massive sampling: generate 10,000 samples per subtask.
    - Model‑generated tests:
      - Ask the model to write C++ test input generators and validators per subtask.
      - Generate 256 random test inputs and accept each input that passes at least 75% of validators (Section 3.2, “Clustering”).
    - Clustering by behavior: group programs if they produce identical outputs on the 256 accepted inputs.
    - Reranking:
      - Score each sample by (i) a learned scoring function, (ii) errors on model‑generated tests, and (iii) failing public tests.
      - Give each cluster an aggregate score (mean of its samples minus a penalty for repeated submissions from that cluster). Tune penalty weights via random search on prior IOI problems by simulating submissions (Section 3.2, “Reranking”).
    - Submission strategy:
      - Up to 50 submissions total (IOI limit), allocated round‑robin across subtasks starting from the hardest.
      - When a subtask is solved, stop sampling for it.
      - For supersets of solved subtasks, filter candidates that don’t match outputs on previously solved easier subtasks to rapidly narrow viable options (Section 3.2, “Submission”).

- System 3 — `o3` (scaled general-purpose LRM; Section 4):
  - Goal: rely on learned test‑time behaviors rather than human‑designed pipelines.
  - Observed emergent behavior: for some problems, `o3` writes a simple brute‑force solution to verify correctness of its optimized solution by cross‑checking outputs, thereby catching subtle errors (Figure 6).
  - Evaluation procedure differs from `o1-ioi` (Section 4.2):
    - Single prompt per IOI problem (no subtask‑specific prompts).
    - Sample 1,024 solutions per problem; choose the 50 with the highest test‑time compute (i.e., the most deliberate reasoning).
    - No clustering or manual selection heuristics; yet, solutions often cover many or all subtasks.

- CodeForces contest simulation (Appendix B):
  - Data: Division 1 contests from late 2023 and 2024 (post‑training cutoff), excluding one with an interactive problem (Appendix B.1).
  - Contamination check: embedding search to ensure test problems are unseen (Appendix B.1).
  - Grading: run against full hidden tests; following AlphaCode, allow up to 10 independent submissions; mark solved if any one passes (`pass@10`, Appendix B.2).
  - Time scoring fairness: because models can think in parallel, the contest score for solved problems is set to the median score of humans who solved the same problem with the same number of failed attempts (Appendix B.3).
  - Rating estimation: compute the single model rating that maximizes the likelihood of observed rankings vs human ratings, using the Elo‑style logistic model described in Appendix B.4.

- IOI 2024 evaluations:
  - Live competition with `o1-ioi`: 10 hours, six problems, 50 submissions per problem (Section 3.4).
  - Retrospective eval with `o3`: same six problems, 50‑submission limit; 1,024 samples/problem; pick 50 highest compute (Section 4.2).
  - “Gold” threshold ≈ 360 points (Section 4.2 and Figure 7).

- Software engineering benchmarks (Section 5):
  - HackerRank Astra: 65 realistic, multi‑file coding tasks across frameworks. No public tests, so test‑time heuristics like public‑test filtering are not available (Section 5.1).
  - SWE‑bench Verified: curated 500 GitHub issues with corrected grading and clearer specs; each model gets up to 5 tries to produce a patch; evaluations averaged over 3 trials; system failures aren’t penalized but retried (Section 5.2).

## 4. Key Insights and Innovations
- Emergent learned test‑time strategies from general RL (fundamental):
  - `o3` learns to self‑validate by generating a brute‑force checker and cross‑comparing with its optimized solution (Figure 6). This behavior replaces much of the human‑designed selection pipeline needed for `o1-ioi`, demonstrating that end‑to‑end RL can teach models how to test their own code.

- General-purpose RL scaling beats domain‑specific pipelines (fundamental):
  - Despite `o1-ioi`’s strong domain tailoring and complex test‑time engineering, a later `o3` checkpoint surpasses it with a simpler sampling policy (Section 4.2). On IOI 2024, `o3` achieves gold within the standard 50‑submission limit, while `o1-ioi` needed either hand‑engineered selection or relaxed submission limits to reach gold (Figure 7).

- Two knobs that reliably raise reasoning performance: more RL training compute and more test‑time compute (broad, empirically grounded insight):
  - Figure 2 shows monotonic gains in competitive mathematics (AIME) both when increasing RL training compute (“o1 AIME accuracy during training”) and when allocating models more test‑time compute.

- A practical, reproducible contest‑simulation methodology for CodeForces (incremental but important):
  - The paper details how to approximate fairness (Appendix B.3), guard against data leakage (Appendix B.1), and map performance to an estimated CodeForces rating via an Elo‑like model (Appendix B.4). This framework helps the community compare models against strong human baselines.

## 5. Experimental Analysis
- Evaluation methodology
  - CodeForces (Sections 2.1, 3.3, 4.1; Appendix B):
    - Division 1 contests after training cutoff.
    - Full hidden tests; `pass@10` acceptance; rating estimation as in Appendix B.4.
    - Time advantage corrected by replacing model time with median human score at the same failure count (Appendix B.3).
  - IOI 2024 (Sections 3.4, 4.2):
    - `o1-ioi` competed live under official rules (50 submissions/problem).
    - `o3` evaluated retrospectively with the same limits and a simple selection rule (top‑compute 50 of 1,024 samples).
  - HackerRank Astra (Section 5.1):
    - 65 multi‑file tasks; report `pass@1` and average proportion of tests passed. No public tests; hand‑crafted test-time heuristics are moot here.
  - SWE‑bench Verified (Section 5.2):
    - 500 curated GitHub issues; 5 attempts per task; percent of issues resolved.

- Main quantitative results
  - CodeForces ratings
    - Figure 1:
      - `gpt‑4o`: 808 (11th percentile)
      - `o1‑preview`: 1258 (62nd)
      - `o1`: 1673 (89th)
    - Figure 3 (o1-ioi ablations):
      - `o1-ioi` vanilla: 1807 (93rd)
      - + public‑test filtering: 2092 (96th)
      - + full test‑time strategy (clustering + reranking): 2214 (98th)
    - Figure 5:
      - `o3` (early checkpoint): 2724 (99.8th)
    - Appendix B.6 Table 1 provides a per‑contest, per‑problem breakdown (e.g., contest 1951 shows solutions up to G=3100 rating, H/I unsolved; contest 1916 solves up to H2=2700).
  - IOI 2024 scores
    - Figure 4 (o1‑ioi, live contest):
      - 50 submissions with hand‑crafted selection: 213 points (49th percentile).
      - If randomly selecting 50 submissions from the 10k pool: 156 points.
      - If allowed 10k submissions (no selection strategy): 362.14 points (above gold threshold).
    - Figure 7 (o3 retrospective vs o1‑ioi):
      - `o1‑ioi` (50 submissions): 213
      - `o1‑ioi` (10k submissions): 362.14
      - `o3` (50 submissions, simple top‑compute selection): 395.64 (gold)
  - Emergent test‑time behavior
    - Figure 6: `o3` writes and runs a brute‑force solver to validate outputs from an optimized solution, a behavior not specified by humans.
  - HackerRank Astra (Figure 8):
    - `gpt‑4o`: 50.91% pass@1; 69.52% avg score
    - `o1‑preview`: 60.89% pass@1; 75.55% avg score
    - `o1`: 63.92% pass@1; 75.80% avg score
  - SWE‑bench Verified (Figure 9):
    - `gpt‑4o`: 33.2% correct
    - `o1‑preview`: 41.3%
    - `o1`: 48.9%
    - `o3`: 71.7%

- How convincing are the experiments?
  - The CodeForces methodology addresses key fairness issues (parallel “thinking”) and data leakage (Appendix B.1–B.4). The large, unseen contest set and per‑problem breakdowns in Table 1 support the headline rating claims.
  - The IOI analysis isolates the impact of test‑time pipelines vs general RL. Figure 4 acts as an ablation: random 50 vs engineered selection vs lifting the submission limit. Figure 7 then shows a later `o3` checkpoint beating `o1‑ioi` with a simpler selection rule.
  - The software‑engineering results (HackerRank Astra, SWE‑bench Verified) generalize the reasoning gains beyond competitive programming (Figures 8–9).

- Failure cases and caveats
  - The very hardest CodeForces problems (e.g., 3300–3500 rating) remain frequently unsolved, and overall solve rate still trails the very top humans (Figure 10).
  - The CodeForces simulation allows `pass@10` against full tests (Appendix B.2), different from how humans experience pretests during the contest, though authors argue Division 1 pretests are typically strong.
  - Interactive problems were excluded (Appendix B.1), so results do not cover that category.

## 6. Limitations and Trade-offs
- Assumptions and scope:
  - Heavy reliance on a secure code‑execution sandbox and tool use (Section 2). Real‑world environments with nondeterminism, external services, or complex build systems may pose additional challenges not captured here.
  - The CodeForces simulation and IOI retrospective results are careful but still approximations of live human competition dynamics (Appendix B.2–B.4).

- Compute and scalability:
  - `o1-ioi` depends on very large test‑time sampling (10k samples/subtask), plus custom clustering and reranking (Section 3.2). This is expensive and domain‑specific.
  - `o3` uses less hand engineering but is a larger, more compute‑intensive model trained with substantially more RL compute (Sections 4–5). The paper does not disclose precise compute budgets, which makes cost–benefit analysis hard.

- Transparency:
  - While the paper explains inference behaviors and evaluation in detail, it does not detail `o3`’s architecture or exact RL training algorithmic choices. That limits reproducibility of the training process even though evaluation is well described.

- Generalization boundaries:
  - Interactive programming tasks were excluded in CodeForces (Appendix B.1).
  - Some contest‑specific features (e.g., pretests vs full tests, time pressure, hacking in CodeForces) are not fully replicated.
  - The retrospective IOI evaluation for `o3` samples only 1,024 solutions per problem and uses highest‑compute selection; different sampling/selection policies could change scores.

- Measurement choices:
  - CodeForces time‑scoring adjustment uses median human scores for similar fail counts (Appendix B.3). Although sensible, alternative adjustments (e.g., exact wall‑clock model times at fixed GPU budgets) could yield different estimated ratings.

## 7. Implications and Future Directions
- How this work shifts the landscape:
  - It provides strong evidence that scaling general‑purpose RL produces LRMs that learn sophisticated test‑time strategies (like self‑verification) without domain‑specific heuristics, while achieving state‑of‑the‑art results in competitive programming (Figures 5–7). This is a step from “engineer the inference pipeline” toward “train the behavior.”

- Practical applications:
  - Competitive programming assistance, automated bug‑fixing, and feature implementation in real codebases, suggested by gains on SWE‑bench Verified (Figure 9) and HackerRank Astra (Figure 8).
  - Education and contest training: models that can generate brute‑force checkers and cross‑validate solutions offer novel pedagogy and debugging workflows (Figure 6).

- Research directions enabled:
  - Learned testing strategies: formalize and strengthen the emergent behaviors (e.g., test generation, property checks, differential testing) so models can systematically validate their own code as problems get harder.
  - Cost‑aware reasoning: optimize the trade‑off between test‑time compute and accuracy (Figure 2) with dynamic compute allocation, e.g., deciding when to invoke brute‑force validation vs trusting an optimized solution.
  - Broader benchmarks: extend beyond non‑interactive problems to interactive protocols and multi‑component systems; include runtime constraints that penalize slow brute‑force checks to encourage algorithmic efficiency.
  - Interpretability of reasoning: analyze chains of thought and learned verification routines to understand failure modes on very hard problems (e.g., 3300–3500 CodeForces tasks in Table 1).

- Standards and evaluation:
  - The contest‑simulation blueprint in Appendix B (data leakage checks, pass@k policy, rating estimation, time‑scoring correction) can serve as a community reference for fair, apples‑to‑apples evaluation of LRMs on competitive programming.

> Bottom line: With `o3`, general-purpose RL training—not domain‑specific test‑time heuristics—emerges as the more robust route to top‑tier coding and reasoning performance, achieving a 2724 CodeForces rating (99.8th percentile; Figure 5) and a gold‑medal score on IOI 2024 with only 50 submissions (Figure 7), while also transferring to real‑world software tasks (Figures 8–9).
