# DIVERSITY EMPOWERS INTELLIGENCE: INTEGRATING EXPERTISE OF SOFTWARE ENGINEERING AGENTS

**ArXiv:** [2408.07060](https://arxiv.org/abs/2408.07060)

## 🎯 Pitch

This paper introduces Diversity Empowered Intelligence (DEI), a meta-framework that leverages the diverse strengths of multiple software engineering (SWE) agents by using a large language model committee to select the best solution from a pool of candidate bug-fix patches. By systematically harnessing intra- and inter-agent diversity, DEI dramatically improves automated bug resolution rates—far surpassing the best individual agent and even outperforming many closed-source solutions—advancing the state of collaborative AI for real-world software engineering challenges.

---

## 1. Executive Summary
This paper introduces DEI (Diversity Empowered Intelligence), a meta-module that sits on top of existing software engineering (SWE) agents and selects the best bug-fix patch from multiple candidate patches produced by different agents or different runs of the same agent. By using a large language model (LLM) as a code-review “committee,” DEI leverages diversity among agents to significantly increase the rate of successfully resolved real GitHub issues on SWE-Bench Lite, achieving up to 55.0% resolve rate in a top-performing committee and 34.3% with an all–open-source committee (Table 1).

## 2. Context and Motivation
- Problem addressed
  - LLM-based SWE agents can fix real bugs but no single agent is consistently best across all issues. The best open-source agent on SWE-Bench Lite solves only 27.3% of issues (Table 1), and different agents tend to solve different, only partially overlapping subsets of issues (Figure 1a).
  - Missing capability: a principled way to harness this diversity by combining agents so that the overall system solves markedly more issues.

- Why this matters
  - Real-world impact: Automating bug fixing across large repositories has huge practical value—developers face complex, interdependent codebases where locating and repairing faults is difficult and time-consuming (Section 1).
  - Theoretical significance: The work formalizes a meta-policy that chooses among specialized agent policies by context, extending single-agent frameworks into a coordinated, diversity-aware system (Section 3.3.1–3.3.2).

- Prior approaches and gaps
  - Individual SWE agents (e.g., OpenDevin, Moatless Tools, Agentless) emphasize different tools or workflows: some run code to reproduce bugs, others focus on search/localize/repair without execution (Section 1).
  - Multi-agent literature often uses static workflows, free-form group chat, or hierarchical task assignment (Section 2), but there is limited systematic evaluation of diversity across SWE agents and limited methods that re-rank patch candidates across agents at scale.

- Positioning of this work
  - The paper measures both intra-agent diversity (different runs of the same agent) and inter-agent diversity (different agents) and introduces DEI, a meta-policy that selects among multiple patch candidates using an LLM-based review and scoring pipeline (Section 3.3.3; Figure 2). It shows this meta-layer unlocks substantial latent performance.

## 3. Technical Approach
This section explains both the formal framework and its practical implementation.

- Problem formalization (Section 3.3.1)
  - The task of fixing an issue is cast as a Contextual Markov Decision Process (`CMDP`): 
    - `S` = states (e.g., repository state, open files), `C` = contexts (repository + issue description), `A` = actions (e.g., search, edit, run tests).
    - The objective is to pick actions that maximize cumulative reward (e.g., tests pass after the patch). Equation (1) gives the value function: maximize expected total reward over a trajectory.
  - Specialized agents and meta-policy (Section 3.3.2)
    - Each agent policy `π_i` is specialized to some context distribution `ρ_i` (Equation (2)).
    - DEI’s meta-policy `π_DEI` selects the most suitable agent per context to maximize overall return across all contexts (Equation (3)). Intuitively, it’s a “router” that chooses which agent’s solution to adopt for each issue instance.

- Practical instantiation: DEIBASE (Section 3.3.3; Figure 2)
  - Goal: pick the best patch among multiple candidates for the same issue (from N different agents or N different runs).
  - Inputs for each candidate patch (Step 1: Input Construction)
    - Issue description.
    - Relevant context: code snippets identified as likely relevant. To fit LLM context limits and guide attention, DEIBASE uses relevant code spans identified by the open-source Moatless Tools agent (Section 3.3.3).
    - Code before the patch and code after the patch (not a diff). This design choice simplifies LLM reading compared to unified diff format (Section 3.3.3).
  - Review process (Step 2: Explanation Generation)
    - The LLM generates a structured set of explanations in a fixed order, where each step builds on the previous:
      1) Issue explanation: what the bug is and why it matters.
      2) Context explanation: why each provided code span is relevant.
      3) Location explanation: whether the patch edits the faulty code region.
      4) Patch explanation: how the patch fixes the bug.
      5) Conflict detection: whether the patch breaks or conflicts with other relevant code.
    - The prompt explicitly instructs the LLM to reference earlier explanations when producing later ones (Section 3.3.3).
  - Scoring and selection (Step 3: Patch Scoring)
    - Based on its own reasoning, the LLM assigns a score (1–10) guided by a rubric (e.g., wrong location is a severe error with heavy penalty; minor style issues are a small penalty).
    - DEIBASE optionally uses multiple “votes” (independent scoring passes) and averages the scores to reduce LLM variability (Figure 4). In most experiments, 10 votes per candidate are used.
    - The system then selects the top-scoring candidate(s). For evaluation, they study `n@k` with `n=1`: pick one candidate from k available (Section 4.1.2).

- Why this approach?
  - Evaluation can be easier than generation: LLMs often judge correctness more reliably than they can author correct code from scratch. DEIBASE exploits this asymmetry by using an LLM as an evaluator rather than an author (Section 3.3.3).
  - Token and comprehension constraints drive inputs: limiting to relevant spans and providing pre/post code avoids diff parsing complexity and keeps within context limits (Section 3.3.3).
  - Diversity-aware routing: the CMDP meta-policy view (Equations (2)–(3)) motivates dynamically choosing among diverse specialized agents, instead of trying to build a single “universal” agent.

## 4. Key Insights and Innovations
- Quantifying diversity across agents and runs (Section 4.2.1; Figure 3; Table 2)
  - Novelty: A systematic measurement using metrics `Union@k`, `Intersect@k`, `Average@k` (Section 4.1.2).
  - Insight: Inter-agent diversity is large. With 10 different agents, the oracle union (if you could always choose the correct patch) is 54.3% vs. 26.6% average per agent (Table 2, top block, k=10).
  - Significance: There is a big untapped headroom if one can select the right patch per instance.

- A simple, general meta-policy that turns diversity into accuracy (Sections 3.3.2–3.3.3; 4.2.2)
  - Novelty: Treat final patches as the “action space” of a meta-policy and use an LLM review committee to re-rank them (Figure 2).
  - Significance: This architecture is model-agnostic and sits atop any agent(s); it does not require re-engineering existing tools or workflows.

- LLM-based code review with structured reasoning improves selection (Figure 4; Table 3)
  - Novelty: Structured multi-part explanations (issue/context/location/patch/conflict) before scoring.
  - Evidence: Removing explanations reduces performance across settings (Table 3). Adding more review “votes” generally increases resolve rates (Figure 4).

- Multi-agent committees outperform best single agents on the leaderboard (Table 1)
  - Evidence:
    - Open-source-only committee: 
      > “DEIBASE-Open” achieves 34.3% vs. the best open baseline at 27.3% (Agentless), a +7.0 absolute improvement (Table 1).
    - Best-performing committee:
      > “DEIBASE-1” reaches 55.0% on SWE-Bench Lite, topping the leaderboard subset shown (Table 1).
  - Significance: Diversity does empower intelligence; light-weight meta-selection yields substantial gains without redesigning agents.

## 5. Experimental Analysis
- Evaluation setup (Section 4.1)
  - Dataset: SWE-Bench Lite (300 real GitHub issues with hidden unit tests that reflect whether a patch fixed the bug).
  - Agents:
    - Inter-agent diversity: 10 distinct leaderboard entries, both open- and closed-source, each with 26–31% resolve rate (Section 4.1.1; Appendix A.1 lists the order used).
    - Intra-agent diversity: 10 independent runs each for Agentless, Moatless Tools, and Aider (three strong open-source agents).
  - Metrics (Section 4.1.2)
    - `Resolve rate`: fraction of issues resolved by a single system.
    - `Union@k`: oracle best-case across k candidates (upper bound if selection is perfect).
    - `Intersect@k`: adversarial worst-case (issues solved by all k).
    - `Average@k`: expected performance under random selection.
    - `n@k` with `n=1`: performance of the re-ranker (DEI picks one from k). A perfect re-ranker matches `Union@k`.

- Main results (Section 4.2; Figure 3; Table 2; Table 1)
  - Diversity headroom:
    - 10 different agents (k=10): 
      > Union@k = 54.3%, Average@k = 26.6%, Intersect@k = 4.7% (Table 2).  
      This shows large diversity: many issues are solved by someone, but not consistently.
  - DEI selection improves over random:
    - 10 different agents (k=10): 
      > DEI achieves 35.7% vs. Average@k = 26.6% (∆ = +9.1) (Table 2).
    - 10 runs of Agentless (k=10): 
      > DEI 26.0% vs. Average@k 20.4% (∆ = +5.6) (Table 2).
    - 10 runs of Aider (k=10): 
      > DEI 24.7% vs. 21.7% (+3.0) (Table 2).
    - 10 runs of Moatless (k=10): 
      > DEI 26.3% vs. 15.9% (+10.4) (Table 2).
    - Trend with k: Improvements grow with more candidates, then plateau or fluctuate slightly (Figure 3; Section 4.2.2).
  - Leaderboard-style committees (Table 1):
    - Best committee (“DEIBASE-1”): 
      > 55.0% resolve rate, the highest in the shown leaderboard slice.
    - Committee of strong closed-source agents (“DEIBASE-2”): 
      > 37.0%.
    - Open-source-only committee (“DEIBASE-Open”): 
      > 34.3%, beating many closed-source systems and the best single open-source system (27.3%).
    - The paper highlights this as a 25% relative improvement for the open-source group:  
      > from 27.3% to 34.3% (Abstract; Table 1).

- Ablations and diagnostics (Section 4.3; Figure 4; Table 3)
  - More committee votes help (Figure 4):
    - In three of four settings, even 1 vote already beats the average; where it doesn’t, 3 votes suffice to surpass the average.
    - Performance generally increases with more votes, supporting the use of score averaging to reduce LLM variance.
  - Explanations matter (Table 3):
    - With explanations vs. without: 
      > Open Agents: 34.6 vs. 32.3,  
      > Agentless: 26.0 vs. 23.0,  
      > Aider: 24.6 vs. 23.3,  
      > Moatless: 25.6 vs. 25.3.

- Do the experiments support the claims?
  - Yes, for the studied setting. Multiple, independent metrics show large diversity headroom and consistent improvements from DEI over random selection, across both inter-agent and intra-agent scenarios (Figure 3; Table 2).
  - The leaderboard-style results (Table 1) demonstrate competitive real-world performance, including with open-source-only committees.
  - Caveats: No statistical significance tests are reported; results depend on chosen candidate pools and the particular LLM used as reviewer (gpt-4o for committees in Table 1), so generality to other LLM evaluators is not directly shown.

## 6. Limitations and Trade-offs
- Assumptions and dependencies
  - DEIBASE assumes access to multiple patch candidates per issue—either from multiple agents or multiple runs—which may not be available in some deployments.
  - The reviewer LLM is central. Reported committees use gpt-4o as the evaluator (Table 1). Performance may vary with other evaluators; evaluator bias or failure modes (e.g., over-trusting superficial edits) are possible.
  - Relevant code spans are sourced from Moatless Tools (Section 3.3.3), and are not adapted per candidate patch. If spans miss important context or over-include noisy code, review quality may suffer.

- Scalability and cost
  - Computational cost grows with number of candidates and number of committee votes (Figure 4 uses up to 10 votes per candidate). In large-scale CI pipelines, this may translate to non-trivial latency and API cost.
  - As k increases, gains eventually plateau and can slightly fluctuate (Figure 3), suggesting diminishing returns and the need for more advanced selection strategies for large candidate pools.

- Scope of evaluation
  - The main benchmark is SWE-Bench Lite (300 issues). While widely used, it is a subset of real-world scenarios; generalization to enterprise-scale repos, non-Python ecosystems, or different bug types remains to be tested.
  - The method evaluates by selecting among finished patches; it does not attempt to improve the generation process itself (e.g., by providing targeted feedback to agents).

- Failure modes and open questions
  - Without executing tests during review, the LLM may sometimes prefer plausible but incorrect patches; the gap between DEI and the oracle `Union@k` (Table 2) indicates room for better verification.
  - Order effects: For inter-agent aggregation, the paper fixes a specific agent order (Appendix A.1), which could influence `@k` curves for intermediate k, though not the endpoints (k=1, k=10).

## 7. Implications and Future Directions
- How this changes the field
  - The work reframes “which agent should I use?” as “use many and select,” providing a practical, plug-in meta-layer that can immediately improve performance without touching existing agent implementations (Sections 3.3.2–3.3.3).
  - The empirical finding that inter-agent diversity is especially valuable (Figure 3, top-left; Table 2) encourages the community to cultivate heterogeneous agent designs rather than optimize a single pipeline in isolation.

- Follow-up research enabled or suggested
  - Stronger evaluators: Train a dedicated patch-ranking model (possibly fine-tuned on review explanations and pass/fail outcomes) and/or incorporate static analysis and lightweight execution (e.g., targeted unit tests or sandboxed checks) to close the gap to `Union@k`.
  - Better context retrieval: Make relevant code spans patch-dependent, not only issue-dependent (Section 3.3.3 hint), to reduce noise and increase the reviewer’s diagnostic precision.
  - Smarter selection under budget: Learn to pick a diverse subset of agents per issue (agent portfolio optimization) to maximize marginal gains while controlling cost.
  - Feedback to generation: Use reviewer explanations to guide agents in iterative repair (a loop between generation and evaluation) rather than one-shot selection.

- Practical applications
  - CI/CD systems: Deploy DEI as a post-generation patch selector across multiple internal or third-party agents to improve fix rates on automated bug reports.
  - Large engineering organizations: Maintain a “committee” of specialized agents (e.g., for specific libraries, testing frameworks, or coding styles) and use DEI to route issues to the most promising candidate patch.
  - Educational tooling: Provide structured review feedback (issue/context/location/patch/conflict) for human learners or junior developers, not just for automated selection.

Key takeaways in one place:
- Diversity is a resource. The gap between `Average@k` and `Union@k` is large—e.g., 26.6% vs. 54.3% with 10 agents (Table 2).
- A lightweight meta-policy—an LLM-based code review and scoring pipeline—can exploit this diversity to produce sizable gains (e.g., +9.1 points over random selection for 10 agents in Table 2; 34.3% with an open-source committee in Table 1).
- Structured explanations and multiple votes further stabilize and improve selection (Figure 4; Table 3).
- There remains substantial headroom to approach the oracle union, inviting research on stronger and more verifiable evaluators.
