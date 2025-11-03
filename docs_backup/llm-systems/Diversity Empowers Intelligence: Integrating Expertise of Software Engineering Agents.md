# Diversity Empowers Intelligence: Integrating Expertise of Software Engineering Agents

**ArXiv:** [2408.07060](https://arxiv.org/abs/2408.07060)
**Authors:** Kexun Zhang, Weiran Yao, Zuxin Liu, Yihao Feng, Zhiwei Liu, Rithesh Murthy, Tian Lan, Lei Li, Renze Lou, Jiacheng Xu, Bo Pang, Yingbo Zhou, Shelby Heinecke, Silvio Savarese, Huan Wang, Caiming Xiong
**Institutions:** Carnegie Mellon University, Salesforce Research, University of Texas at Austin, other institutions (co-authors)

## 🎯 Pitch

This paper introduces DEI (Diversity Empowered Intelligence), a meta-policy that leverages the distinct strengths of multiple software engineering agents to enhance bug-fixing performance by up to 55% on SWE-Bench Lite. By using LLM-based code reviews to re-rank bug-fixing patches, DEI transforms agent diversity from a challenge into an advantage, ensuring more effective resolution and showcasing the power of diverse agent collaboration in software maintenance.

---

## 1. Executive Summary
This paper introduces DEI (Diversity Empowered Intelligence), a meta-policy that sits on top of existing software engineering (SWE) agents and selects the best bug-fixing patch from multiple candidate patches. Its significance is twofold: it empirically shows that different SWE agents (and even repeated runs of the same agent) solve very different issues, and it turns that diversity into performance gains by using an LLM-based code review committee to re-rank patches—achieving up to 55.0% resolve rate on SWE-Bench Lite and a 25% improvement within an all–open-source agent group (Table 1).

## 2. Context and Motivation
- Problem addressed
  - LLM-based SWE agents can fix real GitHub bugs but no single framework excels across all issues. Agents show strong specialization: they solve different subsets of problems even when their overall success rates are similar (Figure 1a–b).
  - The key gap: the community lacks a principled way to leverage this diversity—i.e., to coordinate multiple agents and pick the right solution for each issue.

- Why it matters
  - Practical: Automated bug fixing reduces engineering effort and accelerates maintenance across large codebases. SWE-Bench Lite uses real issues and hidden tests; passing them implies tangible fix quality (Section 3.1).
  - Scientific: Understanding and exploiting diversity across agents can raise ceilings beyond any individual agent by combining complementary strengths (Figure 1c).

- Prior approaches and shortcomings
  - Individual SWE agents use different tools and workflows (Section 2): e.g., OpenDevin executes code to replicate bugs, while others like Moatless Tools and Agentless do not.
  - Multi-agent systems exist (Section 2), but common patterns—static workflows, group chat ensembles, or hierarchical assignment—do not directly address selecting the best fix among candidate patches for the same issue across heterogeneous agent designs.

- Positioning
  - The paper formalizes agent diversity (Section 3.2) and frames selection among multiple agents as a meta-policy optimization problem (Section 3.3.2). It then implements a practical LLM-based reviewer, DEIBASE, that re-ranks candidate patches (Section 3.3.3, Figure 2).

## 3. Technical Approach
At a high level, DEI treats each SWE agent as a specialized policy and learns a meta-policy that chooses which agent’s output to use for each issue. DEIBASE is a simple instantiation: it gathers multiple candidate patches, asks an LLM to “review” them using structured prompts, assigns a score, and submits the top-ranked patch.

- Formalization (Section 3.3.1–3.3.2)
  - The paper models SWE agents in a contextual Markov decision process (CMDP). Intuitively:
    - `S` is the state space (e.g., repository state).
    - `C` is the context (repository info + issue description).
    - `A` is the action space (tools such as search/edit/run tests).
    - The agent follows a policy `π(st, c)` and aims to maximize cumulative reward (Eq. 1).
  - Consider `N` agent policies `{π1,…,πN}`, each strong in sub-distribution `ρi` of contexts. A single policy may underperform outside its niche (Eq. 2).
  - DEI defines a meta-policy `π_DEI` that selects the best agent for each context to maximize expected reward across all contexts (Eq. 3). Practically, this becomes selecting the best candidate patch for each issue.

- DEIBASE: LLM code-review committee (Section 3.3.3; Figure 2)
  - Inputs per candidate patch:
    - Issue description (from SWE-Bench Lite).
    - Relevant context snippets (extracted by an agent; the paper uses Moatless Tools spans to keep prompts within context limits).
    - Code “before patch” and “after patch” as separate blocks (easier for LLMs to parse than unified diffs).
  - Review procedure (prompted, in a fixed order to scaffold reasoning):
    1) Issue explanation: articulate the bug’s symptoms/impact.
    2) Context explanation: relate each provided code snippet to the issue.
    3) Location explanation: assess whether the patch edits the faulty code.
    4) Patch explanation: reason how the changes address the issue.
    5) Conflict detection: check for contradictions with other relevant code.
  - Scoring and selection:
    - The LLM assigns a score from 1–10 based on detailed rubrics (e.g., wrong location is a severe error).
    - Candidates are ranked by score; the highest-scoring patch is submitted.
    - To reduce reviewer non-determinism, DEIBASE can “vote” by scoring a patch multiple times and averaging the scores (Figure 4).

- Why this design?
  - Evaluation can be easier than generation: LLMs may more reliably judge if a patch addresses the problem than they can reliably generate a correct fix from scratch (Section 3.3.3).
  - Separate “before” and “after” code reduces cognitive load versus unified diffs (Section 3.3.3).
  - Using Moatless’s relevant code spans saves tokens and focuses attention (Section 3.3.3).

- Metrics to quantify diversity and meta-selection (Section 4.1.2)
  - `Resolve rate`: fraction of issues a single system solves.
  - `Union@k`: number solved by any of k candidates (an “oracle” upper bound).
  - `Intersect@k`: number solved by all k candidates (a consistency lower bound).
  - `Average@k`: average over the k single-candidate results (random choice baseline).
  - `n@k`: performance when a reranker selects `n` candidates from `k` (DEIBASE reports `n=1`). If reranking is perfect, `1@k` equals `Union@k`.

## 4. Key Insights and Innovations
- Demonstration of substantial agent diversity (Section 4.2.1; Figure 3; Table 2)
  - Novelty: A systematic measurement of both inter-agent and intra-agent diversity in SWE agents using `Union@k`, `Intersect@k`, `Average@k`.
  - Significance: For 10 different agents, `Union@10 = 54.3%` while `Average@10 = 26.6%` (Table 2). This reveals enormous untapped potential if one can reliably pick the right candidate per issue.

- Meta-policy framing with a practical LLM reranker (Sections 3.3.2–3.3.3)
  - Novelty: Casting “pick the right agent output” as a CMDP meta-policy (Eq. 3) and instantiating it with an LLM-based code review committee that scores patches using structured, explanation-first prompting.
  - Significance: Converts heterogeneous agent outputs into a common evaluation substrate and operationalizes “diversity as strength.”

- Explanation-led scoring pipeline (Section 3.3.3; Table 3)
  - Novelty: Enforcing a chain of explanations (issue → context → location → fix → conflicts) before scoring.
  - Significance: Ablation shows explanations improve reranking quality, e.g., on the open agents group, 34.6% with explanations vs. 32.3% without (Table 3).

- Multi-vote reviewer to stabilize LLM judgments (Figure 4)
  - Incremental but effective: Averaging several LLM scoring passes improves stability and performance across settings; for Moatless runs, performance rises from roughly the Average@10 baseline (~16%) to ~26% with 10 votes (Figure 4).

## 5. Experimental Analysis
- Evaluation setup (Section 4.1)
  - Dataset: SWE-Bench Lite (300 real-world issues with hidden tests; Section 4.1.1; also explained in Section 3.1).
  - Agents:
    - Inter-agent diversity: 10 distinct agents with similar leaderboard resolve rates (26–31%). The order for `@k` accumulation is fixed (Appendix A.1 lists the agent order).
    - Intra-agent diversity: 10 repeated runs each for Agentless, Aider, and Moatless under identical parameters (Section 4.1.1).
  - Metrics: Resolve rate, Union@k, Intersect@k, Average@k, and 1@k (DEI) as described in Section 4.1.2.

- Main quantitative results
  - Diversity is large; potential gains are high (Section 4.2.1; Figure 3; Table 2):
    - 10 different agents:
      - Average@10 = 26.6%
      - Union@10 = 54.3% (oracle upper bound)
      - DEI 1@10 = 35.7% (+9.1 absolute over Average@10)
    - 10 runs of a single agent:
      - Agentless: Average@10 = 20.4%, Union@10 = 34.7%, DEI 1@10 = 26.0% (+5.6)
      - Aider: Average@10 = 21.7%, Union@10 = 38.0%, DEI 1@10 = 24.7% (+3.0)
      - Moatless: Average@10 = 15.9%, Union@10 = 35.3%, DEI 1@10 = 26.3% (+10.4)
    - Interpretation: Different agents provide more complementary strengths than repeated runs of a single agent—the gap between Average@k and Union@k is largest for the cross-agent group (Figure 3 top-left). DEI’s improvement (1@k − Average@k) grows with k but tends to plateau (Section 4.2.2).

  - Leaderboard-level comparisons (Table 1):
    - DEIBASE-1 (mixed top agents): 55.0% resolve rate, which is the highest reported on SWE-Bench Lite at the time of writing, surpassing strong closed-source systems (e.g., 50.6% Cosine Genie).
    - DEIBASE-2 (five closed-source agents): 37.0%, improving over the best member in the group (e.g., 31.3% Factory Code Droid).
    - DEIBASE-Open (only open-source candidates): 34.3%, a 25% relative improvement over the best individual open-source agent in the group (27.3% Agentless). This beats several closed-source competitors.

- Ablation studies and robustness checks
  - Effect of explanations (Table 3):
    - Open agents: 34.6% (with explanations) vs. 32.3% (without).
    - Agentless runs: 26.0% vs. 23.0%.
    - Aider runs: 24.6% vs. 23.3%.
    - Moatless runs: 25.6% vs. 25.3%.
    - Conclusion: Explanations consistently help, though the margin varies.
  - Effect of votes (Figure 4):
    - More reviewer votes (independent LLM scorings averaged) generally increase resolve rate. E.g., for Moatless runs, performance rises steadily toward ~26% at 10 votes versus an Average@10 baseline around 16%.
  - Ordering caveat (Section 4.1.2; Appendix A.1):
    - `@k` values depend on the order in which candidates are added; the paper fixes orders (e.g., random-shuffled chronological order for cross-agent experiments) to keep results interpretable.

- Do the experiments support the claims?
  - Yes, within scope. Figures 3–4 and Tables 1–3 provide converging evidence that:
    - Diversity is substantial (large Union@k − Average@k gaps).
    - DEI’s reranking is meaningfully better than random (1@k > Average@k across settings).
    - The approach scales to leaderboard-winning results (Table 1).

## 6. Limitations and Trade-offs
- Assumptions and design choices
  - The reviewer does not execute code or tests; it uses static analysis of “before/after” code plus provided context (Section 3.3.3). If runtime behavior is crucial, misjudgments are possible.
  - Relevant context comes from Moatless Tools’ spans (Section 3.3.3). If spans miss key code, the reviewer may score incorrectly. The paper notes potential gains from making spans patch-specific rather than issue-only.
  - DEIBASE is a reranker: it cannot produce new patches—only pick among candidates.

- Scope limitations
  - The meta-policy formalism (Eq. 3) allows per-context agent selection, but the implemented DEIBASE does not dynamically orchestrate agents during problem solving. It operates post hoc over completed patches (Section 3.3.3).
  - Results are on SWE-Bench Lite (300 issues). While representative, generalization to larger or different repositories is untested here.

- Computational and practical costs
  - Generating multiple candidate patches (from multiple agents or multiple runs) and scoring each with many votes can be costly in LLM tokens and wall-clock time (Figure 4 suggests up to 10 votes per candidate).
  - The best reported DEI systems use closed-source LLMs (e.g., gpt-4o) as reviewers (Table 1), which may limit reproducibility or increase cost for some users. The open-source agent group still relies on a closed-source reviewer.

- Performance plateaus and sensitivity
  - Gains from adding more candidates can saturate and even fluctuate slightly (Section 4.2.2). This suggests the current scoring rubric and prompting could be further optimized for large committees.
  - `@k` metric values are order-dependent; although the paper fixes ordering, alternative orders could yield slightly different curves (Section 4.1.2).

## 7. Implications and Future Directions
- Impact on the field
  - The work reframes “agent specialization” from a liability into an asset. By quantifying diversity and adding an effective meta-selector, it shows that multi-agent combinations can leapfrog single-agent ceilings (Table 2; Figure 3).
  - It strengthens the case that LLMs may evaluate code more reliably than they can generate correct fixes outright, at least in many cases (Section 3.3.3).

- Practical applications
  - As an LLM-based code review assistant that selects among multiple candidate patches from different tools/agents. This can be integrated into CI pipelines to triage candidate fixes before running expensive test suites.
  - As a leaderboard-level optimizer: teams can keep using their preferred agents and add DEI as a light-weight meta-layer to increase success without refactoring their systems (Table 1).

- Research directions
  - Dynamic, step-level meta-control: Move from post hoc reranking to real-time agent orchestration that selects tools/agents through a trajectory (closing the gap between Eq. 3’s vision and DEIBASE’s current implementation).
  - Richer evidence for scoring: Incorporate executable checks—compile, run tests, static analysis, or fault localization signals—so the reviewer bases scores on both static reasoning and runtime feedback.
  - Patch-specific retrieval: Generate context spans tailored to each candidate patch, not only the issue (Section 3.3.3 hints this could improve reviewer accuracy).
  - Confidence estimation and risk control: Calibrate reviewer scores and use them to decide when to abstain, request more candidates, or trigger additional agent runs.
  - Cost-aware committees: Optimize the number and type of candidates and reviewer votes to maximize performance per token or per minute (Figure 4 shows diminishing returns beyond a point).

Quoted highlights from the paper
- Diversity potential:
  > “For the ‘10 different agents’ setting, as k approaches 10, the distinct issues resolved are 2× the average number of issues resolved by a single agent in the group.” (Section 4.2.1; Figure 3; Table 2)
- DEI gains:
  > “A group of open-source SWE agents, with a maximum individual resolve rate of 27.3% on SWE-Bench Lite, can achieve a 34.3% resolve rate with DEI” (Abstract; Table 1).
- Explanation necessity:
  > “For all 4 settings we evaluated, DEIBASE with explanations performs slightly better than DEIBASE without explanations.” (Section 4.3; Table 3)

In sum, the paper’s central message—diversity empowers intelligence—holds up empirically: different agents solve different bugs, and an LLM reviewer that explains first and scores second can reliably pick better fixes than chance, pushing ensemble performance above the best individual agent.
