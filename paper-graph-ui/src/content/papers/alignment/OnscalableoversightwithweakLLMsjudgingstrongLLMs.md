# On scalable oversight with weak LLMs judging strong LLMs

**ArXiv:** [2407.04622](https://arxiv.org/abs/2407.04622)

## 🎯 Pitch

This paper provides the first broad, systematic benchmark of scalable oversight protocols, using weaker language models to judge stronger ones across nine diverse tasks ranging from extractive QA to mathematics and multimodal reasoning. It shows that debate between two strong LLMs enables less capable judges to supervise more accurately than one-agent consultancy and reduces the risk of amplifying incorrect outputs—but debate's advantage over simple question answering depends on the task and type of oversight gap. These results rigorously inform the path toward safe supervision of superhuman AI, highlighting both the promise and the limits of debate as a scalable alignment solution.

---

## 1. Executive Summary
This paper benchmarks scalable oversight protocols—ways for a weaker judge to supervise stronger AI systems—by running large language models (LLMs) as both agents and judges across nine tasks (extractive QA, closed QA, and multimodal). It finds that debate between two agents generally helps weak judges more than one‑agent consultancy and is less likely to amplify an agent’s wrong belief, though debate does not consistently outperform direct question answering baselines.

## 2. Context and Motivation
- Problem addressed
  - How can weaker evaluators (e.g., humans or weaker LLMs) reliably supervise stronger AI systems? This is the “scalable oversight” problem. 
  - The paper evaluates two families of protocols that might make weak oversight work:
    - `Debate`: two AIs argue opposite sides; a judge decides.
    - `Consultancy`: one AI answers and responds to judge’s questions.
- Why it matters
  - Misaligned or simply overconfident models can exploit weak oversight signals, leading to unsafe or incorrect behavior (Introduction, citing Denison et al. 2024). If oversight scales, it can reduce risks such as deceptive behavior by stronger models.
- Prior approaches and gaps
  - Most prior LLM debate studies focus on a single “extractive reading comprehension” dataset where debaters can quote a hidden article and the judge cannot read it (information asymmetry). This setup favors debate but does not reflect other capability gaps (math, logic, coding, multimodal) that matter for real oversight.
  - Previous results are mixed: some human‑debater studies saw little benefit; LLM‑based studies showed gains mostly on that one extractive dataset (Related Work; Section 1).
- Positioning of this work
  - Extends evaluation to nine datasets covering three gap types (Figure 1, Table 1; Appendix F):
    - `Extractive QA` with information asymmetry (QuALITY, BoolQ, GPQA‑Extractive).
    - `Closed QA` without extra sources (MMLU, GSM8KQA, PrOntoQA, TruthfulQA, GPQA).
    - `Multimodal` (MMMU).
  - Compares protocols in both “assigned‑role” and “open‑role” versions:
    - `Assigned`: systems are forced to argue a given answer (correct or incorrect).
    - `Open`: a system chooses which answer to argue for, mimicking training-time behavior.
  - Studies multiple judge strengths (Gemma 7B, GPT‑3.5, Gemini Pro 1.0, Gemini Pro 1.5) while typically using the strongest model (Gemini Pro 1.5) as the agent (Section 3.3, 4.1).

## 3. Technical Approach
Step‑by‑step design of the evaluation framework.

- Core entities
  - `Judge`: the weaker LLM deciding the final answer.
  - `Debaters`/`Consultant`: stronger LLM(s) producing arguments. Debaters argue opposite sides; the consultant defends a single answer.
  - `Information asymmetry`: in extractive QA tasks, debaters/consultant see the article; the judge does not (Section 3.1).
- Protocols benchmarked (Section 3.2; Figure 1)
  - Direct QA baselines (no agent help):
    - `QA without article`: judge answers from the question only (works for all tasks).
    - `QA with article`: judge sees the full article (extractive tasks only).
  - Assigned‑role oversight protocols:
    - `Consultancy`: consultant is randomly assigned either the correct or incorrect answer (50/50) and tries to convince a questioning judge for several rounds.
    - `Debate`: two debaters argue opposite answers for several rounds; the judge chooses.
  - Open‑role oversight protocols (more “training‑like”):
    - `Open consultancy`: consultant chooses which answer to defend (by matching the consultant’s own direct‑QA choice; Section 3.2.3).
    - `Open debate`: one debater—the `protagonist`—defends the answer that same model would pick in direct QA; the other (`antagonist`) gets the opposite side.
- Interaction mechanics
  - Rounds: main results use 3 rounds (Section 4.1); ablations also test 1 round (Figure A.2).
  - Turn style: simultaneous vs sequential (Section 3.2.2); ablated in Figure A.6.
  - Best‑of‑N: for debate, four sampled arguments per turn with a model‑based selector that picks “the most persuasive one” (Section 4.1 footnote; Figure A.3). Unlike prior work using log‑probabilities vs a dummy argument, selection is by model judgment among the actual candidates.
  - Chain‑of‑thought (CoT): agents produce hidden reasoning; only their final arguments are shown to the judge (Section 3.2).
- Passage verification tool for extractive tasks (Appendix G; Figure I.1–I.2)
  - Debaters/consultants can include `<passage>...</passage>` excerpts from the hidden article.
  - The system verifies exact string matches:
    - If verified: tags become `<v_passage>...</v_passage>`.
    - If not: tags become `<u_passage>...</u_passage>`.
  - Judges are told to trust only verified passages (Judge prompts, Appendix H.4.1/H.3.1).
- Tasks and how they’re made binary (Appendix F)
  - `QuALITY`, `BoolQ`: directly binary.
  - `GPQA-Extractive`: article is an expert explanation of the correct answer; agents must extract from it.
  - `MMLU`: binary subset formed by picking the correct option and one random wrong option.
  - `GSM8KQA`: create two executable Python solutions; the correct/incorrect answers come from code that does/doesn’t match the official answer (Appendix F.2).
  - `PrOntoQA`: construct incorrect variants by negating logical statements in the chain-of-thought (Appendix F.2).
  - `TruthfulQA`, `GPQA`: binaryized multiple‑choice.
  - `MMMU`: multimodal image+text questions; only Gemini Pro 1.5 supports it (Appendix F.3; Figure 2).
- Models and capability gaps (Section 3.3)
  - Agents: primarily `Gemini Pro 1.5` (strongest) as debaters/consultant.
  - Judges: `Gemma 7B`, `GPT‑3.5`, `Gemini Pro 1.0`, `Gemini Pro 1.5`. The judge is equal or weaker than the agents.
- Measurement and statistics
  - Metric: `judge accuracy` versus ground‑truth labels (Sections 3 and 4).
  - For open‑role: also measure `win‑rate` (how often the judge selects the side the protagonist/consultant advocated; Figure 3 top).
  - Significance tests: paired permutation tests (10,000 resamples) for accuracy differences (Appendix A; Figure A.1). Error bars are 95% CIs (Appendix D).
- Persuasiveness via Elo (Section 4.3; Appendix E)
  - Create 1‑turn simultaneous debates across five debater settings (Gemma 7B, GPT‑3.5, Gemini Pro 1.0, Gemini Pro 1.5, Gemini Pro 1.5 with Best‑of‑4).
  - Cross‑play 512 pairings and judge with Gemini Pro 1.5 to estimate an Elo skill for each debater.
  - Expected win probability uses the standard Elo formula:
    > Appendix E: expected win-rate of i against j is `1 / (1 + 10^((E_j - E_i)/500))`.
  - Compute aggregate Elo, plus separate Elo when assigned the correct or incorrect side.

## 4. Key Insights and Innovations
- Broad, capability‑diverse evaluation of scalable oversight (Figure 1; Table 1; Sections 3–4)
  - Novelty: moves beyond one reading-comprehension dataset to nine tasks covering information asymmetry (extractive), intrinsic reasoning gaps (math, logic, knowledge in closed QA), and multimodal reasoning.
  - Significance: reveals where debate helps and where it does not, which prior single‑task studies could not establish.
- Open‑role analysis that probes training‑time risks (Section 4.2; Figure 3)
  - Novelty: introduces `open debate` and evaluates `open consultancy` more comprehensively. In both, agents choose which side to argue for, revealing whether oversight amplifies an agent’s own mistaken belief.
  - Insight: when the agent is wrong, open debate helps the judge resist persuasion much more than open consultancy (Figure 3 bottom).
- Clear separation of information vs reasoning assistance (Sections 4.1 and A.1)
  - Insight: In extractive tasks, `QA with article`—giving the judge the source directly—beats debate/consultancy for all judge models (Figure 2). This shows that when the gap is “missing information,” direct access helps more than mediated arguments.
- Elo‑based persuasiveness linked to judge accuracy (Section 4.3; Figure 4)
  - Finding: higher debater Elo correlates with higher judge accuracy, especially on extractive tasks (Figure 4 bottom left). This supplies preliminary evidence that as agent skill scales, debate can deliver better oversight signals.

## 5. Experimental Analysis
- Evaluation setup and datasets
  - Nine tasks × 128 questions each ≈ 5 million model calls (Section 1).
  - Main config: 3 rounds, simultaneous turns, Best‑of‑4 for debaters with a model‑based selector; judges are zero‑shot (Section 4.1). Ablations vary turns, Best‑of‑N, few‑shot, CoT, turn order, and debate orderings (Figures A.2–A.8; B.2–B.8).
- Main results across protocols (Figure 2; Appendix A)
  - Debate vs consultancy (assigned‑role):
    > Figure 2 and Figure A.1: “Debate consistently outperforms consultancy across all tasks.”  
    This holds for extractive, closed, and multimodal task types (except that multimodal’s difference is less pronounced; see A.1).
  - Debate vs direct QA:
    - Extractive QA with asymmetry: 
      > Section 4.1 and Figure 2: Debate > QA without article; `QA with article` is best for all judges.
    - Closed QA: mixed results (Section 4.1).
      - With very weak judge (Gemma 7B): QA without article > Debate.
      - `GPT‑3.5` judge: Debate > QA without article with `p = 0.037` (Figure A.1).
      - `Gemini Pro 1.0` judge: No significant difference (`p = 0.1`).
      - `Gemini Pro 1.5` judge (same strength as debaters): Debate < QA without article.
    - Multimodal: Consultancy is significantly worse; Debate ≈ QA without article (not statistically significant; `p = 0.11`, Section 4.1).
  - Takeaway: Debate helps more when the gap is missing information (extractive) than when the gap is pure reasoning or knowledge skill (closed, multimodal).
- Open‑role findings (Section 4.2; Figure 3)
  - Frequency of choosing the correct side:
    > Figure 3 caption: Protagonist/consultant chooses correctly in 88% (extractive), 84% (closed), 71% (multimodal).
  - Win‑rate vs accuracy trade‑off:
    > Figure 3 top: Open consultancy yields higher or equal win‑rate than open debate, meaning judges agree with the consultant more often.
    - But higher win‑rate does not always mean higher judge accuracy.
  - Crucial robustness difference when the agent is wrong:
    > Figure 3 bottom: When the consultant picks the wrong answer, judges are often convinced in open consultancy (low accuracy, light blue bars).  
    > In open debate, when the protagonist is wrong, judge accuracy is much higher (light red bars).
  - Interpretation: Open debate reduces risk of amplifying agents’ mistakes compared to open consultancy.
- Ablations and robustness checks
  - Number of rounds: 1 vs 3 has no consistent effect (Figure A.2; B.2).
  - Best‑of‑N sampling for debaters: little to no consistent benefit (Figure A.3; B.3).
  - Few‑shot prompting for judges: no consistent gains; can cause failures for the weakest judge due to long contexts (Figure A.4; B.4 caption).
  - Judge chain‑of‑thought: neutral or harmful (Figure A.5; B.5). The authors hypothesize that weak judges’ own CoT may distract from stronger agents’ arguments.
  - Turn style: simultaneous vs sequential—no significant difference (Figure A.6; B.6).
  - Positional bias and order: judges do exhibit positional bias (Figure A.8), but evaluating both orderings does not change mean accuracy; randomizing answer position suffices (Figure A.7; B.7–B.8).
- Elo and scaling (Section 4.3; Figure 4; Appendix E)
  - Debater Elo separates models and shows a consistent advantage when assigned the correct side (Figure 4 top and middle).
  - Judge accuracy increases with debater Elo on extractive tasks (Figure 4 bottom left). On closed tasks, the correlation is weaker.
  - Additional plots with different judges and aggregation appear in Appendix E (Figures E.1–E.2).
- Failure analysis (Appendix C)
  - In `PrOntoQA`, most judged mistakes (11/12 reviewed) come from the judge’s own logical errors, not from irresistible deception by the dishonest debater.
  - In `TruthfulQA`, errors often stem from ambiguity that dishonest debaters exploit or from missing external facts; sometimes the judge is inattentive to key aspects.
- Overall assessment
  - The experiments are extensive and carefully controlled with statistical tests (Figure A.1) and ablations. The conclusions are well supported:
    - Debate is a better oversight protocol than consultancy when consultants might be assigned the wrong side or choose it incorrectly.
    - Debate’s benefits over direct QA are conditional: clear for information asymmetry, mixed otherwise.

## 6. Limitations and Trade-offs
- Inference‑only evaluation (Section 5: Conclusion; Limitations)
  - No training via these protocols (no self‑play or RL). Debate is judged as a one‑shot inference method, which may understate its potential when trained.
- Judge fine‑tuning mismatch (Section 1 and Conclusion)
  - Judges are instruction‑tuned and RLHF‑trained mostly on direct QA formats, not on evaluating debates. This likely favors the direct QA baselines.
- Agent alignment and helpfulness bias
  - Agent models are RLHF‑trained for honesty; dishonest roles may be artificially constrained (Limitations; also discussed relative to prior work).
- Open debate design
  - Only one debater is truly “open” (the protagonist); the other is auto‑assigned (Section 3.2.3). Both choosing freely might yield different dynamics (future work suggested in Section 4.2).
- Human relevance
  - LLM judges stand in for humans; real human judges may behave differently, especially at superhuman capability gaps (Limitations).
- Task scope and tools
  - The extractive setting relies on exact‑match passage verification, which may not reflect realistic evidence checking when humans supervise (Appendix F–G; Limitations).
- Computational costs
  - Large‑scale multi‑agent interactions (≈5M generations) imply non‑trivial runtime and cost; Best‑of‑N adds sample overhead without clear benefit (Figures A.3, E.1).

## 7. Implications and Future Directions
- What changes in the field
  - The work reframes debate as most promising when the oversight gap is primarily missing information and when avoiding error amplification is critical. It tempers expectations for debate vs direct QA on reasoning‑only tasks unless judges are trained to evaluate debates.
- Research enabled or suggested
  - Train debate and consultancy agents with the judge’s signal (self‑play / RL) and measure whether judge accuracy and debater Elo co‑improve (Conclusion; Future work).
  - Fine‑tune judges specifically for debate/consultancy evaluation; test whether this closes the gap vs direct QA (Section 1 caveats; Conclusion).
  - Human‑in‑the‑loop studies to validate LLM‑judge findings (Conclusion).
  - Explore richer scalable oversight protocols:
    - Debate with `cross‑examination` (questioning the opponent’s claims with tools), iterative amplification, or market‑based schemes (Future work; Related Work).
    - Open debate where both debaters can choose the same side (Section 4.2 suggestion).
    - Introduce tools (code execution, retrieval, multimodal tools) asymmetrically to model real weak‑strong gaps (Future work).
  - Robustness under distribution shift and against misaligned agents (Future work).
- Practical applications
  - Oversight for AI assistants that have private tool access or proprietary data: debate can reveal and stress‑test claims when the human (or weak judge) cannot directly access the source (extractive tasks illustrate this).
  - Safer preference‑learning pipelines: open debate is less likely than open consultancy to amplify an agent’s mistaken beliefs (Figure 3 bottom), which is crucial when using overseer feedback to train future systems.

> Key quantitative highlights to remember:
> - Debate > Consultancy across all task types in assigned‑role settings (Figure 2; Figure A.1).
> - In extractive QA, `QA with article` > Debate > `QA without article` (Figure 2).
> - In closed QA, Debate vs `QA without article` is mixed: depends on judge strength; only GPT‑3.5 shows a significant win for Debate (`p=0.037`, Figure A.1).
> - Open‑role: when the agent chooses incorrectly, judges are much less misled in open debate than in open consultancy (Figure 3 bottom).
> - Stronger debaters (higher Elo) modestly increase judge accuracy, especially for extractive tasks (Figure 4 bottom).
