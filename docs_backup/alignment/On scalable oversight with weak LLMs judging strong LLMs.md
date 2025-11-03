# On scalable oversight with weak LLMs judging strong LLMs

**ArXiv:** [2407.04622](https://arxiv.org/abs/2407.04622)
**Authors:** Zachary Kenton, Noah Y. Siegel, János Kramár, Jonah Brown‑Cohen, Samuel Albanie, Jannis Bulian, Rishabh Agarwal, David Lindner, Yunhao Tang, Noah D. Goodman, Rohin Shah
**Institutions:** Google DeepMind

## 🎯 Pitch

This paper pioneers a comprehensive evaluation of 'scalable oversight' protocols, demonstrating that AI debates (two AI systems arguing) outperform consultancy (one AI advocating) across diverse tasks, notably excelling when judges lack access to key information. This innovation offers significant implications for AI safety, highlighting structured debate's potential to prevent weak oversight from reinforcing erroneous beliefs, thus enhancing the reliability of supervising advanced AI systems.

---

## 1. Executive Summary
This paper evaluates whether “scalable oversight” protocols let a weaker supervisor accurately judge work done by stronger AI systems. Using large language models (LLMs) as both agents and judges across nine datasets (text and multimodal), it compares debate (two AIs argue), consultancy (one AI argues), and direct question answering baselines. The core finding is that debate consistently beats consultancy across tasks, but its advantage over direct QA depends on the task: it helps when the judge lacks key information (extractive tasks) and is mixed otherwise (Figures 2–4).

## 2. Context and Motivation
- Problem addressed
  - How can humans (or weak models standing in for humans) reliably supervise systems that are stronger than they are? The paper studies this “scalable oversight” problem, where the goal is to design supervision protocols that remain accurate even when the supervisee is more capable (Introduction, p. 1).
- Why this matters
  - Practically: Human feedback is central to training aligned AI (e.g., reinforcement learning from human feedback, RLHF). If weak oversight can be exploited, models may learn undesirable behaviors (e.g., reward tampering) as shown in recent work (Introduction, p. 1).
  - Theoretically: Debate has connections to interactive proofs, suggesting that structured interaction among strong agents can help weaker judges verify complex claims (Introduction, p. 2).
- Prior approaches and shortcomings
  - Debate has mostly been tested on a single reading comprehension dataset (QuALITY) with artificial “information asymmetry” (judges can’t access the full article), limiting how broadly the conclusions generalize (Introduction, p. 2).
  - Consultancy (a single assistant arguing for one side) is common in RLHF-like settings but may amplify mistakes if the assistant is wrong (Section 3.2).
- This paper’s positioning
  - Broadens evaluation beyond a single dataset and beyond pure information-asymmetry scenarios: adds math, coding, logic, and multimodal tasks (Figure 1; Section 3.1; Table 1).
  - Introduces “open-role” variants (open consultancy and open debate) to analyze whether protocols amplify an agent’s own mistakes (Section 3.2.3; Figure 3).

Key terms (defined when first used):
- `Scalable oversight`: training or evaluation setups where a weaker supervisor can still provide accurate judgments about stronger systems.
- `Debate`: two AIs advocate opposing answers; a judge chooses which side is correct (Section 3.2.2).
- `Consultancy`: one AI argues for a given answer; the judge asks questions and then decides (Section 3.2.2).
- `Open` variants: the AI chooses which answer to argue for (Section 3.2.3), rather than being assigned.
- `Information asymmetry`: the agent has access to information (e.g., an article) that the judge does not (Figure 1; Section 3.1).
- `Elo rating`: a standard skill-rating system from games; here it quantifies how persuasive a debater is via cross-play win rates (Appendix E).

## 3. Technical Approach
The study is empirical and large-scale (~5 million generation calls across experiments; Section 1, p. 3). It varies three axes: tasks, oversight protocols, and capability gaps between judges and agents.

A. Tasks (Section 3.1; Table 1; Figure 1)
- Three categories, converted to binary-choice format (correct vs. one randomly selected incorrect option).
  1) `Extractive QA` with information asymmetry
     - Judge does not see the article; agents can quote it.
     - Datasets: QuALITY, BoolQ, GPQA-extractive (Appendix F.1).
     - A “passage verification tool” marks quoted text as verified `<v_passage>` if it exactly matches the article, otherwise `<u_passage>` (Appendix G; Figure I.1–I.2).
  2) `Closed QA` without information asymmetry
     - No article; both judge and agents see the same prompt (Appendix F.2).
     - Datasets: MMLU, GSM8KQA (binary variant using code-generated candidate solutions), PrOntoQA (synthetic logic), TruthfulQA (adversarial truthfulness), GPQA (very hard science).
  3) `Multimodal` without information asymmetry
     - MMMU (images + text; Appendix F.3).
- Each dataset contributes 128 questions (Section 1, p. 3); only Gemini Pro 1.5 supports images.

B. Protocols (Section 3.2; Figure 1)
- Direct QA baselines
  - `QA without article`: judge answers with no assistance; for extractive tasks, the judge also lacks the article (Section 3.2.1).
  - `QA with article`: only for extractive tasks; the judge receives the full article (Section 3.2.1).
- Assigned-role scalable oversight protocols (Section 3.2.2)
  - `Consultancy`: one consultant is assigned an answer (correct or incorrect with 50/50 probability) and argues for it over several rounds; the judge can ask questions.
  - `Debate`: two debaters are assigned opposing answers and argue over several rounds; the judge sees the transcript and decides.
  - Turn structure: default 3 rounds; “simultaneous” (both submit each round without seeing current-round opponent’s argument) vs “sequential” (the second debater can respond to the first in the same round) (Section 3.2.2).
  - Chain-of-thought (“Thinking”) is private to debaters/consultants; only “Argument” is shown to judges (Section 3.2).
- Open-role variants (Section 3.2.3)
  - `Open consultancy`: the consultant argues for the answer it would choose under direct QA; implemented by filtering assigned-role runs where the assignment matches the model’s own preferred answer.
  - `Open debate`: one “protagonist” debater is the one whose assigned answer matches its own direct QA choice; the other (antagonist) is assigned the opposite. This reuses existing assigned-role debates.

C. Models and capability gaps (Section 3.3)
- Judges: `Gemma7B (V1, instruction-tuned)`, `GPT-3.5`, `Gemini Pro 1.0`, `Gemini Pro 1.5` (the multimodal model).
- Agents: Mostly `Gemini Pro 1.5` for consultancy/debate in the main experiments; later varied to compute Elo (Section 4.3).
- Design rationale: vary the judge to be weaker than the agent to emulate human supervisors overseeing more capable systems (Section 3.3).

D. Default experimental settings (Section 4.1)
- Assigned-role main comparison (Figure 2): 3Rounds; debate uses “best-of-4” sampling for each debater’s turn, where the debater model itself selects the most persuasive of 4 independently generated candidates (footnote 1 on p. 8). Judges are zero-shot prompted.
- Statistics: 95% confidence intervals and paired permutation tests for significance (Appendix D; Figure A.1).
- Ablations: number of turns, best-of-N, few-shot judges, judge chain-of-thought (CoT), debate turn style (simultaneous vs sequential), and positional bias via swapping answer order (Appendix A; Figures A.2–A.8).

E. Persuasiveness via Elo (Section 4.3; Appendix E)
- Generate 1-turn simultaneous debates across five debater settings: `Gemma7B`, `GPT-3.5`, `Gemini Pro 1.0`, `Gemini Pro 1.5 (BoN=1)`, `Gemini Pro 1.5 (BoN=4)`.
- Cross-play 512 pairings (balanced across datasets), judged by `Gemini Pro 1.5`.
- Fit Elo by maximum likelihood to match observed win rates; report separate Elo when assigned the correct vs incorrect side, plus aggregate Elo (Figure 4, Appendix E).

## 4. Key Insights and Innovations
1) Multi-capability evaluation of scalable oversight
- What’s new: moves beyond the standard QuALITY reading comprehension setting to include logic (PrOntoQA), math/coding (GSM8KQA), very hard science questions (GPQA), truthfulness (TruthfulQA), and multimodal MMMU (Table 1; Section 3.1).
- Why it matters: oversight must work across capability gaps that are not only about hidden information (Introduction, p. 2). This study tests whether debate/consultancy scale across qualitatively different asymmetries.

2) Open-role protocols to test amplification of model mistakes
- What’s new: `open consultancy` and `open debate` analyze whether a weak judge will be persuaded by a strong model’s own (possibly wrong) choice (Section 3.2.3).
- Why it matters: During training, if the model prefers the wrong answer, a weak supervisor may reinforce it. Open-role analysis surfaces that risk directly (Figure 3).

3) Clear empirical picture of debate vs. consultancy and vs. direct QA
- Novelty: Provides cross-task evidence that debate improves over consultancy, but its advantage over direct QA depends on information asymmetry and relative model strengths (Figure 2; Section 4.1).
- Significance: Helps calibrate expectations—debate is not a silver bullet across all settings, but it is reliably better than single-consultant protocols, and it can help when judges lack key information.

4) Scaling signal via Elo is positive but modest
- What’s new: Elo analysis across debaters indicates that stronger debaters yield higher judge accuracy—more clearly on extractive tasks—but the effect is smaller and less consistent than prior single-dataset studies (Section 4.3; Figure 4).
- Why it matters: Debate’s promise for scalable oversight hinges on judge accuracy improving as agent capabilities scale; this work finds such a trend, but only weakly.

## 5. Experimental Analysis
A. Evaluation methodology (Sections 3–4)
- Datasets: Nine datasets across three categories (Table 1; Section 3.1).
- Metrics:
  - `Judge accuracy`: whether the judge picked the ground-truth option.
  - `Win rate` (open-role): fraction of cases where the judge chose the consultant’s/protagonist’s side (Figure 3, top).
  - `Elo rating` for debaters (Section 4.3; Appendix E).
- Setup:
  - Assigned-role main study uses `Gemini Pro 1.5` as agent and varies judge from `Gemma7B` up to `Gemini Pro 1.5` (Figure 2).
  - Open-role analysis re-labels a subset of assigned-role runs based on agents’ own QA choices (Section 3.2.3).
  - Significance via paired permutation test (Figure A.1); 95% CIs (Appendix D).

B. Main quantitative results
- Debate vs. Consultancy (assigned-role)
  - Result: “Debate consistently outperforms consultancy across all tasks” (Summary list item 1, p. 3; Figure 2). The advantage is statistically significant in most judge–task combinations (Figure A.1).
- Debate vs. Direct QA
  - Extractive QA (with information asymmetry):
    - `QA with article` is best when available (judge sees the article), outperforming all other protocols (Figure 2; Section 4.1, “Extractive tasks”). This is expected because the weak judge receives the full information.
    - Debate > `QA without article` for most judges—except the weakest judge (`Gemma7B`) where debate ≈ QA without article (Section 4.1).
  - Closed QA (no information asymmetry):
    - Debate > consultancy again (Figure 2).
    - Debate vs `QA without article` is mixed and judge-dependent:
      - For `GPT-3.5`, debate does better with p = 0.037 (Section 4.1).
      - For `Gemini Pro 1.0`, difference is not significant (p = 0.1).
      - For `Gemini Pro 1.5` (judge = debaters), debate underperforms direct QA (Section 4.1; Figure 2).
      - For `Gemma7B`, direct QA > debate.
  - Multimodal:
    - Consultancy is clearly weaker; debate and direct QA are statistically similar (p = 0.11; Section 4.1; Figure 2).
- Open-role protocols (Figure 3)
  - Consultant/protagonist correctness frequencies:
    > “The consultant/protagonist chooses the correct answer in 88% (extractive), 84% (closed), and 71% (multimodal) of questions” (Figure 3 caption).
  - Win rate vs judge accuracy:
    - Open consultancy yields higher (or equal) win rates than open debate—judges tend to agree more with a lone consultant (Figure 3, top). But higher win rate doesn’t always mean higher accuracy.
    - When the model is correct:
      - Open consultancy can slightly outperform open debate (especially extractive tasks; Figure 3, bottom).
    - When the model is incorrect:
      - Open debate substantially reduces the judge’s tendency to be convinced by the wrong answer compared to open consultancy (Figure 3, bottom). This means open debate is less likely to amplify wrong beliefs.
- Persuasiveness and scaling via Elo (Section 4.3; Figure 4)
  - Being assigned the correct side confers an advantage (correct-Elo > incorrect-Elo across models; Figure 4, top/middle).
  - Stronger debaters (higher aggregate Elo) correlate with higher judge accuracy on extractive tasks (Figure 4, bottom-left). On closed tasks, correlations are weaker and the range is narrower (Figure 4, bottom-right).
  - Best-of-4 increases debater Elo but does not reliably increase judge accuracy (Figure A.3 vs Figure 4), suggesting limited utility of inference-time selection alone.
- Ablations and robustness checks (Appendix A)
  - Number of rounds: 3 vs 1 shows no consistent benefit (Figure A.2).
  - Best-of-N: little to no consistent accuracy benefit (Figure A.3).
  - Few-shot judges: no consistent improvements (Figure A.4).
  - Judge CoT: often harms or has no effect (Figure A.5). Authors hypothesize judges may ignore stronger agents’ reasoning and rely on their own weaker chain-of-thought (Section 4.1).
  - Turn style (sequential vs simultaneous): no significant difference (Figure A.6).
  - Positional bias: judges show bias toward answer position, especially in debate, but evaluating both orders reduces variance not mean accuracy (Figures A.7–A.8). Randomizing answer order is usually sufficient.
- Failure analysis (Appendix C)
  - On PrOntoQA, 11/12 judge errors (with `Gemini Pro 1.5` judge) arise from the judge’s own logical mistakes; on TruthfulQA, many errors stem from question ambiguity exploited by the dishonest side (Appendix C).

C. Do the experiments support the claims?
- Support for “debate > consultancy” is strong and consistent (Figure 2; Figure A.1).
- Support for “debate beats direct QA” is conditional:
  - Clear under information asymmetry when the judge lacks the article (extractive tasks): debate > QA without article (Figure 2).
  - Mixed when there is no information asymmetry: sometimes debate helps, sometimes it doesn’t, and for a very strong judge debate can even hurt (Section 4.1).
- Open-role findings convincingly show that open debate is less prone than open consultancy to amplifying wrong model beliefs (Figure 3).
- Scaling signal via Elo is present but modest (Figure 4).

## 6. Limitations and Trade-offs
- Inference-only evaluation
  - The study measures judgment accuracy without training debaters via self-play or judges via specialized supervision for debate. The central promise of scalable oversight is about training dynamics; this work is a proxy (Section 5, “Limitations”).
- Judge fine-tuning bias
  - Judges may be better trained for direct QA than for judging debates, potentially favoring direct QA in closed tasks (Section 1, caveat on p. 4).
- RLHF-tuned agents
  - The debaters/consultants have been aligned for helpfulness/honesty via RLHF, which may disadvantage the “dishonest” side in assigned-role settings and blur conclusions about worst cases (Section 5, “Limitations”).
- Quote verification tool realism
  - The extractive tasks rely on automated passage verification (`<v_passage>`/`<u_passage>`), which might not reflect real-world human oversight contexts (Appendix F.1; Appendix G).
- Self-preference bias
  - Judges from the Gemini family may underrate GPT-3.5 debaters, hinting at model-family preference effects (Section 4.3).
- Mixed benefit over direct QA
  - Where there is no information asymmetry and the judge is already strong, debate provides limited or negative marginal value (Figure 2, closed tasks; Section 4.1).
- Computational cost
  - Multi-round debate or consultancy has higher inference cost than direct QA; ablations show more rounds and best-of-N do not consistently improve accuracy (Figures A.2–A.3).

## 7. Implications and Future Directions
- How it changes the landscape
  - Provides a comprehensive, multi-domain picture of when debate helps weak judges and when it doesn’t. It firmly establishes that “debate > consultancy” across settings (Figure 2), while challenging the idea that debate universally improves over direct QA.
- Practical takeaways
  - If a weak supervisor cannot access relevant information (e.g., source documents), structured debate among stronger agents can improve their decisions (extractive tasks; Figure 2).
  - When using a single-advocate (consultancy) approach in training, beware amplification of wrong beliefs—open consultancy shows judges are easily persuaded when the consultant is incorrect (Figure 3, bottom).
  - For safety training, open debate is preferable to open consultancy because it reduces the chance of reinforcing wrong answers (Figure 3).
- Research directions
  - Train for debate: Move beyond inference-only; train debaters with self-play and judges with supervised or RL objectives tailored to judging (Section 5 “Future work”).
  - Judge fine-tuning: Build judges specialized for debate adjudication; test whether this closes the gap with direct QA especially in closed tasks (Introduction caveat on p. 4).
  - Protocol variants: Explore cross-examination, iterated amplification, and settings where both debaters freely choose positions (Section 5 “Future work”).
  - New asymmetries: Tool use, code execution, retrieval, and modality access asymmetries to mirror real oversight scenarios (Section 5 “Future work”).
  - Human studies: Replace LLM judges with humans to validate whether findings transfer and to quantify cognitive load and reliability (Section 5 “Limitations/Future work”).
  - Robustness: Stress-test debate under distribution shift and with misaligned debaters (Section 5 “Future work”).

> Bottom line (Figures 2–4): Debate reliably improves over consultancy and can help when judges lack information, but its advantage over direct QA is conditional. As a scalable oversight mechanism, debate shows promise—especially in configurations that reduce the amplification of wrong model beliefs (Figure 3)—yet realizing its full benefit likely requires training judges and debaters specifically for the debate setting.
