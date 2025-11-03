# Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena

**ArXiv:** [2306.05685](https://arxiv.org/abs/2306.05685)

## 🎯 Pitch

This paper introduces a scalable and automated framework for evaluating chat-oriented large language models (LLMs) by leveraging a strong LLM (like GPT-4) as an impartial judge, and proposes two innovative benchmarks: MT-Bench (a suite of challenging, expert-written multi-turn prompts) and Chatbot Arena (a live, crowdsourced battle platform). Demonstrating that GPT-4’s judgments align with human preferences over 80% of the time, the work addresses key biases, offers practical mitigation strategies, and lays the groundwork for fast, explainable, and cost-effective evaluation methods—critical for advancing chatbot development and alignment with real user needs.

---

## 1. Executive Summary (2-3 sentences)
This paper introduces a scalable way to evaluate chat-oriented large language models (LLMs) by using a strong LLM as an automatic judge (“LLM-as-a-judge”) and by releasing two complementary benchmarks, `MT-Bench` (80 expert-written, multi-turn prompts) and `Chatbot Arena` (a live crowdsourced battle platform). Using GPT-4 as the judge, the method aligns closely with human preferences—over ~80% agreement—while diagnosing systematic biases and proposing mitigations (e.g., position swapping, reference-guided grading), enabling fast, explainable, and cost-effective evaluation of chatbots.

## 2. Context and Motivation
- Problem addressed
  - Modern chatbots are evaluated mostly on closed-ended, short-form tests (e.g., multiple-choice), which do not reflect how people use them in open-ended, multi-turn conversations. The paper targets the gap between “capability” benchmarks and “human preference” in realistic chat settings (Introduction; §2.1).
- Why it matters
  - Real users care about helpfulness, instruction-following, and conversational quality, not just factual recall or multiple-choice accuracy. Figure 1 shows a case where a fine-tuned chatbot provides more useful follow-up content than a strong base model, despite similar scores on traditional benchmarks (Introduction; Table 8).
- Shortcomings of prior approaches
  - Core-knowledge benchmarks (e.g., MMLU, HellaSwag) focus on short answers and do not capture conversational helpfulness (§2.1).
  - Instruction-following datasets (Flan, Self-Instruct, NaturalInstructions) broaden tasks but are still largely static and not multi-turn (§2.1).
  - Conversational benchmarks (e.g., CoQA) lack the diversity and difficulty to separate today’s top chatbots (§2.1).
- Positioning of this work
  - The paper proposes (i) two preference-centric evaluation settings—`MT-Bench` (curated, challenging multi-turn prompts across 8 categories) and `Chatbot Arena` (real-world pairwise battles), and (ii) a systematic study of `LLM-as-a-judge`, including its biases and fixes (§2; §3). It argues for a hybrid evaluation framework: combine capability benchmarks with preference-based evaluation judged by a strong LLM (Conclusion; §5).

## 3. Technical Approach
The approach has three intertwined components: benchmarks, LLM-judging protocols, and bias/robustness controls.

A) Benchmarks and data pipelines
- `MT-Bench` (§2.2; Table 1)
  - 80 multi-turn (two-turn) questions designed to stress instruction-following and conversational ability across 8 categories: writing, roleplay, extraction, reasoning, math, coding, STEM, humanities.
  - Each sample has two turns to test continuity and adherence. For example, a writing prompt followed by a style-constrained rewrite (Table 1).
  - 58 expert raters (mostly graduate students) provided ~3K controlled votes over model pairs and turns (§4.1; Appendix C.1).
- `Chatbot Arena` (§2.3; Appendix C.2)
  - A live web platform where users ask any question to two anonymous chatbots in parallel and vote for the better response; models are revealed after voting. Over ~30K votes collected in one month, with a 3K-vote sample used for evaluation (§4.1; Table 6).

B) LLM-as-a-judge: judging formats and prompts (§3.1; Appendix A)
- Pairwise comparison
  - The judge sees a prompt plus two answers and outputs which is better or tie; prompt emphasizes avoiding position/length biases (Figure 5).
- Single-answer grading
  - The judge assigns a 1–10 rating to one response; more scalable for many models, though it can be less discriminative at fine margins (Figure 6).
- Reference-guided grading
  - For domains like math/reasoning, the judge is given a reference solution (often generated independently by the judge first), then evaluates each answer relative to it (Figure 8, Figure 10).

C) Multi-turn judging design (§3.5)
- Pitfall: If each turn is judged in isolation, the judge can misrefer to the wrong prior response (Figure 16).
- Fix: Present the full A and B two-turn conversations in a single prompt so the judge can correctly focus on the second turn (Figure 9).

D) Diagnosing and mitigating judge biases (§3.3–§3.4)
- Position bias: Preference for the first-displayed answer (Table 2).
  - Mitigation: Swap A/B order and declare a winner only if consistent (conservative), or randomize positions at scale (aggressive) (§3.4).
  - Few-shot judging improves consistency (GPT-4 consistency rises from 65.0% to 77.5%; Table 12) but increases cost and may import new biases.
- Verbosity bias: Favoring longer answers even when redundant.
  - “Repetitive list” attack shows high failure for Claude-v1 and GPT-3.5 (91.3%) but much lower for GPT-4 (8.7%) (Table 3; Figure 12).
- Limited math/reasoning grading
  - Even strong judges can be steered by wrong answers (Figures 13–15).
  - Chain-of-thought (CoT) helps somewhat, but the judge can repeat the same error (Figure 15; Table 4 shows failures drop 14/20 → 6/20 with CoT).
  - Reference-guided grading helps most (14/20 → 3/20 failures; Table 4).
- Self-enhancement bias: A judge favoring its own model family
  - Mixed/unclear evidence; GPT-4 seems to favor itself by ~10% in win rate, Claude-v1 by ~25%, while GPT-3.5 does not (aggregate trends; §3.3, Figure 3b discussion).

E) Agreement metric and setups (§4.1; Appendix D.3)
- `Agreement` = probability that two randomly sampled judges (from two judge types) agree on a randomly sampled question.
- Two setups:
  - `S1`: includes non-tie, tie, and order-inconsistent votes; inconsistent counted as ties (random baseline 33%).
  - `S2`: only non-tie votes (random baseline 50%).

## 4. Key Insights and Innovations
- A practical pipeline for preference-based evaluation at scale
  - Novelty: Demonstrates that a strong LLM (GPT-4) can approximate human preference judgments across both curated and crowdsourced settings with high agreement (≥80% on non-tied votes) (Table 5, Table 6). This is more than a leaderboard—it is a method for scaling evaluation without excessive human labor (§4.2).
- Two complementary benchmarks tailored to preference judgments
  - `MT-Bench` isolates multi-turn instruction-following across diverse categories; `Chatbot Arena` captures “in-the-wild” usage diversity (Appendix C.2). The combination enables both controlled studies and real-world validation (§2.2–§2.3; §4).
- Systematic bias analysis of LLM-as-a-judge, with concrete mitigations
  - Position and verbosity biases are measured and addressed (Table 2, Table 3); math/reasoning grading is enhanced via reference-guided judging (Table 4).
- Multi-turn judging prompt design that reduces misreferencing
  - Presenting full conversations for both assistants in a single prompt meaningfully lowers misjudgment risk (Figure 9 vs. the failure example in Figure 16).

These are more than incremental tweaks: the paper establishes a methodology for using LLMs to proxy human preferences reliably, while making explicit and mitigating judge-specific biases.

## 5. Experimental Analysis
Evaluation design
- Models evaluated on MT-Bench: `GPT-4`, `GPT-3.5`, `Claude-v1`, `Vicuna-13B`, `Alpaca-13B`, `LLaMA-13B` (§4.1).
- Human judges
  - MT-Bench: 58 expert labelers (~3K votes) (Appendix C.1).
  - Chatbot Arena: 2114 unique IPs; 3K-vote sample for analysis (§4.1; Table 6).
- Judge variants: Pairwise (Figure 5), Single-answer grading (Figure 6), Reference-guided (Figures 8, 10), with bias controls like swapping (§3.1–§3.5).
- Metrics: agreement (S1 vs S2), average win rate, category-wise win rate.

Main quantitative results
- Agreement with humans (controlled MT-Bench)
  - > “The agreement under setup S2 (w/o tie) between GPT-4 and humans reaches 85%, which is even higher than the agreement among humans (81%).” (Table 5, First turn: G4-Pair vs Human 85% vs Human vs Human 81%; Second turn similar at 85% vs 82%).
  - GPT-4 single-answer grading also aligns well with GPT-4 pairwise and humans (Table 5).
- Agreement with humans (crowdsourced Arena)
  - > “G4 vs H: 87% (S2, non-ties)” (Table 6).
  - Non-tie agreements between GPT-4 and other LLM judges are ~94–96% (suggesting when they do commit to a non-tie, they often agree with GPT-4’s pick; Table 6 S2 row for G4 vs G3.5 and G4 vs C).
- Bias analyses
  - Position bias (Table 2): GPT-4 shows the highest consistency among judges (65–66%), but bias remains; Claude-v1 and GPT-3.5 are more biased toward the first position. The “rename” prompt reveals a name bias for Claude-v1.
  - Verbosity bias: “Repetitive list” attack succeeds 91.3% of the time on Claude-v1 and GPT-3.5, but only 8.7% on GPT-4 (Table 3; Figure 12).
  - Math/reasoning grading: Reference-guided reduces errors substantially (Table 4: failures drop from 14/20 default → 3/20 reference-guided).
- Multi-turn vs single-turn outcomes
  - Win-rate curves from LLM judges closely track human preferences across both MT-Bench and Arena (Figure 3; Figure 4).
  - Agreement rises with performance disparity: > “from 70% to nearly 100% as win-rate difference widens” (Figure 2).
- Category-wise differentiation (Table 7)
  - GPT-4 leads in most categories; GPT-3.5 close on math/coding overall win rates, but GPT-4 still outperforms GPT-3.5 in direct pairwise or single grading within those categories (Table 7 discussion).
- Benchmark complementarity with standardized tests (Table 8)
  - MT-Bench scores (GPT-4 single-answer grading) distinguish aligned chatbots (e.g., Vicuna) from base models even when standardized benchmarks (MMLU, TruthfulQA) do not shift as much. Example:
    - > `LLaMA-13B`: MMLU 47.0; MT-Bench 2.61
    - > `Vicuna-13B`: MMLU 52.1; MT-Bench 6.39
    - > `GPT-4`: MMLU 86.4; MT-Bench 8.99 (Table 8)

Ablations and robustness checks
- Few-shot judging increases consistency (GPT-4 65.0% → 77.5%) but may import bias and quadruple prompt cost (Table 12; §3.4).
- Multi-turn prompt design prevents misreferencing errors (Figure 16 vs Figure 9; §3.5).
- Additional position-bias slices by category and model-pair show bias shrinks when model quality difference is large (Table 10, Table 11).

Do the experiments support the claims?
- The combination of controlled (MT-Bench) and uncontrolled (Arena) human data, consistent agreement statistics (Tables 5–6), and bias stress tests (Tables 2–4) make a compelling case that GPT-4 can serve as a high-quality proxy for human preference judgments when using the prescribed mitigations. The paper is careful to show failure modes (Figures 13–15) and to quantify gains from mitigations.

## 6. Limitations and Trade-offs
- Scope limited to “helpfulness” (Discussion §6)
  - Safety (harmfulness, honesty) is largely out of scope; adapting prompts might extend to these axes, but the paper does not evaluate that empirically.
- Residual biases
  - Position bias persists even for GPT-4 (Table 2); verbosity bias exists (Table 3), though GPT-4 is more robust.
  - Self-enhancement bias is not conclusively measured due to confounds (§3.3).
- Reasoning/math judging remains delicate
  - Even with CoT or references, judges can be misled by context or propagate errors (Figures 13–15; Table 4).
- Cost and latency
  - Few-shot prompts (to improve consistency) increase cost (~4× longer prompts; §3.4). Pairwise scaling is quadratic in number of models, while single-answer grading may tie more often (§3.1).
- Data assumptions
  - Arena relies on crowdsourced inputs; while reflective of real usage, it inherits platform biases in user demographics and prompt distribution (Appendix C.2).
- Multi-turn fidelity
  - Prompt design matters; simplifying to turn-by-turn judging risks misreferencing (Figure 16).

## 7. Implications and Future Directions
- Field impact
  - Establishes `LLM-as-a-judge` as a credible, explainable, and scalable approximation of human preference judgments, with empirical grounding that rivals human–human agreement (Tables 5–6). This can accelerate research cycles and reduce evaluation cost.
- Hybrid evaluation norm
  - The results motivate a standard practice combining capability tests (e.g., MMLU) with preference-based evaluation using a strong LLM judge and bias mitigations (§5; Conclusion).
- Practical applications
  - Fast leaderboard updates; model selection and A/B testing; iterative alignment training (e.g., RLHF data curation); regression testing for conversational assistants.
- Research directions
  - Safety/ethics dimensions: adapt prompts/judging criteria to honesty and harmlessness (§6).
  - Better bias controls: more robust debiasing (beyond swapping and few-shot), measure and mitigate self-enhancement rigorously.
  - Stronger math/reasoning judges: integrate verified solvers or external tools into the reference-guided pipeline (Table 4 suggests large gains are possible).
  - Open-source judges: Fine-tuned open models can approximate GPT-4 judging quality at lower cost; early evidence shows a fine-tuned `Vicuna-13B` improves consistency from 11–16% to 65% and reaches 85.5% agreement on non-ties versus labels in a held-out test (Appendix F; Table 15).

> Bottom line: With `MT-Bench` and `Chatbot Arena` plus a carefully engineered `LLM-as-a-judge` pipeline (prompts, bias controls, and reference-guided judging), this work provides a practical blueprint for preference-based evaluation of chatbots that is fast, inexpensive, and—within stated limits—well-aligned with human judgments.
