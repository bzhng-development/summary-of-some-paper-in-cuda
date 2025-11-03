# Judging LLM‑as‑a‑Judge with MT‑Bench and Chatbot Arena

**ArXiv:** [2306.05685](https://arxiv.org/abs/2306.05685)
**Authors:** Lianmin Zheng, Wei‑Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric P. Xing, Hao Zhang, Joseph E. Gonzalez, Ion Stoica
**Institutions:** 

## 🎯 Pitch

This paper introduces 'LLM-as-a-judge,' leveraging GPT-4 to evaluate chat models on open-ended, multi-turn tasks with a new methodology and two benchmarks. By achieving over 80% agreement with human preferences, it offers a scalable and cost-effective alternative to human evaluation, advancing the development of more aligned and effective conversational AI systems.

---

## 1. Executive Summary (2-3 sentences)
This paper introduces LLM-as-a-judge: using a strong language model (notably `GPT-4`) to evaluate other chat models on open-ended, multi-turn tasks. It contributes two complementary, publicly available benchmarks—`MT-bench` (80 two-turn questions) and `Chatbot Arena` (crowdsourced pairwise battles)—and shows that a careful `GPT-4` judge matches human preferences with over 80% agreement while identifying and mitigating key judging biases.

## 2. Context and Motivation
- Problem addressed
  - Modern chat assistants are trained to align with human preferences (e.g., via RLHF), but standard benchmarks (like MMLU) largely test narrow, closed-ended capabilities and fail to reflect how humans experience chatbot quality in open, multi-turn conversations.
  - The paper highlights a mismatch: fine-tuned assistants can be preferred by users yet show little or no improvement on conventional benchmarks (see Figure 1 and Table 8).
- Why it matters
  - Practical: Product teams need rapid, scalable, and preference-aligned evaluations to ship better assistants. Human evaluation is costly and slow.
  - Scientific: The field lacks benchmarks that (1) capture instruction-following in multi-turn settings and (2) reflect user utility across diverse, open-ended tasks.
- Prior approaches and gaps
  - Existing benchmarks fall into:
    - Core knowledge (e.g., MMLU, HellaSwag): closed-ended, short answers; struggle to distinguish alignment quality (Section 2.1).
    - Instruction-following datasets (e.g., Flan, Self-Instruct): more diverse tasks but limited conversational depth (Section 2.1).
    - Conversational datasets (e.g., CoQA): often not diverse or challenging enough for current assistants (Section 2.1).
  - None provides a scalable way to measure human preference across open-ended, multi-turn interactions.
- Positioning
  - The paper proposes a new evaluation framework—use a strong, already-aligned LLM as a judge—and contributes two datasets designed to measure and validate agreement with humans (Sections 2–4). It also systematically studies biases in LLM judges and mitigations (Section 3).

## 3. Technical Approach
The paper presents both the evaluation artifacts (benchmarks) and the judging methodology.

- Benchmarks designed to elicit human preference signals
  - `MT-bench` (Section 2.2; Table 1)
    - A set of 80 carefully curated, two-turn prompts spanning eight categories: writing, roleplay, extraction, reasoning, math, coding, STEM knowledge, and humanities.
    - Each item has a first-turn prompt and a second-turn follow-up that tests instruction-following and conversational coherence (Table 1 shows examples).
  - `Chatbot Arena` (Section 2.3)
    - A live, crowdsourced platform for head-to-head model “battles” with anonymous models. Users chat with two models in parallel, then vote on the preferred response. Over 30K votes collected; 3K randomly sampled for analysis (Appendix C.2; Section 4.1).

- LLM-as-a-judge variations (Section 3.1)
  - `Pairwise comparison`: Provide the question and two answers; the judge picks A, B, or tie, with a short explanation (prompt in Figure 5).
  - `Single-answer grading`: Provide the question and one answer; the judge assigns a 1–10 score (Figure 6). This scales better (linear in number of models) but may blur subtle pairwise differences.
  - `Reference-guided grading`: Provide a “reference” solution (e.g., for math), then ask the judge to compare answers to it (Figure 8). This improves correctness when the judge might otherwise be misled.
  - Multi-turn judging: Present each full two-turn conversation inline to prevent context errors (Figure 9 for pairwise; Figure 10 for single-answer with references). The paper shows that splitting turns across prompts causes wrong cross-references (Figure 16).

- Biases identified and how they are measured (Section 3.3)
  - `Position bias`: Judges favor the first-position answer. Measured by swapping A/B and checking consistency (Table 2).
  - `Verbosity bias`: Judges prefer longer, repetitive answers even if not higher quality. Tested via a “repetitive list” attack that duplicates list items without adding information (Table 3; Figure 12).
  - `Self-enhancement bias`: A judge might prefer outputs from the same model family. Analyzed via win-rate shifts across judges (Figure 3b); evidence is suggestive but not conclusive.
  - Limited math/reasoning grading: Judges can be misled by the provided answers, even for problems they can solve in isolation (Figures 13–15).

- Mitigations (Section 3.4)
  - `Swapping positions`: Run the judge twice with A/B swapped; declare a win only if consistent; otherwise tie.
  - `Few-shot judge`: Add demonstrated judging examples to the prompt; increases GPT-4 consistency from 65.0% to 77.5% (Appendix D.2, Table 12), at higher cost and potential new biases.
  - `Chain-of-thought (CoT) judge`: Ask the judge to solve the problem first, then grade (Figure 7); reduces some failures but still vulnerable to being anchored by given answers (Figure 15).
  - `Reference-guided judge`: Generate a reference solution first and present it to the judge; math grading failure rate improves from 14/20 to 3/20 (Table 4).

- Experimental setup for agreement estimation (Section 4.1; Appendix D.3)
  - `Agreement` is defined as the probability that two randomly sampled judges (of specified types) agree on a randomly sampled question.
    - Setup S1 includes ties and inconsistent votes (random baseline ≈ 33%).
    - Setup S2 uses only non-tied votes (random baseline = 50%).
  - Human evaluations:
    - `MT-bench`: 58 expert labelers (~3K votes), each evaluated ≥20 random questions (Appendix C.1).
    - `Chatbot Arena`: 3K single-turn votes sampled from ~30K, 2114 unique IPs (Section 4.1).

## 4. Key Insights and Innovations
- Scalable, explainable evaluation via LLM judges
  - Novelty: Positions an aligned LLM as a surrogate human evaluator on open-ended, multi-turn tasks, with prompts that require a verdict and explanation (Figures 5–10).
  - Significance: Achieves human-level agreement while being fast and automatable. The explanations make judgments interpretable.

- Two complementary benchmarks that surface human preferences
  - `MT-bench`: Small, high-quality, two-turn tasks designed to probe instruction-following and category-specific skills (Table 1).
  - `Chatbot Arena`: A dynamic, in-the-wild setting that captures real user interactions and preferences at scale (Section 2.3; Appendix C.2).
  - Significance: Together they capture both controlled and naturalistic preferences, enabling robust validation of judges.

- Systematic bias analysis and practical mitigations
  - The paper does not assume judges are unbiased. It measures position and verbosity biases quantitatively (Tables 2–3), exposes math/reasoning failures (Table 4; Figures 13–15), and proposes actionable fixes (swapping, few-shot, CoT, references). This elevates the work from a proposal to a validated methodology.

- Hybrid evaluation perspective
  - Insight: Preference-based benchmarks and traditional capability benchmarks capture different axes of quality; both are needed (Table 8). This reframes “evaluation” as a composite of capabilities and human alignment.

## 5. Experimental Analysis
- Evaluation methodology
  - Datasets and models (Section 4.1)
    - `MT-bench`: 80 two-turn prompts; models evaluated include `GPT-4`, `GPT-3.5`, `Claude-v1`, `Vicuna-13B`, `Alpaca-13B`, `LLaMA-13B`.
    - `Chatbot Arena`: 3K votes sampled from ~30K (single-turn); models include the above plus `Vicuna-7B`, `Koala-13B`, `Dolly-12B`.
  - Judges and metrics
    - Judges: `GPT-4` (pairwise and single-answer), `GPT-3.5`, `Claude-v1`, and humans; agreement computed under S1 and S2 (Section 4.1; Appendix D.3).
    - Additional metrics: `win rate` (average head-to-head wins against other models; Figures 3–4).
  - Bias quantification setups
    - Position bias: generate two similar answers per question (via GPT-3.5 with temperature 0.7), swap order, and measure consistency (Table 2).
    - Verbosity bias: “repetitive list” attack on 23 answers with numbered lists (Table 3; Figure 12).
    - Math/reasoning failures: judge elementary problems; measure failure rates under different prompts (Table 4; Figures 13–15).

- Main quantitative findings
  - Human–LLM judge agreement
    - `MT-bench`: 
      - Pairwise `GPT-4` vs humans (S2, non-ties): 85% agreement on both turns (Table 5).
      - Human–human agreement (S2): 81–82% (Table 5). 
      - Quote:
        > “The agreement under setup S2 (w/o tie) between GPT-4 and humans reaches 85%, which is even higher than the agreement among humans (81%).” (Section 4.2; Table 5)
    - `Chatbot Arena`:
      - Pairwise `GPT-4` vs humans (S2): 87% agreement (Table 6).
      - `GPT-4` single-answer grading (S2) also aligns strongly with pairwise and humans (85–89% depending on comparator; Table 6).
    - Agreement strengthens when model quality gaps widen: 
      > “Agreement between GPT-4 and human progressively increases … from 70% to nearly 100% as win rate difference grows.” (Figure 2)

  - Bias magnitudes and defenses
    - Position bias (Table 2):
      - `GPT-4` shows the least bias but still only 65–66% consistent under default/rename prompts; many judges over-prefer the first answer (e.g., `Claude-v1`: 75% biased toward first, 24% consistency).
      - Few-shot examples increase `GPT-4` consistency to 77.5% (Appendix D.2, Table 12).
      - Swapping-based voting (conservative win only if consistent) is used in subsequent experiments.
    - Verbosity bias (Table 3):
      - “Repetitive list” attack succeeds against `Claude-v1` and `GPT-3.5` 91.3% of the time; `GPT-4` is much more robust (8.7% failure).
    - Math/reasoning grading (Table 4):
      - Failure rate on 10 math questions (20 judgments due to swaps): default 14/20, CoT 6/20, reference-guided 3/20. Reference guidance is most effective.

  - Model rankings and category breakdowns
    - Win-rate curves from LLM judges closely track human results on both `MT-bench` (Figure 3) and `Arena` (Figure 4).
    - On `MT-bench`, `GPT-4` dominates across categories; `GPT-3.5` is close in math/coding but behind in reasoning (Table 7). 
      - Example:
        > `GPT-4` win rates: STEM 76.6%, Humanities 72.2%; Reasoning 49.3%, Math 66.1%.  
        > `GPT-3.5` win rates: STEM 52.8%, Humanities 53.8%; Reasoning 32.6%, Math 63.8%. (Table 7)
    - Traditional vs preference benchmarks (Table 8):
      - Fine-tuning on dialogue data (e.g., `Vicuna`) greatly boosts `MT-bench` scores (e.g., `Vicuna-13B` 6.39) without necessarily raising MMLU to the same extent (52.1). 
      - Quote:
        > “No single benchmark can determine model quality… comprehensive evaluation is needed.” (Section 5; Table 8)

- Are the experiments convincing?
  - Yes, for the core claim that `GPT-4` can approximate human preferences: large agreement margins over random baselines (S2 random = 50%) on two distinct datasets; consistent model rankings across judges; careful bias analysis and mitigations.
  - The paper also probes judge failure modes with concrete adversarial and math tests, and reports both successes and shortcomings (Tables 2–4; Figures 11–16).

- Ablations, failure cases, robustness checks
  - Bias quantification and mitigations (position, verbosity).
  - Prompting ablations (few-shot vs zero-shot) with cost/benefit trade-offs (Appendix D.2).
  - Prompt design for multi-turn judging; show that presenting full conversations reduces reference errors (Figure 16 vs Figure 9/10).

## 6. Limitations and Trade-offs
- Reliance on a specific strong judge
  - Most results hinge on `GPT-4`. This introduces cost, API dependence, and potential model-specific biases (Section 6; Appendix F shows promising but not yet equivalent open-source judges).
- Bias mitigation is partial
  - Position and verbosity biases are reduced but not eliminated. Even `GPT-4` shows non-trivial position effects (Table 2); verbosity attacks still succeed sometimes (Table 3).
- Math/reasoning grading remains imperfect
  - CoT helps but can be anchored by provided answers (Figure 15). Reference-guided grading improves accuracy but adds complexity and compute (Table 4).
- Preference scope and safety
  - The paper focuses on “helpfulness” and aggregates multiple dimensions (accuracy, relevance, creativity) into one score (Section 6). It does not evaluate honesty or harm avoidance, though the methods could be adapted.
- Pairwise vs single-answer scaling
  - Pairwise evaluations scale quadratically with the number of models. Single-answer grading scales linearly but is coarser (more ties; Section 3.1, Section 4.2).
- Data and ecological validity
  - `MT-bench` is small by design (80 questions) and expert-labeled; `Arena` is broad but depends on who shows up to vote and what they ask. Both are valuable but have different biases.

## 7. Implications and Future Directions
- How this work shifts the field
  - Establishes LLM-as-a-judge as a credible, explainable proxy for human preference at scale. This enables rapid iteration on chat models without prohibitive human labeling costs.
  - Reframes evaluation as two axes: core capabilities (traditional benchmarks) and human alignment (preference benchmarks). The paper advocates a hybrid evaluation stack (Section 5; Table 8).

- Follow-up research enabled
  - Open-source judges trained to match human preferences with low bias (Appendix F shows initial success fine-tuning `Vicuna-13B` to reach S2 agreement of 85.5% vs 87% for `GPT-4` in one setting).
  - Robustness to adversarial styles (e.g., verbosity) and debiasing techniques beyond swapping/CoT/few-shot—potentially using verifiers or external tools for grounding.
  - Extending beyond helpfulness to evaluate safety, honesty, and other value dimensions with specialized rubrics or constrained prompts (Section 6).
  - Better math/reasoning judging with formal references, programmatic checkers, or hybrid human-in-the-loop for hard cases.

- Practical applications
  - Continuous, scalable model benchmarking and A/B testing for assistant development (`Chatbot Arena` already functions as a live platform).
  - Training reward models: LLM judge outputs (with explanations) can supervise preference models cheaply.
  - Model selection and deployment policies that optimize for user-perceived quality, not just accuracy on closed-ended tests.

> Core takeaway: With careful prompt design, bias checks, and simple mitigations (like position swapping and reference guidance), a strong LLM judge—especially `GPT-4`—can approximate human preferences on open-ended, multi-turn tasks with agreement on par with inter-human agreement (≈85% in non-tie settings; Tables 5–6). The accompanying `MT-bench` and `Chatbot Arena` provide complementary testbeds to track both controlled and in-the-wild performance.
