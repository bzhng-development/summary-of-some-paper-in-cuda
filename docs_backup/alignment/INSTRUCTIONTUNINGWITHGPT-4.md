# INSTRUCTION TUNING WITH GPT-4

**ArXiv:** [2304.03277](https://arxiv.org/abs/2304.03277)

## 🎯 Pitch

This paper pioneers the use of GPT-4 as a high-quality generator of both instruction-following data and feedback for fine-tuning open-source language models, specifically LLaMA. By leveraging GPT-4 to produce and evaluate 52K English and Chinese instruction–response pairs, the authors demonstrate significant boosts in zero-shot performance, showing that open-source models can reach quality levels competitive with commercial systems—greatly advancing scalable alignment and generalization in multilingual AI assistants.

---

## 1. Executive Summary
This paper shows how to use `GPT-4` as an automatic data generator and judge to instruction-tune open-source language models, specifically `LLaMA` 7B. By replacing earlier GPT‑3.5–generated data with 52K `GPT-4`–generated instruction–response pairs (and adding Chinese variants plus GPT‑4–produced feedback), the tuned models achieve stronger zero-shot performance on unseen tasks and competitive quality relative to larger or proprietary systems, as evidenced by human and GPT‑4–based evaluations (Sections 2–4; Figures 3–5).

## 2. Context and Motivation
- Problem addressed
  - Building language models that reliably follow natural-language instructions without expensive human-written data or feedback. The work targets scalable “instruction tuning”—supervised fine-tuning on instruction–response pairs—using machine-generated data (Abstract; Section 1).
- Why it matters
  - Instruction-following ability underpins practical assistants (question answering, writing, coding). High-quality supervision improves zero-shot generalization (i.e., performing new tasks without task-specific training), making models more useful in the real world (Introduction).
- Prior approaches and gaps
  - Self-Instruct methods bootstrap training data using a strong “teacher” model (e.g., GPT‑3.5) to produce instructions and responses (Wang et al., 2022a). Open-source models like Alpaca and Vicuna rely on GPT‑3.5 outputs or ShareGPT logs, but these can be limited in response quality and breadth, and are costly to extend to cross-lingual settings or to gather comparison data for reward modeling (Introduction; Section 3.2).
- Positioning
  - This is, to the authors’ knowledge, the first attempt to use `GPT‑4` both as:
    - A data generator for instruction–response pairs in English and Chinese (Section 2; Algorithm 1).
    - A feedback source to score and compare model outputs for reward modeling and evaluation (Section 2 “Comparison Data”; Section 3.2; Figure 2).
  - The paper provides a controlled comparison: instruction-tuning `LLaMA` 7B on the same 52K instructions as Alpaca but replacing the GPT‑3.5 outputs with GPT‑4 outputs, enabling direct measurement of the teacher swap (Section 2; Section 3.1).

## 3. Technical Approach
Step-by-step pipeline

- Data generation (Section 2; Algorithm 1)
  - Base instruction set: reuse the 52K unique instructions from Alpaca (English) to ensure an apples-to-apples test of teacher quality (GPT‑3.5 vs GPT‑4).
  - Prompting: two templates handle cases with and without extra input context. The unified template is:
    - “Below is an instruction … Write a response that appropriately completes the request.”
    - It embeds `### Instruction`, optional `### Input`, and expects `### Response:` (Algorithm 1, lines 3–10).
  - GPT‑4 call settings:
    - `model="gpt-4"`, `temperature=1.0`, `top_p=1.0`, `max_tokens=512` (Algorithm 1, lines 12–18).
    - Temperature and top‑p allow diverse responses; `max_tokens=512` caps response length.
  - Chinese data: Translate the 52K instructions to Chinese using ChatGPT, then ask GPT‑4 to respond in Chinese, producing a parallel Chinese instruction-following set (Section 2, item (2)).
  - Comparison data for feedback: For each prompt, collect responses from three systems (GPT‑4, GPT‑3.5, OPT‑IML) and have GPT‑4 rate them on a 1–10 scale and provide pairwise comparisons, forming training data for a reward model (Section 2, item (3); Figure 2).
  - Unnatural Instructions test: Generate GPT‑4 answers on a separate benchmark of 68K synthetic instruction–response triplets to measure alignment on “unusual” instructions (Section 2, item (4)).

- Instruction-tuning (Section 3.1)
  - Models trained
    - `LLaMA-GPT4` (7B) on the 52K English instruction–response pairs from GPT‑4.
    - `LLaMA-GPT4-CN` (7B) on the 52K Chinese instruction–response pairs from GPT‑4.
  - Training schedule: follows Alpaca’s setup to isolate the effect of switching to GPT‑4 data (Section 3.1).

- Reward modeling (Section 3.2; Figure 2)
  - Purpose: estimate user preference for responses to enable ranking or future RLHF.
  - Data: From GPT‑4’s 1–10 ratings of multiple responses per prompt, convert to pairwise preferences (higher‑scored `y_h` vs lower‑scored `y_l`), producing many training pairs per prompt.
  - Model: `OPT 1.3B` used as the reward model `r_θ`.
  - Loss: pairwise logistic preference loss encourages `r_θ(x, y_h)` > `r_θ(x, y_l)`:
    - In words: increase the score gap so that preferred responses receive higher scalar rewards.
    - In notation (Section 3.2): minimize `log(σ(r_θ(x, y_h) − r_θ(x, y_l)))`, where `σ` is the sigmoid.
  - Use in this paper: only for re-ranking multiple decoded samples per prompt during evaluation; no reinforcement learning is applied to the base model here (Section 4.3; Conclusions).

- Why these choices?
  - Reusing Alpaca’s 52K instructions isolates the effect of teacher quality (GPT‑3.5 vs GPT‑4) without confounding from instruction design or data size (Section 2).
  - GPT‑4 as both generator and judge lowers the cost of high-quality supervision and preference data, addressing a key bottleneck in RLHF pipelines (Section 2; Section 3.2).
  - Chinese translation plus GPT‑4 responses test cross-lingual generalization and yield a ready-made Chinese instruction-tuned model (Section 2, item (2)).

- Data characteristics (Figure 1)
  - The paper probes stylistic/content differences between GPT‑3.5 and GPT‑4 outputs by extracting root verb–direct object noun pairs from each response.
    - Unique pairs: GPT‑4 (5,229) vs GPT‑3.5 (6,133) (Figure 1c).
    - GPT‑4 tends to produce longer responses on average, but GPT‑3.5 shows a longer tail in length distribution due to the Alpaca team’s iterative deduplication process not used here (Figure 1d; Section 2 “Data Statistics”).

## 4. Key Insights and Innovations
- Using GPT‑4 as a high-quality teacher for instruction tuning
  - What’s new: swap GPT‑3.5 teacher outputs with GPT‑4’s on the same 52K Alpaca instructions, producing English and Chinese datasets (Section 2).
  - Why it matters: This isolates the effect of teacher quality and shows notable gains in zero-shot helpfulness and competitive performance against stronger baselines (Figure 3a; Figure 4c–d).

- GPT‑4–generated comparison data to train a reward model cheaply
  - What’s new: GPT‑4 acts as an automatic rater and comparator across responses from multiple models (GPT‑4, GPT‑3.5, OPT‑IML), producing large-scale preference data (Section 2, item (3); Figure 2).
  - Why it matters: Preference data is expensive when collected from humans; this enables training a working reward model (`OPT` 1.3B) that aligns with GPT‑4 judgments and improves decoding via re-ranking (Section 3.2; Figure 4a–b).

- Cross-lingual instruction tuning with Chinese data
  - What’s new: A parallel 52K Chinese instruction-following dataset and a Chinese-tuned model `LLaMA-GPT4-CN` (Section 3.1; Section 4.3 Figure 5c).
  - Why it matters: Demonstrates that the pipeline extends beyond English, and that a Chinese-tuned model substantially improves over an English-tuned model when evaluated in Chinese (Figure 5c: 64% vs 35% relative score vs GPT‑4).

- Transparent evaluation with both humans and GPT‑4
  - What’s new: Combine human HHH evaluation (helpful, honest, harmless) on user-oriented tasks with GPT‑4 automatic pairwise scoring on challenging prompts (Section 4.2; Section 4.3).
  - Why it matters: Provides triangulated evidence. Human judges prefer the GPT‑4–tuned model over the GPT‑3.5–tuned Alpaca for helpfulness (Figure 3a), and GPT‑4–based evaluation ranks the GPT‑4–tuned model above Alpaca and raw LLaMA (Figure 4c–d).

These are primarily methodological and empirical innovations (stronger teacher, automated feedback, cross-lingual extension) rather than theoretical advances.

## 5. Experimental Analysis
- Datasets and evaluation setup (Section 4.1)
  - Human evaluation: 252 “User-Oriented-Instructions-252” prompts covering practical applications (writing, coding, etc.). MTurk interface enforces Helpful/Honest/Harmless comparisons between two models’ outputs (Appendix A.1; Figure 7).
  - GPT‑4 automatic evaluation: 80 challenging prompts synthesized in the Vicuna evaluation set (Section 4.1; Figure 4). GPT‑4 assigns 1–10 scores to each model’s output in pairwise comparisons; the total over 80 items (max 800) is used. Results are reported relative to the opponent model’s total score (Figure captions).
  - Unnatural Instructions: 68,478 instruction–response triplets; 9K used for ROUGE‑L analysis, grouped by ground-truth response length (Figure 6).

- Baselines and comparators (Figures 3–5)
  - Open models: `LLaMA` (13B), `Alpaca` (13B, GPT‑3.5 tuned), `Vicuna` (13B).
  - Commercial systems: `ChatGPT`, `Bard`, `GPT‑4`.
  - This work: `LLaMA-GPT4` (7B), and a re-ranked variant `LLaMA-GPT4 (R1)` using the reward model’s top-1 selection from five decoded samples (Figure 4a–b).

- Human HHH evaluation results (Figure 3)
  - LLaMA‑GPT4 vs Alpaca (Figure 3a; 252 prompts):
    - Helpfulness:
      > LLaMA‑GPT4 wins 54.12%; Alpaca wins 19.74%; ties 26.14%.
    - Honesty:
      > LLaMA‑GPT4 31.39%; Alpaca 25.99%; ties 42.61%.
    - Harmlessness:
      > Alpaca 25.43%; LLaMA‑GPT4 16.48%; ties 58.10%.
    - Takeaway: Switching to GPT‑4 responses notably improves perceived helpfulness and slightly honesty, while harmlessness is similar overall (high ties) with a slight edge to Alpaca.
  - LLaMA‑GPT4 vs GPT‑4 (Figure 3b):
    - Helpfulness:
      > GPT‑4 44.11%; LLaMA‑GPT4 42.78%; ties 13.11%.
    - Honesty:
      > LLaMA‑GPT4 37.88%; GPT‑4 37.48%; ties 24.64%.
    - Harmlessness:
      > GPT‑4 35.36%; LLaMA‑GPT4 31.66%; ties 32.98%.
    - Takeaway: On these user-oriented tasks, the 7B LLaMA tuned on GPT‑4 data is surprisingly close to GPT‑4 itself across HHH criteria.

- GPT‑4 automatic evaluation on 80 challenging prompts (Figure 4)
  - Effect of reward-model re-ranking for `LLaMA-GPT4`:
    - Against ChatGPT (Figure 4a):
      > Baseline 609:666 (91%); top‑1 re-ranked 624:667 (94%); others 85–92%.
    - Against GPT‑4 (Figure 4b):
      > Baseline 606:726 (83%); top‑1 re-ranked 631:722 (87%); others 83–85%.
    - Takeaway: The reward model’s top‑1 selection yields consistent, modest gains (2–4 percentage points).
  - Overall standings against strong opponents (Figures 4c–d):
    - Against ChatGPT (Figure 4c):
      > LLaMA (13B) 72%; Alpaca (13B) 83%; Vicuna (13B) 99%; LLaMA‑GPT4 (7B) 91%; LLaMA‑GPT4 (7B, R1) 94%; Bard 96%; ChatGPT 100%; GPT‑4 118%.
    - Against GPT‑4 (Figure 4d):
      > LLaMA (13B) 71%; Alpaca (13B) 80%; Vicuna (13B) 89%; LLaMA‑GPT4 (7B) 83%; LLaMA‑GPT4 (7B, R1) 87%; Bard 88%; ChatGPT 91%; GPT‑4 100%.
    - Takeaway: Despite using only 7B parameters, `LLaMA‑GPT4` outperforms raw LLaMA and Alpaca (both 13B) and closes part of the gap to Vicuna, Bard, and ChatGPT.

- Chinese evaluations (Figure 5)
  - When all models answer in English and outputs are translated to Chinese, compared against GPT‑4’s translated Chinese outputs (Figure 5a):
    > LLaMA (13B) 67%; Alpaca (13B) 76%; Vicuna (13B) 93%; LLaMA‑GPT4 (7B) 87%; LLaMA‑GPT4 (R1) 89%; Bard 92%; ChatGPT 95%; GPT‑4 100%.
  - When all models answer in English but compared to GPT‑4’s Chinese outputs generated directly from Chinese prompts (Figure 5b):
    > Scores are slightly higher for GPT‑4 itself (109%), and relative ordering remains consistent (e.g., LLaMA‑GPT4 (R1) 94%).
    - Insight highlighted in the text: GPT‑4’s own translated answers outperform its native Chinese answers on this benchmark, consistent with stronger English capability (Section 4.3).
  - When all models are prompted and answer in Chinese (Figure 5c):
    > Alpaca (13B) 33%; LLaMA‑GPT4 (7B) 35%; LLaMA‑GPT4‑CN (7B) 64%; Vicuna (13B) 79%; GPT‑4 92%.
    - Takeaway: Training directly on Chinese GPT‑4 data (`LLaMA‑GPT4‑CN`) substantially improves over English‑tuned `LLaMA‑GPT4` when evaluated in Chinese (64% vs 35%).

- Unnatural Instructions (Figure 6; 9K samples; ROUGE‑L)
  - Mean ROUGE‑L:
    > Alpaca 0.39; GPT‑4 0.37; LLaMA‑GPT4 0.34.
  - Trend vs ground-truth response length:
    > For longer expected answers (length > 10), GPT‑4 and LLaMA‑GPT4 close the gap or outperform, suggesting better handling of creative/long-form outputs; shorter answers favor Alpaca, likely because GPT‑4‑style chatty elaborations dilute n‑gram overlap with concise reference answers (Section 4.3; Figure 6 bars and legend).
  - Takeaway: ROUGE‑L, which rewards literal overlap, can penalize high-quality but more verbose or stylistically different answers; the authors caution that lower ROUGE for GPT‑4‑style outputs may not reflect worse usefulness (discussion around Figure 6).

- Do the experiments support the claims?
  - Yes, for core claims:
    - Human and GPT‑4 evaluations both favor the GPT‑4–tuned model over GPT‑3.5–tuned Alpaca, especially on helpfulness and overall pairwise quality (Figure 3a; Figure 4c–d).
    - Reward-model ranking provides consistent incremental gains (Figure 4a–b).
    - The Chinese-tuned model yields large gains in Chinese tasks (Figure 5c).
  - Caveats:
    - Heavy reliance on GPT‑4 as the evaluator raises potential circularity (the judge agrees with the teacher). Human HHH results help mitigate this, but are limited to 252 prompts (Section 4.2; Figure 3).

## 6. Limitations and Trade-offs
- Data sourcing and scope
  - The 52K instructions are reused from Alpaca; there is no new instruction induction or iterative filtering, and the collection is a one-time generation rather than a self-expanding set (Section 2 “We leave it as future work…”). This may limit instruction diversity.
  - The output distribution differs from Alpaca’s, partly because Alpaca iteratively removed similar instructions while this work did not (Figure 1d discussion).
- Model scale and training
  - Only the 7B `LLaMA` base is instruction-tuned, while some baselines are 13B and commercial systems are far larger, leaving potential headroom untested (Conclusions).
  - RLHF is not applied to update the policy; the reward model is used only for decoding-time ranking (Section 4.3; Conclusions). End-to-end RLHF might yield larger gains but was not explored.
- Evaluation design
  - GPT‑4 provides much of the scoring in automatic evaluations (Figures 4–5), introducing bias toward its own style. Human evaluation adds balance but is smaller-scale (252 items) and mostly single-turn (Section 4.2).
  - ROUGE‑based analysis penalizes verbose, chat-like responses; results on Unnatural Instructions should be interpreted with this limitation (Figure 6).
- Language and task coverage
  - Chinese instruction set is produced by translating English instructions with ChatGPT before answering with GPT‑4, so it may inherit translation artifacts; truly native Chinese instruction design is not covered (Section 2 item (2)).
  - Multi-turn dialogue, tool use, or grounded tasks are not addressed; all setups are single-turn instruction following (Algorithm 1; Section 3.1).

## 7. Implications and Future Directions
- How this changes the landscape
  - It validates a practical recipe: use a stronger proprietary model (`GPT‑4`) to bootstrap high-quality instruction data and automatic preference labels, then fine-tune and re-rank open models to get competitive assistants at modest parameter counts (Sections 2–4; Figures 3–5).
  - It shows cross-lingual portability: the same pipeline yields meaningful Chinese instruction followers and highlights the benefit of native-language tuning (Figure 5c).

- Follow-up research enabled or suggested
  - Scale both data and models
    - Expand instruction–response pairs well beyond 52K and tune larger LLaMA variants (e.g., 13B and above). The paper anticipates combining GPT‑4 data with multi-turn datasets like ShareGPT for further gains (Conclusions).
  - Full RLHF with machine feedback
    - Move beyond decoding-time ranking to train the base policy via reinforcement learning using the GPT‑4–derived reward model (Section 3.2; Conclusions). Investigate how far “AI feedback” can substitute for human feedback and where it fails.
  - Diversify evaluators and metrics
    - Incorporate human evaluations at larger scale; use task-grounded metrics and robustness checks to reduce style bias from a GPT‑4 judge. For short-answer tasks, employ exactness-focused metrics alongside semantic similarity to mitigate verbosity penalties (Figure 6 discussion).
  - Richer tasks and multilingual expansion
    - Extend to multi-turn dialogues, tool integration, and domain-specific settings (coding, math, safety-sensitive advice). Create native instructions in more languages rather than translated ones (Section 2 item (2)).
  - Data curation and quality control
    - Re-introduce iterative instruction generation and deduplication to expand diversity (noted as missing here relative to Alpaca), and analyze how instruction composition affects generalization (Figure 1d discussion).

- Practical applications
  - Organizations can train capable assistants on commodity hardware by:
    - Collecting instruction–response data with GPT‑4 using the provided prompt template and settings (Algorithm 1).
    - Fine-tuning smaller open models (`LLaMA` 7B) and using a lightweight reward model (`OPT` 1.3B) to re-rank outputs, achieving substantial gains over untuned or GPT‑3.5–tuned baselines (Figures 4c–d).
  - Cross-lingual deployments benefit from generating language-specific responses (Figure 5c), rather than relying solely on translation.

Overall, the work provides a clear, reproducible pathway to upgrade open models using GPT‑4 as both teacher and critic, with evidence from human and automatic evaluations that this pipeline delivers tangible improvements, particularly in helpfulness and cross-lingual capability.
