# INSTRUCTION TUNING WITH GPT-4

**ArXiv:** [2304.03277](https://arxiv.org/abs/2304.03277)

## 🎯 Pitch

This paper pioneers the use of GPT-4 to generate both instruction-following training data and automated feedback for fine-tuning open-source language models like LLaMA. By leveraging 52,000 English and Chinese responses written and rated by GPT-4, the authors create instruction-tuned models that surpass previous GPT-3.5–based benchmarks and approach the quality of proprietary models such as ChatGPT and GPT‑4, all while releasing their datasets and code to the community. This approach significantly reduces the need for costly human annotation and introduces scalable, high-quality evaluation and training pipelines, accelerating the development of competitive, accessible AI assistants.

---

## 1. Executive Summary
This paper shows how to use GPT-4 to generate both training data and automatic feedback to instruction-tune open-source language models, specifically LLaMA. Using 52K GPT-4–written answers (English and Chinese) and GPT-4–provided ratings, the authors fine-tune 7B-parameter LLaMA models that outperform prior GPT‑3.5–based instruction-tuning (e.g., Alpaca) and approach ChatGPT/GPT‑4 quality on several benchmarks, while also releasing the datasets and code.

## 2. Context and Motivation
- Problem addressed:
  - Instruction-tuned models learn to follow natural-language instructions. Building high-quality instruction-following models typically requires costly human-written prompts/responses and preference labels.
  - Prior “self-instruct” methods use a strong model (teacher) to synthesize instructions and answers for supervised fine-tuning, reducing human effort. However, most open-source efforts used GPT‑3.5-level data; the impact of GPT‑4 as a teacher and as an automatic judge had not been systematically explored.

- Why this matters:
  - High-quality instruction-following is central to practical assistants. Lowering the human labeling cost while improving quality can accelerate open, capable systems.
  - Reliable evaluation remains difficult; scalable automatic judging could make development more efficient.

- Prior approaches and gaps:
  - Human-supervised instruction tuning with human preference data (e.g., RLHF) improves alignment but is expensive.
  - Self-Instruct (Wang et al., 2022a) showed teacher-generated data is effective; Alpaca (Taori et al., 2023) used 52K GPT‑3.5 responses; Vicuna used ShareGPT conversations. These pipelines lacked:
    - Higher-quality teacher data from GPT‑4.
    - Machine-generated comparison data to train reward models without human raters.
    - A multilingual (Chinese) counterpart built from the same instruction set.

- Positioning:
  - The paper uses GPT‑4 for three roles: answer generator, rater (feedback provider), and evaluation judge. It releases a 52K English/Chinese dataset, trains instruction-tuned LLaMA models and a reward model, and evaluates with human HHH criteria and automatic GPT‑4 judging (Sections 1, 2, 3, 4).

## 3. Technical Approach
Step-by-step pipeline

1) Data generation (Section 2; Algorithm 1):
- Base instruction set: reuse the 52K unique instructions from Alpaca’s Self-Instruct collection.
- Prompting templates: two variants—one for instructions with extra input context and one without. Algorithm 1 shows the exact template and API call.
  - Quote (Algorithm 1): 
    > “Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request… ### Instruction: {instruction} … ### Input: {input} … ### Response:”
  - GPT‑4 decoding settings: 
    > `temperature = 1.0`, `top_p = 1.0`, `max_tokens = 512`, using `model="gpt-4"` (Algorithm 1 lines 13–17).

- Four datasets produced (Section 2):
  - English instruction-following: 52K GPT‑4 answers.
  - Chinese instruction-following: translate the 52K instructions to Chinese with ChatGPT, then have GPT‑4 answer in Chinese.
  - Comparison/feedback data: for each prompt, GPT‑4 assigns scores 1–10 to responses from multiple models (GPT‑4, GPT‑3.5, OPT‑IML), and performs pairwise comparisons—used to train a reward model (Section 2, Figure 2).
  - GPT‑4 answers for the “Unnatural Instructions” core set (68K) to quantify gaps (Section 2).

- Data statistics: linguistic patterns and length distributions (Figure 1).
  - GPT‑4 outputs are generally longer than GPT‑3.5 (Figure 1d). 
  - Distribution differences in verb–noun pairs are visualized (Figures 1a–1c).

2) Models (Section 3.1):
- Base model: `LLaMA 7B`.
- Supervised fine-tuning (SFT) settings follow Taori et al. (2023).
- Two SFT models:
  - `LLaMA-GPT4`: trained on the 52K English GPT‑4 answers.
  - `LLaMA-GPT4-CN`: trained on the 52K Chinese GPT‑4 answers.

3) Reward modeling (Section 3.2; Figure 2):
- Purpose of a reward model: predict a scalar preference score for a given prompt+response; used to rank multiple candidate generations.
- Training data: GPT‑4 assigns numeric scores (1–10) to multiple responses per prompt. For each prompt, form pairwise preferences from the scored set; each pair `(y_l, y_h)` is a lower- and higher-scored response.
- Model: `OPT 1.3B` as the reward model backbone.
- Objective (pairwise Bradley–Terry-style logistic loss):
  - Quote (Section 3.2):
    > minimize `log(σ(rθ(x, yh) − rθ(x, yl)))`, where `σ` is the sigmoid.
  - Intuition: encourage the model to give higher reward to the better response.

- Use at inference time (Section 4.3): generate multiple samples (e.g., 5 per prompt) and rank them with the reward model to pick the top response (“R1”).

4) Evaluation setup (Section 4):
- Human evaluation with HHH criteria on 252 unseen “User-Oriented Instructions” (Section 4.2; Appendix A.1, Figure 7).
  - HHH definitions:
    - Helpfulness: helps users achieve goals.
    - Honesty: provides true information, expresses uncertainty as needed.
    - Harmlessness: avoids harmful or unsafe content.
- Automatic evaluation with GPT‑4 as judge on 80 “Vicuna-Instructions” questions (Section 4.3; Figure 4).
  - GPT‑4 rates the quality between two models for each question (scores 1–10), then the paper sums across 80 questions; results reported as “relative scores” vs a strong opponent (ChatGPT or GPT‑4), normalized by a full score of 800 (Figures 4c–4d).
- Chinese evaluation variants (Figure 5):
  - Translate English responses to Chinese vs directly ask Chinese questions; compare models against GPT‑4 in both settings.
- ROUGE‑L on Unnatural Instructions (Figure 6): overlap metric between model responses and nominal answers; analyzed by ground-truth answer length.

Why these design choices
- GPT‑4 as teacher and judge: to leverage state-of-the-art quality for both data generation and scalable evaluation (Sections 1–3).
- 52K size: matches Alpaca to isolate the effect of teacher quality (Section 3.1), while keeping cost manageable.
- Pairwise reward modeling: standard and data-efficient way to learn from preference comparisons; avoids costly human pairwise labels by using GPT‑4 (Section 3.2).
- Two languages: test cross-language generalization and enable a Chinese instruction-tuned model (Section 2).

## 4. Key Insights and Innovations
- GPT‑4–generated instruction-following data improves SFT quality over GPT‑3.5–generated data at the same scale.
  - Evidence: on human Helpful/Honest/Harmless evaluation with 252 instructions (Figure 3a), LLaMA‑GPT4 wins decisively on helpfulness and is comparable on honesty/harmlessness where ties dominate.
  - On automatic judging across challenging questions (Figure 4c–4d), the 7B LLaMA‑GPT4 variants outperform 13B Alpaca and base LLaMA.

- Machine-generated “preference” data from GPT‑4 is sufficient to train a reward model that meaningfully improves output selection.
  - Evidence: when generating 5 responses per question and ranking with the reward model, “top‑1” selection beats the baseline single-sample decoding against both ChatGPT and GPT‑4 (Figures 4a–4b).
    - Quote (Figure 4a, vs ChatGPT): 
      > Baseline “B”: `609 : 666 = 91%`; Top‑1 ranked group: `624 : 667 = 94%`.
    - Quote (Figure 4b, vs GPT‑4):
      > Baseline “B”: `606 : 726 = 83%`; Top‑1 ranked group: `631 : 722 = 87%`.

- Cross-language instruction-tuning is viable with this pipeline, but English remains stronger and translating English answers to Chinese can score higher than native Chinese generation.
  - Evidence (Figures 5a–5c):
    - Against GPT‑4 with Chinese answers produced by translating from English (Figure 5a): LLaMA‑GPT4 (7B, R1) reaches `620 : 693 = 89%`, while Vicuna (13B) is `639 : 688 = 93%`.
    - When GPT‑4 generates Chinese natively (Figure 5b), normalized scores drop slightly across models.
    - A Chinese‑trained SFT model (`LLaMA‑GPT4‑CN`) improves over English‑trained when asked to respond in Chinese (`445 : 694 = 64%` vs LLaMA‑GPT4’s `253 : 723 = 35%`, Figure 5c).

- Behavioral closeness to GPT‑4 emerges, especially for longer, more open-ended responses; pure overlap metrics can undervalue helpful, chatty outputs.
  - Evidence (Figure 6): 
    - Average ROUGE‑L: Alpaca 0.39, LLaMA‑GPT4 0.34, GPT‑4 0.37.
    - However, for longer ground-truth answers (length >10), LLaMA‑GPT4 and GPT‑4 surpass Alpaca (bar differences +0.0562 and +0.0132 in the 6–10 and >10 bins), reflecting better performance on creative or multi-sentence tasks.
    - The paper notes both GPT‑4 and LLaMA‑GPT4 often include correct content plus extra helpful context, which can lower ROUGE despite being preferable (Section 4.3; Figure 6 discussion).

Overall, the innovations are practical and integrative rather than theoretical: a higher-quality teacher, a machine-labeled reward model, and a multilingual dataset assembled into a coherent, reproducible pipeline.

## 5. Experimental Analysis
Evaluation methodology

- Datasets (Section 4.1):
  - User-Oriented-Instructions‑252: 252 curated instructions reflecting real user applications (e.g., StackOverflow, Overleaf).
  - Vicuna-Instructions‑80: 80 challenging questions spanning knowledge, math, counterfactuals, roleplay, coding, etc.
  - Unnatural Instructions: 68,478 synthetic instruction–input–output triplets; the paper samples 9K for ROUGE‑L analysis (Figure 6 caption).

- Metrics:
  - HHH human ratings (Appendix A.1 shows the exact MTurk form; Figure 7).
  - Automatic GPT‑4 judging (pairwise scores 1–10 summed to a total of 800; Figures 4–5).
  - ROUGE‑L overlap for Unnatural Instructions (Figure 6).

- Baselines and systems under test:
  - Baselines: LLaMA 13B (untuned), Alpaca 13B (GPT‑3.5‑tuned), Vicuna 13B (ShareGPT‑tuned), Bard, ChatGPT, GPT‑4.
  - Proposed: LLaMA‑GPT4 7B (SFT on 52K GPT‑4 English), LLaMA‑GPT4‑CN 7B (SFT on 52K GPT‑4 Chinese), with and without reward-model ranking (“R1”).

Key quantitative results

- Human HHH on 252 instructions (Figure 3):
  - LLaMA‑GPT4 vs Alpaca:
    - Quote:
      > Helpfulness: Alpaca 19.74%, LLaMA‑GPT4 54.12%, Tie 26.14% (Figure 3a).
      > Honesty: Tie dominates at 42.61%; the two models split the remainder with small differences (Figure 3a).
      > Harmlessness: Tie dominates at 58.10% (Figure 3a).
  - LLaMA‑GPT4 vs GPT‑4:
    - Quote:
      > Helpfulness: GPT‑4 44.11%, LLaMA‑GPT4 42.78%, Tie 13.11% (Figure 3b).
      > Honesty: LLaMA‑GPT4 37.88%, GPT‑4 37.48%, Tie 24.64% (Figure 3b).
      > Harmlessness: GPT‑4 35.36%, LLaMA‑GPT4 31.66%, Tie 32.98% (Figure 3b).
  - Interpretation: LLaMA‑GPT4 is clearly more helpful than Alpaca, and roughly on par with GPT‑4 on this small human eval; honesty/harmlessness judgments are often “tie,” indicating difficulty distinguishing safety/accuracy at this granularity.

- Automatic GPT‑4 judging on 80 Vicuna questions (Figures 4a–4d):
  - Reward-model selection helps:
    - Quote (vs ChatGPT, Figure 4a):
      > Baseline `609 : 666 = 91%` → Top‑1 ranked `624 : 667 = 94%`.
    - Quote (vs GPT‑4, Figure 4b):
      > Baseline `606 : 726 = 83%` → Top‑1 ranked `631 : 722 = 87%`.
  - All-chatbot comparisons (vs ChatGPT, Figure 4c):
    - Quote:
      > LLaMA (13B): `502 : 698 = 72%`; Alpaca (13B): `585 : 704 = 83%`; Vicuna (13B): `649 : 652 = 99%`; LLaMA‑GPT4 (7B): `609 : 666 = 91%`; LLaMA‑GPT4 (7B, R1): `624 : 667 = 94%`; Bard: `634 : 660 = 96%`; ChatGPT: `759 : 759 = 100%`; GPT‑4: `613 : 521 = 118%`.
  - All-chatbot comparisons (vs GPT‑4, Figure 4d):
    - Quote:
      > LLaMA (13B): `520 : 732 = 71%`; Alpaca (13B): `593 : 746 = 80%`; Vicuna (13B): `640 : 716 = 89%`; LLaMA‑GPT4 (7B): `606 : 726 = 83%`; LLaMA‑GPT4 (7B, R1): `631 : 722 = 87%`; Bard: `633 : 722 = 88%`; ChatGPT: `652 : 714 = 91%`; GPT‑4: `760 : 760 = 100%`.
  - Interpretation: With only 7B parameters and 52K samples, LLaMA‑GPT4 surpasses 13B Alpaca and 13B base LLaMA, and approaches Vicuna/Bard/ChatGPT. Reward-model reranking gives a consistent bump.

- Chinese evaluations (Figure 5):
  - Translate-to-Chinese setting (Figure 5a):
    - Quote:
      > LLaMA‑GPT4 (7B): `607 : 700 = 87%`; LLaMA‑GPT4 (7B, R1): `620 : 693 = 89%`; Vicuna (13B): `639 : 688 = 93%`; ChatGPT: `652 : 684 = 95%`.
  - Native Chinese-question setting (Figure 5b):
    - Quote:
      > LLaMA‑GPT4 (7B): `618 : 686 = 90%`; LLaMA‑GPT4 (7B, R1): `629 : 672 = 94%`; Vicuna (13B): `658 : 677 = 97%`; ChatGPT: `658 : 679 = 97%`; GPT‑4: `680 : 626 = 109%`.
  - Fully in Chinese (models asked to answer in Chinese, Figure 5c):
    - Quote:
      > Alpaca (13B): `233 : 707 = 33%`; LLaMA‑GPT4 (7B): `253 : 723 = 35%`; LLaMA‑GPT4‑CN (7B): `445 : 694 = 64%`; Vicuna (13B): `545 : 691 = 79%`; GPT‑4: `626 : 680 = 92%`.
  - Interpretation: Training on Chinese GPT‑4 answers substantially improves Chinese performance (64% vs 35%). Translating English answers to Chinese often scores better than asking directly in Chinese, suggesting stronger English competence in models and judge.

- ROUGE‑L on Unnatural Instructions (Figure 6):
  - Quote:
    > Average ROUGE‑L: Alpaca 0.39; LLaMA‑GPT4 0.34; GPT‑4 0.37.
  - By ground-truth length:
    - For short references (0–2, 3–5 tokens), LLaMA‑GPT4 is behind GPT‑4 (−0.043, −0.009).
    - For longer references (6–10, >10), LLaMA‑GPT4 surpasses GPT‑4 (+0.0132, +0.0562 delta with GPT‑4 shown on bars).
  - Interpretation: Overlap metrics penalize helpful extra text; LLaMA‑GPT4 and GPT‑4 look better as tasks demand longer reasoning/output.

Do the experiments support the claims?
- Yes, for the central claims:
  - “GPT‑4 data improves instruction tuning over GPT‑3.5 data” is supported by HHH helpfulness gains (Figure 3a) and automatic GPT‑4 judging (Figures 4c–d).
  - “GPT‑4 feedback enables useful reward modeling” is supported by consistent improvements from reranking (Figures 4a–b).
  - “Comparable to GPT‑4 on HHH small set” is suggested by the near parity numbers in Figure 3b—but the sample is small and tie rates are high.
- Mixed findings:
  - ROUGE‑L average favors Alpaca (Figure 6), but analysis by length and qualitative interpretation suggest ROUGE underestimates quality for chat-like answers.

Ablations and robustness
- Present:
  - Reranking ablation (baseline vs top‑k ranks) shows monotonic improvements (Figures 4a–b).
  - English vs Chinese training and evaluation comparisons (Figure 5).
  - Output length and linguistic distribution analysis (Figure 1).
- Missing:
  - No full ablation on dataset size, temperature, or prompt variants.
  - No safety stress tests beyond coarse HHH “harmlessness” votes.
  - No variance/confidence intervals; automatic judging relies on a single judge (GPT‑4).

## 6. Limitations and Trade-offs
- Reliance on GPT‑4 as both teacher and judge:
  - Potential bias toward GPT‑4’s style and preferences; automatic evaluations may favor GPT‑4-like outputs.
  - Human evaluation is limited (252 prompts) with high tie rates for honesty/harmlessness (Figure 3), so fine-grained alignment properties remain uncertain.

- Data scale and diversity:
  - Only 52K instruction–answer pairs (matching Alpaca) and a single-pass collection without iterative de-duplication or error correction (Section 2 and Figure 1 discussion).
  - Instructions are inherited from Alpaca’s set; the domain coverage mirrors that distribution.

- Methodological scope:
  - Supervised fine-tuning only; the reward model is used only at decoding time, not to further train the policy via RLHF (Section 6).
  - Safety alignment is not separately optimized beyond SFT and HHH measurement; no constitutional or safety RL.

- Cross-language pipeline:
  - Chinese instructions are obtained by translating English with ChatGPT; translation errors and cultural nuances may limit data fidelity (Section 2).
  - Results show better performance when translating English answers to Chinese (Figure 5a) than generating in Chinese directly (Figure 5b), indicating a gap in native Chinese capabilities.

- Metrics:
  - ROUGE‑L penalizes extra helpful content (Figure 6), complicating comparisons to datasets with short references.
  - Automatic GPT‑4 judging is not independent and lacks inter-judge reliability estimates.

- Computational choices:
  - All GPT‑4 generations use `temperature=1.0`, `top_p=1.0` (Algorithm 1), which may introduce variability and verbosity; no sensitivity analysis is provided.

## 7. Implications and Future Directions
- How this changes the landscape:
  - Demonstrates that high-quality, machine-generated instruction data and preference labels from GPT‑4 can substantially close the gap between open-source SFT models and proprietary assistants, even at smaller model scales (7B).
  - Provides a reproducible recipe—prompt templates, decoding settings, and a reward-modeling scheme—plus released data/code that others can extend.

- Follow-up research enabled/suggested (Section 6 “Conclusions”):
  - Scale up data and models:
    - Quote:
      > “It would be promising to continue collecting more GPT‑4 instruction-following data … and train larger LLaMA models for higher performance.”
  - Move from reranking to full RLHF with machine-generated feedback:
    - Quote:
      > “It is natural to continue to train LLMs with reward models, for example for reinforcement learning using machine-generated feedback.”
  - Improve multilingual coverage:
    - Expand beyond Chinese; test native prompts/answers across languages with targeted evaluation sets.

- Practical applications:
  - Rapidly bootstrapping instruction-following assistants in new domains or languages using GPT‑4 as a data/feedback generator.
  - Cost-effective preference modeling for reranking or RLHF when human comparisons are scarce.
  - Creating evaluation harnesses where GPT‑4 can triage model regressions across diverse tasks (with caution about bias).

In sum, this paper offers a practical, well-documented pathway to leverage GPT‑4 as both a data generator and a scalable judge. The results—particularly the strong helpfulness gains and the effectiveness of reward-model reranking—are convincing on the chosen benchmarks (Figures 3–5). The main caveats are evaluation dependence on GPT‑4, limited human studies, and the absence of full RLHF training—clear opportunities for the next wave of work.
