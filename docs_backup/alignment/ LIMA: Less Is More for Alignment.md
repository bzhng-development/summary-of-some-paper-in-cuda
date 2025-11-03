# LIMA: Less Is More for Alignment

**ArXiv:** [2305.11206](https://arxiv.org/abs/2305.11206)
**Authors:** Yvann Schwaller, Brian McMahan, Angela Jiang, Vlad Firoiu, Evan Hubinger, Jonathan Ward, Pedro A. Ortega, Markus J. Song, Jacob Andreas, Sébastien Bubeck, Samuel R. Bowman
**Institutions:** Microsoft Research, NYU, Anthropic

## 1. Executive Summary (2–3 sentences)

LIMA fine-tunes a strong pretrained language model (LLaMA 65B) on just 1,000 carefully curated prompt–response pairs (about 750k tokens) with standard supervised training—no RLHF—and still reaches competitive assistant-level quality. Its core finding is that most “capability” lives in pretraining; alignment primarily teaches style and formatting, so high-quality, diverse, but tiny alignment data can be enough to unlock impressive performance.

## 2. Context and Motivation

- The problem/gap
  - Modern large language models (LLMs) are built in two stages: (1) huge unsupervised pretraining on raw text to learn general-purpose capabilities, and (2) alignment to human tasks/preferences, often via instruction tuning on millions of examples and reinforcement learning from human feedback (RLHF). This second stage is expensive in compute, data collection, and engineering.
  - The paper asks: how much of alignment is truly needed if the base model is strong? Is alignment mostly about teaching the model the “style” of a helpful assistant rather than learning new task-specific knowledge?

- Why it matters
  - Real-world impact: If strong performance can be achieved with minimal supervision, labs and companies can align models faster, cheaper, and with less specialized infrastructure. This lowers the barrier to usable assistants and accelerates iteration cycles.
  - Theoretical significance: It challenges the default assumption that scaling alignment data and RLHF is always necessary, and introduces the idea that alignment might be “superficial”—primarily about response format and tone—because the knowledge is already there from pretraining.

- Prior approaches and their shortcomings
  - Large-scale instruction tuning (e.g., multi-million example datasets) and RLHF pipelines (collecting millions of preference comparisons) have achieved strong results but are data- and compute-heavy (Section 1, citations to [Chung et al., 2022], [Ouyang et al., 2022]).
  - Distillation-based methods automate dataset creation but often optimize for quantity over quality and can inherit teacher biases or errors (Section 2.2).

- How this paper positions itself
  - The Superficial Alignment Hypothesis (Section 2): “A model’s knowledge and capabilities are learnt almost entirely during pretraining, while alignment teaches it which subdistribution of formats should be used when interacting with users.”
  - LIMA tests this by fine-tuning LLaMA 65B on exactly 1,000 high-quality, diverse, assistant-style demonstrations (Table 1) and evaluating whether this is enough to match or beat models trained with far larger instruction tuning datasets and RLHF.

## 3. Technical Approach

At a high level: start with a strong base model, collect a small but diverse and high-quality dataset with a consistent assistant style, fine-tune with standard supervised learning, and evaluate through pairwise preference judgments.

- Data curation (Section 2; Table 1)
  - 1,000 training examples total, roughly 750,000 tokens (split into exactly 1,000 sequences).
  - Sources (Table 1):
    - Stack Exchange STEM: 200 examples (avg input 117 tokens, output 523)
    - Stack Exchange Other: 200 examples (avg input 119, output 530)
    - wikiHow: 200 examples (avg input 12, output 1,811)
    - Reddit r/WritingPrompts (via Pushshift): 150 examples (avg input 34, output 274)
    - Super-Natural Instructions (50 NL generation tasks): 50 examples (avg input 236, output 92)
    - Manually authored by Paper Authors (Group A): 200 examples (avg input 40, output 334)
  - Held-out for evaluation:
    - Dev set: 50 prompts (Group A)
    - Test set: 70 r/AskReddit prompts (titles only) + 230 author prompts (Group B)
  - Key curation choices (Sections 2.1–2.2):
    - Stack Exchange selection emphasizes quality and diversity: sample across 75 STEM and 99 non-STEM exchanges, pick high-score self-contained questions, take top answer with strong positive score (≥10), and filter to align with “helpful assistant” style. Automatic filters remove too short (<1,200 chars) or too long (>4,096) answers, first-person writing, and references to other answers; HTML stripped except code/list formatting.
    - wikiHow provides uniformly high-quality “how to” content; titles used as prompts; article bodies used as responses; minor preprocessing for style.
    - Reddit r/AskReddit used only for test prompts due to top comments often being humorous rather than “helpful.” r/WritingPrompts included for creative writing quality.
    - Manually authored 200 training examples aim for a consistent assistant tone (acknowledge the question, then answer) to encourage clear, stepwise structure—akin to “let’s think step-by-step” prompting (Section 2.2).
    - Safety examples: 13 in training with careful refusals/explanations; 30 safety-related prompts in test (Section 2.2, 4.3).
    - Micro-dataset for structural formatting: later, they show adding just six structure-constrained examples unlocks much better adherence to complex output formats (Appendix E; Figure 13).

- Model and fine-tuning setup (Section 3)
  - Base model: `LLaMA 65B` (Touvron et al., 2023).
  - Tokenization for dialogue: Introduce a special `EOT` token (End of Turn) after each utterance. This works like EOS to halt generation but avoids any pretrained semantics tied to the usual EOS token.
  - Training regime:
    - Optimizer: AdamW (β1=0.9, β2=0.95, weight decay 0.1)
    - Learning rate: 1e-5 linearly decaying to 1e-6, no warmup
    - Epochs: 15; batch size 32; context length truncated at 2048 tokens
    - Residual dropout schedule: linearly increase from 0.0 at bottom layer to 0.3 at the last layer (0.2 for smaller models). Residual dropout means applying dropout on residual connections to regularize—borrowed from Ouyang et al. (2022)—to help avoid overfitting when data is tiny.
    - Checkpoint selection: Although perplexity on a held-out Stack Exchange set increases (usually a bad sign), generation quality improves (Appendix B; Figure 9). They therefore select checkpoints manually between epochs 5–10 using the 50-example dev set’s generation quality.

- Evaluation generation settings (Section 4.1)
  - For each test prompt, sample one response per model:
    - Nucleus sampling with p=0.9, temperature=0.7, repetition penalty=1.2, max tokens=2048.

- Preference evaluation protocol (Section 4)
  - Pairwise comparisons: For each prompt, show annotators two responses (LIMA vs a baseline) and ask which is better, or mark a tie. The same instructions are also given to GPT-4 for an automated “Turking” check (i.e., assessing if GPT-4 agrees with human preferences).
  - Baselines (Section 4.1):
    - Alpaca 65B (52k instruction-tuning examples)
    - OpenAI DaVinci003 (trained with RLHF)
    - Google Bard (PaLM-based)
    - Anthropic Claude (trained with Constitutional AI, a form of AI-feedback reinforcement learning)
    - OpenAI GPT-4 (trained with RLHF)
    - All baseline outputs sampled April 2023.
  - Inter-annotator agreement (Section 4.1): They use “tie-discounted accuracy” (1 point for full agreement; 0.5 if one annotator chose tie; 0 otherwise). Agreement among humans is high: crowd–crowd 82%, crowd–author 81%, author–author 78%. GPT-4 aligns with humans at ~78–79%, “passing” this annotation task as a surrogate judge.

- Ablations (Section 5)
  - Use LLaMA 7B for stability and speed on ablations. Train on 2,000+ examples, sample five outputs per prompt, and auto-score helpfulness via GPT-3.5 on a 1–6 Likert scale (Appendix D).
  - Vary data diversity (Stack Exchange vs wikiHow), response quality (filtered vs unfiltered), and quantity (2k–32k examples).
  - Key to isolate: What matters more—diversity, quality, or quantity?

- Multi-turn dialogue test (Section 6)
  - LIMA trained on single-turn interactions still shows surprisingly coherent multi-turn behavior. Then they add just 30 dialogue examples (20 edited from Stack Exchange threads + 10 hand-authored), retrain on 1,030 examples total, and re-evaluate dialogue.

## 4. Key Insights and Innovations

- A tiny, high-quality alignment set can be enough to “unlock” the pretrained model’s generality
  - What’s new: LIMA uses only 1,000 examples to fine-tune a 65B model and reaches competitive assistant performance without RLHF (Abstract; Sections 1, 4).
  - Why it matters: It strongly supports the idea that alignment mostly teaches the model “how to talk” (format, tone, refusal style) rather than core skills or knowledge. This reframes alignment strategy around curation and diversity over scale.

- The Superficial Alignment Hypothesis gets concrete evidence
  - Novel hypothesis: Alignment is mainly about learning response style; capabilities are acquired during pretraining (Section 2).
  - Evidence:
    - LIMA follows specific formats and style from very few examples, including out-of-distribution tasks like speculative fiction or itineraries (Abstract; Sections 4.3).
    - LIMA achieves good multi-turn dialogue with zero dialogue training, and adding only 30 dialogues makes it significantly better (Section 6; Figure 7, Figure 8).
    - A small number of safety-style examples yield safe refusals for most sensitive prompts (Section 4.3).

- Data diversity and quality beat raw quantity for alignment
  - New finding: Scaling up the number of training examples alone doesn’t improve generation quality; diversity and high-quality responses do (Section 5).
    - Filtered Stack Exchange (diverse, high-quality) outperforms wikiHow (uniform “how-to” prompts) and unfiltered Stack Exchange (same diversity but lower quality). Figure 5 shows average helpfulness score 3.83 (filtered SE) vs 3.49 (wikiHow) vs 3.33 (unfiltered SE).
    - Increasing training examples from 2k to 32k yields a plateau (Figure 6).

- Perplexity isn’t a reliable proxy for alignment quality
  - Insight: During fine-tuning, validation perplexity on Stack Exchange worsens while human-judged quality improves (Appendix B; Figure 9). That suggests style alignment can diverge from standard language modeling metrics; human-in-the-loop or task-grounded validation is needed.

- Minimal, targeted examples can unlock new structured behaviors
  - When LIMA initially struggles with complex formatting requirements, adding just six structure-focused examples (e.g., multi-section marketing plan, bullet-pointed summaries) dramatically improves adherence to requested formats (Appendix E; Figure 13). This is a practical recipe: teach the format once and the model generalizes.

Overall, the fundamental innovation is not an algorithmic trick but a sharp reframing: treat alignment as style transfer done via rigorously curated, diverse, high-quality demonstrations, rather than a data- or RL-intensive learning problem.

## 5. Experimental Analysis

- Evaluation setup
  - Datasets:
    - Test prompts: 300 total = 70 r/AskReddit titles + 230 manually authored by a separate author group (Table 1).
    - Additional analyses on safety prompts (30 items) and out-of-distribution prompts (20 items within a 50-item manual sample; Section 4.3).
  - Metrics:
    - Pairwise human preference (win/tie/lose) vs five baselines (Figure 1).
    - GPT-4 as a surrogate judge for the same comparisons (Figure 2).
    - Absolute ratings: manual labeling into Fail/Pass/Excellent over 50 random test prompts (Figure 3).
    - For ablations: GPT-3.5 helpfulness scores (1–6) with 95% CI (Section 5).

- Main quantitative results (Section 4.2; Figures 1–3)
  - Pairwise human preferences (Figure 1):
    - Versus Alpaca 65B (52k examples): LIMA wins 53%, ties 21%, loses 26%.
    - Versus DaVinci003 (RLHF): LIMA wins 44%, ties 21%, loses 35%.
    - Versus Bard: LIMA wins 33%, ties 25%, loses 42%. So LIMA is at least as good 58% of the time.
    - Versus Claude: LIMA wins 24%, ties 22%, loses 54%.
    - Versus GPT-4: LIMA wins 18%, ties 25%, loses 57%. So “equal or preferred” in 43% of cases.
  - GPT-4 as judge (Figure 2) confirms trends:
    - LIMA beats Alpaca 65B 64% wins, 19% ties, 17% loses.
    - Beats DaVinci003 54% wins, 23% ties, 23% loses.
    - Versus GPT-4 itself: LIMA still wins 19% and ties 15% (loses 66%).
  - Absolute quality (Figure 3):
    - On 50 sampled prompts: 50% Excellent, 38% Pass, 12% Fail.
    - Out-of-distribution slice (20 items): 45% Excellent, 35% Pass, 20% Fail (Section 4.3).
  - Safety (Section 4.3):
    - On 30 sensitive prompts, 80% responses are “safe,” including 6 out of 10 explicit malicious-intent prompts yielding refusals.

  Representative quote:
  > “In a controlled human study, responses from LIMA are either equivalent or strictly preferred to GPT-4 in 43% of cases; ... 58% when compared to Bard and 65% versus DaVinci003.” (Abstract; consistent with Figures 1–2)

- Ablations (Section 5; Figures 5–6)
  - Diversity matters:
    - 2,000-example training sets: filtered Stack Exchange (diverse prompts) scores 3.83, wikiHow (homogeneous prompts) 3.49 (Figure 5).
  - Quality matters:
    - Filtered vs unfiltered Stack Exchange: 3.83 vs 3.33 (Figure 5).
  - Quantity alone doesn’t:
    - Scaling from 2k to 32k examples plateaus; doubling data repeatedly doesn’t reliably improve quality (Figure 6).

- Multi-turn dialogue (Section 6; Figure 7–8)
  - Zero-shot dialogue (no dialogue data): Across 10 live chats, responses are “surprisingly coherent” but off-distribution; fail rate rises within a few turns.
    - Turn-level distribution (Figure 7): Excellent 45.2%, Pass 35.7%, Fail 19.1%.
  - After adding 30 dialogue chains (1,030 examples total): Big jump in dialogue quality.
    - Turn-level distribution (Figure 7): Excellent 76.1%, Pass 21.7%, Fail 2.2%.
    - Conversation-level: “significantly better” in 7/10 conversations and tied in 3/10.
    - Failure rate drops from 15 fails per 42 turns to 1 fail per 46 turns.

- Do the experiments support the claims?
  - Yes, convincingly for the scope tested. The pairwise human studies across 300 prompts with strong baselines demonstrate the core claim: tiny, high-quality alignment can compete with instruction-tuned and RLHF models on many prompts. The ablations cleanly isolate that diversity and response quality, not sheer data volume, drive improvements. The multi-turn section and format-constraint experiments show new behaviors can be unlocked with just tens of examples.
  - Caveat: This is one base model (LLaMA 65B). Results may rely on base model strength; the paper partially explores smaller models (7B) for ablations but the headline comparisons are at 65B.

- Failure cases and conditions (Section 4.3; Figure 4)
  - LIMA sometimes fails on safety where malicious intent is implicit (last column in Figure 4 shows an unsafe “advice” behavior).
  - It can produce weak responses under adversarial prompts or unlucky sampling (Section 7).
  - It initially struggles with complex output structures until given a few targeted examples (Appendix E; Figure 13).

## 6. Limitations and Trade-offs

- Assumptions and scope
  - Requires a very strong pretrained base model (LLaMA 65B). The “less is more” result depends on substantial capabilities already learned via massive pretraining (Abstract; Sections 1–2). Smaller or weaker base models likely need more supervision or won’t reach similar quality.
  - The approach assumes alignment is largely about style/format (Superficial Alignment Hypothesis). This may hold broadly for many assistant tasks, but tasks that require tools, retrieval, or precise safety/legal constraints may need more specialized training or systems integration.

- Data curation cost and bias
  - While the dataset is small, making it diverse, high-quality, and stylistically consistent is labor-intensive (Section 7).
  - Some training/test prompts were authored by different subgroups of the authors, but there was “significant contact between the groups” before annotation, which “resulted in certain shared priors” (Section 2.2). This could inflate performance on author-written test prompts.

- Safety is only lightly covered
  - Only 13 safety/harms-relevant training examples; 80% safety on 30 sensitive tests is promising but not production-grade (Section 4.3). Implicit harm scenarios remain tricky (Figure 4, right column, unsafe advice).

- Evaluation ceilings and breadth
  - Strong product models (GPT-4, Claude) still outperform LIMA overall (Figure 1), even though LIMA wins a non-trivial fraction.
  - Benchmarks rely on pairwise preferences and human/GPT-4 judges; no standardized task benchmarks (e.g., MMLU, factuality stress tests) are reported here.

- Metric mismatch
  - Perplexity becomes unreliable during alignment. This complicates automated validation and early stopping; human-verified dev sets or task-based proxies are necessary (Appendix B; Figure 9).

- Compute realities
  - Even if data is small, fine-tuning a 65B model is still compute-intensive and may require specialized hardware and careful engineering. So “less is more” refers to alignment data and pipeline complexity, not necessarily total compute/storage ease.

## 7. Implications and Future Directions

- How this changes the landscape
  - Alignment-by-curation: The paper suggests a new center of gravity—curate a tiny, diverse, high-quality, and stylistically consistent dataset and do simple supervised fine-tuning. This challenges the assumption that large-scale instruction tuning or RLHF is always necessary for high-quality assistant behavior.
  - Operationally, labs with limited resources can still produce strong assistants if they have access to a strong base model and can invest in careful data curation.

- Practical applications
  - Rapid prototyping of domain assistants: With tens to hundreds of carefully written examples, you can shape a model’s tone, structure, and refusal behavior for a specific domain (e.g., medical advice style guides, legal memos, enterprise support).
  - Safety guardrails via few-shot style: Add small, targeted safety examples to bias refusals and safe alternatives without building a full RLHF pipeline.
  - Complex formatting and workflows: Teach templates (e.g., reports, marketing plans, grant proposals) with a handful of exemplars (Appendix E), then generalize broadly.

- Follow-up research ideas
  - What’s the minimal boundary? Systematically map how dataset size, diversity, and quality interact with base model size to hit specific performance targets. The plateau in Figure 6 suggests a “diversity frontier” rather than a “quantity frontier.”
  - Better automatic validation signals than perplexity: Can we learn a proxy for style alignment quality that correlates with human preference, avoiding expensive human evaluation?
  - Safety scaling with small data: Identify the minimal set of safety exemplars needed to robustly handle implicit and adversarial harms. Explore hybrid systems (e.g., rule-based postprocessing) combined with LIMA-style fine-tuning.
  - Multimodal or tool-augmented LIMA: Extend the “curate a tiny dataset” philosophy to tool-use (APIs, retrieval) or multimodal outputs, giving a few canonical exemplars to unlock those capabilities.
  - Data compositionality: Evaluate how adding a small number of “capability unlockers” (like the 30 dialogues or 6 structure-constrained examples) composes—can we modularly layer capabilities by adding small shards of curated tasks?

Reasoning for this analysis
- The argument that alignment is mostly about style is supported by multiple converging results: strong pairwise preferences with only 1,000 examples (Figure 1–2), high absolute quality (Figure 3), out-of-distribution generalization (Section 4.3), and dramatic improvements from tiny “capability unlockers” (Section 6; Appendix E).
- The ablations isolate causes: diversity and response quality consistently improve results while quantity alone does not (Figures 5–6), matching the style-learning hypothesis rather than a data-scaling hypothesis.
- The approach’s limits (safety coverage, dependency on base model strength, need for careful curation) are either explicitly discussed (Section 7) or visible in failure cases (Figure 4).
- Together, these patterns make the central claim—“less is more for alignment”—credible for strong pretrained LLMs, while also pointing to where more work is needed for robust, production-grade systems.

Key citations to ground claims
- Dataset composition and sizes: Table 1; Sections 2–2.2
- Training protocol and EOT token: Section 3
- Preference results vs baselines: Figures 1–2; Section 4.2
- Absolute quality and OOD generalization: Figure 3; Section 4.3
- Safety behavior: Section 4.3; Figure 4 (unsafe case example)
- Ablations on diversity/quality/quantity: Section 5; Figures 5–6
- Multi-turn dialogue improvements with 30 examples: Section 6; Figures 7–8
- Perplexity vs generation quality anticorrelation: Appendix B; Figure 9
- Format-constrained unlock with six examples: Appendix E; Figure 13

Selected quotes for emphasis
- > “responses from LIMA are either equivalent or strictly preferred to GPT-4 in 43% of cases; ... 58% when compared to Bard and 65% versus DaVinci003” (Abstract)
- > “Overall, these remarkable findings demonstrate the power of pretraining and its relative importance over large-scale instruction tuning and reinforcement learning approaches.” (end of Section 1; echoed in Sections 4, 5, 6)
- > “Scaling up quantity alone might not [improve alignment]; scaling up input diversity and output quality have measurable positive effects.” (Section 5; Figures 5–6)