# LIMA: Less Is More for Alignment

**ArXiv:** [2305.11206](https://arxiv.org/abs/2305.11206)

## 🎯 Pitch

LIMA demonstrates that a large pretrained language model can be aligned to act as a helpful AI assistant using only 1,000 high-quality prompt–response examples—without any reinforcement learning or massive instruction datasets. This challenges prevailing assumptions: LIMA matches or outperforms leading models like GPT-4 on almost half of evaluation prompts, revealing that most usable knowledge arises from pretraining, while alignment chiefly sets output style—thereby slashing the data and compute needed to create capable, aligned AI assistants.

---

## 1. Executive Summary
LIMA shows that a very large pretrained language model can be aligned to behave like a helpful assistant using only 1,000 carefully curated prompt–response examples, without reinforcement learning or massive instruction datasets. On 300 challenging prompts, human raters find LIMA’s answers equivalent or better than GPT-4 in 43% of cases and often preferred over other strong systems (Figures 1–2), suggesting that most task knowledge resides in pretraining while alignment primarily teaches response style and format.

## 2. Context and Motivation
- Problem addressed:
  - Modern large language models (LLMs) require “alignment” to produce helpful, safe, and user-appropriate outputs. Typical alignment uses millions of instruction-following pairs and/or reinforcement learning from human feedback (RLHF), which are expensive and slow to iterate.
- Why it matters:
  - If high-quality alignment could be achieved with far fewer examples, organizations could deploy capable assistants more easily, and researchers could study alignment mechanisms with less compute and data.
- Prior approaches and gaps:
  - Instruction tuning: supervised fine-tuning on very large multi-task datasets with millions of examples [e.g., references in Section 1]. These improve following instructions but demand massive curation/compute.
  - RLHF: learn a reward model from human preferences and optimize the model with reinforcement learning (Section 1). RLHF improves conversational helpfulness/harmlessness but requires millions of comparisons and skilled annotators.
  - Missing understanding: how much of “being a helpful assistant” is skill/knowledge vs. output style and formatting? How much data is actually necessary?
- Positioning:
  - LIMA tests the “Superficial Alignment Hypothesis” (Section 2): most knowledge and capabilities are learned during pretraining; alignment mainly teaches which response format/style to use. LIMA fine-tunes a 65B-parameter LLaMA on only 1,000 handpicked examples and measures how far this goes.

Definitions (selective):
- `Alignment`: guiding a pretrained model to produce helpful, safe, and appropriately formatted responses to user prompts.
- `Instruction tuning`: supervised fine-tuning on prompt–response pairs to make a model follow instructions.
- `RLHF`: reinforcement learning from human feedback where a learned reward model scores outputs and the base model is optimized to maximize this score.

## 3. Technical Approach
Step-by-step methodology (Sections 2–3, Table 1):

1) Base model
- Start from `LLaMA-65B`, a strong pretrained LLM.

2) Alignment data design: small, diverse prompts; uniform response style
- Size: exactly 1,000 prompt–response pairs (~750k tokens).
- Sources and counts (Table 1):
  - Stack Exchange STEM: 200; Stack Exchange Other: 200. 
  - wikiHow: 200.
  - Reddit r/WritingPrompts: 150 (creative stories).
  - Super-Natural Instructions: 50 (one example per selected task).
  - Manually authored by paper authors (“Group A”): 200.
- Dev and test prompts (Table 1):
  - Dev: 50 prompts (Group A).
  - Test: 300 prompts total: 230 manually authored (Group B) + 70 r/AskReddit.
- Curating for assistant style (Section 2.1–2.2):
  - Stack Exchange answers filtered for quality and style: remove too short/long, first-person phrasing, references to other answers; strip links/images, keep code blocks/lists.
  - wikiHow articles reformatted to answer style.
  - Reddit WritingPrompts chosen manually for high-quality creative responses.
  - Manually authored answers emphasize a uniform assistant tone: acknowledge question, then solve it; a structure that implicitly scaffolds reasoning.
  - Safety coverage: 13 training examples involve rejecting unsafe requests; 30 safety-relevant items are held out for test (Section 2.2).

3) Training protocol (Section 3)
- Objective: standard supervised next-token loss on the assistant’s response, conditioned on the prompt.
- Conversation delimiter: introduce a special `EOT` (end-of-turn) token at the end of each utterance to mark speaker turns without reusing the generic `EOS`.
- Hyperparameters:
  - 15 epochs; optimizer AdamW (β1=0.9, β2=0.95, weight decay=0.1).
  - Learning rate: start 1e-5, linearly decay to 1e-6; no warmup.
  - Batch size 32; context length 2048 tokens (truncate longer sequences).
  - Residual dropout applied across layers: linearly increases from 0 at bottom to 0.3 at the top layer. This follows an RLHF recipe (Ouyang et al., 2022) but used here in pure supervised fine-tuning.
- Checkpoint selection:
  - Perplexity on a small validation set anticorrelates with generation quality (Appendix B; Figure 9). Hence, checkpoints are manually chosen between epochs 5–10 using the 50-example dev set.

4) Evaluation pipeline (Section 4.1)
- Generation settings for all models: nucleus sampling with p=0.9, temperature 0.7, repetition penalty 1.2; max output 2048 tokens.
- Preference tests:
  - Pairwise comparisons on 300 test prompts. Each trial shows a prompt and two model responses; annotators choose the better answer or tie (Appendix C).
  - Baselines: `Alpaca 65B` (52k instruction-tuning set), `DaVinci003` (OpenAI RLHF model), `Bard` (PaLM-based), `Claude` (52B, Constitutional AI/RL from AI feedback), `GPT-4`. Responses collected in April 2023.
  - Two annotation regimes:
    - Human crowdworkers; inter-annotator agreement (tie-discounted accuracy): crowd–crowd 82%, crowd–author 81%, author–author 78%.
    - GPT-4-as-judge using the same instructions; agreement with humans around 78–79%.
- Absolute quality audit:
  - 50 random test prompts labeled as Fail, Pass, Excellent (Figure 3).
  - Additional 20 “out-of-distribution” (format-wise unseen) checks within this analysis.
- Safety analysis:
  - 30 potentially unsafe prompts from test; check refusal/safe guidance rate (Section 4.3).
- Ablations on alignment data (Section 5, Figure 5–6):
  - Use `LLaMA-7B` to cheaply test factors: prompt diversity, response quality, and dataset size. Evaluate with GPT-3.5 grading responses on a 1–6 helpfulness scale (Appendix D).
- Multi-turn dialogue emergence (Section 6, Figures 7–8):
  - Live chats on 10 dialogue seeds.
  - Compare zero-shot dialogue ability (trained only on single-turn data) vs. adding just 30 dialogue chains to the training set.

Why these design choices?
- Small but high-quality and diverse prompts test the hypothesis that alignment mainly conveys style/format, not knowledge (Section 2).
- Uniform assistant tone aims to make behavior predictable and generalizable to unseen prompts.
- Introducing `EOT` disentangles conversational turn-taking from generic end-of-sequence behavior.
- Residual dropout improves stability during fine-tuning very large models on small datasets.
- GPT-4-as-judge serves as an independent consistency check; the “Turking Test” style agreement indicates the rubric is well-posed.

## 4. Key Insights and Innovations
1) Minimal supervised data can yield competitive assistant quality
- What’s new: Fine-tuning on only 1,000 carefully curated examples (no RLHF, no massive synthetic sets) produces answers that humans prefer to “DaVinci003” 44% of the time and find at least as good as “Bard” 58% of the time (wins+t ties; Figure 1).
- Significance: Challenges the assumption that alignment needs millions of examples or RLHF to reach strong performance; suggests pretraining already encodes capabilities while alignment mostly teaches expression.

2) The Superficial Alignment Hypothesis is empirically supported
- Evidence: With this small dataset, 88% of responses meet prompt requirements (Figure 3: 50% Excellent, 38% Pass, 12% Fail) on a hard test set. Out-of-distribution checks show similar quality (45% Excellent, 35% Pass, 20% Fail; Section 4.3).
- Why it matters: Reframes alignment as largely a “formatting/style selection” problem atop pretrained knowledge, enabling more efficient alignment strategies.

3) Data diversity and quality matter more than sheer quantity for alignment
- Novelty: Ablations (Figure 5) show filtered, diverse Stack Exchange data yields higher helpfulness than homogeneous wikiHow or unfiltered data; scaling from 2k to 32k examples shows little improvement (Figure 6).
- Implication: Alignment “scaling laws” differ from pretraining; more of the same format does not help—diversity and editorial quality do.

4) Multi-turn dialogue emerges with zero-shot supervision and improves with 30 examples
- Finding: Even without dialogue training, the model sustains coherent multi-turn exchanges but fails frequently; adding only 30 dialogue chains raises the proportion of Excellent turns from 45.2% to 76.1% and reduces failures from 15/42 to 1/46 turns (Figure 7; examples in Figure 8).
- Significance: Reinforces that conversation skills are largely latent from pretraining and are activated with minimal supervision.

5) Perplexity is a poor proxy for aligned generation quality
- Observation: As training proceeds, validation perplexity increases while human/GPT-graded quality also increases (Appendix B, Figure 9), indicating standard language modeling metrics can be misleading for alignment.

## 5. Experimental Analysis
Evaluation methodology
- Datasets and prompts:
  - 300 test prompts comprising 230 manually authored by a held-out group and 70 r/AskReddit (Table 1). A 50-prompt dev set is used for checkpoint selection.
- Metrics:
  - Pairwise human preference with tie option (Section 4.1).
  - Inter-annotator agreement via tie-discounted accuracy; high agreement among humans and between humans and GPT-4 (Section 4.1).
  - Absolute audit: Fail/Pass/Excellent rates (Section 4.3).
  - Ablation helpfulness scores (1–6 Likert) using GPT-3.5 Turbo with a fixed rubric (Appendix D).
- Baselines:
  - `Alpaca 65B` (52k instruction tuning).
  - `DaVinci003` (RLHF).
  - `Bard` (PaLM-based), `Claude`, `GPT-4` (RLHF and related alignment training).
- Decoding setup:
  - All models sampled with the same p=0.9, temperature=0.7, repetition penalty=1.2; max 2048 tokens (Section 4.1).

Main quantitative results
- Human preferences (Figure 1):
  - vs Alpaca 65B: LIMA wins 53%, ties 21%, loses 26%.
  - vs DaVinci003: LIMA wins 44%, ties 21%, loses 35%.
  - vs Bard (April): LIMA wins 33%, ties 25%, loses 42% → equal or better in 58%.
  - vs Claude (April): LIMA wins 24%, ties 22%, loses 54%.
  - vs GPT-4 (April): LIMA wins 18%, ties 25%, loses 57% → equal or better in 43%.
- GPT-4-as-judge preferences (Figure 2) mirror the trend:
  - For example, vs GPT-4 itself: LIMA wins 19%, ties 15%, loses 66%.
- Absolute quality assessment (Figure 3):
  - 50% Excellent, 38% Pass, 12% Fail on 50 random test prompts.
- Safety (Section 4.3; Figure 4 right column shows a failure):
  - Safe responses on 80% of 30 safety-relevant prompts; 6/10 malicious-intent prompts are safely handled.
  - Some implicit harm scenarios slip through (e.g., suggesting medication for a neighbor’s dog).
- Ablations (Section 5):
  - Diversity and quality (Figure 5):
    - Filtered Stack Exchange (diverse, high quality): 3.83 helpfulness.
    - Unfiltered Stack Exchange (diverse, lower quality): 3.33.
    - wikiHow (homogeneous “how-to” prompts): 3.49.
  - Quantity (Figure 6):
    - Increasing examples from 2k to 32k on filtered Stack Exchange yields a flat quality curve near ~3.8–3.9, indicating diminishing returns without broader prompt diversity.
- Multi-turn dialogue (Section 6; Figure 7):
  - Zero-shot (no dialogue training) vs +30 dialogues:
    - Excellent proportion increases from 45.2% to 76.1%.
    - Failures drop from 15/42 to 1/46 turns.
    - Overall conversations: fine-tuned version is better in 7/10 and tied in 3/10.

Do the experiments support the claims?
- Support for the minimal-data alignment claim is strong: clear head-to-head preferences vs large-data and RLHF baselines (Figure 1), plus absolute quality audit and out-of-distribution checks (Section 4.3).
- The “style vs knowledge” hypothesis is consistent with multi-turn emergence and dramatic improvements from a handful of dialogue examples (Section 6) and with format-specific examples enabling complex structures (Appendix E, Figure 13).
- Reliability caveats exist: safety lapses (Figure 4), non-robust decoding in some cases (Section 7), and dependence on a high-capacity base model.

## 6. Limitations and Trade-offs
- Dependence on a very strong base model:
  - Results hinge on `LLaMA-65B`. Smaller models can work for ablations, but the competitive assistant performance is shown for 65B (Sections 3–4). Organizations without such a base model may not replicate performance.
- Manual, high-effort curation:
  - The 1,000 examples are meticulously filtered and authored for consistent assistant style (Sections 2.1–2.2). Section 7 acknowledges the “mental effort” does not scale easily.
- Safety coverage is thin:
  - Only 13 safety-oriented training examples and 30 test prompts; safe behavior is 80%—not product-grade (Section 4.3). Implicit harm remains a risk (Figure 4, “Unsafe” example).
- Generalization boundaries:
  - While out-of-distribution performance is promising (Section 4.3), some structured tasks fail without a few targeted examples (Appendix E), revealing sensitivity to format-specific supervision.
- Metric mismatch and selection:
  - Perplexity anticorrelates with quality (Appendix B, Figure 9), so the model selection uses a 50-example dev set and manual inspection—a potential source of bias.
- Evaluation scope and sources:
  - Test prompts include many authored by the research team; a footnote (Section 2.2) notes shared priors between author groups, which could bias style alignment toward LIMA’s training distribution. Nonetheless, external sources (AskReddit) help diversify.

## 7. Implications and Future Directions
- Field impact:
  - Shifts emphasis from massive instruction or RLHF pipelines to targeted curation of small, high-quality alignment sets. This lowers the barrier to developing capable assistants on top of strong base models.
  - Suggests an alternative “alignment scaling law”: prioritize prompt diversity and response quality, not just quantity (Section 5).
- Practical applications:
  - Rapid bootstrapping of domain assistants by curating a few hundred to a thousand style-consistent exemplars (e.g., legal, medical, customer support), especially where base models already encode domain knowledge.
  - Efficient upgrades: a handful of examples can unlock new structured-output capabilities (Appendix E) or improve multi-turn dialog (Section 6).
- Research directions:
  - Systematic methods for data curation: active selection to maximize prompt diversity and style coverage; automated quality filters grounded in human rubrics.
  - Safety and robustness with minimal data: explore “Constitutional”-style rules or AI-feedback loops but constrained to small datasets; study how few, carefully designed counterexamples can close safety gaps.
  - Understanding alignment mechanics: why perplexity diverges from helpfulness (Appendix B); how `EOT` design and residual dropout influence small-data fine-tuning stability.
  - Generality across base models: replicate the LIMA recipe on different architectures and smaller base models; quantify the minimal base capability needed for the “less is more” regime to hold.

Block-quoted highlights
- Human preference head-to-heads (Figure 1):
  > LIMA wins 44% vs DaVinci003, 53% vs Alpaca 65B; and is equal or better than Bard 58% of the time.
- Absolute quality (Figure 3):
  > 50% Excellent, 38% Pass, 12% Fail on 50 random test prompts.
- Ablations (Figure 5–6):
  > Diverse, filtered Stack Exchange data outperforms homogeneous or unfiltered sources; increasing data from 2k to 32k examples shows little gain without new diversity.
- Multi-turn improvement (Figure 7):
  > Adding just 30 dialogue chains raises Excellent turns from 45.2% to 76.1% and nearly eliminates failures.

Overall, LIMA demonstrates that with a capable base model, alignment can be achieved primarily by teaching “how to answer,” not “what to know.” The work reframes alignment as selective, high-quality style conditioning, opening a path to lighter-weight, faster, and more interpretable alignment pipelines.
