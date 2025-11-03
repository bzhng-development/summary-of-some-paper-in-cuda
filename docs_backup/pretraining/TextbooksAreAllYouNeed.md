# Textbooks Are All You Need

**ArXiv:** [2306.11644](https://arxiv.org/abs/2306.11644)

## 🎯 Pitch

This paper introduces phi-1, a 1.3B-parameter language model for code that rivals or outperforms much larger models on standard coding benchmarks—achieving 50.6% pass@1 on HumanEval and 55.5% on MBPP—by training almost exclusively on 'textbook-quality' data. By rigorously filtering web code and generating synthetic code textbooks and exercises, the authors demonstrate that high-quality, highly instructive data enables small models to dramatically outperform much larger ones, challenging the conventional belief that scaling model size or dataset volume is the only reliable route to strong AI performance. These results have profound implications for the democratization, efficiency, and sustainability of future AI systems, suggesting that 'better' data can be more valuable than 'more' data.

---

## 1. Executive Summary (2-3 sentences)
This paper introduces `phi-1`, a 1.3B-parameter code-focused language model trained mostly on “textbook-quality” data—carefully filtered web code and GPT-3.5–generated textbooks and exercises. With a fraction of the model size and training tokens of prior systems, `phi-1` reaches pass@1 scores of 50.6% on HumanEval and 55.5% on MBPP, showing that data quality can bend conventional scaling laws (Table 1; Section 2).

## 2. Context and Motivation
- Problem/gap:
  - Modern code LLMs typically gain accuracy by scaling parameters and data volume. This is costly and environmentally impactful, and it implicitly assumes “more data” beats “better data.” The paper asks: can high-quality, curriculum-like data allow small models to rival much larger ones?
- Importance:
  - Practical: Lower compute and smaller models democratize code generation and reduce environmental cost (Introduction).
  - Scientific: It probes whether scaling laws can be “bent” by data quality, not just by adding compute or parameters (Introduction; [EL23] motivation).
- Prior approaches and shortcomings:
  - Most code LLMs (e.g., CodeGen, StarCoder, Replit, PaLM-Coder; Table 1) train on massive web code corpora (hundreds of billions to trillions of tokens) like The Stack. Manual inspection shows such data often lacks properties that teach algorithmic reasoning: self-containedness, meaningful computation, clarity, and balance (Section 2, bullet list).
  - Some works clean data or study scale, but few construct a deliberately educational, balanced “textbook” curriculum for code and test its impact on scaling behavior.
- Positioning:
  - The paper adopts the “textbook-quality data” lens from TinyStories [EL23] and extends it to code. It builds a training set intended to teach rather than merely expose code, then shows this alters performance-vs-scale trade-offs (Figures 2.1, Table 1).

## 3. Technical Approach
The project proceeds as a pipeline: curate data → pretrain → finetune → probe capabilities and decontamination.

- Core idea: replace indiscriminate scale with curated, educational content:
  - “Textbook-quality data” means data that is self-contained, instructive, and balanced across concepts (Section 2). The authors operationalize this in three datasets.

1) Filtering large web code with an LLM-aided classifier (Section 2.1)
- Starting pool: deduplicated Python files from The Stack + StackOverflow (35M samples, >35B tokens).
- Label a small subset (~100k samples) for “educational value” using GPT-4 as an annotator.
- Train a random forest classifier on embeddings from a pretrained CodeGen model to predict educational value for all samples.
- Keep only high-value code/text pairs, yielding ~6B tokens of “filtered code-language” data.
- Why this matters: removes boilerplate and non-instructive code (Figure on “Educational values deemed by the filter”), enabling the model to see concise, algorithmically meaningful snippets instead of configuration glue.

2) Synthetic “Textbook” data generation (<1B tokens) with GPT-3.5 (Section 2.2)
- Goal: provide natural-language-heavy explanations interleaved with code, focused on reasoning/algorithmic skills.
- Diversity mechanism: inject randomness and constraints into prompts (e.g., specify topics and audience) so the generator explores varied content instead of repeating stock examples (Section 2.2, “trick” inspired by TinyStories).
- Example: a mini linear algebra section on singular vs nonsingular matrices with Python functions (Section 2.2).

3) Synthetic “CodeExercises” (~180M tokens) with GPT-3.5 (Section 2.2)
- Format: function docstrings describing tasks + code solutions—explicitly aligned to the HumanEval-style function-completion task.
- Diversity: constrain function names to force variation (Section 2.2).
- Decontamination: later sections conduct n‑gram and deeper similarity checks, and even retraining on pruned data (Section 5).

These two synthetic datasets plus the filtered web dataset form:
- `CodeTextbook` = filtered web code (~6B tokens) + synthetic textbooks (<1B tokens) for pretraining.
- `CodeExercises` = synthetic exercises (~180M tokens) for finetuning.

4) Model architecture and training (Section 2.3)
- Architecture:
  - Decoder-only Transformer with FlashAttention; parallel MHA/MLP; Rotary Position Embeddings; CodeGen tokenizer.
  - `phi-1` (1.3B params): 24 layers; hidden size 2048; MLP 8192; 32 attention heads of size 64.
  - `phi-1-small` (350M params): 20 layers; hidden 1024; MLP 4096; 16 heads of size 64.
- Training setup:
  - Objective: next-token prediction on 2048-token sequences separated by `<|endoftext|>`.
  - Optimizer/schedule: AdamW, warmup–decay; dropout 0.1; fp16; 8×A100; DeepSpeed; batch size details in Section 2.3.
- Pretraining:
  - Train on `CodeTextbook` for 36k steps; checkpoint at 24k steps used as `phi-1-base`, which corresponds to ~8 epochs and “a little over 50B” tokens seen in total (Section 2.3).
  - Compute footprint: 770 GPU-hours for 1.3B (Figure 2.1 caption).
- Finetuning:
  - Start from `phi-1-base`; finetune on `CodeExercises` for 6k steps; best checkpoint every 1k steps used as `phi-1` (Section 2.3).
  - Compute: ~7 hours on the same hardware.

5) Why this approach over alternatives?
- Rather than adding parameters/tokens, the pipeline increases the “teaching signal” per token, hypothesized to reshape scaling behavior (Introduction and Section 2).
- Filtering with a learned classifier is more scalable than hand-curation; synthetic generation is guided to ensure diversity and coverage (Section 2.2).
- Notably, the paper intentionally does not use training tricks like Fill-in-the-Middle or Multi-Query Attention (Section 2.3), isolating the effect of data quality.

6) Emergence after finetuning (Section 3)
- Although `CodeExercises` only contains short Python tasks with basic libraries, finetuning produces improvements on:
  - Logical understanding (custom simulation example; Section 3.1).
  - Using external libraries not present in finetuning data, e.g., PyGame and Tkinter (Section 3.2) with side-by-side completions from `phi-1`, `phi-1-base`, and `phi-1-small`.
  - Basic chat-style interaction (Section 3, “Chat mode example”).
- Mechanistic interpretation: finetuning reorganizes and consolidates knowledge acquired during pretraining, enhancing the model’s ability to apply it (Section 3, opening paragraph).

7) Decontamination and robustness evaluation (Sections 4–5)
- Concern: did finetuning leak HumanEval-like tasks?
- Two lines of evidence:
  - New “unconventional” problems (50 items) written by a separate team; graded by GPT-4 using rubric-like prompts (Section 4; Table 2).
  - Strong-form pruning: compute similarity between `CodeExercises` and HumanEval using both embedding distance (CodeGen-350M embeddings) and AST-based edit distance (“match rate”), prune up to 40%+ of exercises, retrain, and re-evaluate (Section 5; Table 3; Appendix C examples).

Definitions of uncommon terms used:
- `pass@1`: probability that the single generated solution passes unit tests; more precisely, the proportion of problems solved when sampling a single output.
- `AST match rate τ`: a measure of syntactic similarity between two code snippets based on edit distance between their Abstract Syntax Trees; higher τ means more similar trees (Section 5.2).
- `Embedding distance`: L2 distance between code embeddings produced by a pretrained model; captures semantic similarity (Section 5.2).

## 4. Key Insights and Innovations
- High-quality, curriculum-like data changes the returns to scale (fundamental innovation).
  - Instead of amassing tokens, `phi-1` relies on carefully filtered and synthetic “textbook” content. The 1.3B model trained on ~7B pretraining tokens plus ~180M finetuning tokens achieves 50.6% on HumanEval (Table 1), rivaling or beating many models 10–100× larger in either parameters or tokens.
  - Figure 2.1 shows the biggest accuracy jump comes from finetuning on `CodeExercises`, not from extra compute or more raw data.
- LLM-aided data filtering yields tangible gains even without synthetic data (incremental but important).
  - On a 350M model, HumanEval accuracy saturates at 12.19% on unfiltered Stack after ~200B tokens, but rises to 17.68% after switching to the filtered subset and to 20.12% when adding synthetic textbooks (Section 2.1).
- Finetuning on small, focused exercises “unlocks” latent capabilities learned during pretraining (novel empirical observation).
  - After finetuning on basic exercises, `phi-1` becomes better at tasks absent from finetuning (Section 3), such as using external libraries (PyGame, Tkinter) with correct API calls (Figures in Section 3.2). This suggests consolidation/organization effects beyond mere memorization.
- Strong-form decontamination methodology (methodological contribution).
  - Beyond n‑gram checks, the paper uses embedding- and AST-based similarity to prune any `CodeExercises` entries similar to HumanEval at varying thresholds, retrains, and shows high performance persists (Table 3). This is stronger than common overlap metrics alone (Section 5).

## 5. Experimental Analysis
- Evaluation setup:
  - Benchmarks and metrics:
    - HumanEval (pass@1) and MBPP (pass@1) for generative code tasks (Abstract; Table 1).
    - A new set of 50 “unconventional” problems designed to avoid training overlap; GPT-4 grades logic/understanding on a 0–10 scale converted to percentages (Section 4; Table 2).
  - Baselines:
    - A broad set of contemporary code LLMs, including CodeGen variants, StarCoder, Replit, PaLM-Coder, etc. (Table 1).
  - Ablations and comparisons:
    - 350M vs 1.3B models; varying data sources (The Stack vs CodeTextbook); with/without `CodeExercises` finetuning; more training tokens vs higher-quality tokens (Figure 2.1).
  - Training compute references:
    - `phi-1-base` at 51B tokens uses 770 GPU-hours; a baseline trained on The Stack+ uses 1,090 GPU-hours and 76B tokens (Figure 2.1 caption).

- Main quantitative results:
  - Overall accuracy:
    - Quote (Table 1): 
      > `phi-1` (1.3B, ~7B pretraining tokens + 180M finetuning) reaches “50.6% HumanEval and 55.5% MBPP,” outperforming many larger models trained on hundreds of billions to trillions of tokens.
  - Scaling by data quality (Figure 2.1):
    - For 1.3B models, moving from The Stack+ to `CodeTextbook` boosts pass@1 (17% → 29%), and finetuning on `CodeExercises` further jumps to 51%.
    - For 350M models, accuracy improves similarly with `CodeTextbook`, and `phi-1-small` reaches 45% HumanEval (Figure 2.1).
  - LLM-graded “understanding” on new problems (Table 2):
    - `phi-1`: 52% (understanding score) vs StarCoder 51% (even though StarCoder is 15.5B).
    - The ranking matches HumanEval performance, suggesting genuine capability rather than test leakage (Section 4).
  - Data filtering effect (Section 2.1):
    - Quote:
      > “350M models on unfiltered Stack saturate at 12.19% after ~200B tokens; filtered subset achieves 17.68% after 36k steps; adding synthetic textbooks improves to 20.12%.”
  - Decontamination via pruning (Table 3):
    - Even after removing 42.5k–354k of 879.5k exercises (τ from 0.95 to 0.8), the retrained model’s overall HumanEval pass@1 remains higher than StarCoder-Prompted (e.g., 45.1–50.6% vs 41.5%).
    - Quote (Table 3):
      > “Even after heavily pruning our dataset, `phi-1` still outperforms StarCoder-Prompted by a large margin.”

- Do the experiments support the claims?
  - The consistent improvements across ablations (Figure 2.1), strong baselines (Table 1), robust decontamination (Table 3), and a separate, newly authored evaluation (Table 2) collectively support the central claim: high-quality, curriculum-like data enables small models to perform competitively.
  - Caveat: Using GPT-4 as a grader (Section 4) removes the need for unit tests but introduces potential subjectivity and bias; however, it is complemented by standard benchmarks and decontamination analyses.

- Failure cases and robustness (Appendix B; Section 3):
  - The paper documents weaknesses post-finetuning, including sensitivity to prompt phrasing/length, struggles with ambiguous NL instructions, and difficulty with counting/spatial layout in GUIs (Appendix B; examples provided).

## 6. Limitations and Trade-offs
- Assumptions and scope:
  - The model is specialized for Python; multi-language code generation and domain-specific APIs are not central targets (Conclusion).
  - The data curation strategy assumes that educational quality can be reliably approximated by a classifier trained on GPT-4–labeled samples (Section 2.1). If the labeling heuristic is biased, the filter may skew topics or styles.
- Scenarios not addressed:
  - Complex, large-scale software engineering (e.g., multi-file repositories, advanced frameworks) is outside the finetuning distribution (“short Python tasks using only basic libraries”; Section 3).
  - Robust handling of long, noisy, or ambiguous prompts remains weak (Appendix B).
- Computational/data constraints:
  - Although compute is drastically lower than prior SOTA, pretraining still processes ~50B tokens (multi-epoch training over a ~7B-token corpus), which is nontrivial for smaller labs (Section 2.3).
  - Some proprietary details of synthetic data generation are omitted, limiting full reproducibility (Abstract end; Introduction).
- Open questions:
  - How far can “textbook quality” scale—does performance keep improving with better curricula or does it plateau?
  - To what extent does synthetic-data “recursion” risk narrowing model diversity over generations (Related Work discussion around [SSZ+23; GWS+23; MMJ+23])?

## 7. Implications and Future Directions
- Field-level impact:
  - This work shifts emphasis from “how big?” to “how well taught?” It provides a concrete, replicable template showing that educationally curated and synthetic curricula can rival brute-force scaling (Table 1; Figure 2.1).
- Follow-up research directions:
  - Generalize curricula to other languages (Java, C++, JS) and multi-file projects; explore curriculum design theory for LLMs (ordering, prerequisites, difficulty pacing).
  - Improve synthetic generation fidelity with higher-quality teachers (e.g., GPT-4), as the paper notes GPT-3.5 data has high error rates (Conclusion).
  - Combine textbook-quality pretraining with known training tricks (FIM, MQA) and retrieval augmentation to further boost performance (Section 2.3 mentions not using these).
  - Formalize “data diversity” metrics and coverage analyses for code curricula (Conclusion: the paper notes a lack of methodology for measuring diversity/redundancy).
  - Expand decontamination methods and standardized protocols across code benchmarks, including semantic similarity and AST-level measures (Section 5; Appendix C).
- Practical applications:
  - Organizations can train smaller, specialized code assistants tuned for educational settings (tutors), onboarding, and basic automation with lower compute budgets.
  - Curriculum-driven pretraining could improve robustness in safety-critical coding tasks by emphasizing clarity and correctness in training data.
  - Teaching-focused datasets may be used to fine-tune larger foundation models to improve reasoning and adherence to instructions without massive additional data.

> Key takeaway: “Textbooks are all you need” is not hyperbole in this context; carefully filtered and synthetic, didactic data can substitute for vast quantities of raw web code, enabling a 1.3B model to compete with models trained on two to three orders of magnitude more tokens (Table 1), with strong evidence that finetuning on small, well-designed exercises unlocks broader capabilities (Section 3; Figure 2.1).
