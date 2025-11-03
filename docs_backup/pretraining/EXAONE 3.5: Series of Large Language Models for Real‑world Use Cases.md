# EXAONE 3.5: Series of Large Language Models for Real‑world Use Cases

**ArXiv:** [2412.04862](https://arxiv.org/abs/2412.04862)
**Authors:** LG AI Research, Soyoung An, Kyunghoon Bae, Eunbi Choi, Kibong Choi, Stanley Jungkyu Choi, Seokhee Hong, Junwon Hwang, Hyojin Jeon, Gerrard Jeongwon Jo, Hyunjik Jo, Jiyeon Jung, Yountae Jung, Hyosang Kim, Joonkee Kim, Seonghwan Kim, Soyeon Kim, Sunkyoung Kim, Yireun Kim, Yongil Kim, Youchul Kim, Edward Hwayoung Lee, Haeju Lee, Honglak Lee, Jinsik Lee, Kyungmin Lee, Woohyung Lim, Sangha Park, Sooyoun Park, Yongmin Park, Sihoon Yang, Heuiyeen Yeen, Hyeongu Yun
**Institutions:** LG AI Research

## 🎯 Pitch

EXAONE 3.5 presents a groundbreaking suite of instruction-tuned language models capable of bilingual, long-context reasoning with exceptional efficiency, achieving top results on real-world benchmarks while utilizing significantly less computational power. By introducing novel Korean RAG benchmarks, EXAONE 3.5 not only enhances language model capabilities but catalyzes further research in bilingual retrieval and application, marking a crucial step in deploying powerful AI on resource-constrained devices.

---

## 1. Executive Summary
EXAONE 3.5 is a family of instruction-tuned large language models (LLMs) in three sizes (`32B`, `7.8B`, `2.4B`) that target strong real‑world instruction following and long‑context reasoning up to 32K tokens. The work matters because it delivers bilingual (Korean/English) models that achieve top performance on several real‑world and long‑context benchmarks while using notably less pre‑training compute than comparable open models (Table 3), and it contributes new Korean long‑context RAG benchmarks.

## 2. Context and Motivation
- Problem and gap
  - Many users need LLMs that 1) follow instructions robustly in everyday tasks, 2) reason with long inputs typical in RAG pipelines, and 3) run at a range of cost and device constraints. Prior EXAONE 3.0 (7.8B) delivered good bilingual capability, but context length was limited (4K), and there was clear demand for both larger/faster models and much smaller on‑device models (Section 1).
- Importance
  - Real‑world usage increasingly centers on RAG and long documents (contracts, reports, web search results). If the model cannot reliably retrieve and reason over 10K–30K tokens, downstream applications suffer. Cost and deployability (e.g., small models) are also decisive for academic and industrial adoption.
- Prior approaches and gaps
  - Recent open models (e.g., Qwen 2.5, Gemma 2, Llama 3.x) reach high general benchmarks but often trade off cost or bilingual strength, and many do not support 32K context across sizes. Long‑context training often risks catastrophic forgetting of prior capabilities or relies on much higher pre‑training budgets.
- Positioning
  - EXAONE 3.5 positions itself as a cost‑efficient, bilingual, long‑context instruction‑tuned family (2.4B/7.8B/32B) with:
    - 32K token context across all sizes (Table 1).
    - A two‑stage pre‑training process with replay to prevent forgetting (Section 2.2.1).
    - Careful test‑set decontamination (Section 2.2.2; Figure 4; Table 10).
    - Supervised fine‑tuning and staged preference optimization for alignment (Sections 2.3.1–2.3.2; Figures 1–2).
    - New long‑context Korean RAG benchmarks (Section 3.4; Appendix D.2).

## 3. Technical Approach
This is an empirical systems paper: it defines three model configurations, a pre‑ and post‑training pipeline, and an evaluation suite.

- Architecture (Table 1)
  - All models are decoder‑only Transformers with:
    - `pre-normalization` residual blocks and `SwiGLU` non‑linearity (a gated activation that improves optimization stability).
    - `GQA` (Grouped Query Attention): queries are grouped to share key/value projections, reducing memory/computation while retaining performance.
    - Rotary position embeddings with large `RoPE theta = 1,000,000`, which keeps positional encodings well‑behaved for long contexts.
    - `BBPE` tokenizer (byte‑level BPE) with a 102,400 vocabulary designed to split coverage roughly 50% Korean / 50% English.
    - 32K maximum sequence length for all sizes.
    - Size‑specific choices (Table 1), e.g., `32B` uses 64 layers, model width 5,120, 40 attention heads with 8 K/V heads; `2.4B` ties input/output embeddings to save parameters (“Tied word embedding: True”).

- Two‑stage pre‑training (Section 2.2; Table 2)
  - Stage 1 trains on a large multi‑domain web‑scale corpus (9T tokens for `7.8B`, 6.5T for `2.4B` and `32B`).
  - Stage 2 “long‑context fine‑tuning” (Section 2.2.1) extends the effective context to 32K using the positional interpolation method [7]. To avoid catastrophic forgetting, a replay strategy reuses part of Stage‑1 data (Section 2.2.1). A key procedural change: in Stage 2, long documents are trained in their intact form rather than chunked, directly exercising long‑range dependencies.

- Decontamination (Section 2.2.2; Figure 4; Table 10)
  - Goal: remove any training examples that leak benchmark test content. The pipeline:
    1) Normalize test items (keep alphanumeric only).
    2) Build a substring pool from all unique 50‑character sliding windows.
    3) For a candidate training example, sample N=10 substrings and flag as contaminated if any match occurs.
  - Table 10 shows real overlap examples for MMLU and KMMLU that were removed.

- Compute efficiency (Table 3)
  - Compute is approximated by “model size × training tokens.” EXAONE `32B` (6.5T tokens) requires 1.0× relative compute; Qwen 2.5 `32B` at 18T requires 2.77×; Gemma 2 `27B` at 13T requires 1.69×. The argument is that EXAONE 3.5 matches or beats long‑context/real‑world performance using lower training budgets.

- Post‑training alignment (Section 2.3; Figures 1–2)
  - Supervised fine‑tuning (SFT) (Section 2.3.1; Figure 1):
    - Build a “knowledge taxonomy” from 8M web pages (e.g., “Math → Algebra → Arithmetic sequence,” “Arts → Music → Jazz”) and generate instructions grounded in those topics.
    - Use an “instruction evolution” step (a method inspired by [58]) to systematically increase difficulty and variety, producing diverse instruction‑response pairs.
  - Preference optimization (Section 2.3.2; Figure 2):
    - Use direct alignment algorithms (DAAs) such as `DPO` and `SimPO`. Build preference data by sampling N responses from multiple models for each prompt `x`, then ranking them with a reward model to choose best `y_w` and worst `y_l`.
    - Validate preference pairs with a second reward model and keep only pairs where both reward models agree above a threshold. 
    - Train in stages `M0 → M1 → M2` to mitigate over‑optimization (a known risk where models overfit the reward model; [38]).

- Long‑context focus in evaluation and training
  - Needle‑in‑a‑Haystack (NIAH; Section 3.4.1; Figure 3) is used up to 32K tokens in English and Korean. The “needle” is a specific sentence inserted at random depth; the task is to retrieve it verbatim. EXAONE shows near‑perfect accuracy at all depths/lengths.
  - RAG‑oriented benchmarks: `LongBench`, extended `LongRAG` with explicit unanswerable cases (Appendix D.2.3), and new Korean datasets `Ko‑LongRAG` and `Ko‑WebRAG` (Appendix D.2.4–D.2.5). For `LongRAG`, the prompts explicitly instruct models to answer “Unanswerable” when evidence is missing.

- Evaluation protocol details (Section 3.1; Appendix D)
  - Mix of automatic metrics (exact match, accuracy, F1, ROUGE) and `LLM‑as‑a‑judge` (GPT‑4o‑2024‑08‑06 or GPT‑4‑1106) for open‑ended outputs (Tables 4, 6–8; Appendix D.2–D.3).
  - For general‑domain tasks, use zero‑shot prompts and, where specified, zero‑shot chain‑of‑thought (`CoT`) prompts with answer parsing (Appendix D.3). Greedy decoding, max generation length 2,048.

Definitions of less common terms:
- `GQA` (Grouped Query Attention): an attention variant where multiple query heads share a smaller set of key/value heads, reducing memory and latency.
- `RoPE theta`: a scaling factor in rotary position embeddings; larger values help preserve positional distinctions at long sequence lengths.
- `DAA`/`DPO`/`SimPO`: families of preference optimization that train the model to prefer human‑preferred responses without explicit reinforcement learning.
- `LLM‑as‑a‑judge`: using a strong model (e.g., GPT‑4o) to grade another model’s outputs when no exact ground truth exists.
- `NIAH`: synthetic long‑context test where a “needle” sentence must be found in a very long “haystack.”

## 4. Key Insights and Innovations
- Cost‑efficient long‑context training with replay
  - What’s new: A two‑stage pre‑training scheme that switches from chunked data (Stage 1) to full, intact documents (Stage 2) with a replay buffer to prevent catastrophic forgetting (Section 2.2.1). 
  - Why it matters: Enables all model sizes to support 32K tokens while maintaining prior capabilities. Evidence: near‑perfect NIAH retrieval across lengths and depths in both languages (Figure 3).

- Strong real‑world instruction following under tight compute
  - What’s new: A focus on “real‑world” instruction datasets and evaluation, plus staged preference optimization with double reward‑model agreement to curate high‑quality preference pairs (Section 2.3.2; Figure 2).
  - Why it matters: On MT‑Bench, Arena‑Hard, AlpacaEval, IFEval, etc., EXAONE outperforms similar‑size open baselines (Table 6). This supports the claim that alignment and data design choices, not just scale, drive practical gains.

- Bilingual long‑context RAG evaluation resources
  - What’s new: Extension of `LongRAG` with unanswerables and creation of `Ko‑LongRAG` and `Ko‑WebRAG` (Section 3.4.2; Appendix D.2.3–D.2.5).
  - Why it matters: These datasets stress both retrieval and generation under long contexts, including Korean web search scenarios. EXAONE leads on these tasks (Table 7), demonstrating a key real‑world capability.

- Small model that punches above its weight
  - What’s new: A `2.4B` model trained for 32K context with strong instruction following and RAG performance (Tables 5–7).
  - Why it matters: It tops or matches larger models (≤9B) in real‑world and long‑context averages and is competitive in general benchmarks (Table 5), enabling on‑device or resource‑constrained deployments.

Incremental vs. fundamental:
- Incremental: Architecture choices (SwiGLU, GQA, RoPE) are established techniques. 
- More fundamental for this work: The training recipe to reliably scale long‑context across sizes with replay; the alignment pipeline with staged DAAs and dual reward‑model filtering; and the creation of new Korean long‑context RAG evaluations.

## 5. Experimental Analysis
- Evaluation setup (Section 3; Table 4; Appendix D)
  - Categories:
    - Real‑world instruction following: MT‑Bench, LiveBench (2024‑08‑31), Arena‑Hard‑v0.1, AlpacaEval 2.0 LC, IFEval, KoMT‑Bench, LogicKor; metrics include LLM‑as‑judge win rates/scores and instruction strict accuracy.
    - Long context: NIAH (EN/KR), LongBench, extended LongRAG, Ko‑LongRAG, Ko‑WebRAG; metrics include F1/ROUGE, LLM‑as‑judge scores, and accuracy.
    - General domain: GSM8K (CoT), MATH (CoT), HumanEval, MBPP, GPQA (CoT), ARC‑C, BBH (CoT), MMLU (CoT), KMMLU (CoT); zero‑shot prompts with standardized parsing (Appendix D.3).
  - Baselines: recent open models across sizes (Appendix D.1), including Qwen 2.5, Llama 3.1/3.2, Gemma 2, Phi‑3, Yi 1.5.

- Headline results (Tables 5–8; Figure 3)
  - Real‑world use cases (macro average; Table 5 and Table 6)
    - > “`EXAONE 3.5 32B` average 74.3 vs `Qwen 2.5 32B` 69.8” (Table 5).
    - Per‑benchmark (32B; Table 6): “Arena‑Hard 78.6 vs 67.0; AlpacaEval 60.6 vs 41.0; IFEval 81.7 vs 78.7.” MT‑Bench is tied (8.51 vs 8.49). LiveBench is lower (43.0 vs 50.6).
    - `7.8B` and `2.4B` models also lead their peer groups on the macro average (Table 6).
  - Long‑context (macro average; Table 7)
    - > “`EXAONE 3.5 32B` average 71.1 vs `Qwen 2.5 32B` 66.9,” with strong Korean RAG scores: Ko‑LongRAG 85.3 and Ko‑WebRAG 82.3.
    - `7.8B` average 66.6 vs Qwen 2.5 7B at 56.1; `2.4B` average 63.4 vs Qwen 2.5 3B at 40.7.
    - NIAH: near‑perfect retrieval across depths and lengths in EN/KR (Figure 3).
  - General domain (macro average; Table 8)
    - Mixed: `Qwen 2.5 32B` leads (78.7) over `EXAONE 32B` (74.8), especially on MATH (+6.0), MBPP (+7.1), BBH (+7.4).
    - `EXAONE 2.4B` leads its size class with 63.3 vs Qwen 2.5 3B at 62.1 and Llama 3.2 3B at 54.9.
  - Compute vs. performance (Table 3)
    - > “Qwen 2.5 32B needs 2.77× the compute of EXAONE 3.5 32B,” yet EXAONE matches or surpasses it on long‑context and real‑world averages (Tables 5, 7).

- Safety/harmlessness (Table 9)
  - On a 10,000‑item Korean trustworthiness benchmark: 
    - > “Overall accuracy: 87.1% (32B), 85.6% (7.8B), 72.2% (2.4B).” 
    - High in “Hate” and “Illegal” subcategories for larger models; smaller model lags.

- Do the experiments support the claims?
  - The long‑context and real‑world instruction‑following advantages are well supported by:
    - Near‑perfect NIAH (Figure 3).
    - Dominance on Ko‑LongRAG/Ko‑WebRAG (Table 7).
    - Strong wins on Arena‑Hard/AlpacaEval/IFEval (Table 6).
  - General‑domain superiority is not claimed; indeed, Qwen 2.5 32B leads there (Table 8), which is consistent and increases credibility.
  - Robustness checks:
    - Decontamination is described rigorously (Section 2.2.2; Figure 4; Table 10).
    - LongRAG is extended with unanswerable cases and explicit instructions to answer “Unanswerable” (Appendix D.2.3), a good stress test for RAG reliability.
  - Caveats:
    - Several benchmarks use `LLM‑as‑a‑judge` (Table 4) with GPT‑4o/4‑1106. The paper acknowledges separability issues with earlier judges and switches to GPT‑4o (footnote in Table 4). This improves reliability but still introduces potential bias and variance.

- Missing ablations
  - No ablation on replay vs. no‑replay for long‑context, or on the effect of dual reward‑model filtering in preference optimization.
  - No cost breakdown showing training time/throughput from GQA or memory savings from embedding tying.

## 6. Limitations and Trade-offs
- Assumptions and scope
  - Heavy use of web data and synthetic preferences; while decontamination is strong, residual leakage is always possible (Section 2.2.2).
  - `LLM‑as‑a‑judge` evaluation may favor certain response styles or language fluency; although prompts and judges are specified (Table 4; Figures 5 and 8), inter‑judge reliability is an open concern.

- Scenarios not directly addressed
  - Multimodal inputs are out of scope (text‑only).
  - Ultra‑long contexts beyond 32K are not evaluated, nor are memory‑compression methods for >32K contexts.

- Computational and data constraints
  - While compute is lower than some peers (Table 3), training still requires multi‑trillion tokens and major GPU resources (Table 2). The license is research‑only (Appendix B), limiting commercial deployment without separate agreement.

- Performance trade‑offs
  - General‑domain reasoning/coding lags Qwen 2.5 32B on several tasks (Table 8).
  - Smaller model (`2.4B`) performs very well given size but still shows safety gaps relative to larger models (Table 9).

- Open questions
  - How much of the real‑world/long‑context gain comes from data recipes versus model architecture? Ablations could quantify this.
  - How sensitive are results to the judge choice and prompt phrasing in LLM‑as‑a‑judge settings?

## 7. Implications and Future Directions
- Field impact
  - Demonstrates that carefully engineered data/finetuning pipelines can deliver strong long‑context and instruction following at lower training budgets (Table 3) and across sizes, including small models suitable for edge/on‑device scenarios (Table 5).
  - Provides Korean long‑context RAG benchmarks that will likely catalyze research in bilingual retrieval and generation (Appendix D.2.4–D.2.5).

- Practical applications
  - High‑reliability RAG for enterprise search and knowledge bases (Ko‑LongRAG/Ko‑WebRAG results in Table 7).
  - Long‑document analytics: contracts, scientific articles, and reports (32K context; Figure 3).
  - Bilingual assistants for Korean/English markets (vocab and results across KO/EN in Tables 6–7).

- Follow‑up research
  - Ablations on long‑context replay, RoPE scaling, and instruction evolution to isolate contribution of each technique.
  - Robust evaluation beyond LLM‑as‑a‑judge: human studies, pairwise calibration, and adversarial testing for long‑context hallucinations.
  - Safety advancement: improve smaller models’ trustworthiness (Table 9) and analyze bias across more languages/domains.
  - Methods to extend context further with compression or memory modules while retaining the replay‑based stability shown here.
  - Public release and standardization of Ko‑LongRAG/Ko‑WebRAG leaderboards to benchmark bilingual RAG more broadly.

> Bottom line: EXAONE 3.5 contributes a well‑engineered training/evaluation pipeline that turns modest compute into strong real‑world and long‑context performance (Tables 5 and 7), introduces valuable Korean RAG benchmarks, and shows that a 2.4B model can meaningfully compete in these scenarios. General‑domain reasoning is competitive but not best‑in‑class (Table 8), highlighting a clear trade‑off and an opportunity for future optimization.
