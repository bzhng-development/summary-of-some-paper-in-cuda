# Let there be answers: DeepSeek LLM Technical Report

**ArXiv:** [2401.02954](https://arxiv.org/abs/2401.02954)
**Authors:** DeepSeek AI team
**Institutions:** DeepSeek (Hangzhou DeepSeek Artificial Intelligence Basic Technology Research Co., Ltd.)

## 🎯 Pitch

DeepSeek LLM unveils two open-source language models trained on a vast bilingual corpus, redefining scaling laws with a novel measure of 'model scale' that accurately predicts compute allocation and hyperparameter tuning. This transformative approach not only outperforms existing models like LLaMA-2 in several benchmarks but also optimizes resource usage, enabling more efficient and effective language model training at scale, particularly in advancing bilingual capabilities and reasoning tasks.

---

## 1. Executive Summary (2–3 sentences)
DeepSeek LLM introduces two open-source decoder-only language models (`7B` and `67B`) trained from scratch on a bilingual 2-trillion-token corpus, and—more importantly—revisits scaling laws to make compute allocation and hyperparameter selection predictable. It proposes a new way to measure “model scale,” derives empirical formulas for the optimal batch size and learning rate as compute grows, and shows that data quality shifts the optimal trade-off between model size and data volume; guided by these findings, the 67B model outperforms LLaMA‑2 70B on many code, math, reasoning, and Chinese benchmarks (Section 5; Table 5–6).

## 2. Context and Motivation
- Problem addressed
  - Open-source LLM training has converged on a few fixed sizes (7B, 13B, 34B, 70B) without clear guidance on how to scale models and datasets as compute budgets increase (Introduction, p.3). Prior “scaling laws” disagree on the optimal split between model size and data size—Kaplan et al. tend to favor larger models, while Chinchilla (Hoffmann et al.) favors more data for a given compute budget (Section 3, Table 4).
  - Hyperparameters that matter most at scale (batch size, learning rate) are often tuned ad hoc. It is unclear how to set them when compute changes (Section 3.1).

- Why it matters
  - Practically: Bad scaling choices waste millions of GPU-hours. A systematic, predictive recipe reduces cost and risk for open-source LLM training.
  - Scientifically: Reconciling scaling laws clarifies what governs generalization as models and datasets grow, and how dataset quality interacts with scaling (Section 3.3).

- Prior approaches and gaps
  - Earlier scaling laws used parameter count to represent “model scale” and approximated compute as `C = 6 N D` (parameters × tokens). This ignores attention cost and can misestimate compute, especially for small models or long contexts (Section 3.2; Table 3).
  - Hyperparameter guidance was empirical and conflicting (Section 3.1), and open-source projects seldom publish end-to-end recipes usable across budgets.

- Positioning
  - DeepSeek contributes: (1) a new model-scale proxy that tracks actual training FLOPs per token; (2) compute-conditioned formulas for optimal batch and learning rate; (3) evidence that better data shifts the compute-optimal split toward larger models; and (4) two strong bilingual models trained using these rules (Sections 2–4).

## 3. Technical Approach
This work contains three intertwined parts: a full pretraining pipeline, an alignment pipeline, and a scaling-law program that governs both.

- Pretraining data pipeline (Section 2.1)
  - Corpus: 2T tokens, primarily Chinese and English.
  - Processing stages:
    - Deduplication across the entire CommonCrawl (not just within a dump), which removes far more duplicates—“dedup across 91 dumps eliminates ~89.8% duplicates vs 22.2% within a single dump” (Table 1).
    - Filtering with both linguistic and semantic checks to increase information density.
    - Remixing to rebalance underrepresented domains.
  - Tokenization: Byte-level BPE with pre-tokenization barriers between newlines/punctuation/CJK; digits split; 100,015 learned tokens + 15 specials, trained on ~24 GB multilingual text. A padded training vocabulary of 102,400 is used to keep compute efficient (Section 2.1).

- Model architecture (Section 2.2; Table 2)
  - Base design follows LLaMA: Pre-Norm with `RMSNorm`, `SwiGLU` in FFN (intermediate size `8/3 * d_model`), `Rotary` positional embeddings.
  - 67B uses `Grouped-Query Attention (GQA)`—a memory/time-optimized variant that groups multiple query heads to share key–value projections—reducing inference cost.
  - Macro design is deeper rather than wider: 7B = 30 layers (`d_model=4096`, 32 heads); 67B = 95 layers (`d_model=8192`, 64 heads, 8 KV heads). The choice of depth aids pipeline parallel partitioning (Section 2.2).
  - Context length: 4096 tokens for both.

- Optimization and scheduler (Section 2.3)
  - AdamW with β1=0.9, β2=0.95, weight decay 0.1; init std=0.006; gradient clip=1.0.
  - A multi-step learning rate schedule: warm up to max LR over 2000 steps; then decay to 31.6% of max after 80% of tokens, then to 10% after 90% (Figure 1a). This matches cosine LR in final quality but enables straightforward “continual training” by reusing the first training phase (Figure 1a–b).

- Training system (Section 2.4)
  - Framework: HAI‑LLM with data/tensor/sequence parallelism and 1F1B pipeline parallelism; FlashAttention; ZeRO‑1 optimizer state sharding; fused ops; bf16 forward/backward with fp32 gradient accumulation.
  - Asynchronous checkpointing every 5 minutes (“lose no more than 5 minutes in worst failures”).
  - Evaluation served via vLLM.

- Scaling laws program (Section 3)
  - Notation and key definitions
    - Compute budget `C` is modeled as the product of per-token model FLOPs `M` and number of training tokens `D`: `C = M * D` (Section 3.2).
    - `M` is “non-embedding FLOPs per token”—a replacement for parameter-based proxies. It counts attention compute and excludes vocabulary softmax cost, which improves fidelity (Equation (2)).
    - `IsoFLOP profile`: fit performance across multiple model/data pairs that share the same total compute `C`. This helps identify the best split of `C` across `M` and `D` (Section 3.2).
    - “Generalization error” is measured as bits-per-byte (BPB) on a held-out validation set (100M tokens) drawn from the same distribution as training (Figure 4–5).

  - Step 1: Hyperparameter scaling (Section 3.1)
    - Small-model grid searches show a wide plateau of near-optimal batch/learning-rate pairs (Figure 2a).
    - Using multi-step LR to reuse an initial phase, they sweep many compute budgets (1e17–2e19 FLOPs) and fit power laws for near-optimal hyperparameters:
      > Equation (1): `η_opt = 0.3118 * C^-0.1250`, `B_opt = 0.2920 * C^0.3271`  
      The near-optimal band is wide (Figure 3), suggesting robust choices across budgets.
    - Validation at larger budgets shows fitted values sit in the center of the optimal region (Figure 2b).

  - Step 2: Optimal split of model vs. data (Section 3.2)
    - Replace parameter-based proxies `6N1` (non-embedding params) and `6N2` (all params) by `M`. Table 3 shows those proxies can misestimate compute by up to ~50% for small models or long sequences, while `M` tracks actual training flops (Equation (2); Table 3).
    - For eight budgets (1e17–3e20), they evaluate ~10 `M`/`D` allocations each, using the hyperparameters predicted by Equation (1), and fit “IsoFLOP” curves (Figure 4a).
      > Equation (4): Optimal scaling laws are `M_opt = 0.1715 * C^0.5243` and `D_opt = 5.8316 * C^0.4757`  
      Hence model scale grows slightly faster than data scale as compute increases (Figure 4b–c).
    - A performance-vs-compute curve (Figure 5) trained on small budgets accurately predicts outcomes at ~1000× larger compute, matching the final `7B` and `67B` checkpoints (blue stars).

  - Step 3: Data-quality sensitivity (Section 3.3)
    - Fitting the same IsoFLOP method on three datasets (early in-house, current in-house, and OpenWebText2) yields different exponents (Table 4):
      > Table 4: `a` (model exponent) increases from 0.450 → 0.524 → 0.578 as data quality rises; `b` (data exponent) decreases symmetrically.  
      Interpretation: Higher-quality data rewards allocating more compute to model size.

- Alignment (Section 4)
  - Data: 1.5M bilingual instruction examples including 1.2M “helpful” (31% general language, 47% math, 22% code) and 300K “safety” items.
  - Supervised fine-tuning (SFT): `7B` for 4 epochs (LR 1e-5) and `67B` for 2 epochs (LR 5e-6); monitor a “repetition ratio” metric across 3868 prompts. Heavy math SFT can induce repetition in smaller models; a two-stage SFT (all data → then conversational-only) lowers repetition without harming code/math scores (Table 12).
  - Direct Preference Optimization (`DPO`): 1 epoch, LR 5e-6, batch size 512, cosine decay; preference pairs built from model-generated candidates for helpfulness and harmlessness. DPO improves open-ended dialogue quality with little change to benchmark numbers (Appendix Table 17).

## 4. Key Insights and Innovations
- New, more faithful model-scale measure (`M`: non-embedding FLOPs/token)
  - What’s different: Unlike proxies based on parameter counts, `M` includes attention cost and excludes vocabulary-softmax cost (Equation (2)).
  - Why it matters: It reduces error in compute estimation (Table 3) and yields more accurate predictions of large-scale performance (Appendix A.2, Figure 6c vs 6a–b).

- Compute-conditioned hyperparameter laws
  - Novelty: Empirical power laws for near-optimal batch size and learning rate as functions of compute (Equation (1); Figure 3).
  - Significance: Converts hyperparameter tuning for new budgets into a plug-in formula, enabling consistent training across 10^3× compute (Figure 2b, Figure 5).

- Data quality alters optimal compute allocation
  - Discovery: With higher-quality corpora, the optimal exponent shifts toward more model scaling (Table 4).
  - Implication: Investing in data quality (not just volume) unlocks larger, better-performing models under the same token budget.

- Practical scheduler choice enabling restartable training
  - Multi-step LR matches cosine in final loss (Figure 1a) but facilitates “continual training” by reusing the first stage (Figure 1b). This is a pragmatic improvement for long, interrupted runs.

- Depth-first, GQA-enabled 67B design
  - Departing from width expansion common in GQA models, the 67B expands depth (95 layers) to improve performance while keeping inference efficient (Section 2.2; Table 2).

## 5. Experimental Analysis
- Evaluation methodology (Section 5; Appendix A.6)
  - Multiple-choice tasks (e.g., HellaSwag, MMLU, ARC, C‑Eval, CMMLU) use perplexity-based scoring; for ARC/OpenBookQA they apply “unconditional normalization,” and length normalization elsewhere.
  - Generation tasks (e.g., TriviaQA, NQ, DROP, GSM8K, MATH, HumanEval, MBPP, BBH, AGIEval, CLUEWSC, CMath) are evaluated with greedy decoding and task-specific parsers.
  - Language modeling uses BPB on Pile-test.
  - Open-ended ability is judged on Chinese AlignBench and English MT‑Bench using GPT‑4-based rubric scripts with prescribed temperatures (Section 5.2).

- Headline base-model results (Table 5)
  - Against LLaMA‑2 70B, DeepSeek‑67B Base wins broadly, especially in code and math. Selected comparisons:
    > Table 5 (English): HumanEval 42.7 vs 28.7; MBPP 57.4 vs 45.6; GSM8K 63.4 vs 58.4; MATH 18.7 vs 13.5; BBH 68.7 vs 62.9; MMLU 71.3 vs 69.0.  
    > Table 5 (Chinese): CHID 92.1 vs 55.5; C‑Eval 66.1 vs 51.4; CMMLU 70.8 vs 53.1; CMath 63.0 vs 53.9; CCPM 88.5 vs 66.2.  
    > Pile-test BPB 0.642 (lower is better) vs 0.649.
  - Interpretation: The bilingual training plus scaling regimen gives strong reasoning/code math performance and much stronger Chinese knowledge.

- Chat-model results and trade-offs (Table 6)
  - SFT + DPO boosts reasoning and program synthesis substantially:
    > 67B Chat: GSM8K 84.1; MATH 32.6; HumanEval 73.8; DROP 71.9; BBH 71.7.  
  - Some tasks drop post-fine-tuning, especially cloze/sentence-completion style:
    > HellaSwag falls (67B Base 84.0 → 67B Chat 75.7); WinoGrande also dips.  
    Rationale in Section 5.1.2: pure language modeling sometimes suits these tasks better than instruction-following style.

- Open-ended evaluations (Section 5.2)
  - Chinese AlignBench (Table 7):
    > 67B Chat achieves an overall 6.43, improved to 6.69 with DPO; it ranks above ChatGPT (gpt‑3.5‑turbo‑0613, 6.08) and close to older GPT‑4 variants across categories.
  - English MT‑Bench (Table 8):
    > 67B Chat scores 8.35 (comparable to GPT‑3.5‑turbo 8.39); DPO lifts it to 8.76, behind GPT‑4‑1106‑preview (9.26).

- Held-out, recency-sensitive tests (Section 5.3; Table 9)
  - LeetCode contests: 67B Chat pass@1 = 17.5 vs Qwen‑72B Chat 12.7.
  - Hungarian High-School Math: 58 vs Qwen‑72B 52 (human-graded).
  - Instruction Following (IFEval): 55.5 vs Qwen‑72B 50.8; 7B Chat reaches 41.2.

- Safety evaluation (Section 5.4; Tables 10–11)
  - A 2400-prompt human-labeled safety suite across discrimination, legality, IP, and other risks shows high safe-answer rates (e.g., 486/500 on discrimination, 473/500 on legal rights).
  - On the Do‑Not‑Answer dataset, 67B Chat scores 97.8, higher than ChatGPT (97.7) and GPT‑4 (96.5) (Table 11).

- Scaling-law validations
  - Performance-vs-compute prediction trained on small budgets tracks the final 7B and 67B checkpoints (Figure 5), substantiating the use of `M` and the IsoFLOP methodology for forward planning.

- Ablations and diagnostic studies
  - Two-stage SFT reduces repetition without harming code/math and improves instruction following (Table 12).
  - Adding 20M multi-choice (MC) questions during alignment boosts MC benchmarks greatly (e.g., MMLU 49.4 → 60.9; C‑Eval 47.0 → 71.3) but not generative QA (TriviaQA unchanged; Table 13). The team therefore excludes MC-heavy instruction during pretraining/fine-tuning to avoid benchmark overfitting (Section 5.5).
  - System prompt helps large models but can slightly hurt small ones on MT‑Bench (Table 14).
  - DPO’s effect on benchmarks is small (Appendix Table 17), consistent with its goal of improving open-ended dialogue quality.

- Do the experiments support the claims?
  - Yes on three fronts: (1) predictive scaling with `M` and IsoFLOP (Figures 4–5); (2) compute-conditioned hyperparameters producing strong models (Figure 2–3; Table 5–6); (3) quality-weighted allocation between `M` and `D` (Table 4).
  - The paper is transparent about trade-offs: fine-tuning can degrade some cloze-style tasks; MC-heavy alignment can inflate MC benchmarks without helping generative QA (Section 5.5).

## 6. Limitations and Trade-offs
- Assumptions and scope
  - Hyperparameter laws depend only on total compute `C`; the paper notes optimal regions vary slightly with the specific `M`/`D` split and other factors not modeled (Section 3.1). Extending the formulas to include model architecture or optimizer nuances is open.
  - Generalization is measured on an in-house validation set drawn from the same distribution as training (Section 3.2). Out-of-distribution scaling behavior may differ.

- Data and language coverage
  - Pretraining is largely Chinese and English (Conclusion). Performance in other languages is “delicate,” and Chinese data may omit some topics.

- Model constraints
  - Context length is 4096 tokens; long-context scaling is not addressed.
  - Compute and infrastructure needs remain high despite optimizations (Section 2.4); replication requires multi-parallel training with ZeRO, pipeline, flash attention, etc.

- Fine-tuning trade-offs
  - Instruction tuning can depress cloze-style benchmarks (Table 6) and initially increased repetition for smaller models, requiring two-stage SFT (Table 12).
  - MC-heavy data can create benchmark overfitting without improving real conversational ability (Table 13).

- Safety evaluation
  - The 2400-item safety suite and review rubric are internal; while strong on Do‑Not‑Answer (Table 11), broader third-party red-teaming would strengthen claims.

## 7. Implications and Future Directions
- How this changes the field
  - For open-source LLM builders, this paper offers a practical, compute-grounded recipe: use `M` as the scale variable, set hyperparameters via Equation (1), allocate compute via Equation (4), and expect better returns on model size as data quality rises (Table 4). The result is credible planning for large training runs with fewer blind spots.
  - The empirical link between data quality and optimal compute split reframes “scaling laws” as conditional on corpus properties, not universal constants.

- Follow-up research enabled
  - Theory: Derive or simulate why higher-quality data pushes the optimum toward larger `M`; extend the hyperparameter laws to include optimizer schedule shapes, architecture families (e.g., MoE), and longer contexts.
  - Methods: Automate “quality-aware” sampling that targets the `a/b` exponents in Table 4; calibrate `M` for models with sparse routing or custom attention variants.
  - Evaluation: Public, diversified safety suites and held-out reasoning benchmarks; standardized open-ended evaluation beyond GPT‑4 judges.

- Practical applications
  - Organizations training bilingual or multilingual assistants can use the paper’s recipes to plan compute, size models, and pick hyperparameters without expensive pilot runs.
  - The released `67B` chat model, with strong math/code and Chinese performance (Tables 6–8), is suitable for enterprise assistants, coding copilots, math tutoring, and bilingual knowledge services; the safety results (Tables 10–11) suggest readiness for pilot deployments with additional red-teaming.

Overall, DeepSeek LLM contributes both strong models and, crucially, operational scaling guidance. The key takeaways practitioners can immediately use are: measure “model size” by `M` (not parameters), set `B` and `η` from compute using Equation (1), fit IsoFLOP curves early to predict outcomes (Figures 4–5), and invest in data quality to justify larger models (Table 4).
