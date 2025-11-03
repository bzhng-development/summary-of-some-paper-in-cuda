# Better & Faster Large Language Models via Multi‑token Prediction

**ArXiv:** [2404.19737](https://arxiv.org/abs/2404.19737)
**Authors:** Fabian Gloeckle, Badr Youbi Idrissi, Baptiste Rozière, David Lopez‑Paz, Gabriel Synnaeve
**Institutions:** Not specified on arXiv

## 🎯 Pitch

This paper introduces Multi-Token Prediction (MTP), a groundbreaking approach that allows a language model's architecture to predict multiple future tokens at each step using lightweight output heads. By enhancing sample efficiency and enabling significantly faster inference, MTP promises to lower the costs and latency of real-world LLM applications, while advancing our understanding of long-range decision-making within generative tasks.

---

## 1. Executive Summary (2-3 sentences)
This paper introduces multi-token prediction (MTP): training a language model to predict n future tokens at each position using n lightweight output heads on top of a shared transformer trunk. Across strong empirical studies—especially in code generation—the approach improves sample efficiency and enables 2–6× faster inference via self-speculative decoding, while keeping training compute essentially unchanged (Sections 2–3; Figures 1–3; Tables 1, S2–S5).

## 2. Context and Motivation
- Problem addressed
  - Most large language models (LLMs) are trained with next-token prediction (NTP): at each position, the model predicts only the immediate next token. This “teacher-forced” setup tends to overfit local patterns and underemphasize long-range decisions that determine how a continuation unfolds (Introduction; Section 5.1).
  - Result: inefficiency. Achieving strong generative abilities requires enormous data and compute, and inference is slow because tokens are generated one at a time.

- Why it matters
  - Practical impact: Faster inference and better sample efficiency lower costs and latency for real-world LLM applications (Section 3.2).
  - Scientific significance: Provides evidence that modest changes to the training objective can reweight learning toward “choice points,” improving reasoning and planning (Sections 5.1–5.2).

- Prior approaches and shortfalls
  - Multi-task corruption/masking (e.g., UL2, XLNet, UniLM) mixes denoising tasks, but typically backpropagates through only ~15–25% of tokens and does not directly encourage lookahead planning (Related Work).
  - Multi-token heads have been proposed for faster decoding (e.g., blockwise parallel decoding; Medusa) but mainly as finetuning/inference tricks, not as a pretraining loss to change what the model learns (Related Work).
  - ProphetNet predicts future n-grams but replicates large parts of the model, complicating compute-matched comparisons (Related Work).

- Positioning
  - This work pretrains with MTP end-to-end, at scale (up to 13B parameters and 1T tokens), with a compute-matched architecture that reuses a shared trunk and adds small per-step heads (Figure 1; Section 2). It contributes memory-efficient training, systematic scaling studies, byte-level experiments, and a theory-tinged rationale for why MTP helps.

## 3. Technical Approach
Step-by-step overview

- Objective: predict multiple future tokens
  - In NTP, the loss is the sum over positions of the cross-entropy for the next token:
    - Equation (1): L1 = − Σt log Pθ(xt+1 | xt:1).
  - In MTP, the loss expands to the next n tokens at each position:
    - Equation (2): Ln = − Σt log Pθ(xt+n:t+1 | xt:1).
  - Under a shared-trunk-with-heads factorization, this becomes a sum over i = 1..n of parallel next-token-like losses:
    - Ln = − Σt Σi log Pθ(xt+i | zt:1) • Pθ(zt:1 | xt:1), where zt:1 is the shared trunk representation (Section 2).

- Architecture: shared trunk + n independent output heads
  - The model computes z = fs(xt:1) with the shared transformer trunk, then applies n small head stacks fhi and a shared unembedding fu to produce logits for xt+i (Section 2; Figure 1 top).
  - Design choice: heads are lightweight transformer layers, not duplicate trunks, allowing compute-matched comparisons: when adding n−1 head layers, they remove n−1 layers from the trunk to keep total parameters fixed (Section 3 preface; Table S14).

- Memory-efficient training
  - Challenge: logits dominate memory because vocabulary size V ≫ hidden size d. Naively materializing n heads’ logits and gradients would scale memory as O(nV + d).
  - Solution: sequential head backprop. After computing the trunk once, the training loop runs the forward+backward for each head in sequence, accumulates trunk gradients, and frees per-head logits before moving on (Figure 2). This keeps peak memory at O(V + d) with no inherent runtime cost (Section 2). Measured overhead from framework choices is small (1.02–1.09×) and attributed to FSDP overlap loss, not the method itself (Table S5).

- Inference pathways
  - Standard path: use only the next-token head for normal autoregressive decoding (Section 2).
  - Fast path: self-speculative decoding (blockwise parallel decoding; Stern et al. 2018). The extra heads propose several future tokens in one go; the main head then verifies them. Because the heads were pretrained to be good future predictors, acceptance rates are high, yielding substantial speedups (Section 2; Section 3.2; Tables S2–S3; Figure S10).

- Byte-level variant
  - The method also trains models at byte-level tokenization (vocabulary of 256 bytes), where sequences are longer. For these runs, they use a replicated-unembedding variant (Appendix B) due to implementation convenience (Table S13).

- Fairness of comparisons
  - Across experiments, models are compute/parameter matched: adding head layers is offset by removing the same number from the trunk (Section 3 preface; Table S14).
  - Hyperparameters and training steps are carefully controlled (Table S13).

Intuition via a simple example
- Imagine a code file where a single line chooses a library (`import numpy as np`). That choice constrains many later tokens (APIs, variable names). MTP forces the model to optimize not only the immediate token but the joint likelihood of this decision and the next n−1 tokens. This increases the effective training weight on such consequential choices (Section 5.1; Figure 9 and Appendix L.3).

## 4. Key Insights and Innovations
- Training-time multi-token prediction as an auxiliary loss that emphasizes “choice points”
  - Novelty: pretraining with n-head lookahead on a shared trunk (Figure 1), not just finetuning for speed. The method changes what the model learns: it upweights positions whose choices constrain future text (Section 5.1; Figure 9). The paper quantifies that, for n-token prediction, choice points receive n(n+1)/2 effective loss terms vs. n for inconsequential transitions (Appendix L.3).

- Memory- and compute-matched implementation
  - Innovation: sequential head backprop reduces peak memory from O(nV + d) to O(V + d) (Section 2; Figure 2). This enables plugging MTP into standard training pipelines without sacrificing batch size.

- Self-speculative decoding without a separate draft model
  - By pretraining the extra heads to be future predictors, the model can internally “propose and verify” multiple tokens per step. This yields 2.7–3.1× speedups with 4 heads on text/code (Table S2; Figure S10) and up to ~6.4× on byte-level code with 8 heads (Table S3; Section 3.3).

- Evidence for improved induction and algorithmic reasoning at small-to-mid scales
  - The paper links MTP to earlier formation of “induction heads”—patterns that copy or continue recent sequences (Section 4.1; Figure 7)—and to better generalization on a polynomial arithmetic task (Section 4.2; Figure 8), with performance gains larger than tripling model size (Figure S16).

- Making byte-level modeling practical
  - With n=8 heads, byte-level models approach token-based performance despite 1.7× less data, and gain large inference speedups (Section 3.3; Table 1; Table S3).

## 5. Experimental Analysis
Evaluation design

- Datasets and settings
  - Code pretraining: up to 1T tokens; scaling study uses ≥91B tokens (Section 3.1). Evaluations on MBPP, HumanEval, and APPS/Intro (Table 1).
  - Natural language pretraining: 200B and 500B tokens; evaluated on multiple choice benchmarks (ARC, COPA, HellaSwag, NQ, PIQA, SIQA, TriviaQA; Figure 5 and Appendix G), abstractive summarization (eight datasets; Figure 6; Tables S8–S10), and GSM8K math reasoning (Figure S13).
  - Byte-level pretraining: 314B bytes (~116B tokens equivalent) of code (Section 3.3; Table 1).
  - Synthetic tasks: induction capability (children’s stories with two-token names; Section 4.1; Figure 7) and polynomial arithmetic over F7[X]/(X^5) with 1–10 operations (Section 4.2; Figure 8).
  - Finetuning: CodeContests (Python subset) with reward annotations (Section 3.6; Figure 4).
  - Inference speed: blockwise self-speculative decoding (xFormers) on text and code test sets (Section 3.2; Table S2; Figure S10). Speedups also reported for byte-level (Table S3).

- Metrics
  - Code: `pass@k`—probability that at least one of k samples solves the task, estimated from 200–1000 samples per task (Table 1; Figure 4). They also report temperature-swept “oracle” pass@k (Table S12; Figure 4).
  - Summarization: ROUGE-1/2/3/L F1, precision, recall (Figure 6; Tables S8–S10).
  - Multiple choice: accuracy (Figure 5; Figure S12).
  - Math (GSM8K): `pass@k` over temperatures (Figure S13).
  - Speed: relative throughput/latency vs. standard decoding (Table S2; Figure S10).
  - Induction: accuracy on the second token of seen names (Figure 7).

Main quantitative results

- Scaling behavior on code
  - For small models (≤1.3B), MTP can underperform NTP, but at larger scales it consistently wins (Figure 3).
  - Example at 13B (MBPP pass@1): 
    > “n=4 matches or surpasses baseline at all k; HumanEval pass@100 improves from 56.0 (n=1) to 63.5 (n=4)” (Figure 3; Table S7).
  - Across sizes, improvements grow with scale; this helps explain why MTP’s benefits were previously overlooked (Section 3.1).

- Which `n` works best on code?
  - For 7B models trained on 200B tokens (32k tokenizer), n=4 is best on MBPP and HumanEval at pass@1/10/100 (Table 1).
    > MBPP pass@1: 33.8 (n=4) vs. 30.0 (n=1). HumanEval pass@1: 24.0 (n=4) vs. 22.8 (n=1).
  - On APPS/Intro, n=6 performs best on average (Table 1), suggesting the optimal window depends on data distribution (Section 3.4).

- Multi-epoch robustness
  - Training for 4 epochs on 1T tokens keeps advantages, though reduced:
    > MBPP pass@1: 40.7 (n=1) → 43.1 (n=4); HumanEval pass@100: 83.0 (n=1) → 86.2 (n=4) (Table 1; Section 3.5).

- Byte-level code
  - With 8-byte prediction on 314B bytes:
    > MBPP pass@1 jumps from 19.3 (n=1) to 32.3 (n=8); HumanEval pass@1 from 18.1 to 21.8 (Table 1).
  - Self-speculative decoding speedups reach ~6.4× with 8 heads (Table S3; Section 3.3).

- Inference speedups (token-level)
  - With 4 heads on text and code:
    > Relative speedup ~2.7–3.1×; tokens-per-forward ~3.1–3.5 (Table S2; Figure S10). Section 3.2 reports “3.0× on code, 2.7× on text,” with ~2.5 accepted tokens out of 3 suggestions on code.

- CodeContests finetuning
  - Pretraining with n=4 helps regardless of finetuning loss. The best option is pretrain n=4, finetune with standard next-token loss (n′=1):
    > “Both finetunings of the n=4 model outperform the n=1 baseline across k; n=4→n′=1 is best overall” (Figure 4).

- Natural language: mixed results
  - Multiple-choice/NLL tasks (7B, 200B tokens): n=2 is on par with baseline; n=4 slightly worse (Figure 5; Figure S12).
  - Summarization (after finetuning): both n=2 and n=4 beat baseline in average ROUGE-L F1 at 200B and 500B tokens:
    > Average ROUGE-L F1: +0.51 (n=2) and +0.46 (n=4) at 200B; +0.28 (n=2) and +0.31 (n=4) at 500B (Table S9; Figure 6; Table S10).
  - GSM8K (8-shot): at 200B, n=2 > n=1; at 500B, the baseline catches up and n=4 remains worse (Figure S13).

- Synthetic tasks: induction and arithmetic
  - Induction (two-token names): MTP (n=2) improves accuracy for models up to ~30M params; the gap disappears by ~100M (Figure 7). With higher-quality data mixes, induction forms early for all models, removing MTP’s edge (Figure S14).
  - Polynomial arithmetic: MTP improves both in-domain (≤5 operations) and out-of-domain (>5) accuracy (Figure 8). Gains outweigh those from tripling model size (Figure S16). Benefits persist with pause tokens (Figure S15).

Do the experiments support the claims?
- Yes, for the core claims:
  - Performance: consistent gains in code generation at scale (Figure 3; Table 1).
  - Speed: strong, consistent self-speculative decoding speedups at 2–3× (Table S2; Figure S10) and up to ~6× byte-level (Table S3).
  - Mechanistic links: synthetic tasks show earlier induction and better algorithmic generalization (Figures 7–8, S16).
- Caveats:
  - Gains on multiple-choice NLP tasks are not observed at 7B; effects may require larger models or are task-specific (Figure 5; Appendix G).
  - Optimal n is data/task dependent (Section 3.4; Table 1).

Ablations and alternatives
- Alternative head architectures (linear, causal, anticausal) yield similar outcomes; the simple parallel-head design is competitive (Appendix B; Table S4).
- Finetuning Llama 2 with MTP did not help much, suggesting that changing the loss late may be destabilizing (Appendix D; Table S6).

## 6. Limitations and Trade-offs
- When MTP helps—and when it doesn’t
  - Small models can underperform next-token baselines (Figure 3; Table S7). Benefits emerge “at scale.”
  - On multiple-choice benchmarks at 7B/200B tokens, MTP brings no gains and can slightly regress (Figure 5; Appendix G).
  - On GSM8K, advantages at 200B shrink at 500B, suggesting some benefits are data-regime dependent (Figure S13).

- Choice of `n`
  - The optimal number of predicted tokens depends on domain and tokenizer (Section 3.4):
    - Code (32k tokenizer): n=4 best overall (Table 1).
    - APPS/Intro: n=6 best (Table 1).
    - Byte-level: n=8 most consistent (Table 1; Section 3.3).
  - This introduces a tuning dimension and potential overfitting to evaluation suites.

- Compute and implementation considerations
  - Training overhead is small but non-zero in their implementation due to FSDP overlap loss (1.02–1.09×; Table S5).
  - The approach requires careful training loop engineering (sequential head backprop) and small architectural changes (extra heads).

- Task coverage
  - The study mainly targets generative tasks (code, summarization). Improvements are less clear on discriminative/MCQ evaluations (Section 3.7).

- Theoretical explanation is suggestive, not definitive
  - The “choice point weighting” and “mutual information” arguments (Sections 5.1–5.2; Appendix L.2–L.3) are intuitive and informative but not formal proofs of generalization improvements.

## 7. Implications and Future Directions
- How this changes the landscape
  - MTP offers a simple, drop-in modification to pretraining that:
    - Improves generative quality and diversity on code tasks at scale (Figure 3; Table 1).
    - Enables self-speculative decoding without external draft models, making multi-token-per-step generation practical (Section 3.2; Tables S2–S3).
    - Makes byte-level LLMs more viable by compensating for longer sequences with large speedups and better learning signals (Section 3.3).

- What to explore next
  - Adaptive `n` or loss balancing: automatically adjust the weight/number of future tokens, possibly with dynamic schedules or techniques like loss scaling (Conclusion; Section 7).
  - Tokenizer co-design: revisit vocabulary size choices for MTP vs. NTP to optimize sequence lengths and compute-per-token trade-offs (Conclusion).
  - Embedding-space auxiliary losses: predict future features instead of hard tokens (Conclusion; LeCun 2022 inspiration).
  - Larger natural language models: test whether MTP’s gains on generative NLP tasks grow with scale beyond 7B (Figure 5 hints that size may matter).
  - Integration with reasoning scaffolds: combine MTP with pause tokens, tool-use, or planning structures to further emphasize long-horizon decisions (Appendix K).

- Practical applications
  - Code assistants and IDE copilots: higher pass@k and much faster decoding directly improve developer experience (Table 1; Table S2).
  - High-throughput generation: multi-document summarization and other sequence generation workloads benefit from both improved ROUGE and 3× decoding speed (Figure 6; Table S2).
  - Byte-level domains: domains with rich symbol sets (e.g., logs, binaries, multilingual scripts) can leverage byte-level MTP for unified modeling (Section 3.3).

---

Selected mechanisms and rationale (grounded in paper content)
- Why does MTP emphasize “choice points”? Because future tokens that depend on a decision are harder to predict if the decision is wrong. Counting loss terms shows that, for n heads, a consequential choice contributes n(n+1)/2 related loss terms vs. n for inconsequential choices (Figure 9; Appendix L.3).
- Why can it improve long-range reasoning? The information-theoretic decomposition shows that 2-token prediction increases the effective weight of the mutual information between consecutive tokens, I(X;Y), by a factor of 2 in the loss (Section 5.2). This encourages learning features that connect current decisions to their downstream implications.
- How is training kept memory-efficient? By sequentially backpropagating through each head and freeing logits between heads, peak memory scales with O(V + d), not O(nV + d) (Figure 2; Section 2).
- Why does inference get faster? Because the extra heads propose several plausible future tokens, many of which the main head accepts, reducing the number of forward passes needed to emit L tokens (Section 3.2; Tables S2–S3; Figure S10).

> Representative results
> - Code, 7B/200B tokens, 32k tokenizer: MBPP pass@1 improves 30.0 → 33.8 and HumanEval pass@1 improves 22.8 → 24.0 with n=4 (Table 1).
> - Self-speculative decoding: 4-head models reach ~3× throughput vs. autoregressive decoding at similar batch sizes (Table S2; Figure S10).
> - Byte-level, 7B/314B bytes: MBPP pass@1 improves 19.3 → 32.3 with n=8, while speedups reach ~6.4× with 8 heads (Table 1; Table S3).
> - Summarization: average ROUGE-L F1 improves by +0.51 (n=2) at 200B and +0.28 (n=2) at 500B after finetuning (Table S9; Figure 6).
> - Induction and arithmetic: substantial gains for small-to-mid models; MTP’s effect exceeds that of tripling parameter count on the arithmetic task (Figures 7–8; Figure S16).
