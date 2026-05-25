# Better & Faster Large Language Models via Multi-token Prediction

**ArXiv:** [2404.19737](https://arxiv.org/abs/2404.19737)

## 🎯 Pitch

This paper introduces a novel training paradigm for large language models—multi-token prediction—where models are trained to predict several future tokens simultaneously at each sequence position using multiple output heads over a shared transformer trunk. This approach dramatically boosts sample efficiency and enables up to 3× faster inference speeds, providing significant gains especially in generative tasks like code synthesis. By fostering stronger long-range reasoning and reducing the gap between training and inference, this technique offers a simple yet powerful way to make state-of-the-art language models both better and more efficient.

---

## 1. Executive Summary
This paper introduces multi-token prediction (MTP): during training, a language model predicts several future tokens at once from each position using multiple output heads that share a single transformer trunk. With a memory- and compute-matched implementation (Section 2, Figure 2), this yields better sample efficiency and substantially faster inference via self-speculative decoding, with the largest gains on generative tasks such as code generation (Figure 3, Table 1, Section 3.2).

## 2. Context and Motivation
- Problem addressed
  - Standard large language models (LLMs) are trained with next-token prediction under teacher forcing: the model is fed the true previous tokens and learns to predict only the immediate next one (Equation (1)). Section 1 and Section 5 argue this focuses the model on very local patterns and exacerbates the mismatch between training (teacher-forced) and inference (free-running autoregressive), making learning less sample-efficient and error-prone over longer horizons.

- Why this matters
  - Practical: Reducing data and compute needed to reach strong capabilities directly lowers cost and environmental impact. Faster inference is a major operational bottleneck for LLM deployments; Section 3.2 shows up to 3× decoding speedups (Table S2) without an extra “draft” model.
  - Scientific: The work probes whether training signals that “look ahead” push models to form better internal mechanisms for long-range reasoning (e.g., induction heads) and algorithmic structure (Section 4).

- Prior approaches and their gaps
  - Denoising/permuted objectives (UL2, XLNet, BART-like; Section 6) introduce non-causal signals but typically train on a small fraction of tokens per sequence (often 15–25%), leaving most positions unused for gradient signal and complicating pure left-to-right generation.
  - ProphetNet (predicting future n-grams) anticipates multiple tokens but with larger residual replication and different factorization (Section 6); not compute-matched to next-token baselines at modern LLM scale.
  - Speculative decoding methods (e.g., Medusa; Section 6) speed up inference by adding heads after pretraining or by using a separate draft model, but they do not investigate the effect of teaching a model to predict multiple future tokens during pretraining itself.
  - Scheduled sampling (Appendix L.1) mitigates exposure bias in sequence models but is ill-suited for language modeling: mixing ground-truth and model tokens during training often produces ungrammatical, incoherent contexts.

- Positioning of this work
  - A simple, compute- and parameter-matched MTP architecture trains transformers to predict n future tokens in parallel from each position using n independent heads and a shared trunk (Section 2, Figure 1). It provides:
    - Better generative performance at scale, especially for code (Figure 3, Table 1).
    - Native support for self-speculative decoding with substantial speedups (Section 3.2, Figure S10, Tables S2–S3).
    - Evidence that MTP encourages induction heads and improved algorithmic generalization (Section 4).

## 3. Technical Approach
Step-by-step overview of how the method works.

- Core idea: predict several future tokens at once
  - At each position t in a sequence, compute a shared hidden representation of the prefix, then predict the next n tokens xt+1 … xt+n in parallel via n separate output heads.
  - Loss formulation (Section 2): Instead of minimizing the cross-entropy for only the next token (Equation (1)), minimize the sum of next-n token cross-entropies (Equation (2)):  
    Ln = −∑t log Pθ(xt+n: t+1 | xt:1)  
    With independence across heads conditioned on the shared representation, this becomes a sum over i=1..n of −log Pθ(xt+i | zt:1).

- Architecture (Figure 1, Section 2)
  - Shared trunk `fs`: a standard transformer that encodes the observed prefix xt:1 into a latent representation `z`.
  - n prediction heads `fhi`: each is one transformer layer (Section 2; Appendix B explores alternatives). Heads take `z` and produce logits for a specific offset i.
  - Shared unembedding `fu`: a single vocabulary projection matrix maps head outputs to token logits for efficiency and consistency.
  - Compute-matched design: when adding n−1 head layers, remove n−1 layers from the trunk so total parameter count stays constant (Section 3, “To allow fair comparisons…”; Table S14 lists sizes).

- Memory-efficient training (Figure 2)
  - Challenge: naively materializing logits and gradients for n heads multiplies memory by n (O(nV) with vocabulary size V).
  - Solution: run the trunk forward once; then, for each head in sequence, run forward and immediately backpropagate that head’s loss, accumulating gradients at the trunk and freeing the head’s logits before moving to the next head.  
    Result: reduces peak memory from O(nV + d) to O(V + d), where d is hidden size, with no runtime penalty in principle (Section 2; Table S5 shows only small overhead from an FSDP implementation detail).

- Inference paths (Section 2)
  - Standard decoding: use only the head for xt+1 to generate tokens autoregressively.
  - Fast decoding via self-speculative decoding: use the extra heads to propose multiple future tokens from the current state, then verify/correct them using the next-token head in a blockwise scheme (no separate draft model needed; Section 3.2). This is a specific instantiation of blockwise/Medusa-style decoding (Stern et al., 2018; Cai et al., 2024).

- Why this might help (Section 5; intuitions)
  - Lookahead emphasizes consequential decisions (“choice points”): If a token constrains many future tokens, predicting multiple steps ahead reweights the loss to focus on those tokens (Figure 9; Appendix L.3 quantifies the implicit weight increase by a factor of (n+1)/2 at choice points).
  - Information-theoretic view: With two-step prediction, the mutual information between the next token X and the second-next Y receives double weight in the training signal (Section 5.2; Appendix L.2), encouraging features that matter for future continuation rather than only local correctness.

- Variants explored (Appendix B)
  - Head types: transformer vs. linear probing; causal or anti-causal stacking of heads. Parallel transformer heads perform best and most consistently (Table S4).
  - Byte-level language modeling: MTP extends to predicting multiple future bytes (Section 3.3), with large speed and accuracy benefits.

## 4. Key Insights and Innovations
- An extremely simple, compute-matched MTP architecture at LLM scale
  - What’s new: n independent prediction heads atop a shared trunk, with a sequential backward trick to keep memory flat (Figure 2).
  - Why it matters: Prior multi-token ideas either added heavy replication or were not compute-matched; this design scales to 13B parameters and beyond with standard training stacks (Section 3.1, Table S13/S14).

- Training-time MTP unlocks inference-time speedups without a draft model
  - What’s new: Heads are accurate from pretraining (not just finetuning), enabling blockwise self-speculation that accepts multiple tokens per verification step (Section 3.2).
  - Why it matters: On code, a 4-head model achieves ~3× speedup with ~3.5 tokens retrieved per forward pass (Table S2; Figure S10). With 8–32 heads at the byte level, speedups reach 6.4–10.8× (Table S3).

- Strong gains on generative/code tasks, especially as models scale
  - What’s new: At equal parameter counts, MTP consistently beats next-token training for larger models on code benchmarks (Figure 3; Table 1).
  - Why it matters: Improvements are largest for pass@100, indicating both better solution quality and sampling diversity—crucial for practical code synthesis workflows.

- Multi-byte prediction makes byte-level training competitive
  - What’s new: Predict 8 future bytes at once and recover much of the tokenization gap (Section 3.3, Table 1).
  - Why it matters: Byte-level models avoid tokenizer design and can be universal across languages; MTP makes them far more practical by boosting accuracy and offsetting longer sequence lengths with faster decoding.

- Mechanistic benefits on synthetic reasoning
  - What’s new: Clear formation of induction capability in small models and improved out-of-distribution algorithmic generalization (Section 4.1–4.2; Figures 7–8).
  - Why it matters: Suggests MTP trains internal mechanisms for transferring information across positions (induction heads) and composing operations—capabilities that underwrite long-horizon reasoning.

## 5. Experimental Analysis
- Evaluation setup
  - Compute-matched comparisons: When adding heads (n > 1), the trunk is shortened by n−1 layers so total parameters are equal to the n=1 baseline (Section 3, Table S14).
  - Training data
    - Code models: up to 1T tokens; core comparisons at 200B tokens (0.8 epochs) and 1T tokens (4 epochs) with a 32k tokenizer (Table 1; Section 3.4–3.6).
    - Byte-level code models: 314B bytes (≈116B tokens) with n ∈ {8,16,32} (Section 3.3, Table 1).
    - Natural language: 200B and 500B tokens for 7B models (Section 3.7).
    - Synthetic: children’s stories with renamed entities for induction (Section 4.1); polynomial arithmetic over F7[X]/(X^5) for algorithmic reasoning (Section 4.2).
  - Metrics
    - Code: pass@k on MBPP, HumanEval, APPS/Intro using the standard unbiased estimator with 200 samples per problem and temperature sweeps; oracle temperatures reported in Table S12.
    - NLP: average accuracy on multiple-choice benchmarks (Figure 5, Figure S12). For summarization, ROUGE-1/2/3/L precision/recall/F1 (Figure 6; Tables S8–S10). For GSM8K, pass@k across temperatures (Figure S13).
  - Inference speed: self-speculative decoding throughput/latency relative to standard decoding (Figure S10; Tables S2–S3).

- Main quantitative results
  - Scaling on code (Figure 3; Table S7)
    - Gains grow with model size. For 13B models trained on ~209B tokens of code, MTP (n=4) improves HumanEval pass@100 from 56.0 to 63.5 (+7.5 abs) and MBPP pass@1 from 26.0 to 30.5 (+4.5 abs).
    - For small models (≤0.6B), MTP can underperform, then crosses over to outperform at larger scales (Figure 3).
  - Best n on 7B code models at 200B tokens (Table 1, “200B tokens, 32k tokens”)
    - MBPP: pass@1 improves from 30.0 (n=1) to 33.8 (n=4).  
      > “MBPP pass@1: 30.0 → 33.8; pass@100: 73.7 → 76.9.”  
    - HumanEval: pass@1 improves from 22.8 (n=1) to 24.0 (n=4); pass@100 from 62.0 to 66.1.
    - APPS/Intro is mixed: n=6 performs best (e.g., pass@100: 17.4 → 22.7). This suggests the optimal lookahead window depends on data/task distribution (Section 3.4).
  - Longer training on code (1T tokens; Table 1)
    - MBPP: pass@1 improves from 40.7 (n=1) to 43.1 (n=4).
    - HumanEval: pass@100 improves 83.0 → 86.2 with n=4.
    - Benefits persist across multiple epochs, though the gap narrows (Section 3.5).
  - Byte-level code (Section 3.3; Table 1)
    - With 314B bytes, an 8-byte predictor lifts MBPP pass@1 from 19.3 to 32.3 and HumanEval pass@1 from 18.1 to 21.8.  
      > “8-byte prediction model solves 67% more MBPP problems and 20% more HumanEval problems on pass@1 than next-byte.”
    - Decoding speedups up to 6.39× with 8 heads and up to 10.84× with 32 heads (Table S3).
  - Inference speed (Section 3.2; Figure S10; Table S2)
    - On code with 7B models and 4 heads, relative throughput ≈3.05× with ≈3.50 tokens retrieved per forward. Speedup is consistent across batch sizes (Figure S10).
  - Finetuning on CodeContests (Section 3.6; Figure 4)
    - A 4-token predictor pretrained model, when finetuned either with `n′=4` or with standard next-token (`n′=1`), outperforms a baseline pretrained with next-token across pass@k for all k.  
      > “Next-token finetuning on top of 4-token pretraining appears best overall.” (Figure 4)
  - Natural language results (Section 3.7)
    - Multiple-choice benchmarks: 7B models trained on 200B tokens—`n=2` matches the baseline; `n=4` slightly regresses (Figure 5; Figure S12).
    - Summarization: average ROUGE-L F1 improves for both `n=2` and `n=4`.  
      > “Average ROUGE-L F1: +0.51 (n=2) and +0.46 (n=4) at 200B; +0.28 (n=2) and +0.31 (n=4) at 500B” (Table S9; Figure 6).
    - GSM8K (8-shot): at 200B tokens, `n=2` is clearly better across temperatures; at 500B, the baseline catches up and `n=4` remains worse (Figure S13).

- Ablations and robustness
  - Head architectures: causal/anti-causal/linear variants improve over baseline in some cases but are less consistent than parallel transformer heads (Table S4).
  - Induction capability forms earlier with MTP in small models, but the advantage disappears once models are large enough or trained on higher-quality data that induces induction heads anyway (Figure 7; Figure S14).
  - Algorithmic reasoning: MTP beats increasing model size from 30M to 100M parameters in improving generalization (Figure 8; Figure S16). Adding “pause tokens” does not change the relative advantage (Figure S15).

- Do the experiments support the claims?
  - Yes for the core claims: (1) MTP improves generative performance at scale—especially for code—and (2) it enables significant, robust self-speculative speedups. Evidence spans multiple model sizes, data scales (200B → 1T), and includes synthetic mechanism probes (induction/algorithmic tasks).
  - Mixed for natural language multiple-choice/likelihood tasks: benefits are small or negative at 7B/200B, while summarization improves, suggesting MTP primarily aids generation-focused objectives.

## 6. Limitations and Trade-offs
- Where MTP helps most—and where it doesn’t
  - Strongest gains are on generative tasks (code, summarization) and in settings that benefit from better long-horizon planning. Multiple-choice accuracy shows little gain or slight regression at 7B/200B (Figure 5).
  - Benefits increase with model size; small models can underperform (Figure 3; Table S7), implying limited utility in very low-parameter regimes.

- Choosing `n` is task-dependent
  - On tokenized code, `n=4` works best on average (Table 1, Section 3.4). On APPS/Intro, `n=6` is better; on bytes, larger `n` (e.g., 8) is consistently best. There is no universal `n`, and suboptimal choices can hurt results.

- Compute and training dynamics
  - While the method is designed for no runtime overhead in principle, the specific FSDP implementation loses some overlap, causing a small slowdown (up to ~1.22× at 0.3B; Table S5). This is an engineering artifact that can be fixed.
  - Compute matching removes layers from the trunk to add heads. For tasks relying heavily on deep trunk representations (e.g., some comprehension benchmarks), this trade could be unfavorable.

- Evaluation details to keep in mind
  - Code results use temperature oracles per metric and dataset (Table S12). This is standard but can slightly inflate each model’s best-reported numbers; still, comparisons are fair because all models get the same oracle.

- Not addressed
  - No results beyond 13B parameters or with instruction-tuned / RLHF models; unclear how MTP interacts with alignment and preference optimization.
  - No exploration of mixture-of-experts with MTP heads, or dynamic per-token head usage at training time.

## 7. Implications and Future Directions
- Practical impact
  - Training-time MTP is a low-friction modification that yields better generative models and faster inference without adding a separate draft model. It is particularly compelling for code assistants and services that rely on fast, high-quality generation.
  - Byte-level modeling becomes viable: MTP absorbs much of the performance penalty from longer byte sequences and then recovers the inference cost with large speedups (Section 3.3), enabling universal tokenization strategies.

- Research directions
  - Adaptive or learned lookahead: Automatically choosing or scheduling `n` during training (Section 7 suggests loss balancing, e.g., learned scales as in Défossez et al., 2022).
  - Vocabulary and tokenization co-design: Optimal vocabulary size for MTP may differ from next-token training (Section 7). Jointly tuning vocabulary and `n` could improve compute-per-byte trade-offs.
  - Auxiliary targets in embedding space: Predicting future embeddings or compressed signals (Section 7; LeCun, 2022) might yield stronger long-horizon features with fewer parameters per head.
  - Integration with alignment: Study how MTP interacts with instruction tuning, RLHF, and tool use; investigate whether MTP-trained models are more robust to exposure bias during long chain-of-thought generation.
  - Mechanistic interpretability: Use the synthetic results (Section 4) as a starting point to locate and characterize “lookahead circuits,” induction heads, and choice-point detectors formed under MTP.

- Conceptual takeaway
  - Giving the model a training signal that explicitly cares about what comes next—not just the very next token—rebalances learning toward decisions that shape future text. This reduces the training–inference mismatch and leads to both better generative performance and faster decoding, especially in settings where long-range coherence and planning matter (Sections 3 and 5).
