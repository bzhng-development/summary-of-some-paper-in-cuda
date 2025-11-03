# FAST-DLLM V2: Efficient Block-Diffusion LLM

**ArXiv:** [2509.26328](https://arxiv.org/abs/2509.26328)

## 🎯 Pitch

FAST-DLLM v2 introduces a novel block diffusion language model that transforms any pretrained autoregressive (AR) LLM into a highly efficient, parallel text generator by imbuing it with a carefully designed block-wise attention mask and a two-level (block and sub-block) caching mechanism. This approach dramatically accelerates inference—achieving up to 2.5× speedup—without sacrificing accuracy and requires only a fraction of the training data compared to traditional diffusion models, unlocking practical, low-latency deployment of high-quality LLMs for real-world applications.

---

## 1. Executive Summary (2-3 sentences)
Fast-dLLM v2 converts a standard autoregressive large language model into a block-diffusion generator that can decode many tokens in parallel, while still reusing Key-Value (KV) caches like autoregressive models. With a carefully designed training mask, token-shift objective, and a two-level caching system, it reaches up to 2.5× higher throughput than autoregressive decoding without sacrificing accuracy on a broad benchmark suite (Figures 1, 4, 5; Table 1).

## 2. Context and Motivation
- Problem addressed
  - Autoregressive (AR) LLMs generate one token at a time, making inference inherently sequential and slow (Introduction).
  - Diffusion language models (dLLMs) can predict or refine multiple tokens at once, but practical systems struggle with:
    - Poor compatibility with KV caching due to bidirectional attention,
    - Higher end-to-end latency than AR,
    - Fixed or inflexible sequence lengths,
    - Heavy training data requirements (e.g., Dream uses ~580B tokens) (Introduction; Related Work §2.1–2.3).

- Why it matters
  - Real-time assistants, coding copilots, and batch serving benefit from higher tokens-per-second (latency and throughput).
  - Maintaining AR-level quality while achieving diffusion-level parallelism would enable deployable, faster LLMs.

- Prior approaches and gaps
  - Full-attention dLLMs (e.g., Dream) need massive finetuning data and do not cleanly exploit KV cache (Introduction; §2.1).
  - Approximate caches exist (DualCache, dKV-Cache, dllm-cache, etc.) but they do not fully eliminate recomputation and/or deviate from exact AR computation semantics (§2.3).
  - Block diffusion (e.g., BD3-LM) bridges AR and diffusion by denoising within blocks and moving across blocks left-to-right, but had only been shown at smaller scales (§2.2).

- Positioning
  - Fast-dLLM v2 scales block diffusion to modern LLMs (1.5B and 7B) and introduces an AR-friendly attention mask and hierarchical caching so that:
    - Inter-block behavior is AR and cacheable,
    - Intra-block behavior is bidirectional (diffusion-style) and parallelizable (Method §3.2–3.3; Appendix A.2).
  - It claims data-efficient adaptation from an AR base with ~1B-scale tokens of finetuning, a ~500× reduction vs full attention dLLMs like Dream (Abstract). Appendix A.1 quantifies total tokens processed in their experiments (1.31B–3.15B depending on model size).

## 3. Technical Approach
This section unpacks how Fast-dLLM v2 preserves AR strengths while gaining diffusion parallelism.

- Core setup (Figure 2; §3.2)
  - Start from a pretrained AR model (Qwen2.5-Instruct, 1.5B or 7B).
  - Divide each training sequence of length L into B non-overlapping blocks of size D (they use D=32 by default).
  - Pad sequences so their lengths are multiples of D; padding tokens do not contribute to loss (Block-wise organization).

- Within-block masking with complementary views (Figure 2; §3.2)
  - For each block, randomly mask a subset of tokens with a learned [MASK] embedding.
  - Duplicate the sample to create two “complementary” mask views:
    - View 1 uses mask m (masked tokens),
    - View 2 uses complement 1−m (the previously visible tokens now masked).
  - Benefit: ensures every token is trained as masked and unmasked across the batch, increasing supervision coverage.

- Token-shift prediction (Token shift for prediction; §3.2)
  - Definition: When a token at position i is masked, predict it using the hidden state at position i−1 (one-step shift).
  - Purpose: preserve the AR model’s temporal representation (next-token prediction alignment) while enabling bidirectional denoising within a block.

- Training objective (Equation in §3.2; Appendix A.3)
  - Loss only on masked tokens in each view:
    - ℒ_block(θ) = − E[ Σ_i 1[x_i is [MASK]] · log p_θ(x_i | x_<i, x_block(i)) ].
  - Because both mask and its complement are trained together, the total number of “masked” positions per original sample equals the whole sequence length L, so they do not normalize by the typical “1 / (#masked)” factor (Appendix A.3).

- Attention mask design: AR-friendly block-diffusion (Appendix A.2; Figure 7)
  - During training:
    - Concatenate noised sequence x_t and its targets x_0 along the length dimension, for a total of 2L tokens.
    - Apply a hybrid attention mask ℳ_full with three sub-masks (Figure 7a):
      - Block-diagonal (BD): full bidirectional attention within the same block in x_t (enables intra-block denoising).
      - Offset block-causal (OBC): tokens in x_t can attend to all clean tokens in earlier blocks of x_0 (preserves inter-block causality).
      - Block-causal (BC): x_0 tokens attend causally to their own and previous blocks (standard left-to-right progression).
  - During inference (Figure 7b):
    - Previously decoded clean blocks of x_0 are cached,
    - Only the current noised block x_t^b is computed/updated,
    - That block attends bidirectionally within itself, and causally to cached prefixes—allowing exact KV caching across blocks.

- Hierarchical caching and decoding (Figure 3; §3.3)
  - Block-level KV cache: once a block is finalized, its representations are cached and reused by future blocks (standard AR-style reuse across blocks).
  - Sub-block cache via DualCache:
    - Within the current block, decoding proceeds iteratively and selectively (some tokens may be decoded earlier than others).
    - DualCache keeps both prefix and suffix KV states for the partially decoded block to avoid recomputing the whole block when a few positions change (reused from Fast-dLLM; §3.3 “DualCache for sub-block reuse”).
  - Confidence-aware parallel refinement (Parallel refinement within each block; §3.3)
    - At each iteration, positions whose predicted confidence exceeds a threshold are finalized (unmasked) in parallel.
    - Uncertain positions remain masked for further refinement steps.
    - Speed-quality knob: threshold = 1.0 yields non-parallel baseline; lower thresholds increase throughput but can affect accuracy (Figure 4).

- Batch decoding with padding (end of §3.3)
  - Right-pad sequences with [MASK] to multiples of D.
  - Decode in lockstep across a batch, one block at a time, leveraging GPU parallelism.

- Training protocol and data (Appendix A.1; §4.1)
  - Base models: Qwen2.5-Instruct 1.5B and 7B.
  - Data: LLaMA-Nemotron post-training dataset; block-aligned packing and padding.
  - Hyperparameters:
    - 1.5B: 6,000 steps, LR 2e-5 → ~3.15B tokens processed (256 batch × 2048 context × 6,000).
    - 7B: 2,500 steps, LR 1e-5 → ~1.31B tokens processed.
  - Infrastructure: 64× A100 GPUs, 8–12 hours.

## 4. Key Insights and Innovations
- AR-compatible block diffusion with data-efficient adaptation
  - What’s new: a training+masking scheme that preserves AR causality across blocks while enabling bidirectional attention within blocks (Appendix A.2, Figure 7).
  - Why it matters: enables reusing the KV cache across blocks (a core advantage of AR inference) and requires far less finetuning data than full-attention dLLMs. The abstract highlights a ~500× reduction vs Dream’s 580B tokens; Appendix A.1 shows their runs used ~1.31B (7B) and ~3.15B (1.5B) tokens—still orders-of-magnitude less than 580B.

- Complementary masking plus token-shift objective (Figure 2; §3.2; Table 2)
  - What’s new: two-view complementary masks ensure every token receives both masked and visible supervision; token-shift retains AR-style temporal representations.
  - Impact: ablating these choices reduces performance. Table 2 shows the “+ pad + complementary mask” recipe improves average score from 41.3 (naive shift) to 45.0 on the 1.5B model.

- Hierarchical caching: block-level cache + DualCache inside blocks (Figure 3; §3.3)
  - What’s new: a two-tier cache structure that reuses exact inter-block context and reuses intra-block partial states without full recomputation.
  - Impact: amortizes the cost of bidirectional refinement and supports confidence-aware parallel decoding. Figure 6b shows throughput gains at larger batch sizes when using sub-block cache.

- Confidence-aware parallel decoding (Figure 4; §3.3)
  - What’s new: finalize multiple masked positions in parallel when their confidence exceeds a threshold.
  - Impact: a tunable speed–quality tradeoff. On GSM8K, setting threshold = 0.9 achieves 2.6× throughput (101.7 vs 39.1 tokens/s) with minimal accuracy drop (Figure 4).

Fundamentally, the mask design and training objective align diffusion with AR behavior where it matters (inter-block), unlocking KV cache reuse and practical performance at scale—this is the central innovation beyond incremental speedups.

## 5. Experimental Analysis
- Evaluation methodology (§4.1; Appendix A.4)
  - Models: Fast-dLLM v2 adapted from Qwen2.5-Instruct at 1.5B and 7B.
  - Benchmarks: HumanEval and MBPP (code), GSM8K and MATH (math reasoning), MMLU and GPQA (knowledge), IFEval (instruction following).
  - Tooling: EvalPlus for code; LM-Eval for others.
  - Default inference unless stated: block size = 32, sub-block size = 8, parallel decoding disabled (threshold = 1) (Appendix A.4).

- Main quantitative results
  - Overall performance (Table 1)
    - 1.5B scale:
      - Fast-dLLM v2: Avg 45.0 vs Qwen2.5-1.5B and its FT variant at 44.3.
      - Gains: GSM8K 62.0 (vs 57.0), IFEval 47.0 (vs 41.2), MMLU 55.1 (vs 54.6).
      - Trade-off: MATH 38.1 (down from 46.8).
    - 7B scale:
      - Fast-dLLM v2: Avg 60.3 vs Qwen2.5-7B-FT 59.6; Dream 7B at 57.6.
      - Strong gains in code: HumanEval 63.4 and 58.5 (Plus) vs 52.4 and 48.2 for Qwen2.5-7B-FT; MBPP 63.0 vs 57.1.
      - GSM8K 83.7 (near Qwen2.5-7B-FT 84.1).
      - Trade-offs: MATH 61.6 (down from 72.0 for Qwen2.5-7B-FT); IFEval 61.4 (down from 69.5); MMLU 66.6 (slightly lower than 68.6).
  - Throughput and accuracy trade-offs (Figures 1, 4, 5, 6)
    - Speed–quality at single-sequence level (Figure 4, GSM8K):
      > Threshold = 0.9: throughput 101.7 tokens/s vs non-parallel baseline 39.1 (≈2.6×), with small accuracy reduction.
      > Threshold = 1.0: recovers the non-parallel baseline.
    - End-to-end throughput (Figure 1):
      > “Fast-dLLM v2 (7B) achieves 2.54× higher throughput than Qwen2.5-7B-Instruct while offering comparable accuracy” (Figure 1a).
      > Batch size 1 and 4: Fast-dLLM v2 outperforms all baselines (Figure 1b).
    - Scaling on new hardware (Figure 5):
      > On A100, up to 1.5× higher throughput at batch size 64; on H100, up to 1.8×. Diffusion scales better with batch size.
    - Sub-block granularity and caching (Tables 3–4; Figure 6):
      > Optimal sub-block size at inference is 8 for best average accuracy (Table 3).
      > Changing the global block size at inference to values not used in training degrades performance (Table 4), underscoring the importance of training–inference consistency.
      > Sub-block cache brings notable throughput gains under compute-bound settings (e.g., larger batches) without affecting accuracy (Figure 6a–b).

- Ablations and design validation
  - Complementary masking and padding matter (Table 2): Avg increases from 41.3 → 42.2 (+ pad) → 45.0 (+ pad + CM).
  - Sub-block vs block-size mismatch (Tables 3–4): adjusting sub-block size is safe/tunable; mismatching global block size hurts.

- Do the experiments support the claims?
  - Efficiency: Yes—multiple figures report clear throughput gains, including hardware scaling (Figures 1, 4, 5, 6).
  - Quality: Mixed but competitive—averages meet or exceed AR baselines, with strong code gains; some regressions on MATH and IFEval at 7B (Table 1). Claims of “lossless adaptation” should be read as “on-average competitive,” not uniformly superior across all tasks.

## 6. Limitations and Trade-offs
- Data-efficiency nuance
  - The abstract claims adaptation with “~1B tokens.” Appendix A.1 reports 1.31B tokens for 7B and 3.15B for 1.5B runs. This is still much less than 580B (Dream), but the 1B headline is approximate and varies by model size.

- Task-level trade-offs
  - At 7B, code tasks improve substantially, but MATH and IFEval drop vs AR fine-tunes (Table 1). This suggests the block-diffusion training objective may bias toward certain capabilities.

- Inference consistency requirement
  - Mismatching the global block size at inference degrades performance (Table 4). Systems must keep the same block size used during training, limiting flexibility.

- Complexity and engineering cost
  - Requires specialized attention masks (Appendix A.2), block-aligned packing, and a hierarchical cache (including DualCache) that complicate implementation.
  - Parallel decoding introduces a quality knob (threshold) that must be tuned per task/deployment (Figure 4).

- Scope of validation
  - Base models are Qwen2.5 Instruct variants (1.5B/7B). Generality to larger models, different pretrains, or multilingual/multimodal tasks is not empirically shown here.

- Approximate intra-block reuse
  - While inter-block KV caching is exact, intra-block DualCache is derived from prior work as an efficiency heuristic; subtle distribution shifts inside iterative refinement could affect some tasks (§3.3 “DualCache for sub-block reuse”).

## 7. Implications and Future Directions
- How this changes the landscape
  - Demonstrates that diffusion-style parallelism and AR-style caching can co-exist at LLM scales, breaking the perceived “speed vs cache” trade-off for dLLMs. The attention-mask design (Appendix A.2) and hierarchical caching (Figure 3) are the enabling mechanisms.

- Practical applications
  - Low-latency assistants, code generation, and batched serving on modern accelerators benefit from 1.5–2.5× speedups (Figures 1, 5), especially at higher batch sizes and on newer hardware (H100).

- Follow-up research directions
  - Adaptive block sizing and scheduling:
    - Learn to vary block size by prompt or stage of generation; maintain training–inference consistency via curriculum or multi-size training to avoid the Table 4 degradation.
  - Better math/knowledge retention:
    - Combine block-diffusion training with task-specific objectives (e.g., chain-of-thought, tool use) to recover MATH and IFEval gaps seen in Table 1.
  - Confidence and refinement policies:
    - Automatically tune the confidence threshold per domain/input to optimize the speed–quality frontier (Figure 4).
  - Broader base models and modalities:
    - Validate on larger parameter counts, multilingual corpora, and multimodal LLMs; integrate with retrieval or tool-augmented pipelines.
  - Systems co-design:
    - Kernel-level implementations of the block-diagonal/offset-causal masks and caches (flex-attention is already used; §3.2) for further latency gains.

Block diffusion with AR-compatible masking and caching, as instantiated here, provides a practical path to faster LLMs without wholesale retraining. The reported speedups, balanced with mostly competitive quality, make Fast-dLLM v2 a credible candidate for deployment where inference cost and latency are paramount.
