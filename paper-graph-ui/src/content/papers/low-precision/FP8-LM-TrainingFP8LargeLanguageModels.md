# FP8-LM: Training FP8 Large Language Models

**ArXiv:** [2310.18313](https://arxiv.org/abs/2310.18313)

## 🎯 Pitch

This paper presents FP8-LM, the first end-to-end framework for training large language models (LLMs) entirely in 8-bit floating point precision—not just compute, but also optimizer states and inter-GPU communication. By innovatively addressing the numerical instabilities of FP8, this approach delivers up to 39% lower memory usage and 75% faster training versus standard BF16 methods, all while preserving model quality—making truly massive, cost-efficient LLMs accessible on contemporary GPU hardware.

---

## 1. Executive Summary
This paper introduces an end-to-end FP8 (8‑bit floating point) mixed‑precision framework that trains large language models (LLMs) using 8‑bit numbers not only for computation but also for gradient communication and optimizer states. It achieves large savings in memory and communication while matching the accuracy of BF16 (16‑bit) training, and it accelerates training substantially—up to 75% faster than BF16 for a 175B‑parameter GPT model on H100 GPUs (Table 5), with no changes to typical hyperparameters.

## 2. Context and Motivation
- Problem addressed
  - Training LLMs is extremely costly in compute, memory, and inter-GPU communication. Examples cited include GPT‑3 (175B) and PaLM (540B) requiring thousands of accelerators and large energy footprints (Section 1).
  - Prior mixed‑precision schemes (FP16/FP32, BF16/FP32) reduce cost but still leave significant savings unrealized because gradients, optimizer states, and communications remain high precision in mainstream systems (Section 1, 2).

- Why it matters
  - Reducing precision from 16/32‑bit to FP8 in the full training loop can theoretically yield 2× compute speedup and 50–75% savings in memory and communication (Abstract; Appendix A.1). This enables training larger models or longer contexts on the same hardware (Figure 1; Section 3.2.2).

- Prior approaches and gaps
  - FP16/FP32 mixed precision suffered instability for very large models due to FP16’s limited dynamic range; BF16/FP32 became the standard for LLMs because BF16 has FP32‑like range (Section 2).
  - With Nvidia H100, FP8 became practical; however, Nvidia’s Transformer Engine (TE) uses FP8 mainly for matrix multiplications (GEMMs) while keeping gradients, master weights, and optimizer states in higher precision, so end‑to‑end gains are modest (Section 1; 2).

- Positioning
  - This work targets a full FP8 training stack. It designs:
    - FP8 gradients and FP8 all‑reduce communication with automatic scaling (Section 2.1; Eqs. 1–6).
    - An FP8‑aware optimizer via precision decoupling—8‑bit where safe, 16/32‑bit where needed (Section 2.2; Eqs. 7–8; Table 6).
    - FP8‑compatible distributed parallelism (tensor, pipeline, sequence) and a ZeRO‑style sharding that respects FP8 scaling (Section 2.3; Figure 2–3; Algorithm 1).
  - It demonstrates parity in model accuracy with BF16 for pretraining, SFT, and RLHF, while materially reducing memory and boosting speed (Figures 4–6; Tables 2–5).

## 3. Technical Approach
The framework integrates FP8 across compute, storage, and communication. Below, “FP8” refers to two standardized sub‑formats—E4M3 and E5M2—with different tradeoffs of range and precision (Appendix A.1). Because FP8 has a much narrower dynamic range and fewer mantissa bits than BF16/FP32 (Table 9), the framework relies on careful scaling and selective high‑precision placement.

1) FP8 gradients and FP8 all‑reduce communication (Section 2.1)
- Background definitions
  - `All-reduce`: a distributed operation that sums corresponding elements of tensors across GPUs and makes the sum available to all GPUs.
  - `Underflow/overflow`: values too small/large to be represented in a chosen format; with FP8 this is common if data is not scaled.
  - `Tensor scaling`: multiplying a tensor by a scalar so its values fall within the representable range; in FP8 one typically stores the scaled tensor in FP8 and the scale as a separate factor (Appendix A.2).

- The problem
  - Two standard ways to average gradients across N GPUs:
    - Pre-scaling: divide each local gradient `g_i` by `N` before summing: g = g1/N + ... + gN/N (Eq. 1). This causes underflow in FP8 when N is large.
    - Post-scaling: sum first and divide later: g = (g1 + ... + gN)/N (Eq. 2). This risks overflow during the sum.
  - NCCL (the common communication library) does not natively handle per‑tensor scaling factors during all‑reduce.

- The solution: automatic scaling + shared scaling per gradient tensor
  - Auto‑scaling factor `μ` (Eq. 3): before quantizing to FP8, multiply gradients by a dynamic factor. At each step, measure how many values saturate to FP8’s max. If more than 0.001% saturate, halve `μ` next step to avoid overflow; if saturation stays below threshold, gradually double `μ` over ~1000 steps to reduce underflow.
  - Shared scaling across GPUs (Eqs. 4–6):
    - Each GPU has a local FP8 gradient `g_i'` with its local scale `s_i'`. GPUs first all-gather the per‑tensor scale values (only scalars), compute the global minimum scale `s_g' = min(s_1', ..., s_N')` (Eq. 4), and re‑quantize their gradients to use this shared scale before all‑reduce (Eq. 5).
    - After summation, set the new scale to `s = N * s_g'` (Eq. 6), which is equivalent to averaging the unscaled gradients. This lets standard NCCL handle the all‑reduce (no custom per‑element scale handling), while communicating only one scalar per gradient tensor.

- Why it works
  - Using the global minimum scale avoids overflow across GPUs and keeps all shards on a common quantization grid, enabling a conventional all‑reduce.
  - Auto‑scaling balances the underflow/overflow tradeoff over time without manual tuning.

2) FP8 optimizer with precision decoupling (Section 2.2)
- Background definitions
  - `AdamW` keeps, per parameter, a high‑precision copy of the parameter (`master weight`), the gradient, and two moving averages: the first‑order moment (like a running average of gradients) and the second‑order moment (running average of squared gradients).
  - In mainstream mixed precision, these states are typically FP32 for stability, costing 16 bytes/parameter (Eq. 7).

- Design choices and rationale
  - Keep what is precision‑sensitive high‑precision; reduce where safe:
    - Master weights: FP16 with tensor scaling (or FP32) to reliably apply very small updates (Section 2.2). FP8 here degrades training (ablation in Table 6 & Figure 8).
    - Gradients: FP8 (already handled by the communication scheme in 2.1).
    - First moment (m): FP8 with scaling—its direction matters more than its exact magnitude; tolerates quantization noise.
    - Second moment (v): FP16—squaring small gradients is vulnerable to underflow in FP8 and harms stability/accuracy (Figure 8; FP8 for v diverges).
  - This configuration uses 6 bytes per parameter (Eq. 8), a 2.6× reduction vs. the 16 bytes/param baseline (Eq. 7).

3) FP8‑compatible distributed training (Section 2.3)
- Background definitions
  - `Tensor parallelism`: split large matrix multiplications within a layer across GPUs.
  - `Pipeline parallelism`: split layers across GPUs.
  - `Sequence parallelism`: split the sequence length dimension across GPUs to save activation memory.
  - `ZeRO`: shard optimizer states and gradients across devices to eliminate redundancy.

- FP8 integration
  - Tensor + sequence parallelism with FP8 activations and weights (Figure 2):
    - Convert activations and sharded weights to FP8 at the GEMM boundaries, so forward, backward, and gradient communication run in FP8.
    - Introduce FP8 conversion before the gather/reduce operator `g` that bridges sequence and tensor parallel regions (Figure 2), cutting activation all‑gather/reduce‑scatter communication roughly by a third (Table 7).
  - FP8‑aware ZeRO sharding (Figure 3; Algorithm 1):
    - Challenge: FP8 uses per‑tensor scaling factors; if a tensor is split into chunks, managing scales per chunk complicates communication and correctness.
    - Solution: do not split tensors; instead, greedily assign whole tensors (FP8 array + its scale) to GPUs based on remaining memory (Algorithm 1). This keeps scale/tensor together, simplifies communication, and still balances memory (Table 8).

4) Experimental setup at a glance (Section 3.1; Table 1; Appendix A.3)
- Models: GPT‑style decoder‑only Transformers, sizes 125M, 7B, 13B, 175B; use RoPE positional embeddings and FlashAttention for efficiency.
- Data: A mixture of CommonCrawl, The Pile components, Wikipedia, code, books, etc. (Table 10; Appendix A.3). Instruction tuning uses ShareGPT; RLHF uses Anthropic Helpful/Harmless + OpenAssistant (Section 3.1.1).
- Hardware: Nvidia H100 80GB on Azure NDv5 (Section 3.1.2).
- Training settings: Cosine LR schedule; standard AdamW hyperparameters; sequence length 2048; batch sizes in Table 1. For 175B, they cap to 40B tokens (Table 1) to study system behavior while saving compute/emissions.

## 4. Key Insights and Innovations
1) End‑to‑end FP8 beyond compute: gradients, communication, and optimizer (Sections 2.1–2.3)
- What’s new vs prior FP8 (e.g., Transformer Engine)?
  - TE uses FP8 mainly within GEMM compute but retains FP16/FP32 for gradients, master weights, and communication. This work converts those remaining bottlenecks to FP8 where safe, with targeted high‑precision where necessary (master weights and second moments).
- Why it matters
  - Produces substantial end‑to‑end gains: 28–39% memory reduction vs BF16 for GPT‑7B/13B/175B (Table 5), 63–65% reduction in weight‑related communication (Table 5), and up to 75% throughput gain vs BF16 on a 175B model (Table 5).

2) Automatic gradient scaling for FP8 all‑reduce (Section 2.1; Eqs. 1–6; Figure 7)
- Novelty
  - A simple, hardware‑friendly mechanism that uses a per‑tensor shared scale and a dynamic factor `μ`, avoiding NCCL changes and large overhead while keeping overflow/underflow in check.
- Evidence
  - Across Transformer blocks, auto‑scaling lowers both underflow and overflow compared to pre‑ or post‑scaling alone and improves signal‑to‑noise ratio (Figure 7a–c).

3) Precision decoupling in the optimizer (Section 2.2; Table 6; Figure 8)
- Novelty
  - First moment in FP8 + second moment in FP16 + FP16 master weights (with scaling) strikes the right balance of stability and memory savings.
- Evidence
  - Ablations on GPT‑125M show:
    - FP8 master weights degrade (compare “FP8 #3” with “FP8 #2a/#2b” in Figure 8).
    - FP8 second moment diverges (the “FP8 #4” curve in Figure 8).
    - FP16 master weights with scaling match FP32/BF16 baselines (Figure 8 and Table 6).

4) FP8‑aware distributed parallelism and ZeRO (Section 2.3; Figure 2–3; Table 7–8)
- Distinguishing features
  - FP8 conversions are placed precisely at sequence/tensor parallel boundaries to reduce activation communication without altering model semantics (Figure 2).
  - Whole‑tensor FP8 ZeRO sharding preserves scale factors cleanly (Figure 3), improving memory balance and reducing peak usage (Table 8).
- Impact
  - 34% reduction in activation‑related communication for GPT‑13B and GPT‑175B (Table 7).
  - Lower and more balanced memory use across GPUs (Table 8).

These are more than incremental tweaks: they are the missing glue that makes FP8 viable across the entire training stack.

## 5. Experimental Analysis
- Evaluation design
  - Accuracy: Compare FP8 vs BF16 pretraining loss for GPT‑7B/13B/175B (Figure 4); evaluate zero‑shot downstream performance (Table 2). For SFT and RLHF, compare losses and standard alignment benchmarks (Figures 5–6; Tables 3–4).
  - Systems: Measure GPU memory, throughput, TFLOPS, MFU, and communication volumes under different parallelism settings and micro‑batches (Table 5). Also quantify activation communication (Table 7) and memory balancing in ZeRO (Table 8).
  - Baselines: BF16 training (Megatron‑LM style) and Nvidia TE’s FP8 compute mode (Table 5).

- Main quantitative results
  - Accuracy parity
    - Pretraining loss curves for FP8 and BF16 overlap across scales (Figure 4a–c), with no hyperparameter changes.
    - Zero‑shot downstream accuracy is comparable. Example (Table 2, averages for 7B/13B):
      - GPT‑7B: 58.4 (BF16) vs 58.0 (FP8).
      - GPT‑13B: 61.0 (BF16) vs 60.4 (FP8).
    - SFT (Table 3): similar quality on AlpacaEval (66.15 vs 67.20 win‑rate vs Davinci‑003) and MT‑Bench (5.75 vs 5.70), with FP8 using 14% less GPU memory and 27% higher throughput.
    - RLHF (Table 4): comparable AlpacaEval (72.05 vs 72.42) and MT‑Bench (6.16 vs 6.04), while reducing weights memory by 32% and optimizer states by 62%.
  - System gains
    - Weight‑related communication volume cut by 63–65% across model sizes (Table 5; “Weight‑related Comm. Volume”).
    - Memory savings versus BF16:
      - GPT‑7B: 69.6 GB → 49.4 GB (−29%); GPT‑13B: 68.2 GB → 48.9 GB (−28%); GPT‑175B (micro‑batch 1): 66.1 GB → 40.3 GB (−39%) (Table 5).
    - Speed
      - GPT‑175B on 32 H100s: FP8 reaches 39.3 samples/s at micro‑batch 4 vs 22.4 for BF16 (+75%); and 39.3 vs 28.7 for TE (+37%) (Table 5).
      - Model FLOPs Utilization (MFU) improves to 34.2% for 175B with FP8 micro‑batch 4, outperforming TE’s 24.9% (Table 5).

- Ablations and robustness checks
  - Gradient scaling strategies (Figure 7): auto‑scaling improves SNR while controlling under/overflow compared to pre/post‑scaling.
  - Optimizer precision (Table 6; Figure 8): high‑precision for master weights and the second moment is necessary; otherwise training degrades or diverges.
  - FP8 parallelism reduces activation communication by about one‑third (Table 7).
  - FP8 ZeRO keeps memory well balanced (Table 8).

- Do the experiments support the claims?
  - Yes for system efficiency and short‑to‑mid‑horizon training stability/accuracy. Parity is shown across multiple scales for losses and task metrics (Figure 4; Table 2) and across tuning scenarios (Tables 3–4).
  - Caveat: the 175B experiment is restricted to 40B tokens for cost reasons (Table 1), so very long‑horizon stability (e.g., at trillion‑token scales) is not directly demonstrated.

## 6. Limitations and Trade-offs
- Dependence on hardware and libraries
  - FP8 training assumes access to H100‑class GPUs that natively support FP8 (Appendix A.1). Earlier hardware will not realize these gains.
  - NCCL does not support per‑tensor scale‑aware all‑reduce; the method circumvents this via a shared scalar and re‑quantization (Section 2.1). This is elegant but adds steps and depends on heuristic thresholds (e.g., 0.001% saturation).

- Precision choices are problem‑specific
  - The chosen precision split (FP8 for gradients and first moment; FP16 for second moment and master weights) works well for the tested GPT setups (Section 2.2; Table 6), but other architectures/optimizers might require re‑tuning.

- Training horizon and coverage
  - The 175B model is trained on 40B tokens (Table 1). While sufficient for system benchmarks and early‑phase stability, it does not fully demonstrate end‑to‑end convergence behavior at massive token counts.

- Communication scaling nuances
  - Using the global minimum scale across GPUs (Eq. 4) guarantees safety but may be conservative if one shard has atypically large values, potentially increasing quantization noise elsewhere. Figure 7’s SNR results suggest the trade‑off is acceptable, but behavior under extreme heterogeneity is not deeply explored.

- ZeRO tensor‑as‑a‑whole sharding
  - Keeping tensors whole simplifies scaling but constrains placement granularity. Algorithm 1’s greedy allocator (Section 2.3) balances memory well in reported setups (Table 8), yet more complex cluster topologies might require enhanced placement strategies.

## 7. Implications and Future Directions
- How this changes the landscape
  - Demonstrates that FP8 can be used safely across the entire LLM training stack—not just compute kernels—without accuracy loss and with substantial efficiency gains (Figures 4–6; Tables 2–5). This can lower the barrier to training larger models and/or longer contexts on fixed budgets (Section 3.2.2 and Figure 1).

- What it enables next
  - Systems research: integrate tensor‑scale‑aware primitives directly into collective communication libraries (future NCCL enhancements) to remove remaining overheads.
  - Algorithmic research: explore adaptive/learned scaling schedules beyond the simple threshold rule (Eq. 3), and extend precision‑decoupling ideas to other optimizers (e.g., Adafactor) and architectures (e.g., MoE, diffusion).
  - Even lower precision: given the stability shown with FP8 when carefully engineered, push to 4‑bit training for parts of the stack, building on the same scaling and decoupling principles.

- Practical applications
  - Pretraining: cut training time/cost for very large GPT‑style models.
  - Fine‑tuning and RLHF: reduce memory/compute when multiple models (policy, reference, reward) are loaded concurrently; the reported 32%–62% memory reductions in RLHF (Table 4) are particularly valuable for practitioners.
  - Longer contexts and larger batch sizes on the same hardware footprint (Section 3.2.2): e.g., 175B with 4096 tokens on 32×H100 becomes feasible with this FP8 scheme, whereas TE fits only 2048 (Section 3.2.2).

> Net takeaway: With targeted scaling and precision decoupling, FP8 is not just a compute‑kernel trick—it is a viable, general‑purpose training precision for LLMs that meaningfully reduces cost while preserving quality.
