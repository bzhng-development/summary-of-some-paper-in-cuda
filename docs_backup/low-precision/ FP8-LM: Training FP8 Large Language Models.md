# FP8-LM: Training FP8 Large Language Models

**ArXiv:** [2310.18313](https://arxiv.org/abs/2310.18313)
**Authors:** Houwen Peng, Kan Wu, Yixuan Wei, Guoshuai Zhao, Yuxiang Yang, Ze Liu, Yifan Xiong, Ziyue Yang, Bolin Ni, Jingcheng Hu, Ruihang Li, Miaosen Zhang, Chen Li, Jia Ning, Ruizhe Wang, Zheng Zhang, Shuguang Liu, Joe Chau, Han Hu, Peng Cheng
**Institutions:** Microsoft (Azure)

## 🎯 Pitch

This paper presents a groundbreaking FP8 automatic mixed-precision training framework that extends beyond matrix multiplies to include gradients, optimizer states, and distributed communication for large language models. Utilizing Nvidia H100 GPUs, the framework achieves up to 75% increased throughput and a 39% reduction in real GPU memory usage compared to BF16, all while maintaining model accuracy. These enhancements not only cut computational and memory costs but also pave the way for training larger models and longer contexts on fixed hardware, revolutionizing efficiency in language model training.

---

## 1. Executive Summary
This paper introduces an FP8 automatic mixed‑precision training framework that pushes 8‑bit floating‑point (FP8) beyond matrix multiplies to cover gradients, optimizer states, and distributed communication for training large language models (LLMs). On Nvidia H100 GPUs, it preserves model accuracy without hyper‑parameter changes while cutting real GPU memory by up to 39% and increasing throughput by up to 75% versus BF16 (Table 5), and it outperforms Nvidia’s Transformer Engine (TE) by 37% in end‑to‑end speed on GPT‑175B.

## 2. Context and Motivation
- Problem addressed
  - LLM training is prohibitively expensive in compute, memory, and inter‑GPU communication. Prior “mixed‑precision” practice uses FP16/BF16 for compute and FP32 for some states to stay numerically stable (Sec. 2; Refs. Micikevicius et al., 2017).
  - FP8 on H100 GPUs promises 2× speed and up to 75% savings in memory and communication versus 16/32‑bit formats (Sec. 1), but practical support is limited.

- Why it matters
  - Lowering precision is one of few levers that simultaneously reduces compute, memory, and bandwidth costs. Realizing stable FP8 training enables larger models, longer contexts, and faster training within fixed hardware budgets (Fig. 1).

- Where prior approaches fall short
  - Nvidia Transformer Engine uses FP8 only inside GEMMs (matrix multiplies) but keeps master weights, gradients, and collective communications at higher precision (Sec. 1; Sec. 2). This limits end‑to‑end gains in memory and communication.
  - Naïvely using FP8 causes numerical instabilities (underflow/overflow, quantization error), especially for gradients and optimizer states (Sec. 2).

- Positioning
  - The paper extends FP8 usage end‑to‑end: gradients, optimizer, and distributed parallelism (Sec. 2). It adds two mechanisms to maintain stability: automatic scaling for FP8 gradient communication (Sec. 2.1, Eqs. 1–6) and precision decoupling in the optimizer (Sec. 2.2, Eqs. 7–8; Table 6). It also adapts sequence/tensor parallelism and ZeRO to FP8 (Sec. 2.3; Fig. 2, Fig. 3, Alg. 1).

## 3. Technical Approach
This section explains how the system achieves numerically stable, end‑to‑end FP8 training.

- Background on FP8 formats (Appendix A.1)
  - Two standardized FP8 types balance range and precision: `E4M3` (more precision, smaller range) and `E5M2` (larger range, less precision). Compared to FP16/BF16, FP8 has much narrower representable ranges and fewer mantissa bits (Table 9), increasing risk of underflow/overflow and quantization error.
  - Tensor scaling places values into a “comfortable” representable range before converting to FP8 (Appendix A.2; Fig. 9).

- 3.1 FP8 gradients and FP8 all‑reduce communication (Sec. 2.1)
  - Problem: In data‑parallel training, gradients are aggregated across GPUs via `all-reduce`. Standard choices:
    - Pre‑scaling: divide by number of GPUs `N` before summation, Eq. (1). Risk: underflow at large `N`.
    - Post‑scaling: sum then divide by `N`, Eq. (2). Risk: overflow during summation.
  - Mechanism 1: Automatic scaling per gradient tensor across steps (Eq. 3).
    - Introduce a dynamic factor `μ` that multiplies each local gradient `g_i`. Monitor the fraction of values that saturate at FP8’s maximum; if above 0.001%, halve `μ` next step (avoid overflow). If consistently below the threshold, gradually double `μ` over 1,000 steps (avoid underflow).
  - Mechanism 2: Single shared scaling factor for all GPUs (Eqs. 4–6).
    - Each GPU has a local FP8 gradient tensor `g'_i` with a local scale `s'_i`. Before all‑reduce, the system gathers all scales `s'_i`, computes the global minimum `s'_g = min(s'_1,...,s'_N)` (Eq. 4), and re‑quantizes local gradients to FP8 using the shared `s'_g` (Eq. 5). Then a standard NCCL all‑reduce sums the FP8 values. The final scale becomes `s = N·s'_g` (Eq. 6).
    - Why this helps: only one scalar per gradient tensor is communicated in addition to the FP8 payload, yet all tensors align to a common scale so a normal all‑reduce can be used (no custom collective). This avoids complex per‑chunk scale handling and prevents overflow/underflow during summation.
  - Evidence: Fig. 7 compares pre‑scale/post‑scale vs auto‑scale across Transformer blocks. Auto‑scale reduces underflow and overflow simultaneously (Fig. 7b–c) and yields higher signal‑to‑noise ratio (Fig. 7a).

- 3.2 FP8 optimizer with precision decoupling (Sec. 2.2)
  - Definition: In AdamW, the “master weights” are the high‑precision variables used to accumulate updates; `m` is the first‑moment (EMA of gradients), and `v` is the second‑moment (EMA of squared gradients).
  - Problem: Standard practice stores master weights, gradients, and Adam moments in FP32 for stability, costing 16 bytes per parameter (Eq. 7).
  - Key design choice (“precision decoupling”): allocate precision based on sensitivity (Table 6; Fig. 8).
    - Use FP16 with tensor scaling for master weights. Rationale: weight updates are often tiny; losing precision here harms convergence (Fig. 8 shows FP8 master weights degrade; BF16 master weights also slightly worse than FP16+scaling).
    - Store gradients in FP8.
    - Store first‑moment `m` in FP8. Rationale: direction matters more than fine magnitude details; scaling preserves distribution shape.
    - Store second‑moment `v` in FP16 (not FP8). Rationale: squaring small gradients can underflow in FP8; FP8 `v` diverged in ablation “FP8 #4” (Fig. 8).
  - Outcome: The optimizer memory drops from 16 bytes to 6 bytes per parameter (Eq. 8) while maintaining accuracy across scales (Sec. 3.2.1; Table 2; Fig. 4).

- 3.3 FP8 distributed parallel training (Sec. 2.3)
  - Parallelism definitions:
    - `Tensor parallelism`: split a layer’s parameters across GPUs.
    - `Pipeline parallelism`: split layers across stages on different GPUs.
    - `Sequence parallelism`: split the input sequence (tokens) across GPUs to save activation memory.
    - `ZeRO`: shard optimizer/gradient/master weight states across data‑parallel GPUs to remove redundancy.
  - FP8 in tensor and sequence parallelism (Fig. 2)
    - Place FP8 converters around GEMMs so both forward and backward pass use FP8 weights/activations with tensor scaling. For the boundary operator `g` that converts between sequence and tensor parallel regions, cast to FP8 before all‑gather/reduce‑scatter so activation communication also benefits.
    - Measured effect: activation‑related communication volume is reduced by about one‑third (Table 7: e.g., GPT‑175B, 5.9 GB → 3.9 GB).
  - FP8‑aware ZeRO (Fig. 3; Alg. 1)
    - Problem: ZeRO partitions tensors into chunks; FP8 tensors carry per‑tensor scaling factors, which complicates chunking because each chunk would need consistent scaling.
    - Solution: shard whole FP8 tensors (with their scales) across GPUs using a greedy, memory‑aware assignment (Alg. 1), instead of partitioning into slices. This avoids per‑chunk scale management and reduces communication/compute complexity.
    - Effect: lower and more balanced memory usage across GPUs (Table 8 shows reduced min/max memory vs TE/BF16 across model sizes).

- 3.4 Experimental setup essentials (Sec. 3.1)
  - Models: Decoder‑only Transformers with RoPE and FlashAttention (Sec. 3.1.2; Table 1).
  - Scales: 125M, 7B, 13B, 175B parameters. Training on Azure NDv5 H100 80GB GPUs (Sec. 3.1.2).
  - Data: A 100B‑token mixture from CommonCrawl, C4, OpenWebText, Wikipedia, code, books, etc. (Appendix A.3; Table 10). For GPT‑175B, training is limited to 40B tokens to control cost while evaluating system metrics (Table 1 note).
  - Tuning/Alignment: Instruction tuning on ShareGPT‑style data; RLHF using Anthropic HH and Open‑Assistant data via AlpacaFarm (Sec. 3.1.1).

## 4. Key Insights and Innovations
- FP8 all‑reduce with dual scaling mechanisms is stable and practical
  - What’s new: Combine dynamic “auto‑scale” across steps (Eq. 3) with a single shared per‑tensor scale across GPUs (Eqs. 4–6), enabling standard NCCL all‑reduce over FP8 tensors (Sec. 2.1).
  - Why it matters: Overcomes the underflow of pre‑scaling and the overflow of post‑scaling simultaneously (Fig. 7), unlocking 8‑bit gradient storage and communication. This reduces weight‑related communication volume by 63–65% in practice (Table 5).

- Optimizer precision decoupling that keeps only what truly needs higher precision
  - What’s new: FP8 gradients + FP8 first moment + FP16 second moment + FP16 master weights (with scaling) inside AdamW (Sec. 2.2; Table 6).
  - Why it matters: Cuts optimizer memory from 16B to 6B per parameter (Eq. 8) without accuracy loss (Fig. 4; Table 2). Ablations show which states cannot be lowered (FP8 `v` diverges; FP8 master weights degrade; Fig. 8).

- Extending FP8 into distributed parallelism and ZeRO
  - What’s new: FP8 activation communication for sequence/tensor parallel boundaries (Fig. 2) and a whole‑tensor FP8 ZeRO sharding scheme with a greedy balancer (Fig. 3; Alg. 1).
  - Why it matters: Reduces activation‑related communication by ~34% (Table 7) and lowers/levels GPU memory (Table 8), enabling larger micro‑batches or longer sequences (Sec. 3.2.2 notes 4,096‑token context for 175B on 32×H100, whereas TE fits only 2,048).

- End‑to‑end system acceleration beyond what TE achieves
  - What’s new: Move FP8 beyond GEMMs to gradients, optimizer, and collectives—areas TE retains at higher precision (Sec. 1; 2).
  - Why it matters: Delivers stronger real‑world speed/footprint gains: for GPT‑175B, throughput +75% vs BF16 and +37% vs TE; memory −39% vs BF16 and −42% vs TE (Table 5).

## 5. Experimental Analysis
- Methodology and baselines
  - Pre‑training: Trained GPT‑7B/13B for 100B tokens; GPT‑175B for 40B tokens (Table 1). Compare FP8 (this framework), BF16 (Megatron‑LM‑style), and TE (FP8 in GEMMs only).
  - Downstream: Zero‑shot on HellaSwag, Lambada, BoolQ, PIQA, COPA, Winogrande, ARC‑C/E, OpenBookQA (Table 2).
  - Tuning/Alignment: Instruction tuning (Vicuna‑style) and PPO‑based RLHF (AlpacaFarm) in FP8 vs BF16 (Sec. 3.2.1; Fig. 5–6; Tables 3–4).
  - System metrics: GPU memory, throughput (#samples/s), TFLOPs, model FLOPs utilization (MFU), communication volumes (Table 5; Table 7; Table 8).

- Main quantitative findings
  - Accuracy and learning dynamics
    - Pre‑training loss curves for 7B, 13B, 175B nearly overlap between FP8 and BF16 (Fig. 4), with no hyper‑parameter changes (Sec. 3.2.1).
    - Zero‑shot downstream is comparable across many tasks (Table 2; differences within a point or two).
  - System‑level improvements (Table 5)
    - GPT‑175B, same micro‑batch size (1):
      - Throughput: 22.4 (BF16) → 27.1 (FP8) = +21%.
      - GPU memory: 66.1 GB → 40.3 GB = −39%.
      - Weight‑related communication volume: 23.4 GB → 8.2 GB = −65%.
      - MFU: 39.0% → 23.9% for the micro‑batch 1 row; with larger micro‑batch (4), MFU rises to 34.2%.
    - GPT‑175B, leveraging saved memory to increase micro‑batch size (1 → 4):
      - Throughput: 22.4 → 39.3 = +75% vs BF16.
      - Memory still below BF16 (57.7 GB vs 66.1 GB).
    - Across 7B/13B: similar trends—memory drops 28–29% vs BF16; throughput +38–53% when larger micro‑batches are possible; communication volume −63–64%.
  - Instruction tuning (Table 3; Fig. 5)
    - GPU memory: 51.1 GB (BF16) → 44.0 GB (FP8) = −14%.
    - Throughput: 103 → 131 = +27%.
    - Quality: AlpacaEval win‑rate 66.15 → 67.20; MT‑Bench 5.75 → 5.70 (parity).
  - RLHF (Table 4; Fig. 6)
    - Weights memory: 15,082 MB → 10,292 MB = −32%.
    - Optimizer states: 15,116 MB → 5,669 MB = −62%.
    - Quality: AlpacaEval 72.05 → 72.42; MT‑Bench 6.16 → 6.04 (parity).

- Do the experiments support the claims?
  - Stability and accuracy are strongly supported by overlapping loss curves (Fig. 4) and task scores (Table 2, 3, 4). The ablation on communication scaling (Fig. 7) directly explains why FP8 all‑reduce works reliably.
  - System benefits are documented with per‑model breakdowns (Table 5). A subtlety: the headline 75% throughput gain at 175B leverages a larger micro‑batch enabled by memory savings—at equal micro‑batch, the gain is +21% (Table 5). The paper is explicit about using the saved memory for larger batches/longer contexts (Sec. 3.2.2).

- Ablations and robustness checks
  - Gradient all‑reduce strategies (Fig. 7): Auto‑scale dominates pre/post‑scale on SNR and reduces both underflow and overflow.
  - Optimizer precision settings (Table 6; Fig. 8): FP8 master weights or FP8 second moment harm training; FP16+scaling for master weights and FP16 for the second moment are necessary for stability.
  - Parallel communication (Table 7): FP8 reduces activation all‑gather/reduce‑scatter volume by ~34%.
  - FP8 ZeRO sharding (Table 8): lower min/max memory across GPUs and better balance than BF16/TE.

## 6. Limitations and Trade-offs
- Hardware dependence
  - The gains rely on FP8 support in Hopper‑class GPUs (H100). Earlier architectures or other accelerators may not see the same benefits or require porting the scaling mechanisms.

- Heuristic scaling controls
  - The auto‑scale policy uses a saturation threshold (0.001%) and exponential growth over 1,000 steps (Sec. 2.1). These heuristics work empirically (Fig. 7) but are not theoretically optimized; different models/data might prefer different thresholds or schedules.

- Optimizer precision floor
  - The method still needs FP16 (with scaling) for master weights and FP16 for the second moment; fully FP8 optimizers (including `v`) diverged in ablations (Fig. 8). Memory is greatly reduced (6B/param) but cannot reach the theoretical minimum of “all in FP8” without accuracy loss.

- Throughput comparisons vs BF16
  - The largest throughput gain (+75%) for GPT‑175B is realized when FP8’s memory savings allow a 4× larger micro‑batch (Table 5). At the same micro‑batch, the speedup is +21%. This is a fair and practical scenario (since users will trade saved memory for throughput), but readers should be aware of the distinction.

- Training coverage for 175B
  - The 175B model is trained on 40B tokens (Table 1 note) to evaluate system behavior efficiently. While pre‑training loss trends are similar (Fig. 4c), full‑convergence comparisons on standard compute budgets are not presented.

- Scope of tasks
  - Demonstrations include zero‑shot language tasks, instruction tuning, and RLHF (Tables 2–4). Broader domains (e.g., multilingual, code‑heavy pre‑training at scale, or multi‑modal) and longer training horizons remain future work (Sec. 5 Conclusion).

## 7. Implications and Future Directions
- Field impact
  - This work shows that FP8 can be used end‑to‑end, not just inside GEMMs, without sacrificing accuracy. The combination of FP8 gradients, FP8/FP16 optimizer with precision decoupling, and FP8‑aware distributed training provides tangible cost reductions and opens the door to training larger models and longer contexts on fixed hardware (Fig. 1; Sec. 3.2.2).

- Follow‑up research enabled/suggested
  - Theory and control for scaling policies: Beyond the current heuristic thresholding, one could study optimal control of `μ` and per‑tensor scales to further improve stability and compression (Sec. 2.1).
  - Extending low‑precision boundaries: Explore whether variants of the second moment accumulation can be re‑designed to tolerate FP8 (e.g., log‑space accumulators, blockwise scaling, stochastic rounding) to further reduce memory.
  - Communication primitives: Native FP8 collectives with scale metadata in libraries like NCCL could simplify implementations, reduce overhead of shared‑scale gathering, and potentially increase bandwidth efficiency.
  - Beyond language: The authors plan to apply the FP8 scheme to multi‑modal large models and low‑bit deployment on edge devices (Sec. 5 Conclusion).

- Practical applications
  - Faster, cheaper LLM pre‑training and fine‑tuning, including instruction tuning and RLHF with multiple concurrent models (Table 4).
  - Training longer contexts on the same hardware: the paper notes 175B models can train with 4,096 tokens on 32×H100 with FP8, whereas TE fits only 2,048 (Sec. 3.2.2).
  - Cloud training stacks: The open‑source codebase (aka.ms/MS.AMP) can be integrated into Megatron‑LM‑style pipelines, providing an immediate path to reduce costs for organizations training large models.

> Bottom line: By making gradients, optimizer states, and inter‑GPU communication FP8‑aware—and by engineering the scaling and sharding details—the framework delivers substantial, hardware‑realizable savings with accuracy parity (Fig. 4; Tables 2–4), and system‑level reductions in memory and communication (Tables 5, 7, 8).
