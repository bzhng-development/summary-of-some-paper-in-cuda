# ThunderKittens: Simple, Fast, and Adorable AI Kernels

**ArXiv:** [2410.20399](https://arxiv.org/abs/2410.20399)

## 🎯 Pitch

ThunderKittens (TK) introduces a lightweight CUDA-embedded framework that lets developers write state-of-the-art AI GPU kernels using just a handful of clear, PyTorch-like abstractions—covering warp-, block-, and grid-level parallelism. By focusing on simplicity and efficient hardware mapping, TK consistently matches or outperforms industry-standard kernels (like cuBLAS and FlashAttention-3) across core AI operations, dramatically lowering the barrier to both performance and maintainability for the next generation of ML workloads.

---

## 1. Executive Summary (2-3 sentences)
ThunderKittens (TK) is a CUDA-embedded framework that lets developers write high‑performance AI kernels using a small, opinionated set of abstractions mapped to three GPU hierarchy levels: tiles at the warp level, an asynchronous load–compute–store–finish (LCSF) template at the block level, and grid-level scheduling helpers. Across GEMM, attention, linear attention, and state‑space/FFT convolutions, TK matches or beats strong baselines such as cuBLAS and FlashAttention‑3 while being markedly simpler to write and maintain (Figures 7–9, Table 4).

## 2. Context and Motivation
- The problem: Modern AI models depend on GPU kernels that rarely reach the hardware’s theoretical potential, even for “standard” ops like attention. Example: a widely used attention kernel (FlashAttention‑2) lost 47% performance when moved to NVIDIA H100, and it took two years to create FlashAttention‑3 (Section 1).
- Why this matters:
  - Real‑world impact: Inference/training costs and throughput hinge on kernel efficiency. Under‑utilized tensor cores (specialized matrix units) leave most FLOPs idle (Section 1).
  - Hardware trend: GPUs increasingly devote silicon to tensor cores (e.g., 16× more BF16 tensor‑core FLOPs vs general-purpose pipelines on A100/H100). Kernels must keep tensor cores busy while hiding memory and non‑tensor compute overheads (Section 1).
- Prior approaches and gaps:
  - C++ template libraries (CUTLASS/CuTe) can reach peak performance but are complex to use and tune; users must navigate nested templates and low‑level details like layouts and synchronization (Section 2.3).
  - Compiler approaches (Triton, TVM, XLA) simplify programming but can make it hard to access specialized, newest instructions (e.g., bulk async copies, tensor core variants) and to finely control asynchronous pipelines and register usage (Section 2.3).
- Positioning:
  - TK asks whether a very small, opinionated set of abstractions can cover many high‑performance AI kernels without sacrificing speed. It is CUDA‑embedded (so developers can drop to raw CUDA when needed) and centers on three ideas: tensor‑core‑aligned tiles, an asynchronous worker template (LCSF), and grid scheduling techniques for L2 reuse and launch overheads (Section 3; Figure 1).

Key GPU terms used (defined where they first matter):
- `warp`: 32 threads that execute in lockstep; four warps can form a `warpgroup` on H100.
- `thread block` (or block): A set of warps running on a single streaming multiprocessor (SM), sharing on‑chip `shared memory` (SMEM).
- `grid`: Many blocks launched for a kernel.
- `tensor cores`: Specialized matrix multiply units that dominate GPU compute throughput.
- `HBM`: High‑Bandwidth Memory (global GPU memory). High capacity, highest latency.
- `L2 cache`: On‑chip cache shared across SMs; lower latency than HBM.
- `bank conflict`: SMEM is partitioned into 32 banks. If multiple threads access the same bank simultaneously, accesses serialize and stall.

## 3. Technical Approach
TK maps to the GPU hierarchy with three concrete components and a simple cost model that explains how they interact.

A. Cost model (Section 2.2)
- Idealized kernel time ≈ max of memory costs (HBM/L2/L1/SMEM) and compute costs (tensor cores, ALU/XU/FMA), plus launch/sync overheads:
  > C_overall = max(C_HBM, C_L2, C_L1, C_Shared, C_Tensor, C_ALU, C_FMA, C_XU) + C_Setup + C_Sync
- Goal: Reduce each term and overlap them so the max—not the sum—dominates.

B. Warp level: tiles and PyTorch‑like ops (Section 3.1; Figures 1–2, 4)
- Core data structure: 16×16 `tile` aligned to tensor cores. TK supplies:
  - `rt<M,N>` register tiles (fastest memory, used in compute),
  - `st<M,N>` shared tiles (in SMEM, used for staging),
  - `gl<...>` global layout descriptors (for HBM indexing; compile‑time dims can live in instruction cache to save registers).
- Operations resemble PyTorch/NumPy but run in parallel over tiles: `mma_AB`, `mma_ABt` (tensor core matmuls), `exp`, `row_sum`, `row_max`, `copy`, `zero`, etc. Figure 2 shows how attention can be expressed as a few tile ops mirroring a PyTorch program.
- Layout choice to avoid bank conflicts and support hardware instructions (Section 3.1; Figure 4; Appendix C):
  - TK narrows layout options to just three `swizzled` SMEM layouts with 32B/64B/128B strides, chosen automatically at compile time based on tile width. These layouts keep addresses aligned for hardware‑accelerated instructions (e.g., Hopper’s TMA for bulk copies and WGMMA/HGMMA for matrix ops) while minimizing bank conflicts.
  - Why this matters: A naïve row‑major SMEM layout has severe 8‑way bank conflicts when loading tensor‑core register formats (Figure 4, top left). Swizzled layouts reduce or eliminate conflicts and preserve hardware support (Appendix C.4–C.6).

C. Block level: a single asynchronous worker template (LCSF) (Section 3.2; Figure 5; Table 1; Figure 6)
- Template structure: `Load → Compute → Store → Finish`
  - `load`: One or more warpgroups act as producers, moving tiles from HBM to SMEM.
  - `compute`: Other warpgroups consume SMEM tiles, run tensor‑core math and non‑tensor ops in registers/SMEM.
  - `store`: Producers write outputs from SMEM to HBM asynchronously.
  - `finish`: Wrap up any remaining work/state.
- Built‑ins the template manages:
  - Multi‑stage SMEM buffers for pipelining (N stages). With deeper buffers, the next tiles are prefetched while current tiles are computed. Table 1 shows a GEMM speedup from 260 TFLOPs (1 stage) to 760 TFLOPs (4 stages).
  - `arrive` barriers to signal readiness between producer/consumer stages (Figure 5).
  - Unified async I/O: wraps both `cp.async` and Hopper’s `TMA` with a common interface; TK auto‑creates tensor‑map descriptors for `gl` layouts (Section 3.2).
  - Register budgeting helpers (e.g., `warpgroup::increase_registers`/`decrease_registers`) so compute warps can own larger tiles while producer warps stay thin (Appendix B).
- Tuning occupancy vs. efficiency (Figure 6):
  - Higher occupancy (more active warps) increases overlap but increases contention over registers, SMEM, and issue slots.
  - LCSF expands the Pareto frontier over a synchronous baseline: as the number of worker warpgroups rises from 1 to ~5–6, LCSF maintains higher TFLOPs before contention dominates (Figure 6).

D. Grid level: launch and scheduling (Section 3.3; Tables 2–3)
- Persistent grid: Instead of launching new blocks for each work chunk, TK keeps 132 resident blocks (matching H100 SMs) and hands them new work items. This reduces `C_Setup` and lets blocks preload inputs for the next chunk during `finish`.
  - Table 2 shows GEMM TFLOPs gains at smaller `K` when persistence is turned on (e.g., 4096×4096×256: 309 TFLOPs with persistence vs 271 without; cuBLAS gets 242).
- Block order for L2 reuse: The order in which blocks traverse tiles determines whether data is found in L2 or reloaded from HBM.
  - Table 3 (GEMM): a block order `{8, N, M/8}` consumes only 982 GB/s HBM and reaches 805 TFLOPs, while a naïve `{N, M}` ordering pulls 3070 GB/s and only 392 TFLOPs.
  - Table 3 (attention): an L2‑friendly order `{N, H, B}` uses 213 GB/s for 600 TFLOPs vs row‑major `{B, H, N}` requiring 2390 GB/s for just 494 TFLOPs.

E. Example: attention written with TK (Figures 1–2, 5; Appendix B.3)
- Registers hold `Q` fragments; SMEM buffers stream `K` and `V` tiles from HBM via TMA.
- Compute warps run:
  - `A = Q @ K^T` on tensor cores (`mma_ABt`),
  - softmax in registers using vector ops (`row_max`, `exp2`, `row_sum`, scaling tricks for stability in lines 53–11 of Figure 12),
  - `O += A @ V` on tensor cores (`mma_AB`).
- Barriers ensure that once a tile of `K`/`V` is consumed, producers can overwrite that buffer slot.

Why this approach over alternatives?
- Fixes common pitfalls (bank conflicts, misaligned layouts) by construction (Section 3.1; Figure 4; Appendix C).
- Encourages high tensor‑core utilization through 16×16 tiles and warpgroup MMA ops (WGMMA), but keeps a simple, PyTorch‑like API (Figure 2).
- Unifies asynchronous data movement and compute; developers fill a few template hooks rather than hand‑roll scheduling (Section 3.2; Figure 5).
- Leaves escape hatches: embedded in CUDA/C++, so specialized instructions or non‑templated code can be inserted when needed (Section 3.2 end).

## 4. Key Insights and Innovations
1) Tile‑centric, tensor‑core‑first programming model (Section 3.1; Figures 1–2, 4)
- What’s new: TK standardizes on 16×16 tiles and auto‑selects one of three SMEM `swizzled` layouts with hardware support, minimizing bank conflicts during tensor‑core loads.
- Why it matters: Keeps tensor cores—the dominant compute resource—busy with minimal developer effort; avoids subtle layout bugs that degrade bandwidth (Table 4 shows bank conflicts in a strong baseline vs. none in TK’s attention backward).

2) A single, reusable block‑level template (LCSF) for asynchrony (Section 3.2; Figure 5; Table 1; Figure 6)
- What’s new: Producer/consumer specialization across warpgroups, N‑stage pipelining, unified async I/O, and lightweight barriers built into one template.
- Why it matters: Simplifies overlapping of HBM transfers with compute; enables easy tuning of pipeline depth and occupancy. Table 1 quantifies the benefit of deeper pipelines; Figure 6 shows LCSF expands the performance frontier vs synchronous execution.

3) Grid scheduling for L2 locality and low launch overhead (Section 3.3; Tables 2–3)
- What’s new: Practical, template‑compatible choices—persistent grids to amortize `C_Setup`, and non‑naïve block orders to maximize L2 reuse.
- Why it matters: Tables 2–3 show large, concrete gains (often 1.3–2× TFLOPs or 3× lower HBM traffic) without changing the kernel math.

4) Evidence that a very small abstraction set suffices for a broad kernel family (Section 4; Figures 7–9; Table 4; Appendices B–C)
- What’s new: One set of concepts (tiles + LCSF + grid scheduling) expresses competitive kernels for GEMM, (causal/non‑causal/GQA) attention, rotary embeddings, linear attention variants, long convolutions/SSMs, and Mamba‑2 updates.
- Why it matters: Reduces development cost and time; makes high‑performance kernels accessible even to developers without deep CUDA backgrounds (Section 1, contributions).

## 5. Experimental Analysis
Evaluation setup (Section 4)
- Hardware: NVIDIA H100 80GB SXM, CUDA 12.6; average TFLOPs reported over timed iterations.
- Baselines:
  - GEMM: cuBLAS (Figure 7).
  - Attention: FlashAttention‑3 (FA3) (Figure 8).
  - Linear attention: FLA Triton kernels (Figure 9).
  - FFT/long convolution: FlashFFTConv (Figure 9).
  - Mamba‑2: Triton kernels from Dao & Gu (ICML 2024) (Figure 9).
  - Also compares to PyTorch in some ops (Figure 9).
- Profiling: Nsight Compute to inspect tensor‑core utilization, issue slot utilization, HBM throughput, and stalls due to SMEM/HBM (Table 4).

Main quantitative results
- GEMM (Figure 7; Table 2; Appendix B.1):
  - One TK kernel (~40 lines of device code) competes with cuBLAS. With persistent blocks, TK reaches 600 TFLOPs on 4096×4096×1024 vs cuBLAS 633; for smaller K, persistence helps TK surpass cuBLAS (e.g., K=256: 309 vs 242 TFLOPs; Table 2).
- Attention (Figure 8; Appendix B.3):
  - Non‑causal forward: TK matches FA3 across sequence lengths (top‑left panel).
  - Causal/non‑causal backward: TK outperforms by 10–40% depending on length; e.g.,
    > “TK exceeds FA3 by >40% at short sequences and ~10% at long sequences” (Figure 8, top‑right and bottom‑left/bottom‑right panels).
- Linear attention (Figure 9, two middle-left panels):
  - Polynomial‑based: TK is up to 14× faster than FLA Triton kernels.
  - Learned-kernel variants: up to 6.5× faster than FLA.
- Long convolution / SSM primitives (Figure 9, top‑right):
  - TK outperforms FlashFFTConv by 4.7× at length 4096 and 7.9× at 1024; beats PyTorch FFTs by up to 8.7×.
- Mamba‑2 (Figure 9, top‑left):
  - TK >3× faster than the Triton implementation across shown sequence lengths.
- Other fused ops (Figure 9, bottom‑row):
  - Fused dropout–residual–layernorm and rotary encodings: TK variants outperform popular Triton baselines.

Do profiles justify the claims? (Table 4)
- Attention backward (FA3 vs TK):
  - Similar tensor‑core utilization (61.2% vs 58.2%), but TK has higher issue slot utilization (34.8% vs 25.1%) and higher effective HBM GB/s (490 vs 328).
  - SMEM stalls are dramatically lower:
    > “Shared stalls drop from 0.92 to 0.14 cycles” (Table 4), explained by TK’s conflict‑free layouts vs. up to 9.6‑way bank conflicts detected in FA3 (Section 4.2).
- Long convolution (FlashFFTConv vs TK):
  - Tensor‑core utilization improves from 13.4% to 54.8%; issue slots from 25.5% to 40.0%; HBM TPS doubles (14.8→31.4). SMEM/HBM stalls both shrink (Table 4).

Ablations and trade‑offs
- Pipeline depth: More stages significantly raise GEMM throughput (Table 1: 260→760 TFLOPs from 1→4 stages).
- Occupancy: LCSF remains efficient at higher occupancies than a synchronous design (Figure 6).
- Grid scheduling: Persistent blocks and careful block order both matter; the latter can halve TFLOPs if chosen poorly at large sizes (Table 3 and the GEMM grid discussion in Appendix B.1).

Assessment
- Coverage: Benchmarks touch standard and emerging ops with consistent wins or parity.
- Causality: TK’s improvements are linked to concrete, measurable mechanisms (layout conflicts eliminated; async pipelines; grid locality), supported by Nsight metrics (Table 4) and controlled ablations (Table 1, Figure 6, Table 2–3).
- Caveat: cuBLAS still leads on some large‑K GEMMs (4096×4096×1024: cuBLAS 633 vs TK 600; Table 2), showing room for further tuning.

## 6. Limitations and Trade-offs
- Hardware specificity:
  - Many gains exploit Hopper (H100) features such as `TMA` (bulk async copies) and `WGMMA/HGMMA` (warpgroup MMAs). Portability to other GPUs is conceptually possible (Section 2.1 footnote), but achieving equal performance will require mapping to vendor‑specific instructions.
- Opinionated layouts and tile sizes:
  - The fixed 16×16 register tile and three SMEM swizzle choices work very well for tensor‑core‑centric ops. Workloads that are not matrix/tile‑friendly may need custom extensions (TK allows raw CUDA, but that reduces abstraction benefits).
- Manual tuning remains:
  - Although LCSF abstracts asynchrony, developers still set pipeline stages, choose occupancy (number of load/store vs. compute workers), and pick grid orders. The paper demonstrates how to do this (Tables 1–3; Figure 6), but it is not fully auto‑tuned.
- Scope of evaluations:
  - Results are on H100 80GB SXM with CUDA 12.6. Performance on other GPUs or mixed‑precision regimes beyond FP16/BF16 is not detailed. Some GEMM regimes still favor cuBLAS (Table 2).

Open questions
- How far can the three‑layout policy be pushed for more exotic data types or irregular shapes?
- Can the framework include auto‑schedulers to choose pipeline depth/occupancy/grid order automatically without sacrificing transparency?

## 7. Implications and Future Directions
- Impact on the field:
  - Demonstrates that a small, principled abstraction set can match or outperform hand‑tuned or compiler‑generated kernels across diverse AI ops. This lowers the barrier to building fast custom kernels and may accelerate the adoption of new architectures (Section 5 conclusion).
- Practical applications:
  - Inference providers and latency‑sensitive domains (e.g., high‑frequency trading) can deploy TK kernels to cut costs/latency (Section 1 concluding paragraph). The open‑source repo (Section 5) makes it directly usable.
- Research enabled:
  - Faster prototyping for novel attention variants, hybrid SSM‑Transformer designs, and custom fused ops, since developers can focus on math while TK handles layouts, pipelining, and scheduling (Appendix B shows concise kernels).
  - Systems/compilers work: integrating TK’s LCSF and tile policies into higher‑level auto‑schedulers; extending to multi‑GPU or pipeline parallel contexts; mapping TK to non‑NVIDIA GPUs with analogous instructions.
- Next steps:
  - Auto‑tuning of pipeline stages and occupancy within the LCSF template.
  - Broader precision support and adaptive layout selection for non‑standard shapes.
  - Cross‑vendor backends (AMD/Apple) to validate the hypothesis that the same few abstractions carry across architectures (Section 2.1 footnote).

> Key takeaway: Figures 7–9 and Table 4, together with Tables 1–3 and Figure 6, show that TK’s three‑level design—tiles, asynchronous worker template, and grid scheduling—systematically tackles memory, compute, and overhead terms in the cost model. This combination achieves state‑of‑the‑art or better performance while keeping kernel code concise and comprehensible.
