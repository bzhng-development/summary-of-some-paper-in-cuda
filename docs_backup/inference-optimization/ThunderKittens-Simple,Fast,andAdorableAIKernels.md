# ThunderKittens: Simple, Fast, and Adorable AI Kernels

**ArXiv:** [2410.20399](https://arxiv.org/abs/2410.20399)

## 🎯 Pitch

ThunderKittens introduces a lightweight, C++-embedded framework that radically simplifies GPU kernel development by distilling high-performance programming down to three intuitive abstractions—tile-based data structures, a generalized asynchronous template, and grid-level scheduling aids—all mapped to the GPU's natural hierarchy. By making it easy for even non-experts to write kernels that fully utilize tensor cores and memory, ThunderKittens not only matches or outperforms the best hand-tuned baselines for critical AI operations (e.g., GEMM, attention, long sequence models), but also accelerates AI innovation by reducing both complexity and bottlenecks that slow industry progress.

---

## 1. Executive Summary (2-3 sentences)
ThunderKittens (TK) is a small C++-embedded framework for writing high‑performance GPU kernels using a few opinionated abstractions that align with the GPU’s three levels of parallelism (warp, block, grid). By giving developers 16×16 “tile” data structures with managed layouts, a generalized asynchronous Load‑Compute‑Store‑Finish (LCSF) template inside a thread block, and grid‑level scheduling helpers, TK matches or outperforms state‑of‑the‑art hand‑tuned kernels on NVIDIA H100: it competes with cuBLAS for GEMM and FlashAttention‑3 for inference, and yields large gains on attention backprop (+10–40%), linear attention (up to 14×), and long‑sequence state‑space models (up to 8×) (Figures 7–9, Tables 1–3, Section 4).

## 2. Context and Motivation
- Problem the paper tackles
  - Modern AI models rely on GPU kernels (low‑level implementations of ops like matrix multiply and attention). Writing kernels that both use specialized hardware (tensor cores, async copy engines) and avoid memory bottlenecks is hard; even widely used operations (e.g., softmax attention) regularly underperform on new hardware generations (Section 1).
  - Example: FlashAttention‑2 lost 47% performance when moved to H100, and it took >2 years after H100’s release to recover performance with FlashAttention‑3 (Section 1).
- Why it matters
  - Compute growth on GPUs is concentrated in specialized matrix units (“tensor cores”). To run modern models efficiently, kernels must keep tensor cores busy while hiding memory and synchronization costs (Section 1, cost model in Section 2.2).
  - Poor kernel performance slows training and inference across industry (e.g., LLM serving and long‑sequence models), increasing cost and limiting what architectures are practical.
- Prior approaches and their gaps
  - CUTLASS/CuTe: extremely powerful C++ template libraries, but complex to use and tune; developers must reason about many nested templates and hardware quirks (Section 2.3).
  - Compiler DSLs like Triton: simpler user experience but make it harder to use the newest low‑level instructions or finely control asynchronous execution and registers (Section 2.3).
  - Result: fragmented, hand‑tuned kernels with sophisticated, kernel‑specific schedulers (e.g., FA‑3’s ping‑pong) that are hard to maintain or extend.
- Positioning of this work
  - TK asks whether a small, opinionated set of abstractions can cover many AI kernels without sacrificing peak performance (Section 1).
  - It maps directly onto the GPU hierarchy (warp, block, grid) with minimal concepts, aims for PyTorch‑like ergonomics, and manages error‑prone details (memory layouts, synchronization) for the user (Sections 3.1–3.3, Figures 1–2).

## 3. Technical Approach
TK exposes three layers of abstractions that line up with how GPU hardware actually works (illustrated in Figure 1 and explained in Section 2).

1) Warp‑level: tile data structures and PyTorch‑like operations (Section 3.1)
- What the developer sees
  - Basic data type is a 16×16 “tile” in different memories: register tiles (`rt<…>`), shared‑memory tiles (`st<…>`), and global layout descriptors (`gl<…>`) for HBM indexing (Figure 2).
  - Operations mirror familiar PyTorch ops but run in parallel across tiles: `mma_AB` (matrix multiply), `mm_ABt` (A×B^T), `exp`, `row_sum`, `row_max`, `div_row`, `copy`, etc. (Figures 1–2).
- Why 16×16 tiles
  - They match tensor core block sizes, maximizing time spent on the fastest units (Section 1). Keeping data in tensor‑core‑friendly layouts reduces conversions and data movement.
- Managed memory layouts to avoid bank conflicts
  - Shared memory (SMEM) is fast but split into “banks”; if several threads hit the same bank, accesses serialize (“bank conflicts”), harming bandwidth/latency (Section 2.1).
  - TK narrows the layout choices to three “swizzled” patterns—32, 64, and 128‑byte swizzling—and automatically selects the largest compatible layout for a given tile width (Section 3.1; Appendix C.4–C.6).
    - Effect: minimizes bank conflicts while remaining compatible with special hardware instructions (e.g., HGMMA/WGMMA for tensor cores, TMA for bulk copies).
  - Evidence: Figure 4 shows that naïve row‑major layouts cause up to 8‑way conflicts when loading tensor‑core formats, padded layouts waste memory and break hardware alignment, while TK’s chosen swizzles keep alignment and reduce conflicts (2‑way or none depending on width).
- Safety and correctness
  - TK performs static checks on layouts and operands; e.g., if `mma_AB` requires A row‑major and B column‑major, violating layouts is caught at compile time (Section 3.1).

2) Block‑level: the generalized LCSF asynchronous template (Section 3.2)
- What it is
  - A producer–consumer program template with four phases per iteration: Load, Compute, Store, Finish (Figure 5). Within a block, different warps specialize:
    - “Load/store workers” move tiles between HBM and SMEM using async copy engines (TMA/cp.async).
    - “Compute workers” operate on tiles in registers/SMEM using tensor cores and ALUs.
- How it hides latency
  - Multi‑stage pipelines: an N‑stage SMEM buffer lets load/store run ahead while compute works on prior tiles (Section 3.2). With deeper pipelines, fewer full‑block synchronizations are needed because workers operate on different stages concurrently.
  - Barriers and arrivals (`arrive`) coordinate when inputs are ready or outputs can be evicted (Figure 5).
  - Unified async I/O: same interface wraps both synchronous and asynchronous instructions; TK also auto‑creates TMA “tensor maps” for global memory descriptors (Section 3.2).
- Tuning knobs the developer controls
  - Number of load/store vs compute warps (i.e., occupancy), pipeline depth (number of stages), and tile sizes. TK keeps tuning minimal: a few integers influence a wide range of kernels.
- Evidence that the template works
  - Pipeline depth ablation: for 4096³ GEMM, 1 stage gives 260 TFLOPs, while 4 stages reach 760 TFLOPs (Table 1).
  - Occupancy trade‑off: for attention with D=64 and sequence length 4096, increasing worker warpgroups improves TFLOPs until register/SMEM contention dominates; the LCSF version expands the Pareto frontier relative to a synchronous baseline (Figure 6).

3) Grid‑level: persistent blocks and cache‑aware block ordering (Section 3.3)
- Reducing block launch overhead (“pipeline bubbles”)
  - Persistent grid: launch 132 blocks once on the 132 SMs (H100) and have each block pull successive “tasks” rather than relaunching. TK also overlaps loading the next task into the pipeline while finishing the current one (Section 3.3).
  - Effect on GEMM: persistent mode improves TK’s kernel throughput at small/medium `K`; for `M=N=4096`, TFLOPs rise from 93→108 for `K=64` and 271→309 for `K=256` (Table 2).
- Improving L2 reuse via block order
  - Blocks that touch the same data close together in time can hit in the on‑chip L2 cache instead of reloading from slow HBM (Section 2.1).
  - TK exposes block‑order choices; different orders dramatically change HBM bandwidth usage and TFLOPs:
    - GEMM (16384³): with a block order `{8, N, M/8}`, HBM bandwidth drops to 982 GB/s and throughput reaches 805 TFLOPs; a naïve `{N, M}` order consumes 3070 GB/s and only 392 TFLOPs (Table 3).
    - Attention forward (D=128): cache‑friendly order uses 213 GB/s and hits 600 TFLOPs vs 2390 GB/s and 494 TFLOPs for the naïve order (Table 3).

A simple mental model for performance (Section 2.2)
- TK’s cost model: total time is roughly the maximum of memory and compute pipelines plus overheads:
  - C_overall ≈ max({HBM, L2, L1, SMEM}, {tensor cores, ALU, FMA, exp/XU}) + {setup, sync}.
- TK’s abstractions target each term: hardware‑compatible layouts for SMEM, WGMMA/TMA for compute/memory, deep pipelining to overlap stages, and persistent grids to shrink setup and tail inefficiencies.

## 4. Key Insights and Innovations
- Managed tile layouts that “just work” with tensor‑core instructions
  - What’s new: TK restricts shared‑memory layouts to three hardware‑supported swizzles (32/64/128B) and auto‑selects the best given tile width (Section 3.1; Appendix C). This avoids user‑induced bank conflicts while keeping addresses aligned for TMA/H(G/W)MMA.
  - Why it matters: Many high‑end kernels still incur bank conflicts; NSight profiles in Table 4 show FA‑3’s attention backward experiencing up to 9.6‑way bank conflicts (noted in Section 4.2), while TK’s version shows 85% fewer SMEM stall cycles (0.14 vs 0.92).
- A single, generalized asynchronous template (LCSF) that spans many kernels
  - What’s new: Instead of bespoke schedulers (e.g., FA‑3’s ping‑pong), TK provides one LCSF template with N‑stage buffers and warp specialization; the developer writes four short callbacks (Load, Compute, Store, Finish) using tile ops (Figure 5).
  - Why it matters: It simplifies development while achieving high utilization; pipeline depth and occupancy ablations (Table 1, Figure 6) show the template captures the key performance levers.
- Grid‑level levers exposed but simple
  - What’s new: Lightweight helpers for persistent grids and cache‑friendly block ordering (Section 3.3).
  - Why it matters: For large problems that exceed L2 capacity, launch order can be a first‑order factor (Table 3 shows >2× TFLOP differences and 3× HBM bandwidth changes).
- Small code, big performance
  - Demonstration: a ~40‑line GEMM kernel in TK reaches cuBLAS‑class throughput (Figure 7; Appendix B.1 shows the full kernel). The TK library itself is <1 MB vs 689 MB for cuBLAS and 22 MB for CUTLASS (Appendix A, Table 5).
- Safety via compile‑time checks
  - TK’s tile types encode layout and shape, so many illegal instruction/layout combinations are caught at compile time (Section 3.1). This is unusual among high‑performance GPU frameworks and reduces debugging time.

## 5. Experimental Analysis
- Setup
  - Hardware/software: NVIDIA H100 80 GB SXM, CUDA 12.6 (Section 4). Benchmarks report average TFLOPs over 10 warmups + 10 runs; full C++ timing with Python bindings available (Appendix B).
  - Baselines: cuBLAS for GEMM; FlashAttention‑3 (FA‑3) for attention; FlashFFTConv for long FFT‑based convolution; Flash Linear Attention (Triton) for linear attention; PyTorch for general ops (Section 4 and Figure 9).
- Main quantitative results
  - GEMM
    - TK vs cuBLAS: TK’s single GEMM kernel “competes” with cuBLAS across sizes (Figure 7). Grid‑level persistent launch improves TK at low K (Table 2), e.g.:
      > Table 2: `M=N=4096` — TK improves 93→108 TFLOPs at `K=64`, 271→309 at `K=256`, and remains competitive 565→600 at `K=1024`.
  - Attention (forward/backward, causal/non‑causal, D∈{64,128})
    - Forward inference: TK matches FA‑3 across sequence lengths (Figure 8, top‑left).
    - Backward pass: TK outperforms FA‑3 by 10–40% depending on sequence length (Figure 8, right). For example:
      > Figure 8 (Attn Bkwd, B=16, H=16, D=128): TK’s bars exceed FA‑3 by roughly 40% at short context (768) and ~10% at long (12288).
  - Linear attention
    - TK vs Flash Linear Attention (Triton): up to 14× faster on polynomial kernels and 6.5× on learned feature maps (Figure 9, middle row).
  - Long convolution / state space models (SSMs)
    - TK vs FlashFFTConv: 4.7× faster at sequence length 4096 and 7.9× at 1024; up to 8.7× over PyTorch FFT ops (Figure 9, top‑right and caption).
  - Kernel fusion for common ops
    - Rotary and fused dropout‑residual‑layernorm also run faster than popular Triton baselines (Figure 9, bottom row).
- Profiling evidence (how the speedups arise)
  - Attention backward (FA‑3 vs TK), NSight Compute (Table 4):
    > TK shows higher issue‑slot utilization (34.8% vs 25.1%), higher HBM throughput (490 vs 328 “TPS” units), ~10% fewer HBM stall cycles (1.63 vs 1.83), and 85% fewer SMEM stall cycles (0.14 vs 0.92).
    - Interpretation: TK maintains similar tensor‑core utilization but reduces memory‑system stalls through better layout and synchronization, confirming the design intent of the managed layouts and LCSF.
  - Long convolution (FlashFFTConv vs TK), NSight Compute (Table 4):
    > TK increases tensor‑core utilization ~4.1× (54.8% vs 13.4%), boosts issue‑slot utilization (40.0% vs 25.5%), and reduces both HBM and SMEM stalls.
    - Interpretation: warpgroup MMAs and TK’s pipeline keep tensor cores busy while hiding I/O.
- Ablations and trade‑off studies
  - Pipeline depth (Table 1): 1→4 stages lifts GEMM 260→760 TFLOPs—direct evidence that the LCSF pipeline overlaps memory with compute.
  - Occupancy (Figure 6): performance rises with more workers then drops from contention; TK’s LCSF curve dominates the synchronous baseline across worker counts.
  - Block order and L2 reuse (Table 3): careful ordering roughly halves TFLOPs on large GEMM if chosen poorly and can 10× HBM traffic for attention, illustrating the importance of grid‑level scheduling.
- Assessment of evidence
  - The results are broad (GEMM, attention variants, linear attention, FFT convolution/SSMs, rotary, fused layers), and the profiles pinpoint mechanisms (tensor‑core utilization, bank conflicts, I/O stalls), not just end numbers (Section 4.2, Table 4). Ablations isolate template levers (Table 1, Figure 6), and grid‑level studies show sensitivity to launch order (Table 3). Together, they convincingly support TK’s central claim: a small set of abstractions can achieve state‑of‑the‑art performance across diverse AI kernels.

## 6. Limitations and Trade-offs
- Hardware specificity and portability
  - TK’s flagship results target NVIDIA H100 features (TMA for async copies, WGMMA/HGMMA tensor‑core instructions, 32/64/128B swizzle support). Although Section 2.1 notes the hierarchy is common across vendors, the paper does not present results on AMD/Apple or older NVIDIA GPUs. Portability may require alternative backends or reduced feature use.
- Fixed, opinionated tile choices
  - Register/shared tiles are designed around 16×16 tensor‑core compatibility. While TK supports a range of widths/heights through templating, the best layouts are selected from three swizzles; some shapes still have unavoidable 2‑way conflicts if they fall outside the “sweet spots” (Figure 4 bottom; Appendix C).
- Manual—but small—tuning loop
  - Developers still choose occupancy (number of compute vs producer warps), pipeline depth, and block order. TK makes these few knobs effective, but it does not provide an auto‑tuner out of the box (Sections 3.2–3.3; Figure 6, Tables 1–3).
- Scope boundaries
  - Focus is single‑GPU kernels. Multi‑GPU, inter‑GPU communication, and kernel fusion at whole‑graph level are outside scope. Compiler‑level transformations (operator fusion across a model graph) are left to external frameworks (Section 2.3).
- Debuggability vs abstraction
  - TK improves safety via compile‑time checks, but asynchronous programming with barriers and warp specialization still demands care. While simpler than CUTLASS, it is lower‑level than Triton and requires C++/CUDA familiarity.

## 7. Implications and Future Directions
- How this changes the landscape
  - TK demonstrates that peak‑class GPU kernels need not require massive template stacks or bespoke schedulers. A small, unified template and managed layouts can cover GEMM, attention and its variants, SSMs, and common utility ops—shifting kernel development toward a simpler, reusable pattern (Sections 3–4).
- What it enables next
  - Practical avenues:
    - Rapid prototyping of new AI layers (e.g., novel attention/SSM variants) with competitive performance using a few dozen lines of code (Appendix B shows full kernels).
    - Production deployment in inference providers and latency‑sensitive settings like HFT, where persistent grids and cache‑aware ordering matter (Section 1).
  - Research directions:
    - Auto‑tuning and scheduling search over TK’s small knob set (pipeline depth, occupancy, block order) integrated with model compilers.
    - Cross‑vendor portability layers (mapping tiles/LCSF onto AMD/Apple async copy and matrix units).
    - Automated selection of swizzle/layout at runtime for dynamic shapes or mixed precisions; exploration of broader tile families while retaining hardware compatibility.
    - Graph‑level fusion that emits TK kernels under the hood, combining compiler IRs with TK’s low‑level efficiency (Section 2.3 connection to TVM/XLA/Triton).
- Downstream applications
  - High‑throughput training and inference of long‑context LLMs/SSMs (demonstrated 6–14× speedups on linear attention and FFT conv; Figure 9).
  - Custom kernels for memory‑bound routines (rotary, normalization, fused ops) that can be co‑scheduled with compute‑heavy steps using the same template (Figure 9, bottom).
  - Education and maintainability: TK’s small code footprint (<1 MB; Appendix A, Table 5) and PyTorch‑like API lower the barrier for students and practitioners to learn GPU kernel optimization without sacrificing performance.

Overall, ThunderKittens contributes a concrete, mechanism‑driven answer to a long‑standing question in GPU programming: can we get both simplicity and speed? By aligning a tiny set of abstractions with how GPUs really execute work (tiles in tensor‑core layouts, asynchronous pipelines, and cache‑aware grids), and by verifying with ablations and profiles (Tables 1–4; Figures 6–9), the paper shows that for many AI kernels, the answer is yes.
