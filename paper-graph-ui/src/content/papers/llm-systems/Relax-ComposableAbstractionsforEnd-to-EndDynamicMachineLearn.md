# Relax: Composable Abstractions for End-to-End Dynamic Machine Learning

**ArXiv:** [2311.02103](https://arxiv.org/abs/2311.02103)

## 🎯 Pitch

Relax introduces a unified compiler abstraction that bridges the gap between high-level computation graphs, loop-level tensor programs, and vendor operator libraries, while pioneering first-class symbolic shape annotations to enable end-to-end optimization of dynamic-shape machine learning workloads—including large language models. This cross-level approach empowers ahead-of-time (AOT) compilation with dynamic-shape–aware operator fusion, memory planning, and deployment to a wide range of backends (GPUs, mobiles, browsers), delivering performance and portability previously out of reach for dynamic models and supporting the next generation of machine learning systems.

---

## 1. Executive Summary
Relax is a new compiler abstraction and end-to-end compilation framework that makes dynamic‑shape machine learning models (especially LLMs) run efficiently and portably. It unifies three traditionally separate layers—graph IR, loop‑level tensor programs, and vendor libraries—into a single cross‑level representation, and adds first‑class symbolic shape annotations so the compiler can optimize across levels even when tensor sizes are only known at runtime (§§3–4, Fig. 1).

The result is a practical AOT (ahead‑of‑time) pipeline that delivers competitive performance to state‑of‑the‑art systems across NVIDIA, AMD, and Apple GPUs, enables CUDA Graphs for dynamic models, cuts memory with dynamic‑shape–aware planning, and deploys modern LLMs to emerging backends like mobile and WebGPU (§5, Figs. 14–20, Table 2, Table 3).

## 2. Context and Motivation
- Problem addressed
  - Modern ML workloads, notably large language models (LLMs), contain dynamic shapes: dimensions (e.g., batch size, sequence length, KV-cache length) that are unknown until runtime. This breaks many classic compiler optimizations such as static memory planning and cross‑operator fusion that assume fixed sizes (§1, §2).
  - Existing ML compilers typically use multiple IR layers—graph IR, loop‑level tensor IR, and vendor libraries—and “single‑shot” lowering between them. Each layer tends to treat the others as opaque, which prevents analyses or transformations that require information to flow across layers (§1, §2, Fig. 1).
- Why this matters
  - Real systems need to deploy models onto heterogeneous backends (servers, laptops, phones, embedded devices, browsers). JIT approaches that trace shapes at runtime often don’t fit constrained or sandboxed environments (e.g., mobile, WebGPU). AOT compilation with whole‑program optimization is needed (§1, §2).
  - Without dynamic‑shape reasoning, systems fall back to general-purpose runtime allocators, pay kernel‑launch overheads, and miss fusion opportunities—hurting latency, throughput, and memory usage (§1, §4.3–§4.5).
- Prior approaches and their gaps
  - Graph IRs with unknown shapes: Relay and several MLIR dialects represent dynamic dimensions as “unknown,” which loses relations between dimensions (e.g., output has size 4×n) and limits optimization (§1, §3.2, Fig. 3).
  - JIT tracing: PyTorch’s compiler tracks shapes per traced function, sidestepping cross‑function shape tracking, but limiting AOT portability to environments like mobile/WebGPU (§1).
  - Loop‑level work: Halide, DietCode, CoRA, SparseTIR optimize tensor programs and sometimes call libraries from within kernels, but they do not solve the cross‑level, whole‑program optimization with dynamic shapes (§1, §6).
- Positioning
  - Relax introduces a cross‑level program abstraction that integrates graph‑level, loop‑level (TensorIR), and external libraries in one IR, plus first‑class symbolic shapes tracked across function boundaries. This enables AOT, whole‑program, shape‑aware optimizations that span levels (§§3–4, Fig. 1).

## 3. Technical Approach
Relax consists of two pivotal ideas and a concrete optimization/lowering pipeline.

A) First‑class symbolic shapes with interprocedural tracking (§3.2, Fig. 3; Table 1)
- What: Represent dynamic sizes as symbolic expressions over integer variables (e.g., `n`, `4*n`, `n+1`). These annotations live on values (tensors, tuples, shapes), can be passed as first‑class values, and are preserved through transformations.
- Why: This preserves relations like “flatten of `(n, 2, 2)` has length `4*n`,” enabling the compiler to reason about equality, reuse memory, and validate fusion even when `n` is not known at compile time (Fig. 3).
- How:
  - Forward symbolic deduction: Each operator has a shape rule that computes the output annotation from input annotations/values. Relax performs efficient forward propagation across the program and across function boundaries (§4.1).
  - `match_cast`: For data‑dependent shapes (e.g., `unique`), the compiler allows inserting a checked assertion that a value conforms to a symbolic annotation, with lightweight runtime checks (§3.2, Fig. 3).
  - Function signatures carry shape relations: Functions (including subgraphs) declare parameter/return annotations so callers can infer output shapes without seeing the callee body (§4.1, Fig. 7).
  - Symbolic expressions in parameter types: Fusions can pass additional shape arguments (e.g., `shape(n)`) when fused bodies refer to `2*n` to keep fused functions well‑typed (Fig. 8).

B) Cross‑level abstraction with foreign calls in a single IR (§3.3, Figs. 4–5)
- What: Graph‑level functions can call:
  - Loop‑level tensor programs using `call_tir`, and
  - External vendor/library functions using `call_dps_library`.
- Why: Unifies the three levels so the compiler can partially lower some regions, analyze or transform others, and feed analyses back across levels (Fig. 6).
- How:
  - Destination‑passing style (DPS): Many low‑level functions accept an output buffer to write into. Relax models `call_tir` so the graph IR allocates the destination, passes it to the low‑level kernel, and returns the tensor view (Fig. 5). This keeps low‑level code simple and lets the graph level own memory management.
  - Shape‑aware calls: `call_tir` carries an explicit output annotation and optional symbolic arguments so TensorIR can specialize on static dimensions and keep only truly dynamic ones (Fig. 4).

C) Algorithms and cross‑level optimizations (§4, Figs. 9–13; Algs. 1–3)
1) Dynamic‑shape‑aware operator fusion (graph + tensor levels) (§4.2, Fig. 9)
   - Compute‑pattern analysis on tensor programs (Alg. 1) classifies kernels (e.g., `ElementWise`, `Injective`, `Reduction`, `OutputEwiseFusible`).
   - Graph partitioning/fusion (Alg. 2, “FuseOps”) groups calls (including custom tensor programs) into subgraph functions based on patterns (e.g., fusing `matmul` [OutputEwiseFusible] with following elementwise ops).
   - Cross‑level “FuseTensorIR” merges the tensor programs called in the subgraph into a single TensorIR function, preserving symbolic shapes (Fig. 9).
   - Example: Fusing a custom “quantization decode” loop with a matmul, no bespoke high‑level operator required (Fig. 9, left→right).

2) Dynamic‑shape‑aware memory planning (§4.3, Fig. 10; Alg. 3)
   - Step 1: Lower `call_tir` and `call_dps_library` to explicit allocations and DPS calls so allocations are visible (Fig. 5).
   - Step 2: Perform liveness analysis; maintain a storage pool that compares required sizes using symbolic equality (Alg. 3, lines 7–13).
   - Step 3: Reuse storage if shapes match symbolically; otherwise allocate new. Optionally use user‑provided upper bounds (e.g., max context length) to pre‑allocate once, even with dynamic shapes (§4.3).

3) Cross‑level workspace lifting (§4.4, Fig. 11)
   - Detect global workspace allocations inside tensor programs (e.g., Stream‑K matmul’s partial accumulations).
   - Lift these to graph level as explicit buffers passed into `call_tir` and then include them in global memory planning (Fig. 11).

4) CUDA Graph offloading for dynamic models (§4.5)
   - CUDA Graphs reduce per‑kernel launch overhead by capturing a graph of launches, but require fixed, pre‑allocated memory.
   - After static planning, detect subgraphs that satisfy capture constraints; insert runtime builtins to “capture on first run, replay thereafter” (§4.5). This extends CUDA Graphs to dynamic‑shape models by making memory static at capture time.

5) Partial lowering and operator optimization (§4.6, Fig. 12)
   - Pattern‑match and partially lower subgraphs to vendor libraries (e.g., matmul with specific epilogues to cuBLAS/CUTLASS) while compiling the rest to TensorIR.
   - Complement with schedule rules for TensorIR (and optional autotuning for hard cases) so library and codegen approaches compose (§4.6).

D) End‑to‑end pipeline (§4.7, Fig. 13)
- Order: partial library lowering → generate TensorIR for remaining high‑level ops → fusion → workspace lifting → memory planning → CUDA Graph offloading → build runnable module.
- Build: erase annotations, compute runtime values of symbolic expressions via a compact “symbol table” tensor, generate GPU code for TensorIR, and package with a small VM that issues low‑level calls (§4.7).

Key definitions used above:
- `symbolic shape`: an expression like `(n, 256)` where `n` is a runtime‑known variable tracked symbolically.
- `dataflow block`: a side‑effect‑free straight‑line region that simplifies transformation (Fig. 2).
- `TensorIR`: TVM’s loop‑level IR for writing/scheduling kernels with explicit loops and buffers.
- `destination‑passing style (DPS)`: a callee writes results into a caller‑provided buffer instead of allocating internally.
- `CUDA Graph`: a CUDA feature that records a sequence of GPU operations and replays them to reduce launch overhead; requires fixed memory allocations during capture.

## 4. Key Insights and Innovations
1) Cross‑level program abstraction (fundamental) (§3.3, Fig. 4)
   - What’s new: A single IR where graph nodes can directly call loop‑level kernels (`call_tir`) and vendor libraries (`call_dps_library`), all while the compiler can analyze/transform across these calls.
   - Why it matters: Enables partial lowering, analysis feedback (infer op properties from kernel loops), and cross‑level transformations like workspace lifting (Fig. 6).
   - Difference vs prior: Previous systems largely treated other levels as opaque during graph optimization or did single‑shot lowering, limiting composition and feedback.

2) First‑class, interprocedural symbolic shapes (fundamental) (§3.2, §4.1, Figs. 3, 7–8; Table 1)
   - What’s new: Symbolic shapes with arithmetic expressions propagate across function boundaries, subgraphs, and foreign calls, with runtime checks only when unavoidable (`match_cast`).
   - Why it matters: Retains equalities/relations (e.g., `2*n` reappearing after `transpose`) that enable memory reuse, fusion, and specialization. Forward deduction is fast and sufficient for most cases (§4.1).

3) Dynamic‑shape–aware memory and execution planning (significant) (§4.3–§4.5, Figs. 10–11; Alg. 3)
   - Memory: Uses symbolic equality to reuse buffers and can pre‑allocate using upper bounds to make memory static even when shapes vary (Fig. 10, Alg. 3).
   - Workspace lifting: Moves large temporary buffers out of kernels into the graph so they can participate in global planning (Fig. 11).
   - CUDA Graphs: With static allocations, capture/replay becomes possible for dynamic models (§4.5).

4) Composable partial lowering and fusion with custom tensor programs (significant) (§4.2, §4.6, Fig. 9, Fig. 12)
   - Fusion works even when some ops exist only as custom loop kernels (e.g., quantization decode), because pattern kinds are inferred from TensorIR (Alg. 1) and shape compatibility is preserved symbolically (Fig. 9).
   - Library and codegen approaches compose; the compiler can choose best‑of‑both depending on batch size and backend (Fig. 12, §5.1).

## 5. Experimental Analysis
Setup (§5)
- Models and tasks:
  - LLM decoding latency across batch sizes {1, 16, 32, 64} for `Llama3‑8B`, `Gemma1.1‑7B`, `Qwen2‑7B` with float16 weights/activations (Figs. 14–16).
  - Additional models: `Whisper‑large‑v3` ASR (30‑second transcription time), `LLaVA` (32‑token generation for an image) (Figs. 19–20).
- Hardware:
  - NVIDIA RTX 4090, AMD Radeon 7900 XTX, Apple M2 Ultra; plus emerging platforms (iPhone 14 Pro, Samsung S23, Orange Pi 5, Steam Deck, Jetson Orin, WebGPU on M3 Max) (Table 3).
- Baselines:
  - HuggingFace Transformers (PyTorch eager and `torch.compile`), vLLM, llama.cpp, Faster‑Whisper, WhisperX (where supported) (§5.1, §5.4).
- Metrics:
  - LLMs: per‑token decode latency (ms/token).
  - Mobile/embedded/WebGPU: throughput (tokens/sec), Table 3.
  - Memory: allocated activation memory over varying prefill/decode shapes (Table 2).
- Implementation:
  - Relax on Apache TVM; models compiled once for arbitrary batch sizes and sequence lengths (§5.1).

Main quantitative results
- LLM decode latency (NVIDIA RTX 4090, Fig. 14)
  - Relax is consistently competitive and sometimes best. Quote:
    > “Relax … reduces the decode token latency by up to 27%.” (Fig. 14)
  - Example: For several batch sizes, Relax outperforms HF Transformers and vLLM; llama.cpp is less competitive on NVIDIA GPUs (§5.1).
- LLM decode latency (AMD 7900 XTX, Fig. 15)
  - Relax maintains leading or competitive performance; at batch size 1, it reports up to 1.50× improvement over some baselines (caption of Fig. 15).
- LLM decode latency (Apple M2 Ultra, Fig. 16)
  - Relax is competitive with llama.cpp (which is strong on Apple) and substantially ahead of HF Transformers in many settings (Fig. 16).
- Ablation: where the speedups come from (RTX 4090, Llama3‑8B, Fig. 17)
  - Starting from “no fusion, no partial library, no CUDA Graph,” then add:
    - + operator fusion → noticeable gains.
    - + partial library lowering → the largest jump; up to 27% improvement at larger batch sizes by mapping big matmuls to cuBLAS.
    - + CUDA Graph offloading → an additional ~1–2% by reducing driver launch overhead (§5.2, Fig. 17).
- Memory savings (Table 2)
  - Static memory planning (with upper bounds) vs. runtime allocator:
    > Prefill: 192.7 MiB → 149.7 MiB (−22%); Decode: 150.0 MiB → 88.2 MiB (−40%).
  - This reuse holds even as input shapes change over time (§5.2, Table 2).
- Emerging platforms (Table 3; Fig. 18)
  - Relax enables GPU‑accelerated LLMs on platforms where baselines don’t run or run only on CPU.
  - Throughput (tokens/sec) examples from Table 3:
    > Steam Deck Vulkan: Llama3‑8B 14.0 tok/s; Jetson Orin CUDA: 32.0 tok/s; WebGPU (M3 Max): 37.8 tok/s.
  - On Samsung S24, Relax beats llama.cpp by up to 55% throughput for 4‑bit LLMs (Fig. 18).
- Other models
  - Whisper‑large‑v3 transcription time:
    > Relax is 14% faster than HF Transformers on RTX 4090 and competitive on Apple M2 Ultra (Fig. 19).
  - LLaVA image+text:
    > Relax achieves competitive optimized generation time on both RTX 4090 and Apple M2 Ultra (Fig. 20).

Assessment
- Do experiments support the claims?
  - Yes, the ablation (Fig. 17) shows that composability—library + fusion + CUDA Graph—explains the performance, not just one component.
  - Portability claims are demonstrated across three GPU vendors and in Table 3 for mobile/embedded/WebGPU.
  - Memory planning benefits are quantified and tied to the symbolic‑shape mechanism (Table 2, §4.3).
- Caveats
  - Torch `compile` mode baselines are omitted for some models (e.g., Qwen2‑7B on RTX 4090; Fig. 14 caption notes “lack of support”), and PyTorch/llama.cpp lack Apple GPU or Android GPU support. Where baselines are weaker or absent, absolute competitiveness is clear but relative advantage depends on availability.
  - CUDA Graph gains are modest but consistent (~1–2%; Fig. 17), which matches expectations for launch‑overhead‑dominated regimes.

## 6. Limitations and Trade-offs
- Shape reasoning requires symbolic equality; otherwise the compiler falls back
  - Memory reuse depends on proving symbolic equality (Alg. 3). If expressions differ or are data‑dependent, reuse may be missed. `match_cast` inserts runtime checks but cannot infer relations not stated (§3.2, §4.3).
- Data‑dependent operators need dynamic checks
  - Ops like `unique` produce shapes depending on values. Relax accepts this using `match_cast`, but it introduces runtime validation and possible failure if assumptions are violated (Fig. 3, §3.2).
- Pipeline order and AOT focus
  - The fixed‑order pipeline is engineered for AOT portability (Fig. 13). It avoids JIT‑style profiling/adaptation that might yield extra gains in server settings. Choosing the order (e.g., when to fuse vs. when to dispatch to libraries) can limit some opportunities (§4.7).
- Developer effort shifts to cross‑level patterns and TensorIR
  - While analysis feedback (Alg. 1) reduces per‑op annotations, some expertise is still needed to:
    - Write custom TensorIR for non‑standard ops (e.g., quantization decode).
    - Register partial lowering patterns for libraries and keep them backend‑aware (§4.6, Fig. 12).
- CUDA Graphs need static allocations and steady subgraphs
  - Capture applies only when memory is stable; if a deployment frequently changes upper bounds or toggles graph structure, the benefit may diminish (§4.5).
- Training not covered
  - The paper focuses on inference. Some mechanisms (e.g., gradient memory planning, optimizer state) are unaddressed.
- Precision/layout and library coverage
  - Performance depends on library availability and precision (e.g., FP16). Where vendor libraries lag (new dtypes/layouts), TensorIR must shoulder more optimization work (§4.6, §5.1).

## 7. Implications and Future Directions
- Changes to the landscape
  - Relax demonstrates that dynamic‑shape models can be compiled AOT with performance comparable to specialized systems, while remaining portable (NVIDIA/AMD/Apple GPUs, mobile, WebGPU). The combination of cross‑level IR and symbolic shapes is a compelling template for next‑generation ML compilers (§§3–5).
- What this enables
  - Unified deployment stack: Frameworks can target Relax to run one model binary across desktops, phones, embedded devices, and browsers without per‑backend hand‑kernels (Table 3).
  - Richer dynamic‑shape optimizations: Cross‑level transformations (fusion, workspace lifting) that were previously awkward or impossible become routine because shape relations persist across levels (Figs. 9–11).
  - CUDA Graphs for dynamic models: By statically planning memory via upper bounds, dynamic models can now benefit from capture/replay (§4.5).
- Research directions
  - Stronger shape reasoning: Integrate constraint solvers to infer more complex relations when forward deduction is insufficient, while balancing compile‑time cost (§3.2, §4.1).
  - Automated selection among codegen vs. libraries: Learn‑to‑optimize dispatch that chooses per‑subgraph whether to “compile” or “call a library,” conditioned on runtime ranges of shapes and target hardware (§4.6).
  - Training support: Extend symbolic‑shape planning and cross‑level optimizations to backprop, optimizer states, and distributed training.
  - Broader backends: Apply the same approach to new static execution graph features on emerging GPUs or NPUs (cf. CUDA Graphs; §4.5).
- Practical applications
  - Latency‑sensitive LLM serving with dynamic batching and context lengths (Figs. 14–17).
  - On‑device inference (mobile, embedded) where memory budgets are tight and JIT is infeasible (Table 3).
  - Web‑native ML via WebGPU, enabling privacy‑preserving or low‑latency in‑browser assistants (Table 3).

> In short, Relax’s cross‑level IR and first‑class symbolic shapes turn dynamic shape from an obstacle into a resource the compiler can reason about. This delivers concrete performance and memory wins (up to 27% decode latency reduction and 40% memory reduction in the reported settings; Figs. 14, 17; Table 2) while expanding the set of viable deployment targets (Table 3).
