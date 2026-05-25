# TVM: An Automated End-to-End Optimizing Compiler for Deep Learning

**ArXiv:** [1802.04799](https://arxiv.org/abs/1802.04799)

## 🎯 Pitch

TVM introduces a fully automated compiler stack that transforms high-level deep learning models from popular frameworks into highly optimized, hardware-specific code for CPUs, GPUs, and a wide spectrum of accelerators—including previously unsupported or custom hardware. By unifying graph-level and operator-level optimizations with a novel learning-based auto-scheduler, TVM achieves or exceeds the performance of hand-tuned vendor libraries, enabling rapid deployment and efficient execution of AI workloads across diverse devices without manual tuning or specialized libraries. This breakthrough drastically reduces engineering effort and accelerates innovation by making state-of-the-art inference portable and performant everywhere.

---

## 1. Executive Summary (2-3 sentences)
This paper introduces TVM, an end-to-end optimizing compiler that turns high-level deep learning (DL) models into hardware-specific, high-performance executables for CPUs, GPUs, and specialized accelerators. It combines graph-level rewrites, a tensor-level program representation with new optimization primitives, and a learning-based auto-scheduler to achieve performance competitive with or better than hand-tuned vendor libraries (e.g., cuDNN) across diverse hardware, and it can target novel accelerators without bespoke operator libraries.

## 2. Context and Motivation
- Problem gap
  - DL frameworks (e.g., TensorFlow, MXNet, PyTorch) typically rely on vendor-specific operator libraries (e.g., cuDNN, cuBLAS). This constrains performance portability and makes it costly to support new hardware or new operator variants.
  - Graph-level optimizations alone (e.g., common subgraph elimination, automatic differentiation) are too high-level to realize the backend-specific operator transformations needed for peak speed, especially when operators are fused or when a backend has unusual compute/memory primitives (Section 1; Figure 1).
  - Frameworks face a trade-off: avoid graph rewrites that create unsupported fused operators, or run suboptimal unfused operators (Section 1).
- Why it matters
  - DL now runs from cloud servers to mobile/embedded devices to custom accelerators (FPGAs/ASICs). Each has distinct compute units and memory hierarchies (Figure 1), so manually re-optimizing per device is costly and slows down deployment of new models and hardware.
- Prior approaches and limits
  - Hand-optimized operator libraries: excellent on supported devices/operators, but opaque, hard to fuse across operators, and expensive to port to new hardware or emerging operators (e.g., depthwise convolution).
  - High-level compiler IRs (e.g., TensorFlow XLA, DLVM): help with graph-level rewrites, but still rely on handcrafted operator libraries or fixed lowering rules, requiring substantial backend engineering to reach peak performance (Section 7).
  - Auto-tuning in HPC (e.g., ATLAS, FFTW) and recent ML auto-tuners (e.g., Tensor Comprehensions): powerful, but can require many trials and may not address accelerator-specific synchronization/memory features (Sections 5, 6.1).
- Positioning
  - TVM proposes a full-stack compiler that spans graph rewrites and operator-level code generation with a new “tensor expression + schedule” layer, extends schedule primitives beyond Halide to GPUs/accelerators, and uses an ML-based cost model to efficiently search huge optimization spaces. It targets CPUs, GPUs, and a custom FPGA-based accelerator with a single system (Figure 2; Sections 3–6).

## 3. Technical Approach
TVM’s pipeline (Figure 2) transforms a framework model into deployable code through three layers: graph rewriting, tensor-level code generation with hardware-aware schedules, and automated schedule search.

1) High-level graph rewriting (Section 3)
- Representation
  - TVM uses a computational graph: nodes are tensor operations; edges are data dependencies. Operations have attributes like strides and padding (Figure 3).
- Optimizations
  - Operator fusion: fuse compatible chains of ops to avoid writing intermediates to memory.
    - TVM classifies ops into four categories: `injective` (elementwise maps), `reduction` (e.g., sum), `complex-out-fusable` (e.g., conv2d allows fusing maps on its output), and `opaque` (non-fusable, e.g., sort). It applies generic fusion rules across these categories (Section 3).
    - Impact: reduces memory traffic; Figure 4 shows speedups of 1.2×–2× for fused conv+bn+relu, depthwise-conv+bn+relu, and RNN/LSTM-cell cases on a Titan X GPU.
  - Constant folding: precompute statically resolvable parts.
  - Static memory planning: pre-allocate buffers for intermediate tensors.
  - Data layout transformation: choose backend-friendly internal layouts and insert layout conversions where producer/consumer layouts differ (Section 3). Example: tile data into small blocks to match accelerator tensor primitives (e.g., 4×4 tiles).

2) Operator-level program generation via tensor expressions and schedules (Section 4)
- Tensor expression language
  - Each operator is written as an index-based formula that defines how to compute every output element from inputs (Section 4.1). Example: transposed matrix multiplication uses a reduction axis over `k` to sum products into `C[y, x]`.
  - Crucially, this compute description does NOT specify loop order, tiling, memory placement, or threading. These are controlled by a `schedule`.
- Schedule primitives and lowering (Figure 6)
  - TVM adopts Halide-style separation of compute and schedule for loop transformations, thread binding, and compute locality.
  - It introduces three new families of primitives tailored to GPUs and accelerators:
    - `special memory scope`: place temporary tensors in on-chip shared memories or accelerator-specific buffers to enable cross-thread reuse and explicit memory management (Sections 4.2–4.3).
    - `tensorization`: replace a sub-computation (e.g., an inner 8×8 matrix multiply) with a target-specific “tensor intrinsic” (a hardware micro-kernel) (Section 4.3).
    - `latency hiding`: for accelerators that do not automatically hide memory latency, TVM inserts fine-grained synchronization and interleaves load/compute/store to overlap memory and compute (Section 4.4).
- Example end-to-end scheduling on a specialized accelerator (Figure 5)
  - Start with a high-level matrix multiplication tensor expression.
  - Apply loop tiling to create small tiles (`tile(y, x, k, 8, 8, 8)`).
  - Cache tiles into accelerator-local buffers using `special memory scope` (e.g., `acc_buffer`, `inp_buffer`) so tiles can be reused on-chip.
  - Map the innermost 8×8 compute to a hardware tensor primitive via `tensorize`.
  - TVM lowers the final schedule to hardware-specific low-level code, emitting intrinsic calls (e.g., `vdla.fused_gemm8x8_add`) and DMA copies for tiled data.
- GPU-specific nested parallelism with cooperation (Section 4.2; Figure 7)
  - Concept: separate “shared-nothing” parallelism (each thread only accesses its own data) from “cooperative fetching,” where a group of threads loads a shared tile into `shared` memory for reuse.
  - Mechanism: TVM’s `memory scope` marks compute stages as `shared`; the compiler inserts synchronization barriers so all threads see the shared tile. Cooperative loads reduce global memory traffic and improve occupancy. Figure 7 shows substantial runtime reductions for GEMM on Titan X when cooperative fetching is enabled.
- Tensorization for hardware intrinsics (Section 4.3)
  - Define a `tensor intrinsic` by providing: (1) a high-level tensor expression that describes the intrinsic’s semantics and (2) a lowering rule that emits the correct hardware instructions for reset, compute, and update phases.
  - The `tensorize` schedule primitive pattern-matches a sub-computation in the scheduled loop nest and replaces it with the intrinsic (example: 8×8 GEMM micro-kernel).
  - Benefit: cleanly decouples schedules from hardware-specific instructions; easy to add new accelerators/intrinsics. It also enables using hand-optimized micro-kernels where helpful (e.g., ultra-low-precision bit-serial micro-kernels on mobile CPUs), yielding up to 1.5× speedups (Section 4.3).
- Explicit memory latency hiding for accelerators (Section 4.4; Figures 8–10)
  - Targeted hardware: decoupled access-execute (DAE) accelerators, where the hardware offers separate load (ld), compute (ex), and store pipelines with explicit queues and minimal control logic. The compiler must orchestrate overlap and enforce correctness via dependencies (Figure 9).
  - Virtual threading: programmers write a high-level threaded schedule (as if multithreading exists). TVM lowers it into a single instruction stream with explicit “push/pop dependency” tokens that encode read-after-write and write-after-read constraints between ld/ex/st pipelines (Figure 8). The hardware uses these tokens to recover pipeline parallelism safely.
  - Effect: hides memory latency and increases utilization. On an FPGA-based accelerator, TVM’s latency hiding raised peak compute utilization from 70% to 88% (Figure 10).

3) Automated schedule optimization (Section 5)
- Schedule space specification (Section 5.1)
  - Developers can write schedule templates with tunable knobs (e.g., tile sizes, loop order, unroll factors, memory placement). TVM also provides generic templates that infer knobs from the tensor expression.
  - Realistic models yield billions of valid schedule configurations; exhaustive search is infeasible.
- ML-based cost model (Section 5.2; Figure 13)
  - Input: the lowered loop program (its abstract syntax tree, or AST) and annotations (e.g., vectorized, unrolled).
  - Features: memory access counts and reuse ratios per buffer at each loop level; one-hot encodings of loop annotations; or alternatively, a TreeRNN that directly digests the AST (Figure 13).
  - Model: gradient tree boosting (XGBoost) with a ranking objective (predicts relative order rather than exact runtime). It is fast to train and predict (0.67 ms per prediction on average), enabling tight inner loops of exploration.
  - Continual learning: the model starts with no data and improves as it sees measured runtimes from on-device trials.
- Schedule exploration (Section 5.3; Figure 12)
  - Uses parallel simulated annealing guided by the cost model. It walks to neighboring configurations and keeps moves that are predicted to reduce cost (estimated runtime).
  - Empirical outcome: on Titan X for a ResNet-18 conv2d, the ML-guided explorer finds faster kernels in fewer trials than random search or a genetic algorithm; the curve in Figure 12 shows the ML approach surpasses cuDNN’s baseline more quickly.
- Distributed device pool via RPC (Section 5.4; Figure 11)
  - TVM compiles candidate kernels on the host and runs them on a pool of real devices (e.g., Raspberry Pi, Mali GPU, Nvidia GPU, FPGA) through an RPC interface to collect true runtimes.
  - This enables scalable, fine-grained measurement campaigns across heterogeneous hardware.

Implementation note: Core is ~50k LoC in C++ with Python/Java bindings; the specialized accelerator backend used here was built with ~2k lines in Python (Section 6.4).

## 4. Key Insights and Innovations
- End-to-end performance portability without hand-written operator libraries
  - What’s new: TVM spans from graph-level rewrites (fusion, layout) to low-level kernel codegen for diverse backends, automatically generating fused operators rather than relying on pre-baked kernels (Figure 2; Section 3).
  - Why it matters: unlocks fused, backend-specific optimizations while avoiding combinatorial growth of library code and enabling new hardware targets (Sections 1, 3).
- New schedule primitives that unlock GPU and accelerator performance (Figure 6)
  - `special memory scope` for explicit shared/local memory placement (GPU shared memory, accelerator SRAMs) and compiler-inserted synchronization (Section 4.2).
  - `tensorization` and `tensor intrinsic` declarations to harness emerging tensor compute units and micro-kernels systematically (Section 4.3).
  - `latency hiding` via virtual threading and compiler-inserted dependency tokens for DAE accelerators (Sections 4.4; Figures 8–9).
  - Significance: these extend Halide’s model to modern DL hardware; they are essential for peak GPU/accelerator performance.
- Learning-based schedule search that adapts across hardware (Section 5)
  - What’s different: instead of black-box auto-tuning or hand-written cost models, TVM’s ML model learns from measured data, predicts relative performance of loop programs, and guides exploration efficiently (0.67 ms prediction; Figure 12).
  - Impact: finds state-of-the-art kernels in far fewer trials than random/genetic baselines and can reuse knowledge across related workloads (Figure 12; Table 1).
- Concrete accelerator bring-up case study (VDLA) with compiler-driven latency hiding (Section 6.4)
  - TVM quickly targets a custom FPGA-based accelerator through tensorization and virtual-thread lowering, achieving large kernel speedups (40× on offloaded conv layers) and higher utilization (Figure 10, Figure 21).
  - Significance: demonstrates a practical path to bring up new accelerators without hand-coding vast operator libraries.

## 5. Experimental Analysis
- Setup (Section 6)
  - Hardware
    - Server GPU: Nvidia Titan X.
    - Embedded CPU: ARM Cortex-A53 (Quad-core, 1.2GHz).
    - Embedded GPU: ARM Mali-T860MP4 (Firefly-RK3399 board).
    - Accelerator: FPGA-based “Vanilla Deep Learning Accelerator (VDLA)” on a PYNQ board (Artix-7, dual-core Cortex-A9 @ 667MHz). The design includes a 16×16 8-bit×8-bit→32-bit matrix-vector unit @ 200MHz; ~102.4 GOPS theoretical peak; on-chip buffers: 32KB each for activations, weights, microcode; 128KB register file (Section 6.4; Figure 20).
  - Workloads
    - ResNet-18, MobileNet, LSTM language model, DQN, and DCGAN (Section 6).
  - Baselines
    - Server GPU: MXNet (v1.1) and TensorFlow (v1.7) using cuDNN v7 and cuBLAS v8; TensorFlow XLA (JIT compiled).
    - Embedded CPU: TensorFlow Lite (commit 7558b085).
    - Embedded GPU: ARM Compute Library v18.03.
    - Operator-level baseline: Tensor Comprehensions (commit ef644ba) with 2000 trials per operator; handcrafted low-precision Caffe2 kernels (commit 39e07f7) (Sections 6.1–6.2).
  - Metrics
    - End-to-end inference time (ms/s), relative speedups vs baselines, per-operator speedups; utilization roofline for accelerator (Figure 10).
- Main results
  - Graph fusion benefit
    - > “Fused operators generate up to a 1.2× to 2× speedup by reducing memory accesses.” (Figure 4; Section 3)
  - GPU end-to-end performance (Figure 14)
    - TVM outperforms MXNet, TensorFlow, and TensorFlow XLA on Titan X across models.
    - Reported speedups range from roughly 1.6× to 3.8×; the largest gain (≈3.8×) occurs on DQN due to nonstandard conv settings (4×4 kernel, stride 2) that cuDNN does not optimize well; TVM finds optimized kernels automatically (Section 6.1; Figure 14).
  - GPU operator-level analysis (Figure 15; Table 2)
    - For standard conv2d layers in ResNet-18 (C1–C12), TVM frequently outperforms cuDNN; it also supports a “pretransformed Winograd” path for 3×3 conv (TVM PT).
    - For depthwise conv2d (MobileNet’s D1–D9), both TVM and Tensor Comprehensions find fast kernels, beating MXNet’s hand-crafted baseline; TVM’s gains come from a larger schedule space and ML-guided search (Section 6.1).
  - Embedded CPU end-to-end (Figure 16) and per-operator (Figure 17)
    - TVM beats TFLite on ResNet-18, MobileNet, and DQN on an ARM A53. Per-operator relative speedups are >1× across most conv2d/depthwise conv2d configurations (Table 2 lists exact shapes).
  - Ultra low-precision operators on CPU (Figure 18)
    - For ResNet 2-bit activations, 1-bit weights:
      - Single-threaded TVM exceeds the single-threaded Caffe2 baseline, especially on 1×1 conv layers (C5, C8, C11), where the baseline is less optimized.
      - Multi-threaded TVM further improves performance; 1×1 layers show smaller multi-threading gains due to lower compute intensity (Figure 18; Section 6.2).
  - Embedded GPU end-to-end (Figure 19)
    - On Mali-T860MP4, TVM outperforms ARM Compute Library on ResNet-18, MobileNet, and DQN for both float32 and float16 with speedups ≈1.2×–1.6× (Section 6.3).
  - Accelerator (FPGA) utilization and end-to-end (Figures 10, 21)
    - Latency hiding effectiveness: > “Peak compute utilization increased from 70% with no latency hiding to 88% with latency hiding.” (Figure 10; Section 4.4)
    - Offloaded conv layers ran ≈40× faster than on the Cortex-A9 CPU; overall speedup is limited by unfused/non-accelerated layers (Amdahl’s law), highlighting the value of broader accelerator coverage (Figure 21; Section 6.4).
  - ML-based auto-scheduler efficiency (Figures 12–13)
    - > “The ML-based model starts with no training data and uses the collected data to improve itself.” It identifies faster configs in far fewer trials than random search or genetic algorithms; prediction cost averages 0.67 ms (Figure 12; Section 5.2).
- Do experiments support the claims?
  - Yes. The paper presents:
    - End-to-end improvements on three hardware classes (server GPU, embedded CPU/GPU) across multiple models (Figures 14, 16, 19).
    - Operator-level breakdowns confirming TVM’s kernels are competitive with or exceed highly optimized libraries (Figure 15).
    - A concrete accelerator case study showing both utilization gains (Figure 10) and large kernel speedups (Figure 21).
    - An ablation-like comparison of the auto-scheduler strategies (Figure 12).
- Notable components/ablations
  - Fusion vs no fusion (Figure 4).
  - Cooperative fetching vs not (Figure 7).
  - Latency hiding vs not (Figure 10).
  - ML-guided vs random vs genetic schedule search (Figure 12).

## 6. Limitations and Trade-offs
- Specialization to known tensor shapes
  - TVM’s optimizations (fusion, layout, schedules) are most effective when input tensor shapes are known and fixed. Dynamic-shape workloads may require additional runtime mechanisms or conservative schedules (Section 3 mentions optimizing for a fixed set of input shapes).
- Search cost and engineering overhead
  - Although the ML model reduces trials, hundreds of real-device measurements may still be needed for each operator/hardware pair (e.g., Figure 12 runs up to 800 trials). This creates tuning time and requires access to representative hardware via RPC (Section 5).
  - New accelerators still need `tensor intrinsic` declarations and runtime interfaces. While far less work than a full operator library, this is non-trivial engineering (Section 4.3; Section 6.4 mentions ~2k LoC to add the VDLA backend).
- Incomplete accelerator coverage limits end-to-end gains
  - In the FPGA case, substantial end-to-end time remains on the CPU (residuals, activations), limiting overall speedup despite 40× faster conv layers (Figure 21). Real-world deployments need broader operator coverage on accelerators.
- Model bias and portability of the cost model
  - The learned cost model uses features from loop programs and is trained per hardware target. While it improves with data, it can mis-rank configurations, and cross-device generalization is not guaranteed (Table 1; Section 5.2).
- Scope of workloads
  - The evaluation focuses on inference of CNNs, an LSTM LM, DQN, and DCGAN. Training workloads, dynamic control flow, and very sparse/tensor-irregular computations are not the focus here. Sparse tensor compilation is outside scope (Section 7 cites TACO for sparse).
- Memory constraints and layout transforms
  - Sophisticated layout choices boost speed but add conversion overheads between mismatched producers/consumers. TVM inserts these automatically, but suboptimal layout choices could negate gains if not searched or specified carefully (Section 3).

## 7. Implications and Future Directions
- How this changes the landscape
  - TVM demonstrates that a single, general compiler can meet or exceed vendor libraries while supporting hardware diversity and fused kernels. It reduces the need for per-backend operator libraries and provides a practical path to quickly bring up novel accelerators through declarative intrinsics and latency-hiding schedules (Figures 6, 8–9, 20–21).
- Enabled directions
  - Faster hardware bring-up: define tensor intrinsics and scheduling rules for new accelerators, then let the auto-scheduler find high-performing implementations.
  - Broader operator coverage on accelerators: extend the accelerator ISA (or intrinsic set) to include activations, residuals, and other ops to avoid CPU fallbacks and realize end-to-end speedups (Section 6.4).
  - Improved auto-scheduling: explore richer models (e.g., graph neural nets on loop IR), better transfer learning across operators/devices, and multi-objective search (latency, memory, energy).
  - Dynamic shapes and control flow: integrate runtime shape polymorphism and speculative compilation to expand beyond fixed-shape inference.
  - System integration: use TVM as a portable back-end across frameworks and edge/cloud environments to standardize deployment, reduce engineering overhead, and enable joint graph–kernel optimization.
- Practical applications
  - Edge deployment on phones/embedded devices where vendor libraries are limited or missing operators (depthwise conv, low-precision).
  - Cloud inference services that need to serve diverse models across CPU/GPU fleets with consistent performance.
  - Accelerator prototyping and evaluation (e.g., FPGA/ASIC) where rapid iteration on micro-architectural features (tensor primitives, memory hierarchies) is needed.

> “Experimental results show that TVM delivers performance across hardware back-ends that are competitive with state-of-the-art, hand-tuned libraries for low-power CPU, mobile GPU, and server-class GPUs” and it “achieves speedups ranging from 1.2× to 3.8×” (Abstract; Figures 14, 16, 19).

> “Peak compute utilization increased from 70% with no latency hiding to 88% with latency hiding” on the FPGA-based accelerator (Figure 10).

> The ML cost model “starts with no training data,” updates from measured runs, and predicts in “0.67 ms,” enabling efficient exploration that outperforms random and genetic search (Sections 5.2–5.3; Figure 12).
