# Relax: Composable Abstractions for End-to-End Dynamic Machine Learning

**ArXiv:** [2311.02103](https://arxiv.org/abs/2311.02103)
**Authors:** Ruihang Lai, Junru Shao, Siyuan Feng, Steven S. Lyubomirsky, Bohan Hou, Wuwei Lin, Zihao Ye, Hongyi Jin, Yuchen Jin, Jiawei Liu, Lesheng Jin, Yaxing Cai, Ziheng Jiang, Yong Wu, Sunghyun Park, Prakalp Srivastava, Jared G. Roesch, Todd C. Mowry, Tianqi Chen
**Institutions:** 

## 🎯 Pitch

Relax revolutionizes machine learning deployment by unifying computational graphs, loop-level tensor programs, and vendor libraries into a single IR, introducing first-class symbolic shapes for dynamic-shape models. This breakthrough allows for efficient, cross-level optimizations and static-like memory planning, significantly improving performance and deployability across diverse platforms from servers to mobile devices, reshaping end-to-end ML program compilation.

---

## 1. Executive Summary
Relax is a new compiler abstraction and end-to-end compilation framework that makes dynamic-shape machine learning models (especially modern LLMs) easy to optimize and deploy across many backends. It unifies three traditionally separate layers—computational graphs, loop-level tensor programs, and vendor libraries—into one cross-level IR and adds first-class symbolic shapes so the compiler can reason about dynamic dimensions globally, plan memory statically, and perform cross-level optimizations (Figures 1–2, §3–4).

## 2. Context and Motivation
- Problem addressed
  - Modern ML models, especially large language models (LLMs), have dynamic tensor shapes (e.g., variable batch sizes, sequence lengths, KV cache lengths). Dynamic shapes make key optimizations (like buffer reuse and CUDA Graph capture) difficult because shapes are not fully known at compile time (§1).
  - Existing ML compilers typically treat the computational graph, loop-level kernels, and external library calls as separate layers that are lowered in a single shot. This separation makes it hard to analyze and optimize across layers; moreover, dynamic shapes are often tracked only within individual operators or within a JIT-compiled region, not across function boundaries (§1, §2).

- Why this matters
  - Real-world deployment must target diverse environments (servers, mobiles, vehicles, browsers). Without robust AOT (ahead-of-time) compilation and dynamic-shape awareness, models either cannot run (memory fragmentation/overuse), run inefficiently, or require backend-specific engineering (§1, §5.3).
  - For GPUs, the inability to statically plan memory or capture CUDA Graphs leaves significant performance on the table (§4.5).

- Prior approaches and their gaps
  - High-level graph IRs with “unknown” shapes (Relay, MLIR dialects) do not track relationships (e.g., knowing that one dimension is 4× another), limiting optimization (§1, §3.2).
  - JIT compilers (e.g., PyTorch 2/TorchInductor) can trace and compile dynamic regions but avoid cross-function shape tracking and are harder to deploy AOT to constrained platforms like mobile or WebGPU (§1, §6).
  - Loop-level work (DietCode, CoRA, SparseTIR) optimizes dynamic shape kernels but not end-to-end cross-level optimization (§1, §6).
  - Halide can call external libraries from within kernels, but lacks Relax’s unified cross-level program view and global symbolic shape tracking (§1, §6).

- Positioning
  - Relax brings a holistic, AOT-capable program abstraction that (1) spans graph, tensor program, and library levels within a single IR and (2) introduces first-class symbolic shapes to track relationships across the whole program (§2–§3). This enables new cross-level, dynamic-shape-aware optimizations that prior single-layer or JIT-only systems cannot easily perform.

## 3. Technical Approach
Relax provides a language and compilation methodology that unifies multiple abstraction levels and reasons about dynamic shapes globally. The core pieces:

- Language constructs with annotations and dataflow blocks (§3.1; Figure 2; Table 1)
  - Annotations attach structural info to values (e.g., `Tensor((n, 4), "f32")`, `Shape(["n", "m"])`). Shapes can be symbolic expressions over integer variables (`n`, `m`, `n*4`) rather than just “unknown” (§3.2).
  - Dataflow blocks mark pure, straight-line regions (no side effects), simplifying rewriting like dead code elimination.

- First-class symbolic shapes (§3.2; Figure 3)
  - Instead of “unknown” dimensions, Relax uses symbolic variables and expressions to represent dynamic sizes (e.g., `n`, `n*4`). This preserves relationships across operators and functions, enabling:
    - Memory planning even for dynamic shapes (Figure 10; Algorithm 3).
    - Verifying shape consistency and reusing buffers when shapes are provably equal.
  - `match_cast` asserts a shape relation when static deduction is impossible (e.g., for data-dependent ops like `unique`), and the compiler inserts runtime checks (§3.2, Figure 3).

- Cross-level calls that harmonize graph, kernels, and libraries (§3.3; Figures 4–5)
  - `call_tir`: invoke a loop-level tensor program (TensorIR) from the graph. Semantics follow destination-passing style (DPS): outputs are allocated by the caller and passed in, removing allocation from low-level code (Figure 5).
  - `call_dps_library`: similarly calls a vendor library routine by name. Both carry output shape annotations and symbolic arguments so low-level code can specialize appropriately.

- Cross-level optimization patterns (Figure 6, §3.3)
  - Partial lowering: only some ops are lowered at a time (e.g., to libraries), leaving the rest for later passes.
  - Analysis feedback: analyze tensor programs to infer mathematical properties (element-wise, injective, etc.) instead of manually annotating every high-level op.
  - Cross-level transforms: change both graph and kernel code simultaneously (e.g., lift temporary workspaces out of kernels to participate in global memory planning, Figure 11).

- Shape annotation deduction (§4.1; Figure 7–8)
  - Forward, interprocedural deduction of shapes through operators, subgraph calls, and foreign calls. Function signatures carry shape relations so callers can infer output shapes without inlining (§4.1).
  - Parameter annotations support symbolic expressions (e.g., a fused function parameter `Tensor(("n*2",), "f32")`), and passes can add extra symbolic parameters when needed for fusion (Figure 8).

- Cross-level, dynamic-shape-aware operator fusion (§4.2; Algorithms 1–2; Figure 9)
  - Step 1: Analyze each tensor program to classify its compute pattern (Algorithm 1): `ElementWise`, `Broadcast`, `Injective`, `Reduction`, `OutputEwiseFusible`, or `Opaque`.
  - Step 2: Graph pattern matching and grouping (`FuseOps`, Algorithm 2) uses these labels to form subgraphs (e.g., fuse `ElementWise` into trailing side of a matmul).
  - Step 3: `FuseTensorIR` collapses the tensor programs referenced by the subgraph into a single kernel, preserving symbolic shapes (Figure 9). This even fuses custom kernels (e.g., quantization decode + matmul) that lack high-level operator equivalents.

- Dynamic-shape-aware memory planning (§4.3; Figure 10; Algorithm 3)
  - After lowering `call_tir` and `call_dps_library` to explicit allocations and DPS calls, liveness analysis and a symbolic-aware storage pool decide when existing storage can be reused (Algorithm 3).
  - Symbolic equality proofs determine reuse (e.g., `n*2` equals `2*n`). Users can provide upper bounds for dynamic dims (e.g., max context length), enabling a static plan even with dynamic shapes (§4.3).

- Cross-level workspace lifting (§4.4; Figure 11)
  - If a kernel allocates a large global workspace (e.g., StreamK matmul), a joint transform lifts that allocation to the graph so it participates in global planning and stays static-sized, aiding CUDA Graph capture.

- CUDA Graph offloading for dynamic workloads (§4.5)
  - CUDA Graphs require all accessed global memory to be pre-allocated with fixed sizes, which is normally incompatible with dynamic shapes.
  - With Relax’s static memory planning (including workspaces), eligible subgraphs are captured once and replayed later, bringing graph-capture benefits to dynamic models (§4.5).

- Operator optimization via partial lowering (§4.6; Figure 12)
  - Pattern-match subgraphs (e.g., “matmul + epilogue”) and rewrite them to vendor library calls; other ops use compiler-optimized TensorIR. Analysis-based scheduling rules and optional Ansor-style autotuning cover remaining kernels (§4.6).

- Full pipeline (§4.7; Figure 13)
  - Order: partial library lowering → operator-to-TensorIR lowering → fusion → workspace lifting → memory planning → CUDA Graph offloading → build to runnable module.
  - Runtime shape evaluation: the compiler inserts code to evaluate symbolic expressions from input shapes and store them in a small “shape tensor,” then erases annotations and executes VM instructions plus generated GPU code (§4.7).

Analogy: Think of Relax as a “unified notebook” where graph-level formulas, low-level loops, and vendor calls live together, all annotated with symbolic equations about shapes. Those equations enable the compiler to rearrange and combine code across pages, pre-book memory ahead of time, and even “pre-record” GPU execution (CUDA Graphs)—despite dynamic sizes.

## 4. Key Insights and Innovations
- First-class symbolic shapes across the whole program (fundamental)
  - What’s new: Shapes are symbolic expressions, not “unknowns.” Relations like `len(flatten(x)) = 4*n` are tracked through operators, subgraphs, and foreign calls (Figures 3, 7–8).
  - Why it matters: Enables dynamic-shape-aware memory planning, shape-equality proofs, and cross-level optimizations that require static-like reasoning (Figure 10; Algorithm 3). Also allows runtime-verified assertions (`match_cast`) when deduction is impossible.

- Cross-level abstraction unifying graphs, kernels, and libraries (fundamental)
  - What’s new: The IR natively expresses calls across levels (`call_tir`, `call_dps_library`) with DPS semantics and shape annotations (Figures 4–5).
  - Why it matters: Makes “partial lowering,” “analysis feedback,” and “cross-level transforms” first-class (Figure 6). This is what allows fusion of custom decode kernels with matmul (Figure 9) and lifting workspaces (Figure 11).

- Dynamic-shape-aware operator fusion driven by kernel analysis (novel and practical)
  - What’s new: Pattern inference happens at the tensor program level (Algorithm 1), not by hard-coding properties per high-level op. Fusion then uses symbolic-shape-preserving subgraph creation and kernel merging (Algorithms 2; Figure 9).
  - Why it matters: Reduces engineering effort (no per-op annotations), enables fusing custom ops, and keeps shape relations intact for later passes.

- Static memory planning and CUDA Graph offloading for dynamic models (novel combination)
  - What’s new: With symbolic shape reasoning and upper bounds, Relax generates a static plan even for dynamic workloads, enabling CUDA Graph capture and replay (§4.3–4.5).
  - Why it matters: Historically, CUDA Graphs were reserved for static shapes. Relax extends their benefits to dynamic models, reducing kernel launch overhead (§4.5).

- Workspace lifting across levels (useful cross-level transform)
  - What’s new: Detects kernel-level global allocations and lifts them to graph level (Figure 11).
  - Why it matters: Improves memory reuse and enables static planning/CUDA Graphs for kernels needing large intermediates (e.g., StreamK matmul).

## 5. Experimental Analysis
- Setup and baselines (§5)
  - Models: Llama3-8B, Gemma1.1-7B, Qwen2-7B (float16) for LLM decoding; Whisper-large-v3 for ASR; LLaVA for multimodal generation.
  - Hardware: NVIDIA RTX 4090, AMD Radeon 7900 XTX, Apple M2 Ultra; additional emerging platforms: iPhone 14 Pro (Metal), Samsung S23 (OpenCL), Orange Pi 5 (OpenCL), Steam Deck (Vulkan), Jetson Orin (CUDA), WebGPU on M3 Max (§5.1, §5.3).
  - Baselines: HuggingFace Transformers eager and `torch.compile`, vLLM, and hand-optimized `llama.cpp`. FlashAttention enabled where available (§5.1).
  - Metric: Decode token latency (ms/token) for LLMs (generate 32 tokens; compile once for arbitrary batch/sequence sizes), throughput (tokens/s) on edge devices, transcription/generation time for Whisper/LLaVA (§5.1, §5.3, §5.4).
  
- Main results
  - Competitive or superior LLM decode latency across GPUs
    - On RTX 4090 (Figure 14): 
      > “Relax brings competitive performance across different models and batch sizes, and reduces the decode token latency by up to 27%.”
    - On AMD 7900 XTX (Figure 15):
      > “Relax … brings optimized performance with up to 1.50× under case of batch size 1.”
    - On Apple M2 Ultra (Figure 16):
      > “Relax has competitive performance comparing to the hand-optimized llama.cpp baseline.”
  - Memory planning effectiveness (Table 2; §5.2)
    - Prefill activation memory reduced from 192.7 MiB to 149.7 MiB (−22%).
    - Decode activation memory reduced from 150.0 MiB to 88.2 MiB (−40%).
  - Ablation: contributions of individual techniques (Figure 17; §5.2)
    - Partial library lowering: up to 27% latency reduction at larger batches.
    - Operator fusion: fuses ~20% of ops, reducing global memory traffic and launch count.
    - CUDA Graph offloading: an extra ~1–2% gain by lowering GPU driver overhead.
  - Emerging platforms (Table 3; §5.3): throughput (tokens/s), 4-bit (mobile footnotes vary)
    - iPhone 14 Pro (Metal): Llama 5.1†, Phi-3 13.8, RedPajama-3B 19.5
    - Samsung S23 (OpenCL): Llama 7.9†, Phi-3 13.1, RedPajama-3B 20.5
    - Orange Pi 5 (OpenCL): Llama 2.3, Phi-3 5.0, RedPajama-3B 6.1
    - Steam Deck (Vulkan): Llama 14.0, Phi-3 20.2, RedPajama-3B 22.9
    - Jetson Orin (CUDA): Llama 32.0, Phi-3 59.1, RedPajama-3B 65.2
    - WebGPU (M3 Max): Llama 37.8, Phi-3 68.0, RedPajama-3B 68.6
    - Footnote: mobile Llama results use Llama2-7B (3-bit on iPhone and 4-bit on S23) due to VRAM limits.
    - A direct Android comparison (Figure 18):
      > “Relax delivers up to 55% more throughput” than `llama.cpp` on Samsung S23.
  - Other models (§5.4)
    - Whisper-large-v3 (Figure 19):
      > RTX 4090: 14% faster transcription than HF Transformers; competitive on Apple M2 Ultra where some baselines lack Apple GPU support.
    - LLaVA (Figure 20):
      > Competitive optimized generation time on NVIDIA and Apple platforms.

- Do the experiments support the claims?
  - Yes, for three reasons:
    - Performance: Cross-platform speedups or parity with strong baselines across vendor GPUs and device classes (Figures 14–16, 18–20).
    - Memory: Concrete reductions that enable running on constrained devices and enable CUDA Graph usage (Table 2, §4.5).
    - Breadth: Single compiler pipeline targets servers, mobile, embedded, and browsers (Table 3), showing the practical value of cross-level AOT compilation for dynamic models.

- Notable ablations and robustness
  - The ablation (Figure 17) isolates the effect of fusion, library calls, and CUDA Graphs, clarifying where gains come from and how they compose.
  - The dynamic-shape memory plan is robust to varying sequence lengths/batch sizes across time, avoiding allocator churn (§5.2).

## 6. Limitations and Trade-offs
- Shape reasoning is forward and local by default (§4.1)
  - This keeps compilation fast but may miss deductions that require global constraint solving. Relax uses `match_cast` + runtime checks as a fallback for data-dependent ops (e.g., `unique`, Figure 3).
- Pattern-dependent optimizations (§4.2, §4.6)
  - Fusion and partial library lowering rely on pattern detection. New library integrations still require pattern registration, and unusual operator combinations may need additional engineering.
- CUDA Graph applicability (§4.5)
  - Only subgraphs that meet capture constraints (fixed-size pre-allocated memory, stable execution ordering) are offloaded. Gains are modest (~1–2%) when kernel launch overhead is a small fraction of total time (Figure 17).
- Scope: inference-centric pipeline
  - The paper targets end-to-end inference. Training workloads (with complex control flow, gradient accumulation, and optimizer state) are not evaluated.
- Runtime assertions for `match_cast`
  - While described as lightweight (§3.2), shape assertions still add potential runtime checks when data-dependent shapes are present, and incorrect assumptions raise errors.
- Upper bounds for static planning
  - For best memory predictability, users may provide upper bounds (e.g., max context length). Overly conservative bounds can waste memory; overly tight bounds can fail for larger requests (§4.3).

## 7. Implications and Future Directions
- What changes in the field
  - Cross-level IR + symbolic shapes demonstrates that dynamic-shape models can obtain “static-like” benefits (static planning, CUDA Graph capture) without sacrificing portability. This encourages ML compilers to blur boundaries between graph, kernels, and libraries (Figure 6) and to treat shapes as first-class program entities.
  - Practical deployment unification: the same AOT pipeline reaches servers, phones, embedded devices, and browsers (Table 3), lowering the barrier to edge AI and WebGPU inference.

- Follow-up research enabled/suggested
  - More powerful shape reasoning: augment forward deduction with constraint solvers to reduce reliance on `match_cast` and widen fusion opportunities (§4.1).
  - Automated cross-level scheduling: jointly optimize tiling, fusion, and memory plans with symbolic constraints, perhaps with learning-based search across levels.
  - Robust dynamic-workload graph capture beyond CUDA: apply similar ideas to other static execution graph backends (Vulkan, Metal) as hinted in §4.5.
  - Training support: extend cross-level abstractions and symbolic shapes to backpropagation, mixed-precision training, and optimizer states.
  - Safety and verification: formalize the semantics of symbolic shapes, `match_cast`, and DPS calls to make guarantees about correctness and memory safety.

- Practical applications
  - Turnkey deployment of LLMs and multimodal models to mobile, embedded, and browser environments with predictable memory and competitive performance (Table 3; Figures 18–20).
  - Faster iteration on custom operators (e.g., quantization decode) thanks to kernel analysis feedback and cross-level fusion (Figure 9).
  - Operational efficiency: fewer backend-specific hand-written kernels by combining partial library lowering with compiler-generated kernels (§4.6).

In short, Relax contributes a unified way to express and optimize dynamic ML programs end-to-end. The combination of cross-level IR and first-class symbolic shapes is the key architectural idea that unlocks static planning and cross-layer optimizations for workloads that traditionally forced a trade-off between dynamism and deployability.
