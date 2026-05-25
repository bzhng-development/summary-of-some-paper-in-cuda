# KernelBench: Can LLMs Write Efficient GPU Kernels?

**ArXiv:** [2502.10517](https://arxiv.org/abs/2502.10517)

## 🎯 Pitch

KernelBench introduces the first open-source benchmark expressly designed to evaluate if large language models (LLMs) can automatically generate not just correct, but *efficient* GPU kernels for real-world machine learning workloads. By offering a 250-task suite, an automated evaluation environment, and a unified metric that combines correctness and speedup, KernelBench allows direct, practical measurement of AI-driven code optimization. This matters because showing that LLMs can produce high-performance GPU code could revolutionize AI system development—lowering costs, saving energy, and accelerating innovation by democratizing the creation of fast, hardware-specific kernels that power today’s ML architectures.

---

## 1. Executive Summary (2-3 sentences)
KernelBench is a new, open-source benchmark and evaluation environment that measures whether large language models (LLMs) can write correct and fast GPU kernels—small GPU programs that implement ML operations. It contributes a realistic 250-task suite, an automatic evaluation pipeline, and a new metric, `fastp`, to judge both correctness and speedup. Using KernelBench, the study finds that state-of-the-art reasoning models still outperform PyTorch’s eager execution in fewer than 20% of tasks out of the box (Table 1), but can improve substantially when given execution/profiling feedback (Table 2, Figure 6).

## 2. Context and Motivation
- Problem/gap:
  - Efficient GPU kernels are essential for the speed and cost of ML systems, but writing them is slow and requires specialized expertise. Porting kernels to new hardware can take years (e.g., the evolution from FlashAttention to FlashAttention-3) and often demands algorithmic redesigns (Section 1).
  - Existing LLM code benchmarks largely test correctness, not runtime performance on real workloads (Section 2).
- Importance:
  - Real-world: Faster kernels directly lower ML training/inference cost and energy (Abstract; Section 6.2 Ethics stresses energy savings).
  - Scientific: Tests whether LLMs can reason about hardware, memory hierarchies, and performance—skills beyond syntax-level code generation (Sections 1–2).
- Prior approaches and limitations (Section 2):
  - Hand-engineered libraries (cuDNN, CUTLASS) achieve high performance but are hardware-specific and labor-intensive.
  - Higher-level DSLs/libraries (Triton, ThunderKittens) lower the barrier but still require human programming.
  - Compiler tools (e.g., `torch.compile`, FlexAttention) perform limited automatic graph-level optimizations; they rarely invent new kernels or algorithms.
  - Prior LLM studies focused on translating code or generating small, known kernels (e.g., GEMM), not end-to-end performance on diverse ML workloads.
- Positioning:
  - KernelBench evaluates LLMs in a realistic engineering loop: choose what to optimize, generate code in any language (CUDA, Triton, CUTLASS, PTX, etc.), compile, profile, and refine with feedback (Sections 3, 5.1).
  - The benchmark is evaluation-only and hardware-agnostic so that progress transfers to new GPUs (Sections 3.3, 6.2).

## 3. Technical Approach
Step-by-step overview of the KernelBench framework (Section 3; Appendix A–C, H):

- Task format (Section 3.1; Appendix A):
  - Input: a reference PyTorch `Model` with `__init__` and `forward`, plus `get_inputs()` and `get_init_inputs()` that fix tensor shapes/dtypes for evaluation.
  - Output: an optimized `ModelNew` that can call custom kernels in the `forward`. The LM chooses which ops to replace/fuse, which language to use, and how to optimize (fusion, tiling, using Tensor Cores, etc.).
  - Example integration uses PyTorch’s inline CUDA extensions (Appendix A).

- Task suite (Section 3.2):
  - 250 tasks, three levels by complexity:
    - Level 1 (100 tasks): Single primitive ops (e.g., matmul, convolutions, activations, norms, losses). Beating PyTorch is hard because it already calls expert kernels.
    - Level 2 (100 tasks): Sequences of 3–6 ops—encourages fusion (e.g., matmul + bias + ReLU).
    - Level 3 (50 tasks): End-to-end architectures from popular repos (e.g., AlexNet, MiniGPT), where compiler rules may be insufficient and algorithmic changes help.

- Evaluation methodology (Section 3.3; Appendix B):
  - Correctness:
    - Run both `Model` and `ModelNew` on 5 random inputs (fixed shapes) and check that outputs match in shape and values (Section 3.3; Appendix B.2 explains the choice of 5).
  - Performance:
    - Measure wall-clock runtime of `forward` using CUDA events; warmup 3 runs; average over 100 trials (Appendix B.1). Coefficient of variation < 3%.
    - Speedup = time(Model) / time(ModelNew).
  - Metric `fastp`:
    - Fraction of tasks that are both correct and achieve speedup > `p` (Section 3.3).
    - `fast0` == correctness rate; `fast1` is the fraction of tasks strictly faster than baseline PyTorch Eager.
    - By increasing `p`, you raise the bar for “how much faster” (Figure 3).

- Baselines and hardware (Sections 4, B.4, G):
  - Primary baseline is PyTorch Eager (direct execution of highly optimized kernels).
  - `torch.compile` is reported but not the main focus; it sometimes has runtime overhead on small Level 1 ops (Table 1; Appendix B.4).
  - Main results profiled on NVIDIA L40S; generalization checked on other GPUs (A100, H100, L4, T4, A10G) (Section 4.4; Table 13–15; Figures 8–9).

- Test-time methods to improve LLM generations (Section 5.1):
  - Repeated sampling (Section 5.1.1; Figure 4): Generate k samples with temperature; score the best (`fastp@k`). Increases the chance to find a working/fast variant.
  - Iterative refinement (Section 5.1.2; Figure 5–6; Table 2): Multi-turn loop. At each turn, feed back:
    - `G`: previous generation,
    - `E`: compiler/execution feedback (e.g., NVCC errors, timeouts, correctness verdict),
    - `P`: PyTorch profiler info (operator breakdown and timings).
    - Measure best-so-far quality after N turns (`fastp@N`).
  - Practical orchestration supports compiling on CPU, running on GPU, and parallelizing many experiments (Appendix H).

- Conditioning on knowledge (Section 5.2):
  - Few-shot hardware-aware exemplars (Section 5.2.1; Appendix F): Provide examples that demonstrate fusion, tiling, and shared-memory I/O patterns (e.g., minimal FlashAttention).
  - Hardware specification prompts (Section 5.2.2; Appendix G): Provide GPU specs (TFLOPS, bandwidth, shared memory sizes) and definitions of GPU concepts (warps, thread blocks) to encourage hardware-specific code.
  - Definitions (selective):
    - `kernel`: a function run massively in parallel on the GPU; each invocation is a thread.
    - `fusion`: combining multiple ops into one kernel to reduce slow global memory traffic.
    - `tiling`: partitioning data into blocks that fit faster memories (registers/shared memory) to improve locality.
    - `shared memory`: fast on-chip memory shared by threads in a block.
    - `wmma` (warp matrix multiply-accumulate): Tensor Core instructions that multiply tiles at warp granularity.
    - `PyTorch Eager`: default PyTorch execution mode; calls optimized kernels directly.
    - `torch.compile`: a PyTorch feature that captures and optimizes graphs (e.g., fuses ops) before running.

## 4. Key Insights and Innovations
- A realistic, end-to-end evaluation environment (Sections 3, H):
  - Novelty: The benchmark gives LLMs full agency—pick targets, pick language, generate runnable code, and iterate with compiler/profiler signals. Prior LLM coding benchmarks rarely integrate compilation, execution, and performance feedback at this scale.
  - Significance: Success on KernelBench maps directly to faster ML systems (Abstract; Section 6.2).

- The `fastp` metric (Section 3.3):
  - Novelty: Jointly measures correctness and an adjustable performance threshold, unifying “works” and “is fast enough.”
  - Significance: Captures speedup distribution (Figure 3), supports raising performance requirements over time, and allows `p < 1` during training.

- Evidence on current LLM capabilities and failure modes (Section 4):
  - Insight: Frontier reasoning models generate fewer compile/runtime failures (Figure 2) but still struggle with functional correctness and speed—fewer than 20% of tasks faster than PyTorch Eager in one-shot (Table 1).
  - Significance: Highlights that performance-aware code generation requires more than syntax fluency; it needs algorithmic and hardware reasoning.

- Feedback-driven test-time improvement (Section 5.1):
  - Novelty: Systematically compares repeated sampling versus multi-turn refinement with compiler/profiler feedback.
  - Significance: Iterative refinement with execution and profiling feedback can more than double `fast1` in some settings—e.g., DeepSeek-R1 on Level 2 rises from 36% to 72% within 10 turns (Figure 6; Table 2).

## 5. Experimental Analysis
- Setup and metrics:
  - Hardware: NVIDIA L40S for main results; cross-GPU checks on H100, A100, L4, T4, A10G (Table 13; Section 4.4).
  - Metric: `fast1` (faster than PyTorch Eager and functionally correct); `fast0` for correctness only (Section 3.3).
  - Timing: 3 warmups + 100 measured trials; mean used (Appendix B.1). Correctness on 5 random inputs (Appendix B.2).

- One-shot baseline (Section 4.1):
  - Main quantitative finding (Table 1):
    > “LM-generated kernels achieve a speedup over PyTorch Eager in fewer than 20% of tasks.”
    - Example `fast1` over PyTorch Eager:
      - `OpenAI o1`: Level 1 10%, Level 2 24%, Level 3 12%.
      - `DeepSeek R1`: Level 1 12%, Level 2 36%, Level 3 2%.
      - `GPT-4o`: Level 1 4%, Level 2 5%, Level 3 0%.
      - `Llama 3.1-70B`: Level 1 3%, Level 2 0%, Level 3 0%.
  - Speedup distribution: Most correct kernels are still slower than PyTorch; fewer than ~15% exceed 1× across levels (Figure 3).
  - Failure modes (Figure 2):
    - Reasoning models (o1, R1) have fewer compile/runtime failures but similar rates of functional incorrectness compared to others.

- Cross-hardware generalization (Section 4.4; Table 14; Figures 8–9):
  - Level 1 `fast1` is relatively stable across GPUs; Level 2 varies. For example:
    > DeepSeek-R1 Level 2 `fast1` is 36% on L40S vs 47% on A10G (Table 14).
  - Interpretation: One-shot generated kernels may be sensitive to device characteristics; portability of performance is limited.

- Test-time methods (Section 5.1):
  - Repeated sampling (Figure 4):
    - DeepSeek-V3 Level 2: `fast1` rises from 4% (one-shot) to 37% at k=100.
    - Gains saturate when a model’s base success probability is very low (e.g., 34 convolution variants remained unsolved even with k=100).
  - Iterative refinement (Table 2; Figure 6):
    - With a fixed budget of 10 model calls, iterative refinement generally beats repeated sampling in 5/6 settings (Table 2).
    - Strongest case: DeepSeek-R1 Level 2 `fast1` jumps from 36% (single) → 62% with execution feedback `G+E` → 72% with `G+E+P` (Table 2; Figure 6).
    - Correctness (fast0) also increases dramatically: DeepSeek-R1 reaches 95% (Level 1) and 92% (Level 2) correct by turn 10 with `G+E+P` (Table 9).
  - Qualitative turn-by-turn trajectories (Appendix D.4):
    - Example: 2D conv (Level 1, Problem 63) improves runtime from 9.1 ms to 1.43 ms over 10 turns but still trails PyTorch Eager at 0.47 ms (Table 5).
    - Some tasks never become correct despite 10 turns (Table 8).

- Conditioning on knowledge (Section 5.2):
  - Few-shot best-practice examples (Section 5.2.1; Appendix F):
    - Overall `fast1` may drop due to more aggressive, error-prone code (Table 10), but within-task improvements appear.
    - For GEMM-like tasks, 77% of Level 1 variants are faster than the paper’s one-shot baseline due to tiling, though still slower than PyTorch Eager (Table 11).
    - A subset of Level 2 tasks surpass PyTorch Eager via shared-memory I/O management (Table 12).
  - Hardware-spec prompts (Section 5.2.2; Table 15):
    - Minimal overall impact for Llama 3.1 and DeepSeek-V3; mixed improvements for o1/R1.
    - Encourages hardware-specific attempts: R1 often tries `wmma` on H100 (example kernel in Appendix G, Figure 10), but many fail to compile. A few outliers achieve ≥2× speedups vs the one-shot baseline.

- Notable generated kernels (Section 6.1; Appendix D):
  - Algorithmic simplification: Multiply diagonal matrix without forming it—treat it as row-wise scaling—achieves 13× speedup (D.1).
  - Fusion:
    - Fused GeLU: 2.9× (D.2).
    - Softsign: 1.3× (D.2).
    - A small chain (matmul + divide + sum + scale): 2.6× (D.2).
    - Partial fusion in attention: 1.9× (D.2).
  - Memory hierarchy:
    - Cosine similarity using shared memory reduces global traffic: 2.8× (D.3).
    - Triplet margin loss with shared-memory reductions: 1.9× (D.3).

- Do the experiments support the claims?
  - Yes for difficulty and headroom: One-shot `fast1` is low (Table 1; Figure 3) and failure analysis shows functional errors are pervasive (Figure 2).
  - Yes for test-time feedback value: Both repeated sampling and iterative refinement increase `fast1`, with larger and more consistent gains from iterative refinement that ingests execution/profiler feedback (Table 2; Figure 6; Table 9).
  - Mixed for hardware conditioning: Hardware specs in prompts help only a little overall, though they induce some hardware-specific code (Section 5.2.2; Table 15; Figure 10).

## 6. Limitations and Trade-offs
- Functional correctness remains the bottleneck:
  - Even reasoning models that reduce compile/runtime failures still often produce wrong outputs (Figure 2). Correctness feedback is less granular than compiler errors, making it harder to fix (Section 5.1.2; Table 9 discussion).
- Data scarcity and complexity:
  - CUDA is a low-resource language in common open-source corpora (~0.073% of The Stack v1.2; Section 4.2), which likely limits LLM proficiency, especially for niche instructions (`wmma`).
- Evaluation choices:
  - Correctness checked on 5 random inputs (Appendix B.2). This is pragmatic but not formal verification; subtle bugs may slip through.
  - Compile time is excluded; only runtime is measured (Appendix B.4). This favors heavy specialization that may incur large compile costs in practice.
  - The primary baseline is PyTorch Eager; `torch.compile` can be faster on Levels 2–3 but has runtime overhead on small ops (Appendix B.4; Tables 1 and 4).
- Generalization across hardware:
  - Performance portability is limited; Level 2 results vary across GPUs (Section 4.4; Table 14; Figures 8–9).
- Scope and coverage:
  - Focused on NVIDIA GPUs; other accelerators (TPUs, IPUs) are future work (Section 6.3).
  - The benchmark is evaluation-only; no training data is provided (Section 3.3), which is appropriate for evaluation but does not directly guide model training.

## 7. Implications and Future Directions
- How this changes the landscape:
  - It reframes LLM code-gen evaluation around real performance, not just syntactic correctness. The `fastp` metric and end-to-end environment create a target for performance-aligned LLMs and agents.
- Research opportunities:
  - Data and training:
    - Curate/open-source high-quality CUDA/Triton/CUTLASS/PTX corpora; emphasize hardware features and correctness (Section 6.3).
    - Train or fine-tune models with performance signals—e.g., RL on `fastp`, curriculum on `p` thresholds.
  - Agentic workflows:
    - Strengthen multi-turn tool use: integrate compiler logs, profilers, and autotuners; design search strategies that combine repeated sampling with refinement (Sections 5.1, H).
  - Abstractions:
    - Target higher-level DSLs (Triton, ThunderKittens, CUTLASS) to improve compile success and ease Tensor Core use (Section 6.3).
  - Verification and robustness:
    - Combine random testing with static analysis or bounded verification to catch subtle errors (Appendix B.2).
  - Hardware awareness:
    - Learn device-specific strategies (e.g., asynchronous copy, Tensor Core tiling) and portability heuristics; explore cross-device transfer and conditional codegen (Section 5.2.2).
- Practical applications:
  - Speed up bespoke ML operators, fuse common op chains in production models, and prototype kernels for new architectures faster than manual efforts.
  - Reduce compute cost and energy at scale by harvesting even modest speedups across widely used workloads (Ethics; Section 6.2).

Overall, KernelBench provides a rigorous, scalable way to measure and drive progress on performance-aware code generation. The current frontier models show promise—in particular when guided by execution/profiling feedback—but there is substantial room to improve correctness, hardware specialization, and the ability to exploit modern GPU features.
