# KernelBench: Can LLMs Write Efficient GPU Kernels?

**ArXiv:** [2502.10517](https://arxiv.org/abs/2502.10517)

## 🎯 Pitch

KernelBench introduces the first comprehensive framework and benchmark to rigorously evaluate whether large language models can generate both correct and high-performance GPU kernels for real-world machine learning workloads. By testing LMs across 250 diverse PyTorch tasks and introducing a new 'fastp' metric measuring both correctness and speedup, KernelBench reveals that current models struggle to match expert-tuned baselines out-of-the-box, but can improve dramatically through iterative feedback. This matters because efficient GPU kernels underpin scalable, cost-effective, and energy-efficient AI systems, and automating this expertise could profoundly accelerate innovation and deployment across the ML ecosystem.

---

## 1. Executive Summary
KernelBench introduces an evaluation framework and metric to test whether language models (LMs) can write GPU kernels that are both correct and fast for real machine‑learning workloads. Across 250 PyTorch tasks, frontier “reasoning” LMs can occasionally match or beat expert baselines, but out‑of‑the‑box success is rare; iterative refinement using compiler/runtime feedback substantially improves outcomes (e.g., `fast1` up to 72% on operator‑sequence tasks), yet the benchmark remains challenging.

## 2. Context and Motivation
- Problem and gap
  - Writing efficient `GPU kernels`—small programs executed by many parallel GPU threads—is hard and time‑consuming. It demands deep knowledge of hardware details (e.g., thread blocks, shared memory, specialized instructions) and performance techniques (e.g., tiling, operator fusion). Section 1 highlights a real case: FlashAttention kernels took years to arrive and to be ported to new hardware.
  - Existing LM code benchmarks largely emphasize functional correctness, not wall‑clock speed on real workloads. There is no standardized way to test if LMs can act like kernel engineers who iterate with compiler and profiler feedback.
- Why it matters
  - Performance kernels impact cost, energy, and feasibility of ML systems. Faster kernels cut training/inference time and energy consumption at scale (Section 1).
- Prior approaches and limitations
  - Hand‑engineered libraries (cuDNN, CUTLASS; Section 2) are fast but hardware‑specific and labor‑intensive.
  - Higher‑level kernel DSLs/libraries (Triton, ThunderKittens; Section 2) reduce effort but still require human coding.
  - Compilers (e.g., `torch.compile`) offer limited, rule‑based optimizations, mainly fusion (Section 2; Appendix B.4).
  - LM code generation on HPC tasks has focused on translation or a small set of classic kernels (e.g., GEMM) rather than diverse, modern ML workloads (Section 2).
- Positioning
  - KernelBench (Sections 3–4) supplies a realistic, automated environment that:
    - Feeds LMs real PyTorch workloads.
    - Lets LMs pick what to optimize and how (CUDA/PTX/Triton/CUTLASS etc.).
    - Provides execution, compiler, and profiler feedback for iterative improvement.
    - Evaluates both correctness and speed via a new metric, `fastp`.

## 3. Technical Approach
- What a KernelBench task looks like (Section 3.1; Figure 1; Appendix A)
  - Input: a PyTorch reference module `Model` with `__init__`, `forward`, and helper functions, plus callable generators `get_inputs()` and `get_init_inputs()` that produce tensors with fixed shapes/dtypes (important because optimal kernels are shape- and dtype‑dependent).
  - Output: a new module `ModelNew` that embeds custom kernels (commonly via PyTorch’s inline CUDA extension) and can use any optimization strategy (fusion, tiling, recompute) and any language level (CUDA C++, `PTX` assembly, Triton, CUTLASS, ThunderKittens).
  - The LM decides which operations to replace and how (Section 3.1).
- Task coverage (Section 3.2)
  - 250 tasks across three “levels,” increasing in difficulty and realism:
    - Level 1 (100 tasks): single primitive ops (matmul, conv, activations, norms, losses). Note: PyTorch often calls highly tuned closed‑source kernels, making this a high bar.
    - Level 2 (100 tasks): short sequences of ops (3–6 ops) that invite kernel fusion.
    - Level 3 (50 tasks): end‑to‑end modules from popular repos (e.g., AlexNet, MiniGPT), containing many ops.
- Evaluation protocol (Section 3.3; Appendix B)
  - Correctness: compare outputs of `Model` and `ModelNew` on 5 random inputs per task. Five is a pragmatic balance between catching errors and throughput; an experiment with 100 kernels showed failures were typically 0/5 or 0/100 (Appendix B.2).
  - Performance: measure mean wall‑clock `forward` time over 100 trials with warm‑ups, using CUDA events; coefficient of variation < 3% (Appendix B.1).
  - Metric `fastp`: the fraction of tasks where the generated kernel is both correct and achieves speedup > `p` over the baseline (PyTorch Eager is the primary baseline). `fast0` equals correctness rate; `fast1` is “correct and faster than baseline” (Section 3.3).
- Baseline prompting and decoding (Section 4.1; Appendix C.1)
  - One‑shot: a single in‑context example (adding two tensors) to show formatting/inlining; greedy decoding.
- Test‑time methods that mimic engineer workflows (Section 5.1; Figure 5)
  - Repeated sampling (Section 5.1.1): draw `k` diverse samples (higher temperature), select the best; report `fastp@k`.
  - Iterative refinement (Section 5.1.2): multi‑turn process. Each turn gives the LM its last code `G`, compiler/execution feedback `E` (including errors/timeouts) and optionally a PyTorch profiler summary `P`. Report `fastp@N` (best by turn `N`).
- Conditioning on optimization knowledge (Section 5.2)
  - Few‑shot examples that demonstrate best practices (fusion for GeLU, tiling for GEMM, a minimal FlashAttention showcasing shared‑memory I/O).
  - Hardware‑aware prompts: pass GPU specs (e.g., SM registers, TFLOPS, bandwidth) and concise GPU concepts. Intended to test whether LMs adapt to hardware properties (Appendix G.2).
- Cross‑hardware evaluation
  - Main results on NVIDIA L40S; also run generated kernels on H100, A100, L4, T4, A10G to study generalization (Section 4.4; Appendix G.1, Table 13, Table 14, Figures 8–9).
- System engineering for throughput (Appendix H)
  - Pipeline: large‑scale parallel LM inference → CPU pre‑compile with `nvcc` (caching) → isolated GPU evaluation; for iterative runs, a GPU orchestration FSM with per‑turn compilation/execution.

Definitions for uncommon terms used above:
- `Operator fusion`: merging multiple tensor ops into a single kernel to avoid extra global‑memory reads/writes.
- `Tiling`: splitting computation into blocks (“tiles”) to improve cache/shared‑memory reuse.
- `PTX`: NVIDIA’s low‑level GPU assembly language.
- `Tensor Cores` / `wmma`: specialized hardware units and APIs for fast matrix multiplications at low precision.
- `torch.compile`: a PyTorch facility that captures and optimizes graphs (e.g., with fusion rules); compile‑time excluded from runtime comparisons (Appendix B.4).

## 4. Key Insights and Innovations
- A benchmark and environment that mirror real kernel engineering (Sections 3–4; Figure 1)
  - Novelty: Unlike prior code gen tasks or single‑kernel suites, KernelBench supplies 250 modern ML workloads with fixed shapes/dtypes, lets LMs choose what to optimize, and integrates compile/runtime/profiler feedback. Success directly translates to practical speedups.
  - Significance: It bridges LM code generation with performance engineering rather than stopping at functional correctness.
- The `fastp` metric (Section 3.3)
  - Novelty: Jointly measures functional correctness and a configurable speedup threshold over strong baselines. `fast0` isolates correctness; `fast1` demands both correct and faster.
  - Significance: Prevents inflated scores from trivially correct but slow code and allows raising the bar by increasing `p` as methods improve (Figure 3).
- Evidence that feedback loops matter (Section 5.1; Table 2; Figure 6; Table 9)
  - Novelty: Systematic test‑time use of compiler/runtime/profiler feedback to iteratively refine kernels.
  - Significance: Converts many compile/runtime failures into correct code and surfaces faster implementations; e.g., `DeepSeek‑R1` Level 2 improves from `fast1=36%` (single attempt) to `72%` after 10 feedback turns with execution+profiler signals.
- First wide‑angle picture of LM kernel‑writing capabilities and failure modes (Sections 4–6)
  - Insight: Even strong LMs often fail correctness checks; reasoning models reduce execution failures but still struggle with functional equivalence (Figure 2). Most correct kernels are slower than PyTorch (Figure 3; Figure 7).
  - Significance: Sets a clear research agenda—data scarcity for CUDA (0.073% in The Stack; Section 4.2), hardware adaptation, discovering and validating performance strategies (fusion, tiling, shared memory, tensor‑core use).

## 5. Experimental Analysis
- Evaluation setup (Appendix B; Section 4.1)
  - Hardware: Main results on NVIDIA L40S (48GB, Ada; Appendix B). Cross‑hardware on H100, A100, L4, T4, A10G (Appendix G.1, Table 13).
  - Measurement: 3 warm‑ups + 100 runs; mean time; CV < 3% (Appendix B.1). Correctness on 5 random inputs per task (Appendix B.2).
  - Baselines: PyTorch Eager (primary). `torch.compile` variations reported separately; runtime overhead can make Level‑1 Eager faster for small ops (Appendix B.4, Table 4).
- One‑shot headline numbers (Table 1; Figure 3; Figure 2; Figure 7)
  - Out‑of‑the‑box `fast1` (correct and faster than PyTorch Eager) is low:
    > Table 1 (L40S, `fast1` over Eager): `OpenAI‑o1` = 10% (L1), 24% (L2), 12% (L3); `DeepSeek‑R1` = 12%, 36%, 2%; `GPT‑4o` = 4%, 5%, 0%; `Claude‑3.5 Sonnet` = 10%, 7%, 2%; `Llama‑3.1 70B` = 3%, 0%, 0%.
  - Failure modes:
    > Figure 2: reasoning models (o1, R1) reduce execution failures but functional‑correctness failures remain high across all models.
  - Speed profile:
    > Figure 3: at `p=1`, fewer than 15% of correct kernels are faster than Eager across levels; increasing `p` rapidly decreases `fastp`.  
    > Figure 7: among correct kernels, the median speedup is below 1 for Levels 1 and 3 and only slightly above 1 for Level 2.
- Cross‑hardware generalization (Section 4.4; Appendix G.1)
  - Generated kernels’ speedups vary with GPU:
    > Table 14: `DeepSeek‑R1` Level‑2 `fast1` shifts from 36% (L40S) to 47% (A10G) and 42% (H100).  
    - Interpretation: kernels are often tuned implicitly to the device they were first run on; portability is weak without explicit hardware conditioning.
- Test‑time scaling and feedback (Section 5.1; Figure 4; Table 2; Figure 6; Table 9)
  - Repeated sampling helps but has diminishing returns:
    > Figure 4: On Level 2, `DeepSeek‑V3` rises to `fast1=37%` at `k=100` from 4% one‑shot; some tasks (e.g., 34 conv variants) never yield a correct sample even at `k=100`.
  - Iterative refinement is especially effective with execution/profiler feedback:
    > Table 2 (10‑call budget): `DeepSeek‑R1` Level‑2 `fast1` improves from 36% (single attempt) → 62% (G+E) → 72% (G+E+P).  
    > Figure 6: the `fast1@N` curve for `DeepSeek‑R1` Level‑2 keeps rising through 10 turns.  
    > Table 9: correctness (`fast0`) for `DeepSeek‑R1` reaches 95%+ on Levels 1–2 after 10 turns with execution feedback; remaining mistakes are functional mismatches rather than crashes.
- Conditioning on optimization knowledge (Section 5.2; Appendix F, G.2)
  - Few‑shot optimization exemplars:
    > Appendix F, Table 10: few‑shot prompts often lengthen generations and increase execution failures, reducing overall `fast1`; however, within correct solutions, LMs more frequently apply tiling/fusion.  
    > Appendix F, Table 11: for 77% of Level‑1 GEMM variants, tiling beats the one‑shot LM baseline (but usually remains slower than PyTorch Eager due to lack of tensor‑core use).  
    > Appendix F, Table 12: level‑2 kernels with aggressive shared‑memory I/O can outperform Eager on several tasks (e.g., “Conv2d InstanceNorm Divide” 0.082 ms vs 0.090 ms).
  - Hardware‑aware prompts:
    > Table 15: passing GPU specs yields small net changes overall; some `o1`/`R1` generations attempt `wmma` (tensor‑core) code (Appendix G.2, Figure 10), but many fail to compile; a few outliers achieve ≥2× speedups.
- Case studies of “what worked” (Section 6.1; Appendix D)
  - Operator fusion:
    > GELU fused into one kernel: 2.9× (Appendix D.2); a fused matmul/div/sum/scale: 2.6×; a fused masking+scale+ReLU in attention: 1.9×.
  - Memory hierarchy use:
    > Cosine similarity with shared‑memory reductions: 2.8×; Triplet margin loss with shared memory: 2.0× (Appendix D.3).
  - Algorithmic trick:
    > Diagonal matrix × dense matrix rewritten as row scaling (no explicit diagonal construction): 13× (Appendix D.1).
- Do the experiments support the claims?
  - Yes, in three ways:
    - Breadth and realism: 250 tasks, many without known human kernels, across three levels (Section 3.2).
    - Rigor: automated correctness checks, stable timing, multiple hardware backends (Sections 3.3, 4.4; Appendix B, G).
    - Analysis: breakdown of errors (Figure 2), speed distributions (Figure 3, Figure 7), search vs feedback (Figure 4; Table 2; Figure 6), and concrete kernel examples (Section 6.1; Appendix D).
  - Results are mixed and conditional:
    - Out‑of‑the‑box is weak (<20% `fast1` in Table 1), but feedback and search can lift performance substantially (Table 2).  
    - Hardware knowledge helps sporadically; compiler/profiler feedback is the reliable lever.

## 6. Limitations and Trade-offs
- Assumptions and design choices
  - Correctness via 5 random inputs is probabilistic; rare corner cases could pass (Appendix B.2). This is common in kernel testing but not a formal proof of equivalence.
  - Speed comparisons exclude compile time; `torch.compile` runtime overhead can penalize small ops (Appendix B.4). Benchmark focuses on steady‑state runtime.
- Coverage limits
  - Only GPUs are evaluated. Other accelerators (TPUs, specialized ASICs) are out of scope (Section 6.3).
  - Many Level‑1 PyTorch ops are backed by heavily optimized, sometimes closed‑source kernels, setting a high bar that may obscure LM progress unless tensor cores or advanced tiling are used.
- Data and modeling constraints
  - CUDA appears only 0.073% in a popular open‑source code corpus (The Stack v1.2), which likely hinders LM fluency (Section 4.2). This manifests as frequent compiler errors and functional mismatches.
  - Some APIs lacked temperature control (e.g., for `DeepSeek‑R1`), limiting fair repeated‑sampling comparisons (Table 2 note).
- Generalization and robustness
  - Cross‑hardware variability is significant for multi‑op tasks (Section 4.4; Table 14; Figures 8–9).  
  - Few‑shot optimization prompts increase code length and execution failures (Appendix F, Table 10), highlighting a trade‑off between ambition and reliability.
- Open questions
  - How to systematically elicit tensor‑core use and correctness (few successful `wmma` kernels; Section 5.2.2)?  
  - How to choose which feedback to surface per turn (granularity of correctness signals vs rich compiler errors; Table 9 discussion)?

## 7. Implications and Future Directions
- How this changes the landscape
  - KernelBench reframes “code generation” to “performance‑aware system building.” The `fastp` metric and multi‑turn feedback loops make performance a first‑class target, not an afterthought.
  - Because tasks are real PyTorch workloads, progress on the benchmark directly yields deployable speedups (Section 6.2).
- Research avenues it enables (Section 6.3)
  - Data and training: curate/open‑source high‑quality CUDA/Triton/PTX corpora; explore performance‑aligned finetuning and reasoning curricula for hardware‑aware coding.
  - Abstractions and toolchains: have LMs target higher‑level DSLs (Triton, CUTLASS, ThunderKittens) to reduce surface area for errors and to tap tensor‑core kernels more reliably.
  - Agentic workflows: combine search (sampling) with closed‑loop compilation, profiling, and autotuning; incorporate cost models that predict occupancy, memory throughput, and arithmetic intensity.
  - Benchmark evolution: raise `p` in `fastp`, expand workloads, and extend to other accelerators; compare against more compiler backends and runtime settings.
- Practical applications
  - Auto‑tuning kernels for new models or hardware with minimal human effort (portability).  
  - Accelerating bespoke operators (losses, fusions) not covered by vendor libraries.  
  - Continuous performance regression testing in ML systems, with LMs proposing fixes guided by profiler traces.

Key citations to ground claims:
- Task design and metric: Sections 3.1–3.3; Figure 1; Appendix A/B.
- One‑shot results and failure analysis: Section 4; Table 1; Figure 2; Figure 3; Figure 7.
- Cross‑hardware behavior: Section 4.4; Appendix G (Table 13–15; Figures 8–10).
- Feedback/search improvements: Section 5.1; Figure 4; Table 2; Figure 6; Table 9.
- Few‑shot/hardware conditioning: Section 5.2; Appendix F (Tables 10–12) and G.2 (Table 15, Figure 10).
- Kernel case studies: Section 6.1; Appendix D.
