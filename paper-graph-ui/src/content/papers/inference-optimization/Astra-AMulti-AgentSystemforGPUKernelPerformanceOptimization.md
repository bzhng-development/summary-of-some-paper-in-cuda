# Astra: A Multi-Agent System for GPU Kernel Performance Optimization

**ArXiv:** [2509.07506](https://arxiv.org/abs/2509.07506)

## 🎯 Pitch

Astra introduces the first LLM-based multi-agent system designed specifically for GPU kernel optimization. By coordinating specialized agents for code generation, testing, profiling, and planning, Astra autonomously transforms existing CUDA kernels into faster, production-ready versions—achieving an average 1.32× speedup on real-world LLM serving workloads. This innovation addresses a longstanding bottleneck in deploying large language models at scale, promising substantial cost, efficiency, and energy gains in mission-critical AI infrastructure.

---

## 1. Executive Summary
Astra is a multi-agent system that uses large language models (LLMs) to automatically optimize existing GPU kernels (CUDA functions) for speed while preserving correctness. Starting from real kernels used in SGLang (a large-scale LLM serving framework), Astra’s specialized agents iteratively generate code, test functionality, profile runtime, and plan next steps, achieving an average 1.32× speedup over baselines in five rounds (Table 2), and outperforming a single-agent baseline (1.32× vs. 1.08×; Table 3).

## 2. Context and Motivation
- Problem addressed
  - Optimizing `GPU kernels` (low-level GPU functions written in CUDA) is difficult, time-consuming, and requires expert knowledge about GPU hardware, memory hierarchies, and parallelism.
  - Real deployments often run below hardware peak performance; porting to new hardware or models can cause regressions (e.g., “FlashAttention-2… suffered a 47% performance drop when first ported to NVIDIA’s H100 GPUs,” Introduction, p. 1).
- Why it matters
  - Kernel performance directly determines cost, throughput, and energy efficiency for training and serving large language models (LLMs). SGLang is “deployed at scale and responsible for generating trillions of tokens per day,” so even modest kernel gains have significant real-world impact (Introduction, p. 2).
- Prior approaches and their limits
  - Manual tuning (e.g., cuDNN) delivers strong performance but “demands extensive manual effort” and must be redone as hardware evolves (Introduction, p. 2).
  - Compiler/DSL systems (TVM, Triton, Mirage, ThunderKittens, etc.) reduce user burden but “require substantial engineering effort to develop and must be continuously adapted as hardware evolves” (Introduction, p. 2).
  - LLM-based kernel generation has focused on translating from high-level PyTorch to CUDA (KernelBench and follow-ups), which adds translation difficulties and error risks (Introduction, pp. 2–3).
- Positioning
  - Astra targets optimization of existing CUDA kernels, not translation from Python. This matches production reality (kernels already exist) and narrows the task to true performance optimization (Introduction, p. 3).
  - It uses a multi-agent design aligned with how human engineers work (testing, profiling, planning, coding), rather than a single monolithic agent (Section 3.2 and Figure 1).

Key terms defined:
- `GPU kernel`: a function executed in parallel on GPU threads, often performance-critical due to data movement and compute intensity.
- `CUDA`: NVIDIA’s programming model for writing GPU kernels.
- `SGLang`: a production LLM serving framework from which the evaluated kernels are taken.
- `LLM agent`: an LLM configured with tools and role-specific prompts to perform tasks such as code writing, testing, or profiling.

## 3. Technical Approach
Astra decomposes kernel optimization into four collaborating LLM agents that iterate in a loop (Figure 1; Algorithm 1).

- Overall workflow (Algorithm 1)
  1. Test construction and baseline profiling
     - The `TestingAgent` builds a test suite `T` from the baseline kernel `S0` (Algorithm 1, step 1).
     - The `ProfilingAgent` measures baseline performance on `T` (step 2).
     - A log records code, correctness, and performance per round (steps 3–4).
  2. Iterative optimization for `R` rounds (default `R = 5`; Implementation, p. 5)
     - `PlanningAgent` proposes targeted modifications using previous code and signals (correctness pass/fail and runtime numbers) (step 9).
     - `CodingAgent` applies the suggestions to produce a new kernel candidate (step 10).
     - `TestingAgent` validates correctness against `T` (step 11), and `ProfilingAgent` measures time (step 12).
     - Results are appended to the log for traceability and future decisions (steps 13–16).

- Correctness and performance criteria (Section 3.1)
  - Correctness is functional equivalence to the baseline kernel on a set of test inputs. The paper formalizes this using a discrepancy metric `d(·,·)` and tolerance `ε`, then instantiates it with a finite test suite `T`. A candidate is correct if:
    - max over inputs in `T` of `d(S′(x), S(x)) ≤ ε`.
  - Performance metric is speedup `σ(x) = τ(S, x) / τ(S′, x)` for runtime `τ`, aggregated as the geometric mean across `T`:
    - σ_T = [∏_i τ(S, x_i) / τ(S′, x_i)]^(1/m).
  - Rationale: geometric mean is standard for ratios and reduces outlier influence (Section 3.1).

- Agent roles and why each is necessary (Section 3.2; Figure 1)
  - `TestingAgent`: constructs diverse test cases and enforces correctness. Separation helps avoid the optimizer “learning to the test.”
  - `ProfilingAgent`: measures runtime across input shapes to give performance feedback grounded in hardware.
  - `PlanningAgent`: converts correctness/performance signals into actionable optimization plans.
  - `CodingAgent`: performs code transformations (loop changes, memory access restructuring, intrinsics).
  - Why this decomposition? The paper demonstrates single-agent systems can make biased choices (e.g., generating “unrepresentative test inputs” that mislead profiling for Kernel 1; Section 5.2), while specialization reduces such failure modes.

- Pre- and post-processing (Section 3.2)
  - Pre-processing: kernels from SGLang depend on many internals, so the team extracts standalone versions as inputs to Astra.
  - Post-processing: optimized kernels are integrated back (“monkey-patched”) into SGLang and validated against the full framework. Reported speedups are vs. original SGLang kernels, ensuring drop-in compatibility and realistic timing.

- Experimental setup specifics (Section 4)
  - Hardware: NVIDIA H100.
  - LLM: `OpenAI o4-mini`, zero-shot prompting (no fine-tuning).
  - Optimization rounds: `R = 5`.
  - Runs per input shape: 20 warm-ups + 100 timed repetitions.
  - Shapes: chosen from modern LLMs (e.g., LLaMA-7B/13B/70B).
  - Final correctness evaluation uses manually designed tests, not just agent-generated ones.

How Astra optimizes in practice (illustrative examples; Section 5.3):
- Loop-invariant code motion: compute scalars once outside the inner loop (Figure 2).
- Reduction restructuring: use warp-level shuffles (register-only) before shared-memory aggregation (Figure 3).
- Vectorized memory access: use `__half2` to load two FP16 values at once (Figure 4).
- Fast math intrinsics: replace division and standard `expf` with reciprocal–multiply and `__expf` (Figure 5).

Definitions for uncommon CUDA terms:
- `warp`: a group of 32 threads that execute the same instruction on NVIDIA GPUs.
- `__shfl_down_sync`: an intrinsic that moves data between threads within a warp via registers (fast, no shared memory).
- `shared memory`: fast on-chip memory shared by threads in a block; requires synchronization.
- `__half`/`__half2`: 16-bit floating-point types; `__half2` packs two half-precision numbers for vectorized operations.
- `fast math intrinsics`: device functions like `__expf` or `__frcp_rn` that trade small precision for speed.

## 4. Key Insights and Innovations
- Multi-agent optimization pipeline for kernels (fundamental innovation)
  - Unlike single-agent systems, Astra mirrors the human engineering workflow with distinct roles for testing, profiling, planning, and coding (Figure 1; Algorithm 1). Table 3 empirically shows this division leads to better outcomes and avoids pathologies (Kernel 1 slowdown with single-agent due to poor test construction).
- Optimize existing CUDA kernels instead of translating from high-level specs (significant shift in scope)
  - Prior LLM work often uses Python-to-CUDA translation as the task (e.g., KernelBench). Astra targets the production reality where kernels exist and need tuning. This avoids translation errors and keeps the focus on performance (Introduction, pp. 2–3).
- Zero-shot LLMs can autonomously apply expert-level GPU optimizations (new capability demonstrated)
  - Case studies show the agents reliably apply key HPC techniques: loop-invariant hoisting (Figure 2), warp-level reductions (Figure 3), vectorized loads (`__half2`, Figure 4), and fast math intrinsics (Figure 5). This demonstrates practical skill transfer from textual guidance to correct, efficient CUDA code.
- Closed-loop evaluation tied to the real serving framework (practical significance)
  - Even though optimization occurs on extracted kernels, final validation and speedups are measured relative to the original SGLang implementation, ensuring drop-in viability and meaningful performance gains in context (Section 3.2; Section 4; Table 2).

## 5. Experimental Analysis
- Evaluation methodology (Section 4)
  - Kernels from SGLang:
    - `merge_attn_states_lse` (Kernel 1): merges two attention states with log-sum-exp normalization (Table 1).
    - `fused_add_rmsnorm` (Kernel 2): elementwise add plus root-mean-square normalization.
    - `silu_and_mul` (Kernel 3): applies SiLU activation and multiplies by a gate.
  - Metrics:
    - Correctness via agreement with original SGLang kernels on diverse shapes.
    - Performance via geometric-mean speedup across shapes; timings averaged over 100 reps after 20 warm-ups.
  - Setup:
    - NVIDIA H100; `o4-mini`; R=5; manual final test suites.

- Main quantitative results (Section 5.1; Table 2)
  - Reported exactly:
    > Table 2: “Average 1.32×” speedup across 3 kernels; all pass correctness.
  - Per kernel (baseline time → optimized time; speedup):
    - Kernel 1: 31.4 µs → 24.9 µs, 1.26×; Lines of code increase +87%.
    - Kernel 2: 41.3 µs → 33.1 µs, 1.25×; LoC +50%.
    - Kernel 3: 20.1 µs → 13.8 µs, 1.46×; LoC +59%.

- Multi-agent vs single-agent (Section 5.2; Table 3)
  - Exact summary:
    > Table 3: multi-agent average 1.32× vs single-agent 1.08×, both correct.  
    > Kernel 1: single-agent 0.73× (slowdown), multi-agent 1.26×.
  - Diagnostic insight: the single-agent generated “unrepresentative test inputs,” biasing profiling and misguiding optimization for the most complex kernel. Specialization in Astra mitigated this.

- Case studies: what actually changed in the code (Section 5.3; Figures 2–5)
  - Kernel 1 (Figure 2): move expensive math (`expf`, division) out of the element loop into a precompute step; the inner loop becomes memory loads, multiply-adds, store. This reduces instruction count and hot-loop latency.
  - Kernel 2 (Figure 3): replace a shared-memory tree reduction (multiple `__syncthreads()`; diminishing active threads) with:
    - warp-local register reductions via `__shfl_down_sync` (no shared memory, no sync),
    - brief shared-memory finalize for inter-warp sums. This improves arithmetic intensity and reduces synchronization overhead.
  - Kernel 3 (Figures 4–5):
    - Vectorized global reads via `__half2`, halving the number of memory transactions compared to scalar `__half`.
    - Replace division with reciprocal–multiply and use `__expf` for faster exponentials. This increases throughput on the arithmetic pipeline.

- Shape sensitivity (Section 6.1; Table 4)
  - Exact examples:
    > Kernel 1, shape [512, 40, 128]: 32.4 µs → 20.6 µs, 1.57×.  
    > Kernel 1, shape [768, 32, 256]: 32.5 µs → 32.5 µs, 1.00× (no gain).
  - Interpretation: speedups depend on parallelism, memory coalescing, and compute/memory balance at each shape. Astra optimizes for generality rather than shape-specific tuning, so gains vary.

- Do the experiments support the claims?
  - Yes, for the stated scope:
    - All optimized kernels pass correctness against the real framework implementation.
    - Consistent performance improvements across three representative kernels drawn from production, with detailed microarchitectural explanations and code diffs.
    - The head-to-head with a matched single-agent baseline (same tools, rounds) isolates the benefit of specialization and closed-loop division of labor.

- Missing analyses
  - No ablations by agent role (e.g., removing the planning or profiling agent) beyond the single-agent comparison.
  - No breakdown of how often each optimization pattern is discovered or how many rounds each kernel needed.
  - No formal numerical tolerance `ε` or discrepancy metric specification in final tests (though the formalism is defined in Section 3.1).

## 6. Limitations and Trade-offs
- Manual pre/post-processing (Section 6.2)
  - Kernels must be extracted and simplified before Astra can optimize them; integration back into SGLang is manual. This is a non-trivial bottleneck for scale.
- Scope and generality
  - Only three kernels evaluated, though they are representative and production-relevant. Broader coverage is needed to claim generality across operator types and models.
- Shape-agnostic optimization
  - Astra does not tune to specific shapes (Section 6.1). This favors generality but sometimes leaves performance on the table compared to autotuners that specialize per shape.
- Correctness depends on testing
  - Functional validation relies on finite test suites (Section 4). While robust for practice, it does not guarantee equivalence for all inputs.
- Code complexity increases
  - Optimized kernels have 50–87% more lines of code (Table 2). This may affect long-term maintainability unless wrapped in templates/DSLs.
- Hardware specificity
  - Results are on NVIDIA H100; some intrinsics or performance characteristics may differ on other architectures or vendors.

## 7. Implications and Future Directions
- Field impact
  - Demonstrates that multi-agent LLM systems can carry out nontrivial, hardware-aware code optimization, not just code synthesis. This complements compilers/DSLs by automating expert tuning passes from natural-language plans.
- Practical applications
  - Immediate drop-in speedups for LLM serving frameworks like SGLang (validated integration, Section 3.2), with potential cost and latency benefits at production scale.
  - Potential to assist kernel developers by generating optimization candidates, regression tests, and profiling reports.
- Research avenues
  - Scaling beyond three kernels and beyond SGLang to vLLM, PyTorch, TorchTitan (Section 6.2).
  - Automating pre/post-processing with program analysis tools, stubs, and harness generation; adding human-in-the-loop checkpoints for safety and maintainability.
  - Combining Astra with training-based improvements (RL, contrastive learning) to amplify optimization skill; the paper’s zero-shot success suggests substantial headroom (Section 5, Introduction).
  - Integrating with compiler autotuners (TVM, Triton) where agents propose search spaces or seed high-quality schedules; or emitting parameterized templates for per-shape autotuning.
  - Adding formal verification or symbolic checking for tighter correctness guarantees, aligned with recent equivalence-checking benchmarks.

Key takeaways in the authors’ own numbers and figures
- > “Average 1.32×” speedup over three SGLang kernels with five rounds and zero-shot `o4-mini` (Table 2).
- > Multi-agent vs single-agent: “1.32× vs 1.08×” average, with a single-agent slowdown on the most complex kernel (Table 3).
- How the gains happen:
  - Loop-invariant hoisting (Figure 2),
  - Warp-level shuffle reductions (Figure 3),
  - `__half2` vectorized loads (Figure 4),
  - Fast math intrinsics and reciprocal–multiply (Figure 5).

In sum, Astra reframes LLM-based kernel optimization as a multi-agent, closed-loop engineering workflow grounded in correctness and hardware profiling, demonstrating credible speedups on production kernels and establishing a practical path to scale these capabilities.
