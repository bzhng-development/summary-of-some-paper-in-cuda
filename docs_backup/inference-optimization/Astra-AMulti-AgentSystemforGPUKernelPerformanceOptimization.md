# Astra: A Multi-Agent System for GPU Kernel Performance Optimization

**ArXiv:** [2509.07506](https://arxiv.org/abs/2509.07506)

## 🎯 Pitch

Astra introduces the first large language model (LLM)-powered multi-agent system dedicated to optimizing existing GPU kernels by orchestrating specialized agents for code generation, testing, profiling, and iterative planning. Unlike past LLM approaches focused on code translation, Astra works directly on production CUDA code and delivers an average 1.32× speedup on real LLM-serving workloads with zero additional training, enabling seamless and correct integration back into deployed systems. This approach not only automates a historically labor-intensive process but also unlocks tangible efficiency improvements in industry-scale AI deployments, marking a significant advance for both high-performance computing and practical machine learning infrastructure.

---

## 1. Executive Summary
Astra introduces a multi-agent, LLM-driven system that automatically optimizes existing GPU kernels (CUDA functions) by coordinating four specialized agents for planning, coding, testing, and profiling. On three real kernels extracted from the SGLang LLM-serving framework, Astra delivers an average 1.32× speedup with zero training, while preserving correctness and reintegrating back into production code (Table 2).

## 2. Context and Motivation
- Problem addressed
  - Writing fast GPU kernels remains labor-intensive and hardware-specific; even expert code can underperform after hardware changes or for new workloads. The paper notes a real example: when FlashAttention-2 was ported to NVIDIA H100s, performance initially dropped by 47% until FlashAttention-3 introduced major optimizations years later (Section 1).
- Why it matters
  - LLM training/serving throughput and cost depend on kernel efficiency. SGLang, the source of evaluated kernels, is deployed at scale “generating trillions of tokens per day,” so even modest speedups yield real-world savings (Section 1).
- Prior approaches and their limitations
  - Manual tuning (e.g., cuDNN) achieves high performance but requires repeated, expensive engineering cycles (Section 1).
  - Compiler/DSL-based systems (e.g., TVM, Triton, Mirage, ThunderKittens) reduce effort but still need substantial system engineering and often shape-/hardware-specific tuning (Sections 1–2).
  - LLM-based kernel generation exists (e.g., KernelBench and several agentic or training-based methods), but mostly translates PyTorch to CUDA rather than improving existing production kernels, adding translation errors and extra complexity (Sections 1–2).
- Positioning of this paper
  - Astra targets optimization of existing CUDA implementations extracted from SGLang, not translation from high-level Python. It designs a multi-agent pipeline to iteratively refine code with correctness checks and runtime profiling, claiming more reliable performance improvements and easier reintegration (Figure 1; Sections 1, 3.2).

## 3. Technical Approach
Astra treats kernel optimization as an iterative, multi-stage process. It formalizes correctness and performance, decomposes the workflow into specialized LLM agents, and loops through suggestion → implementation → validation → profiling.

- Problem formulation (Section 3.1)
  - Given a baseline kernel `S` and an optimized kernel `S'`, the goal is to make `S'` faster while staying functionally equivalent to `S`.
  - Correctness: since exact equivalence is impractical to prove, Astra uses a finite test suite `T = {(xi, yi)}` where `yi = S(xi)`. `S'` is considered correct if outputs are within a tolerance `ε` on all test cases: `max_i d(S'(xi), yi) ≤ ε`.
  - Performance: For each input `x`, `σ(x) = τ(S, x) / τ(S', x)` where `τ` is runtime. Over a test suite, Astra reports the geometric mean speedup: `σ_T = (Π_i τ(S, xi)/τ(S', xi))^(1/m)`. The geometric mean is preferred because it aggregates ratios correctly and reduces outlier influence.
- Multi-agent architecture (Figure 1; Section 3.2)
  - `Testing Agent`: builds or runs test cases and validates candidate kernels against baseline outputs.
  - `Profiling Agent`: times kernel executions on the test suite.
  - `Planning Agent`: uses test and profiling feedback to suggest targeted changes.
  - `Coding Agent`: edits the previous kernel implementation to apply the plan.
- Optimization loop (Algorithm 1)
  1) Initialize: Testing Agent creates a test suite `T`; Profiling Agent profiles the baseline `S0`.
  2) Iterate R times (R=5 in experiments, Section 4): Planning Agent proposes edits based on previous round’s correctness and performance; Coding Agent produces a new kernel `S_new`; Testing Agent validates; Profiling Agent re-measures runtime. The system logs every round’s code, pass/fail, and performance.
- Pre-/post-processing for practicality (Section 3.2)
  - Pre-processing: SGLang kernels have internal dependencies. The authors manually extract and simplify them into standalone kernels that Astra can optimize.
  - Post-processing: optimized kernels are re-integrated (“monkey-patched”) back into SGLang and validated against the original framework implementation. Reported speedups are relative to original SGLang kernels, measured in the full framework.
- Implementation choices (Section 4)
  - Agents are implemented with the OpenAI Agents SDK (tool abstraction) and powered by `o4-mini` in zero-shot prompting mode. Hardware: NVIDIA H100 GPUs. Each performance measurement uses 20 warm-ups plus 100 repetitions for robustness.

How it works in practice (example path through the loop)
- Planning Agent might suggest “hoist loop-invariant computations out of the inner loop.”
- Coding Agent edits the kernel accordingly.
- Testing Agent runs the modified kernel on diverse tensor shapes and checks numerical tolerances against the baseline.
- Profiling Agent measures speedups to determine whether the change improved performance.
- The process repeats, compounding multiple improvements while guarding correctness.

Why multi-agent rather than single-agent?
- Kernel optimization spans distinct skills: writing CUDA, selecting representative tests, interpreting profiles, planning next steps. The paper’s single-agent comparison shows that when one agent does everything, it can make poor early choices (e.g., unrepresentative tests), bias profiles, and degrade results (Table 3 and Section 5.2).

## 4. Key Insights and Innovations
- Multi-agent decomposition of kernel optimization
  - What’s new: The system formalizes a role-specialized loop—testing, profiling, planning, coding—rather than asking one LLM to do everything (Figure 1; Algorithm 1).
  - Why it matters: Table 3 shows an average 1.32× speedup for the multi-agent setup versus 1.08× for a single agent under the same tools/rounds. Notably, the single-agent even slows down Kernel 1 (0.73×), while the multi-agent achieves 1.26×.
- Optimize existing production CUDA, not translate from Python
  - What’s new: Prior LLM efforts like KernelBench typically generate CUDA from PyTorch, which is error-prone and misaligned with production needs (Sections 1–2).
  - Why it matters: Astra starts from SGLang’s kernels and returns drop-in replacements validated in the full framework (Section 3.2), improving real-world deployability.
- Zero-shot prompt-only system with measurable gains
  - What’s new: No extra training (no supervised finetuning or RL) beyond `o4-mini` prompting (Section 1, Abstract).
  - Why it matters: Demonstrates that prompt engineering plus a structured agent loop can yield nontrivial performance gains (1.32× average; Table 2), leaving headroom for training-based enhancements (Section 2).
- LLMs can autonomously apply expert GPU-optimization techniques
  - Evidence: Case studies (Section 5.3; Figures 2–5) show:
    - Loop-invariant hoisting (Figure 2) to remove repeated `exp` and division from hot loops.
    - Warp-level shuffle reductions (Figure 3) to keep partial sums in registers and minimize synchronization.
    - Vectorized memory loads with `__half2` (Figure 4) to double effective bandwidth for FP16 data.
    - Fast math intrinsics (`__expf`, `__frcp_rn`) and reciprocal–multiply in place of division (Figure 5) to reduce latency.

## 5. Experimental Analysis
- Evaluation methodology (Section 4)
  - Kernels: three real kernels from SGLang (Table 1):
    - `merge_attn_states_lse`: mixes two value vectors `Va`, `Vb` using exponentiated scores with log-sum-exp normalization; outputs both the mixed values and the resulting log-normalizer.
    - `fused_add_rmsnorm`: computes `y = (x + r) / sqrt(mean((x + r)^2) + ε) ⊙ w` (RMSNorm fused with residual add and scaling).
    - `silu_and_mul`: computes `out = SiLU(x) ⊙ g` with `SiLU(z) = z / (1 + e^-z)`.
  - Metrics:
    - Correctness: pass/fail against the original SGLang outputs using diverse tensor shapes and numerical tolerance (Sections 3.1 and 4).
    - Performance: runtime speedup over the baseline SGLang kernels; measurements are averaged over multiple shapes and repetitions; speedups reported as geometric means (Section 3.1, 4).
  - Setup: NVIDIA H100 GPUs; 20 warm-ups + 100 measured runs per shape; five optimization rounds (R=5).
- Main quantitative results (Table 2)
  - “All three optimized kernels are correct.”
  - Speedups and code size:
    - Kernel 1 (`merge_attn_states_lse`): 1.26× speedup, Lines of Code (LoC) +87% (124 → 232).
    - Kernel 2 (`fused_add_rmsnorm`): 1.25× speedup, LoC +50% (108 → 163).
    - Kernel 3 (`silu_and_mul`): 1.46× speedup, LoC +59% (99 → 157).
  - Average across kernels: “1.32× speedup” with correctness preserved.
- Multi-agent vs. single-agent (Table 3; Section 5.2)
  - Quote:
    > Kernel 1 baseline 31.4 µs: Single-Agent 0.73× vs. Multi-Agent 1.26×  
    > Kernel 2 baseline 41.3 µs: Single-Agent 1.18× vs. Multi-Agent 1.25×  
    > Kernel 3 baseline 20.1 µs: Single-Agent 1.48× vs. Multi-Agent 1.46×  
    > Average: Single-Agent 1.08× vs. Multi-Agent 1.32× (all correct).
  - Diagnosis: The single-agent underperformed on Kernel 1 due to unrepresentative tests that biased profiling; the multi-agent avoided this pitfall with dedicated testing and profiling roles (Section 5.2).
- Case studies: mechanisms behind speedups (Section 5.3)
  - Kernel 1: Hoisting loop-invariant terms (Figure 2) eliminates repeated `expf` and division from the inner loop, leaving only multiply-adds and memory ops.
  - Kernel 2: Warp-level reduction with `__shfl_down_sync` (Figure 3) keeps partial sums in registers (intra-warp), then briefly uses shared memory for inter-warp aggregation—reducing synchronization and memory traffic versus pure shared-memory tree reduction.
  - Kernel 3: Vectorized `__half2` loads (Figure 4) and fast math intrinsics with reciprocal–multiply vs. division (Figure 5).
- Shape sensitivity (Table 4; Section 6.1)
  - Quote:
    > Kernel 1: [512, 40, 128] shows 1.57×, while [768, 32, 256] shows 1.00×.  
    > Kernel 2: ranges from 1.07× to 1.33× across tested shapes.  
    > Kernel 3: consistently ~1.47–1.50× across shapes.
  - Interpretation: Gains can vary by tensor shape (e.g., occupancy, memory coalescing). Astra does not tune to a specific shape; it aims for generally good improvements (Section 6.1).
- Are the experiments convincing?
  - Strengths:
    - Real production kernels; reintegration into SGLang; correctness validated against framework outputs (Section 3.2).
    - Clear mechanisms illustrated via code deltas (Figures 2–5).
    - Controlled profiling protocol (warm-ups, repetitions) and geometric-mean reporting.
    - Single-agent ablation isolates the value of specialization (Table 3).
  - Caveats:
    - Only three kernels; pre-/post-processing is manual; generality across frameworks is untested (Section 6.2).
    - LoC increases substantially (Table 2), which could affect maintainability or readability.

## 6. Limitations and Trade-offs
- Scope and generality
  - Evaluation covers three kernels from one framework (SGLang); behavior on a broader set or other frameworks (vLLM, PyTorch, TorchTitan) is future work (Section 6.2).
- Manual pre/post processing
  - Kernel extraction and reintegration are manual, non-trivial steps that may limit scalability (Section 3.2, 6.2).
- Correctness guarantees
  - Correctness is established via finite tests and numerical tolerances, not formal verification; equivalence for all inputs is not guaranteed (Section 3.1).
- Performance variability
  - Speedups depend on tensor shapes (Table 4) and may be neutral in some cases (e.g., 1.00× for Kernel 1 at [768, 32, 256]).
- Code complexity
  - Optimized kernels are longer (+50–87% LoC; Table 2), potentially increasing maintenance overhead or the chance of corner-case bugs. Though they pass the test suite, more code can be harder to reason about.
- Reliance on model/tooling
  - The approach depends on the capabilities of a specific LLM (`o4-mini`) and the Agents SDK tooling. Different models or tool configurations may produce different outcomes.

## 7. Implications and Future Directions
- Impact on the field
  - Demonstrates that a structured, multi-agent LLM workflow can produce meaningful, correctness-preserving performance gains on production GPU kernels without additional training. This complements compiler/DSL ecosystems by offering a new, agentic optimization layer (Sections 1–2, 5).
- Practical applications
  - Immediate utility for LLM-serving/training stacks where many specialized kernels exist: integrating Astra into CI/CD could automatically propose optimizations, validate them on representative shapes, and surface candidates for human review.
- Research directions
  - Automate pre-/post-processing: agentic extraction of kernels from complex codebases, dependency pruning, and safe reintegration (Section 6.2).
  - Broaden coverage: more kernels and frameworks (vLLM, PyTorch, TorchTitan) and more GPU architectures to test cross-hardware robustness (Section 6.2).
  - Stronger correctness: combine property-based testing, fuzzing, differential testing against multiple baselines, or formal methods for select kernels (Section 3.1’s correctness framing suggests this trajectory).
  - Joint optimization with compilers: use agents to drive Triton/TVM autotuners, propose schedule templates, or select intrinsic use under hardware counters.
  - Learning-enhanced agents: layer supervised finetuning or RL (cited in Section 2) atop the agent loop to further increase success rates and speedup magnitudes.
  - Shape-aware strategies: while Astra avoids shape-specific tuning (Section 6.1), a hybrid mode could auto-detect deployment distributions and maintain a small per-shape portfolio of optimized kernels.

Overall, Astra reframes GPU-kernel optimization as a cooperative, tool-using conversation among specialized LLM agents, showing measurable gains on real production code. The system design, empirical results (Tables 2–4), and transparent code-level case studies (Figures 2–5) make a strong case that multi-agent LLMs can now perform nontrivial systems optimization work and are ready to be integrated into the optimization toolchain alongside compilers and human experts.
