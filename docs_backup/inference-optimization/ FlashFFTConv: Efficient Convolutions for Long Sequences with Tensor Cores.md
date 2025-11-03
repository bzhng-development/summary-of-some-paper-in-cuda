# FlashFFTConv: Efficient Convolutions for Long Sequences with Tensor Cores

**ArXiv:** [2311.05908](https://arxiv.org/abs/2311.05908)
**Authors:** Daniel Y. Fu, Hermann Kumbong, Eric Nguyen, Christopher Ré
**Institutions:** Stanford University (Hazy Research)

## 🎯 Pitch

FlashFFTConv revolutionizes long-sequence convolutions by optimizing the Fast Fourier Transform (FFT) for modern GPUs, leveraging tensor cores and kernel fusion to enhance speed and efficiency. This innovation bridges the performance gap with Transformer models, enabling competitive wall-clock speeds and scalability in domains like language, vision, and genomics, ultimately maximizing throughput and context length capabilities.

---

## 1. Executive Summary (2-3 sentences)
FlashFFTConv is a system that makes long-sequence convolutions fast and memory‑efficient on modern GPUs by rewriting the Fast Fourier Transform (FFT) as a series of matrix multiplications that run on tensor cores and by aggressively fusing kernels to cut data movement. This matters because long convolutions power state‑of‑the‑art non‑Transformer sequence models, yet standard FFT implementations underutilize hardware and spend most time on memory I/O; FlashFFTConv closes that gap and enables longer contexts, higher throughput, and in several cases better accuracy for the same compute.

## 2. Context and Motivation
- Problem addressed
  - Long-sequence convolution models use filters as long as the input; naive time‑domain convolution is quadratic in sequence length. Using the FFT makes the operation O(N log N), but off‑the‑shelf FFTs run poorly on GPUs for long sequences: they don’t use tensor cores and incur heavy memory traffic between GPU memory levels (Section 1; Figure 1 middle right).
- Why it’s important
  - Convolutional sequence models (e.g., Hyena, S4, M2) deliver competitive or state‑of‑the‑art quality in language, time series, vision, and genomics, and can be more stable and better‑scaling in context length than attention (Section 1). Yet they lag Transformers in wall‑clock speed, limiting adoption and maximum context.
- Where prior approaches fall short
  - Classical FFT algorithms broadcast matrix operations across batch and channels, not sequence (Figure 3, top left), so fusing the end‑to‑end pipeline becomes infeasible once sequences exceed the capacity of on‑chip memory (SRAM). Tensor cores remain largely idle because the FFT is dominated by small transforms and permutations, which are memory‑bound (Sections 1–2.2).
- Positioning relative to existing work
  - FlashFFTConv builds on the idea that systems advances can unlock better models (cf. FlashAttention). It leverages a structured factorization of FFTs—the order‑`p` Monarch decomposition (Section 2.1; Figure 2)—and reorients it to exploit tensor cores and to enable kernel fusion at long lengths, adding domain‑specific optimizations for real signals and causal padding (Section 3.1). It also introduces convolutional analogues of attention sparsity—partial and frequency‑sparse convolutions (Section 3.3).

## 3. Technical Approach
Step-by-step, how FlashFFTConv turns a slow FFT‑based convolution into a tensor‑core‑friendly, IO‑efficient pipeline.

- What the computation is
  - A long convolution `y = u * k` (sequence length `N`) is computed via the FFT as:
    - Conceptually: convolve in time by multiplying in frequency.
    - Formally (Equation 1): `u * k = F^{-1}(F u ⊙ F k)`, where `⊙` is elementwise multiply.
  - In practice, convolution layers reuse the kernel FFT `k_f = F k` across a batch, leaving just an FFT of each input, a pointwise multiply by `k_f`, and an inverse FFT (Section 1).

- Why naive FFTs are slow on GPUs
  - Two bottlenecks arise at long `N` (Section 1):
    - Poor use of specialized matrix units (tensor cores): FFTs are implemented as many tiny operations and permutations rather than large matrix multiplies.
    - Costly I/O across the memory hierarchy (HBM → SRAM → registers): as sequences get large, intermediate tensors can no longer be kept on‑chip, so kernels can’t be fused; padding for causality and real↔complex conversions add extra traffic (Figure 1 left and middle right).

- Core idea 1: Monarch FFT decomposition mapped to matrix multiplies
  - Monarch decomposition expresses an `N × N` FFT matrix `F_N` as a product of `p` block‑structured transforms (Section 2.1; Figure 2). For order‑2 (two‑way):
    - `F_N = P (I_{N2} ⊗ F_{N1}) D P^{-1} (I_{N1} ⊗ F_{N2}) P`, with:
      - `⊗` Kronecker product (applies many small FFTs in parallel),
      - `P` permutations (reshape/transpose bookkeeping),
      - `D` diagonal twiddle factors (phase corrections).
  - Higher order `p` recursively applies this factorization, trading more, smaller matrix multiplications for additional permutations (Section 2.1). Crucially, each factor can be executed as matrix multiplications sized to tensor cores.

- Core idea 2: Broadcast along the sequence to enable fusion
  - Classical FFT variants broadcast the small FFTs over batch and channels (Figure 3 top left), which forces loading many sequences concurrently to fill tensor cores—unsuitable for long sequences.
  - FlashFFTConv flips the broadcast dimension to the sequence itself (Figure 3 top right):
    - Each small transform multiplies a block along the length dimension, and the algorithm runs in parallel across batch (`B`) and channels (`H`).
    - The expensive global permutations become fast on‑chip matrix transposes (Figure 3 bottom).
    - Result: only a single sequence needs to live in SRAM per SM, allowing kernel fusion up to 32K tokens on A100/H100 and fused innermost steps even beyond (Section 3.1; Algorithm 1).

- Core idea 3: Kernel fusion and recomputation
  - For long sequences, inner matrix multiplications and elementwise ops are fused and kept on‑chip; only the outermost steps touch HBM (Section 3.1).
  - Backward pass uses recomputation rather than storing large intermediates (e.g., re‑do `F u` instead of saving it), cutting memory footprint and I/O (Section 3.1; Tables 16–17).

- Domain‑specific optimizations for sequence learning
  - Real‑to‑real FFT: since inputs and kernels are real, a standard trick computes a size‑`N` real FFT via a size‑`N/2` complex FFT (Appendix A.1). FlashFFTConv implements this “decimation in time” method to halve FFT cost (Section 3.1, “Domain‑Specific Optimizations”).
  - Implicit causal padding: causal convolutions zero‑pad inputs; FlashFFTConv recognizes these zeros and skips half of the outermost matmuls in the FFT/iFFT (Section 3.1).
  - Fuse common gating: many long‑conv blocks use multiplicative gating `y = v ⊙ ((u ⊙ w) * k)`; FlashFFTConv fuses the two elementwise multiplications into the FFT pipeline to avoid extra HBM reads/writes (Section 3.1 and Table 4).

- Order‑`p` cost model: choosing how much to factorize
  - Intuition: larger `p` yields smaller matmuls (fewer FLOPs) but introduces more intermediate results (more I/O). There is an optimal `p` that depends on sequence length and hardware (Section 3.2).
  - Plain language version of Equation 2: total cost = compute time of the `p` matmul stages + I/O time to move the `p` intermediate results through memory. It accounts for whether each matmul meets the minimal size to run on tensor cores.
  - Formal (Equation 2): `C = B H ∑_{i=1}^p [ 16 N N_i / γ(N_i) + 4 N / ω(i) ]`, where:
    - `N = Π_i N_i` is the factorization, `μ` is the tensor‑core tile size (e.g., 16 on A100/H100),
    - `γ(N_i)` equals tensor‑core FLOPs `τ_M` if `N_i ≥ μ`, else general FLOPs `τ_G`,
    - `ω(i)` is the bandwidth of the memory level used at stage `i` (HBM vs SRAM),
    - Empirical constants for A100 are given in Appendix C/Table 19.
  - Figure 4 shows per‑token cost vs sequence length for `p ∈ {2,3,4}` on A100:
    - At short lengths, higher `p` hurts (matrices fall below tensor‑core size; “Matrices Too Small”).
    - At mid/long lengths, higher `p` helps until SRAM becomes the bottleneck (the bump for `p=3` near 32K–64K is from exhausting SRAM; `p=4` regains ground by further factoring).

- Architectural extensions: sparsity mapped to skipped matmuls
  - Partial convolutions: learn shorter kernels (like local attention). Implementation: skip parts of the FFT pipeline corresponding to trailing zeros in time domain; reduces memory and enables sliding‑window extension to longer contexts (Section 3.3; Section 4.3).
  - Frequency‑sparse convolutions: zero out portions of `k_f` (frequency response). Implementation: skip specific blocks inside the Monarch matmuls that would multiply by zero, yielding actual compute savings without changing outputs (Section 3.3; Appendix A.4 explains which blocks can be skipped in 4‑way decomposition).

- Low‑level CUDA execution (Appendix A.2–A.5)
  - Uses the WMMA API to run 16×16×16 fp16/bf16 matmuls on tensor cores; carefully aligns data layouts so accumulator fragments can be reused as inputs to avoid SRAM round‑trips (Algorithm 2).
  - Double‑buffered I/O across memory levels, vectorized loads/stores, warp‑level tiling across `B` and `H`.
  - Hardware support currently targets A100/H100; V100 not supported due to different tensor‑core tile sizes (Appendix A.5).

## 4. Key Insights and Innovations
- Turning FFTs into tensor‑core matmuls at long sequence lengths
  - Novelty: Adapts the Monarch factorization specifically to broadcast over the sequence dimension, not batch/channels (Figure 3), so the FFT’s heavy lifting becomes well‑sized matmuls for tensor cores.
  - Significance: Converts a traditionally memory‑bound primitive into compute‑efficient steps with high FLOP utilization; enables fusion and reduces HBM traffic (Section 3.1).
- IO‑aware order‑`p` factorization with a simple roofline‑style cost model
  - Novelty: Equation 2 blends compute throughput (tensor‑core vs general) with memory bandwidth at each factorization stage; selects `p` based on sequence length and per‑GPU constants (Figure 4; Appendix C/Table 19).
  - Significance: Explains when and why to change decomposition order as `N` grows; predicts the “bumps” seen in practice (e.g., `p=3` bump near 32K–64K).
- Domain‑specific fusion for real, causal, gated long‑conv blocks
  - Novelty: Integrates “real FFT via N/2 complex FFT” (Appendix A.1), implicit causal padding, and common gating `y = v ⊙ ((u ⊙ w) * k)` into the fused pipeline (Algorithm 2).
  - Significance: Delivers the largest measured speedups—up to 7.93× vs PyTorch for gated convolutions (Table 4)—and the biggest memory savings in end‑to‑end models (Tables 16–17).
- Convolutional analogues of sparse/approximate attention
  - Novelty: Defines partial (time‑domain) and frequency‑sparse (frequency‑domain) convolution schemes that map cleanly to “skipping blocks” in the Monarch matmuls (Section 3.3; Appendix A.4).
  - Significance: Enables longer‑sequence modeling (first single‑nucleotide embeddings of the longest human genes at 2.3M bp; Table 8) and further runtime savings at the same or better quality (Table 9).

## 5. Experimental Analysis
- Evaluation setup
  - Benchmarks span synthetic kernels and full models across modalities and sequence lengths from 256 to 4M (Tables 3–4, 11–17).
  - Models include M2‑BERT‑base (masked LM), Hyena small (GPT‑style), a long‑conv model on Long Range Arena Path‑X/Path‑512, SaShiMi (audio), and HyenaDNA (genomics) (Sections 4.1–4.2; Table 5).
  - Metrics: wall‑clock time, sequences/tokens per second, memory footprint, FLOP utilization, perplexity (PPL), GLUE score, and task accuracy.

- Main results (quantitative, with citations)
  - Convolution kernels
    - Forward speedups vs PyTorch up to 6.54× for plain conv (Table 3, 1K seq), and up to 7.93× for gated conv (Table 4, 1K seq).
    - Memory savings up to 8.21× for conv (Table 3/16, 256 seq) and 6.65× for gated conv (Table 4/17, 256 seq), still 2.6–2.8× at million‑token scale (Tables 16–17).
    - Backward pass is also faster: 1.45–6.43× over PyTorch depending on length (Table 15).
  - End‑to‑end model throughput (Table 5)
    - M2‑BERT‑base (128 tokens): 1.9× sequences/s.
    - Hyena‑s‑4K: 1.7× sequences/s.
    - Path‑X conv model (16K): 2.4× images/s.
    - SaShiMi (64K audio): 1.3× clips/s (convolutions are a smaller fraction of end‑to‑end time here).
    - HyenaDNA‑1M: 4.4× sequences/s by enabling a 4× larger batch than PyTorch.
  - Quality at fixed compute (Table 1)
    - “More training for the same budget” effect: higher throughput lets models see more tokens.
    - Reported gains: 
      > M2‑BERT‑base average GLUE: 77.6 → 80.9  
      > Hyena‑s perplexity on The Pile: 13.4 → 11.1
  - Long‑context capability (Table 2)
    - Path‑512 (sequence 256K): prior convolutional setups OOM, yet
      > FlashFFTConv achieves 96.1% accuracy.  
      Path‑X (16K) remains at 96.9% (no regression).
  - Transformer comparison (Table 6, same 2.7B params)
    - Tokens/s: Hyena with FlashFFTConv is faster at 2K, 8K, 16K (1.1×, 1.3×, 1.5×).
    - FLOP utilization end‑to‑end: FlashFFTConv ~56–62% vs FlashAttention‑v2 66–79%. Despite lower utilization, convolution has fewer FLOPs, so wall‑clock wins.
  - Partial convolutions (Tables 7–8)
    - Training memory reduction with little/no loss in quality for Hyena‑s‑8K: convolution kernel can be shortened down to 2K with essentially unchanged PPL (Table 7).
    - Extending HyenaDNA to longer sequences via sliding window over short filters:
      > At 4M length, PPL matches or slightly improves (2.91 → 2.90) while enabling embedding of the longest human genes (Table 8; Appendix Figure 5).
  - Frequency‑sparse convolutions (Table 9; Appendix A.4)
    - Zeroing 50–79% of frequency coefficients leaves PPL unchanged (2.91–2.90), and
      > yields up to 1.4× extra speedup in the convolution.  
      Quality starts to degrade beyond ~84% sparsity.

- Do the experiments support the claims?
  - Speed and memory: Yes. Multiple sequence lengths, forward/backward breakdowns, and end‑to‑end models show consistent gains. Tables 3–4 and 11–15 confirm that the claimed “up to 7.93×” kernel speedup is achieved in realistic guarded/gated convs; memory reductions match the recomputation/fusion story (Tables 16–17).
  - Quality at fixed compute: Plausible and quantified. Throughput gains translate to more tokens seen; the measured improvements (Table 1) are in line with scaling‑law expectations and are calibrated against larger baselines (Appendix B.2, Table 18).
  - Long‑context and sparsity: Convincing. Path‑512 success (Table 2) demonstrates concrete new capability; partial/frequency‑sparse results include both quality and speed metrics (Tables 7–9) and explainability via skip‑patterns (Appendix A.4).

- Ablations and robustness
  - Fusion‑only (no tensor cores) baselines show that tensor‑core matmuls are necessary for long‑sequence performance; otherwise the kernel becomes compute‑bound on general ALUs and runs out of SRAM beyond 32K (Table 3, “Fusion‑Only/cuFFTdx”).
  - Domain‑specific fusions (gating, causal) provide additional measurable gains (Table 4; Appendix B.1 Tables 13–14).
  - Cost‑model sanity: Figure 4 explains where each `p` dominates and matches observed performance transitions (e.g., SRAM limit bump for `p=3` near 32–64K).

## 6. Limitations and Trade-offs
- Hardware specificity
  - Current implementation is optimized for NVIDIA A100/H100 tensor‑core tile sizes and memory characteristics (Appendix A.5; Appendix C/Table 19). Older GPUs (e.g., V100) are not supported; portability to non‑GPU accelerators is future work.
- Diminishing speedups at extreme lengths
  - At multi‑million tokens, speedups over PyTorch narrow (e.g., 1.3–1.8× at 2–4M; Tables 3–4, 11–14) as SRAM and HBM I/O dominate and only outermost stages remain unfused.
- Order‑`p` selection depends on hardware and `N`
  - The optimal factorization changes with sequence length and memory limits (Figure 4). Mis‑tuned `p` (e.g., matrices too small for tensor cores) can lose much of the benefit.
- Precision and numerical considerations
  - The implementation leans on fp16/bf16 tensor cores (Appendix A.2). Although standard in DL, some applications may require higher precision or careful scaling to avoid numerical artifacts in frequency space when using sparsity.
- Applicability scope
  - The strongest benefits show up for long convolutions with shared kernels across batch (standard in sequence models). Workloads with tiny kernels (e.g., typical 2D convs in vision) aren’t the target; those already use different fast paths.
- Sparsity patterns and learned kernels
  - Frequency‑sparse zeroing is applied post‑pretraining for HyenaDNA; while small to moderate sparsification preserves quality (Table 9), the optimal pattern may be model‑ and task‑dependent (Appendix A.4). Learning sparsity during training remains open.

## 7. Implications and Future Directions
- How this changes the landscape
  - By making long‑sequence convolutions competitive in wall‑clock time and memory with highly optimized attention, FlashFFTConv removes a key systems barrier. This strengthens the case for convolution‑ and state‑space‑based architectures in domains where long context, stability, or linear‑time scaling is advantageous (Section 4.2; Table 6).
- Research directions enabled
  - Hardware‑aware algorithm design for other fast transforms (e.g., wavelets, Chebyshev) via Monarch‑style factorizations and broadcast‑along‑sequence patterns.
  - Learning partial/frequency‑sparse structure end‑to‑end (jointly selecting skip‑blocks in the Monarch matmuls) as a principled analogue to sparse attention patterns (Section 3.3; Appendix A.4).
  - Automated `p`‑selection and tiling using the cost model (Equation 2) plus online profiling; extending to heterogeneous clusters (HBM tiers, NVLink/PCIe effects).
  - Portability beyond NVIDIA: re‑implement using platform‑agnostic libraries (e.g., CUTLASS variants) and adapt to emerging accelerators with different tensor‑core shapes (Appendix A.5).
- Practical applications
  - Long‑context language models that maintain high throughput beyond 8–16K tokens (Table 6).
  - High‑resolution vision or spatial reasoning tasks requiring hundreds of thousands of tokens (Table 2).
  - Genomics at single‑nucleotide resolution over multi‑megabase regions, including embedding the longest human genes (Table 8; Appendix Figure 5).
  - Time‑series and audio models operating at native sampling rates over long windows with efficient training and inference (Table 5, SaShiMi).

> In sum, FlashFFTConv reframes the FFT for long convolutions as a sequence‑broadcasted, tensor‑core‑friendly pipeline with IO‑aware factorization and fusion (Algorithms 1–4; Figures 2–4). The measured kernel‑level and end‑to‑end gains (Tables 3–6, 11–17), the quality improvements at fixed compute (Table 1), and the new long‑context capabilities (Tables 2, 7–9) collectively support its central claim: efficient FFT convolutions at long sequence lengths are achievable on today’s accelerators, and doing so unlocks both better models and new applications.
