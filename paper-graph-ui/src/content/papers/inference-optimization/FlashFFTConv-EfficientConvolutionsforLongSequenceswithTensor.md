# FlashFFTConv: Efficient Convolutions for Long Sequences with Tensor Cores

**ArXiv:** [2311.05908](https://arxiv.org/abs/2311.05908)

## 🎯 Pitch

FlashFFTConv introduces a groundbreaking approach to long-sequence convolutions by reimagining Fast Fourier Transform (FFT) operations as matrix multiplications optimized for GPU tensor cores, while fusing computations to minimize costly memory traffic. This innovation breaks through the hardware bottlenecks of conventional FFT convolutions, delivering up to 7.93× faster performance and significant memory savings—enabling state-of-the-art results in domains like language, vision, and genomics, with longer context windows and superior model quality on the same compute budget.

---

## 1. Executive Summary (2-3 sentences)
FlashFFTConv is a system that makes very long 1D convolutions fast and memory‑efficient on modern GPUs by restructuring the Fast Fourier Transform (FFT) into matrix multiplies that run on tensor cores and by fusing the FFT, pointwise multiply, and inverse FFT into a single IO‑aware kernel. This solves a key bottleneck in long‑sequence modeling: conventional FFT‑based convolutions underutilize GPU compute and suffer heavy memory traffic, whereas FlashFFTConv delivers up to 7.93× faster convolutions, large memory savings, and enables longer contexts and better quality at the same compute budget (Tables 3–5, Table 1).

## 2. Context and Motivation
- Problem addressed
  - Many high‑performing sequence models (language, time series, vision, genomics) use very long 1D convolutions whose filter length matches the sequence length. The efficient way to compute such “long filters” is the FFT convolution: transform input and filter to frequency space, multiply elementwise, then transform back (Equation (1), Sec. 2.1). On modern accelerators, this is still slow in wall‑clock time.
- Why it matters
  - Long‑sequence tasks are increasingly common: long‑context language modeling, raw‑audio modeling at 44–64 kHz, high‑resolution images unrolled into long sequences, and DNA modeling at single‑base resolution (Sec. 1). Faster long convolutions raise feasible context windows (e.g., up to 4M tokens) and can improve quality under fixed compute by increasing training throughput (Table 1; DNA results in Tables 8–9).
- Where prior approaches fall short
  - Current FFT convolutions waste hardware potential in two ways (Sec. 1; Fig. 1 middle right):
    - They don’t leverage tensor cores (specialized matrix‑multiply units) well; on an H100, tensor cores are ~15× faster than general arithmetic (1.0 PFLOP/s vs 67 TFLOP/s; Sec. 2.2).
    - They incur heavy IO across the memory hierarchy (HBM↔SRAM↔registers; Fig. 1 left), especially at long sequence lengths where on‑chip memory can’t hold intermediate results, breaking kernel fusion.
- Positioning relative to existing work
  - Systems like FlashAttention show that IO‑aware kernels can transform end‑to‑end efficiency for Transformers. FlashFFTConv plays an analogous role for FFT‑based convolutions: it reexpresses the FFT as a sequence of matrix multiplies (via a Monarch decomposition) to run on tensor cores and designs fusion strategies that scale to very long sequences (Secs. 3.1–3.2; Figs. 2–4). It also introduces “partial” and “frequency‑sparse” convolution variants analogous to sparse/approximate attention (Sec. 3.3).

## 3. Technical Approach
The core idea is to compute FFT‑based convolutions using matrix multiplies on tensor cores while keeping intermediate data on‑chip as much as possible.

- Primer: FFT convolution
  - A long convolution is computed by transforming input `u` and kernel `k` to frequency domain, multiplying component‑wise, and transforming back: `(u * k) = F^{-1}(F u ⊙ F k)` (Equation (1), Sec. 2.1). This reduces algorithmic cost from O(N^2) to O(N log N) but is typically IO‑bound in practice.

- Step 1 — Rewrite FFT as tensor‑core GEMMs using Monarch decomposition
  - Monarch decomposition: a structured factorization of the N‑point FFT matrix into a sequence of small FFTs, permutations, and “twiddle” diagonal corrections (Fig. 2; Sec. 2.1). An order‑2 version for `N = N1 N2` expresses `F_N` as `P (I_{N2} ⊗ F_{N1}) D P^{-1} (I_{N1} ⊗ F_{N2}) P`, where `⊗` is Kronecker product, `P` is a reshape/transpose permutation, and `D` is the diagonal twiddle matrix.
  - Key choice: broadcast matrix multiplies across the sequence dimension, not across batch/hidden (Fig. 3 top right vs top left; Sec. 3.1). Why this matters:
    - It lowers the per‑SM on‑chip memory requirement because each SM only needs a single sequence at a time, enabling fusion for much longer sequences (up to ~32K on A100/H100; Sec. 3.1).
    - Permutations become simple on‑chip transposes in SRAM instead of HBM shuffles (Fig. 3 bottom).

- Step 2 — Fuse kernels and minimize IO
  - FlashFFTConv fuses inner matrix operations (FFT/iFFT substeps) with elementwise multiplications (by the frequency‑domain kernel `k_f`) so that data stays in registers/SRAM, writing to HBM only at outer boundaries (Sec. 3.1, “Kernel Fusion and Recomputation”).
  - Backward pass uses recomputation: intermediates like `F u` are not stored to HBM; they’re recomputed, which trades extra compute for large memory savings (Sec. 3.1; memory results in Tables 16–17).

- Step 3 — Domain‑specific optimizations for ML convolutions
  - Real‑valued FFT shortcut: for real inputs/filters (standard in ML), a complex FFT of length `N/2` suffices to compute a real FFT of length `N` (Appendix A.1). FlashFFTConv implements this “one‑stage decimation in time,” cutting FFT length and cost roughly in half (Sec. 3.1).
  - Zero‑padding for causality: input/output padding is common when models need causal convolutions. FlashFFTConv exploits known zeros to skip parts of the outermost matrix multiplies in FFT and iFFT (Sec. 3.1), reducing IO and FLOPs.
  - Tiling on `(B, H)`: to amortize filter and twiddle loads, the kernel tiles over batch `B` and hidden `H` (Algorithm 1).

- Step 4 — Cost model to choose decomposition order p
  - Order‑p Monarch: recursively applies the decomposition, trading fewer FLOPs (smaller submatrices) for more IO (more stages). The cost model (Equation (2), Sec. 3.2) combines compute and IO:
    - Compute term scales with matrix size in each stage and the effective FLOP rate `γ(N_i)` that switches from general cores to tensor cores when the matrix side `N_i` exceeds the tensor‑core tile size `µ` (e.g., 16 on A100/H100; Sec. 3.2).
    - IO term accounts for bandwidth where each stage’s intermediate is stored (`ω(i)` chooses HBM or SRAM; Sec. 3.2).
  - Fig. 4 shows that optimal `p` depends on sequence length: higher `p` helps at long sequences (more SRAM/IO pressure), but at short sequences it can create submatrices too small to hit tensor‑core peak (early bumps labeled “Matrices Too Small for Tensor Cores”).

- Step 5 — Two sparsity‑style extensions that drop compute by skipping blocks
  - Partial convolutions (Sec. 3.3): learn a shorter time‑domain filter by zeroing its tail, analogous to local attention. In the matrix view, this means certain FFT blocks don’t need to be computed.
  - Frequency‑sparse convolutions (Sec. 3.3; Appendix A.4): zero selected frequency bins of `k_f`, letting the kernel skip corresponding sub‑blocks of the matrix multiplies. Appendix A.4 explains which decomposed matmuls can be bypassed when you sparsify along each of the four reshaped dimensions and lists specific block patterns used (Table 10).

- Implementation notes (Appendix A.2, A.3, A.5)
  - Uses CUDA WMMA tensor cores with 16×16×16 tiles; reuses accumulator registers as subsequent inputs whenever a transpose isn’t needed (Algorithm 2), avoiding SRAM round‑trips.
  - Double‑buffering across memory levels and vectorized fp16/bf16 ops reduce stalls and raise throughput.
  - Provides 3‑ and 4‑way decomposition kernels (Algorithms 3–4) and currently supports A100/H100 (Appendix A.5).

- Putting it all together: Algorithm 1 (Sec. 3.1) shows an order‑2 fused pipeline for a single layer:
  - Load `F`, `F^{-1}`, twiddle vectors, and `k_f`.
  - Reshape input to square blocks, do column FFT, twiddle multiply, row FFT, elementwise multiply by `k_f`, then inverse steps, writing back transposed outputs.

## 4. Key Insights and Innovations
- Turn FFT into tensor‑core GEMMs in an IO‑aware way (fundamental)
  - Prior FFTs are IO‑bound and don’t engage tensor cores. FlashFFTConv’s Monarch factorization and “broadcast over sequence” dramatically change the execution pattern so FFTs become tensor‑core GEMMs with on‑chip transposes (Fig. 3; Sec. 3.1), unlocking up to PFLOP‑class compute (Sec. 2.2).
- A compute‑plus‑IO cost model that selects decomposition order (methodological)
  - Equation (2) formalizes the tradeoff between compute and memory traffic, with a practical switch `γ(N_i)` that reflects when matmuls are big enough for tensor cores. Fig. 4 visualizes why `p=2,3,4` differ across N and where SRAM limits kick in.
- Domain‑specific FFT shortcuts for ML (incremental but impactful)
  - Real‑to‑real FFT halving (Appendix A.1) and exploiting causal zero‑padding (Sec. 3.1) reduce work at minimal complexity cost; these are classic DSP ideas applied carefully to the fused kernel.
- Sparse analogues for convolutions that map to block skipping (conceptual + practical)
  - Partial and frequency‑sparse convolutions (Sec. 3.3; Appendix A.4) are easy to implement in this matrix view (“skip blocks”), directly yielding extra speedups without changing the model interface. They also enable new capabilities (longer contexts via partial filters; Table 8).

## 5. Experimental Analysis
- Evaluation setup
  - Microbenchmarks: compare FlashFFTConv with PyTorch FFT convolution and a “fusion‑only no tensor‑cores” ablation that effectively mirrors NVIDIA cuFFTdx (Table 3). Measure forward latency and memory, across sequence lengths 256 to 4M; batch=64, hidden=768; H100‑SXM for timing (Tables 3–4; full sweeps in Tables 11–15).
  - End‑to‑end models across modalities (Table 5): M2‑BERT‑base (seqlen 128), Hyena‑s‑4K, long‑convs for Path‑X (16K), SaShiMi for raw audio (64K), and HyenaDNA‑1M (genomics). Architecture‑specific fusion (e.g., multiplicative gating) is included where relevant.
  - Transformer comparison: Hyena‑2.7B vs GPT‑2.7B with FlashAttention‑v2 on A100 at seqlen 2K/8K/16K (Table 6). Report throughput and end‑to‑end FLOP utilization.
  - Quality under fixed compute: Train equal compute budgets but faster throughput lets FlashFFTConv see more data/steps (Table 1).
  - Sparse variants: Partial convolutions (Table 7, Table 8) and frequency‑sparse convolutions (Table 9).

- Main results
  - Convolution speed and memory
    - Forward convolution speedup up to 6.54× over PyTorch (Table 3; 1K tokens). For gated convolutions, up to 7.93× (Table 4; 1K). Speedup remains >1× up to 4M length (Tables 3–4, 11–14). The “fusion‑only” ablation confirms tensor cores are crucial at longer sequences—without them, SRAM runs out by 32K (Table 3, “Fusion‑Only/cuFFTdx”).
    - Memory footprint drops by 5–8× at short/medium lengths and ~2.6–2.8× at million‑token scales (Tables 3–4; full numbers in Tables 16–17), driven by recomputation and fusion.
  - End‑to‑end throughput
    - Speedups across tasks (Table 5): 1.3× (SaShiMi, where convolution isn’t dominant) to 4.4× (HyenaDNA, where PyTorch forced batch=1 but FlashFFTConv allows batch=4).
  - Transformer comparison
    - At 2K/8K/16K context, Hyena‑2.7B with FlashFFTConv has higher throughput than GPT‑2.7B with FlashAttention‑v2 despite slightly lower FLOP utilization (62% vs 66–79%). Reported speedups are 1.1×, 1.3×, and 1.5× as sequence grows (Table 6). This stems from lower algorithmic FLOPs of convolution relative to attention (Sec. 4.2).
  - Quality under fixed compute budget
    - Faster training translates to higher quality given the same compute budget (Table 1):
      - M2‑BERT‑base: +3.3 GLUE points (77.6 → 80.9).
      - Hyena‑s: perplexity 13.4 → 11.1 on The Pile.
    - The magnitude is comparable to doubling parameter count for the baselines (Appendix B.2, Table 18).
  - Longer‑sequence capability
    - High‑resolution vision (Path‑512; 256K tokens): first successful solution at 96.1% accuracy; PyTorch runs out of memory (Table 2). Path‑X (16K) matches prior 96.9%.
    - Genomics: partial convolutions extend HyenaDNA to 4M context with negligible perplexity change (2.91→2.90; Table 8). A t‑SNE of embeddings for long genes including Dystrophin (2.3M bp) showcases the new capability (Appendix B.3, Fig. 5).
  - Sparse variants
    - Partial convolutions: for Hyena‑s‑8K, kernel length can be reduced to 2K with no perplexity degradation while dropping memory from 32.5 GB to 8.4 GB (Table 7).
    - Frequency‑sparse convolutions: up to 79% of frequency bins can be zeroed without loss, improving convolution speed by up to 1.4×; at 75% sparsity, DNA perplexity slightly improves (-0.01), suggesting denoising of high‑frequency components (Table 9; patterns in Appendix A.4).

- Are the experiments convincing?
  - The microbenchmarks are thorough (powers of two up to 4M; Tables 11–15) and include ablations isolating the role of tensor cores and fusion (Table 3).
  - End‑to‑end results span five modalities/tasks and include a strong baseline (FlashAttention‑v2) with matched parameter counts (Table 6).
  - The fixed‑compute comparison clearly ties efficiency to quality (Table 1), though it relies on “more tokens/steps within the same compute” rather than identical training schedules (details in Appendix C.2).

- Conditions and trade‑offs observed
  - Speedup is largest when FFT IO dominates and matmul sizes hit tensor cores; it tapers at the longest lengths where SRAM/HBM traffic reappears (Fig. 4; Tables 3–4).
  - Tasks where convolution isn’t the main cost (e.g., SaShiMi with SSM components) see smaller end‑to‑end gains (Table 5).

## 6. Limitations and Trade-offs
- Hardware specialization
  - Current kernels target A100/H100 tensor cores with 16×16×16 tiles and use CUDA WMMA APIs (Appendix A.2). Older GPUs (e.g., V100) aren’t supported due to different tensor‑core sizes (Appendix A.5).
- Kernel specialization and engineering overhead
  - The implementation compiles kernels per sequence length and decomposition, with tuned tiling/unrolling (Appendix A.2–A.3). This yields peak performance but increases engineering complexity and may limit portability.
- IO vs compute trade‑offs persist at extreme scales
  - Even with fusion, very long sequences still pay extra IO at outer stages (e.g., SRAM↔HBM for 4‑way decompositions), which reduces the relative speedup (Fig. 4; long‑N rows in Tables 3–4).
- Assumptions about signal properties
  - Gains rely on real‑valued inputs/filters (Appendix A.1) and on common causal padding patterns (Sec. 3.1). Complex‑valued convolutions or uncommon padding schemes may reduce benefits.
- Scope limitations
  - The paper focuses on 1D long convolutions and single‑GPU kernels. Multi‑GPU FFT plans, 2D/3D FFTs, and non‑GPU accelerators are outside current support (Appendix A.5), though the ideas may transfer.
- Sparse variants require pattern design
  - Frequency‑sparse patterns were manually crafted (Appendix A.4, Table 10). Automated or learned sparsity schedules are not explored and could affect robustness across domains.

## 7. Implications and Future Directions
- Field impact
  - FlashFFTConv makes long‑convolution models practical at scales previously dominated by optimized Transformers. It narrows the systems gap—achieving end‑to‑end utilization close to FlashAttention‑v2 (Table 6)—while retaining the algorithmic benefits of convolutions (lower FLOPs, stability).
- Research enabled
  - Longer‑context modeling: language models beyond 32K–64K tokens; genomics at multi‑million base‑pair windows (Table 8); high‑resolution vision benchmarks such as Path‑512 (Table 2).
  - New “sparsity for convolutions”: partial/frequency‑sparse designs suggest learned block‑skip schedules, dynamic sparsity, and compression techniques that are easy to implement in the matrix view (Sec. 3.3; Appendix A.4).
  - Hardware‑algorithm co‑design: the cost model (Eq. (2); Fig. 4) invites adaptive selection of `p` per layer, sequence length, or device, and generalizes to other fast transforms (Walsh‑Hadamard, DCT) and to non‑GPU accelerators.
- Practical applications
  - Faster training/inference for long‑context LLM alternatives (Hyena‑style models), efficient raw‑audio generation and ASR with long receptive fields, and genomics models that embed entire long genes at single‑base resolution.
- Promising next steps
  - Multi‑GPU and pipeline‑parallel FFT fusion; automatic `p` selection using runtime telemetry; integration into compiler stacks (e.g., PyTorch 2.0) for automatic fusion; learning frequency‑sparse patterns end‑to‑end; extending to 2D/3D FFTs for vision and scientific computing.

> Key references inside the paper: Equation (1) (FFT convolution), Figure 2 (Monarch decomposition), Figure 3 (broadcast‑over‑sequence and SRAM transposes), Figure 4 (cost model trade‑offs), Algorithm 1/3/4 (order‑2/3/4 fused kernels), Tables 3–4 (microbench speedups), Table 5 (end‑to‑end throughput), Table 6 (comparison with FlashAttention‑v2), Table 1 (quality under fixed compute), Table 2 (Path‑512 success), Tables 7–9 (partial and frequency‑sparse), Appendix A.1–A.5 (FFT details, CUDA, sparsity patterns, hardware).
