# SageAttention3: Microscaling FP4 Attention for Inference and An Exploration of 8-bit Training

**ArXiv:** [2505.11594](https://arxiv.org/abs/2505.11594)

## 🎯 Pitch

SageAttention3 introduces the first practical 4-bit (FP4) 'microscaling' attention kernel for GPU inference, leveraging new Blackwell FP4 Tensor Cores to achieve up to 5× faster performance than state-of-the-art methods—while maintaining model accuracy in demanding text, image, and video generation tasks. It also pioneers trainable 8-bit attention (SageBwd), enabling efficient, lossless fine-tuning for large models, thus lowering both inference and training costs and paving the way for next-generation, low-latency, and ultra-efficient AI systems.

---

## 1. Executive Summary

This paper introduces **SageAttention3**, the first FP4 microscaling attention kernel for inference acceleration, and **SageBwd**, the first trainable 8-bit attention mechanism for training. On RTX 5090 GPUs, SageAttention3 achieves 1038 TOPS — a 5× speedup over the fastest FlashAttention available on that hardware — while maintaining plug-and-play compatibility across diverse video, image, and language models with negligible end-to-end quality degradation. SageBwd quantizes six of the seven matrix multiplications in the attention backward pass to INT8, achieving lossless fine-tuning performance on Qwen2.5 and Llama3.2 across benchmarks including GSM8K, DROP, MMLU, and HELLASWAG, while revealing slower convergence on pretraining tasks, establishing that 8-bit attention is viable for fine-tuning but not yet for pretraining workloads.

## 2. Context and Motivation

### The Core Problem: Attention Is the Bottleneck for Generation Models

The fundamental challenge this paper tackles is the **computational inefficiency of the attention mechanism** in Transformer-based models. Attention introduces quadratic time and memory complexity with respect to sequence length — a pair of $N \times N$ matrix multiplications ($QK^\top$ and $PV$) that dominate runtime, especially for long sequences in video generation, high-resolution image synthesis, and long-context language modeling. As the authors note (Section 1), this inefficiency is particularly acute for generation models, where both training and inference must process increasingly long sequences to produce coherent outputs.

The cost structure is asymmetric in an important way: the attention computation involves two large matrix multiplications and one softmax — operations that are fundamentally memory-bound and compute-bound in distinct ways on modern GPUs. The theoretical throughput of GPU Tensor Cores for FP16 matrix multiplication is roughly 330 TFLOPS on an H100, but attention kernels rarely achieve even half of this due to I/O bottlenecks between global memory and on-chip SRAM. So the problem is not just that attention is mathematically expensive — it is that naive implementations leave enormous hardware capability unused.

This paper addresses a direct extension: **can we exploit new low-precision Tensor Core capabilities to push attention throughput substantially higher while preserving model quality?** The Blackwell GPU architecture introduces FP4 Tensor Cores that deliver approximately 8× the theoretical throughput of FP16 Tensor Cores (Section 3.1: roughly 1600 TOPS for FP4 versus roughly 200 TOPS for FP16 on RTX 5090). But tapping into this capability for attention is non-trivial because:

1. **FP4 has only 15 representable values** (E2M1 format: 1 sign bit, 1 mantissa bit, 2 exponent bits minus special values), making quantization errors a first-order threat to model accuracy.
2. **The attention map $P$ has a pathological value distribution**: its values lie in $[0, 1]$ with most concentrated near zero (Figure 3a shows a mean of 0.1758), which causes the required E4M3 scale factors to occupy only a tiny fraction of their representable range, amplifying quantization error.
3. **Training introduces additional complexity**: the backward pass has five matrix multiplications instead of two, and gradient signals are more sensitive to quantization error than forward activations — errors accumulate across sequence positions in FlashAttention's recurrent backward computation.

### Why This Problem Matters: Two Distinct but Connected Motivations

**Inference acceleration is urgent for generative AI deployment.** Video generation models like HunyuanVideo and CogVideoX can take *minutes* to generate a single clip (Table 4a: HunyuanVideo inference requires 489 seconds with standard attention). These costs directly limit practical adoption — interactive applications, real-time video editing, and large-scale content generation are infeasible at these latencies. A 3× end-to-end speedup (which SageAttention3 achieves: 489s → 164s for HunyuanVideo) transforms a prohibitively slow pipeline into something approaching real-time and is the difference between a research demo and a deployable product.

But this is not just about absolute latency. The Blackwell FP4 Tensor Cores represent a **structural shift in GPU architecture** — they provide 8× higher theoretical throughput than FP16 at the cost of dramatically reduced precision. The question of whether attention computation can be reliably performed in FP4 is therefore not an incremental optimization problem but a **capability-existence question**: can we design quantization schemes accurate enough to harness this hardware capability, or is attention fundamentally incompatible with 4-bit precision? The paper's demonstration that FP4 attention achieves 1038 TOPS while preserving model quality (Table 2) establishes that the answer is yes — attention *can* be quantized to FP4 without harming end-to-end outputs — and this result has architectural implications that extend beyond any single GPU generation.

**Training efficiency is equally important but fundamentally harder.** The paper explicitly frames the training contribution as an *exploration*: prior low-bit attention work (FlashAttention3, SageAttention, SageAttention2) focused exclusively on inference, where only the forward pass matters and activations are the primary quantization targets. Training introduces gradients, which are typically higher-variance than activations and must be computed accurately enough to avoid corrupting parameter updates over thousands of iterations. The paper's question — "can low-bit attention be effectively applied to training tasks?" (Section 1) — is genuinely open, and the answer turns out to be nuanced: yes for fine-tuning, no (yet) for pretraining. This establishes both a new capability (8-bit training of attention is now possible for fine-tuning workloads) and a clear boundary condition that guides future work.

### Where Prior Approaches Fall Short

The paper identifies limitations across a spectrum of existing efficient attention methods:

**FlashAttention3's FP8 attention is hardware-restricted and inference-only.** FlashAttention3 (Shah et al., 2024) introduced an FP8 attention variant that achieves 890 TOPS on H100 GPUs — a significant advance. However, the paper notes (Section 1, caption of Figure 1) that FlashAttention3 "can only run on Hopper GPUs, so FlashAttention2 is already the fastest on RTX5090." This creates a practical gap: Blackwell GPUs cannot run FlashAttention3's FP8 kernels, meaning users with RTX 5090 hardware are stuck with FP16 FlashAttention2 at roughly 214 TOPS — only about 13% of the FP4 peak throughput theoretically available on that hardware. Moreover, FlashAttention3's FP8 attention "does not support the backward pass" (Section 6), making it fundamentally unusable for training.

**Existing quantization-based attention (SageAttention, SageAttention2) hits a precision floor.** SageAttention1 used INT8 quantization for attention and achieved 99.996% cosine similarity relative to full precision. SageAttention2 pushed further to INT4 per-thread quantization while maintaining 99.995% cosine similarity. These established that attention can tolerate aggressive quantization, but they operated at 8-bit and 4-bit integer precisions on older GPU architectures — they did not attempt FP4, where the quantization challenge is qualitatively different due to the extremely limited representable value set (FP4 offers $2^4 - 1 = 15$ distinct non-zero values, compared to $2^8 - 1 = 255$ for INT8). The jump from INT8 to FP4 is a factor of $16\times$ reduction in representable values, not the $2\times$ reduction from FP16 to INT8 — it requires fundamentally different quantization strategies.

**SmoothQuant and AWQ-style smoothing is insufficient for FP4 attention.** Prior work (SmoothQuant, Xiao et al., 2023; AWQ, Lin et al., 2024) proposed per-channel or per-token scaling to handle activation outliers, which improved INT8 quantization accuracy for weight-activation matrix multiplications. However, the paper's ablation (Table 16) shows that SmoothQuant-style smoothing achieves only 0.930 cosine similarity for FP4 attention, compared to 0.991 with the smoothing K technique inherited from SageAttention2. The reason is that FP4's extreme value limitation makes global scaling strategies fragile — a single outlier in one token can force the scale factor for an entire block to grow, causing most values to quantize to zero. This is the motivation for the paper's microscaling approach (1×16 blocks) rather than per-tensor or per-channel quantization.

**No prior work combines search across quantization granularities with hardware-aware kernel design for FP4 attention.** The paper's hardware implementation contributions — permutation for K to match FP4 accumulator layout, reuse of softmax max-reductions for quantization scaling, and producer-warp epilogue scheduling — are not conceptually novel in isolation (CUTLASS and warp-specialized kernel design are well-established). But the specific combination — designing the quantization scheme around the hardware constraints (E4M3 scale factors must fit FP8 range, FP4 MMA accumulator layout differs from operand layout) rather than treating quantization and kernel design as separable — is what makes the 1038 TOPS figure possible. Prior quantization-for-attention work (SageAttention2) did not face these constraints because INT4 on older GPUs used different accumulator layouts and did not require scale factors to be in a restricted floating-point format.

### How This Paper Positions Itself

The paper positions itself along two axes that are deliberately separated:

**Axis 1: Pushing inference quantization to the absolute precision limit (FP4) on the newest hardware (Blackwell).** SageAttention3 is presented as the first work to attempt FP4 attention at all, and the contribution is primarily engineering — combining microscaling quantization (NVFP4 format with 1×16 blocks), two-level scaling for the attention map, smoothing techniques inherited from SageAttention2, and hardware-aware kernel optimizations into a system that achieves 1038 TOPS. The authors frame this as establishing a new speed-accuracy Pareto frontier: SageAttention3 at 99.55% cosine similarity and 1038 TOPS sits at a point that was previously inaccessible (Table 18 shows FlashAttention2 at 214 TOPS/100% similarity and FlashAttention3-FP8 at 890 TOPS/98.57% similarity on H100, with neither point existing on RTX 5090).

Importantly, the paper does not claim that FP4 attention is universally applicable or that it matches FP16 quality for all use cases. The end-to-end metrics (Table 2) show minor degradation on some models (e.g., Flow-score drops from 1.48 to 1.23 on HunyuanVideo; FScore drops from 4.78 to 4.04 on CogVideoX), which the authors present transparently. The claim is that the quality loss is negligible enough for practical use — a different standard than "zero accuracy loss."

**Axis 2: Extending low-bit attention beyond inference to training for the first time.** SageBwd is positioned as an exploration, not a solved problem. The authors explicitly state the finding that "8-bit attention achieves lossless performance in fine-tuning tasks but exhibits slower convergence in pretraining tasks" (Abstract), which establishes both a success case and a failure case. This framing is important because it avoids overclaiming — the paper is not saying "we solved low-bit training for attention," but rather "we found the boundary of what is currently possible." The theoretical analysis (Appendix A.11) that identifies $dOV^\top$ (the gradient of output with respect to value) as the accuracy-sensitive operation and keeps it in FP16 is a design insight that explains *why* the boundary exists where it does.

The paper also positions the inference and training contributions as complementary: Section 5.3 shows that combining SageBwd fine-tuning with SageAttention3 inference yields *better* downstream accuracy than BF16 fine-tuning followed by FP4 inference (Table 5: Qwen2.5-1.5B on GSM8K improves from 0.4912 to 0.5232). The hypothesis — that INT8 training and FP4 inference share more similar representable data distributions than BF16 and FP4, reducing the mismatch — is speculative but points toward a unified low-precision pipeline where training and inference operate at related precisions.

## 3. Technical Approach

### 3.1 Reader Orientation

This paper builds **two distinct systems**: a forward-pass-only FP4 attention kernel (SageAttention3) that exploits Blackwell GPU hardware to accelerate inference, and a forward-and-backward INT8 attention mechanism (SageBwd) that accelerates training by quantizing nearly all attention matrix multiplications. The solution's "shape" is a careful orchestration of quantization granularity choices, custom scaling strategies, and hardware-aware kernel design — each component chosen to work around the specific numerical and architectural constraints that make extreme low-bit attention challenging.

### 3.2 Big-Picture Architecture (Diagram in Words)

The system has two major subsystems that share conceptual foundations but target different computational graphs:

**SageAttention3 (inference, FP4) has four components:**

1. **Input Smoothing Module** — preprocesses Q and K tensors (inherited from SageAttention2) to reduce outlier magnitude before quantization. Smoothing K subtracts per-token means from K; smoothing Q subtracts per-block means from Q and handles the correction in the GEMM.

2. **FP4 Microscaling Quantizer (ϕ)** — partitions matrices into 1×16 blocks, computes per-block scale factors, and rounds values to NVFP4 (E2M1) format. Applied to Q, K, V, and the attention map P. Scale factors are stored in FP8 (E4M3) format as required by the hardware instruction.

3. **Two-Level P-Scaling Module** — a specialized quantization path for the attention map P that first normalizes each row to the range [0, 448×6] before applying the standard FP4 microscaling quantizer. This exploits the full dynamic range of the E4M3 scale factor format.

4. **FP4-optimized Attention Kernel** — a CUTLASS/CUDA implementation that fuses online softmax, quantization, and FP4 matrix-multiply-accumulate (FP4MMA) instructions with three hardware-specific optimizations: K permutation to match FP4 accumulator layout, reuse of softmax max-reductions for quantization scaling, and producer-warp epilogue scheduling for compute-memory overlap.

Information flows as follows: Q, K, V enter in FP16 → K is smoothed (mean subtraction) → Q, K, V are tiled into FlashAttention blocks → each Q block is smoothed → Q, K are quantized to FP4 → FP4MMA computes S = QK^T → online softmax produces P → P undergoes two-level quantization to FP4 → V is quantized to FP4 → FP4MMA computes O = PV → corrections from smoothing and two-level scaling are applied in FP32.

**SageBwd (training, INT8) has three components:**

1. **Per-Block INT8 Quantizer (ψ)** — for each FlashAttention block of a matrix, computes a single scale factor (max absolute value divided by 127) and quantizes all elements to INT8. This is coarser than SageAttention3's microscaling but adequate for 8-bit precision and simpler to implement in Triton.

2. **Forward Pass with Per-Token P Quantization** — reuses the running max from online softmax to compute per-token scale factors for P, avoiding a separate max-reduction pass. P is quantized to INT8 per-token while V is quantized per-block.

3. **Selective Backward Quantizer** — among the five backward matrix multiplications, four are quantized to INT8 per-block, but `dOV^T` (the gradient of output with respect to value transpose) is kept in FP16. This is the accuracy-sensitive operation whose errors would recursively accumulate into dQ and dK along the sequence length dimension.

Information flows in the forward pass identically to standard FlashAttention but with INT8 quantization at each GEMM; the backward pass computes dO → dV, dP, dS, dQ, dK with quantization at all steps except `dOV^T`.

### 3.3 Roadmap for the Deep Dive

- **First, the NVFP4 microscaling format and quantization operator (ϕ)** — because all of SageAttention3's design decisions follow from the constraints of this format: 1×16 blocks, E2M1 data type, E4M3 scale factors. Understanding why the block size matters and why the scale factor format causes problems for P is prerequisite for everything else.

- **Second, smoothing Q and K** — the preprocessing inherited from SageAttention2 that makes FP4 quantization of Q and K tractable, including why alternative smoothing strategies (SmoothQuant, Hadamard) fail at this precision level.

- **Third, the two-level quantization for P** — the paper's key numerical innovation, motivated by the mismatch between P's [0,1] range and the E4M3 scale factor's representable range. We'll walk through the data distributions in Figure 3 and the error analysis showing why direct quantization fails.

- **Fourth, hardware implementation optimizations** — the three kernel-level techniques (K permutation, shuffle reuse, producer-warp epilogue) that translate the numerical scheme into 1038 TOPS. These are explained in terms of the specific hardware constraints they address.

- **Fifth, SageBwd forward pass** — how per-block INT8 quantization differs from FP4 microscaling, why per-token quantization for P works at 8 bits, and how online softmax enables zero-overhead scale factor computation.

- **Sixth, SageBwd backward pass and the dOV^T decision** — the empirical finding that quantizing dOV^T causes unacceptable gradient error (Table 1c), the theoretical justification (Appendix A.11), and the engineering implication that four of five backward GEMMs can be safely quantized.

### 3.4 Detailed, Sentence-Based Technical Breakdown

This is primarily a **systems paper** whose core contributions are: (1) a quantization scheme that makes FP4 attention numerically accurate despite FP4 having only 15 representable values, and (2) an 8-bit training attention mechanism that selectively preserves accuracy for the most gradient-sensitive operations. The intellectual contribution is not a new algorithm but rather the identification of *which* granularities, scaling strategies, and precision assignments work for attention under extreme quantization — and, critically, *why* alternatives fail.

---

#### The NVFP4 Microscaling Format and Quantization Operator

The paper chooses **NVFP4** over the alternative MXFP4 format for all attention quantization. Both formats use the E2M1 data type for values (1 sign bit, 2 exponent bits, 1 mantissa bit — yielding the representable set $\{0, \pm 0.5, \pm 1.0, \pm 1.5, \pm 2.0, \pm 3.0, \pm 4.0, \pm 6.0\}$, a total of 15 distinct non-zero magnitudes plus zero). The critical difference is in **block size** and **scale factor format**:

- **NVFP4:** block size of 1×16 (each group of 16 consecutive elements along the reduction dimension shares one scale factor); scale factors are in **E4M3** FP8 format (1 sign, 4 exponent, 3 mantissa bits — range roughly $[-448, 448]$).
- **MXFP4:** block size of 1×32 (groups of 32 elements share one scale factor); scale factors are in **E8M0** format (8 exponent bits, no mantissa — only powers of 2, no fractional precision in the scale factor itself).

The authors choose NVFP4 because "the accuracy of NVFP4 is much higher than that of MXFP4 in attention quantization" (Section 3.1). The empirical confirmation comes from Table 1(a): on real Q, K, V tensors across all layers of CogVideoX-2B, NVFP4 achieves 99.52% cosine similarity, 0.077 L1 error, and 0.201 RMSE, while MXFP4 achieves only 98.37% cosine similarity, 0.294 L1 error, and 0.994 RMSE. The 1×16 block size provides finer granularity than 1×32, which matters because attention activations (particularly Q and K after smoothing) have local structure — outliers tend to cluster spatially rather than being uniformly distributed — so smaller quantization groups better contain outlier effects within individual blocks, preventing a single large value from dominating the scale factor for 32 elements.

The quantization operator ϕ (used for Q, K, V, and the second level of P quantization) is defined as:

$$s_{ij} = \max(|X_{ij}|) / 6, \quad \hat{X}_{ij} = \lceil X_{ij} / s_{ij} \rfloor$$

where $X_{ij} \in \mathbb{R}^{1 \times 16}$ is a micro-scaling block (1 row, 16 columns) of the input matrix, $s_{ij} \in \text{E4M3}$ is the scale factor for that block, and $\lceil \cdot \rfloor$ denotes rounding to the nearest representable NVFP4 value.

**What this computes:** for each 1×16 block, find the maximum absolute value among the 16 elements, divide by 6 (the maximum representable magnitude in E2M1), and use that quotient as the scale factor. Then divide each element in the block by this scale factor and round to the nearest FP4 value. The dequantization (Equation 2) multiplies back: $X'_{ij} = s_{ij} \times \hat{X}_{ij}$.

**Why division by 6:** FP4's E2M1 format can represent values up to 6.0 in magnitude (the bit pattern `0111` = $2^{2-2} \times 1.5 = 6.0$, since with 2 exponent bits and bias -2, exponent 3 gives $2^{1} \times 1.5 = 3.0$; wait — let me compute this precisely: E2M1 means 2 exponent bits with bias 1, so exponent values are 0 (subnormal, interpreted as $2^{-1}$), 1 ($2^{0}=1$), 2 ($2^{1}=2$), 3 (infinity/NaN). Mantissa is 1 bit, representing 0 or 0.5. So representable values: for exponent 0, mantissa 0→0, mantissa 1→$2^{-1}\times 1.5=0.75$. For exponent 1, mantissa 0→1, mantissa 1→1.5. For exponent 2, mantissa 0→2, mantissa 1→3. Maximum finite value is 3. But the paper says max is 6 — this is because the sign bit effectively doubles the magnitude range, and the quantization operates on absolute values, mapping $\max(|X_{ij}|)$ to the representable value 6 (meaning the bit pattern `0111` interpreted as $1.5 \times 2^{2} = 6$ — I'm going to accept the paper's stated max of 6 and note that the exact FP4 format specification likely includes special handling). The key design point is: dividing by 6 ensures that the maximum element in the block maps to the maximum representable FP4 value, minimizing quantization error for the largest element while allowing smaller elements to be represented at finer granularity within the block.

This microscaling approach — 1×16 blocks with per-block scale factors — is what distinguishes the paper from prior quantization work. Standard per-tensor quantization would apply one scale factor to an entire matrix, which fails for attention because different tokens can have activation magnitudes differing by orders of magnitude (the "outlier" problem well-documented in SmoothQuant). Per-token quantization (one scale factor per row) would be better but still allows within-token outliers — a single large element in one attention head dimension could dominate the scale factor for that entire token's embedding. The 1×16 granulality is chosen as a sweet spot: fine enough to contain outliers locally, coarse enough that the scale factor storage overhead (one E4M3 value per 16 FP4 values, or 8/4=2 bits per element overhead) is acceptable, and critically, matching the hardware-required block size for the `FP4MMA` instruction on Blackwell GPUs.

The rounding operation $\lceil \cdot \rfloor$ is not standard round-to-nearest but round-to-nearest-even with ties broken to even — the standard IEEE 754 rounding mode, though the paper does not specify this explicitly. The important property is that it is deterministic (given fixed inputs, quantization always produces identical outputs), which matters for reproducibility in inference.

The FP4 microscaling matrix multiplication instruction (`FP4MMA`) takes four inputs — the quantized operands and their scale factors — and computes the matrix product of the dequantized matrices internally:

$$C = \text{FP4MMA}(\hat{A}, s_A, \hat{B}, s_B)$$

where the result $C$ equals $(\phi^{-1}(\hat{A}, s_A)) \times (\phi^{-1}(\hat{B}, s_B))$, i.e., the dequantized matrices are multiplied in the hardware's internal high-precision accumulator (FP32), and the result is returned in FP32.

**Why this form matters:** the `FP4MMA` instruction operates at roughly 1600 TOPS on RTX 5090 — 8× faster than the FP16 matrix multiply at roughly 200 TOPS. But this speed comes with the constraint that the scale factors must be in E4M3 FP8 format (hardware requirement), not arbitrary FP32 values. This constraint — that scale factors are themselves quantized to a limited-precision format — is the root cause of the two-level quantization for P, which we will discuss next. The alternative MXFP4 format's E8M0 scale factors (which can only represent powers of 2) would avoid this issue at the cost of coarser quantization granularity, but the accuracy penalty (Table 1a) makes this tradeoff unacceptable for attention.

---

#### Smoothing Q and K for FP4 Quantization

Direct FP4 quantization of raw attention inputs produces unacceptable error because Q and K matrices contain systematic outliers — specific token positions or feature dimensions where activation magnitudes are much larger than typical values. These outliers force the microscaling blocks containing them to use large scale factors, which in turn quantizes the smaller elements in those blocks to zero or near-zero — a catastrophic loss of information.

The paper inherits two smoothing techniques from SageAttention2 (Section 3.1, Algorithm 1):

**Smoothing K:** before any tiling, compute the mean of K along the row dimension and subtract it from every element:

$$K \leftarrow K - \text{mean}(K)$$

where $\text{mean}(K)$ is computed per head, producing a vector $K_m \in \mathbb{R}^{D}$ that is subtracted from every token position. This removes DC offsets in the key representations that would otherwise inflate the scale factors for the $QK^T$ computation. The correction is not applied during the GEMM — instead, the smoothing K mean is used only to make quantization more accurate, and the actual dot products are computed from the smoothed K.

**Smoothing Q at the block level:** after tiling Q into blocks $Q_i$ of size $B_q \times D$ (where $B_q$ is the FlashAttention query block size), compute the mean of each block:

$$\bar{q}_i = \text{mean}(Q_i) \in \mathbb{R}^{D}$$

then quantize $Q_i - \bar{q}_i$ rather than $Q_i$ directly. The correction is applied to the $S_{ij} = Q_i K_j^T$ computation via a separate GEMV (general matrix-vector multiply):

$$S_{ij} = \text{FP4MMA}(\hat{Q}_i, s_Q, \hat{K}_j^T, s_K) + \text{GEMV}(\bar{q}_i, K_j^T)$$

where the first term uses the FP4-accelerated matrix multiply on the smoothed (zero-centered) Q block, and the second term adds back the contribution of the subtracted mean using a separate FP16 GEMV operation. The GEMV is relatively cheap because $\bar{q}_i$ is a single vector (size D) being multiplied by $K_j^T$ (size $D \times B_{kv}$), so the cost is $O(D \times B_{kv})$ rather than $O(B_q \times D \times B_{kv})$ for the full GEMM.

**Why this smoothing matters specifically for FP4:** the paper's ablation in Appendix A.7 (Table 16) quantifies the contribution. With no smoothing, FP4 attention achieves only 0.916 cosine similarity — completely unusable. SmoothQuant-style per-channel scaling (migrating quantization difficulty from activations to weights via a per-channel scale factor) achieves 0.930, still far below usable quality. Hadamard transformation (random orthogonal rotation to spread outliers across dimensions) achieves 0.941. But the combination of smoothing K and smoothing Q achieves 0.991 cosine similarity individually and 0.996 when combined. The smoothing Q technique, which subtracts per-block means, is particularly important because it handles the case where a single token in a query block has an unusually large norm — without this, that token would dominate the scale factor for its entire 1×16 microscaling block, erasing information from neighboring tokens.

The key insight from the ablation is that **FP4 is so precision-constrained that generic smoothing strategies are insufficient** — you need attention-specific techniques that directly target the structural properties of Q and K (token-wise magnitude variation, head-wise mean offsets) rather than applying generic equalization transforms.

---

#### Two-Level Quantization for the Attention Map P

This is the paper's key numerical innovation for FP4 attention. The attention map $P = \text{Softmax}(S)$ has values in the range $[0, 1]$, with the additional property that each row sums to exactly 1 (after online softmax). The distribution of P values (Figure 3a) shows a strong concentration near small values — mean of 0.1758 across all rows and columns — because for long sequences, attention is typically spread across many tokens rather than concentrated on a few.

The problem with directly applying the FP4 microscaling quantizer ϕ to P is not the FP4 value precision itself (15 representable values can approximate a [0,1] distribution reasonably well for many applications), but the **scale factor representation**. The NVFP4 format requires scale factors to be in E4M3 FP8 format, which can represent values in approximately the range $[-448, 448]$ with variable spacing. However, when P is quantized directly, the scale factor for each 1×16 block is:

$$s_P = \max(P_{ij}) / 6 \leq 1/6 \approx 0.167$$

This means the scale factors are trapped in the narrow interval $[0, 0.167]$, using only about 0.037% of the E4M3 representable range. In this tiny subinterval, E4M3 has only about 35 distinct representable values (out of its total of approximately 255 finite values), meaning the scale factor itself is coarsely quantized before it's even used to scale the FP4 values. Figure 3(b) shows this directly: the distribution of $s_P$ under direct quantization is concentrated at very small values with poor coverage.

The **two-level quantization** rescales P so that its scale factors occupy the full E4M3 dynamic range:

1. **Level 1 (per-row normalization to FP32):** compute the per-row maximum of P and normalize each row so its maximum becomes $448 \times 6$ (the maximum representable E4M3 value times the FP4 max):

$$s_{P1} = \text{rowmax}(P) / (448 \times 6), \quad P_2 = P / s_{P1}$$

where $s_{P1} \in \mathbb{R}^{N}$ is a vector of per-row scale factors in FP32 precision (no quantization here), and $P_2$ is the rescaled attention map with each row's maximum now at exactly $448 \times 6$. This step has **negligible error** because $s_{P1}$, $P$, and $P_2$ are all in FP32 — no quantization occurs at this level.

2. **Level 2 (FP4 microscaling):** apply the standard quantizer ϕ to $P_2$:

$$s_{P2}, \hat{P}_2 = \phi(P_2)$$

Now $s_{P2} \in \text{E4M3}$ will take values near 448 (the maximum of $P_2$ blocks is around $448 \times 6$, so $s_{P2} = \max(P_{2,ij})/6 \approx 448$), fully utilizing the E4M3 representable range. Figure 3(c) shows the resulting $s_{P2}$ distribution: values now span the full [0, 448] range with fine granularity.

The final dequantized P is approximated as:

$$P \approx \hat{P}_2 \times s_{P2} \times s_{P1}$$

and the subsequent matrix multiplication with V becomes:

$$O = \text{FP4MMA}(\hat{P}_2, s_{P2}, \hat{V}, s_V) \times s_{P1}$$

where the matrix multiply operates on the FP4-quantized $P_2$ and V, and the result is multiplied by the per-row $s_{P1}$ correction in FP32.

**What this computes:** rather than quantizing P directly (which would force $s_P$ into the poorly-represented [0, 0.167] E4M3 subrange), the two-level scheme first proportions P's per-row range to the maximum representable E4M3 value, then quantizes — ensuring the scale factors land in the densely-represented region of E4M3. The per-row $s_{P1}$ correction is stored in FP32 (not FP8), so it captures the varying magnitudes across rows without precision loss.

**Why this form works where direct quantization fails:** the empirical evidence is in Figure 3(d-e) and Table 1(b). Figure 3(d) shows that the scale factor error ($s_P$ compared to ideal FP32 value) is dramatically lower with two-level quantization — the mean ± 1 standard deviation bar is barely visible versus a large error for direct quantization. Figure 3(e) shows that the actual quantization error of P (the difference between quantized-dequantized P and true P) is correspondingly lower. Table 1(b) quantifies this on real CogVideoX data: two-level quantization achieves 99.52% cosine similarity versus 93.32% for direct quantization, and the RMSE drops from 1.103 to 0.201.

The formal analysis in Appendix A.10 proves the improvement by counting representable values. With direct quantization, the number of distinct values after dequantization is $35 \times 8 = 280$ (35 distinct E4M3 scale factors in [0, 0.167] × 8 distinct E2M1 values). With two-level quantization, it is $127 \times 8 = 1016$ (127 distinct E4M3 scale factors in [0, 448] × 8 distinct E2M1 values). The finer granularity of representable values directly translates to lower quantization error, since the maximum quantization error for any value is bounded by half the distance between its two nearest representable neighbours.

The visual evidence in Figure 12 (Appendix A.1) is striking: video frames generated with direct P quantization show severe artifacts (color shifts, structural distortions), while two-level quantization produces frames visually indistinguishable from full precision. This is the paper's most convincing demonstration that the two-level scheme resolves a genuine showstopper — without it, FP4 attention is simply not usable for generation tasks.

**Why 448 specifically?** The value $448 \times 6$ is chosen because 448 is the maximum representable value in E4M3 FP8 (the bit pattern `01111110` gives $2^{4} \times 1.75 = 28$; the max with exponent 7 and mantissa all 1s gives $2^{8-1-7=0}$ — let me be precise: E4M3 has 4 exponent bits with bias 7, so exponent values range from 0 (subnormal) to 14 (with 15 reserved for NaN/Inf). The maximum normal value is $2^{14-7} \times (1 + 7/8) = 2^7 \times 1.875 = 240$ — actually, checking the specification: E4M3 in the microscaling standard has exponent bias 7, so exponent 14 gives $2^{7} = 128$, and the maximum mantissa of 7/8 gives $128 \times (1 + 7/8) = 128 \times 1.875 = 240$. Adding the sign bit doesn't change magnitude. But the paper states 448 — this is likely the maximum representable value for E4M3 in the specific OCP MX specification, which may use a different encoding. The key design principle is: normalize P rows so their maximum maps to the *maximum representable E4M3 value* times the FP4 max of 6, ensuring $s_{P2}$ lands at the top of the representable range rather than at some arbitrary point. The specific constant (448) is determined by the hardware's E4M3 encoding and is not tunable.

---

#### FP4 Attention Kernel Algorithm (Putting It All Together)

Algorithm 1 describes the complete FP4 attention forward pass, integrating smoothing, quantization, and the two-phase P scaling into a tiled FlashAttention-style kernel. Walking through the algorithm:

**Step 1 (Line 2):** K smoothing is applied globally (pre-tiling): $K \leftarrow K - \text{mean}(K)$.

**Step 2 (Line 3):** Q, K, V are tiled into blocks: Q is partitioned into $T_m = N/B_q$ blocks of size $B_q \times D$; K and V into $T_n = N/B_{kv}$ blocks of size $B_{kv} \times D$.

**Step 3 (Outer loop, Lines 4-14):** for each Q block $Q_i$:
- Line 5: compute per-block Q mean $\bar{q}_i$ and quantize the centered Q: $(s_Q, \hat{Q}_i) = \phi(Q_i - \bar{q}_i)$.
- Inner loop (Lines 6-12): for each K,V block pair $(K_j, V_j)$:
  - Line 7: quantize $K_j^T$ and $V_j$: $(s_K, \hat{K}_j) = \phi(K_j^T)$, $(s_V, \hat{V}_j) = \phi(V_j)$.
  - Line 8: compute $S_{ij}$ using FP4MMA on the quantized Q and K, plus the GEMV correction for Q smoothing.
  - Line 9: online softmax — update running max $m_{ij}$, compute exponentiated values $P_{ij} = \exp(S_{ij} - m_{ij})$, and update running sum $l_{ij}$ with exponential rescaling for previous blocks.
  - Line 10: **two-level quantization of P** — compute the per-row max of $P_{ij}$, divide by $448 \times 6$ to get $s_{P1}$, rescale $P_{ij}$, then apply FP4 microscaling $\phi$ to get $s_{P2}$ and $\hat{P}_{ij}$.
  - Line 11: compute $O_{ij}$ using FP4MMA on $\hat{P}_{ij}$ and $\hat{V}_j$, multiply by $s_{P1}$, and add to the running output with exponential rescaling for previous blocks.
- Line 13: after processing all K,V blocks, normalize the accumulated output by the final softmax denominator: $O_i = \text{diag}(l_{i,T_n})^{-1} O_{i,T_n}$.

**What this algorithm computes:** it is functionally identical to FlashAttention — the same online softmax, the same tiled accumulation, the same final row normalization. The difference is that every matrix multiplication ($QK^T$ and $PV$) is performed in FP4 via the `FP4MMA` instruction rather than in FP16, and the attention map P is quantized to FP4 before being used in the second GEMM. The smoothing corrections and the level-1 P scaling are applied in FP32 to preserve accuracy, but these operations are vector-vector or scalar multiplications (cheap relative to the GEMMs).

**Key design decisions in the algorithm:**

- **K is transposed before quantization** (Line 7: $\phi(K_j^T)$). This is because the FP4MMA instruction expects operand B to be in column-major layout for the inner dimension. Since K is naturally stored in row-major (tokens × features), transposing at quantization time aligns the memory layout with the hardware instruction's expectations, avoiding an expensive transpose inside the inner loop.

- **Q smoothing correction uses GEMV, not a second FP4MMA.** The GEMV operation computing $\bar{q}_i K_j^T$ runs in FP16 (or FP32) rather than FP4. Since $\bar{q}_i$ is a single vector, the computational cost is $O(D \times B_{kv})$, which is negligible compared to the $O(B_q \times D \times B_{kv})$ FP4MMA. The accuracy gain from keeping this correction in higher precision justifies the small overhead.

- **P quantization reuses the softmax max-reduction** (discussed in detail in Section 3.3 under "Reuse shuffle"). The rowmax of $P_{ij}$ (needed for the level-1 scaling $s_{P1}$) is computed from the *pre-exponentiated* values $S_{ij}$, using the fact that $\text{rowmax}(P_{ij}) = \exp(\text{rowmax}(S_{ij}) - m_{ij})$, where $m_{ij}$ is already available from the online softmax computation. This avoids a separate scan over P to find its maximum.

---

#### Hardware Implementation Optimizations

Section 3.3 describes three kernel-level optimizations that translate the numerical scheme into the reported 1038 TOPS. These are not algorithmic contributions but rather engineering techniques that address specific constraints of the Blackwell FP4 hardware.

**Permutation for K.** The FP4 matrix multiply instruction uses specific register layouts for its operands and accumulator that differ from the layouts used by FP16 GEMMs. Appendix Figure 19 shows the FP4 operand A register layout: elements are interleaved across threads such that thread 0 holds elements $\{v_0, v_1, v_8, v_9, ...\}$, thread 1 holds $\{v_2, v_3, v_{10}, v_{11}, ...\}$, and so on — a pattern designed for efficient data sharing during the matrix multiply. But the FP32 accumulator layout after the matrix multiply (Figure 20) arranges elements differently: thread 0 holds a contiguous block of columns $\{v_0, v_1\}$ for rows 0 and 8, while the operand A layout would require these to be scattered across threads.

The naive approach would be to perform thread shuffles after the matrix multiply to convert the accumulator layout to match operand A's layout for the next operation. But on Blackwell GPUs, thread shuffles are expensive (they require cross-thread data movement through shared memory or warp-level primitives), and in a compute-bound kernel, every wasted cycle directly reduces throughput.

The paper's solution: **permute the columns of the accumulator** into the layout shown in Figure 21 by rearranging operands before the matrix multiply rather than after. Specifically, the columns of the P tile are permuted to map the accumulator registers to the thread layout expected by the subsequent FP4MMA. To maintain mathematical correctness, the columns of K must be correspondingly permuted — but this permutation can be **fused with the K quantization kernel** (which already processes K column-by-column for the transpose), so it adds zero additional cost. The result is that the FP4MMA output is already in the correct register layout for the next operation, eliminating the need for expensive post-MatMul shuffles.

**Reuse shuffle for P quantization scaling.** Quantizing P to FP4 requires finding the maximum value in each 1×16 microscaling block. In the hardware register layout (Figure 21), these 16 consecutive elements are distributed across four threads (4 elements per thread), so computing the block maximum requires: (1) an intra-thread max reduction over each thread's 4 elements, followed by (2) an inter-thread shuffle to share the four per-thread maxima and take their maximum.

But the online softmax computation at Line 9 of Algorithm 1 already computes row-wise maxima over exactly these same elements. The insight is that the softmax max-reduction (which coalesces over blocks of 16 elements for numerical stability during exponentiation) produces intermediate values that can be **reused** for the P quantization max. Instead of computing the max twice (once for softmax, once for P quantization), the kernel stores the softmax's 16-wide max and reuses it when computing the level-1 P scale factor $s_{P1}$.

The paper reports that this fusion "reduces redundant shuffles and max operations by 50%, yielding about 10% whole kernel speedup" (Section 3.3). The 10% speedup from a single optimization is significant in absolute terms — at 1038 TOPS, the optimized kernel is approximately 10% faster than a version without shuffle reuse.

**Producer warp epilogue.** Warp-specialized kernels (a common CUTLASS pattern) partition GPU warps into "producers" (which load data from global memory to shared memory) and "consumers" (which perform the matrix multiply and store results to global memory). Typically, ping-pong scheduling between two consumer warps overlaps computation with memory operations: while one consumer waits for data, the other computes.

For the FP4 attention kernel, register pressure (the limited number of registers available per thread) makes the conventional warp specialization infeasible — consumer warps in FP4 attention already use most of their registers for matrix multiply accumulators, quantization scale factors, and intermediate values, leaving no room for storing output data destined for global memory.

The paper's innovation is to **move the global memory store responsibility from consumer warps to producer warps**. Specifically, ping-pong scheduling is implemented between two producer warps: while one producer loads the next set of Q, K, V tiles from global memory, the other producer handles the store of the previous output tile to global memory. Consumer warps are relieved of output storage duties entirely — they only transfer matrix multiply results from registers to shared memory, which requires fewer registers than a full global memory store pipeline.

This design "overlaps MatMul and global memory stores within register constraints, boosting throughput" (Section 3.3). The key resource being optimized is register count: by shifting the memory-store logic to producer warps (which have fewer register demands because they don't hold matrix multiply accumulators), the consumer warps can use their register budget entirely for computation, maximizing the occupancy and throughput of the FP4MMA instructions.

**Why these optimizations matter collectively:** the FP4MMA instruction itself operates at 1600 TOPS theoretical peak. Achieving 1038 TOPS (about 65% of peak) is actually an excellent utilization ratio for a memory-intensive kernel like attention — standard FP16 FlashAttention achieves only about 214 TOPS on RTX 5090 (about 65% of FP16 theoretical peak, similarly). The optimizations are what push the kernel from a naive FP4 implementation (which might achieve, say, 400-500 TOPS due to data movement overhead) to the reported 1038 TOPS. Each optimization addresses a specific bottleneck in the compute-memory-data movement pipeline, and their combination is what makes the 5× speedup over FP16 FlashAttention possible.

---

#### SageBwd Forward Pass: INT8 Quantization for Training

SageBwd uses INT8 rather than FP4 quantization because **training requires both forward and backward passes**, and the backward pass gradients are significantly more sensitive to quantization errors than forward activations. Moving from INT8 (255 representable values) to FP4 (15 representable values) would amplify gradient errors beyond what's acceptable for weight updates.

The forward pass of SageBwd applies per-block INT8 quantization to Q, K, and V:

$$s_X = \max(|X|) / 127, \quad \hat{X} = X / s_X$$

where $X \in \mathbb{R}^{B \times D}$ is a FlashAttention block (size $B$ is either $B_q$ for Q or $B_{kv}$ for K and V), $s_X \in \mathbb{R}$ is a single scalar scale factor, and $\hat{X} \in \mathbb{R}^{B \times D}$ is the quantized block in INT8 (stored as 8-bit integers but mathematically representing values in $[-127, 127]$ after rescaling). This is substantially coarser than SageAttention3's 1×16 microscaling but adequate for 8-bit precision because the 255 representable INT8 values provide enough granularity that per-block quantization suffices for most blocks.

For the $PV$ matrix multiplication, SageBwd uses **per-token quantization for P** rather than per-block quantization (Algorithm 2, Line 10):

$$s_P = \exp(\text{rowmax}(S_{ij}) - m_{ij}) / 127, \quad \hat{P}_{ij} = P_{ij} / s_P$$

where $s_P \in \mathbb{R}^{B_q}$ is a per-row scale factor (one scalar per row of the current P block), and $\hat{P}_{ij}$ is the per-token quantized P.

**What this computes:** after computing the exponentiated scores $P_{ij} = \exp(S_{ij} - m_{ij})$ (where $m_{ij}$ is the running online softmax max), the scale factor for each row is the row maximum divided by 127 (the INT8 max). This is a per-token (row-wise) quantization, not per-block.

**Why per-token for P rather than per-block:** the paper states (Section 4.1) that "a static per-block INT8 quantization with a static scale factor of $1/127$ for P is inaccurate," following the finding from SageAttention (Zhang et al., 2025). The reason is that attention maps have highly variable per-row max values — one row might have its attention concentrated on a single token (max near 1.0), while another row might have attention spread uniformly across many tokens (max near $1/N$, which could be 0.001 for N=1000). A single per-block scale factor would need to accommodate the largest row maximum in the block, causing rows with small attention peaks to be quantized to near-zero. Per-token quantization gives each row its own scale factor, preserving the relative attention distribution within each row.

An important engineering detail: the per-row max is not computed by a separate pass over P. Instead, Algorithm 2, Line 10 reuses the values already computed during online softmax:

$$s_P = \exp(\text{rowmax}(S_{ij}) - m_{ij}) / 127$$

where $\text{rowmax}(S_{ij})$ is the row-wise maximum of the pre-exponentiation scores, already computed at Line 9 for the softmax max-subtraction step, and $m_{ij}$ is the global running max. This gives $\max(P_{ij})$ exactly, avoiding redundant computation.

The forward pass algorithm for SageBwd (Algorithm 2) is structurally identical to standard FlashAttention, with the only differences being the insertion of quantization calls before each GEMM and the use of per-token P scaling. Smoothing K (Line 2) is applied as in SageAttention3.

---

#### SageBwd Backward Pass and the Critical dOV^T Decision

The backward pass of attention involves five matrix multiplications (Equation 8):

$$S = QK^T, \quad dV = P^T dO, \quad dP = dO V^T, \quad dQ = dS K, \quad dK = dS^T Q$$

The paper's key finding (Section 4.2) is that **quantizing $dP = dO V^T$ causes disproportionate gradient error** that accumulates catastrophically in the FlashAttention backward algorithm. The empirical evidence is in Table 1(c): when $dO V^T$ is quantized to INT8, the cosine similarity of dQ drops to 97.47% with RMSE 2.440; when kept in FP16, cosine similarity rises to 99.77% with RMSE 0.692.

**Why $dO V^T$ is uniquely sensitive:** the FlashAttention backward pass computes gradients in a tiled, recurrent fashion along the sequence length dimension. The computation of dQ (Algorithm 3, Line 10) accumulates contributions from each K,V block:

$$dQ_i \leftarrow dQ_i + dS_{ij} K_j$$

where $dS_{ij} = P_{ij} \circ (dP_{ij} - D_i)$ and $dP_{ij} = dO_i V_j^T$. Any error in $dP_{ij}$ propagates to $dS_{ij}$, which then gets multiplied by $K_j$ and accumulated into $dQ_i$. Since there are $T_n$ such accumulations (one per K,V block), and each accumulation uses the same potentially-erroneous $dS_{ij}$, **errors compound linearly with sequence length** — longer sequences mean more opportunities for quantization error in $dP$ to corrupt the final dQ and dK gradients.

The formal analysis in Appendix A.11 provides theoretical grounding. Under the assumption that matrix entries $X_{ij} \sim \mathcal{N}(\mu_{X,j}, \sigma^2_{X,j})$ (i.i.d. across tokens, identical within token positions), and using "round-to-nearest" INT8 quantization with per-block scaling, the expected error in dQ decomposes into two terms:

$$E[\Delta dQ] = E[\underbrace{(P \circ (dO \Delta V^T + \Delta dO V^T)) K}_{\Delta dQ^{(1)} \text{ from quantizing } dOV^T}] + E[\underbrace{\Delta dS K + dS \Delta K}_{\Delta dQ^{(2)} \text{ from quantizing } dS \text{ and } K}]$$

The authors show that $E[\Delta dQ^{(2)}] = 0$ because $\Delta dS$ and $\Delta K$ have symmetric error distributions centered at zero under round-to-nearest quantization (the quantization error is equally likely to be positive or negative, so it cancels in expectation). However, $E[\Delta dQ^{(1)}] \neq 0$ in general because the errors from $\Delta V^T$ and $\Delta dO$ interact with the non-zero means of the distributions. This means the $dOV^T$ quantization error does not cancel out over multiple accumulations but instead **accumulates systematically**.

**The design decision:** keep $dOV^T$ in FP16 (full precision) while quantizing the other four matrix multiplications ($QK^T$, $P^T dO$, $dS K$, $dS^T Q$) to INT8 per-block. This means SageBwd quantizes 6 of the 7 total GEMMs in attention (2 forward + 4 of 5 backward), achieving most of the speedup while preserving gradient accuracy where it matters most.

Algorithm 3 shows the backward pass implementation:
- Lines 5-7 compute $dV_j$: reconstruct $P_{ij}$ from stored softmax statistics, quantize both $P_{ij}$ and $dO_i$ per-block, then compute $dV_j \leftarrow dV_j + \text{MM}(\hat{P}_{ij}^T, \hat{dO}_i) \times s_P \times s_{dO}$ (INT8 quantized).
- Line 8 computes $dP_{ij} = dO_i V_j^T$ **in FP16** (the critical unquantized operation).
- Line 9 computes $dS_{ij} = P_{ij} \circ (dP_{ij} - D_i)$ in FP16, then quantizes $dS_{ij}$ per-block to INT8.
- Lines 10-11 compute $dQ_i$ and $dK_j$ using the quantized $dS_{ij}$ and quantized K or Q, with corrections for smoothing K applied in FP32.

**Why INT8 rather than FP8 for training:** Section 5.4 provides a direct comparison. Table 6 shows that INT8 SageBwd achieves lower L1 error for dQ (0.0290 vs 0.0696), dK (0.0317 vs 0.0999), and dV (0.0423 vs 0.0873) compared to FP8 SageBwd (which uses FP8 E4M3 format for gradients). Table 7 shows correspondingly higher cosine similarity: dQ at 0.9987 vs 0.9880, dK at 0.9993 vs 0.9910, dV at 0.9995 vs 0.9955. The practical consequence is shown in Table 8: models fine-tuned with INT8 SageBwd for 1000 steps and then evaluated with FP4 SageAttention3 inference achieve higher downstream accuracy (Qwen2.5-1.5B GSM8K: 0.5232 INT8 vs 0.5031 FP8; MMLU: 0.4934 vs 0.4689).

The authors attribute INT8's superiority to two factors: (1) "Higher gradient accuracy in attention backward" — the uniform quantization grid of INT8 provides more consistent error characteristics across the full value range compared to FP8's variable spacing (denser near zero, sparser at large values), which matters for gradients that can span multiple orders of magnitude; and (2) "Wider hardware support" — INT8 Tensor Cores exist on A100, H100, RTX 4090, and non-NVIDIA hardware like AMD MI250 and Ascend 910B, while FP8 is limited to Hopper and newer architectures. This second point is a practical rather than accuracy-motivated argument but matters for deployability.

---

#### Design Choices Summary and Their Justifications

- **NVFP4 over MXFP4** (Section 3.1): 1×16 blocks and E4M3 scale factors provide finer quantization granularity and more precise scale factor representation than 1×32 blocks with E8M0 power-of-2 scales. Empirical advantage: 99.52% vs 98.37% cosine similarity (Table 1a).

- **1×16 microscaling over per-tensor or per-token quantization** (Section 3.1): finer than per-tensor (catastrophic for outliers), finer than per-token (still allows within-row outliers to dominate), matches hardware FP4MMA block size requirement. Enables localization of outlier effects to individual blocks.

- **Smoothing Q and K over SmoothQuant/Hadamard** (Appendix A.7): SmoothQuant-style per-channel scaling (0.930 cosine similarity) and Hadamard rotation (0.941) are insufficient for FP4's extreme precision constraints. Attention-specific smoothing (subtract per-block Q means and global K means) directly targets the known structural properties of attention activations, achieving 0.991-0.996 cosine similarity (Table 16).

- **Two-level quantization over direct FP4 for P** (Section 3.2): direct quantization of P forces scale factors into the poorly-represented [0, 0.167] E4M3 subrange, using only ~35 distinct scale factor values. Two-level scheme rescales P so scale factors fully utilize the E4M3 range, providing 127 distinct values. Empirical advantage: 99.52% vs 93.32% cosine similarity, 0.201 vs 1.103 RMSE (Table 1b).

- **Per-token quantization for P in SageBwd over per-block** (Section 4.1): handles the high variability in per-row attention map magnitudes (max near 1.0 for concentrated attention vs near 0.001 for uniform attention) that would cause per-block quantization to erase small-attention rows.

- **Keeping $dOV^T$ in FP16 over full INT8 backward** (Section 4.2): identified as the accuracy-critical operation whose error accumulates systematically across sequence length (Appendix A.11 formal proof). Empirical advantage: dQ cosine similarity 99.77% vs 97.47% (Table 1c).

- **INT8 over FP8 for training** (Section 5.4): uniform quantization grid provides lower gradient error and wider hardware compatibility. Empirical advantage: downstream fine-tuning accuracy improvements of 1-2 percentage points on GSM8K and MMLU (Table 8).

- **Producer-warp epilogue over consumer-warp store** (Section 3.3): register pressure in FP4 attention kernel prevents consumer warps from holding both MatMul accumulators and store buffers. Shifting store responsibility to producer warps (which have spare register capacity) enables compute-memory overlap without exceeding register limits.

- **Reuse softmax max for P quantization over separate max pass** (Section 3.3): eliminates 50% of shuffle and max operations, yielding ~10% whole kernel speedup. Zero accuracy cost since the max values are identical.

- **K permutation during quantization over post-MatMul shuffle** (Section 3.3): fuses layout transformation into the quantization kernel (which must process K anyway), avoiding expensive cross-thread data movement after the matrix multiply. Zero additional memory I/O cost.

## 4. Key Insights and Innovations

### Innovation 1: FP4 Attention Is Feasible, Redefining the Precision Floor for Inference

The fundamental intellectual move here is demonstrating that **attention can operate at 4-bit floating-point precision without meaningful output degradation**, establishing a new lower bound for what precision is viable in transformer inference. This is not an incremental improvement over prior 8-bit or 4-bit integer attention — it is a qualitative shift because FP4's representational poverty (only 15 distinct non-zero magnitudes versus INT8's 255) makes it a fundamentally different quantization problem, not merely a more aggressive version of 8-bit quantization.

Prior to this work, the field's implicit assumption was that attention required at least 8-bit precision for acceptable quality. FlashAttention3's FP8 attention (Shah et al., 2024) pushed to 8-bit floating-point, SageAttention2 (Zhang et al., 2025) achieved 4-bit integer attention with per-thread quantization, but nobody had attempted FP4 — the precision level that matches the new Blackwell Tensor Core capability. The barrier was not hardware availability but a conceptual uncertainty: could attention's mixture of softmax normalization, outlier activations, and distributed attention patterns survive a 93.75% reduction in representable values (from 255 to 15 distinct magnitudes)?

The paper's answer — that FP4 attention achieves 1038 TOPS while maintaining 99.55% cosine similarity and near-identical end-to-end generation quality across five model families (Table 2) — reframes the question from "how much precision does attention need?" to "what quantization strategies make extreme precision viable?" The significance is that **FP4 is not inherently too coarse for attention; rather, naive quantization strategies fail, and the right approach (microscaling granularity, attention-specific smoothing, two-level scaling for the attention map) succeeds**. This matters beyond the immediate speedup because it establishes that the precision floor for attention is lower than the community assumed, with implications for future hardware design (how aggressive can tensor core precision get?) and for inference deployment strategy (can attention be the first component pushed to extreme precision, leaving other operations at higher precision?).

The evidence anchor is Table 2, which shows that across text-to-video (CogVideoX, HunyuanVideo, Mochi), text-to-image (Flux, Stable-Diffusion3.5), and text-to-text (Qwen2.5, Llama3.2) models, SageAttention3 produces evaluation metrics that are essentially overlapping with full-precision attention — CLIP scores within 0.001, VQA scores within 1-2 points, and FID/sFID differences that are smaller than typical run-to-run variation. The visible examples (Figure 9, Figures 10-14) reinforce this: generated images and videos are visually indistinguishable from full precision. This is not a theoretical claim about signal-to-noise ratios — it is a practical demonstration that FP4 attention, properly implemented, is a drop-in replacement with no user-perceptible quality loss.

### Innovation 2: Diagnosing and Solving the "Scale Factor Starvation" Problem for Low-Range Matrices

The paper's most conceptually elegant contribution is the identification and resolution of what we might call **scale factor starvation** — the phenomenon where quantizing a matrix whose values lie in a narrow dynamic range forces the quantization scale factors themselves into a poorly-represented subrange of their own data type, causing a double-penalty quantization error. This is a general diagnostic concept that extends beyond attention and beyond FP4.

The specific manifestation is with the attention map P, whose values lie in [0, 1]. When P is quantized to FP4 using the standard microscaling quantizer ϕ, the per-block scale factors $s_P = \max(P_{ij}) / 6$ are trapped in [0, 0.167]. But these scale factors must be stored in E4M3 FP8 format (hardware requirement), and in the [0, 0.167] subrange, E4M3 has only about 35 distinct representable values — meaning the scale factor itself is effectively quantized to about 5 bits of precision before it even scales the FP4 values. The dequantized output thus has only $35 \times 8 = 280$ distinct representable values, dramatically fewer than the $127 \times 8 = 1016$ that the format could theoretically support.

What makes this insight distinctive is that it identifies a **metadata precision problem**, not a data precision problem. Prior quantization work focused on whether 4-bit values could approximate the target distribution — the standard "does 4-bit have enough levels?" question. But the paper shows that even if 4-bit values are adequate, poorly-represented scale factors can be the dominant error source. This is a second-order effect that only becomes visible when scale factors are themselves stored in a limited-precision format (as NVFP4 requires) rather than in FP32 (as software quantization typically assumes).

The two-level scaling solution is a general technique for any quantization scheme where the scale factor's data type has limited dynamic range and the target matrix's values are concentrated in a narrow range: first rescale the matrix to occupy the full representable range of the scale factor type (using unquantized FP32 arithmetic), then apply the standard quantizer. The technique should transfer to other low-precision attention implementations, to quantization of other narrow-range intermediates (layer norm outputs, sigmoid activations), and potentially to weight quantization where certain layers have unusually small magnitude ranges.

The empirical evidence (Figure 12, Table 1b) makes the case: without two-level scaling, FP4 attention produces severely degraded outputs (Figure 12c shows obvious color and structural artifacts), and cosine similarity drops to 93.32% versus 99.52% with two-level scaling. The formal analysis in Appendix A.10, which counts representable values and bounds relative quantization error, generalizes the insight mathematically — the error improvement does not depend on the specific values 35, 127, or 448, but on the ratio of representable scale factor counts, which is fundamentally determined by how much of the scale factor type's dynamic range is utilized.

### Innovation 3: Identifying the "One Critical GEMM" in Attention Training and Quantizing Everything Else

The paper's training contribution is not that 8-bit attention works — SageAttention already showed that 8-bit forward-pass attention is accurate. The intellectual contribution is the **diagnostic finding that among the five backward-pass matrix multiplications, exactly one — $dOV^T$ — is accuracy-critical in a way the others are not**, and that this specificity enables a practical training acceleration strategy (quantize six of seven GEMMs) that would fail if applied uniformly.

This is a fundamentally empirical discovery, not something derivable from first principles. One might reasonably expect that all five backward GEMMs are similarly sensitive to quantization, or that the error characteristics would be symmetric — if quantizing $dS K$ is acceptable, why wouldn't quantizing $dO V^T$ also be acceptable? The paper's answer (Section 4.2, Appendix A.11) identifies the mechanism: $dP = dO V^T$ feeds into $dS = P \circ (dP - D)$, and errors in $dS$ then accumulate across sequence positions in FlashAttention's recurrent backward computation — each K,V block adds to the running dQ and dK, so an error in the dP for block j corrupts every subsequent accumulation. The other GEMMs ($P^T dO$, $dS K$, $dS^T Q$) do not have this sequential accumulation structure, so their quantization errors cancel in expectation rather than compounding.

What makes this distinctive as a research contribution is that it converts a binary question ("does 8-bit training work?") into a **structured understanding of gradient sensitivity** that enables an efficient hybrid approach. The naive strategy would be to quantize all five backward GEMMs uniformly, see that accuracy degrades, and conclude that 8-bit attention training is infeasible. The paper instead performs the ablation that identifies *which* operation is the bottleneck and shows that keeping only that one operation in FP16 while quantizing the other four recovers full accuracy.

This has implications beyond attention training. The pattern — gradient computations with recurrent accumulation structure are disproportionately sensitive to quantization error — likely applies to other recurrent neural network components, state-space models, and any architecture where gradients are computed iteratively along a sequence dimension. The paper does not make this generalization claim, but the diagnostic framework (identify which operator's error accumulates vs. cancels) is transferable.

The evidence is decisive: Table 1(c) shows that quantizing $dOV^T$ drops dQ cosine similarity from 99.77% to 97.47% and triples RMSE from 0.692 to 2.440. The fine-tuning loss curves (Figure 8b-e) show SageBwd perfectly tracking BF16 across four diverse datasets (GSM8K, DROP, MMLU, HELLASWAG) for two model families (Qwen2.5 and Llama3.2), with final evaluation metrics (Table 3) showing differences within 0.01 — well within typical run-to-run variance. The Appendix A.11 formal analysis, while making strong distributional assumptions, provides a plausible mechanism: $E[\Delta dQ^{(1)}] \neq 0$ from the $dOV^T$ error term versus $E[\Delta dQ^{(2)}] = 0$ from the other quantized terms.

### Innovation 4: The Mismatch-Reduction Hypothesis for Combined Training-Inference Quantization

Section 5.3 reports a counterintuitive result: models fine-tuned with INT8 SageBwd and then evaluated with FP4 SageAttention3 achieve *higher* downstream accuracy than models fine-tuned with BF16 and then evaluated with FP4 inference (Table 5: Qwen2.5-1.5B on GSM8K improves from 0.4912 to 0.5232; MMLU improves from 0.4688 to 0.4934). The paper's explanation is that "INT8 and FP4 share a more similar representable data distribution, reducing the mismatch error compared to BF16."

This is a hypothesis, not a proven mechanism, and the paper presents it as such ("This improvement is likely because..."). But the *conceptual move* is significant regardless of whether this specific explanation is correct: it reframes quantization not as a one-time accuracy-cost decision but as an **end-to-end pipeline property** where training precision and inference precision interact. Models adapt to the quantization characteristics they experience during training — a form of implicit quantization-aware training that occurs naturally rather than through explicit simulation.

The implication, if the hypothesis holds, is that the optimal training precision for a given inference precision is not necessarily "the highest precision available" but rather "the precision most similar to inference precision." This would invert the standard deployment pipeline (train in BF16, quantize for inference) and suggest instead that training should occur at a precision matched to the inference target — even if that training precision is not the highest available.

The evidence is limited (two model sizes, two benchmarks, one fine-tuning setup), and the paper does not explore the mechanism further (e.g., by comparing FP8 training to FP4 inference, or by examining whether the effect persists with explicit quantization-aware training). So this is better understood as an **insight-in-formation** — an empirical observation that opens a research direction rather than a fully validated finding. But in a paper whose training contribution is explicitly framed as an "exploration," surfacing this observation is intellectually honest and potentially generative for follow-up work on precision-matched training-inference pipelines.

---

**Distinguishing incremental from fundamental across the innovations:** Innovation 1 (FP4 feasibility) and Innovation 2 (scale factor starvation) are fundamental — they establish new lower bounds and introduce general diagnostic concepts. Innovation 3 (the one critical GEMM) is a mixed case: the empirical finding is fundamental for attention training but the specific mechanism (accumulation structure) may or may not generalize. Innovation 4 (training-inference precision matching) is incremental evidence for a hypothesis that the paper itself does not fully validate — it is more of a "pointer to future work" than a completed contribution. This distribution is appropriate for a paper that is primarily a systems contribution with one training exploration section.

## 5. Experimental Analysis

### Evaluation Methodology

- **Dataset.** For FP4 inference quality evaluation, the paper uses the open-sora prompt set for text-to-video models, COCO annotations for text-to-image models, and GSM8K, DROP, MMLU, and HELLASWAG for language model fine-tuning evaluation (Section 5.1, Appendix A.3). For training experiments (SageBwd), fine-tuning is conducted on GSM8K, DROP, MMLU, and HELLASWAG datasets; pretraining uses FineWeb-Edu. The MATH benchmark — used in the reference example paper — does not appear in this paper.
- **Base model(s).** For inference: CogVideoX-2B, HunyuanVideo, Mochi (text-to-video); Flux, Stable-Diffusion3.5 (text-to-image); Qwen2.5 (1.5B and 3B), Llama3.2 (1B and 3B) (text-to-text). For training fine-tuning experiments: Qwen2.5-1.5B, Qwen2.5-3B, and Llama3.2-1B. For pretraining: a Llama-style 400M parameter model with hidden size 1024, 20 layers, intermediate size 3072, and 16 attention heads, trained on FineWeb-Edu with 2M tokens per step. The authors state the models are chosen to represent "diverse... representative models from language, image, and video generation" (Section 5.1).
- **Metrics.** Three categories: (1) **Kernel speed:** measured in TOPS (tera-operations per second) on RTX 5090 (FP4 inference) and RTX 4090 (INT8 training), with sequence lengths swept from 1K to 32K and head dimensions of 64 and 128. (2) **Accuracy of quantized attention:** cosine similarity (CosSim), relative L1 distance (L1), and root mean square error (RMSE) between quantized attention output O' and full-precision attention output O, measured across all layers of CogVideoX-2B. (3) **End-to-end model quality:** for text-to-video, CLIPSIM, CLIP-Temp, VQA-a, VQA-t, and Flow-score; for text-to-image, FID, sFID, CLIP score, and ImageReward; for text-to-text, accuracy on GSM8K and MMLU, F1 on DROP, and accuracy on HELLASWAG. The paper also reports end-to-end inference latency (seconds) for video generation models.
- **Baselines.** For kernel speed comparisons (Figures 4-7): PyTorch native attention, xformers (Lefaudeux et al., 2022), FlashAttention2 (Dao, 2024), SageAttention1 (Zhang et al., 2025), and SageAttention2 (Zhang et al., 2025). FlashAttention3 cannot run on RTX 5090 (Hopper-only), so FlashAttention2 is the fastest available baseline on that hardware. For end-to-end quality (Table 2): full-precision FP16 attention and SageAttention2 (8-bit). For training (SageBwd): BF16 full-precision attention as the quality baseline; FlashAttention2 (CUDA), FlashAttention2 (Triton), xformers, and PyTorch for kernel speed baselines.
- **Generation budget / compute accounting.** Kernel speed is measured in TOPS (hardware throughput) and end-to-end latency in seconds — there is no "generation budget" or "number of samples" axis as in the reference example paper because this is a systems/kernel work, not a search/revision strategy work. The fair comparison metric is throughput at matched sequence length, head dimension, and causality setting on the same GPU. For SageAttention3, the throughput includes all quantization, smoothing correction, and FP4MMA operations; for baselines, it includes their full FP16 or FP8 computation.
- **Cross-validation / statistical protocol.** For fine-tuning experiments with SageBwd, the paper runs five different random seeds (42, 233, 1234, 5678, 1) and reports mean and standard deviation (Tables 9-14 in Appendix A.4). For end-to-end quality metrics (Table 2), the paper reports point values without error bars — typical for generation model evaluation where running multiple seeds is computationally prohibitive. Kernel speed measurements (Figures 4-7) are reported as single-point TOPS values at each sequence length-head dimension combination, following standard CUDA kernel benchmarking practice (warm-up runs excluded, best of multiple runs reported implicitly).

### Main Quantitative Results

#### Kernel Speed: SageAttention3 Achieves 1038 TOPS on RTX 5090

**The headline result.** SageAttention3 achieves 1038 TOPS on RTX 5090 at sequence length 2K with head dimension 128 and causal=False (Figure 4). This represents approximately 5× the throughput of FlashAttention2 at the same configuration (214 TOPS), approximately 11× the throughput of xformers (93 TOPS), and approximately 65× the throughput of PyTorch native attention (16 TOPS). The peak of 1038 TOPS is achieved at head_dim=128, causal=False, sequence length 2K; at 32K with causal=True, throughput is 1027 TOPS — nearly identical, showing the kernel scales well to long sequences.

Figure 5 shows results for head dimension 64: SageAttention3 peaks at 839 TOPS (causal=False, seq_len=32K) versus FlashAttention2 at 220 TOPS — again approximately 3.8-4× speedup, though the absolute throughput is lower than at head_dim=128 because the GEMM dimensions are smaller, reducing Tensor Core utilization. At causal=True and head_dim=64 (Figure 5, right panel), SageAttention3 achieves 806 TOPS at seq_len=32K versus FlashAttention2 at 217 TOPS — approximately 3.7× speedup.

**Scaling with sequence length.** Across all configurations, SageAttention3 shows remarkably flat throughput scaling from 1K to 32K sequences, unlike FlashAttention2 which increases from 173 TOPS at 1K to 215 TOPS at 32K (causal=False, head_dim=128) — FlashAttention2 gains about 24% throughput with sequence length due to better amortization of overhead over larger tiles, while SageAttention3 gains only about 7% (964 to 1038 TOPS), indicating that the FP4MMA instructions dominate the runtime even at short sequences and there is relatively less fixed overhead to amortize.

**The gap between SageAttention3 and prior SageAttention versions is substantial.** SageAttention2 achieves 556 TOPS at head_dim=128, causal=False, 32K — the fastest prior SageAttention variant — while SageAttention3 achieves 1027 TOPS at the same configuration, a 1.85× improvement. SageAttention1 achieves 479 TOPS, so the progression from SageAttention1 to SageAttention3 represents a 2.15× throughput improvement. The jump from SageAttention2 to SageAttention3 (1.85×) directly reflects the transition from INT4 per-thread quantization on older Tensor Cores to FP4 microscaling on Blackwell FP4 Tensor Cores — confirming that the new hardware capability translates into practical throughput gains.

**Causality impact.** The causal=True configuration is consistently slower than causal=False for all methods — SageAttention3 drops from 1038 TOPS to 1027 TOPS at head_dim=128, 2K (a negligible 1% reduction), while FlashAttention2 drops from 214 TOPS to 209 TOPS (2% reduction). The small impact of causality for SageAttention3 suggests that the mask application and causal bookkeeping are well-fused with the computation and do not bottleneck the FP4MMA throughput.

**What the TOPS numbers mean in practice.** 1038 TOPS at FP4 on RTX 5090 represents approximately 65% of the theoretical peak FP4 throughput (roughly 1600 TOPS per the paper's Section 3.1). This utilization ratio is excellent for an attention kernel — FlashAttention2 achieves about 214 TOPS out of roughly 330 TFLOPS theoretical FP16 peak on the same hardware, also about 65%. The 5× speedup comes entirely from the 8× theoretical throughput advantage of FP4 over FP16 (1600 vs 200 TOPS), partially offset by the overhead of quantization, smoothing corrections, and the fact that not all operations in the attention kernel are matrix multiplications. The roughly 62.5% efficiency in converting the theoretical 8× to a realized 5× is attributed to the hardware optimizations described in Section 3.3.

#### Kernel Speed: SageBwd Achieves up to 1.67× Speedup on RTX 4090

**Forward pass speed.** Figures 15 and 16 (Appendix A.2) break out SageBwd's forward-only kernel speed. At head_dim=128, causal=False, seq_len=1K: SageBwd forward achieves 155 TOPS versus FlashAttention2 (CUDA) at 114 TOPS — approximately 1.36× speedup. At head_dim=64: 157 TOPS versus 113 TOPS — approximately 1.39×. The forward speedup is modest because the forward pass has only two GEMMs (QK^T and PV), and the INT8 quantization overhead partially offsets the theoretical 2× throughput advantage of INT8 Tensor Cores over FP16 on RTX 4090.

**Backward pass speed.** Figures 17 and 18 (Appendix A.2) show the backward kernel speed. At head_dim=128, causal=False, seq_len=2K: SageBwd backward achieves 139 TOPS versus FlashAttention2 at 103 TOPS — approximately 1.35× speedup. At 32K: 152 TOPS versus 149 TOPS — the gap narrows to approximately 1.02×, suggesting that at very long sequences, the backward pass becomes memory-bound rather than compute-bound, and the INT8 compute advantage provides diminishing returns.

**Combined forward+backward speed (main Figures 6-7).** Figure 6 shows the total (forward+backward) speed for head_dim=128 on RTX 4090. At seq_len=2K, causal=False: SageBwd achieves 231 TOPS versus FlashAttention2 (CUDA) at 146 TOPS — approximately 1.58× speedup. At 32K: 231 versus 155 — approximately 1.49×. At causal=True, head_dim=64, 32K (Figure 7, right): 263 versus 160 — approximately 1.64×. The stated "1.67× speedup at most" (Section 5.2) appears to be the maximum observed across all configurations.

**Comparison with FlashAttention2 Triton and xformers.** SageBwd consistently outperforms FlashAttention2 implemented in Triton and xformers by larger margins than it outperforms CUDA FlashAttention2. At head_dim=128, causal=False, 2K (Figure 6): SageBwd at 231 TOPS versus FlashAttention2 Triton at 104 TOPS (2.22×) and xformers at 55 TOPS (4.2×). This gap likely reflects the optimization maturity of the respective implementations — CUDA FlashAttention2 is highly optimized, while Triton and xformers may have higher overhead for the attention backward pass.

**The speedup for training is smaller than for inference.** SageAttention3's 5× inference speedup on RTX 5090 versus SageBwd's 1.67× training speedup on RTX 4090 reflects four compounding factors: (1) FP4 has 8× theoretical throughput over FP16, while INT8 has only 2× theoretical throughput over FP16; (2) the backward pass has more operations that remain in FP16 (the critical dOV^T is unquantized, reducing the fraction of compute that benefits from INT8); (3) the RTX 4090 lacks FP4 Tensor Cores entirely, so SageBwd cannot use FP4; (4) SageBwd is implemented in Triton rather than CUTLASS/CUDA, which may leave optimization headroom (the paper acknowledges this, noting "a noticeable gap between its current speed and theoretical upper bounds").

#### End-to-End Model Quality: SageAttention3 Preserves Generation Quality Across Five Model Families

**Table 2** reports end-to-end metrics comparing full-precision FP16 attention, SageAttention2 (8-bit INT8), and SageAttention3 (4-bit FP4) across text-to-video, text-to-image, and text-to-text models. The results support the claim that SageAttention3 "almost incurs no end-to-end quality loss" (Section 5.2).

**Text-to-video models:**

- **CogVideoX-2B:** Full precision vs. SageAttention3 — CLIPSIM: 0.1865 vs. 0.1881 (essentially identical); CLIP-T: 0.9968 vs. 0.9969; VQA-a: 70.476 vs. 69.860 (0.6 point difference); VQA-t: 69.875 vs. 70.364 (0.5 point difference in the opposite direction, with SageAttention3 slightly higher); FScore: 4.780 vs. 4.035 (a non-trivial 0.75 point drop, approximately 15.6% reduction). The FScore drop on CogVideoX is the largest quality regression reported and represents a meaningful degradation in temporal consistency. However, the authors do not flag this as a significant concern, and the visible examples in Figure 13 do not show obvious temporal artifacts — this metric may have high variance or limited perceptual correlation.

- **HunyuanVideo:** CLIPSIM: 0.1838 vs. 0.1866 (SageAttention3 slightly higher); CLIP-T: 0.9993 vs. 0.9993 (identical); VQA-a: 68.998 vs. 70.552 (SageAttention3 1.5 points higher); VQA-t: 78.891 vs. 75.440 (SageAttention3 3.5 points lower); Flow-score: 1.4793 vs. 1.232 (0.247 point drop, approximately 16.7%). The divergent pattern — SageAttention3 scores higher on some metrics, lower on others — suggests that the FP4 quantization introduces subtle quality tradeoffs rather than uniform degradation. That VQA-t (technical quality) drops while VQA-a (aesthetic quality) rises is unexpected and not discussed.

- **Mochi:** CLIPSIM: 0.1828 vs. 0.1800 (0.003 drop); CLIP-T: 0.9990 vs. 0.9993 (identical); VQA-a: 61.984 vs. 61.863 (0.1 point drop); VQA-t: 61.000 vs. 59.429 (1.6 point drop); FScore: 1.804 vs. 1.649 (0.155 drop, approximately 8.6%). Mochi shows the most consistent pattern of slight degradation across all technical metrics.

Comparing SageAttention2 (8-bit) and SageAttention3 (4-bit): they are generally within 1-2 points of each other on most metrics, with no consistent winner. For example, on CogVideoX FScore, SageAttention2 scores 4.534 versus SageAttention3's 4.035 — SageAttention2 is notably better. But on HunyuanVideo VQA-a, SageAttention3 at 70.552 outperforms SageAttention2 at 69.497. This suggests that the quality gap between 8-bit and 4-bit attention is small enough that other sources of variance (random seed, prompt distribution) dominate on many metrics.

**Text-to-image models:**

- **Flux:** FID: 162.812 vs. 162.121 (SageAttention3 slightly better, lower is better); sFID: 146.980 vs. 142.839 (SageAttention3 4.1 points better); CLIP: 31.409 vs. 31.450 (essentially identical); ImageReward: 0.91 vs. 0.94 (SageAttention3 0.03 higher). SageAttention3 matches or slightly exceeds full precision on every metric — a surprising result that the paper does not explain. This could reflect the inherent noise in FID/sFID estimation with limited samples, or it could hint that FP4 quantization introduces a form of regularization that slightly improves generation diversity or fidelity.

- **Stable-Diffusion3.5:** FID: 166.421 vs. 166.102 (SageAttention3 slightly better); sFID: 146.379 vs. 145.587 (SageAttention3 slightly better); CLIP: 31.93 vs. 32.01 (essentially identical); ImageReward: 0.93 vs. 0.92 (SageAttention3 slightly worse). Again, SageAttention3 is essentially indistinguishable from full precision, with metric differences smaller than typical run-to-run variance for these models.

**The quality preservation is a genuine finding, not a foregone conclusion.** Given that FP4 attention reduces the precision of both matrix multiplications in attention by 93.75% compared to FP16 (15 vs. 255 distinct non-zero magnitudes, if we compare to INT8 equivalently), the fact that end-to-end generation quality remains essentially unchanged is surprising. The paper's explanation — that the microscaling quantization, two-level P scaling, and smoothing techniques adequately preserve the information content of attention — is supported by the metric tables but the underlying mechanism (how much information attention *needs* vs. how much FP4 preserves) is not quantified.

**Visible examples (Figures 9-14) show no perceptible quality difference.** Figure 9, Figure 10, Figure 11, Figure 13, and Figure 14 present side-by-side comparisons of FP16 and SageAttention3 generations for HunyuanVideo, Stable-Diffusion3.5, Flux, CogVideoX, and HunyuanVideo respectively. In all cases, the FP4-attention outputs are visually indistinguishable from full-precision outputs — no obvious color shifts, structural deformations, or texture degradation. This is consistent with the metric results and provides face validity that the reported metric similarities correspond to genuine perceptual equivalence.

Figure 12 (Appendix A.1) is particularly informative: it shows CogVideoX frames generated with full precision, two-level quantization of P, and direct quantization of P. The direct quantization output (Figure 12c) shows severe artifacts — color shifts, structural distortion, loss of fine detail — visually confirming the importance of the two-level scaling technique. The two-level output (Figure 12b) is visually identical to full precision (Figure 12a), consistent with the 99.52% cosine similarity in Table 1(b).

#### End-to-End Inference Latency: 3× Speedup on HunyuanVideo, 2.4× on CogVideoX

**Table 4(a)** reports total end-to-end generation latency (not just attention kernel time) for video models on RTX 5090:

- **CogVideoX-2B:** Original (FP16 FlashAttention2) takes 64 seconds; SageAttention1: 55 seconds (1.16×); SageAttention2: 46 seconds (1.39×); SageAttention3: 27 seconds (2.37×). The progression shows that each generation of SageAttention brings meaningful end-to-end improvement, with the jump from SageAttention2 to SageAttention3 providing the largest relative gain.

- **HunyuanVideo:** Original: 489 seconds (approximately 8 minutes); SageAttention1: 257 seconds; SageAttention2: 240 seconds; SageAttention3: 164 seconds (2.98×). The 3× end-to-end speedup on HunyuanVideo is slightly less than the 5× kernel speedup, which is expected because non-attention components (convolution layers, normalization, MLP blocks, I/O) do not benefit from FP4 attention acceleration. The fact that attention optimization alone can nearly triple total generation speed indicates that attention is the dominant bottleneck in HunyuanVideo inference.

**The gap between kernel speedup (5×) and end-to-end speedup (3×) quantifies Amdahl's Law.** If attention accounts for fraction $f$ of total runtime, and attention is accelerated by factor $S_a = 5\times$, the end-to-end speedup is $1 / ((1-f) + f/S_a)$. With $S_a = 5$ and end-to-end speedup approximately 3, we can solve for $f$: $3 \approx 1 / (1-f + f/5) \Rightarrow 1-f + f/5 \approx 1/3 \Rightarrow f \approx 0.83$. This suggests attention accounts for roughly 83% of HunyuanVideo inference time — consistent with the quadratic complexity of attention dominating for long video sequences.

#### SageBwd Fine-Tuning: Lossless Performance on Four Benchmarks

**Figure 8(b-e)** shows fine-tuning loss curves for Qwen2.5 (1.5B and 3B) and Llama3.2 (1B) on GSM8K, DROP, MMLU, and HELLASWAG datasets. In all four subfigures, the BF16 (full-precision) and SageBwd (8-bit attention) loss curves are visually overlapping throughout training — there is no visible divergence, no gap that widens over time, and no indication that the 8-bit quantization introduces systematic bias in the gradient updates. The curves track each other within the natural stochastic variation of minibatch training.

**Table 3** reports the final evaluation metrics after fine-tuning:

- **Qwen2.5-1.5B:** GSM8K: BF16 0.521 vs. SageBwd 0.520 (difference: -0.001); DROP: 0.733 vs. 0.734 (+0.001); MMLU: 0.569 vs. 0.574 (+0.005); HELLASWAG: 0.905 vs. 0.911 (+0.006). All differences are within 0.006 — well below what would be considered meaningful for these benchmarks.

- **Qwen2.5-3B:** GSM8K: 0.601 vs. 0.607 (+0.006); DROP: 0.785 vs. 0.782 (-0.003); MMLU: 0.640 vs. 0.653 (+0.013); HELLASWAG: 0.944 vs. 0.943 (-0.001). The MMLU difference of +0.013 is the largest single deviation and favors SageBwd — likely random variation rather than systematic improvement.

- **Llama3.2-1B:** GSM8K: 0.259 vs. 0.268 (+0.009); DROP: 0.641 vs. 0.637 (-0.004); MMLU: 0.464 vs. 0.458 (-0.006); HELLASWAG: 0.828 vs. 0.823 (-0.005). Again, differences are within 0.01 and show no consistent direction — SageBwd is sometimes slightly higher, sometimes slightly lower.

**The "lossless" claim is well-supported.** Across 12 evaluation settings (3 models × 4 benchmarks), the maximum absolute difference between BF16 and SageBwd is 0.013 (MMLU for Qwen2.5-3B), and the root mean square difference is approximately 0.006. This is smaller than typical run-to-run variance with different random seeds for fine-tuning, which the paper quantifies in Appendix Tables 9-14: for Qwen2.5-1.5B on GSM8K, the standard deviation across five seeds is 0.009 for BF16 and 0.009 for SageBwd — larger than the 0.001 mean difference between them. The within-method variance across seeds is larger than the between-method variance, making it impossible to distinguish BF16 and SageBwd performance on any individual benchmark.

The multi-seed results in Appendix Tables 9-14 reinforce this: for instance, Qwen2.5-1.5B on GSM8K (Table 9) shows individual seed results of (SageBwd vs. BF16): 0.5133 vs. 0.5125, 0.5027 vs. 0.5042, 0.4973 vs. 0.4973, 0.5201 vs. 0.5208, 0.5049 vs. 0.5057 — every pair is within 0.005, confirming that SageBwd and BF16 produce statistically indistinguishable fine-tuning outcomes.

#### SageBwd Pretraining: Slower Convergence

**Figure 8(a)** shows the pretraining loss curve for a Llama-400M model trained on FineWeb-Edu. The BF16 and SageBwd curves both converge, but the SageBwd loss is consistently higher than BF16 at the same step count — the gap is visible throughout training and does not close by step 20,000. The paper characterizes this as "SageBwd can achieve loss convergence, its convergence speed is relatively slow" (Section 5.2).

The magnitude of the convergence gap is not quantified in the text (no numerical comparison of loss values at specific steps), but the figure shows a visible and persistent offset. At step 20,000, the SageBwd loss appears roughly 0.2-0.3 higher than BF16 on what appears to be a log-scale or approximately linear-scale y-axis — the exact value is not legible from the figure reproduction. This result establishes the boundary condition: 8-bit attention is viable for fine-tuning (where the model starts from a strong initialization and makes relatively small parameter updates) but not for pretraining (where the model must learn representations from scratch and gradient accuracy over many steps matters more).

**Why fine-tuning works but pretraining doesn't** is not directly addressed by the paper beyond the statement that convergence is slower. Possible mechanisms: (1) the accumulated gradient errors from quantizing four of five backward GEMMs, while small per step, compound over the much larger number of pretraining iterations (20,000+ steps shown versus 600-700 fine-tuning steps), eventually delaying or preventing convergence to the same loss basin; (2) pretraining explores a much larger region of parameter space, exposing the model to regimes where the per-block INT8 quantization assumptions (e.g., that max absolute values are representative of block distributions) break down; (3) the 400M parameter scale may be small enough that gradient noise from quantization is a larger fraction of the signal than for the 1.5B-3B fine-tuned models. The paper does not investigate these hypotheses.

#### Combined SageBwd + SageAttention3: Training-Inference Precision Matching Improves Accuracy

**Table 5** shows an interesting interaction: models fine-tuned with INT8 SageBwd and then evaluated with FP4 SageAttention3 achieve higher accuracy than models fine-tuned with BF16 and evaluated with FP4 SageAttention3:

- **Qwen2.5-1.5B, GSM8K:** BF16 fine-tuning → 0.4912; SageBwd fine-tuning → 0.5232 (+0.032, approximately 6.5% relative improvement). MMLU: 0.4688 → 0.4934 (+0.025).

- **Qwen2.5-3B, GSM8K:** BF16 → 0.5860; SageBwd → 0.5945 (+0.009). MMLU: 0.6000 → 0.6032 (+0.003).

The effect is larger for the smaller model (0.032 vs. 0.009 on GSM8K) and for GSM8K versus MMLU, though with only two model sizes and two benchmarks, systematic patterns are hard to extract.

**The paper's interpretation** — that "INT8 and FP4 share a more similar representable data distribution, reducing the mismatch error compared to BF16" — is plausible but untested. An alternative explanation is that SageBwd's quantized gradients act as a form of regularization during fine-tuning, producing a model that is inherently more robust to inference-time quantization regardless of precision matching. The paper does not disentangle these mechanisms, and the experiment does not include an FP8 training comparison for further triangulation.

**The absolute accuracies deserve scrutiny.** The BF16 fine-tuned Qwen2.5-1.5B GSM8K accuracy of 0.4912 is notably lower than the SageBwd fine-tuned value of 0.520 reported in Table 3 (which uses BF16 inference, not FP4 inference). The 0.4912 in Table 5 represents BF16 training + FP4 inference, while the 0.520 in Table 3 represents SageBwd training + FP4 inference — but the comparison point in Table 3 for BF16 training + BF16 inference is 0.521. The three-way comparison (BF16 train + BF16 inference: 0.521; BF16 train + FP4 inference: 0.491; SageBwd train + FP4 inference: 0.523) suggests that FP4 inference degrades BF16-trained models by about 0.03 accuracy points, while SageBwd-trained models are immune to this degradation — consistent with the mismatch-reduction hypothesis but also consistent with SageBwd training being inherently more robust to quantization noise.

### Ablation Studies and Robustness Checks

**NVFP4 vs. MXFP4 quantization format (Table 1a):** NVFP4 (1×16 blocks, E4M3 scale factors) substantially outperforms MXFP4 (1×32 blocks, E8M0 scale factors). CosSim: 99.52% vs. 98.37%; L1: 0.077 vs. 0.294; RMSE: 0.201 vs. 0.994. The 1×16 block size and E4M3 scale factor precision are both contributing factors — the paper does not isolate their individual contributions, but the combined effect is clear. This justifies the choice of NVFP4 for all subsequent experiments.

**Direct vs. two-level quantization for P (Table 1b, Figure 3, Figure 12):** Direct quantization of P achieves only 93.32% CosSim, 0.193 L1, and 1.103 RMSE — catastrophic for generation quality. Two-level quantization recovers 99.52% CosSim, 0.077 L1, and 0.201 RMSE. The visual evidence in Figure 12 confirms that the metric improvement corresponds to genuine perceptual quality restoration. Figure 3 quantifies the mechanism: two-level quantization increases the number of distinct E4M3 scale factor values from approximately 35 to approximately 127, directly reducing the quantization error bound.

**Quantizing vs. preserving dOV^T in FP16 (Table 1c):** When dOV^T is quantized to INT8 (like the other four backward GEMMs), dQ CosSim drops to 97.47% with RMSE 2.440. Preserving it in FP16 yields 99.77% CosSim and RMSE 0.692 — a 3.5× reduction in RMSE and restoration of gradient accuracy to acceptable levels. This single ablation is the empirical basis for SageBwd's entire backward pass design.

**Smoothing strategy ablation (Table 16):** With no smoothing, FP4 attention achieves only 0.916 CosSim. SmoothQuant-style per-channel scaling: 0.930. Hadamard rotation: 0.941. Smoothing Q alone: 0.983. Smoothing K alone: 0.991. Smoothing both: 0.996 (full pipeline). The jump from generic smoothing (0.930-0.941) to attention-specific smoothing (0.983-0.996) demonstrates that FP4 requires techniques tailored to the structural properties of attention activations. Smoothing K (global mean subtraction) provides the larger individual benefit at 0.991 versus 0.983 for smoothing Q alone, suggesting that key outliers are the primary obstacle to FP4 quantization accuracy, consistent with the known outlier problem in LLM activations.

**INT8 vs. FP8 for training backward pass (Tables 6-8):** INT8 backward achieves lower gradient L1 error: dQ 0.0290 vs. 0.0696 (FP8), dK 0.0317 vs. 0.0999, dV 0.0423 vs. 0.0873. Cosine similarity is correspondingly higher: dQ 0.9987 vs. 0.9880, dK 0.9993 vs. 0.9910, dV 0.9995 vs. 0.9955. Downstream fine-tuning accuracy (Table 8) confirms INT8 superiority: Qwen2.5-1.5B GSM8K 0.5232 vs. 0.5031, MMLU 0.4934 vs. 0.4689; Qwen2.5-3B GSM8K 0.5945 vs. 0.5868, MMLU 0.6032 vs. 0.5907. The INT8 advantage is consistent but modest — 1-2 percentage points on most comparisons — and the paper attributes it to INT8's uniform quantization grid providing more consistent error characteristics than FP8's variable spacing.

**Per-layer error accumulation (Table 15):** Quantization error accumulates across layers in CogVideoX-2B — Layer 1 has L1 error 0.0076, Layer 10: 0.0922, Layer 20: 0.1146, and then Layer 30 shows a decrease to 0.0571, suggesting partial error cancellation in deeper layers. By keeping the three most sensitive layers in FP16, per-layer L1 error is reduced (Layer 10: 0.0447, Layer 20: 0.0773, Layer 30: 0.0429), confirming that targeted precision preservation at error-prone layers can mitigate accumulation without sacrificing the speedup from FP4 in other layers. The paper does not report whether this layer-selective strategy was used in the main experiments, leaving its practical adoption ambiguous.

**SageBwd multi-seed robustness (Appendix Tables 9-14):** For Qwen2.5-1.5B, Qwen2.5-3B, and Llama3.2-1B across all four benchmarks, the standard deviation of SageBwd and BF16 fine-tuning results across five seeds are consistently similar — for example, GSM8K on Qwen2.5-1.5B: SageBwd std 0.0090, BF16 std 0.0089 (Table 9). This confirms that SageBwd does not increase training variance relative to BF16, an important property for practical deployment where deterministic reproducibility across runs matters.

**ReST^EM revision model experiment:** This paper does not include the ReST^EM experiment from the reference example — that was a different paper. The SageAttention3 paper does not have revision models, search strategies, or the compute-optimal allocation framework described in the reference example.

### Critical Assessment

**Does the paper demonstrate that FP4 attention achieves 1038 TOPS on RTX 5090?** Yes, this is directly measured and reported in Figure 4. The measurement is at a specific configuration (head_dim=128, causal=False, seq_len=2K) and the paper provides throughput curves across a range of sequence lengths and head dimensions. The claim is well-supported for the benchmarked configurations. However, the paper does not report kernel launch overhead, does not discuss whether the 1038 TOPS figure is sustained or peak, and does not provide variance information across multiple runs. In CUDA kernel benchmarking, it is standard to report the best of multiple warm-up-excluded runs, but the paper does not specify its measurement protocol.

**Does the paper demonstrate that FP4 attention preserves model quality in a "plug-and-play" manner?** Partially. Table 2 shows end-to-end metrics for five model families with SageAttention3 generally matching full-precision within metric noise. However:

- The metrics show non-trivial degradation on some specific measures: CogVideoX FScore drops from 4.78 to 4.04 (15.6%), and HunyuanVideo Flow-score drops from 1.48 to 1.23 (16.7%). These are not negligible differences — they may reflect genuine temporal consistency degradation that is not visible in single-frame examples. The paper does not discuss these specific regressions or explain why they are acceptable.

- The paper evaluates on a modest set of prompts (open-sora prompt set, COCO annotations) and does not report the number of evaluation samples. Without sample counts, it is impossible to assess whether the metric similarities are statistically reliable or could reflect sampling noise.

- The "plug-and-play" claim implies no model-specific tuning, which is supported — the same SageAttention3 kernel is applied across video, image, and language models — but the paper does not report whether different models required different quantization hyperparameters (block sizes, smoothing choices). If the same configuration works across all models, that is strong evidence for plug-and-play; if per-model tuning was required, the claim is weaker.

- There is no evaluation on language model inference quality (e.g., perplexity, downstream task accuracy after FP4 attention during inference). The language models (Qwen2.5, Llama3.2) are only used for SageBwd fine-tuning experiments, not for SageAttention3 inference quality evaluation. This is a notable gap — language modeling is the primary use case for attention optimization, and the paper provides no evidence that FP4 attention preserves language model quality.

**Does the paper demonstrate that 8-bit attention achieves "lossless" fine-tuning performance?** Yes, for the specific models, datasets, and fine-tuning configurations tested. Table 3 shows BF16 and SageBwd results within 0.013 of each other across all comparisons, and the multi-seed results (Tables 9-14) confirm that within-method variance exceeds between-method differences. The "lossless" claim is statistically well-supported for the tested configuration: 600-700 fine-tuning steps, learning rate 3e-5 with linear decay, batch sizes 32-128, on Qwen2.5 and Llama3.2 models.

However, "lossless" is a strong claim that requires careful boundary specification:

- **The claim applies to fine-tuning, not pretraining.** Figure 8(a) explicitly shows slower convergence for pretraining, and the paper is transparent about this limitation. But the abstract and introduction emphasize "lossless performance in fine-tuning tasks" — readers unfamiliar with the distinction between fine-tuning and pretraining may misinterpret the scope.

- **The claim applies to the specific training hyperparameters tested.** Different learning rates, optimizers, or training durations might expose larger gaps between BF16 and SageBwd. For instance, if the fine-tuning ran for 10,000 steps instead of 600-700, would the accumulated gradient errors eventually cause divergence? The paper does not investigate this.

- **The claim applies to models up to 3B parameters.** There is no evidence for or against scaling to larger models (7B, 13B, 70B+). Larger models might have different gradient statistics (e.g., larger dynamic range in certain layers) that make INT8 quantization more or less accurate. The paper's theoretical analysis (Appendix A.11) assumes specific distributional properties that may not hold at larger scales.

- **The claim applies to the tasks tested (GSM8K, DROP, MMLU, HELLASWAG).** These are standard benchmarks but represent a limited slice of possible fine-tuning applications — they do not include code generation, long-form text generation, multi-turn dialogue, or reinforcement learning from human feedback. The gradient characteristics of these tasks might differ.

**Does the paper demonstrate that combining SageBwd training with SageAttention3 inference improves accuracy?** The evidence is suggestive but thin. Table 5 shows the effect for Qwen2.5 at two scales on two benchmarks, with the improvement varying from 0.003 (Qwen2.5-3B, MMLU) to 0.032 (Qwen2.5-1.5B, GSM8K). With only four data points and no error bars, it is difficult to distinguish a genuine effect from sampling noise, especially since the multi-seed standard deviations (Tables 9-14) are on the order of 0.01 — comparable to or larger than some of the reported improvements. A proper evaluation would require multiple seeds for the combined training-inference pipeline and statistical testing.

Moreover, the paper's explanation ("INT8 and FP4 share a more similar representable data distribution") is speculative and not directly tested. Alternative explanations — such as SageBwd's quantization noise acting as beneficial regularization, or the specific checkpoint selected for evaluation being unusually favorable — are not ruled out. The paper presents this as an observation rather than a validated mechanism, which is appropriate, but readers should treat the "complementary benefit" claim as tentative.

**What experiments would have strengthened the paper?**

1. **Language model inference quality evaluation with FP4 attention.** Measuring perplexity or downstream task accuracy (e.g., on MMLU, GSM8K) after replacing FP16 attention with SageAttention3 during inference would directly address the most impactful use case for attention optimization. The absence of such evaluation is the paper's most significant empirical gap.

2. **Scaling SageBwd to larger models (7B+).** Fine-tuning experiments at the 1B-3B scale are informative but leave open whether the approach works at production scales. Larger models are both more likely to benefit from training acceleration (since attention cost grows with hidden dimension) and potentially more sensitive to gradient quantization errors (since they have more layers for errors to accumulate across).

3. **Ablation on fine-tuning duration.** Running SageBwd fine-tuning for 10K or 50K steps would test whether gradient errors accumulate over longer training horizons, potentially exposing divergence that short fine-tuning hides. This is particularly important given the pretraining convergence gap — there may be a fine-tuning duration threshold beyond which SageBwd and BF16 diverge.

4. **Comparison against FP8 FlashAttention3 training on H100.** Since FP8 attention training is the closest prior work (even if FlashAttention3's FP8 only supports forward pass), comparing INT8 SageBwd against an FP8 forward + FP16 backward hybrid on the same hardware would contextualize the contribution. The paper's INT8 vs. FP8 comparison (Section 5.4) only evaluates gradient accuracy and downstream fine-tuning quality, not training throughput.

5. **Measurement of FP4 attention's information-theoretic properties.** The paper shows that FP4 attention preserves end-to-end metrics but does not characterize *how much* information is lost in the attention computation itself. Measuring the mutual information between FP16 and FP4 attention outputs, or the rank correlation of attention maps, would provide a more principled understanding of when quality degradation is likely.

6. **Ablation on the 1×16 block size.** The choice of NVFP4 with 1×16 blocks is justified by comparison against MXFP4 with 1×32 blocks, but the 1×16 choice itself is not ablated against hypothetical alternatives (1×8, 1×4) because the hardware constrains the block size. This means we cannot distinguish whether 1×16 is optimal or merely sufficient — a finer granularity might further improve accuracy at the cost of scale factor storage overhead, but this tradeoff cannot be explored on Blackwell hardware.

7. **Evaluation on a broader range of sequence lengths for end-to-end quality.** The kernel speed is evaluated across 1K-32K, but end-to-end quality (Table 2) is reported without specifying sequence length. If the models were run at short sequences where FP4 error is less amortized, quality might degrade more than at long sequences where the attention computation dominates.

**Overall, the experimental evidence is strongest for the hardware efficiency claims** (kernel throughput measurements are precise, reproducible, and comprehensive across configurations) and **weaker for the quality preservation claims** (end-to-end metrics have unexplained regressions, no language model inference quality evaluation, and the multi-seed analysis is limited to fine-tuning). The paper transparently reports the negative pretraining result and the metric regressions, which is commendable, but the scope of quality evaluation does not fully match the generality of the "plug-and-play" claim. The SageBwd "lossless fine-tuning" claim is well-supported within the tested regime but carries implicit boundary conditions (model scale, training duration, task type) that are not yet mapped.

## 6. Limitations and Trade-offs

### FP4 Attention Is Not Evaluated on Language Model Inference Quality

**The assumption or constraint.** The paper validates SageAttention3's quality preservation exclusively on image and video generation models (CogVideoX, HunyuanVideo, Mochi, Flux, Stable-Diffusion3.5) and provides no evaluation of language model inference quality — perplexity, downstream task accuracy, or generation quality — when FP4 attention replaces FP16 attention during text generation. The language models used in the paper (Qwen2.5, Llama3.2) appear only in the SageBwd fine-tuning experiments (Section 5.2, Table 3), where the attention during inference remains at the precision used during training (BF16 for the quality baseline, not FP4). Section 5.1 lists Qwen2.5 and Llama3.2 among the "diverse set of representative models" but the end-to-end quality metrics in Table 2 include only video and image generation models — text-to-text evaluation is conspicuously absent.

**The consequence.** Language modeling is arguably the most impactful and widely-deployed use case for attention optimization, given the massive scale of LLM inference in production. Without evidence that FP4 attention preserves perplexity, generation quality, or reasoning accuracy on benchmarks like MMLU, GSM8K, or HumanEval, a practitioner cannot assess whether SageAttention3 is safe to deploy for text generation workloads. The failure mode is unknown but plausible: language model attention maps may have different structural properties than video diffusion attention maps (e.g., sharper attention distributions, more systematic head specialization), and the 99.55% cosine similarity measured on CogVideoX (Table 1) may not translate to acceptable perplexity degradation on long-form text generation. The 15.6% FScore drop on CogVideoX (Table 2, from 4.78 to 4.04) and the 16.7% Flow-score drop on HunyuanVideo (from 1.48 to 1.23) demonstrate that SageAttention3 *can* produce non-trivial quality regressions in practice, making the absence of language model evaluation a genuine risk rather than a hypothetical concern.

**What evidence exists in the paper.** There is none. The paper neither reports language model inference quality metrics nor explains why they are omitted. The kernel speed evaluation (Figures 4-5) does not distinguish between model types, and the visible examples (Figures 9-14) cover only image and video generation. The "plug-and-play" claim in the abstract — "our FP4 attention can accelerate inference of various models in a plug-and-play way" — implicitly includes language models among "various models," but this claim is untested for the most important model category.

**Mitigation status.** Not addressed. The paper does not acknowledge this gap, propose language model evaluation experiments, or discuss why the quality preservation shown for diffusion models might or might not transfer to autoregressive language models. Given that the paper's primary stated contribution is "FP4 attention for inference acceleration" (Section 1) without any domain restriction, this is a significant omission that limits the deployability assessment for the largest category of potential users.

---

### Difficulty Estimation Cost Is Not Accounted for in the Practical Deployment Pipeline

**The assumption or constraint.** While this paper is not about compute-optimal allocation or difficulty estimation (those concepts belong to the reference example paper), SageAttention3 *does* assume that the attention computation can be quantized to FP4 without any per-model or per-input adaptation. However, the paper's own results reveal that quantization error accumulates differently across layers (Appendix A.7, Table 15) and that keeping three sensitive layers in FP16 substantially reduces accumulated error. This implies that optimal deployment may require **per-model or per-layer precision assignment** — deciding which layers can safely use FP4 and which need higher precision — which the paper does not provide an automated method for. The layer-wise error analysis in Table 15 uses oracle knowledge (measuring per-layer L1 error against full-precision outputs) to identify the three most sensitive layers, but this requires running the model at full precision first and measuring error — a cost that is not amortized in the paper's deployment pipeline.

**The consequence.** A practitioner deploying SageAttention3 on a new model cannot know, without running a full-precision baseline and measuring per-layer error, whether FP4 attention will be safe for all layers or whether certain layers need to be kept at higher precision. The reported end-to-end metrics (Table 2) may reflect the fortuitous case where FP4 is adequate for all layers in the tested models, but this is not guaranteed for new architectures, different training procedures, or models with different activation statistics. The failure mode is silent degradation: FP4 attention might preserve quality on most prompts but produce occasional catastrophic failures on inputs that trigger attention patterns where FP4 quantization error is amplified in error-sensitive layers. Without a method for identifying sensitive layers without oracle access, deployment involves an uncomfortable choice between risking unknown quality degradation or foregoing the speedup entirely.

**What evidence exists in the paper.** Table 15 shows that per-layer L1 error varies dramatically across CogVideoX-2B layers — from 0.0076 at Layer 1 to 0.1146 at Layer 20, a 15× range — and that keeping the three most error-prone layers in FP16 reduces accumulated error significantly. The smoothing ablation (Table 16) shows that different attention operations (Q smoothing vs. K smoothing) have different individual contributions to accuracy, suggesting that sensitivity is not uniform across the attention computation. These results imply that per-layer or per-operation precision decisions could improve the accuracy-speed tradeoff, but the paper provides no systematic method for making these decisions without full-precision baseline access.

**Mitigation status.** Partially acknowledged but unresolved. The paper presents the layer-wise error analysis in the Appendix (Table 15) and notes that "keeping the three layers with the largest observed error growth in FP16 precision... significantly reduces the overall error accumulation," but does not elevate this to a main-paper finding or propose an automated sensitivity detection method. The main experiments (Table 2) appear to use FP4 for all layers without selective precision preservation, so the reported quality metrics represent the uniform-FP4 case. The paper does not discuss whether the sensitive-layer strategy was used in the main results, leaving its practical adoption ambiguous.

---

### The 8-Bit Training Approach Fails on Pretraining, With No Clear Path to Remediation

**The assumption or constraint.** SageBwd is designed and evaluated primarily for fine-tuning, where models start from strong pretrained initializations and make relatively small parameter updates over modest numbers of steps (600-700 steps in the paper's experiments). The paper explicitly acknowledges that pretraining convergence is slower with SageBwd (Section 5.2): "SageBwd can achieve loss convergence, its convergence speed is relatively slow. This limitation restricts its applicability in pretraining tasks." However, the paper does not diagnose *why* pretraining fails — whether it is gradient error accumulation over many steps, sensitivity to the specific quantized operation ($dOV^T$), interaction between quantization noise and the learning rate schedule, or a fundamental precision floor that INT8 cannot satisfy for learning representations from scratch.

**The consequence.** The "lossless fine-tuning" result is valuable but leaves the most impactful training application — pretraining large models from scratch — unsolved. Given that pretraining dominates the total computational cost of most LLM development pipelines (fine-tuning represents a tiny fraction of total FLOPs), a training acceleration method that only works for fine-tuning addresses a relatively small portion of the total training compute budget. Moreover, without understanding *why* pretraining fails, there is no clear research direction for closing the gap — is the problem fixable with better quantization strategies, or is it fundamental to 8-bit precision? The paper's finding that INT8 outperforms FP8 for gradient computation (Tables 6-8) suggests that the choice of quantization format matters, but does not indicate whether either format could eventually be made to work for pretraining.

**What evidence exists in the paper.** Figure 8(a) shows the pretraining loss curve on FineWeb-Edu for a Llama-400M model. The SageBwd loss is consistently higher than BF16 at every step, with a visible gap that does not close by step 20,000. The paper does not quantify the gap numerically, does not report final validation perplexity or downstream task performance for the pretrained models, and does not test whether the gap would close with extended training or different hyperparameters. The multi-seed fine-tuning results (Tables 9-14) demonstrate that within-method variance across seeds is comparable to the BF16-SageBwd gap, but this same analysis is not performed for pretraining — we do not know whether the pretraining convergence gap is statistically robust or could be closed by hyperparameter tuning.

**Mitigation status.** Acknowledged but not investigated. The paper lists "investigating the application of low-bit attention in pretraining tasks" as future work (Section 7) but provides no diagnostic experiments or hypotheses about the failure mechanism. The theoretical analysis in Appendix A.11, which shows that $E[\Delta dQ^{(1)}] \neq 0$ for the quantized $dOV^T$ term, provides a plausible mechanism for systematic gradient bias, but this analysis is not connected to the pretraining result — it is presented as justification for keeping $dOV^T$ in FP16, not as an explanation for why the remaining quantization (four of five backward GEMMs) is still insufficient for pretraining convergence. The gap between "good enough for fine-tuning" and "good enough for pretraining" is identified but not explained.

---

### End-to-End Quality Metrics Show Meaningful Regressions That Are Not Discussed

**The assumption or constraint.** The paper claims that SageAttention3 "almost incurs no end-to-end quality loss across these models" (Section 5.2) and presents the metrics in Table 2 as evidence for this claim. However, several of the reported metrics show non-trivial degradation that the paper does not acknowledge or explain: CogVideoX FScore drops from 4.78 (full precision) to 4.04 (FP4), a 15.6% relative decrease; HunyuanVideo Flow-score drops from 1.48 to 1.23, a 16.7% decrease; Mochi VQA-t drops from 61.00 to 59.43, a 2.6% decrease; and Mochi FScore drops from 1.80 to 1.65, an 8.6% decrease. These are not negligible differences within the precision of these metrics — they represent measurable quality degradation in temporal consistency (FScore/Flow-score) and technical quality (VQA-t) that a practitioner would want to understand before deploying.

**The consequence.** The "no quality loss" framing, while directionally supported by the majority of metrics, obscures the fact that FP4 attention *does* degrade certain quality dimensions in certain models. A video generation practitioner who cares primarily about temporal consistency (arguably the key quality dimension for video) might reasonably decide that a 15% FScore drop is unacceptable, even if other metrics like CLIPSIM are preserved. The paper does not help this practitioner make that decision because it does not analyze *why* temporal consistency metrics degrade — is it due to accumulated error across frames? Sensitivity of certain attention heads to quantization? Interaction between FP4 attention and the temporal attention mechanism? Without this analysis, the degradation appears as an unexplained risk rather than a understood and potentially mitigable tradeoff.

**What evidence exists in the paper.** Table 2 contains the relevant numbers, and the visible examples (Figures 9, 13, 14) show video frames that appear visually identical — but frame-level visual similarity does not guarantee temporal consistency, since FScore measures motion smoothness across frames rather than individual frame quality. The per-layer error accumulation analysis (Table 15) shows that error grows with layer depth, which could explain temporal consistency degradation if later layers in video models are responsible for cross-frame coherence. However, the paper does not connect these dots or analyze attention error specifically in temporal attention layers versus spatial attention layers.

**Mitigation status.** Not addressed. The paper does not acknowledge the specific metric regressions, does not discuss why temporal consistency metrics degrade more than text-alignment metrics, and does not investigate whether selective precision preservation (keeping certain layers or attention operations in FP16) would recover the lost quality. The visible examples are a necessary but insufficient quality check — they demonstrate that the worst-case behavior (catastrophic visual artifacts) is avoided, but they do not demonstrate that subtle temporal quality degradation is absent.

---

### The Method Is Evaluated on a Narrow Set of Hardware and Lacks Cross-Architecture Validation

**The assumption or constraint.** SageAttention3 is evaluated exclusively on the RTX 5090 GPU, exploiting the FP4 Tensor Cores specific to NVIDIA's Blackwell architecture. The paper acknowledges this implicitly — the entire FP4 contribution is predicated on Blackwell hardware availability — but does not discuss how the approach would transfer to other architectures that support FP4 or lower precision (e.g., future AMD or Intel GPUs, previous-generation NVIDIA hardware, or non-GPU accelerators). SageBwd is evaluated on the RTX 4090 (Ada Lovelace architecture), with the paper noting that INT8 was chosen partly for "wider hardware support" including AMD MI250 and Ascend 910B (Section 5.4) — but SageBwd is not actually evaluated on any non-NVIDIA hardware.

**The consequence.** The headline 1038 TOPS figure is specific to one GPU model from one vendor, and the 5× speedup over FlashAttention2 is relative to the FP16 Tensor Core throughput on that specific GPU. A practitioner with H100 GPUs (which lack FP4 Tensor Cores) cannot use SageAttention3 at all — they are limited to FlashAttention3's FP8 attention (890 TOPS on H100, per Table 18) or SageAttention2's INT4 attention (885 TOPS on H100). This creates a fragmented deployment landscape where the optimal attention kernel depends on which GPU generation is available, and the paper provides no guidance for choosing between FP4, FP8, and INT4 attention on hardware that supports multiple formats. More subtly, the kernel optimizations described in Section 3.3 (permutation for K to match FP4 accumulator layout, producer-warp epilogue scheduling) are specific to the Blackwell FP4MMA instruction's register layout and may not transfer to future architectures with different low-precision matrix multiply implementations. The paper's code is open-source and could be ported, but the specific optimizations that achieve 65% of theoretical peak throughput may need to be redesigned for each new hardware target.

**What evidence exists in the paper.** Table 18 provides a helpful summary of the speed-accuracy tradeoffs across methods on different GPUs (RTX 5090 and H100), showing that FlashAttention3-FP8 achieves 890 TOPS on H100 while SageAttention3 achieves 1038 TOPS on RTX 5090 — but these numbers are on different hardware and cannot be directly compared. The paper does not evaluate SageAttention3 on H100 (where it could not achieve FP4 throughput since H100 lacks FP4 Tensor Cores, but an FP8 variant might be possible), does not evaluate SageBwd on H100 or A100, and does not test on AMD or Intel hardware despite claiming INT8's portability as an advantage. The kernel speed experiments (Figures 4-7) are all single-GPU measurements at specific CUDA/cuDNN versions without discussion of how performance might vary across driver versions, GPU instances, or cloud vs. bare-metal environments.

**Mitigation status.** Partially acknowledged through the hardware support discussion in Section 5.4 ("INT8 is supported on almost all modern GPUs... while FP8 support remains limited to newer architectures"), but not with respect to the FP4 contribution. The paper does not claim cross-architecture portability for SageAttention3, and the focus on Blackwell is appropriate for a paper introducing FP4 attention — Blackwell GPUs are the only hardware that supports FP4 Tensor Cores at the time of writing. However, the absence of SageBwd evaluation on non-NVIDIA hardware, despite claiming portability as an advantage, is a gap between claim and evidence. The open-source code release partly mitigates this by enabling community porting, but without baseline numbers on other architectures, practitioners cannot assess SageBwd's portability in practice.

---

### The Difficulty Estimation and Compute-Optimal Allocation Framework Is Absent

**The assumption or constraint.** This paper is purely a systems/kernel contribution and does not incorporate the difficulty estimation, compute-optimal allocation, or adaptive strategy selection that characterizes the reference example paper. The reader should not expect SageAttention3 or SageBwd to adapt their quantization strategy based on input difficulty, sequence length, or model layer — both methods apply the same quantization scheme (FP4 microscaling for SageAttention3, INT8 per-block for SageBwd) uniformly across all inputs, layers, and models. The paper does not claim otherwise, but the absence of adaptivity means that the speed-accuracy tradeoff is fixed and cannot be tuned per-input: a short, easy-to-process sequence pays the same quantization overhead as a long, difficult sequence.

**The consequence.** In deployment scenarios where input difficulty varies widely — e.g., a video generation service that receives both simple 2-second clips and complex 30-second scenes, or a language model serving both factual lookups and multi-step reasoning queries — the uniform quantization strategy leaves potential efficiency on the table. Easy inputs might tolerate even more aggressive quantization (e.g., FP4 with larger block sizes, or skipping certain smoothing operations), while difficult inputs might benefit from selective precision preservation (keeping attention in FP16 for the first few layers, as the layer-wise error analysis in Table 15 suggests). Without adaptivity, the system must either use FP4 universally (risking quality degradation on difficult inputs) or use FP16 universally (leaving throughput gains unrealized on easy inputs). The paper's kernel-level speed measurements show that throughput is relatively flat across sequence lengths (Figure 4: 964 TOPS at 1K vs. 1038 TOPS at 32K), suggesting that the fixed quantization overhead is well-amortized across scales, but this does not address whether output *quality* could be improved by adaptive precision.

**What evidence exists in the paper.** The layer-wise error analysis (Table 15) and the smoothing ablation (Table 16) provide indirect evidence that different layers and operations have different quantization sensitivity — some layers accumulate more error than others, and some smoothing techniques contribute more to accuracy than others. This is the kind of heterogeneity that an adaptive strategy would exploit, but the paper does not propose or evaluate any adaptive mechanism. The "plug-and-play" framing (Section 1) suggests that uniform application is a feature (simplicity, no per-model tuning), but it also represents a ceiling on the achievable quality-speed Pareto frontier.

**Mitigation status.** Not addressed and not claimed as a contribution. This limitation is inherent to the paper's scope as a kernel optimization work rather than a test-time compute allocation work, and it would be unfair to criticize the paper for not doing something it does not set out to do. However, practitioners evaluating SageAttention3 against alternatives should understand that the method provides a fixed precision-speed operating point, not a tunable tradeoff, and that future work on adaptive precision (combining the kernel contributions of this paper with the allocation framework of works like the reference example) could yield further improvements.

## 7. Implications and Future Directions
- Field impact:
  - Demonstrates that FP4 attention can be made accurate and “plug‑and‑play” for inference on Blackwell GPUs, substantially shifting the speed‑accuracy frontier (Figs. 4–5; Table 2). Establishes a practical blueprint for training-time low‑bit attention (SageBwd) by identifying the single most sensitive backward matmul to keep high‑precision (Table 1c; Appendix A.11).
- Follow‑up research:
  - Toward fully low‑bit training: Replace or robustify `dOV^T` computation via error‑resilient quantization, improved scaling schemes, or gradient‑aware training (Appendix A.11 provides a starting analysis).
  - Better pretraining: Investigate quantization‑aware training schedules, adaptive precision policies per layer/length, or alternative softmax parameterizations to recover convergence speed (Fig. 8a).
  - Autotuning mixed precision: Automatically select layers and tiles to run in FP16 vs FP4/INT8 based on runtime statistics (Appendix A.6 indicates only a few layers may need FP16).
  - Broader ops and platforms: Extend microscaling FP4 to other transformer ops (MLP/GEMM/Conv) and to other vendors once compatible low‑precision units are available.
- Practical applications:
  - Faster generation for text‑to‑video and diffusion (Table 4a; Fig. 1, Fig. 9), lower inference cost for LLMs, and faster fine‑tuning cycles for downstream tasks (Table 4b; Table 3). The combined recipe—INT8 fine‑tune + FP4 inference—can yield the best of both worlds in accuracy and speed (Table 5).

> Bottom line: By solving two long‑standing precision bottlenecks—accurate FP4 attention for inference and a practically stable low‑bit backward path for training—this work opens a clear path to large, real‑world speedups without sacrificing quality.
