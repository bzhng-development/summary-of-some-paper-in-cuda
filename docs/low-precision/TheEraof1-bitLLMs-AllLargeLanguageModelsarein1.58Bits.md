# The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits

**ArXiv:** [2402.17764](https://arxiv.org/abs/2402.17764)

## 🎯 Pitch

This paper introduces BitNet b1.58, a Transformer-based large language model where every weight is constrained to the ternary set {−1, 0, +1}, achieving state-of-the-art language modeling performance at just 1.58 bits per weight. By enabling highly efficient inference—dramatically reducing memory, latency, and energy costs compared to traditional 16-bit models—while matching full-precision accuracy at scale, BitNet b1.58 paves the way for scalable, sustainable, and edge-friendly AI deployments and inspires new hardware optimized for ultra-low-bit computation.

---

## 1. Executive Summary (2-3 sentences)
This paper introduces `BitNet b1.58`, a Transformer-based large language model (LLM) where every weight is ternary `{−1, 0, +1}` (“1.58-bit” weights) and activations are 8-bit. It achieves comparable language modeling quality to full-precision FP16/BF16 LLaMA-style baselines starting at 3B parameters while dramatically reducing inference latency, memory footprint, throughput cost, and estimated energy—enabling a new compute paradigm with almost no multiplications (Fig. 1; Tables 1–3; Fig. 2–3).

## 2. Context and Motivation
- Problem addressed
  - Inference for modern LLMs is expensive in memory, latency, and energy because matrix multiplications in FP16/BF16 dominate cost. Memory bandwidth for loading weights from DRAM is a major bottleneck (Sec. 1).
  - Post-training quantization (PTQ) to 4–8 bits reduces costs but is suboptimal in accuracy and still relies on floating-point multiplies during matmul (Sec. 1, with prior work [XLS+23, FAHA23, LTT+23, CCKS23, TCS+24]).

- Why it matters
  - Power is the limiting factor for performance in many chips; reducing arithmetic energy accelerates computation (Sec. 1, citing [Hor14]). Smaller weights also reduce DRAM bandwidth and capacity needs, improving both throughput and latency in deployed systems (Sec. 1).

- Prior approaches and gaps
  - BitNet (1-bit) architectures remove multiplications by constraining weights to ±1, but modeling capacity can suffer and there is no explicit mechanism to “ignore” features (weight=0) (Sec. 1–2).
  - PTQ methods compress after training; they retain floating-point multiplications and require calibration procedures; accuracy degradation can be dataset/model-size dependent.

- Positioning of this work
  - `BitNet b1.58` is trained from scratch with ternary weights `{−1, 0, +1}` and 8-bit activations, preserving the no-multiplication paradigm of 1-bit transformers while adding a zero weight for feature filtering and better modeling capacity (Sec. 2). It adopts LLaMA-like architectural components for compatibility with common LLM tooling (Sec. 2).

## 3. Technical Approach
High-level mechanism: replace every dense `nn.Linear` layer in a Transformer with `BitLinear`, whose weights are constrained to ternary values `{−1, 0, +1}`. During inference, multiplying by −1/0/+1 reduces to sign flips, zeroing, and additions; matrix multiplication no longer needs floating-point multiplications (Fig. 1; Sec. 1–2).

- Weight quantization to 1.58 bits (“ternary”)
  - Core idea: scale the full-precision weight matrix by the mean absolute value and then round-and-clip to {−1, 0, +1}.
  - In plain language: compute a single scalar `γ` per weight matrix equal to the average magnitude of its entries; divide all weights by `γ`, round each to the nearest integer in [−1, 1], and clip to this range to avoid overflow.
  - Formalization (Sec. 2, Eqs. (1)–(3)):
    - `γ = (1/(nm)) Σ_ij |W_ij|` (Eq. 3)
    - `W_f = RoundClip(W/(γ+ε), −1, 1)` (Eq. 1)
    - `RoundClip(x, a, b) = max(a, min(b, round(x)))` (Eq. 2)
  - Rationale for this design:
    - Single-scale “absmean” normalization stabilizes ternarization by matching each weight matrix’s dynamic range to the 3-level codebook. Adding `0` to the original ±1 codebook enables explicit “feature filtering,” i.e., the model can drop unhelpful signals via zero weights (Sec. 1–2).

- Activation quantization to 8 bits
  - Activations are quantized to 8 bits with symmetric per-token scaling to `[-Q_b, Q_b]`, removing zero-point quantization (Sec. 2).
  - Why symmetric and per-token? It simplifies implementation and system-level optimization and, in experiments, has “negligible effects” on performance (Sec. 2). Symmetry also matches the ternary, zero-centered weight distribution.

- Computation model and kernels
  - Because weights are ternary, matrix multiplication reduces to integer additions and sign flips; multiplications are largely eliminated (Sec. 1–2; Fig. 1).
  - In GPU experiments, a 2-bit kernel from Ladder [WMC+23] is used to implement fast low-bit operations (Sec. 3), indicating the system-level feasibility even without specialized hardware.

- Architecture and training recipe
  - `BitNet b1.58` adopts LLaMA-like modules: `RMSNorm` [ZS19], `SwiGLU` [Sha20], rotary positional embeddings [SAL+24], and removes all biases (Sec. 2).
  - It is trained from scratch on the same data and tokens as full-precision baselines to ensure matched comparisons (Sec. 3: 100B tokens on RedPajama [Com23]; and a separate 2T-token experiment against StableLM-3B [TBMR]).

- Implementation friendliness
  - The LLaMA-alike design means it can be integrated with HuggingFace, vLLM [KLZ+23], and llama.cpp with minimal changes (Sec. 2).

Analogy: think of each weight as a tiny switch with three positions: “pass the signal” (+1), “invert the signal” (−1), or “ignore it” (0). With only these switches, the network avoids expensive multiplication, and computing matrix-vector products becomes flipping signs and summing integers.

## 4. Key Insights and Innovations
- Ternary weights with explicit zero (“1.58-bit”) as a Pareto improvement
  - What’s new: extends 1-bit `{−1, +1}` weights to ternary `{−1, 0, +1}` using a simple absmean scaling and rounding (Sec. 2).
  - Why it matters: keeps the no-multiplication paradigm and low memory footprint while improving modeling capacity via explicit “feature filtering” (zeros). The paper reports accuracy parity with FP16 starting from 3B parameters (Tables 1–2), which is a qualitative step beyond prior 1-bit results.

- Symmetric per-token 8-bit activation quantization
  - What’s new: activations are scaled to `[-Q_b, Q_b]` per token, removing zero points (Sec. 2).
  - Why it matters: simplifies hardware/software (no asymmetric zero-point handling) with negligible measured accuracy impact, and halves activation precision vs FP16—important for KV cache memory in long-context inference (Sec. 4).

- End-to-end system-level efficiency with measured speedups
  - What’s new: demonstrates practical inference benefits on GPUs using an existing low-bit kernel (Sec. 3), not just theoretical energy models.
  - Why it matters: Latency, memory, batch size, and throughput improvements are large and grow with model size (Fig. 2; Table 3), making the approach immediately relevant for serving.

- Energy scaling perspective and “new scaling law” equivalences
  - What’s new: estimates 71.4× lower arithmetic energy for matmul on 7nm and shows end-to-end energy advantages that increase with model size (Fig. 3).
  - Why it matters: It reframes how to scale LLMs—bigger models may be cheaper to deploy if weights are 1.58-bit, leading to equivalence rules of thumb (e.g., 70B 1.58-bit ≈ 13B FP16 in cost; Sec. 3, “BitNet b1.58 is enabling a new scaling law…”).

Fundamental innovations: the ternary no-multiplication compute paradigm at LLM scale and the demonstration of accuracy parity at multi-billion parameter scale. Incremental but important: the specific quantization choices (absmean, symmetric per-token) and LLaMA-compatible design for ecosystem adoption.

## 5. Experimental Analysis
- Setup
  - Pretraining: Both `BitNet b1.58` and reproduced LLaMA FP16 baselines are trained on RedPajama for 100B tokens to match data/tokens (Sec. 3).
  - Metrics: Validation perplexity on WikiText2 and C4; zero-shot accuracy on ARC-Easy/Challenge, HellaSwag, Winogrande, PIQA, OpenBookQA, BoolQ using lm-evaluation-harness (Sec. 3).
  - System measurements: GPU runtime memory and latency using FasterTransformer with Ladder’s 2-bit kernel; time per output token (Sec. 3). Throughput measured on two 80GB A100s with pipeline parallelism (Sec. 3; Table 3).
  - Energy: Estimated using component-wise arithmetic energy on 7nm (Horowitz model) and reported end-to-end energy trends (Fig. 3).

- Main quantitative results
  - Accuracy and perplexity parity starting at 3B:
    - Perplexity (Table 1): at 3B, `BitNet b1.58` PPL = 9.91 vs LLaMA FP16 = 10.04; at 1.3B and 700M, small gaps remain.
    - Zero-shot accuracy (Table 2): at 3B, average 50.2 vs 49.7 (BitNet higher).
  - Latency and memory (Fig. 2; Table 1):
    - At 3B: latency 2.71× faster (1.87 ms vs 5.07 ms); memory 3.55× lower (2.22 GB vs 7.89 GB) while matching perplexity (Table 1).
    - Scaling trends (Fig. 2, left/right): speedup grows with size—1.67× (1.3B), 2.90× (7B), 3.68× (13B), 4.10× (70B); memory reductions similarly grow—2.93×, 4.40×, 5.12×, 7.16×.
  - Throughput and batch size (Table 3):
    - 70B models on 2×A100: max batch size 176 vs 16 (11.0×), throughput 2977 vs 333 tokens/s (8.9×).
  - Energy (Fig. 3):
    - Arithmetic energy composition: majority INT8 adds for `BitNet b1.58`; FP16 adds and multiplies for LLaMA. Estimated 71.4× arithmetic energy savings for matmul on 7nm.
    - End-to-end energy advantage increases with model size: 18.6× (1.3B) up to 41.2× (70B).
  - More data (Table 4, 2T tokens):
    - Against StableLM-3B (2T tokens), `BitNet b1.58 3B` scores higher on all five tasks; average 74.34 vs 73.22.

- Representative citations of results
  > Table 1: “BitNet b1.58 3B … Latency 1.87 ms (2.71×), Memory 2.22 GB (3.55×), PPL 9.91 vs LLaMA 3B PPL 10.04.”

  > Table 2: “BitNet b1.58 3B: Avg. 50.2 vs LLaMA 3B: 49.7.”

  > Figure 2: “BitNet b1.58 70B is 4.1× faster … memory 7.16× lower.”

  > Table 3: “70B: Max batch 176 (11.0×), Throughput 2977 tokens/s (8.9×).”

  > Figure 3: “71.4× arithmetic ops energy savings; end-to-end energy 18.6× to 41.2× across sizes.”

- Do the experiments support the claims?
  - For equal-size comparisons on matched data/tokens, the 3B and larger results convincingly show quality parity or superiority with strong system-level wins (Tables 1–2, Fig. 2–3).
  - Throughput and memory experiments use established codebases (FasterTransformer; Ladder kernel), lending credibility to the reported speedups on current GPUs (Sec. 3).
  - Energy is estimated (not measured) based on a standard model; trends are plausible but depend on hardware assumptions (Fig. 3).

- Ablations and robustness
  - The excerpt does not report ablations isolating the impact of: absmean vs alternative quantizers, per-token activation scaling, or the contribution of the zero state (0) to accuracy.
  - No instruction-following or chat fine-tuning evaluations are reported here; evaluations are zero-shot on common benchmarks plus perplexity.

- Conditions and trade-offs observed
  - At small sizes (700M–1.3B), `BitNet b1.58` lags slightly in perplexity and average zero-shot accuracy (Tables 1–2), but the gap closes and reverses by 3B. System gains exist at all scales.

## 6. Limitations and Trade-offs
- Training from scratch instead of post-training quantization
  - The approach relies on full pretraining with ternary weights and 8-bit activations. It does not address converting an existing FP16 model to 1.58-bit with minimal retraining (Sec. 2–3).

- Embeddings remain full-precision
  - Memory and speed improvements increase with size partly because embeddings (kept in full precision) occupy a smaller fraction of total parameters in larger models (Sec. 3, Memory discussion). This slightly limits benefits at small scales.

- Hardware dependencies and energy estimates
  - Latency and memory gains are measured on GPUs, but the largest energy savings are extrapolated from a 7nm energy model (Fig. 3 left). Real-world energy depends on platform specifics and system components beyond arithmetic (Fig. 3 right suggests non-matmul costs are non-negligible at smaller scales).

- Limited task coverage and analysis
  - Accuracy comparisons focus on perplexity and a standard zero-shot suite. There is no report here on instruction tuning, safety/alignment evaluations, multilingual performance, robustness to distribution shift, or long-context reasoning beyond KV-cache memory implications.

- Small-model performance
  - Below 3B parameters, `BitNet b1.58` shows small quality deficits vs FP16 (Tables 1–2). The ternary constraint may hurt expressivity when capacity is limited.

- Implementation specifics not deeply detailed
  - The excerpt does not specify optimizer, learning-rate schedules, or gradient quantization. Practical stability and convergence details for training such low-bit models at scale remain an area where guidance would help adoption.

## 7. Implications and Future Directions
- How this changes the landscape
  - A credible path to “no-multiplication” LLM inference: by achieving parity with FP16 at 3B and above, 1.58-bit weights plus 8-bit activations shift the cost-quality frontier, enabling larger models to be served at the cost of much smaller FP16 models (Sec. 3, “new scaling law”).
  - It invites hardware specialization: integer-add-dominant matmuls with ternary weights motivate accelerators optimized for bit-serial or lookup-based arithmetic (Fig. 1; Sec. 4 “New Hardware for 1-bit LLMs”).

- Follow-up research enabled or suggested
  - Quantization methodology
    - Ablate and optimize quantizers (e.g., per-channel vs per-matrix scaling; alternatives to absmean; learned codebooks).
    - Explore activation precision further; the paper notes potential lossless compression of activations to 4 bits “or even lower” for KV caches (Sec. 4).
  - Training algorithms
    - Study optimizers and regularizers tailored for ternary weights; investigate gradient quantization and straight-through estimators at scale.
    - Investigate curriculum or distillation methods to transfer from FP16 checkpoints to 1.58-bit with minimal retraining.
  - Architectures
    - 1.58-bit Mixture-of-Experts (MoE) to cut FLOPs while also reducing memory and communication overhead (Sec. 4). With smaller parameters and activations, more experts may fit on a single device.
    - Long-sequence models: 8-bit (and potentially 4-bit) KV caches directly address the main practical barrier for long-context inference (Sec. 4).
  - Systems and hardware
    - Integrate low-bit matmuls into mainstream inference stacks (e.g., vLLM, Triton kernels) and CPUs for edge/mobile, where the paper argues 1.58-bit is “more friendly” (Sec. 4).
    - Co-design with new hardware (e.g., LPUs), building dataflows specialized for ternary weights and integer additions (Sec. 4).

- Practical applications
  - Cost-effective deployment of larger models for chatbots, code assistants, and enterprise search where latency and energy budgets are strict.
  - Edge and mobile on-device inference with significantly lower memory and power, enabling privacy-preserving or offline use cases (Sec. 4).
  - High-throughput batch serving: 8.9× throughput at 70B on 2×A100 (Table 3) suggests strong benefits for large-scale API providers.

In summary, `BitNet b1.58` demonstrates that LLMs with ternary weights and 8-bit activations can match full-precision accuracy at multi-billion parameter scales while delivering large, increasing system-level gains. The combination of a simple absmean ternarization, symmetric per-token activation quantization, and LLaMA-compatible design makes this a compelling direction for both immediate deployment and future hardware-software co-design.
