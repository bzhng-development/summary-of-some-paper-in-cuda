# AWQ: Activation-aware Weight Quantization for On-Device LLM Compression and Acceleration

**ArXiv:** [2306.00978](https://arxiv.org/abs/2306.00978)

## 🎯 Pitch

AWQ introduces a novel, hardware-friendly quantization method that compresses large language models to 4-bit weights by using activation statistics to identify and protect a tiny subset of essential weights via per-channel scaling—delivering minimal accuracy loss. Coupled with the TinyChat inference system, this approach enables 3×–4× real-world speedups and democratizes state-of-the-art LLM and multimodal model deployment on edge devices (even with scarce memory like Raspberry Pi), unlocking private, low-latency AI for widespread, resource-constrained applications.

---

## 1. Executive Summary
AWQ (Activation-aware Weight Quantization) is a post‑training method that compresses large language models (LLMs) to 4-bit weights while preserving accuracy and yielding practical speedups on edge devices. Its core idea is to identify and protect a tiny fraction of “salient” weight channels using activation statistics and a mathematically motivated per‑channel scaling, and to pair this with TinyChat, a lightweight inference system that turns the 4× memory reduction into 3×–4× measured speedups (Figures 3 and 9).

## 2. Context and Motivation
- Problem addressed
  - On-device LLM inference is desirable for privacy, latency, and cost, but is limited by model size and memory bandwidth. A 175B‑parameter model occupies hundreds of GB in FP16 (see Introduction), far beyond edge hardware. Low‑bit weight-only quantization (e.g., W4A16: 4‑bit weights, 16‑bit activations) reduces memory and I/O but often degrades accuracy at very low bit‑widths.

- Why it matters
  - Edge use cases (assistants, chatbots, robotics) operate at batch size 1; generation is memory‑bound (roofline analysis in Figure 3b), so reducing weight bandwidth directly improves latency. Deploying strong models on laptops, mobile GPUs, or even Raspberry Pi expands accessibility (Figures 1 and 10).

- Prior approaches and gaps
  - QAT (quantization-aware training) is expensive and hard to scale to LLMs (Related Work).
  - PTQ (post‑training quantization) methods like RTN (round-to-nearest) are simple but lose accuracy at 3–4 bits.
  - GPTQ (Frantar et al., 2022) reconstructs weights using second‑order information but can overfit the calibration set and become brittle across domains or modalities (Figure 8b). It also needs implementation tricks (reordering) on some models and still struggles at 2‑bit (Table 9).

- Positioning
  - AWQ keeps the simplicity and data‑efficiency of PTQ, avoids backpropagation/reconstruction, and focuses on protecting the few most influential weight channels based on activation scale (Sections 3.1–3.2, Table 1). It adds a system layer (TinyChat) to realize the theoretical benefits in real‑world runtimes (Section 4).

## 3. Technical Approach
AWQ has two components: a quantization algorithm and an inference system.

A) Quantization algorithm: Activation-aware Weight Quantization (Section 3)
- Background terms
  - Quantization maps FP values to low‑bit integers. Weight‑only grouped quantization means only weights are quantized; scales are shared within small groups (e.g., `g128` means each group has 128 weights).
  - A “channel” here refers to an input channel of a linear layer (matrix column); per‑channel scaling scales each such column and compensates by inverse scaling of the corresponding input feature.

- Step 1: Empirical observation—only ~0.1%–1% of channels are “salient”
  - Keeping just 0.1–1% of channels in FP16 massively reduces the perplexity loss versus naive INT3 RTN on OPT models, but only if those channels are chosen using activation magnitude (Table 1).
    - Example: OPT‑6.7B under INT3/g128—RTN PPL 23.54; keeping only 1% FP16 channels chosen by activation gives PPL 11.39 (Table 1; Figure 2b).
  - Selecting large‑norm weight channels or random channels barely helps (Table 1).

- Step 2: Replace mixed precision with equivalent per‑channel scaling
  - Mixed precision (some channels left FP16) is hardware‑unfriendly. Instead, AWQ derives a scaling trick that achieves a similar effect without mixed formats (Figures 2c and 2 caption; Section 3.2).
  - Key quantization equation (Eq. 1): `Q(w) = Δ * Round(w/Δ)`, where `Δ = max(|w|) / 2^(N-1)` for N‑bit quantization.
  - If a salient weight `w` is scaled up by `s > 1` and the corresponding activation is scaled down by `1/s`, the linear map is functionally equivalent but the relative quantization error on that weight is reduced (Eqs. 2–3).
    - Intuition: rounding error is roughly uniform and independent of `s`, but dividing the output by `s` reduces the error magnitude; if `Δ' ≈ Δ`, the error shrinks by about `1/s` (Eq. 3, Table 2).
    - Caution: scaling too much increases group max `Δ'`, which hurts non‑salient channels. Table 2 shows OPT‑6.7B PPL improves from 23.54 (s=1) to 11.92 (s=2), but worsens when s=4 because `Δ'` starts increasing for 21.2% of groups.

- Step 3: Data‑driven search for per‑channel scales
  - Objective (Eq. 4): choose a per‑input‑channel scale vector `s` that minimizes post‑quantization output MSE on a small calibration set, while compensating inputs by `diag(s)^{-1}` (which can be fused into preceding ops).
  - Practical search space (Eq. 5): set `s = s_X^α` where `s_X` is the per‑channel average activation magnitude measured offline; tune a single scalar `α ∈ [0,1]` by grid search (20 values). Weight clipping is applied to further reduce MSE.
    - Why this design: it encodes the “activation‑awareness” (saliency ≈ large activation magnitude) and balances protecting salient channels against increasing `Δ'` for others. It avoids unstable gradient‑based optimization and backprop (Section 3.2).

- Step 4: Quantize with grouping
  - Use INT4 or INT3 with group size 128 unless noted (Section 5.1). Grouping improves the accuracy vs per‑tensor quantization.

B) System: TinyChat for fast W4A16 inference (Section 4)
- Why speedups are possible
  - On-device generation is much slower than context encoding (310 ms vs 10 ms for 200‑token context + 20‑token generation; Figure 3a) and is memory‑bound with low arithmetic intensity (~1 FLOP/byte in FP16; Figure 3b).
  - Weight transfer dominates memory traffic (Figure 3c). Moving weights to 4‑bit increases arithmetic intensity roughly 4× (to ~4 FLOPs/byte), lifting the roofline-bound performance.

- Implementation techniques
  - On‑the‑fly dequantization: fuse dequantization with GEMM kernels to avoid writing dequantized weights to DRAM (Section 4.2).
  - SIMD‑aware weight packing on CPUs: reorder/pack 4‑bit weights to align with 128‑bit NEON registers and unpack an entire vector in three SIMD instructions using bitwise AND/shift with a 128‑bit mask (Figure 4). Yields up to 1.2× speedup in dequantization on ARM.
  - GPU packing: pack 8 weights as `{0,2,4,6,1,3,5,7}` to match GPU access patterns (Section 4.2).
  - Kernel fusion: fuse layer‑norm ops; fuse QKV projections and positional embedding computation; preallocate/update KV caches inside attention kernels (Section 4.2). This reduces launch overhead that is comparable to short FP16 kernel runtimes (~0.01 ms on RTX 4090).

## 4. Key Insights and Innovations
- Activation‑aware saliency is the right signal
  - Novelty: Instead of relying on weight norms or reconstruction, AWQ identifies “salient” channels by their activation magnitude and protects them (Table 1, Figure 2). This differs from GPTQ’s second‑order reconstruction that can overfit a small calibration set.
  - Significance: With only 0.1–1% protected channels (conceptual experiment), INT3 PPL collapses from RTN’s 23.54 to 11.39 on OPT‑6.7B (Table 1), revealing a small set of crucial channels.

- Equivalent per‑channel scaling reduces error without mixed precision
  - Novelty: The error analysis (Eqs. 1–3) shows scaling up a channel and inversely scaling its inputs shrinks its relative quantization error by roughly `1/s` while keeping computation in uniform low‑bit format. Prior work typically either uses mixed precision or per‑tensor scaling.
  - Significance: Comparable accuracy to a mixed-precision “protect 1% in FP16” strategy without hardware complexity (Table 3).

- Minimal‑data, calibration‑robust PTQ
  - Novelty: The search space `s = s_X^α` uses only per‑channel average activation magnitudes from a small calibration set; no backprop or reconstruction (Section 3.2).
  - Significance: Needs 10× fewer calibration sequences than GPTQ to reach similar or better PPL (Figure 8a). When calibration and evaluation distributions differ (PubMed vs Enron), AWQ’s degradation is only +0.5–0.6 PPL, while GPTQ worsens by +2.3–4.9 (Figure 8b).

- System co‑design (TinyChat) that converts memory savings to speed
  - Novelty: Platform‑aware packing and kernel fusion specifically for W4A16 where dequantization happens inside the main compute loop (Section 4.2, Figure 4).
  - Significance: 3.2–3.9× measured speedup over HuggingFace FP16 across devices (Figure 9), and state‑of‑the‑art speed on Jetson Orin vs AutoGPTQ, llama.cpp, and exllama (Figure 10).

## 5. Experimental Analysis
- Evaluation setup (Section 5.1)
  - Quantization: weight‑only INT3/INT4, group size 128; α searched over 20 grid points in [0,1]; small calibration set from The Pile to avoid task overfitting.
  - Models: LLaMA/Llama‑2 (7B–70B), OPT (1.3B–30B), Mistral‑7B, Mixtral‑8×7B, instruction‑tuned Vicuna (7B/13B), and VLMs (OpenFlamingo‑9B; VILA‑7B/13B).
  - Metrics: WikiText‑2 perplexity (PPL↓) for language modeling; GPT‑4 preference for Vicuna (Figure 5); COCO CIDEr for OpenFlamingo (Table 6); multi‑VLM benchmarks for VILA (Table 7); MBPP pass@k and GSM8K accuracy (Table 8); tokens/sec for runtime (Figures 9–10).

- Main quantitative results
  - Llama family (Table 4):
    - Llama‑2‑7B INT3/g128: PPL 6.24 (AWQ) vs 6.66 (RTN) vs 6.43 (GPTQ); INT4: 5.60 (AWQ) vs 5.73 (RTN).
    - Llama‑2‑70B INT3: 3.74 (AWQ) vs 3.98 (RTN) vs 3.86 (GPTQ); INT4: 3.41 (AWQ) vs 3.46 (RTN).
    - Similar consistent gains on original LLaMA (7B–65B).
  - Mistral/Mixtral (Table 5):
    - Mistral‑7B: INT4 PPL 4.30 (AWQ) vs 4.14 (FP16); INT3 PPL 4.83 (AWQ). Mixtral‑8×7B: INT4 PPL 6.05 vs FP16 5.94; INT3 PPL 6.52.
  - Instruction‑tuned Vicuna (Figure 5):
    - Across 160 pairwise GPT‑4 comparisons (80 prompts × two orderings), AWQ has the most “quantized wins” for both 7B and 13B, outperforming RTN and GPTQ in preference scoring.
  - Multi‑modal OpenFlamingo on COCO (Table 6):
    - INT4/g128, 32‑shot CIDEr: 80.53 (AWQ) vs 77.13 (RTN) and 74.98 (GPTQ); degradation vs FP16 reduces from −4.57 (RTN) to −1.17 (AWQ).
    - INT3/g128, 32‑shot: 74.47 (AWQ) vs ~64.8 (RTN/GPTQ); strong improvement at 3‑bit.
  - Multi‑modal VILA across 11 benchmarks (Table 7):
    - Near-lossless INT4 quantization; e.g., VILA‑7B VQAv2 80.1 (AWQ) vs 80.3 (FP16); POPE 85.3 vs 86.3.
  - Coding and math (Table 8):
    - CodeLlama‑7B‑Instruct on MBPP: pass@1 improves to 40.64 (AWQ) from 38.53 (FP16) and 37.51 (RTN); pass@10 49.25 (AWQ) ≈ 49.77 (FP16).
    - GSM8K: Llama‑2‑70B 56.40 (AWQ) ≈ 56.41 (FP16); smaller models also improve over RTN/GPTQ.
  - Extreme low‑bit (INT2/g64; Table 9):
    - AWQ complements GPTQ: e.g., OPT‑13B PPL 13.25 (AWQ+GPTQ) vs 16.74 (GPTQ). RTN fails catastrophically.
  - Data efficiency and robustness (Figure 8):
    - Calibration size: similar PPL with only 16 sequences for AWQ vs 192 for GPTQ (8–12× smaller).
    - Domain shift (PubMed vs Enron): +0.5–0.6 PPL for AWQ vs +2.3–4.9 for GPTQ.
  - System speedups (Figures 9–10, Table 10):
    - RTX 4090: 2.7–3.9× over HuggingFace FP16; FP16‑optimized TinyChat already speeds up Llama‑2‑7B from 52 to 62 tok/s, and W4 adds another 3.1×.
    - Jetson Orin: 3.5× average gain vs FP16; e.g., Llama‑2‑13B runs at 21 tok/s (Figure 9b).
    - Laptop RTX 4070 with 8GB: Llama‑2‑13B at 33 tok/s (FP16 OOM; Figure 9c).
    - Against other 4‑bit systems on Orin (Figure 10a): Llama‑2‑7B 39.1 tok/s (TinyChat) vs 22.5 (AutoGPTQ), 15.9 (llama.cpp), 13.4 (exllama).
    - Raspberry Pi 4B (Figure 10b): up to 0.7 tok/s for 7B models; demonstrates extreme portability.
    - VILA VLMs (Table 10): 2.9–3.1× speedups and enabling 13B on consumer GPUs.

- Do the experiments support the claims?
  - Accuracy: Tables 4–8 consistently show AWQ ≥ RTN and typically ≥ GPTQ at 3–4 bits across multiple architectures and tasks, including instruction‑tuned and multi‑modal models.
  - Robustness and data‑efficiency: Figure 8 directly evaluates calibration size and domain shift.
  - System: Figures 3, 9, and 10, plus Table 10, connect roofline analysis to realized speedups across devices and models.

- Ablations and diagnostics
  - “Protect 1% FP16” vs scaling (Table 3): scaling nearly matches FP16‑mix precision without hardware penalties.
  - Effect of `s` on `Δ'` and error (Table 2): shows the tradeoff across different scaling magnitudes and motivates the α search.

## 6. Limitations and Trade-offs
- Assumptions and scope
  - Weight‑only quantization: activations remain FP16 (W4A16). The method does not address activation quantization or KV‑cache compression, which matter for very large models or multi‑query attention scenarios.
  - Grouped quantization design: results predominantly use group size 128; other group sizes or per‑tensor/per‑row schemes are not exhaustively explored.

- Dependence on activation statistics
  - AWQ relies on per‑channel average activation magnitude from a calibration corpus (Eq. 5). While Figure 8 shows robustness to distribution shift, the approach still depends on the calibration data capturing representative activation scales (e.g., extreme out-of-distribution inputs may alter saliency).

- Very low bit-widths
  - At INT2, AWQ alone is not sufficient; it must be combined with GPTQ to reach practical PPLs (Table 9). This indicates the scaling trick does not fully mitigate quantization artifacts at extreme compression.

- System trade-offs and coverage
  - TinyChat focuses on batch‑size‑1, generation‑heavy, edge settings. It does not target high‑throughput server inference with large batches, where different bottlenecks and kernel choices apply.
  - Platform-specific optimizations (e.g., NEON packing in Figure 4) require hardware-aware engineering; portability to every accelerator or CPU ISA may require additional development.
  - Some model families required extra care even in prior work (e.g., GPTQ reorder for LLaMA/OPT); while TinyChat supports many families (Figure 10), universal coverage of all emerging architectures may lag the ecosystem.

- Metrics and tasks
  - PPL is a strong but indirect measure of quality; the paper complements it with instruction, coding, math, and VLM evaluations, yet broader end‑task coverage (reasoning, safety, multilinguality) could further validate generalization.

## 7. Implications and Future Directions
- Field impact
  - Methodologically, AWQ reframes PTQ for LLMs around activation‑aware saliency and equivalent scaling, offering a simple, training‑free path to 3–4‑bit weight‑only quantization with strong generalization (Tables 4–8, Figure 8). System‑wise, TinyChat demonstrates that W4A16 can deliver substantial real‑world speedups on edge hardware (Figures 9–10).

- Follow‑up research enabled
  - Extending activation‑aware ideas to activation quantization (W4A4) and KV‑cache quantization while maintaining generalization.
  - Adaptive or learned per‑layer/per‑group search spaces beyond `s = s_X^α`, possibly with lightweight meta‑optimization or Bayesian search to tune α and clipping jointly.
  - Dynamic, input‑aware scaling at inference time for non‑stationary workloads, with minimal overhead.
  - Combining AWQ with sparsity or low‑rank adaptation for further compression while preserving accuracy.

- Practical applications
  - Private, offline assistants and coding copilots on laptops and mobile GPUs.
  - Robotics and embedded systems requiring guaranteed latency under tight power budgets.
  - Multimodal on-device applications (e.g., captioning, VQA) where AWQ shows minimal degradation (Tables 6–7).
  - Democratized research and deployment of larger models on modest hardware, e.g., running 13B–70B variants on consumer GPUs and Jetson devices (Figures 9–10), and even enabling experimentation on Raspberry Pi.

> Core takeaway: By protecting a tiny set of activation‑salient channels through per‑channel scaling (Eqs. 1–5) and by engineering the runtime around 4‑bit weight bandwidth, AWQ + TinyChat makes low‑bit, on‑device LLMs both accurate and fast, without retraining and with strong robustness to calibration data choice.
