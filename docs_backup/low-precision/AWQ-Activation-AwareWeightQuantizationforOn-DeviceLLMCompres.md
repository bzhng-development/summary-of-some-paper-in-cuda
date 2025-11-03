# AWQ: Activation-Aware Weight Quantization for On-Device LLM Compression and Acceleration

**ArXiv:** [2306.00978](https://arxiv.org/abs/2306.00978)

## 🎯 Pitch

AWQ introduces a novel activation-aware, post-training quantization method for large language models (LLMs) that preserves accuracy at extremely low bit-widths by identifying and protecting the most important weight channels based on activation statistics. This innovation, implemented in the highly efficient TinyChat inference framework, enables up to 4× smaller models and 3–4× faster on-device inference—democratizing the deployment of advanced LLMs on consumer hardware and resource-constrained edge devices without sacrificing performance.

---

## 1. Executive Summary
Activation-aware Weight Quantization (AWQ) is a post‑training, weight‑only quantization method for large language models (LLMs) that reduces memory and speeds up on‑device inference by moving weights to 4 bits while preserving accuracy. Its core idea is to detect “salient” weight channels using activation statistics and protect them by an equivalent scaling transformation, avoiding hardware‑unfriendly mixed precision; the companion system TinyChat implements fast 4‑bit inference across GPUs and CPUs, delivering 3–4× speedups and enabling very large models on edge devices (Figures 1, 3, 9).

## 2. Context and Motivation
- Problem addressed
  - On‑device LLMs face severe memory and bandwidth limits; naive deployment of FP16 models is often impossible (e.g., GPT‑3 at 350 GB in FP16; Abstract; Introduction).
  - Even when models fit, the generation stage is memory‑bound: arithmetic intensity is ≈1 FLOP/byte in FP16 (Figure 3b), so weight traffic dominates latency (Figure 3c).

- Why it matters
  - Reduces cloud dependence, latency, and privacy risks by enabling local inference (Introduction).
  - Lowers serving cost and broadens access to strong LLMs on consumer hardware and embedded devices (Figure 1; TinyChat computer demo).

- Prior approaches and their gaps
  - QAT (quantization‑aware training) is too expensive for LLMs (Related Work).
  - PTQ (post‑training quantization) often degrades accuracy at low bits. The strongest prior for weight‑only PTQ, `GPTQ`, reconstructs layers using second‑order information but can overfit the calibration set and lose generalization, and for some models needs fragile layer reordering (Related Work; discussion around GPTQ; Table 4).
  - W8A8 (both weights and activations at 8-bit) methods like SmoothQuant simplify kernels but still leave heavy memory traffic in generation and require more hardware support (Section 4; Figure 3b–c).

- Positioning
  - AWQ targets weight‑only low‑bit quantization (W4A16 or W3A16) and is training‑free. It proposes a mechanism to identify and protect salient weight channels using activation statistics and an equivalent scaling instead of mixed precision (Sections 3.1–3.2; Figure 2). TinyChat converts the 4× weight‑memory reduction into measured speedup with dequantization‑fused kernels and platform‑aware packing (Section 4; Figure 4).

## 3. Technical Approach
AWQ comprises two tightly linked parts: a quantization algorithm (Sections 3.1–3.2) and an inference system, TinyChat (Section 4), that realizes the algorithm’s theoretical benefits.

- Definitions (selective)
  - `weight-only quantization`: compress only model weights to low-bit integers; keep activations in higher precision (e.g., FP16).
  - `grouped quantization (g128)`: weights are partitioned into groups of 128 elements; each group shares a scale/zero‑point (here per‑group symmetric quantization; Section 5.1).
  - `perplexity (PPL)`: lower is better; a standard language‑modeling metric.
  - `arithmetic intensity`: FLOPs per byte moved; indicates whether a workload is compute‑ or memory‑bound (Figure 3b).

A) Observation: a tiny fraction of weights matters much more
- Finding: protecting just 0.1–1% of weight channels can drastically improve low‑bit accuracy if the channels are selected using activation magnitude, not weight magnitude (Section 3.1).
  - Evidence: On OPT models at INT3‑g128, keeping 1% of channels in FP16 chosen by activation lowers PPL from 119.0→16.9 (OPT‑1.3B), 23.54→11.39 (OPT‑6.7B), 46.04→10.43 (OPT‑13B); choosing by weight norm barely helps (Table 1).
- Intuition: channels with larger average activation magnitude process more important features; errors on these channels harm outputs more.

B) Why mixed precision is undesirable and how AWQ avoids it
- Mixed precision (keeping that 1% in FP16) is accurate but hardware‑inefficient for storage/layout and kernel design (Section 3.1; Figure 2b).
- AWQ’s key maneuver: protect salient channels using an equivalent scaling, not by keeping them at FP16 (Figure 2c).

C) Quantization error analysis and the scaling trick
- Basic quantization (Equation 1): for an N‑bit group, `Q(w) = Δ·Round(w/Δ)` where `Δ = max(|w|)/2^(N−1)`.
- Equivalent transformation (Equation 2): multiply a salient weight channel by `s>1` and inversely scale the corresponding input activation by `1/s`:
  - Compute `Q(w·s)·(x/s) = Δ'·Round((w·s)/Δ')·(x/s)`.
- Error reasoning (Equation 3):
  - The rounding error for `Round(·)` behaves roughly uniformly within half a quantization step (~0.25 scaled by Δ).
  - If scaling one channel does not change the group’s maximum (`Δ'≈Δ`), the error for that channel scales down by `1/s`, i.e., relative error shrinks by `Δ'/Δ · 1/s ≈ 1/s`.
- Trade‑off: larger `s` better protects salient channels but risks increasing `Δ'` when it changes a group’s max; that amplifies errors for non‑salient channels (Table 2 shows this effect).
  - Empirically, `s=2` gives the best PPL on OPT‑6.7B INT3‑g128: 23.54 (s=1) → 11.92 (s=2); pushing to `s=4` worsens PPL to 12.36 due to Δ changes affecting 21.2% of channels (Table 2).

D) Turning the idea into a robust, data‑light search
- Goal: minimize layer output difference after quantization while considering both protected and unprotected channels (Equation 4).
  - Objective: `s* = argmin_s || Q(W·diag(s))·(diag(s)^{-1}·X) − W·X ||`, where `X` are cached activations from a small calibration set.
- Search space (Equation 5): choose scale per input channel as `s = s_X^α`, where `s_X` is that channel’s average activation magnitude and `α∈[0,1]` controls aggressiveness; select `α` by a small grid search (20 points; Section 5.1). Apply weight clipping to reduce quantization MSE.
  - Why this design: ties protection directly to activation scale (saliency), uses a single hyperparameter per layer, and avoids backpropagation or reconstruction—reducing overfitting and data needs (Sections 3.2, 5.3; Figure 8).

E) Quantization settings used in experiments
- Low‑bit weight‑only INT4/INT3 with group size 128 (Section 5.1).
- Calibration: a few sequences from The Pile (to avoid task‑specific overfitting; Section 5.1).
- No backpropagation, no second‑order reconstruction.

F) Why AWQ accelerates on device, and the TinyChat system
- Bottleneck analysis (Figure 3):
  - Generation is slower than context (310 ms vs 10 ms for 20 vs 200 tokens on LLaMA‑7B; Figure 3a).
  - Generation is memory‑bound with FP16 (`~1` FLOP/byte; Figure 3b); weights dominate memory traffic (Figure 3c).
  - Moving weights from 16→4 bits multiplies arithmetic intensity by ≈4× (to ~4 FLOP/byte; Figure 3b), unlocking speedups if dequantization overhead is controlled.
- TinyChat design (Section 4):
  - On‑the‑fly dequantization fused into GEMM kernels to avoid writing dequantized weights to DRAM; used for both matrix‑matrix and matrix‑vector paths.
  - SIMD‑aware weight packing to minimize bit manipulation per weight; e.g., for ARM NEON 128‑bit SIMD, offline reorder enables unpacking 32 4‑bit weights with three vector ops instead of 3×32 scalar ops (Figure 4), improving linear‑layer latency (right panel of Figure 4).
  - Kernel fusion: fuse LayerNorm operations; fuse QKV projections and positional embeddings; pre‑allocate KV caches; reduces kernel‑launch overhead (Section 4.2).
  - Cross‑platform backends (CUDA/PTX, NEON, AVX) with a PyTorch frontend; C++ lowering for CPU to reduce overhead (Section 4).

## 4. Key Insights and Innovations
- Activation‑aware saliency and protection via equivalent scaling (fundamental)
  - Novelty: uses activation statistics (average magnitude) to identify important weight channels, then protects them by scaling weights and inversely scaling inputs (Equations 2–5), not by mixed precision (Figure 2).
  - Significance: yields large accuracy gains at very low bits without training or reconstruction; keeps hardware‑friendly uniform bitwidth (Tables 1–3).

- Data‑efficient, generalizable PTQ (fundamental)
  - No backpropagation or second‑order reconstruction—only channel‑wise activation means and a lightweight `α` grid search (Section 3.2).
  - Evidence: achieves strong accuracy with 10× fewer calibration sequences than GPTQ and is more robust to calibration/evaluation distribution shift (Figure 8a–b).

- System realization that converts memory savings into speed (significant engineering)
  - Fused dequantization and platform‑aware packing (Figure 4) plus kernel fusion translates W4A16 into 3–4× measured speedup across devices (Figure 9; Section 4.2).
  - Practical impact: enables 70B‑parameter Llama‑2 on a single Jetson Orin 64GB; runs 13B at interactive speeds on an 8GB laptop GPU (Figure 9).

- Generalization to instruction‑tuned and multi‑modal LMs (new capability)
  - Demonstrates low‑bit quantization for instruction‑tuned Vicuna (Figure 5) and multi‑modal models (OpenFlamingo and VILA) with minimal loss and sometimes gains over RTN/GPTQ (Tables 6–7), which prior weight‑only PTQ often struggled to preserve due to overfitting/reconstruction issues.

- Complementary to GPTQ at extreme low bits (incremental but useful)
  - Combining AWQ with GPTQ further improves INT2 results, e.g., on OPT‑13B PPL 16.74 (GPTQ) → 13.25 (AWQ+GPTQ) (Table 9).

## 5. Experimental Analysis
- Evaluation setup (Section 5.1)
  - Models: LLaMA/Llama‑2 (7B–70B), OPT (1.3B–30B), Mistral‑7B, Mixtral‑8×7B, instruction‑tuned Vicuna, and multi‑modal OpenFlamingo‑9B and VILA‑7B/13B.
  - Quantization: INT4/INT3, group size 128, weight‑only; `α` grid size 20; calibration from The Pile; no re‑training.
  - Metrics: WikiText‑2 PPL for language modeling; task‑specific metrics for code (MBPP), math (GSM8K), image captioning (COCO CIDEr), and multi‑modal leaderboards (Table 7).
  - Baselines: FP16, `RTN` (round‑to‑nearest), `GPTQ`, and `GPTQ‑R` (with layer reordering).

- Main accuracy results
  - LLaMA/Llama‑2 language modeling (Table 4)
    - INT3‑g128: On Llama‑2‑13B, PPL 5.32 (AWQ) vs 5.41 (GPTQ‑R), 5.48 (GPTQ), 5.52 (RTN). On LLaMA‑65B, 3.95 (AWQ) vs 4.21 (GPTQ‑R), 4.17 (GPTQ), 4.24 (RTN).
    - INT4‑g128: On LLaMA‑30B, 4.21 (AWQ) vs 4.22 (GPTQ‑R) and 4.23 (RTN); on Llama‑2‑70B, 3.41 (AWQ) vs 3.43 (GPTQ‑R) and 3.46 (RTN).
    - Takeaway: consistent, sometimes clear improvements over GPTQ/RTN across sizes.
  - Mistral/Mixtral (Table 5)
    - Mistral‑7B INT4 PPL 4.30 vs FP16 4.14; Mixtral‑8×7B INT4 PPL 6.05 vs FP16 5.94. Loss is small; AWQ holds up on GQA/MoE architectures.
  - Instruction‑tuned Vicuna (Figure 5)
    - With INT3‑g128, AWQ yields more “wins” vs FP16 than RTN or GPTQ on both 7B and 13B when judged by GPT‑4 on 80 prompts (160 pairwise comparisons). This indicates better preservation of instruction‑following ability.
  - Multi‑modal models
    - OpenFlamingo‑9B on COCO captioning (Table 6):
      - INT4‑g128, 32‑shot: degradation vs FP16 is −1.17 with AWQ, much less than −4.57 (RTN) and −6.72 (GPTQ). Similar pattern across 0/4/8/16 shots. At INT3, AWQ cuts the 32‑shot drop from ~−16.9 (RTN/GPTQ) to −7.23.
    - VILA‑7B/13B on 11 benchmarks (Table 7):
      - INT4‑g128: AWQ is essentially lossless across VQAv2, GQA, VizWiz, ScienceQA‑IMG, TextVQA, POPE, MME, MMB, SEED, LLaVA‑Bench, and MM‑Vet.
  - Programming and math (Table 8)
    - CodeLlama‑7B‑Instruct on MBPP: pass@1 40.64 (AWQ) vs 37.51 (RTN) and 31.97 (GPTQ); near FP16 38.53. Llama‑2 on GSM8K: AWQ matches or nears FP16 at 70B and improves over RTN/GPTQ at 7B/13B.
  - Extreme low bits (Table 9)
    - INT2‑g64 on OPT shows AWQ+GPTQ improves over GPTQ alone, e.g., 6.7B PPL 16.65 → 15.71; 13B 16.74 → 13.25.
  - Ablations on scaling (Tables 2–3; Figure 2)
    - Scaling only 1% salient channels by `s=2` reduces OPT‑6.7B INT3 PPL from 23.54→11.92 (Table 2).
    - AWQ (with search and clipping) tracks or beats “1% FP16 mixed precision” while remaining hardware‑friendly (Table 3).

- Data efficiency and robustness (Figure 8)
  - Calibration size: achieves better PPL with 16 sequences than GPTQ with 192 sequences on OPT‑6.7B INT3‑g128 (Figure 8a).
  - Distribution shift: when calibrating on PubMed and evaluating on Enron (and vice versa), AWQ’s PPL worsens by only 0.5–0.6 vs 2.3–4.9 for GPTQ (Figure 8b).

- System speedups (Figure 9; Table 10; Figure 10)
  - Desktop RTX 4090: 2.7–3.9× over HuggingFace FP16 across Llama‑2, MPT, Falcon; e.g., Llama‑2‑7B improves from 52 tokens/s (HF FP16) → 62 (FP16 fused kernels) → 194 (W4A16 AWQ), a ~3.7× end‑to‑end gain (Figure 9a).
  - Laptop RTX 4070 8GB: runs Llama‑2‑13B at 33 tokens/s (FP16 cannot fit 7B) (Figure 9c).
  - Jetson Orin: 3.5× on average (Figure 9b); TinyChat outperforms AutoGPTQ, llama.cpp, exllama by 1.2–3.0× on Orin and also supports more model families (Figure 10a).
  - Raspberry Pi 4B: up to 0.7 tokens/s on 7B models, demonstrating feasibility on very constrained hardware (Figure 10b).
  - VILA multi‑modal: up to ~3× throughput gains on A100/4090/Orin (Table 10).

- Do the experiments support the claims?
  - Broad model coverage (decoder LLMs, instruction‑tuned, MoE, VLMs), multiple bitwidths, strong baselines, and ablations on scaling and calibration lend credibility. Results consistently favor AWQ on accuracy and show real speedups from TinyChat.

## 6. Limitations and Trade-offs
- Assumptions in the error analysis
  - Relies on `Δ'≈Δ` when scaling salient channels so the group max is unchanged (Section 3.2). If scaling alters the group max, non‑salient channels’ error grows; Table 2 quantifies this (e.g., at s=4, 21.2% of groups see Δ change and PPL worsens vs s=2).
  - Assumes rounding error is roughly uniform; not always exact but empirically adequate (Equation 3 discussion).

- Scope limits
  - Weight‑only quantization: activations stay FP16 (W4A16). This leaves memory and latency from activations/ KV caches unaddressed; it targets the dominant weight traffic but not all bottlenecks (Figure 3c).
  - Focus on linear layers and standard Transformer blocks; specialized components outside this may require extra engineering.

- Heuristic search
  - The scaling policy uses a per‑layer `α` with grid search (Equation 5). While simple and effective, it is heuristic; there is no closed‑form optimum guarantee, and the best `α` may vary with model/layer.

- Dequantization cost on CPUs
  - Even with packing, on‑the‑fly dequantization adds bit‑ops and FMAs; speedups on some CPUs may be modest (ARM NEON shows up to ~1.2× improvement for the linear‑layer unpack path; Figure 4 right), though end‑to‑end gains are still substantial in TinyChat.

- Extreme low bits still hard
  - At INT2, AWQ+GPTQ improves over GPTQ but remains far from FP16 (Table 9). The method makes INT2 more practical but not lossless.

- Calibration still needed
  - Although minimal and robust (Figure 8), AWQ still needs some calibration data to estimate activation magnitudes; completely data‑free settings are not evaluated.

## 7. Implications and Future Directions
- How this changes the landscape
  - Demonstrates that activation‑aware, training‑free, weight‑only PTQ can preserve generalist ability (including instruction‑tuned and multi‑modal tasks) and deliver real on‑device speedups. This lowers the barrier for deploying strong LLMs on edge hardware, from laptops to Jetson to Raspberry Pi (Figures 1, 9–10; Tables 6–7, 10).

- Follow‑up research enabled or suggested
  - Joint weight+activation quantization with AWQ‑style protection to further reduce bandwidth, possibly moving toward W4A8 or lower while maintaining accuracy.
  - Dynamic, runtime‑aware scaling that adapts `α` per input distribution or per decoding phase (context vs generation).
  - Training‑time regularization to make models more AWQ‑friendly (e.g., constraining activation distributions to sharpen saliency).
  - Hardware/ISA support for native INT4×FP16 or mixed‑precision dot‑products to cut dequantization cost; co‑design with TinyChat‑like kernels.
  - Extending the method to KV‑cache compression and attention‑specific quantization (e.g., softmax paths), which dominate memory in long‑context generation.
  - Formal guarantees on error bounds when Δ changes; smarter grouping that isolates salient channels to prevent Δ inflation.

- Practical applications
  - On‑device assistants in privacy‑sensitive domains (healthcare, finance), offline copilots for code or math tasks, and embedded VLMs for robotics or wearables.
  - Cloud cost reduction via higher model density per GPU and improved throughput in memory‑bound regimes.

> Representative quantitative highlights:
> - “Scaling up the 1% salient channels with s=2 improves OPT‑6.7B INT3‑g128 PPL from 23.54 to 11.92” (Table 2).
> - “Across LLaMA/Llama‑2 scales, AWQ beats RTN and GPTQ on PPL in both INT3 and INT4” (Table 4).
> - “On OpenFlamingo‑9B, INT4‑g128 32‑shot degradation is −1.17 with AWQ vs −4.57 (RTN) and −6.72 (GPTQ)” (Table 6).
> - “TinyChat delivers 2.7–3.9× speedup on RTX 4090 and ~3.5× on Jetson Orin; Llama‑2‑7B goes from 52 tokens/s (HF FP16) to 194 tokens/s (W4A16 AWQ)” (Figure 9).
