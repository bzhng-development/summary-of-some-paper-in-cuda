# EFFICIENTLLM: EFFICIENCY IN LARGE LANGUAGE MODELS EVALUATION ON ARCHITECTURE PRETRAINING, FINE-TUNING, AND BIT-WIDTH QUANTIZATION

**ArXiv:** [2505.13840](https://arxiv.org/abs/2505.13840)

## 🎯 Pitch

EfficientLLM introduces the first comprehensive, large-scale empirical benchmark evaluating LLM efficiency across architecture design, fine-tuning, and inference quantization, using over 100 model–technique pairs on real-world, production-grade hardware. By jointly measuring memory, compute, latency, throughput, energy, and compression, it reveals that all efficiency methods involve nuanced trade-offs—with optimal solutions depending on task, model scale, and deployment constraints—empowering practitioners to make data-driven decisions for building faster, cheaper, and more sustainable large AI systems.

---

## 1. Executive Summary (2–3 sentences)
EfficientLLM is a large-scale, end-to-end benchmark that measures how to make Large Language Models (LLMs) more efficient across three critical stages: architecture pretraining, fine-tuning, and inference via quantization. It runs over 100 model–technique pairs on production-class GPUs and introduces fine-grained metrics (memory, compute, latency, throughput, energy, compression) to quantify real-world trade-offs; key findings include that no single method is Pareto-optimal, optima depend on task and scale, and many techniques transfer to vision and vision–language models (Figures 2–5; Sections 2, 5–6).

## 2. Context and Motivation
- Problem addressed
  - There is no comprehensive, empirical, end-to-end benchmark that evaluates LLM efficiency techniques under realistic deployment conditions. Prior work tends to study a single technique in isolation, at limited scale, often without energy measurement or cross-stage comparability (Section 1).
- Why this matters
  - Training and serving LLMs is extremely expensive in compute, memory, energy, and money. For instance, GPT‑3’s training required ~3,640 PF-days and an estimated >$4.6M (Introduction). Deployment costs also scale with model and context size; energy and carbon footprints are substantial (Sections 1, 5.1.2).
- Prior approaches and their gaps
  - Architecture: Many “efficient attention” variants, Mixture‑of‑Experts (MoE), and positional encodings exist, but their measured trade-offs on modern accelerators and at multiple scales are rarely compared head‑to‑head.
  - Fine‑tuning: Parameter‑Efficient Fine‑Tuning (PEFT) methods (LoRA and variants) are abundant, yet it is unclear which method is best for which model size or latency/energy budget.
  - Inference: Quantization and serving optimizations are common, but their energy/latency/memory/accuracy trade-offs are not consistently reported across families and scales (Sections 3–4).
- How this paper positions itself
  - It introduces a unified taxonomy (architecture pretraining, fine‑tuning, inference) and runs a hundred‑scale study on a modern cluster (48× GH200; 8× H200) with standardized metrics (AMU, PCU, AL, TT/ST/IT, AEC, MCR) to enable apples‑to‑apples comparisons (Figure 1; Sections 1, 5.1, 5.2).
  - It extends beyond text to Large Vision Models (LVMs) and Vision‑Language Models (VLMs), testing transferability of efficiency techniques (Section 6).

## 3. Technical Approach
This is an empirical benchmark with carefully controlled hardware, software, and metrics, evaluating techniques across three lifecycle stages.

- Hardware and software
  - Pretraining studies: 48× NVIDIA GH200 (96 GB) with NVLink/InfiniBand; 3D parallelism via Megatron‑Core (tensor, pipeline, data parallel) (Section 5.3 “Hardware and Training Framework”).
  - Fine‑tuning studies: 8× NVIDIA H200 (141 GB) using LlamaFactory; DeepSpeed ZeRO‑3 Offload for full fine‑tuning when needed (Section 5.4).
  - Inference studies: GH200 nodes; serving on optimized inference servers (Section 5.5).

- Metrics (all formally defined in Section 5.1)
  - `AMU` (Average Memory Utilization): time‑averaged memory used over total device memory (Eq. 1).
  - `PCU` (Peak Compute Utilization): average actual GPU utilization / theoretical peak; reported only where it varies meaningfully (PEFT; Eq. 2 and footnote).
  - `AL` (Average Latency): average per‑iteration/request time including compute and communication (Eq. 3).
  - Throughput: `TT` (tokens/second/parameter for pretraining; Eq. 4), `ST` (samples/second/parameter for fine‑tuning; Eq. 5), `IT` (tokens/second for inference; Eq. 6).
  - `AEC` (Average Energy Consumption): average power over time, from integrated energy (Eqs. 7–8).
  - `MCR` (Model Compression Rate): size reduction adjusted by performance retention (Eq. 9).
  - A composite “Efficiency Score” used in some visualizations is a weighted harmonic combination of normalized resource metrics (Appendix, Eq. 12; Figures 4, 7c). Normalization recipes are in Appendix (Eqs. 10–11).

- Architecture pretraining evaluations (Sections 4.4 and 5.3)
  - Efficient attention
    - `MQA` (Multi‑Query Attention): share key/value across heads; only queries are per‑head. Reduces KV‑cache footprint and speeds decoding (Section 4.4.2).
    - `GQA` (Grouped‑Query Attention): share K/V within groups of heads—intermediate between MHA and MQA (Section 4.4.2).
    - `MLA` (Multi‑Head Latent Attention): compress the KV cache into a low‑rank latent (dimension `dc << d`), then up‑project per head on the fly: `c_t = h_t W_DKV` and `k_i = c_t W_UK_i`, `v_i = c_t W_UV_i`. This shrinks KV memory while keeping per‑head expressivity (Section 4.4.2).
    - `NSA` (Native Sparse Attention): a tri‑branch design—global compression, selective global attention, and local sliding window—with learned gates `g_cmp, g_slc, g_win` blending the three (Section 4.4.2).
  - Positional encoding (Section 4.4.3)
    - Compares Rotary (`RoPE`), absolute (fixed and learnable), a relative scheme (“Relate”), and “None” (no PE).
  - Sparse modeling via `MoE` (Mixture‑of‑Experts) (Section 4.4.4)
    - Conditional computation: a router activates only top‑k experts per token (top‑2 in experiments), which reduces FLOPs per token while increasing total parameters. Trade‑off: extra routing and memory to store all experts.
  - Attention‑free alternatives (Section 4.4.5)
    - `Mamba` (state‑space model with selective updates): linear‑time sequence modeling, high speed and low memory.
    - `RWKV` (recurrent variant that trains parallely and infers recurrently).
    - `Pythia` is used here as an attention‑lite baseline for comparison (Table 6).

  - Experimental design
    - Base backbone for pretraining sweeps uses Qwen2.5‑style decoder (0.5B, 1.5B, 3B), trained on FineWeb‑Edu (350B tokens)—an educational subset designed to improve reasoning/factuality (Sections 5.2, 5.3).

- Training and tuning efficiency evaluations (Section 4.5.2; Section 5.4)
  - `PEFT` (Parameter‑Efficient Fine‑Tuning): adapt only small add‑on modules or selected weights.
    - `LoRA`: keep `W0` frozen and learn `ΔW = α A B` with low rank `r << min(m,n)`; merge after training (Section 4.5.2).
    - `LoRA‑plus`: different learning rates for `A` and `B` to improve optimization (Section 4.5.2).
    - `RSLoRA`: rank‑stabilized scaling (`α = 1/√r`) to make higher ranks stable (Section 4.5.2).
    - `DoRA`: decompose weights into magnitude and direction; update direction via LoRA‑like term and learn a magnitude vector (Section 4.5.2).
    - `PiSSA`: initialize low‑rank factors from principal singular vectors/values of `W0`; train `W = A B + R` (Section 4.5.2).
    - `Freeze`: freeze most parameters (e.g., initial layers) for minimal latency/compute.
    - `Full*`: full fine‑tuning with DeepSpeed ZeRO‑3 Offload; batch sizes halved to fit memory (Table 7 note).
  - Data: `OpenO1‑SFT` (77k English/Chinese instruction‑reasoning samples) and a domain dataset `Medical‑o1‑reasoning‑SFT` (Sections 5.2, 5.4).

- Inference efficiency via bit‑width quantization (Section 5.5)
  - Precisions evaluated: `bfloat16` (bf16), `float16` (fp16), and post‑training `int4` (4‑bit weights). `int8` is excluded due to instability/unsupported kernels on GH200 in their setup (Section 5.5 “Note on Int8 Quantization”).
  - Metrics include aggregate task score across MMLU‑Pro, BBH, GPQA, IFEval, MATH, MuSR; `IT` (tokens/s), `AMU`, `AEC`, `MCR` (Table 9; Appendix Table 16 for per‑benchmark scores).

- Cross‑modal extensions (Section 6)
  - LVMs: insert efficient attention and MoE into DiT‑style diffusion backbones (DiT‑XL/2, DiT‑L/8, DiT‑B/4) and evaluate FID (Fréchet Inception Distance; lower is better) and efficiency metrics (Tables 10–11).
  - VLMs: PEFT on LLaVA‑1.5 (7B), Qwen2.5‑VL‑7B, InternVL‑3‑38B, QvQ‑72B (Table 12).
  - LVM fine‑tuning: PEFT and full FT for Wan 2.1‑1.3B (video) and Stable Diffusion 3.5‑Medium (Table 13).

## 4. Key Insights and Innovations
1) A truly end‑to‑end, hardware‑grounded efficiency benchmark for LLMs  
- What’s new: A unified evaluation across architecture pretraining, PEFT, and inference quantization, run on modern GH200/H200 clusters with energy tracking and modality‑agnostic metric collection (Figure 1; Sections 1, 5–6).  
- Why it matters: It enables evidence‑based decisions across the LLM lifecycle rather than isolated, anecdotal choices.

2) A principled metric suite that captures the real bottlenecks  
- What’s new: Fine‑grained metrics—`AMU`, `PCU`, `AL`, `TT/ST/IT`, `AEC`, `MCR`—with explicit formulas (Section 5.1).  
- Why it matters: They quantify memory saturation, compute utilization, latency–throughput trade‑offs, energy cost, and compression in a way FLOPs/params alone cannot.

3) Quantified, cross‑stage trade‑offs and scale dependence  
- Fundamental finding: “No single technique achieves Pareto optimality” (Figure 2; Section 2.1).  
  - Example: MoE improves accuracy and reduces per‑token FLOPs but inflates memory by ~40% and adds routing overhead; in their measurements, a `1.5B×8` MoE has `AMU = 76.53 GB` vs `44.82 GB` for a dense `1.5B` (Table 5).  
  - Example: `int4` reduces memory/energy up to ~3.9× with a ~3–5% average task score drop (Table 9; Abstract).  
- Scale dependence: `RSLoRA` surpasses `LoRA` on models ≥14B (Table 7); freezing layers yields the lowest latency across sizes (Tables 7–8).

4) Transferability to LVMs and VLMs  
- What’s new: The same efficiency tricks validated on LLMs often help in vision (Tables 10–13).  
- Why it matters: It suggests common efficiency principles across modalities; e.g., `MQA/GQA` improve FID in DiT backbones (Table 10), and `RSLoRA/PISSA` scale well for large VLMs (Table 12).

## 5. Experimental Analysis
- Evaluation setup (Sections 5.2–5.5)
  - Models: LLaMA‑3 series, Qwen‑2.5 (7B/14B/32B), DeepSeek‑R1 distill variants (1.5B/8B/14B), Phi‑3.5/4, Yi‑34B; LVMs (Stable Diffusion 3.5‑Medium, Wan 2.1); VLMs (LLaVA‑1.5, Qwen‑VL‑7B, InternVL‑3‑38B, QvQ‑72B) (Tables 2, 15).
  - Datasets: FineWeb‑Edu‑350B (pretraining sweeps), OpenO1‑SFT (PEFT), Medical‑o1‑reasoning‑SFT (medical PEFT), ChatQA (VLMs), plus LVM training corpora (Sections 5.2, 6.1–6.3; Table 13).

- Main quantitative results and what they mean

  Architecture pretraining (Section 5.3; Figures 3a–c; Tables 3–6)
  - Efficient attention (Table 3, 1.5B scale)
    > `MLA` achieves the best language quality (PPL 7.79), but uses more memory/latency (52.93 GB; 0.2537 s/iter).  
    > `MQA` minimizes memory/latency (AMU 42.24 GB; AL 0.1298 s/iter) with slightly higher PPL (8.23).  
    > `NSA` has the lowest average energy (AEC 598 W) albeit with higher latency (0.5962 s/iter).  
    > `GQA` is a middle ground (PPL 8.09; AL 0.1283 s; AMU 44.87 GB; AEC 652.7 W).

    Interpretation: Choose `MLA` for quality‑critical pretraining, `MQA` for memory‑constrained or latency‑sensitive settings, `NSA` for power‑constrained training (Figure 3a).

  - Positional encoding (Table 4, 1.5B)
    > `RoPE` gives the lowest perplexity (PPL 8.09).  
    > A relative scheme (“Relate”) yields the best efficiency—lower latency (0.1246 s/iter), highest throughput (TT 8.98×10⁻²), and lower `AMU` (43.94 GB)—with a small PPL trade‑off (8.29).

  - Mixture‑of‑Experts vs dense (Table 5)
    > `MoE 1.5B×8 (top‑2)` improves PPL to 7.10 vs dense 1.5B’s 8.09 and even dense 3B’s 7.58, and boosts throughput (TT 1.25×10⁻¹).  
    > But it inflates memory (AMU 76.53 GB) and energy (AEC 692.45 W) and slightly increases latency.

    Takeaway: MoE can be compute‑efficient per token and more accurate, at a memory/energy premium (Figure 3c).

  - Attention‑free alternatives (Table 6)
    > `Mamba` reduces memory and energy by ~25% (e.g., 1.5B: AMU 30.25 GB; AEC 510.64 W) and improves latency (0.1025 s), but with worse PPL (9.48 vs 8.09 baseline).  
    > `RWKV` and `Pythia` show mixed patterns—moderate efficiency gains but larger PPL penalties.

    Trade‑off: attention‑free models can be attractive for strict memory/power budgets, but today typically underperform dense Transformers on PPL.

  PEFT (fine‑tuning) on OpenO1‑SFT (Table 7; Figure 4)
  - Small models (1–3B)
    > `LoRA‑plus` often has the lowest loss under similar memory, e.g., LLaMA‑3.2‑1B loss 0.7442; LLaMA‑3.2‑3B loss 0.5791.  
    > `Freeze` gives the lowest latency by ~3× (e.g., 1B: 0.2542 s/iter vs ~1.16–2.15 for others) with good loss (0.6425), making it ideal when interactivity matters.

  - Mid/large models (≥14B)
    > `RSLoRA` outperforms `LoRA` in both loss and latency; e.g., Qwen‑2.5‑14B loss 0.4126 vs LoRA 0.4795 (Table 7).  
    > `DoRA` tends to have high latency (e.g., up to 8.93 s/iter) despite stable loss—best suited to batch fine‑tuning, not interactive settings.

  - Diminishing returns of full FT at scale
    > Full* FT becomes less attractive as parameters grow (e.g., Mistral‑Small‑24B loss 1.2805 vs PEFT 0.3757–0.3975) and is more energy/memory intensive (Table 7).

  Domain PEFT (Medical‑o1; Table 8)
  - `Freeze` again gives best latency (e.g., 8B: 0.4632 s/iter) and strong loss (1.0120).  
  - `LoRA‑plus` and `RSLoRA` are competitive in loss across sizes; `DoRA` remains latency‑heavy.

  Inference quantization (Table 9; Figure 5; Appendix Table 16)
  - Memory/energy/throughput
    > `int4` substantially reduces memory and often increases throughput: e.g., Qwen‑2.5‑32B `AMU` 48.30 GB vs 71.33 GB (bf16) and `IT` 19.20 vs 17.54 tok/s.  
    > Compression ratios (`MCR`) approach 3.7–3.9× for several models (e.g., DeepSeek‑R1‑14B: 3.6965; Phi‑4: 3.9157).

  - Accuracy impact
    > Average task score typically drops modestly (~3–5 pp): e.g., DeepSeek‑R1‑14B 0.4719 (bf16) → 0.4361 (int4). Appendix Table 16 shows per‑benchmark variations (e.g., MATH is more sensitive).

  - bf16 vs fp16
    > On GH200/H200, bf16 often has lower latency/energy than fp16 (e.g., DeepSeek‑R1‑1.5B AEC 144.39 W bf16 vs 158.96 W fp16; Figure 5).  
    > Notably, a few models (e.g., Phi‑4, Yi‑34B) show higher AEC under int4—serving stack and kernel maturity matter (Table 9).

  Cross‑modal transfer (LVMs and VLMs; Section 6)
  - Efficient attention for LVMs (Table 10)
    > `MQA/GQA` consistently improve image generation quality: DiT‑XL/2 FID drops from 19.47 (MHA) to 8.93 (MQA) and 8.71 (GQA).  
    > NSA/MLA show mixed results depending on model size and efficiency target.

  - MoE for LVMs (Table 11)
    > Improves FID and throughput while raising AMU/AEC—e.g., DiT‑B/4 FID 68.38 → 45.62 and TT 1.39e‑5 → 2.09e‑5, with AMU 15.51 → 18.95 GB.

  - PEFT for VLMs (Table 12; Figure 7c)
    > `LoRA‑plus` is best for LLaVA‑1.5 (loss 0.9716).  
    > `PISSA` leads for Qwen‑VL‑7B (loss 0.3156) and Intern‑VL‑3‑38B (0.3635).  
    > `RSLoRA` wins at 72B scale (QvQ‑Pre‑72B loss 0.1434), echoing the LLM trend that `RSLoRA` scales better.

  - Fine‑tuning LVMs (Table 13)
    > Full FT gives the best loss on Wan 2.1‑1.3B (0.104) and SD3.5‑Medium (0.204), but `GLORA`/`LoHA` provide strong trade‑offs with much lower AMU/latency.

- Do the experiments support the claims?
  - Yes, the study repeatedly demonstrates quantifiable trade‑offs and scale dependence across stages and modalities (Figures 2–5, 7; Tables 3–13). The use of energy and memory metrics, in addition to latency/throughput/accuracy, strengthens real‑world relevance.
  - Ablations/comparisons exist across families, sizes, and techniques; per‑metric radar and bar plots (Figures 3–5, 7) align with tabled numbers.

- Failure cases and caveats
  - Int8 inference is excluded due to GH200 kernel instability (Section 5.5).  
  - Some int4 cases show higher energy (Phi‑4, Yi‑34B), highlighting that quantization benefits depend on kernels and serving stack (Table 9).

## 6. Limitations and Trade-offs
- Coverage limits (Section 8.1)
  - The study focuses on three efficiency axes (architecture, PEFT, quantization). It omits, e.g., long‑context KV‑cache strategies, retrieval and alignment (RLHF) cost/quality trade‑offs, speculative decoding, and advanced serving schedulers.
- Hardware specificity
  - Results are on GH200/H200 clusters; behavior on TPUv4/TPU‑v5p, consumer GPUs, or heterogeneous clusters may differ (Section 8.1).
- Scale in pretraining sweeps
  - Architecture pretraining results are at 0.5B–3B; conclusions may shift at ≥10B during pretraining (Table 3). The MoE memory/power overhead could scale differently for larger systems.
- Metrics and economics
  - Metrics are averaged; they don’t capture transient spikes or tail latencies in multi‑tenant serving (Section 8.1). Economic cost models (cloud pricing, amortization) are not included.
- PCU scope
  - `PCU` is meaningfully reported for PEFT; pretraining/inference showed near‑constant utilization on their stack (footnote in Section 5.1), so PCU comparisons there are intentionally limited.

## 7. Implications and Future Directions
- How this changes the landscape
  - EfficientLLM provides a common yardstick to reason about LLM efficiency trade‑offs across the entire lifecycle. It moves the discussion from isolated improvements to multi‑objective, hardware‑aware optimization (Figures 2–5; Sections 2, 5–6).
- Practical guidance distilled from the benchmark
  - Architecture pretraining (Table 3; Figure 3a):  
    - Use `MQA` for memory/latency‑bound settings; `MLA` for quality‑first; `NSA` when power is the primary constraint.  
    - Consider relative PE schemes for efficiency (Table 4), and MoE when memory budget allows (Table 5).
  - Fine‑tuning (Tables 7–8; Figure 4):  
    - `LoRA‑plus` is a solid default for ≤3B; `RSLoRA` for ≥14B; `Freeze` for the lowest latency; avoid `DoRA` for interactive use.  
    - Full FT yields diminishing returns at very large scales.
  - Inference (Table 9; Figure 5):  
    - Prefer `int4` when memory/throughput dominate and small accuracy drops are acceptable; otherwise `bf16` is a strong default on Hopper GPUs.
- Research directions (Section 8.2)
  - Vector‑valued scaling laws balancing loss with latency/memory/energy across compute budgets.  
  - Memory‑aware MoE routing and unified theory for compute–memory trade‑offs.  
  - Robust post‑training quantization for long contexts (activation outliers), including joint weight–activation schemes.  
  - Hardware‑aware auto‑schedulers that jointly optimize data/tensor/pipeline/expert parallelism in heterogeneous clusters.  
  - Cross‑modal PEFT designs that generalize across language, vision, audio, and tooling.

- Downstream applications
  - Cost‑constrained deployments (on‑device, edge), green AI initiatives (minimizing AEC), rapid domain adaptation (PEFT recipes per scale), and capacity planning (predictable memory/latency profiles for a chosen technique mix).

> Bottom line: The benchmark shows efficiency is a multi‑objective optimization problem with context‑dependent optima. With standardized metrics and cross‑stage evidence, EfficientLLM turns folklore into data-driven guidance for building faster, cheaper, and greener foundation models.
