# EfficientLLM: Efficiency in Large Language Models

**ArXiv:** [2505.13840](https://arxiv.org/abs/2505.13840)
**Authors:** Zhengqing Yuan, Weixiang Sun, Yixin Liu, Huichi Zhou, Rong Zhou, Yiyang Li, Zheyuan Zhang, Wei Song, Yue Huang, Haolong Jia, Keerthiram Murugesan, Yu Wang, Lifang He, Jianfeng Gao, Lichao Sun, Yanfang Ye
**Institutions:** 

## 🎯 Pitch

EfficientLLM presents the first comprehensive benchmark evaluating the real-world efficiency of large language models across their lifecycle, using hardware-aware metrics on production GPUs. By quantifying trade-offs like memory vs. quality vs. energy, it empowers practitioners to make informed decisions for model deployment, impacting efficiency and sustainability in AI operations.

---

## 1. Executive Summary (2–3 sentences)
EfficientLLM introduces the first end-to-end, large-scale benchmark that measures the real-world efficiency of large language models (LLMs) across three lifecycle stages: architecture pretraining, fine-tuning, and inference via quantization. Using six hardware-aware metrics gathered on production-class GPUs, it evaluates 100+ model–technique pairs and distills actionable trade-offs (e.g., memory vs. quality vs. energy) that practitioners can use to choose attention variants, PEFT methods, and bit-widths for deployment (Sections 1, 2; Figure 1; Section 5.1).

## 2. Context and Motivation
- Problem addressed
  - Despite a flood of efficiency techniques for LLMs, there has been no comprehensive, empirical, and hardware-grounded benchmark that compares them across the full lifecycle—pretraining architecture choices, parameter-efficient fine-tuning (PEFT), and inference-time quantization—using consistent metrics and modern accelerators (Sections 1, 2).
  - Existing studies often evaluate a single technique in isolation, on limited model scales, or via theoretical FLOPs rather than measured latency, memory, and energy (Section 1).

- Why this matters
  - Real deployments care about memory (can the model fit?), latency (is it responsive?), throughput (can it serve traffic?), and energy/carbon cost (can we afford to run it?). FLOPs or “params” are poor proxies for these concerns (Section 5.1).
  - Different stakeholders need different guidance: model designers (architecture pretraining), applied practitioners (fine-tuning under hardware budgets), and deployment engineers (quantization trade-offs) (Section 1; Figure 1).

- Prior approaches and gaps
  - Work on efficient attention, MoE, PEFT, and quantization exists, but results are incomparable due to different setups, scales, and missing energy measurements; many focus on idealized metrics rather than end-to-end measurements (Sections 1, 4, 7).
  - No prior benchmark runs all three axes together (architecture, tuning, inference) on a production-scale cluster while reporting memory, compute utilization, latency, throughput, energy, and compression consistently (Sections 1, 5).

- Positioning
  - EfficientLLM defines a unified evaluation taxonomy and six orthogonal metrics—Average Memory Utilization (AMU), Peak Compute Utilization (PCU), Average Latency (AL), Throughput (TT/ST/IT), Average Energy Consumption (AEC), and Model Compression Rate (MCR)—and applies them across >100 model–technique pairs from 0.5B to 72B parameters on GH200/H200 GPUs (Sections 1, 5.1, 5.2; Tables 2, 3–13).

## 3. Technical Approach
This is an empirical benchmarking framework, not a single model. It is organized along three axes, each with concrete techniques, datasets, hardware, and metrics.

- Measurement framework and metrics (Section 5.1)
  - Memory: `AMU` (Average Memory Utilization): time-averaged device memory used (Eq. 1).
  - Compute utilization: `PCU` (Peak Compute Utilization): ratio of actual vs. peak GPU utilization, averaged over time (Eq. 2). The authors note PCU is informative mainly for PEFT runs; training and inference PCU hovered near saturation in their infra, so PCU is reported for fine-tuning scenarios (footnote in Section 5.1.1).
  - Latency: `AL`: mean time per training iteration or inference request (Eq. 3).
  - Throughput: `TT` (tokens/param/s for pretraining), `ST` (samples/param/s for fine-tuning), and `IT` (tokens/s for inference) (Eqs. 4–6).
  - Energy: `AEC` (Average Energy Consumption): time-average power draw (Eq. 8).
  - Compression: `MCR`: size_original / size_compressed × (perf_compressed / perf_original), so aggressive compression is penalized if it hurts quality (Eq. 9).

- Experimental infrastructure and coverage
  - Architecture pretraining: Qwen2.5-based models at 0.5B, 1.5B, 3B parameters, trained on FineWeb-Edu (350B tokens) on 48× GH200 GPUs, using Megatron-Core with 3D parallelism (Section 5.3 “Hardware…”).
  - Fine-tuning (PEFT): Llama-3.2 {1B, 3B}, Llama-3.1 8B, Qwen-2.5 {7B, 14B}, Mistral Small 24B, Mistral 7B, trained on OpenO1-SFT and Medical-O1 on 8× H200 GPUs with LLaMA-Factory; full fine-tuning uses DeepSpeed ZeRO-3 Offload (Section 5.4; Table 7, Table 8).
  - Inference (quantization): DeepSeek-R1 distills, Qwen2.5, Phi-3.5/4, Yi-34B; precisions bf16, fp16, int4; experiments on GH200 nodes (Section 5.5; Table 9). Int8 was excluded due to kernel instability on Hopper (Section 5.5 “Note on Int8 Quantization”).
  - Extension beyond text: Large Vision Models (DiT-based Stable Diffusion 3.5; Wan 2.1) and Vision-Language Models (LLaVA-1.5, Qwen2.5-VL-7B, InternVL-3-38B, QvQ-Pre-72B) to test transferability of techniques (Section 6; Tables 10–13; Figure 7).

- Techniques benchmarked
  - Efficient attention (Section 4.4.2; Table 3; Figure 3a)
    - `MQA` (Multi-Query Attention): all heads share one K and V, reducing the KV-cache size and memory bandwidth.
    - `GQA` (Grouped-Query Attention): partition heads into groups; heads in a group share K,V (trade-off between MQA and full MHA).
    - `MLA` (Multi-Head Latent Attention): compress K,V into a low-rank latent cKV = h W_DKV (dc ≪ d) and reconstruct per-head K,V with up-projections, substantially shrinking the cache (Section 4.4.2).
    - `NSA` (Native Sparse Attention): three-branch decomposition—compression (global summary), selection (choose top blocks), and sliding window—combined by learned gates; designed for hardware-aligned sparsity (Section 4.4.2).
    - Also: `MoBA` (Mixture of Block Attention) with top-k block selection for efficiency (Section 4.4.2).
  - Positional encodings (Section 4.4.3; Table 4; Figure 3b)
    - RoPE; absolute (fixed and learnable variants); “Relate” (a relative approach; Section 5.3.2 labels).
  - Sparse models (MoE) (Section 4.4.4; Table 5; Figure 3c): Top-k gated experts per token.
  - Attention-free sequence models (Section 4.4.5; Table 6): Mamba (SSM), RWKV (RNN-style), and “Pythia” entry (linear attention baseline in this paper’s ablation set).
  - PEFT methods (Section 4.5.2; Tables 7–8; Figure 4)
    - LoRA; LoRA-plus (different LR for A,B); `RSLoRA` (rank-stabilized scaling α=1/√r); `DoRA` (decomposes weight into direction+scale, applies low-rank to direction); `PiSSA` (SVD-initialized low-rank); `Freeze` (freeze first 8 layers); `Full*` (DeepSpeed ZeRO-3 full fine-tuning).
  - Inference quantization (Section 4.6; Table 9; Figure 5): bf16, fp16, int4; “Average performance” aggregates task scores (Section 5.1.4 and Appendix Table 16 list MMLU-Pro, BBH, GPQA, IFEval, MATH, MuSR).

- Normalization note
  - Many visualizations are min–max normalized per metric so that “higher is better” even for latency/energy (footnote on p. 2 and Figure captions). Always check the raw tables for absolute values (e.g., Tables 3–6, 7–13).

## 4. Key Insights and Innovations
1) A multi-objective, hardware-measured efficiency lens (fundamental)
- What’s new: Six complementary metrics (AMU, PCU, AL, TT/ST/IT, AEC, MCR) capture the real deployment bottlenecks, not just FLOPs or params (Section 5.1).
- Why it matters: It reveals trade-offs invisible to FLOPs—for example, NSA consumes less energy but has higher latency; MLA improves quality but inflates memory (Table 3).

2) No single method is Pareto-optimal; trade-offs are quantifiable and context-dependent (fundamental)
- Evidence: The consolidated ranking in Figure 2 and per-technique results (Tables 3–6, 7–9) show each method improves at least one metric while regressing another. Example: int4 offers ~3.6–3.9× compression but ~3–5 percentage point average-task performance drop (Table 9; Figure 5).
- Why it matters: Practitioners must pick techniques based on the dominant bottleneck: memory, latency, energy, or quality (Section 2.1).

3) Scale- and task-dependent PEFT optima (novel empirical finding)
- For 1–3B models: LoRA-plus often gives the lowest loss under tight AMU (Table 7: Llama‑3.2‑1B loss 0.7442; 3B: LoRA-plus 0.5791).
- ≥14B: RSLoRA is more efficient on loss and latency (Table 7: Qwen‑2.5‑14B RSLoRA loss 0.4126 vs LoRA 0.4795; also competitive latency).
- Freeze consistently achieves the lowest tuning latency (e.g., Llama‑3.2‑1B AL 0.2542 s/iter; Llama‑3.1‑8B 0.7369 s/iter) at some quality cost (Table 7).
- Significance: Guidance flips with scale—do not extrapolate PEFT winners from small models to large ones (Figure 4; Section 2.1).

4) Architecture choices with clear resource/quality profiles (practical innovation)
- MQA: best memory–latency frontier (e.g., 1.5B AMU 42.24 GB; AL 0.1298 s/iter) (Table 3).
- MLA: best perplexity (1.5B PPL 7.79; 3B PPL 7.29) but higher memory (Table 3).
- NSA: lowest energy (e.g., 0.5B AEC 594 W; 1.5B 598 W) but much higher latency (Table 3).
- MoE: better quality at less FLOPs per token but more VRAM and routing overhead (Table 5; Section 2.1). At 1.5B×8, PPL 7.10 vs dense 3B 7.58, but AMU rises to 76.53 GB (Table 5).

5) Transfer to vision (LVMs) and vision-language (VLMs) (incremental but useful)
- GQA/MQA reduce FID substantially vs MHA in DiT LVMs (e.g., DiT-XL/2 FID: MHA 19.47 → GQA 8.71) (Table 10).
- MoE improves FID and throughput for LVMs at higher AMU/AEC (Table 11).
- PEFT for VLMs: PISSA strong at 7–38B, RSLoRA shines at 72B (Table 12; Figure 7c).

## 5. Experimental Analysis
- Evaluation methodology
  - Datasets: FineWeb-Edu 350B tokens for pretraining (Section 5.2.2); OpenO1-SFT and Medical-O1 SFT for fine-tuning (Sections 5.2.2, 5.4); standard reasoning/coding/math/multitask benchmarks for inference aggregation (MMLU-Pro, BBH, GPQA, IFEval, MATH, MuSR, Section 5.1.4; Appendix Table 16).
  - Hardware: GH200 cluster (pretraining, inference) and H200 node (fine-tuning) with NVLink/InfiniBand; Megatron-Core for pretraining; LLaMA-Factory for PEFT; DeepSpeed ZeRO-3 for Full* (Sections 5.3–5.5).
  - Metrics: AMU, PCU, AL, TT/ST/IT, AEC, MCR (Section 5.1). Visuals are normalized (Figure 3–5; Figure 7), but tables report raw values.

- Main quantitative results (selected, with citations)
  - Efficient attention trade-offs (Table 3; Figure 3a)
    - 1.5B: MLA best quality (PPL 7.79) but higher AMU (52.93 GB); MQA lower memory (42.24 GB) and low latency (0.1298 s/iter); NSA lowest energy (598 W) but far higher latency (0.5962 s/iter).
    - 3B: MLA again best PPL 7.29; MQA/GQA remain memory-friendly; NSA energy-minimal but slow.
  - Positional encodings (Table 4; Figure 3b)
    - At 1.5B: RoPE has the best PPL (8.09). “Relate” (relative PE) achieves the best efficiency: lowest AL (0.1246 s/iter), lowest AMU (43.94 GB), highest TT, and lower AEC (646 W) though slightly worse PPL (8.29).
  - MoE vs dense (Table 5; Figure 3c)
    - 1.5B×8 (Top‑2): PPL 7.10 (better than dense 3B 7.58 and dense 1.5B 8.09), but AMU 76.53 GB (vs 43.94–44.82 GB for dense), AL increases (0.142 s/iter), and AEC increases (692 W). Throughput TT improves to 1.25e‑1.
  - Attention-free models (Table 6)
    - At 1.5B: Mamba lowers AMU (30.25 GB vs 44.82 GB) and AEC (511 W vs 653 W) and has lower latency (0.1025 vs 0.1280 s/iter), but PPL worsens (9.48 vs 8.09). Similar trends at 0.5B and 3B.
  - PEFT efficiency (Table 7; Figure 4a)
    - Small models (1–3B): LoRA-plus/PISSA yield lowest loss (e.g., Llama‑3.2‑3B PISSA 0.5137; LoRA‑plus 0.5791). Freeze gives lowest AL (e.g., 0.4252 s/iter at 3B).
    - Medium–large (8–24B): Freeze still fastest to iterate (Llama‑3.1‑8B AL 0.7369; Mistral‑Small‑24B Freeze 1.4815 s/iter), but RSLoRA often best loss (Mistral‑Small‑24B 0.3818).
    - At 14B (Qwen‑2.5‑14B): RSLoRA achieves best loss (0.4126) with competitive latency (2.7855 s/iter), better than LoRA (0.4795) and LoRA-plus (0.4621) (Table 7).
  - Medical fine-tuning (Table 8)
    - Freeze again offers the lowest latency (e.g., Llama‑3.2‑1B AL 0.2123; Llama‑3.1‑8B 0.4632) and strong loss among methods; DoRA has much higher latency without clear wins.
  - Quantization (Table 9; Figure 5; Appendix Table 16)
    - Memory/throughput/energy: int4 consistently reduces AMU by ~1.5–2× vs 16-bit and increases tokens/s, while lowering AEC.
      - Example: Qwen2.5‑32B AMU 71.33 (bf16) → 48.30 (int4); IT 17.54 → 19.20 tokens/s; AEC 279 W → 215 W; MCR ≈ 3.69 (Table 9).
      - DeepSeek‑R1 Distill Qwen‑14B: AMU 51.83 → 34.21; IT 24.74 → 26.40; AEC 212 W → 191 W; “Avg perf.” 0.4719 → 0.4361 (≈3.6 points drop) (Table 9).
    - bf16 vs fp16: bf16 generally has lower latency and energy on Hopper (e.g., DeepSeek‑R1 Distill Qwen‑1.5B AEC 144 W vs 159 W; tokens/s 39.68 vs 37.70) (Table 9; Section 2.1).
    - Task sensitivity: MATH is more fragile under quantization (Appendix Table 16 shows bigger drops for int4 on MATH for Qwen‑2.5‑14B: 0.1700 → 0.0529).
  - Vision (LVM/VLM) transfer (Tables 10–13; Figure 7)
    - Attention in LVMs: DiT‑XL/2 FID drops from 19.47 (MHA) to 8.71 (GQA) or 8.93 (MQA) (Table 10).
    - LVM MoE: DiT‑B/4 130M→130M×8 improves FID 68.38 → 45.62 and ups throughput, but increases AMU and energy (Table 11).
    - VLM PEFT: Qwen2.5‑VL‑7B PISSA best loss 0.3156; Intern‑VL‑3‑38B PISSA 0.3635; QvQ‑Pre‑72B RSLoRA 0.1434 (Table 12).

- Do the experiments support the claims?
  - Yes for trade-offs: Side-by-side tables make the compromises explicit (e.g., MLA quality vs memory; Freeze latency vs loss; int4 memory/energy vs small score drop).
  - Caveats:
    - Architecture pretraining tests are at 0.5–3B scale; extrapolation to 30–70B is suggested but not empirically verified (Tables 3–6).
    - Some plots are normalized; always consult the raw tables for absolute magnitudes (Figures 3, 5, 7 notes).
    - Int8 was excluded due to instability on Hopper; thus int4 comparisons omit an important real-world baseline (Section 5.5).

- Ablations/robustness
  - Multiple attention mechanisms and PEFT variants tested across several model sizes, plus cross-modality checks (Sections 5.3–5.5, 6).
  - Limitations explicitly documented (Section 8.1).

## 6. Limitations and Trade-offs
- Scope and scale
  - Architecture pretraining results are for 0.5–3B; behavior may differ at 30–70B+ where KV-cache, bandwidth, and routing costs dominate differently (Tables 3–6).
  - Some important topics are “out of scope”: RLHF/alignments costs, test-time acceleration (e.g., speculative decoding), and detailed systems-level training optimizations (Section 7; Section 8.1).

- Metrics and normalization
  - Visual figures normalize metrics so higher=better; this can visually minimize large absolute latency/energy gaps—cross-check tables (Figure 3, Figure 5, Figure 7; footnote on normalization).
  - PCU is reported mainly for PEFT because pretraining/inference PCU clustered near saturation in their infra (Section 5.1.1).

- Hardware dependence
  - Findings (e.g., bf16>fp16) are tied to Hopper-generation GPUs with bf16 Tensor Core preference; different accelerators could shift conclusions (Sections 2.1, 5.5, 8.1).

- Quantization omissions
  - Int8 instability on GH200 means the widely used int8 baseline is absent; int4 comparisons are strong but incomplete (Section 5.5).

- MoE memory inflation and system complexity
  - While MoE lowers active FLOPs and improves quality, VRAM usage and routing overhead increase notably (e.g., AMU 76.53 GB for 1.5B×8 vs 43–45 GB dense) (Table 5). This stresses memory capacity and expert load balancing (Sections 2.1, 4.4.4, 5.3.3).

## 7. Implications and Future Directions
- How this changes the landscape
  - Provides a practical “efficiency compass”: a shared set of metrics, datasets, and hardware-measured results to pick methods based on the dominant bottleneck—memory, latency, energy, or accuracy (Section 1; Section 5).
  - Shifts discussion from FLOPs to measurable deployment outcomes; e.g., recommending MQA when memory is tight, RSLoRA beyond ~14B for better loss–latency, and int4 when small accuracy drops are acceptable (Sections 2, 5).

- Follow-up research enabled/suggested (Section 8.2)
  - Multi-objective scaling laws (accuracy–latency–energy–memory Pareto curves) rather than single-objective compute-optimality.
  - Memory-aware MoE routing and KV-cache–aware attention for ultra-long contexts (sparse routing under hard memory ceilings).
  - Robust post-training quantization for long sequences (joint weight–activation quantizers that handle outliers).
  - Benchmarks that integrate alignment (RLHF/DPO) cost and test-time strategies (speculative decoding, early exit) with the same metrics.
  - Hardware-aware auto-schedulers for heterogeneous clusters (auto-tuning data/tensor/pipeline/expert parallelism).

- Practical applications
  - On-device or memory-constrained inference: choose `MQA` (Table 3) and `int4` (Table 9) to maximize tokens/s per GB and reduce energy.
  - Latency-sensitive interactive fine-tuning: use `Freeze` (lowest AL in Tables 7–8), accepting moderate loss increase.
  - Quality-first offline pretraining: use `MLA` (lowest PPL in Table 3); consider `RoPE` for best PPL and `Relate` when training speed/energy dominate (Table 4).
  - High-throughput serving under budget: prefer `bf16` over `fp16` on Hopper (lower AL and AEC; Table 9) and `int4` if small score drops are acceptable; evaluate task sensitivity (Appendix Table 16—MATH is fragile).

> Three distilled takeaways with numbers:
> - Architecture: 1.5B MLA achieves best PPL 7.79 but AMU 52.93 GB; 1.5B MQA raises PPL to 8.23 but cuts AMU to 42.24 GB and AL to 0.1298 s/iter (Table 3).  
> - Tuning: Qwen‑2.5‑14B `RSLoRA` loss 0.4126 vs `LoRA` 0.4795 and `LoRA-plus` 0.4621; `Freeze` yields the lowest AL 0.6227 s/iter (Table 7).  
> - Inference: Qwen‑2.5‑32B `int4` improves tokens/s 17.54 → 19.20, cuts AMU 71.33 → 48.30, and AEC 279 → 215 W, with “Avg perf.” 0.5523 → 0.5095 (Table 9).

Overall, EfficientLLM provides the most comprehensive, hardware-grounded map to date of where efficiency wins and what you pay for them—separating myths from measured trade-offs and giving practitioners a reliable basis for decisions across pretraining, fine-tuning, and deployment.
