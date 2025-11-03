# QeRL: Beyond Efficiency -- Quantization‑enhanced Reinforcement Learning for LLMs

**ArXiv:** [2510.11696](https://arxiv.org/abs/2510.11696)
**Authors:** Wei Huang, Yi Ge, Shuai Yang, Yicheng Xiao, Huizi Mao, Yujun Lin, Hanrong Ye, Sifei Liu, Ka Chun Cheung, Hongxu Yin, Yao Lu, Xiaojuan Qi, Song Han, Yukang Chen
**Institutions:** NVIDIA, MIT, The University of Hong Kong (HKU), Tsinghua University

## 1. Executive Summary (2–3 sentences)
QeRL is a reinforcement learning (RL) training framework for large language models (LLMs) that combines 4‑bit NVFP4 weight quantization with `LoRA` adapters and introduces an Adaptive Quantization Noise (`AQN`) mechanism to improve both efficiency and performance. It speeds up the RL rollout phase (generation of training trajectories) by up to 1.5× end-to-end for 7B–14B models (Table 3), enables 32B RL training on a single H100 80GB GPU (Table 8), and—crucially—uses quantization noise to raise policy entropy and enhance exploration, matching or surpassing 16‑bit `LoRA` and even rivaling full-parameter finetuning on math benchmarks (e.g., GSM8K 90.8% and MATH500 77.4% for 7B, Table 1–2).

Reasoning for this summary:
- The core idea is an efficiency/performance co-design: hardware-friendly NVFP4 for speed/memory, and controlled noise for exploration. Reported results (Fig. 1, Table 1–3) directly substantiate speedups and accuracy on standard reasoning tasks.

## 2. Context and Motivation
- Problem addressed:
  - RL is essential for enabling LLMs to discover robust multi-step reasoning strategies, but it is expensive in memory and time because RL requires multiple concurrent models (policy, reference), long rollouts, and multi-stage computation (rollouts → rewards → logits → updates) (Introduction, Fig. 2, Sec. 1).
  - The rollout phase (sampling long sequences to compute rewards) is the dominant bottleneck (Sec. 1; Fig. 1, Fig. 2).

- Why this matters:
  - Faster, cheaper RL makes it feasible to post-train larger LLMs for reasoning, with direct impacts on math/logic tasks and broader decision-making applications (Sec. 1).
  - Reducing GPU memory enables training larger models or using fewer devices; e.g., first 32B RL training on a single H100 80GB (Abstract; Table 8).

- Prior approaches and shortcomings:
  - Parameter-efficient finetuning (e.g., `LoRA`, Tina) reduces trainable parameters but does not speed up rollouts because forward passes still run on full-precision weights (Sec. 1; Fig. 2a).
  - Quantized rollout models (e.g., FlashRL) reduce compute but create precision mismatches between rollout and logits models (e.g., 8-bit vs. 16-bit), requiring importance sampling corrections and, often, holding two model precisions in memory—raising memory usage (Sec. 1).
  - `QLoRA` (with NF4) is popular for SFT, but in RL it slows generation by 1.5–2× because NF4 requires LUT unpacking and remapping before matmuls (Sec. 1; “This slowdown occurs because … lookup table …”).

- How this paper positions itself:
  - It switches to hardware-efficient FP4 quantization (`NVFP4`) and integrates it end-to-end in rollout and prefill using the Marlin kernel, while keeping updates in `LoRA`. This directly attacks rollout speed and memory (Fig. 2c; Sec. 3.1).
  - It reframes quantization noise from a liability (in SFT) to a feature: in RL, the noise increases policy entropy and aids exploration; then it adds `AQN` to schedule and localize that noise over training (Fig. 3; Sec. 3.2–3.3; Fig. 5).

Reasoning about gaps and positioning:
- The paper’s key insight is that RL’s need for exploration reverses the usual stance on quantization noise. The proposed NVFP4 + `LoRA` + `AQN` combination is crafted to align hardware acceleration (Marlin, NVFP4) with algorithmic advantages (policy entropy, scheduled noise).

## 3. Technical Approach
This section explains how QeRL works, from quantization to RL optimization and noise scheduling.

- Core components and terms (briefly defined where nonstandard):
  - `Rollout`: generating model completions during RL to collect rewards.
  - `Prefill`: the initial prompt processing (key/value cache build) before autoregressive decoding.
  - `NVFP4`: a 4‑bit floating-point format with a global FP32 scale and per-block FP8 (E4M3) scalers; block size 16; optimized on NVIDIA Hopper/Blackwell (Sec. 2 “NVFP4 Quantization”).
  - `Marlin` kernel: a mixed-precision autoregressive inference kernel that accelerates dequantization×matmul for formats like NVFP4 (Frantar et al., 2024; Sec. 3.1).
  - `LoRA`: low-rank adapters that update a small number of parameters while freezing base weights (Eq. 2; Sec. 2).
  - `GRPO` and `DAPO`: RL objectives for LLMs with group-based advantages; `GRPO` includes a KL penalty to a reference model; `DAPO` removes it to avoid capping exploration (Sec. 3.1, Eq. 3–4).

A. Weight-only NVFP4 quantization with `LoRA` updates
- Quantization background:
  - Quantization maps FP weights W∈R^{d×k} to a small codebook, then reconstructs them during inference (Eq. 1).
- NVFP4 dequantization (key mechanism for speed):
  - It uses a global FP32 scale `SFP32` and per-block FP8 (E4M3) scalers `SE4M3` for blocks of 16 weights. Dequantizing 4‑bit codes W˜ yields high-precision Wˆ:
    - Wˆ = SFP32 · (SE4M3 ⊙ W˜) (Eq. 6),
    - where ⊙ broadcasts the FP8 scaler across the block. This structure lets GPUs fuse dequantization with matmul efficiently (Marlin), avoiding the LUT overhead of `NF4` (Sec. 2 and Sec. 1).
- Training structure in QeRL (Fig. 2c):
  - Forward passes (rollouts and prefill) use NVFP4 weights with Marlin kernels for speed and memory efficiency.
  - Only `LoRA` matrices (low-rank A, B in Eq. 2) are trained and kept in 16-bit for stable gradients.
  - There is no need to maintain a separate 8/16‑bit full-precision copy for rollouts, avoiding importance-sampling corrections and duplicate memory (contrast with FlashRL; Sec. 1).

Design choice rationale:
- NVFP4 vs. NF4:
  - NF4 relies on a LUT mapping at runtime, adding overhead in decoding; NVFP4’s block scalers are faster to apply on NVIDIA hardware (Sec. 1; Table 3, Fig. 11 show the realized speed).

B. RL optimization (GRPO/DAPO)
- GRPO objective (Eq. 3):
  - For each prompt q, sample G candidate outputs {o_i}. Compute rule-based rewards and token-level advantages A_i,t. The objective includes clipped ratios (PPO-style) and a KL penalty to the reference policy π_ref:
    - J(θ) = E[ (1/G) Σ_i (1/|o_i|) Σ_t min( r_i,t A_i,t, clip(r_i,t,1−α,1+α) A_i,t ) − β D_KL(π_θ || π_ref) ],
    - where r_i,t = π_θ(o_i,t|q) / π_{θ_old}(o_i,t|q), and A_i is normalized over the group (Eq. 4).
- DAPO differences:
  - Uses token-level policy gradients and removes the KL penalty to avoid limiting exploration (Sec. 3.1). This can lead to higher entropy but needs stabilization; QeRL’s scheduled noise supports this.

C. Quantization noise as exploration and `AQN`
- Observation:
  - Quantization error ∆ϵ = Q(θ) − θ acts like parameter noise (Eq. 5), perturbing logits and increasing sampling entropy H(π(·|q)) after softmax (Sec. 3.2).
  - Empirically, 4‑bit models (NVFP4/MXFP4/NF4) trained with `LoRA` have higher entropy than 16‑bit `LoRA` (Fig. 5), which accelerates reward learning (Fig. 4, Fig. 7).

- Problem with static noise:
  - Quantization noise is deterministic once weights are quantized; it does not adapt across training stages (Sec. 3.2).

- `AQN`: converting static to dynamic, controlled exploration (Sec. 3.3):
  1) Per-layer stochastic noise vector:
     - Sample a channel-wise vector Z_noisy ∈ R^{1×d} for each quantized linear layer per forward pass, Z_noisy ~ N(0, σ² I), then define augmented noise:
       - ∆ϵ′ = Z_noisy + (Wˆ − W) (Eq. 7).
  2) Exponential noise schedule:
     - Decay σ over K stages with training step k:
       - σ(k) = σ_start ⋅ (σ_end/σ_start)^{(k−1)/(K−1)} (Eq. 8; Fig. 9 and Appendix Fig. 15 show schedulers; exponential works best later in training).
  3) Noise merging into RMSNorm (zero parameter overhead):
     - To avoid storing explicit noise vectors and to keep NVFP4×BF16 kernels intact, QeRL does not add Z_noisy to quantized weight tensors directly. Instead, it folds Z_noisy into the layer-normalization scale vector:
       - Use distributivity X·(Z_noisy + Wˆ) = X·Z_noisy + X·Wˆ (Eq. 9).
       - Replace RMSNorm’s scaling weight w with w_noise = Z_noise + w (Eq. 10), which turns additive noise on inputs into multiplicative noise on rows of the subsequent weight matrix (Eq. 11–12; Fig. 6 shows where this applies across W_q, W_k, W_v, W_gate, W_up).
     - Initial σ_start is small (1e−2) since multiplicative noise is more sensitive in deep nets (Sec. 3.3).

- Implementation in GRPO training (Appendix Algorithm 1):
  - Split M steps into K stages. At each step, compute σ per Eq. 8, update π_{θ_old} by applying `AQN`, sample rollouts, compute rewards, advantages, and update π_θ (Algorithm 1, lines 5–15).
  - Stage 0 uses σ=0 (pure quantization noise). Later stages gradually reduce σ to transition from exploration to exploitation.

Why this approach over alternatives:
- It keeps a single quantized model for rollout/prefill (no duplicate precision models), leverages hardware-friendly NVFP4 dequantization for speed, and turns quantization noise into a tunable signal that meaningfully increases exploration (entropy) in RL—unlike static quantization or NF4-based `QLoRA`.

Reasoning behind the mechanism:
- The mathematical identity (Eq. 9–12) ensures `AQN` does not break optimized NVFP4×BF16 matmuls while implementing per-layer, per-forward stochasticity. The entropy–reward link is supported by measured entropy curves and faster reward growth (Fig. 4–5, Fig. 7, Appendix Fig. 14).

## 4. Key Insights and Innovations
- Quantization noise improves RL exploration (fundamental insight):
  - What’s new: Contrary to SFT where quantization noise hurts learning, in RL it raises policy entropy and accelerates exploration (Fig. 3). Direct evidence shows higher entropy under 4‑bit quantization than 16‑bit `LoRA` (Fig. 5; Appendix Fig. 14).
  - Why it matters: Higher entropy helps avoid early collapse to suboptimal actions, leading to faster reward growth (Fig. 4, Fig. 7) and better final accuracy (Table 1–2).

- NVFP4 + Marlin speeds rollouts without precision mismatch (practical innovation):
  - What’s new: Use NVFP4 for weight-only quantization with per-block FP8 scaling and a global FP32 scale, enabling fast dequant×matmul on NVIDIA hardware (Sec. 2, Eq. 6).
  - Why it matters: 1.2–1.5× end-to-end training speedups for 7B–14B (Table 3) and >2× rollout throughput for 14B/32B (Fig. 11) while cutting model memory to ~25–30% of BF16 (Table 3, 5–8). It avoids running two precisions simultaneously and avoids importance-sampling corrections (Sec. 1).

- Adaptive Quantization Noise (`AQN`) with noise sharing (algorithmic innovation):
  - What’s new: A staged, exponentially decayed Gaussian noise injected per forward pass and merged into RMSNorm, creating multiplicative noise on rows of subsequent weights with zero extra parameters (Eq. 8–12; Fig. 6; Appendix G).
  - Why it matters: It tunes exploration over training time, resulting in steadier and higher reward curves than static quantization (Fig. 8) and performs best under exponential decay (Fig. 9).

- Enabling 32B RL on a single H100 80GB GPU (system-level achievement):
  - What’s new: Demonstrates feasibility of big-model RL with limited hardware (Abstract; Table 8).
  - Why it matters: Substantially lowers the barrier to scaling RL-based reasoning improvements.

Reasoning on novelty vs. incrementality:
- The exploration finding and `AQN` mechanism are conceptual algorithmic contributions beyond simple efficiency tweaks. NVFP4+Marlin integration is a systems advance enabling those algorithmic benefits to be practical at scale.

## 5. Experimental Analysis
- Evaluation methodology:
  - Datasets:
    - GSM8K: 7.5k problems, samples per prompt G=8 (Sec. 4.1).
    - BigMath: 122k problems, G=16; medium-to-high difficulty (levels 3–5 for 7B/14B; 4–5 for 32B) (Sec. 4.1).
  - RL algorithms:
    - `GRPO` for GSM8K (Eq. 3–4; Sec. 3.1), `DAPO` for BigMath (Sec. 3.1).
  - Backbones:
    - Qwen2.5-3B/7B/14B/32B-Instruct without math-specific SFT (Sec. 4.1).
  - Quantization pipelines:
    - Weight-only NVFP4/MXFP4 via AWQ calibration (256×2048 tokens from OpenThoughts-114k) and Marlin kernels (Sec. 4.1).
    - `QLoRA` uses NF4 with default configs (Sec. 4.1).
  - Metrics and inference:
    - Main metric: Pass@1 accuracy on GSM8K, MATH500, AIME24/25, AMC23; inference with temperature 0.6, top‑p 0.95, max length 4096 (Sec. 4.1).
  - Hyperparameters:
    - See Table 4 (Appendix E): AdamW-8bit, LR=1e−5 for 4‑bit models (5e−6 for BF16 `LoRA` to avoid collapse), batch=128, clip range [0.2, 0.28], response length up to 4096/8192 (GSM8K/BigMath).

- Main quantitative results:
  - GSM8K (GRPO), Table 1:
    - 7B:
      - BF16 base: 76.3.
      - BF16 Full: 91.2.
      - BF16 `LoRA`: 88.1.
      - NVFP4 `LoRA`: 88.5; with `AQN`: 90.8 (+13.5 over base), essentially matching full finetuning.
      - NF4/MXFP4 `LoRA`: 85.0/86.4 (lower than NVFP4 `LoRA`).
    - 3B:
      - BF16 base: 61.2.
      - BF16 Full: 84.4.
      - BF16 `LoRA`: 76.1.
      - NVFP4 `LoRA`: 83.3; with `AQN`: 83.7 (+22.6 over base), within 0.7 of full finetuning.
  - BigMath (DAPO), Table 2:
    - 7B average across MATH500/AIME24/AIME25/AMC23:
      - BF16 `LoRA`: 35.7; NVFP4 `LoRA`: 37.0; `+AQN`: 36.4; BF16 Full: 37.3.
      - On AMC23: NVFP4 `LoRA` reaches 47.5 vs. 42.5 (BF16 `LoRA`) and 45.0 (BF16 Full).
    - 14B average:
      - BF16 `LoRA`: 40.2; NVFP4 `LoRA`: 40.5; `+AQN`: 42.0; BF16 Full: 43.3.
      - On AMC23: `+AQN` achieves 57.5, surpassing BF16 Full 55.0.
    - 32B average:
      - BF16 `LoRA`: 42.2; NVFP4 `LoRA`: 41.4; `+AQN`: 45.6; BF16 Full: 46.2.
      - On AIME25: `+AQN` 19.2 vs. BF16 Full 23.3 vs. BF16 `LoRA` 13.3.
  - Reward/Entropy dynamics:
    - Training rewards rise faster under quantized models than 16‑bit `LoRA`; NVFP4 converges to better final rewards than MXFP4 (Fig. 4).
    - QeRL (NVFP4 `LoRA` + `AQN`) increases rewards in 7B/14B within ~200 steps, whereas 16‑bit `LoRA` often needs 500+ steps (Fig. 7; Appendix Fig. 12–13).
    - Entropy is higher for QeRL over the training trajectory, especially early (Fig. 5; Appendix Fig. 14), consistent with improved exploration.

- Efficiency results:
  - End-to-end training speed (per GRPO step, single H100) and model memory:
    - 7B: QeRL model size 5.9 GB vs. 15.2 GB BF16; speedup ×1.5 (batch=2), ×1.2 (batch=8). `QLoRA` is slower (×0.7–0.8) (Table 3).
    - 14B: QeRL 10.6 GB vs. 29.6 GB; speedup ×1.4 (batch=2), ×1.2 (batch=8). `QLoRA` again slower (×0.7–0.9) (Table 3).
    - 32B: NVFP4 enables single-GPU training with per-step times 10.6–12.2 s; BF16 `LoRA` OOM at similar settings (Table 8).
  - Rollout throughput (tokens/s), batch=1 (Fig. 11; Table 9):
    - 14B: QeRL reaches ≈2× speed vs. BF16 across `LoRA` ranks 16–64.
    - 32B: ≈2.2–2.3× rollout speedups.

- Ablations and robustness:
  - `AQN` benefits:
    - Adding scheduled noise beyond default quantization improves reward stability and final levels (Fig. 8).
  - Noise schedules:
    - Exponential decay performs best later in training (Fig. 9; Appendix Fig. 15 shows schedule curves).
  - `LoRA` rank:
    - Ranks 16–128 show similar reward growth; rank 16 converges slightly faster (Fig. 10); rollout speed declines as rank increases for both BF16 and NVFP4 (Table 9).
  - Learning rate:
    - QeRL tolerates larger LRs (e.g., 3e−5) without collapse due to noise stabilizing updates, achieving ≈2× faster reward growth; BF16 `LoRA` becomes unstable at high LR (Appendix Fig. 16–17).

- Do the experiments support the claims?
  - The paper triangulates evidence:
    - Speed/memory gains are measured both end-to-end and at rollout; NVFP4 consistently outperforms NF4 and BF16 on throughput (Table 3, 5–9; Fig. 11).
    - Accuracy and reward gains are shown across two RL algorithms (GRPO, DAPO), three model scales (7B/14B/32B), and multiple math benchmarks (Table 1–2, Fig. 4, Fig. 7).
    - Entropy measurements align with the exploration mechanism (Fig. 5; Appendix Fig. 14).
  - Caveats:
    - Most tasks are math; broader generalization to coding or non-reasoning tasks is untested (Appendix K).
    - Some comparisons (e.g., full finetuning) are close; on average, QeRL matches or slightly trails BF16 Full, with some tasks where it surpasses (e.g., AMC23 for 14B, Table 2).

Reasoning assessment:
- The consistency between entropy increases, reward dynamics, and final accuracies, plus ablations on noise scheduling and rank, makes a compelling case for the exploration mechanism and for practical speedups from NVFP4. The scope is focused (math RL), so generality should be tested in future work.

## 6. Limitations and Trade-offs
- Assumptions and scope:
  - The exploration benefit hinges on RL objectives where higher entropy helps (e.g., GRPO/DAPO) and on tasks with verifiable rewards (math). It is not shown for other domains (code, general language tasks) (Appendix K).
  - `AQN` assumes that controlled, multiplicative noise via RMSNorm does not destabilize training; careful σ scheduling is required (Sec. 3.3; Fig. 9).

- Hardware and software constraints:
  - The speedups depend on NVFP4 support and Marlin-like kernels on NVIDIA Hopper/Blackwell GPUs (Sec. 2; Sec. 4.3). Portability to other accelerators/formats may limit gains.
  - The framework uses weight-only quantization; activation quantization is not explored here.

- Training stability and hyperparameters:
  - `DAPO` removes KL penalties to encourage exploration; combined with noise, this could risk drifting behavior unless schedules are tuned (Sec. 3.1; Fig. 9).
  - Multiplicative noise is sensitive in deep networks; σ must start very small (1e−2) (Sec. 3.3).

- Scale and generalization:
  - Demonstrated up to 32B; no results for 70B+ (Appendix K). For very large models, new memory/comms bottlenecks may appear.
  - Results concentrate on math reasoning; the benefit of quantization-induced exploration may vary with reward design and domain.

- Efficiency measurement nuances:
  - Reported end-to-end speedups are averaged over early steps (shorter outputs); with longer outputs later in training, speed advantages likely increase (Sec. 4.3), but full-step distributions are not shown.
  - Some speed baselines need gradient checkpointing to fit (14B/32B), which interacts with latency (Table 7–8).

Reasoning on risks:
- The most significant externality is reliance on specific hardware (NVFP4 support). Algorithmically, the main risk is over-exploration if σ is not decayed appropriately or if the task prefers low-entropy policies.

## 7. Implications and Future Directions
- Field-level impact:
  - Makes RL post-training of larger LLMs more accessible by reducing memory and time—e.g., single-GPU 32B RL (Table 8). This can democratize experimentation with RL-based reasoning and accelerate iteration cycles.
  - Reinterprets quantization noise as a controllable exploration tool in RL for LLMs, suggesting a new design axis for low-precision training algorithms.

- Practical applications:
  - Training reasoning-specialized assistants for math competitions, tutoring, and scientific QA, where accurate self-rewarding signals exist.
  - Budget-constrained labs can run RL training with fewer GPUs, enabling broader participation.

- Research directions:
  - Beyond math: Test QeRL on code generation, tool-use, or dialogue RL where reward functions differ; examine if entropy gains translate similarly.
  - Alternative noise mechanisms:
    - Learnable or state-dependent noise schedules; per-layer or per-block σ; colored (non-Gaussian) noise variants linked to token uncertainty.
    - Combine `AQN` with quantization-aware training (QAT) to co-optimize robustness and exploration.
  - Broader quantization formats:
    - Explore MXFP4 trade-offs (fast early rewards, Fig. 4) vs. NVFP4 (better final rewards); hybrid schemes across layers.
  - Larger scales and multi-node:
    - Validate on 70B+ models, measure comms bottlenecks, and integrate with pipeline/tensor parallelism under NVFP4.
  - Safety and alignment:
    - With DAPO’s high-entropy policy and `AQN`, study how to maintain alignment; reintroduce adaptive KL penalties when needed.

Reasoning about implications:
- The central insight—use precision-induced noise to benefit RL—should generalize to other structured compression (e.g., sparsity) if the induced perturbations can be scheduled. The systems result (rollout acceleration via NVFP4) enables practical exploration of such ideas at scale.

> Key evidence anchors:
> - Rollout speed and memory: “QeRL achieves 1.2–1.5× end-to-end training speedups… model sizes 25–30% of BF16” (Table 3; Fig. 11; Table 5–9).
> - Exploration via entropy: “4‑bit quantized models have consistently higher entropy and faster reward growth than 16‑bit `LoRA`” (Fig. 5; Fig. 4; Fig. 7; Appendix Fig. 14).
> - Accuracy: “7B QeRL reaches GSM8K 90.8% and MATH500 77.4%, matching or surpassing 16‑bit `LoRA` and near full finetuning” (Table 1–2).
> - Mechanism: “`AQN` injects Gaussian noise per forward pass and merges it into RMSNorm to create multiplicative noise without parameter overhead” (Eq. 8–12; Fig. 6; Algorithm 1).