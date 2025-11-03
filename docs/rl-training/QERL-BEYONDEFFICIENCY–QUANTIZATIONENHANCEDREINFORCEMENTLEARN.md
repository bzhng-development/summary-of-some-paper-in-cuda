# QERL: BEYOND EFFICIENCY – QUANTIZATION ENHANCED REINFORCEMENT LEARNING FOR LLMS

**ArXiv:** [2510.11696](https://arxiv.org/abs/2510.11696)

## 🎯 Pitch

QeRL introduces a transformative reinforcement learning framework for LLMs by synergizing ultra-efficient 4-bit NVFP4 quantization, low-rank adaptation (LoRA), and an adaptive quantization noise (AQN) scheduler. This trio not only slashes memory use and rollout times—enabling end-to-end RL speedups of 1.2–1.5× (and up to 2× in generation)—but also boosts exploration and final accuracy, achieving results on par with full-parameter fine-tuning on demanding reasoning tasks. The innovation makes scalable RL training feasible even for massive models (up to 32B) on a single H100 GPU, unlocking faster, cheaper, and more capable LLMs for real-world deployments.

---

## 1. Executive Summary
QeRL is a training framework that combines 4‑bit weight quantization (NVFP4) with low‑rank adapters (LoRA) and a new Adaptive Quantization Noise (AQN) mechanism to make reinforcement learning (RL) for large language models (LLMs) both faster and more effective. It speeds up the rollout-heavy parts of RL by 1.2–1.5× end‑to‑end and over 2× in pure generation throughput while reducing memory, and it matches or surpasses 16‑bit LoRA and QLoRA accuracy—reaching 90.8% on GSM8K with a 7B model and 77.4% on MATH‑500—approaching full fine‑tuning (Abstract, Fig.1; Table 1; Table 2).

## 2. Context and Motivation
- Problem addressed
  - RL substantially boosts LLM reasoning but is computationally expensive: multiple models run concurrently (e.g., policy and reference in GRPO), long rollouts dominate time and memory, and RL is sample‑inefficient (Sec.1).
  - Existing parameter‑efficient fine‑tuning (PEFT) methods (e.g., LoRA) reduce trainable parameters but not rollout cost; quantized rollouts help but often require keeping multiple precisions in memory to correct precision mismatches (e.g., FlashRL uses importance sampling), increasing memory (Sec.1).
  - QLoRA uses 4‑bit NormalFloat (NF4), but its packed format requires lookup table mapping before matmul, slowing generation; empirically, QLoRA rollouts are 1.5–2× slower than LoRA (Sec.1, Fig.2b).

- Why it matters
  - Reasoning tasks need long generation traces; rollout time dominates RL training. Cutting rollout memory and latency directly lowers cost and enables larger models (Sec.1).
  - The paper also uncovers a counter‑intuitive effect: properly controlled quantization noise increases policy entropy, improving exploration and rewards in RL (Fig.3, Fig.5, Sec.3.2)—contrasting SFT, where quantization noise usually hurts training.

- Prior approaches and gaps
  - LoRA-based RL (e.g., Tina) reduces trainable parameters but leaves rollouts slow (Sec.1).
  - Quantized rollouts (e.g., FlashRL) require both low‑bit and 16‑bit models simultaneously to fix logit precision mismatches, increasing memory (Sec.1).
  - QLoRA’s NF4 reduces memory but slows generation; that is particularly harmful in RL where rollouts dominate (Sec.1, Fig.2b).

- Positioning
  - QeRL replaces NF4 with NVFP4, a 4‑bit floating‑point format with per‑block FP8 scaling that modern NVIDIA GPUs accelerate. It keeps only one quantized model in memory, uses LoRA for gradients, and introduces AQN to convert static quantization noise into a controllable exploration signal (Fig.2c; Sec.3.1–3.3).

Definitions (selective)
- `Rollout`: the process of sampling model completions used to compute rewards and policy gradients in RL.
- `LoRA`: Low‑Rank Adaptation; adds trainable low‑rank matrices to frozen weights to reduce trainable parameters (Sec.2).
- `QLoRA`: LoRA training on quantized weights using the NF4 format (Dettmers et al., 2023a).
- `NVFP4`: a 4‑bit floating‑point weight format with FP8 block scalers (E4M3) and FP32 global scale; optimized kernels exist on NVIDIA Hopper/Blackwell GPUs (Sec.2).
- `GRPO` and `DAPO`: policy optimization variants for LLM RL. GRPO uses group‑relative advantages and a KL penalty; DAPO removes the KL penalty and uses token‑level policy gradients, enabling more exploration (Sec.3.1).

## 3. Technical Approach
QeRL has three pillars (Fig.2c; Sec.3):

1) NVFP4 weight quantization for rollout and prefilling, with LoRA for learning
- Weights are quantized to NVFP4 and kept frozen; LoRA adapters are trained for updates (Sec.2, Eq.2).
- NVFP4 dequantization uses dual scaling (Sec.3.3, Eq.6):
  - A global FP32 scale `S_FP32`.
  - Per‑block FP8 scales `S_E4M3` (per 16‑element block).
  - Dequantization reconstructs `Ŵ = S_FP32 · (S_E4M3 ⊙ W̃)`, where `W̃` is 4‑bit packed weights.
- This format allows fast NVFP4 × BF16 matmuls through the Marlin kernel (Sec.3.1, Sec.4.3), avoiding NF4’s LUT unpacking overhead (Sec.1).
- Result: one quantized model serves both rollout and logit evaluation (no second‑precision copy), and LoRA layers carry gradients (Fig.2c).

2) Adaptive Quantization Noise (AQN) for exploration
- Observation: quantization introduces a (normally harmful in SFT) bias/noise that flattens logits and increases output entropy, which is beneficial in RL as an exploration mechanism (Sec.3.2; Fig.3; Fig.5, Fig.14).
- Challenge: quantization noise is static/deterministic; RL needs exploration that ramps down over time.
- Mechanism (Sec.3.3):
  - For each quantized linear layer, sample a channel‑wise Gaussian noise vector per forward pass: `Z_noisy ~ N(0, σ^2 I)` of shape `1×d`, where `d` is the input dimension (Eq.7).
  - Combine with inherent quantization noise: `Δε' = Z_noisy + (Ŵ − W)`. This creates dynamic parameter noise without touching int4 storage.
  - Control σ with an exponential decay scheduler across K stages (Eq.8): high early exploration, lower later exploration.

3) Zero‑overhead noise injection via RMSNorm “merging”
- Directly adding high‑precision noise to quantized weights breaks int4 kernels and increases parameters.
- Insight: leveraging the identity `X · (Z_noisy + Ŵ) = X·Z_noisy + X·Ŵ` (Eq.9), channel‑wise additive noise on activations can be absorbed into the scale vector of the following LayerNorm (here RMSNorm), creating an equivalent multiplicative noise on weights (Eq.10, Appendix G Eq.11–12).
  - Implemented as `RMSNorm_noise(x) = w_noise ⊙ x / sqrt(mean(x^2)+δ)`, with `w_noise = Z_noise + w`.
  - This turns additive `Z_noisy` into row‑wise multiplicative noise `(Z_w + I)` on the linear layer’s weights after normalization (Appendix G).
- Noise sharing practicalities (Sec.3.3; Fig.6):
  - Share the same noise across layers that share the same RMSNorm: in attention `W_q, W_k, W_v` share one, and in the feed‑forward network `W_gate, W_up` share another. This maintains zero extra parameters and keeps fast NVFP4 × BF16 kernels intact.

RL objective and training loop
- GRPO objective (Eq.3) maximizes clipped policy ratios weighted by group‑relative advantages and a KL penalty to a reference policy; DAPO removes KL (Sec.3.1).
- Group‑relative advantage (Eq.4) normalizes each sampled completion’s reward within its group of G samples.
- Training loop (Algorithm 1; Appendix F):
  - Stage 0: `σ=0` (only quantization noise).
  - Later stages: sample `π_θ_old ← π_θ + N(0, σ^2)` and draw G completions per prompt, compute rewards, compute per‑token advantages, and update the LoRA parameters with GRPO/DAPO.
  - σ follows the exponential schedule from `σ_start` to `σ_end` across K stages (Eq.8).

Why these design choices
- NVFP4 over NF4: removes the decode bottleneck and leverages hardware acceleration (Sec.1, Fig.2b vs Fig.2c; Sec.4.3).
- Dynamic noise (AQN) instead of static: matches RL’s need for exploration early and exploitation later (Sec.3.2–3.3).
- RMSNorm merging: preserves fast kernels and parameter efficiency; multiplicative noise is known to be effective for exploration (Sec.3.3 and citations therein).

## 4. Key Insights and Innovations
- Quantization‑as‑exploration for RL (conceptual advance)
  - Novel insight: quantization noise increases policy entropy and accelerates reward growth in RL, opposite to its negative effect in SFT (Sec.3.2; Fig.3, Fig.5, Fig.14).
  - Significance: turns a supposed “bug” (quantization noise) into a “feature” for RL exploration without extra cost.

- Adaptive Quantization Noise (AQN) with zero‑overhead injection (methodological innovation)
  - Dynamic, per‑forward channel‑wise Gaussian noise with exponential decay (Eq.7–8), merged into RMSNorm to avoid any additional parameters or kernel changes (Eq.9–10; Fig.6; Appendix G).
  - Significance: controllable exploration that is easy to deploy in existing Transformer blocks and compatible with NVFP4 kernels.

- NVFP4+LoRA training path that removes QLoRA’s rollout bottleneck (systems contribution)
  - A single NVFP4 policy model is used for both rollout and logits evaluation—no mixed‑precision duplication or importance sampling (Fig.2c).
  - Result: 1.2–1.5× end‑to‑end RL speedups vs 16‑bit LoRA and avoidance of QLoRA’s 0.7–0.9× slowdowns (Table 3; Sec.4.3). Generation throughput improvements exceed 2× on larger models (Fig.11).

- Practical scaling result: 32B RL on one H100 80GB (systems milestone)
  - The framework “enables RL training of a 32B LLM on a single H100 80GB GPU” (Abstract; Tables 7–8 show OOM for BF16 LoRA while QeRL runs).

Incremental vs fundamental
- Incremental: swapping NF4 for NVFP4 and using Marlin kernels is a systems optimization.
- Fundamental: reframing quantization noise as useful exploration and designing AQN to control it is a conceptual and algorithmic contribution with broader implications for RL‑with‑LLMs.

## 5. Experimental Analysis
Evaluation setup
- Tasks and data (Sec.4.1):
  - GSM8K (7.5k math word problems; G=8 samples/prompt).
  - BigMath (122k math problems; G=16), with difficulty levels 3–5 (7B/14B) and 4–5 (32B).
- Models: Qwen2.5‑3B/7B/14B/32B‑Instruct (no math‑specialized pre‑finetuning) (Sec.4.1).
- Quantization and kernels:
  - NVFP4/MXFP4 weight‑only quantization via AWQ calibration (256 sequences of length 2048) (Sec.4.1).
  - NF4 for QLoRA baselines (default config).
  - Marlin kernel accelerates NVFP4 × BF16 in rollout/prefill (Sec.3.1; Sec.4.3).
- RL algorithms: GRPO (GSM8K) and DAPO (BigMath) (Sec.4.1; Sec.3.1).
- Metrics and inference: pass@1 accuracy on GSM8K, MATH‑500, AIME 2024/2025, AMC 23; T=0.6, top‑p=0.95, max length 4096 (Sec.4.1).
- Hyperparameters: batch 128, 8/16 samples per prompt, off‑policy updates for GSM8K and on‑policy for BigMath, clip range [0.2, 0.28], noise range `σ_start=1e−2` to `σ_end=5e−4`, LoRA rank mostly 32, LR 1e−5 for QeRL/QLoRA and 5e−6 for BF16 LoRA (Table 4).
- Hardware: H100‑80GB; speedups measured on 1 GPU; final large‑scale training on 8 GPUs (Sec.4.1).

Main quantitative results
- GSM8K (GRPO, Table 1)
  - 3B:
    - BF16 LoRA: 76.1
    - NVFP4 LoRA + AQN (QeRL): 83.7
    - BF16 full fine‑tuning: 84.4
  - 7B:
    - BF16 LoRA: 88.1
    - NVFP4 LoRA + AQN: 90.8
    - BF16 full fine‑tuning: 91.2
  - Takeaway: QeRL surpasses 16‑bit LoRA by +7.6 (3B) and +1.7 (7B), and approaches full FT within ≤0.7 points.

- BigMath → evaluation on MATH‑500, AIME, AMC (DAPO, Table 2)
  - 7B (Average over four benchmarks):
    - BF16 LoRA: 35.7
    - NVFP4 LoRA + AQN: 36.4
    - BF16 full: 37.3
    - Notable: MATH‑500 = 77.4 (QeRL) vs 77.0 (BF16 LoRA) and 77.4 (full).
  - 14B (Average):
    - BF16 LoRA: 40.2
    - QeRL: 42.0
    - BF16 full: 43.3
    - Notable: AMC‑23 = 57.5 (QeRL) > 55.0 (full), showing targeted strength (Table 2).
  - 32B (Average):
    - BF16 LoRA: 42.2
    - QeRL: 45.6
    - BF16 full: 46.2
    - Notable AIME‑25 = 19.2 (QeRL) vs 13.3 (BF16 LoRA).

- Reward and entropy dynamics
  - Faster reward growth: on BigMath, QeRL’s reward rises sharply within ~200 steps vs >500 for 16‑bit LoRA (Fig.7; Appendix H Fig.12–13).
  - Higher policy entropy: quantized LoRA consistently shows higher entropy than 16‑bit LoRA, especially early in training (Fig.5, Fig.14), explaining better exploration (Sec.3.2).

- Efficiency and memory
  - End‑to‑end RL speedup (Table 3):
    - 7B: ×1.5/×1.4/×1.2 (batch 2/4/8) vs BF16 LoRA; QLoRA slows to ×0.7–0.9.
    - 14B: ×1.4/×1.2/×1.2; QLoRA slows to ×0.7–0.9.
  - Generation throughput (Fig.11; Tab.9):
    - 14B, rank‑16: 95.3 tokens/s (QeRL) vs 65.4 (LoRA) → ~1.46×; bar chart notes “2.1×” relative to QLoRA.
    - 32B, rank‑16: 58.0 vs 34.0 → ~1.7×; bar chart notes “2.3×” relative to QLoRA.
  - Memory footprint (Table 3):
    - 7B: 5.9 GB (QeRL) vs 15.2 GB (BF16 LoRA).
    - 14B: 10.6 GB (QeRL) vs 29.6 GB (BF16 LoRA).
  - Single‑GPU 32B RL: BF16 LoRA runs OOM while QeRL trains with gradient checkpointing (Tables 7–8).

Ablations and robustness
- AQN effectiveness (Fig.8): injecting adaptive noise (vs default static quantization noise) yields steadier reward growth and pushes improvements near convergence.
- Noise scheduler (Fig.9; Appendix Fig.15): exponential decay gives the most stable late‑stage improvements compared to linear, cosine, logarithmic.
- LoRA rank (Fig.10; Tab.9): ranks 16/32/64/128 show similar trends; rank‑16 converges slightly faster and is efficient.
- Learning rate sensitivity (Appendix Fig.16–17): QeRL remains stable at LR=3e‑5 and converges ~2× faster; 16‑bit LoRA tends to collapse at high LR.

Do the experiments support the claims?
- Yes, for the stated scope. Multiple model sizes (3B–32B), two RL algorithms (GRPO/DAPO), strong baselines (BF16 LoRA, full FT, QLoRA), and extensive ablations support both the efficiency and the “quantization‑as‑exploration” story (Sec.4.2–4.3; Fig.4, Fig.5, Fig.7–11; Tables 1–3, 5–9).
- The most compelling pieces are:
  - Reward/entropy dynamics (Fig.5, Fig.7, Fig.14).
  - Speed/memory tables showing both faster rollouts and smaller footprints (Table 3; Fig.11; Tables 5–9).
  - Accuracy parity or gains on math benchmarks (Tables 1–2).

## 6. Limitations and Trade-offs
- Hardware dependence
  - The speedups rely on NVFP4 support and optimized kernels (Marlin) available on NVIDIA Hopper/Blackwell GPUs (Sec.2; Sec.4.3). On other hardware, benefits may reduce.

- Task/domain scope
  - Evaluations focus on math reasoning benchmarks (GSM8K, MATH‑500, AIME, AMC). No results on code, general dialogue, or multimodal tasks (Sec.4.1; Sec.5 Conclusion; Appendix K).

- Model scale beyond 32B
  - While a 32B model trains on a single H100 80GB with QeRL (Tables 7–8), performance and feasibility for ≥70B remain untested (Appendix K).

- RL‑specific dynamics and tuning
  - AQN requires scheduling (σ_start, σ_end, stages K) and careful integration; poorly tuned noise can destabilize training (Sec.3.3; Fig.9).
  - Results were measured with specific GRPO/DAPO settings (no explicit entropy or KL losses in Table 4). Generality to other RL regimes (e.g., different reward structures or constraints) is unproven.

- Quantized base models underperform before RL
  - Weight‑only quantization reduces raw accuracy (Table 1, rows without LoRA), and the gains manifest after RL training. This is a training‑time remedy rather than a general quantization improvement for inference-only scenarios.

- Compute remains significant
  - RL is still costlier than SFT; although QeRL reduces rollout cost and memory, total training still involves long horizons and large batches (Appendix K).

## 7. Implications and Future Directions
- Shift in how we view quantization in RL
  - By demonstrating that quantization noise can serve as a built‑in exploration mechanism, QeRL bridges compression and RL theory. This invites systematic study of “noise‑aware” RL schedules and quantization formats tailored for exploration (Sec.3.2–3.3; Fig.3, Fig.5).

- Practical impact: lower barrier to large‑model RL
  - Memory and speed gains make RL feasible for larger models and on fewer GPUs. The single‑GPU 32B result suggests RL fine‑tuning could become routine in more labs and production teams (Abstract; Tables 7–8).

- Extensions and research directions
  - Beyond math: test on code generation, tool use, and general instruction‑following to assess how quantization‑driven exploration interacts with different reward structures.
  - Broader noise design: explore alternative noise distributions, layer‑wise schedules, or adaptive controllers that react to online reward/entropy signals rather than pre‑set schedules (Fig.9 hints scheduler choice matters).
  - Combine with activation quantization/QAT: integrate low‑bit activations or quantization‑aware training to further reduce memory and possibly shape exploration at activation level.
  - Algorithmic integration: unify AQN with explicit entropy bonuses or KL constraints (e.g., marrying AQN with DAPO’s no‑KL regime vs GRPO’s KL penalty; Sec.3.1).
  - Scaling studies: push beyond 32B to 70B+ to test whether QeRL’s speed/memory gains and exploration benefits persist.

Key takeaway
> QeRL reframes quantization from a necessary evil into a controllable exploration tool for RL, and couples it with a hardware‑aligned 4‑bit format (NVFP4) and zero‑overhead noise injection. The result is both faster and often more accurate RL training than 16‑bit LoRA and QLoRA, with evidence across 3B–32B models and multiple math benchmarks (Fig.1; Tables 1–3).
