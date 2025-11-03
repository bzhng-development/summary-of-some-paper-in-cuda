# Every Attention Matters: An Efficient Hybrid Architecture for Long-Context Reasoning

**ArXiv:** [2510.19338](https://arxiv.org/abs/2510.19338)

## 🎯 Pitch

This paper introduces Ring-linear-2.0, a new family of large language models that combine linear attention for most layers with periodic softmax attention, dramatically slashing memory and compute costs for long-context reasoning. By further integrating advanced FP8 fused kernels and a robust training–inference alignment procedure for reinforcement learning, these models achieve up to 10x faster inference and robust, stable training over exceptionally long contexts—all while preserving state-of-the-art reasoning performance. This innovation makes practical, economical deployment of long-context LLMs vastly more feasible in real-world applications such as agents, code assistants, and knowledge retrieval.

---

## 1. Executive Summary
This paper proposes a long-context large‑language‑model family, `Ring-linear-2.0`, that mixes two kinds of attention: fast, constant‑state “linear attention” for most layers and standard softmax attention sparingly. It couples this architecture with FP8 kernel fusions and a training–inference alignment procedure for reinforcement learning (RL), yielding large speedups and stable long‑horizon training while maintaining competitive reasoning quality (Figures 6–8; Tables 2–3).

## 2. Context and Motivation
- Problem addressed
  - Long‑context reasoning (agents, code generation, retrieval, test‑time scaling) demands processing sequences well beyond 32K tokens. Standard softmax attention has quadratic compute with sequence length and a Key–Value (`KV`) cache that grows linearly with output length, inflating both compute and memory I/O during decoding (Section 2.2; Figure 4).
- Why it matters
  - Practical deployments increasingly rely on “test‑time scaling”—letting models “think longer” by generating more tokens—and on long input contexts. Without better attention mechanisms and efficient kernels, cost and latency make these use cases uneconomical (Introduction; Section 2.2.4).
- Prior approaches and gaps
  - Pure linear-time sequence models (RetNet, Lightning Attention, Mamba, Gated Linear Attention, DeltaNet—Section 1) reduce theoretical complexity and keep a constant‑size state. However:
    - Pure linear models often underperform at large scale and on retrieval (Section 1; Section 2.2.2).
    - Their advantages tend to appear only beyond ~8K tokens, while much pretraining still sits in the 4–8K range; MoE computation can dominate and blunt efficiency gains during pretraining (Section 1).
    - Community kernels for linear attention were fragmented and lacked support for advanced inference features like speculative decoding with tree masks (Section 3.4).
  - Hybrid designs mixing linear and softmax attention exist (e.g., Minimax M1, GPT‑OSS, Qwen3‑Next; Section 1) but lack a systematic exploration of the optimal ratio, full‑stack kernel fusions in FP8, and training–inference alignment for long‑horizon RL.
- Positioning
  - `Ring-linear-2.0` targets this gap with: (a) a tuned hybrid ratio found by scaling‑law analysis (Section 2.2.2; Figure 3), (b) infrastructure‑level FP8 kernel fusion (“linghe”) for training and inference (Section 3; Figure 6), and (c) a module‑by‑module alignment method that stabilizes RL for long Chain‑of‑Thought outputs (Section 5.2; Figures 10–12).

## 3. Technical Approach
At a glance, the system has three pillars: a hybrid linear–softmax architecture, compute kernels and FP8 training/inference optimizations, and a training pipeline (continued pretrain → SFT → RL) with training–inference alignment.

- Model architecture (Section 2; Figure 2; Table 1)
  - The model is organized into layer groups. Each group has `M` linear‑attention blocks followed by one softmax attention block (“Grouped Query Attention”, `GQA`).
    - `GQA` reduces KV cache by sharing keys/values across multiple query heads.
  - The feedforward is a high‑sparsity `MoE` (Mixture‑of‑Experts) with a 1/32 activation ratio (only ~3% of expert parameters are active per token), optimized with choices like shared experts, sigmoid routing without auxiliary load‑balancing loss, and Multi‑Token Prediction (`MTP`) heads (Section 2.1; Figure 2).
  - Two open models:
    - `Ring-mini-linear-2.0`: 16.4B total parameters, 1.6B active (957M non‑embedding), `d_model=2048`, `n_layers=20`, `n_experts=256`, `top_k=8`, `n_heads=16`, `n_kv_heads=4`, hybrid ratio `1:4` (one softmax after 4 linear layers), 128K context (Table 1).
    - `Ring-flash-linear-2.0`: 104.2B total, 7.4B active (6.1B non‑embedding), `d_model=4096`, `n_layers=32`, same MoE settings, `n_heads=32`, hybrid ratio `1:7`, 128K context (Table 1).

- Linear attention core (Section 2.2.1; Equations 1–4)
  - Intuition: replace the pairwise attention computation with a recurrent accumulation of “key–value outer products,” so decoding only needs a constant‑size state per head.
  - Formalization:
    - Standard attention output: `O = Q (K^T V)` (Eq. 1).
    - With fixed decay `λ` (Lightning Attention form), the t‑th output is
      - `o_t = q_t ∑_{s≤t} λ^{t-s} k_s^T v_s` (Eq. 2),
      - maintained via a recurrent state `kv_t = λ kv_{t-1} + k_t^T v_t`, `o_t = q_t (kv_t)` (Eq. 3).
    - The state `kv_t ∈ R^{d×d}` is the constant‑size KV cache for linear attention (Eq. 4). This makes memory independent of sequence length in decoding.

- Why a hybrid architecture (Section 2.2.2–2.2.3)
  - Pure linear attention underperforms retrieval/extrapolation; a small number of softmax layers repairs this (Section 2.2.2).
  - The paper fits compute–loss scaling laws (Chinchilla‑style) for different ratios (Figure 3).
    - > “the hybrid linear architecture consistently outperforms the pure softmax architecture… a large layer group size (e.g., `M=7`) performs well under high FLOP budgets” (Figure 3).
  - Final choices: `M=7` for the 104B model, `M=4` for the 16B model (Section 2.2.2; Table 1).

- Design choices inside linear attention (Section 2.2.3)
  - `Grouped RMSNorm`: normalize locally per tensor‑parallel rank to avoid all‑reduce (cuts communication).
  - `Partial RoPE` after `QK` normalization: RoPE on half the dimensions lowered LM loss by ~0.004 (Section 2.2.3; Figure 2).
  - `Head-wise decay`: a power‑law schedule across heads improved LM loss by ~0.04 over linear decay and helped downstream tasks (Section 2.2.3).

- Decoding cost model (Section 2.2.4; Figure 4)
  - Decoding speed is bound by memory bandwidth to the KV cache/state. Linear attention’s constant‑size state plus `GQA` yields substantially smaller memory access growth than softmax, `GQA`, or `MLA` alone as sequence length increases (Figure 4).

- Kernel and FP8 training/inference optimizations (Section 3)
  - Fused kernels to minimize memory traffic and activation size (Figure 5), including:
    - Linear attention gate fusion (transpose + grouped norm + sigmoid gate).
    - `permute/unpermute` fused with padding by modifying the routing map.
    - `QK` normalization fused with split, partial RoPE, and transpose.
    - Router casting done in‑kernel (BF16→FP32) to reduce I/O.
    - A single Triton prefill kernel for linear attention by re‑partitioning Q/K and V (instead of 2–4 kernels; Section 3.1).
  - FP8 training (Section 3.2):
    - Quantization fusion: e.g., make `SiLU` output quantized tensors directly, halving I/O from ~`8MN` to `4MN` for input shape `[M, N]`.
    - State‑aware recomputation: during backward, only produce quantized `x^T` needed for `dw = x^T y`; during forward (non‑recompute) only produce quantized `x`—reduces redundant compute and quantize ops.

- Inference system integration (Section 3.4; Figures 7–8)
  - Integrated fused linear attention kernels into SGLang and vLLM, expanding mode coverage and throughput.
  - Implemented the first linear attention kernel supporting tree masks, enabling speculative decoding for a hybrid linear model; available in the offline framework `Flood` and being ported to SGLang (Section 3.4).

- Training pipeline and schedulers (Section 4; Figure 9)
  - Start from softmax‑based `Ling-base-2.0-20T` checkpoints; convert each linear layer’s `QKV` into MHA weights along head dimension and randomly initialize new gate/RMSNorm parameters (Section 4).
  - Two‑stage continued pretraining:
    1) “Continued training” at 4K context to recover base capabilities—600B tokens (`mini`), 1T tokens (`flash`).
    2) “Mid‑training”: extend contexts 4K → 32K → 128K while increasing high‑quality reasoning data.
  - Use a `WSM` (Warmup‑Stable‑Merge) LR schedule (merge mid‑training checkpoints instead of explicit decay), recovering ≥98% of the base models across categories (Figure 9).

- Post‑training: SFT and RL with alignment (Section 5)
  - SFT data mixes hard reasoning (math, code, science, logic) with general tasks and re‑synthesized function‑calling to match broader calling patterns (Section 5.1).
  - RL training on carefully filtered, appropriately difficult data at long contexts (often 64K) to avoid truncation and reach a higher ceiling (Section 5.2).
  - Training–inference alignment (Section 5.2.1; Figures 10–12):
    - Systematically feed the same inputs through training and inference engines and match activations layer‑by‑layer, fixing modules that drift.
    - Key fixes:
      - Use FP32 state for linear‑attention KV accumulation at inference; BF16 causes error growth (Figure 11).
      - FP32 `lm_head` math via a custom GEMM that takes BF16 inputs but accumulates in registers to control cost.
      - Ensure identical `RMSNorm` eps, FP32 residuals, and unfused residual‑norm in both stacks.
      - Match RoPE numerics between PyTorch training and inference operators.
      - Use the same attention backend (e.g., FlashAttention) and align prefill–decode numerics.
      - Make MoE deterministic: stable `topk`, fixed permutation/summation orders, same operators.
    - RL objective: with alignment, PPO can safely use rollout‑engine probabilities for clipping (Eq. 5), instead of bias‑inducing recomputed training probabilities (Eq. 6). This improves rewards and stabilizes the probability gap (Figure 12).

## 4. Key Insights and Innovations
- Hybrid ratio chosen by scaling laws (Figure 3)
  - Novelty: explicitly fit compute–loss curves to select the number of linear layers per softmax layer, rather than ad‑hoc design. The optimal `M` depends on FLOP budget; they fix `M=7` (flash) and `M=4` (mini) to balance efficiency and quality (Section 2.2.2).
  - Significance: keeps most of the model in linear attention without losing retrieval/extrapolation, outperforming pure softmax in the scaling regime explored.

- Constant‑state linear attention with targeted architectural tweaks (Section 2.2.1–2.2.3)
  - Novelty: a practical Lightning‑style linear attention implementation with `Grouped RMSNorm`, `QK` norm, partial RoPE, and head‑wise power‑law decay, each justified by measured LM‑loss improvements.
  - Significance: delivers linear scaling in sequence length for compute and constant state for decoding, addressing the I/O bottleneck (Figures 4, 7–8).

- FP8 end‑to‑end kernel fusion and “state‑aware recompute” (Section 3; Figure 5)
  - Novelty: quantization fused into surrounding ops (e.g., `SiLU`), single‑kernel prefill for linear attention, and recomputation that emits different quantized tensors depending on whether a backward pass needs `x` or `x^T`.
  - Significance: large training throughput gains—up to +77% on the 16B model and +57% on the 104B model (Figure 6)—and higher inference throughput (Figures 7–8).

- Training–inference alignment for long‑horizon RL in hybrid linear MoE models (Section 5.2; Figures 10–12)
  - Novelty: a systematic, module‑level alignment process (KV cache precision, lm_head precision, norms, RoPE, attention backend, MoE determinism) tailored to hybrid linear+MoE, coupled with using rollout probabilities in PPO after alignment (Eq. 5).
  - Significance: prevents RL collapse, yielding steadily increasing reward and test scores (Figure 13).

- First tree‑mask–compatible linear‑attention kernel for speculative decoding (Section 3.4)
  - Novelty: enabling tree‑based speculative decoding in a linear‑attention setting, previously blocked by mask support.
  - Significance: reduces small‑batch latency for long-context decoding while keeping linear‑attention efficiency.

## 5. Experimental Analysis
- Evaluation setup (Section 6.1; Tables 2–3; Figure 1)
  - 17 benchmarks covering:
    - Mathematical reasoning: AIME’24/’25, OlympiadBench, CNMO’24, LiveMathBench, TheoremQA.
    - Coding/agents: HumanEval+, MBPP+, LiveCodeBench (LCB), Codeforces Elo, Spider, BFCL‑Live.
    - General reasoning/knowledge: GPQA‑Diamond, SciBench, DROP, MuSR, Multi‑LogiEval.
  - Comparisons:
    - For `Ring-mini-linear-2.0`: vs `Ring-mini-2.0`, `Qwen3‑8B‑Thinking`, `GPT‑OSS‑20B‑Medium` (Table 2).
    - For `Ring-flash-linear-2.0`: vs `Ring-flash-2.0`, `Qwen3‑32B‑Thinking`, `Gemini‑2.5‑Flash`, `GPT‑OSS‑120B‑Medium`, `Seed‑OSS‑36B‑Instruct`, `Qwen3‑Next‑80B‑A3B‑Thinking` (Table 3; Figure 1).

- Main quantitative results (Tables 2–3)
  - `Ring-mini-linear-2.0` (Table 2):
    - Math: matches or slightly trails `Ring-mini-2.0` by small margins (e.g., AIME’25 73.65 vs 74.06; TheoremQA 69.69 vs 70.09) while staying competitive with `Qwen3‑8B‑Thinking`.
    - Coding: wins on Codeforces Elo (83.84 vs Qwen’s 73.31) and is competitive on HumanEval+ / MBPP+/LCB.
    - General reasoning: mixed—e.g., GPQA‑Diamond 65.69 (near GPT‑OSS‑20B 65.53), strong on DROP 83.20 but behind `Ring-mini-2.0` (88.55).
    - Takeaway: despite using mostly linear attention and only 1.6B active params, performance remains comparable to strong 8–20B baselines across tasks.
  - `Ring-flash-linear-2.0` (Table 3; Figure 1):
    - Math: very strong. AIME’25 86.51 (close to `Ring‑Flash‑2.0` 86.98; near top vs `Qwen3‑Next‑80B‑A3B` 87.80), OlympiadBench 87.36, CNMO’24 84.98.
    - Coding: LCB 70.37 (better than `Ring‑Flash‑2.0` 70.76≈, `Qwen3‑32B` 62.33, and `Gemini‑2.5‑Flash` 61.40), Codeforces Elo 90.24 (high, similar to `Ring‑Flash‑2.0` 90.23).
    - General reasoning: GPQA‑Diamond 74.49 (above `Qwen3‑32B` 68.40 but below `Gemini‑2.5‑Flash` 82.80 and `Qwen3‑Next‑80B‑A3B` 77.20); DROP 89.66 (competitive).
    - Overall: broad competitiveness with larger models, with standout math/coding strengths (Figure 1 highlights AIME’25, LCB, Codeforces, GPQA).

- Efficiency results
  - Training throughput (Figure 6):
    - `Ring-mini-linear-2.0`: +21% with fused kernels vs baseline FP8; +77% when fused kernels allow `TP=1` (from `TP=2`) at same micro‑batch size.
    - `Ring-flash-linear-2.0`: +25% with fused kernels; +57% after also doubling micro‑batch (from 1→2) and adjusting pipeline parallelism.
  - Inference throughput (Figures 7–8):
    - Prefill: linear models overtake softmax models past 8K context and accelerate rapidly; beyond 128K they reach “>2.5×” vs `Ring‑2.0` and “>8×” vs baseline dense models (Figure 7a; Figure 8a, text in Section 3.4).
    - Decode: past 4K generated tokens, linear models exceed `Ring‑2.0`; at 64K they achieve “>2× vs Ring‑2.0” and “>10× vs baseline” (Figure 7b; Figure 8b, Section 3.4).
  - KV/state memory access scaling (Figure 4): hybrid linear grows much slower with sequence length than `GQA` or `MLA`, explaining decode throughput gains.

- Ablations and robustness
  - Hybrid ratio/scaling laws (Figure 3): larger `M` (more linear per softmax) performs well at higher FLOP budgets; hybrid curves dominate pure softmax.
  - Linear‑module design tweaks (Section 2.2.3): partial RoPE (~0.004 LM loss improvement) and power‑law head‑wise decay (~0.04) show measured benefits.
  - RL stability from alignment (Figures 10–13):
    - > Each added fix (KV cache, `lm_head`, `RMSNorm`, attention backend, RoPE) “contributes to improved training efficiency and stability” (Figure 10).
    - > After alignment, PPO with rollout probabilities yields higher rewards and keeps the training‑inference probability gap small (Figure 12).
    - Training reward and test metrics (AIME’25, LCB) rise steadily over RL steps (Figure 13).

- Do the experiments support the claims?
  - Yes for efficiency: clear, repeated speedups in training throughput (Figure 6) and long‑context prefill/decode (Figures 7–8), with a mechanistic explanation (Figure 4).
  - Yes for capability retention: continued pretraining restores ≥98% of base performance (Figure 9).
  - Yes for RL stability: alignment ablations and PPO probability choices directly correlate with reward stability (Figures 10–12).
  - Quality vs top competitors: strong but not always SOTA (e.g., GPQA vs Gemini‑2.5, Table 3); strengths are most pronounced in math/coding and long‑context efficiency.

## 6. Limitations and Trade-offs
- Softmax layers remain bottlenecks
  - Even though few, softmax layers still incur quadratic cost and KV cache growth, limiting the full potential of the hybrid approach (Conclusion).
- Memory overhead from attention heads
  - To maintain effectiveness, the linear module keeps the same head count for `Q`, `K`, and `V`, which “brings heavy memory overhead” (Conclusion).
- Benefits are context‑length dependent
  - Linear attention’s advantages emerge strongly past ~8K context (Section 1; Figures 7–8). In shorter contexts, dense/MoE compute may dominate, muting gains.
- Knowledge retention trade‑offs
  - Continued pretraining shows slight deficits in reasoning/professional knowledge vs the original base (Figure 9), likely due to knowledge forgetting.
- Engineering complexity and reproducibility
  - The gains rely on extensive kernel fusions, FP8 quantization strategies, and a careful alignment pipeline (Sections 3 and 5). Portability to all stacks and hardware may require significant effort.
- Evaluation breadth
  - While 17 benchmarks are covered, robustness to adversarial retrieval, multimodal long‑context tasks, or non‑English domains is not detailed.

## 7. Implications and Future Directions
- Field impact
  - This work shows that long‑context reasoning can be made economical by combining linear attention (for constant‑state decoding) with just enough softmax layers to preserve retrieval/extrapolation—chosen via scaling laws (Figure 3). The full‑stack engineering (FP8 kernels, inference integration, RL alignment) demonstrates that architectural ideas must be matched with systems work for impact.
- Follow‑up research enabled
  - Attention research:
    - Reduce memory by decoupling head counts for `Q`, `K`, `V` in linear modules (Conclusion).
    - Explore adaptive or learned head‑wise decay schedules and partial‑RoPE strategies.
    - Push toward even sparser softmax layers (or smarter placement) while preserving retrieval.
  - Systems:
    - Generalize tree‑mask linear attention kernels and speculative decoding across toolchains (SGLang/vLLM).
    - Extend FP8 fusion patterns to more activations and attention variants; co‑design with compiler‑level fusion.
  - RL and alignment:
    - Standardize module‑wise alignment tests and metrics; make rollout‑probability PPO the default once aligned.
    - Study alignment for multi‑GPU/MoE routing nondeterminism at larger scales and for multimodal models.
- Practical applications
  - Long‑document assistants, multi‑step code agents (LCB/Codeforces results), retrieval‑augmented generation with very long contexts (128K+), and test‑time scaling settings that require thousands of “thinking” tokens—all with substantially lower serving cost and higher throughput (Figures 7–8; Section 3.4).

> Bottom line: `Ring-linear-2.0` combines a scaling‑law‑tuned hybrid architecture with deep systems optimizations and alignment‑aware RL to deliver long‑context efficiency without sacrificing reasoning quality—showing a credible path to economical, capable long‑context LLMs.
