# Every Attention Matters: An Efficient Hybrid Architecture for Long‑Context Reasoning

**ArXiv:** [2510.19338](https://arxiv.org/abs/2510.19338)
**Authors:** Ling Team, Bin Han, Caizhi Tang, Chen Liang, Donghao Zhang, Fan Yuan, Feng Zhu, Jie Gao, Jingyu Hu, Longfei Li, Meng Li, Mingyang Zhang, Peijie Jiang, Peng Jiao, Qian Zhao, Qingyuan Yang, Wenbo Shen, Xinxing Yang, Yalin Zhang, Yankun Ren, Yao Zhao, Yibo Cao, Yixuan Sun, Yue Zhang, Yuchen Fang, Zibin Lin, Zixuan Cheng, Jun Zhou
**Institutions:** Ant Group

## 1. Executive Summary (2-3 sentences)
This report introduces a “hybrid” long-context language model architecture named `Ring-linear`, which mixes two attention mechanisms—`linear attention` (constant memory, linear-time) and `softmax attention` (standard transformer attention)—to retain reasoning/retrieval quality while dramatically improving speed and memory at long sequence lengths. Two open-source models—`Ring-mini-linear-2.0` (16B total parameters, 1.6B active) and `Ring-flash-linear-2.0` (104B total, 7.4B active)—achieve substantially lower inference/training cost via system-level kernels (`FP8` fused ops with the `linghe` library), carefully tuned hybrid ratios, and a novel training–inference alignment for reinforcement learning (`RL`), while maintaining competitive or state-of-the-art performance on several long-context reasoning benchmarks (see Figure 1, Tables 2–3).

## 2. Context and Motivation
- Problem addressed:
  - Large Language Models (`LLMs`) increasingly rely on long-context capabilities (handling very long input/output sequences) for agents, code generation, and test-time scaling (inference-time techniques that allocate more compute or tokens to improve accuracy). Traditional `softmax attention` scales quadratically in compute with sequence length `n` and stores a `Key–Value cache (KV cache)` that grows linearly with `n`, creating an inference and I/O bottleneck (Section 1).
  - `Linear attention` variants reduce compute to linear in `n` and use constant-size state, but pure linear models often underperform on retrieval and certain tasks, and their training-time efficiency advantages only dominate at >8K context—where many pretraining corpora still sit around 4–8K (Section 1).
- Why this matters:
  - Long-context support is now a practical requirement for reasoning models (agents, long CoT—“chain-of-thought”—generation, large documents). Reducing inference cost enables longer contexts and heavier test-time scaling, directly improving accuracy and enabling new applications under realistic resource constraints (Section 1).
- Prior approaches and gaps:
  - Existing attention variants—`MHA` (Multi-Head Attention), `MQA` (Multi-Query Attention), `GQA` (Grouped-Query Attention), `MLA` (Multi-head Latent Attention in DeepSeek-V3)—optimize KV-cache memory and bandwidth, but still inherit softmax’s quadratic compute or non-constant state (Section 1).
  - Pure `linear attention` families—RetNet, Lightning Attention, Mamba, Gated Linear Attention, DeltaNet—achieve O(n d^2) compute (for head dimension `d`) and constant state, but can underperform in retrieval and not always excel in the 4–8K pretraining regime (Section 1).
- Positioning:
  - `Ring-linear` adopts a hybrid design that interleaves many `linear attention` layers with occasional `softmax attention` layers, seeking the “best of both”: softmax for precision in retrieval/extrapolation, linear for speed and constant-state decoding. The paper provides (a) principled hybrid ratios via scaling law studies (Figure 3; Section 2.2.2), (b) an efficiency-first MoE (Mixture-of-Experts) backbone guided by prior scaling laws (Section 2.1), and (c) extensive system kernels and `FP8` training/inference optimizations (Section 3).

## 3. Technical Approach
This section explains the architecture, the hybrid attention mechanism, and the system-level engineering that makes it efficient in practice.

- Overall model design (Figure 2; Table 1; Section 2.1):
  - Backbone: A sparse `MoE` transformer with a very low activation ratio (only ~1/32 of parameters active per token), which drastically reduces per-token compute.
    - `MoE` refers to a layer having many parallel “experts” (independent feedforward networks) and a `router` that picks the top-`k` experts per token (`ntop_k` in Table 1). This activates only a small fraction of total parameters per token and improves compute efficiency.
    - Model sizes:
      - `Ring-mini-linear-2.0`: 16.4B total parameters, 1.6B active; 20 layers; `d_model=2048`; `nexperts=256`; `ntop_k=8`; 16 attention heads; 4 KV heads; context length up to 128K; hybrid ratio 1:4 (one softmax layer per 4 linear layers) (Table 1).
      - `Ring-flash-linear-2.0`: 104.2B total, 7.4B active; 32 layers; `d_model=4096`; same `nexperts/top_k`; 32 attention heads; 4 KV heads; context length 128K; hybrid ratio 1:7 (Table 1).
  - Layer grouping: The model is organized into groups of `M+1` layers—`M` linear-attention blocks followed by one `GQA` softmax-attention block (Section 2.2.2). Chosen `M`: 4 (mini) and 7 (flash), based on scaling-law analysis (Figure 3).
  - Additional design features (Figure 2; Section 2.1):
    - `QK-Norm`: Normalization applied to queries (`Q`) and keys (`K`) to stabilize attention scores.
    - `Partial-RoPE`: `RoPE` (Rotary Position Embedding; encodes positions via rotations in feature space) applied to only half of the dimensions in linear attention, improving LM loss by ~0.004 (Section 2.2.3).
    - `Head-wise decay`: Linear-attention state uses a power-law decay per head (versus linear), improving LM loss by ~0.04 and downstream task performance (Section 2.2.3).
    - First block uses dense MLP (not MoE) for stability (Figure 2).
    - `MTP` (Multi-Token Prediction) layers and SFT/RL for reasoning quality (Sections 2.1, 5).

- Hybrid linear attention mechanics (Sections 2.2.1–2.2.2; Equations 1–4; Figure 3):
  - Definitions:
    - `Softmax attention`: Standard transformer attention with pairwise token interactions, O(n^2) compute, KV cache size grows with `n`.
    - `Linear attention`: Rewrites attention as a recurrent update of a fixed-size “state” summarizing past tokens, reducing compute to O(n d^2) and keeping a constant-size state during decoding.
  - Linear attention formulation:
    - The core operation is expressed as `O = Q (K^T V)` (Equation 1), which can be realized in a recurrent manner as “Lightning Attention” (Section 2.2.1).
    - Token-by-token, output `o_t` equals `q_t` times a decayed sum of outer products of past keys and values:
      - `o_t = q_t (kv_t)`, with `kv_t = λ kv_{t-1} + k_t^T v_t` (Equations 2–4).
      - Here, `kv_t` is a `d x d` matrix that serves as a constant-size “state” (KV cache) shared across time. `λ` is a decay factor (head-wise and power-law tuned in this work).
    - Why this helps: Instead of storing all past keys/values (size grows with sequence length), the model stores a single `d x d` matrix per head, updated at each new token. This makes memory use independent of sequence length and reduces memory bandwidth pressure (Section 2.2.1).
  - Hybrid pattern (“M linear + 1 softmax”):
    - The model alternates `M` linear-attention layers with one softmax-attention layer. This retains softmax’s strengths in retrieval/extrapolation while keeping most layers efficient (Section 2.2.2). 
    - Figure 3 shows scaling-law fits (loss vs FLOPs) where hybrid variants outperform pure softmax across budgets; larger `M` (e.g., 7) performs better at higher budgets. Final choices: `M=7` for `Ring-flash-linear-2.0` and `M=4` for `Ring-mini-linear-2.0`.

- Key architectural choices enabling efficiency and quality (Section 2.2.3):
  - `Grouped RMSNorm`: `RMSNorm` (Root Mean Square Layer Normalization; normalizes activations by their RMS) is executed locally per tensor-parallel rank to avoid costly cross-device synchronization (“all-reduce”). This reduces communication in forward/backward (Section 2.2.3).
  - `Partial RoPE` after QK normalization in linear attention reduces LM loss by ~0.004 (Section 2.2.3).
  - `Head-wise power-law decay` in linear attention reduces LM loss by ~0.04 and notably improves downstream performance (Section 2.2.3).

- Decoding cost analysis (Figure 4; Section 2.2.4):
  - In inference decoding, `KV cache` access is often the bottleneck due to limited GPU memory bandwidth. Figure 4 shows how KV/state memory access scales with sequence length for `Hybrid Linear`, `GQA`, and `MLA`. The hybrid-linear’s constant-state design flattens the growth curve, leading to significantly better scaling as length increases (Section 2.2.4).

- System and kernel engineering for performance (Section 3; Figure 5):
  - `Kernel fusion`: Combine multiple GPU operations (kernels) into one to reduce memory reads/writes and launch overhead (Section 3.1).
  - `FP8` mixed precision (`8-bit floating point`) with custom quantization fusion: Fusing quantization with activations/normalization (e.g., `SiLU`) cuts I/O roughly in half for those kernels, boosting throughput (Section 3.2).
  - `State-aware recompute`: In training checkpointing/recompute, only compute/store the needed transposed or non-transposed quantized tensors, avoiding redundant work (Section 3.2).
  - Specialized linear-attention kernels: Replace multi-kernel prefill with a single high-performance Triton kernel by re-partitioning Q/K and V (Section 3.1).
  - End-to-end: Integrated in `SGLang` and `vLLM` inference engines for both prefill (process input context) and decode (generate tokens), plus `Flood` framework support for speculative decoding with tree masks (Sections 3.4, speculative decoding paragraph).

- Training and post-training (Sections 4–5):
  - `Continued pretraining`: Initialize from `Ling-base-2.0-20T` checkpoints, convert QKV projections for linear attention, then two-stage continued training (4K then 4K→32K→128K context), using a `WSM` learning-rate schedule via checkpoint merging (Section 4).
    - Result: `Ring-linear-base-2.0` recovers ≥98% of original base performance across most categories; slight deficits in reasoning and professional knowledge likely due to knowledge forgetting (Figure 9; Section 4).
  - `SFT` (Supervised Fine-Tuning): High-quality reasoning-heavy datasets plus general capabilities, 128K context; function-calling data re-synthesized for general patterns; use an earlier epoch checkpoint to avoid overfitting and leave headroom for RL (Section 5.1).
  - `RL` (Reinforcement Learning): Long-context (e.g., 64K) training with carefully filtered, sufficiently difficult data; main innovation is “training–inference alignment” to remove numeric and implementation mismatches that make standard on-policy algorithms unstable in long-output MoE models (Sections 5.2–5.2.2).

## 4. Key Insights and Innovations
- Hybrid linear–softmax attention with principled ratios (Sections 2.2.2–2.2.4; Figure 3; Table 1):
  - What’s new: A carefully tuned “M linear + 1 softmax” pattern (M=4 or 7) shown via scaling-law fits to outperform both pure softmax and less aggressive hybrid ratios, especially at higher training compute budgets (Figure 3).
  - Why it matters: Retains softmax’s retrieval/extrapolation strengths while making most layers linear-time and constant-memory, unlocking long-context inference efficiency (Figure 4). This is a design-level innovation beyond incremental tweaks.

- Linear attention that is performant in practice, not just in theory (Sections 2.2.1–2.2.3; 3.1–3.4):
  - What’s new: An implementation that sustains the theoretical O(n d^2) gains by solving real bottlenecks—KV/state representation, head-wise decay, QK-Norm+Partial-RoPE, grouped RMSNorm, fused kernels, and integration into major inference stacks.
  - Why it matters: Many linear-attention schemes underperform at <8K or in real decoding pipelines. Here, throughput crosses over around 8K and scales sharply afterward, achieving >2.5× prefill and >2× decode throughput over the softmax Ring-2.0 counterparts at long contexts (Figures 7–8). This turns linear attention’s theoretical promise into tangible end-to-end wins.

- Training–inference alignment for stable long-horizon RL in MoE models (Sections 5.2–5.2.2; Figures 10–12):
  - What’s new: A systematic, module-by-module alignment of numerical precision and implementation (KV cache precision, LM head in-register FP32 GEMM, RMSNorm epsilon and fusion behavior, RoPE equivalence, consistent attention backends, deterministic MoE routing). 
  - Why it matters: It eliminates a major hidden source of “on-policy” RL instability in long-output MoE LLMs. With alignment, the PPO objective can safely use rollout probabilities (Equation 5) instead of biased recomputed training probabilities (Equation 6), yielding higher reward and lower divergence (Figure 12). This is a fundamental and widely applicable systems insight.

- High-performance FP8 training and fused kernels (`linghe`) (Sections 3.1–3.3; Figure 6):
  - What’s new: Extensive kernel fusion including quantization fusion, “state-aware” recompute, and single-kernel linear-attention prefill; optimized routers and MoE dispatch with reduced I/O/activation memory.
  - Why it matters: Reported training throughput improvements up to +77% (mini) and +57% (flash) over strong baselines (Figure 6), and ~90% inference efficiency gains. This shifts compute and cost frontiers for long-context training.

Incremental vs. fundamental:
- Fundamental: The hybrid ratio validated by scaling laws; the training–inference alignment framework enabling unbiased on-policy RL with long outputs; the constant-state linear attention integrated into mainstream inference engines.
- Incremental but impactful: QK-Norm + partial RoPE refinements; head-wise power-law decay; grouped RMSNorm for no-comm norms; kernel fusions targeted to FP8/GEMM bottlenecks.

## 5. Experimental Analysis
- Evaluation setup (Section 6.1; Figure 1; Tables 2–3):
  - Benchmarks span three categories:
    - Mathematical reasoning: `AIME’24/’25`, `OlympiadBench`, `CNMO’24`, `LiveMathBench`, `TheoremQA`.
    - Agent & coding: `Humaneval+`, `MBPP+`, `LiveCodeBench (LCB)`, `Codeforces Elo`, `Spider`, `BFCL-Live`.
    - General reasoning: `GPQA-Diamond`, `SciBench`, `DROP`, `MuSR`, `Multi-LogiEval`.
  - Baselines:
    - For `Ring-mini-linear-2.0`: `Ring-mini-2.0`, `Qwen3-8B-Thinking`, `GPT-OSS-20B-Medium` (Table 2).
    - For `Ring-flash-linear-2.0`: `Ring-flash-2.0`, `Qwen3-32B-Thinking`, `Gemini-2.5-Flash`, `GPT-OSS-120B-Medium`, `Seed-OSS-36B-Instruct`, `Qwen3-Next-80B-A3B-Thinking` (Table 3).
  - Inference efficiency methodology (Sections 3.4; Figures 7–8):
    - `Prefill throughput`: input tokens/sec at batch size 1.
    - `Decode throughput`: output tokens/sec at batch size 64.
    - Normalized relative to baselines (`Qwen3-8B` for mini; `Qwen3-32B` for flash).

- Key quantitative results and comparisons:
  - Efficiency (Figures 7–8):
    - `Ring-mini-linear-2.0` (Figure 7): Prefill surpasses `Ring-mini-2.0` beyond 8K and reaches >2.5× at long contexts (>128K); decode surpasses beyond ~4K and reaches >2× by 64K. Both substantially outpace dense `Qwen3-8B`.
    - `Ring-flash-linear-2.0` (Figure 8): Prefill up to ~7–8× normalized over `Qwen3-32B` at 128K; decode >8× at long lengths; consistent gains over `Ring-flash-2.0` (softmax) and `Qwen3-Next-80B-A3B` (a hybrid competitor).
  - Training throughput (Figure 6):
    - `Mini`: +77% versus FP8 Megatron baseline when moving from TP=2 to TP=1 enabled by fused kernels and recompute; +21% just from fusion.
    - `Flash`: +25% from fusion; +57% total after increasing micro-batch size and adjusting pipeline.
  - Continued pretraining recovery (Figure 9):
    - Both base models recover ≥98% of performance vs. original Ling-base-2.0 across NLU, math, code, reasoning, knowledge domains; minor dips in reasoning/pro knowledge suggest manageable knowledge forgetting.
  - RL stability and learning (Figures 10–13):
    - Module-by-module fixes raise reward and stabilize training (Figure 10).
    - Using rollout probabilities (Equation 5) improves reward and reduces probability discrepancy (Figure 12).
    - Training reward and test scores (AIME’25 and sampled LCB) trend upward throughout RL (Figure 13).
  - End-task accuracy (Tables 2–3; Figure 1):
    - `Ring-mini-linear-2.0` (Table 2): Matches or modestly exceeds `Ring-mini-2.0` on many tasks, e.g., `AIME’24` 79.95 vs 79.69; `LCB` 59.53 vs 62.56 (slightly lower); `Codeforces Elo` 83.84 vs 84.80 (close); `DROP` 83.20 vs 88.55 (lower). Comparable with `Qwen3-8B-Thinking` and `GPT-OSS-20B-Medium` across the board.
    - `Ring-flash-linear-2.0` (Table 3; Figure 1):
      - Strong math/coding scores: `AIME’25` 86.51 vs `Qwen3-32B-Thinking` 75.47 and `Gemini-2.5-Flash` 72.00; near `Qwen3-Next-80B-A3B` 87.80. `LiveCodeBench` 70.37 exceeds `Qwen3-32B` 62.33 and `Gemini-2.5-Flash` 61.40; close to `Qwen3-Next` 71.97.
      - General reasoning mixed: `GPQA-Diamond` 74.49 beats `Qwen3-32B` 68.40 but trails `Gemini-2.5-Flash` 82.80 and `Qwen3-Next-80B-A3B` 77.20; `DROP` 89.66 is strong but below `Qwen3-Next` 92.05 and `Seed-OSS-36B` 91.17.
- Do the experiments support the claims?
  - Efficiency claims are strongly supported:
    - Figures 7–8 clearly show prefill/decode throughput scaling advantages at long contexts, consistent with constant-state linear attention design (Figure 4). Training speedups (Figure 6) are detailed with configuration and kernel changes.
  - Quality claims:
    - The models achieve competitive to state-of-the-art results on several reasoning tasks (Figure 1; Tables 2–3). However, “SOTA across multiple benchmarks” is true in a subset (not universal). For example, `GPQA-Diamond` is below `Gemini-2.5-Flash` and `Qwen3-Next-80B-A3B`, while math/coding are highly competitive or better.
- Ablations and robustness:
  - Architectural ablations show specific gains from `Partial-RoPE` (~0.004 LM loss) and `power-law head-wise decay` (~0.04 LM loss) (Section 2.2.3).
  - RL alignment ablations (Figure 10) isolate contributions from KV cache precision, LM head precision, RMSNorm, attention, and RoPE—each improving reward and stability.
  - Failure cases:
    - Slight deficits in reasoning and professional knowledge after continued pretraining (knowledge forgetting; Figure 9).
    - Linear attention underperforms at short contexts (<8K) for throughput crossover; hybrid design mitigates this but does not remove it (Sections 2.2.2, 3.4).

## 6. Limitations and Trade-offs
- Architectural assumptions and trade-offs:
  - Constant-state linear attention stores a `d x d` matrix per head. Although independent of sequence length, it can be memory-heavy when the number of heads is large. The conclusion explicitly notes: 
    > “the linear attention module maintains an identical attention head count for Q, K, and V, which brings heavy memory overhead” (Section 7).
  - Residual softmax layers are still computationally and memory intensive; they remain a bottleneck within the hybrid stack (Conclusion).
  - Hybrid ratio choice:
    - While scaling laws (Figure 3) guide the choice of `M`, the optimal ratio may depend on domain, model size, and compute budget. Fixed ratios might not be optimal for all tasks or latency targets.
- Data and training regime constraints:
  - Efficiency gains of linear attention are most pronounced >8K tokens; common pretraining windows are still 4–8K (Section 1). The authors work around this via continued pretraining and long-context mid-training, but the pretraining regime itself could limit linear attention’s benefits.
  - Knowledge forgetting during continued pretraining (Figure 9) suggests a careful curriculum is needed to preserve specialized knowledge.
- Systems and portability:
  - Heavy reliance on custom fused kernels (`linghe`), Triton code paths, and precise alignment steps may complicate portability, reproducibility, and maintenance across different hardware and stacks (Sections 3, 5.2.1).
  - Speculative decoding for linear attention requires specialized tree-mask-compatible kernels; currently in `Flood`, being ported to `SGLang` (Section 3.4).
- RL stability is contingent on alignment:
  - The on-policy PPO formulation (Equation 5) works well only after meticulous training–inference alignment. Without it, the system falls back to biased recomputation (Equation 6) and instability (Sections 5.2.1–5.2.2).
- Not universal SOTA:
  - While extremely competitive and often leading on select math/coding tasks, general-reasoning results are mixed; models like `Gemini-2.5-Flash` or `Qwen3-Next-80B-A3B` lead on some benchmarks (Table 3).

## 7. Implications and Future Directions
- How this work changes the field:
  - It operationalizes linear attention for real, long-context LLMs by combining architectural, algorithmic, and systems innovations:
    - Architectural: A validated hybrid pattern that scales gains with compute (Figure 3) and yields constant-state decoding (Figure 4).
    - Systems: FP8 fused kernels and end-to-end integration into major inference engines, enabling the theoretical benefits to show up in wall-clock metrics (Figures 6–8).
    - RL methodology: A principled, reproducible process for training–inference alignment that makes on-policy RL practical for long-output MoE models (Figures 10–12).
- Follow-up research enabled:
  - Dynamic hybridization: Learn or adapt the number/placement of softmax layers based on task or input properties, possibly per-layer adaptive routing between softmax and linear attention.
  - State compression for linear attention: Replace `d x d` per-head states with low-rank or shared-factor states to reduce constant memory without losing performance (addresses the conclusion’s “heavy memory overhead”).
  - Unified pretraining curricula: Better long-context curricula to avoid knowledge forgetting while leveraging linear attention’s strengths earlier (Figure 9).
  - Broader RL alignment toolkits: Standardized precision/operator conformance across training/inference stacks (FlashAttention versions, RoPE variants, RMSNorm epsilons), making the alignment process automatic.
  - Speculative decoding and tree masks: Generalize linear-attention kernels supporting complex masks and integrate with mainstream engines (`SGLang`, `vLLM`) and techniques like `lookahead` to reduce latency further (Section 3.4).
- Practical applications:
  - Long-document agents, retrieval-augmented generation with very long contexts, code generation with extended history, and any application requiring stable long chain-of-thought with reasonable latency/cost.
  - Test-time scaling: Scenarios where one can trade more tokens/compute for higher accuracy become more affordable, further pushing reasoning performance (Section 1; Figure 1).

Block-quoted, paper-grounded highlights:
- On the bottleneck of softmax attention:
  > “the computational complexity of traditional Softmax Attention grows quadratically with sequence length, while I/O overhead increases linearly with output length” (Section 1).
- On linear attention’s state and complexity:
  > “This linear formulation facilitates recurrent prediction with a commendable complexity of O(nd^2)... [and] requires only constant storage throughout the whole generation process” (Section 2.2.1; Equations 1–4).
- On hybrid ratios and scaling laws:
  > “the hybrid linear architecture consistently outperforms the pure softmax attention architecture… a large layer group size (e.g., M = 7) performs well under high FLOP budgets” (Figure 3; Section 2.2.2).
- On inference throughput scaling:
  > “Ultimately, [Ring-linear] achieves more than 2.5× the [prefill] throughput of Ring-2.0 and over 8× that of the baseline models for context lengths beyond 128K… [and] more than twice the [decode] throughput of Ring-2.0 and exceed baseline performance by over tenfold” (Figures 7–8; Section 3.4).
- On training–inference alignment and RL stability:
  > “In extreme cases, the output probability for the same token can be 0 during training and 1 during inference… [alignment across] KVCache, LM Head, RMSNorm, Attention, and RoPE” (Sections 5.2–5.2.1; Figures 10–11).
  > “After systematic training–inference alignment, using rollout probabilities… yields higher rewards… and maintains [probability differences] within a more stable range” (Figure 12; Equation 5 vs 6).
- On limits and next steps:
  > “the linear attention module maintains an identical attention head count for Q, K, and V, which brings heavy memory overhead… the remaining softmax attention modules introduce additional computational bottlenecks” (Conclusion, Section 7).

In brief: The `Ring-linear` series shows that “every attention” can matter—by strategically mixing linear and softmax attention, engineering the kernels and quantization, and aligning training with inference, it delivers a practical long-context reasoning stack with strong accuracy, significantly lower cost, and a clear roadmap for further gains.