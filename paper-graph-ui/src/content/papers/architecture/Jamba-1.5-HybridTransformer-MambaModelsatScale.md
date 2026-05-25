# Jamba-1.5: Hybrid Transformer-Mamba Models at Scale

**ArXiv:** [2408.12570](https://arxiv.org/abs/2408.12570)

## 🎯 Pitch

Jamba-1.5 pioneers a hybrid architecture that fuses Transformer attention, Mamba state-space models, and sparse Mixture-of-Experts (MoE) to deliver instruction-tuned large language models with unprecedented efficiency. By enabling 256K-token long contexts with strong throughput and dramatically reduced memory requirements—thanks to bespoke INT8 quantization and activation stabilization—Jamba-1.5 makes high-quality, long-context LLMs practical and cost-effective for real-world applications, pushing the boundaries of what open-weight models can achieve.

---

## 1. Executive Summary (2–3 sentences)
Jamba-1.5 introduces two instruction-tuned large language models that scale a hybrid architecture combining Transformer attention, Mamba state-space layers, and sparse Mixture-of-Experts (MoE). The core significance is practical: these models sustain very long contexts (effective 256K tokens) with much lower memory and strong throughput, while keeping competitive quality, enabled by a new INT8 quantization scheme for expert layers and a stabilization technique for activations.

## 2. Context and Motivation
- Problem addressed
  - Modern LLMs struggle to handle very long contexts efficiently: attention’s memory grows linearly with the number of tokens through the key/value (KV) cache, and runtime grows at least linearly with sequence length during decoding. MoE models add further memory pressure because most parameters live in expert MLPs that must be served efficiently.
  - The paper targets: (i) efficient long-context inference with low memory and high throughput, and (ii) maintaining or improving task quality relative to similarly sized open-weight models.

- Why this matters
  - Real applications increasingly require large working contexts: multi-document QA, codebases, contracts, logs, or long-form content. Serving such contexts is costly; many models cannot even run beyond 64K–128K tokens on common hardware. The authors aim to unlock 256K tokens on 8×80GB GPUs with good latency.

- Prior approaches and their gaps
  - Pure Transformers: strong quality but expensive long-context memory (large KV caches).
  - Pure state-space models (e.g., Mamba): efficient sequence handling but can underperform on general tasks compared to Transformers.
  - Hybrid SSM–Transformer models up to ~8B parameters have shown promise (cited in Section 2 with [6, 37]), but scaling to much larger sizes and making them practical at 256K tokens remained open.
  - Quantization approaches (e.g., GPTQ) often require calibration or trade accuracy/performance, and FP8 needs H100s.

- Positioning
  - Section 2 presents Jamba-1.5 as a scaled-up hybrid architecture (Transformer + Mamba + MoE) with two sizes: `Jamba-1.5-Mini` (12B active) and `Jamba-1.5-Large` (94B active, 398B total). Section 3.1 introduces `ExpertsInt8`, an INT8 method for MoE/MLP weights that integrates into vLLM kernels, aiming to combine low memory movement with BF16 compute. Section 3.2 adds an “Activation Loss” to keep activations within a safe numeric range to work with FP16 activation paths in some inference libraries.

## 3. Technical Approach
Step-by-step view of how Jamba-1.5 works and is served.

- Hybrid backbone (Section 2)
  - Components:
    - Transformer attention layers: provide global information pooling across the entire context.
    - Mamba layers: sequence state-space layers with linear-time processing that maintain a compact recurrent “state” instead of storing large token-by-token caches. This reduces memory pressure compared to attention-heavy stacks.
    - MoE MLPs: sparse experts where only a small subset of expert networks is executed per token.
  - Organization for `Jamba-1.5-Large`:
    - 9 blocks; each block has `l = 8` layers.
    - Ratio of attention to Mamba layers `a:m = 1:7` (one attention layer interleaved among seven Mamba layers). This choice came from prior ablations in Jamba and is supported by follow-up work (Section 2 referencing [24], [6], [37]).
    - MoE replaces a single MLP layer every `e = 2` layers.
    - `n = 16` experts per MoE layer, with top-`K = 2` routing per token (only two experts run for each token).
    - Hidden size 8192; 64 query heads and 8 KV heads.
  - Active vs total parameters:
    - “Total” counts all parameters (including all experts). “Active” counts only the subset used per token (e.g., top-2 experts), which determines inference cost. `Jamba-1.5-Large` has 398B total, but 94B active; `Jamba-1.5-Mini` has 52B total, 12B active (Table 1).

- Why this hybrid?
  - Mamba layers cut KV cache load because their computation does not require storing per-token attention keys/values, while periodic full attention layers recover global context aggregation. Section 2 notes a 1:7 attention-to-Mamba ratio worked best in their tests.
  - Figure 1 (Section 2) compares Mamba-1 vs Mamba-2 in isolation and in hybrid form. Although Mamba-2 beats Mamba-1 in pure SSM stacks, the hybrid “Mamba-1 + Attention” outperforms “Mamba-2 + Attention” at 350M and 1.3B scales, suggesting benefits of Mamba-2 (e.g., larger state size) are less critical when full attention is interleaved.

- Memory advantage at long context (Table 1)
  - KV cache at 256K tokens:
    - `Jamba-1.5-Large`: 9GB vs LLaMA-3.1-70B: 80GB and Mistral-Large-2: 88GB.
    - `Jamba-1.5-Mini`: 4GB vs Mixtral-8×7B and LLaMA-3.1-8B: 32GB.
  - This roughly order-of-magnitude reduction derives from fewer attention layers and fewer KV heads (64 query heads but only 8 KV heads), and heavy use of Mamba layers.

- Serving optimization: `ExpertsInt8` quantization (Section 3.1; Figure 2)
  - Observation: >85% of weights lie in MoE layers; >90% in MoE or MLPs.
  - Technique:
    - Quantize MoE/MLP weights to INT8 and store them as INT8 on disk.
    - At inference, dequantize to BF16 inside vLLM’s fused MoE kernel right before compute. This keeps compute in fast BF16 kernels but halves memory traffic from HBM to on-chip SRAM because INT8 weights are moved until the last moment.
  - Why this design?
    - Avoids long calibration (unlike many post-training quantization methods) — quantization occurs at model load in seconds.
    - Works on A100 (FP8 does not); still allows BF16 activations (important for numerical stability).
    - In Figure 2: latency matches FP8 on H100 and substantially beats GPTQ on A100 across batch sizes, while keeping quality (the paper reports no loss of quality with this method).
  - Engineering integration:
    - Implemented directly in vLLM’s `fused_moe` kernel (PR linked in Section 3.1), minimizing extra overhead and leveraging existing memory management (PagedAttention [18]).

- Stabilization during training: “Activation Loss” (Section 3.2)
  - Problem: Some experts and the last Mamba layer produced extremely large activations during pre-training (up to 4×10^6). While BF16 tolerated them during training, inference stacks that keep activations in FP16 can overflow (FP16 max ≈ 65K).
  - Solution: Add a small auxiliary loss term proportional to the mean-square of activations, with coefficient `α` (used `α = 1e−5` for Large). This gently penalizes very large activations.
  - Effect:
    - Reduced maxima to ~2–3K (Section 3.2).
    - No observed training quality degradation, even with `α` up to 1e−3 in experiments.
    - Validated by running full evaluation with FP16 activations and matching BF16 results without NaNs/overflows.

- Training data and post-training (Section 5)
  - Infrastructure: H100s; FSDP, tensor parallelism, sequence parallelism, and expert parallelism (adapts MegaBlocks).
  - Data/stages:
    - Pre-training on a mixture of web, code, books, and scientific articles (last updated March 2024), with multilingual coverage (English, Spanish, French, Portuguese, Italian, Dutch, German, Arabic, Hebrew).
    - Mid-training phase emphasizing long documents to strengthen long-range capabilities.
    - Post-training: supervised fine-tuning over high-quality conversational/skill-specific/long-context data with heavy use of synthetic data generation and filtering (Section 5.3). Pipelines cover table QA, document QA, tool use (Glaive function-calling v2) including parallel function calling, and steerability via validated constraints.

## 4. Key Insights and Innovations
- Scaling a hybrid SSM–Transformer–MoE to 94B active parameters with 256K effective context (Sections 1–2; Tables 1, 4)
  - What’s new: Prior hybrid demonstrations were limited to ≤8B scale. This work shows the design remains efficient and high-quality at 94B active parameters and 398B total, with confirmed 256K effective length on RULER (Table 4).
  - Why it matters: It combines Transformer-quality with SSM efficiency at production-relevant scales.

- `ExpertsInt8`: fast, calibration-free INT8 for MoE/MLP weights integrated into vLLM fused kernels (Section 3.1; Figure 2)
  - What’s new: On-the-fly dequantization inside the fused MoE kernel so weights move as INT8 through memory but compute in BF16. Works on A100 where FP8 is unavailable.
  - Why it matters: Matches FP8 latency on H100 and outperforms GPTQ on A100, without reported quality loss, enabling 256K contexts on 8×80GB GPUs for the Large model.

- Activation Loss to bound activation magnitudes with no quality hit (Section 3.2)
  - What’s new: A simple but effective squared-activation penalty added late in training to curb rare but extreme activations; brings FP16 activation paths into play without numerical issues.
  - Why it matters: Practical reliability during inference across diverse kernels and precisions.

- Empirical finding on hybrid choice: Mamba‑1 + Attention > Mamba‑2 + Attention (Figure 1)
  - What’s new: While Mamba‑2 beats Mamba‑1 in pure SSM stacks, in the interleaved hybrid setting Mamba‑1 + Attention trains better (lower loss) at 350M and 1.3B scales.
  - Why it matters: Guides architecture decisions for hybrids: Mamba‑2’s benefits (e.g., larger state) are less important when attention periodically aggregates globally.

## 5. Experimental Analysis
- Evaluation setup and baselines
  - Broad benchmark coverage (Section 6):
    - Academic: MMLU, MMLU‑Pro, GPQA, ARC‑C, BBH, HumanEval, GSM8K, instruction following (IFEval), function-calling (BFCL), safety (RealToxicity, TruthfulQA). Results in Table 2.
    - Chatbot: Arena‑Hard and WildBench (Table 3).
    - Long‑context: RULER (synthetic suite across 4K→256K; Table 4) and ∞BENCH on long novels (EN.MC, EN.QA; Table 5).
    - Multilingual: m‑MMLU for several languages (Table 6).
  - Serving performance: throughput/latency vs context length in Figures 3–4; INT8 quantization latency in Figure 2.
  - Hardware: Latency/throughput for Mini on 2×A100 80GB; Large on 8×A100 80GB; batch size 1; 512 output tokens (captions below Figures 3–4).

- Headline quantitative results
  - Long‑context effectiveness
    - RULER: The only models with confirmed effective length 256K. 
      - > “Jamba‑1.5‑Large: Avg 95.7; 93.9 at 256K” (Table 4)
      - > “Jamba‑1.5‑Mini: Avg 92.6; 86.1 at 256K” (Table 4)
      - Others degrade sharply beyond 32–64K (e.g., LLaMA‑3.1‑70B: effective length 64K; avg 89.6; 66.6 at 128K; Table 4).
    - ∞BENCH (100K‑token novels):
      - > EN.MC: Large 80.4 vs LLaMA‑3.1‑70B 78.2 and Mistral‑L‑2 36.9; Mini 76.9 vs LLaMA‑3.1‑8B 65.1 (Table 5).
      - > EN.QA: Mini 40.6 vs 27.1; Large 34.9 vs 36.7 (Table 5). Mixed but generally strong.

  - Academic benchmarks (Table 2; mix of own runs and reported values)
    - Large model quality is competitive with peers of similar active size:
      - > MMLU: Large 80.0 vs LLaMA‑3.1‑70B 83.6; Mistral‑L‑2 82.5.
      - > ARC‑C: Large 93.0 vs 94.8 (LLaMA‑3.1‑70B).
      - > BBH: Large 65.5 vs 69.0 (LLaMA‑3.1‑70B), 70.8 (Mistral‑L‑2).
      - > HumanEval: Large 71.3 vs 80.5 (LLaMA‑3.1‑70B) and 92 (Mistral‑L‑2).
      - > BFCL (function calling): Large 85.5 vs ~85 for LLaMA‑3.1‑70B and Mistral‑L‑2.
    - Mini model:
      - > MMLU: 69.7 (≈ LLaMA‑3.1‑8B 69.4; Gemma‑2‑9B 71.3).
      - > ARC‑C: 85.7 (best among the three in Table 2 row).
      - > BFCL: 80.7 vs LLaMA‑3.1‑8B 76.1.

  - Chatbot benchmarks (Table 3)
    - > Arena‑Hard: Large 65.4 vs LLaMA‑3.1‑70B 55.7; behind Mistral‑L‑2 (70.4).
    - > WildBench: Large 48.5 vs 49.8 (LLaMA‑3.1‑70B) and 56.3 (Mistral‑L‑2).
    - Mini outperforms LLaMA‑3.1‑8B substantially on both.

  - Throughput and latency (Figures 3–4)
    - Across context lengths (up to 256K), both Mini and Large show substantially lower latency than comparably sized baselines and retain higher throughput at long contexts. 
    - > Figure 4 shows LLaMA‑3.1‑405B cannot run beyond ~100K context on the same 8×80GB hardware, while Jamba‑1.5‑Large runs to 256K.

  - Quantization latency (Figure 2)
    - On H100: ExpertsInt8 ≈ FP8 latency for Jamba models and Mixtral baselines (Figures 2a, 2c, 2d, 2e).
    - On A100: ExpertsInt8 significantly faster than GPTQ (Figure 2b).

- Do the experiments support the claims?
  - Efficiency and long‑context claims are strongly supported by:
    - KV cache comparisons (Table 1),
    - end‑to‑end latency/throughput curves (Figures 3–4),
    - RULER effective length of 256K with high accuracy (Table 4),
    - practical feasibility (Large runs 256K on 8×80GB with ExpertsInt8).
  - Quality claims are supported but nuanced:
    - On many standard tasks, Jamba‑1.5‑Large is close to but below LLaMA‑3.1‑70B and Mistral‑L‑2 SOTA on several benchmarks; it excels in some (e.g., ARC‑C) and matches in function calling (Table 2).
    - Chatbot results show Large beats LLaMA‑3.1‑70B on Arena‑Hard but trails Mistral‑L‑2 (Table 3).
  - Ablations and robustness:
    - Architecture ablation: Figure 1 convincingly shows hybrid Mamba‑1 + Attention outperforming Mamba‑2 + Attention at smaller scales.
    - Training stabilization: Section 3.2’s FP16 validation suggests Activation Loss solves overflow without hurting metrics, though detailed pre/post accuracy tables are not shown.
    - Quantization quality: Figure 2 focuses on latency; the “no loss of quality” assertion is stated in Section 3.1 but not backed by a dedicated accuracy delta table. However, the end‑to‑end evaluation of the quantized models at 256K context indirectly supports acceptable quality.

- Caveats and mixed results
  - GSM8K reporting for LLaMA‑3.1 models needed a “flexible” evaluation to match known numbers (Table 2 note), and Mistral‑L‑2 struggled with ARC‑C on the authors’ runs. These show some evaluation irregularities common in LLM benchmarking.

## 6. Limitations and Trade-offs
- Benchmark scope vs real tasks
  - RULER’s synthetic tasks measure retrievability and tracking under long contexts; they are not full proxies for complex long‑document reasoning. ∞BENCH helps, but evaluated only two English tasks (Table 5).
- Quality vs efficiency
  - While Jamba‑1.5 often matches or is close to peers, it does not consistently win on the most stringent academic benchmarks (Table 2). The win is primarily efficiency and long‑context capacity.
- Quantization evidence
  - ExpertsInt8 quality impact is asserted but not tabulated with accuracy deltas pre/post quantization; the paper shows latency benefits (Figure 2) but not detailed accuracy parity under quantization.
- Hardware and batching
  - Throughput/latency curves use batch size 1 (Figures 3–4). Multi‑tenant serving dynamics (e.g., high batch sizes, variable sequence mixes) are not extensively profiled.
- Training data transparency
  - The data mixture is described at a high level (Section 5.1), with reliance on synthetic data and internal filters (Section 5.3). Full datasets are not released, limiting exact reproducibility of training.
- Architectural choices
  - The 1:7 attention:Mamba ratio works well here (Section 2) but may be sensitive to domain/data; there is not a broad sweep across ratios at large scale.
- MoE specifics
  - Load balancing, expert utilization statistics, and routing stability are not deeply analyzed; though they adopt MegaBlocks for expert parallelism (Section 5.1), detailed behavior under skewed routing is not shown.

## 7. Implications and Future Directions
- Field impact
  - Demonstrates that hybrid SSM–Transformer–MoE models can deliver 256K practical contexts with low KV memory and strong throughput on widely available 8×80GB GPU nodes. This shifts the feasibility frontier for long-context applications from research demos to deployable systems.

- Practical applications
  - Long-document QA and analysis (contracts, legislation, technical manuals).
  - Codebase understanding (large repositories), log analysis, and multi‑file reasoning.
  - Enterprise assistants needing tool use (supported by strong BFCL scores) and long‑memory sessions.
  - Scalable serving on A100‑class hardware using ExpertsInt8, not only H100.

- Research directions
  - Hybrid design exploration:
    - Optimize attention:Mamba ratios per depth and per domain; study dynamic scheduling of attention layers.
    - Revisit Mamba‑2 in hybrid settings at larger scales or with altered state sizes to probe the Figure 1 finding.
  - Quantization advances:
    - Extend ExpertsInt8 to attention projections and non‑MoE layers where beneficial; publish accuracy parity studies across many tasks.
    - Combine with activation-aware quantization to further cut memory without precision loss.
  - Training for long context:
    - Systematic recipes for mid‑training/post‑training data that retain long‑range skills without hurting general quality.
    - More naturalistic long‑context benchmarks beyond RULER and ∞BENCH (e.g., multi‑document reasoning with citations).
  - Safety and alignment:
    - The paper outlines OECD‑aligned behavioral tenets (Section 7); making the 60‑tenet set public and measurable can foster comparable safety evaluations.
  - Efficient multi‑tenant serving:
    - Characterize throughput/latency under mixed sequence length and large batch sizes; study scheduling with MoE load balancing and paged KV caches.

> Bottom line: Sections 2–4 and Tables 1, 4, 5, alongside Figures 2–4, make a compelling case that the Jamba hybrid plus ExpertsInt8 achieves a rare combination—very long effective context (256K), low memory, and solid quality—on affordable hardware. The main innovations are engineering and systems‑oriented, with credible measurements, and they open a practical path to long‑context LLM deployments.
