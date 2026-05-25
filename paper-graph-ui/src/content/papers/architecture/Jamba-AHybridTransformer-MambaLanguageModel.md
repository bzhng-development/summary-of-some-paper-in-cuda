# Jamba: A Hybrid Transformer-Mamba Language Model

**ArXiv:** [2403.19887](https://arxiv.org/abs/2403.19887)

## 🎯 Pitch

Jamba introduces a novel hybrid language model architecture that interleaves Transformer attention layers with efficient Mamba state space layers and incorporates Mixture-of-Experts MLPs, delivering both high-quality performance and unprecedented efficiency. This approach enables state-of-the-art long-context inference—supporting up to 256K tokens in production—with an 8x smaller memory footprint compared to traditional Transformers, unlocking practical applications for large context windows on a single GPU and setting a new standard for scalable, high-throughput language modeling.

---

## 1. Executive Summary (2-3 sentences)
This paper introduces Jamba, a large language model that interleaves Transformer layers with Mamba (state space) layers and inserts Mixture-of-Experts (MoE) MLPs to achieve high quality, high throughput, and drastically lower memory usage. The key outcome is long-context inference at production scale—up to 256K tokens—with an 8x smaller attention key-value cache than vanilla Transformers, while matching or approaching the performance of strong baselines like Mixtral-8x7B and Llama‑2‑70B.

## 2. Context and Motivation
- Problem addressed:
  - Standard Transformer decoders are memory- and compute-heavy for long contexts because they must store a “key-value” (`KV`) cache for each attention layer and recompute attention over the entire prefix at every generated token (Section 1).
  - Recurrent-style architectures summarize history in a small state but historically train slowly and struggle with long-range dependencies.
- Why it matters:
  - Real applications (analytics over long documents, retrieval-augmented generation, log/code analysis) increasingly need context windows in the 100K–1M range. The Transformer KV cache becomes the limiting factor for fitting such contexts on commodity GPUs (Sections 1–2).
- Prior approaches and their gaps:
  - Pure Transformers: excellent quality but poor long-context efficiency (KV cache grows with number of attention layers and sequence length) (Section 1).
  - Pure state space models (SSMs) like Mamba: linear-time sequence modeling and small memory footprint, but lag behind same-sized Transformers on standard LLM benchmarks (Section 1).
  - Earlier hybrids (e.g., S4 with local attention, Hyena/StripedHyena, H3) explored mixing attention and SSMs but either at smaller scale, different layer mixing patterns, or with weaker performance than strong Transformer baselines (Section 1).
- How this work positions itself:
  - Jamba is a large-scale, production-grade hybrid that:
    - Interleaves attention and Mamba layers at a tuned ratio.
    - Adds MoE to selected MLPs to boost total capacity without increasing active compute.
    - Demonstrates state-of-the-art long-context performance and throughput while fitting on a single 80GB GPU (Sections 1–3).

## 3. Technical Approach
Step-by-step overview of the Jamba architecture and implementation.

- Core architectural idea: the “Jamba block”
  - A `Jamba block` contains `l` layers formed by mixing attention and Mamba layers at a ratio `a:m` (Section 2; Figure 1).
  - Each layer is either:
    - A Transformer-style self-attention layer followed by an MLP, or
    - A Mamba layer followed by an MLP (Figure 1b).
  - Some MLPs are replaced by MoE layers to increase capacity while keeping compute modest (Section 2).

- Key components explained
  - Mamba layer (state space model; SSM):
    - Processes sequences with a recurrent-like state update that is linear in sequence length and does not require storing a KV cache. This makes it memory- and compute-efficient for long contexts (Sections 1–2).
    - In Jamba, Mamba layers receive RMSNorm normalization internally for stability at scale (Section 6.4; Figure 9).
    - No explicit positional encoding is used; the Mamba layers provide implicit position information, making RoPE optional (Section 6.5; Table 8).
  - Attention layer:
    - Provides explicit content-based retrieval over the entire context (standard in Transformers).
    - Uses Grouped-Query Attention (`GQA`) to reduce KV size (Section 2).
  - Mixture of Experts (`MoE`):
    - Replaces some MLPs with an expert router that chooses `K` experts among `n` per token. Only the chosen experts run, so “active parameters” per token remain low while total capacity grows (“available parameters”) (Section 2).
    - Jamba uses load balancing to keep expert usage even (Section 2).
  - Terminology:
    - `Available parameters`: total parameters across all experts.
    - `Active parameters`: the subset of parameters actually used per token (e.g., top‑2 experts out of 16).
    - `KV cache`: tensors of keys and values stored for future attention; scales with the number of attention layers, attention heads, and sequence length.

- Design space (Section 2)
  - `l`: layers per Jamba block.
  - `a:m`: ratio of attention to Mamba layers.
  - `e`: how often MoE replaces the MLP (e.g., every 2 layers).
  - `n`: number of experts per MoE layer.
  - `K`: number of experts activated per token.

- Implemented configuration (fits on one 80GB GPU while keeping quality high)
  - 4 Jamba blocks, each with:
    - `l = 8` layers.
    - `a:m = 1:7` (one attention, seven Mamba per block).
    - `e = 2` (MoE every other layer).
    - `n = 16` experts; `K = 2` active experts per token (Section 3.1; Figure 1).
  - Result: 32 total layers with only 4 attention layers and 28 Mamba layers. Active params ~12B, total available params ~52B (Sections 1, 3.1).
  - Supporting details: 64K BPE tokenizer (digits as separate tokens), SwiGLU activations, GQA, MoE load balancing; no explicit positional encodings (Section 2).

- Why these choices?
  - Ratio `a:m = 1:7`:
    - Ablations show hybrids outperform pure Mamba and pure attention; and 1:3 vs 1:7 deliver similar quality, so 1:7 is chosen for higher efficiency (Section 6.1; Table 4; Figure 6).
  - MoE every other layer with 16 experts, top‑2 routing:
    - Balances memory, compute, and inter-GPU communication during expert-parallel training/inference (Section 3.1).
    - Empirically improves the hybrid’s quality (Section 6.3; Table 7).
  - No positional encoding:
    - Similar quality with and without RoPE; Mamba layers provide implicit positional information (Section 6.5; Table 8).
  - Stabilization:
    - Adding RMSNorm inside Mamba layers eliminates loss spikes at 7B-scale training (Section 6.4; Figure 9).

- System implementation and training setup
  - Training on NVIDIA H100 with in-house framework supporting FSDP, tensor parallelism, sequence parallelism, and expert parallelism (Section 4).
  - Data: in-house mixture of web, books, and code (curated and deduplicated; last update March 2024) (Section 4).
  - Inference experiments often on A100 80GB (throughput and memory fit analyses) (Sections 3.1–3.2).

- How the hybrid reduces memory and improves throughput
  - KV cache scales with the number of attention layers; Jamba uses only 4 attention layers out of 32 total, cutting the KV cache roughly 8x vs a full-Transformer of similar depth (Section 2; Table 1).
  - For long sequences, attention FLOPs dominate; replacing most of them with Mamba layers improves compute efficiency, especially as context grows (Section 2; Figure 3b).

## 4. Key Insights and Innovations
- Hybrid attention–Mamba design at production scale
  - Novelty: Interleaving Mamba and attention at a high Mamba ratio (1:7) with MoE at scale, delivering performance comparable to strong Transformers while drastically lowering KV memory (Sections 1–3).
  - Significance: Enables 256K context with a 4GB KV cache vs 32GB for Mixtral and 128GB for Llama‑2‑7B at the same context length (Section 2; Table 1).
- Demonstrated long-context efficiency without sacrificing quality
  - Evidence: On 4×A100 GPUs, Jamba’s throughput at 128K context is ~3× Mixtral’s; Llama‑2‑70B does not fit this window (Section 3.2; Figure 3b). Yet benchmark quality remains comparable to Mixtral and Llama‑2‑70B on many tasks (Section 5.1; Table 2).
- MoE integrated into a hybrid SSM–attention model
  - Novelty: MoE MLPs inside a hybrid architecture, every other layer, with 16 experts and top‑2 routing (Sections 2–3.1).
  - Impact: Clear quality gains at 7B scale vs the same hybrid without MoE (Section 6.3; Table 7).
- Insight into in-context learning (ICL) behavior of SSMs
  - Finding: Pure Mamba models perform poorly on tasks requiring strict format adherence and ICL-like behavior (e.g., IMDB, QuAC, NarrativeQA), but the hybrid recovers performance comparable to attention-only models (Section 6.2; Table 6).
  - Mechanistic hint: Visualization shows attention heads in the hybrid locking onto label tokens from the shots, resembling “induction heads” (Section 6.2; Figure 8).
- Removing explicit positional encodings
  - Discovery: Comparable results with and without RoPE when Mamba precedes attention, suggesting Mamba encodes position implicitly (Section 6.5; Table 8).

## 5. Experimental Analysis
- Evaluation methodology
  - Benchmarks (Section 5.1):
    - Reasoning and QA: HellaSwag (10‑shot), WinoGrande (5‑shot), ARC-E (0‑shot), ARC‑C (25‑shot), PIQA (0‑shot), Natural Questions (5‑shot), BoolQ (10‑shot), QuAC (0‑shot), GSM8K (3‑shot CoT), HumanEval (pass@1), MMLU (5‑shot), BBH (3‑shot).
  - Long-context tests (Section 5.2):
    - Synthetic retrieval: Needle-in-a-Haystack up to 256K tokens (Figure 4).
    - Few-shot classification with many shots (to lengthen context): TREC‑Fine, NLU Intent, Banking77, CLINC150 up to ~128K tokens (Figure 5).
    - Long-context QA from L‑Eval formatted as 3‑shot: NarrativeQA, LongFQA, NQ, CUAD, SFiction (Table 3).
  - Throughput and memory:
    - 80GB single-GPU fit and KV cache comparison (Figure 2; Table 1).
    - End-to-end throughput (encoding+decoding) by batch size and context length (Figure 3).

- Main quantitative results
  - Memory footprint and fit:
    - KV cache at 256K context: 
      > Table 1: “Jamba 4GB vs Mixtral 32GB; Mistral 32GB; Llama‑2 7B 128GB.”
    - Single A100 80GB fit:
      > Figure 2: Jamba supports ~2× longer contexts than Mixtral and ~7× longer than Llama‑2‑70B on one 80GB GPU.
  - Throughput:
    - Batch scaling at 8K context:
      > Figure 3a: Jamba achieves ~3× the throughput of Mixtral and supports larger batch sizes (e.g., Mixtral cannot fit batch 16).
    - Scaling with context length:
      > Figure 3b: At 128K context, Jamba’s throughput is ~3× Mixtral’s; Llama‑2‑70B does not fit.
  - Standard academic benchmarks:
    - Overall, Jamba is competitive with Mixtral‑8x7B and Llama‑2‑70B:
      > Table 2 (selected): HellaSwag 87.1 (Jamba) vs 86.7 (Mixtral) and 85.3 (Llama‑2‑70B); WinoGrande 82.5 (Jamba) vs 81.2 (Mixtral); GSM8K 59.9 (Jamba) vs 60.4 (Mixtral); MMLU 67.4 (Jamba) vs 70.6 (Mixtral) and 69.8 (Llama‑2‑70B).
  - Long-context evaluations:
    - Needle-in-a-Haystack:
      > Figure 4: Near-perfect retrieval across depths up to 256K despite having only 4 attention layers.
    - Many-shot classification:
      > Figure 5: Jamba outperforms Mixtral on TREC‑Fine and Banking77 as shots increase; parity on NLU Intent and CLINC150.
    - Long-context QA:
      > Table 3: Average F1 0.44 (Jamba) vs 0.43 (Mixtral); wins on LongFQA and NQ.
  - Ablations and diagnostics:
    - Hybrid vs pure models (1.3B, 250B tokens):
      > Table 4: Hybrid outperforms pure attention and pure Mamba across HellaSwag, WinoGrande, OLLM, log-prob.
    - Hybrid vs pure models (7B, 50B tokens):
      > Table 5: Hybrid exceeds pure attention and pure Mamba on OLLM (15.4 vs 13.7 and 14.0) and on several log-prob datasets.
    - SSM ICL limitations:
      > Table 6: IMDB 48.8 (Mamba) vs 84.1 (Attention) and 90.9 (Hybrid); similar gaps on QuAC and NarrativeQA, traced to formatting adherence failures (Section 6.2).
    - Effect of MoE:
      > Table 7: OLLM 18.9 (Hybrid+MoE) vs 15.4 (Hybrid no-MoE); consistent improvements on HellaSwag, WinoGrande, NQ, and log-prob.
    - Stability with RMSNorm:
      > Figure 9: RMSNorm inside Mamba eliminates loss spikes at large scale.
    - Positional information:
      > Table 8: Comparable results with and without RoPE.

- Do the experiments support the claims?
  - Yes, for the stated scope:
    - Memory/throughput advantages are clearly quantified (Table 1; Figures 2–3).
    - Quality parity with strong baselines is demonstrated on a standard suite (Table 2) and long-context tasks (Table 3).
    - Ablations justify core design choices (Tables 4–8; Figures 6–7,9) and provide a plausible mechanism for why hybrids outperform pure SSMs on ICL-heavy tasks (Section 6.2; Figure 8).
  - Caveats:
    - The released model is a base (not instruction-tuned or safety-aligned), so comparisons on instruction-following tasks are out of scope (note under Figure 1; “Important notice”).
    - Throughput numbers are “end-to-end” and not maximally optimized; the paper notes further optimizations for hybrid models could increase the gap (Section 3.2).

## 6. Limitations and Trade-offs
- Assumptions and design constraints
  - Only 4 attention layers in the released configuration; tasks that rely heavily on global pattern matching via attention might benefit from more attention layers (implied by the a:m trade-off discussion in Section 2 and Figure 1).
  - The memory and throughput advantages depend on using fewer attention layers and on GQA; changing these knobs changes trade-offs (Section 2).
- Scenarios not fully addressed
  - Instruction-following, safety alignment, and moderation are not provided; the base model should not be deployed with end users without additional tuning (notice under Figure 1).
  - Cross-lingual, multimodal, or tool-use capabilities are not covered.
- Computational and data constraints
  - Although the model fits 256K on one 80GB GPU with 8‑bit weights (Section 3.1; Figure 2), some experiments (e.g., high-throughput or multi-GPU) still require substantial hardware.
  - The training corpus is proprietary; replicability of data distribution is limited (Section 4).
- Open technical questions
  - Nature of ICL in SSMs: the paper presents evidence that pure Mamba struggles with ICL-like formatting, and that attention heads in the hybrid show induction-like behavior (Section 6.2); a full theory is still open.
  - Optimal hybrid ratios and placement strategies beyond `a:m = 1:7` for different compute/memory budgets (Section 6.1 explores only a few points).

## 7. Implications and Future Directions
- Field-level impact
  - Jamba demonstrates that hybrid attention–SSM architectures can retain Transformer-level quality while achieving drastic gains in long-context efficiency. This challenges the “attention-only” default for large context windows and motivates new training/inference systems tailored to hybrids (Sections 1–3; 5).
- Follow-up research enabled
  - System optimizations for hybrids: specialized kernels, KV caching strategies for few attention layers, pipelining with Mamba states (Section 3.2 hints more gains are possible).
  - Mechanistic interpretability of hybrids: mapping how attention heads and Mamba states collaborate to produce ICL (Section 6.2; Figure 8).
  - Architecture search over `a:m`, `e`, `n`, `K` to meet diverse latency/memory budgets, including edge deployment.
  - Further study of positional information in hybrids and when explicit encodings help or harm (Section 6.5).
  - Training stability recipes for large SSM components (RMSNorm variants, scaling laws; Section 6.4).
- Practical applications
  - Long-document assistants (legal, financial, scientific) and code analysis tools that need 100K–256K context.
  - Cost-effective deployment on fewer or smaller GPUs due to the smaller KV cache and higher throughput at long contexts (Table 1; Figures 2–3).
  - Open checkpoints (Apache 2.0; model link in Abstract) enable community fine-tuning for instruction following, safety, and domain specialization.

Overall, Jamba provides a concrete, reproducible path to long-context LLMs that preserve quality while dramatically improving memory and throughput, supported by design-motivated ablations and large-scale evaluations across standard and long-context benchmarks.
