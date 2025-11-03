# InfiniteHiP: Extending Language Model Context Up to 3 Million Tokens on a Single GPU

**ArXiv:** [2502.08910](https://arxiv.org/abs/2502.08910)

## 🎯 Pitch

InfiniteHiP introduces a novel, training-free framework that enables large language models to handle up to 3 million tokens on a single commodity GPU—achieving this by uniting efficient hierarchical block-sparse attention, dynamic position encoding for out-of-length generalization, and smart KV-cache offloading that avoids forgetting any context. This combination delivers unprecedented scalability and speed, empowering production LLM deployments for massive-context applications like retrieval-augmented generation and long-form analysis, without expensive retraining or hardware upgrades.

---

## 1. Executive Summary
InfiniteHiP is a training‑free inference framework that lets existing large language models process million‑token prompts on a single GPU by combining three ideas: a modular hierarchical pruning scheme that builds a block‑sparse attention mask, a dynamic positional encoding strategy for out‑of‑length (OOL) generalization, and a KV‑cache offloading system that spills rarely used tokens to host memory. As a result, it processes up to 3M tokens on a single 48GB L40s without permanently discarding context and achieves large speedups over dense attention and prior sparse/offloading systems (Abstract; Figures 1, 5; Tables 3–4).

## 2. Context and Motivation
- The problem
  - Long prompts make attention slow (quadratic in sequence length) and inflate the key‑value (`KV`) cache linearly with length, stressing GPU memory during generation (Introduction).
  - Most pretrained LLMs do not generalize beyond their training context window because their positional encoding (e.g., RoPE) does not extrapolate well (Introduction).
- Why this matters
  - Practical applications (RAG systems, multi‑document QA, long‑form analysis) require hundreds of thousands to millions of tokens while keeping prior content accessible (Introduction; Section 5).
  - Production deployments need both speed and faithful retrieval of any part of the context (no “forgetting” due to eviction).
- Prior approaches and gaps
  - FlashAttention2 reduces memory traffic but not compute; it still computes dense attention (Related Works; Table 3).
  - Token‑selection methods speed prefill but not decoding (e.g., MInference), or rely on fixed representatives that may miss query‑specific tokens (InfLLM) (Related Works).
  - KV eviction (e.g., H2O) saves memory but permanently forgets information that may be needed later (Related Works).
  - HiP Attention offloads KV to CPU and fetches on demand, but its pruning uses heuristics with global synchronizations that limit GPU parallelism and speed (Related Works; Section 1).
  - RoPE extrapolation methods (Dynamic‑NTK, Self‑Extend) extend position indexing but do not address compute/memory cost (Section 5.1).
- Positioning
  - InfiniteHiP integrates three components—modular pruning, dynamic RoPE, and efficient offloading—into a single inference pipeline that targets both speed and OOL generalization without training (Section 4; Figure 1).

## 3. Technical Approach
The system consists of three interlocking parts (Figures 1–2; Section 4; Appendix A–D).

1) Modular hierarchical context pruning (how the sparse mask is built)
- Key observation (chunk sparsity)
  - In real attention maps, “top‑k” keys concentrate in a small number of contiguous chunks: in a 128K context using Llama‑3.1‑8B, fewer than 2% of chunks contain more than 12.5% of the top‑2048 keys, and about 75% of 64‑token chunks contain none of the top‑2048 keys (Figure 2a).
- Goal
  - For each query block, approximate the top‑K highest‑score key blocks and compute attention only there (block‑sparse attention).
- Pruning stages (S(i))
  - Each stage S(i) is parameterized by `(bq, lc, k)` (query block size, key chunk size, tokens to keep); see Section 4 (“In formal notation…”) and Appendix A.1.
  - Pipeline per stage (Figure 2b; Algorithm 2):
    1) Partition candidate key indices from the previous stage into equal‑size chunks of length `lc`.
    2) For each chunk, pick a representative token via hierarchical top‑1 selection (Algorithm 3): recursively split the chunk in half, compare one token from each half against the current query block (maximum over queries and heads), and keep the better half; this takes `O(log lc)` steps and reads at most two keys per split. This avoids scanning all keys and is highly parallelizable on the GPU.
    3) Compute an estimated chunk score by taking the maximum attention score between the query block and that chunk’s representative token, maximized over heads and queries in the block.
    4) Keep only the top `K = k / lc` chunks and pass their indices to the next stage (Equations (1)–(3)).
- Design details that make it practical
  - Sink and streaming tokens: always retain the first `nsink` prompt tokens and the most recent `nstream` tokens to preserve global anchors and recency (Section 4).
  - Stage caching: during decoding, the sparse mask is not recomputed every step; each stage refreshes every `n_refresh^(i)` steps to exploit temporal locality (Section 4 “Sparse Attention Mask Caching”; Algorithm 4; Table 4 shows hit ratios rise steeply as more stages are cached).
  - Default hyperparameters (Appendix F) use three stages with progressively smaller chunks `(256, 32, 8)` and `k` shrinking `(32K, 8K, 2–4K)`, query blocks of size 64, sink=256, stream=1024.

2) Dynamic RoPE for out‑of‑length generalization (how positions are assigned when context > training window)
- Terms
  - `RoPE` (rotary positional embedding) rotates queries/keys with sinusoidal matrices parameterized by position id; it encodes order without positional vectors.
  - “OOL generalization” means handling positions longer than those seen during pretraining without fine‑tuning.
- Why RoPE needs care here
  - When contexts exceed the training length, naive RoPE indexing harms attention patterns. Also, early layers of many LLMs exhibit sliding‑window‑like locality that block‑sparse attention must approximate (Appendix D; Figure 10).
- Strategy (Section 4 “Dynamic RoPE…”; Appendix A.1, B–D)
  - During pruning (mask selection), mix two styles:
    - Chunk‑indexed RoPE for the first 3 layers: assign one position id per key chunk (offset by `nstream` from the current query); this injects strong locality and helps approximate sliding windows in early layers.
    - Relative‑style RoPE for later layers: assign two nearby position ids to the left/right halves during top‑1 selection, which emphasizes content while preserving relative proximity.
  - During the final sparse attention compute, use StreamingLLM‑style RoPE: assign sequential position ids to the selected keys (including sink and stream), with the most recent key sharing the query’s position (Section 4; Xiao et al., 2024b).
- Evidence the mix matters
  - Ablation with ∞Bench En.MC at 300K context shows using chunk‑indexed in layers 1–3 and relative in 4–32 yields 74.23% vs 68.55% for relative‑only (Appendix D, Table 6).
  - A cross‑combination ablation (Table 5) indicates Relative (pruning) + Streaming (sparse attention) is among the strongest pairings.

3) KV‑cache offloading with unified memory and LRU (how memory fits on a single GPU)
- Terms
  - `KV cache` stores past keys and values so the model does not recompute them during decoding; it grows linearly with context length.
  - `UVM` (Unified Virtual Memory) lets GPU kernels access CPU memory pages that are transparently fetched over PCIe on demand.
  - `LRU` eviction discards the least recently used entries when the GPU cache is full.
- Mechanism (Section 4 “KV Cache Offloading”; Algorithm 4)
  - Maintain two GPU‑resident key banks: one for mask‑selection kernels and one for the final block‑sparse attention (separate access patterns).
  - Keep a GPU page table mapping global token indices to their current GPU‑bank slot.
  - On a GPU cache miss, fetch the missing KV pages from host memory via UVM and place them into the GPU bank; evict cold pages using LRU.
  - Stage‑wise mask caching reduces how often Stage‑1 (the only part linear in context length) runs and therefore how often CPU memory must be touched (Table 4, “Cached Stages”).
- Paged block‑sparse attention kernel
  - A FlashAttention‑like kernel adapted to block‑sparse masks and paged KV (Section 4 “Implementation”), with FlashDecode for decoding and PagedAttention‑style memory management.

Complexity and parallelism
- Stage‑1 pruning is `O(Tq·Tkv)` but reads at most two keys per chunk during top‑1 selection and is implemented as a single Triton kernel with no global synchronization (Appendix A.1). Later stages are `O(Tq)`. In practice, heavy GPU parallelism and caching make it faster than earlier hierarchical methods (Table 3).

## 4. Key Insights and Innovations
- Modular, GPU‑friendly pruning that exploits chunk sparsity
  - Innovation: a stack of pruning stages that each (a) select a representative key per chunk using a binary‑search‑like top‑1 estimator and (b) pass only the best chunks onward (Figure 2b; Algorithms 2–3).
  - Why it matters: enables high‑recall, query‑adaptive masks with strong GPU parallelism. The selected‑key recall is higher than HiP and InfLLM by 4.72 and 1.57 percentage points, respectively (Figure 6a).
- A layer‑wise mix of RoPE indexers geared to attention locality
  - Innovation: combine chunk‑indexed RoPE in early layers with relative/streaming styles later (Section 4; Appendices B–D).
  - Why it matters: preserves sliding‑window‑like behavior in shallow layers (Figure 9, 10) and yields stronger OOL generalization; En.MC rises from 68.55% (relative‑only) to 74.23% with the mixed scheme (Appendix D, Table 6).
- Stage‑wise mask caching plus LRU‑backed UVM offloading
  - Innovation: cache masks per pruning stage and maintain a separate GPU KV bank for masking vs attention; fetch misses from CPU memory and evict via LRU (Algorithm 4; Table 4).
  - Why it matters: drastically cuts decoding latency under offloading. At 256K, “Runtime (Flash)” falls to 325 µs vs InfLLM’s 1,186 µs (≈3.65× faster), and mask hit ratios exceed 98% when early stages are cached (Table 4).
- Practical single‑GPU million‑token inference
  - Innovation: an end‑to‑end SGLang integration with paged block‑sparse kernels and UVM offloading (Section 4 “Implementation”; Figure 5).
  - Why it matters: delivers usable throughput on commodity or cloud GPUs. On a 24GB RTX 4090 at 1M tokens, throughput is ≈3.20× higher than the SGLang Runtime with FlashInfer (SRT) estimate (Figure 5, left). On a 48GB L40s at 3M tokens, throughput is ≈7.25× higher than SRT estimate (Figure 5, right).

## 5. Experimental Analysis
- Evaluation setup (Section 5)
  - Benchmarks
    - LongBench (average length ≈32K) spanning QA, summarization, few‑shot, code (Table 1).
    - ∞Bench (beyond 100K tokens) covering synthetic retrieval tasks and NLU (Table 2).
    - Additional: RULER (Appendix E.2), passkey retrieval on DeepSeek‑R1‑distilled Qwen2.5‑14B (Appendix E.1), scaling curves (Figures 3–4).
  - Models: Llama‑3/3.1‑8B Instruct and Mistral 0.2‑7B Instruct; additional tests on Gemma2‑9B and EXAONE‑7.8B (Figures 3–4; Table 10).
  - Baselines (Section 5.1): FA2 (dense, truncation), Dynamic‑NTK, Self‑Extend (RoPE tricks), LM‑Infinite and StreamingLLM (sink+stream windows), H2O (eviction), InfLLM (chunk representatives), and HiP Attention.
  - System setup: Triton kernels, SGLang integration; decoding latency and throughput measured on RTX4090 and L40s (Tables 3–4; Figure 5; Appendix E.4).
- Main quantitative results
  - LongBench (Table 1)
    - Llama‑3‑8B: Absolute average 47.72 for InfiniteHiP vs 44.47 (InfLLM) and 42.47 (FA2). Relative score 100.00 vs 92.83 and 87.69.
    - Mistral‑7B: Absolute average 42.71 for InfiniteHiP vs 41.46 (best baseline). Relative score 99.85 vs 96.99 (InfLLM‑12K).
  - ∞Bench (Table 2)
    - Llama‑3‑8B: Relative score 98.17 (3K window) vs 89.07 (InfLLM). With different refresh schedules (3K‑fast/flash), relative scores remain ≈98.
    - Mistral‑7B: Relative score up to 99.09 (5K) vs 94.77 (InfLLM‑16K).
  - OOL scaling (Figures 3–4)
    - En.MC rises with context for InfiniteHiP even beyond the pretrained window, while dense baselines degrade after 128K.
    - On short‑context models (Gemma2 8K, EXAONE 3/3.5), InfiniteHiP yields large gains at extended lengths (Figure 4; Table 10).
  - Latency without offloading (Table 3; 1M‑token context)
    - Decoding attention latency: 234 µs (InfiniteHiP 3K) vs 4,645 µs (FA2) and 1,222 µs (InfLLM‑12K), i.e., ≈19.85× and ≈4.98× faster, respectively. Prefill is also faster than FA2 (20.29×) and slightly faster than InfLLM.
    - Enabling dynamic RoPE (“Extend”) adds ≈1.6× overhead to prefill and ≈5% to decoding (Table 3, “Ours with Extend”).
  - Latency with KV offloading (Table 4)
    - At 256K, “Runtime (Flash)” is 325 µs (InfiniteHiP 3K‑fast) vs 1,186 µs (InfLLM‑12K), ≈3.65× faster; similar factors at 512K and 1,024K.
    - Stage caching reduces decoding from 9,803 µs (no caches) to 110 µs (all stages cached) at 256K; mask hit ratio jumps from 71.7% to 98.8% (Table 4).
  - Throughput (Figure 5; Appendix E.4 Tables 11–12)
    - RTX4090 @1M: Estimated 40 tokens/s (offload‑flash) vs 12.5 (SRT estimate), ≈3.2× higher.
    - L40s @3M: Estimated 23.8 tokens/s (offload‑flash) vs 3.3 (SRT estimate), ≈7.25× higher.
- Ablations and diagnostics
  - Mask quality: selected‑key recall beats HiP by 4.72 points and InfLLM by 1.57 points (Figure 6a).
  - Depth of pruning: more stages (N=3) improve En.MC over N=2 (74.24 vs 70.31), both surpass dense FA2 at 128K (67.25) (Figure 6b).
  - RoPE combinations: Relative (pruning) + Streaming (sparse attention) among the strongest; chunk‑indexed everywhere hurts, but using it only in early layers helps (Table 5; Appendix D, Table 6).
- Do results support the claims?
  - Speedups: Yes; multiple tables show order‑of‑magnitude decoding gains vs FA2 and multi‑fold gains vs InfLLM, both with and without offloading (Tables 3–4).
  - Accuracy: On LongBench and ∞Bench, InfiniteHiP matches or exceeds best baselines while pruning more aggressively (Table 1–2). OOL generalization is supported by scaling curves and by strong performance on short‑context models when extended (Figures 3–4; Table 10).
  - Practicality: SGLang throughput plots demonstrate single‑GPU million‑token decoding with realistic tokens/s (Figure 5).

> “InfiniteHiP enables the processing of up to 3 million tokens on a single L40s 48GB GPU … without any permanent loss of context information.” (Abstract; Figure 1; Section 4 “KV Cache Offloading”)

> “Our framework achieves an 18.95× speedup in attention decoding for a 1 million token context without requiring additional training.” (Abstract; corroborated by Table 3, which shows ≈19.85× vs FA2)

> “We implement InfiniteHiP on the SGLang serving framework, achieving a 7.24× speedup in end‑to‑end decoding on a 3M token context while using only 3.34% of the VRAM required by FA2.” (Section 1, contributions)

## 6. Limitations and Trade-offs
- Assumptions about attention structure
  - The pruning relies on “chunk sparsity” and “attention locality.” While empirically supported (Figure 2a; Figures 9–10), pathological inputs with uniformly distributed long‑range attention could reduce pruning efficacy.
- Stage‑1 cost and prefill latency
  - Stage‑1 is `O(Tq·Tkv)` and dominates when masks must be computed; caching amortizes it in decoding but not in first‑token latency. Appendix G acknowledges that even linear‑ish systems still yield multi‑minute TTFT for 1M‑token prompts on consumer hardware.
- Memory‑bandwidth sensitivity under offloading
  - Accessing CPU memory over PCIe adds ≈31.5× latency versus VRAM (Section 5.3 “Latency with KV Offloading”). InfiniteHiP mitigates this via caching and masking, but decoding remains memory‑bound at extreme lengths (Table 4 “Offload (µs)” rows).
- Overheads from dynamic RoPE
  - OOL generalization adds ≈1.6× prefill overhead and ≈5% decoding overhead due to extra sin/cos reads (Table 3).
- Hyperparameter tuning
  - Performance depends on stage sizes, keep‑rates, and refresh intervals (Appendix F). Defaults work broadly, but task‑specific tuning can shift the accuracy/latency trade‑off (Appendix G discusses block‑size trade‑offs).
- Hardware and software dependencies
  - Relies on Nvidia UVM and Triton kernels; portability to other accelerators or runtimes is non‑trivial.
- Scope
  - The work focuses on inference, not training; it does not address training with million‑token sequences or model‑internal reparameterizations.

## 7. Implications and Future Directions
- How it shifts the landscape
  - Demonstrates that million‑token prompts are feasible on a single commodity‑class GPU without discarding context and without fine‑tuning, by co‑designing sparse attention, positional indexing, and memory management. This lowers the barrier to real‑world long‑context applications.
- Practical applications
  - Retrieval‑augmented generation over massive corpora, contract or codebase analysis, long‑horizon agents and logs, multi‑document summarization and QA, and interactive assistants that must retain full conversation history.
- Follow‑up research
  - Reducing TTFT: integrate speculative decoding/prefill or lazy initialization specifically tailored to modular pruning (Appendix G).
  - Memory efficiency: combine with KV quantization/compression (e.g., KVQuant, DeepSeek‑style compression) inside the same offloading framework (Appendix G).
  - Adaptive pruning policies: learn per‑layer or per‑task stage parameters online; develop reliability checks that detect when chunk sparsity fails and fall back gracefully.
  - Broader hardware support: extend UVM‑style paging and graph‑capturable kernels to non‑Nvidia accelerators; multi‑GPU paging across NVLink.
  - Training‑time synergy: pretraining with objectives that sharpen chunk sparsity or encourage stable attention locality could further improve pruning accuracy and OOL robustness.

Overall, InfiniteHiP is a well‑engineered, training‑free path to practical long‑context LLM inference. Its key strength is the coherent combination of algorithmic, architectural, and systems ideas that together deliver both speed and accuracy at unprecedented context scales, as evidenced across benchmark quality, latency, and throughput (Tables 1–5; Figures 3–5).
