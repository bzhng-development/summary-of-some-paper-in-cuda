# LLM in a flash: Efficient Large Language Model Inference with Limited Memory

**ArXiv:** [2312.11514](https://arxiv.org/abs/2312.11514)

## 🎯 Pitch

'LLM in a flash' presents a novel hardware-aware system that enables large language models—those far exceeding device DRAM capacity—to run efficiently by storing parameters in flash memory and intelligently streaming only the needed weights into DRAM on demand. Through techniques like windowing, row-column bundling, and optimized memory management, the method cuts inference latency by up to 20x compared to naïve approaches and lets devices run models up to 2x larger than available RAM. This breakthrough democratizes on-device LLM inference, making powerful models accessible on everyday hardware without sacrificing performance or privacy.

---

## 1. Executive Summary
This work introduces “LLM in a flash,” a hardware-aware inference method that runs large language models whose weights do not fit in DRAM by storing them on flash (SSD) and loading only what is needed, when it is needed. By reducing how much is read from flash and reading it in larger, more contiguous chunks—plus efficient DRAM management—the system runs models up to 2x larger than available DRAM and achieves large per‑token latency reductions (up to 4x on CPU, 7x on Apple Metal, and 20x on NVIDIA GPU vs. naive loading), as shown in Table 3 and summarized in Section 1 and Figure 1.

## 2. Context and Motivation
- Problem addressed
  - On most personal devices (laptops, phones), DRAM is too small to hold all weights of modern LLMs (e.g., a 7B model needs >14 GB in FP16). Standard inference loads the entire model into DRAM, which caps model size and inflates first‑token latency (Section 1; Introduction).
  - Flash memory (SSDs) has much larger capacity but much lower bandwidth and higher per-read latency than DRAM (Figure 2a). Small, random reads from flash are particularly slow (Figure 2b).

- Why it matters
  - Enables on-device inference of larger models without cloud offload, improving privacy, availability, and interactive latency (Section 1). It also reduces the need for aggressive compression that can affect accuracy.

- Prior approaches and gaps
  - Model compression (quantization, pruning) reduces parameter size but still assumes the whole model resides in DRAM at inference time; it doesn’t help when the model fundamentally exceeds DRAM (Related Works; Section 6/E).
  - Offloading systems (e.g., FlexGen) move weights between GPU, DRAM, and flash, but often target settings where the full model fits in DRAM+GPU and are bound by flash→DRAM bandwidth when that assumption breaks (Appendix E).
  - Sparse-activation systems (e.g., DejaVu) exploit that many FFN neurons are inactive per token but still assume GPU/DRAM residency for the subset of weights (Appendix E).

- Positioning of this work
  - Treats SSD as the primary weight store for models larger than DRAM and designs the inference stack around flash’s strengths and weaknesses. Two central levers guide the method (Section 2.2; Section 3):
    - Load less data from flash (via activation sparsity prediction and a sliding window of recently used neurons).
    - When loading, read larger, contiguous chunks and parallelize I/O (bundling FFN row/column pairs; use of multithreaded reads).

## 3. Technical Approach
The system targets per-token inference when model weights primarily live on flash. Total latency per token is decomposed into:
- I/O time to read weights from flash,
- DRAM memory-management overhead to add/remove the needed weights,
- Compute time for the actual forward pass (Section 3).

Three coordinated components reduce I/O and make the remaining I/O faster, while keeping DRAM operations cheap.

1) Reduce how much data is loaded per token (Section 3.1)
- Selective persistence in DRAM
  - Keep embeddings and all attention weights permanently in DRAM, roughly one-third of the model (Section 3.1, “Selective Persistence Strategy”). This avoids reloading these frequently used parts and confines flash reads to the FFN.
- Exploit activation sparsity with a predictor
  - Observation: FFN activations are highly sparse with ReLU-like activations; e.g., OPT‑6.7B shows ~97% sparsity in FFNs; Falcon‑7B can be “relufied” to ~95% with similar accuracy; Llama‑2 with FATReLU reaches ~90% (Section 3.1).
  - Mechanism: Train a small low‑rank predictor per layer that predicts which FFN neurons will be positive after the activation (Figure 3a/b; Section 3.1 “Anticipating ReLU Sparsity”). It consumes only the current layer’s attention output (not previous FFN outputs), improving timing and accuracy for weight loading. Training uses 10k C4 samples, 2 epochs on A100 (Section 3.1).
  - Outcome: Only rows/columns corresponding to predicted‑active FFN neurons are loaded from flash. False negatives are rare and close to zero pre‑activation (Figure 3a), minimizing impact on outputs; zero-shot metrics are preserved (Table 1).
- Sliding Window of active neurons across tokens
  - Idea: Consecutive tokens tend to reuse many of the same neurons. Maintain a DRAM cache of FFN rows/columns predicted active in the most recent k tokens, so only the new, token‑specific delta is read from flash (Figure 4b; Section 3.1 “The Sliding Window Technique”).
  - Behavior: Let sagg(k) be the fraction of unique neurons used by the last k tokens. As k grows, the incremental new neurons per token sagg(k+1) − sagg(k) shrinks (Figure 4a). Choose k as large as DRAM allows to reduce per‑token flash reads.

2) Increase throughput of the flash reads that remain (Section 3.2)
- Hardware facts to exploit
  - Flash is fast for large sequential reads but slow for small random reads due to latency-to-first-byte and multi-stage read pipelines (OS, driver, controller) (Section 2.2).
  - Throughput increases with chunk size and with multiple concurrent threads (Figure 2b).
- Row–column bundling for FFN weights
  - In a Transformer FFN, the ith neuron uses: column i of the “up‑projection” (input→hidden) and row i of the “down‑projection” (hidden→output). Store these together on flash so fetching neuron i is a single contiguous read of size 2 × d_model × num_bytes (Figure 5; Section 3.2).
  - Benefit: Doubles chunk size versus storing rows/columns separately, increasing effective random-read throughput (Table 2 shows throughput improvement from 1.25 GB/s to 2.25 GB/s when bundling is combined with the rest).
- Parallelized reads
  - Use 32 threads to overlap latency and saturate the flash controller, which helps reach the “upper bound” random-read throughput regime seen in Figure 2b (Implementation Details in Section 4.1).

3) Keep DRAM management overhead low (Section 3.3)
- Preallocate and compact
  - For each layer, preallocate a matrix of size Reqi × 2d_model (Reqi is the max number of cached neurons for the chosen window), plus auxiliary arrays: `pointer` (original neuron indices), `bias`, `num_used`, and `last_k_active` (which neurons were active in the last k tokens) (Figure 6).
- Constant‑time deletion and amortized insertion
  - Deletion: If some neurons are no longer in the sliding window, swap their rows with the last occupied rows to keep the active block contiguous; decrease `num_used` (Figure 6, “1. Start deletion” → “2. Deletion complete”).
  - Insertion: Append new required neurons contiguously at the end (“3. Insertion complete”). This avoids expensive reallocation and copying.
  - Inference layout: The first half of the preallocated matrix holds the up‑projection; the second half (transposed) holds the down‑projection. Reordering neurons in the hidden space doesn’t change the FFN output, so this layout preserves correctness (Section 3.3).

Practicalities and extensions
- File-system caching is disabled during throughput benchmarking to measure true flash performance under constrained DRAM (Appendix C, “Caching Considerations,” F_NOCACHE/DirectIO).
- On Apple Silicon, the implementation uses MLX’s Metal kernels for dynamic shapes and unified memory (`MTLStorageModeShared`) to eliminate unnecessary copies (Appendix C).
- Speculative decoding is supported by updating the sliding window with multiple draft tokens at once and choosing which tokens’ neurons to keep based on the acceptance ratio α (Section 5.2).

## 4. Key Insights and Innovations
- Hardware-informed cost model for “weights on flash”
  - Novelty: Rather than trying to squeeze the full model into DRAM, treat flash as the primary store and optimize for its constraints: reduce read volume and make remaining reads large and parallel (Sections 2 and 3).
  - Significance: Unlocks models up to 2x DRAM capacity with competitive latency (Abstract; Figure 1).

- Activation-aware selective loading with a per-layer low‑rank predictor
  - Novelty: A lightweight predictor placed after the attention sublayer in each block forecasts which FFN neurons will be nonzero, so only those neuron weights are read (Section 3.1; Figure 3b). This placement differs from prior work that needed the previous FFN output.
  - Significance: Substantially reduces flash I/O with minimal accuracy loss in zero-shot tasks (Table 1; Appendix B.2 and Table 4). Predictor overhead is small in time and memory (Appendix B.3).

- Sliding-window caching of recently used neurons
  - Novelty: A simple but effective temporal locality mechanism tailored to token-by-token generation that reduces incremental reads as the window expands (Figure 4a/b).
  - Significance: With k=4–5, per-token flash reads shrink markedly (Section 3.1; OPT example in Appendix C.1).

- Row–column bundling of FFN weights
  - Novelty: Re-layout FFN weights on disk by pairing each up‑projection column with its matching down‑projection row to double per-neuron chunk size and raise random-read throughput (Figure 5, Section 3.2).
  - Significance: Converts otherwise scattered small reads into fewer, larger reads; Table 2 shows throughput rising from 1.25 GB/s to 2.25 GB/s when bundling is active alongside other techniques.

- Efficient DRAM data structure and in‑place compaction
  - Novelty: A pointer-indexed, preallocated, contiguous buffer per layer that supports O(1) deletions via swap‑with‑end and append‑only insertions (Figure 6; Section 3.3).
  - Significance: Avoids frequent reallocation/copying that would otherwise erode the I/O gains.

A negative but informative result: bundling neurons by co-activation (“closest friends”) led to repeatedly loading very active neurons and was counterproductive (Appendix D; Figure 12). This clarifies why the row–column bundling (structural) is preferable to co-activation bundling (statistical) for flash I/O.

## 5. Experimental Analysis
Evaluation setup (Section 4.1)
- Hardware
  - Apple M1 Max (1 TB SSD), Apple M2 Ultra (2 TB SSD), and Linux with NVIDIA RTX 4090.
  - CPU runs use float32; Apple Metal runs use float16; RTX uses bfloat16.
- Models
  - `OPT-6.7B`, `Falcon-7B` (relufied), `Persimmon-8B`, `Phi-2 (2.7B)`, and `Llama 2-7B` (sparsified via FATReLU) (Section 4.1).
- Workload
  - Single-sequence generation: prompt = first 128 tokens, then generate 256 tokens from C4 validation subset (Section 4.1).
- Memory budget
  - Approximately half the model size available in DRAM/GPU memory for all but `Phi-2` (set to 65% due to lower sparsity) (Table 3; Section 4.2).
- Baselines
  - Naive: load all needed weights from flash each token.
  - Hybrid: keep half the model in DRAM and read the other half every token (theoretical best I/O for methods that don’t use sparsity) (Section 4.1 “Baselines”).

Key quantitative results
- End-to-end per-token latency (Table 3)
  - `OPT-6.7B`:
    - CPU: Naive 3182 ms → All (predictor + windowing + bundling) 669 ms (≈4.8x faster).
    - Apple Metal (M1): 2389 ms → 565 ms (≈4.2x).
    - Apple Metal (M2): 2270 ms → 305 ms (≈7.4x).
    - NVIDIA GPU: 2218 ms → 84 ms (≈26x).
    - With speculative decoding: 60 ms total (additional ≈1.4x over All; also shown with variability in Table 5).
  - `Falcon-7B` CPU: 3095 ms → 706 ms (≈4.4x). Hybrid is 1947 ms, so All beats both Naive and the theoretical hybrid baseline.
  - `Persimmon-8B` CPU: 3806 ms → 1041 ms (≈3.7x); Hybrid 2495 ms.
  - `Phi-2 (2.7B)` CPU: 1287 ms → 546 ms (≈2.4x); Hybrid 711 ms.
  - `Llama 2-7B` CPU: 3095 ms → 994 ms (≈3.1x); Hybrid 1903 ms.
- I/O ablation on `OPT-6.7B` (M1 Max, FP16) (Table 2)
  - Starting from Naive I/O of 2196 ms (13.4 GB read at 6.1 GB/s), introducing the predictor halves traffic to 6.7 GB (1090 ms).
  - Adding windowing reduces flash traffic to 0.9 GB but throughput drops to 1.25 GB/s due to scattered reads (738 ms I/O).
  - Bundling doubles the read chunk size, yielding the same 0.2 GB traffic at 1.25 GB/s → 164 ms.
  - With a hybrid DRAM layout plus bundling, throughput rises to 2.25 GB/s for the same 0.2 GB → 87 ms I/O.
- Memory–latency tradeoff (Figure 7)
  - On GPU, increasing the fraction of model kept in DRAM from 35% to 80% steadily lowers Load‑From‑Flash and memory‑management times; compute is near-flat.
- Long-generation robustness (Figure 8)
  - Generating up to 1000 tokens shows average flash latency stays stable; the first few tokens are slower due to cold cache fill, but no SSD thermal throttling trends are observed.
- Accuracy and predictor overhead
  - Zero-shot accuracy: Table 1 and Table 4 show marginal or no drop on Arc-Easy/Challenge and HellaSwag when using predictors (e.g., `OPT-6.7B` 66.1/30.6/50.3 vs. 66.2/30.6/49.8).
  - MMLU for `Persimmon-8B` can be maintained by using larger predictors in later layers (Figure 10a). For `Phi-2`, relufication plus distillation keeps MMLU ≈52 with predictors (Figure 10b).
  - Predictor compute cost is small: for `OPT-6.7B`, <2.4% of non-embedding FLOPs; 2.75% of CPU time on M1 Max and 4.8% on RTX GPU (Appendix B.3).

Do the experiments support the claims?
- The per-token latency gains are repeatedly demonstrated across multiple models and backends in Table 3, and the step-by-step I/O reductions are isolated in Table 2. The stability across generation lengths (Figure 8) and the memory–latency curve (Figure 7) reinforce that the design scales with different DRAM budgets and session lengths.
- The accuracy checks (Table 1, Table 4; Figures 10a–b, 11) suggest low‑rank prediction and sparsity do not materially degrade zero‑shot performance in the tested settings.
- Ablations and negative results (Table 2; Appendix D) illuminate what does and does not help, strengthening causal interpretation.

Caveats within the results
- Gains depend on having substantial FFN sparsity and predictor precision; `Phi-2` shows smaller gains (≈2.4x) due to lower sparsity (Table 3; Appendix C.4).
- Power: instantaneous power is lower for the sparse method, but total energy can be higher due to longer generation time (Section 5.3).

## 6. Limitations and Trade-offs
- Assumptions about sparsity
  - The approach assumes high FFN activation sparsity (≥90%) to materially reduce flash I/O. Models with SwiGLU often need relufication or FATReLU fine‑tuning to reach such sparsity (Section 3.1; Appendix C.5), which may require task-specific adaptation and can slightly affect accuracy (Figure 11; Llama‑2 MMLU notes in Appendix C.5).

- Predictor training and tuning
  - Requires per-layer predictors trained on samples (10k C4 examples, 2 epochs) with threshold tuning and layer-specific ranks (Appendix B, Table 4). While overhead is modest at inference, training adds a setup cost.

- Single-sequence focus
  - Experiments run batch size = 1 to prioritize KV-cache and model-size constraints (Section 4.1). Multi-batch or prompt-processing phases are not explored; interactions between multiple concurrent sequences and the sliding window are open.

- DRAM budgeting and window management
  - The window size k must be tuned to DRAM availability; too small increases I/O, too large increases DRAM usage and memory-management overhead (Figure 7; Section 4.3). Early tokens pay a heavier I/O cost to “warm” the cache (Figure 8).

- Dependence on flash behavior and OS stack
  - Benefits hinge on SSDs with good random-read scaling via chunk size and multithreading (Figure 2b). Filesystem cache was disabled to measure true I/O; real deployments may need careful cache policy to avoid DRAM pressure (Appendix C, “Caching Considerations”).

- Energy trade-off
  - Despite lower instantaneous power, the sparse approach can consume more total energy due to longer time-to-generate (Section 5.3), which matters for battery-powered devices.

## 7. Implications and Future Directions
- What changes now
  - Treating SSD as the primary store reshapes how on-device LLM inference can be architected. With the right I/O patterns and minimal DRAM residency, devices can serve models roughly twice their DRAM size with acceptable latency (Figure 1; Abstract; Table 3).

- Practical applications
  - Private, offline assistants on laptops; edge deployments where cloud is unavailable; developer workflows where larger models can run locally without GPU-class DRAM.
  - Appendix F sketches smartphone feasibility when combined with 4‑bit quantization, provided device kernels support low‑bit compute and the same sparsity holds (Table 6).

- Follow-up research enabled
  - Smarter bundling: The co-activation negative result (Appendix D; Figure 12) suggests investigating more sophisticated bundling strategies (e.g., disjoint cluster bundles that avoid reloading hot neurons).
  - Multi-batch, multi-session memory managers: Extending the sliding window to shared/evolving caches across conversations or users.
  - Joint design with compression: Integrating quantization/pruning with flash-aware layouts and predictors while preserving accuracy.
  - Power/thermal modeling: A systematic measurement of energy vs. latency trade-offs and thermal constraints for long sessions (Section 8).
  - Architectures with built-in flash-friendly layouts: File formats or weight sharding schemes aligned to predictor access patterns and SSD block sizes.
  - Broader decoding strategies: The speculative decoding prototype (Table 5; Section 5.2) could be combined with other parallel or early-exit decoding methods.

Quoted highlights (grounding in the paper)
- Flash vs DRAM capacity/bandwidth:
  > “Flash memory offers significantly higher capacity but suffers from much lower bandwidth compared to DRAM…” (Figure 2a)
- Random-read throughput improves with chunk size and threads:
  > “The throughput for random reads in flash memory increases with the size of sequential chunks and the number of threads.” (Figure 2b)
- End-to-end improvements with the full method:
  > For `OPT‑6.7B` on NVIDIA GPU: “Naive 2218 ms … All 84 ms” (Table 3)
- Stepwise I/O reduction:
  > “Using predictors, windowing, and bundling… I/O latency [drops] from 2196 ms to 87 ms” for `OPT‑6.7B` on M1 Max (Table 2)
- Capability claim:
  > “Run models 2x larger than the device’s DRAM capacity and speed up inference up to 4x, 7x, and 20x compared to naive implementations in CPU, Metal, and NVIDIA GPU backends, respectively.” (Section 1; also reflected in Table 3)

In sum, LLM in a flash is a well-motivated, hardware-aligned design that demonstrates sizeable latency gains by pairing activation-aware selective loading with flash-friendly I/O and DRAM-efficient data structures. It opens a clear path for larger on-device models and invites a wave of systems-and-algorithms co-design around SSD-centric inference.
