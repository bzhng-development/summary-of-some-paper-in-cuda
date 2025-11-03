# LLM in a flash: Efficient Large Language Model Inference with Limited Memory

**ArXiv:** [2312.11514](https://arxiv.org/abs/2312.11514)
**Authors:** Keivan Alizadeh, Iman Mirzadeh, Dmitry Belenko, S. Karen Khatamifard, Minsik Cho, Carlo C. Del Mundo, Mohammad Rastegari, Mehrdad Farajtabar
**Institutions:** Apple (Apple Machine Learning Research)

## 🎯 Pitch

This paper revolutionizes large language model (LLM) inference by utilizing a hardware-aware scheme that leverages flash storage, activation sparsity, and innovative data layout, effectively doubling model-to-DRAM size capacity and reducing per-token latency by up to 20× on GPUs. This advancement enables practical on-device LLM deployment, enhancing privacy, responsiveness, and offline availability, crucial for expanding AI capabilities on devices with limited DRAM resources.

---

## 1. Executive Summary
This paper introduces a practical way to run large language models (LLMs) that are bigger than a device’s DRAM by storing weights on flash storage and loading only the needed parts on demand during inference. The core contribution is a hardware-aware inference scheme—built around activation sparsity, a small predictor, a sliding window cache, and a “row–column bundling” data layout—that cuts flash I/O while making the inevitable reads much faster, enabling up to 2× model-to-DRAM size and large per-token latency reductions (e.g., 4× on CPU and 20× on GPU versus a naive approach; Figure 1 and Table 3).

## 2. Context and Motivation
- Problem addressed
  - Many devices lack enough DRAM to hold a full LLM. Even a 7B model in 16-bit format needs ~14 GB just for parameters; that exceeds typical smartphone DRAM and is tight for many laptops (Introduction).
  - Standard practice is to load the entire model into DRAM; this hard-caps model size and wastes energy/time when only a fraction of weights are used per token.

- Why this matters
  - On-device inference promises privacy, responsiveness, and availability offline, but is constrained by DRAM capacity. Flash storage (e.g., NVMe SSD) is larger but has limited bandwidth and poor random-read performance (Section 2; Figure 2a and 2b).
  - The paper targets a realistic setting: models larger than DRAM where weights must live on flash. It aims to minimize end-to-end per-token latency by reducing bytes read and increasing effective flash throughput.

- Prior approaches and gaps
  - Compression: pruning/quantization reduce model size but often still require loading the entire compressed model into DRAM (Related Works).
  - Offloading frameworks (e.g., FlexGen): move weights/KV-cache between GPU, DRAM, and flash to stretch GPU memory, but still assume the model fits in system DRAM and remain bound by flash→DRAM throughput under DRAM scarcity (Section 6 and Appendix E).
  - Activation sparsity methods (e.g., DejaVu): exploit which neurons are active to reduce compute/memory, but assume GPU/DRAM residency of weights or access patterns unsuited to flash (Appendix E).
  - Gap: A method specifically engineered to serve LLMs directly from flash when DRAM is insufficient, with flash characteristics (latency-to-first-byte, random vs sequential throughput, parallelism) baked into the algorithm.

- Positioning
  - The work reframes inference as a flash I/O optimization problem: minimize bytes transferred and maximize throughput of the remaining reads while keeping compute overhead low (Section 3). It complements compression/sparsity work and can be combined with it (Section 7; Appendix F).

## 3. Technical Approach
At a high level, the system keeps some weights permanently in DRAM and fetches the rest on demand from flash. The design is governed by a simple cost model: per-token latency ≈ flash I/O time + DRAM memory management + compute (Section 3). The method focuses on shrinking flash I/O and boosting flash read throughput.

Step-by-step pipeline for one generated token:
1) Decide which weights will be used
   - Exploit activation sparsity in each transformer block’s Feed-Forward Network (FFN): typically >90% of intermediate neurons are zero after ReLU-like activations in suitable models (Section 3.1; citing rates for OPT 6.7B, Falcon 7B, Llama 2 with FATReLU).
   - Use a tiny per-layer “low-rank predictor” to forecast which FFN neurons will be active for the current token, so only those weights are needed (Section 3.1; Figure 3b).
     - What is a “low-rank predictor”? A small, trained linear module that takes the current layer’s attention output and predicts a binary mask indicating which FFN neurons will be nonzero after activation. It approximates the high-dimensional “up-projection” with a low-rank surrogate to cheaply estimate which entries will be positive after ReLU. The predictor is trained on a subset of data (10k C4 samples, 2 epochs; Appendix B) to minimize false negatives, using a balanced loss.
     - Why this predictor? Without it, you’d need to load the full up-projection matrix to discover who activates. With it, you only load the rows/columns for predicted-active neurons (Figure 3a shows false negatives are rare and close to zero anyway; Table 1 shows negligible zero-shot accuracy change).

2) Keep the “always-hot” part in DRAM
   - Store embeddings and all attention weights permanently in DRAM; these are about one-third of the model and essential every step (Selective Persistence Strategy in Section 3.1).

3) Reuse recent neurons via a sliding window cache
   - Observation: the union of active neurons over recent tokens grows sublinearly; successive tokens reuse many neurons (Figure 4a).
   - Maintain a cache in DRAM containing the union of predicted-active neurons for the last k tokens (“window size” k). For the new token, load only the neurons in the current prediction that are not already in the window, and evict neurons that fall out of the window (Figure 4b).
   - Formalization: if sagg(k) is the cumulative unique-activated neurons over the last k tokens, each new token requires loading roughly sagg(k+1) − sagg(k) from flash, which decreases as k grows (Section 3.1).

4) Lay out weights on flash to read larger chunks per activation
   - Flash hardware is bad at many small random reads because each read pays a fixed “latency to first byte” (Section 2.2; Figure 2b). So the system:
     - Bundles the up-projection column and the matching down-projection row for the same FFN neuron contiguously in the flash file (“row–column bundling,” Section 3.2; Figure 5).
     - Rationale: When an FFN neuron activates, both its up-projection (to compute the hidden vector) and its down-projection (to project back) are needed. Reading them as a single contiguous record doubles the chunk size to 2·d_model elements, amortizing the per-read overhead and increasing effective throughput (Section 3.2).
     - Negative result: trying to bundle “co-activated friends” of a neuron backfired because highly active neurons are everyone’s “closest friend,” causing repeated reloads (Appendix D).

5) Read from flash in parallel, in “large enough” chunks
   - Empirical hardware study shows random-read throughput rises with chunk size and threading (Figure 2b). The system uses ~32 KiB reads or larger and 32 threads to saturate controller parallelism (Implementation details; Section 4.1).
   - Benchmarks are done without OS page cache to get true device throughput (Appendix C).

6) Manage DRAM-resident weights with O(1) reallocation
   - Naively appending/removing rows in the active FFN matrices would cause frequent reallocations and large copies.
   - Instead, preallocate a fixed-size matrix per layer large enough for the chosen window (estimated from data), of shape `Req_i × 2·d_model`, plus:
     - `pointer` array mapping matrix rows back to original neuron indices,
     - `bias` vector for up-projection biases,
     - `num_used` to track how many rows are occupied,
     - `last_k_active` to record recently used neuron IDs (Figure 6; Section 3.3).
   - Efficient updates each token (Figure 6):
     - Deletions: swap rows to keep the matrix densely packed and decrement `num_used`. Cost is O(c·d_model) for c deletions.
     - Insertions: read missing neurons’ bundled records from flash and append to the end (no reallocation).
     - Compute: treat the first half of the matrix as the “up” weights and the transposed second half as the “down” weights. Because FFN activation order doesn’t affect the final sum, arbitrary row order is fine (Section 3.3).

7) Compute the layer with the active subset
   - Run the FFN on only the cached active neurons for the window, then proceed to the next layer and next token. For GPU backends on Apple Silicon, custom Metal kernels (from MLX) and unified shared memory avoid shape-dynamism cliffs and extra copies (Appendix C).

Design choices and why
- Keep attention and embeddings in DRAM: these are widely reused, about one-third of model size, and avoid frequent random accesses (Section 3.1).
- Use a predictor from current layer’s attention output (not previous FFN): deferring prediction to later in the computation yields more accurate masks without extra dependency on last layer’s FFN (Section 3.1).
- Windowing rather than full caching: the window bounds memory growth and matches reuse behavior (Figure 4a).
- Row–column bundling vs. fine-grained reads: bundling increases chunk size for each needed neuron, which is essential for flash throughput (Section 3.2; Figure 2b).
- Preallocation + swap-delete: avoids costly reallocation/copies, keeping memory management time small (Section 3.3).

Cost model in plain language (Section 3)
- Flash I/O latency = bytes transferred ÷ achieved throughput. Reduce both numerator (by predicting and caching) and increase denominator (by bundling and threading).
- Memory management adds overhead proportional to the number of rows evicted/inserted; keep this low by windowing and preallocation.
- Compute is orthogonal here; the paper keeps compute standard and focuses on I/O and memory.

## 4. Key Insights and Innovations
- Activation-aware on-demand weight paging from flash
  - What’s new: selective loading of only the FFN neurons predicted to fire (Section 3.1), with attention/embeddings pinned in DRAM.
  - Why it matters: shrinks bytes read per token by orders of magnitude compared to loading entire layers, making flash-based inference viable (Table 2, rows comparing “Naive” vs “Predictor + Windowing”).

- Sliding-window neuron cache across tokens
  - What’s new: maintain the union of active neurons over the last k tokens so only deltas are loaded/evicted each step (Section 3.1; Figure 4a–b).
  - Why it matters: successive tokens share many active neurons; the incremental load per token drops as k increases, significantly reducing I/O (Figure 4a).

- Row–column bundling data layout for flash
  - What’s new: put each neuron’s up-projection column and down-projection row next to each other on disk to double read chunk size (Section 3.2; Figure 5).
  - Why it matters: flash throughput is limited by per-read overhead; bundling increases effective throughput from ~1.25 GB/s to ~2.25 GB/s in the sparse-read regime on M1 Max (Table 2, last two rows), nearly halving I/O time.

- DRAM-side data structure for zero-reallocation updates
  - What’s new: preallocated matrices plus pointer/metadata to support O(1) insert-at-end and swap-delete operations, avoiding costly reshapes and copies (Section 3.3; Figure 6).
  - Why it matters: memory management overhead becomes small compared to compute and I/O (e.g., OPT-6.7B GPU “All” shows 34 ms memory vs 30 ms I/O and 20 ms compute per token; Table 3).

- Hardware-informed throughput tuning
  - What’s new: measure real flash characteristics (Figure 2b), disable OS caches for true device throughput (Appendix C), and use 32 parallel threads with ≥32 KiB reads.
  - Why it matters: turns many small random reads into fewer larger, parallelized reads, which is essential for the overall speedups.

These are fundamental system-level innovations for flash-resident LLM inference rather than incremental tweaks to model architecture or compression.

## 5. Experimental Analysis
- Evaluation setup
  - Hardware: Apple M1 Max (1 TB SSD), Apple M2 Ultra (2 TB SSD), Linux with NVIDIA RTX 4090 (24 GB). Apple backends tested with CPU float32 and Metal float16; RTX with bfloat16 (Section 4.1).
  - Models: OPT-6.7B, Falcon-7B (relufied/sparsified), Persimmon-8B, Phi-2 (2.7B, relufied), Llama 2-7B (sparsified by FATReLU) (Section 4.1).
  - Workload: single-sequence decoding; prompts of 128 tokens then generate 256 new tokens; about half of model size available in DRAM unless noted (Section 4.1).
  - Baselines: “Naive” reload of needed half-model per token; “Hybrid” keeps half in DRAM, loads the rest per token without sparsity. Both use the best possible theoretical I/O (Section 4.1, Baselines).
  - Measurement: per-token latency broken into I/O, memory management, and compute (Table 3). Additional I/O-only analysis in Table 2.

- Key quantitative results
  - End-to-end per-token latency (Table 3):
    - OPT-6.7B:
      > Naive GPU: total 2218 ms; “All” (predictor + windowing + bundling + optimized memory): 84 ms (≈26× faster); with speculative decoding: 60 ms.
      > Naive CPU: 3182 ms; “All” CPU: 669 ms (~4.8× faster).
      > Metal M1: 2389 → 565 ms; Metal M2: 2270 → 305 ms.
    - Falcon-7B (CPU): Naive 3095 ms; Hybrid 1947 ms; “All” 706 ms.
    - Persimmon-8B (CPU): Naive 3806 ms; Hybrid 2495 ms; “All” 1041 ms.
    - Phi-2 2.7B (CPU): Naive 1287 ms; Hybrid 711 ms; “All” 546 ms.
    - Llama 2-7B (CPU): Naive 3095 ms; Hybrid 1903 ms; “All” 994 ms.
  - I/O-only ablations on OPT-6.7B 16-bit (M1 Max; Table 2):
    - Naive contiguous read: move 13.4 GB at 6.10 GB/s → 2196 ms I/O per token.
    - With predictor only: move 6.7 GB → 1090 ms.
    - + Windowing: move 0.9 GB at 1.25 GB/s (sparse random reads) → 738 ms.
    - + Bundling: move 0.2 GB at 1.25 GB/s → 164 ms.
    - + Higher throughput via configuration: 0.2 GB at 2.25 GB/s → 87 ms.
  - Memory–latency trade-off (Figure 7):
    > As the percentage of model kept in DRAM grows from ~35% to ~80%, the “Load From Flash” component drops sharply, showing a continuous latency–memory curve for tuning to a device’s DRAM budget.
  - Long-generation stability (Figure 8):
    > No signs of SSD thermal throttling over 1000-token generations. I/O latency is highest in the first few tokens (cold cache) and stabilizes afterward. Nucleus sampling (diverse outputs) did not hurt performance versus greedy decoding in long runs.
  - Speculative decoding (Section 5.2; Table 5):
    > Using a draft length λ=4 and an acceptance-aware windowing strategy yields ~1.4× speedup over the already-optimized “All” GPU pipeline (84.6 → 60.2 ms), close to the 1.58× theoretical ideal.
  - Accuracy impact of predictors:
    - Zero-shot tasks remain nearly unchanged (Table 1, Table 4), e.g., OPT-6.7B ArcEasy 66.1→66.2; HellaSwag 50.3→49.8.
    - MMLU changes are minor or recoverable by using larger predictors where layers are less sparse (Figure 10a for Persimmon-8B) or by using distillation when relufying (Phi-2, Figure 10b).

- Do the experiments support the claims?
  - Yes, the ablations (Table 2) isolate the effect of each component on I/O; the end-to-end breakdown (Table 3) shows consistent speedups across models and backends; and robustness checks (Figure 8 long generations; Section 5.1) address practical concerns like SSD throttling and sampling diversity.
  - Hardware awareness is validated by throughput curves (Figure 2b) and the benefits of bundling and threading.

- Noteworthy negative/neutral findings
  - Co-activation-based bundling harms performance by repeatedly loading “celebrity neurons” that co-activate with many others (Appendix D).
  - Power: instantaneous power is lower for the sparse pipeline, but total energy can be higher because decoding may take longer (Section 5.3), suggesting a speed–energy trade-off that needs further study.

## 6. Limitations and Trade-offs
- Assumptions and prerequisites
  - Significant FFN activation sparsity (often ≥90%) is needed to get large I/O savings. The method uses ReLU or FATReLU “relufication” to ensure sparsity in some models (Section 3.1; Appendix C.4–C.5).
  - Attention and embeddings must fit in DRAM; these are ~1/3 of model size (Section 3.1).
  - Single-sequence, single-batch decoding is the primary focus; extensions to multi-batch/prompt processing are future work (Section 8).

- Accuracy vs efficiency
  - Predictors occasionally misclassify, but false negatives are small and near zero (Figure 3a); still, aggressive predictor thresholds or too-small ranks can degrade accuracy (Table 4; Figure 10a–b). There is a tunable accuracy–I/O trade-off.
  - Relufication can affect accuracy; distillation or larger predictors can recover some losses (Appendix C.4, Figure 10b).

- Hardware/software constraints
  - Requires careful data layout on flash and a custom runtime that manages per-layer preallocated buffers, 32-threaded I/O, and unified memory on Apple Silicon to avoid shape-dynamism slowdowns (Appendix C).
  - Flash throughput for small random reads is fundamentally limited; bundling and parallelism mitigate but do not erase this gap versus DRAM (Figure 2a–b).

- Resource trade-offs
  - Larger window k reduces I/O but increases DRAM usage (Figure 7). Devices with less DRAM must use smaller k and accept higher latency.
  - Predictors add a small computation and parameter overhead (e.g., 2–5% of inference time in reported setups; Appendix B.3).

- Open questions
  - How to optimally set window size adaptively across layers and generation stages?
  - Can better predictors or architectural cues (e.g., gating logits in GLU variants) remove the need for separate predictor modules (Appendix C.5 “Alternative approaches”)?
  - Comprehensive power/energy profiling across devices is deferred (Section 5.3; Section 8).

## 7. Implications and Future Directions
- How this changes the landscape
  - Establishes flash-resident LLM inference as a viable regime, not just a stopgap: models up to ~2× DRAM size can be served with competitive latencies, expanding the set of devices that can run mid-size LLMs locally (Figure 1; Table 3).
  - Reorients optimization from “fit the model in DRAM” to “minimize and accelerate flash I/O,” a direction orthogonal to standard compression and complementary with it (Section 7; Appendix F).

- Practical applications
  - On-device assistants on laptops/phones where DRAM is tight but fast NVMe/flash exists.
  - Edge devices that require privacy and offline operation.
  - Systems that must cold-start quickly or serve sporadic queries without keeping the whole model hot in memory.

- Follow-up research enabled or suggested
  - Better neuron-activity predictors: smaller, more accurate, possibly learned jointly during fine-tuning; exploit architecture-specific signals (e.g., gates in GLU/FATReLU models; Appendix C.5).
  - Advanced bundling strategies: beyond row–column, explore grouping that increases chunk sizes without repeatedly loading “celebrity” neurons (Appendix D).
  - Adaptive windowing: layer-wise and time-varying k driven by observed reuse and DRAM headroom; integrate with speculative decoding acceptance dynamics (Section 5.2).
  - Combine with quantization: load 4-bit weights directly from flash with specialized kernels on mobile SoCs (Appendix F), preserving activation sparsity behavior (Table 6).
  - Multi-sequence batching and prompt processing under DRAM limits: schedule-aware caching and eviction policies for overlapping sequences (Section 8).

- Broader systems insights
  - The paper shows that flash’s “latency to first byte” and parallelism characteristics should shape ML software design. Reading slightly more data in larger, coalesced chunks can be faster than reading only what you need in tiny pieces (Section 2.2; Figure 2b; Table 2).
  - The compute graph can remain mostly unchanged; most gains come from the data layer and memory management. This lowers adoption barriers for existing models/runtimes.

Quotes and key references to anchor the above:
- “The throughput for random reads in flash memory increases with the size of sequential chunks and the number of threads” (Figure 2b).
- “By bundling columns of the up project and rows of the down project layer, we can load 2× chunks” (Figure 5).
- “Aggregated neuron usage… slope is decreasing… only a small fraction of new weights need to be loaded” (Figure 4a–b).
- OPT-6.7B I/O ablation: from 2196 ms (naive) → 87 ms (predictor + windowing + bundling + higher throughput) (Table 2).
- End-to-end OPT-6.7B: GPU 2218 ms → 84 ms (All), → 60 ms with speculative decoding (Table 3; Table 5 for means and stddevs).

Overall, “LLM in a Flash” offers a clear, actionable recipe to make flash-first inference not only possible but fast, opening a new system design space for on-device and edge deployment of LLMs beyond DRAM limits.
