# ZeRO: Memory Optimizations Toward Training Trillion Parameter Models

**ArXiv:** [1910.02054](https://arxiv.org/abs/1910.02054)

## 🎯 Pitch

ZeRO introduces the Zero Redundancy Optimizer, a set of innovative memory optimizations that eliminate redundant training state copies across GPUs and optimize communication patterns, enabling the efficient training of deep learning models with hundreds of billions—and even up to a trillion—parameters. By fundamentally rethinking data and model parallelism, ZeRO dramatically increases model size and training speed on today's hardware, removing a major barrier to scaling and democratizing large model training for the broader AI community.

---

## 1. Executive Summary (2-3 sentences)
This paper introduces ZeRO (Zero Redundancy Optimizer), a set of system-level memory optimizations that removes redundant copies of training states across GPUs and re-schedules communication so very large neural networks can be trained efficiently. It enables models with 100B+ parameters to train at high throughput on today’s hardware and shows, via concrete memory and communication analysis, that trillion-parameter training is feasible with a modest GPU count.

## 2. Context and Motivation
- Problem addressed
  - Training very large models (billions to trillions of parameters) quickly hits GPU memory limits. The largest part of memory during training is not just the model weights, but a combination of optimizer states (e.g., Adam’s momentum/variance), gradients, and activations.
  - Basic data parallelism (replicating the whole model on each GPU) offers good compute/communication efficiency but wastes memory because all model states are duplicated on every GPU. Section 1 notes that baseline data parallelism runs out of memory around 1.4B parameters on 32GB GPUs.
  - Model parallelism (splitting a layer’s parameters across GPUs) reduces per-GPU memory, but introduces heavy per-layer communication and smaller compute granularity; it scales poorly across nodes. A 40B model across two DGX-2 nodes achieved only about 5 TFLOPS per V100 (<5% of peak), as reported in Section 1.

- Why it matters
  - Larger models have repeatedly yielded large accuracy gains in NLP and beyond (Section 1), but the field was hitting a memory/scale wall. Making more memory available per device without sacrificing efficiency materially expands what models can be trained and by whom.

- Prior approaches and their shortfalls
  - Data parallelism (DP): memory-inefficient due to state replication; efficient communications (one all-reduce per step) but cannot fit very large models (Sections 1–3).
  - Model parallelism (MP): fits larger models but at high communication cost and worse per-GPU efficiency, especially across nodes (Sections 1–2.1).
  - Pipeline parallelism (PP): can reduce memory, but constrains batch sizes, requires complex scheduling, and may deviate from standard training semantics (Section 2.1).
  - CPU offloading: drastically slower due to PCIe bandwidth limits, sometimes spending up to 50% of time transferring tensors (Section 2.2.2).

- Position of this work
  - ZeRO keeps the low-communication, coarse-grained compute of DP, but removes its memory redundancy by partitioning training states across data-parallel ranks and moving only what is needed when it is needed (Sections 4–5, 7).
  - It also shrinks “residual” memory (activations, temporary buffers, and fragmentation) to prevent secondary bottlenecks (Section 6).

## 3. Technical Approach
ZeRO consists of two complementary parts:
- ZeRO-DP: reduces memory for model states (optimizer states, gradients, parameters) while keeping DP-like communication volume and compute granularity (Sections 4.1, 5).
- ZeRO-R: reduces the remaining “residual” memory (activations, temporary buffers, and fragmentation) (Sections 4.2, 6).

Foundational memory accounting (Section 3):
- With mixed precision and Adam, training an `Ψ`-parameter model needs:
  - `2Ψ` bytes for FP16 parameters + `2Ψ` bytes for FP16 gradients.
  - `4Ψ` (FP32 master weights) + `4Ψ` (momentum) + `4Ψ` (variance) = `12Ψ` more bytes for optimizer states.
  - Total for model states: `16Ψ` bytes (Figure 1 caption and Section 3.1).
- Activations can be enormous: a GPT-2 1.5B model with sequence length 1024 and batch size 32 needs ~60 GB for activations; checkpointing lowers this to ~8 GB, but for a 100B model activation memory is still ~60 GB even with checkpointing (Section 3.2).

Step-by-step: ZeRO-DP (Sections 5, 7)
- Notation
  - `Nd`: degree of data parallelism (number of DP processes).
  - “Partition” means each DP process owns and stores only 1/`Nd` of the tensor.

- Stage 1 — `Pos` (partition optimizer states; Section 5.1)
  - What: Shard Adam’s FP32 master weights, momentum, and variance across `Nd` processes. Each process keeps and updates only its shard, then shares the updated parameters once per step.
  - How it works: At the end of the step, an all-gather collects updated parameters so every process has the full parameter view for the next forward pass.
  - Memory effect: `4Ψ + (KΨ / Nd)` instead of `4Ψ + KΨ` for model states (Figure 1; `K=12` for Adam). Example in Figure 1: a 7.5B model with `Nd=64` drops from 120 GB to 31.4 GB for model states.

- Stage 2 — `Pg` (partition gradients; Section 5.2)
  - What: As backprop produces each layer’s gradients, they are reduce-scattered directly to the process that owns the corresponding parameter shard. The owner accumulates the reduced gradient; non-owners can free it immediately.
  - How it works: Replace the usual DP all-reduce with a reduce-scatter over gradient “buckets,” so each rank only keeps the gradients it needs to update.
  - Memory effect: Gradient memory shrinks from `2Ψ` to `2Ψ/Nd`. Combined with `Pos`, total model-state memory approaches `2Ψ + 14Ψ/Nd` (Section 5.2).
  - Communication volume: Same as DP. A reduce-scatter (`Ψ`) plus an all-gather (`Ψ`) per step equals `2Ψ`—the same as the DP all-reduce (Section 7.2.1).

- Stage 3 — `Pp` (partition parameters; Section 5.3)
  - What: Each process stores only its parameter shard. For layers it does not own, it temporarily receives the needed shard just-in-time, uses it, then discards it.
  - How it works: Pipeline the all-gather of parameters across the forward (and reverse for backward): before a partition’s forward, the owner broadcasts its shard; after use, the shard is dropped. Repeat in reverse order for backward.
  - Memory effect: All model-state memory now scales as `16Ψ/Nd` (Figure 1).
  - Communication volume: `3Ψ` per step (reduce-scatter for gradients (`Ψ`) + two pipelined all-gathers (`2Ψ`)), i.e., only 1.5× the baseline DP volume (Section 7.2.2).

Step-by-step: ZeRO-R (Section 6)
- `Pa`: partitioned activation checkpointing (Section 6.1; see also Section 8)
  - Background: Model parallelism typically replicates activations (intermediate tensors) across MP GPUs, wasting memory. “Activation checkpointing” stores only a subset of activations and recomputes the rest on demand; ZeRO-R goes further.
  - What: Partition the saved activation checkpoints across MP ranks (rather than replicate). When an activation is needed, perform an all-gather to reconstruct it, then discard it again.
  - Why it’s viable: In transformers the arithmetic intensity (compute per byte moved) is very high (≥10K and increasing with hidden size), so the all-gather cost is small compared to computation (Section 4.2.1).
  - Memory effect: Reduces activation checkpoint memory by the MP degree. Example (Section 6.1): a 100B model with MP=16 would need ~33 GB per GPU for activation checkpoints; `Pa` cuts this to ~2 GB. Optional `Pa+cpu` offloads these 2 GB per GPU to CPU for near-zero activation memory at extra PCIe/NVLink traffic.
  - Communication effect: In Megatron-LM-style MP, each transformer block has six all-reduces, total `≈12×(seq_len×hidden_dim)` elements moved. `Pa` adds one all-gather of `(seq_len×hidden_dim)` per block—less than 10% overhead (Section 8).

- `CB`: constant-size fused buffers (Section 6.2)
  - Background: Frameworks fuse many small tensors into large buffers to get high bandwidth for collective operations. But fusing the entire model can make temporary buffers scale linearly with model size (e.g., a 3B model needs a 12 GB FP32 fused buffer).
  - What: Cap fused buffer sizes at a constant, performance-efficient size to bound memory while retaining high bandwidth.

- `MD`: on-the-fly memory defragmentation (Section 6.3)
  - Background: Checkpointing and backprop interleave long- and short-lived tensors, creating fragmentation that can cause out-of-memory errors despite free space (Section 3.2 notes OOM with >30% “free” memory in extreme cases).
  - What: Pre-allocate contiguous arenas for long-lived tensors (checkpoints, parameter gradients) and copy into them as they are produced. This both reduces OOM risk and speeds up allocations.

Communication summary (Sections 7–8)
- Baseline DP: one all-reduce = reduce-scatter (`Ψ`) + all-gather (`Ψ`) ⇒ `2Ψ` elements moved per step (Section 7.1).
- ZeRO-DP:
  - `Pos+g`: reduce-scatter (`Ψ`) + all-gather (`Ψ`) ⇒ `2Ψ` (same as DP; Section 7.2.1).
  - `Pos+g+p`: add two pipelined all-gathers for forward/back ⇒ `3Ψ` (1.5× DP; Section 7.2.2).
- ZeRO-R `Pa`: in Megatron-LM style MP, adds <10% of MP’s communication per transformer block (Section 8), while enabling much larger batch sizes that can reduce DP communication pressure.

Implementation scope evaluated (Section 10)
- The evaluated system, “ZeRO-100B,” includes `Pos+g` plus all ZeRO-R components. Stage 3 (`Pp`) is analyzed but not part of the reported implementation (Sections 1, 10).

## 4. Key Insights and Innovations
- Partition, don’t replicate, model states in DP (fundamental)
  - Novelty: Data-parallel training typically replicates parameters, gradients, and optimizer states on every GPU. ZeRO-DP shards all three across `Nd` processes, keeping only the necessary subset locally, and re-materializes full tensors only when needed (Sections 5.1–5.3).
  - Significance: Reduces per-GPU model-state memory by up to 4× (`Pos`), then 8× (`Pos+g`), and finally by `Nd`× (`Pos+g+p`), with only a modest 1.5× communication increase in the last stage (Figure 1; Sections 5, 7).

- Temporal scheduling of states (mechanism-level insight; fundamental)
  - Insight: Not all states are needed all the time. For each layer’s forward/backward, only that layer’s parameters/activations are required at that moment (Section 4.1).
  - Mechanism: Pipeline parameter all-gathers per layer, discard immediately after use, and reduce-scatter gradients as soon as they are produced (Sections 5.2, 5.3, 7.2.2).

- Partitioned activation checkpointing across MP ranks (new capability)
  - Difference from prior MP: Traditional MP replicates activations; ZeRO-R partitions them and reassembles on demand (Section 6.1).
  - Impact: Reduces activation memory by the MP degree and, if needed, supports CPU offload (`Pa+cpu`). This is crucial when model-state memory is tamed by ZeRO-DP and activations become the next bottleneck (Sections 4.2.1, 6.1).

- Memory fragmentation control and bounded temporaries (practical innovations)
  - `MD` reduces OOMs and allocator overhead by segregating long/short-lived tensors (Section 6.3).
  - `CB` prevents temporary buffers from scaling with model size while still achieving high throughput (Section 6.2).

Collectively, these are not just incremental tweaks; they re-architect DP to have the memory efficiency of MP without sacrificing DP’s communication efficiency and ease of use (Sections 4–5).

## 5. Experimental Analysis
Evaluation setup (Section 10.1; Tables 5–10)
- Hardware: 400× NVIDIA V100 32GB GPUs (25 DGX-2 nodes), with 800 Gbps internode bandwidth.
- Models: GPT-2–style transformers with varying layers/hidden sizes to reach 1.5B–170B parameters (Table 4 and detailed configs in Tables 5–10).
- Baselines:
  - Without MP: PyTorch DistributedDataParallel (DDP).
  - With MP: Megatron-LM (open-source as of Sept 2019).
- ZeRO configuration: “ZeRO-100B” = `Pos+g` + ZeRO-R (`Pa`, `CB`, `MD`), with or without MP as specified (Table 3).

Headline results (Sections 1, 10.2–10.6; Figures 2–8)
- Throughput and scale
  > Figure 2: ZeRO-100B sustains ≈15 PFLOPS aggregate and ~38 TFLOPS/GPU on 400 GPUs for 8B–100B models, up to 10× faster than the Megatron-LM baseline at large scales.  
  > Figure 3: Super-linear speedup for a 60B model from 64→400 GPUs—performance more than doubles when doubling GPUs in the 64–400 range.

- Model size enabled
  > Figure 2 and Table 5: ZeRO-100B efficiently trains up to 170B parameters on 400 GPUs; the Megatron baseline degrades beyond ~40B as inter-node MP becomes communication-bound.  
  > Figure 4 and Table 10: Without any MP, ZeRO-100B trains models up to 13B parameters on 128 GPUs; DDP runs out of memory around 1.4B.

- Memory scaling (analysis corroborated by measurement)
  > Table 1: With `Nd=64`, a 7.5B model’s model-state memory drops from 120 GB (DP) to 31.4 GB (`Pos`), 16.6 GB (`Pos+g`), and 1.88 GB (`Pos+g+p`).  
  > Table 2: The “measured max model size” with `Pos` matches the “max theoretical” values, supporting the accuracy of the memory analysis (e.g., MP=16, GPUs=1024, `Pos` supports ~121.6B parameters; measured: 100B due to practical overheads).

- Activation memory and throughput trade-offs
  > Figure 6: Enabling `Pa` increases max trainable model size (e.g., 40B→60B) by reducing activation memory; adding `Pos+g` further boosts to ~140B.  
  > Figure 7: Max cached memory per iteration falls as more ZeRO-R/D-P components are enabled; `Pa+cpu` shows the biggest drop for very large models (e.g., 100B), while effects are smaller for 40B.  
  > Figure 8: `Pa+cpu` can reduce throughput when PCIe/NVLink transfers dominate (e.g., 60B), but it is necessary to run the largest configurations (e.g., 170B) without OOM.

- Real model training
  > Section 10.6; Figure 5: Turing-NLG (17B parameters) trained end-to-end with ZeRO-100B achieves WebText-103 perplexity of 10.21 and sustains 41.4 TFLOPS/GPU.

Do the experiments support the claims?
- Efficient scaling to 100B+: Yes. The direct throughput measurements (Figure 2) and the super-linear scaling (Figure 3) convincingly demonstrate practical efficiency for 8B–100B models, with working runs up to 170B.
- Memory reduction claims: Theoretical reductions are validated both by measured max model size (Table 2) and by the ability to run much larger models/batch sizes (Figures 6–7).
- Communication overhead: While Stage 3 (`Pp`) is analyzed but not implemented in ZeRO-100B, the measured performance with `Pos+g` plus ZeRO-R is consistent with the “no extra communication over DP” property of `Pos+g` (Section 7.2.1).
- Ablations/robustness: Figures 6–8 isolate the effects of `Pa`, `Pa+cpu`, and `Pos+g`, showing when each helps and when offloading hurts throughput. The baseline comparisons use per-GPU throughput, with a note that baseline sometimes used fewer GPUs (Table 10 notes), which slightly favors the baseline; ZeRO still outperforms.

## 6. Limitations and Trade-offs
- Stage 3 (`Pp`) communication overhead and implementation scope
  - `Pp` raises per-step communication from `2Ψ` to `3Ψ` (Section 7.2.2). ZeRO-100B did not include `Pp`, so trillion-parameter results are analytical projections (Sections 1, 9), not measured.

- Dependence on high arithmetic intensity and transformer-like workloads
  - `Pa` assumes compute/memory ratios high enough to hide activation all-gather costs (Sections 4.2.1, 8). Models with lower arithmetic intensity or very short sequence lengths could see different trade-offs.

- Offloading trade-offs
  - `Pa+cpu` reduces activation memory nearly to zero but can reduce throughput due to host-device bandwidth limits; it is beneficial only when GPU memory is otherwise the bottleneck (Figure 8).

- Network characteristics matter
  - Although `Pos+g` matches DP communication volume, absolute performance still depends on interconnect quality; the baseline MP’s poor scaling across nodes (Figure 2) illustrates how sensitive large-model training is to bandwidth/latency. ZeRO reduces inter-node pressure but does not eliminate dependence on a decent fabric.

- Generality and evaluation scope
  - Experiments focus on GPT-like transformers. The methods are generic, but performance/memory benefits may differ for architectures with very different activation patterns or operator sets.

- Compute-power gap to 1T end-to-end training
  > Section 9 estimates that even with ZeRO’s memory/communication efficiency, training a 1T model end-to-end on a 1024-GPU cluster could take many months; exaFLOP-scale compute would be needed for practical wall-clock times.

- Engineering complexity
  - Sharded optimizers, reduce-scatter scheduling, and dynamic parameter broadcasting increase system complexity compared to plain DP. While the API is kept simple (Section 10.1), the runtime is sophisticated.

## 7. Implications and Future Directions
- Field impact
  - By bringing DP’s efficiency to the memory footprint of MP, ZeRO removes the principal system barrier to training extremely large models. This changes the scaling “rulebook”: model size can grow roughly linearly with the number of GPUs (Figure 1; Table 1), without resorting to fine-grained cross-node MP.
  - Ease of use matters: training up to 13B parameters without MP (Figure 4) democratizes large-model experimentation to teams without specialized MP expertise or premium intra-node fabrics.

- What this enables next
  - Trillion-parameter exploration: With `Pos+g+p`, the analysis shows that a 1T model’s model-state memory (≈16 TB for Adam in 16-bit training; Section 1) can be distributed across ~1024 GPUs so that each GPU stores ≈16 GB—well within device limits.
  - Better optimizers: Since ZeRO amortizes optimizer-state memory, richer adaptive optimizers with larger state (Section 2.3) become practical at scale.
  - New 3D parallel strategies: Combining ZeRO-DP with in-node MP (and optional pipeline parallelism where beneficial) opens flexible hybrid strategies tuned to specific clusters (Section 1, “ZeRO and MP”).
  - Activation-aware training: Partitioned checkpointing and selective offloading suggest further work on compiler/runtime scheduling that co-optimizes compute, memory, and communication.

- Practical applications
  - Training frontier language models (e.g., Turing-NLG 17B; Section 10.6), large multi-modal transformers, and very deep vision backbones.
  - Scaling studies: Researchers can probe scaling laws and data regimes without being blocked by DP memory duplication.

In summary, ZeRO rethinks data-parallel training as a memory-sharded, temporally scheduled system that preserves DP’s communication efficiency. The measured results (Figures 2–4, 6–8; Tables 1–2, 5–10) show strong speed and scale benefits up to 170B parameters on current hardware, and the analysis (Sections 5–8) credibly supports trillion-parameter feasibility from a memory/communication standpoint. The remaining obstacle to 1T training is compute budget, not memory capacity.
