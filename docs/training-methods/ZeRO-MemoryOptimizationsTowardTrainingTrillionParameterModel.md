# ZeRO: Memory Optimizations Toward Training Trillion Parameter Models

**ArXiv:** [1910.02054](https://arxiv.org/abs/1910.02054)

## 🎯 Pitch

ZeRO (Zero Redundancy Optimizer) introduces a breakthrough approach to large model training by eliminating redundant memory across distributed GPU clusters, enabling deep neural networks with up to hundreds of billions—and potentially trillions—of parameters to be trained efficiently on commodity hardware. By partitioning optimizer states, gradients, parameters, and activation checkpoints, ZeRO dramatically increases model size and training speed while maintaining usability and scalability, directly addressing the memory bottleneck that limits today’s state-of-the-art deep learning systems. This innovation paves the way for democratizing training of ever-larger models, unlocking new frontiers in AI research and application.

---

## 1. Executive Summary
This paper introduces ZeRO (Zero Redundancy Optimizer), a set of system-level techniques that eliminate redundant memory in distributed training so very large neural networks can fit and run efficiently on commodity GPU clusters. By partitioning optimizer states, gradients, parameters, and activation checkpoints across devices—and carefully scheduling the required communication—ZeRO enables training models up to hundreds of billions of parameters today, and lays out a path to trillion-parameter training with current-generation hardware.

## 2. Context and Motivation
- The problem: Training ever-larger models (billions to trillions of parameters) quickly runs out of device memory; naïvely adding more GPUs with standard data parallelism does not reduce per-device memory and thus fails to scale. Section 1 shows that with 32 GB GPUs, basic data parallelism runs out of memory beyond ~1.4B parameters.
- Why it matters:
  - Practical: Larger models have delivered step-change accuracy improvements in NLP and beyond (Sec. 1 cites BERT-large 0.3B, GPT-2 1.5B, Megatron 8.3B, T5 11B).
  - Systems: Memory, not only compute, has become the bottleneck; without solving it, training speed and model size hit a wall.
- Where prior approaches fall short:
  - Data parallelism (DP): Replicates all model states on each GPU, so memory scales with model size, not with number of devices. Efficient communication/computation, poor memory efficiency.
  - Model parallelism (MP): Splits layers across devices, which saves memory but incurs fine-grained communication every layer. Works within a node (fast NVLink/NVSwitch), degrades across nodes (slower Infiniband). The paper reports <5% of peak on a 40B model across two DGX-2 nodes (Sec. 1).
  - Pipeline parallelism (PP): Splits model by layers with micro-batching, but requires large batches to hide pipeline bubbles (hurts convergence, consumes more activation memory) or uses stale parameters (convergence concerns). See Sec. 2.1 for GPipe vs. PipeDream trade-offs.
  - CPU offload / virtual memory: Moves tensors to host memory, but PCIe bandwidth becomes the bottleneck; up to 50% of time can be transfers (Sec. 2.2.2).
- Positioning: ZeRO aims to retain DP’s high efficiency while matching or surpassing MP’s memory savings, by removing every major source of redundancy in memory across data-parallel workers (Sec. 4). It augments this with activation and buffer management to handle “residual” memory.

Key terms (defined only when uncommon):
- Model states: the training-time tensors tied to parameters: the parameters themselves, their gradients, and optimizer states (e.g., Adam’s running averages of gradients and squared gradients). These dominate memory in large models (Sec. 3.1).
- Residual states: everything else that uses memory during training—activations, temporary buffers, and memory fragmentation overhead (Sec. 3.2).
- Nd: data-parallel degree (number of DP processes/GPUs). Nm: model-parallel degree.
- All-reduce, reduce-scatter, all-gather: standard collective communication operations. For large tensors, time is dominated by total data volume moved (Sec. 7.1).

## 3. Technical Approach
The ZeRO framework has two pillars (Sec. 4): ZeRO-DP to eliminate redundancy in model states, and ZeRO-R to reduce residual memory. Each pillar comprises concrete techniques and a communication schedule that keeps the approach efficient.

A. ZeRO-DP: partition model states, not computation
ZeRO-DP has three stages, applied cumulatively (Sec. 5, Fig. 1):
- Stage 1 — Pos (Optimizer state partitioning):
  - Idea: With Adam in mixed precision, optimizer states consume K=12 bytes per parameter in FP32 (FP32 copy of weights + momentum + variance), while parameters and gradients in FP16 add 2Ψ + 2Ψ bytes (Ψ = number of parameters; Sec. 3.1). Instead of replicating optimizer states on each DP rank, shard them across Nd ranks; each rank owns and updates only its 1/Nd shard.
  - How it works:
    - Partition optimizer states into Nd equal shards; rank i holds shard i and updates only parameters in that shard (Sec. 5.1).
    - After computing all gradients (still with a full parameter replica at this stage), each rank applies updates only to its shard, then an all-gather step distributes the updated full parameter tensor to all ranks for the next step.
  - Memory effect: reduces optimizer state memory from KΨ to KΨ/Nd; model-state memory becomes 4Ψ + KΨ/Nd (Fig. 1; Table 1).
  - Communication: unchanged vs. DP (Sec. 7.2.1) because Pos itself doesn’t change gradient aggregation volume.

- Stage 2 — Pg (Gradient partitioning):
  - Idea: Since each rank updates only its parameter shard, it only needs the gradient for that shard. Do a reduce-scatter (instead of all-reduce) so each rank ends with its shard’s reduced gradients and can discard the rest immediately (Sec. 5.2).
  - How it works:
    - During backprop, as gradients for a bucket of parameters are ready, bucketize by destination shard and perform a reduce-scatter to the owner rank; then free gradient memory for that bucket. The paper uses bucketization to overlap compute and communication (Sec. 5.2).
  - Memory effect: gradient memory drops from 2Ψ to 2Ψ/Nd; combined with Pos, total model-state memory approaches 2Ψ + 14Ψ/Nd (Fig. 1; Table 1).
  - Communication: still the same as DP—reduce-scatter volume Ψ plus one all-gather of updated parameters Ψ, total 2Ψ (Sec. 7.2.1).

- Stage 3 — Pp (Parameter partitioning):
  - Idea: Also shard parameters across DP ranks; a rank stores only its parameters. When other layers need remote parameters for compute, fetch them just-in-time, then discard (Sec. 5.3).
  - How it works:
    - Forward pass: For each layer/partition, the owning rank broadcasts the needed parameter shard to all ranks; after computing that layer, ranks discard the broadcast parameters.
    - Backward pass: Do the same in reverse order (need parameters to compute gradients).
    - End-of-step: Gradient reduce-scatter and local optimizer update on each shard. No need to all-gather full parameters persistently; shards are broadcast on demand next iteration (Sec. 7.2.2).
  - Memory effect: the classic 16Ψ model-state memory (Sec. 3.1) becomes 16Ψ/Nd—linear reduction with the number of GPUs (Fig. 1; Table 1).
  - Communication: increases modestly to 3Ψ per step (1.5× baseline DP’s 2Ψ) due to two parameter all-gathers (one in the forward, one in the backward) plus a gradient reduce-scatter Ψ (Sec. 7.2.2).
  - Why acceptable: The extra 50% communication buys Nd-fold memory reduction without changing compute granularity (DP-level), which keeps per-GPU efficiency high compared to fine-grained MP (Sec. 4.1a).

Concrete numbers (Fig. 1; Table 1):
- For a 7.5B model at Nd=64, model-state memory per GPU:
  - Baseline DP: 120 GB
  - Pos: 31.4 GB
  - Pos+g: 16.6 GB
  - Pos+g+p: 1.96 GB

B. ZeRO-R: reduce “residual” memory (activations, buffers, fragmentation)
- Pa — Partitioned activation checkpointing (Sec. 6.1):
  - Activation checkpointing saves only selected activations during forward and recomputes others in backward to reduce memory at 33% extra compute (Sec. 3.2). In MP, activations are often replicated across GPUs even though model states are partitioned.
  - ZeRO-R partitions checkpointed activations across MP ranks (Nm-way) and all-gathers them on demand right before reuse in recomputation/backward. This removes the replication.
  - Optional Pa+cpu: offload partitioned checkpoints to CPU memory to further reduce GPU memory, relying on high arithmetic intensity (lots of compute per byte moved) to hide PCIe transfers for very large models (Sec. 4.2.1).
  - Communication analysis: In Megatron-LM’s transformer block, MP uses six all-reduces per block with total volume proportional to 12 × (seq_len × hidden_dim). Pa adds just one all-gather of size (seq_len × hidden_dim) per block—often <10% of MP traffic (Sec. 8).

- CB — Constant-size fused buffers (Sec. 6.2):
  - Fusing many small tensors into one large buffer (e.g., for all-reduce) improves bandwidth utilization, but if the fused buffer scales with model size, it can consume tens of GB (example: 3B params → 12 GB FP32 buffer).
  - ZeRO uses a fixed, performance-tuned buffer size once models get large, keeping buffers big enough for throughput but bounded for memory.

- MD — Memory defragmentation (Sec. 6.3):
  - Fragmentation arises from mixing short-lived tensors (recomputed activations, activation gradients) with long-lived ones (checkpoints, parameter gradients).
  - ZeRO pre-allocates contiguous pools for long-lived tensors and copies into them on the fly; this reduces allocator overhead and prevents “out of memory” despite sufficient total free memory.

Design choices and rationale:
- Keep DP’s computational granularity and low communication volume (Sec. 4.1a) while removing redundancy (Sec. 4.1b) and exploiting temporal locality—“not all states are needed all the time” (Sec. 4.1c).
- Partition activations in MP rather than rely solely on recomputation, because replicated activations become the next bottleneck once model states are optimized (Sec. 4.2).

Implementation footprint:
- The paper releases a production implementation in PyTorch called ZeRO-100B (Pos+g from ZeRO-DP plus ZeRO-R) and integrates with Megatron-LM for MP (Sec. 10.1; Table 3 lists configurations C1–C5).

## 4. Key Insights and Innovations
- Zero-redundancy partitioning across DP ranks (fundamental):
  - Instead of replicating model states on every GPU (DP), shard them without sacrificing DP’s efficiency. This is the core innovation; memory per GPU scales roughly as 1/Nd with all three stages (Sec. 5 and Fig. 1).
- Communication-balanced scheduling (fundamental):
  - The parameter shard all-gathers are pipelined with layer execution (broadcast just before use and discard after), keeping the added communication to 1.5× DP while achieving Nd-fold memory reduction (Sec. 7.2.2).
- Treating activations as another redundant state (new capability):
  - MP reduces parameter memory but historically replicates activations. Partitioning checkpoints across MP ranks (Pa) removes this hidden replication and costs only a small fraction of MP’s normal comm volume (<10%; Sec. 8).
- Practical residual-memory engineering (incremental but crucial):
  - Constant-size fused buffers (CB) and on-the-fly defragmentation (MD) prevent “death by buffers/fragmentation” that often sinks deep models before parameters are the limit (Sec. 6.2–6.3).
- Usability: “DP as usual” (practical innovation):
  - ZeRO-100B exposes a DP-like interface; models do not need to be rewritten for MP/PP (Sec. 10.1). This unlocks large-model experimentation without model refactoring.

Why these matter:
- Together, they make the aggregate GPU memory of the cluster available to the model while preserving DP efficiency. With all three stages, a trillion-parameter model’s ~16 TB of model states (with mixed-precision Adam) can be spread across 1024 GPUs → ~16 GB/GPU (Sec. 1; Fig. 1 narrative).

## 5. Experimental Analysis
Evaluation setup (Sec. 10.1):
- Hardware: 400 NVIDIA V100 (32 GB) GPUs across 25 DGX-2 nodes; 800 Gbps inter-node bandwidth.
- Models: GPT-2–like transformers with varying depth/hidden size to reach 1.5B–170B parameters. Tables 4–10 list the exact layer counts, hidden sizes, attention heads, and batch sizes for each figure.
- Baselines:
  - Without MP: PyTorch Distributed Data Parallel (DDP).
  - With MP: Megatron-LM (state of the art MP at the time).
- Implementations compared:
  - ZeRO-100B (Pos+g + ZeRO-R, sometimes with MP).
  - Baseline MP alone for large models; DDP alone for small-to-mid.

Main results:
- Model size and throughput gains (Fig. 2):
  - With MP+ZeRO-100B, they train up to 170B parameters, while Megatron alone degrades beyond 40B and is untenable by 80–170B due to inter-node communication (NVSwitch 300 GB/s vs Infiniband 12.5 GB/s; Sec. 10.2).
  - Throughput: sustained ~15 PFLOPs aggregate; ~38 TFLOPs/GPU on 100B models (Fig. 2).  
    > “ZeRO runs 100B parameter models on a 400 Nvidia V100 GPU cluster with over 38 TFLOPs per GPU, and aggregate performance over 15 petaflops” (Sec. 10.2).
  - Speedup: up to 10× over Megatron baseline at large scales (Fig. 2 bars and triangles).

- Super-linear scaling (Fig. 3):
  - For a 60B model, increasing GPUs from 64 → 400 more than doubles per-GPU performance as GPUs double; total throughput shows super-linear growth (Fig. 3). Reason: as Nd grows, Pos+g frees memory, allowing larger per-GPU batch sizes and higher arithmetic intensity (Sec. 10.3).

- “Democratization” without MP (Fig. 4 and Table 10):
  - With ZeRO-100B and no MP, they train models up to 13B parameters on 128 GPUs at >40 TFLOPs/GPU (Fig. 4; Table 10). Baseline DDP runs out of memory around 1.4B and is <20 TFLOPs/GPU.  
    > “ZeRO-100B can train models with up to 13B parameters without MP... existing systems (e.g., PyTorch DDP) run out of memory with 1.4B” (Sec. 10.4).

- Memory behavior and ablations (Figs. 6–8; Table 3):
  - Configs C1–C5 progressively add Pa and Pos+g and then Pa+cpu (Table 3).
  - Max model size increases from 40B (C1) → 60B (C2, via Pa’s 16× MP activation reduction) → 140B (C4, via Pos+g halving model-state memory) → 150B (C5, via Pa+cpu) at fixed batch settings (Fig. 6).
  - Max cached memory drops across configs; Pa+cpu shows a pronounced drop at 100B but not at 40B because activation memory dominates only at very large scales (Fig. 7 narrative in Sec. 10.5).
  - Throughput per GPU generally increases as memory decreases (larger possible batch sizes), with the caveat that Pa+cpu can hurt performance due to PCIe transfers unless necessary (Fig. 8 and Sec. 10.5).

- Theory vs. practice alignment (Table 2):
  - The “measured max model size” with Pos matches the “max theoretical model size” from memory analysis across MP and GPU counts (e.g., 1 MP, 64 GPUs: 7.6B vs. 6.2B measured; trend matches), validating the memory model is realistic.

- Real model to SOTA (Sec. 10.6; Fig. 5):
  - Turing-NLG (17B) trained end-to-end with ZeRO-100B achieves Webtext-103 perplexity 10.21 and sustains 41.4 TFLOPs/GPU (Fig. 5).

Communication analysis support:
- DP baseline volume: 2Ψ (reduce-scatter + all-gather, Sec. 7.1).
- Pos+g: still 2Ψ (reduce-scatter Ψ + parameter all-gather Ψ at step end, Sec. 7.2.1).
- Pos+g+p: 3Ψ (two parameter all-gathers—forward and backward—plus gradient reduce-scatter, Sec. 7.2.2).
- Pa adds one all-gather per transformer block, typically <10% of Megatron’s MP comm (Sec. 8).

Do the experiments support the claims?
- The throughput and scale results (Figs. 2–4) substantiate the efficiency and size claims for the implemented subset (Pos+g + ZeRO-R). The 1-trillion claim is presented as a memory-feasibility analysis (Table 1; Sec. 1 and Sec. 9) rather than an empirical result; the paper explicitly discusses the compute-time gap (Sec. 9).

## 6. Limitations and Trade-offs
- Stage 3 (Pp) not fully evaluated at scale in the implementation:
  - ZeRO-100B uses Pos+g + ZeRO-R; the full “Nd-fold” parameter sharding (Pp) is analyzed and would add 1.5× communication (Sec. 7.2.2) but is not the default in the 100B-scale experiments (Sec. 10). So the 1T parameter result is a feasibility analysis (Table 1) rather than an end-to-end run.
- Communication vs. memory trade-off:
  - Pp increases communication volume to 3Ψ (1.5× DP; Sec. 7.2.2). On clusters with very weak interconnects, this can bottleneck, especially across nodes.
- CPU offload trade-offs:
  - Pa+cpu reduces activation memory to near zero but adds two PCIe transfers per checkpoint and can reduce throughput unless batch size would otherwise be very small or the model would not run (Sec. 6.1; Sec. 10.5 explains the 60B case where C5 underperforms C4).
- Convergence with large global batch sizes:
  - ZeRO’s super-linear scaling comes from increasing per-GPU batch sizes (Sec. 10.3), but very large total batch sizes can harm convergence beyond a “critical batch size” (footnote to Sec. 2, citing [8]). The method does not solve the algorithmic limits of scaling batch size.
- Assumptions in memory accounting:
  - Memory formulas assume mixed-precision Adam with K=12 extra bytes per parameter (Sec. 3.1). Other optimizers or precisions would change constants.
- Training time at trillion scale:
  - Even with memory solved, compute is the bottleneck. Sec. 9 estimates ~140 days to train a 1T model on a 1024 V100 cluster (keeping sequence length and dataset size constant), likely >1 year with realistic increases; an exaFLOP system is suggested for practical times.
- Scope of evaluations:
  - Most experiments report throughput and capacity; only Turing-NLG reports task-level quality (perplexity). Broader task coverage and convergence studies at 100B+ are not presented here.

## 7. Implications and Future Directions
- Field impact:
  - ZeRO reframes memory scaling: with full ZeRO-DP (Pos+g+p), per-GPU memory for model states scales as 1/Nd (Sec. 5.3), so aggregate GPU memory becomes a single pool the model can inhabit. This removes memory as the primary barrier for scaling model size and shifts the bottleneck to compute and interconnects.
  - It makes DP the default for scaling large models, with MP used selectively (e.g., to reduce activation memory, or when DP-only batch sizes become too large; Sec. 1 “ZeRO and MP”).
- Practical applications:
  - Train >10B parameter models without rewriting models for MP/PP (Fig. 4), enabling more teams to explore large-model regimes (Sec. 10.4).
  - Combine with MP within a node (e.g., Nm=16 on DGX-2) and DP across nodes (Nd=64) to fit trillion-parameter models on ~1024 GPUs with manageable batch sizes (Sec. 1; Table 2).
- System-level research directions:
  - Automating scheduling decisions: When to enable Pp, Pa, Pa+cpu given hardware/network constraints (Sec. 8’s comparisons imply a policy controller).
  - Multi-tier memory: Extending Pa+cpu to NVMe/remote memory tiers with overlap strategies; the arithmetic intensity arguments in Sec. 4.2 suggest feasibility for very large models.
  - Communication optimization: Further reduce the 1.5× overhead in Pp via topology-aware collectives, compression, or kernel fusion.
  - Optimizer design: With memory constraints loosened, explore richer optimizers that were previously impractical (Sec. 2.3), or conversely pair ZeRO with memory-efficient optimizers (Sec. 2.2.3).
- Scientific directions:
  - Large-scale convergence: Systematically chart critical batch sizes and optimization behavior for 100B–1T models, beyond the single 17B case (Sec. 10.6).
  - Activation management: Explore learned checkpointing policies or structure-aware partitioning for models beyond transformers.

Selected, grounded takeaways:
- Memory feasibility at scale:
  > “With all three stages enabled, ZeRO can train a trillion-parameter model on just 1024 GPUs… 16TB divided by 1024 is 16GB” (Sec. 1; Fig. 1 rationale).
- Communication boundedness:
  > “Pos+g incurs the same communication volume as DP; Pp adds at most a 50% increase” (Sec. 7.2.1–7.2.2).
- Activation comm overhead:
  > “Pa adds one all-gather per block, less than one tenth of Megatron-LM MP’s communication” (Sec. 8).
- Real-world speed and scale:
  > “15 PFLOPs aggregate, up to 10× faster than Megatron alone; super-linear scaling for 60B models from 64 to 400 GPUs” (Figs. 2–3; Sec. 10.2–10.3).

In short, ZeRO’s key mechanism—sharding every training-time state across data-parallel workers and communicating it only when needed—transforms how memory scales in distributed training. The experimental ZeRO-100B implementation demonstrates strong practical benefits today, and the full ZeRO-DP analysis outlines a credible path to trillion-parameter training once compute becomes available.
