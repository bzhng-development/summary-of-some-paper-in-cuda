# Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM

**ArXiv:** [2104.04473](https://arxiv.org/abs/2104.04473)

## 🎯 Pitch

This paper introduces PTD-P, a unified scheme that combines pipeline, tensor, and data parallelism to train trillion-parameter language models efficiently across thousands of GPUs. Through a novel interleaved pipeline schedule and specialized communication and computation optimizations, the approach achieves unprecedented throughput—up to 502 petaFLOP/s (52% of peak)—enabling models at the trillion-parameter scale to be trained in a matter of months rather than years. This breakthrough dramatically reduces resource barriers for frontier NLP research and sets a new standard for large-scale model training practicality and efficiency.

---

## 1. Executive Summary
This paper shows how to train very large Transformer language models efficiently on GPU clusters by combining three kinds of parallelism—pipeline, tensor, and data—in a single system called `PTD-P`. Its key advance is a new, memory-efficient, interleaved pipeline schedule plus communication and kernel optimizations that together sustain 502 petaFLOP/s on 3072 NVIDIA A100 GPUs while training a 1-trillion–parameter GPT model (Table 1), achieving 52% of the hardware’s peak throughput and projecting practical end-to-end training times (∼84 days for 450B tokens; §5.1, Eq. (4)).

## 2. Context and Motivation
- Problem addressed
  - Training state-of-the-art language models now requires hundreds of billions to a trillion parameters (Figure 1). Two obstacles dominate (§1):
    - Limited GPU memory cannot hold all parameters and activations.
    - The compute required makes single-GPU training infeasible (e.g., GPT‑3 would take “~288 years” on one V100; §1).
- Why it matters
  - Larger models yield better zero/few-shot performance and power important applications (summarization, dialog, search, code; §1). Making trillion-parameter training practical broadens who can build such models and shortens iteration cycles.
- Prior approaches and their limitations
  - Data parallelism scales by replicating the model and splitting data, but it hits two walls (§1, §3.3.1): small per-GPU batch sizes reduce utilization and increase communication; and the maximum number of devices equals the batch size.
  - Tensor (intra-layer) model parallelism (Megatron-LM) works well within an 8-GPU node but suffers across nodes: required all-reduces cross slower inter-node links and small per-rank matrix multiplies hurt GPU efficiency (§1, §2.3, §3.2).
  - Pipeline (inter-layer) model parallelism reduces per-device memory but either:
    - Uses “all-forward then all-backward” (GPipe) with large activation memory and a pipeline “bubble” that wastes time (§2.2.1, Figure 3), or
    - Uses 1F1B (one forward/one backward per stage) but still has a bubble and frequent pipeline flushes to maintain strict optimizer semantics (§2.2.1).
  - ZeRO-3 shards optimizer state and parameters across data-parallel ranks, but when used without model parallelism it communicates heavily across nodes and slows as GPU count grows (Table 2, Figure 10).
- How this work positions itself
  - The paper composes pipeline, tensor, and data parallelism (`PTD-P`) and introduces a new interleaved 1F1B pipeline schedule that shrinks the pipeline bubble while keeping memory low (§2.2.2, Figure 4). It also introduces a scatter/gather communication trick that exploits multiple NICs per node to make the interleaved schedule feasible at scale (§4.1, Figure 9).

## 3. Technical Approach
At a high level, `PTD-P` splits the model and work three ways:
- `tensor model parallelism` splits each layer’s large matrix multiplications across GPUs in the same node,
- `pipeline model parallelism` splits different groups of layers across nodes and runs microbatches through them in a pipeline,
- `data parallelism` replicates the (already sharded) model across groups of nodes to process more data per iteration.

The paper’s method comprises five parts.

1) How the model is partitioned (tensor vs pipeline)
- Tensor model parallelism (Megatron-style; §2.3, Figure 5)
  - MLP layer: split the first weight matrix `A` by columns so each rank computes `Y_i = GeLU(X A_i)` independently; split the second matrix `B` by rows so each rank produces a shard of the output and then all-reduces to combine. This needs two all-reduces per forward and two per backward across `t` tensor-parallel ranks.
  - Self-attention: split attention heads across ranks; queries/keys/values are computed per-rank and combined by an output projection that is row-partitioned (again requiring all-reduces).
  - Why this design: it minimizes synchronization (the nonlinearity sits between the two MLP matmuls so each rank can apply it locally), and uses all-reduces only where mathematically unavoidable.
- Pipeline model parallelism (§2.2)
  - A global batch is split into `microbatches`. These stream through `p` pipeline stages (each stage is a contiguous block of layers unless interleaving is used; see below).
  - Strict optimizer semantics are preserved by draining (“flushing”) the pipeline at the end of every batch so all microbatches finish before the optimizer step. This avoids weight staleness but introduces idle time—the `pipeline bubble`.
  - The bubble fraction under GPipe’s “all-forward then all-backward” is `(p−1)/m` where `m` is the number of microbatches (Figure 3, §2.2.1). It is efficient only when `m ≫ p`, but that requires storing many activations at once.
  - PipeDream-Flush 1F1B schedule (§2.2.1): warm up, then alternate one forward and one backward per stage (“1F1B”), then drain. Bubble time is the same as GPipe, but the number of in-flight microbatches is limited by `p`, which greatly lowers activation memory.

2) Novel interleaved 1F1B pipeline schedule (§2.2.2, Figure 4)
- Idea: give each physical device `v` smaller “model chunks” (virtual stages) instead of one large contiguous stage. The device alternates between its chunks in a carefully arranged interleaved 1F1B timeline.
- Effect:
  - Each chunk has roughly `1/v` of the original compute, so the effective pipeline bubble is divided by `v`. Formally, if a forward microbatch takes `t_f` and a backward takes `t_b`, the bubble time becomes `((p−1)(t_f + t_b))/v`, corresponding to a bubble fraction `(1/v) · (p−1)/m` (§2.2.2).
  - Constraint: the number of microbatches in a batch must be a multiple of the pipeline degree (`m` multiple of `p`) to align interleaving (Figure 4).
  - Trade-off: communication (activation tensors between pipeline stages) also increases by `v` because each device participates in `v` virtual stages (§2.2.2). The paper mitigates this with scatter/gather below.

3) Communication optimization: scatter/gather across nodes (§4.1, Figure 9)
- Observation: with tensor parallelism, the output activations of a stage are replicated across the `t` tensor-parallel ranks. Without care, the same full tensor would be sent `t` times across nodes to the next pipeline stage.
- Optimization: split the activation tensor into `t` chunks, send one chunk per NIC/rank across nodes (leveraging the 8 NICs in a DGX A100 node), then NVLink all-gather the chunks at the receiver to reconstruct the full activation. This reduces cross-node traffic by a factor of `t` and better exploits intra-node bandwidth.
  - Quantitatively, per-microbatch cross-node payload per adjacent stage pair drops from `b·s·h` to `(b·s·h)/t` where `b` is microbatch size, `s` is sequence length, and `h` is hidden size (§4.1).

4) Choosing the degrees of parallelism and microbatch size (§3)
- Notation (§3.1): total GPUs `n = p·t·d`, global batch `B`, microbatch `b`, microbatches per pipeline `m = (B/d)/b`.
- Heuristics derived from analysis and experiments:
  - Use tensor parallelism within a node and pipeline parallelism across nodes (§3.2 “Takeaway #1”): tensor-parallel all-reduces are fast on NVLink/NVSwitch but costly across nodes; pipeline’s point-to-point is cheaper across nodes.
  - Use as much model parallelism `M = p·t` as needed to fit the model; scale further with data parallelism `d` (§3.3.2 “Takeaway #2”).
  - Microbatch size trade-off (§3.4): larger `b` improves math efficiency of kernels but reduces `m = (B/d)/b`, enlarging the bubble. Ignoring communication, iteration time behaves like `(B/(d·b) + p − 1) · (t_f(b) + t_b(b))` (Eq. (1)), so the optimal `b` depends on the model and `p,d`. Figure 8 shows an example where `b=4` is best.
  - Interleaving requires `m` to be a multiple of `p` (§2.2.2).
- Activation recomputation (§3.5): to cut memory, store only stage inputs and recompute inner activations during backward. It increases compute but is often necessary to fit large batch sizes (Figure 17).

5) System engineering to stay compute-bound (§4.2)
- Data layout changed from `[batch, seq, heads, hidden]` to `[seq, batch, heads, hidden]` to unlock strided batched GEMM and avoid transposes.
- Fused elementwise kernels via PyTorch JIT: bias+GeLU and bias+dropout+add.
- Custom fused scale-mask-softmax kernels (general and causal) reduce memory traffic.
- Communication via NCCL on a fat-tree network; training code is an extension of Megatron-LM (§4).

Auxiliary formulas for accounting and planning
- Parameter count (Eq. (2)): an analytic approximation to map `layers l`, `hidden h`, `vocab V`, and `sequence s` to total parameters `P`.
- FLOPs per iteration (Eq. (3)): counts forward+backward (including recomputation). Used to report achieved teraFLOP/s and to estimate training time.
- Training time estimator (Eq. (4)): for tokens `T`, parameters `P`, GPU count `n`, and measured per-GPU throughput `X`, time ≈ `(8·T·P)/(n·X)`. This holds under typical regimes where terms in Eq. (3) simplify (§5.1).

## 4. Key Insights and Innovations
- Interleaved 1F1B pipeline schedule that reduces idle time without growing activation memory (§2.2.2, Figure 4)
  - What’s new: previous schedules either saved memory (1F1B) but kept the bubble, or reduced the bubble by increasing batch/memory (GPipe). Interleaving splits each physical stage into multiple virtual stages, cutting the bubble by a factor `v` while keeping only `O(p)` in-flight microbatches in memory. The cost—`v`× more pipeline communication—is offset by the scatter/gather optimization.
  - Why it matters: in practice it raises throughput by “as much as 10%” over non-interleaved 1F1B at the same memory footprint (§Abstract, §2.2.2; Figure 12 shows the uplift is largest at smaller batch sizes).
- Scatter/gather cross-node activation transfer (§4.1, Figure 9)
  - What’s new: a topology-aware way to exploit tensor-parallel replication so that cross-node communication uses all NICs with disjoint payloads, followed by a fast intra-node all-gather on NVLink/NVSwitch.
  - Why it matters: makes the extra communication of interleaving affordable at scale; Figure 18 shows up to 11% throughput improvement on the 175B GPT model with interleaving.
- A principled recipe for composing parallelism types (`PTD-P`) with simple rules of thumb (§3 “Takeaways”)
  - What’s new: a clear, validated division of labor—tensor parallelism within nodes, pipeline across nodes, data parallelism for scale-out—and an analysis that explains when each helps or hurts (Figures 13–15).
  - Why it matters: naive combinations can halve throughput (§Abstract). The paper’s guidance yields near-linear scaling to 3072 GPUs (Table 1).
- Kernel and layout fusion to keep training compute-bound (§4.2, §5.8)
  - What’s new: specific fused kernels (bias+GeLU; bias+dropout+add; fused softmax) and a layout choice that eliminate expensive transposes.
  - Why it matters: substantial gains—“+19%” throughput for the 175B model and “+11%” for the 530B model (§5.8)—that accumulate with the parallelism innovations.

## 5. Experimental Analysis
- Setup (§5): 
  - Cluster: NVIDIA Selene with DGX A100 nodes (8×80GB A100 per node), NVLink/NVSwitch intra-node, 200 Gbps HDR Infiniband inter-node, fat-tree network.
  - Models: GPT-family configurations with varying layers/hidden/heads; all use sequence length `s=2048` and vocabulary `V=51,200`. Parameter counts computed by Eq. (2).
  - Metrics: per-GPU teraFLOP/s (end-to-end, including data loading and optimizer), aggregate petaFLOP/s. FLOPs computed with Eq. (3).
- Main scaling result (Table 1; §5.1)
  - Weak scaling from 1.7B to 1008B parameters with increasing GPU counts and batch sizes.
  - Quote: “achieved end-to-end training throughput of 163 teraFLOP/s per GPU (52% of theoretical peak) and an aggregate throughput of 502 petaFLOP/s on a GPT model with a trillion parameters using 3072 GPUs.”
  - Throughput improves as models grow (larger GEMMs increase GPU utilization) while communication is contained via `PTD-P`.
  - Training-time estimates (Eq. (4)): 
    - Quote: “175B parameters, 300B tokens, 1024 A100s → 34 days” (§5.1).
    - Quote: “1T parameters, 450B tokens, 3072 A100s → 84 days” (§5.1).
- Comparison to ZeRO‑3 without model parallelism (Table 2, Figure 10; §5.2)
  - For GPT‑3‑175B with fixed global batch 1536:
    - At 384 GPUs, ZeRO‑3 achieves 144 TFLOP/s/GPU vs `PTD-P` 153 (6% higher).
    - At 1536 GPUs, ZeRO‑3 drops to 44 TFLOP/s/GPU while `PTD-P` maintains 141 (3.2× higher).
  - For 530B: ZeRO‑3 could not fit on 560 GPUs at microbatch 4; at 640 GPUs it reaches 138 TFLOP/s/GPU; `PTD-P` delivers 159–171 TFLOP/s/GPU at comparable scales (Table 2). 
  - Quote: “By doubling the number of GPUs (keeping the batch size the same), `PTD-P` outperforms ZeRO‑3 by 70% for both models due to less cross-node communication.” (§5.2; see Figure 10 trend).
  - Caveat: the ZeRO baseline here does not combine with model parallelism; the paper notes it could be combined.
- Pipeline parallelism studies (§5.3)
  - Weak scaling with more pipeline stages (Figure 11): at small batch sizes, throughput drops as `p` increases due to the bubble `(p−1)/m`; at larger batch sizes, scaling improves as the bubble is amortized.
  - Interleaved vs non-interleaved (Figure 12): interleaving yields higher throughput at small–moderate batch sizes; the advantage shrinks as batch grows because (a) the bubble shrinks for the default schedule, and (b) interleaving’s extra communication scales with batch.
- Interactions among parallelism types and microbatch (§5.4–§5.5)
  - Tensor vs pipeline (Figure 13): with 64 GPUs, best performance occurs when tensor-parallel size equals the number of GPUs per node (8) and the remaining factor goes to pipeline (the `(t,p)=(8,8)` point). Excess tensor parallelism across nodes hurts due to cross-node all-reduces.
  - Pipeline vs data (Figure 14): for a fixed batch, increasing pipeline depth reduces throughput because `(p−1)/m` grows. Pipeline should be used only as needed to fit the model; data parallelism should provide additional scale.
  - Tensor vs data (Figure 15): for large batches and microbatch 1, data-parallel all-reduces are infrequent, while tensor parallelism requires all-reduces every microbatch and fragments GEMMs—leading to worse performance as tensor size grows.
  - Microbatch choice (Figure 16): for a 91B model with `(t,p)=(8,8)`, the best microbatch is 2; larger `b` improved math efficiency but increased the bubble; the optimum is model- and setup-dependent. The simple timing model (Eq. (1)) helps select `b`.
- Memory and communication ablations
  - Activation recomputation (Figure 17): at small batches it can reduce throughput “by up to 33%” due to extra forward compute, but it enables larger batches where throughput is “up to 2× higher” than the best no-recompute point because the bubble shrinks.
  - Scatter/gather (Figure 18): with interleaving on the 175B model, this optimization yields “up to 11%” higher throughput.
  - Fused operators (§5.8): +19% for 175B; +11% for 530B.
  - Effective bisection bandwidths achieved at 1T scale: 892 GB/s for point-to-point pipeline traffic and 12.9 TB/s for data-parallel all-reduce (§5.9), indicating careful partitioning kept communication within the network’s capacity.
- Checkpointing throughput (§5.10): trillion-parameter checkpoints are 13.8 TB; parallel load reached the filesystem’s 1 TB/s peak read bandwidth; writes reached 273 GB/s (40% of peak).

Assessment: The experiments are broad (scaling to 3072 GPUs), include head-to-head baselines (ZeRO‑3), and provide ablations (schedules, microbatch, recomputation, comms optimizations). Together they substantiate the claim that the `PTD-P` composition plus interleaving and engineering delivers state-of-the-art throughput for trillion-parameter training.

## 6. Limitations and Trade-offs
- Dependence on high-performance hardware and topology
  - The design assumes fast intra-node links (NVLink/NVSwitch) and multiple high-bandwidth NICs per node to make tensor-parallel all-reduces and scatter/gather efficient (§4.1, §5.9). On clusters with weaker interconnects, the interleaved schedule’s extra communication and tensor-parallel all-reduces may dominate.
- Interleaving constraints and overheads
  - Interleaving requires the number of microbatches per batch `m` to be a multiple of the pipeline degree `p` (§2.2.2). It also increases pipeline communication by a factor `v` (the number of virtual stages per device), which must be mitigated by scatter/gather; without it, interleaving can lose to the default schedule at large batches (§5.3.2).
- Strict semantics with pipeline flushes limit overlap
  - The method maintains strict optimizer semantics by flushing every batch (§2.2). Alternatives (asynchronous or bounded-staleness) could further increase utilization but are not explored here.
- Manual configuration; no automatic search
  - Although the paper provides heuristics, it does not automatically explore `(p,t,d)`, microbatch size, or activation checkpointing strategies (§1, §3). Suboptimal choices can be 2× slower (§Abstract, §3.2).
- Compute–memory trade-offs
  - Activation recomputation can hurt small-batch throughput by up to 33% (Figure 17). The best microbatch size is model- and pipeline-dependent (Figure 16), so tuning is required.
- Scope of model architectures and tasks
  - The pipeline partitioning discussion assumes symmetric Transformer stacks (same block repeated; §2.2). More heterogeneous architectures require different partitioning strategies not treated here.
- Baseline scope
  - The ZeRO‑3 comparison does not include hybrid ZeRO + model parallelism, which could perform better than ZeRO‑3 alone (§5.2).

## 7. Implications and Future Directions
- How this changes the field
  - It demonstrates that trillion-parameter dense Transformer training with strict optimizer semantics is practical on today’s GPU clusters when parallelism is composed thoughtfully. The concrete recipe—tensor parallelism within nodes, pipeline across nodes with interleaving, and data parallelism for scale-out—has already influenced large-scale training stacks.
- Follow-up research enabled/suggested
  - Automated parallelism planners that jointly search `(p,t,d)`, microbatch size, and checkpoint placement using the cost models from §3 and empirical feedback.
  - Adaptive or elastic interleaving that adjusts `v` based on network load and batch size; better overlap of communication and computation for the interleaved schedule.
  - Hybridization with ZeRO (sharded optimizer/parameters) and activation partitioning to further reduce memory, enabling even deeper pipelines or longer sequences.
  - Exploring relaxed pipeline semantics (e.g., PipeDream‑2BW, PipeMare; §6 Related Work) with convergence guarantees, potentially improving utilization beyond strict flushes.
  - Extending the approach to other architectures (e.g., encoder–decoder, multi-task stacks) and to sparsely-activated models (Mixture-of-Experts) where the communication/computation balance differs.
- Practical applications
  - Faster pretraining of general-purpose LLMs (search, dialog, code), domain-specific LMs (biomed, legal), and very long-context models by reinvesting saved time into longer sequences or more tokens.
  - Cost-effective scaling in cloud and on-prem clusters: the scatter/gather trick shows how to exploit multi-NIC nodes; the training-time estimator (Eq. (4)) helps plan cluster size and schedule.

> Bottom line: Section §5 and Table 1 show that with the interleaved pipeline schedule, scatter/gather communications, and fused compute kernels, `PTD-P` sustains 163 TFLOP/s per A100 (52% of peak) at 3072 GPUs, reaching 502 PFLOP/s aggregate, and—using Eq. (4)—reduces the time-to-train a trillion-parameter GPT model to a few months without relaxing optimizer semantics.
