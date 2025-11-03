# Efficient Long‑context Language Model Training by Core Attention Disaggregation

**ArXiv:** [2510.18121](https://arxiv.org/abs/2510.18121)
**Authors:** Yonghao Zhuang, Junda Chen, Bo Pang, Yi Gu, Yibo Zhu, Yimin Jiang, Ion Stoica, Eric Xing, Hao Zhang
**Institutions:** 

## 1. Executive Summary
The paper introduces `Core Attention Disaggregation (CAD)`, a system design that separates the parameter-free softmax-based attention computation (`softmax(QK^T) V`, called `core attention`, or `CA`) from the rest of a transformer, and schedules CA on a shared pool of “attention servers.” This solves persistent load imbalance in long-context large language model (LLM) training—especially with document packing—by balancing the quadratic CA compute independently from the linear compute and memory of other layers, yielding up to 1.35× end-to-end training throughput improvements while removing stragglers in data and pipeline parallelism (see §6, Figure 10).

## 2. Context and Motivation
- Problem or gap:
  - In transformers, `core attention (CA)` scales quadratically with sequence length `l` (O(l^2)), while most other layers scale linearly (O(l)). When these are colocated on the same devices, document length variance (common in long-context training) makes some batches or pipeline stages much slower—these are “stragglers” that stall the rest (Figure 1; Table 1).
  - `Document packing` concatenates variable-length documents into a fixed-size chunk to improve throughput, but two chunks with the same token count can have very different CA FLOPs. For example, one 4K-token document has ~4× the CA FLOPs of four 1K-token documents (Figure 1 caption and narrative).
- Why it matters:
  - Applications increasingly demand 100K–1M token contexts: long chain-of-thought (CoT) reasoning and repository-level code agents (Introduction, §1). To train LLMs to operate effectively at these lengths, long documents must be present in the training data (often upsampled), which amplifies packing-induced skew.
  - In distributed training:
    - `Data parallelism (DP)`: replicas process different chunks; the slowest replica at the gradient barrier stalls all others (stragglers).
    - `Pipeline parallelism (PP)`: different microbatches traverse stages concurrently; any microbatch with heavier CA makes its stage slow, creating pipeline “bubbles” that idle subsequent stages (Introduction, §1; §2.2).
- Prior approaches and their limitations:
  - Variable-length data chunks: redistribute documents to equalize total compute (FLOPs). This unbalances activation memory (which scales with total tokens), hitting OOM (out-of-memory) constraints and failing to equalize CA compute at long contexts. Figure 4a shows memory divergence increases with DP size; Figure 4b shows idle time fraction grows to 19% at DP=4 and 55% at DP=8 for 512K-token workloads (both from §3.2).
  - `Context parallelism (CP)`: shard sequences across GPUs; `per-document CP` shards each document to balance compute/memory per rank. However, it introduces heavy `all-gather` of `K`/`V` states (the key/value vectors needed to compute attention), whose cost grows with global tokens (Figure 3a shows all-gather latency share rises to ~40% on 32 nodes; §3.2). It also forces the last CP rank to store large aggregated KV for backward, creating growing memory pressure (Figure 3b shows KV memory fraction grows to ~30% at 16 nodes).
  - Combining variable-length packing and per-document CP inherits both downsides and still struggles to avoid PP bubbles (Figure 6; §3.2).
- Positioning:
  - The paper isolates the `core attention` (weightless, stateless softmax-based computation) as the right boundary to disaggregate. By scheduling CA independently on a shared pool of “attention servers,” it balances compute without upsetting memory, and decouples CA from DP and PP so stragglers are eliminated across both (§3.3; Figure 2).

## 3. Technical Approach
- Key concepts and definitions:
  - `Core attention (CA)`: the parameter-free attention computation `softmax(QK^T) V` (with masking), where `Q` is query, `K` is key, `V` is value. Unlike broader “attention” (which includes linear projections and layer norm), CA has no trainable parameters and minimal transient state (IO-aware kernels recompute `softmax` in backward and avoid storing the full attention matrix; §2.1).
  - `CA-task`: a unit of CA work defined by a query shard `q(t)` and its context `KV` shard `kv(t)`. Any document can be split into token-level shards; each shard’s CA can be computed independently given its `Q` and the necessary `K,V` tokens (§4.1).
  - `Attention servers`: GPUs that process CA-tasks. They dynamically batch multiple CA-tasks (possibly from different documents and pipeline stages) into a single high-occupancy kernel (e.g., FlashAttention) without losing efficiency (§4.1).
- Why disaggregate CA:
  - Formalizing compute/memory imbalance:
    - Define per-document compute as `FLOPs(l) = α l^2 + β l` (α from CA, β from other layers), and activation memory as `M(l) = γ l` (since modern kernels avoid storing the quadratic attention matrix P; §3.1).
    - For packed microbatches, balancing both compute and memory across batches requires matching both sum of tokens and sum of token-squared, which is rarely possible in practice (§3.1).
  - CA’s enabling properties:
    - `Statelessness`: no parameters, negligible intermediate state. Scheduling CA reduces to pure compute balancing.
    - `Composability`: modern IO-aware kernels (e.g., FlashAttention v2) maintain high utilization when fusing batches of variable-length shards; throughput depends primarily on total fused tokens, not document origin. Profiling shows high throughput when shard length ≥ 128 tokens (the kernel tile size), with padding penalties below that (Figure 5; §3.3).
- DistCA runtime (Figure 2; §4.1):
  - Flow:
    1. Devices process context-independent layers (linear projections Q/K/V/O, feed-forward networks (FFN), layer norm, position embeddings). These are token-wise and scale ≈ O(l) in both compute and memory (§2.1).
    2. After this phase, each document is split into CA-tasks: `{t_d0_0, t_d0_1, ..., t_d1_0, ...}` (token-level shards).
    3. A central CPU scheduler assigns each CA-task to an attention server and determines how to shard each document to balance compute while minimizing communication.
    4. Each attention server receives the needed `Q` and `KV` tensors (via GPU all-to-all), dynamically re-batches its assigned tasks into one kernel call (e.g., FlashAttention), computes CA outputs, and returns outputs to the originating devices for the next context-independent layers (Figure 2).
  - `In-place attention servers`:
    - Instead of dedicating separate GPUs solely to CA (which underutilizes memory because CA is stateless and FFN dominates memory; Figure 3b), each GPU time-shares between “attention server” and “normal” roles. This keeps both compute and memory utilization high (§4.1).
  - `Ping-pong execution` (Figure 7; §4.1):
    - To hide the communication cost of sending `Q`, `K`, `V`, each microbatch is split into two equal-size “nano-batches” (Ping, Pong). The system interleaves their execution phases so the communication of one overlaps with the computation of the other. It also overlaps intra-node `NVLink` traffic (for tensor parallelism) with inter-node `InfiniBand` transfers (from CA disaggregation).
    - Additionally, it fuses the “post-CA” computation of the previous layer with the “pre-CA” of the current layer (both context-independent), improving overlap (§4.1).
  - `Pipeline parallelism (PP) integration` (Figure 8; §4.1):
    - CA does not involve weights, so CA-tasks from different PP stages are indistinguishable and can be batched together on attention servers.
    - To avoid idle time during role switching (attention server vs context-independent compute), all PP stages perform the same phase within a tick—either all forward or all backward—by deferring selected backward microbatches into the drain-down bubbles without increasing ticks per iteration (Figure 8).
    - During pipeline warm-up and drain-down when some stages are idle, those GPUs serve as attention servers to run CA-tasks.
- Communication-aware scheduling (§4.2, Appendix B):
  - Objective: minimize load imbalance (in FLOPs) across attention servers while minimizing communication volume (bytes of `Q` and `KV` sent).
  - Profiler: benchmarks CA over a grid of query lengths and KV lengths to interpolate execution time and throughput; if in the saturation region, derive time from max measured throughput (§4.2).
  - Scheduling units: `Item` is either a full document or a shard (at multiples of the kernel block size, 128 tokens). Each Item’s CA computation maps to a CA-task (§4.2).
  - Steps:
    1. Compute ideal per-server load `F¯` (sum of Item FLOPs / number of servers). Partition servers into `surplus` (load > `F¯`) and `deficit` (load < `F¯`), sort deficits descending (§4.2).
    2. For each deficit server `d`, select migration from surplus server(s):
       - For candidate Item, compute maximum transferable compute `ΔFmax = min(FItem, Ssource, Ddestination)`.
       - Estimate communication cost `Vcomm` for moving a shard with `ΔFmax` FLOPs (Appendix B gives `Comm(n_q, n_kv) = n_q * size_q + n_kv * size_kv`, with an analytic choice of `n_q` minimizing communication, subject to causal-mask constraints; §B).
       - Rank candidates by efficiency `E = ΔFmax / Vcomm`; migrate the best. If `ΔFmax < FItem`, split Item into two sub-Items and move the `ΔFmax` shard (§4.2).
    3. Terminate when each server’s load is within `ε F¯` (tolerance) or further moves have negligible `E`. This balances compute while controlling communication (§4.2).
  - Communication upper bound (Appendix A):
    - Under InfiniBand bandwidth `B` and per-token context-independent compute time `t`, communication can be fully overlapped with compute if:
      - For `s` shards, `t·l ≥ l · (h_q + h_kv (s+1)/2) / B`, giving `s ≤ 2 (tB - h_q)/h_kv - 1`.
      - With Llama-34B parameters (Table 5), 50 GB/s IB, 50% MFU of H200, the computed bound is `s ≈ 31` shards without incurring communication overhead (§A, Equation (1)).

## 4. Key Insights and Innovations
- Disaggregating exactly `core attention (CA)` is the right boundary:
  - Novelty: prior “attention” bundling includes linear projections and normalization; CAD isolates only `softmax(QK^T) V`, which is stateless and compute-heavy. This separation lets CA be scheduled and balanced independently (Table 1; §2.1; §3.3).
  - Significance: removes DP and PP stragglers by decoupling the quadratic computation from linear compute and memory. Balancing CA no longer forces memory imbalance (contrast Figure 4a/4b and §3.2).
- Token-level partitioning plus kernel composability:
  - Novelty: any document can be sharded at token granularity; shards from different documents and PP stages can be fused into one efficient CA kernel call. The kernel’s throughput depends on aggregate tokens, not origin, provided each shard meets the tile size (128 tokens for FA2; Figure 5; §3.3).
  - Significance: flexible, fine-grained load balancing without sacrificing kernel efficiency, unlike uniform CP splits that penalize short documents or early tokens under causal masking (§3.2).
- In-place attention servers with ping-pong overlap:
  - Novelty: devices time-share roles so CA’s statelessness avoids memory underutilization; ping-pong splits allow overlapping inter-node CA transfers with intra-node tensor-parallel traffic and with compute (Figure 7; §4.1).
  - Significance: communication is effectively hidden (“fully overlapped” in most configurations; Figure 11), enabling throughput gains without additional GPU memory pressure (Figure 3b).
- Communication-aware greedy scheduler:
  - Novelty: balances per-server CA FLOPs subject to communication costs, with a heuristic efficiency score `E = ΔFmax / Vcomm`, analytic communication minimization (Appendix B), and tolerance control (§4.2).
  - Significance: achieves near-perfect compute balance while keeping communication within overlap budgets (Figure 12 shows the trade-off window where latency remains flat as communication drops by ~20–25%).

## 5. Experimental Analysis
- Evaluation setup:
  - Models: LLaMA-3 8B and 34B; key dimensions in Table 2 (e.g., 8B: 32 layers, hidden 4096; 34B: 48 layers, hidden 8192).
  - Hardware: up to 512 H200 GPUs (DGX H200 nodes; 8 GPUs per node; §6.1).
  - Parallelisms: `TP=8` fixed (intra-node), grid search `DP`, `PP`, and `CP` for baselines; DistCA replaces CP with CAD, uses sequential document placement with fixed tokens per device for context-independent layers (§6.1).
  - Datasets:
    - “Pretrain”: synthetic distribution upsampling long documents per common practice (e.g., Fu et al., 2024; §6.1).
    - “ProLong”: public long-context mixture (Gao et al., 2025), higher proportion of long documents than Pretrain (§6.1).
  - Baseline: WLB-LLM’s workload-balanced 4D parallelism (Wang et al., 2025c), combining variable-length packing and per-document CP; reimplemented and swept DP/CP degrees (deferred execution not implemented; §6.1).
  - Config grids:
    - 3D (no PP): Table 3 lists MaxDocLen (128K–512K), batch sizes, #GPUs (64–256).
    - 4D (with PP): Table 4 lists MaxDocLen (up to 512K for 8B, 384K for 34B), batch sizes, #GPUs (8B: 64–256; 34B: 128–512; §6.2).
- Main results:
  - 3D (no PP):
    - DistCA consistently outperforms WLB-LLM with 1.05–1.20× speedup (Figure 9). Quote:
      > DistCA shows 1.07–1.20× speedup on Pretrain, and 1.05–1.12× on ProLong; scaling is more favorable across #GPUs and MaxDocLen (Figure 9; §6.2).
    - Trends:
      - Larger speedups on “Pretrain” than “ProLong” (more short documents, harder for WLB to balance; §6.2).
      - For 34B, speedup increases at higher MaxDocLen (more input diversity exacerbates WLB imbalance; §6.2).
      - For 8B, speedups higher at lower MaxDocLen when total tokens per batch are large; all-gather overhead dominates as CA FLOPs per token shrink, but communication remains fixed (due to total tokens; §6.2).
  - 4D (with PP):
    - DistCA achieves up to 1.35× end-to-end speedups and generally better scaling (Figure 10). Quote:
      > For 8B, DistCA yields 1.15–1.30× on Pretrain and 1.10–1.35× on ProLong; for 34B, up to 1.25× on ProLong and 1.15× on Pretrain (Figure 10; §6.2).
    - Reasons:
      - DistCA balances CA across pipeline stages and uses idle stages during warm-up/drain-down to serve CA (Figure 8; §4.1; §6.2).
      - WLB-LLM hits OOM at high CP or DP, and PP amplifies load imbalance (pipelines bubbles), worsening throughput (§6.2; see also Figure 6 for tension between CP and DP).
  - Communication overlap and system overhead (ablation; Figure 11):
    - Three configurations: DistCA (full system), “Signal” (only 1 byte transfers; isolates pure compute imbalance), and “Single Stream” (no ping-pong; comm on same stream as compute).
    - Quote:
      > DistCA nearly matches “Signal” latency across 8B/34B, 8/16 nodes—showing communication is almost fully overlapped; “Single Stream” incurs 10–17% higher latency due to lack of overlap, except 8B on 8 nodes where compute is too small to hide communication (Figure 11; §6.3).
  - Scheduler tolerance factor (ablation; Figure 12):
    - Trade-off between load imbalance (compute) and communication volume. Quote:
      > For 8B, latency remains flat for tolerance 0–0.20 while communication drops ~20–25%; for 34B, tolerance <0.10 is too strict—comm can’t be fully hidden; too large tolerance raises latency roughly linearly due to imbalance (Figure 12; §6.3).
- Support for claims:
  - Throughput gains up to 1.35× are shown across model sizes, datasets, and scales (Figures 9–10).
  - Communication is effectively hidden via ping-pong, evidenced by near-identical latency to “Signal” (Figure 11).
  - DP/PP stragglers are addressed: decoupling CA from DP and PP scheduling, balanced CA-tasks across attention servers, and synchronized PP phases (Figure 8) collectively remove straggler-induced bubbles. This is further contextualized by failure of variable-length packing and per-document CP at scale (Figures 3, 4, 6).
- Caveats:
  - Baseline implementation did not include WLB’s “deferred execution” mechanism (Algorithm 1 in Wang et al., 2025c), which could reduce imbalance—future work needed for apples-to-apples comparison (§6.1).
  - Memory fragmentation due to variable tensor shapes for CA-tasks caused PyTorch GC overhead in 34B 4D experiments, limiting performance (noted explicitly in §6.2).

## 6. Limitations and Trade-offs
- Assumptions and boundary conditions:
  - Long-context workloads with enough compute per token to overlap communication; if compute is too small relative to network bandwidth, overlap may be insufficient (seen in 8B/8-node case; Figure 11).
  - Modern attention kernels must sustain high MFU on fused, variable-length shards; shards shorter than the kernel tile (128 tokens in FA2) reduce efficiency due to padding (Figure 5; §3.3).
- Design constraints:
  - Scheduler currently restricts each CA-task to a `Q` shard with the full `K,V` context range (head-tail style), reducing flexibility. Allowing partial `KV` contexts per task could further reduce communication but complicates correctness and kernel fusion (§8).
  - Communication model is conservative, sometimes overestimating bytes by ignoring `K,V` already resident on destination devices; may result in non-minimal transfers (§8).
- System overheads:
  - Memory fragmentation and dynamic allocation overhead due to varying CA-task shapes cause CPU-side delays and degrade performance, especially for large models under 4D parallelism (§6.2).
- Scalability and hardware:
  - The Appendix A bound on shard count `s` assumes specific bandwidth and MFU; different interconnects or models may reduce the number of shards that can be overlapped (§A). While larger models increase `t` and can increase `s`, smaller models or slower networks may constrain shardability.
- Scope limitations:
  - CAD replaces context parallelism; it does not aim to optimize tensor parallelism beyond standard intra-node use (TP=8).
  - Evaluation focuses on training throughput; inference scenarios (e.g., KV cache behavior) are not studied here.

## 7. Implications and Future Directions
- Field impact:
  - CAD reframes long-context LLM training as a two-pool computation: stateless, quadratic CA balanced across a shared server pool, and linear, memory-bound context-independent layers handled per model pipeline. This architectural decoupling directly addresses load imbalance and stragglers in DP and PP, a long-standing bottleneck in scaling training to 100K–1M contexts (Introduction; §3).
- Enabled directions:
  - Dedicated attention server pools:
    - While the paper uses in-place time-sharing to preserve memory utilization, a dedicated pool could further lower CA latency, improve fault tolerance, and enable performance isolation for multi-tenant training jobs (§8).
  - Stronger scheduling models:
    - Extend the scheduler to allow CA-tasks with partial `KV` contexts, incorporate reuse-aware communication modeling (credit for already resident KV), and adopt global optimization methods (e.g., multiphase or continuous relaxation) while staying practical for large batches (§8; §4.2).
  - Memory and launch optimization:
    - Static allocation and CUDA Graphs to eliminate fragmentation and GC overhead in 34B 4D settings (§6.2).
  - Integration with model/system co-design:
    - Combine CAD with inference-side disaggregation (e.g., expert disaggregation in MoE settings), or with new attention kernels that optimize for token-level fusion across jobs.
- Practical applications:
  - Training LLMs for long chain-of-thought reasoning and repository-level code tasks (Introduction), where 100K–1M contexts are necessary. CAD’s throughput and scaling enable cost-effective training runs and reduce iteration times at scale.
  - Distributed training pipelines in cloud or on-prem GPU clusters where DP/PP stragglers are common; CAD can improve hardware utilization, reduce idle fractions, and maximize throughput under memory constraints (Figures 4, 9–10).

> Overall, the paper demonstrates that isolating the parameter-free `softmax(QK^T) V` computation and treating it as a composable, stateless service is sufficient to fix the scaling pathologies of long-context training. The DistCA system makes this practical with time-shared attention servers, ping-pong overlap, and a communication-aware scheduler, yielding consistent gains (up to 1.35×) and better scaling while keeping memory balanced.