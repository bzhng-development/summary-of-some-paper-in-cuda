# MegaScale-Infer: Serving Mixture-of-Experts at Scale with Disaggregated Expert Parallelism

**ArXiv:** [2504.02263](https://arxiv.org/abs/2504.02263)

## 🎯 Pitch

MegaScale-Infer introduces a novel system architecture for serving massive Mixture-of-Experts (MoE) large language models by decoupling attention and expert modules within each model layer, allowing them to be independently scaled and deployed on heterogeneous hardware. Through a tailored ping-pong pipeline parallelism and a custom many-to-many (M2N) communication library, the system dramatically boosts GPU utilization and slashes inference costs—achieving up to 1.9× higher throughput and 1.5–2.0× lower serving cost compared to state-of-the-art systems. This innovation addresses a key bottleneck in practical MoE deployment, unlocking significant efficiency gains vital for large-scale, cost-sensitive AI services.

---

## 1. Executive Summary
MegaScale-Infer is a system for serving Mixture-of-Experts (MoE) large language models that splits each model layer into two independently scaled parts—`attention` and `experts` (the FFN sub-networks)—and connects them with a high-performance many-to-many (M2N) network layer. It adds a “ping-pong” micro-batch pipeline that overlaps computation with communication and introduces a specialized M2N communication library to make token routing fast and stable. Across 132B–317B MoE models, MegaScale-Infer delivers up to 1.9× higher per-GPU decoding throughput than state-of-the-art serving systems and up to 1.86× higher throughput per unit cost under heterogeneous GPUs (Figures 8(a) and 9(a), Abstract).

## 2. Context and Motivation
- Problem addressed
  - During inference, MoE sparsity routes each token to a small subset of experts (top-k). This reduces per-expert batch sizes during decoding, making the expert FFNs memory-bound and under-utilized on GPUs despite MoE’s compute savings (Section 2.3; Figure 1(b)).
  - Attention during decoding is inherently memory-bound (due to key–value cache lookups over all past tokens), while FFNs are compute-efficient only when batch size is large enough to amortize weight loads (Section 2.1; Figure 1(a)).
  - With realistic latency targets and KV-cache memory limits, the total batch size cannot be made arbitrarily large, so per-expert batches shrink as the number of experts grows—further depressing utilization (Section 2.3; Roofline analysis and the Mixtral 8x22B example).

- Why it matters
  - Low GPU utilization inflates inference costs for large-scale MoE services. The paper targets cost and energy efficiency at production scale (nearly 10,000 GPUs; Section 8), with demonstrated 1.5–2.0× cost reductions in deployment.

- Limitations of prior approaches
  - Traditional model parallelism (tensor parallelism and pipeline parallelism) does not address MoE-induced per-expert batch fragmentation and can add communication overhead (Section 2.2).
  - Expert parallelism improves expert GEMM shapes but requires expensive all-to-all token dispatch per MoE layer (Figure 2(b)).
  - Disaggregation for long-context dense models (e.g., Infinite-LLM) replicates attention to grow batch size, but it does not tackle MoE’s dynamic token routing and top-k sparsity, and assumes simpler communication (Section 2.4).

- Positioning
  - MegaScale-Infer goes beyond prefill/decoding disaggregation used in prior systems by disaggregating within the layer: attention on one set of GPUs and experts on another (Figure 3). It then:
    - Replicates attention (data parallelism) to aggregate requests.
    - Scales experts (expert parallelism) to keep expert GEMMs compute-bound.
    - Introduces a ping-pong micro-batch pipeline and a purpose-built M2N library to hide and stabilize token routing costs (Sections 3–5).

## 3. Technical Approach
The system’s core is “disaggregated expert parallelism”: separate attention and experts—each with its own parallelism strategy and hardware—then add a pipeline and communication layer that make this split efficient (Figure 3).

- Architecture and parallelism choices
  - Attention nodes
    - Replicate attention parameters and store the KV cache (data parallelism). Use tensor parallelism inside a node to exploit high intra-node bandwidth (NVLink) (Section 3).
    - Rationale: decoding attention is memory-bound and benefits from large memory capacity/bandwidth and cheap replication.
  - Expert nodes
    - Each node hosts parameters for one expert; all expert nodes form an expert-parallel group (Figure 3). Use tensor parallelism inside a node (Section 3).
    - Rationale: experts should get as many aggregated tokens as possible, making their GEMMs compute-bound and efficient.

- Ping-pong micro-batch pipeline (Figure 4; Section 4.1)
  - Problem: After disaggregation, attention and experts would be idle while waiting for each other or for network transfers.
  - Solution: Split the global batch into m micro-batches and shuttle them in a ping-pong fashion through attention → experts → attention for each MoE layer, twice per layer (A2E and E2A).
  - Key conditions to keep GPUs busy and hide communication (Equations (1)–(3)):
    - Balance compute: `Ta ≈ Te` (attention vs expert time per micro-batch).
    - Communication faster than compute: `Tc < Tf`, where `Tf = max(Ta, Te)`.
    - Enough micro-batches to fill the pipeline and cover two communications per layer:
      - `m × Tf ≥ 2 × (Tf + Tc)` ⇒ `m ≥ 2 × (1 + Tc/Tf)`.
      - With fast links (`Tc < 0.5 Tf`), m ≥ 3 is sufficient; slower links need m ≥ 4 (Section 4.1).
  - Latency model (Equations (4)–(5)):
    - Per-micro-batch iteration latency bounded by `(Ta + Te + 2Tc) + m Tf (L − 1) ≤ Titer ≤ m Tf L`.
    - Total iteration latency for the global batch: `Ttotal = (Ta + Te + 2Tc) + Tf (mL − 1)`.

- Deployment plan search with a performance model (Algorithm 1; Sections 4.1–4.2)
  - Search space: tensor parallel sizes `tpa` (attention) and `tpe` (experts), number of attention nodes `na`, number of micro-batches `m`, and global batch size `B` (subject to a latency SLO).
  - Modeling compute times using GEMM arithmetic intensity and measured constants (Table 2; Section 4.2):
    - Attention time per micro-batch `Ta ≈ k1 ba + k2`, with `ba` the per-attention micro-batch size; includes memory-bound KV-cache reads proportional to `ba × s` and TP sync overhead.
    - Expert time per micro-batch `Te ≈ k3 be + k4`, with `be` the per-expert micro-batch size.
    - Relationship: `ba × m × na = be × m × E/K = B` (total tokens conserved), so we set `na ≈ (k1 E)/(k3 K)` to balance `Ta` and `Te` (Constraint 1).
  - Modeling communication (Equation (6)):
    - For A2E and E2A, time is the slower of send and receive: 
      - `Tc = max{ (ba h K / tpa) / (Wa × Util(ba h K / tpa)), (be h / tpe) / (We × Util(be h / tpe)) }`,
      - where `Wa, We` are per-GPU link bandwidths and `Util(msg_size)` is the measured bandwidth utilization curve vs. message size.
  - Constraints:
    - Pipeline constraints (Equations (1)–(3)).
    - SLO on time-between-tokens: `Titer ≤ SLO` (Equation (7)); SLO is set to 150 ms in evaluations (Section 7.1).
    - GPU memory capacity for attention: `4 m ba s h L / g + 2 Pa < tpa Ca` (Equation (8); GQA groups `g`; bfloat16; Pa is attention parameter size; Ca per-GPU memory).
  - Objective: maximize throughput per unit cost = `(B / Ttotal) / (tpa na Costa + tpe E Coste)` by simulating plans and picking the best (Algorithm 1; Section 4.2).
  - Practical bounds: `Nm` (max micro-batches) is 4—too many micro-batches degrade expert GEMM efficiency—and GPU-per-node choices are typically {1,2,4,8}, keeping search tractable (Algorithm 1 commentary).

- High-performance M2N communication library (Section 5; Figures 6–7; Figure 5)
  - Why NCCL struggles for MoE-style M2N token dispatch:
    - Extra GPU→CPU proxy copies (issue #852), group op batching limits (max 8 ops), and general per-group setup overhead inflate latency; tail latency spikes with more receivers (Figure 5).
    - GPU-side synchronization and memory access add instability at high percentiles (Section 5; references [41,83]).
  - Design: CPU-driven RDMA with stream-aware blocking, avoiding GPU-to-CPU copies and GPU synchronizations.
    - Sender flow (Figure 6): wait on CUDA event (previous kernel), block the CUDA stream using `cuStreamWaitValue32`, do RDMA write-with-immediate from pre-registered GPU buffers, poll completion queue, then unblock the CUDA stream via a shared flag.
    - Receiver flow (Figure 7): ensure target buffer is free, block stream, poll CQ to ensure arrival, perform a GDRCopy-based flush for GPU visibility, then unblock the stream.
    - Traffic tuning: prioritize ACKs on separate high-priority queues to prevent head-of-line blocking; adjust congestion control for unbalanced traffic (Section 5).
  - Comparison to DeepEP (GPU-to-GPU comms):
    - GPU kernels can push higher packet rates for tiny messages but consume SMs and need intricate low-level tuning (e.g., PTX, L2 usage). In MegaScale-Infer’s regime (hundreds of KB per sender–receiver pair), a CPU thread saturates NIC bandwidth while keeping GPUs fully available for compute (Section 5).

- Additional implementation details
  - Fused kernels: fuse all-gather with subsequent GEMM using Flux to overlap intra-node TP comms with compute; fuse gating, top-k selection, token scatter preparation, and related memory-bound steps (Section 6).
  - Expert load balancing: on-device redundancy for “hot” experts using a greedy approximation that minimizes the max per-node cost `max_j Cj` with allocation fractions `x_{i,j}` and expert activity costs `a_i` (Section 6).
  - Code footprint: about 4.9k C/C++ and 5k Python LOC for the M2N library (Section 6).

- Heterogeneous deployment (Section 4.3; Table 3)
  - Map attention to GPUs with high per-cost memory capacity and bandwidth (e.g., H20), and experts to GPUs with high per-cost compute (e.g., L40S). Table 3 quantifies GB/GB/s/TFLOPS per dollar.
  - Also improves throughput per watt because H20 and L40S respectively offer efficient bandwidth and compute per power (Section 4.3; Figure 10).

## 4. Key Insights and Innovations
- Disaggregated expert parallelism is a new within-layer split for MoE serving.
  - What’s new: previous work disaggregates phases (prefill vs decoding); MegaScale-Infer disaggregates modules within each layer (attention vs experts) and scales them independently (Figure 3).
  - Why it matters: it converts sparse, memory-bound expert computation into compute-bound GEMMs by aggregating tokens from many attention replicas, unlocking FFN efficiency (Sections 2.3 and 3).

- Ping-pong pipeline that provably hides communication
  - What’s new: a micro-batch pipeline that enforces concrete conditions (Equations (1)–(3)) to overlap two bidirectional communications per MoE layer with compute (Figure 4).
  - Why it matters: it keeps both sides busy despite per-layer token routing, which would otherwise create frequent bubbles.

- A purpose-built, CPU-driven M2N library for token routing
  - What’s new: a stream-aware, RDMA write-with-immediate design that avoids GPU-to-CPU copies, NCCL group overheads, and GPU synchronization; adds traffic-aware ACK prioritization and congestion control tuning (Section 5; Figures 6–7).
  - Why it matters: drastically lowers both median and tail latencies and improves throughput for the large-message regime typical in MoE routing (Figures 11–12), enabling communication to be fully hidden by the ping-pong pipeline.

- Heterogeneous deployment that matches hardware strengths to module characteristics
  - What’s new: formalizes attention-on-memory-rich GPUs and experts-on-compute-efficient GPUs, then evaluates end-to-end cost and power (Section 4.3; Table 3; Figures 9–10).
  - Why it matters: yields up to 3.24× higher decoding throughput per cost versus strong baselines on H20 and 1.80× higher decoding throughput per watt (Figures 9(a) and 10(a)).

- A practical, search-based deployment planner grounded in a simple, profile-calibrated performance model
  - What’s new: closed-form constraints and simple linear models for compute plus an empirical link model for communication enable a fast search of `tpa, tpe, na, m, B` under SLO and memory constraints (Algorithm 1; Section 4.2).
  - Why it matters: finds configurations where `Ta ≈ Te` and `m` suffices to hide communication, maximizing throughput per dollar given real hardware and workloads.

## 5. Experimental Analysis
- Setup (Section 7.1)
  - Hardware
    - Homogeneous: 8 nodes with 8× 80GB Ampere GPUs each, NVLink (400 GB/s intra-node), 8× 200 Gbps NICs per node.
    - Heterogeneous: H20 nodes (900 GB/s NVLink, 4× 400 Gbps NICs) and L40S nodes (PCIe intra-node, 2× 400 Gbps NICs).
  - Models (Table 4)
    - Mixtral-8×22B (141B params, 8 experts, top-2), DBRX (132B params, 16 experts, top-4), Scaled-MoE (317B params, 32 experts, top-4).
  - Workload: in-house production traces; median input length 571 tokens, output length 159; bfloat16 weights/activations/KV (Section 7.1).
  - Metrics
    - Primary: decoding throughput (tokens/s) per GPU for homogeneous, and per unit cost for heterogeneous; latency SLO is TBT ≤ 150 ms (Section 7.1).
    - Also: end-to-end throughput including prefill; throughput per unit power; M2N microbench latencies/throughput (Section 7.1).
  - Baselines: vLLM and TensorRT-LLM (both with TP/PP; TRT-LLM also supports EP). Prefill/decoding are evaluated separately for fairness (Section 7.1).

- Main results
  - Homogeneous decoding throughput (Figure 8(a))
    - MegaScale-Infer vs baselines:
      - Mixtral-8×22B and DBRX: up to 2.56× higher per-GPU decoding throughput over vLLM and 1.28× over TensorRT-LLM.
      - Scaled-MoE (multi-node): 7.11× over vLLM and 1.90× over TensorRT-LLM.
    - Interpretation: disaggregation and ping-pong overlap sustain FFN utilization even at scale, while baselines suffer from inter-node overhead and per-expert batch shrinkage.
  - Latency (TBT) (Figure 8(b))
    - Despite adding cross-node comms per layer, mean TBT is comparable to baselines, indicating communication is largely hidden by the pipeline and M2N efficiencies.
  - End-to-end throughput (prefill + decoding) on homogeneous GPUs (Figure 8(c))
    - Gains are smaller (up to 1.18×) because prefill is compute-bound and not improved by the decoding-focused design; still shows net benefits.
  - Heterogeneous decoding throughput per cost (Figure 9(a))
    - With attention on H20 and experts on L40S, MegaScale-Infer achieves up to 3.24× (vs vLLM on H20) and 1.86× (vs TensorRT-LLM on H20) higher throughput per dollar.
    - Mean TBT remains comparable or slightly better than L40S-only baselines (Figure 9(b)).
  - Heterogeneous end-to-end throughput per cost (Figure 9(c))
    - Offloading expert compute to L40S (cheaper compute) yields up to 1.66× end-to-end throughput per cost improvement versus H20 baselines.
  - Throughput per watt (Figure 10)
    - MegaScale-Infer achieves 1.80× (decoding) and 1.72× (end-to-end) higher throughput per unit power due to matching module characteristics to energy-efficient hardware.

- M2N microbenchmarks (Section 7.3; Figures 11–12)
  - Varying message sizes (2 KB–8 MB), with M=N=8:
    - Median latency reduced by up to 80.8% and P99 by up to 96.2% vs NCCL; throughput improves by up to 9.9× (Figure 11).
    - For the typical 256 KB size, median latency −68.2%, P99 −92.9%, throughput +4.2× (Figure 11).
  - Varying number of senders/receivers (M=N=4–32) at 256 KB:
    - Tail latency consistently lower (−54.7% to −96.9%), throughput +3.3× to +5.8× (Figure 12).
  - Takeaway: the library’s design choices and traffic tuning materially stabilize and accelerate token dispatch at the scales and sizes relevant to MoE inference.

- Ablations and diagnostics
  - Value of disaggregation and M2N (Figure 13)
    - Disaggregation alone (with NCCL) yields up to 4.66× over a colocated baseline by aggregating tokens across attention replicas.
    - Replacing NCCL with the custom M2N adds up to another 1.53× by hiding comms fully (meeting `Tc < Tf`).
  - Effect of micro-batch count m (Figure 14)
    - m=1 (no pipeline) under-utilizes GPUs; m=2 gives ~1.9× throughput; m=3 allows overlap of comm and compute, adding 1.10×/1.28×/1.38× more for Mixtral/DBRX/Scaled-MoE; larger m shows diminishing returns in a high-bandwidth testbed.
  - Choosing the right attention replication (Figure 15)
    - For DBRX, increasing attention DP from 1→8 shifts the bottleneck from attention to experts, maximizing normalized throughput without raising TBT. Further DP increases hurt by idling attention while experts compute—evidence for the “`Ta ≈ Te`” balance rule (Constraint 1).

- Deployment evidence (Section 8; Figure 16)
  - Production-scale deployment (∼10k GPUs) reduces cost by 1.5–2.0×.
  - Real traffic shows large expert load skew (Figure 16(a)); decoding expert loads are stable over time while prefill is more volatile (Figures 16(b)–(c)), motivating static/periodic balancing for decoding and more frequent adjustments for prefill.
  - Attention load imbalance arises from variable sequence lengths; they batch to a target per-node compute time using profiled operator runtime curves (Section 8).

- Overall assessment
  - The experimental design isolates decoding (where the method’s benefits accrue) and provides ablations that tie observed gains to the proposed mechanisms (pipeline fill, comm-latency reduction, balance of `Ta` and `Te`).
  - Results are consistent across models, scales, and hardware, and the microbenchmarks validate the communication substrate that underpins the pipeline-overlap claim.

## 6. Limitations and Trade-offs
- Balance and pipeline assumptions
  - The ping-pong pipeline relies on `Ta ≈ Te` and `Tc < Tf`. When compute balance or communication regimes change (e.g., very slow networks or very small messages due to extremely high expert counts), the overlap can break down or require m ≥ 4 (Equations (1)–(3); Section 4.1).
  - The planner depends on profiling-derived constants (`k1..k4`) and measured bandwidth utilization curves; workload drift or software updates can invalidate them, requiring periodic re-profiling (Section 4.2).

- Specialized communication stack and hardware
  - The M2N library assumes RDMA, GPUDirect, and GDRCopy are available and well-tuned; not all deployments have these capabilities (Section 6).
  - CPU-driven communication wins at hundreds of KB per connection (their regime), but for very small messages and very high degrees, a GPU-driven approach like DeepEP could outperform it (Section 5, “Comparison with DeepEP”).

- Scope focus on decoding
  - The largest gains come during decoding. Prefill benefits mainly via heterogeneous cost savings, not raw speedups (Figures 8(c), 9(c)).

- Memory footprint and replication
  - Attention replication across `na` nodes increases total memory for attention parameters and KV caches (Equation (8)), potentially limiting max batch sizes on smaller-memory GPUs (Section 4.2).

- Load-imbalance dynamics
  - Expert popularity skews change over time; the paper proposes on-device redundancy and periodic plans but does not detail an online reactive scheme (Section 6; Section 8, Figure 16).

## 7. Implications and Future Directions
- Changing the design space of MoE serving
  - Moving from monolithic, layer-colocated serving to module-level disaggregation enables independent scaling and hardware specialization. This sets a template for disaggregating other layer components (e.g., attention variants, normalization, or MoE variants).

- Practical deployment guidance
  - The conditions and model (Equations (1)–(8)) offer a practitioner’s checklist: balance `Ta`/`Te`, ensure `Tc < Tf`, pick m ≥ 3 or 4, and select `na` and `tpa/tpe` to satisfy memory and SLO constraints. Table 3 plus Figures 9–10 provide a concrete rationale for pairing memory-rich GPUs with attention and compute-efficient GPUs with experts.

- Follow-on research opportunities
  - Adaptive runtime planning: close the loop by continuously re-estimating `k1..k4`, bandwidth utilization curves, expert hotness, and sequence length distributions to re-optimize `na, m, B, tpa, tpe` online.
  - Hybrid CPU/GPU communication: dynamically choose CPU- vs GPU-driven dispatch depending on message size and degree, potentially leveraging both DeepEP-like GPU queues and MegaScale’s CPU RDMA for different layers or traffic regimes (Section 5).
  - Scheduling and fairness: integrate sequence-length-aware packing and expert hotness into a global scheduler that co-optimizes pipeline utilization, latency SLOs, and multi-tenant isolation.
  - Extending to training or fine-tuning: the same disaggregation and M2N mechanisms could be adapted to optimize MoE training steps that involve frequent token re-partitioning.
  - Hardware co-design: given the success of ACK prioritization and congestion tuning (Section 5), NIC/driver features for token-routing collectives (M2N/N2M) could further reduce tail latencies and CPU overhead.

- Applications
  - Cost-optimized production serving of very large MoE LLMs.
  - Cloud providers combining heterogeneous GPU fleets (e.g., H20 and L40S) to improve both cost and power efficiency (Figures 9–10).
  - Long-context serving alongside MoE (prefill/decoding and attention/expert disaggregation are compatible), and scenarios with bursty expert popularity where on-device redundancy helps.

> Overall, MegaScale-Infer demonstrates that splitting attention and experts across independently scaled, possibly heterogeneous hardware—then carefully filling a micro-batch pipeline while stabilizing token routing—can transform MoE decoding from memory-bound inefficiency into high-utilization, cost-effective serving (Figures 1(c), 8–10, 11–12).
