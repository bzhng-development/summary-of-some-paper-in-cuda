# MegaScale-Infer: Serving Mixture-of-Experts at Scale with Disaggregated Expert Parallelism

**ArXiv:** [2504.02263](https://arxiv.org/abs/2504.02263)

## 🎯 Pitch

MegaScale-Infer pioneers a novel system for serving large-scale Mixture-of-Experts (MoE) language models by decoupling attention and feed-forward (expert) modules within each Transformer layer, allowing independent scaling and specialized hardware for each. This architectural disaggregation is paired with a smart, micro-batched 'ping-pong' pipeline and an optimized M2N communication library, transforming MoE token sparsity from a throughput bottleneck into an efficiency advantage. As a result, MegaScale-Infer significantly boosts GPU utilization and cost-effectiveness, achieving up to 1.9× greater per-GPU throughput and slashing inference costs by 1.5–2× compared to state-of-the-art systems—a crucial advancement for scalable, economical AI deployment.

---

## 1. Executive Summary
MegaScale-Infer introduces a new way to serve Mixture-of-Experts (MoE) large language models by disaggregating the attention and feed-forward (expert) parts of every Transformer layer onto different GPU pools, then stitching them together with a micro-batched “ping‑pong” pipeline and a specialized M2N communication library. This architecture turns MoE’s token sparsity from a utilization problem into an advantage, delivering up to 1.90× higher per‑GPU decoding throughput than strong baselines on homogeneous clusters and up to 1.86× higher throughput per unit cost under heterogeneous deployment (Figures 8 and 9; Abstract).

## 2. Context and Motivation
- Problem addressed
  - During decoding (token-by-token generation), attention is memory‑bound because it must read the key–value (KV) cache of all prior tokens, while the feed‑forward network (FFN) becomes compute‑efficient only at large batch sizes (§2.1).
  - In MoE, each token activates only K of E experts (e.g., top‑2 of 8). This sparsity reduces the number of tokens per expert, making each expert’s batch small and underutilizing GPUs (§2.3, Figure 1b).

- Why it matters
  - Serving costs and throughput: Low GPU utilization during decoding dominates production inference cost. The paper quantifies that for an A100 GPU (312 TFLOPs, 2 TB/s), a dense FFN needs batch size b ≥ F/B = 156 to saturate compute. With MoE top‑2-of‑8 and batch 156, each expert sees 39 tokens—only ~25% of ideal utilization (util ≈ topk/E = 2/8; §2.3).

- Prior approaches and their limits
  - Standard parallelism: tensor parallelism (TP), pipeline parallelism (PP), and expert parallelism (EP) (§2.2). TP adds communication; EP suits MoE but still suffers from small per-expert batches in decoding.
  - Long-context disaggregation (e.g., Infinite‑LLM) focuses on dense attention/KV memory pressure in long-context settings and does not address MoE’s token routing complexity (§2.4).
  - Prefill/decoding disaggregation (e.g., DistServe, Splitwise) removes interference between phases but leaves MoE sparsity during decoding unaddressed (§3).

- Positioning
  - MegaScale-Infer goes beyond phase disaggregation and separates attention and experts within each layer (§3, Figure 3). It pairs this with (1) micro‑batched ping‑pong pipelining to keep both sides busy (Figure 4), (2) module-specific parallelism (replicate attention; EP for experts), (3) a search-guided deployment plan backed by a performance model (Algorithm 1), and (4) a high-performance M2N communication library to make token dispatch practical at scale (§§4–5).

## 3. Technical Approach
At a high level, the system splits each Transformer layer’s computation into two GPU pools—attention nodes and expert nodes—and moves micro‑batches back and forth between them in a carefully tuned pipeline (Figures 3–4).

- Key terms used once for clarity
  - KV cache: intermediate attention states (keys and values) saved per past token to speed up decoding (§2.1).
  - MoE: a layer that replaces the dense FFN with many experts (independent FFNs) and a gating network that routes each token embedding to the top‑K experts (§2.2, Figure 2a).
  - Expert Parallelism (EP): different experts live on different devices; tokens are dispatched to the devices that host their selected experts (Figure 2b).
  - M2N communication: token dispatch between M attention senders and N expert receivers (and the reverse direction for aggregation) (§4).

Step-by-step design

1) Disaggregated expert parallelism (DEP)
- Architecture (Figure 3; §3)
  - Attention nodes: replicate attention parameters and store KV caches; use intra-node TP to exploit NVLink.
  - Expert nodes: each node stores one expert’s parameters; all expert nodes together form the EP group; intra-node TP is used as needed.
  - Requests are batched globally, then split into micro‑batches that “ping‑pong” between attention and expert nodes in every MoE layer.

- Why disaggregate?
  - Aggregating requests from multiple attention replicas increases tokens seen by each expert across the instance, turning expert compute from memory‑bound to compute‑bound (§2.4; Figure 1c).
  - It enables heterogeneous deployment—attention on GPUs with strong memory bandwidth/capacity; experts on GPUs with cost‑effective compute (§4.3; Table 3).

2) Ping‑pong pipeline parallelism (PPP)
- Problem: If attention and experts are separated, each side would idle while waiting for the other or for network transfers (§4.1).
- Solution: Split the global batch into m micro‑batches and run them in a wavefront so that when attention is computing on micro‑batch i, experts are computing on i‑1 and communication for i or i‑1 is overlapped (Figure 4).
- When does it work? Three conditions derived and enforced by the deployment planner (§4.1):
  - Balance compute: Ta ≈ Te (Eq. 1), where Ta and Te are per‑micro‑batch compute times on attention and expert nodes.
  - Communication shorter than compute: Tc < Tf, with Tf = max{Ta, Te} (Eq. 2).
  - Enough micro‑batches to cover two crossings per layer: m × Tf ≥ 2 × (Tf + Tc), i.e., m ≥ 2 × (1 + Tc/Tf) (Eq. 3). In fast networks (Tc < Tf/2), m ≥ 3 is sufficient (§4.1).
- Latency model: For L MoE layers, the iteration latency per micro‑batch is bounded by (Ta + Te + 2Tc) + mTf(L − 1) ≤ Titer ≤ mTfL; total latency of the global batch is Ttotal = (Ta + Te + 2Tc) + Tf(mL − 1) (Eq. 4–5).

3) Deployment plan search with a calibrated performance model
- Search space and constraints (§4.2; Algorithm 1, Table 1)
  - Variables: attention TP size (tpa), expert TP size (tpe), number of attention nodes (na), number of micro‑batches (m), global batch size (B).
  - Constraints: service-level objective on time‑between‑tokens (Titer ≤ SLO; Eq. 7), and memory capacity for attention GPUs to hold KV cache and attention parameters (4 m b_a s h L / g + 2Pa < tpa Ca; Eq. 8).
- Simulation and balancing
  - Attention compute time is modeled as Ta ≈ k1 ba + k2, experts as Te ≈ k3 be + k4, where ba and be are per‑micro‑batch token counts per attention/expert node and k’s come from profiling (§4.2; Table 2 lists GEMM shapes considered).
  - Balance condition (Eq. 1) translates to choosing na so that na = (k1 E) / (k3 K) (since ba m na = be m E/K = B; §4.2).
  - Communication time per micro‑batch Tc is the max of A→E and E→A transfers, each estimated from measured link utilization curves as Tc = max{ bahK/tpa / (Wa×Util(…)), beh/tpe / (We×Util(…)) } (Eq. 6).
  - The planner enumerates feasible (tpa, tpe), chooses na to meet Ta ≈ Te, tries m ∈ {3,4,…}, and binary‑searches the largest B that respects SLO; it outputs the plan with best throughput per unit cost (Algorithm 1).

4) High‑performance M2N communication library (§5; Figures 5–7)
- Motivation from measurement: standard NCCL incurs extra copies through a CPU proxy, processes peer operations in small groups, and involves GPU synchronization—leading to higher median and tail latency, especially as the number of receivers grows (Figure 5).
- Design choices (CPU‑orchestrated RDMA; no superfluous copies or GPU synchronizations)
  - Pre‑registered GPU buffers; synchronization via CUDA events to ensure producer kernels finished (Figures 6–7).
  - Block the CUDA stream with cuStreamWaitValue32 while host issues RDMA Write with immediate to all receivers and polls their completion queues (CQs); then unblock the stream by writing a shared flag (§5).
  - Receivers poll CQs and perform a GPU-visible flush with GDRCopy to ensure data visibility, then unblock (§5).
  - Traffic optimizations: prioritize ACK packets on separate high‑priority queues and fine‑tune congestion control, which stabilizes tail latency under unbalanced traffic (§5).
- Rationale vs DeepEP (§5): CPU‑driven data plane avoids consuming GPU SM resources and cache contention; at the typical per‑peer sizes here (hundreds of KB), a single CPU thread can saturate the link. For much smaller messages and many QPs, GPU‑driven approaches may win.

5) Implementation extras (§6)
- Fused kernels: (i) fuse intra‑node TP all‑gather with subsequent GEMM using Flux; (ii) fuse gating, top‑K selection, token counting/weighting, and scatter into one pass to reduce memory traffic (§6).
- Expert load balancing: replicate hot experts on device proportionally to observed popularity from recent traffic to minimize max per‑node cost with a greedy approximation (§6).
- Code: a PyTorch extension with ~4900 C/C++ and ~5000 Python LoC; relies on GPUDirect and GDRCopy (§6).

6) Heterogeneous deployment (§4.3; Table 3)
- Insight: attention is memory‑bound and KV‑heavy; experts are compute‑bound. Table 3 shows per‑cost memory bandwidth/capacity favor H20, while per‑cost TFLOPs favor L40S. The planner enumerates hardware pairings and often chooses H20 for attention and L40S for experts.

## 4. Key Insights and Innovations
- Disaggregated expert parallelism (DEP) within each layer
  - Novelty: goes beyond phase disaggregation by splitting attention and experts and scaling them independently (§3, Figure 3).
  - Why it matters: raises tokens per expert by consolidating demand from multiple attention replicas, recovering FFN compute efficiency lost to MoE sparsity (Figure 1c).

- Ping‑pong pipeline with principled conditions
  - Novelty: a micro‑batch pipeline across attention and experts with explicit conditions to fully hide communication (Eqs. 1–3; Figure 4).
  - Significance: prevents idle time on either side and makes per‑layer cross‑cluster communication practical without hurting latency beyond SLO (§4.1).

- Performance‑model‑guided deployment search
  - Novelty: a compact, profile‑calibrated model ties together compute balance, network utilization vs. message size, memory limits, and SLO to choose tpa/tpe/na/m/B (Algorithm 1; Eqs. 4–8).
  - Significance: ensures plans fill the pipeline (m), balance module times (na), and keep M2N under compute (Tc < Tf), maximizing throughput per cost (§4.2).

- A purpose‑built M2N communication layer
  - Novelty: CPU‑driven RDMA Writes with immediate, no GPU‑to‑CPU copies, no NCCL group overhead, explicit stream blocking/unblocking, plus prioritized ACK and tuned congestion control (§5; Figures 6–7).
  - Significance: reduces median and tail latency and raises throughput substantially vs. NCCL in the token‑dispatch regime (Figures 11–12).

- Heterogeneous hardware co‑design
  - Novelty: uses per‑cost metrics (Table 3) to place attention on bandwidth‑rich H20 and experts on compute‑efficient L40S (§4.3).
  - Significance: amplifies gains into throughput per dollar and per watt improvements (Figures 9–10).

These are fundamental architectural shifts (DEP, PPP, M2N) combined with a principled planner; not just kernel‑level tweaks.

## 5. Experimental Analysis
- Setup (§7.1)
  - Clusters: (i) 8 nodes with 8×80GB Ampere GPUs each (NVLink intra‑node; 200 Gbps NICs), (ii) heterogeneous cluster with H20 (900 GB/s NVLink; 4×400 Gbps) and L40S (PCIe; 2×400 Gbps).
  - Models (Table 4): Mixtral‑8×22B (E=8, K=2), DBRX (E=16, K=4), Scaled‑MoE 317B (E=32, K=4).
  - Workload: production traces; median prompt 571 tokens; median output 159 tokens; bfloat16 for weights/activations/KV.
  - Baselines: vLLM and TensorRT‑LLM with standard optimizations; all methods evaluated with prefill and decoding temporally separated to avoid interference (§7.1).

- Primary metric and SLO
  - Decoding throughput per GPU or per cost (heterogeneous), with Time‑Between‑Tokens (TBT) SLO = 150 ms (§7.1).

- Main results on homogeneous Ampere (Figure 8)
  - Decoding throughput per GPU:
    - DEP+PPP+M2N (MegaScale-Infer) vs baselines: 
      > “achieves 2.56× and 1.28× higher per‑GPU decoding throughput than vLLM and TensorRT‑LLM” across Mixtral and DBRX (Figure 8a).
      - On the largest model (Scaled‑MoE 317B), where inter‑node comms dominate for baselines, improvements reach
      > “7.11× vs vLLM and 1.90× vs TensorRT‑LLM” (Figure 8a).
  - Latency:
    - Despite per‑layer cross‑node transfers, TBT is comparable to baselines (Figure 8b), because communication is overlapped by PPP and accelerated by M2N.
  - End‑to‑end throughput (prefill + decoding):
    - Gains are smaller since prefill is compute‑bound and not helped by DEP; still up to 1.18× better (Figure 8c).

- Heterogeneous results (H20 attention + L40S experts; Figure 9)
  - Decoding throughput per cost:
    - Compared to vLLM (H20) and TensorRT‑LLM (H20):
      > “up to 3.24× and 1.86×” improvement, respectively (Figure 9a).
  - Latency:
    - TBT comparable to baselines; slightly better than L40S‑only deployments (Figure 9b).
  - End‑to‑end throughput per cost:
    - Up to 1.66× improvement (Figure 9c).
  - Power efficiency:
    - Throughput per watt improved by 1.80× (decoding) and 1.72× (end‑to‑end) due to matching bandwidth‑per‑watt (H20) and compute‑per‑watt (L40S) to workload (§7.2; Figure 10).

- M2N micro‑benchmarks (Figures 11–12)
  - Varying message sizes (2 KB to 8 MB), M=N=8:
    - Median latency reduced by up to 80.8%; P99 reduced by up to 96.2%; throughput up to 9.9× higher vs NCCL (Figure 11).
    - At a representative size (≈256 KB per peer), improvements are
      > “68.2% lower median latency, 92.9% lower P99, and 4.2× higher throughput” (Figure 11).
  - Scaling M and N (4 to 32) at 256 KB:
    - Tail latency reduced by 54.7%–96.9%; throughput improved by 3.3×–5.8× (Figure 12).
  - Takeaway: the token‑dispatch regime is where M2N’s design decisions pay off.

- Ablations (Figures 13–15)
  - Effect of disaggregation and M2N:
    - Disaggregating attention and experts alone (using NCCL) yields up to 4.66× speedup over colocated baselines; adding M2N gives up to an additional 1.53× (Figure 13).
  - Micro‑batches (m):
    - Going from m=1 (no pipeline) to m=2 approximately halves idle time (≈1.9× throughput). Raising to m=3 enables full overlap of comms and compute, adding 1.10×–1.38× more (Figure 14). Larger m brings diminishing returns in high‑bandwidth settings.
  - Balancing attention replicas (DP) for DBRX:
    - Throughput scales linearly and latency stays flat as DP grows from 1→4 (attention bottleneck). At DP=8, Ta≈Te and throughput peaks. Beyond that, experts become the bottleneck and latency rises while normalized throughput falls (Figure 15).

- Deployment at scale
  - In production on ~10,000 GPUs and, under heterogeneous deployment, reduces cost by 1.5–2.0× for the same traffic (§8).
  - Real traffic analysis finds both expert and attention imbalances; the system uses expert replication and batch composition to balance runtime (§8; Figure 16).

- Do the experiments support the claims?
  - Yes, across three MoE models, two clusters, two strong baselines, and with ablations that isolate each component’s effect. The separation of prefill and decoding for all systems ensures a fair apples‑to‑apples decoding comparison (§7.1). The M2N advantages are validated via focused micro‑benchmarks (Figures 11–12).

## 6. Limitations and Trade-offs
- Additional per‑layer communication
  - DEP introduces two cross‑pool transfers per MoE layer (A→E, E→A). PPP hides most of it, but the SLO constraint (Eq. 7) and pipeline‑fill condition (Eq. 3) can force choices (e.g., m≥3 or 4) that slightly increase per‑token latency (Figure 8b).
- Network and system assumptions
  - The design assumes RDMA‑capable networking and benefits from multiple high‑speed NICs per node (e.g., 200–400 Gbps; §7.1). Commodity Ethernet without RDMA or with limited bandwidth would reduce Tc‑hiding effectiveness.
- CPU involvement in the data plane
  - The M2N library uses CPU‑orchestrated RDMA. This is ideal for hundreds‑of‑KB messages and modest QP counts (§5), but for very small messages and many QPs, GPU‑driven approaches (e.g., DeepEP) may achieve higher peak packet rates at the cost of GPU SM time (§5).
- Planning model portability
  - The Ta/Te linear models and network Util(·) curves are learned from profiling (§4.2). Porting to new hardware or drivers requires re‑profiling to maintain accuracy.
- Memory pressure on attention GPUs
  - Attention nodes hold KV caches; Eq. (8) shows memory grows with micro‑batches m, sequence length s, hidden size h, and layers L. Extremely long contexts or very large L may limit feasible m or batch size B.
- Expert load balancing via replication
  - Replicating hot experts (for load balance) consumes additional memory capacity and requires periodic replanning (§6, §8), which adds operational complexity.

## 7. Implications and Future Directions
- How this changes the field
  - Establishes that disaggregating attention and experts is a first‑order architectural lever for MoE inference—comparable in impact to prefill/decoding disaggregation—unlocking high utilization despite MoE sparsity (Figures 1c, 8, 9).
  - Demonstrates that a targeted communication layer (M2N) can materially change system‑level outcomes for token routing (Figures 11–12).

- Follow‑up research enabled
  - Adaptive runtime planning: dynamically adjust m, na, and even hardware assignment as load and sequence-length distributions shift.
  - Unified CPU/GPU communication strategies: combine CPU‑based M2N and GPU‑driven paths (à la DeepEP) and switch based on message size/QP count (§5).
  - Cross‑phase co‑design: extend DEP to optimize prefill as well (e.g., selective attention offload or attention‑specific accelerators) without hurting latency (§7.2).
  - Smarter gating/dispatch: incorporate pre‑gating or token clustering to reduce communication volume (related to §6 and [47] in References).
  - Wider heterogeneity: include future bandwidth‑rich/compute‑lean accelerators for attention and compute‑dense cards for experts; integrate per‑watt planning (Figures 9–10; Table 3).

- Practical applications
  - Cost‑optimized deployment of production MoE models (e.g., Mixtral, DBRX‑style architectures) in cloud and on‑prem clusters, with substantial savings under heterogeneous fleets (§7.2, §8).
  - Multi‑tenant serving platforms: DEP lets operators independently scale attention (memory‑bound) and expert (compute‑bound) pools and apply targeted load‑balancing policies (Figure 16; §8).

In short, MegaScale-Infer provides a principled, end‑to‑end recipe—architecture, pipeline, planner, and communication—to make large‑scale MoE decoding efficient and economical. The combination of DEP, PPP, and M2N is broadly applicable to other conditional‑compute architectures that exhibit similar sparsity‑induced underutilization.
