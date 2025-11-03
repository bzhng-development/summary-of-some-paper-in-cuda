# Insights into DeepSeek‑V3: Scaling Challenges and Reflections on Hardware for AI Architectures

**ArXiv:** [2505.09343](https://arxiv.org/abs/2505.09343)
**Authors:** Chenggang Zhao, Chengqi Deng, Chong Ruan, Damai Dai, Huazuo Gao, Jiashi Li, Liyue Zhang, Panpan Huang, Shangyan Zhou, Shirong Ma, Wenfeng Liang, Ying He, Yuqing Wang, Yuxuan Liu, Y. X. Wei
**Institutions:** (Not explicitly listed in the abstract)

## 🎯 Pitch

This paper introduces DeepSeek-V3/R1, a novel hardware-model co-design strategy that allows large language models (LLMs) to be run efficiently on a 2,048-GPU cluster by targeting memory, computation, and bandwidth limitations through Multi-head Latent Attention and FP8 training. This breakthrough significantly reduces the cost and hardware requirements associated with LLMs, democratizing advanced AI capabilities for a broader range of organizations, and offering a pathway for more accessible AI innovation.

---

## 1. Executive Summary (2-3 sentences)
This paper presents a hardware–model co-design blueprint behind DeepSeek‑V3/R1, showing how to train and serve state-of-the-art large language models (LLMs) on a relatively modest 2,048‑GPU H800 cluster by attacking the three dominant bottlenecks: memory, computation, and interconnect bandwidth. The work combines architectural choices (Multi-head Latent Attention, Mixture-of-Experts with hardware-aware routing, Multi‑Token Prediction) with low‑precision FP8 training and a cost‑effective multi‑plane network, and distills concrete requirements for next‑generation AI hardware.

## 2. Context and Motivation
- Problem this work targets
  - Modern LLMs are hitting the “AI memory wall” and interconnect limits: memory for inference (especially Key/Value cache, or `KV cache`) scales with context length; compute efficiency falls when attention becomes memory bound; and Mixture‑of‑Experts (`MoE`) requires costly all‑to‑all communication.
  - Section 2.1 frames the memory trajectory: 
    > “LLMs generally require significant memory resources, with memory demands increasing by more than 1000% per year, while high‑speed memory capacity … typically [grows] less than 50% per year” (Sec. 2.1).
  - For MoE, expert parallelism hinges on network bandwidth and latency; the interconnect often becomes the throughput limiter (Sec. 2.3.2).
- Why it matters
  - Cost and accessibility: state‑of‑the‑art models commonly require clusters with “tens or even hundreds of thousands of GPUs or TPUs” (Sec. 1.1). That bar excludes most organizations. This paper demonstrates how a smaller cluster can still reach top performance by co‑designing models, software, and network fabric.
  - User experience and reasoning models: reasoning‑style systems (o1/o3, DeepSeek‑R1, etc.) rely on rapid token generation at long lengths; poor inference throughput directly degrades product usability (Sec. 2.3.4).
- Prior approaches and gaps
  - Dense models scale parameter count and compute uniformly, leading to huge training and serving costs.
  - KV‑reduction techniques (GQA/MQA) compress cache size but don’t tackle communication‑heavy MoE issues (Sec. 2.1.2).
  - Quantization is popular for inference (e.g., GPTQ/AWQ) but less so for training; FP8 training existed in vendor libraries but lacked open, large‑scale demonstrations and practical recipes for MoE (Sec. 3.1).
  - Datacenter networks usually use three‑tier fat trees or proprietary fabrics; cost‑effective, low‑latency alternatives that scale cleanly remain challenging (Sec. 5.1).
- How this paper positions itself
  - It is not a full model report (that’s [26]); instead it explains the co‑designed mechanisms that made DeepSeek‑V3/R1 efficient on 2,048 H800s: Multi‑head Latent Attention (MLA) to shrink KV cache, MoE with routing tuned to the H800’s asymmetric bandwidths, FP8 fine‑grained training, and a multi‑plane, two‑layer fat‑tree network (Fig. 1, Secs. 2–5). It also crystallizes hardware features the community should build next (Secs. 3.1.2, 3.2.2, 4.4.2, 4.5.2, 5.2.2, 6).

## 3. Technical Approach
At a glance (Fig. 1), DeepSeek‑V3 integrates four pillars:
1) memory‑efficient attention (MLA), 2) sparse computation (DeepSeekMoE), 3) low‑precision training (FP8 with fine‑grained scaling), and 4) inference accelerators (Multi‑Token Prediction). These are then matched to hardware and network strategies (Secs. 4–5).

- Multi‑head Latent Attention (`MLA`, Sec. 2.1.2; Fig. 1, bottom-left)
  - What it solves: the `KV cache` stores per‑token Keys and Values from all attention heads during decoding; this becomes both memory and bandwidth bound. 
  - How it works: instead of caching head‑wise K/V tensors, MLA compresses them into a much smaller “latent” vector using a trained projection. At inference, only this latent `c_t^KV` is cached (Fig. 1 shows latent vectors), and per‑head K/V are reconstructed on the fly via learned projections.
  - Why this choice: it reduces per‑token KV memory dramatically without rethinking the Transformer stack, and it shifts less traffic through memory‑bound GEMV paths.

- Mixture‑of‑Experts (`MoE`) with hardware‑aware routing (Sec. 2.2 and 4.3; Fig. 1, lower-right)
  - MoE recap: many “experts” (Feed‑Forward Networks) exist, but only a small subset is activated per token, guided by a gating router. This keeps compute per token low while growing total parameters.
  - DeepSeekMoE design highlights:
    - One shared expert plus multiple “routed experts” (Fig. 1).
    - Node‑Limited Routing (Sec. 4.3): in an 8‑node, 256‑expert setting (4 experts/GPU), each token is routed to up to 4 nodes. IB transfers to a node happen once, then intra‑node NVLink forwards to the specific GPUs. This “deduplicates” inter‑node traffic and exploits higher intra‑node bandwidth.
    - Example (Sec. 4.3): if a token needs 9 experts (8 routed + 1 shared), naïvely spread across 8 nodes, inter‑node time is “8t”. With node‑limited routing and NVLink forwarding, it reduces to “Mt” where M ≤ 4.
  - Implementation: Expert Parallelism (`EP`) uses two all‑to‑all phases—`dispatch` (send token activations to experts) and `combine` (gather and reduce expert outputs). DeepSeek’s `DeepEP` library overlaps these with compute (Sec. 2.3.1) and reaches near‑line‑rate bandwidth (Fig. 7).

- FP8 mixed‑precision training with fine‑grained scaling (Sec. 3.1; Fig. 1)
  - What is FP8: 8‑bit floating point formats (e.g., E4M3/E5M2) reduce memory/compute cost relative to BF16. 
  - How training is stabilized:
    - High‑precision accumulation (to internal registers) to curb rounding error.
    - Fine‑grained scaling: tile‑wise 1×128 quantization for activations and block‑wise 128×128 for weights.
    - Custom FP8 GEMMs (open‑sourced as `DeepGEMM`, [77]).
  - Practical mapping: Fig. 1 annotates which forward/backward paths run in FP8 (e.g., attention and FFN core GEMMs) and which keep BF16/FP32 for numerically sensitive ops.

- Multi‑Token Prediction (`MTP`) for inference speed (Sec. 2.3.3; Fig. 1, top)
  - What it does: adds shallow, single‑layer “heads” that predict the next 2–4 tokens cheaply. The main model then verifies these in parallel (a form of self‑drafting speculative decoding).
  - Why it helps: reduces sequential decoding steps, increasing effective tokens/sec while preserving accuracy.
  - Reported behavior:
    > “An MTP module achieves an acceptance rate of 80%–90% for predicting the second subsequent token, increasing generation TPS by 1.8×” (Sec. 2.3.3).

- Communication compression: FP8 dispatch and `LogFMT` exploration (Sec. 3.2)
  - Dispatch runs in FP8 (1 byte/element), halving traffic vs BF16; combine currently BF16 for accuracy, though FP8/E5M6/hybrids are being tested (Sec. 3.2).
  - `LogFMT‑nBit` (novel): a block‑local logarithmic quantizer that maps |x| to log‑space, linearly quantizes within the tile’s dynamic range, and decodes with exp. 
    - Observations: LogFMT‑8 outperforms E4M3/E5M2 on 7B‑scale tests for residual‑branch simulation; LogFMT‑10 approximates BF16 (Sec. 3.2).
    - Not deployed due to encode/decode overhead (50%–100%) on current GPUs (Sec. 3.2.1). The paper recommends native compression/decompression units in future NICs/I/O dies (Sec. 3.2.2).

- Hardware and parallelism choices for H800 (Sec. 4)
  - Context (Fig. 2): H800 has reduced NVLink bandwidth (400 GB/s total per node, ~160 GB/s achievable per direction cited in Sec. 4.3) vs H100, but each node includes eight 400 Gbps IB NICs.
  - Design choices (Sec. 4.2):
    - Avoid Tensor Parallelism during training (too NVLink‑heavy); optionally use for latency‑critical inference.
    - Use `DualPipe` pipeline parallelism to overlap computing with MoE comms and reduce bubbles.
    - Push Expert Parallelism (EP) hard over IB using DeepEP; all‑to‑all > 40 GB/s per GPU (Fig. 7).
  - Overlap strategy for throughput (Sec. 2.3.1): dual micro‑batch overlap decouples MoE/MLA compute and their respective dispatch/combine steps so communication is hidden behind another batch’s compute; production also separates prefill (big batches) and decode (latency‑critical) onto different EP group sizes.

- Cluster network: Multi‑Plane Two‑Layer Fat‑Tree (`MPFT`) (Sec. 5.1; Figs. 3–6; Table 3)
  - Each GPU‑NIC pair belongs to one of eight planes; cross‑plane traffic is forwarded intra‑node via NVLink/PCIe (Fig. 3).
  - The ideal design (Fig. 4) is NICs with multiple physical ports bonded into one logical interface with native out‑of‑order placement; current ConnectX‑7 falls short of this ideal, so the deployed MPFT uses per‑GPU NICs per plane.
  - Why MPFT: a two‑layer topology lowers cost and latency vs a three‑layer fat tree while still scaling to >10k endpoints (Table 3).

- Low‑latency I/O path: InfiniBand GPUDirect Async (`IBGDA`) (Sec. 5.2.3)
  - GPUs directly post RDMA work requests and ring the NIC “doorbell,” removing CPU proxy threads. This reduces control‑plane latency and improves many‑small‑packet sends.

## 4. Key Insights and Innovations
- MLA reduces KV cache by multiples without hurting generality (Sec. 2.1.2; Table 1)
  - What’s new: a trained latent space that collapses all heads’ K/V into a compact vector cached per token; per‑head K/V are reconstructed on demand.
  - Why it matters: KV cache per token drops to 70 KB for DeepSeek‑V3 vs 327 KB (Qwen‑2.5‑72B) and 516 KB (LLaMA‑3.1‑405B) (Table 1).
  - Quote:
    > “DeepSeek‑V3 (MLA) 70.272 KB … Qwen‑2.5 72B (GQA) 327.680 KB … LLaMA‑3.1 405B (GQA) 516.096 KB” (Table 1).

- Hardware‑aware MoE routing (Node‑Limited Routing) that “deduplicates” inter‑node traffic (Sec. 4.3)
  - What’s new: tie the router’s Top‑K expert selection to node groups so that IB traffic is minimized and high‑bandwidth intra‑node NVLink forwards within the node.
  - Why it matters: all‑to‑all is the dominant cost in EP; reducing IB fan‑out directly boosts throughput. This is a fundamental systems innovation rather than a minor tuning.

- Practical FP8 training recipe for large MoE models with fine‑grained scaling (Sec. 3.1; Fig. 1)
  - What’s new: a full, open recipe that uses tile/block‑wise scaling, high‑precision accumulation, and custom kernels (`DeepGEMM`) to make FP8 training robust in MoE.
  - Why it matters: halves activation/weight memory vs BF16 and lifts compute throughput; in small/medium ablations, accuracy loss is ≤0.25% (Sec. 2.4).

- Multi‑Token Prediction (MTP) as a built‑in, training‑time feature to enable speculative decoding (Sec. 2.3.3; Fig. 1)
  - What’s new: lightweight one‑layer heads for next‑token(s) that can be validated in parallel; this is integrated and trained jointly.
  - Why it matters: real‑world acceptance rate “80–90%” for the second token yields “1.8×” tokens/sec, which is critical for reasoning‑length outputs.

- Multi‑Plane, two‑layer fat tree that matches single‑plane multi‑rail performance at lower cost/latency (Sec. 5.1; Figs. 5–6; Table 3)
  - What’s new: MPFT shows all‑to‑all performance comparable to MRFT thanks to NCCL PXN pathing (Figs. 5–6) while enabling >10k endpoints with two switching tiers.
  - Why it matters: achieves cost/latency comparable or better than three‑tier fat trees and competitive with Slim Fly (Table 3).

## 5. Experimental Analysis
- What is evaluated and how
  - System‑level communication and training throughput, not end‑task quality (those are in [26]). The paper measures NCCL all‑to‑all bandwidth/latency, DeepEP kernels under EP traffic, protocol latency, and end‑to‑end training throughput/MFU on 2,048 GPUs.
- Main quantitative findings
  - KV cache reduction via MLA (Table 1):
    > “70.272 KB per token” with MLA vs “327.680 KB” (Qwen‑2.5‑72B, GQA) and “516.096 KB” (LLaMA‑3.1‑405B, GQA).
  - Training compute cost advantage of MoE (Table 2; seq len 4096):
    > DeepSeek‑V3 MoE: “250 GFLOPs/token” vs dense 72B: “394 GFLOPs/token” and dense 405B: “2448 GFLOPs/token.”
  - Theoretical decode upper bound (Sec. 2.3.2):
    > With 400 Gbps IB and dual‑microbatch overlap: “14.76 ms TPOT (≈67 tok/s)”; with GB200 NVL72‑class scale‑up bandwidth: “>0.82 ms TPOT (≈1200 tok/s).” These are analytical upper limits, not measured end‑to‑end.
  - MTP effectiveness (Sec. 2.3.3):
    > “80–90% acceptance for the second token … 1.8× TPS improvement.”
  - All‑to‑all and EP kernel performance (Figs. 5–7):
    - Fig. 5/6: MPFT vs MRFT all‑to‑all bandwidth/latency are “nearly identical” from 32 to 128 GPUs.
    - Fig. 7: DeepEP achieves >40 GB/s per GPU for both dispatch and combine across 16–128 GPUs, “nearly saturating the 400 Gbps NIC bandwidth.”
  - End‑to‑end training throughput on 2,048 GPUs (Table 4):
    > Tokens/day: “272.80B (MPFT) vs 272.52B (MRFT)”; MFU (causal): “38.94% vs 38.90%”; 1F1B time: “13.95 s vs 14.00 s.” Differences are within noise.
  - Protocol latency (Table 5):
    > For 64‑byte messages, intra‑leaf: IB “2.8 μs” vs RoCE “3.6 μs”; cross‑leaf: IB “3.7 μs” vs RoCE “5.6 μs.” NVLink intra‑node is “3.33 μs.”
  - Network cost/scalability (Table 3):
    > MPFT: “16,384 endpoints … $72M total … $4.39k per endpoint,” versus a 3‑layer fat tree at “65,536 endpoints … $491M … $7.5k per endpoint.”
- Robustness checks and ablations
  - FP8 training ablations show ≤0.25% relative loss on 16B and 230B DeepSeek‑V2 models before integrating FP8 into V3 (Sec. 2.4).
  - LogFMT is validated on ~7B dense models (residual‑branch simulation) and found superior to E4M3/E5M2 at 8 bits; 10‑bit approaches BF16 (Sec. 3.2), but not deployed due to encode/decode overhead.
- Do the experiments support the claims?
  - For system performance and cost effectiveness, yes: network microbenchmarks (Figs. 5–7), protocol latency (Table 5), and large‑scale training throughput (Table 4) directly support the MPFT + EP design and the routing/overlap strategies. The KV cache and GFLOPs/token tables (1–2) quantify MLA and MoE efficiency.
  - For end‑task quality, this paper references the technical report [26]; it focuses on systems efficiency rather than benchmarks like MMLU or coding/math datasets.
  - The 14.76 ms and 0.82 ms TPOT are theoretical best‑cases under aggressive overlap assumptions (Sec. 2.3.2); they motivate network design but are not empirical end‑to‑end latency numbers.

## 6. Limitations and Trade-offs
- Architectural and hardware assumptions
  - The routing/co‑design assumes strong intra‑node bandwidth (NVLink) and multiple NICs per node (H800 with eight 400 Gbps IB NICs, Fig. 2). Other platforms with different scale‑up/scale‑out ratios may need different routing policies.
  - Dual micro‑batch overlap and prefill/decode disaggregation assume batched, mixed workloads (Sec. 2.3.1); single‑stream, low‑batch decoding will see less overlap benefit.
- FP8 constraints on current GPUs (Sec. 3.1.1)
  - Accumulation precision inside Tensor Cores is limited (e.g., 13 fraction bits accumulated into FP22 registers), which can hurt stability; the paper calls for configurable or FP32 accumulation (Sec. 3.1.2).
  - Fine‑grained scaling introduces dequantization overhead as partial results move between Tensor Cores and CUDA cores (Sec. 3.1.1).
- Communication compression trade‑offs (Sec. 3.2)
  - LogFMT improves quantization quality at same bit‑width but encode/decode adds “50%–100%” overhead on current GPUs; thus not deployed (Sec. 3.2.1).
  - Combine remains BF16 for accuracy; pushing combine to FP8/E5M6 may incur minor quality loss (work in progress, Sec. 3.2).
- Scale‑up/scale‑out mismatch and SM contention (Sec. 4.4.1)
  - Due to unbalanced NVLink vs IB bandwidths, the software pipeline uses GPU SMs to forward, reduce, and manage data layouts, consuming “up to 20 SMs” per H800 for communication chores during training. This steals compute from kernels.
- Bandwidth contention during inference (Sec. 4.5.1)
  - PCIe/NVLink traffic (e.g., CPU↔GPU KV cache movement) can contend with EP all‑to‑all, causing latency spikes. Dynamic prioritization is not well supported by today’s interconnects (Sec. 4.5.2).
- Multi‑plane deployment gaps (Sec. 5.1)
  - The ideal MPFT requires NICs with multiple bonded ports and native out‑of‑order placement (Fig. 4). ConnectX‑7 lacks this; cross‑plane traffic needs intra‑node forwarding, adding extra hops/latency in some inference paths.
- Evaluation scope
  - This paper emphasizes systems efficiency. Comprehensive model quality results, safety, and long‑context accuracy trade‑offs are in the technical report [26]; they are not reproduced here.

## 7. Implications and Future Directions
- How this work shifts the field
  - It demonstrates that with the right co‑design, a 2,048‑GPU H800 cluster can train and serve frontier‑class MoE models cost‑effectively by:
    - Slashing KV cache with MLA (Table 1).
    - Keeping compute/token low with MoE (Table 2).
    - Making FP8 training practical at scale (Sec. 3.1; Fig. 1).
    - Turning the network into a first‑class design element (Node‑Limited Routing, MPFT, DeepEP; Secs. 4–5).
  - It reframes “bigger clusters only” into “smarter co‑design,” broadening who can build advanced LLMs.
- Enabled applications
  - On‑prem and personal agents: MoE activates only a small parameter subset per token (e.g., V2: 21B active of 236B; V3: 37B active of 671B; Sec. 2.2.1), enabling high TPS on commodity servers; the paper notes nearly “20 TPS” on a ~$10k consumer‑GPU server using KTransformers (Sec. 2.2.2).
  - Reasoning models and RL fine‑tuning benefit from higher tokens/sec (Sec. 2.3.4).
- Concrete hardware directions distilled from the bottlenecks observed
  - Precision and quantization support (Secs. 3.1.2, 3.2.2)
    - Configurable or FP32 accumulation in Tensor Cores.
    - Native fine‑grained scaling inside Tensor Cores (group‑scale GEMMs).
    - Built‑in compression/decompression units (FP8/custom formats like LogFMT) on NICs/I/O dies.
  - Unifying scale‑up and scale‑out fabrics (Sec. 4.4.2)
    - A “Unified Network Adapter” or I/O die that speaks both intra‑node (NVLink‑class) and inter‑node (IB/Ethernet) and can forward packets to specific GPUs with policy routing.
    - Dedicated communication co‑processors to offload packet handling, memory copies, type casts, and reduce/broadcast to hardware.
    - Hardware synchronization primitives with memory semantics (acquire/release) to fix ordering without software fences (Sec. 4.4.2; expanded as “Region Acquire/Release” in Sec. 6.4).
    - Dynamic interconnect QoS and prioritization across EP/TP/KV traffic; integrate NICs on I/O dies and connect CPU↔GPU via scale‑up fabric to remove PCIe bottlenecks (Sec. 4.5.2).
  - Ethernet as a competitive AI fabric (Sec. 5.2.2)
    - Specialized low‑latency RoCE switches (e.g., Slingshot‑like, Broadcom AIFH‑style).
    - Adaptive routing (packet spraying) instead of pure ECMP; better congestion control (VOQ, RTT‑based CC, programmable CC).
  - In‑network compute and compression (Sec. 6.5)
    - Hardware multicast for EP dispatch and small‑scope in‑network reduce for EP combine.
    - LogFMT‑class compression supported natively in the fabric.
  - Memory‑centric accelerators (Sec. 6.6)
    - DRAM‑stacked accelerators (e.g., SeDRAM‑like) for memory‑bound inference.
    - System‑on‑Wafer integration to push bandwidth and capacity for ultra‑large models.
  - Reliability (Sec. 6.1)
    - Stronger error detection beyond ECC (checksums, hardware redundancy), plus vendor‑supplied diagnostic toolkits to catch silent data corruption.

In sum, the paper provides a detailed, reproducible playbook for making large MoE LLMs efficient on today’s constrained hardware, and it translates concrete pain points into a clear set of hardware features the ecosystem can build toward. If you are designing next‑gen AI systems or fabrics, the recommendations in Secs. 3–6 form a prioritized roadmap grounded in real bottlenecks observed while training and serving DeepSeek‑V3/R1.
