# DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving

**ArXiv:** [2401.09670](https://arxiv.org/abs/2401.09670)

## 🎯 Pitch

DistServe pioneers a novel serving architecture for large language models by separating the 'prefill' and 'decoding' stages onto different GPUs and independently optimizing their resource allocation and parallelism. This disaggregation eliminates interference between the two crucial phases, enabling up to 7.4× higher request throughput or meeting 12.6× stricter latency service-level objectives compared to leading systems—dramatically lowering operational costs and ensuring top-tier response quality for latency-critical applications like chatbots and coding assistants.

---

## 1. Executive Summary

DistServe introduces a serving architecture that **disaggregates the prefill and decoding phases** of LLM inference onto separate GPUs, co-optimizing resource allocation and parallelism strategies for each phase independently to maximize **per-GPU goodput** — the maximum request rate that can be served while meeting both time-to-first-token (TTFT) and time-per-output-token (TPOT) latency constraints for a target SLO attainment. Evaluated across OPT models (13B–175B) on chatbot, code completion, and summarization workloads, DistServe achieves up to **7.4× higher request rates** and sustains **12.6× tighter SLO scales** compared to state-of-the-art colocated serving systems, establishing that disaggregation eliminates the prefill-decoding interference and resource coupling that force existing systems to over-provision — though only when sufficient bandwidth exists to absorb the KV cache transmission between phases and when the base model's per-phase latency characteristics can be profiled in advance.

## 2. Context and Motivation

### The Core Problem: LLM Inference Is a Dual-Phase Process with Conflicting Requirements

The fundamental challenge this paper tackles is that serving large language models (LLMs) involves two computationally distinct phases that existing systems treat as a single, unified workload. When a user submits a prompt, the **prefill phase** processes all input tokens in parallel to produce the first token of the response. After that, the **decoding phase** generates subsequent tokens one at a time, with each new token depending on all previously generated tokens. This split is inherent to autoregressive generation and cannot be circumvented — but the paper argues that *forcing these two phases to share the same GPUs creates inherent conflicts* that existing systems cannot resolve.

Why does this matter? The two phases have fundamentally different computational shapes:

- **Prefill is compute-bound** when the input prompt is non-trivial (roughly 512+ tokens for a 13B model on an A100 GPU). It performs a large number of matrix multiplications in parallel across all input tokens, saturating the GPU's floating-point units. Its performance directly determines TTFT — how long the user waits before seeing the first response token.

- **Decoding is memory-bandwidth-bound** because each step processes only a single new token but must load the entire model weights and the accumulated KV cache from GPU memory. Its performance determines TPOT — the average time between subsequent tokens.

These two phases have *different latency requirements across different applications.* As the paper notes (Section 1):

> "Different applications place varying demands on each metric. For example, real-time chatbots prioritize low TTFT for response promptness, while TPOT only remains important until it is faster than human reading speed (i.e., 250 words/min). Conversely, document summarization emphasizes low TPOT for faster generation of the summary."

The serving system, then, must satisfy *both* constraints simultaneously while minimizing cost per query. This is what the paper formalizes as **per-GPU goodput** — the maximum request rate at which the system meets SLO attainment targets (e.g., 90% of requests satisfy both TTFT and TPOT requirements) on each GPU provisioned. Higher per-GPU goodput means fewer GPUs (and therefore lower cost) to serve a given traffic load.

The problem is that optimizing one phase's latency tends to *conflict with* optimizing the other's, creating a fundamental tension that the paper argues cannot be adequately resolved within the colocated architecture used by all major LLM serving systems prior to this work.

### Why This Problem Is Important

The paper identifies several drivers that make this optimization critical:

**Cost pressure on LLM services.** Deploying LLMs at scale requires massive GPU clusters. The paper cites Reuters (2023) noting that processing an end-to-end LLM query can be "substantially slower than a standard search query" — and since LLM services compete with traditional search engines and other Internet services on response time, providers face pressure to over-provision GPUs to meet latency targets. Any architecture that increases per-GPU goodput directly reduces the number of GPUs needed, translating to lower operational costs.

**Heterogeneous application requirements.** LLMs are deployed across a widening range of use cases — real-time chatbots, code completion assistants, document summarization, and more — each with different sensitivity to TTFT versus TPOT. A chatbot user expects the first token quickly (TTFT) but is tolerant of inter-token delays as long as they're below human reading speed. A code completion engine needs low TTFT for interactive use and low TPOT to produce completions rapidly. A summarization system may tolerate a long initial wait (since the user understands the document is long) but then expects rapid token generation. A one-size-fits-all serving architecture that treats all applications identically will inevitably over-provision for some and under-deliver for others.

**The gap between what colocated systems achieve and what is possible.** Figure 1 in the paper provides the motivating example in stark terms. When serving a 13B LLM on an A100 GPU with a synthetic workload (512-token input, 64-token output) and latency requirements emulating a summarization task, existing colocated systems achieve roughly **1.6 requests per second (rps)** before violating SLOs. But when the prefill and decoding phases are served independently on separate GPUs, the prefill-only system achieves **5.6 rps** and the decode-only system achieves **10 rps**. By allocating 2 GPUs for prefill and 1 for decoding, the overall system could handle 10 rps — equivalent to **3.3 rps per GPU**, or **2.1× the per-GPU goodput** of the colocated system. This gap represents wasted GPU capacity and unnecessary cost.

**The scaling trend makes this worse.** As LLMs grow larger (the paper evaluates up to 175B parameters) and context lengths increase (with newer models reaching 1M+ token windows), the computational disparity between prefill and decoding widens — prefill becomes more compute-intensive relative to decoding, and the interference between them intensifies. The paper argues that the colocated architecture's problems compound at scale.

### Where Existing Approaches Fall Short

The paper situates existing LLM serving systems along two dimensions: how they batch requests and how they parallelize computation. It identifies specific limitations in both.

#### Continuous Batching: Maximizing Throughput at the Expense of Latency

The dominant technique in modern LLM serving systems is **continuous batching** (introduced by Orca, Yu et al., 2022, and adopted by vLLM, DeepSpeed-MII, and others). Traditional static batching waits for all requests in a batch to complete before moving to the next batch, which wastes GPU time when some requests finish early. Continuous batching instead dynamically adds new requests into the current batch as slots free up, keeping the GPU busier and improving overall throughput (total tokens generated per second across all users).

However, the paper identifies a critical flaw:

> "Colocating and batching the prefill and decoding computation to maximize the overall system throughput... is cost-effective for service providers. However, in the presence of SLOs, present approaches struggle to maintain both high service quality and low cost."

The problem is that continuous batching groups prefill and decoding jobs together in the same GPU execution, and these jobs have *vastly different execution times*. Figure 2 demonstrates this: adding a single prefill job (processing 1024 input tokens) to a batch of decoding requests increases the batch execution time dramatically. The decoding jobs in that batch must wait for the lengthy prefill to complete, inflating TPOT. Conversely, adding decoding jobs to prefill extends TTFT because the GPU's compute resources are now shared.

This is what the paper calls **prefill-decoding interference** — a direct consequence of two operations with different computational profiles sharing the same hardware. The interference is not marginal: Figure 2 shows that with a 1024-token input, adding one prefill to a decoding batch roughly doubles the latency, and this effect worsens with longer inputs and larger batches.

#### Chunked Prefill with Piggyback: Trading One Problem for Another

The paper specifically analyzes SARATHI (Agrawal et al., 2023), implemented in DeepSpeed-MII, which attempts to mitigate prefill-decoding interference through **chunked prefill with piggyback**. The idea: instead of processing a long prefill in one shot (which would block all concurrent decoding jobs), split it into smaller chunks and interleave those chunks with decoding jobs. This reduces the maximum decoding delay but introduces new problems:

**1. It trades TTFT for TPOT, but doesn't eliminate the tradeoff.** As the paper explains (Section 2.3):

> "If the chunk size is set much lower than the inflection point that can saturate the GPU, then the prefill job will have a longer execution time since it competes with the decoding job in the same batch and cannot solely utilize the GPU resources."

A small chunk size means prefill never fully uses the GPU's compute capacity, extending TTFT. A large chunk size means fewer opportunities to piggyback decoding jobs, reducing TPOT improvement.

**2. It introduces O(N²) KV cache reload overhead.** When a prefill is split into N chunks, each subsequent chunk must reload the KV caches from all previous chunks from HBM to SRAM to compute attention. The total KV cache memory traffic becomes:

$$\text{KV cache loads} = N + (N-1) + \ldots + 1 = O(N^2)$$

compared to O(N) for processing the prefill in a single pass. This overhead grows quadratically with the number of chunks, making the approach increasingly expensive for longer contexts.

**3. The interference is reduced, not eliminated.** Even with piggyback, some decoding jobs still wait on prefill chunks, and prefill still competes with decoding for memory bandwidth. The paper's experimental results (Figure 8, right column) confirm that while DeepSpeed-MII improves over naive continuous batching (vLLM), it still underperforms disaggregation significantly — by 1.6×–7.4× in request rate on chatbot workloads.

#### Sequential Scheduling: Ineffective Due to Queuing Delays

One might ask: why not simply unbatch prefill and decoding and run them sequentially? The paper explains why this doesn't work (Section 2.3):

> "Unbatching prefill and decoding jobs and scheduling them sequentially does not mitigate the interference. Decoding jobs may experience longer queuing delays due to waiting for ongoing prefill jobs on GPUs. Moreover, batches dedicated to decoding often lead to GPU underutilization."

If the system prioritizes prefill, decoding jobs queue up behind prefill jobs and experience high TPOT. If it prioritizes decoding, prefill jobs queue and TTFT spikes. There's no scheduling policy within a colocated architecture that can satisfy both latency requirements simultaneously without over-provisioning.

#### Resource and Parallelism Coupling: One Size Must Fit Two Different Phases

Beyond scheduling, the paper identifies a deeper structural problem: when prefill and decoding share GPUs, they necessarily share resource allocation and parallelism strategies. Yet each phase has fundamentally different preferences for how resources and parallelism should be configured:

- **Prefill benefits from intra-operator parallelism** (tensor parallelism) because it's compute-bound. Splitting large matrix multiplications across GPUs accelerates execution time, directly reducing TTFT. However, this requires high-bandwidth interconnects (NVLINK) and introduces communication overhead that grows with model size.

- **Decoding benefits from inter-operator parallelism** (pipeline parallelism) when batch sizes are large enough, because it can linearly scale throughput with additional GPUs. But at small batch sizes (common under stringent TPOT requirements), it may benefit more from intra-op parallelism to reduce per-step latency.

- **The optimal batch size differs:** Prefill saturates GPU compute with relatively few requests (a single 512-token sequence for a 13B model), so large prefill batches don't help and only increase queuing delay. Decoding is heavily bandwidth-bound for small batches and *requires* batching many requests to achieve high GPU utilization — but the batch size is constrained by GPU memory (KV caches) and by the TPOT SLO (larger batches increase per-step latency).

When colocated, the system must pick *one* parallelism configuration that serves both phases. The paper states (Section 2.3):

> "In existing systems, due to coupling, resource allocation and parallelism plans are tailored to satisfy the more demanding of TTFT and TPOT, which may not be ideal for the other. This often leads to resource over-provisioning to meet both SLOs."

For instance, if TPOT is the tighter constraint, the system might allocate more GPUs with inter-op parallelism to increase decoding throughput — but this configuration may not reduce prefill latency enough for the TTFT requirement, forcing the system to add even more GPUs (or replicate instances) until both slack are met. The coupling prevents fine-grained optimization.

#### Model Parallelism in Serving: Under-explored for Goodput

The paper notes that AlpaServe (Li et al., 2023) explored using model parallelism to improve LLM serving throughput through statistical multiplexing — spreading requests across pipelined GPUs to smooth out variations in execution time. However:

> "It only targets the non-autoregressive generation."

AlpaServe operates in a setting where the model produces the entire output in one shot (or processes a batch of prefill-only requests), avoiding the prefill-decode interaction entirely. Its insights don't directly transfer to the autoregressive setting where prefill and decoding interleave.

Existing production systems (vLLM, DeepSpeed-MII, TensorRT-LLM) primarily offer intra-operator parallelism (tensor parallelism) but don't systematically search over the combined space of intra- and inter-operator parallelism strategies for each phase. This is partly because, in a colocated architecture, the search space is inherently constrained — the parallelism strategy must work for both phases simultaneously, limiting the potential gains from specialized configurations.

### How This Paper Positions Itself

DistServe proposes that the colocated architecture is fundamentally the wrong starting point. The paper's central claim is that **disaggregating prefill and decoding onto separate GPUs** eliminates both the interference and coupling problems at once:

1. **No interference:** Since prefill and decoding never share a GPU, neither phase blocks the other. Prefill instances focus solely on processing prompts quickly; decoding instances focus on generating tokens efficiently.

2. **Independent scaling:** Each phase can use its own parallelism configuration tailored to its computational characteristics and its specific latency SLO. The paper's placement algorithms (Section 4) search over this expanded space to find the per-phase configuration that maximizes per-GPU goodput.

3. **Independent batching:** Prefill instances can batch only when it improves goodput (i.e., when input lengths are below the compute-bound threshold). Decoding instances can accumulate larger batches without prefill interference, approaching the compute-bound regime where inter-op parallelism becomes beneficial.

The cost is **KV cache transmission** between prefill and decoding instances — the intermediate states generated during prefill must be sent to the decoding instance that will continue the request. The paper acknowledges this overhead (Section 3.3) but argues it's manageable: for a 512-token request on OPT-66B, the KV cache is ~1.13 GB. At 10 rps, that's ~90 Gbps — well within the capacity of NVLINK (600 GB/s between A100 GPUs) or InfiniBand (800 Gbps) in modern clusters. The placement algorithms are explicitly designed to ensure KV cache transfer occurs over high-bandwidth links.

The paper positions disaggregation not as an incremental improvement but as an architectural shift that **expands the optimization space** enough to enable the 2–7× goodput improvements demonstrated in evaluation. The key insight is that the constraints of colocation are not just inconvenient — they're the *primary bottleneck* limiting per-GPU goodput, and removing them through disaggregation enables optimizations that were previously impossible.

Importantly, the paper frames this work in the context of **goodput optimization**, drawing a connection to prior systems like Pollux (Qiao et al., 2021) for training workloads and Clockwork (Gujarati et al., 2020) for traditional DNN serving:

> "Optimizing goodput is a hot topic in DL applications... DistServe is the first work to optimize the goodput for autoregressive LLM inference."

This positions DisServe as filling a specific gap: extending goodput-optimized serving from non-autoregressive or training settings to the more challenging autoregressive inference case, where the dual-phase nature creates unique optimization opportunities that prior systems couldn't exploit.

## 3. Technical Approach

### 3.1 Reader Orientation

DistServe is a serving system that runs the two phases of LLM inference — prefill (processing the input prompt) and decoding (generating output tokens one by one) — on **separate sets of GPUs**, with each phase independently configured for its own parallelism strategy and resource allocation. It solves the problem that colocating these two phases on the same GPUs creates interference (prefill blocks decoding and vice versa) and forces both phases to share the same parallelism configuration, which is suboptimal because prefill is compute-bound and benefits from tensor parallelism while decoding is memory-bandwidth-bound and benefits from different strategies depending on batch size. The solution takes the shape of a **two-level optimization**: first, a placement algorithm searches over all feasible parallelism configurations for each phase independently, using a simulator to estimate the resulting per-GPU goodput (the maximum request rate that meets latency SLOs for a target percentage of requests), and selects the configuration that maximizes goodput; second, the selected configurations are replicated and deployed onto physical GPU nodes with a bandwidth-aware placement that keeps KV cache transmission on high-speed intra-node links.

### 3.2 Big-Picture Architecture (Diagram in Words)

The DistServe system consists of five major components working together:

1. **Placement Algorithm Module** — An offline optimizer that takes the model specification, workload characteristics, latency SLOs (TTFT and TPOT requirements), SLO attainment target (e.g., 90%), and cluster topology as input. It searches over parallelism configurations (intra-op and inter-op degrees) for both prefill and decoding phases, uses a custom simulator to estimate the per-GPU goodput of each configuration under the given workload, and outputs a **placement plan**: the number of prefill instances, the number of decoding instances, their parallelism strategies, and which physical GPUs they should occupy. This module runs once before deployment (or on workload shifts detected by the profiler) and takes under 1.3 minutes in the largest evaluated setting.

2. **Centralized Controller** — A runtime component that receives all incoming requests via a RESTful API (OpenAI-compatible). It implements a simple First-Come-First-Served (FCFS) dispatching policy: each new request is routed to the prefill instance with the currently shortest queue. After the prefill instance completes and produces the first token and KV cache, the controller dispatches the request to the least-loaded decoding instance for subsequent token generation. The controller also coordinates KV cache transmission between instances.

3. **Prefill Instances** — GPU workers (powered by a parallel execution engine built on Ray actors) that each hold a complete copy of the LLM weights and execute only the prefill phase. Upon receiving a request, a prefill instance processes all input tokens in parallel through the model, produces the first output token, generates the KV cache for all input positions, and retains this KV cache in GPU memory. It does NOT perform any decoding steps. Multiple prefill instances can feed into a single decoding instance to accumulate larger decoding batches. Prefill instances are configured with the parallelism strategy chosen by the placement algorithm (e.g., 3-way inter-op × 3-way intra-op for OPT-175B on ShareGPT).

4. **Decoding Instances** — GPU workers that each hold a complete copy of the LLM weights and execute only the decoding phase. A decoding instance receives KV caches and first tokens from prefill instances (via a "pull" mechanism — the decoding instance fetches KV caches as needed rather than having them pushed), then generates subsequent tokens autoregressively using continuous batching across all active requests. Because no prefill jobs compete for GPU resources, decoding instances can accumulate larger batches without inflating TPOT. Their parallelism configuration is independently optimized (e.g., 4-way intra-op × 3-way inter-op for OPT-175B on ShareGPT).

5. **Workload Profiler** — A monitoring component that tracks key workload parameters (average input/output lengths, arrival rate distribution) over time. If it detects a significant shift in workload patterns, it triggers a re-run of the placement algorithm to produce a new optimized configuration, which is then applied by reloading model weights onto the reconfigured instances. This replanning happens on the timescale of hours, while the algorithm itself runs in seconds to minutes.

**Information flow at runtime:** A request arrives at the controller → the controller dispatches it to the prefill instance with the shortest queue → the prefill instance processes the prompt, producing the first token and KV cache → the KV cache remains in the prefill instance's GPU memory as a buffer → the controller assigns a decoding instance → the decoding instance pulls the KV cache from the prefill instance over NVLINK or InfiniBand → the decoding instance generates subsequent tokens, streaming them back to the client, until a termination token is produced.

### 3.3 Roadmap for the Deep Dive

- **First**, the formal definition of per-GPU goodput and the optimization objective, since everything else in the system is designed to maximize this quantity.
- **Second**, the simulator that estimates SLO attainment for a given configuration, because both placement algorithms depend on it and it embodies the paper's analytical latency models for prefill and decoding execution.
- **Third**, the high node-affinity placement algorithm (Algorithm 1), which is the simpler case and introduces the core search methodology — enumerating parallelism configurations, simulating goodput, and replicating.
- **Fourth**, the low node-affinity placement algorithm (Algorithm 2), which adds the constraint that prefill and decoding instances must share the same node to use NVLINK for KV cache transfer, requiring co-optimization of intra-node configurations.
- **Fifth**, the online scheduling enhancements — pipeline bubble reduction, the "pull" mechanism for KV cache transfer, and periodic replanning — which address practical deployment issues that the analytical models and offline optimization don't fully capture.

### 3.4 Detailed, Sentence-Based Technical Breakdown

This is primarily a **systems design paper** whose core idea is that a disaggregated architecture, combined with an automated search over per-phase parallelism configurations guided by a workload-aware simulator, eliminates the prefill-decoding interference inherent in colocated serving systems and enables substantially higher per-GPU goodput than is possible under colocation.

---

#### The Optimization Objective: Per-GPU Goodput

The paper does not state a single closed-form objective function, but the optimization target is defined conceptually throughout Sections 1, 2, and 4. The system aims to maximize **per-GPU goodput**, which is:

$$\text{Goodput} = \max\left\{ R \;\middle|\; \text{SLO\_attainment}(R, \text{TTFT}_{\text{SLO}}, \text{TPOT}_{\text{SLO}}, \text{config}, W) \geq A_{\text{target}} \right\}$$

where $R$ is the request arrival rate, $\text{TTFT}_{\text{SLO}}$ and $\text{TPOT}_{\text{SLO}}$ are the application's latency requirements, $\text{config}$ is the set of parallelism strategies and instance counts for both prefill and decoding phases, $W$ characterizes the workload distribution (arrival process, input length distribution, output length distribution), and $A_{\text{target}}$ is the SLO attainment target (typically 90% or 99% — the fraction of requests that must satisfy both TTFT and TPOT constraints).

**What it computes:** For a given system configuration and workload, this finds the highest sustainable request rate at which the fraction of requests meeting both latency SLOs remains at or above the target attainment percentage. The per-GPU goodput is then this maximum rate divided by the total number of GPUs used by the configuration. Higher per-GPU goodput means fewer GPUs are needed to serve a given traffic load, translating directly to lower cost per query.

**Why this form:** This formulation captures the fundamental tradeoff in LLM serving: throughput (total tokens per second) is not the right metric because it can be maximized by packing requests into large batches, which inflates individual request latency. Goodput explicitly couples throughput with latency constraints, matching the real-world requirement that service providers must meet SLOs while minimizing cost. The SLO attainment target (90% or 99%) acknowledges that tail latencies are unavoidable in stochastic systems and that service-level agreements typically permit a small fraction of violations.

The placement algorithms don't solve this optimization analytically. Instead, they enumerate feasible configurations, use a discrete-event simulator to estimate $\text{SLO\_attainment}(R, \ldots)$ for each configuration, and then binary-search over $R$ to find the maximum rate that meets the attainment target. The per-GPU goodput is computed by dividing this rate by the GPUs used, and the configuration with the highest per-GPU goodput is selected.

---

#### The Simulator: Estimating SLO Attainment Without Real Deployment

The placement algorithms in Sections 4.1 and 4.2 both depend on a simulator that can estimate, for a given parallelism configuration and workload distribution, what fraction of requests will meet the TTFT and TPOT SLOs at a given arrival rate. Building this simulator requires two components: **(1) an analytical latency model** that predicts the execution time of prefill and decoding operations on the target GPU, and **(2) a discrete-event simulation** that models request arrivals, queuing, batching, and scheduling.

**Why a simulator rather than analytical queuing models:** The paper's analysis in Section 3 uses M/D/1 queuing theory to derive insights about parallelism preferences (Equations 1–3), but these analytical models assume uniform input lengths and simple FCFS scheduling without batching. Real workloads have variable input/output lengths, and the prefill phase may batch multiple short requests or process long requests individually depending on the compute-bound threshold $L_m$. A discrete-event simulator can model these complexities accurately while remaining fast enough for an inner loop of configuration search.

##### Analytical Latency Model for LLM Inference

The simulator's core is an analytical latency model (detailed in Appendix A) that predicts execution time for both prefill and decoding steps given the model architecture, batch composition, and GPU characteristics. The model focuses on matrix multiplications (GEMMs), which dominate LLM inference latency, and treats attention operations separately due to their specialized FlashAttention kernels.

**Model symbols (architecture):**
- $h$: hidden size (divided by tensor parallelism degree if used)
- $n$: number of attention heads (divided by tensor parallelism degree)
- $s$: head size, where $h = n \cdot s$
- $m$: FFN intermediate size (divided by tensor parallelism degree)

**Batch symbols:**
- $B$: batch size (number of requests)
- $l_0, l_1, \ldots, l_{B-1}$: input length of each request in the batch
- $t$: total number of tokens in the batch, $t = \sum_{i=0}^{B-1} l_i$
- $t_2$: squared sum of input lengths, $t_2 = \sum_{i=0}^{B-1} l_i^2$
- $b$: block size in FlashAttention kernel (typically 16 or 32)

**Prefill phase latency model:**

The prefill phase involves four GEMM operations (QKV projection, attention output projection, FFN input, FFN output) and one attention operation. The GEMMs have arithmetic intensity $O(t)$, and since $t$ typically reaches several hundred tokens, they are compute-bound on A100 GPUs. Their latency is modeled from total FLOPs:

$$T_1 = C_1 \cdot (4th^2 + 2thm)$$

where $C_1$ is a hardware-specific coefficient determined through profiling, $4th^2$ accounts for the QKV, attention output, and FFN input projections (each $th^2$, plus the QKV having $3h$ output dimension: $t \cdot h \cdot 3h = 3th^2$, and attention output: $t \cdot h \cdot h = th^2$, totaling $4th^2$), and $2thm$ accounts for the two FFN GEMMs (input: $t \cdot h \cdot m = thm$, output: $t \cdot m \cdot h = thm$).

For the attention operation with FlashAttention, the kernel processes each request's tokens independently. For one head and a request of length $l$, it performs approximately $2sl + 3sl \cdot (l/b) \approx 3sl \cdot (l/b)$ memory accesses and $2sl^2 + sl(l/b) \approx 2sl^2$ FLOPs, giving arithmetic intensity $\approx 2b/3$. With $b=16$ this is 10.677, and with $b=32$ it is 21.333 — both well below the A100's compute-bound threshold of ~156, making attention memory-bound. The total attention latency across all heads and requests is:

$$T_2 = C_2 \cdot n \cdot \sum_{i=0}^{B-1} \frac{3sl_i^2}{b} = C_2 \cdot \frac{3nst_2}{b} = C_2 \cdot \frac{3ht_2}{b}$$

where $C_2$ is a memory-bandwidth coefficient.

Total prefill latency:

$$T_{\text{Prefill}} = C_1 \cdot (4th^2 + 2thm) + C_2 \cdot \frac{3ht_2}{b} + C_3$$

where $C_3$ captures fixed overheads (Python runtime, kernel launch, system noise).

**What this computes:** Given a batch of prefill requests with varying input lengths, this predicts the wall-clock time to process all of them through the model in one forward pass. The first term models the compute-bound GEMM time (scaling with total tokens $t$), the second term models the memory-bound attention time (scaling with squared lengths $t_2$ because attention is quadratic in sequence length), and the third term captures constant overhead.

**Why this form:** The separation into compute-bound and memory-bound components reflects the fundamental distinction on GPUs: operations with high arithmetic intensity (FLOPs per byte loaded) are limited by compute throughput and modeled by FLOP-count, while operations with low arithmetic intensity are limited by memory bandwidth and modeled by data-movement volume. Using $t_2$ (sum of squared lengths) rather than $t^2$ (square of sum) correctly accounts for the fact that FlashAttention processes each request's attention independently — a batch of one 1024-token request has very different attention cost from a batch of four 256-token requests even though both have $t=1024$.

**Decoding phase latency model:**

In decoding, each step processes $B$ requests where each request contributes exactly one new token. The GEMMs have arithmetic intensity $O(B)$, and since $B$ is typically much smaller than model dimensions, they are memory-bound. The total memory traffic for the four GEMMs is approximately $8Bh + 4h^2 + 2hm + 2Bm$. Since model dimensions dominate batch size ($h, m \gg B$), the constant $4h^2 + 2hm$ terms dominate:

$$T_3 = C_4 \cdot (4h^2 + 2hm)$$

For the decoding attention with FlashAttention, each request of length $l_i$ (total tokens generated so far) requires approximately $3sl_i$ memory accesses and $2sl_i$ FLOPs, remaining memory-bound:

$$T_4 = C_5 \cdot n \cdot 3s \sum_{i=0}^{B-1} l_i = C_5 \cdot 3ht$$

Total decoding step latency:

$$T_{\text{Decoding}} = C_4 \cdot (4h^2 + 2hm) + C_5 \cdot 3ht$$

where the overhead term is absorbed into $C_4$ since $4h^2 + 2hm$ is already a constant.

**What this computes:** For a single decoding step, this predicts the time to generate one new token for each of the $B$ active requests in the batch. The first term is constant per step (dominated by loading model weights), and the second term grows linearly with the total KV cache length $t$ across all batched requests (because attention must access all previous tokens' KV entries).

**Why this form:** The constant term explains why decoding benefits from larger batches — the fixed cost of loading weights is amortized across more requests. However, the $C_5 \cdot 3ht$ term explains why TPOT increases with batch size: as more requests are batched together, the total KV cache that must be traversed grows, increasing attention latency. This creates an optimal batch size that balances GPU utilization against per-step latency, which the simulator can capture by modeling each decoding step's composition.

The coefficients $C_1$ through $C_5$ are determined through profiling: the system measures actual execution times for a few known batch configurations, then interpolates to find the coefficient values. The paper reports in Section 6.4 that the simulator's SLO attainment predictions match real measurements within 2% error across all tested configurations, confirming the model's accuracy.

##### Discrete-Event Simulation

Armed with the analytical latency model, the simulator proceeds as follows:

1. **Workload generation:** The workload distribution $W$ (arrival process, input/output length distributions) is fitted from historical request traces. The simulator samples new traces from this distribution to avoid overfitting to a specific trace. The arrival process is modeled as Poisson with rate $R$.

2. **Instance modeling:** Each instance (prefill or decoding) is modeled as a queue with its specific parallelism configuration. The parallelism configuration determines how many GPUs form the instance and how the model layers are partitioned (affecting the per-step execution time via the analytical model) and the memory capacity (determining maximum batch size based on KV cache storage).

3. **Prefill scheduling:** For prefill instances, the batch composition logic follows the principle from Section 3.1: if a single request's input length exceeds the compute-bound threshold $L_m$ (determined by profiling the specific model and GPU), it is scheduled alone. If shorter, multiple requests are batched together to approach $L_m$ total tokens. The scheduling policy is FCFS within each instance, and the controller dispatches to the instance with the shortest queue.

4. **Decoding scheduling:** For decoding instances, continuous batching is used — new requests join the active batch as soon as they arrive from prefill, and completed requests leave immediately. The batch size is limited by GPU memory (KV cache capacity) and can grow until either memory is exhausted or the per-step latency (from the analytical model) would violate TPOT.

5. **Event processing:** The simulator advances through discrete events: request arrivals, prefill completions, KV cache transfers, decoding step completions, and request completions. At each event, it records whether the request's observed TTFT and TPOT meet the SLOs. After processing the full trace, it computes the fraction of requests that met both SLOs, producing the $\text{SLO\_attainment}$ estimate.

6. **Binary search for goodput:** To find the maximum rate for a configuration, the simulator performs binary search over $R$: it starts with a range, tests the midpoint rate, and narrows the range based on whether the attainment target is met. The maximum rate meeting the target, divided by the total GPUs in the configuration, gives the per-GPU goodput.

**Design choice — simulator over analytical model:** The paper could have attempted a unified queuing-theoretic analysis like the M/D/1 model in Section 3.1, but this would break down under variable-length requests, batching, and the interaction between prefill and decoding instances. The simulator captures these real-world complexities while remaining fast enough for the configuration search (minutes, not hours). The accuracy verification in Section 6.4 (Table 2) validates this choice: the simulator's predictions differ from real measurements by at most 2%.

---

#### Placement Algorithm for High Node-Affinity Clusters (Algorithm 1)

When the deployment cluster has high cross-node bandwidth (e.g., InfiniBand at 800 Gbps), KV cache transmission between any two GPUs is fast enough to be negligible. In this setting, prefill and decoding instances can be placed arbitrarily across the cluster, and the optimization decouples: find the best parallelism configuration for prefill instances independently, find the best for decoding instances independently, then replicate both to meet the target traffic rate.

**Inputs to Algorithm 1:**
- $G$: the LLM specification (architecture, weight sizes)
- $N$: maximum number of nodes a single instance can span
- $M$: number of GPUs per node
- $C$: GPU memory capacity
- $W$: workload distribution
- $R$: target traffic rate

**Algorithm structure (paraphrased from the paper):**

The algorithm enumerates all feasible $(inter\_op, intra\_op)$ pairs for a single instance, where $intra\_op$ ranges from 1 to $M$ (the GPUs within one node, since intra-op parallelism requires high-bandwidth NVLINK typically available only intra-node) and $inter\_op$ ranges from 1 to $\lfloor N \times M / intra\_op \rfloor$ (total GPUs across nodes divided by the intra-op group size). For each configuration, it checks that the model fits in GPU memory: the per-GPU memory requirement is $G.size / (inter\_op \times intra\_op)$, which must be less than $C$.

For each feasible configuration, it applies the parallelism to create a partitioned model $\hat{G}$, then runs two separate simulations:

- `simu_prefill`($\hat{G}, W$): Treats $\hat{G}$ as a prefill-only instance. The simulator runs the workload $W$ through this instance configuration, varying the arrival rate via binary search to find the maximum rate that meets the TTFT SLO attainment target. This maximum rate, divided by the number of GPUs in the configuration, is the per-GPU goodput for prefill under this config. The algorithm tracks the configuration `configp` with the highest per-GPU prefill goodput.

- `simu_decode`($\hat{G}, W$): Same process for a decoding-only instance, finding the maximum rate meeting the TPOT SLO attainment target, and tracking the configuration `configd` with the highest per-GPU decoding goodput.

After finding the best per-phase configurations independently, the algorithm determines replication counts: to serve the target traffic rate $R$, it needs $n = \lceil R / configp.goodput \rceil$ prefill replicas and $m = \lceil R / configd.goodput \rceil$ decoding replicas. The final placement is $(n, configp, m, configd)$.

**Key design insight — why search prefill and decoding independently:** In the high-bandwidth regime, the only coupling between prefill and decoding is the KV cache transmission, which is assumed negligible. Under this assumption, the per-GPU goodput of a prefill instance does not depend on how many decoding instances exist or their configuration — it only depends on the prefill instance's own ability to meet TTFT. Similarly for decoding. This independence allows the two optimization problems to be completely separated, reducing the search space from the Cartesian product of all prefill and decoding configurations to the sum.

**Complexity:** The enumeration covers $O(M \times N \times M) = O(NM^2)$ configurations. With typical values of $N=4$ (nodes per instance) and $M=8$ (GPUs per node), this is roughly 256 configurations. Each configuration requires a binary search simulation, but simulations are fast (discrete-event, seconds each). The total solving time is reported as under 1.3 minutes (Section 6.5).

**Why this works:** The key enabling factor is disaggregation itself — because prefill and decoding are separated onto different instances, the per-phase goodput depends only on that phase's own parallelism configuration, not on the other phase's choices. In a colocated system, the equivalent search would need to evaluate combined $(prefill\_config, decode\_config)$ pairs jointly because they share GPUs, and the search would be constrained by the need for a single configuration to work for both phases.

---

#### Placement Algorithm for Low Node-Affinity Clusters (Algorithm 2)

When cross-node bandwidth is limited (e.g., the evaluation cluster has only 25 Gbps between nodes), KV cache transmission between prefill and decoding instances on different nodes becomes a bottleneck. The solution is to **colocate the prefill and decoding segments of each pipeline stage on the same node**, forcing all KV cache transfers to use the high-bandwidth intra-node NVLINK.

**The key constraint:** With inter-op (pipeline) parallelism, the model is split into stages, each assigned to different GPUs. When both prefill and decoding instances use pipeline parallelism with the same number of stages, their corresponding stages (e.g., layers 0–11 on stage 0, layers 12–23 on stage 1, etc.) can be placed on the same physical node. The KV cache from prefill stage $i$ needs to be transferred to decoding stage $i$, and if both are on the same node, this transfer uses NVLINK rather than the cross-node network.

**Instance segments:** The paper introduces the concept of **instance segments** — one stage of a pipelined instance. An instance with $inter\_op$ degree of pipeline parallelism is divided into $inter\_op$ segments, each running one pipeline stage. Each segment occupies some number of GPUs determined by its intra-op parallelism degree. The constraint in Algorithm 2 is that for each stage index $i$, the prefill segment $i$ and decoding segment $i$ must be placed on the same node.

**Inputs to Algorithm 2:** Same as Algorithm 1, but with the additional implicit constraint that prefill and decoding instances must share nodes along pipeline stage boundaries.

**Algorithm structure (paraphrased from the paper):**

The algorithm begins by enumerating possible inter-op parallelism degrees: `inter_op` ranges from 1 to $N$ (the node limit per instance). For each `inter_op`, it calls `get_intra_node_configs(G, M, C, inter_op)` to enumerate all possible intra-node parallelism configurations for one segment of a pipeline stage. A single segment's GPUs are constrained to fit within one node (up to $M$ GPUs), and the per-GPU memory must accommodate the stage's model weights: the total model size divided by `inter_op` stages, further divided by the intra-op degree within the segment, must be less than $C$.

Then, for each pair of intra-node configurations $(P_p, P_d)$ for prefill and decoding segments respectively, the algorithm checks a colocation constraint: both segments must fit on the same node simultaneously. This means $P_p.num\_gpus + P_d.num\_gpus \leq M$ — the GPUs used by the prefill segment plus those used by the decoding segment must not exceed the GPUs available in one node.

For each feasible pair, the algorithm constructs a full prefill instance $\hat{G}_p$ and decoding instance $\hat{G}_d$ with the chosen inter-op degree and the respective intra-node configurations repeated across all stages. It then runs a combined simulation `simulate($\hat{G}_p, \hat{G}_d, W$)` that models the full disaggregated system: prefill instances process prompts using FCFS with batching logic, KV caches transfer between corresponding stages (modeled with the intra-node NVLINK bandwidth), and decoding instances generate tokens using continuous batching.

The per-GPU goodput of the combined system is computed as the maximum rate meeting both TTFT and TPOT SLO attainment targets, divided by the total GPUs ($inter\_op \times (P_p.num\_gpus + P_d.num\_gpus)$ for one pair of prefill and decoding instances). The configuration with the highest per-GPU goodput is selected, and then replicated $n = \lceil R / config.goodput \rceil$ times to meet the target traffic rate.

**Key design difference from Algorithm 1:** In Algorithm 1, the per-phase goodputs are simulated independently and the replication counts are computed separately. In Algorithm 2, the combined goodput of one prefill instance paired with one decoding instance is simulated jointly, because their relative sizing (how many prefill GPUs vs. decoding GPUs per node) affects the overall goodput. The combined simulation captures effects like: if prefill is the bottleneck, adding more prefill GPUs per node (at the expense of decoding GPUs) might improve overall goodput; if decoding is the bottleneck, the opposite holds.

**Why the intra-node constraint matters for correctness:** If prefill segment $i$ and decoding segment $i$ were on different nodes, KV cache transfer between them would use the slow cross-node link. For a 512-token request on OPT-66B, the KV cache is ~1.13 GB. At 25 Gbps (the evaluation cluster's cross-node bandwidth), this transfer takes ~360 ms — a substantial fraction of typical TTFT/TPOT budgets and a bottleneck at high request rates. By forcing colocation on the same node, the transfer uses NVLINK at 600 GB/s, reducing the transfer time to ~2 ms, which is negligible relative to inference times.

**Complexity:** The enumeration is over `inter_op` (up to $N$), and for each, the Cartesian product of possible prefill intra-node configs and decoding intra-node configs. Each intra-node search is bounded by the GPUs per node (typically 8), so the product is manageable. The inner-loop simulation is more expensive than in Algorithm 1 because it models both phases jointly, but the total running time remains under 80 seconds for the largest configuration tested (Section 6.5, Figure 12).

**Design choice — why not always use Algorithm 2:** Algorithm 2 is more constrained and may find configurations with lower per-GPU goodput than Algorithm 1, because forcing prefill and decoding segments to share nodes limits flexibility. For example, if the optimal prefill configuration uses 8 GPUs per node and the optimal decoding configuration also uses 8 GPUs per node, they cannot be colocated (would need 16 GPUs per node). The system would have to compromise on one or both. If high cross-node bandwidth is available, Algorithm 1 avoids this compromise entirely, placing prefill and decoding on different nodes optimized independently. The paper's ablation study (Section 6.4, Figure 11) shows that "DistServe-High" (Algorithm 1, assuming high bandwidth) outperforms "DistServe-Low" (Algorithm 2) by a notable margin, confirming that node-affinity constraints reduce achievable goodput.

---

#### Online Scheduling Enhancements

The placement algorithm produces an optimized static configuration, but real deployments face dynamic challenges that require runtime adaptations. DistServe implements three online scheduling mechanisms:

**Reducing pipeline bubbles due to variable-length requests (Section 4.3):**

When prefill instances use pipeline (inter-op) parallelism, each stage processes its portion of the model layers and passes intermediate activations to the next stage. If different requests in the pipeline have different input lengths, their execution times per stage vary, causing some stages to idle waiting for others — these are pipeline bubbles. The paper addresses this by scheduling requests to balance execution time across batches.

The key insight is that **the number of new tokens in a batch is a reliable indicator of the batch's execution time** for both prefill and decoding. For prefill, the system profiles the target model and GPU to determine the shortest prompt length $L_m$ needed to saturate the GPU (making it compute-bound). It then schedules prefill batches so that the total sequence length in the batch is close to $L_m$: batches of multiple short requests whose combined length approximates $L_m$, or individual requests longer than $L_m$ scheduled alone. For decoding instances, $L_m$ is set to the largest batch size that fits in GPU memory, and batches are filled to approach this size.

**Why this works:** By keeping the per-stage workload roughly constant across batches, pipeline stages progress at similar rates, minimizing idle time. Without this scheduling, a batch of one 1024-token request followed by a batch of one 64-token request would cause the first stage to finish quickly for the second batch and then wait while the later stages process the first batch's larger activations.

**The "pull" mechanism for KV cache transmission (Section 4.3):**

Instead of prefill instances pushing KV caches to decoding instances as soon as prefill completes (which could overwhelm decoding instances during traffic bursts), DistServe uses a **pull-based approach**: decoding instances fetch KV caches from prefill instances as they are ready to begin decoding a request. The prefill instance's GPU memory serves as a **queuing buffer** for completed KV caches.

**What this prevents:** During bursty arrivals, many prefill jobs may complete nearly simultaneously, generating a flood of KV caches. If these were pushed to decoding instances, the decoding instances' memory could be exhausted, or their network bandwidth saturated. With pull, the decoding instance controls the flow — it only fetches a KV cache when it has memory capacity and CPU cycles to begin decoding that request. The prefill instance can continue processing new prompts independently, simply retaining completed KV caches until they're fetched.

**Why this matters for goodput:** The pull mechanism prevents a failure mode where burst-induced KV cache overflow causes request drops or memory-related slowdowns on decoding instances, which would reduce SLO attainment. It decouples the pacing of the two phases: prefill can run as fast as GPUs allow, and decoding proceeds at its own sustainable rate without being destabilized by prefill bursts.

**Periodic replanning for workload shifts (Section 4.3):**

The placement algorithm optimizes for a specific workload distribution $W$. If the workload pattern changes significantly (e.g., users start submitting much longer prompts, or the mix of short vs. long outputs shifts), the previously optimal configuration may become suboptimal. DistServe includes a **workload profiler** that monitors key parameters — average input length, average output length, arrival rate distribution — over time.

If the profiler detects a significant distribution shift, it triggers a re-run of the placement algorithm using recent historical data to fit a new workload distribution. The algorithm runs in seconds to minutes (Section 6.5), and applying the new configuration requires reloading model weights onto reconfigured instances, which the paper states "can be completed within minutes — far shorter than the hourly scale at which real-world workload variations tend to occur."

**Design rationale:** This replanning loop makes the system adaptive without requiring continuous online optimization, which would be too slow to run in the inner loop of request scheduling. The assumption is that workload patterns change slowly (over hours), so occasional re-optimization (minutes) is sufficient to track them. The paper does not implement or evaluate this replanning mechanism — it is described as a design feature — so its effectiveness in responding to real workload shifts remains unverified.

**Preemption and fault tolerance (discussed but not implemented):**

The paper acknowledges two limitations of its current runtime design. First, the FCFS scheduling policy in prefill instances can cause a **convoy effect**: a long request arriving slightly before several short requests forces all the short requests to wait behind it, increasing their TTFT. Incorporating preemptive scheduling (e.g., from FastServe, Wu et al., 2023) could mitigate this by allowing short requests to jump ahead of long ones in the queue. The paper notes this is compatible with DistServe's architecture but not implemented.

Second, the dependency between prefill and decoding instances introduces **fault propagation risk**: if one decoding instance fails, all prefill instances feeding it lose their downstream processing path, potentially crippling the service. In a pure replication-based system (where each replica handles both phases independently), a single replica failure doesn't affect others. The paper identifies fault tolerance as important future work but does not propose a solution.

---

#### The Combined Optimization Flow

Stepping back, the full optimization flow works as follows:

1. **Profile the model and hardware:** Before deployment, measure $L_m$ (the compute-bound threshold input length) and calibrate the analytical latency model coefficients $C_1$ through $C_5$ by running a few known batch configurations on the target GPUs. These are one-time measurements per model-hardware pair.

2. **Characterize the workload:** Fit distributions to historical request traces for the arrival process (Poisson rate), input lengths, and output lengths. This workload characterization $W$ is input to the simulator.

3. **Run the placement algorithm:** Based on cluster bandwidth (high vs. low node affinity), execute either Algorithm 1 or Algorithm 2. The algorithm enumerates parallelism configurations, simulates each, and finds the configuration with maximum per-GPU goodput while meeting the SLO attainment target.

4. **Deploy the configuration:** Provision the specified number of prefill and decoding instances with their respective parallelism strategies on the physical GPUs according to the placement constraints. Load model weights onto each instance.

5. **Serve requests:** The controller dispatches incoming requests to prefill instances (shortest-queue FCFS). Prefill instances batch requests according to the $L_m$-based policy. KV caches are pulled by decoding instances. Decoding instances use continuous batching.

6. **Monitor and replan:** The workload profiler tracks key parameters. If a significant distribution shift is detected, re-run from step 2 with recent data, produce a new configuration, and redeploy.

The whole system is designed so that the heavy optimization work (steps 1–4) happens offline, and the runtime path (steps 5–6) is lightweight — FCFS dispatching, simple batching logic, and pull-based KV cache transfer — minimizing per-request overhead while benefiting from the carefully chosen static configuration.

---

#### Summary of Key Design Choices and Their Justifications

- **Disaggregation over colocation:** Eliminates prefill-decoding interference (Figure 2) and enables per-phase parallelism optimization. The cost is KV cache transmission, which is made negligible through bandwidth-aware placement.

- **Per-phase parallelism search over manual configuration:** The optimal parallelism strategy depends on the model, hardware, workload, and SLOs in non-obvious ways (e.g., intra-op is better for prefill at low rates but inter-op becomes better at high rates, as shown by the queuing analysis in Section 3.1). Exhaustive search with a simulator finds configurations that a human operator would be unlikely to guess (e.g., the 3×3 prefill and 4×3 decoding configuration chosen for OPT-175B on ShareGPT in the end-to-end experiments).

- **Analytical latency model over pure profiling:** Profiling every possible batch composition and parallelism configuration would be combinatorially infeasible. The analytical model captures the key structure (compute-bound GEMMs vs. memory-bound attention, quadratic attention scaling) in a parameterized form that generalizes across batch compositions.

- **Simulator over closed-form queuing models:** Variable-length requests, batching, and the interaction between prefill and decoding make closed-form analysis intractable for realistic workloads. The simulator handles these complexities while maintaining sufficient accuracy (≤2% error) and speed (seconds per configuration) for the optimization loop.

- **Pull-based KV cache transfer over push:** Decouples the pacing of prefill and decoding, preventing burst-induced memory exhaustion on decoding instances without requiring explicit coordination or flow control.

- **Two separate algorithms for high vs. low node affinity:** Recognizes that cluster topology is a first-class constraint. Forcing Algorithm 2 in high-bandwidth clusters would leave performance on the table; using Algorithm 1 in low-bandwidth clusters would cause KV cache transmission to become a bottleneck. The system adapts its optimization to the available hardware.

- **FCFS with $L_m$-based batching over more complex scheduling:** The paper's analysis shows that for compute-bound prefill, batching additional requests doesn't improve GPU efficiency — it only increases queuing delay for all requests in the batch. The $L_m$-based policy avoids this by not batching beyond the saturation point. FCFS is simple, predictable, and works well when combined with per-instance shortest-queue dispatching.

## 4. Key Insights and Innovations

### Innovation 1: Disaggregation Is Not Just an Architecture Choice — It Is the Enabling Condition for Per-Phase Optimization

The dominant mental model in LLM serving prior to DistServe treated the prefill and decoding phases as two operations that happen to run on the same GPU — a scheduling problem to be managed, not an architectural constraint to be removed. Systems like Orca (Yu et al., 2022), vLLM (Kwon et al., 2023), and SARATHI (Agrawal et al., 2023) all accepted colocation as a given and built increasingly sophisticated batching and scheduling mechanisms on top of it: continuous batching to interleave prefill with decoding, chunked prefill to break long prefill into smaller pieces, piggybacking to fill otherwise-idle GPU cycles. Each of these innovations made incremental progress on the interference problem, but each also introduced new tradeoffs — chunked prefill trades TTFT for TPOT and incurs O(N²) KV cache reload overhead, piggybacking is only effective when chunk sizes leave room for decoding jobs, and continuous batching forces both phases to share a single parallelism configuration.

DistServe's fundamental conceptual move is to **reject the premise that colocation is necessary**. The paper argues that colocation is not an inherent requirement of LLM serving — it is an artifact of implementation convenience (both phases share model weights, so hosting them together saves GPU memory) that has been mistaken for an architectural constraint. By disaggregating, DistServe doesn't just mitigate prefill-decoding interference — it **eliminates the category of problem entirely**. There is no interference because the two phases never compete for the same GPU. There is no coupling because each phase provisions its own GPUs with its own parallelism strategy.

The significance of this move goes beyond the performance numbers. It reframes the optimization problem from "how do we schedule two conflicting workloads on shared hardware?" to "how do we independently provision two distinct services with different computational profiles and latency requirements?" This is a **fundamental shift** in how the problem is posed, not an incremental improvement over prior scheduling techniques. The consequence is that the entire design space expands: parallelism strategies, batch sizing policies, and instance counts can all be optimized per-phase without constraint. The paper's placement algorithms (Section 4) search over this expanded space and find configurations that would be literally impossible under colocation — for instance, the OPT-175B configuration where prefill uses 3-way inter-op × 3-way intra-op while decoding uses 3-way inter-op × 4-way intra-op, a heterogeneous assignment that no colocated system could express.

The cost of this move is KV cache transmission, and the paper's diagnostic contribution is to **quantify that cost and show it is manageable** across realistic deployment scenarios. Figure 10(b) shows that even for OPT-175B (where KV caches are largest), over 95% of requests experience transmission delays under 30 ms, and the total transmission accounts for less than 0.1% of system execution time. This converts a potential objection ("disaggregation adds network overhead") into a quantified, bounded cost that is dwarfed by the gains from eliminating interference. The bandwidth-aware placement algorithms (Algorithms 1 and 2) ensure this cost stays low across diverse cluster topologies.

The evidence that this reframing is more than a conceptual exercise appears in the ablation study (Figure 11): even when vLLM is augmented with a search over parallelism configurations ("vLLM++"), it achieves the same performance as default vLLM because the parallelism search space under colocation is too constrained to find improvements. The interference bottleneck dominates any gains from better parallelism. Disaggregation is what **unlocks** the optimization space that the placement algorithms then exploit.

### Innovation 2: Goodput as the Optimization Target Rather Than Throughput — and Why It Demands a Different Architecture

Prior LLM serving systems (vLLM, Orca, DeepSpeed-MII) optimize for **throughput** — total tokens generated per second across all users. This is a natural metric when the goal is to maximize hardware utilization: keep the GPU busy by packing as many tokens as possible into each batch, using continuous batching to fill slots as they open. Under throughput optimization, colocation is actually *beneficial* because it allows prefill and decoding tokens to fill the same batch, increasing GPU utilization beyond what either phase could achieve alone.

DistServe's key insight is that **throughput and goodput are fundamentally different optimization targets** that pull the system architecture in opposite directions. Goodput — the maximum request rate that meets latency SLOs for a target fraction of requests — explicitly couples throughput with latency constraints. Under goodput optimization, packing more tokens into a batch is counterproductive if it pushes TPOT or TTFT past the SLO threshold. The system must find the configuration that maximizes sustainable request rate *subject to* latency constraints, which often means running at lower GPU utilization than throughput optimization would dictate.

This distinction is not merely terminological. It has architectural consequences. The paper demonstrates (Figure 1) that a colocated system optimized for throughput achieves only 1.6 rps under latency constraints, while a disaggregated system optimized for goodput achieves 3.3 rps per GPU — 2.1× higher. The colocated system is maximizing the wrong objective: it keeps GPUs busier but violates latency constraints under lower load. The disaggregated system accepts lower per-GPU utilization in exchange for meeting SLOs at higher request rates.

The paper's connection to prior goodput optimization work — Pollux (Qiao et al., 2021) for training, Clockwork (Gujarati et al., 2020) and Shepherd (Zhang et al., 2023) for traditional DNN serving — positions DistServe as extending this line of thinking to autoregressive LLM inference, where the dual-phase nature makes the throughput-goodput divergence particularly acute. In traditional DNN serving, a single inference is a single operation, and queuing theory fairly directly relates throughput and latency (more batching → higher throughput → higher latency). In LLM serving, the relationship is more complex because prefill and decoding have different latency sensitivities and different batching behaviors, making throughput a poor proxy for goodput.

The significance here is that **DistServe doesn't just improve goodput within an existing architecture — it shows that optimizing for goodput requires a different architecture than optimizing for throughput**. The colocated architecture that dominates prior work is throughput-optimal (maximizing batch packing) but goodput-suboptimal (sacrificing latency guarantees). Disaggregation is goodput-optimal because it eliminates the throughput-goodput tension: prefill instances can optimize for TTFT without concern for TPOT, and decoding instances can optimize for TPOT without prefill interference. This is a diagnostic contribution: it explains *why* prior systems struggle with latency SLOs despite high throughput, and it identifies the architectural feature (colocation) that creates the tradeoff.

Concurrent work like Splitwise (Patel et al., 2023) arrives at a similar disaggregation insight, which the paper acknowledges, confirming that this reframing from throughput to goodput is a convergent insight across independent research efforts — further evidence that it captures a genuine structural property of LLM serving rather than being an artifact of DistServe's specific implementation choices.

### Innovation 3: The Difficulty of Automating Goodput-Optimal Configuration — and Why a Simulator-Based Search Is the Pragmatic Solution

On its face, the problem DistServe solves might seem straightforward: pick parallelism configurations for prefill and decoding instances, and replicate until you can handle the traffic. The paper's contribution here is not the idea of searching over configurations (which is standard practice in systems like Alpa, Zheng et al., 2022) but rather the **identification that this search is necessary, non-trivial, and cannot be reduced to simple rules of thumb** in the disaggregated setting.

Section 3 provides the analytical foundation: the M/D/1 queuing analysis (Equations 1–3) shows that prefill's parallelism preference depends on the arrival rate and SLO stringency — intra-op parallelism is better at low rates (reduces execution time), while inter-op parallelism becomes better at high rates (reduces queuing delay). The crossover point depends on the speedup coefficient $K$ (which itself depends on model architecture, input length, and communication bandwidth) and on the TTFT SLO. For decoding (Section 3.2), the preference depends on batch size and TPOT stringency. These are not intuitions a human operator could reliably apply — they require quantitative modeling of the specific model, hardware, workload, and SLOs.

The novelty is in **identifying that this configuration problem is hard enough to require systematic search, and tractable enough that a simulator-based approach works**. The hardness comes from three factors: (1) the interaction between parallelism strategy and queuing behavior is non-linear (the M/D/1 equations show how execution time and queuing delay trade off differently under different parallelism choices), (2) real workloads have variable-length inputs and outputs that break closed-form analysis, and (3) the low node-affinity case couples prefill and decoding configuration through the per-node GPU constraint. The tractability comes from the high predictability of DNN workloads (noted by prior work like Clockwork and AlpaServe) and the modest size of the search space (~256 configurations in the largest setting).

The paper's decision to build a custom simulator rather than relying on analytical models or pure profiling is a pragmatic insight. Analytical models (like the M/D/1 equations in Section 3) provide conceptual guidance but cannot capture variable-length requests, batching policies, or the interaction between prefill and decoding. Pure profiling on real hardware would be combinatorially infeasible — the number of batch compositions and parallelism configurations is too large. The analytical latency model (Appendix A) bridges this gap: it captures the key computational structure (compute-bound GEMMs scaling with total tokens, memory-bound attention scaling with squared lengths) in a parameterized form that generalizes across configurations, and the coefficients are calibrated with a small number of profiling runs. The result is a simulator accurate to within 2% of real measurements (Table 2) while fast enough to serve as the inner loop of a configuration search (under 1.3 minutes for the largest setting, Figure 12).

This is an **engineering contribution with conceptual significance**: it demonstrates that automated goodput-optimal configuration is feasible for LLM serving, and it provides a methodology (analytical latency model + discrete-event simulation + exhaustive search) that can be adapted to new models, hardware, and workloads. The relatively modest accuracy requirement (2% error on SLO attainment is acceptable for configuration selection) means the approach doesn't need to be perfect — it just needs to rank configurations correctly, which is an easier problem than predicting absolute performance.

### Innovation 4: Verifying That Disaggregation's Benefits Compound with Scale — and That the Approach Has Sharp Boundaries

The paper's evaluation does more than demonstrate that DistServe outperforms baselines. It systematically characterizes **where disaggregation helps and where it doesn't**, establishing boundary conditions that are as informative as the positive results.

**The compounding benefit with model scale.** Across the three OPT model sizes (13B, 66B, 175B) on the chatbot workload (Figure 8), DistServe's advantage over vLLM grows from 2.0× to 4.6× in sustainable request rate. This is not accidental: larger models have larger weight matrices, meaning prefill steps take longer and cause proportionally more interference when batched with decoding in colocated systems. The disaggregation benefit is not a fixed multiplier — it scales with the computational disparity between the two phases, which widens with model size and input length. This suggests that as LLMs continue to grow, the case for disaggregation strengthens.

**The workload-dependent benefit.** On code completion (Figure 9a), DistServe achieves 5.7× higher rate than vLLM — a larger gap than on chatbot (1.6×–7.4× for DeepSpeed-MII, 2.0×–4.6× for vLLM) because code completion has stringent TTFT requirements that are particularly harmed by prefill-decoding interference. On summarization (Figure 9b), the gap is 4.3× over vLLM, driven by long input lengths creating large prefill jobs that severely delay colocated decoding. These variations are not noise — they reflect the mechanism: disaggregation helps most when prefill-decoding interference is most severe, which occurs when (a) prefill is long (high TTFT impact from sharing GPUs with decoding) or (b) TPOT requirements are stringent (decoding is sensitive to prefill-induced delays).

**The sharp boundary at hard SLO targets.** The appendix results (Figures 13 and 14) show that under a 99% SLO attainment target, DistServe's advantage grows further — 3×–8× over vLLM, 1.32×–8× over DeepSpeed-MII. This is because colocated systems exhibit higher tail latency variance (prefill-decoding interference creates unpredictable delays), making it harder to meet stringent attainment targets. Disaggregation reduces this variance by isolating the phases, directly translating to better tail latency behavior.

**The bandwidth dependence.** The ablation study (Figure 11) shows that "DistServe-High" (algorithm assuming high cross-node bandwidth) outperforms "DistServe-Low" (algorithm constrained to intra-node KV cache transfer) by a notable margin. This quantifies the cost of limited network infrastructure: when prefill and decoding cannot be placed independently across nodes, the optimization is more constrained and goodput suffers. This finding provides guidance for cluster provisioning — investing in higher cross-node bandwidth pays off in better disaggregation efficiency.

**The limitation on hard problems is not a failure of disaggregation.** DistServe cannot help when the model's per-phase execution time is fundamentally too high to meet SLOs regardless of configuration — if a single prefill step on the largest available parallelism configuration still exceeds the TTFT SLO, no amount of disaggregation or optimization will help. The paper doesn't encounter this regime in its evaluation, but the principle is clear: disaggregation eliminates interference overhead, but it doesn't reduce the intrinsic execution time of each phase. For extremely tight SLOs, the solution is faster hardware or model compression, not disaggregation.

This systematic boundary characterization is a **methodological contribution**: it provides a template for evaluating LLM serving systems that goes beyond aggregate throughput numbers to ask *under what conditions* a technique is effective, and it quantifies the sensitivity of performance to workload characteristics, model scale, SLO targets, and cluster topology. This kind of analysis makes the results actionable — a practitioner can assess whether their deployment scenario falls within the regime where disaggregation helps.

## 5. Experimental Analysis

### Evaluation Methodology

**Dataset.** The paper evaluates on three real-world LLM application datasets. For the chatbot application, it uses the **ShareGPT dataset**, a collection of user-shared conversations with ChatGPT (Section 6.1). For code completion, it uses the **HumanEval dataset** (Chen et al., 2021), including 164 programming problems with function signatures or docstrings. For summarization, it uses the **LongBench dataset** (Bai et al., 2023), specifically the summarization task, with input lengths capped at 2048 because OPT's absolute positional embedding does not support longer sequences. No explicit train/test split is used — the datasets serve as sources from which requests are sampled to construct evaluation workloads.

**Base model(s).** All experiments use the **OPT model family** (Zhang et al., 2022): OPT-13B (26 GB), OPT-66B (132 GB), and OPT-175B (350 GB), all in FP16 precision. The paper states this choice follows prior work on LLM serving (vLLM, Kwon et al., 2023) and notes that OPT uses the classic multi-head attention (MHA) rather than newer memory-efficient mechanisms like grouped-query attention (GQA) or multi-query attention (MQA), deliberately putting more pressure on KV cache transmission overhead. The chatbot workload is tested on all three model sizes; code completion and summarization are tested on OPT-66B only.

**Metrics.** The primary metric is **SLO attainment** — the percentage of requests that satisfy both TTFT and TPOT latency requirements — plotted as a function of either request rate or SLO scale. From these curves, the paper extracts two key quantities: (1) **per-GPU goodput**, defined as the maximum per-GPU request rate at which SLO attainment meets or exceeds a target (typically 90%), and (2) **minimum sustainable SLO scale**, determined by fixing the request rate and linearly scaling both TTFT and TPOT requirements downward until attainment falls below the target. Per-GPU goodput is computed by dividing the total sustainable rate by the number of GPUs used in the configuration. The paper evaluates SLO attainment at both 90% and 99% targets (main results use 90%; Appendix C reports 99%).

**Baselines.** Two state-of-the-art colocated serving systems are compared:
- **vLLM** (Kwon et al., 2023): A widely-used LLM serving system supporting continuous batching and paged-attention for KV cache management. Since vLLM supports only intra-operator parallelism, the paper follows the original vLLM paper's configuration: intra-op = 1 for OPT-13B, intra-op = 4 for OPT-66B, and intra-op = 8 for OPT-175B.
- **DeepSpeed-MII** (2023): DeepSpeed Model Implementations for Inference, which supports chunked-prefill with piggyback (the SARATHI approach, Agrawal et al., 2023). Its intra-op parallelism is set identically to vLLM for OPT-13B and OPT-66B. DeepSpeed-MII cannot serve OPT-175B in the paper's setup because its underlying kernel implementation requires `vocab_size/intra_op` to be a multiple of 8; with intra-op = 8, this condition fails for OPT's vocab_size of 50272, and with intra-op = 4, the model causes an out-of-memory error.

The paper does NOT compare against concurrent disaggregated systems (Splitwise, TetriInfer, DéjàVu) since they appeared simultaneously and are acknowledged only in the related work.

**Generation budget / compute accounting.** The paper does not use a "generation budget" in the sense of FLOPs or token counts as the unit of comparison. Instead, systems are compared by their **per-GPU goodput** — the maximum sustainable request rate on a given GPU allocation while meeting SLO targets. The hardware is fixed (A100-80GB GPUs), and the total GPU count is accounted for in the per-GPU division. For the placement algorithm search, the compute cost of configuration enumeration is measured separately (Section 6.5, Figure 12) and reported in wall-clock time, not FLOPs. The KV cache transmission overhead is measured via latency breakdown (Section 6.3) and shown to be negligible as a fraction of total request processing time.

**Cross-validation / statistical protocol.** No cross-validation is reported. The evaluation uses real deployments on a physical cluster (4 nodes, 32 NVIDIA SXM A100-80GB GPUs, NVLINK intra-node, 25 Gbps cross-node). Request arrival times are generated using a Poisson distribution with varying rates; requests are sampled from each dataset to construct workloads matching the application's input/output length distributions (shown in Figure 7). The paper reports SLO attainment curves as a function of rate or SLO scale, with the 90% attainment level marked by vertical lines, but does not report confidence intervals or multiple runs with different random seeds. The simulator accuracy is verified in a separate experiment (Table 2) by comparing predicted vs. real SLO attainment at different rates, showing ≤2% error.

---

### Main Quantitative Results

#### Chatbot Application Across Model Scales

The first row of Figure 8 shows SLO attainment vs. per-GPU request rate for all three OPT model sizes on the ShareGPT dataset, with the 90% attainment level indicated by vertical lines marking the maximum per-GPU goodput.

**OPT-13B:** DistServe sustains approximately **2.0× higher request rate** than vLLM and approximately **1.6× higher** than DeepSpeed-MII at the 90% SLO attainment target. The vLLM curve drops below 90% at a per-GPU rate just above 1.0 req/s, while DistServe maintains above 90% past 2.0 req/s.

**OPT-66B:** DistServe achieves approximately **4.6× higher** per-GPU rate than vLLM (which falls below 90% around 0.25 req/s, while DistServe sustains roughly 1.15 req/s) and approximately **7.4× higher** than DeepSpeed-MII (which drops below 90% well before 0.5 req/s, while DistServe exceeds 1.1 req/s). The gap widens substantially compared to OPT-13B, consistent with larger models experiencing more severe prefill-decoding interference under colocation.

**OPT-175B:** DistServe sustains roughly **3.0× higher** per-GPU rate than vLLM. The DistServe curve maintains above 90% attainment at approximately 0.75 req/s, while vLLM falls below 90% around 0.25 req/s. DeepSpeed-MII is absent from this comparison due to the compatibility issue noted above. The paper highlights that DistServe's chosen placement for OPT-175B is non-trivial: prefill uses inter-op = 3, intra-op = 3; decoding uses inter-op = 3, intra-op = 4 — a heterogeneous configuration that no colocated system could express.

The second row of Figure 8 evaluates robustness to SLO stringency: the request rate is fixed, and both TTFT and TPOT requirements are scaled simultaneously by a multiplicative factor called **SLO Scale** (lower is more stringent). At 90% attainment:

- **OPT-13B:** DistServe sustains roughly **3.2× more stringent SLO** than vLLM (SLO Scale ≈ 0.25 vs. 0.8) and roughly **1.8× more stringent** than DeepSpeed-MII (≈ 0.25 vs. 0.45).
- **OPT-66B:** DistServe achieves roughly **2.5× more stringent SLO** than vLLM (≈ 0.65 vs. 1.6) and roughly **1.7× more stringent** than DeepSpeed-MII.
- **OPT-175B:** DistServe sustains roughly **1.8× more stringent SLO** than vLLM.

The paper attributes vLLM's poor performance primarily to TPOT violations: colocating prefill and decoding significantly slows decoding, and with stringent TPOT requirements on chatbot applications, many requests violate the TPOT SLO even though TTFT is met for most requests. DeepSpeed-MII shows better performance on larger models because chunked-prefill mitigates some interference, but as discussed in Section 2.3, chunked prefill is slower than full prefill, so it struggles to meet TTFT SLO as a sacrifice for better TPOT.

#### Code Completion Application (OPT-66B)

Figure 9(a) shows SLO attainment for the code completion task on HumanEval. At the 90% attainment target:

- DistServe sustains **5.7× higher per-GPU request rate** than vLLM and **1.6× higher** than DeepSpeed-MII.
- Under the SLO Scale analysis, DistServe achieves approximately **1.4× more stringent SLO** than both vLLM and DeepSpeed-MII.

The paper notes that code completion's stringent TTFT requirement (Table 1: 0.125s TTFT, 0.2s TPOT) makes TTFT the binding constraint for all systems. DistServe eliminates decoding interference on prefill instances, and its placement algorithm automatically increases intra-op parallelism for prefill to reduce average prefill latency, enabling more requests to meet the tight TTFT target.

#### Summarization Application (OPT-66B)

Figure 9(b) shows results for the summarization task on LongBench, which features much longer input lengths (average 1738.3 input tokens vs. 755.5 for ShareGPT; see Figure 7). At the 90% attainment target:

- DistServe sustains **4.3× higher per-GPU request rate** than vLLM and **1.8× higher** than DeepSpeed-MII.
- Under the SLO Scale analysis, DistServe achieves roughly **12.6× more stringent SLO** than vLLM and **2.6× more stringent** than DeepSpeed-MII. The vLLM SLO scale curve barely reaches 90% even at the loosest tested setting (SLO Scale ≈ 10), while DistServe maintains 90% attainment down to approximately SLO Scale = 0.8.

The paper attributes vLLM's severe underperformance to the long input lengths: large prefill jobs cause a substantial slowdown in the colocated decoding phase, making it impossible to meet the TPOT requirement (Table 1: 0.15s TPOT). Even though the TTFT SLO is loose (15s), vLLM's TPOT violations dominate. DeepSpeed-MII's chunked prefill alleviates some decoding slowdown but at the cost of increased prefill time, and it still struggles with the TPOT constraint.

#### Results Under 99% SLO Attainment Target

Appendix C (Figures 13 and 14) reports the same experiments with a more stringent 99% SLO attainment target. The paper notes that the performance gap widens:

- On chatbot (Figure 13), DistServe sustains **3×–8× higher rate** and **1.24×–6.67× more stringent SLO** compared to vLLM, and **1.32×–8× higher rate** and **1.20×–1.58× more stringent SLO** compared to DeepSpeed-MII, depending on model size.
- On code completion (Figure 14a), DistServe continues to substantially outperform both baselines.
- On summarization (Figure 14b), the gap remains large, consistent with the 90% attainment results.

The widening gap at higher attainment targets is consistent with the paper's claim that disaggregation reduces tail latency variance: colocated systems exhibit unpredictable prefill-decoding interference that creates long-tail latency spikes, which are particularly penalized under high attainment targets (where even 1% of requests can violate SLOs).

---

### Ablation Studies and Robustness Checks

**Simulator accuracy (Table 2):** The SLO attainment reported by the simulator is compared against real measurements on the physical testbed for both vLLM and DistServe-Low across request rates from 1.0 to 4.0 req/s when serving OPT-66B on ShareGPT. The maximum error is consistently below 2%: for example, at 2.0 req/s, vLLM's real SLO attainment is 52.8% vs. simulated 51.0% (1.8% error), and DistServe-Low's real attainment is 99.3% vs. simulated 99.3% (0.0% error). The worst-case error across all configurations is 1.8 percentage points. This validates that the analytical latency model (Appendix A) combined with discrete-event simulation captures the key performance factors accurately enough for configuration ranking.

**Parallelism search under colocation (Figure 11, "vLLM++"):** To test whether the benefits attributed to disaggregation could instead be achieved by better parallelism configuration in a colocated system, the paper implements "vLLM++" — vLLM augmented with enumeration over different intra-op parallelism strategies, selecting the one with best per-GPU goodput. The result shows "vLLM++" has the **same performance as default vLLM** because the optimal configuration for the colocated setting happens to be the default intra-op setting. The paper interprets this as evidence that prefill-decoding interference is the dominant bottleneck under colocation — adjusting parallelism within the constrained colocated space cannot overcome the fundamental interference, so the parallelism search yields no improvement.

**High vs. low node-affinity placement (Figure 11):** Two versions of DistServe are compared via simulation (since the physical testbed lacks high cross-node bandwidth): **DistServe-High** (Algorithm 1, assuming high cross-node bandwidth, no colocation constraint between prefill and decoding) and **DistServe-Low** (Algorithm 2, constrained to colocate prefill and decoding segments on the same node). When serving OPT-66B on ShareGPT, DistServe-High achieves notably higher SLO attainment than DistServe-Low at the same request rates, and sustains a higher rate at the 90% attainment target. The paper attributes this to the additional flexibility of Algorithm 1: when prefill and decoding instances can be placed independently across nodes, the per-phase parallelism configurations can be optimized without the constraint that each node must accommodate both a prefill and decoding segment.

**Latency breakdown and transmission overhead (Figure 10):** To quantify the cost of disaggregation, the paper decomposes the total processing time of all requests into five stages: prefill queuing, prefill execution, transmission, decoding queuing, and decoding execution. For OPT-175B on ShareGPT across rates from 0.03 to 0.28 req/s:

- **Transmission accounts for less than 0.1% of total execution time** at all rates. Even for OPT-175B, where KV caches are largest (the per-layer hidden size is larger and there are more layers), the transmission overhead is negligible.
- The CDF of absolute transmission time (Figure 10b) shows that over **95% of requests experience transmission delays under 30 ms** across all three model sizes, despite the testbed having only 25 Gbps cross-node bandwidth. This is enabled by the low node-affinity placement algorithm (Algorithm 2), which forces KV cache transfers between corresponding pipeline stages to use intra-node NVLINK (600 GB/s between A100 GPUs) rather than the cross-node link.

The paper notes that OPT was chosen specifically because it uses MHA rather than GQA/MQA, producing larger KV caches and thus putting more pressure on transmission overhead. DistServe would show even lower transmission overhead on newer models with memory-efficient attention.

**Algorithm running time (Figure 12):** The execution time of the placement algorithms is profiled on an AWS m5d.metal instance with 96 CPU cores as the number of GPUs available to a single instance increases from 2 to 32:

- Both **DistServe-Low and DistServe-High scale well** with GPU count, and running time is independent of model size (since the simulator only simulates discrete events, the per-event computation does not depend on model dimensions).
- At the largest tested configuration (32 GPUs), DistServe-High runs in approximately **20 seconds** and DistServe-Low in approximately **75 seconds**.
- DistServe-Low becomes slower than DistServe-High as GPU count increases because the search for prefill and decoding configurations in Algorithm 2 is coupled (the Cartesian product of intra-node configurations creates a larger search space), while in Algorithm 1 the two phases are searched independently and parallelized.
- Both algorithms are highly parallelizable since the simulations for different parallelism configurations are independent.

The paper argues that execution times of seconds to minutes are acceptable because the algorithm runs once before deployment (or on workload shifts detected by the profiler), and the redeployment time (reloading model weights) is on the order of minutes — shorter than the hourly scale of real-world workload variations.

**Workload distribution characteristics (Figure 7):** The input and output length distributions of the three datasets are characterized: ShareGPT has average input 755.5 tokens and output 200.3 tokens; HumanEval has average input 171.3 tokens and output 98.2 tokens; LongBench has average input 1738.3 tokens and output 90.7 tokens. These distributions explain some of the workload-dependent performance differences: LongBench's long inputs create large prefill jobs that exacerbate colocation interference, while HumanEval's shorter inputs reduce this effect. The input/output length distributions are used by the simulator to generate realistic request traces from fitted distributions.

---

### Critical Assessment

#### Do the experiments support the claim that disaggregation achieves up to 7.4× higher request rates?

The claim that DistServe serves "up to 7.4× more requests" (Abstract) is numerically supported by Figure 8(b): on the OPT-66B chatbot workload, DistServe's maximum per-GPU rate is approximately 7.4× that of DeepSpeed-MII at the 90% attainment target. However, this is a **best-case comparison against the weakest baseline** (DeepSpeed-MII) on a specific model size. Against vLLM, the same configuration achieves 4.6×. On OPT-13B, the advantage is 2.0× over vLLM and 1.6× over DeepSpeed-MII — substantially smaller. The "up to 7.4×" framing is technically accurate but masks substantial variation across baselines and model scales. The more representative range across the full evaluation is roughly 1.6×–5.7× over the stronger of the two baselines for any given configuration.

The comparison also embeds an asymmetry: DistServe's configurations are automatically optimized by the placement algorithm, while the baselines use fixed (paper-recommended) parallelism settings. The "vLLM++" ablation (Figure 11) partially addresses this by showing that even when vLLM is given the same configuration search, it achieves the same performance as default vLLM — the interference bottleneck dominates. But this ablation was only run on OPT-66B with ShareGPT in simulation, leaving open the question of whether a more exhaustive search (including inter-op parallelism or different scheduling policies) could narrow the gap on other configurations.

#### Do the experiments support the claim of 12.6× tighter SLO?

The "12.6× tighter SLO" claim (Abstract) is supported by Figure 9(b): on the summarization workload with OPT-66B, DistServe achieves 90% attainment at approximately SLO Scale = 0.8 while vLLM fails to reach 90% even at SLO Scale ≈ 10, giving a ratio of roughly 12.5–12.6. However, this number requires careful interpretation. The SLO Scale metric simultaneously tightens both TTFT and TPOT while keeping their ratio constant relative to the application's default requirements. For summarization, the default TTFT SLO is already loose (15s; Table 1), while TPOT is stringent (0.15s). Scaling both by 12.6× means the effective TPOT requirement becomes roughly 0.15/12.6 ≈ 12 milliseconds, which is close to or below the minimum achievable per-step latency for OPT-66B on A100 GPUs regardless of the serving architecture. The practical meaning of "12.6× tighter SLO" is therefore that DistServe continues meeting TPOT requirements at scales where vLLM cannot meet even the loosest tested configuration — but the absolute TPOT values at those extremes may approach hardware limits where neither system can operate. The SLO Scale metric is useful for comparing relative robustness but should not be interpreted as implying that DistServe can deliver 12.6× faster per-token latency in absolute terms.

#### Do the experiments support the claim that disaggregation eliminates prefill-decoding interference?

The **mechanism** is well-demonstrated by the latency breakdown (Figure 10), which shows that in DistServe, prefill and decoding execution times are independent — there is no stage where one phase blocks the other. The SLO attainment curves (Figures 8 and 9) show that DistServe maintains high attainment at rates where colocated systems have already degraded, consistent with the absence of interference.

However, the experiments do not **directly isolate** interference as the causal factor. A colocated system with ideal scheduling (which does not exist) might hypothetically achieve some of the same gains. The paper's argument is that ideal scheduling is impossible under colocation because the two phases fundamentally conflict — but this is a theoretical claim about the impossibility of simultaneously optimizing TTFT and TPOT on shared hardware, not an experimental finding. The experiments demonstrate correlation (disaggregation → higher goodput), and the analysis in Sections 2 and 3 provides the causal mechanism, but there is no experimental manipulation that independently varies interference while holding other factors constant.

The ablation with "vLLM++" (Figure 11) provides the strongest evidence that interference is the binding constraint: even with optimized parallelism, colocated performance doesn't improve because the bottleneck is the phase interaction, not the parallelism configuration. But this is a single data point (OPT-66B on ShareGPT) and assumes that the vLLM scheduling policy is optimal for the given parallelism configuration, which may not hold.

#### Do the experiments support the claim that per-phase parallelism optimization is necessary and beneficial?

The placement algorithm's chosen configurations (Table 3 in Appendix B) demonstrate that DistServe **does** select heterogeneous per-phase parallelism: for OPT-175B on ShareGPT, prefill uses 3-way pipeline × 3-way tensor, while decoding uses 3-way pipeline × 4-way tensor. For OPT-13B, prefill uses 2-way tensor and decoding uses 1-way tensor (no parallelism). These are configurations that a human operator would not trivially guess, and they differ between the two phases, confirming that per-phase optimization is being exercised.

However, the experiments do not include an ablation where DistServe uses the **same** parallelism for both phases (as a colocated system would be forced to do) but with disaggregation, to quantify how much of the gain comes from disaggregation alone vs. from the per-phase parallelism tuning. The comparison against vLLM confounds these two effects: DistServe differs from vLLM both in disaggregation and in having per-phase parallelism optimization. The "vLLM++" result suggests that parallelism tuning alone doesn't help under colocation, but it doesn't answer the question of how much per-phase tuning helps *given* disaggregation. The "DistServe-High" vs. "DistServe-Low" comparison (Figure 11) shows that more flexible placement (which enables better per-phase optimization) improves goodput, but this is about placement constraints, not about the value of heterogeneous parallelism per se.

#### Missing experiments and robustness concerns

**Single model family.** All experiments use OPT models. The paper argues OPT was chosen to stress-test KV cache transmission (since OPT uses MHA rather than GQA/MQA), but this means the results may not generalize to the LLaMA family or other widely-used architectures that employ more efficient attention mechanisms. The paper acknowledges that DistServe would show better performance on GQA/MQA models (Section 6.1), but this claim is not experimentally verified.

**Single GPU architecture.** All experiments use NVIDIA A100-80GB GPUs. The compute-bound vs. memory-bound thresholds, the intra-op speedup coefficient $K$, and the optimal batching strategies all depend on the GPU's compute throughput, memory bandwidth, and interconnect speed. The paper's analytical latency model (Appendix A) is parameterized by hardware-specific coefficients, suggesting the approach should transfer to other GPUs, but no experimental evidence is provided for this claim.

**No evaluation of the replanning mechanism.** Section 4.3 describes a workload profiler that detects pattern shifts and triggers re-optimization, and Section 6.5 reports algorithm running times that would enable replanning within minutes. However, no experiment evaluates whether replanning actually improves goodput when workload characteristics change. The experiments use static workload distributions (Poisson arrivals with fixed per-dataset length distributions), so the replanning capability is described but untested.

**No evaluation of fault tolerance or preemption.** The paper explicitly acknowledges that fault tolerance and preemptive scheduling are not implemented (Section 4.3). In a disaggregated architecture, the dependency chain (prefill → KV cache transfer → decoding) introduces failure modes that don't exist in replicated colocated systems. Without experimental evidence that these failure modes are manageable, the system's robustness in production remains an open question.

**Cluster scale is modest.** The evaluation uses a 4-node, 32-GPU cluster. Whether the placement algorithms scale to production clusters with hundreds or thousands of GPUs — where network topology becomes more complex and the configuration search space may grow — is not addressed. The paper shows algorithmic scaling in simulation up to 32 GPUs per instance (Figure 12), but the cluster-wide optimization across many instances is not evaluated at larger scales.

**No latency constraint violation analysis by phase.** The SLO attainment metric aggregates violations of both TTFT and TPOT. The paper qualitatively notes that vLLM tends to violate TPOT on chatbot workloads and that DistServe's TTFT is unaffected by decoding load, but it does not provide a breakdown of SLO violations by constraint type (TTFT vs. TPOT) across the three systems. Such a breakdown would directly quantify how much of the improvement comes from fixing each constraint independently.

**Workload representativeness.** The evaluation uses Poisson arrivals, which may not capture the bursty, time-of-day-dependent patterns of real LLM services. The paper's pull-based KV cache mechanism is specifically motivated by burst tolerance (Section 4.3), but no bursty workload experiment is conducted to verify that the pull mechanism works as intended under realistic traffic spikes.

#### Conditions under which the claims hold

The paper's central claim — that disaggregation improves per-GPU goodput — holds clearly under the evaluated conditions: OPT models (13B–175B) on A100 GPUs, with applications spanning a range of TTFT/TPOT requirements and input lengths from ~170 to ~1740 tokens, in a cluster with intra-node NVLINK but limited cross-node bandwidth. Within this envelope, DistServe consistently and substantially outperforms both vLLM and DeepSpeed-MII.

The paper does NOT claim (and the experiments do not establish) that disaggregation is beneficial when:
- **Cross-node bandwidth is very low and model parallelism across nodes is required**, forcing KV caches to traverse slow links. DistServe-Low's constraint of colocating prefill and decoding segments on the same node addresses this partially, but if the model is so large that a single stage exceeds one node's GPU memory, or if the optimal per-phase configurations cannot be colocated within a node, Algorithm 2 may fail to find a valid placement or may settle for substantially suboptimal goodput.
- **Models are small enough that a single GPU can handle both phases without interference.** For OPT-13B on ShareGPT, DistServe's advantage is 2.0× over vLLM — meaningful but smaller than for larger models. For even smaller models (not tested), the disaggregation overhead (additional GPU memory for duplicate weights, KV cache transmission) might outweigh the interference elimination benefit.
- **Latency requirements are extremely loose** relative to model execution time, such that both phases can easily meet SLOs even with interference under colocation. In this case, colocated continuous batching may achieve higher throughput (not goodput) by better GPU utilization. The paper explicitly notes this in Section 7: "In offline applications that are not latency-sensitive... the effectiveness of DistServe may be compromised."
- **GPU memory is extremely scarce**, such that duplicating model weights for prefill and decoding instances is infeasible. The paper's OPT-175B configuration in Table 3 shows that DistServe uses significant GPU resources: 9 GPUs per prefill instance and 12 GPUs per decoding instance under the chosen parallelism strategy. In resource-constrained settings (a few GPUs), disaggregation may not be viable.

## 6. Limitations and Trade-offs

### The Difficulty Estimation Cost Is Unaccounted for in Headline Efficiency Gains

**The assumption or constraint.** The entire compute-optimal framework depends on estimating prompt difficulty *before* deciding how to allocate the inference budget. The paper's method requires generating 2048 samples per question to create the difficulty bins, as acknowledged in Section 3.2:

> "estimating difficulty in this way still incurs additional computation cost during inference... our experiments do not account for this cost largely for simplicity"

**The consequence.** The reported 4× efficiency gains (e.g., "16 generations matching 64 generations of best-of-N" in Figures 4 and 8) are computed *after* difficulty is known, without amortizing the cost of learning it. Generating 2048 samples per question costs more compute than the largest test-time budgets studied (256–512 generations). In a realistic deployment where difficulty estimation must be done online for each new question, the total cost would be difficulty estimation + strategy execution, and the former could dwarf the latter. The 4× figure should therefore be understood as an **upper bound on achievable efficiency** — the realized gain in a production system could be substantially smaller or even negative once difficulty estimation is fully accounted for.

**What evidence exists in the paper.** The paper provides no experiment that measures the total end-to-end cost including difficulty estimation. The predicted (non-oracle) difficulty bins perform nearly as well as oracle bins (Figures 4 and 8), demonstrating that ground-truth labels aren't needed, but this still uses 2048 samples + PRM scoring per question. The paper does not evaluate cheaper difficulty estimation methods (e.g., using fewer samples, training a lightweight classifier from question text, or adaptive estimation integrated into the solution process).

**Mitigation status.** The paper explicitly identifies this as a key avenue for future work in Section 3.2:

> "We leave the exploration of this exploration-exploitation tradeoff to future work"

and again in Section 8:

> "future work can also explore pretraining or finetuning models to directly predict difficulty of a question"

No mitigation is implemented or evaluated in the current work. The difficulty estimation cost is the single largest barrier between the paper's demonstrated results and practical deployment.

---

### Results Are from a Single Benchmark (MATH) and a Single Model Family (PaLM 2-S*)

**The assumption or constraint.** All experiments use exclusively the MATH benchmark (Hendrycks et al., 2021) — 500 test questions drawn from high-school competition math — with PaLM 2-S* (Codey) as the only base model evaluated. The authors state they "believe this model is representative of the capabilities of many contemporary LLMs" (Section 4), but this claim is asserted, not demonstrated.

**The consequence.** Several aspects of the paper's findings could be specific to the MATH-PaLM 2-S* combination and not generalize:

- **The PRM's over-optimization behavior** (beam search degrading on easy problems, Figure 3 right) depends on the specific distribution of PaLM 2-S*'s outputs. A model with different calibration properties or different error patterns could exhibit different over-optimization thresholds, changing which strategies are optimal at which difficulty levels.

- **The revision model's ability to learn from incorrect in-context examples** depends on the base model's capacity for in-context learning from its own outputs. LLM families vary substantially in this capability.

- **MATH problems require primarily symbolic reasoning** — mathematical derivation, algebraic manipulation, logical deduction. It is unknown whether the difficulty-dependent patterns (sequential revisions dominating on easy problems, beam search dominating on medium, neither helping on hard) transfer to other reasoning domains such as code generation, logical reasoning, scientific question-answering, or to tasks requiring factual recall rather than inference. The paper's claim in Section 4 that MATH was chosen because "test-time compute is expected to help most when the model already possesses the necessary knowledge" implicitly limits the scope to reasoning-over-existing-knowledge tasks.

- **MATH has clean ground-truth answers** graded by exact string matching (via the Lightman et al., 2022 grading function). This enables both the difficulty estimation (via pass@1 computation) and the PRM training pipeline (via Monte Carlo rollout correctness checking). Many important LLM applications — open-ended generation, creative writing, complex multi-step planning, dialogue — lack clean binary correctness signals, requiring fundamentally different verifier training and difficulty estimation approaches that the paper does not explore.

**What evidence exists in the paper.** None. There are no cross-benchmark or cross-model-family experiments. The authors' stated belief in PaLM 2-S*'s representativeness is not supported by any evidence. The paper also does not provide confidence intervals on SLO attainment curves, making it impossible to assess whether the observed strategy rankings per difficulty bin are statistically stable given the 500-question test set (split into quintiles of ~100 each, further split by two-fold cross-validation, meaning strategy selection relies on ~50 questions per fold per bin).

**Mitigation status.** The limitation is acknowledged implicitly by the scope of the experimental section but is never explicitly discussed as a limitation. The paper does not frame generalization to other benchmarks or model families as future work. This is the most fundamental open question about the paper's contributions: whether the difficulty-conditioned compute-optimal framework transfers to other domains and models, or whether the specific patterns observed (beam search optimal on medium difficulty, revisions optimal on easy, neither on hard) are idiosyncratic to MATH and PaLM 2-S*.

---

### The 14× Larger Model Baseline Is Not Compute-Optimally Trained and Uses Greedy Decoding

**The assumption or constraint.** The FLOPs-matched comparison in Section 7 compares PaLM 2-S* with compute-optimal test-time strategies against a model with ~14× more parameters. The larger model is created by scaling parameters while holding training data fixed, following the LLaMA paradigm (Touvron et al., 2023). The authors acknowledge this departs from compute-optimal pretraining:

> "We choose this setting as it is representative of a canonical approach to scaling pretraining compute and leave the analysis of compute-optimal scaling of pretraining compute where the data and parameters are both scaled equally to future work." (Section 7)

Additionally, the larger model uses only **greedy decoding** with no test-time compute augmentation of its own.

**The consequence.** The reported advantage of test-time compute over pretraining (e.g., +27.8% relative improvement on easy questions at R << 1 for revisions, Section 7) is measured against a baseline that is weaker than it could be in two ways:

1. **Parameter-only scaling is suboptimal.** Hoffmann et al. (2022) established that compute-optimal pretraining scales model size and training data equally. A Chinchilla-optimal model trained with the same total FLOPs budget would allocate some of the additional compute to more training tokens rather than solely to more parameters. This Chinchilla-optimal larger model would likely outperform the parameter-only-scaled version, narrowing or potentially reversing the reported advantages of test-time compute.

2. **Greedy decoding is a weak inference baseline.** The ~14× larger model is evaluated with greedy decoding only — no majority voting, no best-of-N, no search. Giving the larger model even a modest test-time compute budget (say, best-of-8 or best-of-16) would create a much stronger comparison that answers a more relevant question: "given a fixed total FLOPs budget, should I invest in a larger model with simple inference or a smaller model with elaborate inference?" The current comparison answers a less realistic question: "should I use a smaller model with elaborate inference or a larger model with *no* inference-time optimization?" If the larger model with best-of-8 outperforms the smaller model with compute-optimal strategies, the policy implication changes dramatically.

**What evidence exists in the paper.** The FLOPs-matched results (Figure 9, Figure 1 bar charts) show test-time compute outperforming the larger model on easy-to-medium problems at low R values but losing on hard problems and at high R values. However, these results are entirely conditional on the specific (parameter-only, greedy-decoding) baseline. The paper does not provide any comparison against a larger model that also uses test-time compute, nor against a Chinchilla-optimal larger model. The sensitivity of the conclusions to these baseline choices is not explored.

**Mitigation status.** The paper acknowledges the parameter-only scaling deviation from compute-optimal pretraining (Section 7) and frames the Chinchilla-optimal comparison as future work. The greedy decoding choice is not discussed as a limitation. No experiments or analyses explore how the FLOPs-matched conclusions would change under a stronger inference baseline for the larger model. This means the paper's most policy-relevant claim — that test-time compute can substitute for pretraining — is supported only under a specific, relatively weak pretraining baseline, and the magnitude of the effect may be overstated.

---

### Compute-Optimal Strategies Require Pre-Computation Per Difficulty Bin Per Budget Level, Which Does Not Scale to Fine-Grained or Dynamic Allocation

**The assumption or constraint.** The compute-optimal policy is implemented as a **lookup table**: pre-compute, via two-fold cross-validation on a held-out set, which strategy (search algorithm, beam width, lookahead depth, sequential-to-parallel ratio) performs best for each of five difficulty quintiles at each discrete budget level. The paper uses five bins and sweeps budgets at powers of 2 (Section 3.2, Section 5.3).

**The consequence.** This approach has several scaling limitations that prevent finer-grained optimization:

- **The five-bin discretization is coarse.** Questions within the same quintile can have substantially different difficulty — a question at the hard end of bin 3 and one at the easy end of bin 3 receive the identical strategy, even though different strategies might be optimal. A finer-grained partitioning (10 bins, 20 bins) would require proportionally more validation data per bin to reliably estimate the best strategy, and the current 500-question test set already provides only ~50 questions per fold per bin with five bins. The paper does not explore the sensitivity of results to the number of bins or whether performance improves with finer discretization.

- **The strategy is static per question.** Once a question is assigned to a difficulty bin at the start, the entire budget is spent according to the pre-computed strategy for that bin. There is no mechanism for **dynamic adjustment mid-computation** — for instance, starting with a few parallel samples, assessing the verifier's scores on those samples to refine the difficulty estimate, and then allocating the remaining budget differently. Such an adaptive scheme could potentially outperform the static lookup approach, especially for questions near bin boundaries or when the initial difficulty estimate is uncertain.

- **The policy does not generalize across budgets.** The optimal strategy at budget N=16 might differ from the optimal strategy at N=256 for the same difficulty bin (as evidenced by Figure 3 right, where beam search is best at low budgets but best-of-N catches up at high budgets). The lookup table captures this by storing separate strategies per budget level, but this means the number of entries grows with the number of budget levels evaluated. If a deployment needs to support arbitrary budgets (not just powers of 2), the lookup approach doesn't interpolate.

- **Two-fold cross-validation on a 500-question test set provides limited statistical power.** With five bins, each fold contains ~50 questions per bin. The "best" strategy selected on 50 questions may not be reliably better than the second-best strategy. The paper does not report standard errors on per-bin strategy performance, so it is unclear whether the selected strategies are statistically distinguishable from near-optimal alternatives.

**What evidence exists in the paper.** The five-bin discretization is used throughout. The paper reports compute-optimal scaling curves (Figures 4 and 8) that aggregate across bins, but does not compare against finer-grained binning or against continuous difficulty-conditioned policies. The paper does not evaluate adaptive/dynamic allocation strategies. The only evidence that the binning choice is reasonable is that the compute-optimal curves outperform baselines, but this does not establish that five bins are sufficient rather than being arbitrarily chosen. The predicted difficulty bins perform similarly to oracle bins (Figures 4 and 8), which validates the PRM-based difficulty proxy but does not address the discretization question.

**Mitigation status.** The paper does not discuss the limitations of the discretized lookup-table approach or propose finer-grained or dynamic alternatives. The cross-validation protocol is described but its statistical limitations given the 500-question test set are not analyzed. This is an implicit assumption that five bins are sufficient, without evidence or discussion.

---

### The Revision Model's 38% Correct-to-Incorrect Reversion Rate and the Sensitivity of Revision Training to Methodology

**The assumption or constraint.** The revision model is trained exclusively on sequences where all in-context answers are incorrect followed by a correct target answer (Section 6.1). At test time, the model produces a chain of revisions where earlier steps may generate correct answers. The paper reports:

> "approximately 38% of correct answers get converted back to incorrect ones using a naive approach" (Section 6.1, describing the correct-to-incorrect reversion problem)

Additionally, an attempt to further optimize the revision model using ReST^EM (Singh et al., 2024) — an on-policy RL-style training method — caused performance to substantially degrade:

> "additional sequential revisions substantially hurt performance" (Appendix K, Figure 16)

**The consequence.** The revision approach has two fragility issues:

1. **The 38% reversion rate means the revision model actively degrades correct solutions.** The mitigation — using majority voting or verifier-based selection across the entire revision chain rather than taking the final output — is an imperfect patch. It means the system generates revisions that are known to be harmful ~38% of the time when the current answer is correct, and relies on a post-hoc selection mechanism to recover the correct answer. This wastes generation budget (generating revisions that undo correct work) and makes the system's behavior non-monotonic: more revisions don't monotonically improve the probability of a correct final answer. The paper does not explore training the model to recognize when no revision is needed (i.e., including correct-to-correct trajectories in the training data), which would be a more principled solution.

2. **The ReST^EM result (Appendix K) suggests that revision training is highly sensitive to the data generation procedure.** The offline data construction approach (pairing independently sampled correct and incorrect solutions post-hoc, using edit distance to select a "close" incorrect answer) works, but the on-policy RL approach (ReST^EM) fails catastrophically — fully sequential revisions drop to approximately 33.5% accuracy compared to 38.5% at the optimal ratio. The authors hypothesize that "on-policy data collection in ReST^EM exacerbates spurious correlations in revision data," but this explanation is post-hoc and the underlying cause is not diagnosed. This implies that the positive revision results depend on specific, somewhat delicate training choices (offline pairing, edit-distance-based selection, early stopping before validation loss diverges) that may not transfer robustly to new models, domains, or training pipelines.

**What evidence exists in the paper.** The 38% reversion rate is reported in Section 6.1 without a dedicated figure or table quantifying it across difficulty bins or revision depths — it's stated as a single aggregate number. Figure 6 (left) shows pass@1 gradually improving through the revision chain, but this is the post-selection result (after majority voting or verifier selection recovers correct answers that were later reverted). The raw per-step correctness without selection is not shown, so the reader cannot directly observe the reversion pattern. The ReST^EM failure is documented in Appendix K (Figure 16) and briefly discussed.

**Mitigation status.** The reversion problem is partially mitigated by within-chain selection (majority voting or verifier-based selection across all revisions), but the paper does not claim this is a complete solution. The root cause — training only on incorrect-to-correct trajectories — is not addressed. The ReST^EM failure is presented as a negative result without a proposed fix. The paper does not frame revision training robustness as an area for future work, despite the evidence that the approach is fragile.

---

### No Evaluation of Latency or Wall-Clock Time: The Framework Optimizes Generation Count, Not Real Time

**The assumption or constraint.** The entire compute-optimal framework measures test-time compute in **number of generations** (complete solutions sampled), which is a proxy for total FLOPs but ignores **latency** — the wall-clock time required to produce an answer. The paper acknowledges this unit of measurement in Section 3.1 (defining $N$ as the generation budget) but never discusses the latency implications of different strategies.

**The consequence.** Different strategies with the same generation budget can have radically different wall-clock latencies:

- **Sequential revisions are inherently serial.** A strategy that allocates 64 generations as one chain of 64 sequential revisions requires 64 sequential forward passes through the model. Each revision depends on the previous one — they cannot be parallelized. A parallel strategy (best-of-64) can execute all 64 generations simultaneously given sufficient hardware, producing a result in the time of a single forward pass.

- **Beam search has partial serialization.** At each step of beam search, the next set of beams cannot be generated until the current step's beams have been scored and pruned. This introduces step-level serial dependencies that best-of-N avoids.

- **Lookahead search compounds serialization.** Each lookahead rollout of $k$ steps adds $k$ sequential forward passes to the scoring of each beam, further increasing latency.

The compute-optimal policies in Section 5.3 and Section 6 favor strategies that can have high latency despite moderate generation budgets. For example, on easy problems (bin 1–2), the optimal revision strategy is purely sequential revisions (Figure 7 right), which maximizes serial dependency. On medium problems (bin 3), beam search with $M=4$ is preferred over best-of-N (Figure 3 right), introducing step-level serialization.

For **latency-sensitive applications** — interactive chatbots, real-time code completion, voice assistants — the wall-clock time to first token and time between tokens are the user-facing metrics. A strategy that achieves higher accuracy at a given generation budget but takes 10× longer in wall-clock time may be unacceptable regardless of its FLOPs-efficiency. The paper's chatbot and code completion evaluations (Section 6) use datasets from these very applications, making the absence of latency analysis particularly notable.

**What evidence exists in the paper.** None. There is no measurement of wall-clock latency for any strategy or hardware configuration. The generation budget $N$ is used as the sole compute metric throughout. The paper does not discuss the latency implications of sequential vs. parallel strategies, nor does it propose latency-aware allocation policies.

**Mitigation status.** Not addressed. The paper does not acknowledge this as a limitation or suggest latency-aware extensions. For a paper whose primary motivation is meeting real-time SLOs (TTFT and TPOT requirements in Section 1), the absence of any latency analysis in the inference-time compute framework is a significant gap between the analytical setup and practical deployment considerations. The connection between generation budget and wall-clock latency depends on hardware (GPU count, model parallelism), batch size, and strategy serialization, none of which are modeled or measured in the compute-optimal framework.

## 7. Implications and Future Directions
- Landscape impact:
  - Shifts LLM serving design from maximizing aggregate throughput to meeting per-request TTFT/TPOT SLOs cost-effectively. Phase-specialized scheduling and resource allocation become first-class considerations.
- Practical applications:
  - Real-time chat, code assistants, and interactive tools benefit from lower TTFT and stable TPOT without GPU over-provisioning; providers can cut cost per query while meeting user-perceived latency targets (Abstract; §6.2).
- Follow-up research:
  - Integrate preemption and fault tolerance that respect the two-phase dependency structure (§4.3).
  - Reduce memory duplication via shared-weight schemes or memory disaggregation.
  - Extend placement to multi-tenant clusters with fairness/priority and dynamic SLOs.
  - Combine with speculative decoding, long-context models (GQA/MQA, 1M-token contexts), and KV-cache streaming systems; the paper argues disaggregation remains valuable as prefill-vs-decode disparities grow with longer contexts (§7 “Long-context scenarios”).
  - Explore online learning for workload prediction and adaptive batching/parallelism beyond periodic replanning.

> Key quantitative takeaway: “DistServe can serve 7.4× more requests or 12.6× tighter SLO, compared to state-of-the-art systems, while staying within latency constraints for >90% of requests” (Abstract; detailed per-task improvements in Fig. 8–9).
