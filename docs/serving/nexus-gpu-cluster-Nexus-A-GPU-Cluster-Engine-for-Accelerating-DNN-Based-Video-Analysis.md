## 1. Executive Summary

Nexus addresses the critical challenge of serving Deep Neural Networks (DNNs) for video analysis at high GPU utilization while strictly adhering to latency Service Level Objectives (SLOs), a problem where conventional systems like TensorFlow Serving and Clipper fail due to inefficient batching of specialized models and coarse-grained scheduling. By introducing "squishy bin packing" to co-schedule variable-sized batches, automated query optimization for multi-stage pipelines, and "prefix batching" to jointly execute the shared layers of specialized models (e.g., variants of ResNet-50), Nexus achieves request throughput rates **1.8–12.7× higher** than state-of-the-art baselines on 16-GPU clusters while maintaining a **99%** success rate within latency constraints. In large-scale deployments on **100 GPUs** across **7 applications** and **12 models**, the system sustains **84%** of theoretical optimal utilization and violates latency SLOs on only **0.27%** of requests.

## 2. Context and Motivation

To understand the necessity of Nexus, one must first appreciate the economic and technical paradox of modern Deep Neural Network (DNN) inference: while GPUs offer massive computational power at a low cost per operation, they are notoriously difficult to keep busy. This section dissects the specific gap Nexus fills, the economic imperative driving it, the limitations of prior systems, and how Nexus redefines the scheduling landscape.

### The Core Problem: The Utilization-Latency Trade-off in Video Analysis

The paper targets a specific, high-stakes scenario: **cloud-scale video analysis services**. Imagine a platform where thousands of tenants simultaneously analyze thousands of video streams (e.g., traffic monitoring, game stream analysis, security surveillance). As illustrated in **Figure 1**, these pipelines typically involve sampling frames, applying lightweight logic to find regions of interest, and then running heavy DNNs on those regions.

The fundamental problem is not simply "running DNNs fast." It is running them **efficiently** on a cluster of GPUs while strictly adhering to **latency Service Level Objectives (SLOs)**.

*   **The Economic Driver:** GPUs are expensive capital assets with immense capacity (e.g., >100 TFLOPS). **Table 1** quantifies the disparity: executing a ResNet-50 model on a CPU costs roughly **$4.22** per 1,000 invocations and takes **1,130 ms**, whereas a GPU reduces the cost to **$0.48** and latency to **6.2 ms**. However, this cost advantage only materializes if the GPU is kept at **sustained high utilization**.
*   **The Utilization Gap:** A single video stream rarely generates enough frames to saturate a GPU. For instance, **Section 2.1** calculates that to fully utilize an NVIDIA V100 (125 TFLOPS) on a tiny model like LeNet (20 MOPs), the system would need to process **6.25 million inputs per second**. No single application provides this volume. Therefore, the system must aggregate work across many tenants and applications.
*   **The Conflict:** Aggregating work requires **batching** (grouping multiple inputs to process simultaneously). However, batching increases latency. If the batch grows too large, the processing time exceeds the SLO (e.g., a "live" alert must happen within tens of milliseconds). The core challenge is a complex optimization problem: *How do we pack variable-sized batches from diverse applications onto shared GPUs to maximize throughput without ever violating individual latency deadlines?*

### Why Prior Approaches Fall Short

Before Nexus, the state-of-the-art relied on systems like **Clipper** and **TensorFlow Serving**. While these systems introduced adaptive batching, they operate under assumptions that break down in large-scale, heterogeneous video analysis workloads. The paper identifies three critical deficiencies in these prior approaches:

#### 1. The "Whole-Model" Batching Constraint
Conventional systems assume that batching is only possible when the **entire** model is identical for all inputs.
*   **The Reality of Transfer Learning:** Modern applications rarely use generic models. Instead, they use **specialized models** created via transfer learning—taking a base model (like ResNet-50) and retraining only the final layers to detect specific objects (e.g., "Ford F-150" vs. "Toyota Camry").
*   **The Failure Mode:** Because specialized models differ in their final layers, conventional systems treat them as completely distinct entities. They cannot batch a request for "Ford detection" with "Toyota detection," even though 95% of the computation (the early layers) is identical. This forces the GPU to run many small, inefficient batches, destroying utilization.

#### 2. Coarse-Grained "Bin Packing"
Existing schedulers treat model execution as a static bin-packing problem where the "size" of a task is fixed.
*   **The "Squishy" Reality:** In DNN inference, the cost of a task is not fixed; it is **"squishy."** As defined in **Section 4**, the processing cost of an input depends entirely on the batch size it is grouped into. A batch of 4 might take 50ms, while a batch of 16 might take 100ms.
*   **The Failure Mode:** Prior schedulers cannot reason about this variability. They often allocate a GPU to a single model type or use simple round-robin scheduling that ignores the complex interplay between batch size, duty cycle, and latency SLOs. They fail to solve what Nexus calls **"squishy bin packing"**: packing tasks where the ball size changes depending on how many other balls are in the bin.

#### 3. Lack of Query-Level Optimization
Video analysis pipelines are rarely single-model operations. They are often **multi-stage queries** (e.g., Step 1: Detect car; Step 2: Recognize license plate).
*   **The Failure Mode:** Systems like Clipper treat each stage independently. If a user specifies a 400ms SLO for the whole query, prior systems might arbitrarily split this evenly (200ms per stage) or rely on static configurations.
*   **The Missed Opportunity:** As shown in **Section 4.2**, the optimal split depends on the workload characteristics (e.g., does the first stage filter out 90% of images, or expand them?). A static or naive split leaves performance on the table, bottlenecking the entire pipeline.

### Positioning Nexus: A Paradigm Shift in Granularity and Control

Nexus positions itself not merely as an incremental improvement but as a fundamental re-architecture of the serving stack, moving from **opaque container-based serving** to **fine-grained cluster-scale orchestration**.

| Feature | Prior Systems (Clipper, TF Serving) | Nexus Approach |
| :--- | :--- | :--- |
| **Batching Unit** | **Whole Model:** Requires exact model match. Specialized variants cannot be batched together. | **Prefix Batching:** Identifies common subgraphs (prefixes) across specialized models and batches them jointly, executing unique suffixes sequentially (**Section 6.3**). |
| **Scheduling Logic** | **Static/Fixed Cost:** Assumes fixed execution times; struggles with variable batch impacts on latency. | **Squishy Bin Packing:** Explicitly models execution latency as a function of batch size ($\ell(b) = \alpha b + \beta$) to optimize duty cycles (**Section 6.1**). |
| **Optimization Scope** | **Per-Model:** Optimizes individual model throughput in isolation. | **Query-Aware:** Automatically splits end-to-end latency SLOs across multi-stage pipelines to maximize global throughput (**Section 6.2**). |
| **Resource Scope** | **Node-Level:** Often relies on external cluster managers (Mesos, Kubernetes) for placement, lacking DNN-specific awareness. | **Cluster-Wide:** A global scheduler that performs detailed GPU placement, reasoning about co-scheduling groups of DNNs to minimize interference (**Section 5**). |

### Theoretical and Practical Significance

The significance of Nexus lies in its resolution of the **NP-hard scheduling problem** inherent in this domain. **Appendix A** proves that even a restricted version of this scheduling problem (Fixed-rate GPU Scheduling) is strongly NP-complete. Prior work often sidestepped this complexity with heuristics that sacrificed efficiency.

Nexus confronts this complexity head-on by:
1.  **Decoupling Control and Data Planes:** It introduces a slow control plane (seconds/minutes) for solving the complex "squishy bin packing" and query optimization, and a fast data plane (milliseconds) for executing the resulting schedule (**Figure 6**).
2.  **Embracing Heterogeneity:** Instead of forcing uniformity, it exploits the structural similarities in heterogeneous workloads (via prefix batching) to recover the batching gains that transfer learning traditionally destroys.

In essence, Nexus shifts the abstraction level from "serving a model" to "serving a cluster of DNN fragments," enabling the high utilization required to make cloud-scale video analysis economically viable.

## 3. Technical Approach

This section details the architectural and algorithmic innovations that allow Nexus to solve the "squishy bin packing" problem. Unlike prior systems that treat DNN serving as a simple queueing problem, Nexus treats it as a complex, multi-variable optimization task where the cost of computation changes dynamically based on how requests are grouped. We will walk through the system's three-plane architecture, the mathematical formulation of its scheduler, the logic behind its query optimizer, and the runtime mechanisms that enforce these decisions on the GPU.

### 3.1 Reader orientation (approachable technical breakdown)
Nexus is a cluster-scale operating system specifically designed to slice, dice, and reassemble Deep Neural Network (DNN) computations across many GPUs to maximize hardware usage without missing deadlines. It solves the problem of inefficient GPU utilization in video analysis by breaking the rigid rule that "only identical models can be batched together," instead allowing the system to batch shared parts of different models and dynamically adjust batch sizes based on real-time traffic patterns.

### 3.2 Big-picture architecture (diagram in words)
The system operates across three distinct logical planes that interact at different time scales, as depicted in **Figure 6**:
*   **The Management Plane (Hours to Weeks):** This is the ingestion layer where developers upload DNN models and application containers. Its primary responsibility is to profile each model to generate a "batching profile"—a lookup table that maps batch sizes to execution latencies and memory usage.
*   **The Control Plane (Seconds to Minutes):** This is the brain of the system, centered around a **Global Scheduler**. It ingests real-time workload statistics (request rates) and latency SLOs from the data plane. Its responsibility is to solve the complex "squishy bin packing" optimization problem, deciding exactly which models run on which GPUs, what batch sizes they should use, and how to split latency budgets for multi-stage queries. It outputs two critical data structures: a **Routing Table** (sent to frontends) and an **Execution Schedule** (sent to backends).
*   **The Data Plane (Milliseconds to Seconds):** This is the high-speed execution layer consisting of distributed **Frontends** (inside application containers) and **Backends** (on GPU nodes). Frontends receive user requests, consult the Routing Table to find the correct backend, and dispatch the request. Backends receive streams of requests, buffer them according to the Execution Schedule, perform "prefix batching" on the GPU, and return results.

### 3.3 Roadmap for the deep dive
To fully understand how Nexus achieves its performance gains, we will dissect the system in the following logical order:
1.  **The Scheduling Core (Squishy Bin Packing):** We first explain the fundamental algorithm that decides how to pack variable-sized batches onto GPUs, as this is the primary innovation over prior static schedulers.
2.  **Query-Level Optimization:** We then expand the scope to multi-stage pipelines, explaining how Nexus automatically splits latency budgets across dependent models to prevent bottlenecks.
3.  **Runtime Execution & Prefix Batching:** We dive into the GPU node itself to explain how Nexus physically executes batches of "different" models by identifying and merging their common computational prefixes.
4.  **Adaptive Dispatch & Rate Control:** Finally, we cover the feedback loop mechanisms that handle bursty traffic, ensuring the system drops requests early rather than failing latency SLOs due to inefficient small batches.

### 3.4 Detailed, sentence-based technical breakdown

#### The Three-Plane Architecture and Data Flow
Nexus decouples the slow, complex optimization logic from the fast, latency-critical data path to ensure stability and performance.
*   **Management and Profiling:** When a new model is uploaded, the system does not assume its performance characteristics; instead, a profiler executes the model with varying batch sizes to construct a precise **batching profile**. This profile captures the non-linear relationship between batch size and latency, which is critical because DNN latency does not scale linearly with input count due to memory access overheads.
*   **Control Loop Dynamics:** The Global Scheduler operates in discrete time intervals called **epochs**, typically lasting **30 to 60 seconds**, though the system enforces a minimum epoch duration of **10 seconds** to prevent oscillation from frequent reconfigurations. During each epoch, the scheduler collects request rate statistics ($R_i$) and current latency SLOs ($L_i$) from the data plane, runs its optimization algorithms, and pushes updated routing tables and schedules to the nodes.
*   **Data Plane Execution:** Upon receiving a request, the frontend library performs a lightweight lookup in its local routing table to identify the specific backend node and GPU assigned to that model session. The backend node maintains a queue for each assigned model; however, instead of executing immediately, it waits until the accumulated requests match the **target batch size** dictated by the current epoch's schedule, or until the latency budget for the oldest request is about to expire.

#### Algorithm 1: Squishy Bin Packing for Individual Sessions
The core scheduling challenge is that the "size" of a task (its execution time) shrinks as you pack more of them together, a phenomenon Nexus terms **"squishy bin packing."**
*   **Problem Formulation:** The scheduler aims to minimize the total number of GPUs used while ensuring that for every session $i$, the worst-case latency does not exceed $L_i$. The worst-case latency for a request is defined as the sum of the **duty cycle** ($d$, the time to cycle through all models on a GPU) and the **batch execution time** ($\ell(b)$).
*   **Handling Saturated Sessions:** For models with very high request rates that require one or more dedicated GPUs, the scheduler first calculates the maximum feasible batch size $B_i$ such that $2 \cdot \ell_{k_i}(B_i) \leq L_i$. The factor of 2 accounts for the worst-case scenario where a request misses the current batch and must wait for the next one. The system allocates $\lfloor R_i / T_i \rfloor$ dedicated GPUs, where $T_i$ is the throughput at batch size $B_i$, and passes the remaining "residual" load to the next stage.
*   **Residual Load Optimization:** For the remaining low-rate sessions that do not fill a GPU, Nexus formulates an integer programming problem to pack multiple sessions onto shared GPUs. The objective is to minimize $\sum_j g_j$ (the number of active GPUs), subject to constraints ensuring that the sum of batch execution times for all sessions on a GPU fits within the duty cycle $d_j$, and that $d_j + \ell_{k_i}(b_{ij}) \leq L_i$ for every session $i$ on GPU $j$.
*   **Greedy Approximation Strategy:** Since the exact integer programming solution is computationally expensive (taking hours for 25 sessions), Nexus employs a greedy heuristic inspired by the **Best-Fit Decreasing** algorithm. It sorts residual sessions by their "occupancy" (the fraction of the duty cycle they consume) in descending order.
*   **Merging Logic:** The algorithm attempts to merge sessions into existing duty cycles by reducing the duty cycle to the minimum of the merged components ($d = \min(d_1, d_2)$). It then recalculates the required batch size for the session with the larger original duty cycle to fit the new, tighter cycle ($b'_1 = d \cdot r_1$). If the sum of the new batch execution times fits within the new duty cycle ($\ell(b'_1) + \ell(b_2) \leq d$) and memory constraints are met, the merge is accepted. This approach explicitly handles the "squishy" nature of the problem by dynamically adjusting batch sizes during the packing process.

#### Algorithm 2: Complex Query Optimization
Video analysis applications often consist of dependent stages (e.g., detect object $\to$ recognize identity), forming a directed acyclic graph (DAG) of models.
*   **The Latency Split Challenge:** A user specifies a single end-to-end latency SLO ($L$) for the entire query, but the system must decide how to partition this budget among the individual stages ($L_1, L_2, \dots$). An even split is often suboptimal because different stages have different batching efficiencies and filtering properties.
*   **Throughput Coupling:** The throughput of a pipeline is constrained by its slowest stage relative to the data volume it processes. If stage $X$ produces $\gamma$ outputs for every input (where $\gamma$ is the expansion factor), the system must ensure that the downstream stage $Y$ can process $\gamma \cdot R_X$ requests. The optimization goal is to find batch sizes $b_v$ for each node $v$ that minimize the total GPU count $\sum \frac{R_v \cdot \ell_v(b_v)}{b_v}$ subject to the constraint that the sum of latencies along any path from root to leaf is $\leq L$.
*   **Dynamic Programming Solution:** For tree-structured dependency graphs, Nexus uses dynamic programming to solve this efficiently. It defines a function $f(u, t)$ representing the minimum GPUs required to execute the subtree rooted at model $u$ within a time budget $t$.
*   **State Space Discretization:** To make the continuous time budget solvable, the algorithm discretizes the time domain into segments of length $\epsilon$, resulting in a state space of size $L/\epsilon$. The algorithm iterates through all possible time splits $k$ for the current node and the remaining budget $t-k$ for its children, selecting the split that minimizes the total GPU count. The time complexity of this approach is quadratic in $L/\epsilon$, allowing it to run efficiently within the epoch timescale.

#### Runtime Mechanism: Prefix Batching and GPU Multiplexing
Even with perfect scheduling, efficiency is lost if the GPU cannot physically batch specialized models. Nexus breaks the conventional "whole-model" batching barrier.
*   **Identifying Common Prefixes:** When models are ingested, Nexus computes a hash of every sub-tree in the model's computational graph. By comparing these hashes, the system identifies when two different models (e.g., a "Ford detector" and a "Toyota detector") share identical early layers (the prefix) but differ only in their final classification layers (the suffix).
*   **Fragmented Execution:** At runtime, instead of loading two separate full models, the backend loads the shared prefix once. Requests for both models are aggregated into a single large batch and processed through the shared prefix layers simultaneously.
*   **Suffix Separation:** After the prefix computation, the batch is split back into sub-batches corresponding to the specific suffix required for each request. These smaller sub-batches are then processed sequentially through their respective unique output layers. This technique recovers the batching efficiency that transfer learning typically destroys, allowing the system to amortize the fixed cost of the heavy prefix computation over a much larger number of requests.
*   **CPU-GPU Overlapping:** To further maximize utilization, the backend employs a multi-threaded architecture where CPU threads handle pre-processing (decoding, resizing) and post-processing for the *next* and *previous* batches, respectively, while a dedicated thread launches the GPU kernel for the *current* batch. This ensures the GPU never idles waiting for CPU preparation, a bottleneck that **Figure 10** shows can reduce throughput by up to **7.4×** in tight-SLO scenarios.

#### Adaptive Batching and Early Drop Policy
In bursty traffic conditions, strictly adhering to a "first-come, first-served" policy can force the system into inefficient small batches, causing a cascade of latency violations.
*   **The Failure of Lazy Dropping:** Traditional systems use "lazy dropping," where a request is only dropped after it has already missed its deadline. **Figure 5** demonstrates that this approach fails under Poisson arrival patterns because the system wastes resources trying to process an old request with a tiny remaining time budget, forcing a small batch size that increases the fixed cost per request for everyone else.
*   **Early Drop Strategy:** Nexus implements an "early drop" policy where the dispatcher scans the request queue with a sliding window equal to the optimal batch size determined by the scheduler. It identifies the first request that has sufficient time budget to accommodate the full batch execution latency.
*   **Aggressive Pruning:** All requests preceding this "feasible" request are immediately dropped, regardless of whether they have technically timed out yet. This counter-intuitive move sacrifices a few requests to preserve the large batch size for the remaining majority, thereby maintaining high GPU throughput and ensuring that the **99%** of served requests meet their SLOs. **Figure 9** confirms that this strategy yields up to **25%** higher throughput compared to lazy dropping under variable load conditions.

#### Design Choices and Trade-offs
Nexus makes several deliberate design choices that distinguish it from general-purpose cluster managers like Mesos or Kubernetes.
*   **Centralized vs. Distributed Scheduling:** Unlike fully distributed schedulers that react instantly but locally, Nexus uses a centralized Global Scheduler running at the epoch granularity (30-60s). This choice allows the system to solve the globally optimal "squishy bin packing" problem, which requires a holistic view of all workloads, rather than making greedy local decisions that lead to fragmentation.
*   **Deterministic Scheduling vs. Best-Effort:** The system prioritizes deterministic adherence to latency SLOs over fairness or throughput maximization in isolation. By calculating strict duty cycles and enforcing them via the runtime, Nexus guarantees that no single "noisy neighbor" application can starve others, a common failure mode in multi-tenant GPU clusters.
*   **Model Specialization Support:** The decision to implement prefix batching at the system level rather than requiring application-level changes is crucial. It allows developers to use standard transfer learning workflows (retraining only the last layer) while automatically gaining the performance benefits of batching, removing the burden of manual model fusion from the application developer.

## 4. Key Insights and Innovations

Nexus does not merely optimize existing serving pipelines; it fundamentally redefines the abstractions used to schedule GPU workloads. The system's success relies on three core insights that challenge conventional wisdom in deep learning systems: that batching units can be smaller than whole models, that task costs are dynamic variables rather than static constants, and that sacrificing individual requests can preserve cluster-wide efficiency.

### 4.1 Innovation: Prefix Batching as a First-Class Primitive
**The Insight:** The most significant conceptual leap in Nexus is the decoupling of the "batching unit" from the "model identity." Prior systems (Clipper, TensorFlow Serving) operate on the assumption that batching is only valid when inputs are processed by *identical* computational graphs. This assumption renders transfer learning—a ubiquitous practice where models share 90%+ of their layers but differ in final classification heads—disastrous for utilization, as it forces the GPU to process many small, isolated batches.

**Why It Is Fundamental:** Nexus introduces **Prefix Batching** (Section 6.3), treating the common subgraphs of specialized models as independent, batchable entities.
*   **Distinction from Prior Work:** Previous approaches like MCDNN or Mainstream focused on sharing prefixes to *avoid redundant computation* for a single input across multiple models. Nexus flips this logic: it shares prefixes to *aggregate distinct inputs* from different models into a single large batch.
*   **Significance:** This innovation recovers the batching efficiency that model specialization traditionally destroys. As demonstrated in **Figure 15**, this allows throughput to scale linearly with the number of model variants, achieving up to **110% higher throughput** compared to non-prefixed execution. More critically, it alters the memory economics of the cluster: **Figure 15(b)** shows that adding new specialized models incurs negligible memory overhead (only the unique suffix layers), whereas conventional systems would require loading full model copies, quickly exhausting GPU memory. This transforms the cluster from a collection of static model silos into a fluid pool of compute fragments.

### 4.2 Innovation: "Squishy" Bin Packing and Dynamic Duty Cycles
**The Insight:** Classical bin-packing algorithms assume items have fixed sizes. Nexus recognizes that in DNN serving, the "size" of a task (its execution time) is a **squishy variable** that shrinks as more items are packed together. The cost of processing a request is not intrinsic to the model; it is a function of the batch size it is grouped into, governed by the latency equation $\text{batch\_lat}(b) = \alpha b + \beta$ (Equation 1).

**Why It Is Fundamental:** This insight necessitates a new class of scheduling algorithms that simultaneously solve for *placement* and *batch size*.
*   **Distinction from Prior Work:** Conventional schedulers allocate resources based on peak or average throughput, treating latency as a constraint to be checked post-allocation. Nexus's **Squishy Bin Packing** (Section 6.1) integrates the batching profile directly into the packing logic. It calculates a **duty cycle** for each GPU—the time required to cycle through all assigned models—and dynamically adjusts the batch size of each model to fit within that cycle while meeting its specific SLO.
*   **Significance:** This approach unlocks utilization in low-to-moderate load regimes where dedicated GPUs are wasteful. By merging residual loads from multiple applications onto a single GPU and shrinking their batch sizes just enough to fit the combined duty cycle, Nexus achieves **84% of theoretical optimal utilization** on mixed workloads (Section 7.4). The theoretical weight of this innovation is underscored by **Appendix A**, which proves that even a restricted version of this scheduling problem is **strongly NP-complete**, justifying Nexus's move away from simple heuristics to a sophisticated, profile-guided greedy approximation.

### 4.3 Innovation: Query-Aware Latency Splitting via Dynamic Programming
**The Insight:** In multi-stage video pipelines (e.g., Detect $\to$ Track $\to$ Recognize), the end-to-end latency SLO is a shared resource. A naive, even split of this budget across stages often creates artificial bottlenecks. Nexus identifies that the optimal split depends on the **expansion factor** ($\gamma$) of each stage—whether a stage filters data ($\gamma < 1$), passes it through ($\gamma = 1$), or expands it ($\gamma > 1$, e.g., detecting multiple cars in one frame).

**Why It Is Fundamental:** This shifts the optimization scope from the *model* level to the *query* level.
*   **Distinction from Prior Work:** Existing systems typically require developers to manually tune latency budgets per stage or rely on static, equal splits. Nexus automates this via a **Dynamic Programming** algorithm (Section 6.2) that explores the state space of possible time allocations to minimize total GPU consumption.
*   **Significance:** The performance impact is non-linear and workload-dependent. As shown in **Figure 17**, when the expansion factor $\gamma$ is high (one input yields many outputs), the downstream stage becomes the bottleneck, requiring a larger latency budget to enable larger batches. Nexus's automated splitter adapts to these dynamics, yielding **13–55% higher throughput** compared to even-split baselines. This capability allows Nexus to treat complex, multi-model applications as single schedulable units, maximizing global cluster efficiency rather than local stage efficiency.

### 4.4 Innovation: The "Early Drop" Policy for Throughput Preservation
**The Insight:** In high-variance traffic patterns, strictly adhering to a First-Come-First-Served (FCFS) discipline can be catastrophic for throughput. If the system waits for an old, straggling request to form a batch, it may be forced to execute a tiny batch to meet that request's deadline, thereby amortizing the fixed kernel launch cost ($\beta$) over very few items.

**Why It Is Fundamental:** Nexus challenges the standard reliability metric of "minimize dropped requests" in favor of "maximize throughput of served requests within SLO."
*   **Distinction from Prior Work:** Systems like Clipper use **Lazy Dropping**, discarding a request only after it has definitively missed its deadline. **Figure 5** illustrates how this leads to poor performance under Poisson arrivals, as the system chases deadlines with inefficient small batches. Nexus implements **Early Drop** (Section 6.3), proactively discarding requests that *would* force a sub-optimal batch size, even if they haven't technically timed out yet.
*   **Significance:** This counter-intuitive strategy preserves the "sweet spot" of batch sizes for the majority of traffic. By sacrificing a small fraction of requests early, the system maintains high GPU utilization for the remaining 99%. **Figure 9** quantifies this gain, showing that Early Drop achieves up to **25% higher throughput** than Lazy Dropping while maintaining the same SLO compliance rate. This represents a philosophical shift in system design: prioritizing the efficiency of the *collective* workload over the latency of the *individual* outlier.

### Summary of Contributions
| Innovation | Type | Key Differentiator | Impact |
| :--- | :--- | :--- | :--- |
| **Prefix Batching** | Fundamental | Batches *fragments* of different models, not just whole identical models. | Enables efficient transfer learning; **110%** throughput gain; reduces memory footprint. |
| **Squishy Bin Packing** | Fundamental | Treats task cost as a dynamic variable dependent on batch size. | Solves NP-hard packing problem; achieves **84%** of theoretical optimal utilization. |
| **Query-Aware Splitting** | Fundamental | Optimizes latency budgets across multi-stage DAGs based on data expansion. | Adapts to workload dynamics; **13–55%** throughput gain over static splits. |
| **Early Drop Policy** | Strategic | Proactively drops requests to preserve optimal batch sizes. | Prevents throughput collapse under bursty loads; **25%** gain over lazy dropping. |

These innovations collectively move the field from **static, model-centric serving** to **dynamic, cluster-centric orchestration**, proving that high utilization and strict latency SLOs are not mutually exclusive if the system is willing to reason about the internal structure of DNNs and the statistical nature of the workload.

## 5. Experimental Analysis

The evaluation of Nexus is designed to rigorously test its core hypothesis: that fine-grained, batching-aware scheduling and execution can drastically improve GPU utilization in video analysis clusters without violating strict latency Service Level Objectives (SLOs). The authors employ a multi-faceted experimental strategy, ranging from controlled single-application case studies to large-scale, multi-tenant deployments, supported by detailed ablation studies to isolate the contribution of each system component.

### 5.1 Evaluation Methodology

#### Workloads and Datasets
The evaluation utilizes **seven distinct video analysis applications** modeled after real-world scenarios, as detailed in **Table 4**. These workloads vary significantly in complexity, model architecture, and data characteristics:
*   **Game Analysis:** Recognizes numbers and icons in Twitch streams (20 streamers, 1 week of data). This workload uses tiny, specialized models (LeNet variants) and a larger ResNet-50 variant.
*   **Traffic Monitoring:** A two-stage pipeline (Object Detection $\to$ Face/Car Recognition) processing 20 camera feeds. It employs SSD, VGG-Face, and GoogleNet-car models.
*   **Other Applications:** Include dance performance rating, billboard audience analysis, bike rack occupancy, Amber Alert vehicle matching, and corporate logo auditing.
*   **Data Characteristics:** The inputs range from 24/7 live streams to finite video files looped for simulation. Crucially, the workloads exhibit varying degrees of **model specialization** (amenable to prefix batching) and **query complexity** (ranging from 1-stage "QA-1" to 5-stage "QA-5" pipelines).

#### Metrics and Success Criteria
The primary metric for success is **Throughput**, defined as the maximum request rate the system can sustain while ensuring that **at least 99%** of requests are served within their latency SLO.
*   **Bad Rate:** The percentage of requests that either exceed their latency deadline or are explicitly dropped by the system. The target is a bad rate $< 1\%$.
*   **Utilization Efficiency:** Measured as the ratio of GPUs used by Nexus to the theoretical lower bound (the minimum number of GPUs required if models ran at peak isolated throughput with perfect batching).

#### Baselines and Competitors
Nexus is compared against two state-of-the-art baselines:
1.  **Clipper:** A prediction serving system that supports adaptive batching but lacks cluster-scale, DNN-aware scheduling.
2.  **TensorFlow Serving (TF Serving):** A high-performance serving system that lacks native support for latency SLOs and cross-model batching.

To ensure a fair comparison, the authors implemented a **batch-oblivious scheduler** for the baselines. This scheduler allocates GPUs proportionally to request rates and splits query latency budgets evenly across stages—strategies that Nexus explicitly aims to improve upon. Notably, Clipper's internal adaptive batching was retained, as it is orthogonal to the cluster-level scheduling innovations being tested.

#### Experimental Setup
Experiments were conducted on commercial cloud infrastructure using two cluster scales:
*   **Small Scale:** A **16-GPU** cluster (NVIDIA GTX 1080 Ti or K80) for single-application case studies and ablation tests.
*   **Large Scale:** A **100-GPU** cluster (NVIDIA K80) for multi-application, long-running deployments to test scalability and stability under variable loads.

### 5.2 Single-Application Case Studies

The authors first isolate specific applications to quantify performance gains under controlled conditions.

#### Game Analysis: The Power of Prefix Batching and Overlap
In the game analysis scenario, the system must process frames with a tight **50ms latency SLO**. The workload involves tiny LeNet variants and a ResNet-50 variant.
*   **Quantitative Results:** As shown in **Figure 10**, Nexus achieves a throughput of **4,120 req/s**. In stark contrast, Clipper manages only **324 req/s**, and TF Serving reaches **440 req/s**. This represents a massive speedup of **9.4× to 12.7×**.
*   **Ablation Insights:** The ablation study reveals why the baselines fail.
    *   **Overlapping (OL):** Disabling CPU-GPU overlapping causes throughput to plummet by **7.4×** (from 4,120 to roughly 550 req/s, inferred from the chart bars). The tight 50ms SLO combined with ~10ms preprocessing time forces small batch sizes; without overlapping, the GPU sits idle waiting for the CPU, destroying utilization.
    *   **Prefix Batching (PB):** While OL is the dominant factor here, disabling all Nexus features (PB, Squishy Scheduling, Early Drop, OL) results in a **48% drop** in throughput relative to the full system (falling to ~2,143 req/s), proving that multiple optimizations contribute synergistically.
    *   **Baseline Failure:** The baselines performed so poorly on the tiny LeNet models that the authors excluded them from the final comparison to be "maximally fair," reporting results where baselines only ran the ResNet model. Even with this advantage, Nexus still outperformed them by over 4×.

#### Traffic Monitoring: Query Optimization and Robustness
The traffic application involves a heavier, two-stage pipeline (Detection $\to$ Recognition) with a relaxed **400ms SLO**.
*   **Quantitative Results:** **Figure 11** shows Nexus achieving **534 req/s**, compared to **227 req/s** for TF Serving and **297 req/s** for Clipper. This is a **1.8× to 2.4×** improvement.
*   **Query Analysis (QA) Impact:** Since this is a multi-stage pipeline, the **Query Analysis** feature is critical. Disabling QA (forcing an even latency split) reduces throughput by **19%** (from 534 to 433 req/s). This confirms that intelligently allocating more time to the bottleneck stage (object detection) yields significant gains.
*   **Diurnal Variation:** **Figure 12** compares performance during "rush hour" (high complexity, more objects per frame) vs. "non-rush hour."
    *   Throughput naturally drops during rush hour due to increased computational load per frame.
    *   However, Nexus maintains its relative advantage. During rush hour, Nexus achieves **264 req/s** vs. **146 req/s** for TF Serving (**1.8×** gain). While the relative gain is slightly lower than the non-rush hour **2.4×**, the system remains robust under heavy load.

### 5.3 Large-Scale Multi-Application Deployment

To validate scalability, Nexus was deployed on a **100-GPU cluster** running all seven applications simultaneously with varying Poisson arrival rates.

*   **Optimality:** On a controlled 16-GPU setup, Nexus used an average of **11.7 GPUs** to handle the workload, while the theoretical lower bound was **9.8 GPUs**. This indicates Nexus achieves **84%** of the theoretical optimal utilization, a remarkably high efficiency for a system handling heterogeneous, latency-constrained workloads.
*   **Adaptability:** **Figure 13** illustrates Nexus's response to a sudden workload spike.
    *   When request rates surged at $t=326s$, Nexus detected the change within **12 seconds** and automatically provisioned additional GPUs.
    *   When demand subsided at $t=644s$, it de-provisioned resources with a **10-second lag**.
    *   Throughout this volatility, the system maintained an average bad rate of only **0.27%**, well below the 1% threshold. Sporadic spikes above 1% occurred only during the brief reconfiguration windows, demonstrating stable control plane dynamics.

### 5.4 Micro-Benchmarks and Sensitivity Analysis

The authors conducted targeted micro-benchmarks to isolate the contribution of specific algorithms.

#### GPU Multiplexing
**Figure 14(a)** compares Nexus against baselines when running multiple copies of the Inception model on a single GPU.
*   **Interference Management:** TF Serving and Clipper suffer from interference as model count increases. Nexus, by explicitly scheduling duty cycles, achieves **1.4–2.1×** higher throughput than TF Serving and **1.9–9.8×** higher than Clipper.
*   **SLO Sensitivity:** **Figure 14(b)** shows that as the latency SLO relaxes (50ms $\to$ 200ms), the performance gap narrows but Nexus consistently leads, leveraging the extra slack to form larger, more efficient batches.

#### Prefix Batching Scalability
**Figure 15** analyzes the impact of adding specialized model variants (ResNet-50 with different final layers).
*   **Throughput:** With 10 model variants, prefix batching yields **110% higher throughput** compared to executing them separately. Without prefix batching, the system is forced to run small sub-batches to meet SLOs, drastically reducing efficiency.
*   **Memory Efficiency:** **Figure 15(b)** highlights a critical resource saving. Adding variants with only 1 unique fully connected layer ("1 FC") consumes negligible extra GPU memory with prefix batching. In contrast, without it, the GPU runs out of memory almost immediately as full model copies are loaded. This confirms that prefix batching is essential for hosting diverse, specialized models on limited hardware.

#### Squishy Scheduling Robustness
**Figure 16** compares Nexus's squishy bin packing against a batch-oblivious baseline across five different workload mixes (varying models, SLOs, and request rates).
*   **Consistent Gains:** Nexus outperforms the baseline in every scenario. The gains are most pronounced (**up to 64%**) when handling mixed request rates (Zipf distribution), where the ability to merge residual loads is most valuable. Even in the simplest mix (fixed models/SLOs), Nexus provides an **11%** gain.

#### Complex Query Analysis
**Figure 17** evaluates the query optimizer under different expansion factors ($\gamma$) and SLOs.
*   **Dynamic Adaptation:** The baseline (even split) performs poorly when $\gamma$ is high (one input generates many outputs), as it starves the downstream stage of time. Nexus adapts the split, achieving **13–55% higher throughput**. The gain is largest (55%) when $\gamma=10$ and the SLO is tight (300ms), precisely where intelligent scheduling matters most.

#### Early Drop Policy
**Figure 9** contrasts the "Early Drop" policy with the conventional "Lazy Drop."
*   **Throughput Preservation:** Under Poisson arrivals with high fixed costs (small $\alpha$), lazy dropping forces the system into inefficient small batches, collapsing throughput. Early Drop proactively sacrifices stragglers to maintain optimal batch sizes, yielding up to **25% higher throughput** while maintaining the same 99% SLO compliance.

### 5.5 Critical Assessment

The experimental results convincingly support the paper's claims. The evidence is robust across multiple dimensions:
1.  **Magnitude of Gains:** The throughput improvements (1.8× to 12.7×) are not marginal; they represent a fundamental shift in efficiency. The largest gains occur in scenarios specifically designed to break prior systems: tight SLOs with small models (Game) and highly specialized models (Prefix Batching).
2.  **Attribution:** The ablation studies are thorough. They successfully decompose the total gain into contributions from overlapping, prefix batching, squishy scheduling, and query analysis. For instance, the dramatic drop in Game analysis performance when disabling Overlapping (OL) clearly identifies the bottleneck in tight-SLO regimes, while the Traffic analysis ablation highlights the value of Query Analysis (QA).
3.  **Scalability:** The 100-GPU deployment demonstrates that the centralized scheduler does not become a bottleneck. The system reacts to load changes within seconds and maintains high utilization (84% of optimal) even with complex, mixed workloads.
4.  **Realism:** The use of real video datasets (Twitch, traffic cameras) and realistic arrival patterns (Poisson, Zipf) strengthens the validity of the results. The inclusion of "rush hour" vs. "non-rush hour" analysis adds a layer of practical relevance often missing in systems papers.

**Limitations and Trade-offs:**
*   **Epoch Latency:** The system relies on an epoch-based scheduler (30-60s). While effective for minute-scale trends, it may react slower than purely local, reactive schedulers to sub-second bursts, relying on the "Early Drop" policy to bridge the gap. The paper acknowledges this trade-off but shows the bad rate remains low (0.27%).
*   **Implementation Complexity:** The solution requires significant engineering effort (10k lines of C++, custom profilers, graph hashing for prefixes). This complexity is the price paid for the performance gains, making Nexus harder to adopt than off-the-shelf containers, though the paper argues the economic benefits (GPU cost savings) justify the effort.
*   **Baseline Configuration:** While the authors took care to implement a "batch-oblivious" scheduler for fairness, one could argue that a highly tuned, manual configuration of Clipper/TF Serving might narrow the gap. However, the paper's point is precisely that *manual* tuning is infeasible at scale, and Nexus automates what humans cannot easily compute (the NP-hard squishy bin packing).

In conclusion, the experiments provide compelling evidence that Nexus's novel approach to batching, scheduling, and query optimization solves the utilization-latency trade-off that plagues current DNN serving systems. The results are consistent, quantitative, and clearly attributed to the specific innovations proposed.

## 6. Limitations and Trade-offs

While Nexus demonstrates significant performance gains, its design involves specific trade-offs and relies on assumptions that constrain its applicability in certain scenarios. Understanding these limitations is crucial for determining where Nexus is the appropriate solution versus where simpler systems might suffice.

### 6.1 The Latency of Centralized Control (Epoch Granularity)
The most fundamental architectural trade-off in Nexus is the decoupling of the **Control Plane** (global scheduler) from the **Data Plane** (runtime execution). To solve the NP-hard "squishy bin packing" problem, the Global Scheduler operates in discrete time intervals called **epochs**, typically lasting **30 to 60 seconds**, with a hard minimum of **10 seconds** to prevent oscillation (**Section 5**).

*   **The Constraint:** This design means Nexus cannot react instantaneously to sub-second traffic spikes. If a sudden burst of requests arrives mid-epoch, the scheduler cannot immediately re-allocate GPUs or adjust batch sizes globally.
*   **The Mitigation Strategy:** To bridge this gap, Nexus relies heavily on its **Early Drop** policy at the data plane level. As shown in **Figure 13**, during a workload spike at $t=326s$, the system detects the change and allocates more GPUs within **12 seconds**. However, during this reaction window, the "bad rate" (requests missing SLOs or being dropped) spikes. The paper reports an average bad rate of **0.27%**, but acknowledges that sporadic violations occur specifically during these reconfiguration windows.
*   **Implication:** Nexus is optimized for workloads with statistical stability over tens of seconds (e.g., video streams, analytics dashboards). It is less suitable for applications requiring millisecond-level elasticity or those with extremely erratic, flash-crowd traffic patterns where even a 10-second reaction time results in unacceptable service degradation.

### 6.2 Dependency on Accurate Profiling and Stationarity
Nexus's scheduling algorithms are **profile-guided**. The "squishy bin packing" logic and query optimization depend entirely on the accuracy of the **batching profiles** generated during the management phase (**Section 5**). These profiles map batch sizes to execution latencies ($\ell(b)$).

*   **The Assumption:** The system assumes that the execution characteristics of a model are relatively **stationary** and predictable. It assumes that a profile generated on a specific GPU type (e.g., NVIDIA K80 or V100) remains valid throughout the epoch.
*   **The Weakness:** The paper does not explicitly address how Nexus handles **noisy neighbors** at the hardware level (e.g., thermal throttling, PCIe contention from other non-Nexus processes) or **dynamic voltage/frequency scaling (DVFS)** that might alter execution times unpredictably. If the actual latency $\ell(b)$ deviates significantly from the profiled value due to hardware variability, the calculated duty cycles could become invalid, leading to SLO violations.
*   **Cold Start Overhead:** The reliance on profiling implies a **cold start cost**. When a new, unseen model is ingested, the system must pause to run benchmarks across various batch sizes to generate a profile before it can be optimally scheduled. For environments with thousands of ephemeral models that are used only once, this profiling overhead could negate the efficiency gains.

### 6.3 Structural Constraints on Model Specialization
The **Prefix Batching** innovation (**Section 6.3**) is powerful but structurally constrained. It relies on the ability to identify identical subgraphs (prefixes) across different models.

*   **The Limitation:** This technique is highly effective for **transfer learning** scenarios where models share early layers (feature extractors) and differ only in the final classification layers (suffixes). However, it offers diminishing returns if:
    1.  **Divergence is Early:** If specialized models differ in their early layers (e.g., different input resolutions requiring different convolutional strides, or entirely different architectures like mixing CNNs with Transformers), the common prefix is small or non-existent.
    2.  **Complex DAGs:** While Nexus handles tree-structured queries well, the complexity of identifying and merging prefixes in highly complex, cyclic, or deeply nested DAGs with multiple merge/split points is not fully explored. The current implementation hashes sub-trees; if the graph structure varies slightly (even if semantically similar), the hash mismatch prevents batching.
*   **Memory vs. Compute Trade-off:** As shown in **Figure 15(b)**, while prefix batching saves memory for models with small suffixes ("1 FC"), the memory savings decrease as the unique suffix grows ("2 FC", "3 FC"). If the specialized part of the model is large, the system still faces memory pressure, limiting the number of co-located models.

### 6.4 The "Drop" Philosophy and Application Suitability
Nexus explicitly adopts a service model where **dropping requests** is an acceptable mechanism to maintain overall throughput and SLO compliance for the remaining 99% of traffic (**Section 5**).

*   **The Assumption:** This philosophy is well-suited for **video analysis**, where dropping a single frame is often inconsequential because the next frame arrives shortly after (temporal redundancy).
*   **The Limitation:** This approach is **not universally applicable**. For workloads where every request is critical and non-redundant (e.g., financial fraud detection, medical diagnosis from a single scan, or security alert verification where a missed event is catastrophic), a system that proactively drops requests via the "Early Drop" policy is unacceptable.
*   **Alternative Not Supported:** The paper briefly mentions that the system *could* be configured to delay requests rather than drop them, but notes that "many of our techniques would still be effective." However, the evaluation focuses exclusively on the drop model. The performance characteristics of Nexus under a strict **no-drop, high-latency-tail** constraint are not quantified. In such a scenario, the "squishy bin packing" might fail to find a feasible solution, potentially leading to cluster-wide gridlock rather than graceful degradation.

### 6.5 Implementation Complexity and Ecosystem Integration
Finally, there is a significant engineering trade-off between performance and complexity.

*   **Implementation Burden:** Nexus is a substantial system, comprising roughly **10,000 lines of C++** and requiring deep integration with GPU drivers, container runtimes (Docker/Kubernetes), and DNN frameworks (Caffe, TensorFlow, Darknet). It requires custom modifications to handle prefix batching and GPU multiplexing, which are not native to standard frameworks.
*   **Adoption Barrier:** Unlike Clipper or TensorFlow Serving, which can be deployed as off-the-shelf containers, Nexus requires a dedicated deployment effort and ongoing maintenance of the custom runtime.
*   **Framework Dependency:** The ability to perform prefix batching depends on the DNN framework's ability to expose the computational graph and allow partial execution. While the paper supports Caffe, Caffe2, TF, and Darknet, the rapid evolution of the deep learning ecosystem (e.g., the rise of PyTorch's dynamic graphs or new hardware accelerators like TPUs/NPUs) poses a maintenance challenge. Adapting Nexus's graph hashing and prefix execution logic to every new framework version or architecture requires continuous engineering effort.

### Summary of Trade-offs

| Feature | Benefit | Trade-off / Limitation |
| :--- | :--- | :--- |
| **Epoch-Based Scheduling** | Solves global NP-hard optimization; stable placement. | **Slow reaction time** (10s+); relies on "Early Drop" to handle bursts. |
| **Profile-Guided Batching** | Maximizes efficiency via precise latency modeling. | **Cold start overhead**; assumes **stationary** hardware performance; sensitive to noise. |
| **Prefix Batching** | Enables batching of specialized models; saves memory. | Only works for models with **shared prefixes**; ineffective for early-divergence architectures. |
| **Early Drop Policy** | Preserves high throughput and SLOs for 99% of traffic. | **Unsuitable for critical/non-redundant workloads** where every request must be served. |
| **Custom Runtime** | Fine-grained GPU control; CPU-GPU overlapping. | **High implementation complexity**; tight coupling to specific DNN frameworks. |

In conclusion, Nexus is a highly specialized engine optimized for **high-throughput, latency-constrained, redundant video workloads** running on stable clusters. Its limitations in reaction time, drop tolerance, and implementation complexity make it less suitable for general-purpose, low-latency, or critical-request serving scenarios where simplicity and absolute reliability take precedence over raw throughput efficiency.

## 7. Implications and Future Directions

Nexus fundamentally alters the landscape of deep learning serving by shifting the unit of optimization from the **individual model** to the **cluster-wide fragment**. Prior to this work, the prevailing wisdom dictated that efficient serving required homogenizing workloads—forcing applications to use identical models to enable batching or accepting low utilization for specialized tasks. Nexus dismantles this constraint, proving that heterogeneity (via transfer learning) and high utilization are not mutually exclusive if the system possesses fine-grained visibility into the computational graph. This paradigm shift has profound implications for cloud economics, system architecture, and future research trajectories.

### 7.1 Reshaping the Landscape: From Model Silos to Compute Fluidity

The most significant impact of Nexus is the decoupling of **model identity** from **batching eligibility**.
*   **Economic Viability of Specialization:** Before Nexus, the cost of model specialization (retraining only the output layers for specific tenants) was high because it destroyed batching efficiency, forcing providers to either provision dedicated GPUs (wasteful) or deny specialization (reducing accuracy). Nexus makes specialization economically viable by recovering batching gains through **prefix execution**. This enables a new class of "hyper-personalized" video analytics services where thousands of tenants can run unique models on shared infrastructure without prohibitive costs.
*   **The End of "Black Box" Serving:** Traditional serving systems (e.g., TensorFlow Serving) treat the DNN as an opaque function $f(x)$. Nexus demonstrates that treating the DNN as a transparent, decomposable graph of linear algebra operations allows for orders-of-magnitude improvements in scheduling. This suggests that future serving stacks must be **graph-aware**, exposing internal layer boundaries to the scheduler rather than hiding them behind API wrappers.
*   **Redefining the SLO-Throughput Frontier:** By introducing **squishy bin packing** and **early drop**, Nexus proves that strict latency SLOs do not require sacrificing throughput. Instead, it shows that throughput collapses often stem from rigid scheduling policies (like lazy dropping) that force inefficient small batches. The field must now view latency compliance and throughput maximization as a joint optimization problem solvable via dynamic duty cycles, rather than a trade-off managed by over-provisioning.

### 7.2 Catalyst for Follow-Up Research

Nexus opens several fertile avenues for future investigation, particularly in extending its core innovations to broader contexts:

*   **Dynamic Graph Merging for Heterogeneous Architectures:** Nexus currently relies on hash-matching to find identical subgraphs, which works well for transfer learning variants of the same architecture (e.g., ResNet-50 variants). Future research could explore **semantic graph merging**, where the system identifies *functionally equivalent* but structurally different layers (e.g., merging a $3\times3$ convolution with two $1\times1$ convolutions) or merges across different architectures (e.g., EfficientNet and ResNet) using approximation techniques. This would extend prefix batching beyond strict structural identity.
*   **Reinforcement Learning for Squishy Packing:** The current scheduler uses a greedy heuristic to solve the NP-hard packing problem. Given the stochastic nature of video workloads, **Deep Reinforcement Learning (DRL)** agents could be trained to predict optimal duty cycles and batch sizes in real-time, potentially outperforming static heuristics in highly volatile environments where request rates fluctuate faster than the 30-second epoch.
*   **Cross-Hardware Prefix Batching:** Nexus is designed for homogeneous GPU clusters. A natural extension is **heterogeneous prefix batching**, where the heavy shared prefix is executed on high-throughput GPUs (e.g., V100/A100), and the lightweight, diverse suffixes are offloaded to lower-cost accelerators (e.g., TPUs, FPGAs, or even CPUs). This would require a scheduler capable of reasoning about data movement costs between heterogeneous devices.
*   **Training-Serving Co-Design:** Nexus optimizes inference. Future systems could apply similar "fragment-based" thinking to **distributed training**. If multiple tenants are fine-tuning similar models, could their gradient computations for shared prefixes be batched or synchronized jointly to accelerate the training loop? This would blur the line between training and serving clusters.

### 7.3 Practical Applications and Downstream Use Cases

The techniques pioneered by Nexus are immediately applicable to any domain requiring high-volume, low-latency inference on diverse data streams:

*   **Smart City Infrastructure:** Municipalities deploying thousands of cameras for traffic monitoring, pedestrian safety, and crime detection often need specialized models for local conditions (e.g., specific vehicle types, regional license plates). Nexus allows a single city-wide cluster to host hundreds of these specialized detectors efficiently, rather than requiring a dedicated server per intersection.
*   **Personalized Retail Analytics:** Retail chains analyzing shopper behavior can deploy unique models for each store layout or product category. Nexus enables a central cloud service to process video feeds from thousands of stores simultaneously, running store-specific object detectors (e.g., "Store A sells shoes, Store B sells groceries") on shared hardware with per-store latency guarantees.
*   **Live Sports and Esports Broadcasting:** As demonstrated in the "Game Analysis" case study, broadcasters can run real-time analytics (player tracking, score recognition, ad verification) on thousands of concurrent streams. The ability to handle bursty traffic (e.g., during a goal or playoff moment) via **early drop** ensures the system remains stable without crashing under load.
*   **Industrial IoT and Quality Control:** Manufacturing lines often require highly specialized defect detection models for different product lines. Nexus allows a factory to run a unified analytics cluster that switches between or co-runs models for different assembly lines, adapting batch sizes dynamically based on production speed.

### 7.4 Reproducibility and Integration Guidance

For practitioners and researchers looking to adopt or build upon Nexus, the following guidance clarifies when and how to apply these techniques:

#### When to Prefer Nexus (or Nexus-inspired Architectures)
*   **High Tenant Diversity:** If your workload involves **many specialized models** (variants of a base model) rather than a single monolithic model, Nexus's **prefix batching** is essential. Standard servers will fail to batch these efficiently.
*   **Strict Latency + High Throughput:** If you must meet tight SLOs (e.g., &lt;100ms) while maximizing GPU utilization, the **squishy bin packing** and **duty cycle** management are superior to standard queue-based schedulers.
*   **Video/Stream Workloads:** The **early drop** policy is specifically designed for redundant data streams (video frames) where dropping a few inputs is acceptable to save the batch. **Do not use this approach** for non-redundant, critical transactions (e.g., financial fraud detection) where every request must be processed; in those cases, the "drop" philosophy is inappropriate.

#### Integration Challenges and Strategies
*   **Framework Dependency:** Nexus requires deep integration with the DNN framework to expose the computational graph and enable partial execution (prefix/suffix splitting).
    *   *Strategy:* If using modern frameworks like PyTorch or TensorFlow 2.x, researchers should leverage **FX graph capture** (PyTorch) or **SavedModel graph analysis** (TF) to replicate the subgraph hashing logic. The core idea is to intercept the execution graph, identify common prefixes via hashing, and rewrite the execution plan to merge batches before the divergence point.
*   **Profiling Overhead:** The system relies on accurate batching profiles ($\ell(b)$).
    *   *Strategy:* Implement an automated profiling pipeline that runs upon model ingestion. For dynamic environments, consider **online profiling** where the system periodically re-measures latency for active batch sizes to adjust for hardware drift or thermal throttling.
*   **Scheduler Complexity:** Implementing the full integer programming solver is likely overkill for smaller clusters.
    *   *Strategy:* Start with the **greedy best-fit decreasing** algorithm described in **Algorithm 1**. It captures 90% of the benefit with a fraction of the complexity. Focus first on implementing the **duty cycle** concept: force the GPU to cycle through models in a fixed time window, adjusting batch sizes to fit, rather than letting models run to completion arbitrarily.

#### Reproducibility Note
The paper provides a fully implemented system in ~10k lines of C++. However, reproducing the exact results requires careful attention to:
1.  **Hardware Specificity:** The batching profiles are hardware-dependent. Results on V100s will not directly translate to A100s or K80s without re-profiling.
2.  **Workload Traces:** The evaluation uses specific video traces (Twitch, traffic cameras). Using synthetic Poisson arrivals may underestimate the benefits of **early drop**, which shines under the bursty, correlated arrivals typical of real video streams.
3.  **Baseline Configuration:** As noted in **Section 7.2**, fair comparison requires disabling the baseline's ability to assume fixed costs. Researchers must ensure their baselines are configured with the same "batch-oblivious" constraints to avoid apples-to-oranges comparisons.

In summary, Nexus provides a blueprint for the next generation of AI infrastructure: systems that are **aware** of model internals, **adaptive** to workload dynamics, and **aggressive** in optimizing global efficiency over local simplicity. Its legacy will likely be the standardization of "fragment-based" scheduling in future cloud AI platforms.