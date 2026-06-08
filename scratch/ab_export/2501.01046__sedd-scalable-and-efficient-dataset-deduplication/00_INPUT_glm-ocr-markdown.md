# GLM-OCR markdown input — 2501.01046

# SEDD: Scalable and Efficient Dataset Deduplication with GPUs

_This is the GLM-OCR output (the LLM input). 61,858 chars._

---

SEDD: Scalable and Efficient Dataset Deduplication with GPUs

Youngjun Son
Graduate School of Data Science
Seoul National University
Seoul, Korea
jun041577@snu.ac.kr

Chaewon Kim
Department of Computer Science
Seoul National University
Seoul, Korea
chaewon@aces.snu.ac.kr

Jaejin Lee
Graduate School of Data Science
Seoul National University
Seoul, Korea
jaejin@snu.ac.kr

Abstract

Dataset deduplication is widely recognized as a crucial preprocessing step that enhances data quality and improves the performance of large language models. A commonly used method for this process is the MinHash Locality-Sensitive Hashing (LSH) algorithm. Recently, GPU-accelerated frameworks such as NVIDIA NeMo Curator have been introduced to handle large-scale corpora; however, they remain suboptimal due to high communication overhead from physical data shuffling and underutilization of GPU resources. In this paper, we propose SEDD, a high-performance GPU-accelerated deduplication framework optimized for distributed cluster environments. SEDD introduces a computationally efficient, partially reusable hash function, alongside highly optimized GPU kernels and a hardware-aware automatic parameter selection mechanism. By replacing traditional data shuffling with a streaming-based approach, SEDD significantly mitigates communication bottlenecks.

Our framework outperforms the CPU-based deduplication tool in SlimPajama by up to 158× and the GPU-based tool in NVIDIA NeMo Curator by up to 7.8× when processing 30 million documents on a node with four GPUs. Notably, SEDD dramatically accelerates the previously time-consuming MinHash signature generation phase, achieving speedups of up to 375× over the CPU baseline. Despite these gains in efficiency, SEDD maintains high deduplication fidelity, with duplicate document sets achieving Jaccard similarities of over 0.95 compared to those identified by the standard MinHash algorithm. In large-scale experiments, the deduplication of 1.2 trillion tokens is completed in just 3 hours on an 8-node 32-GPU V100 cluster. The related code is publicly available on GitHub (https://github.com/mcrl/SEDD).

Keywords

Data Deduplication, Data Preprocessing, Parallel Computing

1 Introduction

Pretrained language models (PLMs) have demonstrated remarkable performance across a wide range of applications [5, 12, 29, 38–40], with their capabilities continuing to improve as training datasets scale [17, 27]. However, web-scale corpora used for modern LLM training often contain a substantial amount of duplicate and near-duplicate content [14, 24]. Such redundancy not only wastes significant computational and storage resources during training but also introduces skewed data distributions due to over-represented documents [23, 36]. Consequently, large-scale dataset deduplication has become a critical preprocessing step in modern LLM data pipelines. However, scalable near-duplicate detection over trillion-token corpora remains prohibitively expensive with existing approaches.

Duplicate detection methods can be broadly categorized into two approaches: exact matching, which identifies identical strings or hash values, and approximate matching, which detects documents with high content similarity [1]. While exact matching is computationally efficient, it fails to capture semantically similar documents with minor variations. In contrast, approximate matching is significantly more computationally intensive. A widely adopted technique for approximate matching is MinHash LSH [18], which provides a scalable approximation of the Jaccard similarity [4]. Despite its popularity, existing MinHash LSH implementations often struggle to efficiently process datasets at the trillion-token scale due to high computational cost and limited hardware utilization.

Recently, NVIDIA introduced NeMo-Curator [20], a GPU accelerated deduplication pipeline that significantly outperforms traditional CPU-based approaches. However, our analysis shows that its performance remains suboptimal in many practical environments, where excessive communication during data shuffling stalls the pipeline and results in low GPU utilization. These limitations prevent current systems from fully exploiting the massive parallelism offered by modern GPUs.

To address these challenges, we propose SEDD, a high-performance GPU-based deduplication framework that maximizes end-to-end processing throughput. By combining computationally efficient hash functions with highly optimized GPU kernels and communication-aware pipeline design, SEDD removes major bottlenecks in existing frameworks, including excessive data shuffling overhead and poor GPU utilization. As a result, our system can deduplicate 1.2 trillion tokens in just 3 hours using an 8-node cluster equipped with 32 V100 GPUs.

The contributions of this paper are summarized as follows:

- **End-to-end GPU deduplication framework.** We present SEDD, a framework that achieves 7× speedup over NeMo-Curator—the fastest existing GPU-based baseline—without compromising deduplication accuracy. Our design enables efficient processing of trillion-token-scale datasets on standard GPU clusters.

- **Optimized MinHash LSH computation.** We accelerate MinHash generation and comparison by adopting lightweight, reusable hash functions and carefully optimized GPU kernels that improve memory efficiency and parallel utilization.

- **Communication-efficient data pipeline.** We eliminate major data shuffling bottlenecks through a streaming-based execution.

strategy. Combined with double buffering, SEDD reduces total communication overhead to about 20% of that of existing frameworks.

2 Background
This section describes the background and related work to SEDD. We examine the impact of data deduplication on the language models. Second, we describe the MinHash LSH algorithm, a standard approach for approximate duplicate detection. Finally, we introduce existing implementations, focusing on the architectural bottlenecks in distributed CPU and GPU-based frameworks that motivate our proposed design.

2.1 Effects of Data Deduplication
Data deduplication is the process of identifying and removing redundant entries to ensure data uniqueness, thereby improving both training efficiency and data quality for PLMs. Figure 2 illustrates examples of duplicate documents in the RealNews [42] dataset, where document structures are nearly identical. Such redundancy can hinder the effective learning of language models.

Prior studies have shown that duplicated or near-duplicate samples negatively affect model evaluation and generalization. Lee et al. [23] report that deduplication improves validation perplexity, while Allamanis [2] demonstrate that duplicated code samples degrade performance on code understanding tasks. Moreover, when near-duplicate documents appear in both training and test sets, evaluation metrics can be artificially inflated, leading to an overestimation of the model’s capability. Removing these overlaps enables a more faithful assessment and reduces overfitting caused by memorization of repeated sequences, ultimately improving generalization [1, 23, 37].

In addition to improving model quality, deduplication substantially reduces computational overhead during training. By eliminating redundant content, the model processes fewer repeated tokens, resulting in faster training and lower operational costs without sacrificing—and often improving—model performance [36].

2.2 MinHash
MinHash [4] is a technique approximating the Jaccard similarity between two documents. The standard MinHash-based deduplication pipeline consists of four stages: MinHash signature generation, duplicate pair generation, construction of a union graph, and extraction of the final duplicate list.

MinHash generation. As illustrated in Figure 1, each document is first decomposed into a set of shingles (or n-grams). Given $H$ hash functions $f_1, \ldots, f_H$, each shingle is mapped to an integer value, producing $H$ hashed representations. The MinHash signature of a document is then formed by taking the minimum hash value for each function, resulting in a signature vector of length $H$. For $N$ documents, this process produces an $H \times N$ signature matrix.

Duplicate pair generation. Given the MinHash signatures, we estimate the similarity between documents by comparing their signatures element-wise. The similarity is approximated as the fraction of matching entries, which serves as an estimator of the Jaccard similarity. Document pairs whose estimated similarity exceeds a predefined threshold $\theta$ are considered near-duplicates and added to the duplicate set.

Constructing a union graph. From the set of duplicate pairs, we build a union graph that groups related documents. Each document is represented as a node, and an edge connects two nodes if they are identified as duplicates. The connected components of this graph correspond to clusters of near-duplicate documents.

Generating the final list of duplicates. For each connected component, we select a representative document (e.g., the document with the lowest index) and remove the remaining documents from the dataset. This procedure yields the final deduplicated corpus while preserving a single instance from each near-duplicate group.

2.3 MinHash LSH
Computing similarity between all document pairs using MinHash signatures can incur prohibitive computational cost at large scale. To address this issue, MinHash LSH (Locality-Sensitive Hashing) combines MinHash with LSH methods.

The MinHash LSH partitions each signature vector of length $H$ into bands and hashes each band into a bucket, resulting in a more compact set of candidate document pairs. Specifically, the signature vector is divided into $b$ bands, each containing $r$ rows (so that $H = b \times r$). Each band is then hashed into one of $k$ buckets. Two documents are considered candidate pairs if at least one of their corresponding band hashes maps to the same bucket. This process significantly reduces the number of pairwise comparisons to a smaller subset of document pairs. For each candidate pair, the full signature vectors are compared. The pair is classified as a duplicate if their similarity exceeds a predefined threshold. For further details, please refer to the Appendix A.

MinHash LSH has become the de facto standard for document deduplication in the development of large language models [1, 6, 25, 28]. Although MinHash LSH provides a scalable approximation, it still imposes a significant computational and execution-time burden. Surprisingly, the system-level efficiency has been overlooked in prior research. This motivates the design of SEDD, which prioritizes hardware-aware optimizations to address the growing scale of modern datasets.

2.4 Limitations of Existing Implementations
CPU Baseline. SlimPajama [35] employed CPU-based MinHash LSH using the Python library datasketch. We identified implementation issues in the publicly released code and corrected them, and hereafter, we will refer to this corrected version as the CPU baseline throughout the paper. The CPU baseline operates identically to the MinHash LSH process described in Section 2.3, with one key difference. The difference lies in how comparisons are performed between documents grouped into the same bucket. When a bucket contains $D$ documents, the original approach performs pairwise comparisons among all documents, resulting in $O(D^2)$ time complexity. However, in the CPU baseline, comparisons are conducted between the first document’s signature vector entering the bucket and those of subsequently arriving documents, resulting in $O(D)$. Since the CPU baseline relies on the datasketch Python library,

it is inherently inefficient for highly parallel workloads such as deduplicating billions of documents.

Notably, search-oriented libraries such as FAISS [21] and ScaNN [15] are excluded from our comparison, as text deduplication for dataset curation focuses on direct token overlap and symbolic similarity rather than high-dimensional vectorization.

**GPU Baseline.** For our primary baseline, we evaluate NVIDIA NeMo Curator [20], which currently represents the state-of-the-art in high-performance dataset deduplication. This framework provides a GPU-accelerated MinHash LSH pipeline that leverages cuDF [31] and Dask [11]. While NeMo Curator offers a simplified mode that identifies duplicates based solely on bucket mapping, we specifically compare against the version that performs explicit Jaccard similarity computation. This ensures a fair and rigorous evaluation by maintaining high-fidelity standards consistent with CPU baseline and accounting for reductions in false positives.

Despite its status as a high-performance framework, NeMo Curator exhibits several architectural inefficiencies when scaling to large datasets. First, its reliance on Dask mandates a physical data shuffling phase. This process requires repartitioning the entire dataset by bucket IDs and writing intermediate results to disk, incurring substantial I/O latency and communication overhead. Second, Nemo Curator employs a sparse bucketing strategy, using complex hash functions to map documents to a high-cardinality bucket space. Adopting the same strategy as the CPU baseline to minimize computational costs, it performs only 1-to-N comparisons within each bucket by selecting a single anchor document and comparing it against the remaining documents. Such sparse tasks may fail to provide sufficient parallel work to saturate GPU Streaming Multiprocessors, potentially leading to hardware underutilization. Finally, Nemo Curator follows a sequential, decoupled execution model in which each stage operates independently. This architecture requires persisting large-scale intermediate results to disk in Parquet format between stages, which increases I/O overhead. Furthermore, since each processing stage can begin only after the previous data movement completes, the framework fails to overlap communication with computation. This lack of pipeline optimization prevents the system from hiding data transfer latency, leading to a significantly longer total wall-clock time.

3 Design and Implementation of SEDD

We develop SEDD, a deduplication framework optimized for distributed GPU clusters with multiple GPUs per node. SEDD is specifically designed to handle massive datasets exceeding the trillion-token scale. The end-to-end architecture is illustrated in Figure 3.

3.1 Overview of SEDD’s Pipeline

The deduplication process in SEDD is divided into two primary phases: (1) parallel hashing and (2) streaming-based extraction with pairwise comparison.

In the first phase, raw datasets stored in formats such as JSONL are processed in parallel. Each node in the cluster spawns $N_{\text{GPU}}$ processes, where $N_{\text{GPU}}$ corresponds to the number of available GPUs. Each process manages loading document text and indices into CPU memory buffers, which are then transferred to the GPU. On the GPU, multiple CUDA threads concurrently generate MinHash signatures and assign bucket IDs for each band. These results are then returned to the CPU and persisted as intermediate hash files, maintaining a one-to-one mapping to the original input files.

The second phase identifies and verifies duplicate candidates through a streaming-based approach. Unlike the GPU baseline, which relies on a physical shuffle to reorganize data on disk, SEDD uses an on-the-fly streaming extraction strategy. Each process $j$ is assigned a specific subset of bands ($B = N_{\text{band}} / N_{\text{GPU}}$) and scans the intermediate hash result files to identify signatures belonging to the same bucket ID. To manage memory constraints and hide I/O latency, each process extracts and buffers signatures for $C$ buckets.

at a time. This allows pipelined execution, where the system can fetch the next batch of document data while the GPU processes the current batch. During each iteration, the extracted signatures for the $C$ buckets are dispatched to the GPU for similarity verification.

Unlike the baselines that rely on $O(D)$ anchor-based heuristics, SEDD performs an exhaustive all-pairs comparison within each bucket in $O(D^2)$, where $D$ denotes the number of documents in the bucket. Despite the theoretically higher computational complexity, SEDD’s hardware-optimized GPU kernels achieve superior throughput. By performing more comprehensive comparisons, SEDD aims to reduce false negatives. Finally, the similarity results are aggregated to construct a global union graph for the final deduplication decision.

3.2 Hash Functions

MinHash LSH performs hashing twice. The first hashing involves calculating the MinHash values for each document to generate signature matrices. The second hashing assigns bucket IDs to the bands of the signature vector of each document. The CPU and GPU baselines and SEDD differ in implementing these hashing steps.

**Hashing in the CPU baseline.** For MinHash generation, the CPU baseline uses a widely used cryptographic hash function SHA-1 [13], which is relatively slow when processing large data. Documents are mapped to the same bucket only if their signature vectors are identical in a particular band.

**Hashing in the GPU Baseline.** Since the primary focus of MinHash generation is preventing collisions rather than cryptographic security, the GPU baseline uses the non-cryptographic MurmurHash3 algorithm [3] for MinHash generation. It produces a 32-bit or 128-bit hash value, enabling faster computation than SHA-1. In the LSH stage, the values of each band are mapped to buckets using the results of the MD5 hash function [32], which generates a 128-bit fingerprint by encoding a string of any length.

**Hashing in SEDD.** For MinHash generation, SEDD adopts a Rabin-Karp rolling hash[22], a computationally efficient non-crypto graphic hash function that enables partial reuse across adjacent shingles. Let $s = c_1c_2\ldots c_k$ be a $k$-gram shingle that consists of characters. We define $f(s) = \sum_{i=1}^{k} c_i \cdot q^{i-1}$ such that $q$ is a constant larger than the character alphabet size. Then SEDD’s hash function $h(s)$ for MinHash generation is defined as follows:

$$h(s) = f(s) \mod p$$ (1)

where $p$ is a sufficiently large prime number (e.g., $p = 4, 294, 967$ in our experiments), allowing both $p$ and $q$ to be represented using 32-bit integers. Unlike SHA-1 and MurMurHash3 in the CPU and GPU baselines, the proposed hash function in Equation 1 allows reusing the hash value of the previous shingle. For example, let $t = c_2 \ldots c_k c_{k+1}$ be the next $k$-shingle of a shingle $s = c_1 c_2 \ldots c_k$. Then, $h(t)$ is computed as follows:

$$h(t) = f(t) \mod p$$
$$= \left( \frac{f(s) - c_1}{q} + c_{k+1} q^{k-1} \right) \mod p$$

Instead of recomputing $f(t)$ from scratch, only a few arithmetic operations—a single multiplication, two additions, and one division—are required. This rolling formulation substantially reduces the computational cost compared to recomputing independent hashes of the baselines.

The second hashing phase assigns bucket IDs. While the GPU baseline uses high-entropy hashes (e.g., MD5) to map bands into a sparse, high-cardinality space, SEDD deliberately employs a dense bucketing strategy. SEDD sums the $r$ values in each band and uses the remainder of the division by the predefined constant number of buckets, $K$. This approach maps the sparse signature space $\mathbb{N}^r$ into a compact, dense space $\mathbb{N}^K$. The choice of $K$ is a critical trade-off. A smaller $K$ increases the collision probability, leading to larger buckets and more comparisons ($O(N^2/K)$), while a larger $K$ increases the overhead of managing fragmented buffers ($O(NK)$ in our streaming setup). To balance the overhead of stream management against the cost of pairwise comparisons, we model the total processing time $T(K)$ as $T(K) \approx aNK + b \frac{N^2}{K}$, where $a$ and $b$ are system-dependent constants. Differentiating with respect to $K$, we derive the optimal number of buckets $K_{opt} \propto \sqrt{N}$. In practice, we set $K = 4\sqrt{N}$ based on empirical calibration, which consistently produced bucket sizes sufficient to sustain high GPU occupancy during the comparison phase, while avoiding excessive fragmentation overhead in the streaming pipeline.

Further discussions on the suitability of the rolling hash function and the empirical analysis regarding the effect of $K$ can be found in Appendix B and Appendix D, respectively.

### 3.3 GPU-based Pairwise Comparison

Given a bucket containing $D$ documents, let $S \in \mathbb{Z}^{D \times H}$ be the signature matrix where $S_i$ represents the signature vector of the $i$-th document with length $H$. The pairwise deduplication task requires computing a binary similarity matrix $M \in \{0, 1\}^{D \times D}$, defined as:

$$M_{ij} = \mathbb{I} \left( \sum_{k=1}^{H} \mathbb{I} \left( S_{i,k} = S_{j,k} \right) \geq \theta \cdot H \right)$$

where $\mathbb{I}(\cdot)$ is the indicator function and $\theta$ is the similarity threshold. To reduce redundant computation, we evaluate only the upper triangular portion of $M$. Since this ‘count-match’ operation is not supported by standard BLAS(Basic Linear Algebra Subprograms) libraries, we implement a custom CUDA kernel optimized for GPU execution. Although the operation is not a conventional matrix multiplication, it exhibits similar memory access patterns. Thus, we adopt a GEMM-inspired tiled design to improve data locality. Each thread is assigned to a single document pair and accumulates the number of matching hash values across the signature dimension. The kernel iteratively loads chunks of signature values into shared memory and reuses them across multiple comparisons, reducing global memory traffic. After processing all tiles, document pairs whose match count exceeds the threshold are written to the output buffer and returned to the CPU for union-graph construction.

### 3.4 Other Optimizations

This section explains the optimization details when calculating signature matrices and generating duplicate pairs.

**Hardware-Aware Parameter Tuning.** To ensure scalability across diverse hardware configurations, SEDD adopts a memory-aware execution strategy that adapts to the available CPU and GPU resources. Prior to execution, the system automatically determines several key parameters to maximize resource utilization while maintaining stable operation. Specifically, SEDD decides (1) whether intermediate results are retained in memory or offloaded to storage, (2) the number of buckets processed concurrently during the streaming extraction stage, and (3) the maximum bucket size permitted for GPU-based pairwise comparison. The detailed parameter selection procedure is described in Appendix C.

**Communication-computation overlapping.** To maximize performance, we implement communication-computation overlapping techniques, such as double buffering [7]. When the GPU performs computation, a buffer containing a batch of files is simultaneously transferred to the GPU. This overlapping strategy is applied to both the minhash generation stage and the comparison stage.

### 4 Experiment

This section compares SEDD against the existing CPU and GPU implementations of MinHash LSH. We first compare the computation speed and accuracy for deduplication. Then, we provide a detailed analysis of the architectural factors contributing to this efficiency.

### 4.1 Evaluation Environment

**Target system configuration.** We use a 8-node GPU cluster with a storage node for our experiments. Each node is equipped with four NVIDIA Tesla V100 GPUs, with each GPU having 32GB of memory. The detailed target system configuration is in Appendix E.

**Comparison baselines.** To ensure a fair and rigorous evaluation, we align the algorithmic hyperparameters for MinHash LSH across all implementations. We adopt the standard configuration of the CPU baseline, utilizing 128 hash functions partitioned into $b = 16$ bands and $r = 8$ rows. Following Lee et al. [23], we generate shingles using 5-grams and set the Jaccard similarity threshold to 0.8. All implementations are tuned to their best-performing configurations under identical hardware constraints.

**Datasets used.** The datasets we used in our experiments are the RealNews dataset [42] and C4 [30]. The RealNews dataset is a large English corpus of news articles from Common Crawl, and C4 is a filtered version of Common Crawl. We select these datasets for two primary reasons. First, their significant scale—comprising over 30 million documents—is essential for evaluating the computational limits of traditional MinHash LSH and for quantifying

the acceleration provided by SEDD. Second, as established in prior research [23], deduplicating these specific corpora has been shown to improve the downstream performance of language models. Since this effectiveness is well-documented, we focus on the efficiency of the deduplication process itself, while including supplementary experiments to verify the consistency of these effects.

Following SlimPajama [35], we preprocess the datasets before deduplication. We apply NFC normalization to remove non-Unicode characters, ensuring that a letter followed by a combining character becomes a single combined character. We also filter out documents with less than 200 characters.

4.2 Processing Speed

We divide the overall MinHash LSH process into two main phases: the Generation phase, which includes MinHash generation and bucket mapping, and the Comparison phase, which includes pairwise comparisons and merging.

We used the complete 100GB RealNews dataset and the sampled C4 dataset, which consists of 100GB randomly sampled from the full 750GB C4 dataset, for this experiment. The CPU baseline processing time is measured using Python’s multiprocessing module across 64 logical CPU cores, while both the GPU baseline and SEDD are measured using four V100 GPUs in a single node. The results are summarized in Table 1. For deduplication on the RealNews dataset, SEDD is about 158 times faster than the CPU baseline and about 7.8 times faster than the GPU baseline. Notably the MinHash generation phase alone is roughly 375 times faster than the CPU baseline. Similarly, on the C4 dataset, SEDD achieves a speedup of 136 over the CPU baseline and about 7.3 over the GPU baseline. As described in Section 3.2, SEDD uses $K = 4\sqrt{N}$ buckets and processes $C$ buckets concurrently during the streaming stage. Further analysis of these hyperparameters and the ablation results are reported in Appendix D.

4.3 Deduplication Accuracy

Evaluating the accuracy of deduplication on web-scale corpora is inherently challenging due to the absence of labeled ground truth. To address this, we adopt different strategies depending on the size of dataset: we utilize brute-force MinHash as a pseudo-ground truth for small datasets and employ multiple proxy evaluations—including consistency checks with GPU baseline and downstream model performance for large-scale datasets.

Small Dataset. For datasets of manageable size (0.1M and 1M documents sampled from RealNews), we treat the result of a brute-force all-pairs MinHash comparison (without LSH approximation) as the oracle, or Exact MinHash. To ensure the sampled subsets contain a statistically significant number of duplicates for evaluation, we employ stratified sampling rather than random selection, preserving the cluster structures of the original corpus.

We define the set of near-duplicates as the collection of all document pairs identified as similar. We measure the fidelity of SEDD by calculating the Jaccard similarity between the set retrieved by SEDD and the set retrieved by Exact MinHash. As shown in Table 2, SEDD achieves a set similarity score exceeding 0.95 relative to Exact MinHash. Specifically, on the 1M dataset, SEDD identified 283,264 documents as duplicates, of which 281,355 overlapped with the ground truth. This high alignment confirms that SEDD captures nearly all duplicate candidates found by the computationally expensive exact method, effectively minimizing false negatives despite its optimized hashing and bucketing strategy.

Large Datasets. For datasets exceeding 30 million documents, running Exact MinHash becomes computationally impractical. Consequently, we demonstrate the accuracy of SEDD through two indirect evaluation methods. First, we evaluate the consistency of SEDD by measuring its overlap with NeMo Curator, the current state-of-the-art GPU baseline. As detailed in Table 3, SEDD successfully identifies over 97% of the near-duplicates found by NeMo while uncovering additional instances. Second, we utilize

downstream language model performance as a proxy for data quality. We trained LLaMA 3.2-1B[12] on three dataset variations: (1) non-deduplicated raw data, (2) data deduplicated by NeMo, and (3) data deduplicated by SEDD. Detailed configurations are in Appendix F. As summarized in Table 5, the model trained on the SEDD-deduplicated dataset performs as well as the model trained on the non-deduplicated version, despite using 10% fewer tokens. Crucially, its performance is comparable to the NeMo-deduplicated baseline across all downstream tasks. This consistency in model performance serves as strong evidence that SEDD efficiently identifies and removes redundant data without compromising the linguistic integrity or the information density of the training corpus.

4.4 Hash Functions

Computational performance. We compare the computational performance of our reusable hash functions with MurmurHash3 [3], a widely used non-cryptographic hash function adopted in the GPU baseline. To ensure a fair comparison, we implement a CUDA version of MurmurHash3 and measure only the GPU kernel execution time corresponding to the hash computation, excluding other components of the MinHash generation pipeline (e.g., data loading, memory transfers). As reported in Table 4, across datasets of varying sizes, our hash function consistently outperforms MurmurHash3 in terms of kernel execution speed. On the RealNews dataset, the proposed hash function achieves a 4.85× speedup over MurmurHash3.

Imbalance in Buckets. We conducted a thorough analysis of the distribution of documents across buckets in the C4 dataset to assess potential imbalances during our MinHash LSH process. As shown in Figure 4, the distribution of the number of documents per bucket varies slightly for each rank. However, most buckets cluster around the mean value, which is calculated by dividing the total number of documents by the total number of buckets.

Outlier buckets—defined as those whose sizes exceed the mean by more than twice the standard deviation—account for only about 1.2% of all buckets. Additionally, the pairwise comparisons within these outliers represent merely 3.6% of the total per-band comparisons, indicating a negligible impact. The difference between the actual number of pairwise comparisons and the expected number based on a perfectly uniform distribution is approximately 1.97%, which is not statistically significant.

These results indicate that, while minor imbalances are present, their effect on the overall number of comparisons is minimal. Thus, the distribution can be considered effectively uniform in practice. Additionally, the number of pairwise comparisons conducted per rank is nearly constant, further supporting the notion of a balanced workload distribution across ranks. Detailed experimental results can be found in the Appendix G.

4.5 Scalability

By varying the number of GPUs, the size of the dataset, and the underlying hardware architecture, we evaluate the comprehensive scalability and robustness of SEDD.

Number of GPUs. As illustrated in Figure 5, we evaluate the execution time and speedup as the number of GPUs increases from 1 to 32. Configurations with eight or more GPUs correspond to a multi-node setup. For the RealNews dataset, SEDD achieves a 50× speedup over the CPU baseline even with a single GPU, and completes the process in under 90 seconds when using 32 GPUs. Throughout the scaling process, SEDD consistently maintains at least a 7× speedup over the GPU baseline. For the full C4 dataset, execution time decreases nearly linearly as the number of GPUs increases. In particular, deduplication of 365M documents is completed in approximately 25 minutes using 32 GPUs. Interestingly, the speedup over the GPU baseline increases as we transition to a multi-node environment, indicating that SEDD’s streaming extraction and communication–computation overlapping effectively mitigate synchronization and I/O overheads that typically limit distributed LSH pipelines. We also observe that scaling efficiency depends strongly on dataset size. For relatively smaller datasets such as RealNews, the workload per GPU becomes insufficient as more devices are added, and fixed communication and synchronization overheads begin to dominate, leading to performance saturation beyond eight GPUs. In contrast, larger datasets such as C4 provide sufficient computational intensity to fully utilize additional hardware resources, allowing SEDD to sustain strong scaling and higher speedups in large-scale cluster environments.

Robustness across GPU Architectures. To verify the portability of SEDD, we evaluate its performance across different GPU architectures beyond our V100 cluster. We conducted additional tests on a workstation equipped with NVIDIA RTX 3090 GPUs. Despite the architectural differences, SEDD successfully deduplicated the RealNews dataset in 119 seconds using 4 GPUs. This confirms that our hardware-aware parameter selection and optimized CUDA kernels effectively adapt to varying hardware specifications.

4.6 Analysis of Communication Efficiency

To further investigate the architectural advantages of SEDD, we conduct a comparative analysis of communication latency and volume during the deduplication of the RealNews dataset in a 4-GPU environment. Figure 6 illustrates the breakdown of data transfers into three categories: Host-to-Device (H-to-D), Device-to-Host (D-to-H), and Device-to-Device (D-to-D).

As shown in the results, the GPU baseline suffers from substantial communication overhead, with a total latency of 187.7 seconds and a communication volume reaching 3.45 TB. This inefficiency is primarily driven by the physical shuffling phase, which requires extensive D-to-D and D-to-H data movements to reorganize signatures into their respective buckets across distributed GPUs. In contrast, SEDD demonstrates significantly higher efficiency, reducing the communication latency to 35.7 seconds (a 5.2x speedup) and the total volume to 398.9 GB (an 8.6x reduction).

The reduction in communication volume is particularly notable; by utilizing a streaming extraction strategy, SEDD eliminates the need for large-scale D-to-D shuffling. Instead, it only performs the necessary H-to-D and D-to-H transfers for signature processing. Furthermore, SEDD effectively mitigates the remaining communication latency through the communication-computation overlapping techniques described in Section 3.4. By pipelining data transfers with GPU kernel execution, SEDD hides the majority of the data movement overhead, ensuring that communication does not become a bottleneck as the dataset scale increases.

5 Conclusion

We present a framework called SEDD, which performs MinHash LSH-based deduplication efficiently on GPUs. SEDD significantly outperforms the CPU baseline included in SlimPajama by up to 158× and the GPU baseline included in NVIDIA NeMo Curator by up to

7.8× on a single node with four GPUs when processing a dataset with 30M documents. In a multi-node and multi-GPU environment, SEDD completes deduplication of 1.2 trillion tokens in approximately 3 hours. These performance gains are enabled by a series of GPU-oriented system optimizations, including reusable rolling hash functions for efficient MinHash generation and a streaming execution pipeline that overlaps communication and computation. As a result, SEDD achieves both high throughput and high accuracy: documents identified as duplicates exhibit a Jaccard similarity of 0.95 or higher compared to those found by the standard MinHash algorithm. Extensive experimental results demonstrate that SEDD is a practical and scalable solution for deduplication in the large-scale datasets used for modern LLM training.

Acknowledgement

This work was partially supported by the National Research Foundation of Korea (NRF) under Grant No. RS-2023-00222663 (Center for Optimizing Hyperscale AI Models and Platforms), and by the Institute for Information and Communications Technology Promotion (IITP) under Grant No. 2018-0-00581 (CUDA Programming Environment for FPGA Clusters) and No. RS-2025-02304554 (Efficient and Scalable Framework for AI Heterogeneous Cluster Systems), all funded by the Ministry of Science and ICT (MSIT) of Korea. It was also partially supported by the Korea Health Industry Development Institute (KHIDI) under Grant No. RS-2025-25454559 (Frailty Risk Assessment and Intervention Leveraging Multimodal Intelligence for Networked Deployment in Community Care), funded by the Ministry of Health and Welfare (MOHW) of Korea. Additional support was provided by the BK21 Plus Program for Innovative Data Science Talent Education (Department of Data Science, Seoul National University, No. 519999014569) and the BK21 FOUR Program for Intelligent Computing (Department of Computer Science and Engineering, Seoul National University, No. 4199990214639), both funded by the Ministry of Education (MOE) of Korea. This work was also partially supported by the Artificial Intelligence Industrial Convergence Cluster Development Project, funded by the MSIT and Gwangju Metropolitan City. Research facilities were provided by the Institute of Computer Technology (ICT) at Seoul National University.

References

[1] Alon Albalak, Yanai Elazar, Sang Michael Xie, Shayne Longpre, Nathan Lambert, Xinyi Wang, Niklas Muennighoff, Bairu Hou, Liangming Pan, Haewon Jeong, et al. 2024. A survey on data selection for language models. arXiv preprint arXiv:2402.16827 (2024).

[2] Miltiadis Allamanis. 2019. The adverse effects of code duplication in machine learning models of code. In Proceedings of the 2019 ACM SIGPLAN International Symposium on New Ideas, New Paradigms, and Reflections on Programming and Software. 143-153.

[3] Austin Appleby. 2012. MurmurHash3, 2012. URL: https://github.com/aappleby/smhasher/blob/master/src/MurmurHash3.cpp (2012).

[4] Andrei Z Broder. 1997. On the resemblance and containment of documents. In Proceedings. Compression and Complexity of SEQUENCES 1997 (Cat. No. 977B100171). IEEE, 21–29.

[5] Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. 2020. Language models are few-shot learners. Advances in neural information processing systems 33 (2020), 1877–1901.

[6] Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. 2020. Language models are few-shot learners. Advances in neural information processing systems 33 (2020), 1877–1901.

[7] John Cheng, Max Grossman, and Ty McKercher. 2014. Professional CUDA c programming. John Wiley and Sons.

[8] Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot, Ashish Sabharwal, Carissa Schoenick, and Oyvind Tafford. 2018. Think you have solved question answering? try arc, the ai2 reasoning challenge. arXiv preprint arXiv:1803.05457 (2018).

[9] Yann Collet. 2012. xxHash. GitHub repository. https://github.com/Cyan4973/xxHash.

[10] Together Computer. 2023. RedPajama: an Open Dataset for Training Large Language Models. https://github.com/togethomputer/RedPajama-Data

[11] Dask. 2024. Dask: a Python library for parallel and distributed computing. https://github.com/dask/dask

[12] Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiaesha Letman, Akhil Mathur, Alan Schelten, Amy Yang, Angela Fan, et al. 2024. The laima 3 herd of models. arXiv preprint arXiv:2407.12783 (2024).

[13] D. Eastlake 3rd and P. Jones. 2001. US Secure Hash Algorithm 1 (SHA1). RFC 3174 (Informational). http://www.ietf.org/rfc/rfc3174.txt Updated by RFC 4634.

[14] Yanai Elazar, Akshita Bhagia, Ian Magnusson, Abilasha Ravichander, Dustin Schwenk, Alane Suhr, Pete Walsh, Dirk Groeneveld, Luca Soldain, Sameer Singh, Hanna Haijishirzi, Noah A. Smith, and Jesse Dodge. 2024. What’s In My Big Data? arXiv:2310.20707 [cs.CL] https://arxiv.org/abs/2310.20707

[15] Ruiqi Guo, Philip Sun, Erik Lindgren, Quan Geng, David Simcha, Felix Chern, and Sanjiv Kumar. 2020. Accelerating Large-Scale Inference with Anisotropic Vector Quantization. In International Conference on Machine Learning. https://arxiv.org/abs/1908.10396

[16] Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. 2020. Measuring massive multitask language understanding. arXiv preprint arXiv:2009.03300 (2020).

[17] Jordan Hoffmann, Sebastian Borgeaud, Arthur Mensch, Elena Buchatskaya, Trevor Cai, Eliza Rutherford, Diego de Las Casas, Lisa Anne Hendricks, Johannes Welbl, Aidan Clark, et al. 2022. Training compute-optimal large language models. arXiv preprint arXiv:2203.15556 (2022).

[18] Piotr Indyk and Rajeev Motwani. 1998. Approximate nearest neighbors: towards removing the curse of dimensionality. In Proceedings of the thirtieth annual ACM symposium on Theory of computing. 604–613.

[19] Paul Jaccard. 1912. The distribution of the flora in the alpine zone. 1. New phytologist 11, 2 (1912), 37–50.

[20] Joseph Jennings, Mostofa Patwary, Sandeep Subramanian, Shrimai Prabhumoye, Ayush Dattagupta, Vibhu Jawa, Jiwei Liu, Ryan Wolf, Sarah Yurick, and Varun Singh. 2024. NeMo-Curator: a toolkit for data curation. https://github.com/NVIDIA/NeMo-Curator

[21] Jeff Johnson, Matthijs Douze, and Hervé Jégou. 2019. Billion-scale similarity search with GPUs. IEEE Transactions on Big Data 7, 3 (2019), 535–547.

[22] Richard M Karp and Michael O Rabin. 1987. Efficient randomized pattern-matching algorithms. IBM journal of research and development 31, 2 (1987), 249–260.

[23] Katherine Lee, Daphne Ippolito, Andrew Nystrom, Chiyuan Zhang, Douglas Eck, Chris Callison-Burch, and Nicholas Carli. 2022. Deduplicating Training Data Makes Language Models Better. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), Smaranda Muresan, Preslav Nakov, and Aline Villavicencio (Eds.). Association for Computational Linguistics, Dublin, Ireland, 8424–8445. doi:10.18653/v1/2022.acl-long577

[24] Ian Magnusson, Akshita Bhagia, Valentin Hofmann, Luca Soldain, Ananya Harsh Jha, Oyvind Tafford, Dustin Schwenk, Evan Pete Walsh, Yanai Elazar, Kyle Lo, et al. 2023. Paloma: A benchmark for evaluating language model fit. arXiv preprint arXiv:2312.10523 (2023).

[25] Rabeeh Karimi Mahabadi, Sanjeev Satheesh, Shrimai Prabhumoye, Mostofa Patwary, Mohammad Shoeybi, and Bryan Catanzaro. 2025. Nemotron-cc-math: A 133 billion-token-scale high quality math pretraining dataset. arXiv preprint arXiv:2508.15096 (2025).

[26] Geoff Pike. 2014. Introducing FarmHash. Google Open Source Blog. https://opensource.googleblog.com/2014/03/introducing-farmhash.html.

[27] Jack W Rae, Sebastian Borgeaud, Trevor Cai, Katie Millican, Jordan Hoffmann, Francis Song, John Aslanides, Sarah Henderson, Roman Ring, Susannah Young, et al. 2021. Scaling language models: Methods, analysis & insights from training gopher. arXiv preprint arXiv:2112.11446 (2021).

[28] Jack W Rae, Sebastian Borgeaud, Trevor Cai, Katie Millican, Jordan Hoffmann, Francis Song, John Aslanides, Sarah Henderson, Roman Ring, Susannah Young, et al. 2021. Scaling language models: Methods, analysis & insights from training gopher. arXiv preprint arXiv:2112.11446 (2021).

[29] Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J Liu. 2020. Exploring the limits of transfer learning with a unified text-to-text transformer. Journal of machine learning research 21, 140 (2020), 1–67.

[30] Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, Peter J Liu, et al. 2020. Exploring the limits of transfer learning with a unified text-to-text transformer. J. Mach. Learn. Res. 21, 140 (2020), 1–67. https://arxiv.org/abs/1910.10683

[31] RAPIDS. 2024. cuDF - GPU DataFrames. https://github.com/rapidsai/cudf

[32] Ronald Rivest. 1992. RFC1321: The MD5 message-digest algorithm.

[33] Keisuke Sakaguchi, Ronan Le Bras, Chandra Bhagavatula, and Yejin Choi. 2021. Winogrande: An adversarial winograd schema challenge at scale. Commun. ACM 64, 9 (2021), 99–106.

[34] J.T. Schwartz. 1980. Fast Probabilistic Algorithms for Verification of Polynomial Identities. J. ACM 27, 4 (Oct. 1980), 701–717. doi:10.1145/322217.322225

[35] Zhiqiang Shen, Tianhua Tao, Liqun Ma, Willie Neiswanger, Joel Hestness, Natalia Vassilieva, Daria Soboleva, and Eric Xing. 2023. Slimpajama-dc: Understanding data combinations for lm training. arXiv preprint arXiv:2309.10818 (2023).

[36] Ben Sorscher, Robert Geirhos, Shashank Shekhar, Surya Ganguli, and Ari Morcos. 2022. Beyond neural scaling laws: beating power law scaling via data pruning. Advances in Neural Information Processing Systems 35 (2022), 19523–19536.

[37] Kushal Tirumala, Daniel Simig, Armen Aghajanyan, and Ari Morcos. 2023. D4: Improving lm pretraining via document de-duplication and diversification. Advances in Neural Information Processing Systems 36 (2023), 53983–53995.

[38] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, Aurelien Rodriguez, Armand Joulin, Edouard Grave, and Guillaume Lample. 2023. LLaMA: Open and Efficient Foundation Language Models. arXiv:2302.13971 (cs.CL).

[39] Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babai, Nikolay Bashlykov, Soumya Batra, Prajwal Bhargava, Shruti Bhosale, et al. 2023. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288 (2023).

[40] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Lukasz Kaiser, and Ilia Polosukhin. 2017. Attention is all you need. Advances in neural information processing systems 30 (2017). https://arxiv.org/abs/1706.03762

[41] Rowan Zellers, Ari Holtzman, Yonatan Bisk, Ali Farhadi, and Yejin Choi. 2019. Hellaswag: Can a machine really finish your sentence? arXiv preprint arXiv:1905.07830 (2019).

[42] Rowan Zellers, Ari Holtzman, Hannah Rashkin, Yonatan Bisk, Ali Farhadi, Franziska Roesner, and Yejin Choi. 2019. Defending against neural fake news. Advances in neural information processing systems 32 (2019).

Appendix

A MinHash

A.1 Jaccard Similarity

Jaccard similarity [19], $J(A, B)$, is a metric to quantify the similarity of two finite sets, $A$ and $B$:

$$J(A, B) = \frac{|A \cap B|}{|A \cup B|}$$

(4)

When used as a metric for finding document similarity, it is defined as the number of common words between the documents divided by the total number of words in them. While this is the most intuitive way to measure similarity, calculating the Jaccard similarity directly between documents in a large corpus is not feasible due to the computational cost. Thus, the MinHash algorithm [4], which approximates Jaccard similarity, is often used instead.

A.2 MinHash LSH

The critical difference between MinHash and MinHash LSH lies in the stage of generating duplicate pairs. In MinHash LSH, the signature column vector in the signature matrix of each document is divided into $b$ bands, each of which has $r$ integers. As shown in Figure 7, LSH hashes the bands in each document to $k$ buckets. Let $S_i$ be the signature vector of the document $D_i$. For a given signature vector $S_1$ and $S_2$, when at least a pair of bands $B_{S_1}$ from $S_1$ and $B_{S_2}$ from $S_2$ hashed to the same bucket, we tag them as candidate pairs, potentially similar documents. For example, in Figure 7, $D_2$ and $D_N$ are potentially similar documents because they have bands hashed to the same bucket. Then, LSH performs pairwise comparisons on each document’s signature vectors of length $b \times r$ for the candidate pairs. If the similarity between the two vectors exceeds a predefined threshold, the two documents are considered a duplicate pair. Once the union graph is constructed for each band, the union graphs for all bands are merged to obtain the final union graph.

B Suitability of SEDD’s Hash Functions

In general, a good hash function for MinHash should have the following properties:

- Determinacy: The same input produces the same output.
- Uniformity: The hash function should distribute outputs uniformly across the range.
- Collision resistance: Two distinct inputs should unlikely produce the same hash value.

The proposed hash function satisfies these properties. Determinacy and uniformity are straightforward. Collision resistance is also met with a large value of $p$. When uniformity is satisfied, the probability of a collision approaches $\frac{1}{p}$, which is very small. This follows the same intuition as the Schwartz–Zippel lemma [34]. This ensures collisions are rare when $p$ is sufficiently large.

In many contexts, hash functions are required to resist preimage attacks, making it computationally difficult to reverse-engineer the input from the hash value. However, it is not a concern in MinHash because the hash function is not used for security purposes. Moreover, while other non-cryptographic hash functions such as FarmHash [26] and xxHash [9] exist, our design provides the rolling property, which allows efficient reuse of previously computed results when sliding over substrings. This property yields better throughput for string hashing compared to FarmHash or xxHash.

C Hardware-aware Parameter Selection

SEDD employs an automated, hardware-aware mechanism to determine execution parameters, aiming to maximize throughput while maintaining stable operation across diverse hardware environments.

Adaptive Storage Strategy. The system first determines whether intermediate results—specifically the signature matrix and bucket IDs—should be kept in main memory or offloaded to secondary storage. Let $N$ denote the total number of documents, $H$ the signature length, and $n_{\text{proc}}$ the number of CPU processes per node. The required memory for intermediate buffers is estimated as

$$M_{\text{req}} = N \times (H + 1) \times \text{sizeof(int)} \times n_{\text{proc}}.$$

This estimate is compared against the available host memory $M_{\text{avail\_cpu}}$. If

$$M_{\text{req}} \leq \alpha \cdot M_{\text{avail\_cpu}},$$

the system operates fully in memory to eliminate disk I/O overhead. Otherwise, SEDD automatically activates a disk-based offloading strategy and persists intermediate hash results in Parquet format to prevent memory exhaustion when scaling to trillion-token datasets. We use a conservative safety margin of $\alpha = 0.2$ to account for transient memory usage from the operating system and runtime buffers.

Dynamic Pipelining with Parameter $C$. During the streaming extraction phase, each process handles $C$ bucket IDs concurrently to reduce the number of full-dataset scans. The value of $C$ is dynamically tuned to utilize available CPU memory without triggering swap overhead. Given $n_{\text{proc}}$ concurrent processes per node and $K$ total buckets, SEDD determines $C$ using the constraint

$$C \cdot n_{\text{proc}} \cdot \left( \frac{N}{K} \right) \cdot H \cdot \text{sizeof(int)} \leq \alpha \cdot M_{\text{avail\_cpu}},$$

where $M_{\text{avail\_cpu}}$ denotes the currently available memory and $\alpha$ is the safety margin factor. Maximizing $C$ within this bound reduces redundant file accesses and helps hide I/O latency, as more buckets can be verified in a single pass over the stored hash results.

GPU Memory Constraint Management. Finally, the system enforces a hard limit on the maximum bucket size $D_{\text{max}}$ to ensure compatibility with the available GPU memory $M_{\text{gpu}}$. Since the comparison kernel performs exhaustive $O(D^2)$ pairwise checks, the required memory for processing a bucket depends on both the number of documents $D$ and the signature length $H$. SEDD ensures that

$$D \times H \times \text{sizeof(int)} < \beta \cdot M_{\text{gpu}}.$$

where $\beta$ is the safety margin factor. If a bucket exceeds $D_{\text{max}}$, it is partitioned into smaller sub-units to maintain memory safety and prevent Out-of-Memory (OOM) errors during kernel execution.

D Ablation on System-Level Design Choices

To better understand the contribution of each component in SEDD, we conduct a series of ablation studies on key system-level design choices. In particular, we analyze how different architectural optimizations and parameter settings affect the overall performance of MinHash LSH-based deduplication.

Ablation study. We provide a quantitative breakdown of the performance gains contributed by each architectural component of SEDD. Table 6 summarizes the execution time for deduplicating the RealNews dataset using 4 GPUs under different system configurations.

The results show that the fully optimized version of SEDD achieves the best performance, completing the pipeline in 135.2 seconds. Notably, even when using the standard MurmurHash3, our framework executes in 147.7 seconds, which is already substantially faster than traditional baselines due to our hand-tuned GPU kernel implementation. Replacing MurmurHash3 with our proposed rolling hash further accelerates the hash generation stage, reducing the overall execution time by an additional 12.5 seconds.

Furthermore, disabling the double buffering mechanism significantly degrades performance, increasing the execution time to 189.0 seconds. This result highlights the importance of overlapping storage I/O with GPU computation in achieving high throughput.

The effect of $C$. $C$ denotes the number of buckets processed by each process in a single step, meaning that a larger $C$ allows more buckets to be handled per file I/O operation. While we set $C$ to a practical value considering CPU memory constraints, we further analyze its impact on deduplication performance.

Table 7 reports the file I/O time measured during the comparison stage on the RealNews dataset while varying $C$. The results show that increasing $C$ consistently reduces the file I/O overhead, demonstrating that $C$ plays a crucial role in improving throughput. Therefore, in practice, we set $C$ to the largest value permitted by available system resources to minimize file I/O cost.

The effect of $K$. As discussed earlier, the optimal number of buckets can be expressed as $K = kN^{1/2}$. We evaluate the deduplication performance on the RealNews dataset using different values of $k$ and describe the process for selecting an appropriate configuration. As shown in Table 8, increasing $k$ beyond a certain threshold does not increase the number of deduplicated documents, while the comparison time begins to plateau. Based on experiments across multiple datasets, we find that setting $k = 4$ provides the best trade-off between computational cost and deduplication effectiveness for SEDD.

E System Configuration

We use a 8-node GPU cluster with a storage node for our experiments. Each node is equipped with four NVIDIA Tesla V100 GPUs, with each GPU having 32GB of memory. Table 9 provides the detailed target system configuration.

F Language model configuration

The detailed configuration of the LLaMA 1B model used in the language model experiments described in Section 4.3 is provided in Table 10.

G Load imbalance analysis

Table 11 presents the per-rank statistics of bucket sizes and outlier comparisons in the C4 dataset. Each row corresponds to a rank (i.e., a band in the MinHash LSH process), and the columns are defined as follows:

- **Mean**: the average number of documents per bucket for the given rank.
- **StdDev**: the standard deviation of the number of documents across buckets, indicating the spread of bucket sizes.

- **Comparisons**: the total number of pairwise comparisons for the rank.
- **Outliers**: the number of buckets whose size exceeds the mean by more than two standard deviations.
- **%Out**: the percentage of buckets that are considered outliers relative to the total number of buckets.
- **OutComp**: the total number of pairwise comparisons performed within outlier buckets.
- **%OutComp**: the proportion of comparisons in outlier buckets relative to the total number of per-rank comparisons.

The number of pairwise comparisons reported in Table 11 is computed based on the actual number of documents in each bucket during the MinHash LSH process on the C4 dataset. For a bucket containing $n$ documents, the total number of pairwise comparisons is calculated using the formula $n(n-1)/2$, which corresponds to summing all pairs of documents within the bucket. Similarly, the OutComp column accounts for comparisons in outlier buckets (buckets whose size exceeds the mean by more than two standard deviations) using the same $n(n-1)/2$ calculation per bucket. Outlier buckets—defined as those whose size exceeds the mean by more than two standard deviations—constituted a small fraction of all buckets, typically around 0.59%–1.94% per rank. The number of pairwise comparisons within these outlier buckets (OutComp) accounted for only approximately 2.62%–4.36% of the total per-rank comparisons, indicating that the contribution of outliers to the overall computation is minimal and can be considered negligible.

Table 11: Per-rank statistics of bucket sizes and pairwise comparisons in the C4 dataset.

| Rank | Mean | StdDev | Comparisons | Outliers | %Out | OutComp | %OutComp |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 0 | 1,790.82 | 331.52 | 47,516,269,551 | 264 | 0.92 | 3,091,775,767 | 3.25 |
| 1 | 1,790.82 | 238.53 | 46,756,449,092 | 387 | 1.35 | 3,594,688,373 | 3.84 |
| 2 | 1,790.82 | 248.64 | 46,827,051,391 | 373 | 1.30 | 3,408,740,924 | 3.64 |
| 3 | 1,790.82 | 425.97 | 47,541,685,437 | 243 | 0.85 | 2,933,144,397 | 3.02 |
| 4 | 1,790.82 | 215.24 | 46,604,982,146 | 380 | 1.33 | 3,167,172,510 | 3.40 |
| 5 | 1,790.82 | 259.04 | 46,902,741,222 | 310 | 1.08 | 3,558,005,800 | 3.79 |
| 6 | 1,790.82 | 232.75 | 46,717,393,506 | 371 | 1.29 | 3,397,637,751 | 3.63 |
| 7 | 1,790.82 | 331.54 | 47,516,420,018 | 278 | 0.97 | 2,487,274,545 | 2.62 |
| 8 | 1,790.82 | 147.99 | 46,254,877,288 | 556 | 1.94 | 3,504,375,831 | 3.79 |
| 9 | 1,790.82 | 210.98 | 46,578,954,192 | 415 | 1.45 | 3,480,028,218 | 3.73 |
| 10 | 1,790.82 | 196.33 | 46,493,455,190 | 387 | 1.35 | 3,247,192,167 | 3.49 |
| 11 | 1,790.82 | 282.27 | 47,082,957,868 | 224 | 0.78 | 3,363,723,606 | 3.57 |
| 12 | 1,790.82 | 181.62 | 46,413,733,631 | 419 | 1.46 | 3,197,735,770 | 3.44 |
| 13 | 1,790.82 | 334.38 | 47,543,508,465 | 169 | 0.59 | 4,151,084,255 | 4.36 |
| 14 | 1,790.82 | 193.71 | 46,478,801,548 | 457 | 1.59 | 3,434,621,567 | 3.69 |
| 15 | 1,790.82 | 210.72 | 46,577,389,821 | 463 | 1.62 | 3,278,460,404 | 3.52 |

Table 12: Mean-Only Assumption vs Actual Comparisons Per Rank

| Rank | Expected Comparisons | Actual Comparisons | Delta (A-E) | RelErr % |
| :--- | :--- | :--- | :--- | :--- |
| 0 | 45,940,962,914 | 47,516,269,551 | 1,575,306,637 | 3.32 |
| 1 | 45,940,962,914 | 46,756,449,092 | 815,486,178 | 1.74 |
| 2 | 45,940,962,914 | 46,827,051,391 | 886,088,477 | 1.89 |
| 3 | 45,940,962,914 | 47,541,685,437 | 1,600,722,523 | 3.37 |
| 4 | 45,940,962,914 | 46,604,982,146 | 664,019,232 | 1.42 |
| 5 | 45,940,962,914 | 46,902,741,222 | 961,778,308 | 2.05 |
| 6 | 45,940,962,914 | 46,717,393,506 | 776,430,592 | 1.66 |
| 7 | 45,940,962,914 | 47,516,420,018 | 1,575,457,104 | 3.32 |
| 8 | 45,940,962,914 | 46,254,877,288 | 313,914,374 | 0.68 |
| 9 | 45,940,962,914 | 46,578,954,192 | 637,991,278 | 1.37 |
| 10 | 45,940,962,914 | 46,493,455,190 | 552,492,276 | 1.19 |
| 11 | 45,940,962,914 | 47,082,957,868 | 1,141,994,954 | 2.43 |
| 12 | 45,940,962,914 | 46,413,733,631 | 472,770,717 | 1.02 |
| 13 | 45,940,962,914 | 47,543,508,465 | 1,602,545,551 | 3.37 |
| 14 | 45,940,962,914 | 46,478,801,548 | 537,838,634 | 1.16 |
| 15 | 45,940,962,914 | 46,577,389,821 | 636,426,907 | 1.37 |

Total 735,055,406,624 749,806,670,366 14,751,263,742 1.97

Table 12 reports the comparison counts per rank under the Min-Hash LSH process, illustrating the difference between a hypothetical uniform distribution and the actual computation. Each column is defined as follows:

- **Expected Comparisons**: the total number of pairwise comparisons assuming that all buckets contain exactly the same number of documents.
- **Actual Comparisons**: the total number of pairwise comparisons actually performed for the rank.
- **Delta (A-E)**: the difference between the actual and expected comparisons.

- **RelErr %**: the relative error expressed as a percentage of the expected comparisons.

As shown in Table 12, while individual ranks exhibit small deviations from the expected counts, the total number of actual comparisons across all ranks differs from the perfectly uniform case by only approximately 1.97%. This indicates that although minor imbalances exist in document distribution among buckets, their overall impact on the total number of pairwise comparisons is minimal.