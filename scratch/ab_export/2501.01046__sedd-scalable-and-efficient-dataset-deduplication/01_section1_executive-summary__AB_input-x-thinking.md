# Section 1 — Executive Summary — A/B

**2501.01046 — SEDD: Scalable and Efficient Dataset Deduplication with GPUs**


_input chars: PyMuPDF=69,841 · GLM-OCR=61,858_



## PyMuPDF input · THINK-NONE

## 1. Executive Summary

This paper introduces **SEDD**, a high-performance GPU-accelerated deduplication framework optimized for distributed cluster environments. Evaluated on the RealNews and C4 datasets using a cluster of NVIDIA V100 GPUs, SEDD introduces a computationally efficient, partially **reusable rolling hash function** (enabling incremental computation across adjacent shingles via Equation 2) alongside a **streaming-based extraction strategy** that replaces physical data shuffling with on-the-fly pipelined execution and **hardware-aware automatic parameter selection** (dynamically tuning bucket count 𝐾 and concurrency 𝐶 to balance GPU occupancy against fragmentation overhead). On a single node with four GPUs processing 30 million documents, SEDD outperforms the CPU-based deduplication tool in SlimPajama by up to 158× and the GPU-based tool in NVIDIA NeMo Curator by up to 7.8×, while completing deduplication of 1.2 trillion tokens in just 3 hours on an 8-node 32-GPU V100 cluster—establishing that near-duplicate detection at trillion-token scale becomes practical only when communication bottlenecks are systematically eliminated through streaming execution and communication-computation overlapping.


## GLM-OCR input · THINK-NONE

## 1. Executive Summary

This paper introduces **SEDD** (Scalable and Efficient Dataset Deduplication with GPUs), a high-performance GPU-accelerated framework for approximate document deduplication in distributed cluster environments. SEDD targets the MinHash Locality-Sensitive Hashing (LSH) pipeline—standard in LLM data curation—and accelerates it through a **computationally efficient, partially reusable hash function** (a Rabin-Karp rolling hash that reuses adjacent shingle computations via modular arithmetic), highly optimized GPU kernels for pairwise signature comparison, and a **streaming-based extraction** strategy that replaces physical data shuffling with on-the-fly bucket processing. On a node with four V100 GPUs processing 30 million documents, SEDD achieves up to **158× speedup over the CPU baseline** and **7.8× speedup over NVIDIA NeMo Curator**, while completing deduplication of a 1.2-trillion-token corpus in only **3 hours on an 8-node 32-GPU V100 cluster**. Despite these throughput gains, the framework maintains high deduplication fidelity—duplicate document sets achieve Jaccard similarities exceeding 0.95 relative to exact MinHash—establishing that hardware-aware, communication-minimizing design can dramatically accelerate large-scale data curation without sacrificing accuracy.