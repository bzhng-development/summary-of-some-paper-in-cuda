# Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism

**ArXiv:** [1909.08053](https://arxiv.org/abs/1909.08053)

## 🎯 Pitch

Megatron-LM introduces a straightforward, intra-layer model parallelism technique that enables training transformer language models with billions of parameters by making only a few changes to standard PyTorch code. This innovation not only breaks the memory barrier for single devices but also achieves impressive scaling efficiency—demonstrated by state-of-the-art results and robust performance at the scale of up to 8.3 billion parameters and 512 GPUs—thus opening the door to larger, more powerful NLP models and better downstream task performance.

---

## 1. Executive Summary
Megatron-LM introduces a simple, intra-layer model-parallel method that lets standard PyTorch train transformer language models with billions of parameters by inserting only a few communication operations. It achieves strong throughput and accuracy at unprecedented scales (e.g., an 8.3B-parameter GPT-2 and a 3.9B-parameter BERT) and sets new state-of-the-art results, while sustaining 15.1 PFLOPs with 76% scaling efficiency on 512 GPUs (Section 5.1, Figure 5).

## 2. Context and Motivation
- Problem addressed
  - Very large transformer language models no longer fit in the memory of a single accelerator, and naïve scaling via larger batches can harm optimization (Section 2.3). The question is how to split a transformer across multiple GPUs with minimal code changes, minimal communication, and strong efficiency.
- Why it matters
  - Larger language models improve downstream NLP task performance substantially (Section 1). Being able to train multi-billion parameter models expands capability for generative modeling, reading comprehension, and transfer learning—demonstrated by state-of-the-art (SOTA) results on WikiText103, LAMBADA, and RACE (Sections 5.2–5.3, Tables 3 and 5).
- Prior approaches and shortcomings
  - Data parallelism only works if the whole model fits on a single device (Section 2.3).
  - Pipeline model parallelism (e.g., GPipe) requires nontrivial pipeline scheduling and can suffer “pipeline bubbles” and optimizer changes (Section 2.3).
  - Distributed tensor frameworks (e.g., Mesh-TensorFlow, FlexFlow) often require new languages/compilers or substantial rewrites (Section 2.3).
  - Parameter sharing (e.g., ALBERT) reduces memory but caps model capacity (Section 2.3).
- Positioning
  - Megatron-LM targets intra-layer (tensor) model parallelism for transformers that:
    - Is implemented with a handful of collectives inserted into a standard PyTorch model (Sections 3 and B.1).
    - Is orthogonal to pipeline parallelism (can be combined later, Section 3).
    - Focuses on reducing communication by careful tensor partitioning and by fusing loss with parallel logits (Sections 3 and Figure 4).

## 3. Technical Approach
This section explains how Megatron-LM splits the transformer across GPUs to minimize synchronization while keeping most compute local.

Key terms (defined when first used):
- `model parallelism`: splitting different parts of a model’s parameters and computations across multiple devices.
- `data parallelism`: each device holds a full copy of the model and processes a different minibatch; gradients are averaged across devices.
- `intra-layer (tensor) parallelism`: partitioning the tensors (weights/activations) within a layer across devices, rather than assigning whole layers to devices.
- `GEMM`: general dense matrix–matrix multiplication; the workhorse of transformer layers.
- `all-reduce`: a collective communication that sums (or otherwise reduces) tensors across devices and returns the result to all of them.

A transformer layer (Figure 2) has two heavy blocks: multi-head self-attention and a two-layer MLP. Megatron introduces tensor-parallel patterns for both (Figure 3), such that each layer requires only two all-reduces in the forward pass and two in the backward pass (Figure 4).

3.1 MLP block parallelization (Section 3; Equations (1)–(3); Figure 3a)
- The MLP consists of:
  - First GEMM: `X @ A`, then nonlinearity `GeLU` (Gaussian Error Linear Unit), then
  - Second GEMM to project back to hidden size.
- Two possible ways to partition the first GEMM:
  1) Split `A` by rows and `X` by columns. This would require summation before applying `GeLU` because `GeLU` is nonlinear (Equation (2)): synchronization point appears mid-block.
  2) Split `A` by columns. Then each GPU computes `Y_i = GeLU(X @ A_i)` independently (Equation (3)), avoiding synchronization before the nonlinearity.
- Megatron chooses (2): column-parallel first GEMM; row-parallel second GEMM.
  - Each GPU processes its slice `A_i` locally through `GeLU` and provides its local output to the second GEMM without communication.
  - A single all-reduce is needed only after the second GEMM in the forward pass to combine partial results before dropout/residual (Figure 3a).
- Autograd helpers `f` and `g` (Code 1)
  - `f`: identity in forward; all-reduce in backward (used where gradients need to be summed).
  - `g`: all-reduce in forward; identity in backward (used where activations must be summed).
  - These encapsulate the only communications you add to the baseline PyTorch transformer.

3.2 Self-attention parallelization (Section 3; Figure 3b)
- Partition by attention head:
  - Split the `Q`, `K`, and `V` projection matrices column-wise so each GPU holds a subset of heads and computes attention for its heads locally—no immediate synchronization.
  - The output projection after attention is split row-wise, consuming local outputs directly.
- Result: the entire attention block, like the MLP, needs only one all-reduce in forward and one in backward; hence two per block, four per layer total (Figure 4).

3.3 Embedding and output logits (Section 3)
- Transformers often tie input and output embeddings.
- Input embedding table `E` is huge (`hidden_size × vocab_size`). Megatron partitions `E` column-wise by vocabulary (each GPU holds a slice of the vocab). After lookup, an all-reduce (`g`) follows the embedding to combine.
- Output logits would naively require an expensive all-gather across the full vocabulary (`batch × seq_len × vocab_size`), which is prohibitive.
  - Megatron fuses the parallel output projection with cross-entropy loss so that only per-example scalar losses are communicated, reducing communication from O(`b×s×v`) to O(`b×s`) (Section 3).

3.4 Reducing communication by duplicating cheap ops (Section 3)
- Dropout, residual adds, and layer normalization are duplicated on each GPU instead of broadcasting intermediate results. This avoids extra communication at the cost of a small amount of extra compute.
- Each GPU keeps a private copy of layer norm parameters and updates them locally (Section 3).

3.5 Hybrid with data parallelism (Appendix B.1; Figure 8)
- GPUs are grouped along two dimensions:
  - Model-parallel groups: each group holds slices of one model instance (e.g., 8 GPUs).
  - Data-parallel groups: replicas of the model-parallel group processing different data shards (e.g., 64 groups).
- Gradient all-reduces are done within each data-parallel group; tensor-parallel all-reduces are done within each model-parallel group.

3.6 Randomness correctness (Appendix B.2)
- To keep dropout consistent:
  - Residual-path dropout (outside tensor-parallel regions): same RNG seed on all GPUs → identical masks.
  - Dropout inside tensor-parallel regions: per-GPU RNG seeds → different masks for different slices, preserving randomness across the combined operation.

3.7 Training and systems setup (Sections 4 and 5)
- Data pipeline (Section 4.1): 174 GB of deduplicated text from Wikipedia, CC-Stories, RealNews, OpenWebText; for BERT also BooksCorpus. De-duplication via LSH with Jaccard > 0.7; remove short docs.
- Optimization (Section 4.2): mixed precision with dynamic loss scaling, Adam with weight decay, grad clipping, dropout 0.1, activation checkpointing per layer.
- Hardware (Section 5): up to 32 DGX-2H nodes (512 V100 32GB), NVSwitch intra-node (300 GB/s GPU–GPU), 8× InfiniBand per node (100 GB/s total).

3.8 Evaluation specifics you need to interpret results
- Weak scaling vs strong scaling:
  - Weak scaling: increase problem size (here, model parameters) with number of GPUs, seeking constant per-GPU load (Section 5.1).
  - Strong scaling: fixed problem size, more GPUs; diminishing returns when communication dominates (Appendix D.2).
- Perplexity metric (Appendix E.1): exponentiated average cross-entropy over tokens (Equation (4)); computed with sliding 1024-token windows and overlap of 32 tokens due to transformer’s fixed context.
- LAMBADA accuracy (Appendix E.2): last word prediction over 4–5 sentence contexts; for subword models, the whole multi-subword target must be correct.

## 4. Key Insights and Innovations
- Intra-layer partition patterns that eliminate mid-block sync
  - Column-parallel first GEMM + row-parallel second GEMM in MLP; head-parallel attention with row-parallel output (Section 3; Figure 3).
  - Significance: reduces communication to just two all-reduces per layer in each direction (Figure 4), yielding high efficiency (77% weak-scaling efficiency at 8-way model parallel; Figure 5).
- Fused vocabulary-parallel logits with loss
  - Compute loss directly on partitioned logits to avoid an all-gather of size `b×s×v` (Section 3).
  - Significance: communication drops from tens of millions of elements to `b×s`, unlocking large-vocab training at scale.
- Minimal, PyTorch-native implementation via `f`/`g` autograd functions
  - Identity/all-reduce conjugate ops inserted at a few points (Section 3, Code 1).
  - Significance: no compiler, no model rewrite; simple to adopt and orthogonal to pipeline parallelism.
- Architecture tweak that enables large BERT to train stably
  - Rearranging layer normalization and residual connections (pre-norm style; Figure 7b) eliminates training instabilities that appear as size grows (Section 5.3).
  - Significance: enables monotonic gains up to 3.9B parameters on multiple downstream tasks (Table 5), where earlier BERT variants degraded with size.

## 5. Experimental Analysis
5.1 Setup and metrics
- Scaling study (Section 5.1; Table 1; Figure 5)
  - Four GPT-2–style configurations from ~1.2B to 8.3B parameters; hidden size per head fixed at 96; up to 8-way model parallelism and 64-way data parallelism (total 512 GPUs).
  - Baseline: 1.2B model on a single V100 sustains 39 TFLOPs (30% of peak).
  - Weak scaling target: keep ≈1B parameters per GPU; increase GPUs → larger total model.
- GPT-2 evaluation (Sections 5.2 and 4.2; Table 2; Figure 6; Table 3)
  - Training: sequences of 1024 tokens, batch size 512, 300k iterations; cosine LR schedule.
  - Metrics: validation perplexity; zero-shot WikiText103 perplexity and LAMBADA accuracy.
- BERT evaluation (Section 5.3; Table 4; Table 5; Figure 7)
  - Pretraining up to 2M iterations (1.5M for 3.9B); SOP (sentence order prediction) replaces NSP; whole-word n-gram masking.
  - Finetune on MNLI, QQP, SQuAD 1.1/2.0 (dev), and RACE (test); report median of 5 seeds; ensemble results also shown (Appendix A lists hyperparameters).

5.2 Main quantitative results
- Scaling efficiency and throughput (Section 5.1; Figure 5)
  - Model-parallel weak scaling: 8.3B on 8 GPUs achieves 77% of linear scaling vs 1.2B on 1 GPU.
  - Model + data parallel (512 GPUs): 8.3B achieves 74% efficiency; total sustained throughput 15.1 PFLOPs; overall scaling efficiency 76% vs the single-GPU baseline.
- GPT-2 accuracy (Section 5.2; Table 3; Figure 6)
  - Larger models converge faster and to lower validation perplexity (Figure 6).
  - Zero-shot test results:
    - 355M: 19.31 (WikiText103 PPL), 45.18% (LAMBADA).
    - 2.5B: 12.76 PPL, 61.73% acc.
    - 8.3B: 10.81 PPL, 66.51% acc, surpassing prior SOTA: “15.79 PPL” (WikiText103) and “63.24%” (LAMBADA) (Table 3).
  - Training time per epoch for 8.3B on 512 GPUs: ~2.10 days (Table 2).
- BERT scaling and downstream gains (Section 5.3; Table 5; Figure 7)
  - With pre-norm arrangement (Figure 7b), validation perplexity decreases monotonically as size increases (1.58 → 1.30 → 1.16).
  - Finetuning median results (Table 5; trained tokens ratio normalized to 336M = 1):
    - MNLI (m/mm): 336M 89.7/90.0 → 1.3B 90.9/91.0 → 3.9B 91.4/91.4.
    - QQP: 92.3 → 92.6 → 92.7.
    - SQuAD 1.1 F1/EM: 94.2/88.0 → 94.9/89.1 → 95.5/90.0.
    - SQuAD 2.0 F1/EM: 88.1/84.8 → 90.2/87.1 → 91.2/88.5.
    - RACE test accuracy: 83.0 → 87.3 → 89.5 (single models); 3.9B ensemble hits 90.9 on RACE (Table 5).
- Robustness/controls
  - Data leakage checks: 8-gram overlap—WikiText103: ≤10.8%; LAMBADA: ≤1.4% (Section 5.2).
  - Evaluation details ensure fair PPL computation with detokenization and context-windowed loss (Appendix E.1).

5.3 Ablations and diagnostics
- Attention heads vs scaling (Appendix D.1; Table 7)
  - For the 8.3B model at 8-way tensor parallel:
    - 16 heads: 82% efficiency; 24 heads: 80%; 32 heads: 77%.
  - Interpretation: more heads shrink per-head GEMMs and enlarge softmax, modestly increasing communication/overhead.
- Strong scaling of a fixed 1.2B model (Appendix D.2; Table 8)
  - Speedup with 2/4/8 GPUs: 1.64×/2.34×/2.98× at fixed batch size 8—diminishing returns as memory bandwidth and comms dominate.

5.4 Do the experiments support the claims?
- Yes, on three axes:
  - Efficiency: Clear weak-scaling curves (Figure 5) and large, sustained FLOPs (Section 5.1).
  - Capability: Training truly large models (8.3B GPT-2; 3.9B BERT) with standard PyTorch code paths.
  - Accuracy: Monotonic gains with size; SOTA on key benchmarks (Tables 3 and 5).
- Caveat: Results are measured on high-end NVSwitch/Infiniband clusters; generalization to commodity interconnects is not evaluated.

## 6. Limitations and Trade-offs
- Hardware dependence and network assumptions
  - The approach relies on fast collective communication (NVSwitch intra-node and multi-IB inter-node). Efficiency may drop on slower interconnects; no results are reported for such setups (Section 5 infrastructure).
- Communication still bounds scaling at extremes
  - Even with minimized collectives (two all-reduces per block), weak-scaling efficiency is <100% and degrades with more heads (Figure 5; Table 7).
  - Strong scaling shows diminishing returns past a few GPUs for a fixed-size model (Appendix D.2).
- Memory duplication for non-parallel ops
  - Layer norm parameters and computations are duplicated on every GPU (Section 3). This adds modest memory/compute overhead; beneficial for comms but not free.
- Model shapes and padding constraints
  - Vocabulary is padded to make per-GPU vocab divisible by 128 and the model-parallel degree (Section 5.1); this is an extra detail practitioners must handle.
- Scope
  - Experiments are limited to up to 8-way intra-layer model parallelism and up to 512 GPUs via data parallelism; models larger than ~16B parameters are only discussed conceptually as requiring hybrid inter-node and inter-layer parallelism (Conclusion).
- Optimization and stability beyond pre-norm BERT
  - The paper identifies pre-norm (Figure 7b) as key for scaling BERT, but does not provide a broader stability analysis across architectures and tasks or compare to other stabilizing tricks (e.g., Adafactor, different initializations).

## 7. Implications and Future Directions
- Field-level impact
  - Establishes a practical recipe for “tensor parallelism” in transformers that has since influenced many large-model training systems: split along heads and hidden columns, fuse logits with loss, and limit collectives to a few all-reduces per layer.
  - Demonstrates that simply making models larger (if you can train them efficiently) continues to push SOTA across generative and comprehension benchmarks (Tables 3 and 5).
- What this enables next
  - Combining tensor parallelism with pipeline parallelism and sharded optimizers to go well beyond 16B parameters (Conclusion).
  - Applying the same patterns to other architectures (e.g., XLNet, T5) and harder tasks like long-context QA or summarization (Conclusion).
  - Knowledge distillation from multi-billion–parameter “teacher” models into smaller “students” for deployment (Conclusion).
- Practical applications
  - Organizations with multi-GPU servers can adopt this PyTorch-native approach to train very large language models for:
    - High-quality text generation and completion (GPT-2 style).
    - Reading comprehension and multi-sentence reasoning (RACE).
    - General-purpose representation learning for transfer to downstream tasks (BERT-style).
  - The minimal-code-change design lowers barriers to entry for research groups with access to adequate compute.

> Most actionable takeaway: If you already have a PyTorch transformer, you can scale it to multi-billion parameters by (i) column-splitting the first linear and QKV projections, (ii) row-splitting the second linear/output projections, (iii) inserting only two all-reduces per block in forward and backward (Figure 4), and (iv) fusing the parallel output logits with cross-entropy to avoid a giant all-gather (Section 3). With pre-norm for BERT (Figure 7b), accuracy scales smoothly with size.
