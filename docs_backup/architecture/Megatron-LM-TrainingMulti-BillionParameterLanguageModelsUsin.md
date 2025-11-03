# Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism

**ArXiv:** [1909.08053](https://arxiv.org/abs/1909.08053)

## 🎯 Pitch

Megatron-LM presents a simple and efficient intra-layer model parallelism method—requiring only minimal changes to PyTorch code—enabling the training of transformer language models with billions of parameters across hundreds of GPUs. This innovation makes it practically feasible to scale language models far beyond the memory of a single device, unlocking new state-of-the-art NLP results and democratizing the training of extremely large models by eliminating the need for custom frameworks or compilers.

---

## 1. Executive Summary (2-3 sentences)
This paper introduces Megatron-LM, a simple but highly effective way to train transformer language models with billions of parameters by splitting work inside each layer across many GPUs (“intra-layer model parallelism”) and combining it with standard data parallelism. The result is both practical and scalable: models up to 8.3B parameters train efficiently on 512 NVIDIA V100 GPUs with sustained 15.1 PFLOPs and strong accuracy gains, including new state-of-the-art results on WikiText-103 perplexity, LAMBADA accuracy, and RACE reading comprehension (Abstract; Sections 3–5; Table 3; Table 5; Figure 5).

## 2. Context and Motivation
- Problem/gap:
  - Very large transformer models (multi-billion parameters) do not fit in the memory of a single GPU, and naïvely distributing them either requires specialized compilers/frameworks or suffers from poor efficiency (Section 2.3).
  - Existing solutions (e.g., pipeline model parallelism in GPipe, Mesh-TensorFlow) require model rewrites and/or custom compilers; they also face pipeline bubbles and complexity in scheduling (Section 2.3).
- Importance:
  - Larger language models consistently improve performance on a wide range of NLP tasks (Section 1; Section 2.1). Unlocking training for larger models has immediate impact: better text modeling (perplexity), reading comprehension, and cloze-style tasks.
- Prior approaches and shortcomings:
  - Data parallelism scales batch size but cannot exceed the single-device model memory limit; it also risks convergence degradation at very large batches (Section 2.3).
  - Pipeline parallelism reduces per-device memory but introduces pipeline bubbles and scheduling complexity; some formulations rely on asynchronous updates that hurt convergence consistency (Section 2.3).
  - Distributed tensor compilers (e.g., Mesh-TensorFlow) are powerful but require new languages/compilers and model rewrites (Section 2.3).
- Positioning:
  - Megatron-LM uses intra-layer tensor partitioning tailored to transformer structure, implemented with “a few communication primitives” in native PyTorch—no compiler or model rewrite—while being orthogonal to pipeline parallelism (Section 3; Abstract).

## 3. Technical Approach
Megatron-LM splits the heavy linear algebra inside each transformer layer across GPUs in a way that minimizes synchronization, keeps computations local, and uses just a few collective communications.

Key terms (selective definitions):
- `all-reduce`: a collective operation that sums (or otherwise reduces) tensors across all GPUs and makes the reduced result available on each GPU.
- `GEMM`: general matrix-matrix multiplication, the dominant compute in transformer layers.
- `weak scaling`: increasing the problem size (here, model size and/or batch) proportionally with the number of GPUs, aiming to keep per-GPU work roughly constant.

Step-by-step design (Section 3; Figures 3–4; Equations 1–3; Code 1):
1) Transformer block decomposition
   - Each layer has a self-attention block and a two-layer MLP (Figure 2).
   - The approach introduces model parallelism inside both blocks.

2) Column-parallel first MLP projection, row-parallel second projection (Figure 3a)
   - Consider the first MLP projection Y = GeLU(XA) (Eq. 1).
   - If you split the weight A along columns (A = [A1, A2]), each GPU multiplies the full input X with its shard (XA1, XA2) and applies GeLU independently:
     > “[Y1, Y2] = [GeLU(XA1), GeLU(XA2)]” (Eq. 3)  
     This avoids a synchronization before GeLU (contrast with splitting along rows, which would require a sync before the nonlinearity; Eq. 2).
   - The second MLP projection is split along its rows to consume the sharded GeLU results without communication (Figure 3a).
   - Only one `all-reduce` is needed after the second projection in the forward pass (and one in backward). Megatron encodes this with complementary autograd functions:
     > `g` performs all-reduce in forward (identity in backward); `f` does identity in forward and all-reduce in backward (Figure 3; Code 1).

3) Attention: shard by heads and row-parallel output (Figure 3b)
   - Multi-head attention is naturally parallelizable by attention heads.
   - Keys/Queries/Values (K/Q/V) projections are split column-parallel so that different heads live on different GPUs; no immediate communication is needed within attention (each GPU computes attention for its heads locally).
   - The attention output’s final linear projection is row-parallel, directly consuming the sharded attention output without communication (Figure 3b).

4) Minimizing communication per layer (Figure 4)
   - With the above sharding, one transformer layer needs only:
     > “two all-reduces in the forward pass and two in the backward pass” (Figure 4).

5) Embedding and output softmax with vocabulary parallelism (Section 3)
   - Input/output embedding matrix (shape H×V) is large because vocabulary size V is big (e.g., 50k). Megatron splits the embedding across the vocabulary dimension (`E = [E1, E2, …]`, column-wise).
   - After the input embedding, an `all-reduce` (`g`) reassembles the full hidden states for subsequent computation.
   - For the output softmax, naïvely all-gathering logits would communicate `batch × sequence × vocab`—prohibitively large. Instead, Megatron fuses the parallel matmul with the cross-entropy loss so each GPU computes partial losses and only `batch × sequence` scalars are communicated:
     > “Communicating scalar losses instead of logits is a huge reduction in communication” (Section 3).

6) Duplicate cheap, non-matrix ops; synchronize only where needed (Section 3)
   - Dropout, residual adds, and layer normalization are duplicated on every GPU to avoid broadcasts.
   - Each model-parallel worker updates only its local parameters; duplicated parameters (e.g., LayerNorm’s scale/shift) are kept identical by construction (no extra sync needed).

7) Random-number generation consistency (Appendix B.2)
   - Dropout outside model-parallel regions must be synchronized across GPUs (identical patterns) to keep residual additions consistent; they seed RNGs identically.
   - Dropout inside model-parallel regions must differ across GPUs for proper randomness; they maintain a separate, uniquely-seeded RNG per worker.

8) Hybrid with data parallelism (Appendix B.1; Figure 8)
   - GPUs are grouped both for model parallel (within-server groups) and data parallel (across same-position GPUs in each model-parallel group). Gradients are all-reduced within each data-parallel group.

9) Training environment and optimization (Sections 4–5)
   - Mixed-precision with dynamic loss scaling on V100 Tensor Cores (Section 4.2).
   - Weight initialization `N(0, 0.02)` and scale-down of residual branch weights by `sqrt(1/(2N))` to stabilize deep residuals (Section 4.2).
   - Adam with weight decay 0.01, gradient norm clipping 1.0, dropout 0.1, activation checkpointing per layer (Section 4.2).

Why these choices?
- Column-parallel first projections and head-parallel attention avoid a synchronization point before nonlinearities, which is crucial for scaling (discussion around Eq. 2–3; Figure 3).
- Row-parallel second projections match the data layout of the preceding sharded outputs, keeping computation local and only adding a single `all-reduce` per block.
- Vocabulary-parallel softmax with fused loss avoids a massive all-gather of logits (`b × s × v`), replacing it with per-example scalar communications (Section 3).
- Duplicating cheap ops avoids frequent small communications that would hurt throughput.

## 4. Key Insights and Innovations
1) Intra-layer tensor parallelism that needs only four collectives per layer
   - Novelty: Shard MLP and attention so that each transformer layer requires just “two all-reduces in forward and two in backward” (Figure 4). Prior systems often synchronize more frequently or rely on compilers.
   - Significance: Keeps GPUs compute-bound and yields excellent weak scaling up to 8-way model parallelism (Figure 5).

2) Vocabulary-parallel output with fused loss
   - Novelty: Split the embedding/softmax by vocabulary and fuse the parallel matmul with cross-entropy so only losses are communicated, not logits (Section 3).
   - Significance: Dramatically reduces communication volume for very large vocabularies (tens of thousands of tokens), enabling throughput at scale.

3) Minimal changes in native PyTorch
   - Novelty: Implement model parallelism using a couple of custom autograd functions (`f` and `g`) that wrap `all-reduce` at precise graph points (Figure 3; Code 1).
   - Significance: No model rewrite or custom compiler; orthogonal to pipeline parallelism, so it can be combined with it.

4) Stability fix for very large BERT models via LayerNorm/residual ordering (Figure 7)
   - Novelty: Reorder LayerNorm and residual connections (architecture “b” in Figure 7) to eliminate instabilities seen when scaling BERT beyond ~BERT-Large.
   - Significance: Enables monotonic gains up to 3.9B parameters on multiple downstream tasks (Table 5). This is a practical recipe for training deeper/wider BERT-like models.

Incremental vs. fundamental:
- The sharding strategies are targeted, engineering-focused innovations that, together, constitute a fundamental usability step: training billion-scale transformers efficiently without specialized tooling.
- The LayerNorm/residual reordering is a fundamental stability insight for deep BERT-like models.

## 5. Experimental Analysis
Evaluation methodology and setup
- Hardware: up to 32 DGX-2H servers (512 V100 32GB GPUs), 300 GB/s intra-node (NVSwitch) and 100 GB/s inter-node (8× InfiniBand) bandwidth (Section 5).
- Datasets for pretraining (Section 4.1):
  - Aggregate of Wikipedia, CC-Stories, RealNews, OpenWebText; deduplicated with LSH (Jaccard > 0.7), documents <128 tokens filtered; 174 GB total. BooksCorpus is added for BERT but excluded for GPT-2 to avoid overlap with LAMBADA.
- Optimization (Section 4.2):
  - Mixed precision; Adam with weight decay 0.01; grad clip 1.0; dropout 0.1; activation checkpointing.
  - GPT-2: sequence length 1024; batch 512; 300k iterations; LR 1.5e-4 with 3k warmup then cosine decay to 1e-5.
  - BERT: vocab 30,522; replace next sentence prediction with sentence order prediction; whole-word n-gram masking; batch 1024; LR 1e-4, warmup 10k, linear decay over 2M iters.

Scaling experiments (Sections 5.1; Figure 1; Figure 5; Table 1)
- Models range from 1.2B to 8.3B parameters; model-parallel degree 1, 2, 4, 8 (Table 1).
- Baseline: single-GPU 1.2B model sustains 39 TFLOPs (≈30% of V100 peak), a strong baseline (Section 5.1).
- Weak scaling efficiency:
  > Model-parallel: 95% (2 GPUs), 82% (4 GPUs), 77% (8 GPUs).  
  > Model+data-parallel (64-way data parallel): 96% (64 GPUs), 83% (128), 79% (256), 74% (512) (Figure 5).
- End-to-end throughput:
  > “Sustain 15.1 PFLOPs” for the 8.3B model on 512 GPUs with “76% scaling efficiency” vs. the single-GPU baseline (Abstract; Figure 1/5).
- Sensitivity analyses (Appendix D):
  - More attention heads slightly reduce scaling efficiency (e.g., 82% at 16 heads vs. 77% at 32 heads for the 8.3B/8-way setup; Table 7), due to smaller per-head GEMMs and larger softmaxes.
  - Strong scaling for the 1.2B model (fixed batch): speedups of 1.64× (2 GPUs), 2.34× (4), 2.98× (8) (Table 8), showing diminishing returns as communication dominates.

GPT-2 language modeling results (Section 5.2; Table 2–3; Figure 6; Appendix E)
- Configurations (Table 2): 355M, 2.5B, 8.3B; total GPUs 64, 128, 512; time per epoch ≈0.86, 2.27, 2.10 days.
- Evaluation protocol:
  - WikiText-103 perplexity computed with overlapping sliding windows (context 1024, overlap 32), normalized by original word-level token count to match prior work (Appendix E.1).
  - LAMBADA cloze accuracy: teacher-forced subword prediction; an answer is correct only if all subword tokens match (Appendix E.2).
- Results:
  - Validation perplexity decreases monotonically with model size, converging faster and lower (Figure 6).
  - Zero-shot test performance (Table 3):
    > 355M: WT103 ppl 19.31; LAMBADA 45.18%  
    > 2.5B: WT103 ppl 12.76; LAMBADA 61.73%  
    > 8.3B: WT103 ppl 10.81; LAMBADA 66.51%  
    > Previous SOTA: WT103 ppl 15.79; LAMBADA 63.24%
  - Data leakage checks: 8-gram overlap ≤10.8% (WT103 test) and ≤1.4% (LAMBADA test) with the training set after preprocessing, consistent with prior practice (Section 5.2).

BERT bi-directional modeling and finetuning (Section 5.3; Figure 7; Table 4–5)
- Model sizes (Table 4): 336M (24×1024, 16 heads), 1.3B (24×2048, 32 heads), 3.9B (48×2560, 40 heads).
- Stability insight: with the reordered LayerNorm/residual (Figure 7b), larger models train stably and yield lower training loss than the original ordering (Figure 7).
- Finetuning setup: MNLI, QQP, SQuAD 1.1, SQuAD 2.0, and RACE; hyperparameter sweeps on batch size and LR; report median over 5 seeds; ensembles where noted (Table 5; Appendix A).
- Results (Table 5):
  > Megatron-336M: MNLI 89.7/90.0; QQP 92.3; SQuAD1.1 94.2/88.0; SQuAD2.0 88.1/84.8; RACE 83.0% (dev/test split m/h shown)  
  > Megatron-1.3B: MNLI 90.9/91.0; QQP 92.6; SQuAD1.1 94.9/89.1; SQuAD2.0 90.2/87.1; RACE 87.3%  
  > Megatron-3.9B: MNLI 91.4/91.4; QQP 92.7; SQuAD1.1 95.5/90.0; SQuAD2.0 91.2/88.5; RACE 89.5%  
  - On RACE test set, single-model 3.9B achieves:
    > “90.9%” accuracy vs. previous SOTA “89.4%” (Table 5 caption row and Abstract).  
  - 5-way ensembles further boost SQuAD and RACE (Table 5).

Assessment of claims
- Efficiency claims are backed by clear weak-scaling graphs (Figure 5) and a strong single-GPU baseline (39 TFLOPs, Section 5.1).
- Accuracy claims are supported by monotonic gains with model size for both GPT-2 and BERT (Figure 6; Table 3; Table 5).
- Methodological details (Appendix E) make the evaluation of perplexity and LAMBADA comparable to prior work.

## 6. Limitations and Trade-offs
- Hardware assumptions:
  - Results rely on very high intra-node bandwidth (NVSwitch 300 GB/s) and substantial inter-node bandwidth (100 GB/s via 8 IB links; Section 5). Clusters without such interconnects may see lower efficiency.
- Communication still matters:
  - Although minimized, each layer incurs four all-reduces (Figure 4). As model depth increases or interconnect quality decreases, communication can bottleneck.
  - Efficiency drops slightly as attention heads increase (Table 7), since per-head matrices shrink and softmax gets larger—a design trade-off between model expressivity and throughput.
- Memory and optimizer state:
  - ADAM maintains additional state per parameter, increasing memory pressure (Section 2.3). Activation checkpointing helps, but memory remains a constraint for >16B models (Section 6, Future Work).
- Scope of modeling:
  - The work demonstrates GPT-2-style left-to-right and BERT-style masked pretraining; other architectures (e.g., encoder-decoder T5) are suggested but not explored here (Section 6).
- Data considerations:
  - While deduplication and overlap checks are performed (Section 4.1; Section 5.2), the aggregate corpus (web-scale) may still contain biases or artifacts not analyzed in this paper.
- Strong scaling limits:
  - For fixed model/batch, speedup saturates beyond a few GPUs (2.98× with 8 GPUs on 1.2B; Table 8), indicating diminishing returns when communication dominates.

## 7. Implications and Future Directions
- Field impact:
  - Demonstrates that billion-scale transformer training can be achieved with modest code changes in PyTorch, lowering the barrier to scaling models. This shifted the community toward practical intra-layer tensor parallelism as a standard tool.
- Enabled directions (Section 6):
  - Scale further by combining this intra-layer parallelism with pipeline (inter-layer) and inter-node parallelism for models >16B parameters.
  - Improve optimizer efficiency/memory (e.g., state sharding or low-memory optimizers) to push parameter counts higher.
  - Apply the technique to other model families (XLNet, T5) and to more challenging tasks (generative QA, summarization, dialogue).
  - Use knowledge distillation to compress these very large pretrained teachers into smaller, efficient students.
- Practical applications:
  - Training very large language models with SOTA zero-shot performance (Table 3) for text completion and long-context reasoning.
  - Building stronger bidirectional encoders for downstream understanding tasks (Table 5), improving tools for search, QA, and reading comprehension.

Quote highlights for quick recall
- Throughput and scaling:
  > “We sustain 15.1 PetaFLOPs… with 76% scaling efficiency” (Abstract; Figure 1; Figure 5).
- Communication budget per layer:
  > “two all-reduces in the forward path and two in the backward path” (Figure 4).
- GPT-2 zero-shot SOTA:
  > “WikiText103 ppl 10.81; LAMBADA 66.51%” (Table 3).
- BERT SOTA on RACE:
  > “90.9% compared to SOTA 89.4%” (Abstract; Table 5).
- Stability fix:
  > Rearranging LayerNorm/residual “enables stable training with lower training loss” for larger BERTs (Figure 7).

In sum, Megatron-LM’s core contribution is a clear, implementation-friendly recipe for splitting the heavy computation inside transformer layers to fit and train multi-billion-parameter models efficiently—validated by strong systems scaling and state-of-the-art NLP results.
