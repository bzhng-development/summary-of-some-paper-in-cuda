# Liger Kernel: Efficient Triton Kernels for LLM Training

**ArXiv:** [2410.10989](https://arxiv.org/abs/2410.10989)

## 🎯 Pitch

Liger-Kernel introduces a suite of open-source, highly optimized Triton GPU kernels that fuse common operations used in large language model (LLM) training, drastically reducing memory usage and kernel launch overhead. By targeting the 'last-mile' inefficiencies of PyTorch's operation execution with numerically precise, operation-specific fused kernels, Liger-Kernel delivers up to 60% memory savings and 20% faster throughput across popular LLMs. This breakthrough enables training larger models, with longer sequences and bigger batches, unlocking new regimes of LLM training that were previously constrained by hardware limitations.

---

## 1. Executive Summary
Liger-Kernel is an open-source suite of Triton GPU kernels that fuse common operations used in large language model (LLM) training to reduce memory traffic and kernel launch overhead. It targets “last‑mile” efficiency by replacing PyTorch’s step-by-step execution with fused, numerically careful kernels, yielding large end-to-end gains: roughly 20% average throughput improvement and up to ~60% peak GPU memory reduction across popular LLMs (Section 4.2), with much larger wins for specific kernels (Figures 2–3).

## 2. Context and Motivation
- Problem or gap
  - Training modern LLMs is increasingly constrained by GPU memory and kernel launch overheads. In eager-mode PyTorch, every operation incurs Python dispatch and a CUDA launch; intermediate activations are materialized for backprop, inflating memory (Section 2).
  - The key bottleneck is memory traffic between high-bandwidth memory (HBM) and on-chip SRAM. Even when floating-point arithmetic is fast, repeatedly moving tensors in and out of HBM stalls compute (Section 2.2).

- Why this matters
  - Real-world impact: Memory spikes limit feasible batch size, sequence length, and vocabulary size, directly inflating training cost or making some regimes (e.g., long context, multi-head decoding) infeasible.
  - Theoretical significance: Fused algorithms (like FlashAttention) leverage hierarchical memory to change effective complexity of core operations (e.g., avoiding materialization of large intermediates).

- Prior approaches and their limits
  - Model compilers (torch.compile, TVM, XLA, nvFuser; Section 2.1) fuse general graphs but often struggle to apply algorithm-specific tricks that exploit structure (e.g., attention tiling).
  - Custom fused kernels exist (FlashAttention, xFormers, Unsloth; Section 2.3), but coverage across all LLM training hotspots (norms, activations, loss, position encodings, and output-layer loss for huge vocabularies) is incomplete, and integration/testing can be hard.

- How this work positions itself
  - Liger-Kernel focuses on a curated set of high-impact, training-centric kernels, implemented in Triton for portability and ease of contribution.
  - It complements compilers by delivering algorithm-aware fusions and numerics, packaged behind a minimal, patch-in-place API and backed by convergence, correctness, and performance tests (Sections 3.1, 3.3, 3.4).

## 3. Technical Approach
Liger-Kernel rewrites specific training hotspots as fused Triton kernels. General pattern: reshape inputs to two-dimensional matrices of shape `(B×T, H)` (batch × sequence collapsed, hidden dim last), process each row independently in parallel (Section 3.2), and fuse steps that would otherwise be separate kernels. Below is a walk-through by kernel, emphasizing how each fusion reduces memory traffic or recomputation.

- Design choices across kernels
  - Operate row-wise on `(B×T, H)` so Triton can map each row to one program instance/warp set (Section 3.2).
  - Cache small per-row scalars (e.g., inverse RMS) computed in forward for reuse in backward, avoiding recomputations and extra reads (Figures 2d, 3d; Section 3.2 RMSNorm).
  - When a parameter (e.g., `γ`/`β`) is shared across rows, aggregate gradients efficiently (atomic or two-stage reductions; footnote 8 in Section 3.2).
  - Ensure tensor contiguity before kernel calls to avoid illegal memory access and numerical divergence (Section 3.3.4).
  - Guard against integer overflow in Triton program IDs by using `int64` for very large dimensions (Section 3.3.1).

- Fused `RMSNorm` (Equations (1)–(2), Section 3.2)
  - What it does: Computes root-mean-square normalization and scaling in one pass, caching `RMS(x)` for backward. Output: `y = (x / RMS(x)) ⊙ γ`.
  - Backward: Reuses cached RMS to compute gradients w.r.t. `x` and `γ` without extra memory reads.
  - Why it helps: Avoids writing/reading the normalized vector and separate scaling step; aggregates `γ` gradients across rows efficiently.

- Fused `LayerNorm` (Equations (3)–(4), Section 3.2)
  - What it does: Centers, normalizes, scales, and biases in one pass: `y = ((x − mean)/RMS(x − mean)) ⊙ γ + β`. Caches `inv_rms`.
  - Backward: Computes gradients for `x`, `γ`, `β` using the cached scalar(s). Efficient reduction across rows for `γ`, `β`.

- Fused RoPE for queries/keys (Equations (5)–(6), Section 3.2)
  - RoPE (Rotary Positional Embedding) rotates pairs of hidden dimensions by position-dependent angles (here, implemented in the Hugging Face style, not the Su et al. full dense matrix).
  - What it does: Applies rotations to both `q` and `k` in a single kernel, exploiting the block-sparse 2×2 rotation structure (no materialization of a dense `d×d` matrix).
  - Implementation trick: Store sinusoid angles in a flattened 1D tensor and reuse block patterns to minimize memory reads.

- Fused `SwiGLU` and `GeGLU` elementwise (Equations (7)–(14), Section 3.2)
  - These are gated MLP activations that multiply a “gate” branch with a “value” branch.
  - Forward: Given pre-projected `x1 = W x + b` and `x2 = V x + c`, compute `y = SiLU(x1) ⊙ x2` (SwiGLU) or `y = GELU(x1) ⊙ x2` (GeGLU).
  - Backward: Fuse derivative formulas so the gate’s derivative and the product with `x2` are computed together, and avoid saving full activation tensors by recomputing `SiLU`/`GELU` in backward (Figures 3b–3c show memory savings).

- Fused `CrossEntropy` (CE) with online-softmax and in-place gradients (Equations (15)–(16), Section 3.2)
  - Challenge: For vocabulary size `V`, materializing logits `x ∈ R^V` for many tokens explodes memory.
  - What it does:
    - Computes softmax in an online (streaming) manner and immediately forms `∇xL = y − t`, storing gradients in place of logits to avoid simultaneous storage of both.
    - Uses numerically safe log operations to improve stability.
  - Benefit: Reduces both kernel count and peak memory by overwriting the logit tensor with its gradient, and avoids extra buffers for `softmax`.

- Fused Linear + CrossEntropy (FLCE) with input chunking (Equations (17) and Figure 1, Section 3.2)
  - Problem: The final linear projection to vocabulary (`W^T h`) creates a massive logits tensor when `V` is large (e.g., 128k or 256k), dominating peak memory (discussion around Gemma example, Section 3.2).
  - Key idea:
    - Flatten the 3D hidden states `(B, T, H)` to `(BT, H)` and process them in chunks along the `BT` dimension.
    - For each chunk: compute `x = W^T h`, call the non-fused CE kernel that does online-softmax and in-place gradient write, then immediately backpropagate to accumulate `∇h = W ∇x` and `∇W += h (∇x)^T` (Equation (17)).
    - Never materialize the full `(BT, V)` logits; only a chunk’s logits exist at any time and are overwritten by gradients.
  - Chunk size heuristic:
    - Choose a power-of-two chunk size to keep GPU utilization high while keeping memory close to the hidden size. The paper suggests setting the chunk size “closer to the hidden dimension size” and gives a rule of thumb formula (Section 3.2, “In practice, we set the chunk size … closer to the hidden dimension size…”).
  - Gradient normalization remark:
    - If the training loss uses mean reduction across tokens, per-chunk gradients should be scaled by `chunk_size / (B×T)` to match the true mean (Section 3.2, “Remark”).

- Testing and integration practices (Section 3.3, 3.4)
  - Correctness checks: Compare against pure PyTorch or Hugging Face references on both regular and odd shapes; strict tolerances for fp32 (atol=1e-7, rtol=1e-5) and bf16 (atol=1e-3, rtol=1e-2), sometimes relaxed with convergence tests (Section 3.3.1).
  - Performance testing: Use realistic shapes (e.g., batch 4, hidden 2048, multiple sequence lengths) to mirror production (Section 3.3.2).
  - Convergence testing: Small-scale training runs that verify identical logits, weights, and loss over steps even when tensor layouts differ from unit tests (Section 3.3.3).
  - Contiguity: Make tensors contiguous before kernel invocation; a non-contiguous gradient in SDPA once caused loss divergence—fix by enforcing contiguity (Section 3.3.4).
  - Integration: One-line patch via `AutoLigerKernelForCausalLM`, model-specific `apply_liger_kernel_to_*` helpers, or compose `LigerLayerNorm`/`LigerCrossEntropyLoss` directly (Section 3.1). Works with Trainer, TRL’s `SFTTrainer`, Axolotl, LLaMA-Factory (Section 3.4).

## 4. Key Insights and Innovations
- Operation-specific fusion beats general compilation for targeted wins
  - Insight: Many training bottlenecks stem from a handful of high-frequency operations where algorithm-aware fusion (e.g., caching RMS scalars; in-place CE gradient) yields outsized gains. This complements general-purpose compilers (Section 2.1 vs. 2.2).
  - Significance: Demonstrates large kernel-level speedups (up to 7–8× for norms and RoPE at large hidden sizes; Figures 2d–2f) and substantial memory drops (up to ~5× for CE at V=163,840; Figure 3a).

- In-place gradients and online-softmax for loss computation
  - What’s new: The CE kernel computes gradients during forward and overwrites logits (`x`) with `∇x` (Section 3.2 CE). This avoids co-existing large tensors and cuts memory spikes.
  - Why it matters: At very large `V`, this is the difference between feasible and OOM. The paper reports “approximately 3× faster execution” and “~5× less memory” for the CE kernel at `V=163,840` (Figures 2a and 3a).

- Fused Linear + CE with input chunking (FLCE)
  - What’s different: Rather than compute all logits then loss, Liger chunks the input stream, immediately consumes logits inside CE (which writes gradients in-place), and backpropagates per chunk (Figure 1; Equation (17)).
  - Why it matters: Enables training with huge vocabularies and multi-head decoding (Medusa) by keeping the logits tensor from ever materializing at full size (Section 4.2, Medusa).

- Practical kernel engineering for robustness
  - Non-obvious but important: explicit handling of int32 program-id overflow to avoid illegal memory access at very large sizes (Section 3.3.1), and a documented contiguity requirement with a real failure example (RoPE gradient in SDPA; Section 3.3.4).
  - Significance: These are the kinds of issues that derail production training; documenting fixes increases reliability across diverse settings.

## 5. Experimental Analysis
- Evaluation methodology
  - Kernel-level microbenchmarks (Section 4.1): Single NVIDIA A100 80GB, 10 runs each. Vary:
    - For CE: vocabulary sizes {40,960; 81,920; 122,880; 163,840}.
    - For GeGLU/SwiGLU: sequence lengths {4,096; 8,192; 12,288; 16,384}.
    - For RMSNorm/LayerNorm/RoPE: hidden sizes {4,096; 8,192; 12,288; 16,384}.
    - Report median speed and memory with [0.2, 0.8] quantiles as error bounds (Figures 2–3).
  - End-to-end fine-tuning (Section 4.2): 4× A100 80GB, bfloat16, AdamW + cosine LR schedule, sequence length 512, Alpaca dataset. Throughput and peak memory measured after 20 steps; means over 5 runs with standard errors (Figures 4–8).
  - Medusa use case (Section 4.2, “Medusa”): 8× A100 80GB, variable sequence length, batch size 4. Compares Stage-1 (only heads train) and Stage-2 (heads + backbone) for 3 and 5 heads (Figures 9–12).

- Main quantitative results
  - Kernel microbenchmarks (Figures 2–3):
    - CE kernel:
      > “approximately 3× faster … and … approximately 5× less memory for a vocab size of 163,840” (Figures 2a, 3a).
    - GeGLU/SwiGLU:
      > Speed at parity with baseline (Figures 2b–2c); peak memory reduced by ~1.6× at sequence length 16,384 due to recomputation in backward (Figures 3b–3c).
    - RMSNorm:
      > “~7× reduction in execution time and roughly 3× reduction in peak memory” at hidden size 16,384 (Figures 2d, 3d).
    - LayerNorm:
      > “approximately 30% reduction in execution time … with minimal memory overheads” (Figures 2e, 3e).
    - RoPE:
      > “approximately 8× speedup with approximately 3× lower memory” at hidden size 16,384 (Figures 2f, 3f).
  - End-to-end fine-tuning (Figures 4–8):
    - LLaMA 3-8B at batch 64:
      > “42.8% increase in throughput” and “54.8% reduction in GPU memory usage” (Figure 4).
    - Qwen2 at batch 48:
      > “25.5% throughput” gain and “56.8% memory” reduction (Figure 5).
    - Gemma 7B at batch 48:
      > “11.9% throughput” and “51.8% memory” reduction (Figure 6).
    - Mistral 7B at batch 128:
      > “27% increase in throughput … 21% drop in GPU memory” (Figure 7).
    - Phi-3 at batch 128:
      > “17% … throughput … 13% [lower] memory” (Figure 8).
  - Medusa (multi-token prediction) with 128k vocabularies where logits materialization often OOMs:
    - Using Liger FLCE avoids OOM and improves both throughput and peak memory in Stage-1 and Stage-2 with 3 or 5 heads (Figures 9–12). The paper notes standard errors are usually <1% and often not visible.

- Do experiments support the claims?
  - Yes for performance/memory: The kernel-level plots show consistent speed/memory benefits across a range of realistic shapes. End-to-end results across five distinct LLM families indicate improvements persist when integrated with real training stacks and optimizers (Section 4.2).
  - Convergence/accuracy: The paper emphasizes convergence tests and exactness checking (Section 3.3), but the end-to-end section focuses on throughput and memory; it does not report fine-tuned model quality metrics on Alpaca. That said, the kernels for norms/activations/loss reuse exact formulas (Equations (1)–(17)), and correctness tests include tight tolerances with convergence checks.

- Ablations, failure cases, robustness
  - Ablations: Three gradient-aggregation implementations for normalization layers were evaluated; they chose a two-stage aggregation method from FlashAttention for better performance than plain PyTorch and atomic-only (footnote 8 in Section 3.2).
  - Robustness safeguards:
    - Contiguity requirement documented with a real divergence incident (Section 3.3.4).
    - Integer overflow avoidance by moving program IDs to `int64` (Section 3.3.1).
  - Missing ablations:
    - No systematic sensitivity study on the FLCE chunk-size heuristic.
    - Results are primarily on A100 GPUs; cross-hardware variability is not deeply explored (though CI mentions AMD and Intel funding; Section 6.2).

## 6. Limitations and Trade-offs
- Scope and coverage
  - Kernel set is focused on norms, RoPE, gated activations, and loss layers. Attention and MLP linear matmul kernels are not the central subject here (beyond the final projection in FLCE). Users may still rely on other libraries (e.g., FlashAttention) for attention (Sections 2.2–2.3, 3.2).
- Hardware and software constraints
  - Evaluations center on NVIDIA A100 GPUs; while Triton is portable, real performance can vary by architecture, compiler version, and driver stack. The paper notes AMD and Intel CI support (Section 6.2) but provides no cross-vendor benchmarks.
- Numerical behavior and tolerances
  - Although exact formulas are implemented, bf16 often requires looser tolerances (Section 3.3.1). The need to sometimes relax tolerances suggests potential edge-case sensitivity.
- Dataflow assumptions
  - FLCE depends on chunking over `(BT)` and assumes the projection weight `W` fits device memory and can be accumulated correctly across chunks. Extremely unbalanced shapes (e.g., very small `H` vs. very large `V`) might reduce GPU utilization without careful chunk tuning (Section 3.2, FLCE heuristic).
- Integration pitfalls
  - Requires contiguous tensors; non-contiguity can silently produce divergence (Section 3.3.4). Also, model code changes upstream (e.g., Hugging Face internals) could require re-patching or adaptation.

## 7. Implications and Future Directions
- How this changes the landscape
  - By packaging algorithm-aware fusions behind an easy patching API, Liger-Kernel lowers the barrier to training LLMs efficiently—especially in high‑vocabulary, long‑sequence, or multi-head decoding regimes where memory is the primary limiter.
  - The results show that modest, well-targeted fusions deliver substantial end-to-end gains, encouraging a “toolkit” approach alongside general compilers.

- Follow-up research enabled or suggested
  - Dynamic chunk-size selection: An auto-tuner for FLCE that adapts to `B, T, H, V`, GPU model, and current residency of weights/activations could further improve utilization.
  - Expanded kernel coverage: Fusing more of the MLP pipeline (e.g., linear projections + activation + dropout) and attention variants (e.g., block-sparse, long-context) with unified memory-aware scheduling.
  - Cross-hardware validation: Systematic benchmarking on AMD and Intel GPUs and across CUDA generations to quantify portability of gains.
  - Accuracy studies: Beyond convergence checks, report task metrics on standard fine-tuning/continued-pretraining suites to verify that numerical changes do not impact final model quality.

- Practical applications and downstream use
  - Resource-constrained training: Fine-tuning large vocab models (e.g., LLaMA‑3 8B with 128k vocab) on fewer/smaller GPUs by reducing peak memory (Figures 4–8).
  - Multi-token prediction frameworks (Medusa) and other multi-head objectives where parallel heads would otherwise multiply logits memory (Figures 9–12).
  - Longer context training: Memory headroom from FLCE and fused norms can be traded for longer sequence lengths or larger batch sizes.
  - Distributed training stacks: Works with PyTorch FSDP, DeepSpeed ZeRO and ZeRO++ (Sections 1, 3.4), enabling integration into existing production pipelines with minimal code changes.

Overall, the paper provides a clear, well-engineered set of kernels with careful numerics, pragmatic testing guidance, and tangible end-to-end wins. The most fundamental contributions are the in-place, online CE and the chunked fused linear+CE, which directly address the dominant memory spikes in modern LLM training.
