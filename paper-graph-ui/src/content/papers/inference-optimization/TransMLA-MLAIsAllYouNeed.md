# TransMLA: MLA Is All You Need

**ArXiv:** [2502.07864](https://arxiv.org/abs/2502.07864)

## 🎯 Pitch

TransMLA introduces a practical framework to convert any Grouped-Query-Attention (GQA) language model into a Multi-Head Latent Attention (MLA) model, fully compatible with DeepSeek’s cutting-edge inference stack. By compressing the key–value cache with minimal performance loss and unlocking advanced MLA optimizations, TransMLA enables dramatic inference speedups—over 10x in some settings—without requiring complete retraining, thus empowering model providers to maximize efficiency, serve longer contexts, and instantly leverage the latest hardware acceleration advances.

---

## 1. Executive Summary
TransMLA presents a practical, mostly training-free procedure to convert any Grouped-Query-Attention (GQA) large language model into a Multi-Head Latent Attention (MLA) model that runs on the highly optimized DeepSeek inference stack. It compresses the key–value (KV) cache with minimal quality loss, enables the MLA “Absorb” inference mode, and achieves large real-world speedups (up to 10.6x at 8K context; Figure 5a) while preserving model behavior after light fine-tuning.

## 2. Context and Motivation
- Problem addressed
  - Modern LLMs are increasingly bottlenecked by KV-cache memory movement rather than compute. The KV cache stores past attention “keys” and “values” for every token; its size grows linearly with context length and dominates memory bandwidth during decoding.
  - Many high-quality open and proprietary models are trained with `GQA` (grouped-query attention), which reduces KV size compared to standard multi-head attention, but hardware/runtime ecosystems are now optimized around DeepSeek’s `MLA` layout and kernels, leaving model providers with sunk costs in GQA checkpoints (Abstract; Section 1).

- Why it matters
  - Shrinking the KV cache without re-training from scratch brings immediate throughput gains, lower serving costs, and longer feasible contexts on commodity accelerators (Section 1; Figure 5, Table 4).
  - A path to migrate existing GQA models to MLA unlocks DeepSeek-specific optimizations (vLLM integration, SGLang, FP8, multi-token prediction), broadening impact beyond a single model family (Abstract; Section 5.4).

- Prior approaches and gaps
  - Architectural choices: `MQA` (one shared KV head) and `GQA` (groups of shared KV heads) cut KV size but degrade quality vs. full MHA (Section 2).
  - Post-training compression: KV quantization, head sharing, token pruning, or custom cache schemas (e.g., DuoAttention, KiVi, H2O) save memory but require nonstandard runtimes or re-expansion steps that blunt speedups (Section 2).
  - DeepSeek’s `MLA`: architected to pre-train with compressed KV and a decoupled positional path, enabling the “Absorb” inference trick; however, existing checkpoints in the wild are largely GQA and not MLA (Section 3.3).

- Positioning
  - TransMLA offers a conversion framework: it proves MLA is strictly more expressive than GQA for a fixed KV budget (Appendix A; Figure 1a), then provides a sequence of equivalence-preserving transformations plus low-rank compression that turns any GQA checkpoint into an MLA checkpoint compatible with DeepSeek kernels (Section 4; Figure 1b).

## 3. Technical Approach
The pipeline converts a GQA layer to an MLA layer that supports the MLA “Absorb” inference mode. Key terms:
- `KV cache`: the per-token memory storing all past keys (`K`) and values (`V`).
- `RoPE`: rotary positional embedding; encodes token position by rotating pairs of feature dimensions with sin/cos at fixed frequencies (Eq. 1; Section 3.1).
- `Absorb`: MLA’s inference-time reparameterization that collapses per-head projections into a shared latent KV, delivering MQA-like runtime while retaining multi-head expressivity (Eqs. 9–10; Section 3.3).
- `RoRoPE`: a new rotation-and-PCA procedure that concentrates positional information into a small subspace (Section 4.2; Figure 2).
- `FreqFold`: groups adjacent RoPE frequencies (which are similar) so their principal components can be learned jointly, increasing how much positional information fits into the target subspace (Appendix C; Figure 7).
- `BKV` (Balanced Key–Value): rescales keys vs. values before joint PCA to avoid the larger-norm keys dominating the compression (Eq. 20; Appendix D; Figure 4).

Step-by-step

1) Merge all GQA K/V heads into one latent head (no quality change)
- Mechanism: Introduce per-query-head matrices `WUK_i` and `WUV_i` that “select” the original K/V group for that head. Initialize them as block identity selectors so each query head still attends to its original K/V group. Merge all K and all V into a single concatenated latent vector `c_KV_t = [c_K_t; c_V_t] = [W^K; W^V] x_t` (Eq. 11).
- The attention then becomes (Eqs. 12–15): compute queries `q_t`, apply a “big RoPE” to the merged key portion `c_K_t`, use `WUK_i` to route attention scores, and `WUV_i` to route values back to each head. KV cache size is unchanged at this stage; this step is algebraic refactoring that enables the next steps.

2) Decouple RoPE while preserving attention scores (RoRoPE)
- Key observation: RoPE uses the same rotation frequency pattern for the same pair of dimensions across all heads. Therefore, rotating the stacked (across-head) real and imaginary parts by the same orthogonal matrix `U_l` for each frequency `l` leaves all RoPE dot-products unchanged (Eq. 19; proof in Appendix B).
- Procedure (Figure 2; Section 4.2):
  - For each RoPE frequency pair `l`, collect the key outputs across heads (both real and imaginary parts).
  - Compute an orthogonal rotation `U_l` via PCA that concentrates the variance into the first few coordinates (the “first head positions” after rotation).
  - Apply the same `U_l` to both real and imaginary channels to preserve RoPE dot-products (constraint proven in Appendix B).
  - Keep RoPE only on the concentrated dimensions (call them `K_rope`); remove RoPE from the rest (call them `K_nope`) to make them compatible with MLA’s Absorb.

3) Increase positional capacity via frequency folding (FreqFold)
- Motivation: If RoPE information from all heads is forced into a single dimension per frequency, capacity can be insufficient. Adjacent RoPE frequencies are very similar; grouping them lets PCA find shared components in a higher-dimensional folded block (Appendix C; Figure 7).
- Mechanism: Concatenate the 2g-dimensional segments for multiple nearby frequencies and run a single PCA to obtain multiple principal components allocated to one head’s RoPE channel(s). Proposition 2 (Appendix C.2–C.3) formalizes why PCA on the concatenated block captures at least as much variance as running separate PCAs and then combining.

4) Balance `K_nope` and `V` before joint compression (BKV)
- Observation: After extracting `K_rope`, the norm of the remaining key features (`K_nope`) is still much larger than the value features `V`, so naïve joint PCA of `[K_nope; V]` ignores `V` (Section 4.3; Figure 4a).
- Fix: Scale `K_nope` by `1/α`, where `α = E[||K_nope||^2] / E[||V||^2]` (Eq. 20), to equalize magnitudes during PCA. Multiply the corresponding up-projection by `α` afterwards to keep the overall function unchanged (Appendix D.1).

5) Low-rank KV projection with joint PCA
- Concatenate the balanced activations `[K_nope'; V]` over a small calibration set and run PCA to learn a projection `R_KV` (Appendix D.2).
- Replace the original projections by a low-rank bottleneck:
  - Down: `WDKV' = R_KV^T WDKV` (Eq. 35) stores a compressed latent `c_KV` in the cache.
  - Up: `WUKV' = WUKV R_KV`, which decomposes into head-wise `WUK` and `WUV` in the MLA parametrization (Eqs. 36–37).
- Optional: compress queries similarly (Section 5.4 distinguishes “Low-rank Q” vs. “Full-rank Q” in speed plots).

6) Enable MLA Absorb inference mode
- With RoPE isolated to a small shared key vector and the content K/V routed through a low-rank latent, the layer supports the MLA Absorb form (Eqs. 9–10; Section 3.3):
  - Training-time: behaves MHA-like (per-head activations), ensuring optimization stability.
  - Inference-time: collapses to a shared latent key/value per token (MQA-like runtime), but per-head diversity re-emerges via learned up-projections on the fly.

7) Expressiveness guarantee (why this preserves capability)
- Appendix A proves the strict expressiveness ordering at equal KV budget: `GQA < MLA_factorized < MQA`. MLA with decoupled RoPE uses an MLA_factorized core for content and an MQA-style shared positional stream, hence is more expressive than GQA while using the same KV budget (Appendix A.3; Figure 6, mirrored in Figure 1a).

Analogy for RoRoPE/FreqFold
- Think of each RoPE frequency as a “note” played across many instrument tracks (heads). RoRoPE finds a rotation that mixes tracks so the loudest parts of each note move into the first track. FreqFold groups nearby notes into short chords and learns a joint mix so the first track can carry richer positional melody, letting other tracks drop the positional effect (NoPE) without losing the song.

## 4. Key Insights and Innovations
- Theoretical expressiveness advantage of MLA over GQA under equal KV budget
  - Novelty: A constructive mapping showing any GQA can be represented as MLA with one extra projection, while the reverse does not hold; rank-based arguments further separate MLA_factorized and MQA (Appendix A.2–A.3).
  - Significance: Justifies switching to MLA not only for speed, but for representation capacity at the same cache cost (Figure 1a; Appendix A Figure 6).

- RoRoPE: Rotation-invariant decoupling of RoPE across heads
  - What’s new: A provably invariant orthogonal rotation per RoPE frequency that concentrates positional content into chosen dimensions of one head, enabling RoPE removal from other heads without changing any attention scores before compression (Eq. 19; Appendix B).
  - Why it matters: Makes the Absorb trick possible on converted GQA checkpoints, which prior KV-compression methods couldn’t do efficiently due to RoPE entanglement (Section 4.2; Figure 2).

- FreqFold: Multi-frequency PCA for higher positional capacity
  - What’s new: Joint PCA over clusters of nearby RoPE frequencies, with a formal variance-preservation advantage (Proposition 2, Appendix C).
  - Why it matters: Retains more positional detail in `K_rope` while keeping most heads entirely RoPE-free. Empirically, 4D-FreqFold is a sweet spot for LLaMA 3 8B (Figure 3b).

- Balanced Key–Value (BKV) joint compression
  - What’s new: A simple, activation-based rescaling that equalizes key/value magnitudes before joint PCA, plus an algebraically exact inverse rescaling of the up-projection (Eq. 20; Appendix D).
  - Why it matters: Prevents the values from being washed out by higher-norm keys, reducing perplexity spikes under aggressive KV compression (Figure 4).

- Full compatibility with DeepSeek MLA and runtime ecosystems
  - What’s new: Converted checkpoints run directly on DeepSeek’s MLA code paths (vLLM, SGLang) and benefit from existing optimizations (Section 5.4).
  - Why it matters: Translates to real throughput gains across hardware without custom kernels (Figure 5; Table 4).

## 5. Experimental Analysis
- Evaluation setup
  - Models converted: `smolLM-1.7B` and `LLaMA-2-7B` (Section 5.1).
  - Benchmarks: 6 zero-shot multiple-choice tasks—MMLU, ARC (Easy/Challenge), PIQA, HellaSwag, OpenBookQA, Winogrande (Table 1).
  - Phases: before conversion (original), immediately after conversion (0 tokens), and after light pre-training/fine-tuning with 300M–6B tokens (Table 3; Appendix E).
  - Compression settings: KV cache reduced to 31.25%, 18.75%, 12.5%, and 7.03% of original (i.e., −68.75%, −81.25%, −87.5%, −92.97%) (Table 1 headings).
  - Inference throughput: vLLM across three GPUs (165.2 TFLOPS/24GB; 312 TFLOPS/40GB; 320 TFLOPS/64GB) with equal prefill/decoding lengths (Section 5.4; Table 4).

- Main quantitative findings
  - Training-free quality at moderate compression is strong:
    - LLaMA-2-7B (original avg = 59.85): after TransMLA at −68.75% KV, average = 58.20 (drop ≈ 1.65 points) (Table 1, “LLaMA-2-7B, TransMLA, 0 tokens, −68.75%”).
  - Extreme compression still coherent, recoverable with light training:
    - LLaMA-2-7B at −92.97% KV, 0 tokens: avg = 43.26, still “meaningful” outputs (Abstract; Table 1).
    - With 6B tokens: avg = 58.68, nearly back to the 59.85 original across the 6 tasks (Table 1, last block).
  - Outperforms concurrent conversion method (MHA2MLA) under the same budgets:
    - Example: LLaMA-2-7B, −68.75%, 0 tokens: TransMLA 58.20 vs. MHA2MLA 37.90 (Table 1).
    - smolLM-1.7B: at −68.75%, 0 tokens: TransMLA 51.95 vs. MHA2MLA 40.97; after modest training, TransMLA surpasses MHA2MLA trained on more tokens (Table 1).
  - Real speedups:
    - Up to 10.6x at 8K context on the 165.2 TFLOPS/24GB GPU with −92.97% KV (Figure 5a).
    - Table 4 shows raw throughput: at 8K context on the 312 TFLOPS/40GB GPU, LLaMA-2-7B = 218.51 tokens/s vs. TransMLA = 1118.18 tokens/s (≈5.12x).
    - Longer contexts increase gains; at 32K context, the original runs out of memory on smaller GPUs while TransMLA still delivers hundreds of tokens/s (Table 4).

- Ablations and diagnostics
  - RoRoPE and FreqFold:
    - Figure 3a: Norm concentration—after RoRoPE, key dimensions with large norms cluster into the first head; 4D-FreqFold amplifies this effect further, preparing for RoPE removal.
    - Figure 3b: During progressive RoPE removal, “RoRoPE + 4D-FreqFold” maintains far lower log-perplexity than MHA2MLA; at 90% removal, RoRoPE+4D ≈ 2 vs. MHA2MLA ≈ 6.
  - BKV:
    - Figure 4a: Before balancing, `K_nope` dominates `V`; after balancing, norms align.
    - Figure 4b: Across both weight-based and activation-based PCA, the balanced variants consistently reduce perplexity, with activation-based PCA best overall.

- Do the experiments support the claims?
  - Conversion fidelity: Yes, particularly at −68.75% KV with 0 tokens on LLaMA-2-7B (minimal average drop of ~1.65; Table 1), and strong recoverability at extreme compression with 6B tokens.
  - Speedups: Yes, demonstrated across hardware and contexts, including memory-limited regimes where the original OOMs (Figure 5; Table 4).
  - Mechanism value: Norm plots and removal curves convincingly show why RoRoPE, FreqFold, and BKV matter (Figures 3–4).

- Qualitative examples
  - Even the −92.97% model without additional training produces coherent text, and simple SFT further improves outputs (Appendix G, Table 5).

## 6. Limitations and Trade-offs
- Training-free is partial
  - At moderate compression (−68.75%), training-free performance is strong; at extreme compression (−92.97%), some light pretraining (up to 6B tokens) is needed to recover near-baseline performance (Table 1).
- Coverage of model families
  - Experiments are reported for smolLM-1.7B and LLaMA-2-7B. Although the method claims to convert many GQA models (LLaMA, Qwen, Gemma, Mistral/Mixtral), empirical validation beyond these two is not shown in this version (Section 6: “needs to be validated across a broader range of models”).
- FreqFold trade-off
  - Overly aggressive folding can harm accuracy; for LLaMA 3 8B, 4D-FreqFold is the “sweet spot,” but higher fold widths degrade performance (Figure 3b).
- Simplicity of BKV
  - BKV is a scalar norm-balancing heuristic; more advanced multi-objective or subspace-weighted methods might further improve joint PCA (Section 6).
- Benchmarks and tasks
  - Evaluations are on six common sense QA datasets; broader tasks (code, instruction-following, long-context retrieval quality) and human preference metrics are not included here.
- Implementation sensitivity
  - The pipeline relies on PCA computed from calibration data (WikiText-2 for some analyses; Appendix D.2), which introduces choices about sampling and stability across domains.

## 7. Implications and Future Directions
- Impact on the field
  - Provides a principled, reproducible path to migrate the vast ecosystem of GQA-based checkpoints into an MLA form that runs faster on commodity hardware. The expressiveness proof (Appendix A) reframes MLA not just as an efficiency trick but as a strictly stronger representation under the same cache budget.

- What this enables next
  - Compositions with other efficiency tools: quantization (including FP8), sparsity/pruning, token selection/pruning, and multi-token prediction—all directly on the converted MLA models (Abstract; Section 6).
  - Better balancing and compression: replace BKV with learned or optimization-based balancing, or use data-aware subspace weighting to push closer to training-free conversion at extreme compression (Section 6).
  - Broader validation: replicate on Qwen, Gemma, Mistral/Mixtral, LLaMA 3/4; test on reasoning-heavy and long-context tasks, including KV-sharing across layers or streaming workloads.

- Practical applications
  - Cloud and edge inference: serve long-context chat and RAG workloads on smaller GPUs where original models OOM, while sustaining 4–10x throughput gains (Figure 5; Table 4).
  - Cost-effective model distillation and hosting: quickly convert existing high-quality checkpoints to an MLA format, regain quality with a small number of tokens, and deploy on the DeepSeek-compatible stack.

> Headline result: “By compressing 93% of the KV cache in LLaMA-2-7B, TransMLA reaches a 10.6x speedup at 8K context while maintaining meaningful outputs, and needs only 6B tokens of fine-tuning to recover comparable performance across multiple benchmarks” (Abstract; Section 5.1; Figure 5a; Table 1).

In sum, TransMLA is both a theoretical argument and a practical recipe: it shows why MLA is a better target than GQA for a fixed KV budget, and it provides the rotations, folding, and balanced PCA needed to get there from any GQA checkpoint—unlocking immediate, hardware-agnostic speedups with minimal retraining.
