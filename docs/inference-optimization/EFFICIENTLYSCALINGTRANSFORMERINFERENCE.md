# EFFICIENTLY SCALING TRANSFORMER INFERENCE

**ArXiv:** [2211.05102](https://arxiv.org/abs/2211.05102)

## 🎯 Pitch

This paper presents a principled, hardware-aware framework for scaling generative inference in very large Transformer models, introducing analytical models and optimal partitioning strategies that maximize efficiency and minimize latency on distributed accelerators like Google TPU v4. By combining multi-dimensional tensor partitioning, advanced memory management, and specific attention mechanisms, the authors set a new Pareto frontier—demonstrating fast, cost-effective inference for models exceeding 500 billion parameters, an achievement crucial for deploying large language models in both latency-sensitive and high-throughput production environments.

---

## 1. Executive Summary
This paper develops a principled, hardware-aware method for running very large Transformer models fast and cheaply at inference time by choosing the right way to split (“partition”) model tensors across many accelerator chips and by optimizing memory traffic. It delivers an analytical model and a set of concrete partitioning layouts that together set a new latency–efficiency Pareto frontier on Google TPU v4 for 500B+ parameter models, achieving, for PaLM 540B, 29 ms/token decode latency with int8 weights and up to 76% model FLOPS utilization (MFU) in large-batch prefill (Abstract; Figure 1; Tables 2–3).

## 2. Context and Motivation
- Problem addressed
  - Generative inference for very large Transformers must process one token at a time (a “decode step”), so it has much less parallelism than training and is therefore latency- and bandwidth-sensitive (Section 1).
  - Large models do not fit on a single chip; they require multi-chip partitioning and careful handling of the persistent attention key/value (“KV cache”) whose size grows with batch and context length (Section 2; Section 3.3).
- Why it matters
  - Real applications span interactive chat (tight latency) and offline processing (high throughput/low cost). Getting both to work at 100B–500B scale determines the practical utility of LLMs in production (Section 1; Section 2.1).
- Prior approaches and gaps
  - Training-time parallelism systems (e.g., Megatron, GSPMD, Alpa) provide general sharding but don’t explain which layout is best for inference’s unique bottlenecks or for different phases (prefill vs. decode) (Related Work, Section 6).
  - Existing inference suites such as FasterTransformer rely on limited forms of tensor and pipeline parallelism and hit communication bottlenecks as tensor-parallel degree grows (Figure 9; Section 5).
  - Multiquery attention (MQA) reduces KV cache size but loses its memory benefits if sharded over heads in the obvious way, because K/V must be replicated across devices (Section 3.3; Figure 4b).
- Positioning of this work
  - Provides an analytical framework that predicts communication time for different layouts and shows how to select the best one as a function of model size, batch size, sequence length, and chip count (Sections 2–3; Appendix A).
  - Introduces an attention partitioning for MQA that shards across batch during decode to realize KV-cache savings (Section 3.3; Figures 4c, 5b).
  - Combines these with low-level scheduling/communication overlap and int8 weight quantization (Sections 3.5–3.6).

Key terms used throughout (defined once):
- `prefill`: the initial pass over the full input context where the model processes B×Linput tokens in parallel (Section 2.2).
- `decode`: the autoregressive phase that generates tokens one step at a time; each step depends on previous outputs (Section 2.2).
- `KV cache`: per-layer tensors that store past attention keys and values for every sequence; needed to attend to prior tokens without recomputing (Section 2).
- `MFU (Model FLOPS Utilization)`: observed throughput divided by the theoretical peak FLOPS of the hardware configuration; higher MFU means better hardware efficiency (Section 2).
- `reduce-scatter` / `all-gather` / `all-reduce` / `all-to-all`: standard multi-chip collectives to sum, shard, or reshuffle tensors across devices (Section 3.1; Figure A.1).
- `weight-stationary` vs. `weight-gathered` layouts: strategies that either keep weights fixed on each chip and move activations, or keep activations fixed and move weights (Sections 3.2.1–3.2.3).

## 3. Technical Approach
The paper builds an end-to-end system by aligning high-level partitioning choices with the actual sources of time: compute, memory traffic (weights + KV cache), and communication.

A) Cost model and phases (Section 2)
- Latency has two parts:
  - Prefill: parallel over the entire input sequence (good parallelism).
  - Decode: a loop of Lgen steps; each step is a full forward pass on the newest token (poor parallelism).
- Compute cost: a decoder-only model with N parameters performs about 2N floating-point operations per token (Kaplan scaling; Section 2).
- Memory cost: weights and KV cache must be loaded from on-device HBM to compute cores each forward pass; at large batches/long contexts, KV cache dominates memory time (Section 2; 2.1).
- Communication cost: depends on which tensor dimensions are sharded and which collectives are required (Section 3.1; Appendix A.1).
- Objective: pick partitioning to minimize total latency (or maximize MFU) subject to hardware limits and application goals.

B) Partitioning the feedforward (FFN) layers (Section 3.2)
The FFN dominates FLOPs, so its sharding determines most of the compute/communication.

1) 1D weight‑stationary (Section 3.2.1; Figure 2a)
- What it does: shard each E×F weight by one dimension (typically F, the intermediate dimension), keep the shard local, all‑gather inputs and reduce‑scatter outputs between the two matmuls in the FFN “block”.
- Communication behavior: per forward pass, activations of shape `BLE` are aggregated; communication time scales roughly as Tcomm ≈ 2BLE / (network bandwidth), independent of chip count nchips (Section 3.2.1; Appendix A.1).
- Limitation: as nchips grows, memory and compute scale down, but this constant communication term becomes the bottleneck.

2) 2D weight‑stationary (Section 3.2.2; Figure 2b)
- What it does: shard weights along both model dimension E and FFN dimension F so each device holds a smaller, closer-to-square chunk; alternate the aggregation axis across the two FFN matmuls so no chip ever needs the full activation (mechanically, one reduce-scatter/all-gather happens along E, the other along F).
- Key result: with optimal split X along E and Y×Z along F, communication time scales as
  - Tcomm = 8BLE / (√nchips × network bandwidth), minimized by choosing X = 0.5√nchips and YZ = 2√nchips when F ≈ 4E (Appendix A.2.1).
- Why it matters: unlike 1D, communication decreases with more chips (∝ 1/√nchips), so latency keeps dropping as we scale up.

3) Weight‑gathered layouts (Section 3.2.3; Figure 2c; Figure A.2)
- When to use: at very large tokens-per-batch BL (e.g., prefill with many sequences), the activation outputs become larger than the weights, so moving weights can be cheaper than moving activations.
- How it works:
  - Keep activations stationary on each chip; all‑gather weights over N chips just-in-time for the two FFN matmuls.
  - Three variants differ in how widely weights are gathered: X‑only, XY, or XYZ (full) gathering; activations are partitioned to match (Figure A.2).
- Optimal choice and cost:
  - Choose N (the number of chips to all‑gather weights over) to balance weight vs. activation traffic: N = √(BL·nchips / F).
  - Communication time becomes Tcomm = 4E·√(BLF) / (√nchips × network bandwidth) (Appendix A.2.2).
- Design choice: store weights on device in the same ExFyz layout as 2D weight‑stationary so the system can switch layouts between prefill (often weight‑gathered) and decode (weight‑stationary) without re-sharding the parameters (Section 3.2.3).

C) Partitioning the attention layer with multiquery attention (MQA) (Section 3.3; Figures 4–5)
- Background: MQA emits multiple query heads but shares a single key and a single value head across all queries; this reduces KV cache size by a factor of nheads (Section 3.3).
- Pitfall: If we shard attention over heads (as is typical for multi-head attention), MQA’s single K/V head must be fully replicated on each chip, erasing the memory benefit (Figure 4b).
- Proposed layout for decode (Figures 4c and 5b):
  - Shard by batch (B) across devices so each chip only loads and uses the slice of the KV cache for its subset of sequences.
  - Pay a small extra all‑to‑all to reshuffle the much smaller Q/K/V inputs and outputs.
  - Why it works: during decode, KV cache (many past tokens) is orders of magnitude larger than per-step Q/K/V (one token per sequence), so trading a small communication on Q/K/V for a large reduction in KV memory loads is beneficial (Section 3.3).
- Prefill exception: during prefill, Q has many tokens and reuses the same K/V; the amortized KV load is not the bottleneck, so the head-sharded layout remains preferable (Section 3.3).

D) Block structure: parallel attention/FFN layers (Section 3.4)
- Use the “parallel” Transformer block (as in PaLM) where attention and FFN start from the same layer-normed input and are fused:
  - Benefits: one layer norm instead of two; larger fused matmuls; one fewer all‑reduce per layer along the E/F axis (Section 3.4).

E) Low-level and numerical optimizations (Section 3.5–3.6)
- Looped CollectiveEinsum: schedule reduce‑scatter/all‑gather overlapping with the corresponding matmuls, improving end-to-end performance by ~1.4× vs. a naive compiler schedule (Section 3.5).
- Prefer reduce‑scatter into hidden dims (E/F) to expose more overlap opportunities (Section 3.5).
- Miscellaneous kernels: faster sampling, softmax/swish, incremental prefill, better in-memory layouts (Section 3.5).
- Quantization: convert bfloat16 weights to int8 using AQT, reducing weight loading time (especially impactful at small batch sizes); matmuls still use bfloat16 activations (Section 3.6).

F) Implementation and hardware (Section 4; Section 4.4; Section 5)
- Framework: JAX + XLA; derived from T5X codebase (Section 4).
- Hardware: TPU v4 (275 TFLOPS bfloat16, 32 GiB HBM @1200 GB/s, 270 GB/s interconnect in 3D torus) (Section 4).
- Practical tweak: pad PaLM 540B heads from 48 to 64 to enable cleaner partitioning on 64+ chips (adds ~3% parameter overhead, recovered by better partitioning) (Section 4).

## 4. Key Insights and Innovations
1) Closed-form, phase-aware partitioning strategy for FFN (Sections 3.2 and A.2)
   - Novelty: explicit formulas for communication time for 1D vs. 2D weight‑stationary and multiple weight‑gathered variants, with the optimal sharding splits derived analytically.
   - Significance: enables principled selection of layout as batch size (BL), model dims (E, F), and chip count vary. In practice this flips prefill from weight‑stationary to weight‑gathered at large BL, while decode stays 2D weight‑stationary (Figure 7).

2) Batch‑sharded multiquery attention for decode (Section 3.3; Figures 4–5)
   - Novelty: for MQA, shard attention over batch (not heads) during decode so each device loads only its KV-cache slice; use all‑to‑all to reshuffle small Q/K/V tensors.
   - Significance:
     - Dramatically lowers memory time for long contexts and large batches, enabling much longer contexts: up to 32–64× longer maximum context length than head-sharded variants on 64 chips (Table 1).
     - Yields growing latency wins as context length increases (Figure 8).

3) Use of parallel attention/FFN with collective-fused scheduling (Sections 3.4–3.5)
   - Novelty: systematically align block-level fusion (parallel block) with collective scheduling (Looped CollectiveEinsum) and reduction axes to hide communication.
   - Significance: ~1.4× speedup vs. naive schedules (Section 3.5), and fewer all‑reduces per layer (Section 3.4).

4) End-to-end Pareto frontier at 500B scale with int8 weights (Figure 1; Tables 2–3)
   - Innovation is integrative: combining the theoretical partitioning choices, MQA batch sharding, fusion, overlap, and int8 weights.
   - Significance:
     - Low-latency interactive setup with PaLM 540B: 29 ms/token decode (int8 weights) and the ability to process 64-input + 64-output tokens with a 1920‑token history in 1.9 s total on 64 chips (Figure 1; Section 4).
     - Large-batch prefill MFU of 76% (Figure 7; Table 2).

## 5. Experimental Analysis
Evaluation setup
- Models: PaLM family (8B, 62B, 540B) using parallel block and multiquery attention (Section 4); a Megatron-like model is also evaluated for cross-suite comparison (Appendix D, Table D.1).
- Hardware: up to 256 TPU v4 chips; most headline results on 64 chips (Section 4).
- Metrics:
  - Latency (prefill per forward pass; decode per generated token).
  - Cost measured as chip‑seconds per token = nchips × time / (B×L), directly proportional to dollars; inversely proportional to MFU (Section 4.4).
  - MFU to normalize across different hardware when compared to FasterTransformer on NVIDIA A100 (Section 5).
- Workloads:
  - Context length 2048 for main Pareto plots (Figure 1).
  - Prefill vs. decode analyzed separately due to distinct parallelism/memory patterns (Sections 2.2, 3.2–3.3).

Main quantitative results
- Overall Pareto frontiers (Figure 1)
  - Decode: “minimum latency is ~3× lower than batch‑512 latency,” showing the latency–cost trade-off as batch decreases (Figure 1, left).
  - Prefill: high MFU and low cost even at moderate batch sizes due to weight‑gathered layouts; batch‑512 prefill cost is ~2× lower than batch‑512 decode (Figure 1, right; Section 4.4).
- Concrete PaLM 540B configurations (Table 2)
  - Low-latency setting on 64 chips (int8 weights):
    - Prefill 2048 tokens at batch 1 with 2D weight‑stationary FFN: MFU 43%, 0.29 s.
    - Decode 64 tokens at batch 64 with 2D weight‑stationary FFN and batch‑sharded attention: MFU 14%, 1.82 s.
  - High-throughput setting on 64 chips (bfloat16 weights):
    - Prefill 2048 tokens at batch 512 using XYZ weight‑gathered FFN: MFU 76%, 85.2 s for the entire batch.
    - Decode 64 tokens at batch 512: MFU 33%, 6.0 s for the entire batch (Table 2).
- FFN partitioning behavior
  - 2D vs. 1D weight‑stationary during decode: latency per token improves with chip count for 2D, while 1D flattens due to communication bottlenecks (Figure 6).
  - Prefill layout switch: as tokens per batch increase, MFU transitions from 2D weight‑stationary to weight‑gathered, peaking at 76% MFU at very large batches (Figure 7).
- MQA partitioning and long-context capability
  - Max context length on 64 chips with 30% memory reserved for KV cache (Table 1):
    - Multihead (dhead=128): 1320 (B=128) and 330 (B=512).
    - Baseline MQA, head-sharded (dhead=256): 660 and 165.
    - Optimized MQA, batch-sharded: 43,000 and 10,700.
  - Latency vs. context length (decode, 8-layer proxy): batch-sharded MQA increasingly outperforms as context grows; at long contexts, attention becomes only 8–31% of runtime (Figure 8; Section 4.2).
- Comparison to FasterTransformer (Figure 9; Appendix D)
  - MFU vs. latency for a 60‑input, 20‑output benchmark: PaLM implementation on 64 TPU v4 achieves the best absolute latency and higher MFU at most points; notably, at 64‑way tensor parallelism it maintains ~44% MFU, whereas FasterTransformer’s 32‑way setup peaks near 33% MFU and degrades at 32‑way vs. 16‑way (Figure 9; Section 5).
  - Full numeric tables for several input/output settings show the same MFU–latency pattern (Tables D.2–D.4).

Do the experiments support the claims?
- The phase-aware layout selection is directly corroborated by MFU and latency trends (Figures 6–7).
- The MQA batch-sharding design both increases max context length by up to 32–64× (Table 1) and lowers decode step time at long contexts (Figure 8).
- The end-to-end Pareto plots and example configs quantify low-latency and high-throughput operating points on a 540B model (Figure 1; Table 2).
- Cross-suite comparisons normalize differences via MFU and show scalability advantages at higher parallel degrees (Figure 9; Section 5). The paper notes hardware differences explicitly.

Ablations and robustness
- Ablations are analytical + empirical:
  - 1D vs. 2D vs. weight‑gathered FFN (Figures 6–7).
  - Head‑sharded vs. batch‑sharded MQA (Figures 4–5, 8; Table 1).
  - Parallel vs. serial block: serial increases decode latency by ~14% at batch 512 on 64 chips (Section 4.3).
- Failure modes:
  - Without batch‑sharded MQA, memory limits prevent long contexts at 540B scale (Figure 8, dotted line; Table 1).
  - Some weight‑gathered layouts exhaust memory unless communication/computation overlap is carefully implemented (Section 3.5).

## 6. Limitations and Trade-offs
- Hardware specificity vs. generality
  - The derivations assume access to fast collective primitives and a high-bandwidth, low-diameter interconnect (3D torus on TPU v4). While the formulas for collective costs are general (Appendix A.1), absolute numbers and optimal splits may differ on other topologies (Section 7 notes generalization, but no direct GPU result beyond MFU comparisons).
- Latency vs. cost
  - Extremely low latency requires many chips and small batch sizes, which reduces MFU and raises cost per token (Figure 1 left; Section 2.1).
- Activation quantization not used
  - Only weights are int8. Large-batch scenarios remain compute‑bound with bfloat16 matmuls; activation quantization could further reduce cost but is not implemented (Section 3.6; Section 4.4).
- Scope: dense, decoder-only models
  - The analysis targets dense models; mixture-of-experts or encoder–decoder specifics are not explored (Section 7 suggests this as future direction).
- Operational complexity
  - Switching layouts between prefill and decode, and using different attention sharding per phase, adds system complexity and requires careful orchestration (Sections 3.2–3.3; 4.1).
- Quality considerations for MQA
  - The work assumes MQA (as in PaLM) without re-evaluating quality trade-offs vs. multihead in this paper; results focus on efficiency and memory (Section 4; Table 1 discusses capacity limits, not task accuracy).

## 7. Implications and Future Directions
- Practical guidance for serving very large LLMs
  - Use 2D weight‑stationary for decode on many chips to avoid a communication plateau (Figure 6).
  - For prefill at large BL, switch to weight‑gathered (often XYZ) to push MFU into the 70%+ range (Figure 7).
  - With MQA models, shard attention by batch during decode to unlock long contexts and lower memory time (Figures 4–5; Table 1; Figure 8).
  - Prefer parallel attention/FFN blocks and fuse collectives with matmuls to hide communication (Sections 3.4–3.5).
  - Apply int8 weight quantization for low-latency regimes (Section 3.6; Figure 1 left).
- Impact on the field
  - Shifts inference design from “use as much tensor parallelism as possible” to “choose phase- and tensor-dimension-aware partitioning with provable communication scaling,” enabling 500B‑class models to meet both interactive and offline needs on multi-chip clusters.
- Research directions
  - Activation quantization and communication compression to further reduce cost at large batch sizes (Section 3.6; Section 7).
  - Extending the analytical framework to sparse/MoE models and adaptive computation to reduce FLOPs per token (Section 7).
  - Porting the batch‑sharded MQA approach and 2D layouts to different interconnects (multi-node NVLink, Ethernet fabrics) and validating optimal splits under those bandwidth/latency profiles (Section 7).
- Downstream applications
  - Long-context interactive systems (summarization, chat with long histories) become feasible at 500B scale due to the 32×–64× context length expansion (Table 1).
  - High-throughput offline scoring/distillation pipelines benefit from weight‑gathered prefill and high MFU (Table 2; Figure 1 right).

Selected, grounded highlights
- “29 ms per token during generation (using int8 weight quantization) … on the PaLM 540B model” (Abstract; Figure 1 left; Table 2 decode).
- “76% MFU during large-batch-size processing of input tokens [prefill]” (Abstract; Figure 7; Table 2 prefill).
- “Optimized multiquery attention enables up to 32× larger context lengths” (Table 1; Section 4.2).
- “2D weight‑stationary communication scales as 1/√nchips” with optimal X,Y,Z (Appendix A.2.1).
- “Weight‑gathered communication scales with √(BL)” and becomes preferable at large tokens per batch (Appendix A.2.2; Figure 3).
- “Our 64‑way tensor parallelism maintains ~44% MFU, whereas FasterTransformer’s 32‑way peaks near 33% MFU” (Figure 9; Section 5).
