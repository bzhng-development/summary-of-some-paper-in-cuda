# EFFICIENTLY SCALING TRANSFORMER INFERENCE

**ArXiv:** [2211.05102](https://arxiv.org/abs/2211.05102)

## 🎯 Pitch

This paper introduces a principled analytical framework and a suite of engineering techniques for scaling the inference of massive Transformer models (500B+ parameters) across many accelerator chips, achieving record-setting latency and throughput on TPU v4. By deriving optimal partitioning strategies for both feedforward and attention layers and overlapping communication with computation, the authors demonstrate new Pareto frontiers—like enabling a 540B-parameter PaLM model to generate tokens at just 29 ms/token and up to 76% FLOPs efficiency. These innovations make practical, cost-effective inference for ultra-large language models achievable, a critical advancement for real-world applications demanding both high speed and scalability.

---

## 1. Executive Summary
This paper presents a principled way to run very large Transformer language models fast and cheaply at inference time by choosing the right way to split computation and tensors across many accelerator chips. It introduces an analytical model and several partitioning strategies (for both feedforward and attention layers), plus low‑level communication/computation overlap, that together set new latency–throughput Pareto frontiers on 500B+ models on TPU v4. With these techniques, PaLM 540B achieves 29 ms/token generation latency at batch 64 with int8 weights and up to 76% Model FLOPs Utilization (MFU) in large‑batch prefill (Figure 1, Tables 2–3).

## 2. Context and Motivation
- Specific problem and gap
  - Generative inference for very large Transformers (100B–500B+ parameters) is hard because:
    - Models and their per‑sequence attention state do not fit on one chip; they must be split across many chips, which introduces chip‑to‑chip communication (Section 2).
    - Autoregressive generation proceeds one token at a time; each step depends on the previous tokens, limiting parallelism and making latency targets tight (Introduction; Section 2.1).
    - The attention `KV cache` (stored keys/values for past tokens) grows with batch size and context length and must be reloaded every decode step, causing memory‑bandwidth bottlenecks (Sections 2 and 2.1).
  - Prior work offers training‑time parallelism (e.g., Megatron, GSPMD, Alpa) and inference frameworks (e.g., FasterTransformer, DeepSpeed Inference), but either:
    - Depend on a fixed set of parallelism modes or black‑box search over options (Section 1), or
    - Do not provide a simple analytical way to choose the best partitioning for given hardware, batch size, and latency goals.

- Why it matters
  - Real‑world deployments (chatbots, code assistants, ranking, distillation) need either very low latency for small batches or very high throughput for large batches. The paper targets both with a single framework (Section 2 and Figure 1).
  - Memory and communication dominate cost at scale; getting the partitioning “right” can reduce dollars per token and make long‑context inference feasible (Sections 2.1 and 3.3).

- Positioning relative to existing work
  - Builds a compact analytical model to reason about communication and memory time and uses it to derive when each partitioning strategy is optimal (Sections 3.2 and Appendix A).
  - Augments model‑level ideas (PaLM’s parallel blocks and multiquery attention) with partitioning and kernel scheduling tailored for inference (Sections 3.3–3.5).
  - Demonstrates superior MFU vs. latency trade‑offs versus FasterTransformer on 500B‑class models, especially at high degrees of tensor parallelism (Figure 9; Appendix D).

## 3. Technical Approach
Key terms used throughout (defined when first introduced):
- `prefill`: the initial forward pass over the B×L_input input tokens that exist before decoding; it can run fully in parallel (Section 2.2).
- `decode`: the autoregressive generation phase; L_gen steps run sequentially, one new token per step per sequence (Section 2.2).
- `KV cache`: per‑layer tensors of past keys and values that must be kept for attention; dominates memory traffic during decode (Introduction; Section 2).
- `MFU (Model FLOPs Utilization)`: measured throughput divided by the hardware’s peak FLOPs; higher MFU means better hardware utilization (Section 2).
- `all-gather`, `reduce-scatter`, `all-reduce`, `all-to-all`: standard collective communication primitives; Appendix A.1 details their costs.

The paper’s methodology consists of three layers:
1) an analytical communication model; 2) partitioning designs for feedforward and attention; 3) low‑level scheduling to overlap communication with compute.

A) Analytical communication model (Appendix A.1)
- For an all‑gather over K chips, each chip sends/receives D/K bytes to (K–1) peers. The communication time simplifies to approximately D / (network_bandwidth), ignoring the factor (K–1)/K for large K.
- Reduce‑scatter has the same form but with D equal to the input size; an all‑reduce is roughly two such phases.
- Using this, the paper derives closed‑form communication times for different partitionings and then picks parameters (how many shards along each axis) that minimize them.

B) Feedforward (MLP) partitioning strategies (Section 3.2)
Let `E = d_model`, `F = d_ff`, `B` = batch size, `L` = tokens processed this pass.

1) 1D weight‑stationary (Section 3.2.1; Figure 2a)
- Idea: shard each `E×F` weight matrix along one dimension (often F) across `nchips` and keep weights fixed (“stationary”) on each chip. Activations move between chips using one all‑gather and one reduce‑scatter per MLP.
- Communication time per layer is:
  - T_comm ≈ 2·B·L·E / network_bandwidth (Equation in Section 3.2.1).
- Pro: simple; works well for few chips.
- Con: communication time does not decrease with more chips, so it becomes the bottleneck at scale.

2) 2D weight‑stationary (Section 3.2.2; Figure 2b)
- Idea: shard weights along both E and F so each chip holds a roughly square block; alternate which axis you aggregate activations over across the two matmuls in the MLP.
- With `d_ff = 4E`, optimizing the number of shards per axis yields (Appendix A.2.1):
  - Choose X = 0.5·√nchips partitions along E and YZ = 2·√nchips along F.
  - Communication time per layer becomes:
    - T_comm ≈ 8·B·L·E / (√nchips · network_bandwidth)
- Why better: communication scales down as 1/√nchips, so adding chips keeps cutting latency even when comm‑bound (Figure 6 shows it outperforms 1D at 64 chips).

3) Weight‑gathered variants (Section 3.2.3; Figure 2c and Figure A.2)
- Motivation: When `B·L` is very large (e.g., prefill with long inputs or big batches), activations are much larger than weights. It can be cheaper to keep activations stationary and move weights.
- Mechanism: Before the two MLP matmuls, all‑gather weights over N chips (N can be X, X·Y, or X·Y·Z), then compute locally; this removes one activation collective and shrinks the other.
- Optimal N (Appendix A.2.2): N ≈ √(B·L·nchips / F)
- Resulting communication time per layer:
  - T_comm ≈ 4·E·√(B·L·F) / (√nchips · network_bandwidth)
- Transition: Communication for weight‑stationary grows ∝ B·L; for weight‑gathered it grows ∝ √(B·L). Thus, beyond a certain `B·L`, weight‑gathered is cheaper (Figure 3). Figure 7 empirically shows MFU jumping as prefill switches to weight‑gathered at large batches.

C) Attention partitioning with multiquery attention (Section 3.3; Figures 4–5)
- Background: Multihead attention stores a separate KV pair per head. `Multiquery attention (MQA)` shares a single K and V across all heads, reducing KV cache size by `n_heads`.
- Naïve MQA layout (shard by heads) replicates the single K/V on each chip, losing the memory saving (Figure 4b).
- Proposed decode‑time layout: shard attention by `batch` instead of `heads` (Figure 4c and Figure 5b).
  - Each chip holds a different slice of sequences’ KV cache; cost to load KV per chip drops by `nchips` (memory‑time win).
  - Requires an `all-to-all` to reshuffle small Q/K/V projections so that compute lines up with KV sharding; but this communication is cheap compared to KV memory loads during decode.
- Prefill exception: During prefill, Q has many tokens and each is matched against the same K/V; amortization makes KV loads less dominant. The method therefore shards by heads during prefill and by batch during decode (Section 3.3).
- Payoff: Enables much longer contexts and faster decode (Table 1; Figure 8).

D) Parallel attention/FFN blocks (Section 3.4)
- PaLM uses a “parallel” Transformer block: one layernorm feeds both MLP and attention, computed in parallel, then summed.
- Benefits:
  - One fewer layernorm per layer (lower latency at small batch).
  - Fusion opportunities: input projections (`W_Q` with MLP’s input matrix) and output projections (`W_O` with MLP’s output) can be combined into larger matmuls, improving MFU.
  - Eliminates one of the two all‑reduces per layer (for the `d_ff/n_heads` axis), halving that communication.

E) Low‑level scheduling and kernel work (Section 3.5)
- Uses a `Looped CollectiveEinsum` pattern to overlap collectives with matmuls; reduces end‑to‑end latency by about 1.4× compared to a naïve compiler‑scheduled baseline.
- Carefully chooses which dimensions to reduce‑scatter into (often `E`/`F` rather than `B`/`L`) to expose more overlap.
- Plus memory‑layout tuning, fast sampling/top‑k, faster softmax/activation, and incremental prefill support.

F) Quantization (Section 3.6)
- Converts bfloat16 weights to int8 using AQT for memory‑time savings; no activation quantization. This helps especially at small batches where weight loads dominate latency.

G) Experimental setup (Section 4)
- Hardware: up to 256 TPU v4 chips; each has 32 GiB HBM (1200 GB/s), 270 GB/s interconnect in a 3D torus, and 275 TFLOPs bfloat16 (Section 4).
- Models: PaLM 8B, 62B, 540B. For 540B, heads are padded from 48 to 64 to partition better on 64+ chips (3% MFU loss recouped by better sharding).
- Metrics: latency (total; and separately for prefill and decode), throughput, and MFU (Section 2).
- Typical context length 2048 for main Pareto plots (Figure 1). Additional long‑context experiments in Figure 8 and Table 1.

## 4. Key Insights and Innovations
1) Communication‑aware choice between 2D weight‑stationary and weight‑gathered MLPs
   - What’s new: A compact analytical model plus closed‑form optimums for how many shards to use along each axis (Appendix A.2.1–A.2.2).
   - Why it matters: It clarifies when to switch strategies:
     - For decode (small `B·L`), 2D weight‑stationary minimizes latency and scales with chip count (Figure 6).
     - For prefill (large `B·L`), weight‑gathered wins and reaches 76% MFU on PaLM 540B (Figure 7).
   - Difference from prior work: Megatron‑style 1D tensor parallelism (Section 3.2.1) stops scaling due to constant communication overhead; the 2D/weight‑gathered mix preserves scaling and utilization.

2) Batch‑sharded multiquery attention for decode
   - What’s new: In decode, shard attention by batch so each chip owns a slice of the KV cache, avoiding K/V replication (Figures 4c and 5b).
   - Why it matters:
     - Dramatically lowers memory time during decode (the dominant cost at long contexts), letting the system trade a small all‑to‑all on tiny Q/K/V for a big KV load reduction.
     - Enables much longer contexts with fixed memory: Table 1 shows the maximum context length on 64 chips with 30% memory reserved for KV cache increases up to 32×–64× compared to multihead/baseline MQA (e.g., batch 512: 165 → 10,700 tokens).
   - Prior work typically shards attention over heads; that replicates the single MQA KV head and forfeits the memory advantage (Figure 4b).

3) Parallel block structure and projection fusion reduce per‑layer communication and increase effective matmul size
   - Significance: Fewer collectives and more efficient matmuls raise MFU and cut latency. In a head‑to‑head comparison, serial blocks raise decode latency by 14% (Section 4.3).

4) Communication/compute overlap via Looped CollectiveEinsum
   - Significance: Achieves ≈1.4× speed‑up over a straightforward compiler schedule (Section 3.5), which is crucial when communication would otherwise dominate.

Overall, the combination is more than incremental: it provides a general recipe (with formulas) to pick the right partitioning for any given model/hardware/latency target and backs it with system‑level implementation details that realize the predicted gains.

## 5. Experimental Analysis
- Evaluation methodology
  - Workloads: Inference passes (prefill and decode) on PaLM 8B, 62B, 540B. Context length typically 2048 for main Pareto curves (Figure 1). Separate long‑context study up to tens of thousands of tokens (Table 1; Figure 8).
  - Hardware: TPU v4 slices with 3D torus interconnect. Chip counts vary; main results often use 64 chips (Section 4).
  - Metrics: latency (per forward pass and per generated token), cost in chip‑milliseconds per token, and MFU (Section 2). Prefill and decode are reported separately.

- Main quantitative findings
  - New Pareto frontier for latency vs. cost and MFU:
    - Decode (Figure 1, left): On PaLM 540B with int8 weights at batch 64 and 64 chips, generation latency reaches
      > “29 ms per token” (Introduction; reiterated around Figure 1 and Table 2).
      At similar latency, bfloat16 weights are ≈36.9 ms/token (Section 4.4).
    - Prefill (Figure 1, right): With weight‑gathered layouts, MFU reaches
      > “76% during large‑batch prefill” on PaLM 540B (Introduction; Table 2 uses batch 512 prefill).
  - Decode partitioning: 2D weight‑stationary scales better than 1D as chip count grows (Figure 6). At 64 chips on 540B, the 2D layout shows markedly lower per‑token latency.
  - Prefill partitioning: As tokens per batch grow, MFU jumps when switching from 2D weight‑stationary to weight‑gathered MLPs (Figure 7). With L=2048 and 64 chips, MFU climbs to 76% at ≈10^6 tokens (batch 512 × 2048).
  - Multiquery attention layout:
    - Maximum context length (Table 1, 64 chips, 30% memory for KV):
      - Multihead: 1320 (B=128), 330 (B=512)
      - Baseline MQA (sharded by heads): 660 (B=128), 165 (B=512)
      - Optimized MQA (sharded by batch for decode): 43,000 (B=128), 10,700 (B=512)
    - Decode latency vs. context (Figure 8, 8‑layer subset, B=256): optimized MQA maintains lower latency than both multihead and baseline MQA as context grows; attention takes only 8–31% of runtime even at 8k–32k tokens (Section 4.2).
  - End‑to‑end configurations (Tables 2–3):
    - PaLM 540B low‑latency: prefill B=1 in 0.29 s, decode B=64 in 1.82 s (total for 2048 input + 64 output tokens ≈ 2.11 s, matching the “interactive chatbot” scenario in Section 4.4).
    - PaLM 540B high‑throughput: prefill B=512 in 85.2 s, decode B=512 in 6.0 s with MFU 76% and 33% respectively (Table 2).
    - Similar patterns hold for PaLM 62B (Table 3), with proportionally lower latency due to model size.
  - Comparison to FasterTransformer (Section 5; Figure 9; Appendix D):
    - Across the 60‑input/20‑output benchmark at many batches, the implementation achieves higher MFU at equal or lower latency. For example, at moderate latencies it reaches ≈44–46% MFU at 64‑way tensor parallelism, while FasterTransformer’s TP32 tops out at ≈33% MFU (Figure 9).
    - Detailed tables (D.2–D.4) list latency and MFU per batch for the standard 20/8, 60/20, and 128/8 token setups.

- Do the experiments support the claims?
  - Yes, for system‑level performance claims. The predicted transitions (1D→2D→weight‑gathered as B·L grows) appear in Figures 6–7. The multiquery layout’s memory advantage is quantified by the large increases in max context length (Table 1) and reduced attention share of runtime at long contexts (Figure 8).
  - The FasterTransformer comparison normalizes by MFU to control for differing hardware; while not identical hardware, this is a fairer comparison than raw throughput (Section 5).

- Ablations / robustness
  - The paper provides analytical derivations (Appendix A) and then validates the predicted scaling with chip count and batch size (Figures 6–7).
  - It also isolates the effect of the parallel vs. serial block (Section 4.3) and weight quantization (Section 4.4: int8 reduces decode latency from 36.9 to 28.5 ms/token at B=64).
  - Failure/edge cases: memory exhaustion with multihead/baseline MQA at long contexts is highlighted (Figure 8 dotted line; Table 1).

- Conditions and trade‑offs
  - Decode vs. prefill differ: decode is sequential and memory‑bandwidth‑dominated by KV cache; prefill is parallel and often compute‑ or activation‑communication‑dominated. The system switches partitioning between phases (Section 4.1).
  - Int8 weights help small‑batch latency (weight load bound) more than large‑batch throughput (compute bound), so cost benefits are bigger at low latency (Section 4.4).

## 6. Limitations and Trade-offs
- Hardware specificity vs. generality
  - Derivations use a 3D torus topology and TPU v4 bandwidths (Sections 3.1 and 4). The qualitative conclusions generalize, but the exact optimal shard counts (e.g., X=0.5√nchips) assume `d_ff ≈ 4E` and accessible torus shapes (Appendix A.2.1).
- Model assumptions
  - The largest wins in attention rely on `multiquery attention`. Models trained without MQA would not benefit from the batch‑sharded decode layout, and multihead attention remains memory‑limited at long contexts (Section 4.2; Table 1).
- Quantization scope
  - Only weights are quantized to int8; activations remain bfloat16. The paper suggests activation quantization could further reduce communication and compute, but it is not implemented (Section 3.6, 4.4).
- Benchmarking scope
  - Evaluations are system‑throughput/latency benchmarks, not accuracy evaluations. The method changes runtime behavior, not model outputs, but quantization and large‑context runs could interact with quality budgets in practical systems (quality is not studied here).
- Complexity and engineering effort
  - Achieving the reported gains depends on careful kernel fusion and scheduling (Looped CollectiveEinsum), plus explicit collective placement (Section 3.5). Porting or reproducing on other stacks may be non‑trivial.
- Communication limits at extreme scale
  - Though 2D weight‑stationary continues to improve with √nchips, all approaches remain fundamentally constrained by memory bandwidth and interconnect efficiency. The paper itself points to sparsity/MoE and compressed communication as future levers (Conclusions).

## 7. Implications and Future Directions
- Changes to the field
  - Provides a clear, formula‑backed playbook to choose partitioning for LLM inference:
    - Use 2D weight‑stationary for decode and for small to medium `B·L`.
    - Switch to weight‑gathered for large‑batch prefill.
    - With MQA models, shard attention by batch during decode to unlock long contexts.
  - Demonstrates that 64‑way tensor parallelism can still achieve high MFU (≈44%) with the right 2D layout (Figure 9), countering the notion that tensor parallelism stops scaling beyond 16–32 ways.

- Follow‑up research enabled/suggested
  - Activation quantization and communication compression to further reduce latency and cost in both prefill and decode (Sections 3.6 and Conclusions).
  - Extending the analytical model to heterogeneous clusters, multi‑node GPU interconnect topologies, or hybrid pipeline/tensor parallelism.
  - Automatic partitioning planners using these closed‑form cost models as objectives, avoiding black‑box searches while adapting to workload and hardware changes.
  - Applying similar batch‑sharding ideas to other stateful components (e.g., cache‑efficient retrieval augmentation).

- Practical applications
  - Interactive assistants: With
    > “0.29 s prefill (2048 tokens) + 1.82 s to generate 64 tokens” on PaLM 540B‑int8 at 64 chips (Table 2),
    real‑time chat and coding use‑cases become feasible.
  - Offline large‑scale scoring or distillation: With
    > “76% MFU prefill” and high MFU decode at batch 512 (Table 2),
    cost per token drops for massive throughput jobs.
  - Long‑context tasks (e.g., summarization, extended conversations): Optimized MQA enables tens of thousands of tokens per context under fixed memory (Table 1), opening new application regimes.

In sum, the paper delivers both a conceptual framework (closed‑form communication models guiding partitioning decisions) and a practical system (2D/weight‑gathered MLPs, batch‑sharded MQA decode, fused parallel blocks, overlapped collectives) that materially lowers the latency and cost of LLM inference at 500B+ scale.
