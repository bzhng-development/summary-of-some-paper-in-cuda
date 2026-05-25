# RECURRENT DRAFTER FOR FAST SPECULATIVE DECODING IN LARGE LANGUAGE MODELS

**ArXiv:** [2403.09919](https://arxiv.org/abs/2403.09919)

## 🎯 Pitch

Recurrent Drafter (ReDrafter) introduces a novel speculative decoding technique that pairs large language models with an efficient RNN-based drafter, enabling the generation and verification of multiple tokens per LLM forward pass—dramatically speeding up inference without sacrificing exactness of outputs. By leveraging a GPU-friendly dynamic tree attention to eliminate redundant computation and training the drafter through knowledge distillation, ReDrafter achieves state-of-the-art acceleration (up to 2.8× on GPUs, 2.3× on Apple Silicon) while preserving model fidelity. This innovation is key for real-time and resource-constrained applications, slashing latency and infrastructure costs while keeping outputs identical to the original LLM.

---

## 1. Executive Summary
ReDrafter is a speculative decoding method that accelerates large language model (LLM) inference by pairing the LLM with a small recurrent drafter, then verifying many drafted tokens at once using a single LLM forward pass. The key advances are: (a) a lightweight RNN-based drafter conditioned on the LLM’s last hidden state, (b) a GPU-friendly “dynamic tree attention” that removes duplicate prefixes across beam-search candidates, and (c) training the drafter via knowledge distillation so its predictions closely match the LLM. On MT-Bench with Vicuna models and an Nvidia H100, ReDrafter reaches up to 2.8× speedup while preserving exact LLM outputs (Figure 1, Table 1); it also delivers up to 2.3× on Apple Silicon GPUs using MLX (Figure 1, Table 2).

## 2. Context and Motivation
- Problem addressed
  - LLM text generation is slowed by memory bandwidth and sequential token-by-token decoding. Speculative decoding mitigates this by letting a smaller “draft” model propose multiple future tokens, which the LLM then verifies in one pass.
- Why it matters
  - Real-time assistants, on-device inference, and cost-sensitive serving all demand higher tokens-per-second. Cutting LLM forward passes directly lowers latency and server costs without changing model quality—if the final outputs remain identical to the base LLM (Section 3.4, Figure 2).
- Prior approaches and gaps
  - Detached drafter models: use a smaller separate LLM; simple to reuse but adds system complexity and may misalign with the target LLM (Section 2).
  - Attached, non-recurrent heads (e.g., Medusa): predict multiple future tokens in parallel from the same hidden state, but each head is independent; accuracy drops as the prediction horizon grows, hurting acceptance rates and requiring many heads with separate parameters (Section 2).
  - Recurrent speculative approaches: improve accuracy by using recurrence, but can underutilize GPUs due to reduced parallelism, eroding speedup despite high acceptance rates (Section 2).
- Positioning
  - ReDrafter combines recurrence (to improve prediction accuracy) with a dynamic, GPU-friendly verification scheme (to reclaim parallel efficiency), plus knowledge distillation (to shift work from inference to training), yielding state-of-the-art speedups on both server GPUs and Apple Metal GPUs (Figure 1, Table 1, Table 2).

## 3. Technical Approach
At a high level, each decoding step alternates between a small drafter proposing multiple tokens and the LLM verifying them, then accepting the longest correct prefix (Figure 2).

Key terms explained briefly:
- Speculative decoding: a method where a small model proposes several next-token candidates, and the large model only verifies them, reducing the number of large-model forward passes.
- Drafter: the small model used to propose tokens.
- Beam search: a search procedure that keeps the top-K partial sequences at each step (here used inside the drafter).
- Dynamic tree attention: a runtime-constructed attention mask that exploits shared prefixes among beam candidates so the LLM verifies only unique tokens once.
- Knowledge distillation: training the drafter to match the LLM’s predictive distribution rather than ground-truth tokens, improving alignment.

Step-by-step mechanics (Sections 3.1–3.4, Figures 2–4):
1) Inputs to the drafter
   - The LLM runs one normal forward step to produce the next token and its last-layer hidden state `h` (Figure 2, green token). This token is “guaranteed” correct.
   - The drafter then conditions on:
     - `h`: the LLM’s last-layer output for the current position.
     - The embedding of the latest accepted token (and subsequent drafted tokens), fed recurrently (Figure 3).

2) Recurrent drafter architecture (Section 3.1, Figure 3)
   - Hidden state initialization: `g1 = [s1, h]` where `s1 := e1` is the embedding of the LLM’s just-generated token and `[...]` denotes concatenation.
   - Recurrence: for t = 2…T (T = draft horizon), update the drafter state with a one-layer RNN:
     - `st = f(U st-1 + W e_t + b)`
     - `gt = [st, h]`
   - Output head: a small MLP with skip connections, ending in a softmax over the vocabulary.
   - Parameter sharing across time steps: the same drafter parameters are reused for all drafted positions, keeping the drafter compact even for larger horizons (Section 3.1).

   Why recurrence? Unlike independent heads, the drafter’s next-step prediction can depend on previously drafted tokens, capturing short-range dependencies and improving the chance that the LLM will later accept them (Section 3.1).

3) Drafter beam search (Section 3.2, Figure 4)
   - The drafter performs beam search with width `BW` and length `T` to explore multiple candidate continuations and rank them by probability.
   - Wider beams raise the chance that at least one candidate matches the LLM’s continuation, increasing accepted tokens per step—but also increase FLOPs for verification (Section 3.2).

4) Dynamic tree attention for verification (Section 3.3, Figure 4)
   - Problem: beam candidates often share long prefixes. If passed naively to the LLM for verification, duplicated tokens waste compute (Figure 4a).
   - Solution: compress candidates into a “packed beam” by deduplicating shared prefixes and building a tree-structured attention mask so each token attends only to its valid ancestors (Figure 4b–c).
   - GPU-friendly construction: the beam candidates have equal length T, enabling a pure tensor implementation to find shared prefixes without building a sequential trie. Appendix A.1 shows a five-line PyTorch routine `dedup_prefix` that:
     - Creates a 3D equality tensor `matches = beam[:, :, None] == beam[:, None, :]`.
     - Uses `cumsum` over prefix length to find identical prefixes across candidates.
     - Uses `argmax` to map each prefix to the smallest-index candidate sharing it.
   - After deduplication, a masked-attention pass over the packed tokens lets the LLM compute all verification logits in one forward pass, reusing compute for shared prefixes (Figure 4b–c).

5) Acceptance and progression (Section 3.4, Figure 2)
   - The LLM produces log-probabilities for all proposed tokens in the packed beam.
   - The system finds the longest prefix in the beam that matches the LLM’s predictions; that entire prefix is accepted and appended to the output context.
   - This guarantees the final generated text is exactly what the LLM would have produced with standard greedy decoding (Figure 2, Section 3.4). Selection can be greedy matching or rejection sampling; experiments mainly use greedy (Section 4.1).

6) Training via knowledge distillation (Section 3.5; Equations (1)–(2))
   - Objective: make the drafter’s joint distribution over the next T tokens close to the LLM’s (`min KL(pllm(y1:T) || pdraft(y1:T))`; Equation (1)).
   - Practical training: at position `t` in a training sequence, collect the LLM’s T-token continuation `ŷt+1:t+T` (with ground-truth context `y1:t`) and maximize the drafter likelihood `pdraft(ŷt+1:t+T | y1:t)` (Equation (2)).
   - Only the drafter is trained; the LLM is frozen, ensuring inference outputs remain unchanged (Section 3.5).

Implementation notes (Appendix A.2):
- MLX on Apple Silicon requires attention to lazy evaluation, JIT compilation stability, and data types (e.g., `float16` and low-bit quantization can be faster than `bfloat16`/`float32`), which influence end-to-end performance.

## 4. Key Insights and Innovations
- Recurrent, LLM-conditioned drafter (Section 3.1, Figure 3)
  - What’s new: replaces many independent heads (as in Medusa) with a single shared-parameter RNN conditioned on the LLM’s hidden state.
  - Why it matters: recurrence models local token dependencies, delivering higher accuracy for longer draft horizons without expanding parameter count. This translates into more tokens accepted per LLM call and better speedups (Table 1 shows higher Tokens/Step than Medusa across Vicuna 7B/13B/33B).
- Dynamic tree attention built with tensor ops (Section 3.3, Figure 4, Appendix A.1)
  - What’s new: a GPU-friendly procedure to deduplicate shared prefixes among equal-length beam candidates and build the required attention masks—without a slow trie.
  - Why it matters: reduces the number of verification tokens by 30–60% across a wide range of beam sizes (Figure 6 left), especially valuable when compute is tight (Figure 6 right).
- Distillation that shifts work from inference to training (Section 3.5, Section 4.3.3, Table 4)
  - What’s new: sequence-level distillation that matches the LLM’s next-T-token distribution, with the LLM frozen.
  - Why it matters: improves acceptance rates and speedups by roughly 10% across beam widths (Table 4), while preserving exact final outputs.
- Hardware-aware beam design and masking (Sections 3.2, 4.2, 4.3.1)
  - What’s new: explicit analysis of the beam width/length trade-offs under different hardware regimes (H100 vs. Apple M1/M2 GPUs), plus batch-size effects (Table 2, Table 3, Figure 7).
  - Why it matters: shows how to tune speculative decoding to specific devices—narrow beams are optimal on Apple GPUs, wider beams on server GPUs—maximizing real-world speedup.

## 5. Experimental Analysis
- Setup and metrics (Section 4)
  - LLMs: Vicuna 7B, 13B, 33B.
  - Datasets/benchmarks: MT-Bench and AlpacaEval (Section 4.1; Figure 5 shows category-wise results).
  - Hardware/implementations:
    - PyTorch on Nvidia H100 (Section 4.1).
    - MLX on Apple Silicon GPUs: M1 Max and M2 Ultra (Section 4.2).
  - Metric definitions:
    - Speedup: vs. standard autoregressive decoding throughput.
    - Tokens/Step: average number of drafted tokens accepted by the LLM per decoding step; higher is better because it reduces the number of LLM passes.

- Headline results (Table 1; Section 4.1)
  - MT-Bench (temperature 0, greedy decoding):
    - Vicuna 7B: ReDrafter 2.80× speedup, 4.20 Tokens/Step vs. Medusa 2.39×/2.55 and EAGLE 2.69×/3.96.
    - Vicuna 13B: ReDrafter 2.80×/4.21 vs. Medusa 2.40×/2.61 and EAGLE 2.74×/4.00.
    - Vicuna 33B: ReDrafter 2.61×/3.87 vs. Medusa 2.51×/2.53 and EAGLE 2.80×/3.71.
  - AlpacaEval shows a similar pattern, with ReDrafter leading on 7B and 13B and competitive on 33B (Table 1 bottom).
  - Category-level consistency: Figure 5 shows ReDrafter’s Tokens/Step and speedups are strong across MT-Bench and AlpacaEval subcategories, though some speedup gap remains vs. theoretical Tokens/Step due to overheads.

- On-device MLX results (Table 2; Section 4.2, Figure 7)
  - M1 Max, Vicuna 7B: best at beam width (BW)=1, 1.32× speedup (Tokens/Step 2.15).
  - M2 Ultra:
    - Vicuna 7B: best at BW=3, 1.52× (Tokens/Step 2.44).
    - Vicuna 13B: best at BW=2 or 3, 1.94× (Tokens/Step ≈2.82–2.94).
    - Vicuna 33B: speedup grows with BW up to 2.28× at BW=4 (Tokens/Step 2.56).
  - Quote:
    > “Despite the limited compute resources… we observed a memory bottleneck… ReDrafter effectively mitigates this bottleneck, resulting in up to 2.3× speedup” (Section 1 and 4.2; Table 2 shows up to 2.28×).

- Ablations and diagnostics
  - Beam width × batch size (Table 3; Section 4.3.1)
    - Per-request speed (TPS) is highest with large beam and small batch (e.g., BW=64, BSZ=1–2 gives ~110 TPS).
    - System throughput (TPS×BSZ) peaks with moderate beam and large batch (e.g., BW=2, BSZ=80 ≈ 1636 TPS×BSZ), but large BW at large BSZ can hit OOM.
    - Takeaway: tune for latency (large BW, small BSZ) or throughput (moderate BW, large BSZ).
  - Dynamic tree attention effectiveness (Figure 6; Section 4.3.2)
    - Token compression: packed-beam tokens are 30–60% fewer than uncompressed (Figure 6 left) across beams up to 70×5 tokens.
    - When compute-bound (batch size > 4), dynamic tree attention substantially boosts both per-request TPS and total throughput (Figure 6 right). When compute is abundant (BSZ ≤ 4), benefits are small.
  - Knowledge distillation (Table 4; Section 4.3.3)
    - Distillation (T=5, temperature 0) yields ≈10% improvements in both speedup and Tokens/Step across beam widths; e.g., at BW=16, speedup rises from 1.80 to 1.92 and Tokens/Step from 2.87 to 3.09.

- Do the experiments support the claims?
  - Yes, on server GPUs (H100), ReDrafter consistently outperforms Medusa and matches or exceeds EAGLE in speedup while leading in Tokens/Step on 7B/13B (Table 1).
  - On Apple GPUs, careful tuning shows meaningful gains despite limited compute/memory, and the dynamic tree attention delivers predictable compression (Table 2, Figure 6, Figure 7).
  - The guarantee of identical outputs is preserved because the LLM verifies and only accepts matching prefixes (Section 3.4, Figure 2).

## 6. Limitations and Trade-offs
- Hardware utilization vs. recurrence (Sections 2, 3.1, 4.3.1)
  - Recurrence improves prediction accuracy but reduces parallelism compared to fully parallel heads; GPU utilization must be recovered by batching, beam tuning, and efficient masking. Large beams raise verification FLOPs and can hurt speedup on weaker GPUs (Table 2).
- Dynamic tree attention assumptions (Section 3.3, Appendix A.1)
  - The fast tensor-based construction exploits equal-length beams. If candidate lengths vary, extending the same trick may be non-trivial and could reintroduce trie-like overheads.
- Scope of decoding strategies (Section 4.1)
  - Main results use greedy decoding (temperature 0). Behavior under diverse sampling strategies (e.g., temperature sampling, nucleus sampling) is not extensively reported here.
- Model/ecosystem coverage
  - Evaluations focus on Vicuna (7B/13B/33B). Generalization to other architectures, multilingual setups, or highly domain-specific LLMs is promising but not demonstrated here.
- Memory-bound large models on device (Section 4.2)
  - For very large models (e.g., 33B on Apple GPUs), overall TPS may be dominated by memory/IO constraints; speculative decoding still helps, but absolute throughput remains low without quantization or further compression.
- Training data and objective (Section 3.5, 4.3.3)
  - Distillation uses next-T-token continuations from the target LLM with temperature 0; broader temperature ranges or richer sequence-level objectives could further improve robustness but are not explored.

## 7. Implications and Future Directions
- Field impact
  - ReDrafter shows that combining recurrence (for accuracy) with dynamic verification (for parallel efficiency) sets a new state-of-the-art speed/acceptance balance for speculative decoding (Table 1). It also demonstrates practicality across execution stacks (PyTorch/CUDA and MLX/Metal), informing device-specific tuning (Figure 7, Table 2–3).
- Near-term applications
  - Latency-sensitive assistants and chat agents (server-side and on-device).
  - Cost-efficient LLM serving (fewer LLM passes per token).
  - On-device inference in constrained environments (Apple Silicon) where memory bandwidth is a bottleneck, with further gains possible via quantization (Section 4.2).
- Research directions
  - Stronger distillation: temperature schedules, sequence-level objectives beyond Eq. (2), and training on sampled, diverse continuations to improve acceptance under non-greedy decoding (Section 5).
  - Variable-length dynamic tree attention: fast tensorized prefix-trie analogs for heterogeneous lengths.
  - More expressive drafters: gated RNNs or lightweight transformers while retaining shared parameters; explore hybrid parallel–recurrent designs to trade off utilization vs. accuracy.
  - Wider LLM coverage: different families, multilingual tasks, and long-context settings; integrate with serving frameworks (the paper notes integration with TensorRT-LLM).
  - Joint optimization with quantization/pruning: co-design drafter, attention masks, and low-bit kernels for mobile-class GPUs.

In short, ReDrafter contributes a principled, implementable path to faster LLM inference: a compact recurrent drafter aligned by distillation, verified efficiently through dynamic tree attention, and tuned carefully to hardware. The method preserves exact LLM outputs while significantly reducing effective decoding steps per token, making it compelling for both datacenter and on-device deployments.
