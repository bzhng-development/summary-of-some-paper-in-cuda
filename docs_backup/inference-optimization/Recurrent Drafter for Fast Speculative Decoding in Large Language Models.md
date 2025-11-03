# Recurrent Drafter for Fast Speculative Decoding in Large Language Models

**ArXiv:** [2403.09919](https://arxiv.org/abs/2403.09919)
**Authors:** Yunfei Cheng, Aonan Zhang, Xuanyu Zhang, Chong Wang, Yi Wang
**Institutions:** Apple Machine Learning Research

## 🎯 Pitch

ReDrafter introduces an innovative speculative decoding method that significantly accelerates LLM generation by integrating a small recurrent model and a novel GPU-friendly dynamic tree attention approach. This advancement not only reduces computational overhead by increasing accepted tokens per pass but also enhances the feasibility of deploying faster, more efficient on-device language models, thereby improving user experience in both datacenter and mobile environments.

---

## 1. Executive Summary
ReDrafter is a speculative decoding method that attaches a small recurrent model to an LLM to propose several future tokens, then verifies them in bulk with the LLM so that multiple tokens are accepted per LLM forward pass. Its key innovations—an RNN-based drafter conditioned on the LLM’s last hidden state, a GPU-friendly “dynamic tree attention” to avoid duplicate computation, and distillation-based training—deliver up to 2.8× speedup on Nvidia H100 (PyTorch) and up to 2.3× on Apple Silicon GPUs (MLX), with identical outputs to standard autoregressive decoding (Abstract; Fig. 2; Table 1; Sec. 4.2).

## 2. Context and Motivation
- The problem: LLM generation is throughput-limited by memory bandwidth. Every new token requires a pass through the large model, making inference slow and costly.
- Why it matters:
  - Practical: Faster generation improves user experience and enables on-device assistants where compute and bandwidth are limited (Sec. 4.2).
  - Systems-level: Reducing LLM calls shifts work from memory-bound operations to compute-friendly ones, increasing hardware utilization (Sec. 2).
- Prior approaches and their gaps (Sec. 2):
  - Detached draft models: Use a separate smaller model to propose tokens (e.g., using a small variant of the same family). Downsides include training/integration overhead and alignment challenges.
  - Attached heads (e.g., Medusa): Multiple independent heads predict T future positions from the LLM’s hidden state. This leverages parallelism but ignores local temporal dependencies, leading to weaker predictions as T grows and an “exponentially large set of feasible candidate token sequences” to verify (Sec. 1 and 2).
  - Recurrent drafters (e.g., EAGLE variants): Add recurrence to better model local dependencies, improving accuracy but reducing GPU parallelism; overhead can erode net speedup (Sec. 2).
- Positioning: ReDrafter combines a lightweight recurrent draft model (to raise acceptance rates) with a new verification-time optimization—dynamic tree attention—that recovers parallelism and reduces duplication, plus knowledge distillation to improve alignment, achieving state-of-the-art speedups across hardware (Sec. 1, 3.1–3.5, 4.1–4.2).

## 3. Technical Approach
ReDrafter’s decoding loop consists of four steps (Fig. 2; Sec. 3.1–3.4):

1) Get one guaranteed next token from the LLM
- The LLM runs one standard autoregressive step to produce the next token and its last-layer hidden state `h`. This token is the “anchor” that will always be accepted (green in Fig. 2).
- Guaranteeing at least one token per step ensures progress and provides a starting point and conditioning signal for the drafter.

2) Draft multiple future tokens with a recurrent head (Sec. 3.1; Fig. 3)
- Drafter input: the LLM’s last-layer hidden state `h` (a compact summary of the current context) and the embedding of the last accepted token.
- Recurrent update: A simple one-layer RNN maintains a hidden state that concatenates the drafter’s recurrent state `s_t` with `h`:
  - Intuition: `h` injects the LLM’s view of the context; the RNN captures short-range temporal structure across the drafted tokens.
  - Update rule (described in plain language, then notation): The drafter takes the previous recurrent state and the embedding of the current token it just drafted, applies a linear transform plus nonlinearity to obtain the next recurrent state, and concatenates it with `h` to predict the next token via an MLP and softmax.
  - In notation (Sec. 3.1): initialize `g1 = [s1, h]` with `s1 := e1` the embedding of the just-verified token. For step `t`, update `st = f(U st−1 + W e_t + b)` and set `g_t = [s_t, h]`. Then predict next-token probabilities with a small MLP+softmax.
- Why recurrence? It explicitly models local token-to-token dependencies across the drafted span, improving draft accuracy and thus acceptance (Sec. 3.1). Parameters are shared across positions, so the drafter size does not grow with the drafting horizon.

3) Explore alternatives with beam search (Sec. 3.2; Fig. 4)
- Beam search makes the drafter propose several likely continuations:
  - `beam width` = how many alternative sequences are kept at each step.
  - `beam length` = how many tokens each candidate sequence attempts to predict ahead.
- Rationale: A wider beam raises the chance that at least one candidate will match the LLM for many tokens, increasing accepted tokens per step and reducing total LLM calls (Sec. 3.2). However, verifying more candidates costs more FLOPs; the best width depends on hardware (Sec. 4.3.1, Table 3; Sec. 4.2, Table 2).

4) Verify efficiently with “dynamic tree attention” (Sec. 3.3; Fig. 4)
- Problem: Many beam candidates share long prefixes. Naively verifying each candidate separately repeats work for the shared parts (Fig. 4a shows 15 tokens to verify across three candidates of length 5).
- Solution: De-duplicate shared prefixes across the beam and construct a single “packed” verification pass with a custom attention mask that encodes the tree dependencies between tokens (Fig. 4b–4c).
  - Practical effect: The example in Fig. 4 reduces 15 tokens to 8 after packing.
  - Implementation detail: The packed-verification attention mask ensures each token only attends to its valid prefix path in the candidate tree (Fig. 4b–4c).
  - GPU-friendly construction: Instead of building a trie (hard to parallelize), ReDrafter exploits the fact that all candidates have the same length and derives shared-prefix indices with tensor operations. Appendix A.1 gives a 5-line `dedup_prefix` routine that finds for each prefix the smallest-indexed candidate that shares it, yielding a `prefix_tree`. A subsequent `pack_beam` function creates the packed sequence (Fig. 4c).

5) Accept the longest matching prefix and repeat (Sec. 3.4)
- The LLM runs once on the packed beam to compute log-probabilities for all proposed tokens. The system then identifies the longest prefix where the drafter’s proposal exactly matches the LLM’s next-token predictions.
- Selection can be greedy (exact token matches) or based on rejection sampling; the paper uses greedy decoding with temperature=0 in most experiments (Sec. 4.1).
- Accepted tokens are appended to the context; their final states become inputs for the next step. The process loops until stopping. Outputs are guaranteed to be identical to pure autoregressive decoding (Fig. 2: “ReDrafter ensures that the tokens it generates are identical to those produced by the LLM”).

Training the drafter with knowledge distillation (Sec. 3.5; Eqs. (1)–(2))
- Objective: Make the drafter’s joint distribution over the next `T` tokens match the LLM’s, not just the single ground-truth token. This raises acceptance likelihood.
  - Plain-English view: Teach the drafter to mimic the LLM’s probability distribution over the next few steps, so the drafter proposes what the LLM itself would generate.
  - Formalization: minimize KL divergence `KL(p_llm(y1:T) || p_draft(y1:T))` (Eq. (1)), implemented as an expected negative log-likelihood over short sequences sampled from the LLM: `E_{p_llm}[-log p_draft(y_{t+1:t+T} | y_{1:t})]` (Eq. (2)).
- Practical detail: Distillation is local—at each position, the LLM predicts T future tokens conditioned on the ground-truth context. Only the drafter is trained; the LLM is frozen (Sec. 3.5).

Implementation footprint
- PyTorch on Nvidia H100 for server-class evaluation (Secs. 4.1, 4.3).
- MLX on Apple Silicon GPUs for on-device evaluation (Sec. 4.2), with system-level tips on data types, lazy evaluation, and JIT behavior (Appendix A.2.2).

## 4. Key Insights and Innovations
- RNN-conditioned-on-LLM hidden state (Sec. 3.1; Fig. 3)
  - Novelty: Combines the LLM’s last-layer hidden state `h` with a small recurrent model that drafts several steps ahead. This captures short-range dependencies across draft tokens that independent heads (e.g., Medusa) miss.
  - Significance: Improves draft accuracy and thus acceptance rate, converting compute into fewer LLM calls. Stays lightweight by sharing parameters across positions.
- Dynamic tree attention for verification (Sec. 3.3; Fig. 4; Appendix A.1)
  - Novelty: A tensor-only, GPU-friendly way to de-duplicate shared prefixes across beam candidates and construct an attention mask that encodes the candidate tree. Avoids a runtime trie build.
  - Significance: Cuts verification tokens by 30–60% in practice (Fig. 6-left), particularly benefiting compute-limited settings and larger beams.
- Distillation over short future sequences (Sec. 3.5; Table 4)
  - Novelty: Distills the joint distribution over the next `T` tokens (not just the next token) from the LLM to the drafter, improving agreement where it matters—multi-token acceptance.
  - Significance: Yields ≈10% gains in speedup and accepted tokens per step across beam widths (Table 4).
- Balanced system design that travels across hardware (Secs. 4.1–4.2)
  - Novelty: The paper evaluates server GPUs and on-device GPUs, surfacing different optimal settings (small beams on M1/M2; large beams on H100).
  - Significance: Demonstrates practicality in both datacenter and on-device contexts, with code paths tuned for each (PyTorch vs. MLX).

## 5. Experimental Analysis
Evaluation setup (Sec. 4)
- Base LLMs: Vicuna 7B, 13B, 33B.
- Benchmarks: MT-Bench and AlpacaEval (Sec. 4.1). Decoding mostly at temperature=0 (greedy).
- Metrics:
  - “Speedup”: wall-clock speed versus standard autoregressive decoding.
  - “Tokens/Step”: average number of tokens accepted per LLM forward pass—higher is better.
- Baselines: Autoregressive (AR), Medusa, and EAGLE (Sec. 4.1).
- Implementations: PyTorch on H100 (Sec. 4.1) and MLX on Apple Silicon (Sec. 4.2).

Main results on H100 (Table 1; Fig. 5)
- Overall:
  - MT-Bench: 
    - Vicuna-7B: ReDrafter 2.80× speedup, 4.20 Tokens/Step. EAGLE: 2.69×, 3.96. Medusa: 2.39×, 2.55.
    - Vicuna-13B: ReDrafter 2.80×, 4.21; EAGLE 2.74×, 4.00; Medusa 2.40×, 2.61.
    - Vicuna-33B: ReDrafter 2.61×, 3.87; EAGLE 2.80×, 3.71; Medusa 2.51×, 2.53.
  - AlpacaEval:
    - 7B: ReDrafter 2.69×, 4.06; EAGLE 2.43×, 3.61; Medusa 2.19×, 2.42.
    - 13B: ReDrafter 2.78×, 4.02; EAGLE 2.49×, 3.62; Medusa 2.26×, 2.45.
    - 33B: ReDrafter 2.43×, 3.61; EAGLE 2.59×, 3.29; Medusa 2.31×, 2.31.
- Takeaways:
  - ReDrafter attains the highest Tokens/Step in all settings (Table 1), and the highest speedups on 7B/13B. On 33B, ReDrafter’s Tokens/Step is best, but EAGLE’s speedup is slightly higher (Table 1).
  - Performance is consistent across MT-Bench categories (Fig. 5). The paper notes “a gap between Tokens/Second and speedup,” attributed to speculative-decoding overheads (Sec. 4.1).

On-device (Apple Silicon, MLX) results (Sec. 4.2; Table 2)
- M1 Max (Vicuna-7B):
  - Best speedup ≈1.32× at beam width 1; speed drops at larger beams due to verification FLOP costs (Table 2).
- M2 Ultra:
  - 7B and 13B: Best speedups ≈1.52× (7B, BW=3) and ≈1.94× (13B, BW=2–3), with 2.4–2.8 Tokens/Step (Table 2).
  - 33B: Very low per-request TPS (e.g., 1.15–1.33), but still 1.97–2.28× speedups because decoding becomes memory/IO-bound; fewer LLM calls still pay off (Table 2; Sec. 4.2).
- Interpretation:
  - Wider beams are beneficial only up to the hardware’s compute limit; beyond that, verification cost dominates and speedups fall (Sec. 4.2).

Ablations and robustness
- Beam width vs. batch size (Sec. 4.3.1; Table 3)
  - Latency (per-request TPS) is maximized at large beam width and tiny batch size: ≈110 TPS at BW=64, BSZ=1–2.
  - Throughput (TPS×BSZ) peaks at moderate beam width and large batch size: ≈1636 at BW=2, BSZ=80.
  - OOM arises at high batch with wide beams; practical deployments must tune for their objective (latency vs throughput).
- Dynamic tree attention effectiveness (Sec. 4.3.2; Fig. 6)
  - Token compression: Reduces verification tokens by 30–60%, with stable compression ratios across beam sizes; distributions reported up to 99th/1st percentiles (Fig. 6-left).
  - Throughput under load: With beam width=45 and beam length=5, tree attention gives higher TPS and TPS×BSZ once batch size >4 (Fig. 6-right). When compute is plentiful (batch ≤4), both versions perform similarly.
- Distillation gains (Sec. 4.3.3; Table 4)
  - Quote: “distillation lead to an approximate 10% increase in the speedup and the average accepted tokens per step.”
  - Example (Vicuna-7B, BSZ=1): at BW=64, speedup improves from 1.99× to 2.18×; Tokens/Step from 3.30 to 3.58. Benefits are present from BW=1 through BW=64.

System and implementation observations (Appendix A.2.2)
- MLX specifics: float16 > bfloat16 for speed; low-bit quantization helps. MLX uses lazy evaluation and JIT; frequent `array.item()` can fragment traces and slow kernels. The team uses Instruments to profile CPU/GPU timelines.

Overall support for claims
- The paper provides detailed comparisons against strong baselines (Medusa, EAGLE), cross-hardware evidence (H100, M1 Max, M2 Ultra), and careful ablations isolating each innovation (beam size/batch, tree attention, distillation). The gains in Tokens/Step and speedup are consistent and well-explained by system trade-offs (Secs. 4.1–4.3; Fig. 5–6; Tables 1–4).

## 6. Limitations and Trade-offs
- Parallelism vs. recurrence (Sec. 2)
  - While the RNN improves draft accuracy, it reduces pure parallelism compared to fully independent heads. ReDrafter compensates with dynamic tree attention and beam search, but the drafter still runs sequentially across drafted tokens.
- Verification cost at large beams (Sec. 4.2; Table 2)
  - Wider beams increase LLM verification FLOPs; on compute-limited devices (e.g., M1 Max), the optimal beam can be as small as 1. Over-widening the beam hurts speedup.
- Hardware- and model-size sensitivity (Sec. 4.2)
  - On small GPUs or very large models (Vicuna-33B on M2 Ultra), inference can be dominated by memory/IO, making absolute TPS low even if speedup is >2×.
- Training overhead and data needs (Sec. 3.5; 4.3.3)
  - Distillation shifts cost from inference to training. Building a distillation corpus of LLM rollouts adds preprocessing time and storage. The drafter must be retrained per LLM or after LLM updates.
- Assumed equal candidate lengths in dynamic tree attention (Sec. 3.3; Appendix A.1)
  - The tensorized dedup relies on all candidates having the same drafted length. Variable-length candidate sets would require additional engineering.
- Decoding regime coverage
  - Most tests use greedy decoding (temperature=0; Sec. 4.1). Behavior under diverse sampling settings (top-p, high temperature) is not deeply explored here.
- Scope of LLM families
  - Experiments focus on Vicuna. While the approach is model-agnostic in principle, results on other architectures are not reported in this version.

## 7. Implications and Future Directions
- Field impact
  - ReDrafter clarifies that small recurrent drafters, when paired with GPU-friendly beam de-duplication, can surpass independent-head methods in accepted tokens per step without sacrificing wall-clock performance. This reframes the design space for speculative decoding around “accurate local sequence modeling + efficient verification,” rather than maximizing head parallelism alone.
- Practical applications
  - Datacenter serving: Higher throughput or lower latency for chat and code assistants by tuning beam width vs batch size (Table 3).
  - On-device AI: Meaningful 1.3–2.3× speedups on Apple Silicon at modest beams make local assistants more viable, especially when combined with 4-bit quantization and careful MLX engineering (Sec. 4.2; Appendix A.2.2).
  - Systems integration: The dynamic tree attention primitive can also accelerate detached drafters (Sec. 3.3), suggesting a reusable building block for token-tree verification.
- Research directions
  - Stronger distillation and training curricula: The paper notes opportunities to “enhanc[e] draft model training through more advanced distillation techniques” (Sec. 5). This includes sequence-level or teacher-free variants, better negative sampling, and longer-horizon T.
  - Adaptive beam control: Dynamically adjust beam width/length per step or per hardware load to stay at the latency/throughput sweet spot discovered in Table 3 and Fig. 6.
  - Broader decoding regimes: Evaluate under diverse sampling strategies, multilingual tasks, and long-context settings. The packed-beam attention mask could be extended to variable-length candidates and deeper token trees.
  - Compiler/runtime co-design: Further kernel fusion for dynamic tree attention and packed-beam masks, integration with inference engines (the paper mentions collaboration to integrate into TensorRT-LLM in Acknowledgments).

> Bottom line: ReDrafter demonstrates that an RNN drafter, when coupled with dynamically constructed, tensorized tree attention and trained via local sequence distillation, can reliably turn one LLM forward pass into several accepted tokens. This yields state-of-the-art speculative decoding speedups across both datacenter and on-device hardware (Fig. 1; Table 1; Table 2), while preserving exact equivalence to standard autoregressive outputs (Fig. 2).
