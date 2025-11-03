# Break the Sequential Dependency of LLM Inference Using LOOKAHEAD DECODING

**ArXiv:** [2402.02057](https://arxiv.org/abs/2402.02057)

## 🎯 Pitch

This paper introduces Lookahead Decoding, a novel, exact, and parallel decoding algorithm for large language models (LLMs) that breaks the traditional one-token-at-a-time bottleneck of autoregressive decoding. By reformulating generation as a fixed-point problem, Lookahead Decoding enables multi-token, parallel decoding without auxiliary draft models, significantly accelerating inference—achieving up to 1.8× speedup on single GPUs and 4× speedup in multi-GPU settings—while preserving generation quality. This advancement unleashes the full compute potential of modern accelerators, lowering latency for interactive LLM applications and opening new directions for scalable, high-throughput language generation.

---

## 1. Executive Summary (2–3 sentences)
This work introduces Lookahead Decoding, an exact (distribution-preserving) parallel decoding algorithm that reduces the number of sequential steps needed to generate text with large language models (LLMs). It replaces the strictly one-token-at-a-time bottleneck of autoregressive decoding with a “decode, predict, and verify in the same step” procedure that exploits unused compute, scales to multiple GPUs, and integrates with FlashAttention. Reported gains include up to 1.8× throughput on MT-Bench in single-GPU settings and up to 4× with multi-GPU strong scaling on code completion tasks (Abstract; Fig. 5–7).

## 2. Context and Motivation
- Problem addressed: Standard autoregressive decoding generates exactly one token per step in sequence order, so the total latency is proportional to the number of tokens generated. Each step is memory-bandwidth bound (due to causal attention’s dependence on all previous tokens), which underutilizes modern accelerators’ compute capacity (Sec. 1; Sec. 2, “Causal Attention in Decoder Models”).
- Importance:
  - Real-world: Latency is a critical bottleneck for chatbots, search assistants, and code assistants. Reducing time-to-first-answer and tokens/s latency directly improves user experience and service costs.
  - Theoretical/system-level: Demonstrates that sequential generation can be reframed as a parallelizable fixed-point problem, opening design space beyond token-at-a-time decoding (Sec. 2).
- Prior approaches and their limits:
  - Speculative decoding uses a small “draft” model to guess multiple tokens and the base model to verify them in parallel (Sec. 2, “Guess-And-Verify Paradigm,” Eq. 2). Speedup is capped by acceptance rate α: tokens that fail verification must be regenerated, and training/maintaining a draft model that generalizes is non-trivial (Sec. 1; Sec. 4.1).
  - Other variants rely on special-purpose draft heads or retrieval (Medusa, OSD, EAGLE, REST; Sec. 6), which add components and still hinge on acceptance quality.
- Positioning: Lookahead Decoding removes the draft model entirely by generating and verifying multiple disjoint n-grams in parallel using the base LLM itself. It converts excess compute (otherwise idle due to memory-bound behavior) into fewer decoding steps while preserving the exact output distribution (Sec. 3; Appendix B).

## 3. Technical Approach
The core idea is to convert autoregressive decoding into a parallel, exact algorithm that (a) generates many candidate n-grams ahead, (b) verifies multiple candidates concurrently, and (c) accepts as many as possible—without deviating from the base model’s output distribution.

Key terms (paper-specific):
- `n-gram`: a contiguous sequence of n tokens.
- `Jacobi iteration`: a fixed-point method that updates all variables of a system in parallel from their previous values. Here, it provides a lens to parallelize token updates across positions (Sec. 2, “Jacobi Decoding”).
- `Lookahead branch`: the part of the algorithm that produces candidate tokens in future positions by following the recent “trajectory” of token updates (Sec. 3.1).
- `Verification branch`: the exact check (by the base LLM) that determines how many speculative tokens can be accepted at once, preserving the output distribution (Sec. 3.2; Algorithms 3–4).
- Window parameters: `W` (lookahead distance), `N` (n-gram length), `G` (max number of candidates verified per step). See Fig. 1 and Fig. 2.

Step-by-step mechanism
1) Recast decoding as a fixed-point problem (Sec. 2)
   - Standard greedy decoding solves m coupled equations sequentially (Eq. 1).
   - Define f(yi, y1:i−1, x0) = yi − argmax PM(yi | y1:i−1, x0). The full decode is the solution to f = 0 for all i=1..m (Eq. 3).
   - Jacobi decoding updates all yi in parallel from a previous guess y^t−1 to y^t, converging in at most m iterations and sometimes producing multiple correct tokens in one iteration (Algorithm 1; discussion under “Jacobi Decoding”).
   - Limitation: parallel updates put correct tokens in wrong positions and later iterations often overwrite correct ones, so wall-clock speedup is hard to realize directly (Sec. 2, “Limitations of Jacobi Decoding”).

2) Lookahead Decoding’s two-branch design (Sec. 3; Fig. 1)
   - Maintain a fixed-size 2D window over (time steps × token positions). Use the last N−1 steps of history to form many disjoint n-grams that end at future positions up to W tokens ahead (Sec. 3.1). This is the “lookahead branch.”
   - Cache each n-gram in an `n-gram pool` (Sec. 3).
   - In parallel, run a “verification branch” that selects up to G cached n-grams whose first token equals the last generated token and verifies them concurrently using the target LLM (Sec. 3.2; Algorithm 3 for greedy; Algorithm 4 for sampling).
   - Any verified prefix is accepted—i.e., multiple tokens can be appended in one step—while preserving the exact output distribution (proof sketch in Appendix B).

   Intuition with a toy example (Fig. 2b; Fig. 1):
   - Suppose at step t you have tokens from steps t−3, t−2, t−1 forming partial paths. You “look ahead” W positions and, for each position, use the last N−1 tokens from these paths to predict the next token (the blue/red/green/orange stacks in Fig. 2b). These sequences yield many potential 4-grams if N=4 (Sec. 3.1).
   - You then select the n-grams that can legally attach to the current output (they start with the current last token) and verify them all at once with the LLM (Sec. 3.2).
   - Accept as many as pass verification (possibly several tokens at once), then slide the 2D window forward and repeat (Fig. 1, steps 1–4).

3) Running “decode, predict, and verify” in the same forward pass (Sec. 3.3)
   - A custom attention mask enables the two branches to co-exist without illegal information flow (Fig. 2a vs. 2b). Tokens may only attend to positions with a larger position index than themselves (consistent with causality).
   - FlashAttention integration: the attention pattern of Lookahead Decoding is baked into FlashAttention to retain its memory-saving benefits, yielding ~20% end-to-end speedup over a naïve PyTorch implementation (Sec. 3.3).

4) Multi-GPU “Lookahead Parallelism” (LP) (Sec. 3.4; Fig. 3)
   - Observation: lookahead computations form disjoint branches that don’t interact during the forward pass. Assign different branches (and different verification candidates) to different GPUs, each holding a full model replica.
   - Communication is limited to synchronizing accepted tokens after the forward step; during the forward pass, there is near-zero inter-GPU communication (contrast with standard tensor/pipeline parallelism which communicates on the critical path).

5) Cost model and “scaling law” (Sec. 4)
   - Per-step compute scales with the effective number of input tokens processed in that step, roughly proportional to (W + G) × (N − 1) FLOPs (Sec. 5.5; Table 4 discussion).
   - Define `S`, the step compression ratio, as:
     > “S = (#generated tokens) / (#LOOKAHEAD DECODING steps)” (Eq. 6).
   - To relate S to speculative capacity, they model “good speculations” every f steps and use the expected number of accepted tokens per step E(#tokens) to obtain:
     > “S = (f − 1 + E(#tokens)) / f” (Eq. 7).
   - Using an analysis analogous to speculative decoding with b parallel speculations of length γ (Eq. 5; Appendix C), the formulation and empirical curves (Fig. 4a–b) suggest S can grow roughly linearly with log(FLOPs) as W and N increase, highlighting a trade-off between per-step compute and fewer steps.

6) Exactness for sampling (Algorithm 4; Appendix B)
   - Challenge: preserving the output distribution for non-greedy sampling typically requires storing full token distributions. Lookahead solves this by generating n-grams greedily in the lookahead branch and performing a rejection/acceptance process during verification that provably reproduces the target distribution (Appendix B theorem and proof outline).
   - Practically, the `n-gram pool` only stores the chosen tokens, not full distributions, enabling manageable memory.

## 4. Key Insights and Innovations
- Draft-free, exact multi-token parallelism
  - What’s new: Generate and verify many disjoint n-grams using the base LLM itself—no draft model or datastore (Sec. 3; Fig. 1).
  - Why it matters: It avoids acceptance-rate ceilings and generalization issues of draft models (Sec. 1; Sec. 4.1). Gains come from trading unused compute for fewer steps, not from relying on a separate model’s guesses.

- Two-branch, one-step execution with a custom mask
  - What’s new: A single forward pass both produces lookahead n-grams and verifies multiple candidates using an attention mask that preserves causality and isolation between branches (Sec. 3.3; Fig. 2b).
  - Why it matters: Maximizes accelerator utilization without changing the base model, and integrates cleanly with FlashAttention for ~20% additional speedup (Sec. 3.3).

- Distribution-preserving verification with minimal memory
  - What’s new: A sampling verification algorithm (Algorithm 4) that maintains the exact output distribution while only caching greedy tokens in the pool; the correctness is proved (Appendix B).
  - Why it matters: Prior tree-based speculative sampling (e.g., SpecInfer) requires managing many distributions; Lookahead’s disjoint n-gram verification keeps memory modest while staying exact.

- Lookahead Parallelism (LP) for multi-GPU strong scaling
  - What’s new: Partition disjoint lookahead branches and candidate verifications across GPUs that each host a full model copy, avoiding comms on the forward pass (Sec. 3.4; Fig. 3).
  - Why it matters: Achieves up to 4× throughput improvements on code tasks when scaling from 1 to 8 GPUs (Fig. 6–7), unlike conventional TP/PP which often slow down single-batch inference due to communication (Fig. 6–7; “TP/PP bring slowdowns”).

- A simple, actionable scaling law
  - What’s new: A formulation and empirical evidence that step compression S increases with log(FLOPs) (Sec. 4; Fig. 4a–b).
  - Why it matters: Gives operators a principled knob to trade per-step compute for fewer steps, and explains why speedups are better on GPUs with more available compute headroom.

## 5. Experimental Analysis
Evaluation design (Sec. 5; Table 1)
- Models: LLaMA‑2 Chat (7B, 13B, 70B) and CodeLlama variants (7B, 13B, 34B) in FP16, batch size 1 unless noted.
- Hardware:
  - S1: A100 80GB; 70B uses pipeline parallelism across 2 GPUs.
  - S2: DGX with 8× A100 40GB + NVLink.
- Datasets and tasks:
  - MT‑Bench (multi-turn chat; diverse, unique tokens), GSM8K (math QA), HumanEval (code completion and infill), MBPP (instruction-based code), ClassEval (class-level code completion). For code, max generation length is 512 (HumanEval) and 2048 (ClassEval) (Sec. 5; Table 1).
- Baselines: HuggingFace greedy decoding; FlashAttention; and, in multi-GPU, standard tensor parallelism (TP) and pipeline parallelism (PP).

Main quantitative results
- Single-GPU speedups without FlashAttention (Fig. 5):
  - MT‑Bench: 1.45–1.64× (7B–70B).
  - GSM8K: 1.70–1.89×.
  - MBPP: 1.75–1.87×.
  - HumanEval completion: up to 2.25×; infill: ~1.40–1.55×.
  - Pattern: Code completion benefits more (more repetitive tokens), and smaller models see higher gains (compute headroom constraints; Sec. 5.1).
- With FlashAttention and multi-GPU scaling (Fig. 6–7):
  - FlashAttention alone gives ~1.05–1.10× over PyTorch baselines.
  - Lookahead + FlashAttention, 1 GPU:
    - MT‑Bench (7B): 1.9× vs autoregressive + FlashAttention (Fig. 6).
    - HumanEval/ClassEval (7B): ~2.7–2.8× (Fig. 6).
    - For 13B, similar trends with slightly smaller factors (Fig. 7).
  - Lookahead Parallelism (LP) strong scaling:
    - Up to ~3.9–4.0× on HumanEval/ClassEval when going to 8 GPUs (Fig. 6–7).
    - TP/PP slow down single-batch inference (0.71–0.82×), highlighting LP’s advantage (Fig. 6–7).
- Sampling quality (Table 2; Sec. 5.3):
  - CNN/DM and XSum ROUGE scores are essentially unchanged compared to autoregressive decoding at temperature 0.0 and 1.0.
  - Example (CNN/DM, T=1.0): ROUGE‑1 36.55 vs 36.53, ROUGE‑2 13.20 vs 13.27, ROUGE‑L 22.68 vs 22.71; speedups 1.46× and compression ratio S=1.64×.
  - Similar for XSum with ~1.50–1.60× speedups and S~1.67–1.77.
- Ablations (Table 3; Sec. 5.4):
  - Using prompt lookup alone (no lookahead branch) yields 1.44× speedup (S=1.55; row ②).
  - Unbalanced settings suffer: small verification branch (G=1) with large lookahead (W=30) achieves 1.61×/S=1.79 (row ⑦), worse than balanced W=15,G=15,N=5 at 1.78×/S=1.96 (row ⑧).
  - Best reported here: N=5, W=15, G=15 with prompt as reference reaches 1.88× and S=2.05 (row ⑨).
- Accuracy preservation under greedy decoding and advanced supports:
  - Bit-identical behavior holds in FP32 tests; FP16 numerical differences are comparable to HuggingFace’s own FP16 deviations (Appendix E).
  - Adding FlashAttention or LP does not change compression ratio S (Appendix E: ≤0.3% difference with FlashAttention; ≤0.1% with LP).

Do experiments support the claims?
- Yes, for latency/throughput gains in single-batch, latency-sensitive settings:
  - Broadly consistent 1.5–2.3× gains on single GPU (Fig. 5) and strong scaling up to ~4× with LP (Fig. 6–7).
  - Exactness demonstrated via ROUGE parity under sampling (Table 2) and FP32 identity under greedy (Appendix E).
- Conditions and trade-offs are clearly shown:
  - Gains are higher on smaller models and code tasks (Sec. 5.1).
  - TP/PP baselines slow in single-batch; LP helps (Fig. 6–7).
  - Compute-bound regimes or weaker GPUs reduce gains (Fig. 8).

## 6. Limitations and Trade-offs
- Extra per-step compute is required:
  - Per-step FLOPs grow roughly as (W + G) × (N − 1); recommended configurations on A100 (Table 4: 7B→W=15,N=5; 13B→W=10,N=5; 34B→W=7,N=5) imply 56–120× more per-step FLOPs (Sec. 5.5).
  - Works best when decoding is memory-bound and spare compute is available; can underperform in compute-bound scenarios (large batches, smaller GPUs) (Sec. 5.5; Fig. 8).
- Diminishing returns:
  - The scaling law implies step compression increases roughly with log(FLOPs) (Sec. 4; Fig. 4), meaning exponential compute is needed for linear step reduction.
- Memory and engineering complexity:
  - Requires a custom attention mask and FlashAttention integration (Sec. 3.3).
  - The `n-gram pool` grows over time; the verification branch is capped to G candidates to manage cost (Sec. 3.2).
- Multi-GPU memory cost:
  - LP replicates the full model per GPU (not sharded), increasing memory usage versus TP/PP (Sec. 3.4).
- Modeling assumptions:
  - The analytical framework (Sec. 4) assumes an average acceptance behavior and independence akin to speculative decoding; actual acceptance can vary by task/dataset.

## 7. Implications and Future Directions
- How this changes the landscape:
  - Demonstrates that exact multi-token parallel decoding is possible without auxiliary models or data stores. This can become a default, “drop-in” acceleration path for latency-critical inference whenever compute headroom exists.
  - Recasts decoding as parallel fixed-point progress (Sec. 2), encouraging further cross-pollination between numerical methods and inference algorithms.
- Practical applications:
  - Low-latency chat and interactive coding assistants where batch sizes are small and memory bandwidth dominates (e.g., IDE plugins, copilots).
  - Multi-GPU inference clusters favoring minimal communication paths; LP offers strong-scaling throughput improvements (Fig. 6–7).
  - Environments already using FlashAttention can incorporate Lookahead’s custom pattern for additional gains (Sec. 3.3).
- Research directions:
  - Adaptive scheduling: dynamically tune W, N, G based on real-time acceptance or predicted “good speculation” frequency f (Sec. 4).
  - Hybrid methods: combine Lookahead with lightweight draft heads or retrieval to boost candidate quality while keeping exact verification.
  - Better candidate selection: learning-based filters to rank `n-gram pool` entries before verification could raise acceptance per step.
  - Hardware co-design: accelerators that efficiently realize the custom mask and disjoint-branch execution would expand the viable FLOPs envelope.
  - Broader decoding regimes: extend exactness proofs and efficient verification to more sampling schemes and constrained decoding (e.g., grammars, structured outputs).

> Headline results to remember:
> - “Speed up autoregressive decoding by up to 1.8× on MT‑Bench and 4× with strong scaling on multiple GPUs in code completion tasks” (Abstract; Fig. 5–7).
> - “Applying FlashAttention … brings about 20% end-to-end speedup compared to a straightforward implementation” (Sec. 3.3).
> - Exact output distribution preserved under both greedy and sampling (Algorithms 3–4; Table 2; Appendix B/E).

Overall, Lookahead Decoding is a principled, draft-free method to compress decoding steps by leveraging parallel compute that autoregressive decoding leaves on the table, with solid empirical gains where memory bandwidth—not compute—is the bottleneck.
