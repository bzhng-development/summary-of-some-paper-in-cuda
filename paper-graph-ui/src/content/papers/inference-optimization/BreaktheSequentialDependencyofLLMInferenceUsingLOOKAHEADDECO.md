# Break the Sequential Dependency of LLM Inference Using LOOKAHEAD DECODING

**ArXiv:** [2402.02057](https://arxiv.org/abs/2402.02057)

## 🎯 Pitch

This paper presents LOOKAHEAD DECODING, a novel parallel decoding algorithm that dramatically speeds up large language model (LLM) inference by generating and verifying multiple candidate n-grams in parallel—without needing an external draft model or altering model output distributions. By unlocking the compute potential of modern accelerators and breaking the traditional step-by-step bottleneck, this method slashes response latency, delivering up to 1.8× faster inference on a single GPU and scaling to nearly 4× with multi-GPU setups—crucial for powering next-generation conversational AI and low-latency LLM applications.

---

## 1. Executive Summary (2-3 sentences)
This paper introduces LOOKAHEAD DECODING, a lossless, parallel decoding algorithm that accelerates large language model (LLM) inference without using a separate draft model or changing the output distribution. By generating and verifying multiple future n-grams in parallel within each decoding step, it reduces the number of steps required, achieving up to 1.8× speedup on MT-Bench with a single GPU and up to ~4× with multi-GPU strong scaling on code tasks (Figures 6–7).

## 2. Context and Motivation
- Problem addressed:
  - Autoregressive decoding (generating one token at a time conditioned on all previous tokens) is memory-bandwidth-bound and underutilizes modern accelerators’ compute, causing high latency (Abstract; §1).
  - Two core inefficiencies: it produces only one token per step, and each step underutilizes GPU compute because the attention pattern makes inference I/O-bound (§1, “However, current LLMs…”).

- Why this matters:
  - Low-latency generation is critical for chatbots, search, and code completion (§1). Reducing end-to-end latency at batch size 1 (common in interactive settings) improves user experience and enables new applications.

- Prior approaches and their gaps:
  - Speculative decoding accelerates decoding using a cheaper draft model to propose tokens, then verifies them with the base model (Eq. 2; §2 “Guess-And-Verify Paradigm”).
  - Limitations:
    - Speedup is capped by the acceptance rate α (fraction of draft tokens accepted) and can degrade when α is low (§1; §4.1, Eq. 4).
    - Draft models require training, do not generalize across base models/datasets, and complicate deployment (§1).

- Positioning of this paper:
  - Develops an exact method (no approximation in token distribution) that needs no auxiliary model or datastore.
  - Exploits the observation that decoding can be reinterpreted as solving a nonlinear system via Jacobi iteration (§2, Eq. 3; “Jacobi Decoding”), then harvests parallelizable structure from that view.
  - Scales with compute: shows a scaling law linking fewer decoding steps to per-step log(FLOPs) (§4.2).

## 3. Technical Approach
The method rethinks decoding as a parallel process that generates multiple future n-grams and verifies them in the same forward pass.

- Key terms (defined when first used):
  - `n-gram`: a contiguous sequence of `n` tokens.
  - `Jacobi iteration`: an iterative method for solving systems of equations where all variables are updated in parallel using values from the previous iteration.
  - `Acceptance rate` α: the probability that a proposed next token is the one the base model would produce (used in speculative-style verification).
  - `W`, `N`, `G`: algorithm controls — lookahead window size (`W` future positions), n-gram length (`N` tokens), and maximum candidates to verify per step (`G`) (§3.1–§3.2).

Step-by-step mechanism:

1) Reformulate decoding to expose parallelism (§2)
- Standard greedy autoregressive decoding solves a chain of `m` problems, each picking the next token from the model’s conditional distribution (Eq. 1).
- Define a nonlinear system f(y) = 0 where each equation encodes that the chosen token equals the model’s argmax at that position given the prior tokens (Eq. 3).
- Jacobi decoding updates all positions simultaneously from a previous “trajectory” y^(t−1) to y^(t) (Algorithm 1). Although it can generate multiple tokens per iteration, many end up in wrong positions and get overwritten later (§2 “Limitations of Jacobi Decoding”).

2) Core idea: look ahead along the Jacobi trajectory and cache n-grams (§3; Fig. 1)
- Maintain a fixed-size 2D window across time (previous Jacobi iterations) and sequence (future positions). At each new step `t`, use tokens from the previous `N−1` steps and predict `W` new tokens — one at each of the next `W` positions — in parallel (§3.1).
- From each “diagonal” across the past `N−1` steps plus the new prediction, form disjoint `N`-grams and store them in an `n-gram pool` for later verification (Algorithm 2, lines 27–31; Fig. 2b example with `W=5, N=4`).

3) Verification to keep the exact output distribution (§3.2)
- Search the `n-gram pool` for up to `G` “promising” candidates that start with the model’s last generated token (Algorithm 2, lines 21–26).
- Verify candidates in parallel, like speculative decoding:
  - Greedy case: accept a token if the base model’s argmax at that position equals the candidate; stop at first mismatch (Algorithm 3). This preserves exact greedy outputs (§3.2; Appendix E shows FP32 equality to HF greedy on MT-Bench).
  - Sampling case: progressively accept/reject each candidate token using the base model’s probabilities, re-normalizing when a candidate is rejected (Algorithm 4). The method stores only the sampled token per `n-gram` by enforcing greedy selection during n-gram generation, avoiding storage of full distributions (§3.2). Appendix B proves the output distribution equals that of the base model.

4) Do “decode, predict, and verify” in one forward pass (§3.3)
- Use a custom attention mask so tokens in the lookahead branch only see allowed past tokens, and tokens in the verification branch only see the ongoing prefix they are verifying (Fig. 2b).
- Integrate with `FlashAttention` by hard-coding the new attention pattern with adjustable `W`, `N`, `G` (§3.3). This yields ~20% extra end-to-end speedup over a PyTorch implementation (Fig. 6–7; §5.2).

5) Scale across GPUs with Lookahead Parallelism (LP) (§3.4)
- Replicate the entire model on each GPU (data-style parallelism over tokens), assign disjoint lookahead branches and verification candidates to devices, and avoid inter-device communication during the forward pass. Only synchronize accepted tokens afterward (Fig. 3).
- Unlike tensor or pipeline parallelism, LP keeps communication off the critical path of each decoding step (§3.4).

6) Tuning knobs and compute/latency trade-off (§4)
- Define `S` (step compression ratio) = number of AR steps divided by number of LOOKAHEAD steps (Eq. 6).
- With batched speculations of size `b=G=W` and speculation length `γ=N−1`, the expected accepted tokens per “good” step follows the speculative-style formula with batch (Eq. 5).
- Assume only 1 out of every `f` steps finds a “good” speculation while the rest fall back to AR, giving:
  - S = (f − 1 + E(#tokens)) / f (Eq. 7).
- Since per-step FLOPs scale roughly with `(W + G) * (N−1)`, they argue fewer steps can be obtained roughly linearly with log(FLOPs) for large enough `N` (§4.2; Fig. 4).

Design choices and rationale:
- Keep a fixed-size window: bounds memory footprint and stabilizes throughput (§3.1).
- Use greedy generation in the lookahead branch so only token IDs (not full distributions) need to be cached, enabling sampling verification without large memory (Algorithm 4; §3.2).
- Set `G ≈ W` to balance generation and verification (§3.2).
- Custom mask + FlashAttention: reclaims otherwise idle compute under memory-bound decoding (§3.3).

## 4. Key Insights and Innovations
- Parallel n-gram speculation from a Jacobi trajectory (fundamental)
  - Novelty: reframes decoding as parallel fixed-point updates and mines the trajectory to build many disjoint, verifiable `n`-grams in every step (Fig. 1; §3.1).
  - Significance: avoids the need for a draft model while still proposing many future tokens, using compute that is otherwise idle in memory-bound decoding.

- Unified “decode, predict, verify” in one pass with a custom attention mask (system/algorithmic)
  - Novelty: a single forward pass contains both the lookahead and verification branches without cross-visibility (Fig. 2b; §3.3).
  - Significance: enables effective parallelism on a single GPU and compatibility with FlashAttention (§3.3), yielding ~20% additional speedup (§5.2).

- Lookahead Parallelism (LP) for multi-GPU strong scaling (systems)
  - Novelty: token-distribution parallelism where each GPU holds a full model replica and computes disjoint branches independently (Fig. 3; §3.4).
  - Significance: achieves near-linear strong scaling for latency-sensitive single-batch inference, unlike tensor/pipeline parallelism which incurs communication on the critical path (§3.4; Figures 6–7).

- Output-distribution-preserving verification for disjoint n-grams, including sampling (theoretical + practical)
  - Novelty: extends speculative-style verification to a set of disjoint n-grams while storing only tokens (not distributions) by forcing greedy lookahead generation (Algorithms 3–4).
  - Significance: maintains exact distribution under sampling; Appendix B provides a proof. Table 2 shows unchanged ROUGE scores alongside speedups.

- Scaling law linking step reduction to per-step log(FLOPs) (analytical insight)
  - Novelty: Eq. 7 connects `S` to the expected accepted tokens and the frequency of good speculations; combined with Eq. 5 (batched acceptance) and the fact FLOPs ∝ `(W+G)(N−1)`, this predicts linear step reduction vs. log(FLOPs) (§4.2; Fig. 4).
  - Significance: clarifies when more compute translates to lower latency and motivates multi-GPU scaling.

## 5. Experimental Analysis
- Evaluation setup (§5; Table 1)
  - Models: `LLaMA-2` (7B, 13B, 70B), `CodeLlama` (7B, 13B, 34B), `CodeLlama-Python` (7B, 13B).
  - Hardware:
    - S1: single A100 80GB for 7B/13B/34B; 70B uses 2×A100 with pipeline parallelism (§5).
    - S2: DGX with 8×A100 40GB + NVLink, used to evaluate LP and FlashAttention (§5.2).
  - Datasets/tasks:
    - MT-Bench (multi-turn chat), GSM8K (math), MBPP (instruction code gen), HumanEval (completion + infill), ClassEval (class-level completion) (§5; Table 1).
    - Summarization (CNN/DailyMail, XSum) for distributional quality under sampling (Table 2).
  - Baselines:
    - HuggingFace greedy; variants with FlashAttention. For distributed: TP (DeepSpeed), PP (Accelerate) (§5).

- Main results (throughput in tokens/s, speedup vs. baseline)
  - Single-GPU end-to-end (no FlashAttention/LP): Figure 5.
    - MT-Bench: 7B 1.64×, 13B 1.51×, 70B 1.45×.
    - GSM8K: 7B 1.89×, 13B 1.72×, 34B 1.70×.
    - MBPP: 7B 1.87×, 13B 1.75×, 34B 1.76×.
    - HumanEval (completion): 7B 2.25×, 13B 2.26×, 34B 1.72×.
    - HumanEval (infill): 7B 1.55×, 13B 1.40×.
    - Interpretation: larger gains on code completion tasks (more repetition → higher acceptance; §5.1).
  - With FlashAttention and LP (multi-GPU strong scaling): Figures 6–7.
    - For 7B:
      - MT-Bench: single-GPU FA lookahead 1.90×; LP+FA scales to 2.05× on 8 GPUs (Fig. 6).
      - HumanEval: LP+FA reaches 3.87× on 8 GPUs (Fig. 6).
      - ClassEval: LP+FA reaches 3.99× on 8 GPUs (≈4×; Fig. 6).
    - For 13B:
      - MT-Bench: single-GPU FA lookahead 1.67×; LP+FA up to 1.97× (8 GPUs; Fig. 7).
      - HumanEval: LP+FA up to 3.42× (8 GPUs; Fig. 7).
      - ClassEval: LP+FA up to 3.79× (8 GPUs; Fig. 7).
    - Contrast with TP/PP: both slow down at single-batch due to communication (0.71–0.82× across tasks; Figures 6–7), confirming LP’s advantage.
    - FlashAttention adds ~20% on top of PyTorch lookahead (§5.2).
  - Quality under sampling (Table 2):
    - CNN/DailyMail (temperature 1.0): ROUGE-1/2/L essentially unchanged (36.55/13.20/22.68 AR vs. 36.53/13.27/22.71 LA); speedup 1.46×; S=1.64×.
    - XSum (temperature 1.0): ROUGE-1/2/L unchanged (19.15/4.53/12.84 AR vs. 19.20/4.53/12.87 LA); speedup 1.50×; S=1.67×.
    - At temperature 0.0 (greedy), scores are identical; speedups increase slightly (1.57–1.60×).
    - Appendix E further shows FP32 identity with HF greedy and negligible differences with FA/LP.
  - Ablations (Table 3; MT-Bench, LLaMA-2-7B-Chat, FA on):
    - Prompt lookup baseline (no lookahead window): 1.44×, S=1.55×.
    - Minimal lookahead with `W=1` and large `G` gives limited gains (e.g., (5,1,30) without prompt-as-reference is 1.04×).
    - Heavy lookahead with tiny verification (5,30,1) = 1.61× shows verifying capacity matters.
    - Balanced branches perform best: (5,15,15) without prompt-as-reference 1.78× (S=1.96); adding prompt-as-reference rises to 1.88× (S=2.05).
    - Takeaway: both lookahead breadth and verification bandwidth are necessary; prompt-as-reference can further boost candidate quality (§5.4).

- Do the experiments support the claims?
  - Yes for single-batch latency: consistent 1.5–2.3× single-GPU gains (Fig. 5) and strong scaling to ~4× with LP+FA on code tasks (Fig. 6).
  - Quality preserved: unchanged ROUGE on summarization (Table 2) and FP32 identity with greedy (Appendix E). Sampling correctness is supported by the proof in Appendix B.

- Notable patterns and conditions:
  - Gains are larger on code tasks (more repetitive patterns increase acceptance; Fig. 5; §5.1).
  - Speedups diminish on larger models at fixed hardware because per-step FLOPs saturate GPU compute sooner (§5.1).
  - LP outperforms TP/PP for latency at batch size 1; TP/PP incur communication overhead (Figures 6–7).

> “FlashAttention-integrated LOOKAHEAD DECODING shows 1.8× speedups for the 7B model on MT-Bench… strong scaling to multiple GPUs reaches ~4× on ClassEval” (Figures 6 and 7; §5.2).

## 6. Limitations and Trade-offs
- Extra compute per step:
  - Per-step FLOPs scale with `(W + G) * (N−1)` (§5.5). Recommended single-GPU A100 configs imply 56–120× extra per-step FLOPs for 34B→7B models (Table 4).
  - Since decoding is memory-bandwidth-bound, many of these FLOPs are “free” on A100 at batch size 1, but not on smaller GPUs or when the workload becomes compute-bound (Fig. 8 and §5.5).

- Diminishing returns:
  - The step compression `S` grows only linearly with log(FLOPs) for sufficiently large `N` (§4.2). Achieving further reductions requires exponentially more per-step compute.

- Hardware and workload dependence:
  - Gains shrink on devices with less compute headroom (e.g., RTX 3090 vs. A100; Fig. 8).
  - Large batches (compute-bound regimes) or very large models on limited hardware can reduce or negate gains (§5.5).

- Implementation complexity:
  - Requires a custom attention mask and modifications to FlashAttention (§3.3), which increases engineering effort.
  - Multi-GPU LP needs full model replication on each GPU, raising memory requirements (§3.4).

- Candidate availability:
  - While the `n-gram pool` grows over time, actual acceptance depends on task structure; domains with fewer local repetitions (e.g., creative open-ended chat) yield smaller speedups than code completion (Fig. 5).

- Search/verification caps:
  - A cap `G` is needed to limit verification cost; setting `G` too small underutilizes good candidates, too large increases per-step compute (§3.2, §5.4).

Open questions:
- How to dynamically tune `W`, `N`, `G` per prompt or per step to optimize acceptance vs. compute?
- How large can the `n-gram pool` grow in long generations, and what is the best policy for pool management beyond windowing?

## 7. Implications and Future Directions
- Field impact:
  - Establishes a new family of “model-only” parallel decoding methods that preserve exact output distribution and do not require draft models or datastores. The Jacobi-trajectory view could inspire other parallelization strategies for sequence models (§2–§3).

- Practical applications:
  - Low-latency, single-batch inference for chat assistants, code completion, and search.
  - Multi-GPU serving for latency-sensitive endpoints: LP provides strong scaling without inter-step communication (Figures 6–7).
  - Deployments already compatible with FlashAttention (§3.3), making integration into modern inference stacks feasible.

- Follow-up research directions:
  - Adaptive control: learn policies to set `W`, `N`, `G` online based on confidence/entropy to maximize `S` under compute budgets.
  - Pool quality: smarter n-gram retrieval (beyond exact-start matching) or lightweight scoring to prioritize high-acceptance candidates, while maintaining exactness through verification.
  - Synergy with other accelerations: combine with quantization, KV-cache optimizations, or even draft-model speculative decoding to blend their advantages.
  - Generalization: extend the attention-mask idea to other architectures (e.g., MoE, state-space models) or to multilingual/tokenization regimes where n-gram statistics differ.
  - Systems: standardize attention pattern APIs in FlashAttention-like libraries to lower adoption cost; explore heterogeneous scheduling across GPUs/NPUs.
  - Theory: refine the scaling law by modeling acceptance α as a function of domain/model/prompt and deriving optimal `W,N,G` under resource constraints.

> Overall, LOOKAHEAD DECODING shows that the sequential dependency in autoregressive decoding can be relaxed—without approximation—by precomputing and verifying multiple future n-grams per step. The method turns otherwise idle compute into latency reduction, and its benefits grow with available FLOPs, especially under multi-GPU strong scaling.
