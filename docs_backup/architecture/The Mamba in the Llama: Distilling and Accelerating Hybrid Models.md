# The Mamba in the Llama: Distilling and Accelerating Hybrid Models

**ArXiv:** [2408.15237](https://arxiv.org/abs/2408.15237)
**Authors:** Junxiong Wang, Daniele Paliotta, Avner May, Alexander M. Rush, Tri Dao
**Institutions:** Unknown (not explicitly listed in sources)

## 🎯 Pitch

This paper introduces an innovative method to convert pretrained Transformer models into faster, efficient Mamba linear RNNs by leveraging existing attention weights and a refined distillation process. Delivering up to 2.6x speedups and improved throughput without sacrificing performance, this approach offers a groundbreaking solution for deploying long-context language models efficiently, crucial for applications requiring high throughput and low memory use.

---

## 1. Executive Summary (2-3 sentences)
This paper shows how to convert a pretrained Transformer language model into a faster linear RNN (Mamba)–based model by reusing the Transformer’s attention weights, then recover the teacher’s chat abilities with a lightweight distillation pipeline. It also introduces a hardware-aware speculative decoding algorithm tailored for Mamba and hybrid Transformer–Mamba models, yielding 1.6–2.6x speedups and up to ~300 tokens/s throughput while preserving quality on standard chat and general benchmarks (see Table 1, Table 2, Table 3).

## 2. Context and Motivation
- Problem addressed
  - Autoregressive Transformers are slow and memory-heavy at long sequence lengths due to quadratic attention cost and the growing key-value (KV) cache. This is a bottleneck for long-context tasks (e.g., multi-document reasoning, codebase navigation) and large-batch agent workflows. See Introduction (pp. 1–2).
  - Linear RNN architectures (e.g., Mamba/Mamba2) can match or beat Transformers at small/medium scales and are much faster at inference, but the strongest open models are still Transformers and linear RNNs trained from scratch at scale have lagged behind on downstream tasks (Introduction; §2.1).

- Why it matters
  - Deployment-centric need: higher throughput and lower memory for long contexts and large batches.
  - Compute reality: most training investment is in Transformers; replacing them wholesale is impractical. Making an already-trained Transformer run like a linear RNN would deliver immediate inference benefits.

- Prior approaches and gaps
  - Linear RNNs/SSMs (e.g., Mamba, RetNet, RWKV) show attractive scaling, but top performance still comes from large Transformers; training Mamba-style models from scratch to parity needs trillions of tokens (Introduction; §5).
  - Distilling Transformers into other sub-quadratic architectures has been explored (e.g., Hyena; Related Work §7), but either at small scales or with noticeable performance degradation (Table 6, left).

- How this work positions itself
  - It proposes a direct, weight-reusing mapping from Transformer attention heads to a Mamba block (§2.2–§2.3; Figure 1; Algorithm 1), then a multistage distillation pipeline (§3) that reconstructs instruction-following and alignment.
  - It develops a speculative decoding method that is specifically adapted to RNN/SSM constraints (§4; Figure 2; Algorithm 2), enabling real speedups even on modern GPUs (Table 1; Figure 3).

## 3. Technical Approach
The paper has two technical pillars: (A) distilling attention into Mamba to form a hybrid or pure linear-RNN model, and (B) a speculative decoding algorithm that fits Mamba’s execution model.

A) Distilling attention into a linear RNN (Mamba)

1) Start from a principled link between attention and linear RNNs (§2.1)
   - If you “linearize” attention by dropping the softmax, the output at time t can be written as a dot-product between current query and a running sum over past key–value products. In symbols:
     - The RNN form (Equation (1)): h_t = A_t h_{t-1} + B_t x_t; y_t = C_t^T h_t.
     - Linearized attention corresponds to choosing A_t as a causal mask scalar m_{t-1,t}, B_t as the key projection applied to the current token, C_t as the query projection applied to the current token, and x_t as the value projection (all defined via the original attention’s W_K, W_Q, W_V; see §2.1 for the mapping).
   - Problem: this naïve linearization has too little capacity (hidden state is only size N per head) and performs poorly without softmax (§2.1).

2) Expand capacity using Mamba’s state-space parameterization (§2.2)
   - Mamba parameterizes a continuous-time state-space model (SSM): h′(k) = A h(k) + B(k) x(k), y(k) = C(k) h(k), then discretizes it at runtime to get the RNN (Discretize in Algorithm 1; see §2.2).
   - Key idea: keep the Transformer-derived B_t and C_t (from K and Q), and expand the hidden state by a factor N′ (per head) via SSM discretization without explicitly materializing huge matrices. The expansion is controlled by learned A (diagonal) and learned step sizes Δ_t produced by a small MLP (Algorithm 1 lines 13–15; “Disc” function). This increases modeling capacity with minimal parameter overhead and efficient kernels (§2.2).

3) Attention-to-Mamba initialization and hybridization (§2.3; Figure 1; Algorithm 1)
   - Map attention head projections directly into a Mamba block:
     - Use W_Q → C_t, W_K → B_t, W_V → x_t; learn A and Δ (Figure 1; Algorithm 1 lines 10–16).
     - Keep the original Transformer MLP blocks (feedforward layers) initially frozen, and replace some or all attention layers with Mamba blocks (hybrid). Shapes and heads are kept similar to the Transformer head structure (§2.3; Figure 1).
   - Stepwise replacement (progressive hybridization):
     - Empirically best: replace attention layers in stages—first keep every 2nd attention layer, distill; then keep every 4th layer, distill; etc. (§2.3).

4) Distillation pipeline redoing post-training at low cost (§3)
   - Distillation focuses on re-learning SFT and preference alignment on top of the transferred MLPs and newly inserted Mamba blocks; initially freeze MLPs (§3; training details §5.1).
   - Supervised fine-tuning with distillation (Equation (2)):
     - Combine sequence-level KD (using teacher-generated pseudo-label responses) with token-level KL between teacher and student distributions. Loss L = α·NLL on teacher outputs + β·KL(teacher || student) (§3; Eq. (2)).
   - Preference optimization by DPO as a distillation objective (Equations (3)–(4)):
     - Instead of RL, optimize directly on preference pairs (y_w preferred over y_l) with a student–teacher ratio that keeps the student close to the teacher (Eq. (4)). The paper asserts this is the first use of DPO for distilling aligned behavior (§3; Eq. (3)–(4)).

B) Speculative decoding adapted to Mamba (§4; Figure 2; Algorithm 2)

1) Challenge with RNNs
   - Speculative decoding uses a small “draft” model to propose several next tokens and a large “verifier” model to accept or reject them in batches. Transformers verify drafts in parallel using the KV cache. RNNs do not have a KV cache; they have one big hidden state h_t, so “rewinding” to t′ during verification would require caching many h’s or recomputing (§4.1).
   - RNN training-time parallel modes avoid materializing intermediate states; naively using them for short speculative bursts is inefficient (§4.1).

2) Multi-step verification kernel that avoids large caches (§4.2; Figure 2; Algorithm 2; Figure 3)
   - Core primitive: MultiStep takes one cached state h_i and a run of tokens y_{j:k}, and returns the verifier logits y_{j:k} plus the hidden states at j and k without materializing every intermediate state in GPU memory:
     - y_{j:k}, h_j, h_k ← MultiStep(h_i, y_{1:n}, i, j, k; A, B, C, Δ) (§4.2).
   - Algorithm 2 maintains only a single cached RNN state, reuses it to verify each speculative chunk, and advances or rewinds by recomputing just the necessary part. This matches Mamba’s “compute-heavy, memory-light” sweet spot (§4.2).
   - For hybrid models, Transformer layers verify in parallel as usual, and Mamba layers verify with the multi-step kernel (§4.2).

3) Hardware optimizations for speedups on modern GPUs (§4.3; Table 1; Figure 3)
   - Fused kernels reduce overhead: recomputation, multi-step decoding, and caching are fused for both draft and verifier models. This is important on H100 GPUs where GEMMs are so fast that launch/caching overheads otherwise dominate (§4.3).
   - Result: consistent speedups on both RTX 3090 and H100 for pure Mamba verifiers (Table 1), and additional speedups for hybrid verifiers (Table 5).

## 4. Key Insights and Innovations
- Weight-reuse initialization from attention into Mamba heads (fundamental innovation)
  - Novelty: directly bootstrap Mamba blocks from the teacher’s Q/K/V projections, then expand capacity via SSM discretization (Figure 1; Algorithm 1). This bridges architectures with minimal new parameters.
  - Significance: drastically reduces compute required to “convert” a trained Transformer into a fast linear RNN or hybrid, without pretraining from scratch (§2.2–§2.3; §5.1).

- Stepwise hybrid distillation that keeps some attention layers (pragmatic yet impactful)
  - Novelty: progressively replace attention layers and distill at each step (§2.3). Freeze MLPs initially to preserve knowledge and learn the new attention→Mamba interactions (§3).
  - Significance: 50% attention hybrids closely match teacher quality; even 25% attention remains competitive, while 0% attention degrades (Table 2–Table 4, Table 6). This provides a tunable quality–speed knob.

- Using DPO as a distillation alignment objective (methodological innovation)
  - Novelty: reformulate alignment as distillation with a teacher-conditioned DPO loss (Equations (3)–(4)), avoiding RL and stabilizing training (§3).
  - Significance: improved downstream results over SFT-only or DPO-only; best when combined (Table 6, right).

- Speculative decoding for linear RNNs via a multi-step kernel (systems innovation)
  - Novelty: verification that recomputes only the needed states without materializing intermediate hidden states, fused into hardware-friendly kernels (Figure 2; Algorithm 2; Figure 3).
  - Significance: 1.6–2.6x end-to-end speedups, up to ~300 tok/s for 7B on H100, with both pure Mamba and hybrid models (Table 1; Table 5).

- Emergent length extrapolation after distillation (empirical insight)
  - Observation: despite distilling on sequences of length ~2k, distilled models retrieve perfectly in Needle-in-a-Haystack up to 10k (3B) and 16k (8B), with one 8B hybrid performing well to ~38k (Figure 4). This suggests the Mamba hybrids inherit and even enhance long-context abilities.

## 5. Experimental Analysis
- Evaluation methodology and setup (§5.1)
  - Target teachers: Zephyr-7B (chat-tuned Mistral 7B) and Llama-3/3.1/3.2 Instruct 8B/3B.
  - Students: pure Mamba and hybrids (Mamba/Mamba2 with 50%, 25%, 12.5%, 0% attention).
  - Distillation data and stages:
    - Stage 1 KD on teacher-generated data from UltraChat and UltraFeedback using combined loss in Eq. (2); freeze MLPs.
    - Stage 2 SFT on GenQA, InfinityInstruct, OpenHermes 2.5; unfreeze all params (§5.1).
    - Stage 3 distilled preference optimization with DPO using UltraFeedback/SimPO datasets (§3; §5.1).
  - Compute: Each hybrid model distilled with ~20B tokens; ≤5 days on 8×80G A100 for the Zephyr/Llama-3 8B setting (§5.1). For Llama-3.1/3.2 sets, 8 days on 8×A100 or 4 days on 8×H100 (§5.1).

- Benchmarks and metrics
  - Chat: MT-Bench (GPT-4 judged), AlpacaEval v2 length-controlled and overall win rates (Table 2).
  - General tasks: LM Eval Harness (10 tasks; accuracy/normalized accuracy) with an overall average (Table 3).
  - Open LLM Leaderboard + ZeroEval for ARC-Challenge, HellaSwag, MMLU, Winogrande, TruthfulQA, GSM8K, CRUX (Table 4).
  - Long context: Needle-in-a-Haystack (Figure 4).
  - Speed: Pure Mamba speculation (Table 1; Figure 3) and hybrid speculation (Table 5).

- Main quantitative results
  - Chat parity (or better) with 50% attention hybrids:
    - Llama-3 8B teacher vs distilled hybrids (Table 2):
      - MT-Bench: teacher 8.00; `Mamba-Llama3 (50%)` 7.35; `Mamba2-Llama3 (50%)` 7.32.
      - AlpacaEval LC-win%: teacher 22.90; `Mamba-Llama3 (50%)` 29.61; `Mamba2-Llama3 (50%)` 26.78.
      - Lower attention ratios reduce quality; pure Mamba (`0%`) drops to MT-Bench 5.64 and LC-win 14.49 (Table 2).
    - Zephyr-7B teacher vs hybrids (Table 2):
      - MT-Bench: teacher 7.34; `Mamba-Zephyr (50%)` 7.31.
      - AlpacaEval LC-win%: teacher 13.20; `Mamba-Zephyr (50%)` 20.66.
  - General tasks competitiveness and sometimes surpassing teachers or from-scratch SSM baselines (Table 3):
    - `Mamba2-Llama3.1 (50%) DPO` average 65.31 vs `Llama-3.1-8B-Instruct` average 64.48.
    - Both hybrid families outperform TRI Mamba-7B (57.65 avg) and NVIDIA Hybrid Mamba-8B (59.60 avg), despite far less training data (Table 3).
  - Open Leaderboard and ZeroEval (Table 4):
    - On GSM8K (zero-shot chat-style evaluation), `Mamba-Llama3 (50%)` 67.85 and `Mamba2-Llama3 (50%)` 59.36, beating `RecurrentGemma-9B-it` (38.51) and `Falcon Mamba-7B-instruct` (41.32). CRUX scores are also competitive.
  - Long-context retrieval beyond distillation length (Figure 4):
    - 3B hybrids perfect to 10k; 8B hybrids perfect to 16k; one 8B hybrid remains strong up to ~38k.
  - Speedups with speculative decoding:
    - Pure Mamba verifiers (Table 1): 2.8B verifier on 3090 up to 2.6x, H100 up to 1.85x; 7B verifier on 3090 ~2.1x, H100 ~2.0x, with throughput up to ~421 tok/s (2.8B, H100) and ~272 tok/s (7B, H100).
    - Hybrids (Table 5): `Mamba-Zephyr (50%)` 1.8x with a 2-layer draft at K=4; `Mamba-Zephyr (25%)` 1.88x; `Mamba-Llama3 (50%)` ~1.6x. Larger draft models increase acceptance but add overhead.
  - Ablations that justify design choices:
    - Attention initialization is critical: without it, perplexity and downstream metrics collapse (Table 7 left; Table 8).
      > LAMBADA ppl 6.20 vs 55.01; MT-Bench 6.69 vs 1.04; AlpacaEval LC-win 14.11% vs 0.02% (Table 8).
    - Mamba blocks are necessary: removing them (keeping only surviving attention layers) fails badly (Table 9).
    - Freezing MLPs during the first stage helps the student focus on learning new attention→Mamba dynamics (Table 7 left; “Froz” vs “-Froz”).
    - Stepwise interleaving and progressive replacement give lower perplexity (Table 7 right).
    - Combining SFT + DPO yields the best chat results (Table 6 right).

- Do the experiments support the claims?
  - Yes, for the intended scope:
    - Quality: 50% hybrids closely track or exceed teacher chat metrics on AlpacaEval, and match or narrowly trail on MT-Bench (Table 2).
    - Efficiency: measurable speculative decoding speedups for both pure Mamba and hybrids across GPUs (Table 1, Table 5), with kernel-level analysis (Figure 3).
    - Robustness checks: multiple teachers (Zephyr, Llama-3/3.1/3.2), scales (3B, 8B), and a range of benchmarks (Table 2–4). Ablations isolate each design contribution (Table 6–9).
  - Caveats:
    - Pure Mamba (0% attention) loses too much quality under this recipe (Table 2–3).
    - Acceptance rates and end-to-end speedups depend on the draft model size/quality (Table 5).

## 6. Limitations and Trade-offs
- Quality vs. “attention removed”
  - Pure linear-RNN students underperform: `0% attention` plainly degrades (Tables 2–3). The recipe currently works best as a hybrid.

- Dependence on good initialization
  - Reusing Q/K/V projections is essential; without it, distillation does not converge to competitive performance (Table 8). This assumes access to teacher weights in compatible formats (e.g., head shapes, grouped query attention).

- Distillation scope and data
  - The approach redoes post-training (SFT + alignment) rather than full pretraining. This is its strength (low compute) but means the student’s knowledge is bounded by what the MLPs and teacher outputs transfer during these stages (§3; §5.1).
  - Training used ~20B tokens and specific instruction/feedback datasets; different domains or safety datasets may need re-tuning (§5.1).

- Speculative decoding practicalities
  - Speedups vary with draft quality and kernel engineering; H100s required fused kernels for gains (§4.3).
  - Implementations rely on specialized kernels (multi-step verification, circular buffers for Mamba’s convolutional part) and careful memory management (§4.3, footnote 1).

- Long-context evaluation scope
  - Needle-in-a-Haystack is a synthetic retrieval stress test; broader long-context reasoning tasks (e.g., multi-document Q&A with distractors) are not extensively reported here (Figure 4).

- Scaling and generality
  - Results are strong at 3B–8B. Applying the method to much larger teachers/students or to non-Llama/Mistral families is promising but not yet shown.

## 7. Implications and Future Directions
- Changing the deployment calculus
  - This work demonstrates a viable “convert-and-tune” path: take a strong Transformer, reuse its attention projections to initialize Mamba blocks, distill alignment, and deploy a hybrid with sizable speed and memory benefits—without retraining from scratch (§2.3; §3; Table 2–5). This opens a practical route to customize the inference profile of existing LLMs.

- Research enabled by this bridge
  - Toward attention-free LLMs: Can the hybrid ratio be pushed further (e.g., <25% attention) with additional techniques—better kernels, stronger distillation curricula, or softmax-mimicking expansions (cf. §2.2 and Related Work [kernelized approximations])?
  - Distillation objectives: DPO worked well as a distillation objective (Eq. (4), Table 6 right). Exploring other preference- or constraint-aware distillation losses (safety, factuality, tool-use) is a clear next step.
  - Draft models for hybrids: Smaller yet high-acceptance draft models (e.g., layer-drop, knowledge-distilled drafts, or training-free drafting) could unlock larger speedups (Table 5; §4.3).
  - Long-context training signals: The observed length extrapolation (Figure 4) suggests Mamba hybrids can internalize long-range mechanisms even when distilled at short contexts; combining with retrieval or memory modules could expand capabilities further.
  - Systems integration: The multi-step kernel complements quantization, tensor parallelism, and batching strategies; combining them systematically could yield further throughput gains.

- Practical applications
  - Long-context assistants (legal/medical docs, codebase reasoning), agentic systems with large batch decoding, and on-device or edge deployments where KV cache size and latency are critical.
  - Enterprises with existing Transformer checkpoints can adopt this pipeline to reduce serving costs while retaining chat quality.

> Bottom line: The paper delivers a concrete recipe—weight-reuse initialization + staged hybrid distillation + RNN-suitable speculative decoding—that transfers Transformer capabilities into a faster linear-RNN form. The 50% hybrid matches or exceeds teacher quality on many chat and general tasks (Table 2–3) and achieves meaningful inference speedups (Table 1, Table 5), offering a practical path to efficient LLM deployment.
