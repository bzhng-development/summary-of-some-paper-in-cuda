# Scaling Latent Reasoning via Looped Language Models

**ArXiv:** [2510.25741](https://arxiv.org/abs/2510.25741)
**Authors:** Rui-Jie Zhu, Zixuan Wang, Kai Hua, Tianyu Zhang, Ziniu Li, Haoran Que, Boyi Wei, Zixin Wen, Fan Yin, He Xing, Lu Li, Jiajun Shi, Kaijing Ma, Shanda Li, Taylor Kergan, Andrew Smith, Xingwei Qu, Mude Hui, Bohong Wu, Qiyang Min, Hongzhi Huang, Xun Zhou, Wei Ye, Jiaheng Liu, Jian Yang, Yunfeng Shi, Chenghua Lin, Enduo Zhao, Tianle Cai, Ge Zhang, Wenhao Huang, Yoshua Bengio, Jason Eshraghian
**Institutions:** Not specified in the arXiv entry

## 1. Executive Summary

Reasoning behind this summary: The paper’s central idea is to move “thinking” from explicit text at inference time into the model’s latent computation during pre‑training. This is done by looping shared transformer blocks and learning when to stop iterating, so small models can spend more internal compute on hard inputs without growing parameter count.

Core contribution and significance:
- It introduces `Ouro`, a family of Looped Language Models (`LoopLM`) that build iterative latent reasoning into pre‑training via parameter‑shared recurrence and a learned early‑exit gate with entropy regularization (Figure 3; Sections 3.1–3.4).
- Trained on 7.7 trillion tokens (Figure 4; Section 4), Ouro‑1.4B and Ouro‑2.6B match or exceed 4B–8B dense transformers across diverse reasoning benchmarks (Figures 1–2; Tables 7–9), while offering improved safety alignment and more causally faithful internal reasoning (Sections 7.1–7.2; Figures 8–9). This matters because it provides a third scaling axis—recurrent depth—yielding 2–3× parameter efficiency for real‑world deployments.

## 2. Context and Motivation

Reasoning behind the context: The field has mainly scaled parameters, data, and inference‑time chain‑of‑thought (CoT). This paper asks whether architectural recurrence can be a more efficient way to scale computation without parameter bloat, and whether that holds in the multi‑trillion token regime.

- Problem/gap addressed
  - Most LLMs “think” by generating explicit CoT at inference, which adds tokens and latency and leaves pre‑training under‑utilized for reasoning (Abstract).
  - Prior looped or recursive transformers showed promise on smaller scales [7–16], but there is little evidence these benefits persist at practical scales (Section 1).
  - Key questions posed (Section 1):
    1) Does weight‑shared recurrence improve capabilities akin to adding more unshared layers?
    2) Are gains monotonic in loop count, and how do adaptive computation mechanisms behave at scale?

- Importance
  - Parameter efficiency is increasingly critical: large dense models are expensive to deploy (Section 1). Efficiently allocating compute per input (“easy vs hard”) yields better latency/cost profiles.
  - The work reframes “scaling” to include iterative latent computation (“depth of computation” decoupled from parameter count), potentially enabling strong small models.

- Prior approaches and their limits
  - CoT expends tokens to think, inflating context and latency [6].
  - Parameter sharing existed (e.g., ALBERT [20]) but was not widely used for reasoning in modern LLMs (Section 2).
  - Latent pondering variants exist [15, 16, 24, 25], but were tested at modest scales and often biased gates to exit early (geometric priors), under‑exploring deeper computation (Section 3.3; Appendix A and Figure 10).

- Position relative to existing work
  - Builds on the Universal Transformer idea [7]—tying weights and iterating block(s)—but adds:
    - An entropy‑regularized expected loss objective with a uniform prior over exit steps (Section 3.3; Eq. (3), Appendix A/Figure 10).
    - A dedicated second training stage to directly train exit gates based on observed loss improvements (Section 3.4; Eq. (4)–(5)).
    - Evidence at scale: 7.7T tokens, with models at 1.4B and 2.6B matching 4B and 8B baselines (Figures 1–2; Tables 7–9).
  - Provides mechanistic evidence that recurrence improves knowledge manipulation, not storage capacity (Section 6; Figure 6; Figure 7), and offers a theoretical view of efficient latent reasoning (Section B.5; Theorem 2).

## 3. Technical Approach

Reasoning behind the approach: The design must allow a fixed‑size model to “think longer” internally and decide when to stop, while keeping training stable at scale. The approach ties layer weights, computes iteratively in latent space, and learns an early‑exit policy that optimizes the compute‑accuracy trade‑off.

- Core architecture and objective
  - `LoopLM` defines `F^(t)` as repeatedly applying the same hidden stack `HL` `t` times, then decoding via `lmhead` (Section 3.1; Eq. (1)):
    - `F^(t)(·) = lmhead ◦ HL ◦ HL ◦ ... ◦ HL ◦ emb(·)` (t times).
  - Standard next‑token loss at each step `t` (Section 3.1; Eq. (2)):
    - `L^(t) = E_x [∑_{i=1}^{M-1} -log Pr(F^(t)(x_{1:i}) = x_{i+1})]`.
    - This ensures each recurrent step is a competent predictor; deeper steps should improve loss.

- Adaptive early‑exit gating (`Q`‑exit)
  - At each recurrent step `t`, the model produces an early‑exit probability `λ_t(x) = σ(Gate(F^(t)(x)))` (Section 3.2).
  - These form a distribution over exit steps via survival probabilities (Algorithm 1; Section 3.2):
    - `q_ϕ(t|x) = λ_t(x) ∏_{j=1}^{t-1} (1 - λ_j(x))`.
    - The cumulative distribution `CDF(t|x) = ∑_{i=1}^t q_ϕ(i|x)` is compared to a threshold `q ∈ [0,1]`.
    - Deterministic `Q`‑exit: `t_exit(x) = min{t : CDF(t|x) ≥ q}`. Lower `q` exits earlier (less compute), higher `q` allows more loops (more compute).

- Stage I: entropy‑regularized pretraining with uniform prior
  - Training objective combines expected step‑loss with an entropy term (Section 3.3; Eq. (3)):
    - `L = ∑_{t=1}^{Tmax} q_ϕ(t|x) · L^(t) - β·H(q_ϕ(·|x))`, where `H` is entropy.
  - Variational interpretation: an ELBO with uniform prior on exit step (Section 3.3):
    - Using `KL(q_ϕ || π)`, uniform `π` simplifies to maximizing entropy; unlike geometric priors, uniform avoids bias toward shallow exits and encourages exploration of all depths (Appendix A; Figure 10).

- Stage II: specialized adaptive gate training
  - Train the gates to stop when further loops don’t improve the loss (Section 3.4).
  - Measure per‑token improvement from step `t-1` to `t` (Eq. (4)):
    - `I^(t)_i = max(0, L^(t-1)_{i,stop} - L^(t)_{i,stop})`.
  - Compute ideal continuation probability `w^(t)_i = σ(k·(I^(t)_i - τ))` with slope `k=50` and threshold `τ=0.005`.
  - Train gate via weighted cross‑entropy (Eq. (5)):
    - `L^(t)_adaptive = - (1/M) ∑_i [ w^(t)_i·log(1-λ^(t)_i) + (1-w^(t)_i)·log(λ^(t)_i) ]`.
    - Averaged over steps: `L_adaptive = (1/Tmax) ∑_{t=2}^{Tmax} L^(t)_adaptive`.
  - This penalizes both underthinking (stopping too soon) and overthinking (looping past diminishing returns) (Section 3.4).

- Model architecture and training pipeline
  - Transformer blocks: decoder‑only, `MHA` with `RoPE`, `SwiGLU` FFN, “sandwich” `RMSNorm` for stability (Section 4.1; Table 1).
  - Two model sizes:
    - `Ouro-1.4B`: 24 layers, `d_model=2048`, 49,152 vocab (Table 1).
    - `Ouro-2.6B`: 48 layers, `d_model=2048`, same vocab (Table 1).
  - 7.7T token pipeline (Figure 4; Section 4.3–4.4):
    - Warmup + Stable Training (3T).
    - Split streams: keep 1.4B vs upcycle to 2.6B (duplicating layers—parameter sharing eases upcycling; Section 4.3.2).
    - Stable Training (3T), CT Annealing (1.4T, higher quality math/code; Table 4), LongCT (20B tokens, 64K contexts), Mid‑Training (300B tokens with high‑quality SFT‑style mixtures).
  - Key stability choices (Section 4.3.1; Table 5):
    - Reduced recurrent steps from 8 to 4 after instability (loss spikes/grad oscillations).
    - Batch size increased 4M→8M tokens for more stable gradients.
    - Entropy regularization coefficient `β` reduced from 0.1→0.05 to lower conflict with task loss and reduce bias against deeper steps.
    - Sequence lengths progressed: 4K → 16K → 64K → 32K.

- Data composition and SFT
  - Stage 1: massive web (Nemotron‑CC, MAP‑CC), some Chinese initially; later removed due to tokenizer inefficiency (Section 4.1–4.2; Tables 2–3).
  - Stage 2: high‑quality math/code (Nemotron‑CC‑Math‑v1, MegaMath HQ, OpenCoder; Table 4).
  - Stage 3: long contexts (ProLong‑64K; 20B tokens).
  - Stage 4: mid‑training with diverse SFT‑style QA/CoT mixtures, decontamination, ChatML formatting (300B tokens effective; Section 4.2.1).
  - Final reasoner SFT: ~8.3M examples focusing on math/code/science/chat (Table 6); 2 epochs, 32K max length (Section 4.4).

- Inference efficiency trick: KV cache sharing
  - Prefill requires separate KV caches per step; reusing during prefill degrades >10 points on GSM8K (Section 5.4.2).
  - Decoding can reuse only last‑step KV or averaged KV across steps, reducing memory by 4× with minimal performance loss (Table 14):
    - GSM8K: 78.92 (full cache) vs 78.85 (last‑step) vs 78.73 (averaged).
    - MATH‑500: 82.40 (full) vs 80.40 (last‑step) vs 78.52 (averaged).

- Early exit strategy comparisons
  - `Q`‑exit gate (untrained vs adaptively trained), static depth baseline, and heuristic based on hidden‑state differences were compared (Figure 5; Section 5.4.1).
  - The adaptively trained gate dominated the accuracy/compute trade‑off; hidden‑state diff was competitive but slightly behind; static had diminishing returns (Figure 5).

- Scaling law analysis for LoopLM
  - Empirical scaling laws for total loss and step‑wise loss fit well (R²≈0.96 for total loss across size, data, max depth; Section D.2; Figure 15).
  - Generalizability demonstrated across unseen sizes/data/max‑depth (Appendix E).
  - Observed phenomenon: with small models at larger data, shallow step losses can increase as the gate reallocates mass toward deeper steps to reduce expected loss (Section D.3; Figures 17–18; explained by the expected loss structure and entropy regularization).

- Mechanistic insight and theory
  - Knowledge capacity is unchanged by recurrence (~2 bits per parameter across looped and non‑looped models; Figure 6 left; Section 6.1).
  - Manipulation capability improves substantially:
    - Modular arithmetic (`Mano`): looped models outperform iso‑parameter baselines (Figure 6 right; Section 6.2).
    - Natural language 3‑hop QA: looped models learn with fewer samples and faster (Figure 7; B.3.1).
  - Theoretical latent reasoning efficiency: graph reachability with O(log D) recurrence (Section B.5; Theorem 2), illustrating why repeated, parallelizable latent passes can be more efficient than token‑level rationalization.

## 4. Key Insights and Innovations

Reasoning behind the selection: The most distinctive elements are those that directly enable parameter efficiency and faithful latent reasoning at scale: the adaptive gate training, the uniform‑prior entropy objective, the mechanistic results, and practical inference affordances.

- Iterative latent reasoning built into pre‑training (fundamental)
  - What’s new: Tie transformer blocks and reuse them for `t` recurrent steps (`F^(t)`; Eq. (1)). Every step is supervised (Eq. (2)), and a gate decides when to stop (Section 3.1–3.2).
  - Why it matters: Adds a third scaling axis (“compute depth per input”) without growing parameters; avoids CoT’s longer outputs and context bloat (Section 1; Figure 3). Performance peaks at trained depth and improves safety with more steps (Sections 5.3–7.1).

- Entropy‑regularized early exit with a uniform prior (innovative training objective)
  - What’s new: Expected step loss plus entropy (Eq. (3)) encourages unbiased exploration of all depths; uniform prior avoids shallow bias observed with geometric priors (Appendix A; Figure 10).
  - Why it matters: The gate learns input‑difficulty‑dependent termination, not merely minimizing average cost; depth utilization is better, and compute/accuracy Pareto frontier shifts up (Appendix A; Section 5.4.1).

- Stage II adaptive gate training using observed loss improvements (innovative supervision)
  - What’s new: Gate trained to continue only when `I^(t)` (Eq. (4)) warrants it, via a token‑wise cross‑entropy aligning gate outputs to “ideal” continuation probabilities (Eq. (5)).
  - Why it matters: Improves compute‑efficiency vs static depth and entropy‑only gates by 2–3% accuracy at the same average rounds (Figure 5).

- Mechanistic evidence: recurrence boosts knowledge manipulation, not storage (fundamental)
  - What’s new: Controlled synthetic tasks separate storage and manipulation (Section 6). Capacity is ~2 bits/parameter regardless of looping (Figure 6 left), but manipulation tasks (modular trees; multi‑hop QA) benefit strongly (Figure 6 right; Figure 7).
  - Why it matters: Explains why small LoopLMs match larger dense models on reasoning‑heavy benchmarks while not necessarily beating them on knowledge‑heavy ones (Appendix B.4; Table 15).

- Faithfulness and safety properties improve with recurrent depth (important capability)
  - Faithfulness: Intermediate predictions change across steps; linear probes show decisions are made within a step but revised by subsequent recurrence (Figure 9 left/right; Section 7.2). In contrast, some CoT models’ final answers are essentially determined before the “thinking” text.
  - Safety: Harmfulness score and harmful rates decrease as recurrent steps increase, even in extrapolation (Figure 8a). PCA shows better separation of benign/harmful prompts at greater depth (Figure 8b), suggesting recurrent refinement helps safety judgments (Section 7.1).

- Practical inference affordances (incremental but impactful)
  - KV cache reuse during decoding yields 4× memory reduction with negligible loss (Table 14).
  - Built‑in draft–verify: intermediate heads at step `s` can propose tokens, final head at step `T` verifies, enabling speculative decoding without training a separate draft model (Section 7.3).
  - Anytime generation: deeper loops refine predictions monotonically in expectation, enabling compute‑budget‑aware generation with `Q`‑exit (Section 7.3).

## 5. Experimental Analysis

Reasoning behind evaluation: To test whether the architecture and training actually yield parameter‑efficient reasoning, the work compares 1.4B/2.6B LoopLMs to larger dense baselines across many benchmarks, analyzes performance vs recurrent depth, and validates the gate and KV cache strategies.

- Evaluation methodology
  - Base models: Compared against Qwen2.5/Qwen3, Gemma3, Llama3(.1/.2) using standardized harnesses (`lm-eval-harness`, `evalplus`; Section 5.1; Appendix C.1).
  - Reasoning models: AIME 2024/2025, OlympiadBench, GPQA, SuperGPQA, BeyondAIME, HLE; single in‑house harness; LLM‑as‑judge with fixed rubric and settings (Section 5.2; Table 9).
  - Depth and extrapolation: Performance measured across `t` steps; training used Tmax=4 but extrapolated to 5–8 during inference (Section 5.3; Tables 10–13).

- Main quantitative results
  - Parameter efficiency at scale (Figures 1–2; Tables 7–9):
    - `Ouro-1.4B` (R4) ≈ 4B dense models on many tasks:
      - `BBH`: 71.02 vs Qwen3‑4B at 70.95 (Table 7).
      - `GSM8K`: 78.92 vs Qwen3‑4B at 72.86 (Table 7).
      - `MATH500`: 82.40 vs Gemma3‑4B at 68.60 and Qwen3‑4B at 59.60 (Table 7).
    - `Ouro-2.6B` (R4) ≈ or > 8B dense:
      - `MMLU-Pro`: 55.73 vs Qwen3‑8B at 53.72 (Table 8).
      - `BBH`: 80.46 vs Qwen3‑8B at 77.65 (Table 8).
      - `MATH500`: 90.85 vs Qwen3‑8B at 62.30; Gemma3‑12B at 83.20 (Table 8).
      - `ARC-C`: 66.40 vs Qwen3‑8B at 66.10 (Table 8).
  - Advanced reasoning benchmarks (Figure 2; Table 9):
    - `Ouro-1.4B-Thinking R4`: OlympiadBench 71.55, BeyondAIME 34.0; competitive with Qwen3‑4B (73.18 and 31.0).
    - `Ouro-2.6B-Thinking R4`: OlympiadBench 76.44 (exceeds Qwen3‑8B at 75.25), BeyondAIME 39.0 (exceeds Qwen3‑8B at 38.0); AIME pass@10 also strong (AIME24 90.0; AIME25 76.7).
  - Recurrent depth behavior (Section 5.3; Tables 10–13):
    - Base models’ performance improves from T=1→4 and degrades when extrapolating to 5–8 (Tables 10–11).
    - Reasoning SFT models peak around T=3–5 depending on task; performance declines beyond trained depth (Tables 12–13).
  - Early exit and efficiency (Section 5.4.1; Figure 5):
    - At same average rounds, the adaptively trained gate yields ~2–3% higher accuracy vs entropy‑only gate.
    - Hidden‑state difference heuristic is competitive but slightly behind trained gate; static exit shows diminishing returns.
  - KV cache reuse (Section 5.4.2; Table 14):
    - 4× memory reduction at decode with minimal loss using last‑step or averaged KV caches.
  - Safety and faithfulness (Section 7):
    - Safety: harmfulness score and harmful rates decrease as recurrent steps increase (Figure 8a); Thinking models reach harmful rate 0.009 (1.4B) and 0.003 (2.6B) at 4 steps—comparable to Qwen3‑4B-Thinking (Section 7.1).
    - Faithfulness: ROC AUC of probes shows answers form within a step but change across steps (Figure 9 left); step agreement matrix shows deliberate revisions across steps (A[2,3]=551; A[3,4]=361 out of 1000; Figure 9 right).
  - Scaling law fit (Section D; Figures 15–20; Appendix E):
    - Total loss fits across size/data/depth with R²≈0.96; generalizes to unseen sizes and depths.
    - Step‑wise loss scaling also fits (R²≈0.80–0.89), supporting predictable training dynamics for looped architectures.

- Are claims convincingly supported?
  - The parameter‑efficiency claims are supported by broad, multi‑benchmark comparisons against strong baselines using common harnesses (Tables 7–9; Figures 1–2).
  - Mechanistic claim (manipulation > storage) is supported by controlled synthetic tasks, capacity plots (~2 bits/parameter), and reasoning‑heavy category gains in MMLU (Table 15; Section 6.1–6.2; Appendix B.4).
  - Safety and faithfulness claims are backed by quantitative measures (HEx‑PHI trends across steps; PCA separation; Quora probes and step agreement; Figures 8–9).
  - Adaptive gate superiority is shown via accuracy/compute curves (Figure 5).
  - Practicality is bolstered by the KV cache sharing results (Table 14).

- Ablations and robustness checks
  - Prior choice for the gate: uniform vs geometric priors tested; uniform yielded better convergence and depth utilization (Appendix A; Figure 10).
  - Extrapolation tests beyond trained depth show safety improves while task scores can degrade (Tables 10–13; Figure 8a).
  - Depth‑wise probes and disagreement offer internal consistency checks (Figure 9).
  - RL alignment attempts reported but did not improve over SFT (Section 4.5), suggesting infrastructure gaps and saturation.

- Mixed/conditional results and trade‑offs
  - In small‑scale experiments, standard transformers sometimes outperform LoopLM at the same recurrent count, especially as loop count rises (Section D.1; Figures 13–14; Table 18), though the gap narrows with model size.
  - Performance generally peaks at the trained depth; extrapolation to deeper steps may reduce scores while improving safety (Section 5.3; Figure 8a).
  - Tokenizer limits (Latin‑centric 49,152 vocab) constrained multilingual/math symbol coverage; Chinese data was removed after Stage 1 due to inefficiency (Section 4.1–4.2).

## 6. Limitations and Trade-offs

Reasoning behind limitations: The reported successes depend on careful choices (uniform prior, gate training, stability hyperparameters) and still present trade‑offs (depth extrapolation, RL infrastructure, tokenizer, safety vs accuracy).

- Assumptions and dependencies
  - `Uniform prior` for exit steps is key to exploring deeper recurrence; geometric priors push shallow exits and under‑explore depth (Appendix A; Figure 10).
  - Training stability relies on reducing loops (8→4), increasing batch size, and reducing `β` (Section 4.3.1); deeper recurrence remains challenging.
  - Gate training uses a greedy loss‑improvement heuristic (Eq. (4)–(5)), which may not capture global optimal stopping.

- Scenarios not addressed or partially addressed
  - Extrapolation beyond trained depth: task performance declines (Tables 10–13), though safety improves (Figure 8a); robust training at ≥8 loops remains open.
  - RL alignment with dynamic early exit was not effective due to infrastructure limitations (Section 4.5), leaving open whether RL could further improve reasoning.

- Computational/data constraints
  - Pre‑training used 7.7T tokens—substantial compute; success may depend on this scale (Figure 4; Table 5).
  - Vocab of 49,152 from SmolM2 is Latin‑optimized; Chinese and specialized math tokens suffered (Chinese removed after Stage 1; Section 4.1–4.2).

- Model and evaluation caveats
  - Small‑scale analyses (Section D.1) show a performance gap favoring standard transformers at higher loop counts, although the gap narrows with size.
  - LLM‑as‑judge protocols introduce judgment variance; while rubrics are fixed, any automatic judge has limitations (Section 5.2).
  - Decontamination is claimed but in open ecosystems complete elimination is difficult to guarantee (Section 4.2.1).

- Theoretical idealization
  - The reachability proof (Section B.5; Theorem 2) uses simplified single‑head transformers with thresholded normalization and specific matrix encodings; bridging to mainstream multi‑head, floating‑point implementations is not proven.

- Trade‑offs observed
  - `Safety vs accuracy`: increasing recurrence improves safety (Figure 8a) but can reduce benchmark scores when extrapolating beyond trained depth (Tables 10–13).
  - `Compute vs latency`: adaptive early exit saves compute but requires gate inference; static depth is simpler but less efficient (Figure 5).
  - `Cache management`: prefill must keep separate caches, increasing memory; decoding can reuse last‑step or averaged KV (Section 5.4.2; Table 14).

## 7. Implications and Future Directions

Reasoning behind implications: The work argues for a “third axis” of scaling—recurrent latent depth—with demonstrated parameter efficiency and safety/faithfulness benefits. Next steps are clear: deeper stable training, finer‑grained adaptive routing, better infrastructure, and broader data.

- How this changes the landscape
  - Establishes recurrent latent depth as a practical scaling axis alongside parameters and data (Conclusion).
  - Demonstrates that small models can match much larger ones on reasoning by “thinking more” internally instead of generating long CoT text (Figures 1–2; Tables 7–9).
  - Points to internal latent trajectories as faithful reasoning substrates, closing the gap between articulated justifications and final answers (Section 7.2; Figure 9).

- Follow‑up research enabled or suggested
  - Stability and training at deeper loops (≥8), possibly with improved normalization, optimizer schedules, or layer variants (Section 4.3.1).
  - Token‑level or mixture‑of‑recursions routing for per‑token adaptive depth [19], potentially combining with MoE experts for fine‑grained compute allocation.
  - Better priors or adaptive regularization schedules; investigate when uniform vs learned priors perform best.
  - Infrastructure for RL alignment with dynamic compute (Section 4.5), including rollout systems that can track variable depth and early exits.
  - Richer vocabularies (math symbols; multilingual scripts) and math‑optimized tokenization to expand capability.

- Practical applications and downstream use cases
  - Edge deployment of reasoning‑capable small models (1.4B/2.6B) with adaptive compute for latency and cost control (Figures 1–2; KV cache reuse Table 14).
  - Built‑in speculative decoding via step‑s proposals and step‑T verification; no separate draft model needed (Section 7.3).
  - Joint acceleration and pre‑emptive safety: screen draft tokens at intermediate steps before streaming (Section 7.3).
  - Anytime generation: set `Q`‑exit thresholds to match latency budgets; stream from intermediate steps while deeper loops verify/refine (Section 7.3).

In sum, this work shows how looping weight‑tied transformer blocks and learning adaptive halting during pre‑training can yield parameter‑efficient, safer, and more faithful reasoning. The technical mechanisms—entropy‑regularized expected loss with a uniform prior, specialized gate training, and KV cache reuse—translate into practical efficiency, while synthetic and theoretical analyses clarify why recurrence improves manipulation rather than storage. The main challenges ahead are scaling stable recurrence beyond four steps, strengthening infrastructure for dynamic‑compute RL, and broadening data/tokenization for multilingual and symbolic domains.