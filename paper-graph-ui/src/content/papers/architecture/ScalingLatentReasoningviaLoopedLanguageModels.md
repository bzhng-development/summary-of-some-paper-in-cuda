# Scaling Latent Reasoning via Looped Language Models

**ArXiv:** [2510.25741](https://arxiv.org/abs/2510.25741)

## 🎯 Pitch

Ouro introduces Looped Language Models (LoopLMs), a breakthrough architecture that weaves iterative, non-textual reasoning directly into pre-training by repeatedly applying the same stack of Transformer layers. This enables models with just 1.4B–2.6B parameters to match or outperform much larger (4B–8B+) LLMs on challenging reasoning and math tasks, drastically improving parameter efficiency while enabling adaptive compute at inference. By providing deep, faithful internal reasoning with lower resource demands and built-in safety gains, LoopLMs mark a new direction for scaling large language models beyond sheer parameter counts—unlocking powerful AI under practical constraints.

---

## 1. Executive Summary (2-3 sentences)
This paper introduces Ouro, a family of “Looped Language Models” (LoopLMs) that reuse the same stack of Transformer layers multiple times per token to perform iterative, latent (non-textual) reasoning during pre‑training. Trained on 7.7 trillion tokens, Ouro’s 1.4B and 2.6B models with four recurrent loops match or surpass much larger standard Transformers (4B–8B) on difficult reasoning tasks while using fewer parameters, and they provide an adaptive early‑exit mechanism that trades compute for accuracy on the fly.

## 2. Context and Motivation
- Problem addressed
  - Modern LLMs usually scale capability by adding parameters and/or by generating long chain‑of‑thought (CoT) text at inference. Both incur high cost and latency. The paper targets parameter efficiency and built‑in reasoning without relying on lengthy outputs.
- Why this matters
  - Smaller, faster models lower deployment cost and latency, widen access, and reduce infrastructure demands. Architectures that compute “deeper” internally—without growing parameter count—could become a third scaling axis alongside model size and data (Section 1).
- Limitations of prior approaches
  - Parameter sharing is known (e.g., ALBERT), but most shared‑weight or recurrent LLMs have only been explored at modest scale; it is unclear if benefits persist in trillion‑token regimes.
  - Inference‑time CoT spends compute by generating many tokens, grows context length, and can produce post‑hoc rationalizations not causally tied to the answer (Section 7.2; Figure 9).
  - Adaptive halting methods (e.g., PonderNet) often bias toward shallow computation via geometric priors, which can under‑explore deeper reasoning (Section 3.3; Appendix A, Figure 10).
- Positioning
  - Ouro scales the “looped depth” idea to 7.7T tokens and adds two key ingredients: (i) an entropy‑regularized training objective with a uniform prior that encourages exploration of all depths; (ii) a dedicated Stage‑II gate‑training loss that explicitly ties halting to measured loss improvements (Sections 3.3–3.4). It then shows strong parameter‑efficiency and new properties in safety and faithfulness.

## 3. Technical Approach
The core idea is to “reapply the same stack of Transformer layers multiple times” per token, refining hidden states in latent space rather than generating additional textual reasoning.

- Architecture and notation (Section 3.1; Eq. 1–2; Figure 3)
  - Let `HL(·)` be a stack of `L` Transformer layers (e.g., 24 or 48 layers). Instead of a single pass, LoopLM composes `HL` with itself `t` times: `F^(t) = lmhead ∘ HL ∘ HL ∘ … ∘ HL ∘ emb(·)`.
  - Each “reapplication” is a recurrent step (also called `loop step` or `recurrent depth`), indexed by `t = 1..Tmax`. All steps share parameters.
  - Training uses the usual next‑token objective at each loop depth (Eq. 2): `L^(t)` is the cross‑entropy when decoding from the state after step `t`. Intuition: deeper loops should produce better predictions on harder tokens.

- Adaptive early exit: how the model decides when to stop looping (Section 3.2; Algorithm 1)
  - The model computes a halting gate `λ_t(x) = σ(Gate(F^(t)(x)))` at each step and forms a stopping distribution over steps `q_φ(t | x)` using a survival‑probability formulation.
  - Inference uses a deterministic “Q‑exit” rule (Algorithm 1): accumulate the CDF over steps and stop at the first step where CDF ≥ threshold `q ∈ [0,1]`. This single knob trades compute for accuracy at deployment time.

- Training the exit behavior in two stages
  - Stage I: entropy‑regularized exploration (Section 3.3; Eq. 3; Appendix A)
    - Objective: expected task loss across steps minus β times the entropy of `q_φ(·|x)` (Eq. 3).
    - Viewed as an ELBO with a uniform prior over steps; unlike geometric priors that favor shallow halts, the uniform prior avoids biasing the model to stop early and forces it to explore all depths (Section 3.3 “Alternative perspective”; Appendix A Figure 10 shows lower loss and better stability with the uniform prior).
  - Stage II: focused gate training (Section 3.4; Eq. 4–5)
    - Freeze the language model; train only the gates.
    - Define per‑token improvement `I_i^(t) = max(0, L_{i,stop}^{(t−1)} − L_{i,stop}^{(t)})` (Eq. 4); map it to an “ideal continue” probability via a sharp sigmoid around a threshold τ (continue if the latest step materially improved loss).
    - Optimize a cross‑entropy between this “ideal continue/stop” target and the predicted gate `λ` (Eq. 5). This penalizes both “underthinking” (stopping too soon) and “overthinking” (looping without benefit).

- Implementation and training pipeline (Section 4; Figure 4; Table 5)
  - Base architecture: decoder‑only Transformer with MHA + RoPE + SwiGLU, “sandwich” RMSNorm before attention and FFN blocks.
  - Two model sizes with the same 49,152‑token vocabulary: Ouro‑1.4B (24 layers, 2048 hidden), Ouro‑2.6B (48 layers, 2048 hidden) (Section 4.1).
  - Training recipe totals 7.7T tokens across four stages (Section 4.2–4.3; Figure 4; Tables 2, 5):
    - Stage 1a/1b: 6T tokens (web + code + math); initial 8 loops caused instability, reduced to 4 loops for stability; “upcycling” creates the 2.6B variant by stacking layers (Section 4.3.2).
    - Stage 2 (CT Annealing): 1.4T tokens of higher‑quality math/code/general data; sequence length extended to 16K (Table 4).
    - Stage 3 (LongCT): 20B tokens at 64K context for long‑context ability.
    - Stage 4 (Mid‑training): 300B tokens of diverse SFT‑style QA/CoT mixtures in ChatML format.
  - SFT for reasoning (“Ouro‑Thinking”): 8.3M examples emphasizing math and code (Section 4.4; Table 6). RL attempts (Section 4.5) did not beat SFT due to infrastructure limits with dynamic depth.

- Inference efficiency: KV‑cache sharing across loops (Section 5.4.2; Table 14)
  - During decoding (not prefilling), reusing only the final loop’s KV cache (“last‑step reuse”) matches the full 4× cache baseline within 0.2–2 points on GSM8K/MATH500 while cutting memory 4× (Table 14).

## 4. Key Insights and Innovations
1. Iterative latent reasoning as a third scaling axis (Figures 1–2; Tables 7–9)
   - Reusing depth via loops lets small models match bigger dense Transformers on hard reasoning tasks after the same pre‑training. This is distinct from inference‑time CoT: compute is spent on hidden‑state refinement rather than long outputs.

2. Entropy‑regularized halting with a uniform prior (Section 3.3; Appendix A, Figure 10)
   - The uniform prior over exit steps avoids the shallow‑computation bias of geometric/Poisson‑lognormal priors. Empirically it yields lower training loss and stabler convergence in a 776M LoopLM (Appendix A, left panel), indicating better exploration of deep steps.

3. Greedy, loss‑improvement‑driven gate training (Section 3.4; Eq. 4–5; Figure 5)
   - A dedicated Stage‑II objective aligns halting with measured gains in token‑level cross‑entropy. On MMLU, this training makes adaptive exit uniformly better than static exits and naive heuristics; at the same average loop count, it gives ~2–3% absolute accuracy over an untrained gate (Figure 5).

4. Evidence that looping improves knowledge manipulation, not storage (Section 6; Figure 6–7)
   - On synthetic “Capo” biographies, both looped and non‑looped models store ~2 bits/parameter (Figure 6 left), so loops do not increase capacity.
   - On “Mano” modular arithmetic and multi‑hop QA, looped models generalize with fewer samples and higher accuracy than iso‑parameter and often iso‑FLOP non‑looped baselines (Figure 6 right; Figure 7).

5. Safer and more faithful reasoning with depth (Section 7; Figures 8–9)
   - Safety (HEx‑PHI) improves as loops increase—even beyond training depth (Figure 8a). PCA shows harmful vs. benign prompts become more separable at deeper steps (Figure 8b).
   - Faithfulness: step‑to‑step predictions actually change and converge, indicating causal latent computation rather than post‑hoc rationalization. Linear probes cannot predict the final decision from earlier‑step states (Figure 9 left), and adjacent steps disagree substantially (Figure 9 right) up to the trained depth, which is what a revising reasoning process should look like.

## 5. Experimental Analysis
- Evaluation setup (Sections 5.1–5.3, 5.4)
  - Base models: compared to Qwen2.5/Qwen3/Gemma3/Llama3.* across MMLU, MMLU‑Pro, BBH, ARC‑C/E, HellaSwag, Winogrande, GSM8K, MATH500, HumanEval/+, MBPP/+ (Tables 7–8). Same harness for fairness (Section 5.1).
  - Reasoning models (“Ouro‑Thinking”): AIME 2024/2025, OlympiadBench, GPQA, SuperGPQA, BeyondAIME, HLE with an in‑house judge protocol (Section 5.2; Table 9).
  - Recurrent‑depth sweeps and extrapolation: evaluate T=1..8 though training used T=4 (Tables 10–13).
  - Early‑exit strategies: static exit vs. hidden‑state delta threshold vs. learned gates with/without Stage‑II training (Figure 5).
  - Efficiency: decode‑time KV‑cache sharing variants (Table 14).

- Main quantitative results
  - Parameter efficiency at scale
    - Ouro‑1.4B (R4) vs 4B baselines (Table 7):
      - Reasoning heavy: BBH 71.02 (Ouro‑1.4B) vs 70.95 (Qwen3‑4B); GSM8K 78.92 vs 72.86; MATH500 82.40 vs 59.60.
      - General knowledge: MMLU 67.35 vs 73.19 (trails), but MMLU‑Pro 48.62 vs 51.40 (close).
    - Ouro‑2.6B (R4) vs 8–12B baselines (Table 8):
      - MMLU 74.60 (Ouro‑2.6B) near Qwen3‑8B 76.63; MMLU‑Pro 55.73 vs 53.72; BBH 80.46 vs 77.65; MATH500 90.85 vs 62.30; MBPP 80.40 vs 79.00.
      - On coding, HumanEval 78.7 trails Qwen3‑8B 84.8; HumanEval+ 70.7 trails 75.3.
  - Advanced reasoning suites (Table 9; Figure 2)
    - AIME24 pass@1: Ouro‑1.4B‑Thinking 65.0, Ouro‑2.6B‑Thinking 64.7 (competitive with Qwen3‑4B 61.3; below Qwen3‑8B 73.0).
    - AIME25 pass@1: Ouro‑2.6B‑Thinking 50.3 vs Qwen3‑4B 51.3 (close); Qwen3‑8B 66.7 is higher.
    - OlympiadBench: 76.44 (Ouro‑2.6B‑Thinking) vs 75.25 (Qwen3‑8B).
    - BeyondAIME: 39.0 (Ouro‑2.6B‑Thinking) vs 38.0 (Qwen3‑8B).
  - Depth behavior (Tables 10–13)
    - Performance rises steeply from T=1 to T=4 (trained depth), then degrades mildly at T>4. For the 1.4B base model: MMLU peaks at 67.45 at T=4, drops to 64.49 at T=8 (Table 10).
    - Reasoning SFT peaks around T=3–5 depending on benchmark (Tables 12–13).
  - Early exit (Figure 5)
    - For the same average exit round, the Stage‑II trained gate yields the best MMLU accuracy; hidden‑state‑difference is a strong heuristic but consistently below the trained gate; untrained gate lags by ~2–3%.
  - KV‑cache sharing (Table 14)
    - “Last‑step only” achieves GSM8K 78.85 (vs 78.92 full) and MATH500 80.40 (vs 82.40 full) with 4× lower cache.
  - Safety and faithfulness (Figures 8–9)
    - Harmfulness scores and rates drop as recurrent steps increase (Figure 8a), including in extrapolated T>4.
    - PCA of top‑layer states shows clearer harmful/benign separation at larger T (Figure 8b).
    - On Quora Question Pairs, linear probes show within‑step predictability but cross‑step revisions (Figure 9 left); step‑to‑step label agreement is far below 100% up to T=4 (Figure 9 right), indicating real latent updates.

- Do the experiments support the claims?
  - Parameter efficiency for reasoning is strongly supported (Tables 7–8; Figures 1–2).
  - “Loops improve manipulation not capacity” is supported by:
    - Capacity: near‑identical bits/parameter on Capo (≈2 bits/param; Figure 6 left).
    - Manipulation: superior accuracy/sample‑efficiency on Mano and multi‑hop QA (Figure 6 right; Figure 7).
  - Adaptive exit’s practical value is demonstrated by clear Pareto improvements in accuracy vs. average loops (Figure 5).
  - Safety/faithfulness signals are suggestive and consistent (Figures 8–9), though they rely on proxy measures (HEx‑PHI judged by GPT‑4o; linear probes/agreements).

- Notable ablations/robustness
  - Prior choice for halting: uniform prior yields lower loss and better stability than geometric priors (Appendix A, Figure 10).
  - Recurrent depth extrapolation: base models degrade modestly beyond T=4; SFT peaks vary (Tables 10–13).
  - KV‑cache reuse alternatives: first‑step cache is catastrophic; average or last‑step caches work (Table 14).
  - Category analysis on MMLU (Appendix B.4; Table 15) shows largest gains in reasoning‑heavy categories (e.g., Elementary Math +156%) and minimal gains in fact‑recall categories (e.g., Global Facts +8%).

## 6. Limitations and Trade-offs
- Assumptions and training choices
  - Trained with at most four loops; performance degrades beyond the trained depth (Tables 10–13). This limits “free” extrapolation to deeper thinking.
  - Exit‑gate training needs two stages and careful stability settings (Section 4.3.1), including reduced β and batch scaling; initial 8‑loop training was unstable (Section 4.3.2).
- Scope and evaluation constraints
  - Vocabulary lacks Chinese tokens; Chinese data was dropped after Stage 1 due to poor tokenization efficiency (Section 4.1–4.2), reducing multilingual generality.
  - Some reasoning benchmarks use LLM‑as‑judge (Section 5.2), which introduces grader bias; safety uses GPT‑4o as judge (Section 7.1).
- Compute and data
  - 7.7T tokens and long contexts imply substantial compute/time cost. Gains may depend on the large, curated data mixture (Tables 2–4).
- Mixed results by domain
  - On general knowledge (MMLU), the 1.4B model trails the best 4B baselines (Table 7); coding scores for 2.6B lag the 8B best (Table 8).
- Open questions
  - Why RL generalizes to using fewer rounds at inference when trained at fixed depth (Section 4.5) is not yet understood.
  - Formal guarantees on faithfulness are not provided; evidence is observational (probes/agreements).

## 7. Implications and Future Directions
- Field‑level impact
  - Establishes recurrent depth as a practical, scalable axis of LLM capability. For reasoning‑centric tasks, looping can substitute for parameters—e.g., a 2.6B LoopLM competes with 8B dense models (Table 8)—and enables adaptive computation via early exit.
- Practical applications
  - Latency/compute‑aware deployment via Q‑exit thresholding (Algorithm 1), anytime generation, and built‑in draft‑and‑verify speculative decoding using intermediate heads (Section 7.3).
  - Memory‑efficient decoding with last‑step cache reuse enables deployment on constrained hardware (Section 5.4.2; Table 14).
  - Safer outputs through deeper latent refinement, even at the same parameter count (Figure 8).
- Research directions
  - Train at deeper loop counts and study methods to improve extrapolation beyond the trained depth (Section 8).
  - Better theoretical underpinnings of latent reasoning with tied weights; the paper provides an `O(log D)` construction for graph reachability with loops (Appendix B.5), suggesting efficiency advantages versus discrete/continuous CoT.
  - Improved RL infrastructure for dynamic‑depth models and principled safety/faithfulness evaluation that goes beyond proxy graders.
  - Multilingual tokenization and specialized vocabularies (e.g., math/code symbols) to lift limits acknowledged in Section 4.1.

> Primary takeaway: Looping the same layers turns depth from a static architectural choice into an input‑adaptive computation budget, yielding strong parameter efficiency and emergent benefits in safety and faithfulness when trained at scale with a uniform‑prior, entropy‑regularized halting objective and a loss‑improvement‑aligned gate.
