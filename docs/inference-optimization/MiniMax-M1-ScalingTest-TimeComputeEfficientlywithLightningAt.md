# MiniMax-M1: Scaling Test-Time Compute Efficiently with Lightning Attention

**ArXiv:** [2506.13585](https://arxiv.org/abs/2506.13585)

## 🎯 Pitch

MiniMax-M1 is the first open-weight, large-scale reasoning model to efficiently scale test-time compute by combining a hybrid Mixture-of-Experts architecture with the Lightning Attention mechanism, enabling unprecedented 1M-token context windows and up to 80K-token generations at a fraction of standard compute cost. It also introduces CISPO, a novel reinforcement learning algorithm that preserves critical, rare reasoning steps during training, allowing for rapid and stable scaling of long-form reasoning skills—making MiniMax-M1 especially powerful for software engineering, tool use, and long-context tasks that previously would have been prohibitively expensive or inaccessible.

---

## 1. Executive Summary (2-3 sentences)
MiniMax-M1 introduces an open‑weight large reasoning model that scales “test‑time compute” (the computation spent while generating long chains of thought) far more efficiently by combining a hybrid attention architecture with Lightning Attention and a Mixture‑of‑Experts (MoE). It also proposes CISPO, a new reinforcement learning (RL) objective that preserves gradient contributions from rare but crucial reasoning tokens, enabling fast and stable RL to very long outputs (up to 80K tokens) and 1M‑token inputs.

## 2. Context and Motivation
- Problem addressed
  - Long, high‑quality reasoning typically improves as models spend more FLOPs during generation, but standard softmax attention has quadratic cost in sequence length, making very long thoughts and contexts prohibitively expensive. Section 1 frames this as a “test‑time compute” scaling problem.
  - Existing large reasoning models (LRMs) like o1 and DeepSeek‑R1 use RL to elicit longer reasoning but still rely largely on softmax attention; linear/sparse alternatives have shown promise but were rarely validated at scale for state‑of‑the‑art LRMs (Section 1; related work list).

- Why it matters
  - Real workloads—software engineering, tool‑use, and long‑document understanding—require both extended thinking and very long contexts. M1 targets up to 1M input tokens and up to 80K generated tokens, making such tasks tractable with lower compute (Table 1, Section 1).

- Shortcomings of prior approaches
  - Quadratic attention limits long inputs and long generations.
  - Prior linear/SSM/RNN variants (e.g., Performer, RetNet, Mamba) either have limited validation at scale or are not open (Section 1).
  - Common RL algorithms (PPO/GRPO/DAPO) clip large token updates, which inadvertently suppresses rare “fork” tokens (e.g., “However,” “Recheck”) that drive deep reasoning (Section 3.1, “Issues of Token Clipping”).

- Positioning
  - M1 is an open‑weight, large MoE model that interleaves Lightning Attention (a linear‑time variant implemented in an I/O‑aware way) with occasional softmax layers (“hybrid attention”), enabling near‑linear scaling for long sequences during both inference and RL (Section 1; Figure 1 Right).
  - RL is scaled with a new algorithm, CISPO, and a suite of engineering recipes so training completes in 3 weeks on 512 H800s (≈$0.53M rental) while achieving competitive performance with strong open‑weight models and favorable long‑context/tool‑use results against leading closed models (Abstract; Sections 1 and 3; Table 2).

## 3. Technical Approach
This section walks through M1’s architecture, training pipeline, RL algorithm, and long‑length scaling strategy.

- Model architecture: hybrid attention + MoE
  - Size: 456B total parameters with 45.9B active per token; 32 experts (Section 1).
  - Hybrid attention pattern: one transformer block with standard softmax attention follows every seven “transnormer” blocks that use Lightning Attention (Section 1).
    - Lightning Attention (LA): a linear‑attention variant (from Qin et al. 2022a; 2024b,c) implemented to be I/O‑efficient; cost grows roughly linearly with sequence length rather than quadratically.
    - Design intuition: mostly linear attention for scalability, but periodic softmax blocks to preserve global expressivity and calibration.
  - Native context: up to 1M tokens (Table 1). Output (thinking) length: up to 80K tokens for the released M1‑80k; 40K for M1‑40k (Table 1, Section 1).

- Why this design is efficient
  - Figure 1 (Right) shows theoretical inference FLOPs vs generation length: M1 uses <50% the FLOPs of DeepSeek‑R1 at 64K tokens and about 25% at 100K. This directly targets the test‑time compute bottleneck for long reasoning (Section 1).

- Pretraining and SFT (Section 2)
  - Continual pretraining
    - 7.5T additional tokens with higher proportions of STEM/code/reasoning (70%) and curated QA; refined parsing, cleaning, and semantic deduplication (Section 2.1, “Training Data”).
    - Recipe: constant LR 8e‑5 for 2.5T tokens, then decay to 8e‑6 over 5T; MoE aux‑loss coefficient reduced; larger micro‑batch to soften aux‑loss impact (Section 2.1, “Training Recipe”).
    - Long‑context extension in four stages (32K→...→1M). A smooth schedule prevents gradient explosions linked to differing decay rates across early vs. late LA layers (“earlier layers focus more on local information,” Section 2.1, “Long Context Extension”).
  - Supervised fine‑tuning (SFT)
    - Injects reflection‑style chain‑of‑thought (CoT) patterns across math, coding, STEM, writing, QA, and multi‑turn chat; ≈60% math+coding to seed later RL for long reasoning (Section 2.2).

- RL algorithm: CISPO (Section 3.1)
  - Background and problem
    - Standard PPO/GRPO/DAPO clip token‑level updates when the importance ratio `r_i,t` is large, which often happens for low‑probability “fork” tokens that initiate deeper reasoning. Once clipped early, those tokens stop contributing to later off‑policy updates, impeding the emergence of long CoT (Section 3.1 “Issues of Token Clipping”).
  - CISPO’s idea
    - Preserve all token gradients but stabilize learning by clipping only the importance sampling (IS) weights, not the token updates themselves.
    - Start from the REINFORCE objective with IS correction (Eq. 3), then replace `r_i,t` with a clipped version `ŝr_i,t` within a range `[1 - ε_IS_low, 1 + ε_IS_high]` (Eq. 5), and optimize the token‑level group‑relative advantage objective (Eq. 4, building on GRPO’s advantage in Eq. 2).
      - `Â_i,t` is a group‑normalized advantage; no KL penalty is used (Section 3.1).
      - A general masked form (Eq. 6–7) shows PPO‑style clipping as a special case, unifying strategies.
    - Practical choice: only tune the upper clip `ε_IS_high`; set the lower side very large (effectively unbounded below), keeping gradients from all tokens (Section 3.1).
  - Outcome
    - CISPO reduces variance, maintains exploration entropy via rare tokens, and empirically reaches DAPO‑level performance with half the steps on AIME 2024 using Qwen2.5‑32B (Figure 2).

- RL with the hybrid architecture: stability recipes (Section 3.2)
  - Precision mismatch fix
    - During RL, probabilities computed in training vs. inference diverged due to precision differences, particularly from large activations in the LM head. Making the LM output head FP32 realigned them: correlation improved from ≈0.987 to ≈0.997 and stayed stable (Figure 3 and text under “Computational Precision Mismatch…”).
  - Optimizer settings
    - Gradients span 1e‑18 to 1e‑5 and are weakly correlated across steps; AdamW with `β1=0.9, β2=0.95, eps=1e‑15` avoided non‑convergence seen with common settings like (0.9, 0.999, 1e‑8) (same subsection).
  - Early truncation via repetition detection
    - If 3,000 consecutive tokens each exceed probability 0.99, generation is cut to prevent pathological loops and stabilize gradients (same subsection).

- RL environments and rewards (Section 4)
  - Rule‑verifiable tasks (Section 4.1)
    - Mathematical reasoning: high‑quality, deduplicated competition problems; pass@10 filtering to keep moderate difficulty; ≈50K samples (details on cleaning, overlap removal, and reformatting in Section 4.1).
    - Logical reasoning: 41 tasks generated with the SynLogic framework; difficulty bounded by model solvability; ≈53K instances (Section 4.1).
    - Competitive programming: public problems; test suites generated where needed; filtered by pass rates; ≈30K (Section 4.1).
    - Software engineering: execution‑based sandbox derived from SWE‑bench—run tests for rewards; includes bug localization, repair, and test synthesis; several thousand samples (Section 4.1).
  - General‑domain tasks via reward models (Section 4.2)
    - With ground truth but hard to rule‑check: use a Generative Reward Model (GenRM) trained and validated on human‑annotated comparisons; graded rewards and Best‑of‑N selection checks (Section 4.2.1).
    - Without ground truth (instruction following, creative writing): pairwise preference scoring against vetted reference answers; additional rule‑based checks for constraint satisfaction; bias minimization via multiple‑blind and position‑switched judgments; “Swiss Round” scoring to choose references (Section 4.2.1).
    - Length‑bias mitigation: continuous online monitoring for reward hacking toward verbosity; if detected, recalibrate GenRM and apply reward shaping/normalization (Section 4.2.2).

- Curriculum for mixing tasks (Section 4.3)
  - Start RL with rule‑verified reasoning tasks, then gradually blend in general‑domain tasks, balancing verifiable skills with broader assistant abilities.

- Extending the thinking budget to 80K (Section 5)
  - Data curation: filter out easy items using the 40K model; emphasize harder math/coding; downsample synthetic reasoning that caused repetitive, destabilizing patterns (Section 5).
  - Staged length expansion: 40K → 48K → 56K → 64K → 72K → 80K, advancing when perplexity stabilizes and the 99th percentile length nears the current cap (Section 5).
  - Preventing late‑sequence collapse: early stopping for repetition; combine sample‑level loss with token‑level normalization; reduce gradient clip threshold and `ε_IS_high` (Section 5).

- Training budget and availability
  - Full RL completes in 3 weeks on 512 H800s (~$534,700 rental) (Abstract; Section 3). Models are released with vLLM and Transformers support (end of Section 1).

## 4. Key Insights and Innovations
- Hybrid attention that actually scales long reasoning in a frontier‑scale LRM
  - Distinctive aspect: seven Lightning Attention blocks followed by one softmax block repeat; native 1M context and efficient long generations (Section 1).
  - Why it matters: Figure 1 (Right) shows near‑linear compute scaling; at 100K tokens, M1 uses about one‑quarter the FLOPs of DeepSeek‑R1. This is a fundamental efficiency advancement for long CoT.

- CISPO: clip IS weights, not token updates (Section 3.1)
  - Difference from PPO/GRPO/DAPO: preserves gradients from rare, high‑leverage reasoning tokens by avoiding token‑level clipping, while keeping updates stable via IS weight clipping (Eq. 4–5).
  - Impact: On AIME 2024 with Qwen2.5‑32B, CISPO matches DAPO performance with 50% of training steps and outperforms GRPO at equal steps (Figure 2). This is a methodological innovation with clear training‑efficiency gains.

- Engineering fixes enabling RL at scale with the hybrid architecture (Section 3.2)
  - FP32 LM head to eliminate train‑vs‑infer probability drift (Figure 3), tuned AdamW for tiny gradients, and a probability‑based early truncation rule to avoid degenerate loops. These are practical but essential to make RL stable at ultra‑long lengths.

- Realistic, verifiable SE sandbox and length‑bias‑aware reward modeling (Sections 4.1 and 4.2)
  - Execution‑based rewards for real repos align training with practical software engineering; continuous monitoring and recalibration reduce GenRM length bias to prevent reward hacking in long CoT.

- Efficient long‑length RL schedule (Section 5)
  - Staged length expansion with quality monitors and adjusted losses/clip thresholds is a robust recipe for moving from 40K to 80K thinking budgets.

Together, these are primarily fundamental innovations in efficiency (hybrid attention) and RL optimization (CISPO), complemented by impactful engineering and data contributions.

## 5. Experimental Analysis
- Evaluation setup (Section 6)
  - Decoding: temperature 1.0, top‑p 0.95 for all tasks.
  - Benchmarks and metrics:
    - Math: AIME 2024/2025 (average pass rate over 32 samples) and MATH‑500 (Section 6.1).
    - Coding: LiveCodeBench (contamination‑controlled; report average pass rate over 16 samples) and FullStackBench (Section 6.1).
    - Reasoning & knowledge: GPQA‑Diamond (pass@32), HLE without tools, ZebraLogic, MMLU‑Pro (Section 6.1).
    - Software engineering: SWE‑bench Verified using an Agentless‑style pipeline with two‑stage localization (Section 6.1).
    - Long context: OpenAI‑MRCR at 128K and 1M, and LongBench‑v2 (Section 6.1).
    - Agentic tool use: TAU‑bench airline and retail scenarios (max 40 steps; generic system prompt; GPT‑4.1 as the user model) (Section 6.1).
    - Factuality: SimpleQA (short‑form factuality) (Section 6.1).
    - General assistant: MultiChallenge (GPT‑4o judged) (Section 6.1).

- Capabilities and headline numbers (Table 2, Figure 1, Table 1)
  - Context and generation limits:
    - Quote: “Max Input 1M; Max Output 80K” for `MiniMax‑M1‑80k` versus 128K/64K for `DeepSeek‑R1` and 200K/32K for `Claude 4 Opus` (Table 1).
  - Long‑context efficiency:
    - Quote: “M1 consumes <50% of FLOPs at 64K tokens and ≈25% at 100K vs DeepSeek‑R1” (Figure 1 Right, Section 1).

- Math and coding (Table 2)
  - AIME 2024: `M1‑80k` 86.0%; behind `DeepSeek‑R1‑0528` 91.4% but ahead of `Qwen3‑235B‑A22B` 85.7% and most open‑weight baselines.
  - AIME 2025: `M1‑80k` 76.9%; behind `R1‑0528` 87.5%.
  - MATH‑500: `M1‑80k` 96.8%, competitive but not state‑leading.
  - LiveCodeBench: `M1‑80k` 65.0%, on par with `Qwen3‑235B‑A22B` 65.9%; `o3`/`Gemini‑2.5` ≈76–77%.
  - FullStackBench: `M1‑80k` 68.3% > `Qwen3‑235B‑A22B` 62.9% and close to top closed models near 69–70%.

- Reasoning & knowledge (Table 2)
  - GPQA‑Diamond: `M1‑80k` 70.0%, trailing `R1‑0528` 81.0% and closed models (o3 83.3, Gemini‑2.5 86.4).
  - HLE (no tools, text‑only subset): `M1‑80k` 8.4%—lower absolute scores across open models without tools.
  - ZebraLogic: `M1‑80k` 86.8%, above `Qwen3‑235B‑A22B` 80.3 but below closed and `R1` (≈95%).
  - MMLU‑Pro: `M1‑80k` 81.1%, slightly below top open models (84–85% ranges for others in Table 2).

- Software engineering (Table 2)
  - SWE‑bench Verified: `M1‑80k` 56.0% and `M1‑40k` 55.6%, close to `R1‑0528` 57.6% and far above other open‑weights (e.g., `Qwen3‑235B‑A22B` 34.4). This aligns with their execution‑based RL on real repos (Section 4.1).

- Long‑context (Table 2)
  - OpenAI‑MRCR (128K): `M1‑40k` 76.1% and `M1‑80k` 73.4%, beating `o3` 56.5 and `Claude 4` 48.9, nearing `Gemini‑2.5` 76.8.
  - OpenAI‑MRCR (1M): `M1‑40k` 58.6% and `M1‑80k` 56.2% vs `Gemini‑2.5` 58.8; other models not reported at 1M.
  - LongBench‑v2: `M1‑80k` 61.5% > `DeepSeek‑R1‑0528` 52.1 and `Qwen3‑235B‑A22B` 50.1.

- Agentic tool use (Table 2)
  - TAU‑bench (airline): `M1‑80k` 62.0%, best among open‑weights and above `Gemini‑2.5` 50.0; close to `Claude 4 Opus` 59.6.
  - TAU‑bench (retail): `M1‑40k` 67.8% > `M1‑80k` 63.5; top closed model `Claude 4 Opus` 81.4.

- Factuality and assistant ability (Table 2)
  - SimpleQA: `M1‑80k` 18.5% outperforms most open‑weights except `DeepSeek‑R1` 27.8; behind `o3` 49.4 and `Gemini‑2.5` 54.0.
  - MultiChallenge: both `M1` variants 44.7, roughly comparable to `R1‑0528` 45.0 and `Claude 4 Opus` 45.8; below `o3` 56.5 and `Gemini‑2.5` 51.8.

- Do longer thoughts help? (Section 6.2; Figure 4)
  - Quote: “Average response lengths on AIME and LiveCodeBench exceed 20,000 tokens,” with AIME 2024 accuracy rising from ~68% to ~80% as training proceeds. Curves show a strong correlation between longer outputs and higher accuracy.

- Algorithmic ablation (Figure 2; Section 3.1)
  - On AIME 2024 with Qwen2.5‑32B, CISPO reaches DAPO performance with 50% steps and beats GRPO across steps (“2× speedup” annotation in Figure 2).

- Robustness/stability diagnostics (Figure 3; Section 3.2)
  - Probability alignment between train and infer modes improved to ≈0.997 correlation after the FP32 head fix and remained stable during training.

- Overall assessment
  - The evidence convincingly supports:
    - Efficiency: FLOPs scaling advantage (Figure 1 Right), 1M/80K limits (Table 1), and successful long‑length RL (Figure 4).
    - Competitiveness: Near‑SOTA among open‑weights overall, with pronounced strengths in long‑context tasks and realistic tool/SE scenarios (Table 2).
  - Results are mixed on math/coding versus the very latest `DeepSeek‑R1‑0528` and on factuality versus top closed models; this nuance is transparent in Table 2.

## 6. Limitations and Trade-offs
- Performance trade‑offs
  - Math and coding competitions: `M1‑80k` trails the latest `DeepSeek‑R1‑0528` on AIME and GPQA (Table 2).
  - Short‑form factuality remains behind top closed models (SimpleQA; Table 2).
- Reward model risks
  - Even with online monitoring and recalibration (Section 4.2.2), GenRMs can encode biases (e.g., toward length). The paper’s mitigations reduce but do not eliminate this risk; reliance on a learned judge remains a potential failure point for open‑ended tasks.
- Training complexity and reproducibility
  - Stability depends on engineering details: FP32 LM head, AdamW hyperparameters tuned for very small gradients, and repetition‑based early truncation (Section 3.2). Replicating results requires these kernels/recipes.
- Compute requirements
  - Although efficient relative to alternatives, full RL still needs 512 H800s for 3 weeks (Section 3), which is significant for many groups.
- Architecture caveats
  - Hybrid attention mixes linear and softmax layers; the exact ratio (7:1) is a design choice. The paper does not present a systematic study of ratios or where softmax is most beneficial, leaving optimality open.
- Long‑length pathologies
  - Section 5 details late‑sequence collapse and repetition during length scaling; mitigations work in practice, but this reveals fragility at extreme lengths and dependence on careful scheduling and loss normalization.

## 7. Implications and Future Directions
- How it changes the landscape
  - Demonstrates that an open‑weight, frontier‑scale LRM can combine linear‑time attention with occasional softmax to support 1M‑token contexts and very long CoT at substantially lower FLOPs (Figure 1 Right; Table 1), without sacrificing competitive performance across many domains (Table 2).
  - Introduces CISPO—a general alternative to PPO/GRPO/DAPO—that is simple, stable, and empirically faster for reasoning RL (Figure 2; Eqs. 3–6).

- Follow‑up research enabled
  - Systematic studies on hybrid attention ratios, placement, and interaction with MoE.
  - Generalization of CISPO across tasks, model sizes, and off‑policy update depths; theoretical analysis of bias introduced by IS clipping.
  - Better reward models for reasoning: length‑invariant scoring, decomposition by correctness vs. verbosity, and tool‑assisted verification signals.
  - Methods for detecting and preventing late‑sequence collapse automatically at even larger thinking budgets; curriculum schedulers that adaptively set length caps.
  - Kernel and memory optimizations for 1M‑token contexts in open inference stacks (vLLM/Transformers support is already provided).

- Practical applications
  - Long‑form analysis: legal/finance/regulatory documents and scientific literature where 1M‑token context is transformative.
  - Software engineering agents: repository‑scale reasoning with execution‑verified rewards (SWE‑bench style), patch synthesis, and test generation.
  - Multi‑step tool use: TAU‑bench results indicate viability for agentic workflows with policy constraints.
  - Enterprise agents requiring extended, auditable chains of thought at controlled FLOPs budgets.

> In sum, MiniMax‑M1 provides a concrete, open blueprint for scaling test‑time compute efficiently—architecturally via hybrid Lightning Attention and algorithmically via CISPO—backed by strong long‑context and agentic performance, while leaving clear avenues for improving peak reasoning and factuality against the very latest closed and open competitors.
