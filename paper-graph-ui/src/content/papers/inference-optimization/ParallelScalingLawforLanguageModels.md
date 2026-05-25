# Parallel Scaling Law for Language Models

**ArXiv:** [2505.10475](https://arxiv.org/abs/2505.10475)

## 🎯 Pitch

This paper introduces PARSCALE, a novel paradigm for scaling language model capacity by running multiple parallel forward passes with learnable input transformations and dynamically aggregating the outputs—boosting capability without increasing parameter count. Unlike traditional scaling which balloons memory or slows inference, PARSCALE achieves similar gains to parameter scaling (O(log P) with P parallel streams) while dramatically cutting both memory and latency overhead, as validated by theory and large-scale experiments. This breakthrough enables deployment of more powerful LLMs in memory- and compute-constrained environments, and provides new insight into how computation and parameters contribute to model ability.

---

## 1. Executive Summary
This paper introduces PARSCALE: a way to scale a language model’s capability by running P parallel forward passes with small, learnable perturbations of the input and then dynamically aggregating the P outputs, without increasing the model’s core parameters. It derives and validates a “parallel scaling law” showing that adding parallel streams behaves like increasing parameters by roughly O(log P), and demonstrates better inference efficiency (memory and latency) than conventional parameter scaling (Figures 1 and 4; Section 3).

## 2. Context and Motivation
- Problem the paper addresses
  - Mainstream scaling strategies either increase parameters (space-heavy) or increase generated reasoning tokens at inference (time-heavy). The paper asks whether there is a universal and inference-efficient way to raise capability without large memory or time penalties (Introduction, Section 1).
- Why it matters
  - Practical: Large models (e.g., hundreds of billions of parameters) are hard to deploy on edge devices due to memory. Inference-time scaling (more reasoning tokens) can be slow and scenario-limited. The paper targets a method that is memory-efficient and broadly applicable, enabling stronger models in low-resource settings (Section 1; Table 1).
  - Theoretical: It offers a quantitative lens on the contributions of compute vs. parameters to model capability, extending the family of LLM scaling laws (Sections 3.1–3.2).
- Prior approaches and their limits
  - Parameter scaling (dense or MoE): boosts quality but raises inference memory a lot (Table 1).
  - Inference-time scaling (e.g., chain-of-thought, repeated sampling): improves reasoning but often requires specialized data or verifiers and substantially increases inference time (Section 5 “Related Work”).
  - Classifier-Free Guidance (CFG) in diffusion: uses multiple passes and a contrastive aggregation rule, but lacks theory and relies on hand-designed transformations (Section 2).
- Positioning
  - PARSCALE generalizes the idea behind CFG from “two passes with a fixed rule” to “P passes with learnable input transformations and learnable aggregation,” then formalizes a scaling law tying parallel compute to effective parameter increases (Sections 2 and 3).

## 3. Technical Approach
Step-by-step mechanism (Sections 2, 3; Figure 1):
1. Core idea: parallel streams
   - A standard Transformer LM `fθ` is kept fixed structurally. For each input x, the method creates P transformed versions x₁…x_P, runs P parallel forward passes, and aggregates the P next-token distributions into one output (Equation (2)).
   - Definitions:
     - `parallel stream`: one forward pass on a transformed version of the input.
     - `prefix tuning`: a parameter-efficient method where small trainable “prefix” embeddings are injected into each attention layer’s Key/Value pathways, effectively steering each stream differently.
     - `KV-cache`: cached Keys/Values from self-attention during decoding; here, different KV-caches implement different stream prefixes.
2. Input transformation: prefix tuning
   - Each stream uses distinct learned prefixes across layers; this can be seen as using different KV-caches per stream (Appendix A, “Input Transformation”).
   - Random initialization suffices to induce diversity across streams; alternatives (LoRA, BitFit) were tested but yielded similar performance, so prefix tuning is chosen for minimal invasiveness (Appendix A; Table 6).
3. Output aggregation: dynamic, learnable mixture
   - Concatenate the P output logits and feed them to an MLP that outputs P aggregation weights via softmax (Equation (6)). A small label-smoothing term keeps weights from collapsing to one stream, similar to load-balancing in MoE (Appendix A; Equation (7)).
   - Added parameters are tiny: ≈0.2% per stream (Section 2).
4. Training and inference
   - The same P-way compute and aggregation are used both during training and inference. This is crucial: unlike inference-only tricks (beam search, self-consistency), the model learns to exploit parallel computation (Section 2; Appendix H).
5. Theoretical framework relating compute and parameters (Section 3.1)
   - Baseline assumption: Each stream’s converged loss follows a Chinchilla-type law (Lemma 3.1, Equation (3)): Lᵢ = (A/N)^α + E, where N is parameter count, E is data entropy baseline, and A, α are constants.
   - Proposition 1 (Equation (4)): Aggregating P streams reduces loss as
     L = ( A / [ N · P^(1/α) · DIVERSITY ] )^α + E,
     where DIVERSITY = [(P − 1)ρ + 1]^(-1/α), and ρ is the correlation of residual errors across streams.
   - Intuition: If different streams err independently (ρ≈0), averaging reduces error like an ensemble. High correlation (ρ≈1) yields little gain; negative correlation could yield even more reduction.
6. Practical, fitted scaling law (Section 3.2)
   - Empirically, across many runs, gains grow roughly logarithmically with P. The paper therefore fits
     L = ( A / [ N · (k·log P + 1) ] )^α + E  (Equation (5)),
     with excellent fit (R² up to 0.998; Figure 2; Tables 8–11 in Appendix E).
   - Interpretation: Increasing P behaves like increasing effective parameters by a factor proportional to log P.
7. Implementation and experimental design
   - Models: Qwen-2.5-like dense architecture; 36 layers fixed, varying width to get 0.5B–4.4B non-embedding parameters; P ∈ {1,2,4,8} (Appendix C, Table 7).
   - Pre-training for scaling-law fits: 42B tokens, batch size 1024, sequence length 2048, standard Adam, bfloat16; two datasets: Pile (general) and Stack-V2 Python (code/reasoning) (Section 3.2; Appendix C).
8. Efficiency analysis (Section 3.3; Figure 4)
   - Memory: PARSCALE adds minimal parameters; it increases KV-cache size but reuses backbone weights, so memory grows slowly vs. parameter scaling (Figures 4a–4d).
   - Latency: At small batch sizes, decoding is memory-bound; shifting to compute-bound via parallel streams can lower latency growth compared to bigger models (Figures 4e–4h). Quantified advantage at batch size 1: “22× less memory increase and 6× less latency increase” for a 1.6B model at P=8 vs parameter scaling to equal performance (Section 3.3; Figure 1(3) and Figure 4).
9. Two-stage training for production budgets (Section 4.1; Figure 5; Table 4)
   - Stage 1: train normally on 1T tokens.
   - Stage 2: enable PARSCALE and train only 20B tokens more. The new prefix/aggregation parameters adapt quickly (after ≈0.0002T tokens; Figure 5), delivering most of the gains at tiny additional cost.
10. Applying to an off-the-shelf model (Section 4.2; Figure 6)
    - Continual pre-training of Qwen2.5-3B on Pile/Stack-V2: lower training loss with higher P (Figures 6a–6b).
    - Parameter-efficient fine-tuning (freeze backbone, train only new PARSCALE parameters): code Pass@1 improves from 47.4% to 53.0% and Pass@10 from 73.1% to 78.2% when increasing P from 1 to 8 (Figure 6c).

## 4. Key Insights and Innovations
- A general parallel scaling law that links compute and parameters (Sections 3.1–3.2)
  - Novelty: Formalizes how P parallel streams change the loss, first via Proposition 1 (Equation (4)) and then via an empirical law with log P (Equation (5)).
  - Significance: Converts “more parallel compute” during train/inference into an equivalent “effective parameter increase” ≈ O(log P). This reframes capacity as a function of both parameters and parallel compute, not parameters alone (Figures 2–3).
- Learnable, end-to-end parallelization mechanism (Section 2; Appendix A)
  - Different from prior inference-only tricks (CFG, beam search) or specialized verifiers: both the input perturbations (prefixes) and aggregation weights are learned during training, making the model intrinsically capable of using parallel compute.
  - Practicality: Adds ≈0.2% parameters per stream and requires no architectural change to the backbone.
- Inference efficiency gains at small batch sizes (Section 3.3; Figure 4)
  - Fundamental distinction vs parameter scaling: PARSCALE reuses parameters and expands compute; at typical edge settings (batch size 1–2), it achieves the same capability with much smaller memory and lower latency growth.
- Compute vs. memorization vs. reasoning (Figure 2; Tables 2–3)
  - Observation: Gains from parallel compute are larger on code/math (reasoning-heavy) than on general knowledge (memorization-heavy). Fitted k is higher on Stack-V2 Python (k=0.39) than on Pile (k=0.33), suggesting parallel compute contributes more to reasoning ability (Section 3.2).

## 5. Experimental Analysis
- Evaluation setup (Sections 3–4; Appendix G)
  - Scaling-law fitting: Train 24 runs per dataset (P ∈ {1,2,4,8} × N across 6 sizes) on 42B tokens from Pile and Stack-V2 Python; report final training loss.
  - Downstream pretraining checkpoints: code (HumanEval/+, MBPP/+) and general tasks (WinoGrande, HellaSwag, OpenBookQA, PiQA, ARC, SciQ), using lm-eval-harness and EvalPlus (Appendix G).
  - Inference efficiency: measure GPU memory and end-to-end latency across batch sizes {1,2,4,8} and prompt+output lengths {64,128,256,512} using llm-analysis (Section 3.3).
  - Large-scale two-stage run: 1.8B model to 1T tokens (normal) + 20B tokens (PARSCALE) with a data mix emphasizing general, math, and code (Section 4.1; Table 4).
  - Off-the-shelf experiments: Qwen2.5-3B continual pretraining and PEFT (Section 4.2; Figure 6).
- Main quantitative results
  - Scaling-law fit
    - Excellent fits for Equation (5): R²=0.9978 on Stack-V2 Python and R²=0.9987 on Pile (Figure 2; Tables 8–11). The loss reductions align closely across P increments (1→2→4→8), consistent with a log P trend.
  - Downstream after 42B-token pretraining (Tables 2–3)
    - Code average (HumanEval(+), MBPP(+)): at 1.6B parameters, increasing P from 1 to 8 raises average accuracy from 33.9% to 39.1% (+5.2). At 4.4B, P=1→8 raises 39.2% to 45.4% (+6.2) (Table 2).
    - General tasks: improvements are smaller but consistent; e.g., at 2.8B, P=1→8 improves from 55.2% to 58.1% (+2.9) (Table 3).
  - Inference efficiency (Figure 4; Section 3.3)
    - Quote: “When batch size is 1, for a 1.6B model and scaling to P=8 … 22× less memory increase and 6× less latency increase compared to parameter scaling that achieves the same performance” (Section 3.3; Figure 1(3), Figure 4e).
    - Trend: Memory growth under PARSCALE is modest even as batch sizes rise (Figures 4a–4d). Latency advantage diminishes as batches get larger (4e–4h) but remains favorable up to batch=8 for the tested sizes.
  - Two-stage 1T-token training (Table 4; Figure 5)
    - General average: 56.0 (P=1) → 58.6 (P=8), +2.6 points.
    - GSM8K (4-shot): 28.7 → 38.4 (+9.7 absolute; ~34% relative).
    - HumanEval Pass@1/10 (averaged with MBPP): code averages also increase across P (Table 4 bottom).
    - Convergence: New PARSCALE parameters stabilize quickly in Stage 2 (≈0.0002T tokens; Figure 5).
  - Instruction tuning (Table 5)
    - On IFEval (0-shot), MMLU (5-shot), GSM8K (4-shot), increasing P improves all three: IFEval 54.1 → 59.5, MMLU 34.2 → 41.7, GSM8K 50.3 → 56.1 when P=1→8.
  - Off-the-shelf and PEFT (Figure 6)
    - Continual pretraining: training loss decreases faster with higher P on both Stack-V2 Python and Pile (Figures 6a–6b).
    - PEFT (freeze backbone): average code Pass@1 47.4 → 53.0 and Pass@10 73.1 → 78.2 as P increases to 8 (Figure 6c), demonstrating “dynamic parallel scaling”: swap P at deployment while keeping the same backbone.
  - Ablations and pivot experiments (Appendix A; Table 6)
    - Output aggregation matters: dynamic weighted sum with label smoothing (ε=0.1) performs best.
    - Input transformation choice matters little relative to P: prefix tuning vs LoRA/BitFit differ by ~0.1% loss; the dominant factor is the number of parallel computations.
  - Beam search vs. PARSCALE (Appendix H; Table 30)
    - On math benchmarks, beam search only helps slightly at 2 beams, then degrades as beams increase. PARSCALE improves consistently as P grows, underscoring the need to learn to use parallel compute during training.
  - Data repetition robustness (Appendix D; Figure 7)
    - With repeated epochs on OpenWebText, PARSCALE resists overfitting better than parameter scaling at the point where validation loss spikes, likely due to fewer new parameters.
- Do the experiments support the claims?
  - The scaling-law fits are strong across two distinct datasets and six model sizes (Figure 2), and contour plots visualize the interplay of parameters vs P (Figure 3).
  - Efficiency measurements directly target practical metrics (memory, latency) and show advantages in the small-batch regime relevant to edge devices (Figure 4).
  - Generality is evidenced by pretraining from scratch, two-stage training to production scale, and adaptation to an existing model with PEFT (Sections 4.1–4.2).
  - Caveats: the empirical law is fit up to P=8 and N≤4.4B (non-embedding) for the fitting runs; generalization to much larger P or N is an extrapolation (Section 6 below).

## 6. Limitations and Trade-offs
- Theoretical assumptions and gaps (Section 6; Section 3.1)
  - Proposition 1 hinges on the correlation ρ between stream residuals. ρ is difficult to model a priori and is treated as constant in a secondary fit that underperforms the log P model (Appendix E; Tables 12–15). Why diversity scales logarithmically with P remains an open question.
- Empirical scope
  - Parallel streams were evaluated up to P=8 and model sizes up to 4.4B non-embedding for the scaling-law fits (Figure 2). Behavior for P≫8 or very large N is not established.
  - The primary scaling-law dataset size is fixed at 42B tokens (Section 3.2). The law’s dependence on data scale and quality (beyond the two-stage 1T demonstration) is left for future work (Section 4; Discussion).
- Compute during training
  - Training cost scales roughly linearly with P (more floating-point operations), even though inference is efficient. The two-stage strategy cuts additional token budget to ~2% (20B/1T), but large-P training is still compute-heavy (Section 4.1).
- Efficiency trade-offs with batch size
  - Latency gains are most pronounced at small batch sizes; as batch increases, decoding becomes compute-bound and PARSCALE’s latency advantage declines, though it remains favorable in tests up to batch=8 (Figure 4).
- Task dependence
  - Gains are larger on reasoning-oriented tasks (Stack-V2 Python, GSM8K) than on general knowledge tasks (Pile, MMLU). If a workload is dominated by memorization, parameter scaling may be more cost-effective (Section 3.2; Tables 2–3, 4).
- Engineering considerations
  - Requires integrating prefix-tuned streams and a gating MLP into inference stacks; modest but non-zero complexity. Label smoothing in the aggregator is needed to avoid load imbalance early in training, similar to MoE stability concerns (Appendix A).

## 7. Implications and Future Directions
- Reframing scaling: compute as first-class capacity
  - By quantifying that P parallel streams ≈ O(log P) parameter increase (Equation (5); Figure 3), the work encourages “inference-optimal” designs that balance parameters and parallel compute under memory and latency budgets (Discussion, “Training Inference-Optimal Language Models”).
- Practical deployment pathway
  - For edge devices and small-batch settings, PARSCALE offers a route to higher capability without the memory footprint of larger models (Figures 4a, 4e). The ability to freeze the backbone and vary P at deployment (“dynamic parallel scaling”) enables adaptive quality/latency trade-offs (Section 4.2; Figure 6c).
- Synergy with other techniques
  - The method complements inference-time reasoning (e.g., chain-of-thought). In the 1T two-stage experiments, CoT plus PARSCALE yields additional GSM8K gains (Table 4 and Section 4.1).
  - Potentially complementary to sparse MoE: MoE is parameter-heavy but latency-friendly; PARSCALE is compute-heavy but memory-friendly. Combining them could target both constraints (Discussion).
- Research directions
  - Theory: Model and measure ρ and DIVERSITY more precisely; determine when diversity can grow faster than log P; study asymptotics for P≫8 (Discussion).
  - Data scaling: Fit parallel scaling laws jointly over parameters, compute, and data quantity/quality; analyze behavior under repeated data and domain shifts (Appendix D).
  - Adaptive and selective parallelism: Learn when and where in a sequence to invoke higher P; allocate streams dynamically per token or per task.
  - Broader domains: Apply PARSCALE to other architectures (vision, speech) and to non-generation tasks to explore general parallel scaling laws (Conclusion).

> Core takeaway: Section 3 shows that “scaling P times of parallel computation is similar to scaling parameters by O(log P),” validated by high-R² fits (Figure 2), while Section 3.3 and Figure 4 show that, at small batch sizes, the same capability can be reached with “up to 22× less memory increase and 6× less latency increase” than parameter scaling. This provides both a conceptual and practical path to more capable models under tight inference budgets.
