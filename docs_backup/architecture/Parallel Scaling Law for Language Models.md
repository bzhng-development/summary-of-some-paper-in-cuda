# Parallel Scaling Law for Language Models

**ArXiv:** [2505.10475](https://arxiv.org/abs/2505.10475)
**Authors:** Mouxiang Chen, Binyuan Hui, Zeyu Cui, Jiaxi Yang, Dayiheng Liu, Jianling Sun, Junyang Lin, Zhongxin Liu
**Institutions:** 

## 🎯 Pitch

PARSCALE revolutionizes language model scaling by leveraging parallel computation streams with learnable diversity, offering a new path to augment model capacity without increasing parameters or token generation. This innovation drastically reduces memory and latency overheads of traditional scaling methods, providing a viable solution for deploying large models efficiently on memory-constrained devices, ultimately enhancing reasoning capabilities in applications like coding assistants and math solvers.

---

## 1. Executive Summary
This paper introduces PARSCALE, a way to scale language models by running P parallel forward passes of the same backbone with learnable input transformations and then dynamically aggregating the P outputs. The key finding is a new “parallel scaling law”: for a fixed-parameter model, increasing parallel streams to P behaves like increasing parameters by about O(log P) while being far more inference‑efficient in memory and latency (Figures 1 and 4, Equation (5)).

## 2. Context and Motivation
- Problem the paper targets
  - Standard ways to make language models better either increase parameters (“parameter scaling”) or increase the number of generated reasoning tokens at inference (“inference‑time scaling”). Both are expensive: parameters cost memory, and long chains of thoughts cost time (Section 1; Figure 1(1)).
  - The paper asks whether we can increase “compute” without increasing parameters or generated tokens, and whether that compute translates into capacity in a predictable way.

- Why this matters
  - Edge deployment is memory‑limited. Very large models (e.g., 672B parameters) are impractical to run on devices (Section 1).
  - Reasoning models that rely on very long thoughts can be slow and sometimes “overthink” (Section 1).
  - Understanding how “computation” (not just parameters) contributes to capacity is a fundamental question (end of Section 1).

- Prior approaches and gaps
  - Parameter scaling: effective but memory‑heavy.
  - Inference‑time scaling: improves reasoning but adds latency and often needs special data or verifiers (Table 1, Section 5).
  - Classifier‑Free Guidance (CFG) in diffusion models uses extra forward passes at inference to improve quality, but lacks a unifying theory and is not trained end‑to‑end for parallel compute (Background, Equation (1)).

- Positioning
  - PARSCALE scales parallel computation during both training and inference while keeping the backbone parameters nearly fixed. It proposes a theoretical law (Equation (4)), validates a practical law (Equation (5), Figure 2), and analyzes inference efficiency (Figure 4).
  - It also shows how to retrofit existing models with minimal post‑training, including parameter‑efficient variants (Section 4.2; Figure 6).

## 3. Technical Approach
PARSCALE = parallel computation with learnable diversity + dynamic aggregation.

- Core pipeline (Figure 1(1), Equation (2))
  1. Take the same input and create P “versions” via learnable transformations: x1,…,xP.
     - Implementation: prefix tuning (Appendix A). Each stream gets its own set of soft “prefix” tokens injected into every attention layer’s key and value paths. These act as stream‑specific “hints” that steer the model to think differently.
     - Resource note: per stream this adds ≈0.2% additional parameters (Section “Implementation Details and Pivot Experiments” and Appendix A/Table 6).
     - KV cache: each stream keeps its own key/value cache during decoding; this is the main memory overhead of PARSCALE besides the tiny prefixes (Section 3.3).
  2. Run the same backbone fθ in parallel on all P versions. No changes to the backbone architecture.
  3. Aggregate the P output distributions with dynamic weights:
     - Compute weights w1,…,wP via a small MLP applied to the concatenated stream logits, then Softmax (Appendix A, Equation (6)).
     - To prevent “weight collapse” onto a single stream early in training, use label smoothing so every stream keeps a minimum weight (Appendix A, Equation (7), smoothing ϵ=0.1).
  4. The final next‑token distribution is the weighted sum ∑i wi fθ(xi) (Equation (2)). This is used both in training (loss backpropagates through all streams and the aggregator) and inference.

- Why this design
  - Prefix tuning cleanly creates diverse streams without changing the backbone and with tiny parameter cost (Appendix A/Table 6 shows alternatives—LoRA, BitFit, linear/static averaging—yield similar trends; the number of parallel computations P is the dominant factor).
  - Dynamic aggregation lets the model learn, token by token, which stream is most useful (Appendix A). Visualizations (Appendix I, Tables 31–33) show locality: nearby tokens often prefer the same stream, suggesting the aggregator learns stable “modes.”

- Theoretical framework (Section 3.1)
  - Start from the Chinchilla scaling law for each stream: after convergence, a model’s loss obeys Li = (A/N)^α + E (Lemma 3.1, Equation (3)), where:
    - N: number of parameters,
    - E: entropy of natural text,
    - A, α > 0: dataset/model constants.
  - Aggregate P streams by averaging their predicted distributions (simplified case with equal weights to make analysis tractable), and analyze the residuals relative to the true distribution.
  - Main result (Proposition 1):
    > L = ( A / (N · P^(1/α) · DIVERSITY) )^α + E,  with  DIVERSITY = [(P − 1)ρ + 1]^(-1/α)
    - ρ is the correlation between per‑stream residuals; lower correlation (more diversity) yields larger gains.
    - When ρ = 1 (identical predictions), PARSCALE degenerates to the single‑stream case.
    - When ρ = 0 (independent residuals), loss decreases like 1/P (discussion under Proposition 1).

- Practical, empirically fitted law (Section 3.2)
  - Observed losses follow a log trend in P. The paper fits:
    > L = ( A / (N · (k·log P + 1)) )^α + E  (Equation (5))
    - k quantifies benefit from parallel computation; larger k means more benefit from increasing P.
    - Fits on Stack‑V2‑Python and Pile achieve R^2 ≈ 0.998 (Figure 2). k is higher on code (0.393) than on the general corpus (0.334), suggesting parallel compute helps reasoning/coding more (Figure 2 captions).

- Training and implementation (Section 3.2; Appendix C)
  - Backbones: dense Qwen‑2.5 architecture, 36 layers fixed; vary width to get 0.5B–4.4B non‑embedding parameters (Appendix C/Table 7). Fixing depth keeps latency comparisons fair.
  - Pretraining for scaling-law fitting: 42B tokens, no repeat; P ∈ {1,2,4,8}; batch size 1024; sequence length 2048 (Section 3.2). Report EMA‑smoothed final training loss.
  - Inference-cost analysis: use the llm‑analysis framework to model GPU memory and latency across batch sizes 1–8 with various input/output lengths (Section 3.3; Figure 4).

- Two‑stage training to reduce training compute (Section 4.1)
  - Stage 1: standard pretraining on 1T tokens (no PARSCALE).
  - Stage 2: PARSCALE on 20B tokens (2% extra data) to teach the aggregator/prefixes; loss spike from random init vanishes after ~0.0002T tokens, then steady gains follow the same log trend (Figure 5).

- Applying to off‑the‑shelf models (Section 4.2)
  - Continual pretraining: start from Qwen‑2.5‑3B and continue on Pile/Stack‑V2; larger P lowers loss (Figures 6a, 6b).
  - PEFT variant: freeze backbone; only train prefixes + aggregator. Code generation improves as P increases even with a frozen 3B backbone (Figure 6c).

## 4. Key Insights and Innovations
- A third scaling axis: parallel compute with learnable diversity
  - What’s new: A general mechanism to increase compute at both training and inference without increasing backbone parameters or output length: P parallel streams with learned input perturbations and learned token‑wise aggregation (Figure 1(1), Equation (2)).
  - Why it matters: It reframes compute as a first‑class scaling handle, reusable with existing models and data.

- A parallel scaling law linking compute and parameters (Section 3)
  - Theoretical generalization (Equation (4)) ties loss to P, N, and the diversity term involving correlation ρ of residuals.
  - Practical law (Equation (5)) shows effective capacity grows like N·(k log P + 1), i.e., scaling P behaves like adding O(N log P) parameters (Figure 2). This is a conceptual step beyond prior scaling laws that only couple N and data.

- Efficiency at inference via parameter reuse (Section 3.3; Figure 4)
  - Result: For the same loss improvement, PARSCALE has far smaller memory growth and modest latency growth than parameter scaling, especially at small batch sizes typical of edge use.
  - Example: With batch size 1 and a 1.6B backbone, increasing P to 8 uses “22× less memory increase” and “6× less latency increase” than parameter scaling to a comparable capacity (Figure 4 annotations; Section 3.3).

- A practical recipe to retrofit models and control capacity at runtime (Sections 4.1–4.2)
  - Two‑stage training reduces the extra training compute to only 2% more tokens (Figure 5).
  - “Dynamic parallel scaling”: freeze the backbone and switch P at deployment to trade capacity vs. speed (Figure 6c).

## 5. Experimental Analysis
- Evaluation setup
  - Scaling‑law fitting: pretrain 24 runs per dataset (6 N values × 4 P values) on 42B tokens of Stack‑V2‑Python (code‑heavy) and Pile (general text). Fit Equation (5) using Huber loss and L‑BFGS (Appendix E). The fitted curves and parameters are in Figure 2, with R^2 up to 0.998.
  - Downstream tasks after 42B‑token pretraining:
    - Code: HumanEval(+), MBPP(+) (Tables 2, 16–23).
    - General: WinoGrande, HellaSwag, OpenBookQA, PiQA, ARC, SciQ (Tables 3, 24–29).
  - Inference efficiency: GPU memory vs. loss and latency vs. loss across batch sizes 1, 2, 4, 8 and multiple sequence lengths (Figure 4).
  - Large‑scale, production‑like training: 1.8B model trained on 1T tokens (Stage 1), then 20B tokens with PARSCALE (Stage 2). Evaluated on 7 general tasks, 3 math tasks (GSM8K, GSM8K‑CoT, MATH), and 8 code tasks (Table 4). Instruction tuning with 1M SmolTalk examples (Table 5).
  - Off‑the‑shelf Qwen‑2.5‑3B: continual pretraining (loss curves) and PEFT (code gen metrics) in Figure 6.

- Main quantitative findings
  - Scaling law and “compute ≈ O(log P) parameter gain”
    - Fitted law: L = (A / (N·(k log P + 1)))^α + E fits tightly (Figure 2). k is larger on Stack‑V2‑Python (0.3935) than on Pile (0.3345), suggesting parallel compute helps reasoning/coding more than memorization (Figure 2 captions).
    - Predicted loss contours (Figure 3) flatten as N grows, meaning larger backbones benefit more from increasing P at fixed N.
  - Downstream after 42B‑token pretraining
    - Code tasks show stronger gains. Example: at 1.6B parameters, average code score rises from 33.9% (P=1) to 39.1% (P=8); at 4.4B, from 39.2% to 45.4% (Table 2).
    - General tasks also improve but less. Example: at 1.6B, average general score rises from 53.1% (P=1) to 55.7% (P=8) (Table 3).
    - Claim supported: “parameters mainly help memorization; compute mainly helps reasoning” (Section 3.2 discussion with Figure 2).
  - Inference efficiency (Figure 4)
    - Memory: For a fixed loss target, PARSCALE adds far less memory than parameter scaling across batch sizes. The gain is largest at small batches because memory is dominated by parameters; PARSCALE adds only prefixes and more KV cache.
    - Latency: At batch size 1, scaling P to 8 in a 1.6B model costs roughly “6× less latency increase” than scaling parameters to match loss (Figure 4e caption text on plot). At larger batches the gap narrows as decoding becomes compute‑bound, but PARSCALE remains favorable up to batch size 8 (Figures 4e–4h).
  - Two‑stage training on 1T + 20B tokens (Table 4, Figure 5)
    - Adaptation is quick: after ~0.0002T tokens in Stage 2, the extra parameters (prefixes/aggregator) begin to help and the loss drops below P=1 (Figure 5).
    - Performance: from P=1 to P=8, average increases are +2.6% on general tasks, +7.3% on math, +4.3% on code (Table 4). GSM8K rises from 28.7% to 38.4% (absolute +9.7; relative +34%).
    - With CoT, GSM8K improves further (Table 4 bottom block), showing PARSCALE can complement inference‑time scaling.
  - Instruction tuning (Table 5)
    - IFEval improves from 54.1% (P=1) to 59.5% (P=8). MMLU 5‑shot rises from 34.2% to 41.7%. GSM8K 4‑shot rises from 50.3% to 56.1%.
  - Off‑the‑shelf, frozen backbone (Figure 6)
    - Continual pretraining reduces loss more for larger P on both Stack‑V2‑Python and Pile (Figures 6a, 6b).
    - PEFT (frozen backbone): averaged code Pass@1 increases from 47.4% to 53.0% and Pass@10 from 73.1% to 78.2% as P goes from 1 to 8 (Figure 6c).
  - Additional analyses
    - Pivot experiments (Appendix A/Table 6): the specific input transformation (prefix vs. LoRA/BitFit) and aggregator choice matter little compared to increasing P; dynamic weighted sum with label smoothing performs best.
    - Repeated data (Appendix D/Figure 7): with OpenWebText repeats, PARSCALE overfits less than parameter scaling (for the same loss target), hinting at regularization benefits when data is limited.
    - Beam search vs. PARSCALE (Appendix H/Table 30): increasing beams does not replicate PARSCALE’s gains, and can even harm performance at larger beam sizes, underscoring the value of training‑time parallel compute rather than inference‑only resampling.

- Do experiments support the claims?
  - The scaling-law fit is strong and consistent across two corpora (Figure 2; Appendix E/Tables 8–11).
  - Efficiency comparisons are systematic across batch sizes and sequence lengths (Figure 4).
  - Gains are largest on coding and math tasks, aligning with the interpretation that parallel compute helps reasoning skills (Tables 2–4).
  - The theoretical formula with an explicit correlation ρ (Equation (4)) fits less tightly than the empirical log law (Appendix E/Tables 12–15), leaving open theoretical questions about why benefits scale log‑linearly with P (acknowledged in Section 6).

## 6. Limitations and Trade-offs
- Compute cost
  - Training compute increases roughly linearly with P because there are P forward/backward passes (Section 4.1). The two‑stage recipe limits extra data to 2%, but the second stage still uses P× FLOPs.
  - Inference compute also grows with P, although latency remains favorable at small batches due to GPU‑friendly parallelism (Section 3.3). At larger batch sizes, decoding becomes compute‑bound and the latency advantages shrink (Figures 4f–4h).

- Memory behavior
  - KV cache scales with P. For very long contexts or very large P, this can become non‑negligible, especially at higher batch sizes (Section 3.3 discussion).

- Scope of validation
  - P is evaluated up to 8; behavior for P ≫ 8 is unknown. Section 6 explicitly lists whether performance saturates and if growth can exceed O(log P) as open questions.
  - Backbones are up to ~4.4B non‑embedding parameters for scaling‑law fitting, with a deeper dive at 1.8B for large‑data training (Sections 3–4). Extrapolation to much larger models is untested here.

- Theory vs. practice
  - The diversity term involves an average correlation ρ of residuals; ρ is not modeled or controlled directly, and the empirical log‑P fit outperforms the closed‑form with constant ρ (Appendix E).
  - The inference cost analysis uses a standardized framework (llm‑analysis) and averaged settings; real‑hardware variance and kernel optimizations could shift absolute numbers (Section 3.3).

- Task mix and data
  - Gains on general knowledge tasks are smaller than on reasoning/coding (Tables 2–4). If an application is dominated by memorization, parameter scaling may still be more cost‑effective.

- Implementation complexity
  - Serving requires running P parallel streams with separate KV caches and a token‑wise aggregator. While feasible, it is more complex than single‑stream decoding.

## 7. Implications and Future Directions
- How it changes the landscape
  - Establishes “parallel compute” as a third, practical scaling axis alongside parameters and data. The empirical law (Equation (5)) lets practitioners predict capacity gains from increasing P for a fixed backbone (Figure 3 contours).
  - Provides an inference‑efficient path to higher capacity for memory‑limited scenarios (smartphones, cars, robots) where batch sizes are small (Section 3.3).

- Follow‑up research directions (Section 6)
  - Theory: explain why benefits scale ∝ log P; characterize or actively optimize the residual correlation ρ across streams; study limits for large P.
  - Budget‑aware design: extend inference‑optimal scaling laws to include P, N, memory, and latency constraints (Discussion, first paragraph of Section 6).
  - Training schedules: optimize the two‑stage boundary (how much second‑stage data is needed for different N and P).

- Practical applications
  - Retrofitting pre‑trained models: add prefixes + aggregator, run a short PARSCALE stage, and optionally keep the backbone frozen for dynamic capacity scaling at deployment (Section 4.2; Figure 6c).
  - Reasoning‑heavy workloads: coding assistants and math solvers benefit most (Tables 2 and 4). PARSCALE also stacks with inference‑time techniques like chain‑of‑thought (Table 4, bottom).
  - Systems design: combine PARSCALE (memory‑friendly) with sparse MoE (latency‑friendly) to balance resource trade‑offs (Section 6).

Key citations to anchor statements:
- Mechanism: Equation (2), Appendix A (prefix tuning, dynamic weighting), Figure 1(1).
- Theory: Lemma 3.1 (Equation (3)), Proposition 1 (Equation (4)); empirical law Equation (5).
- Fits and trends: Figure 2 (A, k, α, E; R^2), Figure 3 contours.
- Efficiency: Section 3.3; Figure 4 (memory/latency arrows and annotations).
- Two‑stage training: Section 4.1; Figure 5; Table 4.
- Instruction tuning: Table 5.
- Off‑the‑shelf and PEFT: Section 4.2; Figure 6.
- Ablations and robustness: Appendix A (Table 6), Appendix D (Figure 7), Appendix H (Table 30), Appendix I (Tables 31–33).

Definitions used above (selectively):
- `prefix tuning`: adding a small number of learnable “prefix” embeddings into the keys/values of each attention layer so the same backbone can produce different behaviors per stream.
- `KV cache`: cached keys/values per layer used during autoregressive decoding; here each stream keeps its own cache.
- `Chinchilla scaling law`: an empirical relation that, after convergence, a language model’s cross‑entropy loss decreases as a power law in 1/N (Equation (3)).
- `DIVERSITY` and `ρ`: terms from Proposition 1 quantifying how similar the per‑stream residual errors are; lower ρ (less correlated errors) yields more benefit from averaging streams.
