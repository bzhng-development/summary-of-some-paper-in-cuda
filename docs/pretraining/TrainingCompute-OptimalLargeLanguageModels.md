# Training Compute-Optimal Large Language Models

**ArXiv:** [2203.15556](https://arxiv.org/abs/2203.15556)

## 🎯 Pitch

This paper presents a rigorous empirical study into the optimal allocation of training compute between model size and data for large language models. The authors demonstrate, through training over 400 models and three complementary analyses, that for a fixed compute budget, model size and number of training tokens should be scaled equally—contradicting previous dogma. This insight led to the creation of Chinchilla, a smaller but better-trained model that outperforms much larger models like Gopher and GPT-3 on a wide range of tasks, showing that smarter scaling dramatically boosts efficiency and performance in large language model development.

---

## 1. Executive Summary (2-3 sentences)
This paper asks a practical question: if you know your training compute budget, what is the best way to split it between making a language model bigger versus showing it more data? By training more than 400 models and building three complementary analyses, the paper finds a simple rule: for compute‑optimal training, scale model size and training tokens in roughly equal proportion. Using this rule, the authors train Chinchilla (70B parameters on 1.4T tokens) with the same compute as Gopher (280B on 300B tokens) and show consistent, often large, accuracy gains across many benchmarks (Sections 3–4; Figure 1; Table 1).

## 2. Context and Motivation
- Problem addressed
  - Given a fixed training compute budget `C` (measured in FLOPs, i.e., the number of floating‑point operations), what combination of model parameters `N` and training tokens `D` minimizes final pre‑training loss `L(N, D)`? The paper formalizes this as minimizing `L(N, D)` subject to `FLOPs(N, D) = C` (Equation 1, Section 3).

- Why it matters
  - Training and inference are extremely costly for large language models (LLMs), and in practice one often knows the available accelerators and time upfront (Section 1). Choosing `N` and `D` suboptimally wastes compute and yields worse models. Moreover, inference and fine‑tuning cost scale with model size, so compute‑optimal training can reduce downstream costs (Section 4).

- Prior approaches and their gaps
  - Earlier scaling law work (Kaplan et al., 2020) recommended growing `N` much faster than `D` under more compute (roughly `N ∝ C^0.73`, `D ∝ C^0.27`; Table 2). Many subsequent LLMs thus fixed the number of tokens near ~300B while dramatically increasing parameters (Table 1). This practice likely under‑trained models on data, given the available compute (Sections 1–2).

- How this paper positions itself
  - Re-examines compute‑optimal scaling using a richer experimental design that varies both `N` and `D`, tunes learning‑rate schedule length to match training horizon, and fits three independent estimators of the compute‑optimal frontier (Section 3; Figures 2–4). It then validates the prediction by training a new model, Chinchilla, that is smaller but trained on more tokens than Gopher using the same compute (Section 4).

## 3. Technical Approach
The goal is to estimate the compute‑optimal allocation functions `N_opt(C)` and `D_opt(C)` that minimize final pre‑training loss subject to a fixed compute budget. The paper builds three complementary estimators.

Key concepts (defined where first used):
- `FLOPs` (floating point operations): a standardized measure of compute used during training. For transformers, a common approximation is `FLOPs ≈ 6 N D` (Section 3.3; Equation below Figure 4). The paper also provides a detailed FLOP accounting and shows it closely matches `6ND` (Appendix F; Table A4).
- `Tokens`: total number of training subword tokens processed during pre‑training.
- `IsoFLOP profile`: a set of experiments that hold total compute `C` fixed while varying `N` and `D` to map loss across model sizes for that compute level (Section 3.2; Figure 3).
- `Training curve envelope`: for multiple runs at different `N` and training horizons, the envelope is the pointwise minimum loss as a function of compute, revealing which `(N, D)` pair is best at each `C` (Section 3.1; Figure 2 left).

A. Approach 1 — Minimum‑over‑training‑curves (Section 3.1; Figure 2)
- Design
  - Train model families with fixed parameter counts from ~70M to 10B, each for four different training horizons (implemented via cosine learning‑rate schedules decaying by 10× over horizons that vary by 16×; Section 3.1 and Appendix B).
  - For every run, smooth and interpolate the training loss as a function of FLOPs, producing a continuous mapping `C → L` for that run.
  - For each compute level, take the minimum loss across all runs to build the envelope, and record the `(N, D)` that achieved it.
  - Fit power laws `N_opt ∝ C^a`, `D_opt ∝ C^b` to the envelope points (Figure 2 center/right).
- Important implementation choice
  - The cosine schedule length is matched to the planned number of training tokens; overestimating the schedule length by >25% harms performance (Appendix B; Figure A1). This avoids systematically over‑penalizing shorter‑horizon runs, a key difference from Kaplan et al. (2020).

B. Approach 2 — IsoFLOP profiles (Section 3.2; Figure 3)
- Design
  - Select nine target compute budgets from `6×10^18` to `3×10^21` FLOPs.
  - For each compute level, train a range of model sizes (up to 16B parameters here), setting the number of tokens so that total FLOPs is constant. Use a cosine schedule length that matches the implied number of tokens (Section 3.2).
  - Plot final loss versus parameters for each compute level. Each curve shows a clear U‑shaped “valley,” revealing an optimal parameter count for that budget (Figure 3 left). Fit a parabola to locate the minimizer.
  - Fit power laws relating the optimal `N` and `D` to `C` (Figure 3 center/right).

C. Approach 3 — Parametric loss model and closed‑form frontier (Section 3.3; Figures 4; Equations 2 and 4)
- Model
  - Propose a simple, interpretable form for final loss after training on `D` tokens with a model of size `N`:
    - `L̂(N, D) = E + A / N^α + B / D^β` (Equation 2).
    - Terms correspond to: irreducible entropy `E`, functional approximation gap due to finite `N`, and optimization/statistical suboptimality from finite `D` (Section D.2).
- Fitting
  - Fit parameters `(A, B, E, α, β)` by minimizing a robust Huber loss between `log L̂(N,D)` and observed `log L` over all experiments from Approaches 1–2, using L‑BFGS with a grid of initializations (Section 3.3; Equation 3; Section D.2).
  - The fitted exponents are `α ≈ 0.34`, `β ≈ 0.28` in the decomposition (Equation 10), which imply a roughly balanced role of `N` and `D`.
- Compute‑optimal frontier
  - Under the approximation `FLOPs(N, D) ≈ 6 N D`, minimizing `L̂(N, D)` under a fixed `C` yields closed‑form:
    - `N_opt(C) = G (C/6)^a`, `D_opt(C) = G^(-1) (C/6)^b`, with `a = β/(α+β)`, `b = α/(α+β)`, and `G = ((αA)/(βB))^(1/(α+β))` (Equation 4). This defines the “efficient frontier” (Figure 4 left, blue curve).

D. Cross‑dataset sanity checks
- The IsoFLOP analysis replicated on C4 and on a GitHub code dataset shows very similar scaling exponents (Appendix C; Figure A2; Table A2), suggesting the conclusion is not specific to MassiveText as long as training stays in the <1 epoch regime.

E. Training Chinchilla (Section 4; Table 4; Appendix A)
- To test the prediction, the authors train `Chinchilla` (70B parameters) on 1.4 trillion tokens from MassiveText with an adjusted subset mix (Table A1), using the same total compute as `Gopher` (≈`5.76×10^23` FLOPs; Figure 2 center/right, green marker; Figure 4 vertical dashed line).
- Architectural and training differences vs. Gopher:
  - 80 layers; `d_model=8192`, 64 heads, `kv_size=128`; AdamW optimizer (instead of Adam), bfloat16 compute with float32 master weights, a slightly modified tokenizer (no NFKC normalization) (Table 4; Section 4.1; Appendix G).
  - Batch size doubles mid‑training (1.5M→3M tokens), cosine LR schedule, same training infrastructure (TPU + JAX/Haiku) (Section 4.1).

## 4. Key Insights and Innovations
- Equal‑proportion compute scaling of parameters and data is optimal
  - All three methods independently estimate `a ≈ b ≈ 0.5` for `N_opt ∝ C^a` and `D_opt ∝ C^b` (Table 2). This contrasts sharply with Kaplan et al. (2020) (`a=0.73`, `b=0.27`) and implies that recent mega‑models are under‑trained on data relative to their size (Figure 1; Table 3).
  - Significance: If you double compute, you should roughly double both model size and the number of tokens, rather than skewing heavily to size. This changes how future budgets should be spent.

- Methodological correction: match learning‑rate schedule length to training horizon
  - The paper shows that setting the cosine schedule too long (e.g., 2× the training steps) degrades loss, and even a 25% overshoot is harmful (Appendix B; Figure A1). Prior work fixed schedule length across runs, systematically disadvantaging shorter‑horizon experiments and biasing conclusions toward larger `N`.

- Three ways to recover the same frontier
  - The training‑curve envelope (Figure 2), IsoFLOP valleys (Figure 3), and the parametric loss surface with a closed‑form frontier (Figure 4) triangulate to the same conclusion. This redundancy strengthens the claim beyond a single fitting procedure.

- Practical validation at scale: Chinchilla beats much larger models using the same compute
  - Training a 70B model on 1.4T tokens (4× the data, 4× smaller model vs. Gopher), but with equal compute, yields consistent gains across language modeling, reasoning, reading comprehension, and QA (Section 4.2; Figures 5–7; Tables 6–9). This anchors the scaling rule in a head‑to‑head comparison at scale.

- Quantified “how under‑trained” current LLMs are
  - Table 3 projects the tokens and FLOPs needed to train compute‑optimal models at various sizes. Example: a 175B model would need ≈`3.85×10^24` FLOPs and ≈`3.7T` tokens; a 280B model would need ≈`9.90×10^24` FLOPs and ≈`5.9T` tokens. Most current models used far fewer tokens.

## 5. Experimental Analysis
Evaluation methodology (Section 4.2; Table 5)
- Datasets and tasks
  - Language modeling: Wikitext‑103, The Pile with 19 subsets (e.g., arXiv, GitHub, stackexchange), C4, etc. (Figure 5; Table A5).
  - Reasoning/knowledge: MMLU (57 tasks), BIG‑bench (62 tasks) (Tables 6; A6; A7).
  - Reading comprehension: LAMBADA, RACE‑m/h (Table 7).
  - Closed‑book QA: Natural Questions (NQ), TriviaQA (both unfiltered and filtered) (Table 9).
  - Bias/toxicity: Winogender coreference and unconditional toxicity via PerspectiveAPI (Table 10; Section 4.2.7).

- Metrics
  - Language modeling: bits‑per‑byte (bpb), perplexity.
  - QA/MC: accuracy; for NQ/TriviaQA, also k‑shot settings.
  - Toxicity: mean/percentile toxicity scores.

Main results and comparisons
- Language modeling quality improves across the board
  - On all 19 Pile subsets, Chinchilla lowers bpb compared to Gopher (Figure 5; Table A5). Example: arXiv bpb `0.627` vs `0.662`; GitHub `0.337` vs `0.377`.
  - Wikitext‑103 perplexity: `7.16` (Chinchilla) vs `7.75` (Gopher) (Section 4.2.1). The paper notes potential train/test leakage due to 4× more data and thus cautions against over‑interpreting raw LM metrics (Section 4.2.1).

- MMLU: state‑of‑the‑art at the time, with broad gains
  - Quote: “Chinchilla reaches a state‑of‑the‑art average accuracy of 67.6%… greater than a 7.6% improvement over Gopher” (Table 6; Figure 6). It improves on 51/57 tasks, ties on 2, and loses on 4 (Figure 6; Table A6). Notably >90% on four subjects.

- BIG‑bench: large average improvement
  - Average accuracy 65.1% vs 54.4% for Gopher (+10.7%), with worse performance on only 4 of 62 tasks (Figure 7; Table A7).

- Reading comprehension
  - LAMBADA (zero‑shot): 77.4% vs 74.5% (Table 7).
  - RACE‑m/h (few‑shot): 86.8%/82.3% vs 75.1%/71.6% (Table 7).

- Closed‑book QA
  - NQ: 5‑shot 31.5% vs 24.5%; 64‑shot 35.5% vs 28.2%; also stronger than GPT‑3’s 29.9% at 64‑shot (Table 9).
  - TriviaQA (unfiltered/test): 0‑shot 67.0% vs 52.8%; 64‑shot 72.3% vs 61.3%; beats GPT‑3’s 71.2% at 64‑shot (Table 9).
  - TriviaQA (filtered/dev): 64‑shot 64.6% vs 57.2%, now within 7.9% of open‑book SOTA (FiD + distillation 72.5%) though the setting differs (Table 9).

- Bias/toxicity analyses
  - Winogender coreference: Higher accuracy overall (78.3% vs 71.4%), with the largest gain on female “gotcha” examples (+10.0%), but improvements are uneven (Table 10).
  - Unprompted toxicity: Mean and 95th percentile toxicity scores are very similar to Gopher (0.087 vs 0.081 mean; 0.238 vs 0.230 at 95th percentile), suggesting better LM loss does not increase unconditional toxicity (Section 4.2.7).

Support strength and ablations
- The scaling claim is supported by three independent estimation methods (Table 2; Figures 2–4) and a targeted, same‑compute validation at scale (Section 4). The paper also provides a direct head‑to‑head small‑compute test against Kaplan‑recommended size (`≈2.80B` vs `4.74B`) showing the compute‑optimal recommendation performs better (Appendix D.4; Figure A4).
- Robustness: The IsoFLOP scaling result replicated on C4 and GitHub code datasets with similar exponents (Appendix C; Table A2).
- Hyperparameter insight: Matching the cosine schedule to training tokens is empirically critical (Appendix B; Figure A1).
- Implementation differences (AdamW vs Adam, optimizer precision) are documented and shown to help, but they are not the main driver of gains (Appendix G; Figures A6–A7). The primary difference remains the compute‑optimal size/data reallocation.

## 6. Limitations and Trade-offs
- Assumptions about the regime
  - Analyses and experiments assume the “infinite data” sense that training tokens are less than the size of the whole corpus (footnote 2, Section 1), and most runs are <1 epoch (Section 5). The multi‑epoch regime is not studied, leaving open whether the same scaling holds when reusing tokens.

- Power‑law modeling and curvature
  - The compute‑optimal frontier is fit by power laws, yet the paper observes concavity (negative curvature) at higher budgets (Appendix E; Figure A5). This could make very large‑compute extrapolations optimistic about `N_opt`, potentially favoring even smaller models at extreme budgets.

- Limited large‑scale replications
  - Only two comparable high‑compute runs exist (Gopher and Chinchilla), so validation at intermediate scales is limited by cost (Section 5).

- Data requirements and quality constraints
  - Compute‑optimal training at larger `N` would require trillions of additional high‑quality tokens (Table 3). Acquiring and curating such datasets is hard; train–test leakage risks increase (Section 5).

- Scope of analysis
  - The study targets dense autoregressive transformers. Routed Mixture‑of‑Experts or retrieval‑augmented models may change the compute–data trade‑off (Section 2), and the paper does not jointly optimize other axes (e.g., depth‑to‑width, batch size) beyond standard heuristics (Section 2, “Estimating hyperparameters”).

- Safety
  - While unconditional toxicity does not worsen, the model still reflects biases (Winogender results; Section 4.2.7). Safety, privacy, and fairness considerations remain and may scale with the volume of data used (Section 5; model card Table A8).

## 7. Implications and Future Directions
- How this changes practice
  - Budget planning: When you know your compute, plan to scale parameters and tokens together (roughly 1:1 in log space). This reorients training roadmaps that had emphasized parameter count growth with relatively fixed data budgets (Table 2; Figure 1).
  - Downstream efficiency: Smaller, better‑trained models can outperform much larger ones while cutting inference and fine‑tuning costs, easing deployment on modest hardware (Section 4).

- What this enables
  - Dataset strategy: The field should invest more in collecting and curating larger, higher‑quality corpora; results suggest data quality will be increasingly pivotal (Section 5).
  - Architectural choices: Since equal scaling improves compute efficiency, combining it with retrieval (to effectively increase `D`) or MoE routing (to increase effective `N` at similar FLOPs) could yield further gains; the paper’s methodology (Approaches 1–3) can be reused to find new compute‑optimal frontiers in these settings (Sections 2–3).

- Research directions
  - Multi‑epoch and curriculum: Test whether the scaling rule persists when the same tokens are revisited, and how curricula or deduplication affect the `1:1` scaling.
  - Frontier curvature: Model and exploit the observed curvature at high compute (Appendix E) to refine `N_opt(C)` and `D_opt(C)` predictions.
  - Joint hyperparameter optimization: Extend the loss model to include batch size, optimizer, depth‑to‑width ratio, and schedule shape; the parametric framework (Equation 2) is amenable to such extensions.
  - Safety‑aware data scaling: Develop methods for toxicity/bias mitigation that do not suppress minority dialects or viewpoints (Section 4.2.7, Appendix H), especially as data scales to trillions of tokens.

- Practical applications
  - Pre‑training pipelines: Organizations can re‑target existing compute to train smaller but better‑fed models, improving accuracy and reducing inference memory/latency.
  - Model selection: Table 3 provides a recipe book for how many tokens to plan for a given `N` to lie on the compute‑optimal frontier (e.g., ~1.5T tokens for ~67B parameters; ~3.7T for 175B).

> Core takeaway, grounded in three analyses (Table 2; Figures 2–4) and a direct at‑scale test (Section 4): with a fixed compute budget, you get the best language model by growing parameters and data together. Chinchilla’s across‑the‑board improvements over much larger models (Tables 6–9; Figures 5–7) demonstrate the practical payoff of this rule.
