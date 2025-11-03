# ReFT: Representation Finetuning for Language Models

**ArXiv:** [2404.03592](https://arxiv.org/abs/2404.03592)

## 🎯 Pitch

ReFT introduces a new paradigm for adapting large language models by learning to intervene directly on their hidden representations, rather than updating model weights. Its flagship method, LoReFT, delivers top-tier performance across a range of tasks while being up to 65× more parameter-efficient than leading weight-based approaches like LoRA. This approach not only unlocks practical advantages—such as faster, memory-light finetuning and deployability—but also bridges model interpretability and control, making deeper insights about a model’s internal workings actionable for efficient, tailored adaptation.

---

## 1. Executive Summary (2-3 sentences)
This paper introduces Representation Finetuning (ReFT), a family of methods that adapt large language models (LLMs) by editing a small number of hidden representations during the forward pass instead of updating model weights. Its main instance, Low-rank Linear Subspace ReFT (`LoReFT`), learns low-rank, subspace-limited edits that are 15×–65× more parameter-efficient than popular weight-based methods like LoRA, while matching or exceeding their performance on many benchmarks (Figure 1, Tables 1–4).

## 2. Context and Motivation
- Problem addressed
  - Finetuning large LLMs is expensive; Parameter-Efficient Fine-Tuning (`PEFT`) methods reduce cost by updating a small subset of weights. However, nearly all strong PEFTs still change weights, not representations. Section 1 argues that internal representations encode rich, causally-relevant semantics (supported by interpretability work), so directly intervening on them may be a more powerful and efficient adaptation mechanism.
- Why it matters
  - Practical: Lower memory, faster training, easier model deployment. ReFT can learn tiny “edits” (kilobytes to megabytes) that steer a frozen base model at inference-time, making it attractive for on-device and multi-task deployment.
  - Scientific: Bridges model interpretability (which manipulates representations to test causal mechanisms) with model control, turning insights like “concepts live in linear subspaces” into an efficient finetuning procedure (Sections 3.1–3.2).
- Prior approaches and their gaps (Section 2)
  - Adapter-based PEFTs insert small modules per layer; they perform well but add inference-time latency and parameters that cannot be merged into the base weights.
  - LoRA/DoRA update low-rank weight decompositions; strong performance and no inference overhead after merging, but still require storing and merging weight deltas.
  - Prompt-based methods (e.g., prefix-tuning) tune soft tokens; easy to apply but often underperform and increase inference cost with longer prompts.
  - Representation editing (e.g., activation addition, RED) can steer models but typically uses fixed vectors or per-layer biases; they do not offer a comprehensive, learnable, task-specific training framework equivalent to finetuning.
- Positioning
  - ReFT generalizes representation editing into a finetuning framework: learn small, task-specific “interventions” that modify selected hidden states at chosen layers and positions during inference (Definitions 3.1–3.2). `LoReFT` is a principled, low-rank subspace version that inherits causal interpretability ideas (DAS/DII) and scales to strong LLMs (Sections 3.1–3.2, Figure 2).

## 3. Technical Approach
ReFT in one sentence: keep the base model frozen, and learn tiny functions that overwrite a few hidden representations at selected layers/positions during the forward pass.

- Core concepts (Section 3.3)
  - `Intervention`: A tuple ⟨`Φ`, `P`, `l`⟩ (Definition 3.1), where `Φ` is a learnable edit function applied to hidden states at a set of token positions `P` in layer `l`. After layer `l` computes its hidden states h^(l), the model overwrites h_p^(l) with Φ(h_p^(l)) for p ∈ P (Equation 7). Later layers then consume the edited states.
  - `ReFT method`: A set of non-overlapping interventions {I1, …, If} (Definition 3.2). Each has its own parameters; together they implement the full finetuning behavior.

- Why interventions instead of weight updates?
  - Interventional interpretability shows that human-meaningful features (gender, number, attributes, logic) are often linearly encoded in hidden states; causal interchange interventions can steer predictions by altering low-dimensional subspaces (Section 3.1). ReFT turns this into trainable finetuning.

- The interpretability link (Section 3.1)
  - Distributed Interchange Intervention (`DII`): swaps part of a representation in a low-dimensional subspace to emulate a counterfactual. If b is a base hidden vector and s is a counterfactual vector, and `R` is an r×d projection with orthonormal rows (a basis for an r-dimensional subspace), then:
    - DII(b, s, R) = b + Rᵀ(Rs − Rb)  (Equation 1).
  - Distributed Alignment Search (`DAS`) learns `R` by maximizing the probability of “desired” counterfactual outputs when such subspace swaps are performed. This motivates learning the subspace and the source of the counterfactual signal.

- LoReFT: Low-rank Linear Subspace ReFT (Section 3.2)
  - Goal: Learn how to edit only the subspace that matters.
  - Edit rule:
    - Φ_LoReFT(h) = h + Rᵀ(Wh + b − Rh)  (Equation 2).
  - Intuition:
    - `R` (r×d, orthonormal rows) defines the low-dimensional subspace to edit.
    - `Rh` are the current coordinates in that subspace.
    - `Wh + b` (dimension r) is a learned, task-specific “target” for those coordinates, computed as a simple linear projection plus bias.
    - The update adds Rᵀ(target − current), i.e., it changes only the chosen subspace and leaves components orthogonal to that subspace untouched. Figure 2 (right) visualizes this: the edit is confined to the subspace spanned by rows of `R`.
  - What is learned? ϕ = {R, W, b}. The base model remains frozen.

- DiReFT: a simpler ablation (Section 3.2)
  - Edit rule:
    - Φ_DiReFT(h) = h + W₂ᵀ(W₁h + b)  (Equation 3).
  - Differences:
    - No orthogonality constraint and no explicit “difference” term (−Rh).
    - Conceptually like applying LoRA directly on hidden states at specific positions. It’s even lighter-weight but typically slightly worse than LoReFT in performance (Tables 1–4).

- Where and when to edit (Section 4.1)
  - The method edits only `p` prefix and `s` suffix token positions (e.g., first 3 and last 3 tokens), and only in a small set of layers `L`. Optionally, tie edit parameters across positions in a layer to save parameters.
  - This makes inference cost independent of full prompt length (because the number of edited positions is fixed), unlike soft prompting that grows with prompt length.

- Training objectives (Section 3.2)
  - Generation tasks (decoder-only or encoder–decoder): minimize token-level cross-entropy with teacher forcing (Equation 4).
  - Classification (encoder-only): add a small classifier on top of the `[CLS]` (first token) representation and minimize cross-entropy (Equations 5–6). Only the classifier and ReFT parameters are trained.

- Implementation and library (Appendix A)
  - `pyreft` (on top of `pyvene`) provides generic training/inference hooks for interventions into HuggingFace models. The code snippet in Appendix A shows how to attach a single LoReFT intervention to a Llama-2 layer’s residual stream.

- Inference overhead (Appendix H)
  - Unlike LoRA, ReFT edits can’t be merged into weights, so they incur a small runtime cost. However, because edits are applied to a fixed number of positions and select layers, the overhead is modest. Figure 11 shows that with rank-8 and 10 intervened layers applied to the last prompt token, the added end-to-end latency is roughly 0.05 seconds for generating 256 tokens, with a near-linear relation to prompt length and rank.

## 4. Key Insights and Innovations
- A. Finetuning by editing hidden representations—formulated as a learnable, low-rank, subspace intervention (Sections 3.1–3.3).
  - Difference from prior work: Moves beyond fixed, hand-crafted steering vectors (e.g., activation addition) by learning the subspace `R` and the target mapping `Wh + b` jointly for the task.
  - Significance: Subspace-limited edits ensure control and interpretability—only a known r-dimensional slice of the representation is modified.

- B. The ReFT framework unifies and generalizes many representation editing techniques (Section 3.3; Appendix B).
  - Many existing methods (e.g., RED, activation addition, RepE) fit as special cases of `Φ` and `⟨Φ, P, l⟩`. This makes ReFT a common language for representation-based control and a drop-in alternative to PEFTs.

- C. Strong instance `LoReFT`: orthogonally constrained, difference-based subspace edits (Equation 2).
  - Why it matters: The orthogonality of `R` and the difference term (−Rh) stabilize learning and keep edits confined, often outperforming simpler `DiReFT` (Tables 1, 3, 4; Appendix E Table 17).

- D. Extreme parameter efficiency with competitive or state-of-the-art performance (Figure 1).
  - LoReFT uses 0.014–0.031% of base model parameters in experiments, yet reaches or surpasses LoRA/DoRA and adapters on commonsense, instruction tuning, and GLUE (Tables 1, 3, 4).

- E. New capabilities from subspace training: composability and memorization (Appendix F–G).
  - A single rank-1 subspace edit can memorize long sequences (up to 2,048 tokens with 100% prefix match; Figures 3–4) or up to 256 input–output mappings (Figures 9–10). Subspace partitions can be trained for different skills and composed at inference (Appendix G.1), hinting at modular, puzzle-piece-like control.

## 5. Experimental Analysis
- Evaluation setup (Section 4; Appendices C–D)
  - Models: LLaMA-1 7B/13B, Llama-2 7B, Llama-3 8B; RoBERTa-base/large.
  - Benchmarks:
    - Commonsense reasoning: 8 datasets (BoolQ, PIQA, SIQA, HellaSwag, WinoGrande, ARC-e/c, OBQA), trained jointly on a combined set (COMMONSENSE170K) (Section 4.2).
    - Arithmetic reasoning: 4 datasets (AQuA, GSM8K, MAWPS, SVAMP), trained on MATH10K with chain-of-thought style outputs (Section 4.3).
    - Instruction-following: Ultrafeedback with evaluation on Alpaca-Eval v1.0 win rate vs text-davinci-003 (Section 4.4).
    - Natural language understanding: GLUE with RoBERTa models and careful validation/test split (Section 4.5; Appendix C.3).
  - Baselines: Prefix-tuning, series/parallel adapters, BitFit, RED, LoRA, DoRA (Sections 4.2–4.5).
  - Hyperparameter selection: Unlike many PEFT papers, tuning is done on held-out development sets, not test sets (Section 4.1; Appendix D.1). For commonsense/arithmetic, tuning was done on a GSM8K dev subset and reused across tasks to test robustness.

- Main results and takeaways
  - Commonsense reasoning (Table 1):
    - LoReFT achieves the best average scores across model sizes with tiny parameter budgets. Examples:
      > LLaMA-7B: LoReFT 80.2 avg at 0.031% params vs DoRA 78.1 and LoRA 74.7.  
      > LLaMA-13B: LoReFT 83.3 avg at 0.025% vs DoRA 81.5 and LoRA 80.5.  
      > Llama-2 7B: LoReFT 81.8 avg at 0.031% vs best DoRA 80.5.  
      > Llama-3 8B: LoReFT 86.6 avg at 0.026% vs best DoRA 85.2.
    - LoReFT often wins by large margins on HellaSwag, WinoGrande, and ARC-c (e.g., LLaMA-13B HellaSwag 95.1 and WinoGrande 87.2).
    - DiReFT is slightly worse but close, confirming the value of LoReFT’s orthogonality/difference design.
  - Arithmetic reasoning (Table 2):
    - Results are mixed. LoReFT underperforms LoRA/Adapters on average but beats prefix-tuning.  
      > LLaMA-7B avg: LoReFT 42.6 vs LoRA 46.9 (AQuA 21.4, GSM8K 26.0).  
      > LLaMA-13B avg: LoReFT 49.6 vs LoRA 51.1.
    - Authors attribute this in part to long chain-of-thought generations diluting the impact of fixed-position edits and intrinsic difficulty (Section 4.3).
  - Instruction-following (Table 3):
    - On Llama-2 7B and Ultrafeedback, LoReFT reaches top win rate with minute parameter count:
      > LoReFT: 85.60% win-rate at 0.0039% params vs LoRA 81.48%, RED 81.69%, and even full finetuning 80.93%.
      - “Half” rank LoReFT: 84.12% at 0.0019%.  
      - Low-resource (1K examples): 81.91%, still competitive with PEFT baselines.
  - GLUE (Table 4; Table 13 with SDs):
    - RoBERTa-base: LoReFT 84.2 avg at 0.015%, essentially tied with RED 84.3 and close to full finetuning 85.6.
    - RoBERTa-large: LoReFT 88.2 avg at 0.014%, slightly above RED 88.0 and close to FT 88.6.
    - Strong per-task results on SST-2, RTE, STS-B; DiReFT trails more on small models.
  - Additional checks and ablations:
    - Epoch-matched runs (Appendix D.3, Tables 14–15) show LoReFT’s commonsense gains persist even when trained for the same epochs/effective batch as DoRA/LoRA, addressing concerns that longer training solely drives the gains.
    - Parametrization ablations (Appendix E, Table 17) indicate that variants with similar parameter counts perform similarly, but LoReFT’s full form is consistently strong; fully removing either the orthogonality or difference components degrades performance modestly.
    - Inference overhead (Appendix H, Figure 11): overhead grows with rank, layers, and number of intervened positions but remains small for practical settings (e.g., rank-8, 10 layers, one position ≈ 0.05s for 256 tokens).

- Do the experiments support the claims?
  - Yes for commonsense, instruction tuning, GLUE: strong quantitative margins at extreme parameter efficiency (Figure 1 + Tables 1, 3, 4).  
  - Mixed for arithmetic reasoning: performance trails LoRA/Adapters (Table 2), and the paper explicitly frames this as a current limitation where long chain-of-thought reduces edit impact (Section 4.3).

- Qualitative and capability studies
  - Memorization (Appendix F):
    > With a single rank-1 LoReFT on the last token at a single layer, the model can reproduce up to 2,048 tokens of “Alice in Wonderland” with 100% prefix match (Figures 3–4). For scrambled text or random tokens, recall plunges (Figures 5–8), suggesting pretraining familiarity and linguistic structure matter. A single rank-1 edit can also store up to 256 ID→token mappings (Figures 9–10).
  - Composability (Appendix G.1):
    > Partition the rank-8 subspace into two orthogonal halves and train them on separate skills (German continuations vs instruction following). At inference, activating both partitions yields combined behavior—illustrated by qualitative examples—hinting at modular, additive control.

## 6. Limitations and Trade-offs
- Where it currently underperforms
  - Arithmetic and long chain-of-thought tasks: LoReFT lags LoRA/Adapters (Table 2). Fixed-position edits diminish in influence over very long generations (Section 4.3).
- Inference overhead
  - Unlike LoRA, LoReFT cannot be merged into weights; it requires runtime hooks. Overhead is small but nonzero and increases with rank, number of layers, and positions (Appendix H, Figure 11).
- Hyperparameter sensitivity and search space
  - Choosing positions (`p`, `s`), layers `L`, rank `r`, and whether to tie parameters materially affects performance (Section 4.1; Appendix D.2 gives tuning advice). While simpler than full finetuning, there is still a tuning burden.
- Scope of model families
  - Experiments largely use the LLaMA family and RoBERTa; vision-language or other architectures remain to be explored (Section 5).
- Assumptions
  - LoReFT presumes useful features are linearly encoded in subspaces and that editing a small number of positions/layers can steer behavior. This aligns with causal/linear findings but may not hold equally across all tasks or models.
- Stability on small datasets
  - GLUE results on small tasks (e.g., RTE) show variability typical of PEFTs; the paper uses controlled validation splits and multiple seeds (Section 4.5; Table 13).

## 7. Implications and Future Directions
- How this changes the landscape
  - ReFT reframes finetuning as subspace editing rather than weight updating. This bridges interpretability and control: if tasks can be induced by low-rank, position-specific edits, then finetuning can become more modular, explainable, and shareable (tiny artifacts per task).
- Practical applications
  - Rapid personalization and domain adaptation: store small “skill patches” and activate them per task or user.  
  - Multi-task composition: learn orthogonal subspace partitions for different abilities and combine them at inference (Appendix G.1).  
  - Deployment at scale: ship a frozen base model plus many kilobyte-sized ReFT adapters for different customers or tasks with minimal memory overhead.
- Research opportunities
  - Automated intervention search: learn positions/layers/ranks adaptively (Section 5 mentions interest in automating the hyperparameter search).
  - Richer intervention functions: beyond linear subspaces (e.g., nonlinear maps, conditional gating, token- or context-adaptive interventions).
  - Long-form reasoning: design schedule-aware or recurrent interventions that refresh the effect during generation to address chain-of-thought degradation.
  - Compositionality theory: when and how do orthogonal subspaces compose without interference? Early positive signs appear in Appendix G.1; formalizing this could enable “skill libraries.”
  - Cross-modal and non-Transformer models: apply ReFT to vision-language models (e.g., LLaVA), speech, or sequence models beyond Transformers (Section 5).
  - Safety and control: Appendix G.2 shows few-shot style/safety transfer (a 5-shot “GOODY-2”-like refusal style) with 32K parameters trained in ~30s; systematic, auditable control edits become feasible.

Overall, ReFT—and notably LoReFT—demonstrates that small, principled edits to hidden representations at a few layers/positions can deliver state-of-the-art parameter efficiency with strong performance on many tasks (Figure 1; Tables 1, 3, 4), opening a path to modular, interpretable, and highly deployable model adaptation.
