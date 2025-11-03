# Compact Language Models via Pruning and Knowledge Distillation

**ArXiv:** [2407.14679](https://arxiv.org/abs/2407.14679)

## 🎯 Pitch

This paper introduces a practical framework for deriving an entire family of compact, high-performance language models from a single large pretrained LLM by unifying structured pruning across multiple axes with highly data-efficient knowledge distillation. By applying this to the 15B Nemotron-4 model, the authors produce smaller MINITRON models (8B and 4B) using up to 40× fewer training tokens than training from scratch, with comparable or superior performance to similar community models. This work significantly lowers the cost and barrier to producing versatile LLM families, accelerating real-world deployment and democratizing access to powerful language models.

---

## 1. Executive Summary
This paper presents a practical recipe to turn one well-trained large language model (LLM) into a family of smaller, competitive models by combining structured pruning with data‑efficient knowledge distillation. Applied to the 15B-parameter `Nemotron‑4` teacher, the method produces the `MINITRON` 8B and 4B models using up to 40× fewer additional training tokens than training those sizes from scratch, while matching or beating similarly sized community models on standard benchmarks (Figure 1, Tables 2–4).

## 2. Context and Motivation
- Problem addressed
  - Building an LLM “family” (e.g., 15B, 8B, 4B) typically requires training each size from scratch—expensive in data and compute. The paper asks whether one can train a single large model once and derive smaller variants via structured pruning plus light retraining (Section 1).
- Why it matters
  - Real deployments need various model sizes for latency and cost. Reducing the marginal cost of each additional size makes model families practical for more organizations. The paper quantifies the savings: producing 8B and 4B models from a 15B teacher uses up to 40× fewer tokens per additional model and yields a 1.8× reduction in total compute to train a 15B/8B/4B family (Section “Cost Savings for Training a Model Family”).
- Prior approaches and limitations
  - Structured pruning has been widely studied, but:
    - Many LLM pruning works focus on a single axis (depth or width) and/or require gradients, large memory, and substantial fine‑tuning data (e.g., LLM-Pruner [33], SliceGPT [4], Sheared LLaMA [53]; see Related Work, Section 5).
    - Existing works rarely explore data‑efficient retraining via knowledge distillation (KD) from the unpruned teacher after structural pruning.
    - Users lack clear, empirically validated “best practices” that cover how to rank/prune across axes, combine pruning axes, and retrain effectively.
- Positioning relative to existing work
  - This paper unifies depth and width pruning with an activation-only importance metric computed via forward passes on a tiny calibration set (1024 samples), plus a KD‑based retraining regime that minimizes data use (Sections 2–3). It distills empirically grounded best practices and demonstrates state‑of‑the‑art results against modern pruning baselines (Table 4).

## 3. Technical Approach
The pipeline (Figure 2) has four stages: compute importance, rank and prune, lightweight architecture search, and retraining via distillation.

1) Importance estimation (Section 2.2)
- Goal: quantify the “importance” of each layer (depth), attention head, MLP neuron, and embedding channel (width) using only forward passes on a small calibration dataset `D` (1024 samples drawn from the pretraining mix).
- Width importance (for each Transformer block):
  - Heads: sum the squared L2 norm of each head’s output across batch and sequence:
    - “F_head^(i) = Σ_{B,S} ||Attn(X W_Q,i, X W_K,i, X W_V,i)||²” (Section 2.2).
  - MLP neurons: aggregate the pre‑activation magnitude of each neuron, computed from the i‑th row of the first MLP weight matrix `W1`:
    - “F_neuron^(i) = Σ_{B,S} X · (W1_i)^T” (Section 2.2; `W1_i` is the i‑th row).
  - Embedding channels: sum the LayerNorm output of channel i:
    - “F_emb^(i) = Σ_{B,S} LN(X)_i” (Section 2.2).
  - Aggregation choice matters. The best-performing reduction of these activations across batch and sequence is “batch=L2, sequence=mean”; see Table 13 (zero‑shot) and Figure 5 (post‑retraining).
- Depth importance:
  - Perplexity (“PPL”) drop: remove one layer at a time and measure the perplexity increase (Shortened LLaMA–style; Section 2.2).
  - Block Importance (“BI”): compute 1 − cosine similarity between a layer’s input and output in one forward pass (Equation for `BI_i` in Section 2.2). BI is faster and can extend to multiple contiguous layers.

2) Rank and prune (Section 2.3)
- Rank units in each axis by importance and trim the corresponding weights to the target sizes.
- Special handling for attention heads:
  - After pruning from L heads to K heads, the method “merges” information by adding residuals from pruned heads into kept heads, inspired by “Layer Collapse” for depth (Section 2.3). This preserves useful information and boosts accuracy.
  - For grouped-query attention (GQA), this merging is applied to query heads only (Section 2.3).
- After pruning, all affected matrices in MLP, MHA, LayerNorm, and embeddings are reshaped to match the new dimensions.

3) Lightweight architecture search (Section 2.3, Figure 3; Table 12)
- Enumerate feasible architectures within a narrow parameter range around the target (±5%), varying:
  - Number of layers (depth), attention heads, MLP expansion factor, and embedding size.
- Each candidate undergoes short retraining (about 1.8B tokens; “lightweight RT”) to stabilize rankings; then the best candidate is selected for full retraining (Sections 2.3 and 4.3).
- Figure 9 shows rankings stabilize after ≈300 steps (about 1.35B tokens), justifying lightweight retraining in the search loop.

4) Retraining via knowledge distillation (Section 3; Figure 4)
- Teacher–student setup: teacher is the original `Nemotron‑4 15B`, the student is the pruned model.
- Losses (with plain-language meaning first):
  - Logit distillation `L_logit`: make the student’s next‑token distribution match the teacher’s using soft targets and KL divergence at temperature τ:
    - “L_logit = (1/l) Σ_{k=1..l} Loss(p_t^k(x,τ), p_s^k(x,τ))” (Section 3).
    - `p(x,τ)` is the softmax at temperature τ (definition in Section 3).
  - Optional intermediate‑state losses `L_is`: align hidden states between teacher and student at selected layers; the student’s hidden states are linearly upscaled to teacher dimensionality (Figure 4 and Section 3).
    - `L_is` can include: embedding output loss `L_emb`, encoder block output loss `L_o`, input loss `L_i`, and attention relation loss `L_att` (Appendix A.4).
  - Cross‑entropy `L_CLM` with ground‑truth labels (standard language modeling loss).
- Total loss:
  - `L = L_CLM + L_logit + α·L_is`, with a dynamic weight `α = L_logit / L_is` to balance magnitudes (Section 3).
- Empirical choices:
  - KLD works best for `L_logit` compared to MSE, cosine, or reverse KLD (Tables 15–16).
  - Temperature τ = 1.0 performs best; top‑K logits truncation hurts unless K is very large, and gives no benefit over not truncating (Appendix A.3).
  - When depth is not heavily reduced, using `L_logit` alone works best; when depth is significantly reduced, add selected `L_is` terms (Best Practices #6–#7, Section 4.1; Tables 17–18).

5) One‑shot vs. iterative pruning
- For importance estimation: iterative re‑ranking/pruning brings no benefit after retraining; single‑shot is sufficient (Table 14).
- Across axes: pruning width alone outperforms depth pruning after some retraining, and a simple depth+width combination can be helpful depending on target (Table 10, Figure 6; also Table 1).
- For aggressive compression to 4B, a two‑step path (15B→8B→4B) with retraining at each step yields substantially better final accuracy than 15B→4B in one shot (Table 11, last two rows).

6) Where to prune in multi‑phase training
- If the teacher was trained with multi‑phase pretraining (web-heavy phase followed by cleaner data phase), prune the final‑phase checkpoint and retrain on a portion of that cleaner phase data (Table 20).

Implementation details:
- All pruning/distillation is implemented in NVIDIA Megatron‑LM (Section 4). Final model architectures are in Table 5.

## 4. Key Insights and Innovations
1) Activation-only, unified importance scoring across depth and width
- What’s new: a single, forward‑pass‑only method to score layers, heads, MLP neurons, and embedding channels with a tiny calibration set (1024 samples) (Section 2.2). This avoids gradient computation and large memory overhead common in prior pruning methods.
- Why it matters: enables practical, low‑overhead pruning at LLM scale. The study shows aggregation choice is crucial; “batch=L2, seq=mean” ranks best (Table 13, Figure 5).

2) Empirical best practices that change pruning decisions
- Ten best practices (Section 4.1) synthesize extensive ablations across axes, losses, pruning order, and retraining. Two that significantly affect design:
  - Width pruning beats depth pruning—but only after some retraining (Table 10; Figure 6; also Table 1).
  - Distillation with `L_logit` (KLD) alone usually beats using ground‑truth `L_CLM` or adding many intermediate-state losses unless depth is heavily reduced (Tables 15–18).

3) Residual merging for pruned attention heads
- A simple “head residual carryover” adds pruned heads’ information back into kept heads, analogous to layer collapse for depth (Section 2.3). This improves accuracy with negligible overhead, and is tailored for grouped‑query attention.

4) Lightweight architecture search with short retraining to stabilize rankings
- Rather than complex Bayesian/genetic search, the paper enumerates a small, practical search space and relies on ~1.8B‑token retraining to reveal the best candidate (Figure 9; Table 12). Rankings change materially in the first ~300 steps and then stabilize.

5) Cost‑effective model family creation
- Quantified saving: training the additional 8B and 4B models via prune+distill uses ~40× fewer tokens each; total family compute is 1.8× lower than training all sizes from scratch (Section “Cost Savings for Training a Model Family”).

## 5. Experimental Analysis
Evaluation setup (Section 4)
- Data and training
  - Teacher: `Nemotron‑4 15B` trained on an 8T token curated dataset; continued training data (“CT”) available (Section 4).
  - Lightweight retraining for search: ~1.8B tokens (400 steps). Final retraining budgets vary; e.g., Table 18 shows 18.9B vs 94B tokens for 8B ablations.
  - Importance estimation uses a calibration set `D` of 1024 samples (Section 4).
- Benchmarks and metrics (Section 4)
  - Knowledge/logic: `MMLU`, `ARC‑Challenge`, `HellaSwag`, `Winogrande`, `TruthfulQA`, `GSM8K`.
  - Coding: `HumanEval`, `MBPP` (pass@1; T=0.2, top‑p=0.95).
  - Summarization: `XL‑Sum (en)`.
  - Instruction-tuned evaluations: `MT‑Bench`, `IFEval`, `ChatRAG‑Bench`, `Berkeley Function Calling Leaderboard (BFCL)`.

Main results: quality vs. similarly sized models (Tables 2–3)
- `MINITRON 8B` (derived from 15B):
  - Beats previous generation `Nemotron‑3 8B` while using 40× fewer tokens, and is competitive with community baselines:
  - Selected numbers from Table 2:
    - MMLU (5‑shot): 63.8 vs `Nemotron‑3 8B` 54.7; comparable to `Mistral 7B` 64.1 and `Gemma 7B` 64, near `Llama‑3 8B` 65.3.
    - HellaSwag (10‑shot, acc_norm): 80.7 vs `Nemotron‑3 8B` 78.5; still below 84.6 of the 15B teacher.
    - GSM8K (5‑shot): 51.3 vs `Nemotron‑3 8B` 24.0; close to `Gemma 7B` 50.
    - HumanEval (pass@1): 31.6 vs `Nemotron‑3 8B` 20.7; close to `Gemma 7B` 32.
  - Quote:
    > Table 2: MINITRON 8B uses “40× fewer training tokens than Nemotron‑3 8B” and shows improved MMLU (+9.1 points) and coding performance (HumanEval +10.9 points).
- `MINITRON 4B`:
  - Outperforms `Gemma‑2B` and is competitive with `Phi‑2`/`Qwen2‑1.5B` on many tasks, despite using far fewer tokens than those models’ pretraining budgets (Table 3):
    - MMLU: 58.6 (vs `Phi‑2` 57.5, `Gemma‑2B` 42.0).
    - HellaSwag: 75.0 (vs `Phi‑2` 75.2).
    - HumanEval: 23.3 (vs `Phi‑2` 50.0; coding remains challenging for small base models without code‑heavy pretraining).
    - GSM8K: 24.1 (well below `Qwen2‑1.5B` 58.5; math reasoning remains a gap).

Against pruning baselines (Table 4)
- Quote:
  > Table 4 (8B range): `MINITRON` reaches MMLU 63.8 and HellaSwag 80.7, vs LLM-Pruner 25.2/67.8, SliceGPT 37.1/55.7, LaCo 45.9/64.4, ShortGPT 54.7/66.6—substantial gains despite fewer non‑embedding parameters.
  > Table 4 (4B range): `MINITRON 4B` achieves MMLU 58.6 and HellaSwag 75.0, far above ShortGPT’s 43.96/53.02 and Sheared LLaMA’s 26.4/70.8.

Instruction‑tuned evaluation (Tables 6–9)
- `MINITRON 4B‑instruct` (SFT using Nemotron‑4‑340B instruction data) shows strong downstream capability:
  - MT‑Bench: 6.46, outperforming `Gemma‑2B‑IT` (5.19) and `StableLM‑2 Chat 1.6B` (5.42) (Table 6).
  - IFEval: 68.76% strict prompt‑level accuracy; `Gemma‑2B‑IT` reports 28.70% (loose) (Table 7).
  - ChatRAG‑Bench: 41.11 average vs `Gemma‑2B‑IT` 33.31 (Table 8).
  - BFCL v2: 53.09 average, beating `Gemma‑2B‑IT` 41.63 and even `Llama‑3‑8B‑instruct` 50.51 (Table 9).

Ablations and supporting studies
- Width vs. depth pruning:
  - After ~200 retraining steps, width‑only beats depth‑only and depth+width for the same target size (Table 10 and Figure 6). Table 1 similarly shows width pruning becomes superior after distillation.
- Distillation vs. conventional training (iso‑compute) (Table 11):
  - Quote:
    > Distilling a pruned 4B student (100B tokens) reaches HellaSwag 52.04 and MMLU 42.45, vs random‑init 4B trained with the same compute (150B tokens) at 46.22/24.36 (or even 400B tokens at 48.23/26.24).
- Loss selection for KD:
  - `L_logit` with KLD outperforms MSE, cosine, reverse KLD (Tables 15–16). Using `L_logit` alone generally works best unless depth is reduced a lot (Tables 17–18).
- Aggregation metric for importance:
  - “batch=L2, seq=mean” ranks best both before and after retraining (Table 13, Figure 5).
- Single‑ vs multi‑phase retraining:
  - Pruning the phase‑2 (cleaner) checkpoint and retraining on phase‑2 data yields better results than mixing phase‑1+2 after pruning (Table 20).
- Architecture search:
  - Practical enumerations around target parameter budgets (Table 12) plus 1.8B‑token retraining yields stable rankings (Figure 9).
  
Do the experiments support the claims?
- Yes, for the scales tested (≤15B teacher) and the chosen tasks. The study is methodical: it contrasts pruning axes, validates importance aggregation choices, and probes distillation loss design. It shows large, consistent margins over pruning baselines (Table 4) and competitive performance versus from‑scratch peers for similar sizes (Tables 2–3), all with a clear compute‑saving story (Section “Cost Savings…”).

## 6. Limitations and Trade-offs
- Reliance on a strong teacher
  - The approach assumes access to a high‑quality, fully trained large model (here, `Nemotron‑4 15B` with 8T tokens). Compute savings apply to the “additional sizes,” not to producing the first large model.
- Data representativeness for importance estimation
  - Importance is computed from a tiny calibration set (1024 samples). If this set poorly represents deployment domains, importance rankings could mislead pruning (Section 2.2).
- Scale tested
  - Best practices (e.g., “width > depth after retraining”) are established up to 15B parameters. It is unclear whether they hold at much larger scales or with very different architectures (Section 4.1, Table 10).
- Task coverage and mixed results
  - While `MINITRON 8B` is broadly strong, `MINITRON 4B` shows weaker math/coding compared to specialized or heavily code‑trained models (Table 3: GSM8K and HumanEval).
- Distillation cost and teacher inference overhead
  - KD adds a full teacher forward pass, which is non‑trivial. Even “lightweight” retraining uses up to tens of billions of tokens for final models (e.g., 94B in Table 18), although still far less than from‑scratch pretraining.
- Structured changes can affect specialized capabilities
  - The paper does not evaluate long‑context behavior or retrieval-augmented settings post‑pruning. Pruning embeddings/heads could interact with such capabilities.

## 7. Implications and Future Directions
- Field impact
  - A practical path to economical model families: train one strong model once, then derive smaller variants with minimal extra data. The detailed best practices (Section 4.1) make pruning+KD actionable for teams beyond major labs.
- Applications
  - Deployable small/medium LLMs for edge or latency‑constrained environments.
  - Rapid “spin‑offs” of task‑ or domain‑specific models by pruning a generalist teacher and retraining on small domain data.
  - As shown with `MINITRON 4B‑instruct`, instruction‑tuned compact models can be strong for function calling and RAG (Tables 8–9).
- Research directions
  - Multi‑teacher or mixture‑of‑experts distillation for pruned students.
  - Automatic calibration set selection to improve importance estimates for target domains.
  - Extending best practices to larger teachers and to architectures with long‑context mechanisms.
  - Joint compression with quantization and low‑rank adaptation; the paper notes LoRA could be used during the lightweight search phase (Section 2.3).
  - Theoretical understanding of why width pruning overtakes depth pruning after short retraining (Table 10, Figure 6).

In sum, this paper contributes a carefully tested, end‑to‑end procedure to compress LLMs—spanning importance scoring, pruning across multiple axes, efficient retraining by KD, and practical architecture search—backed by extensive ablations and competitive downstream results.
