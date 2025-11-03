# ReFT: Representation Finetuning for Language Models

**ArXiv:** [2404.03592](https://arxiv.org/abs/2404.03592)
**Authors:** Zhengxuan Wu, Aryaman Arora, Zheng Wang, Atticus Geiger, Dan Jurafsky, Christopher D. Manning, Christopher Potts
**Institutions:** Stanford University

## 🎯 Pitch

ReFT introduces a novel approach to fine-tuning language models by editing hidden representations instead of weights, significantly reducing parameter count and computation costs. This method not only achieves state-of-the-art performance with 15×–65× fewer parameters but also bridges interpretability and control, paving the way for more modular and adaptive AI systems with practical applications in domain-specific tasks and rapid model specialization.

---

## 1. Executive Summary
ReFT (Representation Finetuning) is a new way to adapt language models by directly editing a small set of hidden representations at inference time instead of changing model weights. Its main instantiation, LoReFT, learns low‑rank interventions on selected tokens and layers, achieving state‑of‑the‑art or highly competitive results on several benchmarks while using 15×–65× fewer trainable parameters than strong weight‑based PEFT methods like LoRA (Figures 1; Tables 1, 3, 4).

## 2. Context and Motivation
- Problem addressed
  - Finetuning large language models (LMs) is expensive in memory and compute. Parameter‑efficient finetuning (PEFT) reduces cost by training a small subset of weights (e.g., LoRA, adapters), but still updates weights and often adds inference‑time overhead (Section 2).
  - Prior interpretability work shows that hidden representations encode task‑relevant concepts in near‑linear subspaces (Section 3.1). This suggests directly editing representations could be both powerful and efficient, yet there has been no general, trainable framework to do so across tasks.

- Why this matters
  - Practical: faster, cheaper adaptation of large models (single GPU runs in this paper) with tiny task‑specific footprints (often <0.04% of base parameters; Tables 1–4).
  - Scientific: bridges interpretability and control. If linear subspaces steer behavior, we can learn and re‑use them, offering a path toward interpretable, modular adaptation (Sections 3–5; Appendix G).

- Prior approaches and their gaps
  - Adapters add small networks between layers; they often impose inference overhead because they can’t be merged into base weights (Section 2).
  - LoRA/DoRA learn low‑rank weight updates that can be merged at inference, but still operate on weights, not representations (Section 2).
  - Representation‑editing techniques (e.g., activation addition, concept erasure) steer models but are typically hand‑crafted, not learned end‑to‑end for a task and not presented as a general finetuning framework (Appendix B).
  
- Positioning
  - ReFT is a general framework that learns interventions on hidden states during the forward pass (Definitions 3.1–3.2; Figure 2). LoReFT is a strong, low‑rank subspace instantiation of ReFT that is a drop‑in replacement for PEFT in practice (Section 3.2).

## 3. Technical Approach
At a high level, ReFT freezes the base LM and learns a small set of “edits” that are applied to selected hidden states during the forward pass. The key design choices are where (layers), when (token positions), and how (intervention function) to edit.

- Preliminaries
  - A Transformer LM computes a sequence of hidden vectors h(0) → … → h(m); predictions come from the final layer h(m) (Section 3; standard setup).
  - ReFT edits certain h(l) values at specific positions P and layer l with a parametric function Φ (Definition 3.1). Multiple non‑overlapping interventions form a ReFT method (Definition 3.2). The general intervention operation is Equation (7).

- Why edit representations?
  - Causal abstraction studies show that many concepts are encoded linearly in hidden spaces. A tool called distributed interchange intervention (DII) swaps the component of a representation within a learned low‑rank subspace to test causality (Equation (1); Section 3.1). If such subspaces govern behavior, learning to intervene in them may steer models reliably.

- LoReFT: Low‑rank Linear Subspace ReFT (core method)
  - Intuition: find a low‑dimensional subspace where small edits produce desired outputs, and learn a projection W and bias b that produce the edit, all while only touching a few tokens and layers.
  - Mechanism (Equation (2); Figure 2, right):
    - Choose a low‑rank subspace R ∈ R^{r×d} with orthonormal rows (r ≪ d).
    - For a hidden vector h, LoReFT computes an edit only inside that subspace by replacing Rh with a learned linear projection Wh + b, and then lifting back to the full space with R^T:
      - ΦLoReFT(h) = h + R^T(Wh + b − Rh).
    - Trainable parameters are ϕ = {R, W, b}. The base LM weights stay frozen.
  - How it relates to DII (Equation (1)): DII requires a counterfactual source representation s; LoReFT replaces this with a learned projected source Rs = Wh + b, turning DII into a trainable, task‑driven edit.

- DiReFT: a simpler, faster ablation (Equation (3))
  - Removes the orthogonality constraint and the subtraction term:
    - ΦDiReFT(h) = h + W2^T(W1 h + b), with low‑rank W1, W2 ∈ R^{r×d}.
  - This resembles LoRA applied to hidden states rather than weights (Section 3.2), trading some performance for efficiency and simpler training (Tables 1–4; Appendix E).

- Where and when to edit
  - Positions: intervene at a small number of prefix tokens (first p positions) and suffix tokens (last s positions). Since p + s is fixed, the overhead does not grow with prompt length (Section 4.1).
  - Layers: choose a set L of layers (often all, then shrink). Optionally tie parameters across positions within a layer to reduce parameters further (Section 4.1; Appendix D.2).

- Training objectives
  - Generation: standard language‑model cross‑entropy with teacher forcing (Equation (4)).
  - Classification: attach a small MLP head to the [CLS] representation and train it jointly with the interventions (Equations (5)–(6)).

- Implementation and usability
  - A general library (`pyreft`) wraps any HuggingFace model and inserts interventions (Appendix A), making ReFT a practical replacement for PEFT.
  - Inference overhead is small when editing only prompt tokens; Appendix H shows end‑to‑end generation adds ≈0.05s when intervening on 10 layers with rank 8 at the last prompt token (Figure 11).

- Why this design?
  - Low‑rank linear subspaces are supported by interpretability evidence for linear encodings (Section 3.1; citations therein).
  - Editing a few tokens concentrates compute where it matters (setup tokens often control behavior), avoiding per‑token overhead during long decodes (Section 4.1; Appendix H).
  - Orthogonality (LoReFT) stabilizes edits in a well‑conditioned subspace (Appendix E compares parametrizations).

## 4. Key Insights and Innovations
- A general finetuning framework over representations (ReFT)
  - Novelty: defines finetuning as a set of forward‑pass interventions I = ⟨Φ, P, l⟩, not weight updates (Definitions 3.1–3.2).
  - Significance: unifies and extends prior representation control methods (activation addition, RED, RepE) as special cases of ReFT (Appendix B), turning interpretability insights into a trainable finetuning paradigm.

- LoReFT: low‑rank, linear subspace editing that is task‑trainable
  - Novelty: replaces DII’s counterfactual source with a learned projection (Equation (2)), enabling end‑to‑end training on downstream objectives while editing only a few positions.
  - Significance: delivers strong performance at extremely low parameter counts (0.015%–0.031% in many settings; Tables 1–4), often beating or matching LoRA/DoRA.

- Efficiency without obvious performance sacrifice in many tasks
  - Innovation: decouples adaptation from weight updates; editing a few hidden states has near‑constant inference overhead and scales well with model size (Figure 1; Appendix H).
  - Significance: 15×–65× fewer trainable parameters than LoRA while being SOTA or competitive on commonsense, instruction tuning, and GLUE (Figure 1; Tables 1, 3, 4).

- Interpretability leverage and new capabilities
  - Insight: learned low‑rank subspaces can be composed like “puzzle pieces” to combine abilities (e.g., German completion + instruction following; Appendix G.1).
  - Evidence: a rank‑1 intervention can memorize long sequences or many key–value pairs, highlighting the expressive control achievable via small subspaces (Appendix F).

## 5. Experimental Analysis
- Evaluation setup
  - Models: `LLaMA-1 7B/13B`, `Llama-2 7B`, `Llama-3 8B`, `RoBERTa-base/large`. Training uses bfloat16 on a single A100 or RTX 6000 (Section 4).
  - Baselines: Prefix tuning, Series/Parallel Adapters, BitFit, LoRA, DoRA, and RED (Sections 4.2–4.5; Tables 1–4).
  - Hyperparameter selection avoids test‑set hill‑climbing; tuning is done on dev sets (Section 4.1; Appendix D.1). For commonsense/arithmetic, settings are transferred from a GSM8K dev set (with fewer epochs for the larger commonsense corpus).

- Benchmarks and metrics
  - Commonsense reasoning: eight datasets (e.g., BoolQ, PIQA, HellaSwag, WinoGrande; Section 4.2), accuracy on each test set (Table 1).
  - Arithmetic reasoning: AQuA, GSM8K, MAWPS, SVAMP; accuracy on final answer only (Section 4.3; Table 2).
  - Instruction tuning: UltraFeedback → evaluate with Alpaca‑Eval v1.0 win‑rate vs text-davinci-003 using GPT‑4 as judge (Section 4.4; Table 3).
  - GLUE classification: eight tasks; standard task metrics reported as “accuracy” aggregate (Section 4.5; Table 4).

- Headline results (parameter counts in parentheses are % of base LM)
  - Commonsense reasoning (Table 1)
    - LLaMA‑13B: 
      > LoReFT 83.3 avg (0.025%) vs DoRA 81.5 (0.681%) and LoRA 80.5 (0.670%).
    - LLaMA‑7B:
      > LoReFT 80.2 avg (0.031%) vs DoRA 78.1 (0.838%) and LoRA 74.7 (0.826%).
    - Llama‑2 7B:
      > LoReFT 81.8 avg (0.031%) vs DoRA 79.7 (0.838%).
    - Llama‑3 8B:
      > LoReFT 86.6 avg (0.026%) vs DoRA 85.2 (0.710%).
    - These reflect 20–30× parameter reductions relative to LoRA with higher accuracy.
    - Fairness check: With the same number of epochs as DoRA (3 epochs), LoReFT still outperforms on commonsense (Table 14; LLaMA‑13B 83.1 avg).

  - Instruction tuning (UltraFeedback → Alpaca‑Eval win‑rate; Table 3)
    - Llama‑2 7B:
      > LoReFT 85.60 win‑rate (0.0039%) vs LoRA 81.48 (0.1245%), RED 81.69 (0.0039%), and full finetuning 80.93 (100%).
    - Efficiency sensitivity: halving LoReFT’s rank (0.0019%) still yields 84.12; training on only 1K examples yields 81.91.
    - The win‑rate approaches GPT‑3.5 Turbo 1106 (86.30), showing strong long‑form generation capability.

  - GLUE (Table 4; average accuracy)
    - RoBERTa‑large:
      > LoReFT 88.2 avg (0.014%) vs LoRA 88.1 (0.225%) and RED 88.0 (0.014%).
    - RoBERTa‑base:
      > LoReFT 84.2 avg (0.015%) ≈ RED 84.3 (0.016%) and close to LoRA 84.7 (0.239%).
    - LoReFT is competitive at an order‑of‑magnitude lower parameter count than LoRA (≈16× fewer).

  - Arithmetic reasoning (Table 2; average accuracy)
    - LLaMA‑7B:
      > LoReFT 42.6 avg < LoRA 46.9; DiReFT 40.6.
    - LLaMA‑13B:
      > LoReFT 49.6 avg < LoRA 51.1; DiReFT 48.0.
    - These tasks require long chain‑of‑thought generations; ReFT’s edits, applied only at select tokens, influence a long trajectory less strongly (Section 4.3 discussion).

- Ablations and robustness
  - Parametrization variants of Φ (Appendix E; Table 17) show LoReFT’s design—orthonormal subspace and difference term—tends to perform best or on par among low‑parameter variants.
  - Equal‑epoch comparisons (Appendix D.3; Tables 14–15) suggest ReFT’s gains on commonsense are not an artifact of longer training.
  - Inference overhead: measured end‑to‑end and found small when intervening at prompt tokens only (Appendix H; Figure 11).

- Qualitative capabilities
  - Compositionality: partition LoReFT’s subspace (orthogonal slices) to combine skills learned on different datasets into a new behavior without joint training (Appendix G.1).
  - Memorization capacity: a single rank‑1 LoReFT can reproduce long texts (e.g., up to 2,048 tokens with 100% prefix match in many layers of LLaMA‑1 7B/13B) and memorize ≈256 key–value pairs (Appendix F; Figures 3–10). This illustrates strong control power of tiny interventions.

- Overall assessment
  - The experiments convincingly support that representation‑level finetuning can match or exceed weight‑based PEFT on several major tasks while using far fewer parameters, especially on commonsense and instruction following (Tables 1 and 3).
  - Performance is mixed on arithmetic chain‑of‑thought tasks (Table 2), clarifying the method’s current limits and motivating follow‑ups.

## 6. Limitations and Trade-offs
- Where ReFT underperforms
  - Chain‑of‑thought arithmetic: LoReFT/DiReFT trail LoRA/adapters (Table 2). Edits applied to a few prompt tokens may fade over long generation trajectories; tasks that need stable multi‑step internal computations may require broader or recurrent interventions (Section 4.3 discussion).

- Hyperparameter sensitivity and search space
  - Users must choose positions (prefix/suffix), layers, rank r, and whether to tie parameters (Section 4.1; Appendix D.2). While the paper offers heuristics (Appendix D.2), this adds tuning complexity compared to simpler “attach to Wq/Wk/Wv” recipes.

- Inference overhead
  - Unlike LoRA (mergeable into weights), ReFT must run interventions during the forward pass. Overhead is small when editing only prompt tokens (Appendix H), but non‑zero; editing generated tokens would increase it.

- Assumptions about linear structure
  - LoReFT relies on linear subspaces with orthonormal bases (Section 3.2). If task‑relevant mechanisms are highly non‑linear or distributed across many positions, a small linear subspace at a few positions may be insufficient.

- Scope of evaluation
  - Main results are on LLaMA‑family and RoBERTa models; vision‑language and other architectures remain to be studied (Section 5, “Limitations”).

## 7. Implications and Future Directions
- Field impact
  - Reframes finetuning: from weight editing to representation editing. This shift directly leverages interpretability’s “linear subspace” findings for practical, trainable model control, and it demonstrates that tiny, well‑placed edits can rival large weight updates (Sections 3–4; Figure 1).

- Practical applications
  - Lightweight domain or task adapters with tiny disk footprints (e.g., instruction tuning with ≈1 MB of parameters; Table 3 note). Potential for on‑device or multi‑task deployments where many small adapters are swapped at runtime.
  - Safety/style steering: Appendix G.2 shows few‑shot “ethos” tuning (GOODY‑2 style refusals) with only five examples and 32,772 trainable parameters, trained in ≈30 seconds, suggesting rapid specialization is feasible.

- Research directions
  - Automated search for where to intervene: learn positions/layers alongside subspaces; dynamic policies that intervene during generation when needed.
  - Better long‑horizon control: extend ReFT to recurrent or multi‑step interventions that sustain influence across chain‑of‑thought reasoning.
  - Compositional subspaces: formalize how orthogonal subspaces combine, including conflict resolution and task routing (Appendix G.1 hints that compositions can work).
  - Cross‑modal and broader model families: apply to vision‑language models (e.g., LLaVA; Section 5) and non‑Transformer architectures.
  - Interpretability feedback loop: use learned subspaces to map behaviors to mechanisms; conversely, use mechanistic insights to design more structured Φ functions (Section 5, “ReFT and model interpretability”).

In sum, LoReFT operationalizes a long‑standing interpretability hypothesis—that linear subspaces encode behavior—into a practical, trainable finetuning method. It offers a compelling efficiency–performance trade‑off and opens a path toward modular, composable, and interpretable adaptation of large models.
