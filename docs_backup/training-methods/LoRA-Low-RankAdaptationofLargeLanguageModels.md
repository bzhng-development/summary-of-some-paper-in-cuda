# LoRA: Low-Rank Adaptation of Large Language Models

**ArXiv:** [2106.09685](https://arxiv.org/abs/2106.09685)

## 🎯 Pitch

LoRA introduces a simple yet transformative method to adapt massive language models by freezing their original weights and learning only a tiny, low-rank update in each layer. This approach slashes the number of trainable parameters—and training memory use—by orders of magnitude, while matching or exceeding the quality of full fine-tuning and adding zero inference overhead. LoRA makes it feasible and efficient to deploy multiple specialized models on colossal architectures like GPT-3, unlocking broader access and agility for real-world NLP applications.

---

## 1. Executive Summary
LoRA (Low-Rank Adaptation) is a method for adapting very large language models by freezing the original weights and learning only a small, low‑rank update inside each layer. This reduces the number of trainable parameters by up to 10,000× and training memory by up to 3× while matching or improving the quality of full fine‑tuning and adding no extra inference latency (Abstract; Section 4.2; Table 4).

## 2. Context and Motivation
- Problem addressed
  - Full fine‑tuning rewrites all parameters of a pre‑trained model. With modern models (e.g., GPT‑3 with 175B parameters), storing, training, and deploying a separate fine‑tuned copy per task becomes prohibitively expensive in memory, compute, and operational complexity (Introduction; footnote 1; Section 2).
- Why this matters
  - Real-world deployment requires many task‑specialized models. Keeping a full copy per task multiplies memory/storage cost (hundreds of GBs per model) and slows iteration. Reducing trainable parameters without sacrificing quality lowers the hardware barrier to entry and enables multi‑tenant, task‑switchable services (Abstract; Section 4.2).
- Prior approaches and their gaps (Section 3)
  - Adapter layers: small trainable bottleneck modules inserted between Transformer blocks. They save parameters but add sequential computation at inference, increasing latency—especially at online batch size 1. On GPT‑2 medium, adapters increase latency by 20–30% for short sequences (SeqLen 128) even with small parameter counts (Table 1; Appendix B).
  - Prompt/prefix tuning: optimize special “prompt” tokens instead of model weights. It reduces effective sequence length for the actual task and is harder to optimize; performance often degrades when using many special tokens (Figure 2; Section 3).
- Positioning
  - LoRA targets a different axis: keep the pre‑trained weights frozen and express the task‑specific weight change as a low‑rank matrix inside existing layers. This preserves the original inference graph (no added depth), keeps full sequence length, and can be merged into base weights for zero inference overhead (Sections 4.1–4.2).

## 3. Technical Approach
LoRA reframes “how to adapt” as “how to parameterize the change.”

- Core reparameterization (Section 4.1; Figure 1)
  - For any dense weight `W0 ∈ ℝ^{d×k}`, LoRA models its task‑specific update as a low‑rank product:
    - `ΔW = B A`, where `B ∈ ℝ^{d×r}` and `A ∈ ℝ^{r×k}` with small rank `r << min(d,k)`.
    - The forward pass becomes:
      - `h = W0 x + (α/r) · (B A) x`  (Equation 3; the `(α/r)` scaling stabilizes training across different `r`).
  - Initialization: `A` is Gaussian‑initialized, `B` is zero‑initialized, so the model starts exactly at the pre‑trained function (`ΔW=0`), avoiding cold‑start regressions (Section 4.1).
  - Only `A` and `B` are trainable; `W0` is frozen.

- Why low rank?
  - Prior evidence suggests the effective dimensionality of fine‑tuning is small (intrinsic dimension). LoRA hypothesizes the “intrinsic rank” of the necessary update is also low, so a product `B A` can capture the essential directions of change (Section 1; Section 4.1).

- Where LoRA is inserted in Transformers (Section 4.2)
  - Focus on multi‑head attention projections: `Wq`, `Wk`, `Wv`, `Wo` (commonly treated as `d_model × d_model`).
  - In most experiments, LoRA is applied only to `Wq` and `Wv` for simplicity and parameter efficiency; MLP layers and LayerNorms are frozen (Section 4.2; Section 5.1 baseline definitions).

- How inference keeps zero latency (Section 4.1)
  - At deployment, explicitly merge: `W = W0 + B A`. The layer then performs a single matrix multiply as usual. To switch tasks, subtract the current `B A` and add another task’s `B'A'`—a quick, memory‑light operation.

- Why this design over alternatives (Sections 3 and 4.1–4.2)
  - Unlike adapters, LoRA does not add new sequential layers, so it adds no extra inference ops after merging (Table 1 shows adapters can slow online inference by 20–30%).
  - Unlike prefix tuning, LoRA does not consume sequence length and is easier to optimize (Figure 2 shows prefix performance can drop with more prompt tokens).

- Practical impact and numbers (Section 4.2)
  - Training memory: On GPT‑3 175B, LoRA reduces VRAM from ~1.2 TB to ~350 GB (while keeping the same model parallelism) and increases per‑GPU throughput from 32.5 to 43.1 tokens/s (footnote 5).
  - Storage: With `r=4` and adapting only `Wq`/`Wv` across 96 layers, a task checkpoint is ~35 MB vs. ~350 GB for a full fine‑tuned model (footnote 4).
  - Parameter count: For LoRA on attention with rank `r`, the trainable parameters scale as `|Θ| = 2 × L̂ × d_model × r` where `L̂` is the number of adapted matrices (Section 5.1).

- Formal objective, in plain terms (Section 2; Equations 1–2)
  - Full fine‑tuning maximizes the conditional likelihood of targets given inputs by updating all parameters `Φ` (Eq. 1).
  - LoRA constrains the adaptation to be generated from a much smaller parameter set `Θ` that defines `ΔΦ(Θ)` (Eq. 2). The model you optimize is `Φ0 + ΔΦ(Θ)`, with `|Θ| << |Φ0|`.

## 4. Key Insights and Innovations
- Low‑rank updates suffice—even at extreme scale
  - Novelty: Instead of learning full matrices, LoRA learns a low‑rank update (`B A`) per adapted weight.
  - Significance: On GPT‑3 175B, ranks as low as `r=1–2` often match or surpass full fine‑tuning quality with millions (not billions) of trainable parameters (Table 6; Table 4).
  - Why it works: Subspace analysis shows the top singular direction(s) of the learned update are stable and shared across different `r` (Figure 3), suggesting the “useful” update space is extremely low dimensional.

- Zero‑latency parameter‑efficient adaptation
  - Novelty: The update merges into the original weight (`W0 + B A`), leaving the runtime graph unchanged (Section 4.1).
  - Significance: Unlike adapters, this avoids inference slowdowns; Table 1 shows adapters increase single‑example latency by 20–30% on GPT‑2 medium (SeqLen 128), whereas LoRA adds none after merging.

- Which matrices matter for adaptation
  - Insight: Under a fixed parameter budget, adapting both `Wq` and `Wv` outperforms adapting `Wq` or `Wk` alone (Table 5). This guides practical placement of LoRA for best quality per parameter.

- What adaptation “does” to the pre‑trained model
  - Insight: The learned update `ΔW` aligns with—but does not duplicate—the dominant directions of the original `W`. It amplifies “non‑dominant” directions that matter for the task (Section 7.3; Table 7).
  - Evidence: Projecting `Wq` onto the subspace spanned by `ΔWq` shows strong correlation vs. random, but little overlap with `Wq`’s own top singular directions; the amplification factor can exceed 20× for `r=4` (Table 7; Section H.4).

## 5. Experimental Analysis
- Evaluation setup
  - Models and tasks:
    - NLU: RoBERTa base/large (125M/355M) and DeBERTa XXL (1.5B) on GLUE (eight tasks: MNLI, SST‑2, MRPC, CoLA, QNLI, QQP, RTE, STS‑B) (Section 5.2–5.3; Table 2).
    - NLG: GPT‑2 medium/large on E2E NLG, DART, WebNLG (Section 5.4; Tables 3, 13, 14).
    - Large‑scale: GPT‑3 175B on WikiSQL, MNLI‑matched (MNLI‑m), and SAMSum (Section 5.5; Table 4).
  - Baselines (Section 5.1):
    - Full fine‑tuning; Adapter variants (Houlsby “AdapterH,” Pfeiffer/LayerNorm placement “AdapterL/P,” and AdapterDrop); Prefix‑embedding and Prefix‑layer tuning; Bias‑only (BitFit).

- Main quantitative results
  - GPT‑3 175B (Table 4):
    > LoRA (4.7M params) achieves “MNLI‑m 91.7%, WikiSQL 73.4%, SAMSum 53.8/29.8/45.9 (R1/R2/RL)” while full fine‑tuning (175B params) achieves “89.5%, 73.8%, 52.0/28.0/44.5.”
    - With more LoRA parameters (37.7M), WikiSQL improves to 74.0% and MNLI‑m stays ~91.6%.
    - Prefix methods often do worse and show non‑monotonic behavior as parameters increase (Figure 2; Table 15).
  - GLUE with RoBERTa and DeBERTa (Table 2):
    > RoBERTa‑base: LoRA average 87.2 vs. full fine‑tuning 86.4; RoBERTa‑large: LoRA 89.0 vs. full fine‑tuning 88.9; DeBERTa‑XXL: LoRA 91.3 vs. full fine‑tuning 91.1.
  - GPT‑2 on E2E NLG (Table 3):
    > GPT‑2‑medium: LoRA (0.35M params) “BLEU 70.4, NIST 8.85, METEOR 46.8, ROUGE‑L 71.8, CIDEr 2.53,” outperforming full fine‑tuning and adapters with similar parameter counts.
  - Latency study (Table 1; Appendix B):
    > Adapter layers increase inference latency by up to “30%” at batch size 1 and SeqLen 128; LoRA introduces no such overhead after merging.

- Ablations and diagnostics
  - Placement ablation (Table 5): Adapting `Wq+Wv` beats `Wq` or `Wk` alone under the same parameter budget.
  - Rank ablation (Table 6): For GPT‑3 175B, ranks as low as `r=1–2` can match near‑best performance for `Wq+Wv`; training only `Wq` needs higher `r`.
  - Subspace analyses (Figures 3–4; Section 7.2): The top singular directions of updates are shared across `r` and across seeds, indicating a very low “intrinsic rank.”
  - Low‑data regime (Table 16): On MNLI‑m with only 100 examples, LoRA reaches 63.8% vs. full fine‑tuning 60.2%, while prefix methods perform poorly (37.6–48.3%).

- Do the experiments support the claims?
  - Yes, across architectures (RoBERTa, DeBERTa, GPT‑2, GPT‑3), LoRA delivers comparable or better accuracy with orders‑of‑magnitude fewer trainable parameters and no added inference latency. The ablations (Tables 5–6) and analysis (Section 7; Table 7) explain why small ranks work and which weights to adapt.
  - Caveats: Some datasets show small deltas (e.g., GLUE averages are within ~1 point), and a few results are near ties. However, at GPT‑3 scale, LoRA clearly outperforms full fine‑tuning on MNLI‑m and SAMSum (Table 4).

## 6. Limitations and Trade-offs
- Assumptions
  - The adaptation lies in a low‑rank subspace (“intrinsic rank” is small). If a task requires large structural changes (e.g., a very different language), a small `r` may be insufficient (Section 7.2, footnote 6).
- Scope choices
  - Most experiments adapt only attention projections (`Wq`, `Wv`); MLP and LayerNorm layers remain frozen (Section 4.2). It remains open whether adapting more components consistently helps.
- Task batching and multi‑tenant serving
  - If you merge `B A` into `W0` for zero latency, a single forward pass can only use one task’s LoRA weights. Serving a heterogeneous batch with different LoRA modules requires either unmerged weights (slight overhead) or batching per task (Section 4.2).
- Memory during deployment
  - You must still store the large base model (e.g., 350 GB for GPT‑3 175B). LoRA reduces per‑task storage (e.g., ~35 MB), not the base (Section 4.2; footnote 4).
- Hyperparameters and stability
  - While LoRA reduces tuning burden via the `(α/r)` scaling (Section 4.1), `r`, learning rate, and placement still matter. Prefix‑layer combinations can be sensitive (Appendix E).
- Not all competing methods scale smoothly
  - Prefix‑based methods degrade beyond certain parameter counts (Figure 2; Table 15). Although this highlights LoRA’s advantage, it also signals optimization sensitivity in alternative paradigms.

## 7. Implications and Future Directions
- Field impact
  - LoRA makes large‑model adaptation practical: millions of trainable parameters instead of billions, scalable to GPT‑3 175B with better or equal quality and zero added inference latency (Table 4). This shifts the default from “fine‑tune a copy” to “attach a tiny low‑rank head.”
- Practical applications
  - Multi‑tenant services: host a single base model in GPU memory and hot‑swap many tiny LoRA modules for different tasks or customers (Section 4.2).
  - Edge or latency‑sensitive settings: no added inference depth; avoids the adapter latency penalty (Table 1).
  - Rapid domain customization: cheap to train and store specialized variants (footnote 4: 100 tasks add ~3.5 GB, not ~35 TB).
- Research directions (Section 8; Appendix E; Section 7)
  - Combine LoRA with other parameter‑efficient methods (e.g., prefix tuning) for orthogonal gains; early results show LoRA+PrefixEmbed further boosts WikiSQL (Table 15).
  - Decide placement and rank automatically: develop principles or algorithms to select which matrices to adapt and at what rank (Section 7.1–7.2 suggest `Wq+Wv` and very small `r` often suffice).
  - Theory of adaptation: LoRA’s subspace analyses connect fine‑tuning to amplifying specific low‑rank feature directions; formalizing this could explain when and why large models adapt efficiently (Section 7.2–7.3; Table 7).
  - Beyond attention: Evaluate adapting MLP/LayerNorms or combining LoRA with tensor‑product parameterizations for even greater parameter efficiency (Section 6 discussion).

> Key takeaways anchored in the paper’s evidence:
> - “Reduce trainable parameters by 10,000× and GPU memory by 3×” at GPT‑3 scale (Abstract; Section 4.2).
> - “Zero additional inference latency” because `W0 + B A` can be merged at deployment (Section 4.1).
> - “Ranks as low as 1–2 can suffice” for strong results when adapting `Wq`/`Wv` (Table 6), and adapting `Wq+Wv` is the best use of a fixed budget (Table 5).
> - “Outperforms full fine‑tuning on GPT‑3 MNLI‑m and SAMSum with 4.7M parameters” (Table 4), while adapters introduce 20–30% latency in online settings (Table 1).
