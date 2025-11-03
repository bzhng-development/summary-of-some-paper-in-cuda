# Compact Language Models via Pruning and Knowledge Distillation

**ArXiv:** [2407.14679](https://arxiv.org/abs/2407.14679)
**Authors:** Saurav Muralidharan, Sharath Turuvekere Sreenivas, Raviraj Joshi, Marcin Chochowski, Mostofa Patwary, Mohammad Shoeybi, Bryan Catanzaro, Jan Kautz, Pavlo Molchanov
**Institutions:** NVIDIA (implied, by NVlabs and associated authors)

## 🎯 Pitch

This paper introduces a pioneering method to compress large language models into smaller versions using structured pruning and knowledge distillation, bypassing the need for the original pretraining data. With innovations like 'teacher correction' and a downstream-task-driven pruning approach, it enables the creation of highly efficient models with significant speed improvements and reduced training costs, broadening accessibility for organizations without proprietary data access.

---

## 1. Executive Summary
This paper demonstrates a practical recipe to shrink large language models (LLMs) into small language models (SLMs) using structured pruning plus knowledge distillation without needing the original pretraining data. The key additions are a lightweight “teacher correction” stage that adapts the teacher to the available distillation data and a downstream-task-driven criterion for depth pruning; together they yield state‑of‑the‑art 8B and strong 4B models with far fewer training tokens and faster inference (Figures 1–2, Tables 1–2).

## 2. Context and Motivation
- Problem addressed
  - Training a full family of LLM sizes from scratch (e.g., 8B/70B/405B) is extremely expensive in data and compute. Prior “Minitron” work showed pruning + distillation can cheaply derive smaller models from a single large model—but assumed access to the original pretraining dataset (Introduction, first two paragraphs; [2]).
  - With many modern models trained on private data, the distiller often cannot access the original corpus. Using a different dataset for distillation causes suboptimal guidance from the teacher (Teacher Correction section; Figure 2).

- Importance
  - Practical: Enables organizations to produce high-quality smaller models tailored for deployment constraints (e.g., memory- or latency-limited) without retraining from scratch or accessing proprietary corpora.
  - Scientific: Provides evidence that carefully designed pruning + distillation can reach or exceed teacher performance on some tasks while using 10–40× fewer tokens (Tables 1–2; Insights, General point 2).

- Prior approaches and gaps
  - Earlier structured pruning for LLMs (e.g., Sheared LLaMA, SliceGPT, ShortGPT) focused on pruning strategies, often relying on perplexity metrics and/or access to original data ([6–9]).
  - The original Minitron recipe showed strong results with pruning + distillation but assumed original pretraining data and used perplexity (validation loss) for layer importance; it also used a lightweight NAS step to select architectures ([2]).
  - Gaps: No solution for distillation without original pretraining data; depth-pruning saliency based purely on validation loss can underperform on downstream tasks (Methodology → Pruning; Figures 8–9).

- This paper’s positioning
  - Adapts the Minitron pipeline for the “no pretraining data access” setting via `teacher correction`.
  - Replaces perplexity-only depth-pruning saliency with a downstream-task-based criterion (Winogrande) and prefers dropping contiguous layer blocks (Methodology → Pruning; Figures 8–9).
  - Skips NAS, using architecture insights from [2], to keep the method simple and practical (Pruning; Table 3).

## 3. Technical Approach
The pipeline has three main stages (Figure 1): (1) teacher correction, (2) pruning, (3) distillation-based retraining. A subsequent alignment stage is used for instruction tuning.

A. Teacher correction (Figure 2; Teacher Correction section)
- What it is
  - “Teacher correction” is a short, low‑learning‑rate fine‑tuning of the original teacher model on the dataset available for distillation (∼100B tokens; Figure 1 shows 127B). It adapts the teacher to the token distribution and content of the new dataset.
- Why it’s needed
  - When the distillation dataset differs from the teacher’s pretraining data, the teacher’s probability distribution over subword tokens can be a poor fit, reducing the quality of distillation signals (Teacher Correction section).
- How it’s done
  - Fine‑tune the teacher for ~100B tokens with small learning rates, short warm‑up (120 steps), and the same batch size/decay schedule as the original model (Teacher Correction section).
- Variants explored
  - Two ways to use correction (Figure 5):
    1) Fully correct the teacher, then prune+distill.
    2) Start from the original teacher; correct it “in parallel” while distilling the student.
  - Result: Both are on par; correction can happen concurrently with distillation (Figure 5).

B. Pruning (Figures 3, 8, 9; Pruning section)
- Structured pruning: remove whole structures (not individual weights), such as entire layers (`depth` pruning), whole neurons or MLP channels, attention heads, or embedding channels (`width` pruning). This preserves regular tensor shapes and is hardware‑friendly.
- Importance estimation for width axes
  - Use a small calibration set (1024 samples) and forward passes only to collect activations.
  - Rank the importance of neurons (MLP), attention heads (MHA), and embedding channels (LayerNorm/embeddings) using activation magnitudes; aggregate across batch with L2‑norm and across sequence with the mean. Then prune least important ones in a single shot (no iterative pruning) (Pruning; “Importance Estimation”).
- Depth pruning (whole layers)
  - Evaluate the sensitivity of removing contiguous layer blocks by two metrics:
    1) Language modeling validation loss/perplexity (LM val loss).
    2) Downstream task accuracy; they found Winogrande (a commonsense pronoun resolution benchmark) to be a strong indicator (Pruning; Figures 8–9).
  - Key empirical findings guiding design choices (Figures 8–9):
    - Early and late layers are most important by LM val loss (Figure 8).
    - Removing non‑contiguous layers may look good on LM loss (dashed line in Figure 8) but can hurt downstream accuracy.
    - Dropping a contiguous block of layers selected via Winogrande yields better downstream performance than picking non‑contiguous layers by per‑layer importance (Figure 9).
- Chosen pruning strategies for the two families (Pruning; Table 3)
  - Llama‑3.1‑Minitron‑4B‑Width: keep all 32 layers and all attention heads, shrink hidden size (4096→3072) and MLP dim (14336→9216).
  - Llama‑3.1‑Minitron‑4B‑Depth: keep hidden sizes, but halve depth (32→16).
  - MN‑Minitron‑8B (from Mistral NeMo‑12B): keep depth, shrink hidden (5120→4096) and MLP dim (14336→11520).

C. Distillation (Figure 2; Retraining with Distillation; Distillation section)
- Distillation goal
  - Train the pruned student to match the teacher’s output probability distribution at each next‑token step.
- Loss used
  - Forward KL divergence between the teacher’s and student’s logits-probabilities (no cross‑entropy to ground truth labels):
    - Intuition: minimize information lost when approximating the teacher with the student; encourages the student to cover the teacher’s distribution mass (Figure 2; Retraining with Distillation).
- Training setup
  - Llama‑3.1‑Minitron: 94B tokens; peak LR 1e‑4; min LR 1e‑5; warm‑up 40 steps; global batch 1152; context 8192 (Table 4).
  - MN‑Minitron‑8B: 380B tokens; peak LR 1e‑4; min LR 4.5e‑7; warm‑up 60 steps; global batch 768; context 8192 (Table 4).
  - Hardware: 32 NVIDIA DGX H100 nodes (Distillation section).

D. Instruction tuning and alignment (Instruction Tuning section)
- Pipeline (for all models):
  1) Math and code SFT.
  2) Instruction SFT.
  3) Two rounds of Reward‑aware Preference Optimization (RPO) to align with preference data (Instruction Tuning section).
- Tooling: NeMo‑Aligner (open‑source alignment toolkit) [17].

E. Datasets
- Pretrained teachers: Mistral‑NeMo‑12B and Llama‑3.1‑8B (trained by their creators on proprietary corpora).
- For pruning, teacher correction, and distillation: Nemotron‑4 “continued training (CT)” dataset (Training Details → Dataset; [13,14]).
  - This is the single dataset used throughout the pipeline when the original pretraining data is unavailable.

## 4. Key Insights and Innovations
1) Teacher correction for data-misaligned distillation
- What’s new
  - A short fine‑tune of the teacher on the available distillation dataset before, or even during, student training (Figure 2; Teacher Correction section).
- Why it matters
  - It fixes a common real‑world obstacle: lack of access to the original pretraining data.
  - Empirically, it reduces LM validation loss of the student by >6% and accelerates convergence (Figure 4; Insights, General point 1). Figure 5 shows on‑the‑fly correction performs on par with full pre‑correction.

2) Downstream‑task‑driven depth pruning with contiguous layer drops
- What’s new
  - Instead of ranking layers only by LM loss, they select which contiguous block of layers to remove based on downstream accuracy (Winogrande), which better predicts end‑task performance (Pruning; Figures 8–9).
- Why it matters
  - Even when non‑contiguous layer removal improves LM loss, it can harm downstream accuracy; picking a contiguous block by Winogrande yielded better results (Figure 9).

3) Single‑shot width pruning while retaining attention heads
- What’s refined
  - For width pruning, they prune MLP and embedding channels but keep attention heads intact, guided by activation statistics (Pruning; Table 3; Insights, General point 3).
- Why it matters
  - For Llama‑3.1→4B, the width‑pruned model consistently outperforms the depth‑pruned one at the same parameter budget on base tasks (Figure 7, Table 1), and also on most instruction‑tuned benchmarks (Table 2).

4) Strong results with far fewer tokens and practical speedups
- What’s impactful
  - The MN‑Minitron‑8B student learned from 380B tokens vs. the Llama‑3.1‑8B teacher’s 15T pretraining tokens, yet competes with or beats similarly sized models (Table 1).
  - The Llama‑3.1‑Minitron‑4B Depth model achieves up to 2.7× throughput over Llama‑3.1‑8B; the Width model reaches 1.8×, both in FP8 with TensorRT‑LLM (Figure 10).

These are mostly practical innovations (data‑agnostic distillation, saliency criterion choice, pruning design) that cumulatively deliver large real‑world gains.

## 5. Experimental Analysis
A. Evaluation setup
- Base model benchmarks (Table 1)
  - Downstream language tasks: MMLU (5‑shot), Winogrande (5‑shot), ARC‑Challenge (25‑shot), HellaSwag (10‑shot), TruthfulQA (0‑shot), XL‑Sum English (20% subset; 3‑shot).
  - Coding/program synthesis: HumanEval (0‑shot, n=20) and MBPP (0‑shot). pass@1 with T=0.2, top‑p=0.95 for code tasks.
- Instruction‑tuned benchmarks (Table 2)
  - MT‑Bench (GPT‑4‑Turbo judge), MMLU (0‑shot), GSM8K (0‑shot), GPQA (0‑shot), HumanEval, MBPP, IFEval, and BFCLv2 (function calling; live score).
- Models compared
  - Teachers: Llama‑3.1‑8B and Mistral‑NeMo‑12B.
  - Students: MN‑Minitron‑8B (from Mistral‑NeMo‑12B), Llama‑3.1‑Minitron‑4B‑Width and ‑Depth (from Llama‑3.1‑8B).
  - External baselines include Gemma2‑7B/2B, Qwen2‑1.5B, Phi‑2‑2.7B, etc. (Tables 1–2).

B. Main quantitative results
- Base models (Table 1)
  - MN‑Minitron‑8B vs similarly sized 8B models:
    - MMLU: 69.5 vs Llama‑3.1‑8B’s 65.3 (+4.2).
    - GSM8K: 58.5 vs 48.6 (+9.9).
    - HellaSwag: 83.0 vs 81.8 (+1.2).
    - MBPP: 43.8 vs 42.3 (+1.5).
    - Winogrande: 80.4 vs 77.3 (+3.1).
    - TruthfulQA: 47.6 vs 45.0 (+2.6).
    - XL‑Sum: 32.0 vs 30.0 (+2.0).
  - MN‑Minitron‑8B vs its 12B teacher:
    - It actually surpasses the 12B base on GSM8K (58.5 vs 55.7) and HumanEval (36.2 vs 23.8) while staying close elsewhere (Table 1; Insights, Mistral→MN‑Minitron‑8B point 1).
  - Llama‑3.1‑Minitron‑4B‑Width vs ‑Depth (same param budget):
    - MMLU: 60.5 (Width) vs 58.7 (Depth).
    - GSM8K: 41.2 (Width) vs 16.8 (Depth).
    - Winogrande: 76.1 (Width) vs 73.2 (Depth).
    - Overall, Width is stronger for base capabilities (Table 1; Figure 7).
- Instruction‑tuned models (Table 2)
  - MN‑Minitron‑8B (aligned) vs Llama‑3.1‑8B (aligned):
    - MMLU: 70.4 vs 69.4.
    - GSM8K: 87.1 vs 83.8.
    - IFEval: 84.4 vs 80.4.
    - BFCLv2 (Live): 67.6 vs 44.3.
    - MT‑Bench: 7.86 vs 7.78.
    - HumanEval/MBPP are roughly comparable (71.3/72.5 vs 72.6/72.8).
  - 4B models vs small open baselines:
    - Llama‑3.1‑Minitron‑4B‑Width achieves GSM8K 79.76, MBPP 65.1, IFEval 79.54—stronger than Phi‑2, Gemma2‑2B, and Qwen2‑1.5B on most metrics (Table 2).

C. Ablations and analyses
- Teacher correction improves student convergence
  - With correction, MN‑Minitron‑8B’s LM validation loss is consistently lower over training; the drop exceeds 6% (Figure 4; Insights, General point 1).
  - Doing correction during distillation matches the performance of pre‑corrected teachers (Figure 5).
- Pruning + distillation vs alternatives (Figure 6)
  - Four settings compared for MN‑Minitron‑8B:
    1) Random initialization + distillation,
    2) Random pruning + distillation,
    3) Pruning + standard LM loss training,
    4) Pruning + distillation (proposed).
  - Proposed (4) achieves the best and fastest convergence, demonstrating orthogonal benefits of both informed pruning and distillation.
- Width vs depth pruning (Figure 7)
  - For Llama‑3.1→4B, width pruning yields lower initial loss and better training curves than depth pruning at equal parameter budgets.
- Depth pruning criterion (Figures 8–9)
  - LM‑loss‑based layer removal suggests non‑contiguous drops can look good for LM loss (Figure 8, dashed line).
  - However, downstream accuracy (Winogrande) is higher when dropping one contiguous block (e.g., layers 16–31) than when dropping “best” 16 non‑contiguous layers by LM loss (Figure 9). This finding drives the paper’s choice.

D. Runtime performance (Figure 10)
- TensorRT‑LLM FP8 throughput on a single H100 80GB GPU:
  - Llama‑3.1‑Minitron‑4B‑Depth: average 2.7× faster than Llama‑3.1‑8B.
  - Llama‑3.1‑Minitron‑4B‑Width: average 1.8× faster.
  - FP8 provides ~1.4× boost over BF16 in their measurements.
  - Larger batches are possible for 4B models due to lower memory footprint.

E. Do results support the claims?
- Yes, the combination of teacher correction, thoughtful pruning, and logit‑only distillation produces students that are competitive or superior to same‑size baselines and, in cases, rival larger teachers (Tables 1–2).
- The ablations directly isolate the value of correction (Figures 4–5) and pruning+distillation (Figure 6). Depth‑pruning analysis ties design choices to downstream outcomes (Figures 8–9).

F. Mixed results or trade‑offs
- Base depth‑pruned 4B model shows a large reasoning drop on GSM8K (16.8%) relative to width‑pruned (41.2%), though instruction tuning narrows this gap (Table 2).
- On some code metrics, 4B aligned models are behind the best 8B aligned results; 8B aligned models remain stronger for code (Table 2).

## 6. Limitations and Trade-offs
- Dependence on a sizable correction/distillation corpus
  - Teacher correction uses ~100B tokens and distillation uses up to 380B tokens (Figure 1; Table 4). While far less than 15T pretraining, this is still substantial and requires significant compute (32 DGX H100 nodes).
- Dataset effects and potential shifts
  - Teacher correction can slightly change the teacher’s performance across tasks (some improve, some degrade; Teacher Correction section; Table 1 notes). The approach assumes the CT dataset is representative of desired capabilities.
- Design choices tailored to evaluated models
  - The specific depth blocks and width dimensions pruned are based on empirical analyses for Llama‑3.1‑8B and Mistral‑NeMo‑12B. Generalization to very different architectures or tokenizers, while plausible, is not explicitly validated.
- Distillation signal limited to logits
  - They use forward‑KL on logits only (Figure 2; Retraining with Distillation), omitting intermediate feature matching or ground‑truth CE loss. This is simpler and works well here, but might limit benefits in settings where hidden‑state alignment helps (not explored).
- Skipping NAS
  - The paper sidesteps the lightweight neural architecture search from [2], opting for manual configs (Pruning; Table 3). This improves practicality but could miss better‑performing architectures for some targets.
- Evaluation breadth
  - The paper focuses on widely used benchmarks (Tables 1–2). It does not report robustness to distribution shifts, safety/hallucination metrics, or multilingual breadth beyond XL‑Sum English subset.

## 7. Implications and Future Directions
- How this changes the landscape
  - It makes pruning+distillation viable when original pretraining data is unavailable—a common legal and practical constraint—by introducing teacher correction. This unlocks economical creation of high‑quality SLMs from a single well‑trained teacher (Figure 1; Teacher Correction).
  - Provides evidence that carefully chosen pruning (width or contiguous depth) plus distillation can match or exceed same‑size models and even surpass a larger teacher on selected tasks (Tables 1–2).

- Practical applications
  - Deployable 4B and 8B models with strong instruction‑following, reasoning, and function‑calling, suitable for edge servers or latency‑sensitive applications; the 4B variants offer 1.8–2.7× throughput boosts in FP8 (Figure 10).
  - Model families: Train one large model once; derive several smaller sizes via pruning+distillation with correction, drastically reducing total training cost.

- Research directions
  - Lighter correction techniques: Explore LoRA or layer‑norm‑only tuning to cut the ~100B token requirement (Teacher Correction section notes this as future work).
  - Multi‑objective pruning criteria: Combine LM loss with a basket of downstream tasks, plus calibration/robustness metrics, to choose depth blocks more systematically (Figures 8–9 hint at task sensitivity).
  - Layer‑wise dynamic depth at inference: Findings that early/late layers are most important (Figure 8) suggest conditional computation or routing strategies.
  - Enhanced distillation signals: Add hidden‑state, attention, or loss‑mixing (teacher CE + KL) where beneficial, or curriculum‑style correction schedules.
  - Integration with quantization and sparse kernels: Combine structured pruning with low‑precision formats (FP8/BF16/int8) and hardware‑aware compilation (TensorRT‑LLM) for further speedups.

> Bottom line: With “teacher correction,” downstream‑aware depth pruning, and simple width‑pruning heuristics, the Minitron approach provides a reproducible, compute‑efficient path to high‑quality small models without access to proprietary pretraining data—validated by strong base and instruction‑tuned results and tangible inference speedups (Figures 1–10, Tables 1–4).
