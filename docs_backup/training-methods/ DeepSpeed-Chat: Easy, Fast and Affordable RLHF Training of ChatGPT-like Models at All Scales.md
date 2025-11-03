# DeepSpeed-Chat: Easy, Fast and Affordable RLHF Training of ChatGPT-like Models at All Scales

**ArXiv:** [2309.07973](https://arxiv.org/abs/2309.07973)
**Authors:** Jeff Rasley, Minjia Zhang, Shuaiwen Leon Song, Shivaram Venkataraman, Samyam Rajbhandari, Jared Casper, Adam David Jaffe, Yuxiong He
**Institutions:** Microsoft

## 🎯 Pitch

DeepSpeed-Chat revolutionizes the training of ChatGPT-style models with its end-to-end system, combining inference and training optimizations into a unified 'Hybrid Engine.' This innovation dramatically reduces costs and time, democratizing access to advanced RLHF, enabling smaller labs to develop domain-specific models efficiently and cost-effectively.

---

## 1. Executive Summary
DeepSpeed-Chat introduces a complete, end-to-end system and toolkit for training ChatGPT-style models with Reinforcement Learning from Human Feedback (RLHF) that is both easy to use and highly efficient. Its central contribution is a unified “Hybrid Engine” that fuses DeepSpeed’s inference and training optimizations, enabling fast, affordable Step-3 RLHF training across model sizes from 1.3B to 175B parameters with strong throughput and memory scaling (e.g., OPT-13B in 9 hours for ≈$290 on 8×A100-80G; Table 1).

## 2. Context and Motivation
- Problem addressed
  - There is no accessible, cost-efficient, end-to-end RLHF pipeline capable of training large “ChatGPT-like” models. Existing open-source efforts provide instruction-tuned models or pieces of the pipeline but not a full, scalable RLHF stack that ordinary labs can run.
  - The most difficult stage is Step 3 (RLHF with PPO), which interleaves expensive generation (inference) with policy optimization (training) and stresses both memory and throughput.

- Why it matters
  - RLHF is the dominant technique for aligning LLMs with human preferences. Making it affordable and turnkey materially lowers the barrier to building domain-specific assistants and research prototypes.
  - System efficiency directly determines who can participate: single-GPU support and strong multi-node scaling democratize access.

- Prior approaches and limitations
  - Community models (e.g., ChatLLaMA, Alpaca, Vicuna, Dolly) emphasize instruction tuning but not full RLHF.
  - RLHF implementations built on “vanilla” PyTorch or frameworks like HuggingFace DDP and Colossal-AI suffer from low utilization and limited model sizes. The paper reports:
    - Single GPU: >10× throughput improvement over alternatives (Figure 3).
    - Multi-GPU: 6–19× over Colossal-AI and 1.4–10.5× over HuggingFace DDP (Figure 4).
    - Larger models fit: Colossal-AI runs up to 1.3B on one GPU and 6.7B on one A100-40G node; DeepSpeed-Chat runs ≈6.5B on one GPU and ≈50B on the same node (Section 5.2).

- Positioning
  - DeepSpeed-Chat provides (1) a one-script training experience, (2) a complete pipeline faithful to InstructGPT’s three steps, and (3) a new `Hybrid Engine` that unifies training/inference to accelerate Step 3 (Sections 1–4; Figure 1 and Figure 2).

## 3. Technical Approach
This section explains the pipeline, design choices, and the Hybrid Engine’s mechanics.

- Terminology (defined on first use)
  - `RLHF`: Finetuning a model using a reward signal derived from human preference data.
  - `SFT` (Supervised Finetuning): Standard supervised training on human-written responses.
  - `Reward model (RW)`: A model trained to score responses by preference rankings.
  - `PPO` (Proximal Policy Optimization): A reinforcement learning algorithm that updates a policy (the “actor”) while constraining it not to deviate too far from a reference policy.
  - `Reference model`: A frozen copy of the actor used to compute a KL-penalty so the policy stays close to its supervised origin.
  - `Critic model`: The value function used in PPO to estimate long-term return (advantage).
  - `ZeRO`: A memory-optimization family that shards optimizer states, gradients, and parameters across GPUs to fit larger models.
  - `LoRA`: Low-Rank Adaptation; trains small low-rank adapters instead of full weights to reduce memory and compute.
  - `Tensor parallelism (TP)`: Splits layers’ computations across GPUs to increase effective memory for a single model replica.
  - `KV cache`: Stores key/value projections from previous generated tokens to avoid recomputation during autoregressive decoding.
  - `EMA` (Exponential Moving Average): Maintains a smoothed parameter copy that can yield better final quality.

- Full RLHF pipeline (Figure 1)
  1) Step 1: `SFT`  
     - Start from a pretrained model and finetune on curated prompt–response pairs.
  2) Step 2: `Reward model` training  
     - Train a smaller model to rank multiple responses to the same prompt; the loss encourages correct preference orderings.
  3) Step 3: `RLHF with PPO`  
     - The SFT model becomes the `actor`. For each prompt, the actor generates a response (inference).  
     - The reward is computed by the `RW`, combined with a KL penalty to the frozen `reference` model to prevent drift.  
     - PPO updates the actor using a `critic` for advantage estimation.  
     - Optional features:
       - `EMA`: collect an EMA checkpoint for evaluation.
       - `Mixture training`: blend the next-token prediction loss with PPO to preserve general capabilities (e.g., SQuAD2.0 retention noted in Section 3).

- Data handling
  - `Data abstraction and blending` unify different datasets into a common format and split/blend them coherently across the three stages (Section 3).
  - Benchmark setting for Step 3 is explicit: 1 epoch over 135M tokens: 67.5M prompt tokens (≈131.9k prompts of length 256) + 67.5M generated tokens (≈131.9k responses of length 256) with a max global batch size of 0.5M tokens (1024 prompt–response pairs) per step (footnote under Table 2 and linked BenchmarkSetting.md).

- Hybrid Engine (Figure 2; Section 4)
  - Design goal: Step 3 alternates between generation (inference) and PPO updates (training). The engine seamlessly switches modes and reconfigures memory/parallelism to suit each phase.
  - How it works
    - Inference phase (experience generation):
      - Use inference-adapted kernels and a lightweight memory manager for the `KV cache` and intermediate activations.
      - Prefer `tensor parallelism` to fit large models across multiple GPUs while minimizing cross-GPU traffic during autoregressive decoding.
      - Result: much higher tokens-per-second than training-optimized stacks (Section 4).
    - Training phase (PPO updates):
      - Switch to DeepSpeed Training engine using `ZeRO` sharding and optional `LoRA` to maximize batch size and fit model states efficiently.
      - Re-map parameters and buffers between inference- and training-friendly layouts, and re-allocate memory to favor the current phase (“Data remapping, switch parallelism, memory management” in Figure 2).
    - The engine toggles via standard model `train()` and `eval()` modes, hiding complexity from the user while composing optimizations from both inference and training stacks (Section 4).

- Ease-of-use interface
  - One command runs all three stages and produces a usable assistant:
    - Example: `python train.py --actor-model facebook/opt-13b --reward-model facebook/opt-350m --deployment-type singlenode` (Section 2.1).
  - An RLHF API allows custom loops:
    - `DeepSpeedRLHFEngine` + `DeepSpeedPPOTrainer` expose `generate_experience()` and `train_rlhf()` for flexible experimentation (Section 2.3).

- Why these design choices?
  - Generation is memory-bandwidth bound and dominates end-to-end time if not optimized (Figure 5). Inference kernels + TP are best here.
  - PPO training is compute-bound and benefits from large batches and sharded states; `ZeRO` + optional `LoRA` are best here.
  - Switching strategies per phase yields high “effective throughput” (Section 5.3).

## 4. Key Insights and Innovations
- Unifying training and inference into a single `Hybrid Engine` (fundamental innovation)
  - What’s new: Automatic switching between inference-optimized and training-optimized stacks, including different parallelism and memory layouts per phase (Figure 2).
  - Why it matters: The generation phase dominates Step 3 runtime; accelerating it while preserving efficient training yields large end-to-end gains. The paper reports up to 9× faster generation than HuggingFace and 15× than Colossal-AI (Figure 5), which translates into 6–19× overall Step-3 throughput gains on multi-GPU nodes (Figure 4).

- End-to-end, reproducible RLHF pipeline faithful to InstructGPT with optional quality-preserving features (incremental but important)
  - Includes SFT, reward modeling, PPO; adds `EMA` and `mixture training` that many open-source reproductions omit (Section 3; Figure 1). This makes results more representative of production-grade RLHF.

- Memory–throughput co-design that scales from one GPU to multi-node clusters (significant system contribution)
  - `ZeRO` + `LoRA` for training; `TP` + inference kernels for generation; dynamic reconfiguration between phases (Section 4).
  - Practical effect: single-GPU training of nontrivial models and support for very large models on clusters. Table 3 shows max single-GPU model sizes (e.g., OPT-13B on A100-80G).

- Data abstraction and blending layer (useful engineering contribution)
  - A common format and split/blend logic across the three stages simplifies multi-dataset RLHF experiments (Section 3).

## 5. Experimental Analysis
- Evaluation methodology
  - Focus: system efficiency, scalability, and affordability for Step 3 (the bottleneck). The paper reports end-to-end times and Azure cost approximations, throughput comparisons to two frameworks (HuggingFace DDP and Colossal-AI), and time breakdowns across phases.
  - Hardware: single GPUs, single-node 8×A100-40G and 8×A100-80G DGX nodes, and up to 64×A100-80G multi-node clusters (Tables 1–2; Figures 3–4, 6–7).
  - Data/recipe for Step 3: 135M total tokens, sequence lengths 256 (prompts) + 256 (generated), max global batch 0.5M tokens per step (footnote under Table 2).

- Main quantitative results
  - End-to-end training time and cost
    - Single node (8×A100-80G), Step 3:
      > Table 1: OPT-13B in 9 hours (≈$290), OPT-30B in 18 hours (≈$580), OPT-66B in 2.1 days (≈$1620); OPT-6.7B in 4.1 hours (≈$132).
    - Multi-node (64×A100-80G), Step 3:
      > Table 2: OPT-13B in 1.25 hours (≈$320), OPT-30B in 4 hours (≈$1024), OPT-66B in 7.5 hours (≈$1920), OPT-175B in 20 hours (≈$5120).
    - Stage-by-stage breakdowns:
      > Table 4 (OPT-13B on one 8×A100-40G node): Step 1 = 2.5 hr, Step 2 = 0.25 hr, Step 3 = 10.8 hr, Total = 13.6 hr.  
      > Table 5 (OPT-66B on 8 nodes of 8×A100-80G each): Step 1 = 82 min, Step 2 = 5 min, Step 3 = 7.5 hr, Total = 9 hr.  
      > Table 6 (OPT-1.3B on single A6000 48G): Step 1 ≈ 2900 s, Step 2 ≈ 670 s, Step 3 ≈ 1.2 hr, Total ≈ 2.2 hr.

  - Single-GPU feasibility
    > Table 3: Max single-GPU model sizes include OPT-13B (A100-80G), OPT-6.7B (A6000-48G or A100-40G), OPT-2.7B (V100-32G).

  - Throughput comparisons
    - Single GPU (Figure 3): DeepSpeed-Chat achieves over 10× higher Step-3 throughput than competing stacks; alternatives often hit out-of-memory for larger models.
    - Multi-GPU (Figure 4): 6–19× faster than Colossal-AI and 1.4–10.5× faster than HuggingFace DDP across models up to 13B on 8×A100-40G.
    - Generation-phase speedups (Figure 5): Up to 9× vs HuggingFace and 15× vs Colossal-AI at the generation-heavy portion of Step 3.

  - Effective throughput and scalability
    - Figure 6: Effective throughput (TFlops/GPU) is highest for mid-sized models (≈6.7B–66B). Even at 175B the per-GPU efficiency remains higher than at 1.3B, though lower than the 6.7B–66B peak due to batch-size limits.
    - Figure 7: Super-linear scaling at small scale (more GPUs free memory for larger per-GPU batches under `ZeRO`), transitioning to near-/sub-linear scaling when the global batch cap (1024 sequences of length 512) becomes the bottleneck.

- Do the experiments support the core claims?
  - Yes for system efficiency: multiple figures show consistent, large speedups and detailed time/cost breakdowns tied to precise hardware and a specified training recipe.
  - Less so for model quality: the paper includes a short example conversation and mentions `mixture training` for benchmark retention (Section 3), but provides no quantitative quality metrics (e.g., human eval, win rate, or held-out benchmarks).

- Ablations and diagnostics
  - Phase breakdown (Figure 5) explains why accelerating inference (generation) is decisive.
  - Effective throughput decomposition (Figure 6) and the generation/training split (~20% vs ~80% of total computation; Section 5.3) clarify where different optimizations matter.
  - No ablations on `EMA` or `mixture training` hyperparameters are reported; no sensitivity analyses across datasets/network fabrics are shown.

- Robustness checks and failure cases
  - OOM behavior for baseline systems appears in Figures 3–4 (marked by missing bars), highlighting DeepSpeed-Chat’s broader model-size envelope.
  - The paper does not report robustness to data/domain shift or instability in PPO training.

## 6. Limitations and Trade-offs
- Quality evaluation is limited
  - No quantitative human preference metrics or standardized alignment benchmarks are reported. Performance claims are primarily about speed, scale, and cost.

- Assumptions tied to the reported numbers
  - All time/cost figures for Step 3 rely on a specific recipe: one epoch over 135M tokens with sequence lengths of 256/256 and a max global batch of 0.5M tokens (footnote under Table 2). Different datasets, lengths, or batch caps will change results.

- Algorithmic scope
  - PPO is the only RL algorithm implemented and evaluated in the paper. Other RLHF variants (e.g., DPO, RLAIF, KL-free methods) are not analyzed here.

- Hardware and software ecosystem bias
  - Results are on NVIDIA A100-class hardware with DeepSpeed’s kernels and sharding stack. Portability to other accelerators or interconnects is untested in this paper.

- Communication/memory trade-offs
  - The engine switches between `TP` (generation) and `ZeRO` (training). While this is beneficial overall, switching incurs data remapping and may add synchronization overhead; the costs are not separately quantified.
  - Very large models (e.g., 175B) see reduced per-GPU efficiency due to batch-size limits (Figure 6), even though absolute time is still competitive at scale (Table 2).

- Scope of single-GPU support
  - Table 3 reports model sizes that fit on a single GPU, but full end-to-end times are only detailed for smaller models (e.g., OPT-1.3B; Table 6). Running large models on one GPU may still be impractically slow for real projects.

## 7. Implications and Future Directions
- How this changes the landscape
  - DeepSpeed-Chat makes full RLHF training practical on commodity-to-cloud hardware, narrowing the gap between research labs and large organizations. The one-script pipeline and RLHF APIs lower the barrier to entry for alignment research and domain adaptation.

- What follow-up research it enables
  - Algorithmic: Plug in alternative RLHF objectives (e.g., DPO, IPO, RLAIF), test different KL-control schemes, or evaluate the effects of `EMA`/`mixture training` quantitatively.
  - Systems: Explore auto-tuning for phase-specific parallelism, overlap communication with PPO compute, or extend to heterogeneous clusters and non-NVIDIA accelerators.
  - Data: Study data blending strategies, active preference collection, and cross-lingual RLHF at scale using the provided abstractions.

- Practical applications
  - Cost-effective alignment of specialty assistants (coding, biomedical, legal) via RLHF with modest GPU budgets.
  - Rapid experimentation with reward models and PPO schedules using the provided APIs (Section 2.3).
  - Scaled training for very large models when clusters are available, with predictable time/cost profiles (Tables 1–2).

In sum, the paper’s primary advance is systems-level: a unified engine and pipeline that substantially accelerate and scale Step-3 RLHF. The experimental evidence for efficiency is strong (Figures 3–7; Tables 1–6), while model quality evaluation remains an open area for deeper study.
