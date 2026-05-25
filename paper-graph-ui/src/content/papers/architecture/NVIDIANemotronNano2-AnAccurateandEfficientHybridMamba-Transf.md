# NVIDIA Nemotron Nano 2: An Accurate and Efficient Hybrid Mamba-Transformer Reasoning Model

**ArXiv:** [2508.14444](https://arxiv.org/abs/2508.14444)

## 🎯 Pitch

Nemotron Nano 2 unveils NVIDIA-Nemotron-Nano-9B-v2, a cutting-edge 9B-parameter hybrid Mamba–Transformer model meticulously engineered for high-throughput, long-chain reasoning. By leveraging innovative architectural pruning, distillation, and multi-stage alignment, it enables 128k-token context inference with up to 6× faster generation than competitors—all on a single 22 GiB GPU—unlocking practical, efficient deployment for demanding reasoning workloads in real-world settings.

---

## 1. Executive Summary (2-3 sentences)
Nemotron Nano 2 introduces `NVIDIA-Nemotron-Nano-9B-v2`, a 9B-parameter hybrid Mamba–Transformer model designed specifically for long-chain reasoning with far higher generation throughput at comparable accuracy to similarly sized baselines. Starting from a 12B base model trained on 20T tokens, it combines improved data/recipes, long-context pretraining to 128k tokens, multi-stage alignment (SFT, DPO, GRPO, RLHF), and pruning+distillation (Minitron) to run 128k-context inference on a single 22 GiB A10G GPU while achieving up to 6.3× higher output-token throughput than `Qwen3-8B` in generation-heavy settings (Figure 1).

## 2. Context and Motivation
- Problem addressed
  - Reasoning assistants generate long “thinking” traces before final answers. Generating many output tokens is the bottleneck; attention-heavy models slow down as sequence length grows and key–value (KV) cache memory explodes. The goal is to keep accuracy while dramatically increasing throughput and enabling 128k context on a 22 GiB GPU (§1, §4).
- Why it matters
  - Real systems (math, coding, tool use) often require 8k–16k generated tokens. Faster generation lowers latency and cost, enabling on-prem or edge deployment on modest GPUs. Long-context capability lets a single model handle large documents and multi-step traces (§1; Figure 1).
- Prior approaches and gaps
  - Pure Transformers are accurate but generation is costly for long sequences (quadratic attention scaling and KV cache size).
  - Recent hybrids (e.g., Jamba; cited in §1) and `Nemotron-H` replace many attention layers with selective state-space model layers such as Mamba to reduce complexity. However, prior work lacked: (a) a full, high-accuracy open model that sustains 128k context on a single 22 GiB GPU, (b) a carefully evaluated distillation-and-pruning path for reasoning models under memory/throughput constraints, (c) explicit “thinking budget” control (§1, §§3–4).
- Positioning
  - Nemotron Nano 2 builds on the `Nemotron-H` hybrid architecture, but adds: large new datasets (including high-fidelity math), FP8 training, a 512k-seq long-context extension, staged alignment with truncation-aware SFT, and a compression pipeline targeting 128k context on A10G (§§2–4). It releases checkpoints and most data (§1).

## 3. Technical Approach
Step-by-step overview across pretraining, long-context extension, alignment, and compression.

- Hybrid architecture (base model; §2.1, Figure 2, Table 1)
  - Structure: 62 layers total with only 6 self-attention layers (~8% of depth) evenly dispersed; 28 Mamba-2 layers; 28 FFN blocks (Figure 2).
  - Key dims: hidden 5120, FFN 20480, Grouped-Query Attention (GQA) with 40 Q heads / 8 KV heads; Mamba-2 with 8 groups, state dimension 128, head dimension 64, conv window 4; squared ReLU FFN; RMSNorm; separate input/output embeddings; no dropout, no biases (Table 1).
  - Why Mamba-2: Mamba is a state-space sequence model (SSM) whose per-token compute scales linearly and doesn’t maintain a growing KV cache like attention. Replacing most attention with Mamba accelerates long generation while retaining a few attention layers for global interactions (§2.1).
- Pretraining data and curriculum (§§2.2–2.3; Figure 3)
  - Curated data: updated Common Crawl (`Nemotron-CC-v2`), multilingual slices for 15 languages, a new high-fidelity math extraction pipeline (`Nemotron-CC-Math-3+` and `-4+` subsets), and curated GitHub code with license filtering and deduplication (§§2.2.1; Appendix A lists accepted licenses).
  - Synthetic data: STEM QA; regenerated math dialogues (`MIND`-style) from higher-quality math web text; multilingual DiverseQA; code QA; academic QA; and SFT-style data for general, math, code, and a new Fundamental Reasoning (FR) SFT set targeting logical/analytical reading comprehension (§§2.2.2, 2.3.2).
  - Three-phase curriculum (Figure 3): start broad/diverse; then pivot to higher-quality sources; switch points at 60% and 90% of tokens (§2.3).
  - Multilingual ablation (Table 2): diverse QA translated from English crawl (`DiverseQA-crawl`) yields the best Global-MMLU scores, so it gets higher weight (§2.3.1).
  - FR SFT ablation (Table 3): adding 5% FR-SFT improves MMLU-Pro from 44.24 to 56.36 and math average by ~1.8 points, with no harm to reasoning or code averages (§2.3.2).
- Numerics and training schedule (§§2.4–2.5)
  - FP8 training (E4M3 format) for all tensors with block-wise quantization; first/last four linear layers in BF16; optimizer state in FP32; stable training observed (§2.4).
  - Scale: 20T tokens; sequence length 8192; global batch 768; WSD learning rate schedule (stable LR 4.5e-4, min 4.5e-6); Adam β1=0.9, β2=0.95; weight decay 0.1 (§2.5).
- Long-context extension to 128k (§2.6; Table 4)
  - Strategy: continuous pretraining (CPT) with even longer sequence length 512k at a small constant LR (4.5e-6) to avoid cutting coherent long docs during Concat&Chunk. Uses 8-way tensor parallel + 16-way context parallel, batch size ensuring ~6M tokens/batch. 18.9B tokens in this phase (§2.6).
  - Long-doc QA synthesis: extract long academic docs >32k tokens; chunk; synthesize QAs and append; allocate 20% of blend to this data (§2.6).
  - Ablation (Table 4): training at 512k with synthetic long-doc QA yields the best RULER-128k (81.04 for `Nemotron-H-8B` ablation).
- Alignment pipeline (§3; Figure 4)
  - SFT Stage 1: ~80B tokens across math, coding, science, tool-calling, multilingual, safety, conversational. Concatenate to ~128k sequences; 10% of prompts include “empty” reasoning traces so the model can answer with thinking “off” (§§3.1–3.2).
  - SFT Stage 2: focus on tool-calling without concatenation to recover tool patterns (§3.2).
  - SFT Stage 3: reinforce long-context; introduce truncated reasoning traces (1–2k tokens) that still end with correct answers to teach budgeted thinking (§3.2, §3.4).
  - IFEval RL: apply rule-checked instruction-following reward; improves IFEval while other benchmarks fluctuate, so careful checkpointing is needed (§3.2).
  - DPO (Direct Preference Optimization): on-policy data inside a verifiable multi-step tool environment (Workbench) to strengthen multi-step/multi-turn tool use; focuses on BFCL v3 tasks (§3.2).
  - GRPO (Group Relative Policy Optimization): preference-based RL on helpfulness/chat (Arena-Hard) using `HelpSteer3` prompts; generate with/without thinking traces; a Qwen-based reward model evaluates rollouts (§3.2).
  - Model merging: linear interpolation of checkpoints to trade off reasoning vs chat; α≈0.5 works well (§3.2).
  - Budget control mechanism (§3.4; Figure 5): limit the number of `<think>` tokens. When the token budget is reached, the system closes `</think>` at the next newline (or forces closure within +500 tokens). Truncation-trained SFT makes outputs well-formed and avoids “compensating” by moving long rationales into the final answer (Figure 5b).
- Compression for A10G (22 GiB) at 128k (§4)
  - Constraint: inference memory budget set to 19.66 GiB (22.06 GiB minus framework buffer and 1.3 GiB for a vision encoder) while supporting 128k context and batch≥1 (§4.2).
  - Throughput target: measure vLLM throughput at ISL/OSL=8k/16k at max fitting batch on A10G (§4.2).
  - Importance estimation (lightweight, forward-only; §4.1)
    - Layer importance: iteratively remove a candidate layer, compute logit MSE vs original, prune the lowest-impact layer at each step.
    - FFN and embedding channel importance: aggregate neuron activations over a 1024-sample calibration set (mean and L2 norms) to rank/prune (§4.1).
    - Mamba head importance: nested activation-based scores per group to prune heads while respecting group structure (Taghibakhshi et al., 2025); in this work, small compression ratios made head pruning less beneficial (§4.2.2).
  - Lightweight NAS under memory constraint (§4.2)
    - Search over depth (remove 6–10 layers from 62), embedding width (4480–5120), FFN size (13440–20480), Mamba heads (112–128). Two-stage strategy:
      1) pick depth: after 6B KD tokens, 56 layers clearly outperform 54 and 52 (51.48 vs 47.35 and 44.92 average reasoning; Table 9);
      2) fix depth=56 and search width. Top-3 candidates distilled for 19B tokens and benchmarked (Table 10). “Candidate 2” (hidden 4480, FFN 15680, 128 Mamba heads) gives best accuracy (63.02 avg) with strong throughput (156.42 toks/s/GPU at 8k/16k, batch 8).
  - Distillation schedule (§4.3; Figure 6; Table 11)
    - Distill with KL-divergence on logits (“teacher” is the 12B model).
    - Reasoning model:
      - Depth-only KD (~60B tokens at 8,192).
      - Width-pruned KD: ~50B at 8,192; ~25B at 49,152; ~1B at 262,144.
      - DPO → GRPO → short KD at 262,144 to recover drops → RLHF → model merging (Figure 6 traces per benchmark).
      - Data mix ablation (Table 11): 70% post-training Stage 2 + 30% pretraining gives the best math accuracy after ~6B KD.
    - Base model:
      - KD after depth pruning (~120B at 8,192) → width-pruned KD (~360B at 8,192) → long-context KD (~2.5B at 524,288) using only pretraining data (§4.3).

## 4. Key Insights and Innovations
- Hybrid Mamba–Transformer optimized for generation-heavy reasoning
  - What’s new: Only ~8% attention layers with Mamba-2 elsewhere (Figure 2), tuned for long outputs and long contexts.
  - Why it matters: Enables up to 6.3× higher throughput at 8k/16k tokens on A10G (Figure 1 right) while matching or exceeding accuracy vs `Qwen3-8B` across AIME24/25, LiveCodeBench, BFCL v3, and RULER-128k (Figure 1 left).
- High-fidelity math pretraining corpus and FR SFT
  - Math dataset (`Nemotron-CC-Math-3+/-4+`): preserves equations across web formats via a Lynx-render + LLM standardization pipeline, delivers gains in math, code, and general knowledge (details §2.2.1; summarized in Mahabadi et al. 2025).
  - FR SFT ablation (Table 3): targeting analytical/logical reading comprehension materially improves MMLU-Pro (44.24 → 56.36) and math average.
  - Significance: Improves reasoning at higher difficulty levels without harming code or commonsense (§2.3.2).
- Long-context extension via 512k CPT with synthesized long-doc QA (§2.6; Table 4)
  - Insight: Training at 512k (not 128k/256k) reduces document fragmentation and best boosts RULER-128k. This yields strong long-context scores for both base (Table 5) and aligned models (Table 8).
- Truncation-aware SFT for explicit “thinking budget” control (§3.4; Figure 5)
  - Mechanism: Cap `<think>` tokens; close tag sensibly; teach the model with shortened traces so it does not spill reasoning into the final answer or continue “thinking” after closure.
  - Result: After truncation training, accuracy vs budget curves are stable, final answers remain concise, and responses are well-formed (Figure 5b).
- Compression under explicit memory/throughput constraints (§4)
  - Extension of Minitron to reasoning models: forward-only sensitivity scoring + NAS over depth/width subject to 19.66 GiB at 128k context; stage-wise KD to recover accuracy (Tables 9–11, Figure 6).
  - Outcome: 12B→9B while retaining 56 layers and shrinking widths; final `Nano-9B-v2` matches or beats similarly sized baselines and runs 128k on a 22 GiB A10G (Figure 1; §4.4).

## 5. Experimental Analysis
- Evaluation setup and metrics
  - Framework: based on `lm-evaluation-harness` with math grading via `math-verify` and EvalPlus variants for code; pass@1/avg@32 for code; standardized multiple-choice for general reasoning; long-context via RULER (13 tasks) (§2.7).
  - Throughput: vLLM output generation at ISL/OSL=8k/16k on A10G (§4.2).
- Main quantitative results
  - Throughput and accuracy vs `Qwen3-8B` (Figure 1):
    - Quote: “up to 6.3× higher throughput” at 8k/16k output-heavy setting (right panel).
    - Accuracy: comparable or better on AIME24 (81.9 vs 75.8), AIME25 (72.0 vs 69.3), LiveCodeBench (71.1 vs 59.5), BFCL v3 (66.9 vs 66.3), RULER-128k (78.9 vs 74.1) (left panel).
  - Base-model comparisons (Table 5):
    - `12B Base` vs `Qwen3-8B Base`: MMLU 78.24 vs 76.44; GSM8K 91.66 vs 84.00; MATH 83.54 vs 55.40; AIME24 pass@32 56.67 vs 20.00; HumanEval+ avg@32 61.03 vs 57.55; RULER-128k 84.74 (Gemma3-12B 80.70).
    - `9B Base` (pruned) remains competitive: e.g., GSM8K 91.36; MATH 80.50; HumanEval+ 58.50; RULER-128k 82.22.
  - Multilingual (Table 6):
    - Global-MMLU-Lite average: `12B Base` 75.13; `9B Base` 69.94; `Qwen3-8B Base` 72.81; `Gemma3-12B Base` 71.88.
    - MGSM average: `12B Base` 85.94; `9B Base` 84.67; `Qwen3-8B Base` 80.93; `Gemma3-12B Base` 66.33.
  - Aligned 12B reasoning model (Table 8):
    - AIME-2024: 85.42 (vs Qwen3-8B 75.83, Qwen3-14B 81.53).
    - AIME-2025: 76.25 (vs 69.31, 66.6).
    - MATH-500: 97.75 (vs 96.3, 96.85).
    - GPQA-Diamond: 64.48 (vs 59.61, 64.53).
    - LiveCodeBench: 70.79 (vs 59.5, 63.08).
    - IFEval (strict): 89.81 (vs 89.39, 91.32).
    - BFCL v3: 66.98 (vs 66.34, 68.01).
    - RULER@128k: 83.36 (vs 74.13, 73.55).
    - ArenaHard: 74 (vs 78.4, 87.7) — chat helpfulness remains a trade-off that is partly mitigated via checkpoint merging (§3.2).
- Ablations and robustness
  - Multilingual data selection: `DiverseQA-crawl` best average (Table 2).
  - FR-SFT efficacy: large lift on MMLU-Pro and small math gain (Table 3).
  - Long-context training length: 512k + synthetic QA best on RULER-128k (Table 4).
  - Depth sensitivity: accuracy correlates strongly with depth; 56-layer target chosen (Table 9).
  - Architecture selection under constraint: Candidate 2 best accuracy/throughput balance (Table 10).
  - Distillation data mix: 70% reasoning SFT + 30% pretraining best early math (Table 11).
  - Post-RL recovery and merging: Figure 6 shows DPO→GRPO boost tool use and instruction following, temporary dips on MMLU-Pro recovered by post-GRPO KD; RLHF improves chat alignment but induces drops that are mitigated by model merging.
- Do the results support the claims?
  - Yes for the central claims: Figure 1 demonstrates simultaneous accuracy parity/superiority and large throughput gains; Tables 5–6 show strong base model performance; Table 8 shows robust reasoning benchmarks and long-context strength. The observed trade-offs (e.g., ArenaHard) are transparently addressed via targeted stages and merging (§3.2; Figure 6).

## 6. Limitations and Trade-offs
- Accuracy vs chat helpfulness
  - ArenaHard remains lower than the strongest baselines (Table 8). The pipeline uses RL and checkpoint interpolation to trade off reasoning and chat, indicating inherent tension between these skills (§3.2, Figure 6).
- Tool-calling sensitivity to batching/concatenation
  - Concatenating samples to ~128k during SFT Stage 1 degraded tool-calling; a separate non-concatenated Stage 2 was needed (§3.2).
- Compression boundaries
  - Mamba head pruning brought limited benefit at the relatively modest compression ratios used; the final design prunes FFN/embeddings and layers but leaves many Mamba heads, constraining further size reduction (§4.2.2).
- Compute and data intensity
  - 20T-token pretraining + multiple KD and RL stages represent significant compute. Achieving 512k CPT and long-context KD requires distributed context parallelism (§2.6, §4.3).
- Assumptions about data quality
  - Large reliance on synthetic data (STEM, FR SFT, multilingual DiverseQA). While ablations support choices (Tables 2–3), broader generalization and contamination controls are challenging at this scale (decontamination used for math corpus; §2.2.1).
- Budget control edge cases
  - The inference system forces `</think>` if no newline appears within +500 tokens. This is a pragmatic heuristic; pathological formatting or adversarial prompts could still produce awkward closures (§3.4).

## 7. Implications and Future Directions
- How this changes the landscape
  - Demonstrates that hybrid Mamba–Transformer models can deliver state-of-the-art reasoning accuracy at much higher output throughput, and that a 9B model can sustain 128k context on a single 22 GiB GPU. This lowers the barrier to deploy capable reasoning assistants in constrained environments (Figure 1; §4.4).
- Follow-up research enabled
  - Adaptive hybridization: learn to allocate attention vs Mamba per layer or per token budget dynamically.
  - More aggressive but safe compression: improved Mamba head/channel importance estimation; structured low-rank methods; mixed-precision KV caches for the residual attention layers.
  - Learned thinking budgets: integrate budget prediction policies conditioned on task difficulty; joint training for optimal compute–accuracy trade-offs beyond fixed heuristics (§3.4).
  - Robust long-context training: curriculum that mixes retrieval, summarization, and tool use at 128k–512k to further improve RULER-like generalization (§2.6).
  - Safety and multilingual robustness: expand guardrails and quality checks in synthetic translation; investigate fairness/cultural bias with richer multilingual evaluations (Table 6; §3.1 Safety).
- Practical applications
  - On-prem copilots for math, coding, and tool-augmented workflows where long outputs and 100k+ contexts are common (e.g., enterprise knowledge, software repositories).
  - Cost-effective batch generation for agents that deliberate extensively before acting, thanks to higher output-token throughput (Figure 1).
  - Scalable function-calling agents with verified multi-step tool use (BFCL v3 and WorkBench setup; §3.2).

> Overall takeaway: by pairing Mamba-2 layers with a small number of attention layers, using high-fidelity math/FR data and long-context CPT, and compressing via forward-only importance scoring plus staged distillation, Nemotron Nano 2 delivers a 9B model that is both fast at long-form reasoning and competitive on accuracy across math, science, code, tool use, and long-context benchmarks (Figures 1, 5–6; Tables 5–6, 8–11).
