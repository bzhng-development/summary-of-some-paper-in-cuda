# Llama-Nemotron: Efficient Reasoning Models

**ArXiv:** [2505.00949](https://arxiv.org/abs/2505.00949)

## 🎯 Pitch

Llama-Nemotron debuts a family of open, efficient 'reasoning' language models—spanning 8B to 253B parameters—combining state-of-the-art multi-step reasoning with remarkably fast, memory-efficient inference and a user-controlled reasoning switch. By leveraging neural architecture search tailored for hardware, advanced distillation, targeted pretraining, and large-scale reinforcement learning, Llama-Nemotron not only outperforms previous open models but also democratizes scalable reasoning for enterprises and researchers, enabling accessible deployment and dynamic control of answer depth across diverse real-world applications.

---

## 1. Executive Summary (2-3 sentences)
Llama‑Nemotron introduces an open family of “reasoning” language models—`LN‑Nano` (8B), `LN‑Super` (49B), and `LN‑Ultra` (253B)—that combine strong multi‑step problem solving with high inference efficiency and a user‑controllable reasoning toggle (“detailed thinking on/off”). Using a hardware‑aware neural architecture search, targeted distillation/pretraining, supervised reasoning traces, and large‑scale reinforcement learning (RL), the flagship `LN‑Ultra` matches or exceeds state‑of‑the‑art open models on reasoning while running faster and in less memory, especially on 8×H100 GPUs (Figures 2 and 4; Tables 1 and 5).

## 2. Context and Motivation
- Problem addressed
  - Recent “reasoning LLMs” achieve high accuracy by generating long chains of thought, but they are expensive to run and often slow at inference because they expand many tokens per answer (Section 1). This constrains both user experience and the feasibility of large agent systems.
  - Users also need control: many tasks do not benefit from verbose multi‑step reasoning, so a model should switch style without swapping models (Section 1).

- Why it matters
  - Inference cost and latency have become a limiting factor for system‑level intelligence and real‑world deployment (Section 1). Efficient reasoning enables broader, cheaper, and faster use of advanced capabilities in applications like STEM tutoring, code assistants, and agent pipelines.

- Prior approaches and gaps
  - Strong closed and open reasoning models (e.g., OpenAI’s o1, DeepSeek‑R1) achieve accuracy but typically require heavy hardware (e.g., 8×H200 for DeepSeek‑R1) and lack explicit, open, hardware‑optimized architectures with controllable reasoning modes (Sections 1–2; Figure 4 caption).
  - Open instruction‑tuned models (e.g., Llama 3.x) are efficient but generally weaker on multi‑step reasoning without additional post‑training, and they do not provide a dynamic reasoning toggle (Sections 1, 3).

- Positioning
  - Llama‑Nemotron bridges the gap by: (1) redesigning base Llama models for inference efficiency through a NAS framework (`Puzzle`) and FFN Fusion (Section 2), (2) recovering and enhancing quality via distillation and continued pretraining (Section 2.2), then (3) adding reasoning via supervised traces and RL (Sections 4–5), all while exposing a simple runtime switch for reasoning behavior (Sections 1, 3).

## 3. Technical Approach
Step‑by‑step pipeline spanning five stages (Sections 1–2, 4–6):

1) Creating inference‑optimized backbones with `Puzzle` NAS (Section 2; Figure 3)
- What `Puzzle` does:
  - Builds a “library” of alternative transformer “blocks” for each layer using block‑wise local distillation: each candidate block is trained to mimic the original layer’s behavior while trading off accuracy vs. speed/memory (Section 2; Figure 3 Step 1).
  - Available variants include:
    - Attention removal: omit attention in some layers to reduce compute and KV‑cache memory (Section 2).
    - Variable FFN width: shrink or widen the feed‑forward network’s hidden size at different ratios (Section 2).
    - Other options are supported (e.g., grouped‑query attention, linear attention, no‑ops), but the two above dominated the LN models’ efficiency/quality trade‑off (Section 2).
  - Assembling the final model:
    - A mixed‑integer programming (MIP) solver selects one candidate block per layer to optimize accuracy under deployment constraints (throughput, latency, memory, hardware) (Section 2; Figure 3 Step 2).
- Why this design:
  - It directly searches the accuracy‑efficiency Pareto frontier per layer and per hardware target instead of naively compressing the whole network, allowing heterogeneous layers specialized for throughput or memory (Section 2.1).

2) Vertical depth reduction with `FFN Fusion` (LN‑Ultra only) (Section 2; “Vertical Compression with FFN Fusion”)
- Mechanism:
  - After `Puzzle` removes attention from some layers, adjacent FFN blocks can appear. `FFN Fusion` replaces consecutive FFNs with fewer but wider FFNs that execute in parallel, reducing sequential depth and inter‑GPU communication (Section 2; FFN Fusion).
- Benefit:
  - Improves latency and utilization—crucial on multi‑GPU nodes (Section 2).

3) Recovery training: knowledge distillation and continued pretraining (CPT) (Section 2.2; Table 1)
- Purpose:
  - Restores/boosts quality after architectural changes and improves inter‑block compatibility.
- Setup:
  - `LN‑Super`: 40B tokens of distillation on Distillation Mix (Section 2.2).
  - `LN‑Ultra`: 65B distillation + 88B CPT on Nemotron‑H phase 4 (Section 2.2).
- Outcome before any reasoning SFT/RL (Table 1):
  - `LN‑Ultra‑CPT` meets or exceeds strong baselines on challenging tasks, e.g. MATH500 80.4 vs. 69.6 for Llama‑3.1‑405B, and RULER‑128K 83.2 vs. 73.7 (Table 1).

4) Reasoning‑focused supervised fine‑tuning (SFT) with a runtime reasoning toggle (Section 4; Section 3; Table 2)
- Reasoning toggle:
  - A lightweight system instruction—`"detailed thinking on"` or `"detailed thinking off"`—conditions the model to include or hide chains of thought. Format rewards later enforce `<think> ... </think>` tags when “on” and their absence when “off” (Sections 1, 3, 5.1).
- Synthetic data curation (Section 3; Table 2):
  - Math: harvested problems from AoPS forums; invalid/problematic items filtered; final answers extracted for automatic checking; decontamination against benchmarks; multi‑solution generation with DeepSeek‑R1 (16 samples) and Qwen2.5‑Math‑7B (64); wrong answers removed via LLM‑based equivalence checking (Section 3.1.1).
  - Code: 28,904 programming tasks aggregated (TACO, APPS, CodeContests, CodeForces) with strict decontamination; DeepSeek‑R1 produces multi‑sample solutions with explicit reasoning in `<think>` tags; code segments validated (Tree‑Sitter); ~488K Python samples (Section 3.1.2). Scaling study indicates larger, harder datasets keep improving coding performance—no early plateau (Section 3.1.2 “Data Scaling Insights”).
  - Science: mixture of real and synthetic MCQs across physics/biology/chemistry; decontaminated against GPQA/MMLU/MMLU‑Pro; DeepSeek‑R1 generates multi‑trace solutions; majority voting when gold answers are missing (Section 3.1.3).
  - General: open‑domain instructions/prompts with multiple responses filtered by a 70B reward model; augmented with safety/function‑calling datasets (Sections 3.1.4, 3.2).
  - Paired “reasoning on/off” responses to teach the toggle, with off‑mode responses generated by Llama‑3.1‑Nemotron‑70B‑Instruct or Llama‑3.3‑70B‑Instruct (Section 3.2).
  - Overall blend: 33,011,757 samples; notably math is 66.8% and code 30.6% of the corpus (Table 2).
- Model‑specific SFT (Section 4.2):
  - `LN‑Nano`: three‑stage SFT, starting with reasoning‑only to avoid degenerate repetition, then mixing in non‑reasoning, finishing with chat/instruction/tool‑calling; effective sequence length 32k, global batch 256 (Section 4.2).
  - `LN‑Super`: one epoch over full SFT dataset; seq length 16k; global batch 256; fixed LR 5e‑6 (Section 4.2).
  - `LN‑Ultra`: sequence packing to ~24k effective length; global batch 256; cautious LR schedule (warmup to 1e‑5 then cosine to 1e‑6); training required restarts due to gradient instabilities after the first epoch (Section 4.2).

5) Reinforcement learning for scientific reasoning (LN‑Ultra) (Section 5; Figures 5–6)
- Why RL: SFT distilled from a teacher caps performance at the teacher’s level; RL enables surpassing it (Section 5).
- Algorithm:
  - `GRPO` (Group Relative Policy Optimization): sample groups of responses per prompt and update toward those that score better relative to the group (Section 5.1).
- Rewards (Section 5.1):
  - Accuracy reward: an LLM judge (Llama‑3.3‑70B‑Instruct) checks if the final answer matches ground truth (numeric, sentence, paragraph).
  - Format reward: enforce `<think>` tags when “on” and their absence when “off”, similar to DeepSeek‑R1 (Section 5.1).
- Hard‑example curriculum (Section 5.1; Figure 6):
  - Precompute pass rates using `LN‑Super`; discard easy items (pass rate ≥ 0.75); progressively shift batches from easier to harder using a Gaussian target distribution over pass rates. Improves stability and accuracy (Figure 6).
- Scale and settings (Section 5.1, 5.2):
  - 72 nodes × 8×H100; generation with `vLLM`, training with `Megatron‑LM`; FP8 decoding; ~140k H100 hours; sampling 16 responses/prompt at temperature=1, top_p=1; global batch 576, 2 optimizer steps per rollout (Sections 5.1–5.2).

6) Instruction following and RLHF (Section 6)
- Short `RLOO` run (leave‑one‑out style RL) trains compliance with multi‑step instructions using a verifier reward (Section 6.1).
- RLHF via Reward‑aware Preference Optimization (`RPO`) improves helpfulness/chat quality on HelpSteer2 prompts using the 70B reward model; two online RPO iterations lift `LN‑Super` Arena‑Hard from 69.1 to 88.1 (Section 6.2). `LN‑Ultra` uses GRPO for a brief RLHF run (Section 6.2). `LN‑Nano` uses offline RPO for reasoning control then instruction following (Section 6.2).

7) System and memory engineering to make RL feasible (Section 5.2)
- Co‑locate generation and training on the same GPUs, hot‑swapping weights via shared memory; finely tuned parallelism: tensor=8 with sequence parallel, context=2, pipeline=18, data=2; `vLLM` tensor=8, data=72 (Section 5.2.1).
- Memory profiling (GPU/CPU/dev/shm), reshaping heavyweight tensors (one had 13B elements ≈ 26 GB in BF16), and balancing pipeline with identity layers to avoid OOM while keeping >90% utilization (Section 5.2.2).
- FP8 generation path in `vLLM` with per‑token activation scaling and per‑tensor weight scaling; meta‑tensor init avoids materializing BF16 engines; enables cudagraphs and yields ~1.8× generation speedup, peaking at 32 tokens/s/GPU/prompt (Section 5.2.3).

8) Deployment targets (Section 2.1; Figure 4)
- `LN‑Super`: optimized for a single H100 at tensor‑parallel=1 with ≥2.17× throughput over Llama‑3.3‑70B‑Instruct even when the latter uses TP=4; supports ~300k cached tokens at FP8 (Section 2.1).
- `LN‑Ultra`: optimized for a single 8×H100 node; 1.71× latency reduction versus Llama‑3.1‑405B‑Instruct; up to 3M cached tokens in FP8 and 600k in BF16 on an H100 node (Section 2.1); better accuracy‑throughput trade‑off than Llama‑3.1‑405B and DeepSeek‑R1 (Figure 4).

## 4. Key Insights and Innovations
- Hardware‑constrained heterogeneous architecture search that removes attention where safe and compresses FFNs (Section 2; Figure 3)
  - What’s new: per‑layer block replacement chosen by a MIP solver under real deployment constraints (latency/throughput/memory), not just uniform pruning/quantization.
  - Why it matters: achieves large throughput gains without large quality loss; enables “right‑sizing” each layer for specific hardware budgets (Section 2.1; Figure 4).

- `FFN Fusion` to reduce sequential depth (Section 2; “Vertical Compression with FFN Fusion”)
  - What’s new: exploit attention‑removed stretches to merge adjacent FFNs into fewer, wider parallel FFNs.
  - Why it matters: reduces inter‑layer hops and improves GPU utilization—key for multi‑GPU latency (Section 2).

- A practical reasoning toggle learned end‑to‑end and enforced by rewards (Sections 3–5)
  - What’s new: a simple, lightweight system prompt (“detailed thinking on/off”) paired with data and format‑rewards that reliably add or hide chain‑of‑thought spans (`<think>` tags).
  - Why it matters: users can pick concise answers for routine questions or deep reasoning for hard tasks without switching models (Sections 1, 3, 5.1).

- Curriculum‑driven RL with FP8 generation inside the training loop (Sections 5.1–5.2)
  - What’s new: combine RL on hard‑filtered questions, a pass‑rate curriculum, and an FP8 `vLLM` path that speeds decoding 1.8× and re‑enables cudagraphs (Section 5.2.3).
  - Why it matters: makes very large‑scale reasoning RL feasible on commodity 8×H100 nodes and enables surpassing the teacher (Figures 5–6; Table 5).

- Open, enterprise‑permissive release with full post‑training dataset and code (Abstract; Section 1)
  - What’s new: open weights (`LN‑Nano`, `LN‑Super`, `LN‑Ultra`, plus `LN‑Ultra‑CPT`), the `Llama‑Nemotron‑Post‑Training‑Dataset`, and training codebases (NeMo, NeMo‑Aligner, Megatron‑LM).
  - Why it matters: enables reproducibility and downstream research on reasoning efficiency and RL at scale.

## 5. Experimental Analysis
- Evaluation methodology (Section 7.1)
  - Reasoning: AIME24/25 (competition math), GPQA‑Diamond (graduate‑level science), MATH500 (math proof‑style problems), LiveCodeBench (fresh, contamination‑controlled coding).
  - Non‑reasoning: IFEval for instruction following, BFCL V2 Live for tool/function calling, Arena‑Hard for conversational preference.
  - Settings: 32k evaluation context (even if trained at 16–24k, extended context improves long reasoning completion); reasoning‑on uses temperature 0.6, top‑p 0.95; reasoning‑off uses greedy decoding; up to 16 completions prompt, report average pass@1 (Section 7.1). Checkpoints selected on a reasoning subset; small reasoning benchmarks can have high variance (Section 7.1).

- Efficiency vs. accuracy (Figure 4; Section 2.1)
  - Quote:
    > LN‑Ultra “consistently outperforms DeepSeek‑R1 and Llama‑3.1‑405B in both accuracy and efficiency” on GPQA‑Diamond across two throughput settings; it runs on 8×H100 while R1 requires 8×H200 (Figure 4).

- Quality after CPT (before SFT/RL) (Table 1)
  - Quote:
    > `LN‑Ultra‑CPT` vs. Llama baselines: MMLU 88.1 (vs. 88.6 for 405B), MATH500 80.4 (vs. 69.6 for 405B), HumanEval 88.4 (vs. 86.0 for 405B), RULER‑128K 83.2 (vs. 73.7 for 405B).
  - Interpretation: architectural optimization plus short distillation/CPT can meet or exceed the 405B baseline on several tasks even before reasoning SFT (Section 2.2).

- LN‑Nano results (Table 3)
  - Strong for its size on math and code:
    - GPQA‑Diamond reasoning‑on 54.1% vs. 49.0% for DeepSeek‑R1‑Distilled‑Llama‑8B; MATH500 95.4% vs. 89.1% (Table 3).
    - LiveCodeBench (2408–2502) 46.6% pass@1, beating 8B baselines (Table 3).
  - Reasoning toggle control: off‑mode reduces verbosity and sometimes accuracy (expected), but tool calling (BFCL) stays nearly unchanged (63.9/63.6) (Table 3).

- LN‑Super results (Table 4)
  - Reasoning‑on performance competitive with larger/distilled peers:
    - GPQA‑Diamond 66.7% (vs. 65.2% DeepSeek‑R1‑Distilled‑Llama‑70B; 58.8% QwQ‑32B) (Table 4).
    - AIME25 60.0% (vs. 55.0% DeepSeek‑R1‑Distilled‑70B) (Table 4).
    - MATH500 96.6% (on par with QwQ‑32B’s 96.2) (Table 4).
  - Non‑reasoning alignment:
    - IFEval up to 89.2 after targeted instruction‑following RL; Arena‑Hard 88.3 after RLHF (Table 4; Section 6.2).
  - Mixed results:
    - LiveCodeBench lags (45.5%) because SFT used an older dataset version (Section 7.3).

- LN‑Ultra results and surpassing the teacher (Table 5; Figures 5–6)
  - Quote:
    > GPQA‑Diamond reasoning‑on 76.0% for `LN‑Ultra`, surpassing DeepSeek‑R1 at 71.5% and also exceeding Llama‑4 Maverick (69.8) and Llama‑3.1‑405B (43.4) (Table 5).
    > AIME24 80.8% vs. 79.8% R1; AIME25 72.5% vs. 70.0% R1 (Table 5).
    > LiveCodeBench (2410–2502) 68.1%, well above Llama‑4 Behemoth 49.4 and Maverick 43.4 (Table 5).
  - RL effect:
    - `LN‑Ultra‑SFT` approaches R1, but RL is “critical for surpassing DeepSeek‑R1, particularly on GPQA” (Table 5 and Section 7.4). Figure 5 shows steady GPQA‑D improvements during RL; Figure 6 shows the curriculum beats random batching.

- LLM‑as‑a‑Judge generalization (Table 6)
  - Quote:
    > On JudgeBench, `LN‑Ultra` attains 79.14 overall, above DeepSeek‑R1 at 73.14, second only to o3‑mini(high) 80.86. `LN‑Super` (69.71) surpasses o1‑mini (65.71) (Table 6).

- Do the experiments support the claims?
  - Yes for both axes:
    - Efficiency: `Puzzle` + `FFN Fusion` meet specific deployment targets and yield better accuracy‑throughput than strong baselines (Figure 4; Section 2.1).
    - Quality: Across diverse reasoning and non‑reasoning benchmarks, the models are competitive or leading; RL lifts `LN‑Ultra` beyond the teacher on GPQA, aligning with the stated motivation for RL (Table 5; Figures 5–6).
  - Trade‑offs are candidly reported: instruction following vs. conversationality (IFEval vs. Arena‑Hard) and a coding shortfall for `LN‑Super` due to dataset versioning (Section 7.3).

## 6. Limitations and Trade-offs
- Reliance on powerful teacher models and LLM‑based rewards
  - SFT uses DeepSeek‑R1 traces extensively; RL’s accuracy reward depends on a 70B LLM judge (Section 5.1). This can propagate teacher biases and judge errors; there is no human‑verified gold for every sample.

- RL at scale is resource‑intensive
  - ~140k H100 hours, 72 nodes, complex parallelism, and custom FP8 decoding (Sections 5.1–5.2). This limits accessibility for many labs.

- Instabilities and engineering complexity
  - `LN‑Ultra` SFT required restarts due to gradient explosions (Section 4.2). The RL stack needed weight hot‑swapping, shared‑memory staging, and careful pipeline balancing to avoid OOM (Section 5.2.2).

- Mode‑control vs. content leakage
  - The toggle and format reward enforce presence/absence of `<think>` tags (Section 5.1), but the paper does not quantify rare failures to comply or cases where implicit reasoning still leaks into the answer in off‑mode.

- Evaluation scope and variance
  - Reasoning benchmarks like AIME have small test sizes and high variance (Section 7.1). Although decontamination is applied in data pipelines (Sections 3.1.1–3.1.3), residual contamination risk always exists in large‑scale web‑sourced corpora.

- Coverage gaps
  - RL for reasoning is applied only to `LN‑Ultra`; smaller models may benefit less from RL per preliminary observations (Section 5). Some capabilities (e.g., long‑context beyond 32k) are not evaluated despite 128k context support (Section 1; Section 7.1).

## 7. Implications and Future Directions
- Field impact
  - Demonstrates that hardware‑aware heterogeneous architectures plus targeted distillation can yield large, fast reasoning models without sacrificing quality, and that RL can push them past their teachers at acceptable cost on 8×H100 hardware (Figure 4; Tables 1 and 5).
  - Establishes a practical, user‑visible reasoning toggle that lets one model serve both concise assistant use and deep step‑by‑step problem solving (Sections 1, 3, 5).

- Enabled research directions
  - Open dataset and code allow:
    - Replicating large‑scale reasoning RL with FP8 decoding.
    - Studying per‑layer NAS choices and FFN Fusion effects on different hardware (Section 2; code releases).
    - Investigating curriculum schedules based on difficulty estimates beyond pass rates (Section 5.1; Figure 6).
  - Robustness work: auditing LLM‑judge reward accuracy, calibrating the toggle’s reliability, and measuring leakage of reasoning in off‑mode.

- Practical applications
  - High‑throughput assistants that switch to deep reasoning only when needed (customer support, tutoring).
  - Scientific and engineering copilots where GPQA‑Diamond‑level reasoning matters; `LN‑Ultra` leads open models there (Table 5).
  - Coding agents leveraging strong LiveCodeBench performance with controllable verbosity.
  - Enterprise deployment on common H100 nodes, aided by throughput and memory improvements (Section 2.1; Figure 4).

- Next steps suggested by the paper’s findings
  - Extend RL to smaller models if compute‑efficient curricula/rewards can close the gap (Section 5).
  - Refresh SFT coding data for `LN‑Super` to match `LN‑Ultra`’s coding gains (Section 7.3).
  - Explore broader multi‑objective alignment to better balance IFEval compliance and conversational preference (Section 7.3; Section 6.2).
  - Generalize FFN Fusion and NAS policies to longer contexts and different transformer variants (e.g., hybrid architectures; Section 2).

Overall, Llama‑Nemotron provides a full, open blueprint—from hardware‑aware architecture search through reasoning RL—for building fast, controllable reasoning LLMs. The empirical results (Figures 2, 4–6; Tables 1, 3–6) substantiate both the efficiency gains and the reasoning quality, with `LN‑Ultra` setting a new open baseline on difficult scientific reasoning while remaining deployable on standard H100 nodes.
