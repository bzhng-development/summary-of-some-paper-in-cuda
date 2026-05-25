# HERMES 3 TECHNICAL REPORT

**ArXiv:** [2408.11857](https://arxiv.org/abs/2408.11857)

## 🎯 Pitch

Hermes 3 introduces a family of openly available, highly steerable instruction-tuned large language models (8B, 70B, 405B parameters) built on Llama 3.1, distinguished by their neutral alignment to system prompts and robust agentic tool-use capabilities. By combining a curated, diverse training regimen with innovative token-level support for multi-step reasoning, structured outputs, retrieval-augmented generation, and tool-calling, Hermes 3 enables precise, reliable, and transparent control—meeting the needs of developers seeking flexible, enforceable workflows without the constraints or refusals imposed by commercial safety-aligned models. This opens new horizons for enterprise, research, and open-source communities by delivering state-of-the-art performance with unparalleled controllability and interpretability.

---

## 1. Executive Summary
Hermes 3 is a family of open‑weight, instruction‑tuned models (8B, 70B, 405B parameters) built on Llama 3.1 that emphasizes strong steerability to system prompts, transparent tool use, and “agentic” reasoning features. The largest model, `Hermes 3 405B`, achieves state‑of‑the‑art results among open models on several public benchmarks, while introducing a training and data recipe that makes the model follow instructions neutrally and adhere closely to requested formats.

## 2. Context and Motivation
- Problem addressed
  - Base (foundation) LLMs are trained on broad text and can be hard to steer reliably; instruction‑tuned (“instruct”) variants solve this in part but often embed safety policies that overrule user requests. Hermes 3 targets two gaps:
    - High steerability to arbitrary system prompts without baked‑in refusals.
    - Built‑in support for tool use, retrieval‑augmented generation (RAG), and interpretable multi‑step reasoning with structured tags.
  - Evidence in the paper: Section 1 and 2 emphasize steerability, neutral alignment, and agentic capabilities; Figure 1 shows precise format control (“Shakespeare prose”); Figures 7–9 demonstrate evaluation, RAG citations, and agentic coding.

- Why it matters
  - Real‑world: Developers want assistants that can be strictly steered (e.g., enterprise‑specific compliance style guides) and can transparently call tools and cite sources. Section 2.1 highlights structured outputs and tool/RAG integration.
  - Research: Tests whether careful instruction tuning plus curated synthetic data and special tokens can deliver both performance and controllability without heavy RL or restrictive safety layers.

- Prior approaches and shortcomings
  - Closed models (e.g., popular commercial chatbots) are steerable but often refuse requests “on moral grounds” and are not openly adjustable (Section 1).
  - Open instruct models exist (e.g., Llama 3.1 Instruct), but lack Hermes 3’s combination of neutral instruction‑following, rich agentic tags, and explicit tool/RAG training with citations (Section 2.1).

- Positioning
  - Hermes 3 is presented as a neutrally aligned, generalist instruct model suite with transparent agent capabilities, trained with a compact but carefully curated mixture (390M tokens; Table 1), and a two‑stage optimization (SFT then DPO; Section 4).

## 3. Technical Approach
Step‑by‑step overview of how Hermes 3 is built and why these choices were made.

- Base models and context
  - Built by fine‑tuning Llama 3.1 (“Herd of Models”) at 8B, 70B, and 405B parameters with a 128K token context window (Section 4.1).

- Data mixture and filtering (what they train on and why)
  - Mixture totals ~390M tokens with emphasis on instruction following and diverse domains (Table 1):
    - 60.6% General Instructions; plus Domain Expert, Math, Roleplaying, Coding, Tool/Agentic/RAG, Content Generation, and Steering/Alignment.
  - Generation and curation
    - Combines existing high‑quality sources and domain‑specific synthetic instructions inspired by Evol‑Instruct (Section 3).
  - Filtering pipeline
    - Enforces token length thresholds; removes refusals, misformatted responses, and incomplete turns; prioritizes outputs from stronger models (Section 3).
  - Design rationale
    - The mix addresses weaknesses in older Hermes models (e.g., code/math/roleplay/agentics coverage) and trains the model to precisely follow system and instruction prompts (Section 3).

- Agentic tokens and structured reasoning (how “reasoning” is made explicit)
  - Uses reserved tokens like `<SCRATCHPAD>`, `<REASONING>`, `<PLAN>`, `<INNER_MONOLOGUE>`, `<SOLUTION>`, `<EXPLANATION>`, `<UNIT_TEST>`, etc., to encourage interpretable intermediate steps (Section 2.1).
  - Figure 9 shows these tags orchestrating a multi‑stage coding plan (restatement → reasoning → plan → schemas → diagram → reflection → solution → explanation → unit tests).

- Tool use and RAG (how the model interacts with external systems)
  - Hermes Function Calling standard: tool definitions are JSON schemas inside `<tools>`, and invocations appear in `<tool_call>` / `<tool_response>` (Section 2.1; footnote link).
  - RAG citation protocol: sources are cited inline with `<co: doc_id></co>` (Figure 8). This explicit schema trains the model to expose provenance.

- Supervised fine‑tuning (SFT) details (how training is optimized)
  - Optimizer/schedule: AdamW, weight decay 0.01, peak LR 7e‑6 for 8B/70B with cosine decay and 300 warmup steps (Section 4.1; Figure 2 shows a learning‑rate sweep that picks 7e‑6).
  - Label masking: training loss ignores tokens in the instruction and tool output sections so the optimizer focuses on assistant responses and tool use (Section 4.1).
  - Sample packing: heterogeneous conversations are packed into 8K‑token sequences using FlashAttention‑2’s variable‑length function to avoid cross‑sample attention while keeping 96% packing efficiency (very little padding) (Section 4.1; Figure 3).
  - Checkpoint selection: for each size, select the epoch with best normalized average across GPT4All benchmarks, AGIEval, IFEval, and MT‑Bench (Section 4.1). Table 2 illustrates 70B metrics across 4 epochs; epoch 3 is chosen.

- DPO preference optimization (what it is and how used)
  - DPO (Direct Preference Optimization) fine‑tunes a reward preference without explicit RL. To save memory at large scale, they train a LoRA adapter instead of full‑model DPO (Section 4.2).
  - LoRA setup: rank r=32, α=16, dropout 0.05 on all linear layers (Section 4.2).
  - Optimizer: RMSProp with peak LR 3e‑6 and linear decay; NEFTune (noisy embeddings) with α=5 for regularization (Section 4.2).
  - Outcome: modest but positive gains on 8B (Table 4), negligible at larger scales, so they keep SFT checkpoints for 70B/405B.

- Compute and scaling mechanics (how they trained the largest model)
  - 8B/70B: trained on six HGX4 nodes (each HGX has 8× H100 SXM5 GPUs) with PyTorch FSDP (fully sharded data parallelism) (Section 4.1).
  - 405B: requires ≥7 HGX nodes even at 8K context with CPU parameter offloading; offloading slows training by ~45% (Section 4.1). Final run uses 16 HGX nodes with effective batch size 128 and LR 3.5e‑6 (half of smaller models), selected by trials (Section 4.1; Table 3 summarizes batch sizes, LR, GPU hours; Figure 4 shows loss curves).

- Inference/quantization for evaluation
  - 405B evaluations use FP8 weight quantization with vLLM’s llm‑compressor (round‑to‑nearest, channelwise activations, per‑token scales) to fit inference efficiently (Section 5).

## 4. Key Insights and Innovations
- Neutral, high‑fidelity instruction following through system‑prompt sensitivity
  - Hermes 3 is trained to “fit inside” the provided system prompt and follow it strictly, not revert to a generic “helpful assistant.” Figure 1 shows style‑constrained counting in Shakespearean prose; Figure 6 shows that an empty system prompt can yield a non‑assistant persona. This is a different alignment target than refusal‑heavy assistants and matters when enterprises need exact persona control.

- Interpretable agentic scaffolding with reserved tags
  - The model is explicitly trained to use XML‑like tags for multi‑step problem solving, internal monologue, planning, diagrams, and unit tests (Section 2.1; Figure 9). Unlike generic chain‑of‑thought, these tags establish a structured contract for intermediate outputs (e.g., `<PLAN>`, `<EXPLANATION>`), making agent behaviors more auditable.

- First‑class tool use and RAG with explicit citations
  - Hermes Function Calling trains the model to read tool JSON schemas and to emit `<tool_call>`/`<tool_response>` blocks; RAG answers cite sources inline with `<co:doc_id>` (Section 2.1; Figure 8). This explicit formatting helps downstream systems parse tool invocations and verify provenance, reducing “black‑box” behavior.

- Efficient SFT with variable‑length sample packing and targeted loss
  - Ignoring instruction/tool tokens in the loss focuses learning on assistant responses and tool usage (Section 4.1). FlashAttention‑2 varlen packing achieves 96% token efficiency at an 8K training context (Figure 3), a practical advance for mixed‑length dialog corpora.

- Transparent training decisions and compute scaling for 405B
  - The report details minimum viable hardware, the need for CPU offload, and resulting 45% slowdown (Section 4.1). It also documents epoch selection via a cross‑benchmark criterion (Table 2), LR sweep (Figure 2), and quantified training curves (Figure 4), which is valuable for reproducibility.

## 5. Experimental Analysis
- Evaluation methodology
  - Benchmarks include GPT4All suite (ARC‑E/C, BoolQ, HellaSwag, OpenBookQA, PIQA, WinoGrande) plus AGIEval, IFEval, MT‑Bench during training/selection (Section 4.1), and broader final evaluations: BBH, MATH Lvl 5, GPQA, MuSR, MMLU, MMLU‑PRO (Section 5).
  - 405B evaluations are run with FP8 quantization (Section 5). Epoch selection uses a min‑max normalized average across GPT4All, AGIEval, IFEval, MT‑Bench (Section 4.1; Table 2).

- Main quantitative results (Table 5, “Final downstream task evaluations”)
  - 405B vs Llama 3.1 Instruct 405B
    - Clear gains on several reasoning and commonsense tasks:
      - AGIEval: 61.84 vs 58.60
      - ARC‑C: 69.45 vs 66.04
      - HellaSwag (10‑shot): 90.19 vs 88.34
      - GPQA: 44.84 vs 42.66
      - TruthfulQA (MC2 0‑shot): 65.57 vs 64.83
    - Mixed or lower on others:
      - IFEval (Strict): 84.87 vs 87.09
      - MATH Lvl 5 (4‑shot): 30.85 vs 35.98
      - MMLU (5‑shot): 85.02 vs 86.14
      - MMLU‑PRO (5‑shot): 54.14 vs 63.51
      - MT‑Bench: 8.93 vs 9.17
  - 70B vs Llama 3.1 Instruct 70B
    - Improvements on reasoning/commonsense:
      - AGIEval: 56.18 vs 48.26
      - ARC‑C: 65.53 vs 63.40
      - HellaSwag: 88.19 vs 86.42
      - TruthfulQA: 63.29 vs 59.91
    - Lower on instruction‑following and knowledge exams:
      - IFEval: 81.21 vs 87.25
      - MATH Lvl 5: 20.80 vs 29.24
      - MMLU: 79.09 vs 82.27
      - MMLU‑PRO: 47.24 vs 52.94
  - 8B vs Llama 3.1 Instruct 8B
    - Some improvements (e.g., ARC‑C: 58.11 vs 55.12; HellaSwag: 82.83 vs 80.01; OpenBookQA: 47.80 vs 43.20), but mixed elsewhere.

- Support for claims
  - The report’s central claim—strong steerability and competitive open‑weight performance—is supported by:
    - Demonstrations of formatting control and persona sensitivity (Figures 1 and 6).
    - Tool/RAG transparency (Figure 8) and agentic planning (Figure 9).
    - Quantitative gains on reasoning‑heavy tasks (AGIEval, ARC‑C, HellaSwag) across sizes (Table 5).
  - Caveats:
    - Instruction‑following strictness (IFEval) and knowledge/maths exams (MMLU/MMLU‑PRO/MATH) sometimes lag the Llama 3.1 Instruct baselines (Table 5). This suggests a trade‑off between neutral steerability and raw exam‑style knowledge performance.

- Ablations and diagnostic studies
  - Learning‑rate sweep justifies 7e‑6 for 8B trials (Figure 2).
  - Epoch‑selection analysis for 70B shows how metrics evolve (Table 2).
  - DPO ablation on 8B:
    - Small overall improvements after DPO: e.g., TruthfulQA 56.43 → 58.69; AGIEval 40.17 → 41.26; GPT4All 72.03 → 72.30; with a slight dip on Big‑Bench (44.57 → 43.04) (Table 4).
    - Reward margins curve shown in Figure 5.
  - Training loss curves across sizes in Figure 4.

- Qualitative evidence
  - Figure 7 (MT‑Bench‑style judgment) shows the model can evaluate multi‑turn answers and produce a calibrated score with explanation.
  - Figure 8 shows strict source citation behavior in RAG.
  - Figure 9 shows end‑to‑end agentic coding output using the tag scaffolding.

## 6. Limitations and Trade-offs
- Performance trade‑offs across benchmarks
  - While Hermes 3 excels on several reasoning benchmarks, it is behind Llama 3.1 Instruct on exam‑style knowledge and math at multiple scales (Table 5: MMLU, MMLU‑PRO, MATH). This indicates that the training mixture or alignment target may emphasize format/agentic behavior and general reasoning over textbook exam performance.

- Instruction‑following metrics not always superior
  - On IFEval (Strict), Hermes 3 trails Llama 3.1 Instruct at 405B and 70B (Table 5), despite an emphasis on strict system prompt adherence elsewhere. This suggests that instruction‑style preferences learned by Hermes 3 (e.g., use of tags, neutrality) do not perfectly match IFEval’s scoring rules.

- Heavy reliance on synthetic and curated instruction data
  - The SFT dataset is relatively small for such large models (390M tokens; Table 1) and includes substantial synthetic data (Section 3). Synthetic coverage is powerful for controllability but risks narrowness or artifacts that don’t fully generalize to real‑world requests.

- Computational constraints at 405B
  - Training the 405B model required CPU parameter offloading and ≥7 HGX nodes, incurring ~45% slowdown (Section 4.1). This raises the bar for replication and further tuning; they also note future need for higher‑dimensional parallelism (tensor + data) to avoid excessively large batch sizes (Section 4.1).

- Quantization during evaluation
  - 405B results are under FP8 quantization (Section 5). While efficient, this can sometimes change accuracy slightly relative to full‑precision inference, complicating direct comparisons.

- Safety/neutrality implications
  - The model is intentionally “neutrally aligned” and resists moral refusals (Section 1). This design relies on external system‑level guardrails rather than built‑in model refusals. It’s a deliberate choice but shifts responsibility to application designers.

## 7. Implications and Future Directions
- Field impact
  - Hermes 3 demonstrates that open instruct models can combine strong steerability, transparent agent tooling, and competitive benchmark performance, advancing the feasibility of open, auditable AI agents. The explicit tagging scheme for reasoning, planning, and citations provides a concrete interface for building trustworthy agent systems.

- Practical applications
  - Enterprise assistants with strict persona/style control (Figure 1).
  - Tool‑driven agents that must expose calls/results for auditing, e.g., finance, healthcare, legal domains (Section 2.1).
  - Retrieval‑centric assistants that must cite sources consistently (Figure 8).
  - Developer copilots that output plans, diagrams, and tests alongside code (Figure 9).

- Suggested follow‑up research
  - Improving exam‑style knowledge and math without sacrificing steerability—e.g., targeted SFT or curriculum augmentation for MMLU/MATH tasks.
  - Stronger instruction‑following generalization on IFEval while preserving neutrality.
  - Scaling the DPO phase for larger models (beyond LoRA) or exploring alternative preference‑learning methods that complement the agentic tag regime.
  - Training without FP8 constraints for evaluation fairness, or systematically quantifying FP8 impacts.
  - More extensive robustness checks: long‑context stress tests (the base is 128K), out‑of‑domain tool schemas, noisy retrieval corpora, and adversarial prompts targeting the tag scaffolding.

- System design recommendations
  - Given the model’s neutrality and sensitivity to system prompts (Figures 1 and 6), production systems should:
    - Provide explicit, unambiguous system prompts that encode safety/compliance rules.
    - Validate/tag outputs programmatically by parsing the XML‑like tags.
    - Log and verify tool calls and citations, using the structured `<tool_call>`/`<co>` formats to enable audit trails.

> Overall, Hermes 3 shows that careful instruction tuning with structured agentic tags, tool/RAG schemas, and a curated instruction dataset can produce highly steerable, open models that outperform open baselines on several reasoning tasks (Table 5), while revealing trade‑offs on instruction‑following benchmarks like IFEval and knowledge exams like MMLU/MATH.
