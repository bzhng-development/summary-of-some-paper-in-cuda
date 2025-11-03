# Hermes 3 Technical Report

**ArXiv:** [2408.11857](https://arxiv.org/abs/2408.11857)
**Authors:** Ryan Teknium, Jeffrey Quesnelle, Chen Guang
**Institutions:** 

## 🎯 Pitch

Hermes 3 introduces a groundbreaking approach to language modeling by fine-tuning Meta’s Llama 3.1, achieving highly steerable and neutrally aligned models that excel in structured reasoning and tool use. This innovation is crucial for developers needing precise LLM control in diverse applications, enhancing transparency and reliability while redefining user interaction with AI systems.

---

## 1. Executive Summary
Hermes 3 is a family of instruction-following (“instruct”) and tool-using language models built by fine‑tuning Meta’s Llama 3.1 at three sizes (8B, 70B, 405B). It aims to be highly steerable and “neutrally aligned,” meaning it follows the given system and user instructions closely—including for role‑play, tools, and structured reasoning—while achieving strong open‑weight benchmark performance, especially at the 405B scale.

## 2. Context and Motivation
- Problem addressed
  - Instruct models are how most users interact with large language models (LLMs), but many existing models are:
    - Difficult to steer precisely across varied use cases (tools, RAG, structured reasoning).
    - Over‑constrained by refusals or safety heuristics that sometimes block legitimate requests.
    - Not consistently trained to output transparent, structured intermediate reasoning that helps with planning or tool use.
  - Hermes 3 targets these gaps by combining fine‑tuned instruction following, explicit tool use, and agentic/structured reasoning within one model family (Section 2, 2.1).

- Why this matters
  - Practical impact: Developers need LLMs that can plan, call tools with well‑formed arguments, cite sources in RAG, and maintain personas in long conversations (128K context via Llama 3.1) while remaining controllable by a system prompt (Section 2).
  - Research impact: It explores how training data and formatting (special tokens, schemas) change an LLM’s ability to perform multi‑step, interpretable reasoning and to adhere to directives (Section 2.1, Section 3).

- Prior approaches and limitations
  - Base/foundation models are general but not steerable for imperative tasks.
  - Many chat/instruct models rely on strong “helpful assistant” defaults and safety rules that sometimes override explicit user/system prompts (Introduction).
  - Tool use and RAG often require extra scaffolding outside the model or idiosyncratic prompting; consistent structure is not guaranteed.
  - Reasoning formats (scratchpads, plans) are not always part of the pretraining/fine‑tuning distribution.

- Positioning
  - Hermes 3 positions itself as a generalist, neutrally aligned instruct model trained to:
    - Obey the system prompt precisely (Section 2).
    - Use standardized tags for reasoning and tools (Section 2.1).
    - Support long‑context, multi‑turn role‑playing and structured workflows.
  - The work argues that safety/guardrails are best applied at the application/system level rather than by model‑level refusals: “For Hermes, there is no such thing as latent thoughtcrime.” (Introduction).

## 3. Technical Approach
Step‑by‑step overview of how Hermes 3 is built and how it works.

- Base models and context
  - Fine‑tunes Llama 3.1 “Herd of Models” at 8B, 70B, and 405B parameters, with 128K context at inference (Section 4.1). Fine‑tuning is done at 8K tokens per sequence for compute efficiency (Section 4.1 and 4.1 hardware notes).

- Instruct + agentic capabilities
  - Trained to follow a `system prompt` (a persistent meta‑instruction that sets behavior) very literally. Sensitivity is “particularly pronounced” at 405B; with an empty system prompt it does not default to a helpful persona (Figure 6, Section 2).
  - Introduces structured, interpretable reasoning and planning via reserved tokenizer tokens (Section 2.1):
    - `<SCRATCHPAD>`, `<REASONING>`, `<INNER_MONOLOGUE>`, `<PLAN>`, `<EXECUTION>`, `<REFLECTION>`, `<THINKING>`, `<SOLUTION>`, `<EXPLANATION>`, `<UNIT_TEST>`.
    - These tags serve as scaffolding for multi‑step problem solving (e.g., plan → execute → reflect) and for code tasks (Figure 9 shows a full agentic coding workflow with plan, schemas, diagram, reflection).
  - Tool use and RAG:
    - Tools follow the “Hermes Function Calling” standard: tool JSON Schemas in `<tools>`, invocations in `<tool_call>`, and tool outputs in `<tool_response>` (Section 2.1).
    - RAG responses cite sources inside `<co: doc_id></co>` tags; Figure 8 shows citations used to ground claims.
  - Reward modeling and judging:
    - Inspired by MT‑Bench, the models can assess answer quality across turns (Figure 7).

- Data mixture and curation (Section 3, Table 1)
  - Total ≈ 390M tokens for SFT; 69% are output/response tokens and 31% are input/instruction tokens:
    - General Instructions 60.6% (236M), Domain Expert 12.8% (50M), Math 6.7% (26M), Roleplaying 6.1% (24M), Coding 4.5% (18M), Tool Use/Agentic/RAG 4.3% (17M), Content Generation 3.0% (12M), Steering & Alignment 2.5% (10M).
  - Generation/curation emphasizes diversity (including role‑play and agentics) and strict instruction following:
    > Filtering included “removal of refusals and improperly formatted responses, elimination of conversations with missing or empty turns, and prioritization of conversations generated by the strongest models.” (Section 3)

- Supervised fine‑tuning (SFT) procedure (Section 4.1)
  - Optimization: AdamW with weight decay 0.01; cosine decay after 300 warmup steps; peak LR chosen via sweep (Figure 2), selecting 7×10⁻⁶ for 8B/70B; 3.5×10⁻⁶ for 405B (half of smaller models’ LR).
  - Labeling strategy:
    - Uses an “ignore index” for all tokens in the instruction and tool output sections; the loss focuses on response/tool‑use generation only (Section 4.1).
  - Efficient sample packing:
    - Heterogeneous conversation lengths are packed into 8K sequences using FlashAttention‑2’s variable‑length attention to prevent cross‑attention across samples (Figure 3). Achieves 96% packing efficiency (4% padding).
  - Checkpoint selection:
    - Epoch chosen using a min–max normalized average over GPT4All tasks (ARC‑E/C, BoolQ, HellaSwag, OpenBookQA, PIQA, WinoGrande), AGIEval, IFEval, and MT‑Bench (Section 4.1). Table 2 shows how 70B behaves across epochs.

- Hardware and scaling (Section 4.1, Table 3)
  - 8B/70B trained on six HGX4 nodes (8× H100 each) with PyTorch FSDP.
  - 405B required ≥7 HGX nodes with CPU parameter offloading (≈45% slowdown). Final run: 16 HGX nodes; effective batch size 128; training time 2086 GPU hours (Table 3). This highlights memory limits at 405B; future runs likely need tensor parallelism in addition to data parallelism.

- Preference optimization (DPO) phase (Section 4.2)
  - Trains a LoRA adapter instead of full model to avoid in‑memory duplication of “policy” and “reference” models.
    - LoRA: low‑rank adaptation that injects small trainable matrices into linear layers (here r=32, α=16, dropout 0.05 on all linear layers).
  - Optimizer: RMSProp, peak LR 3×10⁻⁶ with linear decay; NEFTune (α=5) adds small noise to embedding vectors during training to improve robustness.
  - Effect: modest improvement for 8B (Table 4), negligible for 70B/405B; therefore final larger models use SFT‑only checkpoints.

- Inference quantization for 405B evaluations (Section 5)
  - FP8 weight quantization with llm‑compressor for vLLM; round‑to‑nearest weights, channelwise activations, per‑token scales.

Definitions of less‑standard terms used above:
- `system prompt`: a persistent instruction that frames how the model should interpret all subsequent user inputs (e.g., “speak like Shakespeare,” Figure 1).
- `RAG` (Retrieval‑Augmented Generation): the model receives retrieved documents and must ground its answer in them, citing sources.
- `DPO` (Direct Preference Optimization): a pairwise preference training method that pushes the model to prefer “chosen” over “rejected” responses without training a separate reward model.
- `LoRA`: technique for parameter‑efficient fine‑tuning by learning low‑rank updates to large layers.
- `NEFTune`: adds controlled noise during instruction tuning to improve generalization.
- `FlashAttention‑2 variable‑length`: an attention kernel that supports packed sequences without cross‑sample attention and with better GPU utilization.
- `FSDP`: Fully Sharded Data Parallel training that shards model parameters and optimizer states across GPUs.
- `FP8 quantization`: 8‑bit floating point weights/activations at inference to reduce memory and increase throughput.

## 4. Key Insights and Innovations
- Highly steerable, “neutral” instruction following
  - Distinctive stance: the model is trained to follow the system prompt and user instructions “exactly and neutrally,” minimizing moralistic refusals (Introduction, Section 2).
  - Why it matters: increases applicability in role‑play, simulation, and developer workflows where system prompts define behavior. Figure 6 demonstrates that the 405B model does not default to “helpful assistant” without a system prompt.

- Structured, interpretable agentic scaffolds built into training
  - Use of reserved tokens for explicit reasoning phases—`<PLAN>`, `<EXECUTION>`, `<REFLECTION>`, etc. (Section 2.1)—is a notable integration of agentic process into the SFT data distribution.
  - Significance: improves transparency and reliability for multi‑step tasks and tool use; examples include code planning with Pydantic schemas and UML/mermaid diagrams (Figure 9) and multi‑turn judging (Figure 7).

- Unified tool use and RAG conventions
  - Standardized, schema‑driven function calling via `<tools>`, `<tool_call>`, `<tool_response>` and explicit RAG citations via `<co: doc_id></co>` (Section 2.1; Figures 8–9).
  - Significance: lowers the burden on application scaffolding; makes agent executions auditable and easier to debug.

- Training efficiency choices for heterogenous instruction data
  - Ignore‑label strategy to focus loss on desired outputs (Section 4.1) and efficient variable‑length sample packing with 96% packing efficiency (Figure 3).
  - Significance: maintains throughput and avoids cross‑sample leakage while handling diverse conversation lengths.

Incremental versus fundamental:
- Incremental: optimizer schedules, LR sweep (Figure 2), hardware configurations, and ignore‑label masking are engineering best practices.
- More fundamental: the systematic use of structured tags for agentic reasoning and standardized tool/RAG schemas during SFT, coupled with neutrality to system prompts, meaningfully shifts how an instruct model can be controlled and audited.

## 5. Experimental Analysis
- Evaluation methodology (Section 4.1 and 5)
  - During SFT: checkpoints selected via a combined score over:
    - GPT4All set (ARC‑E/C, BoolQ, HellaSwag, OpenBookQA, PIQA, WinoGrande), AGIEval, IFEval, and MT‑Bench (Section 4.1, Table 2).
  - Final evaluations (Table 5): includes AGIEval, ARC‑C/E, BoolQ, BBH, GPQA, HellaSwag, IFEval, MATH Level 5, MMLU, MMLU‑PRO, MT‑Bench, MuSR, OpenBookQA, PiQA, TruthfulQA, Winogrande. 405B evaluated under FP8 quantization (Section 5).

- Key quantitative results (Table 5; numbers below are Hermes 3 vs Llama 3.1 Instruct at the same size)
  - 405B:
    - Wins: AGIEval 61.84 vs 58.60; ARC‑C 69.45 vs 66.04; HellaSwag 90.19 vs 88.34; TruthfulQA 65.57 vs 64.83; PiQA 85.96 vs 84.93; MuSR 48.26 vs 47.58; OpenBookQA 48.80 vs 48.60.
    - Mixed/Losses: MMLU 85.02 vs 86.14; MATH L5 30.85 vs 35.98; IFEval 84.87 vs 87.09; BBH 75.37 vs 76.25; Winogrande 86.27 vs 86.82; GPQA 44.84 vs 42.66 (small win); BoolQ 88.93 vs 89.52 (slight loss).
    - Takeaway: Stronger on commonsense and reasoning‑heavy tasks like ARC‑C/E and HellaSwag, but lower on knowledge‑heavy/academic measures like MMLU and math (MATH L5).
  - 70B:
    - Notable wins: AGIEval 56.18 vs 48.26; HellaSwag 88.19 vs 86.42; MuSR 50.67 vs 47.08; TruthfulQA 63.29 vs 59.91.
    - Losses: IFEval 81.21 vs 87.25; MMLU 79.09 vs 82.27; MATH L5 20.80 vs 29.24; GPQA 37.67 vs 40.09.
  - 8B:
    - Hermes trades some knowledge benchmarks for better commonsense in places: HellaSwag 82.83 vs 80.01; TruthfulQA 58.69 vs 53.99; but lower on MMLU 64.79 vs 68.05 and IFEval 62.25 vs 80.15.

- Training phase diagnostics
  - LR sweep (Figure 2) guided SFT LR choice of 7×10⁻⁶ for 8B/70B.
  - Epoch comparison for 70B (Table 2): MT‑Bench peaked at epoch 3 (8.99), IFEval improved steadily through epoch 4 (86.61), but combined score favored epoch 3 overall due to drops elsewhere (e.g., GPT4All average fell at epoch 4).

- Preference optimization effect (Table 4; Figure 5)
  - 8B DPO yields small but consistent gains (e.g., TruthfulQA 56.43 → 58.69, IFEval 66.17 → 66.70).
  - For larger models, DPO impact was negligible; they ship SFT checkpoints (Section 4.2). This suggests the larger models already capture preference signals from SFT data or are saturated for the available DPO dataset.

- Do experiments support claims?
  - The “neutrally aligned” and steerable behavior is evidenced qualitatively:
    - Figure 6: without a system prompt, 405B does not default to the “helpful assistant.”
    - Figure 1: under a specific system prompt (speak like Shakespeare), it follows style and counts letters accurately.
    - Figure 8: RAG citation tags are used properly.
    - Figure 9: multi‑stage agentic coding with planning and schemas.
  - Performance claims are mixed but encouraging. The 405B wins on several reasoning‑style public benchmarks and is competitive with Llama 3.1 Instruct 405B on average (Table 5), though it trails on MMLU and MATH.

- Missing ablations and robustness checks
  - No ablation isolating the impact of reasoning tags, tool/RAG schemas, or data categories (Table 1) on downstream scores.
  - No explicit safety‑behavior or refusal‑rate analysis despite the neutrality goal.
  - No long‑context (≥32K/128K) benchmark evidence, even though the model supports 128K context (Section 2).

## 6. Limitations and Trade-offs
- Assumptions and scope
  - Assumes applications will supply an explicit and well‑crafted `system prompt`; behavior is highly sensitive to it (Section 2; Figure 6).
  - Relies on structured tags and tool schemas being used consistently by the application layer; deviation may reduce reliability.

- Data and training constraints
  - SFT corpus is relatively small for modern LLMs (≈390M tokens), heavy on synthetic/curated instructions (Section 3). This may limit knowledge breadth and contribute to lower MMLU/MATH scores (Table 5).
  - Fine‑tuning context length is 8K (Section 4.1), not the full 128K, which can reduce training‑time exposure to long‑context patterns.

- Computational constraints
  - 405B requires large multi‑node clusters; CPU parameter offloading caused ≈45% slowdown (Section 4.1). Future runs need higher‑dimensional parallelism (tensor parallel) to scale efficiently.

- Mixed benchmark profile
  - Strong on commonsense/reasoning (ARC‑C/E, HellaSwag, TruthfulQA), weaker on formal knowledge/math (MMLU, MATH L5) at 405B and 70B (Table 5). IFEval (strict instruction‑following) also trails Llama 3.1 Instruct at 70B/405B.

- Safety and societal considerations
  - The neutrality stance reduces refusals and shifts responsibility to the system layer (Introduction). This increases flexibility but also places more burden on downstream safety controls; the paper does not quantify misuse safeguards or refusal behavior.

## 7. Implications and Future Directions
- How this changes the landscape
  - Demonstrates that tightly integrating agentic scaffolding (reasoning tags, tool schemas, explicit citations) into SFT can yield models that are easier to control, audit, and integrate into multi‑tool pipelines—without requiring extensive external prompt engineering.
  - Reinforces that alignment “style” (neutral vs safety‑heavy) meaningfully changes user experience and that high‑capacity models can be highly sensitive to the system prompt.

- Follow‑up research directions
  - Data ablations: quantify contributions of each data category (Table 1) and of each structured token family to downstream tasks.
  - Long‑context evaluation: benchmark at 32K–128K to validate conversation and retrieval claims, including tool traces over long sessions.
  - Preference optimization at scale: investigate why DPO helps 8B but not 70B/405B; explore pairwise data quality, better reference policies, or alternative preference methods.
  - Safety tooling at the system layer: develop and evaluate modular guardrails that complement the model’s neutrality (e.g., policy‑driven filters, trace auditing of `<tool_call>` and `<co>` citations).
  - Math/knowledge improvements: augment data or training (e.g., curriculum, specialized math solvents, code‑interpreter tools) to address MATH/MMLU gaps.

- Practical applications
  - Agentic assistants that plan and execute workflows with visible reasoning and tool traces (Figures 8–9).
  - Enterprise chatbots requiring strict persona control and citation‑grounded answers.
  - Code assistants that generate plans, schemas, tests, and explanations in a single, auditable output.
  - LLM‑as‑a‑judge or self‑review pipelines leveraging the built‑in evaluation and reflection formats (Figure 7).

Quoted anchors for key points:
- Neutral alignment and steerability emphasis:
  > “Crucially, our training data strongly encourages the model to follow the system and instruction prompts exactly and neutrally.” (Introduction)
- Structured tags for agentic reasoning:
  > “Utilizing the extra reserved tokens… making use of the <SCRATCHPAD>, <REASONING>, <INNER_MONOLOGUE>, <PLAN>, <EXECUTION>, <REFLECTION>, <THINKING>, <SOLUTION>, <EXPLANATION>, and <UNIT_TEST> tokens.” (Section 2.1)
- Tool and RAG conventions:
  > “Tools can be specified… <tools>… <tool_call>… <tool_response>…” and “the model has been trained to cite retrieval sources using the <co> tag.” (Section 2.1)
- Data scale and composition:
  > “The final dataset… approximately 390 million tokens… 270 million (69%) were output tokens… 120 million were input tokens.” (Section 3; Table 1)
- Training efficiency:
  > “Multiple samples are packed together… using… Flash Attention 2… selected a target sequence length of 8192… packing is achieved at a 96% efficiency.” (Section 4.1; Figure 3)
- Hardware limits for 405B:
  > “Minimum system configuration… seven HGX nodes… CPU parameter offloading… 45% drop in training efficiency.” (Section 4.1)
- DPO design and impact:
  > “LoRA… r=32, α=16, dropout=0.05… RMSProp… 3×10⁻⁶… NEFTune α=5… DPO provided a moderate but positive impact on… 8B… negligible… for larger models.” (Section 4.2; Table 4)
- Final performance snapshot:
  > “Evaluations… FP8 quantization… Table 5: Final downstream task evaluations.” (Section 5)

Overall, Hermes 3 is best understood as a carefully fine‑tuned, open‑weight instruct family that prioritizes exact steerability and traceable agentic behavior. Its benchmark profile shows strong commonsense and reasoning with some trade‑offs in formal knowledge/math—setting a clear agenda for targeted future improvements.
