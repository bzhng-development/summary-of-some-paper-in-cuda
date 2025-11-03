# Attention Heads of Large Language Models: A Survey

**ArXiv:** [2409.03752](https://arxiv.org/abs/2409.03752)
**Authors:** Zifan Zheng, Yezhaohui Wang, Yuxin Huang, Shichao Song, Mingchuan Yang, Bo Tang, Feiyu Xiong, Zhiyu Li
**Institutions:** 

## 🎯 Pitch

This paper offers a groundbreaking framework for understanding how attention heads in decoder-only LLMs facilitate reasoning by systematizing their roles into a cognitively-inspired, four-stage process: Knowledge Recalling, In-Context Identification, Latent Reasoning, and Expression Preparation. By consolidating scattered mechanistic insights into a cohesive model, it not only clarifies how these heads collaborate but also provides actionable methodologies to enhance LLM performance, offering both practical solutions for reducing errors and boosting interpretability in AI research.

---

## 1. Executive Summary
This survey systematizes how attention heads inside decoder‑only large language models (LLMs) contribute to reasoning. It proposes a four‑stage, cognitively inspired framework—Knowledge Recalling (KR), In‑Context Identification (ICI), Latent Reasoning (LR), and Expression Preparation (EP)—and maps dozens of known “special heads” and the methods used to discover them onto this framework (Figure 7; Sections 4–5). The work matters because it translates scattered mechanistic findings into an operational picture of how heads collaborate across layers (Figures 8–9), and it clarifies experimental toolkits and benchmarks to study them (Section 6).

## 2. Context and Motivation
- Problem addressed
  - LLMs achieve strong performance yet remain largely black boxes. Understanding their internal reasoning “bottlenecks”—especially the role of `attention heads`—is a central challenge (Introduction; Figure 1; Section 1).
  - Prior interpretability surveys emphasized early Transformer variants or techniques, often predating modern decoder‑only LLMs and emergent capabilities (Section 3.3). This leaves a gap in head‑level mechanism understanding for current LLMs.

- Why it matters
  - Practical impact: Mechanistic insight enables targeted interventions (e.g., reduce hallucinations, improve truthfulness, strengthen long‑context retrieval). Section 1 and Section 4.4 describe heads like `Truthfulness`, `Accuracy`, and `Retrieval` that can be manipulated to improve behavior.
  - Theoretical significance: Heads are natural “units” of inference in Transformers. Clarifying how they read from and write to shared `residual streams` (Figure 4; Section 3.2.1) provides a coherent account of information flow through layers.

- Where prior approaches fall short
  - Older surveys focus on attention variants or encoder models (e.g., BERT) rather than decoder‑only LLMs with emergent behaviors (Section 3.3).
  - Many mechanistic studies analyze one task or one head family in isolation; cross‑task generality and inter‑head collaboration are underexplored (Section 8.1).

- Positioning
  - Scope: decoder‑only LLMs, attention heads as the focus (Section 2), with FFN mechanisms summarized later (Section 7.1).
  - Contribution: a unifying framework for head functions across tasks and layers, plus a clear taxonomy of experimental methods (Sections 4–5) and evaluation resources (Section 6).

## 3. Technical Approach
This is a survey that builds a structured model of how attention heads operate, grounded in the Transformer’s math and a cognitive analogy.

Step 1 — Formalize how a head works in the model
- Layer computation (Section 3.1; Equations 1–2):
  - Each layer has two residual blocks. The first adds the multi‑head attention output to the input; the second adds the FFN output:  
    `X_{ℓ,1} = X_{ℓ,0} + Σ_h Attn_ℓ^h(X_{ℓ,0})` (Eq. 1) and `X_{ℓ+1,0} = X_{ℓ,1} + FFN_ℓ(X_{ℓ,1})` (Eq. 2).
- Single‑head computation (Section 3.1; Eq. 3–4):
  - A head computes attention weights via queries and keys and writes values through an output matrix:  
    `Attn_ℓ^h(X) = softmax(Q K^T) V O` (Eq. 3).
  - Expanded view (Eq. 4):  
    `W_Q W_K^T` is the `QK matrix` (controls “who to read from”), and `W_V O` is the `OV matrix` (controls “what to write back”).  
  - Key concept: heads “read” from the shared `residual stream` via QK and “write” back via OV (Figure 4; Section 3.2.1).

Terminology used throughout (Section 3.2):
- `Residual stream`: the running sum of embeddings and previous layer outputs at each position; it is the channel all heads and FFNs read from/write to (Figure 4).
- `Activation patching`: replace an intermediate activation with one from a different run to test causal contributions (Figure 5; Section 3.2.2).
- `Ablation`: remove or zero out components or activations to measure effect (Section 3.2.2).
- `Logit lens`: map an intermediate vector through the unembedding to see which tokens it favors (Section 3.2.2).

Step 2 — A four‑stage cognitive framework for head functions (Section 4; Figures 6–7)
- Motivated by cognitive models (OAR and ACT‑R; Section 4.1), the survey maps head roles to four stages:
  1) `Knowledge Recalling (KR)` — retrieve parametric or experience‑like knowledge to seed reasoning.
  2) `In‑Context Identification (ICI)` — locate and summarize structural, syntactic, and semantic information in the prompt.
  3) `Latent Reasoning (LR)` — integrate and transform information to infer answers or intermediate states.
  4) `Expression Preparation (EP)` — aggregate, amplify, and align internal results with surface tokens for output.
- Layer‑wise pattern: shallow layers skew toward KR/ICI, mid‑layers to ICI/LR, deep layers to LR/EP (Figure 8), with exceptions on specific tasks.

Step 3 — Map concrete head types into the framework (Section 4; Figure 7; Tables 2–3)
- KR examples (Section 4.2; Table 2):
  - `Memory Head` retrieves relevant parametric knowledge triggered by enriched entity features written by shallow FFNs (Section 4.2).
  - Task‑specific biases: `Constant Head` and `Single Letter Head` in MCQA initialize attention over choice letters (Section 4.2); `Negative Head` encodes a prior toward “No”-like outputs in binary decision tasks (Section 4.2).
- ICI examples (Section 4.3):
  - Overall structure: `Previous Head`, `Positional Head`, `Rare Words Head`, `Duplicate Head` (Section 4.3.1).
  - Long context: `Retrieval Head` and `Global Retrieval Head` locate target tokens in long sequences (“needle‑in‑a‑haystack”) (Section 4.3.1).
  - Syntax: `Subword Merge Head` merges split subwords; `Syntactic Head` marks subjects/objects/modifiers; `Name/Letter Mover Heads` copy key items to the `[END]` position; `Negative Name Mover Head` prevents unwanted copying (Section 4.3.2).
  - Semantics: `Context Head`; `Content Gatherer Head` moves answer‑relevant text to `[END]`; `Sentiment Summarizer` writes sentiment near `[SUM]`; `Subject/Relation Heads` encode attributes; `Semantic Induction Head` captures semantic relations (Section 4.3.3).
- LR examples (Section 4.4):
  - In‑context learning:
    - `Summary Reader` reads `[SUM]` to infer a sentiment label (Section 4.4.1).
    - `Function Vector`: mid‑layer head outputs combine into a vector that encodes the task mapping (Section 4.4.1).
    - `Induction Heads`: detect patterns like “... A B ... A → predict B” by matching “previous token” features supplied by `Previous Head` (Section 4.4.1).
    - `In‑context Head`: compares `[END]` features with label features, weighting labels by similarity (Section 4.4.1).
  - Effectiveness:
    - `Truthfulness`, `Accuracy`, `Consistency` heads correlate with desirable behaviors and can be steered; `Vulnerable Head` overreacts to spurious inputs (Section 4.4.2).
  - Task‑specific:
    - `Correct Letter Head` maps answer text to the right choice letter in MCQA; `Iteration Head` updates an iterative state; `Successor Head` increments ordinals; `Inhibition Head` suppresses misleading candidates (Section 4.4.3).
- EP examples (Section 4.5; Table 3):
  - `Mixed Head` aggregates outputs from earlier heads (e.g., Subject/Relation/Induction) into a concise representation for unembedding.
  - Signal amplification: `Amplification Head` and `Correct Head` boost logits of the correct choice near `[END]`.
  - Instruction alignment: `Coherence Head` maintains language consistency; `Faithfulness Head` aligns Chain‑of‑Thought with actual internal computation.

Step 4 — Explain collaboration (“circuits”) across heads (Section 4.6; Figure 9)
- IOI example: a multi‑stage circuit integrates KR (Subject/Relation trigger “human name”), ICI (Duplicate and Name Mover heads spotlight “John/Mary” at `[END]`), LR (Induction and Previous Heads aggregate evidence; `Inhibition Head` suppresses “John”), and EP (`Amplification Head` boosts “Mary”)—see Figure 9 for the full pathway.
- Additional examples:
  - Parity/iteration (Eq. 5): a `Mover Head` forwards the `[EOI]` index to `[END]`; an `Iteration Head` queries “are you position t?” and updates the state (Section 4.6).

Step 5 — Methods to discover and validate head functions (Section 5; Figure 10)
- Modeling‑Free (no new models) (Table 4):
  - `Modification‑Based`:
    - Directional addition/subtraction assume linear feature directions (e.g., “sentiment direction”) and add/remove them at specific heads to measure output effects (Section 5.1).
  - `Replacement‑Based`:
    - Zero/mean ablation replace a head’s activation with zeros or dataset‑means.
    - Naïve activation patching swaps activations from a “corrupted” prompt (e.g., swapping the name “Mary”→“Alice”) at specific heads to test causal roles (Figure 5; Section 5.1).
- Modeling‑Required (new models or metrics) (Table 5):
  - `Training‑Required`:
    - Probing: train a classifier on head activations to detect functional heads (Section 5.2).
    - Simplified model training: learn a small attention‑only or two‑layer model on synthetic tasks to study head formation (Section 5.2).
  - `Training‑Free`:
    - Scoring functions: `RetrievalScore_ℓ^h` (Eq. 6) measures how often a head assigns top attention to the true target; `NAS_ℓ^h` (Eq. 7) quantifies negative bias by contrasting attention to “Yes/No” tokens (Section 5.2).
    - Information Flow Graph (IFG): build a token‑level graph of information transfer and prune to the most impactful edges to reveal routes (Section 5.2).

Step 6 — Evaluation resources (Section 6)
- Mechanism exploration benchmarks (Table 6) simplify tasks to token‑level readouts (e.g., IOI, sentiment templates in Figure 11, induction datasets).
- Common evaluation (Table 7) tests whether manipulating heads improves broader capabilities (e.g., TruthfulQA, MMLU, long‑context retrieval).

## 4. Key Insights and Innovations
- A cognitively grounded, four‑stage framework for head functions (Section 4; Figure 6)
  - Novelty: Instead of listing heads piecemeal, the survey maps them to KR/ICI/LR/EP and explains where they tend to reside in depth (Figure 8).
  - Significance: Clarifies “who does what, when” during inference, making inter‑head roles and transitions explicit.

- A comprehensive taxonomy of special heads with concrete mechanisms (Figure 7; Sections 4.2–4.5; Tables 2–3)
  - Novelty: Brings together disparate findings—e.g., `Induction`, `Mover`, `Retrieval`, `Inhibition`, `Amplification`, `Truthfulness` heads—under one operational vocabulary tied to QK/OV roles.
  - Significance: Offers ready‑made handles for targeted interventions (e.g., suppress `Vulnerable Head`, amplify `Correct Letter Head`), and connects many works to common primitives (read via QK, write via OV).

- A unifying view of head collaboration as circuits (Section 4.6; Figure 9)
  - Novelty: Shows end‑to‑end flows across stages on concrete tasks (IOI, parity), not just single‑head anecdotes.
  - Significance: Encourages circuit‑level design and evaluation—e.g., combining `Name Mover`, `Induction`, and `Inhibition` heads to steer outputs.

- Methodological reframing of interpretability toolkits (Section 5; Figure 10)
  - Novelty: Re‑organizes techniques by modeling dependency (Modeling‑Free vs Modeling‑Required) and by how activations are altered (Modification vs Replacement).
  - Significance: Helps practitioners pick the right tool for the causal question (e.g., linear feature tests via directional addition; logical elimination via zero ablation; route discovery via IFG).

These are foundational rather than incremental: they consolidate mechanisms into a functional theory of head roles and provide a methodological map to probe and edit them.

## 5. Experimental Analysis
Because this is a survey, it synthesizes experimental designs and results rather than presenting a single new empirical study. Key elements:

- Evaluation methodology (Section 6)
  - Mechanism‑level datasets (Table 6):
    - Sentiment templates `ToyMovieReview` and `ToyMoodStory` (Figure 11) to isolate sentiment features and test `Sentiment Summarizer` and `Summary Reader`.
    - IOI to examine `Name Mover`, `Induction`, and `Inhibition` circuits (Figure 9).
    - Induction/iteration/succession datasets to study `Induction`, `Iteration`, `Successor` heads.
    - World‑capital and LREl to probe factual recall (`Memory`/`Mixed` heads).
  - System‑level benchmarks (Table 7):
    - Knowledge/logic: MMLU, TruthfulQA, LogiQA, MQuAKE.
    - Sentiment: SST/SST‑2, ETHOS.
    - Long context: Needle‑in‑a‑Haystack.
    - Text comprehension: AG News, TriviaQA, AGENDA.

- Metrics and causal tests
  - `Logit lens` maps intermediate activations to token logits to quantify intervention effects (Section 3.2.2).
  - Direct/indirect/total effects when patching (Figure 5).
  - Head‑specific scores: `RetrievalScore` (Eq. 6) for long‑context retrieval ability; `NAS` (Eq. 7) for negative‑bias diagnosis.

- Representative findings the survey grounds in figures/equations
  - “Needle‑in‑a‑Haystack” ability is attributable to `Retrieval Heads`, made measurable by `RetrievalScore_ℓ^h` (Section 5.2, Eq. 6).
  - Binary decision bias is quantifiable via `NAS_ℓ^h`; high values indicate attention skew toward negative tokens (Section 5.2, Eq. 7).
  - Circuits for IOI integrate KR→ICI→LR→EP, with explicit head roles and dataflow (Figure 9).
  - Layer distribution aligns with KR→EP progression (Figure 8), but deep layers sometimes return to KR/ICI for specific tasks (Section 4.6).

- Do the experiments support the claims?
  - The survey consistently ties mechanisms to causal tools (activation patching, ablations) and to simplified tasks that expose token‑level effects (Section 5; Table 6). The use of direct/indirect/total effects (Figure 5) and logit‑lens readouts adds quantitative grounding.
  - However, as a survey, it does not present meta‑analyses or unified effect sizes across models/tasks; evidence is task‑ and model‑specific (explicitly noted in Section 8.1 on generalizability).

- Ablations and robustness checks
  - Replacement methods (zero/mean ablation) test necessity of a head (Table 4).
  - Directional addition/subtraction test linearity and feature causal potency (Section 5.1).
  - IFG route pruning tests whether the discovered circuit is sufficient to carry most of the effect (Section 5.2).

- Failure modes and mixed results
  - `Vulnerable Heads` can over‑attend to irrelevant forms, harming accuracy (Section 4.4.2).
  - `Negative Heads` can inject prior bias in binary tasks (Section 4.2), requiring correction (Eq. 7 gives a way to detect it).
  - Heads are not universally stable across models or tasks (Section 8.1).

## 6. Limitations and Trade-offs
- Assumptions
  - Many methods assume meaningful linear directions in activations (e.g., sentiment or truthfulness vectors) that can be added/subtracted (Section 5.1). This may not capture non‑linear interactions or feature entanglement in all contexts.
  - Circuit descriptions assume modularity of head roles and sparse pathways; real models may use overlapping or distributed mechanisms.

- Scope constraints
  - Focus on decoder‑only LLMs and attention heads (Section 2). FFN mechanisms are summarized but not the main emphasis (Section 7.1).
  - Many head discoveries rely on synthetic or templated tasks to cleanly expose mechanisms (Table 6), which may differ from open‑ended applications.

- Generalizability and transfer
  - Mechanisms validated on one model may not transfer to others; cross‑series reproducibility is underexplored (Section 8.1 “Lack of Mechanism Transferability”).
  - Circuits found for IOI or color‑object tasks are not yet shown to hold broadly across task families (“Lack of task generalizability”; Section 8.1).

- Collaboration coverage
  - While notable circuits are mapped (Figure 9), a comprehensive account of multi‑head collaboration across all layers and tasks remains open (Section 8.1).

- Theoretical underpinnings
  - Despite strong empirical tooling, formal guarantees or proofs of necessity/sufficiency for circuits are limited (“Absence of theoretical supports”; Section 8.1).

- Practical trade‑offs
  - Interventions (e.g., boosting `Amplification` heads) may improve one metric but risk overfitting to specific templates or reduce robustness elsewhere; the survey encourages careful evaluation (Section 6), but standardized trade‑off reporting is still rare.

## 7. Implications and Future Directions
- How this work changes the landscape
  - It reframes attention‑head interpretability from isolated observations to a staged theory of reasoning that aligns with model math (QK/OV read‑write; Eq. 4) and cognitive analogies (Figure 6). This helps researchers and practitioners discuss and target interventions at the right stage and depth (Figure 8).

- Enabled follow‑ups
  - Circuit‑level editing: Combine `Name Mover` + `Induction` + `Inhibition` + `Amplification` edits to steer IOI‑like phenomena (Figure 9).
  - Bias detection/correction: Use `NAS` (Eq. 7) to find and mitigate `Negative Heads` in safety‑critical binary decisions.
  - Long‑context optimization: Identify `Retrieval Heads` via `RetrievalScore` (Eq. 6) for KV‑cache compression or latency reduction while preserving retrieval (Section 4.3.1; Section 5.2).
  - Truthfulness and consistency: Probe for `Truthfulness`, `Accuracy`, `Consistency` heads and perform inference‑time intervention (Section 4.4.2).

- Research directions highlighted (Section 8.2)
  - Tackle complex tasks (open‑ended QA, math, tool use) to test whether the KR→EP framework scales without heavy templating.
  - Prompt‑robust mechanisms: study why small prompt changes flip outcomes and which heads/circuits mediate this sensitivity.
  - New experimental designs: tests for mechanism indivisibility and universality; automated discovery (e.g., scaling IFG, circuit search).
  - Integrate `Machine Psychology` (Section 7.2): design behavioral experiments that map cleanly onto KR/ICI/LR/EP and residual‑stream operations.
  - Build a comprehensive interpretability framework that covers attention‑FFN co‑operation (Section 7.1) and offers theoretical grounding.

- Practical applications
  - Safety and reliability: steer `Truthfulness`/`Faithfulness` heads during inference to reduce hallucinations and improve Chain‑of‑Thought fidelity (Section 4.5).
  - Multilingual and instruction following: leverage `Coherence Heads` to keep output language consistent with the prompt (Section 4.5).
  - Retrieval‑augmented systems: identify and preserve `Retrieval` circuits when compressing or pruning models (Section 4.3.1; Section 5.2).
  - MCQA and structured tasks: use `Content Gatherer` + `Correct Letter` + `Amplification` pipelines to improve option mapping and final choice selection (Sections 4.3.3, 4.4.3, 4.5).

> Anchors for further reading in the paper: Figures 3–4 (model math and residual streams), Equations 3–4 (QK/OV read‑write), Figure 7 (taxonomy of heads), Figures 8–9 (layer distribution and collaboration), Table 4–5 (method taxonomies), Equations 6–7 (scoring functions), and Tables 6–7 (evaluation resources).
