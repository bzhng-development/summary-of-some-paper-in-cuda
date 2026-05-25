# Squid: Long Context as a New Modality for Energy-Efficient On-Device Language Models

**ArXiv:** [2408.15518](https://arxiv.org/abs/2408.15518)

## 🎯 Pitch

Squid introduces an innovative decoder-decoder architecture that treats long textual context as a new modality, compressing extensive input with a lightweight language model into efficient memory embeddings, which are then processed by a larger model for final response generation. This design slashes energy consumption and inference latency by up to tenfold without sacrificing quality, making advanced language models vastly more practical and sustainable for on-device and edge applications where speed and battery life are mission-critical.

---

## 1. Executive Summary (2-3 sentences)
Squid proposes a two-stage “decoder–decoder” pipeline that compresses long input text into a small set of learned `memory tokens` using a compact 0.5B-parameter language model, then feeds those tokens to a 7B-parameter language model to generate answers (Figure 1; Section 3.1). By treating long context as its own “modality” and repurposing a multimodal `projector` to adapt embeddings across models, Squid aims to retain task accuracy while cutting inference time and energy use for on-device scenarios (Abstract; Sections 1 and 3).

## 2. Context and Motivation
- Problem/gap addressed
  - Long contexts (e.g., multi-document prompts, multi-turn chat histories) dramatically increase computation and energy consumption in large language models (LLMs), which is a critical blocker for on-device deployment where battery life and latency matter (Introduction).
  - The cost grows because standard LLMs process every input token with the full model; long prompts make inference slow and power-hungry, harming interactive use cases like mobile assistants (Introduction).

- Why this matters
  - Real-world: On-device assistants, offline tools, IoT devices, and wearables need fast, energy-efficient inference under memory and power constraints (Abstract; Introduction).
  - Methodological: Efficiently handling long contexts without sacrificing quality is a core challenge across RAG systems, summarization, QA, and multi-turn dialogue.

- Prior approaches and limitations
  - Retrieval-Augmented Generation (RAG) offloads knowledge to a retriever, then inserts selected passages into the prompt [8, 9]. This helps avoid memorizing everything inside the model, but still leads to long input prompts and compute spikes when many passages are retrieved. It also adds system complexity (Section 1).
  - KV-cache optimizations and stateful serving (e.g., LLMaaS) reduce repeated computation across turns [10, 22], but do not fundamentally shrink the prompt presented to the model at each step (Section 1).
  - Prompt compression methods:
    - Token pruning or extractive selection (e.g., LongLLMLingua [35], Selective-Context [34], document rerankers [29–31]) can remove less relevant text but may miss salient details (Section 2).
    - Abstractive or learned compression (e.g., RECOMP [33], Prompt-SAW [32], Gist tokens [13], ICAE [14], AutoCompressor [12]) can be effective but introduce extra computation for compression, sometimes require architectural changes (e.g., attention masks), and can suffer alignment issues between compressed and original text (Sections 1–2).
  - Long-context transformer variants (e.g., Longformer [47], Performer [49], RMT [51, 52]) alter attention mechanisms but still scale with sequence length and may be challenging to deploy on-device (Section 2).

- How Squid positions itself
  - Treats “long context” as a new `modality`, akin to images in vision-language models (VLMs), and reuses the multimodal recipe: a modality-specific encoder plus a `projector` that maps into the main LLM’s embedding space (Abstract; Section 3.1; Table 1).
  - Uses a small decoder (`πs`, 0.5B) to distill long context into a handful of `memory tokens`, then a large decoder (`πl`, 7B) to answer questions using those tokens plus the user query. This is meant to drastically reduce the number of tokens the 7B model processes while preserving task-relevant information (Section 3.1; Figure 1).

## 3. Technical Approach
Step-by-step overview with key mechanisms and equations

- High-level pipeline (Figure 1; Section 3.1)
  - Inputs: a long context `C` (e.g., retrieved documents or conversation history) and a short user query `Q`.
  - Stage 1 (compression): A compact 0.5B-parameter decoder `πs` (based on `Qwen2 0.5B`) encodes `C` and writes a condensed representation into `N` special `memory tokens`.
  - Projection: An MLP `Φ` (`projector`) converts the 0.5B model’s embeddings (dimension 896) into the 7B model’s embedding space (dimension 3584).
  - Stage 2 (reasoning/response): The 7B decoder `πl` (based on `Qwen2 7B`) consumes the user query `Q` together with the projected memory embeddings `E` to generate the answer `R`.

- Why a “decoder–decoder” design?
  - `πs` is itself a standard generative decoder (not an encoder-only model) that is repurposed to “encode” the long context; this allows reuse of powerful pretrained language modeling abilities to decide what to store in the memory tokens (Section 3.1).
  - The large model `πl` remains architecturally untouched; it simply receives additional tokens (the projected memory embeddings) as if they were normal prompt tokens. This avoids invasive changes to the main LLM (Section 3.1).

- Memory tokens: how the compression works (Section 3.2; Equations 5–7)
  - Define `N` special tokens `[memory_0] ... [memory_{N-1}]` and add them to the 0.5B model’s vocabulary.
  - Append these tokens to the end of the original context:
    - `C' = (c1, ..., cL, [memory_0], ..., [memory_{N-1}])` (Eq. 5), where `L` is the length of the context.
  - Run `πs` over `C'`:
    - `Z = πs(C') ∈ R^{(L+N) × d_s}` (Eq. 6), where `d_s` is the embedding size of `πs` (896 for `Qwen2 0.5B`).
  - Extract the final `N` positions corresponding to the memory tokens:
    - `M = Z_{L+1:L+N} ∈ R^{N × d_s}` (Eq. 7). This `M` is the compact latent summary of the long context.
  - Intuition: think of the memory tokens as “sticky notes” appended to the end of the context, which the small model fills in with the distilled gist needed for answering questions later.

- Projector and main model consumption (Section 3.1; Equations 2–4)
  - Map `M` to the 7B model’s embedding space using the MLP projector `Φ`:
    - `E = Φ(M)` (Eq. 3). This aligns the 0.5B embeddings (dimension 896) to the 7B embeddings (dimension 3584).
  - Feed the user query `Q` plus `E` into the 7B model:
    - `R = πl(Q, E)` (Eq. 4).
  - Claimed compression: If the raw context of length `L` were fed directly to the 7B model, it would process `L × 3584`-dimensional embeddings; by compressing to `N` memory tokens, the 7B model processes far fewer context tokens (Section 3.1). The paper reports compression rate `ρ = L / N` up to 8 “without compromising quality” (Section 3.1), though ablations over different `ρ` are not shown (see Section 6 of this analysis).

- Training procedure (Section 3.3; Equations 8–13; Table 1)
  - Squid uses a three-stage curriculum reminiscent of VLM training (compared to LLaVA in Table 1).
  - Stage 1: Restoration training (autoencoding) to teach the system to preserve information through compression.
    - Pipeline: `E = Φ(πs(C))` (Eq. 8); the 7B model learns to reconstruct `C` from `E`: `Ĉ = πl(E)` (Eq. 9).
    - Objective: minimize difference between `Ĉ` and the original `C`. Prompts/special tokens can guide reconstruction (Section 3.3.1).
  - Stage 2: Continual training (predict the continuation from partial context).
    - Split `C = (C1, C2)`. Compress `C1`: `E1 = Φ(πs(C1))` (Eq. 10).
    - Train the 7B to generate `C2` from `E1`: `Ĉ2 = πl(E1)` (Eq. 11).
    - Objective: minimize discrepancy between `Ĉ2` and `C2` (Section 3.3.2).
  - Stage 3: Instruction fine-tuning (IFT) for final task behavior.
    - Given context `C` and query `Q`, compress `C` and train the 7B model to produce the task response: `E = Φ(πs(C))`, `R̂ = πl(Q, E)` (Eqs. 12–13; Section 3.3.3).

- Data used across stages (Section 3.4)
  - Restoration: 100K diverse context samples.
  - Continual: 100K distinct context samples.
  - IFT: 1M question-answer pairs paired with relevant long contexts across ~20 domains, drawing from sources like The Pile [39], Natural Questions, BookCorpus, and arXiv (Section 3.4).

- Inference path (Section 3.1)
  - For each query: run `πs` once over the long context to fill `N` memory tokens, project to `E`, then run `πl` on `[Q, E]`. The 7B model never sees the full context `C` directly; it only sees the compressed memory tokens.

## 4. Key Insights and Innovations
- Treat long context as a new “modality” with a VLM-style projector (Abstract; Section 3.1; Table 1)
  - Novelty: Instead of feeding long text directly or designing new attention mechanisms, Squid repurposes multimodal alignment ideas. The long text is “encoded” by a small LLM and then projected into the big LLM’s embedding space, just as images are projected for VLMs.
  - Significance: This enables token-level compression that integrates seamlessly with an unmodified 7B LLM, facilitating deployment.

- Decoder–decoder compression via learned `memory tokens` (Sections 3.1–3.2; Eqs. 5–7)
  - Novelty: The compressor is a generative decoder (0.5B) that writes summaries into appended tokens learned for this purpose, rather than a separate encoder or handcrafted pruning.
  - Significance: Leverages pretrained language knowledge in the small model to decide what to keep, potentially improving alignment between what’s compressed and what the 7B model needs for downstream tasks.

- Three-stage training to preserve information and usability (Section 3.3; Eqs. 8–13; Table 1)
  - Novelty: Combines restoration (autoencoding), continuation prediction, and instruction tuning—mirroring successful VLM curricula (cf. LLaVA in Table 1) but applied to long-text-as-modality.
  - Significance: Restoration ensures information survivability through compression; continuation teaches coherence; instruction tuning delivers task-following behavior.

- Efficiency through asymmetric computation (Section 3.1)
  - Insight: Processing the full context with a small 0.5B model is much cheaper than with a 7B model; the 7B only processes a short set of memory tokens, reducing its computational load. The paper reports compression rates up to `ρ ≈ 8` without quality loss (Section 3.1), and a measured latency reduction of 4.79× on an A100 GPU (Table 4).

## 5. Experimental Analysis
- Evaluation methodology (Section 4)
  - Datasets
    - Testing uses 3,740 (context, prompt, response) samples selected from the Prompt-with-Context (PWC) dataset used in ICAE [14], choosing contexts under 512 words “to align with the default maximum context length of the Squid model” (Section 4.1; Table 2).
    - Task types include: Contextual QA, Numeric QA, Rephrasing, Summarization, Title/Keywords, Continuation, with counts listed in Table 2.
  - Setup
    - Hardware: single NVIDIA A100 80GB GPU on Azure (Section 4.3).
    - Baselines: `Qwen2-7B` without compression; `AutoCompressor` [12] (based on Llama-2-7B) for pairwise quality comparisons (Section 4.3; Table 6).
  - Metrics
    - Latency (seconds) for end-to-end inference.
    - “Compression quality” scored by GPT-4 as a judge (given input prompt and question), reported as “Correctness (%)” per category (Table 5).

- Main quantitative results
  - Latency (Table 4; Section 4.3)
    - The paper presents the following:
      > Average inference time (s) by Squid 4.32s  
      > Average inference time (s) by Qwen2-7B 20.71s  
      > Improvement Factor 4.79×
    - Interpretation: Even though Squid runs two models (0.5B + 7B), the 7B model sees only a few memory tokens, not the full long context; this reduces its heavy compute. The 0.5B pass is cheap enough that total time drops substantially.
  - Correctness evaluated by GPT-4 (Table 5; Section 4.3)
    - The paper reports:
      > Contextual QA: 97.76%  
      > Numeric QA: 98.53%  
      > Rephrasing: 99.22%  
      > Summarization: 99.62%  
      > Title/Keywords: 100.00%  
      > Continuation: 100.00%  
      > Weighted average: 98.53%
    - Interpretation: On these short-to-moderate contexts (<512 words), compressed memory tokens appear to preserve task-relevant information very well by GPT-4’s rubric.
  - Pairwise model comparisons (Table 6; Section 4.3)
    - Squid vs AutoCompressor (Llama-2-7B-based):
      > Win 95.1% | Lose 0.0% | Tie 4.9% | Win+Tie 100.0%
    - Squid vs Qwen2-7B (base of Squid’s 7B):
      > Win 23.6% | Lose 32.2% | Tie 44.2% | Win+Tie 67.8%
    - Interpretation: Squid achieves parity or near-parity with its 7B base model on a substantial fraction of cases despite compressing context; and it strongly outperforms AutoCompressor in this evaluation.

- Qualitative restoration example (Table 3; Section 4.2)
  - In an autoencoding test, the restored output differs by one rare word: “stuttered” → “stalled,” preserving meaning:
    > “When the hands finally stuttered to a stop, …”  
    > → “When the hands finally stalled to a stop, …”
  - Interpretation: Restoration preserves semantic content even when lexical substitutions occur.

- Convincingness and gaps
  - Strengths
    - Clear latency improvement vs. a non-compressed baseline on identical hardware (Table 4).
    - High task correctness by a strong external judge (GPT-4) across multiple task types (Table 5).
    - Head-to-head comparison with a learned compression baseline (AutoCompressor) shows strong wins (Table 6).
  - Gaps and caveats
    - Context lengths in testing are capped at <512 words (Section 4.1; Table 2), which limits evidence about very long contexts that motivate the method. The paper claims compression rates up to `ρ = 8` (Section 3.1), but does not supply ablations across `N`, `L`, or `ρ`.
    - Energy-efficiency claim of 10× is stated in Abstract but no direct energy measurements (e.g., Joules, power draw over time) or device-side experiments are provided; latency is measured on an A100 GPU rather than on-device NPUs/CPUs (Abstract; Section 4.3).
    - “Correctness” relies on GPT-4 judging; while reasonable, it may inflate agreement on paraphrased outputs and is not a domain-ground-truth metric. No human evaluation or exact-match metrics are reported.
    - The comparison to AutoCompressor is cross-base-model (Qwen2 vs Llama-2), which complicates strict fairness.

## 6. Limitations and Trade-offs
- Assumptions and scope
  - The approach assumes a small decoder (`πs`) can reliably distill all task-critical information from long contexts into a small `N`-token summary. There is no systematic analysis of failure modes when content is highly dispersed or demands exact quoting (Sections 3.1–3.2; 4.3).
  - Evaluation focuses on contexts under 512 words (Section 4.1), leaving open how Squid performs on truly long inputs (e.g., tens of thousands of tokens), where the method’s benefits should be largest.

- Computational trade-offs
  - Squid still processes the entire long context once with the 0.5B model. Benefits come from sparing the 7B model from seeing the whole context. If `L` becomes extremely large, the 0.5B pass could become a bottleneck; this is not profiled across `L` (Sections 3.1, 4.3).
  - The final system must load both 0.5B and 7B models on-device (or at least run them in sequence). For some devices, memory/thermal constraints may still be tight; the paper does not report memory footprint or device-side throughput (Abstract; Sections 1, 4.3).

- Evaluation and measurement constraints
  - No on-device energy measurements despite the central energy-efficiency claim; all benchmarks run on an A100 GPU (Abstract; Section 4.3).
  - No ablations on number of memory tokens `N`, projector capacity, or training stages’ contribution (e.g., removing restoration training) to understand which components are critical.
  - Correctness is judged by GPT-4 rather than ground-truth labels or human raters; this can bias toward semantic similarity over exactness, especially for numeric QA (Table 5).

- Generalization and robustness
  - Seen domains: The training uses large, diverse corpora (Section 3.4), but the tests are limited to PWC-derived tasks. Robustness to domain shift, multi-document reasoning, and adversarially long contexts remains untested in the paper.
  - Compression fidelity vs. query specificity: The paper does not distinguish query-agnostic vs. query-aware compression. If the task requires different parts of the context depending on the query, a single pre-compressed memory might omit needed details.

## 7. Implications and Future Directions
- How this changes the landscape
  - Reframing long-text handling as a “modality alignment” problem bridges LLM compression with established multimodal engineering practices (projectors, multi-stage alignment training). This can simplify integration: the main LLM remains unchanged and sees fixed-size memory tokens, potentially standardizing long-context adapters.

- Practical applications
  - On-device assistants that need to consult large local stores (emails, notes, documents) without uploading data to the cloud; the small model can compress locally, and the big model can reason over compact memories.
  - RAG pipelines: compress retrieved passages into memory tokens before passing them to the reasoner, enabling larger retrieval batches without overwhelming the main LLM’s context window.
  - Multi-turn chat memory: replace ever-growing histories with rolling memory tokens that capture relevant state.

- Research directions
  - Rigorous on-device studies: measure Joules per query, thermal throttling behavior, and latency on NPUs/CPUs across representative devices to substantiate the energy claims.
  - Long-context stress tests: evaluate on standardized suites (e.g., LongBench-style tasks, “Needle-in-a-Haystack” tests) with contexts far exceeding 512 words, and report ablations over `N`, `ρ = L/N`, and projector capacity.
  - Query-aware compression: condition the 0.5B compressor on the user query `Q` so memory tokens encode exactly what the 7B will need, potentially improving precision for sparse signals.
  - Architectural variants: explore cross-attention projectors (e.g., Q-Former-like [21]) instead of an MLP `Φ`, or hierarchical memory tokens written at multiple layers/positions to capture both global and local details.
  - Continual and streaming settings: study how to update memory tokens incrementally as new context arrives, and how this interacts with KV-cache optimizations.
  - Fair, apples-to-apples baselines: compare against other learned compression approaches using the same base model (e.g., Qwen2-7B) and standardized evaluation judges or human raters.

In sum, Squid’s core idea—compress long text with a small decoder into learned memory tokens and feed those into a large decoder via a modality projector—offers a clean, modular path to speed up and scale down long-context processing. The latency gains reported on GPU are promising (Table 4), and correctness judged by GPT-4 is high (Table 5), but comprehensive on-device energy measurements, long-context stress testing, and ablations will be key to establish the method’s reliability and boundaries.
