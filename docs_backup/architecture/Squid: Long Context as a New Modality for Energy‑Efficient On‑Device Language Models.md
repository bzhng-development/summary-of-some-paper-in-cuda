# Squid: Long Context as a New Modality for Energy‑Efficient On‑Device Language Models

**ArXiv:** [2408.15518](https://arxiv.org/abs/2408.15518)
**Authors:** Wei Chen, Zhiyuan Li, Shuo Xin, Yihao Wang
**Institutions:** Nexa AI (implied)

## 🎯 Pitch

Squid innovatively introduces long text as a distinct modality, compressing it into 'memory tokens' via a small decoder, reducing the input for larger models and maintaining response quality. This approach significantly cuts latency and power usage, essential for efficient, high-quality on-device applications such as personal assistants and chatbots.

---

## 1. Executive Summary (2–3 sentences)
Squid introduces a two-stage “decoder–decoder” pipeline that treats long textual context as a new modality: a small 0.5B-parameter decoder compresses the context into a short sequence of learned “memory tokens,” which a 7B-parameter decoder then consumes via a simple projector to generate answers. This design substantially reduces the number of tokens that the large model must process, targeting energy and latency reductions for on-device use while preserving response quality (Abstract; Section 3.1; Equations 1–4).

## 2. Context and Motivation
- The specific problem or gap:
  - Long prompts (multi-turn chats, document QA, RAG) dramatically increase inference cost and latency for on-device large language models (LLMs), where power and memory are limited (Introduction).
  - Processing long sequences directly in a 7B-class model is expensive; attention cost and memory traffic scale poorly with input length.
- Why this matters:
  - On-device models enable privacy, offline use, and reduced round-trip latency. Yet, long-context workloads can drain battery and degrade user experience in assistants and chatbots (Introduction).
- Prior approaches and shortcomings:
  - Retrieval-Augmented Generation (RAG) reduces what needs to be fed into the model, but adds a retrieval stack and can still require heavy reading of retrieved content (Related Works: [8], [9]).
  - KV-cache optimizations (compression/swapping/stateful execution) reduce repeated compute but not the cost of the initial long context pass, and often trade memory for complexity (Related Works: [10], others).
  - Prompt compression methods (token pruning, abstractive/extractive summarization, specialized encoders like AutoCompressor, Gist tokens, ICAE) reduce length but introduce extra steps, overhead, or alignment gaps between compressed and original text (Related Works: [12–15, 33–35, 41–45]; “Some other works… do not address the alignment issue,” Introduction).
- Positioning:
  - Squid reframes long text as a distinct modality, borrowing from multimodal LLM design (e.g., LLaVA): a small model produces a compact representation (“memory tokens”) that a large LLM consumes via a projector (Section 3.1; Figure 1; Table 1). This avoids changing the large LLM’s architecture and aims to keep response quality while cutting the large model’s input length.

## 3. Technical Approach
Step-by-step overview of the pipeline (Section 3; Figure 1):

- Key components (with definitions for paper-specific terms):
  - `πs` (“small decoder”): a 0.5B-parameter transformer decoder (Qwen2-0.5B) that reads the full long context `C` (Section 3.1).
  - `memory tokens`: a small set of learned special tokens appended to the end of the context when feeding `πs`. Their output embeddings are extracted to serve as a compact “summary” of the entire context (Section 3.2; Equations 5–7).
  - `Φ` (“projector”): a small multi-layer perceptron (MLP) that maps the `πs` embedding space (dimension 896) to the large decoder’s embedding space (dimension 3584), reusing the “image projector” pattern from vision-language models (Section 3.1).
  - `πl` (“large decoder”): a 7B-parameter transformer decoder (Qwen2-7B) that reads the user query `Q` plus the compact context tokens output by `Φ` and generates the response `R` (Section 3.1).

- Inference flow:
  1. Append `N` learned `memory tokens` to the end of the long context `C`. Formally, `C' = (c1,…,cL, [memory_0], …, [memory_{N-1}])` (Equation 5).
  2. Run the small decoder `πs` over `C'` to get embeddings `Z ∈ R^{(L+N)×ds}` (Equation 6).
  3. Extract just the last `N` rows corresponding to the memory tokens to get a compact representation `M ∈ R^{N×ds}` (Equation 7). This step distills context information into `N` vectors.
  4. Map `M` through the projector `Φ` to get `E ∈ R^{N×dl}` in the large model’s embedding space (Equation 3; Section 3.1).
  5. Feed the user query `Q` and the projected compact context `E` into the large decoder `πl` to produce the response `R` (Equation 4).

- Compression ratio and cost:
  - If the original context length is `L`, and the small decoder outputs `N` memory tokens, the compression ratio is `ρ = L/N` (Section 3.1). The paper reports “ρ can reach up to 8 without compromising quality” (Section 3.1).
  - Compute savings intuition: the small decoder (0.5B) processes the full `L+N` tokens, which is cheaper than having the 7B model do so; the 7B model then processes only `|Q| + N` tokens rather than `|Q| + L` (Section 3.1). This shifts most of the long-sequence burden to a much smaller model and reduces the large model’s attention and memory costs.

- Training procedure (multi-stage; Section 3.3; Table 1 draws analogy to LLaVA):
  - Stage 1: Restoration (autoencoding)
    - Goal: ensure the compact representation preserves enough information to reconstruct the original text.
    - Mechanism: compress full context `C` into `E = Φ(πs(C))`, then train the large decoder to reconstruct `Ĉ = πl(E)` with a reconstruction loss, optionally guided by special prompts (Equations 8–9; Section 3.3.1).
  - Stage 2: Continual training (predict continuation)
    - Goal: teach the large decoder to continue text using only the compressed prefix representation.
    - Mechanism: split `C` into `C1` and `C2`. Compress `C1` (to `E1`), then predict `Ĉ2 = πl(E1)` with a generation loss against `C2` (Equations 10–11; Section 3.3.2).
  - Stage 3: Instruction fine-tuning (task use)
    - Goal: perform instruction-following given an external query `Q` and the compressed context.
    - Mechanism: compress `C` to `E`, then train `πl` to produce `R̂ = πl(Q, E)` with a supervised response loss (Equations 12–13; Section 3.3.3).

- Data for training (Section 3.4):
  - Stage 1: 100K contexts (diverse domains).
  - Stage 2: 100K different contexts (for continuation learning).
  - Stage 3: 1M (context, question, answer) pairs across 20 domains; sources include The Pile, Natural Questions (augmented with longer contexts), BookCorpus, and arXiv.

- Design choices and why they matter:
  - Treat long context as a modality. This makes it natural to reuse the “projector” idea from vision-language models so the large LLM can ingest fixed-length “external” embeddings without changing its architecture (Section 3.1; Figure 1; Table 1).
  - Use a decoder (not an encoder) to compress. Unlike encoder-decoder schemes, `πs` is itself a decoder (Qwen2-0.5B). This aligns with the backbone architecture and leverages autoregressive training dynamics for the memory tokens (Section 3.1–3.2).
  - Append memory tokens at the end. This provides `πs` a fixed “scratch space” to write the distilled summary, which the pipeline then extracts directly by indexing those positions (Equations 5–7; Section 3.2).
  - Simple MLP projector. An MLP suffices to map 896-dim embeddings to 3584-dim (Section 3.1), mirroring image-to-text projection in LLaVA-like systems, reducing engineering complexity.

## 4. Key Insights and Innovations
- Long text as a new modality (fundamental innovation)
  - What’s new: The long textual context is not fed directly to the big model; it is encoded into a fixed number of “memory tokens,” conceptually similar to image tokens in vision-language models (Section 3.1; Figure 1).
  - Why it matters: It decouples the cost of long context from the big model and enables plug-and-play integration via a projector. This bridges long-context NLP and multimodal LLM design.

- Decoder–decoder architecture with learned memory tokens (fundamental innovation)
  - What’s new: A small decoder produces a learned, trainable summary via appended memory tokens, rather than using a separate encoder or an external summarizer (Sections 3.1–3.2; Equations 5–7).
  - Why it matters: It leverages the same architecture family (decoder) for both stages and gives a direct way to supervise what those memory tokens should contain (via restoration and continuation losses), improving alignment between compression and downstream generation.

- Multi-stage curriculum modeled after VLM training (incremental-to-innovative)
  - What’s new: A three-phase process—restoration (autoencoding), continuation (next-part generation), and instruction tuning—mirrors the staged alignment used in LLaVA while being tailored to long text (Section 3.3; Table 1).
  - Why it matters: It systematically teaches the large decoder to (1) trust the compact representation, (2) use it to continue sequences, and (3) answer questions grounded in it.

- Compression without modifying the large LLM’s architecture (incremental but practical)
  - What’s new: The projector mediates between `πs` and `πl` embeddings, so the 7B model can remain largely unchanged (Section 3.1; Figure 1).
  - Why it matters: This reduces engineering overhead for adoption and facilitates swapping in different small compressors or large decoders.

## 5. Experimental Analysis
- Evaluation methodology (Section 4):
  - Test data:
    - 3,740 (context, prompt, response) samples drawn from the PWC (Prompt-with-Context) test set used by ICAE, filtered to contexts shorter than 512 words (Section 4.1).
    - Six task types: Contextual QA, Numeric QA, Rephrasing, Summarization, Title/Keywords, Continuation. Table 2 reports counts and examples (e.g., Contextual QA is 56.36% of the test set).
  - Hardware and baselines:
    - Single NVIDIA A100 80GB GPU on Azure (Section 4.3).
    - Baselines: Qwen2-7B (no compression) for latency and quality parity, and AutoCompressor (Llama-2-7B) for head-to-head comparison (Section 4.3; Table 6).
  - Metrics:
    - Latency (seconds per inference) for Squid vs. Qwen2-7B (Table 4).
    - “Correctness” judged by GPT-4 across categories (Table 5).
    - Pairwise win/lose/tie comparisons (Table 6).
  - Restoration example (Table 3): qualitative demonstration that reconstructed text matches original aside from a single synonym (“stuttered” vs “stalled”).

- Main quantitative results:
  - Latency (Table 4):
    > Squid average inference time: 4.32 s; Qwen2-7B: 20.71 s; Improvement factor: 4.79×.
  - Compression quality (Table 5; judged by GPT-4):
    - Contextual QA: 97.76%
    - Numeric QA: 98.53%
    - Rephrasing: 99.22%
    - Summarization: 99.62%
    - Title/Keywords: 100.00%
    - Continuation: 100.00%
    - Weighted average: 98.53%
  - Head-to-head comparison (Table 6):
    - Squid vs. AutoCompressor: Win 95.1%, Tie 4.9%, Lose 0.0%.
    - Squid vs. Qwen2-7B: Win 23.6%, Tie 44.2%, Lose 32.2% (Win+Tie 67.8%).

- Do the experiments support the claims?
  - Latency: The 4.79× speedup on an A100 (Table 4) supports the claim that reducing tokens for the 7B model materially lowers latency. It is close to—but slightly below—the “5×” reduction highlighted in the Abstract.
  - Quality: High correctness rates across diverse task types (Table 5) indicate the compressed representation preserves enough information for GPT-4 to judge answers as correct in most cases.
  - Energy efficiency: The Abstract claims “a 10-fold improvement in energy efficiency,” but the paper does not present direct energy measurements (e.g., device power draw, joules per token) or on-device benchmarks; only latency on an A100 is reported (Section 4.3; Table 4). This gap limits how strongly one can accept the energy claim.
  - “Long context” generalization: The test set filters to contexts under 512 words (Section 4.1). That does not exercise extreme long-context regimes (thousands to tens of thousands of tokens), which is where the approach is most conceptually advantageous. The paper states a compression ratio up to 8 “without compromising quality” (Section 3.1), but no ablation shows quality as a function of `ρ` or absolute context length.

- Missing or limited analyses:
  - No ablations on number of memory tokens `N`, compression ratio `ρ`, or projector capacity.
  - No breakdown of latency contributions (small vs. large decoder) or effect of query length `|Q|`.
  - Quality is judged by GPT-4; human or exact-match metrics for numeric QA are not reported.
  - Restoration performance is shown as a single illustrative example (Table 3) rather than quantitative metrics (e.g., ROUGE/BLEU/BERTScore between original and restored text).
  - Evaluation runs on server-class GPU, not on representative mobile/edge NPUs/CPUs where the approach targets deployment.

## 6. Limitations and Trade-offs
- Assumptions and scope:
  - The small decoder must be capable of distilling long contexts into a fixed-size memory efficiently. If contexts grow extremely long or are highly diverse, the fixed `N` memory tokens may bottleneck fidelity (Section 3.2).
  - The large decoder is expected to learn to rely on the projected memory tokens. Without strong restoration/continuation training, it may underutilize or misinterpret them (Section 3.3).
- Scenarios not addressed or under-tested:
  - Extremely long contexts (multi-thousand tokens) are not directly evaluated; the test set caps at <512 words (Section 4.1).
  - On-device hardware measurements (runtime, power, thermals) are absent; all latency numbers are from an A100 GPU (Section 4.3).
  - Robustness to noisy or adversarial contexts, multilingual settings, and domain shift are not reported.
- Computational constraints:
  - The small decoder still processes the entire context sequence of length `L+N`. While cheaper than running the 7B model on `L`, its cost still grows with the square of sequence length in standard self-attention (Section 3.1–3.2). For very large `L`, the small model may become the bottleneck.
  - The pipeline introduces an additional forward pass (small decoder + projector) before the large decoder. For short contexts, this overhead might offset benefits.
- Methodological caveats:
  - Energy efficiency claims are not empirically substantiated with power/energy measurements; only latency is reported (Abstract vs. Section 4.3).
  - Quality evaluation relies on GPT-4 as a judge (Table 5), which can introduce bias or variability. No human evaluation or task-specific exact-match metrics are reported.
  - Training details are coarse-grained (e.g., whether `πl` is frozen during early stages is not specified). This makes reproducibility and interpretation of the curriculum’s contribution harder (Section 3.3).

## 7. Implications and Future Directions
- How this work shifts the landscape:
  - It reframes long-context processing as a modular, multimodal-style problem. Treating long text as “another modality” enables reuse of projector-based integration and may generalize to other structured externals (e.g., retrieved knowledge graphs, audio transcripts) (Section 3.1; Table 1).
  - It suggests a practical, architecture-light path to long-context efficiency: compress with a small decoder, keep the large decoder unchanged, and bridge with a projector (Figure 1).
- Follow-up research enabled or suggested:
  - Rigorous long-context benchmarks: Evaluate across 8K–128K token contexts with controlled compression ratios `ρ`, and report accuracy vs. latency/energy trade-offs.
  - On-device studies: Measure energy (joules per token), battery impact, and thermals on mobile NPUs/CPUs/GPUs using frameworks cited in Related Works (e.g., Llama.cpp, ExecuTorch, MediaPipe).
  - Adaptive memory allocation: Learn `N` dynamically based on context complexity or task needs, or use hierarchical memory tokens to scale with `L`.
  - Alternative compressors: Replace `πs` with efficient attention (e.g., linear/sparse attention) or retrieval-augmented small decoders to handle extremely long inputs before distillation.
  - Alignment and ablations: Quantify how each training stage (restoration, continuation, instruction tuning) contributes; test different projector architectures (e.g., cross-attention/Q-Former) vs. MLP.
  - Faithfulness guarantees: Introduce constraints or post-hoc checks that memory tokens faithfully preserve critical facts (especially for numeric QA and safety-critical tasks).
- Practical applications:
  - On-device assistants that must read long emails, documents, or chat histories within tight latency/energy budgets.
  - RAG pipelines where retrieved passages are distilled into memory tokens on-device before final generation, reducing context length while maintaining grounding.
  - Edge settings (IoT, wearables, in-vehicle systems) where long logs or transcripts must be summarized and queried locally.

Block-quoted references to specific claims and results:
- From the Abstract:
  > “a 10-fold improvement in energy efficiency and a 5-fold reduction in latency… without losing quality of the response.”
- From the latency benchmark (Table 4):
  > Average inference time (s) by Squid: 4.32s; by Qwen2-7B: 20.71s; Improvement Factor: 4.79×.
- From the compression quality benchmark (Table 5):
  > Weighted average correctness: 98.53% (Contextual QA: 97.76%, Numeric QA: 98.53%, Rephrasing: 99.22%, Summarization: 99.62%, Title/Keywords: 100.00%, Continuation: 100.00%).
- From the methodology (Section 3.1; Equations 1–4; Section 3.2; Equations 5–7):
  > `M = πs(C)`; `E = Φ(M)`; `R = πl(Q, E)`; with memory tokens appended and extracted as the compact representation.
- From the training curriculum (Section 3.3; Equations 8–13):
  > Restoration: `Ĉ = πl(Φ(πs(C)))`; Continuation: `Ĉ2 = πl(Φ(πs(C1)))`; Instruction FT: `R̂ = πl(Q, Φ(πs(C)))`.

Overall assessment: Squid’s core idea—compressing long text into learned memory tokens with a small decoder and feeding them to a large decoder via a projector—is elegant, practical, and well-motivated by multimodal architectures. The latency gains reported on GPU are strong. To fully validate the “on-device, long-context” promise, future work should provide on-device energy measurements, stress-test truly long sequences, and include detailed ablations on memory-token capacity and training stages.
