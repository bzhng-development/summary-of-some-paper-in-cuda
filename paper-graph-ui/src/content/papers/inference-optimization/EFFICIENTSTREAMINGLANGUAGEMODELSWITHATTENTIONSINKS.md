# EFFICIENT STREAMING LANGUAGE MODELS WITH ATTENTION SINKS

**ArXiv:** [2309.17453](https://arxiv.org/abs/2309.17453)

## 🎯 Pitch

This paper introduces StreamingLLM, a lightweight yet powerful inference-time method that allows existing large language models (LLMs) to process arbitrarily long input streams efficiently and with stable performance, simply by retaining a tiny portion of initial 'attention sink' tokens in conjunction with a sliding window of recent tokens. This approach requires no re-training and achieves up to 22× speedup over existing high-quality baselines, breaking the barriers for deploying LLMs in real-world streaming applications such as long conversations and always-on assistants. The authors further demonstrate that pretraining with a dedicated 'sink token' makes LLMs even more robust in streaming, offering an immediately practical solution to one of the biggest bottlenecks in scalable, real-time language modeling.

---

## 1. Executive Summary (2-3 sentences)
The paper introduces StreamingLLM, a simple inference-time framework that lets existing large language models generate over “infinite” input streams efficiently and stably by retaining a tiny set of initial “attention sink” tokens together with a sliding window of recent tokens. It further shows that pretraining with a single dedicated “sink token” makes streaming even more robust. This solves the practical problem of long-running conversations and other streaming applications without retraining and with up to 22.2× decoding speedup compared to the only high-quality baseline that recomputes context for every step.

## 2. Context and Motivation
- Problem addressed:
  - Two bottlenecks prevent LLMs from being used for long, continuous streams (Section 1):
    - Memory/time explosion from caching all past tokens’ key/value states (`KV cache`) during decoding.
    - Poor length extrapolation: many LLMs degrade once input length exceeds the pretraining attention window (Figure 3; also Press et al., 2022 observations cited in Section 2).
- Why this matters:
  - Real deployments (multi-turn chat, day-long sessions, long-running agents) need “always-on” generation with bounded memory and stable quality (Section 1, Applications in Appendix A).
- Prior approaches and their shortcomings:
  - Dense attention: attends to the full past; time is quadratic in sequence length and cache grows unbounded; quality drops beyond training length (Figure 1a; Figure 3 “Dense attention fails once length surpasses pre-training window”).
  - `Window attention`: keep only the most recent L tokens in the KV cache (Beltagy et al., 2020). It is efficient but collapses exactly when the sequence exceeds cache size—because evicting initial tokens breaks the model (Figure 1b; Figure 3).
  - `Sliding window with recomputation`: for each new token, recompute KVs for the most recent window from raw text. Quality is good but cost is prohibitive (O(T·L²)) (Figure 1c), leading to large per-token latency (Figure 10).
  - Context-window extension methods (RoPE scaling, YaRN, fine-tuning) can enlarge a finite window but do not enable “infinite” streaming and do not guarantee good long-context usage (Section 2).
- Positioning:
  - The paper targets length extrapolation to effectively unlimited streams without model fine-tuning. It is complementary to context window extension and can be combined with those methods to broaden the “recent window” part of the cache (Section 1 and Section 4.3, Figure 9).

## 3. Technical Approach
The core idea: models heavily allocate attention to a few initial tokens regardless of their meaning (“attention sinks”). Keep those initial KVs plus a rolling window of recent KVs so attention distributions remain stable while memory and latency stay bounded.

Key terms used below:
- `KV cache`: the stored Key and Value tensors for past tokens used by attention during decoding.
- `Window attention`: cache only the last L tokens.
- `Sliding window with recomputation`: before generating each new token, recompute KVs for the last L tokens from text (no long-term cache).
- `Attention sink`: tokens that receive disproportionately high attention scores across layers/heads even if they are not semantically relevant (Figures 2, 11–13).
- `Sink token`: a dedicated, learned placeholder prepended during pretraining to act as a universal attention sink.
- `RoPE` and `ALiBi`: widely used relative positional encodings; StreamingLLM supports both (Section 3.2).
  
Step-by-step:

1) Observation: attention sinks
- Attention maps for Llama‑2‑7B show that beyond the first two layers, many heads concentrate attention on the very first tokens (Figure 2). This holds for long inputs (Figure 11) and even for a 4096-token position where the first token grabs a large fraction of attention mass in most layers (Figure 12).
- Crucial experiment (Table 1): with `window attention` and a 1024-token cache on Llama‑2‑13B, perplexity explodes to 5158.07 once the first tokens are evicted. Re-adding just 4 initial tokens (plus 1020 recent) restores perplexity to 5.40, and replacing those 4 initial tokens with 4 newline tokens still recovers perplexity (5.60). This shows the special role is positional, not semantic.

2) Why attention sinks arise (Section 3.1; Equation 1):
- Softmax attention must sum to 1 over all visible tokens. When a query has no strong matches, the model still needs to allocate probability mass somewhere. Initial tokens, visible to all later tokens during training, become natural “dumping grounds” and thus act as “attention sinks.” Removing them distorts the softmax denominator and shifts attention distributions.

3) StreamingLLM cache layout (Section 3.2; Figure 4):
- Maintain two disjoint parts of the KV cache:
  - `Attention sinks`: a small, fixed set of the earliest tokens’ KVs (empirically 4 tokens suffice; see Table 2).
  - `Rolling KV cache`: the last L recent tokens (sliding window).
- Complexity and memory remain O(L) per decoding step while keeping the anchor that stabilizes attention scores (Figure 1d).

4) Positional encoding inside the cache (Section 3.2):
- Critical detail: compute relative positions within the cache, not in the original text timeline. If the cache currently holds tokens [0,1,2,3,10,11,12,13] as in Figure 4, they are assigned contiguous positions [0..7] for attention/position encoding.
- Implementation for common encodings:
  - RoPE: store keys before the rotary transform; at each step, apply the correct rotation to the keys in the rolling part using cache-relative positions.
  - ALiBi: apply a contiguous linear bias over the cache range (avoid “jumps”).
- This prevents positional “gaps” that would otherwise degrade attention (Section 3.2).

5) Pretraining with a dedicated sink token (Section 3.3):
- Two options are tested on 160M-parameter models:
  - `Zero Sink` (SoftMax-off-by-one; Equation 2): modify attention to SoftMax1(x) = exp(xi) / (1 + Σj exp(xj)), equivalent to prepending an all-zero Key/Value token in attention. This helps but still leaves some reliance on initial tokens (Table 3).
  - `Learnable Sink token`: prepend a single trainable token to all training sequences. This centralizes the sink role and, at inference, only the sink token needs to be kept in the cache to stabilize streaming (Table 3).
- Regular task performance and convergence are unaffected by adding a sink token (Figure 6; Table 4).

Why this design?
- Retaining a tiny set of perpetual “sinks” keeps attention distributions similar to the training regime, avoiding catastrophic shifts when the window slides (Section 3; Figure 3).
- Using cache-relative positions avoids position-encoding pathologies as tokens are evicted and re-indexed (Section 3.2).
- A learnable sink token eliminates the need to keep several initial content tokens and clarifies the model’s “dump” target (Table 3; Figure 7).

## 4. Key Insights and Innovations
- Discovery and characterization of “attention sinks” (fundamental):
  - Novel empirical insight: across many layers/heads and across model families/scales, the earliest tokens absorb large attention mass regardless of content (Figures 2, 11–13; Section 3.1). Quantitatively, at position 4096 the first token often receives >50% attention in many layers (Figure 12). This reframes why window attention fails when initial tokens are evicted.
- StreamingLLM cache strategy (practical, low-overhead):
  - Simple recipe: keep a handful of initial tokens’ KVs plus a sliding window of recent tokens (Figure 4). No fine-tuning; compatible with standard LLMs using relative positions (Section 3.2). This is a minimal change that reliably prevents collapse.
- Cache-relative position handling (subtle but pivotal):
  - Assign positions within the cache rather than the original timeline and adjust RoPE/ALiBi accordingly (Section 3.2). This removes “jumps” that would otherwise break relative-encoding assumptions as tokens are evicted.
- Pretraining with a single dedicated sink token (forward-looking training recipe):
  - Adding one learnable sink token at the start of every training sequence allows stable streaming with only that sink present at inference (Table 3; Figure 7) and does not harm standard zero-shot task accuracy (Table 4). This is a clean architectural/training knob that turns an emergent quirk into a controlled mechanism.

## 5. Experimental Analysis
Evaluation setup and baselines:
- Long-text language modeling: concatenated PG19 test set (100 books) for perplexity (Section 4.1), with cache sizes set to half of pretraining windows for clarity (e.g., 1024 for Falcon/Pythia/MPT, 2048 for Llama‑2).
- Streaming QA:
  - ARC-Easy/Challenge concatenated into a single stream; exact-match accuracy at each answer position (Section 4.3; Table 5).
  - New “StreamEval” benchmark: queries every 10 lines about content 20 lines back; models tested up to ~120k tokens and under varying query-answer distances (Section 4.3; Figure 8, Figure 9; Appendix C Table 7).
- Baselines: dense attention, window attention, sliding window with recomputation (oracle-quality but slow), and two 32k-extended models for complementarity (Figure 9).
- Models: Llama‑2 [7B,13B,70B], Falcon [7B,40B], Pythia [2.8B,6.9B,12B], MPT [7B,30B] (Section 4).

Main results:
- Quality on long texts:
  - 20k-token sequences (Figure 3): 
    - Dense attention breaks beyond training length.
    - Window attention collapses when initial tokens are evicted.
    - StreamingLLM matches sliding-window-recomputation perplexity.
  - Concrete example (Table 1, Llama‑2‑13B, cache≈1k):
    > Window-only: 5158.07 PPL vs StreamingLLM 4+1020: 5.40 PPL; replace the 4 initial tokens with “\n” and still get 5.60 PPL.
  - Super-long sequences: >4 million tokens (Figure 5):
    > “Perplexity remains stable throughout” across Llama‑2, Falcon, Pythia, and MPT models and scales.
- How many initial tokens are needed (Table 2; ablation):
  - Quality jumps from failure to stability once 4 initial tokens are kept with the recent window. Adding more than 4 gives diminishing returns.
  - Example (Llama‑2‑7B; cache≈4k): PPL 3359.95 (0+4096) → 9.59 (4+4092) with little gain beyond 4.
- Cache size ablation (Table 6):
  - Bigger cache does not always reduce perplexity (e.g., Llama‑2‑7B: 9.32 PPL at 4+1020 vs 9.59 at 4+4092), echoing broader evidence that LLMs do not fully exploit long contexts.
- Streaming QA:
  - ARC (Table 5): window attention’s accuracy collapses (e.g., Llama‑2‑70B‑Chat: 0.12%/0.32%), dense attention OOMs, while StreamingLLM matches or slightly exceeds one-shot accuracy (e.g., 91.37%/80.20% vs one-shot 91.29%/78.50%).
  - StreamEval (Figure 9): StreamingLLM maintains reasonable accuracy up to ~120k tokens, while dense/window fail at pretraining length or cache size respectively. Appendix C (Table 7) shows accuracy declines predictably when the query-answer distance exceeds the cache capacity (i.e., the required evidence has been evicted).
- Speed and memory (Figure 10):
  > Per-token decoding speedup up to 22.2× over the recomputation baseline, with similar memory footprint. StreamingLLM latency grows roughly linearly with cache size, versus the baseline’s quadratic growth.
- Pretraining with a sink token:
  - Streaming perplexity (Table 3): with a learnable sink token, stable streaming is achieved using just that sink. The `Zero Sink` (SoftMax1) helps but still needs extra initial tokens.
  - Regular benchmarks (Table 4; Figure 6): no degradation in zero-shot accuracy or convergence trends compared to a vanilla model.
  - Attention maps (Figure 7): with a sink token, attention consistently targets the sink across layers/heads, reducing reliance on other initial tokens.

Robustness and additional analyses:
- Cross-family generality: attention sink behavior and StreamingLLM stability are shown on Llama‑2, Falcon, Pythia, MPT (Figures 2–5).
- Encoder models: a similar sink phenomenon appears in BERT, where [SEP] acts as an attention sink in many layers (Appendix H, Figure 14).
- LongBench (Appendix D, Table 8): When prompts need both the very beginning and end, a small number of “initial” tokens (e.g., 4) may underperform naive truncation that preserves 1750 initial and 1750 final tokens. Matching the truncation pattern within StreamingLLM (1750+1750) restores parity, underscoring that performance depends on what information remains inside the cache.

Do the experiments support the claims?
- Yes for the core objectives: stable quality with O(L) memory/time, zero fine-tuning, and large inference speedups versus recomputation, across multiple model families and very long streams (Figures 3, 5, 10; Tables 1–2, 5–6).
- The limits are also clearly evidenced: once required evidence is outside the cache, accuracy falls (Appendix C Table 7), and more cache is not always better due to imperfect long-context use (Table 6).

## 6. Limitations and Trade-offs
- Not a context-length extender:
  - StreamingLLM does not increase a model’s attendable window; it only ensures stable behavior within a fixed-size cache (Section A “Limitations”; Appendix C). Tasks that need evidence beyond the cache (e.g., long-document QA/summarization) will still suffer once relevant tokens are evicted.
- Dependence on relative positional encodings:
  - The method is designed for relative encodings like RoPE/ALiBi (Section 3.2). Models with strictly absolute positional embeddings would need additional adaptation.
- Sensitivity to keeping the sink:
  - If sink tokens are not retained, collapse can occur (Table 1; Table 3 “0+1024” rows). Pretraining with a dedicated sink token mitigates this by formalizing the sink role (Table 3; Figure 7).
- Imperfect long-context utilization:
  - Increasing cache size does not guarantee lower perplexity (Table 6), highlighting broader LLM limitations (“lost in the middle”-type effects).
- Practical engineering details:
  - RoPE support requires caching pre-rotary keys and re-rotating each step; ALiBi requires careful contiguous biasing (Section 3.2). Both are straightforward but must be implemented correctly.

## 7. Implications and Future Directions
- How this changes the landscape:
  - Provides a drop-in, training-free recipe for stable, efficient “infinite” streaming on existing LLMs. This decouples a model’s pretraining context length from its practical deployment horizon (Conclusion). It is already adopted in several serving stacks (Impact Statement).
- Practical applications:
  - Multi-turn chatbots, assistants, and agents that run continuously; short-document QA; code assistants; any scenario where only recent context matters but uptime is long (Section A Applications). Pairing with context extension increases the usable “recent” portion of context (Section 4.3; Figure 9).
- Recommended training practice:
  - Prepend a learnable sink token during pretraining to centralize the sink function without harming standard performance (Section 3.3; Table 4; Figure 7). This provides a principled target for “excess attention” and simplifies streaming.
- Research directions:
  - Improve long-context usage so quality increases monotonically with larger cache (Table 6).
  - Explore architectural changes like SoftMax1 (Equation 2) at scale and understand their trade-offs (Table 3).
  - Dynamic or learned strategies for the number and identity of sink tokens; head-wise or layer-wise sink control.
  - Integration with retrieval/memory systems so evicted but important content can be re-injected into the rolling cache.
  - Extend analysis of attention sinks across modalities (images, speech) and architectures (the BERT and ViT parallels in Appendix H suggest a general phenomenon).

In short, StreamingLLM turns an emergent quirk—attention sinks—into a controlled mechanism that unlocks efficient, stable streaming for today’s LLMs, with a clean pretraining tweak (a sink token) that makes it even more reliable.
