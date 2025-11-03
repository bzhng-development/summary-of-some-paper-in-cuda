# EFFICIENT STREAMING LANGUAGE MODELS WITH ATTENTION SINKS

**ArXiv:** [2309.17453](https://arxiv.org/abs/2309.17453)

## 🎯 Pitch

This paper introduces StreamingLLM, a new inference-time method that enables large language models to handle arbitrarily long streaming inputs efficiently and stably, without any fine-tuning. By retaining a handful of 'attention sink' tokens from the start of the session together with a sliding window of recent tokens, StreamingLLM breaks through memory and latency barriers while sustaining high accuracy, making it practical to deploy chatbots and assistants that operate over long, ongoing conversations in real-world settings.

---

## 1. Executive Summary
This paper introduces StreamingLLM, a simple inference-time method that makes existing decoder-only language models work stably on arbitrarily long, streaming inputs without fine‑tuning. It does so by keeping a tiny set of “attention sink” tokens from the beginning of the session together with a sliding window of recent tokens, achieving constant memory, linear-time decoding in the window size, and accuracy that matches much slower recomputation baselines.

## 2. Context and Motivation
- Problem addressed
  - Production chatbots, assistants, and other streaming applications require models to sustain long, ongoing interactions. Two blockers:
    1) Caching every past token’s attention states (“KV cache”) grows memory and latency over time.
    2) Most LLMs break down when inputs exceed their training context length (length extrapolation failure).
  - Figure 1 contrasts common strategies and shows they either become intractable (dense attention), or collapse once the first tokens are evicted (window attention), or are accurate but far too slow (sliding window with re‑computation).

- Importance
  - Real-world assistants need stable, long-running sessions without blowing up memory or resetting context. The paper’s approach decouples the model’s pretraining window from its usable stream length, enabling always-on deployments (Section 1, Figure 1).

- Prior approaches and their gaps
  - Dense attention: Full cache and full attention at each step; time O(T^2) and increasing memory; performance often degrades beyond training window (Figure 1a; Figure 3).
  - Window attention: Keep only the last L tokens; fast and memory-bounded, but performance collapses as soon as initial tokens’ keys/values are dropped (Figure 1b; Figure 3).
  - Sliding window with re-computation: For each new token, rebuild the cache for the most recent L tokens by re-encoding them; accurate but quadratic in the window size per step O(T·L^2) and thus slow (Figure 1c; Figure 10).
  - Context extension and positional tricks (e.g., RoPE scaling, ALiBi): They extend the maximum attendable context but do not solve infinite streaming, and performance beyond the new window still degrades (Section 2).

- Positioning
  - StreamingLLM lies in “length extrapolation” (Section 2): it does not expand the model’s context window. Instead, it makes models reliable on arbitrarily long inputs by stabilizing attention when using a fixed, small cache.

## 3. Technical Approach
Key terms (defined only where uncommon):
- KV cache: During autoregressive decoding, each token’s attention “Key” and “Value” vectors are stored so future tokens can attend to them without recomputation.
- Window attention: Keep only the KVs for the most recent L tokens.
- Attention sink: Tokens that consistently receive high attention mass across layers and heads, not because of semantic relevance but because the attention softmax needs to distribute probability somewhere; in standard LLMs, the first few tokens often become such sinks (Section 3.1, Figure 2).
- Perplexity (PPL): Standard language modeling metric; lower is better.
- RoPE/ALiBi: Relative positional encoding schemes widely used in modern LLMs.

Step-by-step mechanism
1) Diagnose why window attention fails
   - Empirical finding: Across many layers/heads, models allocate disproportionately high attention to the earliest tokens in a sequence (Figure 2; also quantified for long inputs in Figure 12).
   - Removing those tokens’ KVs drastically changes the softmax normalization and destabilizes attention, causing sky-high perplexity once the sequence exceeds the cache size (Figure 3; Table 1).
   - The paper explains this via the attention softmax. If token 1 routinely has a much larger attention logit (`x1 >> xj` in Equation 1), dropping it removes a large part of the denominator and warps the distribution:
     > Equation (1), Section 3.1: Softmax over scores `x` where the initial token has x1 >> others, so removing it shifts all normalized weights.

   - Crucially, this “sink” role is positional, not semantic. Replacing the original first four tokens with four newline tokens still restores performance when combined with the recent window (Table 1: “4'\n'+1020” gives PPL 5.60, comparable to “4+1020” at 5.40).

2) StreamingLLM cache layout (“rolling KV cache with attention sinks”)
   - Maintain a fixed-size cache that is the union of:
     - A tiny, pinned set of initial tokens (attention sinks; four usually suffice).
     - A sliding window of the most recent tokens (Figure 4).
   - Every new token evicts the oldest token from the sliding part, but the sink tokens’ KVs are never evicted.
   - Positions are assigned relative to the current cache, not absolute time in the original long stream (Section 3.2). Example (Figure 4): if the cache holds tokens [0,1,2,3,6,7,8] while decoding the 9th token, they are renumbered as [0..6,7] locally in-cache. This keeps relative distances coherent when the cache rolls.

3) How this integrates with positional encodings
   - RoPE: Cache keys before applying rotary transformation, then at each step re-apply RoPE to the rolling keys with their new local positions (Section 3.2).
   - ALiBi: Apply a contiguous linear bias based on the local in-cache distances (no “jumps”; Section 3.2).
   - This detail is critical: treating positions within the cache avoids artifacts from “skipping” positions in a long stream.

4) How many sink tokens?
   - Ablations show one or two is often insufficient; four is generally enough; more gives diminishing returns (Table 2).

5) Optional: pretraining with a dedicated sink token
   - Two variants (Section 3.3; Table 3):
     - “Zero sink” (SoftMax1 in Equation 2): equivalent to prepending a token with zero key/value so softmax mass can flow there. It helps but still requires extra initial tokens in streaming setups.
     - “Learnable sink token”: prepend a dedicated, trainable token to every training sample. Models then rely on that single token to collect unneeded attention (Figure 7), enabling stable streaming with only that token plus the recent window (Table 3).
   - Figure 6 and Table 4 show pretraining with a sink token neither hurts convergence nor standard NLP performance.

Why this works
- The approach honors what existing models already do: concentrate surplus attention on a few always-visible positions. By pinning those positions in the cache and renumbering positions locally, the attention softmax operates in-distribution even as the session grows indefinitely.

Complexity and implementation
- Time: O(T·L), where T is total tokens processed and L is cache size (recent window plus sinks). Memory: O(L). Compare to dense attention O(T^2) and sliding-window recomputation O(T·L^2) (Figure 1; Figure 10).
- No model changes are required for the base StreamingLLM method; it’s an inference-time cache policy and position handling. Pretraining with a sink token is optional but can further simplify deployment (Section 3.3).

## 4. Key Insights and Innovations
- Discovery of the “attention sink” phenomenon (fundamental)
  - Novel observation: initial tokens receive large attention across layers/heads even when semantically irrelevant (Figure 2; Figure 12; Section 3.1). Replacing them with newline tokens still restores performance (Table 1), confirming a positional rather than semantic effect.
  - Significance: Explains why window attention collapses upon evicting the first tokens and reveals a general property across architectures (see also Appendices H for BERT and ViT parallels).

- StreamingLLM: retain a few sink tokens + recent window (practical, high impact)
  - New cache policy and position handling that stabilizes attention for infinite streams without fine‑tuning (Section 3.2; Figure 4). Empirically matches the accuracy of the recomputation baseline while being far faster (Figure 3; Figure 10).
  - Orthogonal to context-extension methods: can be combined to increase the size of the sliding window (Figure 9; discussion in Section 4.3).

- Pretraining with a dedicated sink token (conceptual and practical)
  - Introduce one learnable token at the start of every sequence during pretraining so the model does not need several initial tokens as sinks (Section 3.3; Figure 7).
  - Table 3 shows that with a sink token, stable streaming perplexity can be achieved with only that token plus recent tokens, whereas vanilla models need multiple initial tokens.

- Cache-local positional treatment for RoPE/ALiBi (necessary design detail)
  - Storing pre-rotary keys (RoPE) and reapplying position transforms, or using contiguous ALiBi biases, ensures the cache remains a coherent “mini-sequence” despite rolling (Section 3.2). This implementation insight is key to making the approach work in practice.

## 5. Experimental Analysis
Evaluation setup
- Models: Llama‑2 (7B/13B/70B), MPT (7B/30B), Falcon (7B/40B), Pythia (2.9B/6.9B/12B), plus custom 160M models for sink-token pretraining (Sections 4, 4.2).
- Datasets and tasks:
  - Long-form language modeling on PG‑19 books (20K tokens and a 4‑million-token concatenation) using perplexity (Figures 3 and 5).
  - Streaming QA with instruction-tuned models on ARC‑Easy and ARC‑Challenge concatenated in a single stream (Table 5).
  - StreamEval: a purpose-built benchmark where the model is queried every 10 lines about content ~20 lines earlier (Figures 8 and 9; Appendix C provides distance sensitivity).
  - LongBench subset to probe summarization and QA with long inputs (Appendix D; Table 8).
- Baselines: dense attention, window attention, and sliding window with recomputation (Figures 1, 3, 10). Context-extended models are also probed (LongChat‑7b‑v1.5‑32k; Llama‑2‑7B‑32K‑Instruct) to show complementarity (Figure 9).

Main quantitative results
- Language modeling stability and accuracy
  - On 20K-token inputs, StreamingLLM matches the recomputation baseline; dense and window methods collapse (Figure 3).
  - On a 65K-token PG‑19 book with Llama‑2‑13B: Figure 1 summarizes PPL for predicting the 7th token far beyond training length: dense 5641, window 5158, recomputation 5.43, StreamingLLM 5.40.
  - Table 1 (Llama‑2‑13B, PG‑19 book): window-only `0+1024` PPL 5158.07; adding four initial tokens `4+1020` drops PPL to 5.40. Replacing those initial tokens with four newline tokens achieves 5.60.
  - Four sink tokens are generally enough across families/scales, with diminishing gains beyond four (Table 2).
  - StreamingLLM remains stable over 4 million tokens for all tested model families/scales (Figure 5).

- Efficiency
  - Decoding latency: StreamingLLM is up to 22.2× faster per token than sliding window with recomputation, and the speed advantage grows with cache size due to the latter’s quadratic behavior (Figure 10).
  - Memory: Similar footprint to the recomputation baseline and bounded by the cache size (Figure 10).

- Streaming QA
  - Concatenated ARC streams with instruction-tuned Llama‑2‑Chat models: dense fails with OOM; window attention accuracy is near random; StreamingLLM reaches the one-shot baseline (e.g., for Llama‑2‑7B‑Chat on ARC‑Easy: one-shot 71.25% vs StreamingLLM 71.34%; Table 5).

- StreamEval
  - Accuracy remains reasonable as input grows towards 120K tokens, while dense and window attention fail at their respective limits (Figure 9).
  - Appendix C (Table 7) shows accuracy degrades when the distance between query and answer exceeds the cache; within-cache distances are handled well.

- Pretraining with a sink token
  - Similar convergence (Figure 6) and similar zero-shot accuracy across seven NLP tasks (Table 4).
  - Streaming with only the sink token and recent tokens is viable (Table 3; “Learnable Sink” row). Attention maps confirm the model learns to route surplus attention to the sink (Figure 7).

Ablations and robustness
- Number of initial tokens: Four is the reliable default (Table 2).
- Cache size: Increasing cache size does not always reduce perplexity (Table 6), consistent with broader evidence that LLMs under-utilize long contexts.
- Generality of the phenomenon: Attention sinks also appear in BERT and Vision Transformers, where specific tokens or patches act as global “registers” (Appendix H Figure 14; discussion in Appendix B).

Assessment
- The experiments are diverse (models, scales, tasks) and convincingly support the central claims: (i) attention sinks exist and matter; (ii) pinning a few initial tokens plus a sliding window stabilizes long-stream decoding; (iii) the approach is fast and memory-efficient; (iv) an explicit sink token during pretraining further simplifies streaming use.

## 6. Limitations and Trade-offs
- What StreamingLLM does not do
  - It does not extend the model’s intrinsic context window; it just makes streaming over infinite time possible by focusing on the recent window plus sinks. Information outside the cache cannot be used (Discussion in Section A; Appendix C Table 7 shows accuracy drops to zero when query-answer distance exceeds cache).
  - Not a memory or retrieval mechanism for distant facts; tasks like long-document QA or summarization that require seeing far-back content beyond the cache are not solved (Section A and Appendix D Table 8).

- Assumptions and applicability
  - Requires relative positional encodings (e.g., RoPE, ALiBi) and the ability to reassign local positions in-cache (Section 3.2).
  - Assumes that early tokens act as sinks in the target model. This holds broadly in tested families/scales (Figure 2; Table 2), but bespoke models might deviate.

- Quality vs cache size
  - Bigger caches do not always yield lower perplexity (Table 6), suggesting models may underuse long contexts—an external limitation of current LLMs.

- Pretraining variant caveats
  - “Zero sink” (SoftMax1; Equation 2) helps partially but still often needs additional initial tokens (Table 3).
  - A dedicated sink token requires modifying pretraining data and pipeline (Section 3.3), which is not always feasible.

- Edge cases
  - If a deployment requires precise reasoning over specific old content far outside the cache, StreamingLLM alone is insufficient—must be paired with retrieval, memory, or context-extension methods (Figure 9 notes complementarity with LongChat‑32k and Llama‑2‑32k).

## 7. Implications and Future Directions
- How this work shifts the landscape
  - It provides a practical, drop-in way to run LLMs on unbounded streams with stable quality and constant memory—decoupling model training window from deployment session length (Conclusion; Section 5).
  - The technique has already been integrated into major inference stacks (Impact Statement): NVIDIA TensorRT‑LLM, Intel Extension for Transformers, HuggingFace Transformers, MLC LLM.

- Practical applications
  - Always-on assistants and multi-turn chat that run for days without resets.
  - Streaming summarization within a moving window (e.g., call centers, live monitoring) where only recent context matters.
  - Complement to context-extension: use StreamingLLM to handle unbounded time, and expand the sliding window as hardware allows (Figure 9).

- Follow-up research directions
  - Train with a dedicated sink token by default to simplify streaming deployment (Section 3.3; Figure 7; Table 3).
  - Combine with retrieval and memory modules to recall content beyond the cache when needed.
  - Improve models’ utilization of longer contexts so that larger caches consistently improve quality (Table 6; aligns with “Lost in the Middle” observations cited in Section C).
  - Explore theoretical underpinnings of attention sinks and their relation to softmax normalization across different architectures (Appendices F–H).

> Representative results:
> - PG‑19 long text modeling: dense attention fails beyond training length; window attention collapses once initial tokens are evicted; StreamingLLM matches recomputation in PPL while being far faster (Figures 1 and 3; Figure 10).
> - 4M-token streams: perplexity remains stable for Llama‑2, Falcon, Pythia, MPT (Figure 5).
> - Streaming QA on ARC with Llama‑2‑Chat: dense OOM; window near random; StreamingLLM ≈ one-shot baseline (Table 5).
> - Pretraining with a sink token: similar convergence and task performance (Figure 6; Table 4), with stable streaming using just that token plus recent window (Table 3; Figure 7).

In short, StreamingLLM is a small but powerful change in cache policy and positional handling grounded in a clear empirical insight (attention sinks). It enables efficient, stable, infinite-horizon decoding with existing LLMs and suggests a simple pretraining tweak (a sink token) to make streaming even more robust.
