# LM2: Large Memory Models

**ArXiv:** [2502.06049](https://arxiv.org/abs/2502.06049)
**Authors:** Jikun Kang, Wenqi Wu, Filippos Christianos, Alex J. Chan, Fraser Greenlee, George Thomas, Marvin Purtorab, Andy Toulis
**Institutions:** 

## 🎯 Pitch

LM2 innovatively integrates a trainable memory bank within a decoder-only Transformer, using gated mechanisms akin to LSTM to enhance long-context reasoning without disrupting attention dynamics. This dual-path design significantly improves the model's ability to store and recall facts across extensive inputs, facilitating more effective multi-step reasoning and offering a pragmatic solution for applications requiring robust long-range memory and reasoning capacities, such as legal document analysis and multi-document QA systems.

---

## 1. Executive Summary
LM2 is a decoder‑only Transformer that adds an explicit, trainable memory bank inside every decoder block and controls its read/write via gates, much like an LSTM’s input/forget/output gates. This simple but carefully integrated memory pathway lets the model store and retrieve long‑range facts without disrupting the normal attention flow, yielding large gains on long‑context, multi‑step reasoning benchmarks and modest gains on general knowledge tests.

## 2. Context and Motivation
- Problem addressed
  - Long‑context reasoning: standard Transformers struggle when essential facts are spread across very long inputs (e.g., dozens of thousands of tokens) and when answers require multi‑step or relational reasoning. The paper highlights the “needle‑in‑a‑haystack” failure mode in which relevant facts are sparsely embedded in long text (Introduction).
  - Retaining and using information over many steps: models often compress or forget earlier details, hurting multi‑hop inference and counting/list aggregation.

- Why it matters
  - Practical impact: long reports, legal documents, logs, multi‑document QA, and code bases commonly exceed normal context windows. Robust long‑range memory can reduce retrieval overhead and enable end‑to‑end reasoning on large contexts.
  - Scientific significance: tests the limits of Transformers’ context handling and explores explicit memory mechanisms as a principled alternative to pure scaling or retrieval.

- Prior approaches and their limits
  - Retrieval‑Augmented Generation (RAG): narrows input to retrieved chunks but can miss multi‑hop dependencies and inter‑chunk reasoning; performance can degrade when evidence must be pieced together across many retrieved items (Related Work; Section 4.1).
  - Recurrent memory Transformers (e.g., RMT, Transformer‑XL): pass short memory tokens or recurrent states between segments. These often summarize prior segments rather than maintain a rich, addressable store; performance can drop at larger contexts (Discussion around MemReasoner’s drop from 60.6 to 18.5 as context >16K; Introduction; Kuratov et al., 2024).
  - Sparse/global attention variants (Longformer, BigBird, etc.): improve efficiency but do not provide an explicit read–write memory with learnable gates (Related Work).

- Positioning of this work
  - LM2 inserts a separate auxiliary memory bank into a decoder‑only Transformer and couples it with cross‑attention reads and gated writes. Crucially, it preserves the original attention pathway and adds a complementary “memory flow” that is dynamically mixed in (Figure 1; Section 2). This aims to combine general‑purpose LLM behavior with targeted, explicit memory.

## 3. Technical Approach
LM2 augments each decoder block with a memory module that is read via cross‑attention and updated via gating. The standard self‑attention path remains intact; the memory path is added as a skip connection.

Step‑by‑step

1) Base model and memory bank
- Backbone: a Llama‑3–style decoder‑only Transformer with 16 blocks, model dimension 2,048, feed‑forward inner size 8,192, and 32 attention heads (Section 3).
- Memory bank `M`: a set of `N` memory “slots,” each with dimension `d = 2048`. The text lists `M ∈ R^{N×d×d}` but then uses projections consistent with `M ∈ R^{N×d}`; functionally, each slot holds a vector‑like memory representation. Slots are initialized to the identity (conceptually giving a neutral starting point) (Section 2.1).

2) Reading from memory via cross‑attention (Memory Information Flow)
- Inputs: token embeddings `E ∈ R^{T×d}` (sequence length `T`), memory bank `M ∈ R^{N×d}` (Section 2.1).
- Project to queries/keys/values (Equation 1):
  - `Q = E_t W_Q`, `K = M_t W_K`, `V = M_t W_V` (learned `W_* ∈ R^{d×d}`).
- Compute attention over memory slots:
  - `A = softmax(Q K^T / √d) ∈ R^{T×N}`
  - Retrieve a memory‑conditioned representation: `E_mem = A V ∈ R^{T×d}`.
  - Causal masking and optional top‑k keep memory read focused on relevant past steps (Section 2.1).
- Output gate controls how much memory to inject into the main flow (Equation 2–3):
  - `g_out = σ(E_mem W_out)`; then the gated memory contribution is `E_gated = g_out · M_t` (the dot denotes element‑wise/broadcast gating).
- Combine with the block’s self‑attention output via a residual path:
  - `E_next = E_attn + E_gated` (end of Section 2.1).
- Intuition: the model learns where to look in memory (via cross‑attention), how much to use (via `g_out`), and mixes the retrieved signal into the normal attention stream through a skip connection.

3) Writing to and maintaining memory (Memory Updates)
- Three gated phases within each block (Figure 2; Section 2.2), analogous to LSTM gating but applied to a shared memory bank:
  - Input gate: how much of the new memory read to write
    - `g_in = σ(E_t W_in)` (Equation 4).
  - Forget gate: how much of old memory to keep
    - `g_forget = σ(E_mem W_forget)` (Equation 5).
  - Update rule: combine bounded new content with retained old content
    - `M_{t+1} = g_in · tanh(E_mem) + g_forget · M_t` (Equation 6).
- Intuition: the memory avoids catastrophic overwrite by deciding what to retain (forget gate) and what to add (input gate). Using `tanh` bounds new content to keep the memory numerically stable.

4) Where the memory lives and how often it is used
- Memory modules are inserted in all 16 decoder blocks for best perplexity and task performance (Section 3; Section 4.3; Figure 5). Variants with fewer memory‑enabled blocks converge more slowly and reach higher perplexity.

5) Pre‑training data and scale
- Total parameters: ~1.2B in the base Transformer + ~0.5B in the memory modules = 1.7B (Section 3).
- Data: 248B tokens focused on language (exclude Python code) from the SmolLM‑Corpus—28B synthetic textbooks/stories + 220B educational web content from FineWeb‑Edu (Section 3).

6) Interpreting and probing the memory
- Neuron‑Explainer probes show individual memory slots specialize: some focus on factual content and Q/A structure, others on formatting cues; irrelevant slots show negative activations (Section 4.4; qualitative Examples “Explanation 4.1–4.3”).
- Test‑time adaptation: cross‑attention heatmaps shift toward task‑relevant tokens as inference proceeds, reflecting memory updates during generation (Figure 6a–b; Section 4.5).

Design choices and rationale
- Preserve the normal Transformer path: LM2 keeps the standard attention outputs untouched and only adds a gated memory skip. This protects general capabilities while letting memory help when useful (Figure 1; Section 2.1).
- Gated writes with `tanh`: prevents noisy overwrites and runaway activations during long sequences (Equation 6).
- Spread memory across blocks: multiple read/write opportunities per layer aid both depth‑wise reasoning and temporal coverage (Figure 5).

Analogy
- Think of each decoder block as both “thinking” (self‑attention/MLP) and “consulting/filing” a shared notebook. At each step, it:
  - Reads: flips to relevant notebook pages (cross‑attention).
  - Decides: how much of what it read to use (output gate).
  - Writes: updates the notebook carefully (input/forget gates) so later blocks/tokens can benefit.

## 4. Key Insights and Innovations
- A. Dual‑path decoder: memory as a complementary flow, not a replacement
  - What’s new: The model maintains an unaltered self‑attention path while injecting a gated memory signal via a residual connection (Figure 1; Section 2.1). Prior recurrent/memory models often entangle memory with the main path or compress prior context into a few tokens.
  - Why it matters: protects general LLM competence while enabling long‑range recall; empirically, LM2 improves both long‑context reasoning and general MMLU performance (Tables 1–2).

- B. Cross‑attention into a persistent, gated memory bank
  - What’s new: Instead of recurrent “memory tokens” passed between segments, LM2 uses an addressable bank read by cross‑attention and written with LSTM‑style gates (Section 2.1–2.2; Equations 1–6).
  - Why it matters: gives the model targeted retrieval (via attention) plus controlled persistence (via gates), improving multi‑hop, counting, and list/set tasks (Figure 3; Table 1).

- C. Memory across many layers improves learning dynamics
  - What’s new: Integrating memory into more decoder blocks lowers perplexity and speeds convergence compared to sparse integration (Figure 5).
  - Why it matters: shows that memory is not just a “top‑layer cache”; distributing read/write opportunities throughout the stack yields better language modeling and reasoning.

- D. Early signs of interpretability
  - What’s new: Using Neuron‑Explainer on memory slots reveals specialization (e.g., factual retrieval vs. structural cues) and changing attention to relevant tokens during generation (Section 4.4–4.5; Figure 6).
  - Why it matters: explicit memory opens the door to diagnosing and coaching what gets stored and retrieved, potentially enabling safer and more controllable LMs.

## 5. Experimental Analysis
Setup
- Datasets and tasks
  - BABILong (Kuratov et al., 2024): long‑context extension of bAbI with tasks like single/multi‑supporting facts, relation tracking, yes/no, counting, lists/sets, negation, and indefinite knowledge (Appendix A). Contexts range 0K (bAbI‑like) up to 128K tokens (Tables 1 and 3).
  - MMLU: broad general‑knowledge multiple‑choice benchmark across subjects and difficulty levels (Section 4.2; Table 2).

- Models compared (Section 4)
  - `LM2-1.7B`: the proposed model.
  - `vanilla-Llama-1.7B`: same base architecture and pretraining data scale as LM2 but without memory.
  - `RMT-1.7B`: Recurrent Memory Transformer with Llama‑1.7B backbone, fine‑tuned on bAbI (Section 4).
  - `Llama‑3.2‑1.2B`: Meta’s 1.2B model (more high‑quality tokens, fewer parameters) as a capability reference.
  - `Llama‑3.2‑1.2B‑RAG`: the 1.2B model with retrieval augmentation for long contexts.

Main results
- BABILong, short to moderate contexts (0K–4K)
  - At 0K (bAbI‑style): 
    > LM2 averages 92.5% vs RMT 76.4%, vanilla‑Llama 75.0%, and Llama‑3.2‑1.2B 40.7% (Table 1).
  - At 4K:
    > LM2 averages 55.9% vs RMT 38.4%, vanilla‑Llama 42.2%, Llama‑3.2‑1.2B 36.8% (Table 1).
  - Interpretation: even without extreme contexts, the memory module improves structured reasoning and factual recall.

- BABILong, very long contexts (≥8K up to 128K)
  - Aggregated 8K–128K:
    > LM2 averages 39.9% vs RMT 35.5%, vanilla‑Llama 31.2%, Llama‑3.2‑1.2B 28.2%, and RAG 32.3% (Table 1).
  - Detailed per‑length (Table 3) shows mixed task‑wise wins at extreme lengths; e.g., at 128K LM2 wins some tasks (e.g., counting qa7: 91.0) but RMT edges LM2 on others (e.g., qa1 17.0 vs 15.0). Overall, LM2 remains competitive or better across most categories.

- Capability profile (Figure 3)
  - Grouping tasks into Single‑step, Multi‑step, Relation Tracking, Basic Query, and Negation & Uncertainty:
    > LM2 dominates most groups except it is relatively closer on Relation Tracking where RAG can be strong due to precise chunking and retrieval.

- MMLU (general knowledge)
  - Averages:
    > LM2 29.4% vs vanilla‑Llama 28.0% and RMT 26.5% (Table 2).
  - By category:
    > Humanities: 32.2 (LM2) vs 28.7 (vanilla)  
    > Social Sciences: 31.6 (LM2) vs 29.2 (vanilla)  
    > STEM: 28.1 (LM2) vs 27.2 (vanilla)
  - Interpretation: the explicit memory pathway does not harm general performance; it yields small but consistent improvements.

Ablations and analyses
- Memory placement ablation (Figure 5)
  - More blocks with memory → lower perplexity; 16‑block integration clearly outperforms 1‑block and 6‑block setups.
  - 1‑block memory matches vanilla’s final level but learns more slowly, implying optimization benefits from distributed memory flows.

- Memory interpretability and test‑time behavior
  - Memory slots specialize (Section 4.4), and cross‑attention shifts from generic tokens to task‑relevant spans as tokens are generated (Figure 6a–b).

Do the experiments support the claims?
- The long‑context gains are sizeable at 0K–4K and remain at ≥8K, though margins shrink and per‑task outcomes can be mixed at 128K (Tables 1 and 3). This pattern is consistent with the goal: better memory over long contexts with reasonable robustness.
- The MMLU improvements, though modest, are important because they show no degradation of general abilities (Table 2).
- Ablations (Figure 5) and analyses (Figures 4–6) provide mechanism‑level plausibility: the memory is used, updated, and influences attention.

## 6. Limitations and Trade-offs
- Computational overhead
  - Parameter cost: +0.5B parameters for the memory modules (1.7B total vs 1.2B base; Section 3).
  - Runtime: cross‑attention against `N=2048` memory slots in every block adds compute and memory usage. The paper does not provide explicit latency or throughput benchmarks.

- Notational inconsistency and shaping details
  - The text alternates between `M ∈ R^{N×d×d}` and `M ∈ R^{N×d}` (Section 2.1). Equations (1)–(3) match the latter. While this does not invalidate results, implementers need clarity on shapes and broadcasting in `E_gated = g_out · M_t`.

- Scope of comparisons
  - RMT is fine‑tuned on bAbI (Section 4), while LM2 and vanilla are pre‑trained from scratch on a large corpus. Although this follows prior practice, fine‑tuning might emphasize certain skills for RMT; conversely, LM2’s broader pretraining could help generalization. More matched fine‑tuning across models would strengthen claims.

- Absolute accuracy at extreme lengths
  - While LM2 leads on average, accuracies at 128K remain modest for several tasks (Table 3). LM2’s large advantage on counting (qa7) contrasts with narrower margins—or occasional deficits—on single‑fact lookups at 128K.

- Persistence beyond a single example
  - The memory updates are described within decoder blocks during a forward pass; persistent memory across separate prompts/sessions is not addressed. This limits use as an external “long‑term memory” across tasks without re‑encoding context.

- Data domain and robustness
  - Pretraining excludes code and focuses on educational content and synthetic textbooks (Section 3). Behavior in other domains (e.g., noisy web forums, legal contracts, multi‑modal inputs) remains to be shown.

## 7. Implications and Future Directions
- How this changes the landscape
  - LM2 demonstrates that adding an explicit, gated memory bank across layers can materially improve long‑context reasoning without sacrificing general abilities. It bridges classic gated memory ideas (input/forget/output) with modern Transformers via cross‑attention.

- Follow‑up research enabled
  - Memory design
    - Learnable memory size and sparsity (adaptive `N`, product‑key or k‑NN addressing).
    - Shared vs per‑layer memory; tying or factorizing memory parameters to reduce overhead.
    - Alternative write rules (e.g., Hebbian/associative updates; contrastive memory objectives).
  - Training strategies
    - Supervised memory supervision: teaching the model where to write/read (e.g., via rationales or retrieval traces).
    - Curriculum over context length and reasoning depth; joint training with chain‑of‑thought signals.
  - Persistence and control
    - Session‑persistent or user‑profile memories with safety controls; selective clearing/forgetting policies.
    - Tool‑augmented LM2: combine with RAG to seed memory with retrieved facts, then let gates integrate and track multi‑hop dependencies.
  - Evaluation
    - Broader long‑context suites (e.g., book‑level QA, long‑form code understanding).
    - Stress tests for temporal consistency, contradiction handling, and catastrophic memory overwrite.

- Practical applications
  - Long‑document QA and summarization where facts must be tracked and combined over tens of thousands of tokens.
  - Log analysis and incident investigation (counting, lists/sets, and multi‑step causal chains are strong points per Table 1 and Figure 3).
  - Educational assistants that must retain structured facts across sections and apply them in later questions.
  - Conversational systems that benefit from explicit, interpretable memory modules for on‑the‑fly adaptation.

Overall takeaway
- By decoupling storage/retrieval from immediate token‑level processing and re‑injecting it through a gated skip path, LM2 offers a practical route to stronger long‑range reasoning while preserving general LLM behavior. The method is simple enough to implement, shows clear gains on BABILong (Tables 1 and 3), and slightly improves MMLU (Table 2), making it a compelling baseline for future memory‑augmented Transformers.
