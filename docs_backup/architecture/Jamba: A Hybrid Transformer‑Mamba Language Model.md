# Jamba: A Hybrid Transformer‑Mamba Language Model

**ArXiv:** [2403.19887](https://arxiv.org/abs/2403.19887)
**Authors:** Opher Lieber, Barak Lenz, Hofit Bata, Gal Cohen, Jhonathan Osin, Itay Dalmedigos, Erez Safahi, Shaked Meirom, Yonatan Belinkov, Shai Shalev‑Shwartz, Omri Abend, Raz Alon, Tomer Asida, Amir Bergman, Roman Glozman, Michael Gokhman, Avashalom Manevich, Nir Ratner, Noam Rozen, Erez Shwartz, Mor Zusman, Yoav Shoham
**Institutions:** AI21 Labs

## 🎯 Pitch

Introducing Jamba, a groundbreaking model that interleaves Transformer attention with Mamba state-space layers and incorporates mixture-of-experts (MoE) MLPs. This innovative design delivers Transformer-level accuracy with drastically reduced memory and increased throughput, enabling efficient long-context processing up to 256K tokens on a single 80GB GPU—opening new possibilities for complex natural language processing applications, such as multi-document analysis and long-form content generation.

---

## 1. Executive Summary
This paper introduces Jamba, a large language model that interleaves Transformer attention layers with Mamba state‑space layers and adds mixture‑of‑experts (MoE) MLPs. The hybrid design delivers Transformer‑level quality while dramatically reducing memory and increasing throughput for long contexts—supporting up to 256K tokens with an 8× smaller key–value (KV) cache than attention‑only models (Table 1) and up to 3× higher throughput at long sequences (Figure 3b).

## 2. Context and Motivation
- Problem addressed
  - Attention‑only LLMs struggle with long contexts because the KV cache grows linearly with the number of attention layers and tokens, driving memory up and throttling throughput. Inference also reprocesses the entire KV cache for every generated token, reducing speed (Section 1).
  - Pure recurrent/SSM models (e.g., Mamba) maintain a single summary state and are efficient for long sequences, but they typically underperform Transformers at comparable scale and have difficulty with certain in‑context learning behaviors (Section 1; Section 6.2).

- Why this matters
  - Long‑context reasoning (e.g., multi‑document QA, legal and financial analysis) often exceeds 32K–128K tokens. Making such contexts practical on commodity accelerators enables new applications (Sections 1–3; Table 1; Figure 2).
  - Higher throughput at long contexts reduces serving cost and latency (Figure 3).

- Prior approaches and their limits
  - Attention‑only MoE models like `Mixtral‑8x7B` achieve high quality but retain large KV caches and slow down on long contexts (Table 1, Figure 3).
  - Earlier attention–SSM hybrids work at smaller scale or underperform strong Transformer baselines (Section 1: related efforts such as S4+local attention, H3, StripedHyena).

- How this paper positions itself
  - Jamba is a production‑grade hybrid with:
    - Interleaved attention and Mamba layers at a 1:7 ratio in each block (Figure 1; Section 3.1).
    - MoE in half of the MLPs (every other layer) with 16 experts and top‑2 routing, yielding 52B total (“available”) parameters but only 12B “active” parameters per token (Sections 2–3.1; Table 1).
    - A design that fits on a single 80GB GPU even for very long contexts, while matching or approaching the quality of larger Transformer baselines on standard benchmarks (Sections 3–5; Table 2).

## 3. Technical Approach
Jamba combines three components inside repeated “Jamba blocks.”

- Architecture building blocks (Figure 1; Section 2)
  - Layer types:
    - `Transformer layer`: RMSNorm → Attention → RMSNorm → MLP.
    - `Mamba layer`: RMSNorm → Mamba (a state‑space layer) → RMSNorm → MLP.
    - `Mamba MoE layer`: same as Mamba layer but MLP is replaced by an MoE MLP.
  - Jamba block configuration variables (Section 2):
    - `l`: number of layers per block.
    - `a:m`: ratio of attention to Mamba layers inside the block.
    - `e`: how often to replace the MLP by an MoE MLP (e.g., every 2 layers).
    - `n`: number of experts per MoE layer.
    - `K`: number of experts activated per token (top‑K routing).

- Concrete instantiation released (Section 3.1; Figure 1)
  - 4 Jamba blocks; each block has `l = 8` layers with an `a:m = 1:7` mix.
  - MoE applied every `e = 2` layers.
  - Per MoE layer: `n = 16` experts, `K = 2` active experts per token.
  - This yields 32 total layers with just 4 attention layers overall—greatly shrinking the KV cache.
  - Additional details: grouped‑query attention (GQA), SwiGLU activations, MoE load balancing, 64K BPE vocabulary with digit‑level tokens, no dummy space token, and crucially, no explicit positional encoding (no RoPE) because Mamba provides implicit position information (Section 2; Section 6.5).

- Why these design choices
  - Attention vs. Mamba ratio: Ablations at 1.3B parameters show both 1:3 and 1:7 hybrids outperform pure attention and pure Mamba in loss and downstream tasks, with negligible difference between 1:3 and 1:7; 1:7 is more compute‑ and memory‑efficient (Table 4; Figure 6).
  - MoE placement and size: Using MoE every other layer with 16 experts and top‑2 routing increases model capacity without growing per‑token compute excessively, striking a balance so the model fits on a single 80GB GPU using int8 weights while leaving memory for large inputs (Section 3.1).
  - Stabilizing Mamba at scale: Adding RMSNorm inside Mamba layers prevents loss spikes seen in 7B‑scale training (Figure 9; Section 6.4).
  - Positional encodings: The hybrid works comparably with and without RoPE; omitting explicit positions simplifies the stack (Table 8; Section 6.5).

- How it achieves memory and throughput improvements
  - KV cache scales with the number of attention layers and sequence length. With only 4 attention layers out of 32, Jamba’s KV cache is ~8× smaller than a same‑depth Transformer (Section 2; Table 1).
  - Mamba layers process sequences with a recurrent state rather than pairwise attention, making long sequences cheaper. Increasing the Mamba fraction improves throughput, especially at long contexts (Section 2; Figure 3b).

- Training and data
  - Trained on NVIDIA H100 GPUs with a framework supporting FSDP, tensor/sequence parallelism, and expert parallelism (Section 4).
  - Data is an in‑house mix of web, books, and code (last updated March 2024) with quality filtering and deduplication (Section 4).

- Practical fit
  - With int8 weights, the released 12B‑active/52B‑available model fits on a single 80GB GPU and supports very long inputs. Figure 2 shows double the maximum context length of Mixtral‑8x7B and 7× Llama‑2‑70B on a single A100 80GB card.

Definitions of less common terms used above:
- `Mamba layer`: a state‑space model layer that maintains a compact hidden state summarizing the sequence as it scans it. Unlike attention, it does not store a token‑by‑token KV cache.
- `Mixture‑of‑Experts (MoE) MLP`: an MLP replaced by a collection of `n` expert MLPs. A learned router selects the top‑`K` experts per token. “Available parameters” counts all experts; “active parameters” counts only the selected experts per token.
- `KV cache`: the per‑token keys and values stored by attention layers so the model can attend to prior tokens during generation. Memory grows with sequence length and number of attention layers.

## 4. Key Insights and Innovations
- Hybrid Attention–Mamba architecture that scales
  - What’s new: A production‑grade interleaving of attention and Mamba with only 4 out of 32 layers using attention in the released model (Figure 1; Section 3.1).
  - Why it matters: It keeps Transformer‑like capabilities while dramatically shrinking the KV cache and increasing long‑context throughput (Table 1; Figures 2–3).

- MoE integrated into a hybrid SSM stack
  - What’s new: MoE is applied to every other MLP in a hybrid Attention–Mamba model with 16 experts and top‑2 routing (Section 3.1).
  - Significance: It pushes total capacity to 52B parameters while keeping only 12B active per token, enabling single‑GPU deployment and strong task performance (Table 1; Table 2).
  - Empirical support: At 7B scale, adding MoE to the hybrid improves multiple benchmarks and log‑probabilities (Table 7).

- Evidence for complementary strengths: attention aids in‑context learning, Mamba aids efficiency
  - Observation: Pure Mamba underperforms on tasks requiring strict output format adherence and few‑shot induction (IMDB, QuAC, NarrativeQA), while the hybrid matches the attention‑only model (Table 6).
  - Mechanism: Visualization reveals “induction‑like” attention heads in the hybrid that reference prior label tokens in few‑shot prompts (Figure 8). This supports the hypothesis that attention layers inject in‑context learning behaviors that Mamba alone struggles to acquire (Section 6.2).

- Long‑context capability without explicit positional encodings
  - Finding: The hybrid model works similarly with and without RoPE; Mamba layers likely provide implicit position signals (Table 8).
  - Practical impact: One less moving part to tune and scale at ultra‑long contexts (Section 6.5).

## 5. Experimental Analysis
- Evaluation setup
  - Benchmarks span common sense reasoning, comprehension, math/code, and aggregate suites: HellaSwag, WinoGrande, ARC‑E/ARC‑C, PIQA, BoolQ, QuAC, GSM8K (3‑shot CoT), HumanEval, NQ (closed‑book), TruthfulQA, MMLU, and BBH (Section 5.1; Table 2).
  - Long‑context tests include:
    - Synthetic “needle in a haystack” retrieval up to 256K tokens (Figure 4).
    - Naturalistic few‑shot classification with thousands of in‑context examples across TREC‑Fine, NLU Intent, Banking77, CLINC150 (Figure 5).
    - Long‑document QA (3‑shot) on NarrativeQA, LongFQA, NQ, CUAD, SFiction (Table 3).
  - Throughput measured end‑to‑end (encode+decode) with: (a) single A100‑80GB, int8, 8K context, variable batch; (b) four A100s, no quantization, variable context up to 128K (Section 3.2; Figure 3).
  - Training infrastructure on H100s; proprietary in‑house data (Section 4).

- Main quantitative results
  - Memory and fit:
    - KV cache at 256K tokens: “Jamba: 4GB vs. Mixtral: 32GB vs. Mistral 7B: 32GB vs. Llama‑2 6.7B: 128GB” (Table 1).
    - On an A100‑80GB, Jamba supports about 2× the context of Mixtral and 7× Llama‑2‑70B (Figure 2).
  - Throughput:
    - At long contexts, Jamba reaches up to 3× Mixtral’s throughput; Llama‑2‑70B does not fit at 128K (Figure 3b).
    - With 8K input, int8, single A100, Jamba handles larger batches and achieves ~3× the throughput of Mixtral (Figure 3a).
  - Standard benchmarks (Table 2; selected highlights):
    - Reasoning/Comprehension: Jamba is competitive with Mixtral and Llama‑2‑70B. For example, HellaSwag 87.1 (Jamba) vs 86.7 (Mixtral), 85.3 (Llama‑2‑70B); BoolQ 88.2 (Jamba) vs 88.4 (Mixtral), 85.0 (Llama‑2‑70B).
    - Aggregates: MMLU 67.4 (Jamba) vs 70.6 (Mixtral) and 69.8 (Llama‑2‑70B); BBH 45.4 (Jamba) vs 50.3 (Mixtral).
    - Code: HumanEval pass@1 29.3 (Jamba) vs 34.8 (Mixtral).
    - Takeaway: Jamba is often close to or matching strong baselines but lags on some aggregate and coding metrics.
  - Long‑context capability:
    - Needle‑in‑a‑haystack: near‑perfect retrieval depth across lengths up to 256K (Figure 4).
    - Many‑shot classification: Jamba outperforms Mixtral on TREC‑Fine and Banking77 with thousands of in‑context examples; parity on NLU Intent and CLINC150 (Figure 5).
    - Long‑document QA: Average F1 0.44 (Jamba) vs 0.43 (Mixtral) across LongFQA, CUAD, NarrativeQA, NQ, SFiction (Table 3).

- Ablations and diagnostics
  - Hybrid beats pure attention or pure Mamba at 1.3B/250B tokens on multiple metrics, with little difference between 1:3 and 1:7 ratios (Table 4; Figure 6).
  - At 7B/50B tokens, hybrid still outperforms both pure variants on several metrics and in training loss (Table 5; Figure 7).
  - Pure Mamba fails on format‑sensitive few‑shot tasks (IMDB 48.8 vs 84.1 attention; hybrid 90.9), while hybrid matches attention‑only (Table 6).
  - Adding MoE improves the 7B hybrid across OLLM, HellaSwag, WinoGrande, NQ, and log‑probabilities (Table 7).
  - Training stabilization: adding RMSNorm inside Mamba layers eliminates loss spikes at 7B scale (Figure 9).
  - Positional encoding ablation: hybrid with and without RoPE shows nearly identical results (Table 8).

- Do the experiments support the claims?
  - Yes for efficiency and long‑context capability: the memory/throughput advantages are directly measured (Table 1; Figures 2–3) and the model retrieves information up to 256K tokens (Figure 4).
  - Mixed for pure task quality: Jamba is broadly competitive but not uniformly best; it trails Mixtral on MMLU, BBH, and HumanEval (Table 2).
  - The ablations substantiate design choices (attention:Mamba ratio, need for a few attention layers for in‑context learning, MoE benefits, normalization for stability).

## 6. Limitations and Trade-offs
- Model alignment and safety
  - The released model is a pretrained base without alignment, instruction‑tuning, or moderation; it should not be deployed to end users without further adaptation (Important notice beneath Figure 1; Section 2).

- Quality vs. efficiency
  - Using only 4 attention layers shrinks the KV cache but may limit peak performance on some knowledge‑heavy or reasoning benchmarks relative to larger attention‑only models (Table 2: MMLU, BBH).

- Scale of ablations
  - Several ablation insights come from smaller models (1.3B and 7B trained on 50B–250B tokens). While indicative, they are not full‑scale replications of the 12B‑active/52B‑available model (Section 6).

- Data transparency
  - The training dataset is in‑house and not publicly detailed beyond high‑level composition (web/books/code) and March 2024 cut‑off (Section 4). This may limit reproducibility and bias audits.

- Software ecosystem maturity
  - The hybrid stack has not yet benefited from the years of engineering optimizations available for pure Transformers; further speedups are expected, but not yet realized (Section 3.2).

- Potential behavior gaps
  - Pure Mamba struggles with in‑context format following; the hybrid fixes this with a small number of attention layers (Section 6.2). Edge cases that require heavy cross‑token relational reasoning might still benefit from more attention capacity.

## 7. Implications and Future Directions
- Field impact
  - Demonstrates that attention–SSM hybrids can match Transformer‑level accuracy at similar active parameter counts while unlocking ultra‑long contexts and higher throughput. This broadens the design space beyond attention‑only models for long‑document applications.

- Practical applications
  - Long‑form retrieval‑augmented generation, legal and financial document analysis, multi‑session conversational histories, multi‑file code comprehension, and any workload where 100K–256K contexts reduce external retrieval complexity or simplify system design.

- Research avenues enabled
  - Architecture search over attention:Mamba ratios, expert frequency (`e`), expert count (`n`), and top‑K routing to optimize specific latency/memory/quality targets (Section 2’s design space).
  - Understanding emergent in‑context learning in hybrids: mapping which minimal number/placement of attention layers suffice and how “induction heads” arise (Figure 8; Section 6.2).
  - Training theory and stabilization for large SSMs: normalization schemes, initialization, and optimization to prevent loss spikes (Figure 9).
  - Improved long‑context evaluations and real‑world workloads: beyond synthetic needles and aggregate F1, measuring downstream task latency, cost, and reliability at 128K–1M tokens (Sections 3.2, 5.2).
  - System optimization: kernels and runtimes specialized for Mamba and hybrid stacks to further widen the throughput gap over attention‑only models (Section 3.2).

Quoted highlights for quick reference:
- “KV cache (256K, 16‑bit): Jamba 4GB vs. Mixtral 32GB vs. Mistral 32GB vs. Llama‑2 6.7B 128GB.” (Table 1)
- “Jamba enables 2× Mixtral and 7× Llama‑2‑70B context length on a single A100‑80GB.” (Figure 2)
- “At 128K tokens, Jamba’s throughput is 3× Mixtral; Llama‑2‑70B does not fit.” (Figure 3b)
- “Hybrid outperforms pure attention and pure Mamba in ablations; 1:3 and 1:7 are similar—choose 1:7 for efficiency.” (Table 4; Figure 6)
- “MoE improves the hybrid across multiple metrics at 7B scale.” (Table 7)
- “Near‑perfect needle retrieval up to 256K tokens despite only 4 attention layers.” (Figure 4)
- “Adding RMSNorm inside Mamba layers prevents loss spikes at scale.” (Figure 9)
- “Omitting explicit positional encodings yields comparable quality.” (Table 8)
