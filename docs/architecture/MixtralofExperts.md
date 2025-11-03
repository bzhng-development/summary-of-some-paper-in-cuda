# Mixtral of Experts

**ArXiv:** [2401.04088](https://arxiv.org/abs/2401.04088)

## 🎯 Pitch

Mixtral 8x7B is a groundbreaking sparse Mixture-of-Experts (SMoE) language model that routes each token to only 2 out of 8 experts per layer, activating just 13B parameters per token out of 47B total. This innovative architecture achieves or surpasses the performance of much larger dense models like Llama-2-70B and GPT-3.5 across key benchmarks—including code, mathematics, and multilingual tasks—while delivering drastically lower inference costs. By providing state-of-the-art open weights and efficient deployment tools, Mixtral paves the way for more accessible, scalable, and cost-effective advanced language AI.

---

## 1. Executive Summary
Mixtral 8x7B is a sparse Mixture‑of‑Experts (SMoE) transformer that replaces each feed‑forward block with 8 experts and routes each token to only 2 of them per layer. By activating 13B parameters per token while holding 47B total (“sparse”) parameters, it matches or beats much larger dense models (notably Llama‑2‑70B and GPT‑3.5) on many benchmarks, especially code, math, and multilingual tasks (Table 2, Table 3, Table 4). This matters because it delivers near‑frontier capability at significantly lower inference compute.

## 2. Context and Motivation
- Problem gap
  - Scaling language models has traditionally required activating all parameters for every token, making inference costly. This limits deployability and throughput, especially for long contexts and interactive systems.
  - Conditional computation—only using some parameters per token—has been explored in earlier Mixture‑of‑Experts systems, but there has not been an open‑weights model that simultaneously delivers competitive general performance, strong math/code/multilingual ability, 32k‑token context, and practical inference tooling.
- Why it matters
  - Real‑world: Lower compute per token translates to faster responses at low batch sizes and higher throughput at large batch sizes (Abstract; “Size and Efficiency,” p. 4), enabling cost‑effective deployment.
  - Scientific: Demonstrates that SMoE can scale capability without scaling per‑token compute, and provides routing analyses that inform future MoE design (Section 5, pp. 6–8).
- Prior approaches and shortcomings
  - Dense transformers (e.g., Llama‑2‑70B) activate all parameters per token—high quality but expensive.
  - Prior routed/MoE models (e.g., GShard) used conditional computation but did not replace every FFN block with MoE and/or were not available as high‑quality open weights. Mixtral positions itself as a fully open, high‑performing SMoE with efficient kernels and deployment integrations (Abstract; Section 2; “Code” and “Webpage” links).
- Positioning
  - Mixtral uses the Mistral 7B architecture as a base but replaces FFNs with MoE and supports a fully dense 32k context (Section 2, Table 1). It aims for best‑in‑class open performance with only 13B active parameters, demonstrating outsized gains on math, code, and multilingual tasks (Abstract; Table 2; Table 4).

## 3. Technical Approach
At a high level, Mixtral is a decoder‑only transformer where each layer’s feed‑forward network (FFN) is replaced by a Mixture‑of‑Experts (MoE) block.

Key components and how they work:
- Sparse Mixture‑of‑Experts (SMoE)
  - Definition: An MoE layer contains multiple alternative sub‑networks (“experts”). A small “router” chooses which experts should process each token, so only a subset run per token.
  - In Mixtral, “each layer is composed of 8 feedforward blocks (i.e. experts). For every token, at each layer, a router network selects two experts to process the current state and combine their outputs.” (Abstract; Figure 1)
  - The gating function: compute logits `x · Wg`, keep the Top‑K entries (here K=2), apply a Softmax over those K values to get weights, then compute the weighted sum of the selected experts’ outputs (Section 2.1). In notation (p. 2–3):
    - `G(x) := Softmax(TopK(x · Wg))`
    - `y = Σ_i Softmax(Top2(x · Wg))_i * SwiGLU_i(x)`
  - Why it reduces compute: with 8 experts per layer but K=2, only 2 expert FFNs run per token per layer, so active compute scales with K, not with total experts (Section 2.1).
- Experts
  - Each expert is a standard FFN using `SwiGLU` (a gated MLP activation variant) as in a vanilla transformer (Figure 1; p. 3).
- Router and load balancing
  - The router picks Top‑2 experts per token. Efficient execution uses specialized GPU kernels (Megablocks) that cast MoE FFNs as sparse matrix multiplications and handle variable token‑to‑expert assignments (Section 2.1, p. 2–3).
  - Distribution across GPUs uses Expert Parallelism (EP): tokens assigned to an expert are shipped to that expert’s device and returned after computation (Section 2.1). EP raises load‑balancing challenges because some experts may receive many more tokens; Mixtral discusses this explicitly (p. 2–3).
- Model size and context
  - Architecture hyperparameters are in Table 1: `dim=4096`, `n_layers=32`, `n_heads=32`, `n_kv_heads=8`, `hidden_dim=14336`, `num_experts=8`, `top_k_experts=2`, `context_len=32768`, `vocab_size=32000`.
  - “Mixtral supports a fully dense context length of 32k tokens” (p. 2) and successfully retrieves information anywhere in that window (Figure 4, left).
- Parameter accounting
  - “Sparse parameter count” = total parameters across all experts (≈47B in Mixtral).
  - “Active parameter count” = parameters used per token (≈13B with K=2). This is proportional to inference compute (Section 2.1; “Size and Efficiency,” p. 4).
- Training and instruction tuning
  - Pretraining: multilingual data; 32k context (Abstract; Section 3.1).
  - Instruction tuning: supervised fine‑tuning (SFT) followed by Direct Preference Optimization (`DPO`, a preference‑learning method that aligns outputs to preferred responses) (Section 4).
- Inference stack
  - Optimized with `Megablocks` CUDA kernels and integrated into `vLLM`; deployable via `Skypilot` (Abstract).

Why these design choices:
- K=2 over 8 experts per FFN
  - Strong compute/quality trade‑off: increases total capacity without increasing per‑token compute beyond two experts (Section 2.1).
- SwiGLU experts
  - A well‑performing FFN variant used in Mistral 7B; keeps expert definition simple while focusing novelty on routing (Section 2; [18]).
- Dense 32k context
  - Enables long‑document use cases and is validated by retrieval accuracy and perplexity improvements with longer context (Figure 4).

Analogy:
- Imagine each token entering a “committee meeting” at each layer. There are 8 specialists (experts), but the token only consults the top 2, as chosen by a quick vote (router logits + Top‑2 + Softmax). The final advice is a weighted average of those two specialists’ recommendations.

## 4. Key Insights and Innovations
- High‑performing open SMoE at low active compute (fundamental)
  - Novelty: Fully open‑weights SMoE where “each token has access to 47B parameters, but only uses 13B active parameters during inference” (Abstract) and yet outperforms or matches much larger dense models (Table 2, Table 3).
  - Significance: Demonstrates state‑of‑the‑art open performance with much lower per‑token compute, showing that routed capacity can deliver outsized gains in code and math (Table 2; Figure 3).
- 32k dense context with strong retrieval and better perplexity (incremental but impactful)
  - Evidence: Figure 4 (left) shows “100% retrieval accuracy” on the Passkey task across positions and lengths; Figure 4 (right) shows perplexity on “proof‑pile” decreases as context increases.
  - Significance: Validates long‑range reasoning/retrieval ability at large context.
- Strong multilingual capability through scaled capacity and data upsampling (incremental)
  - Evidence: Table 4 shows Mixtral 8x7B beating Llama‑2‑70B on ARC‑C, HellaSwag, and MMLU in French, German, Spanish, and Italian (e.g., French MMLU 70.9% vs 64.3%).
  - Significance: Improved non‑English utility without sacrificing English performance.
- Router behavior is more syntactic than topical; high temporal locality (new analysis)
  - Evidence: Figure 7 shows expert assignment distributions look similar across varied domains (ArXiv, PubMed, PhilPapers), with only slight divergence for DM Mathematics. Figure 8 highlights tokens like “self” (Python) and “Question” (English) being routed consistently, suggesting syntactic cues. Table 5 quantifies locality: e.g., “first choice” expert repetition between consecutive tokens at layer 15 is 27.9% on ArXiv vs 12.5% random expectation; “first or second choice” repetitions reach 62–67% across sources (Table 5).
  - Significance: Guides systems design (caching, scheduling) and suggests that expert specialization may emerge around syntax/structure more than topic.

## 5. Experimental Analysis
- Evaluation protocol
  - Benchmarks span commonsense, world knowledge, reading comprehension, math, code, and popular aggregates (Section 3; Figure 2; Figure 3). Datasets include HellaSwag, WinoGrande, PIQA, SIQA, OpenBookQA, ARC‑Easy/Challenge, CommonsenseQA; NaturalQuestions, TriviaQA; BoolQ, QuAC; GSM8K, MATH; HumanEval, MBPP; MMLU, BBH, AGI‑Eval.
  - They re‑evaluate all baselines with a unified pipeline (Figure 2 caption). Noted differences: (1) MBPP uses the hand‑verified subset; (2) TriviaQA is evaluated without Wikipedia contexts (p. 5, “Evaluation Differences”).
- Main results (selected highlights)
  - Across broad benchmarks vs Llama‑2‑70B (Table 2):
    - `MMLU`: Mixtral 8x7B 70.6% vs 69.9%.
    - `MBPP (pass@1)`: 60.7% vs 49.8%.
    - `GSM8K (8‑shot maj@8)`: 74.4% vs 69.6%.
    - `HumanEval`: 40.2% vs 29.3%.
    - Mixtral trails slightly on some commonsense metrics: `HellaSwag` 84.4% vs 85.4%; `WinoGrande` 77.2% vs 80.4%.
  - Against GPT‑3.5 and Llama‑2‑70B (Table 3; different prompt shots than Table 2):
    - `MMLU`: 70.6% (Mixtral) vs 70.0% (GPT‑3.5) vs 69.9% (Llama‑2‑70B).
    - `HellaSwag (10‑shot)`: 86.7% vs 85.5% vs 87.1%.
    - `ARC‑Challenge (25‑shot)`: 85.8% vs 85.2% vs 85.1%.
    - `MBPP (pass@1)`: 60.7% vs 52.2% vs 49.8%.
    - `GSM‑8K (5‑shot)`: 58.4% vs 57.1% vs 53.6%.
    - For the instruction‑tuned variants, MT‑Bench: 8.30 (Mixtral‑Instruct) vs 8.32 (GPT‑3.5‑Turbo‑1106) (Table 3), and the LMSys Arena shows Elo 1121, beating Claude‑2.1, Gemini Pro, GPT‑3.5 versions, and Llama‑2‑70B‑chat (Figure 6).
  - Multilingual (Table 4):
    - Example (Spanish): ARC‑C 55.4% vs 50.5%; HellaSwag 77.6% vs 74.5%; MMLU 72.5% vs 66.0%.
  - Long context (Figure 4):
    - “100% retrieval accuracy regardless of the context length or the position” on Passkey; perplexity on proof‑pile decreases monotonically with more context.
  - Bias (Section 3.3; Figure 5):
    - “BBQ accuracy” 56.0% (Mixtral) vs 51.5% (Llama‑2‑70B).
    - BOLD shows higher average (more positive sentiment) and similar or lower variance (less intra‑group bias).
- Efficiency analysis (p. 4, “Size and Efficiency”)
  - Active parameters (13B) drive compute; memory is set by sparse parameters (47B), still below Llama‑2‑70B but higher than a 13B dense model.
  - SMoE introduces extra overhead from routing and memory loads; best for batched workloads where arithmetic intensity is higher.
- Do the experiments support the claims?
  - For code and math, the gains are substantial and consistent across both table settings (Tables 2–3; Figure 3). For general knowledge (MMLU), Mixtral is competitive or slightly better. Commonsense tasks are mixed: it wins some but trails on WinoGrande and sometimes HellaSwag depending on shot settings.
  - Long context tests convincingly demonstrate retrieval capability (Figure 4), though real‑task long‑context benchmarks are not reported here.
  - The bias analysis indicates a favorable trend (Figure 5) but is not exhaustive.
- Ablations and diagnostics
  - Section 5 provides a routing analysis rather than classic ablations. Key findings:
    - Expert selection is not strongly domain‑specialized (Figure 7), with limited divergence on DM Mathematics.
    - Strong temporal locality: repeated expert selections across consecutive tokens much higher than random—e.g., “first choice” repetitions at layer 15: 27–28% across domains vs 12.5% random; “first or second choice” 61–67% vs ≈46% random (Table 5; Figure 10).
  - Implications: locality may cause expert over‑subscription under EP but also enables caching (p. 7).

## 6. Limitations and Trade-offs
- Compute vs memory and overhead
  - While only 13B parameters are activated per token, all 47B parameters must be stored in memory; SMoE routing also introduces overhead and increased memory loads, so utilization is best at larger batches (p. 4, “Size and Efficiency”).
- Load balancing in Expert Parallelism
  - Router‑induced token clustering can overload some experts, creating bottlenecks; the paper flags this as a key systems challenge (p. 2–3). The routing analysis (Table 5) shows temporal locality that can exacerbate over‑subscription.
- Mixed performance on certain benchmarks
  - On some commonsense tasks (e.g., WinoGrande) Mixtral lags Llama‑2‑70B (Table 2). Reading‑comprehension plots (Figure 3) suggest Mixtral is not best in that category.
- Evaluation caveats
  - Differences in evaluation protocol—MBPP hand‑verified subset; TriviaQA without Wikipedia context—may affect comparability (p. 5).
- Limited details on training data and techniques
  - Pretraining is described broadly as multilingual with 32k context; granular dataset composition and training schedule are not detailed here.
- Expert specialization
  - Analyses suggest experts do not strongly specialize by topic (Figure 7). While not necessarily negative, it raises questions about how to intentionally induce useful specialization.

## 7. Implications and Future Directions
- Field impact
  - Mixtral shows that routed capacity can rival or exceed large dense models at a fraction of active compute. This is a compelling blueprint for future open models to scale via SMoE rather than pure density.
- Practical applications
  - Cost‑effective deployment for long‑context assistants, code generation, math problem solving, and multilingual agents—helped by vLLM integration and cloud deployment via Skypilot (Abstract).
- Research directions
  - Routing and specialization
    - Explore alternative gating (e.g., expert‑choice routing, load‑aware gates) to improve balance, reduce overhead, and possibly promote meaningful specialization (Section 2.1; [35]).
    - Leverage temporal locality for expert‑output caching and scheduling (p. 7; [11]).
  - Compute policies
    - Dynamic K (vary number of experts by token or layer) to trade quality/latency adaptively.
  - Long‑context capabilities
    - Move beyond synthetic passkey retrieval to evaluate and train for real‑task long‑context reasoning (document QA, multi‑document retrieval).
  - Safety and bias
    - Extend BBQ/BOLD analysis, combine with preference‑modeling (DPO) to steer behavior, and broaden multilingual fairness evaluation (Section 3.3; Section 4).
  - Systems
    - Better EP strategies and kernel optimizations to reduce router overhead and memory traffic; further integration into high‑throughput inference stacks.

Key citations from the paper’s content:
- “Each layer is composed of 8 feedforward blocks… a router network selects two experts…” (Abstract; Figure 1).
- “Each token has access to 47B parameters, but only uses 13B active parameters during inference.” (Abstract).
- “Mixtral supports a fully dense context length of 32k tokens.” (p. 2; Table 1).
- “100% retrieval accuracy” for passkey across positions/lengths (Figure 4 left); decreasing perplexity with longer context (Figure 4 right).
- Strong code/math gains and competitive MMLU (Table 2; Table 3; Figure 3).
- Multilingual gains across FR/DE/ES/IT (Table 4).
- Bias metrics: BBQ 56.0% vs 51.5% (Figure 5).
- Routing locality and lack of strong domain specialization (Figure 7; Figure 8; Table 5; Figure 10).
