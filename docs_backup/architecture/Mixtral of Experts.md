# Mixtral of Experts

**ArXiv:** [2401.04088](https://arxiv.org/abs/2401.04088)
**Authors:** Albert Q. Jiang, Alexandre Sablayrolles, Antoine Roux, Arthur Mensch, Blanche Savary, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Emma Bou Hanna, Florian Bressand, Gianna Lengyel, Guillaume Bour, Guillaume Lample, Lélio Renard Lavaud, Lucile Saulnier, Marie-Anne Lachaux, Pierre Stock, Sandeep Subramanian, Sophia Yang, Szymon Antoniak, Teven Le Scao, Théophile Gervet, Thibaut Lavril, Thomas Wang, Timothée Lacroix, William El Sayed
**Institutions:** 

## 🎯 Pitch

Mixtral 8x7B introduces a pioneering sparse Mixture-of-Experts design, selecting the top 2 out of 8 experts per input token, effectively leveraging 47B parameters while using only 13B per token at inference. This innovative architecture outperforms larger models like LLaMA-2-70B by enhancing accuracy and efficiency, making advanced AI more accessible and cost-effective across diverse applications, from multilingual tasks to complex math problems.

---

## 1. Executive Summary (2-3 sentences)
Mixtral 8x7B is a large language model that uses a sparse Mixture-of-Experts (SMoE) design: every layer contains 8 “experts,” but for each input token only the top 2 are activated and combined. This yields a model with access to 47B parameters while using only 13B per token at inference, enabling strong accuracy–efficiency trade-offs: across many benchmarks, Mixtral matches or surpasses much larger models such as LLaMA‑2‑70B and GPT‑3.5 (see Table 2 and Table 3).

## 2. Context and Motivation
- Problem/gap:
  - Scaling dense models (where all parameters process every token) quickly becomes compute- and memory-prohibitive. Even high-performing open-weight models like LLaMA‑2‑70B require 70B active parameters per token at inference.
  - There is a need for open, efficient models that deliver near state-of-the-art performance, handle long contexts, and work well across languages.
- Importance:
  - Practical: Lower active compute per token reduces inference latency and cost, enabling broader deployment (Section 1; “Size and Efficiency”).
  - Scientific: Demonstrates how conditional computation via SMoE can scale capacity without proportional inference cost (Section 2.1).
- Prior approaches and shortcomings:
  - Dense Transformers (e.g., LLaMA/Mistral 7B) use all parameters on every token—efficient at small scale, but expensive when scaling up.
  - Earlier MoE systems like GShard used MoE layers in alternating blocks and more elaborate second-expert routing; they showed promise but came with training and routing complexity (Section 2.1; comparison to GShard [21]).
- Positioning:
  - Mixtral generalizes the Mistral 7B architecture but replaces every feedforward block with an MoE layer, using K=2 of 8 experts per token (Table 1; Section 2.1).
  - It aims to set a new accuracy–efficiency point for open models, with 32k fully dense context support and strong multilingual and long-context performance (Introduction; Sections 3 and 3.1–3.2).

## 3. Technical Approach
Step-by-step overview
1. Base architecture: decoder-only Transformer
   - Mixtral follows Mistral 7B’s modifications but supports a fully dense 32k token context and replaces all feedforward sub-blocks with MoE layers (Section 2; Table 1).
   - Key hyperparameters (Table 1):
     - `dim=4096`, `n_layers=32`, `n_heads=32`, `head_dim=128`, `hidden_dim=14336`
     - `n_kv_heads=8` (for multi-query attention style key/value sharing)
     - `context_len=32768`, `vocab_size=32000`
     - `num_experts=8`, `top_k_experts=2`

2. Mixture-of-Experts (MoE) layer
   - Concept: An `expert` is a standard Transformer feedforward block (here implemented as `SwiGLU`, a gated feedforward variant) with its own parameters.
   - For each token at each layer, a lightweight `router` (a learned linear layer) produces a score for every expert. Only the top `K=2` experts are selected; their outputs are weighted and summed (Section 2.1; Figure 1).
   - Notation and mechanism (Section 2.1):
     - The router computes `x · Wg` (logits over experts), applies `TopK` to keep only the largest K logits, then a softmax over those:
       - `G(x) := Softmax(TopK(x · Wg))`
     - The output is the weighted sum of expert outputs:
       - `y = Σ_i Softmax(Top2(x · Wg))_i · SwiGLU_i(x)`
     - Intuition: For each token, the model “chooses” two specialists whose outputs are blended, rather than invoking all eight experts.

3. Active vs. total parameters
   - Because only 2 of 8 experts are active per token per layer, the compute scales with the `active` parameters (13B) while the model’s `sparse` total is larger (47B). This gives high capacity without paying the full inference cost of a dense 47B model (Section 2.1; Introduction).
   - Quote:
     > “Each token has access to 47B parameters, but only uses 13B active parameters during inference.” (Abstract; Section 2.1 elaborates the active vs. sparse distinction)

4. Efficient execution and parallelism
   - The paper relies on specialized kernels and parallelism strategies to make MoE fast and scalable:
     - `Megablocks` treats MoE FFNs as large sparse matrix multiplications to accelerate execution and handle variable token-to-expert routing (Section 2.1; [13]).
     - `Expert Parallelism (EP)` places different experts on different GPUs; tokens are routed to the device holding their selected experts and results are returned (Section 2.1). This raises load-balancing challenges because different experts may receive uneven loads.
   - Engineering note: Mixtral integrates with `vLLM` and `TensorRT-LLM/Triton` to run efficiently in open-source stacks (Section 1).

5. Training and fine-tuning
   - Pretraining: multilingual data with a 32k-token context window (Introduction; Section 3.1).
   - Instruction-tuning: a supervised fine-tuning (SFT) stage followed by `Direct Preference Optimization (DPO)`, a method that optimizes model behavior based on preferred responses in paired comparisons (Section 4; [25]).

6. Long-context behavior
   - The model is evaluated on a passkey retrieval task over very long prompts; results show perfect retrieval regardless of position or length (Section 3.2; Figure 4 left).

Clarifying a few terms
- `SwiGLU`: A feedforward block that uses a “swish”-like gate to modulate activations, often improving performance over ReLU/GeLU FFNs.
- `Active parameters`: parameters used for a specific token during inference (here 13B).
- `Sparse (total) parameters`: the full parameter set across all experts (here 47B), only a subset of which is active per token.
- `TopK`: operation that keeps only the K highest scores and masks the rest to −∞ before softmax, ensuring only K experts receive nonzero weights (Section 2.1).
- `Pass@1` (code): fraction of problems solved by the single top sample from the model.
- `maj@k` (math): accuracy after sampling k solutions and taking the majority vote.

Design choice rationale
- Choosing `K=2` of `n=8` experts balances capacity and compute: increasing `n` expands total capacity, while keeping `K` small keeps inference cost nearly constant (Section 2.1).
- Making every FFN an MoE layer (rather than every other block as in GShard) maximizes the benefit of conditional computation at all depths (Section 2.1).
- The 32k full-context support and multilingual upsampling are aimed at real-world workloads (long documents; multilingual usage) (Section 3.1; 3.2).

## 4. Key Insights and Innovations
1) All-FFN MoE with Top‑2 routing yields strong accuracy–efficiency gains
- What’s new: Every FFN is replaced by an MoE layer; the router picks 2 of 8 experts for each token (Figure 1; Section 2.1). This is more aggressive than designs that only place MoE in some layers.
- Why it matters: It achieves capacity comparable to much larger dense models but with only 13B active parameters per token. Results show Mixtral matches or beats LLaMA‑2‑70B and GPT‑3.5 on many benchmarks (Figure 2; Table 2; Table 3).
- Quote:
  > “Mixtral outperforms or matches Llama 2 70B on all benchmarks. In particular, it is vastly superior in mathematics and code generation.” (Figure 2 caption)

2) Long-context capability with robust retrieval
- What’s new: Fully dense 32k context training and testing on passkey retrieval (Section 3.2).
- Why it matters: Demonstrates reliable use of long prompts without positional brittleness.
- Quote:
  > “Mixtral achieves a 100% retrieval accuracy regardless of the context length or the position of the passkey in the sequence.” (Figure 4 left)

3) Strong multilingual performance via upsampling
- What’s new: Pretraining significantly upsamples multilingual data (Section 3.1).
- Why it matters: Mixtral outperforms LLaMA‑2‑70B on ARC‑Challenge, HellaSwag, and MMLU in French, German, Spanish, and Italian (Table 4).

4) Instruction tuning with DPO reaches top open-weight performance on human evals
- What’s new: SFT + DPO yields Mixtral‑Instruct scoring 8.30 on MT‑Bench (Section 4).
- Why it matters: Human evaluation (LMSys Arena) shows it surpasses GPT‑3.5‑Turbo, Claude‑2.1, Gemini Pro, and LLaMA‑2‑70B‑chat (Figure 6).

5) Router behavior is more syntax‑ than domain‑driven, with strong temporal locality
- What’s new: Analysis reveals experts aren’t cleanly specialized by topic; instead, routing correlates with syntactic patterns, and consecutive tokens often use the same experts (Section 5; Figures 7–8; Table 5; Figure 10).
- Why it matters: Suggests optimization opportunities (caching, better load balancing) and revises assumptions about how MoE specialization emerges.

## 5. Experimental Analysis
Evaluation setup
- Tasks and settings (Section 3):
  - Commonsense reasoning (0-shot): HellaSwag, Winogrande, PIQA, SIQA, OpenBookQA, ARC‑Easy/Challenge, CommonsenseQA
  - World knowledge (5-shot): NaturalQuestions, TriviaQA
  - Reading comprehension (0-shot): BoolQ, QuAC
  - Math: GSM8K (8-shot, `maj@8`), MATH (4-shot, `maj@4`)
  - Code: HumanEval (0-shot, pass@1), MBPP (3-shot, pass@1; hand-verified subset)
  - Aggregates: MMLU (5-shot), BBH (3-shot), AGIEval (3–5-shot)
  - Long-context: passkey retrieval; perplexity vs. context length (Figure 4)
  - Bias: BBQ, BOLD (Section 3.3; Figure 5)

Main quantitative results
- Overall comparative performance (Table 2; Figure 2; Figure 3):
  - Against LLaMA‑2‑70B, Mixtral 8x7B uses 5× fewer active parameters (13B vs. 70B) yet “outperforms or matches” on almost all benchmarks (Figure 3 caption).
  - Concrete differences (Table 2; Mixtral vs. LLaMA‑2‑70B):
    - Math and code standout:
      - MATH (`maj@4`): 28.4% vs. 13.8%
      - GSM8K (`maj@8`): 74.4% vs. 69.6%
      - HumanEval (0-shot, pass@1): 40.2% vs. 29.3%
      - MBPP (3-shot, pass@1): 60.7% vs. 49.8%
    - MMLU (5-shot): 70.6% vs. 69.9%
    - Commonsense (selected): ARC‑Challenge 59.7% vs. 56.5%; PIQA 83.6% vs. 82.6%
    - Reading comprehension is mixed; overall figures suggest smaller or no advantage there (Figure 3).
- Comparison with GPT‑3.5 on standard evals (Table 3):
  - Quote of headline numbers:
    > “MMLU: 70.6% (Mixtral) vs. 70.0% (GPT‑3.5); HellaSwag (10‑shot): 86.7% vs. 85.5%; ARC‑Challenge (25‑shot): 85.8% vs. 85.2%; MBPP: 60.7% vs. 52.2%; GSM‑8K (5‑shot): 58.4% vs. 57.1%; MT‑Bench: 8.30 (Mixtral‑Instruct) vs. 8.32 (GPT‑3.5‑Turbo‑1106).”
  - Note the two GSM‑8K results: Table 2 uses `maj@8` (74.4%); Table 3 reports 5‑shot single-sample (58.4%). The better `maj@k` result reflects ensembling via multiple samples.

- Multilingual (Table 4):
  - Mixtral beats LLaMA‑2‑70B in French/German/Spanish/Italian on ARC‑Challenge, HellaSwag, and MMLU.
  - Example (French): ARC‑C 58.2% vs. 49.9%; HellaSwag 77.4% vs. 72.5%; MMLU 70.9% vs. 64.3%.

- Long-context (Figure 4):
  - Passkey retrieval: 100% across positions and lengths.
  - Perplexity on proof-pile subset decreases monotonically as context increases, indicating the model uses added context effectively.

- Bias (Figure 5):
  - Quote:
    > “BBQ accuracy: 56.0% (Mixtral) vs. 51.5% (LLaMA‑2‑70B). BOLD sentiment score shows higher averages and similar or lower standard deviations in several categories.”
  - Interpretation: Higher average sentiment score = more positive sentiment; lower standard deviation = less intra-group bias variability.

- Routing analysis (Section 5; Figures 7–10; Table 5):
  - No strong expert specialization by domain across layers 0, 15, 31 (Figure 7).
  - Syntax-like patterns: visually, code indentation or repeated words route consistently to the same experts, especially at first/last layers (Figure 8).
  - Temporal locality: “same expert in consecutive tokens” is far above random at deeper layers—e.g., across domains at layer 15, “first or second choice” repetition is ~62–67% vs. ~46% expected by chance (Table 5; Figure 10).

Evaluation protocol notes and fairness
- The paper re-runs all LLaMA baselines with its own pipeline and notes two deviations:
  - MBPP uses the hand-verified subset.
  - TriviaQA is evaluated without providing Wikipedia contexts (Section 3, “Evaluation Differences”).
- These choices likely make MBPP results more reliable and TriviaQA stricter, but they also mean numbers may differ from other publications.

Do the experiments support the claims?
- Yes, for efficiency and accuracy: Tables 2–3 and Figures 2–3 consistently show Mixtral matching or exceeding LLaMA‑2‑70B and GPT‑3.5 on many tasks while using far fewer active parameters.
- The strongest evidence is in math, code, multilingual, and long-context retrieval (Table 2, Table 4, Figure 4).
- Results are more mixed in reading comprehension (Figure 3), which the paper acknowledges visually (lower relative advantage).
- The routing analysis provides mechanistic insight rather than just reporting scores (Section 5).

Missing ablations or robustness checks
- No ablation over `K` (number of experts per token) or “MoE everywhere” vs. “MoE in some layers.”
- No detailed analysis of router load balancing strategies, capacity factors, or spillover handling.
- No comparison to more recent proprietary models beyond GPT‑3.5; human evals (Figure 6) include Claude‑2.1 and Gemini Pro, but not GPT‑4-class models.

## 6. Limitations and Trade-offs
- Compute/memory and deployment:
  - Inference compute scales with active parameters (13B), but memory scales with the full sparse parameter count (47B). This is still less than 70B, but not “small.” SMoE adds routing overhead and more memory transactions (Section “Size and Efficiency”).
  - Quote:
    > “SMoEs introduce additional overhead due to the routing mechanism and increased memory loads… They are more suitable for batched workloads where one can reach a good degree of arithmetic intensity.” (Size and Efficiency)
- Parallelism and load balancing:
  - Expert Parallelism requires shuffling tokens across devices; poor load balance can bottleneck throughput (Section 2.1).
  - Temporal locality in routing (Table 5; Figure 10) can cause over-subscription of particular experts in batches, complicating EP scheduling (Section 5).
- Scope of multilingual evaluation:
  - Strong results are shown for four Western European languages; the paper does not report across a broader set (e.g., low-resource or non‑Indo‑European languages).
- Mixed performance on some categories:
  - Reading comprehension shows less consistent gains (Figure 3).
- Limited ablations:
  - No controlled experiment varying number of experts, K, or comparing alternate gating schemes beyond Top‑2 (Section 2.1 mentions alternatives but does not ablate).

## 7. Implications and Future Directions
- Field impact:
  - Mixtral validates SMoE as a practical path to high-capacity, cost-efficient open models: 47B total capacity with 13B active per token, outperforming or matching 70B dense baselines in many areas (Table 2; Figure 2).
  - Long-context reliability (32k) and strong multilingual results broaden the utility of open-weight LLMs for real-world deployments.
- Practical applications:
  - Cost-effective inference for coding assistants, math tutors, multilingual chatbots, and document-heavy workflows (legal, scientific, enterprise) where 32k contexts and high throughput matter.
  - Open Apache‑2.0 licensing enables commercial integration (Section 1).
- Engineering directions:
  - Exploit routing locality for caching or communication-aware scheduling (Section 5; “locality can be leveraged for caching” with reference [11]).
  - Improve EP load balancing using router-aware batching or dynamic capacity allocation.
  - Kernel and runtime improvements (e.g., vLLM + Megablocks + TensorRT‑LLM) to further reduce routing/memory overhead.
- Research directions:
  - Study expert specialization: Can training objectives or data assignment induce clearer semantic specialization, aiding controllability and interpretability?
  - Ablate `K`, number of experts, and placement (every layer vs. selective) to map the efficiency frontier.
  - Explore adaptive `K` per token or per layer (dynamic compute).
  - Extend multilingual evaluations and training to more languages and scripts, especially low-resource settings.
  - Investigate alignment methods beyond DPO for safety, bias reduction (BBQ/BOLD results are promising but limited in scope; Figure 5).

Overall, Mixtral 8x7B demonstrates that a well-engineered, fully‑MoE Transformer with Top‑2 routing can deliver top-tier performance among open models while substantially reducing active inference compute, with particularly strong gains in math, code, and multilingual tasks and robust long-context behavior.
