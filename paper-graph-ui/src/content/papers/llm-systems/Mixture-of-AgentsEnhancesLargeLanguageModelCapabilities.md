# Mixture-of-Agents Enhances Large Language Model Capabilities

**ArXiv:** [2406.04692](https://arxiv.org/abs/2406.04692)

## 🎯 Pitch

This paper introduces the Mixture-of-Agents (MoA) framework, a novel methodology where multiple large language models (LLMs) collaborate by reading and improving on each other’s outputs through layered aggregation and synthesis—all via prompting alone. MoA achieves state-of-the-art performance on alignment and instruction-following benchmarks by harnessing the diverse strengths of existing LLMs, outperforming even closed-source giants like GPT-4o, while also enabling greater cost-effectiveness and flexibility without any additional training. This approach signals a powerful shift away from monolithic models, making superior LLM capabilities more accessible, scalable, and interpretable across the AI ecosystem.

---

## 1. Executive Summary
This paper introduces Mixture-of-Agents (MoA), a layered way to combine multiple large language models (LLMs) so that each model reads and improves on the others’ outputs rather than working in isolation. By iteratively aggregating and synthesizing multiple responses, MoA achieves state-of-the-art results on several alignment and instruction-following benchmarks while remaining cost- and latency-competitive (Table 2, Figure 5).

## 2. Context and Motivation
- Problem addressed
  - There are many strong but differently skilled LLMs (some better at instruction following, others at coding, etc.). The question is how to harness their complementary strengths without retraining or modifying model internals. Section 1 frames this as leveraging the “collective expertise” of multiple LLMs.

- Why this matters
  - Training ever-larger single models is extremely costly (“several trillion tokens” and large compute; Section 1). If coordination among existing models can extract more capability, we can improve quality without retraining, and potentially reduce cost and latency.

- Gap in prior work
  - Prior multi-model approaches often:
    - Select one best output via ranking or routing (model ensemble, reranking; Section 4.2), which discards information in non-selected candidates.
    - Require training specialized routers or fusion models (e.g., GENFUSER, or reward-guided routing; Section 4.2).
    - Rely on interactive multi-agent debates, which introduce complex protocols and latency (Section 4.2).
  - Standard Mixture-of-Experts (MoE) mechanisms work inside a single neural network layer with gating over experts, not at the model level (Section 2.3).

- Positioning
  - MoA is a prompt-only, model-level ensemble: multiple full LLMs produce candidate answers; other LLMs then read all prior answers and synthesize an improved response. No training or parameter access is required, and models can be swapped in and out (Section 2.2). The paper also articulates and tests a general phenomenon it calls “collaborativeness,” where LLMs improve when given other models’ outputs (Figure 1).

## 3. Technical Approach
MoA is a layered architecture where each layer contains several “agents” (LLMs) that read the same input and all outputs from the previous layer, then generate new outputs that are aggregated in the next layer (Figure 2).

- Core concepts
  - `Proposers`: models that generate candidate responses to the user prompt (Section 2.1).
  - `Aggregators`: models that read multiple candidates and produce a refined, single response (Section 2.1).
  - `Collaborativeness`: the empirical property that an LLM tends to produce higher-quality output when it sees outputs from other models, even if those are weaker (Figure 1; Sections 1 and 2.1).

- Notation and data flow (Section 2.2, Equation (1))
  - Let `l` be the number of MoA layers and `n` the number of agents per layer.
  - At layer `i`, the agents `A_{i,1}, …, A_{i,n}` each generate an output from the input text `x_i`. All outputs are combined via an Aggregate-and-Synthesize prompt (Table 1), denoted by `⊕`. The prompt explicitly tells the aggregator to synthesize, critique, and refine, rather than copy:
    > “Your task is to synthesize these responses into a single, high-quality response… critically evaluate… not simply replicate the given answers…” (Table 1)
  - Formally:
    - `y_i = ⊕_{j=1..n} [A_{i,j}(x_i)] + x_1`
    - `x_{i+1} = y_i`
    - Here `+` is text concatenation, and `⊕` means “apply the synthesis prompt to the set of model outputs.”

- Practical execution (Section 2.2)
  - Only one LLM is needed in the last layer to produce the final response (i.e., the final aggregator reads all prior text and outputs one answer).
  - Agents can be reused across layers (same model can appear in multiple layers) and within a layer (same model sampled multiple times with temperature). The latter is called the “single-proposer” variant.

- Design choices and rationale
  - Layered aggregation rather than one-shot selection
    - Figure 4a shows that generating a new synthesized output (MoA) outperforms an LLM-based ranker that merely selects the best proposer response (LLM-Ranker prompt in Table 5). This indicates real fusion, not just picking the best candidate.
  - Diversity and number of proposers
    - Table 3 shows performance rises with the number of proposers `n` and that diverse models (“multiple-proposer”) outperform multiple samples from the same model (“single-proposer”).
  - Model roles (who proposes vs. who aggregates)
    - Table 4 reveals specialization: some models are better aggregators (e.g., `Qwen1.5-110B-Chat`) while others excel as proposers (e.g., `WizardLM-8x22B`). This informs which models to place in the final aggregation layer.

- Analogy to Mixture-of-Experts (Section 2.3, Equation (2))
  - Standard MoE: `y_i = sum_j G_{i,j}(x_i) E_{i,j}(x_i) + x_i`, where a learnable gate `G` weights expert networks `E`. This requires internal access and training.
  - MoA: performs expert selection and fusion at the model level using natural language prompts; no gating network or weight updates; full LLMs act as experts and the aggregator LLM plays the role of both gate and combiner via instruction-following.

- Example pipeline
  1) Layer 1 (`n` proposers): each generates an answer to the user prompt.
  2) Layer 2 (`n` agents): each reads all Layer-1 answers plus the original prompt and produces improved candidates.
  3) Final layer (one aggregator): reads all prior candidates and outputs the final response (Figure 2).

## 4. Key Insights and Innovations
- Demonstration of “collaborativeness” across modern LLMs (Figure 1; Section 2.1)
  - Novel observation: LLMs score higher when they see other models’ answers—even weaker ones. This is not obvious a priori and motivates the whole MoA approach.
  - Significance: it justifies multi-LLM pipelines without model retraining.

- Layered synthesis (MoA) rather than single-stage reranking (Sections 2.2 and 3.3; Figure 4a)
  - Innovation: iterative aggregation layers give progressively better responses (Figure 4a trends up from Layer 1 to later layers).
  - Why it matters: Ablation against a strong LLM-based ranker baseline shows generation + fusion outperforms selection-only, indicating MoA extracts signal from multiple candidates.

- Empirical role specialization and diversity as design levers (Section 3.3; Tables 3–4)
  - New capability: choose which models to place as proposers vs aggregators and how many proposers to use for a budget-accuracy target.
  - Importance: practical recipes—e.g., use more diverse proposers and pick a strong aggregator—consistently improve results (Table 3).

- Cost/latency–quality Pareto frontier for multi-LLM inference (Section 3.4; Figure 5)
  - Contribution: operational analysis tying LC win rate to both dollar cost and TFLOPs (as a latency proxy), showing MoA and its “Lite” variant can be more cost-effective than single large closed models at similar quality levels.

## 5. Experimental Analysis
- Evaluation methodology (Section 3.1)
  - Benchmarks
    - AlpacaEval 2.0: 805 real-use instructions; compares each model’s answer against a reference answer using a GPT‑4 judge, with a “length-controlled” metric (`LC win rate`) to remove length bias.
    - MT-Bench: multi-turn questions graded by GPT‑4.
    - FLASK: 12 fine-grained skill scores (e.g., correctness, factuality, robustness).
  - Models in MoA
    - Open-source proposers: `Qwen1.5-110B-Chat`, `Qwen1.5-72B-Chat`, `WizardLM-8x22B`, `LLaMA-3-70B-Instruct`, `Mixtral-8x22B-v0.1`, `dbrx-instruct` (Section 3.1).
    - Default MoA has 3 MoA layers; final aggregator is `Qwen1.5-110B-Chat`.
    - Variants:
      - `MoA w/ GPT-4o`: same pipeline but uses `GPT-4o` as the final aggregator.
      - `MoA-Lite`: 2 layers; final aggregator `Qwen1.5-72B-Chat` for lower cost.

- Main results
  - AlpacaEval 2.0 (Table 2a)
    > `MoA w/ GPT-4o`: LC win 65.7 ± 0.7%, Win 78.7 ± 0.2%  
    > `MoA` (open-source only): LC win 65.1 ± 0.6%, Win 59.8 ± 0.3%  
    > `MoA-Lite`: LC win 59.3 ± 0.2%, Win 57.0 ± 0.7%  
    > `GPT-4 Omni (05/13)`: LC win 57.5%  
    - Takeaway: MoA (open-source only) surpasses GPT‑4 Omni by 7.6 LC win points; with GPT‑4o as final aggregator, it peaks at 65.7 LC win.
  - MT-Bench (Table 2b)
    > `MoA w/ GPT-4o`: 9.40 ± 0.06 (1st turn 9.49, 2nd 9.31)  
    > `MoA`: 9.25 ± 0.10 (1st 9.44, 2nd 9.07)  
    > `GPT-4 Turbo (04/09)`: 9.31  
    > `GPT-4 Omni (05/13)`: 9.19  
    - Takeaway: MoA variants are competitive at the top of the leaderboard; improvements are smaller here because many models already score >9/10.
  - FLASK (Figure 3)
    - MoA improves over its own final aggregator (`Qwen1.5-110B-Chat`) on robustness, correctness, efficiency, factuality, commonsense, insightfulness, and completeness. It also beats GPT‑4 Omni on correctness, factuality, insightfulness, completeness, and metacognition. It is slightly worse on conciseness.

- Ablations and mechanism checks
  - MoA vs LLM-Ranker (Figure 4a; Table 5 for the ranker prompt)
    - MoA outperforms the ranker baseline across layers, indicating the final aggregator is not just picking the best candidate but combining them.
  - Similarity-to-quality correlation (Figure 4b, Appendix Figure 6)
    - Within each prompt, the aggregator’s final answer is more similar (BLEU, TF-IDF, or Levenshtein) to the highest-quality proposer answers; the Spearman correlation between similarity and win rate is positive across layers and aggregators. This supports the claim that the aggregator preferentially incorporates better parts of the candidates.
  - Number and diversity of proposers (Table 3)
    > With 2 layers and `Qwen1.5-110B-Chat` as final aggregator:  
    > n=6: multiple-proposer 61.3% vs single-proposer 56.7%  
    > n=3: 58.0% vs 56.1%  
    > n=2: 58.8% vs 54.5%  
    > n=1: 47.8% (same)  
    - Takeaway: more proposers help; diversity helps more than repeated samples from one model.
  - Role specialization (Table 4)
    > As aggregator vs as proposer (2-layer setup)  
    > `Qwen1.5-110B-Chat`: 61.3% vs 56.7% (strong aggregator)  
    > `WizardLM-8x22B`: 52.9% vs 63.8% (strong proposer, weaker aggregator)  
    > `LLaMA-3-70B-Instruct`: 45.0% vs 60.6% (strong proposer, weaker aggregator)
  - Case studies (Tables 6–7)
    - When at least one proposer is very strong, the final answer closely integrates its content and boosts the overall preference to 0.99 (Table 6).  
    - Even when all proposers are mediocre, the aggregator can still assemble a better-than-any-single-candidate answer (final preference 0.33 vs near 0.0–0.16 for inputs; Table 7).
  - Reasoning benchmark (MATH) (Table 8)
    - Across aggregators, accuracy improves with additional MoA layers (e.g., `Qwen1.5-110B-Chat`: 0.500 → 0.570 → 0.576), suggesting applicability beyond pure instruction following.

- Budget and latency analysis (Section 3.4; Figure 5)
  - Cost vs LC win rate (Figure 5a)
    - Shows a Pareto frontier: for a given performance, there are cheaper MoA variants than large closed models.  
    - The paper reports “MoA-Lite can match GPT‑4o’s cost while achieving higher quality” and “outperforms GPT‑4 Turbo by ~4% while being more than twice as cost-effective.”
  - TFLOPs vs LC win rate (Figure 5b)
    - TFLOPs used as a latency proxy. MoA variants lie on the Pareto front. (Caveat noted in caption: TFLOPs for GPT‑4 are estimated from community rumors of an 8×220B MoE.)

- Do the experiments support the claims?
  - Yes, across three benchmarks and multiple ablations, the data consistently shows: (a) layered synthesis beats ranking; (b) diversity and more proposers help; (c) specializations matter; (d) strong, consistent gains on AlpacaEval and meaningful gains on FLASK/MT‑Bench. The MATH results also suggest generality to reasoning tasks.

## 6. Limitations and Trade-offs
- Latency and user experience (Limitations paragraph in Section 5)
  - The final output cannot start streaming until the final aggregation layer completes, increasing Time to First Token (TTFT). The paper suggests mitigating by reducing layers or doing chunk-wise aggregation in future work.

- Verbosity vs conciseness (Figure 3)
  - MoA tends to be less concise than baselines. The synthesis process encourages completeness, which can lengthen answers.

- Dependence on aggregator quality (Tables 2, 4)
  - Performance heavily depends on a strong final aggregator. Poor aggregators degrade results (e.g., `dbrx-instruct` as aggregator yields 41.5% in Table 4).

- Evaluation biases
  - Benchmarks like AlpacaEval and MT‑Bench rely on a GPT‑4-based judge. While LC win rate controls for length (Section 3.1), cross-model judging can introduce systemic biases shared with the judge model.

- Cost and compute
  - Although MoA can be cost-effective at a given quality (Figure 5), it still runs multiple models per query. Cost and TFLOPs rise with the number of layers and proposers (Table 3; Figure 5), so careful design is required.

- Architectural assumptions
  - MoA assumes models can be prompted to act as synthesizers without training. Some tasks might benefit from a learned fusion model or task-specific aggregation prompts, which MoA does not include.

- Scope
  - The paper focuses on single-turn prompts and general instruction following. Multi-turn, tool-using, or tightly constrained generation settings may require modified aggregation prompts or coordination strategies.

## 7. Implications and Future Directions
- How this changes the landscape
  - MoA shows that model-level composition—without training—can surpass single-model performance and even top closed models in some alignment benchmarks (Table 2a). This reframes progress from “build a bigger single LLM” to “compose strong, diverse LLMs effectively.”

- Practical applications
  - High-stakes drafting (policy, legal, technical writing) where completeness and factuality matter; MoA can produce more robust, cross-checked outputs (Figure 3).  
  - Domain aggregation: mix domain experts (code, math, medical) as proposers and use a general aggregator to produce a unified answer.  
  - Cost-aware deployments: use `MoA-Lite` to hit a target quality at lower dollar/latency budgets (Figure 5).

- Follow-up research directions
  - Adaptive routing: learn to choose the number and identity of proposers per prompt to minimize cost while preserving quality (extends Section 3.4).  
  - Streaming/low-TTFT aggregation: chunk-wise or incremental synthesis to reduce perceived latency (Limitations in Section 5).  
  - Task-specific aggregation prompts: customize Table 1’s synthesis instruction by task (e.g., safety-critical domains).  
  - Hybrid training: fine-tune a lightweight aggregator model on MoA transcripts to further improve fusion while retaining most of the zero-shot flexibility.  
  - Reliability analyses: formal studies on when and why “collaborativeness” holds (Figure 1) and failure cases where misleading proposer content might bias the aggregator.

Overall, this paper provides both a practical recipe and a set of design principles—use diverse proposers, pick a strong aggregator, and aggregate in layers—that together deliver consistent gains across strong baselines while offering an attractive cost–quality–latency trade-off.
