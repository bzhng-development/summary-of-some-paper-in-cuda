# Mixture-of-Agents Enhances Large Language Model Capabilities

**ArXiv:** [2406.04692](https://arxiv.org/abs/2406.04692)
**Authors:** Junlin Wang, Jue Wang, Ben Athiwaratkun, Ce Zhang, James Zou
**Institutions:** Not specified on arXiv

## 🎯 Pitch

The paper presents a novel Mixture-of-Agents (MoA) model that combines multiple large language models (LLMs) through layered synthesis without training, achieving state-of-the-art results on alignment-style benchmarks. This innovative approach enhances robustness and factuality, offering a cost-effective and scalable solution for deploying AI by integrating the strengths of existing models, marking a significant shift towards collaborative AI systems.

---

## 1. Executive Summary (2-3 sentences)
This paper introduces a simple but powerful way to combine multiple large language models (LLMs) without training: a layered Mixture-of-Agents (`MoA`) where each “agent” is an LLM that reads the other agents’ outputs and synthesizes a better response. With only prompting (no fine-tuning), `MoA` achieves state-of-the-art scores on alignment-style benchmarks, including a 65.1% length-controlled (LC) win rate on AlpacaEval 2.0 using only open-source models, surpassing GPT-4 Omni (57.5%) (Table 2a).

## 2. Context and Motivation
- Problem gap
  - Many high-quality LLMs exist, each with different strengths (e.g., reasoning, coding, instruction following). The open question is how to harness “collective expertise” across models without expensive joint training or complex infrastructure.
  - Scaling a single model further is extremely costly and data-hungry; combining existing models could be more practical.
- Why it matters
  - Real-world assistants often need robustness, factuality, and breadth. A collaborative approach can mitigate any single model’s blind spots.
  - Practically, organizations already call multiple APIs/models; a principled, effective way to aggregate them increases quality per dollar.
- Prior approaches and their limits
  - Model ensembles and reranking choose the “best” among candidate answers, but do not synthesize them into a superior response (Related Work §4.2). 
  - Multi-agent debate/discussion methods let agents talk, but typical setups can be complex, symmetric, or not optimized for final synthesis (§4.2).
  - Mixture-of-Experts (MoE) requires training and internal access to a model’s weights/activations; it is not directly applicable to black-box LLMs (§2.3).
- Positioning
  - `MoA` is a black-box, prompt-only analogue of MoE that runs at the model level: it stacks layers of LLM agents, each layer reading and combining all previous responses (§2.2, Figure 2). It is compatible with API-accessible models and requires no fine-tuning.

## 3. Technical Approach
Step-by-step mechanism (what actually happens):
1. Roles
   - `Proposers`: agents that generate diverse candidate answers. A good proposer adds useful details or perspectives even if its standalone answer would not “win.”
   - `Aggregators`: agents that read multiple candidate answers and synthesize them into a single, higher-quality response (§2.1).
2. Layered architecture
   - Organize agents into `l` layers; each layer `i` has `n` agents `A_{i,1} ... A_{i,n}` (Figure 2).
   - Layer 1 proposers generate responses independently to the user prompt `x1`.
   - From Layer 2 onward, each agent sees:
     - the original prompt `x1`, and
     - all responses produced by the previous layer, wrapped in an “Aggregate-and-Synthesize” instruction (Table 1).
   - Notation (Equation (1), §2.2, paraphrased):  
     - Let `A_{i,j}(x_i)` be agent j’s output on layer i’s input `x_i`.  
     - Concatenate the previous layer’s outputs and the original prompt, apply the aggregation prompt `⊕`, and pass that to agents in the next layer.  
     - The final output is produced by a single aggregator in the last layer to save cost.
3. The “Aggregate-and-Synthesize” prompt (Table 1)
   - A fixed instruction that tells the aggregator to: read the set of model responses, critically evaluate them (recognizing bias or errors), and synthesize a refined, accurate, comprehensive answer (not just repeat one candidate).
4. Single- vs. multiple-proposer settings
   - `Multiple-proposer`: each proposal is from a different LLM.
   - `Single-proposer`: generate multiple samples from the same LLM (e.g., using temperature) to create diversity (§2.2).
5. Selecting models for layers (§2.1)
   - Two criteria guide agent choice:
     - performance (place stronger models later as aggregators), and
     - diversity (use heterogeneous models as proposers to maximize complementary information).
6. Analogy to Mixture-of-Experts (§2.3)
   - In trained MoE, a learned gate mixes internal “experts” by weighting their activations (Equation (2)).  
   - In `MoA`, the “gate” is implicit: a general-purpose LLM reads natural-language candidate answers and decides what to keep, refute, or merge—purely via prompting. No weight changes are needed.
7. Default implementation (§3.1)
   - Proposers drawn from open-source models: `Qwen1.5-110B-Chat`, `Qwen1.5-72B-Chat`, `WizardLM-8x22B`, `LLaMA-3-70B-Instruct`, `Mixtral-8x22B-v0.1`, `dbrx-instruct`.
   - Three `MoA` layers by default; final aggregator is `Qwen1.5-110B-Chat`.  
   - Variants:
     - `MoA w/ GPT-4o`: uses GPT-4o as the final aggregator (quality-first variant).
     - `MoA-Lite`: two layers; final aggregator `Qwen1.5-72B-Chat` (cost-oriented).
8. How it behaves in practice (Case Study, Tables 6–7)
   - When some proposers are strong (Table 6, “Smooth” by Rob Thomas), the aggregator preserves high-quality facts (e.g., “12 weeks at the top of the Billboard Hot 100”) while combining style and structure across answers.
   - When no proposer is very strong (Table 7, “How do you become an author?”), the aggregator still merges partial strengths (e.g., “finish your work,” “self-publishing,” “marketing”) into a better-structured plan.

## 4. Key Insights and Innovations
- Discovery of “collaborativeness” in LLMs (§2.1; Figure 1)
  - Insight: Many LLMs produce better answers when shown other models’ outputs—even if those other outputs are worse on their own.  
  - Evidence: Figure 1 shows LC win rates on AlpacaEval 2.0 increase when models read peers’ answers. This is more than just ensembling; it reveals a generalizable synthesis capability in LLMs.
- A layered, prompt-only `Mixture-of-Agents` (§2.2; Figure 2)
  - Novelty: Extends MoE concepts to black-box LLMs via natural language rather than learned routing. The aggregator’s “gating” is reasoning over text, not learned weights.
  - Significance: Achieves SOTA-quality improvements without training, enabling immediate use with off-the-shelf models and APIs.
- Synthesis beats selection (Figure 4, left; Appendix Table 5)
  - Using an LLM ranker to pick the “best” proposal underperforms `MoA`, which writes a new, integrated response. This shows the value of aggregation (reasoning over multiple answers) vs. mere reranking.
- Diversity and width matter (Table 3)
  - Increasing the number of proposers monotonically improves quality; diverse models outperform multiple samples from one model. This is a scalable design knob (width) that improves results without retraining.
- Role specialization (Table 4)
  - Some models are excellent aggregators (e.g., `Qwen1.5-110B-Chat`), others are better as proposers (e.g., `WizardLM-8x22B`). Tuning roles per model is a practical optimization for real systems.

## 5. Experimental Analysis
- Benchmarks and metrics (§3.1–§3.2)
  - AlpacaEval 2.0: 805 general instructions; judged by a GPT-4-based evaluator. Key metric is `LC win rate` (length-controlled win rate) to reduce length bias.
  - MT-Bench: conversational tasks; GPT-4-based scoring with turn-level breakdown.
  - FLASK: 12 skill-specific scores (e.g., correctness, factuality, robustness).
  - MATH (Appendix D, Table 8): accuracy on math reasoning problems (to test generalization beyond instruction following).
- Main quantitative results
  - AlpacaEval 2.0 (Table 2a)  
    > LC win rate: `MoA w/ GPT-4o` = 65.7±0.7%; `MoA` (open-source only) = 65.1±0.6%; `MoA-Lite` = 59.3±0.2%; `GPT-4 Omni (05/13)` = 57.5%.
    - Interpretation: Both `MoA` variants outperform GPT-4 Omni; the open-source-only `MoA` exceeds GPT-4 Omni by +7.6 points LC win.
  - MT-Bench (Table 2b)  
    > Average score: `MoA w/ GPT-4o` = 9.40±0.06; `MoA` = 9.25±0.10; `GPT-4 Turbo (04/09)` = 9.31; `GPT-4 Omni (05/13)` = 9.19.  
    - Interpretation: `MoA` achieves top or near-top scores on an already saturated benchmark (where many strong models are near 9/10).
  - FLASK (Figure 3)
    - `MoA` improves over the single final aggregator (`Qwen1.5-110B-Chat`) on multiple skills: robustness, correctness, efficiency, factuality, commonsense, insightfulness, completeness.  
    - It also surpasses GPT-4 Omni on correctness, factuality, insightfulness, completeness, and metacognition, but is slightly less concise.
  - MATH (Table 8)
    - Layering helps across different aggregators (e.g., `Qwen1.5-110B-Chat` rises from 0.500 at Layer 1 to 0.576 at Layer 3; `LLaMA-3-70B` rises from 0.456 to 0.578). This suggests the approach generalizes to structured reasoning tasks, not only instruction following.
- Ablations and analyses
  - Aggregation vs. ranking (Figure 4, left; Appendix Table 5): the LLM ranker baseline trails `MoA` significantly, showing synthesis brings value beyond picking an existing response.
  - “Does the aggregator just copy the best proposal?”  
    - Correlation analysis (Figure 4, right; Appendix Figure 6): the aggregator’s final answer is more similar (BLEU/TF-IDF/Levenshtein) to higher-quality proposals (positive Spearman correlations). However, final performance exceeds simple selection, indicating true synthesis beyond copying.
  - Number and diversity of proposers (Table 3):  
    > With 2 layers and `Qwen1.5-110B-Chat` as final aggregator, LC win rises with `n`:  
    > `n=1`: 47.8% (both settings) → `n=2`: 58.8% (multi) / 54.5% (single) → `n=3`: 58.0% / 56.1% → `n=6`: 61.3% / 56.7%.  
    - Takeaway: More proposals help; heterogeneity helps more than multiple samples from the same model.
  - Role specialization (Table 4):  
    > As aggregator vs. proposer LC win examples: `Qwen1.5-110B-Chat` (61.3% vs. 56.7%); `WizardLM-8x22B` (52.9% vs. 63.8%).  
    - Takeaway: Some models shine as synthesizers; others as idea generators.
  - Cost and latency trade-offs (Figure 5)
    - Cost: `MoA-Lite` matches GPT-4o’s cost while achieving higher quality; it outperforms GPT-4 Turbo by ~4% LC win at less than half the cost (Figure 5a, text below the figure).
    - Latency proxy (`tflops`, Figure 5b): `MoA` variants lie on the Pareto frontier, suggesting better quality per unit compute than GPT-4 Turbo and GPT-4o for the same LC win rate.
- Do the experiments support the claims?
  - The paper evaluates across three popular alignment-style benchmarks and adds a reasoning dataset (MATH). It also includes ablations that test core mechanisms (aggregation vs. ranking, number/diversity of proposers, role specialization) and a budget analysis (cost and latency proxies).  
  - Caveat: Most evaluations use GPT-4-based judges (AlpacaEval, MT-Bench, and FLASK), which is common but can introduce evaluator bias. The paper mitigates this partly by reporting multiple benchmarks and by showing MoA variants that do not use GPT-4o as an aggregator.

## 6. Limitations and Trade-offs
- Sequential aggregation increases Time to First Token (TTFT)
  - The final answer cannot start streaming until the last layer finishes; this can degrade user experience (Limitations section). The paper suggests limiting layers or exploring chunk-wise aggregation.
- Computational overhead
  - Although cost-effective relative to some strong baselines (Figure 5), `MoA` still runs multiple large models per query. Latency and compute can grow with the number of layers and proposers.
- Reliance on prompt quality and judge bias
  - Performance depends on the `Aggregate-and-Synthesize` prompt (Table 1). The evaluator is GPT-4-based for primary benchmarks, which may not perfectly reflect human judgments, although AlpacaEval’s LC metric correlates strongly with humans (Spearman 0.98 noted in §3.1).
- No explicit truth-checking
  - The aggregator is instructed to be critical, but there is no veracity verification mechanism (e.g., retrieval or fact-checking). In adversarial or noisy settings, strong but wrong proposals might sway the synthesis.
- Scope
  - The method is demonstrated for text tasks with general instruction following and math reasoning. Multimodal, tool-use-heavy, or safety-constrained deployments are not analyzed here.
- Architecture search not exhaustively explored
  - The paper shows clear role specialization (Table 4) and width effects (Table 3), but does not provide an automated way to optimally route or select agents per task instance.

## 7. Implications and Future Directions
- How this changes the landscape
  - `MoA` shows that today’s best performance can be achieved by orchestrating existing LLMs with only prompting. This lowers the barrier to building state-of-the-art systems and encourages thinking in “systems-of-models,” not only “bigger single models.”
- Practical applications
  - Enterprise assistants that need robust, well-rounded answers can combine internal and external models safely via `MoA`.  
  - Content generation, summarization, and question answering pipelines can use `MoA` to merge strengths (e.g., factuality from one model, style from another).  
  - Cost-sensitive products can deploy `MoA-Lite` to beat premium models at similar or lower cost (Figure 5a).
- Follow-up research directions
  - Streaming/low-latency `MoA`: chunk-wise or token-wise aggregation to reduce TTFT (Limitations).  
  - Dynamic routing: learn or infer per-query proposer/aggregator assignments, possibly with lightweight routers (§4.2 related work on routing).  
  - Verification-augmented aggregation: integrate retrieval or tool calls so aggregators can check claims before synthesis.  
  - Trainable aggregator: pretrain or fine-tune a smaller “fusion” model on synthetic `(proposals → synthesis)` data to approximate `MoA` at lower runtime cost.  
  - Scaling width: explore many more heterogeneous proposers (Table 3 suggests monotonic gains), including specialized models (coding, math, safety).
  - Safety and robustness: study behavior under conflicting, biased, or adversarial proposals and design safeguards.

Overall, `Mixture-of-Agents` is a prompt-only, layered collaboration scheme that consistently turns multiple imperfect answers into a better final one. The method’s strength comes from a simple but effective insight—LLMs are good at reading and improving other LLMs’ outputs—and the paper backs this with solid ablations (Tables 3–4, Figure 4), strong benchmark results (Table 2, Figure 3), and practical cost/latency analysis (Figure 5).
