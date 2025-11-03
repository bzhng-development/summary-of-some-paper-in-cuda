# LIMO: Less is More for Reasoning

**ArXiv:** [2502.03387](https://arxiv.org/abs/2502.03387)
**Authors:** Yixin Ye, Zhen Huang, Yang Xiao, Ethan Chern, Shijie Xia, Pengfei Liu
**Institutions:** 

## 🎯 Pitch

LIMO demonstrates that large language models can achieve advanced mathematical reasoning by leveraging a meticulously curated set of just 800 training examples, challenging the prevailing assumption that large-scale datasets are necessary. This innovative approach not only cuts down on data costs but also shifts the focus to the quality of cognitive templates, making sophisticated AI reasoning more accessible and efficient.

---

## 1. Executive Summary
LIMO shows that strong mathematical reasoning in large language models (LLMs) can be unlocked with a very small, carefully curated training set: 800 examples. Using simple supervised fine-tuning on a knowledge-rich base model (`Qwen2.5-32B-Instruct`), LIMO attains competition-level results (e.g., 63.3% on AIME24 and 95.6% on MATH500) and outperforms models trained on 100× more data (Table 1), suggesting that quality and structure of examples, not just quantity, are decisive.

## 2. Context and Motivation
- Problem/gap addressed
  - The field commonly assumes that complex reasoning (especially math) requires massive supervised datasets, and that supervised fine-tuning (SFT) mainly memorizes rather than generalizes. Section 1 summarizes this prevailing view and its costs, citing works that scale instruction data to tens or hundreds of thousands of items.
- Why this matters
  - Practically: Large reasoning datasets are expensive to build and fine-tune on. Cutting data by two orders of magnitude could make high-quality reasoning models far cheaper and more accessible.
  - Conceptually: If sophisticated reasoning can be elicited with minimal, well-chosen demonstrations, the bottleneck shifts from data volume to curation quality and using inference-time computation.
- Prior approaches and limitations
  - Scaling instruction reasoning data (e.g., NuminaMath, OpenThoughts; Section 2.1–2.2) improves performance but is data- and compute-heavy and risks memorization (Section 2.1).
  - Test-time scaling (longer chains, multiple samples) helps but does not say how to most efficiently train models to use such computation (Section 2.2).
- Positioning
  - LIMO proposes the Less-Is-More Reasoning Hypothesis (Sections 1 and 3): if pre-training has already encoded rich domain knowledge and test-time compute allows long reasoning, then only a small number of “cognitive template” exemplars are needed to trigger sophisticated reasoning. LIMO operationalizes this with a disciplined data curation pipeline and a compact SFT stage.

## 3. Technical Approach
This is an empirical recipe that couples selective data curation with a modest SFT stage.

Key terms used below
- `reasoning chain` (often called chain-of-thought/CoT): a step-by-step solution path with intermediate steps and checks.
- `pass@1`: the accuracy when only the top model output is considered.
- `Less-Is-More Reasoning (LIMO) Hypothesis`: In knowledge-rich foundation models, minimal but precisely organized examples that expose long, explicit reasoning can elicit sophisticated reasoning behaviors (Sections 1 and 3).

A. Data curation pipeline (Section 3; Figure 2)
1) Build a large candidate pool of math problems
   - Sources include NuminaMath-CoT, DeepScaleR, AIME (pre-2024), MATH, and diverse Chinese curricula/exams (Section 3.1.1).
2) Two-stage difficulty filtering to keep only challenging, reasoning-intensive items
   - Stage 1 (coarse): Use `Qwen2.5-Math-7B-Instruct` with short-CoT; discard any problem it solves within four attempts (Section 3.1.1).
   - Stage 2 (fine): Use `DeepSeek-R1-Distill-Qwen-32B`; sample 32 attempts per problem and compute empirical success rate. Keep problems solved in only 1–3 of 32 attempts (Section 3.1.1). Result: 2,125 hard problems (`LIMO-Pool`).
   - Deduplicate against all evaluation benchmarks via n-gram matching to reduce contamination risk (Section 3.1.1).
3) Construct and score candidate reasoning chains for each kept problem (Section 3.1.2)
   - Generate multiple solutions per problem using strong reasoners: `DeepSeek R1`, `DeepSeek-R1-Distill-Qwen-32B`, and `QwQ-32B` (Section 3.1.2).
   - Score chains with a rule-based metric emphasizing qualities the authors want the model to imitate (Section 3.1.2):
     - Elaborated Reasoning: longer, fully spelled-out logic (30% weight; measured by length).
     - Self-Verification: explicit checks (“check”, “verify”) of intermediate results (20%).
     - Exploratory Approach: considering alternatives (“perhaps”, “might”) (25%).
     - Adaptive Granularity: connective reasoning (“therefore”, “since”) (25%).
     - Frequencies are normalized by text length to avoid rewarding verbosity alone.
   - For each problem, select the highest-scoring chain.
4) Rank all problem–solution pairs by their chain-quality scores and choose the top 800 to form the final `LIMO` dataset (Section 3.1.2).

Design rationale
- The pipeline enforces two conditions in the LIMO hypothesis:
  - Only keep problems that demand multi-step reasoning (difficulty filtering).
  - Only keep solutions that model the intended cognitive processes (chain scoring), so that they serve as high-signal “cognitive templates.”

B. Fine-tuning recipe (Section 4)
- Base model: `Qwen2.5-32B-Instruct` (chosen for strong pre-trained math/code knowledge; Section 1 and 6.3.3).
- SFT setup:
  - Full-parameter fine-tuning.
  - Sequence length: up to 16,384 tokens (covers all SFT responses).
  - Optimizations: DeepSpeed ZeRO-3 and FlashAttention-2.
  - Learning rate 5e-6 with cosine decay; no warmup; 15 epochs; batch size 64 (Section 4).
- Intuition: With only 800 high-quality exemplars, the model learns to write longer, self-checking, step-structured solutions, leveraging its pre-trained knowledge during inference.

C. Evaluation protocol (Section 5)
- Benchmarks
  - In-domain: AIME24, MATH500, AMC23.
  - Out-of-distribution (OOD): OlympiadBench; new multilingual Chinese sets (CHMath, Gaokao 2024, Kaoyan, GradeSchool); STEM benchmarks (MinervaMath) and GPQA (Section 5).
- Decoding and metrics
  - Zero-shot chain-of-thought; pass@1 everywhere.
  - For small test sets (<50 problems; AIME24, AMC23, CHMath): 4 samples at temperature 0.6 and unbiased pass@1 (per Chen et al., 2021).
  - For larger sets: one greedy sample.
  - Max output length 32,768 tokens (Section 5).

D. Baselines for comparison (Section 6.1; Table 1)
- Strong proprietary or open models: `OpenAI-o1-preview`, `QwQ-32B-Preview`, and the `Qwen2.5-32B-Instruct` base model.
- Two large SFT datasets on the same backbone to isolate the effect of data quality vs quantity:
  - `OpenThoughts-114k` (multi-domain synthetic reasoning chains).
  - `NuminaMath-100k` (random subset with CoT).

## 4. Key Insights and Innovations
1) A concrete, testable “Less-Is-More” hypothesis for reasoning (Sections 1 and 3)
   - Novel angle: It asserts the limiting factor is not task complexity per se but (a) how much prerequisite knowledge exists in pre-training and (b) whether SFT examples demonstrate the right cognitive processes. This reframes reasoning training as elicitation rather than acquisition.

2) A disciplined, dual filter for question difficulty plus an explicit chain-quality scorer (Sections 3.1.1–3.1.2; Figure 2)
   - Different from prior large-scale data collection: LIMO keeps only items that are both hard to solve and whose solutions embody long, exploratory, self-verifying reasoning. The scoring rubric operationalizes “quality” beyond correctness or length alone.

3) Empirical evidence that curated quality beats scale by a wide margin (Table 1; Section 6.2)
   - 800 LIMO examples (same backbone) outperform training with 114k or 100k examples from popular datasets across 10 benchmarks, including substantial OOD generalization. This is a strong, quantified demonstration that “what” you train on matters more than “how much.”

4) Clear diagnostic ablations isolating what makes LIMO work (Section 6.3; Figures 3–8)
   - Chain quality, question difficulty, pre-training quality, model size, and sample count are each varied in controlled ways. This suite of studies transitions the work from a recipe to a set of causal insights.

Overall, these contributions go beyond an incremental dataset release; they provide a methodology and evidence base for highly data-efficient reasoning elicitation.

## 5. Experimental Analysis
A. Evaluation setup
- Benchmarks and decoding settings described in Section 5.
- All models evaluated with pass@1; small benchmarks use unbiased pass@1 from 4 samples; larger ones use one greedy output.

B. Main results (Table 1)
- In-domain
  - AIME24: LIMO 63.3% vs `QwQ-32B-Preview` 50.0%, `OpenAI-o1-preview` 44.6%, base `Qwen2.5-32B-Instruct` 16.5%.
  - MATH500: LIMO 95.6% vs `QwQ-32B-Preview` 89.8% and `OpenAI-o1-preview` 85.5%.
  - AMC23: LIMO 96.3% vs `QwQ-32B-Preview` 83.6%.
- Out-of-domain
  - OlympiadBench: 67.6% (LIMO) vs 58.5% (QwQ-32B-Preview) and 45.3% (base).
  - CHMath (Chinese league 2024): 84.2% vs 68.5% (QwQ-32B-Preview).
  - Gaokao 2024: 91.1% vs 80.1% (QwQ-32B-Preview) and 72.1% (base).
  - Kaoyan: 83.9% vs 70.3% (QwQ-32B-Preview) and 48.2% (base).
  - GradeSchool: 76.2% vs 63.8% (QwQ-32B-Preview) and 56.7% (base).
  - MinervaMath: 52.2% vs 41.2% (base) and 39.0% (QwQ-32B-Preview).
  - GPQA: 70.7% (LIMO) close to 73.3% (`OpenAI-o1-preview`).
- Average across all benchmarks
  - LIMO: 78.1%
  - QwQ-32B-Preview: 66.9%
  - OpenAI-o1-preview: 61.1%
  - Base model: 49.9%
  - SFT with large datasets on the same backbone:
    - `OpenThoughts-114k`: 58.3%
    - `NuminaMath-100k`: 32.3%

> Table 1 shows that with only 800 examples, LIMO reaches 63.3% on AIME24 and 95.6% on MATH500 and achieves the highest average (78.1%) across 10 benchmarks, outperforming models trained on 100k–114k items on the same backbone.

C. Do the experiments support the claims?
- Yes, multiple angles reinforce the central thesis:
  - Cross-benchmark superiority with tiny SFT data.
  - OOD strength across different languages and exam styles.
  - Diagnostics that isolate the effect of chain quality and question difficulty (Sections 6.3.1–6.3.2; Figures 3–4).

D. Ablations and diagnostics (Section 6.3)
- RQ1: Chain quality matters (Figure 3)
  - When training on reasoning chains bucketed from low- to high-quality (L1→L5), accuracy increases monotonically on both AIME24 and MATH500. The L5 condition performs best. This connects the rule-based chain-quality score to measurable gains.
- RQ2: Question difficulty matters (Figure 4)
  - Training only on harder problems boosts performance even without in-domain overlap:
    - AIME24 accuracy climbs to 51.5% when trained on an AIME-level set (“Advanced-500”), a +16 percentage point gain over easier sets.
    - MATH500 reaches 91.2% despite zero direct training on MATH problems, indicating transfer from tougher problems to simpler ones.
- RQ3: Pre-training quality matters (Figure 5)
  - Fine-tuning the same 800 LIMO examples on different but architecturally similar backbones shows large gaps:
    - `Qwen1.5-32B-Chat`: 9.2% (AIME24) and 65.2% (MATH500).
    - `Qwen2.5-32B-Instruct`: 63.3% and 95.6%.
  - The 54.1-point AIME24 gain and 30.4-point MATH500 gain suggest that richer pre-trained math/code corpora enable the “less data” strategy to work.
- RQ4: Model size scaling (Figure 6)
  - AIME24 increases from 2.5% (3B) to 68.3% (72B).
  - Saturation: 32B vs 72B is close on MATH500 (95.6% vs 94.8%) and less dramatic on AIME24 (63.3% vs 68.3%), indicating diminishing returns past ~32B for these benchmarks.
- RQ5: Sample efficiency (Figures 7–8)
  - With only 400 curated samples, performance already jumps:
    - AIME24: 16.5% → 57.5%.
    - MATH500: 79.4% → 94.8%.
  - Gains plateau around 800–1,200 examples; 2,000 gives the best but only slightly higher scores (69.6% AIME24, 95.8% MATH500), implying early saturation.

E. Robustness and potential concerns
- Deduplication against test benchmarks is done via n-gram matching (Section 3.1.1), but semantic duplicates may slip through; still, broad OOD improvements and the Chinese-language tests reduce the risk that gains come from leakage.
- Complex answers use an LLM-based evaluator (Section 5); while common practice, this can introduce evaluator bias. Numerical answers are rule-checked.
- The evaluation allows up to 32k output tokens (Section 5), which may increase inference cost; but this aligns with the hypothesis that reasoning benefits from test-time compute.

## 6. Limitations and Trade-offs
- Dependence on prior knowledge in the base model (Sections 1, 6.3.3)
  - The approach presumes that pre-training already encodes rich math knowledge. Results with `Qwen1.5-32B-Chat` (9.2% AIME24) underline that without a strong foundation, small SFT datasets won’t suffice.
- Reliance on long, verbose chains (Section 3.1.2)
  - The chain-quality scorer rewards markers like “check/verify” or “perhaps/might.” This may favor stylistic verbosity and could overlook concise but rigorous proofs. There’s a risk of conflating quality with particular rhetorical patterns.
- Data curation compute vs training compute
  - Although SFT data are few, the filtering pipeline is compute-intensive: it samples 32 attempts with a strong model across a large pool and uses multiple strong models to generate candidate chains (Section 3.1). This is cheaper than massive SFT, but not free.
- Domain and modality scope
  - The study focuses on math. It tests transfer to STEM QA (Minerva) and GP-level QA (GPQA), but does not demonstrate the same pipeline for code or multi-modal reasoning (Section 5).
- Inference-time cost
  - The framework is predicated on long chain generation at inference (32k max tokens; Section 5). Deployments that need short latency and small outputs may not reap the full benefits.
- Open questions
  - The theoretical “minimum” data size remains open (Section 6.3.5 notes early saturation but not a lower bound).
  - The rule-based chain scorer is heuristic; whether learned or human-judged quality measures would be superior remains to be seen.

## 7. Implications and Future Directions
- How this changes the landscape
  - Shifts the focus from “scale the number of training samples” to “curate a tiny set of high-signal exemplars that exploit pre-trained knowledge and test-time compute.” This reframing could make advanced reasoning more accessible to smaller labs and products.
- Follow-up research enabled
  - Active learning for chain quality: iteratively query a model for problems that most improve long-CoT reasoning, then select or edit solutions using richer, learned quality models rather than keyword heuristics (Section 6.3.5 suggests room for optimization).
  - Beyond math: apply the LIMO pipeline to programming, scientific reasoning, or planning—domains where pre-training already contains substantial knowledge and long inference helps.
  - Better chain-quality metrics: replace keyword-based scoring with models trained to detect logical coherence, error-checking, and minimal hallucination; or use process supervision signals (e.g., verifier models).
  - Cost-aware inference: combine LIMO-style training with test-time routing, early-exit strategies, or dynamic token budgets to keep long chains only when needed.
  - Theory of elicitation thresholds: formalize how pre-training coverage, model size, and chain exemplars jointly determine when complex reasoning emerges.
- Practical applications
  - Education/tutoring: models that show explicit, self-checked solutions can be safer and more instructive.
  - Assessment/grading: robust step-by-step solvers for math competitions and standardized tests.
  - Scientific/engineering workflows: long-form derivations with built-in self-verification reduce silent errors.

> In short, Sections 3–6, Table 1, and Figures 3–8 collectively show that with a strong base model and careful data selection emphasizing long, self-verified reasoning, supervised fine-tuning on only 800 problems can elicit high-level mathematical reasoning and robust generalization.
