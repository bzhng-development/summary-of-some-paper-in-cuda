# LIMO: Less is More for Reasoning

**ArXiv:** [2502.03387](https://arxiv.org/abs/2502.03387)

## 🎯 Pitch

LIMO overturns the conventional belief that large language models require massive supervised datasets to master sophisticated mathematical reasoning. By meticulously curating just 800 high-quality problems and solutions and fine-tuning a pre-trained foundation model, LIMO achieves state-of-the-art performance surpassing models trained on 100x more data—demonstrating that, for well-pretrained LLMs, a handful of strategically chosen cognitive templates can unlock robust, generalizable reasoning skills. This breakthrough not only democratizes the creation of powerful reasoning models by slashing data and compute costs but also reshapes our understanding of how knowledge and reasoning capabilities emerge in LLMs.

---

## 1. Executive Summary
LIMO shows that strong mathematical reasoning in large language models (LLMs) can be elicited with just hundreds of carefully chosen examples rather than the tens or hundreds of thousands typically used. By curating 800 high-quality problems and solutions and applying straightforward supervised fine-tuning to `Qwen2.5-32B-Instruct`, the model attains state-of-the-art or near–state-of-the-art results on difficult math and STEM benchmarks (e.g., 63.3% on AIME24, 95.6% on MATH500; Table 1), and generalizes out of distribution and cross-lingually.

## 2. Context and Motivation
- Problem addressed
  - Can complex, multi-step reasoning be taught to LLMs with very small amounts of supervision? Section 1 argues that conventional wisdom holds that reasoning needs massive training sets because models otherwise just memorize patterns and fail to generalize.
- Why this matters
  - Practical: Massive supervised datasets are expensive to collect and train on, and they slow iteration. Lowering the data requirement makes high-quality reasoning models far more accessible.
  - Scientific: If sophisticated reasoning emerges with a handful of examples, then pre-training may already encode much of the required domain knowledge; post-training might mainly “unlock” it.
- Where prior approaches fall short
  - Large-scale SFT on reasoning data (e.g., 100k+ CoT examples) brings gains but risks memorization and can degrade OOD generalization (Section 2.1; criticism in Mirzadeh et al., 2024; Chu et al., 2025).
  - Test-time compute scaling improves performance (Section 2.2), but most systems still rely on large training corpora.
- Positioning
  - LIMO advances the “quality-over-quantity” line from LIMA (alignment with ~1k examples) to the harder domain of mathematical reasoning (Section 2.3). It hypothesizes that with a knowledge-rich foundation model and adequate inference-time “workspace,” only a small set of carefully designed examples is needed to trigger long-form reasoning (Sections 1 and 3).

## 3. Technical Approach
The LIMO method has three pillars: a hypothesis about eliciting reasoning, a data curation pipeline that operationalizes the hypothesis, and a lightweight training recipe.

A. The LIMO Hypothesis (Sections 1, 3)
- Claim in plain words: If a foundation model has already absorbed the necessary domain knowledge during pre-training, then a small number of strategically designed demonstrations can teach it how to “use what it knows” through extended reasoning.
- Two determinants of success:
  1) The model’s latent knowledge base is sufficiently complete.
  2) The post-training examples act as effective “cognitive templates”—explicit, step-by-step chains that demonstrate deliberation, self-checking, and problem decomposition.

B. Data curation pipeline (Section 3; Figures 2 and pages 3–5)
1) Build a large candidate pool
   - Sources include NuminaMath-CoT, DeepScaleR, historical AIME (pre-2024), MATH, and Chinese curricular exams across levels. The initial scale is “tens of millions” of problems.
2) Two-stage difficulty filtering
   - Stage 1 (coarse filter): Use `Qwen2.5-Math-7B-Instruct`. Any problem it answers correctly within 4 attempts is discarded, removing trivial items.
   - Stage 2 (fine filter): Use a stronger reasoning model `DeepSeek-R1-Distill-Qwen-32B`. For each remaining problem, sample 32 solutions and compute an empirical success rate. Retain problems solved only 1–3 times out of 32, keeping items that are challenging but not impossible. This yields 2,125 problems (`LIMO-Pool`).
   - Deduplication: n-gram matching against all evaluation benchmarks to avoid overlap (Section 3.1.1).
3) Construct and score reasoning chains
   - Generate solutions for each retained problem using three strong reasoners: `DeepSeek-R1`, `DeepSeek-R1-Distill-Qwen-32B`, and `QwQ-32B` (Section 3.1.2).
   - Manually analyze qualities that matter for eliciting reasoning, then quantify with a rule-based scoring scheme:
     - Elaborated reasoning (weight 30%): lengthier, fully spelled-out logic.
     - Self-verification (20%): frequency of validation terms (e.g., “check,” “verify”).
     - Exploratory approach (25%): tentative reasoning markers (“perhaps,” “might”).
     - Adaptive granularity (25%): connective phrases indicating structured deduction (“therefore,” “since”).
     - All counts are normalized by text length to avoid bias toward longer responses.
   - For each problem, keep the highest-scoring solution. Rank all problem–solution pairs and select the top 800 to form the final LIMO training set.
   - Intuition: Beyond correctness, solutions should model how to think—exploration, checking, and explicit step structure—so the model internalizes the process, not just the answer.

C. Training recipe (Section 4)
- Base model: `Qwen2.5-32B-Instruct`.
- Supervised fine-tuning (no RL), full-parameter training with DeepSpeed ZeRO-3 and FlashAttention-2.
- Hyperparameters: learning rate 5.0e-6 with cosine decay, no warmup; 15 epochs; batch size 64; response sequences capped at 16,384 tokens during training.
- Rationale for choices:
  - No warmup and multiple epochs help the model quickly adapt to the high-quality, long-form reasoning demonstrations (Section 4).
  - Long sequence budget allows training on lengthy chains so the model learns to “think in long form.”
- Inference-time budget: outputs up to 32,768 tokens during evaluation (Section 5), aligning with the test-time compute scaling perspective that longer chains provide more cognitive workspace.

D. Evaluation protocol (Section 5)
- Zero-shot chain-of-thought setting with `pass@1` (probability the final answer of the first chosen output is correct).
- Decoding:
  - Large benchmarks (MATH500, OlympiadBench, Gaokao, Kaoyan, GradeSchool, MinervaMath, GPQA): greedy decoding, single sample.
  - Small benchmarks (AIME24, AMC23, CHMath): sample 4 outputs with temperature=0.6, compute unbiased `pass@1` (Chen et al., 2021).
- Answer checking: rule-based for numeric answers; LLM-based evaluator for complex formats (Section 5).

Analogy for clarity: Think of pre-training as letting the model “study” a huge library. LIMO’s examples are not more library books; they are a small set of “worked solutions” that show the model how to reason step by step and double-check itself. At test time, we let the model keep enough scratch paper (long output budget) to “show its work.”

## 4. Key Insights and Innovations
- A minimal-data route to complex reasoning (Sections 1 and 6.2; Table 1; Figure 1)
  - Novelty: Demonstrates competition-level math performance using only 800 SFT examples—about 1% of the data size used in common open datasets (e.g., 100k+ in NuminaMath-100k or OpenThoughts-114k in Table 1).
  - Significance: Reduces the cost barrier to training strong reasoning models while improving generalization.
- A structured definition of “reasoning chain quality” (Section 3.1.2; Figure 2)
  - Novelty: Goes beyond correctness to quantify valuable properties of reasoning (elaboration, self-verification, exploration, adaptive granularity) via a weighted, length-normalized heuristic.
  - Significance: Ablations (Figure 3) show higher-quality chains produce better models, isolating the role of “how” a solution is written.
- Difficulty-first problem selection using model-based filtering (Section 3.1.1)
  - Novelty: Two-tier difficulty filtering based on empirical success rates under strong reasoners ensures retained items naturally elicit longer reasoning.
  - Significance: Ablations (Figure 4) reveal that harder problems in training improve performance even on easier benchmarks, indicating transfer from challenging to broader tasks.
- The Less-Is-More Reasoning Hypothesis (Sections 1 and 3)
  - Conceptual advance: Proposes that the threshold to elicit sophisticated reasoning depends on (a) pre-trained knowledge completeness and (b) the ability of few, high-quality exemplars to serve as “cognitive templates.”
  - Evidence: Section 6.3.3 (Figure 5) shows dramatic sensitivity to the pre-trained backbone (Qwen2.5 vs. Qwen1.5), and Section 6.3.5 (Figures 7–8) shows diminishing returns beyond ~800 samples.

## 5. Experimental Analysis
- Datasets and metrics (Section 5)
  - In-domain: AIME24, MATH500, AMC23.
  - Out-of-domain: OlympiadBench, CHMath (Chinese math league 2024), Gaokao (CN college entrance 2024), Kaoyan (CN graduate entrance), GradeSchool (new elementary math benchmark), plus MinervaMath and GPQA for broader STEM reasoning.
  - Metric: `pass@1`. Decoding setups differ by set size (see Section 3D above).

- Baselines (Section 6.1; Table 1)
  - Strong proprietary/open models: `OpenAI-o1-preview`, `QwQ-32B-Preview`.
  - Backbone without SFT: `Qwen2.5-32B-Instruct`.
  - SFT with large datasets on the same backbone: `OpenThoughts-114k`, `NuminaMath-100k`.

- Main quantitative results (Table 1; Section 6.2)
  - In-domain
    - AIME24: 
      > LIMO 63.3 vs QwQ-32B-Preview 50.0; OpenAI-o1-preview 44.6; base 16.5; NuminaMath-100k 6.5; OpenThoughts-114k 50.2.
    - MATH500:
      > LIMO 95.6 vs QwQ-32B-Preview 89.8; o1-preview 85.5; base 79.4.
    - AMC23:
      > LIMO 96.3 vs QwQ-32B-Preview 83.6; o1-preview 81.8; base 64.0.
  - Out-of-domain and cross-lingual
    - OlympiadBench:
      > LIMO 67.6 vs QwQ-32B-Preview 58.5; base 45.3.
    - CHMath (Chinese):
      > LIMO 84.2 vs QwQ-32B-Preview 68.5; base 27.3.
    - Gaokao (Chinese):
      > LIMO 91.1 vs QwQ-32B-Preview 80.1; base 72.1.
    - Kaoyan (Chinese):
      > LIMO 83.9 vs QwQ-32B-Preview 70.3; base 48.2.
    - GradeSchool:
      > LIMO 76.2 vs QwQ-32B-Preview 63.8; base 56.7.
    - MinervaMath:
      > LIMO 52.2 vs o1-preview 47.1; base 41.2.
    - GPQA:
      > LIMO 70.7, close to o1-preview 73.3; base 48.0.
  - Average over all benchmarks:
    > LIMO 78.1 vs QwQ-32B-Preview 66.9; o1-preview 61.1; OpenThoughts-114k 58.3; NuminaMath-100k 32.3.

- Do experiments support the claims?
  - Data efficiency: Yes. Table 1 shows 800 examples outperform 100k–114k datasets trained on the same backbone.
  - Generalization: Strong OOD and cross-lingual gains (Table 1: large improvements on CHMath, Gaokao, Kaoyan, and GradeSchool).
  - Hypothesis about pre-training and exemplars:
    - Backbone sensitivity (Section 6.3.3; Figure 5):
      > With the same LIMO set, Qwen2.5-32B achieves 63.3 (AIME24) vs Qwen1.5-32B’s 9.2; and 95.6 (MATH500) vs ~65.2, a >30-point gap. This ties success to a knowledge-rich base model.
    - Reasoning-chain quality matters (Section 6.3.1; Figure 3):
      > Training on L5 (highest-quality) chains outperforms L1–L4 consistently on both AIME24 and MATH500, validating the scoring scheme.
    - Problem difficulty matters (Section 6.3.2; Figure 4):
      > Moving from Simple-500 to Advanced-500 increases AIME24 accuracy by 16 points; training solely on harder AIME-like items still yields 91.2% on MATH500, indicating transfer.
    - Sample efficiency and diminishing returns (Section 6.3.5; Figures 7–8):
      > 400 samples already lift AIME24 from 16.5 to 57.5 and MATH500 from 79.4 to 94.8; improvements plateau after ~800 samples (2k reaches 69.6/95.8, only marginal gains).

- Experimental setup details that bolster credibility
  - Output length up to 32k tokens enables long chains (Section 5).
  - For small sets (AIME24, AMC23, CHMath), they use multiple samples and unbiased `pass@1`, reducing sampling bias (Section 5).
  - Deduplication from evaluation sets using n-gram checks (Section 3.1.1).

- Caveats in the evidence
  - LLM-based answer graders for complex formats introduce evaluation variance (Section 5).
  - Dedup via n-gram matching may miss paraphrases; however, they explicitly state the check (Section 3.1.1).
  - Compute for curation (32 samples per problem in stage-2 filtering and multi-sampler solution generation) is nontrivial, though it’s a one-time cost (Section 3.1.1 and 3.1.2).

## 6. Limitations and Trade-offs
- Dependency on a strong pre-trained backbone
  - Section 6.3.3 (Figure 5) shows that without a math- and code-rich pre-training corpus (Qwen2.5 vs Qwen1.5), minimal SFT is insufficient (9.2% vs 63.3% on AIME24). LIMO’s recipe assumes extensive pre-trained knowledge.
- Heuristic scoring of reasoning chains
  - The quality metric relies on keyword-based counts and length (Section 3.1.2). While effective in ablations, it may reward verbosity or specific phrasing styles and might not capture deeper logical soundness beyond the final correctness filter.
- Curation compute vs. training compute
  - Although training uses only 800 examples, constructing that set required multi-model sampling and 32-shot difficulty estimation for many candidates (Sections 3.1.1–3.1.2), which can be computationally expensive for new domains.
- Inference-time cost
  - The approach leans on long output budgets (up to 32,768 tokens; Section 5). This improves accuracy but increases latency and cost per query.
- Scope
  - The strongest results are in mathematics and math-heavy STEM. While MinervaMath and GPQA are included, broader reasoning domains (e.g., legal, medical, commonsense) are not evaluated, leaving generality to other domains as open.
- Evaluation artifacts
  - Some benchmarks use greedy decoding while others use sampling (Section 5). LLM-based grading for complex answers can introduce subtle biases.
- Data contamination risk is mitigated but not eliminated
  - Deduplication uses n-gram matching (Section 3.1.1); near-duplicate or paraphrastic leakage is harder to detect.

## 7. Implications and Future Directions
- How this changes the field
  - Establishes a credible, empirically validated path to high-level reasoning without massive SFT: prioritize a small number of high-difficulty, high-quality, long-form examples and give the model ample inference-time budget (Sections 1, 6.2, Figures 7–8).
  - Reframes “training for reasoning” as “eliciting existing knowledge with cognitive templates,” shifting attention from dataset size to exemplar design.
- What follow-up research this enables
  - Automated data selection: Replace heuristic quality scoring with learned evaluators for chain quality, or use active learning to choose the next best problems (Section 6.3.5 hints at this).
  - Domain transfer: Apply LIMO to programming, scientific reasoning, or multi-modal settings to test the hypothesis beyond math.
  - Efficiency research: Co-design training and inference budgets—e.g., when can shorter chains with better self-verification match long chains?
  - Robust quality metrics: Develop content-aware logic checks (e.g., step-level consistency proofs) rather than keyword proxies.
  - Backbone analysis: Quantify how much and what kind of pre-training data is “enough” to cross the elicitation threshold (Section 6.3.3 suggests strong sensitivity).
- Practical applications
  - Education/tutoring: Models that show their reasoning and verify steps are useful for pedagogy.
  - Competitive math and STEM assistance: High AIME24 and MATH500 scores (Table 1) indicate utility for problem solving and solution drafting.
  - Low-resource model development: Institutions without massive compute can fine-tune competitive reasoners by investing in careful example selection rather than huge datasets.

In short, Sections 3–6 collectively make the case that with the right backbone, a small set of carefully crafted, difficult, and explicitly reasoned examples—and sufficient test-time “scratch space”—can substitute for orders of magnitude more training data, yielding both in-domain strength and robust OOD generalization.
