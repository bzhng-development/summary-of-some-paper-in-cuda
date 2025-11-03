# Thinking Augmented Pre‑training

**ArXiv:** [2509.20186](https://arxiv.org/abs/2509.20186)
**Authors:** Liang Wang, Nan Yang, Shaohan Huang, Li Dong, Furu Wei
**Institutions:** 

## 🎯 Pitch

The paper introduces Thinking Augmented Pre-Training (TPT), a method that enhances data efficiency and model performance by appending automatically generated "thinking trajectories" during pre-training. By simplifying the learning of complex tokens, TPT significantly reduces the reliance on large-scale data, showing that smaller models trained with TPT can compete with those using trillions of tokens, thereby lowering computational costs and broadening AI accessibility.

---

## 1. Executive Summary
- This paper introduces Thinking Augmented Pre-Training (`TPT`): a data-centric method that appends automatically generated step-by-step “thinking trajectories” to ordinary pre-training text and then trains models with standard next-token prediction on the concatenated sequence.
- By making difficult tokens easier to learn and implicitly allocating more training compute to reasoning-heavy content, `TPT` improves data efficiency and final performance. With only 100B training tokens, an 8B model trained with `TPT` approaches or surpasses stronger baselines trained on trillions of tokens (Figures 1–2, Table 1) and yields large gains after instruction tuning (Table 2).

## 2. Context and Motivation
- Problem addressed
  - Modern LLMs are limited by high-quality data availability while compute keeps scaling. Some “high-value” tokens—typically final answers, conclusions, or compressed facts—are hard to learn from a single next-token step because the underlying rationale spans many latent steps (Introduction; Figure 1b example where the answer “890” depends on several algebraic steps).
- Why it matters
  - Data scarcity is now a central bottleneck for frontier models. Improving how much value is extracted per token can lower training cost and reduce the reliance on ever-larger web corpora (Section 1).
- Prior approaches and gaps
  - Data selection/prioritization focuses on picking “valuable and learnable” tokens but still asks the model to learn hard targets in one step (Section 1; references to Mindermann et al., 2022; Lin et al., 2024).
  - RL-style “reasoning pre-training” (e.g., RPT, OpenAI o1/DeepSeek-R1) improves reasoning but requires expensive online rollouts and token-level credit assignment (Related Work; Section 2’s “Scalability” paragraph).
  - Hidden-thoughts/latent-thought methods (Reasoning CPT, BoLT) showed promise but at smaller scales and narrower domains (Related Work).
- Positioning
  - `TPT` is a simple, scalable, document-level augmentation pipeline that:
    - Requires no human annotation or online RL.
    - Works for pre-training from scratch and for “mid-training” (continual pre-training) on existing checkpoints.
    - Naturally “upsamples” difficult/valuable content by generating longer thinking for such cases (Section 4; Figure 4).

## 3. Technical Approach
Goal: make hard tokens more learnable by exposing the intermediate reasoning that would produce them.

- Core concept: `thinking trajectory`
  - Definition: automatically generated, multi-step explanatory text that simulates how an expert would analyze the given document. It is wrapped in tags like `<think> ... </think>` in many examples (Figure 1b; Appendix Table 8).
- Data augmentation pipeline (Section 2)
  1. Start with a document `d` from the pre-training corpus.
  2. Use an off-the-shelf LLM (e.g., DeepSeek-R1-Distill-Qwen-7B for mid-training; Qwen3-8B for scratch pre-training; Appendix A.1) with a fixed prompt to generate a `thinking trajectory` `t`.
     - Prompt summary (Section 2): “Simulate an expert’s in-depth thought process… focus on complex and informative aspects… use Feynman technique…”.
     - Generation settings (Appendix A.1): input truncated to 2k tokens; generate up to 8k thinking tokens; temperature 0.6; top-p 0.9. Generation stops at `</think>` to avoid summarizing again beyond the thought.
  3. Concatenate `[d; t]` to form a single training sample `x`.
  4. Train with the standard next-token prediction objective over the entire sequence:
     - Plain-language: predict the next token given all prior tokens, where the prior includes both the original document and the generated thought.
     - Equation (1): `min L = -(1/N) * Σ log p(x_i | x_<i)` where `N` is tokens in `[d; t]`.
- Key design choices and logic
  - Why document-level augmentation?
    - It scales: no need for per-token rollouts or interactive RL (Section 2, “Scalability”).
    - It transforms noisy web text into an “LLM-friendly” didactic format that the model can learn from more easily (Section 2, “LLM-friendly Data Format”; Figure 2 training-loss gap).
  - Why does this allocate training compute dynamically?
    - Harder documents induce longer thoughts (Section 4), hence more tokens, hence more gradient steps on those samples—analogous to test-time scaling but applied at training time (Section 2, “Dynamic Allocation of Training Compute”; Figure 4).
  - Example (Figure 1b; Appendix Table 8)
    - A single-token answer “890” is difficult to learn directly. The trajectory reveals the Remainder Theorem reasoning chain, decomposing the task so the model can learn the path to that token.
- Training regimes implemented (Section 3)
  - Pre-training from scratch under abundant data (100B token budget).
  - Pre-training under constrained unique data (10B raw-token pool; 40B training budget via repeated epochs).
  - Mid-training (continual pre-training) on public checkpoints (1.5B–7B) followed by instruction SFT.
- Implementation details of experiments (Appendix A)
  - Corpora: `FineWeb-Edu` and `MegaMath-Web-Pro-Max`, mixed via sampling weights (Appendix A.1).
  - Packing: documents are packed into 8k-token samples during pre-/mid-training; SFT uses up to 32k context (Table 5).
  - Hardware and cost: thought generation ~20k A100 GPU hours for 100B tokens; training on MI300; 8B for 100B tokens takes ~1 week (Appendix A.1).

## 4. Key Insights and Innovations
- Thinking as a data transform (fundamental)
  - Novelty: Instead of adding new data sources or using RL, `TPT` turns existing text into didactic, reasoning-rich training signals by appending generated step-by-step thoughts (Section 2).
  - Significance: It improves the learnability of hard tokens and reduces the reliance on scarce high-quality web data (Figures 1–2; Table 1).
- Training-time compute reallocation (conceptual)
  - Insight: Thinking length positively correlates with domain difficulty and “reasoning intensity,” so models naturally spend more tokens where it matters (Section 4; Figure 4 shows Advanced reasoning ≈ 50% longer thoughts than None).
  - Significance: This is a training-time analogue of test-time scaling—without special schedulers or heuristics.
- Broad-stage applicability (practical)
  - Works for pre-training from scratch, mid-training, and SFT pipelines (Section 3). Gains persist after instruction tuning and across tasks (math, code, general reasoning) and sizes (1.5B→8B) (Tables 1–3).
- Surprising ablation: smaller teacher can be better (insight)
  - Using a 1.5B model to generate thinking sometimes outperforms a 7B generator for downstream 3B students (Table 4), suggesting that “teacher complexity” is not a monotonic driver of useful thoughts and hinting at teacher–student compatibility effects.

## 5. Experimental Analysis
- Evaluation setup
  - Base-model evaluation (Section 3.1; Appendix A.2):
    - Datasets: GSM8K, MATH, BoolQ, MMLU, MMLU-Pro.
    - Protocol: few-shot CoT for GSM8K/MATH and MMLU/MMLU-Pro; zero-shot for BoolQ; aggregated score averages the five tasks (Figure 2 right).
    - Note: if the answer extractor fails, score is zero—so early training can be below random guess (Section 3.1).
  - Instruction-tuned evaluation (Sections 3.1–3.3; Appendix A.2):
    - After SFT on Mixture-of-Thoughts (350k examples; 32k context). Tasks span math (MATH-500, AIME24/25, GSM8k, HMMT), coding (HumanEval, LiveCodeBench v4_v5), and general knowledge (GPQA-Diamond, MMLU-Pro, JEEBench). Pass@1 is the main metric. Sampling temperatures and multiple samples per problem are specified (Appendix A.2).
- Main quantitative results
  - Pre-training under abundant data (100B tokens, 8B model; Figure 2; Table 1):
    - Training loss is much lower with thoughts, indicating more learnable signal.
    - Average across five base tasks:
      - Quote:
        > Table 1: `TPT-8B` (100B) avg 43.9 vs `Vanilla-8B` (100B) avg 26.2.
      - Math-heavy gains:
        > GSM8K: 50.1 vs 19.2; MATH: 21.8 vs 9.1.
      - Competitiveness with a much larger-data model:
        > `TPT-8B` (100B) approaches `LLaMA-3.1-8B` (15T) average 46.8.
  - After SFT on 2B-token Mixture-of-Thoughts (Table 2):
    - Quote:
      > AIME24: 35.2 vs 1.0 (`Vanilla-8B→SFT`), MATH-500: 82.4 vs 33.8, LCB: 23.4 vs 1.9, GPQA: 45.2 vs 27.7, MMLU-Pro: 59.8 vs 29.0.
    - Even exceeds `LLaMA-3.1-8B-Instruct` on all listed benchmarks (Table 2).
  - Pre-training under constrained unique data (10B raw tokens; 40B training budget; Figure 3; Table 7):
    - Despite seeing each raw document only once (thoughts make samples longer), `TPT` continues improving while vanilla saturates.
    - Quote (Table 7):
      > Average: 32.6 (`TPT-8B`) vs 16.6 (vanilla); GSM8K: 30.5 vs 6.7; MATH: 12.9 vs 4.8.
  - Mid-training across model families and sizes (Table 3):
    - `TPT-Qwen2.5-1.5B` vs OpenR1-1.5B:
      > MATH-500: 82.3 vs 79.6; AIME24: 28.5 vs 20.8; HEval: 63.4 vs 45.1; MMLU-Pro: 50.5 vs 43.5; JEEBench: 50.3 vs 38.2.
    - `TPT-LLaMA-3B` vs OpenR1-LLaMA-3B:
      > AIME24: 18.6 vs 5.8; AIME25: 17.5 vs 7.1; HEval: 65.2 vs 45.7; LCB: 20.0 vs 13.9; MMLU-Pro: 55.5 vs 45.8; JEEBench: 42.4 vs 26.6.
    - `TPT-Qwen2.5-7B` is strong and competitive with the teacher-style distillation model:
      > AIME24: 57.5 vs 53.2 (`DS-Distill-Qwen-7B`) and maintains high scores across tasks.
  - Ablations and scaling studies
    - Different thought generators (Table 4):
      > Back-thinking prompt and random focus point yield small changes; using a smaller 1.5B generator can improve several metrics over 7B.
    - Mid-training token budget (Figure 5):
      > Moving from 0B (direct SFT) to 100B thinking mid-training steadily improves AIME24, MATH-500, GPQA-Diamond, and LCB for both 1.5B and 3B models; for LLaMA-3B, AIME24 jumps by ~15 points.
    - SFT epochs (Figure 6):
      > More SFT epochs generally help; models with thinking mid-training start higher and stay better across 0.5–5 epochs, indicating that mid-training forms a stronger base.
    - Vanilla mid-training baseline (Appendix Table 6):
      > 40B tokens of ordinary text mid-training provides little benefit and can hurt coding performance vs direct SFT, highlighting that the gains come from thinking augmentation rather than extra tokens alone.
- Do the experiments support the claims?
  - Yes for data efficiency and cross-domain improvements: consistent gains across sizes, training stages, and benchmarks; strong math improvements and meaningful boosts in code and general knowledge after SFT.
  - The training-loss gap (Figure 2 left) supports the “more learnable” claim; the thinking-length analysis (Figure 4) supports the “dynamic compute allocation” mechanism.

## 6. Limitations and Trade-offs
- Dependence on synthetic thoughts
  - Quality of generated thinking is only as good as the teacher model and prompt; errors or hallucinations in trajectories can teach wrong reasoning patterns. The paper does not quantify thought correctness or noise rates.
- Compute and token budget implications
  - Thoughts lengthen each sample substantially (up to 8k extra tokens per document; Appendix A.1). Although the authors argue “3× data efficiency,” the absolute training FLOPs still increase per document; overall cost trade-off depends on budget and objectives.
- Style bias and distribution shift
  - The training distribution becomes “didactic” and tagged with `<think>`; this may bias generation style or encourage verbose reasoning even when not needed. The paper evaluates benefits but does not evaluate potential downsides like verbosity at inference.
- Scope of scaling evidence
  - Results scale to 100B tokens and 1.5B–8B models. It remains unknown whether the same multiplicative gains persist at multi-trillion-token, >30B-parameter scales (Conclusion acknowledges future scaling).
- Safety and contamination checks
  - No explicit analysis of safety filtering for generated thoughts or contamination risk from teacher models. While datasets are public, the paper does not audit overlap between training and evaluation beyond standard practices.
- Domain variance
  - Gains are largest in math and reasoning-heavy tasks; some general knowledge and coding gains require SFT and larger contexts. Pure base-model improvements on non-reasoning tasks (e.g., BoolQ) are present but smaller than math gains (Table 1).

## 7. Implications and Future Directions
- How this changes the landscape
  - Establishes a simple, scalable alternative to RL-based reasoning pre-training: augment data with thinking and keep standard next-token training. This lowers the barrier to improving reasoning capability and data efficiency with commodity infrastructure.
- Practical applications
  - Training compact models that perform well on math/coding without massive proprietary corpora.
  - Continual improvement of existing checkpoints via mid-training, then instruction tuning for chat assistants specialized in analytical tasks (education, research copilots, debugging assistants).
- Follow-up research enabled
  - Thought-quality modeling: automatic filters or verifiers to score thought usefulness/correctness before training.
  - Teacher–student matching: systematically study why a smaller thought generator can help (Table 4) and how to select or ensemble teachers.
  - Adaptive budgeting: control thought length by document difficulty to further optimize compute.
  - Integration with other pipelines: combine with rewriting/quality filtering (Section 6) or with RL-based post-training for hybrid benefits.
  - Prompt optimization and diversity: multiple prompts/thought styles, or “back-thinking” variants (Appendix A.3; Table 4) with principled selection.
  - Scaling studies: push beyond 100B tokens and larger model sizes, and measure effects on non-reasoning domains and safety.

Overall, `TPT` reframes “reasoning” as a data-engineering problem: expose the hidden steps during training so base models can learn difficult tokens more reliably. The approach’s simplicity, consistent empirical gains (Tables 1–3), and interpretability of its mechanism (Figure 4) make it a compelling addition to LLM training toolkits.
