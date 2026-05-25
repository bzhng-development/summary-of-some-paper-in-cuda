# THINKING AUGMENTED PRE-TRAINING

**ArXiv:** [2509.20186](https://arxiv.org/abs/2509.20186)

## 🎯 Pitch

This paper proposes Thinking-augmented Pre-Training (TPT), a simple yet transformative method that augments language model training data by appending automatically generated step-by-step reasoning—'thinking trajectories'—to each text sample. By making the implicit reasoning behind challenging tokens explicit, TPT dramatically boosts data efficiency and reasoning performance, allowing models to achieve state-of-the-art capabilities with up to 3× less training data—an essential advance as high-quality pre-training data becomes increasingly scarce.

---

## 1. Executive Summary
- The paper introduces Thinking-augmented Pre-Training (`TPT`), a simple data-engineering method that turns ordinary text into “explain-then-text” training samples by appending automatically generated reasoning (“thinking trajectories”) to each document, then trains with standard next-token prediction on the concatenated sequence.
- Across pre-training from scratch and continual “mid-training,” `TPT` yields large gains in reasoning-heavy benchmarks and sharply improves data efficiency (≈3× fewer training tokens to reach a given score; see Figure 1a and Figure 2). With only 100B training tokens, an 8B model approaches the performance of models trained on 15T tokens (Table 1).

## 2. Context and Motivation
- Problem addressed
  - LLMs need immense amounts of high-quality text, but the web’s “good” data is finite and increasingly exhausted. Merely scaling compute and tokens is not enough when many valuable tokens are hard to learn because they compress the result of multi-step reasoning into a single target token (Section 1; Figure 1b).
- Why this matters
  - Practically: future model scaling is constrained by data scarcity. Making each document more “learnable” increases the value extracted per token.
  - Conceptually: next-token prediction struggles when the next token implicitly encodes long chains of reasoning (e.g., answering “890” without the intermediate math; Figure 1b).
- Prior approaches and gaps
  - Data selection/prioritization keeps only “learnable” and “worth learning” tokens (e.g., Lin et al., 2024; Mindermann et al., 2022), but still asks the model to learn a difficult token in one step.
  - Test-time prompting (e.g., chain-of-thought) or RL-style “reasoning” training (OpenAI o1; DeepSeek-R1) increases inference compute, not training learnability.
  - Reinforcement Pre-Training (`RPT`) improves pre-training but is compute-heavy due to online rollouts and token-level control (Section 1).
  - Reasoning CPT and BoLT generate “latent thoughts,” but were evaluated at much smaller scales (≤8B tokens) or narrow domains (Section 1).
- Positioning of this work
  - `TPT` is a training-time, data-centric method that:
    - Requires no human labels and no online RL.
    - Works at document level for any text domain.
    - Scales to 100B training tokens and multiple model families (Sections 2–3).
    - Naturally allocates more training compute to harder content via longer generated thinking (Section 4).

## 3. Technical Approach
- Core idea in plain terms
  - For each document `d` in the pre-training corpus, automatically generate an expert-like “thinking trajectory” `t` that explains or reasons about `d`. Concatenate them into one sequence `x = [d; t]`. Train the model to predict every next token in `x` (Section 2).
- What is a “thinking trajectory”?
  - A model-generated, step-by-step analysis of the document that focuses on “complex and informative aspects” (prompt in Section 2). It is similar to chain-of-thought but is produced during data preparation, not at inference. Example shown in Figure 1b and Appendix Table 8.
- Generation prompt and controls
  - Prompt (Section 2): “Simulate an expert’s in-depth thought process… Use Feynman technique…”; for ablations, a “random focus point” variant is in Appendix A.3.
  - Practical settings (Appendix A.1): truncate document to ≤2k tokens; generate up to 8k thinking tokens with temperature 0.6 and top-p 0.9; stop at `</think>` to avoid redundant summaries.
  - Thinking models: Qwen3-8B for from-scratch pre-training; DeepSeek-R1-Distill-Qwen-7B for mid-training studies (Appendix A.1). Surprisingly, a 1.5B thinking generator sometimes works even better (Table 4).
- Training objective
  - Standard next-token prediction over the concatenated sequence:
    - Equation (1): minimize `- (1/N) * Σ log p(x_i | x_<i)` over all tokens of `[d; t]`.
  - No special losses, no RL, no extra supervision; this makes it easy to scale (Section 2).
- Why this design?
  - Document-level augmentation is scalable and agnostic to data format.
  - By breaking a “hard next token” into many explanatory steps, the model receives a path to generalize beyond memorizing answers (Figure 1b and Section 2 “Dynamic Allocation of Training Compute”).
  - Thinking trajectories are naturally longer for challenging domains, effectively up-sampling them with more training tokens (Section 4; Figure 4).
- Training setups used to test generality (Section 3; Appendix A.1, A.2):
  - From-scratch pre-training (8B models; 100B tokens; Figure 2; Table 1).
  - Constrained-data pre-training (limit raw documents to 10B tokens; 40B training budget; Figure 3; Table 7).
  - Mid-training (continual pre-training) of existing checkpoints: Qwen2.5-1.5B/7B and LLaMA‑3.2‑3B on 100B augmented tokens, then supervised fine-tuning (`SFT`) on the public 350k-sample “Mixture-of-Thoughts” dataset distilled from DeepSeek-R1 (Section 3.3; Table 3; Appendix A.1).
  - Implementation: MI300 GPUs; 8B pre-training to 100B tokens takes about a week; thinking-data generation ≈20k A100 GPU hours (Appendix A.1).

Analogy: If the original data asks “What’s 890?” the model must “jump” to the answer in one step. `TPT` teaches the model the staircase—polynomial division, Remainder Theorem, divisibility—so that predicting the final token becomes a sequence of small, learnable steps (Figure 1b; Appendix Table 8).

## 4. Key Insights and Innovations
- Thinking-augmented data as a universal, scalable pre-training format
  - Novelty: Append automatic, expert-like reasoning to any document, then train with the usual language-model objective (Section 2). No new loss, no task structure, no labels.
  - Why it matters: It turns hard-to-learn targets into decomposed sequences, improving generalization beyond memorization (Figure 1b).
- Dynamic training compute allocation emerges automatically
  - Observation: “High-value” documents (math, physics; advanced reasoning) produce longer thinking trajectories, thus receiving more training tokens and compute (Section 4; Figure 4 shows ≈50% longer thoughts for “Advanced Reasoning” vs “No Reasoning”).
  - Significance: Mimics “test-time scaling” benefits (longer reasoning improves accuracy) but moves them into training, improving learnability with fixed inference cost if desired (Section 2).
- Data efficiency at scale
  - Claim grounded by results: With 100B tokens, `TPT-8B` reaches an average score 43.9 vs 26.2 for a vanilla 8B trained identically (Table 1) and trends toward models trained on orders of magnitude more data (e.g., LLaMA‑3.1‑8B trained on 15T tokens scores 46.8; Table 1; Figure 2).
  - Figure 1a visualizes ≈3× data efficiency.
- Simplicity over RL or EM pipelines, yet strong gains across stages
  - In contrast to RL-based `RPT` or EM-style bootstrapping in BoLT, `TPT` requires only generation + standard training, yet improves from-scratch pre-training and mid-training, and continues to help after SFT (Sections 3.1–3.3; Tables 1–3, Figures 2–3).

## 5. Experimental Analysis
- Evaluation methodology
  - Base-model (no SFT) evaluation (Appendix A.2): average across five datasets—`GSM8K`, `MATH`, `BoolQ`, `MMLU`, `MMLUPro`—with specified shot settings and strict answer extraction (Section 3.1; Figure 2; Table 1).
  - Post-SFT evaluation: ten challenging benchmarks spanning math (`MATH-500`, `AIME24`, `AIME25`, `GSM8K`, `HMMT`), code (`HumanEval`, `LiveCodeBench v4/v5`), and general reasoning (`GPQA-Diamond`, `MMLUPro`, `JEEBench`) using Pass@1; multiple samples per question for stability (Section 3.3; Appendix A.2; Table 3).
- Main results, with specific numbers
  - From-scratch pre-training (abundant data; 100B tokens):
    - Training loss: `TPT` has far lower loss, signaling more learnable/less noisy data (Figure 2, left; authors caution distributions differ).
    - Downstream performance curve: `TPT` overtakes vanilla after ≈20B tokens and widens the gap to 100B (Figure 2, right).
    - Final scores (Table 1):
      > `Vanilla-8B (100B)` average 26.2 vs `TPT‑8B (100B)` 43.9. On math: `GSM8K` 19.2→50.1, `MATH` 9.1→21.8.  
      > `LLaMA‑3.1‑8B (15T)` achieves 46.8—only modestly above `TPT‑8B` trained on 150× fewer tokens.
  - Constrained-data setting (only 10B raw-document tokens available; 40B training budget):
    - `TPT` keeps improving while vanilla plateaus (Figure 3).
    - Final (Table 7):
      > `TPT‑8B (40B)` average 32.6 vs `Vanilla‑8B (40B)` 16.6. Notably `GSM8K` 30.5 vs 6.7 and `MATH` 12.9 vs 4.8.
  - Mid-training + SFT (Table 3):
    - Consistent gains across model sizes and families. Examples:
      > `TPT‑LLaMA‑3B` vs OpenR1‑LLaMA‑3B on `AIME24`: 18.6 vs 5.8; on `MMLUPro`: 55.5 vs 45.8; on `GPQA‑D`: 41.7 vs 32.8.  
      > `TPT‑Qwen2.5‑7B` vs OpenR1‑7B on `AIME24`: 57.5 vs 50.5; on `GPQA‑D`: 54.7 vs 52.1; on `JEEBench`: 73.6 vs 69.1.
    - With SFT, a `TPT`-pretrained 8B surpasses `LLaMA‑3.1‑8B‑Instruct` on all five reported benchmarks (Table 2):
      > AIME24: 35.2 vs 5.4; MATH‑500: 82.4 vs 49.4; LCB: 23.4 vs 9.4; GPQA: 45.2 vs 31.4; MMLUPro: 59.8 vs 43.6.
- Ablations and analyses
  - Thinking generator variations (Table 4):
    - “Back-thinking” (fine-tuned generator that writes thoughts inside `<think>...</think>`) and “random focus” prompt yield similar performance to default—small deltas.
    - Smaller thinking generator (`DeepSeek‑R1‑Distill‑Qwen‑1.5B`) sometimes outperforms 7B (e.g., `AIME24` 17.7 vs 11.7), suggesting “simpler thoughts” may be easier to learn.
  - Token budget scaling for mid-training (Figure 5):
    > Increasing from 0→100B thinking-augmented tokens steadily raises scores on `AIME24`, `MATH-500`, `GPQA‑D`, `LiveCodeBench` for both 1.5B and 3B models.
  - SFT epochs (Figure 6):
    > Without mid-training, a 3B base barely solves `AIME24`; with `TPT` mid-training, performance starts higher and stays higher across 0.5–5 SFT epochs. SFT alone is insufficient for strong reasoning.
  - Vanilla mid-training control (Table 6):
    > Continual training on plain text (40B) does not help and even hurts code (e.g., `LCB` 5.7) compared to direct SFT (13.9). This isolates thinking augmentation as the driver of gains.
  - Thinking-length distribution (Section 4; Figure 4):
    > Math and physics documents trigger the longest thoughts; “Advanced reasoning” has ≈50% longer thoughts than “No reasoning,” effectively up-sampling hard content.
- Do the experiments support the claims?
  - The breadth (from-scratch, constrained data, mid-training, SFT) and consistent margins—especially in math/code/general-reasoning suites—provide strong evidence that `TPT` improves learnability and data efficiency.
  - Caveat: training-loss curves are not directly comparable due to distribution differences (Figure 2 caption); nonetheless the downstream metrics and multiple baselines (Tables 1–3) are compelling.

## 6. Limitations and Trade-offs
- Dependence on thought quality
  - Generated thinking can be verbose, partially incorrect, or stylistically biased. The method assumes that “explanation is easier to learn than answer,” even if explanations are imperfect (Section 2; examples in Appendix A.5). No explicit mechanism filters incorrect thoughts.
- Compute and memory overhead
  - Thinking trajectories can add up to 8k tokens per sample (Appendix A.1), increasing training tokens and sequence lengths. Although this is the lever that boosts learnability, it raises training cost and may require long-context models (they train with 8k context for pre/mid-training; 32k for SFT/inference; Table 5).
  - Data generation itself is non-trivial (≈20k A100 GPU hours; Appendix A.1).
- Potential distribution shift
  - Longer thoughts disproportionately up-sample math/advanced content (Figure 4). While beneficial for reasoning, it could distort domain balance if not managed (they do apply sample weights when mixing datasets; Appendix A.1).
- Scale beyond 100B unknown
  - Results scale cleanly up to 100B mid-training tokens (Figure 5), but behavior at trillion-token scale remains to be validated.
- Inference-time behavior
  - Many evaluations allow long thinking at inference (up to 32k tokens; Section 3.3), which can increase latency/cost. It remains an open question how well `TPT` transfers when constrained to short outputs.
- Generalization to non-reasoning tasks
  - Benchmarks emphasize math/code/general reasoning. Effects on style, summarization, or safety-alignment tasks are not reported.

## 7. Implications and Future Directions
- How this changes the landscape
  - `TPT` reframes “data scaling” as “explanation scaling”: instead of scraping more web text, create more learnable content per document. This offers a path to stronger reasoning without trillion-token corpora (Figure 1a; Table 1).
  - It bridges a gap between test-time chain-of-thought and training: the model practices decomposed reasoning during pre-training/mid-training, not only when prompted at inference.
- Practical applications
  - Upgrading existing open-source models via mid-training to achieve large reasoning gains before SFT (Table 3) is attractive for organizations without the budget for massive pre-training runs.
  - Domains that benefit most—STEM education tools, math/code assistants, scientific reading and verification—align with where thinking trajectories naturally lengthen (Figure 4).
- Follow-up research
  - Thought quality control: automatic correctness checks, verifiers, or self-consistency filters to prune harmful thoughts.
  - Budgeted thinking: adaptively choose thought length per document to trade off compute and benefit; e.g., learn a policy to allocate thinking tokens.
  - Co-evolution of thought generator and learner: iterate generation with the current model’s weaknesses (EM- or RL-style) while keeping the pipeline simple.
  - Prompt/program synthesis: diversify thinking styles (proofs, counterexamples, sketches) beyond a single prompt (Appendix A.3 hints at random focus).
  - Transfer under constrained inference: train with thoughts but distill into short-answer variants that keep the gains with minimal inference cost.

Overall, the paper demonstrates a clear, scalable mechanism—attach generated explanations to training data—that substantially improves reasoning and data efficiency. The method’s simplicity, strong empirical results (Figures 1–3; Tables 1–3), and analyses (Figure 4; Table 4; Figures 5–6) make it a practical addition to large-scale LLM training pipelines, while leaving rich avenues for refining thought generation quality, adaptivity, and efficiency.
