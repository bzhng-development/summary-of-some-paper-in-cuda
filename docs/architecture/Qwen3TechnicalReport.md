# Qwen3 Technical Report

**ArXiv:** [2505.09388](https://arxiv.org/abs/2505.09388)

## 🎯 Pitch

Qwen3 introduces a single large language model family that seamlessly integrates both chat-style (non-thinking) and complex, step-by-step reasoning (“thinking”) capabilities, with a user-controllable 'thinking budget' to balance response speed and reasoning depth on demand. This unified architecture—spanning dense and Mixture-of-Experts models from 0.6B to 235B parameters and supporting 119 languages—eliminates the need to switch between separate chat and reasoning models, significantly reducing deployment complexity and inference costs while achieving state-of-the-art results across reasoning, coding, math, and multilingual benchmarks. Qwen3’s innovations inclusively empower both high-performance large models and competitive small-scale models, democratizing advanced AI for a global audience.

---

## 1. Executive Summary (2-3 sentences)
Qwen3 is an open-weight family of large language models that unifies two behaviors—concise answers (“non-thinking”) and explicit multi-step reasoning (“thinking”)—inside a single model, with a user-controllable “thinking budget” that caps how many reasoning tokens are used (Section 4.3, Table 9; Figure 2). Spanning dense and Mixture-of-Experts (MoE) models from 0.6B to 235B parameters and trained on 36 trillion tokens across 119 languages (Sections 2–3), Qwen3 achieves state-of-the-art or competitive results on coding, math, agent, and multilingual benchmarks, while offering strong small models through an efficient strong-to-weak distillation pipeline (Sections 3.3, 4.5; Tables 3–21).

## 2. Context and Motivation
- Problem addressed
  - Today’s best systems often separate fast, chat-optimized models from slow, reasoning-optimized models. Users and developers must switch models or maintain complex routing to balance speed vs. reasoning depth.
  - Prior open models also show large compute costs for high performance, and smaller open models typically lag behind proprietary systems.
  - Multilingual coverage in earlier open releases (e.g., Qwen2.5) is limited in language breadth.
- Why it matters
  - Many real-world applications need “on-demand reasoning”: everyday queries should be fast and cheap, whereas hard problems need step-by-step thinking. A single model that can switch modes and throttle its reasoning saves integration complexity and compute costs.
  - Widening multilingual support enables global deployment and fairer access.
- Where prior work falls short
  - Separate models for chat vs. reasoning force manual model selection and complicated serving. Dedicated reasoning models are slower and costly for simple tasks.
  - Smaller open models are expensive to post-train to high quality; RL-heavy pipelines are costly and unstable.
  - Multilingual support and long-context reasoning are uneven.
- Qwen3’s positioning
  - Unifies “thinking” and “non-thinking” in one architecture using simple control flags and a budget (Section 4.3, Table 9; Figure 2).
  - Introduces a cost-effective strong-to-weak distillation pipeline that transfers capability from larger to smaller models with ≈10× fewer GPU hours than RL (Table 21).
  - Substantially expands language coverage (119 languages) and long-context capability (up to 128K with scaling; Section 3.2, Appendix A.1.1 Table 23).

## 3. Technical Approach
Step-by-step overview of how Qwen3 is built and aligned.

- Model family and architectures (Section 2)
  - Dense models: `0.6B, 1.7B, 4B, 8B, 14B, 32B` (Table 1).
  - MoE models: `30B-A3B` and flagship `235B-A22B` with 128 total experts, 8 activated per token (Table 2).
    - MoE (Mixture-of-Experts): a network that routes each token to a subset of specialized “experts”; here, only 8 of 128 experts process each token, lowering active compute and memory.
  - Key design choices in dense models: Grouped Query Attention (GQA) for efficient attention; SwiGLU activation; Rotary Position Embeddings (RoPE); RMSNorm with pre-normalization; removal of QKV bias; addition of `QK-Norm` to stabilize attention (Section 2).
  - MoE specifics: fine-grained expert segmentation and a global-batch load-balancing loss to promote expert specialization (Section 2). Unlike Qwen2.5-MoE, Qwen3 MoE has no shared experts.

- Tokenizer and context length
  - Qwen BBPE tokenizer with 151,669 vocabulary size (Section 2).
  - Context: dense and MoE models support up to 128K context tokens; training extended via a “long context stage” and inference scaling techniques (Section 3.2; Appendix A.1.1).

- Pre-training corpus and procedure (Section 3)
  - Scale and diversity: 36T tokens covering 119 languages and dialects (Section 3.1).
  - Data sourcing:
    - Extract text from PDF-like documents using `Qwen2.5-VL`, then refine with `Qwen2.5` (Section 3.1).
    - Synthesize domain-specific data with `Qwen2.5-Math` and `Qwen2.5-Coder` (Section 3.1).
    - Multilingual annotation system assigns fine-grained labels (e.g., educational value, domain, safety) to >30T tokens, enabling instance-level mixture optimization (Section 3.1).
  - Three-stage pre-training (Section 3.2):
    1) General Stage S1: >30T tokens, 4,096 length; builds general knowledge across 119 languages.
    2) Reasoning Stage S2: +≈5T higher-quality, STEM/coding/synthetic-heavy tokens at 4,096 length; faster LR decay.
    3) Long-Context Stage: hundreds of billions of tokens with 32,768 length; 75% of samples are 16K–32K long.
       - RoPE base frequency increased from 10,000 to 1,000,000 via ABF; YARN and Dual Chunk Attention (DCA) enable 4× length scaling at inference (Section 3.2).
         - ABF: Adjusts positional embedding frequency range for better extrapolation.
         - YARN: A method to extend usable context length without retraining.
         - DCA: An attention scheme that divides context into chunks to manage long sequences efficiently.
  - Scaling laws: hyperparameters (e.g., learning rate, batch size) predicted per model and stage from extensive experiments (Section 3.2).

- Unifying “thinking” and “non-thinking” via post-training (Section 4; Figure 1)
  - Definitions
    - `Thinking mode`: model emits a hidden “reasoning transcript” inside `<think>...</think>` before giving the final answer.
    - `Non-thinking mode`: model answers directly; an empty `<think></think>` block is kept for consistent formatting.
    - `Thinking budget`: a user-defined cap on the number of tokens the model may spend inside `<think>...</think>` (Section 4.3 and Figure 2).
  - Four post-training stages (Figure 1):
    1) Long-CoT Cold Start (Section 4.1): Build initial long chain-of-thought skills using carefully filtered math, code, and logical problems with verifiable answers or unit tests. The dataset excludes trivial prompts and responses that can be solved without reasoning, avoiding “shortcut learning.”
    2) Reasoning RL (Section 4.2): Use GRPO (a policy-gradient-style method) on 3,995 “query–verifier” pairs not seen in Stage 1. Large batch sizes, many rollouts, off-policy training, and tuned entropy schedules stabilize learning. Result: AIME’24 accuracy on `Qwen3-235B-A22B` rises from 70.1 to 85.1 in 170 RL steps (Section 4.2).
    3) Thinking Mode Fusion (Section 4.3): Continual supervised fine-tuning (SFT) merges thinking/non-thinking behavior in one model.
       - Control flags in the chat template: `/think` and `/no think` (Table 9).
       - For non-thinking samples, include an empty `<think>` block to keep output format stable.
       - By default the model thinks; some SFT samples omit `/think` to teach this default.
       - The “thinking budget” behavior emerges naturally: when the `<think>` length hits the budget, the server inserts a stop instruction and the model moves to the final answer based on partial reasoning.
    4) General RL (Section 4.4): A multi-task reward suite across >20 tasks improves instruction-following, formatting (e.g., honoring `/think` flags), preference alignment, tool use (agents), and specialized scenarios like RAG. Rewards combine:
       - Rule-based checks (e.g., exact format/constraints),
       - Model-based with references (teacher compares to known answers),
       - Model-based without references (learned reward from human preferences).

- Strong-to-Weak Distillation for small models (Section 4.5)
  - Off-policy distillation: train students on teacher outputs in both `/think` and `/no think` to seed reasoning and mode-switching.
  - On-policy distillation: the student generates responses; training minimizes KL divergence to teacher logits, aligning the student’s probability distribution.
  - Outcome: Compared to reinforcement learning from the same 8B checkpoint, on-policy distillation achieves higher scores with ≈1/10 GPU hours (Table 21).

- Inference protocol for evaluations (Section 4.6)
  - Thinking mode: temperature 0.6, top-p 0.95, top-k 20; presence penalty 1.5 only for creative writing.
  - Non-thinking: temperature 0.7, top-p 0.8, top-k 20; presence penalty 1.5.
  - Output length 32,768 (extended to 38,912 on AIME tasks to allow more thinking).

## 4. Key Insights and Innovations
- Unified control of reasoning vs. speed inside one model
  - Novelty: Use of `/think` and `/no think` flags with an internal `<think>` channel and a “thinking budget” allows per-request control without switching models (Table 9; Section 4.3).
  - Why it matters: Simplifies deployment and enables adaptive compute—spend more tokens only when the task benefits from deeper reasoning (Figure 2 shows performance scales with larger budgets).
- Thinking Mode Fusion as a training recipe
  - Different from prior practice where separate chat and reasoning models are maintained, Stage 3 SFT fuses behaviors while preserving format consistency by always including a `<think>` block (Table 9). Stage 4 RL then reinforces instruction following, alignment, and tool use (Section 4.4, Table 22).
- Efficient strong-to-weak distillation
  - Significance: Students (e.g., 14B, 8B, 4B) inherit both reasoning and non-thinking abilities and achieve strong performance with far less compute than running a full RL pipeline (Table 21).
  - Distillation from teacher logits expands the student’s exploration space, improving not just pass@1 but also pass@64 in math/coding, unlike RL which did not improve pass@64 from the same starting point (Table 21).
- Scalable MoE with improved specialization and cost-efficiency
  - MoE design (128 total experts, 8 active) with global-batch load balancing improves specialization without shared experts (Section 2). Results show MoE base models match dense models with ≈1/5 activated parameters (Section 3.3, conclusions (2)).
- Substantially expanded multilingual coverage and long-context capability
  - Training on 119 languages and building instance-level mixtures with labeled data (Section 3.1) yields robust multilingual performance across instruction following, knowledge, math, and logical reasoning (Tables 10, 11–20; Appendix Tables 24–37).
  - Long-context methods (ABF, YARN, DCA) enable strong non-thinking performance on RULER up to 128K (Appendix Table 23).

## 5. Experimental Analysis
- Evaluation design (Sections 3.3 and 4.6; Tables 3–20)
  - Base (pretrained) models are tested on 15 benchmarks spanning general knowledge (MMLU, MMLU-Pro, MMLU-Redux, BBH, SuperGPQA), math/STEM (GPQA, GSM8K, MATH), coding (EvalPlus, MultiPL-E, MBPP, CRUX-O), and multilingual tasks (MGSM, MMMLU, INCLUDE).
  - Post-trained models are evaluated under both modes on broader suites: general tasks (MMLU-Redux, GPQA-Diamond, C-Eval, LiveBench), alignment (IFEval strict, Arena-Hard, AlignBench v1.1, Creative Writing v3, WritingBench), math and text reasoning (MATH-500, AIME’24/’25, ZebraLogic, AutoLogi), agent/coding (BFCL v3, LiveCodeBench v5, CodeForces Elo), and multilingual tasks (Multi-IF, INCLUDE, MMMLU-14, MT-AIME2024, PolyMath, MLogiQA). Sampling settings and output lengths are fixed per mode (Section 4.6).
  - AIME scoring uses 64 samples per question with averaged accuracy (Section 4.6). LiveCodeBench uses official prompts (non-thinking) and a freer prompt (thinking) to allow reasoning (Section 4.6).

- Main quantitative results
  - Base-model highlights
    - `Qwen3-235B-A22B-Base` vs strong open bases:
      - It leads on most of 15 benchmarks vs. DeepSeek-V3-Base, Qwen2.5-72B-Base, and Llama‑4-Maverick-Base (Table 3). For example:
        > MMLU-Pro: 68.18 (Qwen3-235B) vs 59.84 (DeepSeek‑V3), 63.91 (Llama‑4‑Maverick), 58.07 (Qwen2.5‑72B)  
        > EvalPlus (coding): 77.60 vs 63.75 (DeepSeek‑V3) and 68.38 (Llama‑4‑Maverick)
    - `Qwen3-32B-Base` competes with or beats larger models:
      - Outperforms Qwen2.5‑72B‑Base on 10/15 benchmarks despite <1/2 parameters (Table 4). Examples:
        > MMLU-Pro: 65.54 (Qwen3‑32B) vs 58.07 (Qwen2.5‑72B)  
        > MultiPL-E: 67.06 vs 58.70
      - Strong vs Llama‑4‑Scout-Base (MoE 109B total, 17B activated): wins on all 15 benchmarks (Table 4).
    - Smaller bases (8B/4B/1.7B/0.6B) show large gains over Qwen2.5 peers, especially in STEM/coding (Tables 6–8). Example for `Qwen3-8B-Base`:
      > MATH: 60.80 vs 49.80 (Qwen2.5‑7B) (Table 6).
  - Post-trained, thinking mode (Tables 11, 13, 15, 17, 19)
    - Flagship `Qwen3-235B-A22B (Thinking)` achieves:
      > AIME’24 85.7 and AIME’25 81.5; LiveCodeBench v5 70.7; BFCL v3 70.8; CodeForces Elo 2056 (98.2% percentile) (Table 11)  
      > Beats DeepSeek‑R1 on 17/23 benchmarks and is competitive with OpenAI‑o1, Grok‑3‑Beta(Think), and Gemini 2.5 Pro (Table 11).
    - `Qwen3-32B (Thinking)` surpasses QwQ‑32B on 17/23 benchmarks and rivals OpenAI‑o3‑mini (medium) especially on alignment and multilingual tasks (Table 13).
    - Distilled `Qwen3-14B (Thinking)` and `Qwen3-30B-A3B (Thinking)` are close to QwQ‑32B on reasoning and strong on agents/coding (Table 15).
    - Edge models `Qwen3-8B/4B (Thinking)` beat R1‑Distill baselines across many metrics, including AIME and ZebraLogic (Table 17).
  - Post-trained, non-thinking mode (Tables 12, 14, 16, 18, 20)
    - `Qwen3-235B-A22B (Non-thinking)`:
      > Exceeds GPT‑4o‑2024‑11‑20 on 18/23 benchmarks and beats leading open models like DeepSeek‑V3 and Llama‑4‑Maverick on many metrics (Table 12).  
      > For LiveBench: 62.5 vs 60.5 (DeepSeek‑V3) and 59.5 (Llama‑4‑Maverick).
    - `Qwen3-32B (Non-thinking)` performs on par with or better than Qwen2.5‑72B‑Instruct on general tasks and clearly stronger on alignment/multilingual metrics (Table 14).
    - Distilled `Qwen3-14B` and MoE `Qwen3-30B-A3B` beat Gemma‑3‑27B‑IT and Qwen2.5‑32B‑Instruct in many categories with fewer activated parameters (Table 16).
  - Multilingual breadth and depth
    - Benchmarks cover instruction following, knowledge, math, and logic across dozens of languages (Table 10). Qwen3 models show strong averages across Spanish, French, Portuguese, Italian, Arabic, Japanese, Korean, Indonesian, Russian, Vietnamese, German, Thai (Appendix Tables 24–35).
    - On the 80-language Belebele evaluation, `Qwen3-32B (Thinking)` achieves the highest or second-highest scores across most language families vs. Gemma‑3‑27B‑IT and Qwen baselines (Appendix Table 37).
  - Long-context ability
    - On RULER, non-thinking Qwen3 models outperform Qwen2.5 of similar sizes; thinking mode slightly degrades for retrieval-like tasks (Appendix Table 23).
  - Mode-fusion and general-RL effects (Table 22)
    - After Stage 3 and 4, large gains in instruction following, alignment, and tool-use stability:
      > IFEval strict prompt: +5.4 then +6.6 in thinking; ToolUse: +7.1 then +15.1 points (thinking).  
    - Trade-off: slight drops on hardest math/coding when compared directly to the Stage-2 reasoning specialist:
      > AIME’24 thinking: 83.8 → 81.9 → 81.4; LiveCodeBench thinking: 68.4 → 67.2 → 65.7 (Table 22).

- Do the experiments support the claims?
  - The breadth of baselines (open and closed) and consistent superiority on many metrics support claims of state-of-the-art open performance. The staged analysis (Table 22) transparently shows the fusion trade-off: general versatility improves, top-end math/coding peak slips slightly.
  - Distillation vs RL compute and quality is backed by Table 21 with concrete GPU-hour comparisons and pass@64 improvements.

- Ablations, failure cases, robustness
  - Ablation on training stages (Table 22) shows what each stage contributes.
  - Long-context thinking-mode degradation (Appendix Table 23) highlights a setting where reasoning verbosity can hurt retrieval tasks.
  - Figure 2 provides budget-to-performance scaling curves, validating the “pay for more thinking” premise.

## 6. Limitations and Trade-offs
- Compute and data scale
  - Pre-training uses 36T tokens and long sequences (Section 3), implying substantial compute and curation effort that may limit replication. Synthetic data from prior Qwen models could carry biases or contamination risks if not carefully filtered.
- Reasoning vs generality trade-off
  - Integrating non-thinking capabilities (Stages 3–4) slightly reduces peak scores on the hardest math/coding compared to the Stage-2 reasoning specialist (Table 22). Applications that prioritize maximum math/coding accuracy might prefer thinking mode with larger budgets, or a specialist checkpoint.
- Reliance on teacher quality for distillation
  - Strong-to-weak distillation inherits strengths and weaknesses of the teachers; performance ceilings for small models are bounded by teacher competence (Section 4.5, Table 21).
- Long-context reasoning interference
  - In thinking mode, long-context retrieval tasks see slight degradation (Appendix Table 23), suggesting the `<think>` stream can distract retrieval if not budgeted or prompted carefully.
- Controllability depends on prompting and server logic
  - Budget control uses an injected stop instruction (Section 4.3). This requires a serving layer that monitors token counts and edits the stream; robustness to prompt injection or adversarial inputs is not deeply analyzed here.
- Evaluation coverage
  - While benchmark coverage is broad, real-world agent safety, adversarial robustness, and cross-lingual safety are not comprehensively audited. Reward-model reliance (Stage 4) can introduce preference drift if not regularly recalibrated.

## 7. Implications and Future Directions
- How this changes the landscape
  - Qwen3 operationalizes “budgeted reasoning”: a single model that can toggle and meter its thinking per request, confirmed by smooth scaling in Figure 2. This creates a practical bridge between chat speed and reasoning depth without model switching.
  - Open availability (Apache 2.0) across sizes and architectures lowers barriers to building reasoning-capable systems, including edge-scale deployments (0.6B–8B) with strong-to-weak distillation.
- Follow-up research enabled
  - Automatic budget allocation: learn to predict the minimal thinking budget needed per query to hit a target accuracy/latency trade-off (building on Figure 2).
  - Better fusion strategies: preserve the Stage‑2 reasoning peak while adding non-thinking skills, perhaps via multi-head decoders or separate reasoning channels with shared backbones.
  - Long-context thinking improvements: mitigate interference observed in Appendix Table 23 via retrieval-aware `<think>` constraints or tool-mediated reasoning.
  - Distillation science: explore multi-teacher and cross-task distillation schedules to further uplift small models with minimal compute (Table 21 suggests large headroom).
  - Agent RL at scale: Section 5 (Conclusion) points to scaling agent RL with real environment feedback—fertile ground given strong BFCL and CodeForces results.
- Practical applications
  - Developer assistants and code agents: strong coding metrics (e.g., LiveCodeBench v5 70.7 and CodeForces 2056 for `Qwen3-235B-A22B`, Table 11) support IDE integration and competitive programming tutoring.
  - Math tutoring and STEM tools: high AIME/MATH-500 scores, with budgeted reasoning to fit latency constraints.
  - Multilingual assistants: 119-language coverage and strong Belebele results (Appendix Table 37) provide wide geographic applicability.
  - Enterprise chat and RAG: General RL’s format/instruction following and tool-use stability (Table 22) help build reliable agents with structured output and tool invocation.

Overall, Qwen3 combines architectural refinements, massive multilingual pre-training, a carefully staged alignment pipeline, and an efficient distillation method to deliver a single family of models that users can steer between speed and depth—validated across comprehensive benchmarks and ablations.
