# Qwen3 Technical Report

**ArXiv:** [2505.09388](https://arxiv.org/abs/2505.09388)
**Authors:** An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, Chujie Zheng, Dayiheng Liu, Fan Zhou, Fei Huang, Feng Hu, Hao Ge, Haoran Wei, Huan Lin, Jialong Tang, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Yang, Jiaxi Yang, Jing Zhou, Jingren Zhou, Junyang Lin, Kai Dang, Keqin Bao, Kexin Yang, Le Yu, Lianghao Deng, Mei Li, Mingfeng Xue, Mingze Li, Pei Zhang, Peng Wang, Qin Zhu, Rui Men, Ruize Gao, Shixuan Liu, Shuang Luo, Tianhao Li, Tianyi Tang, Wenbiao Yin, Xingzhang Ren, Xinyu Wang, Xinyu Zhang, Xuancheng Ren, Yang Fan, Yang Su, Yichang Zhang, Yinger Zhang, Yu Wan, Yuqiong Liu, Zekun Wang, Zeyu Cui, Zhenru Zhang, Zhipeng Zhou, Zihan Qiu
**Institutions:** 

## 🎯 Pitch

Qwen3 introduces a groundbreaking unified framework for large language models that seamlessly integrates rapid response and deep reasoning, allowing users to adjust the 'thinking budget' to balance speed and accuracy. This innovation not only reduces the need for multiple models and optimizes compute cost but also democratizes access with multilingual capabilities, making it a versatile tool for diverse global applications and pushing the boundaries of open-source AI.

---

## 1. Executive Summary
Qwen3 is an open-weight family of large language models that unifies “thinking” (explicit multi-step reasoning) and “non-thinking” (short, direct answers) in a single model and lets users control a “thinking budget” to trade off latency for accuracy. Spanning dense and Mixture-of-Experts (MoE) variants from 0.6B to 235B parameters, Qwen3 is pre-trained on 36 trillion tokens across 119 languages and shows state-of-the-art or highly competitive results across reasoning, coding, agents, and multilingual benchmarks (e.g., Table 11–20).

## 2. Context and Motivation
- Problem addressed
  - Today, developers often switch between a chat-optimized model for fast responses and a separate reasoning model for hard problems. This is cumbersome to deploy, introduces latency overhead, and wastes compute when heavy reasoning isn’t needed. Qwen3 targets this gap by integrating both behaviors in one model with user control over how much “thinking” to allocate (Introduction; Section 4.3 “Thinking Mode Fusion”).
- Why it matters
  - Real-world systems must adapt to mixed workloads: quick answers for simple queries and deep reasoning for complex tasks (e.g., math/coding/agents). Compute costs and latency vary widely by task difficulty; fine control over inference-time effort can cut costs without sacrificing accuracy when it matters (Abstract; Figure 2).
  - Open, multilingual capability (119 languages) broadens accessibility and reproducibility for research and global applications (Abstract; Section 3.1).
- Prior approaches and their limits
  - Proprietary reasoning models (e.g., o1/o3 series) and open ones (e.g., DeepSeek-R1) improved intelligence via reinforcement learning but require separate models and do not expose a simple, production-ready interface to turn reasoning on/off with budgets (Introduction; Sections 4.1–4.2).
  - Earlier Qwen2.5 models offered strong chat or task-specific variants (e.g., Qwen2.5-Math/Coder) but still separated reasoning and non-reasoning capabilities across models (Introduction).
- Positioning
  - Qwen3 offers: (a) unified modes via a chat template switch, (b) thinking budgets, (c) a strong-to-weak distillation pipeline to cheaply produce capable small models, and (d) MoE improvements for cost-effective scaling (Sections 2, 3, 4).

## 3. Technical Approach
Qwen3 is a model family (dense and MoE) plus a training pipeline (pre-training and post-training) designed to deliver controllable reasoning.

- Model family and architecture (Section 2, Tables 1–2)
  - Dense models: `Qwen3-0.6B/1.7B/4B/8B/14B/32B` with up to 128K context.
    - Uses `GQA` (Grouped Query Attention) to reduce KV cache memory while preserving multi-head benefits; `SwiGLU` feed-forward layers; `RoPE` positional encoding; `RMSNorm` with pre-normalization; removes `QKV` bias; and adds `QK-Norm` to stabilize attention (Table 1).
    - Context extension to 128K is enabled at inference by `YARN` and `Dual Chunk Attention (DCA)` after raising RoPE’s base frequency via `ABF` (Sections 3.2 and A.1.1).
  - MoE models: `Qwen3-30B-A3B` and flagship `Qwen3-235B-A22B` with 128 experts, 8 active per token (22B activated params for the flagship). They use fine-grained expert segmentation (specializing sub-parts of layers) and a global-batch load balancing loss to encourage expert specialization; unlike Qwen2.5-MoE, shared experts are removed (Section 2; Table 2).
  - Tokenizer: byte-level BPE with 151,669 vocabulary (Section 2).
  - Definitions
    - `Mixture-of-Experts (MoE)`: A layer with many specialized sub-networks (“experts”). A router chooses a few experts per token; only those are evaluated, reducing cost per token.
    - `QK-Norm`: Normalizes query/key vectors before attention dot-products to stabilize training (Section 2).
    - `ABF`, `YARN`, `DCA`: Techniques to extend context. ABF raises RoPE’s base frequency to preserve positional resolution at long lengths; YARN rescales positions at inference; DCA processes long inputs by chunking attention efficiently (Section 3.2; A.1.1).
- Pre-training data and stages (Sections 3.1–3.2)
  - Scale and diversity: 36T tokens across 119 languages and domains (code, STEM, books, reasoning, synthetic). Additional tokens extracted from PDFs by a vision-language model (Qwen2.5-VL) then refined by Qwen2.5; synthetic data from math/coder models expands scarce domains (Section 3.1).
  - Instance-level mixture control: a multilingual annotation system labels 30T tokens across quality, domain, education value, and safety; mixtures are tuned at instance level using ablations on proxy models (Section 3.1).
  - Three stages:
    1) General (S1): >30T tokens at 4k context for broad knowledge and multilinguality.
    2) Reasoning (S2): +~5T higher-quality STEM/coding/reasoning/synthetic tokens at 4k, with faster learning rate decay to sharpen reasoning.
    3) Long-context: hundreds of billions of long documents up to 32,768 tokens (75% are 16k–32k). RoPE base raised from 10k to 1,000,000 (ABF), with YARN and DCA for 4× extrapolation at inference (Section 3.2).
  - “Token” denotes a unit of text the model reads/writes (e.g., a subword).
- Post-training pipeline (Figure 1; Sections 4.1–4.6)
  - Stage 1 — Long-CoT cold start (Section 4.1)
    - Build a curated math/code/STEM dataset with verifiable answers. Two filters:
      - Query filter: drop unverifiable or multi-subquestion prompts and easy ones solvable without chain-of-thought (CoT).
      - Response filter: from N candidates by `QwQ-32B`, keep only correct, non-repetitive, non-guessy, consistent, and stylistically clean reasoning paths; human checks when needed.
    - Train briefly to imprint long-CoT patterns without overfitting (so later RL can improve).
    - Definition: `Chain-of-Thought (CoT)`: a detailed step-by-step reasoning trace that leads to an answer.
  - Stage 2 — Reasoning RL (Section 4.2)
    - Use 3,995 “query–verifier” pairs not seen before, chosen to be learnable-yet-hard and to cover sub-domains.
    - Optimize with `GRPO` (a policy-gradient method akin to PPO tailored to group-relative rewards), large batch sizes and many rollouts per query; mix in off-policy updates; control policy entropy for stable exploration.
    - Reported effect: 
      > “AIME’24 score of `Qwen3-235B-A22B` increases from 70.1 to 85.1 over 170 RL steps.” (Section 4.2)
  - Stage 3 — Thinking Mode Fusion (Section 4.3; Table 9)
    - Goal: integrate “non-thinking” capability without losing the improved thinking model.
    - Supervised fine-tuning (SFT) on a dataset that mixes thinking and non-thinking examples.
      - Interface: user can add `/think` or `/no think` in system/user messages; the assistant wraps internal reasoning in `<think>...</think>`. For non-thinking, the assistant emits an empty think block to keep internal format consistent (Table 9).
    - `Thinking budget`: allow the model to think up to a token limit; when limit is reached, inject a “stop-thinking” instruction and force a final answer based on partial reasoning:
      > “Considering the limited time by the user, I have to give the solution based on the thinking directly now.</think>...” (Section 4.3)
    - Key observation: budgeted answering works without special training—the fusion step enables it to emerge.
  - Stage 4 — General RL (Section 4.4)
    - Broaden capabilities and robustness across ~20 task types:
      - Instruction/format following (e.g., honoring `/think` flags, using `<think>` tags correctly).
      - Preference alignment (helpfulness and style).
      - Agent ability (tool calling over multi-turn interactions with environment feedback).
      - Scenario-specific tasks like RAG with hallucination penalties.
    - Rewarding strategies: (1) rule-based checks, (2) model-based scoring against references, and (3) preference-trained reward models (no references) (Section 4.4).
  - Strong-to-Weak Distillation (Section 4.5)
    - Purpose: cheaply produce capable small models that inherit both reasoning and mode-switching.
    - Two phases:
      - Off-policy distillation: feed student both `/think` and `/no think` teacher outputs to bootstrap both skills.
      - On-policy distillation: let the student generate; align student logits to teacher logits via KL divergence, per prompt and per requested mode.
    - Definitions
      - `Logits`: the unnormalized scores before softmax; matching logits aligns token-by-token predictions.
      - `KL divergence`: a measure of how one probability distribution differs from another (used to train the student to mimic the teacher).
      - `on-policy` vs `off-policy`: learning from the student’s own rollouts vs from externally provided teacher trajectories.

## 4. Key Insights and Innovations
- Unified “thinking vs non-thinking” with user control (fundamental)
  - What’s new: a single model obeys `/think` or `/no think` flags and supports a “thinking budget” (Table 9; Section 4.3). Non-thinking uses an empty `<think>` block to keep internal structure consistent.
  - Why it matters: no more model-switching in production; developers can dial up reasoning only when needed. Figure 2 shows smooth accuracy improvements as the thinking budget increases on AIME’24, AIME’25, LiveCodeBench v5, and GPQA-Diamond.
- Distillation-first recipe for small models (fundamental)
  - What’s new: a combined off-policy/on-policy distillation pipeline that transfers both reasoning depth and controllable modes from large teachers (Section 4.5).
  - Why it matters: cuts the cost of building competitive small models dramatically. Table 21 shows on-policy distillation from an 8B checkpoint outperforms additional RL while using ~1/10 GPU hours, and increases exploration headroom (Pass@64).
- MoE design and training refinements (incremental but important)
  - No shared experts; fine-grained expert segmentation; and a global-batch load-balancing loss encourage sharper specialization (Section 2). The flagship `Qwen3-235B-A22B` activates only 22B parameters per token (Table 2), achieving strong accuracy with lower per-token compute than similarly strong models (Tables 11–12).
- Data and long-context engineering at scale (incremental with broad impact)
  - 36T multilingual tokens (119 languages), instance-level mixture optimization, and high-quality synthetic/stem/code corpora (Section 3.1).
  - Long-context capability (128K) with ABF+YARN+DCA; RULER results in Appendix A.1.1 show solid long-context performance in non-thinking mode (Table 23).

## 5. Experimental Analysis
- Evaluation setup (Section 4.6; Tables 11–20; A.1)
  - Categories and datasets:
    - General: MMLU-Redux, GPQA-Diamond, C-Eval, LiveBench (2024-11-25).
    - Alignment: IFEval (strict prompt), Arena-Hard, AlignBench v1.1; writing: Creative Writing v3, WritingBench.
    - Math & text reasoning: MATH-500, AIME’24/’25, ZebraLogic, AutoLogi.
    - Agent & coding: BFCL v3 (function calling), LiveCodeBench v5 (2024.10–2025.02), Codeforces Elo (CodeElo).
    - Multilingual: Multi-IF (8 langs), INCLUDE (44), MMMLU (14), MT-AIME2024 (55), PolyMath (18), MLogiQA (10) (Table 10).
  - Inference settings (Section 4.6):
    - Thinking mode: temperature 0.6, top-p 0.95, top-k 20; output length up to 32,768 (38,912 on AIME).
    - Non-thinking: temperature 0.7, top-p 0.8, top-k 20; presence penalty 1.5; same output lengths.
- Headline results (post-trained)
  - Flagship `Qwen3-235B-A22B (Thinking)` vs strong reasoning baselines (Table 11):
    - Math/agents/coding:
      > AIME’24: 85.7 vs 79.8 (DeepSeek-R1)  
      > AIME’25: 81.5 vs 70.0 (DeepSeek-R1)  
      > BFCL v3: 70.8 vs 56.9 (DeepSeek-R1)  
      > LiveCodeBench v5: 70.7 vs 64.3 (DeepSeek-R1)  
      > Codeforces Elo: 2056 (98.2%ile) vs 2029 (98.1%ile, DeepSeek-R1)
    - Text reasoning (mixed):
      > GPQA-Diamond: 71.1 vs 71.5 (DeepSeek-R1), below Gemini 2.5-Pro at 84.0.
    - Multilingual strengths:
      > MT-AIME2024: 80.8 vs 73.5 (DeepSeek-R1) and 76.9 (Gemini 2.5-Pro)  
      > MLogiQA: 77.1 vs 73.8 (DeepSeek-R1), 75.6 (Gemini 2.5-Pro)
    - Takeaway: open-source state-of-the-art overall among peers, especially in math/agents/coding; slightly behind top proprietary models on some knowledge QA (e.g., GPQA-Diamond).
  - `Qwen3-235B-A22B (Non-thinking)` vs strong chat baselines (Table 12):
    - Alignment and general:
      > Arena-Hard: 96.1 (tops all listed models)  
      > LiveBench: 62.5 vs 60.5 (DeepSeek-V3) and 52.2 (GPT-4o 2024-11-20)
    - Reasoning without thinking (hard): still better than open sources but far below thinking mode:
      > AIME’24: 40.1 vs 39.2 (DeepSeek-V3); both much lower than their thinking scores.
- Mid-size `Qwen3-32B`
  - Thinking (Table 13): consistently beats `QwQ-32B`.
    > AIME’25: 72.9 vs 69.5 (QwQ-32B)  
    > BFCL v3: 70.3 vs 66.4  
    > ZebraLogic: 88.8 vs 76.8  
    > Arena-Hard: 93.8 vs 89.5
  - Non-thinking (Table 14): approaches or beats much larger `Qwen2.5-72B-Instruct` on many practical metrics.
    > Arena-Hard: 92.8 vs 81.2 (Qwen2.5-72B-Instruct)  
    > AutoLogi: 78.5 vs 66.1  
    > Multi-IF: 70.7 vs 65.3
- Lightweight models (Tables 15–20)
  - `Qwen3-30B-A3B (Thinking)` is comparable to `QwQ-32B` on reasoning with fewer activated parameters:
    > AIME’24: 80.4 vs 79.5; ZebraLogic: 89.5 vs 76.8; BFCL v3: 69.1 vs 66.4 (Table 15).
  - `Qwen3-14B (Thinking)` also competes strongly:
    > AIME’25: 70.4; BFCL v3: 70.4 (Table 15).
  - Non-thinking variants of 14B/30B outperform prior Qwen2.5-32B on many tasks, including alignment and LiveBench (Table 16).
  - Edge-scale 8B/4B/1.7B/0.6B show consistent gains over prior models at similar or larger sizes (Tables 17–20).
- Base (pre-trained) models (before instruction-tuning)
  - Despite fewer activated parameters, `Qwen3-235B-A22B-Base` tops most open-source bases, beating DeepSeek-V3 Base on 14/15 benchmarks (Table 3). Examples:
    > MMLU-Pro: 68.18 vs 59.84 (DeepSeek-V3 Base)  
    > EvalPlus: 77.60 vs 63.75  
    > MATH: 71.84 vs 62.62
  - `Qwen3-32B-Base` vs `Qwen2.5-72B-Base`: wins in 10/15 metrics despite less than half the parameters; large gains on reasoning, coding, and multilingual (Table 4).
- Ablations and robustness checks
  - Thinking budget scaling: Figure 2 shows monotonic gains with more thinking tokens across AIME’24, AIME’25, LiveCodeBench v5, GPQA-Diamond.
  - Distillation vs RL cost/benefit (Table 21):
    > On-policy distillation: AIME’24 74.4 (pass@64=93.3) with ~1,800 GPU hours  
    > RL: AIME’24 67.6 (pass@64=90.0) with ~17,920 GPU hours  
    Distillation is both stronger and ~10× cheaper in this setup.
  - Stage-wise effects (Table 22 for 32B):
    - Thinking Mode Fusion (Stage 3) improves instruction following and general tasks (e.g., IFEval +5.4, LiveBench +2.3) and enables mode switching (ThinkFollow 88.7), but slightly reduces top-end math/coding thinking scores (AIME’24 −1.9; LiveCodeBench −1.2).
    - General RL (Stage 4) further boosts alignment and agent/tool reliability (ToolUse +15.1, Arena-Hard +4.4), and nearly eliminates mode-switching errors (ThinkFollow 98.9), while math/coding thinking scores stay similar or slightly down.
  - Long-context results: RULER shows strong non-thinking performance (e.g., `Qwen3-235B-A22B` avg 95.0; Table 23). In thinking mode, long-context retrieval tasks degrade somewhat—likely because added “thinking” doesn’t help retrieval and can distract (A.1.1).

Assessment: The experiments are extensive, cover many capabilities, and include ablations on budgets, training stages, and distillation. Claims of unified controllable thinking, cost-effective scaling, and multilingual breadth are well-supported with numbers across multiple benchmarks and sizes (Tables 3–23, Figure 2). Where results are mixed (e.g., slight regressions on math/coding after Stage 3/4; GPQA-Diamond vs top proprietary models), the paper presents the trade-offs explicitly (Table 22; Table 11).

## 6. Limitations and Trade-offs
- Mode fusion trade-off
  - Integrating non-thinking and general alignment via SFT/RL can slightly degrade peak math/coding reasoning performance (Table 22; AIME’24 −0.5 from Stage 3 to Stage 4; LiveCodeBench −1.5). The paper accepts this for overall versatility.
- Thinking harms some retrieval tasks
  - RULER long-context in thinking mode is weaker than non-thinking, suggesting reasoning verbosity can interfere with pure retrieval (Appendix A.1.1, Table 23).
- RL data scope
  - Reasoning RL uses 3,995 curated query–verifier pairs (Section 4.2). While high-quality, this is small relative to the breadth of tasks; generalization and coverage depend on distillation and broader SFT/RL in Stage 4.
- Budget control granularity
  - The “thinking budget” is a token limit plus a textual stop instruction (Section 4.3). It works well (Figure 2), but it is a heuristic: it doesn’t guarantee the model allocates tokens to the “right” sub-problems or avoids wasteful digressions.
- Compute and engineering complexity
  - Training involves large-scale data pipelines, MoE routing with global-batch balancing, multi-stage RL, and distillation. Although distillation reduces costs substantially for small models (Table 21), pre-training and flagship post-training remain compute-intensive.
- Benchmark coverage vs reality
  - Despite many benchmarks, real-world agent tasks, safety, and tool ecosystems are vast. The reward design (Section 4.4) is comprehensive but inevitably incomplete; out-of-distribution behaviors will require further stress testing.

## 7. Implications and Future Directions
- How the work shifts the field
  - Production-friendly controllable reasoning: `/think` and a thinking budget provides a simple, standardizable interface for developers to dial performance vs latency. This can become a common pattern in LLM APIs.
  - Open, strong MoE at scale with efficient activation (22B per token) narrows the open-vs-closed gap on difficult reasoning and coding tasks (Tables 11–12).
  - A general recipe for strong small models: the off-policy + on-policy distillation pipeline is a practical blueprint for creating capable edge models at a fraction of RL cost (Section 4.5; Table 21).
- Follow-up research
  - Smarter budget policies: learn to allocate thinking tokens adaptively within a response (not just a hard cap), possibly with meta-controllers or uncertainty estimates.
  - Preserve peak reasoning while aligning: explore multi-objective or modular training so Stage 3/4 improvements don’t erode top math/coding performance (Table 22).
  - Thinking-aware retrieval: design inference-time strategies that suppress unnecessary thinking during retrieval-heavy segments to avoid RULER-style degradations (A.1.1).
  - Longer contexts and multimodality: extend beyond 128K context and integrate the same controllable thinking interface into multimodal pipelines (Appendix A; Introduction).
  - Agent RL at scale: expand Stage 4’s agent training with richer tool environments and long-horizon tasks (Conclusion).
- Practical applications
  - Enterprise assistants that toggle depth on demand (cost-efficient helpdesk, code review, analytics).
  - Developer tools and code generation with controlled deliberation for hard cases (LiveCodeBench, Codeforces results in Tables 11, 13, 17).
  - Multilingual education, translation, and STEM tutoring in 100+ languages (Tables 10, 24–37).
  - Function-calling and workflow orchestration with reliability gains from Stage 4 rewards (BFCL v3, ToolUse in Table 22).

Overall, Qwen3 contributes a coherent system—models, data, training, and interface—that makes controllable reasoning practical and affordable across scales, while remaining open and multilingual. The experiments demonstrate strong average performance and clear cost/performance levers, and they surface concrete trade-offs that motivate the next round of research.
