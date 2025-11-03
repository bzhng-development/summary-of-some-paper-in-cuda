# EXAONE 4.0: Unified Large Language Models Integrating Non‑reasoning and Reasoning Modes

**ArXiv:** [2507.11407](https://arxiv.org/abs/2507.11407)
**Authors:** LG AI Research
**Institutions:** LG AI Research

## 🎯 Pitch

EXAONE 4.0 revolutionizes large language models by seamlessly integrating fast utility and deep reasoning modes within a single system, scaling up to 128K token contexts with an efficient hybrid attention design. This breakthrough not only streamlines real-world applications like legal and financial document processing but also enhances AI's role in math, coding, and multilingual tasks, significantly reducing the need for multiple specialized models.

---

## 1. Executive Summary (2–3 sentences)
EXAONE 4.0 introduces a single family of large language models that can operate in two complementary modes—`NON-REASONING` (fast, concise answers) and `REASONING` (deliberate, step-by-step thinking)—while supporting long contexts (up to 128K tokens), agentic tool use, and three languages (English, Korean, Spanish). The paper’s significance lies in unifying these modes in one model, scaling long-context capability with a cost-aware hybrid attention design, and improving test-time reasoning through a new reinforcement learning algorithm, `AGAPO`, which together deliver top-tier math/coding results and competitive tool use versus much larger models (Tables 3–6).

## 2. Context and Motivation
- Problem addressed
  - Bridging a usability gap: earlier EXAONE variants specialized separately in instruction following (EXAONE 3.5) and deep reasoning (EXAONE Deep). Many real applications need both rapid utility and robust reasoning without switching models.
  - Preparing for “agentic AI”: practical deployment requires reliable tool calling over multi-turn, long-horizon tasks, plus long-context processing for realistic documents and RAG workflows (Sec. 1).
- Why it matters
  - Real-world impact: enterprise workflows (legal, financial, engineering) involve long documents, tool APIs, and correctness-sensitive tasks like math and coding. A unified model reduces integration overhead and failure modes that arise from model switching.
  - Theoretical significance: demonstrates that sparse global attention (rather than global attention everywhere) with the right training recipe can maintain strong long-context performance while reducing compute (Fig. 1, Sec. 2.1; Sec. 2.3).
- Prior approaches and gaps
  - Many open-weight models are either “non-reasoning” chat assistants or “reasoning” specialists requiring long chain-of-thought outputs, but not both in one model with clean mode control.
  - Long-context models often rely on global attention in every layer, which is computationally heavy; or use chunking schemes that can hurt performance or are hard to integrate (Sec. 2.1).
  - RL methods for reasoning (e.g., GRPO) struggle with hard samples and handling groups of all-incorrect trajectories; PPO-style clipping can mute learning signals for critical low-probability “fork” tokens (Sec. 2.4.2).
- Positioning
  - EXAONE 4.0 integrates two modes inside one model and pairs it with: (1) a hybrid local–global attention stack; (2) a new RL algorithm `AGAPO`; (3) a two-stage preference alignment that preserves correctness while reducing verbosity (Sec. 2.4 and Fig. 3).
  - The series spans a high-performance mid-size model (32B) and an on-device oriented small model (1.2B)—both released for research (Abstract; Table 1; Appendix B).

## 3. Technical Approach
Step-by-step overview of the system, from architecture through training and post-training.

- Architecture (Sec. 2.1; Fig. 1–2; Table 1)
  - Hybrid attention with a 3:1 ratio of local to global layers (Fig. 1).
    - `Local attention` uses sliding-window attention with a window size of 4K tokens. Sliding-window attention is a sparse attention pattern that restricts each token’s attention to its neighborhood; here it “slides” over the sequence to maintain continuity and is widely supported in open-source frameworks (Sec. 2.1).
    - `Global attention` layers do not use RoPE for QK positional encoding (to reduce length bias and keep a global view; Sec. 2.1).
    - Rationale: Recent results show that a minority of global-attention layers with sufficiently large local windows can achieve strong long-context behavior while reducing compute versus global attention everywhere (Sec. 2.1; citations [14, 15, 36]).
  - Repositioned normalization: `QK-Reorder-LN` (Fig. 2; Sec. 2.1).
    - Applies RMSNorm to queries and keys before attention and again after attention output. This mitigates the variance growth problem associated with Pre-LN architectures at depth and empirically improves downstream task performance (Sec. 2.1; refs [42, 56]).
  - Additional design choices (Table 1):
    - `GQA` heads (Generalized Multi-Query Attention) to reduce KV cache size and improve throughput at long sequence lengths.
    - `SwiGLU` activations, vocabulary size 102,400 with BBPE tokenizer, and long maximum context: 131,072 tokens (32B) and 65,536 (1.2B).

- Pretraining and data curation (Sec. 2.2; Table 2)
  - 32B model: 14 trillion tokens; 1.2B: 12T—roughly doubling EXAONE 3.5 data for enhanced world knowledge and better “cognitive behaviors” for reasoning (Sec. 2.2; refs [12, 13]).
  - Emphasis on domain curation for STEM, coding, and math where robust ground truth and verification are crucial.

- Context-length extension (Sec. 2.3)
  - Two-stage extension: 4K → 32K → 128K (32B); 4K → 64K (1.2B).
  - Continuous validation with the Needle-In-A-Haystack (NIAH) test to ensure the model can retrieve specific “needle” strings anywhere in the long input without short-context degradation (Sec. 2.3; [16]).

- Post-training pipeline (Fig. 3; Sec. 2.4)
  - Large-scale SFT (Sec. 2.4.1)
    - Five domains: World Knowledge, Math/Code/Logic, Long Context, Agentic Tool Use, Multilinguality.
    - Unified-mode SFT: mix `REASONING` and `NON-REASONING` data; tuned ratio so the model does not “overthink” when the `NON-REASONING` mode is active. The paper sets a token ratio of `REASONING:NON-REASONING = 1.5:1` after ablation (Sec. 2.4.1).
    - Targeted resampling: a second pass focusing on high-quality REASONING data for Code and Tool Use to correct domain imbalance.
  - Reasoning RL with `AGAPO` (Sec. 2.4.2)
    - Dataset categories: math, code, science, instruction following; filtering removes trivially easy items (generate 8 responses with the SFT model, drop those where all 8 are correct).
    - Rewarding:
      - Math and science: rule-based correctness (plus an LLM judge if a science answer is marked incorrect by rules).
      - Code: pass/fail on provided test cases for the final code block.
      - Instruction following: binary reward based on constraint satisfaction.
    - The `AGAPO` algorithm (Asymmetric Sampling and Global Advantage Policy Optimization) modifies GRPO/PPO-style training:
      - Remove PPO clip loss to avoid suppressing updates from low-probability “fork tokens” that matter for exploration (Sec. 2.4.2: “Remove Clipped Objective”).
      - Asymmetric sampling retains groups where “all responses are incorrect” and assigns small negative rewards, enabling learning from failure (Sec. 2.4.2: “Asymmetric Sampling”).
      - Two-level advantage: compute a group-level LOO (leave-one-out) advantage, then normalize across the batch to get a `global advantage` that handles all-incorrect groups better (Eq. (2)).
      - Apply a sequence-level cumulative KL penalty to stay faithful to SFT behavior (Sec. 2.4.2).
    - Objective (Eq. (1)): maximize the globally normalized advantage-weighted log-probabilities of complete responses, minus a KL term to the reference model.
  - Preference Learning with a hybrid reward (Sec. 2.4.3)
    - Use SimPER (a reference-free DPO-style method) on on-policy samples produced after RL (4–16 responses per query).
    - Two stages:
      - Stage 1: “Conciseness + verifiable reward” to pick the shortest correct response—improves token efficiency without hurting correctness in REASONING mode.
      - Stage 2: Human-aligned “Preference + language consistency” (partly reusing Stage-1 data for stability). For REASONING data, label preferences on the final answer only, not the thinking steps.

- Agentic tool-use data construction (Sec. 2.4.1)
  - Beyond single API calls, build multi-turn, long-horizon user–agent interactions with execution feedback and iterative planning so the model learns the structure of goal-directed tool use.

- Multilinguality (Sec. 2.4.1)
  - Adds Spanish while keeping the same tokenizer as EXAONE 3.5/Deep; curates Korean data including expert and education-oriented content; evaluates EN–ES translation with WMT24++ using LLM-as-a-judge (Appendix D.5).

## 4. Key Insights and Innovations
- Unifying two response modes in one model (Fig. 3; Sec. 2.4.1)
  - What’s new: A single model reliably toggles between `NON-REASONING` (concise answers) and `REASONING` (long CoT), with explicit data balancing (1.5:1 token ratio) to avoid “over-deliberation” on simple tasks.
  - Why it matters: Reduces deployment friction and UX inconsistency that come from switching models for different tasks.
- Cost-conscious long-context design (Sec. 2.1; Fig. 1; Sec. 2.3)
  - What’s new: A 3:1 local-to-global attention stack with 4K sliding windows, no RoPE on global layers, and progressive extension to 128K.
  - Why it matters: Preserves global understanding with far less global attention, enabling 128K contexts without the full compute burden of global-all-layers models, while maintaining short-context performance.
- `QK-Reorder-LN` normalization placement (Fig. 2; Sec. 2.1)
  - What’s new: RMSNorm applied on Q/K before attention and again after attention output to mitigate depth-related variance issues observed with Pre-LN.
  - Why it matters: Empirically improves downstream tasks for deep stacks despite slightly more compute—an incremental but meaningful stability upgrade.
- `AGAPO` RL for reasoning (Sec. 2.4.2; Eqs. (1–2))
  - What’s new: Removes PPO clipping, keeps hard “all-incorrect” groups via asymmetric sampling, and introduces a two-level (group then global) advantage normalization, plus sequence-level cumulative KL.
  - Why it matters: Strengthens signal for exploration and learning from failure, which is crucial for long-chain reasoning where early low-probability tokens determine entire solution paths. This is a substantive algorithmic advance over GRPO.
- Two-stage preference alignment for concise yet accurate reasoning (Sec. 2.4.3)
  - What’s new: A hybrid reward design that first optimizes for shortest correct answers, then aligns for human preferences and language consistency.
  - Why it matters: Guards against reasoning bloat—a common failure mode after RL—and maintains usability in both modes.

## 5. Experimental Analysis
- Evaluation design (Sec. 3)
  - Benchmarks span six categories (World Knowledge, Math/Coding, Instruction Following, Long Context, Agentic Tool Use, Multilinguality), including up-to-date and “decontaminated” variants when available:
    - World Knowledge: MMLU-REDUX, MMLU-PRO, GPQA-DIAMOND (Sec. 3.1).
    - Math/Coding: AIME 2025, HMMT Feb 2025, LiveCodeBench V5/V6 (Sec. 3.1).
    - Instruction Following: IFEVAL, MULTI-IF(EN) (Sec. 3.1).
    - Long Context: HELMET (excluding LongCite—Appendix D.1), RULER (D.2), LongBench (EN) (D.3); plus an in-house Korean Ko-LongBench (D.4).
    - Tool Use: BFCL-V3 and TAU-BENCH (with `gpt-4.1-2025-04-14` as the user simulator; Sec. 3.1).
    - Multilingual: Korean (KMMLU-PRO, KMMLU-REDUX, KSM, KO-LONGBENCH) and Spanish (MMMLU-ES, MATH500-ES, WMT24++ EN↔ES with judge prompt shown in D.5).
  - Setup details:
    - Multi-sample evaluation for low-shot tasks to reduce variance: e.g., GPQA-DIAMOND n=8; AIME/HMMT n=32; LiveCodeBench, TAU-BENCH, MATH500 n=4 (Sec. 3.3).
    - REASONING mode decoding: temperature 0.6, top-p 0.95; NON-REASONING uses greedy for n=1 (Sec. 3.3).
    - Reasoning budget study varies max “thinking tokens” from 1K to 64K, and if the budget is hit, the model is forced to output the answer (Table 7).

- Main quantitative results (highlights; see Tables 3–6)
  - Math/Coding leadership at mid-size:
    - “32B REASONING” surpasses much larger frontier models on several math/coding tasks:
      - AIME 2025: 
        > Table 3: EXAONE 4.0 32B (REASONING) = 85.3 vs Qwen3 235B (REASONING) = 81.5; DeepSeek R1-0528 = 87.5.
      - HMMT Feb 2025:
        > Table 3: 72.9 vs Qwen3 235B (REASONING) = 62.5; DeepSeek R1-0528 = 79.4.
      - LiveCodeBench V6:
        > Table 3: 66.7 vs Qwen3 235B (REASONING) = 58.9; DeepSeek R1-0528 = 70.3.
    - “1.2B REASONING” is strong for its size:
      > Table 5: AIME 45.2; HMMT 34.0; LiveCodeBench V6 45.3—competitive or better than larger small-size baselines like SmolLM 3B.
  - World knowledge remains competitive:
    - “32B REASONING”:
      > Table 3: MMLU-REDUX 92.3, MMLU-PRO 81.8, GPQA-DIAMOND 75.4 (close to frontier models; DeepSeek R1 = 81.0).
  - Instruction following:
    - “32B NON-REASONING”:
      > Table 4: IFEVAL 84.8 and MULTI-IF(EN) 71.6—on par with strong baselines (e.g., Qwen3 235B MULTI-IF 72.5).
  - Long-context:
    - “32B NON-REASONING”:
      > Table 4: RULER 88.2 (Qwen3 235B = 90.6), HELMET 58.3 (Gemma3 27B = 58.3; Qwen3 32B = 54.5 official), LongBench 48.1 (Gemma3 27B = 51.5).
    - HELMET task-wise and RULER length-wise breakdowns are detailed in Appendices D.1–D.2 (Tables 9–10; Fig. 4).
  - Agentic tool use:
    - “32B REASONING”:
      > Table 3: TAU-BENCH Airline 51.5 (DeepSeek R1-0528 = 53.5); Retail 62.8 (Qwen3 235B = 63.9).
    - “32B NON-REASONING”:
      > Table 4: BFCL-V3 65.2 (Qwen3 235B = 68.0).
  - Multilingual:
    - Korean (“32B NON-REASONING”):
      > Table 4: KMMLU-REDUX 64.8; KO-LONGBENCH 76.9—strong even versus larger baselines.
    - Spanish (“32B NON-REASONING”):
      > Table 4: MMMLU-ES 80.6; MATH500-ES 87.3; WMT24++ ES 90.7.

- Does the evidence support the claims?
  - Yes for the core claims:
    - Unified-mode effectiveness: competitive instruction following (IFEVAL/MULTI-IF) coexists with strong reasoning (AIME/HMMT/LiveCodeBench) in a single model (Tables 3–4).
    - Long-context viability: solid RULER/HELMET/LongBench scores without global attention in all layers (Tables 9–11).
    - RL advantages: math/coding gains versus same-class and some frontier models suggest the RL and preference pipeline effectively enhances reasoning while controlling verbosity (Tables 3, 5, 7).
  - Caveats:
    - Some baseline numbers are from external reports (marked by asterisks), so environment parity is imperfect (Table notes in 3–6).
    - TAU-BENCH uses an LLM simulator (gpt-4.1) for the user role, which introduces judge/model dependencies (Sec. 3.1).

- Ablations, robustness, and failure modes
  - Mode balance: the paper reports that too much REASONING data leads the model to “think long” even in NON-REASONING mode; they set 1.5:1 after ablation (Sec. 2.4.1), though quantitative ablation tables are not shown.
  - Reasoning budget: reducing the “thinking tokens” from 64K to 32K yields small drops except AIME for the 32B model (−12.3 pt) (Table 7). This indicates performance is robust to shorter thinking in most settings, but some math problems benefit from longer deliberation.
  - Long-context validation: NIAH tests at each extension stage ensure robust retrieval of deeply buried information (Sec. 2.3).

## 6. Limitations and Trade-offs
- Architectural and training assumptions
  - Hybrid attention assumes that periodic global layers plus 4K local windows suffice for long-range reasoning (Sec. 2.1). Some tasks with intricate cross-document interactions may still favor denser global patterns.
  - The QK-Reorder-LN placement improves stability “despite more computation” (Sec. 2.1; Fig. 2), implying a compute trade-off.
- Data and evaluation dependencies
  - Heavy pretraining (14T tokens; Table 2) is a significant resource requirement; reproducing this approach at scale may be out of reach for many.
  - Some benchmarks rely on LLM-based judging (e.g., science in RL verification when rules fail; WMT24++ judge in D.5), which can bias results toward the judge’s preferences.
- Mode integration risks
  - The paper notes that too much REASONING data during SFT makes the model overuse long CoT even when not desired (Sec. 2.4.1). The chosen 1.5:1 ratio is a practical fix, but the optimal ratio may vary by domain and future models.
- Tool-use maturity
  - Tool-use performance is “competitive” rather than SOTA (Tables 3–4). Long-horizon planning and recovery from API errors remain open challenges in the broader literature and are not specifically dissected here.
- License constraints
  - The released weights are for research and education only; commercial use is prohibited without a separate license (Appendix B: “EXAONE AI Model License Agreement 1.2 - NC”). This limits immediate commercial adoption.

## 7. Implications and Future Directions
- How this changes the landscape
  - A credible blueprint for unified-mode LLMs: EXAONE 4.0 shows you can pack both “fast utility” and “deep reasoning” into one model without tanking either—if you engineer the data mix, RL objective, and preference pipeline carefully (Sec. 2.4; Fig. 3).
  - Cost-aware long context at scale: Demonstrates that a minority of global-attention layers, if combined with sufficiently large local windows and progressive length extension, can reach 128K tokens with strong performance (Fig. 1; Sec. 2.3; RULER/HELMET).
  - RL for reasoning that learns from failure: `AGAPO`’s asymmetric sampling and global advantage are promising for any domain where “all incorrect” rollouts still contain learning signal (Sec. 2.4.2).

- Follow-up research
  - Systematic ablations:
    - Quantify effects of local window size and global-layer placement across depths and tasks.
    - Compare `AGAPO` against GRPO/DAPO variants under controlled conditions to isolate which components (clip removal, asymmetric sampling, global advantage) drive gains.
  - Adaptive mode control:
    - Explore automatic switching between `NON-REASONING` and `REASONING` based on problem hardness detection, with cost-aware stopping rules (building on the “reasoning budget” study in Table 7).
  - Robust long-horizon tool use:
    - Add datasets with noisy tool feedback, partial failures, and recovery; measure planning reliability and correction loops, not just single-call accuracy.
  - Multilingual expansion:
    - Extend beyond Spanish while maintaining English/Korean parity under the fixed tokenizer; investigate how far the shared vocabulary stretches before performance regresses.

- Practical applications
  - Enterprise document QA and RAG over long contexts (legal, finance, compliance): 128K contexts and strong retrieval recall (RULER/HELMET) are directly useful (Sec. 3.1; D.1–D.2).
  - Engineering and data science copilots: math/coding strength (AIME/HMMT/LiveCodeBench) suggests competitive assistance for algorithmic problem solving and code generation.
  - Agentic automation: competitive BFCL/TAU-BENCH scores imply readiness for function-calling assistants that orchestrate APIs in multi-step workflows.

> In short, EXAONE 4.0’s main contribution is a practical recipe—spanning hybrid attention, unified-mode SFT, `AGAPO` RL, and two-stage preference learning—that yields a single, versatile model with strong math/coding reasoning, competitive tool use, and long-context handling (Figs. 1–3; Tables 1–7).
