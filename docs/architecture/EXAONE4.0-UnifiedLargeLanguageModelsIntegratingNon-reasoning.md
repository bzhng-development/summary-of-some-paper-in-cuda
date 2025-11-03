# EXAONE 4.0: Unified Large Language Models Integrating Non-reasoning and Reasoning Modes

**ArXiv:** [2507.11407](https://arxiv.org/abs/2507.11407)

## 🎯 Pitch

EXAONE 4.0 pioneers a unified large language model family that seamlessly blends both rapid, direct response generation ('NON-REASONING' mode) and sophisticated, step-by-step reasoning ('REASONING' mode) within a single system, while supporting 128K-token contexts via an efficient hybrid attention mechanism. This innovation eliminates the need for separate models for usability and reasoning, enabling dynamic adaptation for diverse real-world and agentic AI scenarios— from concise Q&A to complex mathematical, coding, and long-context tasks— all with robust multilingual and advanced tool-use support. By merging performance and flexibility, EXAONE 4.0 sets a new standard for practical, scalable, and instructable AI agents.

---

## 1. Executive Summary
EXAONE 4.0 introduces a single family of large language models that natively supports two operating styles—`NON-REASONING` (fast, concise answers) and `REASONING` (deliberate, chain-of-thought style)—within one model, while scaling context to 128K tokens via a hybrid local–global attention design (Figure 1, §2.1, §2.3). It further proposes a new reinforcement learning algorithm (`AGAPO`) and a two-stage preference alignment scheme to boost verifiable reasoning without degrading instruction following (Figure 3, §2.4), and demonstrates strong math/coding and competitive tool-use performance for both a mid-size 32B and an on-device 1.2B model (Tables 3–6).

## 2. Context and Motivation
- Problem addressed
  - Real users want both quick, direct responses and deep, step-by-step reasoning, but most systems optimize for one style at a time. Earlier EXAONE releases split these capabilities: EXAONE 3.5 emphasized “real-world usability” (instruction following) while EXAONE Deep emphasized reasoning (§1).
  - Long-context use (summarization, RAG, document QA) is increasingly common, but full global attention at 100K+ tokens is computationally costly (§1, §2.1). 
  - Agentic workflows need robust tool calling over multi-step, multi-turn tasks (§1, §2.4.1).
  - Multilingual support beyond English–Korean is desirable without hurting existing languages (§1).

- Why it matters
  - Consolidating the two response styles reduces operational complexity (one model to deploy, not two), and enables applications to dynamically choose speed vs accuracy.
  - Efficient long-context handling directly affects enterprise and government scenarios that require processing contracts, legal filings, technical manuals, or multi-document evidence (§1, §2.3, Appendix D).
  - Tool use underpins modern agent systems (e.g., data retrieval, code execution, transactions), so stable function calling and multi-step planning are foundational for “agentic AI” (§1, §3.1).

- Shortcomings of prior approaches
  - Long-context: models that use global attention in every layer are accurate but expensive; chunked attention can be easier to implement but may lose cross-chunk information. EXAONE 4.0 replaces “global everywhere” with a hybrid approach (3 local : 1 global) and uses sliding-window local attention to keep both stability and efficiency (§2.1; Figure 1).
  - Reasoning RL: popular GRPO-style training relies on clipped PPO objectives, ignores all-incorrect groups, and normalizes advantages only within groups—each of which can weaken learning signals for complex reasoning (§2.4.2).

- Positioning
  - EXAONE 4.0 unifies modes, scales context to 128K with hybrid attention (Figure 1), adjusts normalization to mitigate depth-related variance (Figure 2), and introduces AGAPO to strengthen verifiable reasoning signals (Eq. 1–2, §2.4.2). It also broadens multilingual coverage to Spanish while keeping the same tokenizer/vocab as earlier EXAONE models (Table 1, §1).

## 3. Technical Approach
Step-by-step overview of the system design and training pipeline.

- Model configurations (§2.1; Table 1)
  - Two sizes: `32B` (64 layers, d_model=5120) and `1.2B` (30 layers, d_model=2048).
  - `GQA` (Grouped Query Attention), `SwiGLU` feed-forward, `RMSNorm` throughout.
  - Max context: 131,072 (32B) and 65,536 (1.2B); shared BBPE tokenizer, 102,400 tokens.

- Hybrid attention with sliding windows (Figure 1, §2.1)
  - Local attention uses a `sliding window` over recent tokens; EXAONE sets a 4K-token window to protect short-context quality.
  - Global attention is applied periodically (ratio `Local:Global = 3:1`) to preserve the ability to integrate information across the whole sequence while reducing compute compared to full-global.
  - Design choices:
    - Avoids Rotary Position Embeddings (`RoPE`) for the global attention path to reduce length bias and preserve a “global view” (§2.1).
    - Prefers sliding-window attention (well-supported, theoretically stable) over chunked attention (§2.1).

- Normalization change: `QK-Reorder-LN` (Figure 2, §2.1)
  - Motivation: Pre-LN Transformers can accumulate output variance with depth, causing “dead” layers in deep stacks (§2.1).
  - Mechanism: apply `RMSNorm` to the `Query` and `Key` inputs before attention scoring and again after the attention output—shown to improve downstream performance despite extra compute (§2.1, Figure 2).

- Context-length extension to 128K (§2.3)
  - Two-stage procedure: pretrain at 4K, extend to 32K, then to 128K, validating each step with the Needle-In-A-Haystack (NIAH) test until “green light” across segments is achieved (§2.3).
  - For the 1.2B model: extended to 64K.

- Pretraining data scale and curation (§2.2; Table 2)
  - `32B`: 14 trillion tokens (≈2× EXAONE 3.5’s 6.5T); `1.2B`: 12T tokens. Aim: expand world knowledge and expose reasoning-relevant “cognitive behaviors” via curated STEM and similar content.
  - Compute: 2.69×10^24 FLOPs (32B), 8.65×10^22 FLOPs (1.2B) (Table 2).

- Supervised fine-tuning (large-scale SFT) and unified mode training (§2.4.1; Figure 3)
  - SFT data split into `NON-REASONING` vs `REASONING`, and across five domains: World Knowledge, Math/Code/Logic, Long Context, Agentic Tool Use, Multilinguality.
  - Long-context SFT varies both the length and the position of key information to train retrieval over dispersed evidence (§2.4.1).
  - Agentic tool-use data emphasizes multi-step, multi-turn interactions with environment feedback—not just single function calls (§2.4.1).
  - Unified training mixes both modes; token ratio `REASONING:NON-REASONING = 1.5:1`. Higher reasoning ratios made the model “act reasoning” even when not requested, so this ratio balances modes (§2.4.1).
  - After unification, a second pass reuses high-quality `REASONING` data in Code and Tool Use to correct domain imbalance (§2.4.1).

- Reasoning RL with `AGAPO` (§2.4.2; Eq. 1–2; Figure 3)
  - Task domains: math, code, science, instruction following. Filtering removes “too easy” items (all 8 SFT samples correct) to focus on informative cases (§2.4.2).
  - Rewards:
    - Math: rule-based verifier of final answers.
    - Code: final code block must pass tests.
    - Science: rule-based verifier first; if incorrect, an LLM-judge checks flexibly.
    - Instruction-following: 1 if all constraints met, else 0 (§2.4.2).
  - Core innovations relative to GRPO:
    - Remove PPO-style clipping: enables low-probability, exploratory tokens (often critical in branching reasoning) to influence gradients (§2.4.2 “Remove Clipped Objective”).
    - Asymmetric sampling: keep groups where all responses are incorrect and assign small negative rewards (“negative reinforcement”) to push away from bad reasoning paths (§2.4.2 “Asymmetric Sampling”).
    - Group & Global advantages: compute leave-one-out (LOO) advantage per group, then normalize across the entire mini-batch to calibrate rewards for all-incorrect groups (§2.4.2 “Group&Global Advantages”).
    - Sequence-level cumulative KL: regularize toward the SFT policy at the sequence level to preserve prior capabilities (§2.4.2).
  - Objective (Eq. 1–2): maximize a sum of log-likelihoods weighted by global advantages minus a KL penalty to a reference policy.

- Two-stage preference learning with hybrid rewards (§2.4.3; Figure 3)
  - Framework: `SimPER` (preference optimization without a fixed reference). Dataset is on-policy: generate 4–16 responses per prompt from the RL model, then score with a hybrid reward that mixes verifiable correctness, preference, language consistency, and conciseness (§2.4.3).
  - Stage 1: promote token efficiency by preferring the shortest correct solution—chosen = shortest among correct; rejected = longer or incorrect. Keeps `REASONING` quality while curbing verbosity (§2.4.3).
  - Stage 2: focus on human alignment—use preference and language consistency rewards; only the final answer (not the intermediate thoughts) is preference-labeled for reasoning data. Sample some Stage-1 data to stabilize training (§2.4.3).

- Multilingual support and tokenizer reuse (§1, Table 1, §2.4.1)
  - Adds Spanish while preserving English/Korean performance; uses the same tokenizer and vocabulary to avoid regressions (§1, Table 1; §2.4.1 Multilinguality).

## 4. Key Insights and Innovations
- Unified dual-mode model (fundamental)
  - What’s new: both `NON-REASONING` and `REASONING` modes in a single model, trained jointly with a carefully chosen token ratio (1.5:1) and then harmonized via two-stage preference learning (Figure 3; §2.4.1–§2.4.3).
  - Why it matters: reduces deployment complexity and enables adaptive use (fast answers when possible, deeper reasoning when needed) without swapping models.

- Hybrid long-context attention (fundamental)
  - What’s new: a 3:1 sliding-window-to-global attention schedule with a 4K local window, skipping RoPE on the global path to minimize length bias (Figure 1; §2.1).
  - Why it matters: brings 128K context within reach while controlling compute; Appendix D shows competitive results on RULER and HELMET at long lengths.

- `QK-Reorder-LN` normalization (incremental but impactful)
  - What’s new: RMSNorm applied to queries/keys before attention and again after attention output to counter variance growth in deep Pre-LN stacks (Figure 2; §2.1).
  - Why it matters: improves downstream stability and quality, especially for deep models (64 layers in the 32B).

- `AGAPO` reasoning RL (fundamental)
  - What’s new: removal of PPO clipping, inclusion of all-incorrect groups via asymmetric sampling, two-level advantage estimation (group LOO → global normalization), and sequence-level cumulative KL (Eq. 1–2; §2.4.2).
  - Why it matters: strengthens gradient signals for exploratory reasoning tokens and harder problems, addressing known limitations of GRPO-style training.

- Long-context SFT design and on-policy preference selection (incremental)
  - What’s new: SFT that varies both where and how key info appears across long inputs; preference data selected from the model’s own outputs using hybrid rewards (§2.4.1, §2.4.3).
  - Why it matters: better prepares the model for dispersed-evidence reasoning and aligns generation quality/length trade-offs with task needs.

## 5. Experimental Analysis
- Evaluation methodology (§3.1–§3.3)
  - Coverage: World Knowledge (MMLU-Redux, MMLU-Pro, GPQA-Diamond), Math/Coding (AIME 2025, HMMT Feb 2025, LiveCodeBench V5/V6), Instruction Following (IFEVAL, Multi-IF EN), Long Context (HELMET, RULER, LongBench), Tool Use (BFCL-V3, TAU-Bench), and Multilinguality (Korean: KMMLU-Pro/Redux, KSM, KO-LongBench; Spanish: MMMLU ES, MATH500 ES, WMT24++).
  - Baselines include mid-size (e.g., Qwen3-32B) and frontier (>200B, e.g., Qwen3-235B, DeepSeek R1-0528) (§3.2; Table 8).
  - Decoding and sampling: in REASONING, temperature 0.6, top-p 0.95; presence penalty 1.5 for `32B` only; n-samples vary per benchmark (e.g., n=32 for AIME/HMMT), and accuracy averaged across samples (§3.3).
  - Long-context for small baselines (Qwen3 0.6B/1.7B) extended via `YaRN` to 64K for fair comparison (Appendix D).

- Main quantitative results (selected highlights; Tables 3–6)
  - Math/Coding (strength of EXAONE 4.0):
    - `32B (REASONING)`: 
      > Table 3: AIME 2025 = 85.3; HMMT Feb 2025 = 72.9; LiveCodeBench V6 = 66.7.
      These exceed Qwen3-235B (AIME 81.5, HMMT 62.5, LCB-V6 58.9) and approach DeepSeek R1 on some tasks.
    - `1.2B (REASONING)`:
      > Table 5: AIME 2025 = 45.2; HMMT = 34.0; LCB-V6 = 45.3.
      Strong for its size; competitive with or better than 1.7B–3B baselines.
  - World Knowledge:
    - `32B (REASONING)`:
      > Table 3: MMLU-Redux = 92.3; MMLU-Pro = 81.8; GPQA-Diamond = 75.4.
      Near frontier scores on MMLU(-Pro); GPQA trails DeepSeek R1 (81.0) but leads many mid-size baselines.
  - Instruction Following:
    - `32B (REASONING)`: 
      > Table 3: IFEVAL = 83.7; MULTI-IF (EN) = 73.5.
    - `32B (NON-REASONING)`:
      > Table 4: IFEVAL = 84.8; MULTI-IF (EN) = 71.6.
      High alignment while retaining reasoning strength.
  - Long Context (mid-size; Tables 4 and Appendix D):
    - `32B (NON-REASONING)`:
      > Table 4: HELMET = 58.3; RULER = 88.2; LongBench V1 = 48.1 at 128K.
      Appendix D.2 shows RULER 88.18 at 128K (Table 10), close to Qwen3-235B’s 90.60.
  - Tool Use:
    - `32B (REASONING)`:
      > Table 3: BFCL-V3 = 63.9; TAU-Bench Airline = 51.5; Retail = 62.8.
      TAU-Bench Airline approaches DeepSeek R1 (53.5), notable given parameter gap; Retail is close to R1 (63.9).
    - `1.2B (REASONING)`:
      > Table 5: TAU-Bench Retail = 28.1—highest among compared small models.
  - Multilingual:
    - Korean `32B (REASONING)`:
      > Table 3: KMMLU-Pro (KO) = 67.7; KSM = 87.6.
    - Spanish `32B (REASONING)`:
      > Table 3: MMMLU (ES) = 85.6; MATH500 (ES) = 95.8.
    - Translation (WMT24++ EN↔ES judged by gpt-4.1):
      > Table 4: 90.7 for `32B (NON-REASONING)`. Judge prompt shown in Appendix D.5.

- Reasoning budget study (Table 7, §3.5)
  - Reducing the allowed “thinking tokens” degrades performance, but moderately for many cases:
    > Table 7: `32K` vs `64K` budget—AIME 2025 drops from 85.3 → 74.8 (−12.3 points) for 32B; LiveCodeBench V6 remains stable (66.7 → 67.3).
  - This quantifies the compute–quality trade-off and suggests 32K budgeting is often acceptable outside the hardest math.

- Robustness/ablations
  - The paper reports the reasoning-budget ablation (Table 7). It mentions ablations guiding the 1.5:1 mode ratio choice (§2.4.1) but does not present numbers. No ablation quantifies the impact of `QK-Reorder-LN` or the 3:1 attention ratio.

- Overall assessment
  - The evidence strongly supports claims of leading math/coding accuracy at the mid-size scale (Table 3) and competitive long-context capability (Appendix D). Tool-use performance is respectable relative to much larger models. The lack of detailed ablations for architectural and RL choices limits causal attributions for some gains.

## 6. Limitations and Trade-offs
- Assumptions and scope
  - Verifiable rewards are central to RL and preference selection (§2.4.2–§2.4.3). Tasks without clear verifiers (e.g., creative writing, open-domain reasoning without ground truth) may not benefit as much.
  - Preference learning labels only the final answer in reasoning mode (§2.4.3). This may under-optimize intermediate reasoning quality/style.

- Compute and data
  - Pretraining scale is very high (14T tokens; 2.69×10^24 FLOPs for 32B; Table 2). While the hybrid attention reduces inference cost at long lengths, training such models remains expensive.
  - Reasoning often uses large token budgets (up to 64K generated tokens in math/coding benchmarks, §3.3), which can be costly in production.

- Long-context trade-offs
  - While RULER/HELMET scores are competitive, results are not top across all tasks (Appendix D). Summarization and some retrieval subtasks show room for improvement (e.g., HELMET Summarization sub-scores around mid-20s for 32B; Table 9).

- Evaluation gaps
  - No reported ablation measuring the independent contributions of: hybrid attention vs full-global; `QK-Reorder-LN` vs Pre-LN; AGAPO vs GRPO; or the exact effect of Stage-1 conciseness preference.
  - Tool-use evaluation covers two popular suites (BFCL-V3, TAU-Bench), but broader real-world agent tasks (with unreliable tools, noisy environments) are not assessed.

- Licensing and deployment
  - The public weights are released under a non-commercial license (Appendix B), which may limit direct commercial adoption even if the technical capabilities are attractive.

## 7. Implications and Future Directions
- How this changes the landscape
  - Unified-mode models simplify deployment and enable adaptive reasoning depth without model swaps. The hybrid attention design and demonstrated 128K handling show a practical path to long-context at mid-size scales.
  - The `AGAPO` principles—especially asymmetric sampling with global-normalized advantages—offer a blueprint for stronger verifiable-reasoning RL.

- Follow-up research
  - Controlled ablations:
    - Vary the `Local:Global` ratio and window size; compare sliding-window to chunked attention under identical data/compute.
    - Isolate gains from `QK-Reorder-LN`.
    - Head-to-head AGAPO vs GRPO on the same seeds and data, including sensitivity to negative-reward magnitudes.
  - Dynamic compute:
    - Learn to adapt the reasoning budget per query (building on §3.5) and stop early when confidence is high.
  - Richer verifiers:
    - Expand verifiable rewards to domains like scientific QA with symbolic checkers, or to multi-step tool chains with environment simulators.
  - Multilingual scaling:
    - Extend beyond Spanish while preserving English/Korean quality using shared tokenizers; measure cross-lingual transfer in math/code.
  - Agentic robustness:
    - Evaluate on noisy tool outputs, partial failures, and long-horizon planning benchmarks; incorporate self-correction loops.

- Practical applications
  - Document-heavy workflows: legal/finance/technical QA and summarization at 128K context (HELMET/LongBench tasks; §3.1, Appendix D).
  - Code assistants: strong LiveCodeBench scores (Tables 3 and 5) suggest reliable code synthesis with tests.
  - Education and STEM support: near-frontier math reasoning at the mid-size scale (AIME/HMMT in Table 3) and multilingual math (MATH500 ES in Tables 3 and 5).
  - Agents: competitive function calling and simulated customer support (BFCL-V3, TAU-Bench; Tables 3–6) provide a foundation for production-grade tool-using assistants.

Overall, EXAONE 4.0 offers a cohesive architecture–training recipe that unifies rapid and deliberate behaviors, scales context efficiently, and advances RL for verifiable reasoning. While some design choices would benefit from deeper ablation, the reported results—especially in math/coding and long-context handling—indicate a strong, practically deployable step toward agentic, multilingual LLMs.
