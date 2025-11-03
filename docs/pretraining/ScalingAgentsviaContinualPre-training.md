# Scaling Agents via Continual Pre-training

**ArXiv:** [2509.13310](https://arxiv.org/abs/2509.13310)

## 🎯 Pitch

This paper introduces Agentic Continual Pre-training (Agentic CPT), a novel intermediate training stage designed to endow large language models with intrinsic agentic behaviors—such as robust tool use, multi-step reasoning, and adaptive decision-making—before post-training. By synthesizing massive, diverse agentic data (First-order and High-order Action Synthesis) and applying this method to create AgentFounder-30B, the authors demonstrate substantial new state-of-the-art performance on deep research and tool-use benchmarks, surpassing both open-source and commercial competitors. This work redefines how we build agentic AI, closing the performance gap for open-source systems and laying the foundation for more capable and reliable autonomous research assistants.

---

## 1. Executive Summary
This paper introduces Agentic Continual Pre‑training (Agentic CPT), an additional training stage inserted between standard pre‑training and post‑training to build “agentic foundation models” that already possess tool‑use and long‑horizon decision‑making priors. Using this stage plus two scalable data‑synthesis methods—First‑order Action Synthesis (FAS) and High‑order Action Synthesis (HAS)—the paper trains AgentFounder‑30B, which achieves new state‑of‑the‑art results on multiple deep‑research benchmarks while retaining broad tool‑use skills (e.g., 39.9% on BrowseComp‑en and 31.5% Pass@1 on HLE; Tables 1–2).

## 2. Context and Motivation
- Problem addressed:
  - Post‑training only (SFT/RL) on general‑purpose LLMs underperforms on complex “agentic” tasks like web research and multi‑step tool use (Section 1). Agentic tasks require models to plan, adapt, and decide under evolving external feedback; general LLMs lack such “agentic inductive biases.”
  - The paper reframes alignment for agents as agentic alignment: matching expert‑like behaviors over entire decision chains (reasoning steps, tool calls, recovery from tool failures), not just final answers (Section 1).
- Why it matters:
  - Practical: High‑quality “deep research” assistants must search, browse, compute, and synthesize over long horizons. Commercial systems show strong performance, but open‑source agent models lag (Section 1; performance gaps on BrowseComp in Table 1).
  - Scientific: Post‑training struggles because it must simultaneously (i) teach entirely new behaviors and (ii) align to demonstrations, creating “optimization tension” and limited exploration of the huge policy space (Section 1).
- Shortcomings of prior approaches:
  - SFT/RL on general models depends on limited, trajectory‑level supervision, often locking models into imitating specific paths (Section 1).
  - Full trajectory generation with live tools is expensive and slow; open‑source implementations rarely scale tool‑grounded data (Section 2.2.2, “Scalability Challenges”).
- Positioning:
  - The paper inserts an intermediate CPT stage focused on agentic behavior formation itself (Figure 2). Rather than relying solely on post‑training, it builds an agent‑ready base model and then applies SFT/RL. It also proposes tool‑free, offline data synthesis for scale (Sections 2.2–2.3).

## 3. Technical Approach
The approach modifies the standard pipeline and introduces scalable synthetic data and a progressive training strategy.

- Pipeline redesign (Figure 2; Section 2.1)
  - Standard pre‑training: next‑token prediction on broad corpora (Equation (1)).
    - Equation (1) uses cross‑entropy over the next token P(xt+1 | x1…xt).
  - New Agentic CPT in two stages:
    - Stage 1: ~200B tokens, context 32K; absorbs broad planning patterns and multi‑step reasoning cues without requiring ground‑truth trajectories (Section 2.1).
    - Stage 2: ~100B tokens, context 128K; trains on longer, higher‑quality agentic sequences to improve long‑horizon planning (Section 2.1).
  - Post‑training: supervised fine‑tuning (SFT) with a mixture of general instruction data and agent trajectories. Three variants (SFT‑A/B/C) are tested (Section 3.1.1).

- First‑order Action Synthesis (FAS): zero external supervision or tool calls (Section 2.2)
  - Goal: Create abundant, diverse contexts that induce planning and reasoning, cheaply and at scale.
  - Step 1 — Knowledge‑to‑Question transformation (Section 2.2.1; Figure 3):
    - Build an “entity‑anchored open‑world memory”: map entities to dense, time‑stamped declarative statements harvested from evolving sources (web, search logs, Wikipedia). This is not a rigid knowledge graph; it emphasizes statement density and recency.
    - Sample clusters of entities + statements to synthesize multi‑style questions (fact retrieval, numerical, multi‑hop, synthesis). This converts static text into dynamic problem contexts requiring retrieval and reasoning.
    - Example in Section 2.2.1: from statements about Paris (Louvre 8.7M visitors in 2024; 2023 bedbug crisis; 2025 Paris Air Show orders), generate an obliquely phrased question whose answer is “Riyadh Air.”
  - Step 2 — Planning Action Synthesis (Section 2.2.2; Figure 4-left):
    - For each question, generate diverse “first‑step” analyses and predicted actions (e.g., which tool to call next) but do not execute tools. Instead of repeating the same question K times, diversify at the question level: produce K different but related questions from the same underlying memory, then synthesize a first step for each. This reduces repetition and broadens action‑space coverage.
    - Quality control via “LLM‑as‑judge”: reject samples whose reasoning/actions are unlikely to reach the needed knowledge. Appendix B.1 shows filtering removes 43.5% and raises retained accuracy to 82% (Figure 9).
  - Step 3 — Reasoning Action Synthesis (Section 2.2.3):
    - Two‑step chain‑of‑thought without tools:
      1) Decompose the question into sub‑questions and produce a speculative answer A1 using the model’s internal knowledge.
      2) Provide the mapped requisite knowledge from the memory and ask the model to refine A1 into A2, correcting logic and citing clues.
    - Accept only samples where A2 yields the correct final answer (checked by LLM‑as‑judge). This creates high‑quality, logic‑guided chains for later CPT.

- High‑order Action Synthesis (HAS): reuse real trajectories and create step‑wise decision spaces (Section 2.3; Figure 5)
  - Motivation: Post‑training produces many discarded or single‑use trajectories because overall success labels are coarse and step‑level rewards are uncertain. The insight is to turn each step into a local decision with alternatives.
  - Step‑level scaling (Section 2.3, (1)):
    - For each real trajectory T = {(S1,R1)…(SK,RK)} and step k with context Ck = (Q,S1,R1,…,Sk−1,Rk−1), generate N alternative “thought+invocation” candidates A_k = {S_k^(1)…S_k^(N)} without executing tools. Keep the original S_k, shuffle the N+1 options, and remember the original option’s index n_k.
  - Contrastive decision‑action synthesis (Section 2.3, (2); Figure 5):
    - For each step, explicitly write: “I will choose option n_k,” then append the real environment response R_k that followed the original action. At the end, append an overall binary outcome text (“My decision is Correct/Incorrect”) using the original trajectory’s success J∈{0,1}.
    - This lets the model learn to choose among alternatives under realistic context and feedback without noisy step‑level rewards or live tool costs. It repurposes sub‑optimal trajectories into rich, stable training signals.

- Progressive two‑stage CPT strategy (Section 2.1; Section 3.4.1)
  - Stage 1 (32K): absorb massive FAS and short HAS sequences—cheap to generate/learn.
  - Stage 2 (128K): focus on longer, carefully curated HAS to teach long‑horizon planning. Table 4 shows Stage 1+2 beats Stage 1 alone (+4.1 Pass@1 on BrowseComp‑en).

- Implementation at inference time (Appendix A.1)
  - Tools provided to agents: `Search` (Google top‑10 results), `Visit` (page fetch + goal‑conditioned summarization), `Google Scholar`, `Python Interpreter`, and `File Parser`.

Glossary of paper‑specific terms:
- `Agentic alignment` (Section 1): consistency with expert‑like behaviors across multi‑step reasoning and tool interactions in dynamic environments.
- `Deep research agent`: an LLM‑based system that autonomously orchestrates search, browsing, computation, and synthesis to answer complex, knowledge‑intensive tasks (Section 1).
- `Pass@n`: proportion of questions solved when sampling up to n candidate runs per question (used across Tables and Figures).
- `FAS`/`HAS`: the two proposed synthetic data pipelines described above.
- `MoE` (mixture‑of‑experts): a model architecture that routes tokens to subsets of expert sub‑networks; Appendix B.4 analyzes expert activations.

## 4. Key Insights and Innovations
- Adding a dedicated agentic CPT stage (Figure 2; Section 2.1)
  - What’s new: inserts an intermediate scaling phase that pre‑aligns the base model with agent behaviors before SFT/RL.
  - Why it matters: reduces the “dual‑burden” during post‑training (learning capabilities and alignment at once). Evidence: during identical SFT, CPT models converge to lower loss—baseline 0.8656 vs. 0.7953 for the best CPT model (Figure 7).
- Tool‑free, offline, large‑scale action data (FAS) (Sections 2.2.1–2.2.3; Figure 3–4)
  - What’s new: transforms static web knowledge into diverse question contexts and synthesizes planning and reasoning actions without paying API costs for live tool calls.
  - Why it matters: enables hundreds of billions of tokens for CPT that are grounded in realistic tasks. Filtering methodology in Appendix B.1 demonstrates quality control (accuracy among retained samples: 82%; Figure 9).
- Turning trajectories into decision problems (HAS) (Section 2.3; Figure 5)
  - What’s new: expands each step with multiple plausible actions and creates a “contrastive decision” record that pairs the chosen option with the actual next state and final outcome.
  - Why it matters: converts discarded or sub‑optimal trajectories into stable, abundant supervision that teaches decision‑making, not just path imitation.
- Long‑context, two‑stage CPT strategy (Section 3.4.1; Table 4)
  - What’s new: Stage 1 learns broad patterns cheaply; Stage 2 teaches long‑horizon dependencies with 128K contexts.
  - Why it matters: consistently improves Pass@1/Pass@3 (e.g., +2.9 on GAIA Pass@1, Table 4), indicating better handling of extended workflows.
- Scaling laws for agentic capabilities (Figure 6)
  - Model scale: Average accuracy rises from 20.4% (1B) → 32.7% (4B) → 48.9% (30B‑A3B), surpassing larger baseline systems (DeepSeek‑V3.1 at 43.0% and Kimi‑K2 at 29.6%; Figure 6a).
  - Data scale: Average Pass@3 improves from 54.2% to 62.2% as CPT tokens scale 0B→315B with logarithmic returns; Stage‑2 long‑context continues to add gains at 65B and 315B (Figure 6b).

## 5. Experimental Analysis
- Evaluation setup (Section 3.1)
  - Benchmarks:
    - General web search: BrowseComp‑en/zh, GAIA (text‑only subset of 103 problems), Xbench‑DeepSearch, WebWalkerQA (Section 3.1.3).
    - Scenario‑targeted: DeepResearch Bench (RACE Overall), SEAL‑0 (misleading/conflicting results), Frames (multi‑perspective reasoning), HLE (expert‑level QA), AcademicBrowse (literature navigation) (Section 3.1.3).
  - Baselines: general LLMs with tools (e.g., Qwen3‑30B, Claude‑4‑Sonnet), commercial deep‑research agents (OpenAI o3, DeepResearch, etc.), and leading open‑source agents (GLM‑4.5, DeepSeek‑V3.1, WebSailor, etc.) (Section 3.1.2).
  - Tools and hyper‑parameters: five core tools; temperature 0.85, repetition penalty 1.1, top‑p 0.95; max 128 tool calls; 128K context (Section 3.1.4).
- Main results (Tables 1–2)
  - General web search (Table 1):
    - AgentFounder‑30B: 39.9 (BrowseComp‑en), 43.3 (BrowseComp‑zh), 72.8 (GAIA), 73.0 (Xbench‑DeepSearch), 71.9 (WebWalkerQA).
    - Relative position:
      - Beats all open‑source agents on 4/5 benchmarks and is near or above some commercial offerings (e.g., 72.8 on GAIA vs. 70.5 for OpenAI‑o3 in Table 1).
      - On BrowseComp‑zh, performance is strong but not best among open‑source (49.2 for DeepSeek‑V3.1 vs. 43.3); the paper attributes this in part to lower Chinese data coverage and search bias (Section 3.2).
  - Scenario‑targeted (Table 2):
    - AgentFounder‑30B achieves 31.5% Pass@1 on HLE, 47.9% on DeepResearch Bench (RACE Overall), 89.6% on Frames, 43.9% on SEAL‑0, and 75.3% on AcademicBrowse.
    - Notable: first open‑source model above 30% Pass@1 on HLE, and substantially above open‑source peers on AcademicBrowse (75.3 vs. 65.0 for DeepSeek‑V3.1).
- Do experiments support claims?
  - Evidence for “pre‑aligned agentic base helps all SFTs” (RQ2): Using the same SFT corpora, AgentFounder‑Base outperforms Qwen3‑30B‑Base across all three SFT variants (Table 3). Example: with SFT‑B, BrowseComp‑en jumps from 28.6 → 39.9 Pass@1; HLE from 27.0 → 31.5.
  - Evidence for two‑stage CPT (RQ3): Stage 1+2 consistently beats Stage 1 only, including a +4.1 Pass@1 on BrowseComp‑en and +8.0 Pass@3 on BrowseComp‑zh (Table 4).
  - Evidence for FAS and HAS contributions (RQ4): With 50B single‑stage CPT, FAS alone improves over non‑CPT across tasks; adding HAS further improves in most metrics, notably BrowseComp‑zh Pass@1 from 37.0 (FAS) → 40.1 (FAS+HAS) (Table 5).
  - Evidence for scaling (RQ5): Figure 6 shows monotonic improvements with both model size and CPT token count; Figure 7 shows strictly better SFT convergence as CPT data increases.
  - Behavioral analyses:
    - Tool‑use patterns adapt to task type (Figure 8): heavy‑tailed usage for BrowseComp‑en/HLE versus conservative usage for WebWalker/GAIA.
    - General tool‑use remains strong on ACEBench: 70.0 vs. 67.2 for the same backbone without CPT (Table 6).
    - Diversity scaling: Pass@n on BrowseComp‑en increases from 31.5 (n=1) to 75.8 (n=16) (Appendix B.2; Figure 10), indicating the model maintains diverse solution strategies—consistent with HAS’s training signal.
    - Difficulty sensitivity on GAIA: largest drop at Level‑3 (Pass@1 50.0; Pass@3 58.3) versus Level‑1 (79.5; 87.2), showing remaining gaps on the hardest problems (Appendix B.3; Figure 11).
    - MoE diagnostics: CPT balances expert activations in late layers, reducing “dead experts” (Appendix B.4; Figure 12), which may contribute to training stability cited in Section 3.6.1.
    - Accuracy vs. tool‑call turns shows expected trend—higher accuracy with fewer turns but non‑trivial success even >40 calls (17.5% average; Appendix B.5; Figure 13).
- Mixed/conditional findings:
  - Chinese browsing (BrowseComp‑zh) lags the best open‑source baseline (49.2 for DeepSeek‑V3.1). The paper links this to training‑data language mix and possible search engine bias (Section 3.2).
  - In a 50B single‑stage setting, adding HAS slightly lowers GAIA Pass@1 vs. FAS alone (72.8 → 69.9) while improving Pass@3 (80.6 → 82.5), suggesting a variance/recall trade‑off (Table 5).

> Table 1: “AgentFounder‑30B … 39.9% on BrowseComp‑en, 43.3% on BrowseComp‑zh, 72.8% on GAIA, 73.0% on Xbench‑DeepSearch, 71.9% on WebWalkerQA.”

> Table 2: “AgentFounder‑30B … 31.5% (HLE Pass@1), 47.9% (DeepResearch RACE Overall), 89.6% (Frames), 43.9% (SEAL‑0), 75.3% (AcademicBrowse).”

Overall, the experimental suite is broad (10 benchmarks), includes ablations (Tables 3–5), scaling (Figure 6), convergence evidence (Figure 7), and behavior analyses (Figures 8, 10–13), which together make a convincing case that Agentic CPT yields a stronger agentic foundation.

## 6. Limitations and Trade-offs
- Dependence on synthetic supervision:
  - FAS/HAS rely on LLM‑as‑judge filtering and correctness checks (Sections 2.2.2–2.2.3; Appendix B.1). Judge errors can pass flawed reasoning into CPT. The final answers are validated, but intermediate step quality is only indirectly controlled.
  - HAS does not execute alternative tool actions; it assumes semantic plausibility without verifying actual outcomes (Section 2.3). This may teach good decision heuristics but not true environment‑level causality.
- Language/domain skew:
  - Lower BrowseComp‑zh relative to the best open‑source baseline is attributed to less Chinese data and possible search bias (Section 3.2). Agentic CPT’s data mixture may under‑represent certain languages or domains.
- Compute and data scale:
  - CPT uses very large token budgets (e.g., 200B + 100B in Section 2.1 and up to 315B in scaling experiments; Figure 6b) and long contexts (128K in Stage 2). This brings significant compute/memory costs.
- Tool set and environment scope:
  - Experiments use five core tools (Appendix A.1) and a specific browsing stack (Google search, Jina). Different environments or APIs (e.g., non‑Google ecosystems) may change performance due to retrieval quality.
- Evaluation coverage:
  - GAIA evaluation uses only the text‑only subset (103 questions; Tables 1 and Section 3.2), so multimodal research capabilities are not tested here.
- Reproducibility of some baselines:
  - Several comparator scores come from provider reports or prior work (Tables 1–2). Differences in tool backends and prompts may affect cross‑paper comparability.

## 7. Implications and Future Directions
- Field impact:
  - Establishes “agentic foundation models” as a new target: pre‑align agent behavior during CPT so that post‑training can focus on alignment refinements rather than capability acquisition (Figure 2; Section 3.6.1).
  - Demonstrates that large‑scale, tool‑free synthesis (FAS) and trajectory reuse (HAS) can unlock open‑source agents that approach or surpass commercial systems on several tasks (Tables 1–2).
- Immediate follow‑ups enabled by this work:
  - Stronger multilingual CPT: expand entity‑anchored memories and trajectory sources in under‑represented languages to address BrowseComp‑zh gaps (Section 3.2).
  - Verified step‑level supervision: combine HAS with selective live execution or simulator feedback to label alternatives, turning contrastive decisions into grounded causal signals.
  - Better judges and filters: calibrate LLM‑as‑judge with meta‑evaluation or human audits to further increase FAS/HAS data reliability (Appendix B.1 shows a promising 82% retained accuracy).
  - Multi‑modal and multi‑agent CPT: extend FAS/HAS to images/tables/videos (the paper cites a multimodal agent in related work) and to cooperative/competitive agent settings.
  - Safety and robustness: integrate SEAL‑style adversarial retrieval during CPT to harden agents against misinformation and tool failures (Section 3.1.3; Table 2).
- Practical applications:
  - Enterprise research assistants (market analysis, patent/literature reviews), scientific tooling (systematic evidence synthesis), investigative journalism support, and education (explanatory research reports).
  - The reported ACEBench gains (Table 6) suggest portability to general tool‑use agents beyond web research.

In short, the paper’s central move—teaching agentic behavior during continual pre‑training with scalable, mostly offline data—reframes how open‑source communities can train robust research agents. The combination of FAS, HAS, and long‑context two‑stage CPT materially advances both capability and efficiency, while leaving clear avenues for multilingual, multimodal, and causally grounded extensions.
