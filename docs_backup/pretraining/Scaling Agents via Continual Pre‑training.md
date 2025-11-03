# Scaling Agents via Continual Pre‑training

**ArXiv:** [2509.13310](https://arxiv.org/abs/2509.13310)
**Authors:** Liangcai Su, Zhen Zhang, Guangyu Li, Zhuo Chen, Chenxi Wang, Maojia Song, Xinyu Wang, Kuan Li, Jialong Wu, Xuanzhong Chen, Zile Qiao, Zhongwang Zhang, Huifeng Yin, Shihao Cai, Runnan Fang, Zhengwei Tao, Wenbiao Yin, Chenxiong Qian, Yong Jiang, Pengjun Xie, Fei Huang, Jingren Zhou
**Institutions:** Alibaba Tongyi Lab

## 🎯 Pitch

Agentic Continual Pre‑training (CPT) revolutionizes training of agentic foundation models by inserting an intermediate learning stage that pre-aligns systems for planning, tool use, and reasoning. This approach significantly enhances performance on complex research tasks, providing a robust framework for scalable and efficient training of autonomous agents, marking a pivotal step towards building versatile AI for real-world applications.

---

## 1. Executive Summary
This paper introduces Agentic Continual Pre‑training (Agentic CPT), a new training stage inserted between standard pre‑training and post‑training to build “agentic foundation models” that already know how to plan, invoke tools, and reason in multi‑step environments. Using Agentic CPT plus a scalable data synthesis pipeline (FAS and HAS), the authors train AgentFounder‑30B, which achieves state‑of‑the‑art results on 10 deep‑research benchmarks while keeping general tool‑use strong (e.g., 39.9% on BrowseComp‑en, 31.5% Pass@1 on HLE; Table 1 and Table 2).

## 2. Context and Motivation
- Problem addressed
  - Modern large language models (LLMs) are increasingly used as autonomous “agents” that must plan, invoke tools (e.g., search, browsing, code), and adapt across multiple steps. However, open‑source agent systems that train only with post‑training (Supervised Fine‑Tuning, Reinforcement Learning) underperform on complex agentic tasks (Introduction; Sec. 1).
  - The paper identifies a root cause: using general‑purpose foundation models forces post‑training to teach both new agentic capabilities and their alignment with expert behavior at once, creating “optimization tensions” and overfitting to specific trajectories instead of robust decision‑making (Sec. 1).

- Why it matters
  - “Deep research agents” (defined here as models that autonomously search, browse, code, and synthesize to answer research‑level questions) are now practical products (e.g., OpenAI Deep Research). Reliable, general, and efficient training for such agents has immediate real‑world impact (Introduction; Related Work 4.1).

- Limitations of prior approaches
  - SFT/RL depend on high‑quality, full trajectories that are costly and sparse; they supervise “what happened” deterministically instead of “what could be done” (Sec. 1).
  - General LLMs lack “agentic inductive biases,” so post‑training must teach both capability and alignment simultaneously, often locking models into imitation rather than exploration (Sec. 1).
  - Even the best open‑source agents (e.g., WebSailor‑72B 12.0 on BrowseComp‑en; GLM‑4.5 26.4; DeepSeek‑V3.1 30.0) lag far behind proprietary systems (OpenAI o3 49.7; OpenAI Deep Research 51.5) on BrowseComp‑en (Table 1).

- Positioning
  - The paper reframes the training pipeline by inserting Agentic CPT (Sec. 2.1, Fig. 2). This stage pre‑aligns models to agentic behavior before SFT/RL, using large‑scale, offline‑synthesized agentic data that covers planning, reasoning, and step‑wise decision spaces—without calling external tools during data creation.

## 3. Technical Approach
The approach has three pillars: a revised training pipeline, a scalable data synthesis method (FAS and HAS), and a progressive two‑stage CPT training schedule.

- Pipeline at a glance (Sec. 2.1; Fig. 2)
  - Start from a pre‑trained base (e.g., `Qwen3‑30B‑A3B‑Base`).
  - Agentic CPT Stage 1: ~200B tokens at 32K context; next‑token prediction (Eq. 1) on mixed agentic corpora to acquire preliminary tool‑use patterns and multi‑step reasoning.
  - Agentic CPT Stage 2: ~100B tokens at 128K context; higher‑quality, longer agentic samples to refine long‑horizon planning and action spaces.
  - Post‑training: SFT (and compatible with RL) on instruction data and agent trajectories to specialize for deep research tasks.

- Learning objective (Eq. 1, Sec. 2.1)
  - Both CPT stages use standard next‑token prediction with cross‑entropy loss:
    - Predict the next token `x_{t+1}` given context `x_1..x_t` via `softmax(W_o h_t)`.
    - Key point: no special loss is introduced; agentic behavior is taught through data design.

- Data Synthesis I: First‑order Action Synthesis (FAS) (Sec. 2.2; Fig. 3–4)
  - Goal: scale training contexts and first‑step actions without paying for real API calls.
  - Step A: Knowledge‑to‑Question transformation (Sec. 2.2.1; Fig. 3)
    - Build an entity‑anchored “open‑world memory”: map entities (e.g., “Paris”) to dense, time‑stamped statements harvested from web corpora, search results, and other sources. Unlike knowledge graphs, this focuses on per‑entity statement density rather than fixed relations.
    - Sample clusters of entities and statements to synthesize multi‑style questions (factual retrieval, numerical computation, multi‑hop reasoning, synthesis). The “Paris” example shows how three recent statements combine into a riddle that implicitly requires retrieval (Sec. 2.2.1, Example).
  - Step B: Planning Action Synthesis (Sec. 2.2.2; Fig. 4)
    - For each question, generate multiple analyses and predicted first actions (e.g., which tool to call next), but do not execute tools—keeping generation cheap and scalable.
    - Diversity trick: instead of K variants of the same question, generate first‑step actions for K different questions based on the same memory, widening context coverage.
    - Quality control: use “LLM‑as‑Judge” to reject samples whose first step is unlikely to reach the needed knowledge (“knowledge alignment verification”).
    - Effectiveness: weakly supervised filtering raises retained accuracy from 50% to 82% while removing 43.5% of data; semantic errors dominate rejected cases (Fig. 9; Appendix B.1).
  - Step C: Reasoning Action Synthesis (Sec. 2.2.3)
    - Two‑step reasoning without tools:
      1) Decompose question into sub‑questions and hypothesize answers using internal knowledge to form a preliminary answer A1.
      2) Provide the mapped requisite knowledge (from the memory) to refine A1 into A2, correcting logic and synthesizing evidence.
    - Reject sampling again checks that A2’s final answer matches ground truth; the surviving chain‑of‑thought is treated as reliable (Sec. 2.2.3, Example).

- Data Synthesis II: High‑order Action Synthesis (HAS) (Sec. 2.3; Fig. 5)
  - Goal: reuse abundant, imperfect trajectories from SFT/RL by transforming them into step‑wise decision problems rather than single “imitate the best path” sequences.
  - Mechanism:
    - At each step `k` in a trajectory, construct the context `C_k` (original question plus prior steps and tool responses).
    - Generate N alternative “thought + invocation” candidates for the next step without executing tools; combine with the original step to produce `N+1` options and shuffle them (record the original option’s position `n_k`).
    - Create a contrastive “decision trace”: enumerate options, insert a local decision statement (“I will choose option n_k”), then append the real observed response `R_k`. At the end, append an overall success/failure judgment (`J ∈ {0,1}`).
  - Why it helps:
    - Teaches the model to recognize viable decision options and to link choices to downstream consequences, without needing fragile step‑level rewards.
    - Expands the local action space at every step, encouraging exploration and avoiding over‑fitting to one specific trajectory (Sec. 2.3; Fig. 5).

- Progressive two‑stage CPT (Sec. 2.1)
  - Stage 1 (32K) focuses on FAS and shorter HAS to bootstrap agentic patterns economically.
  - Stage 2 (128K) focuses on high‑quality HAS and long contexts to learn long‑horizon planning. Table 4 shows this yields consistent gains over Stage‑1‑only training.

- Post‑training configurations (Sec. 3.1.1 Data; Table 3)
  - Three SFT settings on top of the CPT base:
    - `SFT‑A`: general conversation first, then React‑style trajectories with explicit reasoning.
    - `SFT‑B`: two stages, but each uses a balanced mix of general and React‑style data.
    - `SFT‑C`: two stages with general conversation and React trajectories with summarized (shortened) reasoning.
  - This tests whether the CPT base adapts robustly to different post‑training mixes.

- Evaluation tools and inference settings (Sec. 3.1.4; Appendix A.1)
  - 5 tools: `Search`, `Visit` (goal‑conditioned page summarization), `Google Scholar`, `Python Interpreter`, `File Parser`.
  - Inference: temperature 0.85, repetition penalty 1.1, top‑p 0.95; max 128 tool calls; 128K context.

## 4. Key Insights and Innovations
- Agentic CPT as a new intermediate training stage (Fundamental)
  - What’s new: Insert a large‑scale, agent‑behavior‑focused continual pre‑training stage between pre‑training and SFT/RL (Fig. 2).
  - Why it matters: It “pre‑aligns” the base to agent behaviors, letting SFT focus on specialization instead of teaching basic planning/tool patterns. Evidence: SFT loss is consistently lower for CPT models compared to the same baseline on identical SFT corpora (Fig. 7: baseline final loss 0.8656 vs. 0.7953 for the best CPT model).

- Scalable, tool‑free agentic data synthesis (FAS) (Practical + Conceptual)
  - What’s new: Transform static web knowledge into dynamic multi‑style questions, then synthesize planning and reasoning actions without executing tools (Sec. 2.2; Fig. 3–4).
  - Why it matters: Avoids high API costs and enables massive offline scaling. Filtering boosts retained data accuracy to 82% (Fig. 9), providing affordable, reliable supervision signals for planning and logic.

- Step‑wise decision modeling via HAS (Conceptual + Practical)
  - What’s new: Convert trajectories into multi‑option decision traces at every step, merging original and synthetic alternatives with real tool responses and a final success label (Sec. 2.3; Fig. 5).
  - Why it matters: Reuses “suboptimal” or discarded trajectories safely, teaches decision‑making rather than mere imitation, and avoids unstable step‑level rewards.

- Two‑stage long‑context CPT with clear scaling behavior (Methodological)
  - What’s new: A progressive regimen (32K→128K) that first builds agentic priors, then trains long‑horizon planning with 128K context windows.
  - Why it matters: Delivers consistent improvements over single‑stage training (Table 4) and exhibits a clear logarithmic scaling law with more CPT tokens (+8.0% Pass@3 from 0B→315B; Fig. 6b), indicating predictable returns from further scaling.

## 5. Experimental Analysis
- Evaluation setup (Sec. 3.1)
  - Benchmarks (10 total) span general web search and scenario‑targeted tasks:
    - General: BrowseComp‑en/zh, GAIA (text‑only subset of 103 Qs), Xbench‑DeepSearch, WebWalkerQA (Sec. 3.1.3).
    - Scenario‑targeted: DeepResearch Bench (RACE Overall), SEAL‑0, Frames, HLE, Academic Browse.
  - Metrics: Pass@1 and Pass@3 (chance the top‑k final answer is correct); tool‑equipped evaluation with a standard 5‑tool suite (Sec. 3.1.4; Appendix A.1).
  - Baselines: strong general LLMs with tools, commercial deep‑research agents, and top open‑source agents (Sec. 3.1.2).

- Main results (Tables 1–2)
  - General web search (Table 1)
    > AgentFounder‑30B: 39.9% (BrowseComp‑en), 43.3% (BrowseComp‑zh), 72.8% (GAIA), 73.0% (Xbench‑DeepSearch), 71.9% (WebWalkerQA).
    - Beats all open‑source agents on BrowseComp‑en by a large margin (DeepSeek‑V3.1: 30.0; GLM‑4.5: 26.4).
    - Competitive with or better than commercial systems on some tasks (e.g., GAIA 72.8% vs o3 70.5%).
    - On BrowseComp‑zh, it trails DeepSeek‑V3.1 (49.2) and OpenAI o3 (58.1), which the paper attributes to less Chinese data and possible search tool bias (Sec. 3.2).
  - Scenario‑targeted (Table 2)
    > AgentFounder‑30B: 31.5% Pass@1 (HLE), 47.9% (DeepResearch Bench RACE Overall), 89.6% (Frames), 43.9% (SEAL‑0), 75.3% (Academic Browse).
    - First open‑source model above 30 on HLE (31.5%), surpassing several commercial agents listed.
    - Substantial lead on Frames (89.6%) over both open‑ and closed‑source entries listed.
    - Strong on Academic Browse (75.3%), emphasizing research utility.

- Does CPT really help post‑training? (Table 3)
  - For all three SFT data mixtures (SFT‑A/B/C), models initialized from `AgentFounder‑30B‑Base` outperform those from `Qwen3‑30B‑A3B‑Base` on BrowseComp‑en/zh, GAIA, and HLE.
  - Example: with `SFT‑B`, CPT boosts BrowseComp‑en from 28.6 to 39.9 (+11.3) and BrowseComp‑zh from 35.6 to 43.3 (+7.7). GAIA sees modest gains (+1.0), HLE gains +4.5 (Table 3).

- Two‑stage vs. single‑stage CPT (Table 4)
  > Adding Stage 2 (128K) improves Pass@1 by +4.1 (BrowseComp‑en), +2.9 (BrowseComp‑zh), +2.9 (GAIA), and Pass@3 by +2.1/+8.0/+0.9 respectively.

- FAS vs. FAS+HAS (Table 5)
  - FAS alone already lifts performance over Non‑CPT. FAS+HAS is complementary: it improves several metrics, with some variance (e.g., GAIA Pass@1 dips −2.9 but Pass@3 rises +1.9), indicating normal evaluation variance and complementary benefits.

- Scaling studies (Fig. 6)
  - Model size scaling: average accuracy rises from 20.4% (1B) → 32.7% (4B) → 48.9% (30B). The 30B CPT model surpasses larger non‑CPT baselines like DeepSeek‑V3.1 (43.0%) and Kimi‑K2 (29.6%) (Fig. 6a).
  - Data scaling: Pass@3 improves logarithmically from 54.2% (0B) → 62.2% (315B), with biggest gains early (+3.8% by 15B) and continued improvements with Stage‑2 long‑context at 65B and 315B (Fig. 6b).

- Training dynamics (Fig. 7)
  > On identical SFT data, CPT models converge to lower cross‑entropy loss than baseline; best CPT model reaches 0.7953 vs 0.8656 (Fig. 7).

- Agent behavior analyses
  - Tool‑use adaptation: heavy‑tailed tool calls for open‑ended tasks (BrowseComp‑en, HLE) vs. conservative usage for structured tasks (WebWalker, GAIA) (Fig. 8).
  - General tool‑use benchmark: ACEBench overall 70.0 vs 67.2 for the base model, indicating CPT improves broad tool skills (Table 6).
  - Sampling diversity: Pass@1→Pass@16 grows from 31.5%→75.8% on BrowseComp‑en, showing preserved solution diversity (Fig. 10).
  - Difficulty sensitivity: GAIA performance drops with task level (Level‑3 Pass@1=50.0%) (Fig. 11).
  - MoE routing becomes more balanced post‑CPT in later layers, potentially aiding stability and avoiding “dead experts” (Fig. 12).
  - Accuracy vs. tool‑turns: higher success with fewer turns; still non‑trivial success even beyond 40 calls (avg. ~17.5%) (Fig. 13).

- Convincingness and caveats
  - Strengths:
    - Broad, multi‑benchmark superiority among open‑source agents, with some commercial‑level scores (Tables 1–2).
    - Systematic ablations link gains to CPT, two‑stage training, and data types (Tables 3–5; Figs. 6–7).
  - Caveats:
    - Some commercial baselines are reported from official sources and may not share identical tool stacks or constraints, limiting strict apples‑to‑apples comparisons (Sec. 3.1.2–3.1.4).
    - GAIA evaluation is on the text‑only subset (103 questions), not the full multimodal benchmark (Table 1 note; Sec. 3.2).

## 6. Limitations and Trade-offs
- Dependence on synthetic supervision
  - FAS and HAS avoid real tool calls during data creation, which is scalable but risks a mismatch between “synthetic options” and real web behaviors. LLM‑as‑Judge filtering helps (Fig. 9) but may import judge biases; no human audit statistics are reported (Sec. 2.2.2, 2.2.3, B.1).

- Language and search‑engine bias
  - BrowseComp‑zh performance lags top models (Table 1). The paper cites limited Chinese data and Google Search bias in Chinese contexts (Sec. 3.2), pointing to sensitivity to language distribution and tool ecosystems.

- Compute and data cost
  - Though data collection is API‑free, CPT uses very large corpora (≈300B tokens total) and long‑context training up to 128K (Sec. 2.1), which is compute‑intensive.

- Backbone and generality
  - Experiments start from the Qwen3 family; cross‑backbone generality is plausible but not demonstrated. Results may vary for architectures with different routing (e.g., non‑MoE vs MoE) or tokenizer vocabularies.

- Evaluation comparability
  - Some competitive numbers are drawn from official leaderboards or providers and may reflect different tool budgets, browsing policies, or time‑varying web content (Sec. 3.1.2–3.1.4). The paper mitigates this by re‑running many baselines under a standardized setup, but not all.

- Scope of tasks
  - GAIA results use the text subset only (Table 1 note), and the work focuses on web/tool‑centric research tasks. Capabilities for non‑web embodied control or specialized enterprise tools are not evaluated.

## 7. Implications and Future Directions
- Field‑level impact
  - Introducing Agentic CPT reframes how we train agent systems: instead of relying solely on post‑training to infuse agentic skills, pre‑train those skills with targeted data at scale. The consistent SFT loss reductions (Fig. 7) and scaling laws (Fig. 6) suggest that “agentic foundation models” can become a new baseline for downstream agent development.

- Research enabled
  - Better use of imperfect trajectories: HAS shows how to transform sparse, noisy RL/SFT logs into rich, step‑wise decision data without fragile step‑rewards (Sec. 2.3). This invites research into:
    - Learning option priors and step‑value estimators during CPT.
    - Unifying CPT and RL via offline reinforcement learning with synthetic branching.
  - Data synthesis science:
    - FAS’s entity‑anchored memory (Fig. 3) could evolve into continually updated open‑web memories with provenance and temporal consistency; integrating retrieval‑grounded consistency checks beyond LLM‑as‑Judge is a natural next step.
    - Active data acquisition loops: use failure analysis (e.g., Fig. 11, Fig. 13) to target hard cases and calibrate tool budgets.

- Practical applications
  - Enterprise research assistants (market, legal, clinical literature) that demand long‑horizon browsing and citation grounding.
  - Academic discovery support (strong Academic Browse and Google Scholar tool; Table 2; Appendix A.1).
  - Robust multi‑perspective synthesis (Frames 89.6%; Table 2) useful for journalism, policy analysis, and intelligence reporting.

- Immediate next steps
  - Strengthen multilingual coverage and search‑engine diversity to remove performance asymmetries (BrowseComp‑zh; Table 1).
  - Extend to multimodal deep‑research (images, PDFs with figures) and specialized tools (databases, citation managers).
  - Replace or augment LLM‑as‑Judge with calibrated reward models and human audits for filtering and validation.
  - Explore curriculum CPT schedules (e.g., gradually increasing option count in HAS, or staged tool ontologies) and cross‑backbone replication to test generality.

Overall, this paper’s main contribution—Agentic CPT plus scalable FAS/HAS data synthesis—offers a concrete and reproducible path for building agentic foundation models that learn to plan, decide, and use tools before post‑training, with strong empirical support across diverse benchmarks (Fig. 2; Tables 1–5; Figs. 6–8, 10–13).
