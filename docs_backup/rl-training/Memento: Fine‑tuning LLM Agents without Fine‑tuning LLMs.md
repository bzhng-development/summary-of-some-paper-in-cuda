# Memento: Fine‑tuning LLM Agents without Fine‑tuning LLMs

**ArXiv:** [2508.16153](https://arxiv.org/abs/2508.16153)
**Authors:** Huichi Zhou, Yihang Chen, Siyuan Guo, Xue Yan, Kin Hei Lee, Zihan Wang, Ka Yiu Lee, Guchun Zhang, Kun Shao, Linyi Yang, Jun Wang
**Institutions:** 

## 🎯 Pitch

Memento revolutionizes large language model (LLM) agents by enabling continual improvement without retraining, using a memory-augmented decision process that learns from episodic traces. This approach allows agents to adapt to open-ended environments with less computational overhead, achieving state-of-the-art results on complex benchmarks and paving the way for scalable, adaptive AI systems.

---

## 1. Executive Summary (2-3 sentences)
This paper introduces Memento, a way to make Large Language Model (LLM) agents improve over time without updating the LLM’s own weights. It formalizes agent planning as a memory-augmented decision process where the agent retrieves, adapts, and learns from past “cases” (episodic traces) via an online reinforcement-learning policy over memory, achieving state-of-the-art or near state-of-the-art results on long-horizon “deep research” benchmarks (e.g., GAIA, DeepResearcher) while remaining training-free for the base LLMs (Abstract; §1; Figure 1; Table 1–2).

## 2. Context and Motivation
- Problem addressed:
  - LLM agents need to operate in open-ended, changing environments (web research, tool use, multi-step reasoning). Two common strategies fall short:
    - Static workflows (handcrafted “reflection” and tool-use scripts) are rigid and don’t adapt post-deployment.
    - Fine-tuning LLMs (supervised or RL) is computationally expensive and not practical for continual online learning (§1; §2.1).
- Importance:
  - Real-world agents must learn continually from success and failure, incorporate new information, and generalize to out-of-distribution tasks—without constant retraining (§1; §2.3).
- Prior approaches and gaps:
  - Parametric updates: RL or supervised fine-tuning can improve behavior but are costly and susceptible to catastrophic forgetting (§2.1).
  - Retrieval-Augmented Generation (RAG): pulls text from static corpora; lacks mechanisms for online, selective learning from the agent’s own experience (§2.1).
  - Existing memory systems often append experiences indiscriminately (risking “swamping” where retrieval cost outweighs benefit) and lack a learned policy for what to recall (§2.3).
- Positioning:
  - Memento reframes planning as a Memory-based Markov Decision Process (M-MDP) with a learnable case-retrieval policy (case-based reasoning, CBR), enabling the agent to adapt by learning to select the right prior cases—without changing LLM weights (§3; Figure 2).

## 3. Technical Approach
High-level idea: treat planning as a single-step or few-step decision problem where the “action” is choosing which past case(s) to recall and adapt. The base LLM is frozen; only a lightweight policy over memory is learned.

A. Memory-Based MDP (M-MDP) and CBR policy
- Formalization (§3; Figure 2):
  - States `s` and actions `a` are text sequences (prompts/plans).
  - Memory `M` is a set of past cases `c = (s, a, r)`, where `r` is a scalar reward (e.g., success/failure).
  - At time t:
    1) Retrieve: pick a case `c_t ~ µ(c | s_t, M_t)`.
    2) Reuse & Revise: let the LLM generate an action `a_t ~ p_LLM(a | s_t, c_t)` using the retrieved case as context.
    3) Evaluation: receive `r_t`.
    4) Retain: append `(s_t, a_t, r_t)` to memory (i.e., memory grows).
    5) Transition: move to next state (`s_{t+1}`) via environment dynamics (§3, the factorized trajectory probability immediately under Eq. (1)).
- Overall policy (Eq. 1):
  - The agent’s action distribution is a mixture over retrieved cases:
    - π(a | s, M) = Σ_{c∈M} µ(c | s, M) · p_LLM(a | s, c).
  - Intuition: “what I do now” is a blend of “how I adapt” prior experiences most relevant to the current situation.

B. Learning the retrieval policy with maximum-entropy RL
- Objective (Eq. 3–6): maximize expected reward plus an entropy bonus on the retrieval policy `µ` to encourage diversity in recalled cases.
- Optimal form (Eq. 7): the best retrieval policy is a softmax over a learned Q-value for each case:
  - µ*(c | s, M) ∝ exp(Q*(s, M, c)/α), where `α` controls exploration/entropy.
- Q-learning update (Eq. 8): a soft Q-learning temporal-difference update for Q(s, M, c) with a log-sum-exp backup over next-step cases.

C. Making Q-learning practical for text states
Two implementations of the Q function are offered; both leave the base LLM untouched.

1) Kernel-based (non-parametric) Q approximation (§3; Eq. 9–11):
  - Store an episodic memory `𝒟` of tuples `(s, c, Q)`.
  - Approximate Q(s, M, c) as a similarity-weighted average of past Q-values for the same case `c`, using a trainable kernel `k_θ(s, s')` (Eq. 9).
  - Train `θ` by minimizing a TD loss (Eq. 10) with a derived gradient (Eq. 11).
  - Intuition: similar states should value the same case similarly; learning focuses on the similarity function rather than a global function approximator.

2) Neural Q for single-step CBR in planning (§4.2; Eq. 14–16):
  - In Memento’s planner, CBR is applied in a single planning step, so the TD target collapses to the immediate reward (no bootstrapping). The Q-function becomes a supervised predictor of “will this case help?” (Eq. 14).
  - Reward is binary (`r∈{0,1}`), so cross-entropy is used for stability instead of MSE (Eq. 15).
  - Retrieval uses Top-K cases with the highest predicted Q(s, c; θ) (Eq. 16), making selection deterministic and interpretable.

D. The Memento system architecture (§4; Figure 3)
- Planner–Executor loop (“plan-and-act”):
  - Planner: LLM-based CBR agent (default planner: `GPT-4.1`). It queries the Case Memory and generates a decomposed plan into subtasks, leveraging retrieved cases in the prompt.
  - Executor: an LLM running actions (default: `o3` for GAIA, `o4-mini` otherwise), connected to external tools via the Model Context Protocol (`MCP`). Each subtask is executed autonomously, reading and updating a Tool Memory.
- Three memory modules:
  - `Case Memory`: vectorized store of cases `(s, a, r)` for planning with Write (Eq. 12) and Read (Eq. 13 or Eq. 16).
  - `Subtask Memory`: textual log of current subtasks and results to support iterative replanning.
  - `Tool Memory`: textual log of tool calls and outputs per subtask to support reasoning and analysis.
- MCP-based tools (§4.3):
  - External information: metasearch (`searxng`), crawling (`Crawl4AI`).
  - Multimodal processing: images, audio, video, documents, spreadsheets.
  - Reasoning utilities: `Code` tool (sandbox with Python/shell, whitelisted libraries) and `Math`.
- Training loop sketch (Algorithm 1):
  - At each timestep: retrieve case(s), let the LLM act, observe reward, append the new case, and update either the kernel (non-parametric Q) or neural Q (parametric) via stored transitions.

E. Why these design choices?
- Keeping the LLM frozen avoids the cost and risk of fine-tuning and catastrophic forgetting (§2.1).
- Case-Based Reasoning (CBR) mirrors human analogy-making and reuses rich, structured trajectories, not just raw documents (§1; §2.3).
- A learnable retrieval policy avoids memory “swamping” by prioritizing high-utility cases (§2.3; Appendix B, Table 7).
- Single-step Q for planning simplifies learning (stable, supervised signal) while still enabling continual improvement (§4.2, “Parametric Memory Retrieval”).

## 4. Key Insights and Innovations
- Memory-augmented MDP for agent planning (fundamental):
  - Planning is framed as retrieving and adapting prior cases via a learned policy (Eq. 1; Figure 2). This is a principled decision-theoretic formulation that extends beyond static prompt engineering or RAG by explicitly modeling retrieval as an “action” with value (§3).
- Learning to retrieve, not to generate (fundamental):
  - The base LLM’s generation is fixed; optimization focuses on which experiences to recall. This decouples agent learning from model weights, enabling continual, low-cost adaptation via memory read/write (Eq. 7–8; Algorithm 1; §4.2).
- Two complementary memory realizations (practical innovation):
  - Non-parametric (similarity-based; Eq. 13) is simple and efficient; Parametric (neural Q; Eq. 15–16) learns which cases generalize best for a state. Ablations show both add value, especially in continual learning (Figure 1c; Table 4; Table 5).
- Clean separation of roles with MCP tooling (system innovation):
  - A planner specialized for high-level decomposition and a tool-savvy executor standardized via MCP allow scalable tool ecosystems and more reliable long-horizon behavior (§4; Figure 3; §4.3).

## 5. Experimental Analysis
A. Evaluation setup
- Benchmarks (§5.1):
  - GAIA: 450 long-horizon tasks (val 150, test 300) with 3 difficulty levels and exact-match grading.
  - DeepResearcher: seven open-domain QA datasets—`NQ`, `TQ`, `HotpotQA`, `2Wiki`, `MusiQue`, `Bamboogle` (125 samples), `PopQA` (512 each except Bamboogle)—targeting multi-hop web research.
  - `SimpleQA`: 4,330 single-hop factual questions (factual accuracy focus).
  - `HLE` (“Humanity’s Last Exam”): 2,500 diverse academic questions (long-tail, expert domains).
- Metrics (§5.2):
  - GAIA: `Exact Match (EM)` after normalization.
  - Others: `macro-F1` and `Partial Match (PM)` (GPT-4o-mini judge; same prompt as in DeepResearcher).
- Models/config (§5.3):
  - Planner: `GPT-4.1`; Executor: `o3` (GAIA), `o4-mini` (others); image: `GPT-4o`; video: `Gemini 2.5 Pro`; audio: `AssemblyAI`.
  - Case encoding: `SimCSE` + cosine similarity (non-parametric); neural Q is a 2-layer MLP initialized from SimCSE embeddings (parametric).

B. Main results
- DeepResearcher (Table 1):
  - Average across seven datasets: 
    > “Memento (GPT-4.1 + o4-mini) achieves 66.6% F1 and 80.4% PM,”
    substantially above prompt-only baselines (`CoT + RAG`: 37.7/43.2) and training-based systems (`DeepResearcher`: 51.8/60.5).
  - Notable per-dataset peaks: `2Wiki` 81.4 F1 / 94.1 PM, `Bamboogle` 86.2 / 92.8, `HotpotQA` 66.5 / 81.6 (Table 1).
- GAIA leaderboard (Table 2; Figure 1a):
  - Validation (Pass@3): 
    > “87.88% overall (L1 96.23, L2 90.70, L3 61.54),”
    ranked top-1 on validation among public agent frameworks.
  - Test (Pass@1): 
    > “79.40% overall (L1 90.32, L2 75.47, L3 71.43),”
    competitive among top entries.
- SimpleQA and HLE (Figure 4):
  - SimpleQA accuracy:
    > “95.0%,”
    outperforming WebSailor (93.5) and other web agents.
  - HLE PM:
    > “24.40%,”
    second on the shown bar chart, just 0.92 points behind GPT-5 at 25.32.

C. Ablations and analyses
- Component-wise contributions (Table 5):
  - Adding live tools to an “offline executor” helps in some settings (SimpleQA: +28.8 F1 / +63.3 PM) but can hurt on contaminated open-domain QA (DeepResearcher: −18.0 / −2.1).
  - Introducing planning (Memento w/o CBR) boosts all three benchmarks markedly (e.g., DeepResearcher: +29.1 F1 / +11.5 PM over Online Executor).
  - Adding CBR (full Memento) yields consistent extra gains (DeepResearcher: +6.7 F1 / +8.2 PM; SimpleQA: +3.7 / +5.3; HLE: +4.5 / +7.0).
- Continual learning curves (Figure 1c; Table 4):
  - Over five iterations, accuracy steadily rises; parametric CBR > non-parametric > no CBR.
    > Iter-5 (DeepResearcher): 85.44 (param CBR) vs 84.85 (non-param) vs 84.47 (no CBR baseline in this setup).
- How many retrieved cases? (Table 3):
  - Best at `K=4` (DeepResearcher avg: 64.5 F1 / 78.5 PM). Larger K plateaus or slightly degrades, highlighting the value of small, high-quality memories.
- OOD generalization (Figure 1d; §5.5.4):
  - Training cases from `NQ/TQ/HotpotQA/2Wiki` improve OOD datasets `MusiQue/Bamboogle/PopQA` by
    > +4.7% to +9.6% absolute,
    showing CBR helps transfer.
- Tool-use and cost (Figure 5–6; §6.1–6.2):
  - Code/search/crawl dominate and increase with task difficulty (Figure 5).
  - Token costs scale mostly via input context (integrating many tool outputs), not output length (Figure 6; Level-3 avg input ~121k tokens, output ~9.8k).
- “Fast vs slow thinker” planners (Table 6; §6.3):
  - Using `GPT-4.1` as “fast” planner with `o3` executor achieves the highest GAIA val Pass@1 average:
    > 70.91% vs 63.03% when using `o3` as both planner and executor.
  - Slow, deliberative planning often produced overly verbose or direct answers instead of structured plans, confusing the executor.

D. Do the experiments support the claims?
- The method’s core claims—training-free LLMs with learned memory retrieval can match or beat training-based systems—are supported by:
  - Strong aggregate numbers vs. prompt-only and RL-trained baselines (Table 1; Table 2).
  - Ablations isolating the incremental effects of tools, planning, and case-based retrieval (Table 5).
  - Robustness via OOD tests and continual-learning curves (Figure 1c–1d; Table 3–4).
- Caveats:
  - Some benchmarks rely on closed-source models and private leaderboards.
  - Evidence of dataset contamination complicates interpreting “online search vs parametric knowledge” on open-domain QA (§5.5.2).

## 6. Limitations and Trade-offs
- Assumptions about reward and granularity:
  - Binary rewards and single-step CBR for planning simplify learning (Eq. 15), but may not reflect nuanced, delayed credit assignment in longer horizons (§4.2).
- Memory growth and curation:
  - The system appends cases continuously (Eq. 12). Although learned retrieval mitigates swamping, explicit eviction/decay strategies are not the main focus; the authors note quick saturation with ~3k training cases and diminishing returns (end of §5.5.3).
- Computation shifts from training to inference:
  - No LLM fine-tuning costs, but token usage can be high for complex tasks (Figure 6), as inputs accumulate tool outputs, plans, and histories.
- Scope of learning:
  - CBR is applied to planning; the executor itself is not endowed with a learned memory policy beyond textual Tool Memory logs. Complex low-level execution strategies may still depend on base LLM capabilities (§4; Figure 3).
- Data contamination and evaluation confounds:
  - On DeepResearcher, an “online executor” without planning underperforms “offline executor,” suggesting overlap in parametric knowledge vs web retrieval that can mislead naive tool use (Table 5c; §5.5.2). This complicates clean isolation of search benefits.
- Dependence on powerful base LLMs:
  - Many gains assume strong planners/executors (e.g., GPT-4.1, o3, o4-mini). Table 6 shows weaker planners degrade GAIA performance.

## 7. Implications and Future Directions
- Field-level implications:
  - Demonstrates a credible path to “learning agents” without LLM fine-tuning: learn which experiences to recall, not how to generate. This reframes many agentic systems as memory-policy learners on top of frozen, general-purpose LLMs (Eq. 1; Eq. 7).
  - Encourages richer, structured episodic memories (cases) and policy-learning over them, beyond static RAG corpora.
- Practical applications:
  - Long-horizon web research, enterprise workflows with evolving tools, data science automation, compliance research, and support agents that must learn from operations logs without retraining core models (§4.3; results on GAIA, SimpleQA, HLE).
- Research directions:
  - Multi-step CBR and temporal credit assignment: extend beyond single-step planning to handle delayed rewards and per-subtask feedback (Eq. 5–8).
  - Memory curation and safety: principled forgetting, counterfactual evaluation of cases, provenance tracking, and privacy-preserving case banks (§2.3; Appendix B).
  - Cross-agent knowledge sharing: federated or organization-level case banks (cf. Agent-KB; §2.3) with learned retrieval that personalizes by task/domain.
  - Learning for execution: bring parametric/non-parametric CBR to the executor’s tool-calling decisions directly, unifying planning and acting under one memory policy.
  - Cost-aware retrieval and planning: jointly optimize token cost (Figure 6) with utility; learn to summarize or compress cases into reusable “skills” (cf. Eq. 9 kernel weighting; related §2.3 works on distilled guidelines).

Overall, Memento shows that carefully designed, learnable memory systems can deliver continual improvements in agent behavior, rivaling training-heavy approaches while keeping underlying LLMs frozen. The combination of an explicit M-MDP, a learned retrieval policy, and a modular planner–executor architecture provides a clear blueprint for scalable, adaptive LLM agents.
