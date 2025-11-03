# The Landscape of Agentic Reinforcement Learning for LLMs: A Survey

**ArXiv:** [2509.02547](https://arxiv.org/abs/2509.02547)
**Authors:** Guibin Zhang, Hejia Geng, Xiaohang Yu, Zhenfei Yin, Zaibin Zhang, Zelin Tan, Heng Zhou, Zhongzhi Li, Xiangyuan Xue, Yijiang Li, Yifan Zhou, Yang Chen, Chen Zhang, Yutao Fan, Zihu Wang, Songtao Huang, Yue Liao, Hongru Wang, Mengyue Yang, Heng Ji, Michael Littman, Jun Wang, Shuicheng Yan, Philip Torr, Lei Bai
**Institutions:** 

## 🎯 Pitch

This survey introduces 'Agentic Reinforcement Learning' as a concept to extend large language models beyond single-turn interactions, using a POMDP framework to enable multi-step, adaptive decision-making. By unifying the action space for language and environment interaction, it paves the way for more reliable and scalable AI agents, addressing the critical gaps in planning, exploration, and temporal credit assignment, essential for real-world applications like autonomous software engineering and multimodal reasoning.

---

## 1. Executive Summary (2–3 sentences)
This survey defines “Agentic Reinforcement Learning (Agentic RL)” for large language models (LLMs) and formally separates it from classical preference-based reinforcement fine-tuning (RLHF-style post-training). It provides a unified POMDP-based framework, a twofold taxonomy (capabilities and tasks), and a consolidated map of algorithms, environments, benchmarks, and tooling—synthesizing 500+ works to show how RL turns heuristic LLM agents into adaptive, long-horizon decision-makers.

## 2. Context and Motivation
- Problem addressed
  - LLMs are increasingly deployed as “agents” that plan, use tools, search the web, maintain memory, and collaborate. Yet most prior RL-for-LLMs work focused on single-turn alignment (e.g., RLHF), which is a degenerate one-step setting that does not capture multi-step, partially observable, interactive decision-making (Section 2; Table 1).
  - Terminology and evaluation across agent papers are inconsistent, making it hard to compare methods or generalize across domains (Introduction; “Research Gap and Our Contributions”).
- Why this matters
  - Realistic applications (web research, software engineering, robotics, GUIs) are sequential and partially observable. Treating LLMs as single-step text generators misses planning, exploration, temporal credit assignment, and interaction costs—core properties of real tasks (Sections 1–2; Figure 2).
  - Post-2024 models (e.g., DeepSeek-R1, OpenAI o1/o3) highlight emergent reasoning under RL-style training; a unifying view is needed to direct research and engineering (Introduction; Section 2).
- Prior approaches and their limits
  - Preference-based RL fine-tuning (“PBRFT”; RLHF/DPO family) optimizes single responses against human/AI preferences with either a reward model or pairwise preferences (Section 2; Equations 10 and 15). It improves alignment but assumes full observability, single-step episodes, and outcome-only rewards (Table 1).
  - Prompted “agents” (e.g., ReAct) interleave reasoning and actions but often rely on hand-designed loop heuristics or SFT on traces, with limited adaptivity and weak guarantees (Section 3.2; Figure 5).
- How this survey positions itself
  - It formalizes Agentic RL as a POMDP where LLM policies choose both text and structured actions across time, receive step-wise/terminal rewards, and learn with policy-gradient or preference methods (Sections 2.1–2.7; Equations 5, 7–11).
  - It introduces a twofold taxonomy—by capability (planning, tool use, memory, self-improvement, reasoning, perception) and by task domain (search, code, math, GUI, vision, embodied, multi-agent)—to organize the fast-growing literature (Figure 1; Sections 3–4).
  - It catalogs environments and frameworks to enable reproducible Agentic RL research (Section 5; Tables 9–10).

## 3. Technical Approach
This is a survey; its “method” is a formal framework plus a taxonomy that reinterprets prior work.

Step-by-step framework (Section 2; Figure 2; Table 1)
- Decision model
  - PBRFT (single-turn alignment) is a one-step MDP: the “state” is just the prompt; the policy outputs one text sequence; the episode terminates; reward is a single scalar from a reward model or preferences (Section 2.1; Equation 1; Table 1).
  - Agentic RL is modeled as a POMDP: at each time t the agent receives an observation `o_t = O(s_t)` from hidden state `s_t`, chooses an action `a_t`, and the environment transitions stochastically to `s_{t+1}` (Sections 2.1–2.4; Equations 2, 4, 7).
- Action space (Section 2.3)
  - `A = A_text ∪ A_action` (Equation 5).  
    • `A_text`: free-form language for communication/intermediate thoughts.  
    • `A_action`: structured, environment-interactive acts (e.g., `call("search", "query")`, `click(x,y)`), often delimited in the output stream by `<action_start> ... <action_end>`.  
    • Actions can be composite (sequences of primitives), unifying primitive and higher-level operations.
- Rewards (Section 2.5)
  - PBRFT: `R(s0, a) = r(a)`—a single outcome score (Equation 8).  
  - Agentic RL: `R_agent(s_t, a_t)` can mix sparse task success with denser sub-rewards—e.g., unit-test pass, verifier success, intermediate progress (Equation 9).
- Objectives (Section 2.6)
  - PBRFT: maximize expected reward of one response (Equation 10).  
  - Agentic RL: maximize discounted cumulative return over horizons `T>1` (Equation 11), enabling exploration and long-term credit assignment.
- Optimization algorithms (Section 2.7)
  - REINFORCE (Equation 12): increases the log-probability of sampled actions weighted by returns minus a baseline; simple but high-variance.
  - PPO (Equations 13–14): uses ratio clipping to stabilize on-policy updates and a value critic for advantages.
  - DPO (Equation 15): bypasses an explicit reward model by optimizing pairwise preferences under a KL constraint to a reference model.
  - GRPO (Equations 16–17): removes the value critic using group-relative advantages (normalize returns within sampled groups). This reduces compute and stabilizes reasoning RL; many practical variants are summarized in Table 2.

How the survey structures the field (Figures 3–5; Sections 3–4)
- Capability taxonomy (Section 3)
  - Planning (Section 3.1): RL as an external guide (e.g., learn a value/heuristic to guide search) vs RL as an internal driver (directly optimize the LLM’s planning policy).
  - Tool use (Section 3.2; Figure 5): from ReAct-style prompting/SFT to Tool-Integrated Reasoning (TIR) optimized by RL, then toward long-horizon TIR with better temporal credit assignment.
  - Memory (Section 3.3; Table 3): from RAG-style retrieval to token-level or latent memory with RL-controlled write/retrieve/forget; future: structured (graph/temporal) memory under RL.
  - Self-improvement (Section 3.4): from verbal self-correction at inference, to internalizing reflection via RL, to fully autonomous self-training (e.g., self-play, curriculum generation).
  - Reasoning (Section 3.5): fast vs slow thinking; RL for slow, deliberate reasoning with process supervision and test-time scaling.
  - Perception (Section 3.6): transfer R1-style RL to multimodal; promote active perception via grounding, tool use, and imagination (image/video generation during reasoning).
- Task taxonomy (Section 4; Figure 6)
  - Search & research, code agents, math (informal and formal), GUI, vision, embodied agents, multi-agent systems, and others—each with RL methods and benchmarks (Tables 4–8).

Why these design choices
- POMDP formalization and `A_text ∪ A_action` cleanly unify language and tool/environment interaction (Sections 2.1–2.3), making it possible to study when to communicate, think, retrieve, or act.
- Reward shaping and process-level signals address sparse, delayed feedback (Section 2.5; Sections 3.2, 3.7).
- GRPO and its family (Table 2) target the compute cost and instability of PPO with a critic, enabling wide adoption in reasoning/tool-use RL.

## 4. Key Insights and Innovations
1) A principled separation between single-turn “PBRFT” and multi-turn “Agentic RL”
- What’s new: A formal, symbol-based comparison (Table 1; Equations 1, 5, 7–11) recasts agent training as POMDP optimization, not one-shot text alignment.
- Why it matters: It exposes the core gaps—partial observability, multi-action sequences, exploration, and credit assignment—that determine success in real interactive settings (Sections 2.1–2.6; Figure 2).

2) Unifying action space for language and environment interaction
- What’s new: `A = A_text ∪ A_action` (Equation 5) and the idea that `A_action` can be composite—covering tool calls, GUI ops, movement, code execution—while `A_text` enables communication and deliberation (Section 2.3).
- Why it matters: This lets a single policy learn when to “think,” when to “say,” and when to “do”—the essence of agentic behavior.

3) Codification of RL roles across core agentic capabilities
- What’s new: A capability-centric taxonomy shows RL as (i) external guidance (learned value/heuristic to steer search) and (ii) internal driver (directly optimize the policy for plans, tool calls, memory, reflection) (Sections 3.1–3.6; Figures 3–5).
- Why it matters: It clarifies where RL actually adds adaptivity beyond SFT/prompting pipelines and highlights the bottleneck of temporal credit assignment in long-horizon TIR (Section 3.2, “Prospective”).

4) Consolidation of the algorithmic landscape with emphasis on GRPO-style reasoning RL
- What’s new: A clear exposition of REINFORCE, PPO, DPO, GRPO with equations (12–17) and a comparative table of practical variants (Table 2).
- Why it matters: The group-relative approach (GRPO) is a practical workhorse for reasoning RL: it reduces compute by removing the critic while keeping PPO-style stability—crucial for scaling (Section 2.7).

5) A practical compendium of environments and frameworks
- What’s new: Broad coverage of web/GUI/code/science/embodied/game environments and RL frameworks for agent training (Section 5; Tables 9–10).
- Why it matters: Enables reproducibility and accelerates experimental progress in Agentic RL.

## 5. Experimental Analysis
This survey synthesizes results across domains rather than executing new experiments. Below are representative evaluation setups and outcomes as reported in Sections 3–5.

Evaluation methodology used across domains
- Datasets/environments: web research (WebArena, Mind2Web; Section 5.1.1), GUI (AndroidWorld, OSWorld; Section 5.1.2), code (SWE-bench, LiveCodeBench, BigCodeBench; Section 5.1.3), math informal (MATH, GSM8K) and formal (Lean-based miniF2F, ProofNet; Sections 4.3.1–4.3.2; Table 6), embodied (VLN/robotics), and multi-agent collaborative settings (Section 4.7).
- Metrics: task success rate (web/GUI), pass@k (code/math), browse benchmarks (BrowseComp), formal proof verification rate (formal math), trajectory efficiency (e.g., tool-call counts), and process-step accuracy when step-level rewards are used (Sections 4.1–4.7).
- Baselines: SFT-only or prompt-only agents (e.g., ReAct), PPO/DPO-based alignments, and non-agentic LLMs; for formal reasoning, search-based expert-iteration pipelines (Sections 2.7, 4.3.2).

Main findings by domain
- Search & research (Section 4.1; Table 4)
  - RL over live or simulated search improves long-horizon information seeking.  
    > “ASearcher … enabling long-horizon search (40+ tool calls)” (Section 4.1.1).  
    > Closed-source OpenAI Deep Research achieves “51.5% pass@1 on BrowseComp” (Section 4.1.2).
  - Techniques include masking retrieved tokens, two-stage PPO for “when” vs “how” to search, step-wise rewards, and curriculum/self-search to avoid API costs (Section 4.1.1).
- Code agents (Section 4.2; Table 5)
  - Outcome rewards: unit-test pass and pass@k are standard. Large-scale RL yields strong repository-level performance (DeepSWE; Qwen3-Coder) (Section 4.2.3).
  - Process rewards: compiler/runtime/error traces and line-level mutation scores improve credit assignment (StepCoder, PSGPO, PRLCoder) (Section 4.2.1).
  - Iterative refinement: RL from execution feedback reduces attempts and boosts pass rates in multi-turn debugging (RLEF, µCode) (Section 4.2.2).
- Math reasoning (Section 4.3; Table 6)
  - Informal: RL with tool-integrated reasoning improves accuracy and induces self-correction behavior (ARTIST, ToRL, ZeroTIR). Data-efficient settings are notable:  
    > “1-shot RLVR … with only 1 example performs close to using a 1.2k-example dataset” (Section 4.3.1).
  - Formal: RL with binary verifier feedback achieves strong proof success; adding subgoal-level process rewards improves sample efficiency and interpretability (DeepSeek-Prover v1.5/v2, Leanabell-Prover-v2) (Section 4.3.2).
  - Notable result:  
    > “rStar2-Agent … achieves average pass@1 of 80.6% on AIME24 and 69.8% on AIME25” (Section 4.3.1).
- GUI agents (Section 4.4; Table 7)
  - RL over static traces improves step-level grounding and action prediction (GUI-R1, UI-R1, InFiGUI-R1, AgentCPM, UI-Venus).
  - Online RL with asynchronous rollouts scales to real devices and dynamic environments (WebAgent-R1, DiGiRL, MobileGUI-RL, ComputerRL), with curriculum filtering and recovery mechanisms improving stability (Sections 4.4.2–4.4.3).
- Vision and embodied (Sections 4.5–4.6)
  - Vision: RL transfers R1-style gains to multimodal reasoning, including verifiable localization rewards (IoU, mAP) and active perception via grounding and tools (Visual-RFT, Vision-R1, Ground-R1; Section 3.6).
  - Embodied: GRPO-style RL improves navigation/manipulation planning in VLA models, though sim-to-real remains a bottleneck (VLN-R1, TGRPO; Section 4.6).
- Multi-agent (Section 4.7; Table 8)
  - RL formalizations such as Dec-POMDPs (MAGRPO) support joint training with decentralized execution. RLCCF, MAPoRL, and FlowReasoner use multi-dimensional rewards (accuracy, complexity, efficiency) to train cooperative agents.

Ablations/robustness highlights (where discussed)
- Step/process rewards vs outcome-only: step-wise feedback consistently improves sample efficiency and stability (Sections 3.2, 3.7; Table 2’s PSGPO, Step-DPO, Step-GRPO).
- Offline/simulated search vs live APIs: simulated “self-search” and curriculum (SSRL, ZeroSearch) improve stability/cost, with transfer to online APIs at inference (Section 4.1.1).
- Compute scaling: multiple places report continued gains from longer RL horizons, consistent with “Agent RL Scaling Law” (Section 6.2, “Computation”; Section 3.5 cites RLVR scaling).

Do the experiments support the claims?
- The aggregated results consistently show RL improving long-horizon competence when rewards are verifiable (tests, proofs, localized metrics) and/or process-shaped (Sections 4.1–4.4). However, heterogeneity in benchmarks and metrics limits direct cross-paper comparisons (Section 5).

## 6. Limitations and Trade-offs
Assumptions and scope (Sections 1–2; “Primary focus” box)
- Focus is on “how RL empowers LLM-based agents” rather than alignment for harmfulness, pure non-LLM RL, or static-benchmark gains. Results depend on availability of verifiable rewards or good proxies.

Scenarios not addressed or challenging
- Temporal credit assignment for long-horizon Tool-Integrated Reasoning is still underdeveloped; sparse trajectory rewards make optimization brittle (Section 3.2, “Prospective”; Section 3.7).
- Sim-to-real for embodied agents: online RL on physical robots is costly and risky (Section 4.6).

Computational and scalability constraints (Section 6.2)
- RL training is compute-intensive; stability is sensitive to hyperparameters. Larger models risk “capability boundary collapse” without careful advantage shaping or hybrid policies (Section 6.2, “Model Size”; [RL-PLUS]).

Reliability and safety risks (Section 6.1)
- Security: reward hacking can entrench unsafe tool use; indirect prompt injection via tools/memory; multi-agent cascades (Section 6.1, “Security”).
- Hallucination: outcome-only RL can reinforce spurious reasoning and reduce abstention (“hallucination tax”), unless combined with process-level factuality checks or calibrated training data (Section 6.1, “Hallucination”).
- Sycophancy: preference-based rewards may overfit to agreeable responses; requires sycophancy-aware reward design and co-optimization of policy and reward models (Section 6.1, “Sycophancy”).

Open questions (Section 6.4)
- Mechanistic debate: does RL mainly “amplify” pretraining capabilities by better sampling/selection, or can it install qualitatively new reasoning? Evidence is mixed across tasks and data exposure.

## 7. Implications and Future Directions
How this survey changes the landscape
- By reframing LLM agent training as POMDP optimization with a unified action space (`A_text ∪ A_action`) and step-wise rewards, it sets a common language for comparing methods across planning, tools, memory, and perception (Sections 2–3; Figures 2–5).
- The compendium of environments/frameworks (Section 5; Tables 9–10) lowers barriers to running controlled Agentic RL at scale.

What follow-up research it enables
- Better temporal credit assignment for TIR: step-level verifiers, hierarchical advantages, and curricula that target decision bottlenecks (Sections 3.2, 3.7).
- RL for structured memory: policies that create, link, and abstract graph/temporal memories (Section 3.3, “Prospective”).
- Meta-reflection: learn a policy over reflective strategies—when to verify, branch, or search—to balance latency vs reliability (Section 3.4, “Prospective”).
- Co-evolving environments: automated reward learning and curriculum generation (EnvGen) to keep agents in their “zone of proximal development” (Section 6.3).

Practical applications and use cases
- Search & research assistants that plan multi-source investigations and synthesize long reports (Section 4.1; “Deep Research” systems).
- Autonomous software engineering: repository-level debugging, refactoring, and test generation under verifiable signals (Section 4.2).
- Mathematical assistants: informal problem solving with code tools and formal theorem proving with proof assistants (Section 4.3).
- GUI and OS agents: robust, long-horizon desktop/mobile automation via online RL (Section 4.4).
- Multimodal agents: grounded visual/video reasoning, active perception, and tool-augmented understanding (Sections 3.6, 4.5).
- Embodied robotics: navigation/manipulation with VLA models improved by trajectory-level RL (Section 4.6).

> In short, the survey argues—by formalization (Sections 2.1–2.7), taxonomy (Sections 3–4), and a curated ecosystem (Section 5)—that reinforcement learning is the key lever that transforms LLMs from “text predictors” into adaptive, reliable, and scalable decision-making agents, while also surfacing the remaining scientific challenges in safety, training efficiency, and environment design (Section 6).
