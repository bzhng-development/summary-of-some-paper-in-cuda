# Agent Learning via Early Experience

**ArXiv:** [2510.08558](https://arxiv.org/abs/2510.08558)
**Authors:** Kai Zhang, Xiangchao Chen, Bo Liu, Tianci Xue, Zeyi Liao, Zhihan Liu, Xiyao Wang, Yuting Ning, Zhaorun Chen, Xiaohan Fu, Jian Xie, Yuxuan Sun, Boyu Gou, Qi Qi, Zihang Meng, Jianwei Yang, Ning Zhang, Xian Li, Ashish Shah, Dat Huynh, Hengduo Li, Zi Yang, Sara Cao, Lawrence Jang, Shuyan Zhou, Jiacheng Zhu, Huan Sun, Jason Weston, Yu Su, Yifan Wu
**Institutions:** 

## 1. Executive Summary (2-3 sentences)

Reasoning: The field needs a reward-free way for language agents to learn from their own interactions because many real-world environments don’t expose reliable rewards, and expert demonstrations alone don’t generalize well. This work proposes a concrete, scalable way to turn an agent’s own early interactions into supervision signals that improve performance and set up stronger reinforcement learning (RL) later.

The paper introduces the early experience paradigm: a reward-free way for language agents to learn from the future states caused by their own proposed actions. It operationalizes this through two methods—implicit world modeling and self-reflection—and shows consistent gains across eight diverse environments, improved out-of-domain robustness, and stronger post-RL performance (Sections 4–5; Table 2; Table 3; Figure 3).

## 2. Context and Motivation

Reasoning: To understand why this is needed, we first define the gap: RL is often impractical; imitation learning (IL) is brittle. The authors ask a natural question: can the agent’s own exploratory interactions supply useful supervision without explicit rewards?

- Problem addressed
  - Many environments that language agents operate in lack verifiable reward signals (e.g., websites do not provide clear success/failure feedback), or require long, inefficient interaction horizons that make RL unstable and costly (Introduction; Figure 1).
  - Supervised fine-tuning (SFT) on expert demonstrations (also called imitation learning, IL) is the current default, but it generalizes poorly because expert data covers a narrow slice of state space (Introduction; Related Work §2.1).
- Importance
  - Practically: Agents need to improve from their own interactions when rewards are missing or sparse. Without this, they remain brittle and require expensive, hard-to-scale expert data.
  - Theoretically: It offers a middle ground between IL (reward-free but static) and RL (dynamic but reward-dependent), aligning with the “era of experience” vision while acknowledging present-day constraints (Figure 1; Introduction).
- Prior approaches and limitations
  - IL/SFT: Dense, easy-to-train supervision but limited environment exposure; no learning from the consequences of the agent’s own (non-expert) actions; distribution shift at deployment (Eq. (1); §3.1, citing Ross et al. 2011).
  - RL: Effective when rewards and simulators exist (Atari, Go), but often infeasible or brittle in real-world agent settings due to missing rewards, long horizons, and infrastructure gaps (§2.1).
  - World models for agents exist, but usually as separate simulators; they add planning overhead and complexity (§2.2).
  - Prompt-time self-reflection often fails without external feedback and doesn’t update model parameters (§2.2).
- Positioning
  - The paper proposes early experience (Figure 1, §4): let the agent propose actions at expert states, execute them, and use resulting future states as supervision. This is reward-free, scalable, and integrates into existing SFT pipelines.
  - Two strategies under this paradigm:
    - Implicit world modeling (IWM): learn to predict next states from (state, action) pairs, using the same model as the policy (§4.2; Eq. (3)).
    - Self-reflection (SR): generate grounded rationales comparing expert actions to agent-sampled alternatives using the observed next states, and train on these rationales plus the expert action (§4.3; Eq. (4)).

## 3. Technical Approach

Reasoning: The core idea is to elevate the agent’s own exploratory interactions—specifically, the next states they cause—into direct learning signals. The two methods implement this principle differently: one as a prediction auxiliary task embedded in the policy, the other as contrastive, grounded reasoning about why better actions outperform worse ones.

- Formal setup (§3)
  - Decision-making is modeled as an MDP `M = (S, A, T, R, γ, ρ0)`, where states `s ∈ S` are textual observations (e.g., webpage DOM/accessibility text, tool outputs), actions `a ∈ A` are discrete choices (clicks, tool calls, text), and the policy is `πθ(a | s)`.
  - IL objective on expert trajectories `D_expert = {(si, ai)}` is standard cross-entropy (Eq. (1) `L_IL`), but suffers from distribution shift and no exposure to non-expert consequences (§3.1).

- Early experience data construction (§4.1)
  - For each expert state `si` in `D_expert`, sample `K` alternative actions `Ai = {a_i^1, …, a_i^K}` from the current policy `πθ(·|si)`. Execute each `a_i^j` in the environment to obtain the next state `s_i^j ~ T(si, a_i^j)`.
  - Construct a rollout dataset `D_rollout = {(si, a_i^j, s_i^j)}` (Eq. (2)), containing only non-expert actions to diversify experience.
  - Intuition: even without reward labels, the observed next states encode implicit feedback (e.g., error messages, page changes, tool outputs), revealing the consequences of choices.

- Method 1: Implicit World Modeling (IWM) (§4.2)
  - Training signal: Predict the next state `s_i^j` given the pair `(si, a_i^j)` using the same language model that serves as the policy. Objective (Eq. (3)):
    - `L_IWM = - Σ_{(si,a_i^j,s_i^j)∈D_rollout} log pθ(s_i^j | si, a_i^j)`
  - Design choices and rationale:
    - States are text, so next-state prediction reduces to standard next-token prediction—no separate simulator needed.
    - Using the policy model for next-state prediction “grounds” the policy in environment dynamics and provides a warm-up before resuming action supervision (two-stage: IWM then IL, §4.2).
    - Exposing the policy to many non-expert `(s, a)` pairs plus their outcomes builds robustness against distribution shift.
  - Example implementations:
    - WebShop: use offline textual summaries of the next page state after executing either expert or non-expert actions; final dataset size ~122,954 triplets (§B.2).
    - ALFWorld: for each state, sample 8 non-expert actions and include the expert action to form 189,279 triplets; predict textual consequences like “Nothing happens.” for invalid moves (§B.1).

- Method 2: Self-Reflection (SR) (§4.3)
  - Data: For each expert `si → ai → si+1` and each alternative `a_i^j → s_i^j`, prompt an LLM to produce a grounded chain-of-thought `c_i^j` explaining why the expert `ai` is preferable, using the differences between `si+1` and `s_i^j` (Self-Reflection Prompt Template in §4.3).
  - Training objective (Eq. (4)):
    - `L_SR = - Σ_{(si,a_i^j,c_i^j)∈D_refl} log pθ(c_i^j, ai | si)`
    - Mix `D_refl` with `D_expert` and train jointly; keep any expert-provided rationales if available.
  - Design choices and rationale:
    - Contrastive reasoning teaches general decision principles (e.g., respecting budgets, satisfying constraints), rather than rote action mapping.
    - Grounding matters: unlike untested rationales (e.g., STaR-style), SR’s explanations are based on actual next states from executing alternatives, reducing hallucination risk (§6.1; Table 4).
  - Example:
    - WebShop reflection emphasizes why “click [non-ears blue]” satisfies color and price constraints while alternatives violate them (§B.2). Tau-Bench reflections filter out invalid function calls using environment feedback (§B.4).

- Implementation discipline and fairness in comparisons (§5.1)
  - Use consistent prompting and decoding across methods. Fix the total optimization steps to match the IL baseline; IWM uses one epoch for world modeling then resumes IL so total steps match (§5.1).
  - Evaluate with official metrics/validators of each benchmark. Train/eval on up to 8×H100 GPUs (§5.1; Table 1).

- How this concretely improves learning
  - The agent directly experiences the outcomes of its own non-expert actions; these future states become supervision. For IWM, they teach dynamics; for SR, they fuel grounded reasoning on why some actions are suboptimal.
  - This is reward-free and scales with the agent’s own exploratory breadth (e.g., larger branching factor `K`; §6.3; Figure 4b).

## 4. Key Insights and Innovations

Reasoning: The novelty is not just two techniques, but the paradigm shift: upgrade an agent’s reward-free interactions into usable, high-signal supervision, then show it works across settings and scales.

- Early experience paradigm (Figure 1; §4)
  - Novelty: A middle ground between imitation (reward-free but static) and RL (reward-based but often infeasible). It uses the agent’s own exploratory next states as training targets—no external reward required.
  - Significance: Practical now (reward-free) and a strong bridge to RL later (Figure 3).

- Implicit world modeling as an auxiliary objective within the policy (§4.2; Eq. (3))
  - Different from prior “separate world model” approaches: no extra module, no planning overhead; simply predict next-state text with the policy model itself.
  - Significance: Lightweight, scalable grounding in environment dynamics; yields large gains in structured, transactional environments (e.g., WebShop) where next states are predictable (Table 2).

- Grounded self-reflection that compares expert and agent actions using observed outcomes (§4.3; Eq. (4))
  - Different from prior prompt-only reflection: generates rationales that are grounded in executed alternative outcomes (not hypothetical), and actually updates model parameters.
  - Significance: Teaches repairable decision-making and constraint prioritization; particularly effective for long-horizon, logic-heavy tasks (TravelPlanner +12.8–15.0 absolute; Table 2).

- Comprehensive, cross-domain validation and RL warm-start benefits (§5; Figure 3)
  - Beyond in-domain improvements, early experience enhances out-of-domain (OOD) robustness (Table 3).
  - Under identical GRPO recipes, early-experience checkpoints reach higher post-RL ceilings than IL-only starts (Figure 3; Tables 5–6, 8).

## 5. Experimental Analysis

Reasoning: To judge whether the claims hold, we check coverage (environments, models), fairness (training budgets), metrics, and specific quantitative improvements—including ablations and baselines.

- Evaluation setup (§5.1; Table 1)
  - 8 environments spanning embodied simulation (ALFWorld), scientific simulation (ScienceWorld), travel planning (TravelPlanner), multi-turn tool use (BFCLv3, Tau-Bench), retrieval + reasoning (SearchQA), and web navigation (WebShop, WebArena-Lite).
  - Models: Llama-3.2-3B, Qwen-2.5-7B, Llama-3.1-8B, plus scale-up experiments to Llama-3.3-70B and Qwen-72B for WebArena-Lite (Figure 5; Table 10).
  - Training budget parity: IL step count chosen by best val performance, then fixed for all methods; IWM uses a split of world modeling then IL with the same total updates; SR uses the same number of epochs as IL (§5.1).

- Main results (in-domain) (Table 2; also Tables 5–10 in Appendix B)
  - Consistent improvements across domains and models:
    - WebShop (transactional web): large jumps with IWM and SR. For Llama-3.2-3B, success rate improves from 41.8 (IL) to 60.2 (IWM) and 52.7 (SR): 
      > “WebShop, Llama-3.2-3B: Ours-IWM 60.2; Ours-SR 52.7” (Table 6).
    - ALFWorld (embodied): IL 80.5 → IWM 85.9; SR 85.2 for Llama-3.1-8B (Table 2; Table 5).
    - ScienceWorld (lab simulation): Llama-3.1-8B jumps from IL 54.7 to SR 68.0 (+13.3) (Table 2).
    - TravelPlanner (long-horizon planning): absolute gains +8.9 to +15.0 across models; e.g., Llama-8B IL 17.2 → SR 32.2 (+15.0) (Table 2; Table 9).
    - BFCLv3 (multi-turn APIs): +4–8 improvements; e.g., Llama-3.2-3B IL 21.3 → SR 29.3 (+8.0) (Table 7).
    - SearchQA (F1): smaller but consistent gains (e.g., Llama-8B IL 41.0 → IWM 44.3; SR 41.8) (Table 2; Table 8).
    - WebArena-Lite (noisy, open web): modest gains (e.g., Llama-8B IL 4.9 → IWM 8.5; SR 8.5) (Table 2; Table 10).

  - Takeaway: Early experience is particularly strong where next-state dynamics are consistent (IWM) or where reasoning over constraints is key (SR). Even in open, noisy environments (WebArena), gains are steady.

- Out-of-domain generalization (OOD) (Table 3)
  - ALFWorld (Llama-8B): IL 63.3 → IWM 78.1 (+14.8); SR 72.7 (+9.4).
  - BFCLv3 (Llama-3B OOD avg): IL 5.3 → IWM 8.9 (+3.6); SR 13.8 (+8.5).
  - SearchQA (Llama-8B): IL 47.4 → IWM 49.6 (+2.2); SR 50.7 (+3.3).
  - The authors argue the relative gains can match/exceed in-domain rates (e.g., SearchQA), indicating that learning from one’s own outcomes improves robustness to unseen states (§5.3).

- RL after early experience (Figure 3; Tables 5, 6, 8)
  - Using GRPO with identical hyperparameters across initializations:
    - WebShop (Llama-3.2-3B): post-RL success 82.0 (IL) vs 92.2 (IWM) vs 89.8 (SR) (Table 6; Figure 3a).
    - ALFWorld (Llama-3.2-3B): post-RL 78.9 (IL) vs 97.7 (IWM) vs 99.2 (SR) (Table 5; Figure 3b).
    - SearchQA (Llama-3B): post-RL F1 44.8 (IL) vs 50.0 (IWM) vs 46.3 (SR) (Table 8; Figure 3c).
  - Quote:
    > “Checkpoints from early-experience methods (IWM, SR) consistently lead to higher post-RL ceilings than imitation-only starts” (Figure 3 caption).
  - Also: RL directly from raw pretrained models underperforms and is unstable (§5.4).

- Baseline comparisons (Table 4; §6.1)
  - Long CoT (prompt-time, test-time scaling) helps slightly in some prompt-only cases but often breaks after SFT on expert trajectories lacking rationales (e.g., WebShop Llama-8B: IL 47.3 → +Long CoT 0.0). 
    > “+Long CoT … WebShop -47.3; ALFWorld -54.7 vs IL” (Table 4).
  - STaR-style rationales (ungrounded): low match rates and hallucinations degrade performance (WebShop Llama-8B: 47.3 → 25.0).
  - Early experience outperforms both by using grounded next states from actual environment rollouts.

- Data efficiency and branching factor (Figure 4)
  - Less human data: On WebShop, early experience using just 1/8 of the expert trajectories surpasses IL trained on the full dataset; ALFWorld shows a similar effect at 1/2 (Figure 4a).
  - Branching factor `K` (number of alternative actions per state):
    - IWM improves steadily with larger `K`; SR peaks at moderate `K` (e.g., 2–4), likely due to cognitive load and reduced contrast if many alternatives also succeed (Figure 4b).

- Model scaling (Figure 5; Table 10)
  - WebArena-Lite: gains persist from 3B → 8B → 70B. Even under LoRA for 70B, early experience remains on the top performance curve:
    > “Llama-3.3-70B: IL 13.3 vs IWM 16.4 vs SR 15.2” (Table 10; Figure 5).

- Overall assessment
  - The evaluation is broad (8 environments), multi-model, and includes OOD and RL-follow-up. Training budgets are controlled. The paper provides detailed environment-specific data construction (Appendix B). The evidence convincingly supports: (1) reward-free early experience improves policies over IL; (2) the gains transfer to OOD; (3) early experience yields stronger RL warm starts.

## 6. Limitations and Trade-offs

Reasoning: The approach is practical but not a panacea. It trades reward design for exploration design, next-state fidelity, and careful data construction.

- Assumptions and dependencies
  - Access to an interactive environment capable of executing alternative actions to yield next states; not all real systems allow this at scale (implicit in §4.1 and Appendix B).
  - Textual state representations (or summaries) must be sufficiently informative for prediction and reasoning (e.g., WebShop uses offline textual summaries of next pages; §B.2).
- Short-horizon supervision
  - Both IWM and SR primarily leverage single-step transitions `(si, a_i^j) → s_i^j`. They do not directly optimize long-horizon returns or multistep credit assignment without rewards (Limitations and Future Work).
- Data construction complexity
  - SR requires carefully designed prompts and quality filtering (e.g., WebArena-Lite filtered to 3,190 high-quality reflections; §B.8). Poorly grounded reflections can hurt (Table 4 discussion).
  - Some environments needed additional engineering (e.g., TravelPlanner gym; §B.7).
- Computational considerations
  - While reward-free, the approach still requires generating and executing multiple alternatives per state (branching factor `K`), which can be costly in high-latency environments (§6.3; Figure 4b).
- Mixed benefits by domain
  - Gains on open, noisy environments (WebArena-Lite) are modest, though consistent (Table 10). This suggests diminishing returns where dynamics are highly variable or observations are extremely cluttered.
- Potential failure modes
  - SR can become non-monotonic at large `K` due to diminished contrast or cognitive load (Figure 4b).
  - For highly stochastic environments, next-state prediction may capture spurious correlations unless summaries/filters are robust (cf. SearchQA world-modeling uses query-focused summaries; §B.5).

## 7. Implications and Future Directions

Reasoning: The key effect of this work is to give the community a scalable, reward-free training stage that is useful on its own and improves RL later. The natural next steps involve longer horizons, active data collection, and tighter integration with RL.

- How this changes the landscape
  - Establishes early experience as a standard mid-training stage: reward-free, scalable, and effective across environments (Figure 1; §5).
  - Provides a practical path to the “era of experience” by building agents that already understand dynamics and decision trade-offs before any RL (Figure 3).
- Enabled follow-ups
  - Long-horizon early experience: extend beyond single-step supervision to multi-step counterfactual branches and temporal abstractions; combine with trajectory-level consistency checks (not yet addressed; Limitations and Future Work).
  - Better selection of alternatives: active branching policies to maximize information gain (adaptive `K`).
  - Richer self-supervised objectives: e.g., contrastive next-state ranking, masked dynamics infilling, or consistency regularization across multiple next-state summaries.
  - Cross-environment transfer: use world-model-like representations learned in one domain to warm up another (not explored here).
  - Hybrid pipelines: interleave early experience with sparse verifiable rewards where available (continual learning setting; Limitations and Future Work).
- Practical applications
  - Web agents (shopping, form filling, dashboard analytics): IWM’s large improvements on WebShop suggest robust gains where UI transitions are regular (Table 6).
  - Tool-use assistants (customer support, operations): SR’s strength on BFCLv3 and Tau-Bench indicates better argument checking and tool sequencing (Table 7).
  - Scientific/embodied systems: improved procedural reliability and stronger RL ceilings (ScienceWorld, ALFWorld; Tables 5, 8; Figure 3).
  - Planning: constraint satisfaction and budget adherence in itinerary and operations planning (TravelPlanner; Table 9).

In sum, the paper shows that the consequences of an agent’s own actions—even without rewards—contain enough signal to materially improve policy learning. By formalizing early experience and instantiating it via implicit world modeling and grounded self-reflection, the work bridges imitation learning and reinforcement learning in a way that is both immediately practical and strategically important for building more general, resilient agents.