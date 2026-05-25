# Agent Learning via Early Experience

**ArXiv:** [2510.08558](https://arxiv.org/abs/2510.08558)

## 🎯 Pitch

This paper introduces 'Early Experience,' a scalable, reward-free paradigm that enables language agents to learn directly from the outcomes of their own exploratory actions—without relying on external rewards or curated expert supervision. By operationalizing this with two methods—Implicit World Modeling and Self-Reflection—the authors demonstrate consistent improvements in effectiveness, generalization, and downstream reinforcement learning performance across eight diverse environments. This work bridges the gap between imitation learning and reinforcement learning, unlocking robust, practical training for language agents in real-world settings where reward feedback is often unavailable.

---

## 1. Executive Summary
This paper proposes Early Experience, a reward-free training paradigm for language agents that uses the outcomes of the agent’s own exploratory actions as supervision. It operationalizes the idea with two concrete methods—Implicit World Modeling (predict next state after taking an action) and Self-Reflection (learn textual rationales comparing expert vs. agent-sampled actions)—and shows consistent gains across eight diverse environments, improved out-of-domain robustness, and higher ceilings when later fine-tuned with reinforcement learning (RL).

## 2. Context and Motivation
- Problem addressed
  - Training language agents with RL in real environments is hard because many settings lack verifiable rewards (e.g., websites that don’t return ground-truth success/failure signals) or require long, inefficient rollouts for credit assignment (multi-turn tool use). See Introduction (§1) and Figure 1 (“Era of Experience vs. Era of Human Data”).
  - Supervised fine-tuning (SFT) on expert demonstrations avoids rewards but generalizes poorly, scales expensively, and exposes the agent to a narrow slice of environment states (§2.1).
- Why this matters
  - Practical: Agents must operate on the web, in operating systems, or with APIs where dense, trustworthy rewards are rare. Being able to improve without rewards widens where agents can be trained and deployed.
  - Scientific: Bridging imitation learning (IL) and RL by extracting supervision from the agent’s own interactions advances the broader goal of “learning from experience” without relying on reward functions (§1).
- Prior approaches and limitations
  - Imitation learning/behavior cloning: learns from expert state–action pairs only; suffers from distribution shift and never sees the consequences of non-expert actions (Eq. 1; §3.1).
  - RL for agents: promising but brittle in language environments, often relying on approximate rewards, hand-tuned recipes, or fragile infrastructure (§2.1).
  - World models for agents: often trained as external simulators; planning-time overhead and integration complexity (§2.2).
  - “Self-reflection” prompting: mostly inference-time methods that falter without feedback/rewards (§2.2).
- Positioning
  - Early Experience sits between IL and RL (Figure 1, center). It requires no rewards but still learns from interaction by converting future states into supervision (§4). Two strategies are studied systematically: Implicit World Modeling and Self-Reflection (§4.2–4.3).

## 3. Technical Approach
The paper formalizes the environment as a Markov Decision Process (MDP) M = (S, A, T, R, γ, ρ0), where `S` are states (e.g., webpage content, tool outputs), `A` are actions (clicks, tool calls, text replies), and the policy `πθ(a|s)` maps states to action distributions (§3).

Step-by-step pipeline

1) Start from an expert dataset (`D_expert`)
- `D_expert = {(s_i, a_i)}_{i=1}^N` is the usual imitation learning set of expert state–action pairs (§3.1).
- Standard IL objective: minimize negative log-likelihood of expert action at each state (Eq. 1).

2) Generate Early Experience rollouts (`D_rollout`) by exploring alternatives to the expert action
- For each expert state `s_i`, sample `K` alternative actions `a_i^1, …, a_i^K` from the current policy `πθ(·|s_i)` (policy-sampled, not human) and execute each in the environment to get the resulting next state `s_i^j ~ T(s_i, a_i^j)` (§4.1).
- Collect triples `D_rollout = {(s_i, a_i^j, s_i^j)}` (Eq. 2). These “future states” are informative outcomes that require no reward signal.

Why this approach? It exposes the model to what happens after non-expert actions, giving direct, grounded feedback about consequences—precisely what IL lacks (§3.1).

3) Two training strategies that use `D_rollout`

A. Implicit World Modeling (IWM) (§4.2; Figure 2, left)
- Core idea: use the same policy network to predict the next state `s_i^j` given `(s_i, a_i^j)`, integrating a “world model” implicitly into the policy.
- Objective: next-token prediction of the next-state text
  > Eq. 3: `L_IWM = - Σ log p_θ(s_i^j | s_i, a_i^j)`
- Training schedule: two-stage
  - Stage 1: train with `L_IWM` over many `(state, action) → next-state` examples to internalize dynamics.
  - Stage 2: continue with standard IL on `D_expert` to focus on good actions.
- Design choices and rationale:
  - Use one model for both action selection and state prediction to “bake in” dynamics without a separate simulator (contrasts with explicit world-model planners; §2.2).
  - Works well when transitions are regular/predictable (e.g., e-commerce sites and simulators; §5.2).

B. Self-Reflection (SR) (§4.3; Figure 2, right)
- Core idea: generate a chain-of-thought `c_i^j` that explains why the expert action `a_i` is better than an alternative `a_i^j`, grounded in the observed outcomes `s_{i+1}` (expert next state) vs. `s_i^j` (alternative’s next state).
- Data construction:
  - For each expert state `s_i`, pair the expert outcome with K alternative outcomes and prompt a model to write a step-by-step rationale comparing them (Self-Reflection Prompt Template in §4.3).
  - Collect `D_refl = {(s_i, a_i^j, c_i^j)}`.
- Training objective: jointly predict the rationale and the (expert) action from the current state
  > Eq. 4: `L_SR = - Σ log p_θ(c_i^j, a_i | s_i)`
- Practice: mix `D_refl` with `D_expert`. This teaches general decision criteria (e.g., prioritizing constraints) instead of merely copying actions (§4.3).
- Design choices and rationale:
  - It converts suboptimal actions into dense comparison-based supervision, without explicit rewards.
  - Helps where task success involves satisfying constraints and multi-step reasoning (planning, tool use; §5.2).

4) Optional: Reinforcement Learning warm-started from Early Experience (§5.4; Figure 3)
- Where verifiable rewards are available (WebShop, ALFWorld, SearchQA), apply RL (GRPO) starting from IL vs. IWM vs. SR checkpoints under identical RL budgets.
- Finding: IWM/SR starts consistently lead to higher post-RL performance (§5.4; Tables 5–8 and Figure 3).

Concept clarifications (paper-specific terms)
- Early Experience: reward-free training that uses future states produced by the agent’s own exploratory actions as supervision (§4).
- Implicit World Modeling: next-state prediction task integrated into policy learning, not a separate simulator (§4.2).
- Self-Reflection (training-time): generating and training on grounded rationales that compare expert vs. agent-sampled actions using their observed outcomes (§4.3). This is different from purely prompt-based, inference-time reflection (§2.2).
- Branching factor `K`: number of alternative actions sampled per expert state (§4.1; ablated in Figure 4b).

Simple example (from §4.3)
- WebShop: If the goal says “under $20,” clicking a $30 red shirt yields a next state showing a $30 price. The reflection encodes: “It fits color but violates budget; the blue shirt at $15 satisfies both.” The model learns to prioritize constraints over superficial matches.

## 4. Key Insights and Innovations
- A practical bridge between IL and RL without rewards (fundamental)
  - The paradigm reframes “experience” as seeing what happens after your own actions, not as receiving numeric rewards (Figure 1; §4). This cleanly fills the gap where rewards are unavailable but interaction is possible.
- Implicit World Modeling integrated into the policy (novel integration)
  - Predicting next states with the same parameters used for the policy (Eq. 3; Figure 2) grounds action selection in environment dynamics without planning-time overhead or separate simulators (§4.2). Especially effective in structured, predictable environments (§5.2).
- Grounded Self-Reflection that compares outcomes (novel supervision signal)
  - Generates rationales tied to observed outcomes—unlike ungrounded rationales or inference-only reflection (contrast with STaR; §6.1 and Table 4). This trains general decision principles (constraints, tool-use logic) that transfer across contexts (§4.3).
- Strong warm-starts for RL (practical significance)
  - When rewards are available, early-experience checkpoints consistently lead to higher RL ceilings (Figure 3; Tables 5–8), indicating synergistic layering: experience first, reward later (§5.4).
- Scalability knobs and data efficiency (practical)
  - Works with less human data: with only 1/8 demonstrations on WebShop, Early Experience surpasses IL trained on 100% (Figure 4a). The branching factor `K` can be tuned to trade coverage vs. prompt complexity (Figure 4b).

## 5. Experimental Analysis
Evaluation setup (§5.1; Table 1 and Appendix B)
- Environments (8 total, diverse):
  - Embodied/simulated: `ALFWorld`, `ScienceWorld`
  - Planning: `TravelPlanner`
  - Multi-turn tool use: `BFCLv3`, `Tau-Bench`, `SearchQA`
  - Web navigation: `WebShop`, `WebArena-Lite`
- Models: `Llama-3.2-3B`, `Qwen-2.5-7B`, `Llama-3.1-8B` (plus scaling to 70B in §6.4).
- Baselines: Prompting (instruction-tuned), Imitation Learning (behavior cloning).
- Early Experience methods: IWM and SR, trained under the same update budgets as IL. Up to 8× H100 GPUs used.
- Metrics: Success rate (%) for most tasks; F1 for SearchQA.
- Additional comparisons/ablations:
  - Strong prompting with longer chains-of-thought (Long CoT) and STaR-style rationales (§6.1; Table 4).
  - Data fraction (Figure 4a), branching factor `K` (Figure 4b), model size (Figure 5).
  - Out-of-domain (OOD) splits for ALFWorld, BFCLv3, and SearchQA (Table 3).

Main quantitative results (highlights; full numbers in Table 2 and Appendix B)
- Overall effectiveness (Table 2)
  - WebShop: On `Llama-3.2-3B`, success improves from IL 41.8% to IWM 60.2% (+18.4); on `Llama-3.1-8B`, IL 47.3% → IWM 58.6% (+11.3).
  - TravelPlanner: Large gains with SR across all models, e.g., `Llama-3.1-8B` final pass rate 17.2% (IL) → 32.2% (SR) (+15.0).
  - ScienceWorld: `Llama-3.1-8B` rises from 54.7% (IL) to 68.0% (SR) (+13.3).
  - BFCLv3: On `Llama-3.2-3B`, IL 21.3% → SR 29.3% (+8.0).
  - Tau-Bench: On `Qwen-2.5-7B`, IL 33.9% → SR 39.5% (+5.6).
  - SearchQA (F1): On `Llama-3.1-8B`, IL 45.4 → IWM 48.0 (+2.6) and SR 48.0 (+2.6).
  - WebArena-Lite: Gains are modest but consistent; e.g., `Llama-3.1-8B` IL 4.9% → IWM 8.5% (+3.6).
- Action-space and observation regimes (§5.2)
  - Closed/finite action sets (ALFWorld, ScienceWorld, TravelPlanner): IWM internalizes transitions; SR repairs long-horizon plans (big SR gains in TravelPlanner).
  - Structured but large (BFCLv3, Tau-Bench): SR helps when errors are logical/tool-ordering; IWM reduces misuse.
  - Open action sets (SearchQA, WebArena): both methods still help, though gains are smaller due to combinatorial choices and noisy observations.

Out-of-domain generalization (Table 3)
- ALFWorld OOD with `Llama-3.1-8B`: IL 63.3% → IWM 78.1% (+14.8) and SR 72.7% (+9.4).
- BFCLv3 OOD with `Llama-3.2-3B`: IL 5.3% → IWM 8.9% (+3.6) and SR 13.8% (+8.5).
- SearchQA OOD F1 with `Qwen-2.5-7B`: IL 47.0 → SR 51.2 (+4.2) and IWM 49.5 (+2.5).
- Takeaway: Early Experience recovers a substantial share of OOD performance lost by IL (§5.3).

RL following Early Experience (Figure 3; Tables 5–8)
- WebShop (`Llama-3.2-3B`):
  > Table 6: IL+GRPO 82.0% vs. IWM+GRPO 92.2% and SR+GRPO 89.8%.
- ALFWorld (`Llama-3.2-3B`):
  > Table 5: IL+GRPO 92.2% vs. IWM+GRPO 97.7% and SR+GRPO 99.2%.
- SearchQA (All F1; `Llama-3.2-3B`):
  > Table 8: IL+GRPO 44.8 vs. IWM+GRPO 50.0 and SR+GRPO 46.3.
- Pattern (Figure 3): early-experience starts consistently yield higher post-RL ceilings under identical RL budgets (§5.4).

Ablations and controls
- Long CoT and STaR comparisons (§6.1; Table 4)
  - `Llama-3.1-8B` on WebShop: IL 47.3%; Long CoT on the IL model degrades to 0.0%; STaR-style training (ungrounded rationales) brings 25.0% (−22.3 from IL). Early Experience: IWM 58.6% (+11.3), SR 58.2% (+10.9).
  - Conclusion: grounded, outcome-based supervision matters; ungrounded rationales or inference-only reasoning are unreliable without feedback.
- Data efficiency (Figure 4a)
  - WebShop: with 1/8 demonstrations, Early Experience already beats IL trained on the full dataset.
  - ALFWorld: Early Experience with 1/2 demonstrations matches/exceeds full-data IL.
- Branching factor `K` (Figure 4b)
  - IWM improves steadily with larger `K` (more diverse transitions).
  - SR improves for small–moderate `K` (2–4) but can be non-monotonic when too many alternatives include other success paths, lowering contrast.
- Model scaling (Figure 5; Table 10)
  - WebArena-Lite averages improve at 70B as well: `Llama-3.3-70B` IL 13.3% → IWM 16.4% and SR 15.2%. Similar trends for `Qwen-2.5-72B` (IL 12.7% → IWM 17.6%, SR 15.8%).

Do the experiments support the claims?
- Breadth: Eight environments with varied action/observation structures (Table 1) and multiple model families demonstrate robustness (§5.2).
- Depth: OOD splits (Table 3), RL warm-start (Figure 3, Tables 5–8), ablations on data amount and `K` (Figure 4), and comparisons to strong alternatives (Table 4) substantiate the central claims: reward-free, grounded supervision improves effectiveness and generalization and sets up better RL.

## 6. Limitations and Trade-offs
- Focus on short horizons (§7, “Limitations”)
  - Both IWM and SR mainly use one-step transitions; extended credit assignment without rewards remains open.
- Reliance on environment stepability
  - Early Experience requires being able to execute alternative actions and capture next states. In locked-down or costly environments (e.g., real websites with rate limits, CAPTCHAs), rollout collection may be non-trivial (Appendix B notes engineering per environment).
- Quality and representation of next states
  - Some environments need summarization/normalization of noisy observations (e.g., SearchQA summarizes retrievals; WebArena uses accessibility trees with summarization; Appendix B.2, B.5, B.8). Poor summaries could bias supervision.
- Sensitivity to branching factor and prompt length
  - Very large `K` can hurt SR by diluting contrast (Figure 4b). Long outcome/context windows increase memory/compute.
- Compute and data engineering
  - Although capped at 8× H100 in reported runs (§5.1), collecting and storing large `D_rollout` sets (e.g., 122,954 triplets for WebShop in Appendix B.2) requires infrastructure and careful de-duplication/validation pipelines.
- Not a substitute for rewards where available
  - Early Experience is complementary. When verifiable rewards exist, RL still improves further; the contribution is stronger warm-starts, not replacing RL (§5.4).

## 7. Implications and Future Directions
- How this work changes the landscape
  - It establishes a practical, scalable middle ground: agents can learn from interaction without rewards by converting future states into training targets (Figure 1; §4). This “era of early experience” provides a clean on-ramp from imitation to reinforcement.
- Research enabled/suggested
  - Multi-step/long-horizon early experience: leveraging multi-step rollouts and temporal abstractions without rewards (noted as open in §7).
  - Richer self-supervised objectives on future states: contrastive predictors, representation learning across environments (§7).
  - Unified continual training: interleave Early Experience with RL as rewards gradually become available (§5.4 and §7).
  - Better outcome summarization: robust, reliable “state compressors” for noisy observations (Appendix B.5, B.8).
  - Policy-aware exploration: smarter selection of alternatives beyond random/policy sampling to maximize learning per step (Appendix B details several sampling heuristics).
- Practical applications
  - Web agents that improve via sandboxed browsing without platform feedback (WebShop, WebArena).
  - Multi-turn tool-use assistants that reduce misuse and improve argument selection (BFCLv3, Tau-Bench).
  - Search-augmented QA systems that learn to compose queries and reason over retrievals with fewer demonstrations (SearchQA).
  - Planning agents that internalize constraints and budgets from state transitions and reflections (TravelPlanner).

In short, by formalizing and validating Early Experience with two concrete, implementable methods (IWM and SR) and comprehensive experiments (Tables 2–10; Figures 1–5; Eqs. 1–4), the paper shows that learning from the consequences of one’s own actions—without rewards—substantially improves language-agent robustness and sets up stronger RL fine-tuning when rewards do exist.
