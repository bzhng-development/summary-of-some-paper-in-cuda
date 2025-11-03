# ZEROSEARCH: Incentivize the Search Capability of LLMs without Searching

**ArXiv:** [2505.04588](https://arxiv.org/abs/2505.04588)

## 🎯 Pitch

ZEROSEARCH introduces a novel reinforcement learning framework that teaches large language models (LLMs) to search effectively—without ever accessing a real search engine during training. By replacing expensive and unstable live web searches with a fine-tuned simulation LLM that generates controlled, curriculum-adjusted documents, ZEROSEARCH eliminates API costs and stabilizes training, while achieving performance that matches or surpasses models trained on real-world queries. This breakthrough enables scalable, practical, and cost-effective development of LLMs with strong retrieval and reasoning skills, accelerating research and broad deployment of LLMs that interact with external tools.

---

## 1. Executive Summary (2-3 sentences)
ZEROSEARCH is a reinforcement-learning (RL) framework that trains large language models (LLMs) to use search effectively without querying a real search engine during training. It replaces live search with a fine-tuned “simulation LLM” that generates controlled, useful-or-noisy documents and employs a curriculum that gradually increases noise, yielding better or comparable search-and-reasoning performance to real-search RL while eliminating API cost (Sections 3–4; Tables 3–4, 8).

## 2. Context and Motivation
- Problem addressed
  - Training LLMs to search during reasoning improves accuracy and reduces hallucinations, but RL training that interacts with real search engines suffers from:
    - Uncontrolled document quality (noisy or too-good results destabilize training).
    - Prohibitively high API costs for frequent rollouts (Sections 1, 2.2).
- Why this matters
  - Practical: Tools like web search are fundamental to updating knowledge and grounding answers; lowering cost and increasing stability enables broader deployment and research (Section 1).
  - Scientific: Provides a general recipe for using LLMs as controllable simulators of external tools, expanding RL-with-tools without expensive, rate-limited environments (Sections 1, 3).
- Prior approaches and gaps
  - Retrieval-Augmented Generation (RAG): integrates search into generation but often needs careful prompting or heavy test-time compute (e.g., MCTS), and can be expensive at inference (Section 2.1).
  - RL for search:
    - On static corpora (e.g., Wikipedia) misses realistic web dynamics (Section 2.2).
    - With live engines (e.g., DeepResearcher, WebThinker): better realism but unstable document quality and high API cost (Section 2.2).
- Positioning
  - ZEROSEARCH uses an LLM, fine-tuned to simulate a search engine that can produce both useful and noisy documents on demand. It controls difficulty via a curriculum, stabilizes RL training, and removes API costs, while still evaluating with real Google search at inference to prove transfer (Sections 3, 4.3–4.4).

## 3. Technical Approach
Step-by-step overview
1) Interaction template that enforces structure
- The policy model reasons, searches, and answers using tags (Table 1, Section 3.2):
  - `<think>...</think>`: internal reasoning (no answer here).
  - `<search>...</search>`: emits a query.
  - `<information>...</information>`: the environment returns documents for the query.
  - `<answer>...</answer>`: concise final answer.
- Why: separating reasoning, retrieval, and answering makes the decision process explicit and reduces format errors (Section 3.2).
- Example (Table 9) shows multi-turn search followed by a short `<answer>`.

2) Replace live search with a “simulation LLM”
- Goal: a frozen simulator `π_ψ` that, given a query, outputs K short documents that either contain helpful evidence (“useful”) or distracting text (“noisy”).
- How the simulator is trained (Search Simulation Tuning; Section 3.3; Table 2):
  - Collect multi-turn trajectories where an LLM interacts with a real search engine until an answer is found.
  - Extract (query, document) pairs and use an LLM judge to label documents as “useful” if they contain sufficient info to answer the query, else “noisy.”
  - Lightweight SFT (supervised fine-tuning) teaches the simulator to generate 5 documents per query, with quality controlled by a simple prompt switch:
    > “You are the Google search engine. … generate five [useful / noisy] documents … The user is trying to answer [question] whose answer is [ground truth]. … documents should contain [useful / noisy] information.” (Table 2)
  - Including the question and ground-truth answer in the SFT prompt broadens the simulator’s coverage (Section 3.3). This is a pragmatic choice to ensure the simulator can produce useful evidence even if it did not memorize it during pretraining.

3) Curriculum rollout that controls difficulty
- During RL rollouts, each time the policy emits a `<search>` tag, its query is sent to the simulator, which returns 5 documents.
- Difficulty is governed by the probability of returning noisy documents at training step `i`, using an exponential schedule (Section 3.4, Eq. 1):
  - `p_i = p_s + ((b^(i/m) - 1) / (b - 1)) * (p_e - p_s)`  
    where `p_s`/`p_e` are start/end noise probabilities, `m` is total training steps, and `b=4` by default.
- Intuition: start easy (mostly useful docs), then gradually increase noise to force robust search-and-reasoning (Section 3.4).

4) RL objective and optimization
- Objective (Section 3.1):
  - `maximize J(θ) = E[r_ϕ(x, y)] - β * D_KL(π_θ(y | x; π_ψ) || π_ref(y | x; π_ψ))`
  - `π_θ`: policy model; `π_ref`: reference model; `r_ϕ`: reward function; `π_ψ`: fixed simulator. `β` controls the KL penalty to keep the policy close to the reference for stability.
- Reward design (Section 3.5):
  - Rule-based F1 score between the predicted final answer and ground truth:
    - `IN`: overlapping words; `PN`: words in prediction; `RN`: words in ground truth.
    - `reward = 2 * IN / (PN + RN)`
  - They avoid exact match (EM) during training because it led to “reward hacking” (overly long answers that increase chance of including the target string).
  - No extra formatting reward—empirically, the template sufficed (Section 3.5; Table 9 shows clean formatting).
- Algorithms supported (Section 3.6; Fig. 1):
  - REINFORCE, PPO, and GRPO. PPO uses a value model and GAE; GRPO uses group-based relative advantages (Figure 1 depicts both PPO and GRPO pipelines with the same simulated environment).

5) Stabilization via loss masking on external tokens
- Tokens of documents returned by the simulator are masked out of the policy loss because the policy did not generate them (Section 3.6). This prevents unstable gradients from being attributed to the policy for text it did not control.

6) Experimental setup in brief (Section 4.3)
- Policy backbones: `Qwen-2.5-7B`, `Qwen-2.5-3B`, `LLaMA-3.2-3B` (Base and Instruct variants).
- Simulator default: fine-tuned `Qwen-2.5-14B-Instruct`.
- Training hardware: 4 H20 GPUs for simulator server, another 4 H20 GPUs for RL.
- Evaluation uses real Google Web Search via SerpAPI with 5 retrieved documents for all methods to ensure fairness (Section 4.3). This directly tests whether policies trained in simulation transfer to real search.

Analogy
- The simulator acts like a “flight simulator” for search: affordable, controllable, and scalable. Curriculum alters weather conditions (document noise) to train robust pilots (policies) before flying in the real sky (Google at evaluation time).

## 4. Key Insights and Innovations
- LLM-as-search-engine simulator with controllable quality (Sections 3.3–3.4)
  - Novelty: A fine-tuned LLM can mimic a search engine’s role and be directed to emit useful or noisy documents by flipping a keyword in the prompt (Table 2), something live engines cannot offer.
  - Significance: Enables zero-API-cost RL and precise difficulty control that stabilizes learning (Figures 2–3, Table 6).
- Curriculum-based rollout that degrades document quality over time (Section 3.4; Eq. 1)
  - Different from prior RL-with-search work that relies on whatever the live engine returns; here difficulty is scheduled, yielding smoother and stronger learning:
    > ZEROSEARCH “initially lags behind [real-search training] but eventually surpasses it with less fluctuation” (Section 5.1; Figures 2a–b, 3).
- Loss masking for external tokens (Section 3.6; Appendix C)
  - Practical stabilization: Do not backprop through simulator-generated tokens. Ablation shows measurable gains:
    > With masking vs. without on LLaMA-3.2-3B-Base: “36.07 vs. 34.53” average EM (Table 7; Figure 4c).
- Scaling the simulator improves real-world transfer (Section 5.2; Table 4)
  - Fundamental claim: A 7B simulator matches Google; a 14B simulator surpasses it for training the same 3B policy, when all are evaluated with real Google search:
    > “SFT-7B … comparable to Google … SFT-14B even surpasses it” (Table 4; averages: 33.53 for SFT-7B, 34.47 for SFT-14B, vs. 32.81 for Google).
- Cost-effective large-scale RL (Appendix D; Table 8)
  - API cost drops to zero; compute cost is modest for SFT and amortizable across runs:
    > ~64k queries cost “$586.7” with Google vs. “$17.7–$70.8” for SFT-3B/7B/14B on AWS A100s (Table 8).

## 5. Experimental Analysis
- Evaluation design (Sections 4.1–4.3)
  - Datasets
    - Single-hop QA: `NQ`, `TriviaQA`, `PopQA`.
    - Multi-hop QA: `HotpotQA`, `2WikiMultiHopQA`, `Musique`, `Bamboogle`.
    - Training data for RL: merge NQ + HotpotQA to form a unified set (Section 4.3).
  - Metric
    - Exact Match (EM) for evaluation (Section 4.1). Note: training reward uses F1 (Section 3.5).
  - Baselines (Section 4.2)
    - Prompt-only: Direct Answer, Chain-of-Thought (CoT), RAG, RA-Agent, Search-o1.
    - RL: R1 (no search), Search-R1 (real search). All RL baselines use F1 reward for fairness (Section 4.2).
  - Fairness controls (Section 4.3)
    - At evaluation, every method retrieves via SerpAPI-Google, 5 documents fixed.

- Main quantitative results (Section 4.4; Table 3)
  - Across three backbone families and Base/Instruct variants, ZEROSEARCH attains the best or tied-best averages:
    - `Qwen-2.5-7B-Base`:  
      > Search-R1-base avg: “39.51”; ZEROSEARCH-base avg: “40.93” (Table 3).  
      With striking PopQA gain: “54.60 → 60.40”.
    - `Qwen-2.5-7B-Instruct`:  
      > Search-R1-inst avg: “38.17”; ZEROSEARCH-inst avg: “39.08”.
    - `Qwen-2.5-3B-Base`:  
      > Search-R1-base avg: “32.81”; ZEROSEARCH-base avg: “34.47”.
    - `LLaMA-3.2-3B-Base`:  
      > Search-R1-base avg: “34.21”; ZEROSEARCH-base avg: “36.07”.
  - Takeaway: Simulation-trained policies transfer to real Google at test time and often outperform policies trained with real search.

- Simulator choice ablation (Section 5.2; Table 4; Qwen-2.5-3B-Base as policy)
  - Prompted simulators < SFT simulators of same size.
  - SFT-7B ≈ Google; SFT-14B > Google at training time for the same policy, with real-Google evaluation:
    > Averages: “SFT-7B 33.53”, “SFT-14B 34.47”, “Google 32.81”.

- RL algorithm ablation (Section 5.4; Table 5)
  - All three work; REINFORCE is strongest and most stable here:
    > Averages: “REINFORCE 34.47”, “GRPO 33.17”, “PPO 32.67”.

- Curriculum vs. random noise (Section 5.5; Table 6)
  - Curriculum consistently wins:
    > On Qwen-2.5-3B-Base: “34.47 (Curriculum) vs. 32.59 (Random)”.

- Training dynamics (Section 5.3; Figure 2c)
  - Early: interaction turns drop as the policy learns the correct format; rewards increase slowly.
  - Later: both turns and rewards increase then stabilize, while difficulty continues to rise due to the curriculum.

- Stability and transfer curves (Section 5.1; Figures 2a–b and Appendix A, Figure 3)
  - ZEROSEARCH yields smoother reward curves than real-search RL and ultimately higher plateaus across base and instruct models.

- Cost analysis (Appendix D; Table 8)
  - > For ~64k queries over ~12h: “Google: $586.7” vs. “SFT-14B: $70.8” (GPU cost), all with “$0” API for the simulation route.

- Case studies (Appendix E; Tables 9–10)
  - Show multi-turn searches, consistent formatting, and clear separation between “useful” and “noisy” simulated documents.
  - Example snippet:  
    > For “Who is the spouse of the person who played the sergeant major in We Were Soldiers?” ZEROSEARCH searches the actor (Sam Elliott), then queries his spouse and answers “Katharine Ross” (Table 9).

Assessment of support
- The methodology-to-results chain is strong:
  - Controlled difficulty (curriculum) correlates with smoother learning and better EM (Figures 2–3; Table 6).
  - Masking external tokens gives both smoother training and better EM (Figure 4c; Table 7).
  - Larger, fine-tuned simulators yield stronger policies that transfer to real search (Table 4).
- Caveat: The simulator SFT prompt includes the ground-truth answer to ensure the simulator can create useful evidence (Section 3.3). While evaluation uses real Google, this design choice could make the simulated environment more informative than typical web search pages and may partially explain training efficiency.

## 6. Limitations and Trade-offs
- Assumptions and environment design
  - The simulator returns exactly five ~30-word documents per query and can be instructed to be “useful” or “noisy” (Table 2). Real web search varies greatly in format, length, and relevance.
  - An LLM judge labels training documents as useful/noisy (Section 3.3); mislabels could bias the simulator toward certain styles or spurious features.
  - The SFT prompt exposes ground-truth answers to the simulator (Section 3.3). This is practical for coverage, but it departs from real search and could make the simulator “too informative.”
- Scope of tasks
  - Benchmarks are QA-centric (Section 4.1). The framework does not address tasks needing deep browsing, form filling, code execution, or long-page reading beyond short snippets.
- Reward shaping
  - Only final-answer F1 is rewarded (Section 3.5). There is no signal for intermediate reasoning quality, tool-use efficiency, or format beyond the template—though the template worked well empirically (Table 9).
- Hyperparameter sensitivity
  - Curriculum hyperparameters (`p_s`, `p_e`) are set per model family (Appendix F). The paper does not deeply explore sensitivity or auto-tuning.
- Compute and infrastructure
  - Training requires GPU servers for the simulator and policy (Section 6; Appendix D). Although cheaper than APIs at scale, smaller labs still need access to decent hardware, and the simulator can be underutilized during policy updates (Appendix D).

## 7. Implications and Future Directions
- Field impact
  - Shows that LLMs can serve as controllable simulators for RL-with-tools, not just environments with fixed text corpora. This reduces cost and unlocks stable, curriculum-driven training environments for search and potentially other tools (code search, API calling, browsing).
- Practical applications
  - Building low-cost, robust research agents that can search the web; enterprise assistants that learn to consult internal knowledge bases with simulated retrieval during training; domains with limited or rate-limited APIs (e.g., medical literature search behind institutional gateways).
- Follow-up research
  - Richer simulators: multi-source, style-diverse, and page-structure-aware document generation (tables, figures, hyperlinks), possibly learned from scraped SERP/page distributions without including ground-truth answers in the simulator prompt.
  - Adaptive curricula: noise that responds to policy competence, with adversarial or contrastive evidence generation to harden retrieval and verification.
  - Multi-skill simulation: combining search with browsing, citation verification, and tool-use planning; adding intermediate rewards for reasoning quality and tool efficiency.
  - Generalization tests: time-sensitive queries, domain shifts, multimodal retrieval (images/tables), and tasks beyond QA.
  - Efficiency: sharing one simulator across multiple RL jobs (Appendix D), asynchronous rollouts, and distilled simulators for smaller footprints.

> Core takeaway: ZEROSEARCH turns the challenge of unstable, costly real-world search into a controllable, curriculum-driven RL environment. With a sufficiently capable simulator (7B–14B) and careful training design (structured prompts, F1 rewards, token masking), policies trained entirely in simulation transfer to real Google search and often outperform policies trained with real search engines (Tables 3–4; Figures 2–3).
