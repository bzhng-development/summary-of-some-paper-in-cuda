# Search-R1: Training LLMs to Reason and Leverage Search Engines with Reinforcement Learning

**ArXiv:** [2503.09516](https://arxiv.org/abs/2503.09516)

## 🎯 Pitch

SEARCH-R1 pioneers a new way to train large language models to reason step-by-step while deciding how and when to use real-time web search, using reinforcement learning instead of static retrieval or simple prompting. By interleaving autonomous search actions and internal reasoning, and stabilizing RL with a novel retrieved-token loss masking trick, SEARCH-R1 delivers substantial gains on knowledge-intensive tasks. This approach overcomes the limits of previous retrieval-augmented strategies, enabling LLMs to provide more accurate, up-to-date answers essential for reliable AI in fast-evolving or long-tail domains.

---

## 1. Executive Summary
SEARCH-R1 trains a language model to interleave its own step‑by‑step reasoning with live web search using reinforcement learning (RL), rather than relying only on prompting or static retrieval. It delivers sizable accuracy gains on seven question‑answering benchmarks by learning when and how to search, while keeping training stable through a simple outcome reward and a key stabilization trick called retrieved‑token loss masking.

## 2. Context and Motivation
- Problem the work targets
  - Large language models (LLMs) often need up‑to‑date or long‑tail facts to reason correctly, but their pretraining knowledge can be stale or incomplete. Existing retrieval methods either (a) run a one‑shot retrieval step before generation (standard Retrieval‑Augmented Generation, RAG), or (b) prompt the model to “use tools” like a search engine during inference.
- Why this matters
  - Accurate use of external knowledge is central to reliable assistants, enterprise search, scientific question answering, and other knowledge‑intensive tasks. Simply scaling parametric knowledge is insufficient for freshness and coverage.
- Gaps in prior approaches
  - RAG typically retrieves once using the user question as the query; the retrieved passages may be suboptimal for multi‑step reasoning, leading to irrelevant or insufficient context (Section 2.1).
  - Prompted tool‑use (e.g., ReAct/IRCoT) requires the model to already know how to search effectively; generalization is limited, and no learning occurs during deployment (Section 2.1).
  - Supervised tool‑use training needs high‑quality labeled trajectories (search queries and step‑by‑step reasoning), which are expensive at scale (Section 2.1).
  - Search is non‑differentiable, so end‑to‑end gradient learning is not straightforward.
  - RL has recently proven effective for improving reasoning (e.g., DeepSeek‑R1), but how to integrate real search, maintain stability, and design rewards is unclear (Section 1).
- How this work positions itself
  - SEARCH-R1 brings RL to the “search‑while‑reasoning” setting and addresses three core challenges: (1) an RL framework that treats search as part of the environment with stable optimization, (2) multi‑turn, interleaved reasoning and search, and (3) simple but effective outcome‑only rewards (Section 1).

## 3. Technical Approach
SEARCH-R1 turns “search as a tool” into a learnable behavior inside an RL training loop. The system alternates between generating tokens and—when the model decides—calling a real search engine, then continues reasoning with the retrieved text.

- High‑level loop (Section 3.2; Algorithm 1; Table 1)
  1. The LLM receives a question wrapped in a minimal instruction template that enforces structure:
     - Think inside `<think>...</think>`.
     - If knowledge is missing, call search by generating `<search> query </search>`.
     - The system responds by injecting `<information> retrieved_passages </information>`.
     - Repeat “think → optionally search” any number of times.
     - When ready, output the final prediction inside `<answer>...</answer>`.
  2. This continues until an action budget is reached or the answer tag appears.
  3. The retrieved content becomes part of the next decoding context, so reasoning is conditioned on real‑time search results.

- RL objective with search in the environment
  - Classical RL for LLMs optimizes the expected reward with a KL penalty to keep the policy close to a reference model. SEARCH-R1 extends this to include the search engine `R` as part of the environment:
    - Equation (1)/(6) defines the goal: maximize `E[rϕ(x, y)] − β D_KL(πθ(·|x;R) || πref(·|x;R))`, where:
      - `πθ` is the trainable “policy LLM,”
      - `πref` is a frozen “reference LLM,”
      - `y` is the whole trajectory containing both generated tokens and search insertions,
      - `R` indicates that generation is interleaved with retrieval (Section 3.1; Appendix A).
  - Why this matters: the policy is explicitly trained on trajectories that include real search calls and retrieved text, so the model learns to decide when to search, what to search for, and how to use results.

- Stabilization via retrieved‑token loss masking (Section 3.1; Table 4; Figure 3)
  - During training, token‑level losses (policy gradients and KL regularization) are computed only on LLM‑generated tokens. Tokens copied from `<information>...</information>` are masked out and do not contribute to the loss.
  - Intuition: gradients should not encourage the model to “imitate” the retrieved passages, which are exogenous to the model. Masking prevents spurious learning signals and improves stability.

- Two compatible RL optimizers (Section 3.1)
  - PPO (Proximal Policy Optimization): an actor‑critic method with a “clipped” objective (Equation (2)). Advantages are estimated with GAE (Generalized Advantage Estimation).
  - GRPO (Group Relative Policy Optimization): samples a group of responses for the same prompt and uses the group’s average reward as a baseline, avoiding a learned critic (Equation (3)).
  - Both use the same rollout mechanism and the same masking of retrieved tokens (including inside the KL term for GRPO).

- Reward function (Section 3.4)
  - Outcome‑only reward: at the end of a trajectory, extract the answer from `<answer>...</answer>` and compute a rule‑based correctness signal such as exact‑match (EM) with the gold answer (Equation (4)).
  - No process rewards and no learned reward models are used. This keeps the system simple and avoids reward‑model brittleness.

- Experimental pipeline (Section 4.3; Appendix B)
  - Base models: `Qwen2.5-3B` and `Qwen2.5-7B` (both Base and Instruct variants).
  - Retrieval: 2018 Wikipedia dump; `E5` dense retriever; top‑k = 3 by default (Appendix B).
  - Training set: merged NQ + HotpotQA training data; evaluation on 7 datasets (Section 4.1).
  - Default RL: PPO for 500 steps; batch size 512; sequence length 4096; max response 500 tokens; action budget B=4; training on 8×H100 GPUs (Appendix B).

## 4. Key Insights and Innovations
- Treating search as part of the RL environment with interleaved generation (Sections 3.1–3.2)
  - Novelty: the trajectory `y` explicitly includes search calls and retrieved text; the policy is optimized on such trajectories, not just on pure text generations.
  - Significance: the model learns “when to search” and “what to query” as latent skills, improving generalization beyond handcrafted prompts.

- Retrieved‑token loss masking (Section 3.1; Table 4; Figure 3)
  - Novelty: a simple but critical masking trick that excludes retrieved passage tokens from gradient updates and KL computations.
  - Significance: stabilizes RL and meaningfully boosts accuracy. For `Qwen2.5‑7B‑base` with PPO, average EM increases from 0.343 without masking to 0.431 with masking (Table 4).

- Minimal, outcome‑only rewards are enough (Section 3.4)
  - Novelty: avoids complex process supervision or neural reward models; relies solely on final correctness measured by EM.
  - Significance: simplifies training and still yields large gains across diverse QA tasks (Tables 2–3), showing that RL can teach effective search behavior using only final answers as feedback.

- Multi‑turn, structured reasoning with explicit control tokens (Section 3.2; Table 1; Algorithm 1)
  - Novelty: a lightweight protocol based on four tags—`<think>`, `<search>`, `<information>`, `<answer>`—enables iterative reasoning, retrieval, and termination.
  - Significance: supports decomposition, self‑verification, and flexible numbers of search calls during training and inference (case studies in Appendix J).

- Empirical insights on RL choices and dynamics (Section 5)
  - PPO vs. GRPO: GRPO converges faster but can become unstable; PPO is steadier and often achieves the best final performance (Figure 2a; Table 3).
  - Base vs. Instruct: Instruct models learn faster initially, but base and instruct reach similar end performance after RL (Figure 2b; Appendix E).
  - Response‑length dynamics: responses first shorten (less filler), then lengthen as the model learns to search more and include retrieved evidence (Figure 2c–d).

## 5. Experimental Analysis
- Evaluation setup (Sections 4.1–4.3)
  - Datasets (7 total):
    - General QA: Natural Questions (NQ), TriviaQA, PopQA.
    - Multi‑hop QA: HotpotQA, 2WikiMultiHopQA, Musique, Bamboogle.
  - Metric: Exact Match (EM)—whether the predicted answer text matches a gold string (Section 4.3).
  - Baselines (Section 4.2):
    - No retrieval: Direct inference; Chain‑of‑Thought (CoT).
    - With retrieval or tool use: RAG; IRCoT; Search‑o1.
    - Training‑based: SFT; RL reasoning without search (R1); rejection sampling fine‑tuning with search rollouts.
  - Fairness controls: same retriever, same number of retrieved documents (k=3), same corpora, same pre‑trained LLMs (Section 4.2; Appendix B).

- Main results (Table 2; Qwen2.5‑7B and 3B)
  - `Qwen2.5‑7B‑base`:
    - Average EM: 
      - RAG = 0.304, R1‑base = 0.276, Rejection Sampling = 0.348.
      - SEARCH‑R1‑base = 0.431.
    - Example per‑dataset gains:
      - On NQ: 0.480 (SEARCH‑R1‑base) vs. 0.349 (RAG).
      - On HotpotQA (multi‑hop): 0.433 vs. 0.299 (RAG).
  - `Qwen2.5‑3B‑instruct`:
    - Average EM: 0.325 (SEARCH‑R1‑instruct) vs. 0.270 (RAG) and 0.224–0.229 (R1).
  - Takeaway: consistent improvements across both in‑domain (NQ, HotpotQA) and out‑of‑domain datasets (TriviaQA, PopQA, 2Wiki, Musique, Bamboogle). The abstract highlights average relative improvements of 24% (7B) and 20% (3B), which align with Table 2.

- PPO vs. GRPO (Table 3; Figure 2a)
  - For `Qwen2.5‑7B‑base`, average EM:
    - PPO: 0.431; GRPO: 0.350.
  - For `Qwen2.5‑3B‑instruct`, average EM:
    - PPO: 0.325; GRPO: 0.336.
  - Convergence behavior: GRPO often rises quicker but shows reward collapse later; PPO stays stable (Figure 2a; Appendix F).

- Ablations and robustness checks
  - Retrieved‑token masking (Table 4; Figure 3):
    - `Qwen2.5‑7B‑base` PPO mean EM rises from 0.343 (no mask) to 0.431 (mask).
  - Number of retrieved passages (Appendix G; Figure 6; Table 7):
    - Top‑k = 3 yields the best final average EM (0.431) vs. k=1 (0.375) and k=5 (0.400).
    - Interpretation offered: k=1 hurts recall; k=5 introduces noise that can degrade both learning and inference.
  - GRPO group size (Appendix H; Figure 7; Table 8):
    - Group size = 1 (REINFORCE) generalizes best on average (0.410) compared to size 3 (0.363) and 5 (0.350), despite slower convergence.
  - Training dynamics (Section 5.3; Figure 2c–d):
    - As steps progress, valid search calls increase, response length first decreases then increases as the model starts to incorporate more retrieved content.
  - Case studies (Appendix J):
    - Successes show multi‑turn querying and self‑verification (Tables 10, 12, 13, 15, 18, 19).
    - Failures reveal query‑writing mistakes and susceptibility to misleading retrieval (Tables 11, 14, 16, 20).

- Do the experiments support the claims?
  - Yes, the breadth of datasets, consistent gain over strong RAG/tool‑use baselines, and multiple ablations (masking, optimizer, top‑k, group size) substantiate that learning to search through RL is both feasible and beneficial under the presented setup. The method remains competitive across model sizes and both base/instruct variants.

## 6. Limitations and Trade-offs
- Dependence on retrieval quality and corpus coverage
  - Experiments use the 2018 Wikipedia dump and an E5 retriever with k=3 by default (Appendix B). Different domains, non‑Wikipedia sources, or noisy web search could change outcomes.
- Reward design is task‑specific
  - Outcome reward is exact‑match of a short answer string (Section 3.4). This is ideal for factual QA but less suited to tasks with long, free‑form answers, multiple valid phrasings, or subjective judgments.
- Compute and engineering costs
  - Training requires multi‑GPU infrastructure (8×H100), an online rollout server (vLLM), and repeated search calls (Appendix B). This is heavier than standard SFT/RAG pipelines.
- Stability vs. speed trade‑off
  - GRPO trains faster but can collapse; PPO is more stable but slower (Figure 2a; Appendix F). Tuning is nontrivial.
- Protocol reliance
  - The approach depends on special tokens (`<think>`, `<search>`, `<information>`, `<answer>`) and an action budget B (Algorithm 1). Porting to other tool APIs or multimodal settings requires re‑engineering the protocol and the parser.
- Query decomposition remains fragile
  - Failure cases show that the model can write poor queries or be misled by irrelevant passages (Appendix J; Tables 14, 16, 20), indicating open challenges in query planning and evidence assessment.

## 7. Implications and Future Directions
- Field impact
  - SEARCH-R1 demonstrates that outcome‑only RL can teach LLMs non‑differentiable skills like querying a search engine in a multi‑turn manner—without curated process supervision. This bridges the gap between “prompted tool use” and “learned tool use,” suggesting a viable path to train generalist agents that coordinate reasoning with tools.
- Research opportunities
  - Reward design: move beyond EM to semantic equivalence, citation faithfulness, or uncertainty‑aware rewards; explore process rewards that penalize spurious searches.
  - Retrieval strategy: dynamic top‑k, query planning, query reformulation, and reranking; learning to decide when to stop searching.
  - Multi‑tool/Multimodal: extend the same RL‑with‑environment abstraction to calculators, APIs, and vision tools; explore cross‑modal retrieval and reasoning.
  - Off‑policy data and scaling: combine SEARCH-R1 with logged trajectories (e.g., human or synthetic) and larger models (evidence in Appendix C shows consistent gains at 14B).
- Practical applications
  - Enterprise question answering with evolving knowledge bases; research assistants that cite up‑to‑date sources; fact‑checking and report generation with live references; customer support and analytics that require current policies or inventory data.

> Key quantitative takeaways:
> - On `Qwen2.5‑7B‑base`, SEARCH‑R1 reaches average EM 0.431 vs. 0.304 (RAG) and 0.276 (R1 without search) across seven datasets (Table 2).
> - Retrieved‑token masking is crucial: 0.431 (with mask) vs. 0.343 (without) on the same setup (Table 4).
> - PPO is more stable than GRPO (Figure 2a), and top‑k=3 retrieval balances recall and precision best (Table 7).

Overall, SEARCH-R1 is a practical and conceptually clear framework for training LLMs to search while they think, delivering measured improvements and offering a foundation for broader tool‑augmented reasoning with RL.
