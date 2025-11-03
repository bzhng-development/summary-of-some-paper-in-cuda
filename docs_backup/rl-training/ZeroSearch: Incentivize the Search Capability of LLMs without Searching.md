# ZeroSearch: Incentivize the Search Capability of LLMs without Searching

**ArXiv:** [2505.04588](https://arxiv.org/abs/2505.04588)
**Authors:** Hao Sun, Zile Qiao, Jiayan Guo, Xuanbo Fan, Yingyan Hou, Yong Jiang, Pengjun Xie, Fei Huang, Yan Zhang
**Institutions:** Tongyi Lab, Alibaba Group

## 🎯 Pitch

ZEROSEARCH revolutionizes how language models learn to use search engines by employing a simulation approach with controlled document quality, avoiding real search API costs and instability. This framework enhances accessibility and scalability of RL-based search, enabling more stable training and outperforming real-search-trained models, marking a significant leap in efficient, large-scale LLM training.

---

## 1. Executive Summary (2-3 sentences)
ZEROSEARCH is a reinforcement-learning (RL) framework that teaches a language model to use a search engine without ever calling a real search API during training. It replaces live search with a fine-tuned “simulation LLM” that generates search-like documents whose quality can be precisely controlled, and it trains the policy with a curriculum that gradually increases retrieval difficulty. This eliminates API cost, stabilizes training, and in extensive evaluations (Table 3) matches or surpasses models trained with a real web search engine.

## 2. Context and Motivation
- Problem addressed
  - LLMs need up-to-date, external information but their knowledge is static, which leads to hallucinations and outdated answers (Section 1).
  - RL-based “searcher” models trained against live search engines face two key obstacles (Section 1):
    - Uncontrolled document quality: search returns are noisy and unstable for training.
    - Prohibitive API costs: hundreds of thousands of rollouts make training expensive and rate-limited.
- Why it matters
  - Practical: cost controls training scale. Table 8 estimates ~64k queries cost $586.7 via Google’s SerpAPI vs $0 API cost when using a simulated engine, with only $17.7–$70.8 of GPU cost depending on simulator size.
  - Scientific: a controllable, scalable environment helps study tool-use policies (how and when to search) without confounds from an unpredictable external service.
- Prior approaches and gaps
  - Retrieval-Augmented Generation (`RAG`) improves grounding but often needs complex prompts or heavy test-time scaling (Section 2.1).
  - RL for search has used static corpora (e.g., Wikipedia) or live search engines (Search-R1, DeepResearcher, WebThinker; Section 2.2). Static corpora lack real-world complexity; live engines are noisy and expensive.
- Positioning
  - ZEROSEARCH trains “to use a real search engine with simulated searches during training” (Section 1). It aims to preserve the benefits of RL-based search policies while removing live-API dependency and adding explicit control over document quality (Sections 3.3–3.4).

## 3. Technical Approach
The framework replaces the external search engine with a `simulation LLM` during RL training and restores a real engine only at evaluation time.

- RL objective without a real search engine (Section 3.1)
  - Intuition: treat the simulation LLM as the environment. The policy LLM produces a multi-turn trajectory y (think → search → answer) conditioned on the simulation engine’s returned documents.
  - Formal objective: maximize expected reward while keeping the policy close to a reference model via a KL penalty:
    - max over policy `πθ`: E[rϕ(x, y)] − β D_KL(πθ(y|x; πψ) || πref(y|x; πψ))
    - `πψ` is the frozen simulation LLM; `πref` is a fixed reference policy; `rϕ` is a rule-based reward (Section 3.5).
  - Practical consequence: training stability improves by comparing the policy to a known reference, and the simulator provides the retrieval context.

- Multi-turn training template (Table 1; Section 3.2)
  - The policy answers in three stages using explicit tags:
    - `<think>`: internal reasoning after each new piece of information.
    - `<search>query</search>`: when the model decides it lacks knowledge; this triggers the simulator to return “web-like” snippets inside `<information> ... </information>`.
    - `<answer>`: final concise answer.
  - Why: the explicit separation of reasoning/search/answer enforces structured decision-making and makes tool use transparent.

- Building the simulation LLM (“Search Simulation Tuning”; Section 3.3; Table 2)
  - Data creation pipeline:
    1. Collect real interaction traces by prompting an LLM to use a real web search engine until it answers a question.
    2. Extract query–document pairs from these traces.
    3. Use an LLM judge to label whether each document contains enough information to answer the query (“useful”) or not (“noisy”).
  - Light SFT (supervised fine-tuning):
    - Fine-tune a small-to-mid-size LLM (3B/7B/14B) to generate five short “documents” per query, with document quality controlled by a single word in the prompt: “[useful / noisy]” (Table 2).
    - During SFT, the prompt also includes the question and its ground-truth answer to expand the simulator’s coverage (Table 2). This is only for training the simulator—not used during RL rollout.
  - Result: a simulator that can be prompted to produce either helpful or misleading retrieval snippets on demand.

- Curriculum-based rollout with controlled noise (Section 3.4; Eq. (1))
  - At RL training time, each `<search>` call is routed to the simulator, which returns five documents.
  - A noise schedule controls the fraction of “noisy” vs “useful” documents over training steps:
    - pi = ps + ((i/m − 1)/(b − 1)) × (pe − ps) with default base b = 4, where i is the current step and m the total steps.
    - `ps` = initial noise probability; `pe` = final noise probability.
  - Why: start easy (mostly useful documents) so the policy learns the format and basic tool use; then increase difficulty (more noise) to force robust reasoning.

- Reward design (Section 3.5)
  - Problem with exact match (EM) as training reward: the policy tends to “hack” the reward by outputting long answers to increase the chance of overlap.
  - Solution: use token-level F1 between prediction and ground truth:
    - r(x, y) = 2 × IN / (PN + RN)
    - `IN`: overlap count; `PN`: predicted length; `RN`: ground-truth length.
  - Evaluation still uses EM (Section 4.1), but training uses F1 to avoid reward hacking.

- Loss masking for simulator tokens (Section 3.6; Appendix C; Figure 4c; Table 7)
  - Only the policy’s own generated tokens receive gradients. The externally generated simulator tokens are masked out to prevent instability.
  - Effect: measurable performance gain and smoother training (Table 7 shows Avg. 36.07 with masking vs 34.53 without; Figure 4c shows better reward curve with masking).

- Algorithms and implementation (Sections 3.6, 4.3, F)
  - RL optimizers: `REINFORCE`, `PPO`, `GRPO` are all supported.
  - Hardware: simulation server on 4× H20 GPUs; RL on another 4× H20 GPUs (Appendix F).
  - Default: REINFORCE; default simulator: fine-tuned `Qwen-2.5-14B-Instruct`.

- How a single training episode works (simplified)
  - Input question arrives with the template.
  - Policy writes `<think>`, decides to `<search>who wrote …</search>`.
  - Simulator receives the query and, based on the current noise probability, returns five short snippets inside `<information> … </information>`.
  - Policy reads them, thinks, may search again, then writes `<answer> … </answer>`.
  - Reward computed by F1; gradients computed only on policy tokens; KL penalty to reference stabilizes learning (Figure 1 diagrams PPO/GRPO variant).

## 4. Key Insights and Innovations
- LLM-as-search-engine simulator with controllable quality
  - Novelty: Instead of interacting with live search, the environment is a fine-tuned LLM that can emit realistic “search results” and whose quality is switched by a prompt keyword (Table 2).
  - Why it matters: enables zero API cost, scalable throughput (can scale with GPUs), and systematic control of document noise—two pain points in prior RL-with-search work (Sections 1, 3.3).
- Curriculum rollout that gradually increases retrieval difficulty
  - Different from prior constant-noise or uncontrolled-web settings, the noise schedule (Eq. 1) makes training start easy and become harder (Section 3.4).
  - Impact: smoother and higher rewards vs real-search training; Figures 2a–b show ZEROSEARCH initially trails but then surpasses Search-R1 with fewer fluctuations; Figure 3 shows similar trends across other backbones.
- Simple, robust reward design plus loss masking
  - Using F1 (not EM) as reward curbs reward hacking (Section 3.5). Masking simulator tokens prevents gradient noise (Section 3.6).
  - Significance: improves stability and final accuracy (Table 7: +1.5 Avg EM with masking).
- Demonstrated portability and competitiveness
  - Works with base and instruction-tuned models from 3B to 7B across Qwen and LLaMA families, and with RL algorithms REINFORCE/GRPO/PPO (Tables 3, 5).
  - In head-to-head comparisons, the simulated-engine–trained policies match or outperform real-search–trained ones (Table 3).

## 5. Experimental Analysis
- Evaluation setup (Sections 4.1–4.3)
  - Datasets
    - Single-hop QA: NQ, TriviaQA, PopQA.
    - Multi-hop QA: HotpotQA, 2WikiMultiHopQA, Musique, Bamboogle.
  - Metric
    - EM (exact match) for reporting; training reward uses F1 (Section 4.1).
  - Baselines
    - Prompting: Direct Answer, CoT, RAG.
    - Iterative RAG agents: `RA-Agent`, `Search-o1`.
    - RL: `R1` (no search), `Search-R1` (RL with real search engine).
  - Controls for fairness
    - All methods retrieve a fixed five documents per search (Section 4.3).
    - At evaluation, every model uses the same real web search via SerpAPI/Google (Section 4.3). This isolates the effect of training regime.

- Main results (Table 3)
  - Qwen-2.5-7B-Base
    - ZEROSEARCH-base: Avg EM 40.93.
    - Search-R1-base (real search during training): 39.51.
    - Per-dataset highlights: TriviaQA 66.40 vs 61.40; PopQA 60.40 vs 54.60; HotpotQA 32.00 vs 31.20.
  - Qwen-2.5-7B-Instruct
    - ZEROSEARCH-inst: 39.08 vs Search-R1-inst 38.17; better on HotpotQA (34.60 vs 32.80).
  - Qwen-2.5-3B-Base
    - ZEROSEARCH-base: 34.47 vs Search-R1-base 32.81; gains on NQ (43.00 vs 40.60) and HotpotQA (33.80 vs 29.20).
  - LLaMA-3.2-3B-Base
    - ZEROSEARCH-base: 36.07 vs Search-R1-base 34.21; big gains on NQ (43.40 vs 41.20) and HotpotQA (32.20 vs 29.60).
  - Takeaway
    - ZEROSEARCH consistently matches or beats the strongest real-search RL baseline across families/sizes.

- Reward dynamics and behavior analyses
  - Reward curves (Figures 2a–b and 3)
    - ZEROSEARCH shows smoother increases and eventually exceeds Search-R1, consistent with controlled difficulty and reduced environment noise.
  - Interaction turns (Figure 2c)
    - Early in training, turns drop (policy learns formatting and to avoid redundant searches) while reward rises slowly; later, both turns and reward increase, then stabilize as the policy learns to retrieve and reason effectively even under higher noise.

- Simulator choice ablation (Table 4; Section 5.2)
  - With Qwen-2.5-3B-Base as the policy:
    - Prompted simulators (no SFT) underperform SFT simulators.
    - `SFT-7B` simulator matches Google (Avg 33.53 vs 32.81); `SFT-14B` surpasses Google (34.47).
  - Insight: light SFT substantially narrows the style gap and improves “useful/noisy” control fidelity.

- RL algorithm ablation (Table 5; Section 5.4)
  - Using Qwen-2.5-3B-Base: REINFORCE performs best (Avg 34.47), slightly ahead of GRPO (33.17) and PPO (32.67).
  - All three work, supporting framework generality.

- Curriculum vs random rollout (Table 6; Section 5.5)
  - Curriculum improves Avg EM:
    - Qwen-2.5-3B-Base: 34.47 vs 32.59.
    - LLaMA-3.2-3B-Base: 36.07 vs 34.84.

- Loss masking ablation (Figure 4c; Table 7)
  - With masking: Avg 36.07 vs 34.53 without; reward curves also more stable.

- Cost analysis (Table 8; Appendix D)
  - > “Google ~64,000 queries … API cost $586.7.”
  - > “SFT-14B … GPU cost $70.8” with zero API cost.
  - Practical note: simulator GPU utilization is bursty; sharing one simulator across multiple RL jobs can improve cost-efficiency.

- Qualitative evidence (Appendix E, Table 9 and Table 10)
  - Interaction examples show structured behavior (think → search → multi-turn → answer) and correct final answers.
  - Useful vs noisy simulator outputs are clearly distinguishable, and useful docs tend to contain the correct answer.

- Do the experiments support the claims?
  - Yes, on three dimensions:
    - Accuracy: ZEROSEARCH generally outperforms real-search RL (Table 3).
    - Stability and learning dynamics: smoother curves and eventual higher training reward (Figures 2–4).
    - Cost and scalability: orders-of-magnitude reduction in API cost with the ability to scale simulator throughput via GPUs (Table 8).

## 6. Limitations and Trade-offs
- Dependence on simulator quality and initial SFT
  - The simulator requires light SFT on labeled useful/noisy pairs (Section 3.3). While far cheaper than full RL with search, it still needs an initial collection of search traces and an LLM judge for labeling—an extra pipeline step whose scale is not fully quantified.
- Fidelity gap with the open web
  - Even after SFT, simulated outputs may not capture real-world web diversity (freshness, layout quirks, domain biases). The curriculum helps, but true web dynamics (rate limits, inconsistent snippet quality, long-tail sites) are not modeled.
- Reward focuses only on answer accuracy
  - The rule-based F1 reward ignores citation quality, transparency, and safety (Section 3.5)—it neither rewards verifiable quotes nor penalizes unsupported claims.
- Compute and infrastructure
  - Requires GPU servers for the simulator (Appendix D). Although cheaper than APIs at scale, small teams without GPUs may find this a barrier. GPU utilization is uneven during RL (rollout heavy, update light).
- Hyperparameter sensitivity
  - Noise schedule parameters (`ps`, `pe`) are model-specific (Appendix F). Tuning may be needed for new backbones or domains.
- Evaluation metric mismatch
  - Training uses F1 while reporting EM; this is by design to reduce reward hacking (Section 3.5), but introduces a train–test metric shift.

## 7. Implications and Future Directions
- Field impact
  - Demonstrates a practical path to train tool-using LLMs via RL without live APIs. The “LLM-as-environment” paradigm—with controllable difficulty—can generalize to other tools beyond search (e.g., database querying, code execution sandboxes, specialized APIs).
- Follow-up research enabled
  - Better simulators
    - Domain-specific simulators (biomedical, legal) with stronger fidelity; explicit modeling of ranking, snippet attribution, and freshness.
    - Adversarial simulators that stress-test policies (e.g., plausible but subtly misleading snippets).
  - Richer rewards
    - Learned reward models that value attribution, calibration, and safety; multi-objective rewards that jointly optimize correctness and justified evidence.
  - Scaling and algorithms
    - Larger policies, off-policy RL or actor–critic variants, and integration with test-time search planning (e.g., MCTS) to combine policy learning with search-time tree expansion.
  - Sim2Real transfer studies
    - Formal analyses of how simulator noise schedules and labeling quality influence real-web generalization; automatic schedule tuning.
- Practical applications
  - Enterprise RAG systems that need low-cost continual training and predictable behavior.
  - Agents that perform “deep research” with minimal API cost for education, competitive intelligence, and investigative journalism.
  - Privacy- or compliance-sensitive deployments where external calls are restricted; simulation training can be conducted entirely on in-house hardware.

> Key takeaways anchored in the paper:
> - “ZEROSEARCH … incentivizes the capabilities of LLMs to use a real search engine with simulated searches during training” (Section 1), avoiding unstable document quality and high API cost.
> - “A 7B retrieval module achieves comparable performance to the real search engine, while a 14B retrieval module even surpasses it” (Section 1; detailed in Table 4).
> - “ZEROSEARCH consistently outperforms baseline methods” across datasets and model families (Section 4.4; Table 3), with smoother and higher reward trajectories than real-search RL (Figures 2–3).
