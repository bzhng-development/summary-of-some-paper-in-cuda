# Mutual Reasoning Makes Smaller LLMs Stronger Problem-Solvers

**ArXiv:** [2408.06195](https://arxiv.org/abs/2408.06195)
**Authors:** Zhenting Qi, Mingyuan Ma, Jiahang Xu, Li Lyna Zhang, Fan Yang, Mao Yang
**Institutions:** Microsoft Research Asia

## 🎯 Pitch

rStar introduces a groundbreaking "self-play mutual reasoning" approach that transforms small language models (SLMs) into significantly improved problem solvers without requiring fine-tuning or stronger teacher models. By integrating enhanced human-like search with Monte Carlo Tree Search and peer verification, rStar showcases substantial accuracy gains in math and commonsense benchmarks, notably boosting affordable and accessible AI solutions for cost-sensitive applications.

---

## 1. Executive Summary (2-3 sentences)
This paper proposes rStar, a “self‑play mutual reasoning” procedure that turns small language models (`SLMs`, e.g., 3–8B parameters) into much stronger problem solvers at inference time—without fine‑tuning or a stronger teacher model. rStar couples a richer, human‑like search with Monte Carlo Tree Search (MCTS) and a second, similarly capable SLM acting as a verifier; together they generate and mutually validate multi‑step reasoning, yielding large gains on math and commonsense benchmarks (e.g., GSM8K jumps from 12.51% to 63.91% for `LLaMA2-7B`; Table 2).

## 2. Context and Motivation
- Problem addressed
  - Smaller models are attractive for cost and latency but are notably worse at multi‑step reasoning than larger models. Even with standard prompting like Chain‑of‑Thought (`CoT`), 7–8B models struggle (e.g., Mistral‑7B reaches only 36.5% accuracy on GSM8K with few‑shot CoT; Section 1).
  - Improving reasoning without a stronger, supervising LLM (no fine‑tuned value model, no GPT‑4 teacher) remains hard.

- Why it matters
  - Practical: Many organizations deploy SLMs for cost/privacy; better reasoning extends their usefulness to math and logic tasks without expensive retraining.
  - Scientific: Tests whether inference‑time search and peer verification, rather than more parameters or supervised data, can unlock latent reasoning ability.

- Prior approaches and gaps
  - Single‑round prompting (CoT, few‑shot) improves transparency but often underperforms on complex reasoning (Sections 1–2).
  - Multi‑round sampling (Self‑Consistency, SC) improves over single traces by majority vote, but requires that many sampled traces are correct; SLMs rarely meet that condition (Section 3.1).
  - Tree search methods (e.g., ToT, RAP) often use a narrow action space (one way to extend a trace), limiting exploration; reward signals can be unreliable when SLMs self‑evaluate (Appendix A.1 shows near‑random self‑evaluation in RAP when its r1 is randomized, Table 6).
  - Training separate value/reward models can overfit and needs labeled data (Section 2).

- Positioning
  - rStar expands the search space with five “human‑like” reasoning actions and replaces self‑judgment with mutual verification by a second, equally capable SLM (“mutual consistency”). It thereby tackles both exploration (better candidate solutions) and selection (more reliable identification of good ones) without external supervision (Fig. 2, Section 3).

## 3. Technical Approach
rStar is a two‑stage, inference‑time procedure: (1) generate multiple candidate reasoning trajectories with a richer MCTS, and (2) verify them via mutual consistency using a second SLM.

Key terms used below:
- `SLM`: a small language model (≈3–8B parameters).
- `Trajectory`: a full reasoning path from the original problem to a final answer, composed of intermediate steps `s1, s2, …, sd` (Section 3.1).
- `Action space`: the set of available moves for extending a trajectory during search.
- `Mutual consistency`: a verification scheme where a second SLM, given partial steps from a trajectory, completes the reasoning; if it reaches the same answer, the trajectory is considered validated (Section 3.3).

Step-by-step:

A) Generate candidate trajectories with richer MCTS (Section 3.2, Fig. 3)
- Core idea
  - Use MCTS to grow a tree from the question (root) through intermediate steps to terminal nodes (complete solutions). Each edge corresponds to choosing an action that prompts the SLM to produce the next step.
- Rich, human‑like action space (five actions; Section 3.2)
  - `A1` One‑step thought: propose exactly the next reasoning step.
  - `A2` Remaining steps: “fast think” to complete all remaining steps directly (standard CoT‑style continuation).
  - `A3` Next sub‑question + answer: decompose the problem into a smaller sub‑question, then answer it (least‑to‑most prompting).
  - `A4` Re‑answer sub‑question: if a sub‑question may be wrong or brittle, answer it again with few‑shot CoT to improve reliability.
  - `A5` Rephrase the question: rewrite the problem into explicit conditions to reduce misunderstandings.
  - Some actions have ordering constraints (e.g., `A4` can only follow `A3`; `A5` applies only to the root; Section 3.2).
  - Why this design: Single‑action search (e.g., only decompose or only step‑forward) often gets stuck; the five actions mimic human flexibility—decompose when helpful, otherwise compute directly, revisit mistakes, or clarify the statement (Section 3.2, Fig. 3).
  - Evidence: An ablation on GSM8K (200 samples) shows accuracy increases as more actions are enabled: from 70.5% with only `A3` (RAP‑like) to 75.0% with all five (Table 1).

- Reward function tailored to SLMs (Section 3.2)
  - Challenge: SLM self‑evaluation of intermediate steps is unreliable.
  - Design:
    - Initialize `Q(s, a) = 0` for unexplored nodes.
    - When a terminal node is reached, compute its reward `Q(sd, ad)` as the confidence from self‑consistency majority voting (i.e., the likelihood the final answer is correct across sampled completions).
    - Back‑propagate this terminal reward to every node along the path: `Q(si, ai) ← Q(si, ai) + Q(sd, ad)` for i=1..d−1.
  - Why this design: It rewards actions by their empirical contribution to correct final answers (AlphaGo‑style credit assignment), avoiding direct self‑judgment on intermediate steps (Section 3.2).
  - Node selection uses UCT (Upper Confidence Bound for Trees):
    - Equation: UCT(s, a) = Q(s,a)/N(s,a) + c * sqrt(ln N_parent(s) / N(s,a)) (Section 3.2).
    - Interpretation: Prefer actions with high average reward but still explore less‑tried actions.

- Rollout details (Section 4.1)
  - 32 rollouts per problem; max depth `d=5` for most datasets, `d=8` for MATH.
  - Branching: up to 5 children per depth for `A1` and `A3`, 1 for others.
  - Output: a set of candidate trajectories (and their rewards/confidences).

B) Verify trajectories with mutual consistency (Section 3.3, Fig. 2 and Fig. 4)
- Problem: Picking the best single trajectory based only on MCTS reward is hard; many SLM‑generated traces are partially wrong.
- Mechanism:
  - Introduce a second SLM `M̂` (similar capability) as a discriminator.
  - For a candidate trajectory `t = x ⊕ s1 ⊕ … ⊕ sd`, pick a random split point `i < d`.
  - Provide `M̂` with the question and the prefix `x ⊕ s1 ⊕ … ⊕ si−1` as “partial hints,” and ask it to complete the remaining reasoning (Section 3.3, Fig. 4).
  - If `M̂`’s completed answer matches the original trajectory’s answer, label `t` as “validated.”
- Why partial hints: They reduce difficulty and variance, increasing the chance that `M̂` can correctly finish the reasoning and thus provide informative feedback (Section 3.3).
- Final selection: Among validated trajectories, choose the one with the highest product of (i) the MCTS terminal reward and (ii) the terminal confidence from rollouts (Section 3.3).

C) Implementation specifics (Section 4.1)
- Models: five SLMs—`Phi3-mini (3.8B)`, `LLaMA2‑7B`, `Mistral‑7B`, `LLaMA3‑8B`, `LLaMA3‑8B‑Instruct`.
- Discriminator: `Phi3-mini-4k` by default (3.8B), run in parallel for efficiency; when `Phi3` is the generator, it self‑discriminates (Section 4.1).
- Discriminator hinting: random split between 20% and 80% of the steps are given as prefix (Section 4.1).

Analogy: Think of two students solving a problem—one explores multiple solution outlines using different tactics (break down, compute directly, rephrase, retry a subpart), while the other, shown partial work, tries to finish the solution. If both independently reach the same answer, confidence increases (Section 3.3’s “peer verification” rationale).

## 4. Key Insights and Innovations
- Rich action space for reasoning search (fundamental)
  - What’s new: Five complementary actions (`A1–A5`, Section 3.2) replace the typical single action (e.g., only decomposing or only stepping).
  - Why it matters: Better exploration generates higher‑quality candidates. Table 1 shows monotonic gains as actions are added (70.5% → 75.0% on GSM8K subsample).

- Mutual consistency verification with a peer SLM (fundamental)
  - What’s new: Instead of self‑verification or majority voting across random samples, rStar cross‑checks a candidate with a second SLM that receives partial hints and must independently complete to the same answer (Section 3.3, Fig. 4).
  - Why it matters: It provides supervision‑free yet informative feedback that is more robust than self‑judgment and avoids training reward models (Table 5 left shows consistent gains over majority voting and self‑verification).

- AlphaGo‑style credit assignment without intermediate self‑grading (incremental but impactful)
  - What’s new: Rewards flow from successful terminals back to earlier steps; intermediate self‑evaluation is avoided (Section 3.2).
  - Why it matters: Appendix A.1 (Table 6) suggests SLMs’ self‑ratings can be near random; rStar’s design sidesteps this pitfall.

- Strong gains without a stronger teacher model (pragmatic innovation)
  - What’s new: Both generator and discriminator are SLMs; no GPT‑4 teacher is required (though it can be used; Table 5 right).
  - Why it matters: Makes the approach broadly usable in constrained settings.

## 5. Experimental Analysis
- Evaluation setup (Section 4.1)
  - Datasets: four math word‑problem datasets—`GSM8K`, `GSM‑Hard`, `SVAMP`, `MATH‑500`—and one commonsense dataset `StrategyQA`.
  - Metrics: accuracy (percent of correctly answered questions).
  - Baselines (Sections 4.1–4.2): Zero-shot and few-shot CoT; Self‑Consistency (SC@8/64/128); tree‑search methods `ToT` and `RAP`.
  - Rollouts: 32 per question; depth `d=5` (most) and `d=8` (MATH‑500).

- Main quantitative results (Tables 2–3)
  - Across models and tasks, rStar substantially improves over all baselines.
  - GSM8K (Table 2):
    - `LLaMA2‑7B`: few‑shot CoT 12.51% → rStar 63.91% (“+51.4 points”).
    - `Mistral‑7B`: 36.46% → 81.88%.
    - `LLaMA3‑8B‑Instruct`: 74.53% → 91.13%.
    - Quote: “rStar boosts GSM8K accuracy from 12.51% to 63.91% for LLaMA2‑7B, from 36.46% to 81.88% for Mistral‑7B, from 74.53% to 91.13% for LLaMA3‑8B‑Instruct” (Abstract; detailed in Table 2).
  - GSM‑Hard (Table 2):
    - `Mistral‑7B`: 13.57% (few‑shot CoT) → 37.91%.
    - `LLaMA3‑8B‑Instruct`: 25.63% → 37.53%.
  - SVAMP (Table 2):
    - `LLaMA3‑8B`: 76.90% (few‑shot) → 90.00%.
    - `LLaMA2‑7B`: 48.10% → 74.90%.
  - StrategyQA (Table 2):
    - Modest but consistent gains: e.g., `LLaMA3‑8B`: 64.05% (few‑shot) → 67.69%.
  - MATH‑500 (Table 3):
    - `LLaMA3‑8B‑Instruct`: rStar 42.94% vs. best baseline SC@128 at 33.80% (+9.14 points).
    - `Phi3‑mini‑4k`: rStar 48.60% vs. SC@128 at 45.60%.

- Generator vs. discriminator contributions
  - Generator alone (majority voting) is strong: on GSM8K, `LLaMA3‑8B‑Instruct` improves to 88.70%—already better than ToT and RAP (Table 2).
  - Adding the discriminator further lifts accuracy: e.g., `LLaMA3‑8B‑Instruct` 88.70% → 91.13% (Table 2).
  - Discriminator robustness:
    - Against trajectories from different generators, rStar’s discriminator outperforms majority voting and self‑verification (Table 5, left).
    - Using different models as the discriminator barely changes accuracy; `GPT‑4 (2024‑05‑01)` gives 92.57% vs. `Phi3‑Mini‑Instruct` 91.13% on GSM8K (Table 5, right).

- Ablations and diagnostics
  - Action space ablation: Adding actions systematically improves accuracy (Table 1).
  - Generator comparison: rStar’s generator outperforms RAP and SC, both with majority voting and with rStar’s discriminator (Table 4).
  - Self‑evaluation vs. rStar reward: Adding self‑evaluation (Ours+Self‑eval) reduces performance compared with rStar’s back‑prop reward (Table 4).
  - Sensitivity to rollouts: rStar improves accuracy with as few as 2 rollouts and keeps improving with more; RAP saturates or declines after 4 in some settings (Fig. 5).
  - Self‑rewarding unreliability: Randomizing RAP’s intermediate self‑score `r1` barely changes results; randomizing the terminal confidence `r2` hurts (Appendix A.1, Table 6).

- Efficiency and cost
  - Inference‑time overhead is substantial: on GSM8K, average ≈ 149–167 model calls and ≈ 349k–367k generated tokens per question for `Mistral‑7B` and `LLaMA2‑7B` (Appendix A.2, Table 7).
  - Computation scales linearly with the number of rollouts; verification can be parallelized (Section 4.1; Appendix A.2).

- Overall assessment
  - The experiments are comprehensive: five models, five datasets, strong baselines, and extensive ablations. The gains are large in math tasks and consistent in commonsense. The evidence convincingly supports the claims that (i) richer search improves candidate quality, and (ii) mutual consistency selects better solutions without a teacher.

## 6. Limitations and Trade-offs
- Compute overhead at inference time
  - Many rollouts and long traces: ≈150–170 calls and ≈350k tokens per problem (Table 7). This may be too expensive for real‑time or large‑scale deployments without batching/parallelization.

- Agreement ≠ correctness
  - Mutual consistency can validate wrong answers when both SLMs follow the same flawed hint or bias. While results show strong net gains, the method does not guarantee correctness (Section 3.3 rationale; Table 5 shows strong but not perfect verification).

- Domain and tool use
  - The approach assumes reasoning can be expressed in natural language steps. Tasks requiring external tools (symbolic solvers, retrieval, calculators) are not integrated here. The method also avoids trained reward models or external supervision, which may cap performance on some domains (Sections 2–3).

- Hyperparameters and design choices
  - Performance depends on depth `d`, the mix of actions, and rollout count (Section 4.1; Fig. 5). Tuning may be dataset‑specific.

- Intermediate reward signal
  - Terminal confidence comes from self‑consistency on the final answer. If majority voting is weak (few correct completions), reward estimates may be noisy—though the second‑stage discriminator mitigates this (Section 3.2; Table 5).

## 7. Implications and Future Directions
- How this changes the landscape
  - rStar demonstrates that SLMs already contain latent reasoning ability that can be unlocked by better search and peer verification, challenging the notion that dramatic reasoning leaps require bigger models or supervised fine‑tuning (Fig. 1; Table 2).

- Practical applications
  - Math tutoring, automated grading, data cleaning with logic constraints, operations planning, and lightweight on‑device assistants that need stronger reasoning without server‑side large models.

- Follow‑up research
  - Smarter, cheaper search:
    - Adaptive rollouts based on early confidence; learned action priors to prune the tree; caching and reusing sub‑traces.
  - Stronger verification:
    - Multi‑peer or committee verification; integrating external tools for step‑checking; hybrid symbolic‑neural validators.
  - Task‑aware actions:
    - Extending the action set for domains like code reasoning, scientific QA, or spatial/diagram problems.
  - Semi‑supervised refinement:
    - Use validated trajectories as training data for lightweight fine‑tuning, potentially closing the loop between inference‑time gains and model updates—without a stronger teacher.
  - Safety and robustness:
    - Diagnose when mutual agreement is misleading; introduce adversarial peers to reduce confirmation bias.

In sum, rStar’s combination of a diversified action space for MCTS and mutual peer verification offers a compelling, supervision‑free path to boost SLM reasoning. The approach trades inference cost for accuracy and reliability, but it opens clear avenues to make small models far more capable where compute at test time is acceptable.
