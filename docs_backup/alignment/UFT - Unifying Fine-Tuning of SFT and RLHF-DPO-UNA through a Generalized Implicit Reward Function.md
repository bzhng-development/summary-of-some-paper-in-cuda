# UFT: Unifying Fine‑Tuning of SFT and RLHF/DPO/UNA through a Generalized Implicit Reward Function

**ArXiv:** [2410.21438](https://arxiv.org/abs/2410.21438)
**Authors:** Zhichao Wang, Bin Bi, Zixu Zhu, Xiangbo Mao, Jun Wang, Shiyu Wang
**Institutions:** 

## 🎯 Pitch

This paper introduces a unified framework for preference learning in large language models, bridging the gap between reinforcement learning and supervised finetuning by revealing their shared gradient structure. By decoupling algorithms from data collection settings, it redefines the landscape into four modular components—Model, Data, Feedback, and Algorithm—enabling more flexible, safe, and efficient alignment methods that adapt to varied constraints, paving the way for innovative hybrid approaches.

---

## 1. Executive Summary (2-3 sentences)
This survey proposes a unified framework for preference learning in large language models (LLMs) that decomposes the field into four interacting elements—Model, Data, Feedback, and Algorithm—plus Evaluation as a downstream component (Figure 5; Section 3). It bridges the long-standing divide between reinforcement learning (RLHF-style) and supervised finetuning (e.g., DPO-style) by showing they share a common gradient form (Eq. 1) and by decoupling algorithm choice from online/offline data collection, providing a taxonomy (Figure 2), running examples (Figures 3–4), and practical guidance (Algorithm 1).

## 2. Context and Motivation
- Problem/gap addressed
  - Preference alignment—making an LLM’s outputs match human values and choices—is now a necessary step before deployment, yet methods sprawl across RL-based and SFT-based camps with different data pipelines, feedback sources, and objectives (Section 1).
  - Prior categorizations split the field into “RLHF vs. SFT/DPO,” which obscures shared structure and creates artificial barriers for method transfer, comparison, and hybridization (Section 1; discussion before Figure 2).
- Why it matters
  - Practical: Alignment guards against unsafe, toxic, or misleading outputs and improves helpfulness with relatively small amounts of data (Section 1; Figure 1’s example).
  - Scientific: A unified view clarifies how optimization signals flow from feedback to gradients and reveals how data collection (on-policy/off-policy) and algorithms can be recombined (Section 3; Eq. 1; Algorithm 1).
- Prior approaches and their limits
  - RLHF-style: requires learned reward models and online rollout, typically via PPO; strong but compute-intensive and sample-inefficient, and training can be unstable (Section 6.1; Eq. 6).
  - SFT-style (e.g., DPO/IPO): learns directly from pairwise preferences offline; efficient and simple but can overfit to annotations and sometimes depresses likelihood of “chosen” responses (Sections 6.2; Eq. 7–8; critiques in [3], [30], [147]).
  - On-policy vs. off-policy confusion: “online” is often conflated with “RL-based,” and “offline” with “SFT-based,” which is not necessary (Section 3; Appendix A).
- How this paper positions itself
  - Offers a single gradient-based lens that covers both RL-based and SFT-based methods (Eq. 1) and a process view that separates data collection from optimization (Algorithm 1).
  - Organizes the space by four elements—Data, Feedback, Algorithm, Evaluation—rather than by RL vs. SFT or online vs. offline (Figure 5; Figure 2).

## 3. Technical Approach
This is a methodological survey that builds a unified conceptual and operational framework rather than a new training algorithm. The key parts:

1) Definition of preference learning (Section 2)
- Goal: Given a human preference distribution `P(x, y)` over prompts `x` and responses `y`, train a policy `π_θ'` such that its outputs align better, i.e., `P(x, y_{θ'}(x)) > P(x, y_{θ}(x))`.
- Intuition: The environment encodes preferences; “better” outputs receive higher feedback (labels, ranks, or rewards).

2) A single gradient view that unifies RL and SFT (Section 3; Eq. 1)
- Plain-language summary: Any alignment method updates model parameters by weighting token-level log-likelihood gradients with a “gradient coefficient” that encodes preference information derived from feedback.
- In notation (Eq. 1): the update is an expectation over data pairs `(q, o)` of token-level terms `δ_A(r, q, o, t) ∇θ log πθ(o_t | q, o_<t)`. The algorithm `A`, the type of feedback `r`, and the data source determine `δ_A`.
- Why this matters: The gradient coefficient `δ_A` is the common currency through which a reward, a preference label, or a listwise rank actually steers learning. This makes RLHF and DPO variations comparable in a single mathematical frame.

3) Decoupling algorithm from online/offline data collection (Section 3; Algorithm 1; Appendix A)
- Definitions:
  - `on-policy` (online) data: gathered from the current model during training and scored in real time by an “environment” (e.g., a reward model, a judge).
  - `off-policy` (offline) data: prepared ahead of training from any source (humans, stronger LLMs, earlier checkpoints).
- Crucial stance:
  > “We do not use online/offline nor RL/SFT as criteria for classifying algorithms. Instead, we decouple the algorithm from the online/offline setting.” (Section 3; before Figure 5)
- Algorithm 1 (Section 3) is the generic loop:
  - If a reference model is needed, initialize it (e.g., DPO).
  - For online: sample responses from `π_θ` and obtain feedback from the environment `E` on the fly.
  - For offline: read batches from a stored dataset with labels or scores.
  - Update `π_θ` using the chosen algorithm `A` on triplets (Data, Feedback, Model[, Reference]).
  - Iterate.

4) Four-element taxonomy with concrete options (Figures 2 and 5; Sections 4–7)
- Data (Section 4)
  - On-policy sampling: Top-k/Nucleus sampling; beam search; Monte Carlo Tree Search (MCTS) for multi-step reasoning (Section 4.1).
  - Off-policy sources:
    - Human: WebGPT, OpenAI Human Preferences, HH-RLHF, SHP (Section 4.2).
    - LLM-generated: RLAIF, Open-Hermes-Preferences, UltraFeedback, UltraChat (Section 4.2).
- Feedback (Section 5; Figure 6)
  - Direct feedback (no learned reward): human labels; hand-crafted rules such as final-answer correctness (math), unit tests (code), QE scores (translation), human edits (summarization) (Section 5.1).
  - Model-based feedback:
    - Reward models:
      - Pairwise Bradley–Terry preference modeling (Eq. 2–3).
      - Binary classifiers when correctness is objective (Eq. 4).
      - Enhancements: better data via AI feedback and BoN variants; ensembles; fine-grained and process rewards; training stabilizers (Section 5.2.1).
    - Pair-wise scoring models (lightweight discriminators such as PairRanker) (Section 5.2.2).
    - LLM-as-a-Judge (prompted grading, self-rewarding, meta-judging, mixture of judges; task-specific critics/verifiers) (Section 5.2.3).
- Algorithms (Section 6; Table 1; Figures 3–4)
  - Point-wise: rejection sampling + SFT (Eq. 5), PPO (Eq. 6), ReMax (baseline-subtracted REINFORCE), KTO (prospect-theoretic objective with single-label preferences) (Section 6.1).
  - Pair-wise contrasts: CoH, SLiC, DPO (Eq. 7–8), IPO (bounded margin), f-DPO (general f-divergences), EXO (reverse-KL form), DPO-positive (penalizes likelihood collapse), ORPO (single-stage odds-ratio), SimPO (reference-free), online variants (sDPO, TR-DPO) and domain/multi-objective adaptations (Section 6.2).
  - List-wise contrasts: RRHF, PRO, calibration- and reweighting-based listwise methods, and GRPO (reference-free PPO variant with group-normalized rewards) (Section 6.3).
  - Training-free alignment: input optimization (prompt rewriting, norm retrieval, in-context restyling) and output optimization (paraphrasing modules, logits control, rewindable decoding, reward-guided decoding, ICDPO) (Section 6.4).
- Evaluation (Section 7)
  - Rule-based: accuracy/EM/ROUGE on tasks with ground truth (Section 7.1).
  - LLM-based: pairwise comparison, single-answer grading, and reference-guided scoring; evaluators (GPT-4 or fine-tuned open models); biases and meta-evaluation suites (Section 7.2).

5) Visual, step-by-step exemplars (Figures 3–4)
- The paper walks through data collection, feedback generation, and model updates for PPO, DPO (online/offline), RFT, RAFT, and ReMax to concretize how the pieces plug together.

## 4. Key Insights and Innovations
- A single gradient-coefficient lens (Eq. 1) that unifies RLHF and SFT-style methods
  - What’s different: Rather than contrasting objectives (reward maximization vs. preference classification), both are cast as token-level likelihood gradients scaled by a feedback-derived coefficient `δ_A` (Section 3).
  - Why it’s significant: It explains “how” preferences shape learning across algorithms, enabling principled comparisons and hybrids (e.g., online DPO, reference-free listwise PPO like GRPO).
- Decoupling algorithmic choice from online/offline data collection
  - What’s different: Online/offline is treated as a property of data and feedback availability, not of the optimization algorithm (Section 3; Appendix A).
  - Why it’s significant: DPO can be run online if a real-time evaluator exists (Section 3), and PPO can be run with offline batches if rewards are precomputed, expanding feasible design space.
- A practical, end-to-end process view (Algorithm 1; Figure 5) plus a comprehensive taxonomy (Figure 2)
  - What’s different: The field is reorganized by four elements—Data, Feedback, Algorithm, Evaluation—rather than by legacy categories.
  - Why it’s significant: It makes design choices modular and composable, helping practitioners plug-and-play data sources (e.g., MCTS rollouts), feedback types (e.g., PRMs vs. rules), and algorithms (e.g., SimPO vs. PPO) for their constraints.
- Clarifications that resolve common confusions
  - Online vs. offline: explicitly discussed as a definitional ambiguity in preference learning (Appendix A).
  - ReMax vs. GRPO: why they can be viewed as pairwise vs. listwise when thinking in terms of baseline computation for the gradient coefficient, even though optimization is point-wise once advantages are computed (Appendix B).
- Curated, mechanism-focused treatment of feedback and evaluation
  - Feedback: distinguishes direct, reward-model, pairwise scorer, and LLM-judge routes with examples and known pitfalls (Section 5).
  - Evaluation: systematically catalogs methods and known biases such as position and verbosity effects (Section 7.2.3), with meta-evaluation benchmarks.

Overall, the unification and decoupling are fundamental innovations; the taxonomy and running examples are high-value, integrative contributions that substantially lower the barrier to entry.

## 5. Experimental Analysis
This is a survey and does not introduce new empirical results. Instead, it systematizes how experiments are conducted in preference learning and what evidence practitioners typically rely on.

- Evaluation methodology landscape (Section 7)
  - Rule-based evaluation (Section 7.1): used when tasks have ground truth; common metrics include Accuracy, F1, Exact Match, ROUGE. Benchmark families are listed for factual knowledge (e.g., MMLU), math (GSM8K, MATH), reasoning (BBH), QA (TriviaQA, NQ), and coding (MBPP, HumanEval; repo-level SWE-Bench, ML-Bench).
  - LLM-based evaluation (Section 7.2):
    - Pairwise comparison: evaluators choose between two responses; aligns well with humans but scales quadratically with number of models (Section 7.2.1). AlpacaEval and its length-controlled variant are highlighted.
    - Single-answer grading: scalar scores per response; efficient but may miss subtle differences and can be noisy (Section 7.2.1).
    - Reference-guided grading: required for objective tasks (math, translation) when references exist (Section 7.2.1).
    - Evaluator models: priority models (e.g., GPT-4) vs. fine-tuned smaller open models; trade-offs in cost, reproducibility, and generalization (Section 7.2.2).
    - Known biases/limitations: position bias, verbosity preference, similarity bias, and difficulty with domains like math/reasoning; addressed via meta-evaluation suites such as FairEval, MT-Bench, LLMEval, and LLMBar’s adversarial set (Section 7.2.3).
- Algorithmic exemplars and losses (Figures 3–4; Table 1; Section 6)
  - Table 1 compiles exact loss forms for popular pairwise/listwise algorithms (e.g., DPO, IPO, f-DPO, ORPO, SimPO, RRHF, PRO), enabling apples-to-apples comparisons of objectives.
  - Figures 3–4 provide end-to-end schematics for PPO, DPO (online/offline), RFT, ReMax, and RAFT—covering data sampling, feedback computation (e.g., reward, advantage), and update steps.
- Does the evidence support the paper’s claims?
  - The claim is conceptual: many algorithms fit into the unified gradient and four-element process. The survey supports this with explicit equations (Eq. 1, 5–8), algorithm schemas (Algorithm 1), and loss tables (Table 1). It also shows, with concrete examples (Figures 3–4), how different pipelines instantiate the same template.
  - No new quantitative comparisons are made; rather, the paper cites prior works’ reported strengths/weaknesses (e.g., PPO instability and compute (Section 6.1), DPO overfitting (Section 6.2)).
- Ablations/failure cases/robustness
  - Not applicable as a primary contribution. However, Section 7.2.3 consolidates evaluator failure modes, and Section 5.2.1 catalogues reward model failure mitigations (ensembles, regularization, fine-grained/process rewards).

## 6. Limitations and Trade-offs
- Assumptions in the framework
  - Treats “feedback” broadly as any signal that can shape `δ_A` in Eq. 1 (labels, rewards, ranks). This is unifying but abstracts away feedback quality, calibration, and adversarial behavior (Section 3).
  - Assumes that mapping a method into the gradient-coefficient form is sufficiently informative for design and comparison; nuances like exploration strategies or credit assignment may still differ materially across methods.
- Scope limitations
  - Focused on textual preference alignment; does not cover hallucination mitigation, multi-modal alignment, or instruction tuning per se (Section 2).
  - The dataset list is representative, not exhaustive, reflecting the rapidly growing space (Section 4.2).
- Open definitional disputes
  - What counts as “online” can vary: some argue that using a fixed reward model is, in effect, offline with respect to the true environment (Appendix A). The survey adopts a pragmatic definition—if feedback is generated in real time, it is “online.”
- Algorithm-specific trade-offs (from collated evidence)
  - PPO: powerful but compute- and memory-heavy; sample-inefficient and unstable to tune (Section 6.1).
  - DPO and variants: simple and effective but can overfit pairwise labels, depress likelihood of chosen responses, or limit diversity with implicit forward-KL dynamics; many variants patch these issues at the cost of extra complexity or new hyperparameters (Section 6.2; Table 1).
  - Reward models: prone to overoptimization and uncertainty; mitigation via ensembles and regularizers adds training overhead and infrastructure complexity (Section 5.2.1).
  - LLM-as-a-Judge: scalable oversight but introduces biases (position, verbosity, similarity) and struggles on hard reasoning unless supplemented (Section 5.2.3; Section 7.2.3).
- Missing quantitative synthesis
  - The survey does not provide meta-analysis across methods or standardized re-implementation results; readers must rely on cited papers for empirical comparisons.

## 7. Implications and Future Directions
- Field-level impact
  - Normalizes thinking in terms of Data–Feedback–Algorithm–Evaluation, making it easier to compose pipelines that match constraints (e.g., combine on-policy MCTS sampling with pairwise scoring and SimPO).
  - Encourages cross-pollination: e.g., running DPO online, using listwise-normalized advantages (GRPO-style) with SFT-style objectives, or employing process rewards in offline SFT pipelines.
- Research directions explicitly highlighted (Section 8)
  - Better preference data: diverse, scalable synthetic preference generation with high fidelity (e.g., West-of-N, RLAIF-style pipelines) and advanced sampling like MCTS for richer trajectories.
  - Reliable and scalable feedback: expand domains where direct, verifiable feedback exists (compilers, theorem provers) and develop scalable oversight methods (self-rewarding, recursive reward modeling, weak-to-strong).
  - Algorithmic advances: methods that approach upper bounds set by data/feedback while being robust to noise and efficient at scale (e.g., reference-free preference optimization, stable online SFT, listwise-normalized objectives).
  - Evaluation: reduce evaluator bias and cost; build comprehensive, diverse meta-evaluation suites that better reflect open-ended alignment quality.
- Practical applications and use
  - Builders can use Algorithm 1 and Figures 3–4 to instantiate end-to-end pipelines tuned to their constraints (e.g., limited labels: use off-policy data + pairwise scoring + SimPO; verifiable tasks: direct rules + rejection sampling fine-tuning).
  - Safety and governance teams can mix LLM-judge systems with calibrated, reference-guided checks for high-stakes domains, while tracking known biases (Section 7.2.3).
  - Research groups can prototype hybrids that the unified view suggests—for example, online SimPO with listwise candidate normalization, or process-level PRMs plugged into SFT objectives.

> Core takeaway (Section 3; Figure 5): alignment methods differ mainly in how they produce and transform feedback into the gradient coefficient `δ_A` and in how they obtain data (on-policy/off-policy). Once this is clear, RLHF and SFT approaches become interoperable modules rather than opposing camps.
