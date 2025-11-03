# RewardBench: Evaluating Reward Models for Language Modeling

**ArXiv:** [2403.13787](https://arxiv.org/abs/2403.13787)
**Authors:** Nathan Lambert, Valentina Pyatkin, Jacob Morrison, LJ Miranda, Bill Yuchen Lin, Khyathi Chandu, Nouha Dziri, Sachin Kumar, Tom Zick, Yejin Choi, Noah A. Smith, Hannaneh Hajishirzi
**Institutions:** Allen Institute for AI, University of Washington, University of Southern California, University of Washington (Yejin Choi), University of Washington (Hannaneh Hajishirzi)

## 🎯 Pitch

RewardBench offers the first comprehensive framework to evaluate reward models (RMs) critical in RLHF for language models, providing a standardized benchmark covering diverse tasks like safety and reasoning. By revealing systematic strengths and weaknesses in RMs, it enables targeted improvements, impacting safer and more effective AI deployments, and paving the way for refined RLHF outcomes.

---

## 1. Executive Summary
RewardBench is a benchmark, dataset, and codebase for evaluating reward models (RMs)—the components that turn human preferences into learning signals during RLHF for language models. It fills a long-standing gap by offering standardized, carefully curated, and task-diverse tests (chat, adversarial instruction following, safety, and reasoning) plus a public leaderboard of 80+ models, enabling rigorous comparison across classifier-style RMs and implicit RMs trained with Direct Preference Optimization (DPO).

The benchmark reveals systematic strengths and weaknesses across model families, including where DPO-trained models generalize poorly and how different RMs balance refusal vs. compliance on safety prompts. This matters because RMs directly shape the behavior and values of RLHF-tuned systems.

## 2. Context and Motivation
- Problem addressed
  - There is no widely accepted way to evaluate reward models, despite their central role in RLHF. Most public discussion and evaluation target policies (the chatbots) rather than the RMs that guide them during training (Section 1).
  - Existing validation sets used in prior RM work (e.g., Anthropic HH, OpenAI Summarize) have low accuracy ceilings (≈60–70%) due to annotator disagreement and are not designed for out-of-distribution challenges (Section 1; citing Wang et al., 2024).

- Why it matters
  - RMs determine what a model is rewarded for; they encode the “values” that RLHF optimizes toward. Understanding RM behavior is thus central to safety, instruction following, and reasoning quality in deployed systems (Abstract; Section 1).
  - Open-source RM training resources and tests are sparse; researchers and practitioners lack reliable tools to choose or improve RMs.

- Prior approaches and gaps
  - Policy-side evaluations such as AlpacaEval, MT-Bench, and Chatbot Arena measure chatbot quality using LLM-as-a-judge, not RM fidelity (Section 2).
  - Existing preference test sets are noisy, multi-turn, or not designed to stress RMs with subtle, verifiable comparisons (Sections 1, 5.3).
  - New preference datasets (e.g., UltraFeedback, Nectar) do not include dedicated held-out test sets for RM evaluation (Section 1).

- Positioning
  - RewardBench provides a unified evaluation framework and dataset targeted specifically at RMs, with carefully curated prompt–chosen–rejected trios that allow objective, verifiable scoring (Sections 4–4.2; Figure 1). It also compares explicit classifier RMs with implicit DPO RMs and generative LLM-as-a-judge models on the same tasks (Sections 5, B; Tables 2–4, 7–8).

## 3. Technical Approach
RewardBench evaluates whether an RM can assign higher “reward” to a verified-better answer than to a verified-worse answer for the same prompt.

- What a reward model is
  - An RM predicts which of two responses is preferred for a given prompt. Classifier-style RMs are trained with pairwise preference data using a logistic model of comparisons (Bradley–Terry) (Section 3).
  - The modeled probability that y1 is preferred over y2 given prompt x is
    - Intuition: if the RM’s internal score r(x, y) is higher, y should be more preferred.
    - Formalization (Eq. 1): p(y1 ≻ y2 | x) = exp(r(x,y1)) / [exp(r(x,y1)) + exp(r(x,y2))].

- Implicit reward via DPO
  - DPO avoids explicitly training a separate RM; it defines a reward from the policy’s log-probabilities relative to a reference model (Section 3).
  - Reward for response y: r(x,y) = β log[π(y|x)/π_ref(y|x)] + β log Z(x) (Eq. 2). The comparison between y1 and y2 reduces to comparing their log-probability ratios (no Z(x) needed when comparing).

- RewardBench dataset design (Section 4.1; Table 1)
  - All items are single-turn prompt–chosen–rejected trios, manually verified where needed.
  - Sections and intent:
    - Chat: standard instruction-following where quality differences are large (AlpacaEval; MT-Bench; 358 prompts).
    - Chat Hard: subtle or adversarial instruction-following differences, including “near-miss” prompts and GPT-4-crafted traps (LLMBar, MT-Bench hard; 456 prompts).
    - Safety: mix of should-refuse (dangerous/offensive) and should-respond prompts that include trigger words (XSTest, Do-Not-Answer, AI2 refusal data; 740 prompts).
    - Reasoning: precise correctness in math and code; reference solutions vs. minimally buggy solutions (PRM-Math; HumanEvalPack in six languages; 1,431 prompts).
    - Prior Sets: legacy preference test sets (Anthropic HH helpful/harmless/HHH; SHP; OpenAI Summarize; 17.2k items) used for backward comparability, but down-weighted for the final score due to noise (Section 4.1–4.2).

- Scoring protocol (Section 4.2; Figure 1)
  - Each RM scores prompt+chosen and prompt+rejected independently.
  - A “win” occurs if r(x, y_chosen) > r(x, y_rejected). Accuracy is the fraction of wins; 50% is random.
  - Aggregation:
    - Per section, accuracy is a weighted average across its subsets.
    - Reasoning increases PRM-Math weight so math and code contribute equally (Section 4.2).
    - Prior Sets are averaged but counted at half weight in the overall score “due to noise, lack of clearly defined tasks, etc.” (Section 4.2).
  - For DPO models, if the specified reference model is available, r(x, y) is computed via the log-probability ratio with the reference; if missing, performance degrades significantly (Table 7).

- Data curation choices that matter
  - To reduce length bias (RM tendency to reward longer outputs), some subsets pair models with similar answer lengths (e.g., AlpacaEval Length) and ensure the chosen answer is not simply longer (Appendix B; Appendix H.2; Table 10).
  - Chat Hard emphasizes minimal, verifiable differences (e.g., wrong subject or context, or near-duplicate prompts with a twist) that LLM-as-a-judge systems often mishandle (Section 4.1; LLMBar).

- Implementation and release
  - A common inference stack supports classifier RMs, DPO models, and generative judges; text–score pairs for all inputs are released for further analysis (Contributions list in Section 1; Code/Data links on the title page).

## 4. Key Insights and Innovations
- A dedicated, cross-domain RM benchmark with curated, verifiable contrasts
  - What’s new: a single framework comparing chat quality, adversarial instruction following, safety refusal/compliance balance, and token-level reasoning correctness (Sections 4–4.2; Table 1). Many contrasts are constructed so only subtle but decisive facts (e.g., a one-token code bug) separate chosen from rejected (Reasoning; Table 13).
  - Why it’s significant: RMs should score these cases near 100% if they truly capture preference-relevant quality; observed ceilings and variance reveal concrete failure modes (Sections 5.2, E.1).

- Systematic comparison of classifier RMs vs. DPO-trained models
  - What’s new: both are evaluated as reward scorers, not just as chat policies. RewardBench shows how DPO performs on preference tests and where it fails (Sections 5.1–5.3; Tables 2–4, 7, 14).
  - Why it’s significant:
    - DPO models often underperform on legacy preference test sets (Prior Sets) while doing better on modern chat and reasoning subsets (Table 14 vs. Tables 2–4).
    - If the required DPO reference model is absent, “reference-free” scoring collapses by −6 to −35 points depending on the model (Table 7), an operational insight for practitioners.

- Taxonomy of safety behavior measured directly on RMs
  - What’s new: separate measurement of “should-refuse” and “should-respond” behaviors uncovers three modes: balanced, over-refusal, and over-compliance (Section 5.2; Table 6).
  - Why it’s significant: RMs that indiscriminately reward refusals (or compliance) are undesirable for deployed systems; these diagnostics make the trade-off visible.

- Evidence that “hard” instruction following and fine-grained reasoning remain open
  - What’s new: adversarial instruction-following sets (LLMBar) and near-duplicate prompts expose frequent RM mistakes; even strong models can drop below random in some adversarial subsets (Section 5.2; Table 5).
  - Why it’s significant: it pinpoints where to improve data and RM training—exactly the kinds of cases that lead to LLM-as-a-judge failures and brittle RLHF outcomes.

## 5. Experimental Analysis
- Evaluation setup
  - Models: 80+ publicly available models spanning 0.4B to 70B parameters, including classifier RMs (e.g., `Starling-RM-34B`), DPO models (`tulu-2-dpo-*`, `zephyr-*`, Qwen-Chat), and generative judges (GPT-4, Gemini 1.5, Claude 3) (Sections 5, E; Tables 2–4, 8–9).
  - Metric: accuracy on pairwise “chosen vs. rejected” comparisons; 50% random baseline (Section 4.2; Figure 1).
  - Datasets: see Section 3 above (Section 4.1; Table 1). Total primary test size ≈2,958 prompts; plus Prior Sets (≈17.2k items) with half weight in the overall score (Section 4.1–4.2).

- Main quantitative results
  - Overall leaderboard (open models; Table 2)
    - Top classifier RM: `RLHFlow/ArmoRM-Llama3-8B-v0.1` scores 89.0 overall with standout Reasoning 97.3 and Chat 96.9; Chat Hard 76.8; Safety 92.2; Prior Sets 74.3.
    - Other high performers include `pair-preference-model-LLaMA3-8B` (85.7) and `FsfairX-LLaMA3-RM-v0.1` (83.6). DPO policies rank lower overall (e.g., `tulu-2-dpo-70b` at 76.1).
  - Scaling effects for DPO (Table 3)
    - `tulu-2-dpo` shows consistent gains with size: 7B (71.7) → 13B (73.4) → 70B (76.1).
    - Qwen-Chat shows weaker and non-monotonic scaling; e.g., 7B (68.7), 14B (69.8), 72B (68.2), likely reflecting distribution shift issues (Section 5.1 discussion after Table 4).
  - 7B-class comparison (Table 4)
    - `zephyr-7b-alpha` (73.4) vs. `zephyr-7b-beta` (71.8): filtering UltraFeedback for refusals reduces Safety performance in `beta`, illustrating dataset design trade-offs (Section 5.1).
    - Strong 7B classifier RMs (e.g., `Eurus-RM-7b` at 81.6) outperform many DPO 7B models on Chat Hard and Reasoning.
  - Safety behavior (Table 6)
    - Balanced, strong performers: `ArmoRM-Llama3-8B-v0.1` (92.2 Safety overall; 93.0 Dangerous; 97.0 Offensive; 100.0 Should Refuse; 87.2 Should Respond), `Starling-RM-34B` (88.2).
    - Over-refusal tendency: Qwen chat models have high “Should Refuse” but low “Should Respond” (e.g., `Qwen1.5-14B-Chat`: 80.5 vs. 41.6), signaling many false refusals.
    - Over-compliance: `UltraRM-13b` scores poorly on Dangerous/Offensive (18.0/21.0) but very high on “Should Respond” (94.8).
  - Chat Hard difficulty (Table 5)
    - Even top models struggle on adversarial LLMBar subsets. `ArmoRM-Llama3-8B-v0.1` averages 76.8 on Chat Hard but dips to 66–70 on the hardest adversarial sets.
    - Some DPO models do well on specific adversarial subsets (e.g., Qwen excels on Neighbor) but lag overall, underscoring specialization vs. generalization trade-offs.
  - Reasoning breadth and ceilings (Table 13)
    - Best accuracy reaches 97.3 (`ArmoRM-Llama3-8B-v0.1`), but the distribution across models is wide—from below random to near perfect—indicating substantial headroom for many RMs.
    - The reasoning sets rely on tiny, verifiable differences (e.g., a one-token bug), making them a strong test of fine-grained reward discrimination.
  - DPO “reference-free” penalty (Table 7)
    - Removing the required reference model for DPO scoring drops performance sharply (e.g., `Mixtral-8x7B-Instruct` 82.2 → 64.2; −18.0 overall; Safety −35.3), emphasizing the necessity of using the correct reference at inference.
  - Generative judges vs. classifier RMs (Table 8)
    - Best generative judge (`gemini-1.5-pro-0514`) reaches 88.1 overall (Chat 92.3; Chat Hard 80.6; Safety 87.5; Reason 92.0), close to the best classifier RM but still slightly behind the top RewardBench score of 89.0 (Table 2).
    - Many generative judges (and open-weight LLMs) lag substantially behind specialized classifier RMs on the adversarial and reasoning-heavy components.

- Do the experiments support the claims?
  - Yes, via:
    - Diverse, curated test suites with verifiable correctness (Sections 4.1; Appendix F).
    - Cross-family comparisons showing repeatable patterns (Tables 2–6, 13–14).
    - Diagnostics on DPO inference requirements (Table 7).
    - Distributional analyses showing subset difficulty and output score shapes (Appendix E.1–E.2; Figures 2–7).
  - Caveats:
    - The final correlation between RewardBench scores and downstream RL training success is not yet established (Conclusion; Appendix A).

- Failure cases and robustness checks
  - Many RMs fail on subtle adversarial instruction-following (LLMBar) or prefer refusals too often (Table 5–6).
  - The paper inspects length bias and constructs subsets to mitigate it (Appendix B, H.2; Table 10), though more statistical analysis is left for future work.

## 6. Limitations and Trade-offs
- Assumptions and scope
  - Single-turn evaluations dominate; multi-turn dynamics of RM scoring in RLHF are only exposed through Prior Sets (Anthropic Helpful is multi-turn) and not emphasized due to noise (Sections 4.1, 5.3).
  - Implicit assumption that higher pairwise accuracy on curated trios correlates with better RLHF outcomes; this correlation is not yet empirically established (Conclusion).

- Data and curation constraints
  - Much of the data is semi-automatically curated then manually verified, not annotated from scratch by humans; spurious correlations can remain, especially in reasoning formats (Appendix A).
  - Potential test-set contamination is possible for popular sources like AlpacaEval and MT-Bench (Appendix A).

- Metric and aggregation choices
  - Accuracy on pairwise comparisons does not measure calibration, reward scale, or the shape of reward distribution—factors known to affect RL training stability (Appendix E.2). Prior Sets are down-weighted (0.5×), reflecting their noisiness (Section 4.2).

- Computational considerations
  - Running the full suite across many large models is expensive (≈1000 A100 GPU hours for 75 models; Appendix C).
  - DPO evaluation requires the correct reference model; when unavailable, results can be misleading (Table 7; Appendix B).

- Safety data caveats
  - Safety subsets include harmful content in rejected answers; while necessary to test refusals, they create exposure risks and require care in handling (Appendix A; Section 4.1 Safety).

## 7. Implications and Future Directions
- How this changes the landscape
  - RewardBench provides the first broadly adopted, RM-focused yardstick, enabling apples-to-apples comparisons across classifier RMs, DPO policies used as implicit RMs, and generative judges. The leaderboard (link on title page) already surfaces non-obvious winners and weaknesses.
  - The adversarial and fine-grained subsets reveal where reward modeling needs targeted improvements: subtle instruction following, refusal calibration, and token-level correctness in reasoning.

- What research it enables
  - Studying links between RewardBench scores and RLHF outcomes: best-of-N sampling and PPO training guided by different RMs (mentioned as ongoing in the Conclusion).
  - Analyzing the role of RM output distributions (Appendix E.2) and regularization (e.g., KL β, epochs) on RL training stability (Appendix B).
  - Building better preference datasets for hard cases (Chat Hard, Reasoning) where current ceilings are well below 100% (Appendix E.1; Section 5.2).

- Practical applications
  - Selecting RMs for specific deployment goals:
    - Safety-sensitive systems can pick models with balanced refusal/response behavior (Table 6).
    - Code/math tutoring or code review assistants benefit from RMs that achieve near-perfect Reasoning discrimination (Table 13).
    - Evaluation teams can use RewardBench to audit and tune refusal policies in DPO pipelines, armed with evidence that reference-free inference undercuts performance (Table 7).
  - System-level integration: small RMs can serve as safety filters or preference gates around large generative models (Appendix B, “Safety In or After RLHF”).

> Representative takeaways with references:
> - “ArmoRM-Llama3-8B-v0.1 reaches 89.0 overall, with 97.3 on Reasoning and 92.2 on Safety” (Table 2).
> - “DPO models require the correct reference model at inference; removing it drops scores by 6–35 points” (Table 7).
> - “Adversarial instruction following remains hard; many models fall near or below random on specific LLMBar subsets” (Table 5; Section 5.2).
> - “Balanced refusal behavior is measurable and varied; some models over-refuse, others over-comply” (Table 6).

In sum, RewardBench reframes RM evaluation as a first-class problem, provides the tools and data to study it, and surfaces concrete, actionable failure modes. The most impactful next step is to tie these evaluation signals to downstream RLHF training outcomes, closing the loop between RM diagnostics and policy quality.
