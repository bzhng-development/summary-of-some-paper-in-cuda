# RewardBench: Evaluating Reward Models for Language Modeling

**ArXiv:** [2403.13787](https://arxiv.org/abs/2403.13787)

## 🎯 Pitch

RewardBench introduces the first comprehensive benchmark and open-source toolkit specifically for evaluating reward models—the critical yet understudied components that align language models to human values via RLHF. By systematically scoring and comparing both classifier-based and Direct Preference Optimization reward models across dimensions like chat quality, adversarial robustness, safety, and reasoning, RewardBench uncovers key strengths, weaknesses, and scaling behaviors, empowering both the research community and practitioners to build safer, more reliable AI systems—and to better understand which values are truly being encoded in modern LLMs.

---

## 1. Executive Summary
RewardBench introduces the first comprehensive, reproducible benchmark and toolkit for evaluating reward models (RMs)—the components at the heart of reinforcement learning from human feedback (RLHF) used to align large language models (LLMs). It standardizes how to score, compare, and analyze both classifier-based and Direct Preference Optimization (DPO)–trained models across chat quality, adversarial instruction following, safety (refusal behavior), and reasoning (math and code), revealing clear strengths, weaknesses, and scaling trends (e.g., Tables 2–6, 13).

## 2. Context and Motivation
- Problem addressed
  - RLHF aligns LLMs with human preferences by training a separate reward model that scores responses, which are then optimized via reinforcement learning. Despite the centrality of reward models, there has been no standardized way to evaluate them directly; most evaluation is focused on the final chat models (“policies”).
  - Existing validation sets used for RM work (e.g., Anthropic Helpful/Harmless, OpenAI Summarize) have low ceilings due to label disagreement and often max out around 60–70% agreement, limiting diagnostic power (Introduction; Related Works).
  - New preference datasets (e.g., UltraFeedback, Nectar) lack public test sets, making comparable, generalizable RM evaluation difficult (Introduction).

- Why it matters
  - Practically: Reward models encode the “values” enforced during alignment. If their behavior is biased, inconsistent, or fragile, downstream chat models inherit those issues (Conclusion; Appendix B).
  - Scientifically: Understanding RM behavior clarifies how RLHF works, where it fails, and how different training choices (classifier vs DPO, base model, scale) change outcomes (Sections 4–5).

- Prior approaches and shortcomings
  - Indirect evaluation via model-vs-model “win rates” (e.g., MT-Bench, Chatbot Arena) tests the final chat model, not the RM that guided training (Related Works).
  - RM studies using legacy preference test sets can be noisy, narrow, or saturated (Section 5.3; Table 14), and often miss adversarial or fine-grained reasoning failure modes.

- How this work positions itself
  - RewardBench provides:
    - A curated dataset of prompt–chosen–rejected trios across four core skill areas (chat, adversarial “chat hard,” safety, reasoning) plus a “prior sets” compatibility suite (Table 1).
    - A unified scoring and inference framework for both classifier-based RMs and DPO-trained models (Sections 3–4; Figure 1).
    - A public leaderboard and code to reproduce results and analyze per-prompt score distributions (Leaderboards; Code links; Appendix E.2).

## 3. Technical Approach
This section explains how RewardBench evaluates reward models and why its design choices matter.

- What is a reward model (RM)?
  - In RLHF, an RM takes a prompt `x` and a completion `y` and outputs a scalar “reward” `r(x, y)` whose relative value is used to prefer one response over another.
  - Classifier RMs are trained from pairwise preferences by maximizing the likelihood that the chosen answer gets higher reward than the rejected one, often implemented by adding a linear head to an LM (Section 3).

- Scoring procedure (how models are compared)
  - Each datapoint contains `prompt`, `chosen`, and `rejected` responses, verified so that `chosen` should be preferred for factual, safety, or quality reasons.
  - Every RM independently scores prompt+chosen and prompt+rejected. A “win” is counted if the chosen score is higher. Accuracy is the fraction of wins over a subset (Section 4.2; Figure 1).
  - Results are normalized within sections via per-prompt weighted averages; the final RewardBench score is a weighted average across sections. The “Prior Sets” section is down-weighted by 0.5 due to noise and task ambiguity (Section 4.2).

- How classifier RMs are trained and scored (mechanics)
  - Preference modeling uses the Bradley–Terry formulation. Intuitively, a larger reward for an answer means a higher probability of preference. In notation (Eq. 1):
    - The probability that `y1` is preferred over `y2` for prompt `x` is: `exp(r*(x,y1)) / (exp(r*(x,y1)) + exp(r*(x,y2)))`.
  - The training loss maximizes the log-likelihood that chosen beats rejected across the dataset:
    - `E[log(1 + exp(rθ(x, y_rejected) - rθ(x, y_chosen)))]` (Section 3).

- How DPO-trained models are scored (mechanics)
  - DPO directly optimizes a policy using preferences, inducing an implicit reward without training a separate RM. The implicit reward for completion `y` is (Eq. 2):
    - `r(x, y) = β log(π(y|x)/π_ref(y|x)) + β log Z(x)`.
  - At evaluation, RewardBench compares the policy’s log-ratio scores for chosen vs rejected; higher log-ratio wins (Section 3). This requires the correct reference model `π_ref` used during DPO training.

- Dataset design and why it looks this way (Table 1; Section 4.1)
  - Chat (basic instruction following): Pairs drawn from AlpacaEval and MT-Bench—clear wins by stronger models over weaker ones; a special “Length” subset controls for response length (length bias is a known confound; Appendix B; Table 10).
  - Chat Hard (adversarial instruction following): Uses MT-Bench near-ties plus LLMBar adversarial sets to probe subtle instruction changes and “trick” prompts that break LLM-as-a-judge tools (Table 5; Table 11).
  - Safety: Distinguishes three behaviors—correctly refusing harmful content, not over-refusing safe prompts with trigger words (XSTest), and recognizing “do-not-answer” cases (Table 6; Table 12).
  - Reasoning: Verifies whether RMs can spot small but critical bugs or reasoning errors—uses HumanEvalPack code with buggy vs correct solutions and PRM-Math (process-reward data with verified wrong intermediate steps) (Table 13).
  - Prior Sets: Compatibility with commonly used legacy test sets (Anthropic Helpful/Harmless/HHH, SHP, Summarize) to assess generalization and potential overfitting (Table 14). Down-weighted because of noise and ceiling effects (Section 4.2).

- Weighting and normalization
  - Accuracy is the main metric (50% is random). Section-level scores are weighted averages of subsets by prompt count, except Reasoning, where PRM-Math is upweighted so math and code contribute equally; Prior Sets are unweighted internally but count for half the weight of other sections in the final average (Section 4.2).

- Implementation notes that affect interpretation
  - DPO scoring strongly depends on the exact reference model; using the “wrong” reference collapses performance near random (Appendix B; Table 7).
  - Chosen responses are designed to be similar length or shorter than rejected where possible to reduce length bias (Appendix B; Figure 9).
  - Most subsets are single-turn instructions to keep inputs comparable across RMs (Appendix F).

## 4. Key Insights and Innovations
- A unified way to evaluate very different RM families
  - Novelty: RewardBench scores both classifier RMs and DPO-trained policies as RMs using a single pairwise-accuracy protocol and shared inference stack (Section 4.2; code release). This reveals that DPO models can perform competitively on some sections but not others, and that they are highly sensitive to the choice/availability of the reference model (Table 7).

- Adversarial “Chat Hard” reveals failure modes masked by standard chat tests
  - Significance: Many models that ace standard chat comparisons struggle to detect subtle instruction shifts, topic swaps, or deliberately unhelpful responses crafted to fool judges (Table 5; Table 11). This diagnoses a core weakness in current preference training.

- A safety evaluation that disentangles three distinct behaviors
  - Innovation: RewardBench separates “should refuse,” “should respond despite trigger words,” and “do-not-answer” to characterize refusal propensity precisely (Table 6; Table 12). It identifies three behavior clusters: balanced, over-refusing, and under-refusing models.

- Reasoning tests that penalize superficial heuristics
  - Contribution: Tiny diffs (1–2 tokens) in code and verifiable math step errors stress-test whether RMs detect substantive correctness rather than surface style. Top models reach >95% accuracy; many models fall near chance, revealing large headroom (Table 13).

- Public leaderboard + per-prompt reward distributions
  - Utility: Released text-score pairs and distribution plots expose score calibration and margin patterns across datasets and models—useful to debug training and study length bias, reward shapes, and OOD shifts (Appendix E.1–E.2; Figures 2–7).

## 5. Experimental Analysis
- Evaluation setup
  - Metric: Accuracy on pairwise comparisons; random baseline = 50% (Section 4.2).
  - Datasets: 2,958 prompts across Chat, Chat Hard, Safety, Reasoning; plus Prior Sets (Table 1; Appendix F).
  - Models: >80 open and closed models, including classifier RMs (e.g., `Starling-RM-34B`, `UltraRM-13b`), DPO chat models (`tulu-2-*`, `Qwen1.5-*`), and LLM judges (GPT-4, Gemini) (Tables 2–4, 8–9).

- Main quantitative results (selected)
  - Overall leaderboard (open-weight models; Table 2)
    - > “`RLHFlow/ArmoRM-Llama3-8B-v0.1` achieves a RewardBench score of 89.0 with 96.9 (Chat), 76.8 (Chat Hard), 92.2 (Safety), 97.3 (Reasoning), 74.3 (Prior Sets).”
    - > “`RLHFlow/pair-preference-model-LLaMA3-8B`: 85.7 overall; 98.3 (Chat), 65.8 (Chat Hard), 89.7 (Safety), 94.7 (Reasoning), 74.6 (Prior Sets).”
    - These Llama-3–based RMs dominate particularly on Chat Hard and Reasoning—sections where most models struggle.
  - Scaling analysis for DPO models (Table 3)
    - > “`tulu-2-dpo-70b` 76.1 > 73.4 (`13b`) > 71.7 (`7b`) overall.” Gains are consistent across Chat, Safety, and Reasoning.
    - Qwen1.5 Chat shows less monotonic scaling across sections (e.g., `72B` overall 68.2 vs `14B` 69.8), suggesting distribution-shift sensitivity.
  - Chat Hard breakdown (Table 5; Table 11)
    - Only a few models surpass 70% on the adversarial aggregates. For example:
      - > “`ArmoRM-Llama3-8B-v0.1` averages 76.8 on Chat Hard (86.5 on MTBench Hard; 93.0 LLMBar-Natural; 67.9 Neighbor; 77.2 GPTInst; 66.0 GPTOut; 69.6 Manual).”
      - Some models that do well on this section perform poorly overall (e.g., `Qwen1.5-14B-Chat` averages 70.2 on Chat Hard yet 69.8 overall; Table 3 and Table 11), implying over-specialization to certain judge-like features.
  - Safety behavior clusters (Table 6; Table 12)
    - Balanced models:  
      > “`ArmoRM-Llama3-8B-v0.1`: 92.2 average; 93.0 (Dangerous), 97.0 (Offensive), 100.0 (Should Refuse), 87.2 (Should Respond), 79.4 (Do Not Answer).”
    - Over-refusal tendency (high on Should Respond; low on Should Refuse/Do-Not-Answer):
      > “`Qwen1.5-14B-Chat`: 76.3 average; 93.0/83.0 on Dangerous/Offensive but only 41.6 on Should Respond, 90.4 on Do Not Answer.”
    - Under-refusal (high on Should Respond; low on Should Refuse):
      > “`UltraRM-13b`: 56.0 Safety avg; 66.2 (Should Refuse), 94.8 (Should Respond), 37.5 (Do Not Answer).”
  - Reasoning performance (Table 13)
    - > “`ArmoRM-Llama3-8B-v0.1`: 97.3 average; 98.7 PRM Math and 95–98% across code languages.”  
      Many models score between 60–80; some fall below random on PRM-Math, highlighting sensitivity to process-level correctness checks.
  - DPO needs the correct reference model (Table 7)
    - Removing the reference model during evaluation drops DPO performance substantially:
      > “`Mixtral-8x7B-Instruct` falls from 82.2 to 64.2 (−18.0 overall; −28.5 Safety; −35.3 Reasoning).”
      > “`tulu-2-dpo-13b` falls from 78.8 to 62.9 (−15.9 overall; −36.5 Safety).”
    - This underscores that DPO-as-RM evaluation is meaningful only with the original `π_ref`.
  - LLM-as-a-judge vs classifier RMs (Table 8)
    - LLM judges are strong but do not top classifier RMs on RewardBench:
      > “`Gemini-1.5-pro (0514)` 88.1 or 80.7 (two runs shown), `GPT-4-0125` 84.3, `GPT-4o-2024-05-13` 83.3—still below `ArmoRM-Llama3-8B` 89.0.”  
      > Best open-weight judge, `Meta-Llama-3-70B-Instruct`, scores 75.4 overall—well below top classifier RMs.

- Do the experiments support the claims?
  - Yes, by breadth:
    - Cross-family comparisons (classifier vs DPO vs generative judges).
    - Stress tests (Chat Hard, PRM-Math, buggy code) that expose real deficits not visible in easy chat sets.
    - Sensitivity checks (DPO ref-free scoring; Table 7) and scaling trends (Table 3).
  - And by transparency:
    - Released per-prompt score logs and distribution plots make it possible to inspect calibration, margins, and potential artifacts (Appendix E.1–E.2; Figures 2–7).

- Notable robustness/ablations and failure cases
  - Length bias is explicitly considered; a “Length” subset balances average response lengths while preserving difficulty. Many models still exceed 90% here, implying some ability to judge beyond length proxies, but full causal analysis is left for future work (Appendix B; Table 10).
  - Prior Sets results are mixed and down-weighted due to dataset noise and potential contamination; DPO models often perform poorly here even when they do well elsewhere (Section 5.3; Table 14). This cautions against over-reliance on legacy preference sets.

## 6. Limitations and Trade-offs
- Reliance on curated and reformatted data rather than fresh, large-scale human preference labels
  - Many pairs originate from prior evaluations (e.g., AlpacaEval, MT-Bench, LLMBar, PRM800k, HumanEvalPack), with manual verification to reduce noise (Section 4.1; Appendix F). The paper is explicit that this is a pragmatic first step; some spurious correlations may remain (Appendix A).

- Binary accuracy metric
  - A single thresholded measure hides calibration and magnitude differences (partly mitigated by releasing raw scores). Reward magnitudes and distributions differ across RMs (Appendix E.2), and the field still lacks a consensus on desirable reward distributions for downstream RL (Section 5.1, “Different Shapes of Reward Functions”).

- Single-turn focus
  - Most subsets are single-turn instructions; multi-turn preference modeling is only weakly represented via “Prior Sets” and is down-weighted because of noise (Section 4.1; 4.2). Multi-turn RM behavior remains underexplored.

- DPO evaluation dependency
  - Meaningful DPO scoring requires the exact `π_ref`. Many open models do not document `π_ref` unambiguously, which can confound benchmarking (Appendix B; Table 7).

- Compute and scaling
  - Running the full suite for ~75 models consumed ~1000 A100-GPU hours; this is tractable for labs but non-trivial for many practitioners (Appendix C).

- Potential contamination and ceiling effects in Prior Sets
  - Legacy evaluation sets may have been used during training by some models or are inherently noisy, which muddies interpretation. RewardBench down-weights them for the final score (Section 5.3; Table 14).

## 7. Implications and Future Directions
- How this changes the landscape
  - RewardBench sets a practical standard for evaluating reward models directly. It reveals that:
    - Classifier-based RMs currently outperform generative judges and many DPO-as-RM models on hard reasoning and adversarial instruction following (Tables 2, 5, 13).
    - Scaling DPO helps but does not eliminate weaknesses in “Chat Hard” or safety balance (Table 3, Table 6).
    - Safety needs multi-dimensional assessment; aggregate “safety scores” hide over- vs under-refusal trade-offs (Table 6).

- What research it enables
  - Training science for RMs
    - Study how RM score distributions (shape, calibration) affect PPO or best-of-N sampling (Appendix E.2; Section 6 Conclusion).
    - Systematic exploration of DPO hyperparameters (β regularization, epochs), `π_ref` choices, and hybrid training (Appendix B).
    - Incorporate fine-grained, step-level feedback signals (e.g., process reward models) into RM training and evaluation (Section 2; Reasoning).

  - Data curation and generalization
    - Expand “Chat Hard” with new adversarial transformations; build multi-turn and tool-use RM evaluations.
    - Create new, human-labeled test sets targeted at decision boundaries (similar to PRM-Math but for non-code tasks).

  - Safety alignment
    - Use RewardBench’s safety decomposition as a quick “safety profile” during training iterations—detecting drift toward over- or under-refusal (Table 6).
    - Integrate small, specialized safety RMs as system guardrails and evaluate them specifically on the relevant sections (Appendix B, “Safety In or After RLHF”).

- Practical applications
  - Model selection: Choose RMs that match product goals—e.g., high `Chat Hard` for consistency under prompt perturbations, or balanced safety behavior for consumer deployments.
  - Diagnostics: Use per-prompt logs to locate systematic errors (e.g., topic swaps in LLMBar-Neighbor, false refusals in XSTest Should Respond).
  - Vendor evaluation: Closed models can be compared apple-to-apple against open RMs across standardized tasks (Table 8; overall leaderboard).

> Closing observation (Section 6): “A crucial next step is to correlate performance in REWARDBENCH to RLHF usefulness.” The toolkit is ready for such studies (e.g., best-of-N ranking experiments, PPO fine-tuning with different RMs), which will help connect RM evaluation to downstream user experience and safety guarantees.
