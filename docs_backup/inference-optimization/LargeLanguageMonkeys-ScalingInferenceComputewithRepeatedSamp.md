# Large Language Monkeys: Scaling Inference Compute with Repeated Sampling

**ArXiv:** [2407.21787](https://arxiv.org/abs/2407.21787)

## 🎯 Pitch

This paper introduces a simple yet powerful technique: dramatically boosting problem-solving performance of language models by generating many independent solutions at inference time and selecting a correct one using an automatic verifier. The authors reveal that coverage—how often at least one solution is correct—scales log-linearly with the number of samples, establishing 'inference-time scaling laws' akin to those that revolutionized model training. This insight shows that weaker, cheaper models paired with repeated sampling can outperform stronger models on real-world tasks, offering both practical benefits for domains like software engineering and new principles for budgeting AI compute.

---

## 1. Executive Summary
This paper shows that simply sampling many independent solutions from a language model at inference time—then using a verifier to select a correct one—yields large, predictable gains in problem‑solving coverage across math, coding, and formal proof tasks. It further uncovers inference‑time scaling laws: coverage grows approximately log‑linearly with the number of samples and is well modeled by an exponentiated power law, enabling weaker, cheaper models plus many samples to beat stronger models with a single try (e.g., 56% vs. 43% solved on SWE‑bench Lite with 250 attempts; Figure 2).

## 2. Context and Motivation
- Problem addressed
  - Most LLM applications give the model a single attempt per problem, even though model training benefits enormously from more compute. The paper addresses how to systematically “scale inference compute” by drawing many candidate solutions per problem and selecting from them (Figure 1).
- Why this matters
  - Practical: In domains with automatic verifiers (e.g., unit tests, proof checkers), more attempts translate directly into more solved problems—important for software engineering and formal reasoning.
  - Scientific: The work suggests inference‑time scaling laws (Sections 2–3), analogous to training scaling laws, to help plan compute budgets.
- Prior approaches and gaps
  - Chain‑of‑thought and related prompting spend more tokens per sample but still usually take one attempt; self‑consistency/majority vote helps but typically with small sample counts and without characterizing scaling at large k.
  - AlphaCode showed benefits at very large sample counts in coding contests, but broad characterization across tasks/models and cost–benefit analysis was missing.
- Positioning
  - The paper isolates two determinants of success with repeated sampling (Section 1): 
    - Coverage: does any sample solve the problem?
    - Precision: can we identify the correct sample among many?
  - It systematically studies how coverage scales (up to 10,000 samples), models that scaling with a simple law (Equation 3), and examines precision with common verifiers (Figure 7).

Key terms (defined when first used):
- Repeated sampling: generating multiple independent responses for the same prompt using non‑zero temperature.
- Coverage: fraction of problems for which at least one sampled response is correct (with a perfect verifier). In code/proof, this equals pass@k.
- Precision: ability of a selection method (e.g., tests, proof checker, majority vote, reward model) to pick a correct response from the set of samples.
- Verifier: any mechanism that can automatically check correctness (e.g., unit tests, proof checker). When such a tool exists, “oracle verification” is effectively available.

## 3. Technical Approach
Step‑by‑step methodology (Figure 1):
1. Generate samples
   - For each problem, sample k independent candidate solutions from an LLM with positive temperature (k up to 10,000 for math/code; Section 2).
2. Select final answer
   - If an automatic verifier exists (proof checker for Lean; unit tests for CodeContests and SWE‑bench Lite), run it on each candidate and choose a passing solution; otherwise use a selection rule such as majority vote or a reward model (Section 4).

Metrics and estimation
- The core metric is coverage/pass@k: the probability that at least one of the k samples solves the problem. To estimate it from N generated samples per problem, they use the unbiased pass@k estimator from Chen et al. (2021) (Equation 1, Section 2), where `C_i` is the number of correct samples among N for problem i:
  - pass@k = average over problems of 1 − [(N − C_i choose k) / (N choose k)].
- Precision is evaluated by applying selection rules to the same sample sets and computing success rate (fraction of problems where the selected final answer is correct).

Tasks, models, and sampling budgets (Sections 2 and A–B)
- Tasks (five): GSM8K, MATH (math word problems; no automatic verifier), MiniF2F‑MATH (Lean proofs; automatic), CodeContests (coding with hidden tests; automatic), SWE‑bench Lite (real GitHub issues; automatic via repo tests).
- Models (varied): Llama‑3 family (8B, 70B; base and instruct), Gemma (2B, 7B), Pythia (70M–12B), and DeepSeek‑Coder‑V2‑Instruct for SWE‑bench Lite (Sections 2.1–2.2).
- Sampling details include temperatures, few‑shot setups, and token limits for each dataset (Appendix A–B). Example: CodeContests uses T=0.6 and top‑p=0.95; SWE‑bench Lite uses an agent framework (Moatless Tools) and T=1.6 with up to 250 independent trajectories per issue (Appendix B.1).

Modeling inference‑time scaling (Section 3)
- Observation: plotting coverage against the number of samples on a log x‑axis often yields approximate straight lines over several orders of magnitude (Figure 2, Figure 5).
- Model: fit an exponentiated power law to coverage c as a function of samples k (Equations 2–3):
  - log(c) ≈ a·k^b  ⇒  c ≈ exp(a·k^b)
  - Fits are shown in Figure 5 and Appendix C.2 for many model–task pairs, with small average errors for most curves.

Comparing model families (Section 3.2; Figure 6)
- Within a model family (e.g., Llama‑3), coverage curves resemble S‑curves with similar slopes but different horizontal offsets. The paper visualizes this by horizontally shifting each curve so it passes through the same anchor point (k/pass@k^−1(c), c).

Cost modeling (Section 2.3)
- FLOPs estimate per token: 2·(parameters + 2·layers·token_dim·context_len), summed over prompt and decode tokens and multiplied by number of completions. They plot coverage vs. total inference FLOPs for Llama‑3 8B vs. 70B (Figure 4) to reveal which model is best at fixed compute.

Caveats in verification (Section 4.2)
- SWE‑bench Lite has flaky tests on 11.3% of problems (Table 3; Appendix B.2), so results include majority voting over repeated runs for those cases; trends remain after removing such problems (Figure 9).
- CodeContests has false negatives: 35/122 Python3 tasks include “correct” solutions that fail provided tests because tests sometimes enforce a single acceptable output or include mutated inputs violating the spec (Section 4.2.2).

## 4. Key Insights and Innovations
- Inference‑time scaling laws for repeated sampling
  - Novelty: Coverage often follows an exponentiated power law c ≈ exp(a·k^b) over 4+ orders of magnitude in k (Equations 2–3; Figure 5; Appendix C.2).
  - Significance: This provides a simple predictive model for how much more coverage one can buy by increasing sample count, analogous to training scaling laws.
- Weak models amplified via many samples
  - Evidence: On SWE‑bench Lite, DeepSeek‑Coder‑V2‑Instruct rises from 15.9% with one attempt to 56% with 250 attempts, surpassing the single‑attempt state‑of‑the‑art of 43% (Figure 2). On MATH, tiny models like Pythia‑160M jump from 0.27% pass@1 to 57% pass@10k (Figure 3).
  - Why this matters: Budget‑constrained users can trade more attempts for stronger single‑shot capability.
- Precision is the bottleneck without automatic verifiers
  - Insight: Majority voting and reward‑model selection plateau around ~100 samples even while coverage keeps rising toward >95% (Figure 7). For Llama‑3‑8B‑Instruct on MATH, coverage grows from 82.9% (k=100) to 98.44% (k=10,000), but majority vote/reward‑model success barely moves (40.50%→41.41%).
  - Implication: To harvest the benefits at large k, verifiers must “find the needle in the haystack.”
- Cost–performance trade‑offs clarified
  - FLOPs view: With fixed inference FLOPs, the “best” model size depends on the task (Figure 4). 8B beats 70B on MiniF2F, GSM8K, and MATH; 70B is better on CodeContests.
  - Dollar view: For SWE‑bench Lite, five attempts of DeepSeek‑Coder‑V2‑Instruct solve more issues than single attempts of GPT‑4o or Claude 3.5 Sonnet while costing 3–5× less (Table 1).

## 5. Experimental Analysis
Evaluation design
- Datasets and verifiers (Section 2)
  - With automatic verifiers: MiniF2F‑MATH (Lean proof checker), CodeContests (hidden test cases), SWE‑bench Lite (project unit tests via Moatless Tools agent).
  - Without automatic verifiers: GSM8K and MATH; here, “coverage” uses an oracle that counts a problem solved if any sample’s final answer matches (Equation 1).
- Models and sample budgets
  - Diverse sizes/families: Llama‑3 (8B/70B; base/instruct), Gemma (2B, 7B), Pythia (70M→12B), DeepSeek‑Coder‑V2‑Instruct.
  - Budgets: up to 10,000 samples per problem for math/code; up to 250 agent trajectories for SWE‑bench Lite (Sections 2.1–2.2, B.1).
- Metrics
  - Coverage/pass@k (Equation 1) is the primary metric; success rate for selection rules (precision) where no automatic verifier exists (Figure 7).

Main quantitative results
- Broad coverage gains with increasing k (Figure 2; Figure 3)
  - GSM8K/MATH (oracle): smooth growth toward near‑perfect coverage at large k for Llama‑3 8B/70B; similar trends across Gemma and Pythia on MATH (Figure 3).
  - CodeContests: consistent gains for Llama‑3 and Gemma; Gemma‑2B rises >300× from 0.02% (pass@1) to 7.1% (pass@10k) (Section 2.2).
  - MiniF2F‑MATH: both Llama‑3 models improve with k (Figure 2).
  - SWE‑bench Lite: DeepSeek‑Coder‑V2‑Instruct scales from 15.9% to 56% solved at k=250, beating single‑attempt SOTA of 43% (Figure 2). The trend is robust to removing flaky problems (54.14% vs. 56.00%; Figure 9).
- Family‑wise alignment of scaling curves (Figure 6)
  - After shifting coverage curves to share an anchor point, models within the same family trace similar shapes, suggesting constant multiplicative increases in k to move between coverage levels.
- Precision studies on math word problems (Section 4.1; Figure 7; Table 2)
  - Majority vote, reward‑model best‑of‑N, and reward‑weighted voting improve initially but saturate before 100 samples, creating a widening gap to the oracle coverage curve.
  - Human grading of 105 correct chains‑of‑thought shows >90% are logically faithful even on hard problems (Table 2), indicating the information exists for better verifiers.
- Cost analyses (Section 2.3; Figure 4; Table 1)
  - FLOPs curves show task‑dependent optimal model size at fixed compute.
  - SWE‑bench Lite cost: five DeepSeek attempts cost 10.8 USD total to attain 29.62% solved versus single attempts of GPT‑4o (24.00%, 39 USD) and Claude 3.5 Sonnet (26.70%, 51 USD) (Table 1).

Ablations, failures, robustness
- Scaling‑law fit quality varies by task; MiniF2F‑MATH deviates more (Figure 5).
- Pythia models achieve zero coverage on CodeContests even at 10,000 samples (Section 2.2), likely due to less code‑specific training data.
- Dataset/verification caveats
  - SWE‑bench Lite: 11.3% flaky tests; trends stable when removing those tasks (Appendix B.2; Figure 9; Table 3).
  - CodeContests: 35/122 tasks have false‑negative tests (Section 4.2.2).
  - GSM8K: one mislabeled ground truth prevents 100% coverage for Llama‑3‑70B (Appendix E).

Overall assessment
- The experiments convincingly support that coverage scales strongly with k across tasks and models (Figures 2–3) and that simple selection rules do not keep up (Figure 7). The cost analyses are informative but rely on approximations for FLOPs and API pricing; still, they substantiate practical trade‑offs (Figure 4; Table 1).

## 6. Limitations and Trade-offs
- Dependence on verifiers
  - Where automatic verifiers exist, benefits translate directly into success. Without them (GSM8K/MATH), common selectors plateau early (Figure 7), leaving unrealized coverage at high k.
- Single‑turn, independent attempts
  - The study uses independent attempts with fixed prompts/hyperparameters (Section 5: “we explore only a simple version”). Multi‑turn or feedback‑driven strategies might be more compute‑efficient.
- Scaling‑law generality
  - The exponentiated power law fits many but not all curves well (e.g., MiniF2F‑MATH deviates; Figure 5). Parameters a, b vary by task/model; extrapolation beyond observed k is uncertain.
- Dataset and testing artifacts
  - Flaky or imperfect test suites (SWE‑bench Lite, CodeContests) complicate precision and inflate/deflate measured success (Sections 4.2.1–4.2.2; Appendix B.2).
- Compute and latency trade‑offs
  - Repeated sampling increases total compute and may increase user‑perceived latency unless engineered for throughput. FLOPs estimates omit system‑level efficiency details (Section 2.3, Discussion).
- Domain coverage
  - The paper focuses on pass–fail tasks; subjective generative tasks (e.g., creative writing) need different verifiers.

## 7. Implications and Future Directions
- Practical deployment guidance
  - Treat inference compute as a tunable budget: for tasks with reliable verifiers, prefer many samples from a cheaper model when cost or access to frontier models is constrained (Figure 4; Table 1).
  - Engineer inference systems for high‑throughput multi‑sample workloads—batching and shared‑prefix attention optimizations (Section 5) can reduce cost versus naïvely firing many independent API calls.
- Research directions
  - Better verifiers at scale: train or design verifiers that can identify rare correct samples among many, leveraging faithful chains‑of‑thought (Figure 7; Table 2). Explore step‑wise verification, process supervision, or proof‑style formalization for math word problems.
  - Diversity‑aware sampling: beyond temperature, incorporate higher‑level diversification (metadata conditioning, sampling trajectories; Section 5 “Solution Diversity”).
  - Multi‑turn and feedback: integrate iterative refinement, execution feedback, and learning from previous attempts to increase per‑attempt success (Section 5).
  - Theory of inference scaling: formalize when the exponentiated power law emerges, what determines parameters a and b, and how they relate to model calibration and problem difficulty.
- Downstream applications
  - Software engineering agents (SWE‑bench Lite) and competitive programming (CodeContests) can benefit immediately where test suites exist.
  - Formal methods: repeated sampling plus proof checking (MiniF2F‑MATH) shows sizeable headroom without changing model weights.

Selected supporting quotes and pointers
- “Coverage … scales with the number of samples over four orders of magnitude” and “can be modelled with an exponentiated power law” (Abstract; formalized in Equations 2–3; fits in Figure 5 and Appendix C.2).
- SWE‑bench Lite: “from 15.9% with one sample to 56% with 250 samples” outperforming single‑sample SOTA 43% (Figure 2; Appendix B/Figure 9).
- Precision bottleneck: “majority voting and reward models … plateau beyond several hundred samples,” e.g., 40.50%→41.41% while coverage grows 82.9%→98.44% (Figure 7; Section 4.1).
- Cost trade‑offs: fixed‑FLOPs curves swap the preferred model size by task (Figure 4); API cost comparison favors repeated sampling with a cheaper model (Table 1).

In short, this work reframes problem solving with LLMs as a two‑part system—coverage and precision—and shows that scaling inference compute with repeated sampling yields predictable, often dramatic gains. The next frontier is building verifiers that scale in tandem so those gains translate into realized task performance across domains that lack automatic checking.
