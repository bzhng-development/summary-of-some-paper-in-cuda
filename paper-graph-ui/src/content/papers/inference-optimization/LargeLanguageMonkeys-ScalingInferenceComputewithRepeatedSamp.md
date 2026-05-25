# Large Language Monkeys: Scaling Inference Compute with Repeated Sampling

**ArXiv:** [2407.21787](https://arxiv.org/abs/2407.21787)

## 🎯 Pitch

This paper pioneers a simple yet powerful approach to scaling large language model performance by repeatedly sampling many independent candidate solutions at inference time and leveraging automatic verifiers (like unit tests or proof checkers) to select correct answers. The authors establish predictable, near log-linear scaling laws for how 'coverage'—the chance of solving a problem—increases with the sample count, revealing that even weaker, cheaper models can outperform state-of-the-art systems when given enough attempts. This unlocks a new, practical direction for amplifying LLM performance, especially in domains where solutions can be automatically validated, leading to breakthroughs in both capability and cost-effectiveness—for example, solving 56% of SWE-bench Lite issues by sampling 250 times with an open-source model, far surpassing previous single-sample results.

---

## 1. Executive Summary
This paper studies inference-time scaling for large language models (LLMs) by a very simple procedure: generate many independent answers for each problem (“repeated sampling”) and use a verifier to pick a correct one. It shows that the probability that at least one sample is correct (“coverage”) increases smoothly—often almost log‑linearly—with the number of samples over 4 orders of magnitude, and that this can turn cheap or weaker models into strong performers on tasks that have automatic verifiers (code, formal proofs), including a new result on SWE-bench Lite where 250 samples from an open model solve 56% of issues (Figure 2).

## 2. Context and Motivation
- Problem/gap
  - Training-time scaling has produced large gains, but inference-time compute is usually capped at a single try per problem. The paper asks: what if we deliberately spend more inference compute by sampling repeatedly, and can we do so predictably and cost‑effectively? (Introduction; Figure 1).
- Why it matters
  - Real-world tasks like coding and formal proofs provide automatic verifiers (unit tests, proof checkers). If we can generate a correct candidate among many, we can reliably detect it—turning extra inference compute directly into task success (Section 2).
  - Without verifiers (e.g., math word problems), we still need a method to identify the rare correct sample; the paper evaluates how far common methods scale (Section 4).
- Prior approaches and their limits
  - Chain-of-thought and deliberation increase tokens per answer but still usually yield a single attempt.
  - Best-of-N with majority voting or reward models helps, but it is unclear how performance grows with large N (hundreds to thousands).
  - AlphaCode showed benefits from massive sampling for competitive programming, but a systematic, cross-task view and simple scaling laws for inference-time compute were missing.
- Positioning
  - The work isolates and measures two properties that determine practical gains from repeated sampling:
    - `coverage`: whether any sample is correct (Problem 1 in Figure 1).
    - `precision`: whether we can identify a correct sample among many (Problem 2 in Figure 1).
  - It characterizes how coverage scales with sample count, proposes a compact model of this relationship, and studies verification methods and cost/performance trade-offs (Sections 2–4).

## 3. Technical Approach
- Core procedure (Figure 1)
  1) Generate `k` independent samples per problem using a positive temperature (stochastic decoding).
  2) Use a verifier to select a final answer.
     - Automatic verifiers (code unit tests; Lean proof checker) provide ground-truth pass/fail for each sample.
     - For tasks without automatic verifiers (math word problems), use selection methods like majority voting or a reward model.
- Definitions (Section 2)
  - `coverage` (a.k.a. “pass@k” in coding): fraction of problems where at least one of the `k` samples is correct. It is an upper bound on achievable success with a perfect verifier.
  - `precision`: ability of the selection method to pick a correct sample from the batch of k generations.
  - `oracle verifier`: a hypothetical perfect selector that picks a correct sample whenever one exists; using it, success equals coverage.
- Estimating coverage (Equation 1)
  - They first generate `N` samples per problem (e.g., 10,000), count `C_i` correct among them, and then compute an unbiased estimate of pass@k:
    - pass@k = average over problems of 1 − [C(N−C choose k) / (N choose k)].
  - Plain-language intuition: if we know how many of the N are correct, the probability that a random draw of k (without replacement) contains at least one correct one is 1 minus the chance we picked only incorrect ones.
- Experimental design (Sections 2, A, B)
  - Tasks
    - GSM8K (grade-school math word problems; 128 test items; Section 2), no automatic verifier.
    - MATH (harder math problems; 128 items), no automatic verifier.
    - MiniF2F-MATH (formalized math problems in Lean; 130 items), automatic proof checker.
    - CodeContests (competitive programming; Python3; 140 items without images), automatic unit tests.
    - SWE-bench Lite (real GitHub issues; automatic unit tests; multi-turn tool-augmented agent; 300 issues; Section 2.1 and Appendix B).
  - Models
    - Llama-3-8B/-70B Instruct; Gemma-2B/-7B; Pythia 70M–12B; DeepSeek-Coder-V2-Instruct (Sections 2.1–2.2).
  - Sampling budgets
    - Up to 10,000 samples/problem for single-turn tasks; up to 250 independent agent attempts/issue on SWE-bench Lite (Section 2.1).
  - Prompting and decoding details (Appendix A)
    - CodeContests: temperature 0.6, top-p 0.95, 2 few-shots; max 1024 tokens.
    - MATH: temperature 0.6, 5 fixed few-shots; max 512 tokens.
    - GSM8K: temperature 0.6, 5 sampled few-shots; max 512 tokens.
    - MiniF2F (Lean): temperature 0.5; max 200 tokens; standard Lean proof-checking.
  - SWE-bench Lite agent setup (Appendix B)
    - DeepSeek-Coder-V2-Instruct + Moatless Tools agent framework; 250 independent attempts; temperature selected by a sweep on 50 issues (best at 1.6).
- Modeling the scaling curve (Section 3)
  - Empirical observation: coverage tends to increase roughly log‑linearly with number of samples (Figures 2–3).
  - Compact model (“exponentiated power law”, Equations 2–3):
    - log(c) ≈ a·k^b, or equivalently c ≈ exp(a·k^b), with `a,b` fitted to measured coverage at multiple k.
  - Same-family similarity: when overlaying coverage curves of different sizes within a model family, the shapes are very similar and differ mostly by a horizontal shift in log‑k (Figure 6).
- Cost modeling (Section 2.3)
  - Approximate total inference compute with FLOPs (floating-point operations), summing per-token costs over prompt and decoded tokens and multiplying by number of completions (Section 2.3, formula shown).
  - Also compare real API dollar costs for SWE-bench Lite (Table 1).

## 4. Key Insights and Innovations
- Repeated sampling yields predictable, large coverage gains across tasks and models
  - Novelty: a systematic, cross-task, cross-model characterization up to 10,000 samples/problem.
  - Significance: even small/base models gain dramatically in pass@k; e.g., Gemma‑2B on CodeContests rises from pass@1 0.02% to pass@10k 7.1% (Section 2.2; Figure 3).
- A simple scaling law often fits coverage vs. sample count
  - Innovation: the `c ≈ exp(a·k^b)` fit captures the observed log-linear trend over several orders of magnitude (Section 3.1; Figures 5 and 10). Errors are small on many datasets; exceptions (e.g., MiniF2F) reveal where the model is imperfect.
  - Impact: provides a planning tool to forecast returns from more samples.
- Within-model-family “same-shape, shifted” sampling curves
  - Insight: for a given task, different sizes within a family (Llama, Gemma, Pythia) trace S‑curves of similar slope after aligning at a common coverage anchor (Figure 6).
  - Implication: coverage improvements from additional samples are multiplicative and comparable across sizes; the primary difference is how many samples are needed to reach a target coverage.
- Cost-performance trade-offs favor many samples from cheaper models in some settings
  - Evidence: with fixed FLOPs, smaller Llama‑3‑8B‑Instruct outperforms 70B on MiniF2F, GSM8K, MATH, while 70B dominates CodeContests (Figure 4). In API costs on SWE-bench Lite, 5 attempts of DeepSeek-Coder-V2-Instruct solve 29.62% issues at $10.8 total, beating single attempts of GPT‑4o (24% at $39) and Claude 3.5 Sonnet (26.7% at $51) (Table 1).
  - Takeaway: inference-time compute can be “reallocated” between model size and number of samples.
- Verification is the bottleneck without automatic checkers
  - Finding: coverage keeps rising up to 10k samples, but majority vote or reward-model-based selection plateau around ~100 samples, failing to “find the needle in the haystack” (Figure 7). For MATH with Llama‑3‑8B‑Instruct, coverage grows from 82.9% @100 to 98.44% @10,000, while majority vote grows only from 40.50% to 41.41% (Section 4.1; Figure 7).
  - Importance: scalable verification is necessary to fully exploit repeated sampling on open-ended tasks.

## 5. Experimental Analysis
- Evaluation methodology
  - Datasets and verifiers
    - Automatic verifiers: MiniF2F (Lean proof checker), CodeContests (hidden tests), SWE-bench Lite (unit tests).
    - No automatic verifier: GSM8K, MATH; evaluate coverage (oracle upper bound) and selection methods (majority vote; reward-model best-of-N; reward-model-weighted vote) (Section 4.1).
  - Metrics
    - Primary: coverage (pass@k) via unbiased estimator (Equation 1).
    - Success rate with specific selectors on GSM8K and MATH (Figure 7).
  - Baselines
    - Single-attempt GPT‑4o on all tasks, and single-attempt SOTA for SWE-bench Lite (CodeStory Aide + mixed models, 43%) (Figure 2).
- Main quantitative results
  - Cross-task coverage growth (Figure 2)
    - GSM8K/MATH/MiniF2F/CodeContests (Llama‑3 8B/70B): coverage increases smoothly with k; with enough samples, both Llama models exceed single-attempt GPT‑4o.
    - SWE-bench Lite (DeepSeek‑Coder‑V2‑Instruct + Moatless Tools):
      > “15.9% with one sample to 56% with 250 samples, outperforming the single-sample state-of-the-art of 43%” (Abstract; Figure 2).  
      Flaky tests analysis shows the trend holds with or without those issues removed (Appendix B; Figure 9).
  - Scaling across sizes/families (Figure 3)
    - Broadly increasing coverage across Llama, Gemma, and Pythia on MATH; some of the largest proportional gains occur for small models (e.g., Pythia‑160M from pass@1 0.27% to pass@10k 57%).
    - Exception: on CodeContests, Pythia models have near-zero coverage even at 10k samples, likely due to less code data (Section 2.2).
  - Inference compute vs. coverage (Figure 4)
    - With fixed FLOPs, small vs. large model preference depends on task: 8B wins on MiniF2F, GSM8K, MATH; 70B wins on CodeContests.
  - Cost in dollars on SWE-bench Lite (Table 1)
    > DeepSeek‑Coder‑V2‑Instruct (5 attempts): 29.62% solved at $10.8 total cost (1×); GPT‑4o (1 attempt): 24% at $39 (3.6×); Claude 3.5 Sonnet (1 attempt): 26.7% at $51 (4.7×).
  - Scaling law fits (Figures 5 and 10)
    - The exponentiated power law `c ≈ exp(a·k^b)` visually tracks measured curves with small mean errors on many settings (e.g., Llama‑3‑8B‑Instruct on CodeContests, mean error ≈ 0.002 ± 0.0015; Figure 5).
    - Not perfect on MiniF2F (Llama‑3‑8B‑Instruct shows larger deviation; Figure 5), indicating limits of the model.
  - Verification scaling on GSM8K and MATH (Figure 7)
    - Majority vote, reward model best‑of‑N, and reward‑weighted vote initially improve but saturate by ~100 samples, leaving a widening gap to the oracle coverage curve.
    - Despite weak selectors, >90% of chains-of-thought in correct samples appear logically sound in a human study (Table 2), suggesting exploitable signal exists for better verifiers.
- Robustness checks, failure cases, and data issues
  - SWE-bench Lite has 34 instances with flaky tests; even gold solutions sometimes fail; majority voting over 11 runs mitigates (Appendix B; list in Table 3).
  - CodeContests test suites sometimes cause false negatives (35 of 122 Python3 problems had “correct” reference solutions failing tests) due to multiple-valid-output cases and mutated inputs violating problem specs (Section 4.2.2).
  - GSM8K includes at least one incorrect ground-truth answer (Appendix E), which explains why Llama‑3‑70B‑Instruct never produced a “correct” sample for that item (noted in Figure 4 caption).

Do the experiments support the claims?
- Yes for coverage scaling: strong, repeated across tasks/models with large k and detailed methodology (Figures 2–3; Appendices A, C).
- Yes for cost/performance trade-offs: both FLOPs-based and real-API-cost comparisons (Figure 4; Table 1).
- The scaling law fit is compelling but not universal (Figures 5, 10).
- The verification bottleneck claim is convincingly demonstrated by the widening gap between selection methods and coverage (Figure 7) and the human CoT audit (Table 2).

## 6. Limitations and Trade-offs
- Reliance on verifiers
  - Automatic verifiers are crucial for fully harvesting coverage gains; on tasks without them, mainstream selectors plateau early (Figure 7). The approach assumes either an oracle or a strong verifier exists.
- Dataset/test quality issues
  - Flaky or imperfect test suites (SWE-bench Lite; CodeContests) introduce false negatives/positives and noise in evaluation (Section 4.2; Appendix B).
- Compute and system constraints
  - Generating thousands of samples per problem is compute-intensive, though batching and shared-prefix optimizations can reduce cost in practice (Section 5). Real-time, low-latency applications may struggle to use very large k.
- Generality of scaling law
  - The exponentiated power law does not fit every setting (e.g., MiniF2F; Figure 5) and is empirical rather than theoretically derived. Extrapolation should be cautious.
- Model/task coverage
  - Some model families (Pythia) do not improve on certain tasks (CodeContests), likely due to training data limitations (Section 2.2).
- Experimental scope
  - Sampling is independent and single-turn for many tasks; the paper does not explore coordinated diversity strategies, multi-turn repair/feedback, or learning from previous attempts (Section 5 “Improving Repeated Sampling”).

## 7. Implications and Future Directions
- How this changes the landscape
  - Inference-time compute becomes a first-class scaling axis alongside training compute. Practitioners can trade model size for number of samples and still reach or surpass strong single-attempt baselines, sometimes at much lower dollar cost (Figure 4; Table 1).
  - For automatically verifiable domains (software engineering, formal methods), repeated sampling is a powerful and simple capability amplifier.
- Research directions
  - Better verifiers for open-ended reasoning
    - The widening coverage–selector gap (Figure 7) motivates research on scalable verifiers (training dedicated verifiers, step-wise verification, self-critique pipelines) and on converting tasks into verifiable formats (e.g., formalization into Lean; Section 5).
  - Diversity and coordination among samples
    - Move beyond pure temperature sampling: condition samples on different high-level plans, use tree-search over solution sketches, or share/learn from failed attempts (Section 5).
  - Multi-turn and feedback-rich sampling
    - Integrate execution feedback (unit tests, proof-checker traces) into sampling loops; study the cost/benefit of fewer, smarter multi-turn attempts vs. many one-shot samples (Section 5).
  - Systems optimization
    - Specialized inference systems for high-throughput repeated sampling (shared-prefix attention, high batching) can further reduce costs (Section 5; refs [34, 6, 66]).
  - Theory of inference scaling
    - Formalize why and when `c ≈ exp(a·k^b)` arises; characterize task/model properties that govern `a` and `b`, and derive optimal k allocation across problems.

Practical applications
- Software engineering assistants that run many parallel fix attempts and validate via tests (SWE-bench Lite: 56% at k=250; Figure 2).
- Competitive programming and code synthesis at scale (large k with unit tests; Figures 2–3).
- Formal theorem proving with proof checkers (MiniF2F gains; Figure 2).
- Any workflow with cheap verification and flexible latency budgets can leverage repeated sampling to convert cheap compute into reliability.

Overall, this paper provides a clear, data-backed blueprint for using repeated sampling to scale inference-time compute—what to expect as k grows, when it pays off, how to estimate costs, and where verification is the limiting factor.
