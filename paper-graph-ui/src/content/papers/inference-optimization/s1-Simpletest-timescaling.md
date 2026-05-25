# s1: Simple test-time scaling

**ArXiv:** [2501.19393](https://arxiv.org/abs/2501.19393)

## 🎯 Pitch

This paper introduces a remarkably simple and transparent recipe for test-time scaling in reasoning language models: just fine-tune an existing model on 1,000 expertly-curated chain-of-thought examples and control reasoning depth at inference using a decoding intervention called 'budget forcing.' This approach yields a 32B-parameter open model (s1-32B) that not only demonstrates true, monotonic accuracy gains as more compute is spent per question, but also surpasses leading closed models on math benchmarks with orders of magnitude less supervision. The simplicity, sample efficiency, and full openness of this method democratize advanced reasoning research and set a new standard for reproducible, controllable interpretability in LLM test-time computation.

---

## 1. Executive Summary (2-3 sentences)
This paper introduces a minimal recipe for “test-time scaling” of reasoning in language models: fine‑tune an existing model on just 1,000 curated chain‑of‑thought examples (`s1K`) and control its thinking length at inference with a simple decoding trick called `budget forcing`. The resulting 32B-parameter model (`s1-32B`) shows consistent accuracy gains as more test-time “thinking tokens” are allowed (sequential scaling), and reaches competitive or better performance than larger or closed models on math benchmarks while using orders of magnitude less training data (Figures 1–2, Table 1).

## 2. Context and Motivation
- Problem/gap addressed
  - Many recent reasoning LLMs rely on heavy training (e.g., reinforcement learning over millions of samples) or complex inference-time search (MCTS, multi-agent debate). Yet, a clear, open, and reproducible demonstration of monotonic “test-time scaling” (more inference compute ⇒ better accuracy) has been missing (Introduction; §6.1–6.2).
  - Existing open replications have improved raw performance but not the “scaling with thinking” behavior associated with systems like OpenAI’s o1 (Figure 1; §6.1).
- Why it matters
  - Practically, controllable inference compute lets users spend more “thinking” on hard questions and less on easy ones, improving efficiency.
  - Scientifically, it separates “reasoning at inference” from “reasoning baked into weights,” enabling systematic study of how compute at test time affects accuracy (§3.2).
- Prior approaches and their shortcomings
  - Heavy RL pipelines (e.g., DeepSeek R1; millions of traces) achieve strong scores but require massive data and infrastructure (§6.1; Table 1).
  - Parallel inference scaling (majority voting, Best-of-N) improves with more samples but doesn’t let later thoughts refine earlier ones; it also doesn’t guarantee compute control (§3.1; Figure 4b).
  - Earlier “length control by prompt” methods are unreliable because models can’t count tokens or hack step budgets (§5.2; Tables 12–13).
- How this work positions itself
  - Minimalism: use supervised fine‑tuning (SFT) on only 1,000 hand‑selected chain‑of‑thought samples plus a decoding-time controller (“budget forcing”) to obtain both strong performance and clean, controllable test‑time scaling (Abstract; §2–3; Figure 1).
  - Openness and efficiency: open weights, data, and code; training took 26 minutes on 16×H100 GPUs for `s1-32B` (§4.1; D. Training details).

## 3. Technical Approach
Key terms
- `Test-time scaling`: increasing the amount of computation the model performs during inference (e.g., generating more “thinking tokens”) to improve accuracy (§3.1).
- `Thinking tokens`: the tokens in a chain‑of‑thought (reasoning) segment produced before the final answer (§4; Figure 5).
- `Sequential` vs `Parallel` scaling: sequential lets later thoughts build on earlier ones (one long trace); parallel runs many independent attempts (e.g., majority vote) (§3.1; Figure 4).
- `Budget forcing`: a decoding-time intervention that either forces the model to stop thinking at a maximum token budget or to continue thinking when it tries to stop (§3.1; Figure 3).
- `End‑of‑thinking delimiter`: a special token marking the boundary between the reasoning trace and the answer. Here, the model is trained with explicit delimiters `<|im_start|>think` … `<|im_start|>answer` (D. Training details).

Step-by-step methodology
1) Build a small but high‑value reasoning dataset (`s1K`)
   - Start with a 59,029‑question pool from 16 sources plus two new sets (Stanford PhD probability and hard brain‑teasers) (Table 7; §2.1). For every question, obtain a reasoning trace + solution by distilling from Gemini 2.0 Flash Thinking (§2.1).
   - Decontaminate against evaluation sets (MATH500, GPQA Diamond, AIME24) by 8‑gram overlap and deduplicate (§2.1; C.5).
   - Filter to 1,000 examples using three principles (Figure 2 left; §2.2):
     - Quality: drop formatting/response errors; keep 51,581 of the pool; also manually select 384 perceived high‑quality samples (C.4).
     - Difficulty: remove items solved by either of two strong base models (Qwen2.5‑7B/32B Instruct) and prefer questions with longer distilled traces; this yields 24,496 candidates (§2.2; C.3).
     - Diversity: classify questions into 50 domains (Math Subject Classification plus sciences) with Claude 3.5 and sample across domains while biasing toward longer traces (Algorithm 1; §2.2; C.4).
   - Note: not all distilled traces are correct (53.6% correct in s1K; 63.0% in the updated s1K‑1.1; §2.2; §A).

2) Fine-tune a base model with minimal SFT
   - Base: `Qwen2.5‑32B‑Instruct` (§4.1).
   - Train only on reasoning and answer spans (no loss on the question) with delimiters, 5 epochs, batch size 16, lr=1e‑5 with cosine decay, AdamW, bf16, sequence length large enough to avoid truncation (D. Training details).
   - Cost: 26 minutes on 16×H100 (≈7 H100 GPU hours), producing `s1‑32B` (§4.1; Figure 9).

3) Control thinking at inference with `budget forcing` (§3.1)
   - To enforce a maximum budget: when the current thinking length exceeds a limit, append the end‑of‑thinking delimiter (and optionally “Final Answer:”), which immediately pushes the model into answer mode (§3.1).
   - To enforce a minimum/extend thinking: suppress the delimiter and append a short nudge like “Wait” to the current trace, prompting self‑reflection and more steps (§3.1; Figure 3).
   - Practical observation: repeated suppression (2×/4×/6×) extends thinking and can improve accuracy, but too many suppressions can induce loops (§4.2; Figure 4a).

4) Define metrics for evaluating test‑time scaling (§3.2)
   - `Control`: fraction of runs that stay within a target compute budget (higher is better; 100% ideal).
   - `Scaling`: average slope of accuracy vs. thinking tokens across several budgets (must be positive to claim scaling).
   - `Performance`: the best accuracy achieved across the budgets.
   - These provide a principled way to compare different compute‑control methods.

5) Baselines against budget forcing (§5.2; E.1–E.2)
   - Token‑conditional control: put a token budget in the prompt (e.g., “Think for up to 2048 tokens”).
   - Step‑conditional control: instruct a number of steps; model counts down in steps; each step ~100 tokens.
   - Class‑conditional control: generic “short thinking” vs “long thinking” prompts.
   - Rejection sampling: resample until a generation fits a desired length (oracle posterior-by-length).
   - Parallel scaling: majority voting of the base model and tree search using REBASE with a process reward model (Figure 7; §6.2).

Design choices and why they were made
- Small SFT with curated diversity/difficulty: the model has seen massive generic pretraining; a tiny, carefully selected set can “activate” reasoning and align the model to use explicit thinking traces (§6.1; echoes LIMA).
- Sequential scaling via budget forcing: later tokens can refine earlier thoughts; forcing provides perfect control and a clear monotonic trend (Tables 3, 12–13; Figure 4a).
- Avoid pure prompt-based length control: unreliable token counting and step hacking undermine control and scaling (Tables 12–13).
- Minimal strings to extend thinking: short signals like “Wait” work best among tried options (Table 4).

## 4. Key Insights and Innovations
1) A tiny, diverse, hard dataset is enough to unlock reasoning with SFT
   - Innovation: `s1K` is only 1,000 CoT pairs but chosen by a principled three‑axis filter (quality, difficulty via model‑solvability and trace length, and topic diversity across 50 MSC domains; §2.2; Figure 2 left; Algorithm 1).
   - Significance: Compared to training on the full 59K pool, `s1K` is nearly as strong while being ~56× cheaper to train (7 vs. 394 H100 GPU hours) and clearly better than naive selections (Table 2).

2) `Budget forcing`: a trivial decoding control that yields clean, controllable test‑time scaling
   - Innovation: Stop thinking by injecting an end‑of‑thinking token; extend thinking by suppressing it and appending “Wait” (Figure 3; §3.1).
   - Significance: Delivers 100% control, positive scaling slope, and best accuracy among tested methods (Table 3). It also extrapolates beyond the model’s default stop behavior (Figure 4a).

3) A simple, explicit metric suite for test-time scaling
   - Innovation: `Control`, `Scaling`, `Performance` (Eqns. 1–3; §3.2) define what “good scaling” means and allow apples‑to‑apples comparisons.
   - Significance: Reveals, for instance, that rejection sampling can exhibit inverse scaling on AIME24 (accuracy decreases as allowed length grows), a non‑obvious failure mode (Figure 6; §5.2).

4) Practical observations that inform future system design
   - Longer training sequence length yields shorter, better test‑time thinking (the model learns to switch to answer earlier), improving both accuracy and inference cost (Table 8; D.1).
   - Simple class‑conditional prompts (“short/long”) influence length but give mediocre control and limited gains (Table 14).
   - Sequential scaling outperforms parallel majority voting at equal budgets (Figure 4b), while adding structured parallel search (REBASE) can extend scaling further at extra cost (Figure 7).

## 5. Experimental Analysis
Evaluation methodology
- Benchmarks (all accuracy/pass@1; temperature 0 unless noted) (§4.1–4.2):
  - `AIME24`: 30 competition math problems; integer answers 000–999; some require figure inputs (Asymptote) (§4.1).
  - `MATH500`: 500 problems (subset defined by OpenAI) (§4.1).
  - `GPQA Diamond`: 198 PhD‑level science questions (§4.1).
- Systems compared (Table 1): o1 family (API only), Gemini 2.0 Flash Thinking (API), DeepSeek r1 and r1‑distill, QwQ‑32B, Sky‑T1, Bespoke‑32B, the base `Qwen2.5‑32B‑Instruct`, and `s1‑32B`.
- Implementation notes: vLLM serving; known nondeterminism in some settings; they mitigate by using full precision for final runs (Appendix B).

Main quantitative results
- Overall performance frontier (Table 1; Figure 2 right):
  - `s1‑32B` (trained on 1K examples) achieves 56.7 (AIME24) / 93.0 (MATH500) / 59.6 (GPQA Diamond).
  - Versus base `Qwen2.5‑32B‑Instruct`: +30.0 points on AIME24 (26.7→56.7), +9.0 on MATH500 (84.0→93.0), +10.6 on GPQA (49.0→59.6).
  - Versus `o1-preview`: higher on AIME24 (56.7 vs 44.6) and MATH500 (93.0 vs 85.5); lower on GPQA Diamond (59.6 vs 73.3).
  - Sample efficiency: r1‑distill uses ~800K examples; `s1‑32B` uses 1K yet lies on the sample‑efficiency Pareto frontier (Figure 2 right).
- Test‑time scaling behavior (Figures 1, 4a):
  - As average thinking tokens increase, accuracy rises on AIME24 from ~50% (no extra forcing) to 57% when suppressing the stop delimiter up to 6× (Figure 1 middle; Figure 4a).
  - Gains flatten with too many suppressions due to repetitive loops (Figure 4a).
- Sequential vs. parallel scaling (Figure 4b; Figure 7):
  - Majority voting with the base Qwen2.5‑32B, even up to 64 samples, lags `s1‑32B` with budget forcing, supporting the value of sequential reasoning (Figure 4b).
  - Adding REBASE (tree search with a process reward model) on top of `s1‑32B` scales better than majority voting and, in this setting, even beyond the sequential‑only curve, at the cost of extra reward‑model compute (Figure 7; §6.2).
- Method ablations for compute control (Table 3):
  - `Budget forcing`: Control 100%, positive scaling (15), best AIME24 peak 56.7.
  - Token‑conditional: without forcing, only 40% control and negative scaling (−24); with forcing, control 100% but lower performance (40.0).
  - Step‑conditional: mediocre control and small scaling; model compensates by inflating tokens per step (Table 13).
  - Class‑conditional: some scaling with “long” prompt, but only 50% control and lower peak (36.7) (Table 14 and Table 3).
  - Rejection sampling: 100% control but strong negative scaling (−35), with concrete examples where longer samples correlate with backtracking and worse correctness (§5.2; Figure 6; E.2).
- Data ablations (Table 2):
  - Random 1K or purely diverse 1K perform poorly on AIME24 (36.7 and 26.7 vs 50.0 for `s1K`).
  - Longest‑trace 1K helps GPQA (59.6) but still underperforms on MATH500 and AIME24.
  - Full 59K training yields 53.3/92.8/58.1 (AIME/MATH/GPQA) but costs ~394 H100 GPU hours, while `s1K` is nearly as good at ~7 hours.
- Extrapolation string (“Wait”) (Table 4):
  - Suppressing stop 2× with “Wait” improves AIME24 from 50.0 to 53.3 and GPQA from 57.6 to 59.6, better than no string or alternatives like “Alternatively” or “Hmm”.
- Training sequence length matters (Table 8; D.1):
  - Training with long sequences (32,768) vs short (4,096) improves AIME24 from 30.0% to 50.0% and reduces thinking tokens dramatically (20,721 → 6,984 on AIME24), indicating the model learns to answer earlier.

Do the experiments support the claims?
- Yes for the core claims:
  - Monotonic test‑time scaling under budget forcing is demonstrated with explicit curves and metrics (Figures 1, 4a; Table 3).
  - Sample‑efficient SFT on 1K curated traces yields large gains over the base model and competitive scores with far less data (Table 1; Table 2; Figure 2 right).
- Caveats:
  - On GPQA Diamond, `s1‑32B` trails closed o1 and open r1 (Table 1).
  - Scaling via repeated “Wait” suppressions eventually stalls or loops (Figure 4a).
  - vLLM nondeterminism is acknowledged; mitigations are described (Appendix B).

Updates (Appendix A)
- A follow‑up `s1.1` regenerates the 1K traces with DeepSeek r1 and improves: e.g., MATH500 up to 95.4, GPQA to 63.6, AIME24 56.7; on the harder AIME 2025 set, `s1.1` reaches 50.0 with budget forcing (Table 5).

## 6. Limitations and Trade-offs
- Assumptions and scope
  - The method assumes access to an initial distilled “thinking” dataset from a strong teacher model (Gemini Thinking, then r1 for s1.1), and that the base model already has general reasoning priors from pretraining (§2.1; §6.1).
  - Reasoning traces can be incorrect; curation accepts some noise (53.6% correct in s1K; §2.2). The approach focuses on eliciting the reasoning process rather than perfect step‑level correctness.
- Where it may not work well
  - Knowledge‑heavy or cross‑domain scientific reasoning where correctness depends on specific facts (GPQA gap vs. o1; Table 1).
  - Very long chains can hit context limits; sequential scaling alone collapses when the window is exceeded (12/30 AIME24 questions fail at 512 steps in Figure 7; §6.2).
  - Excessive “continue thinking” suppressions can induce loops with no further accuracy gains (Figure 4a).
- Compute and control constraints
  - Sequential scaling is bounded by the model’s context window; parallel methods like REBASE extend scaling but add reward‑model compute (Figure 7).
  - Prompt‑based token/step control is unreliable without forcing, and step delimiters add nontrivial token overhead (Tables 12–13; E.1).
- Evaluation caveats
  - vLLM serves output with known nondeterminism across batch sizes, continuation, and tensor parallelism; final evaluations use full precision to reduce variance (Appendix B).
  - Gemini API “recitation error” required manual AIME24 evaluation via the web UI; scores for Gemini on MATH500/GPQA are N/A (Table 1; §4).

## 7. Implications and Future Directions
- How this changes the field
  - Demonstrates that controlled, monotonic test‑time scaling does not require massive RL or complex search pipelines: a tiny, carefully selected SFT set plus a decoding trick suffices (Figures 1–2; Table 3).
  - Introduces a simple, standardized metric suite (`Control`, `Scaling`, `Performance`) that can become a common yardstick for compute‑vs‑accuracy studies (§3.2).
  - Reframes the “reasoning model” problem as partly an inference-time control problem, not only a training-data problem.
- Follow‑up research enabled
  - Better extrapolation: combine budget forcing with diversity in continuation cues (beyond “Wait”), or integrate temperature/frequency penalties to avoid loops (§6.2).
  - RL + budget forcing: test whether models trained with RL policies respond more productively to enforced longer thinking (§6.2).
  - Hybrid scaling: orchestrate sequential scaling within each trajectory and parallel tree search across trajectories (Figure 7).
  - Data curation science: automate the “quality–difficulty–diversity” triage and explore teacher mixtures; study how trace correctness/noise affects SFT efficacy (§5.1; §A).
  - Robust control: develop token‑ or step‑aware decoders that truly honor budgets without external forcing (Tables 12–13).
- Practical applications
  - Cost‑aware deployment: set per‑query compute budgets and only extend thinking when needed, improving latency/cost for easy questions while retaining high accuracy on hard ones (Figures 1, 4a).
  - Education and problem solving: competition math and STEM tutoring where reasoning transparency and adjustable depth are valued (Figure 5).
  - Systems with tight context limits: augment sequential scaling with lightweight parallel search or PRMs when longer contexts are required (Figure 7).

> Result highlights to remember:
> - Minimal SFT (1K examples) + budget forcing ⇒ consistent test‑time scaling and strong scores: 56.7 (AIME24), 93.0 (MATH500), 59.6 (GPQA) (Table 1; Figures 1–2).
> - Budget forcing achieves perfect control and best accuracy among tested methods (Table 3); “Wait” is the most effective continuation cue (Table 4).
> - Carefully curated “quality–difficulty–diversity” data selection is crucial; random or single‑axis selection drops AIME24 by 13–23 points (Table 2).
