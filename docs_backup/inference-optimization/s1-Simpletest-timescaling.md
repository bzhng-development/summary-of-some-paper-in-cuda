# s1: Simple test-time scaling

**ArXiv:** [2501.19393](https://arxiv.org/abs/2501.19393)

## 🎯 Pitch

This paper introduces s1, a minimalist yet powerful recipe for enabling test-time scaling in large language models (LLMs), leveraging just 1,000 carefully curated reasoning samples and a simple inference-time method called budget forcing to control and scale the amount of 'thinking' a model does before answering. By fine-tuning a 32B-parameter LLM for only 26 minutes and applying this controllable test-time compute intervention, the open-source s1-32B achieves or surpasses the performance of much larger, closed-source, or reinforcement learning-based systems on challenging math and science benchmarks. This work is significant because it brings highly sample-efficient, controllable reasoning and test-time scaling to the open community, showing that advanced reasoning abilities can be unlocked without massive resources, and that model performance can be predictably improved simply by allocating more compute at inference.

---

## 1. Executive Summary (2–3 sentences)
This paper introduces s1, a simple and open recipe to make large language models (LLMs) reason better by using extra computation during inference (“test-time scaling”). The two core pieces are: a compact, carefully selected 1,000-example reasoning dataset (s1K) for supervised fine-tuning, and a decoding-time method called budget forcing that precisely controls and extends the model’s “thinking” before answering. With only 26 minutes of fine-tuning on a 32B-parameter model and simple test-time control, s1-32B matches or exceeds several closed and open systems on math and science benchmarks while exhibiting clean, monotonic test-time scaling (Figures 1–2, Table 1).

## 2. Context and Motivation
- Problem addressed
  - Many recent reasoning LLMs improve by spending more compute at inference time (“test-time scaling”), but reproducible, open methods that both (a) scale reliably with extra test-time compute and (b) require modest training resources are scarce.
  - OpenAI’s o1 family validates the promise of test-time scaling but does not reveal methodology; several replications require large-scale reinforcement learning or complex search procedures (Section 1).
- Why this matters
  - Practical: If performance can be improved by allocating more compute at inference without retraining, users can trade latency for accuracy on demand.
  - Scientific: Establishing simple, controllable mechanisms clarifies what behaviors enable scaling and how small amounts of supervision can “activate” latent reasoning ability in pretrained LLMs (Sections 1, 6.1).
- Prior approaches and their gaps
  - Reinforcement learning at scale (e.g., DeepSeek R1) achieves strong results but uses millions of samples and multiple training stages (Section 1) and does not openly demonstrate clean, controllable scaling curves.
  - Tree search and multi-agent methods can be complex, require reward models, and add compute overhead (Section 6.2).
  - Prompt-only controls (“think longer”) are hard to control precisely; token/step counting is unreliable (Section 5.2; Tables 12–14).
- Positioning
  - s1 positions itself as the simplest path to reliable test-time scaling: small SFT on a well-chosen 1K set plus a minimal decoding-time mechanism (budget forcing) that yields precise control, clear positive scaling, and competitive accuracy (Sections 2–4; Figure 1, Table 1).

## 3. Technical Approach
This section explains the two main components: the s1K dataset and budget forcing. It also covers training, baselines, and the evaluation protocol.

- What is “test-time scaling”?
  - Increasing the amount of computation during inference (not training) to improve accuracy. Two modes:
    - `Sequential`: later steps depend on earlier ones (e.g., longer reasoning trace).
    - `Parallel`: multiple independent attempts are produced and aggregated (e.g., majority vote) (Section 3.1).
- What is a “reasoning trace”?
  - The model’s intermediate “thinking” text before the final answer. s1 uses explicit delimiters to separate a thinking phase from an answer phase during training and decoding (Section D).
- s1K: reasoning data curation (Section 2)
  1) Initial collection (Section 2.1)
     - Start with 59,029 questions from 16 sources emphasizing quality, difficulty, and topic diversity (Table 7).
     - New additions include Stanford PhD probability exam problems (`s1-prob`) and “hard” quantitative brainteasers (`s1-teasers`).
     - For each question, obtain a reasoning trace and answer by distilling from Google Gemini 2.0 Flash Thinking API (Section 2.1).
     - Decontaminate against evaluation sets using 8-gram overlap and deduplicate (Section 2.1, C.5).
  2) Final 1K selection (Section 2.2)
     - Quality filtering: remove API/formatting issues → 51,581 questions; add 384 high-trust samples (Section 2.2).
     - Difficulty filtering: remove questions solvable by either `Qwen2.5-7B-Instruct` or `Qwen2.5-32B-Instruct`; reasoning length serves as a proxy for difficulty (Section 2.2).
     - Diversity sampling: classify into 50 domains via the Mathematics Subject Classification; sample uniformly across domains but bias toward longer reasoning traces (Algorithm 1; Section 2.2, C.4). The final s1K spans 51 domains with 4.7M tokens (Table 6; Figure 2-left).
  - Rationale: The combination “Quality + Difficulty + Diversity” yields the best results; using any single criterion degrades performance (Table 2).
- Supervised fine-tuning (SFT) (Section 4.1, D)
  - Base model: `Qwen2.5-32B-Instruct`.
  - Training details: 5 epochs, batch size 16, AdamW (β1=0.9, β2=0.95, wd=1e-4), LR 1e-5 cosine decay with 5% warmup, bfloat16, loss only on reasoning + answer (Section D).
  - Sequence length ablation: training with long sequences (32,768) avoids truncating answers and yields both higher accuracy and shorter inference-time thinking (AIME24: 50% vs 30%; Table 8).
  - Compute cost: 26 minutes on 16×H100 (7 H100-hours; Section 4.1; D), highlighting sample and compute efficiency.
- Budget forcing (Section 3.1; Figure 3)
  - A decoding-time control to set a maximum and/or minimum amount of “thinking tokens.”
  - Mechanism for maximum: if the reasoning exceeds a token budget, forcibly end the thinking by appending the end-of-thinking delimiter (and optionally “Final Answer:”) so the model must answer (Section 3.1).
  - Mechanism for minimum/extension: suppress the generation of the end-of-thinking delimiter and append a short string like “Wait” to encourage the model to continue reflecting before answering (Section 3.1; Figure 3 shows a self-correction on “raspberry”).
  - Repeating the suppression multiple times (2×/4×/6×) lengthens thinking even more, but eventually hits diminishing returns or loops (Figure 4a).
- Baselines for controlling test-time compute (Section 3.1; 5.2)
  - Token-conditional control: instruct the model to think up to N tokens (Figure 10-left; Table 12). Result: poor adherence without forcing; with forcing, control improves but scaling remains worse than budget forcing (Table 3).
  - Step-conditional control: instruct the model to think for N steps (about 100 tokens each) using a countdown format (Figure 10-right; Table 13). Result: model “hacks” the budget by lengthening steps; higher overhead due to step delimiters (Section 5.2).
  - Class-conditional control: two generic prompts (“short thinking” vs “long thinking”) (Table 14). Result: rough control, but not precise; improvements are inconsistent.
  - Rejection sampling: repeatedly sample until the trace length fits a target budget; this produces inverse scaling because shorter correct trajectories are filtered out at larger budgets (Figure 6; Section 5.2).
- Evaluation metrics for test-time scaling (Section 3.2)
  - `Control` (Eq. 1): fraction of runs whose thinking tokens fall within prescribed limits (higher is better; 100% is perfect).
  - `Scaling` (Eq. 2): average slope of accuracy vs compute across budgets (must be positive to show true scaling).
  - `Performance` (Eq. 3): best accuracy achieved across budgets.
  - Budget forcing attains perfect control and clear positive scaling on AIME24 (Table 3).

## 4. Key Insights and Innovations
- A minimal, effective recipe for reasoning
  - Training on only 1,000 carefully selected reasoning examples is sufficient to significantly boost a strong base model and enable clean test-time scaling (Figure 2-right; Table 1).
  - Significance: Yields a new point on the sample-efficiency frontier—near closed models on some tasks without large-scale RL or search (Figure 2-right; Section 4.2).
- Budget forcing: a simple, controllable, and effective scaling mechanism
  - Novelty: Rather than asking a model to count tokens or steps, directly manipulate the decoding boundary between “thinking” and “answering” using explicit delimiters and small continuation prompts like “Wait” (Section 3.1; Figure 3).
  - Why it matters: Achieves 100% control and a positive, monotonic scaling curve (Table 3; Figure 1, Figure 4a); avoids the fragility of token/step counting (Tables 12–13) and the inverse-scaling pitfall of rejection sampling (Figure 6).
- Data selection matters more than data volume at this scale
  - Using the three-criterion selection (Quality + Difficulty + Diversity) dramatically outperforms random, diversity-only, or longest-traces-only subsets (Table 2).
  - Training on the entire 59K pool helps only slightly relative to the curated 1K, but costs ~56× more GPU hours (394 vs 7 H100-hours; Section 5.1).
- Clear measurement framework for test-time scaling
  - The Control/Scaling/Performance triad (Section 3.2) provides a concise, comparable way to analyze scaling methods—useful beyond this paper (Section 6.2).

## 5. Experimental Analysis
- Evaluation setup (Section 4.1)
  - Benchmarks
    - AIME24: 30 math competition problems; integer answers (with Asymptote for figures).
    - MATH500: 500 competition math problems (OpenAI’s selected subset).
    - GPQA Diamond: 198 PhD-level science questions.
  - Metric: accuracy (pass@1) with greedy decoding unless stated.
  - Implementation: lm-evaluation-harness; vLLM for inference; full precision to reduce nondeterminism (Section B).
- Main quantitative results (Figures 1–2; Table 1)
  - s1-32B with budget forcing:
    - AIME24: 56.7%
    - MATH500: 93.0%
    - GPQA Diamond: 59.6%
  - Comparisons (Table 1)
    - `Qwen2.5-32B-Instruct` base: 26.7 / 84.0 / 49.0 → s1-32B yields +30.0, +9.0, +10.6 absolute gains.
    - `o1-preview`: 44.6 / 85.5 / 73.3 → s1-32B exceeds on AIME24 (+12.1) and MATH500 (+7.5), trails on GPQA (-13.7).
    - `r1-distill (800K examples)`: 72.6 / 94.3 / 62.1 → s1-32B trails AIME24 (-15.9), nearly matches MATH500 (-1.3), is close on GPQA (-2.5) with 800× fewer finetuning samples.
  - Test-time scaling behavior (Figures 1, 4)
    - As thinking tokens increase, accuracy rises steadily on all three benchmarks (Figure 1). On AIME24, extending thinking by suppressing the end-of-thinking token 2×/4×/6× improves accuracy but eventually flattens and risks loops (Figure 4a).
    - Majority voting on the base model (parallel scaling) cannot match the sequential scaling achieved by s1-32B with budget forcing (Figure 4b).
- Ablations and robustness checks
  - Data selection (Table 2)
    - `1K-random`: AIME24 36.7%; `1K-diverse`: 26.7%; `1K-longest`: 33.3% → all worse than `s1K` (50.0% without extra extrapolation) and the budget-forced variants.
    - `59K-full`: AIME24 53.3% with forcing—slightly better than 1K but with ~56× training compute (Section 5.1).
  - Scaling method ablations (Table 3)
    - Budget forcing achieves perfect `Control` (100%), good `Scaling` (+15 average slope), and best `Performance` (56.7 on AIME24).
    - Token- and step-conditional controls require additional forcing to be usable and still underperform; class-conditional has decent slope but weak control (Table 3; Tables 12–14).
    - Appended string choice matters: “Wait” yields the strongest extrapolation among tested strings (Table 4).
    - Rejection sampling exhibits inverse scaling: higher budgets yield worse accuracy (Figure 6) because longer traces correlate with earlier mistakes and backtracking (§E.2).
  - Training sequence length (Table 8)
    - Longer training sequences reduce inference-time thinking cost and improve accuracy (AIME24: 50% at 6,984 tokens vs 30% at 20,721 tokens).
- Complementary parallel scaling (Figure 7)
  - Combining s1 with REBASE (a process reward model for tree search) scales better than simple majority vote and, at large budgets, can surpass purely sequential scaling; however, it adds reward model compute not fully counted in the token cost (Figure 7).
- Follow-up release s1.1 (Appendix A; Table 5)
  - Regenerating traces with DeepSeek R1 improves results (e.g., MATH500 95.4%, GPQA 63.6%, AIME24 56.7% with “Wait” 2×). This confirms that higher-quality traces in the same simple pipeline produce further gains.

Do the experiments support the claims?
- The claim of a simple, controllable test-time scaling method is supported by:
  - Monotonic accuracy–compute curves (Figure 1) and 100% control (Table 3) with budget forcing.
  - Stable improvements over the base model and parity/lead over several strong systems on math benchmarks (Table 1).
- Caveats:
  - AIME24 has only 30 items; reported differences can be sensitive to decoding nondeterminism (Appendix B).
  - Some comparisons rely on reported numbers for other models; Gemini’s “recitation error” prevented a full comparison (Section 4.1).

## 6. Limitations and Trade-offs
- Heuristic nature of budget forcing
  - Appending “Wait” and suppressing the end-of-thinking delimiter is a heuristic; repeated forcing can cause loops and flattening of gains (Figure 4a). Choice of appended string matters (Table 4).
- Context window constraints
  - Sequential scaling eventually hits context limits; long traces can overflow (Figure 7 notes 12/30 overflows at 512-step prompting).
- Data source and supervision quality
  - s1K distills traces and answers from Gemini Thinking; only 53.6% of s1K items are graded correct, rising to 63.0% in s1K-1.1 (Section 2.2, Appendix A). Noisy supervision empirically helps but may embed model-specific biases.
  - Grading uses another model (Claude 3.5/3.7) and 8-gram decontamination; residual contamination or grading bias cannot be entirely ruled out (Sections 2.1–2.2; C.3; C.5).
- Evaluation and reproducibility issues
  - vLLM nondeterminism and hardware configuration can change outcomes even with greedy decoding; authors mitigate via full precision (Appendix B).
- Scope of generalization
  - Most evidence comes from math and science QA; generality to other domains (e.g., coding with tool use, multimodal reasoning) is promising but not demonstrated here.
- Comparative compute accounting
  - In Figure 7, REBASE’s extra reward-model cost is not included in token counts, complicating apples-to-apples comparisons at high budgets.

## 7. Implications and Future Directions
- How this changes the landscape
  - Demonstrates that minimal SFT plus simple decoding control can yield robust test-time scaling—no massive RL or complex tree search required. This lowers the barrier to building strong “thinking” models and provides a clean benchmark for studying inference-time algorithms (Figures 1–2; Section 6.1–6.2).
  - The Control/Scaling/Performance triad (Section 3.2) provides a common lens to evaluate novel inference-time methods.
- Practical applications
  - Systems that adapt compute to user budgets: quick answers when latency-sensitive; deeper, double-checked reasoning when accuracy-critical.
  - Safer deployment knobs: operators can cap or extend thinking deterministically to manage cost and quality.
- Follow-up research directions
  - Improving extrapolation: smarter forcing strategies (e.g., rotating prompts beyond “Wait,” introducing temperature/frequency penalties) to avoid loops (Section 6.2).
  - Hybrid methods: combine budget forcing with lightweight process rewards or self-evaluation signals for better late-stage correction without heavy tree search.
  - RL + budget forcing: use RL to train models that respond more productively to forced extensions, potentially improving scaling at long horizons (Section 6.2).
  - Data curation science: refine automatic selection for Quality–Difficulty–Diversity; investigate how correctness rate and trace style impact downstream scaling (Table 2; Appendix A).
  - Beyond math/science: apply the recipe to long-form reasoning in law, medicine, and open-domain problem solving, with task-specific thinking delimiters and guardrails.

> Key takeaway: with only 1,000 curated examples and a two-line decoding intervention (“end-of-thinking” delimiter control plus “Wait”), s1 achieves controllable, monotonic test-time scaling and competitive reasoning performance—providing an open, reproducible foundation for the next wave of inference-time algorithms.
