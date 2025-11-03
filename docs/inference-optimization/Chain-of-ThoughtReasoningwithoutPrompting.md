# Chain-of-Thought Reasoning without Prompting

**ArXiv:** [2402.10200](https://arxiv.org/abs/2402.10200)

## 🎯 Pitch

This paper introduces 'CoT-decoding', a simple yet powerful decoding strategy that enables large language models (LLMs) to perform step-by-step reasoning—known as chain-of-thought (CoT)—without any special prompts or instruction tuning. By inspecting multiple top initial tokens and leveraging answer confidence, the method surfaces latent reasoning paths that standard greedy decoding misses, boosting accuracy on math and commonsense tasks and unlocking a more truthful measurement of LLMs’ intrinsic reasoning abilities. This innovation shifts the focus from prompt engineering to model decoding, lowering deployment barriers and clarifying what pretrained models can accomplish on their own.

---

## 1. Executive Summary (2–3 sentences)
This paper shows that large language models (LLMs) can produce step-by-step “chain-of-thought” (CoT) reasoning without any CoT prompts by changing only how decoding starts. The method, called `CoT‑decoding`, branches on several top first tokens, continues each branch greedily, and then selects or aggregates the branch whose final answer is decoded with the highest confidence; this reliably surfaces hidden reasoning paths and markedly improves accuracy on math and commonsense benchmarks.

## 2. Context and Motivation
- Problem/gap addressed
  - Standard practice elicits reasoning in LLMs through prompt engineering (few-shot or zero-shot CoT prompts) or through instruction tuning with CoT data. Both inject human priors and make it hard to measure a model’s intrinsic reasoning capacity.
  - When queries are posed in direct question-answer (QA) form (no CoT prompt), LLMs often fail on reasoning tasks because greedy decoding tends to jump straight to an answer rather than explore reasoning steps (§2.1; Table 1).
- Why this matters
  - Practical: Removing heavy prompt engineering lowers deployment friction and cost, and enables task-agnostic reasoning.
  - Scientific: A decoding-only probe reveals what pretrained models can already do without additional supervision, clarifying the role of prompts versus model capabilities.
- Prior approaches and their limits
  - CoT prompting and its variants (few-shot, zero-shot, least-to-most, trees of thought) boost reasoning but entangle evaluation with human-designed exemplars and instructions; they also require multiple generations (self-consistency) and careful prompt design (§4 Related Work).
  - Training-based approaches (instruction tuning/distillation) improve reasoning but need curated CoT datasets and significant compute.
  - Standard decoding strategies (greedy, top-k/p sampling, beam search) optimize fluency/diversity rather than reasoning quality; they underperform on reasoning without prompts (Table 4).
- Positioning
  - The paper reframes the problem as a decoding search: CoT paths already exist among alternative early continuations. By exploring and then scoring these paths with a confidence metric tied to the final answer, the method uncovers the model’s latent ability to reason—without changing the prompt or the model.

## 3. Technical Approach
CoT-decoding is a decoding-time search and selection procedure. Key terms:
- `decoding path`: the full continuation generated when decoding from a chosen first token.
- `greedy decoding`: at each step pick the single highest-probability token.
- `top‑k alternative tokens`: the k highest-probability choices for the first token.
- `answer span`: the segment of the model’s output that encodes the final answer (e.g., the last number in a math solution or “even/odd” in parity tasks).
- `answer confidence (Δ)`: for an answer, the average margin between the highest- and second-highest token probabilities at each answer token position; larger margins indicate sharper preference.

Step-by-step procedure (§2.1–2.2; Fig. 1; Fig. 2):
1. Input formatting
   - Wrap the question in a minimal QA format: `Q: [question]\nA:` so the model answers rather than continues the question (§2.1, footnote 1).
2. Early branching (the core novelty)
   - At decoding step 0 (first output token after “A:”), retrieve the `top‑k` tokens (default k=10).
   - For each of these k choices, start a separate decoding path and continue that path with standard greedy decoding until completion (§2.1; Fig. 2 shows why branching early matters).
3. Identify the answer span for each path (§2.2 “Identify the answer spans”)
   - Math-style tasks: extract the last number or, for some models, append “So the answer is” to align and capture the explicit answer continuation.
   - Choice-style tasks: extract the final option (e.g., even/odd; yes/no).
4. Compute answer confidence Δ for each path (§2.2)
   - Intuition: if the model “knows” the answer it will produce those answer tokens with high certainty; the probability gap between the top-1 and top-2 token at each answer position will be large.
   - Formula (plain language): for each answer token, compute (probability of the chosen top token − probability of the runner-up token). Average these margins across all tokens of the answer string to get a single confidence score Δ for that path.
   - This is a “minimum-margin” style measure; alternatives like raw probability or entropy were found less reliable (§2.2; footnote 2).
5. Select or aggregate answers (§2.2 “Aggregation of the decoding paths”; Table 7)
   - Max-path selection: choose the single decoding path with the largest Δ; this often picks a CoT path (88% of the time among top‑10 paths on a 100‑question GSM8K sample, §2.2).
   - Weighted aggregation: group paths by their final answer and sum their Δ scores, then pick the answer with the largest sum. This stabilizes results when small logit differences flip the top path.
6. Optional variations and explored alternatives
   - Branching later in the sequence is less effective because early mistakes constrain later tokens (Fig. 2).
   - Simple sampling (top‑k / nucleus / temperature) without prompts does not reliably surface CoT paths because the first token remains low-diversity and the model still aims for a direct answer (Table 3; Table 4).
   - Combining with CoT prompts further improves results; the same Δ-based aggregation remains competitive with or better than self-consistency (§3.3; Table 7).

Why this design?
- Early branching injects diversity exactly where paths diverge between “direct answer” and “begin reasoning” continuations (Fig. 1 and Fig. 2).
- Δ targets certainty at the decisive locations (answer tokens), not global path probability or length, which can be misleading (Table 2 shows length-normalized or raw log-prob ranking underperform).

## 4. Key Insights and Innovations
- CoT paths exist without prompting; greedy decoding hides them (§2.1; Table 1; Fig. 1)
  - Novelty: reframes reasoning elicitation as decoding search rather than prompt design or training.
  - Significance: demonstrates intrinsic reasoning capability in pretrained LLMs; prompts are not strictly required to generate reasoning, only to surface it.
- Answer-confidence (Δ) is a reliable signal for CoT path selection (§2.2; Table 1; Table 2)
  - Different from prior “majority vote” (self-consistency), which fails when the majority answer is wrong or when CoT paths are a minority.
  - Empirical evidence: choosing the highest-Δ path among top‑10 yields 72.0% vs 44.0% greedy on a GSM8K subset, and 95.0% vs 57.0% on Year Parity (Table 2).
- Early-step branching is the right lever (Fig. 2)
  - Insight: The first token largely determines whether the model commits to a direct answer or starts a reasoning narrative. Branching later inherits earlier mistakes and narrows search.
- A general, task-agnostic decoding recipe
  - Works across model families and scales (Fig. 3, Fig. 4), requires no model changes or CoT data, and can be combined with prompting for further gains (Table 7).

## 5. Experimental Analysis
Evaluation methodology
- Input format: all tasks use the bare QA format `Q: [question]\nA:` to avoid prompt-induced priors (§3).
- Models: `PaLM‑2` (XS/Small/Medium/Large; pretrained and instruction‑tuned), `Mistral‑7B` (pretrained and instruction‑tuned), and `Gemma‑7B` (pretrained) (§3 “Models”).
- Datasets/tasks (§3 “Datasets”)
  - Mathematical: GSM8K (grade-school word problems) and MultiArith (multi-step arithmetic).
  - Commonsense: Year Parity (“Was [person] born in an even or odd year?”).
  - Symbolic/synthetic: Coin Flip, Web of Lies, Multi-step Arithmetic from Big-Bench-Hard; plus two Big-Bench tasks (Sports Understanding and Object Counting).
- Decoding setup
  - Default k=10 branches at the first token; greedy continuation thereafter; answer-span extraction via last number or “So the answer is” alignment; Δ computed over answer tokens (§3; §2.2).

Main quantitative findings
- CoT-decoding vs standard decoding methods (Mistral‑7B, GSM8K; Table 4)
  > “Greedy decoding 9.9%; Top‑k sampling 4.9%; Nucleus sampling 6.4%; Beam search 6.7%; Temperature 7.5%; Self-consistency w/o CoT prompt 12.9%; CoT‑decoding (k=10) 25.1%.”
  - Interpretation: among decoding-only baselines without prompts, CoT‑decoding is the only method that substantially improves reasoning accuracy.
- Across model families (Fig. 3)
  - The bar plots show consistent, often large, gains on GSM8K, MultiArith, and Year Parity for `Mistral‑7B`, `Gemma‑7B`, and `PaLM‑2 Large`. The exact numbers vary by model/task, but the visual deltas are large and consistent.
- On PaLM‑2 Large, no-prompt results and ablations
  - Path-selection criteria (Table 2)
    > Ranking 10 paths by “highest log-prob”: 37.0% (GSM8K subset) / 55.0% (Year Parity); length-normalized log-prob: 51.0% / 57.0%; Δ confidence (CoT‑decoding): 72.0% / 95.0%.
    - Conclusion: Δ is a far better discriminator of correct paths than path probability or length.
  - Sampling vs CoT‑decoding without prompts (Table 3)
    > Self-consistency (10 samples) without CoT prompt: 40.6% on PaLM‑2 L vs CoT‑decoding 63.2% on GSM8K.
- Scaling behavior (PaLM‑2; Fig. 4)
  - GSM8K: CoT‑decoding improves accuracy by +10–30 points across XS→Large, and nears instruction‑tuned performance at Large (63.2% vs 67.8% IT).
  - Year Parity: Greedy remains near chance even as scale grows; CoT‑decoding climbs toward 95% at Large—evidence that failure under greedy decoding masked existing capability.
- Effect of k (number of first-token branches) (Fig. 5)
  - Larger k generally improves accuracy, especially on harder synthetic tasks; instruction‑tuned models benefit less because their CoT paths are already promoted into top few continuations.
- Synthetic reasoning stress tests (PaLM‑2 L; Table 6)
  - Gains persist but diminish as task complexity rises.
  > Example: Coin Flip 2‑rounds 70→94% (greedy→CoT-decoding), but at 4‑rounds only 48→55%.
  - Observation: as the required number of state updates or compositional steps grows (e.g., Multi-step Arithmetic with deeper/longer expressions), correct CoT paths become rarer or lower ranked; larger k helps but doesn’t fully close the gap (Fig. 5 right).
- Combination with CoT prompts (GSM8K; Table 7)
  > Without prompting: Greedy 34.8% (PaLM‑2 L), CoT‑decoding 63.2%;  
  > With zero-shot CoT prompt: Self-consistency 85.3%; CoT‑decoding (agg) + zero-shot CoT prompt 87.0%.
  - Takeaway: CoT‑decoding complements prompting; Δ‑weighted aggregation outperforms majority voting at similar compute.
- Instruction‑tuned models also benefit (Mistral‑7B; Table 5)
  > GSM8K IT: 31.2→38.2% (+7.0); MultiArith IT: 37.8→66.5% (+28.7); Year Parity IT: 62.2→73.5% (+11.3%).
  - Meaning: even after training to produce CoT, the first token can still nudge toward direct answers; early branching plus Δ rescues missed CoT paths.

Qualitative evidence and mechanism checks
- Fig. 1 and Table 1 show concrete paths where the greedy output is wrong/terse, but an alternative first token triggers a full reasoning chain that leads to the correct answer; paths with CoT show markedly higher Δ on the final answer.
- Manual audit on the first 100 GSM8K questions: choosing the path with highest Δ among top‑10 yields a CoT in 88% of cases (§2.2).

Overall assessment
- The experiments substantiate the central claims: (i) alternative first tokens frequently initiate latent CoT trajectories; (ii) answer-token confidence correlates with correct, reasoned paths; (iii) this holds across multiple models and tasks; and (iv) gains are largest where greedy decoding is brittle (e.g., parity queries).

## 6. Limitations and Trade-offs
- Assumptions and dependencies
  - Requires being able to identify an answer span. For open-ended answers, Δ over the final span can be noisier (§2.2; Discussion).
  - The method hinges on a correlation between CoT presence and higher answer-token confidence; while strong empirically, it’s not guaranteed in all settings.
- Coverage and task scope
  - On highly synthetic or compositional tasks requiring 3+ precise state updates, correct CoT paths become rare or deeply ranked; accuracy gains shrink (Table 6; Fig. 5 right).
  - Some tasks reveal persistent weaknesses such as state tracking errors (Coin Flip) or left-to-right arithmetic instead of respecting operator precedence (§3.2 and Table 11).
- Computational cost
  - Complexity grows linearly with k: generating k branches is O(k) compared to O(1) for greedy (Table 7 “Compute” column).
  - Practical overhead: longer decoding budgets and filtering of malformed outputs are sometimes needed (§D “Details on Experimental Settings”).
- Stability and edge cases
  - Branching later in the sequence is less effective and more expensive (Fig. 2).
  - On public models, alternative paths can sometimes regurgitate training-like text; the paper filters such outputs (Appendix D), but this is a general concern in multi-path decoding.

## 7. Implications and Future Directions
- How this changes the landscape
  - Demonstrates that reasoning is partly a decoding/search problem, not just a prompting or training problem. This reframing encourages lightweight, model-agnostic methods to elicit capabilities that are already present.
  - Provides a clean probe of “intrinsic” reasoning by minimizing human priors in prompts, enabling more faithful capability assessment (§3.2).
- Practical applications
  - Drop-in improvement in QA systems where prompts must stay concise or standardized (customer support, educational assistants, structured forms).
  - Safety and reliability: path-level Δ provides a confidence signal tied to answer tokens, useful for abstention or routing.
  - Low-resource settings: boosts reasoning without additional data or fine-tuning; can be combined with prompts or training when available (Table 7).
- Research directions
  - Search strategies: adaptive k, learned first-token selectors, or limited branching beyond step 0; balancing accuracy with compute (Discussion §5).
  - Better answer-span detection for open-ended outputs; leveraging structure-aware decoders.
  - Training with CoT-decoding traces: fine-tune to promote high‑Δ CoT paths to the top-1 greedy path, reducing the need for multi-branch decoding (Discussion).
  - Efficiency: combine with speculative decoding or distillation to cut O(k) overhead (§4 Decoding for efficiency; future work suggestion).
  - Diagnostics: use CoT-decoding to map failure modes (e.g., state tracking) and to study how pretraining distributions shape which CoTs are readily available (§3.2; McCoy et al. 2023 discussion).

In short, CoT‑decoding is a simple, general, and well-evidenced method that uncovers latent reasoning in pretrained LLMs by searching the earliest branching point and selecting answers via a targeted confidence metric. Figures 1–5, Tables 1–7, and the appendices together show both why it works (early-branching plus answer confidence) and where it breaks (complex synthetic tasks and open-ended answers), setting a clear agenda for decoding-aware reasoning research.
