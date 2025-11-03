# Chain‑of‑Thought Reasoning Without Prompting

**ArXiv:** [2402.10200](https://arxiv.org/abs/2402.10200)
**Authors:** Xuezhi Wang, Denny Zhou
**Institutions:** Google DeepMind

## 🎯 Pitch

CoT-decoding revolutionizes the way large language models execute reasoning by uncovering latent paths through a simple, prompt-free branching mechanism, significantly enhancing accuracy on challenging reasoning benchmarks. This approach not only eliminates the need for costly prompt engineering but also offers a clearer assessment of a model's inherent reasoning capabilities, enabling more efficient and reliable AI deployments in practical applications.

---

## 1. Executive Summary
This paper introduces CoT‑decoding, a decoding-time method that elicits chain‑of‑thought (step‑by‑step) reasoning from large language models without any prompting or fine‑tuning. By branching on the first output token, then selecting among the resulting continuations using an answer‑confidence score, the method uncovers latent reasoning paths and substantially improves accuracy on math and commonsense benchmarks across several model families and sizes.

## 2. Context and Motivation
- Problem addressed
  - Most current methods to make language models “reason” depend on crafted prompts (few‑shot/zero‑shot chain‑of‑thought) or instruction‑tuning on large corpora of explanations. This makes it hard to measure a model’s intrinsic reasoning ability and is costly to deploy.
  - When asked to answer directly in a plain QA format, models often respond with a short guess and perform poorly on reasoning tasks.

- Why this matters
  - Practical: Reduces dependence on labor‑intensive prompt engineering and expensive supervised tuning.
  - Scientific: Separates what the model can already do from what carefully designed prompts “teach” it, enabling a fairer assessment of intrinsic capability.

- Prior approaches and their limits
  - Prompting: Few‑shot CoT (e.g., Wei et al., 2022) and zero‑shot CoT (Kojima et al., 2022) rely on human examples or specific phrasings; performance varies with prompts and tasks (§4).
  - Self‑consistency: Sample many CoT responses and majority‑vote (Wang et al., 2023a); still depends on prompts.
  - Instruction‑tuning with CoT data: Effective but costly and still injects human priors into behaviors (§1).

- Positioning
  - This paper shows that much of the observed “lack of reasoning” under direct QA is an artifact of greedy decoding. Without changing the prompt or training, simply exploring alternative first tokens can reveal existing reasoning paths (§2.1, Fig. 1, Table 1).

## 3. Technical Approach
CoT‑decoding is a pure decoding‑time procedure. The input format is the simplest possible QA pattern, `Q: [question]\nA:` (§3).

Key terms (defined when first used here):
- `decoding path`: The full sequence of tokens produced when the model generates an answer.
- `greedy decoding`: Always pick the highest‑probability next token.
- `top‑k token`: One of the k most probable next tokens at a decoding step.
- `chain‑of‑thought (CoT)`: A multi‑step, explicit reasoning trace produced in natural language.

Step‑by‑step method:
1. Early branching at the very first output token (§2.1, Fig. 2).
   - Instead of using the single top‑1 token, enumerate the top‑k alternatives at the first decoding step (default k=10).
   - For each chosen first token, continue the rest of the generation greedily to obtain k distinct decoding paths (plus the k=0 greedy path).

2. Identify the answer span in each path (§2.2, “Identify the answer spans”).
   - For math: extract the final number or, for PaLM‑2 experiments, append “So the answer is” and align the continuation to identify the answer.
   - For categorical tasks (e.g., “even/odd”), parse the final option.
   - For yes/no tasks in symbolic settings, sum probabilities over the valid labels; ignore invalid outputs.

3. Score each path by answer confidence using a margin‑based metric (§2.2; blue numbers in Table 1).
   - Intuition: If a path contains a genuine reasoning process, the model will be more decisive when emitting the final answer tokens.
   - Formalization: For each answer token in a path, compute Δ (delta) = probability(top token) − probability(second token). Average these margins across all answer tokens to get the path‑level answer confidence, written as `Δ_k,answer`.
   - This “minimum‑margin” style score is robust compared to using raw token probability or entropy (§2.2).

4. Select or aggregate paths (§2.2, “Aggregation of the decoding paths”).
   - Max‑path selection: Output the answer from the path with the highest `Δ_k,answer`.
   - Weighted aggregation: Sum the Δ values for identical answers across paths and choose the answer with the largest total Δ (denoted ˜Δ_a = Σ_k Δ_k,a). This reduces sensitivity to small logit differences.

Why early branching?
- Fig. 2 shows that branching at the very first token maximizes path diversity. If the first token is a direct guess like “5,” later branching rarely recovers a reasoned solution for that path. Some tasks (e.g., “year parity”) may benefit from occasional mid‑path branching, but the default is to branch only at step 1 for simplicity and cost (§2.2).

Why not just sample?
- Without CoT prompts, standard sampling tends to produce direct‑answer first tokens with little diversity. CoT‑decoding enforces diversity exactly where it matters—the first step—so the model can enter reasoning trajectories (§2.2, Table 3).

Complexity and implementation details:
- Complexity is O(k) decoding passes instead of O(1) for greedy (Table 7, rightmost column).
- Default k=10; larger k usually helps, especially on harder tasks (§3.1, Fig. 5).
- Ill‑formed responses are filtered with simple heuristics (App. D).
  
Toy intuition:
- Imagine a math word problem where “A:” could start with “5,” “I,” “We,” or “You.” Greedy might choose “5” and immediately answer incorrectly. CoT‑decoding tries “I/We/You/The…” which often lead to a full sentence and a step‑by‑step explanation that computes the right result (Fig. 1; Table 1).

## 4. Key Insights and Innovations
1. Decoding, not prompting, can elicit latent CoT paths (§2.1; Fig. 1; Table 1).
   - Novelty: Prior work focused on prompt engineering or fine‑tuning. This paper shows that many models already contain reasoning trajectories that are simply not selected by greedy decoding.
   - Significance: It reframes poor direct‑QA performance as a decoding artifact, not a hard limitation.

2. A simple margin metric over answer tokens correlates with CoT presence and correctness (§2.2; Table 1, Table 2).
   - Novelty: Use of average top‑2 probability margins at answer tokens (`Δ_k,answer`) to score confidence, rather than log‑probability of the whole sequence.
   - Significance: On the first 100 GSM8K questions, the top‑Δ path among the top‑10 contained a CoT 88% of the time (§2.2). This makes automated CoT path selection practical without any extra models.

3. CoT‑decoding improves reasoning across models, scales, and tasks (§3.1; Fig. 3, Fig. 4; Table 4, Table 5).
   - Novelty: A prompt‑free procedure that boosts accuracy on math and commonsense tasks for PaLM‑2, Mistral‑7B, and Gemma‑7B.
   - Significance: Often doubles or triples accuracy over greedy, and narrows the gap to instruction‑tuned models.

4. Insights into intrinsic capabilities vs. “taught” behaviors (§3.2; Table 6).
   - Novelty: By removing prompts, the paper probes which tasks/models already contain correct CoT paths in their decoding space.
   - Significance: Correct paths are common for simpler or more natural tasks (e.g., small‑step arithmetic, year parity), but rarer for synthetic, multi‑step symbolic tasks—where few‑shot CoT examples likely play a “teaching” role.

5. Complementarity with CoT prompting (§3.3; Table 7).
   - Innovation: Combine CoT‑decoding with zero‑shot CoT prompts and outperform standard self‑consistency at similar compute via Δ‑based aggregation.

## 5. Experimental Analysis
Evaluation setup (§3; App. D):
- Input format: `Q: [question]\nA:` for all tasks unless unnatural (e.g., raw arithmetic expression).
- Branch size: k=10 by default; early branching at the first output token; greedy thereafter.
- Models: PaLM‑2 (XS, Small, Medium, Large; also instruction‑tuned), Mistral‑7B (pretrained and instruct‑tuned), Gemma‑7B.
- Datasets/Tasks:
  - Math: GSM8K; MultiArith.
  - Commonsense: “Year parity” (query “Was [person] born in an even or odd year?”).
  - Symbolic (BBH and related): Coin Flip; Web of Lies; Multi‑step Arithmetic with varying depth and length.
  - Additional synthetic/natural language: Sports Understanding; Object Counting.

Baselines:
- Greedy decoding, temperature/top‑k/top‑p sampling, beam search (Table 4).
- Self‑consistency with and without CoT prompts (Table 3, Table 7).
- Alternate path selection heuristics: highest log‑prob, length‑normalized log‑prob (Table 2).

Main quantitative findings:
- Path selection metrics (PaLM‑2 L, top‑10 paths) (§2.2):
  > Table 2: On GSM8K (first 100 problems), CoT‑decoding reaches 72.0% vs. greedy 44.0%; on Year Parity, 95.0% vs. 57.0%.

- Prompt‑free decoding strategies on GSM8K (Mistral‑7B pretrained) (§3.1):
  > Table 4: Greedy 9.9%; top‑k sampling 4.9%; top‑p 6.4%; beam 6.7%; temperature 7.5%; self‑consistency w/o CoT prompt (10 paths) 12.9%; CoT‑decoding 25.1%.

- Prompt‑free across models (Fig. 3):
  - Mistral‑7B: GSM8K 9.9% → 25.1%; MultiArith 14.3% → 45.7%; Year Parity 35.0% → 66.0%.
  - PaLM‑2 Large: GSM8K 34.8% → 63.2%; Year Parity ~57% → 95% (see Fig. 4).
  - Gemma‑7B: Similar relative gains (Fig. 3).

- Scaling behavior (PaLM‑2, Fig. 4):
  > Fig. 4 (left): On GSM8K, CoT‑decoding lifts Large from 34.8% (greedy) to 63.2% and brings the pretrained model closer to the instruction‑tuned model (67.8%).  
  > Fig. 4 (right): On Year Parity, accuracy remains flat across scales under greedy but rises to near 95% with CoT‑decoding at Large.

- Instruction‑tuned models also benefit (Table 5):
  > Mistral‑7B Instruct: GSM8K 31.2% → 38.2%; MultiArith 37.8% → 66.5%; Year Parity 62.2% → 73.5%.

- Effect of branching width k (Fig. 5):
  - Larger k generally yields higher accuracy, with diminishing returns for instruction‑tuned models (they already surface CoT in early paths).

- Synthetic tasks probing intrinsic ability (PaLM‑2 L, Table 6):
  > Coin Flip: 2/3/4 rounds—greedy 70/53/48% → CoT‑decoding 94/57/55%.  
  > Web of Lies: 3/4/5 statements—76/58/53.6% → 87/63/57.6%.  
  > Multi‑step Arithmetic: accuracy drops sharply as depth/length increase, but CoT‑decoding still helps (e.g., d2,l4: 0% → 16%).  
  > Sports Understanding: small/no gain (58.8% → 58.0%); Object Counting: modest gain (36.0% → 39.2%).

- Combination with CoT prompting (GSM8K test, Table 7):
  > Mistral‑7B: Zero‑shot CoT 17.5%; self‑consistency w/ zero‑shot CoT 39.4%; CoT‑decoding + zero‑shot CoT (agg) 48.4%.  
  > PaLM‑2 L: Zero‑shot CoT 75.1%; self‑consistency w/ zero‑shot CoT 85.3%; CoT‑decoding + zero‑shot CoT (agg) 87.0%.

Ablations and diagnostic observations:
- Alternative path selection heuristics underperform margin‑based confidence (Table 2).
- Sampling without prompts does not reliably uncover CoT paths (Table 3).
- Early branching is more effective than later branching in most cases (Fig. 2).
- Qualitative examples show CoT‑decoding surfaces “free‑form” reasoning different from prompt‑taught templates (App. A, Table 8), clarifying what the model can do intrinsically.

Assessment of evidence:
- The study spans multiple models and tasks, reports consistent numeric gains, and includes ablations on k, path scoring metrics, decoding strategies, and combinations with prompts. The evidence supports the central claims that (1) latent CoT paths exist and (2) a margin‑based selector can find them reliably. Results are strongest on math and parity; gains are smaller on certain synthetic language tasks (Table 6), which is candidly discussed.

## 6. Limitations and Trade-offs
- Computational cost (explicitly discussed in §5 “Discussion and Limitations”; Table 7):
  - CoT‑decoding requires decoding k paths (O(k) compute), vs. O(1) for greedy. Larger k often helps (Fig. 5) but increases latency and cost.

- Reliance on answer span identification (§2.2; App. D):
  - Requires robust parsing of the final answer tokens. Heuristics (last number, final option) or “So the answer is …” may fail in open‑ended or poorly formatted outputs.

- Confidence metric can be overconfident on wrong answers:
  - Δ is a local margin at the answer tokens. In adversarial or ambiguous cases, the model may be confidently wrong, and Δ will still be high. Aggregation helps but does not guarantee correctness (§2.2, “Aggregation”).

- Task coverage and distributional effects (§3.2; Table 6):
  - CoT paths are less prevalent for complex, synthetic reasoning requiring deep state tracking or strict operator precedence; improvements diminish as depth/length grow (e.g., Multi‑step Arithmetic d2,l4: 0% → 16%).

- Branching only at the first token (current default):
  - While empirically effective (Fig. 2), it may miss mid‑sequence opportunities in tasks where early tokens are not discriminative. The paper notes branching later is possible but more expensive and nontrivial to score (§5 “Discussion and Limitations”).

- Evaluation nuances on factual recall:
  - For year‑parity with smaller open models (Mistral‑7B), the pipeline first queries the model for the birth year and uses that as reference, omitting names it fails to recall (App. D). This avoids noisy labels but changes the task slightly.

## 7. Implications and Future Directions
- How this changes the landscape
  - Establishes decoding‑time search and selection as a viable, prompt‑free route to eliciting reasoning. This helps decouple “knowledge already in the model” from “knowledge injected via prompts,” enabling cleaner diagnostics of intrinsic capabilities.
  - Suggests that some gaps between pretrained and instruction‑tuned models can be closed by better decoding alone (Fig. 4, Table 5), reducing the need for extensive supervised CoT data in certain settings.

- Practical applications
  - Drop‑in improvement for math/commonsense QA systems that must operate with minimal prompt engineering (e.g., embedded assistants, tutoring tools).
  - A fallback strategy: if greedy decoding yields a terse guess, trigger CoT‑decoding to recover reasoned responses and more reliable final answers.
  - A ranking signal: Δ can serve as a light‑weight confidence proxy for final answers in structured tasks.

- Research directions
  - Search and efficiency: Explore multi‑step or adaptive branching, perhaps guided by uncertainty or learned policies; combine with speculative decoding to amortize O(k) cost (§5).
  - Better confidence signals: Investigate richer token‑ or span‑level uncertainty measures, or internal activation‑space signals, especially for open‑ended answers (§5 “Discussion and Limitations”).
  - Process supervision without prompts: Integrate discovered CoT paths to fine‑tune models or to train verifiers/critics that do not depend on handcrafted prompts.
  - Task difficulty curriculum: Use CoT‑decoding to map where latent reasoning exists vs. where explicit “teaching” via exemplars is necessary (Table 6), informing targeted data collection.
  - Robust path aggregation: Extend Δ‑based aggregation with structure‑aware voting, minor perturbations of the question, or step‑wise verification (connecting to step‑checking methods in §4).

In short, CoT‑decoding demonstrates that large language models already contain many reasoning trajectories that greedy decoding fails to expose. A simple, prompt‑free branching-and‑selection scheme reliably surfaces those paths and improves accuracy across diverse settings, while also clarifying where genuine reasoning is present vs. where models still need guidance.
