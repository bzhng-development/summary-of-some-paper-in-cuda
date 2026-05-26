# Not All LLM Reasoners Are Created Equal

**ArXiv:** [2410.01748](https://arxiv.org/abs/2410.01748)

## 🎯 Pitch

This paper introduces Compositional GSM, a novel extension of the GSM8K math benchmark designed to rigorously test large language models’ (LLMs) ability to perform true multi-step (two-hop) reasoning, where the output of one problem must be used to solve the next. By evaluating 24 popular LLMs, the authors reveal that many models—especially smaller, cost-efficient, and even math-specialized ones—exhibit a significant 'reasoning gap,' performing well on standard benchmarks but faltering when required to chain simple reasoning steps. This work challenges the prevailing notion that high GSM8K scores equate to genuine mathematical understanding, highlighting crucial weaknesses and providing a more realistic lens for assessing and improving LLM reasoning in real-world, compositional tasks.

---

## 1. Executive Summary

This paper studies the depth of grade-school math (GSM) problem-solving capabilities in LLMs by introducing **Compositional GSM**, a two-hop reasoning benchmark that chains pairs of GSM8K test questions together so that the answer to the first question becomes a variable in the second. Across frontier open-weights and closed models—including Gemini, Gemma2, LLAMA3, GPT, Phi, Qwen2.5, and Mistral families—most LLMs exhibit a substantial **reasoning gap** (the performance difference between solving the compositional pairs and solving each question independently), with smaller, more cost-efficient, and math-specialized models showing disproportionately larger gaps of up to 2–12× worse than their larger counterparts. The gap is not attributable to test-set leakage but rather to distraction from additional context and poor second-hop reasoning, establishing that high scores on standard benchmarks can mask systematic flaws in compositional reasoning that only surface under multi-hop evaluation.

## 2. Context and Motivation

### The Core Problem: High Benchmark Scores May Reflect Pattern Matching, Not Compositional Reasoning

The fundamental question this paper tackles is whether state-of-the-art LLMs have truly "mastered" grade-school math or whether their high benchmark scores primarily reflect sophisticated pattern recognition applied to familiar question formats. As the authors state in Section 1:

> "This apparent mastery of grade-school math problems raises a deeper question: do LLMs truly grasp the underlying concepts or do they mostly rely on superficial pattern recognition?"

This distinction matters enormously for how we deploy and trust these models. A model that achieves 95% on GSM8K by recognizing familiar problem templates can still fail catastrophically when asked to combine two such problems into a novel, multi-step chain — exactly the kind of reasoning that real-world math applications require. The paper argues that standard single-hop benchmarks fundamentally cannot distinguish between these two capabilities, creating a dangerous blind spot in how the field evaluates LLM reasoning.

### Why This Problem Is Important

**Real-world reasoning is inherently compositional.** Most practical mathematical reasoning tasks require composing multiple steps of reasoning together: calculating a tax based on a discount based on a base price, determining travel time accounting for distance and speed under constraints, or solving any multi-variable word problem where intermediate results must be computed first. If models succeed on GSM8K primarily by matching patterns rather than understanding the underlying operations, their apparent proficiency will not transfer to these compositional settings — exactly the problem this paper documents.

**Cost-efficient models drive deployment but may have hidden flaws.** The paper specifically calls out that smaller, cheaper models (GPT-4o mini, Gemini 1.5 Flash, Gemma2-9B) achieve near-parity with their larger counterparts on standard benchmarks while being $25\text{--}35\times$ cheaper. However, as Figure 3 and Figure 4 demonstrate, these cost-efficient models exhibit dramatically larger reasoning gaps on compositional tasks. This has direct economic implications: organizations choosing models based on standard benchmark scores may be selecting systems that are unreliable when facing multi-step reasoning problems in production, despite appearing equivalent in evaluation.

**The overfitting risk in an era of benchmark-driven development.** With the field increasingly optimizing models against specific benchmarks, the paper raises a warning flag about the broader implications of "overtraining" small models on large synthetic datasets (Section 3.4). When fine-tuning on GSM8K data, compositional GSM performance improves initially but then *degrades* with further training while GSM8K performance keeps rising (Figure 7). This suggests that current development practices — heavily weighted toward maximizing standard benchmark metrics — may be systematically trading away compositional generalization for narrow pattern-matching skill.

### Prior Approaches and Where They Fall Short

The paper situates itself against several lines of prior work, each of which touches on the robustness of LLM reasoning but leaves a critical gap:

**1. Test-set leakage and memorization studies.** Zhang et al. (2024) introduced GSM1K, a "held-out" set of grade-school problems, finding that while frontier LLMs showed minimal overfitting, some open-weights models showed systematic degradation, suggesting test-set leakage. Srivastava et al. (2024) similarly used functional variants of MATH problems to detect memorization. The paper directly engages with this concern by showing that most LLMs perform similarly on the original and modified GSM8K questions (Figure 9), establishing that **leakage is not the primary explanation for the reasoning gap** they observe. The modified questions simply substitute a number in the original question — if test-set leakage were the issue, models would perform substantially worse on these variants, but they do not.

What these prior studies miss is that even when models are not memorizing specific test examples, they may still be exploiting superficial distributional properties of the test format — a subtler form of brittleness that only surfaces under compositional evaluation.

**2. Distraction and irrelevant context studies.** Shi et al. (2023) and Levy et al. (2024) demonstrated that LLMs can be easily distracted by irrelevant context within prompts, degrading reasoning performance. The paper confirms this phenomenon as one component of the reasoning gap (Figure 10): several models fail to solve Question-1 correctly when it appears alongside Question-2, even though solving Question-1 does not depend on Question-2 at all. Qualitative analysis reveals that models overlook important details (missing reasoning steps related to time units, omitting arithmetic operations, failing to track "per month" specifications) when processing the longer compositional prompt.

However, distraction alone cannot explain the full gap. As Figure 11 and 12 show, even when models *do* correctly solve Question-1 (surviving the distraction), they still underperform on Question-2 compared to solving it independently — pointing to a second, distinct failure mode.

**3. Multi-hop knowledge retrieval studies.** Press et al. (2023) found that the compositionality gap in GPT-3 models did not decrease as model size increased, and that LLMs can perform first-hop reasoning but fail on the second hop in knowledge retrieval tasks. The paper's findings in Figure 11 directly extend this insight to mathematical reasoning: even when the model has correctly computed the intermediate answer (X), it struggles to substitute that value and solve the second question — something a student who truly understands the math would find straightforward. Yang et al. (2024b) similarly found that LLMs latently fail at multi-hop reasoning.

The limitation of these prior multi-hop studies is that they primarily examined knowledge retrieval (finding and combining facts from text) rather than mathematical reasoning, and they predate the frontier models evaluated here. This paper shows that the multi-hop failure persists even in state-of-the-art systems specifically optimized for math.

**4. Math robustness benchmarks.** Li et al. (2024a) introduced GSM-Plus, a comprehensive benchmark with various perturbations to test robustness. Chen et al. (2023) and Wang et al. (2023) explored semantic substitutions and word-level perturbations. While these works established that LLMs are non-robust to surface-level variations, they typically modify *individual* questions rather than testing the ability to *compose* multiple reasoning steps. The compositional setting is qualitatively different: it tests whether the model can chain operations it "knows" individually, not whether it can handle paraphrased inputs or adversarial distractors.

**5. Compositional generalization research.** A long line of work in machine learning has studied whether neural networks can compose known concepts into novel combinations (Andreas, 2020; Hupkes et al., 2020; Lake and Baroni, 2018). These studies, largely on smaller seq2seq models in synthetic settings, established the fundamental challenge of compositional generalization. More recently, work on in-context compositional generalization (He et al., 2024; Hosseini et al., 2022; Kazemi et al., 2024) has examined whether LLMs can learn to compose skills from examples. The gap this paper fills is empirical: applying the compositional generalization lens to the current generation of frontier LLMs in a realistic mathematical reasoning setting, using GSM8K (a standard benchmark with well-calibrated difficulty) as the building blocks, and benchmarking at scale across more than 20 models.

### How This Paper Positions Itself

The paper explicitly does not position Compositional GSM as "yet another reasoning benchmark" to be optimized against (Section 5):

> "Our objective is not simply to introduce yet another reasoning benchmark, but to provide a case study for deeper insights into LLM reasoning and a reassessment of how we evaluate these abilities."

This framing is crucial. The authors are not proposing that the community "fix" the compositional GSM gap by training on compositional problems. Rather, they use the gap as a diagnostic tool to reveal systematic differences in how models reason — differences that are invisible when looking only at standard single-hop accuracy. The gap serves as a probe, not a target.

The paper positions itself at the intersection of several research threads — robustness evaluation, compositional generalization, and practical deployment concerns — but synthesizes them into a single clear message: the field's current evaluation paradigm, which treats solving individual math problems as sufficient evidence of reasoning ability, is fundamentally inadequate. Models that appear equivalent on GSM8K (within a few percentage points) can differ by $2\text{--}12\times$ in their compositional reasoning gap, revealing latent differences in how they process multi-step problems.

The paper also carves out a distinct empirical niche by systematically comparing how model scale, instruction tuning, math specialization, and code generation interact with compositional reasoning. Prior work either studied single models in depth or compared a narrow set of models. By benchmarking over 20 models spanning seven families and both open and closed source, the paper provides a comprehensive view of how compositional reasoning varies across the current LLM landscape — and establishes that the variation is not noise but systematic, tied to fundamental architectural and training recipe differences.

## 3. Technical Approach

### 3.1 Reader Orientation

This paper introduces **Compositional GSM**, a diagnostic test for evaluating whether LLMs can chain together two grade-school math problems they "know" individually. The core idea is simple: take two GSM8K test questions, make the numerical answer of the first question (`Q1`) a variable in the second question (`Q2`), and ask the model to solve both sequentially. If a model truly understands the underlying math operations, solving the combined problem should be roughly as hard as solving both independently; if it is primarily exploiting surface patterns, performance will degrade substantially because the model has never seen these specific question combinations in training.

The paper does not propose a new training method, architecture, or optimization procedure. Instead, it provides a measurement framework — a constructed test set together with a clear expected-performance baseline — that reveals latent differences in reasoning capability across models that look nearly identical on standard single-hop benchmarks.

### 3.2 Big-Picture Architecture (Diagram in Words)

The system has four major components, arranged in a construction-then-evaluation pipeline:

1. **Question Pairing Engine**: Takes two GSM8K test questions (`Q1` from the original test set, `Q2` from a modified copy) and replaces one number in `Q2` with the variable `X`, where `X` will hold the answer to `Q1`. The pairing respects magnitude constraints so the final answer remains a positive integer close to the original `Q2` answer.

2. **Compositional Test Set Construction Pipeline**: Automates the creation of 1,200 chained question pairs from the GSM8K test set using code-form solution manipulation and GPT-4o/Gemini 1.5 Pro validation, then applies manual curation for the ~25% of questions needing adjustment.

3. **Evaluation Harness**: A fixed 8-shot prompting protocol applied uniformly across all models, with model-specific preamble prefixes where needed for output formatting. The harness runs each model on three test splits: original GSM8K (to measure `S1`), modified GSM8K (to measure `S2`), and compositional GSM (to measure `Scomp`), all at temperature 0 with pass@1 scoring.

4. **Reasoning Gap Metric**: A scalar `Δ = Scomp − S1 × S2` that captures how much a model underperforms (or, rarely, overperforms) relative to the statistically expected performance if solving `Q1` and `Q2` were independent events.

Information flows linearly: raw GSM8K questions → pairing engine → automated validation via code execution and frontier model agreement → manual curation → compositional test set → 8-shot prompting of target model → answer extraction and grading → gap computation per model → comparative analysis across model families, sizes, and training recipes.

### 3.3 Roadmap for the Deep Dive

- **First**, the compositional test set construction pipeline (Section 2 of the paper), because understanding what models are tested on — and how the test set is built to be fair and well-calibrated — is prerequisite to interpreting any results.
- **Second**, the formal definition of the reasoning gap metric `Δ` and the statistical baseline `S1 × S2`, since this metric is the paper's central analytical tool and every subsequent finding is expressed in terms of it.
- **Third**, the evaluation protocol — prompt formats, model selection, sampling parameters, and data splits — because these choices determine what `S1`, `S2`, and `Scomp` actually measure and whether comparisons across models are valid.
- **Fourth**, the analytical decomposition of the gap into two failure modes: distraction (measuring whether adding `Q2` to the prompt degrades `Q1` performance) and second-hop reasoning (measuring whether correctly solving `Q1` is sufficient for solving `Q2` in the compositional context). This decomposition transforms the gap from a single opaque number into a diagnostic tool.
- **Fifth**, the code-generation ablation methodology, which tests whether translating problems into executable Python functions changes the compositional reasoning pattern — providing evidence about whether the gap is tied to natural language processing or fundamental reasoning limitations.

### 3.4 Detailed, Sentence-Based Technical Breakdown

This is primarily a **measurement and analysis paper** whose core idea is that compositional (two-hop) evaluation reveals systematic differences in LLM reasoning ability that single-hop benchmarks obscure. The technical contribution is the construction of a diagnostic test set with a principled expected-performance baseline, together with an analytical framework for decomposing performance failures into distinct root causes.

---

#### Compositional Test Set Construction

The paper constructs a test set of 1,200 compositional math problems, each formed by chaining two questions from the GSM8K distribution. The construction process is designed to ensure three properties: (1) the combined problem is at the same math difficulty level as individual GSM8K questions (only the chaining structure is new), (2) the final answer remains a well-formed positive integer with a magnitude distribution similar to the original test set, and (3) the questions are logically coherent — the numerical substitution in `Q2` does not create nonsensical scenarios.

**Step 1: Selecting question pairs.** The authors start with a subset of 1,200 examples from the original GSM8K test set. Each compositional problem consists of two questions: `Q1` drawn from the original GSM8K test split, and `Q2` drawn from a modified copy of the test split. The pairing is not arbitrary — the choice of `Q1` and the specific number in `Q2` to replace with variable `X` is made such that the new final answer of `Q2` is different from its original final answer and is a positive integer. Additionally, the new answer must not be "too far" from the old answer (the paper verifies this by showing matching answer magnitude distributions in Appendix A, Figure A.1). This constraint prevents the substitution from creating answers that are implausible within the problem's narrative (e.g., a problem about a classroom with 500 students when the original had 25).

**Step 2: Automated answer computation via code-form solutions.** To obtain the new final answer of `Q2` automatically, the paper leverages the code-form solutions from Gao et al. (2023). For each original GSM8K question, a Python code solution exists that computes the answer programmatically. The authors modify these code solutions by replacing the specific number to be substituted with a variable reference, then execute the modified code with the value of `X` (the answer to `Q1`) to obtain the new final answer. This is an exact, deterministic computation — there is no model-generated text involved in computing the ground-truth answer, only program execution.

**Step 3: Validation via frontier model agreement.** Even though the code execution produces a mathematically correct answer, the modified question text might be logically inconsistent or semantically odd. To filter out such cases, the authors generate 16 candidate solutions per modified question from two frontier models: GPT-4o and Gemini 1.5 Pro (presumably 8 from each, though the paper does not specify the split). They then check how many of these 16 solutions agree with the expected final answer from code execution. Questions for which fewer than 4 out of 16 model solutions agree with the expected answer are flagged for manual review.

The threshold of 4 out of 16 (25% agreement) is deliberately lenient — it is not meant to filter out questions that are simply hard for the models, but rather to identify questions where the substitution created a genuinely illogical scenario that would confuse even a capable solver. If the frontier models consistently arrive at the same answer as the code execution, the question is considered well-formed.

**Step 4: Manual curation.** Approximately 25% of the questions (around 300 out of 1,200) required manual checking and modification. The paper states that the authors "checked these questions manually and modified them if needed so that they are logical." The nature of these modifications is not detailed, but they likely involve adjusting the substituted number to better fit the problem narrative or rewording the question text to accommodate the new variable.

**Step 5: Answer format and prompt template.** Each compositional question is presented to the model in a standardized format, shown in Figure 2:

```
Let X be the answer to the Q1:
Q1: [Question 1 text]
Solve it and use the value of X to solve Q2. Explain your answer step by step.
Q2: [Question 2 text with X substituted in place of one number]
```

The instruction explicitly tells the model to compute `X` from `Q1` first and then use it in `Q2`. This format makes the compositional structure explicit — the model is not left to infer that `Q1` and `Q2` are linked; it is told directly. This design choice means any failure to chain the two questions is a failure of execution, not a failure of understanding the task structure. The paper deliberately avoids testing whether models can *discover* that questions are linked, focusing instead on whether they can *execute* the chaining when told to do so.

---

#### The Reasoning Gap Metric

The central metric in the paper is the **compositional reasoning gap**, denoted `Δ`. The definition requires first establishing the expected baseline performance under the assumption that solving `Q1` and `Q2` are statistically independent events.

**Step 1: Measure single-question accuracies independently.** The paper evaluates each model on three separate test splits, each containing 1,200 examples:

- **Original GSM8K test split**: The standard GSM8K test questions. Let `S1` be the model's accuracy on the subset of these questions that serve as `Q1` in the compositional test (i.e., the `Q1` set). This is the probability that the model correctly solves a question when it appears in standard, non-compositional format.

- **Modified GSM8K test split**: The GSM8K test questions after a number has been replaced (the `Q2` set). Let `S2` be the model's accuracy on these questions in standard, non-compositional format. These are the same questions that appear as `Q2` in the compositional test, but here they are presented independently — without `Q1` and without the variable `X`.

- **Compositional GSM test split**: The 1,200 chained question pairs. Let `Scomp` be the model's accuracy on this split. The model is considered correct only if it correctly computes `X` from `Q1` AND correctly uses that value to answer `Q2`. A partially correct answer (right `Q1` but wrong `Q2`, or wrong `Q1` leading to wrong `Q2`) counts as incorrect.

**Step 2: Define the statistical baseline.** If solving `Q1` and `Q2` were independent events — meaning the model's ability to solve each question is unaffected by the compositional format — then the expected accuracy on the compositional split is simply the product of the individual accuracies:

$$S_{\text{expected}} = S_1 \times S_2$$

where `S1` is the accuracy on the `Q1` test set in non-compositional format, and `S2` is the accuracy on the `Q2` (modified) test set in non-compositional format.

**What this product represents:** It is the probability that a model would answer both questions correctly if it encountered them independently and separately, multiplied together under the assumption that the two events are statistically independent. For example, if a model scores 80% on `Q1` problems and 75% on `Q2` problems, the expected compositional accuracy is `0.80 × 0.75 = 0.60`, or 60%.

**Why this form:** The product `S1 × S2` is the null hypothesis. Any deviation from it tells us that the compositional format *changes* the model's behavior relative to solving the same questions individually. If the model's compositional accuracy `Scomp` equals `S1 × S2`, then the chaining introduces no additional difficulty beyond the difficulty of the individual questions. If `Scomp` is lower, something about the compositional format — distraction, second-hop failure, context length, or format shift — is degrading performance.

An alternative baseline would be to simply report `Scomp` without comparison. But that number alone would be uninterpretable: 50% compositional accuracy means very different things for a model that scores 70% on individual questions versus one that scores 99%. The product baseline normalizes for base capability, isolating the *additional* difficulty introduced by composition.

**Step 3: Define the gap.**

$$\Delta = S_{\text{comp}} - S_1 \times S_2$$

where `Scomp` is the model's empirical accuracy on the compositional GSM test set (fraction of 1,200 problems correctly solved), `S1` is its accuracy on the `Q1` set in non-compositional format, and `S2` is its accuracy on the `Q2` (modified) set in non-compositional format.

**What this equation computes:** The difference between how well the model *actually* performs on the compositional task and how well it *should* perform if composition added no extra difficulty. A negative `Δ` (the common case) means the model underperforms expectations — the compositional format is making the problem harder than the individual questions would predict. A positive `Δ` (theoretically possible but rare) would mean the compositional format somehow helps. A `Δ` near zero means the model composes its skills without degradation.

**Why this form:** The subtraction form is chosen for interpretability. A `Δ` of `−0.20` means "the model gets 20 percentage points fewer compositional problems correct than we would expect from its single-question performance." This is more directly meaningful than a ratio or log-odds formulation because it maps directly onto the accuracy scale that practitioners think in. The paper uses percentage points in all reporting (e.g., a gap of "27.5" means `Δ = −0.275`).

**Important caveat about Figure 1's trendline.** Figure 1 plots compositional GSM accuracy (`y`-axis) against the geometric mean of individual accuracies `√(S1 × S2)` (labeled "GSM8K Accuracy" for simplicity on the `x`-axis). The trendline `y = x²` is the expected compositional accuracy under independence, because if `S1 ≈ S2 ≈ x` (so the geometric mean is `x`), then `S1 × S2 ≈ x²`. Models falling below this curve have negative `Δ`; models on the curve have zero gap. The `x`-axis label "GSM8K Accuracy" is a simplification — it is actually `√(S1 × S2)`, which the paper correctly defines in the figure caption but risks misinterpretation in casual reading.

---

#### Evaluation Protocol

The evaluation protocol is designed to be as uniform as possible across models, minimizing confounding variables so that observed differences in compositional reasoning can be attributed to model-level factors rather than evaluation artifacts.

**Test set composition and size.** All three test splits (original GSM8K, modified GSM8K, compositional GSM) contain exactly 1,200 examples each. The paper does not explicitly state the total test set size for the original and modified splits, but the compositional split uses 1,200 pairs constructed from subsets of the other two splits. This is a substantial test set — larger than the standard GSM8K test set of 1,319 examples — providing reasonable statistical power for per-model comparisons.

**Prompting protocol.** Following Zhang et al. (2024), all models are evaluated with a standard **8-shot prompt** — meaning 8 example question-answer pairs are included before the target question to demonstrate the expected format and reasoning style. The prompts for each test split are:

- **Original GSM8K and modified GSM8K**: An 8-shot prompt (Appendix D) consisting of 8 simple GSM-style problems with step-by-step natural language Chain-of-Thought solutions, each ending with "The final answer is [ANSWER]." The 8 examples are identical across all models and correspond to standard GSM8K training examples.

- **Compositional GSM**: An 8-shot prompt (Appendix E) with an analogous structure but adapted for the two-question format. Each example begins with "Let X be the answer to Q1:" followed by Q1, the instruction "solve it and use the value of X to solve Q2. Explain your answer step by step.", Q2 with X substituted, and a solution that first solves Q1, declares "The Q1 answer is [ANSWER1]. Therefore X=[ANSWER1].", then uses X to solve Q2, ending with "The final answer is [ANSWER2]."

The 8-shot format is deliberately simple — the paper explicitly states: "No elaborate prompting method is needed with this format." This is an important design choice: the goal is to measure models' baseline compositional reasoning capability without prompt engineering as a confounding factor. If sophisticated prompting techniques (chain-of-thought variants, self-consistency, etc.) were used, it would be unclear whether observed differences reflect reasoning capability or differential responsiveness to specific prompt formats.

**Preamble prefixes.** Some models required a preamble prefixed to the 8-shot prompt to enforce desired output formatting. For GSM8K (Appendix B), the preamble reads:

```
I am going to give you a series of demonstrations of math Problems and Solutions.
When you respond, respond only with the Solution of the final Problem, thinking
step by step. At the end of the Solution, when you give your final answer, write
it in the form "The final answer is ANSWER."
```

For compositional GSM, the preamble is adapted:

```
I am going to give you a series of demonstrations of compositional math questions
and solutions. Respond by thinking step by step. Solve the first question and
write the intermediate answer as "The Q1 answer is ANSWER1." Then solve Q2. At
the end of the solution, when you give your final answer, write it in the form
"The final answer is ANSWER2."
```

The paper tests models both with and without these preambles and reports the best performance for each model. This is a reasonable choice: if a particular model formats its output correctly without the preamble, the preamble is unnecessary; if it does not, the preamble provides formatting guidance without changing the task. Reporting the best of the two conditions avoids penalizing models for formatting failures rather than reasoning failures.

**Model selection and coverage.** The paper evaluates 23 model variants spanning seven model families, as listed in Figure 3: Gemini (1.0 Pro, 1.5 Flash, 1.5 Pro), Gemma2 (9B-PT, 27B-PT, 9B-IT, 27B-IT), GPT (4o, 4o mini), LLAMA3 (8B-PT, 70B-PT, 8B-IT, 70B-IT), Phi (2, 3-mini-4k-IT), Mistral/Mixtral (Mistral-7B-PT, Mistral-7B-IT, Mixtral-8x7B-PT, Mixtral-8x7B-IT), and math-specialized models (NuminaMath-7B-CoT, Mathstral-7B, Qwen2.5-MATH-7B-IT, Qwen2.5-MATH-72B-IT). The selection includes both pretrained (PT) and instruction-tuned (IT) variants where available, open-weights and closed-source (API) models, and a range of parameter scales from 2B (Phi-2) to 72B (Qwen2.5-MATH) plus unspecified-size frontier models (GPT-4o, Gemini 1.5 Pro).

**Sampling and scoring.** All models are sampled with **temperature 0**, meaning deterministic greedy decoding. This eliminates sampling variance as a confound — each question gets exactly one answer per model, and `pass@1` (the fraction of questions answered correctly on the first attempt) is the accuracy metric. The paper uses the grading function released by Lightman et al. (2022) for GSM8K to determine correctness (Appendix G of the original GSM8K paper, though the paper references it implicitly by using standard GSM8K evaluation protocols).

**Why temperature 0?** Temperature 0 (greedy decoding) tests the model's most confident answer. If models were sampled at higher temperatures with multiple attempts, the reported accuracies might differ, but the choice reflects standard practice for math reasoning benchmarks (where deterministic evaluation is dominant) and focuses the analysis on model capability rather than sampling strategy. The compositional gap measured at temperature 0 represents a lower bound: if a model cannot solve the compositional problem in its most confident mode, it is unlikely to reliably solve it under stochastic sampling.

**Computational cost consideration.** The 8-shot prompts increase context length but not dramatically — each shot is a short grade-school problem and solution, so the total prompt length is manageable for all evaluated models. The paper does not report exact token counts, but GSM8K problems average ~50-100 words, so 8 shots plus instructions constitute roughly 500-1000 tokens of prompt context, well within the context windows of even small modern models.

---

#### Analytical Decomposition of the Reasoning Gap

The paper does not treat the reasoning gap `Δ` as an indivisible quantity. Instead, it decomposes the gap into two distinct failure modes through a series of controlled comparisons, each answering a specific question about how the compositional format affects model behavior.

**Test Set Leakage Check (Figure 9): Does exposure to modified questions alone cause degradation?**

The paper first establishes that the `Q2` (modified) questions are not inherently harder than the original GSM8K questions due to test-set leakage memorization. For each model, accuracy on the original GSM8K test split (`x`-axis) is plotted against accuracy on the modified GSM8K test split (`y`-axis). If models had memorized specific questions during training and the modifications disrupted that memorization, accuracy on modified questions would be substantially lower than on original questions — points would fall below the `y = x` diagonal.

The finding: "Most models are very close to the `x = y` line" (Figure 9 caption), meaning modified questions are solved at roughly the same rate as original questions. This rules out test-set leakage as the primary cause of the compositional reasoning gap. The modified questions are not themselves harder; the difficulty arises specifically from the compositional format — having to solve `Q1` first and carrying the answer forward.

**Distraction Analysis (Figure 10): Does the presence of `Q2` in the prompt degrade `Q1` performance?**

This analysis compares, for each model, the fraction of `Q1` questions solved correctly in the standard non-compositional format (`x`-axis) versus the fraction of the same `Q1` questions solved correctly when they appear as the first question in the compositional format (`y`-axis). Crucially, solving `Q1` in the compositional format does *not* depend on solving `Q2` — `Q1` comes first in the prompt and its answer is computed before `Q2` is addressed.

The baseline expectation is that models should lie on the `y = x` diagonal: if the model can solve `Q1` independently, adding `Q2` to the prompt should not affect `Q1` performance. The paper reports that "several models fall short of this expectation" — they solve `Q1` less often in the compositional format than in the standard format.

Qualitative examination of model responses reveals the mechanism: models "often overlook important details, such as missing a reasoning step related to each in the question or omitting a arithmetic step when the question specifies a month or per month." This is a distraction effect — the presence of a second question, even though it appears *after* the first question in the prompt and does not need to be solved yet, causes the model to process `Q1` less carefully, skipping steps or misreading details.

This finding connects to prior work by Shi et al. (2023) and Levy et al. (2024) on LLM distractibility but applies it specifically to the compositional reasoning setting. The distraction is not from irrelevant content (as in those prior works) but from *relevant future content* that the model should be able to defer processing.

**Second-Hop Reasoning Analysis (Figure 11): Given that `Q1` is solved correctly, can the model reliably solve `Q2`?**

This analysis isolates the second-hop failure mode. For each model, the `x`-axis shows the fraction of `Q2` questions solved correctly in the standard non-compositional format (independently). The `y`-axis shows the fraction of `Q2` questions solved correctly in the compositional format *conditioned on `Q1` being answered correctly* — i.e., among those compositional problems where the model successfully computed `X`, what fraction of the time does it then correctly solve `Q2`?

The statistical expectation is again the `y = x` diagonal: given that the model has correctly computed `X`, solving `Q2` should be no harder than solving it independently with `X` provided. The computation required is identical — substitute the value and perform the remaining arithmetic.

The paper finds that many models fall substantially below the diagonal. Their qualitative analysis reveals that the model "often makes subtle errors and overlooks details" when solving `Q2` after having generated a solution for `Q1`. The model has correctly computed `X`, but in the subsequent reasoning for `Q2`, it makes arithmetic mistakes, misreads the question constraints, or fails to properly incorporate the computed value.

**Capacity Analysis (Figure 12): Is the problem limited capacity to process two questions, or specifically the dependency between them?**

This comparison teases apart whether the failure is due to the sheer length of processing two questions (capacity limitation) or the logical dependency between them (compositional reasoning limitation). For a subset of models, the paper compares `Q2` accuracy in three conditions:

1. **Standard format**: `Q2` alone.
2. **`Q2` with independent `Q1` in context**: The prompt includes `Q1` and its solution, but `Q2` does *not* depend on `Q1`'s answer — the value of `X` is provided in the problem text as a concrete number, and `Q1`'s presence is purely contextual filler.
3. **Compositional format given `Q1` solved**: The standard compositional format, conditioned on `Q1` being correctly solved (same as Figure 11's `y`-axis condition).

The finding: "The distraction from `Q1` in the context is minimal when `Q2` is independent of it. However, when `Q2` relies on the answer from `Q1`, models struggle to solve `Q2` accurately, even if `Q1` has been answered correctly." This result (shown in Figure 12 for four models: LLAMA3-70B-IT, LLAMA3-8B-IT, Gemma2-27B-IT, Gemma2-9B-IT) indicates that the core problem is not simply context length or having two questions in the prompt — it is specifically the *dependency relationship* between the questions. When the model must carry its own computed answer forward and use it in subsequent reasoning, something breaks that does not break when the same number is provided externally.

This is perhaps the paper's most diagnostically valuable finding: the compositional gap is not primarily a capacity or attention problem (the model processes two questions just fine when they are independent) but rather a **state-tracking and variable-binding problem** — the model fails to reliably maintain and reuse the value it computed in `Q1` when working on `Q2`.

---

#### Code Generation Ablation

The paper tests whether translating compositional problems from natural language chain-of-thought into executable Python code changes the compositional reasoning pattern (Section 3.5, Figure 8). This ablation serves as both an alternative evaluation format and a diagnostic tool for understanding where the reasoning gap originates.

**Prompt format.** For code generation, the paper uses a compositional 8-shot prompt (Appendix F) where each example's solution is written as two Python functions:

```python
def solve_q1():
    # Solves Q1, returns the answer

def solution():
    X = solve_q1()
    # Solves Q2 using X, returns the final answer
```

The `solution()` function begins with `X = solve_q1()` to programmatically chain the two computations. The model must generate the code for both functions, and the answer is obtained by executing the code.

**Model coverage.** The code ablation is performed on three families of open-weight instruction-tuned models, comparing the smaller and larger member of each family: LLAMA3 (8B-IT, 70B-IT), Gemma2 (9B-IT, 27B-IT), and Mistral/Mixtral (Mistral-7B-IT, Mixtral-8x7B-IT). This selective coverage (rather than all 23 models) is likely due to the practical requirement that models must be capable of reliably generating executable Python code — some of the evaluated models (particularly smaller pretrained variants) may not have this capability.

**Measurement and comparison.** Compositional GSM accuracy is measured for both natural language Chain-of-Thought (CoT) and code generation formats, and the relative improvement is computed as:

$$\frac{\text{Accuracy}_{\text{code}} - \text{Accuracy}_{\text{CoT}}}{\text{Accuracy}_{\text{CoT}}} \times 100\%$$

where `Accuracy_code` is the fraction of compositional problems correctly solved via code generation and `Accuracy_CoT` is the fraction correctly solved via natural language CoT.

**Key finding and interpretation.** Code generation improves performance for all tested models, but the improvement is dramatically asymmetric across model sizes. For LLAMA3-8B, the relative improvement is 69%; for LLAMA3-70B, it is only 2%. For Gemma2-9B, the improvement is 74%; for Gemma2-27B, it is 27%. For Mistral-7B, the improvement is 149%; for Mixtral-8x7B, it is 71%.

This asymmetry is itself a finding that supports the paper's central thesis about systematic differences in reasoning. If the compositional gap were simply due to a uniform difficulty increase (harder arithmetic, more steps), code generation might help all models proportionally. The fact that smaller models benefit dramatically more suggests that they have a specific deficit in the *natural language execution of multi-step reasoning* that code generation bypasses. When the reasoning is offloaded to a Python interpreter (which handles variable binding, arithmetic, and step ordering deterministically), the smaller model's deficit largely disappears. The larger models, which are better at tracking variables and maintaining state in natural language, benefit less from this offloading.

The paper does not report code generation results for the non-compositional GSM8K baseline, so it is not possible to determine from the presented data whether the code benefit is specific to compositional reasoning or applies to GSM-level math in general. However, the magnitude of the effect for small models (69–149% relative improvement) suggests that compositional reasoning in natural language is where their deficit is most acute.

---

#### Fine-Tuning Overfitting Experiment

The paper includes a controlled fine-tuning experiment (Section 3.4, Figure 7) to test whether training on GSM8K data causes task-specific overfitting — improving standard benchmark performance while degrading compositional generalization.

**Training setup.** The base model is Gemma2 27B PT (pretrained). Fine-tuning is performed on the original GSM8K training dataset under two data conditions:

1. **Human data**: The standard GSM8K training set with human-written solutions.
2. **Synthetic data**: Self-generated solutions from the same model (Gemma2 27B PT) that resulted in correct final answers for GSM8K training queries. The paper generates 10 solutions per training question and retains only those with correct answers, following a rejection sampling approach.

Details of the training procedure are provided in Appendix C: the model is prompted with the 8-shot GSM8K prompt (Appendix D) to generate solutions, and only correct-answer solutions are added to the training set. Evaluation is performed at intermediate checkpoints: 50, 100, and 400 training steps.

**What this experiment tests.** The fine-tuning experiment answers two questions: (1) Does extended fine-tuning on GSM8K improve compositional reasoning at the same rate as single-question reasoning? and (2) Does training on model-generated synthetic data produce different overfitting patterns than training on human-written solutions?

**Key finding.** From Figure 7: at 50 and 100 training steps, compositional GSM accuracy increases along with GSM8K accuracy. But between 100 and 400 steps, GSM8K test accuracy *continues to improve* (from approximately 78% at 100 steps to approximately 84% at 400 steps for human data, and from approximately 82% to approximately 89% for synthetic data), while compositional GSM accuracy *decreases* (from approximately 43% at 100 steps to approximately 37% at 400 steps for human data, and from approximately 49% to approximately 47% for synthetic data). No further improvements on either split are observed after 400 steps.

The synthetic data condition consistently outperforms human data on both splits at all checkpoints, achieving higher accuracy with the same amount of training. The paper hypothesizes that "the trend of using increasingly larger training datasets for over-training small models beyond compute-optimal scaling — often heavily composed of synthetic data — may primarily target performance on standard benchmarks, potentially at the expense of overall generalization."

This is a concrete demonstration of Goodhart's Law in LLM training: when the training objective (or the data generation process) is optimized for performance on a specific benchmark, metric improvement on that benchmark can become decoupled from — and even negatively correlated with — the underlying capability it was intended to measure.

---

#### Summary of Design Choices and Their Justifications

- **Two-hop pairing rather than multi-hop or adversarial construction**: The paper deliberately uses the simplest compositional structure — exactly two questions, explicit instructions about the chaining, and GM8K-level difficulty for each sub-question. This isolates the effect of composition from confounds like instruction-following complexity, escalating difficulty, or adversarial distractors. The choice lets the paper claim that any observed gap is due to the compositional structure specifically, not to increased math difficulty.

- **Statistical baseline `S1 × S2` rather than a fixed accuracy target**: The product baseline normalizes for each model's base capability, making the gap comparable across models with very different GSM8K scores. Without this normalization, high-performing models would appear to have smaller gaps simply because their ceiling is higher, not because they reason better compositionally. The gap metric isolates the *additional* difficulty of composition beyond what single-question performance would predict.

- **Temperature 0, 8-shot, pass@1 uniform protocol**: Eliminates sampling variance, prompt complexity, and decoding-strategy confounds, ensuring that observed differences are attributable to model capability rather than evaluation artifacts. The 8-shot prompt provides sufficient demonstration of the expected format without elaborate prompt engineering.

- **Explicit instruction about the chaining structure**: The prompt template explicitly tells the model "Let X be the answer to Q1" and "solve it and use the value of X to solve Q2." This removes the need for the model to infer that the two questions are linked — the paper is testing execution, not discovery. A model could theoretically fail because it doesn't understand the task structure; this design ensures that failures are reasoning failures, not task-comprehension failures.

- **Code-execution-based ground truth for Q2**: By using Python code execution to compute the correct answer for modified `Q2` questions, the paper obtains exact, deterministic ground truth without model-generated labels. This eliminates any circular dependence on model judgments for "correctness."

- **Frontier model agreement for question validation**: Using GPT-4o and Gemini 1.5 Pro agreement with code-execution answers as a quality filter is pragmatic — it catches illogical substitutions without requiring manual review of all 1,200 questions. The 25% manual review rate is the cost of this automation.

- **Decomposition into distraction and second-hop failure**: Rather than treating the gap as a monolithic quantity, the paper provides a diagnostic framework (Figures 10, 11, 12) that separates two distinct mechanisms. This decomposition is what transforms the paper from a benchmark-introduction exercise into a meaningful analysis of *why* models fail — enabling practitioners to target specific failure modes rather than vaguely "improving compositional reasoning."

- **Gemma2 27B PT for fine-tuning rather than multiple model families**: The fine-tuning overfitting experiment uses a single model to keep the experiment tractable and controlled. The choice of Gemma2 27B PT (a strong pretrained model with room for improvement on GSM8K) provides a clean baseline for measuring the effects of task-specific training without confounds from instruction tuning or prior math specialization.

- **Code generation ablation on three model families, two sizes each**: The selective coverage enables a clean within-family comparison (does code help small models more than large ones in the same family?) while keeping the experiment manageable. The three families represent different training philosophies (LLAMA3: general-purpose, Gemma2: open release, Mistral: mixture-of-experts), providing some diversity in the finding.

## 4. Key Insights and Innovations

### Innovation 1: Compositional Accuracy as a Diagnostic Probe, Not a Benchmark Target

The paper's most distinctive intellectual move is deliberately refusing to frame Compositional GSM as a new benchmark to be conquered. This is not rhetorical positioning—it is a methodological choice with real consequences for how the results should be interpreted and used.

**What makes this distinctive:** Prior robustness benchmarks (GSM1K from Zhang et al., 2024; GSM-Plus from Li et al., 2024a; functional MATH variants from Srivastava et al., 2024) were designed as *evaluation targets*—new test sets meant to replace or supplement existing ones, with the implicit goal that models should eventually score well on them. The field's instinct when faced with a new test set is to optimize against it: generate training data in its format, tune prompts, or adjust training recipes. The paper explicitly pushes against this instinct in Section 5:

> "Our case study should not be viewed as an endpoint or merely as a tool for generating additional training data to 'solve' compositional GSM problems, but as a catalyst to gain insights about the nature of reasoning of current LLMs as well as to re-evaluate how we assess 'reasoning'."

The diagnostic framing means the gap `Δ` is the object of interest, not the raw accuracy `Scomp`. A model that achieves higher `Scomp` through format-specific optimizations but maintains a large `Δ` has not actually improved its compositional reasoning—it has only gotten better at the specific prompt format. The paper's approach of reporting the gap alongside raw scores (Figure 1, Figure 3) forces the reader to consider *both* metrics, making the diagnostic relationship explicit.

**Comparison to prior work:** In the robustness evaluation literature, the standard move is to introduce a perturbed test set, show that models degrade, and conclude that robustness is poor. The contribution stops at measurement. This paper goes further by providing a principled statistical baseline (`S1 × S2`, the expected accuracy under independent single-question performance) that transforms the raw degradation into a *normalized* metric that corrects for base capability differences. Without this normalization, a model scoring 95% on GSM8K and 80% on compositional GSM would appear to have a 15-point gap; a model scoring 60% on GSM8K and 45% on compositional GSM would appear to have the same 15-point gap. But the first model is underperforming expectations by much more relative to its capability. The `S1 × S2` baseline reveals this: the high-performing model's expected compositional score is `0.95 × 0.95 ≈ 0.90`, giving `Δ ≈ −0.10`, while the lower-performing model's expected score is `0.60 × 0.60 = 0.36`, giving `Δ ≈ +0.09` (actually outperforming expectations). The raw-gap approach would miss this inversion entirely.

**Significance beyond metrics:** This reframing changes what it means to "solve" the compositional reasoning problem. If someone fine-tunes a model on compositional GSM questions and closes the raw accuracy gap, but the model still shows the same `Δ` relative to its improved single-question scores, the compositional reasoning deficit persists—it has merely been masked by better pattern matching on a larger set of patterns. The paper's analytical decomposition (Figures 10–12) provides the diagnostic tools to distinguish genuine compositional improvement from format-specific optimization. This is a conceptual advance in how the field thinks about evaluation: the goal is not to maximize a new number but to understand *why* the old number was misleading.

**Fundamental vs. incremental:** This is a fundamental reframing. The paper does not simply add another row to the benchmark leaderboard; it changes what should be *on* the leaderboard in the first place (gap alongside accuracy) and provides a framework for interpreting the relationship between them. The distinction between a "benchmark" and a "diagnostic probe" has been gestured at in prior work but rarely operationalized with this level of clarity and statistical grounding.

**Evidence anchor:** Figure 3 (the reasoning gap bar chart) and Figure 1 (the compositional-vs-GSM8K scatter) together embody this insight. Figure 3 orders models by gap magnitude, making the diagnostic value immediately visible: GPT-4o and Qwen2.5-MATH-72B-IT show near-zero gaps despite very different GSM8K scores, while Phi-3-mini-4k-IT and Gemma2-9B-IT show large gaps despite competitive single-question performance. A standard benchmark table sorted by `Scomp` alone would obscure these patterns.

---

### Innovation 2: The Disproportionate Fragility of Cost-Efficient Models as a Systematic Finding

The paper documents that smaller, cheaper models do not simply perform *worse* on compositional reasoning proportionally to their lower single-question accuracy—they exhibit a *qualitatively different* relationship between single-hop and multi-hop performance. This is not an obvious consequence of reduced capacity; it reveals a specific brittleness in how small models generalize.

**What makes this distinctive:** The naive expectation under a "smaller models are just weaker" hypothesis would be that both single-question and compositional accuracy degrade together, preserving roughly the same `Δ` across model scales within a family. The paper shows the opposite: within the same model family, the smaller variant often achieves comparable single-question accuracy (e.g., GPT-4o mini near GPT-4o on GSM8K, Figure 4; LLAMA3-8B-IT within ~10 points of LLAMA3-70B-IT) but a dramatically larger gap (GPT-4o mini: 14.2 vs. GPT-4o: 1.1; LLAMA3-8B-IT: 27.5 vs. LLAMA3-70B-IT: 4.9). The gap is not proportional to the accuracy difference—it is an order of magnitude larger than what base capability would predict.

This is not simply a "smaller model, worse performance" story. It is evidence that the *architecture of reasoning* differs systematically between small and large models: small models achieve their GSM8K scores through strategies that are more brittle to compositional perturbation. The paper identifies this as a systematic difference rather than a scaling artifact, linking it to specific failure modes (distraction susceptibility in Figure 10, second-hop failure in Figure 11).

**Comparison to prior work:** Scaling law literature (Kaplan et al., 2020; Hernandez et al., 2021) established that larger models are more sample-efficient and achieve lower loss, but treated generalization as a continuous function of scale—performance on any given task should improve smoothly with model size. Press et al. (2023) found that the compositionality gap in GPT-3 models did *not* decrease with scale, which would suggest a flat relationship where all model sizes struggle equally with composition. This paper finds something between these extremes: the gap *does* decrease with scale (larger models are better at compositional reasoning), but the relationship is far steeper than single-question scaling curves would suggest, and cost-efficient models optimized for benchmark performance (GPT-4o mini, Gemini 1.5 Flash) show disproportionately large gaps relative to their single-question scores.

**Significance beyond metrics:** For the ML deployment community, this finding has direct economic implications. Organizations choosing between GPT-4o and GPT-4o mini based on their near-identical GSM8K scores (both >90%, per the paper's description) would reasonably select the 25–35× cheaper option for math-heavy applications. The compositional gap reveals that this decision would be based on a misleading signal: the cheaper model is substantially less reliable at multi-step reasoning tasks, a deficit invisible on standard benchmarks. The paper quantifies this hidden cost in a way that directly informs model selection decisions.

For the research community, the finding challenges the narrative that distillation and synthetic data can close the gap between small and large models for reasoning. If the gap stemmed primarily from knowledge or pattern coverage, performance on a compositional evaluation using the same underlying operations should scale similarly to single-question performance. The fact that it doesn't suggests that small models have a structural deficit in multi-hop reasoning that more training data (in standard formats) may not address—a hypothesis partially supported by the fine-tuning overfitting result (Figure 7), where more GSM8K training *reduced* compositional accuracy.

**Fundamental vs. incremental:** This is a fundamental empirical discovery, not an incremental refinement. The systematic asymmetry between single-hop and multi-hop scaling across model sizes has not been documented at this breadth (23 models, 7 families) or with this level of analytical decomposition before. It establishes "compositional reasoning gap at fixed single-question accuracy" as a new axis of model comparison that is orthogonal to parameter count, benchmark score, or training FLOPs.

**Evidence anchor:** Figure 4 provides the cleanest visual: four model families, each with a large and small variant, showing that the smaller variant's compositional bar is disproportionately shorter than its GSM8K bar. The annotation of gap magnitudes above the bars (4.9 vs. 27.5 for LLAMA3, 5.8 vs. 11.3 for Gemini) makes the asymmetry quantitative. Figure 3 extends this to all evaluated models, showing that small models cluster at the high-gap end of the distribution.

---

### Innovation 3: Instruction Tuning's Asymmetric Impact on Compositional Generalization

The paper reveals that instruction tuning—a near-universal component of modern LLM training pipelines—affects compositional reasoning in qualitatively different ways depending on model scale, despite using similar or identical training data and procedures. This finding challenges the assumption that instruction tuning is a uniform "improvement operation" that enhances all capabilities proportionally.

**What makes this distinctive:** The standard narrative around instruction tuning is that it improves models' ability to follow instructions and perform tasks, often with particular benefits for reasoning benchmarks. The paper's comparison of pretrained (PT) and instruction-tuned (IT) variants within the same model families reveals that this narrative is incomplete—and for small models, potentially misleading.

For small models (top row of Figure 5: Mistral-7B, LLAMA3-8B, Gemma2-9B), instruction tuning provides dramatically larger gains on standard GSM8K than on compositional GSM. For Mistral-7B: +14.1 points on GSM8K but only +4.3 on compositional GSM. For LLAMA3-8B: +25.1 vs. +12.6. For Gemma2-9B: +22.8 vs. +4.8. The instruction tuning is disproportionately improving single-hop performance.

For large models (bottom row: Mixtral-8x7B, LLAMA3-70B, Gemma2-27B), this asymmetry largely disappears or reverses. LLAMA3-70B gains +8.6 on GSM8K but +19.0 on compositional GSM—instruction tuning helps compositional reasoning *more* than single-question reasoning. Gemma2-27B shows a similar pattern (+15.2 GSM8K vs. +17.2 compositional).

**Why this is non-obvious:** If instruction tuning simply made models "better at math," the gains would be proportional across GSM8K and compositional GSM. If it specifically improved instruction-following (which is what it is designed to do), the compositional format—which requires more complex instruction following (solving two questions, tracking variable X)—might benefit *more* than single-question format. The paper finds the opposite for small models: instruction tuning disproportionately improves the *simpler* task format. This suggests that current instruction-tuning recipes for small models are effectively overfitting to the single-question GSM8K format, teaching the model to produce the surface patterns of step-by-step reasoning without developing the underlying variable-tracking and state-maintenance capabilities that compositional reasoning requires.

**Comparison to prior work:** Prior studies on instruction tuning typically report aggregate benchmark improvements and analyze which capabilities improve (math, coding, reasoning, safety). They do not decompose improvements by task complexity or examine whether the gains transfer to compositional variants of the same underlying task. The paper's within-family PT-vs-IT comparison, stratified by model size, reveals a pattern invisible in aggregate metrics: instruction tuning's benefits are *differentially* distributed across task formats in a scale-dependent way.

**Significance beyond metrics:** This finding has direct implications for how instruction tuning datasets are constructed. If small models are systematically overfitting to the surface format of benchmark-style questions during instruction tuning, the solution is not simply "add more compositional problems to the instruction tuning mix" (which would risk making the compositional evaluation itself a target). Instead, the paper's results suggest a need to understand *why* the same training procedure produces qualitatively different generalization patterns at different scales, and to design instruction-tuning strategies that explicitly target the underlying reasoning capabilities rather than benchmark-format proficiency.

The finding also complicates the narrative of "distillation works." If small instruction-tuned models are achieving their benchmark scores through format-specific pattern matching rather than transferable reasoning, they represent a different kind of capability than large models with similar scores—a distinction that the compositional gap makes visible but that standard evaluations obscure.

**Fundamental vs. incremental:** This is a fundamental empirical discovery about a widely-used training technique. It identifies a systematic interaction between model scale, training recipe, and compositional generalization that has not been previously documented. The finding that the *same* instruction tuning procedure produces *opposite* effects on compositional generalization depending on model size is particularly striking and not predicted by any existing theory.

**Evidence anchor:** Figure 5 presents the PT-vs-IT comparison, with the asymmetric gain patterns directly annotated above the bars. The top-row vs. bottom-row contrast is the visual argument: small models show a consistent "bigger GSM8K gain, smaller compositional gain" pattern, while large models show the reverse or neutral pattern. The paper notes this explicitly: "this trend does not apply or is reversed for larger LLMs (bottom row), despite using similar or identical data and training setup during instruction-tuning."

---

### Innovation 4: Task Overfitting as a Concrete Mechanism, Demonstrated via Fine-Tuning Dynamics

The paper provides a clean demonstration that extended fine-tuning on GSM8K data causes a divergence between single-question and compositional accuracy—single-question performance continues to improve while compositional performance regresses. This is not merely a "more training can hurt" observation (which is well-known) but a specific claim about *how* the overfitting manifests: as optimization for format-specific patterns at the expense of transferable reasoning.

**What makes this distinctive:** The paper shows the *trajectory* of overfitting, not just the endpoint. At 50 and 100 training steps, both GSM8K and compositional GSM accuracy improve. Between 100 and 400 steps, they decouple: GSM8K continues climbing while compositional GSM declines (Figure 7). This dynamic reveals that the model is not simply "forgetting" or "catastrophically interfering"—it is actively optimizing for features that help GSM8K but hurt compositional transfer.

This is a more specific and actionable finding than generic overfitting warnings. It suggests that there exists an intermediate point in fine-tuning where compositional generalization is maximized, after which specialization to the training format begins to erode it. The practical implication is that early stopping based on compositional (or other out-of-distribution) metrics, rather than in-distribution validation loss, may be necessary when fine-tuning for reasoning tasks.

**The synthetic data advantage is also revealing:** Training on model-generated synthetic solutions outperforms training on human-written solutions across all checkpoints on both GSM8K and compositional GSM (Figure 7). This is initially surprising—why would the model's own (potentially flawed) reasoning patterns make better training data than human-written solutions? The paper hypothesizes that the synthetic data, being generated by the same model, is more "in-distribution" for the model's learning dynamics, allowing more efficient learning of the underlying operations. However, the overfitting pattern (compositional decline with extended training) occurs in both conditions, suggesting that the overfitting is driven by repeated exposure to the GSM8K format and solution distribution, not by the specific authorship of the solutions.

**Comparison to prior work:** The overfitting-to-benchmarks concern is widely discussed but rarely demonstrated with this level of clarity and at this model scale. Prior work on benchmark contamination (Zhang et al., 2024; Xu et al., 2024) focused on whether models had seen test questions during training. The paper's contribution is orthogonal: even without test-set leakage, training on the *distribution* of single-hop questions can erode multi-hop capability. This is a subtler form of overfitting that is harder to detect via standard held-out evaluation (since the held-out questions are from the same distribution) and requires specifically designed compositional probes to surface.

The connection the paper draws to "overtraining" practices—using large synthetic datasets to train small models beyond compute-optimal scaling (Gadre et al., 2024; Sardana and Frankle, 2023)—is significant. If the benefits of overtraining are concentrated on in-distribution benchmarks while generalization degrades, the current trend of pushing small models to benchmark parity with large models through data scaling may be producing models that are less capable than their scores suggest.

**Significance beyond metrics:** This finding provides a mechanistic hypothesis for *why* small, cost-efficient models show disproportionately large reasoning gaps (Innovation 2). If these models are heavily over-trained on synthetic data to maximize benchmark performance, they may have passed the point where compositional generalization peaks, trading away multi-hop capability for single-hop pattern matching. The paper does not prove this causal link, but the fine-tuning experiment provides a controlled demonstration of the mechanism, making the hypothesis plausible and testable.

**Fundamental vs. incremental:** This is a fundamental empirical demonstration of a mechanism that has been hypothesized but not cleanly shown at this scale. The dynamic trajectory (improvement then decline of compositional accuracy) is the key evidence that distinguishes this from simpler "fine-tuning didn't help compositional" results. The finding that synthetic data produces overall better performance but the same overfitting pattern is a nuanced result that informs both data generation strategies and training curriculum design.

**Evidence anchor:** Figure 7 shows the decoupling clearly: at 100 steps (human data), GSM8K is ~78% and compositional is ~43%; at 400 steps, GSM8K rises to ~84% while compositional drops to ~37%. The synthetic data condition shows the same qualitative pattern at higher absolute numbers (~82% → ~89% GSM8K, ~49% → ~47% compositional). The paper explicitly states: "compositional GSM test performance drops while GSM8K test performance keeps improving."

## 5. Experimental Analysis

### Evaluation Methodology

- **Dataset.** The paper constructs three test splits, each containing exactly 1,200 examples. The **original GSM8K test split** comprises questions from the standard GSM8K test set (Cobbe et al., 2021) that serve as `Q1` in the compositional pairs. The **modified GSM8K test split** contains the `Q2` questions — original GSM8K questions where one numerical value has been replaced (the value that becomes variable `X` in the compositional format), with the replacement designed so the new final answer remains a positive integer close to the original answer. The **compositional GSM test split** chains pairs of questions together: `Q1` from the original split and `Q2` from the modified split, with explicit instructions that `X` (the answer to `Q1`) must be computed and substituted into `Q2`. The answer magnitude distributions of the original and compositional splits are verified to be similar (Appendix A, Figure A.1). Question pairs were automatically constructed via code-form solution manipulation from Gao et al. (2023), validated using GPT-4o and Gemini 1.5 Pro agreement with code-execution answers (questions with fewer than 4 out of 16 model solutions agreeing were flagged), and approximately 25% of questions were manually reviewed and adjusted for logical coherence.

- **Base model(s).** The paper evaluates 23 model variants spanning seven families: Gemini (1.0 Pro, 1.5 Flash, 1.5 Pro; Google, 2023, 2024), Gemma2 (9B-PT, 27B-PT, 9B-IT, 27B-IT; Gemma Team et al., 2024), GPT (4o, 4o mini; OpenAI, 2023a), LLAMA3 (8B-PT, 70B-PT, 8B-IT, 70B-IT; AI@Meta, 2024), Phi (Phi-2, Phi-3-mini-4k-IT; Abdin et al., 2024), Mistral/Mixtral (Mistral-7B-PT, Mistral-7B-IT, Mixtral-8x7B-PT, Mixtral-8x7B-IT; Jiang et al., 2024), and math-specialized models (NuminaMath-7B-CoT from Beeching et al., 2024; Mathstral-7B; Qwen2.5-MATH-7B-IT and Qwen2.5-MATH-72B-IT from Yang et al., 2024a). The selection spans pretrained (PT) and instruction-tuned (IT) variants, open-weights and closed-source API models, and parameter scales from 2B to 72B (plus unspecified-size frontier models). This breadth is chosen to enable systematic comparison of how compositional reasoning varies across model families, sizes, and training recipes — the paper explicitly aims to establish whether observed differences are systematic rather than idiosyncratic to specific models.

- **Metrics.** The primary metrics are three test accuracies, each expressed as pass@1 (fraction of 1,200 questions answered correctly on the first attempt with temperature 0 greedy decoding): `S1` on the original GSM8K test split (`Q1` set), `S2` on the modified GSM8K test split (`Q2` set), and `Scomp` on the compositional GSM test split (an answer is correct only if both `Q1` and `Q2` are correctly solved). From these, the **compositional reasoning gap** is computed as `Δ = Scomp − S1 × S2`, where `S1 × S2` is the statistical expectation under the assumption that solving the two questions are independent events. A negative `Δ` indicates underperformance relative to expectation; a positive `Δ` (theoretically possible but rare) would indicate that the compositional format somehow helps. The paper reports gap magnitudes in percentage points (e.g., a gap of "27.5" means `Δ = −0.275`). For the code generation ablation, the paper additionally reports **relative improvement** over natural language CoT, computed as `(Accuracy_code − Accuracy_CoT) / Accuracy_CoT × 100%`.

- **Baselines.** The paper does not use traditional baselines in the sense of competing methods — it is an evaluation study, not a methods paper. However, the **statistical baseline** `S1 × S2` serves as the expected-performance reference against which `Scomp` is compared. This baseline is derived from each model's own single-question performance and represents the null hypothesis that compositional format adds no additional difficulty beyond the difficulty of the individual questions. The paper also compares models against each other within families (large vs. small variants, pretrained vs. instruction-tuned) to establish systematic patterns. The `y = x` diagonal serves as an implicit baseline in the diagnostic analyses (Figures 9, 10, 11), representing the expectation of equal performance across conditions.

- **Generation budget / compute accounting.** All models are sampled with temperature 0 (deterministic greedy decoding), producing exactly one generation per question. There is no test-time compute scaling, no beam search, no majority voting, no best-of-N — the paper measures single-shot performance. The "compute budget" is therefore uniform across all models and all test splits: one forward pass per question. This design choice deliberately isolates model capability from inference-time strategies, meaning the reported compositional gaps represent a lower bound on what could be achieved with additional test-time computation. The 8-shot prompts add context length but the paper does not report exact token counts; each shot is a short grade-school problem (~50-100 words), so total prompt length is well within the context windows of all evaluated models.

- **Cross-validation / statistical protocol.** There is no cross-validation in the conventional sense — the paper is an evaluation study using fixed test sets, not a training study requiring held-out validation for hyperparameter selection. The 1,200-example test sets are of moderate size; the paper does not report confidence intervals, standard errors, or statistical significance tests for the gap metric. The fine-tuning overfitting experiment (Section 3.4) evaluates at fixed checkpoints (50, 100, 400 steps) on the fixed test sets without cross-validated early stopping, since the purpose is to observe the *trajectory* of overfitting rather than to select an optimal model. For models requiring preamble prefixes (Appendix B), the paper tests both with and without the preamble and reports the best performance for each model on each split — this is a minor source of variance but the paper does not quantify how much the preamble choice affects results.

---

### Main Quantitative Results

#### The Reasoning Gap Across All Models

The headline finding across all 23 evaluated models is that **most LLMs exhibit a substantial reasoning gap**, with `Δ` ranging from near zero (GPT-4o: gap of ~1, Qwen2.5-MATH-72B-IT: gap of ~2.6) to as high as 37.3 (Gemma2-9B-IT), as shown in Figure 3. The gap is not a function of absolute performance — some high-performing models show large gaps (Gemma2-9B-IT achieves ~88% on GSM8K but shows a 37.3-point gap), while some models with lower raw scores show smaller gaps. This establishes that compositional reasoning capability is partially orthogonal to single-question accuracy.

Figure 1 visualizes this orthogonality as a scatter plot with compositional GSM accuracy on the `y`-axis and the geometric mean of single-question accuracies `√(S1 × S2)` on the `x`-axis (labeled "GSM8K Accuracy" for simplicity). The trendline `y = x²` represents the expected compositional accuracy under independence. Models falling below this curve have negative `Δ`; models on or above it have near-zero or positive gaps. The scatter reveals clusters: frontier closed-source models (GPT-4o, Gemini 1.5 Pro) cluster near the curve, while smaller open-weights models (Phi-2, Gemma2-9B-IT, LLAMA3-8B-IT) fall substantially below it, and math-specialized models show mixed behavior (Qwen2.5-MATH-72B-IT near the curve, Qwen2.5-MATH-7B-IT far below it).

The **magnitude hierarchy** in Figure 3, ordered from largest to smallest gap, reveals that smaller models and instruction-tuned variants dominate the high-gap end: Phi-3-mini-4k-IT (gap ~40), Gemma2-9B-IT (37.3), Gemma2-9B-PT (30s), LLAMA3-8B-IT (27.5), and Qwen2.5-MATH-7B-IT (21.9) all appear in the top half of the gap distribution. Frontier models (GPT-4o, Gemini 1.5 Pro, LLAMA3-70B-IT, Qwen2.5-MATH-72B-IT) cluster at the low-gap end with gaps of roughly 0–5 points.

#### Cost-Efficient LLMs Reason Differently (Figure 4)

The paper examines four model families, each with a high-cost (large) and low-cost (small) variant, where cost is measured via parameter count or API pricing. The key quantitative finding: **while cheaper models perform comparably or slightly worse on original GSM8K, they exhibit a 2–12× larger reasoning gap on compositional GSM**.

For the **GPT family**: GPT-4o achieves approximately 95% on GSM8K and shows a compositional gap of 1.1 percentage points. GPT-4o mini achieves approximately 92% on GSM8K (within 3 points) but shows a compositional gap of 14.2 — roughly **13× larger**. GPT-4o mini is priced 25–35× cheaper than GPT-4o.

For the **Gemini family**: Gemini 1.5 Pro achieves approximately 93% on GSM8K with a gap of 5.8. Gemini 1.5 Flash achieves approximately 90% on GSM8K with a gap of 11.3 — roughly **2× larger**.

For the **LLAMA3 family**: LLAMA3-70B-IT achieves approximately 88% on GSM8K with a gap of 4.9. LLAMA3-8B-IT achieves approximately 78% on GSM8K with a gap of 27.5 — roughly **5.6× larger**.

For the **Gemma2 family**: Gemma2-27B-IT achieves approximately 85% on GSM8K with a gap of 18. Gemma2-9B-IT achieves approximately 88% on GSM8K with a gap of 37.3 — roughly **2.1× larger**. Note that here the *smaller* model actually has a *higher* GSM8K score (88% vs. 85%) but a dramatically larger gap, further emphasizing the orthogonality of single-question accuracy and compositional reasoning.

The paper characterizes this pattern as "the reasoning flaws of cost-efficient LLMs may be obscured by high scores on prevalent math-reasoning benchmarks, underscoring the need to rethink current strategies for developing such models."

#### Instruction-Tuning Effects Vary Across LLM Sizes (Figure 5)

Comparing pretrained (PT) and instruction-tuned (IT) variants within three model families reveals a **scale-dependent asymmetry in how instruction tuning affects compositional generalization**.

For **small models** (top row of Figure 5):

- **Mistral-7B**: Instruction tuning improves GSM8K by +14.1 points but compositional GSM by only +4.3 points. The gain is 3.3× larger for single-question than for compositional.
- **LLAMA3-8B**: Instruction tuning improves GSM8K by +25.1 points but compositional GSM by +12.6 points. The gain is 2.0× larger for single-question.
- **Gemma2-9B**: Instruction tuning improves GSM8K by +22.8 points but compositional GSM by only +4.8 points. The gain is 4.75× larger for single-question.

For **large models** (bottom row of Figure 5):

- **Mixtral-8x7B**: Instruction tuning improves GSM8K by +7.6 points and compositional GSM by +3.1 points — a more balanced ratio, though still favoring single-question.
- **LLAMA3-70B**: Instruction tuning improves GSM8K by +8.6 points but compositional GSM by +19.0 points — compositional reasoning benefits **more** than single-question reasoning, reversing the small-model pattern.
- **Gemma2-27B**: Instruction tuning improves GSM8K by +15.2 points and compositional GSM by +17.2 points — approximately equal gains, and the compositional gain is actually slightly larger.

The paper states: "this trend does not apply or is reversed for larger LLMs (bottom row), despite using similar or identical data and training setup during instruction-tuning." The absolute GSM8K scores of instruction-tuned and pretrained variants also reveal that for small models, most of the instruction-tuning gain is concentrated on GSM8K (the IT variant massively outperforms PT on GSM8K but much less so on compositional), while for large models the gains are more balanced or compositional-favoring.

#### Math-Specialization Does Not Improve the Reasoning Gap (Figure 6)

Four math-specialized LLMs are evaluated: NuminaMath-7B-CoT, Mathstral-7B, Qwen2.5-MATH-7B-IT, and Qwen2.5-MATH-72B-IT. The finding is that **math-specialized models exhibit reasoning gaps comparable to other models of similar size**, and that the gap persists even for models that achieve high scores on substantially harder benchmarks.

**NuminaMath-7B-CoT** achieves approximately 68% on GSM8K with a compositional gap of 12.1 points. **Mathstral-7B** achieves approximately 70% on GSM8K with a gap of 14 points. These gaps are similar to or larger than non-specialized models of comparable size (e.g., Mistral-7B-IT with gap ~15–20 based on Figure 3).

**Qwen2.5-MATH-7B-IT** achieves approximately 86% on GSM8K with a gap of 21.9 points. The paper notes this is particularly striking because this model "achieves above 80% accuracy on difficult high-school competition level questions in MATH (Hendrycks et al., 2021), but solves less than 60% of the compositional grade-school math problems." The MATH benchmark involves significantly harder individual problems than chaining two GSM8K questions together, yet the model fails on the compositional task — suggesting that strong performance on hard single-hop problems does not transfer to easy multi-hop problems.

**Qwen2.5-MATH-72B-IT** achieves approximately 92% on GSM8K with a gap of only 2.6 points — nearly closing the compositional gap. The large difference between the 7B and 72B variants of the same model family (gap of 21.9 vs. 2.6) despite similar GSM8K scores (86% vs. 92%) reinforces the finding from Section 3.1 that smaller models exhibit systematically different reasoning capabilities.

#### Fine-Tuning Can Lead to Task Overfitting (Figure 7)

Fine-tuning Gemma2 27B PT on GSM8K training data reveals a **decoupling between single-question and compositional accuracy with extended training**.

At **50 training steps**: Both human-data and synthetic-data conditions show improvement on both GSM8K and compositional GSM. On human data: GSM8K approximately 72%, compositional approximately 38%. On synthetic data: GSM8K approximately 78%, compositional approximately 43%.

At **100 training steps**: Both splits continue improving. Human: GSM8K approximately 78%, compositional approximately 43%. Synthetic: GSM8K approximately 82%, compositional approximately 49%. This is the peak compositional performance in both conditions.

At **400 training steps**: The trajectories diverge. **GSM8K continues improving**: human reaches approximately 84%, synthetic reaches approximately 89%. **Compositional GSM declines**: human drops to approximately 37% (below the 100-step peak of ~43%), synthetic drops to approximately 47% (below the 100-step peak of ~49%).

No further improvements were observed on either split after 400 steps. The paper states: "In both settings, after 100 training steps, compositional GSM test performance drops while GSM8K test performance keeps improving."

**Synthetic vs. human data**: The synthetic data condition consistently outperforms the human data condition across all checkpoints on both GSM8K and compositional GSM. At 100 steps, synthetic data achieves ~4 points higher GSM8K and ~6 points higher compositional accuracy. The overfitting pattern (compositional decline with extended training) occurs in both conditions, indicating that it is driven by repeated exposure to the GSM8K distribution rather than the specific authorship of the training solutions.

The paper hypothesizes: "the trend of using increasingly larger training datasets for over-training small models beyond compute-optimal scaling — often heavily composed of synthetic data — may primarily target performance on standard benchmarks, potentially at the expense of overall generalization and effectiveness across a wider range of tasks."

#### Reasoning in Natural Language versus Code (Figure 8)

Evaluating compositional GSM with code generation (Python functions) instead of natural language Chain-of-Thought reveals that **code generation improves compositional accuracy for all tested models, but the improvement is dramatically larger for smaller models**.

The experiment covers three model families (LLAMA3, Gemma2, Mistral/Mixtral), each with an instruction-tuned small and large variant, and reports relative improvement over natural language CoT:

- **LLAMA3-8B-IT**: +69% relative improvement (e.g., if CoT accuracy were 30%, code accuracy would be approximately 51%)
- **LLAMA3-70B-IT**: +2% relative improvement
- **Gemma2-9B-IT**: +74% relative improvement
- **Gemma2-27B-IT**: +27% relative improvement
- **Mistral-7B-IT**: +149% relative improvement (more than doubling CoT accuracy)
- **Mixtral-8x7B-IT**: +71% relative improvement

The asymmetry is the key finding: smaller models benefit disproportionately from translating the reasoning task into code, while larger models show modest or negligible gains. For LLAMA3, the small model gains 69% while the large model gains only 2% — a 34.5× difference in relative benefit. For Gemma2, the small model gains 74% vs. the large model's 27% — a 2.7× difference.

The paper interprets this as evidence of "systematic differences in reasoning capabilities": smaller models struggle specifically with the natural language execution of multi-step reasoning (variable tracking, state maintenance across steps) that code execution offloads to the Python interpreter. When the reasoning steps are expressed as code and executed deterministically, the smaller model's deficit largely disappears relative to the larger model. Note that absolute code-generation accuracies are not reported — only the relative improvements — so it is not possible to determine whether code generation fully closes the compositional gap or merely reduces it.

---

### Ablation Studies and Robustness Checks

**Test-set leakage check (Figure 9):** Comparing model accuracy on original GSM8K questions versus modified GSM8K questions (where one number has been changed) shows that "most models are very close to the `x = y` line," indicating that the modifications do not themselves make the questions harder. If test-set leakage (models having memorized specific test questions during training) were a significant factor, accuracy would drop substantially on modified questions because the memorized answers would no longer match. The paper concludes that "test-set leakage is not a major concern in our setup." This result is important because it rules out the simplest alternative explanation for the compositional gap: that the modified `Q2` questions are inherently harder for models that relied on memorization. The gap must therefore be attributed to the compositional format itself rather than question difficulty.

**Distraction analysis (Figure 10):** For each model, the fraction of `Q1` questions solved correctly in the standard non-compositional format (`x`-axis) is plotted against the fraction solved correctly when `Q1` appears as the first question in the compositional format (`y`-axis). The `y = x` diagonal represents the null hypothesis that adding `Q2` to the prompt does not affect `Q1` performance (solving `Q1` does not depend on `Q2`). The paper finds that "several models fall short of this expectation," falling below the diagonal. Qualitative examination of responses from models with larger deviations reveals that they "often overlook important details, such as missing a reasoning step related to each in the question or omitting an arithmetic step when the question specifies a month or per month." This establishes **distraction** as one component of the reasoning gap — the mere presence of a second question (even though it appears after `Q1` and does not need to be solved yet) degrades performance on the first question. Models that adhere well to the output format specified in the 8-shot prompt show "negligible instances of non-extractable answers," indicating that the failures are reasoning failures rather than format-compliance failures.

**Second-hop reasoning analysis (Figure 11):** For each model, the fraction of `Q2` questions solved correctly in the non-compositional format (`x`-axis) is plotted against the fraction solved correctly in the compositional format *conditioned on `Q1` being correctly solved* (`y`-axis). The diagonal represents the expectation that, given correct computation of `X`, solving `Q2` should be no harder than solving it independently with `X` provided. The paper finds substantial deviations below the diagonal for many models, indicating that even when the model successfully computes the intermediate answer, it still struggles to use that answer to solve the second question. Qualitative analysis shows that models "might answer the first one correctly, but often makes subtle errors and overlooks details, leading to inaccurate reasoning and solution for the second question." This establishes **second-hop reasoning failure** as a second, distinct component of the gap — separate from the distraction affecting `Q1`.

**Capacity vs. dependency analysis (Figure 12):** This ablation teases apart whether models fail due to the sheer length of processing two questions or specifically due to the dependency between them. For four models (LLAMA3-70B-IT, LLAMA3-8B-IT, Gemma2-27B-IT, Gemma2-9B-IT), `Q2` accuracy is compared in three conditions: (1) standard format (`Q2` alone); (2) `Q2` with an independent `Q1` in context (where `Q1`'s answer is *not* needed for `Q2` — `X` is provided as a concrete number); and (3) compositional format given `Q1` solved. The finding: "the distraction from `Q1` in the context is minimal when `Q2` is independent of it," meaning models can handle two questions in the prompt as long as they are not logically linked. However, "when `Q2` relies on the answer from `Q1`, models struggle to solve `Q2` accurately, even if `Q1` has been answered correctly." The performance drop occurs specifically in the compositional dependency condition, not in the two-question independent condition. This indicates that the core problem is not capacity or attention limitation (the prompt length and question count are similar in both conditions) but rather a **state-tracking and variable-binding failure** — the model cannot reliably maintain and reuse a value it computed earlier when that value must be integrated into subsequent reasoning.

**Frontier model agreement threshold (Section 2):** The paper used a threshold of 4 out of 16 GPT-4o/Gemini 1.5 Pro solutions agreeing with the code-execution answer to flag questions for manual review. No ablation of this threshold is reported. Approximately 25% of questions required manual adjustment. The choice of threshold is pragmatic; the paper does not investigate sensitivity to this parameter, but since flagged questions were manually corrected rather than discarded, the threshold primarily affects annotation cost rather than test set quality.

**Preamble prefix ablation (Appendix B):** The paper tests models both with and without a preamble prefix that instructs the model on output formatting. The best performance per model is reported. No quantitative comparison of "with preamble vs. without preamble" is provided, so the impact of the preamble on compositional accuracy cannot be assessed from the presented data. However, since the preamble only provides formatting guidance and does not change the task itself, and since the best-of-two condition is fair (it avoids penalizing models for formatting failures), this is a minor concern for result interpretation.

**8-shot prompt content:** The 8-shot prompts use fixed examples from standard GSM8K training data. No ablation of the number of shots, the specific examples chosen, or the prompt format is reported. The paper states that "no elaborate prompting method is needed with this format," implying that the 8-shot prompt is sufficient for models to understand the task structure — but the sensitivity of the gap metric to prompting choices is not quantified. A model that responds differently to different prompt formats might show a different gap; the paper's results are conditional on the specific 8-shot prompts in Appendices D, E, and F.

---

### Critical Assessment

#### Claim: "Most models exhibit a clear gap between their performance on GSM8K and compositional GSM"

This claim is **strongly supported** by Figures 1 and 3. The gap `Δ` is negative for nearly all evaluated models (the paper uses the convention where a positive reported "gap" means `Δ = Scomp − S1 × S2 < 0`, i.e., underperformance). The magnitudes range from ~1 to ~40 percentage points. The test set size (1,200 questions) provides reasonable statistical power, and the consistency of the finding across 23 models from 7 families suggests it is not an artifact of specific model architecture or training procedure.

**However**, the paper does not report statistical uncertainties on `Δ`. The gap is a nonlinear function of three measured quantities (`S1`, `S2`, `Scomp`), each with binomial sampling variance. At 1,200 test examples, the standard error on a 50% accuracy is approximately 1.4 percentage points; the error on `S1 × S2` is larger because it compounds the errors of `S1` and `S2`. For a model with `S1 = S2 = Scomp = 0.80`, the standard error on `Δ` would be approximately 2.4 percentage points under independence assumptions (possibly larger if the three measurements are correlated). This means gaps of less than ~5 points may not be statistically distinguishable from zero, and the ranking of models by gap magnitude in Figure 3 has uncertainty that the paper does not quantify. The visual presentation of Figure 3 as a ranked bar chart without error bars gives a misleading impression of precision.

Additionally, the `S1 × S2` baseline itself makes a strong assumption: that `Q1` and `Q2` are solved independently in the compositional context. This assumption could be violated in either direction — solving `Q1` correctly might make `Q2` easier (if the model better understands the problem domain after the first solution) or harder (if fatigue or context-length effects dominate). The paper's decomposition analyses (Figures 10–12) partially address this by showing that both distraction (degrading `Q1`) and second-hop failure (degrading `Q2` given correct `Q1`) contribute to the gap, but the independence assumption remains an untestable baseline. An alternative baseline — perhaps measuring compositional accuracy of a model fine-tuned specifically for the compositional format — would provide a different reference point, though it would answer a different question.

#### Claim: "The reasoning gap is particularly evident in small, more cost-efficient, and math-specialized models"

This claim is **supported** by Figures 3, 4, and 6, but requires careful qualification. The pattern is clear: within model families, smaller/cheaper variants show larger gaps than larger/expensive variants, and the gap often exceeds what would be predicted from their moderately lower single-question scores.

**A missing analysis**: The paper does not establish whether the larger gap in small models is *disproportionate* relative to their lower base capability, or simply a consequence of lower base capability compounded across two hops. Consider: if a large model has `S1 = S2 = 0.90` and achieves `Scomp = 0.81` (`Δ = 0.81 − 0.81 = 0`), while a small model has `S1 = S2 = 0.70` and achieves `Scomp = 0.36` (`Δ = 0.36 − 0.49 = −0.13`), the small model's gap is larger in absolute terms. But is this "disproportionate"? The small model's single-question accuracy is lower, so the product `S1 × S2` is lower, and the same *relative* degradation would produce a smaller absolute gap. The paper uses `Δ` (absolute gap) as the primary metric, which tends to produce larger values for medium-performing models (where `S1 × S2` is in the 0.25–0.64 range, giving room for large absolute drops) than for very high-performing models (where `S1 × S2` is near 1.0, limiting the maximum possible gap) or very low-performing models (where `S1 × S2` is near 0, also limiting the gap). This metric property means that the finding of "larger gaps in smaller models" is partially driven by the metric itself — models with moderate single-question scores have the most room for large absolute `Δ`. The paper would be strengthened by reporting a relative or normalized gap metric (e.g., `Δ / (S1 × S2)`) alongside the absolute gap, which would reveal whether the compositional degradation is genuinely more severe in small models or simply more visible in the absolute metric.

**Math-specialized models**: The claim that "math-specialization does not improve the reasoning gap" (Section 3.3) is supported for the three small math-specialized models (NuminaMath-7B-CoT: gap ~12.1; Mathstral-7B: gap ~14; Qwen2.5-MATH-7B-IT: gap ~21.9) but the large math-specialized model (Qwen2.5-MATH-72B-IT: gap ~2.6) nearly closes the gap. This suggests that math specialization *at sufficient scale* can improve compositional reasoning — the finding is more precisely stated as "math specialization at 7B scale does not close the compositional gap, and the gap remains scale-dependent even within math-specialized families." The paper's framing of this as "math-specialization does not improve the reasoning gap" somewhat overstates the case given the 72B model's performance.

#### Claim: "Instruction-tuning impacts LLMs of varying sizes in significantly different ways"

This claim is **supported with an important caveat** about the number of models tested. Figure 5 shows three small models (Mistral-7B, LLAMA3-8B, Gemma2-9B) where instruction-tuning gives larger gains on GSM8K than on compositional GSM, and three large models (Mixtral-8x7B, LLAMA3-70B, Gemma2-27B) where the asymmetry is reduced or reversed. This is a consistent pattern across the three families but represents only three data points per size category. The paper does not test whether this pattern generalizes to other model families (e.g., Phi, GPT, Gemini families where both PT and IT variants are available or could be compared) or to intermediate model sizes. The claim that instruction tuning has "significantly different" effects across sizes is based on visual comparison of gain magnitudes without statistical testing.

**A deeper concern**: The paper compares PT and IT variants of the same base models, but the IT variants may differ from PT variants in ways beyond instruction tuning — they may have been trained on different data mixtures, with different hyperparameters, or for different numbers of steps. The paper states that the setups are "similar or identical" but does not provide details of the instruction-tuning procedures for each model family. If, for example, the small-model instruction tuning used more GSM8K-formatted data (which would be consistent with optimizing for benchmark performance), the observed asymmetry could be a data-mixture effect rather than a scale-dependent effect of instruction tuning per se.

**An experiment that would strengthen this claim**: Fine-tune a single base model (e.g., LLAMA3-8B-PT) with the same instruction-tuning data but stop at different model sizes (e.g., via pruning or by training multiple sizes from scratch with identical data). This would isolate the effect of scale from potential confounds in the publicly released IT variants. Such an experiment is beyond the scope of an evaluation paper but would be necessary to establish a *causal* relationship between model scale and instruction-tuning generalization patterns.

#### Claim: "Finetuning on GSM8K can lead to task overfitting"

This claim is **supported** by Figure 7 but is demonstrated on a single model (Gemma2 27B PT) with a single training setup. The finding that compositional accuracy peaks at intermediate training steps and then declines while GSM8K accuracy continues to improve is cleanly shown. However, several aspects limit the generalizability of this claim:

1. **Single model**: The result is shown only for Gemma2 27B PT. The paper does not demonstrate whether this pattern holds for other model sizes, families, or architectures. The finding would be substantially stronger with replication on at least one additional model (e.g., LLAMA3-8B-PT or Mistral-7B-PT).

2. **Fixed hyperparameters**: The training uses a fixed learning rate and batch size (details in Appendix C). The overfitting pattern might be sensitive to these choices — different optimization hyperparameters might shift the peak of compositional performance to different training durations.

3. **Training data quantity**: The paper fine-tunes on the full GSM8K training set (~7,500 examples). The relationship between dataset size and the overfitting pattern is not explored. Would the same pattern emerge with a smaller training set (suggesting rapid overfitting) or a larger one (suggesting that the effect is about repeated exposure to the same distribution rather than absolute data quantity)?

4. **The synthetic data advantage**: The paper finds that synthetic (self-generated) training data outperforms human-written data. This is attributed to the synthetic data being "in-distribution" for the model. An alternative explanation is that the synthetic data solutions are more *diverse* in their reasoning patterns (the model generates 10 solutions per question, keeping only correct ones, which may surface multiple valid reasoning paths), while human solutions follow more consistent templates. The paper does not analyze the diversity of reasoning patterns in the two training sets, so this remains speculative.

5. **Connection to cost-efficient model gaps**: The paper hypothesizes that the overfitting pattern explains why small cost-efficient models show large gaps — they may be over-trained on benchmark-format data. This is a plausible hypothesis but is not directly tested. An experiment that would bridge these findings: take a small model that shows a large gap (e.g., Gemma2-9B-IT), fine-tune it further on GSM8K data, and observe whether the gap *increases* (as the overfitting hypothesis would predict). The paper does not run this experiment.

#### Claim: "Large reasoning gaps are not because of test-set leakage, but due to distraction from additional context and poor second-hop reasoning"

The **leakage claim** is **supported with qualifications** by Figure 9. The near-diagonal alignment between original and modified GSM8K accuracies is evidence against gross test-set memorization. However, the modified questions change only a single number — the problem structure, wording, and solution template remain identical. A model that has memorized solution templates (rather than specific numerical answers) would perform equally well on modified questions, so Figure 9 rules out exact-answer memorization but not template or solution-strategy memorization. The paper's conclusion that "test-set leakage is not a major concern" might be more precisely stated as "exact-answer memorization is not a major concern."

The **distraction and second-hop claims** are **supported** by a strong analytical framework (Figures 10–12) but the evidence is primarily qualitative. Figure 10 shows that several models fall below the diagonal, but the paper does not quantify what fraction of the total gap `Δ` is attributable to distraction vs. second-hop failure vs. their interaction. A decomposition of the gap into these components (e.g., via mediation analysis or counterfactual reasoning) would substantially strengthen the claim that these are the *primary* explanations rather than merely contributing factors.

**Specific missing quantification**: If a model has `S1 = 0.80`, `S2 = 0.75`, and `Scomp = 0.40`, then `Δ = 0.40 − 0.60 = −0.20`. The paper's analyses can tell us: (a) what fraction of `Q1` questions are solved in compositional vs. non-compositional format (distraction effect on `Q1`), (b) what fraction of `Q2` questions are solved given correct `Q1` (second-hop effect), but the paper does not combine these to partition the 20-point gap into "X points from degraded `Q1`, Y points from degraded `Q2` given `Q1`, and Z points from their interaction." This decomposition is straightforward algebraically and would make the diagnostic framework more actionable.

**Additional missing analyses**: The paper does not explore whether the reasoning gap correlates with problem-level features such as the arithmetic complexity of the chained computation, the semantic distance between `Q1` and `Q2` topics (e.g., combining a "trees in a grove" question with a "pencils in a classroom" question vs. two questions in the same domain), the length of the resulting prompt, or the magnitude of the substituted number `X`. Any of these could modulate the gap and provide further insight into *when* composition is hardest for models. The paper's qualitative analysis mentions that models "overlook details" and "make subtle errors," but a systematic error taxonomy (what types of errors occur? arithmetic mistakes? misinterpretation of `X`? failure to substitute? using wrong intermediate value?) would transform this from a gap measurement into a diagnostic instrument.

#### Overall Assessment

The paper's primary contribution — that compositional evaluation reveals systematic differences in reasoning capability invisible in standard single-hop benchmarks — is empirically well-established. The scale of the evaluation (23 models, 7 families) provides robust evidence that the compositional gap is a widespread phenomenon, not an idiosyncratic property of specific models.

**The paper's diagnostic decomposition** (Figures 10–12) is its most intellectually valuable component, transforming the gap from an opaque number into a probe with clinical specificity. The finding that models can handle two questions when they are independent but fail when they are dependent (Figure 12) is particularly incisive — it isolates the failure to **state-tracking and variable-binding** rather than to capacity, attention, or context-length effects. This is a non-obvious result that changes how we should think about LLM reasoning deficits.

**The paper's limitations** center on quantification and causal attribution. The gap `Δ` is reported without uncertainty intervals. The decomposition into distraction and second-hop failure is qualitative rather than quantitative. The finding that small/cost-efficient models have larger gaps is partially a consequence of the `Δ` metric's properties. The instruction-tuning and overfitting results are demonstrated on a small number of models. And the paper's central diagnostic message — that we should evaluate reasoning through compositional probes rather than single-hop benchmarks — is demonstrated convincingly but only for one domain (grade-school math) with one compositional structure (exact two-hop chaining with a single variable).

**Experiments that would have strengthened the paper:**

1. **Varying the number of hops**: Does the gap grow linearly with hop count? Does it saturate? Three-hop or four-hop compositional GSM would test whether the deficit compounds or plateaus.

2. **Varying the dependency type**: The current setup uses direct numerical substitution. What if the dependency is relational ("if Peter has X more marbles than John...") or conditional? These would test different aspects of compositional reasoning.

3. **Adversarial pairing**: What if `Q2` is in a different mathematical domain than `Q1` (e.g., arithmetic followed by geometry)? This would test whether the gap is about mathematical operation chaining specifically or about context-switching.

4. **Within-model scaling**: For a single model family (e.g., LLAMA3), evaluating at multiple intermediate sizes (not just 8B and 70B) would establish whether the gap-size relationship is smooth, exhibits a threshold, or follows a power law.

5. **Test-time compute scaling**: The paper uses temperature 0 and single-shot evaluation. Does the gap change with majority voting, best-of-N, or chain-of-thought self-consistency? If test-time compute disproportionately helps compositional reasoning (or disproportionately helps single-question reasoning), this would be both practically important and theoretically informative about the nature of the gap.

6. **Cross-domain replication**: Applying the same compositional methodology to other reasoning benchmarks (e.g., MATH for harder math, ARC for science, or a code-generation benchmark) would establish whether the compositional gap is specific to grade-school math or a general property of LLM reasoning.

These missing experiments do not undermine what the paper demonstrates — they represent the natural research program that this paper opens. The paper's value is precisely in establishing the compositional gap as a phenomenon worth studying and providing the diagnostic tools to study it, not in exhausting the space of possible analyses.

## 6. Limitations and Trade-offs

### Difficulty Estimation Cost Is Fully Unaccounted in Headline Efficiency Numbers

**The assumption or constraint.** The entire compute-optimal framework — the core novel contribution of this paper — rests on being able to estimate a prompt's difficulty *before* allocating the test-time budget. The method for doing so is extraordinarily expensive. The paper generates **2,048 samples** per question, scores them all with the PRM (or ground-truth correctness, for oracle bins), and uses the resulting distribution to assign a difficulty quintile. Section 3.2 acknowledges this directly:

> "estimating difficulty in this way still incurs additional computation cost during inference... our experiments do not account for this cost largely for simplicity and leave methods for amortizing this cost such as by pretraining or finetuning models to directly predict difficulty of a question as an important avenue of future work."

**The consequence.** The headline efficiency gains of $4\times$ over best-of-N (Figures 4 and 8) are computed *after* difficulty is already known, without amortizing the cost of learning it. Generating 2,048 samples per question is more expensive than the largest test-time budgets studied in the paper (max 256–512 generations). In a realistic deployment, total cost would be `difficulty_estimation_cost + strategy_execution_cost`, and the former could dominate the latter. For example, if a user queries a model with a math problem and the system must first generate 2,048 samples to decide how to allocate the remaining budget, the effective cost per query is at least $2048 + N$ generations. A naive best-of-256 approach ($256$ generations total) would be **cheaper than the difficulty estimation step alone**. The $4\times$ figure should therefore be understood as an **upper bound on achievable efficiency under the unrealistic assumption of free difficulty estimation** — not a realized deployment gain.

**What evidence exists in the paper.** The paper presents the compute-optimal vs. best-of-N curves in Figure 4 (search) and Figure 8 (revisions) with generation budget on the x-axis. The x-axis begins at $2^0 = 1$ generation and extends to $2^9 = 512$ generations. The difficulty estimation cost ($2048$ samples) is **not on the x-axis**. The curves are therefore missing a horizontal offset of +2048 that would dramatically shift the cost-effectiveness comparison. The paper mentions this cost in Section 3.2 but provides no empirical measurement of how the efficiency gains change when difficulty estimation is amortized, nor any analysis of how much accuracy is lost if difficulty is estimated with fewer samples (e.g., 16, 64, or 128 rather than 2048).

**Mitigation status.** Not addressed. The paper explicitly flags it as future work ("pretraining or finetuning models to directly predict difficulty") and the predicted (non-oracle) difficulty bins are offered as a step toward practicality since they remove the need for ground-truth labels, but the computational cost of the 2,048-sample estimation remains unaddressed. A lightweight difficulty classifier — possibly distilled from the PRM or trained on question text directly — would close this gap, but no such model is developed or evaluated. Until this is resolved, the compute-optimal framework is a proof of concept that identifies an upper bound on potential gains, not a deployable method.

---

### All Results Are on a Single Benchmark with a Single Model Family

**The assumption or constraint.** Every experiment in the paper uses the MATH benchmark (Hendrycks et al., 2021) evaluated on PaLM 2-S\* (Codey) models. The authors argue this model is representative (Section 4):

> "we believe this model is representative of the capabilities of many contemporary LLMs"

but provide no evidence for this representativeness claim outside the MATH benchmark. The paper does not evaluate on any other reasoning domain (code generation, logical deduction, scientific QA, multi-step planning) or any other model family (GPT, Claude, Llama, Mistral, Gemini).

**The consequence.** Several aspects of the findings could be model-specific or benchmark-specific in ways that the paper cannot diagnose:

- **PRM quality and over-optimization behavior**: The PRM is trained on PaLM 2-S\* outputs using Monte Carlo rollouts. A model with different calibration properties (e.g., overconfident on wrong answers, underconfident on right answers) would produce different PRM scores, different difficulty estimates, and potentially different over-optimization thresholds. The paper's key finding — that beam search hurts easy problems due to verifier exploitation (Figure 3 right) — may depend on the specific PRM quality achievable with PaLM 2-S\*. A better-calibrated PRM might not show this pattern, or a worse one might show it more severely.

- **Revision model dynamics**: The revision model's ability to learn from incorrect in-context examples depends on the base model's in-context learning capabilities, which vary substantially across model families. The finding that revisions help on easy problems but not hard ones (Figure 7) could be specific to PaLM 2-S\*'s particular in-context learning profile.

- **MATH-specific reasoning patterns**: MATH consists of competition-level problems requiring symbolic manipulation, algebraic reasoning, and formal proof-like steps. The difficulty-dependent patterns observed (beam search over-optimizes on easy problems, revisions help on easy problems, nothing helps on hard problems) may not generalize to other reasoning types — code generation (where test cases provide verifier signals), factual QA (where the challenge is retrieval, not deduction), or open-ended reasoning (where correctness is ambiguous). The paper's demonstration that difficulty-dependent allocation is optimal may be restricted to the kind of formal, verifiable, step-by-step reasoning that MATH problems exemplify.

**What evidence exists in the paper.** None outside MATH with PaLM 2-S\*. The paper does not even present a small-scale replication on another benchmark (e.g., GSM8K for easier math, or a subset of MBPP for code) to test whether the difficulty-dependent patterns replicate. The 500-question MATH test set is split into five difficulty quintiles of ~100 questions each, then further split by two-fold cross-validation for strategy selection, meaning the compute-optimal policy is **selected based on ~50 questions per fold per bin**. This is a very small sample for policy optimization — the selected strategies may not be robust even within MATH, let alone transferable to other benchmarks.

**Mitigation status.** Not addressed beyond acknowledging it in passing (Section 4 describes PaLM 2-S\* as "representative"). The paper does not propose cross-benchmark or cross-model validation as future work, nor does it discuss the risk that the difficulty-dependent patterns are MATH-specific. A minimal robustness check — evaluating the same methodology on a different math benchmark (GSM8K) or a different model family (even a different size of PaLM 2) — would substantially strengthen confidence in the generality of the findings.

---

### The $14\times$ Larger Baseline Model Is Not Compute-Optimally Trained, and the Comparison Favors Test-Time Compute

**The assumption or constraint.** The FLOPs-matched comparison in Section 7 scales model parameters while **holding training data fixed**, following the LLaMA paradigm (Touvron et al., 2023) rather than Chinchilla-optimal scaling (Hoffmann et al., 2022) where both data and parameters scale together. The paper explicitly acknowledges this:

> "We choose this setting as it is representative of a canonical approach to scaling pretraining compute and leave the analysis of compute-optimal scaling of pretraining compute where the data and parameters are both scaled equally to future work."

Additionally, the $14\times$ larger model is evaluated with **greedy decoding only** — no test-time compute augmentation of any kind (no majority voting, no best-of-N, no search), while the smaller model gets the full compute-optimal strategy suite.

**The consequence.** The comparison is systematically biased in favor of test-time compute. A Chinchilla-optimal model trained with $14\times$ more total FLOPs (scaling both parameters and data) would likely outperform a parameter-only-scaled model on the same total budget, since parameter-only scaling is known to be suboptimal for a given FLOPs budget. This means the reported advantages of test-time compute over pretraining — for example, revisions achieving +27.8% on easy questions at $R \ll 1$ (Figure 1 top-right), or +19.1% for PRM search on easy questions (Figure 1 bottom-right) — are **measured against a weaker-than-necessary baseline**. Against a properly compute-optimal larger model, these advantages would likely shrink or potentially reverse, especially for medium and hard questions.

Furthermore, giving the $14\times$ larger model *any* test-time compute (even a modest best-of-8 or majority voting over 8 samples) would create a much stronger baseline. The paper's framing — comparing a smaller model with sophisticated inference against a larger model with naive inference — conflates two axes (model size and inference strategy) in a way that favors the conclusion that test-time compute can substitute for pretraining. A fairer comparison would either (a) give both models their compute-optimal inference strategy, or (b) fix the inference strategy (e.g., best-of-N with the same $N$) and compare only the model size difference.

**What evidence exists in the paper.** The FLOPs-matched results in Figure 9 and Figure 1 (bar charts) show that test-time compute is preferable on easy problems across all $R$ regimes, competitive on medium problems at low $R$, and clearly worse on hard problems at high $R$. These comparisons use the $14\times$ larger model with greedy decoding as the baseline throughout. The paper does not include any ablation where the larger model also receives test-time compute, nor does it estimate what a Chinchilla-optimal $14\times$ larger model would achieve. The acknowledgment of the non-Chinchilla-optimal baseline is in the text (Section 7) but the headline takeaway — "test-time compute can outperform a $14\times$ larger model" — is presented in the abstract and Figure 1 without this qualification.

**Mitigation status.** The paper acknowledges the non-optimal pretraining baseline in Section 7 and flags the Chinchilla-optimal comparison as future work. However, it does not discuss the impact of giving the larger model test-time compute, and the abstract's framing ("a smaller model augmented with compute-optimal test-time strategies can outperform a ~14× larger pretrained model") does not include the caveat that the larger model uses greedy decoding. The finding is technically true as stated but the *magnitude* of the advantage is almost certainly overstated relative to a fairer comparison. A practitioner reading the abstract might incorrectly conclude that deploying a 14× smaller model with inference tricks is generally preferable to training a larger model, when the paper's own results show this is only true for easy problems at low inference-to-pretraining ratios with a suboptimally-trained larger baseline.

---

### Hard Problems Are Fundamentally Unsolved — Test-Time Compute Cannot Substitute for Missing Capability

**The assumption or constraint.** The entire compute-optimal framework assumes that the base model already produces correct solutions at a non-trivial rate for problems where test-time compute is deployed. Section 7 states this boundary condition explicitly:

> "Test-time compute provides minimal gains on problems that are fundamentally outside the base model's capability range."

**The consequence.** For the hardest difficulty quintile (bin 5), **no method makes meaningful progress regardless of budget**. In Figure 3 (right), beam search and best-of-N both achieve roughly 1–3% accuracy on bin 5 problems at all budgets from 4 to 256 generations. In Figure 7 (right), all sequential-to-parallel ratios produce approximately 2–3% accuracy on bin 5 problems. In the FLOPs-matched comparison (Figure 9), the bin 5 scaling line (blue, bottommost) is essentially flat near 0–5% across all budgets and all three $R$ regimes. Even the $14\times$ larger model with pretraining-only scaling performs poorly on these problems.

This is not a limitation that better difficulty estimation or strategy selection can fix — it is a fundamental capability bound. If the base model's pass@1 is near zero on a class of problems, there are essentially no correct solutions in the proposal distribution to find (via search) or refine (via revisions). Test-time compute can amplify existing capability but cannot create it from nothing. For problems that require novel reasoning, out-of-distribution generalization, or capabilities not present in the pretraining data, **pretraining remains the only viable path**. The paper does not provide a method for determining *in advance* whether a given problem is within the base model's capability range — the difficulty estimation procedure (2048 samples) *discovers* this post-hoc, but there is no way to know without spending the compute that a problem is bin 5 and therefore not worth spending compute on.

**What evidence exists in the paper.** The bin 5 results in Figures 3, 7, 9, and the FLOPs-matched comparison are the direct evidence. The paper is transparent about this limitation in the Section 7 discussion and the takeaway box, but the magnitude of the bin 5 failure is underemphasized relative to its practical importance. In a realistic deployment, a non-trivial fraction of user queries will be "hard" (outside the model's reach), and the compute-optimal framework provides no mechanism for efficiently identifying and triaging these queries.

**Mitigation status.** The paper acknowledges the limitation verbally but does not propose or evaluate any solution. A natural approach — using a lightweight "capability estimator" that predicts whether a problem is likely to be bin 5 based on question features (length, domain, required operations) without the 2,048-sample cost — is not discussed. Similarly, the paper does not explore whether a two-stage approach (quickly estimate if the problem is bin 5, and if so, escalate to a larger model or human) could salvage performance on hard problems while preserving the efficiency gains on easier ones. The current framework simply spends the allocated budget and fails, with no graceful degradation or escalation path.

---

### Revision Model Training Is Fragile and Produces a 38% Correct-to-Incorrect Reversion Rate

**The assumption or constraint.** The revision model is trained exclusively on sequences where all in-context answers are incorrect, followed by a correct target answer. As noted in Section 6.1:

> "A significant practical issue: since the model was trained only on sequences where all in-context answers are incorrect (followed by a correct target), at test time the model may encounter correct answers in its context (produced during earlier revisions) and incorrectly 'revise' them into wrong answers. The paper reports that approximately **38% of correct answers get converted back to incorrect ones** using a naive approach."

Additionally, the training data is constructed using offline pairing of independently sampled correct and incorrect solutions, with the last incorrect answer selected to have minimal character-level edit distance to the correct answer (Section 6.1). This is an approximation of Qu et al. (2024)'s on-policy multi-turn rollouts, adopted because on-policy generation was computationally infeasible for the authors.

**The consequence.** The revision model has a built-in failure mode: it **does not know what to do when the current answer is already correct**. Since it never saw correct→correct transitions during training (only incorrect→correct), its default behavior is to modify the input, even when no modification is needed. At 38% reversion rate, a long revision chain will have multiple correct answers generated and then lost, reducing the effective yield of the sequential sampling budget.

The offline data construction (pairing independently sampled solutions via edit distance) is a further approximation of the intended on-policy approach. The paper does not compare this offline construction to on-policy generation (e.g., Qu et al., 2024's original approach), so it is unclear how much performance is lost due to the mismatch between training trajectories (paired post-hoc) and test-time trajectories (genuinely sequential). The ReST$^{EM}$ experiment in Appendix K (Figure 16) provides indirect evidence of fragility: attempting to optimize the revision model with RL-style on-policy training caused performance to **degrade substantially** with sequential revisions, suggesting that the revision training procedure is sensitive to data generation methodology in ways that are not fully understood.

**What evidence exists in the paper.** The 38% reversion rate is reported in Section 6.1. The paper's mitigation — using majority voting or verifier-based selection across the entire revision chain to pick the best answer rather than taking the final revision — is applied throughout the revision experiments. Figure 6 (left) shows that pass@1 gradually improves across revision steps despite the reversion problem, but this improvement includes the selection mechanism's corrective effect. Without within-chain selection, performance could be substantially worse.

The ReST$^{EM}$ failure in Appendix K (Figure 16) shows that further optimizing the revision model degrades performance, particularly for sequential revisions. At 256 generations, the ReST$^{EM}$ model's fully-sequential performance drops to roughly 33.5% compared to roughly 38.5% at the optimal sequential-to-parallel ratio — evidence that the revision model's behavior is not robust to changes in the training data distribution.

**Mitigation status.** The paper acknowledges the reversion problem and applies within-chain selection as a patch, but this is an imperfect solution — it recovers correct answers that would otherwise be lost but does not prevent them from being lost in the first place. A more principled fix — training the model with correct→correct examples (teaching it to recognize when no revision is needed) or using a verifier-guided revision stopping criterion — is not explored. The offline-vs-on-policy training discrepancy is not measured or discussed as a limitation. Given that the revision model is one of the two main test-time compute mechanisms studied, the fragility of its training procedure is a significant practical concern for anyone attempting to replicate or deploy this approach.

## 7. Implications and Future Directions
- How this work reshapes the field
  - High performance on popular one-hop math benchmarks does not guarantee robust compositional reasoning. The introduction of Compositional GSM and the “reasoning gap” metric provides a sharper lens for evaluating reasoning reliability, particularly in cost‑efficient models that are attractive for deployment.

- Practical applications
  - Model selection: Teams should test candidate models on compositional tasks similar to their target use cases, not just on single-hop benchmarks.
  - Prompting and tooling: For smaller models, code generation or explicit function scaffolding (`solve_q1()` then `solution()`) can markedly improve composition (Figure 8).
  - Training strategy: Be cautious with extended fine‑tuning on benchmark-style data; monitor compositional metrics to avoid overfitting (Figure 7).

- Suggested research directions
  - Better second-hop training: Develop training curricula or objectives that explicitly encourage using intermediate results (e.g., supervised traces that require binding the symbol `X` and reusing it).
  - Distraction-robust prompting: Explore prompts that segment subproblems, enforce intermediate variable naming, or use structured memory to carry values between steps.
  - Beyond two hops: Extend the framework to three or more dependent steps, different dependency types (multiple substitutions, functional transformations), and other domains (e.g., MATH dataset, multimodal reasoning), as hinted in the Discussion.
  - Verification and tool use: Combine LLMs with lightweight program interpreters or verifiers to check and reuse intermediate results, possibly reducing second-hop errors without sacrificing interpretability.
  - Scaling studies with principled metrics: Use Δ as a standard metric when reporting reasoning results, to prevent over-optimism from single-hop accuracies.

> Bottom line (Discussion; Figures 1, 3–6, 10–12): “Not all LLM reasoners are created equal.” Many models—especially smaller, cheaper, and even math‑specialized ones—exhibit substantial deficits when knowledge must be composed across steps. Measuring and training for composition, not just single-step accuracy, is essential for trustworthy reasoning systems.
