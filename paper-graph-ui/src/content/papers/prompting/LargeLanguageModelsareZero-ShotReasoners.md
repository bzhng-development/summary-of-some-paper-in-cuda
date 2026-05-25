# Large Language Models are Zero-Shot Reasoners

**ArXiv:** [2205.11916](https://arxiv.org/abs/2205.11916)

## 🎯 Pitch

This paper reveals that large language models can perform impressive multi-step reasoning in a true zero-shot setting by simply adding the phrase “Let’s think step by step” before the answer—no hand-crafted examples or task-specific prompt engineering required. By introducing Zero-shot Chain-of-Thought (Zero-shot-CoT), the authors achieve dramatic accuracy improvements across diverse challenging reasoning tasks, demonstrating that LLMs possess powerful, generalizable reasoning abilities that can be unlocked with minimal, universal prompting—reshaping how we evaluate, prompt, and deploy these models for complex cognitive tasks.

---

## 1. Executive Summary
This paper introduces Zero-shot Chain-of-Thought (Zero-shot-CoT): a simple prompting technique that makes large language models reason step-by-step without any task-specific examples by prepending “Let’s think step by step.” before the answer. Using a two-stage prompting pipeline, this single, task-agnostic prompt yields very large gains on multi-step reasoning benchmarks (e.g., MultiArith: 17.7% → 78.7%; GSM8K: 10.4% → 40.7% with `text-davinci-002`) while revealing how performance scales with model size (Figure 3; Tables 1–2).

## 2. Context and Motivation
- Problem addressed
  - Multi-step reasoning (e.g., arithmetic word problems, logic puzzles) has historically been weak for large language models (LLMs) in zero-shot settings, even when those models excel at single-step “intuitive” tasks.
  - Prior improvements came from Chain-of-Thought (CoT) prompting that supplies worked, step-by-step solutions as few-shot exemplars. But this requires careful, task-specific prompt engineering.

- Why this matters
  - Reducing dependence on hand-crafted, per-task exemplars makes LLMs more usable and robust across diverse tasks and domains.
  - Understanding whether LLMs can perform “system-2” reasoning (slow, multi-step deliberation) without examples informs how we should evaluate, prompt, and scale these models.

- Prior approaches and their limits
  - Few-shot CoT: Demonstrates big gains on arithmetic and logic (e.g., Wei et al., 2022), but requires engineered reasoning examples per task (Figure 1a–b).
  - Standard zero-shot prompting: Works for many single-step tasks but struggles on multi-step reasoning; scaling curves are flat on such tasks (Figure 3).
  - No quantitative zero-shot CoT baseline had been established for a broad set of reasoning benchmarks.

- Positioning of this work
  - The paper shows LLMs already possess latent zero-shot reasoning ability that can be triggered by a minimal, task-agnostic instruction. It establishes Zero-shot-CoT as the strongest zero-shot baseline across a wide variety of reasoning tasks (Abstract; Table 1), and analyzes scaling, robustness, and failure modes.

## 3. Technical Approach
Zero-shot-CoT is a two-stage, template-based prompting pipeline designed to (1) elicit reasoning and then (2) extract a clean final answer.

Key terms
- `zero-shot prompting`: asking a model to perform a task using instructions or a template but without any task examples.
- `few-shot prompting`: including a few example question–answer pairs in the prompt.
- `Chain-of-Thought (CoT)`: asking the model to “show its work” by generating intermediate reasoning steps.
- `self-consistency`: sample multiple reasoning paths and use majority vote over the final answers (Wang et al., 2022). Used here as an optional extension in Appendix D/Table 25.
- `system-1 vs system-2 tasks`: fast, intuitive tasks vs slow, multi-step reasoning tasks (Introduction).

Step-by-step pipeline (Figure 2; Section 3)
1) Reasoning extraction
   - Input question `x` is formatted as `Q: [X]. A: [T]` where `[T]` is a reasoning trigger sentence, most notably `Let’s think step by step.` (Section 3.1).
   - The model generates a free-form reasoning continuation `z` (greedy decoding used throughout; Section 4, A.4).
   - Example (Figure 2): After the trigger, the model writes out multi-step calculations or logical deductions.

2) Answer extraction
   - A second prompt concatenates the original prompt and generated reasoning, plus an answer trigger string that matches the desired answer format (e.g., numbers, multiple-choice letter):
     - For numbers: `Therefore, the answer (arabic numerals) is`
     - For multiple-choice: `Therefore, among A through E, the answer is`
     - See Appendix A.5 for the full list and Table 9–10 for prompts used per dataset.
   - The model outputs the final answer `ŷ`, which is then parsed via simple `answer cleansing` heuristics (Section 4 “Answer cleansing”; Appendix A.6), e.g.:
     - Numbers: extract the first number found.
     - Multiple-choice: extract the first capital letter A–E.
     - Yes/No: extract the first “yes” or “no”.

Why this design?
- Two stages enforce a separation between “explain your reasoning” and “state the answer cleanly,” improving formatting reliability without handcrafted few-shot examples or task-specific solutions (Section 3; Figure 2).
- Minimal, general triggers reduce the need for per-task engineering yet reliably induce step-by-step reasoning (Figure 1d).

Implementation details
- Greedy decoding across all methods and models (Section 4 “Baselines” and A.4).
- OpenAI API for `GPT-3`/`InstructGPT`; HuggingFace for `OPT`, `T0`, `GPT-J`, `GPT-Neo`, `GPT-2` (A.4).
- For PaLM, TopK=1 (greedy) and max tokens 256 (A.4).
- Stop sequence “Q:” is used (except in InstructGPT) to prevent the model from generating new Q/A pairs on its own (A.4).

## 4. Key Insights and Innovations
1) A single, task-agnostic trigger can unlock zero-shot reasoning
   - Core idea: prepend `Let’s think step by step.` before the answer to elicit step-by-step reasoning across diverse tasks (Abstract; Figure 1d; Section 3).
   - Significance: Avoids few-shot CoT’s need for curated, task-specific examples while preserving much of the reasoning benefit (Table 2).

2) Two-stage prompting to cleanly extract the answer
   - The explicit “answer extraction” stage with tailored answer triggers yields consistent output formats across tasks (Section 3.1; Appendix A.5), enabling fair evaluation and automation.
   - This is a pragmatic, generally applicable tactic for template-based reasoning.

3) Scaling laws re-emerge for reasoning under Zero-shot-CoT
   - Without CoT, scaling curves for reasoning benchmarks are mostly flat; with Zero-shot-CoT, accuracy increases sharply with model size (Figure 3; Tables 26–27). Example: PaLM 540B on GSM8K, zero-shot 12.5% → Zero-shot-CoT 43.0% (Table 2).

4) Robustness and sensitivity to the prompt’s phrasing
   - In a systematic template study (Table 4), instructive variants (e.g., `Let’s think step by step.`: 78.7% on MultiArith) substantially outperform misleading or irrelevant variants (e.g., `Don’t think. Just feel.`: 18.8%).
   - This both validates the mechanism (explicitly eliciting reasoning) and highlights open problems in automatic trigger design.

5) What few-shot examples really provide
   - Cross-task few-shot CoT improves most when answer formats match, even if the task domain differs (Table 5), supporting the view that in-context examples often teach output format rather than task-specific reasoning. This clarifies why a format-plus-reasoning trigger can go far in zero-shot settings.

## 5. Experimental Analysis
Evaluation setup
- Tasks (12 datasets across 4 categories; Section 4 “Tasks and datasets”; Appendix A.2)
  - Arithmetic: SingleEq, AddSub, MultiArith, AQUA-RAT, GSM8K, SVAMP.
  - Commonsense: CommonsenseQA, StrategyQA.
  - Symbolic: Last Letter Concatenation, Coin Flip.
  - Other logical reasoning: BIG-bench Date Understanding, Tracking Shuffled Objects.
- Models (Section 4 “Models”; Appendix A.3)
  - `InstructGPT-3` variants (`text-ada/babbage/curie/davinci-001`, `text-davinci-002`) used as main.
  - `Original GPT-3` (ada, babbage, curie, davinci).
  - `PaLM` (8B, 62B, 540B).
  - Additional: GPT-2, GPT-Neo, GPT-J, T0, OPT (used in scaling study).
- Baselines (Section 4 “Baselines”)
  - `Zero-shot`: standard zero-shot prompting with answer triggers.
  - `Few-shot` and `Few-shot-CoT`: 8 in-context exemplars from Wei et al. (2022), same across comparisons (Table 2).
  - All use greedy decoding for determinism.
- Metric: Accuracy across all datasets (Tables 1–2).

Main quantitative results
- Large gains on multi-step arithmetic and other logical tasks (Table 1; Table 2)
  - On `text-davinci-002`:
    - MultiArith: Zero-shot 17.7% → Zero-shot-CoT 78.7%.
    - GSM8K: Zero-shot 10.4% → Zero-shot-CoT 40.7%.
    - AQUA-RAT: 22.4% → 33.5%.
    - SVAMP: 58.8% → 62.1%.
    - BIG-bench Date Understanding: 49.3% → 67.5%; Tracking Shuffled Objects: 31.3% → 52.4%.
    - Symbolic tasks: Last Letter 0.2% → 57.6%; Coin Flip 12.8% → 91.4%.
  - On `PaLM 540B` (Table 2; Appendix D/Table 25):
    - GSM8K: Zero-shot 12.5% → Zero-shot-CoT 43.0%; with self-consistency: 70.1%.
    - MultiArith: Zero-shot 25.5% → 66.1%; with self-consistency: 89.0%.
- Comparison to few-shot methods (Table 2)
  - Zero-shot-CoT underperforms Few-shot-CoT (with curated reasoning examples), but beats standard Few-shot prompting even with 8 examples (e.g., MultiArith: Few-shot 33.8% vs Zero-shot-CoT 78.7%).
  - Augmenting few-shot exemplars with the zero-shot trigger (`Zero-Plus-Few-Shot-CoT`) further improves GSM8K (48.7% → 51.5%; Table 2).
- Mixed results on commonsense QA (Table 1; Table 3)
  - CommonsenseQA slightly drops (68.8% → 64.6%), while StrategyQA improves dramatically (12.7% → 54.8%).
  - Error analyses show reasoning often reads sensible but can yield multiple choices instead of a single answer (Table 3, Appendix C/Table 22).

Ablations and robustness
- Prompt phrasing sensitivity (Table 4)
  - Top prompts (MultiArith accuracy): `Let’s think step by step.` 78.7%; `First,` 77.3%; `Let’s think about this logically.` 74.5%.
  - Misleading/irrelevant prompts do not help (e.g., `It’s a beautiful day.` 13.1%).
- Cross-task few-shot exemplar transfer (Table 5)
  - When answer formats match (CommonsenseQA exemplars → AQUA-RAT), Few-shot-CoT gains are meaningful (31.9%) but still below Zero-shot-CoT (33.5%) and task-matched Few-shot-CoT (39.0%).
- Scaling (Figure 3; Tables 26–27)
  - Without CoT: reasoning accuracy barely improves with model size.
  - With Zero-shot-CoT: accuracy rises sharply with model size across GPT-3 and PaLM.

Failure modes and qualitative analysis
- Arithmetic (Appendix C/Table 23–24)
  - Zero-shot-CoT sometimes continues reasoning after reaching the correct answer and changes it (one unnecessary step error).
  - Occasionally fails to start reasoning, merely paraphrasing the question.
  - Few-shot-CoT often fails on ternary operations like `(3 + 2) * 4`.
- Commonsense (Table 3; Appendix C/Table 22)
  - Reasoning is often coherent but can be indecisive (outputs “A, B, and D”) or rely on flexible but wrong world knowledge.

Do the experiments support the claims?
- Yes, strongly for arithmetic, symbolic, and structured logical tasks: large, consistent gains across multiple datasets and models (Tables 1–2; Figure 3).
- Commonsense QA is mixed: StrategyQA benefits substantially even in zero-shot, while CommonsenseQA can drop. The paper is transparent about this (Table 1; Table 3).

Selected supporting quotes
- > “increasing the accuracy on MultiArith from 17.7% to 78.7% and GSM8K from 10.4% to 40.7% with … text-davinci-002” (Abstract; Tables 1–2).
- > “add ‘Let’s think step by step’... to extract step-by-step reasoning” (Section 3; Figure 1d).
- > “performance drastically increases with chain of thought reasoning, as the model size gets bigger” (Figure 3; Tables 26–27).

## 6. Limitations and Trade-offs
- Prompt sensitivity
  - Performance depends on phrasing; not all “reasoning-like” triggers work (Table 4). Designing robust, automatic prompts remains open.
- Model size dependency
  - The method is far less effective for small models; gains materialize with large-scale LLMs (Figure 3; Tables 26–27).
- Two-stage overhead and parsing heuristics
  - Requires two prompts per question and post-hoc answer cleansing (Section 3; A.6), which can be brittle (e.g., if the model outputs multiple numbers).
- Mixed outcomes on commonsense QA
  - Zero-shot-CoT helps some commonsense tasks (StrategyQA) but can hurt others (CommonsenseQA) (Table 1); reasoning text may be plausible yet indecisive (Table 3).
- Data transparency and reproducibility
  - Training data details for closed-source models (InstructGPT, PaLM) are limited; the paper addresses this but cannot fully control for potential data leakage (Discussion “Training Dataset Details”).
- Ethical considerations
  - As with all LLM prompting, the approach can reproduce biases present in training data; chain-of-thought can amplify or mask such biases (Discussion “Limitation and Social Impact”).

## 7. Implications and Future Directions
- Field impact
  - Establishes a new, strong zero-shot baseline for reasoning: you can often get most of the “CoT effect” without any exemplars by adding a single sentence. This reframes zero-shot evaluation and makes reasoning-capable deployment more practical.
  - Reveals that multi-step reasoning can be unlocked by instruction alone, reshaping how we think about in-context learning vs innate model capabilities.

- Practical applications
  - Immediate drop-in prompt for numeracy-heavy assistants, tutoring systems, and logical assistants where few-shot curation is costly or infeasible.
  - Data labeling aid: Zero-shot-CoT can generate rationales to bootstrap datasets or facilitate human-AI collaboration.

- Research directions
  - Automated prompt discovery: Learn or search for robust, domain-agnostic reasoning triggers (Table 4 suggests significant headroom).
  - Verification and calibration: Combine Zero-shot-CoT with verification (e.g., trained verifiers or `self-consistency`, Table 2 and Appendix D) to improve reliability.
  - Reasoning control: Mechanisms to stop when the correct answer is reached and avoid “over-reasoning” that flips the answer (Appendix C/Table 23–24).
  - Commonsense specialization: Investigate why StrategyQA benefits but CommonsenseQA sometimes degrades; explore hybrid prompting or knowledge-grounding.
  - Open model scaling: Replicate and extend with open models to decouple progress from closed APIs, and to analyze data effects more transparently.

In short, this work shows that a minimal, universal instruction can convert many large language models into competent zero-shot reasoners, especially on arithmetic and structured logic tasks. The simplicity, breadth, and scaling behavior make Zero-shot-CoT both a practical tool and a conceptual lens for probing the latent reasoning abilities of LLMs.
