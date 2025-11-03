# Large Language Models are Zero-Shot Reasoners

**ArXiv:** [2205.11916](https://arxiv.org/abs/2205.11916)

## 🎯 Pitch

This paper introduces Zero-shot-CoT, a deceptively simple yet powerful prompting strategy that enables large language models to perform complex multi-step reasoning tasks in a true zero-shot setting—simply by prepending 'Let’s think step by step' to the query. This innovation eliminates the need for hand-crafted, task-specific exemplars, drastically boosting zero-shot performance on challenging reasoning benchmarks and revealing that large language models have broad, underexplored reasoning capabilities that can be unlocked with minimal, universal prompts. This breakthrough both sets a new state-of-the-art baseline for zero-shot reasoning and changes our understanding of the latent cognitive potential in large-scale language models.

---

## 1. Executive Summary
This paper introduces Zero-shot-CoT, a simple prompting technique that turns large language models (LLMs) into effective zero-shot reasoners by prepending a short trigger such as “Let’s think step by step” before the answer. Using a two-stage prompt (first elicit the reasoning, then elicit the final answer), the method yields large accuracy gains on multi-step reasoning tasks without any task-specific examples, achieving, for instance, 78.7% vs 17.7% on MultiArith and 40.7% vs 10.4% on GSM8K with `text-davinci-002` (Table 1, Table 2).

## 2. Context and Motivation
- Problem addressed
  - LLMs perform well on single-step or intuitive tasks (“system-1”), but degrade on tasks requiring slow, multi-step reasoning (“system-2”) such as multi-hop arithmetic and logic (Section 1). Prior gains in such tasks largely came from chain-of-thought (CoT) prompting that supplies step-by-step exemplar answers, i.e., few-shot CoT (Figure 1a–b).
  - The gap: existing CoT methods rely on per-task few-shot examples and careful prompt engineering, while zero-shot baselines on these tasks have been weak and often underreported.

- Importance
  - Practical: Eliminates the need to curate task-specific few-shot exemplars, saving manual effort and enabling plug-and-play reasoning across diverse tasks.
  - Scientific: Reveals that broad, latent multi-step reasoning capabilities exist in LLMs and can be elicited with minimal, task-agnostic prompting (Abstract; Section 1).

- Prior approaches and shortcomings
  - Standard zero-shot prompting: typically asks for “The answer is …” and performs poorly on system-2 tasks (Figure 1c; Table 1).
  - Few-shot CoT: provides step-by-step exemplars and works well, but requires task-specific sample engineering and can be sensitive to example choice and format (Section 4.1; Table 5).
  - Instruction tuning and other prompt programming: often task- or format-specific and do not systematically unlock multi-step reasoning in zero-shot settings.

- Positioning
  - The paper proposes a single, task-agnostic zero-shot trigger (e.g., “Let’s think step by step”) paired with a two-stage prompting scheme that elicits both reasoning and an answer, substantially raising zero-shot performance across arithmetic, symbolic, and logical reasoning tasks (Figures 1d, 2; Table 1; Table 2). It reframes LLMs as capable zero-shot reasoners when prompted to “show their work.”

## 3. Technical Approach
The method is a prompting pipeline—no model training or finetuning—comprising two stages (Section 3; Figure 2):

- Core idea: a task-agnostic `trigger sentence` encourages the model to produce step-by-step reasoning before giving the final answer. Example trigger: “Let’s think step by step.” The key is to separate the elicitation of reasoning from the elicitation of the final answer.

- Stage 1 — Reasoning extraction
  - Format: `Q: [X]. A: [T]` where `[X]` is the question text and `[T]` is the trigger sentence (Section 3.1).
  - The model, using greedy decoding (temperature 0), generates a reasoning trace `z`—a multi-step explanation or computation leading toward an answer.
  - Analogy: you ask a student to “show your work” before they report the final numeric or multiple-choice answer.

- Stage 2 — Answer extraction
  - Inputs: concatenate the original prompt and the generated reasoning: `[X0] [Z] [A]`, where `[A]` is an answer-extraction cue matched to the task’s answer type (Section 3.1).
    - Examples of answer cues (Appendix A.5):
      - Numeric: “Therefore, the answer (arabic numerals) is”
      - Multiple-choice: “Therefore, among A through E, the answer is”
      - Yes/No: “Therefore, the answer (Yes or No) is”
  - The model then outputs the concise final answer.

- Answer cleansing (Section 4; Appendix A.6)
  - A robust post-processing step parses the first valid occurrence of the expected answer format in the model’s text (e.g., first number, first capital letter A–E, first “yes/no”). This reduces errors from verbose or hedged outputs.

- Design choices and why
  - Two-stage separation: The first stage elicits reasoning (which may be verbose and free-form), the second stage narrows to a cleanly formatted answer. In few-shot settings, examples implicitly shape both reasoning and answer formatting; here, the explicit second-stage cue replaces the need for exemplars (Section 3; Figure 2).
  - Single, task-agnostic trigger: Avoids per-task prompt engineering and demonstrates breadth across tasks (Table 1). The paper also probes trigger alternatives (Table 4).

- Practical details
  - Decoding: greedy across all experiments (so the zero-shot results are deterministic) except PaLM self-consistency runs (Section 4; Appendix D).
  - Models: InstructGPT (`text-ada/babbage/curie/davinci-001`, `text-davinci-002`), original GPT-3 (`ada/babbage/curie/davinci`), PaLM (8B/62B/540B), and others for scaling studies (GPT-2, GPT-Neo, GPT-J, T0, OPT) (Section 4; Appendix A.3).
  - Datasets: 12 tasks spanning arithmetic, symbolic, commonsense, and logical reasoning (Section 4; Appendix A.2).

## 4. Key Insights and Innovations
- A single zero-shot trigger reliably elicits multi-step reasoning across diverse tasks
  - Novelty: CoT previously required few-shot demonstrations; here, a fixed phrase (e.g., “Let’s think step by step.”) suffices to induce step-by-step reasoning in zero-shot mode (Figure 1d; Section 3).
  - Significance: Large accuracy gains in arithmetic and symbolic tasks without any examples (Table 1; Table 2).

- Two-stage prompting (reasoning then answer) as a minimal scaffold
  - Novelty: Explicitly separates reasoning generation from answer formatting (Figure 2), which few-shot setups had implicitly handled via curated examples.
  - Significance: Ensures answers appear in the right format and prevents the model from getting “stuck” in a long chain-of-thought without concluding.

- Scaling flips from flat to steep when reasoning is elicited
  - Observation: Standard zero-shot performance scales little with model size (flat curves), while Zero-shot-CoT exhibits strong gains as models get larger (Figure 3a–c).
  - Significance: Suggests that CoT-style prompting unlocks capacity that size alone does not reveal in zero-shot settings.

- Prompt robustness and failure modes are measurable
  - The method is sensitive to the formulation of the trigger: “instructive” phrasings help, “misleading/irrelevant” ones hurt (Table 4). This clarifies that the trigger’s semantics—and not just any extra token—drive gains.
  - Error analyses reveal typical pitfalls: overlong reasoning that changes a correct intermediate result, multiple answers in ambiguous cases, or failure to begin reasoning (Section 4.1 “Error Analysis”; Table 3; Appendix C).

Overall, the fundamental contribution is conceptual and practical: minimal, task-agnostic zero-shot prompting that elicits CoT and dramatically changes zero-shot reasoning performance and scaling behavior.

## 5. Experimental Analysis
- Evaluation setup
  - Tasks (12 total; Section 4; Appendix A.2)
    - Arithmetic: `SingleEq`, `AddSub`, `MultiArith`, `AQUA-RAT`, `GSM8K`, `SVAMP`
    - Symbolic: `Last Letter Concatenation`, `Coin Flip`
    - Commonsense: `CommonsenseQA`, `StrategyQA`
    - Other logical reasoning: `Date Understanding`, `Tracking Shuffled Objects`
  - Metrics: Accuracy (Section 4.1; tables report exact percentages).
  - Baselines: Standard zero-shot (answer-only cues), few-shot (standard exemplars), few-shot-CoT (step-by-step exemplars). For PaLM, also self-consistency (sampling multiple reasoning paths, then majority vote).
  - Decoding and control: Greedy decoding for determinism in zero-shot; fixed example order for few-shot (Section 4).

- Main quantitative results (all with `text-davinci-002` unless noted)
  - Arithmetic and symbolic gains are large (Table 1; Table 2):
    - MultiArith: 
      > Zero-shot 17.7% → Zero-shot-CoT 78.7% (Table 1).  
      > Against few-shot baselines (Table 2): Zero-shot-CoT 78.7% vs Few-shot (8-shot) 33.8% and approaches Few-shot-CoT (8-shot) 93.0%.
    - GSM8K:
      > Zero-shot 10.4% → Zero-shot-CoT 40.7% (Table 1; Table 2).  
      > It surpasses a finetuned GPT-3 175B baseline reported in prior work (33%) and approaches Few-shot-CoT (8-shot) 48.7% (Table 2).
    - AQUA-RAT:
      > Zero-shot 22.4% → Zero-shot-CoT 33.5% (Table 1).
    - SVAMP:
      > Zero-shot 58.8% → Zero-shot-CoT 62.1% (Table 1).
    - Symbolic reasoning:
      > Last Letter: Zero-shot 0.2% → Zero-shot-CoT 57.6% (Table 1).  
      > Coin Flip: Zero-shot 12.8% → Zero-shot-CoT 91.4% (Table 1).
    - Easier single-step arithmetic (SingleEq/AddSub): changes are small, consistent with less need for multi-step reasoning (Table 1).

  - Commonsense and other logical reasoning are mixed (Table 1; Section 4.1)
    - CommonsenseQA:
      > 68.8% (Zero-shot) → 64.6% (Zero-shot-CoT): slight drop.
    - StrategyQA:
      > 12.7% (Zero-shot) → 54.8% (Zero-shot-CoT): large gain (left-column prompts in Table 1). The right-column variant shows milder effects, underscoring sensitivity to answer cues.
    - Date Understanding and Tracking Shuffled Objects:
      > Date Understanding: 49.3% → 67.5%  
      > Shuffled Objects: 31.3% → 52.4%  
      (Table 1)

  - Scaling behavior (Figure 3)
    - On MultiArith, the original GPT‑3 line shows near-flat scaling without CoT but strong gains with Zero-shot-CoT, especially at 175B (`davinci`) (Figure 3a).
    - InstructGPT line likewise shows steep gains at larger sizes (`text-davinci-002`) with Zero-shot-CoT (Figure 3b).
    - On GSM8K with PaLM, Zero-shot-CoT improves from 2.4% (8B) → 10.5% (62B) → 43.0% (540B) vs much flatter zero-shot scaling (Figure 3c; Table 27).

  - PaLM + self-consistency (Appendix D, Table 25)
    - Zero-shot-CoT + self-consistency (40 samples) reaches:
      > MultiArith 89.0%, GSM8K 70.1%, SVAMP 80.5%, AQUA-RAT 46.5%  
      rivaling or surpassing reported few-shot-CoT results in some cases.

  - Prompt robustness (Table 4)
    - “Let’s think step by step.” is best on MultiArith (78.7%).
    - Other instructive variants (“First,” 77.3%; “Let’s think about this logically.” 74.5%) work well.
    - Misleading/irrelevant triggers collapse performance (e.g., “Don’t think. Just feel.” 18.8%).

  - Few-shot-CoT sensitivity to examples (Table 5)
    - Using few-shot exemplars from a different domain but the same answer format can still help (CommonsenseQA exemplars → AQUA-RAT: 31.9% vs Zero-shot 22.4%), yet results remain below Zero-shot-CoT (33.5%).
    - When answer format differs (CommonsenseQA exemplars → MultiArith), performance drops notably vs Zero-shot-CoT.

  - Error analysis and qualitative behavior
    - Commonsense: reasoning text is often plausible even when answers are wrong; models sometimes hedge with multiple options (Table 3; Appendix C).
    - Arithmetic: Zero-shot-CoT may overshoot—continue reasoning past the correct value and change it, or restate the question without reasoning (Section 4.1 “Error Analysis”; Table 24). Few-shot-CoT tends to stumble on ternary operations like `(3 + 2) * 4`.

- Assessment of support for claims
  - The numerical improvements on multi-step tasks are large, consistent, and replicated across model families and sizes (Table 1; Figure 3; Table 25), supporting the central claim that a minimal zero-shot trigger can elicit reasoning and change scaling behavior.
  - Mixed commonsense results are discussed and contextualized with qualitative analyses (Table 3), consistent with the claim that Zero-shot-CoT elicits reasoning even when metric gains are modest.

## 6. Limitations and Trade-offs
- Dependence on model scale
  - Zero-shot-CoT offers limited benefit for smaller models; strong gains appear for larger models (Figure 3; Table 26–27).

- Sensitivity to the phrasing of the trigger and answer cue
  - Performance varies with “instructive” vs “misleading/irrelevant” triggers (Table 4).
  - Some tasks show sensitivity to the answer-extraction cue (left vs right columns in Table 1 for StrategyQA).

- Two-stage complexity and verbosity
  - The method requires two separate prompts per question and can produce long reasoning traces; this has latency and cost implications in practice (Figure 2; Section 3.1).
  - Post-processing (“answer cleansing”) is necessary to parse final outputs, which can be brittle in edge cases (Section 4; Appendix A.6).

- Commonsense is not uniformly improved
  - CommonsenseQA sees a drop (68.8% → 64.6%), while StrategyQA improves strongly in one setting but less consistently in a variant (Table 1). This suggests task-dependent benefits.

- Training-data opacity and potential bias
  - The exact training corpora for several models (InstructGPT variants, PaLM) are not public in detail; the paper notes this as a limitation for interpreting results (Section 5 “Training Dataset Details”). As with all LLM prompting, outputs can reflect training-data biases.

- Faithfulness of reasoning
  - Even when answers are correct, the reasoning text can be partially incorrect or overinclusive (Table 3; Appendix C), raising questions about faithfulness vs plausibility.

## 7. Implications and Future Directions
- Impact on the field
  - Establishes a new, simple, strong zero-shot baseline for reasoning tasks: always try a CoT trigger and two-stage prompting before investing in task-specific exemplars or finetuning (Abstract; Section 6).
  - Shifts focus from “LLMs are few-shot learners” to “LLMs have untapped zero-shot reasoning abilities” that careful prompting can expose (Section 1; Discussion).

- Follow-up research enabled
  - Automatic trigger discovery and optimization: Table 4 shows large variance across triggers; learning or searching for better universal or task-tailored triggers is promising.
  - Reliability and faithfulness: develop verifiers or consistency checks that ensure the produced reasoning is correct and succinct, not just plausible (Table 2 references verifier-based work; Appendix D demonstrates self-consistency).
  - Efficient deployment: compress or prune reasoning while preserving answer accuracy; improve answer-extraction and parsing to reduce errors and latency.
  - Instruction-tuning synergy: investigate combined training or RL from human feedback to make reasoning elicitation more robust, especially for commonsense tasks where results are mixed.
  - Safety and bias auditing: the explicit reasoning text is a valuable artifact for auditing model biases and errors.

- Practical applications
  - Education and tutoring (transparent step-by-step solutions), data labeling (explanations plus answers), program synthesis or tool use with explicit intermediate steps, and automated planning where traceable reasoning is beneficial.

Block-quoted evidence highlights
- Large arithmetic gains (Table 2):
  > MultiArith: Zero-Shot 17.7 → Zero-Shot-CoT 78.7; Few-Shot-CoT (8) 93.0.  
  > GSM8K: Zero-Shot 10.4 → Zero-Shot-CoT 40.7; Few-Shot-CoT (8) 48.7.

- Symbolic reasoning jumps (Table 1):
  > Last Letter: 0.2% → 57.6%.  
  > Coin Flip: 12.8% → 91.4%.

- Scaling changes (Figure 3):
  > Without CoT, curves are flat; with Zero-shot-CoT, performance rises steeply with model size across GPT-3, InstructGPT, and PaLM.

- Prompt robustness (Table 4):
  > “Let’s think step by step.” 78.7% vs “Don’t think. Just feel.” 18.8% on MultiArith.

In sum, Zero-shot-CoT is a mechanism—not a model—that makes LLMs show their work, and when they do, zero-shot reasoning becomes both more accurate and more scalable.
