# Chain-of-Thought Prompting Elicits Reasoning in Large Language Models

**ArXiv:** [2201.11903](https://arxiv.org/abs/2201.11903)

## 🎯 Pitch

This paper introduces chain-of-thought prompting—a simple method where large language models are prompted with examples that include intermediate reasoning steps in natural language. By encouraging step-by-step explanations, the approach dramatically boosts the models’ abilities on tasks that require multi-step reasoning, from math problems to commonsense and symbolic logic. Crucially, the technique unleashes powerful reasoning skills that only emerge in very large models (100B+ parameters), showing that carefully crafted prompts can bridge the gap between knowledge and reasoning in AI without task-specific training.

---

## 1. Executive Summary (2-3 sentences)
This paper introduces chain-of-thought prompting: a way to get large language models (LLMs) to write short, step-by-step reasoning in natural language before giving an answer. Across arithmetic, commonsense, symbolic reasoning, and a robot-planning task, adding these reasoning steps dramatically improves accuracy—especially for very large models (around 100B+ parameters), where the ability to benefit from chain-of-thought “emerges” (Figure 4).

## 2. Context and Motivation
- Problem addressed
  - Few-shot prompting (showing a model a handful of input–output examples and then asking it to generalize) works well for simple tasks, but it is weak on problems that require multi-step reasoning: math word problems, multi-hop commonsense, and symbolic manipulation (Introduction; Sections 3–5).
  - Training or fine-tuning with reasoning rationales is effective but expensive to build at scale and robs models of the “single checkpoint for many tasks” advantage (Introduction).
- Importance
  - Practical: better math and logic improves tutoring, planning, data analysis, and robotics control.
  - Scientific: shows that language models can be guided to perform multi-step reasoning via natural language, not just by learning input–output mappings.
- Prior approaches and gaps
  - Finetuning with rationales or program execution modules can increase reasoning (Related Work; Introduction), but needs large labeled datasets and task-specific training.
  - Standard few-shot prompting often plateaus on reasoning tasks and does not improve much with model size (Figure 4; Rae et al., 2021 cited), leaving a gap between what models “know” and what they can reason out at inference.
- Positioning
  - This work combines the strengths of both worlds: use few-shot prompting (no gradient updates) but enrich each example with a brief chain of natural-language reasoning. The paper studies when and how this works, how robust it is, and why very large models benefit.

## 3. Technical Approach
What is chain-of-thought prompting?
- Definition: For each few-shot exemplar, provide a triple `⟨input, chain_of_thought, output⟩` instead of the usual `⟨input, output⟩`. The model is expected to emulate the pattern at test time by generating an intermediate “chain of thought” and then a final answer (Section 2; Figure 1; Figure 3).
- Example: For “Roger has 5 tennis balls. He buys 2 cans of 3 balls each. How many now?”, the chain-of-thought exemplar is “Roger started with 5… 2 cans × 3 = 6… 5 + 6 = 11. The answer is 11.” (Figure 1, right).

How it works at inference:
1. Produce a small prompt with 4–10 exemplars formatted as above (Appendix G contains all prompts).
2. Append the new question.
3. Decode greedily (no sampling) so the model first writes reasoning steps (one or a few sentences) and then the answer (Section 3.1).
4. Optionally, post-process any equations inside the reasoning with a Python “external calculator” and substitute computed results back in place (Table 1; Appendix B). This reduces arithmetic slip-ups.

Why this design?
- To let the model allocate tokens to intermediate steps and decompose problems (Section 2, bullets 1–3).
- To surface an interpretable trace that can be inspected for errors (Section 2, bullet 2; Appendix D).
- To keep the method inexpensive and model-agnostic—no finetuning, only prompting (Sections 1–2).

What they compare against
- Standard prompting: few-shot `⟨input, answer⟩` exemplars (Section 3.1; Figure 1 left).
- Ablations (Section 3.3; Figure 5; Table 6–7):
  - `Equation only`: Ask the model to write just the equation before the answer (no natural language steps).
  - `Variable compute only`: Ask the model to output a string of dots “...” of certain length before the answer—so it “spends tokens,” but without meaningful steps.
  - `Chain-of-thought after answer`: Put the reasoning steps only after the answer in exemplars, so the model isn’t incentivized to use reasoning to reach the answer.

Experimental design essentials
- Models tested (Section 3.1): `GPT-3` (350M–175B), `LaMDA` (0.4B–137B), `PaLM` (8B–540B), `UL2 20B`, and `Codex` (code-davinci-002).
- Decoding: greedy (Section 3.1); later work suggests sampling + majority vote (“self-consistency”) can help further (Wang et al., 2022a).
- Benchmarks (Figures 3–4; Table 1–5):
  - Arithmetic: GSM8K, SVAMP, ASDiv, AQuA, MAWPS (+ four MAWPS subsets; Table 3).
  - Commonsense: CSQA, StrategyQA, BIG-bench Date Understanding, Sports Understanding, and SayCan robot planning (Figure 7; Table 4).
  - Symbolic: Last Letter Concatenation and Coin Flip state tracking with in-domain and out-of-domain (OOD) lengths (Figure 8; Table 5).
- Metric: accuracy (“solve rate”)—exact answer match or correct choice (Tables 1–5).

## 4. Key Insights and Innovations
1. Eliciting multi-step reasoning by prompting alone
   - Novelty: No finetuning. Just show a handful of worked examples, and very large models start producing their own step-by-step reasoning.
   - Significance: Enables a single, off-the-shelf model to perform many reasoning-heavy tasks without retraining (Sections 1–2).

2. Emergent ability at scale
   - Observation: Chain-of-thought helps only when models are large (~100B+). Smaller models tend to produce fluent but illogical steps and often get worse than standard prompting (Figure 4; Table 2).
   - Why it matters: Shows a qualitative shift with scale—reasoning-like behavior is not a simple linear extrapolation from small models (Appendix A.1; Figure 9–10).

3. Natural language steps outperform “equation only” or empty “extra tokens”
   - Evidence: On GSM8K with LaMDA 137B, `equation only` yields 5.4% vs 14.3% for full chain of thought; `variable compute only` and `reasoning after answer` stay near baseline (6.4% and 6.1%) (Table 6; Figure 5).
   - Insight: It’s not just more tokens; content matters. Writing the reasoning in language helps the model map semantics to operations (Section 3.3; Appendix A.4).

4. Robustness across annotators and exemplars (with variance)
   - Finding: Different chains written by different people or sampled from GSM8K still beat standard prompting by large margins (Figure 6; Table 6–7).
   - Caveat: There is variance (especially for classification-style tasks like Coin Flip where label bias can creep in), but gains persist (Table 7; Section 3.4).

5. Length generalization for symbolic tasks
   - Result: With chain-of-thought, PaLM 540B generalizes to sequences longer than seen in exemplars—e.g., Last Letter Concatenation for 3–4 words after seeing only 2-word exemplars (Figure 8; Table 5).
   - Importance: Suggests the model learns a procedure that scales to longer inputs, not just shallow pattern copying.

## 5. Experimental Analysis
Evaluation setup
- Datasets and tasks
  - Arithmetic: GSM8K (grade-school math; free response), SVAMP/ASDiv/MAWPS (word problems), AQuA (algebra, multiple choice) (Section 3.1; Table 12).
  - Commonsense: CSQA, StrategyQA, Date Understanding, Sports Understanding, SayCan robot action planning (Section 4; Figure 3).
  - Symbolic: Last Letter Concatenation and Coin Flip, with in-domain (same number of steps as exemplars) and out-of-domain (longer sequences) (Section 5).
- Baseline vs. Chain-of-Thought vs. Ablations (Sections 3.1–3.4; Tables 1–7; Figures 4–6, 7–8, 11).

Main quantitative results

Arithmetic (Tables 1–2; Figure 4)
- GSM8K:
  - PaLM 540B: 17.9% (standard) → 56.9% (CoT) → 58.6% with external calculator.
  - GPT-3 175B: 15.6% → 46.9% → 49.6% with calculator.
  - Codex (code-davinci-002): 19.7% → 63.1% → 65.4% with calculator.
  - Claim of new state-of-the-art at the time: PaLM 540B + CoT surpasses prior best (Figure 2; Table 1 footnotes).
- SVAMP and MAWPS:
  - PaLM 540B: SVAMP 69.4% → 79.0%; MAWPS 79.2% → 93.3%.
- AQuA and ASDiv:
  - PaLM 540B: AQuA 25.2% → 35.8%; ASDiv 72.1% → 73.9% (smaller gain).
- Where gains are small
  - For one- or two-step subsets (MAWPS SingleOp/SingleEq/AddSub), improvements are minimal or even negative because baselines are already high (Table 3).

Commonsense (Figure 7; Table 4)
- PaLM 540B:
  - StrategyQA: 68.6% (standard) → 77.8% (CoT), surpassing prior best single-model result of 69.4%.
  - Sports Understanding: 80.5% → 95.4%, outperforms a reported human enthusiast benchmark (84%).
  - Date Understanding: 49.0% → 65.3%.
  - CSQA: modest gain 78.1% → 79.9%.
  - SayCan robot planning: 80.8% → 91.7%.

Symbolic reasoning and OOD generalization (Figure 8; Table 5)
- PaLM 540B:
  - Last Letter Concatenation: in-domain (2 words) 7.6% (standard) → 99.4% (CoT); OOD 3 words 0.2% → 94.8%; OOD 4 words 0.0% → 63.0%.
  - Coin Flip: in-domain 98.1% → 100%; OOD 3 flips 49.3% → 98.6%; OOD 4 flips 54.8% → 90.2%.

Ablations and robustness (Figure 5–6; Tables 6–7)
- On GSM8K with LaMDA 137B:
  - `equation only`: 5.4%; `variable compute only`: 6.4%; `reasoning after answer`: 6.1%; full CoT: 14.3% vs 6.5% standard (Table 6).
- Across annotators on GSM8K with LaMDA 137B:
  - Different writers still beat standard: 15.5% (B), 17.6% (C), 11.1% (concise style) vs 6.5% baseline (Table 6).
- Number of exemplars sensitivity:
  - Improvements hold from 1 to 8 exemplars across several datasets (Figure 11).

Qualitative/error analyses (Appendix D; A.1)
- For 50 GSM8K examples with correct final answers from LaMDA 137B, 49/50 had logically correct chains; only 1 was “correct by chance” (Table 8–9).
- For 50 wrong answers, the main fixable error types were:
  - “Calculator error only” (8%): correct reasoning but arithmetic slip (Table 10).
  - “Symbol mapping error” (16%): misapplied numbers while the reasoning steps were right.
  - “One step missing” (22%).
  - The rest (54%) had semantic misunderstandings or incoherences (Table 11).
- Scaling from PaLM 62B to 540B fixes many “one step missing” and “semantic understanding” errors (Appendix A.1; Figure 9–10).

Do the experiments support the claims?
- Yes, for large models and multi-step tasks. The combination of broad benchmarks, clear baselines, strong numbers (especially on GSM8K, StrategyQA, Sports), and careful ablations gives a convincing picture that chain-of-thought prompting—not just more tokens or better exemplar ordering—drives the gains (Figures 4–8; Tables 1, 4, 6–7).
- Caveats: benefits are small on some tasks (CSQA, easy MAWPS subsets), chain-of-thought can hurt in some settings (e.g., LaMDA 137B on AQuA, Table 2), and results depend on model scale.

## 6. Limitations and Trade-offs
Assumptions and scope
- Relies on very large models: The “emergent” benefit shows up near ~100B parameters (Figure 4; Table 2). Small models often produce illogical chains and may degrade performance.
- Chains are not guaranteed to be truthful or the causal path to the answer:
  - For free-response arithmetic, correct answers usually had correct chains (Appendix D.1), but in multiple-choice or yes/no settings it is easier to reach the right answer with spurious reasoning (Section 6; Appendix D.1–D.2).
- Prompt sensitivity:
  - While robust across annotators and exemplar sets (Figure 6; Tables 6–7), there is still variance, especially for classification-style tasks where label bias and order effects matter (Table 7; Section 3.4).

Computational and practical constraints
- Serving very large models is expensive (Section 6). The approach presumes access to 100B+ parameter LLMs.
- Longer generations: chains add tokens and latency.

Coverage gaps and edge cases
- Limited math tool use: an external calculator helps when equations appear, but the method does not integrate broader tool-use (e.g., retrieval, formal solvers) beyond this post-hoc step (Table 1).
- Dataset dependence: AQuA (algebra multiple-choice) shows mixed behavior; chain-of-thought slightly hurts for LaMDA 137B (Table 2).
- Not a full theory of reasoning: the work demonstrates elicited behaviors but does not prove the model “reasons” in a human-like cognitive sense (Section 6).

Open questions
- What exactly about scale enables the jump in usable chain-of-thought? Preliminary analysis suggests improvements in semantic understanding and step completeness (Appendix A.1), but a causal explanation is open.

## 7. Implications and Future Directions
How this changes the landscape
- Reframes inference-time control: Natural-language scaffolding can reliably “unlock” multi-step reasoning in large models without finetuning.
- Expands the useful capability frontier: Many tasks previously thought to require special architectures or finetuning become approachable with the right prompt.

Follow-up research it enables
- Better decoding with multiple chains:
  - Sample multiple reasoning paths and majority-vote on final answers (“self-consistency”), already shown to improve beyond greedy decoding (Section 3.1; Wang et al., 2022a).
- Verification and tool-use:
  - Train or prompt verifiers to score chains; integrate calculators, search, or program execution to correct arithmetic and factual errors (Table 1; Related Work on verifiers).
- Distillation to smaller models:
  - Use chains generated by large models to finetune smaller ones, reducing serving costs while keeping reasoning gains (Section 6, last paragraph).
- Automated prompt/chain generation:
  - Generate and select chains automatically for new tasks, optimizing over validation sets (Appendix A.2, closing paragraph).
- Broader domains:
  - Apply chain-of-thought to planning (beyond SayCan), scientific problem solving, data analysis pipelines, and interactive tutoring.

Practical applications
- Education/tutoring: Solve and explain math problems with traceable steps (Figure 1; GSM8K results).
- Decision support: Multi-hop QA, commonsense checks, and date reasoning (Figure 7).
- Robotics and agents: Translate natural-language goals into stepwise plans (SayCan results in Table 4; Figure 3).

> Core takeaway (Figure 4; Table 1): With eight worked examples that include brief reasoning, a 540B-parameter model jumps from 17.9% to 56.9% accuracy on GSM8K (58.6% with an external calculator), achieves 79.0% on SVAMP, and 93.3% on MAWPS—evidence that writing down intermediate steps, not just predicting outputs, is a powerful control knob for eliciting reasoning in large language models.
