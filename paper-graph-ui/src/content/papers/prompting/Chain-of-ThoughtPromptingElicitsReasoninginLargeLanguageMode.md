# Chain-of-Thought Prompting Elicits Reasoning in Large Language Models

**ArXiv:** [2201.11903](https://arxiv.org/abs/2201.11903)

## 🎯 Pitch

This paper introduces chain-of-thought (CoT) prompting, a simple yet powerful method for unlocking multi-step reasoning in large language models by providing a few exemplars with intermediate reasoning steps. By eliciting stepwise natural-language rationales at inference time—without any additional training—CoT prompting enables models to dramatically improve performance on complex arithmetic, commonsense, and symbolic reasoning tasks, setting new state-of-the-art results on challenging benchmarks. This approach demonstrates that sufficiently large language models can reason more effectively and flexibly, vastly expanding their practical value for real-world reasoning applications.

---

## 1. Executive Summary
This paper introduces chain-of-thought (CoT) prompting: a simple way to elicit step-by-step natural‑language reasoning in large language models (LLMs) by placing a few example solutions that include intermediate reasoning steps in the prompt. Across arithmetic, commonsense, and symbolic reasoning tasks, CoT prompting dramatically improves performance—often only for sufficiently large models—and reaches state-of-the-art on GSM8K math word problems using PaLM 540B (see Figure 2 and Figure 4).

## 2. Context and Motivation
- Problem addressed
  - Large language models do well on many tasks but still struggle on multi-step reasoning such as math word problems, multi-hop commonsense, and symbolic manipulation. Scaling model size alone has not solved these tasks; standard few-shot prompting (input–output pairs without reasoning) often yields flat scaling curves (Section 1; Figure 4).
- Why this matters
  - Reasoning tasks underpin real applications (education, planning, data analysis, robotics). A general method that unlocks reasoning without fine-tuning would make LLMs more useful and reduce the need for task-specific training.
- Prior approaches and gaps
  - Rationale-augmented training/finetuning teaches models to produce explanations but requires collecting many labeled rationales, which is costly (Section 1).
  - Neuro‑symbolic methods use formal languages or program execution; they often require specialized architectures or supervision (Section 1; Related Work Appendix C).
  - Standard few-shot prompting avoids training but fails on complex reasoning and does not consistently improve with scale (Section 1; Figure 4).
- Positioning
  - This work combines the strengths of rationales and prompting: it uses a few in-context demonstrations that include a short “chain of thought,” requiring no training while giving the model a template for step-by-step reasoning (Section 2; Figure 1, Figure 3). The paper evaluates this across many datasets and model families to show breadth and robustness.

## 3. Technical Approach
Core idea: Instead of prompting with input–answer pairs, provide triples: `⟨input, chain of thought, final answer⟩`. At test time, the model is asked to produce its own chain-of-thought reasoning followed by the answer (Section 2; Figure 1).

How it works in practice
- Few-shot exemplars
  - For math word problems, the prompt contains eight exemplars with step-by-step reasoning (Appendix G, Table 20). The same eight were reused across four math benchmarks to test generality (Section 3.1).
  - For AQuA (multiple-choice algebra), four exemplars from the training set are used (Appendix G, Table 21).
  - For commonsense tasks (CSQA, StrategyQA) and the BIG-bench subsets (Date and Sports Understanding), the authors wrote CoT exemplars or used the first ten examples in the evaluation set as exemplars and evaluated on the remainder (Section 4; Figure 3; Appendix G, Tables 24–27).
  - For the robotic planning dataset SayCan, six exemplars include a short “Explanation” and a program-like “Plan” (Appendix G, Table 28).
  - For symbolic tasks, exemplars show the exact step pattern to follow (last-letter concatenation; coin-flip state tracking; Figure 3; Appendix G, Tables 22–23).
- Decoding and models
  - Greedy decoding (no sampling) is used for all main results; later work shows self-consistency can further help (Section 3.1).
  - Multiple model families are tested: `GPT-3` (350M–175B), `LaMDA` (422M–137B), `PaLM` (8B–540B), `UL2 20B`, and `Codex` (Section 3.1). This breadth lets the paper analyze how effects depend on scale and architecture.
- Why this approach?
  - The hypothesis is that natural-language intermediate steps guide the model to decompose problems, allocate more computation to reasoning, and surface interpretable intermediate structure (Section 2, bullets 1–3).
- Key variants and controls (Ablations; Figure 5; Appendix Tables 6–7)
  - `Equation-only`: prompt the model to output the key equation before the final answer, but no narrative reasoning.
  - `Variable compute only`: prompt the model to output a sequence of dots with length matched to the equation length—controls for “more tokens = more compute” without reasoning content.
  - `Chain of thought after answer`: place the reasoning after the final answer—controls for whether CoT merely “activates” knowledge rather than supports sequential reasoning.
- External calculator (post-hoc tool use)
  - For arithmetic, a simple Python evaluator is applied to equations found in the generated chain-of-thought and its result is propagated to later steps by string matching. This isolates arithmetic errors from reasoning errors (Appendix B, Table 1).

Intuition with a toy example
- Standard prompting on “Roger has 5 balls, buys 2 cans of 3 each—how many now?” expects the model to jump directly to “11.”
- CoT prompting provides an exemplar like: “Roger started with 5. 2 cans of 3 = 6. 5 + 6 = 11.” The model learns to mimic this step structure on new problems, producing intermediate steps that it can reliably follow (Figure 1, right; Figure 3, top left).

## 4. Key Insights and Innovations
- CoT prompting as an emergent ability of scale (fundamental innovation)
  - The boost from CoT appears only for sufficiently large models (~100B parameters). Smaller models often produce fluent but incorrect chains, sometimes doing worse than standard prompting (Figure 4; Section 3.2). This connects reasoning performance to model capacity in a way not captured by standard prompting.
- Natural-language steps matter beyond “more tokens” (mechanistic insight)
  - The `variable compute only` control performs like baseline, and `reasoning after answer` offers little gain (Figure 5). Hence benefits come from sequential reasoning content, not just longer outputs or “priming” knowledge.
- Broad, training-free gains across task families (practical innovation)
  - A single, off-the-shelf model checkpoint is prompted to do arithmetic, commonsense, and symbolic reasoning with only a handful of handcrafted exemplars per task (Sections 3–5; Figure 3), achieving strong results without any fine-tuning.
- Robustness to prompt authors and exemplars (practical insight)
  - Different annotators and alternative exemplar sets from an independent data source (GSM8K training set) still yield large improvements over standard prompting (Figure 6; Appendix Tables 6–7), suggesting the phenomenon is not brittle to writing style.
- Length generalization in symbolic tasks (new capability)
  - CoT helps models generalize to longer sequences than seen in exemplars—for example, concatenating last letters for names with 3–4 words when exemplars had only 2 words (Figure 8; Appendix Table 5). Standard prompting fails on these OOD (out-of-domain) settings.

## 5. Experimental Analysis
Evaluation methodology
- Datasets (Sections 3–5; Figure 3; Appendix Table 12)
  - Arithmetic: `GSM8K` (grade-school multi-step math), `SVAMP`, `ASDiv`, `AQuA` (multiple-choice algebra), and `MAWPS` (with subsets: SingleOp, SingleEq, AddSub, MultiArith).
  - Commonsense: `CSQA`, `StrategyQA`, BIG-bench `Date Understanding` and `Sports Understanding`, and robotics `SayCan`.
  - Symbolic: `Last Letter Concatenation` and `Coin Flip` (state tracking), with in-domain and OOD (longer sequence) splits.
- Baselines and metrics
  - Baseline: standard few-shot prompting (no CoT).
  - Metrics: accuracy/solve rate (%). Where applicable, prior supervised state-of-the-art numbers are reported for context (Figure 4; Figure 7; Appendix Table 1).
- Setup details
  - Same eight math exemplars are reused across datasets except AQuA (Section 3.1; Appendix G).
  - Greedy decoding with a single generation; error analyses of outputs are provided (Sections 3.2, A.1; Appendices D.1–D.2).
  - For LaMDA, results are averaged over multiple random orders of exemplars; standard deviations reported in ablations (Appendix Tables 6–7).

Main quantitative results
- Arithmetic (Figure 4; Appendix Table 1 and Table 2)
  - On `GSM8K`, PaLM 540B improves from 17.9% (standard) to 56.9% (CoT) and to 58.6% with the external calculator.
  - `GPT‑3 175B` jumps from 15.6% to 46.9% with CoT; `Codex (code-davinci-002)` reaches 63.1% (65.4% with calculator).
  - On `MAWPS`, `PaLM 540B` improves from 79.2% (standard) to 93.3% (CoT), near the top across subsets (Appendix Table 3).
  - Gains are largest for harder benchmarks; on one-step subsets (MAWPS SingleOp) the improvement is small or negative (Appendix Table 3).
  - Quote: “Chain-of-thought prompting… achieves new state-of-the-art performance on GSM8K” (Figure 2, Figure 4, Appendix Table 1).
- Commonsense (Figure 7; Appendix Table 4)
  - `PaLM 540B` with CoT:
    - `CSQA`: 79.9% (vs 78.1% standard; small gain).
    - `StrategyQA`: 77.8% (vs 68.6% standard), exceeding the prior single-model best 69.4%.
    - `Date Understanding`: 65.3% (vs 49.0% standard).
    - `Sports Understanding`: 95.4% (vs 80.5% standard), beating the “unaided sports enthusiast” human reference of 84%.
    - `SayCan`: 91.7% (vs 80.8% standard).
- Symbolic (Figure 8; Appendix Table 5)
  - In-domain: `PaLM 540B` with CoT nearly solves both tasks—`Last Letter` 99.4% and `Coin Flip` 100%.
  - OOD (longer sequences): `Last Letter` rises from 0–0.2% (standard) to 94.8% (3 words) and 63.0% (4 words) with CoT; `Coin Flip` from ~49–55% (standard) to 98.6% (3 actors) and 90.2% (4 actors) with CoT.
- Ablations and robustness
  - `Equation-only` helps on simpler datasets but not on GSM8K (Figure 5; Appendix Table 6).
  - `Variable compute only` and `reasoning after answer` perform near baseline, indicating natural-language intermediate steps—not just length or knowledge priming—drive gains (Figure 5).
  - Changing annotators, using concise styles, or sampling exemplars from the GSM8K training set still beats standard prompting by large margins (Figure 6; Appendix Tables 6–7).
  - Performance remains better than baseline across different numbers and orders of exemplars (Appendix Figure 11).
- Error analyses and scaling (Appendix A.1; D.1–D.2)
  - On 50 correct GSM8K examples from LaMDA 137B, 49 chains were logically correct; only one arrived at the right answer by chance (Appendix D.1; Table 8–9).
  - On 50 incorrect examples, errors include calculator mistakes (8%), symbol mapping mistakes (16%), missing one step (22%), and deeper semantic/coherence errors (54%) (Appendix D.2; Tables 10–11).
  - Scaling PaLM from 62B to 540B fixes a substantial fraction of “one step missing” and “semantic understanding” errors (Appendix A.1; Figures 9–10).

Assessment of evidence
- The study spans many tasks, models, and controls. The emergence with scale (Figure 4), ablations (Figure 5), and robustness checks (Figure 6; Appendix Tables 6–7, Figure 11) convincingly show that CoT prompting is a distinct, reliable mechanism for eliciting reasoning in large models.
- Where improvements are small (e.g., CSQA) or absent for small models, the paper documents the conditions; gains are strongest on multi-step reasoning and with very large models.

## 6. Limitations and Trade-offs
- Dependence on model scale
  - The approach works reliably only for very large models (~100B+). Smaller models often produce fluent but incorrect chains (Figure 4; Section 3.2). Serving such large models is costly (Section 6).
- No guarantee of correct reasoning
  - Chains can be wrong even when the final answer is correct (more common in multiple-choice/binary tasks) and can be right while arithmetic inside is wrong (fixed partly by the external calculator) (Appendix D.1–D.2; Table 1).
- Prompt sensitivity remains
  - Although robust across annotators and exemplar sets, performance varies by style and order, especially on some classification tasks (Appendix Tables 6–7; Figure 11). Crafting good CoT exemplars still requires care.
- Annotation cost at scale
  - Few-shot prompting keeps costs minimal, but building large training corpora of high-quality chains for finetuning would be expensive (Section 6).
- Scope of evaluation
  - The work focuses on math, commonsense, and symbolic reasoning. Effects on unrelated tasks (e.g., translation, summarization) are left for future study (Appendix A.3).
- Confounding factors
  - Model size co-varies with training compute and data; the analysis in Appendix A.1 suggests scale fixes certain error types, but causal factors beyond parameter count remain open.

## 7. Implications and Future Directions
- How this changes the landscape
  - CoT prompting expands what off‑the‑shelf LLMs can do without training, turning reasoning from a training-time capability into an inference-time behavior. This reframes standard prompting as a lower bound on model ability (Section 6).
- Practical applications
  - Stepwise math solvers and tutoring systems (GSM8K). Multi-hop QA and fact checking with visible reasoning steps (StrategyQA, Date). Task planning for agents/robots (SayCan) where plans are produced in natural language or simple programs (Figure 3; Appendix Table 28).
- Research directions
  - Inducing CoT in smaller models to reduce cost (Section 6).
  - Automated generation/selection of robust CoT prompts; leveraging self-consistency or verifiers to select better reasoning paths (referenced in Section 3.1 and Section 6; Cobbe et al., 2021; Wang et al., 2022a).
  - Tool use integration beyond a simple calculator (retrievers, program interpreters) to reduce arithmetic/knowledge errors (Appendix B, Table 1).
  - Understanding emergent reasoning: analyze which pretraining data, objectives, and architectural choices enable CoT behavior (Appendix A.1).
  - Safety and faithfulness: evaluate and improve factual correctness of reasoning steps, especially in high-stakes domains (Appendix D.2; discussion in Section 6).

Overall, this paper demonstrates a simple but powerful principle: providing a few demonstrations of step-by-step reasoning in the prompt enables sufficiently large LLMs to internalize and reproduce multi-step solution strategies, substantially improving performance on reasoning-heavy tasks without any parameter updates.
