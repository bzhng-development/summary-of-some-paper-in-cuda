# Stay on topic with Classifier-Free Guidance

**ArXiv:** [2306.17806](https://arxiv.org/abs/2306.17806)

## 🎯 Pitch

This paper introduces the use of Classifier-Free Guidance (CFG)—an inference-time technique popular in diffusion image generation—to autoregressive language models, allowing them to adhere more closely to user prompts without additional training or model modifications. By simply reweighting model predictions with and without the prompt, CFG significantly boosts fidelity on tasks like Q&A, code generation, and assistant instructions—often matching models twice their size in accuracy and compute. This lightweight, training-free method empowers users to control generative outputs more precisely, enabling smaller, cheaper models to achieve state-of-the-art prompt alignment and utility.

---

## 1. Executive Summary (2–3 sentences)
This paper adapts Classifier‑Free Guidance (CFG)—a simple inference-time trick widely used in diffusion image models—to autoregressive language models to make generations stick more closely to the prompt. With no extra training, the method consistently improves zero‑shot benchmarks, long‑form/code generation, and assistant prompts; notably, it pushes LLaMA‑7B to state‑of‑the‑art accuracy on LAMBADA and often matches the performance of models roughly twice the size at comparable inference compute (Sections 3–4; Figure 2; Figure 11).

## 2. Context and Motivation
- Problem addressed:
  - Large language models (LLMs) often drift from the user’s prompt, leading to hallucinations, meandering, or ignoring system instructions (Introduction; Section 1; Table 1).
- Why it matters:
  - Better prompt adherence increases faithfulness, usefulness, and safety of generated text in assistants, long‑form tasks, and code generation.
- Shortcomings of prior approaches:
  - Training‑time fixes (instruction tuning, RLHF) improve adherence but are expensive and not always accessible (Introduction).
  - Decoding heuristics (temperature, nucleus sampling) change diversity but offer weak control over prompt adherence (Section 5).
  - Prior “controlled generation” methods (e.g., PPLM/FUDGE) require extra classifiers or model modifications (Related Work; Appendix B.2).
- Positioning:
  - The paper provides an inference‑only, training‑free method that reweights token probabilities using two passes of the same model—one “with the prompt” and one “without”—to emphasize prompt‑consistent continuations (Sections 2.1–2.2; Equation 7).

## 3. Technical Approach
Classifier‑Free Guidance in diffusion (background)
- In diffusion models, “classifier guidance” adjusts samples using a classifier’s gradient toward a desired condition `c` (Equation 1).
- CFG removes the external classifier by training the same generative model to handle both conditioned and unconditioned inputs and then combining them: `Pc(x|c) ∝ Pθ(x|c)^γ / Pθ(x)^(γ−1)` (Equation 2), where `γ` (“guidance strength”) ≥ 0 amplifies the influence of condition `c`.
- An intuitive view: move away from an “unconditional” representation toward the “conditional” one by a step of size `γ` (Equation 4; vector arithmetic view in Section 2.1).

How CFG is adapted to language models (Section 2.2)
- Key idea: without any retraining, LMs can compute both:
  - a conditional next-token distribution `Pθ(wi | w< i, c)` (normal prompting), and
  - an “unconditional” distribution `Pθ(wi | w< i)` by dropping the prompt `c` from the context window (LMs naturally support this due to finite contexts).
- Combine them at each decoding step in logit space (logits = the unnormalized scores before softmax):
  - `log Pcθ(wi|w< i, c) = log Pθ(wi|w< i) + γ [log Pθ(wi|w< i, c) − log Pθ(wi|w< i)]` (Equation 7).
  - This raises probabilities of tokens favored by the prompted distribution and lowers those favored only by the unprompted one.
- Negative prompting (Section 2.1; Equation 5):
  - Users can also specify an undesired condition `c−` (e.g., a default system prompt). Then move from `c−` toward `c`: 
    - `log Pcθ(wi|w< i, c, c−) = log Pθ(wi|w< i, c−) + γ [log Pθ(wi|w< i, c) − log Pθ(wi|w< i, c−)]`.
  - This highlights how CFG can “emphasize the difference” between two prompts.

Design choices and rationale
- Operate in logits space (Section 2.2):
  - Logits are linearly related to the last hidden layer and easy to manipulate without architecture‑specific surgery; they directly control token probabilities.
- No new training:
  - Unlike diffusion models that need conditioning dropout to learn unconditional paths, decoder LMs can already compute `P(wi|w< i)` by truncating the prompt (Section 2.2).
- Implementation detail in evaluations:
  - For zero‑shot benchmarks they “start the unconditional prompt at the last token of the initial prompt,” i.e., approximate the unconditioned branch by dropping the prefix `c` to the final token to ensure a comparable decoding state (Section 3.1).

Simple analogy
- Think of two advisors at each step:
  - “Prompted advisor” suggests tokens that fit the prompt,
  - “Unprompted advisor” suggests common/average continuations.
  - CFG tells the decoder to favor the prompted advisor by factor `γ` and discount the unprompted one.

## 4. Key Insights and Innovations
- Prompt adherence via out‑of‑the‑box CFG for LMs (fundamental):
  - The paper shows CFG works in autoregressive text generation without any retraining (Section 2.2), unlike diffusion models that need special training. This is a conceptual bridge between diffusion guidance and LM decoding.
- Negative prompting for assistants (new capability):
  - By setting the “negative condition” to a model’s default system prompt and the “positive condition” to a modified system prompt, CFG emphasizes system‑instruction compliance while preserving user relevance (Section 3.4; Figure 5).
- “Small model + CFG ≈ twice‑sized model” at similar inference compute (practical innovation):
  - Across five of nine benchmarks there is no statistically significant difference between “CFG on a smaller model” and “vanilla decoding on a model twice as large” (ANCOVA at p=.01; Section 4; Figure 11; Table 4). This offers a cost/latency alternative when VRAM is constrained.
- Mechanistic explanation of why it works (analysis contribution):
  - CFG reduces sampling entropy (narrows the plausible token set) to a level similar to instruction‑tuned models, while reordering high‑probability tokens toward prompt‑relevant ones (Section 5; Figure 6a–b, Table 3). It is not equivalent to instruction tuning; overlaps are task‑dependent (Figure 7; Table 8).

## 5. Experimental Analysis
Evaluation setup
- Benchmarks (Section 3):
  - Zero‑shot tasks via the LM Harness: ARC‑c/e, BoolQ, HellaSwag, PIQA, SCIQ, TriviaQA, WinoGrande, and LAMBADA (Figures 2, 8–10).
  - Chain‑of‑Thought (CoT) reasoning: GSM8K and AQuA with few‑shot prompts; models: WizardLM‑30B and Guanaco‑65B (Section 3.2; Figure 3; Figure 15).
  - Code generation: HumanEval with CodeGen‑350M/2B/6B at temperatures 0.2/0.6/0.8; metrics: pass@1/10/100 (Section 3.3; Table 2; Tables 5–7; Figures 12–14).
  - Assistant/system‑prompt control with negative prompting: GPT4All‑J v1.3‑jazzy, 1740 system/user prompt combinations; human preference study evaluating “follows system prompt” vs “follows user prompt” (Section 3.4; Figure 5; Appendix G).
  - Additional: Machine translation on WMT14 fr‑en with Bloom‑3B, RedPajama‑3B, and mT0 (Appendix D.1), and targeted code prompts with GPT‑J/CodeGen (Appendix D.2).
- Baselines:
  - “γ = 1” is vanilla decoding; CFG uses γ > 1. For negative prompting, `c−` is the model’s default system prompt and `c` is the edited system prompt (Section 3.4).

Main quantitative results
- Zero‑shot improvements are broad, with exceptions:
  - LLaMA‑7B on LAMBADA (zero‑shot) improves from 73.6% to 81.3% with γ=1.5, surpassing PaLM‑540B’s reported 77.9% (Figure 2b).
  - Many tasks show nontrivial, consistent gains at γ≈1.5 across GPT‑2, Pythia, and LLaMA families (Figures 8–10). ARC‑challenge and WinoGrande are outliers where gains are small or negative (Section 3.1; Figure 2a–b).
- Chain‑of‑Thought (Section 3.2):
  - CFG increases the rate of “validly formatted” answers and boosts accuracy for small γ; too large γ reduces accuracy despite staying valid (Figure 3; Figure 15). 
  - Example (Table 15) shows a GSM8K prompt where γ=1.1 yields the correct chain and answer, while γ=1 (vanilla) diverges and formats incorrectly.
- Code generation (HumanEval; Section 3.3; Table 2):
  - At temperature 0.2, pass@1 rises at small γ and deteriorates at large γ.
    - `CodeGen‑2B`: pass@1 from 19.5% (γ=1.0) → 20.9% (γ=1.5), then down to 16.5% (γ=2.0) (Table 2).
    - `CodeGen‑350M`: pass@1 ~11.0% (γ=1.0), best around γ=1.1 (11.8%), then declines (Table 2).
  - Gains shrink for pass@100 because CFG reduces diversity; tasks with tiny but non‑zero pass rates can benefit less from multiple samples (Section 3.3.2).
  - Targeted code prompts show practical quality gains: e.g., making a 32×32 red image array, “correct return type” rises from 289 to 546 out of 1600 with γ=2 (Table 13).
- Assistant/system‑prompt adherence via negative prompting (Section 3.4):
  - Human study (611 judgments, 71 raters) shows a clear peak at γ=3:
    - “Follows system prompt” wins 75% of pairwise comparisons against γ=1 (no CFG), while “follows user prompt” remains statistically unchanged up to γ≈3 and only degrades for γ≥4 (Figure 5; Table 16 qualitative example).
- Cost vs. model size (Section 4):
  - Although CFG doubles forward passes, the compute‑accuracy trade‑off is competitive with simply using a model twice as large:
    - “Across 5/9 tasks there is no significant difference” (ANCOVA p>.01), with two tasks favoring CFG and two favoring vanilla (Figure 11; Table 4).
- Why it works (Section 5):
  - Entropy reduction: average logit entropy drops from ~5.4 (vanilla) to ~4.7 (CFG‑γ=1.5) (Figure 6a), shrinking the token set needed to cover top‑p=0.9 probability mass (Figure 6b).
  - Not equivalent to instruction tuning: perplexities and top‑p overlaps differ, with higher agreement on longer prompts and certain datasets (Figure 7; Table 8).
  - Visualization confirms token reordering toward prompt‑relevant content (Table 3 for “The dragon flew over Paris, France”).

Do the experiments support the claims?
- Yes, for “CFG improves prompt adherence and often accuracy”: multiple families, datasets, and tasks show reliable gains at modest γ with clear counter‑examples analyzed (ARC‑c and WinoGrande). The assistant study shows human‑perceived gains and the analysis section provides a plausible mechanism (entropy and token reordering).
- Mixed/conditional findings are acknowledged:
  - Gains are γ‑dependent; too much guidance harms diversity and sometimes accuracy (Section 3.2–3.3; Figures 12–14).
  - Instruction‑tuned models and CFG are complementary rather than equivalent (Section 5.2; Figure 7).

## 6. Limitations and Trade-offs
- Tuning required:
  - Optimal `γ` varies by task, model size, and temperature (Figures 8–14). Over‑guidance (large γ) reduces diversity and can harm reasoning/code correctness even while preserving answer format (Figure 3; Table 2).
- Not universally beneficial:
  - Some benchmarks (e.g., ARC‑c, WinoGrande) show little or negative gains (Figure 2a; Figures 8–10).
- Compute/latency:
  - CFG roughly doubles inference FLOPs (two forward passes per token), though VRAM does not double (Section 4). This may be unacceptable in latency‑sensitive settings.
- Safety and robustness:
  - Increasing adherence may also strengthen malicious or injected instructions; the paper explicitly has not stress‑tested CFG against prompt injection or alignment bypass (Conclusion).
- Methodological nuance:
  - The “unconditional” branch is approximated by dropping the prefix; when prompts carry essential state (e.g., long contexts), this approximation might misestimate `P(wi|w< i)` (Section 3.1 implementation note).
- Diversity vs. fidelity:
  - Entropy reduction and narrowed top‑p set (Figure 6) mean fewer diverse continuations—beneficial for adherence but a drawback for creative writing or exploration (Section 5.1).

## 7. Implications and Future Directions
- Field impact:
  - Establishes CFG as a simple, model‑agnostic inference control for LMs, analogous to its role in diffusion models. It offers a practical knob to trade off diversity for prompt faithfulness without training (Sections 2–3, 5).
- Research avenues:
  - Targeted guidance: weight different parts of multi‑stage prompts (system vs. user, or separate CoT vs. final answer) differently; authors note promising early results and many unexplored variants (Section 3.2 discussion).
  - Robustness: benchmark CFG under prompt injection and safety edge cases; standardize risk‑focused evals (Conclusion).
  - Integration with other inference techniques: combine CFG with self‑consistency, reranking, or contrastive decoding (Section 1 claims and Related Work).
  - Learning to set γ: develop adaptive or token‑wise guidance strengths via uncertainty or entropy signals (Section 5.1 suggests entropy as a proxy).
- Applications:
  - Assistant systems: enhance compliance with system policies without retraining; negative prompting lets teams quickly “steer away” from undesired personas (Section 3.4; Figure 5).
  - Long‑form and structured generation: improved staying‑on‑topic for essays, summarization with constraints, and document‑style outputs (Table 1 demonstration).
  - Code synthesis and formal tasks: modest but meaningful pass@1 boosts for smaller models, helpful where latency/VRAM are constrained (Table 2; Tables 5–7).
  - Machine translation and programmatic prompting: early evidence of gains for small γ with multilingual and base models (Appendix D.1–D.2; Table 11; Table 12; Table 13).

> Representative results:
> - “LLaMA‑7B LAMBADA zero‑shot: 73.6 → 81.3 (γ=1.5)” (Figure 2b).
> - “Human eval for assistants: 75% preference for CFG at γ=3; user prompt relevance unchanged until γ≥4” (Figure 5).
> - “CodeGen‑2B HumanEval pass@1: 19.5% → 20.9% (γ=1.5); declines at γ=2.0” (Table 2).
> - “Entropy drops from ~5.4 to ~4.7 with CFG‑γ=1.5” (Figure 6a).
> - “Across 5/9 tasks, a smaller model+CFG is statistically indistinguishable from a 2× larger model without CFG” (Figure 11; Table 4).

Definitions (used selectively)
- `Classifier‑Free Guidance (CFG)`: An inference‑time technique that combines conditional and unconditional model outputs to upweight tokens consistent with a condition `c`, controlled by `γ` (Equations 2, 7).
- `Guidance strength (γ)`: A non‑negative scalar; `γ=1` is vanilla; `γ>1` emphasizes the prompt; too large harms diversity/accuracy.
- `Negative prompting`: Specify an undesired condition `c−` and move away from it toward `c` (Equation 5).
- `Logits`: Pre‑softmax scores for each token; adjusting logits changes the probability distribution.
- `Entropy` (of logits distribution): A measure of uncertainty; lower entropy implies a sharper distribution with fewer plausible tokens.
- `Pass@k`: For code benchmarks, fraction of problems solved if up to k samples per problem are allowed.
- `FLOPs`: Floating‑point operations; a proxy for compute cost during inference.
