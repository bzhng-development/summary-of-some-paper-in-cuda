# Training Large Language Models to Reason in a Continuous Latent Space

**ArXiv:** [2412.06769](https://arxiv.org/abs/2412.06769)

## 🎯 Pitch

This paper introduces Coconut (Chain of Continuous Thought), a novel paradigm where large language models reason in an unconstrained continuous latent space by recycling their hidden states rather than expressing each step as language tokens. By enabling the model to explore multiple potential reasoning paths in parallel and supporting advanced planning—such as breadth-first search—Coconut surpasses traditional chain-of-thought reasoning in both efficiency and performance on complex tasks. This breakthrough paves the way for more flexible, powerful, and efficient machine reasoning by breaking free from the constraints and verbosity of language-based intermediate steps.

---

## 1. Executive Summary (2–3 sentences)
This paper introduces Coconut (“Chain of Continuous Thought”), a way for large language models (LLMs) to reason in a continuous latent space rather than only through natural-language chains of thought. Instead of emitting a word at each step, Coconut loops the model’s last hidden state back as the next input embedding, enabling end-to-end differentiable “latent thoughts” that can represent multiple possible next steps and support search-like planning.

## 2. Context and Motivation
- The gap addressed
  - Standard LLMs must express intermediate reasoning in words (“chain-of-thought,” CoT). This forces every step to be a discrete token decision, even when:
    - Many tokens are filler for fluency rather than reasoning (Section 1).
    - A few critical tokens embody hard planning decisions and receive the same computation budget as easy tokens (Section 1).
  - Cognitive evidence suggests reasoning need not be linguistic; language is optimized for communication, not thought (Introduction).
- Why it matters
  - Practical: CoT can be verbose and slow, incurs token costs, and sometimes commits too early to a wrong path.
  - Theoretical: CoT improves a transformer’s “effective depth” by looping outputs back as inputs (Related Work). If that loop could be in latent space, we might gain the same or more expressivity without language constraints.
- Where prior work falls short
  - CoT and its variants still operate in language space and inherit its limitations (Sections 1–2).
  - “Pause tokens” or filler tokens add compute but do not extend expressivity like CoT and work best for highly parallel tasks (Related Work; Table 1 shows mixed efficacy).
  - Internalizing CoT (e.g., iCoT) removes explicit reasoning tokens but remains trained and evaluated in language space (Related Work; Baselines).
  - Search-augmented methods often bolt on external tree search or specialized algorithms (Related Work), not an intrinsic latent reasoning loop inside the model.
- Positioning
  - Coconut replaces “output a token” with “recycle the hidden state” for some steps. This creates an unconstrained, differentiable reasoning loop that can encode multiple alternatives in a single latent representation (Sections 3 and 5), enabling search-like behavior to emerge without an explicit search algorithm.

## 3. Technical Approach
Coconut lets an LLM switch between two modes during generation (Figure 1):
- Language mode: standard autoregressive decoding.
- Latent mode: “continuous thoughts.” The last hidden state `h_t` of the current position becomes the input embedding for the next position, so no word is produced.

Key mechanisms
- Notation and core loop (Section 3)
  - Standard LM: For tokens `x = (x1,…,xT)`, embeddings `E_t = [e(x1),…,e(xt)]` pass through the transformer to hidden states `H_t`. The final state `h_t = H_t[t,:]` is mapped to token probabilities via `softmax(W h_t)`.
  - Coconut’s latent step: Instead of mapping `h_t` to a token, directly use `h_t` as the input embedding for step `t+1`. No token is emitted. This creates a chain of hidden states (continuous thoughts).
- Mode control with special tokens (Section 3; Figure 2)
  - `<bot>` (“begin of thought”) marks entry into latent mode.
  - `<eot>` (“end of thought”) marks exit back to language mode.
  - During latent mode, the input sequence is: question tokens …, `<bot>`, `h_i`, `h_{i+1}`, … until `<eot>` is inserted, after which normal embeddings resume.
- Training curriculum (Section 3; Figure 2)
  - Goal: teach the model what useful continuous thoughts should accomplish using existing CoT data as weak supervision.
  - Multi-stage schedule inspired by iCoT:
    1) Stage 0: Train on standard language CoT (question → multi-step reasoning in text → answer).
    2) Later stages: progressively remove the first `k` language reasoning steps and replace each removed step with `c` continuous thoughts. Here `c` is a hyperparameter (“number of latent thoughts per removed language step”).
    3) Loss: standard negative log-likelihood, but only on remaining text (mask the question and latent-thought positions). This encourages latent thoughts not to compress the exact removed text, but to carry whatever information helps predict later steps and the answer.
    4) Optimizer state is reset when switching stages (as in iCoT).
  - Compute: when `n` latent thoughts are used in a sample, training performs `n+1` sequential forward passes to generate them, plus a final pass for the loss (Training Details).
- Inference (Section 3)
  - After the question, insert `<bot>`, run a fixed number of latent steps, then insert `<eot>` and resume language decoding. Two strategies for ending latent mode:
    - Fixed length (used in most experiments).
    - A learned binary classifier that decides when to stop (also viable).
  - A knob `k` controls how many continuous thoughts to use at test time; the same model can be run with different `k` (Section 5.1).
- Why this design?
  - Looping hidden states preserves the “depth-increasing” benefit of CoT (Related Work), but without forcing discrete token commitments mid-reasoning.
  - Latent vectors can superpose multiple futures, delaying hard choices until there is enough evidence (Sections 4.4, 5.2–5.4).
  - The curriculum is necessary: direct training from questions to answers with latent thoughts (“w/o curriculum”) fails (Table 1), showing the model needs staged guidance to learn useful latent representations.

Implementation details for experiments (Section 4.2; Appendix)
- Base model: GPT-2.
- Datasets:
  - GSM8K math word problems (open-domain math; Section 4.1).
  - ProntoQA 5-hop logical deduction with synthetic ontologies (trees; Section 4.1).
  - ProsQA (new): logical deduction over randomly generated DAGs that require more planning/search than ProntoQA (Section 4.1; Appendix A.2–A.3).
- Schedules:
  - GSM8K: `c = 2`; several stages up to all-language-steps removed; final stage still uses `3×c` latent thoughts for long-tail longer CoTs (Section 4.2).
  - ProntoQA/ProsQA: `c = 1`; up to 6 stages to cover max 6 steps; final stage uses only latent thoughts (Section 4.2).
  - Stage mixing for analysis: to prevent forgetting earlier modes, Section 5 uses a mixed-stage curriculum during training (sample other stages with probability 0.3) so inference can vary `k`.

## 4. Key Insights and Innovations
- Feeding hidden states back as inputs (“continuous thoughts”) is a new latent reasoning loop
  - Difference: Prior “pause/filler tokens” add discrete placeholders, while Coconut’s thoughts are continuous vectors that are fully differentiable and not constrained to language (Section 3; Table 1 compares to “Pause token”).
  - Significance: Maintains CoT’s expressivity-boosting loop without discrete token commitments, enabling richer intermediate computation and potentially fewer output tokens (Table 1, Table 4).
- Emergent search-like behavior in latent space
  - Observation: Continuous thoughts can encode multiple candidate next steps and delay commitment, behaving like a soft breadth-first search (BFS) over reasoning options (Section 1; detailed in Section 5).
  - Evidence: When forcing the model to “surface” its latent options (switching to language after k latent steps), the probability mass over possible next nodes acts like an implicit value function over the frontier (Figures 6–7). Parallel exploration is visible in first latent steps and narrows as reasoning proceeds (Figure 8).
- A practical curriculum to learn latent reasoning
  - Novelty: The multi-stage removal-and-replacement schedule turns language CoT into supervision for latent thoughts (Figure 2).
  - Necessity: Without this curriculum, performance collapses (“w/o curriculum” rows in Table 1), indicating latent reasoning does not emerge from question–answer pairs alone.
- Planning benefits on graph-like reasoning
  - ProsQA (DAG reasoning) shows large gains from latent reasoning versus plain CoT (Table 1: 97.0% vs 77.5% accuracy), supporting the claim that latent thoughts help when early steps are high-branching and error-prone (Sections 4.4, 5.2–5.4).

## 5. Experimental Analysis
- Evaluation setup
  - Datasets (Section 4.1; Appendix A)
    - GSM8K (math). Training uses a synthetic CoT dataset from prior work for supervision (Section 4.1).
    - ProntoQA (5-hop tree-shaped ontologies; fictional concepts).
    - ProsQA (new). Graph construction yields DAGs with more distractors and deeper planning; average stats in Appendix Table 2 (23 nodes, 36 edges, shortest-path length ≈3.8).
    - Dataset sizes in Appendix Table 3.
  - Baselines and ablations (Section 4.3)
    - `CoT`: supervised CoT.
    - `No-CoT`: direct answer prediction.
    - `iCoT`: progressively hides early tokens to “internalize” CoT; predicts answers directly.
    - `Pause token`: insert special `<pause>` tokens between question and answer (same count as Coconut’s latent thoughts).
    - Coconut variants: “w/o curriculum,” “w/o thought” (same schedule but removing language steps without adding latent ones), and “pause as thought” (replace continuous thoughts with `<pause>` but keep Coconut’s curriculum).
  - Metrics
    - Accuracy (final answer correctness).
    - Efficiency: number of newly generated tokens (Table 1) and wall-clock time per instance (Appendix Table 4).
    - Reasoning-quality categories for ProsQA (Section 5.1): “Correct Path,” “Longer Path,” “Hallucination,” “Wrong Target,” and answer-only cases “Correct Label/Incorrect Label.”
- Main quantitative results
  - GSM8K (Table 1)
    - CoT: 42.9% with 25.0 tokens.
    - Coconut: 34.1% with 8.2 tokens.
    - iCoT: 30.0% (fewer tokens, 2.2).
    - No-CoT: 16.5%; Pause: 16.4%.
    - Takeaway: Coconut outperforms No-CoT and iCoT on this small model, but trails CoT. It achieves substantial token savings vs CoT (≈3× fewer).
    - Ablation: Increasing thoughts per removed step helps—Figure 3 shows accuracy improving as `c` goes from 0→1→2.
    - Interpreting latent states: Figure 4 decodes the first latent thought into tokens like “180” and “9” for a math problem, suggesting it captured intermediate quantities used later.
  - ProntoQA (Table 1)
    - All strong: CoT 98.8%, iCoT 99.8%, Coconut 99.8%. “Pause token” is unstable (77.7% ± 21.0), and “w/o curriculum” fails (52.4%).
    - Takeaway: The task is relatively easy; expressivity/compute is not the bottleneck, but the curriculum is still necessary for latent thoughts.
  - ProsQA (Table 1)
    - Coconut: 97.0% with 14.2 tokens vs CoT: 77.5% with 49.4 tokens (≈3.5× fewer tokens).
    - iCoT reaches 98.2% (8.2 tokens), and Coconut variants “w/o thought” (95.5%) and “pause as thought” (96.6%) also perform well.
    - Takeaway: Planning-intensive DAG reasoning favors latent/internalized approaches; Coconut is highly effective and efficient here.
  - Efficiency: Appendix Table 4
    - Wall-clock time roughly tracks generated tokens. Example: on ProsQA, CoT 0.47 s vs Coconut 0.15 s per instance; on GSM8K, CoT 0.26 s vs Coconut 0.09 s.
- Do experiments support claims?
  - Planning advantage: Yes. On ProsQA’s harder planning setting, Coconut dramatically outperforms CoT and matches/exceeds internalization baselines (Table 1). Error analyses show fewer “Hallucination” and “Wrong Target” outcomes as more latent steps are used (Figure 5).
  - Chaining in latent space: Yes. Accuracy rises with more latent thoughts (`c` on GSM8K; Figure 3), echoing the “depth” effect of CoT but in continuous space.
  - Emergent search: Supported qualitatively and quantitatively by:
    - Case study (Figure 6): CoT hallucinates an edge when it gets stuck; Coconut with `k=2` avoids early commitment and succeeds.
    - Value-function view (Figures 7–8): probability mass over frontier nodes spreads across multiple options in early latent steps (parallel exploration) and concentrates later (focused exploitation).
    - Height correlation (Figure 9): nodes with low “height” (shortest distance to a leaf) are easier to evaluate; latent steps move reasoning closer to terminal states, improving discrimination.
- Additional analyses
  - Interpolating latent vs language reasoning (Figure 5): With the same trained model, increasing `k` (more latent steps before switching back to language) steadily improves final-answer accuracy and reduces hallucination rates on ProsQA.
  - Training stability with more thoughts: Appendix C reports a mild drop and variance spike at `c=3`, tracing to loss spikes at stage transitions; suggests using finer-grained schedules.

## 6. Limitations and Trade-offs
- Reliance on supervised CoT and a curriculum
  - “W/o curriculum” performs poorly across tasks (Table 1). Learning useful latent thoughts directly from question–answer pairs did not succeed in this study; the method currently depends on having CoT-like supervision to bootstrap (Figure 2; Section 4.4).
  - Even with curriculum, training complexity rises: for `n` latent thoughts, training requires `n+1` sequential forward passes (Training Details). KV caching helps but the sequential nature limits parallelism.
- Fixed-length latent segments at inference
  - Most experiments use a fixed number of latent steps; while a stop-classifier is mentioned and works, it is not the default, leaving room for dynamic control research (Inference Process).
- Model and data scope
  - Experiments use GPT-2; scaling behavior on larger models remains to be demonstrated.
  - Only one natural dataset (GSM8K). The strongest gains are on synthetic logical tasks; generalization to diverse real-world domains needs validation.
- Stability and scheduling sensitivity
  - Abruptly increasing latent-thought count (e.g., `c=3`) can destabilize training (Appendix C). Curriculum design (stage granularity, mixing) materially affects results (Sections 4.2, 5.1).
- Interpretability caveat
  - The “implicit value function” is inferred via token probabilities when forced back to language (Figures 7–8). It is a useful probe but not a ground-truth readout of latent content.

## 7. Implications and Future Directions
- Conceptual shift: Reasoning beyond language
  - Coconut demonstrates that an LLM can perform multi-step reasoning entirely in its continuous state space, preserving CoT’s depth benefits while avoiding premature discrete commitments. This opens a path to internal, search-like planning within the model’s forward passes.
- Practical implications
  - For planning-heavy tasks (e.g., theorem proving, program synthesis, multi-hop retrieval), latent reasoning can reduce tokens and time while improving robustness to early mistakes (Table 1; Table 4; Figures 5–6).
  - Deployments concerned with cost and latency may benefit from shorter generations without sacrificing accuracy—especially on structured reasoning problems.
- Research directions
  - Pretraining with latent thoughts: The paper suggests pretraining LMs to use continuous thoughts (Conclusion), potentially improving zero/few-shot planning.
  - Better curricula and stability: Finer-grained token/step removal, dynamic schedules, and mixing strategies to avoid forgetting (Section 5.1; Appendix C).
  - Adaptive control: Learn when to switch modes and how many latent steps to spend (budgeted latent reasoning).
  - Hybrid reasoning: Combine a sparse “language skeleton” with latent completion, or integrate explicit search policies with latent rollouts.
  - Scaling studies: Validate on larger models and broader, real-world tasks; examine interactions with multi-token prediction pretraining (Section 5.2 discussion) and tool use.
  - Interpretability/probing: Develop more direct techniques to read or steer latent searches, beyond projecting back to language (Related Work; Section 5.3).

> Most decisive evidence comes from Table 1 and Figure 5: Coconut matches or beats language CoT on logical tasks while generating far fewer tokens, and its accuracy improves as more reasoning is done in latent space, with reductions in hallucinations and wrong-target paths. Figures 6–9 further unpack how continuous thoughts support parallel exploration early and focused decisions later, yielding a soft BFS-like process inside the model.
