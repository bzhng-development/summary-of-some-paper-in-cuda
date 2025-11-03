# Training Large Language Models to Reason in a Continuous Latent Space

**ArXiv:** [2412.06769](https://arxiv.org/abs/2412.06769)
**Authors:** Shibo Hao, Sainbayar Sukhbaatar, DiJia Su, Xian Li, Zhiting Hu, Jason Weston, Yuandong Tian
**Institutions:** 

## 🎯 Pitch

Coconut revolutionizes multi-step reasoning by enabling large language models to operate in continuous latent space instead of generating each step in natural language. This approach enhances parallel planning and search efficiency, reducing token use while significantly improving problem-solving accuracy, making it ideal for complex reasoning tasks like theorem proving and logic puzzles.

---

## 1. Executive Summary (2–3 sentences)
This paper introduces Coconut (“Chain of Continuous Thought”), a way for large language models (LLMs) to carry out multi‑step reasoning in a continuous latent space instead of generating intermediate steps in natural language. By feeding a model’s last hidden state directly back as the next input embedding (rather than decoding a token), Coconut enables more parallel and reversible planning, yields breadth‑first‑search‑like behavior, and improves accuracy and efficiency on several reasoning tasks (Figures 1–2, Sections 3–5).

## 2. Context and Motivation
- Problem addressed
  - Standard LLMs reason by predicting the next token, so intermediate reasoning is expressed in natural language (“chain‑of‑thought,” CoT). This imposes two constraints:
    - The model must commit to one next token at each step, even when multiple branches should be explored or pruned later.
    - Compute per token is fixed even though many tokens are “for fluency” rather than reasoning, while a few tokens demand complex planning (Section 1).
- Why it matters
  - Practical: Many reasoning problems (e.g., logic, theorem proving, math word problems) benefit from planning, search, and backtracking; token-by-token CoT often over‑commits early and wastes tokens on text rather than computation (Sections 1, 2).
  - Scientific: Cognitive evidence indicates language areas are not central to core reasoning, suggesting the “language channel” is not the ideal substrate for computation (Section 1, citing Amalric & Dehaene 2019; Monti et al. 2012).
- Prior approaches and their gaps
  - CoT improves expressivity by looping outputs back as inputs, effectively deepening the transformer (Related Work; Feng et al., 2023; Merrill & Sabharwal, 2023), but remains discrete and autoregressive in the language space.
  - Concise CoT, filler or pause tokens, and “think‑before‑speaking” methods provide extra “compute tokens” but still operate in the language bottleneck and don’t extend expressivity like CoT does (Related Work; Goyal et al., 2023; Pfau et al., 2024).
  - Explicit search or tool‑use methods add external algorithms or scaffolds (Related Work; Yao et al., 2023), rather than giving the base model an internal planning substrate.
- Positioning
  - Coconut removes the language bottleneck during internal reasoning by looping hidden states, not tokens. It keeps the simplicity and end‑to‑end training of vanilla LLMs but changes the medium of thought from words to vectors (Figures 1–2, Section 3). It also supplies a curriculum to learn these latent “thoughts,” connecting to recent iCoT (“internalized CoT”) training (Deng et al., 2024).

## 3. Technical Approach
Coconut adds a “latent mode” to a standard LLM’s generation loop.

- Base LLM notation (Section 3)
  - Input sequence `x = (x1, …, xT)` with embeddings `Et = [e(x1), …, e(xt)]`.
  - Transformer produces last‑layer hidden states `Ht ∈ R^{t×d}`; the final state at position `t` is `ht = Ht[t, :]`.
  - Usual next‑token distribution: `softmax(W ht)`.

- Two modes of reasoning (Figure 1; Section 3)
  - Language mode: standard autoregressive decoding where the model predicts a token and its embedding becomes the next input.
  - Latent mode: the model does not predict a token. Instead, it feeds its last hidden state back as the next input embedding. The hidden state at each step is treated as a “continuous thought,” i.e., a vector‑space representation of the reasoning state.

- Switching between modes (Section 3)
  - Two special tokens are introduced:
    - `<bot>` (“begin of thoughts”) marks where latent mode starts.
    - `<eot>` (“end of thoughts”) marks where latent mode ends.
  - If latent reasoning spans positions `i … j`, with `xi = <bot>` and `xj = <eot>`:
    - For `i < t < j` (latent steps), the input sequence is `Et = [e(x1), …, e(xi), hi, hi+1, …, ht−1]`.
    - After `<eot>`, the input reverts to token embeddings: `Et = [ … , hj−1, e(xj), …, e(xt)]`.
  - During latent mode, `M(xt+1 | x≤t)` is not defined (no token is produced), though `softmax(W ht)` can still be probed to analyze what token the hidden state “leans toward” (Section 3; used later in Section 5).

- Training procedure (Figure 2; Section 3)
  - Setting: question‑answer tasks where intermediate CoT is available for supervision.
  - Multi‑stage curriculum (inspired by iCoT, Deng et al., 2024):
    - Stage 0: train on regular CoT (question, reasoning steps in language, answer).
    - Stage k (k ≥ 1): replace the first k language reasoning steps with `k × c` continuous thoughts, where `c` (a hyperparameter) is the number of latent thoughts replacing one language step. Insert `<bot>`/`<eot>` around the latent block.
    - Loss: standard negative log-likelihood on the remaining language tokens; mask loss on question tokens and on latent thoughts. This encourages latent thoughts to be useful for predicting future steps, not merely compressing the removed language sentence (Section 3).
    - Optimizer state is reset between stages (as in Deng et al., 2024).
  - Backpropagation through latent thoughts:
    - With `n` latent thoughts in the current stage, training performs `n+1` forward passes: one to produce each latent thought in sequence, and a final pass to compute loss on the remaining text (Section 3, “Training Details”). KV caching can save repeated compute, but the sequential dependence limits parallelism.

- Inference (Section 3)
  - Insert `<bot>` after the question to enter latent mode.
  - When to stop: either
    - train a small classifier to predict when to emit `<eot>`, or
    - fix the latent length to a constant. The experiments use a fixed length for simplicity unless stated otherwise.
  - After `<eot>`, the model continues in language mode to produce explanations and/or the final answer.

- Experimental configuration (Section 4.2)
  - Base model: GPT‑2 (Radford et al., 2019).
  - Optimizer details: learning rate 1e‑4; effective batch size 128; optimizer reset at stage switches.
  - Task‑specific schedules:
    - GSM8K: default `c = 2`, 3 + 1 stages past the initial stage; 6 epochs at stage 0, 3 epochs thereafter.
    - ProntoQA and ProsQA: `c = 1`; 6 stages (maximum of 6 reasoning steps); 5 epochs per stage; model continues training at the last stage until epoch 50 for model selection.

- How Coconut enables “parallel planning”
  - A continuous thought (a single hidden vector) can implicitly encode multiple candidate next steps. By chaining several latent thoughts before emitting language, the model can explore several branches while delaying commitment. Section 5 shows how to probe these branches by forcing a switch back to language after `k` latent thoughts and reading the token probabilities for the next symbolic step.

- Simplified example (Figure 4; Section 4.4)
  - In a math problem requiring “3 × 3 × 60 = 540,” the first continuous thought, when probed with the language head, assigns high probability to tokens like “180” and “9,” indicating it has internally represented alternative partial computations before committing to a single written step.

## 4. Key Insights and Innovations
- Latent “chain of thought” without language tokens (Figures 1–2; Section 3)
  - Novelty: Rather than generating a word and re‑embedding it, Coconut loops the last hidden state itself as the next input. This frees reasoning from the discrete, single‑choice bottleneck.
  - Significance: It preserves the expressivity gains of looping (as with CoT) while avoiding premature commitment to one tokenized step. It also remains fully differentiable end‑to‑end.

- Curriculum to learn useful latent thoughts (Figure 2; Section 3; Table 1 ablations)
  - Novelty: A staged schedule replaces early language steps with latent thoughts and trains toward predicting later steps/answers. This avoids the failure case where the model is asked to invent latent reasoning from scratch.
  - Evidence: Removing the curriculum (“Coconut w/o curriculum”) collapses performance across tasks (e.g., ProsQA: 76.1% vs 97.0% with curriculum; ProntoQA: 52.4% vs 99.8%; Table 1).

- Emergent breadth‑first‑search‑like planning (Sections 5.2–5.4; Figures 6–9)
  - Novelty: When several latent thoughts are chained before producing text, the model maintains multiple candidate branches and prunes them progressively. This is diagnosed by probing token probabilities after `k` latent steps and interpreting them as “values” for frontier nodes (Figure 7).
  - Significance: This behavior reduces early errors such as hallucinating edges or following wrong branches in graph‑structured reasoning tasks (ProsQA/ProntoQA).

- Efficiency in tokens and time with competitive accuracy (Tables 1, 4)
  - Novelty: Latent thoughts cost no output tokens. Coconut often uses far fewer generated tokens and achieves faster wall‑clock time than long CoT while maintaining or improving accuracy on logic tasks.
  - Evidence: On ProntoQA, Coconut reaches 99.8% ± 0.2% with 9.0 tokens vs CoT 98.8% ± 0.8% with 92.5 tokens; clock time 0.11s vs 0.85s (Tables 1 and 4).

## 5. Experimental Analysis
- Tasks and datasets (Section 4.1; Appendix A; Table 3)
  - GSM8K: grade‑school math word problems; test set 1,319. Training data are synthetic CoT generated by Deng et al. (2023).
  - ProntoQA: 5‑hop logical reasoning over synthetic ontologies with fictional concept names; test set 800.
  - ProsQA (proposed in this paper): new logical reasoning dataset built from randomly generated DAGs to enforce deeper planning and search (Appendix A.2). Graph statistics (Appendix A.2, Table 2): average 23 nodes, 36 edges, shortest‑path length ≈ 3.8, ≈ 1.6 shortest paths per problem. Test set 500.

- Baselines and Coconut variants (Section 4.3; Table 1)
  - Baselines:
    - `CoT`: supervised on full language chains.
    - `No‑CoT`: train to output only the answer.
    - `iCoT` (Deng et al., 2024): gradually remove early CoT tokens during training to internalize reasoning; direct answer at inference.
    - `Pause token`: insert special `<pause>` tokens between question and answer to provide extra “compute tokens.”
  - Variants:
    - `Coconut w/o curriculum`: train only on the final‑stage data (questions and answers with full latent thoughts) — no staged replacement.
    - `Coconut w/o thought`: same curriculum schedule but remove language steps without adding latent thoughts (closer to iCoT but matching Coconut’s schedule).
    - `Coconut pause as thought`: use `<pause>` tokens in place of latent thoughts but keep the Coconut curriculum.

- Main quantitative results (Table 1)
  - GSM8K (math, open‑domain)
    - Quote:
      > Coconut: 34.1% ± 1.5 with 8.2 tokens; CoT: 42.9% ± 0.2 with 25.0 tokens; iCoT: 30.0% with 2.2 tokens.
    - Interpretation: CoT still leads in raw accuracy on GSM8K for this small base model, but Coconut beats iCoT and pause‑token baselines; it achieves meaningful gains over `No‑CoT` (16.5%). Section 4.4 and Figure 3 show that increasing the number of latent thoughts per removed step (`c` from 0 → 1 → 2) steadily improves performance, demonstrating a “chaining” effect in latent space similar to CoT’s depth effect. Appendix C notes instability when jumping to `c = 3`.
  - ProntoQA (logic with moderate branching)
    - Quote:
      > Coconut: 99.8% ± 0.2 with 9.0 tokens vs CoT: 98.8% ± 0.8 with 92.5 tokens.  
      > iCoT: 99.8% ± 0.3 with 3.0 tokens; No‑CoT: 93.8% ± 0.7 with 3.0 tokens.
    - Interpretation: Latent reasoning matches or slightly improves accuracy while reducing tokens by ~10× vs CoT. The `pause token` baseline is unstable (77.7% ± 21.0).
  - ProsQA (logic with substantial planning over DAGs)
    - Quote:
      > Coconut: 97.0% ± 0.3 with 14.2 tokens; iCoT: 98.2% ± 0.3 with 8.2 tokens; CoT: 77.5% ± 1.9 with 49.4 tokens.
    - Interpretation: Language‑based CoT offers little benefit over No‑CoT here (77.5% vs 76.7%), indicating that simply verbalizing steps doesn’t help when the task requires deeper planning/backtracking. In contrast, latent/internalized methods (Coconut and iCoT) achieve large gains, consistent with the planning‑first hypothesis.

- Efficiency (Table 4)
  - Quote:
    > Average per‑case inference time (batch size 1, A100): ProntoQA — Coconut 0.11s vs CoT 0.85s; ProsQA — Coconut 0.15s vs CoT 0.47s; GSM8K — Coconut 0.09s vs CoT 0.26s.
  - Interpretation: Wall‑clock time tracks the counts of generated tokens; latent thoughts are “free” in token budget and faster to execute than generating long verbal chains.

- Analyses that probe “how” Coconut works (Section 5)
  - Interpolating between latent and language reasoning (Figure 5, Section 5.2)
    - Method: Force Coconut to use exactly `k ∈ {0..6}` latent thoughts before switching to language. Use a fine‑grained evaluation that categorizes reasoning traces as Correct Path, Longer Path, Wrong Target, Hallucination, or as Correct/Incorrect Label when no path is produced.
    - Finding: As `k` increases (more latent reasoning), final answer accuracy increases and the proportions of “Hallucination” and “Wrong Target” shrink. This shows fewer early mistakes and better planning.
    - Additional observation: Even at `k = 0` (i.e., Coconut forced to write a full CoT), the final answers and paths are more accurate and less hallucinatory than CoT trained purely on language. The mixed‑stage training appears to teach the model to plan ahead (Section 5.2).
  - Diagnosing latent breadth‑first search (Figures 6–8, Section 5.3)
    - Case study (Figure 6): CoT gets stuck and hallucinates an edge (“Every yumpus is a rempus”), Coconut with `k=1` picks a wrong branch (“Wrong Target”), Coconut with `k=2` succeeds. The extra latent step allows the model to keep alternatives alive and prune them later.
    - Value probing (Figure 7): After `k=1`, the model’s probabilities over the next concept (children of the root “Alex”) are roughly 0.33 for “lempus,” 0.32 for “grimpus,” 0.16 for “zhorpus,” and 0.01 for “sterpus,” showing early pruning of “sterpus.” After `k=2`, mass concentrates on “rorpus,” consistent with narrowing the search frontier.
    - Parallelism analysis (Figure 8): The gap between cumulative values of top‑1, top‑2, and top‑3 candidates is larger at the first thought (broad exploration) and narrows at the second thought (focused exploitation).
  - Why latent helps planning (Figure 9, Section 5.4)
    - Define node “height” as shortest distance to a leaf. Lower height means fewer remaining steps and easier evaluation.
    - Finding: Prediction probabilities correlate with height—incorrect nodes are suppressed more reliably when their height is low. Latent reasoning pushes decisions closer to terminal states, where evaluation is easier and less error‑prone.

- Ablations and variants (Table 1; Section 4.4)
  - Curriculum is essential: without it, performance degrades drastically.
  - Replacing latent thoughts with pause tokens (“pause as thought”) plus curriculum can work well on synthetic logic (e.g., ProntoQA 100.0% ± 0.1%) but underperforms Coconut on GSM8K (24.1% vs 34.1%). This supports the claim that GSM8K requires sequential dependency rather than pure parallel compute (Section 4.4).
  - “Coconut w/o thought” (i.e., removing tokens without adding latent thoughts) performs strongly on logic tasks but not on GSM8K—again pointing to task differences in compute vs chaining needs.

- Do the experiments support the claims?
  - Yes for logic/planning: Large, consistent gains over CoT and No‑CoT, with fewer tokens and faster time, plus careful analyses that explain the mechanism (Sections 4–5).
  - Mixed for open‑domain math: CoT still leads in raw accuracy for this small GPT‑2 base, but Coconut narrows the gap and shows monotonic gains with more latent steps (Figure 3). This suggests scaling or better curricula could close or surpass CoT.

## 6. Limitations and Trade-offs
- Training efficiency and complexity (Section 3, “Training Details”)
  - Each latent thought requires its own forward pass; with `n` latent steps per example, training needs `n+1` sequential passes. This limits parallelism and may slow training despite KV‑cache reuse.
- Reliance on curriculum and supervision (Table 1; Section 4.4)
  - Without staged guidance, the model fails to learn useful latent thoughts (“Coconut w/o curriculum”). The method currently depends on CoT data or curated stage schedules.
- Termination and length control (Section 3)
  - Inference uses a fixed latent length or a classifier to predict `<eot>`. Fixed lengths can be suboptimal across examples; classifiers add components to train.
- Base scale and generality
  - Results are on GPT‑2 and synthetic logic datasets plus GSM8K. It remains to be seen how Coconut scales to larger models and diverse real‑world tasks (e.g., multi‑document QA, coding).
- Interpretability and faithfulness
  - Although probing `softmax(W ht)` offers glimpses into latent branches, latent thoughts are not guaranteed to be human‑readable or faithful explanations. Figure 4’s decodings (“180,” “9”) are illustrative but not a general decoding mechanism.
- Stability with many latent steps
  - Appendix C reports instability when increasing `c` to 3 (accuracy drops and variance rises), suggesting sensitivity to curriculum granularity and optimization dynamics.

## 7. Implications and Future Directions
- How this changes the landscape
  - It demonstrates that LLMs can benefit from “thinking in vectors” rather than words during intermediate computation. This reframes reasoning as a loop over latent states, not tokens, and yields emergent planning behavior (BFS‑like exploration) without hard‑coding a search algorithm (Sections 5.2–5.3).
- Near‑term follow‑ups
  - Pretraining with continuous thoughts: The paper points to pretraining latent reasoning so downstream tasks need less or no CoT supervision (Conclusion).
  - Finer‑grained curricula: Add latent thoughts incrementally (Appendix C) and combine with iCoT’s token‑by‑token removal schedule, potentially improving stability and final accuracy.
  - Learnable termination and adaptive depth: Train a controller to decide how many latent steps to run per instance.
  - Multi‑token prediction objectives: Since Coconut benefits from planning ahead (Section 5.2), pair it with multi‑token prediction pretraining (Gloeckle et al., 2024) to improve foresight.
  - Hybrid reasoning: Generate a sparse “skeleton” in language and fill in details in latent space (Appendix C suggestion), or alternate between latent search and explicit verification.
  - Interpretability tools: Develop better probes for latent branches and value estimates, improving trust and debugging.
- Practical applications
  - Systems needing efficient long‑horizon reasoning with tight token budgets (on‑device assistants, educational tutoring for logic/math) could reduce costs by replacing long CoT traces with compact latent thoughts (Tables 1 and 4).
  - Scientific and engineering planning tasks where breadth exploration and late commitment are crucial (e.g., hypothesis generation, program synthesis, theorem proving) may benefit from the BFS‑like latent search documented in Figures 6–8.

Overall, Coconut is a simple but consequential modification—looping hidden states instead of tokens—that enables LLMs to plan in a continuous space. The empirical gains on planning‑heavy logic tasks, the interpretable BFS‑like analyses, and the clear ablation evidence for the curriculum collectively make a strong case for latent reasoning as a promising direction (Sections 4–5).
