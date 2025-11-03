# Stay on topic with Classifier‑Free Guidance

**ArXiv:** [2306.17806](https://arxiv.org/abs/2306.17806)
**Authors:** Guillaume Sanchez, Honglu Fan, Alexander Spangher, Elad Levi, Pawan Sasanka Ammanamanchi, Stella Biderman
**Institutions:** 

## 🎯 Pitch

Introducing Classifier-Free Guidance (CFG) for autoregressive language models, this paper innovatively adapts an image diffusion technique to boost prompt adherence at inference time without retraining. This method enhances output accuracy and consistency, often rivaling much larger models, marking a significant leap in practical language generation while offering theoretical insights into real-time token distribution control.

---

## 1. Executive Summary (2-3 sentences)
This paper adapts Classifier-Free Guidance (CFG) from image diffusion to autoregressive language models as a pure inference-time method that requires no retraining. By combining two forward passes—one conditioned on the prompt and one “unconditioned”—with a tunable weight `γ`, it boosts prompt adherence and accuracy across diverse tasks, sometimes rivaling models twice the size at the same inference compute (Figure 11; Table 4), and sets a zero-shot SOTA for LAMBADA using `LLaMA-7B` (Figure 2b).

## 2. Context and Motivation
- Problem addressed
  - Language models often drift from the user’s intent during generation (hallucinations, meandering, and degradation over long outputs). Standard decoding treats the initial prompt and the generated continuation equally, so the model’s focus on the prompt can fade with time (Section 1; Table 1 shows an assistant ignoring a system instruction unless guided).
- Why it matters
  - Practical: better adherence reduces hallucinations and style drift in assistants, code, and long-form writing.
  - Theoretical: it explores a general way to reshape the token distribution at decode time without changing model parameters.
- Prior approaches and gaps
  - Training-time methods: instruction tuning and RLHF improve alignment but are compute- and data-intensive (Section 1).
  - Inference-time controllability in NLP: PPLM, FUDGE, GeDi, contrastive decoding adjust token distributions but rely on auxiliary discriminators or additional models.
  - CFG in vision removes the external classifier by using the generator’s own conditional vs. unconditional predictions but typically requires training with conditioning dropout.
- Positioning
  - This work extends CFG to text generation and shows that autoregressive LMs already support both conditional and “unconditional” predictions (by dropping the prompt), so CFG works out-of-the-box for text (Section 2.2). The paper evaluates the method broadly (zero-shot QA/commonsense, Chain-of-Thought reasoning, code, translation, assistants) and analyzes why it works (entropy and vocabulary effects; Section 5).

## 3. Technical Approach
Classifier-Free Guidance for language models in one sentence: at each decoding step, compute logits with and without the prompt and form a guided logit vector that moves probabilistic mass toward tokens more consistent with the prompt.

- Background: guidance in generative models (Section 2.1)
  - Classifier Guidance (for diffusion): reweights the sample distribution as `P_b(x|c) ∝ P_θ(x) · P_ϕ(c|x)^γ` (Equation 1), where `c` is a condition (label/text), `γ` is guidance strength, and `P_ϕ` is a classifier over samples.
  - Classifier-Free Guidance (CFG): remove the external classifier using Bayes’ rule, obtaining `P_cθ(x|c) ∝ P_θ(x|c)^γ / P_θ(x)^(γ−1)` (Equation 2). Intuitively, push samples toward what the conditional model prefers and away from what the unconditional model prefers.
- Adapting CFG to autoregressive LMs (Section 2.2)
  - Key observation: decoder-only LMs naturally give both conditional `P_θ(w|c)` and “unconditional” `P_θ(w)` predictions because dropping the prompt `c` is just starting from a later position in the context window.
  - Notation
    - `w` is the output token; `w_1…w_T` a sequence.
    - `c` is the prompt (instruction/context/beginning text).
    - `logits` are the pre-softmax scores over the vocabulary; they form a linear space that is convenient for combination.
  - Step-by-step decoding with CFG (Equation 7)
    1. At time `i`, compute conditional logits `log P_θ(w_i|w_{<i}, c)` using the full context including the prompt.
    2. Compute “unconditional” logits `log P_θ(w_i|w_{<i})` by dropping `c` (the paper’s implementation starts the unconditional stream at the last token of the prompt; Section 3.1).
    3. Combine them in logit space:
       - `log P_cθ(w_i|w_{<i}, c) = log P_θ(w_i|w_{<i}) + γ · (log P_θ(w_i|w_{<i}, c) − log P_θ(w_i|w_{<i}))`
       - This is a vector step of size `γ` from the unconditional logits toward the conditional logits.
    4. Sample the next token from the softmax of the combined logits using any standard sampler (e.g., nucleus).
  - Why this design: operating in logits avoids changing the network (no editing, architecture-agnostic). Autoregressive LMs do not need special training to provide both conditional/unconditional passes, unlike diffusion models (Section 2.2).
- Negative prompting (Sections 2.1, 3.4)
  - Idea: emphasize the difference between a desired condition `c` and a “negative” condition `c̄` (what we want to avoid). In diffusion, Equation 5 generalizes CFG to move away from `c̄` toward `c`. In LMs, the paper instantiates this by setting `c̄` to a model’s default system prompt and `c` to an edited system prompt (e.g., “write a sad response”), guiding generations to follow the edit (Section 3.4; Figure 5).
- How it differs from temperature or penalties
  - Temperature rescales the entire distribution uniformly; CFG changes relative preferences by moving toward “prompt-consistent” logits and away from unconditional ones. Empirically, CFG reduces entropy (Figure 6a) and reorders top tokens (Figure 6b), not merely sharpening them.

## 4. Key Insights and Innovations
- A general, training-free CFG for text (Section 2.2)
  - Novelty: uses the LM’s native ability to compute conditional and “unconditional” logits; no conditioning-dropout retraining or external classifier is needed.
  - Significance: immediately deployable across models and tasks; adjustable strength `γ` offers a precision dial between prompt adherence and diversity.
- Broad, consistent gains across tasks with small `γ` (Sections 3.1–3.4)
  - Zero-shot benchmarks: accuracy gains for GPT‑2, Pythia, and LLaMA families; especially strong on LAMBADA where `LLaMA‑7B` with `γ=1.5` reaches 81.3% vs. 73.6% at `γ=1` (Figure 2b), surpassing much larger closed models cited in the paper’s leaderboard context.
  - Chain-of-Thought (CoT): small `γ` increases parseable, valid answers and improves accuracy; high `γ` keeps answers valid but reduces correctness (Figure 3; Appendix C.4).
  - Code generation (HumanEval): pass@1 improves for low `γ` across CodeGen models; high `γ` harms performance (Table 2; Tables 5–7).
  - Assistants with negative prompts: human raters prefer CFG outputs 75% of the time at `γ=3`, without hurting user-prompt relevance until `γ≥4` (Figure 5).
- Compute/scale equivalence (Section 4)
  - Finding: at equal inference FLOPs, CFG often matches a model twice the size without CFG; in 5/9 tasks the difference is statistically insignificant at `p=.01` (Figure 11; Table 4), with two tasks favoring CFG and two favoring vanilla.
  - Practicality: same VRAM footprint as the smaller model; adds roughly a second forward pass in latency.
- Understanding what CFG changes (Section 5)
  - Entropy and diversity: CFG reduces token-level entropy (mean ~4.7 vs. 5.4; Figure 6a), shrinking the number of tokens in the top‑p mass and increasing focus on prompt-relevant words (Figure 6b).
  - Not equivalent to instruction tuning: entropy looks similar, but vocabulary preferences differ; CFG overlaps ~50% of the top‑p tokens with the vanilla model and has only partial overlap with instruction‑tuned variants (Figures 6b, 7).
  - Prompt-level visualization tool: ranking tokens by the delta `log P(w_t|w_{<t},c) − log P(w_t|w_{<t})` shows what CFG upweights/downweights; in the example “The dragon flew over Paris, France”, Paris/dragon tokens rise while other regions/dates/topics fall (Table 3).

## 5. Experimental Analysis
- Evaluation setup (Sections 3.1–3.4; Appendix C, D)
  - Models: GPT‑2 (small–XL), Pythia (160M–12B), LLaMA (7B–65B), Falcon‑7B (for analyses), CodeGen‑mono (350M/2B/6B), GPT‑J (exploratory), Bloom‑3B, RedPajama‑Incite‑3B, mT0 (translation), GPT4All‑J (assistant).
  - Tasks and metrics
    - Zero-shot: ARC‑c/e, BoolQ, HellaSwag, PIQA, SciQ, TriviaQA, WinoGrande, LAMBADA (accuracy; Figure 2).
    - CoT: GSM8K and AQuA using WizardLM‑30B and Guanaco‑65B; accuracy and % invalid answers (Figure 3; Appendix C.4).
    - Code: HumanEval; pass@k with k∈{1,10,100}, temperatures 0.2/0.6/0.8 (Table 2; Tables 5–7; Figures 12–14).
    - Assistants: human preference for following system prompt and user prompt (611 votes, 71 voters) across `γ∈{1…6}` (Figure 5).
    - Cost study: accuracy vs per-token FLOPs across tasks (Figure 11; Table 4).
    - Mechanism study: entropy/top‑p overlap and comparisons to instruction‑tuned variants (Figures 6–7; Figure 16; Tables 8–10).
- Main quantitative results
  - Zero-shot improvements (Figure 2)
    - Broad gains at `γ=1.5` across families; exceptions include ARC‑c and WinoGrande where gains are inconsistent.
    - LAMBADA standout:
      > `LLaMA‑7B`: 73.6% (γ=1) → 81.3% (γ=1.5) (Figure 2b).  
      This zero-shot score exceeds previously reported large-model baselines cited by the paper (Section 3.1 narrative under Figure 2).
  - Chain-of-Thought (Figure 3; Appendix C.4)
    - With WizardLM‑30B and Guanaco‑65B on GSM8K/AQuA, small `γ` increases both validity and accuracy; too-large `γ` keeps validity high but drops accuracy. The curves in Figure 3 show accuracy peaking around `γ≈1.25–1.5` and % invalid decreasing at small `γ`.
  - Code generation (Table 2; Figures 12–14)
    - At temperature 0.2, pass@1 improves at small `γ`:
      > CodeGen‑2B: 19.5% → 20.9% (γ=1.5); falls to 16.5% at γ=2.0 (Table 2).  
      > CodeGen‑6B: 19.5% → 20.9% (γ=1.5); down to 16.5% at γ=2.0 (Table 2).  
      > CodeGen‑350M: 11.0% → 11.8% (γ=1.1) then declines (Table 2).
    - As `k` increases (pass@10/100), CFG’s benefits diminish or invert because it reduces diversity; larger `k` recovers hard cases that CFG may suppress (Section 3.3.2 discussion).
    - Task-by-task scatter (Figure 15a–c) shows more wins than losses for `γ=1.25` (e.g., at temp=0.6, 24 wins vs. 16 losses).
  - Assistants with negative prompts (Figure 5; Table 16)
    - Human study with GPT4All‑J shows:
      > System-prompt following preference peaks at 75% for γ=3 (Figure 5).  
      > User-prompt relevance remains roughly balanced (~52%) up to γ<4; begins degrading beyond that (Figure 5).
  - Compute vs. accuracy (Figure 11; Table 4)
    - Regression analysis on log-FLOPs vs accuracy across tasks finds:
      > In 5/9 tasks, the regression lines for CFG and vanilla are statistically indistinguishable at p=.01 (Table 4).  
      > Significant differences favor CFG on LAMBADA and SciQ; vanilla on WinoGrande and TriviaQA (Table 4).
  - Mechanistic analyses (Section 5; Figures 6–7; Table 3)
    - Entropy drop:
      > Mean entropy ~4.7 with CFG vs. ~5.4 vanilla across generation steps (Figure 6a).
    - Top‑p overlap:
      > CFG shares ~50% of tokens in top‑p=0.9 with vanilla; overlap profiles differ from instruction-tuned models (Figure 6b; Figure 16).
    - Qualitative visualization:
      > Tokens about dragons and Paris move up; unrelated regions/dates/topics move down (Table 3).
- Do experiments support the claims?
  - Yes, with caveats. The gains are strongest for short-answer zero-shot tasks and pass@1 in code; moderate `γ` consistently helps CoT validity/accuracy. The compute study and human preference study substantiate practical relevance. Where diversity matters (pass@100) or tasks are sensitive (WinoGrande), gains are smaller or negative.
- Robustness and ablations
  - `γ` sweeps are thorough across tasks; multiple temperatures for code; per-task scatters for HumanEval; FLOP-accuracy ANCOVA; entropy/top‑p analyses; negative prompting human evaluation with varying `γ`. Additional appendices cover translation (Table 11), GPT‑J code language adherence (Table 12), and ablations for image-like code generation (Table 13).

## 6. Limitations and Trade-offs
- Tuning required
  - Performance depends on `γ`: small values (≈1.1–1.5) help; large values harm accuracy/diversity (Figures 3, 12–14; Table 2). There is no single best `γ` for all tasks.
- Diversity vs. adherence
  - CFG reduces entropy and narrows the candidate set (Figure 6a–b). This improves pass@1 but can hurt pass@100 and creative variability (Section 3.3.2 discussion).
- Task sensitivity
  - Not all benchmarks improve; e.g., WinoGrande often degrades (Figure 2b; Table 4), and ARC‑c shows mixed impact (Figure 2a).
- Computational cost
  - Inference roughly doubles forward passes (conditional + unconditional). Although VRAM usage is unchanged, latency increases; benefits offset this only when accuracy gains or smaller-model equivalence matter (Section 4; Figure 11).
- Negative prompting setup
  - Assistant experiments use a specific choice of `c̄` (the model’s default system prompt) and may not generalize to all systems or safety constraints (Section 3.4).
- Security and alignment concerns
  - Stronger prompt adherence could amplify prompt injection or attempts to override safety guidance. The paper flags this as an untested risk and calls for standardized safety benchmarks (Conclusion).

## 7. Implications and Future Directions
- Field impact
  - CFG becomes a simple, model-agnostic knob for prompt adherence in LMs. It offers a way to “buy” alignment and accuracy with inference compute rather than training compute, and can make smaller open models competitive with much larger ones in certain tasks (Figure 11).
- Research avenues
  - Adaptive or token-span–specific guidance: guide only certain parts of the prompt or Chain‑of‑Thought segments; weight the reasoning section differently from the final answer (Section 3.2 outlook).
  - Schedule `γ` over time: high at the start for on-topic setup, lower later to restore diversity.
  - Better negative prompting: learn or detect `c̄` automatically; use safety personas or anti-instructions; evaluate against prompt-injection benchmarks.
  - Combine with other decoding controls: contrastive decoding, calibrated temperature, or evidence‑aware decoding (related work cited in Section B).
  - Mechanistic understanding: more granular studies of vocabulary reordering, agreement with instruction-tuned models under varying prompt lengths and perplexities (Figures 6–7; Tables 8–10).
- Practical applications
  - Assistants and tools: stronger system-prompt compliance for style, persona, or form (Figure 5; Table 1).
  - Program synthesis: better single-sample correctness (pass@1) for IDE integration (Table 2).
  - Long-form generation and translation: staying on-topic and faithful to source; small `γ` gains observed in translation for certain prompts/models (Table 11).
  - Low-resource deployment: enable smaller models with CFG to meet accuracy targets that previously required larger models (Figure 11; Table 4).

Overall, the paper shows that a simple two-pass, logit-space interpolation—grounded in the CFG principle—reliably increases prompt adherence across tasks without retraining, clarifies why it works (entropy reduction and vocabulary reordering), and maps the trade-offs and regimes where it helps most.
