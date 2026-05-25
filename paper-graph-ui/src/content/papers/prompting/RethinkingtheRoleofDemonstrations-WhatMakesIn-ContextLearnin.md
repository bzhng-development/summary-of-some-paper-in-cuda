# Rethinking the Role of Demonstrations: What Makes In-Context Learning Work?

**ArXiv:** [2202.12837](https://arxiv.org/abs/2202.12837)

## 🎯 Pitch

This paper challenges conventional wisdom by empirically demonstrating that ground-truth input-label pairs in in-context learning prompts are largely unnecessary for strong performance in large language models—randomly assigned labels in demonstrations result in only marginal accuracy loss. The authors show that the real drivers of ICL gains are simply exposing the model to the label space, input distribution, and the input–label pair format, which means LMs mainly tap into pretrained knowledge rather than learning new tasks at test time. This reframing not only demystifies ICL, but also has profound implications for how we design, prompt, and trust language models in practical and scientific settings.

---

## 1. Executive Summary (2-3 sentences)
This paper empirically dissects what, exactly, demonstrations in in-context learning (ICL) contribute to large language model performance. Across 12 models and 26 tasks, it finds that using ground-truth labels in the demonstrations is surprisingly unnecessary—randomly replacing those labels barely hurts accuracy—while the main gains come from three simpler properties of the prompt: seeing examples of the input distribution, seeing the label space, and following the input–label pair format (Sections 4–5; Figures 1, 3, 8–10). This reframes ICL as mostly leveraging pretrained priors that are “triggered” by the prompt structure, not learning a new input–label mapping at test time.

## 2. Context and Motivation
- The problem addressed
  - In-context learning lets a language model solve a new task by conditioning on a few “demonstrations” (input–label pairs) without gradient updates (Figure 2; Section 1). What remains unclear is which parts of those demonstrations matter. Do models actually learn the mapping from inputs to labels shown in the prompt, or are other factors driving performance?
- Why this matters
  - Practically: ICL is widely used when fine-tuning is costly or unavailable for very large models. Knowing what aspects of a prompt drive performance determines how to design effective prompts cheaply and safely.
  - Theoretically: Understanding whether models “learn” new tasks at inference or just retrieve and calibrate pretrained knowledge affects how we evaluate and improve LMs (Section 6).
- Prior approaches and gaps
  - Many works have optimized prompts (ordering, templates), selected better examples, or trained instruction-following models and meta-ICL models (Section 2). But there has been little empirical decomposition of which demonstration components (correct labels, label set, input distribution, format) are necessary for ICL gains.
  - Related analyses suggest ICL can be cast as Bayesian inference (Xie et al., 2022) and that performance correlates with pretraining term frequency (Razeghi et al., 2022), but they don’t directly test the contribution of each demonstration component in real tasks.
- Positioning
  - This paper isolates four aspects of demonstrations (input–label mapping, input distribution, label space, and format; Figure 7) and systematically tests their impact via controlled ablations (Sections 4–5). The central, counterintuitive finding is that correct input–label pairings matter far less than assumed (Sections 4.1–4.2; Figures 1, 3–6).

## 3. Technical Approach
The paper is empirical. It defines variants of demonstrations to isolate each component, then measures performance across many models and tasks.

- Core setup
  - Models: 12 decoder-only LMs, including `GPT-2 Large (774M)`, `MetaICL (774M)`, `GPT-J (6B)`, `fairseq 6.7B`, `fairseq 13B`, and `GPT-3 (Davinci, ~175B)` (Table 1). Each is evaluated with two inference options:
    - `Direct method`: choose the label `y` that maximizes the conditional probability `P(y | x, context)`.
    - `Channel method`: score labels by how well they explain the input, approximated by “flipping” input and label in the prompt and selecting `y` that maximizes something proportional to `P(x | y, context)` (Section 3; see Min et al., 2021a).
  - Tasks: 26 datasets spanning sentiment, paraphrase, natural language inference, hate speech, multiple-choice QA, and cloze-style completion (Appendix A; Table 2).
  - Demonstrations: `k = 16` input–label pairs by default, sampled uniformly from the training split; five random seeds per dataset (Section 3).
  - Metrics: Macro-F1 for classification; Accuracy for multiple-choice; report macro-averages across datasets (Section 3).

- Decomposition of demonstration aspects (Figure 7)
  - `Input–label mapping`: are inputs paired with their correct labels?
  - `Input distribution`: do the demonstration inputs resemble test inputs (in-domain) or come from unrelated text (out-of-distribution, OOD)?
  - `Label space`: what are the possible outputs (e.g., Positive/Neutral/Negative)? Even if labels are mismatched to inputs, seeing the set can matter.
  - `Format`: the structural pattern of alternating input and label, i.e., “input ⏎ label,” repeated k times.

- Demonstration variants (Sections 4–5; Appendix B/C)
  - Baselines:
    - `No demonstrations`: pure zero-shot; select `y` from the label set without any examples.
    - `Gold labels`: standard ICL with correctly paired demonstrations `(x_i, y_i)`.
  - Key ablations:
    - `Random labels`: replace each demonstration label with a random label from the true label set `C` (uniform; Appendix C.2 also tries sampling from the true label distribution).
    - `Vary fraction correct`: mix correct and incorrect pairings to get `a%` correct labels (Algorithm 1; Figure 4).
    - `Vary k`: number of demonstrations `k ∈ {0, 4, 8, 16, 32}` (Figure 5).
    - `Input distribution`: use OOD inputs (random sentences from CC-News) paired with random labels from `C` (“OOD + Random labels,” Figure 8); sentences are length-matched to inputs (Appendix B).
    - `Label space`: replace label words with random English words (same cardinality as `C`) to erase the true label space (“Random English words,” Figure 9).
    - `Format`: remove input–label pairing by providing only inputs (“No labels”) or only labels (“Labels only”) as the context; compare to their formatted counterparts (Figure 10).
    - Additional checks (Appendix C.3): constant label “answer” for all demos; or using the test input repeated for all demonstration inputs.

- Example (Table 4)
  - For sentiment: replacing “Positive/Neutral/Negative” with random words (“unanimity,” “wave”) keeps the same format and input distribution but removes the true label space. Conversely, “No labels” keeps the input distribution but removes label space and pairing.

Why this design? Each ablation turns one “knob” while trying to hold others fixed, enabling causal attribution:
- Compare `Gold` vs. `Random labels` to test whether correct input–label mapping matters (Section 4.1).
- Compare `In-domain` vs. `OOD` inputs (holding format and label space) to test input distribution (Section 5.1; Figure 8).
- Compare `Random labels (true label set)` vs. `Random English words` (holding inputs and format) to test the role of the label space (Section 5.2; Figure 9).
- Compare formatted pairs vs. unpaired lists (“No labels,” “Labels only”) to test the importance of the pairwise format (Section 5.3; Figure 10).

## 4. Key Insights and Innovations
- Ground-truth input–label mapping in demonstrations is largely unnecessary
  - Finding: Replacing gold labels by random labels in the demonstrations “only marginally” reduces performance across 12 models and many datasets (Sections 4.1–4.2; Figures 1, 3–6).
  - Evidence:
    - Average absolute drop is small: “1.7%” for multi-choice and “2.6%” for classification (Section 4.1; Figure 3).
    - `MetaICL` is especially insensitive: drop is “0.1–0.9%” absolute (Section 4.1).
    - Even with 0% correct labels, models retain most of the improvement over zero-shot:
      > “preserving 92%, 100% and 97% of improvements from using the demonstrations with MetaICL in classification, MetaICL in multi-choice, and GPT-J in multi-choice, respectively” (Figure 4; Section 4.2).
  - Significance: This challenges the assumption that ICL “learns” a new mapping during inference; instead, it suggests models rely on pretrained priors once the prompt tells them what label tokens exist, what the inputs look like, and how to format answers.

- The three drivers of ICL gains: input distribution, label space, and format
  - `Input distribution` matters:
    - Using OOD inputs in demonstrations causes large drops: “3–16% absolute” for `Channel MetaICL`, `Direct GPT-J`, and `Channel GPT-J` (Figure 8; Section 5.1). In one case, OOD was worse than no demonstrations (Direct GPT-J, multi-choice).
    - Intuition: In-domain inputs make the task closer to language modeling behavior the model has seen in pretraining—providing anchors for the LM to “lock onto” (Section 5.1).
  - `Label space` matters (for direct scoring):
    - Replacing the true label set with random English words (keeping inputs and format) significantly hurts `Direct` models by “5–16% absolute” (Figure 9; Section 5.2).
    - `Channel` models barely change (0–2% absolute, sometimes better), likely because they condition on labels to explain inputs and do not have to generate the exact label token (Section 5.2).
  - `Format` (paired input–label structure) is crucial:
    - Removing pairing (inputs only or labels only) is usually no better than zero-shot (Figure 10; Section 5.3).
    - Conversely, keeping the pairwise format enables strong performance even when only one side carries “useful” information:
      > With `Direct MetaICL`, using random sentences (OOD) paired with the true label set retains “95%” (classification) and “82%” (multi-choice) of the gold-demo improvement (Figure 10).
      > With `Channel` models, pairing in-domain inputs with random English words retains “82%–87%–86%–75%” across four settings (Figure 10).

- Meta-training amplifies reliance on simpler prompt aspects
  - Observation: `MetaICL` (trained with an ICL objective) exhibits near-zero sensitivity to the input–label mapping and, in some configurations, to the input distribution or label space (Section 5.4).
  - Hypothesis: Meta-training encourages the model to rely almost exclusively on easier-to-exploit signals (format, seeing examples of inputs or labels) rather than learning the specific mapping on the fly (Section 5.4).

- A stronger zero-shot baseline emerges
  - If you can access unlabeled training inputs, pairing them with random labels already achieves near k-shot performance (Section 6). This redefines what a “zero-shot” baseline should be when unlabeled data is permitted (also discussed in Appendix C.2).

These are fundamental insights, not just incremental tweaks: they reinterpret what ICL is doing mechanistically and how to construct effective prompts with minimal supervision.

## 5. Experimental Analysis
- Evaluation methodology (Section 3; Appendix A/B)
  - Models: 12 LMs; 6 sizes/styles, each with `Direct` and `Channel` inference (Table 1).
  - Datasets: 26 tasks; true low-resource (<10k training examples), diverse domains (Table 2). Evaluation on dev sets (HuggingFace Datasets).
  - Demonstrations: `k = 16` by default; 5 seeds. For `fairseq 13B` and `GPT-3`, a subset of 6 datasets and 3 seeds due to resource limits (Section 3).
  - Templates: “Minimal” dataset-agnostic templates for inputs and labels; also test “Manual” templates (Figure 6; Appendix B; Table 3).
  - Prompt formatting: model-specific separators/newlines to structure “input ⏎ label,” repeated (Appendix B).

- Main quantitative results
  - Replacing gold labels with random labels barely hurts:
    > “models see performance drop in the range of 0–5% absolute” (Section 4.1; Figures 1 and 3). On average: “1.7%” (multi-choice) and “2.6%” (classification) drops.
    > `MetaICL` is the least sensitive (0.1–0.9% absolute; Section 4.1).
  - Fraction-correct ablation (Figure 4):
    - Insensitive trend: even 0% correct labels yields large gains over no demos.
    - Exception: `GPT-J` in classification sees ~“10%” absolute drop when all labels are incorrect, but still outperforms zero-shot.
  - Varying number of demonstrations `k` (Figure 5):
    - Small gains saturate after `k ≥ 8` for both gold and random labels.
    - The gap between gold and random labels remains small: “0.8–1.6%” absolute (except a 4.4% blip at `k = 4` in classification, likely variance).
  - Templates (Figure 6):
    - Manual templates do not consistently beat minimal templates.
    - The “random labels ≈ gold labels” trend holds regardless of template choice.
  - Input distribution (Figure 8):
    - OOD + random labels drops performance “3–16%” absolute for multiple models; in one case worse than zero-shot (Direct GPT-J, multi-choice).
  - Label space (Figure 9):
    - Direct models: significant drops “5–16%” when replacing label set with random English words.
    - Channel models: minimal change (0–2%).
  - Format (Figure 10):
    - Removing input–label pairing (“No labels” or “Labels only”) is typically near or below zero-shot performance.
    - Keeping the pair format allows strong recovery even if only inputs or only labels carry task-relevant information (retaining up to 95%/82% of gold-demo gains in Direct MetaICL; and 82%–87%–86%–75% in Channel settings).
  - Additional analyses
    - Sampling random labels from the true label distribution (instead of uniform) further reduces the performance gap: e.g., from “1.9% → 1.3%” (Channel MetaICL) and “5.0% → 3.5%” (Channel GPT-J) absolute (Appendix C.2).
    - Per-dataset variation exists: some tasks like `financial_phrasebank` show larger drops (up to “~14%” absolute in one model–dataset pair; Appendix C.2; Figure 12).
    - Robustness checks (Appendix C.3): constant label “answer” harms performance (breaks format); repeating the test input for all demos also hurts (changes the structure), underscoring the importance of a natural pairwise format.

- Do the experiments support the claims?
  - Yes. The conclusions hold across:
    - Many models and sizes (774M to 175B), including both public and API-only models (Table 1).
    - Both `Direct` and `Channel` scoring.
    - Many datasets across categories and domains (Table 2).
  - The paper also transparently reports exceptions (e.g., Direct GPT-J classification is more label-sensitive; Section 4.1).

- Notable caveats
  - A few model–dataset pairs show unusual behavior (e.g., `fairseq 13B` Channel no-demos outperforming gold demos in classification; Section 4.1), but these do not overturn the broader pattern.

## 6. Limitations and Trade-offs
- Assumptions and scope
  - The tasks are classification or multiple-choice with short outputs; open-ended generation is out of scope (Section 6; Limitations).
  - The approach assumes at least some implicit input–label mapping is already encoded by pretraining; when that mapping is sparse or absent (e.g., synthetic tasks), ground-truth labels may matter more (Limitations; citing subsequent work and Rong, 2021).
  - The “OOD” corpus is CC-News; other OOD sources might produce different magnitudes of degradation (Appendix B).
- Data and computational constraints
  - For the largest models (`fairseq 13B`, `GPT-3`), only 6 datasets and 3 seeds were used (Section 3). While trends matched smaller models, full coverage is limited.
- Methodological trade-offs
  - The `Channel` method advantage partly stems from conditioning on labels; it is therefore less affected by destroying the label space (Figure 9). This complicates direct comparisons between Direct and Channel.
  - Replacing labels with random English words preserves a label-like token but injects semantics that are unrelated to the task; this is deliberate to remove the true label space, but could have idiosyncratic effects across tasks.
- Open questions
  - How to extend this decomposition to generation tasks where outputs are open-ended? Designing variants that break input–output pairing while preserving output distribution is nontrivial (Section 6; Limitations).
  - How do instruction-following models behave under similar decompositions? Early evidence suggests that irrelevant or misleading instructions can still help (Webson & Pavlick, 2022), but a full analysis is pending (Section 6).

## 7. Implications and Future Directions
- Reframing what ICL “learns”
  - Strictly speaking, the model does not learn a new input–label mapping at test time; it mostly uses pretraining priors once the prompt tells it:
    - what the inputs look like (input distribution),
    - what label tokens exist (label space), and
    - how to format outputs (format).
  - The paper writes:
    > “our findings suggest that LMs do not learn new tasks at test time” (Section 6), if “learning” is defined as acquiring a new input–label mapping from the demonstrations.
  - Under a broader definition (adapting to distributions and format), ICL still “learns” from the prompt (Section 6).

- Practical prompt design
  - If labeled examples are scarce or noisy, you can still get most of the ICL gains by:
    - including in-domain inputs (even without correct labels),
    - making the label set explicit (e.g., enumerating allowed outputs),
    - preserving a clear input–label pair format (Figures 8–10).
  - For `Direct` scoring, the exact label tokens matter more (Figure 9); for `Channel`, the label set matters less but pairing inputs remains crucial (Figure 10).

- Stronger zero-shot baselines
  - When unlabeled training inputs are available, pairing them with random labels and using a good format can approach k-shot performance (Section 6; Appendix C.2). This should be the new baseline for fair comparisons.

- Model training implications
  - Meta-training with an ICL objective can over-emphasize “easy” prompt cues (format, distribution) while ignoring the actual mapping (Section 5.4). Future training schemes might:
    - explicitly encourage models to use input–label correspondences in context (e.g., by adversarially matching formats but swapping mappings),
    - diversify pretraining to cover more task semantics so that “priors” exist for more tasks,
    - or combine light fine-tuning with ICL to bind task semantics when pretraining priors are weak.

- Research directions
  - Extend this decomposition to:
    - open-ended generation (design “wrong but distribution-preserving” outputs),
    - chain-of-thought prompting (initial evidence: some wrong rationales hurt, others don’t; see Madaan & Yazdanbakhsh, 2022),
    - instruction-following (test whether irrelevant/misaligned instructions still help via input/label/format cues).
  - Diagnose dataset-specific behavior (Appendix C.2; Figure 12) to predict when correct labels will matter (e.g., financial sentiment).
  - Theoretical models that reconcile these findings with Bayesian or retrieval-based accounts of ICL (Section 2; Xie et al., 2022).

In short, Figures 1, 3–6 show that “gold labels” are not the main story; Figures 8–10 show what actually matters (input distribution, label space, format). Algorithm 1 and Appendix C detail robustness checks. This reshapes both how we design prompts in practice and how we conceptualize what ICL is doing under the hood.
