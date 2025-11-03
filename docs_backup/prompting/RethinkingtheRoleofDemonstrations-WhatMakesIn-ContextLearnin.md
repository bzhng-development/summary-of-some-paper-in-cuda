# Rethinking the Role of Demonstrations: What Makes In-Context Learning Work?

**ArXiv:** [2202.12837](https://arxiv.org/abs/2202.12837)

## 🎯 Pitch

This paper offers a surprising empirical analysis showing that large language models do not depend on the correct input–label mappings in demonstrations to achieve strong in-context learning performance on classification and multi-choice tasks. Instead, the key drivers are simply showing the model the label space, in-distribution inputs, and a consistent input–label pair format—randomizing the labels within demonstrations barely reduces accuracy. These findings fundamentally reshape our understanding of what LMs 'learn' at inference time, suggesting new ways to construct prompts, improve zero-shot performance without labeled data, and set more accurate expectations for the capabilities and limitations of in-context learning.

---

## 1. Executive Summary (2-3 sentences)
This paper investigates what, exactly, makes in-context learning (ICL) work in large language models (LMs). It shows that correct input–label pairings in demonstrations are surprisingly unimportant for many classification and multiple-choice tasks: replacing the gold labels with random labels only marginally reduces performance (typically 0–5% absolute), while the main benefits come from providing the label space, in-distribution inputs, and a consistent input–label pair format (Figures 1, 3, 5, 8–10; Sections 4–5).

## 2. Context and Motivation
- Problem addressed:
  - In-context learning (ICL) is when a language model performs a new task at inference time by conditioning on a prompt that contains a few input–label examples, called `demonstrations`, without updating model weights (Figure 2; Introduction).
  - The open question: What aspects of the demonstrations actually drive ICL’s performance? Is the model truly “learning” the input–label mapping from the examples, or are other factors at play (Sections 1, 4–5)?
- Why it matters:
  - Understanding ICL’s mechanism:
    - Theoretically: clarifies whether LMs “learn at test time” vs. recall and adapt pretraining knowledge (Section 6, “Does the model learn at test time?”).
    - Practically: guides how to construct prompts to get the most benefit, potentially elevating zero-shot baselines without labeled data (Sections 4–5, 6).
- Prior approaches and gaps:
  - Prior ICL research optimized prompt formats, example selection, and “chain-of-thought” prompting; others studied brittleness and sensitivity (Related Work).
  - There was limited empirical analysis of why ICL beats zero-shot; particularly, the role of ground-truth input–label correspondences was not isolated (Related Work; Sections 1, 4).
- Positioning:
  - This paper provides a systematic empirical decomposition of demonstrations into four components—`input–label mapping`, `input distribution`, `label space`, and `format`—and quantifies their contributions (Figure 7; Sections 4–5). It evaluates 12 models (774M–175B parameters) across 26 datasets, including GPT-3 and meta-trained LMs (Table 1; Section 3), delivering a cross-model, cross-task perspective.

## 3. Technical Approach
The paper is empirical, designed around controlled interventions that isolate specific aspects of in-context demonstrations. The overall logic is: start from standard k-shot ICL, then swap or remove components to test what changes performance.

- Core setup
  - Demonstrations: sequences of k input–label pairs `(x1, y1), …, (xk, yk)` concatenated before a test input `x`, prompting the model to predict a label `y` (Figure 2; Section 3).
  - Two inference methods (scoring strategies) per model (Section 3; Figure 3):
    - `Direct` method: choose the label `y` from the label set `C` that maximizes `P(y | D, x)` where `D` is the demonstration prompt.
    - `Channel` (noisy-channel) method: flip the conditional and choose `y` that maximizes `P(x | D, y)` (the paper notes you can “flip x and y” to convert Direct ↔ Channel; Section 4.1). This is often used to reduce “label bias” when labels are short tokens.
  - Models: 12 decoder-only LMs from 774M to 175B parameters, including GPT-2, GPT-J, fairseq 6.7B/13B, GPT-3, and `MetaICL` (a GPT-2 Large model meta-trained for ICL on many supervised datasets; Table 1; Section 3).
  - Tasks and metrics: 26 low-resource classification and multiple-choice datasets spanning sentiment, paraphrase detection, NLI, hate speech, QA, and sentence completion (Table 2; Section 3). Metrics are Macro-F1 (classification) and Accuracy (multiple-choice).
  - Demonstration size and sampling: default k = 16, uniformly sampled from training data; 5 random seeds (3 seeds for GPT-3/fairseq-13B due to cost; Section 3).
  - Prompt templates: minimal templates by default; also evaluate hand-written “manual” templates from prior work (Section 4.2, Figure 6; Table 3).

- Key experimental variants
  - Baselines to isolate input–label mapping (Section 4.1; Figure 3):
    - `No demonstrations` (zero-shot with label set `C`).
    - `Demos w/ gold labels`: standard k-shot ICL with `(xi, yi)`.
    - `Demos w/ random labels`: pair each input `xi` with a random label `ỹi ∈ C` drawn uniformly; predict with `P(· | x1, ỹ1, …, xk, ỹk, x)` (Section 4.1).
  - Ablations (Sections 4.2, 5):
    - Fractional correctness: mix of correct and incorrect pairs to reach `a%` correctness (Algorithm 1; Figure 4).
    - Vary k: number of examples per prompt from 0 to 32 (Figure 5).
    - Templates: minimal vs. manual (Figure 6).
  - Decomposing demonstration components (Section 5; Figure 7):
    - `Input distribution`: Replace in-prompt inputs with out-of-distribution (OOD) sentences sampled from a news corpus (CC-News) while keeping label tokens and format intact (“OOD + Random Labels”; Figure 8; Appendix B).
    - `Label space`: Replace labels with random English words of the same cardinality as `C`, destroying knowledge of the true label space but preserving format and inputs (Figure 9).
    - `Format`: Remove the input–label pairing structure:
      - `No labels`: concatenate only inputs `x1 … xk` before test input.
      - `Labels only`: concatenate only labels `y1 … yk` before test input.
      - These serve as “no-format” counterparts to the above variants (Figure 10).
    - Additional controls (Appendix C.3):
      - Replace labels with a constant token like “answer” (hurts performance).
      - Replace all inputs with the same test input paired with random labels (also hurts), suggesting that breaking the pattern of distinct input–label pairs disrupts the “format”.

- How choices isolate mechanisms
  - By holding three components fixed and changing the fourth (Figure 7), each ablation shows the marginal impact of that component:
    - Swap gold → random labels: tests reliance on the input–label mapping itself.
    - Use OOD inputs: tests contribution of in-distribution inputs.
    - Use random English labels: tests how much knowing the actual label set matters.
    - Remove input–label pairing: tests whether the paired format cues the model to continue the pattern.

## 4. Key Insights and Innovations
- Novel contribution 1: Ground-truth input–label mapping often matters little in ICL
  - Insight: Performance with demonstrations that pair inputs with random labels is close to performance with gold-labeled demonstrations across many models and datasets (Figure 3; Section 4.1).
  - Evidence:
    - “Replacing the labels in demonstrations barely hurts performance on a range of classification and multi-choice tasks… consistently over 12 different models including GPT-3.” (Abstract; Figure 3).
    - Typical drops are 0–5% absolute; multi-choice is especially robust (≈1.7% drop on average) compared to classification (≈2.6%; Section 4.1; Figure 3).
    - `MetaICL` shows near-zero sensitivity: only 0.1–0.9% absolute drop (Section 4.1).

- Novel contribution 2: The main gains come from label space + in-distribution inputs + paired format
  - Decomposition (Figure 7; Section 5) shows three drivers:
    - Knowing the label space: Direct models lose 5–16% absolute when labels are replaced with random English words, even though the overall paired format is kept (Figure 9; Section 5.2). Channel models are less affected (0–2%).
    - Conditioning on in-distribution inputs: Replacing prompt inputs with OOD sentences reduces performance by 3–16% absolute for several model/task pairs; in one case it is worse than no demonstrations (Figure 8; Section 5.1).
    - Preserving the input–label pair format: Removing the pairing (inputs-only or labels-only) is typically no better than no demonstrations, despite keeping inputs or labels present (Figure 10; Section 5.3).
  - Significance: The benefits come from “contextualizing” the model on the right input distribution and label set while giving the model an input–label pair pattern to imitate, rather than from learning new mappings (Sections 5, 6).

- Novel contribution 3: Meta-training for ICL amplifies reliance on simple aspects and de-emphasizes true mapping
  - Observation: `MetaICL` (meta-trained to do ICL) is even less sensitive to input–label correctness and more sensitive to format (Section 5.4).
  - Example: With Direct `MetaICL`, input–label mapping and input distribution have “nearly zero influence” in some settings, implying the model keys heavily off the label space and format (Section 5.4; Figures 8–10).
  - Significance: This suggests meta-training encourages models to exploit the easiest-to-use properties (format, familiar token distributions) rather than infer true mappings at test time (Section 5.4).

- Novel contribution 4: Stronger zero-shot baselines without any labeled data
  - Finding: You can achieve nearly k-shot performance by pairing unlabeled inputs with random labels to form “fake” demonstrations—this raises the zero-shot baseline substantially (Section 6, “Significantly improved zero-shot performance”).
  - Quote:
    > “It is possible to achieve nearly k-shot performance without using any labeled data, by simply pairing each unlabeled input with a random label and using it as the demonstrations.” (Section 6)

These are fundamental insights (not just incremental improvements) because they challenge the common assumption that ICL’s advantage comes primarily from “learning” the input–label mapping in the prompt.

## 5. Experimental Analysis
- Evaluation methodology (Section 3):
  - Models (Table 1): `GPT-2 Large` (774M), `MetaICL` (774M, meta-trained for ICL), `GPT-J` (6B), fairseq 6.7B and 13B, and `GPT-3` (175B, “Davinci base” API).
  - Tasks (Table 2): 26 datasets in six categories: sentiment, paraphrase detection, NLI, hate speech detection, question answering, sentence completion. All are classification or multiple-choice (no open-ended generation).
  - Prompting:
    - Minimal templates (Table 3) by default; “manual” templates tested for robustness (Figure 6).
    - Demonstrations are k = 16 by default, uniform sampling; 5 random seeds (3 for GPT-3/13B); Macro-F1 (classification) and Accuracy (multi-choice).
    - Direct and Channel scoring are both evaluated for each model (Figure 3).
  - Reproducibility: Code referenced in Appendix B; data via Hugging Face (Section 3; Appendix A/B).

- Main results (Sections 4, 5; Figures 3–6, 8–10, 11–12):
  - Gold vs. random labels (Section 4.1; Figure 3):
    - Average drop replacing gold with random labels: ≈2.6% absolute (classification), ≈1.7% (multiple-choice).
    - `MetaICL` is exceptionally robust (0.1–0.9% drop).
    - Some exceptions noted (Section 4.1 footnote): for a few Direct setups (GPT-2, GPT-J, fairseq 6.7B) the k-shot performance with gold labels was weak on many datasets; Channel fairseq-13B had zero-shot > k-shot in some cases.
  - Fractional correctness (Section 4.2; Figure 4):
    - Performance is relatively insensitive to fraction of correct labels. Even with 0% correct labels (all incorrect pairings), performance often remains far above no-demos baselines.
    - Examples:
      - With MetaICL (classification), 0% correct retains ≈92% of the improvement over no-demos (Figure 4 caption).
      - With MetaICL (multi-choice), 0% correct retains ≈100% of improvement; with GPT-J (multi-choice), ≈97% (Figure 4 caption).
      - GPT-J (classification) is more sensitive: ≈10% absolute drop at 0% correct—but still much better than no-demos (Section 4.2; Figure 4).
  - Varying k (Section 4.2; Figure 5):
    - Using demonstrations helps even for small k (e.g., k=4).
    - Performance saturates beyond k ≈ 8; increasing k to 16 or 32 brings limited gains for both gold and random labels (Figure 5).
    - The gold→random label performance gap remains small across k (≈0.8–1.6% absolute), with one outlier (≈4.4% at k=4, likely high variance).
  - Templates (Section 4.2; Figure 6):
    - Manual templates confirm the trend: random labels ≈ gold labels. Manual templates are not consistently superior to minimal ones.
  - Decomposition into four aspects (Section 5; Figures 8–10):
    - Input distribution (Figure 8):
      - Replace in-prompt inputs with OOD sentences (while keeping format and random labels): performance drops 3–16% absolute for Channel MetaICL, Direct GPT-J, and Channel GPT-J across classification and multiple-choice.
      - In one setting (Direct GPT-J multi-choice), OOD + random labels is worse than no demonstrations.
      - Direct MetaICL is an exception (little sensitivity), attributed to meta-training (Section 5.4).
    - Label space (Figure 9):
      - Replace labels with random English words (same cardinality): Direct models lose 5–16% absolute; Channel models drop 0–2% or even slightly improve.
      - Interpretation: Direct scoring needs to generate the correct tokens; Channel scoring conditions on them and is less sensitive to their identity (Section 5.2).
    - Format of input–label pairs (Figure 10):
      - Inputs-only and labels-only (no pairing) are generally no better than no demos, showing format matters.
      - Keeping format enables strong performance even when using only inputs or only labels:
        - Direct MetaICL classification: using OOD inputs + random labels retains 95% of the improvement of full in-context learning (gold labels) (Figure 10 caption).
        - Direct MetaICL multi-choice: retains 82%.
        - Channel models:
          - Channel MetaICL classification: pairing training inputs with random English words retains 82% of gains.
          - Channel GPT-J classification: 87%.
          - Channel MetaICL multi-choice: 86%.
          - Channel GPT-J multi-choice: 75%.
  - Dataset-level breakdown and label distribution effects (Appendix C.2; Figure 12):
    - Sampling random labels from the true label frequency distribution reduces the gold→random gap further (classification, channel scoring):
      - Channel MetaICL: gap shrinks from 1.9% to 1.3% absolute.
      - Channel GPT-J: 5.0% to 3.5% absolute.
    - Some datasets show larger sensitivity (outliers):
      - For Channel GPT-J, financial_phrasebank shows nearly 14% absolute drop with random labels in one setting (Figure 12).
  - Additional controls that break format degrade performance (Appendix C.3):
    - Using a constant label token (“answer”) or repeating the test input for all demo inputs both hurt, likely because they distort the pairing pattern.

- Do experiments support claims?
  - Yes, convincingly for classification and multiple-choice tasks under both Direct and Channel scoring across many models:
    - The sheer breadth (12 models, 26 datasets; Tables 1–2) and systematic isolation of components (Figures 7–10) support the central claim that mapping correctness is not the main driver.
    - Multiple robustness checks (fractional correctness, k, templates, label distribution sampling, OOD inputs) triangulate the conclusion.
  - Caveats:
    - Mixed results in a few model/dataset combinations (Section 4.1 footnote; Figure 12) indicate task- and model-specific variability.
    - Results are confined to classification/multi-choice; open-ended generation may behave differently (Limitations).

## 6. Limitations and Trade-offs
- Scope limitations (Section 6: Limitation; Appendix C):
  - Task types: Only classification and multiple-choice. Extending to open-ended generation is nontrivial because it’s hard to “break” input–output correspondences while preserving the output distribution (Section 6; “Extensions to generation”).
  - Dataset variability: While averages show small gaps, some datasets (e.g., financial text, certain hate speech datasets) show larger sensitivity to correct labels (Appendix C.2; Figure 12).
  - Synthetic tasks: Prior reports suggest that in synthetic settings with limited inputs, correct labels can matter more (Limitations; citing Rong 2021).
- Assumptions:
  - Access to unlabeled in-distribution inputs for demonstrations is often assumed—even when using random labels. The paper explicitly treats this as an acceptable “zero-shot” enhancement (Section 6, footnote 10).
  - Label set `C` is known for classification. When `C` is unknown, using random English words helps if the format is preserved (Sections 5.2–5.3; Figure 10).
- Computational constraints:
  - For the largest models (GPT-3, fairseq-13B), experiments use fewer datasets (6) and seeds (3) due to cost (Section 3).
- Model training factors are not exhaustively explored:
  - The effect of pretraining data and objectives on the observed phenomena is discussed as future work (review response lines 055–061).
- Interpretation limits:
  - The study is empirical; it does not provide a formal theory of ICL mechanisms, though it references Bayesian interpretations in prior work (Related Work).

## 7. Implications and Future Directions
- Reframing what ICL “learns” at test time (Section 6):
  - Narrow definition (“learns a new input–label mapping from demonstrations”): evidence suggests LMs typically do not learn this at test time for the evaluated tasks.
  - Broader definition (“adapts to input distribution, label space, and format”): LMs do adapt in this sense; demonstrations act as cues to retrieve and apply pretraining knowledge aligned with the task (Section 6).
- Practical prompting guidance:
  - To boost zero-shot performance without labels:
    - Use in-distribution inputs as demonstrations and pair them with random labels, preserving the input–label pair format (Sections 4–5). This can approach k-shot performance (Section 6).
  - When label tokens are unknown or unsuitable:
    - Use random words as stand-ins, but keep the input–label pair format (Figure 10).
  - Prefer Channel scoring when label identity is ill-defined or biased, as it is less sensitive to label token choices (Figure 9).
- Model training strategy:
  - Meta-training for ICL (`MetaICL`) makes the model rely even more on simple prompt aspects (format, label tokens), reducing reliance on true input–label mapping (Section 5.4). This can be beneficial for robustness to wrong labels but may risk over-reliance on superficial patterns.
  - Future work: Analyze how pretraining corpora and objectives shape the model’s ability to internalize “input–label correspondences” that ICL can then surface (Section 6).
- Extending to other paradigms:
  - Instruction following: Evidence elsewhere suggests models can perform well even with irrelevant or misleading instructions (Webson & Pavlick, 2022). This paper’s findings suggest instructions may function like demonstrations—triggering latent capabilities rather than conveying new mappings (Section 6).
  - Chain-of-thought and rationales: Early evidence shows some kinds of wrong rationales harm, while others do not (Appendix citing Madaan & Yazdanbakhsh, 2022). A “four-aspect” decomposition (input distribution, label space, format, mapping) could inform which rationale properties matter.
- Research directions:
  - Identify tasks where correct input–label mapping is truly needed for ICL (e.g., financial or nuanced hate speech tasks; Figure 12).
  - Develop prompting or scoring methods that better extract latent input–label associations when present, or detect when they are absent and require fine-tuning.
  - Extend the decomposition framework to open-ended generation and structured prediction, designing interventions that preserve output distributions while breaking mappings.

Block-quoted highlights to anchor takeaways:
> “Replacing the labels in demonstrations barely hurts performance on a range of classification and multi-choice tasks, consistently over 12 different models including GPT-3.” (Abstract; Figure 3; Sections 4.1, 5)

> “It is possible to retain 95% and 82% of improvements from in-context learning … by simply sampling random sentences from a corpus and randomly pairing them with the label set” (Direct MetaICL, classification and multi-choice; Figure 10 caption; Section 5.3)

> “Removing the format is close to or worse than no demonstrations, indicating the importance of the format.” (Section 5.3; Figure 10)

> “Performance with random labels is very close to performance with gold labels” (Figure 3; Sections 4.1–4.2)

By disentangling what demonstrations contribute, this paper shifts the field’s understanding of ICL from “learning mappings from a few examples” toward “format- and distribution-based cueing of pretrained capabilities,” and provides concrete, widely applicable prompting strategies for practitioners.
