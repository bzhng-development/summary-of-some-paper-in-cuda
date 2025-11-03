# RobustFT: Robust Supervised Fine‑tuning for Large Language Models under Noisy Response

**ArXiv:** [2412.14922](https://arxiv.org/abs/2412.14922)
**Authors:** Junyu Luo, Xiao Luo, Kaize Ding, Jingyang Yuan, Zhiping Xiao, Ming Zhang
**Institutions:** Peking University, UCLA, Northwestern University, University of Washington

## 🎯 Pitch

ROBUSTFT introduces a robust fine-tuning framework for large language models by detecting and correcting noisy training data through a multi-view detection process and context-enhanced relabeling, ensuring improved performance across various benchmarks. This approach is crucial as it prevents performance degradation due to data noise, offering a significant advancement for deploying LLMs in real-world scenarios where data quality can be inconsistent.

---

## 1. Executive Summary (2–3 sentences)
ROBUSTFT is a two-stage supervised fine‑tuning framework that makes large language models (LLMs) robust to noisy training responses by detecting mislabeled or low‑quality samples and then relabeling and filtering them before fine‑tuning. It matters because even modest noise in downstream fine‑tuning data can severely degrade LLM performance; for example, with 30% noise, standard SFT loses 8.9 points on MMLU versus a vanilla model, whereas ROBUSTFT not only avoids that drop but improves over vanilla across five benchmarks (Table 1, Section 4.2.1).

## 2. Context and Motivation
- Problem addressed
  - Downstream SFT data often contains noise: incorrect answers from human errors, processing mistakes, or model hallucinations (Section 1). The core question is whether an LLM can automatically detect and correct such noisy responses so that fine‑tuning still helps rather than harms.
- Why it matters
  - Noisy SFT can catastrophically hurt performance. Figure 1 shows accuracy decreasing rapidly as noise increases; with 30% noisy responses, standard SFT on MMLU underperforms the vanilla model by 8.9 points. This is highly relevant to real deployments where data curation is imperfect.
- Shortcomings of prior approaches
  - Classical noisy‑label methods focus on fixed, discrete labels (classification) and do not handle open‑ended generation or leverage context within datasets (Section 5.1).
  - In LLM settings, self-selection methods perform poorly at flagging noise because models can be overconfident or hallucinate (Section 4.2.1; SelfSelect underperforms SFT on multiple tasks in Table 1).
  - Inference enhancement (e.g., retrieval-augmented generation) helps modestly but does not address noise in training data (Table 1; SelfRAG).
- Positioning
  - ROBUSTFT targets open‑ended LLM SFT with noisy outputs by combining multi‑view noise detection (base prediction, reasoning‑enhanced prediction, original label) and a denoising pipeline (context‑enhanced relabeling + review + entropy filtering), then fine‑tunes on the cleaned set (Sections 3.2–3.5, Figure 2).

## 3. Technical Approach
At a high level, ROBUSTFT builds a high‑quality fine‑tuning dataset from a noisy one by:
1) detecting likely noisy samples,
2) relabeling them using context and reasoning, and
3) keeping only confident relabeled samples.
Finally, it fine‑tunes the model on the union of originally clean and high‑confidence relabeled items (Algorithm 1, Section 3.5).

Step‑by‑step

1) Problem setup (Section 2.2)
- The downstream dataset is `D_task = {(q_i, y_i)}_i`, with queries `q_i` and target responses `y_i`, some of which are wrong.
- Goal: identify mislabeled pairs and denoise them to form a refined set for SFT.

2) Multi‑view noise detection (Section 3.2)
- Generate two independent predictions per sample:
  - Base prediction: `ŷ_i = M(q_i)` (Eq. 1).
  - Reasoning‑enhanced prediction: the model iterates “reasoning then reflection” steps to refine its answer, represented as `ŷ_i^reas = M_Reas(q_i, M_Refl(M_Reas(q_i, ...)))` (Eq. 2). The idea is to encourage more deliberate chains of thought and self‑correction before producing a final response.
- Consistency‑based checking:
  - A `Checker` compares the original label `y_i`, the base prediction `ŷ_i`, and the reasoning‑enhanced prediction `ŷ_i^reas` to output a reliability flag `r_i ∈ {0,1}` (Eq. 3). If the three views agree sufficiently, `r_i=1` (treated as reliable); otherwise `r_i=0` (potentially noisy). The exact consistency metric is an agreement‑based criterion (Figure 2 and Eq. 3).
- Partition:
  - Reliable set `D_clean = {(q_i, y_i) | r_i=1}` and potential‑noise set `D_noise = {(q_i, y_i) | r_i=0}`.

Why this design?
- Single‑view detection (e.g., only base predictions) is unreliable because LLMs can hallucinate and be overconfident. The second, independent view (reasoning+reflection) provides a distinct signal, and combining both with the original label gives a better estimate of label correctness (Section 3.2; ablations “w/o Checker” confirm this in Table 3).

3) Denoising (relabeling) the flagged samples (Section 3.3)
- Retrieve helpful context from clean examples:
  - Compute an embedding for each query using an `Encoder`: `h_i = Encoder(q_i)` (Eq. 4).
  - For each noisy sample `(q_i, y_i) ∈ D_noise`, find the `k` most similar clean queries in `D_clean` using the embeddings. These serve as in‑context examples.
- Context‑enhanced generation:
  - Condition the LLM on the noisy query plus the `k` retrieved clean pairs to produce a new answer `ŷ_i^cont` (Eq. 5). Intuition: learn from closely related, high‑confidence examples to reduce hallucination and bias toward correct patterns.
- Review and synthesis:
  - A `Review Agent` inspects both `ŷ_i^cont` (context‑based) and `ŷ_i^reas` (reasoning‑enhanced) and produces a final relabeled answer `ỹ_i = Review(q_i, ŷ_i^cont, ŷ_i^reas)` (Eq. 6).
  - Conceptually, this acts like a panel discussion: one expert offers context‑guided evidence, another offers deliberate reasoning, and the reviewer resolves conflicts and synthesizes the most reliable answer.

Why this design?
- Context helps constrain the solution to patterns seen in verified clean data; reasoning offers step‑by‑step validation. Their combination is more robust than either alone (Table 3: removing Context‑Enhanced Relabeling or Reasoning‑Enhanced LLM degrades accuracy).

4) Confidence‑based selection via entropy (Section 3.4)
- Compute a token‑level entropy score for each context‑enhanced response `ŷ_i^cont`:
  - `H(ŷ_i^cont) = -(1/N) Σ_j log p(y_ij | q_i, y_i<j)` (Eq. 7), the average negative log probability across tokens. Lower entropy means the model is more certain.
- Keep only the top‑β fraction of relabeled samples with lowest entropy: `D_select = {(q_i, ỹ_i) | rank(H(ŷ_i^cont)) ≤ β |D_denoise|}` (Eq. 8). Default `β = 50%` (validated in Section 4.3.2; Figure 3).

Why entropy?
- For generation, entropy is a natural uncertainty measure. Low entropy correlates with confident, consistent responses; filtering by it trims uncertain relabels (Table 3, “w/o Selection,” shows large drops if this step is removed).

5) Final fine‑tuning (Section 3.5)
- Combine `D_clean` and `D_select` to form `D_ft`.
- Fine‑tune the base model `M` on `D_ft` with standard next‑token log‑likelihood loss: `M' = arg min_M E_{(q,y) ∼ D_ft} [−log p_M(y|q)]` (Eq. 9).

Implementation notes (Section 4.1.3)
- Fine‑tuning uses LoRA adapters for efficiency, trained 2 epochs via LLaMA‑Factory.
- Defaults: selection ratio `β = 50%` and context length `k = 3` (sensitivity in Section 4.3.2, Figure 3). The reasoning‑reflection loop uses multiple iterations (they set `n=4`, Section 4.1.3).

A simple example (conceptual)
- A training item claims “The capital of Australia is Sydney.” Base and reasoning models both predict “Canberra,” disagreeing with the label. The `Checker` flags it as noisy. The system retrieves similar clean QA items about capitals and generates a context‑informed answer (“Canberra”) and a reasoning‑trace answer (“Government seat is Canberra; Sydney is largest city”), then the `Review Agent` selects or synthesizes “Canberra” as the relabel. If the entropy is low (confident), the item is kept for fine‑tuning.

## 4. Key Insights and Innovations
- Multi‑view noise detection for open‑ended generation
  - What’s new: compares three views—original label, base prediction, and a reasoning‑enhanced prediction (Eq. 3)—to flag noisy items. This goes beyond classification‑only noisy‑label methods by leveraging model diversity in generation (Section 3.2, Figure 2).
  - Why it matters: ablation “w/o Checker” lowers ARC from 84.9→82.7 (30% noise) and MMLU from 68.2→65.3 (Table 3), showing multi‑view detection is pivotal.

- Context‑enhanced relabeling with a Review Agent
  - What’s new: for each flagged item, retrieve `k` most similar clean examples and generate a new answer, then combine it with a reasoning‑based answer and have a reviewer synthesize the final label (Eqs. 5–6; Figure 2).
  - Why it matters: both components are necessary—removing Context‑Enhanced Relabeling (CER) or the Reasoning‑Enhanced LLM (REL) consistently reduces accuracy across noise levels (Table 3).

- Entropy‑based selection tailored to generation
  - What’s new: ranks relabeled samples by token‑level entropy and keeps only the most confident half by default (Eq. 7–8).
  - Why it matters: removing selection causes the largest drop among ablations (e.g., MMLU 68.2→65.7 at 30% noise; Table 3). Sensitivity analysis shows performance peaks around β=40–50% and degrades if more noisy samples are let in (Figure 3).

- End‑to‑end “self-contained” denoising before SFT
  - The framework operates using the same or closely related LLMs as “experts” rather than external annotators or fixed noise models (Section 3; Section 5.1 discussion). This makes it adaptable across domains and model sizes (Table 2).

## 5. Experimental Analysis
- Evaluation setup (Section 4.1)
  - Datasets: five benchmarks cover general knowledge (MMLU), grade‑school science reasoning (ARC), biomedical QA (PubMedQA), reading comprehension with numerical reasoning (DROP), and finance text polarity (FPB). Noise rates: 30%, 50%, 70% injected into training splits (Section 4.1.1).
  - Models: LLaMA‑3.2‑3B, LLaMA‑3.1‑8B, Gemma‑2‑9B (Section 4.1.2).
  - Baselines:
    - `Vanilla`: no fine‑tuning.
    - `SFT`: standard supervised fine‑tuning on noisy data.
    - Strong SFT recipes: Hermes‑3, Tulu‑3.
    - Denoising/selection: NoiseAL, SelfLabel, SelfSelect.
    - Inference enhancement: SelfRAG (Section 4.1.2).
  - Training: LoRA via LLaMA‑Factory, 2 epochs (Section 4.1.3).

- Main quantitative results (Table 1; Section 4.2.1)
  - On `LLaMA‑3.1‑8B`:
    - MMLU: ROBUSTFT 68.2/68.0/67.6 vs Vanilla 65.3 and SFT 59.5/47.5/37.3 at 30/50/70% noise.
      - Quote: “↑ vs. SFT 14.6, 43.2, 81.2” relative improvements (Table 1).
    - ARC: 84.9/84.7/84.1 vs Vanilla 82.7 and SFT 70.7/61.7/47.5.
    - PubMedQA: 75.8/75.6/75.0 vs Vanilla 72.0 and SFT 66.4/36.7/32.8.
    - DROP: 90.3/88.5/87.9 vs Vanilla 87.2 and SFT 85.3/78.6/66.4.
    - FPB: 84.4/80.5/76.2 vs Vanilla 75.5 and SFT 79.7/58.4/34.9.
  - Takeaway: standard SFT collapses as noise increases; ROBUSTFT consistently beats both vanilla and all baselines across datasets and noise levels.

- Cross‑architecture results (Table 2; Section 4.2.2)
  - Gains are consistent across sizes:
    - `LLaMA‑3.2‑3B` MMLU: 58.5/58.2/57.9 vs SFT 55.0/48.4/38.3 and Vanilla 54.9.
    - `Gemma‑2‑9B` ARC: 91.8/91.5/90.4 vs SFT 77.9/64.6/55.0 and Vanilla 90.2.
  - Insight: Larger models are not inherently more noise‑robust; small models benefit markedly from denoising (Section 4.2.2).

- Ablations and sensitivity (Sections 4.3.1–4.3.2; Table 3; Figure 3)
  - Removing selection (`w/o Selection`) is most damaging; e.g., ARC 84.9→83.2 (30% noise).
  - Removing `Checker` also hurts notably; e.g., MMLU 68.2→65.3 (30% noise).
  - Varying β (selection ratio) shows a sweet spot around 40–50%; beyond that, performance drops as more uncertain relabels slip in (Figure 3 left).
  - Varying `k` (context length) shows improvements up to `k≈3–5`, then saturation (Figure 3 right).

- Additional diagnostics
  - Perplexity distributions (Figure 4, Section 4.3.3): ROBUSTFT concentrates density at lower perplexities compared to SFT across noise settings, indicating more confident, consistent language modeling.
  - Category‑wise MMLU (Figure 5, Section 4.3.4): noise harms knowledge‑intensive fields (History, Health, Law) more; ROBUSTFT lifts performance broadly rather than only in a few categories.
  - Stability (Figure 6, Section 4.3.5): with five prompt paraphrases using GPT‑4o, mean accuracy remains steady and variance increases only slightly at higher noise.

- Do the experiments support the claims?
  - Yes, across five tasks, three model sizes, and three noise levels, ROBUSTFT systematically outperforms SFT and baseline denoising/selection methods. The ablations isolate the value of each component, and sensitivity/diagnostics explain why the method works (lower uncertainty; better selection).

- Notable comparisons and caveats
  - Some baselines (e.g., SelfLabel) perform competitively on a few tasks (e.g., FPB), but they are inconsistent across datasets (Table 1). ROBUSTFT is consistently top‑performing.

## 6. Limitations and Trade-offs
- Detection details are under‑specified
  - The `Checker` is described as a “consistency‑based mechanism” (Eq. 3, Figure 2), but the exact agreement metric and thresholds are not fully detailed in the main text. This can affect reproducibility and might be sensitive to prompt phrasing.
- Dependence on internal model quality
  - If both base and reasoning‑enhanced models share biases or fail on a domain, the multi‑view agreement might be misleading (false “clean” or “noisy” decisions).
- Reliance on similar clean neighbors
  - Context‑enhanced relabeling assumes `D_clean` contains semantically similar, correct examples. In low‑resource or highly novel domains, nearest neighbors may be scarce or misleading (Section 3.3).
- Extra compute and system complexity
  - ROBUSTFT adds several inference passes: base prediction, reasoning‑reflection loop (multiple iterations), retrieval, context‑conditioned generation, and review. This increases preprocessing time and cost before SFT, especially at scale.
- Entropy as a proxy for correctness
  - Low entropy correlates with confidence, not necessarily correctness. Overconfident wrong answers could pass the filter; conversely, correct but diverse answers might be filtered out. The authors partially mitigate this with multi‑view inputs to the reviewer.
- Noise model and realism
  - The paper injects noise at fixed rates (30/50/70%) but defers details of noise type to the appendix (Section 4.1.3). Real‑world noise can be structured (systematic misconceptions) or adversarial; robustness under such patterns needs further study.
- Parameter clarity
  - Section 4.1.3 mentions `n=4` and `θ=50%` without fully tying them to earlier notation (the main selection ratio is `β`). This minor inconsistency should be clarified for precise replication.

## 7. Implications and Future Directions
- How this changes the landscape
  - It demonstrates that noise‑robust SFT for open‑ended generation is feasible and beneficial, reversing the common outcome where noisy SFT degrades performance (Figure 1 and Table 1). The framework reframes SFT as “detect‑and‑repair before train,” which could become standard practice when curating instruction or QA data.
- Promising follow‑ups
  - Calibrated multi‑view agreement: learn a probabilistic `Checker` that models dependencies between views and outputs a calibrated noise score rather than a hard label.
  - Soft weighting instead of filtering: use entropy and agreement to assign per‑sample weights during SFT (rather than discarding), enabling the model to learn from uncertain but informative cases.
  - More diverse noise and domains: test against adversarially perturbed labels, domain shift, and long‑form generation with multiple correct answers to stress the selection criterion.
  - Efficient implementations: approximate the reasoning‑reflection loop or use smaller specialist “experts” to reduce preprocessing cost.
  - Human‑in‑the‑loop review: route only borderline cases (medium entropy, conflicting views) to annotators, combining ROBUSTFT with targeted human verification for maximal data quality per dollar.
- Practical applications
  - Enterprise fine‑tuning with imperfect logs (customer support, finance, healthcare): detect and fix noisy resolutions before training.
  - Self‑training loops for domain adaptation: harvest task‑specific data from weak sources, denoise with ROBUSTFT, and iterate to grow high‑quality corpora.
  - Continual learning: periodically re‑run ROBUSTFT to maintain data quality as distributions and operating conditions evolve.

> Bottom line: ROBUSTFT’s combination of multi‑view detection, context‑aware relabeling, and uncertainty‑based selection delivers consistent gains over vanilla and noisy SFT, across models and tasks (Tables 1–2). The approach is practical, methodically validated (ablations, sensitivity, perplexity/stability analyses), and opens a clear path to reliable SFT on imperfect real‑world data.
