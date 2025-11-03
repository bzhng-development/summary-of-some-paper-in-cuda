# ON LINEAR REPRESENTATIONS AND PRETRAINING DATA FREQUENCY IN LANGUAGE MODELS

**ArXiv:** [2504.12459](https://arxiv.org/abs/2504.12459)

## 🎯 Pitch

This paper uncovers a direct, quantitative link between the frequency with which facts (subject–object pairs) appear in pretraining data and the emergence of simple, interpretable linear representations of those facts inside language models. By showing that linear structure forms reliably above measurable frequency thresholds, and that these structures can in turn be used to predict pretraining data statistics in opaque models, this work empowers both principled model analysis and targeted data curation—paving the way for designing language models whose knowledge and behaviors can be more deliberately shaped and understood.

---

## 1. Executive Summary (2-3 sentences)
This paper investigates a simple, measurable link between what language models (LMs) see during pretraining and how they internally represent facts: linear representations for factual relations tend to emerge once the relevant words co-occur often enough in the pretraining data. Using OLMo (1B and 7B) and GPT-J (6B), the paper shows strong correlations between subject–object co-occurrence frequency and the quality of linear representations, identifies frequency thresholds at which these structures reliably appear during training, and demonstrates that measurements of linearity can be used to predict how often terms appeared in pretraining—even for a different model with different data.

## 2. Context and Motivation
- Problem addressed
  - Many LM behaviors depend on pretraining data, but we lack mechanistic understanding of what properties of that data cause specific internal representations to form. The paper targets a concrete question: when and why do “linear representations” of factual relations (e.g., mapping France → Paris in a “country–capital” relation) arise inside LMs?
- Why this matters
  - Practical: If certain capabilities require particular training exposure, we can shape desired behaviors by curating data (e.g., ensure enough examples to reach thresholds).
  - Scientific: A mechanistic link between frequency and representational form clarifies how LMs store and recall knowledge, building bridges from data statistics to internal geometry.
- Prior approaches and gaps
  - Prior interpretability work has identified linear structures for some relations but not others and has debated why (e.g., Hernandez et al., 2024; Chanin et al., 2024). Static word embeddings work suggested frequency helps analogical linearity (Ethayarajh et al., 2019), but it was unclear how this extends to contextual LMs and across training time.
  - Data inference efforts often rely on membership inference, memorization, or tokenizer analysis; they do not leverage linear representational measurements to infer pretraining statistics.
- Positioning
  - This paper connects interpretability (linear relation decoding) to pretraining corpus statistics. It quantifies when linear relational embeddings (LREs) emerge, ties emergence to co-occurrence counts across training checkpoints, and uses linearity metrics to infer pretraining term frequencies (Sections 3–5; Figures 1–3; Table 1).

## 3. Technical Approach
The paper’s methodology has two pillars: (A) measuring linear representations for factual relations, and (B) counting pretraining frequencies—both across training time.

A. Measuring linear representations with LREs (Section 2.1; Equation (1))
- What is an LRE?
  - A “Linear Relational Embedding” (LRE) is a linear approximation of the internal computation that maps a subject token to an object token for a fixed relation (e.g., “X plays the Y”). It captures a direction/affine map in the model’s hidden space that can predict or edit the output object given the subject representation.
- How it is computed
  - Setup: Consider an LM processing a few-shot prompt for a relation. Let `s` be the hidden state at a middle layer for the subject token, and `o` be the final hidden state at the last token position used to decode the object (Section 2.1).
  - The LM’s nonlinear mapping from `s` (plus context `c`) to `o` is linearized via a first-order Taylor expansion around example subjects `si`. Formally (Equation (1)):
    - `F(s, c) ≈ W s + b`, where `W = E[∂F/∂s (si, ci)]` and `b = E[F(s, c) − (∂F/∂s)s | (si, ci)]`, averaging over n examples (n=8 here).
  - In practice: They estimate Jacobians from model activations in a 5-shot prompting setup and fit one LRE per relation using 8 examples (Section 3.1).
- Two evaluation metrics (Section 3.1)
  - `Faithfulness`: Does applying the LRE produce the same object token the unedited LM would produce? This checks whether the linear map matches the LM’s own prediction on that instance.
  - `Causality`: If we edit the subject representation with the LRE to target another object in the same relation (e.g., push “Miles Davis” toward “guitar” instead of “trumpet”), how often does the LM’s output change accordingly? This tests whether the linear edit causally controls the model’s prediction across subject–object pairs. The paper prefers `Causality` because `Faithfulness` can be artificially high early in training when the LM predicts the same frequent tokens (Section 3.1).
  - Two variants also appear in Section 5: “Faith Prob.” (log-prob under the LRE of the correct object) and “Hard Causality” (the edited object must become top-1).
- Design choices vs. prior work
  - One `β` scalar per relation to scale the learned `W`, instead of one per model (Section 3.1), to avoid disadvantaging specific relations with different scales.
  - They do not require the 8 examples used to fit the LRE to be correctly predicted by the model, enabling comparison across checkpoints with consistent training examples (Section 3.1; Appendix B shows no notable difference in quality when using incorrect examples; Figure 4/5).
  - Layer selection and hyperparameters are tuned via sweeps for causality/faithfulness (Appendix C; Figures 6–9).

B. Counting pretraining frequencies across time (Section 3.2)
- What to count and why
  - Target statistic: subject–object co-occurrence frequency for factual triples (e.g., “Miles Davis” and “trumpet” appearing together). Prior work shows co-occurrence is a good proxy for mentions of a fact (Elsahar et al., 2018), and it’s what a causal recall task might depend on.
- Two counting routes
  - `WIMBD` (“What’s in My Big Data?”) can count over full corpora (e.g., The Pile, Dolma), but not per training batch or checkpoint (Section 3.2). The paper uses it for GPT-J final counts.
  - `Batch Search` (new tool released): counts exact co-occurrences in the actual tokenized sequences used to train OLMo, enabling per-checkpoint counts (Section 3.2). This avoids overcounts from long documents and matches how the LM sees data (sequence-length-bounded). They search ~10k terms across ~2T Dolma tokens, running on ~900 CPUs in ~1 day.
  - Validation: Batch Search vs. WIMBD closely align on final checkpoints (slope ~0.94, r=0.99; Appendix D; Figure 10); WIMBD slightly overestimates due to whole-document counting.

Experimental setup (Sections 3–4)
- Models and data
  - OLMo-7B (0424) and OLMo-1B (0724), trained on Dolma; intermediate checkpoints up to 2T tokens (Section 4.1).
  - GPT-J (6B), trained on The Pile; only final checkpoint counts via WIMBD (Section 4.1).
- Relations dataset
  - 25 factual relations (e.g., “country–capital”, “person–mother”) with 10,488 unique subjects/objects; “landmark-on-continent” is dropped due to skew (Section 3.1; footnote 2). Prompts are 5-shot; each LRE uses 8 examples.

## 4. Key Insights and Innovations
1) Frequency threshold → emergence of linear structure (Figures 1–2; Section 4.2)
- Novelty: A clear, quantitative relationship is shown between average subject–object co-occurrence frequency and LRE quality across relations and across training time.
- Evidence:
  - Strong correlation between log co-occurrence and LRE `Causality` (r = 0.82; Section 4.2).
  - Thresholds where average `Causality` > 0.9: ≈1,097 (GPT-J 6B), ≈1,998 (OLMo-7B), ≈4,447 (OLMo-1B) mean co-occurrences per relation (Figure 2, top-right table).
  - Crucially, reaching the threshold at any training stage suffices: even early checkpoints show strong LREs for high-frequency relations (Figure 2, red/blue/gray points tracing checkpoints).

2) Linearity tracks—but is not identical to—task accuracy (Section 4.3; Appendix F)
- Novelty: The paper disentangles representational linearity from in-context performance.
- Evidence:
  - Correlation with co-occurrence is higher for `Causality` (0.82) than for 5-shot accuracy (0.74) in OLMo-7B (Section 4.3).
  - Relations can have high few-shot accuracy but low linearity when frequency is low, e.g., “star–constellation name” has 84% 5-shot accuracy yet 44% causality and ~21 avg co-occurrences (Section 4.3).
  - Conversely, some relations show earlier rises in `Causality` than in accuracy (e.g., “food-from-country” has 65% `Causality` vs 42% 5-shot early; the gap closes during training; Section 4.3; Appendix F, Figures 13–14).

3) Using linearity to infer training data frequencies, even across models (Section 5; Figure 3; Table 1)
- Novelty: A regression model trained on LRE metrics predicts object frequencies (and to a lesser extent co-occurrences) in the pretraining corpus; it generalizes to a different LM with different data without direct supervision.
- Evidence:
  - Within-magnitude (10×) accuracy improves markedly when using LRE features vs. LM-only likelihood/accuracy features (Figure 3). Object frequency prediction reaches ~70% accuracy with LRE features; LM-only features are near baselines.
  - Cross-model generalization: a regressor trained on OLMo LRE features predicts GPT-J term frequencies (and vice versa). LRE-based models outperform LM-only features and the mean baseline (Table 1). Feature permutation shows “Hard Causality” contributes ~15% absolute accuracy (Appendix E; Figure 12).

4) A practical tool for per-batch frequency counting (Section 3.2; Appendix D)
- Novelty: A fast, exact “Batch Search” implementation that counts co-occurrences in tokenized sequences as seen by the LM, enabling fine-grained, per-checkpoint analyses. The code is released.

## 5. Experimental Analysis
- Evaluation methodology
  - Relations and prompts: 24 factual relations from Hernandez et al. (2024) after filtering (Section 3.1), 5-shot prompts, 8 examples to fit each LRE.
  - Metrics: `Faithfulness`, `Causality`, “Hard Causality,” and “Faith Prob.” (Section 3.1 and Section 5.1). For frequency prediction, the target label is either object frequency or subject–object co-occurrence frequency from the pretraining data.
  - Data/counting:
    - OLMo: counts per checkpoint via Batch Search on Dolma (Section 3.2; 2T tokens).
    - GPT-J: final counts via WIMBD on The Pile (Section 3.2).
  - Regression setup: Random forest with 100 trees; leave-one-relation-out splits over 4 seeds; features either LM-only (log-prob of correct object and average 5-shot accuracy) or LM+LRE (adds `Faithfulness`, `Causality`, “Faith Prob.,” “Hard Causality”) (Section 5.1).
  - Reported accuracy: “within-magnitude accuracy” (prediction within 10× of the true count), and mean absolute error in natural log space (Section 5.2).

- Main quantitative results
  - Frequency → linearity (Section 4.2; Figure 2)
    > “Co-occurrence frequencies highly correlate with causality (r = 0.82). Subject-only (r = 0.66) and object-only (r = 0.59) correlations are notably lower.”
    > “Mean causality > 0.9 is reached at ≈1,097 (GPT-J), ≈1,998 (OLMo-7B), ≈4,447 (OLMo-1B) average co-occurrences.”
    - Independence from training stage: once a relation’s average co-occurrence crosses the threshold, strong LREs appear even in early checkpoints (Figure 2, red/gray markers).
  - Linearity vs. accuracy (Section 4.3; Appendix F)
    > “Correlation with co-occurrence is 0.82 for causality vs. 0.74 for 5-shot accuracy in OLMo-7B.”
    > “Some relations are accurate but not linear (e.g., ‘star–constellation’), generally at low frequency.”
  - Predicting frequencies from linearity (Section 5.2; Figure 3)
    > “Using LRE features outperforms LM-only features by ~30% absolute for object frequency within-magnitude accuracy; LM-only is near mean/random baselines.”
    > “Object prediction MAE (ln space): 2.1 (LRE+LM) vs. 4.2 (LM-only). Subject–object MAE: 1.9 (LRE+LM) vs. 2.3 (LM-only).”
    - Caveat: The paper notes subject–object prediction is harder and only marginally above baseline in some analyses due to label clustering around the mean (Section 5.2).
  - Cross-model generalization (Section 5.3; Table 1)
    > “Evaluating on GPT-J with a regressor trained on OLMo LRE features yields 0.65±0.12 within-magnitude accuracy for object frequencies, versus 0.42±0.10 with LM-only features and 0.31±0.15 with the mean baseline.”
    > “For subject–object co-occurrences, LRE features reach 0.76±0.12 (Eval on GPT-J) vs. 0.66±0.09 (LM-only) and 0.57±0.15 (mean baseline).”
    - Despite these numbers, the text also warns that subject–object prediction can be close to strong baselines due to tight distributional clustering (Section 5.3).
  - Feature importance (Appendix E; Figure 12)
    > “‘Hard Causality’ is the most important feature for generalization (~15% absolute accuracy impact), followed by faithfulness-related measures (~5%).”

- Robustness and ablations
  - Example selection for LRE fitting: using incorrect examples does not significantly harm causality/faithfulness (Appendix B; Figures 4–5).
  - Layer/β/rank sweeps: per-layer/per-hyperparameter analyses show where LREs work best (Appendix C; Figures 6–9). Layer selection is chosen by maximizing `Causality`.
  - Counting validation: Batch Search vs. WIMBD agreement (Appendix D; Figure 10).
  - Commonsense relations: correlation with frequency exists but is weaker (r ≈ 0.42) and concept-to-co-occurrence mapping is less reliable (Appendix G; Figure 15).

- Do the experiments support the claims?
  - Yes for the core claims. The strong r=0.82 correlation (Figure 2), frequency thresholds observed across models, and cross-checkpoint invariance support the main thesis that pretraining co-occurrence frequency predicts linearity emergence.
  - The regression results (Figure 3; Table 1) credibly show that linearity metrics encode frequency signals absent from LM likelihood/accuracy alone and that this signal transfers across models.
  - Some nuances remain around subject–object prediction difficulty and distributional baselines, which the paper surfaces and contextualizes.

## 6. Limitations and Trade-offs
- Assumptions and approximations
  - Co-occurrence as a proxy for factual mentions: Works well for factual relations (Section 3.2; Elsahar et al., 2018), but less so for commonsense relations where semantics are not simply co-location (Appendix G).
  - Linearity test class: The work centers on LREs (affine, first-order approximations). Other interpretability methods (e.g., sparse autoencoders) may reveal different structure; training and consistency across checkpoints were practical reasons not to use them here (Section 5.1, related-work note).
- Scope limitations
  - Focus on factual relations; limited exploration of non-factual/commonsense relations shows weaker frequency–linearity links (Appendix G).
  - Three models examined; thresholds likely depend on both frequency and scale (Section 4.2), but broader scaling laws are not established.
- Causality vs. correlation
  - While the frequency–linearity link is strong and predictive, the work does not perform counterfactual pretraining to prove causality of exposure (Limitations, Appendix A).
- Computational constraints
  - Batch-level counting at 2T tokens required significant compute (≈900 CPUs for ~1 day; Section 3.2). This may limit widespread adoption without similar infrastructure.
- Prediction challenges
  - Subject–object co-occurrence prediction is difficult: label distributions are tight, so strong baselines already perform well (Sections 5.2–5.3). Some relations generalize poorly (e.g., “star–constellation”; Table 2), likely due to low frequency and domain specifics.

## 7. Implications and Future Directions
- How this changes the landscape
  - Provides a concrete operationalization for when linear relational structure forms in LMs: once average subject–object co-occurrences cross model-dependent thresholds (Figure 2; Section 4.2). This reframes parts of “emergence” as a predictable outcome of exposure frequency, not just model size or architecture.
- Practical applications
  - Data curation for capability shaping: To elicit robust linear relational structure (and thus controllable, steerable behavior via linear edits), ensure key subject–object pairs exceed the identified co-occurrence thresholds during pretraining.
  - Pretraining data inference for closed-data, open-weights models: Measure LRE `Causality`/`Faithfulness` on the model and use the regression to estimate which domains/terms were frequent in its training corpus (Section 5.3; Table 1).
  - Safety/privacy auditing: Combine LRE-based frequency inference with other methods to flag potential overexposure of sensitive entities.
- Research directions
  - Toward causal tests: Controlled pretraining where co-occurrence is systematically varied would directly test causation and refine threshold estimates.
  - Beyond factual relations: Develop better proxies than co-occurrence for relations where semantics are not well captured by co-location; extend to procedural or abstract knowledge.
  - Nonlinear mechanisms: Identify tasks where LREs fail but performance is high (e.g., “star–constellation”) to map when LMs use non-linear strategies (Section 6; discussion in “Linear Representations in LMs”).
  - Integration with other interpretability tools: Combine LREs with sparse autoencoders and tokenizer-based data mixture inference to improve frequency prediction bounds and feature interpretability (Appendix E; related work).
  - Scaling laws: Systematically study how model size and data scale shift the frequency thresholds.

Overall, the paper delivers a clear, experimentally grounded link from pretraining exposure frequency to linear relational structure in LMs, supplies an efficient counting tool, and demonstrates a practical route to infer training data properties from representational measurements—shaping both interpretability research and data-centric training practices.
