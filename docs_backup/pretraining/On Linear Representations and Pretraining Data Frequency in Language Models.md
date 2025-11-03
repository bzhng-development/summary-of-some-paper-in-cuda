# On Linear Representations and Pretraining Data Frequency in Language Models

**ArXiv:** [2504.12459](https://arxiv.org/abs/2504.12459)
**Authors:** Jack Merullo, Noah A. Smith, Sarah Wiegreffe, Yanai Elazar
**Institutions:** University of Washington (Noah A. Smith), University of Michigan or Allen Institute? (Jack Merullo?), ?

## 🎯 Pitch

This paper introduces a groundbreaking method to infer the frequency of factual relations in a language model's pretraining data through the linearity of their internal representations. By establishing frequency thresholds that predict the emergence of linear relational structures, it offers a novel auditing tool for closed-data models, enhancing our ability to steer and interpret language models based on their data exposure.

---

## 1. Executive Summary (2-3 sentences)
This paper shows that when subject–object pairs in a factual relation (e.g., “France–Paris” in “capital of”) appear often enough in a language model’s pretraining data, the model tends to encode that relation as a simple linear transformation between internal representations. It further demonstrates that the strength of this linear structure lets us estimate how frequently specific terms occurred in pretraining—even for a different model with different training data—introducing a new, representation-based way to infer properties of otherwise-unknown corpora.

## 2. Context and Motivation
- Problem addressed
  - Recent interpretability work finds that many capabilities of language models (LMs) can be approximated by simple linear structures in hidden-state space, but these structures do not emerge uniformly across concepts and relations. What governs whether such linear representations form? The paper targets the role of pretraining data frequency in the emergence of linear relational structure (Abstract; §1; §2.1).
- Why it matters
  - Theoretically: Linear structure enables precise interventions (“steering”) and mechanistic analyses. Knowing when linearity appears helps explain why some tasks are easier to localize and edit (Intro; §2.1).
  - Practically: If frequencies set thresholds for linear behaviors, dataset curation can deliberately shape model capabilities and interpretability. Also, if hidden-state linearity encodes data statistics, it may allow auditing closed-data models (§1; §5).
- Prior approaches and their limits
  - Performance-frequency links are well documented: models do better on high-frequency terms, especially in few-shot settings (cited in §1). But existing work largely measures task accuracy, not the structure of internal computations.
  - Linear representations in modern LMs are known (e.g., Linear Relational Embeddings, “LREs”), but variations across relations were unexplained (e.g., some relations are highly linear, others not; §1; §2.1; Hernandez et al., 2024).
  - Static embeddings work tied analogy quality to frequency (Ethayarajh et al., 2019), but this does not directly explain dynamic contextual models (§2.1).
- This paper’s positioning
  - It bridges interpretability (linear structure) and data analysis (pretraining frequency): measuring exact co-occurrences across pretraining and aligning them with the strength of linear relational structure (§3.2; Fig. 1).
  - It extends from correlation to utility: using linearity metrics to predict term frequencies, even across models with different data (§5; Table 1).

## 3. Technical Approach
Step-by-step view of the pipeline:

- What is a Linear Relational Embedding (LRE)?
  - Goal: approximate the computation a transformer performs to map a subject to the correct object in a factual relation using a simple affine transform in hidden-state space.
  - Setup: Place an example of the relation in a natural language prompt (few-shot format). Extract:
    - `s`: the hidden state at the subject token in an intermediate layer.
    - `o`: the final-layer hidden state at the last token position (the one that decodes the answer token) (§2.1).
  - Approximating the computation:
    - Define the model’s mapping from `s` (with few-shot context `c`) to `o` as `F(s, c) = o`.
    - Use a first-order Taylor approximation around observed subject points `si` (from `n=8` in-relation examples) to estimate a linear map `W` and bias `b`:
      - Equation (1): `F(s, c) ≈ W s + b`, with
        - `W = E_{si,ci}[∂F/∂s |_(si,ci)]`
        - `b = E_{si,ci}[F(s, c) - (∂F/∂s) s |_(si,ci)]`
      - Intuition: the Jacobian `∂F/∂s` averaged over few examples gives a relation-specific “directional map” from subjects to objects (§2.1; Eq. 1).
  - Two evaluation metrics (from Hernandez et al., 2024; §3.1):
    - `Faithfulness`: after applying the learned affine transform to the subject’s hidden state, does the model predict the same object token it would have without the edit?
    - `Causality`: can editing a subject’s hidden state reliably switch the predicted object to a different, chosen object from the same relation (e.g., change “Miles Davis → trumpet” to predict “guitar”)? This tests whether the learned transform is not just mimicking the original prediction but can flexibly steer within the relation. The paper prefers `Causality` because early checkpoints can show spuriously high `Faithfulness` (§3.1).
  - Practical choices:
    - Per-relation scale factor `β` multiplies `W` (found beneficial in prior work); here, a separate `β` is tuned per relation rather than per model to avoid disadvantaging specific relations (§3.1; Appendix C).
    - Fitting on incorrectly answered examples also works: the model need not already “know” the relation to estimate the linear map (Appendix B; Figs. 4–5).

- Measuring frequency during pretraining
  - Frequency proxy: `subject–object co-occurrence`—count how often both terms appear together. Prior work shows co-occurrence is a strong proxy for the fact being mentioned in text (§3.2).
  - Challenge: need counts over time, aligned with the actual sequences the model trained on (not just whole documents).
  - Solution: `Batch Search` tool
    - Reconstruct OLMo training batches on Dolma and count co-occurrences inside each tokenized sequence (not entire documents), which better matches the LM’s actual pretraining signal (§3.2).
    - Large-scale implementation: ~10k terms across ≈2T tokens; 900 CPUs; ~1 day. Released as Cython bindings (§3.2).
  - When exact batches are unavailable (for GPT-J + Pile), use WIMBD document-level counts and validate that they closely match Batch Search at the end of training (Appendix D; Fig. 10 shows slope 0.94, r=0.99, with WIMBD slightly overcounting due to document vs. sequence granularity).

- Experimental models, data, and checkpoints
  - Models: `OLMo-7B (0424)`, `OLMo-1B (0724)` pretraining on `Dolma` (open batches and intermediate checkpoints); `GPT-J (6B)` pretraining on `The Pile` (§3; §4.1).
  - Relations dataset: 25 factual relations from Hernandez et al. (2024) (e.g., `country–capital`, `person–mother`), with 10,488 unique subjects/objects. LREs are fit per relation using 8 examples in 5-shot prompts (§3.1; Appx. B lists relations).
  - Checkpoints over time for OLMo: after {41B, 104B, 209B, 419B, 628B, 838B, 1T, 2T} tokens (§4.1).

- Predicting frequencies from representations
  - Task: regress to either object frequency or subject–object co-occurrence counts from features computed on the fully trained model (§5.1).
  - Features:
    - `LM-only` baseline: log probability of the correct answer under the model and few-shot accuracy (5 trials).
    - `LRE+LM`: adds `Faithfulness`, `Causality`, `Hard Causality` (top-1 version), and `Faith Prob` (log-probability of the correct answer under the LRE-edited hidden state) (§5.1).
  - Protocol: random forest regressors (100 trees), leave-one-relation-out (24 runs × 4 seeds), evaluate accuracy “within one order of magnitude” of ground truth and report log-space MAE (§5.1–§5.2).

## 4. Key Insights and Innovations
- Frequency thresholds predict linearity across training time (fundamental)
  - Across OLMo-7B, OLMo-1B, and GPT-J, relations achieve very high `Causality` (≥0.9 on average) once mean subject–object co-occurrences in pretraining exceed a model-specific threshold:
    - GPT-J: ≈1,097
    - OLMo-7B: ≈1,998
    - OLMo-1B: ≈4,447
    - Shown in Fig. 2 (“Co-Occurrence Threshold” table and dashed lines). Crucially, once a relation crosses this threshold, strong linearity appears even in early checkpoints (e.g., at 41B tokens), indicating dependence on cumulative exposure rather than model maturity (§4.2; Fig. 2).
- Co-occurrence frequency correlates more strongly with linearity than either subject or object frequency alone (incremental but clarifying)
  - `Causality` vs. co-occurrences: r = 0.82
  - `Causality` vs. subject frequency: r = 0.66
  - `Causality` vs. object frequency: r = 0.59
  - Reported for OLMo models (Fig. 2; §4.2). This highlights that learning the relation requires seeing subjects and objects together, not just individually.
- Linear structure predicts pretraining frequency—beyond probabilities (new capability)
  - Using `LRE+LM` features substantially improves frequency prediction over `LM-only` (probabilities and accuracy) when evaluated on held-out relations:
    - Object-frequency prediction: ≈70% “within one order of magnitude” vs ≈40% for LM-only (Fig. 3; §5.2), with log-MAE ≈2.1 (LRE+LM) vs ≈4.2 (LM-only).
  - The most important predictor is `Hard Causality` (≈15% drop when permuted), then faithfulness metrics (≈5% drop), indicating linear structure carries unique frequency signal (§5.2; Appendix E, Fig. 12).
- Cross-model generalization of frequency inference (new capability)
  - Train regression on OLMo, evaluate on GPT-J (and vice versa) without access to the target model’s training counts: LRE features outperform LM-only features in all cases (Table 1). For example, predicting object occurrences:
    - Eval on GPT-J: 0.65±0.12 (LRE) vs 0.42±0.10 (LM-only).
  - This suggests a consistent encoding of dataset statistics in linear relational structure across different models and corpora (§5.3; Table 1).

## 5. Experimental Analysis
- Evaluation setup
  - Relations, features, and metrics as in §3 and §5.1.
  - Models/datasets: OLMo-7B/1B on Dolma with batch-level counts and checkpoints; GPT-J on The Pile with WIMBD counts for final snapshot (§3; §4.1).
  - LRE fitting: 8 examples per relation, 5-shot prompts; per-relation β; subject edited at a layer chosen by a causality sweep. Hyperparameter scans over layers, β, and pseudoinverse rank are reported (Appendix C; Figs. 6–9).
- Main quantitative results
  - Formation of linearity vs. frequency
    - Quote:
      > “Co-occurrence frequencies highly correlate with causality (r = 0.82)… notably higher than the correlations with subject frequencies (r = 0.66) and object frequencies (r = 0.59)” (§4.2; Fig. 2).
    - Thresholds for near-perfect linearity (mean `Causality` > 0.9): 1,097 (GPT-J), 1,998 (OLMo-7B), 4,447 (OLMo-1B), shown in Fig. 2 (top-right table) and as dashed lines on scatterplots.
  - Linear structure across pretraining
    - Even at early checkpoints (e.g., 41B tokens in OLMo; red points in Fig. 2), relations that already exceed the co-occurrence threshold show high `Causality`. This supports the “frequency-first” interpretation (§4.2; Fig. 2).
  - Relation to task accuracy
    - Causality correlates with few-shot accuracy, but is not redundant:
      > In OLMo-7B, correlation of `Causality` with co-occurrence is 0.82 vs. 0.74 with 5-shot accuracy (§4.3).
      > Some relations are high-accuracy but low-linearity (e.g., `star–constellation`: 84% 5-shot vs 44% causality; average co-occurrence ≈21; §4.3).
    - Over training, causality sometimes rises before few-shot accuracy; by the final model, their gap averages ≤11% (Appendix F; Figs. 13–14).
  - Predicting pretraining frequencies from linearity (held-out relations)
    - Object frequencies: ≈70% within one order of magnitude for LRE+LM vs ≈40% for LM-only; log-MAE ≈2.1 (LRE+LM) vs ≈4.2 (LM-only) (Fig. 3; §5.2).
    - Subject–object co-occurrences: improvements exist but are smaller and closer to strong baselines (mean-prediction is competitive because counts are tightly clustered), indicating this is a harder target (§5.2; Fig. 3).
    - Feature importance: `Hard Causality` dominates, then faithfulness metrics; correlations among features are documented in Appendix E (Figs. 11–12).
  - Cross-model generalization (closed-data use case)
    - Table 1: LRE features generalize across OLMo ↔ GPT-J better than LM-only probabilities/accuracy. Example:
      > Predicting object occurrences when evaluating on GPT-J: 0.65±0.12 (LRE) vs 0.42±0.10 (LM-only).
    - The approach provides a new, representation-level signal to estimate training data statistics for models whose corpora are not accessible (§5.3; Table 1).
  - Error analysis and failure modes
    - Table 2 shows large overestimates for very low-frequency objects (e.g., `Arcturus → Boötes`: predicted 974,550 vs ground-truth 2,817 occurrences; 346× error). Some relations transfer poorly (e.g., `star–constellation`), and predictions can be sensitive to subject choice (`Prince William` vs `Prince Harry` both mapping to `Princess Diana`) (§5.4; Table 2).
- Ablations and robustness checks
  - Training LREs on incorrect examples: little difference vs. using only correct examples (Appendix B; Figs. 4–5).
  - Hyperparameter sweeps confirm stable layer, β, and rank choices; layer selected based on causality (Appendix C; Figs. 6–9).
  - Document-level vs. sequence-level counting: strong agreement (slope 0.94, r=0.99) with expected overcount when using documents (Appendix D; Fig. 10).
  - Commonsense relations (beyond factual triplets): correlation weakens (r ≈ 0.42) likely because co-occurrence is a noisier proxy for relation mentions (Appendix G; Fig. 15).

Overall assessment: The experiments convincingly support the main claim that subject–object co-occurrence frequency drives the emergence of linear relational structure, more so than marginal frequencies alone, and that linearity encodes dataset statistics useful for frequency inference. The cross-model generalization is particularly compelling for practical auditing. Subject–object frequency prediction remains challenging and some relations (e.g., astronomy) are outliers.

## 6. Limitations and Trade-offs
- Assumptions and proxies
  - Co-occurrence as a proxy for “the fact was mentioned” works best for factual relations; it is noisier for commonsense or definitional relations (§3.2; Appendix G).
  - Frequency thresholds are reported for three models and may depend on model size or architecture (§4.2 hints at scale dependence).
- Causality vs. causation
  - The study shows strong correlations, but cannot make causal claims about how exposure shapes representations without counterfactual pretraining experiments (Appendix A: Limitations).
- Counting constraints
  - Exact per-sequence counts across training rely on reconstructible batch pipelines; for models like GPT-J, the paper falls back to document-level WIMBD counts for the final snapshot (Appendix D).
- Prediction limits
  - Predicting subject–object co-occurrences is substantially harder than predicting object frequency; performance gains over baselines are modest (Fig. 3; §5.2).
  - Some relations transfer poorly (e.g., `star–constellation`), and predictions can be sensitive to subject choice (Table 2; §5.4).
- Scope
  - Focuses on factual recall relations; non-factual concepts (e.g., stereotypes) may not map cleanly to token-level co-occurrence counts (§3.1 note; §3.2).

## 7. Implications and Future Directions
- How this changes the landscape
  - Offers a concrete, measurable bridge between pretraining data statistics and the internal linear structure of LMs. This reframes when and why linear features emerge—from a vague “model capability” story to a data exposure threshold story (Fig. 2; §4.2).
  - Introduces a practical method to infer aspects of a closed model’s training data (frequency ranges) from its representations (Table 1; §5.3), complementing prior text-memorization or membership-inference approaches.
- Practical applications
  - Dataset curation: meeting frequency thresholds for desired relations can make capabilities more linear, and thus more editable/steerable (Intro; Discussion).
  - Auditing: estimate whether a model likely saw enough data about a domain, entity, or relation—useful for safety, fairness, domain coverage, or contamination checks (§5; Discussion).
  - Steering and interpretability: where linearity is strong, interventions (e.g., activation edits) should be more reliable (§2.1; Discussion).
- Research directions
  - Causal tests: controlled pretraining with manipulated frequencies to verify and refine thresholds (Appendix A).
  - Beyond factual relations: build better proxies than simple co-occurrence for commonsense or definitional relations (Appendix G).
  - Better frequency predictors: combine representation features with tokenizer or source-mixture signals (suggested in §6 Discussion; Hayase et al., 2024).
  - Nonlinear alternatives: investigate when models prefer non-linear solutions (e.g., in RNNs, Csordás et al., 2024) and whether similar data thresholds govern those cases (Discussion).
  - Mechanistic mapping: tie LRE layers and Jacobian structure to specific transformer subcircuits (Appendix C layer scans; Figs. 6–7).

> Central takeaway: “Linear representations form at predictable frequency thresholds during training, regardless of when in pretraining those tokens are encountered” (Fig. 2; §4.2). This both explains variation across relations and unlocks a new, representation-based window into models’ pretraining data.
