# Undesirable Memorization in Large Language Models: A Survey

**ArXiv:** [2410.02650](https://arxiv.org/abs/2410.02650)
**Authors:** Ali Satvaty, Suzan Verberne, Fatih Turkmen
**Institutions:** 

## 🎯 Pitch

This paper provides a comprehensive framework for understanding and addressing "memorization" in large language models (LLMs), unifying definitions and methodologies to assess and mitigate the risks of data leakage. By mapping the causal factors and outlining strategies, it enables more effective safeguarding of privacy and intellectual property, crucial for balancing model performance with legal and ethical obligations.

---

## 1. Executive Summary
This survey systematizes what “memorization” means for large language models (LLMs), how it arises, how to detect it, and how to mitigate it. It provides a unified taxonomy (Figure 1), synthesizes empirical findings across training stages and decoding methods, and outlines concrete open questions that connect technical practice with privacy and legal risk (Sections 2–7).

## 2. Context and Motivation
- Problem and gap
  - LLMs sometimes reproduce training data verbatim or near-verbatim, creating risks such as leaking personally identifiable information (PII) or copyrighted text (Abstract; Section 1).
  - Evidence and methods to study memorization are scattered across papers with inconsistent definitions and measurement practices (Sections 2 and 5).
  - There is no single, reliable way to audit memorization without access to the training set, and existing defenses involve utility and scalability trade-offs (Sections 5–6).

- Why it matters
  - Real-world impact: risks to privacy (e.g., PII leakage), intellectual property, and regulatory exposure (Section 7).
  - Scientific significance: clarifying when models generalize vs. memorize, and how scaling laws and training dynamics influence this boundary (Sections 3–4).

- Prior approaches and their limits
  - Prior surveys often treated memorization as one item within broader safety or privacy overviews (Section 1).
  - Empirical work shows memorization depends on model size, data duplication, sequence length, tokenization, and decoding choices but lacked a cohesive map of mechanisms and practical trade-offs (Section 3).
  - Detection techniques (e.g., prefix-based extraction, membership inference) exist but can fail without training-data access, or are statistically ill-posed for instance-level claims (Section 5).

- Positioning
  - The survey focuses narrowly on memorization across: definitions (Section 2), causal factors and training-stage dynamics (Sections 3–4), detection and measurement (Section 5), mitigation (Section 6), and legal/privacy implications (Section 7). Figure 1 presents the overall taxonomy.

## 3. Technical Approach
The paper is a structured synthesis, not a new algorithm. Its “approach” is a taxonomy plus mechanism-level explanations that make disparate results comparable.

- Taxonomy (Figure 1)
  - “Defining Memorization”: forms and tests of memorization (Section 2).
  - “Factors Influencing Memorization”: what makes memorization more or less likely (Section 3).
  - “Memorization at Different Stages”: pre-training vs. fine-tuning vs. RL/post-training dynamics (Section 4).
  - “Detecting Memorization” and “Mitigating Memorization”: attack/measurement vs. defense knobs (Sections 5–6).
  - “Privacy & Legal Risks”: mapping technical leakage to real-world harms and legal theories (Section 7).

- Definitions and tests (Section 2)
  - `Verbatim (exact) memorization`: the model reproduces a string from its training data word-for-word (Section 2.1).
  - `Perfect memorization`: an extreme formalization where the model assigns probability only to training inputs (Section 2.1).
  - `Eidetic` and `k-eidetic` memorization: the model reproduces a training suffix `s` given a prompt `p`, even if `s` appears ≤ `k` times in training (Section 2.1).
  - `Discoverable memorization`: given a training pair (`p`, `s`), the model completes `s` exactly from `p` (Section 2.1).
  - `Approximate memorization`: near matches measured by normalized edit distance rather than exact string equality (Section 2.2).
  - `Prompt-based extractable` and `k-extractable` memorization: the model reproduces a training example when prompted, sometimes from only `k` prefix tokens (Section 2.3).
  - `(n, p)-discoverable extraction`: with repeated sampling, the string appears in at least one of `n` completions with probability ≥ `p` (Section 2.3).
  - `Influence-based (counterfactual) memorization`: a data point is “memorized” if training on it vs. leaving it out materially changes predictions or loss (Section 2.4). Because retraining per point is intractable, studies exclude subsets and compare multiple models (Section 2.4).

- Mechanisms and why they matter (Sections 3–4)
  - Capacity and scale: larger models memorize more and do so faster because they can fit long-tail details (Section 3).
  - Data duplication: repeated examples amplify gradient signal and increase likelihood of verbatim recall; deduplication reduces this (Section 3).
  - Sequence length and tokenization: longer contexts and tokenization choices (e.g., larger BPE vocabulary that turns rare entities into single tokens) change how easily specific sequences are recalled (Section 3).
  - Decoding parameters: stochastic sampling (temperature, top-k, nucleus) exposes memorization hidden by greedy decoding (Section 3).
  - Training-stage dynamics: “parameter drift” causes early examples to be overwritten unless revisited; late-stage training and fine-tuning can re-activate latent memories (Section 4).

- Detection tools (how they work) (Section 5)
  - `Divergence attack`: prompting that pushes the model toward pre-alignment behavior (akin to end-of-text contexts in pretraining), dramatically increasing verbatim leakage (Section 5).
  - `Prefix-based extraction`: seed the model with a training prefix and test whether it autocompletes the rest exactly or approximately (Section 5).
  - `Membership inference attacks (MIA)`: infer whether a specific example was in the training set using per-example loss, calibrated loss (reference model), compression-adjusted loss (zlib entropy), neighborhood comparisons, or “min-k%” token probabilities (Section 5). The survey explains why instance-level MIAs lack well-calibrated null models and urges aggregate use (Section 5).
  - `Soft prompting`: learned continuous prompt embeddings that amplify or suppress leakage; dynamic variants condition the soft prompt on the input prefix to surface context-dependent memories (Section 5).

- Mitigation tools (how they work) (Section 6)
  - Training-time:
    - `Data cleaning`: deduplication and PII scrubbing remove high-risk content before training (Section 6.1).
    - `Differential privacy (DP)` and `DP-SGD`: clip per-example gradients and add calibrated noise; track a privacy budget; reduces extraction and MIA success (Section 6.1). The survey also covers user-level DP and parameter-efficient fine-tuning (PEFT) with DP (Section 6.1).
  - Post-training:
    - `Machine unlearning`: edit or retrain parts of the model to reduce influence from specific data; efficient but lacks formal guarantees (Section 6.2).
    - `ParaPO`: identify memorized sequences, generate summaries, and run preference optimization (DPO) to prefer summaries over verbatim reproductions; reduces unintended verbatim recall with small utility costs (Section 6.2).
  - Inference-time:
    - `MemFree decoding`: use a bloom filter of training n-grams to screen out verbatim n-grams as the model generates; needs training n-grams and misses near-duplicates (Section 6.3).
    - `Activation steering`: manipulate hidden activations (often layer- and head-specific) identified as memory-related, to suppress recall while preserving task performance; pruning-based localization can find small sets of memory-critical units (Section 6.3).

- Legal/privacy framing (Section 7)
  - Maps technical leakage to personal data, copyrighted/proprietary content, and ongoing litigation, highlighting business and policy implications.

## 4. Key Insights and Innovations
- Unifying definitions and tests of memorization (Section 2; Figure 1)
  - Significance: brings multiple, incompatible notions—exact, prompt-based, discoverable, approximate, counterfactual—under a single vocabulary with concrete operationalizations (e.g., prefix-based, (n, p)-discoverability). This clarity enables apples-to-apples evaluation and targeted defenses.

- Mechanistic picture across scales, data, and decoding (Section 3)
  - Novelty in synthesis: the survey connects scaling (log-linear with size), duplication (superlinear effects), sequence length (log increase), tokenization (BPE vocabulary size), and sampling (stochastic decoding) into a coherent causal story of when and why LLMs memorize.

- Training-stage dynamics and persistence (Section 4)
  - Insight: “parameter drift” explains forgetting early examples and the higher susceptibility of late-stage checkpoints; memorization follows predictable scaling laws and can persist through RLHF; fine-tuning can both reduce or amplify leakage depending on method and task attention patterns.

- Measurement realism and limits of MIAs (Section 5)
  - Critical contribution: distinguishes detection methods that require training data (e.g., prefix extraction) from those that do not (MIAs), and explains why instance-level MIAs are statistically unsound without a proper null, recommending aggregate auditing and canary-based tests.

- End-to-end mitigation palette with trade-offs (Section 6)
  - Integration: lays out choices across the lifecycle—data cleaning, DP/PEFT, unlearning/ParaPO, MemFree, activation steering—and highlights when each is practical, what it buys, and what it costs. This is valuable for practitioners building privacy-aware pipelines.

## 5. Experimental Analysis
While the survey does not run new experiments, it curates quantitative evidence and conditions under which specific phenomena appear. Key numbers and setups are cited to sections.

- Drivers of memorization (Section 3)
  - Model size: memorization “scales log-linearly with model size,” and larger LLMs “memorize more rapidly during training,” increasing vulnerability to extraction (Section 3).
  - Data duplication: deduplication substantially reduces recall; one study reports that models trained on original, non-deduplicated data produce “a tenfold increase in memorized token generation” vs. deduplicated training (Section 3).
  - Sequence length: “memorization increases logarithmically with sequence length,” with verbatim reproduction rising by orders of magnitude from 50 to 950 tokens (Section 3).
  - Tokenization: larger BPE vocabularies correlate with more memorization, especially for named entities and rare phrases that become single tokens (Section 3).
  - Decoding: randomized sampling exposes more leakage than greedy decoding; the survey quotes two findings:
    > “optimizing sampling parameters… can substantially increase memorized data extraction, in some cases doubling previous baselines” (Section 3).
    > “randomized decoding nearly doubles leakage risk compared to greedy decoding” (Section 3).

- Detection effectiveness (Section 5)
  - Divergence attack:
    > “achieving up to 150× more verbatim sequences compared to typical user queries” by steering the model toward pre-alignment behavior (Section 5).
  - Prefix-based extraction:
    - Longer prefixes increase exact continuation rates (Section 5).
    - Structured prefixes (e.g., email headers) are especially potent (Section 5).
    - Adversarial prefix generation can trigger leakage even off-distribution, showing memorization can be more pervasive than vanilla prefix attacks reveal (Section 5).
  - MIAs:
    - Techniques include loss, reference-model-calibrated loss, zlib entropy, neighborhood attack, and min-k% probabilities (Section 5).
    - The survey cautions that per-instance MIAs lack calibrated nulls and may be unreliable, recommending aggregate auditing or canary-based MIAs (Section 5).
  - Soft prompting:
    > Attack prompts “increased memorization leakage by up to 9.3%,” while suppression prompts “decreased extraction by up to 97.7%” (Section 5).
    - Dynamic soft prompts conditioned on the input prefix “significantly improved the discoverable memorization rate” over static prompts (Section 5).

- Mitigation outcomes (Section 6)
  - Differential privacy:
    - DP-SGD with pretrained LLMs can approach non-private utility while reducing leakage, validated via canary insertion and MIAs (Section 6.1).
    - User-level DP lowers canary extraction more than record-level DP (Section 6.1).
    - PEFT with DP can match or exceed full-model DP-SGD utility at tight privacy budgets, though concentrating noise in fewer parameters may weaken guarantees (Section 6.1).
  - Post-training:
    - Machine unlearning can be “over 10^5× more computationally efficient than retraining,” but lacks formal guarantees (Section 6.2).
    - ParaPO:
      > Decreases unintended memorization while “slightly” degrading math/knowledge/reasoning performance; preserves desired verbatim recall like quotations (Section 6.2).
  - Inference-time:
    - Activation steering:
      > Reduced memorization “by up to 60%” with minimal performance drop, but depends on layer selection and steering strength (Section 6.3).
    - Localization and pruning:
      > Removing fewer than “0.5% of neurons” (Hard Concrete) led to a “60% drop in memorization accuracy,” but neurons often serve multiple memories (risk of collateral forgetting) (Section 6.3).
    - MemFree decoding works only with access to the training n-gram set and misses near-duplicates (Section 6.3).

- Are the claims supported?
  - The survey consistently links effects to concrete mechanisms (e.g., duplication → gradient reinforcement; BPE vocabularies → single-token rare entities; decoding diversity → revealing hidden high-probability strings), and reports multiple corroborating sources per claim (Sections 3–6).
  - Where evidence is mixed or conditional, it is flagged explicitly: e.g., MIAs are useful for aggregate auditing but unreliable for per-instance determinations (Section 5); PEFT-with-DP may concentrate noise (Section 6.1); steering strength/layer selection are sensitive (Section 6.3).

## 6. Limitations and Trade-offs
- Assumptions and scope
  - Survey scope: no new experiments; the reliability of conclusions depends on the quality and comparability of underlying studies (implicit across Sections 3–6).
  - Definitions vary: different papers operationalize “memorization” differently (exact/approximate/discoverable/counterfactual), which can change measured rates (Section 2).

- What is not addressed or remains uncertain
  - Modality sensitivity: it remains unclear whether duplication and memorization scale similarly across prose, code, math, etc. (Section 3, “Open Questions”).
  - Decoding for privacy: there is no decoding strategy that consistently minimizes leakage across settings; crafting such a strategy is open (Section 3).
  - Training hyperparameters and temporal order: the roles of learning rate, regularization, and the order in which duplicates are seen are underexplored (Section 3, “Open Questions”).
  - Stage attribution: quantifying precisely how much memorization arises in pre-training vs. fine-tuning vs. RL remains open (Section 4).

- Cost and scalability constraints
  - DP-SGD imposes computational overhead and can reduce utility under strict privacy budgets (Section 6.1).
  - MemFree needs access to training n-grams; misses paraphrases (Section 6.3).
  - Activation steering and localization require white-box access, layer/head selection, and can cause collateral forgetting (Section 6.3).
  - Machine unlearning is efficient but lacks formal guarantees, leaving residual risk (Section 6.2).

- Open technical tensions
  - Utility vs. privacy: ParaPO slightly reduces reasoning/knowledge scores; DP budgets trade utility for privacy (Section 6.1–6.2).
  - Generalization vs. memorization: the survey asks how to separate useful skill acquisition from harmful rote recall in larger models (Section 3, “Open Questions”).

## 7. Implications and Future Directions
- How this work changes the landscape
  - Provides a common language and checklist for practitioners: definitions to test, factors to audit (size, duplication, tokenization, decoding), stage-aware risks, and a menu of mitigations with known trade-offs (Figure 1; Sections 2–6).
  - Re-frames MIAs: use them for aggregate auditing, not instance-level claims (Section 5). This helps teams avoid overclaiming and design better privacy evaluations.

- Follow-up research enabled or suggested
  - Detection without training data:
    - Build standardized, training-data-agnostic audits that go beyond MIAs (Section 5, “Open Questions”).
  - Privacy-aware decoding:
    - Design decoding strategies explicitly optimizing for low leakage across distributions, with proofs or empirics that hold beyond greedy vs. random contrasts (Section 3).
  - Stage-aware mitigation:
    - Quantify and predict which data persist through fine-tuning and RL; craft objectives that selectively reduce harmful memorizations while preserving beneficial recall (Section 4).
  - DP at scale and with PEFT:
    - Clarify when PEFT-with-DP maintains end-to-end user-level privacy; develop accountants and noise-placement strategies that avoid privacy “hot spots” (Section 6.1).
  - Mechanistic control:
    - Formalize layer/neuronal selection for activation steering; automate steering-strength tuning with minimal collateral forgetting (Section 6.3).
  - Reasoning models:
    - Benchmarks to distinguish memorized superficial solution patterns from genuine reasoning competence (Section 5, “Reasoning”).

- Practical applications and downstream use
  - Training pipelines:
    - Default to deduplication and PII scrubbing; consider user-level DP when feasible; monitor with canaries and aggregate MIAs (Sections 6.1 and 5).
  - Model release and serving:
    - Use MemFree or similar filters when training data is known; consider activation steering for high-risk deployments with white-box access (Section 6.3).
  - Legal and compliance:
    - Link memorization metrics (e.g., rates of verbatim reproduction under standardized prompts) to risk assessments for PII and copyright; track high-risk content types (entities, URLs, rare phrases) especially under large BPE vocabularies (Sections 3 and 7).

Block-quoted highlights for quick recall
- “up to 150× more verbatim sequences” with divergence attacks (Section 5).
- “tenfold increase in memorized token generation” without deduplication (Section 3).
- Randomized decoding “nearly doubles leakage risk” compared to greedy decoding (Section 3).
- Soft prompts: +9.3% leakage (attack) vs. −97.7% extraction (suppress) (Section 5).
- Activation steering: up to 60% memorization reduction; pruning <0.5% neurons → 60% drop in memorization accuracy (Section 6.3).

Where to look in the paper
- Figure 1: the overall taxonomy tying together definitions, drivers, stages, detection, mitigation, and risks.
- Section 2: precise types of memorization and how to test for them.
- Section 3: factors that increase or decrease memorization and the role of decoding.
- Section 4: how memorization evolves across pre-training, fine-tuning, RL, and distillation.
- Section 5: detection methods, their mechanisms, and when they fail.
- Section 6: mitigation across training, post-training, and inference, with concrete trade-offs.
- Section 7: privacy and legal implications, including live case themes relevant to practitioners.
