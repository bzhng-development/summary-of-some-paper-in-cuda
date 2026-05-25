# Undesirable Memorization in Large Language Models: A Survey

**ArXiv:** [2410.02650](https://arxiv.org/abs/2410.02650)

## 🎯 Pitch

This comprehensive survey delivers the first unified taxonomy of how and why large language models (LLMs) memorize training data, especially in ways that pose privacy, security, and copyright risks. By systematizing the literature across dimensions of granularity, retrievability, and desirability, and by analyzing how memorization is measured, influenced, and mitigated, it offers researchers and practitioners a roadmap to address the core driver behind critical LLM vulnerabilities. As LLMs increasingly permeate real-world applications, understanding and curbing undesirable memorization becomes essential for safe, ethical, and legally compliant AI deployment.

---

## 1. Executive Summary (2-3 sentences)
This survey organizes the rapidly growing literature on how large language models (LLMs) memorize training data—especially in ways that create privacy, security, and copyright risks—into a clear, three-dimensional taxonomy: granularity (what is memorized), retrievability (how it can be elicited), and desirability (whether we want it). It then synthesizes measurement techniques, influencing factors, and mitigation strategies, and proposes a concrete research agenda with underexplored contexts such as conversational agents, retrieval-augmented generation (RAG), multilingual models, and diffusion language models (Sections II–VI; Fig. 2; Fig. 3).

## 2. Context and Motivation
- Problem and gap addressed
  - LLMs can repeat training data verbatim or nearly verbatim, which risks leaking personally identifiable information (PII), secrets, and copyrighted text. The paper examines the phenomenon of memorization itself as the core driver of these risks, not just downstream attacks (Introduction; Section I).
  - Prior surveys either cover memorization broadly in ML/deep learning or treat memorization as one of several privacy/safety issues. This survey narrows to undesirable memorization in LLMs and provides a unified conceptual frame and actionable agenda (Section I-A; Table I).

- Why it matters
  - Real-world stakes:
    - Privacy/security: attack vectors can extract private training data (e.g., PII, API keys) and cause legal exposure (Section I; Section II-C-a).
    - Copyright/IP: reproducing protected text can infringe rights and affect AI governance and compliance (Section I; Section II-C-a).
  - Scientific stakes:
    - Understanding when/how memorization arises illuminates generalization dynamics, model scaling, and training processes (Sections IV, II-A-e).

- Prior approaches and their limits
  - Scattered definitions and metrics make it hard to compare findings across papers (Sections II–III).
  - Many evaluations use exact string match, which underestimates memorization and misses near-duplicates or entity-level leakage (Section III-A; Section II-A-c/d).
  - Defenses like de-duplication and differential privacy exist but involve performance trade-offs and do not directly address all forms of memorization (Section V).

- How this survey positions itself
  - Provides a 3D taxonomy that disentangles “what is memorized” (granularity) from “how it is elicited” (retrievability) and whether it’s “wanted or not” (desirability), resolving common conceptual conflations (Section II; Fig. 3; “Remark” in Section II-B).
  - Catalogs measurement methods beyond string match (exposure, inference attacks, counterfactuality, and adversarial prompt compression) and synthesizes consistent empirical patterns across model sizes, data duplication, prompting, tokenization, decoding, fine-tuning, training dynamics, and forgetting (Sections III–IV; Table II).
  - Articulates targeted, realistic mitigation options and a forward-looking research agenda tied to current deployment contexts (Sections V–VI).

## 3. Technical Approach
This is a survey; its “approach” is a structured synthesis built on a defined literature selection process and a unifying taxonomy.

- Literature curation
  - Data selection combined (1) a seed of 14 foundational works, (2) keyword searches on Google Scholar and arXiv, (3) manual filtering for relevance/quality, and (4) reference chaining to adjacent areas (e.g., training data extraction, membership inference) (Section I-B; Fig. 1).
  - Final set: 99 papers (Fig. 1).

- The taxonomy: three orthogonal dimensions (Section II; Fig. 3)
  1) Granularity (what is recalled)
     - `Perfect memorization`: outputs mirror the training data’s sequence frequency exactly—an idealized upper bound (Section II-A-a; Definition from [23]).
     - `Verbatim memorization` (also called “eidetic”): exact reproduction of a training string `s` from a prompt `p` where `LM(p) = s` (Section II-A-b; [13]); an “exact” variant requires `Argmax(LM(p)) = s` under greedy decoding (Section II-A-b; [25]).
     - `Approximate memorization`: generated text is similar to a training string (e.g., BLEU > 0.75; token-wise Levenshtein distance) to capture near-duplicates (Section II-A-c; [26], [28]).
     - `Entity-level memorization`: recalling an entity or relation when prompted with a subset of linked entities (e.g., outputting a phone number given a name) (Section II-A-d; [30], [31]).
     - `Content/knowledge memorization`: reproducing facts or concepts without the original phrasing; beneficial for knowledge tasks but risky if sources are biased/outdated (Section II-A-e).

  2) Retrievability (how it can be elicited)
     - `Extractable`: an attacker can craft a prompt that induces the model to output a training example without direct access to the training set (Section II-B-a; Definition from [39]).
     - `Discoverable`: when prompted with training prefixes, the model completes with the original suffix; a practical upper bound on extractability (Section II-B-b; [39], [40]). A `k-discoverable` variant constrains the prefix length to `k` tokens (Section II-B-b; [42]).

     Design choice explained: separating retrievability from granularity avoids mixing “what is memorized” with “how we measure it.” Any granularity can be either extractable or discoverable (Section II-B, Remark).

  3) Desirability (should we want this recall?)
     - `Undesirable`: privacy leaks, security risks, copyright violations, bias, inflated benchmark performance due to training-set overlap (Section II-C-a).
     - `Desirable`: knowledge retention, language fluency, and alignment behaviors (Section II-C-b).

- Measurement toolbox (Section III)
  - `String match`: prompt and compare outputs against training data; exact or approximate match. It yields a lower bound (finite sampling cannot cover all prompts) and is sensitive to small edits, hence approximate matching is used to widen coverage (Section III-A).
  - `Exposure` metric and canaries: insert a known “canary” string and compute its exposure `exposure_θ(s[r]) = log2|R| − log2 rank_θ(s[r])`, where `rank` is the model’s preference ordering (Section III-B; Eq. (1); [24], [47]).
  - `Inference attacks`: 
    - Membership inference attacks (MIAs) estimate if a text was in training (often with perplexity) (Section III-C-a).
    - Extraction attacks craft prompts to recover sensitive content (Section III-C-b).
  - `Counterfactual memorization`: measure how including/excluding an example from training changes performance on that example across multiple re-trainings—closely related in spirit to differential privacy (Section III-D; [71]).
  - `Prompt compression (ACR)`: assess if a training string can be elicited by an unusually short adversarial prompt using gradient-based prompt search (GCG), conceptually related to Kolmogorov complexity (Section III-E; [72], [73]).

- Synthesis of influencing factors (Section IV; Table II)
  - Model capacity: memorization increases log-linearly with size and appears earlier in training for larger models (Section IV-A; [25], [39], [74]).
  - Data characteristics: duplication and low-complexity (compressible) strings amplify recall; nouns/numbers are memorized faster (Section IV-B; [23], [25], [28], [77], [78]).
  - Input/prompting: longer prompts, prompt-tuning, and “according to” prompts elevate recall (Section IV-C; [13], [23], [79]–[84]).
  - Tokenization: larger BPE vocabularies shorten sequences and increase memorization odds (Section IV-D; [85]).
  - Decoding: greedy decoding is often most effective at revealing memorization; tuned top-k/p/temperature can further optimize for extraction (Section IV-E; [13], [86]).
  - Fine-tuning: head vs adapter vs full fine-tuning have different leakage profiles; task choice matters (Section IV-F; [90], [91]).
  - Training dynamics: memorization rises with epochs; earlier examples can be forgotten; mid-training shows lower memorization rates (Section IV-G; [23], [71], [93], [94]).
  - Forgetting mechanisms: forgetting curves can be exponential; memorized content is robust and hard to excise (Section IV-H; [25], [93], [99]–[101], [104], [105]).
  - Interpretability: evidence points to early layers storing recall cues and upper layers amplifying them; memorization is distributed and task-dependent (Section IV-I; [101]–[105]).

- Mitigation catalog (Section V)
  - `Data de-duplication`: exact and near-duplicate removal; can reduce memorization by ~10× in unprompted settings but duplicates in prompts still elicit recall (Section V-A; [77]).
  - `Differential privacy (DP-SGD)`: provably reduces exposure but with performance/compute trade-offs; selective DP and careful hyperparameters can help (Section V-B; [24], [69], [120]–[123]).
  - `Unlearning`: remove specific content post hoc via sharded retraining (SISA), light unlearning layers, in-context unlearning, and RL-based dememorization; effective but relies on knowing what to remove (Section V-C; [124]–[129]).
  - `Heuristic methods`: alternating teaching (teacher-student), decoding constraints (MemFree), tuned nucleus sampling, and modified loss like “goldfish loss” to reduce complete sequence memorization (Section V-D; [26], [107], [130], [135]).

## 4. Key Insights and Innovations
- A unifying 3D taxonomy that resolves definitional ambiguity (Section II; Fig. 3)
  - Novelty: It separates “what is memorized” (granularity) from “how it is elicited” (retrievability) and “whether we want it” (desirability), clarifying that any granularity (e.g., entity-level) can be either extractable or discoverable. Prior literature often conflated these dimensions.
  - Significance: Enables apples-to-apples comparison across studies and guides measurement/mitigation choices to the right sub-problem.

- Consolidated, multi-method measurement toolkit with practical trade-offs (Section III)
  - Novelty: Goes beyond exact-match to include exposure/canaries, counterfactuality, MIAs, and adversarial prompt compression (`ACR`), each illuminating different facets of memorization.
  - Significance: Helps practitioners select robust, complementary diagnostics instead of relying on a single, biased estimate (e.g., exact match lower bounds).

- Evidence-backed map of influencing factors and training dynamics (Section IV; Table II)
  - Novelty: Aggregates consistent patterns—memorization scales predictably with size and duplication; prompts/tokenization/decoding materially affect elicitation; memorization and forgetting have distinct temporal signatures.
  - Significance: Turns anecdotal findings into actionable levers for risk assessment and engineering, e.g., deduplication and prefix control.

- Practical mitigation playbook with limits made explicit (Section V)
  - Novelty: Juxtaposes efficient, general defenses (deduplication, DP) with targeted, surgical ones (unlearning, decoding heuristics), and highlights when they fail or trade off performance.
  - Significance: Clarifies that no single defense suffices; combinations must be tailored to threat models and deployment contexts.

- Concrete future agenda anchored in current deployment modalities (Section VI)
  - Novelty: Identifies underexplored but high-impact contexts—conversational agents, RAG, multilingual LLMs, diffusion language models—where memorization risk may differ materially.
  - Significance: Directs research to where real-world deployments are accelerating and where risk-benefit trade-offs are most acute.

## 5. Experimental Analysis
This is a survey; it synthesizes results across many empirical papers. The survey collates evaluation setups and headline numbers, with pointers to methodologies.

- Evaluation methodology landscape (Sections II–III; Table III)
  - Common model families: `Pythia` and `GPT-Neo`, with checkpoints across training for dynamic analyses (Table III).
  - Common datasets: `PILE` predominates; others include `C4`, `OpenWebText`, `Wiki40B`, code corpora, and task-specific corpora (Table III; Sections IV-F, V-A).
  - Prompt and decoding setups: many studies use 32–50 token prefixes and greedy decoding to maximize recall; others sweep top-k/p and temperature to optimize extraction (Table III; Section IV-E).
  - Metrics:
    - Exact/approximate string match (50-token suffix common) (Table III; Section III-A).
    - Exposure via canaries (Section III-B; Eq. (1)).
    - MIA/extraction attack rates (Section III-C).
    - Counterfactuality across multiple retrainings (Section III-D).
    - `ACR` via adversarial prompt compression (Section III-E).

- Key quantitative findings (as reported in cited works)
  - Upper bound via discoverability:
    > “LLMs discoverably memorize roughly 1% of their training datasets” when prompted with ~50-token contexts; this scales log-linearly with model size and data repetition (Section II-B-b; [39]). Similar rates reported for `PaLM` and `MADLAD-400` (Section II-B-b; [43], [44]).
  - Black-box extractability:
    > Querying with millions of short contexts yields “0.1% to 1% of outputs [that] are memorized,” increasing with model size (Section II-B-a; [40]).
  - Effect of data duplication:
    > A sequence repeated 10× in training is generated “~1000× more frequently” than a once-seen sequence (Section IV-B; [23]).
    > Training on deduplicated `C4` reduces unprompted memorization “10×,” though prompts drawn from duplicated examples still elicit true continuations at high rates (Section V-A; [77]).
  - Prompt/tokenization/decoding sensitivity:
    - Longer prefixes and prompt-tuning increase elicitation; larger BPE vocabularies shorten sequences and heighten verbatim recall; greedy decoding is the most revealing, while tuned sampling can further optimize extraction (Sections IV-C/D/E; Table III; [13], [81], [82], [85], [86]).
  - Training dynamics and forgetting:
    - Memorization rises with epochs; earlier examples are more likely forgotten; forgetting curves can be exponential (Sections IV-G/H; [23], [25], [93], [94]).

- Do the experiments support the survey’s synthesized claims?
  - Yes, multiple independent studies converge on consistent patterns: scaling with size and duplication, sensitivity to prefix length and tokenizer, and the ≈1% discoverable upper bound with ≈50-token contexts (Sections II–IV). The survey makes clear where results are lower bounds (finite prompt sampling) and where metrics capture different facets (exposure vs. string match vs. counterfactuality) (Sections III-A/B/D).

- Ablations, failure cases, robustness checks
  - Robustness to prompt style: “MemFree” decoding reduces verbatim overlap but can be bypassed via simple style transfer in prompts (Section V-D; [26]).
  - Unlearning fragility: stress tests show that many unlearning methods remove targeted strings but degrade model utility or leave distributed traces that still enable recall (Section IV-I, V-C; [101]).
  - Discoverable vs extractable gap: sequences can be discoverable but not practically extractable and vice versa; overlap is informative but incomplete (Section II-B-c; [40] vs [39]).

- Conditional results and trade-offs
  - DP-SGD reduces exposure but can hurt utility; “selective DP” and careful hyperparameters can partially bridge the trade-off (Section V-B; [24], [121]).
  - Deduplication reduces unprompted recall strongly but does not fully prevent elicitation when adversaries reuse duplicated prefixes (Section V-A; [77]).

## 6. Limitations and Trade-offs
- Assumptions and scope
  - The survey focuses on “undesirable” memorization but necessarily discusses beneficial memorization (facts/knowledge) to draw clear boundaries (Section II-C; Section VI-B).
  - Many synthesized results derive from open-source model families (`Pythia`, `GPT-Neo`) and datasets (`PILE`), which may not fully represent proprietary frontier models (Table III).

- Measurement constraints
  - String-match estimates are lower bounds due to finite sampling and verification challenges; approximate matching can introduce false positives/negatives (Section III-A; Section II-A-c).
  - Exposure/canary tests operationalize risk but rely on synthetic insertions rather than naturally occurring secrets (Section III-B).
  - Counterfactuality requires many re-trainings and may not scale to frontier models (Section III-D).

- Mitigation trade-offs
  - Differential privacy: provable but can degrade performance and increase compute; poor hyperparameters can fail under strong attacks (Section V-B; [69], [121]).
  - Unlearning: effective for known targets, but requires identifying what to remove; removing distributed traces without utility loss remains difficult (Sections IV-I, V-C; [101], [125]–[129]).
  - Heuristic decoding/training tweaks: reduce extractable verbatim content but are bypassable or only partially effective (Section V-D; [26], [107], [135]).
  - Deduplication: improves both privacy and quality but requires heavy preprocessing and still leaves residual risk via duplicated prompts (Section V-A; [77]).

- Open questions
  - Where precisely is the boundary between “understanding” and “memorization,” and how do we measure it reliably across tasks? (Section VI-C).
  - How do memorization dynamics change in emerging architectures like diffusion language models (text) vs. image diffusion? (Section VI-D-d).
  - How should granular privacy guarantees (entity, sentence, document) be defined and enforced in practice? (Section VI-A.1).

## 7. Implications and Future Directions
- How this work changes the landscape
  - Provides a shared language and framework (granularity × retrievability × desirability) for discussing LLM memorization risks and defenses (Section II; Fig. 3).
  - Equips practitioners with a richer diagnostic kit and a realistic defense menu, highlighting where trade-offs bite (Sections III–V).
  - Anchors future work in concrete deployment contexts where risk and utility tensions are sharpest (Section VI-D).

- Follow-up research enabled or suggested (Section VI)
  - Balancing privacy and performance:
    - Unify counterfactual memorization with differential privacy at multiple granularities (entity/sentence/document/corpus) and develop adaptive privacy controls (Section VI-A.1).
  - Shifting from verbatim to content memorization:
    - Methods that discourage exact reproduction while preserving factual accuracy, with new metrics to evaluate the trade-off and hallucination risks (Section VI-B).
  - Understanding vs memorization:
    - New evaluations that test abstraction and compositionality while controlling for training-set overlap, to draw principled boundaries (Section VI-C).
  - Context-specific studies:
    - Conversational agents: stronger black-box extraction frameworks that respect alignment layers and dialogue controls (Section VI-D-a).
    - RAG: joint analysis of hallucination reduction vs. increased proximity to sources; privacy leakage from retrieved corpora; architecture-level comparisons (Section VI-D-b).
    - Multilingual LLMs: measure whether low-resource settings are more prone to memorization and PII leakage; fairness and safety implications (Section VI-D-c).
    - Diffusion language models: text-specific memorization theory and measurement distinct from vision, accounting for discrete token spaces (Section VI-D-d).

- Practical applications and downstream use
  - For builders and deployers:
    - Integrate deduplication into data pipelines; monitor exposure with canaries; adopt selective DP for sensitive fine-tuning; consider “goldfish loss” or decoding constraints in high-risk settings (Sections V-A/B/D; [135]).
    - Use prompt controls (prefix management, instruction formats) and tokenizer choices mindful of their impact on elicitation (Sections IV-C/D/E).
    - Establish unlearning workflows (SISA-style sharding, light unlearning layers, in-context unlearning) for data subject requests and compliance (Section V-C).
  - For policy and compliance:
    - Use the taxonomy to operationalize “undesirable memorization” in audits; pair discoverability tests (upper bounds) with extractability tests (realistic risk) for due diligence (Sections II-B, III).

Block-quoted, paper-grounded highlights:
- Discoverable upper bound:
  > “LLMs discoverably memorize roughly 1% of their training datasets,” scaling log-linearly with size, repetition, and prefix context length (Section II-B-b; [39]; corroborated by [43], [44]).
- Black-box extraction at scale:
  > “0.1% to 1% of outputs are memorized,” with a strong correlation to model size (Section II-B-a; [40]).
- Duplication effect:
  > “A sequence appearing 10 times in the training data is … generated approximately 1000 times more frequently than a sequence that appears only once” (Section IV-B; [23]).
- Deduplication payoff:
  > Deduplicated models show “10× less memorization” in no-prompt settings; duplicated prompts can still elicit ground-truth continuations (Section V-A; [77]).

Pointers to figures and tables:
- Fig. 1: Literature selection funnel and final count (99 papers) (Section I-B).
- Fig. 2: End-to-end map of taxonomy, measurement, influences, mitigation (a roadmap for readers new to the area).
- Fig. 3: The 3D “memorization cube” (granularity × retrievability × desirability) (Section II).
- Table I: How this survey’s scope differs from prior surveys (Section I-A).
- Table II: One-page summary of influencing factors with key findings (Section IV).
- Table III: Side-by-side of memorization experimental setups across studies—models, datasets, decoding, prompt/match lengths, and which taxonomy cells they evaluate (Sections II–IV).
