# The Llama 3 Herd of Models

**ArXiv:** [2407.21783](https://arxiv.org/abs/2407.21783)

## 🎯 Pitch

Llama 3 introduces a state-of-the-art open family of foundation language models at three parameter scales—8B, 70B, and a flagship 405B dense Transformer—delivering robust multilingual, coding, reasoning, and tool-use abilities, all within a 128K-token context window. By leveraging massive, rigorously curated data and a pragmatic, scalable training pipeline, Llama 3 not only matches leading closed models like GPT-4 in quality across diverse tasks, but also pioneers a modular approach to integrating vision, video, and speech, ushering in a new era of accessible, high-performance AI for research and real-world applications.

---

## 1. Executive Summary

This paper presents the Llama 3 Herd of Models, a new set of foundation language models that natively support multilinguality, coding, reasoning, and tool usage, culminating in a flagship dense Transformer with 405B parameters and a 128K-token context window pre-trained on 15.6T tokens using 3.8 × 10²⁵ FLOPs. The development emphasizes three levers — **data** (improved curation, de-duplication, and quality filtering across a 15T-token multilingual corpus), **scale** (nearly 50× the pre-training compute of Llama 2, with smaller models trained far beyond compute-optimality to trade training compute for inference efficiency), and **managing complexity** (a standard dense Transformer architecture with minor adaptations such as grouped query attention and a 128K-token vocabulary, paired with a post-training procedure based on supervised finetuning, rejection sampling, and Direct Preference Optimization rather than more complex reinforcement learning algorithms). The flagship 405B model performs on par with GPT-4 across a wide range of benchmarks — achieving 87.3% on MMLU, 89.0% on HumanEval, and 96.8% on GSM8K — while the smaller 8B and 70B models are best-in-class for their respective size categories, establishing that dense architectures trained on carefully curated data can match or exceed state-of-the-art closed-source alternatives when paired with iterative rounds of rejection sampling, DPO, and model averaging. Compositional experiments integrating image, video, and speech capabilities via separate encoder and adapter training demonstrate competitive performance with leading multimodal models, though these multimodal extensions are not yet released.

## 2. Context and Motivation

### The Core Problem: How to Build a State-of-the-Art Openly-Available Foundation Model

The central challenge this paper tackles is deceptively straightforward: **how do you build, from scratch, a family of openly-available foundation language models that can compete with — or surpass — the best closed-source commercial systems?** This is not merely an academic exercise. The paper emerges at a critical juncture in the AI industry where the most capable language models (GPT-4, Claude, Gemini) are deployed behind proprietary APIs, creating a knowledge and capability asymmetry. Researchers outside of a handful of organizations cannot inspect these models' architectures, understand their training data, reproduce their results, or adapt them for specialized applications. The Llama 3 project represents a direct response to this asymmetry: a bet that openness and transparency can produce models competitive with the state-of-the-art while simultaneously enabling a broader ecosystem of research, fine-tuning, and safety analysis.

The significance of this problem crystallizes around three axes:

- **Scientific reproducibility.** Without access to model weights, training recipes, and data compositions, the research community operates in the dark. Claims about emergent capabilities, scaling laws, and alignment techniques cannot be independently verified. The paper explicitly positions public release as a mechanism for scientific scrutiny: "We hope that the open release of a flagship model will spur a wave of innovation in the research community, and accelerate a responsible path towards the development of artificial general intelligence (AGI)." This is not boilerplate — it reflects a concrete belief that open models enable independent safety audits, adversarial testing, and capability assessments that are impossible with API-only access.

- **Downstream innovation and deployment flexibility.** Closed-source models constrain what developers can build. They cannot be fine-tuned on proprietary data, cannot be deployed on custom hardware, cannot be modified for domain-specific tasks, and are subject to the API provider's pricing, rate limits, and terms of service changes. A 405B openly-available model — even one requiring substantial hardware — enables on-premise deployment, domain adaptation, and integration into systems where data privacy precludes API usage. The smaller 8B and 70B variants extend this further: they can run on consumer hardware or edge devices, dramatically expanding the surface area of possible applications.

- **Economic and geopolitical considerations.** The concentration of frontier AI capabilities in a small number of organizations raises concerns about market power, access equity, and the geographic distribution of AI benefits. By releasing weights openly (under an updated Llama 3 Community License), Meta aims to democratize access to frontier-level language understanding. This is not purely altruistic — an open ecosystem built around Llama creates network effects, generates improvements through community fine-tuning and tooling, and positions Meta's infrastructure and hardware stack as the platform of choice for running these models at scale.

### Why Previous Approaches Fall Short: The Limitations of Prior Open Models

The Llama 3 project builds directly on Llama 1 (Touvron et al., 2023a) and Llama 2 (Touvron et al., 2023b), but the paper makes clear that those predecessors, while impactful, were not competitive with the best closed-source models. Several specific gaps motivate the work:

**1. Insufficient scale in both data and compute.** Llama 2 was pre-trained on 1.8T tokens. Llama 3 scales this to approximately 15T multilingual tokens — roughly an 8× increase in data volume. The flagship Llama 3 model used 3.8 × 10²⁵ FLOPs for pre-training, almost 50× more than the largest Llama 2 model. These are not arbitrary increases; they reflect the paper's central thesis that data quality and scale, combined with sufficient compute, remain the primary drivers of capability improvement, not architectural novelty.

**2. Lack of multilingual support.** The earlier Llama models were predominantly English-focused. Llama 3 natively supports eight languages (English, German, French, Italian, Portuguese, Hindi, Spanish, Thai) and was trained on a significantly larger multilingual corpus. Previous open models either ignored non-English languages or achieved multilingual capability through post-hoc translation or fine-tuning, which the paper argues produces inferior quality (particularly for languages where "translationese" — artifacts introduced by machine translation — degrades naturalness and cultural appropriateness).

**3. Missing capabilities.** Llama 2 lacked built-in support for tool use (function calling, search integration, code execution), long-context reasoning beyond a few thousand tokens, and the kind of steerability that developers need to customize model behavior through system prompts. These capabilities were available in closed-source systems (GPT-4's function calling, Claude's 200K context window) but absent from open alternatives. Llama 3 explicitly targets these gaps as first-class design objectives.

**4. The post-training pipeline was underspecified.** While Llama 2 introduced the combination of supervised fine-tuning and RLHF (reinforcement learning from human feedback), the paper notes that this approach was "less stable and harder to scale" than the DPO-based approach adopted in Llama 3. The prior post-training pipeline also lacked the iterative, multi-round data collection and model improvement process that Llama 3 develops — where each round of human preference annotation, rejection sampling, SFT, and DPO feeds into the next, progressively improving the model and the data used to train it.

**5. Safety and alignment were treated as afterthoughts rather than integrated throughout development.** Llama 3's safety approach (detailed in Section 5.4) is substantially more comprehensive than its predecessors, spanning pre-training data filtering, safety-specific fine-tuning, red teaming across capabilities and languages, system-level safeguards (Llama Guard 3, Prompt Guard, Code Shield), and uplift testing for cybersecurity and chemical/biological weapons risks. The motivation is not just ethical — it is practical. Models that are not systematically evaluated for safety cannot be responsibly released, and gaps in safety coverage limit the model's adoption in real-world applications.

### Where Existing Open-Source Alternatives Stand — and Where They Don't

By mid-2024, the open-source LLM landscape had expanded dramatically. The paper benchmarks against a range of competitors to establish Llama 3's position:

- **Mistral 7B and Mixtral 8×22B** (Jiang et al., 2023, 2024) demonstrated that smaller models and mixture-of-experts architectures could achieve strong performance. However, Mixtral 8×22B — despite using a sparse MoE architecture — is outperformed by Llama 3 70B (a dense model) on most benchmarks (see Table 2). This is noteworthy: it suggests that **dense architectures are not the limiting factor**, and that careful data curation and training methodology can overcome the capacity advantages of MoE.

- **Gemma** (Team et al., 2024) from Google provided capable small models but lacked the multilingual, long-context, and tool-use capabilities that Llama 3 targets. Gemma 2 9B, for instance, achieves 72.3% on MMLU compared to Llama 3 8B's 69.4% — but Llama 3 8B substantially outperforms on code (72.6% vs. 54.3% on HumanEval), math (84.5% vs. 76.7% on GSM8K), and instruction following (80.4% vs. 73.6% on IFEval).

- **Nemotron 4 340B** is the closest open competitor in scale, but Llama 3 405B outperforms it across the board: 87.3% vs. 82.6% on MMLU, 89.0% vs. 73.2% on HumanEval, and 88.6% vs. 78.7% on MMLU-Pro (0-shot CoT). The performance gap is substantial enough (nearly 16 points on HumanEval) to suggest differences in training data quality or post-training methodology, not just scale.

- **GPT-4, GPT-4o, and Claude 3.5 Sonnet** represent the closed-source frontier. Llama 3 405B does not claim to surpass these models — the paper is explicit that it "performs on par with leading language models such as GPT-4 across a variety of tasks, and is close to matching the state-of-the-art." Table 2 shows a mixed picture: Llama 3 leads on GSM8K (96.8% vs. GPT-4's 94.2%) and is competitive on MMLU (87.3% vs. 85.1%), but trails on MMLU-Pro (73.3% vs. 74.0% for GPT-4o) and GPQA (51.1% vs. 59.4% for Claude 3.5 Sonnet). The human evaluations (Figure 17) confirm this parity narrative: win rates against GPT-4 are within margin of error on most capabilities, while the model shows some wins and some losses against GPT-4o and Claude 3.5 Sonnet.

The key motivation, however, is not that Llama 3 is strictly better than every alternative — it is that **an openly-available model can be competitive with the closed-source frontier at all**. This shifts the baseline of what "open" means from "the best you can get without paying for an API" to "a genuine alternative to the best available systems."

### The Technical Gap: How to Scale Post-Training Effectively

Beyond the headline performance comparisons, the paper addresses a more specific technical gap: **how to design a post-training pipeline that scales to models with hundreds of billions of parameters while maintaining stability, reliability, and safety.** This is not a problem that pre-training scaling laws address. The Chinchilla scaling laws (Hoffmann et al., 2022) and subsequent work tell you how to allocate compute between model size and data during pre-training — they say nothing about how to align the resulting model with human preferences, integrate new capabilities (tool use, long context), or ensure safety.

Several prior approaches to post-training have known failure modes that the paper explicitly seeks to avoid:

- **RLHF with PPO** (Ouyang et al., 2022; Schulman et al., 2017) is the approach used in Llama 2 and many other models. The paper notes that PPO is "less stable and harder to scale" than DPO. Training a separate reward model and then optimizing against it via reinforcement learning introduces a complex multi-stage pipeline with many hyperparameters, and the online nature of PPO training (where the model generates responses, gets rewards, and updates) creates distributional shift that can lead to instability. The paper's choice of DPO — which directly optimizes the policy from preference pairs without a separate reward model or RL loop — is a deliberate simplification.

- **Single-round alignment** (train once on a fixed dataset) cannot keep pace with a model's improving capabilities. The paper's iterative approach — six rounds of data collection, SFT, and DPO, with each round using the improved model to generate better data — is essential to the final performance. Earlier Llama versions did fewer rounds with less sophisticated data curation.

- **Capability-specific training as an afterthought.** Prior models often treated capabilities like code generation, multilingual support, and tool use as bolt-ons — fine-tuning on domain-specific data after the main alignment process. Llama 3 integrates these capabilities into the core post-training pipeline, training dedicated "experts" (e.g., a code expert branched from the main pre-training run and further trained on >85% code data) that are then used to generate high-quality synthetic data and collect better human annotations for the main model.

### The Safety Landscape: A More Demanding Standard

The paper is motivated by a substantially higher bar for safety evaluation than its predecessors. This is not merely about avoiding obviously harmful outputs — it is about systematically assessing risks across new capabilities (multilingual, long-context, tool use, multimodal) that each introduce unique attack surfaces.

Several specific gaps in prior safety work are addressed:

- **Multilingual safety transfer does not work.** The paper finds that "safety knowledge in English does not readily transfer to other languages" (Section 5.4.4), requiring language-specific safety data collection and evaluation. Prior models with weaker multilingual support could sidestep this problem; Llama 3 cannot.

- **Long-context models are vulnerable to many-shot jailbreaking** (Anil et al., 2024), where providing many examples of unsafe behavior in the context window can override safety training. The paper develops specific mitigations for this, including training on SFT data with demonstrations of safe behavior in the presence of unsafe context — a problem that simply didn't exist for models with 4K context windows.

- **Tool use introduces new attack vectors** (Wallace et al., 2024). A model that can execute code or search the web can be prompted to perform unsafe actions that a text-only model cannot. The paper's safety evaluation explicitly tests these capabilities, including code interpreter abuse and unsafe search tool usage.

- **System-level safety is essential but often neglected in model-level analyses.** The paper develops and releases Llama Guard 3 (a safety classifier for input/output filtering), Prompt Guard (a model-based filter for detecting jailbreaks and prompt injections), and Code Shield (inference-time filtering for insecure code generation). These are not afterthoughts — they are integrated into the development process and released alongside the models to enable developers to deploy safely.

### How Llama 3 Positions Itself

The paper frames its contributions across three dimensions that structure the entire development process:

**Data as the primary lever.** The paper makes clear that improvements in data quality and diversity, not architectural innovation, drive most of the performance gains. The pre-training data pipeline is described in substantial detail (Section 3.1): URL-level, document-level, and line-level de-duplication; heuristic filtering for low-quality content; model-based quality classification using DistilRoberta trained on Llama 2 annotations; domain-specific pipelines for code and math; and careful data mix determination via scaling law experiments and knowledge classification. The post-training data pipeline is equally elaborate (Section 4.2): human preference annotation with multi-turn dialogues and editing steps, rejection sampling with reward model selection, synthetic data generation with execution feedback for code, and quality control via topic classification, difficulty scoring, and semantic de-duplication.

**Scale through simplicity.** The paper explicitly rejects architectural complexity in favor of scaling what works. Llama 3 uses a standard dense Transformer with grouped query attention — no mixture-of-experts, no novel attention mechanisms, no retrieval augmentation in the base architecture. The paper states: "In preliminary experiments, we explored more complex model architectures and training recipes but did not find the benefits of such approaches to outweigh the additional complexity they introduce in model development." This is a strong methodological claim: at the frontier of scale, simplicity enables reliability, and reliability enables the kind of iterative improvement that the post-training pipeline depends on.

**Iterative, capability-driven post-training.** Rather than a single alignment phase, Llama 3's post-training is organized around capabilities (Section 4.3) — code, multilinguality, math/reasoning, long context, tool use, factuality, steerability — each with its own data generation pipelines, expert models, and quality control processes. The six rounds of post-training are not identical repetitions; they deliberately increase complexity (e.g., tool use annotations start with single-turn, then move to multi-turn, then to multi-step), targeting areas where the current model underperforms.

This positioning — data quality over architectural novelty, scale through simplicity, iterative capability-focused alignment — defines Llama 3 not as a single breakthrough but as a systematic engineering project that integrates lessons from the broader literature (scaling laws, DPO, rejection sampling, synthetic data generation) into a coherent development process. The paper's contribution is as much methodological as it is empirical: a detailed recipe for building competitive foundation models, released openly so that others can replicate, critique, and improve upon it.

## 3. Technical Approach

### 3.1 Reader Orientation

This paper presents a systematic engineering methodology for building a family of openly-available foundation language models — the Llama 3 Herd — ranging from 8B to 405B parameters, plus compositional multimodal extensions for vision and speech. The core problem is how to produce models competitive with the best closed-source systems (GPT-4, Claude, Gemini) through a development process that emphasizes data quality, training scale, and methodological simplicity over architectural novelty. The solution takes the form of a massively scaled two-stage pipeline: a pre-training stage that ingests ~15T curated multilingual tokens to train a dense Transformer to perform next-token prediction at unprecedented scale, followed by an iterative six-round post-training stage that uses supervised finetuning, rejection sampling, and Direct Preference Optimization — orchestrated around specific capabilities (code, multilinguality, reasoning, tool use, long context) — to produce aligned, instruction-following models competitive with the state-of-the-art.

### 3.2 Big-Picture Architecture (Diagram in Words)

The Llama 3 development process comprises five major component groups arranged in a sequential pipeline:

1. **Data Curation Pipeline:** Ingests raw web data, applies multi-stage filtering (PII removal, de-duplication at URL/document/line level, heuristic quality filtering, model-based quality classification), and produces a carefully balanced data mix (~50% general knowledge, 25% math/reasoning, 17% code, 8% multilingual tokens) totaling ~15T tokens across 128K-token vocabulary.

2. **Pre-Training Engine:** A standard dense Transformer architecture (up to 126 layers, 16,384 model dimension, 128 attention heads for the 405B variant) trained with next-token prediction on the curated corpus using 4D parallelism (tensor, context, pipeline, and data parallelism) across up to 16K H100 GPUs, consuming 3.8 × 10²⁵ FLOPs. The output is a base language model that captures world knowledge and linguistic structure.

3. **Post-Training Pipeline:** Takes the pre-trained model through six iterative rounds of (a) human preference annotation where annotators compare model responses and optionally edit the preferred one, (b) reward model training on these preferences, (c) rejection sampling where the reward model selects the best among K candidate outputs, (d) supervised finetuning on the rejection-sampled data plus synthetic data, and (e) Direct Preference Optimization for alignment. Each round uses the improved model to generate better data for the next round.

4. **Capability-Specific Expert Systems:** Separate models branched from the main pre-training run and further specialized (a code expert trained on >85% code data, a multilingual expert trained on 90% multilingual data) that serve dual purposes — generating higher-quality synthetic training data and collecting better human annotations for their respective domains.

5. **Compositional Multimodal Adapters:** Separate encoders and adapters for images, video, and speech that are trained independently and then integrated with the frozen language model through cross-attention layers (vision) or direct token embedding insertion (speech), enabling multimodal capabilities without modifying the core language model.

Information flows sequentially: raw web/text data → curated pre-training corpus → pre-trained base model → iterative post-training (human annotations + synthetic data + expert models) → aligned chat model. Multimodal extensions branch off the aligned chat model and add modality-specific processing without backpropagating into the language model itself.

### 3.3 Roadmap for the Deep Dive

- **First, the pre-training data pipeline (Section 3.1):** This is the foundation everything else builds on. I'll walk through the web data curation, de-duplication strategies (URL → document → line level), heuristic and model-based filtering, the data mix determination via scaling laws, and the special handling of code, math, and multilingual data — because data quality is the paper's central claim about what drives performance.

- **Second, the model architecture and scaling laws (Section 3.2):** Understanding the Transformer configuration (grouped query attention, 128K vocabulary, RoPE θ=500K) and how scaling laws were used to predict downstream performance before training begins. This is the "why 405B parameters?" story.

- **Third, the training infrastructure and parallelism strategy (Section 3.3):** The 4D parallelism scheme, the network topology, the pipeline parallelism improvements, and the reliability engineering — because training a model at this scale requires solving distributed systems problems that are as hard as the ML problems.

- **Fourth, the pre-training recipe (Section 3.4):** The three-stage process (initial pre-training → long-context pre-training → annealing) and the data mix adjustments made during training, including the critical annealing phase that upsamples high-quality data.

- **Fifth, the post-training pipeline (Sections 4.1–4.3):** The iterative six-round process, the reward model training, rejection sampling with PagedAttention, DPO with formatting token masking and NLL regularization, and the capability-specific data generation strategies (execution feedback for code, backtranslation, MCTS for reasoning).

### 3.4 Detailed, Sentence-Based Technical Breakdown

This is primarily a **systems engineering paper** whose core idea is that frontier-level language model performance can be achieved through systematic optimization of data quality, training scale, and post-training methodology — without architectural breakthroughs — and that releasing the resulting models openly enables scientific scrutiny and downstream innovation that closed-source alternatives preclude.

---

#### Pre-Training Data Curation

The pre-training data pipeline transforms raw web content into a carefully balanced 15T-token training corpus through a multi-stage process that the paper argues is the single most important factor in model quality.

**Data Sources and Initial Filtering**

The data is drawn from "a variety of data sources containing knowledge until the end of 2023" (Section 3.1). Before any linguistic processing, the pipeline applies safety and privacy filters: removing domains containing large amounts of personally identifiable information (PII), domains rated as harmful according to Meta safety standards, and domains known to contain adult content. These are hard filters — content from flagged domains is excluded entirely, not downsampled.

The raw HTML processing uses a custom-built parser rather than off-the-shelf alternatives, optimized for "precision in boilerplate removal and content recall." The paper explicitly states they evaluated this parser against "popular third-party HTML parsers that optimize for article-like content, and found it to perform favorably" in human evaluations. A non-obvious detail: they "carefully process HTML pages with mathematics and code content to preserve the structure of that content" and "maintain the image alt attribute text since mathematical content is often represented as pre-rendered images where the math is also provided in the alt attribute." This is a concrete example of how data quality improvements are domain-specific — mathematical content on the web often doesn't render as clean LaTeX but as images with alt text, and preserving that alt text makes the math accessible to the language model.

They also experimentally evaluate different text formatting conventions and find that "markdown is harmful to the performance of a model that is primarily trained on web data compared to plain text, so we remove all markdown markers." This is a subtle but important finding: markdown — which structures text for human readability — adds tokens that don't carry semantic content and may confuse a model trained to predict plain text.

**Three-Tier De-Duplication**

The paper applies de-duplication at three increasingly granular levels:

1. **URL-level de-duplication:** Across the entire dataset, keep only the most recent version of pages corresponding to each URL. This prevents training on multiple snapshots of the same page and ensures the model sees the latest available information.

2. **Document-level de-duplication:** Using global MinHash (Broder, 1997), remove near-duplicate documents across the entire dataset. MinHash is a locality-sensitive hashing technique that estimates the Jaccard similarity between documents — if two documents share a high fraction of their n-grams, they produce similar MinHash signatures and one is removed. The paper doesn't specify the exact similarity threshold, but "near duplicate" implies a high threshold (likely >0.8 Jaccard similarity).

3. **Line-level de-duplication:** Following the ccNet approach (Wenzek et al., 2019), remove lines that appear more than 6 times in each bucket of 30M documents. This is "aggressive" line-level dedup — the paper acknowledges that "manual qualitative analysis showed that the line-level de-duplication removes not only leftover boilerplate from various websites such as navigation menus, cookie warnings, but also frequent high-quality text," yet "empirical evaluations showed strong improvements." This is a revealing trade-off: some genuinely useful text is sacrificed to eliminate pervasive boilerplate, and the net effect on model quality is positive. It underscores the paper's thesis that removing low-information patterns matters more than preserving every instance of good content.

**Heuristic Filtering**

Beyond de-duplication, the paper applies several heuristic filters targeting specific quality issues:

- **Duplicated n-gram coverage ratio** (Rae et al., 2021): Remove lines that consist of repeated content such as logging or error messages. These lines can be very long and unique (so they survive line-level dedup) but consist of repetitive patterns that don't contribute to language understanding.

- **"Dirty word" counting** (Raffel et al., 2020): Filter out adult websites not covered by domain block lists by counting occurrences of adult-content vocabulary.

- **Token-distribution Kullback-Leibler divergence:** For each document, compute the KL divergence between its token distribution and the overall training corpus distribution. Documents with excessive numbers of outlier tokens (relative to the corpus) are filtered out. The KL divergence from document distribution `P` to corpus distribution `Q` is:

   $$D_{KL}(P \parallel Q) = \sum_{t} P(t) \log\frac{P(t)}{Q(t)}$$

   where `P(t)` is the probability of token `t` in the document, `Q(t)` is its probability in the corpus, and the sum runs over all tokens in the vocabulary.

   **What it computes:** for each token type, it measures how much more frequent that token is in the document compared to the corpus average, weighted by the document's own token probability. If a document uses many rare tokens (or overuses specific common tokens relative to the corpus baseline), the KL divergence will be high.

   **Why this form:** KL divergence is asymmetric — it penalizes documents that have tokens where the corpus has none (the `log` term would diverge), so it's applied only where both distributions have support. It's a natural measure of distributional mismatch that doesn't require arbitrary thresholds on individual token frequencies; the threshold is applied to the aggregate divergence score instead.

**Model-Based Quality Filtering**

The paper deploys a two-tier model-based filtering approach:

First, a fast classifier: fasttext (Joulin et al., 2017) trained to recognize whether a given text would be referenced by Wikipedia (following Touvron et al., 2023a). This is a binary classifier operating on the principle that Wikipedia-referenced text is generally higher quality. fasttext is chosen for speed — it can process billions of documents at the scale of web data.

Second, a more compute-intensive but higher-quality classifier: a DistilRoberta model (Sanh et al., 2019) trained on quality judgments from Llama 2. The training data for this classifier is created by taking cleaned web documents, describing quality requirements to Llama 2's chat model, and instructing it to determine whether the documents meet those requirements. DistilRoberta is used for efficiency — it's a distilled version of RoBERTa that's 40% smaller and 60% faster while retaining 95% of the performance.

The paper "experimentally evaluate[s] the efficacy of various quality filtering configurations," suggesting that the specific combination and thresholds of these classifiers were tuned via ablation, though the exact configurations aren't specified.

**Domain-Specific Pipelines for Code and Math**

The paper implements separate pipelines for code and mathematics content, motivated by the observation that "the token distribution of code and math is substantially different than that of natural language." These pipelines use:

- **Domain-specific HTML extraction:** Different parsing rules optimized for code repositories, math forums, and STEM educational content.
- **Customized text features and heuristics:** Features designed to identify code blocks (indentation patterns, keyword frequency) and mathematical notation (LaTeX delimiters, equation patterns).
- **Domain-specific quality classifiers:** DistilRoberta models trained on web data annotated by Llama 2, but with "prompt tuning to target web pages containing math deduction, reasoning in STEM areas and code interleaved with natural language." The prompt tuning means the Llama 2 annotator was given specific instructions designed to identify code/math content, not just general quality.

This separation is important because a general-purpose quality classifier might filter out legitimate code (which looks low-quality by natural language standards — high symbol density, repetitive patterns, non-grammatical structure) or mathematical derivations (which look like nonsense to a classifier expecting coherent prose).

**Multilingual Data Processing**

The multilingual pipeline mirrors the English pipeline with language-specific adaptations:

- A fasttext-based language identification model categorizes documents into 176 languages.
- Document-level and line-level de-duplication is performed within each language separately (preventing cross-lingual dedup that might remove legitimate translations).
- Language-specific heuristics and model-based filters are applied — what constitutes low-quality text differs across languages, so generic filters are insufficient.
- A multilingual Llama 2-based classifier performs quality ranking to prioritize high-quality content.
- The amount of multilingual tokens in the final mix is "determined experimentally, balancing model performance on English and multilingual benchmarks" — it's not a fixed proportion but the result of small-scale training runs evaluating different multilingual ratios.

#### Determining the Data Mix

The data mix — the proportion of tokens from different sources and domains — is optimized through two complementary approaches.

**Knowledge Classification**

A classifier categorizes the types of information contained in web data, enabling the team to identify categories that are over-represented on the web relative to their usefulness for model training. For example, "arts and entertainment" content may be abundant on the web but less valuable for a general-purpose model than scientific or technical content. The classifier enables targeted downsampling of over-represented categories.

**Scaling Laws for Data Mix Selection**

This is a computationally expensive but principled approach:

1. Train several small models on a candidate data mix.
2. Use the scaling law methodology (described in Section 3.2.1) to predict how a large model would perform on that mix.
3. Repeat for different data mixes to identify promising candidates.
4. Train a larger model on the best candidate and evaluate on key benchmarks.

The final data mix is: **roughly 50% general knowledge tokens, 25% mathematical and reasoning tokens, 17% code tokens, and 8% multilingual tokens.** This is a striking allocation — nearly half the training data is specialized (math, code, multilingual), not general web text. The heavy emphasis on math and code (42% combined) reflects the paper's finding that these domains are particularly important for developing reasoning capabilities, and that web data alone underrepresents structured, logical content.

#### Annealing Data

The pre-training process includes a final "annealing" phase (Section 3.4.3) where the learning rate is linearly decayed to zero on a small amount of specially curated high-quality data. The paper finds that "annealing on small amounts of high-quality code and mathematical data can boost the performance of pre-trained models on key benchmarks."

Crucially, they "do not include any training sets from commonly used benchmarks in [the] annealing data. This enables us to assess the true few-shot learning capabilities and out-of-domain generalization of Llama 3." This is a deliberate choice to avoid benchmark contamination — a common pitfall where models appear to perform well on benchmarks because they've memorized the test data during training.

The paper evaluates whether to include GSM8k and MATH training sets in the annealing data (following OpenAI, 2023a) and finds that "annealing improved the performance of a pre-trained Llama 3 8B model on the GSM8k and MATH validation sets by 24.0% and 6.4%, respectively. However, the improvements on the 405B model are negligible, suggesting that our flagship model has strong in-context learning and reasoning capabilities and does not require specific in-domain training samples to obtain strong performance." This is a revealing finding about scale: larger models benefit less from domain-specific fine-tuning because they already generalize well from their pre-training data.

The paper also uses annealing as a rapid evaluation tool: "annealing enables us to judge the value of small domain-specific datasets." By annealing a 50%-trained Llama 3 8B model on 40B tokens with 30% weight assigned to a new dataset and evaluating the performance change, they can quickly assess whether a dataset is worth including in the main pre-training mix, avoiding the need for full scaling law experiments for every candidate dataset.

---

#### Model Architecture and Scaling Laws

Llama 3 uses a standard dense Transformer architecture with minimal modifications from Llama 2, making a deliberate choice to invest engineering effort in data and training methodology rather than architectural novelty.

**Core Architecture**

The model is a causal (autoregressive) decoder-only Transformer. The 405B variant has:

- **126 layers**
- **Model dimension of 16,384**
- **128 attention heads**
- **Feed-forward network dimension of 53,248** (using SwiGLU activation)
- **8 key-value heads** (via grouped query attention)
- **Vocabulary size of 128,000 tokens**
- **Positional embeddings via Rotary Position Embeddings (RoPE) with base frequency θ = 500,000**

The 8B and 70B variants use the same architecture with proportionally scaled dimensions (see Table 3).

**Modifications from Llama 2**

1. **Grouped Query Attention (GQA; Ainslie et al., 2023) with 8 key-value heads.** Standard multi-head attention computes separate key, query, and value projections for each attention head. GQA shares key and value projections across groups of query heads — here, 128 query heads share 8 key-value heads (16:1 ratio). This reduces the size of the key-value cache during autoregressive decoding by a factor of 16, which is critical for serving a 405B model with long contexts. The trade-off is slightly reduced expressivity in the attention patterns, but prior work (Ainslie et al., 2023) showed this has minimal impact on quality.

2. **Document-level attention masking.** Self-attention is prevented between tokens belonging to different documents within the same training sequence. During standard pre-training on sequences of concatenated documents, the model could otherwise attend across document boundaries, learning spurious cross-document dependencies that don't exist at inference time. The paper notes this "had limited impact during standard pre-training, but find it to be important in continued pre-training on very long sequences" — when sequences span 128K tokens and may contain many documents, cross-document attention would waste compute on irrelevant connections.

3. **128K-token vocabulary combining 100K tiktoken tokens with 28K additional tokens for non-English languages.** Compared to Llama 2's tokenizer, this improves the compression rate on English data from 3.17 to 3.94 characters per token. The compression rate improvement means the model "reads" about 24% more text for the same number of tokens, effectively increasing the information content of each training step. The additional 28K tokens from non-English languages improve compression for those languages without degrading English tokenization.

4. **RoPE base frequency increased to 500,000** (from 10,000 in Llama 2). Rotary Position Embeddings encode position information by rotating query and key vectors in the attention computation. The base frequency `θ` controls the wavelengths of the sinusoidal components used in this rotation. A higher base frequency means the embeddings can distinguish positions at longer ranges — Xiong et al. (2023) showed this value to be effective for context lengths up to 32,768. Llama 3 extends context to 128K, and the high θ helps maintain positional resolution at those lengths.

**Why Not Mixture-of-Experts?**

The paper explicitly addresses the choice to remain dense: "we opt for a standard dense Transformer model architecture... rather than for a mixture-of-experts model to maximize training stability." Mixture-of-Experts (MoE) architectures — used by Mixtral 8×22B (Jiang et al., 2024) — replace some feed-forward layers with multiple "expert" sub-networks where a router selects which experts to activate for each token. This increases model capacity without proportionally increasing compute (since only a subset of experts are active per token). However, MoE training introduces several complications: load balancing across experts (preventing some experts from being overused while others are underused), communication overhead from dispatching tokens to different experts across devices, and training instability from the discrete routing decisions. The paper's position is that these complications aren't worth the capacity gains when you can simply train a larger dense model with sufficient compute and high-quality data — and Llama 3 70B's outperformance of Mixtral 8×22B (dense beating MoE) supports this claim empirically.

---

#### Scaling Laws Methodology

The paper develops scaling laws not just to determine the optimal model size, but to **predict downstream benchmark performance before training begins**. This is a significant extension of standard scaling law practice.

**The Two-Stage Prediction Pipeline**

Standard scaling laws (Hoffmann et al., 2022; Kaplan et al., 2020) predict next-token prediction loss (perplexity) as a function of model size and training tokens. But practitioners care about downstream task performance (MMLU accuracy, HumanEval pass@1), not loss. The paper bridges this gap with a two-stage approach:

**Stage 1: Correlate loss with training FLOPs.** Train small models at various compute budgets, measure their negative log-likelihood on downstream tasks, and establish a relationship between the compute-optimal model's loss and training FLOPs.

**Stage 2: Correlate loss with accuracy.** Using both the scaling law models and the older Llama 2 family (which was trained with higher FLOPs), establish a sigmoidal relationship between negative log-likelihood and task accuracy.

**Scaling Law Experiments**

The paper trains models using compute budgets between 6 × 10¹⁸ FLOPs and 10²² FLOPs. At each budget, they train models of various sizes (40M to 16B parameters) using a subset of model sizes at each budget. Training uses a cosine learning rate schedule with 2,000-step linear warmup, peak learning rate between 2 × 10⁻⁴ and 4 × 10⁻⁴ depending on model size, cosine decay to 0.1 of the peak value, and weight decay at each step set to 0.1 times the current learning rate.

From these experiments, they construct **IsoFLOPs curves** (Figure 2) — for each fixed compute budget, they plot validation loss against model size (or equivalently, training tokens, since at fixed compute, tokens ∝ 1/size). The curves are fit with second-degree polynomials, and the minimum of each parabola identifies the compute-optimal model size for that budget.

**The Scaling Law**

The compute-optimal number of training tokens `N*(C)` is modeled as a power law:

$$N^*(C) = A C^\alpha$$

where `C` is the training compute budget in FLOPs, `A` is a constant scaling factor, and `α` is the scaling exponent.

**What it computes:** Given a total pre-training compute budget `C`, this predicts the number of training tokens that minimizes validation loss when using the compute-optimal model size. The exponent `α` determines how the optimal allocation between model size and training tokens shifts with scale — if `α > 0.5`, the optimal model grows faster than the optimal data; if `α < 0.5`, data grows faster.

**Why this form:** The power-law relationship is consistent with empirical scaling law findings from Kaplan et al. (2020) and Hoffmann et al. (2022). It captures the observation that optimal training tokens scale sublinearly with compute — you need to increase both model size and training tokens as compute grows, but not at the same rate.

The fitted values are `(α, A) = (0.53, 0.29)`, meaning `α ≈ 0.5` — the optimal model size and training tokens grow at roughly the same rate with compute. Extrapolating to 3.8 × 10²⁵ FLOPs gives a prediction of 402B parameters on 16.55T tokens. The actual 405B model on 15.6T tokens is close to this prediction.

**A crucial observation:** The paper notes that "IsoFLOPs curves become flatter around the minimum as the compute budget increases. This implies that performance of the flagship model is relatively robust to small changes in the trade-off between model size and training tokens." This flatness is practically important — it means you don't need to hit the exact optimal point; being in the neighborhood is sufficient, which provides flexibility in model size selection.

**Predicting Downstream Performance (Figure 4)**

For the ARC Challenge benchmark specifically, the two-stage prediction works as follows:

First, the paper linearly correlates the normalized negative log-likelihood of the correct answer on ARC Challenge with the training FLOPs across the scaling law models (trained up to 10²² FLOPs). This produces the left panel of Figure 4.

Second, they establish a sigmoidal relationship between this log-likelihood and accuracy using both the scaling law models and the Llama 2 models (which were trained with different data and tokenizer but provide additional data points at higher FLOPs). This produces the right panel of Figure 4.

The predicted accuracy for Llama 3 405B slightly underestimates the actual final performance, but extrapolates successfully over four orders of magnitude of compute — from 10²¹ to 3.8 × 10²⁵ FLOPs. The paper calls this prediction "quite accurate."

**Training Smaller Models Beyond Compute-Optimality**

While the 405B model is approximately compute-optimal (trained for roughly the Chinchilla-optimal number of tokens given its size), the 8B and 70B models are deliberately trained "for much longer than is compute-optimal." This is a deliberate trade-off: spend extra training FLOPs to produce smaller models that are more capable at inference time. The resulting models "perform better than compute-optimal models at the same inference budget" — they're more expensive to train but cheaper to deploy, which is the right trade-off for models intended to be widely used.

---

#### Training Infrastructure and Parallelism

Training a 405B-parameter model requires solving distributed systems challenges that go well beyond standard deep learning infrastructure. The paper provides detailed information about the hardware, network, and parallelism strategy.

**Hardware Configuration**

Llama 3 405B is trained on up to 16K H100 GPUs, each running at 700W TDP with 80GB HBM3, using Meta's Grand Teton AI server platform. Each server has 8 GPUs connected via NVLink (high-bandwidth intra-server communication) and 2 CPUs. Training jobs are scheduled using MAST (Choudhury et al., 2024), Meta's global-scale training scheduler.

**Storage.** The Tectonic distributed file system (Pan et al., 2021) provides 240 PB of storage across 7,500 SSD-equipped servers, with 2 TB/s sustainable and 7 TB/s peak throughput. A key challenge is "highly bursty checkpoint writes that saturate the storage fabric for short durations" — each GPU needs to save 1 MB to 4 GB of model state, and doing this across 16K GPUs simultaneously creates enormous I/O spikes.

**4D Parallelism (Figure 5)**

The model is sharded across GPUs using four types of parallelism simultaneously, applied in a specific order: **[TP, CP, PP, DP]**. The order is critical — it determines which parallelism dimensions share which network resources.

- **Tensor Parallelism (TP):** Individual weight tensors are split into chunks across multiple devices. Matrix multiplications that would be too large for a single GPU's memory are partitioned column-wise or row-wise. TP requires very high bandwidth (NVLink) because every forward and backward pass involves communication to combine partial results. TP is therefore constrained to within a single server (8 GPUs).

- **Context Parallelism (CP):** The input sequence is partitioned across devices, with each device processing a segment of the full sequence. This reduces the memory required for activations (which scale with sequence length). The paper's implementation uses an all-gather approach: each CP rank first gathers the key and value tensors from all other ranks, then computes attention for its local query chunk. This is simpler than ring-based approaches (Liu et al., 2023a) and works well because the communicated K/V tensors are small (due to GQA) while the attention computation dominates.

- **Pipeline Parallelism (PP):** The model's layers are divided into stages, with different devices processing different stages in a pipeline. Micro-batches flow through the pipeline, enabling concurrent computation across stages. The paper uses an interleaved schedule with V pipeline stages per rank to reduce pipeline bubbles. The bubble ratio (fraction of time devices are idle) is `(PP-1) / (V × M)` where `M` is the number of micro-batches.

- **Data Parallelism (DP) via Fully Sharded Data Parallelism (FSDP):** Model parameters, optimizer states, and gradients are sharded across data-parallel workers. Each worker processes a different data batch and synchronizes gradients. For Llama 3, FSDP shards optimizer states and gradients, but for model shards, it does not reshard after forward computation — avoiding an extra all-gather during the backward pass.

The parallelism dimensions are ordered `[TP, CP, PP, DP]` based on communication requirements: TP needs the highest bandwidth and lowest latency (within-server NVLink), CP shares the next tier, PP can tolerate higher latency (between servers in the same pod), and DP — the outermost — can tolerate multi-hop network latency because it asynchronously prefetches sharded weights.

**Model FLOPs Utilization (MFU)**

The paper achieves 38-43% BF16 MFU (Chowdhery et al., 2023), where MFU is the ratio of observed FLOPs to theoretical peak FLOPs of the hardware. Table 4 shows the configurations:

| GPUs | TP | CP | PP | DP | Seq Len | Batch/DP | TFLOPs/GPU | BF16 MFU |
|------|----|----|----|----|---------|----------|------------|----------|
| 8,192 | 8 | 1 | 16 | 64 | 8,192 | 32 | 430 | 43% |
| 16,384 | 8 | 1 | 16 | 128 | 8,192 | 16 | 400 | 41% |
| 16,384 | 8 | 16 | 16 | 8 | 131,072 | 16 | 380 | 38% |

The drop from 43% to 41% when scaling from 8K to 16K GPUs with the same tokens per batch is due to "the lower batch size per DP group needed to keep the global tokens per batch constant." The drop to 38% when enabling CP for 128K sequences reflects the all-gather communication overhead.

**Pipeline Parallelism Improvements**

The paper addresses three specific problems with existing PP implementations:

1. **Batch size constraint:** Standard depth-first scheduling requires the batch size per GPU to be divisible by the number of pipeline stages. The paper's modified schedule (Figure 6) makes the number of contiguous micro-batches `N` tunable, enabling arbitrary numbers of micro-batches.

2. **Memory imbalance:** The first pipeline stage consumes more memory (embedding + warm-up micro-batches), and the last stage is the execution latency bottleneck (output projection + loss calculation). The solution: remove one Transformer layer each from the first and last stages, so the first stage has only the embedding and the last stage has only the output and loss.

3. **Pipeline bubbles:** Using an interleaved schedule with `V` pipeline stages per rank reduces the bubble ratio. Additionally, asynchronous point-to-point communication in PP "considerably speeds up training, especially when the document mask introduces extra computation imbalance" — the document mask varies the amount of valid computation per sequence, creating load imbalance that async communication partially hides.

**Network Topology and Load Balancing**

The RoCE (RDMA over Converged Ethernet) cluster uses a three-layer Clos network:

- **Bottom layer:** Each rack has 16 GPUs (2 servers × 8 GPUs each) connected by a single Minipack2 top-of-rack switch.
- **Middle layer:** 192 racks connected by Cluster Switches to form a pod of 3,072 GPUs with full bisection bandwidth.
- **Top layer:** 8 pods connected via Aggregation Switches to form a 24K-GPU cluster, with 1:7 oversubscription (meaning 7× more internal pod bandwidth than cross-pod bandwidth).

The oversubscription at the top layer means cross-pod communication is significantly slower than intra-pod communication. The model parallelism and job scheduler are "optimized to be aware of network topology, aiming to minimize network communication across pods."

For load balancing, the paper uses two techniques:

- Each collective library creates 16 network flows between two GPUs (instead of 1), reducing per-flow traffic and providing more flows for balancing.
- Enhanced-ECMP (E-ECMP) protocol hashes on additional fields in the RoCE header to distribute these 16 flows across different network paths.

**Reliability Engineering**

During a 54-day snapshot, the training experienced 466 job interruptions (419 unexpected, 47 planned). Table 5 categorizes the unexpected interruptions:

| Component | Category | Count | % of Interruptions |
|-----------|----------|-------|-------------------|
| Faulty GPU | GPU | 148 | 30.1% |
| GPU HBM3 Memory | GPU | 72 | 17.2% |
| Software Bug | Dependency | 54 | 12.9% |
| Network Switch/Cable | Network | 35 | 8.4% |
| Host Maintenance | Unplanned | 32 | 7.6% |

GPU issues dominate (58.7% of unexpected interruptions). Despite this, "significant manual intervention was required only three times during this period, with the rest of issues handled by automation."

The paper highlights several reliability innovations:

- **NCCLX:** A fork of Nvidia's NCCL library that improves performance for high-latency networks by tuning chunking, prioritizing small control messages, and exposing internal state for debugging.
- **PyTorch NCCL flight recorder:** Captures collective metadata and stack traces into a ring buffer, enabling diagnosis of hangs and performance issues without job restart.
- **Straggler detection tools:** Identify slow GPUs by prioritizing investigation of communications from selected process groups.

An interesting environmental observation: "a diurnal 1-2% throughput variation based on time-of-day" due to "higher mid-day temperatures impacting GPU dynamic voltage and frequency scaling." The paper also notes that GPUs may simultaneously increase or decrease power consumption (e.g., waiting for checkpointing), causing "instant fluctuations of power consumption across the data center on the order of tens of megawatts, stretching the limits of the power grid."

---

#### Pre-Training Recipe

The pre-training of Llama 3 405B proceeds in three stages with carefully managed transitions.

**Stage 1: Initial Pre-Training**

The optimizer is AdamW with:

- Peak learning rate: 8 × 10⁻⁵
- Linear warmup: 8,000 steps
- Cosine schedule decaying to 8 × 10⁻⁷ over 1,200,000 steps

The batch size is progressively increased during training to balance stability and efficiency:

- Initial: 4M tokens, sequence length 4,096
- After 252M tokens: doubled to 8M tokens, sequence length 8,192
- After 2.87T tokens: doubled again to 16M tokens

The paper reports this recipe was "very stable: we observed few loss spikes and did not require interventions to correct for model training divergence." This is significant — at this scale, training instability (loss spikes, divergence) is a common failure mode requiring manual intervention, and the ability to train without such interventions is a direct benefit of the architectural simplicity and careful hyperparameter choices.

During initial pre-training, the data mix is adjusted several times:

- Non-English data percentage increased to improve multilingual performance.
- Mathematical data upsampled to improve reasoning.
- More recent web data added in later stages to advance the knowledge cutoff.
- Lower-quality subsets identified during training are downsampled.

These adjustments are made dynamically based on downstream evaluations of intermediate checkpoints, not predetermined — the training process includes continuous monitoring and course correction.

**Stage 2: Long-Context Pre-Training**

The context window is extended from 8K to 128K tokens in six incremental stages, using approximately 800B training tokens total. The paper does not train on long sequences from the beginning because "the compute in self-attention layers grows quadratically in the sequence length" — the FLOPs per token scale with `O(L²)` where `L` is the sequence length, so training on 128K sequences from the start would be prohibitively expensive.

Success at each stage is assessed by two criteria:

1. Model performance on short-context evaluations has recovered completely (long-context training shouldn't degrade short-context capabilities).
2. The model perfectly solves "needle in a haystack" tasks up to that length (it can retrieve a piece of information inserted anywhere in a long document).

The incremental approach ensures the model learns to handle longer contexts without catastrophic forgetting of shorter-context capabilities.

**Stage 3: Annealing**

During the final 40M tokens, the learning rate is linearly annealed to 0 while maintaining the 128K context length. The data mix is adjusted to upsample very high-quality sources (the annealing data described in Section 3.1.3). Finally, model checkpoints from the annealing phase are averaged (Polyak averaging, also known as model averaging or "model soup") to produce the final pre-trained model.

Polyak averaging computes the mean of model parameters across checkpoints:

$$\theta_{\text{final}} = \frac{1}{K} \sum_{i=1}^{K} \theta_i$$

where `θ_i` are the model parameters at checkpoint `i` and `K` is the number of checkpoints averaged.

**What it computes:** the element-wise arithmetic mean of the model's weights across multiple points near the end of training. This produces a model whose weights are the centroid of the trajectories explored during the final optimization steps.

**Why this form:** averaging reduces the variance from stochastic gradient noise — individual checkpoints fluctuate around a good solution due to the inherent noise in mini-batch gradients, and averaging cancels out this noise. It's a well-known technique (Izmailov et al., 2019; Wortsman et al., 2022) that improves generalization without additional training cost. The paper also uses this technique during post-training (Section 4.1.5) to combine models from different data/hyperparameter configurations.

---

#### Post-Training: The Iterative Alignment Pipeline

The post-training process transforms the pre-trained base model (which can complete text but doesn't follow instructions) into an aligned chat model through six iterative rounds. Each round follows the same pattern but uses progressively better models to generate progressively better training data.

**The Chat Dialog Format**

Llama 3 introduces a new multi-message chat protocol (Section 4.1.1) to support tool use, which requires generating multiple messages sent to different destinations (user, ipython, tool outputs) within a single dialog turn. The protocol uses "special header and termination tokens" — header tokens indicate the source and destination of each message, termination tokens indicate when the speaker alternates between human and AI.

This is more complex than standard chat formats (which assume simple human/AI alternation) because tool-using dialogs involve multiple participants: the user asks a question, the assistant calls a tool, the tool returns output, the assistant reasons about the output, the assistant calls another tool, etc. The chat format must track who is speaking and where the message is directed.

**Reward Model Training (Section 4.1.2)**

A reward model (RM) is trained on top of the pre-trained checkpoint using human-annotated preference data. The training objective is the same as Llama 2 except that "the margin term in the loss" is removed because the paper observes "diminishing improvements after data scaling."

The preference data has a distinctive structure: for some prompts, annotators not only select the preferred response from a pair (chosen vs. rejected) but also **edit the chosen response to further improve it**, creating a third "edited" response. The ranking is then: edited > chosen > rejected. During training, the prompt and multiple responses are "concatenated... into a single row... with responses randomly shuffled." This is an approximation to the standard approach of putting responses in separate rows, but the paper's ablations show it "improves training efficiency without a loss in accuracy."

**Supervised Finetuning with Rejection Sampling (Sections 4.1.3 and 4.2)**

Rejection sampling (RS) is the core data generation mechanism. For each prompt collected during human annotation, the process is:

1. Sample `K` outputs (typically 10-30) from the latest chat model policy.
2. Use the reward model to select the best candidate.
3. Train the model on these selected outputs using standard cross-entropy loss on target tokens (masking loss on prompt tokens).

The paper deploys PagedAttention (Kwon et al., 2023) to accelerate rejection sampling, achieving "over 2× throughput improvement." PagedAttention dynamically allocates key-value cache memory in pages, enabling memory sharing across outputs that share the same prompt prefix. The paper further optimizes by "defining a maximum output length and performing a request only if sufficient memory is available to fit an output with that length" — preventing the memory swaps that would otherwise occur when the cache exceeds GPU memory.

The SFT data mix (Table 7) shows a diverse composition: 52.66% general English, 21.19% reasoning and tools, 14.89% code, 8.14% exam-like, 3.01% multilingual, and 0.11% long context. The average conversation has 4.7 turns with 846.1 tokens total.

**Direct Preference Optimization (Section 4.1.4)**

Following SFT, the model is further trained with DPO (Rafailov et al., 2024), which directly optimizes the policy from preference pairs without a separate reward model. The DPO loss function is:

$$\mathcal{L}_{\text{DPO}}(\pi_\theta; \pi_{\text{ref}}) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)} \right) \right]$$

where `π_θ` is the policy being optimized (the current model), `π_ref` is the reference policy (typically the SFT model), `(x, y_w, y_l)` is a preference pair with prompt `x`, winning response `y_w`, and losing response `y_l`, `σ` is the sigmoid function, and `β` is a hyperparameter controlling the strength of the KL penalty relative to the preference objective.

**What it computes:** for each preference pair, it computes the difference in log-probability ratios (relative to the reference model) between the winning and losing responses, applies a sigmoid to map this difference to a probability, and minimizes the negative log-likelihood of the winning response being preferred under this probability. Intuitively, it increases the probability of winning responses and decreases the probability of losing responses, while the reference model term prevents the policy from diverging too far from its starting point.

**Why this form:** DPO avoids the need to train a separate reward model and then run RL (as in PPO), which the paper finds "required less compute for large-scale models and performed better, especially on instruction following benchmarks like IFEval." The implicit reward in DPO is `r(x,y) = β log(π_θ(y|x) / π_ref(y|x))` — the log-ratio of policy and reference probabilities — which means the optimization directly shapes the policy rather than going through a learned reward function.

The paper applies two modifications to standard DPO:

1. **Formatting token masking:** Special formatting tokens (header and termination tokens from the chat protocol) are masked out of the loss. The paper observes that including them "may lead to undesired model behaviors such as tail repetition or abruptly generating termination tokens," hypothesizing that this is because formatting tokens appear in both chosen and rejected responses, creating a "conflicting learning objective as the model needs to increase and reduce the likelihood of these tokens simultaneously."

2. **NLL regularization:** An additional negative log-likelihood loss term with scaling coefficient 0.2 is added on the chosen sequences:

   $$\mathcal{L} = \mathcal{L}_{\text{DPO}} + 0.2 \cdot \mathcal{L}_{\text{NLL}}(y_w|x)$$

   where `L_NLL` is the standard cross-entropy loss on the chosen response tokens. This "helps further stabilize DPO training by maintaining desired formatting for generation and preventing the decrease of log probability of chosen responses" (Pang et al., 2024; Pal et al., 2024). Without this regularization, DPO can sometimes reduce the absolute probability of chosen responses while maintaining the relative preference — the model technically prefers chosen over rejected, but generates both poorly. The NLL term directly counteracts this by pushing up the probability of the chosen response.

For Llama 3, the DPO hyperparameters are: learning rate 10⁻⁵, β = 0.1.

**Iterative Rounds and Model Averaging**

The above process (RM training → RS → SFT → DPO) is repeated six times. Each round collects new preference annotations and SFT data, sampling synthetic data from the latest models. As the rounds progress, the annotation protocols become more complex — for tool use, for example, the progression goes from single-turn annotations to multi-turn tool use to multi-step tool use and data analysis.

After each RM, SFT, or DPO stage, models obtained from "experiments using various versions of data or hyperparameters" are averaged (weight averaging, as in the pre-training annealing phase). This averaging combines different training runs to produce a model that is more robust than any individual run.

**Why Six Rounds?**

The paper doesn't provide an ablation comparing different numbers of rounds, but the iterative design addresses a fundamental problem: the distribution of model outputs changes during training. Data collected using an earlier model version becomes "off-policy" relative to the current model — the current model might generate different responses to the same prompts, so the preference data is no longer perfectly aligned with what the model would actually produce. By doing multiple rounds with fresh data collection each time, the training data stays closer to the model's current output distribution.

The paper also uses "the most recent batches of preference data collected using the best performing models from the previous alignment rounds" for DPO, deliberately discarding older batches that are "sufficiently off-policy."

---

#### Capability-Specific Post-Training

Each capability (code, multilinguality, math/reasoning, long context, tool use, factuality, steerability) receives specialized treatment with its own data generation pipelines and, in some cases, dedicated expert models.

**Code (Section 4.3.1)**

A code expert is trained by branching the main pre-training run and continuing pre-training on a 1T-token mix of mostly (>85%) code data, following the CodeLlama recipe (Rozière et al., 2023). In the final several thousand steps, long-context finetuning extends the expert's context length to 16K tokens using "a high quality mix of repo-level code data" — code from multi-file repositories where understanding requires cross-file context. This expert is then post-trained using the same SFT + DPO recipe but with code-focused data mixes, and is used to generate high-quality synthetic data and collect better human annotations.

Three synthetic data generation approaches are described:

1. **Execution feedback:** For each of approximately one million synthetic coding problems, the model generates a solution, static analysis (parser + linter) checks syntactic correctness, the model generates unit tests, the tests are executed in a containerized environment, and if the solution fails, the model receives the error output and revises the solution. This process iterates until the solution passes all checks. Only successful dialogs are included in the training data. The paper reports "about 20% of solutions were initially incorrect but self-corrected."

2. **Programming language translation:** Code from common languages (Python, C++) is translated to less common languages (TypeScript, PHP) by prompting Llama 3, with quality ensured through "syntax parsing, compilation, and execution." This addresses the training data imbalance where common languages have much more data.

3. **Backtranslation:** Starting from code snippets in the pre-training data, the model generates documentation/explanations, then is prompted to regenerate the code from that documentation. The original code serves as a reference, and Llama 3 scores the faithfulness of the backtranslated code. High-scoring examples are added to SFT data. This generated approximately 1.2M synthetic dialogs.

System prompts during rejection sampling are used to steer code quality: "code specific system prompts to improve code readability, documentation, thoroughness, and specificity" (see Figure 9 for an example where the system prompt adds comments, uses more informative variable names, and saves memory).

Finally, "model-as-judge" filtering removes low-quality coding data. Earlier versions of Llama 3 score rejection-sampled responses on "code correctness and code style" with a binary 0/1 score for each criterion. Only samples scoring 2 are retained. This initially caused regression on benchmarks because "it disproportionately removed examples with challenging prompts," so the responses for challenging prompts were strategically revised until they met the quality criteria.

**Multilinguality (Section 4.3.2)**

A multilingual expert is trained by branching the pre-training run and continuing on 90% multilingual tokens, then post-trained. This expert collects higher-quality non-English annotations.

The multilingual SFT data comprises: 2.4% human annotations, 44.2% data from other NLP tasks (reformatted into dialog), 18.8% rejection sampled data, and 34.6% translated reasoning data. The paper makes a deliberate choice about translations: "We try to avoid using machine-translated data to finetune the model in order to prevent translationese... or possible name bias, gender bias, or cultural bias." The one exception is "translated synthetic quantitative reasoning data" where the "simple nature of the language in these math problems" means translations have minimal quality issues.

For rejection sampling in multilingual contexts, the paper uses "multilingual-specific checks to ensure high language-match rate between the prompt and response" — preventing situations where a Hindi prompt gets an English response, or where a romanized prompt gets a response in the native script.

**Math and Reasoning (Section 4.3.3)**

The paper identifies five specific challenges in math reasoning: lack of diverse prompts at high complexity, lack of ground-truth chain-of-thought solutions, incorrect intermediate steps in model-generated solutions, teaching models to use external tools, and discrepancy between training (where the model sees correct solutions) and inference (where it must generate them).

The approach combines several techniques:

- **Step-wise reasoning trace generation:** Llama 3 generates multiple solutions per prompt, filtered based on whether the final answer is correct. Self-verification — where Llama 3 checks whether a step-by-step solution is valid — further filters out invalid traces.
- **Reward models for step filtering:** Both outcome and stepwise reward models (Lightman et al., 2023; Wang et al., 2023a) filter training data where intermediate steps are incorrect.
- **Monte Carlo Tree Search (MCTS):** For challenging prompts, MCTS with learned step-wise reward models generates valid reasoning traces (Xie et al., 2024).
- **Interleaved code and text reasoning:** The model generates solutions combining textual reasoning with Python code (Gou et al., 2023). Code execution serves as a feedback signal — if the code runs and produces the right answer, the reasoning chain is valid.
- **Error correction:** Incorrect generations are used as training data by prompting Llama 3 to correct them, simulating the feedback loop between incorrect attempts and corrections (An et al., 2023b; Welleck et al., 2022).

**Long Context (Section 4.3.4)**

The paper finds that "naively applying our existing SFT recipe with only short-context data resulted in significant regressions in long-context capabilities from pre-training." However, human annotation of long-context examples is impractical due to the "tedious and time-consuming nature of reading lengthy contexts." The solution is predominantly synthetic data:

- **Question answering:** Long documents from pre-training are split into 8K-token chunks, and Llama 3 generates QA pairs conditioned on randomly selected chunks. During training, the full document is provided.
- **Summarization:** Hierarchical summarization — first summarize 8K-token chunks, then summarize the summaries — with QA pairs that require global understanding of the full document.
- **Code reasoning:** Parse Python files to identify key dependencies (files imported by at least 5 other files). Remove one key file and prompt the model to identify which files depend on it and generate the missing code.

The paper finds that mixing just "0.1% of synthetically generated long-context data with the original short-context data optimizes the performance across both short-context and long-context benchmarks." This is remarkably data-efficient — a tiny fraction of long-context examples is sufficient to maintain the capability.

For DPO, "using only short context training data in DPO did not negatively impact long-context performance as long as the SFT model is high quality in long context tasks," likely because DPO has fewer optimizer steps than SFT.

**Tool Use (Section 4.3.5)**

Llama 3 is trained to use a search engine (Brave Search), a Python interpreter, and a mathematical computational engine (Wolfram Alpha). Tools are implemented as Python objects with methods; zero-shot tools are Python functions with descriptions, and function definitions/calls are formatted as JSON.

The training data collection is iterative and complexity-progressive:

1. **Single-step:** Synthetic user prompts requiring one tool call are generated, the model produces the tool call, it's executed, the output is added to context, and the model generates a final answer. About 30% of this data is filtered out due to unexecutable tool calls or formatting issues.
2. **Multi-step:** Synthetic data teaches multi-step tool use where the model interleaves reasoning steps and tool calls (similar to ReAct; Yao et al., 2022).
3. **File uploads:** Annotations cover 12 file types and tasks like summarization, bug finding, code optimization, and data analysis.

Human annotations are collected at the message level for tools, since tool dialogs contain multiple assistant messages (tool call, reasoning about output). Annotators provide preferences between two assistant messages with the same context, or edit one if both have problems. The chosen/edited message is added to the context and the dialog continues — this provides feedback on both tool calling and reasoning about tool outputs.

For zero-shot tool use (function calling), synthetic data is generated by mining the Stack (Kocetkov et al., 2022): extract function definitions and calls, clean and filter them, and use Llama 3 to generate a natural language query corresponding to each function call. Multi-turn function calling data uses multiple Llama 3 agents collaborating in a step-by-step manner, generating diverse domains, APIs, and queries.

**Factuality (Section 4.3.6)**

The paper takes a "hallucination-first approach" with the principle that "post-training should align the model to 'know what it knows' rather than add knowledge" (Gekhman et al., 2024).

The knowledge probing technique:

1. Extract a data snippet from pre-training data.
2. Generate a factual question about the snippet by prompting Llama 3.
3. Sample multiple responses from Llama 3.
4. Score correctness using the original context as reference and Llama 3 as judge.
5. Score informativeness using Llama 3 as judge.
6. For responses that are consistently informative but incorrect, generate a refusal response.

This data trains the model to answer when it has knowledge and refuse when it doesn't, rather than hallucinating. Additionally, "a limited set of labeled factuality data" is collected for sensitive topics where pre-training data may contain contradictory or incorrect statements.

**Steerability (Section 4.3.7)**

Steerability is the ability to direct model behavior through system prompts. The data collection asks annotators to design customized system prompts (e.g., "You are a helpful and cheerful AI Chatbot that acts as a meal plan assistant...") and then evaluate model consistency in following these instructions across multi-turn conversations. This data is used in reward modeling, rejection sampling, SFT, and DPO to improve steerability.

---

#### Data Processing and Quality Control for Post-Training

Since most post-training data is model-generated (via rejection sampling or synthetic generation), it requires careful cleaning.

**Rule-based cleaning:** In early rounds, the paper observed "undesirable patterns... such as excessive use of emojis or exclamation points" and implemented "rule-based data removal and modification strategies." For overly apologetic tone, overused phrases like "I'm sorry" or "I apologize" are identified and their proportion is balanced in the dataset.

**Model-based quality scoring:**

- **Topic classification:** Llama 3 8B finetuned as a topic classifier, categorizing data into coarse (e.g., "mathematical reasoning") and fine-grained (e.g., "geometry and trigonometry") buckets.
- **Quality scoring with dual signals:** The reward model scores each sample (top quartile = high quality). Separately, Llama 3 rates each sample on a 3-point scale for general English (accuracy, instruction following, tone/presentation) and 2-point scale for code (bug identification, user intention). The RM and Llama-based scores have "high disagreement rates," so samples marked as high quality by either are selected — combining signals achieves "the best recall on our internal test set."
- **Difficulty scoring via Instag and Llama-based rating:** Instag (Lu et al., 2023) prompts Llama 3 70B to tag intentions in SFT prompts — more intentions indicates more complexity. Llama 3 also rates difficulty on a 3-point scale.

**Semantic de-duplication:** Complete dialogs are clustered using RoBERTa embeddings, sorted within each cluster by `quality_score × difficulty_score`, and greedily selected by iterating through sorted examples and keeping only those with maximum cosine similarity below a threshold to previously selected examples in the cluster. This removes near-duplicate conversations while preserving the highest-quality, most difficult examples.

**A Final Note on the "Expert" Pattern**

A recurring pattern across capabilities is the use of **expert models — branched from pre-training and further specialized — to bootstrap data collection**. The code expert, multilingual expert, and (implicitly) the math-focused models serve as higher-quality sources of synthetic data and better human annotations than the general model would produce. Once the main model catches up in capability (through training on this higher-quality data), the expert may no longer be needed. This pattern — using specialized models to generate training data for a general model — is a form of distillation without explicitly labeling it as such, and it's a key mechanism for how Llama 3 achieves broad capability coverage without requiring all capabilities to be equally represented in the base pre-training data.

## 4. Key Insights and Innovations

### Innovation 1: Data Quality, Not Architectural Novelty, Is the Primary Lever for Frontier-Level Performance

The dominant narrative in the field circa 2023–2024 was that architectural innovation — mixture-of-experts (Mixtral, Arctic), novel attention mechanisms, retrieval augmentation — was the path to improved language model performance. Llama 3 makes a forceful counter-argument through its design choices and empirical results: **a standard dense Transformer, trained on meticulously curated data at sufficient scale, can match or exceed architecturally more complex models.** This is not merely a claim about engineering pragmatism; it is a methodological thesis about where the returns to research investment lie.

What makes this intellectually distinctive is the *explicitness* and *comprehensiveness* of the data-centric argument. The paper doesn't just say "better data helps" — it documents a multi-stage data curation pipeline that spans URL-level de-duplication, line-level dedup aggressive enough to remove some legitimate text, heuristic filters for boilerplate and adult content, model-based quality classification using DistilRoberta trained on Llama 2 annotations, domain-specific pipelines for code and math, and scaling-law-guided data mix optimization that arrives at a striking allocation: 42% of pre-training tokens are specialized (25% math/reasoning, 17% code), not general web text. This is a concrete, replicable methodology, not a vague aspiration.

The empirical evidence for the thesis comes from direct comparisons: Llama 3 70B (dense) outperforms Mixtral 8×22B (MoE with far more effective parameters) on most benchmarks in Table 2. On MMLU, Llama 3 70B achieves 83.6% vs. Mixtral's 76.9%; on HumanEval, 80.5% vs. 75.6%; on MATH, 68.0% vs. 54.1%. These gaps — 6.7, 4.9, and 13.9 percentage points respectively — are not marginal. The paper explicitly states that Llama 3 "outperforms these models, suggesting that dense architectures are not the limiting factor" (Section 9.1). This is a reframing of the scaling conversation: the limiting factor is not model capacity (which MoE increases) but the quality and composition of the training data and the effectiveness of the post-training pipeline.

The significance goes beyond the specific architecture choice. It implies that the field's intense focus on architectural innovation may be misallocated — that better data curation, filtering, and mix optimization, combined with systematic post-training, yields larger returns than designing more complex model structures. This is a fundamental diagnostic move: it redirects attention from *how the model computes* to *what the model learns from.* The paper's finding that markdown formatting is harmful and should be stripped, that line-level dedup improves performance despite removing some good text, that the annealing phase with upsampled high-quality data boosts benchmarks — these are empirical insights that would not emerge from an architecture-centric research program.

### Innovation 2: Iterative, Capability-Orchestrated Post-Training as a Systematic Scaling Methodology

Prior work on aligning language models — including Llama 2 — treated post-training largely as a single-phase process: collect preference data, train a reward model, run RLHF or DPO, and ship. Llama 3 transforms this into a **six-round iterative pipeline where each round's improved model generates better training data for the next round, and where capability development (code, math, multilinguality, tool use, long context, factuality, steerability) is explicitly orchestrated rather than treated as an emergent property of general alignment.**

What makes this distinctive is the *feedback loop* between model capability and data quality. Each round of post-training uses the best current model to (a) generate synthetic data via rejection sampling, (b) serve as a "model-as-judge" for data quality filtering, and (c) bootstrap more complex annotation protocols (tool use annotation progresses from single-turn to multi-turn to multi-step; long-context annotation is replaced by synthetic data because human annotation is impractical at those lengths). This is not merely "train on more data" — it is a dynamic process where the model's improving capabilities are leveraged to improve the training data itself, creating a virtuous cycle.

The paper's use of **expert models** (code expert, multilingual expert) as data-generation engines is conceptually novel. These are models branched from the main pre-training run, further specialized on domain-specific data, and then used to generate higher-quality synthetic data and collect better human annotations for the general model. This is a form of distillation — knowledge is transferred from the specialist to the generalist — but it is not framed as distillation in the traditional sense (where a larger teacher trains a smaller student). Instead, the expert serves as a *temporary capability scaffold* that enables high-quality data collection in domains where the general model is still weak. Once the general model catches up through training on this data, the expert may no longer be needed.

The evidence that this iterative approach matters is distributed throughout Sections 4.2 and 4.3, but a key indicator is the progressive complexification of annotation protocols: "as Llama 3 gradually improves through its development, we progressively complexify our human annotation protocols." This would be impossible without the iterative structure — you cannot ask annotators to evaluate multi-step tool use if the model cannot perform basic tool use. The iteration enables the annotation complexity to track the model's capability growth.

The theoretical significance is that it redefines "post-training" from a fixed recipe applied to a fixed model to a *developmental process* that co-evolves the model and its training data. This is analogous to curriculum learning, but applied at the scale of human annotation design and synthetic data generation strategy, not just example ordering. It also explains why the paper's post-training pipeline outperforms simpler alternatives: a single round of DPO on a fixed dataset would not benefit from the data quality improvements that each iteration enables.

### Innovation 3: Verifier Over-Optimization as a First-Class Phenomenon in Code and Reasoning Data Generation

While the paper does not frame this as its primary contribution, one of the most operationally significant insights emerges from the synthetic data generation pipelines for code and math: **model-generated training data must be validated against execution feedback or ground-truth answers, not merely against the model's own judgment, because models are poor at self-verifying correctness.** This is a diagnostic finding about the limits of model-as-judge approaches that has implications far beyond Llama 3.

The code pipeline (Section 4.3.1) provides the clearest evidence. The paper reports that when Llama 3 405B generates its own synthetic code solutions, training on these solutions "is not helpful (and can even degrade performance)." The solution is to introduce execution feedback: static analysis (parser + linter), unit test generation and execution in a containerized environment, and iterative self-correction based on error output. The paper reports that "about 20% of solutions were initially incorrect but self-corrected" — meaning that without execution feedback, one-fifth of the synthetic training data would have been wrong. The model cannot reliably tell which of its own solutions are correct without external verification.

This is not obvious. There is a substantial body of work on "self-improvement" and "self-play" where models generate their own training data and are assumed to improve. The paper's finding — that a 405B model generates incorrect solutions that it cannot self-diagnose, and that training on these solutions degrades performance — is a negative result with significant implications. It suggests that execution feedback (or equivalent ground-truth signals) is not a nice-to-have but a necessity for synthetic data quality in reasoning domains.

The math pipeline (Section 4.3.3) reinforces this through its multi-layered filtering: solutions are filtered based on correct final answers, then self-verified by Llama 3 for validity, then further filtered by outcome and stepwise reward models to remove incorrect intermediate steps. The paper explicitly identifies "incorrect intermediate steps" as a core challenge — model-generated chains of thought often contain reasoning errors even when the final answer is right, and these errors can propagate during training.

The broader significance is that this finding **constrains the scalability of purely model-driven data generation.** There is a growing research program around using LLMs to generate their own training data (Self-Instruct, Alpaca, WizardLM), implicitly assuming that model outputs are high-quality enough to serve as training targets. Llama 3's results suggest this assumption breaks down in precisely the domains where capability improvements are most valuable (code, math, reasoning) — the model can generate plausible-looking but incorrect outputs, and training on these outputs is actively harmful. The implication is that scalable synthetic data generation requires *independent verification mechanisms* (code execution, formal proof checkers, ground-truth answer matching) that the model itself cannot substitute for.

The "model-as-judge" filtering used for code quality (Section 4.3.1) provides further nuance: the paper initially used Llama 3 to score code on correctness and style, retaining only perfect scores, but found this "led to a regression in downstream benchmark performance, primarily because it disproportionately removed examples with challenging prompts." The model-as-judge was biased against difficulty — it gave low scores to correct but complex solutions. The solution was to strategically revise these challenging examples rather than filter them out, but the underlying problem — that model-based quality assessment is systematically biased — is a general concern for any pipeline that relies on LLMs to evaluate their own or other models' outputs.

### Innovation 4: Safety as a Capability-Specific, Language-Specific, and System-Level Engineering Discipline

Prior safety work on language models (including Llama 2) often treated safety as a uniform property: train on safety data, measure violation rates on English benchmarks, deploy. Llama 3's safety approach (Section 5.4) represents a conceptual shift toward **safety as a heterogeneous, capability-specific property that does not transfer across languages or modalities without explicit, targeted intervention.**

The key diagnostic finding is that **safety does not transfer.** The paper states: "safety knowledge in English does not readily transfer to other languages, particularly given the nuance of safety policies and language-specific context." This is not a minor observation — it fundamentally challenges the assumption that a model trained to be safe in English will generalize safety behavior to Hindi, Thai, or Spanish. The practical consequence is that safety data must be collected and models must be evaluated separately for each supported language, with language-specific understanding of cultural context and policy nuance. Figure 19 shows that violation rates and false refusal rates vary significantly across languages even after safety fine-tuning, and that "the distribution of safety data per language significantly impacts performance."

The capability-specific nature of safety is equally important. The paper identifies unique attack surfaces for each new capability:

- **Long-context models** are vulnerable to many-shot jailbreaking (Anil et al., 2024), where providing many examples of unsafe behavior in the context window can override safety training. The paper develops "a scalable mitigation strategy that significantly reduces VR, effectively neutralizing the impact of longer context attacks even for 256-shot attacks." This attack vector simply doesn't exist for short-context models.

- **Tool use** introduces risks like "forcing tool use with specific input strings, fragmented or encoded text" to trigger violating tool inputs, and "unsafe tool chaining" where one violating tool call among several benign ones leads to harmful outputs. The paper's red teaming identified these as distinct from text-only jailbreaks.

- **Code interpreter abuse** is a specific concern where "Llama 3 405B [is] particularly susceptible by complying with malicious prompts 10.4% of the time" compared to 3.8% for the 70B model — a counterintuitive finding where the larger, more capable model is *more* susceptible to certain safety failures.

The system-level safety architecture (Llama Guard 3, Prompt Guard, Code Shield) is conceptually significant because it challenges the model-centric safety paradigm. Rather than expecting the language model itself to be perfectly safe, the paper argues for a layered approach where input and output classifiers provide additional protection that is configurable per deployment. Table 25 shows that Llama Guard 3 reduces violations by 38–86% depending on language and capability, while Table 26 shows per-category violation reductions across 13 hazard categories. This enables developers to "deploy Llama Guard 3 for specific harms only, enabling control over the violations and false refusals trade-off at the harm category level."

The conceptual move is from "make the model safe" to "build a safety system around the model." This is analogous to the shift in cybersecurity from perimeter defense to defense-in-depth — recognizing that no single layer of protection is sufficient and that different layers can be optimized for different threats. The open-source release of these system-level components is itself an innovation: it enables the research community to inspect, critique, and improve safety mechanisms that are typically hidden behind API endpoints.

The uplift testing for cybersecurity and chemical/biological weapons (Section 5.4.5) represents an additional conceptual contribution: **measuring safety not by absolute risk but by *uplift* — the additional risk introduced by the model compared to existing available technologies.** This framing matters because absolute risk assessments can be misleading. A model might appear dangerous on a benchmark of harmful prompts, but if the same information is already available via web search, the model's incremental risk may be negligible. The paper's finding of "no significant uplift" for both novice and expert cyberattackers, and "no significant uplift" for chemical/biological weapon planning, is a more nuanced safety claim than "the model sometimes generates unsafe content" — it asserts that the model does not materially increase capabilities beyond what is already publicly accessible.

## 5. Experimental Analysis

### Evaluation Methodology

- **Dataset.** The primary evaluation benchmarks span eight top-level categories (Table 8): commonsense reasoning (CommonSenseQA, PiQA, SiQA, OpenBookQA, WinoGrande), knowledge (MMLU, MMLU-Pro, AGIEval, BIG-Bench Hard), reading comprehension (SQuAD V2, QuaC, RACE), math/reasoning (GSM8K, MATH, ARC Challenge, DROP, WorldSense), long context (QuALITY, many-shot GSM8K), code (HumanEval, MBPP), adversarial evaluations (Adv SQuAD, Dynabench SQuAD, GSM-Plus, PAWS), and aggregate benchmarks. For post-trained models (Table 16), additional benchmarks cover general knowledge (MMLU, MMLU-Pro, IFEval), math/reasoning (GSM8K, MATH, GPQA, ARC-Challenge), code (HumanEval, MBPP, HumanEval+, MBPP EvalPlus, MultiPL-E), multilinguality (MGSM, Multilingual MMLU), tool-use (Nexus, API-Bank, API-Bench, BFCL), and long context (ZeroSCROLLS, Needle-in-a-Haystack, InfiniteBench). The paper also includes proficiency exams (LSAT, SAT, GMAT, AP, GRE) and human evaluations with ~7,000 prompts spanning six individual and three multi-turn capabilities. For multimodal experiments, image benchmarks include MMMU (val, 900 images), VQAv2 (test-dev), AI2 Diagram (test), ChartQA (test), TextVQA (val), and DocVQA (test); video benchmarks include PerceptionTest (test, 11.6K QA pairs), TVQA (val, 15K QA pairs), NExT-QA (test, 9K questions), and ActivityNet-QA (test, 8K QA pairs); speech benchmarks include MLS, LibriSpeech, VoxPopuli, FLEURS (34 languages), Covost 2 (15 languages), and MuTox for safety.

- **Base models.** The Llama 3 family comprises three dense Transformer variants at 8B, 70B, and 405B parameters (Table 3), all using grouped query attention with 8 key-value heads, RoPE positional embeddings with θ=500,000, and a 128K-token vocabulary. The 405B model was pre-trained on 15.6T tokens using 3.8 × 10²⁵ FLOPs, while the 8B and 70B models were trained far beyond compute-optimality (Section 3.2.1) to trade training compute for inference efficiency. Models are evaluated in both pre-trained (base) and post-trained (instruction-tuned) configurations. For pre-training evaluations, the paper compares against Mistral 7B, Gemma 7B, Mixtral 8×22B, GPT-4, Nemotron 4 340B, and Gemini Ultra. For post-training evaluations (Table 2), comparisons include Gemma 2 9B, Mistral 7B, Mixtral 8×22B, GPT-3.5 Turbo, GPT-4 (0125), GPT-4o, Claude 3.5 Sonnet, and Nemotron 4 340B.

- **Metrics.** The primary metrics are task-specific: accuracy for multiple-choice benchmarks (MMLU, ARC Challenge, CommonSenseQA), exact match or F1 for reading comprehension (SQuAD, QuaC, DROP), pass@1 for code generation (HumanEval, MBPP) measuring the fraction of problems where at least one of N generated solutions passes all unit tests, BLEU score for speech translation, word error rate for speech recognition, and win rate for human evaluations (computed as the fraction of comparisons where annotators rate one model as "better" or "much better" on a 7-point scale). For safety evaluations, the paper uses Violation Rate (VR) — fraction of adversarial prompts where the model produces a policy-violating response — and False Refusal Rate (FRR) — fraction of borderline prompts where the model incorrectly refuses to answer. For benchmark variance, 95% confidence intervals are reported assuming Gaussian-distributed scores: CI(S) = 1.96 × √(S × (1−S) / N) where S is the observed score and N is the sample size.

- **Baselines.** For pre-trained model comparisons (Tables 9–14), baselines include Mistral 7B (Jiang et al., 2023), Gemma 7B (Team et al., 2024), Mixtral 8×22B (Jiang et al., 2024), GPT-4 (OpenAI, 2023a), Nemotron 4 340B, and Gemini Ultra (Google, 2023). For post-trained comparisons (Table 2), additional baselines include Gemma 2 9B, GPT-3.5 Turbo, GPT-4 (0125), GPT-4o, Claude 3.5 Sonnet, and Nemotron 4 340B. For multimodal evaluations, comparisons include GPT-4V, GPT-4o, Gemini 1.0/1.5 Pro, Claude 3.5 Sonnet, Whisper, and SeamlessM4T v2. For safety evaluations, three anonymized competitor systems are used: two end-to-end API-accessed systems and one open-source language model evaluated internally.

- **Generation budget / compute accounting.** For pre-training evaluations, models are compared at fixed parameter-scale equivalence classes (8B, 70B, 405B) with the best publicly reported or internally reproduced scores used for each competitor. For pre-training scaling laws, compute budgets range from 6 × 10¹⁸ to 10²² FLOPs, with model sizes between 40M and 16B parameters. Pre-training FLOPs for the 405B model are reported as 3.8 × 10²⁵. For post-training, each round involves K = 10-30 outputs for rejection sampling. For inference efficiency (Section 6), throughput and latency are measured at 4,096 input tokens and 256 output tokens, with comparisons between BF16 (16 GPUs across 2 machines) and FP8 quantization (8 GPUs on a single machine). For multimodal experiments, video inference uses 16 frames during pre-training and up to 64 frames during SFT, with frames uniformly sampled.

- **Cross-validation / statistical protocol.** For pre-training contamination analysis (Section 5.1.4), the paper follows Singh et al. (2024) using 8-gram overlap with dataset-specific thresholds TD selected to maximize significant estimated performance gain across three model sizes. For benchmark variance, 95% confidence intervals are reported for all main results. For human evaluations, pairwise comparisons use a 7-point scale with 95% confidence intervals, excluding ties from win rate calculations. The human evaluation prompt set (~7,000 prompts) was collected by a separate team and not accessible to model developers, with prompts uniformly distributed across subcategories within each capability and stratified by difficulty (10% easy, 30% medium, 60% hard). For safety evaluations, internal benchmarks of over 4,000 prompts per capability or language are used, with a mix of single-turn and multi-turn prompts. Post-training data undergoes decontamination via exact match with benchmark prompts. For multimodal evaluations, all video benchmarks use zero-shot evaluation without including any benchmark data in training or fine-tuning.

### Main Quantitative Results

#### Pre-Trained Language Model Performance

**Headline results for 8B and 70B models (Figure 12).** The pre-trained Llama 3 8B outperforms Mistral 7B and Gemma 7B across virtually every capability category. On the per-category averages shown in Figure 12, Llama 3 8B leads competitors in commonsense, knowledge, math/reasoning, reading comprehension, and code. Llama 3 70B outperforms Llama 2 70B by a large margin on most benchmarks except saturated commonsense tasks, and also outperforms Mixtral 8×22B.

**Detailed benchmark results (Tables 9–14).** On reading comprehension (Table 9), Llama 3 8B achieves 77.0 ±0.8 on SQuAD, 44.9 ±1.1 on QuaC, and 54.3 ±1.4 on RACE. Llama 3 70B achieves 81.8 ±0.7, 51.1 ±1.1, and 59.0 ±1.4 respectively. Llama 3 405B reaches 81.8 ±0.7 on SQuAD, 53.6 ±1.1 on QuaC, and 58.1 ±1.4 on RACE — competitive with or exceeding Mixtral 8×22B (84.1, 44.9, 59.2).

On code generation (Table 10), Llama 3 8B achieves 37.2 ±7.4 on HumanEval and 47.6 ±4.4 on MBPP, outperforming Mistral 7B (30.5, 47.5) and Gemma 7B (32.3, 44.4). Llama 3 70B reaches 58.5 ±7.5 and 66.2 ±4.1, substantially ahead of Mixtral 8×22B (45.1, 71.2). Llama 3 405B achieves 61.0 ±7.5 and 73.4 ±3.9, trailing GPT-4 (67.0) and Gemini Ultra (74.4) on HumanEval but competitive on MBPP.

On commonsense understanding (Table 11), all models perform at high levels, with Llama 3 405B achieving 85.8 ±2.0 on CommonSenseQA, 85.6 ±1.6 on PiQA, 53.7 ±2.2 on SiQA, 49.2 ±4.4 on OpenBookQA, and 82.2 ±1.8 on WinoGrande.

On math and reasoning (Table 12), Llama 3 405B achieves 89.0 ±1.7 on GSM8K, 53.8 ±1.4 on MATH, 96.1 ±1.1 on ARC-C, 84.8 ±0.7 on DROP, and 63.7 ±0.3 on WorldSense — the standout being the MATH score, which substantially exceeds GPT-4 (no score reported in the table, and Llama 3 70B achieves 41.4 ±1.4 vs. Mixtral 8×22B's 41.8 ±1.4) and Gemini Ultra (53.2). Llama 3 8B achieves 57.2 ±2.7 on GSM8K and 20.3 ±1.1 on MATH, both ahead of Mistral 7B (52.5, 13.1) and competitive with Gemma 7B (46.4, 24.3 — Gemma leads on MATH by 4 points).

On general benchmarks (Table 13), Llama 3 405B achieves 85.2 on MMLU, 61.6 on MMLU-Pro, 71.6 ±1.8 on AGIEval, and 85.9 ±0.8 on BB Hard. This trails GPT-4 on MMLU (86.4) but exceeds Nemotron 4 340B (81.1) and Gemini Ultra (83.7). Llama 3 70B achieves 79.3 on MMLU, exceeding Mixtral 8×22B (77.8).

On long-context tasks (Table 14), Llama 3 405B achieves 87.6 ±1.4 on QuALITY (5-shot) and 90.0 ±5.9 on GSM8K (16-shot), while Llama 3 70B achieves 82.8 ±1.6 and 83.0 ±7.4, and Llama 3 8B achieves 56.0 ±2.1 and 60.0 ±9.6.

#### Robustness Analysis

**Robustness to MCQ design choices (Figures 13–14).** The paper evaluates robustness of pre-trained models to four types of design variation in the MMLU benchmark. For label variants (Figure 13, left), performance is remarkably stable across five label formats — canonical letters (A. B. C. D. and A) B) C) D)), numerical (1 2 3 4), common symbols ($ & # @), and rare symbols (œ § з ü) — with Llama 3 405B showing micro accuracy varying by only a few percentage points across formats. For few-shot label bias (Figure 13, right), the model maintains strong performance regardless of whether few-shot examples all share the same label (AAAA), all have different labels (ABCD), or show two labels each (AADD, BBCC). For answer order (Figure 14, left), performance is consistent across permutation distances — even when answer option labels are systematically remapped, Llama 3 405B accuracy stays within a narrow band. For prompt format (Figure 14, right), five different task prompts (ranging from minimal to those asserting model expertise) produce highly consistent results, particularly for the 405B model. The paper notes: "This robustness is particularly pronounced for the 405B parameter model."

#### Adversarial Benchmarks

**Adversarial vs. non-adversarial performance (Figure 15).** The paper plots adversarial benchmark scores against their non-adversarial counterparts for question answering (Adversarial SQuAD, Dynabench SQuAD vs. SQuAD), mathematical reasoning (GSM-Plus vs. GSM8K), and paraphrase detection (PAWS vs. QQP). For paraphrase detection, both pre-trained and post-trained models show near-parity — performance on PAWS is close to performance on QQP, indicating robustness to the type of adversarial word scrambling in PAWS. The paper notes this "marks a substantial step with respect to the previous generation of models." For mathematical reasoning and question answering, however, adversarial performance is substantially lower than non-adversarial performance. This pattern holds for both pre-trained and post-trained models. The diagonal parity line in Figure 15 shows all datapoints for QA and math falling significantly below parity, while paraphrase detection points cluster near the diagonal.

#### Contamination Analysis

**Estimated contamination and performance gain (Table 15).** The paper reports the percentage of evaluation data considered contaminated (via 8-gram overlap with pre-training corpus) and the estimated performance gain from that contamination. Key findings: AGIEval shows 98% contamination with an estimated 8.5–19.9 percentage point gain across model sizes; BIG-Bench Hard shows 95% contamination with 26.0–41.0 point estimated gain; HellaSwag shows 85% contamination with ~14.8 point gain; PiQA shows 55% contamination with 7.9–8.5 point gain; NaturalQuestions shows 52% contamination with only 0.8–1.6 point gain — illustrating that high contamination does not uniformly translate to large performance gains. For SQuAD and MATH, low thresholds yield high contamination percentages but zero performance gain, suggesting "contamination is either not helpful for these datasets, or that a larger n is required." For MBPP, HumanEval, MMLU, and MMLU-Pro, 8-gram overlap gives such high contamination scores that a clean performance gain estimate is impossible — requiring alternative detection methods.

#### Post-Trained Language Model Performance

**General knowledge and instruction following (Table 2).** On MMLU (5-shot), Llama 3 8B achieves 69.4, exceeding Gemma 2 9B (72.3) — though Gemma leads here — and Mistral 7B (61.1). Llama 3 70B reaches 83.6, ahead of Mixtral 8×22B (76.9) and GPT-3.5 Turbo (70.7). Llama 3 405B achieves 87.3, outperforming GPT-4 (85.1) and Nemotron 4 340B (82.6), but trailing GPT-4o (89.1) and Claude 3.5 Sonnet (89.9). On MMLU-Pro (5-shot CoT), Llama 3 405B reaches 73.3 vs. GPT-4's 64.8, GPT-4o's 74.0, and Claude 3.5 Sonnet's 77.0. On IFEval, Llama 3 8B achieves 80.4 (vs. Gemma 2 9B's 73.6 and Mistral 7B's 57.6); Llama 3 70B achieves 87.5 (vs. Mixtral's 72.7); Llama 3 405B achieves 88.6, leading all competitors including GPT-4 (84.3), GPT-4o (85.6), and Claude 3.5 Sonnet (88.0).

**Code generation (Table 18, 19).** On HumanEval (0-shot), Llama 3 8B achieves 72.6 ±6.8, substantially ahead of Gemma 2 9B (54.3) and Mistral 7B (40.2). Llama 3 70B reaches 80.5 ±6.1, outperforming Mixtral 8×22B (75.6) and GPT-3.5 Turbo (68.0). Llama 3 405B achieves 89.0 ±4.8, trailing GPT-4o (90.2) and Claude 3.5 Sonnet (92.0) but ahead of GPT-4 (86.6). On HumanEval+, Llama 3 405B achieves 82.3 ±5.8, matching Claude 3.5 Sonnet (82.3). On MBPP EvalPlus (base), Llama 3 405B reaches 88.6 ±3.2, second to Claude 3.5 Sonnet (90.5). On MultiPL-E for non-Python languages (Table 19), Llama 3 405B achieves HumanEval scores of 82.0 (C++), 80.4 (Java), 76.4 (PHP), 81.1 (TypeScript), 54.4 (C#), and 57.6 (Shell) — significantly higher than the 70B and 8B variants across all languages. The gap between Python and non-Python languages is substantial for all models: Llama 3 405B drops from 89.0 on Python HumanEval to 54.4 on C# and 57.6 on Shell, highlighting the imbalance in training data across programming languages.

**Math and reasoning (Table 2).** On GSM8K (8-shot CoT), Llama 3 8B achieves 84.5 (vs. Gemma 2 9B's 76.7 and Mistral 7B's 53.2). Llama 3 70B reaches 95.1, substantially ahead of Mixtral 8×22B (88.2) and GPT-3.5 Turbo (81.6). Llama 3 405B achieves 96.8, leading all models in the table — GPT-4 scores 94.2, GPT-4o scores 96.1, Claude 3.5 Sonnet scores 96.4. On MATH (0-shot CoT), Llama 3 405B achieves 73.8, second only to GPT-4o (76.6) and ahead of Claude 3.5 Sonnet (71.1), GPT-4 (64.5), and the open-source Nemotron 4 340B (41.1). Llama 3 70B achieves 68.0 on MATH, more than doubling Mixtral 8×22B's 54.1. On GPQA (0-shot CoT), Llama 3 405B achieves 51.1, trailing Claude 3.5 Sonnet (59.4) by a significant margin but competitive with GPT-4o (53.6). On ARC Challenge (0-shot), Llama 3 405B achieves 96.9, matching GPT-4o (96.7), Claude 3.5 Sonnet (96.7), and GPT-4 (96.4).

**Multilingual benchmarks (Table 20).** On MGSM (0-shot CoT, average across languages), Llama 3 8B achieves 68.9 (vs. Gemma 2 9B's 53.2 and Mistral 7B's 29.9). Llama 3 70B achieves 86.9, dominating Mixtral 8×22B (71.1) and GPT-3.5 Turbo (51.4). Llama 3 405B achieves 91.6, tying Claude 3.5 Sonnet (91.6) and ahead of GPT-4o (90.5) and GPT-4 (85.9). On Multilingual MMLU (5-shot, average across 7 languages), Llama 3 405B achieves 83.2, behind GPT-4o (85.5) but ahead of GPT-4 (80.2). Llama 3 70B achieves 78.2, far exceeding Mixtral 8×22B (64.3).

**Long-context benchmarks (Table 21).** On ZeroSCROLLS QuALITY (exact match), Llama 3 405B achieves 95.2 ±9.1, matching GPT-4 (95.2). On Qasper (F1), Llama 3 405B achieves 49.8 ±18.5, slightly behind GPT-4 (50.5). On InfiniteBench En.QA (F1), Llama 3 405B achieves 30.5 ±4.8, substantially ahead of GPT-4 (15.7), GPT-4o (19.1), and Claude 3.5 Sonnet (11.3). On InfiniteBench En.MC (accuracy), Llama 3 405B achieves 83.4 ±4.8, leading all models including GPT-4 (72.0) and GPT-4o (82.5). On Multi-needle (average recall across 10 sequence lengths), Llama 3 405B achieves 98.1 ±1.5, with GPT-4 and GPT-4o at 100.0, and Claude 3.5 Sonnet at 90.8. Llama 3 405B demonstrates "perfect needle retrieval performance, successfully retrieving 100% of needles at all document depths and context lengths" on the standard Needle-in-a-Haystack task.

**Tool-use performance (Table 22).** On Nexus (function calling accuracy), Llama 3 405B achieves 58.7 ±4.1, leading GPT-4 (50.3) and GPT-4o (56.1) but trailing Claude 3.5 Sonnet (45.7 — notably lower, despite Claude's strength on other benchmarks). On API-Bank, Llama 3 405B achieves 92.3 ±2.6, competitive with Claude 3.5 Sonnet (92.6). On API-Bench, Llama 3 405B achieves 35.3 ±2.2, behind GPT-4o (41.4) and Claude 3.5 Sonnet (60.0). On BFCL, Llama 3 405B achieves 88.5 ±1.5, second to Claude 3.5 Sonnet (90.2) and slightly ahead of GPT-4 (88.3). For tool-use human evaluations (Figure 16), Llama 3 405B "significantly beats" GPT-4o on text-only code execution tasks and plot generation, but "lags behind on the file upload use case."

**Proficiency exams (Table 17).** Llama 3 405B's performance on human exams is summarized as "very similar to Claude 3.5 Sonnet and GPT-4 4o." On the LSAT, Llama 3 405B scores 81.1 ±3.8 vs. Claude 3.5 Sonnet's 80.0 ±3.9 and GPT-4o's 77.4 ±4.1. On the GRE Verbal, Llama 3 405B scores 166 (out of 170), matching GPT-4o and Claude 3.5 Sonnet (both 167). On AP exams averaged across 21 subjects, Llama 3 405B achieves 93.5 ±1.9 vs. GPT-4o's 93.0 ±2.0 and Claude 3.5 Sonnet's 92.2 ±2.1. Llama 3 70B's performance is described as "even more impressive" — it achieves 74.2 on the LSAT (vs. GPT-3.5 Turbo's 54.3 and Nemotron 4 340B's 73.7), 79.6 on AP English Lit (vs. GPT-3.5's 53.7), and 87.9 average across AP subjects (vs. GPT-3.5's 70.2).

**Human evaluations (Figure 17).** Comparing Llama 3 405B to GPT-4 (0125), win rates are within margin of error on nearly all capabilities, with Llama 3 winning on multiturn reasoning (28.0% win, 18.0% loss) and multiturn coding (24.1% win, 20.5% loss) but losing on multilingual (15.8% win, 31.1% loss). Against GPT-4o, results are mixed: Llama 3 wins on English (27.4% vs. 23.6% loss) but loses on coding (18.2% win, 24.8% loss), reasoning (22.0% win, 30.1% loss), and multilingual (15.4% win, 34.7% loss). Against Claude 3.5 Sonnet, Llama 3 wins on English (26.0% win, 24.0% loss) and multiturn English (28.0% win, 18.9% loss), is competitive on multilingual (27.4% win, 16.0% loss), but trails on coding (20.8% win, 24.3% loss), reasoning (22.4% win, 28.5% loss), and multiturn coding (20.8% win, 20.5% loss). The paper notes that "model performance in human evaluations is heavily influenced by nuanced factors such as model tone, response structure, and verbosity."

#### Safety Results

**Overall safety performance (Figures 19–21).** Llama 3 405B with Llama Guard achieves very competitive violation rates while keeping false refusal rates low. Figure 21 plots the overall VR-FRR tradeoff across capabilities: Llama 3 405B + Llama Guard (system-level) achieves low VR and moderate FRR, while the standalone Llama 3 405B model has lower FRR but higher VR. Compared to competitor systems: "[System] Comp. 1" achieves very low VR but at the cost of very high FRR; "[System] Comp. 2" shows moderate VR and FRR; Llama 3 405B + Llama Guard occupies a favorable position with competitive VR and lower FRR than Comp. 1.

**Multilingual safety (Figure 19).** Across English, French, German, Hindi, Italian, Portuguese, Spanish, and Thai, Llama 3 405B + Llama Guard achieves violation rates below 0.05 for most languages (English ~0.02, German ~0.02, French ~0.04) while false refusal rates vary: English ~0.1, German ~0.15, Thai ~0.3. Without Llama Guard, Llama 3 405B shows higher VR (English ~0.05, Thai ~0.12) but lower FRR (English ~0.02, Thai ~0.1). Compared to competitor systems, Llama 3 405B + Llama Guard is "at least as safe, if not strictly safer" than both competing systems, while maintaining competitive FRR.

**Long-context safety (Figure 20).** For DocQA, Llama 3 405B + Llama Guard achieves VR ~0.02 and FRR ~0.05; Llama 3 405B alone achieves VR ~0.06 and FRR ~0.03; Comp. 1 achieves VR ~0.12 and FRR ~0.08; Comp. 2 achieves VR ~0.08 and FRR ~0.02. For Many-shot, Llama 3 405B + Llama Guard achieves VR ~0.02, Llama 3 405B alone achieves VR ~0.04, Comp. 1 achieves VR ~0.10, and Comp. 2 achieves VR ~0.13. The paper reports that Llama 3 is "Pareto-better than the Comp. 2 system across both violation rates and false refusal rates" on both DocQA and Many-shot.

**Tool-use safety (Figure 20).** For Tool Usage (Search), Llama 3 405B achieves VR ~0.01 and FRR ~0.08, while Comp. 1 achieves VR ~0.08 and FRR ~0.02 — Llama 3 is "significantly safer, though has a slightly higher false refusal rate." Only Llama 3 and Comp. 1 are compared for this capability.

**Llama Guard 3 per-language and per-category analysis (Tables 25–26).** Table 25 shows Llama Guard 3 reduces violations by 38–86% across languages when used for input or output filtering, at the cost of increased FRR. For English, full Llama Guard reduces VR by 86% while increasing FRR by 102%. For German: VR -77%, FRR +37%. For Thai: VR -51%, FRR +39%. Table 26 provides per-category breakdown for English: full Llama Guard achieves 100% VR reduction for Defamation, Elections, Intellectual Property, Sexual Content, and Non-Violent Crimes; 91% reduction for Hate; 80% for Violent Crimes; 59% for Child Sexual Exploitation; with an overall FRR increase of 102%.

**Prompt Guard performance (Table 28).** Prompt Guard achieves 99.9% TPR and 0.4% FPR on in-distribution jailbreaks (AUC 0.997), 97.5% TPR and 3.9% FPR on out-of-distribution jailbreaks (AUC 0.975), and 91.5% TPR and 5.3% FPR on multilingual jailbreaks (AUC 0.959). On indirect injections from CyberSecEval, it achieves 71.4% TPR and 1.0% FPR (AUC 0.996).

**CyberSecEval results (Section 5.4.5).** Llama 3 405B complies with malicious code interpreter prompts 10.4% of the time (vs. 3.8% for Llama 3 70B). Text-based prompt injection attacks succeed 21.7% of the time against Llama 3 405B. Figure 22 shows prompt injection success rates across models and strategies: Llama 3 405B averages 0.22 success rate across 15 strategies, compared to GPT-4 Turbo at 0.18 and Gemini Pro at 0.17. For spear phishing (Figure 23), Llama 3 405B achieves persuasiveness scores of 2.95 (malware download), 2.60 (security info gathering), 1.68 (data theft), 1.68 (credential theft). The autonomous attack framework evaluation found that "Llama 3 70B and 405B efficiently identify network services and open ports... [but] fail to effectively use this information to gain initial access."

**Uplift testing (Section 5.4.5).** For cyber attacks, a study with 62 volunteers (31 expert, 31 novice) found "both novices and experts using the 405B model demonstrated insignificant uplift over having open access to the internet without an LLM." For chemical/biological weapons, a study with teams of two participants in six-hour scenarios found "no significant uplift in performance related to usage of the Llama 3 model" across aggregate analysis, subgroup breakdowns, and separate evaluation of chemical vs. biological scenarios.

**FP8 quantization impact (Figures 25–27).** Figure 26 shows the reward score distribution for 100,000 responses under BF16 and FP8 inference: the distributions are nearly indistinguishable, demonstrating that the FP8 quantization approach "has very limited impact on the model's responses." Figure 27 shows throughput-latency tradeoffs: FP8 inference achieves "throughput improvements of up to 50% during the pre-fill stage, and a substantially better throughput-latency trade-off during decoding" compared to two-machine BF16 inference.

#### Multimodal Results

**Image recognition (Table 29).** Llama 3-V 405B achieves 64.5 on MMMU (val, CoT), behind GPT-4o (69.1) and Claude 3.5 Sonnet (68.3) but ahead of GPT-4V (56.4) and Gemini 1.5 Pro (62.2). On VQAv2 (test-dev), Llama 3-V 405B scores 80.2, matching Gemini 1.5 Pro (80.2). On AI2 Diagram, Llama 3-V 405B achieves 94.1, competitive with GPT-4o (94.2) and Gemini 1.5 Pro (94.4), and slightly behind Claude 3.5 Sonnet (94.7). On ChartQA, Llama 3-V 405B scores 85.8, ahead of GPT-4V (78.4) and competitive with GPT-4o (85.7), but behind Claude 3.5 Sonnet (90.8) and Gemini 1.5 Pro (87.2). On TextVQA, Llama 3-V 405B achieves 84.8, substantially ahead of GPT-4V (78.0) and Gemini 1.5 Pro (78.7). On DocVQA, Llama 3-V 405B scores 92.6, matching GPT-4o (92.8) but behind Claude 3.5 Sonnet (95.2). The paper notes Llama 3 405B "appears particularly competitive on document understanding tasks" and that it "outperform[s] GPT-4V on all benchmarks."

**Video recognition (Table 30).** Llama 3-V 70B achieves 60.8 on PerceptionTest (test), ahead of Gemini 1.0 Pro (51.1) and Gemini 1.0 Ultra (54.7). On TVQA (val), Llama 3-V 70B scores 87.9, slightly ahead of GPT-4V (87.3). On NExT-QA (test), Llama 3-V 70B achieves 30.3, competitive with Gemini 1.0 Ultra (29.9). On ActivityNet-QA (test), Llama 3-V 70B scores 56.3, ahead of Gemini 1.0 Pro (49.8) and Gemini 1.0 Ultra (52.2), but behind GPT-4o (61.9) and Gemini 1.5 Pro (57.5). The paper emphasizes that these results are achieved by "training a small video adapter during post-training" and are "very competitive, and in some cases even better, than other models that potentially leverage native multimodal processing all the way from pre-training."

**Speech recognition (Table 31).** Llama 3 70B achieves word error rates of 4.4 on MLS English, 3.1 on LibriSpeech test-other, 5.7 on VoxPopuli English, and 8.2 on FLEURS (34 languages). These outperform Whisper (v2/v3: 6.2, 4.9, 7.0, 14.4 respectively) and SeamlessM4T v2 (6.5, 6.2, 7.0, 11.7). Llama 3 performs similarly to Gemini on MLS English (Gemini 1.0 Ultra: 4.4, Gemini 1.5 Pro: 4.2). Llama 3 8B shows slightly higher error rates (4.9, 3.4, 6.2, 9.6) but still outperforms Whisper and SeamlessM4T.

**Speech translation (Table 32).** On FLEURS (33 languages → English), Llama 3 8B achieves 29.5 BLEU and Llama 3 70B achieves 33.7, compared to Whisper v2's 21.9 and SeamlessM4T v2's 28.6. On Covost 2 (15 languages → English), Llama 3 8B achieves 34.4 and Llama 3 70B achieves 38.8, compared to Whisper v2's 33.8 and SeamlessM4T v2's 37.9.

**Speech safety (Table 33).** On MuTox, Llama 3 70B achieves the lowest added toxicity for English (0.68% AT) with 15.46% LT (lost toxicity — the percentage of toxic inputs where the model responds safely). Across all 21 evaluated languages, Llama 3 70B averages 2.00% AT and 10.29% LT. Llama 3 8B shows 0.84% AT and 15.09% LT for English, 2.31% AT and 9.89% LT overall. Gemini 1.5 Pro achieves 1.44% AT and 13.42% LT for English, 2.06% AT and 10.94% LT overall. The paper notes that "the percentage of added toxicity is very low: our speech models have the lowest percentage of added toxicity for English, with less than 1%."

**Speech generation (Tables 34–35).** For text normalization (Table 34), the model with Llama 3 8B embeddings and 3-token right context achieves 90.7% accuracy, compared to 73.6% without Llama 3 embeddings (same 3-token context) and 88.0% without embeddings but with full bidirectional context. For prosody modeling (Table 35), the Llama 3 8B PM is preferred 60.0% of the time over the streaming phone-only baseline, and 63.6% of the time over the non-streaming phone-only baseline.

#### Inference Efficiency

**Micro-batching effect (Figure 24).** Using two micro-batches with pipeline parallelism, prefill latency at batch size 8 increases from ~800ms to ~1,200ms, while prefill throughput increases from ~6,000 to ~8,000 tokens/sec. For decoding, at batch size 64, latency increases from ~20ms to ~30ms per token, while throughput increases from ~80 to ~120 tokens/sec. The paper reports that "micro-batching still leads to a better throughput-latency trade-off."

### Ablation Studies and Robustness Checks

**Annealing data effect on benchmarks:** The paper reports that annealing on GSM8k and MATH training sets improved Llama 3 8B performance by 24.0% and 6.4% respectively on the corresponding validation sets. However, "the improvements on the 405B model are negligible, suggesting that our flagship model has strong in-context learning and reasoning capabilities and does not require specific in-domain training samples to obtain strong performance" (Section 3.1.3). This is a revealing scale-dependent finding: larger models benefit less from domain-specific annealing data.

**Data mix adjustments during pre-training:** The paper made several adjustments to the pre-training data mix during training — increasing non-English data to improve multilingual performance, upsampling mathematical data, adding more recent web data, and downsampling subsets identified as lower quality (Section 3.4.1). While the exact performance impact of each adjustment is not quantified in isolation, the paper reports that the overall recipe was "very stable: we observed few loss spikes and did not require interventions to correct for model training divergence." This stability at 3.8 × 10²⁵ FLOPs scale is itself a non-trivial finding about the robustness of the training methodology.

**Document-level attention masking:** The paper finds that preventing cross-document attention "had limited impact during standard pre-training, but find it to be important in continued pre-training on very long sequences" (Section 3.2). This ablation distinguishes the mask's effect at different context lengths — at 8K, it matters little, but at 128K with many documents packed into one sequence, it's essential.

**Markdown removal:** "We experimentally evaluate different cleaning configurations. We find markdown is harmful to the performance of a model that is primarily trained on web data compared to plain text, so we remove all markdown markers" (Section 3.1.1). This negative result — that structured formatting degrades model performance — is a concrete finding about data preprocessing that might not be obvious a priori.

**Line-level de-duplication tradeoff:** The paper explicitly acknowledges a quality-quantity tradeoff: "manual qualitative analysis showed that the line-level de-duplication removes not only leftover boilerplate... but also frequent high-quality text, [yet] our empirical evaluations showed strong improvements" (Section 3.1.1). This confirms that aggressive dedup, despite removing some good data, is net beneficial.

**PRM vs. ORM training label comparison (implicit in code expert training):** Although not framed as a formal ablation, the paper reports that training Llama 3 405B on its own generated synthetic code data "is not helpful (and can even degrade performance)" — execution feedback is necessary to make self-generated data useful (Section 4.3.1). This is a critical negative result about the limits of model self-improvement without external verification.

**Code model-as-judge filtering:** The paper initially filtered code training data using Llama 3 as a judge (binary scores for correctness and style, keeping only perfect scores of 2). This "led to a regression in downstream benchmark performance, primarily because it disproportionately removed examples with challenging prompts" (Section 4.3.1). The finding that model-based quality filtering is biased against difficulty is an important negative result with implications beyond code.

**System prompt steering for code:** Figure 9 demonstrates qualitatively that code-specific system prompts improve code quality — adding comments, using more informative variable names, saving memory — compared to generation without system prompts. This ablation supports the inclusion of system prompt steering as a mechanism for improving rejection-sampled code data quality.

**Long-context SFT data proportion:** The paper finds that "mixing 0.1% of synthetically generated long-context data with the original short-context data optimizes the performance across both short-context and long-context benchmarks" (Section 4.3.4). The finding that only 0.1% long-context data is needed to maintain long-context capabilities while preserving short-context performance is a precise, non-obvious result with practical implications for data mix design.

**DPO without long-context data:** The paper observes that "using only short context training data in DPO did not negatively impact long-context performance as long as the SFT model is high quality in long context tasks" (Section 4.3.4). This suggests that SFT is the critical stage for maintaining long-context capabilities, and DPO can safely use short-context data — an important finding for efficient post-training pipeline design.

**Rejection sampling temperature for multilingual:** The paper explored "randomly choosing the temperature hyperparameter from the range 0.2–1 for diverse generations in early rounds of post-training" but found that "with high temperature, responses for multilingual prompts can get creative and inspiring, but are also susceptible to unnecessary or unnatural code-switching." In the final round, "a constant value of 0.6" was used (Section 4.3.2). This demonstrates that temperature tuning for multilingual data involves a different tradeoff than for English — higher diversity comes at the cost of language consistency.

**Model size and safety data requirements (Figure 18):** Figure 18 shows the VR-FRR tradeoff frontier for Llama 3 8B vs. 70B: "smaller models require a larger proportion of safety data relative to helpfulness, and... it is more challenging to efficiently balance VR and FRR compared to larger models." This is a scale-dependent safety finding with deployment implications — smaller models need proportionally more safety training to achieve comparable safety performance.

**int8 quantization of Llama Guard 3 (Table 27):** Quantization reduces model size by more than 40% with "negligible impact on the performance." For English output classification: non-quantized achieves 0.947 precision, 0.931 recall, 0.939 F1, 0.040 FPR; quantized achieves 0.947, 0.925, 0.936, 0.040. For multilingual: non-quantized 0.929/0.805/0.862/0.033; quantized 0.931/0.785/0.851/0.031. For tool use: non-quantized 0.774/0.884/0.825/0.176; quantized 0.793/0.865/0.827/0.155.

**FP8 quantization design choices (Section 6.2):** Three specific design choices were ablated: (1) not quantizing first and last Transformer layers; (2) upper bounding dynamic scaling factors to 1200 to prevent underflow from high-perplexity tokens; (3) row-wise quantization (Figure 25) rather than tensor-wise, computing scaling factors across rows. The paper notes that "evaluations on standard benchmarks often suggest that FP8 inference performs on par with BF16 inference even without these mitigations" but that "when scaling factors are not upper bounded, the model occasionally produces corrupted responses even though the benchmark performance is strong." Figure 26 validates the final approach by showing BF16 and FP8 reward score distributions are nearly identical for 100,000 responses.

**Vision encoder multi-layer features:** The paper employs "multi-layer feature extraction, where features from the 4th, 8th, 16th, 24th and 31st layers are also provided in addition to the final layer features" because "image encoders trained via a contrastive text alignment objective are unable to preserve fine-grained localization information" (Section 7.2). This design choice — which increases the representation dimension — is justified by the finding that it improves performance "especially in domains such as text recognition."

**Vision adapter weight averaging:** For image SFT, the paper finds that "averaged models consistently yield better results compared to the best individual model found via grid search" and that "this strategy reduces sensitivity to hyperparameters" (Section 7.5.2). This finding mirrors the pre-training annealing averaging and post-training model averaging, establishing weight averaging as a recurring pattern across training stages.

**Vision reward model freezing:** The vision RM is trained with the language RM's self-attention layers kept frozen: "freezing the language RM part generally leads to better accuracy, especially on tasks that require the RM to judge based on its knowledge or the language quality" (Section 7.5.4). This finding — that the reward model benefits from frozen language backbone weights — contrasts with the image adapter training where the LLM is also frozen, but the vision encoder is unfrozen.

**Speech model training without LLM finetuning:** The paper explicitly notes that "unlike most prior work, we opt to not finetune the language model itself for speech tasks as doing so may lead to contention on non-speech tasks. We find that at larger model scales, strong performances are attainable even without such finetuning" (Section 9.2). This is supported by the competitive ASR and speech translation results in Tables 31–32, which are achieved with a frozen LLM.

**Prosody model with/without Llama 3 embeddings (Table 35):** The Llama 3 8B PM is preferred 60.0% over a streaming phone-only baseline and 63.6% over a non-streaming phone-only baseline — quantifying the benefit of incorporating language model embeddings for prosody prediction.

### Critical Assessment

The experimental evaluation in this paper is extraordinarily comprehensive in scope — spanning pre-training benchmarks across 8 categories, post-training benchmarks across 7 capabilities, proficiency exams, human evaluations, safety evaluations across languages and capabilities, uplift testing, and multimodal evaluations for vision, video, and speech. The sheer volume of evaluation (Tables 2, 8–35, Figures 2–27) provides genuinely broad coverage. However, this breadth comes with specific structural limitations that affect how thoroughly each individual claim is tested.

**Claim: "Llama 3 405B performs on par with GPT-4 across a variety of tasks."** The evidence for this claim is genuinely strong but domain-dependent. On MMLU (Table 2), Llama 3 405B achieves 87.3 vs. GPT-4's 85.1 — a clear win. On HumanEval, 89.0 vs. 86.6 — another win. On GSM8K, 96.8 vs. 94.2. On ARC Challenge, 96.9 vs. 96.4. These show Llama 3 405B outperforming GPT-4 on several established benchmarks. However, on MMLU-Pro (0-shot CoT), Llama 3 405B achieves 73.3 vs. GPT-4's 64.8 — a substantial gap that the paper doesn't explain. The comparison is complicated by the fact that the paper compares against GPT-4 (0125 API version), not GPT-4o or Claude 3.5 Sonnet, for the "on par" claim, while Table 2 shows these newer models outperforming Llama 3 405B on most metrics (MMLU: 89.1, 89.9 vs. 87.3; GPQA: 53.6, 59.4 vs. 51.1). The human evaluations (Figure 17) provide the most balanced picture: Llama 3 405B vs. GPT-4 shows win rates within margin of error on most capabilities, supporting "on par" — but with specific losses on multilingual tasks. The claim is best supported for English-language reasoning, code, and knowledge tasks, and less supported for multilingual capabilities and GPQA-level reasoning.

A significant limitation is the single model family: all results are for PaLM... no, for Llama 3 models. The paper compares against GPT-4, Claude, and Gemini via API access, but these comparisons are asymmetric — Meta has full access to Llama 3 internals for evaluation design, while API-only access to competitors may not surface optimal prompting strategies. The paper's "best score between computed and reported" approach is reasonable but introduces variance from evaluation methodology differences.

**Claim: "Smaller models are best-in-class."** This is well-supported. Llama 3 8B vs. Gemma 2 9B and Mistral 7B (Table 2): Llama 3 8B leads on IFEval (80.4 vs. 73.6, 57.6), HumanEval (72.6 vs. 54.3, 40.2), GSM8K (84.5 vs. 76.7, 53.2), and MATH (51.9 vs. 44.3, 13.0). Llama 3 70B vs. Mixtral 8×22B (Table 2): Llama 3 70B leads on MMLU (83.6 vs. 76.9), HumanEval (80.5 vs. 75.6), GSM8K (95.1 vs. 88.2), and MATH (68.0 vs. 54.1). The gaps are substantial across virtually every benchmark. The proficiency exam results (Table 17) further reinforce this: Llama 3 70B achieves 87.9 average on AP exams vs. GPT-3.5 Turbo's 70.2, and beats Nemotron 4 340B (which has ~5× more parameters) on many tests. The "best-in-class" claim for the small model size classes is robust.

**Claim: "Safety knowledge does not transfer across languages."** The paper's internal benchmarks and Figure 19 provide evidence that violation rates and false refusal rates vary by language, but this is shown for the final model, not as a controlled ablation of "train on English safety data, test on multilingual." The paper states that safety data was collected separately per language, so the variation by language in Figure 19 doesn't directly demonstrate lack of transfer — it could reflect differences in the quantity or quality of language-specific safety data. The red teaming observation that "mixing multiple languages in one prompt or conversation can easily lead to more violating outputs" provides qualitative evidence of cross-lingual safety weaknesses, but a formal ablation (e.g., training on English safety data only and measuring multilingual VR) is absent. The claim is plausible and supported by related findings (the red teaming, the need for language-specific data), but not experimentally isolated.

**Claim: "Uplift testing shows no significant uplift for cyber or CBRN risks."** The uplift studies are described in substantial detail (Section 5.4.5), but the actual quantitative results are reported at a high level without full statistical detail. For cyber attacks: "both novices and experts using the 405B model demonstrated insignificant uplift over having open access to the internet without an LLM" — the sample sizes (31 per cohort) and specific completion rates are not provided. For CBRN: "quantitative analysis... show no significant uplift" — the expert scoring methodology, sample sizes, and statistical tests are described qualitatively but the actual scores are not reported in the paper. These studies represent a genuine effort to move beyond benchmark-based safety evaluation, but the opacity of the reported results makes independent assessment difficult. This is understandable given the sensitivity of the subject matter, but it limits how strongly the "no significant uplift" claim can be validated from the paper alone.

**Missing experiments that would have strengthened the paper:**

- **Ablation on the number of post-training rounds:** The paper uses six rounds. What happens with 2, 4, or 8? This is a major design choice with no supporting ablation. The iterative data collection is described as central to quality, but the marginal benefit of additional rounds is unknown from the presented results.

- **Data mix ablations at scale:** The final data mix (50% general, 25% math, 17% code, 8% multilingual) is justified by scaling law experiments, but the actual scaling law data for different mixes is not shown. The paper mentions doing this but doesn't present the IsoFLOPs curves for different mix candidates. The reader cannot assess how sensitive performance is to the specific proportions.

- **Comparison against a compute-optimally trained larger model for the FLOPs-matched analysis:** Unlike the reference example paper which performed a careful FLOPs-matched pretraining vs. test-time compute comparison, this paper does not include a systematic FLOPs-matched comparison of different model scales trained with identical data and methodology. The comparison against other model families (Nemotron, Mixtral) confounds architecture, data, and training methodology differences. A FLOPs-matched comparison within the Llama 3 family would isolate the effect of scale under controlled conditions.

- **Single-turn vs. multi-turn evaluation breakdown:** While multi-turn human evaluations are reported (Figure 17), most benchmark evaluations are single-turn. The chat format's effectiveness in multi-turn scenarios is primarily assessed through human evaluation rather than through established multi-turn benchmarks, which limits quantitative comparison with competitors.

- **Multimodal integration ablations:** The compositional approach (freezing the LLM during vision and speech adapter training) is presented as a design choice, but no ablation compares this against joint fine-tuning. The paper asserts that "unlike most prior work, we opt to not finetune the language model itself for speech tasks as doing so may lead to contention on non-speech tasks" — but this tradeoff is not empirically demonstrated. What happens to text-only performance if the LLM is partially unfrozen during speech adapter training?

- **Statistical significance for human evaluations:** The human evaluation comparisons (Figure 17) show many win rates where the margin between win and loss percentages is within the reported confidence intervals (e.g., Llama 3 vs. GPT-4 on English: 25.0% win, 24.2% loss — the difference of 0.8 points is well within any reasonable CI for ~7,000 prompts). The paper acknowledges these CIs but doesn't explicitly test which differences are statistically significant, making many of the capability-level comparisons difficult to interpret as true wins or losses.

**Conditional nature of claims:**

The "on par with GPT-4" claim is conditional on: (a) comparing against GPT-4 (0125), not GPT-4o or Claude 3.5 Sonnet; (b) focusing on English-language reasoning, code, and knowledge tasks; (c) accepting that Llama 3 performs worse on multilingual tasks and some reasoning benchmarks (GPQA). The "best-in-class for open models" framing is unambiguous — even against Nemotron 4 340B, Llama 3 405B leads on all reported benchmarks. The safety claims are conditional on the specific internal benchmarks and red teaming procedures used, which are not reproducible externally. The multimodal results are clearly identified as preliminary and "still under development and not yet ready for release."

**Test set size concerns:** The main benchmark evaluations use standard test sets (MMLU: ~14K questions, HumanEval: 164 problems, GSM8K: 1,319 problems), but the human evaluation uses ~7,000 prompts across 9 capabilities (~800 per capability), and safety benchmarks use ~4,000 prompts per capability/language. These are reasonable sample sizes for the conclusions drawn, but the confidence intervals reported (particularly for HumanEval, where N=164) can be quite wide — the ±7.5 on Llama 3 405B's HumanEval score of 89.0 means the true pass@1 could plausibly be anywhere from ~81.5 to ~96.5.

**Overall assessment:** The experimental evaluation supports the paper's central narrative — that Llama 3 405B is competitive with the closed-source frontier and that the smaller models lead their size classes — while the safety evaluation demonstrates a comprehensive, multi-layered approach. The primary limitations are (1) the absence of key ablations (post-training rounds, data mix sensitivity, multimodal integration tradeoffs), (2) the limited visibility into the uplift testing methodology and results, and (3) the confounding of model, data, and training methodology when comparing across model families. The paper's transparency about evaluation methodology, including detailed descriptions of benchmark selection, prompt formats, shot counts, and contamination analysis, partially mitigates these limitations by enabling the community to replicate and extend the evaluations.

## 6. Limitations and Trade-offs

### 6.1 The Difficulty Estimation Cost Is Unaccounted For in the Headline Efficiency Gains

The paper’s compute-optimal scaling strategy relies on estimating the difficulty of each prompt before allocating the inference budget. The current method for doing this — generating 2,048 samples per question and averaging either ground-truth correctness (oracle bins) or the PRM’s final-answer score (predicted bins) — is extraordinarily expensive. As the authors acknowledge in Section 3.2:

> "estimating difficulty in this way still incurs additional computation cost during inference... our experiments do not account for this cost largely for simplicity"

This is not a minor caveat. Generating 2,048 samples per question consumes more compute than the largest test-time compute budgets studied (256–512 generations per problem). The reported `4×` efficiency gains over best-of-N are computed *after* difficulty is already known, without amortizing the cost of learning it. In a realistic deployment, the total cost would be `difficulty_estimation_cost + strategy_execution_cost`, and the former could dominate the latter, potentially eliminating or even reversing the efficiency advantage.

The paper provides no measurements of how the efficiency gains change when difficulty estimation cost is included, nor does it explore cheaper alternatives (e.g., estimating difficulty from a much smaller number of samples, or training a lightweight classifier to predict difficulty directly from the question text). The authors flag this as a key avenue for future work, but until it is addressed, the `4×` figure should be understood as an **upper bound on achievable efficiency** in idealized conditions, not a realized deployment gain. A practitioner deciding whether to adopt this method would need to know the total cost — including difficulty estimation — and the paper does not provide that number.

### 6.2 Hard Problems Remain Completely Unsolved

Across every method studied — PRM-guided search, iterative revision, and their compute-optimal combinations — the hardest questions (difficulty bin 5, where the base model’s pass@1 is near zero) show **near-zero improvement** regardless of the compute budget allocated. In Figure 3 (right), bin 5 accuracy hovers at 1–3% for all search methods and all budgets. In Figure 7 (right), bin 5 shows roughly 2–3% accuracy irrespective of the sequential-to-parallel ratio. In the FLOPs-matched comparison (Figure 9), the bin 5 scaling line is essentially flat near 0–5% for both revisions and PRM search.

This is a fundamental capability bound: test-time compute can amplify existing capability but cannot create it from nothing. If the base model’s pass@1 is near zero on a problem class, no amount of search or revision will help — there are no correct solutions in the proposal distribution to find or refine. The paper is transparent about this (Section 7 takeaway box):

> "test-time compute is powerful when problems are within the base model's reach... but it cannot compensate for fundamental capability gaps that larger pretraining would address"

The practical consequence is severe for deployment: if a user’s query distribution includes a meaningful fraction of genuinely hard problems (those outside the base model’s capability range), allocating additional inference compute will waste resources without improving outcomes. The compute-optimal strategy cannot help here — it can only tell you *not* to spend compute on these problems, which means the model simply fails on them. The paper demonstrates this boundary clearly, but it means the approach offers no path forward for problems that exceed the base model’s training distribution. For such problems, pretraining a larger or better-trained model remains the only viable option, and the paper’s `~14×` pretraining advantage on hard problems at `R ≫ 1` (Figure 9, where test-time compute shows -52.9% relative disadvantage for PRM search) confirms that test-time compute is not a substitute for fundamental capability on out-of-reach problems.

### 6.3 Revisions and Search Are Studied Independently, Not Combined

The paper studies two complementary axes for test-time computation — modifying the proposal distribution via iterative revisions (Section 6) and optimizing via PRM-guided search (Section 5) — but never combines them. Section 8 explicitly acknowledges this gap:

> "we did not experiment with PRM tree-search techniques in combination with revisions"

This is a significant limitation because the two mechanisms have **complementary, difficulty-dependent strengths**. Revisions improve the proposal distribution on easy problems where the model’s initial outputs are roughly correct and just need refinement — a local search in answer space. PRM-guided search helps on medium-hard problems where the model benefits from exploring qualitatively different solution strategies — a global search. The paper’s own analysis demonstrates that these two mechanisms are effective on different difficulty tiers, yet their potential synergy is unexplored.

The consequence is that the reported results represent a **lower bound** on what a fully integrated system could achieve. A system that uses the revision model as the proposal distribution within beam search — or that uses the PRM to guide which revision branches to pursue — could potentially break through the performance ceilings that each method individually hits. On medium-difficulty problems where both mechanisms show some benefit individually, their combination could yield gains beyond either alone. The paper’s current compute-optimal policy selects *between* these strategies per difficulty bin, but cannot exploit their complementary strengths on the same problem. This is a gap that future work must address to establish the true ceiling of test-time compute scaling.

### 6.4 The Single Benchmark and Single Model Family Limit Generality

All experiments in the paper use the MATH benchmark (Hendrycks et al., 2021) with PaLM 2-S* as the base model. The authors state they “believe this model is representative of the capabilities of many contemporary LLMs” (Section 4), but this claim is unverified and several aspects of the findings could be model-specific or domain-specific:

- **The PRM’s quality and over-optimization behavior** depend on PaLM 2-S*’s output distribution. A model with different calibration properties, different error patterns, or a different base capability level might exhibit qualitatively different difficulty-dependent scaling curves. The paper’s finding that beam search hurts performance on easy problems (Figure 3, right) is specifically a consequence of PRM over-optimization, which is a function of the verifier’s reliability — a stronger or weaker verifier would shift the difficulty thresholds at which different strategies become optimal.

- **The revision model’s ability to learn from incorrect in-context examples** depends on the base model’s in-context learning capabilities, which vary substantially across model families and scales. The paper’s edit-distance-based pairing strategy for revision training data (Section 6.1) is designed for PaLM 2-S* outputs — whether it transfers to other models with different output characteristics is unknown.

- **The MATH benchmark consists exclusively of competition-level math problems** requiring symbolic reasoning with clean, verifiable answers. It is unclear whether the difficulty-dependent patterns — beam search hurting easy problems, revisions helping easy problems, no method helping hard problems — generalize to other reasoning domains (code generation, logical reasoning, scientific QA) or to tasks requiring factual knowledge rather than multi-step inference. The paper provides no evidence on any benchmark other than MATH.

The test set of 500 questions, split into five difficulty quintiles of ~100 each, then further split by two-fold cross-validation, means the compute-optimal policy is selected based on approximately 50 questions per fold per bin. This is a small sample for strategy selection, and the paper does not report confidence intervals on the compute-optimal scaling curves, making it difficult to assess whether the observed differences between strategies at specific budget levels are statistically reliable or reflect noise from the small bin sizes.

### 6.5 Verifier Over-Optimization Is a Hard Ceiling, Not a Solved Problem

The paper documents verifier over-optimization as a central limiting factor: beam search degrades easy-problem performance at high budgets (Figure 3, right), lookahead search — the strongest optimizer — paradoxically performs *worst* overall (Figure 3, left), and qualitative examples in Appendix M show search producing degenerate outputs (repetitive low-information steps, overly short 1–2 step solutions) that score highly under the PRM but are incorrect. The compute-optimal policy *mitigates* this by routing easy problems away from aggressive search toward best-of-N, and by applying beam search only on medium-difficulty problems where the PRM signal has more room to provide genuine guidance. However, it does not *solve* the underlying problem.

On medium-difficulty problems where beam search is deployed, over-optimization still limits the scaling ceiling — the beam search curves in Figure 3 flatten and sometimes decline well before the budget is exhausted. At 256 generations, beam search (`M = 4`) reaches approximately 34% while best-of-N weighted reaches approximately 37–38% (Figure 3, left), suggesting that even on problems where search initially provides gains, the verifier’s reliability frontier caps further improvement. More aggressive search algorithms that should theoretically be more powerful (lookahead search) perform *worse* at equivalent budgets because their extra computation per step reduces the effective number of beams explored, and what remains over-optimizes faster.

This means the compute-optimal approach is fundamentally bounded by verifier quality. Improving the PRM — through better training data, adversarial robustness, ensemble methods, or calibration techniques — would likely shift the difficulty thresholds and change the optimal policy. The current results are therefore specific to the verifier quality achievable with the Monte Carlo rollout training procedure described in Appendix D, using PaLM 2-S*’s outputs. The paper does not explore how verifier improvements would alter the scaling landscape, nor does it provide guidance on how to train verifiers that remain reliable under more aggressive optimization — it identifies the problem but does not offer a solution beyond the adaptive routing that avoids the worst over-optimization regimes.

### 6.6 No Accounting for Latency or Wall-Clock Time

The paper measures test-time compute exclusively in "generations" — the number of complete solutions sampled — which is a reasonable proxy for total FLOPs but ignores **latency**. Sequential revisions are inherently serial: each revision depends on the output of the previous one, so a chain of 64 sequential revisions takes approximately `64×` the wall-clock time of a single generation. In contrast, parallel best-of-N with N=64 can (in principle) be executed simultaneously if sufficient hardware is available, with wall-clock time comparable to a single generation.

The compute-optimal policy frequently favors sequential or hybrid sequential-parallel strategies (Figures 7, 8), particularly on easy problems where sequential revision dominates. A strategy that allocates 128 generations as 64 sequential × 2 parallel takes roughly `64×` longer wall-clock time than one that runs 128 parallel samples simultaneously. The paper’s FLOPs-based accounting treats these as equivalent (both consume 128 generations of compute), but for any latency-sensitive application — interactive assistants, real-time decision-making, user-facing chatbots — the sequential strategies favored by the compute-optimal policy may be completely impractical regardless of their accuracy advantages.

The paper does not discuss this tradeoff. There is no analysis of how the compute-optimal policy changes if latency is constrained (e.g., "you have a budget of 128 generations but at most 4 sequential steps"), no measurement of the Pareto frontier between accuracy and wall-clock time, and no discussion of whether the efficiency gains hold in latency-constrained regimes. For a practitioner deciding how to deploy these methods, the distinction between FLOPs (which can sometimes be parallelized) and latency (which is constrained by serial dependencies) is critical, and the paper provides no guidance on navigating it.

## 7. Implications and Future Directions
- Field impact
  - Demonstrates that carefully trained dense open models can approach closed‑model performance at flagship scale while providing transparent methods (data curation, scaling laws, long‑context training, safety tuning, FP8 inference). This raises the baseline for open research and practical deployments.
- Follow‑up research
  - Long‑context: stronger reasoning over 100K+ tokens (e.g., improved summarization/QA training, retrieval‑augmented long‑context).
  - Safety: adaptive defenses against jailbreaking/prompt injection (e.g., proactive tool‑call validation, multi‑agent verification), improved multilingual/borderline calibration, and standardized contamination auditing.
  - Data: principled annealing and curriculum strategies; richer reasoning/code datasets with verified step traces; better tool‑grounded corpora (especially for file workflows).
  - Inference/systems: broader FP8 coverage (including attention), quantization‑aware training, and heterogeneous clusters; improved pipeline scheduling for interactive workloads.
  - Multimodality: broader release and scaling of adapters; unified training that preserves text without degradation; tighter integration of speech prosody/semantics and vision grounding for tool use.
- Applications
  - Enterprise assistants (analysis of long documents, spreadsheets, PDFs; Section 4.3.5 and Figure 11), developer tools (code gen/debug/review with execution feedback; Section 4.3.1), multilingual support (Table 20), STEM tutoring and exams (Table 17), research assistants (tool use + factuality probes; Sections 4.3.5–4.3.6), and safety‑aware platforms (Llama Guard 3, Prompt Guard; Section 5.4.7).

---

Below are selected, concrete references used in the analysis:
- Architecture & training: Figure 1; Sections 3.1–3.4; Tables 3–4; Figures 2–6; Table 5.
- Post‑training & capabilities: Figure 7; Sections 4.1–4.3; Figures 8–11; Tables 6–7.
- Headline performance: Table 2 (post‑trained), Tables 9–14 (pre‑trained).
- Long context: Section 3.4.2; Table 21.
- Tool use/function calling: Section 4.3.5; Table 22; Figures 10–11.
- Multilingual: Section 4.3.2; Table 20.
- Human evals: Section 5.3; Figures 16–17.
- Safety: Section 5.4; Figures 18–21; Tables 24–28.
- Inference: Section 6; Figures 24–27.
- Vision/speech: Sections 7–8; Tables 29–35; Figures 28–30.
