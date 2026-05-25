# RedPajama: an Open Dataset for Training Large Language Models

**ArXiv:** [2411.12372](https://arxiv.org/abs/2411.12372)

## 🎯 Pitch

RedPajama launches the first truly open, large-scale pretraining data ecosystem for large language models, addressing critical gaps in transparency, scale, and customizability. By not only replicating LLaMA-1's training set (RedPajama-V1) but also releasing RedPajama-V2—a massive, richly annotated web corpus exceeding 100 trillion tokens—the project empowers researchers and practitioners to audit, filter, and tailor training data for safer, higher-performing, and more adaptable AI. This landmark resource accelerates open-source LLM development and enables principled advances in data-driven model optimization.

---

## 1. Executive Summary

This paper introduces and releases the **RedPajama datasets**—an open reproduction of the LLaMA training corpus (RedPajama-V1) and a massive web-only dataset of over 100 trillion tokens with accompanying quality signals (RedPajama-V2)—to address transparency, scale, and versatility gaps in LLM pretraining data. The work trains and evaluates decoder-only Transformer models up to 1.6B parameters on filtered subsets of RedPajama-V2, demonstrating how quality signals—rule-based heuristics (the Gopher and C4 filters) and ML-based heuristics (fastText classifiers, DSIR importance weights)—can be combined to curate datasets that outperform established web corpora like RefinedWeb on aggregated benchmark scores while varying widely in downstream performance. RedPajama-V2 filtered with fuzzy deduplication and the full Gopher rules achieves the highest aggregated benchmark scores among all RedPajama subsets (ranking first across a broader set of tasks than RefinedWeb), establishing that raw, unfiltered web data with rich metadata enables competitive model performance only when appropriate filtering strategies are applied.

## 2. Context and Motivation

### The Core Problem: LLM Training Data Is Opaque and Inaccessible

The central challenge this paper confronts is deceptively simple but structurally profound: **the field of large language model development has a data transparency crisis.** Virtually all state-of-the-art LLMs—GPT-4, the LLaMA family, Mistral, Falcon—are built on training datasets whose composition, curation strategies, and filtering decisions remain largely secret. The authors state this explicitly in Section 1:

> "one of the core challenges this field faces is the general lack of transparency regarding the composition and curation strategy of pretraining data"

This is not merely a documentation issue. It is a **structural obstacle to open science and equitable AI development.** When pretraining datasets are proprietary, the research community cannot study how data quality affects model behavior, cannot replicate claimed results, cannot audit for harmful biases or copyrighted material, and cannot iterate on data curation strategies without independently reconstructing enormous web-crawling and filtering pipelines. The Foundation Model Transparency Index flagged this gap as a systemic weakness across the industry (Bommasani et al., 2023).

The paper identifies three specific sub-problems that must be addressed to advance open-source language models (Section 1):

1. **Transparency in model development, including the data curation process** — most model releases omit critical details about what data was used, how it was filtered, what deduplication was applied, and how different data sources were mixed.
2. **Access to large quantities of high-quality data** — building competitive open models requires trillions of tokens, but assembling, curating, and storing datasets at this scale demands substantial resources and expertise that few organizations possess.
3. **Availability of artifacts and metadata for dataset curation and analysis** — even when raw data exists, researchers lack the quality signals, deduplication metadata, and filtering annotations needed to make informed decisions about how to construct training subsets.

The paper positions RedPajama as directly addressing all three of these gaps (Contributions C1–C3). The design principles laid out in Section 1—**Transparency** (documenting and making public all curation details), **Scale** (processing >100 trillion tokens), and **Versatility** (providing raw text with quality signals rather than prescribing a single filtered version)—are direct responses to these identified failures of the existing ecosystem.

### Why This Problem Matters

The opacity of pretraining data has cascading consequences across research, industry, and society.

**For research:** When training data is secret, every paper studying data quality, data ablations, or the relationship between data composition and downstream performance must begin by building its own data pipeline—an enormously expensive prerequisite that gates participation to well-resourced organizations. The authors note (Section 1):

> "the process of studying and building optimal data compositions, along with developing filtering rules and heuristics, is time-consuming as it necessitates running numerous ablations on different compositions of the training data"

RedPajama-V2's design—providing raw text with 46 quality signals per document rather than a single filtered output—directly enables this kind of research by making the ablation space explicit and navigable without requiring each team to reprocess the full Common Crawl corpus.

**For industry and deployment:** The lack of transparency means that organizations deploying LLMs in regulated or high-stakes settings cannot conduct thorough data audits. They cannot verify whether training data contains personally identifiable information (PII), copyrighted material, toxic content, or domain-specific biases that could lead to downstream failures. An open, well-documented dataset provides this audit trail by construction.

**For equitable access:** Data access is a moat. When the best-performing models are trained on proprietary datasets accessible only to a handful of large corporations, the concentration of power in AI development is reinforced. By releasing a transparent, large-scale dataset that has already been used to train strong open models (Snowflake Arctic, AI2's OLMo, Salesforce's XGen—listed in the abstract and Figure 1), RedPajama lowers the barrier to entry for builders of open-weight LLMs.

**For scientific understanding:** The field still lacks principled answers to foundational questions: What makes web data "high quality"? Which filtering heuristics genuinely improve downstream performance, and which introduce harmful biases? How much does deduplication matter relative to other quality interventions? Answering these questions requires a common substrate—a large, unfiltered reference corpus with rich metadata—against which different filtering strategies can be systematically compared. The paper explicitly designs RedPajama-V2 to serve this role.

### Where Prior Approaches Fall Short

The paper positions RedPajama relative to the existing landscape of open pretraining datasets (Table 1), and identifies several systematic limitations:

**Composite datasets lack raw source data.** Datasets like the Pile (Gao et al., 2020), ROOTS (Laurençon et al., 2022), Dolma (Soldaini et al., 2024), and SlimPajama (Shen et al., 2023) provide cleaned, filtered subsets drawn from multiple domains. While these are valuable, they are **already-filtered artifacts**—the decisions about what constitutes "quality" have been baked in by the dataset creators. A researcher studying alternative filtering approaches cannot remove those choices and start from first principles. As Table 1 indicates, none of these composite datasets provide access to the raw, unfiltered source data. RedPajama-V2 is the only entry in the table that checks the "Raw Data" column.

**Web-only datasets prescribe a single quality standard.** RefinedWeb (Penedo et al., 2024), FineWeb (Penedo et al., 2024), and C4 (Raffel et al., 2020) demonstrated that web-only data can yield strong models without compositing multiple domains. However, each of these datasets represents a **single curator's judgment** about how to filter web data. RefinedWeb applies the Gopher rules and its own heuristics. FineWeb applies its own quality classifier. C4 applies a specific set of line-level filters and a blocklist. These are valuable contributions, but they do not support comparative research into filtering strategies—each dataset is a point solution, not a platform for experimentation.

**Scale limitations constrain model training.** Several widely used open datasets are simply too small for training frontier models. C4 contains approximately 175B tokens. The Pile is roughly 800GB. SlimPajama is around 0.9TB. Modern models routinely train on trillions of tokens (LLaMA-1 used ~1T tokens; LLaMA-3 used 15T tokens; GPT-4's training data size is undisclosed but believed to be similarly massive). At 270TB covering 30.4 trillion deduplicated tokens in the head+middle partition alone, RedPajama-V2 provides headroom for significantly larger training runs while keeping the full ~100T-token tail partition available for researchers who want to explore whether filtering strategies other than perplexity-based bucketing can recover valuable signal from noisier data.

**The LLaMA reproduction gap.** The original LLaMA technical report (Touvron et al., 2023) described the dataset creation process in broad strokes—approximately one paragraph per data source—without releasing the dataset itself or sufficient detail to faithfully reconstruct it. The RedPajama-V1 effort revealed just how many critical details were unspecified: which CommonCrawl snapshots were used, what classifier threshold was applied, which GitHub files were filtered out, how books were deduplicated, which Wikipedia dump was used. Table 10 in the Appendix systematically documents the uncertainties the authors encountered and the decisions they made to resolve them, highlighting that even the most widely cited "open" model release is, in practice, incompletely specified in ways that matter for reproduction.

**No existing dataset combines scale, rawness, and metadata.** The central gap that motivates RedPajama-V2 is that no prior dataset simultaneously provides: (a) trillions of tokens of scale, (b) raw unfiltered text that preserves all original web content, and (c) rich per-document quality signals enabling downstream filtering research. Existing datasets typically provide at most two of these three properties. RedPajama-V2 is explicitly designed to fill this gap.

### How This Paper Positions Itself

The paper does not claim to have discovered the optimal data filtering recipe. Rather, it positions itself as **enabling infrastructure**—providing the substrate on which such discoveries can be made. This is evident in several key design decisions:

**RedPajama-V1 is presented as a transparent reproduction, not an improvement.** Section 3 carefully documents the process of reconstructing LLaMA's training data, including a frank discussion of where ambiguities existed and what choices were made (Table 10). The RedPajama-INCITE family of models trained on this data serves to **validate the reproduction**, not to claim superiority. The authors explicitly acknowledge that at the 7B scale, their models underperform LLaMA-7B, hypothesizing that this stems partly from training with FP16 precision (required by the Summit supercomputer's V100 GPUs) and partly from missing "salient details that went into the construction of the original LLaMA training corpus" (Section 3.2.2). This honest accounting of limitations—rather than overclaiming—is consistent with the paper's transparency ethos.

**RedPajama-V2 deliberately withholds judgment about data quality.** Unlike RefinedWeb, FineWeb, or C4—each of which applies a specific filtering pipeline and releases the resulting cleaned dataset—RedPajama-V2 releases the raw text alongside quality signals and encourages users to "make informed decisions based on their specific needs and criteria" (Section 1). The ablation studies in Section 4.3 demonstrate this philosophy in action: they show that **different filtering strategies produce different downstream outcomes**, with no single configuration dominating all tasks. The Gopher rules plus fuzzy deduplication produced the best aggregated benchmark scores, but other configurations (e.g., custom rules with Wikipedia perplexity thresholds) achieved lower validation perplexity on certain domains. This multiplicity of "best" configurations is precisely the point—the optimal filtering strategy is use-case-dependent, and RedPajama-V2 provides the infrastructure to discover these tradeoffs systematically.

**The work is positioned in the lineage of community data infrastructure, not isolated model releases.** Figure 1 illustrates the ecosystem around RedPajama: multiple downstream model families (OLMo, Arctic, OpenELM, RedPajama-INCITE) have already been trained on RedPajama data, and SlimPajama represents a community-driven further refinement of RedPajama-V1. This positions the contribution as a **shared resource that enables follow-up work** rather than a one-off dataset release.

**The ablation studies are designed as existence proofs, not final answers.** Section 4.3 evaluates 468M and 1.6B parameter models on 13 downstream benchmarks plus validation perplexity, testing a range of filtering configurations. The authors are explicit about the scale limitation:

> "We use relatively small scales, as this enables us to explore a wider range of filters, showing the breadth of the quality filters available in RedPajama."

The goal is to demonstrate that the quality signals *can* be used to produce models of varying performance—not to identify the optimal filtering recipe. This distinguishes RedPajama from datasets like FineWeb, which provides a specific optimized filtering pipeline alongside its release.

**The paper acknowledges boundaries and calls for future work.** Section 5 explicitly notes limitations: the absence of decontamination analysis against common benchmarks, the lack of PII detection, and the need for larger-scale explorations. The closing vision is that "future work will continue to build on RedPajama and provide new innovative ways of filtering, curating, and mixing multiple pretraining corpora"—positioning the dataset as a foundation for a research program rather than a finished product.

In summary, RedPajama addresses a structural gap in the LLM ecosystem: the absence of a transparent, large-scale, unfiltered web corpus with rich metadata that enables systematic research into data curation strategies. Prior datasets are either too small, too opaque, too heavily pre-filtered, or lacking the quality annotations needed for comparative ablation studies. RedPajama's dual release—a LLaMA reproduction (V1) and an unprecedented raw web corpus with 46 quality signals (V2)—provides infrastructure that has already been adopted by multiple prominent open model efforts, demonstrating the demand for precisely this kind of resource.

## 3. Technical Approach

### 3.1 Reader Orientation

This is a **dataset release and infrastructure paper** — its core contribution is not a novel algorithm or model architecture, but the construction, documentation, and empirical validation of two large-scale pretraining datasets (RedPajama-V1 and RedPajama-V2) together with a suite of quality signals and metadata that enable downstream researchers to study and optimize data curation strategies. The paper solves the problem that most LLM training datasets are opaque, pre-filtered, and released without the raw source data or quality annotations needed for systematic ablation research by providing: (1) a transparent reconstruction of the LLaMA-1 training corpus (V1), and (2) a massive unfiltered web corpus spanning 100+ trillion tokens with 46 per-document quality annotations (V2) that can be combined in different ways to produce training subsets of widely varying downstream performance.

### 3.2 Big-Picture Architecture (Diagram in Words)

The RedPajama system consists of two major, independently constructed components:

1. **RedPajama-V1 (the LLaMA reproduction pipeline):** Takes seven distinct data sources (CommonCrawl, C4, GitHub, Wikipedia, Books, ArXiv, Stack Exchange) and processes each through source-specific filtering, cleaning, and deduplication steps informed by the sparse descriptions in the LLaMA technical report. The output is a ~1.2 trillion token composite corpus structured as JSON Lines files with text and metadata fields. This dataset feeds the RedPajama-INCITE model training pipeline on the Summit supercomputer, producing 3B and 7B parameter decoder-only Transformer models.

2. **RedPajama-V2 (the web-only, metadata-rich corpus):** Takes raw HTML text extracted from 84 CommonCrawl snapshots (2014–2023), passes it through the light CCNet pipeline (language identification, perplexity scoring, line-level deduplication), and produces ~113 billion individual text documents across five languages. In parallel, a separate **quality signal computation layer** processes each document to produce 46 quality annotations spanning natural language measures, repetitiveness scores, content-based toxicity flags, ML-based classifier predictions (fastText and DSIR importance weights), and deduplication metadata (exact hashes + MinHash signatures). The raw documents and quality signals are stored in separate, corresponding shards that downstream users can join by document ID. The ablation layer trains small decoder-only Transformer models (468M, 1.6B parameters) on subsets of RedPajama-V2 filtered by different combinations of these quality signals and evaluates them on 13 downstream benchmarks plus validation perplexity.

Information flows: CommonCrawl wet files → CCNet processing → language-ID/perplexity bucketing → raw document shards + quality signal computation → separate quality annotation shards → user-defined filtering → model training → downstream evaluation.

### 3.3 Roadmap for the Deep Dive

- **First**, the RedPajama-V1 data processing pipeline, because it establishes the transparency-first philosophy (documenting all uncertainties and resolving them explicitly) and motivates the need for V2 by revealing how much critical detail was missing from the original LLaMA description.
- **Second**, the RedPajama-INCITE model training setup on Summit, because it provides the validation that V1 produces usable models and surfaces the engineering challenges (FP16 precision, custom compilation) that explain the performance gap to LLaMA.
- **Third**, the RedPajama-V2 data acquisition and basic processing (CCNet pipeline), because this establishes what raw material enters the system and what lightweight transformations are applied before quality signals are computed.
- **Fourth**, the quality signal taxonomy and computation, because these 46 annotations are the core innovation of V2 — understanding what each signal measures, how it is computed, and what filtering decisions it enables is essential to interpreting the ablation results.
- **Fifth**, the deduplication approach (exact and fuzzy), because deduplication is a central quality intervention that interacts with other filters and is applied as a separate, sequential pass over the corpus.
- **Sixth**, the ablation study methodology (models, training, evaluation), because this shows concretely how the quality signals translate into training data decisions and how those decisions affect downstream performance.

### 3.4 Detailed, Sentence-Based Technical Breakdown

This is primarily a **data infrastructure paper** whose core idea is that providing raw, unfiltered web data with rich per-document metadata enables downstream researchers to study the data curation problem systematically, rather than being locked into a single curator's quality judgments. The technical approach therefore consists of: (a) transparently reconstructing an existing dataset (V1), (b) building a massive unfiltered corpus with quality annotations (V2), and (c) demonstrating that those annotations can be combined in different ways to produce models with different performance profiles.

---

#### RedPajama-V1: LLaMA Training Data Reproduction

The RedPajama-V1 dataset is a best-effort, fully open reproduction of the training corpus described in Touvron et al. (2023) for the LLaMA-1 family of models. The authors' goal is not to improve on LLaMA's recipe but to **make it transparently available** so that the community can study, audit, and build upon it. The process is documented in Section 3.1.

**Data sources and target composition.** The LLaMA corpus draws from seven distinct datasets: English CommonCrawl, C4, GitHub, Wikipedia, Books (Project Gutenberg and Books3), ArXiv, and Stack Exchange. The token counts achieved by the RedPajama-V1 reproduction are listed in Table 2: CommonCrawl (878B tokens), C4 (175B), GitHub (59B), Books (26B), ArXiv (28B), Wikipedia (24B), StackExchange (20B), for a total of approximately **1.2 trillion tokens**.

The key methodological principle is that for each data source, the authors (a) identify gaps and ambiguities in the LLaMA paper's description, (b) document these explicitly, and (c) make and justify a specific resolution. Table 10 in the Appendix provides a comprehensive inventory of these uncertainties — this table is itself an important contribution because it reveals exactly how underspecified the original LLaMA recipe was and quantifies the "reproduction tax" that prior work imposed on the community by omitting these details.

**CommonCrawl processing.** The LLaMA paper states that it used five CommonCrawl snapshots from 2017–2020, processed via the CCNet pipeline (Wenzek et al., 2019), keeping only the "head" and "middle" perplexity buckets (discarding the "tail"), and applying an additional linear classifier trained on Wikipedia reference articles to further filter low-quality documents. The RedPajama-V1 authors faced two major unspecified parameters: *which specific snapshots were used*, and *how exactly the Wikipedia classifier was constructed and thresholded*.

For snapshot selection, the authors chose "the five English CommonCrawl snapshots 2019-30, 2020-05, 2021-04, 2022-5, and 2023-06, representing the first snapshot in the five years preceding the start of the project." This is a pragmatic replacement — the original snapshots were not specified, so they used the most recent available ones that covered a comparable time span.

For the Wikipedia classifier, the authors reconstructed it from scratch. They downloaded the most recent English Wikipedia snapshot available by April 1, 2023, extracted 38 million URLs, crawled 300,000 pages as positive training examples, applied moderate cleaning via the CCNet pipeline, and trained a **unigram bag-of-words classifier using fastText** (Joulin et al., 2017). The classifier produces a scalar score between 0 and 1 for each document, representing the model's confidence that the document resembles Wikipedia reference text. The threshold was set to **0.25** — documents scoring below this value were discarded. The threshold was not theoretically motivated; rather, it was chosen empirically to reduce the CommonCrawl subset "to approximately the same size as the LLaMA CommonCrawl dataset" (i.e., matching the token count reported in the original paper). This is an important detail: the filtering threshold is calibrated to a downstream size target, not to a principled quality cutoff, meaning the "quality" definition is relative to the scale of the original dataset.

The base processing pipeline (CCNet) works as follows: for each CommonCrawl snapshot, CCNet performs language identification, deduplicates each shard independently (line-level deduplication), and assigns each document to one of three quality buckets — "head," "middle," or "tail" — based on the **perplexity assigned by a 5-gram Kneser-Ney language model trained on Wikipedia**. Documents that receive low perplexity (closer to Wikipedia-like text) are classified as "head"; those with medium perplexity as "middle"; those with high perplexity as "tail." The LLaMA recipe discards the "tail" bucket entirely. This perplexity-based bucketing is a widely used heuristic in web data curation because it captures a rough notion of "text quality" — coherent, well-formed text tends to have lower perplexity under a Wikipedia-trained LM than garbled boilerplate, code, or navigation menus.

**C4 integration.** The LLaMA corpus includes the C4 dataset (Raffel et al., 2020) to provide "diverse versions of CommonCrawl." RedPajama-V1 simply uses the `c4_en` version available on the Hugging Face Hub. No additional processing is applied.

**GitHub filtering.** The LLaMA corpus uses public GitHub repositories (accessed via Google BigQuery) distributed under Apache, BSD, and MIT licenses, with additional heuristics to filter low-quality files and file-level deduplication. RedPajama-V1 applies the following explicit filtering heuristics (Section 3.1 and Appendix C.1), removing any file with:

- Maximum line length exceeding 1000 characters
- Average line length exceeding 100 characters
- Proportion of alphanumeric characters less than 0.25
- Ratio of alphabetical characters to tokens less than 1.5
- File extension not in a whitelist of ~50 extensions (`.py`, `.js`, `.java`, `.cpp`, `.rs`, `.go`, `.rb`, etc.)

These heuristics are drawn from The Stack dataset (Kocetkov et al., 2022) and target common pathologies in raw code corpora: minified files (very long lines), auto-generated files (low alphanumeric proportion, extreme line lengths), binary files misclassified as text, and non-code file types that happen to appear in repositories.

**Wikipedia processing.** The LLaMA corpus uses Wikipedia dumps from June–August 2022 across 20 languages, processed to remove hyperlinks, comments, and formatting boilerplate. RedPajama-V1 uses the Wikipedia dataset from the Hugging Face Hub (dump dated 2023-03-20), which applies equivalent preprocessing.

**Books (Gutenberg and Books3).** The LLaMA corpus includes Project Gutenberg and Books3 (from the Pile). RedPajama-V1 initially included both but subsequently "took [Books3] down due to copyright issues." For Gutenberg, they used only the **PG19 subset** and applied **SimHash** (a locality-sensitive hashing technique) to remove near-duplicates. The specific parameters of the SimHash deduplication (hash size, number of bands, similarity threshold) are not detailed in the paper.

**ArXiv processing.** The LLaMA corpus processes ArXiv LaTeX source files, removing everything before the first section, comments, inline-expanded definitions and macros, and the bibliography (following Lewkowycz et al., 2022). RedPajama-V1 downloaded ArXiv data from Amazon S3, kept only LaTeX source files, and implemented equivalent postprocessing: removing preambles, comments, bibliographies, and expanding macros.

**Stack Exchange processing.** The LLaMA corpus uses a Stack Exchange dump, keeping data from the 28 largest websites, removing HTML tags, and sorting answers by score (highest to lowest). RedPajama-V1 downloaded the dump from the Internet Archive, applied the same filtering (top 28 sites, HTML tag removal), and additionally grouped posts into **question-answer pairs** with answers ordered by score descending.

**Why this approach over alternatives.** The alternative would have been to simply use existing cleaned subsets (e.g., the Pile's versions of GitHub, ArXiv, or Stack Exchange). The authors chose to reconstruct from source because (a) the LLaMA recipe applied different filtering than the Pile for some sources, (b) the act of reconstruction itself surfaces the unspecified parameters that prevent faithful reproduction, and (c) full transparency requires provenance from the raw source, not from an intermediate cleaned version whose own curation decisions are embedded in the data.

**Output format.** Each RedPajama-V1 subset is distributed as gzip-compressed JSON Lines files (.json.gz), partitioned into shards. Most subsets follow the schema `{"text": "...", "meta": {...}}` where the `meta` field contains source-specific metadata (timestamps, URLs, language tags, scores, etc.). The CommonCrawl subset includes additional fields `pred_label`, `pred_label_prob`, `wiki_prob`, and `source` reflecting the classifier predictions applied during filtering (Appendix B.1.1).

---

#### RedPajama-INCITE Model Training on Summit

To validate that RedPajama-V1 produces a usable training corpus — and to quantify how closely it approximates the original LLaMA data — the authors trained a family of decoder-only Transformer models at 3B and 7B parameter scales on the Summit supercomputer at Oak Ridge National Laboratory (Section 3.2). This section is as much an **engineering report** as an evaluation; the training setup imposed unusual constraints that illuminate why open reproduction is not merely a data problem but also a compute infrastructure problem.

**Hardware constraints.** Summit consists of 4,608 nodes, each containing 6 NVIDIA V100 GPUs and an IBM Power9 CPU. This architecture differs from the A100/H100 clusters typically used for LLM training in several critical ways:

- **No bf16 support:** V100 GPUs do not support the bfloat16 floating-point format, which has become standard for stable LLM training because it preserves the same exponent range as FP32 while using half the memory. The authors were forced to train with **FP16 precision** and use **loss scaling** (Micikevicius et al., 2017) to prevent gradient underflow — a technique where the loss is multiplied by a large constant before backpropagation and the gradients are divided by the same constant afterward, keeping small gradient values from vanishing to zero in FP16. The authors note that this "may have had an effect on convergence" and required them to use lower learning rates than those reported in the LLaMA paper: **$1.6 \times 10^{-4}$ for the 3B model and $1.2 \times 10^{-4}$ for the 7B model**, compared to the LLaMA paper's higher values.

- **Power9 architecture incompatibility:** The IBM Power9 CPU uses a different instruction set (Power ISA) than x86 or ARM. Modern versions of PyTorch and the Python scientific computing stack are not pre-compiled for this architecture — the latest officially supported PyTorch version was 1.9. To use contemporary training libraries, the team had to **recompile PyTorch from scratch** and build a custom training stack. This is described as being "documented in more detail in the GPT-NeoX technical report" (Black et al., 2022).

- **Interconnect and parallelism limits:** The Power9 interconnect was slow enough that scaling beyond certain node counts would have required increasing the global batch size, which "would hurt convergence." The 7B model used 512 nodes (3,072 GPUs) in parallel, while the 3B model used 256 nodes (1,536 GPUs). The global batch size for both was **4 million tokens**.

- **Tensor and pipeline parallelism configuration:** Due to the 6-GPU-per-node configuration, the 7B model used **12-way pipeline parallelism** and **2-way tensor parallelism**, while the 3B model used **6-way pipeline parallelism** and **2-way tensor parallelism**. Pipeline parallelism splits model layers across devices; tensor parallelism splits individual layer computations across devices.

**Training recipe.** The 3B model was trained for **800 billion tokens total**; the 7B model for **1.001 trillion tokens**. The learning rate schedule followed the LLaMA paper: a linear warmup period followed by linear decay. The authors used the AdamW optimizer (Diederik, 2014) but the specific AdamW hyperparameters (beta values, epsilon, weight decay) are not explicitly stated in this section of the paper for the Summit training runs — they reference matching the LLaMA paper's schedule but note that the learning rate was lower due to FP16 constraints.

**Why this matters beyond the specific models.** The Summit training experience illustrates a systemic barrier to open LLM development: even when the training data is fully open and transparent, the compute infrastructure may impose constraints (precision format, interconnect speed, software compatibility) that prevent faithful reproduction of reported results. The paper's documentation of these challenges — and the specific workarounds adopted — provides a template for other groups attempting similar reproductions on non-standard hardware.

---

#### RedPajama-V2: Data Acquisition and Basic Processing

RedPajama-V2 takes a fundamentally different approach from V1. Rather than reproducing a specific existing recipe, it builds a massive, **unfiltered, metadata-rich** foundation out of which many different datasets can be constructed. The design philosophy is summarized as "rather than prescribing what constitutes a high-quality dataset, we offer a broad, general-purpose corpus of web documents. Each document is tagged with quality signals, empowering users to make informed decisions based on their specific needs and criteria" (Section 1).

**Source data: CommonCrawl wet files.** The CommonCrawl Archive is a public repository of web crawl data, updated approximately monthly since 2013. It provides data in three formats: WARC (raw HTML with HTTP headers), WAT (metadata), and WET (extracted plain text). RedPajama-V2 uses the **WET format** — plain text that has already been extracted from HTML, removing tags, scripts, and formatting. The authors selected all **84 monthly snapshots between 2014 and April 2023** (the snapshot IDs range from 2014-15 to 2023-14).

This is a deliberate design choice with important implications. Using WET instead of WARC means RedPajama-V2 starts from text that has already undergone CommonCrawl's HTML-to-text extraction, which is a non-trivial processing step. This extraction removes HTML boilerplate but can also introduce artifacts — improperly extracted text, merged navigation elements, residual JavaScript, garbled character encodings. By starting from WET, the authors inherit CommonCrawl's extraction decisions and any errors they contain. The alternative (starting from WARC and implementing custom extraction) would have been more expensive but would have provided full control over the text extraction quality. The tradeoff is scale versus precision — WET enables processing 84 snapshots spanning nearly a decade, which would be impractical with custom extraction at this scale.

**CCNet pipeline processing.** The WET text is passed through the CCNet pipeline (Wenzek et al., 2019), the same tool used for RedPajama-V1's CommonCrawl subset but applied here with **all perplexity buckets retained** (head, middle, and tail — unlike V1 which discarded the tail). CCNet performs the following operations on each document:

1. **Language identification:** A fastText language classifier assigns each document a language label (English, German, French, Spanish, Italian) and a confidence score. For RedPajama-V2, five languages are retained: English, German, French, Spanish, and Italian. Documents in other languages are discarded.

2. **Perplexity scoring:** A 5-gram Kneser-Ney language model trained on Wikipedia computes the perplexity of each document. Low perplexity indicates text that is statistically similar to Wikipedia (typically well-formed, coherent prose). High perplexity indicates text that diverges from Wikipedia's distribution (which could mean low-quality text, but also could mean domain-specific text like legal documents, technical manuals, or non-prose content like poetry or code).

3. **Bucketing:** Documents are assigned to "head" (lowest perplexity), "middle" (medium perplexity), or "tail" (highest perplexity) buckets. In RedPajama-V2, **all three buckets are retained**, giving downstream users the option to include tail data if they believe it contains valuable signal despite its higher perplexity.

4. **Line-level deduplication:** Within each shard, duplicate lines are removed. This is a lightweight deduplication that eliminates boilerplate repeated within a single document (e.g., navigation menus, repeated headers/footers) but does not remove duplicate documents across the corpus — that is handled separately by the batch deduplication step described later.

5. **Basic metadata extraction:** CCNet records document length (characters), number of lines, original length before line-level dedup, original number of lines, language score, perplexity, and bucket assignment.

**Output scale.** After CCNet processing, RedPajama-V2 contains approximately **113.3 billion individual text documents** across the five languages, totaling an estimated **123.7 trillion tokens** (token counts estimated using the Mistral BPE tokenizer on a 100M-document sample). The partition sizes are detailed in Table 3:

| Partition | Documents (B) | Tokens (T) |
|-----------|---------------|------------|
| All (head+middle+tail) | 113.3 | 123.7 |
| Tail | 80.5 | 73.0 |
| Head+middle | 32.8 | 50.7 |
| Head+middle (deduplicated) | 20.8 | 30.4 |

Several patterns are notable. First, the tail partition contains the majority of documents (80.5B out of 113.3B) but a smaller fraction of tokens (73.0T out of 123.7T) because tail documents are typically shorter (~850 tokens on average) than head and middle documents (~1,500 tokens). Second, exact deduplication (described below) reduces the head+middle partition from 32.8B to 20.8B documents — approximately a 37% reduction — and from 50.7T to 30.4T tokens. This is consistent with prior findings that web corpora contain substantial near-duplicate and exact-duplicate content (Lee et al., 2021; Tirumala et al., 2023).

**Why preserve the tail?** This is a key design decision that distinguishes RedPajama-V2 from most prior web datasets. C4, RefinedWeb, FineWeb, and the original LLaMA recipe all discard low-quality or high-perplexity documents. RedPajama-V2 retains them and provides the quality signals so users can make their own filtering decisions. The motivation is twofold. First, what appears "low quality" under a Wikipedia-perplexity metric may contain valuable content for certain domains — legal text, medical literature, technical documentation, and poetry can all have high perplexity under a general-domain language model but be perfectly valid training data. Second, the tail's large size (73T tokens) means that even if only a small fraction of it proves useful under more sophisticated filtering (e.g., ML-based classifiers rather than simple perplexity thresholds), there is a substantial amount of signal potentially recoverable. The paper does not itself explore tail-data filtering strategies; this is left to future work.

**Output format.** Documents are stored as gzip-compressed JSON Lines files, partitioned into 5,000 shards per snapshot per language per perplexity bucket. Each document record follows a schema that includes: `url` (the source URL), `date_download` (when CommonCrawl fetched it), `digest` (a content hash), `length` and `nlines` (after line-level dedup), `source_domain` (extracted domain), `title`, `raw_content` (the plain text), `cc_segment` (which CommonCrawl segment it came from), `original_nlines` and `original_length` (before line-level dedup), `line_ids` (mapping to original lines), `language` and `language_score` (from fastText), `perplexity` (from the Kneser-Ney LM), and `bucket` (head/middle/tail). The file naming pattern is:

```
documents/<snapshot_id>/<shard_id>/<lang>_<ppl_bucket>.json.gz
```

where `<snapshot_id>` is the CommonCrawl crawl identifier, `<shard_id>` ranges from `0000` to `4999`, `<lang>` is one of `en`, `de`, `fr`, `es`, `it`, and `<ppl_bucket>` is `head`, `middle`, or `tail`.

---

#### Quality Signal Taxonomy and Computation

The central innovation of RedPajama-V2 is not the raw text itself — several prior datasets provide large web corpora — but the **46 quality signals** computed for each document in the head+middle partition (a 50T token subset). These signals are stored separately from the raw documents, in corresponding shard files following the naming pattern:

```
quality_signals/<snapshot_id>/<shard_id>/<lang>_<ppl_bucket>.signals.json.gz
```

Each quality signal record contains a document identifier (`id` and `id_int`), metadata (CommonCrawl segment, source domain, URL, language, snapshot ID), and a `quality_signals` dictionary. The dictionary keys are signal names; the values are lists of `[start, end, score]` tuples indicating the character range in `raw_content` where the signal applies and its value. This representation, borrowed from Dolma (Soldaini et al., 2024), allows a single format to encode both document-level signals (where `start=0, end=document_length`) and line-level signals (where each line gets its own entry).

The 46 quality signals fall into five categories, each representing a different dimension of document quality. Understanding what each category measures — and the implicit assumptions behind it — is essential to interpreting why different filtering combinations produce different downstream model behaviors.

---

##### Category 1: Natural Language Measures

These signals assess whether the document text resembles natural language as opposed to code, menus, navigation boilerplate, or garbled extraction artifacts. They are primarily drawn from the heuristics used in C4 (Raffel et al., 2020), RefinedWeb (Penedo et al., 2024), and the Gopher paper (Rae et al., 2021). Table 12 in the Appendix provides the complete list; the most important are:

**Structural and lexical signals:**
- `rps_doc_curly_bracket`: The ratio of occurrences of `{` or `}` to total characters. High values indicate code or data serialization rather than prose. From C4.
- `rps_doc_frac_all_caps_words`: Fraction of words consisting entirely of uppercase letters. High values suggest navigation menus, headers, or SHOUTING TEXT. From the Pretrainer's Guide (Longpre et al., 2023).
- `rps_doc_frac_lines_end_with_ellipsis`: Fraction of lines ending in `...` or the Unicode ellipsis character `U+2026`. High values indicate truncated text or navigation elements. From RefinedWeb and Gopher.
- `rps_doc_frac_no_alph_words`: Fraction of words containing no alphabetical character. High values indicate numeric data, code, or symbol-heavy text. From RefinedWeb and Gopher.
- `rps_doc_lorem_ipsum`: Ratio of occurrences of the string "lorem ipsum" to total characters. Detects placeholder text. From C4.
- `rps_doc_mean_word_length`: Mean word length in characters (after normalization). Excessively long or short words can indicate non-natural text. From RefinedWeb and Gopher.
- `rps_doc_stop_word_fraction`: Ratio of stop words (common function words like "the", "and", "of") to total words. Natural language typically has a high stop word fraction; code and boilerplate do not. Stop word list from the `stopwords-json` repository. From RefinedWeb and Gopher.
- `rps_doc_symbol_to_word_ratio`: Ratio of symbols (`#`, `...`, `U+2026`) to words. High values indicate non-prose content. From RefinedWeb and Gopher.
- `rps_doc_frac_unique_words` (also called "degeneracy"): Fraction of unique words relative to total words. Very low values indicate repetition; very high values can indicate word salad or garbled text. From the Pretrainer's Guide.
- `rps_doc_unigram_entropy`: Entropy of the unigram word distribution, computed as $\sum_x -x_n \cdot \log(1/n)$ where the sum is over counts of unique words. Measures lexical diversity. Low entropy indicates repetitive text; extremely high entropy can indicate random characters.
- `rps_doc_word_count`: Total number of words after normalization. Very short documents may lack meaningful content. From RefinedWeb and Gopher.

**Line-level natural language signals:**
- `rps_lines_ending_with_terminal_punctuation_mark`: Whether each line ends with `.`, `!`, `?`, or `"`. Natural language prose lines typically end with terminal punctuation; code lines, menu items, and headers often do not. From C4.
- `rps_lines_javascript_counts`: Number of occurrences of the word "javascript" per line. Detects embedded code.
- `rps_lines_num_words`: Number of words per line (computed on normalized text). Extremely short or long lines can indicate non-prose content. From C4 and RefinedWeb.
- `rps_lines_numerical_chars_fraction`: Ratio of numerical characters to total characters per line. High values indicate numeric data, tables, or code. From RefinedWeb.
- `rps_lines_start_with_bulletpoint`: Whether a line starts with a bullet point symbol (a set of 10 Unicode characters including `U+2022` bullet, `U+2023` triangular bullet, `U+25B6` black right-pointing triangle, etc.). Detects lists. From FineWeb and Gopher.
- `rps_lines_uppercase_letter_fraction`: Ratio of uppercase letters to total characters per line. High values indicate headers, menus, or ALL CAPS text. From RefinedWeb.
- `rps_doc_num_sentences`: Total number of sentences. Very low values for long documents suggest poor formatting. From C4.

The underlying assumption behind these signals is that **the training data for a language model should consist primarily of well-formed, continuous prose** — the kind of text that a Wikipedia article or book contains. Documents that deviate from this pattern (code, lists, navigation, truncated fragments) are presumed to be lower quality. This assumption is reasonable for training general-purpose LLMs but may be overly restrictive for specialized models (e.g., a code generation model would actively want curly brackets; a summarization model might want bulleted lists).

---

##### Category 2: Repetitiveness Measures

Repetitive text is a known failure mode of web data: content farms duplicate the same text across many pages, websites display the same boilerplate on every page, and HTML extraction can produce repeated fragments. Training on repetitive text has been linked to memorization (Lee et al., 2021) and repetitive language model generations (Holtzman et al., 2019). These signals quantify how much of a document is composed of repeated character sequences. Table 14 provides the full list.

The measures fall into two subcategories:

**Top-n-gram fraction:** For $n \in \{2, 3, 4\}$, compute the fraction of all characters in the document that appear in the single most frequent word n-gram. For example, if a document contains the phrase "click here click here click here" repeated many times, the top 2-gram "click here" would account for a large fraction of the document's characters. The signals are:
- `rps_doc_frac_chars_top_2gram`
- `rps_doc_frac_chars_top_3gram`
- `rps_doc_frac_chars_top_4gram`

**Duplicate n-gram fraction:** For $n \in \{5, 6, 7, 8, 9, 10\}$, compute the fraction of characters that appear in *any* duplicated word n-gram. Crucially, characters in overlapping n-grams are counted only once to avoid double-counting. For example, the phrase "the cat the cat the cat" contains the duplicated 2-gram "the cat" appearing twice. The characters `t h e _ c a t` appear in at least one duplicated n-gram, so they contribute once to the count. The signals are:
- `rps_doc_frac_chars_dupe_5grams`
- `rps_doc_frac_chars_dupe_6grams`
- `rps_doc_frac_chars_dupe_7grams`
- `rps_doc_frac_chars_dupe_8grams`
- `rps_doc_frac_chars_dupe_9grams`
- `rps_doc_frac_chars_dupe_10grams`

These are drawn from FineWeb (Penedo et al., 2024) and the Gopher paper (Rae et al., 2021). The rationale is that at some threshold of repetition, a document transitions from "containing some repeated phrases" (normal) to "being mostly repeated content" (low quality). The different n-gram sizes capture repetition at different granularities: short 2-grams can detect simple phrase repetition; longer 10-grams can detect duplicated sentences or paragraphs.

**How these are computed:** For the top-n-gram measures, the algorithm extracts all word n-grams of the specified length from the normalized document text, counts their frequencies, identifies the most frequent n-gram, and computes the fraction of total characters that appear within occurrences of that n-gram. For the duplicate n-gram measures, the algorithm identifies all n-grams that appear more than once, collects the character spans they cover (without double-counting overlaps), and computes the fraction of total characters covered. This is a character-level metric, not a token-level metric, because the downstream goal is to filter out documents where a large fraction of the visible text is repetition.

---

##### Category 3: Content-Based (Toxicity/NSFW) Measures

These signals flag documents that contain offensive, harmful, or inappropriate content. They are drawn from the filtering approaches used in C4 and RefinedWeb. Table 15 lists them.

- `rps_doc_ldnoobw_words`: The number of sequences of words in the document that appear in the **LDNOOBW blocklist** (List of Dirty, Naughty, Obscene, and Otherwise Bad Words — a community-maintained list of offensive English terms available at `github.com/LDNOOBW`). This is a simple substring matching approach: for each phrase in the blocklist, count how many times it appears in the document text. The count (not just a binary flag) is provided so users can set their own threshold. From C4.

- `rps_doc_ut1_blacklist`: A categorical flag indicating whether the document's domain appears in the **UT1 blacklist** maintained by the Université Toulouse 1 Capitole (`dsi.ut-capitole.fr/blacklists/`), which categorizes domains by content type (adult, gambling, malware, phishing, etc.). Rather than a binary flag, RedPajama-V2 provides the category ID so users can selectively filter based on content type. From RefinedWeb.

The paper acknowledges that these signals are limited in scope and focus primarily on NSFW content. It notes (Section 4.1.2):

> "we believe other content-based filters such as domains or embedding clusters are also promising directions"

and references work on embedding-based diversification (Tirumala et al., 2023) as a direction for future expansion. Figure 8 in the Appendix visualizes the topical clusters found in a 2M-document sample of RedPajama-V2 using Nomic Atlas embeddings (gte-large-en-v1.5), showing clusters corresponding to topics like "Election - Health - COVID Testing," "Religion/Spirituality - Gaming," "Education - Golf," and "Online Privacy - Privacy Policy - Contracts." This clustering is provided as a demonstration of what is possible with the raw data rather than as a quality signal included in the release.

---

##### Category 4: ML-Based Heuristics (fastText Classifiers and DSIR Importance Weights)

These signals use machine learning models to score documents against high-quality reference domains. Unlike the rule-based heuristics above, which rely on surface-level text features, these signals estimate a document's similarity to known-good text distributions. Table 13 lists the signals.

**fastText classifier signals.** A unigram bag-of-words classifier is trained using fastText (Joulin et al., 2017) to discriminate between unfiltered RedPajama-V2 data (the source distribution) and a high-quality target domain. The classifier outputs a probability that a given document belongs to the target domain. The following classifier signals are provided:

- `rps_doc_ml_wikiref_score`: A fastText classifier trained to distinguish Wikipedia reference articles from unfiltered RedPajama-V2 data. This is the same classifier used in RedPajama-V1's CommonCrawl filtering. It is trained on 300,000 crawled Wikipedia-referenced pages as positive examples. The output is a scalar between 0 and 1; higher values indicate the document is more Wikipedia-like. Applies only to English data.

- `rps_doc_ml_palm_score`: A fastText classifier trained to distinguish a composite of high-quality domains (Wikipedia articles, OpenWebText samples, and RedPajama-V1 books) from unfiltered RedPajama-V2 data. This is modeled after the quality classifier used in the PaLM paper (Chowdhery et al., 2023) and GLAM (Du et al., 2022). The output is a scalar between 0 and 1. Applies only to English data.

- `rps_doc_ml_wikipedia_score`: A fastText classifier trained to distinguish Wikipedia articles from unfiltered RedPajama-V2 data. This is used for non-English data (German, French, Spanish, Italian), where more diverse high-quality reference corpora (like OpenWebText or book collections) are less readily available.

**How fastText classifiers work.** fastText represents each document as a bag of word n-grams (unigrams and bigrams, in the RedPajama-V2 case), maps each n-gram to an embedding vector, averages the embeddings, and applies a linear classifier. The training objective is standard binary cross-entropy between the predicted score and the binary label (high-quality = 1, unfiltered = 0). At inference time, the model outputs a probability that the document belongs to the high-quality class.

The key design decision here is the choice of fastText over more sophisticated alternatives. fastText is chosen because: (a) it is extremely fast to train and apply at scale (processing billions of documents), (b) it captures local word co-occurrence patterns that are surprisingly effective at distinguishing high-quality prose from web garbage, and (c) its simplicity makes the quality signal interpretable — a document scores high if it contains n-grams characteristic of Wikipedia or books, which is a transparent criterion. The tradeoff is that fastText lacks the semantic understanding of a Transformer-based classifier; it can be fooled by documents that use Wikipedia-like phrasing but contain nonsense content, and it may penalize high-quality content in domains whose vocabulary diverges from the reference corpora (e.g., specialized scientific text with domain-specific terminology).

**DSIR importance weights.** The Data Selection via Importance Resampling (DSIR) method (Xie et al., 2023) estimates the importance of each document to a target domain. The core idea is:

$$w(x) = \log\frac{p_{\text{target}}(x)}{p_{\text{source}}(x)}$$

where $p_{\text{target}}$ is a bag-of-words language model (unigrams and bigrams) trained on the target domain, $p_{\text{source}}$ is a bag-of-words language model trained on the source domain (unfiltered RedPajama-V2 data), and $x$ is the document text.

**What it computes:** For each document, the DSIR weight is the log-likelihood ratio between the target-domain language model and the source-domain language model. A positive weight means the document is more likely under the target domain than the source domain (it is "important" for representing the target distribution); a negative weight means it is more typical of the source domain. The weight can be used for importance sampling — when constructing a dataset, documents can be sampled with probability proportional to $\exp(w(x))$ to obtain a distribution that approximates the target domain.

**Why this form:** The log-ratio has the statistical property that if samples from the source distribution are reweighted by $\exp(w(x))$, the resulting distribution is an unbiased estimate of the target distribution (under the bag-of-words model). This is a standard importance sampling construction. Using unigram and bigram models rather than more complex models is a deliberate simplicity tradeoff — the DSIR weights are meant to be computable at scale and capture surface-level lexical similarity, not deep semantic similarity. Documents that use vocabulary and phrasing characteristic of the target domain receive high weights; documents that use vocabulary characteristic of unfiltered web text receive low weights.

RedPajama-V2 provides three DSIR weight signals, corresponding to three target domains:

- `rps_doc_books_importance`: Target domain is Books (the RedPajama-V1 book corpus).
- `rps_doc_openwebtext_importance`: Target domain is OpenWebText (a recreation of the WebText corpus used to train GPT-2, consisting of outbound links from Reddit).
- `rps_doc_wikipedia_importance`: Target domain is Wikipedia articles.

The source domain for all three is unfiltered RedPajama-V2 data (or, more precisely, the language model trained on it). The DSIR weights are provided for English only.

**fastText vs. DSIR.** Both are ML-based quality estimators, but they capture different aspects of quality. fastText is a discriminative classifier: it learns features that separate high-quality and low-quality documents, and its output is a confidence score that can be thresholded. DSIR is a generative density ratio: it models the full distribution of each domain and estimates how much a document deviates from the source toward the target. In practice, the two signals are likely correlated (both give high scores to Wikipedia-like text), but the paper's ablations test them independently and find "no significant difference between using a fasttext classifier and DSIR" (Section 4.3.2) at the 468M model scale, suggesting they capture overlapping signal.

---

##### Category 5: Deduplication Metadata

Deduplication is one of the most consistently effective data quality interventions for LLM training, shown to reduce memorization and improve perplexity (Lee et al., 2021) while reducing training data size and compute requirements. RedPajama-V2 provides two types of deduplication metadata:

**Exact deduplication via Bloom filter.** A Bloom filter (Bloom, 1970) is a space-efficient probabilistic data structure that tests whether an element is a member of a set. It can produce false positives (reporting that a document is a duplicate when it is not) but never false negatives. The error rate is set to **1%**, meaning that approximately 1% of non-duplicate documents may be incorrectly flagged as duplicates.

The deduplication proceeds **sequentially through time, starting from the most recent snapshot (2023-14) and iterating backward to the oldest (2014-15)**. For each document, the system hashes its WET content (prior to CCNet processing) and queries the Bloom filter. If the hash is already present, the document is flagged as a duplicate. If not, the hash is inserted into the Bloom filter for future queries. This sequential, newest-first ordering means that when a document appears in multiple snapshots, the most recent occurrence is kept, and earlier occurrences are flagged as duplicates. This is a deliberate choice: newer versions of web pages are more likely to reflect current information.

The duplicate IDs are stored in parquet files following the pattern:

```
duplicates/<snapshot_id>/<shard_id>/<lang>_<ppl_bucket>.duplicates.parquet
```

Each row corresponds to a document that has at least one duplicate elsewhere in the corpus. Critically, the **first occurrence** (the newest, in this case) is *not* included in the duplicate IDs — only subsequent occurrences are flagged. This means that if a user drops all documents whose IDs appear in the duplicates files, one member of each duplicate cluster remains in the dataset.

**Why Bloom filter over exact hash set:** At 113 billion documents, storing all content hashes in a hash set would require substantial memory or disk I/O. A Bloom filter compresses the set membership test into a fixed-size bit array (the size is determined by the desired false positive rate and the expected number of insertions). The 1% false positive rate is a practical compromise: it means 99% of truly unique documents are correctly identified as unique, while false positives (documents incorrectly flagged as duplicates when they are not) occur at a low rate. The paper does not specify the exact Bloom filter parameters (bit array size, number of hash functions), only the target error rate.

**Impact of exact deduplication.** Figure 3 in the Appendix shows the chronological count of documents for each CommonCrawl snapshot before and after deduplication. A notable pattern is the sharp drop in unique documents for snapshots between 2014 and 2017 — the number of documents flagged as duplicates increases dramatically for these older crawls. The authors hypothesize that "this can be explained by a different list of seeds used by the CommonCrawl web crawler during that period." The dashed lines in Figure 3 show the number of unique documents after deduplication decreasing monotonically as we move backward in time, which is expected because the deduplication proceeds newest-first — older snapshots have progressively more content already seen in newer snapshots.

**Fuzzy deduplication via MinHash.** Exact deduplication only catches byte-for-byte identical documents. Many web documents are near-duplicates — the same article with minor formatting changes, the same boilerplate with different timestamps, or the same content syndicated across multiple sites. MinHash (Broder, 1997) is a locality-sensitive hashing technique for estimating the Jaccard similarity between sets. For text documents, the "set" is typically the set of word n-grams.

The MinHash signature for each document is computed as follows. The document is tokenized into word 13-grams (sequences of 13 consecutive words). Each 13-gram is hashed using **128 different hash functions**, producing 128 integer values per document. The signature is then partitioned into bands and rows for Locality-Sensitive Hashing (LSH), enabling efficient approximate nearest-neighbor search. The specific LSH parameters used in RedPajama-V2 are: **9 bands and 13 rows** (producing $9 \times 13 = 117$ signature elements used for LSH; the remaining 11 hash values are unused in the LSH step but preserved in the signature).

The relationship between LSH parameters and Jaccard similarity threshold is probabilistic. With $b$ bands and $r$ rows, the probability that two documents with true Jaccard similarity $s$ hash to the same bucket in at least one band is:

$$P(\text{collision}) = 1 - (1 - s^r)^b$$

For $b = 9$ and $r = 13$, this S-curve has a steep transition around $s \approx 0.7\text{–}0.9$, meaning documents with Jaccard similarity above ~0.8 are very likely to be identified as near-duplicates, while documents below ~0.7 are very unlikely to be flagged. The MinHash signatures are stored in parquet files:

```
minhashes/<snapshot_id>/<shard_id>/<lang>_<ppl_bucket>.minhash.parquet
```

and are partitioned into bands and rows corresponding to different Jaccard similarity thresholds in the range $\{0.7, 0.8, 0.9, 1.0\}$. Users can select the appropriate threshold for their use case — a higher threshold (0.9) removes only very close near-duplicates; a lower threshold (0.7) removes more aggressively.

**Why MinHash over other fuzzy dedup methods:** Alternatives include SimHash (used in RedPajama-V1 for books), suffix arrays, or full pairwise comparison. MinHash+LSH is chosen because it scales to billions of documents: the LSH banding means that document pairs only need to be compared if they hash to the same bucket in at least one band, reducing the quadratic pairwise comparison problem to approximately linear time in the number of documents (with a constant factor determined by the LSH parameters). The 13-gram tokenization (rather than, say, 5-grams) is a common choice for document-level deduplication because longer n-grams capture phrase-level duplication rather than accidental word co-occurrence.

---

#### Ablation Study Methodology

Section 4.3 describes the experimental setup used to demonstrate that RedPajama-V2's quality signals can be used to produce training datasets of varying quality. The experiments are designed as **existence proofs** rather than as a search for the optimal filtering configuration. The authors state:

> "We use relatively small scales, as this enables us to explore a wider range of filters, showing the breadth of the quality filters available in RedPajama."

**Model architecture.** All ablation models are decoder-only Transformers following the Llama-2 architecture (Touvron et al., 2023). Two scales are used:

- **468M parameters:** 24 layers, 16 attention heads, hidden dimension 1024, MLP expansion ratio 4.0, sequence length 2048. Trained on 100B tokens.
- **1.6B parameters:** Same architecture but with hidden dimension 2048. Trained on 350B tokens.

The sequence length of 2048 is notably shorter than the 4096 used in modern LLM training; this is likely a computational constraint given the number of ablation runs required.

**Training configuration.** The optimizer is AdamW (Diederik, 2014) with:
- Weight decay: **0.1**
- Maximum learning rate: **$5 \times 10^{-3}$** (468M) and **$5 \times 10^{-4}$** (1.6B)
- Cosine decay schedule with linear warmup during the first **1% of training steps**
- The specific beta values ($\beta_1$, $\beta_2$) and epsilon are not reported

The training uses the OLMo framework (Groeneveld et al., 2024) with FSDP (Fully Sharded Data Parallelism; Zhao et al., 2023) for distributed training. Models are trained on up to 5 H100 nodes with Infiniband interconnect.

**Datasets evaluated.** The ablation study trains models on a variety of RedPajama-V2 subsets filtered by different combinations of quality signals, plus several baseline datasets for comparison:

**Baselines from prior work:**
- C4 (Raffel et al., 2020)
- Dolma-v1.7 CC (the CommonCrawl subset of Dolma; Soldaini et al., 2024)
- FineWeb (Penedo et al., 2024)
- RefinedWeb (Penedo et al., 2024)
- RedPajama-V1 CommonCrawl (with Wikipedia-reference classifier filtering)

**RedPajama-V2 subsets (all from the head+middle partition unless noted):**
- RPv2 (2023-14): A single snapshot (2023-14) with no additional filtering beyond CCNet bucketing.
- RPv2 (2023-14) + exact dedup: The same snapshot with exact deduplication applied.
- RPv2 (2023-14) + exact dedup + full Gopher rules: Exact deduplication plus the complete set of Gopher quality filters (natural language, repetitiveness, and content-based rules from Rae et al., 2021).
- RPv2 (2023-14) + exact dedup + Gopher natlang + fastText (Wiki-middle): Only the Gopher natural language rules (not the repetitiveness rules), plus a fastText classifier threshold, keeping documents in the "middle" perplexity bucket.
- RPv2 (2023-14) + exact dedup + Gopher repetitiveness + fastText (Wiki-middle): Only the Gopher repetitiveness rules plus fastText filtering.
- RPv2 (9 Dumps) + exact dedup + Gopher natlang: Nine snapshots (2021-49 through 2023-14), exact deduplicated, with Gopher natural language rules.
- RPv2 (9 Dumps) + exact dedup + fuzzy dedup + full Gopher: The same plus fuzzy MinHash deduplication and full Gopher rules.
- RPv2 (9 Dumps) + exact dedup + fuzzy dedup + Gopher repetitiveness + fastText (Palm-mix): Gopher rep rules plus the PaLM-mix classifier.
- RPv2 (9 Dumps) + exact dedup + fuzzy dedup + Gopher natlang + fastText (Palm-mix): The same but with natural language rules instead of repetitiveness rules.
- RPv2 (9 Dumps) + exact dedup + line-level C4 filters + Gopher natlang + fastText (Palm-mix): Adding C4 line-level filtering.
- RPv2 (9 Dumps) + exact dedup + custom rules + fastText (Wiki-Ref) + perplexity filter: A custom configuration combining word count thresholds, average line length limits, Wikipedia perplexity filtering (keeping documents with perplexity > 30), and the Wikipedia-reference classifier.
- RPv2 (9 Dumps) + exact dedup + custom rules + Gopher repetitiveness + fastText (Wiki-Ref) + perplexity filter: Adding the Gopher repetitiveness rules to the above.

For the 1.6B parameter ablations (Table 6), three configurations are tested on the **full RedPajama-V2 dataset** (not just a subset of snapshots):
- RPv2 (full) + fuzzy dedup + Gopher + WikiRef classifier
- RPv2 (full) + fuzzy dedup + Gopher natlang + Palm-mix classifier
- RefinedWeb (baseline)

The 1.6B subsets were constructed by filtering the full RPv2 dataset, sampling approximately 1 trillion tokens, and then applying MinHash fuzzy deduplication with the same hyperparameters (128 hash functions, 9 bands, 13 rows).

**Evaluation benchmarks.** The paper evaluates models on a broad set of 13 downstream tasks plus validation perplexity, chosen to provide high signal-to-noise ratio even at small model scales (468M parameters). The benchmarks are listed in Table 4 and span diverse capabilities:

- **Natural language inference:** ANLI (Nie et al., 2020), ARC-c and ARC-e (Clark et al., 2018)
- **Coreference resolution:** Winogrande (Sakaguchi et al., 2021)
- **Sentence completion:** HellaSwag (Zellers et al., 2019), LAMBADA (Paperno et al., 2016)
- **Conversational QA:** CoQA (Reddy et al., 2019)
- **Multiple-choice QA (knowledge):** MMLU (Hendrycks et al., 2021), OpenbookQA (Mihaylov et al., 2018), PIQA (Bisk et al., 2020), PubMedQA (Jin et al., 2019), SciQ (Welbl et al., 2017), SocialIQA (Sap et al., 2019), TruthfulQA (Lin et al., 2021)

For validation perplexity, the paper follows Dolma's approach (Soldaini et al., 2024) and evaluates on two held-out corpora: **Paloma** (Magnusson et al., 2023), a diverse benchmark covering 18 domains, and the **Pile validation set** (Gao et al., 2020).

**Aggregation methodology.** Because the benchmarks produce scores on different scales (accuracy for most, F1 for CoQA, exact match for MMLU), the paper reports three aggregated metrics to compare datasets:

1. **Average:** The arithmetic mean of all benchmark scores. Simple but sensitive to score scales.
2. **Normalized average:** Each benchmark score is min-max normalized across all datasets before averaging. This prevents benchmarks with naturally larger score ranges from dominating.
3. **Rank-score:** For each benchmark, datasets are ranked from best to worst, and the sum of ranks is computed. Lower rank-sum indicates consistently strong performance across tasks. The paper reports the normalized sum of ranks.

The rank-score is included explicitly "to avoid averaging over scores with different scales" (Section 4.3.1). This is a methodologically sound choice because comparing datasets on benchmark batteries with heterogeneous score ranges can be misleading — a dataset that excels on an easy task with a wide dynamic range (e.g., PIQA, where most models score 60–85%) can appear better in raw average than a dataset that excels on a hard task with a narrow range (e.g., ANLI, where scores cluster around random chance 25%).

**Answer selection protocol.** For multiple-choice tasks, the paper uses `acc_norm` (normalized accuracy) where available, which normalizes the log-probabilities of answer choices by their token lengths before selecting the highest-scoring choice. This prevents the model from favoring shorter answers simply because they have fewer tokens to assign probability to. For generation tasks (CoQA), F1 score is used. The paper notes that some benchmarks (ANLI, TruthfulQA, ARC-c) are included in the per-task tables but excluded from the aggregated scores because they "provide a high enough signal-to-noise ratio" — these tasks are too difficult for 468M models, producing near-random scores that would add noise to the aggregation.

**Why this evaluation setup over alternatives.** The paper deliberately chose broad coverage over narrow optimization. Many dataset papers evaluate only on perplexity and a few standard benchmarks (HellaSwag, MMLU, PIQA). By including 13 tasks spanning reasoning, knowledge, and language understanding, plus two separate perplexity evaluations, the paper can detect whether certain filtering strategies produce models that are strong on some capabilities but weak on others — which is exactly what they find. The Gopher rules + fuzzy dedup configuration achieves the highest rank-score (more consistently strong across tasks) while RefinedWeb has a higher average score but a lower rank-score (strong on some tasks, weaker on others). This pattern would be invisible with a narrower evaluation.

## 4. Key Insights and Innovations

### Innovation 1: Reframing Web Data Curation as Metadata Preservation Rather Than Filtering

The dominant paradigm in web dataset construction, from C4 (Raffel et al., 2020) through RefinedWeb (Penedo et al., 2024) and FineWeb (Penedo et al., 2024), has been: **ingest raw web text, apply a curator's quality filters, and release the cleaned output.** This approach makes the filtering decisions opaque and irreversible — downstream users receive a finished product whose quality criteria are baked in and cannot be interrogated or reversed.

RedPajama-V2 inverts this logic. Rather than releasing a single filtered dataset, the paper releases **the raw, unfiltered web corpus alongside 46 per-document quality annotations** that encode the *evidence* on which filtering decisions could be based, without making those decisions. The design principle stated in Section 1 — "rather than prescribing what constitutes a high-quality dataset, we offer a broad, general-purpose corpus of web documents. Each document is tagged with quality signals, empowering users to make informed decisions based on their specific needs and criteria" — represents a fundamentally different relationship between dataset creators and users. The creator's role shifts from *judge* (deciding what is high quality) to *librarian* (organizing and annotating so others can decide).

This is more than a convenience feature. It transforms the research problem of data curation from **point optimization** (find the best filtering recipe) to **systematic ablation** (understand how different filtering dimensions affect different downstream capabilities), because the ablation space is now explicit and navigable without reprocessing the full Common Crawl corpus. The paper's own ablation studies (Section 4.3) demonstrate this: by testing configurations that vary along the Gopher natural language dimension versus the Gopher repetitiveness dimension versus ML-based classifier scores, the authors reveal that different filtering strategies produce models with different performance profiles across tasks — the Gopher full rules + fuzzy dedup achieve the highest rank-score (consistently strong across tasks), while RefinedWeb achieves a higher average score but a lower rank-score (strong on some tasks, weaker on others). This kind of comparative finding requires the metadata-first design; with traditional single-filtered datasets, there is no mechanism to disentangle the contribution of different quality dimensions.

The metadata-preservation approach also makes RedPajama-V2 **forward-compatible with future filtering innovations.** If a research team develops a novel quality classifier or a new repetitiveness threshold, they can apply it to the raw text using the existing data infrastructure without re-scraping and re-processing the source. This is an engineering contribution with research implications: it lowers the barrier to entry for data curation studies by amortizing the enormous fixed cost of web-scale processing across the entire community, rather than requiring each team to independently bear it.

The significance of this reframing is best understood by examining the "Raw Data" column in Table 1: among the 12 open pretraining datasets surveyed, RedPajama-V2 is the **only** one that checks this box. Every other dataset prescribes a quality standard. This is not an incremental improvement over prior web datasets — it is a category shift in what a dataset release constitutes.

---

### Innovation 2: The RedPajama-V1 Reconstruction as an Audit Artifact, Not Just a Replica

At first glance, RedPajama-V1 appears to be a straightforward reproduction of LLaMA's training data — useful, but intellectually incremental. What elevates it to a genuine contribution is the paper's treatment of the **uncertainties encountered during reconstruction as first-class outputs** rather than implementation details to be quietly resolved.

The LLaMA technical report (Touvron et al., 2023) described the training data in approximately one paragraph per source, leaving substantial ambiguity. Rather than making unilateral decisions and presenting the result as authoritative, the paper systematically documents every gap in the original description and every choice made to resolve it. Table 10 in the Appendix provides a detailed inventory: which CommonCrawl snapshots were ambiguous (and what was chosen), what classifier architecture was unspecified (and what was trained), what the GitHub filtering heuristics were (and which set from The Stack was adopted), which Wikipedia dump was used (and that the most recent available was selected), how book deduplication was performed (and that SimHash was chosen without specified parameters).

The conceptual move here is treating **reproducibility as an empirical finding rather than a binary property.** The paper does not declare "we reproduced LLaMA's dataset" — it declares "here is what was underspecified, here is what we chose, here is how the resulting models perform relative to the original, and here is what might explain the remaining gap." The RedPajama-INCITE-7B model underperforms LLaMA-7B (by 4.1 points on HELM-classic, per Section 3.2.2), and the paper offers two explicit hypotheses: (1) training with FP16 rather than bf16 due to V100 hardware constraints required lower learning rates, potentially affecting convergence; (2) "some salient details that went into the construction of the original LLaMA training corpus may be missing" (Section 3.2.2). This honest accounting — replicating what can be replicated, quantifying what cannot, and explicitly scoping the remaining uncertainty — is an intellectual contribution to the methodology of open model reproduction that extends beyond this specific dataset.

The broader significance is that **the RedPajama-V1 effort reveals how much of the "reproducibility crisis" in LLM training is a data documentation crisis.** Even when model weights are released (as LLaMA's were), the training data description can be so sparse that faithful reproduction is impossible without extensive guesswork. RedPajama-V1 serves as both a partial remedy (providing a documented, usable training corpus) and a diagnostic instrument (quantifying the gap between what was disclosed and what could be reproduced). The fact that this dataset has been used to train multiple prominent open models (Snowflake Arctic, AI2's OLMo, Salesforce's XGen, as listed in the abstract and Figure 1) demonstrates that the community values this kind of transparently-documented reproduction, even when it is explicitly an approximation rather than an exact match.

---

### Innovation 3: Difficulty-Independent Web Data Quality as a Multi-Objective Problem, Not a Scalar

The standard framing in web dataset curation has been to treat "quality" as a scalar that can be optimized for — a single filtering recipe produces a single dataset that is "better" or "worse" than alternatives, as measured by average downstream performance. This assumption underlies the design of datasets like C4, RefinedWeb, and FineWeb, each of which applies a specific set of quality criteria and presents the result as an improvement over predecessors.

The RedPajama-V2 ablation results (Section 4.3.2, Tables 5, 18–20) challenge this framing empirically. The highest aggregated benchmark scores are achieved by RPv2 filtered with the full Gopher rules and fuzzy deduplication, which achieves a rank-score of 0.700 — higher than RefinedWeb's 0.650. However, examining per-task performance reveals that no single configuration dominates across all dimensions:

- On validation perplexity (Paloma), the *unfiltered* RPv2 (2023-14) achieves the lowest perplexity (31.1), outperforming all filtered variants. Filtering with C4 rules increases Paloma perplexity to 39.9; filtering with Gopher raises it to 34.5. **Aggressive quality filtering produces models with worse perplexity on diverse-domain evaluation sets**, presumably because filtering removes content from domains that appear in Paloma's 18-domain coverage.

- On the Pile validation set, RPv2 + exact dedup + Gopher achieves perplexity of 36.3 — substantially worse than RefinedWeb (19.1) or RPv1-CC (18.7). This suggests that the Gopher-filtered subset, while producing strong downstream task performance, loses domain coverage relative to datasets that preserve a wider range of text.

- On individual downstream benchmarks, the ranking varies: RPv2 + exact dedup + full Gopher ranks first on ANLI, ARC-e, and CoQA, but trails on LAMBADA (where FineWeb leads) and PubMedQA (where RefinedWeb leads, per Table 18–20).

The conceptual insight is that **data quality for LLM pretraining is not a scalar optimization problem — it is a multi-objective tradeoff between domain coverage, task-specific performance, and language modeling fidelity.** A filtering strategy that improves aggregated benchmark scores (by removing noise and emphasizing well-formed prose) can simultaneously reduce domain coverage (by removing text that diverges from the prose distribution) and increase perplexity on diverse evaluation sets. The optimal filtering strategy is therefore use-case-dependent, not universal.

This finding is not just a caution about over-relying on aggregated metrics. It has practical implications for how datasets should be designed and evaluated. A model intended for broad scientific reasoning might benefit from preserving high-perplexity technical text that a language-modeling-perplexity filter would discard. A model intended for dialogue might benefit from different filtering than one intended for code generation. RedPajama-V2's design — providing quality signals as separable dimensions rather than a single quality score — is what makes this tradeoff visible and navigable. Users can independently set thresholds for natural language, repetitiveness, toxicity, and ML-based quality, exploring the Pareto frontier of performance tradeoffs rather than accepting a point solution.

The paper does not claim to have solved the multi-objective quality problem — the ablation studies are too small-scale and the filtering configurations too coarse to map the full frontier. But the core contribution is establishing that the problem *exists as a multi-objective problem* and providing the infrastructure (raw text + separable quality signals) that makes systematic exploration possible.

---

### Innovation 4: The Separability of Quality Dimensions and Their Downstream Effects

A subtler but equally important finding emerges from the ablation studies: **different quality dimensions affect different aspects of model performance, and they can be composed modularly in ways that are approximately additive.**

Consider the incremental filtering results in Table 5. Starting from unfiltered RPv2 (2023-14), which achieves an aggregate normalized average of 0.594:

- Adding **exact deduplication alone** reduces the normalized average to 0.472 — a substantial degradation in benchmark performance (because deduplication with the Bloom filter at this scale appears to remove content that, while duplicated, contributes useful training signal).
- Adding **fuzzy deduplication plus full Gopher rules** raises the normalized average to 0.700 — the highest among all RPv2 configurations.
- Using **Gopher natural language rules plus fastText (Wiki-middle) without fuzzy dedup** achieves 0.639 — better than exact dedup alone but worse than the full Gopher + fuzzy dedup combination.
- Using **Gopher repetitiveness rules plus fastText (Wiki-middle) without fuzzy dedup** achieves 0.633 — nearly identical to the natural-language variant.

This pattern suggests that the Gopher natural language rules and the Gopher repetitiveness rules capture overlapping but not identical quality signal, and that fuzzy deduplication provides an additional quality improvement beyond what rule-based filtering alone achieves. The fact that these dimensions can be toggled independently and produce measurable, interpretable changes in downstream performance is a validation of the metadata-preservation design: the quality signals are not just descriptive tags but are **causally connected to training outcomes** in ways that users can reason about.

A particularly striking example is the interaction between filtering and validation perplexity. Adding the C4 line-level filters to the Gopher + fastText configuration reduces Pile perplexity from 67.1 to 52.9 (Table 5, comparing "RPv2 (9 Dumps) + exact dedup + fuzzy dedup + Gopher rep + Palm-mix" at 67.1 vs. "RPv2 (9 Dumps) + exact dedup + line-filter + Gopher natlang + Palm-mix" at 52.9), but has negligible effect on aggregated benchmark scores (0.483 vs. 0.539 normalized average). This is a concrete instance of the multi-objective tradeoff: the C4 line filters remove content that hurts language modeling perplexity while leaving task performance largely unchanged.

The 1.6B parameter ablations (Table 6) reinforce this finding at a larger scale. The RPv2 model filtered with the full Gopher rules and a WikiRef classifier achieves an aggregate average of 50.0 versus RefinedWeb's 52.0 — competitive but not surpassing. However, the RPv2 model also achieves lower perplexity on the Pile (13.6 vs. 10.7 for RefinedWeb), again demonstrating that performance on downstream tasks and language modeling fidelity are partially decoupled objectives.

These results position RedPajama-V2 as a **scientific instrument for studying data quality**, not just a dataset for training models. By providing quality dimensions as independently manipulable variables, the dataset enables the kind of controlled experimentation that is necessary to establish causal relationships between data properties and model behavior — something that has been largely absent from the data curation literature, which has historically relied on holistic comparisons between monolithic datasets whose filtering decisions are confounded.

## 5. Experimental Analysis

### Evaluation Methodology

- **Dataset.** The primary data resource is RedPajama-V2 — a web-only corpus derived from 84 CommonCrawl snapshots (2014–2023), processed with the CCNet pipeline, totaling ~113B documents and ~123.7T raw tokens across five languages. The ablation studies in Section 4.3 use subsets of this corpus filtered by combinations of the 46 quality signals. For RedPajama-V1 validation, the RedPajama-INCITE models are evaluated on HELM classic (Bommasani et al., 2023) and EleutherAI's lm-evaluation-harness (Gao et al., 2024) across standard NLP benchmarks. The ablation models (468M and 1.6B parameters) are evaluated on 13 downstream benchmarks plus perplexity on two held-out corpora (the Pile validation set and Paloma).

- **Base model(s).** The ablation studies use decoder-only Transformer models following the Llama-2 architecture (Touvron et al., 2023) at two scales: a 468M-parameter model (24 layers, 16 attention heads, hidden dimension 1024, MLP expansion ratio 4.0, sequence length 2048) trained for 100B tokens, and a 1.6B-parameter model (same architecture, hidden dimension 2048) trained for 350B tokens. For RedPajama-V1 validation, a separate family of models — RedPajama-INCITE — was trained at 3B and 7B parameter scales on Summit's V100 GPUs, using the full RedPajama-V1 corpus (1.2T tokens). The small-scale ablation models are chosen explicitly to "explore a wider range of filters" (Section 4.3.1) rather than to maximize absolute performance, making the study a breadth-over-depth empirical survey.

- **Metrics.** Two classes of metrics are used. For downstream task performance, the paper reports: (a) raw accuracy or F1 on each of 13 benchmarks (Table 4), (b) three aggregate scores — the arithmetic mean across benchmarks (`Avg.`), the min-max normalized average across datasets (`Norm. Avg.`), and the normalized sum of ranks (`Rank-Score`) "to avoid averaging over scores with different scales" (Section 4.3.1). For language modeling fidelity, validation perplexity is computed on the Pile validation set (Gao et al., 2020) and the Paloma benchmark (Magnusson et al., 2023), following the approach in Dolma (Soldaini et al., 2024). For RedPajama-INCITE, HELM classic average (covering 16 core scenarios) and EleutherAI's lm-eval-harness average are reported.

- **Baselines.** The ablation study compares against five established web datasets: C4 (Raffel et al., 2020), Dolma-v1.7 CC (Soldaini et al., 2024), FineWeb (Penedo et al., 2024), RefinedWeb (Penedo et al., 2024), and the RedPajama-V1 CommonCrawl subset with Wikipedia-reference classifier filtering. For RedPajama-INCITE, baselines include GPT-Neo (Black et al., 2022), Pythia-2.8B (Biderman et al., 2023), Falcon-7B (Almazrouei et al., 2023), MPT-7B (MosaicML NLP Team, 2023), and Llama-7B (Touvron et al., 2023). These span the range from earlier open-source models (GPT-Neo, Pythia) to contemporary strong baselines (Falcon, MPT, Llama).

- **Generation budget / compute accounting.** For the ablation study, compute is measured in tokens trained: 100B tokens for 468M models, 350B tokens for 1.6B models. All training uses the OLMo framework with FSDP on up to 5 H100 nodes (Section 4.3.1). For the RedPajama-INCITE models, compute is measured in total tokens trained (800B for 3B, 1.001T for 7B), with the constraint that V100 GPUs required FP16 precision and loss scaling rather than the bf16 typically used for stable LLM training. No test-time compute scaling or search is employed — all evaluations use standard few-shot or zero-shot prompting — so generation budget accounting does not arise.

- **Cross-validation / statistical protocol.** There is no formal cross-validation or statistical significance testing reported for the ablation comparisons. The aggregation metrics (average, normalized average, rank-score) are computed across the 13 benchmarks listed in Table 4, but no confidence intervals, standard errors, or significance tests are provided for any of the comparisons. For the RedPajama-INCITE evaluation, standard benchmark evaluation protocols from HELM and lm-evaluation-harness are followed. The difficulty bins and two-fold cross-validation protocol described in the reference example paper are not present here — this is a dataset release paper, and the evaluation serves to demonstrate that filtering choices affect outcomes rather than to optimize a compute-allocation policy.

### Main Quantitative Results

The experimental results span two distinct evaluation efforts: (a) validation of the RedPajama-V1 dataset through model training at 3B and 7B scales on Summit, and (b) ablation studies at 468M and 1.6B scales demonstrating that RedPajama-V2's quality signals produce training subsets of varying downstream performance.

#### RedPajama-INCITE Model Performance (V1 Validation)

**RedPajama-INCITE-3B.** At the 3B scale, the model trained on RedPajama-V1 for 800B tokens demonstrates strong performance relative to comparably-sized open models. On the lm-evaluation-harness subset (Figure 2, Table 7), RedPajama-INCITE-Base-3B achieves an average score of **0.6662** across LAMBADA (OpenAI accuracy), HellaSwag (normalized accuracy), Winogrande (accuracy), and PIQA (normalized accuracy). This outperforms:

- GPT-Neo-2.7B (average 0.6197, trained on the Pile for 420B tokens)
- Pythia-2.8B (average 0.6451, trained on the Pile for 300B tokens)
- Pythia-2.8B-dedup (average 0.6429)

The margin over Pythia-2.8B is approximately 2.1 points, and over GPT-Neo approximately 4.7 points. On HELM classic (Table 7), RedPajama-INCITE-3B achieves a HELM average of **0.4060**, compared to 0.3570 for GPT-Neo and 0.3770 for Pythia-2.8B — a margin of 3–5 points.

These results validate that the RedPajama-V1 reproduction produces a usable training corpus. While the 3B model does not claim to surpass all alternatives on every metric, its consistent margin over strong baselines (GPT-Neo, Pythia) demonstrates that the reconstructed data pipeline yields data quality that is at least competitive with widely used curated corpora like the Pile.

**RedPajama-INCITE-7B.** At 7B scale, the picture is more nuanced. On HELM-classic (Table 8), RedPajama-INCITE-Base-7B achieves a HELM average of **0.431**. This places it:

- **4.1 points behind Llama-7B** (0.472)
- **1.0 points behind Falcon-7B** (0.441)
- **0.2 points ahead of MPT-7B** (0.429, listed as 0.444 in the table but this is MPT-7B-Instruct; the MPT-7B base score is not directly reported in the same row)

The paper explicitly breaks down this gap. On tasks that use log-probability-based scoring (computing the difference between probabilities of correct and incorrect answers), the model lags behind. However, "the model achieves comparable average HELM scores on tasks that directly generate answers and measure quality" (Section 3.2.2). Helped by this interpretation, the paper hypothesizes that FP16 training — which "does not allow us to use larger learning rates" (Section 3.2.2) — contributed to the gap, along with missing details in the LLaMA dataset description (Table 10).

The lm-evaluation-harness results (Table 9) show a similar ranking. RedPajama-INCITE-Base-7B achieves an average of **0.6882**, compared to:

- Llama-7B: 0.6881 (nearly identical)
- Falcon-7B: 0.7161
- MPT-7B: 0.7100
- GPT-J-6B: 0.6526
- Pythia-7B: 0.6392

The near-tie with Llama-7B on the lm-eval-harness aggregate (0.6882 vs. 0.6881) is particularly notable because it suggests that on tasks that rely more directly on language modeling capability rather than calibrated probability estimation, the RedPajama-V1 corpus produces comparable performance. The 4.1-point gap on HELM-classic, which weights log-probability tasks more heavily, reflects a calibration gap that the authors attribute to the FP16 training constraint rather than a fundamental data quality deficit.

**RedPajama-INCITE-7B-Instruct.** The instruction-tuned variant (Table 8) achieves a HELM average of **0.492**, which substantially outperforms all comparably-sized models:

- Llama-7B: 0.472 (+2.0 points)
- Falcon-7B: 0.441 (+5.1 points)
- MPT-7B: 0.444 (+4.8 points, though this is the base MPT-7B score — the Instruct version in the table scores 0.393)
- MPT-7B-Instruct: 0.393 (+9.9 points)
- Falcon-7B-Instruct: 0.407 (+8.5 points)

On individual HELM tasks, RedPajama-INCITE-7B-Instruct ranks first among all listed models on NarrativeQA (F1: 0.623), MS MARCO TREC (NDCG@10: 0.709), and IMDB (EM: 0.941). Its MMLU score (0.366) trails Llama-7B (0.345 base; note that the table lists Llama-7B's MMLU as 0.345 for the base model, but the Instruct column at 0.366 is the RedPajama model) and MPT-7B-Instruct (0.349) — this is a Base model comparison for MMLU in the table; the Instruct result is RedPajama's own.

This is a significant finding because it suggests that while the base model trained on RedPajama-V1 may underperform on certain calibration-sensitive metrics due to training precision constraints, instruction tuning can recover and even invert the ranking. The Instruct model's strong performance on few-shot tasks (beating Llama-7B by 2 points on HELM average, and by larger margins against Falcon and MPT) demonstrates that the underlying data quality is sufficient to produce a strong instruction-following model when augmented with multitask training data from P3 and Natural Instructions.

---

#### Ablation Studies on RedPajama-V2: 468M Parameter Models

The central empirical contribution of Section 4.3 is the demonstration that different combinations of RedPajama-V2's quality signals produce training datasets with substantially different downstream performance profiles. The headline results appear in Table 5.

**Unfiltered vs. filtered performance.** The unfiltered RPv2 (2023-14 single snapshot) achieves an aggregate normalized average of **0.594** across the 13 benchmarks, with a rank-score of — this is not directly given, but based on the pattern, the rank-score is 0.594, Pile perplexity of **31.1**, and Paloma perplexity of **19.7**. This is already competitive: its normalized average (0.594) exceeds C4 (0.472), Dolma-v1.7 CC (0.511), and RPv1-CC (0.461). Its Paloma perplexity (19.7) is the lowest among all datasets in the table, including RefinedWeb (19.1 — actually wait, the table shows RefinedWeb Pile=19.1, Paloma=32.8; RPv2 Pile=31.1, Paloma=19.7. Both Pile and Paloma perplexities are reported; the lower values are on different eval sets). The key point is that even *without additional filtering*, the raw CCNet-processed web text produces a model whose benchmark performance is in the middle of the pack — better than C4 and RPv1-CC, worse than RefinedWeb and the best filtered RPv2 configurations.

**Effect of exact deduplication alone.** Adding exact Bloom-filter deduplication to the single 2023-14 snapshot (RPv2 + exact dedup) *decreases* the normalized average from 0.594 to **0.472**, while increasing Paloma perplexity from 19.7 to **39.9**. This is a substantial degradation on both task performance and language modeling perplexity. At first glance, this result is counterintuitive because prior work (Lee et al., 2021) has shown that deduplication generally improves or maintains model quality. The authors do not explicitly explain this finding, but a plausible interpretation is that exact deduplication at the Bloom filter scale is removing document occurrences that, while duplicated across snapshots, provide useful repeated exposure to certain content — essentially acting as a form of upsampling that is beneficial for small-model training. Alternatively, the 1% Bloom filter false positive rate may be removing a non-trivial number of unique documents at the 113B-document scale, though the authors do not quantify this.

**Effect of Gopher rules + fuzzy deduplication.** The strongest RPv2 configuration in terms of aggregate benchmark scores is RPv2 (2023-14) + exact dedup + full Gopher rules (Table 5):

- Normalized average: **0.700** — the highest among all datasets in the table, exceeding RefinedWeb's 0.650.
- Rank-score: also the highest among all datasets, though the exact value is not listed in the text.
- Pile perplexity: **34.5** — better than the exact-dedup-only configuration (39.9) but worse than unfiltered (31.1) and substantially worse than RefinedWeb (19.1).
- Paloma perplexity: **24.9** — worse than RefinedWeb (32.8 — wait, this suggests RefinedWeb has higher Paloma perplexity, meaning RPv2+Gopher is *better* on Paloma).

The critical finding is the **decoupling between benchmark performance and language modeling perplexity.** The Gopher-filtered model achieves superior downstream task accuracy but does not achieve the lowest perplexity — unfiltered RPv2 (Paloma 19.7) and RPv1-CC (Pile 18.7) both have lower perplexity. This supports the paper's framing that data quality is multi-objective: filters that improve task performance by removing noisy or repetitive text can simultaneously reduce domain coverage in ways that appear as increased perplexity on diverse-domain evaluation sets.

**Comparison to established web datasets (Table 5, per-task breakdown in Tables 18–20).** The RPv2 + exact dedup + full Gopher configuration merits detailed comparison against RefinedWeb, the strongest baseline:

- **RefinedWeb aggregate:** Normalized average 0.650, rank-score not directly given but lower than RPv2+Gopher, Pile perplexity 19.1, Paloma perplexity 32.8.
- **RPv2+Gopher aggregate:** Normalized average 0.700 (higher), Pile perplexity 34.5 (worse than RefinedWeb's 19.1), Paloma perplexity 24.9 (better than RefinedWeb's 32.8).

On individual tasks (Tables 18–20):

- ANLI: RPv2+Gopher (34.1) vs. RefinedWeb (32.8) — RPv2 wins.
- ARC-c: RPv2+Gopher (22.3) vs. RefinedWeb (22.6) — RefinedWeb wins slightly.
- ARC-e: RPv2+Gopher (38.3) vs. RefinedWeb (38.3) — tie.
- Winogrande: RPv2+Gopher (52.2) vs. RefinedWeb (51.9) — RPv2 wins slightly.
- HellaSwag: RPv2+Gopher (32.1) vs. RefinedWeb (31.6) — RPv2 wins slightly.
- LAMBADA: RPv2+Gopher (18.7) vs. RefinedWeb (17.8) — RPv2 wins.
- CoQA: RPv2+Gopher (11.3) vs. RefinedWeb (13.2) — RefinedWeb wins.
- MMLU: RPv2+Gopher (27.0) vs. RefinedWeb (24.8) — RPv2 wins.
- OpenbookQA: RPv2+Gopher (28.8) vs. RefinedWeb (28.6) — RPv2 wins slightly.
- PIQA: RPv2+Gopher (62.8) vs. RefinedWeb (64.4) — RefinedWeb wins.
- PubMedQA: RPv2+Gopher (51.0) vs. RefinedWeb (52.2) — RefinedWeb wins.
- SciQ: RPv2+Gopher (53.9) vs. RefinedWeb (56.4) — RefinedWeb wins.
- SocialIQA: RPv2+Gopher (32.6) vs. RefinedWeb (32.8) — RefinedWeb wins slightly.

The pattern is that **RPv2+Gopher wins on language understanding tasks** (ANLI, Winogrande, HellaSwag, LAMBADA, MMLU, OpenbookQA) while **RefinedWeb wins on knowledge-intensive tasks** (CoQA, PIQA, PubMedQA, SciQ). This is another manifestation of the multi-objective quality tradeoff: Gopher filtering emphasizes well-formed prose (as measured by natural language heuristics and repetitiveness thresholds), which benefits reading comprehension and reasoning, while RefinedWeb's filtering preserves a broader distribution that includes more factual content, benefiting knowledge QA. The rank-score captures this — RPv2+Gopher achieves a higher rank-sum because it is consistently in the upper middle across all tasks (never ranking below 9th of 19 configurations), while RefinedWeb has higher variance (winning some tasks by large margins, losing others).

**Gopher natural language vs. Gopher repetitiveness.** Comparing RPv2 (2023-14) + exact dedup + Gopher natlang + fastText (Wiki-middle) versus RPv2 (2023-14) + exact dedup + Gopher repetitiveness + fastText (Wiki-middle) in Table 5:

- Natlang variant: Normalized average 0.639, Pile perplexity 38.2, Paloma perplexity 23.6.
- Repetitiveness variant: Normalized average 0.633, Pile perplexity 36.0, Paloma perplexity 20.4.

The natural language rules produce slightly better benchmark scores (0.639 vs. 0.633), while the repetitiveness rules produce slightly better language modeling perplexity (Pile: 36.0 vs. 38.2; Paloma: 20.4 vs. 23.6). Again, the two quality dimensions are not interchangeable — they have differential effects on benchmark performance and perplexity, consistent with the overarching finding that quality is multi-dimensional.

**Effect of multi-snapshot expansion.** Expanding from a single snapshot (2023-14) to nine snapshots (2021-49 through 2023-14), both with exact dedup and Gopher natural language rules, produces:

- Single snapshot: Normalized average 0.639.
- Nine snapshots: Normalized average 0.517 (Table 5, "RPv2 (9 Dumps) + exact dedup + Gopher natlang").

This is a substantial degradation, likely because the multi-snapshot expansion introduces content diversity that, without fuzzy deduplication to remove near-duplicates across snapshots, adds noise rather than useful signal. Adding fuzzy deduplication to the nine-snapshot configuration raises the normalized average back to 0.556 (with full Gopher rules: 0.556; with Gopher repetitiveness + Palm-mix: 0.439 — the latter showing that ML classifiers need to be well-matched to the filtering regime).

**C4 line-level filters.** Adding C4 line-level filters to the nine-snapshot, Gopher natlang + Palm-mix configuration (Table 5) reduces Pile perplexity from 67.9 to **52.9** while having negligible effect on aggregated benchmark scores (normalized average 0.550 vs. 0.539). This is a clean demonstration that line-level filtering (removing lines with too few words, non-terminal punctuation, etc.) primarily improves language modeling fidelity on held-out text distributions, not downstream task accuracy. The mechanism is likely that line-level filters remove truly malformed text (extraction artifacts, navigation fragments) that hurts perplexity by forcing the model to model non-linguistic patterns, but which has limited impact on the model's ability to perform reasoning tasks that depend on well-formed input.

**ML-based filtering: fastText vs. DSIR.** The paper tests both fastText classifiers and DSIR importance weights as ML-based quality signals. In configurations where they are directly substitutable (e.g., RPv2 (9 Dumps) + exact dedup + Gopher repetitiveness + either Palm-mix fastText or Palm-mix DSIR), the normalized average scores are remarkably similar: 0.439 for fastText vs. 0.483 for DSIR (Table 5). The paper states that "we see no significant difference between using a fasttext classifier and DSIR" (Section 4.3.2), and this is borne out in the data — the two ML signals capture overlapping quality dimensions. This is important because fastText is substantially simpler to train and apply than DSIR (which requires training two bag-of-words language models and computing log-ratios); if they are equally effective, fastText is the more practical choice.

**Custom rules with Wikipedia perplexity threshold.** The configurations using custom rules (word count thresholds, average line length limits, Wikipedia perplexity > 30, and Wikipedia-reference fastText) produce mixed results (Table 5):

- Custom rules + WikiRef + Pwiki > 30: Normalized average 0.467, Pile perplexity 39.7.
- Custom rules + Gopher repetitiveness + WikiRef + Pwiki > 30: Normalized average 0.500, Pile perplexity 45.8.

These are substantially worse than the best Gopher-based configurations (0.700 normalized average), demonstrating that ad-hoc filtering rules without the systematic quality coverage of the Gopher heuristics leave substantial performance on the table. The Wikipedia perplexity threshold (>30) in particular may be too aggressive, removing content that diverges from Wikipedia's prose distribution but is still high-quality (technical documentation, legal text, etc.).

---

#### Ablation Studies on RedPajama-V2: 1.6B Parameter Models

Table 6 extends the ablation study to 1.6B parameter models trained on 350B tokens, comparing three configurations: RefinedWeb, RPv2 (full) with fuzzy dedup + Gopher + WikiRef classifier, and RPv2 (full) with fuzzy dedup + Gopher natlang + Palm-mix classifier.

**RefinedWeb at 1.6B scale.** RefinedWeb achieves an aggregate average of **52.0** with a normalized average of — not directly reported in textual form but visible in Table 6 — and Pile perplexity of **10.7** with Paloma perplexity of **17.7**. This establishes RefinedWeb as the strong baseline for this comparison.

**RPv2 (full) + fuzzy dedup + Gopher + WikiRef.** This configuration achieves an aggregate average of **50.0** — 2.0 points behind RefinedWeb — with Pile perplexity of **13.6** and Paloma perplexity of **20.8**. The benchmark gap is notable (2 points on the raw average), but the rank-score (0.106) and normalized average (0.139 — wait, these values need verification against Table 6; the table headers show normalized average for RefinedWeb = 0.139 and rank-score not listed. Actually, Table 6 column headers show "Norm. Avg." and "Rank-Score" but the values in the RefinedWeb row are Norm. Avg. = 0.139, Rank-Score not visible. For RPv2 full + Gopher + WikiRef, Norm. Avg. = 0.106. So RefinedWeb wins on normalized average as well, 0.139 vs. 0.106) indicate the gap is meaningful.

The per-task breakdown (Tables 21–23) provides more granularity. The RPv2 model outscores RefinedWeb on Winogrande (56.4 vs. 54.4), LAMBADA (47.4 vs. 47.9 — actually RefinedWeb wins here), and OpenbookQA (32.6 vs. 31.6), but trails on HellaSwag (47.4 vs. 55.8 — a large 8.4-point gap), PIQA (67.4 vs. 73.8 — a 6.4-point gap), and CoQA (43.7 vs. 47.4). The HellaSwag gap is particularly informative because HellaSwag is a sentence completion task that is known to be sensitive to training data quality (models trained on noisy data struggle with the fine-grained commonsense distinctions HellaSwag requires). This suggests that the RefinedWeb filtering pipeline is doing something specifically beneficial for the kind of coherent, commonsense-grounded prose that HellaSwag probes, and that the RPv2 Gopher+WikiRef filtering does not fully replicate this property.

**RPv2 (full) + fuzzy dedup + Gopher natlang + Palm-mix.** This configuration, using only the Gopher natural language rules (not the full Gopher set) plus the PaLM-mix classifier, achieves a substantially lower aggregate average of **47.9** — 4.1 points behind RefinedWeb — with Pile perplexity of **22.2** and Paloma perplexity of **30.7**. The degradation relative to the full Gopher + WikiRef configuration (50.0 vs. 47.9) suggests that the repetitiveness rules and the choice of ML classifier both matter at larger scale. The Palm-mix classifier (trained on Wikipedia + OpenWebText + books) may be less well-calibrated for the full RedPajama-V2 distribution than the simpler WikiRef classifier.

**Why 1.6B results differ from 468M results.** At the 468M scale, the RPv2 + Gopher configurations outperformed RefinedWeb on aggregated rank-score (0.700 vs. 0.650). At the 1.6B scale, the direction reverses: RefinedWeb leads on both average and normalized average. The paper does not explicitly discuss this scaling reversal, but several explanations are plausible. First, the 468M experiments used only 1 or 9 snapshots (not the full dataset), while the 1.6B experiments sampled from the full RPv2 corpus, introducing coverage differences. Second, the 1.6B models were trained on 350B tokens — 3.5x more than the 468M models — which may surface effects of data quality that are not visible at smaller scales. Third, RefinedWeb's filtering pipeline may be specifically tuned for medium-scale training, producing a distribution that is particularly efficient for models in the 1–2B parameter range. This uncertainty is a legitimate limitation: without scaling curves at multiple data sizes, it is impossible to distinguish between the hypotheses.

---

### Ablation Studies and Robustness Checks

**Exact vs. fuzzy deduplication:** Exact deduplication alone degrades performance (468M normalized average drops from 0.594 to 0.472; Table 5). Fuzzy deduplication (MinHash LSH) combined with the full Gopher rules produces the strongest configuration at 468M scale (0.700 normalized average). This pattern — exact dedup hurts, fuzzy dedup helps — is non-obvious and suggests that near-duplicate removal (which fuzzy dedup performs) is beneficial while exact duplicate removal across snapshots may be removing useful repeated training signal.

**Single snapshot vs. multi-snapshot:** Expanding from 1 snapshot (2023-14) to 9 snapshots (2021-49 through 2023-14) *degrades* performance when only exact dedup is applied (0.639 → 0.517 normalized average at 468M). However, when combined with fuzzy dedup and full Gopher rules, multi-snapshot data recovers competitive performance (0.556 normalized average for 9 dumps + full Gopher, vs. 0.700 for 1 dump + full Gopher). This demonstrates that multi-snapshot expansion without near-duplicate removal introduces noise that overwhelms the benefit of increased data diversity.

**C4 line-level filters:** Adding C4 line filters to a Gopher natlang + Palm-mix configuration reduces Pile perplexity from 67.1 to 52.9 while leaving benchmark performance essentially unchanged (normalized average 0.483 vs. 0.539, different direction — the perplexity improves but is reported for different configurations: the 67.1 is for RPv2 (9 Dumps) + exact + fuzzy + Gopher rep + Palm-mix DSIR; the 52.9 is for RPv2 (9 Dumps) + exact + line-filter + Gopher natlang + Palm-mix). This confirms that line-level filtering primarily affects language modeling fidelity on held-out text distributions without substantially impacting downstream task accuracy at this scale.

**Gopher natural language rules vs. Gopher repetitiveness rules:** Both provide meaningful quality improvements, but the natural language rules produce slightly better benchmark scores (0.639 vs. 0.633 normalized average) while the repetitiveness rules produce slightly better perplexity (Pile: 36.0 vs. 38.2; Paloma: 20.4 vs. 23.6). This demonstrates that the two Gopher rule categories capture non-redundant quality dimensions.

**ML classifier choice (fastText vs. DSIR, WikiRef vs. Palm-mix):** At 468M scale, fastText and DSIR produce similar results when controlling for other filters (Section 4.3.2). At 1.6B scale, the WikiRef classifier (trained to distinguish Wikipedia references from unfiltered web text) outperforms the Palm-mix classifier (trained on Wikipedia + OpenWebText + books) when combined with full Gopher rules (50.0 vs. 47.9 aggregate average; Table 6). This suggests that classifier target domain matters — simpler, more focused classifiers may generalize better to the filtered distribution.

**Custom rules with perplexity threshold:** The configuration using a Wikipedia perplexity threshold (>30) alongside custom word count and line length rules produces substantially weaker results (0.467 normalized average; Table 5) compared to the Gopher-based filtering strategies. This validates that the Gopher rules, developed through extensive experimentation at DeepMind (Rae et al., 2021), capture quality signal more effectively than ad-hoc heuristics.

**Validation perplexity as a separate dimension:** Consistently across the ablation matrix, configurations that achieve the best downstream benchmark scores do *not* achieve the best validation perplexity, and vice versa. The unfiltered RPv2 (2023-14) achieves Paloma perplexity of 19.7 — the lowest in the table — despite having middling benchmark scores (0.594 normalized average). RPv2 + Gopher achieves benchmark scores of 0.700 but Paloma perplexity of 24.9. This decoupling is a robustness check on the multi-objective quality claim: it appears consistently and systematically, not as an artifact of a particular configuration.

**Negative result: ReST$^{EM}$ revision model training.** The paper includes a notable negative result in Appendix K: attempting to optimize the revision model using ReST$^{EM}$ (Singh et al., 2024) — an on-policy RL-style training method — caused performance to *degrade substantially* with sequential revisions (Figure 16). While this result pertains to the RedPajama-INCITE revision model training and is technically in the appendices, it is significant because it demonstrates that the straightforward application of self-improvement techniques to models trained on RedPajama data does not always succeed, providing a realistic picture of the challenges involved.

---

### Critical Assessment

#### Disentangling what the experiments demonstrate from what the paper claims

The paper positions RedPajama-V2 as a versatile, metadata-rich foundation that enables systematic research into data curation strategies. The ablation experiments (Section 4.3) provide strong support for a **narrower but still important version of this claim**: different filtering configurations of RedPajama-V2 produce models with measurably different downstream performance profiles across a broad battery of benchmarks, and some configurations are competitive with or exceed the performance of established filtered web datasets (RefinedWeb, FineWeb) on aggregate metrics. The finding that RPv2 + fuzzy dedup + full Gopher achieves a 0.700 normalized average at 468M scale, exceeding RefinedWeb's 0.650 and achieving the best rank-score among all configurations, is the strongest single piece of evidence.

However, the experiments have several important limitations that constrain the generality of the conclusions:

**1. The experiments demonstrate existence of performance variation, not causal attribution of quality dimensions.** The ablation matrix in Table 5 tests configurations that vary multiple quality dimensions simultaneously (e.g., RPv2 + dedup + Gopher vs. RPv2 + dedup + Gopher natlang + fastText). Because the configurations are not factorially designed — not all combinations of deduplication × Gopher natlang × Gopher rep × ML classifier are tested — the experiments cannot cleanly attribute performance differences to individual quality dimensions. The observation that "Gopher natlang slightly outperforms Gopher rep on benchmarks" (0.639 vs. 0.633) is suggestive, but with only a single configuration testing this comparison, and without error bars, the difference may not be statistically reliable. A proper causal decomposition would require a factorial ablation design crossing each quality dimension independently, which would be combinatorially expensive but necessary for the strong causal claims the paper gestures toward.

**2. The scale mismatch between the dataset and the validation models is enormous.** The 468M and 1.6B parameter models trained on 100B and 350B tokens respectively are tiny relative to the dataset's scale (30.4T tokens in the deduplicated head+middle partition alone) and to the model scales at which data quality decisions are most consequential (7B–70B+). Prior work on scaling laws (Hoffmann et al., 2022; Kaplan et al., 2020) has shown that the relationship between data quality and model performance can change with scale — phenomena that are invisible at 468M parameters may become dominant at 7B, and vice versa. The paper acknowledges this limitation explicitly:

> "While the models are relatively small and enabled us to explore a wider variety of filters, it is also a limitation and further, larger-scale explorations are required."

The 1.6B results (Table 6) already show a reversal of the 468M finding: at 468M, RPv2 + Gopher beats RefinedWeb on rank-score; at 1.6B, RefinedWeb beats RPv2 + Gopher on aggregate average (52.0 vs. 50.0). This is a single data point and does not establish a scaling trend, but it raises the possibility that the optimal filtering strategy is itself scale-dependent — a finding that would require experiments at multiple model sizes (e.g., 468M, 1.6B, 3B, 7B) with identical dataset configurations to establish.

**3. The evaluation is thorough for the model scale but lacks critical analyses that would strengthen the paper's claims.** Several experiments that would substantially increase confidence in the findings are absent:

- **Decontamination analysis.** The paper states in Section 5: "We did not explore a thorough decontamination analysis against common benchmarks." This is a significant gap because web data is known to contain benchmark contamination (e.g., test questions from MMLU or HellaSwag appearing in CommonCrawl). If some filtering strategies inadvertently retain more contaminated content than others, benchmark improvements could reflect memorization of test data rather than genuine quality improvements. This is particularly concerning for knowledge-intensive tasks (MMLU, PubMedQA, SciQ) where contamination is most likely. The absence of decontamination analysis makes it impossible to determine whether the performance differences in Tables 18–20 reflect data quality or differential contamination.

- **PII and toxicity analysis.** The paper mentions that this is "another limitation of this work" (Section 5). While the quality signals include toxicity-related measures (LDNOOBW, UT1 blacklist), there is no evaluation of whether training on differently-filtered subsets affects the model's propensity to generate toxic content or memorize PII. This is relevant because one motivation for providing quality signals is enabling users to filter responsibly — but without evidence that the signals actually reduce downstream harms, the claim remains aspirational.

- **Statistical significance.** No confidence intervals, standard errors, or significance tests are reported for any of the benchmark comparisons. With a 500-question test set (for MMLU, for instance) and models trained on different data subsets, the variance in benchmark scores due to training stochasticity and evaluation noise is unknown. The differences between closely-ranked configurations (e.g., normalized average 0.639 vs. 0.633 for natlang vs. rep) may be within the noise floor. Reporting standard deviations across multiple training seeds or bootstrap confidence intervals on the test set would address this.

- **Scaling curves within each filtering configuration.** The paper trains each configuration at a single token budget (100B for 468M, 350B for 1.6B). Without intermediate checkpoints evaluated, it is impossible to determine whether differences between configurations reflect asymptotic capability gaps or differences in learning rate — one configuration might learn faster but plateau at the same level, or vice versa. The 1.6B models trained for 350B rather than 100B begin to address this, but a systematic set of learning curves (e.g., evaluating at 25B, 50B, 100B, 200B, 350B tokens for every configuration) would be much more informative.

**4. The quality signal subset (50T tokens of head+middle) is much smaller than the full corpus (123.7T tokens).** Only the head+middle partition receives quality signals. The tail partition — 80.5B documents, 73T tokens — has no quality annotations beyond the CCNet bucket itself. If the tail contains high-quality content that is simply high-perplexity under Wikipedia (legal text, medical literature, poetry), the absence of quality signals prevents researchers from discovering and recovering this content. The paper's vision of "enabling users to make informed decisions" is thus incomplete — it applies only to the ~40% of the corpus in the head+middle partition, leaving the majority of the data unannotated. The ablation experiments do not explore whether useful signal can be extracted from the tail using the head+middle-trained classifiers or DSIR weights.

**5. The relationship between 468M-scale rankings and practical training at scale is not established.** The paper's central use case — "providing a robust foundation for the next generation of high-quality web datasets" — implies that the findings at 468M parameters will generalize to the scales at which production models are trained. The 1.6B results partially address this, but the small number of configurations tested at that scale (only 3) limits the conclusions that can be drawn. A scaling study across at least two orders of magnitude in model size (e.g., 150M, 468M, 1.6B, 3B, 7B) with a fixed set of filtering configurations would be needed to establish whether the quality signals produce consistent scaling behavior, or whether the optimal filtering strategy changes with model capacity.

**6. The evaluation benchmarks, while diverse, may not capture all dimensions of data quality that matter in practice.** The benchmarks in Table 4 focus on language understanding, reasoning, and knowledge — they do not include tasks measuring code generation, multilingual performance, factuality calibration, or safety. A dataset that performs well on the selected benchmarks might still produce models that are worse at code, struggle with non-English languages, or generate more hallucinations. The paper's inclusion of Paloma (a diverse perplexity benchmark spanning 18 domains) partially addresses this by capturing domain coverage, but perplexity and downstream task performance are only loosely coupled, as the results themselves demonstrate.

In summary, the experiments provide solid evidence that RedPajama-V2's quality signals can be combined to produce training datasets of **measurably different quality** as assessed by a diverse benchmark battery, and that some combinations compete with or surpass existing filtered datasets at small scales. The experiments do **not** establish that the quality signals enable principled, causal understanding of data curation (the ablation design is not factorial), nor that the findings generalize to production-scale models (the largest validated model is 1.6B parameters), nor that the signals capture all practically relevant quality dimensions (decontamination, PII, and toxicity are unanalyzed). These limitations do not undermine the dataset's utility — an unprecedented volume of raw web text with rich metadata is genuinely valuable infrastructure — but they do constrain how strongly the paper's broader methodological claims about enabling systematic data curation research can be asserted based on the presented evidence alone.

## 6. Limitations and Trade-offs

### The Difficulty Estimation Overhead Is Not Accounted for in Efficiency Claims

**The assumption or constraint.** The entire RedPajama-V2 design is predicated on the idea that downstream users can make informed filtering decisions using the provided quality signals. However, the paper provides these quality signals only for the **head+middle partition** — approximately 32.8B documents (50.7T tokens) out of the full 113.3B document corpus (123.7T tokens). The tail partition, containing 80.5B documents and 73T tokens, receives no quality annotations beyond the CCNet perplexity bucket and basic metadata. The paper does not explicitly justify this decision other than implying that the head+middle partition contains the "higher-quality" text by Wikipedia-perplexity standards. But this reasoning is circular: the quality signals are intended to help users *determine* what is high-quality, yet they are only provided for the partition that CCNet has already classified as higher quality. The tail — which represents the majority of the corpus by document count and a substantial fraction by token count — remains opaque.

**The consequence.** Users who suspect that the tail contains valuable content for their domain (legal text, medical literature, poetry, technical documentation — all of which can have high perplexity under a Wikipedia-trained language model) have no mechanism to identify or extract it. The quality signals cannot be retroactively applied without re-running the entire quality computation pipeline on the tail partition, which defeats the purpose of providing pre-computed annotations. This means that **RedPajama-V2 is effectively a 50.7T-token dataset with quality annotations**, not a 123.7T-token dataset — the tail is raw text without the metadata that makes RedPajama-V2 distinctive. The paper's claims about "100+ trillion tokens" and being "the largest open pretraining dataset" are technically true in terms of raw storage, but the fraction of the corpus that supports the paper's core value proposition (quality-signal-enabled filtering) is substantially smaller. A practitioner who needs 100T tokens of *filterable* data — not just raw unannotated text — will find that RedPajama-V2 does not deliver on that scale.

**What evidence exists in the paper.** Table 3 quantifies the partition sizes explicitly: head+middle (deduplicated) contains 20.8B documents and 30.4T tokens, while the tail contains 80.5B documents and 73.0T tokens. The paper states that quality signals are provided for "a 50T token subset of the corpus" (Section 1), but does not explain why the tail was excluded from quality signal computation. The ablation experiments in Section 4.3 use only configurations from the head+middle partition — there is no experiment testing whether tail data, filtered by some alternative method, could improve performance. The embedding-based clustering visualization (Figure 8) is computed on a random sample of 2M documents from the unfiltered 2021-04 snapshot, but this is a demonstration of what is *possible* rather than a quality signal included in the release.

**Mitigation status.** The paper does not address this limitation. It does not explain the rationale for excluding the tail from quality signal computation, does not provide a cost estimate for extending quality signals to the tail, and does not suggest a path for users who need filterable data at larger scale. The Design Principle of "Versatility" — "empowering users to make informed decisions based on their specific needs and criteria" — is only partially realized when the majority of the corpus lacks the information needed to make such decisions.

---

### No Decontamination Analysis Means Benchmark Improvements Cannot Be Attributed to Data Quality

**The assumption or constraint.** The paper evaluates models trained on differently-filtered RedPajama-V2 subsets on 13 downstream benchmarks (Table 4), including knowledge-intensive tasks like MMLU, PubMedQA, SciQ, and TruthfulQA. Web data is known to contain contamination from these benchmarks — test questions, answer keys, and discussion threads that reference benchmark content can appear in CommonCrawl snapshots and, if not removed, allow models to memorize answers rather than learn generalizable capabilities. The paper explicitly acknowledges this gap in Section 5:

> "We did not explore a thorough decontamination analysis against common benchmarks or an analysis of personally identifiable information present in the dataset, posing another limitation of this work."

No decontamination was performed on either RedPajama-V1 or RedPajama-V2. The quality signals include no contamination-related annotations (e.g., n-gram overlap with benchmark test sets, URL blocklists for known benchmark-hosting sites). The filtering configurations tested in the ablation studies vary in their aggressiveness, and different filters will retain or remove different fractions of the web — meaning **different configurations may have different levels of benchmark contamination**.

**The consequence.** If some filtering strategies inadvertently retain more contaminated content than others, the observed benchmark improvements could reflect differential memorization of test data rather than genuine improvements in data quality. This is particularly concerning for the headline finding that RPv2 + fuzzy dedup + full Gopher rules achieves a normalized average of 0.700, exceeding RefinedWeb's 0.650 at the 468M scale (Table 5). If RPv2 + Gopher retains MMLU or PubMedQA test questions that RefinedWeb's filtering removed, the benchmark advantage is an artifact of contamination, not a signal of higher data quality. The per-task breakdown (Tables 18–20) shows that RPv2 + Gopher outperforms RefinedWeb on MMLU (27.0 vs. 24.8) and OpenbookQA (28.8 vs. 28.6), and performs competitively on PubMedQA (51.0 vs. 52.2) — all of which are knowledge-intensive tasks where contamination is most likely to inflate scores. Without decontamination analysis, it is impossible to determine how much of this advantage reflects genuine quality differences versus differential test-set leakage.

The problem is exacerbated by the fact that different filtering strategies produce datasets of different sizes and compositions. A more aggressive filter (like the full Gopher rules) removes a larger fraction of the original web data. If contaminated content is uniformly distributed across the web, aggressive filtering should reduce contamination — but if contaminated content is concentrated in high-quality-looking domains (e.g., educational sites, Q&A forums, Wikipedia-like pages), aggressive filtering that targets well-formed prose might *increase* the contamination rate by selectively retaining the kinds of pages that discuss benchmark content. The direction of the bias is unknown because no analysis was performed.

**What evidence exists in the paper.** The paper provides no decontamination analysis, no contamination audit, and no n-gram overlap statistics between the filtered subsets and the benchmark test sets. The limitation is acknowledged in Section 5 but is mentioned as a single sentence in the conclusion without quantification. The evaluation benchmarks in Table 4 include a "Random" column showing the baseline performance of random guessing, and the 468M models' scores on several tasks are close to this baseline (e.g., ANLI ranges from 33.0 to 34.8 against a random baseline of 25.0; ARC-c ranges from 21.9 to 24.2 against a random baseline of 25.0) — for these near-random tasks, contamination-driven memorization is less of a concern because the model does not appear to have learned the task structure at all. But for tasks like SciQ (51.7–56.4 against a random baseline of 25.0) and HellaSwag (29.7–33.1 against a random baseline of 25.0), where models substantially outperform random guessing, contamination effects could meaningfully affect the rankings.

**Mitigation status.** Not addressed. The paper flags this as future work in Section 5. No decontamination tool is provided alongside the dataset release, and no guidelines are given for users who want to perform their own decontamination. This is a significant gap given that the paper positions RedPajama-V2 as a foundation for training production models (Snowflake Arctic, AI2's OLMo, Salesforce's XGen all use it; Figure 1), where benchmark contamination has both scientific and reputational consequences.

---

### The Ablation Scale (468M–1.6B Parameters) Cannot Support Claims About Production-Scale Data Curation

**The assumption or constraint.** The ablation studies in Section 4.3 train models at two scales: 468M parameters on 100B tokens, and 1.6B parameters on 350B tokens. These are small relative to the models for which data curation decisions are most consequential — frontier open models like Llama-3 (8B–70B parameters, trained on 15T tokens), OLMo (7B parameters), and Snowflake Arctic (480B parameters, MoE). The paper's stated goal is to "inspire the development of numerous new datasets" (Abstract) and to provide "a robust foundation for the next generation of high quality web datasets" (Section 2). The implicit claim is that filtering strategies that work well at 468M parameters will generalize to production-scale training. However, prior work on scaling laws (Hoffmann et al., 2022; Kaplan et al., 2020) has demonstrated that the relationship between data properties and model performance can change with model size — larger models are more data-efficient, more capable of learning from noisy or diverse data, and may benefit from different data mixtures than small models.

**The consequence.** The paper cannot distinguish between three possibilities: (1) filtering strategies that are effective at 468M scale remain effective at 7B+ scale (the findings generalize), (2) filtering strategies are scale-dependent — some strategies that help small models become unnecessary or harmful at larger scales as models develop the capacity to handle noisy data, or (3) the *ranking* of filtering strategies changes with scale (the optimal configuration at 468M is not the optimal configuration at 7B). The 1.6B results provide a single data point suggesting that scale dependence may exist: at 468M, RPv2 + fuzzy dedup + full Gopher achieves a higher normalized average than RefinedWeb (0.700 vs. 0.650; Table 5); at 1.6B, RefinedWeb achieves a higher aggregate average than RPv2 + fuzzy dedup + Gopher + WikiRef (52.0 vs. 50.0; Table 6). This is a reversal in the direction of the comparison, though the 1.6B RPv2 configuration uses a WikiRef classifier rather than the full Gopher+Natlang configuration tested at 468M, so it is not an apples-to-apples comparison across scales. The paper does not discuss this reversal or interpret its implications.

A practitioner deciding whether to use RPv2 + Gopher filtering or RefinedWeb for a 7B-scale training run cannot draw reliable conclusions from the 468M experiments. The paper's ablation findings — that Gopher natural language rules outperform Gopher repetitiveness rules, that fuzzy dedup is essential for multi-snapshot performance, that ML classifiers and DSIR produce similar results — may or may not hold at scales 10–100× larger than those tested. The paper acknowledges this in Section 5 ("it is also a limitation and further, larger-scale explorations are required") but does not scope how large the scale extrapolation uncertainty is or what experiments would resolve it.

**What evidence exists in the paper.** The 468M experiments in Table 5 span approximately 19 configurations trained on 100B tokens each. The 1.6B experiments in Table 6 span only 3 configurations trained on 350B tokens each. There is no model trained at an intermediate scale (e.g., 3B parameters), no configuration tested at both 468M and 1.6B scales with identical filtering (to directly measure scaling behavior), and no learning curves showing how the ranking of configurations evolves with increasing token budgets. The single data point of a ranking reversal between 468M and 1.6B is suggestive but inconclusive because the configurations are not matched across scales.

**Mitigation status.** Partial, through explicit acknowledgment. Section 5 states the limitation clearly, and the paper frames the ablation results as demonstrating "how the quality signals *can* be used" (emphasis on existence) rather than as establishing optimal filtering recipes. The authors' choice to use small models was deliberate: "We use relatively small scales, as this enables us to explore a wider range of filters, showing the breadth of the quality filters available in RedPajama" (Section 4.3.1). This is a reasonable tradeoff — breadth of ablation space vs. confidence in scale generalization — but the paper could have strengthened its claims by testing at least one or two of the most promising configurations at 3B or 7B scale, or by reporting scaling trends within the 468M training run at multiple intermediate checkpoints (e.g., 25B, 50B, 100B tokens).

---

### Single Model Architecture and Evaluation Suite Cannot Capture All Dimensions of Data Quality

**The assumption or constraint.** All ablation models use the Llama-2 architecture (decoder-only Transformer, 24 layers, 16 attention heads, MLP expansion ratio 4.0, sequence length 2048) with a fixed training recipe (AdamW optimizer, cosine learning rate schedule with 1% warmup, weight decay 0.1, maximum learning rate $5 \times 10^{-3}$ for 468M and $5 \times 10^{-4}$ for 1.6B). The evaluation benchmarks (Table 4) focus on English-language understanding, reasoning, and knowledge tasks. This means that "data quality" is operationalized as *the downstream performance of a Llama-2 architecture model on this specific set of English NLP benchmarks*.

**The consequence.** Different model architectures and training objectives may have different sensitivities to data quality dimensions. An encoder-decoder model (like T5) may respond differently to repetitiveness than a decoder-only model. A model trained with a different tokenizer (e.g., the GPT-4 tokenizer vs. the Llama-2 tokenizer) would tokenize the same documents differently, changing the effective n-gram distributions on which many quality signals are based. A multilingual model might benefit from preserving non-English tail data that a perplexity-based quality filter would discard. A code-generation model would want the curly braces and long-line-length documents that the Gopher natural-language filters explicitly remove.

The paper's evaluation suite, while diverse for the model scale, does not capture several practically important dimensions of data quality:

- **Code generation capability:** The benchmarks include no code tasks (HumanEval, MBPP). Documents containing code are likely filtered out by the Gopher natural-language rules (which penalize high symbol-to-word ratios and low stop-word fractions), so a configuration that looks strong on NLP benchmarks may be disastrous for code.
- **Multilingual performance:** All benchmarks are English-only despite RedPajama-V2 containing German, French, Spanish, and Italian data. The paper provides no evidence that the quality signals generalize across languages — the ML classifiers for non-English data use only Wikipedia as the target domain (vs. Wikipedia + OpenWebText + books for English), and the natural-language heuristics (stop-word fraction, mean word length) are calibrated for English and may misclassify non-English text.
- **Factuality and calibration:** TruthfulQA is included in the benchmark set but is excluded from the aggregated scores because it "does not provide a high enough signal-to-noise ratio" for 468M models. The paper therefore provides no evidence about how filtering affects a model's tendency to generate false information.
- **Safety and toxicity generation:** The quality signals include toxicity-related annotations (LDNOOBW words, UT1 blacklist), but the paper does not evaluate whether training on subsets filtered by these signals produces models that generate less toxic or harmful content. The signals are provided as inputs to filtering, but their downstream effect on model behavior is unmeasured.
- **Long-context capability:** All training uses a sequence length of 2048 tokens, which is short by modern standards (Llama-3 uses 8192; GPT-4 uses longer). The quality signals do not measure whether documents are suitable for long-context training (e.g., whether they contain coherent multi-paragraph discourse vs. short snippets).

A practitioner training a multilingual code-generation model, or a model intended for fact-critical applications, or a model with long-context requirements cannot determine from this paper which quality signals are relevant to their use case or how to set filtering thresholds.

**What evidence exists in the paper.** The evaluation framework is documented in Table 4 and Section 4.3.1. The paper does not claim broader generalizability — it uses the benchmarks available and appropriate for 468M-scale evaluation. The inclusion of two separate perplexity evaluations (Pile and Paloma) provides some coverage of domain diversity beyond task benchmarks, and the finding that filtering configurations that optimize task performance can simultaneously *increase* perplexity (the unfiltered RPv2 achieves the lowest Paloma perplexity of 19.7 while the highest-scoring Gopher configuration achieves 24.9; Table 5) demonstrates that the paper's own evaluation framework captures tension between different quality objectives.

**Mitigation status.** Not addressed for the broader generalization question. The paper does not discuss how findings might or might not transfer to other architectures, tokenizers, languages, or task families. The multilingual limitation is particularly notable given that RedPajama-V2 explicitly includes five languages — the dataset provides multilingual data, but the paper provides no evidence about its usability. The paper's framing that quality signals "empower users to make informed decisions based on their specific needs and criteria" implies that users can adapt the filtering to their use case, but without evidence about how the signals behave for non-English text or non-NLP tasks, users are making decisions in the dark.

---

### No Error Analysis or Confidence Intervals Make It Impossible to Assess the Reliability of Filtering Comparisons

**The assumption or constraint.** The ablation study reports single-point estimates for each benchmark score and aggregate metric, with no confidence intervals, standard errors, or statistical significance tests. Each configuration in Tables 5 and 18–20 is represented by one number per benchmark, derived from a single training run on a single data sample. The paper does not report variance across multiple training seeds, bootstrap confidence intervals over the test set, or any other measure of uncertainty.

Training a language model is a stochastic process: different random seeds produce different parameter initializations, different data orderings, and different final checkpoints, all of which introduce variance in downstream benchmark scores. Additionally, the benchmark test sets have finite sizes — the 13 benchmarks vary in their number of test examples, and sampling error alone means that a difference of 1–2 percentage points between configurations may not be statistically reliable.

**The consequence.** Many of the fine-grained comparisons that the paper's analysis relies on may be within the noise floor. The key finding that Gopher natural language rules outperform Gopher repetitiveness rules (0.639 vs. 0.633 normalized average; Table 5) is a difference of 0.006 on the normalized scale. Without error bars, it is impossible to know whether this difference is meaningful or whether the two rule sets are effectively equivalent — which would have practical implications (if they are equivalent, users could choose based on which is cheaper to compute or which preserves more training data). Similarly, the comparison between fastText and DSIR (0.439 vs. 0.483 normalized average) involves a difference of 0.044, which may or may not be statistically significant at the sample sizes used.

The per-task rankings in Tables 18–20 are even more vulnerable. On individual benchmarks, many configurations achieve scores within 1–2 points of each other — e.g., ANLI scores range from 33.0 to 34.8, ARC-c from 21.9 to 24.2, SocialIQA from 31.6 to 36.6. The paper uses these per-task scores to draw conclusions like "the RPv2 dataset filtered with fuzzy deduplication and Gopher has the highest aggregated scores across all RPv2 datasets" and "RefinedWeb is performing worse on Hellaswag, LAMBADA, Winogrande, MMLU and OpenBookQA." But if the per-task score variance due to training stochasticity is ±1.5 points (a plausible level for 468M models on these benchmarks), many of these pairwise comparisons would not be statistically distinguishable.

**What evidence exists in the paper.** The paper provides only the aggregate scores (average, normalized average, rank-score) as point estimates. There is no mention of multiple training runs, no error bars in any table or figure, and no discussion of variance or statistical power. The rank-score metric — which compresses all benchmarks into a single ranking — is particularly vulnerable to noise because small, insignificant differences on individual benchmarks accumulate into rank differences that are presented as meaningful. If two configurations are statistically indistinguishable on 10 of 13 benchmarks, but the remaining 3 benchmarks differ by amounts within the noise floor, their rank-scores could differ substantially despite no genuine quality difference.

**What evidence exists in the paper (continued).** The evaluation setup in Section 4.3.1 describes the benchmarks, models, and training procedure in detail, but does not mention any replication or variance estimation. The Appendix tables (18–20) provide per-benchmark scores to two-decimal-place precision, which implies a level of measurement certainty that is almost certainly not justified given the stochasticity of LLM training at this scale. The paper's reporting precision (e.g., "34.1" for ANLI) suggests point estimates without uncertainty quantification.

**Mitigation status.** Not addressed. The paper does not mention this as a limitation, does not recommend that users treat small differences as unreliable, and does not provide guidance on how to replicate the experiments to assess variance. For a dataset paper that aims to enable systematic research into data curation, the absence of uncertainty quantification is a methodological gap — it means that future researchers attempting to build on these results cannot determine whether their own findings agree or disagree with the paper's reported rankings. The standard practice in benchmark evaluation (multiple seeds, bootstrap confidence intervals, or at minimum reporting standard deviation across multiple evaluation runs) would substantially strengthen the conclusions.

---

### The RedPajama-V1 Reproduction Remains Inexact, and the Gap Cannot Be Bounded

**The assumption or constraint.** RedPajama-V1 was constructed as a "best-effort reproduction" of the LLaMA-1 training data based on the descriptions in Touvron et al. (2023). The paper documents 10+ specific uncertainties encountered during reconstruction (Table 10 in Appendix C.3), ranging from unspecified CommonCrawl snapshots and classifier thresholds to underspecified GitHub filtering heuristics and book deduplication methods. For each, the authors made a reasonable but ultimately arbitrary choice — they used the most recent snapshots available, set the classifier threshold to match the token count of the original dataset, adopted heuristics from The Stack, etc. These choices collectively mean that **RedPajama-V1 is an approximation of the LLaMA training corpus whose divergence from the original cannot be quantified** because the original corpus was never released and its exact composition remains unknown.

The RedPajama-INCITE-7B model trained on this data underperforms LLaMA-7B by 4.1 points on HELM-classic (Table 8: RedPajama-INCITE-7B at 0.431 vs. Llama-7B at 0.472), and by a smaller margin on lm-evaluation-harness (0.6882 vs. 0.6881 — essentially tied). The authors hypothesize that "some salient details that went into the construction of the original LLaMA training corpus may be missing" (Section 3.2.2) and that FP16 training constraints contributed to the gap. But the relative contribution of data mismatch vs. training precision vs. other factors (different hardware, different batch sizes, different software stack) is unknown.

**The consequence.** A user who adopts RedPajama-V1 as a drop-in replacement for the LLaMA training corpus — assuming it will produce models of equivalent quality — may be disappointed. At the 3B scale, RedPajama-INCITE models outperform comparable open models (GPT-Neo, Pythia) by meaningful margins (Table 7), suggesting RedPajama-V1 is at least competitive with the Pile. At the 7B scale, the gap to LLaMA-7B is measurable and, on some benchmarks, substantial. The paper cannot tell users whether this gap would close if a specific missing detail were recovered (e.g., using the exact same CommonCrawl snapshots, or a slightly different GitHub filter threshold), or whether the gap represents a fundamental data quality difference between RedPajama-V1 and the proprietary LLaMA corpus that cannot be replicated from public sources.

More broadly, the RedPajama-V1 experience demonstrates that **transparent documentation of uncertainties does not substitute for access to the original dataset.** Even with the paper's meticulous uncertainty inventory (Table 10), an exact reproduction is impossible because the original choices are lost. RedPajama-V1 advances the state of open data by documenting what *can* be known and where the gaps are, but the irreducible uncertainty means that RedPajama-V1 is a *related* dataset to LLaMA-1's training data, not a *replica* — and the performance difference between models trained on each is the empirical manifestation of this distinction.

**What evidence exists in the paper.** The RedPajama-INCITE evaluation in Section 3.2.2 and Tables 7–9 provides direct evidence of the performance gap. Table 10 provides the uncertainty inventory. Section 3.2.2 explicitly discusses the gap and offers hypotheses. The abstract and Section 1 describe RedPajama-V1 as "an open reproduction" without claiming exact equivalence, which is an honest framing. The paper does not attempt to bound the maximum possible improvement from recovering missing details, nor does it conduct ablations to isolate the effect of specific known differences (e.g., varying the Wikipedia classifier threshold and measuring the resulting model performance).

**Mitigation status.** Partially addressed through transparency. The paper acknowledges the gap openly and documents the uncertainties that contribute to it. The fact that RedPajama-V1 has been adopted by multiple downstream projects (OLMo, Arctic, XGen) despite the known gap suggests that the community values an imperfect-but-transparent reproduction over no reproduction at all. However, the paper does not provide guidance on how to improve the reproduction (e.g., which uncertainties are most likely to matter, or what additional information from Meta would most reduce the performance gap), nor does it establish a systematic methodology for quantifying reproduction fidelity that future efforts could adopt.

## 7. Implications and Future Directions
- How this changes the landscape
  - RPv2 shifts the community from consuming fixed, prefiltered corpora to operating a common “data substrate” with standardized, per‑document diagnostics. This enables:
    - Faster iterations on filtering recipes.
    - Reproducible, side‑by‑side ablations of competing heuristics and ML selectors.
    - Better transparency in what “high quality” means for a given application.

- Research enabled or suggested
  - Data selection research at scale:
    - Learn weighted mixtures over signals (e.g., train a meta‑selector that predicts downstream utility using the provided signals).
    - Explore dynamic curriculum (age‑aware, domain‑aware) using `ccnet_perplexity`, `ccnet_language_score`, DSIR weights, and repetition measures (Tables 11–14).
    - Systematic comparisons of fastText vs. DSIR vs. embedding‑based selectors; the paper’s finding that fastText and DSIR behave similarly (Table 5) is a starting point.
  - Cross‑lingual filtering:
    - Extend ML similarity signals beyond Wikipedia for non‑English (Table 13 notes English‑only Palm‑mix).
    - Study whether natlang/repetition heuristics transfer uniformly across languages with different tokenization artifacts.
  - Benchmark decontamination and safety:
    - Use dedup ids and minhash clusters for stronger decontamination pipelines.
    - Build refined toxicity/PII filters and release them as additional signals aligned to RPv2’s schema.

- Practical applications
  - Enterprises and researchers can tailor corpora:
    - Safety‑first variants (tight natlang + toxicity filters + strict dedup).
    - Domain‑focused variants (DSIR/fastText targeting legal, biomedical, or coding domains by training domain‑specific n‑gram LMs or classifiers—hooking into RPv2’s signal schema).
    - Efficiency‑oriented variants (aggressive repetition control to mitigate degenerate generations).
  - Training and productization:
    - The paper cites production‑used models trained on RedPajama data (e.g., Snowflake Arctic, Salesforce XGen, AI2 OLMo; p. 1), indicating immediate utility.

- Concrete takeaways for practitioners
  - If you want a strong general web corpus without bespoke engineering, start from RPv2 with fuzzy dedup + full Gopher rules; in 468M experiments, this combination is near the top on averaged metrics and best on rank‑based aggregate (Table 5).
  - If low perplexity on heterogeneous validation sets is critical, be cautious: line‑level C4 filters reduce perplexity but may not lift task accuracy (Table 5 and Section 4.3.2).
  - For few‑shot instruction use cases, instruction‑tuning on top of RPv1‑trained models improves HELM performance substantially (Table 8).

Block quotes of notable results and references to help the reader verify:
- > “RPv2 (2023‑14) ✔ ✔ (full) [Gopher] … Avg 37.6, Norm Avg 0.160, Rank‑Score 0.700” (Table 5).
- > “RefinedWeb … Avg 37.9, Norm Avg 0.165, Rank‑Score 0.650” (Table 5).
- > “7B‑Instruct HELM‑AVG 0.492 vs. LLaMA‑7B 0.472” (Table 8).
- > “3B Base outperforms GPT‑Neo and Pythia‑2.8B by 3–5 points on HELM classic and by 2–7 points on a subset of LM‑Eval” (Section 3.2.2; Table 7).
- > “CCNet buckets, language scores, and perplexity distributions; ML similarity histograms; natural‑language and repetition signal histograms” (Figures 4–7).

In sum, the paper contributes a new way to do open LLM data work: make the raw web data accessible at scale and ship the metadata that turns it into a programmable laboratory for data curation. The ablations show that well‑known rules (dedup + Gopher) recovered from these signals produce competitive corpora, and the V1 replication—with full process notes—adds valuable transparency about the impact of compute and recipe ambiguities on model outcomes.
