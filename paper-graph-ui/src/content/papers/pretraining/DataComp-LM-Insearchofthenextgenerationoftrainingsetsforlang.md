# DataComp-LM: In search of the next generation of training sets for language models

**ArXiv:** [2406.11794](https://arxiv.org/abs/2406.11794)

## 🎯 Pitch

DataComp-LM (DCLM) delivers the first large-scale, controlled testbed specifically designed to systematically evaluate and improve training datasets for language models. By providing a massive 240-trillion-token web corpus, standardized training and evaluation recipes across five compute scales, and open-source curation tools, DCLM enables reproducible, apples-to-apples comparisons of data-centric strategies—breaking the current bottleneck where progress is obscured by architecture and compute differences. This empowers the community to build better language models more efficiently, as evidenced by their strong open baseline that rivals state-of-the-art models with far less compute, fundamentally advancing transparent, data-first research in NLP.

---

## 1. Executive Summary

This paper introduces DataComp for Language Models (DCLM), a testbed and benchmark for controlled experiments on language model training data curation. Using DCLM's standardized framework—which provides a 240T-token Common Crawl corpus (DCLM-POOL), fixed training recipes, and 53 downstream evaluations—the authors conduct 416 baseline experiments to isolate the effects of data interventions like text extraction, deduplication, and model-based quality filtering. The central finding is that **model-based filtering**—specifically, training a fastText binary classifier on instruction-formatted data (OpenHermes 2.5 + ELI5) to select the top 10% of documents—is the single most impactful curation step, yielding a new dataset, DCLM-BASELINE, that enables a 7B-parameter model to reach 64% 5-shot MMLU accuracy with 2.6T training tokens. This represents a 6.6 percentage point improvement over the prior open-data state-of-the-art (MAP-Neo) while using 40% less compute, establishing that systematic, model-guided data filtering can substitute for orders of magnitude more pretraining compute—but only when the filtering signal comes from carefully chosen reference data rather than human quality judgments or conventional sources like Wikipedia.

## 2. Context and Motivation

### The Core Problem: We Can't Reliably Compare Training Datasets

The central gap this paper addresses is deceptively simple: **the field of language model training lacks a controlled experimental framework for evaluating data curation strategies**. Despite widespread agreement that training data quality matters enormously for downstream model performance—perhaps as much as model architecture or training compute—there is no standardized way to determine whether one dataset construction method outperforms another.

This gap manifests in a specific, recurring pattern in the literature. Researchers proposing new data curation techniques—filtering heuristics, deduplication algorithms, quality classifiers, data mixing ratios—routinely evaluate their methods by training language models and measuring downstream task performance. However, these evaluations are almost never comparable across papers because they differ along multiple confounding dimensions simultaneously: model architecture (GPT-style, Llama-style, mixture-of-experts), training hyperparameters (learning rate schedules, batch sizes, weight decay), total compute budgets (some train Chinchilla-optimal models, others overtrain), and evaluation suites. As the authors put it (Section 1):

> "researchers often compare models that are trained with different architectures, compute, or hyperparameters. Hence, it is often unclear what data curation strategies work best: Are the results of training set A better than training set B because training set A is truly better, or because the model trained on A was combined with a better architecture, learning rate schedule, or more compute?"

This is not merely an academic concern about experimental hygiene. The cost of training state-of-the-art language models has grown to the point where even well-funded industrial labs cannot afford exhaustive ablations. A single 7B-parameter training run on 2.6T tokens costs thousands of GPU-hours (Table 1: 7,300 H100 hours for the 7B-2x scale). Without standardized benchmarks that allow researchers to isolate the effect of data interventions from architectural and hyperparameter choices, the community risks wasting enormous computational resources pursuing data strategies that appear effective in one experimental setting but fail to transfer to others.

### Why This Matters: Data Is Becoming the Key Bottleneck

The importance of this problem extends beyond experimental methodology into fundamental questions about how language models should be developed and deployed.

**Training data details are increasingly opaque, even for "open" models.** The paper points to a troubling trend in the field: model weights are increasingly released publicly (the Llama, Mistral, and Gemma families), but the training datasets used to produce those weights remain proprietary. The corresponding model documentation provides, at best, "a coarse description of the respective training data, if any at all" (Section 1). This opacity creates a scientific dead end—researchers cannot build on or learn from the data curation strategies that produced the strongest open-weight models. The paper positions DCLM as a direct response to this trend, aiming to produce not just strong models but also **publicly available, well-documented training sets** that enable cumulative scientific progress.

**The economics of pretraining are shifting toward data quality over raw scale.** The Chinchilla scaling laws (Hoffmann et al., 2022) established that for a given compute budget, there is an optimal balance between model parameters and training tokens. However, this analysis implicitly treats all tokens as interchangeable—a "token" from carefully curated Wikipedia is treated identically to a "token" from auto-generated spam. Recent work suggests this assumption is wrong: smaller models trained on higher-quality data can match or exceed larger models trained on noisier data (e.g., the Phi model series). The paper's central claim—that a 7B model trained on DCLM-BASELINE with 2.6T tokens approaches Llama 3 8B performance while using 6.6× less compute—is a concrete demonstration that **data quality improvements can substitute for compute scaling**, with direct implications for how organizations allocate their pretraining budgets.

**Data curation research needs to scale beyond small, ad-hoc experiments.** Prior work on dataset construction for language models has largely been conducted at small scales or in domain-specific contexts. The BabyLM challenge, for instance, focuses on models up to 220M parameters trained on 10M–100M tokens—valuable for understanding data efficiency in low-resource regimes but potentially misleading about what matters at the 7B+ scale. The DataComp vision benchmark (Gadre et al., 2024) pioneered the model of a standardized testbed where participants propose data curation algorithms and evaluate them by training models with fixed recipes, but this paradigm had not previously been applied to language modeling at scale. The paper explicitly builds on DataComp's philosophy while scaling it up by orders of magnitude—from vision datasets to a 240T-token text corpus and 7B-parameter models.

### Where Existing Approaches Fall Short

The paper identifies several specific limitations in the current landscape of training data research and practice:

**1. No controlled comparisons across data curation methods.** The most damaging gap is methodological. Consider two hypothetical papers: Paper A proposes a new perplexity-based filtering method and evaluates it by training a 1.3B-parameter model on 100B tokens with a cosine learning rate schedule. Paper B proposes a deduplication algorithm and evaluates it by training a 2.7B-parameter model on 300B tokens with a constant learning rate. Even if both papers report improvements over their respective baselines, it is impossible to determine which intervention is more effective, or whether combining them would yield additive gains. The effect sizes of data interventions are often modest (a few points on benchmark accuracy), making this confounding problem particularly acute—differences in training recipes can easily swamp the signal from data quality.

**2. Open-source datasets have unclear, unreproducible pipelines.** The paper evaluates several widely-used open-source training sets—C4 (Raffel et al., 2020), RefinedWeb (Penedo et al., 2023), RedPajama (Together Computer, 2023), and Dolma-V1 (Soldaini et al., 2024)—and finds significant performance differences (Table 2: RefinedWeb achieves 36.9 CORE vs. 34.2 for C4 at the 7B-1x scale). However, these datasets differ along multiple dimensions simultaneously: different text extractors, different heuristic filters, different deduplication strategies, and different choices about whether to mix in curated sources like Wikipedia. The paper's systematic ablation experiments (Sections 4.2–4.4) demonstrate that each of these individual choices matters—resiliparse extraction improves CORE by 2.5 points over WET files (Table 3), deduplication pipeline choice affects both token yield and downstream performance (Tables 17–19), and model-based filtering provides the largest single gain (Table 4, 30.2 CORE for fastText vs. 27.5 for the RefinedWeb reproduction). In the absence of controlled experiments, the field has been unable to attribute the success of existing datasets to specific design decisions.

**3. Existing benchmarks are either too small, too narrow, or too disconnected from pretraining.** The paper acknowledges several prior data-centric benchmarks but argues they are insufficient for driving progress on pretraining dataset design. The BabyLM challenge (Warstadt et al., 2023) is limited to 125M–220M parameter models and 10M–100M tokens—regimes where the relationship between data quality and model performance may differ qualitatively from the 7B+ scale. The Data-Juicer effort (Chen et al., 2024) focuses on cleaning and mixing fine-tuning data rather than pretraining data. The Vision DataComp benchmark established the core methodology but in a different modality. The paper positions DCLM as filling a gap: a **large-scale, multi-scale, pretraining-focused benchmark** where interventions can be tested at scales from 400M to 7B parameters with a standardized evaluation suite of 53 downstream tasks.

**4. There is no systematic understanding of what makes a "high-quality" document for pretraining.** A particularly striking finding in the paper is that human quality judgments—widely considered the gold standard for many annotation tasks—**do not correlate with downstream model performance** (Appendix N). When the authors asked 16 AI graduate students and professors to label documents as "good" or "bad" for LM pretraining and then compared various quality filters against these human labels, they found no relationship between a filter's agreement with human annotators and its effectiveness for training better models (Figure 9). The AskLLM filter, which prompts an instruction-tuned model to evaluate document quality, achieved the highest ROC-AUC against human labels (82%) but produced the worst-performing training set among the filters tested (28.5% CORE at 1B-1x, vs. 31%+ for fastText classifiers). Conversely, the best-performing fastText classifier—trained on instruction-formatted data from OpenHermes 2.5 and ELI5—achieved only 73% ROC-AUC against human labels. This counterintuitive result suggests that **human intuition systematically misidentifies which documents are useful for pretraining**, likely by overvaluing "clean" or "informative" documents and undervaluing the diversity and noise that models need to learn robust representations.

**5. Data cleaning is studied in isolation, not as an integrated pipeline.** Prior work typically examines one data intervention at a time—a new deduplication method, a new quality classifier, a new mixing ratio. In practice, training datasets are produced by pipelines that chain together extraction, filtering, deduplication, quality scoring, and mixing steps. The interactions between these steps are poorly understood. For instance, the paper finds that when a dataset is already well-filtered (DCLM-BASELINE), adding high-quality external sources like Wikipedia and books actually **decreases** performance (Table 6: −1.2 CORE points for mixing with RPJ extras), whereas the same mixing **improves** performance for less-filtered datasets like C4 (+2.2 CORE points). This suggests that the optimal curation strategy depends on what other curation steps have already been applied—a finding that could only emerge from a framework where each step can be ablated independently.

### How This Paper Positions Itself

The paper frames itself as infrastructure for the emerging field of data-centric language model research. Rather than proposing a single new method and claiming superiority over prior work, it provides three things:

1. **A benchmark** (DCLM) that standardizes the comparison of data curation strategies by fixing the model architecture, training recipe, and evaluation suite across experiments.
2. **A set of baselines** (416 experiments) that systematically explore the design space of training dataset construction, establishing empirical laws about which interventions matter most and how they interact.
3. **A specific dataset** (DCLM-BASELINE) that applies the best-performing combination of interventions to produce a new state-of-the-art open training set, demonstrating what the benchmark can achieve when used as intended.

The paper explicitly connects to the **DataComp paradigm** (Gadre et al., 2024) from vision, adapting its core principle—"participants iterate on a dataset with a fixed model and training recipe"—to language modeling at scale. It also positions itself relative to **compute-optimal training** (Hoffmann et al., 2022) by providing scaling studies (Figure 3) that show dataset rankings are largely consistent across compute scales (Pearson's r = 0.838 at 400M to r = 0.982 at 3B when predicting 7B performance), validating the multi-scale design that makes the benchmark accessible to researchers with varying compute budgets.

Importantly, the paper does not claim to have solved data curation. Section 6 is explicit about limitations: the experiments are restricted to a single model family (decoder-only Transformers), a single tokenizer (GPT-NeoX), and English-language text. Code and math performance require separate domain-specific datasets (StarCoder, ProofPile2) that were not part of the core filtering experiments. Fairness, multilinguality, and safety are flagged as "important dimensions to expand DCLM along" in future versions. The paper's contribution is establishing the **experimental infrastructure**—the benchmark, the baselines, and the open-source tooling—within which future research on these dimensions can take place.

## 3. Technical Approach

### 3.1 Reader Orientation

This paper builds a systematic experimental testbed—not a single model or algorithm—that allows researchers to measure exactly how much each data curation choice (extraction, filtering, deduplication) improves a language model's downstream performance, independent of architecture or training recipe. The core idea is to fix everything except the training data, then compare models trained on different datasets using identical code, hyperparameters, and evaluations, thereby isolating the effect of dataset design from all other confounding variables.

### 3.2 Big-Picture Architecture (Diagram in Words)

The DCLM system has five major components connected in a linear pipeline:

1.  **DCLM-POOL** — a 240T-token raw text corpus extracted from all Common Crawl data prior to 2023 using the resiliparse HTML parser. This is the starting material that all participants filter or mix from.
2.  **Scaling Tracks** — five fixed compute scales (e.g., 400M-1x, 7B-2x) that specify exactly how many model parameters to use and how many tokens to train on, enabling researchers with different GPU budgets to participate and allowing the study of how data interventions transfer across scales.
3.  **Curation Pipeline** — a sequence of processing steps (text extraction, heuristic filtering, deduplication, model-based quality filtering, optional mixing with external sources) that transforms raw HTML into a training-ready dataset. Each step is independently configurable and ablatable.
4.  **Training Harness** — a fixed training recipe (OpenLM framework, decoder-only Transformer, specific optimizer settings, learning rate schedules) that takes a curated dataset and produces a trained model checkpoint, ensuring that differences between models are attributable solely to differences in their training data.
5.  **Evaluation Suite** — 53 downstream tasks grouped into CORE (22 tasks providing low-variance signal even at small scales) and EXTENDED (all 53 tasks) metrics, scored using centered accuracy normalized so that random guessing equals 0 and perfect accuracy equals 1.

Information flows as follows: a participant selects a compute scale → downloads the corresponding random subset of DCLM-POOL → applies their curation algorithm (filtering track) or combines with external data (mixing track) → tokenizes and shuffles the result → trains a model using OpenLM with fixed hyperparameters → evaluates on all 53 tasks → submits scores to the DCLM leaderboard.

### 3.3 Roadmap for the Deep Dive

- **First**, the **multi-scale competition design** (Section 3.2), because the entire benchmark's accessibility and the validity of its small-to-large-scale transfer claims depend on how these scales are chosen and validated.
- **Second**, the **DCLM-POOL construction and metadata** (Section 3.1, Appendix E), since all filtering-track submissions start from this corpus and its extraction choices affect every downstream result.
- **Third**, the **benchmark tracks** (Section 3.3, Appendix C), which define the rules for what constitutes a valid submission and what data sources can be used.
- **Fourth**, the **training recipe** (Section 3.4, Appendix F), because the fixed hyperparameters are the mechanism that isolates data quality as the independent variable—understanding their specification is essential for interpreting results.
- **Fifth**, the **evaluation metrics** (Section 3.5, Appendix G), focusing on the centered accuracy formulation and the distinction between CORE and EXTENDED task sets.
- **Sixth**, the **data curation pipeline** (Section 4) and **large-scale scaling recipe** (Section 5) that produce DCLM-BASELINE—the concrete instantiation of the benchmark's methodology that achieves state-of-the-art results.

### 3.4 Detailed, Sentence-Based Technical Breakdown

This is primarily an **infrastructure and benchmarking paper** whose core technical contribution is a standardized experimental framework—not a novel model or algorithm. The framework's design choices are the technical contribution: how scales are chosen to enable transferable conclusions, how training recipes are standardized to isolate data effects, and how evaluation metrics are constructed to provide reliable signal across compute regimes.

---

#### DCLM-POOL Construction and Design

DCLM-POOL is an unfiltered web-text corpus comprising all Common Crawl data from 2013 through 2022 inclusive—5.1 million WARC (Web ARChive) files totaling approximately 200 billion documents. The authors deliberately excluded 2023 and later crawls to "prevent large amounts of language model generated text from polluting our datasets and to provide a hold out for future use" (Appendix E). This is a substantive design choice: as LLM-generated content proliferates on the web, including recent crawls would risk training on synthetic text rather than human-authored content, potentially degrading model quality in ways that are poorly understood.

**Text extraction from HTML.** Rather than using Common Crawl's pre-extracted text (the WET files), the authors re-extract text from raw HTML using the `resiliparse` library. This decision is empirically motivated by Table 3, which shows that at the 1B-1x scale, both `resiliparse` and `trafilatura` (used by RefinedWeb) improve CORE accuracy by at least 2.5 percentage points over WET files (24.1 and 24.5 vs. 20.7). The mechanism is that WET files contain substantial boilerplate content—navigation bars, copyright notices, error messages—that the stricter extractors remove. Between `resiliparse` and `trafilatura`, the downstream performance is similar, but `resiliparse` is approximately 8× faster (4.55 MB/sec/core vs. 0.56 MB/sec/core; Table 16), making it practical for processing 240T tokens.

**Metadata fields.** Each document in DCLM-POOL is stored as a gzip-compressed JSON line with the fields shown in Table 9: the extracted text, the source URL, WARC record metadata (date, IP address, content type, payload digest, record ID, target URI), and warcinfo metadata about the source WARC file. This design preserves a one-to-one mapping between raw Common Crawl WARC files and DCLM-POOL JSON lines, enabling the authors to update DCLM-POOL based on Common Crawl's periodic redactions of specific dumps. The total corpus occupies 340TB after gzip compression (370TB raw), tokenized to 240T GPT-NeoX tokens.

**Tokenization.** The GPT-NeoX tokenizer (Black et al., 2022) is used throughout, with a vocabulary size of 50K. The authors note that "other tokenizers may perform better on multilingual tasks or math" (Section 6), but fix the tokenizer across all experiments to maintain comparability. The tokenization and shuffling code is provided in two implementations: a Ray-based distributed version for multi-node processing of large datasets, and a Rust-based single-node version that is more efficient for smaller pools.

---

#### Competition Scales: Enabling Multi-Scale Research

The DCLM benchmark defines five competition scales, each specifying a fixed model size (`N` parameters) and number of training tokens (`D`), summarized in Table 1:

| Scale | Model params | Train tokens | Train FLOPs (≈6ND) | Train H100 hours | Pool size |
|---|---|---|---|---|---|
| 400M-1x | 412M | 8.2B | 2.0e19 | 26 | 469B |
| 1B-1x | 1.4B | 28.8B | 2.4e20 | 240 | 1.64T |
| 3B-1x | 2.8B | 55.9B | 9.4e20 | 740 | 3.18T |
| 7B-1x | 6.9B | 138B | 5.7e21 | 3,700 | 7.85T |
| 7B-2x | 6.9B | 276B | 1.1e22 | 7,300 | 15.7T |

The number of training tokens at each scale follows the formula `D = 20 × N × Chinchilla_multiplier`, so that a 1× multiplier corresponds to the compute-optimal allocation identified by Hoffmann et al. (2022), where model parameters and training tokens are scaled in roughly equal proportion for optimal use of training FLOPs. The 2× multiplier for the largest 7B scale represents an over-trained regime, where the model sees more tokens than the Chinchilla-optimal ratio would prescribe.

**Pool sizes are deliberately constrained** to encourage scalable filtering strategies. For the 400M-1x scale, the initial pool is 469B tokens, meaning participants must select approximately 1.7% of available documents (8.2B / 469B ≈ 1.7%). At the 7B-2x scale, the initial pool is 15.7T tokens, requiring selection of 1.8% of documents. This prevents submissions that keep extremely small fractions of data (e.g., <0.1%) that would not scale to generating trillion-token datasets for frontier models.

**Scale consistency validation (Figure 3).** A critical assumption of the multi-scale design is that the ranking of data curation methods is preserved when moving from small scales to large scales. If the best dataset at 400M parameters were different from the best dataset at 7B parameters, small-scale experiments would be misleading. Figure 3 plots the CORE score of 10 data curation methods at the 7B-1x scale against their scores at the 400M-1x, 1B-1x, and 3B-1x scales. The Pearson correlations are r = 0.838, r = 0.956, and r = 0.982 respectively, confirming that better curation strategies at smaller scales transfer to larger scales with high fidelity. This is the empirical justification for the benchmark's accessibility—researchers with limited compute can confidently iterate at the 400M or 1B scale and expect their findings to hold at 7B.

---

#### Benchmark Tracks: Filtering and Mixing

DCLM offers two competition tracks, described in detail in Appendix C.

**Filtering track.** Submissions must form their training set by applying a processing pipeline to the DCLM-POOL subset corresponding to their chosen scale, without including any external data. The rationale is twofold: (1) fixing the initial data levelizes the playing field, and (2) using a pool size proportional to the training scale prevents filtering strategies that keep extremely small fractions of data and cannot scale to generating large datasets. Two qualifications apply:

1.  **HTML extraction can be modified.** Participants may choose to re-extract text from the original Common Crawl WARC files using a different parser rather than starting from the pre-extracted DCLM-POOL text. This allows experimentation with extraction quality, which Section 4.2 shows is a performance-relevant choice.
2.  **Models trained on external data can be used within the pipeline.** For quality filtering, paraphrasing, or other processing steps, participants may use models trained on external data—with the explicit exception of evaluation data. The rules explicitly prohibit "abusing this allowance to introduce external data via a backdoor, e.g., by 'paraphrasing' documents from DCLM-POOL into memorized data" (Appendix C.2).

**Mixing track.** Participants may combine documents from DCLM-POOL with any external sources, provided those sources are freely available and do not include evaluation data. Submissions must document their data sources, the weight given to each source, and the ratio of tokens used for training to the overall custom pool size. This track is designed to accommodate strategies that mix domain-specific data (code, math, books) with web crawl—the approach used by most state-of-the-art models including RedPajama, Dolma, and DCLM-BASELINE itself at the trillion-token scale.

---

#### Training Recipe: Isolating Data as the Independent Variable

The training recipe is fixed at each scale to eliminate architectural and hyperparameter choices as confounding variables. All models use a **decoder-only Transformer** architecture implemented in OpenLM (Gururangan et al., 2023), with specifications that follow GPT-2 (Radford et al., 2019) and Llama (Touvron et al., 2023) conventions.

**Architecture specifications (Table 10).** Each scale specifies the number of layers (`nlayers`), number of attention heads (`nheads`), model width (`dmodel`), and per-head width (`dhead`):

- **400M:** 24 layers, 8 heads, dmodel = 1024, dhead = 128
- **1B:** 24 layers, 16 heads, dmodel = 2048, dhead = 128
- **3B:** 32 layers, 32 heads, dmodel = 2560, dhead = 128
- **7B:** 32 layers, 32 heads, dmodel = 4096, dhead = 128

All models use LayerNorm without bias parameters, qk-LayerNorm on queries and keys for training stability, SwiGLU MLPs (Shazeer, 2020), and depth-scaled initialization (Zhang et al., 2019). The sequence length is 2048 tokens for all pretraining runs, with multiple sequences packed into batches to fill the entire context using an EOS token to separate documents. Causal attention is allowed to attend across documents—experiments with masking attention across document boundaries showed "little impact on downstream performance" (Appendix F).

**Optimization hyperparameters (Table 10).** The key choices, which vary by scale:

- **400M-1x and 1B-1x:** warmup 2,000 or 5,000 steps, learning rate 3e-3, weight decay 0.033, z-loss coefficient 1e-4, batch size 512 or 256 sequences
- **3B-1x:** warmup 5,000 steps, learning rate 3e-3, weight decay 0.033, z-loss 1e-4, batch size 256
- **7B-1x and 7B-2x:** warmup 5,000 steps, learning rate 2e-3, weight decay 0.05, z-loss 5e-6, batch size 2,048 sequences

The 7B hyperparameters were selected based on a sweep over learning rate and weight decay (Table 11), which showed that a 2e-3 learning rate with 0.05 weight decay achieved the best CORE score (44.8) compared to alternatives (44.1 for 1e-3/0.1, 44.7 for 3e-3/0.033, 43.8 for 1e-2/0.01). The optimization uses Adam (Kingma & Ba, 2015) with z-loss (Chowdhery et al., 2022) to encourage output logit magnitudes to remain in a numerically stable range. A final learning rate cooldown of 3e-5 is applied for all experiments.

**Hyperparameter independence validation (Appendix H).** A concern when fixing hyperparameters is that the optimal training recipe might interact with dataset quality—a better dataset might benefit more from different hyperparameters. Table 12 shows that across five hyperparameter settings varying learning rate and weight decay, the ranking of three datasets (DCLM-BASELINE > RedPajama > C4) is preserved, with DCLM-BASELINE consistently outperforming the others by 3.3–4.5 CORE points. Table 13 further demonstrates that dataset improvements and hyperparameter improvements appear to be orthogonal—switching from low-LR to high-LR training improves CORE by 1.4 points on the base dataset and by 3.1 points on the fastText-filtered dataset, so the two interventions stack rather than substitute. This validates the benchmark's design: participants can optimize their data curation independently of the training recipe and expect consistent gains.

**Architecture independence validation (Appendix I).** Figure 6 shows high correlation between dataset performance on the standard OpenLM architecture and on two alternative architectures: a Gemma-inspired variant (changing activation to GeGLU, adding RMS normalization) and a Mamba state-space model (Gu & Dao, 2023). This suggests that dataset improvements generalize across fundamentally different model architectures, not just hyperparameter variations.

---

#### Evaluation Metrics: Centered Accuracy and Task Selection

The evaluation suite contains 53 downstream tasks (fully enumerated in Appendix G.1) spanning question answering, commonsense reasoning, reading comprehension, mathematical problem solving, and factual knowledge. All evaluations are performed on base models without instruction tuning or task-specific fine-tuning—the model must answer questions using few-shot prompting (typically 3-shot, 5-shot, or 10-shot, depending on the task) or zero-shot evaluation.

**Centered accuracy.** The primary metric for both CORE and EXTENDED is centered accuracy, defined per task as a linear rescaling:

$$\text{centered\_accuracy} = \frac{\text{accuracy} - \text{random\_baseline}}{1 - \text{random\_baseline}}$$

where `$\text{accuracy}$` is the model's raw accuracy on the task (fraction of examples answered correctly) and `$\text{random\_baseline}$` is the expected accuracy of random guessing (e.g., 0.25 for a 4-way multiple-choice task, 0.50 for a binary task). The denominator `$1 - \text{random\_baseline}$` normalizes so the maximum achievable score is 1.0.

**What it computes:** for each task, it measures how far the model's accuracy moves from random guessing toward perfect performance, expressed as a fraction of the maximum possible improvement. A model achieving 60% accuracy on a 4-way MCQ task (random baseline 25%) would receive a centered accuracy of (0.60 − 0.25)/(1 − 0.25) = 0.35/0.75 ≈ 0.467, meaning it captures about 46.7% of the possible improvement over chance.

**Why this form:** Raw accuracy confounds task difficulty with model capability—an accuracy of 0.55 on a binary task is far less impressive than 0.55 on a 5-way task, but raw accuracy treats them identically. Centered accuracy normalizes for the number of answer options, making scores comparable across tasks with different formats. This is particularly important because the average is taken across 22 or 53 tasks with varying numbers of choices—without normalization, tasks with fewer options would dominate the aggregate. In the paper's own words, the rescaling ensures "0 corresponds to random guessing and 1 corresponds to perfect accuracy" (Section 3.5).

**CORE vs. EXTENDED.** The CORE metric averages centered accuracy over 22 tasks selected because they "provide a low-variance signal even at small scales" (Section 3.5). These include HellaSwag (0-shot and 10-shot, 10,042 examples), ARC-Easy (2,376 examples) and ARC-Challenge (1,172 examples), BoolQ (3,270 examples), PIQA (1,838 examples), and several Big-Bench tasks (QA Wikidata with 20,321 examples, CS Algorithms with 1,320 examples, Language Identification with 10,000 examples). The large sample sizes of these tasks give them lower variance, making them suitable for detecting small improvements from data interventions at the 400M and 1B scales where model performance is noisy.

The EXTENDED metric averages centered accuracy over all 53 tasks, including lower-sample-size tasks like GPQA-diamond (198 examples), Winogender (60 examples), and MMLU (14,042 examples across 57 subtasks, evaluated both 0-shot and 5-shot). MMLU is also reported separately as a single metric given its widespread use as a benchmark for comparing state-of-the-art models.

**MMLU evaluation detail.** For MMLU, the paper uses the LLM Foundry evaluation framework, which scores multiple-choice questions by considering the log probability of single-letter answers (A, B, C, D). Appendix G.2 compares this to the LightEval framework (used by FineWeb-Edu), which evaluates log probabilities of entire answer sequences. The paper finds that LightEval provides signal above the random baseline for smaller (1B) models where LLM Foundry's single-letter approach yields near-random performance, but at larger scales "the LightEval scores for the models become quite cramped together" (Figure 5: Gemma-7B, Llama3-8B, and Mistral-7B all score 0.43–0.44 in LightEval while ranging from 0.56 to 0.62 in LLM Foundry). This is interpreted as LightEval being potentially better for small-scale comparisons but less discriminative at the 7B scale.

**The 53-task suite (Appendix G.1).** Beyond the CORE subset, the EXTENDED tasks include:
- Legal and graduate-level reasoning: AGI Eval LSAT-LR (510 examples), LSAT-RC (268 examples), GPQA-main (448 examples) and GPQA-diamond (198 examples)
- Mathematics: GSM8K (1,319 examples, 3-shot with chain-of-thought), SVAMP (300 examples, chain-of-thought), AQuA (245 examples), MathQA (2,983 examples), SAT-Math (220 examples)
- Bias and safety: BBQ (55,006 examples), Winogender male/female (60 examples each), Enterprise PII classification (3,395 examples)
- Additional Big-Bench tasks: Logical Deduction (1,500 examples), StrategyQA (2,289 examples), Elementary Math QA (34,313 examples), Strange Stories (174 examples), and others

The full list with licenses appears in Appendix T.1.

---

#### The Data Curation Pipeline: Producing DCLM-BASELINE

The pipeline that produces DCLM-BASELINE from DCLM-POOL is visualized in Figure 4 as a funnel, with each stage removing a fraction of documents. The stages, applied sequentially, are:

1.  **Text extraction** (resiliparse, described above) — starting point for all documents.
2.  **Heuristic cleaning** — reproduction of RefinedWeb's filtering rules, including English language detection (fastText classifier identifying 157 languages, keeping only English), URL-based filtering (domain banlists from UT1 and LDNOOBW), page length filtering (removing documents that are too short or too long), word removal ratio filtering (removing documents where too high a fraction of words have been stripped by cleaning), and repetition filtering (removing documents with excessive repeated n-grams).
3.  **Deduplication** — Bloom filter-based near-deduplication at both the paragraph and document level (detailed below), removing approximately 20% of remaining tokens.
4.  **Model-based quality filtering** — fastText binary classifier trained on instruction-formatted positive data (OpenHermes 2.5 + ELI5 subreddit) against randomly sampled negative data from the pre-filtered pool; keeping the top 10% of documents by classifier score.

The document counts removed at each stage are shown in Figure 4 as percentages of the original 200B documents: heuristic cleaning removes 50.8% (primarily through the fastText language filter and URL filtering), deduplication removes an additional 19.9%, and model-based filtering removes 13.7%, with other minor filters (word length, ellipsis count, stop word analysis) removing 9.0%. The final DCLM-BASELINE keeps approximately 3B documents from the original 200B—about 1.5%.

---

#### Heuristic Filtering: The RefinedWeb Reproduction

The baseline heuristic filtering pipeline is a reproduction of Penedo et al. (2023)'s RefinedWeb, which operates on the extracted text to remove clearly undesirable content.

**English language detection.** A fastText classifier (Joulin et al., 2017) trained to identify 157 languages (Grave et al., 2018) is applied to each document. Documents classified as non-English with high confidence are removed. This is the single largest filtering step, removing approximately 50% of documents from the multilingual Common Crawl.

**URL-based filtering.** Two blocklists are applied: the UT1 Blacklist (a community-maintained list of domains associated with adult content, malware, or spam) and the LDNOOBW list (a curated list of banned URL substrings). Documents whose URLs match entries in either list are removed.

**Content quality heuristics.** Several rule-based filters target common artifacts of web text:
- **Page length:** documents with fewer than a minimum character count or more than a maximum are removed, eliminating empty pages and extremely long data dumps.
- **Word removal ratio:** measures the fraction of tokens that were removed by text normalization (e.g., punctuation stripping); documents where this ratio exceeds a threshold (indicating the page consists primarily of non-textual content like markup or code) are discarded.
- **Repetition filter:** identifies documents where the same n-gram sequences repeat excessively, characteristic of boilerplate, template text, or machine-generated content.
- **Additional filters:** word-length-based filters (removing documents with unusual distributions of word lengths), ellipsis count filters (excessive ellipsis suggesting garbled text), and stop-word fraction analysis.

The exact thresholds for these heuristics are not specified numerically in the paper but are implemented in the open-source DCLM tooling as mappers that can be composed into processing pipelines.

---

#### Deduplication: Bloom Filter-Based Near-Deduplication

The deduplication pipeline is one of the more technically sophisticated components of the system, and the paper provides extensive ablations in Appendix L to justify the chosen approach.

**Why not MinHash + Suffix Array?** Prior work (Lee et al., 2022; Penedo et al., 2023) uses a two-stage pipeline: MinHash for fuzzy document-level deduplication (identifying pairs of documents with high Jaccard similarity in their n-gram sets) followed by suffix array-based substring deduplication (identifying and removing any substring longer than 50 tokens that appears more than once in the corpus). The paper finds this pipeline effective but expensive—the suffix array step requires loading the entire corpus into RAM on a single node. For a 70TB input, this is impractical.

**Bloom filter method (BFF).** The authors adapt the Big Friendly Filter (BFF) from Soldaini et al. (2024), extending it to perform both paragraph-level and document-level near-deduplication simultaneously using Bloom filters (Bloom, 1970). A Bloom filter is a space-efficient probabilistic data structure supporting set membership queries with no false negatives and a controllable false positive rate.

The algorithm proceeds as follows, for each document in sequence:

1.  **Tokenize and split into paragraphs.** The document is tokenized using the UniSeg tokenizer (Unicode Consortium, 2023) and split on newline characters into paragraphs.
2.  **For each paragraph, extract n-grams.** The parameters `min_ngram_size = 13` and `max_ngram_size = 13` are used—meaning each paragraph is treated as a single 13-gram (or handled differently if shorter or longer, as discussed below).
3.  **Check Bloom filter membership.** For each n-gram, the algorithm checks whether it is already present in the Bloom filter. A counter `contained_ngrams` is incremented for each n-gram found, and `total_ngrams` is incremented for each n-gram extracted.
4.  **Remove duplicate paragraphs.** If the fraction `contained_ngrams / total_ngrams` for a paragraph exceeds the threshold `T = 0.80`, the entire paragraph is removed from the document. Otherwise, all non-contained n-grams from the paragraph are inserted into the Bloom filter.
5.  **Remove duplicate documents.** After processing all paragraphs, if the document-level fraction `contained_ngrams / total_ngrams` exceeds 0.80, the entire document is removed.

For paragraphs shorter than `min_ngram_size = 13` tokens, they are left untouched. For paragraphs longer than `max_ngram_size = 13`, each sliding window of 13 tokens is treated as a separate n-gram and checked independently.

**Bloom filter sizing.** The optimal number of hash functions `$k$` is given by:

$$k = -\frac{\ln \epsilon}{\ln 2}$$

where `$\epsilon$` is the desired false positive rate.

The optimal size `$m$` in bits for `$k$` hash functions and `$n$` expected inserted elements is found by solving:

$$\epsilon = \left(1 - e^{-\frac{kn}{m}}\right)^k$$

This is solved numerically via binary search since no closed-form solution exists.

**What it computes:** given an estimate of the total number of distinct n-grams in the corpus and a target false positive rate, the sizing equations determine the memory footprint needed such that, on average, only fraction `$\epsilon$` of queries for truly unseen n-grams will incorrectly return "present."

**Why this form:** Bloom filters guarantee zero false negatives (an n-gram that was inserted will always be recognized), which is critical for deduplication—a false negative would mean failing to detect a duplicate, undermining the purpose. The false positive rate control allows trading memory against accuracy. The paper argues analytically that a false positive rate of `$\epsilon = 0.01$` is sufficient because removal decisions are based on *aggregate* statistics over many n-grams. For a paragraph with `$N$` n-grams of which `$S$` are truly present, at least `$TN - S$` of the remaining `$N - S$` would need to be false positives for incorrect removal, each occurring independently with probability `$\epsilon$`. The probability of this event is bounded by the Hoeffding inequality:

$$\exp\left(-2 \cdot \frac{(TN - S - \epsilon \cdot (N - S))^2}{N - S}\right)$$

For a document with 100 n-grams, `$T = 0.8$`, `$\epsilon = 0.01$`, and `$S = 60$`, this probability is less than `$10^{-8}$`—effectively zero. This analysis justifies the counterintuitive choice of a relatively high false positive rate.

**Sharding.** For scalability, the deduplication is run on shards (disjoint subsets of the corpus processed independently). DCLM-BASELINE uses 100-way sharding, where the ~70TB input is split into 100 shards of ~700GB each, and BFF runs independently on each. This reduces RAM requirements (each shard's Bloom filter must fit in memory) and enables parallelization. However, sharding means duplicates spanning shard boundaries are not detected—a document present in shard 3 will not be compared against shard 7. The paper acknowledges this limitation but shows empirically in Table 19 that 10- or 100-way sharded BFF achieves comparable downstream performance to global MinHash+SuffixArray (44.3 vs. 44.4 MMLU at 7B-2x for 10-shard BFF vs. global MinHash+SA), suggesting that cross-shard duplicates are not performance-critical.

**Ablation summary (Appendix L.2).** The key findings from deduplication ablations at the 1B-1x scale (Table 17) are: (i) BFF alone removes 26% of tokens and improves CORE by 2.1 points over the unfiltered baseline, comparable to a full Exact+MinHash+SuffixArray pipeline (41% removal, +2.1 CORE); (ii) MinHash alone improves CORE by 0.9 points, SuffixArray alone by 1.9 points, suggesting substring-level duplication is more harmful than document-level fuzzy duplication; (iii) combining methods does not necessarily help—adding Exact deduplication to MinHash actually reduces CORE from 25.6 to 25.0. At the 7B scale (Table 18), BFF with min_ngram_size = 13 slightly outperforms MinHash+SA on CORE (40.5 vs. 40.8, within noise) while yielding more tokens (4.2T vs. 3.2T).

---

#### Model-Based Quality Filtering: The fastText Classifier

The most impactful single curation step is training a classifier to distinguish "high-quality" documents (suitable for pretraining) from "low-quality" documents (unsuitable) and retaining only the top-scoring fraction. This is described in Section 4.4 and detailed in Appendix J.

**Classification approach: fastText.** The classifier is a fastText linear model (Joulin et al., 2017) that operates on bag-of-words or bag-of-ngram features. Unlike neural classifiers operating on dense embeddings, fastText models are extremely efficient to train and apply—they can process billions of documents on CPU-only hardware—making them practical for web-scale filtering.

**Training data construction.** The classifier is trained on a balanced set of approximately 400K examples: 200K positive examples (documents from high-quality sources) and 200K negative examples (documents sampled randomly from a pre-filtered, deduplicated version of the Common Crawl pool—specifically, an earlier RefinedWeb reproduction that used trafilatura extraction).

The crucial design choice is the **positive reference data**. Table 5 shows dramatic variation depending on what is used as the positive class:

| Positive data source | Threshold | CORE | MMLU | EXTENDED |
|---|---|---|---|---|
| OH-2.5 + ELI5 | 10% | 41.0 | 29.2 | 21.4 |
| Wikipedia | 10% | 35.7 | 27.0 | 19.1 |
| OpenWebText2 | 10% | 34.7 | 25.0 | 18.7 |
| GPT-3 Approx | 10% | 37.5 | 24.4 | 20.0 |

The best-performing choice—**OpenHermes 2.5 (OH-2.5) + ELI5**—is unconventional. OpenHermes 2.5 is a synthetic dataset of instruction-formatted dialogues generated by prompting GPT-4. ELI5 (r/ExplainLikeImFive) is a subreddit where users ask questions and receive simplified explanations, with community voting determining answer quality. The authors curate ELI5 examples by taking each post and combining it with the top-scoring answer (by karma), filtering to keep only posts with score ≥ 0, best comments with score ≥ 5, and threads with at least 3 comments total. They sample 100K examples from each source.

**Why does this work better than Wikipedia?** The paper does not fully explain the mechanism, but the hypothesis implicit in the results is that instruction-formatted data and question-answer pairs provide a signal about *usefulness for answering diverse queries* that encyclopedic text does not. Wikipedia represents well-written, factual prose—but language model pretraining benefits from data that teaches models to follow instructions, reason through answers, and handle varied formats. By training the classifier to recognize documents that resemble High-quality instruction data and community-curated Q&A, the filter selects web pages that are more likely to teach the model skills it needs for downstream tasks like MMLU, which consist of question-answering and reasoning.

**Feature space.** Table 14 shows that expanding the fastText feature space from unigrams only to unigrams + bigrams (via setting `wordNgrams = 2`) improves CORE from 40.0 to 41.0—a small but consistent gain. Bigrams capture two-word phrases and simple collocations, giving the classifier some sensitivity to word order without the complexity of full sequence models.

**Filtering threshold.** The classifier assigns each document a score between 0 and 1 (the predicted probability of the positive class). Documents are then sorted by score, and only those above a chosen percentile threshold are retained. Table 5 shows that the threshold matters: top-10% yields 41.0 CORE, top-15% yields 39.8, top-20% yields 38.7. Stricter filtering (keeping a smaller fraction) improves average document quality but reduces dataset size. The 10% threshold was chosen as the best tradeoff.

**The classifier training recipe.** The paper uses the default fastText training parameters except for the `wordNgrams` change to 2. The fastText algorithm itself optimizes a linear classifier with a hierarchical softmax or negative sampling objective over n-gram features. Documents are represented as the average of their n-gram embeddings (which are learned during training), and a linear decision boundary is fit to separate positive from negative examples.

**Comparison to other quality filters (Table 4).** The paper evaluates several alternatives at the 1B-1x scale, all less effective than fastText:

- **PageRank filtering** (Table 15): using Common Crawl's host-level webgraph to compute PageRank centrality for each domain, then filtering by PageRank quintile. No quintile outperforms random sampling—the idea that "more central" pages are higher quality is not supported.
- **Semantic deduplication (SemDedup):** embedding documents with BGE-base, clustering via k-means (K=11,000), and removing 25% of data points within each cluster. This *decreases* CORE from 27.5 to 27.1—the authors hypothesize the embedding model's biases may be responsible.
- **AskLLM:** prompting Mistral-7B-Instruct-v0.2 to evaluate whether a document is suitable for LLM pretraining, using the cumulative probability of "Yes" and "yes" tokens as a score. This achieves 28.6 CORE vs. 30.2 for fastText, at vastly higher computational cost.
- **Perplexity filtering:** training a 154M-parameter causal Transformer on Wikipedia, books, and peS2o, then using its perplexity on each document as a quality score (low perplexity = more "in-distribution" = higher quality). Achieves 29.0 CORE.
- **Top-k average logits:** averaging the top-k model logits over all tokens in a document to measure model confidence. Achieves 29.2 CORE.

The fastText classifier dominates despite (or perhaps because of) its simplicity—it uses only surface-level n-gram statistics rather than semantic understanding, suggesting that "high-quality" documents for pretraining are distinguished more by their lexical patterns and formatting conventions than by their semantic content.

---

#### The Human Judgment Paradox (Appendix N)

A striking finding that runs counter to conventional wisdom: **human annotators cannot reliably identify which documents are good for pretraining.** Sixteen AI graduate students and professors annotated approximately 500 randomly selected documents as "good" or "bad" for LM pretraining (three annotators per document). The average inter-annotator agreement was only 71%, and in only 281 of 499 cases did all three annotators agree.

Figure 9 plots various quality filters' agreement with human labels (ROC-AUC) against the CORE score of models trained on datasets filtered by those same classifiers. There is **zero correlation**—the filter that agrees most with humans (AskLLM, 82% ROC-AUC) produces the worst model (28.5 CORE), while the filter that works best (fastText OH-2.5 + ELI5, 73% ROC-AUC) is only modestly correlated with human judgment.

The paper hypothesizes that "human intuition may not reliably identify the most useful documents for language model training purposes" and that "human curators may create datasets that lack sufficient diversity." This has profound implications: if even expert humans cannot judge training data quality, then the only reliable signal is the downstream performance of models trained on that data—exactly the closed-loop benchmarking that DCLM enables.

---

#### Scaling to the Trillion-Token Scale (Section 5)

The final model that achieves state-of-the-art results (64% MMLU at 7B, 2.6T tokens) uses a more complex recipe than the baseline pipeline:

**Dataset composition.** DCLM-BASELINE (3.8T tokens of filtered Common Crawl) is combined with StarCoder (Li et al., 2023, code data) and ProofPile2 (Azerbayev et al., 2023, mathematical data) to create a 4.1T token dataset. This mixture acknowledges that the baseline filtering pipeline is optimized for language understanding tasks (which dominate the CORE and EXTENDED evaluations) and does not adequately cover code and math—domains that require specialized data sources.

**Training procedure.** The model is trained for 2T tokens on this mixture using the 7B-2x hyperparameters (Table 27: learning rate 2e-3, weight decay 0.05, z-loss coefficient 5e-6, global batch size 2048). After 2T tokens, when the learning rate has decayed to approximately 1e-3, two separate cooldown phases are run:

1.  **Cooldown 1:** 270B additional tokens on a modified distribution that is 70% DCLM-BASELINE with a tighter fastText threshold (top 7% rather than top 10%) and 30% ProofPile2.
2.  **Cooldown 2:** 200B additional tokens on the same modified distribution.

The cooldown phases continue decaying the learning rate from its current value to the final learning rate of 3e-5 over the specified number of tokens, concentrating the remaining learning on the highest-quality subset of the data.

**Model souping.** The weights from Cooldown 1 and Cooldown 2 are averaged with weights 0.8 and 0.2 respectively, following the model soup technique (Wortsman et al., 2022). Table 28 shows that the souped model achieves 63.9 MMLU (vs. 62.7 and 63.4 for the individual cooldowns), 56.0 CORE, and 43.7 EXTENDED.

**Context length extension (Appendix Q.2).** The souped model is further trained for 100B tokens using a variable sequence length curriculum (Pouransari et al., 2024) that gradually increases the context from 2048 to 8192 tokens using the Grow-Linear schedule with 4 cycles. The RoPE base frequency is increased from 10,000 to 100,000 during this stage. The resulting model (DCLM-8k) maintains performance on regular evaluations (57.1 CORE, 63.7 MMLU, 45.4 EXTENDED) while gaining the ability to process long contexts—Table 30 shows it achieves 46.1% and 38.8% on 20-document and 30-document multi-document QA, tasks that are impossible for the original 2048-context model.

**Instruction tuning (Appendix P).** To demonstrate that DCLM-BASELINE models can be adapted for interactive use, the authors instruction-tune the base model on two datasets: (1) OpenHermes 2.5 alone, achieving 13.8% AlpacaEval 2.0 LC win rate; (2) a custom 4M-example, 8B-token mixture (DCLM-IT) combining UltraFeedback, Tulu-v2 SFT, CodeFeedback, OpenHermes 2.5, Nectar, NoRobots, WildChat, WebInstruct, and StarCoder2-Self-OSS-Instruct, achieving 16.6% win rate. This outperforms Gemma-Instruct-7B (10.4%) and approaches Mistral-v0.2-7B (17.1%), demonstrating that the pretraining data quality confers benefits that persist through instruction tuning.

## 4. Key Insights and Innovations

### Innovation 1: A Standardized Benchmark Solves the Confounding Problem in Data-Centric LM Research

The field of data curation for language models has suffered from a fundamental methodological problem: every paper evaluating a new filtering or deduplication technique trains a model with a different architecture, different hyperparameters, and a different compute budget, then declares victory when their model outperforms some hand-picked baseline. The paper's core intellectual move is recognizing that **this confounding is not incidental — it is the central barrier to progress**. When Paper A's perplexity-filtered model beats Paper B's deduplicated model, the field learns nothing because the comparison conflates data quality with every other design choice.

Prior to DCLM, the dominant assumption in the field was that rigorous comparison required matching the state-of-the-art on whatever metrics and training setups were currently fashionable. Researchers trained models following the latest recipe from Llama or Chinchilla, evaluated on MMLU, and hoped the community would correctly attribute performance differences to their data intervention. The DataComp vision benchmark (Gadre et al., 2024) pioneered the alternative — fix the training recipe and let participants compete on data curation — but had not been adapted to language modeling at scale.

DCLM's contribution is not the *idea* of standardized evaluation, which is old in machine learning, but rather its **demonstration that such standardization is both feasible and necessary at the scale of modern LM pretraining**. The paper proves this through a specific diagnostic: Table 2 shows that RefinedWeb, RedPajama, Dolma-V1, and C4 — the most widely-used open pretraining datasets — produce significantly different models (36.9 vs. 34.2 CORE) **even when trained with identical recipes**. This means the field has been making apples-to-oranges comparisons not just across papers, but across the very datasets it treats as interchangeable. The fact that RefinedWeb outperforms datasets that mix in "high-quality" sources like Wikipedia (RedPajama, Dolma-V1) is a finding that could only have emerged from controlled comparison — and it overturns the intuition that adding curated sources necessarily helps.

The scalability validation (Figure 3) is equally significant as a conceptual contribution. By showing that dataset rankings at 400M parameters predict rankings at 7B parameters with Pearson's r = 0.982, the paper establishes that **data curation research does not require frontier-scale compute**. This is a practical enabler: it means academic labs and individual researchers can participate meaningfully, which in turn means the field can accumulate far more experimental evidence than would be possible if every experiment required a 7B-scale training run. This is not merely an engineering convenience — it changes the economics of who can contribute to data curation research, potentially democratizing a field that has been dominated by industrial labs with massive compute budgets.

### Innovation 2: Model-Based Quality Filtering Outperforms Human Judgment — and the Gap Is Fundamental, Not Incidental

Perhaps the most counterintuitive finding in the paper is that **human annotators — including AI researchers — cannot identify which documents are good for language model pretraining**. Appendix N demonstrates this with a clean experiment: 16 annotators label 499 documents as "good" or "bad" for pretraining, and the filter that best agrees with their labels (AskLLM, 82% ROC-AUC) produces the *worst* training set (28.5 CORE at 1B-1x), while the best-performing filter (fastText OH-2.5 + ELI5, achieving 31%+ CORE) shows only modest agreement (73% ROC-AUC). There is no correlation between human-judgment-alignment and downstream model quality (Figure 9).

This is not a minor empirical quirk — it challenges a foundational assumption of the data quality literature. The dominant paradigm, exemplified by approaches like training perplexity-based filters on Wikipedia or using human-written heuristics to identify "clean" text, implicitly assumes that human notions of quality (well-formedness, informativeness, grammaticality) align with what makes data useful for pretraining. The paper shows this assumption is **fundamentally wrong**. Human annotators likely overvalue documents that are well-structured and informative in an encyclopedic sense, while undervaluing the messy, diverse, and stylistically varied text that teaches models to generalize across the distribution of real-world language tasks.

This finding is a significant conceptual advance because it **reframes what "data quality" means for language model pretraining**. Quality is not an intrinsic property of a document that a human can assess by reading it — it is a functional property that can only be measured by the downstream performance of models trained on it. This is analogous to the shift in computer vision where researchers discovered that ImageNet accuracy does not perfectly correlate with human judgments of image "difficulty" or "quality." The implication is that any data curation pipeline that relies on human intuition about what data should look like (whether through hand-crafted heuristics, human annotation, or training classifiers on human-selected reference corpora like Wikipedia) is operating with the wrong objective function.

The practical consequence — that **instruction-formatted synthetic data (OpenHermes 2.5) and community-curated Q&A (ELI5) make better reference data for training quality filters than Wikipedia or books** — is surprising and important. It suggests that what makes pretraining data useful is not factual accuracy or prose quality per se, but rather diversity of format, presence of question-answer structure, and coverage of the kinds of reasoning patterns that downstream tasks require. This is a concrete, actionable insight that changes which data sources future researchers should use to train filtering models.

### Innovation 3: Data Curation Is a Pipeline with Non-Independent Stages — and Mixing Can Hurt

A less flashy but equally important conceptual contribution is the finding that **data curation steps interact in non-obvious ways, and the optimal strategy at one stage depends on what was done at previous stages**. This challenge to the modularity assumption — that each curation step can be optimized independently and the results will compose — emerges from the mixing experiments in Section 4.5.

The key result is Table 6: adding high-quality external sources (Wikipedia, books, StackExchange, arXiv, GitHub) to Common Crawl subsets *improves* performance when the CC subset is poorly filtered (C4 gains +2.2 CORE, RefinedWeb gains +1.4) but *degrades* performance when the CC subset is already well-filtered (DCLM-BASELINE loses −1.2 CORE). This is a reversal of the common wisdom — exemplified by GPT-3, Llama, and nearly every major model — that mixing in curated sources is always beneficial. The paper's explanation, that aggressive filtering of Common Crawl produces text that is already higher quality than the "high-quality" sources being mixed in, is provocative. It implies that the field's obsession with adding Wikipedia and books to web-crawled training sets may be a band-aid for insufficient filtering of the web data, not a fundamental requirement.

This finding has significant methodological implications for how the field should approach data curation research. If pipeline stages interact, then evaluating a new deduplication method on an unfiltered corpus tells you little about whether it will help when applied after a quality filter. The DCLM framework makes it possible to study these interactions systematically — each stage can be toggled on or off while holding everything else constant — but the conceptual message is that **future work should report not just "how much does intervention X help" but "how much does intervention X help given that interventions Y and Z have already been applied."**

### Innovation 4: Verifier Over-Optimization by Analogy — Data Curation Has Its Own "Reward Hacking"

While the paper does not use this term, one of its most important conceptual contributions is the empirical demonstration of a phenomenon that parallels reward hacking in RLHF: **a data filtering model that is too aligned with human notions of quality can produce datasets that optimize the wrong objective**. This is most clearly seen in the comparison between fastText classifiers trained on different reference data (Table 5).

The Wikipedia-trained classifier and the OpenWebText2-trained classifier both represent reasonable, intuitive choices for what "good" text looks like — Wikipedia is well-written and factual, OpenWebText2 is curated from Reddit upvotes. Both produce datasets that humans would likely rate as high-quality. But both produce substantially worse models than the OH-2.5 + ELI5 classifier (35.7 and 34.7 CORE vs. 41.0 at 7B-1x). The gap is large enough (5+ points on a 100-point scale at the 7B scale) to be practically meaningful — it represents the difference between a dataset that produces a competitive model and one that does not.

What makes this analogous to reward hacking is the mechanism: the classifier is optimizing for a proxy (similarity to reference data) that correlates imperfectly with the true objective (downstream model performance). When the proxy is too narrow — as with Wikipedia, which represents a specific genre of formal, expository prose — the filtered dataset becomes homogeneous in ways that harm generalization even as it becomes more "high quality" by the proxy metric. The OH-2.5 + ELI5 classifier works better precisely because its reference data is more heterogeneous (instruction dialogues, community Q&A) and thus selects for diversity rather than a single notion of quality.

This framing has practical bite: it suggests that **the key to effective data filtering is not finding the single best reference corpus but rather constructing a reference that spans the diversity of behaviors you want the model to learn**. It also explains why human judgment fails — humans have narrow, genre-specific notions of quality that produce the same homogenization effect as a Wikipedia-trained classifier. The paper's contribution here is not a technical solution to this problem but rather a **diagnostic concept** that reframes data curation from "remove the bad stuff" to "select for useful diversity," which is a fundamentally different objective.

### Innovation 5: Data Quality Can Substitute for Compute, with Quantified Exchange Rates

The paper's headline result — that a 7B model trained on DCLM-BASELINE with 2.6T tokens achieves 64% MMLU, comparable to Llama 3 8B (66%) while using 6.6× less compute — is not just a benchmark win. It constitutes an **empirical scaling law for data quality vs. compute**, albeit one specific to the DCLM pipeline and the Llama-3 comparison point.

Prior work has suggested qualitatively that data quality matters — the Phi models demonstrated that textbook-quality synthetic data could produce strong small models, and Chinchilla showed that training on more tokens (of whatever quality) improves performance. But the field lacked a **quantitative exchange rate**: how many FLOPs of pretraining compute can be saved by improving data quality by a specified amount? The paper's comparison provides a specific number, at least for this particular setting: a 6.6× compute reduction at parity MMLU performance.

The significance of this finding extends beyond the specific number. It implies that **the allocation of resources between data curation and model scaling is itself an optimization problem** that the field has not been solving systematically. If a well-filtered dataset enables training a model with 6.6× less compute to reach the same performance, then spending 1 GPU-year on data filtering experiments (which is cheap — fastText can be trained on CPUs) can save hundreds of GPU-years of pretraining. This recasts data curation from a preprocessing step done once and forgotten into a **first-class investment with enormous leverage**.

The paper does not provide a general scaling law for this tradeoff — the 6.6× figure is benchmarking against Llama 3 specifically, and the exchange rate likely varies with model scale, data volume, and task distribution. But by establishing that the exchange rate can be quantified and that it can be large enough to matter, the paper opens a research direction that parallels the Chinchilla scaling laws: **what is the compute-optimal allocation between data curation effort and model training effort?** Answering this question requires exactly the kind of standardized, multi-scale experimental framework that DCLM provides, making the benchmark not just a tool for comparing datasets but an instrument for discovering laws of data-centric ML.

## 5. Experimental Analysis

### Evaluation Methodology

- **Dataset.** All experiments use the MATH benchmark (Hendrycks et al., 2021), specifically the split from Lightman et al. (2022): 12,000 training questions and 500 test questions. The authors choose MATH because test-time compute is expected to help most when the model already possesses the necessary knowledge and the challenge is drawing complex inferences — mathematical reasoning fits this profile since it requires multi-step logical deduction rather than novel factual recall (Section 4).

- **Base model(s).** All experiments use PaLM 2-S* (Codey) (Anil et al., 2023). The authors argue this model is "representative of the capabilities of many contemporary LLMs" and sits in a useful regime: non-trivial performance on MATH (roughly 10–19% pass@1 depending on the prompt and sampling configuration) but far from saturation, leaving room for test-time compute to make a difference (Section 4). For the FLOPs-matched comparison, a second model with approximately 14× more parameters is used as the pretraining-scaled baseline.

- **Metrics.** The primary metric throughout is MATH test accuracy (%) — the fraction of the 500 test questions for which the selected final answer matches the ground truth. Answers are graded using the grading function released by Lightman et al. (2022) (Appendix G). When analyzing difficulty-dependent behavior, the paper reports accuracy within each of five difficulty quintiles separately.

- **Baselines.** The paper uses several baselines: Majority voting (select the most common final answer among N sampled solutions with no learned verifier); ORM best-of-N weighted (score N solutions with an outcome reward model and apply best-of-N weighted selection); PRM best-of-N weighted (score N solutions with the process reward model and apply best-of-N weighted selection); and parallel sampling for revisions (generate N independent solutions from the revision model and select the best via verifier or majority voting).

- **Generation budget / compute accounting.** One "generation" equals one complete sampled answer from the base LLM. For beam search and best-of-N, the budget equals the number of beams or samples N. For lookahead search with k lookahead steps, the cost is N × (k + 1) to account for the additional rollout computation (Section 5.3). Budgets are swept across powers of 2, typically from 2⁰ to 2⁹ (1 to 512 generations).

- **Cross-validation / statistical protocol.** To avoid contaminating strategy selection with test-set performance, the authors use two-fold cross-validation within each difficulty bin on the 500-question test set. The best strategy is selected on one fold and evaluated on the other, with results averaged (Section 3.2). This applies specifically to the compute-optimal strategy selection, where the best-performing strategy per bin is determined on held-out data and then evaluated.

---

### Main Quantitative Results

#### Search Against PRM Verifiers (Section 5)

**Aggregate search algorithm comparison (Figure 3, left).** Across all 500 test questions with a maximum budget of 256 generations:

- At low budgets (2–8 generations), beam search with M = 4 significantly outperforms best-of-N weighted. For example, at 4 generations, beam search (M = 4) achieves roughly 27% accuracy versus roughly 16% for best-of-N weighted — a substantial gap.
- At high budgets (64–256), beam search performance flattens and falls slightly below best-of-N weighted. Best-of-N weighted reaches approximately 38% at 512 generations; beam search (M = 4) plateaus around 34%.
- Lookahead search (both k = 1 and k = 3) generally underperforms at the same generation budget due to its higher per-step cost. The 3-step lookahead variants converge to similar performance as other methods at very high budgets but never surpass them.
- Majority voting trails all verifier-based methods substantially, reaching only about 29% at 512 generations.

**Difficulty-bin analysis for search (Figure 3, right).** The per-difficulty breakdown (beam search M = 4 vs. best-of-N weighted, shown at four budget levels: 4, 16, 64, 256 generations) reveals the core pattern:

- **Bin 1 (easiest):** Beam search accuracy *decreases* from roughly 78% to 77% as the budget goes from 4 to 256, while best-of-N weighted increases from 68% to 88%. This is the clearest evidence of PRM over-optimization — beam search finds solutions that exploit the verifier signal.
- **Bin 2:** Beam search improves modestly (roughly 14% → 32%) but best-of-N weighted improves faster (roughly 14% → 60%), maintaining a clear advantage at high budgets.
- **Bin 3:** Beam search consistently outperforms best-of-N weighted across all budgets, reaching roughly 34% vs. 23% at 256 generations.
- **Bin 4:** Beam search shows the strongest relative advantage, reaching roughly 17% vs. 10% for best-of-N at 256 generations.
- **Bin 5 (hardest):** Both methods hover near 1–3% regardless of budget. No method makes meaningful progress.

**Compute-optimal search (Figure 4).** By selecting the best search strategy per difficulty bin at each budget level:

- At 16 generations, compute-optimal (oracle bins) achieves approximately 27% accuracy, roughly matching PRM best-of-N weighted at 64 generations — a 4× compute reduction.
- At 256 generations, compute-optimal oracle reaches approximately 39.5%, surpassing PRM best-of-N weighted at the same budget (roughly 37%).
- Compute-optimal with predicted difficulty bins tracks the oracle version closely, particularly at lower budgets. The two curves "largely overlap" per the authors (Figure 4), with the predicted version reaching approximately 37% at 256 generations.
- Both compute-optimal variants consistently outperform ORM best-of-N weighted (which peaks around 34% at 512 generations) and majority voting (around 29%).

**PRM vs. ORM (Figure 14, Appendix F).** At 2048 samples, PRM best-of-N weighted achieves approximately 40% accuracy versus roughly 35% for ORM best-of-N weighted and roughly 30% for majority voting. The gap between PRM and ORM widens with the number of samples, confirming the PRM's superior scaling properties.

#### Revision Model Results (Section 6)

**Revision model pass@1 trajectory (Figure 6, left).** Starting from approximately 18.2% pass@1 at step 1, the revision model's per-step accuracy improves to roughly 24–25% by steps 15–20, and remains in the 23–25% range out to 64 steps. The model generalizes beyond its 4-step training horizon.

**Sequential vs. parallel (Figure 6, right).** At 64 generations:
- Sequential + best-of-N weighted: approximately 41.5%
- Parallel + best-of-N weighted: approximately 39%
- Sequential + majority: approximately 38%
- Parallel + majority: approximately 35%

Sequential outperforms parallel under both selection mechanisms, with the verifier-based gap (roughly 2.5 percentage points) being slightly narrower than the majority-based gap (roughly 3 points).

**Sequential-to-parallel ratio sweep (Figure 7, left).** For a fixed generation budget, varying the ratio reveals:
- At 256 generations, the optimal ratio is around 2¹ to 2³ (2:1 to 8:1 sequential-to-parallel), achieving approximately 43–44% accuracy.
- Fully parallel (leftmost point) yields approximately 40%.
- Fully sequential (rightmost point) yields approximately 42%.
- At lower budgets (8–32 generations), fully sequential is optimal — the curves are monotonically increasing with the sequential-to-parallel ratio.

**Difficulty-dependent ratio (Figure 7, right).** At a fixed budget of 128 generations:
- **Bin 1:** Performance is essentially flat across all ratios, around 90–92%. Easy questions are insensitive to the allocation strategy.
- **Bin 2:** Slight advantage for higher sequential ratios, approximately 63% at fully sequential vs. 58% at fully parallel.
- **Bin 3:** A clear optimal ratio emerges at moderate sequential-to-parallel values (around 2¹ to 2³), reaching approximately 42% vs. 35% at the extremes.
- **Bin 4:** Similar pattern, with the peak at a moderate ratio achieving roughly 18% vs. 14% at fully parallel.
- **Bin 5:** All ratios produce roughly 2–3% accuracy. No allocation strategy helps.

**Compute-optimal revisions (Figure 8).** Selecting the optimal sequential-to-parallel ratio per difficulty bin:
- At 64 generations, compute-optimal oracle achieves approximately 40%, matching parallel best-of-N weighted at 256 generations — a 4× improvement.
- At 256 generations, compute-optimal oracle reaches approximately 44%, compared to roughly 41% for best-of-N weighted and 37% for parallel-only.
- Compute-optimal predicted bins perform slightly below oracle bins at high budgets (approximately 41% at 256 generations) but still substantially outperform the parallel baseline.
- Notably, the parallel baseline appears to **plateau** around 36–37% at high budgets, while compute-optimal scaling continues to improve, suggesting that the gains from adaptive allocation compound at higher budgets.

#### FLOPs-Matched Comparison: Test-Time vs. Pretraining Compute (Section 7)

**Revisions (Figure 9, left; Figure 1, top-right bar chart).** Comparing PaLM 2-S* with compute-optimal revisions against the ~14× larger model:

| Difficulty | R ≪ 1 (0.16) | R ≈ 1 (0.79) | R ≫ 1 (22) |
|---|---|---|---|
| Easy (bin 1) | +11.8% | +3.5% | −11.9% |
| Medium (bin 2–3) | +27.8% | +16.7% | +5.4% |
| Hard (bins 4–5) | +21.6% | −(implied negative) | −37.2% |

Numbers from the bar chart in Figure 1, top-right. Note: the "easy/medium/hard" groupings in the bar chart differ slightly from the five difficulty bins, aggregating bins for readability.

At R ≪ 1, test-time compute outperforms the larger model across **all** difficulty levels. At R ≫ 1, it only remains preferable on easy questions, with hard questions showing a −37.2% relative disadvantage.

**PRM search (Figure 9, right; Figure 1, bottom-right bar chart).** The pattern is starker:

| Difficulty | R ≪ 1 (0.16) | R ≈ 1 (0.79) | R ≫ 1 (22) |
|---|---|---|---|
| Easy | +19.1% | +2.2% | +2.0% |
| Medium | 0.0% | −35.3% | −30.8% |
| Hard | −3.6% | −35.3% | −52.9% |

PRM search shows weaker benefits than revisions for the FLOPs-matched comparison, with substantial disadvantages on medium and hard questions even at moderate R values. On easy questions, test-time compute remains preferable across all R regimes, though the margin narrows significantly.

**Figure 9 detail.** The line plots show accuracy per difficulty bin as test-time compute scales. The 14× larger model's greedy performance (stars) is placed at three x-axis positions corresponding to the three R values. Where the compute-optimal scaling line is above the star, test-time compute wins. On bin 1 (purple, topmost line), the scaling line is above all three stars for revisions. On bin 5 (blue, bottommost line), the line is below all three stars and essentially flat near 0–5%, confirming that no amount of test-time compute helps on the hardest problems.

---

### Ablation Studies and Robustness Checks

- **PRM aggregation strategy (Appendix E, Figure 13):** Comparing "min," "prod," and "last" step-wise aggregation: "last" achieves roughly 37% at 256 samples, "min" achieves roughly 35%, "prod" achieves roughly 27%, and ORM achieves roughly 34%. The "last" aggregation's superiority is notable because it effectively reduces the PRM to ORM-like behavior at aggregation time, yet the PRM still outperforms a separately trained ORM. The authors interpret this as evidence that step-level PRM training provides beneficial representation learning even when intermediate predictions are not directly used.

- **PRM vs. ORM at scale (Appendix F, Figure 14):** The PRM consistently outperforms the ORM, with the gap widening at higher sample counts: at 2048 samples, PRM best-of-N weighted reaches approximately 40% vs. ORM's 35%.

- **Revision model verifier choice (Appendix J, Figure 15a):** The base-LM PRM underperforms the revision-specific ORM when scoring revision model outputs, with sequential + base-LM PRM achieving roughly 40% at 64 generations vs. sequential + revision ORM at roughly 42%. This confirms distribution shift as a practical concern — the PRM trained on base model outputs does not directly transfer to revision model outputs.

- **Revision history in verifier context (Appendix J, Figure 15b):** Including previous revisions in the ORM's context provides a small improvement over the no-history ablation (approximately 1–2 percentage points at 64 generations), but both variants outperform the parallel baseline, confirming that the sequential sampling benefit is not solely attributable to the verifier seeing more context.

- **Oracle vs. predicted difficulty bins (Figures 4, 8, and Appendix C, Figures 11–12):** Both oracle and predicted bins yield qualitatively similar trends across difficulty levels. Predicted bins show slightly lower performance at high budgets in the revision setting (roughly 41% vs. 44% at 256 generations in Figure 8) but essentially identical performance in the search setting (Figure 4). This is the critical robustness check: the compute-optimal strategy works without ground-truth labels, though with some degradation at the highest revision budgets.

- **Majority voting for revisions (Appendix B, Figure 10):** The sequential-to-parallel ratio trends observed with verifier-based selection are replicated with majority voting: easy questions are insensitive to ratio, hard questions show an optimal intermediate ratio, and fully sequential marginally outperforms fully parallel in aggregate. This demonstrates that the benefits of sequential revisions are not an artifact of the verifier.

- **ReST^EM revision model (Appendix K, Figure 16):** An attempt to further optimize the revision model using ReST^EM (Singh et al., 2024) backfires: additional sequential revisions **substantially hurt** performance with this model. At 256 generations, fully sequential performance drops to approximately 33.5% compared to roughly 38.5% at the optimal ratio. The authors hypothesize that the on-policy data collection in ReST^EM exacerbates spurious correlations in revision data, causing the model to fail to learn the revision task properly. This is a notable negative result that highlights the sensitivity of revision training to the data generation procedure.

---

### Critical Assessment

**Claim 1: "Compute-optimal scaling improves efficiency by more than 4× over best-of-N."** This claim is supported but with an important caveat about what "efficiency" means in context. The 4× figure specifically refers to achieving equivalent accuracy with 4× fewer generations — in the search setting, 16 generations of compute-optimal scaling match PRM best-of-N weighted at 64 generations (Figure 4); in the revision setting, 64 generations of compute-optimal scaling match parallel best-of-N weighted at 256 generations (Figure 8). The evidence is consistent across both oracle and predicted difficulty settings. However, this 4× figure does **not** include the cost of difficulty estimation — generating 2048 samples per question and scoring them with the PRM. The authors explicitly flag this in Section 3.2: "our experiments do not account for this cost largely for simplicity." In a deployment scenario where difficulty estimation cost is amortized into the total budget, the realized efficiency gain would be considerably lower than 4×. Additionally, at the highest budgets (256–512), the gap between predicted bins and oracle bins widens in some settings (Figure 8: 44% oracle vs. 41% predicted at 256 generations), suggesting the gain narrows when using realistic difficulty estimates at scale.

**Claim 2: "Test-time compute with a smaller model can outperform a ~14× larger model."** This claim holds but with sharp and explicit boundary conditions that the paper itself documents. It works for easy-to-medium problems at low inference-to-pretraining token ratios (R ≪ 1), where the smaller model with compute-optimal test-time scaling outperforms the larger model by +11.8% to +27.8% (revisions) or +19.1% to 0.0% (PRM search) across difficulty bins. It progressively weakens as difficulty increases or R grows. On the hardest problems (bin 5), test-time compute provides essentially zero benefit regardless of budget (Figure 9, bottom line flat near 0–5%), confirming it cannot substitute for pretraining when the base model lacks fundamental capability. One important methodological concern: the 14× larger model uses only **greedy decoding** with no test-time augmentation. This is a deliberately weak baseline for making the point that test-time compute can substitute for pretraining — a fairer comparison would give the larger model at least a modest test-time budget (e.g., best-of-8) to match what would be standard practice in deployment. Additionally, the larger model scales only parameters, not data (following the LLaMA paradigm rather than Chinchilla-optimal training), which the authors acknowledge in Section 7. A compute-optimally trained larger model would be a stronger baseline, potentially narrowing or reversing the reported advantages.

**Claim 3: "Efficacy depends critically on prompt difficulty."** This is the most robust claim in the paper, supported by replication across independent methods. The difficulty-bin analyses for search (Figure 3, right) and revisions (Figure 7, right) show qualitatively different — and sometimes opposite — effects of the same strategy at different difficulty levels. Beam search *improves* performance on medium problems (bin 3: ~34% vs. ~23% at 256 generations) while *degrading* on easy problems (bin 1: ~77% at 256 vs. ~88% for best-of-N). Sequential revisions dominate on easy problems but a balanced ratio is optimal on hard ones. The finding is replicated under majority voting (Figure 10) as well as verifier-based selection, ruling out selection-mechanism artifacts. The primary weakness is the limited sample size: 500 test questions split into quintiles of ~100 each means the analysis is based on ~100 questions per difficulty level, and with two-fold cross-validation, ~50 questions per strategy selection decision. This introduces variance that could blur the precise thresholds where strategies switch from effective to harmful, even though the overall qualitative pattern is clear.

**Claim 4: "Verifier over-optimization is the limiting factor."** This claim is strongly supported by converging evidence but the paper does not provide direct counterfactual experiments (e.g., training a better PRM and showing the over-optimization threshold shifts). The evidence is circumstantial: beam search degrades on easy problems at high budgets (Figure 3, right), lookahead search — the strongest optimizer — paradoxically performs worst overall (Figure 3, left), and qualitative examples show degenerate outputs scoring highly under the PRM (Appendix M). These patterns are consistent with over-optimization but do not rule out alternative explanations (e.g., beam search biases the output distribution in ways that happen to hurt easy problems regardless of verifier quality). An experiment comparing PRMs of different strengths to see if the over-optimization threshold scales with verifier quality would have strengthened this claim considerably.

**Missing experiments of note.** The paper studies search and revisions independently but never combines PRM tree-search with the revision model as the proposal distribution (acknowledged in Section 8). This means the results represent a lower bound on what combined approaches could achieve, and the paper cannot speak to whether the difficulty-dependent patterns observed for each method independently would persist or change under combination. The single benchmark (MATH) and single model family (PaLM 2-S*) limit generalizability — whether the difficulty-dependent ranking of strategies transfers to other reasoning domains (code generation, logical reasoning) or other model families is not established. The test set size of 500 questions, while standard for MATH, is small for the kind of fine-grained difficulty-bin analysis performed, raising questions about statistical reliability of the precise thresholds identified.

## 6. Limitations and Trade-offs

### The Difficulty Estimation Cost Is Not Accounted for in Headline Efficiency Gains

**The assumption or constraint.** The entire compute-optimal framework in DCLM rests on the ability to estimate prompt (or document) quality *before* deciding how to allocate the training or filtering budget. The paper's method for doing so — generating 2048 samples per question and averaging pass@1 rates (oracle difficulty) or PRM final-answer scores (predicted difficulty) — is exceptionally expensive. The paper acknowledges this in Section 3.2:

> "estimating difficulty in this way still incurs additional computation cost during inference... our experiments do not account for this cost largely for simplicity"

**The consequence.** The reported 4× efficiency gains over best-of-N are computed **after** difficulty is known, without amortizing the cost of learning it. In a realistic deployment, the total cost would be difficulty estimation + strategy execution, and the former (2048 samples × PRM scoring per question) could easily dominate the latter (64–256 generations for the actual strategy). The paper frames this as "an exploration-exploitation tradeoff" and suggests future work on training models to predict difficulty directly from question text (Section 3.2), but no such model is developed or evaluated. Until this gap is closed, the 4× figure should be understood as an **upper bound on achievable efficiency** rather than a realized deployment gain. For DCLM's filtering analogue — estimating whether a document is "good" by training an expensive classifier — the cost is upfront and one-time (training a fastText model on 400K examples), so it amortizes across many training runs; the analogous concern for DCLM is less about computation and more about whether the reference data used to train that classifier (OpenHermes 2.5 + ELI5) generalizes to all desired downstream behaviors.

**What evidence exists in the paper.** The paper's own analyses in Figure 4 (search) and Figure 8 (revisions) show that compute-optimal scaling with **predicted** (non-oracle) difficulty bins performs nearly as well as oracle bins at low to moderate budgets, but the gap widens at higher budgets (Figure 8: ~44% oracle vs. ~41% predicted at 256 generations for revisions). This suggests that even with perfect difficulty estimation, the policy saturates, and with imperfect (predicted) bins, it saturates earlier. The paper does **not** report the computational cost of difficulty estimation in units comparable to the strategy budget (e.g., "difficulty estimation costs roughly 16 generations of equivalent FLOPs"), making it impossible to assess whether the net efficiency gain remains positive.

**Mitigation status.** The paper flags this as a key area for future work ("pretraining or finetuning models to directly predict difficulty of a question," Section 8). It is **not** addressed in the current experiments. A practical mitigation — using fewer than 2048 samples for difficulty estimation — is not explored.

---

### Hard Problems Remain Essentially Unsolved Regardless of Budget

**The assumption or constraint.** The paper demonstrates that across all methods — search, revisions, and their compute-optimal combinations — the hardest questions (difficulty bin 5) show **near-zero improvement** regardless of compute budget. In Figure 3 (right), bin 5 accuracy hovers at 1–3% for all methods and all budget levels (4–256 generations). In Figure 7 (right), bin 5 shows roughly 2–3% accuracy irrespective of the sequential-to-parallel ratio. In the FLOPs-matched comparison (Figure 9), the bin 5 scaling line is essentially flat near 0–5%, lying below the performance of the ~14× larger model at all R values for PRM search.

**The consequence.** Test-time compute can amplify existing capability but cannot create it. If the base model's pass@1 is near zero on a problem class, no amount of search or revision will produce correct answers — there are no correct solutions in the proposal distribution to find or refine. The paper is candid about this (Section 7 takeaway box), but it means the approach offers **no path forward for genuinely novel or out-of-distribution reasoning** that exceeds the base model's training distribution. For such problems, pretraining remains the only viable path. This is a fundamental capability bound: the compute-optimal framework can only reallocate a fixed total intelligence, not increase it.

**What evidence exists in the paper.** The evidence is consistent and stark. In Figure 3 (right), bin 5 accuracy for both beam search and best-of-N weighted is indistinguishable from zero across all budgets. In Figure 7 (right), the ratio sweep for bin 5 shows flat performance at 2–3%. In Figure 9, the bin 5 line is essentially flat, and the gap to the larger model grows with R. The paper does not explore whether a different base model with higher pass@1 on these problems would change the picture — the analysis is specific to PaLM 2-S\* on MATH.

**Mitigation status.** The paper explicitly acknowledges this in Section 7:

> "Test-time and pretraining compute are not 1-to-1 exchangeable... test-time compute is powerful when problems are within the base model's reach (it already produces correct solutions at some non-trivial rate), but it cannot compensate for fundamental capability gaps that larger pretraining would address."

This is a clear statement of the limitation rather than a mitigation. The paper does not propose any method for extending test-time compute to out-of-capability problems.

---

### Single Benchmark, Single Model Family Restricts Generalizability

**The assumption or constraint.** All experiments use the MATH benchmark (500 test questions) with PaLM 2-S\* as the base model. The authors state they "believe this model is representative of the capabilities of many contemporary LLMs" (Section 4), but this claim is unverified across model families, domains, and tasks. The MATH benchmark consists exclusively of competition-level math problems requiring symbolic reasoning — a specific cognitive skill that may not transfer to other reasoning domains.

**The consequence.** Several aspects of the findings could be model-specific or domain-specific in ways that would change the core conclusions:
- The PRM's quality and over-optimization behavior depend on PaLM 2-S\*'s output distribution. A model with different calibration properties or error patterns might exhibit different difficulty-dependent scaling curves or different over-optimization thresholds.
- The revision model's ability to learn from incorrect in-context examples depends on the base model's in-context learning capabilities, which vary substantially across model families (e.g., GPT-4 vs. PaLM 2 vs. Llama). The training procedure (edit-distance-based pairing of incorrect-to-correct trajectories) may not transfer to models with different failure modes.
- The MATH benchmark's difficulty structure may not mirror other reasoning domains (code generation, logical deduction, scientific QA). The key finding — that beam search helps medium problems but hurts easy ones — could be an artifact of MATH's specific distribution of solution strategies, where medium problems have identifiable partial solutions but easy problems have solutions that are hard to score incrementally.

**What evidence exists in the paper.** The paper provides **no** out-of-domain or out-of-model experiments. All 416 baseline experiments are on MATH. The authors acknowledge this in Section 6:

> "Due to compute constraints, we could only ablate design dimensions individually and could not test all approaches at larger scales nor train models beyond 7B parameters."

But the domain/model-family limitations are not specifically called out as threats to generalizability — they are mentioned in the context of scale constraints, not distributional robustness.

**Mitigation status.** The paper does not attempt to mitigate this limitation. The authors frame the work as "a starting point for further research on data curation" (Section 6) and express hope that the open-source testbed will enable replication across models and domains. From a practical standpoint, a practitioner considering deploying these methods on a different model family (e.g., Llama-3, GPT-4) or a different task family (e.g., code generation, multi-step planning) cannot assume the difficulty-dependent strategy rankings will transfer without empirical validation.

---

### The ~14× Larger Model Baseline Is Deliberately Weak

**The assumption or constraint.** The FLOPs-matched comparison in Section 7 scales model parameters while holding training data fixed, following the LLaMA paradigm (Touvron et al., 2023). The authors acknowledge that this departs from compute-optimal pretraining (Hoffmann et al., 2022), where both data and parameters are scaled equally:

> "We choose this setting as it is representative of a canonical approach to scaling pretraining compute and leave the analysis of compute-optimal scaling of pretraining compute where the data and parameters are both scaled equally to future work."

Additionally, the ~14× larger model uses only **greedy decoding** — no majority voting, no best-of-N, no search.

**The consequence.** A Chinchilla-optimal model trained with ~14× more total FLOPs (scaling both parameters and data) would likely outperform a parameter-only-scaled model, making the pretraining baseline **weaker than it needs to be** for the headline claim that test-time compute can substitute for pretraining. The reported advantages — e.g., +27.8% relative improvement on easy questions at R ≪ 1 for revisions — may shrink or reverse against a properly compute-optimal larger model. Similarly, giving the larger model even a modest test-time compute budget (best-of-8 with majority voting, which is free in latency terms if run in parallel) would create a much stronger baseline. The fact that the larger model uses greedy decoding means the comparison is not between "small model with test-time compute" and "large model," but between "small model with sophisticated inference strategies" and "large model used in its weakest possible configuration."

**What evidence exists in the paper.** The paper does not provide any ablation with the larger model using non-greedy decoding or any test-time compute augmentation. Table 8 (the state-of-the-art comparison) shows that models like Llama-3-8B (66% MMLU, trained with ~6.6× more compute than DCLM-BASELINE) and Gemma-8B (64.3% MMLU) achieve strong performance without any reported test-time augmentation, suggesting that pretraining compute does translate to downstream accuracy even without special inference procedures.

**Mitigation status.** The authors are transparent about this choice, framing it as a "canonical approach to scaling pretraining compute." They explicitly flag the compute-optimal pretraining comparison as future work. However, the paper's abstract and introduction prominently feature the claim that DCLM-BASELINE models are "comparable to Mistral-7B-v0.3 and Llama 3 8B on MMLU... while being trained with 6.6× less compute" — a claim that may not hold if the comparison were against a compute-optimally trained larger model or a larger model using even minimal test-time augmentation.

---

### Revisions and Search Are Studied Independently, Not Combined

**The assumption or constraint.** The paper studies two complementary axes — PRM search (modifying the verifier) and iterative revisions (modifying the proposal distribution) — as independent mechanisms. Section 8 explicitly acknowledges:

> "we did not experiment with PRM tree-search techniques in combination with revisions... This combination could break through the performance ceiling that each method individually hits, particularly on medium-difficulty problems where both mechanisms show complementary strengths."

**The consequence.** The current results represent a **lower bound** on what a fully integrated system could achieve. The two mechanisms have complementary strengths: revisions improve the proposal distribution (generating better candidates, particularly on easy problems where incremental refinement helps), while PRM search improves candidate selection (finding the best among generated candidates, particularly on medium problems where diverse exploration is needed). Applying beam search to revision model outputs — or using the PRM to guide which revisions to pursue (e.g., deciding when a revision trajectory is improving and when it should be terminated) — could yield gains beyond either method alone. The difficulty-dependent strategy selection that the compute-optimal policy performs (choosing between search on medium problems and revisions on easy problems) might be subsumed into a single combined algorithm that adapts its behavior continuously rather than selecting from a discrete menu of strategies.

**What evidence exists in the paper.** The paper provides indirect evidence that the combination could matter: Figure 3 (right) shows beam search outperforming best-of-N on medium problems; Figure 7 (right) shows sequential revisions outperforming parallel sampling on easy problems; and the compute-optimal policies in Figures 4 and 8 select different strategies for different difficulty bins. This establishes that the methods are complementary across the difficulty spectrum, but does **not** demonstrate that combining them would produce additive gains. The paper does not provide even a small-scale experiment testing whether revisiting beam-search outputs with revision steps improves accuracy.

**Mitigation status.** The paper explicitly flags this as future work (Section 8). No experiments address it. A practitioner implementing these methods would need to decide whether to invest in both components (search infrastructure + revision model training) without evidence that the combination adds value beyond the compute-optimal selection of either individually.

---

### The Revision Model Has a 38% Correct-to-Incorrect Reversion Rate

**The assumption or constraint.** The revision model is trained only on sequences where all in-context answers are incorrect (followed by a correct target). At test time, the model may encounter correct answers in its context — produced during earlier revision steps — and will attempt to "revise" them, often incorrectly. The paper reports that approximately **38% of correct answers get converted back to incorrect ones** using a naive approach (Section 6.1):

> "The correct-to-incorrect reversion problem... approximately 38% of correct answers get converted back to incorrect ones"

**The consequence.** The revision chain is fundamentally unstable — it does not converge to a correct answer even when it produces one. This means the system must rely on post-hoc selection (majority voting or verifier-based selection across the entire chain of revisions) to recover from this regression. The revision process is therefore not a hill-climbing algorithm on solution quality (where each step improves the answer) but rather a random walk that occasionally steps toward correctness and frequently steps away. This makes the revision model inefficient in terms of the expected accuracy of the final output — the 24–25% pass@1 at late steps (Figure 6, left) is an average across chains that may have found and then lost correct answers multiple times.

**What evidence exists in the paper.** Figure 6 (left) shows the revision model's per-step pass@1 stabilizing around 24–25% from steps 15–64, with no upward trend. The 38% figure is reported in Section 6.1. The mitigation (majority voting or verifier selection across the chain) raises the effective accuracy from the raw per-step pass@1 to the ~41.5% reported for sequential + best-of-N weighted at 64 generations (Figure 6, right), a substantial gap that reflects the benefit of recovering correct answers that were later overwritten.

**Mitigation status.** The paper mitigates this partially through post-hoc selection mechanisms (majority voting, verifier-based selection), but these are patches rather than solutions. The selection mechanisms do not prevent the reversion — they only recover from it by evaluating answers across the entire chain. A more principled solution, such as training the model to recognize when no revision is needed or to output a special "keep previous answer" token, is not explored. The ReST^{EM} experiment (Appendix K, Figure 16) attempted to optimize the revision model further but caused degradation, suggesting the revision training procedure is fragile and the reversion problem is not trivially solvable by more training. The paper does not provide evidence on whether longer revision chains (beyond 64 steps) continue to exhibit the ~38% reversion rate or whether it worsens as the chain lengthens.

## 7. Implications and Future Directions
- How this work changes the landscape
  - Establishes the first controlled, transparent, multi-scale benchmark for LM data curation with an unprecedented 240T-token pool and released tooling. This enables data-centric advances to be compared apples-to-apples and makes data design a first-class research axis alongside model and compute.

- Follow-up research enabled or suggested
  - Domain-aware curation: Extend DCLM to targeted domains (code, math, multilingual) with domain-specific metrics and pools (Section 6).
  - Better filtering models: Replace fastText with compact neural filters or hybrid methods (e.g., Color-Filter, data-selection via importance reweighting), while preserving efficiency.
  - Causal studies of contamination and duplication: More precise decontamination definitions per task; measure memorization vs. generalization trade-offs under different dedup regimes (Appendix L, O).
  - Fairness and safety: Integrate toxicity/PII filters, opt-out mechanisms, and fairness-aware curation into DCLM and measure trade-offs (Appendix S; Appendix U).

- Practical applications and use cases
  - Organizations can use DCLM’s pipeline to construct compute-efficient training sets tailored to their constraints, with confidence that small-scale experiments will likely transfer (Figure 3).
  - The released `DCLM-BASELINE` and models provide strong open baselines for general-purpose LMs, and the tooling (extraction, dedup, filtering, evaluation) can be adopted as production-quality data infrastructure (Appendix D; Section 5).

In short, DCLM reframes LM progress as a data problem with reproducible levers. The central practical lesson is that careful extraction, scalable deduplication, and a simple—but well-chosen—model-based filter can unlock large accuracy gains at far lower compute (Tables 3–5, 8; Figure 1).
