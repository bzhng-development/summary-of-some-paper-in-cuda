# DeepSeek-Coder: When the Large Language Model Meets Programming - The Rise of Code Intelligence

**ArXiv:** [2401.14196](https://arxiv.org/abs/2401.14196)

## 🎯 Pitch

DeepSeek-Coder introduces an open-source series of large language models for programming, uniquely trained on 2 trillion tokens of curated, repository-level code and equipped with advanced infilling capabilities and a 16K-token context window. By capturing inter-file dependencies and enabling powerful ‘fill-in-the-middle’ code completion, DeepSeek-Coder not only achieves or exceeds state-of-the-art performance among open models, but even surpasses closed models like GPT-3.5 on key code benchmarks, democratizing advanced code intelligence for research and industry alike.

---

## 1. Executive Summary

This paper introduces the DeepSeek-Coder series, a family of open-source code language models ranging from 1.3B to 33B parameters trained from scratch on 2 trillion tokens across 87 programming languages. The models employ two key mechanisms: **repository-level data construction** during pre-training (arranging files within a project by their dependency order via topological sorting so that cross-file context precedes each file) and a **Fill-In-the-Middle (FIM)** training objective (randomly splitting code into prefix-suffix-middle segments using the PSM mode at a 50% rate to enable code infilling alongside next-token prediction). On the HumanEval multilingual benchmark, DeepSeek-Coder-Base 33B achieves state-of-the-art open-source performance with an average accuracy of 50.3%, surpassing the similarly sized CodeLlama-Base 34B by approximately 9 percentage points, while the 6.7B variant already exceeds CodeLlama's 34B model. After instruction tuning, DeepSeek-Coder-Instruct 33B outperforms OpenAI's GPT-3.5 Turbo on HumanEval (79.3% vs. 76.2%), establishing that open-source code models can match or exceed closed-source alternatives on code generation tasks when trained on high-quality, repository-structured corpora with FIM augmentation.

## 2. Context and Motivation

### The Core Problem: Open-Source Code Models Lag Behind Closed-Source Alternatives

The paper addresses a structural imbalance in the landscape of code intelligence: despite rapid progress in large language models for programming, the most capable models remain proprietary and inaccessible. The fundamental gap is not merely a matter of benchmark scores — it is a question of **who can study, modify, and deploy these models**. As the authors state in the introduction, "the predominance of closed-source models has restricted extensive research and development." When models like OpenAI's GPT-3.5 and GPT-4 (OpenAI, 2023) achieve strong performance on code generation tasks without being specifically trained for them, they demonstrate the potential of large-scale language models for programming — but their closed nature means that researchers cannot inspect their training data, analyze their failure modes, modify their architectures, or deploy them in arbitrary environments without commercial agreements.

This restriction has concrete consequences. Software development tools — code completion engines, bug detection systems, automated refactoring assistants — must operate in diverse environments with varying latency, privacy, and cost constraints. Closed-source models are general-purpose and cannot be adapted to these specific deployment scenarios. An embedded code assistant running on a developer's local machine, for instance, cannot depend on an API call to a proprietary model. Similarly, organizations handling proprietary codebases may be unwilling to send source code to external services for privacy reasons. Open-source models enable on-device deployment, fine-tuning on domain-specific codebases, and transparent security auditing — all of which are impossible or severely constrained with closed-source alternatives.

### The Scale of the Performance Gap (Prior to DeepSeek-Coder)

At the time of this paper's release (January 2024), the performance gap between open-source and closed-source code models was substantial. The authors present baselines in Table 3 that make this concrete. On the HumanEval multilingual benchmark (covering eight programming languages), the leading open-source models achieved the following average accuracies:

- **CodeLlama-Base 34B**, the largest open-source code model at the time, achieved 41.0%
- **StarCoderBase 16B** achieved 28.0%
- **CodeGeeX2 6B** achieved 24.5%

Meanwhile, OpenAI's **GPT-3.5 Turbo** achieved 64.9%, and **GPT-4** reached 76.5% — roughly 1.6× and 1.9× the performance of the best open-source model, respectively. This gap of 24–35 absolute percentage points on HumanEval means that, for a substantial fraction of programming problems, closed-source models succeed where open-source models fail.

The situation on more realistic programming benchmarks was equally stark. On the DS-1000 benchmark (Table 4), which tests real-world data science workflows across libraries like Matplotlib, NumPy, and Pandas, CodeLlama-Base 34B achieved only 34.3% average accuracy — and no closed-source baselines were even reported because the gap was so wide. On the LeetCode Contest benchmark (Table 5), which comprises genuine competition-level problems collected from mid-2023 to early 2024, the best open-source model (Phind-CodeLlama-V2 34B) achieved 13.3% overall accuracy, while GPT-3.5 Turbo managed 23.3% and GPT-4 Turbo reached 40.6%.

### Where Prior Open-Source Approaches Fall Short

The paper identifies several specific shortcomings in existing open-source code models that contribute to this performance gap.

#### File-Level Pre-Training Ignores Cross-File Dependencies

Prior code models — including StarCoder (Li et al., 2023), CodeGen (Nijkamp et al., 2022), and CodeLlama (Roziere et al., 2023) — were trained primarily on **file-level** source code. The training procedure concatenates individual files from repositories without regard to the structural relationships between them. As the authors note in Section 2.2, "large language models for code are mainly pre-trained on file-level source code, which ignores the dependencies between different files in a project."

This is a significant limitation because real-world software development is inherently multi-file. A function defined in one module is imported and called in another; a class declared in one file is extended in a second; a header file in C/C++ defines interfaces implemented elsewhere. When a model is trained only on isolated files, it learns to generate code in a vacuum — it cannot leverage the context of what other files in the same project contain, and it cannot reason about how the code it generates should integrate with existing project structure.

The practical consequence, which the authors explicitly name, is that file-level-trained models "struggle to effectively scale to handle entire project-level code scenarios." This limitation manifests most clearly in cross-file code completion tasks, where a model must fill in code that depends on definitions and imports from other files. The CrossCodeEval benchmark (Ding et al., 2023), used in Section 4.3, is specifically designed to require cross-file context for accurate completion — and prior models perform poorly on it without retrieval augmentation.

#### Insufficient Attention to Fill-in-the-Middle Capabilities

Code completion tools require a capability distinct from simple code generation: the ability to insert code in the **middle** of existing code, given both a prefix (what comes before) and a suffix (what comes after). This task, known as Fill-in-the-Middle (FIM), cannot be learned from standard autoregressive next-token prediction alone. While several prior works explored FIM training — notably InCoder (Fried et al., 2022), SantaCoder (Allal et al., 2023), StarCoder (Li et al., 2023), and CodeLlama (Roziere et al., 2023) — the optimal configuration of FIM training (mode, rate, interaction with next-token prediction) remained poorly understood.

The paper identifies a concrete tension here: aggressive use of FIM training degrades code generation performance. In the ablation experiments of Section 3.1.2 (Figure 3), using a 100% FIM rate — meaning every training sample uses the fill-in-the-middle objective — achieves the best HumanEval-FIM scores but produces "the weakest code completion capability." This is non-obvious because the FIM objective involves rearranging the original text order (the middle segment is moved to the end), which disrupts the left-to-right causal structure that the model relies on for next-token prediction. Finding the right balance between FIM and standard autoregressive training — a mixed-objective approach that yields strong performance on both — was an open question that the paper addresses directly.

#### Training Data Quality and Organization

The paper argues that training data quality, not just quantity, is a key differentiator. While prior models like StarCoder (Li et al., 2023) and CodeLlama (Roziere et al., 2023) also used large-scale code corpora (The Stack dataset, totaling approximately 3 TB), the DeepSeek-Coder authors implement a more aggressive and principled filtering pipeline (Section 2). Their approach involves:

- **Rule-based filtering** adapted from StarCoder but applied more aggressively: removing files with extreme line lengths, low alphabetic character ratios, and language-specific quality heuristics. The filtering reduces the raw data to only **32.8% of its original size**, suggesting that a substantial majority of crawled GitHub code is low-quality.
- **Dependency-aware file ordering** (described in Algorithm 1): files within a repository are arranged using topological sort based on import/inclusion relationships. This ensures that when the model processes a file during training, its dependencies (the files it imports or includes) have already appeared in the context — teaching the model to use cross-file context naturally.
- **Repository-level deduplication** rather than file-level deduplication: the authors argue that deduplicating at the file level risks breaking repository structure by removing individual files while keeping others from the same project. By treating whole repositories as deduplication units, they preserve the structural integrity that their dependency-aware ordering creates.
- **Quality screening with compilers and heuristics**: code with syntax errors, poor readability, or low modularity is filtered out using compiler checks and a learned quality model.

These data curation steps represent a key contribution claim: the paper argues that the quality and organization of training data — specifically, preserving project-level structure and aggressively filtering low-quality code — is at least as important as model size or architecture for achieving competitive code generation performance.

### How This Paper Positions Itself

The paper positions DeepSeek-Coder not as a fundamentally new architecture or training algorithm, but as a combination of **carefully engineered training practices** that collectively close the gap between open-source and closed-source code models. The framing is deliberately practical rather than theoretical: the contributions center on data construction methodology (repository-level organization), training objective configuration (the FIM rate and mode ablation), and comprehensive evaluation across a wide range of benchmarks.

Relative to specific prior works:

- **Vs. StarCoder (Li et al., 2023):** DeepSeek-Coder adopts StarCoder's filtering rules as a starting point but extends them with repository-level dependency analysis and deduplication. StarCoder was also trained with FIM, but the paper argues that the 50% PSM rate with the specific sentinel token format yields better balance than prior configurations.

- **Vs. CodeLlama (Roziere et al., 2023):** CodeLlama was trained via continued pre-training from LLaMA2, meaning it inherits a general-purpose language model as its starting point. DeepSeek-Coder, by contrast, is trained **from scratch** on a code-dominant corpus (87% code, 10% code-related English, 3% Chinese). The paper demonstrates that this code-first approach yields stronger code performance at equivalent model sizes — the 6.7B DeepSeek-Coder matches the 34B CodeLlama on HumanEval, and the 33B model substantially exceeds it. However, the paper also acknowledges the trade-off: training from scratch on code means weaker natural language capabilities, which is why Section 5 introduces DeepSeek-Coder-v1.5, which continues pre-training from the general-purpose DeepSeek-LLM checkpoint to recover natural language performance.

- **Vs. Closed-source models (GPT-3.5, GPT-4):** The paper's explicit goal is to "significantly narrow the performance gap" rather than claim complete parity. The results in Table 3 show that DeepSeek-Coder-Instruct 33B surpasses GPT-3.5 Turbo on HumanEval (79.3% vs. 76.2%) but still trails GPT-4 (84.1%). On harder benchmarks like LeetCode Contest (Table 5), the gap to GPT-4 Turbo remains wide (28.9% vs. 41.8% with chain-of-thought). The paper is candid about this — the contribution is making competitive performance openly accessible, not claiming superiority over the most advanced proprietary systems.

- **Vs. The broader open-source code model landscape:** The paper provides the most thorough evaluation of a single model family across code generation (HumanEval, MBPP, DS-1000, LeetCode), code infilling (HumanEval-FIM), cross-file completion (CrossCodeEval), and program-based math reasoning (GSM8K, MATH, and others). This breadth of evaluation, combined with the release of the LeetCode Contest benchmark as a new resource, positions the paper as a comprehensive empirical contribution rather than a method-focused one.

### Why This Matters Beyond Benchmark Numbers

The paper's contribution has implications that extend beyond the benchmark tables. By demonstrating that repository-level data organization meaningfully improves cross-file code generation (Section 4.3, Table 7), the paper establishes a new standard for how code pre-training corpora should be constructed. Future code models should arguably incorporate dependency structure, not just concatenate random files.

The FIM ablation in Section 3.1.2 (Figure 3) provides actionable guidance for practitioners: a 50% FIM rate in PSM mode offers the best trade-off between infilling and generation capabilities. The finding that 100% FIM severely degrades next-token prediction performance is a warning against naive application of the technique, and the superiority of PSM over MSP (masked span prediction) contradicts the earlier CodeGen2.5 finding that MSP may enhance performance.

Finally, the paper's demonstration that the 6.7B model matches the 34B CodeLlama — a 5× parameter reduction with competitive performance — has practical significance for deployment. Smaller models enable on-device code assistants, lower-latency completion engines, and fine-tuning on consumer hardware. This efficiency advantage, achieved through data quality improvements rather than architectural innovation, shifts the conversation from "how large must a code model be?" to "how well-curated must a code model's training data be?"

## 3. Technical Approach

### 3.1 Reader Orientation

This paper describes how to build a family of decoder-only Transformer language models that can write, complete, and understand code across 87 programming languages. The core problem is that existing open-source code models are trained on isolated source files, which prevents them from learning how different files in a software project depend on each other — a capability that real-world programming demands. The solution takes the form of a **two-axis training strategy**: on the data axis, files within each repository are arranged in dependency order (via topological sorting) so that the model sees cross-file context during pre-training; on the objective axis, standard next-token prediction is combined with a fill-in-the-middle task at a carefully balanced 50% rate, so the model learns both generation and infilling without either capability cannibalizing the other.

### 3.2 Big-Picture Architecture (Diagram in Words)

The system has five major components that operate in sequence:

1. **Data Crawling and Filtering** — raw GitHub repositories are collected, filtered through rules adapted from StarCoder (line length limits, alphabetic ratio thresholds, language-specific heuristics), reducing the corpus to 32.8% of its original size. This step removes obviously low-quality code.

2. **Dependency Parsing and Topological Sorting** — for each repository, import/inclusion relationships between files are extracted using language-specific regular expressions. Files are then arranged in an order that ensures dependencies appear before dependents, producing a single concatenated training sample per repository subgraph. File path comments are prepended so the model learns path information.

3. **Repository-Level Near-Deduplication** — rather than deduplicating individual files (which could break repository structure by removing some files while keeping others), the concatenated repository-level sequences are deduplicated as whole units using the same near-deduplication algorithm from prior work.

4. **Quality Screening and Decontamination** — a compiler and learned quality model remove code with syntax errors, poor readability, or low modularity. N-gram filtering removes any code overlapping with test sets (HumanEval, MBPP, GSM8K, MATH) using 10-gram exact matching and shorter exact-match filtering for strings between 3 and 9 characters.

5. **Model Training with Dual Objectives** — the filtered data is used to train a standard decoder-only Transformer (with Rotary Position Embeddings, SwiGLU activations, and grouped-query attention on the largest variant) using two interleaved objectives: next-token prediction and fill-in-the-middle. The FIM objective splits each document into prefix, suffix, and middle segments, rearranges them in PSM (Prefix-Suffix-Middle) order with sentinel tokens, and asks the model to generate the middle given the prefix and suffix. This dual-objective training runs at a 50% FIM rate (half of documents use FIM, half use standard next-token prediction).

Information flows in one direction: raw GitHub data → filtered files → dependency-sorted repository samples → deduplicated sequences → quality-screened corpus → shuffled training batches → model training with interleaved objectives. After pre-training, the base model can be further fine-tuned on instruction data (to produce DeepSeek-Coder-Instruct) or continue-trained from a general-purpose LLM checkpoint (to produce DeepSeek-Coder-v1.5).

### 3.3 Roadmap for the Deep Dive

- **First**, the data construction pipeline (Sections 2.1–2.4), because the paper's primary claim is that data quality and organization drive performance. I will explain the filtering rules, the dependency parsing algorithm (Algorithm 1), the deduplication strategy, and the quality screening — walking through why each step matters and what would break if it were skipped.

- **Second**, the training objectives (Sections 3.1.1–3.1.2), including the next-token prediction baseline and the FIM mechanism. I will explain the PSM vs. SPM modes, the sentinel token format, the FIM rate ablation results (Figure 3), why 50% PSM was chosen, and the MSP comparison.

- **Third**, the model architecture and optimization (Sections 3.2–3.4), covering the tokenizer, the Transformer configuration across model scales (Table 2), the AdamW optimizer settings, and the three-stage learning rate schedule.

- **Fourth**, the long context extension (Section 3.6), explaining the RoPE parameter modifications, the additional training steps at 16K sequence length, and the theoretical vs. empirical context window limits.

- **Fifth**, the instruction tuning procedure (Section 3.7), covering the Alpaca-format data, the cosine learning rate schedule, and the multi-turn dialogue examples.

This order follows the data through the pipeline — from raw corpus to trained model to instruction-tuned variant — which matches how the system is built and makes the dependencies between components clear.

### 3.4 Detailed, Sentence-Based Technical Breakdown

This is primarily an **empirical systems paper** whose core idea is that training code models on **repository-level data organized by dependency order** and with a **balanced mixed objective** (next-token prediction + fill-in-the-middle at 50% PSM rate) yields substantially better code generation, completion, and cross-file reasoning than prior approaches that use file-level data with simpler or unbalanced objectives. The paper does not propose novel architectures or training algorithms — instead, it demonstrates that careful engineering of the data pipeline and training objective configuration can close performance gaps that were previously attributed to model scale or closed-source advantages.

---

#### The Data Construction Pipeline: From Raw GitHub to Training Corpus

The data pipeline described in Section 2 transforms raw GitHub repositories into a carefully structured training corpus through five sequential stages. The overarching design principle is that **code in the wild is not just text — it has structure, and that structure should be preserved and exploited during training to teach the model about cross-file dependencies and project-level organization**.

##### Stage 1: GitHub Crawling and Rule-Based Filtering

The pipeline begins by collecting all public GitHub repositories created before February 2023 and retaining only files from 87 programming languages (listed in Table 1, which includes languages as diverse as Python, Java, C++, Haskell, Lean, Solidity, and Verilog).

The raw crawl produces an enormous volume of data, most of which is unsuitable for training. The authors adopt the filtering rules from the StarCoder project (Li et al., 2023), which they describe as follows (quoted from Section 2.1):

> "we filter out files with an average line length exceeding 100 characters or a maximum line length surpassing 1000 characters. Additionally, we remove files with fewer than 25% alphabetic characters. Except for the XSLT programming language, we further filter out files where the string '<?xml version=' appeared in the first 100 characters."

For specific file types, additional rules apply:

- **HTML files**: only retained if "visible text constitutes at least 20% of the code and is no less than 100 characters." This removes HTML files that are almost entirely markup with no human-readable content, which would teach the model to generate boilerplate rather than substantive code.
- **JSON and YAML files**: only kept if they have "a character count ranging from 50 to 5000 characters." These file types tend to be either trivial (short configuration snippets under 50 characters) or overwhelmingly large (data dumps exceeding 5000 characters), neither of which provides useful training signal.

The impact of these rules is dramatic: "By applying these filtering rules, we reduce the total amount of data to only 32.8% of its original size." This means that roughly two-thirds of raw GitHub code is discarded as low-quality — a finding that underscores the importance of aggressive filtering for code pre-training. The authors do not treat this as a loss but as a necessary purification: the discarded data would have added noise, taught the model poor coding practices, and consumed training compute that is better spent on higher-quality examples.

The choice to adopt StarCoder's rules directly (rather than designing new ones) is a deliberate engineering decision. StarCoder's filtering was extensively validated, and reproducing it ensures comparability with prior work while freeing the authors to focus their innovation on the repository-level organization steps that follow. However, the paper goes beyond StarCoder by adding compiler-based and model-based quality screening later in the pipeline (Stage 4), which StarCoder did not incorporate.

##### Stage 2: Dependency Parsing — Why File-Level Training Is Insufficient

This is the paper's most distinctive data construction contribution. The central observation is that prior code models "are mainly pre-trained on file-level source code, which ignores the dependencies between different files in a project." In file-level training, each source file is treated as an independent document. The model might see `import numpy as np` in one training example and the actual numpy source code in another, but there is never a training instance where the importing file appears with the imported file's content in context. The model never learns that `from utils.helper import parse_data` means "the function `parse_data` is defined in `utils/helper.py`, and here is what it does."

The consequence, which the paper states explicitly, is that file-level-trained models "struggle to effectively scale to handle entire project-level code scenarios." In practice, this means that when a developer uses a file-level model for code completion in a multi-file project, the model cannot leverage the content of other files in the same repository — it can only use the current file's context and its (implicitly learned) general knowledge of what common libraries do.

The solution is to arrange files within each repository so that **each file's context includes the files it depends on**. The procedure is described in Algorithm 1 and works as follows:

**Step 1: Extract dependency relationships.** For each pair of files within a repository, the algorithm checks whether one file depends on another. Dependencies are identified using language-specific regular expressions that match import/inclusion statements: `import` in Python, `using` in C#, `include` in C/C++, and equivalent constructs in other languages. The function `HASDEPENDENCY(fileA, fileB)` returns true if `fileA` imports, includes, or otherwise references `fileB`. For example, if `main.py` contains `from utils.metrics import compute_accuracy`, then `HASDEPENDENCY(main.py, utils/metrics.py)` is true.

**Step 2: Build a directed dependency graph.** The algorithm constructs an adjacency list `graphs` where `graphs[fileB]` contains all files that depend on `fileB`, and an in-degree dictionary `inDegree` where `inDegree[fileA]` counts how many files `fileA` depends on. The direction of edges is from dependency to dependent: if A depends on B, there is an edge B → A, and A's in-degree increases. This means files with low in-degree have few or no dependencies; files with high in-degree import many other files.

**Step 3: Identify disconnected subgraphs.** A repository may contain multiple independent components — for example, a Python package might have a `server/` directory and a `client/` directory that don't reference each other. The algorithm identifies these disconnected subgraphs using `getDisconnectedSubgraphs(graphs)` so that each subgraph can be processed independently. This prevents spurious ordering constraints between unrelated files.

**Step 4: Modified topological sort for each subgraph.** A standard topological sort selects nodes with zero in-degree, processes them, removes their outgoing edges, and repeats. This works only for directed acyclic graphs (DAGs). However, real-world import graphs often contain cycles — file A imports file B, and file B (perhaps indirectly) imports file A. The authors handle this with a crucial modification:

> "Unlike the standard approach that selects nodes with zero in-degrees, this algorithm selects nodes with minimal in-degrees, which allows it to handle cycles within the graph."

The algorithm repeatedly selects the file with the smallest in-degree that hasn't yet been added to the result list, then decrements the in-degrees of all files that depend on it. When a cycle exists, there will eventually be no node with zero in-degree, but there will be a node with minimal in-degree — and selecting it breaks the cycle. The resulting order is not a strict topological sort (since cycles violate the DAG assumption), but it is a best-effort ordering that places dependencies as early as possible given the graph structure.

**Step 5: Concatenate files with path information.** Once files are sorted within each subgraph, they are concatenated to form a single training sample. Crucially, "a comment indicating the file's path is added at the beginning of each file." This path comment is essential: without it, the model would see a sequence of code from different files with no indication of where one file ends and another begins. The path comment serves as a structural delimiter that teaches the model to associate code with its location in the project hierarchy.

The result is a training corpus where, for any given file, the model has already seen (in its context window) the files that file imports. If `main.py` imports `utils.py` and `models.py`, these files appear earlier in the training sequence. This means the model can learn to attend to cross-file context naturally — when generating code in `main.py`, the representations of `utils.py` and `models.py` are already in the model's hidden state from earlier tokens.

This is not the same as retrieval-augmented generation, which retrieves relevant files at inference time. Rather, dependency-ordered pre-training teaches the model's parameters to **expect and utilize cross-file context** as a fundamental part of code understanding. The experimental validation of this approach comes in Section 4.3 (CrossCodeEval, Table 7), where the authors show that removing repository-level pre-training ("w/o Repo Pre-training") decreases performance across Java, TypeScript, and C# languages — providing causal evidence that the dependency-aware ordering is responsible for the gains.

##### Stage 3: Repository-Level Near-Deduplication

Deduplication is a well-established technique for improving language model training efficiency and reducing memorization of training data. The standard approach in prior code models was to deduplicate at the file level: each source file is compared to every other source file, and near-duplicates are removed. The authors identify a subtle problem with this approach:

> "we perform deduplication at the repository level of code, rather than at the file level, as the latter approach may filter out certain files within a repository, potentially disrupting the structure of the repository."

The logic is clear: if a repository contains two files that are near-duplicates (say, `config_dev.py` and `config_prod.py` with only minor differences), file-level deduplication would remove one of them. This breaks the dependency graph that Stage 2 constructed — another file that imports the removed `config_dev.py` now has a dangling reference, and the structural coherence that the topological sort created is lost.

By contrast, repository-level deduplication treats the **entire concatenated repository sequence** as the unit of comparison. If two repositories are near-duplicates (e.g., a forked project with minimal changes), the entire repository sample is removed. This preserves internal structure for every repository that survives deduplication. The specific near-deduplication algorithm used is the same one applied in prior work (Kocetkov et al., 2022; Lee et al., 2022), which identifies long repetitive substrings and removes samples that share them above a threshold.

The trade-off is that repository-level deduplication might remove genuinely distinct repositories that happen to share boilerplate code (license headers, standard library imports). The authors accept this trade-off because the alternative — broken repository structure — is worse for their training objective. They are optimizing for models that understand project-level code organization, and that organization is destroyed by file-level deduplication.

##### Stage 4: Quality Screening and Decontamination

After the structural organization stages, a final quality pass removes code that would teach the model bad habits:

**Compiler-based filtering:** Code with outright syntax errors is removed by running it through language-appropriate compilers. This is only possible because the data is organized at the repository level — a single file with a missing import might compile correctly when its dependencies appear earlier in the context, but an isolated file with the same missing import would fail. The authors do not specify which compilers are used for which languages, but the principle is universal: if a compiler cannot parse the code, it likely contains errors that would confuse the model.

**Quality model and heuristic rules:** Beyond syntax, the authors apply a learned quality model (architecture unspecified) and heuristic rules targeting "poor readability, and low modularity." Readability heuristics might include metrics like variable name length, comment density, and function length. Modularity heuristics might penalize files with very long functions that could be decomposed, or files that mix unrelated concerns. The precise heuristics are not detailed, which is a minor gap in reproducibility.

**N-gram decontamination:** To prevent test set leakage, the training data is filtered against known benchmarks. The paper describes the procedure precisely:

> "if a piece of code includes a 10-gram string identical to any in the test data, it is excluded from our training data. In cases where the test data comprises strings that are shorter than 10-grams but no less than 3-grams, we use an exact match approach for filtering."

This is more conservative than simple file-level exclusion. Even if a GitHub repository contains a solution to a HumanEval problem embedded in a larger file, the 10-gram matching will catch the overlap and remove the entire training sample. The threshold of 10 tokens is chosen to balance false positives (removing code that coincidentally shares short substrings with test data) against false negatives (missing contamination through paraphrased solutions). Shorter test strings (3–9 tokens) use exact matching rather than n-gram overlap because an exact match of a very short string is more likely to be meaningful contamination.

The benchmarks filtered against are HumanEval, MBPP, GSM8K, and MATH — all benchmarks used in the paper's evaluation (Section 4) and the math reasoning tasks. This decontamination is critical for credible evaluation: without it, the model might have memorized test solutions during pre-training, inflating benchmark scores relative to genuine generalization capability.

**Statistical summary.** Table 1 provides the final data statistics: 798 GB of source code across 603 million files, with the largest languages being Java (18.63%), Python (15.12%), C++ (11.39%), JavaScript (6.75%), TypeScript (7.60%), C# (7.34%), and PHP (7.38%). The long tail includes niche languages like Lean (0.07%), Idris (0.01%), and Bluespec (0.01%). This multi-lingual distribution is intentional — the models are meant to be general-purpose code assistants, not Python specialists. The data composition for the final training mixture is 87% source code, 10% English code-related natural language (GitHub Markdown and StackExchange), and 3% code-unrelated Chinese natural language (Section 2 introduction). The code-related English corpus serves to teach the model about code documentation, library usage discussions, and bug-fixing workflows. The Chinese corpus improves the model's Chinese language proficiency but is kept small to avoid diluting code capability.

---

#### Training Objective 1: Next Token Prediction (Standard Autoregressive LM)

The first training objective is the standard causal language modeling loss used by virtually all decoder-only Transformer models. As described in Section 3.1.1:

> "various files are concatenated to form a fixed-length entry. Then, these entries are used to train the model, enabling it to predict the subsequent token based on the provided context."

Given a sequence of tokens `$x_1, x_2, ..., x_T$`, the model is trained to maximize the probability of each token given all previous tokens:

$$P(x_1, ..., x_T) = \prod_{t=1}^{T} P(x_t \mid x_1, ..., x_{t-1})$$

where `$T$` is the sequence length and `$P(x_t \mid x_1, ..., x_{t-1})$` is the probability assigned to the correct token `$x_t$` by the model's output distribution at position `$t$`.

**What it computes:** The joint probability of the entire sequence, factored as the product of per-token conditional probabilities. At each position, the model takes all preceding tokens as input (through causal self-attention masking, which prevents attending to future positions) and outputs a probability distribution over the vocabulary. The loss is the negative log-likelihood: `$-\sum_{t=1}^{T} \log P(x_t \mid x_1, ..., x_{t-1})$`. Minimizing this loss trains the model to assign high probability to tokens that actually appear in the training data.

**Why this form:** This is the standard maximum-likelihood objective for autoregressive sequence models. The causal factorization ensures that the model can only use past context, which is required for autoregressive generation at inference time (where future tokens are unknown). The per-token decomposition makes the loss additive, enabling efficient gradient computation via teacher forcing (the model always sees the ground-truth previous tokens, not its own predictions). This objective alone is sufficient for code generation — the model learns syntax, idioms, and algorithmic patterns from millions of examples of working code — but it is insufficient for code **infilling**, because the model never learns to condition on both past and future context simultaneously.

---

#### Training Objective 2: Fill-in-the-Middle (FIM) — The PSM Mode

The second training objective addresses a capability that next-token prediction cannot provide: generating code that must fit between existing prefix and suffix code. This is essential for code completion tools, where the user's cursor is in the middle of a file and the model must insert content that is consistent with both what comes before and what comes after.

The FIM approach used in this paper is adapted from Bavarian et al. (2022) and Li et al. (2023). The core idea is to restructure training documents so that the model practices generating the middle of a document given its prefix and suffix.

**Document splitting.** For each code file, the content is divided into three contiguous segments:

- `$f_{pre}$`: the prefix — the beginning portion of the file
- `$f_{middle}$`: the middle — the portion to be predicted
- `$f_{suf}$`: the suffix — the ending portion of the file

The paper does not specify how the split point is chosen (random position? random fraction?), but the standard approach from Bavarian et al. (2022) is to sample a split point uniformly from the document length, then sample a middle span length from a truncated geometric distribution (so shorter middle spans are more common than longer ones). At an FIM rate of 0.5, roughly half of all training documents undergo this splitting, while the other half are presented in their original order for standard next-token prediction.

**The two modes: PSM and SPM.** The paper defines two structural arrangements for presenting the three segments to the model:

- **PSM (Prefix-Suffix-Middle):** The tokens appear in the order `$f_{pre}$`, then `$f_{suf}$`, then `$f_{middle}$`. The model sees the prefix, then the suffix, and must generate the middle knowing what comes on both sides.
- **SPM (Suffix-Prefix-Middle):** The tokens appear in the order `$f_{suf}$`, then `$f_{pre}$`, then `$f_{middle}$`. The suffix appears first, then the prefix, then the middle.

The distinction between PSM and SPM matters because it determines which context the model sees first. In PSM, the prefix (the more "natural" starting point since it is the actual beginning of the file) appears first, which may provide a smoother transition from the standard next-token prediction objective. In SPM, the suffix appears first, which may force the model to reason about what comes after before seeing what comes before — a more challenging but potentially more robust training signal.

**Sentinel token format.** The paper uses three special tokens to demarcate the FIM structure:

```
<|fim_start|> f_pre <|fim_hole|> f_suf <|fim_end|> f_middle <|eos_token|>
```

- `<|fim_start|>` marks the beginning of the FIM-structured sequence
- `<|fim_hole|>` separates the prefix from the suffix, indicating "here is where the middle should go"
- `<|fim_end|>` separates the suffix from the middle, indicating "now generate what goes in the hole"
- `<|eos_token|>` marks the end of the sequence

At training time, the model processes the entire sequence left-to-right with causal attention. The loss is only computed on the `$f_{middle}$` tokens (the portion after `<|fim_end|>`), because the prefix and suffix are provided as context. This is exactly analogous to standard autoregressive training — the model predicts each token given all previous tokens — but the "previous tokens" now include the suffix (which, in the original document order, would have appeared after the middle). The sentinel tokens teach the model to recognize the FIM structure and to switch between "reading mode" (processing prefix and suffix) and "writing mode" (generating the middle).

**Why FIM is implemented at the document level before packing.** The authors state:

> "We implement the Fill-in-the-Middle (FIM) method at the document level before the packing process, as proposed in the original work by Bavarian et al. (2022)."

Packing refers to concatenating multiple documents into a single fixed-length training sequence (to avoid wasting compute on padding tokens). If FIM were applied after packing, the split points could cross document boundaries, creating nonsensical training examples where the prefix comes from one file and the suffix from another unrelated file. By applying FIM at the document level first, each FIM example is a coherent single-file transformation, and packing only concatenates already-formed (FIM or standard) documents.

**The FIM rate ablation and the choice of 50% PSM.** The paper conducts a systematic comparison of FIM configurations using the DeepSeek-Coder-Base 1.3B architecture on a Python-only data subset (Section 3.1.2). Four configurations are tested:

1. **0% FIM rate (fim_0):** Standard next-token prediction only. This is the baseline for code generation capability.
2. **50% FIM rate in PSM mode (fim_0.5):** Half of documents use PSM-structured FIM, half use standard next-token prediction.
3. **100% FIM rate in PSM mode (fim_1.0):** Every document uses FIM. There is no standard next-token prediction training.
4. **50% MSP rate (msp_0.5):** Masked Span Prediction — an alternative infilling objective from T5 (Raffel et al., 2023) where multiple text spans are masked and the model learns to reconstruct them. CodeGen2.5 (Nijkamp et al., 2023) had previously suggested MSP might outperform PSM for FIM.

The evaluation uses two complementary metrics: **HumanEval-Pass@1** (standard code generation) and **HumanEval-FIM** (single-line infilling in Python, where one line of a HumanEval solution is obscured and the model must predict it). The results are shown in Figure 3, with training curves plotted against training steps.

The key findings from Figure 3:

**On HumanEval-FIM** (the infilling metric):
- `fim_1.0` (100% FIM) achieves the highest performance, reaching approximately 0.82 exact match.
- `fim_0.5` (50% FIM) is close behind at approximately 0.80.
- `msp_0.5` (50% MSP) trails at approximately 0.75.
- `fim_0` (no FIM) performs worst at approximately 0.60.

This confirms that FIM training — in any form — substantially improves infilling capability over standard next-token prediction. The 100% FIM rate provides the best infilling, but the margin over 50% is small (roughly 0.02 absolute).

**On HumanEval-Pass@1** (the code generation metric):
- `fim_0` (standard training) achieves the highest performance, reaching approximately 0.20.
- `fim_0.5` (50% FIM) is close behind at approximately 0.18.
- `fim_1.0` (100% FIM) drops sharply to approximately 0.13 — a substantial degradation.
- `msp_0.5` (50% MSP) also underperforms at approximately 0.15.

This reveals the central tension: **100% FIM training severely damages code generation capability**. The authors hypothesize that "the PSM mode may exhibit subtle differences compared to the traditional next-token prediction objective. This is primarily because PSM involves rearranging the order of the original text, potentially impacting the learning dynamics of the model." When every training example has been reordered (prefix-then-suffix-then-middle instead of the natural order), the model's internal representations are optimized for the reordered distribution, not the natural left-to-right distribution that code generation requires. The model learns to expect a `<|fim_hole|>` separator and a suffix after every prefix, which doesn't match standard code generation prompts.

**On MBPP-Pass@1** (a second code generation benchmark):
- The same pattern holds: `fim_0` leads at approximately 0.23, `fim_0.5` is at approximately 0.22, `fim_1.0` drops to approximately 0.18, and `msp_0.5` is at approximately 0.20.

The decision to use **50% PSM rate** follows directly from these results. The authors state:

> "we observe that with a 50% PSM rate, the model outperforms the MSP strategy. To achieve a balance between FIM efficiency and code completion proficiency, we ultimately choose the 50% PSM rate as our preferred training policy."

The 50% rate provides most of the infilling benefit of 100% FIM (HumanEval-FIM of 0.80 vs. 0.82) while preserving most of the code generation capability (HumanEval-Pass@1 of 0.18 vs. 0.20 for no FIM). The gap of 0.02 on each metric represents a small tax paid for having both capabilities in a single model.

**Why PSM over SPM or MSP.** The paper does not explicitly test SPM mode (the ablation only includes PSM variants and MSP), so the choice of PSM over SPM appears to be based on prior work (Bavarian et al., 2022; Li et al., 2023) rather than an ablation in this paper. The superiority of PSM over MSP contradicts the CodeGen2.5 finding that "MSP may enhance FIM performance compared to PSM." The authors do not explain this discrepancy, but possible reasons include differences in the base model architecture, the training data distribution (Python-only subset vs. full code corpus), or the specific implementation of MSP (number of masked spans, masking probability). The result stands as an empirical counterpoint to the CodeGen2.5 claim and a practical recommendation for future work.

---

#### Tokenizer: Byte Pair Encoding with 32K Vocabulary

The tokenizer is a standard Byte Pair Encoding (BPE) tokenizer trained using the HuggingFace Tokenizer library on a subset of the training corpus (Section 3.2). BPE works by starting with individual characters as the initial vocabulary, then iteratively merging the most frequent adjacent token pairs. For example, if "import" appears frequently, the merger "im" + "port" → "import" will be learned early, and later the full word "import" will be a single token rather than six individual characters.

The vocabulary size of 32,000 is a standard choice that balances several considerations:

- **Coverage:** A 32K vocabulary is large enough to represent common code tokens (keywords like `def`, `class`, `import`; operators like `+=`, `->`, `**`; and common identifiers) as single tokens, which is more efficient than representing them as sequences of subword units.
- **Efficiency:** The embedding matrix and output projection layer scale linearly with vocabulary size. At 32K, these matrices are of manageable size even for the 33B model (where the hidden size is 7168, making the embedding matrix 32,000 × 7,168 ≈ 229 million parameters, or roughly 0.7% of total parameters).
- **Out-of-vocabulary handling:** BPE can represent any token as a sequence of subword units, so there is no "unknown token" problem — even code with rare identifiers or unusual formatting can be tokenized. If a token is not in the vocabulary, BPE decomposes it into smaller known subword units.

The choice to train the tokenizer on a subset of the training corpus (rather than using a pre-existing tokenizer like GPT-2's or LLaMA's) ensures that the vocabulary is tuned to the statistical distribution of code. Code has different token frequencies than natural language — symbols like `{`, `(`, `;`, and whitespace patterns are more prevalent, and library-specific identifiers appear with high frequency in their respective ecosystems. A code-specific tokenizer compresses code more efficiently than a general-purpose one, meaning more code fits in the same context window.

---

#### Model Architecture: Decoder-Only Transformers at Three Scales

The DeepSeek-Coder models follow the same architecture as the DeepSeek LLM (DeepSeek-AI, 2024), which is a standard decoder-only Transformer with several modern architectural choices. Table 2 provides the complete hyperparameter specifications for all three model scales.

**What is a decoder-only Transformer?** All three models are autoregressive language models: they process input tokens left-to-right using causal self-attention (each token can only attend to itself and previous tokens, never to future tokens). This architecture, popularized by GPT (Radford et al., 2018) and now dominant in the field, is simpler than encoder-decoder architectures (like T5) because it uses the same parameters for both "understanding" input context and generating output tokens. The trade-off is that decoder-only models cannot naturally condition on bidirectional context (which is why FIM training is necessary for infilling — it artificially provides suffix information by placing it before the middle in the input sequence).

**Rotary Position Embedding (RoPE).** Position embeddings are necessary because self-attention is permutation-invariant — without them, the model would treat "the cat sat on the mat" and "mat the on sat cat the" identically. RoPE (Su et al., 2023) encodes position information by rotating the query and key vectors in attention by an angle proportional to their absolute position. Specifically, for a token at position `$p$` with attention head dimension `$d$`, RoPE applies a rotation matrix:

$$\text{RoPE}(x, p) = R_p \cdot x$$

where `$R_p$` is a block-diagonal rotation matrix with rotation angles `$\theta_i = 10000^{-2i/d}$` for dimension pair `$i$` and position `$p$`. The dot product between rotated query at position `$p_q$` and rotated key at position `$p_k$` depends only on the relative position `$p_q - p_k$`, which means the model naturally learns relative positional relationships without needing explicit relative position biases.

**Why RoPE over learned or sinusoidal position embeddings?** RoPE has two key advantages: (1) the relative position property means the model can generalize to sequence lengths beyond those seen during training (important for the context extension in Section 3.6), and (2) the rotation mechanism is applied to the query and key vectors in attention, which integrates naturally with the attention computation without adding parameters. Learned position embeddings (as in the original GPT) are parameter-inefficient and do not naturally generalize to longer sequences. Sinusoidal embeddings (as in the original Transformer) provide a fixed encoding but do not encode relative position as cleanly in the attention dot product.

**SwiGLU Activation.** All three models use the SwiGLU activation function in their feed-forward networks (the multi-layer perceptron block that follows each attention layer). SwiGLU, introduced by Shazeer (2020), combines the Swish activation with a gated linear unit:

$$\text{SwiGLU}(x) = xW_1 \cdot \text{Swish}(xW_2) \cdot W_3$$

where `$\text{Swish}(z) = z \cdot \sigma(z)$` (the sigmoid-gated linear function), and `$W_1$`, `$W_2$`, `$W_3$` are learned weight matrices. The dimensionality of the intermediate hidden states is given by the "Intermediate size" row in Table 2.

**Why SwiGLU over ReLU or GeLU?** SwiGLU has been shown to outperform both ReLU and GeLU in large-scale language model training (Shazeer, 2020; Touvron et al., 2023), with better training stability and slightly improved downstream performance. The gating mechanism (`$\cdot \text{Swish}(xW_2)$`) provides a learnable input-dependent activation strength, which is more expressive than a simple nonlinearity like ReLU.

**Grouped-Query Attention (GQA) on the 33B model.** The 33B variant uses GQA with a group size of 8. Standard multi-head attention (used in the 1.3B and 6.7B models) computes separate key and value projections for each attention head, which requires storing and computing over `$n\_heads \times d\_head$` key-value pairs per layer. GQA reduces this cost by sharing key and value projections across groups of query heads. With a group size of 8 and 56 total attention heads (Table 2), the 33B model has 56 ÷ 8 = 7 distinct key-value projections, each shared by 8 query heads. This reduces the key-value cache size (critical for inference memory) by a factor of 8 compared to full multi-head attention, while maintaining most of the expressivity.

**Why GQA only on the 33B model?** The memory savings from GQA are most impactful at larger model scales, where the key-value cache for long sequences can dominate GPU memory. For the 1.3B and 6.7B models, the attention heads are fewer (16 and 32 respectively) and the hidden sizes are smaller (2048 and 4096), making the key-value cache less of a bottleneck. The choice of group size 8 (meaning 7 distinct key-value projections for 56 heads) is a standard trade-off that balances memory savings against the risk of reducing attention expressivity too much — smaller group sizes would save less memory, while larger groups (e.g., 56, making it multi-query attention with a single key-value projection) might hurt quality.

**FlashAttention v2.** All models use FlashAttention v2 (Dao, 2023) to accelerate attention computation. Standard attention requires materializing the full `$N \times N$` attention matrix (where `$N$` is the sequence length), which has `$O(N^2)$` memory cost. For a 16K sequence, this is 256 million entries per attention head per layer — prohibitive for large models. FlashAttention avoids this by tiling the computation: it processes the attention matrix in blocks that fit in GPU SRAM (fast on-chip memory), computing softmax in a numerically stable way without ever materializing the full matrix in HBM (slow off-chip memory). This reduces memory usage from `$O(N^2)$` to `$O(N)$` and speeds up computation by reducing HBM accesses.

**Why FlashAttention v2 over v1?** FlashAttention v2 improves on v1 by optimizing the work partitioning between thread blocks and reducing the number of non-matmul operations (which are slower on GPUs than matrix multiplications). For training with 16K context, these optimizations are critical — without them, the attention computation would be a bottleneck that dominates training time.

**Scale-specific configurations.** Table 2 reveals the scaling pattern:

| Hyperparameter | 1.3B | 6.7B | 33B |
|---|---|---|---|
| Hidden size | 2048 | 4096 | 7168 |
| Intermediate size (FFN) | 5504 | 11008 | 19200 |
| Hidden layers | 24 | 32 | 62 |
| Attention heads | 16 | 32 | 56 |
| Attention type | Multi-head | Multi-head | Grouped-query (8) |
| Batch size (tokens) | 1024 | 2304 | 3840 |
| Max learning rate | 5.3e-4 | 4.2e-4 | 3.5e-4 |

The scaling is roughly uniform: doubling parameters from 1.3B to 6.7B (5.2×) increases hidden size by 2×, layers by 1.33×, and intermediate size by 2×. The 6.7B to 33B jump (4.9× parameters) increases hidden size by 1.75×, layers by 1.94×, and intermediate size by 1.74×. The batch size (in tokens) and learning rate follow the scaling laws from DeepSeek LLM (DeepSeek-AI, 2024): larger models train with larger total batch sizes (more tokens per update to reduce gradient noise at scale) and lower learning rates (larger models are more sensitive to parameter updates). The specific scaling law is not detailed in this paper but is referenced as being derived in the companion DeepSeek LLM paper.

---

#### Optimization: AdamW with Three-Stage Learning Rate Schedule

The training optimization follows the configuration established in DeepSeek LLM (DeepSeek-AI, 2024) (Section 3.4).

**AdamW optimizer.** AdamW (Loshchilov and Hutter, 2019) is a variant of Adam that decouples weight decay from the adaptive learning rate computation. The update rule is:

$$\theta_{t+1} = \theta_t - \eta \left( \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda \theta_t \right)$$

where `$m_t$` and `$v_t$` are the first and second moment estimates (exponential moving averages of gradients and squared gradients), `$\hat{m}_t$` and `$\hat{v}_t$` are their bias-corrected versions, `$\eta$` is the learning rate, `$\lambda$` is the weight decay coefficient, and `$\epsilon$` is a small constant for numerical stability.

**What this does:** AdamW maintains running estimates of gradient mean and variance to adapt the learning rate per parameter (parameters with consistently large gradients get smaller effective step sizes; parameters with small gradients get larger step sizes), while applying weight decay separately from the adaptive update. The decoupling is important because in standard Adam, weight decay interacts with the adaptive learning rates in ways that effectively reduce regularization strength for parameters with large gradient variance.

**The hyperparameters** are `$\beta_1 = 0.9$`, `$\beta_2 = 0.95$`. These control the decay rates of the first and second moment estimates: `$\beta_1 = 0.9$` means the first moment estimate has an exponential decay half-life of roughly 6.5 steps (the gradient 6.5 steps ago has weight 0.5). `$\beta_2 = 0.95$` gives a half-life of roughly 13.5 steps for the second moment. The higher `$\beta_2$` is standard — we want the variance estimate to average over a longer horizon than the mean estimate. The weight decay coefficient `$\lambda$` is not explicitly stated in this paper but is presumably 0.1 as in DeepSeek LLM.

**Three-stage learning rate schedule.** The paper uses a step-based schedule (not cosine or linear):

> "For the learning rate scheduling, we implement a three-stage policy, which includes 2000 warm-up steps, and set the final learning rate to 10% of the initial rate. Notably, the learning rate at each stage is scaled down to `$\sqrt{1/10}$` of the preceding stage's rate."

The three stages can be reconstructed as:

- **Stage 0 (warmup):** Learning rate increases linearly from 0 to the maximum value (e.g., 5.3e-4 for 1.3B) over the first 2000 steps. Warmup prevents the model from making destructively large updates before the moment estimates stabilize.
- **Stage 1 (high learning rate):** The model trains at the maximum learning rate for a portion of total training (likely roughly 1/3 of total steps, though the paper does not specify the stage boundaries).
- **Stage 2 (reduced):** Learning rate drops to `$\sqrt{1/10} \approx 0.316$` of Stage 1's rate. So for the 1.3B model, Stage 2 uses approximately `$5.3\text{e-}4 \times 0.316 \approx 1.68\text{e-}4$`.
- **Stage 3 (final):** Learning rate drops again by `$\sqrt{1/10}$` to 10% of the initial rate. For the 1.3B model, this is `$5.3\text{e-}5$`. Training continues at this rate until completion (2 trillion tokens total).

**Why this schedule over alternatives?** Compared to a cosine schedule (which smoothly decays the learning rate to near zero), the three-stage schedule maintains higher learning rates for longer, then drops them sharply. This is motivated by the DeepSeek LLM scaling laws: the model needs high learning rates early in training to escape poor local minima quickly, but needs lower learning rates later to converge precisely. The sharp drops (by `$\sqrt{0.1}$` factor) are more aggressive than cosine decay, potentially allowing the model to settle into distinct optimization basins at each stage. The final learning rate being 10% of the initial (rather than near zero as in cosine decay) means the model never fully stops learning — there is always some plasticity preserved, which may be beneficial if training on a very large, diverse corpus where later batches are not simply repeats of earlier data.

**Batch sizes and model-specific learning rates.** Table 2 shows that batch sizes and learning rates are scaled according to the model size, following the scaling laws from DeepSeek LLM. Larger models use larger batch sizes (to maintain stable gradient estimates despite larger parameter spaces) and lower learning rates (to avoid divergence, since larger models are more sensitive to parameter updates). The specific scaling law is not derived in this paper but is inherited from the companion DeepSeek LLM work. The values are: 1.3B at batch size 1024 with lr 5.3e-4; 6.7B at batch size 2304 with lr 4.2e-4; 33B at batch size 3840 with lr 3.5e-4. The total number of training tokens is 2 trillion for all models (stated in the abstract and Section 1), which means the 1.3B model takes approximately `$2 \times 10^{12} / 1024 \approx 1.95$` billion optimizer steps, while the 33B model takes `$2 \times 10^{12} / 3840 \approx 521$` million steps (roughly 3.75× fewer updates due to the larger batch size).

**Training infrastructure.** The experiments use the HAI-LLM training framework, which combines three parallelism strategies (Section 3.5):

- **Tensor parallelism (Korthikanti et al., 2023):** Individual layers' weight matrices are split across GPUs, with communication required for the forward and backward passes through each layer. This is necessary for very large layers that don't fit on a single GPU.
- **ZeRO data parallelism (Rajbhandari et al., 2020):** Optimizer states, gradients, and parameters are partitioned across GPUs rather than replicated. This dramatically reduces per-GPU memory consumption, enabling larger models without increasing GPU count.
- **PipeDream pipeline parallelism (Narayanan et al., 2019):** Different layers of the model are assigned to different GPUs, with micro-batches pipelined through to keep all GPUs busy. This reduces idle time compared to naive model parallelism.

The hardware consists of clusters with NVIDIA A100 and H800 GPUs, each node containing 8 GPUs interconnected with NVLink/NVSwitch within nodes and InfiniBand between nodes. The A100 and H800 are both high-memory (80GB) datacenter GPUs optimized for large-scale training.

---

#### Long Context Extension: Linear RoPE Scaling to 16K

Section 3.6 describes how the models' context window is extended from the default (presumably 4K tokens, based on the DeepSeek LLM architecture) to 16K tokens. This is necessary for the repository-level code processing that the paper emphasizes — a single repository with dependency-ordered files can easily exceed 4K tokens, and the model needs to attend across the entire concatenated sequence to leverage cross-file context.

**The problem with simply training on longer sequences.** Naively increasing the sequence length at training time is expensive: attention cost scales quadratically with sequence length (`$O(N^2)$`), and the RoPE position encodings, optimized for the original training length, may not generalize well to positions far beyond what was seen during the bulk of training. The model's position encodings for tokens at position 16,000 might map to angles that were never meaningfully updated during training, causing degraded attention patterns.

**The solution: linear RoPE scaling.** The paper adopts the approach proposed by Chen et al. (2023) and kaiokendev (2023):

> "we employed a linear scaling strategy, increasing the scaling factor from 1 to 4 and altering the base frequency from 10000 to 100000."

In standard RoPE, the rotation angle for position `$p$` and dimension pair `$i$` is:

$$\theta_i(p) = p \cdot 10000^{-2i/d}$$

When the sequence length is scaled by a factor `$s = 4$` (from 4K to 16K), applying the same rotation would map the new maximum position to an angle range that the model has never seen. The linear scaling strategy instead computes:

$$\theta_i(p) = \frac{p}{s} \cdot 10000^{-2i/d}$$

This compresses the position encodings: position 16,000 in the extended model receives the same rotation as position 4,000 in the original model. The intuition is that the model has learned useful relative attention patterns at the original positions, and linear scaling preserves those patterns (in relative terms) while accommodating longer sequences.

**Base frequency adjustment.** The base frequency is changed from 10,000 to 100,000. The base frequency determines the overall scale of rotation angles: higher base frequencies mean slower rotation as position increases, which means position encodings repeat less frequently. At base frequency 10,000, the longest wavelength (for `$i=0$`) is `$2\pi \cdot 10000 \approx 62,832$` positions — well beyond 4K, meaning position encodings don't repeat within the original context window. At base frequency 100,000, the longest wavelength is `$2\pi \cdot 100000 \approx 628,319$` positions, which is well beyond even the theoretical 64K maximum. This ensures that position encodings remain nearly monotonic throughout the extended context, avoiding confusing aliasing where two different positions map to nearly identical encodings.

**Training procedure for context extension.** The extended-context training uses:

> "an additional 1000 steps of training, using a batch size of 512 and a sequence length of 16K. The learning rate was maintained as in the final pre-training phase."

This is a relatively small amount of additional training — 1000 steps × 512 batch size × 16384 sequence length ≈ 8.4 billion tokens of 16K-context data, which is only 0.4% of the total 2 trillion token budget. The model has already learned the fundamentals of code generation from the bulk of training; this short adaptation phase only needs to teach the attention mechanism to handle the extended position range. Using the final pre-training phase's learning rate (10% of initial, which is a low rate) prevents catastrophic forgetting of existing capabilities during this adaptation.

**Theoretical vs. empirical context window.** The paper notes:

> "Theoretically, these modifications enable our model to process up to 64K tokens in context. However, empirical observations suggest that the model delivers its most reliable outputs within a 16K token range."

The 4× scaling factor, combined with the base frequency adjustment, means that the RoPE encodings remain well-defined up to roughly 64K (4× the original 16K extended length). However, the model was only trained on 16K sequences, so its attention patterns for positions beyond 16K are based on extrapolation rather than learned behavior. The empirical degradation beyond 16K is expected — without training data at those lengths, the model cannot learn effective attention patterns for very long-range dependencies. The 16K practical limit is the training length, and the 64K theoretical limit represents the RoPE mathematical domain, but these are not the same thing.

---

#### Instruction Tuning: From Base Model to Chat Assistant

Section 3.7 describes the fine-tuning procedure that converts DeepSeek-Coder-Base into DeepSeek-Coder-Instruct, enabling the model to follow natural language instructions for coding tasks.

**Training data format.** The instruction data uses the Alpaca Instruction format (Taori et al., 2023), which structures each example as an instruction-response pair (or a multi-turn conversation). The data is described as "helpful and impartial human instructions" — meaning it covers a range of coding tasks without biasing toward particular styles or solutions. The specific composition of this instruction data is not detailed (e.g., how many examples, what tasks are covered), which is a minor reproducibility gap.

**Dialogue delimiter.** A key implementation detail is the use of a unique delimiter token:

> "To demarcate each dialogue turn, we employed a unique delimiter token  <|EOT|>  to signify the conclusion of each segment."

This token, which appears to be a special Unicode character, serves the same role as `<|endoftext|>` or `</s>` in other models: it tells the model where one turn ends and the next begins. During inference, the model learns to generate ` <|EOT|> ` to signal that its response is complete. This is more reliable than relying on the model to know when to stop based on content alone, since code responses can be arbitrarily long and don't have a natural "end of turn" marker.

**Training hyperparameters.** The instruction tuning uses:

- **Cosine learning rate schedule** with 100 warm-up steps, starting from an initial learning rate of `$1 \times 10^{-5}$`. The cosine schedule smoothly decays the learning rate from its initial value to near zero over the course of training, following `$\eta_t = \eta_0 \cdot \frac{1}{2}(1 + \cos(\pi t / T))$`. The 100 warm-up steps ensure stable early training before the decay begins.
- **Batch size of 4M tokens** — this is a very large batch size in terms of tokens, made possible by the instruction data's relatively short sequences compared to pre-training. A large batch size provides stable gradient estimates for the fine-tuning phase.
- **Total training volume of 2B tokens** — this is 0.1% of the pre-training volume (2 trillion tokens), which is typical for instruction tuning. The model already knows how to write code; the fine-tuning only needs to teach it the conversational format and instruction-following behavior.

**Why cosine schedule instead of three-stage?** The three-stage schedule used in pre-training is designed for long-running training from scratch, where distinct optimization phases (exploration, refinement, convergence) are beneficial. Instruction tuning is a much shorter process (2B tokens vs. 2T tokens), and the model starts from a strong initialization. A smooth cosine decay is appropriate here: it maintains a moderate learning rate for most of training (preventing underfitting) while ensuring the final checkpoints are in a flat minimum near the pre-trained weights (preventing catastrophic forgetting).

**Multi-turn capability.** Figure 4 provides a concrete example of DeepSeek-Coder-Instruct 33B's behavior in a multi-turn dialogue. The user first asks the model to "Write a game snake using pygame," and the model produces a complete, runnable implementation with imports, a game loop, and collision detection. The user then asks to "Add a scoring system in the top left corner," and the model modifies its previous code — introducing a `score` variable, a `display_score` function, and integrating the scoring display into the game loop — with explanations of what was changed and why. This demonstrates that the instruction-tuned model can:

1. Generate complete, functional code from natural language descriptions
2. Maintain state across dialogue turns (it "remembers" the code it just wrote and can modify it)
3. Explain its changes in natural language, making the interaction educational rather than just transactional

The appendix (Figure 5) provides another example involving database creation and data analysis with matplotlib, further demonstrating multi-step task execution with context maintained across turns.

## 4. Key Insights and Innovations

### Innovation 1: Repository-Level Data Organization is a First-Class Training Signal, Not Just More Data

The dominant assumption in code model pre-training prior to this paper was that the fundamental unit of training data is the **individual source file**. StarCoder (Li et al., 2023) concatenated files from The Stack corpus with a fill-in-the-middle objective, but the files were essentially treated as independent documents — the model saw `sorting.py` in one training batch and `binary_search.py` in another, never learning that these files reference each other in practice. CodeLlama (Roziere et al., 2023) inherited this file-level paradigm from its LLaMA2 base, continuing training on a 500B-token code corpus that preserved the document-independence assumption. The implicit belief was that if a model sees enough code files, the cross-file structure — how imports, function calls, and class hierarchies span module boundaries — would be learned incidentally from the statistical patterns within individual files.

This paper makes a clean conceptual break from that assumption. The core move is not just "more data" or "better filtered data" (though the filtering is aggressive, reducing the corpus to 32.8% of its raw size). It is that **the dependency structure between files in a repository is itself a training signal that should be explicitly encoded in the sequence order**, not left for the model to infer from isolated snapshots. Algorithm 1's topological sort — and specifically the modified variant that handles cycles by selecting minimal-in-degree nodes rather than requiring zero-in-degree — is not merely an engineering convenience. It embodies the insight that real-world codebases have messy, cyclic import graphs, and the ordering heuristic that breaks cycles by prioritizing files with fewest remaining dependencies is a principled approximation of how a human developer would read a project: start with the foundational modules that few things depend on, then work upward.

Why this is more than an incremental improvement: prior work had explored dependency-based code understanding, but exclusively at inference time through retrieval augmentation — BM25 or neural retrieval to find relevant cross-file context and prepend it to the model's input (as in the CrossCodeEval benchmark's default setting). The DeepSeek-Coder approach bakes dependency awareness into the **model parameters** through pre-training rather than bolting it on at inference. The ablation in Table 7 ("w/o Repo Pre-training") is the critical piece of evidence: removing repository-level organization and training on file-level data causes systematic degradation across Java, TypeScript, and C# on CrossCodeEval, even when retrieval augmentation is used at test time. This is a causal demonstration that pre-training with dependency structure teaches the model something that retrieval alone cannot replicate — likely because the model learns to attend to cross-file context as an integral part of code understanding, developing internal representations that span module boundaries, rather than treating retrieved context as an external prompt prefix to be processed shallowly.

The significance extends beyond benchmark numbers. This finding reframes how the community should think about pre-training data for code models. It is not enough to feed a model terabytes of code; the **topology** of the code — how files compose into projects, how dependencies propagate, how imports create directed graphs — must be preserved and exploited. This is analogous to how natural language models benefit from preserving document structure (keeping paragraphs contiguous rather than shuffling sentences), but for code, the structure is inter-file rather than intra-file, and the cost of ignoring it (broken import graphs, dangling function calls) is qualitatively more severe than shuffling paragraphs in an essay.

### Innovation 2: Fill-in-the-Middle and Standard Generation Are Competing Objectives That Must Be Balanced, Not Optimized Separately

The field's treatment of Fill-in-the-Middle training prior to this paper was characterized by a natural but flawed intuition: more FIM training should produce better infilling models, and since infilling is just a special case of generation, a model trained entirely with FIM should be strictly better than a model trained partially with FIM. InCoder (Fried et al., 2022) trained on FIM causally, and the SantaCoder/StarCoder lineage (Allal et al., 2023; Li et al., 2023) treated FIM as a data augmentation applied uniformly. The implicit model was additive: FIM capability stacks on top of next-token prediction without interference, so the optimal FIM rate should be as high as possible given the data diversity trade-off.

The ablation in Figure 3 provides a crisp empirical counterargument that amounts to a conceptual reframing. The 100% FIM rate achieves the best infilling (HumanEval-FIM of ~0.82) but the worst generation (HumanEval-Pass@1 of ~0.13), while the 0% FIM rate achieves the reverse (generation ~0.20, infilling ~0.60). The 50% rate occupies a Pareto-optimal middle ground (~0.18 generation, ~0.80 infilling). This is not a case of diminishing returns — where adding FIM helps up to a point — but a genuine **trade-off** where FIM actively damages next-token prediction capability at high rates.

The mechanism the authors hypothesize is subtle and revealing: PSM mode rearranges the order of the original text, so the model learns a distribution where prefixes are always followed by `<|fim_hole|>` and suffixes, never directly by the code that should come next. At 100% FIM, the model's internal representations are optimized for this artificial reordering, and the standard left-to-right generation mode — which the model must use for HumanEval, MBPP, and the vast majority of real-world code generation tasks — becomes a distribution-shift problem. The model expects sentinel tokens and suffix context that are absent during standard generation, degrading its performance.

This reframes FIM training from "how much infilling capability can we add?" to "what is the optimal allocation of a shared representational budget between two competing objectives?" The 50% rate is not arbitrary — it is the empirical solution to a resource allocation problem that the field had not recognized existed. The finding that PSM outperforms MSP (Masked Span Prediction, from T5 and CodeGen2.5) adds another layer: not all infilling objectives are equal in their interference with generation. MSP masks multiple spans rather than moving a single contiguous block, and this more aggressive restructuring apparently causes more interference.

This is a fundamental rather than incremental advance because it changes the optimization problem. Future code model training should not ask "should we use FIM?" but "what FIM rate, mode, and scheduling minimizes the generation-infilling trade-off for our target deployment?" It also implies that architectural innovations that better separate generation and infilling pathways — perhaps through dedicated FIM-specific parameters or multi-task training objectives with explicit task conditioning — could push the Pareto frontier outward, enabling both capabilities at higher levels than the 50-50 compromise allows.

### Innovation 3: Data Quality Engineering, Not Scale, Is the Primary Lever for Closing the Open-Source Gap

The narrative around open-source code models prior to DeepSeek-Coder centered on scale. CodeLlama's strongest model was 34B parameters, trained on 500B tokens of code on top of LLaMA2's 2T pretraining tokens. StarCoder was 15B, trained on 1T tokens from The Stack. The implicit assumption was straightforward: to approach closed-source performance (GPT-3.5 at 64.9% HumanEval average, GPT-4 at 76.5%), open-source models needed more parameters and more data — the same recipe that had worked for general-purpose LLMs.

This paper disrupts that narrative decisively. The DeepSeek-Coder-Base 6.7B model surpasses CodeLlama-Base 34B on HumanEval (44.7% vs. 41.0% average across eight languages, Table 3) — a **5× parameter advantage nullified** by data quality improvements. The 6.7B model achieves this despite being trained from scratch on 2T tokens (comparable total compute to CodeLlama's continued pre-training, but without the benefit of LLaMA2's 2T-token general-domain initialization). On MBPP, the 6.7B model scores 60.6% vs. CodeLlama-34B's 55.2%. On DS-1000 (Table 4), the more realistic data science benchmark, the 6.7B achieves 30.5% vs. 34.3% for CodeLlama-34B — much closer than the 5× parameter difference would predict.

The enabling factor is not one thing but the combination of the data construction pipeline's stages working in concert. The aggressive filtering (discarding 67.2% of raw data), the repository-level deduplication (preserving project structure), the compiler-based quality screening (removing syntactically broken code), and the dependency-aware ordering all contribute. But the fact that the 6.7B model — which is small enough to fine-tune on a single consumer GPU with quantization — can match or approach the 34B model has practical implications that extend beyond benchmarking. It means that careful data curation can substitute for model scale, shifting the economics of code model development. Organizations without access to thousands of GPUs can still train competitive code models by investing in data quality rather than model size.

This is not a theoretical advance about how models learn — it is an empirical finding about **where the marginal return on investment lies** in code model development. The paper demonstrates that the scaling laws that govern general-purpose LLMs (where parameter count and data volume are the primary knobs) may not directly transfer to code models, where data **organization** (not just volume) and data **purity** (not just diversity) are first-order drivers of performance. This insight is reinforced by Section 5's DeepSeek-Coder-v1.5 results: starting from a general-purpose LLM and continuing pre-training on a high-quality code corpus yields gains in natural language understanding at a small cost to code performance, confirming that data composition — not just total code tokens — determines model behavior.

## 5. Experimental Analysis

### Evaluation Methodology

- **Dataset.** The primary benchmark for code generation is **HumanEval** (Chen et al., 2021), consisting of 164 hand-written Python programming problems validated with test cases, evaluated in a zero-shot setting. The Multilingual HumanEval extension (Cassano et al., 2023) adds problems in seven additional languages: C++, Java, PHP, TypeScript, C#, Bash, and JavaScript, for a total of eight languages. **MBPP** (Austin et al., 2021) provides 500 Python problems evaluated in a few-shot setting. For realistic data science workflows, **DS-1000** (Lai et al., 2023) offers 1,000 problems across seven libraries (Matplotlib, NumPy, Pandas, SciPy, Scikit-Learn, PyTorch, TensorFlow). The paper introduces a new **LeetCode Contest benchmark** comprising 180 problems collected from weekly and biweekly LeetCode contests between July 2023 and January 2024, with 100 test cases per problem to ensure coverage. For Fill-in-the-Middle evaluation, the **Single-Line Infilling benchmarks** from Allal et al. (2023) test Python, Java, and JavaScript using line exact match accuracy. Cross-file code completion uses **CrossCodeEval** (Ding et al., 2023), constructed from repositories created between March and June 2023 (after the training data cutoff of February 2023) across Python, Java, TypeScript, and C#. Program-based math reasoning uses seven benchmarks: **GSM8K**, **MATH**, **GSM-Hard**, **SVAMP**, **TabMWP**, **ASDiv**, and **MAWPS**. Natural language evaluation (Section 5) uses **MMLU**, **BBH**, **HellaSwag**, **Winogrande**, and **ARC-Challenge**.

- **Base model(s).** The DeepSeek-Coder family comprises three scales: **1.3B**, **6.7B**, and **33B** parameters, all decoder-only Transformers trained from scratch on 2 trillion tokens. A variant, **DeepSeek-Coder-v1.5 7B**, is produced by continuing pre-training from the general-purpose DeepSeek-LLM-7B-Base checkpoint on an additional 2 trillion tokens. The base models are compared against leading open-source code models: **CodeGeeX2 6B** (Zheng et al., 2023), **StarCoderBase 16B** (Li et al., 2023), and **CodeLlama-Base** at 7B, 13B, and 34B scales (Roziere et al., 2023). Closed-source baselines include **code-cushman-001** (12B, the original GitHub Copilot model), **GPT-3.5-Turbo**, and **GPT-4** (OpenAI, 2023). For instruction-tuned comparisons: **WizardCoder-V1.0 15B**, **CodeLlama-Instruct 34B**, and **Phind-CodeLlama-V2 34B**.

- **Metrics.** For HumanEval, MBPP, and DS-1000, the metric is **Pass@1** — the fraction of problems for which the model's single generated solution passes all test cases. For HumanEval, greedy search (temperature 0) is used, while MBPP uses a few-shot setting. The LeetCode Contest benchmark also uses Pass@1 with greedy decoding. For FIM evaluation, the metric is **line exact match accuracy** (Allal et al., 2023): the generated line must character-for-character match the ground-truth obscured line. CrossCodeEval reports **exact match (EM)** and **edit similarity (ES)**, where edit similarity measures the normalized character-level overlap between the generated code and the reference. For program-based math reasoning (Table 8), the metric is accuracy on the final answer derived from program execution. For natural language benchmarks (Table 10), standard accuracy metrics are used per benchmark. The HumanEval multilingual average is computed as the mean across all eight languages.

- **Baselines.** The paper compares against a comprehensive set:
  * **CodeGeeX2 6B** — second-generation multilingual code model based on ChatGLM2 architecture
  * **StarCoderBase 16B** — trained on 86 languages from The Stack, with FIM objective
  * **CodeLlama-Base** (7B, 13B, 34B) — continued pre-training from LLaMA2 on 500B code tokens
  * **code-cushman-001 12B** — OpenAI's model, original GitHub Copilot backend
  * **GPT-3.5-Turbo** and **GPT-4** — OpenAI's closed-source general-purpose models
  * For instruction-tuned comparisons: **WizardCoder-V1.0 15B**, **CodeLlama-Instruct 34B**, **Phind-CodeLlama-V2 34B**
  * For FIM tasks: **SantaCoder 1.1B**, **StarCoder 16B**, and **CodeLlama-Base** (7B, 13B)
  All baseline results (except GPT-3.5/GPT-4) are re-implemented using the same evaluation scripts and environment for fair comparison (Section 4.1).

- **Generation budget / compute accounting.** For code generation tasks, the paper uses **greedy search** (temperature 0), so the generation budget is 1 sample per problem — there is no sampling-based budget scaling. For FIM evaluation, the model receives prefix and suffix context and generates the middle segment in a single pass. For CrossCodeEval, models generate up to 50 output tokens given up to 512 tokens of cross-file context (retrieved via BM25) within a maximum sequence length of 2048 tokens. The primary "compute" axis is **model scale** (parameters) and **training data volume** (2 trillion tokens for all models), not inference-time sampling. There is no best-of-N, beam search, or revision-based test-time compute scaling in this paper — the focus is on single-pass generation quality.

- **Cross-validation / statistical protocol.** The paper does not employ cross-validation or report confidence intervals. For the LeetCode Contest benchmark, the authors explicitly caution about potential data contamination: "the possibility of data contamination cannot be entirely ruled out. We observed that the GPT-4-Turbo and DeepSeek-Coder models achieved higher scores in the LeetCode Contest held in July and August." This transparency about contamination risk is notable. For CrossCodeEval, the benchmark's construction date (March–June 2023) post-dates the training data cutoff (February 2023), providing a natural contamination control. For the decontamination of training data, n-gram filtering using 10-gram matching and shorter exact-match filtering is applied against HumanEval, MBPP, GSM8K, and MATH test sets.

### Main Quantitative Results

#### Code Generation: HumanEval Multilingual and MBPP

The headline result is that **DeepSeek-Coder-Base 33B achieves state-of-the-art open-source performance with 50.3% average accuracy across eight languages on HumanEval and 66.0% on MBPP** (Table 3). This represents a 9.3 percentage point improvement over CodeLlama-Base 34B's HumanEval average (41.0%) and an 10.8 point improvement on MBPP (55.2%). The 6.7B model already surpasses CodeLlama's 34B: 44.7% vs. 41.0% on HumanEval and 60.6% vs. 55.2% on MBPP.

The per-language breakdown on HumanEval (Table 3) reveals that DeepSeek-Coder-Base 33B's advantage is not uniform across languages. It achieves its strongest results on **C++** (58.4% vs. CodeLlama-34B's 44.7%, a +13.7 point gap), **Python** (56.1% vs. 48.2%, +7.9 points), and **JavaScript** (55.3% vs. 42.2%, +13.1 points). The gap narrows on **PHP** (44.1% vs. 41.0%, +3.1 points) and **Bash** (32.3% vs. 15.8%, +16.5 points). On **C#**, DeepSeek-Coder's 51.3% trails CodeLlama-34B's 48.7% only modestly. The largest relative gap is on Bash, where DeepSeek-Coder more than doubles CodeLlama's performance — likely reflecting the benefits of repository-level training where shell scripts often appear alongside other project files and learnable cross-file patterns emerge.

The performance gap to closed-source models narrows substantially after instruction tuning. **DeepSeek-Coder-Instruct 33B achieves 79.3% on HumanEval Python** (the primary HumanEval metric), surpassing GPT-3.5 Turbo's 76.2% by 3.1 points but still trailing GPT-4's 84.1% by 4.8 points (Table 3). Across the multilingual average, DeepSeek-Coder-Instruct 33B scores 69.2% to GPT-3.5 Turbo's 64.9% — a 4.3 point advantage — while GPT-4 maintains the lead at 76.5%. On MBPP, the 33B Instruct model scores 70.0% vs. GPT-3.5 Turbo's 70.8% (roughly tied) vs. GPT-4's 80.0%.

A striking result is the smaller models' competitiveness. **DeepSeek-Coder-Base 1.3B achieves 28.3% HumanEval average**, comparable to StarCoderBase 16B's 28.0% despite having 12.3× fewer parameters. The 6.7B model (44.7%) exceeds CodeLlama 34B (41.0%) by 3.7 points — meaning DeepSeek-Coder needs only ~20% of the parameters to outperform a model trained on the foundational LLaMA2 checkpoint plus additional code data. This efficiency advantage is the primary quantitative evidence for the paper's claim that data quality and organization (repository-level structuring) substitute for model scale.

On **MBPP** (Table 3), the same pattern holds with even larger gaps: DeepSeek-Coder-Base 1.3B scores 46.2% vs. CodeGeeX2 6B's 36.2% (10 points higher with 4.6× fewer parameters). The 6.7B at 60.6% exceeds CodeLlama 34B's 55.2%. The 33B at 66.0% establishes a clear open-source state-of-the-art.

#### DS-1000: Realistic Data Science Workflows

On DS-1000 (Table 4), which tests practical library usage across seven data science ecosystems, **DeepSeek-Coder-Base 33B achieves 40.2% average accuracy** vs. CodeLlama-Base 34B's 34.3% — a 5.9 percentage point advantage. The gap is larger than on HumanEval, suggesting that repository-level training provides disproportionate benefits for tasks requiring understanding of complex library APIs with multi-file interdependencies.

The per-library breakdown shows DeepSeek-Coder leading across all libraries:
- **Matplotlib**: 56.1% vs. 50.3% (+5.8 points)
- **NumPy**: 49.6% vs. 42.7% (+6.9 points)
- **Pandas**: 25.8% vs. 23.0% (+2.8 points, the narrowest gap)
- **PyTorch**: 36.8% vs. 25.0% (+11.8 points)
- **SciPy**: 36.8% vs. 28.3% (+8.5 points)
- **Scikit-Learn**: 40.0% vs. 33.9% (+6.1 points)
- **TensorFlow**: 46.7% vs. 40.0% (+6.7 points)

The PyTorch gap (11.8 points) is the largest relative improvement — PyTorch codebases tend to have complex multi-file module structures (models defined in separate files, training loops importing from utility modules), which aligns with the paper's claim that repository-level pre-training teaches cross-file dependency understanding. The Pandas gap (2.8 points) is the smallest, which may reflect that Pandas workflows are often self-contained in single scripts or notebooks, providing less opportunity for cross-file learning.

#### LeetCode Contest Benchmark: Competition-Level Programming

This benchmark provides the most difficult test of code generation capability, using genuine competition problems collected after the training data cutoff to minimize contamination risk. **DeepSeek-Coder-Instruct 33B achieves 27.8% overall Pass@1** (Table 5), making it "the only open-sourced model that outperforms OpenAI's GPT-3.5-Turbo in this task" (GPT-3.5 Turbo scores 23.3%). However, **GPT-4 Turbo reaches 40.6%** — a 13.2 point gap that remains wide.

The difficulty-stratified results (Easy/Medium/Hard) reveal where the advantage lies:
- **Easy (45 problems):** DeepSeek-Coder-Instruct 33B scores 57.8%, exceeding GPT-3.5 Turbo's 46.7% by 11.1 points but trailing GPT-4 Turbo's 73.3% by 15.5 points.
- **Medium (91 problems):** 22.0% vs. GPT-3.5 Turbo's 15.4% (+6.6 points) vs. GPT-4 Turbo's 31.9% (-9.9 points).
- **Hard (44 problems):** 9.1% vs. GPT-3.5 Turbo's 15.9% (-6.8 points) vs. GPT-4 Turbo's 25.0% (-15.9 points).

The pattern is revealing: DeepSeek-Coder outperforms GPT-3.5 Turbo on easy and medium problems but underperforms on hard problems, where the gap to GPT-4 Turbo is largest. This difficulty-dependent behavior mirrors the finding from the reference paper (DeepSeek-Coder's companion work) that test-time compute is most effective on easy-to-medium problems — the base model has non-trivial pass@1 on easy problems, and instruction tuning amplifies this, but genuine capability boundaries (hard problems requiring novel algorithmic insight) remain.

The paper introduces **Chain-of-Thought (CoT) prompting** as a performance enhancer: adding "You need first to write a step-by-step outline and then write the code" to the prompt improves DeepSeek-Coder-Instruct 33B from 27.8% to 28.9% overall. The benefit is concentrated on medium problems (+3.3 points, from 22.0% to 25.3%) and hard problems (+2.3 points, from 9.1% to 11.4%), while easy problems slightly decline (57.8% to 53.3%). This suggests that explicit reasoning scaffolding helps the model on problems where the initial solution path is non-obvious but hurts on easy problems where the added verbiage may introduce confusion.

The paper notes a contamination concern specific to this benchmark: both GPT-4 Turbo and DeepSeek-Coder achieved higher scores on problems from July and August 2023 (nearer the data cutoff). The authors "encourage the research community to consider the potential issue of data contamination when evaluating models in future studies using our released LeetCode data" — a commendably transparent acknowledgment.

#### Fill-in-the-Middle Code Completion

The FIM evaluation (Table 6) tests the single-line infilling capability that the 50% PSM training objective was designed to enable. **DeepSeek-Coder-Base 33B achieves a mean line exact match of 81.2% across Python, Java, and JavaScript**, compared to CodeLlama-Base 13B's 75.5% and StarCoder 16B's 69.7%.

The per-language breakdown:
- **Python:** DeepSeek-Coder 33B at 65.4% vs. CodeLlama 13B at 68.3% — CodeLlama leads slightly, which is surprising given DeepSeek-Coder's overall superiority
- **Java:** DeepSeek-Coder 33B at 86.6% vs. CodeLlama 13B at 77.6% — a 9-point gap
- **JavaScript:** DeepSeek-Coder 33B at 82.5% vs. CodeLlama 13B at 80.7% — a narrow 1.8-point gap

The Python underperformance relative to CodeLlama is an anomaly that the paper does not explain. Possible factors: DeepSeek-Coder's FIM training used a global 50% rate across all 87 languages, while Python's specific characteristics (significant whitespace, dynamic typing, decorator-heavy syntax) may benefit from different FIM configurations. CodeLlama's continued pre-training from LLaMA2 may also provide advantages in Python's natural-language-like syntax.

A notable scaling pattern: **DeepSeek-Coder-Base 1.3B (70.4%) already outperforms StarCoder 16B (69.7%) and matches CodeLlama 7B (69.7%)**, despite having a fraction of the parameters. This is the strongest evidence that FIM capability is driven more by training data organization and objective configuration than by model scale. The 6.7B model (80.7%) is competitive with the 33B model (81.2%), suggesting diminishing returns to scale for infilling — the 50% FIM rate may be the primary bottleneck at larger scales.

#### Cross-File Code Completion: Repository-Level Pre-Training Validated

CrossCodeEval (Table 7) provides the most direct test of the paper's central innovation: repository-level data organization during pre-training. The benchmark requires models to complete code that depends on definitions from other files in the same repository, with cross-file context provided via BM25 retrieval. The results compare models at the ~7B scale to isolate the effect of training methodology.

**Without retrieval augmentation**, DeepSeek-Coder-Base 6.7B achieves:
- **Python:** 9.53% EM (vs. CodeLlama 7B's 7.32%, +2.21 points)
- **Java:** 10.80% EM (vs. 9.68%, +1.12 points)
- **TypeScript:** 9.59% EM (vs. 8.19%, +1.40 points)
- **C#:** 5.26% EM (vs. 4.07%, +1.19 points)

**With retrieval augmentation** (using the official BM25 cross-file context):
- **Python:** 16.14% EM (vs. CodeLlama 7B's 13.02%, +3.12 points)
- **Java:** 17.72% EM (vs. 16.41%, +1.31 points)
- **TypeScript:** 14.03% EM (vs. 12.34%, +1.69 points)
- **C#:** 16.23% EM (vs. 13.19%, +3.04 points)

These gains are consistent across all languages but relatively modest (1–3 percentage points). The more telling result is the **ablation**: removing repository-level pre-training ("+ Retrieval w/o Repo Pre-training" in Table 7) causes systematic degradation:
- **Java:** 17.72% → 16.64% (-1.08 points)
- **TypeScript:** 14.03% → 13.23% (-0.80 points)
- **C#:** 16.23% → 14.48% (-1.75 points)
- **Python:** 16.14% → 16.02% (-0.12 points)

The Python degradation is negligible (0.12 points), while C# and Java show more substantial drops. This may reflect Python's import system being more predictable from file-local syntax (import statements unambiguously name the imported module), whereas C# and Java have more complex namespace and package resolution that benefits from explicit cross-file training. The statistical significance of these differences is not reported given the single-evaluation nature of the benchmark.

The edit similarity (ES) metrics show the same pattern: DeepSeek-Coder leads across all languages with retrieval, and removing repository pre-training reduces ES scores. However, the ES values are high across all models (60–67%), indicating that all models generate roughly plausible code even when exact matches fail. The exact match metric is more discriminative for measuring genuine cross-file understanding.

#### Program-Based Math Reasoning

Table 8 tests a capability distinct from code generation: using programming to solve mathematical problems. The PAL (Program-Aided Language Models) method prompts models to interleave natural language reasoning with code execution. **DeepSeek-Coder-Base 33B achieves a 65.8% average across seven math benchmarks**, compared to CodeLlama-Base 34B's 62.0% (+3.8 points).

The strongest gains appear on the harder benchmarks:
- **MATH:** 29.1% vs. 21.2% (+7.9 points) — a 37% relative improvement
- **GSM-Hard:** 54.1% vs. 51.8% (+2.3 points)
- **SVAMP:** 71.6% vs. 70.3% (+1.3 points)
- **TabMWP:** 75.3% vs. 69.8% (+5.5 points)
- **ASDiv:** 76.7% vs. 70.7% (+6.0 points)

The MATH benchmark gap (7.9 points) is particularly notable because MATH problems require multi-step symbolic reasoning rather than simple arithmetic — the kind of task where code generation (writing a Python script to solve the problem) benefits from understanding complex algorithmic patterns. DeepSeek-Coder's repository-level training may help here by exposing the model to more sophisticated code structures (recursive functions, dynamic programming implementations, mathematical library usage) that transfer to math problem-solving.

The 6.7B model's performance (54.7% average) already exceeds CodeLlama 13B's 52.3%, consistent with the efficiency advantage seen in code generation benchmarks.

#### DeepSeek-Coder-v1.5: Recovering Natural Language Capability

Section 5 (Table 10) compares the original DeepSeek-Coder 6.7B (trained from scratch on code) with DeepSeek-Coder-v1.5 7B (continued pre-trained from DeepSeek-LLM-7B-Base). The key trade-off:

**Code performance decreases slightly:**
- HumanEval: 44.7% → 43.2% (-1.5 points)
- MBPP: 60.6% → 60.4% (-0.2 points, essentially unchanged)

**Math reasoning improves dramatically:**
- GSM8K: 43.2% → 62.4% (+19.2 points)
- MATH: 19.2% → 24.7% (+5.5 points)

**Natural language understanding improves across all benchmarks:**
- MMLU: 36.6% → 49.1% (+12.5 points)
- BBH: 44.3% → 55.2% (+10.9 points)
- HellaSwag: 53.8% → 69.9% (+16.1 points)
- Winogrande: 57.1% → 63.8% (+6.7 points)
- ARC-Challenge: 32.5% → 47.2% (+14.7 points)

The instruction-tuned variants show the same pattern: DeepSeek-Coder-Instruct-v1.5 7B achieves 72.6% on GSM8K (vs. 62.8% for Instruct 6.7B) and 34.1% on MATH (vs. 28.6%), at a small cost to HumanEval (64.1% vs. 66.1%). This validates the paper's final claim that "the most effective code-focused Large Language Models (LLMs) are those built upon robust general LLMs" (Section 6). The trade-off is explicit: ~1.5 points of code generation capability buys ~12–16 points of natural language capability — a favorable exchange for applications requiring both code and natural language understanding.

### Ablation Studies and Robustness Checks

**FIM rate and mode ablation (Figure 3, Table 3 implicit):** The paper conducts a controlled experiment using DeepSeek-Coder-Base 1.3B on a Python-only subset, comparing four configurations: 0% FIM rate (standard next-token prediction only), 50% FIM rate in PSM mode, 100% FIM rate in PSM mode, and 50% MSP (Masked Span Prediction) rate. The key finding is the **trade-off between infilling and generation**: 100% FIM achieves the highest HumanEval-FIM score (~0.82) but the lowest HumanEval-Pass@1 (~0.13), while 50% PSM achieves nearly the same infilling (~0.80) with substantially better generation (~0.18). The 50% MSP rate underperforms 50% PSM on both metrics (~0.75 FIM, ~0.15 generation), contradicting CodeGen2.5's suggestion that MSP may enhance FIM performance. The paper attributes the 100% FIM degradation to PSM's text reordering disrupting the model's learning dynamics for standard left-to-right generation. This ablation directly justifies the core design choice of 50% PSM rate.

**Repository-level pre-training ablation (Table 7):** The paper trains a variant of DeepSeek-Coder-Base 6.7B on file-level code corpus only ("w/o Repo Pre-training") and compares it to the standard repository-level-trained model on CrossCodeEval with BM25 retrieval. The repository-level model outperforms the file-level model across Java (+1.08 points EM), TypeScript (+0.80 points), and C# (+1.75 points), with Python essentially unchanged (+0.12 points). This provides the only causal evidence that dependency-aware file ordering during pre-training — not just higher-quality data or better filtering — is responsible for cross-file completion gains. The small Python effect is unexplained but may indicate that Python's explicit import syntax reduces the benefit of seeing imports in context.

**Model scale scaling behavior (implicit in Tables 3, 4, 6):** The consistent pattern across all benchmarks is that DeepSeek-Coder's performance advantage over CodeLlama is largest at smaller model scales and narrows (relatively) at larger scales. On DS-1000 (Table 4): 1.3B underperforms CodeLlama 7B (16.2% vs. 22.1%), 6.7B outperforms CodeLlama 13B (30.5% vs. 26.8%), and 33B outperforms CodeLlama 34B (40.2% vs. 34.3%). This non-monotonic scaling suggests an interaction between model capacity and training data quality: at very small scales (1.3B), the model may lack capacity to absorb the richer structural signals from repository-level data; at moderate scales (6.7B), the data quality advantage manifests fully; at large scales (33B), the advantage persists but data quality and model scale are both contributing. The paper does not explicitly analyze this interaction.

**Language distribution effects (Table 1):** The training data is heavily skewed toward major languages (Java 18.63%, Python 15.12%, C++ 11.39%, TypeScript 7.60%, JavaScript 6.75%, C# 7.34%, PHP 7.38%), with a long tail of niche languages (93.2% of total data in the top 7 languages). The HumanEval results (Table 3) show that DeepSeek-Coder's largest advantages over CodeLlama are on C++ (+13.7 points), JavaScript (+13.1 points), and Bash (+16.5 points) — not necessarily the languages with the most training data. PHP (7.38% of data) shows the narrowest gap (+3.1 points). This suggests that training data quality and repository organization, not just volume per language, drive cross-language transfer. The paper does not conduct a controlled language-by-language ablation.

**Long context extension (Section 3.6, implicit in CrossCodeEval):** The paper extends the context window from 4K to 16K via linear RoPE scaling with base frequency adjustment, followed by 1000 steps of continued training at 16K sequence length. The empirical observation that "the model delivers its most reliable outputs within a 16K token range" despite a theoretical 64K limit is a robustness check on the context extension method: simple linear scaling works up to the trained length but degrades beyond it. CrossCodeEval uses only 2048-token sequences with 512-token cross-file context, so it does not directly test the 16K capability. The paper does not provide a quantitative ablation of context length vs. performance.

**Instruction tuning without architecture changes (Section 3.7, Table 5):** The instruction-tuned models use the same architecture as base models, with only 2B tokens of fine-tuning data. The fact that this produces competitive instruction-following (matching GPT-3.5 on HumanEval, exceeding it on multilingual average) without architectural modifications or larger instruction datasets is an implicit ablation showing that the base model's code understanding transfers efficiently to instruction-following tasks. The multi-turn examples (Figure 4, Figure 5) provide qualitative evidence of this transfer.

**Data decontamination (Section 2.4, implicit in all benchmarks):** The n-gram filtering against HumanEval, MBPP, GSM8K, and MATH test sets is described in detail (10-gram exact matching for longer strings, exact matching for 3–9 grams). This is a robustness check on benchmark validity: the paper shows that performance gains are not due to memorization of test solutions. The LeetCode Contest benchmark, collected after the training data cutoff, provides an additional contamination-free evaluation. The paper's acknowledgment of possible contamination in July–August 2023 LeetCode problems (Section 4.1) is a transparency measure rather than a robustness check per se.

**DeepSeek-Coder-v1.5 as an ablation on training initialization (Table 10):** Comparing DeepSeek-Coder 6.7B (trained from scratch on code) with DeepSeek-Coder-v1.5 7B (continued from a general LLM checkpoint) essentially ablates the importance of general-domain pre-training as an initialization for code models. The result — code performance drops slightly (~1.5 points HumanEval) while natural language performance improves dramatically (~12–16 points) — quantifies the trade-off and supports the paper's ultimate recommendation that future code models be built on general LLM foundations.

### Critical Assessment

#### Claim 1: DeepSeek-Coder achieves state-of-the-art open-source code generation, surpassing all existing open-source models.

This claim is **well-supported** by the HumanEval multilingual benchmark (Table 3), where DeepSeek-Coder-Base 33B's 50.3% average exceeds CodeLlama-Base 34B's 41.0% by 9.3 points, and the 6.7B model's 44.7% already surpasses the 34B CodeLlama. The MBPP results (Table 3) reinforce this with 66.0% vs. 55.2%. However, the claim requires qualification: "state-of-the-art" applies specifically to the set of open-source models evaluated at the time of publication. The paper does not compare against models released near-contemporaneously (e.g., other January 2024 models), and the field moves rapidly. The claim is valid for the specific model versions listed as baselines.

A more significant qualification concerns **benchmark diversity**: HumanEval and MBPP are the two most widely used code generation benchmarks, but they have known limitations. HumanEval's 164 problems are relatively simple, function-level tasks that may not represent real-world programming complexity. MBPP's few-shot setting provides additional context that inflates absolute scores. The DS-1000 benchmark (Table 4) provides a more realistic test, and DeepSeek-Coder's lead narrows there (40.2% vs. 34.3%, a 5.9-point gap vs. 9.3 points on HumanEval). On the LeetCode Contest benchmark (Table 5), the 33B Instruct model achieves 27.8% — the best open-source result, but well below GPT-4 Turbo's 40.6%. The paper would be stronger with results on additional realistic benchmarks (e.g., SWE-bench for repository-level bug fixing, or code review benchmarks), but these were likely not available at submission time.

The HumanEval "baselines re-implemented" protocol (Section 4.1) is both a strength and a concern. It ensures fair comparison by controlling for evaluation scripts and environments, but it may produce different numbers than the original papers reported (the paper does not discuss any discrepancies). This is standard practice but worth noting when comparing the paper's baseline numbers to those in the original CodeLlama or StarCoder papers.

#### Claim 2: Repository-level data construction during pre-training boosts cross-file code generation capability.

This claim is **supported with qualifications** by the CrossCodeEval results (Table 7). The ablation comparing repository-level and file-level pre-training shows gains on Java (+1.08 EM), TypeScript (+0.80 EM), and C# (+1.75 EM), but the effect on Python is negligible (+0.12 EM). The absolute EM scores remain low across all models (5–18%), meaning that exact-match cross-file completion is a hard task that all models struggle with. The repository-level training provides a small but consistent improvement on most languages.

The evidence has several limitations. First, the ablation is conducted only at the 6.7B scale — we do not know whether the benefit scales with model size (does the 33B model benefit more or less from repository-level training?) or whether it interacts with the FIM training objective. Second, the BM25 retrieval augmentation used in the evaluation confounds the measurement: it is unclear how much of the cross-file performance comes from the retrieval mechanism vs. the model's internal cross-file understanding. A pure test without retrieval would isolate the pre-training effect more cleanly, but the paper only reports the "without retrieval" results as a lower baseline (where all models score 4–10% EM). Third, the paper does not extensively analyze *how* repository-level pre-training changes model behavior — does it improve attention to cross-file context? Does it reduce hallucination of non-existent imports? Does it improve consistency between files? Qualitative examples would strengthen this claim significantly.

#### Claim 3: The 50% PSM Fill-in-the-Middle rate represents the optimal balance between code generation and infilling.

This claim is **supported for the specific configuration tested** but has limited generalizability evidence. The ablation in Figure 3 tests exactly five configurations (0%, 50%, 100% FIM in PSM; 50% MSP; and implicit SPM is not tested at all) on a single model scale (1.3B) with a Python-only data subset. The optimality of 50% is inferred from only three PSM data points (0%, 50%, 100%) — intermediate rates (e.g., 25%, 75%) are not tested, so the true optimum could lie elsewhere. The evaluation uses a single FIM benchmark (HumanEval-FIM, single-line Python) and two generation benchmarks (HumanEval, MBPP). Performance on multi-line FIM, FIM in other languages, or more realistic infilling scenarios is not tested.

The finding that PSM outperforms MSP (contradicting CodeGen2.5) is claimed as a contribution, but the comparison is limited to a single MSP configuration (50% rate) without exploring MSP rate variations or MSP mode variants. The superiority of PSM over MSP may be specific to this model architecture, this data distribution, or this training recipe — the paper does not establish generalizability.

Furthermore, the paper does not ablate the FIM sentinel token format (the specific tokens `<|fim_start|>`, `<|fim_hole|>`, `<|fim_end|>`) against alternative formats from prior work, nor the FIM span length distribution (how the split point between prefix, middle, and suffix is chosen). These implementation details could affect the generation-infilling trade-off, but their impact is unexplored.

#### Claim 4: DeepSeek-Coder-Instruct surpasses GPT-3.5 Turbo on code-related tasks.

This claim is **supported on specific benchmarks but with important caveats**. On HumanEval Python, DeepSeek-Coder-Instruct 33B scores 79.3% vs. GPT-3.5 Turbo's 76.2% — a clear win. On the multilingual HumanEval average, the margin is 69.2% vs. 64.9%. On MBPP, the scores are nearly tied (70.0% vs. 70.8%). On DS-1000, GPT-3.5 Turbo baselines are not reported, so the claim cannot be evaluated there. On the LeetCode Contest benchmark, the 33B model's 27.8% exceeds GPT-3.5 Turbo's 23.3%, though GPT-4 Turbo reaches 40.6%.

The caveats: (1) GPT-3.5 Turbo is a moving target — OpenAI updates the model periodically, and the specific version tested is not identified in the paper. (2) The evaluation uses greedy decoding for all models, which may disadvantage GPT-3.5 Turbo if its instruction tuning was optimized for non-zero temperature sampling. (3) The paper does not evaluate on non-code tasks (the DeepSeek-Coder-Instruct model was trained on code-focused instruction data, and the v1.5 comparison in Table 10 suggests the original Instruct model underperforms on natural language benchmarks like MMLU at 37.2%). The claim of "surpasses GPT-3.5 Turbo" should be understood as "surpasses on code generation benchmarks specifically, while likely underperforming on general natural language tasks."

#### Claim 5: DeepSeek-Coder's efficiency (6.7B matching or exceeding CodeLlama 34B) demonstrates that data quality substitutes for model scale.

This claim is **strongly supported** by the consistent pattern across benchmarks. On HumanEval average: 6.7B scores 44.7% vs. CodeLlama 34B's 41.0% (Table 3). On MBPP: 60.6% vs. 55.2%. On DS-1000: the 6.7B matches CodeLlama 13B (30.5% vs. 26.8%) and approaches CodeLlama 34B (34.3%). On FIM: the 6.7B at 80.7% exceeds CodeLlama 13B at 75.5% (Table 6). On CrossCodeEval: the 6.7B outperforms CodeLlama 7B across all languages (Table 7). On program-based math: 6.7B at 54.7% exceeds CodeLlama 13B at 52.3% (Table 8).

The counterargument would be that CodeLlama's 34B model was trained on only 500B code tokens (on top of LLaMA2's 2T tokens of general-domain pre-training), while DeepSeek-Coder 6.7B was trained on 2T code-heavy tokens from scratch. The total training compute is not directly comparable: DeepSeek-Coder's 2T tokens at 6.7B parameters requires roughly 2 × 10^12 × 6.7 × 10^9 × 6 ≈ 8 × 10^22 FLOPs (using the standard 6ND approximation for training FLOPs), while CodeLlama 34B's 500B code tokens (after 2T general tokens) requires roughly 5 × 10^11 × 34 × 10^9 × 6 ≈ 1 × 10^23 FLOPs for the code phase alone, plus the LLaMA2 pre-training cost. The total compute budgets may be closer than the 5× parameter ratio suggests. The paper does not report FLOPs-matched comparisons between training recipes, which would be the gold standard for demonstrating data quality efficiency.

The omission of training compute budgets and FLOPs accounting is a genuine weakness of the experimental design. Without it, the claim that data quality "substitutes for model scale" conflates two factors: data quality improvements and potentially different total training compute. The paper could have strengthened this claim by reporting training FLOPs for each model and ideally including a FLOPs-controlled comparison.

#### Missing experiments that would strengthen the paper:

- **FIM rate sweep**: testing at least 25%, 50%, 75% rates (ideally across multiple model scales) to confirm 50% as optimum rather than just better than 0% and 100%.
- **Repository-level ablation at multiple scales**: is the cross-file benefit larger or smaller at 1.3B and 33B?
- **Retrieval-free CrossCodeEval**: evaluating cross-file completion without BM25 augmentation to isolate the model's internal cross-file understanding.
- **Multi-line FIM evaluation**: the HumanEval-FIM benchmark tests only single-line infilling, but real code completion often requires multi-line insertions. The FIM training claims would be stronger with multi-line FIM results.
- **Training FLOPs reporting**: all efficiency claims about "model scale vs. data quality" would be more convincing with FLOPs totals for the DeepSeek-Coder models and their baselines.
- **Confidence intervals**: none of the benchmark results report variance or statistical significance, which is important given the small test set sizes (164 for HumanEval, 180 for LeetCode).
- **Non-Python language FIM**: the FIM evaluation (Table 6) includes Java and JavaScript, but the FIM rate ablation (Figure 3) used only Python — performance in other languages at the chosen 50% rate is assumed but not demonstrated.

#### What the experiments do demonstrate convincingly:

The experiments strongly demonstrate that careful data engineering — aggressive filtering, repository-level deduplication, compiler-based quality screening, and dependency-aware file ordering — produces a model that outperforms prior open-source code models at equivalent and smaller parameter counts on standard benchmarks. The consistent pattern across HumanEval, MBPP, DS-1000, FIM, CrossCodeEval, and math reasoning benchmarks (Tables 3–8) makes this a robust result.

The experiments also convincingly show the generation-infilling trade-off (Figure 3) and the effectiveness of the chosen 50% PSM compromise, even if the exact optimality is not proven.

The instruction tuning results (Tables 3, 5) convincingly demonstrate that the base model's code understanding transfers to instruction-following with simple fine-tuning, achieving competitive performance with closed-source alternatives on code-specific benchmarks.

The DeepSeek-Coder-v1.5 comparison (Table 10) clearly quantifies the trade-off between code-first and general-first pre-training, providing actionable guidance for practitioners choosing between the two approaches.

#### Where the experimental design falls short of the paper's ambitions:

The paper's introduction frames the contribution as closing the gap between open-source and closed-source code models, but the experiments compare only against two closed-source models (GPT-3.5 Turbo and GPT-4) and only on a subset of benchmarks (HumanEval, MBPP, LeetCode Contest). There is no closed-source comparison on DS-1000, CrossCodeEval, or FIM tasks. The paper would be stronger with more comprehensive closed-source baselines, but this is partially constrained by API access costs.

The claim that repository-level pre-training is the key innovation rests primarily on a single ablation (Table 7, "w/o Repo Pre-training") at a single model scale. More extensive analysis — qualitative examples of cross-file reasoning, attention pattern analysis, probing experiments — would substantially strengthen this claim.

The LeetCode Contest benchmark, while valuable as a new resource, has only 180 problems. The difficulty stratification (45 easy, 91 medium, 44 hard) leaves the hard subset especially small for drawing reliable conclusions. The contamination concern the authors themselves raise further complicates interpretation. This benchmark serves as a useful stress test but cannot bear heavy quantitative conclusions about hard-problem performance on its own.

## 6. Limitations and Trade-offs

### 6.1 The Repository-Level Pre-Training Claim Rests on a Single Ablation at a Single Model Scale

The paper's central methodological innovation — that dependency-ordered, repository-level pre-training improves cross-file code understanding — is supported by exactly one controlled experiment: the comparison of DeepSeek-Coder-Base 6.7B with and without repository-level pre-training on CrossCodeEval (Table 7, "+ Retrieval w/o Repo Pre-training" row). This ablation is conducted at a single model scale (6.7B parameters) and evaluated on a single benchmark with a specific retrieval augmentation (BM25 with up to 512 tokens of cross-file context).

The consequence is that the **generalizability and scaling behavior of this innovation are essentially unknown**. We cannot tell from the paper whether repository-level pre-training provides larger benefits at smaller scales (where the model has less capacity to infer cross-file structure from isolated files), at larger scales (where richer representations might better exploit dependency signals), or whether it saturates at some intermediate scale. The Python results in Table 7 add further uncertainty: the ablation shows only a 0.12 percentage point exact-match difference for Python (16.14% vs. 16.02%), suggesting that repository-level pre-training may be nearly irrelevant for Python cross-file completion, while showing larger effects for C# (1.75 points), Java (1.08 points), and TypeScript (0.80 points). This language-dependence is not explained or discussed.

The paper does not provide qualitative analysis of *how* repository-level pre-training changes model behavior — for instance, whether it improves attention to cross-file context, reduces hallucination of non-existent imports, or improves identifier resolution across file boundaries. Without such analysis, the mechanism remains a black box, and a practitioner cannot predict whether the technique will transfer to their language or codebase structure of interest.

**Mitigation status:** The paper does not attempt to address this gap. The CrossCodeEval ablation is presented as confirmatory evidence (Section 4.3) rather than as a preliminary result requiring deeper investigation. The authors do not acknowledge the single-scale, single-benchmark nature of this evidence as a limitation. Future work that varies model scale, tests additional cross-file benchmarks (e.g., repository-level bug fixing, multi-file refactoring), and conducts mechanistic analysis would substantially strengthen — or potentially qualify — the central claim.

### 6.2 Training Compute Budgets Are Not Reported, Weakening Efficiency Claims

The paper makes a prominent efficiency argument: DeepSeek-Coder-Base 6.7B matches or exceeds CodeLlama-Base 34B across multiple benchmarks (Section 4.1, Table 3), which the authors frame as evidence that data quality and organization can substitute for model scale. However, the paper **never reports the total training FLOPs for any model** — neither for the DeepSeek-Coder family nor for baseline models. This omission makes the efficiency comparison fundamentally ambiguous.

The relevant quantities are not directly comparable: DeepSeek-Coder 6.7B is trained from scratch on 2 trillion code-heavy tokens, while CodeLlama 34B is trained via continued pre-training from LLaMA2-34B on 500 billion code tokens — meaning CodeLlama benefited from LLaMA2's 2 trillion tokens of general-domain pre-training. Using the standard approximation of 6ND FLOPs per training token (where N is parameter count and D is token count), the DeepSeek-Coder 6.7B pre-training cost is approximately 6 × 6.7 × 10^9 × 2 × 10^12 ≈ 8.0 × 10^22 FLOPs. CodeLlama 34B's code-phase cost alone is approximately 6 × 34 × 10^9 × 5 × 10^11 ≈ 1.0 × 10^23 FLOPs, which is already ~25% larger than DeepSeek-Coder's total budget, plus the substantial but unreported cost of the LLaMA2 pre-training phase. The total FLOPs invested in CodeLlama 34B may substantially exceed those invested in DeepSeek-Coder 6.7B, meaning the efficiency comparison is not cleanly parameter-count-controlled.

The consequence is that **the paper's headline efficiency claim — "6.7B matches 34B" — may overstate the contribution of data quality relative to total training compute**. A more precise claim would be "DeepSeek-Coder's training recipe (code-first pre-training + repository-level organization + FIM) achieves better benchmark performance at lower parameter count, but with a total training FLOPs budget that may be comparable to or larger than the baseline when accounting for continued pre-training." This distinction matters for practitioners deciding how to allocate a fixed compute budget: if the total FLOPs to train DeepSeek-Coder 6.7B from scratch approaches the FLOPs for CodeLlama 34B (including its general-domain initialization), then the practical efficiency advantage is narrower than the parameter-count ratio suggests.

**Mitigation status:** The paper does not address this limitation. No training FLOPs totals are reported, and the implicit framing throughout Section 4 treats parameter count as the primary cost axis. The DeepSeek LLM companion paper (DeepSeek-AI, 2024) presumably provides scaling-law details that could enable FLOPs estimation, but this paper does not reference those details in the context of the efficiency comparison. The omission is particularly notable because the paper positions itself in part as an efficiency story ("smaller models can compete with larger ones through better data") but does not provide the accounting needed to verify that story.

### 6.3 Hard Competition-Level Programming Problems Remain Essentially Unsolved

The LeetCode Contest benchmark (Table 5, Section 4.1) provides the paper's most honest assessment of where DeepSeek-Coder's capabilities break down. On the **hard** difficulty subset (44 problems), DeepSeek-Coder-Instruct 33B achieves only 9.1% Pass@1 (with chain-of-thought, 11.4%). This is not just below GPT-4 Turbo's 25.0% — it represents a near-complete failure to solve problems requiring non-trivial algorithmic insight. Even on **medium** difficulty (91 problems), the 33B Instruct model's 22.0% (25.3% with CoT) trails GPT-4 Turbo's 31.9% by a substantial margin. Only on **easy** problems (45 in total) does DeepSeek-Coder exceed GPT-3.5 Turbo (57.8% vs. 46.7%), while still trailing GPT-4 Turbo (73.3%).

These results reveal a **capability boundary** that improved data organization and training objectives do not cross. The paper frames its contribution as closing the gap between open-source and closed-source models (Section 1, Abstract), but the LeetCode results show that this closure is concentrated on problems at or below the model's inherent reasoning ceiling. When problems require genuinely novel algorithmic construction — dynamic programming state definitions, non-obvious greedy choices, complex graph reductions — DeepSeek-Coder performs at a level that would be unacceptable for a primary coding assistant in competitive or interview settings.

The consequence for practitioners is that **deploying DeepSeek-Coder for challenging programming tasks requires careful expectation management**. The model excels at routine code generation (HumanEval, MBPP), handles realistic data science workflows competently (DS-1000), and manages cross-file completion with retrieval augmentation. But when presented with a problem that a skilled human programmer would find genuinely difficult — the kind that appears in the "Hard" category of LeetCode contests — the model's success rate drops to roughly 1 in 10. A developer relying on DeepSeek-Coder as a coding assistant would learn quickly that it can handle boilerplate and common patterns but cannot be trusted for algorithmic innovation.

**Mitigation status:** The paper is transparent about these numbers (Table 5 reports them without obfuscation) and acknowledges the gap to GPT-4 Turbo. However, the framing in the abstract and conclusion — "surpasses existing closed-source models like Codex and GPT-3.5" and "significantly narrowing the performance gap between OpenAI GPT-4 and open-source models" — emphasizes the relative improvement over prior open-source models rather than the absolute capability ceiling. A practitioner reading only the abstract would not learn that hard competition problems are essentially out of reach. The paper suggests chain-of-thought prompting as a partial mitigation (+2.3 points on hard problems, Table 5), but this is a modest improvement that does not change the qualitative conclusion.

### 6.4 The 16K Context Window Is Empirically Unvalidated for Cross-File Tasks

Section 3.6 describes extending the context window from 4K to 16K tokens via linear RoPE scaling, with the explicit motivation that repository-level code processing requires long contexts. The paper states that "the model underwent an additional 1000 steps of training, using a batch size of 512 and a sequence length of 16K." However, **no experiment in the paper evaluates model performance at sequence lengths approaching 16K tokens**. The CrossCodeEval benchmark (Section 4.3, Table 7) uses a maximum sequence length of 2,048 tokens and a cross-file context budget of only 512 tokens. The HumanEval, MBPP, DS-1000, and LeetCode benchmarks involve single-function or single-file generation tasks that rarely exceed a few hundred tokens of context. The FIM benchmark tests single-line infilling.

The consequence is that **the paper's headline capability — 16K context for repository-level code processing — is entirely unvalidated**. The empirical observation that "the model delivers its most reliable outputs within a 16K token range" (Section 3.6) is stated without supporting evidence. We do not know whether the model can effectively attend across 16K tokens of dependency-ordered repository files, whether its cross-file completion accuracy degrades with longer contexts, or whether the RoPE extension introduces subtle attention artifacts at positions beyond the original 4K training length.

This is a significant gap because the repository-level pre-training innovation and the context extension are logically coupled: the entire motivation for dependency-ordered file concatenation is that the model can process entire repositories in a single context window. If the model cannot effectively use 16K contexts — or if performance degrades substantially as context length increases — then the repository-level training signal is partially wasted, because the model never sees full repository sequences at the lengths required for real projects. A typical multi-file Python project can easily exceed 4K tokens (the pre-extension context limit), and many real repositories exceed 16K tokens. If the effective context window in practice is shorter than the concatenated repository length, the dependency-ordering benefit is truncated.

**Mitigation status:** The paper does not address this gap. No experiments test long-context performance, no perplexity measurements at varying context lengths are reported, and no qualitative examples of long-context code generation or completion are provided. The "1000 steps at 16K" adaptation protocol is described (Section 3.6) but its effectiveness is asserted rather than demonstrated. The CrossCodeEval configuration (2048 max tokens) actively avoids testing the long-context capability that the paper claims to provide. This limitation directly affects practitioners considering DeepSeek-Coder for repository-level tasks: they have no evidence that the 16K window is usable beyond the theoretical capability.

### 6.5 Instruction Tuning Data and Methodology Are Insufficiently Specified for Reproducibility

Section 3.7 describes the instruction tuning procedure in only a few sentences: the data uses the Alpaca Instruction format, consists of "helpful and impartial human instructions," uses a cosine learning rate schedule with 100 warm-up steps and an initial learning rate of 1e-5, a batch size of 4M tokens, and a total training volume of 2B tokens. No information is provided about the **size, composition, or sourcing of the instruction dataset**. We do not know how many instruction examples were used, what proportion cover code generation vs. code explanation vs. debugging vs. multi-turn dialogue, whether the data was human-written or model-generated, or whether it was publicly released alongside the model weights.

The consequence is that **DeepSeek-Coder-Instruct is not fully reproducible from the paper alone**. A practitioner attempting to replicate the instruction tuning procedure — perhaps to adapt it to a different base model or a domain-specific code corpus — would need to guess at the critical data composition decisions that produced the reported performance. The instruction tuning transforms the base model from a code generator (DeepSeek-Coder-Base) to a conversational coding assistant (DeepSeek-Coder-Instruct) that can handle multi-turn dialogue, explain its code, and follow refinement requests. The gap between base and instruct performance is substantial: on HumanEval Python, the 33B model improves from 56.1% (Base) to 79.3% (Instruct) — a 23.2 percentage point gain (Table 3). Without knowing the instruction data recipe, we cannot determine how much of this gain comes from the base model's latent instruction-following capability (which might transfer to other instruction datasets) vs. specific properties of the training data (which might not generalize).

This limitation is particularly consequential because the paper explicitly positions DeepSeek-Coder-Instruct as competing with closed-source instruction-tuned models (GPT-3.5 Turbo, GPT-4). Closed-source models are black boxes whose training data is unknown; open-source models are valuable precisely because they enable reproducibility and adaptation. If the instruction tuning procedure is not reproducible, the open-source advantage is partially nullified — the model weights are available, but the recipe for producing competitive instruction-tuned variants from other base models is not.

**Mitigation status:** The paper does not acknowledge this as a limitation. The instruction tuning section (3.7) is notably brief compared to the detailed descriptions of pre-training data construction (Section 2) and training objectives (Section 3.1). The Alpaca format is referenced (Taori et al., 2023), which provides a template structure, but the content and curation of the instruction data — the aspect most likely to affect performance — is unspecified. The paper states that the models are "under a permissive license that allows for both research and unrestricted commercial use" (Abstract), but the license applies to model weights, not to the training data or fine-tuning recipe needed for full reproducibility.

### 6.6 The Difficulty Estimation and Adaptive Allocation Framework from the Reference Paper Is Absent Here

The companion paper analyzed in the reference example develops a framework for **compute-optimal test-time scaling** — estimating problem difficulty and adaptively allocating inference compute between search strategies and revision depths. DeepSeek-Coder, by contrast, uses **fixed greedy decoding** for all code generation benchmarks (Section 4.1). There is no test-time search (no best-of-N, no beam search, no verifier-guided selection), no revision chains, and no difficulty-adaptive allocation. Every problem receives exactly one generation pass with temperature 0.

The consequence is that **the paper does not explore whether test-time compute scaling could close the remaining gap to GPT-4 on hard problems**. The LeetCode Contest results (Table 5) show DeepSeek-Coder-Instruct 33B at 27.8% overall vs. GPT-4 Turbo's 40.6% — a 12.8 percentage point gap. The reference paper's central finding is that test-time compute can provide gains equivalent to a much larger model on problems within the base model's capability range, with up to 4× efficiency improvements over naive best-of-N. It is entirely possible that applying compute-optimal test-time strategies to DeepSeek-Coder — beam search against a trained verifier, sequential revisions, or difficulty-conditioned allocation between parallel and sequential sampling — would narrow the gap to GPT-4 on medium-difficulty LeetCode problems without any additional pre-training. But this hypothesis is untestable from the data in the paper.

This limitation is not a flaw in DeepSeek-Coder per se — the paper's scope is pre-training and instruction tuning, not inference-time optimization — but it means the paper's headline comparison against closed-source models may **understate what the open-source model can achieve** when paired with modern test-time techniques, and simultaneously may **overstate the pre-training advantage** by attributing to data quality what could potentially be achieved by smarter decoding. A GPT-4-level model with unknown test-time optimizations (speculative decoding, verifier-guided sampling, internal chain-of-thought) is compared against DeepSeek-Coder with greedy decoding — an asymmetric comparison that conflates pre-training quality differences with inference strategy differences.

**Mitigation status:** The paper does not address this limitation, and it is not acknowledged. The evaluation protocol (greedy search, Section 4.1) is standard for code generation benchmarks, ensuring comparability with prior work that also uses greedy decoding. However, the reference paper's framework suggests that this standard protocol may systematically underestimate model capability, particularly on harder problems where additional inference compute could provide disproportionate gains. Future work combining DeepSeek-Coder's pre-training recipe with compute-optimal test-time scaling would be a natural integration of the two papers' contributions, but neither paper performs this integration.

## 7. Implications and Future Directions
- Field impact:
  - Demonstrates that open, repository-aware pretraining with balanced FIM can close much of the gap to closed models on realistic code tasks, including competitive programming. This shifts expectations for what open models can deliver and provides methods (repo-level ordering, FIM balance, RoPE scaling) other groups can adopt.
- Practical applications:
  - IDE assistants for multi-file projects (cross-file completion improved in Table 7).
  - Code review and maintenance tools that need to insert code segments precisely (FIM results in Table 6).
  - Data-science copilots with stronger library usage (DS-1000 results in Table 4).
  - Education and programming pedagogy, where step-by-step reasoning plus coding (PAL) can be scaffolded (Table 8).
- Research directions:
  - Longer and more reliable contexts beyond 16K (extend §3.6, e.g., continued pretraining at long lengths, position interpolation variants).
  - Richer repository modeling: integrate static analysis, build graphs, or typed ASTs to deepen cross-file understanding beyond regex-based dependencies (§2.2).
  - Dynamic retrieval and memory: learn to select cross-file context better than BM25; evaluate with larger retrieval budgets.
  - Objective mixtures: explore adaptive schedules that vary FIM rates across training to optimize both infilling and continuation.
  - Robust decontamination and benchmark freshness: standardized, versioned, and continuously refreshed test suites (the paper’s LeetCode benchmark is a step here; §4.1).
  - Multilingual depth: Table 3 shows strong multilingual HumanEval; extending repo-level construction to non-Python ecosystems (e.g., Java/TypeScript build systems) could further raise cross-file scores.

> In sum, DeepSeek-Coder’s core technical package—repository-ordered pretraining, balanced FIM, and long-context RoPE scaling—translates to consistent empirical gains across a wide spectrum of code tasks (Tables 3–8), with strong open-source accessibility and a clear roadmap for building even more capable repository-scale code models.
