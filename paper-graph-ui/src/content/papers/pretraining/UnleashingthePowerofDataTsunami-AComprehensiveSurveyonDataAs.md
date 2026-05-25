# Unleashing the Power of Data Tsunami: A Comprehensive Survey on Data Assessment and Selection for Instruction Tuning of Language Models

**ArXiv:** [2408.02085](https://arxiv.org/abs/2408.02085)

## 🎯 Pitch

This paper delivers the first unified, in-depth survey of data assessment and selection strategies for instruction tuning large language models, organizing the vast literature into a clear taxonomy based on data quality, diversity, and importance. By bridging abstract evaluation metrics with actionable selection mechanisms, it empowers practitioners to identify the most beneficial subsets from massive instruction datasets—boosting performance while reducing training costs. This synthesis not only clarifies the strengths and gaps in current methods, but also provides essential guidance for developing more efficient, robust, and responsible LLM training pipelines.

---

## 1. Executive Summary

This survey systematically organizes and reviews data assessment and selection methods for instruction tuning of large language models, categorizing all approaches into a unified taxonomy of **quality-based**, **diversity-based**, and **importance-based** selection (operationalized respectively as perplexity filtering, k-center greedy clustering, and gradient-based influence estimation). Analyzing representative methods on standard training sets including Alpaca, FLAN v2, and The Pile with models such as LLaMA 2 7B and Mistral 7B, the survey documents that quality-focused selection can match full-dataset performance with as little as 5–10% of the data, while hybrid quality-and-diversity approaches like DEITA and QDIT consistently outperform single-dimensional methods across benchmarks (e.g., MMLU, ARC, HellaSwag) — establishing that compound assessment strategies are necessary for optimal subset construction, though noting that the field lacks a unified definition of "good data" and that existing methods rarely integrate importance-based metrics into their selection pipelines.

## 2. Context and Motivation

### The Core Problem: We Don't Know How to Define "Good Data" for Instruction Tuning

The fundamental challenge this survey addresses is deceptively simple: **given a massive collection of instruction-response pairs for fine-tuning a language model, which ones should you actually use?** This question matters because the instruction tuning datasets available today are enormous, heterogeneous, and — critically — of wildly varying quality. The paper notes that researchers and practitioners now face a situation where "naively training a LLM on all existing instructions may not be optimal and practical" (Section 1). The sheer volume of available data (tens of thousands to millions of examples across open-source collections like Alpaca, FLAN v2, UltraChat, and OpenOrca) creates a selection problem that is both computationally expensive and methodologically underdetermined.

This gap is significant for several practical reasons the authors highlight throughout Section 1:

- **Training efficiency**: Instruction tuning a 7B-parameter model on millions of examples is computationally intensive. If 90%+ of those examples are redundant, low-quality, or unhelpful, the wasted compute is substantial. The survey documents cases where 5% of the data matches full-dataset performance (e.g., IFD on Alpaca with LLaMA, Table 2), implying that naive full-dataset training squanders resources.
- **Performance ceilings**: Poor-quality data doesn't just waste compute — it actively degrades model performance. Misaligned instruction-response pairs, vague or contradictory instructions, and noisy responses introduce supervision noise that limits what the model can learn. The paper's taxonomy of quality dimensions (Section 3) makes explicit that failing to filter for clarity, correctness, and coherence in responses creates an upper bound on achievable instruction-following capability.
- **Lack of principled guidance**: Organizations building instruction-tuned LLMs face a bewildering array of choices: Should they filter by perplexity? By GPT-4 scores? By clustering and diversity sampling? By gradient matching against a validation set? The survey reveals that each of these approaches has been studied in isolation, but no unified framework exists for understanding their relationships, tradeoffs, or complementarity.

### The Field's Fragmented Understanding of "Data Quality"

The paper is motivated by a genuine conceptual fragmentation in how the research community thinks about instruction data quality. The authors observe that "very few studies noticed that there exists no unified dimensions or aspects in measuring data 'quality' where previous works tend to put emphasis on the domain-specific and task-dependent characteristics" (Section 1.2). In other words, every research group studying data selection for instruction tuning has developed its own ad-hoc quality criteria — some emphasizing linguistic fluency, others emphasizing diversity of task coverage, still others emphasizing alignment with a target evaluation distribution — without a shared vocabulary or framework for comparing across approaches.

This fragmentation manifests in several concrete ways:

**Terminological confusion.** The paper notes that "quality, diversity, and importance might be used interchangeably without strict discrimination in previous studies" (Section 1). A method described as "quality-based" by one research group might actually be measuring what another group would call "diversity" (or vice versa). The survey addresses this directly by providing explicit, formalized definitions for each term (Section 1.2):

> "Quality refers to the intrinsic value of the data. High-quality data typically satisfy two conditions: 1) The instructions are clear, accurate, and explicit in explaining the task at hand and the expected behavior of LLMs. 2) The responses are correct, coherent, and pertinent to the instructions."

> "Diversity refers to the variety and richness of the dataset. During training, models that are exposed to datapoints under a wide range of scenarios enjoy a high level of generalization to unseen tasks."

> "Importance refers to the impact of specific data points on the LLM's performance. It implies the necessity of adding one datapoint into the selected subset for instruction tuning."

These definitions are not merely taxonomic — they serve as the organizing principle for the entire survey, structuring a diverse literature into a coherent framework. The fact that such definitions were needed (and were previously absent) underscores how fragmented the field had become.

**No shared evaluation methodology.** The authors observe in Section 7.1 that existing ablation studies on data selection methods "are often carried out by comparing the performance of LLMs fine-tuned with the selected and the full dataset." However, there is no standardized benchmark for evaluating whether a selection method actually identifies "good" data versus simply optimizing for the idiosyncrasies of a particular evaluation set. The paper documents cases where "coreset sampling methods that use losses and gradients as proxies for data quality" fail to show "positive correlation with the selection effectiveness" on downstream benchmarks — meaning the proxy metrics used during selection don't reliably predict actual model performance.

**Individual methods, not cumulative knowledge.** Each data selection paper typically proposes a new metric or algorithm and demonstrates its effectiveness on specific datasets with specific models. But very few studies "ever tried to justify their design and interpret the philosophy behind" (Section 7.2). The field has accumulated techniques — IFD scores, k-center greedy clustering, GPT-4 scoring, EL2N filtering, gradient matching — without understanding how (or whether) these techniques relate to each other, whether they capture complementary or redundant information, or how to combine them effectively.

### Where Existing Approaches Fall Short

The survey identifies specific limitations across prior work, organized along the three axes of its taxonomy:

**Quality-based methods are necessary but insufficient.** Perplexity-based filtering (Ankner et al., 2024; Section 3.2) and GPT-scoring approaches (Chen et al., 2023b; Section 3.3) can identify obviously bad examples — nonsensical responses, mismatched instruction-response pairs, ungrammatical text. But quality filtering alone "cannot detect mismatched instruction-response pairs" (Section 3.1 remark on hand-crafted indicators) and "fail[s] to guarantee the instruction-following capability of LLMs trained on highly-scored datasets." A dataset of perfectly fluent, coherent, high-quality responses that all address the *same narrow set of tasks* would pass quality filters but produce a model with poor generalization. The Alpagasus method (Chen et al., 2023b) reduced Alpaca from 52K to 9K examples based on GPT-3.5 quality scores and maintained performance — but this works because Alpaca was already relatively homogeneous. Against more diverse datasets, pure quality filtering would strip away valuable rare examples.

**Diversity-based methods ignore individual sample quality.** Geometry-based coreset sampling methods — k-center greedy (Sener & Savarese, 2017; Section 4.3), herding (Harvey & Samadi, 2014), and clustering-based approaches (Tirumala et al., 2024) — optimize for coverage of the embedding space. They ensure that the selected subset spans the semantic breadth of the full dataset. But these methods are agnostic to whether any individual datapoint is actually *good*. A diverse subset could include an equal number of high-quality and low-quality examples from each cluster, diluting the training signal. The paper notes this tension explicitly when discussing DEITA (Liu et al., 2023b; Section 4.3), which addresses it by applying quality scoring *first* and then diversity filtering — but this sequential approach means that diverse-but-low-quality examples are permanently discarded, even if they could have been useful for some purposes.

**Importance-based methods are expensive and evaluation-dependent.** Gradient-based influence estimation (LESS, Xia et al., 2024a; Section 5.4) and datamodel-based approaches (DsDm, Engstrom et al., 2024; Section 5.2) identify training examples that most strongly affect model behavior on a target evaluation set. These are the most *principled* methods — they directly optimize for what we care about — but they suffer from fundamental limitations. First, they are computationally intensive: computing per-example gradients or training datamodels at LLM scale requires significant approximation and still incurs substantial cost. Second, and more importantly, they are "highly dependent on the LLMs under development" (Section 5.4 remark), meaning the selected subset is optimal for a specific model architecture on a specific evaluation set. Transfer to a different model or a different downstream task requires recomputing everything. The paper flags this in Section 7.2: "instructions that resemble the most to the testing set or bring about performance gains are judged as 'good' data. However, such 'good' data cannot be easily transferred to another LLM of completely different architecture and parameters."

**Hybrid methods exist but are ad-hoc.** Several methods combine multiple assessment dimensions — LIFT (Xu et al., 2023b) uses both hand-crafted and GPT-based quality indicators plus diversity considerations; QDIT (Bukharin & Zhao, 2023) dynamically weights quality and diversity during greedy selection; FL (Bhatt et al., 2024) combines uncertainty-based importance with diversity sampling. But the paper observes that these combinations are "more or less ad-hoc" (Section 7.2), with the priority order among dimensions "implicitly encoded into the selection of instructions" rather than explicitly reasoned about. There is no systematic understanding of *when* to prioritize quality over diversity over importance, or how to dynamically adjust the tradeoff based on the characteristics of the dataset or the target application.

**The looming problem of scale.** The paper anticipates (Section 7.3) that synthetic data generation via powerful LLMs like GPT-4 will "proliferate cost-effectively with fine-grained control of characteristics such as difficulty and style," leading to "a surge of datapoints (e.g., tens or even hundreds millions) in the short future." At this scale, existing selection methods — many of which require computing pairwise similarities, clustering embeddings, or estimating per-example gradients — become computationally prohibitive. The survey notes that even current methods struggle: the difficulty estimation step in some approaches requires generating 2048 samples per question (referenced in the context of IFD-based methods), which is "extraordinarily expensive" and often exceeds the budget of the actual fine-tuning step.

### How This Paper Positions Itself

The survey's contribution is explicitly **organizational and synthetic**, not methodological. The authors state their goal in Section 1:

> "In this work, we aim to unify a wide array of data assessment and selection methods under the context of instruction tuning of LLMs."

The paper positions itself as filling a specific gap in the survey literature. The authors distinguish their work from four related surveys (Section 1.1):

1. **Liu et al. (2024d)** surveyed the *datasets themselves* (their statistics, sources, domains) but provided no guidance on how to select subsets from them. The present survey "emphasize[s] the selection of instruction-tuning data for the improved downstream performance."

2. **Albalak et al. (2024)** provided a comprehensive overview of data pipeline construction for language models, including utility functions and selection mechanisms, but focused primarily on pre-training corpora while "neglecting the fine-grained analysis of existing selection methods specifically designed for instruction tuning." The present survey "serves as an indispensable extension on the selection of instruction datasets."

3. **Wang et al. (2024a)** categorized dataset selection methods by the type of model used (system of indicators, trainable LLMs, powerful LLMs, small models) rather than by the *characteristics of the data themselves* that are being optimized. The present survey "stems from the characteristics of data themselves, namely quality, diversity, and importance, for categorization of selection methods."

4. **Guo et al. (2022)** and **Zhou et al. (2024b)** surveyed coreset selection in deep learning and data quality measurement tools respectively, but were not specific to instruction tuning or LLMs.

The paper's positioning is thus: while there exist surveys on data selection in general machine learning, and surveys on LLM datasets, and surveys on instruction tuning methods, **no existing survey organizes instruction tuning data selection methods into a unified taxonomy based on the fundamental properties of the data being measured**. The three-axis framework (quality, diversity, importance) is the paper's primary intellectual contribution — it provides a lens through which dozens of disparate papers can be understood as variations on common themes rather than isolated contributions.

The paper also explicitly scopes out certain topics to maintain focus. Bias and fairness in instruction data (Section 1.3) are excluded because "most data selection methods on instruction tuning do not even notice the bias or fairness of data" and "they are in lack of explicitly designing steps to reduce negative impacts of biased data." This is an honest acknowledgment of a gap in the literature that the survey chooses to flag as future work rather than attempt to retrofit into the existing framework.

Finally, the paper positions itself as forward-looking. Section 7 catalogs five open challenges: benchmarking the evaluation of selection methods themselves rather than just downstream model performance, establishing more universal definitions of "good data," scaling selection methods to massive synthetic datasets, maintaining cost-efficiency as LLMs grow larger, and integrating bias and fairness considerations. These challenges are not merely listed — they are presented with concrete suggestions for how the community might address them, making the survey not just a retrospective review but a **research agenda**.

## 3. Technical Approach

### 3.1 Reader Orientation

This paper is a **comprehensive survey** — not a method paper introducing a new technique, but a systematic taxonomy and review of existing methods for evaluating and selecting data for instruction tuning of large language models. The core idea is that the dozens of published data selection techniques, which previously appeared as isolated contributions with their own ad-hoc terminology, can be unified into three fundamental perspectives — **quality, diversity, and importance** — and that understanding the relationships, tradeoffs, and complementarity among these three axes provides both intellectual clarity and practical guidance for building better instruction-tuned LLMs.

The problem the survey solves is **conceptual fragmentation**: given an instruction tuning dataset and a budget constraint (only `$b$` out of `$N$` examples can be used), how should one evaluate each datapoint and decide which to keep? The survey shows that every existing method can be understood as implementing a scoring function `$q(x_i)$` that maps a datapoint to a number, followed by a selection mechanism `$\pi$` that converts those scores into a subset — and that the scoring function always targets some combination of quality, diversity, and importance. The "shape" of the solution is therefore a **unified mathematical framework** (Section 2, Eqs. 2–4) plus a **fine-grained taxonomy** (Sections 3–5) that organizes methods by what aspect of data they measure and how they measure it.

### 3.2 Big-Picture Architecture (Diagram in Words)

The survey's conceptual architecture has two layers. The first is a **unified mathematical abstraction** that captures what all data selection methods share. The second is a **three-branch taxonomy** that captures how they differ.

**Layer 1: The Unified Selection Framework (Section 2)**

Every data assessment and selection method can be expressed as:

1. **Input**: A full instruction tuning dataset `$\mathcal{S} = \{x_i\}_{i=1}^N$`, where each `$x_i$` is a tokenized instruction-response pair (the instruction tokens `$x_i(<t)$` are fed to the model without loss computation; the response tokens `$x_i(\ge t)$` are the target for language modeling loss).
2. **Evaluation function** `$q(\cdot)$`: A scoring rule that assigns a real number to each datapoint. The nature of `$q$` — what it measures and how it's computed — is what distinguishes methods.
3. **Selection mechanism** `$\pi$`: A procedure that takes the full dataset, the scores, and a budget `$b$`, and returns the selected subset `$\mathcal{S}_b \subset \mathcal{S}$` of size at most `$b$`.
4. **Output**: The selected subset `$\mathcal{S}_b$`, which is then used to fine-tune the LLM via standard cross-entropy loss on the response tokens:

$$\mathcal{L} = \sum_{x_i \in \mathcal{S}} \mathcal{L}_i, \quad \mathcal{L}_i = -\sum_{j=t}^{|x_i|} \log P(x_i(j) \mid x_i(<j); \theta)$$

where `$\theta$` denotes all model parameters, `$x_i(j)$` is the `$j$`-th token of datapoint `$x_i$`, and `$t$` is the index where the response part begins (the instruction part `$x_i(<t)$` is included in the conditioning context but excluded from loss computation).

**Layer 2: The Three-Axis Taxonomy (Sections 3–5)**

The survey classifies all methods into three categories based on what `$q(x_i)$` measures:

- **Quality-based** (Section 3): `$q(x_i)$` assesses the intrinsic value of `$x_i$` — whether the instruction is clear and the response is correct, coherent, and relevant. Methods range from hand-crafted linguistic indicators (readability scores, n-gram statistics) to model-based indicators (perplexity, reward model scores) to GPT-4 judging to human annotation.
- **Diversity-based** (Section 4): `$q(x_i)$` assesses how much `$x_i$` differs from other datapoints — whether adding it to the selected subset increases the variety and coverage of the dataset. Methods include lexical diversity metrics (type-token ratio, MTLD), semantic diversity metrics (k-nearest-neighbor distance, Vendi Score), and geometry-based coreset sampling (k-center greedy, herding, clustering-based selection).
- **Importance-based** (Section 5): `$q(x_i)$` assesses how much `$x_i$` contributes to the model's downstream performance — whether learning from this example actually helps the model do better on target tasks. Methods include difficulty/complexity estimation, loss- and error-based influence (forgetting score, memorization), datamodel-based prediction, and gradient-based influence (gradient matching, influence functions).

The survey further subdivides each category by **implementation style**: hand-crafted indicators (explicit formulas requiring no model training), model-based indicators (using trained proxy models or the target LLM itself), and coreset sampling (iterative optimization procedures that jointly consider multiple datapoints).

**Information Flow**

When a practitioner applies these methods:

1. They first **pre-process** the raw instruction dataset: each text sample containing instruction, input, and response is wrapped with a model-specific chat template (e.g., `<|im_start|>user\n...<|im_end|>`) and tokenized using the LLM's tokenizer. This converts human-readable text into the integer token sequences `$x_i$` that the model actually processes.
2. They **choose an assessment dimension** (or combination of dimensions) based on their goals: quality for eliminating noise, diversity for improving generalization, importance for maximizing performance on specific evaluation tasks.
3. They **compute evaluation scores** `$q(x_i)$` for each datapoint using their chosen method(s). This may involve running inference with a pre-trained LLM (perplexity), calling GPT-4 APIs (quality scoring), computing embeddings and clustering (diversity), or training small proxy models (importance).
4. They **apply a selection mechanism** `$\pi$` — either threshold-based filtering (keep all `$x_i$` where `$\tau_{\min} < q(x_i) < \tau_{\max}$`), percentile-based selection (keep top or middle `$P\%$` of scores), greedy iterative selection (repeatedly pick the best-scoring remaining example), or weighted sampling (sample with probability proportional to score).
5. They **fine-tune the LLM** on the selected subset `$\mathcal{S}_b$` using standard language modeling loss.

### 3.3 Roadmap for the Deep Dive

The detailed technical breakdown follows the survey's own organization, but I'll explain the *mechanisms* rather than just cataloguing methods:

- **First**, the unified mathematical formalism (Section 2 of the paper): the problem statement, the selection budget constraint, the two canonical selection mechanisms (greedy and probabilistic), and the pre-processing pipeline that converts raw text into tokenized datapoints. This is the shared language that makes all subsequent methods comparable.

- **Second**, quality-based methods (Section 3 of the paper): how researchers measure whether an instruction-response pair is "good" in absolute terms, organized by *who or what does the judging* — hand-crafted linguistic formulas, trained proxy models, GPT-4 as judge, and human annotators. I'll explain how each approach operationalizes the abstract quality dimensions (clarity, accuracy, explicitness for instructions; correctness, coherence, pertinence for responses).

- **Third**, diversity-based methods (Section 4 of the paper): how researchers ensure the selected subset covers the space of possible tasks and styles, organized by *what kind of diversity is measured* — lexical variety within individual examples, semantic distance between examples, geometric coverage of the embedding space, and bilevel optimization that jointly selects data and trains models.

- **Fourth**, importance-based methods (Section 5 of the paper): how researchers identify which examples actually matter for downstream performance, organized by *what signal is used to estimate importance* — difficulty and complexity of the example itself, losses and errors during training, datamodels that predict influence, and gradients that reveal which examples drive parameter updates.

- **Fifth**, I'll explain the cross-cutting design patterns that recur across categories: the distinction between individual scoring and joint selection, the role of proxy models for computational efficiency, the coupling between assessment and selection mechanisms, and the hybrid approaches that combine multiple dimensions.

### 3.4 Detailed, Sentence-Based Technical Breakdown

This is primarily a **taxonomy and review paper** whose core idea is that instruction tuning data selection methods, despite apparent diversity, can be organized into quality-, diversity-, and importance-based categories, with each category further structured by whether it uses hand-crafted indicators, model-based indicators, or coreset optimization. The survey does not propose a new method but rather provides the conceptual scaffolding for understanding and comparing existing ones.

---

#### The Unified Mathematical Formalism (Section 2)

The paper establishes a shared notation and problem formulation that applies to every selection method it reviews. This formalism is the survey's most important intellectual contribution — it converts an inchoate collection of ad-hoc techniques into instances of a single problem class.

**The datapoint representation.** Every instruction tuning example begins as raw text containing an instruction (possibly with an input field) and a response. The paper formalizes the conversion pipeline:

1. **Template wrapping**: The raw text is wrapped with a model-specific chat template containing special tokens that delineate roles. For example, the Qwen template (Figure 3) inserts `<|im_start|>system`, `<|im_start|>user`, and `<|im_start|>assistant` tokens. This step produces a textual prompt `$p_i$`.

2. **Tokenization**: The prompt `$p_i$` is tokenized using the LLM's associated tokenizer into a sequence of integer token IDs: `$x_i = [x_i(1), x_i(2), \ldots, x_i(n)]$`, where `$n$` is the total number of tokens.

3. **Train/context split**: The sequence is split at index `$t$` into two parts. The instruction part `$x_i(<t)$` — all tokens before `$t$` — serves as conditioning context for the model but is excluded from loss computation (its tokens don't contribute gradients). The response part `$x_i(\ge t)$` — all tokens from `$t$` to the end — is the target for language modeling.

**The training objective.** Given the full tokenized dataset `$\mathcal{S} = \{x_i\}_{i=1}^N$`, instruction tuning minimizes:

$$\mathcal{L} = \sum_{x_i \in \mathcal{S}} \mathcal{L}_i, \quad \mathcal{L}_i = -\sum_{j=t}^{|x_i|} \log P(x_i(j) \mid x_i(<j); \theta)$$

where `$|x_i|$` is the total token count of datapoint `$i$`, `$t$` is the index where the response begins, `$x_i(<j)$` denotes all tokens before position `$j$` (including both instruction context and previously generated response tokens), `$\theta$` represents all model parameters, and `$P(x_i(j) \mid x_i(<j); \theta)$` is the model's predicted probability for the actual token `$x_i(j)$` at position `$j$`.

**What this equation computes:** the standard autoregressive cross-entropy loss summed over all response tokens across all datapoints. At each token position `$j$` in the response, the model sees all preceding tokens (the instruction and the response so far) and must predict the next token. The loss penalizes the model when it assigns low probability to the actual token that appeared in the training data.

**Why this form:** this is the standard causal language modeling objective used throughout the LLM literature. The key design choice is that instruction tokens contribute to the *conditioning context* (they affect predictions indirectly through the model's attention mechanism) but not to the *loss* — the model is only trained to predict response tokens, not to reproduce the instruction. This prevents the model from wasting capacity on memorizing instructions and focuses learning on generating appropriate responses.

**The data selection problem.** The paper formalizes selection as:

$$\mathcal{S}_b = \pi(\mathcal{S}, b, q)$$

where `$\mathcal{S}$` is the full instruction tuning dataset, `$b$` is the maximum allowed size of the selected subset, `$q(\cdot)$` is an evaluation function that scores each datapoint, and `$\pi$` is a selection mechanism that takes the dataset, budget, and scores as inputs and outputs the selected subset.

**What this equation represents:** the selection process takes a scoring function `$q$` (which encodes all assessment logic) and a selection mechanism `$\pi$` (which encodes the strategy for converting scores into a subset) and produces `$\mathcal{S}_b$`, the subset that will actually be used for training. This decomposition separates *what makes data good* (the scoring function) from *how to assemble a good subset* (the selection mechanism).

**Why this decomposition:** it reveals that all methods in the literature share this structure, even when they appear different. A method that filters by perplexity threshold uses `$q(x_i) = \text{PPL}(x_i)$` and `$\pi$` as simple threshold filtering; a method that does k-center greedy clustering uses `$q(x_i) = \min_{x_j \in \mathcal{S}_b} d(g(x_i), g(x_j))$` (distance to nearest selected neighbor) and `$\pi$` as iterative maximization. Understanding them as instances of the same framework enables systematic comparison.

**Two canonical selection mechanisms.** The paper presents two archetypal `$\pi$` implementations that cover most methods in the literature:

**Greedy selection:**

$$\mathcal{S}_b = \pi_{\text{greedy}}(\mathcal{S}, b, q) = \arg\max_{\mathcal{S}' \subseteq \mathcal{S}, |\mathcal{S}'| \le b} \sum_{x_i \in \mathcal{S}'} q(x_i)$$

**What it computes:** iteratively, greedily selects datapoints with the highest individual scores until the budget is met. At each step, it adds the remaining datapoint with the maximum `$q(x_i)$` to `$\mathcal{S}_b$` and removes it from the candidate pool.

**Why this form:** greedy selection is computationally efficient (no combinatorial optimization) and is a natural fit when `$q(x_i)$` represents an absolute quality or importance metric. It assumes that individual datapoint scores combine additively, which ignores interactions between datapoints (adding two examples may be redundant or synergistic) but works well in practice when scores are well-calibrated.

**Probabilistic selection:**

$$\mathcal{S}_b = \pi_{\text{prob}}(\mathcal{S}, b, q) = \text{Sample}(\mathcal{S}, b, p), \quad p(x_i) = \frac{q(x_i)}{\sum_{x_j \in \mathcal{S}} q(x_j)}$$

**What it computes:** samples `$b$` datapoints without replacement from the full set, where each datapoint's selection probability is proportional to its normalized score. High-scoring datapoints are more likely (but not guaranteed) to be selected.

**Why this form:** probabilistic selection introduces randomness that can improve diversity when `$q(x_i)$` is a quality metric — even some lower-quality examples may be selected, preventing the subset from being too homogeneous. It also provides a natural way to combine scores from different assessment dimensions (by multiplying or adding their probabilities). The paper notes that this can be extended to more sophisticated sampling techniques "in accordance with the domains and tasks at hand."

**Two threshold-based filtering mechanisms.** For quality-based methods where scores represent absolute judgments, the paper describes two simpler selection approaches:

$$\mathcal{S}_b = \{x_i \mid \tau_{\min} < f(x_i) < \tau_{\max}, 1 \le i \le N\}$$

**What it computes:** keeps all datapoints whose indicator score falls within a pre-specified range `$[\tau_{\min}, \tau_{\max}]$`. This is hard filtering — a datapoint is either in or out based on fixed cutoffs.

**Why this form:** threshold-based selection is interpretable and requires no comparison between datapoints. It's natural for quality metrics with clear failure modes: discard examples with perplexity above some value, or with GPT-4 quality scores below 3 out of 5.

$$\mathcal{S}_b = \{x_i \mid P_{\min} \le \hat{F}_f(f(x_i)) \le P_{\max}, 1 \le i \le N\}$$

**What it computes:** keeps datapoints whose indicator scores fall within a specified percentile range of the empirical score distribution `$\hat{F}_f$`. For example, setting `$P_{\min}=0$` and `$P_{\max}=0.2$` selects the top 20% of datapoints by score.

**Why this form:** percentile-based selection is robust to the absolute scale of the indicator — it adapts to whatever score distribution the method produces. This is important when comparing across datasets or when the indicator's calibration is unknown. A method that reports scores on `$[0, 1]$` and one that reports on `$[0, 100]$` are treated identically under percentile selection.

---

#### Quality-Based Assessment: The Abstract Dimensions (Section 3 Introduction)

The survey decomposes quality assessment into a formal multi-dimensional structure that unifies disparate measurement approaches. Quality `$q(x_i)$` is an aggregation of instruction quality `$q_I$` and response quality `$q_R$`:

$$q(x_i) = f_q(q_I(x_i(<t)), q_R(x_i(\ge t)))$$

where `$f_q$` is an aggregation function (explicit or implicit) that combines instruction and response assessments, `$x_i(<t)$` denotes the instruction part, and `$x_i(\ge t)$` denotes the response part.

**Instruction quality further decomposes into three sub-dimensions:**

- `$q_I^C$` (clarity): how easy it is for a human or model to understand what task is being requested. Unclear instructions — those with ambiguous referents, contradictory constraints, or missing context — score low.
- `$q_I^A$` (accuracy): how well the instruction aligns with the actual intended task. An instruction might be clear but ask for the wrong thing (e.g., asking for a summary when the user needs a translation).
- `$q_I^E$` (explicitness): how precisely the instruction specifies output constraints like format, length, style, or tone. Vague instructions ("write something about dogs") score low; explicit ones ("write a 200-word paragraph about Golden Retrievers in the style of a Wikipedia article") score high.

These aggregate as: `$q_I(x_i(<t)) = g_I(q_I^C(x_i(<t)), q_I^A(x_i(<t)), q_I^E(x_i(<t)))$` where `$g_I$` is an instruction-specific aggregation function.

**Response quality decomposes into:**

- `$q_R^C$` (correctness): whether the response factually and logically satisfies the instruction's requirements.
- `$q_R^H$` (coherence): whether the response is internally consistent and logically structured, without contradictions or non-sequiturs.
- `$q_R^P$` (pertinence): whether the response directly addresses the instruction rather than digressing or providing irrelevant information.

These aggregate as: `$q_R(x_i(\ge t)) = g_R(q_R^C(x_i(<t)), q_R^H(x_i(<t)), q_R^P(x_i(<t)))$`.

**Why this decomposition matters:** the survey acknowledges that "all the mentioned quality measurement components above are only demonstrative and are not enforced explicitly in the development of existing quality-based methods." In practice, most methods implicitly measure some combination of these sub-dimensions without separating them. The formal decomposition serves an analytical purpose: it provides a vocabulary for understanding *what a particular quality metric actually captures* and *what it misses*. A perplexity-based filter primarily captures fluency (related to clarity and coherence) but says nothing about factual correctness. A GPT-4 score captures multiple dimensions simultaneously but is expensive and difficult to calibrate. The decomposition enables systematic reasoning about method complementarity and blind spots.

---

#### Quality-Based Assessment: Hand-Crafted Indicators (Section 3.1)

**Core mechanism:** Hand-crafted quality indicators use explicit, manually designed formulas — often rooted in linguistic analysis and readability research — to compute quality scores without training any model.

**The general form** for a composite hand-crafted indicator is:

$$\text{IND}_i = f(\text{IND}_1(x_i), \text{IND}_2(x_i), \text{IND}_3(x_i), \ldots, \text{IND}_M(x_i))$$

where each `$\text{IND}_m(\cdot)$` is an explicit, computable function of the datapoint's text, and `$f$` is an aggregation function (often a linear combination with manually tuned or dynamically adjusted weights).

**What this computes:** starting from the raw text of datapoint `$x_i$`, each constituent indicator `$\text{IND}_m$` extracts a numerical feature — vocabulary size, n-gram frequency, semantic similarity to other examples, or intra-sample word redundancy. The aggregation function `$f$` combines these `$M$` features into a single quality score. The paper notes that "meticulous tuning might be needed for the ultimate `$f$`" — the optimal combination weights depend on the dataset and task.

**Why this form:** hand-crafted indicators are computationally cheap (no model training or inference), interpretable (each component has a clear linguistic meaning), and grounded in decades of readability and text quality research. They work well for filtering out obviously bad examples — ungrammatical text, extreme redundancy, nonsensical word sequences — but "cannot detect mismatched instruction-response pairs and therefore fail to guarantee the instruction-following capability of LLMs trained on highly-scored datasets."

**Specific indicator families described:**

**Readability-based indicators** originate from classical text analysis. The paper catalogs three representative formulas:

1. The **Dale-Chall formula** (Chall & Dale, 1995): based on sentence length and the proportion of "difficult" words (words not on a pre-defined list of 3,000 familiar words). Longer sentences with more uncommon words score as less readable.

2. The **Flesch Reading Ease** (Flesch, 1948): computed from average sentence length and average syllables per word. Higher scores indicate easier text.

3. The **Gunning Fog Index** (Gunning, 1952): estimates the years of formal education needed to understand a text, based on average sentence length and the percentage of complex words (three or more syllables).

The paper also references more recent NLP-enabled readability features (Feng et al., 2010; François, 2010; François & Fairon, 2012) that go beyond surface statistics: syntactic complexity measures (parse tree depth, clause density), semantic cohesion metrics, and language-model-based surprisal. Some studies (François, 2010; François & Miltsakaki, 2012) validate up to 46 indicators including lexical features (word frequency, word length distributions), syntactic features (average parse tree height, number of subordinate clauses), and semantic features (latent topic coherence).

**DQI (Data Quality Index)** (Mishra et al., 2020a,b) is a more specialized composite indicator designed specifically for NLP benchmark quality assessment. It has seven components:

- **Vocabulary** analysis of the dataset's word distribution
- **Inter-sample N-gram frequency and relation**: how often N-grams repeat across different examples
- **Inter-sample semantic textual similarity (STS)**: how semantically similar different examples are (high similarity suggests redundancy)
- **Intra-sample word similarity**: how repetitive individual examples are internally
- **Intra-sample STS**: semantic coherence within a single example
- **N-Gram frequency per label**: whether certain N-grams disproportionately appear with certain labels (indicating spurious correlations)
- **Inter-split STS**: similarity between train and test splits

DQI is designed to "quantify the differences between successive benchmarks by giving high scores to generalizable samples and low scores to biased samples" — essentially measuring whether a well-trained model would truly learn the task rather than overfitting superficial patterns. Dang & Verma (2024) further decompose DQI into linguistic indicators (vocabulary, N-gram statistics) and semantic indicators (embedding-based similarities), validating their respective roles in detecting different types of data problems.

**Implementation of selection with indicators:** The paper presents two straightforward filtering approaches. Either set absolute thresholds:

$$\mathcal{S}_b = \{x_i \mid \tau_{\min} < f(x_i) < \tau_{\max}, 1 \le i \le N\}$$

where `$\tau_{\min}$` and `$\tau_{\max}$` are the lower and upper acceptable score boundaries, or use percentile-based selection:

$$\mathcal{S}_b = \{x_i \mid P_{\min} \le \hat{F}_f(f(x_i)) \le P_{\max}, 1 \le i \le N\}$$

where `$\hat{F}_f$` is the empirical cumulative distribution function of all indicator scores and `$P_{\min}$`, `$P_{\max}$` specify the percentile range to retain.

**Why thresholds matter:** the paper notes that both types of thresholds "are hyper-parameters that require task-specific fine-tuning." The choice between absolute and percentile thresholds reflects a tradeoff: absolute thresholds encode explicit quality standards (e.g., "Flesch Reading Ease must be above 60") but don't adapt to dataset-level score distributions; percentile thresholds adapt automatically but can retain low-quality examples if the overall dataset is poor. The survey does not prescribe one approach, treating it as a domain-specific engineering decision.

**Key limitation emphasized by the paper:** hand-crafted indicators operate on surface text properties. They can identify unreadable or nonsensical text but cannot evaluate whether a response *correctly answers* its instruction. A fluent, well-structured response that is factually wrong would score highly on readability metrics while being useless for instruction tuning. This limitation motivates model-based and GPT-based quality assessment, which incorporate semantic understanding.

---

#### Quality-Based Assessment: Model-Based Indicators (Section 3.2)

**Core mechanism:** Model-based indicators use trained machine learning models — either the target LLM itself or smaller proxy models — to compute quality scores. Unlike hand-crafted indicators that rely on explicit linguistic formulas, model-based indicators leverage models' learned representations of language to assess quality.

**The general form** is analogous to hand-crafted indicators but with learnable parameters:

$$\text{IND}_i = f(\text{IND}_{\theta_1}^1(x_i), \text{IND}_{\theta_2}^2(x_i), \text{IND}_{\theta_3}^3(x_i), \ldots, \text{IND}_{\theta_M}^M(x_i))$$

where `$\theta_1, \theta_2, \ldots, \theta_M$` denote the trained parameters of the models computing each sub-indicator.

**Perplexity as a quality proxy.** The most widely used model-based indicator is perplexity, defined for a datapoint `$x_i$` as:

$$\text{PPL}_{x_i} = 2^{\text{NLL}_i}, \quad \text{NLL}_i = \frac{1}{|x_i|} \sum_{j=1}^{|x_i|} -\log P(x_i(j) \mid x_i(<j); \theta)$$

where `$|x_i|$` is the total token count of datapoint `$i$`, `$x_i(j)$` is the `$j$`-th token, `$x_i(<j)$` represents all tokens before position `$j$`, `$\theta$` denotes the parameters of the language model computing the perplexity (which may be different from the model being fine-tuned), and `$P(x_i(j) \mid x_i(<j); \theta)$` is the probability the model assigns to the actual token at position `$j$`.

**What it computes:** perplexity is the exponentiated average negative log-likelihood of the datapoint's tokens under the language model. A lower perplexity means the model finds the text more predictable — it assigns higher probability to the actual sequence of tokens. Text that is highly predictable (conventional, fluent, consistent with the model's training distribution) receives low perplexity; text that is surprising, incoherent, or nonsensical receives high perplexity.

**Why this form:** perplexity has a natural interpretation as the model's uncertainty about the text — it's approximately the number of equally likely next-token choices the model effectively considers at each step. A text with perplexity 10 means the model is, on average, as uncertain as if it were choosing uniformly among 10 equally probable options at each token position. Low-perplexity text is "in-distribution" for the model and thus more likely to be well-formed and learnable.

**How perplexity is used in selection:** Ankner et al. (2024) propose using a small GPT-style reference model (MPT 125M) to compute perplexity for pruning much larger datasets. The key empirical finding is that medium-to-high perplexity samples are most valuable: "samples at the high and medium percentiles are chosen by Eq. 8 for downstream fine-tuning." Very low perplexity samples are too easy (the model already knows them well), and extremely high perplexity samples are likely noise or malformed. The authors demonstrate this using The Pile and Dolma datasets with MPT 1B models, validating that a small proxy model (125M parameters) can effectively filter data for training a 3B model — a 24× reduction in the model size needed for quality assessment.

**Instruction-Following Difficulty (IFD).** One of the most influential quality metrics in the instruction tuning literature is the IFD score (Li et al., 2023a), which measures how much the instruction helps generate the response:

$$\text{IFD}_i = \frac{\text{NLL}_i^{A|Q}}{\text{NLL}_i^A}$$

where the two constituent losses are:

$$\text{NLL}_i^{A|Q} = \frac{1}{|x_i(\ge t)|} \sum_{j=t}^{|x_i|} -\log P(x_i(j) \mid x_i(<j); \theta)$$

$$\text{NLL}_i^A = \frac{1}{|x_i(\ge t)|} \sum_{j=t}^{|x_i|} -\log P(x_i(j) \mid x_i(t \le, <j); \theta)$$

Here `$t$` is the index where the response part begins, `$x_i(<j)$` in the first equation includes both the instruction and previously generated response tokens, while `$x_i(t \le, <j)$` in the second equation includes only the response tokens (the instruction is excluded from the conditioning context). `$\theta$` denotes the parameters of the language model being used for evaluation (typically the same model that will be fine-tuned, after brief "warm-up" training on a small random subset).

**What it computes:** `$\text{NLL}_i^{A|Q}$` is the model's average negative log-likelihood on the response tokens when it can see the instruction as context. `$\text{NLL}_i^A$` is the same quantity but without the instruction — the model must predict the response tokens based only on the preceding response tokens. IFD is their ratio.

**What IFD means operationally:** if the instruction provides useful guidance, the model's perplexity on the response should be lower with the instruction than without it, giving `$\text{IFD}_i < 1$`. A value of `$\text{IFD}_i = 1$` means the instruction provides no help — the model is equally uncertain about the response whether it sees the instruction or not, suggesting either that the instruction is uninformative or the model already knows the response. A value of `$\text{IFD}_i > 1$` means the instruction actually *increases* the model's uncertainty, which the paper interprets as an indicator of misaligned, mismatched instruction-response pairs.

**Why the ratio form:** the ratio normalizes for the absolute difficulty of the response text. A response might have high perplexity simply because it contains rare words or complex syntax. By dividing by the instruction-free perplexity, IFD isolates the *marginal benefit* of the instruction specifically. This makes IFD scores comparable across datapoints with different response complexities.

**Selection with IFD:** samples with `$\text{IFD}_i > \tau_{\max} = 1$` are filtered as invalid. The lower threshold `$\tau_{\min}$` controls the quality-diversity tradeoff: a higher `$\tau_{\min}$` (closer to 1) keeps only examples where the instruction provides substantial help, potentially reducing diversity; a lower `$\tau_{\min}$` keeps more examples but may include ones with low-quality instructions. The paper reports that IFD-based selection achieves competitive or better performance with only 5-10% of the full Alpaca or WizardLM datasets (Table 2).

**The warm-up step critically matters:** the model used to compute IFD scores is first "warmed-up" by fine-tuning on a very small random subset "to learn from brief experience." Without this warm-up, the model would have no instruction-following capability whatsoever, making `$\text{NLL}_i^{A|Q}$` and `$\text{NLL}_i^A$` nearly identical for all examples (since the model can't use instructions at all), and IFD would be uninformative. The warm-up teaches the model the general concept of instruction following, after which it can distinguish helpful from unhelpful instructions.

**Perplexity-based learning complexity.** Jiang et al. (2024c) propose using the variability of perplexity across differently-regularized versions of the same model as a training-free proxy for example difficulty:

$$\tilde{S}(x_i) = \frac{1}{I} \sum_{j=1}^I \text{PPL}_{x_i;\Theta_j}$$

where `$I$` is the number of subnets (obtained by adjusting dropout rate from 10% to 90% in 10% increments, giving `$I=9$`), and `$\Theta_j$` is the `$j$`-th subnet of the original model `$\Theta$` induced by setting different dropout probabilities.

**What it computes:** for each subnet (same architecture, different dropout-induced sparsity patterns), compute the perplexity of the datapoint. Average across subnets to get `$\tilde{S}(x_i)$`. Lower `$\tilde{S}(x_i)$` means the example is "easy" — it is predicted consistently even by heavily regularized (high-dropout) subnets with reduced effective capacity.

**Why this approach:** it simulates the learning trajectory without actually training. In early training when the model has effectively lower capacity (or when using a smaller model), easy examples are learned first. In a data-poor regime (small budget `$b$`), easy examples should be prioritized because they're the ones the model can actually learn from given limited capacity and data. In a data-rich regime (large `$b$`), hard examples become more valuable because easy ones are already well-covered. This provides a principled criterion for adjusting selection strategy based on the budget.

**Reward models as quality scorers.** Both Bukharin & Zhao (2023) and Du et al. (2023) use pre-trained reward models to score instruction-response pairs:

$$R_i = r_\phi(x_i(<t), x_i(\ge t))$$

where `$r_\phi$` is a reward model (e.g., the raft model from Dong et al. 2023, or deberta-v3-large-v2 from OpenAssistant), `$x_i(<t)$` is the instruction/prompt, and `$x_i(\ge t)$` is the response. The reward model `$r_\phi$` is trained on human preference data to predict which of two responses a human would prefer; its score on a single instruction-response pair represents estimated response quality.

**Why reward models:** they directly capture human preferences for helpfulness, correctness, and harmlessness — the exact dimensions that instruction tuning aims to optimize. A reward model score is a more semantically rich quality signal than perplexity (which only measures fluency/predictability) or surface linguistic indicators.

**EL2N and memorization ranking.** Marion et al. (2023) systematically compare three model-based quality indicators. The Error l2-Norm (EL2N) (Paul et al., 2021) is:

$$\text{EL2N}_i = \frac{1}{|x_i|} \sum_{j=1}^{|x_i|} \|P(x_i(<j); \theta) - \mathbf{y}_i(j)\|_2$$

where `$P(x_i(<j); \theta) \in \mathbb{R}^{N_{\text{vocab}}}$` is the model's predicted probability distribution over the vocabulary at position `$j$` given previous tokens, and `$\mathbf{y}_i(j) \in \mathbb{R}^{N_{\text{vocab}}}$` is the one-hot ground-truth vector (all zeros except a 1 at the index of the actual token `$x_i(j)$`). `$N_{\text{vocab}}$` is the vocabulary size.

**What EL2N computes:** the Euclidean distance between the model's predicted probability distribution and the ground-truth one-hot vector at each token position, averaged over the sequence. High EL2N means the model's predictions are far from the correct answer — the example is difficult or the model hasn't learned it yet. Low EL2N means the model confidently assigns high probability to the correct token.

**Why the l2 distance:** it captures both confidence and accuracy simultaneously. A model that assigns probability 0.9 to the correct token has EL2N ≈ 0.46 (the distance from `[0.9, 0.05, 0.05]` to `[1, 0, 0]`). A model that assigns 0.3 to the correct token and spreads the rest has EL2N ≈ 0.84 — much larger. EL2N is more informative than just top-1 accuracy because it captures the model's certainty.

**Memorization ranking** is defined as:

$$\text{MEM}_i = \frac{1}{N_{\text{win}}} \sum_{j=1}^{N_{\text{win}}} \mathbb{1}(\hat{x}_i(M_{\text{offset}} + j) = x_i(M_{\text{offset}} + j))$$

where `$N_{\text{win}}$` is the length of a consecutive token window, `$M_{\text{offset}}$` is a starting offset into the sequence, `$\hat{x}_i(M_{\text{offset}} + j)$` is the model-generated token at position `$M_{\text{offset}} + j$` (given all preceding tokens as context), and `$x_i(M_{\text{offset}} + j)$` is the ground-truth token at that position. The indicator `$\mathbb{1}(\cdot)$` is 1 if the generated and ground-truth tokens match, 0 otherwise.

**What MEM computes:** the fraction of tokens in a specific window of the sequence that the model can regenerate exactly when prompted with the preceding context. High MEM means the model has memorized this exact sequence — it can reproduce it verbatim.

**Why memorization matters for quality:** Biderman et al. (2024) showed that highly memorized examples are often atypical outliers or contaminated test-set examples. Filtering them out can improve generalization by removing spurious memorization patterns and focusing training on genuinely learnable regularities.

**The AFLite method** (Le Bras et al., 2020) for adversarial filtering deserves special attention because it appears across multiple cited works. The procedure:

1. Randomly partition all datapoints into training and validation sets.
2. Train a model (linear classifier or small language model) on the training partition.
3. Evaluate the model on the validation partition, recording whether each validation example was correctly predicted.
4. Repeat `$m$` times with different random partitions.
5. Compute the **predictability score** for each datapoint:

$$\text{PRED}_i = \frac{|\{\hat{x}_i \in \mathcal{E}_i \text{ s.t. } \hat{x}_i = x_i\}|}{|\mathcal{E}_i|}, \quad \mathcal{E}_i = \{\hat{x}_i^{\theta_1}, \hat{x}_i^{\theta_2}, \ldots, \hat{x}_i^{\theta_m}\}$$

where `$\hat{x}_i^{\theta_j}$` is the response generated for datapoint `$i$` by the model trained in iteration `$j$` (with parameters `$\theta_j$`), `$x_i$` is the ground-truth response, and the condition checks whether the generated response exactly matches the ground truth.

**Key detail:** datapoint `$x_i$` is never included in the training set for any iteration where it serves as a validation example. This ensures the predictability score measures how learnable the pattern is from *other* examples, not whether the model can memorize the specific instance.

**What PRED computes:** the fraction of held-out iterations where a model trained on other data could correctly reproduce the response for this datapoint. High PRED means the response is highly predictable from the general patterns in the data — it represents the "easy," learnable case. Low PRED means the response is idiosyncratic or noisy. AFLite filters by keeping only examples with predictability above a threshold, then optionally deletes the bottom `$k$` examples by DQI score.

**Uncertainty-based indicators** (Bhatt et al., 2024) draw from the active learning literature. The paper presents four variants:

$$\text{U}_i^{\text{entropy}} = \frac{1}{|x_i|} \sum_{j=1}^{|x_i|} P(x_i(j) \mid x_i(<j); \theta) \cdot \log P(x_i(j) \mid x_i(<j); \theta)$$

**What this computes:** the average entropy of the model's predictive distribution over tokens. High entropy means the model is uncertain — it spreads probability mass across many possible tokens. Low entropy means the model is confident about what comes next.

$$\text{U}_i^{\text{confidence}} = -\prod_{j=1}^{|x_i|} P(x_i(j) \mid x_i(<j); \theta)$$

**What this computes:** negative product of predicted probabilities for the actual tokens (the sequence-level likelihood). Higher values (closer to 0) mean the model is less confident in the overall sequence.

$$\text{U}_i^{\text{margin}} = -\frac{1}{|x_i|} \sum_{j=1}^{|x_i|} (\beta_1(P(x_i(<j); \theta)) - \beta_2(P(x_i(<j); \theta)))$$

where `$\beta_1$` and `$\beta_2$` denote the largest and second-largest elements of the probability vector `$P(x_i(<j); \theta) \in \mathbb{R}^{N_{\text{vocab}}}$`. This is the average margin between the top two predicted tokens.

$$\text{U}_i^{\text{min-margin}} = -\min_{j \in \{1,\ldots,|x_i|\}} (\beta_1(P(x_i(<j); \theta)) - \beta_2(P(x_i(<j); \theta)))$$

**What this computes:** the minimum (most uncertain) margin across all token positions. It identifies examples where there's at least one position where the model is torn between two viable continuations.

**Important negative result:** Wu et al. (2023) found that these uncertainty-based sampling methods "perform worse than random sampling" on several instruction tuning datasets (Databricks-Dolly, SelfInstruct-Davinci, SelfInstruct-GPT4). This suggests that for instruction tuning specifically, model uncertainty is not a reliable quality signal — uncertain examples may be genuinely ambiguous or poorly written rather than informatively difficult.

**The self-guided data selection approach** (Li et al., 2023a) that introduced IFD also established a paradigm for using the target model itself as the quality evaluator. The key steps:

1. **Warm-up:** fine-tune the pre-trained LLM on a very small random subset (e.g., 100 examples) to give it basic instruction-following capability.
2. **Evaluation:** use the warmed-up model to compute IFD scores for all remaining datapoints.
3. **Selection:** apply threshold-based filtering on IFD scores (Eqs. 7, 8).
4. **Final training:** fine-tune on the selected subset.

**Why this self-guided approach:** it avoids the need for external quality judges (GPT-4, human annotators) or separately trained proxy models. The model that will be fine-tuned is also the model that selects its own training data, creating a closed loop. However, it requires initial warm-up, which the paper notes is "brief" and uses negligible compute relative to the full fine-tuning.

**Validation with small proxy models.** Li et al. (2024b) demonstrate a crucial efficiency result: "both the perplexity and IFD scores inferred from a rather small GPT2-125M are indicative in selecting high-quality datapoints for training LLaMA2-7B and LLaMA2-13B." This means a 125M-parameter model (roughly 1.8% the size of LLaMA2-7B) can effectively filter data for much larger models, dramatically reducing the computational cost of quality assessment. This finding is practically important because it suggests the quality signal captured by perplexity and IFD is model-size-invariant enough that proxy models are reliable.

---

#### Quality-Based Assessment: GPT Score (Section 3.3)

**Core mechanism:** Closed-source LLMs like GPT-3.5 and GPT-4 are used as automated quality judges by prompting them to rate instruction-response pairs on defined criteria. The approach leverages the empirical finding that "powerful language models like ChatGPT highly align with human preference on judging the quality of instructions and responses" (Zheng et al., 2024).

**The formal process:**

$$\text{GPTScore}_i = \mathcal{G}(I_i, p_G)$$

where `$I_i$` is the raw text of the instruction-response pair (before tokenization), `$p_G$` is a carefully designed prompt template that defines the scoring task and grading criteria with output format constraints, and `$\mathcal{G}(\cdot, \cdot)$` represents the end-to-end process of sending the prompt to GPT, receiving the response, and parsing the score from the output.

**The prompt template structure** (Figure 4 of the paper):

```
We would like to request your feedback on the performance of AI assistant
in response to the instruction and the given input displayed following.
Instruction: <instruction>
Input: <input>
Response: <response>
Please rate according to the <dimension> of the response to the instruction
and the input. Each assistant receives a score on a scale of 0 to 5, where
a higher score indicates higher level of the <dimension>.
Please first output a single line containing the value indicating the scores.
In the subsequent line, please provide a comprehensive explanation of your
evaluation, avoiding any potential bias.
```

The template is parameterized by `<dimension>` (e.g., helpfulness, accuracy, clarity, explanation quality) and instantiated with the specific instruction, input, and response text.

**Why this template design:** the prompt includes several deliberate features. The scale (0 to 5) provides fine-grained discrimination without being so granular as to introduce annotation noise. The requirement to output the numeric score on a separate first line enables reliable automatic parsing. The request for a comprehensive explanation serves both as a quality check (if the explanation contradicts the score, the rating may be unreliable) and as a mechanism to improve scoring quality (forcing the model to justify its rating reduces hasty or inconsistent judgments).

**Key finding on relative vs. absolute scoring:** Liu et al. (2023b) argue that "the direct scoring of GPT4 on one single instruction sample is not well-calibrated and instead gives relative ranking of multiple instruction variants at once." In other words, asking GPT-4 to score individual examples produces scores that drift depending on the order of presentation, the recent scoring history, and other context effects. Pairwise comparison — presenting two instruction-response pairs and asking which is better — yields more reliable and consistent rankings. QuRator (Wettig et al., 2024) exploits this by collecting pairwise quality comparisons via GPT-3.5 and using them to fine-tune a small model (Sheared-LLaMA 1.3B) with a DPO-like objective, effectively distilling GPT's pairwise judgment into an efficient scoring model.

**The Alpagasus approach** (Chen et al., 2023b) uses GPT-3.5 to score each Alpaca datapoint on a 0-5 scale for helpfulness and accuracy, then filters to keep only highly-scored examples. The result: 9K examples from the original 52K (17% retention) train a model that matches or exceeds full-dataset performance on benchmarks including BBH, DROP, HumanEval, and MMLU (Table 2). This is the simplest possible quality-based pipeline and its effectiveness was surprising to the community.

**BSDetector** (Chen & Mueller, 2024) goes beyond simple scoring by additionally estimating GPT's confidence in its quality judgments. It uses both self-consistency (generate multiple evaluations and check agreement) and direct confidence elicitation. Only datapoints where GPT is highly confident in its quality assessment are retained; low-confidence datapoints are automatically corrected by GPT rather than discarded. This transforms quality filtering from a binary accept/reject decision into a quality improvement pipeline.

**The cost-quality tradeoff in GPT scoring:** the paper notes (Section 3.3 remark) that "it would be cost-efficient to collect few (e.g., <100K) GPT-scored samples first and then fine-tune an open-source LLM for quality measurement on massive corpus." The GPT API cost for scoring hundreds of thousands of examples is substantial (though far less than human annotation), but a smaller open-source model fine-tuned on these GPT scores can then be applied to millions of examples essentially for free. This distillation approach makes GPT-quality-scoring scalable.

**Multi-dimensional vs. holistic scoring:** Xu et al. (2023b) use GPT-4 to evaluate instruction datasets across four dimensions (accuracy, explanation, clarity, and difficulty) producing weighted composite scores. The paper notes that decomposing quality into dimensions can improve scoring reliability because GPT-4 can focus on one aspect at a time, but it increases the number of API calls (one per dimension per datapoint) by a factor equal to the number of dimensions.

---

#### Quality-Based Assessment: Human Evaluation (Section 3.4)

**Core mechanism:** Human annotators provide quality judgments following detailed guidelines, typically rating on multiple fine-grained dimensions on Likert scales. This is the gold standard for quality assessment but is expensive and slow, limiting its direct applicability to large-scale selection.

**The formal decomposition:**

$$\text{LabelScore}_i = f(\text{LabelScore}_1(x_i), \text{LabelScore}_2(x_i), \ldots, \text{LabelScore}_M(x_i))$$

where `$\text{LabelScore}_m(x_i)$` is the human-provided score for datapoint `$i$` on the `$m$`-th quality dimension, and `$f$` is typically summation or averaging. Scores may be boolean (acceptable/unacceptable) or integer-valued (e.g., 0 to 5 Likert scale).

**The OpenAssistant annotation framework** (Köpf et al., 2024) is the most prominent example. Human annotators evaluate each instruction-response pair along three axes:

1. **Spam detection:** binary judgment of whether the content is genuine or spam.
2. **Guideline adherence:** whether the response follows the specified annotation guidelines (formatting, content restrictions, safety policies).
3. **Quality:** a five-point Likert scale across multiple sub-aspects including creativity, humorousness, politeness, and harmlessness.

**Why this multi-dimensional approach:** different downstream applications prioritize different quality aspects. A creative writing assistant needs high creativity scores; a customer service bot needs high politeness and accuracy; a medical QA system needs high factual correctness and safety. By recording scores on individual dimensions rather than a single holistic quality rating, the dataset can be filtered differently for different use cases.

**The LIMA approach** (Zhou et al., 2024a) takes human quality control to an extreme: rather than filtering a large dataset, they have human annotators *create* a small, carefully curated dataset of 1,000 high-quality examples (750 selected from existing sources, 250 written from scratch). Simple hand-crafted indicators (text length) provide an initial coarse filter, but the final selection and composition is entirely human-driven. The remarkable result — that 1,000 hand-picked examples can produce strong instruction-following performance — motivated much of the subsequent research on quality-based data selection.

**The Dolly dataset** (Conover et al., 2023) represents a middle ground: 15,000 human-generated instruction-response pairs produced by Databricks employees. Despite the emphasis on quality during creation, subsequent analysis (He et al., 2024) found that "imperfect samples still exist" including "low-quality and inaccurate responses, incomplete and vague instructions, problematic texts with toxic language and grammar errors." This highlights that human annotation at scale inevitably introduces noise, and even human-created datasets benefit from quality filtering.

**Key limitation of human evaluation for selection:** the paper notes that "to reduce the inter-annotator inconsistency, detailed guidelines should be prepared for quality measurement." Human judgments of response quality are subjective and vary across annotators, cultures, and time. The OpenAssistant project employs multiple annotators per example and uses aggregation to reduce individual bias, but this multiplies the already-high cost. For selection at the scale of millions of examples, direct human evaluation of every datapoint is infeasible; instead, human judgments are used to train or calibrate automated quality metrics.

**The role of human evaluation in the quality pipeline:** the paper positions human evaluation not as a standalone selection method but as the ultimate reference against which other quality metrics are validated. When GPT-4 scoring or model-based indicators are shown to correlate with human judgments, they gain credibility as proxies. The claim that "GPT scores can be provided for reference during evaluation and selection" reflects this hierarchical relationship: GPT scores approximate human judgment and can be used when human evaluation is too expensive.

---

#### Diversity-Based Assessment: Hand-Crafted Indicators (Section 4.1)

**Core mechanism:** Diversity assessment measures how much variety exists within a single datapoint (lexical richness) and how different datapoints are from each other (semantic spread). Hand-crafted diversity indicators use explicit mathematical formulas to quantify these properties from token sequences and embedding representations.

**The unified formulation:**

$$q(x_i) = f_d(q_L(x_i), q_S(x_i))$$

where `$q_L$` measures lexical diversity (variety of vocabulary within the datapoint), `$q_S$` measures semantic diversity (how different the datapoint's meaning is from others), and `$f_d$` aggregates these into an overall diversity score. "Typically, `$q_L$` often investigates the diversity of n-grams, tokens, words, and sequences. Complementarily, `$q_S$` emphasizes semantic diversity that the variety of representations of the selected datapoints should be maximized in the embedding space."

**Lexical diversity: Type-Token Ratio (TTR) and its descendants.** The foundational lexical diversity metric is:

$$\text{TTR}_i = \frac{|\text{Unique}(x_i)|}{|x_i|}$$

where `$|\text{Unique}(x_i)|$` is the number of distinct token types in datapoint `$i$`, and `$|x_i|$` is the total token count. A TTR of 1.0 means every token is unique (maximum lexical variety); a TTR approaching 0 means heavy repetition.

**Why TTR is problematic:** it's sensitive to text length. Short texts naturally have higher TTR because there are fewer opportunities for repetition. Two texts of different lengths with the same underlying lexical diversity will have different TTRs, making cross-datapoint comparison unreliable. This is "the sensitivity of indicators to text length" — the first of two fundamental problems identified by Bestgen (2023).

**vocd-D** (Malvern & Richards, 1997; Malvern et al., 2004) addresses the length sensitivity through a curve-fitting approach:

$$\text{vocd-D}_i = D$$

where `$D$` is the single parameter estimated by fitting the theoretical curve:

$$\widehat{\text{TTR}}_i^k = \frac{D}{k}\left[\left(1 + 2\frac{k}{D}\right)^{\frac{1}{2}} - 1\right]$$

to empirical TTR values `$\text{TTR}_i^k$` computed on random sub-sequences of varying lengths `$k$`:

$$\text{TTR}_i^k = \frac{|\text{Unique}(x_i(j \le, <j+k))|}{|x_i(j \le, <j+k)|}, \quad 1 \le j \le |x_i| - k$$

**What the procedure does step by step:**

1. For each candidate length `$k$` (typically 10, 20, 30, ..., up to a maximum), randomly sample multiple contiguous sub-sequences of length `$k$` from `$x_i$`.
2. For each sub-sequence, compute its TTR.
3. Average the TTRs across all sub-sequences of that length to get `$\text{TTR}_i^k$`.
4. This produces a curve of `$\text{TTR}_i^k$` versus `$k$`.
5. Fit the theoretical model `$\widehat{\text{TTR}}_i^k$` to this empirical curve by optimizing `$D$`.
6. The resulting `$D$` is vocd-D.

**Why this works:** the theoretical model captures how TTR should decrease with sequence length for a text with a given underlying diversity `$D$`. By fitting the model, we extract a length-independent diversity parameter. A larger `$D$` means the text maintains higher TTR even at longer sequence lengths — it is genuinely more lexically diverse. The random sub-sampling ensures robustness to where the sub-sequences start in the text.

**MTLD (Measure of Textual Lexical Diversity)** (McCarthy & Jarvis, 2010) takes a different approach. Rather than curve-fitting, it partitions the text into contiguous segments that each maintain a target TTR:

$$\text{MTLD}_i = \frac{1}{M} \sum_{m=1}^M |x_i^m|$$

where `$M$` is the number of segments the text is partitioned into, and `$|x_i^m|$` is the length of the `$m$`-th segment. The partitioning rule is forward-sequential: start from the beginning of the text, add tokens one by one, and compute the cumulative TTR. When the cumulative TTR drops below a pre-specified threshold (the paper doesn't specify the exact threshold but references McCarthy & Jarvis 2010, who recommend TTR = 0.72), that's the end of one segment. Start a new segment from the next token. Repeat until the text is exhausted.

**What MTLD measures:** the average length of text segments needed before lexical repetition drives the TTR below the threshold. A text with high lexical diversity can go longer before the TTR drops, so it has longer segments and higher MTLD.

**HD-D (Hypergeometric Distribution Diversity)** (Jarvis, 2013) models type occurrence as draws from a hypergeometric distribution:

$$\text{HD-D}_i = \sum_{t=1}^{|\text{Unique}(x_i)|} \frac{1}{M} \sum_{m=1}^M \mathbb{1}(x_i^m(n) = u_t), \quad 1 \le n \le |x_i^m|$$

where `$u_t \in \text{Unique}(x_i)$` iterates over each distinct token type in the datapoint, `$x_i^m$` is the `$m$`-th randomly sampled sub-sequence of length `$k$`, and `$\mathbb{1}(x_i^m(n) = u_t)$` is 1 if the `$n$`-th token of sub-sequence `$m$` matches type `$u_t$`. This counts, for each type, the probability of observing that type in a random sub-sequence, averaged over `$M$` samples of size `$k$`.

**Why the hypergeometric distribution:** it models sampling without replacement from a finite population — exactly the scenario of drawing a sub-sequence from a text. The HD-D score estimates how many types we'd expect to find in a random sub-sequence, which is a length-normalized measure of lexical richness.

**Simplified n-gram diversity metrics** from the dialogue generation literature: Li et al. (2015) propose distinct-1 and distinct-2:

$$\text{distinct-}n_i = \frac{|\text{Unique n-grams}(x_i)|}{|\text{Total n-grams}(x_i)|}$$

where `$n = 1$` counts unigrams and `$n = 2$` counts bigrams. This is exactly TTR generalized to n-grams — it measures what fraction of the n-grams in the sequence are unique. These metrics are widely used in NLG evaluation (Cao & Clark, 2017; Zhu et al., 2018; Shu et al., 2019; Tevet & Berant, 2020) and inherit both the simplicity of TTR and its length sensitivity.

**Semantic diversity: k-Nearest Neighbor distance.** Moving beyond surface lexical statistics, semantic diversity measures how different datapoints are in embedding space:

$$\text{kNN}_i^j = d(g(x_i), g(\mathcal{N}_j(x_i)))$$

where `$g(\cdot)$` is a sentence encoder (typically Sentence-BERT, Reimers & Gurevych 2019) that maps text to a fixed-dimensional embedding vector `$g(x_i) \in \mathbb{R}^H$`, `$\mathcal{N}_j(x_i)$` is the `$j$`-th nearest neighbor of `$x_i$` in the embedding space (according to the distance metric `$d$`), and `$d(\cdot, \cdot)$` is typically Euclidean distance, cosine distance, or Jaccard coefficient distance.

**What kNN distance means:** a datapoint with a large distance to even its closest neighbor (`$j=1$`) is semantically unique — there's nothing else like it in the dataset. A datapoint with very close neighbors is redundant — other examples already cover its semantic territory. Selection for diversity prioritizes high-kNN datapoints.

**Why this construction:** the k-NN graph (efficiently construable via the method of Dong et al., 2011) captures the local density of the embedding space around each point without requiring explicit clustering. It's more robust than global density measures because it adapts to the varying density of different regions of the embedding space.

**Principal Component Analysis (PCA) variance for diversity** (Xu et al., 2023b). Let `$X = [g(x_1), g(x_2), \ldots, g(x_N)] \in \mathbb{R}^{|\mathcal{S}| \times H}$` be the matrix of all sentence embeddings. Perform PCA:

$$\text{Cov} = Q\Lambda Q^T = \frac{1}{|\mathcal{S}| - 1}(X - \mu_X)^T(X - \mu_X), \quad \mu_X = \frac{1}{|\mathcal{S}|} \sum_{i=1}^{|\mathcal{S}|} X_i$$

Select the top-`$k$` eigenvectors `$V = [v_1, v_2, \ldots, v_k]$` corresponding to the `$k$` largest eigenvalues `$\lambda_1 \ge \lambda_2 \ge \ldots \ge \lambda_k$`. Project each embedding into the reduced space:

$$Y = (X - \mu_X)V$$

where `$Y \in \mathbb{R}^{|\mathcal{S}| \times k}$`. Then compute the per-datapoint variance:

$$\text{Var}_i = \frac{1}{k - 1} \sum_{j=1}^k (Y_{ij} - \mu_i)^2, \quad \mu_i = \frac{1}{k} \sum_{j=1}^k Y_{ij}$$

**What this computes:** for each datapoint, how spread out its representation is across the principal components of the dataset. Datapoints whose embeddings have high variance in the PCA space are "outliers" occupying unique regions. Xu et al. select the top 20% by variance to maximize semantic diversity.

**Why PCA variance works for diversity:** unlike distance-based methods that require `$\mathcal{O}(N^2)$` pairwise comparisons, PCA is `$\mathcal{O}(NH^2 + H^3)$` — linear in dataset size (once the covariance is computed). The top principal components capture the directions of maximum variance in the data, so a datapoint with high variance in this space is far from the dataset centroid along important axes. This provides a computationally efficient diversity signal.

**Dataset-level diversity metrics.** Beyond per-datapoint diversity, the paper presents metrics for measuring the overall diversity of a selected subset `$\mathcal{S}_b$`:

$$\text{D}_{\text{kNN}}(\mathcal{S}) = \frac{1}{|\mathcal{S}|} \sum_{i=1}^{|\mathcal{S}|} \text{kNN}_i^1, \quad x_i \in \mathcal{S}$$

**What it computes:** the average distance from each datapoint to its nearest neighbor in the dataset. A higher average means examples are more spread out and less redundant.

**Cluster-inertia-based diversity** (Du & Black, 2019):

$$\text{D}_{\text{inertia}}(\mathcal{S}) = \sum_{j=1}^K \sum_{x_i \in C_j} \|g(x_i) - \mu_j\|^2, \quad \mu_j = \frac{1}{|C_j|} \sum_{x_i \in C_j} g(x_i)$$

where `$K$` is the number of k-means clusters and `$\mu_j$` is the centroid of cluster `$C_j$`. This is the standard k-means objective — it measures how tightly clustered the embeddings are around their centroids. Surprisingly, **higher** inertia means **more** diversity because the points are more spread out from their cluster centers.

**Cluster-radius-based diversity** (Lai et al., 2020) models each cluster as a multivariate Gaussian:

$$\text{D}_{\text{radius}}(\mathcal{S}) = \sqrt[H]{\prod_{j=1}^H \sigma_j}$$

where `$H$` is the embedding dimension and `$\sigma_j$` is the standard deviation along the `$j$`-th axis of the ellipsoid enclosing the cluster. This is the geometric mean of the standard deviations — a measure of the "volume" occupied by the embeddings.

**Inter-cluster distance diversity** (Dang & Verma, 2024):

$$\text{D}_{\text{ICD}}(\mathcal{S}) = \frac{1}{K} \sum_{j=1}^K \text{div}_{\text{JS}}(P_j \| P_{\neq j})$$

where `$P_j$` is the inverse-document frequency distribution of cluster `$C_j$`, `$P_{\neq j}$` is the IDF distribution of all other clusters combined, and `$\text{div}_{\text{JS}}$` is the Jensen-Shannon divergence. This measures how distinct the vocabulary and topic distribution of each cluster is from the rest — higher divergence means clusters are well-separated in content.

---

#### Diversity-Based Assessment: Model-Based Indicators (Section 4.2)

**Core mechanism:** Model-based diversity indicators use language models or other trained systems to estimate the rarity and variety of datapoints, often drawing on concepts from information theory and ecology.

**Entropy-based diversity.** The vanilla Shannon entropy (Shannon, 1948) of a dataset:

$$\text{D}_{\text{entropy}}(\mathcal{S}) = -\sum_{x_i \in \mathcal{S}} P(x_i \mid \theta) \cdot \log_2(P(x_i \mid \theta))$$

where `$P(x_i \mid \theta)$` is the probability of datapoint `$i$` under the language model `$\theta$`. High entropy means the datapoints have diverse probability masses — the model finds them varying in likelihood — which indicates semantic diversity.

**Why entropy for diversity:** in information theory, entropy measures the expected surprisal. A dataset where all examples have similar probabilities (evenly distributed) has higher entropy than one where a few examples dominate. The key insight is that if a dataset is semantically homogeneous (all examples are about the same topic, in the same style), a language model will assign them similar probabilities; if it's diverse, probabilities will vary widely.

**Rényi entropy** (Rényi, 1961) generalizes Shannon entropy with a parameter `$\alpha > 0, \alpha \neq 1$`:

$$\text{D}_{\text{RE}}^\alpha(\mathcal{S}) = \frac{1}{1 - \alpha} \log_2\left(\sum_{x_i \in \mathcal{S}} P(x_i \mid \theta)^\alpha\right)$$

**Why the `$\alpha$` parameter:** it controls sensitivity to rare vs. common events. When `$\alpha < 1$`, the entropy is more sensitive to rare examples (the `$\alpha$` exponent in `$P(x_i)^\alpha$` amplifies small probabilities). When `$\alpha > 1$`, common examples dominate. When `$\alpha \to 1$`, Rényi entropy converges to Shannon entropy. This flexibility is valuable for diversity measurement under class imbalance — a small `$\alpha$` can detect whether rare categories are present even when they account for a tiny fraction of the data.

**Simpson's Index** (Simpson, 1949), adapted from ecology for biodiversity measurement:

$$\text{D}_{\text{SI}}(\mathcal{S}) = \frac{2 \sum_{x_i, x_j \in \mathcal{S}, i \le j} \mathbb{1}(x_i = x_j \mid \theta)}{|\mathcal{S}|(|\mathcal{S}| + 1)}$$

where `$\mathbb{1}(x_i = x_j \mid \theta)$` is 1 if `$x_i$` and `$x_j$` are equivalent under the model's representation, 0 otherwise. The numerator counts pairs of equivalent examples; the denominator is the total number of pairs. Higher SI means more equivalent pairs — **lower** diversity. This is often reported as `$1 - \text{D}_{\text{SI}}$` to make higher = more diverse.

**The Vendi Score** (Dan Friedman & Dieng, 2023) is a more recent diversity metric based on the eigenspectrum of a similarity kernel:

$$\text{D}_{\text{VS}}^\alpha(\mathcal{S}) = \exp\left(\frac{1}{1 - \alpha} \log_2\left(\sum_{i=1, i \in \text{supp}(\bar{\lambda})}^{|\mathcal{S}|} \bar{\lambda}_i^\alpha \mid \theta\right)\right)$$

where `$\bar{\lambda}_i \mid \theta$` are the normalized eigenvalues of the similarity kernel matrix `$K_{\mathcal{S}} \mid \theta$` (summing to 1), and `$\text{supp}(\bar{\lambda})$` is the set of indices of non-zero eigenvalues. A common kernel implementation uses the Gaussian RBF: `$k(g(x_i \mid \theta), g(x_j \mid \theta)) = \exp(-\frac{1}{2}\|g(x_i \mid \theta) - g(x_j \mid \theta)\|^2)$`.

**What the Vendi Score computes:** it's the exponential of the Rényi entropy of the eigenspectrum. If all eigenvalues are equal (the kernel matrix has maximum effective rank), the data is maximally diverse — no direction dominates. If one eigenvalue dominates (the data lies on a low-dimensional manifold), diversity is low.

**Why the eigenspectrum:** the eigenvalues of the similarity kernel capture the intrinsic dimensionality and spread of the data. A flat spectrum means many independent directions of variation (high diversity); a steep spectrum means a few directions dominate (low diversity). This is fundamentally different from pairwise-distance-based metrics because it captures global dataset structure rather than local neighborhoods.

**Quality-weighted Vendi Score** (Nguyen & Dieng, 2024) extends this by multiplying the diversity score with average quality:

$$\text{D}_{\text{QVS}}(\mathcal{S}_b) = Q(\mathcal{S}_b) \times \text{D}_{\text{VS}}^\alpha(\mathcal{S}_b), \quad Q(\mathcal{S}_b) = \frac{1}{|\mathcal{S}_b|} \sum_{x_i \in \mathcal{S}_b} \text{IND}_i$$

where `$\text{IND}_i$` is any individual quality indicator. This provides a joint quality-diversity score that captures the tradeoff between selecting high-quality examples and maintaining representation across the data manifold.

**Task2Vec-based diversity** (Miranda et al., 2022; Lee et al., 2023). The Task2Vec embedding of a batch `$\mathcal{B}$` of data is the diagonal of the Fisher Information Matrix:

$$\hat{F}_{\mathcal{B}} = \mathbb{E}_{x_i, j, \hat{x}_i(j)}\left[\nabla_\theta \log P(\hat{x}_i(j) \mid x_i(<j); \theta) \cdot \nabla_\theta \log P(\hat{x}_i(j) \mid x_i(<j); \theta)^T\right]$$

where `$x_i$` is a datapoint sampled from the batch, `$\hat{x}_i(j)$` is the `$j$`-th token predicted by the model given the real prefix `$x_i(<j)$`, and the expectation averages over datapoints, positions, and model samples. The Task2Vec embedding is `$\vec{f}_{\mathcal{B}} = \text{diag}(F_{\mathcal{B}})$` — the diagonal entries of the FIM.

**What the FIM diagonal captures:** it encodes how sensitive the model's predictions are to each parameter when processing this batch. Different tasks perturb different subsets of parameters. The diagonal entries form a task signature that can be compared across batches.

The diversity coefficient for a dataset is then:

$$\text{D}_{\widehat{\text{div}}}(\mathcal{S}) = \mathbb{E}_{\mathcal{B}_1, \mathcal{B}_2 \sim \mathcal{S}}[d(\vec{f}_{\mathcal{B}_1}, \vec{f}_{\mathcal{B}_2})]$$

where `$\mathcal{B}_1$` and `$\mathcal{B}_2$` are two batches sampled from `$\mathcal{S}$`, and `$d$` is a distance metric (typically cosine distance). High diversity means batches from different parts of the dataset have dissimilar FIM diagonals — they tax the model in different ways, indicating task variety.

**Why this is a model-based diversity measure:** unlike embedding-based methods that measure semantic similarity of *inputs*, Task2Vec measures diversity of the *learning problem* posed by different data. Two batches could be semantically similar (both about dogs) but present different learning challenges (one is classification, one is generation), and Task2Vec would detect this while embedding similarity would not.

**TagLM-based diversity** (Lu et al., 2023a) uses open-ended intention tagging. A GPT-4-annotated dataset of (datapoint → fine-grained intention tags) is used to train a tagging model. For each datapoint, the model produces a set `$\mathcal{D}_{x_i}$` of atomic tags describing what skills and tasks the example involves. The diversity selection algorithm (Algorithm 1 in the paper) then iteratively selects datapoints that add new tags to the accumulated tag set:

$$\text{Select } x_i = \arg\max_{x_i \in \mathcal{S}} |\mathcal{D}_{x_i} \setminus \mathcal{D}_{\mathcal{S}_b}|$$

where `$\mathcal{D}_{\mathcal{S}_b}$` is the union of tags from all previously selected examples.

**Why this works:** it directly operationalizes task diversity — each new example is chosen because it covers skills not yet represented in the selected subset. This is more interpretable than embedding-space methods because the tags have human-readable labels (e.g., "summarization," "sentiment analysis," "code generation").

---

#### Diversity-Based Assessment: Geometry-Based Coreset Sampling (Section 4.3)

**Core mechanism:** Rather than scoring individual datapoints for diversity and then filtering, coreset sampling methods perform joint selection — they construct a subset that collectively maximizes geometric coverage of the embedding space. These are iterative greedy algorithms that add one datapoint at a time, each time choosing the point that most increases the diversity of the growing subset.

**K-Center Greedy** (Sener & Savarese, 2017; Algorithm 2 of the paper) is the canonical geometry-based diversity method:

$$\min_{\mathcal{S}_b \subset \mathcal{S}, |\mathcal{S}_b| = b} \max_{x_i \in \mathcal{S} \setminus \mathcal{S}_b} \min_{x_j \in \mathcal{S}_b} d(g(x_i), g(x_j))$$

**What this optimizes:** find a subset `$\mathcal{S}_b$` of size `$b$` such that the maximum distance from any unused datapoint to its nearest selected datapoint is minimized. In geometric terms: place `$b$` centers such that the largest "coverage hole" is as small as possible. This is the **minimax facility location problem** (Cornuéjols et al., 1983).

**Why minimax:** it guarantees worst-case coverage. Every datapoint in the full set has at least one selected representative within distance `$\delta$` (the minimized maximum). This ensures no region of the data space is neglected, unlike methods that optimize average distance (which could leave some regions completely uncovered if they're sparse).

**Why the greedy approximation:** exact solution of the k-center problem is NP-hard. The greedy algorithm provides a 2-approximation: the maximum distance achieved by greedy selection is at most twice the optimal maximum distance. The greedy procedure is:

1. Initialize `$\mathcal{S}_b$` with one or more seed points (either random or cluster centroids from k-means).
2. While `$|\mathcal{S}_b| < b$`:
   - Find the datapoint `$u$` that maximizes `$\min_{x_j \in \mathcal{S}_b} d(g(u), g(x_j))$` — the point farthest from all currently selected points.
   - Add `$u$` to `$\mathcal{S}_b$`.
3. Return `$\mathcal{S}_b$`.

**What plays the role of the evaluation function `$q$`:** in k-center greedy, the "score" of a candidate datapoint is not an inherent property but depends on the *current state of the selected subset*: `$q(x_i \mid \mathcal{S}_b) = \min_{x_j \in \mathcal{S}_b} d(g(x_i), g(x_j))$`. This is fundamentally different from quality-based methods where each datapoint has a fixed score independent of what else is selected. The selection mechanism `$\pi$` and the scoring function `$q$` are tightly coupled — you can't pre-compute scores and then filter; you must build the subset iteratively.

**Herding Greedy** (Harvey & Samadi, 2014; Algorithm 3) takes a different geometric approach. Rather than maximizing minimum distances, it minimizes the distance between the centroids of the selected and full datasets:

$$\mu = \frac{1}{N} \sum_{i=1}^N g(x_i)$$

Greedy selection: at each step `$t$`, choose:

$$u = \arg\min_{x_i \in \mathcal{S} \setminus \mathcal{S}_b} \left\|\mu - \frac{1}{|\mathcal{S}_b| + 1} \sum_{x_j \in \mathcal{S}_b \cup \{x_i\}} g(x_j)\right\|_2$$

**What this computes:** add the datapoint that, when included, makes the average embedding of the selected subset as close as possible to the average embedding of the full dataset. In geometric terms: herding "chases the centroid" — it tries to match the first moment (mean) of the full data distribution.

**Why herding vs. k-center:** herding optimizes for representativeness of the *center*; k-center optimizes for *coverage* of the extremes. If the data has a dense central cluster and sparse outliers, herding will heavily sample the center (matching the mean) while k-center will distribute points more evenly including to the periphery. The choice depends on whether you care more about faithfully representing the typical case (herding) or ensuring minority regions aren't missed (k-center).

**QDIT (Quality-Diversity Instruction Tuning)** (Bukharin & Zhao, 2023; Algorithm 5) combines quality and diversity in a single greedy selection loop. At each iteration:

$$u = \arg\max_{x_i \in \mathcal{S} \setminus \mathcal{S}_b} \left[(1 - \alpha) \cdot \text{D}_{\text{FL}}(\mathcal{S}_b \cup \{x_i\}) + \alpha \cdot \text{GPTScore}_i\right]$$

where `$\alpha \in [0, 1]$` controls the quality-diversity tradeoff, and `$\text{D}_{\text{FL}}(\mathcal{S}_b)$` is the facility location diversity measure:

$$\text{D}_{\text{FL}}(\mathcal{S}_b) = \sum_{x_j \in \mathcal{S}} \max_{x_i \in \mathcal{S}_b} \text{sim}(g(x_i), g(x_j))$$

Here `$\text{sim}(\cdot, \cdot)$` is a similarity function (typically cosine similarity). `$\text{D}_{\text{FL}}$` measures how well the selected subset "covers" the full set: for each datapoint in the full set, find its most similar selected neighbor and sum these maximum similarities.

**What the tradeoff parameter `$\alpha$` controls:** `$\alpha = 0$` means pure diversity (select to maximize coverage). `$\alpha = 1$` means pure quality (select the highest GPT-scored examples regardless of diversity). Intermediate `$\alpha$` balances the two — a candidate that's both high-quality and adds coverage will outscore one that's only high-quality (but redundant) or only diverse (but low-quality).

**Why dynamic weighting is important:** unlike sequential approaches (filter by quality first, then diversity), QDIT can select a lower-quality example if it substantially improves diversity, or pass over a diverse example if its quality is too low. The greedy maximization with dynamic tradeoff avoids the "filter-then-forget" problem where quality filtering permanently discards examples that could have contributed to diversity.

**DEITA (Data-Efficient Instruction Tuning for Alignment)** (Liu et al., 2023b; Algorithm 6) takes a sequential approach: quality-first, then diversity-aware. Specifically:

1. Compute a combined complexity-and-quality score: `$\text{GCQ}(x_i) = \mathcal{G}(x_i, p_C \mid \theta_C) \cdot \mathcal{G}(x_i, p_Q \mid \theta_Q)$` where `$\theta_C$` and `$\theta_Q$` are separately trained scoring models for complexity and quality, respectively, with prompts `$p_C$` and `$p_Q$`.
2. Sort all datapoints by GCQ score in descending order.
3. Iterate through the sorted list. For each candidate, check if it's sufficiently dissimilar from all already-selected examples:

$$\text{If } d(g(u), g(\mathcal{N}_0(u))) > \tau, \text{ where } \mathcal{N}_0(u) \in \mathcal{S}_b$$

   then add `$u$` to `$\mathcal{S}_b$`. Here `$\mathcal{N}_0(u)$` is `$u$`'s nearest neighbor in the already-selected set, and `$\tau$` is a diversity threshold.

**Why this sequential approach:** it's simpler than QDIT's dynamic tradeoff and can be implemented with pre-computed scores and a single pass through sorted data. The quality ranking ensures high-quality examples are considered first; the diversity check ensures that once a high-quality example from a region is selected, subsequent high-quality examples from the same region are skipped in favor of exploring new regions. The threshold `$\tau$` controls how similar is "too similar" — lower `$\tau$` allows more near-duplicates, higher `$\tau$` enforces stricter uniqueness.

**D4 Sampling** (Tirumala et al., 2024; Algorithm 7) adds a deduplication step before diversity sampling. The procedure:

1. **SemDeDup** (Abbas et al., 2023): for each of `$K_1$` k-means clusters, within each cluster, iteratively remove the example most similar to the cluster centroid `$\mu_j$` that has cosine similarity > `$\tau$` to any already-kept example. This removes semantic duplicates while keeping the example closest to each duplicate set (the one that appeared first in the greedy order).
2. **Prototypicality-based sampling**: re-cluster the deduplicated set into `$K_2$` clusters. Within each cluster, keep the examples that are **furthest** from the cluster centroid (the "outliers"), discarding the "prototypical" examples close to the center.

**Why discard prototypical examples:** Sorscher et al. (2022) showed that for data pruning, removing the most prototypical (centroid-close) examples and keeping outliers improves performance. The intuition: prototypical examples are highly redundant (many examples in the cluster are similar to the centroid), so discarding them doesn't lose unique information. Outliers capture the cluster's boundary and diversity.

**The proportion kept** is controlled by a percentile threshold: `$\mathcal{S}_b^j = \{x_i \mid \hat{F}_d(d(x_i, \mu_j)) > \frac{b}{K_2}, x_i \in C_j\}$`, where `$\hat{F}_d$` is the empirical CDF of distances to the centroid. This keeps the top `$\frac{b}{K_2}$` fraction of each cluster by distance-to-centroid, ensuring equal representation across clusters.

**Why two clustering passes:** the first pass (SemDeDup with `$K_1$` clusters) removes near-duplicates within fine-grained clusters. The second pass (prototypicality sampling with `$K_2$` clusters) selects diverse representatives across coarser clusters. The two `$K$` values can be tuned: larger `$K_1$` means stricter deduplication (within smaller clusters); larger `$K_2$` means finer-grained diversity control in the output.

**The Easy and Diverse First Sampling** (Jiang et al., 2024c; Algorithm 4) combines learning complexity with diversity by sampling cluster-by-cluster:

1. Cluster all datapoints into `$K$` groups via k-means on normalized embeddings.
2. Within each cluster `$C_j$`, compute the learning complexity `$\tilde{S}(x_i)$` (Eq. 11 from Section 3.2) for each example.
3. Select the `$\frac{b}{K}$` easiest examples (lowest complexity) from each cluster.

**Why equal allocation across clusters:** this enforces diversity by ensuring every cluster is represented in proportion to the number of clusters, not the number of examples in the cluster. A small cluster covering a rare task gets the same representation as a large cluster covering a common one, preventing the selection from being dominated by common tasks. The paper notes this implicitly enforces a diversity constraint: `$\text{D}_{\text{dist}}(\mathcal{S}) = \frac{1}{|\mathcal{S}|} \sum_{x_i \in \mathcal{S}} \min_{j \neq i} d(x_i, x_j) \ge C$`, where `$C$` controls the minimum separation.

**Comparison of diversity sampling approaches** (Alcoforado et al., 2024) finds that "the reverse semantic search performs most consistently and competitively." Reverse semantic search is essentially k-center greedy (Algorithm 2) initialized with the two most dissimilar datapoints rather than random seeds. In contrast, "the limited lexical similarity is sensitive to the hyper-parameter threshold `$\tau$`" and "the ordered clustering is not robust across datasets and fails to select high-quality samples." This validates the minimax facility location objective as a robust diversity criterion.

---

#### Diversity-Based Assessment: Bilevel Optimization-Based Coreset Sampling (Section 4.4)

**Core mechanism:** Bilevel optimization treats data selection and model training as a nested optimization problem: the outer loop selects data, and the inner loop trains the model on the selected data. The objective is typically to find the subset that minimizes the validation loss of the model trained on that subset:

$$\mathcal{S}_b = \arg\min_{\mathcal{S}_b' \subset \mathcal{S}} \sum_{x_i \in \mathcal{S}_b', \theta = \theta^*} \text{NLL}_i^{A|Q}, \quad \text{s.t. } \theta^* = \arg\min_\theta \sum_{x_i \in \mathcal{S}_b'} \text{NLL}_i^{A|Q}$$

**What this structure represents:** the inner loop trains model parameters `$\theta$` on the selected subset `$\mathcal{S}_b'$` by minimizing the language modeling loss. The outer loop evaluates the resulting model on a validation set and adjusts which examples are in `$\mathcal{S}_b'$` to minimize validation loss. This is a closed-loop optimization: the selection depends on the model, and the model depends on the selection.

**Why this is different from other diversity methods:** geometry-based methods (Section 4.3) select data based purely on embedding-space properties, without knowing what the model will actually learn. Bilevel methods directly optimize for the end goal (validation performance), theoretically producing better subsets. The cost is computational: solving a nested optimization is much more expensive than running k-center greedy.

**Why bilevel optimization is hard:** the outer loop's objective depends on `$\theta^*$`, which itself is the solution to an inner optimization problem. Computing the gradient of the outer objective with respect to the selection variables requires differentiating through the entire inner training procedure — extremely expensive for LLMs. Practical methods use various relaxations and approximations.

**The ScaleBiO method** (Pan et al., 2024) addresses this by transforming the bilevel problem into a single-loop minimax formulation:

$$\min_{\mathcal{S}_b'} \max_u \left[\sum_{x_i \in \mathcal{S}_b'} \text{NLL}_i^{A|Q} + \alpha \cdot \text{constraint}(\theta, u)\right]$$

where `$\alpha > 0$` is a multiplier, `$u$` is a proxy variable that tracks the inner-loop model parameters, and the constraint enforces that `$u$` approximates `$\theta^*$`. This single-loop formulation can be optimized with standard gradient methods, avoiding the nested iteration of true bilevel optimization.

**Why the multiplier `$\alpha$`:** it controls the tradeoff between optimizing the outer objective and satisfying the inner-loop constraint. Large `$\alpha$` enforces that the model trained on `$\mathcal{S}_b'$` closely matches the optimal model; small `$\alpha$` allows looser approximation in favor of optimizing the selection.

**Soft weights vs. hard selection.** The paper notes that bilevel methods often use "soft weights" rather than binary inclusion/exclusion. Each example `$x_i$` has a weight `$w_i \in [0, 1]$` representing its contribution to the loss, and the outer loop optimizes these continuous weights rather than discrete selection masks. This enables gradient-based optimization (binary masks produce zero gradients almost everywhere) and "guarantees a higher level of diversity as each sample contributes more or less to the overall optimization" — examples aren't permanently discarded, just down-weighted.

**The GLISTER method** (Killamsetty et al., 2021b) optimizes the outer level on a held-out validation set, minimizing validation loss rather than training loss. The Retrieve method (Killamsetty et al., 2021c) additionally incorporates self-supervised losses from unlabeled data (consistency regularization, entropy regularization) into both inner and outer loops, improving robustness when labeled data is scarce.

**Computational challenges at LLM scale.** The paper acknowledges that bilevel optimization for LLMs is extremely expensive: "it becomes more and more cumbersome to implement the entire pipeline for quality measurement and selection" (Section 7.4). The methods described in this subsection are primarily validated on smaller models and datasets; scaling them to 7B+ parameter LLMs with millions of training examples remains an open challenge.

---

#### Importance-Based Assessment: Hand-Crafted Indicators (Section 5.1)

**Core mechanism:** Importance assessment measures how valuable a datapoint is for the model's learning — typically operationalized as difficulty, complexity, or the amount of novel information the example provides relative to what the model already knows. Hand-crafted importance indicators use readability formulas and domain-specific difficulty labels.

**The unified formulation** for importance mirrors quality and diversity:

$$q(x_i) = f_i(q_C(x_i), q_P(x_i), q_G(x_i))$$

where `$q_C$` measures complexity/difficulty of the example itself, `$q_P$` measures the example's contribution to overall performance (loss/error-based), and `$q_G$` measures gradient-based influence. The paper notes that "most existing methods simply choose one of the above evaluation implementations," so `$f_i$` is typically an exclusive choice function (pick one dimension) rather than a true aggregation.

**Readability as a difficulty proxy.** The same readability formulas described in Section 3.1 (Dale-Chall, Flesch Reading Ease, Gunning Fog Index) can serve dual purpose: they measure both *quality* (well-written text should be at an appropriate reading level for its audience) and *difficulty* (text with advanced vocabulary, complex syntax, and dense information is harder to understand and thus potentially more valuable for training robust models). The paper notes that "samples with intricate grammar, advanced vocabulary, and inference dependency are deemed as difficult ones."

**Domain-specific difficulty labels.** For specialized datasets, difficulty is often explicitly labelled:

- **Math problems**: education level (elementary, high school, university) as in Patel et al. (2021) and Koncel-Kedziorski et al. (2016). The paper cites MATH (Hendrycks et al., 2021) as the standard benchmark, which already includes difficulty levels.
- **Reading comprehension**: question complexity based on required inference depth, number of supporting facts needed, or distractor plausibility.

**How difficulty-based selection works:** select examples at a target difficulty level or spread selection across difficulty levels. Easy examples are learned quickly and provide foundational skills; hard examples challenge the model and improve robustness. The paper notes that "the selection of difficult instruction-response pairs via Eq. 8 allows a wider distribution of performance for the models under investigation" — including hard examples increases the variance of model performance, making evaluation more discriminative.

**The percentage-of-difficult-words approach** (Klare, 1974; Begeny & Greene, 2014): count the fraction of words in a text that appear on a pre-defined "familiar words" list. Words not on the list are considered difficult. This is the simplest possible difficulty indicator and requires only a dictionary lookup.

**Key limitation:** hand-crafted difficulty indicators, like hand-crafted quality indicators, operate on surface text features. They cannot assess whether a datapoint is *important for the specific model being trained* — a text may be lexically complex but cover knowledge the model already possesses from pre-training, making it unimportant for fine-tuning. This motivates model-based importance assessment.

---

#### Importance-Based Assessment: Model-Based Indicators (Section 5.2)

**Core mechanism:** Model-based importance indicators use the target LLM or proxy models to estimate how much the model would benefit from learning a particular example. The paper categorizes these into uncertainty-based, reward score-based, and datamodel-based approaches.

**Prompt uncertainty** (Siddhant & Lipton, 2018; Kung et al., 2023; Nieth et al., 2024) measures how much the model's response varies when the instruction is paraphrased:

$$\text{U}_i^{\text{prompt}} = -\frac{1}{K} \sum_{k=1}^K \sum_{j=t}^{|x_i|} \left|P(x_i(j) \mid x_i(<j); \theta) - P(x_i(j) \mid \tilde{x}_i^k(<j); \theta)\right|$$

where `$K$` is the number of perturbed versions of the instruction, `$\tilde{x}_i^k$` is the `$k$`-th perturbed prompt (same response, paraphrased instruction), and the difference is the absolute change in predicted probability for the ground-truth response token `$x_i(j)$`.

**What this computes:** for each token position in the response, compare the model's predicted probability under the original instruction versus under each of `$K$` paraphrased instructions. Average the absolute differences across perturbations and positions. High prompt uncertainty means the model's predictions change substantially when the instruction is reworded — the model is sensitive to the specific phrasing rather than robustly understanding the underlying task.

**Why high uncertainty implies importance:** an example where the model is sensitive to instruction phrasing is one it hasn't truly mastered. Training on such examples should improve robustness and generalization. Conversely, examples where the model gives the same response regardless of phrasing are already well-learned and provide little marginal training value.

**CAPE (Calibrated Augmented Prompt Ensembles)** (Jiang et al., 2023b) addresses a specific pathology: LLMs after instruction tuning tend to be overconfident. CAPE first transforms all tasks into multiple-choice format, then applies prompt augmentations (paraphrasing, permuting in-context examples, permuting answer choices) to collect diverse predictions. The ensemble of predictions is used to calibrate uncertainty. "Such calibrated uncertainty tells if an instruction-tuned LLM simply memorizes the response to a given prompt rather than truly understanding the instruction" — the key importance signal.

**Reward models for necessity estimation.** Beyond quality scoring (Section 3.2), reward models can measure *necessity* — does the model actually need training on this example? The approach:

1. Prompt the current model `$\theta$` with the instruction `$x_i(<t)$` to generate a response `$\hat{x}_i^\theta(\ge t)$`.
2. Score the generated response with a reward model `$\phi$`: `$\hat{R}_i = r_\phi(x_i(<t), \hat{x}_i^\theta(\ge t))$`.
3. If `$\hat{R}_i$` is high (the model already produces good responses without training on this example), the example is deemed *unimportant* and not selected. If `$\hat{R}_i$` is low (the model's current responses are poor), the example is important and should be included.

**Why this measures importance rather than quality:** a high-quality instruction-response pair might still be unimportant if the model already handles it well. Importance is inherently model-dependent — it measures the gap between current and desired performance, not absolute data quality.

**Datamodel-based importance estimation** (Ilyas et al., 2022; Park et al., 2023). A datamodel is a predictive model — typically linear or a small neural network — that estimates how a model trained on a particular subset will perform on specific evaluation examples:

$$\tau_{\theta_x}(\mathbf{1}_{\mathcal{S}_b}) = \theta_x^T \mathbf{1}_{\mathcal{S}_b}$$

where `$\mathbf{1}_{\mathcal{S}_b} \in \{0, 1\}^{|\mathcal{S}|}$` is a binary indicator vector (1 if the example is in the training subset, 0 otherwise), and `$\theta_x$` is a learned weight vector that predicts the loss on evaluation example `$x$` when training on subset `$\mathcal{S}_b$`. The datamodel parameters are trained to minimize:

$$\theta_{x_j} = \arg\min_\theta \hat{\mathbb{E}}_{\mathcal{S}_i \sim \mathcal{S}_b \subset \mathcal{S}}^{(m)} \left[\mathcal{L}_{\text{reg}}(\tau_\theta(\mathbf{1}_{\mathcal{S}_i})), L_{x_j}(\mathcal{S}_i)\right]$$

where `$\hat{\mathbb{E}}^{(m)}$` is an empirical expectation over `$m$` randomly sampled training subsets, and `$L_{x_j}(\mathcal{S}_i)$` is the actual loss on evaluation example `$x_j$` when the model is trained on subset `$\mathcal{S}_i$`.

**What this does:** by training on `$m$` different random subsets and recording the resulting evaluation losses, the datamodel learns to predict how any subset composition affects performance. The weight `$\theta_x$` for each training example captures its marginal contribution to reducing evaluation loss.

**The TARK estimator** (Park et al., 2023) used in DsDm (Engstrom et al., 2024) provides an efficient way to estimate these datamodel weights. Once trained, the optimal subset for minimizing expected evaluation loss is:

$$\mathcal{S}_b = \arg\min_{\mathcal{S}_b' \subset \mathcal{S}} \hat{L}_{\mathcal{S}_{\text{eval}}}(\mathcal{S}_b'), \quad \hat{L}_{\mathcal{S}_{\text{eval}}}(\mathcal{S}_b') = \frac{1}{|\mathcal{S}_{\text{eval}}|} \sum_{x_j \in \mathcal{S}_{\text{eval}}} \theta_{x_j}^T \mathbf{1}_{\mathcal{S}_b'}$$

**What this computes:** the predicted average loss on the evaluation set if the model is trained on `$\mathcal{S}_b'$`, estimated as a linear combination of per-example datamodel weights. The subset that minimizes this predicted loss is selected.

**Why linear datamodels:** they are the simplest model class that can capture per-example additive effects. While real training dynamics are non-linear (interactions between examples, diminishing returns), linear datamodels with enough training subsets `$m$` provide surprisingly accurate loss predictions. The linearity also makes the final selection step trivial: sort training examples by their average weight on evaluation examples (`$\frac{1}{|\mathcal{S}_{\text{eval}}|} \sum_{x_j \in \mathcal{S}_{\text{eval}}} \theta_{x_j}$`) and take the top `$b$`.

**GPTfluence** (Liu et al., 2024b) models training dynamics as an `$n$`-th order Markov process:

$$\phi_t(x_k) = \sum_{j=1}^n \alpha_j(c_t)\phi_{t-j}(x_k) + \beta(c_t), \quad \forall x_k \in \mathcal{S}_{\text{eval}}$$

$$\alpha_j(c_t) = \sum_{i=1}^{|c_t|} A_{i,j}, \quad \beta(c_t) = \sum_{i=1}^{|c_t|} B_i, \quad \forall x_i \in c_t \subset \mathcal{S}$$

$$A_{ij} = \langle W_{(j)}^T g(x_i)_j, U_{(j)}^T g(x_k) \rangle_F, \quad B_i = \langle W'^T g(x_i)_j, U' g(x_k) \rangle_F$$

where `$c_t$` is the training batch at step `$t$`, `$\phi_t(x_k)$` is the predicted evaluation metric at time `$t$` for evaluation example `$x_k$`, `$g(\cdot)$` extracts BERT or GPT embeddings, `$W_{(j)}$`, `$U_{(j)}$`, `$W'$`, `$U'$` are learnable weight matrices, and `$\langle \cdot, \cdot \rangle_F$` is the Frobenius inner product.

**What this models:** the evaluation performance at time `$t$` depends on: (a) performance at the previous `$n$` timesteps (the Markov memory), weighted by `$\alpha_j$` factors that depend on the current training batch `$c_t$`; (b) an additive contribution `$\beta(c_t)$` from the current batch. The `$\alpha_j$` and `$\beta$` factors are themselves functions of the similarity between training examples in `$c_t$` and the evaluation example `$x_k$`, computed via learned bilinear forms on their embeddings.

**Why modeling dynamics matters:** most datamodels predict *final* performance as a function of the full training set. GPTfluence predicts the entire *trajectory*, which captures effects like catastrophic forgetting (an example might help early but hurt later) and curriculum effects (some examples are most beneficial at specific training stages). This richer modeling enables more nuanced importance estimation but requires tracking metrics throughout training.

**MATES** (Yu et al., 2024) addresses a key limitation of static datamodels: model preferences change during training. Rather than selecting data once before training, MATES maintains a small datamodel that is updated alternately with the main model. The datamodel selects the next batch based on the model's current state, creating a dynamic curriculum. "The datamodel, like a partner, is updated alternatively to adapt to the constantly changing preferences of the model under development."

**DSIR (Data Selection via Importance Resampling)** (Xie et al., 2023) takes a fundamentally different approach: importance is estimated by distributional resemblance to the target evaluation set, without training any datamodel:

$$w_i = \frac{\hat{w}_i}{\sum_{i=1}^{|\mathcal{S}|} \hat{w}_i}, \quad \hat{w}_i = \frac{\hat{p}_{\text{feat}}(h(x_i))}{\hat{q}_{\text{feat}}(h(x_i))}$$

$$\hat{p}_{\text{feat}}(h(x_i)) = \prod_{j=1}^m \gamma_j^{h(x_i)_j}, \quad \hat{q}_{\text{feat}}(h(x_i)) = \prod_{j=1}^m \beta_j^{h(x_i)_j}$$

$$\hat{\gamma} = \frac{1}{\sum_{x_i \in \mathcal{S}_{\text{eval}}} \mathbf{1}^T h(x_i)} \sum_{x_j \in \mathcal{S}_{\text{eval}}} h(x_j), \quad \hat{\beta} = \frac{1}{\sum_{x_i \in \mathcal{S}} \mathbf{1}^T h(x_i)} \sum_{x_j \in \mathcal{S}} h(x_j)$$

where `$h(x_i) \in \mathbb{N}^m$` is a hashed n-gram feature vector of dimension `$m$` (sparse bag-of-ngrams), `$\gamma_j$` is the empirical probability of n-gram `$j$` in the evaluation set, and `$\beta_j$` is the empirical probability in the full training set.

**What DSIR computes:** for each training example, the importance weight `$w_i$` is the ratio of how likely its n-gram features are under the evaluation distribution versus the training distribution. Examples with n-gram patterns that are common in the evaluation set but rare in the overall training set get high weight (they help bridge the distribution gap). Examples with patterns common in training but rare in evaluation get low weight (they're less relevant).

**Why cheap features work:** DSIR uses simple hashed n-grams rather than neural embeddings or datamodels. The paper notes that "cheap approximation of features by bag-of-n-grams achieves similar performance but requires much less computing resources" (Section 7.3). This makes DSIR applicable to billion-token datasets where embedding computation or datamodel training would be prohibitive.

**Selection:** given the importance weights `$w_i$`, the subset `$\mathcal{S}_b$` is obtained by importance-weighted sampling without replacement for `$b$` draws, where the probability of selecting `$x_i$` at each draw (conditioned on not having selected it already) is proportional to `$w_i$`.

---

#### Importance-Based Assessment: Loss and Error-Based Coreset Sampling (Section 5.3)

**Core mechanism:** Rather than pre-computing importance scores, loss- and error-based methods track example behavior during actual or simulated training and select based on observed dynamics. The core insight: examples that are consistently misclassified, frequently forgotten, or cause large loss fluctuations are important — the model struggles with them and benefits from repeated exposure.

**Forgetting score** (Toneva et al., 2018). Track the accuracy of each example across training epochs. An example `$x_i$` undergoes a **forgetting event** at step `$t$` if:

$$\text{acc}_i^{t-1} > \text{acc}_i^t$$

— the model previously classified it correctly but now gets it wrong. Conversely, a **learning event** occurs when `$\text{acc}_i^t < \text{acc}_i^{t+1}$` — the model transitions from wrong to correct.

An example is classified as **unforgettable** if:

$$\text{Unforget}_i = \begin{cases} 1, & \exists t^* < \infty \text{ s.t. } \text{acc}_i^{t} < \text{acc}_i^{t+1} \text{ and } \forall k \ge t^*, \text{acc}_i^k > \text{acc}_i^{k-1} \\ 0, & \text{otherwise} \end{cases}$$

**What this means:** once the model learns the example (accuracy increases), it never subsequently forgets it — the example is stably learned. The forgetting score for an example is the number of forgetting events it experiences during training. High-forgetting examples are difficult for the model to retain; low-forgetting examples are easy.

**Why forgetting implies importance:** examples that are frequently forgotten lie near decision boundaries or require rare capabilities. Training on them — especially multiple times — helps the model stabilize its predictions. Easy, unforgettable examples can be pruned: `$\mathcal{S}_b = \{x_i \mid \text{Unforget}_i = 0, x_i \in \mathcal{S}\}$`. The paper notes that this approach has been validated for both pre-training and instruction tuning (Sorscher et al., 2022; Paul et al., 2021; Zhang et al., 2023a; Jin & Ren, 2024a; Maini et al., 2022).

**Memorization score** (Feldman, 2020; Feldman & Zhang, 2020). The memorization of example `$x_i$` is defined as the performance gap between a model trained with and without that example:

$$\text{Memo}_i = \frac{1}{|x_i| - t} \sum_{j=t}^{|x_i|} \left(P(x_i(j) \mid x_i(<j); \theta_{\mathcal{S}}) - P(x_i(j) \mid x_i(<j); \theta_{\mathcal{S} \setminus x_i})\right)$$

where `$\theta_{\mathcal{S}}$` is the model trained on the full dataset, `$\theta_{\mathcal{S} \setminus x_i}$` is the model trained on the full dataset *except* example `$x_i$`, and the difference is in the predicted probability of the ground-truth response tokens.

**What Memorization captures:** how much the model's ability to predict `$x_i$` depends on having seen `$x_i$` during training. High memorization means the model can't generalize from other examples to this one — it needed to memorize the specific example. Low memorization means the example is consistent with general patterns and doesn't require specific memorization.

**The influence of `$x_i$` on another example `$x_k$`** is analogously:

$$\text{Infl}_{ik} = \frac{1}{|x_k| - t} \sum_{j=t}^{|x_k|} \left(P(x_k(j) \mid x_k(<j); \theta_{\mathcal{S}}) - P(x_k(j) \mid x_k(<j); \theta_{\mathcal{S} \setminus x_i})\right)$$

**What this captures:** how much having trained on `$x_i$` changes the model's predictions on `$x_k$`. Positive influence means training on `$x_i$` helps predict `$x_k$` (increase in probability of correct tokens); negative influence means it hurts.

**Practical approximation via batch sampling.** Computing exact memorization and influence requires training `$|\mathcal{S}| + 1$` separate models (one with each example removed, plus the full model), which is infeasible. The standard approximation (Feldman & Zhang, 2020): sample `$N$` batches `$\mathcal{B}_1, \ldots, \mathcal{B}_N$` of size `$n$` from `$\mathcal{S}$`. For each example `$x_i$`, some batches contain it and others don't. Train a model on each batch. The two probability terms in Memo and Infl are approximated by averaging model outputs across batches that respectively contain or exclude `$x_i$`. The paper notes that this "batch-wise sampling tricks with a greedy principle behind" make the computation tractable.

**Empirical ranking of pruning metrics** (Sorscher et al., 2022): memorization scores outperform random sampling, EL2N scores (Eq. 13), and influence scores (Eq. 55) for pruning datasets to small subsets. This suggests that per-example memorization — which directly measures whether the example is redundant with others — is a stronger signal than difficulty (EL2N) or cross-example influence.

**AME (Average Marginal Effect)** (Lin et al., 2022) uses Shapley-value-inspired estimation. The marginal effect of adding `$x_i$` to a random subset `$\mathcal{S}' \subset \mathcal{S} \setminus \{x_i\}$` is the performance difference on the evaluation set between models trained with and without `$x_i$` in the subset. The AME is the expectation of this marginal effect over random subsets:

$$\text{AME}_i = \mathbb{E}_{\mathcal{S}' \subset \mathcal{S} \setminus \{x_i\}}[L(\mathcal{S}') - L(\mathcal{S}' \cup \{x_i\})]$$

**What this represents:** the average reduction in loss attributable to adding `$x_i$` to a random training subset. Examples with high AME are consistently beneficial regardless of what other examples are present; examples with low AME are redundant or harmful.

**Why AME is related to Shapley values:** the Shapley value averages marginal contributions over all possible subset orderings with appropriate weighting. AME simplifies this by using uniform weighting over random subsets — less computationally demanding but less axiomatic.

---

#### Importance-Based Assessment: Gradient-Based Coreset Sampling (Section 5.4)

**Core mechanism:** Gradient-based methods operate on the principle that the most important training examples are those whose gradients most closely align with the "ideal" update direction — either the average gradient of the full dataset, the gradient of a validation set, or the gradient of a specific evaluation example.

**Gradient matching** (Zhao et al., 2020a; Killamsetty et al., 2021a):

$$\theta^*, \mathcal{S}_b^* = \arg\min_{\theta, \mathcal{S}} d\left(\frac{1}{|\mathcal{S}|} \sum_{x_i \in \mathcal{S}} \nabla_\theta \text{NLL}_i^{A|Q}, \frac{1}{\sum_i w_i} \sum_{x_i \in \mathcal{S}_b} w_i \nabla_\theta \text{NLL}_i^{A|Q}\right)$$

**What this optimizes:** find a subset `$\mathcal{S}_b \subset \mathcal{S}$` with per-example weights `$w_i > 0$` such that the weighted average gradient of the subset approximates the average gradient of the full dataset as closely as possible, under distance metric `$d(\cdot, \cdot)$`.

**Why match gradients:** if a single gradient step on the subset moves parameters in approximately the same direction as a gradient step on the full dataset, training on the subset should produce similar model updates. This is a more direct optimization target than matching data distributions or embedding coverage — it directly optimizes for equivalent optimization trajectories.

**Why this is computationally expensive:** computing `$\nabla_\theta \text{NLL}_i^{A|Q}$` for every example in a large dataset requires a full forward-backward pass per example, then storing `$|\mathcal{S}|$` gradients each of dimension `$|\theta|$` (millions to billions for LLMs). The paper notes that "gradient-based coreset sampling techniques are highly dependent on the LLMs under development" and that "approximation is unavoidable for application on LLMs." The orthogonal matching pursuit algorithm (Killamsetty et al., 2021a) and low-rank approximations (Xia et al., 2024a) provide speedups but lose some precision.

**Influence functions** (Koh & Liang, 2017) provide a different gradient-based importance measure. For a model with optimal parameters `$\theta^*$`, the parameters that would result from up-weighting example `$x_i$` by a small amount `$\epsilon$` can be approximated via a first-order Taylor expansion:

$$\theta_{x_i}^\epsilon = \arg\min_\theta \frac{1}{|\mathcal{S}|} \sum_{x_j \in \mathcal{S}} \text{NLL}_j^{A|Q} + \epsilon \cdot \text{NLL}_i^{A|Q}$$

$$\theta_{x_i}^\epsilon \approx \theta^* - \epsilon H_{\theta^*}^{-1} \nabla_\theta \text{NLL}_i^{A|Q}$$

where `$H_{\theta^*}$` is the Hessian matrix of the loss with respect to parameters `$\theta$`, evaluated at `$\theta^*$`.

**The influence of `$x_i$` on the model parameters:**

$$\text{InflF}_i = \frac{d\theta_{x_i}^\epsilon}{d\epsilon}\bigg|_{\epsilon=0} = -H_{\theta^*}^{-1} \nabla_\theta \text{NLL}_i^{A|Q}$$

**What this represents:** the direction and magnitude of parameter change that would result from slightly increasing the weight of `$x_i$` during training. This is a vector in parameter space — it tells us which parameters `$x_i$` would most strongly affect.

**The influence of `$x_i$` on the loss of another example `$x_j$`:**

$$\text{InflF}_{ij} = -\nabla_\theta \text{NLL}_j^{A|Q}^T H_{\theta^*}^{-1} \nabla_\theta \text{NLL}_i^{A|Q}$$

**What this computes:** the change in the loss on example `$x_j$` that would result from up-weighting `$x_i$`. It's the inner product (under the inverse Hessian metric) of the gradients of `$x_i$` and `$x_j$`. If `$\text{InflF}_{ij}$` is large and negative, up-weighting `$x_i$` would reduce the loss on `$x_j$` — `$x_i$` is helpful for `$x_j$`. If large and positive, `$x_i$` would harm performance on `$x_j$`.

**Why the inverse Hessian appears:** the Hessian captures the local curvature of the loss landscape. A gradient step in a high-curvature direction (large Hessian eigenvalue) produces a smaller parameter change than a step in a low-curvature direction, because the optimization "resists" moving along directions where the loss changes rapidly. The inverse Hessian corrects for this, converting raw gradients into actual parameter changes.

**Computational challenges at LLM scale.** Computing `$H_{\theta^*}^{-1}$` for an LLM with billions of parameters is impossible exactly. The paper surveys several approximations:

- **Hessian-vector products** (Pearlmutter, 1994): compute `$Hv$` for any vector `$v$` without materializing `$H$`, enabling iterative solution of `$H^{-1}g$` via conjugate gradient.
- **EK-FAC** (George et al., 2021; Grosse et al., 2023): Eigenvalue-corrected Kronecker-Factored Approximate Curvature, which approximates the Hessian as a Kronecker product of layer-wise matrices, reducing storage from `$\mathcal{O}(|\theta|^2)$` to `$\mathcal{O}(\sum_{\ell} d_{\ell,\text{in}}^2 + d_{\ell,\text{out}}^2)$`.
- **Arnoldi iteration** (Schioppa et al., 2021): approximates the top eigenspectrum of the Hessian, enabling low-rank approximation of `$H^{-1}$`.

Grosse et al. (2023) successfully scaled influence functions to 52-billion-parameter models using EK-FAC, studying which pre-training data most influenced math abilities, programming skills, cross-lingual generalization, and role-playing behavior. This demonstrates that gradient-based importance, while expensive, is feasible for the largest models with appropriate approximations.

**GraNd score** (Paul et al., 2021):

$$\text{GraNd}_i = \mathbb{E}_\theta\left[\|\nabla_\theta \text{NLL}_i^{A|Q}\|_2\right]$$

**What this computes:** the expected Euclidean norm of the gradient of example `$i$`'s loss with respect to model parameters. Examples with large gradient norms cause large parameter updates — they are "surprising" or difficult for the model.

**Why GraNd ≈ EL2N:** Paul et al. (2021) show empirically that for models trained with cross-entropy loss, the gradient norm is well approximated by the EL2N score (Eq. 13). This is important because EL2N can be computed from model outputs without backpropagation, making it much cheaper than actual gradient computation. The paper notes this connection explicitly: "the GraNd score can be well approximated by EL2N score for efficient data pruning."

**MoSo (Moving-one-Sample-out)** (Tan et al., 2024a) avoids full retraining by using a gradient-based approximator. Rather than literally removing one example and retraining, it identifies examples whose gradients are "consistently aligned with the average gradients of the entire training set." These are the least informative examples — removing them would barely change the overall gradient direction — and can be safely pruned.

**LESS (Low-rank gradient Similarity Search)** (Xia et al., 2024a) uses low-rank gradient approximations to efficiently find training examples whose gradients are most similar to the gradients of evaluation examples. This directly selects data that would most reduce evaluation loss, combining the principle of influence functions (Section 5.4) with computational tractability via dimensionality reduction. The paper reports that LESS with 5% of data can approach full-dataset performance on MMLU, TYDIQA, and BBH (Table 4), demonstrating the practical viability of gradient-based importance at moderate scale.

**Key takeaway from the paper on gradient methods:** "The gradient-based coreset sampling techniques are highly dependent on the LLMs under development, where the gradients describe the model's inherent knowledge and uncertainty about each datapoint. Despite the precision of gradient-based selection methods, it is noted that approximation is unavoidable for application on LLMs. The efficiency and accuracy of various approximation techniques should be considered."

---

#### Cross-Cutting Design Patterns

Several recurring themes span the quality, diversity, and importance categories:

**1. The proxy model strategy.** Computationally expensive assessment methods can often be approximated using smaller, cheaper models. The paper documents this pattern in multiple contexts: Ankner et al. (2024) use MPT 125M to filter data for MPT 1B (Section 3.2); Li et al. (2024b) validate that GPT2-125M perplexity and IFD scores select good data for LLaMA2-7B and 13B (Section 3.2); QuRator (Wettig et al., 2024) distills GPT-3.5 pairwise quality judgments into a Sheared-LLaMA 1.3B scorer (Section 3.3); DSIR (Xie et al., 2023) uses cheap hashed n-gram features rather than neural embeddings for importance estimation (Section 5.2). The paper observes this as a general scaling strategy: "small proxy models have been successfully applied in accelerated fine-tuning of language models" (Section 7.4), and "the scaling law confirms the expected consistent behavior between data quantity and model scale, providing practical guidelines on the development of such proxy LLMs."

**2. Individual scoring vs. joint selection.** Quality-based methods typically compute per-example scores independently (a datapoint's quality score doesn't depend on what else is selected), enabling pre-computation and simple threshold filtering. Diversity-based methods like k-center greedy require joint selection (a datapoint's value depends on the current selected subset) and are inherently iterative. Importance-based methods span both: datamodel-based approaches pre-compute importance weights, while gradient matching requires iterative subset construction. The computational cost of pre-computation vs. iterative selection is a key practical consideration.

**3. The evaluation set dependency.** Importance-based methods "are highly dependent on the LLMs under development" (Section 5.4) and on the evaluation set used to estimate influence. The same datapoint may be highly important for one downstream task and irrelevant for another. This is a fundamental limitation: "instructions that resemble the most to the testing set or bring about performance gains are judged as 'good' data. However, such 'good' data cannot be easily transferred to another LLM of completely different architecture and parameters" (Section 7.2). Quality and diversity, by contrast, are (mostly) model- and task-independent — a well-written, diverse dataset is generally useful regardless of the specific evaluation.

**4. Hybrid methods and the coupling problem.** The paper notes that existing hybrid methods (LIFT, DEITA, QDIT, FL) are predominantly "parallel setups" or "sequential setups." Parallel methods optimize a weighted sum of quality and diversity scores at each selection step (QDIT's dynamic `$\alpha$` weighting). Sequential methods apply quality filtering first, then diversity-based selection on the survivors (DEITA). The paper critiques sequential setups because they "fail to retrieve the candidates that are filtered out in the preceding quality control steps even if those candidates are of high importance or variety." This coupling — the order of operations matters — is a key design consideration when combining multiple assessment dimensions.

**5. The missing importance dimension.** The paper observes that "the importance-based assessment is often overlooked in existing hybrid approaches, implying that the investigation of integrating importance with quality and diversity is of high potentials for future studies." Only a few methods (e.g., LESS) combine importance (via gradient similarity) with quality and diversity considerations. This is identified as a significant gap.

## 4. Key Insights and Innovations

### Innovation 1: The Three-Axis Taxonomy as a Unifying Framework for a Fragmented Literature

The survey's single most important conceptual contribution is the taxonomy itself — the claim that all data assessment methods for instruction tuning can be organized into exactly three fundamental perspectives: **quality, diversity, and importance**. This is not merely a filing system; it is an analytical lens that reveals structure where the prior literature saw only isolated techniques.

**What the field had before.** Prior to this survey, research on instruction tuning data selection was characterized by what the authors call "terminological confusion" (Section 1.2): "quality, diversity, and importance might be used interchangeably without strict discrimination in previous studies." Each paper proposed its own metric — IFD scores, GPT-4 ratings, k-center greedy clustering, gradient matching, EL2N filtering — with its own vocabulary and its own validation setup. There was no shared language for comparing across approaches, no framework for understanding whether two methods were measuring the same underlying property (and thus redundant) or different properties (and thus complementary). The result was a literature that accumulated techniques without accumulating understanding: the field knew *that* certain methods worked on certain datasets, but not *why*, or *when* one method should be preferred over another, or *whether* combining methods would help or hurt.

**What the taxonomy achieves.** By defining quality, diversity, and importance as orthogonal axes — and by providing formal decompositions for each (quality as instruction clarity + response correctness, diversity as lexical variety + semantic spread, importance as difficulty + performance contribution + gradient alignment) — the survey makes several intellectual moves simultaneously:

First, it **retroactively classifies** the existing literature, showing that dozens of seemingly disparate methods are in fact variations on three themes. A perplexity filter and a GPT-4 scorer are both quality-based methods, differing only in *who does the judging* (small proxy model vs. powerful closed-source LLM). K-center greedy and Vendi Score computation are both diversity-based methods, differing only in *what notion of coverage* they optimize (minimax facility location vs. eigenspectrum flatness). This classification is not arbitrary — it is grounded in the mathematical structure of the selection problem (Section 2, Eq. 2–4), where all methods share the form `S_b = π(S, b, q)` and differ only in what `q` measures.

Second, it **reveals blind spots**. The taxonomy makes visible that importance-based assessment is "often overlooked in existing hybrid approaches" (Section 6). Methods like DEITA and QDIT combine quality and diversity, but almost no method systematically integrates all three axes. This is not a random omission — it reflects a genuine difficulty: importance is inherently model-dependent and evaluation-set-dependent, while quality and diversity are (mostly) model-agnostic. The taxonomy doesn't solve this difficulty, but it frames it clearly as a research gap rather than an inexplicable absence.

Third, it **enables systematic comparison**. By decomposing quality into sub-dimensions (instruction clarity, accuracy, explicitness; response correctness, coherence, pertinence) and noting that "all the mentioned quality measurement components above are only demonstrative and are not enforced explicitly in the development of existing quality-based methods" (Section 3 introduction), the survey provides a checklist for evaluating any new quality metric: what sub-dimensions does it actually capture, and what does it miss? A perplexity-based filter primarily captures fluency (related to clarity and coherence) but says nothing about factual correctness. A reward model captures multiple dimensions simultaneously but is expensive and difficult to calibrate. This diagnostic capability is the taxonomy's practical value — it helps practitioners reason about method complementarity rather than treating quality assessment as a black box.

**Why this is foundational rather than incremental.** Taxonomies in mature fields (biology's Linnaean system, chemistry's periodic table) are sometimes dismissed as "mere classification," but they are in fact the prerequisite for theory. Before the taxonomy, the instruction tuning data selection literature had no shared vocabulary, no way to state hypotheses about method complementarity, and no framework for designing ablation studies that isolate *which property* of a selection method drives performance. After the taxonomy, one can ask precise questions: "Does adding diversity-based filtering to quality-based filtering improve generalization, or do they capture redundant information?" "Is the benefit of DEITA due to its quality scoring, its diversity threshold, or the interaction?" "Why does LESS (importance-based) sometimes outperform quality+diversity methods?" The taxonomy doesn't answer these questions, but it makes them *askable* — and that is the mark of a genuine conceptual advance.

The paper's own evidence for the taxonomy's utility is indirect but compelling: the experimental results in Tables 2–4 show that methods from different categories achieve different patterns of performance (quality methods match full-data performance at small fractions; diversity methods improve generalization; importance methods target specific evaluation sets), confirming that the categories capture real functional distinctions. The fact that hybrid methods (DEITA, QDIT) consistently outperform single-axis methods (Tables 2–3) validates the taxonomy's central implication: the axes are complementary, and optimal selection requires considering all three.

---

### Innovation 2: The Unified Selection Formalism as a Rosetta Stone

A deceptively simple but intellectually powerful move is the paper's formalization of all data selection as instances of a single mathematical template: `S_b = π(S, b, q)`, where `q` is an evaluation function and `π` is a selection mechanism. This formalism, presented in Section 2, serves as a "Rosetta Stone" that translates between the vocabularies of otherwise incommensurable methods.

**What the field had before.** Different sub-communities studying data selection used different mathematical frameworks and different terminology. The quality-filtering literature spoke of "thresholds" and "percentile-based filtering." The coreset literature spoke of "facility location objectives" and "submodular maximization." The active learning literature spoke of "uncertainty sampling" and "query strategies." The influence function literature spoke of "Hessian-vector products" and "up-weighting." These communities rarely cited each other, and when they did, they often talked past each other because they lacked a common mathematical language.

**What the formalism reveals.** By decomposing every selection method into a scoring function `q` and a selection mechanism `π`, the survey reveals that:

**The scoring function `q` and the selection mechanism `π` are conceptually independent but practically coupled.** In quality-based threshold filtering (Section 3.1), `q` is a per-example absolute score (perplexity, GPT rating) and `π` is a simple threshold. In k-center greedy (Section 4.3), `q(x_i | S_b)` depends on the *current state* of the selected subset — it's the distance to the nearest already-selected example — and `π` is an iterative maximization. The fact that `q` can be either state-independent or state-dependent is not a bug in the formalism; it's the formalism's key diagnostic insight. It explains why diversity-based methods cannot be implemented as simple pre-compute-then-filter pipelines: their scoring function requires knowing what else has been selected.

**The same `q` can serve different `π` mechanisms, and vice versa.** The paper shows that IFD scores (`q`) can be used with either threshold filtering (`π`, Eq. 7) or percentile-based selection (`π`, Eq. 8), depending on whether absolute or relative quality standards are desired. Conversely, the same greedy selection mechanism (`π`, Algorithm 2) can be driven by different scoring functions — distance-to-nearest-neighbor for diversity, quality-weighted-distance for QDIT's hybrid approach. This decomposition makes the design space explicit: there are ~M ways to compute `q` and ~N ways to implement `π`, yielding M×N possible combinations, most of which have never been explored.

**The budget constraint `b` interacts differently with different `q`-`π` pairs.** For greedy selection (`π_greedy`), the budget directly controls subset size and the algorithm naturally produces nested subsets (the first k selected examples are optimal for budget k). For probabilistic selection (`π_prob`), the budget controls sample count but the subset is random, so different runs with the same budget produce different subsets. This distinction has practical consequences: greedy methods are deterministic and reproducible but may miss rare high-value examples; probabilistic methods explore more broadly but with variance.

**Why this is a genuine innovation rather than mere notation.** The formalism does more than summarize; it **generates hypotheses**. If every method is `S_b = π(S, b, q)`, then one can ask: "What happens if we take the `q` from method A (say, IFD quality scores) and combine it with the `π` from method B (say, k-center greedy)?" The answer is not obvious — IFD scores are absolute quality judgments, and k-center greedy assumes a distance-to-subset scoring — but the question itself is well-posed only because of the formalism. The paper does not explore such cross-method combinations systematically (this is identified as future work in Section 7), but the formalism provides the intellectual scaffolding for doing so.

**Evidence from the survey's own organization.** The formalism is not just stated and abandoned; it genuinely structures the entire survey. Each method in Sections 3–5 is described in terms of its `q` and `π`. The taxonomy (quality/diversity/importance) classifies `q`; the sub-categories (hand-crafted/model-based/coreset) classify how `q` is computed and whether `π` is simple or iterative. This organizational coherence — which would be impossible without the formalism — is evidence that the formalism captures something real about the structure of the problem.

---

### Innovation 3: The Proxy Model Strategy as a General Scaling Principle

While individual papers have used small models to approximate expensive computations for larger models (Ankner et al., 2024; Li et al., 2024b; Wettig et al., 2024), the survey is the first to **identify this as a cross-cutting design pattern** with general applicability across all three assessment dimensions, and to articulate the conditions under which it works.

**What the field had before.** The dominant assumption — often implicit — was that data assessment for a large model requires that large model itself, or an even more capable one. IFD scores were computed using the target model after warm-up. Gradient-based influence required computing Hessian-vector products for the full model. GPT-4 scoring used a model larger and more capable than the one being fine-tuned (GPT-4 vs. LLaMA-7B). The intuition was that only a model of sufficient capability could accurately judge what data would benefit training.

**The survey's reframing.** By systematically cataloging instances where small proxy models work, the survey transforms an empirical curiosity into a **scaling principle**:

- **Quality**: A 125M GPT-2 model's perplexity effectively filters data for a 7B LLaMA model (Li et al., 2024b) — a 56× reduction in the model size needed for assessment.
- **Diversity**: Embedding computation with Sentence-BERT (110M parameters) provides sufficient semantic representations for k-center greedy clustering of datasets destined for models 100× larger.
- **Importance**: DSIR (Xie et al., 2023) uses cheap hashed n-gram features rather than neural embeddings or trained datamodels, yet matches the performance of far more expensive importance estimation methods.

**Why this is conceptually significant.** The proxy model strategy challenges a deeply held assumption: that data quality is an *absolute* property best judged by the most capable model available. Instead, it suggests that for many assessment tasks — particularly quality filtering and diversity measurement — the signal is **model-size-invariant** beyond a relatively low threshold. A 125M model identifies the same obviously-bad examples (nonsensical text, extreme repetition) as a 7B model. A 110M sentence encoder captures the same semantic clusters as a 7B model would. This has profound practical implications: it means data assessment can be decoupled from the model being trained, amortizing the cost of assessment across multiple training runs and model sizes.

**The survey also identifies the limits of this principle.** Importance-based methods — gradient matching, influence functions, datamodels — are explicitly **not** proxy-friendly in the same way, because they measure *model-specific* effects. A datapoint's gradient depends on the specific model parameters; its influence on evaluation loss depends on the specific evaluation set and model architecture. The survey's taxonomy makes this distinction visible: quality and diversity are (mostly) model-agnostic assessment dimensions that benefit from proxy models; importance is model-dependent and requires model-specific computation (with approximations). This boundary condition — *when* proxy models work and *when they don't* — is a genuine conceptual contribution, not just an engineering observation.

**Evidence from Table 2.** The results for PPL-based filtering (Ankner et al., 2024) show that a 125M proxy model selecting data for a 1B training model achieves competitive performance with full-dataset training, and that the "Mid" and "High" perplexity percentiles consistently outperform "Low" percentiles and often match the full dataset. This validates both the proxy model strategy (a small model's perplexity judgments transfer) and the non-monotonic relationship between perplexity and data quality (extremely low perplexity examples are too easy, not the most valuable).

---

### Innovation 4: Difficulty-Conditioned Value of Data — The "Easy vs. Hard" Regime Shift

The survey documents — across multiple methods and multiple assessment dimensions — a fundamental regime shift in what constitutes "good" data based on the **interaction between dataset size and example difficulty**. This is not presented as a single claim but emerges as a recurring pattern that the taxonomy makes visible.

**The pattern.** For quality-based methods: Jiang et al. (2024c) propose learning-complexity-based selection where "in a data-poor regime, easy datapoints are more informative and should be kept first. On the contrary, in a data-rich regime, hard datapoints should be treasured." For diversity-based methods: the D4 sampling approach (Tirumala et al., 2024) discards prototypical (centroid-close) examples and keeps outliers — but this is optimal only when the dataset is large enough that prototypes are redundant. For importance-based methods: the forgetting score (Toneva et al., 2018) identifies examples that are difficult for the model to retain, but these are valuable only if the model already has foundational capabilities from easier examples.

**Why this is a conceptual innovation rather than an obvious observation.** The field's default assumption — reinforced by the "data is the new oil" narrative — was that more data is always better, and that the best data is whatever is most "high-quality" by some absolute metric. The survey shows that this is systematically wrong: the optimal selection strategy depends on the *budget*, not just on the data. In the extreme low-data regime (e.g., LIMA's 1,000 examples, Zhou et al., 2024a), careful human curation of diverse, high-quality examples works. In the moderate-data regime (e.g., Alpagasus's 9K examples from Alpaca's 52K, Chen et al., 2023b), quality filtering to remove noise and redundancy is sufficient. In the high-data regime (millions of examples), learning-complexity-based importance sampling becomes necessary because quality filtering alone would either discard too many useful examples or fail to distinguish among the remaining high-quality ones.

This regime shift has a precise operational interpretation in terms of the survey's formalism. When the budget `b` is very small, the selection mechanism `π` dominates: the choice of which specific examples to include matters enormously. When `b` is large, the scoring function `q` dominates: you can afford to be inclusive, so the main challenge is ranking examples correctly. The interaction between `b` and the optimal `q-π` pair is a dimension of the selection problem that prior work largely ignored, treating budget as a simple constraint rather than a variable that changes *which strategy is best*.

**Evidence across methods.** The experimental results in Tables 2–4 demonstrate this regime shift implicitly. IFD-based selection (Table 2) shows that 5% of Alpaca data can match full-dataset performance on many benchmarks, but critically, this holds only because Alpaca is relatively clean and homogeneous. For more diverse datasets like FLAN v2 (Table 2, FL results), quality+diversity methods require larger fractions (20K–45K examples) to approach full-dataset performance. The optimal budget fraction is not a universal constant — it depends on dataset characteristics and the assessment method.

**The survey doesn't solve this problem — it diagnoses it.** Section 7.3 explicitly identifies the "optimal scale of the selected subset" as an open challenge: "such proportion varies from dataset to dataset. When more instruction datasets from diverse domains and tasks are incorporated, it becomes more difficult to nail down the best selection proportion." The contribution is diagnostic: by making the budget-dependence visible across all three axes of the taxonomy, the survey reframes the data selection problem from "find the best data" (an absolute judgment) to "find the best data *for a given budget*" (a conditional optimization) — a more precise and more useful formulation.

---

### Innovation 5: The Missing Integration — Why Importance-Based Assessment Remains Uncoupled

Perhaps the survey's most provocative contribution is a **negative result masquerading as a gap analysis**: the observation that importance-based assessment is systematically absent from hybrid methods, and that this absence is not an accident but reflects fundamental difficulties that the field has not yet addressed.

**What the survey shows.** The paper documents that existing hybrid methods (DEITA, QDIT, LIFT, FL, InstructionMining) combine quality and diversity — through sequential filtering, dynamic weighted maximization, or ensemble indicator computation — but "the importance-based assessment is often overlooked" (Section 6). Only a handful of methods (LESS, MATES) target importance explicitly, and these are standalone approaches that don't integrate quality or diversity filtering.

**Why this absence is structural, not accidental.** The survey's own analysis reveals three deep reasons:

**1. Importance is evaluation-set-dependent.** A datapoint's importance is defined relative to a specific downstream task and evaluation metric. The same example may be crucial for MMLU performance and irrelevant for HumanEval. Quality and diversity, by contrast, are task-agnostic — a well-written, diverse dataset helps regardless of the specific evaluation. This means importance-based selection cannot be done once and reused across tasks; it must be recomputed for each deployment scenario.

**2. Importance is model-dependent.** Gradient-based influence, datamodel-based prediction, and loss-based forgetting all depend on the specific model architecture and parameters. An example that is high-importance for LLaMA-7B may be low-importance for Mistral-7B. The paper notes this explicitly in Section 7.2: "instructions that resemble the most to the testing set or bring about performance gains are judged as 'good' data. However, such 'good' data cannot be easily transferred to another LLM of completely different architecture and parameters." This transferability problem does not affect quality and diversity assessment to the same degree.

**3. Importance estimation is computationally expensive at scale.** Gradient matching requires per-example gradient computation and storage at LLM scale. Influence functions require Hessian approximations. Datamodel training requires multiple model retrainings on different subsets. The paper's discussion of scaling challenges (Section 7.4) notes that "the cost-efficiency of data assessment and selection diminishes with larger LLMs involved in the pipeline," and this is most acute for importance-based methods that require the target model itself to be involved in computation.

**The significance of this diagnosis.** By making these structural barriers explicit, the survey transforms "no one has done this yet" into "here is why this is genuinely hard." This is a more useful contribution than simply proposing yet another hybrid method, because it frames the research agenda: solving the importance-integration problem requires advances in one or more of (a) model-agnostic importance estimation (so importance scores transfer across models), (b) task-agnostic importance estimation (so scores transfer across evaluation sets), or (c) computationally efficient importance approximation (so estimation is feasible at LLM scale with limited resources). Each of these is a substantive research challenge, and the survey provides the conceptual vocabulary for pursuing them.

**Evidence from the experimental results.** The performance of LESS (Table 4) — which achieves near-full-dataset performance with only 5% of data by targeting importance specifically — demonstrates that importance-based selection can be remarkably effective when done well. The fact that LESS with 5% of data (LLaMA2-7B: 5% LESS achieves 50.2% on MMLU vs. 51.6% full) approaches or exceeds what quality+diversity methods achieve with similar fractions (DEITA 10K on LLaMA2-13B: 55.3% MMLU) suggests that importance captures signal that quality and diversity miss. But LESS is compute-intensive, model-specific, and evaluation-set-dependent — it cannot simply be combined with DEITA's quality+diversity pipeline. The survey's framing makes this tension explicit: we know importance matters, we know how to measure it, but we don't know how to integrate it with the other assessment dimensions in a computationally tractable, model-agnostic way.

## 5. Experimental Analysis

### Evaluation Methodology

- **Dataset.** The survey does not conduct its own experiments. It compiles and compares results from prior work evaluated on standard instruction tuning datasets: Alpaca (52K examples, Taori et al., 2023), WizardLM (70K examples), FLAN v2 (variable size, Longpre et al., 2023), UltraChat (~1.3M examples, Ding et al., 2023), OpenOrca (variable, Lian et al., 2023), Dolly (15K examples, Conover et al., 2023), The Pile (825 GB, Gao et al., 2020), Dolma (3T tokens, Soldaini et al., 2024), C4 (750 GB, Raffel et al., 2020), RedPajama (1.2T tokens, Together Computer, 2023), and several custom mixtures (e.g., ShareGPT + UltraChat + WizardLM for DEITA). Training set sizes in the reported experiments range from 15K (Dolly) to billions of tokens (The Pile). The survey's contribution is comparative analysis of existing results, not new experiments.

- **Base model(s).** Reported results span LLaMA 7B and 13B (Touvron et al., 2023a), LLaMA 2 7B and 13B (Touvron et al., 2023b), Mistral 7B (Jiang et al., 2023a), MPT 1B (Team, 2023), Pythia 410M and 1B (Biderman et al., 2023), GPT-Neo 3B (Black et al., 2022), Chinchilla-optimal 1.3B (Hoffmann et al., 2022), StarCoder 15B (Li et al., 2023b), RoBERTa-Base 125M (Liu et al., 2019), and Sheared-LLaMA 1.3B (Xia et al., 2023). The survey does not systematically vary model scale within individual comparisons; each cited paper typically reports results on one or two model sizes. The choice of models reflects what was publicly available and computationally feasible for the original authors, not a controlled scaling study.

- **Metrics.** All reported results use standard downstream benchmark accuracy. The specific benchmarks vary by cited paper and include: **ARC** (Easy and Challenge sets, Clark et al., 2018), **HellaSwag** (Zellers et al., 2019), **MMLU** (Hendrycks et al., 2021), **TruthfulQA** (Lin et al., 2021), **BBH** (Suzgun et al., 2022), **DROP** (Dua et al., 2019), **HumanEval** (Chen et al., 2021), **GSM8K** (Cobbe et al., 2021), **MATH** (Hendrycks et al., 2021), **AlpacaEval** (Li et al., 2023), **MT-Bench** (Zheng et al., 2024), **SuperGLUE** (Wang et al., 2019), **OBQA** (Mihaylov et al., 2018), **PIQA** (Bisk et al., 2020), **CBT** (Hill et al., 2015), **Winogrande** (Sakaguchi et al., 2021), **BoolQ** (Clark et al., 2019), **COQA** (Reddy et al., 2019), **TriviaQA** (Joshi et al., 2017), **SciQ** (Welbl et al., 2017), **LogiQA** (Liu et al., 2020), **MNLI/QNLI/QQP/RTE** (GLUE benchmark, Wang et al., 2018), **SST-2/MRPC/CoLA/STS-B** (GLUE), **TYDIQA** (Clark et al., 2020), **COPA** (Roemmele et al., 2011), and **LAMBADA** (Paperno et al., 2016). For pre-training data selection experiments (The Pile, Dolma, C4, RedPajama), the metric is typically downstream task accuracy after fine-tuning the pre-trained model, not pre-training perplexity. For direct quality assessment of datasets, some methods report correlation between indicator scores and downstream model performance, but these are not standardized.

- **Baselines.** Every cited paper uses **random selection** as the primary baseline — randomly sampling the same number of examples as the proposed method selects. Some additionally compare against **training on the full dataset** to establish the performance ceiling. Specific method-vs-method baselines are less common; the field largely compares each new method against random selection at matched data budgets. The survey's Tables 2–4 report both the random-sampling baseline and (where available) the full-dataset result for each method, enabling direct comparison of efficiency (how much data is needed to match full-set performance) and effectiveness (whether the method outperforms random at the same budget).

- **Generation budget / compute accounting.** The survey does not impose a unified compute accounting framework across the cited methods. Each original paper measures "budget" differently depending on the selection approach: for threshold-based methods (Alpagasus, IFD), the budget is the **number or fraction of examples retained** (e.g., 5%, 10%, 9K examples); for coreset methods (DEITA, QDIT, DQ), the budget is the **subset size `b`**; for importance-resampling methods (DSIR), the budget is expressed in **tokens or examples sampled**; for pre-training data pruning (PPL, DsDm, MATES), the budget is reported in **billions of tokens** or **percentage of the full corpus**. The survey reports these as given by the original papers without attempting to normalize for computational cost of the selection process itself (e.g., GPT-4 API calls, gradient computation, clustering). This is a significant omission — the cost of *performing* the selection is not included in any budget comparison, making efficiency claims potentially overstated (see Critical Assessment).

- **Cross-validation / statistical protocol.** The survey reports results exactly as published in the original papers, without imposing a uniform statistical protocol. Most cited works use standard train/test splits of the downstream benchmarks and report single-run results without confidence intervals or multiple random seeds. The survey does not re-evaluate any method or conduct meta-analyses; it compiles published numbers. The paper notes this limitation implicitly in Section 7.1 when discussing the need for "a benchmark for documenting and comparing the statistics of the selected instruction-response pairs."

### Main Quantitative Results

The survey's experimental analysis consists of three comparison tables (Tables 2–4) that compile published results organized by the three assessment dimensions. The paper does not run new experiments; its contribution is the comparative framing.

#### Quality-Based Selection Results (Table 2)

**Headline finding: Quality-filtered subsets of 5–20% of original data can match or exceed full-dataset performance across multiple benchmarks, with IFD-based selection on Alpaca achieving this at 5% retention (5% of 52K = 2.6K examples).**

Table 2 compiles results from seven quality-focused methods: IFD (Li et al., 2023a), LIFT (Xu et al., 2023b), PPL (Ankner et al., 2024), InstructionMining (Cao et al., 2023), FL (Bhatt et al., 2024), Alpagasus (Chen et al., 2023b), BSDetector (Chen & Mueller, 2024), AutoDS (Zhang et al., 2024c), and QuRator (Wettig et al., 2024).

**IFD on Alpaca and WizardLM (LLaMA 7B and LLaMA2 7B).** On Alpaca with LLaMA 7B, full-dataset training achieves 42.7% ARC, 76.9% HellaSwag, 41.7% MMLU, 39.6% TruthfulQA, and 26.5% AlpacaEval. With only **5% of data** (approximately 2.6K examples), IFD achieves 53.9% ARC (+11.2 percentage points over full), 79.5% HellaSwag (+2.6 pp), and 34.7% AlpacaEval (+8.2 pp). However, MMLU drops from 41.7% to 36.5% (-5.2 pp) and TruthfulQA drops from 39.6% to 38.3% (-1.3 pp). This pattern — **gains on some benchmarks, losses on others** — is consistent across IFD results and suggests that quality filtering may bias the selected subset toward certain task types at the expense of others. On WizardLM with LLaMA 7B, 10% of data (7K examples) achieves 52.9% ARC (vs. 53.1% full), 79.0% HellaSwag (vs. 77.4% full), and 61.4% AlpacaEval (vs. 62.0% full) — near-parity. On LLaMA2 7B, the results are more mixed: 5% of Alpaca data outperforms the full set on ARC (55.8% vs. 54.4%), HellaSwag (57.9% vs. 78.7% — note the full-set HellaSwag is anomalously reported; Table 2 says 78.7% full but this may be a typesetting issue as other IFD results cluster around 79-80%), and TruthfulQA (44.2% vs. 41.0%), but underperforms on MMLU (46.6% vs. 47.0%). The critical pattern: **IFD does not uniformly improve all benchmarks; improvement is task-dependent.**

**LIFT on Open-Platypus and CodeAlpaca (Mistral 7B and StarCoder 15B).** For general instruction tuning (Open-Platypus, Mistral 7B), LIFT-selected 15K examples outperform random 15K by +3.6 pp on ARC (64.3% vs. 60.7%), +2.4 pp on HellaSwag (84.4% vs. 82.0%), and +5.2 pp on TruthfulQA (49.0% vs. 43.8%). For code generation (CodeAlpaca, StarCoder 15B), LIFT 10K dramatically outperforms random 10K on HumanEval (55.0% vs. 38.1%, +16.9 pp) and MBPP (49.5% vs. 43.1%, +6.4 pp). The large gains on code tasks suggest that quality filtering is particularly impactful when the dataset contains substantial noise (as crowdsourced or synthetically generated code data often does).

**PPL on pre-training data (MPT 1B).** Ankner et al. (2024) compare three perplexity-based selection strategies on The Pile and Dolma: selecting the lowest 50% of examples by perplexity ("Low"), the middle 50% ("Mid"), and the highest 50% ("High"). The key finding is that **"Mid" and "High" percentiles consistently outperform "Low"** across all five task categories (World Knowledge, Commonsense Reasoning, Language Understanding, Symbolic Problem Solving, Reading Comprehension). On The Pile: Mid 50% achieves 16.1% World Knowledge vs. 11.1% for Low 50%; High 50% achieves 12.8% Commonsense Reasoning vs. 5.8% for Low 50%. The full dataset achieves 15.5% World Knowledge and 10.3% Commonsense Reasoning, meaning **Mid 50% matches or exceeds full-dataset performance while using half the data**. This confirms that very low perplexity examples — which are highly predictable and thus "easy" — are the least valuable for training, not the most.

**Alpagasus on Alpaca (LLaMA2 7B and 13B).** GPT-3.5 quality scoring followed by retaining only 9K of 52K examples (17% retention). On LLaMA2 7B: Alpagasus 9K achieves 33.8% BBH (vs. 33.0% full), 26.0% DROP (vs. 25.9% full), 12.2% HumanEval (vs. 11.7% full), but 38.8% MMLU (vs. 40.9% full). On LLaMA2 13B: 38.9% BBH (vs. 38.7% full), 34.4% DROP (vs. 33.8% full), 15.9% HumanEval (vs. 15.7% full), but 46.1% MMLU (vs. 47.9% full). **The MMLU deficit is consistent across both model sizes**, suggesting that GPT-3.5 quality filtering removes examples that are specifically valuable for the knowledge-intensive MMLU benchmark, even though those examples may appear lower-quality by surface-level metrics.

**BSDetector on domain-specific tasks (LLaMA2 7B Chat).** On SQuAD-N (reading comprehension with noise), auto-filtering improves from 49.9% (full) to 59.9%, and auto-correcting further improves to 71.4%. On Emails-N, auto-filtering slightly degrades performance (49.7% vs. 50.7% full), but auto-correcting improves to 52.3%. On DROP-N, both filtering (47.4% vs. 44.7% full) and correcting (50.5%) help. This demonstrates that quality assessment can not only filter but also **repair** low-quality examples, achieving gains beyond what filtering alone can provide.

**AutoDS on OpenWebMath (Mistral 7B).** For mathematical reasoning, AutoDS-selected 2.5B tokens outperform random 2.5B tokens on MATH (16.1% vs. 14.3%) and GSM8K (45.4% vs. 44.1%), while maintaining parity on non-math benchmarks (MMLU: 52.3% vs. 52.2%; HellaSwag: 62.7% vs. 62.2%). The specialized quality assessment for mathematical content improves math performance without degrading general capabilities.

**QuRator on QuRatedPajama (Sheared-LLaMA 1.3B).** Qurator-selected 30B tokens outperform random 30B tokens across all three task categories: Sheared RC (52.1% vs. 50.9%), CR (55.5% vs. 55.0%), WK (15.2% vs. 14.9%). Gains are modest (0.3–1.2 pp) but consistent, validating that distilled pairwise quality judgments from GPT-3.5 into a small scorer are effective for pre-training data curation.

**Cross-method comparison within quality-based methods is not possible** from Table 2 alone because each method uses different datasets, different base models, and different evaluation benchmarks. The survey reports results as-is without attempting meta-analytic normalization.

#### Diversity-Based Selection Results (Table 3)

**Headline finding: Diversity-aware methods that combine quality and diversity (DEITA, QDIT) consistently outperform random selection and pure diversity methods (DQ) at equivalent data budgets, with DEITA 6K on Mistral 7B achieving 61.9% MMLU vs. 58.7% for random 10K — 3.2 pp higher with 40% less data.**

Table 3 compiles results from four diversity-focused methods: DEITA (Liu et al., 2023b), ClusterClip (Shao et al., 2024), QDIT (Bukharin & Zhao, 2023), and DQ (Zhou et al., 2023).

**DEITA on mixed datasets (LLaMA 13B, LLaMA2 13B, Mistral 7B).** DEITA selects 6K–10K examples from a 206K mixed corpus (ShareGPT + UltraChat + WizardLM) using a sequential quality-then-diversity pipeline. On LLaMA 13B: DEITA 10K outperforms random 10K on ARC (59.5% vs. 55.8%, +3.7 pp), HellaSwag (82.0% vs. 80.0%, +2.0 pp), and MMLU (60.6% vs. 47.4%, +13.2 pp — the largest gain). On LLaMA2 13B: DEITA 10K vs. random 10K shows 58.9% vs. 61.5% ARC (-2.6 pp, a rare negative), 82.1% vs. 83.7% HellaSwag (-1.6 pp), but 55.3% vs. 55.2% MMLU (essentially tied) and 54.6% vs. 44.8% TruthfulQA (+9.8 pp). On Mistral 7B: DEITA 6K (not 10K) outperforms random 10K on ARC (57.8% vs. 55.4%), HellaSwag (80.3% vs. 79.2%), MMLU (61.9% vs. 58.7%), and TruthfulQA (59.8% vs. 53.6%). **The 6K vs. 10K comparison is particularly notable: DEITA achieves better performance with 40% less data than random selection uses.** The variability across model families (DEITA helps LLaMA 13B more than LLaMA2 13B on some metrics) suggests that data selection effectiveness is model-dependent, though the underlying mechanism for this interaction is not explored in the survey.

**ClusterClip on OpenOrca and Proof-Pile-2 (Mistral 7B and LLaMA2 7B).** ClusterClip performs balanced sampling across k-means clusters. On OpenOrca (Mistral 7B): ClusterClip 5B tokens achieves 64.3% SuperGLUE (vs. 63.0% uniform, 62.1% random), 58.7% GSM8K (vs. 58.8% uniform — essentially tied), 81.4% OBQA (vs. 78.2% uniform), and 6.9 MT-Bench score (vs. 6.75 uniform). On Proof-Pile-2 (LLaMA2 7B): ClusterClip 20B tokens achieves 7.9% MATH (vs. 7.6% uniform), 24.8% GSM8K (vs. 26.0% uniform — a slight regression), 51.1% MMLU (vs. 50.0% uniform), and 42.8% BBH (vs. 42.9% uniform — tied). **ClusterClip's gains are modest (0.2–3.2 pp over uniform) but consistent across most benchmarks**, with the largest improvements on OBQA and SuperGLUE. The near-tie on GSM8K and BBH suggests that balanced clustering helps more on certain task types than others.

**QDIT on multiple datasets (LLaMA 7B).** QDIT dynamically weights quality (GPT scores) and diversity (facility location) during greedy selection. The results span five datasets at different budget levels. On UltraChat: QDIT 10K outperforms random 10K on MMLU (36.1% vs. 32.1%, +4.0 pp), BBH (32.1% vs. 33.2% — slight loss), ARC (60.7% vs. 58.3%, +2.4 pp), DROP (26.7% vs. 26.2%), and SciQ (86.8% vs. 85.4%). On LMSYS: QDIT 10K achieves 37.3% MMLU vs. 33.1% random (+4.2 pp). On the mixed dataset: QDIT 10K achieves 34.3% MMLU vs. 32.9% random. On Dolly (only 1K examples): QDIT 1K dramatically outperforms random 1K on MMLU (33.8% vs. 28.1%, +5.7 pp), BBH (30.3% vs. 27.3%), and DROP (22.6% vs. 17.3%). **The Dolly results are particularly important: at extremely small budgets (1K examples), QDIT's quality-diversity balancing provides the largest relative gains**, suggesting that the compound strategy is most valuable when the subset is too small for random sampling to achieve adequate coverage.

**DQ on Alpaca (LLaMA 7B).** DQ is a pure diversity method (dataset quantization) without explicit quality scoring. Results on Alpaca: full dataset achieves 32.9% BBH, 26.3% DROP, 41.6% MMLU, 10.0% HumanEval. At 20% retention: 32.7% BBH, 26.7% DROP, 39.8% MMLU, 9.2% HumanEval — generally close to full performance. At 2% retention: 32.9% BBH (matches full), 27.6% DROP (exceeds full by +1.3 pp), but 36.6% MMLU (-5.0 pp) and 8.5% HumanEval (-1.5 pp). **Pure diversity can maintain BBH and DROP performance at 2% data, but MMLU degrades substantially**, confirming that diversity alone cannot compensate for the absence of quality filtering on knowledge-intensive benchmarks. This validates the survey's central thesis: diversity and quality capture complementary signals.

**Cross-method comparison within diversity-based methods.** DEITA consistently shows the largest gains over random baselines, likely because it combines explicit quality scoring (via trained complexity and quality models) with diversity filtering. QDIT shows the next-strongest gains, using dynamic tradeoff between GPT quality scores and facility location diversity. ClusterClip and DQ, which are primarily diversity-driven without explicit quality scoring, show more modest improvements. This gradient — hybrid > quality-weighted-diversity > pure diversity — supports the survey's claim that compound strategies outperform single-dimensional ones.

#### Importance-Based Selection Results (Table 4)

**Headline finding: Importance-based methods can achieve near-full-dataset performance with 5–20% of data, with LESS achieving 50.2% MMLU vs. 51.6% full on LLaMA2 7B (5% data) and MATES consistently outperforming random selection by 1–3 pp across diverse benchmarks at 20% data budgets.**

Table 4 compiles results from five importance-focused methods: DsDm (Engstrom et al., 2024), MATES (Yu et al., 2024), DSIR (Xie et al., 2023), Skill-it (Chen et al., 2024b), and LESS (Xia et al., 2024a).

**DsDm on C4 (Chinchilla-optimal 1.3B).** DsDm uses datamodels to select subsets that minimize expected evaluation loss. Results are mixed compared to random selection: on COPA (63.0% vs. 62.0% random), TriviaQA (7.1% vs. 3.7% — a near-doubling), and COQA (25.5% vs. 18.8%), DsDm provides clear gains. On PIQA (69.0% vs. 68.9% — tied), CBT (88.2% vs. 86.4% — marginal), and Winogrande (51.1% vs. 52.2% — a loss). On OBQA, DsDm underperforms random (31.2% vs. 33.4%). **The inconsistency — gains on some tasks, losses on others — is a signature of importance-based methods that optimize for a specific evaluation set:** DsDm was optimized for the average across all reported benchmarks, and individual task performance reflects the distribution of the evaluation set used during selection.

**MATES on C4 (Pythia 410M and 1B).** MATES uses a dynamically updated datamodel that adapts to the model's changing preferences during training. Results at 20% data: on Pythia 410M, MATES outperforms random 20% on SciQ (66.0% vs. 64.1%, +1.9 pp), ARC-E (41.8% vs. 40.2%, +1.6 pp), and LogiQA (25.7% vs. 24.7%), but ties on ARC-C (25.0% vs. 25.6% — a slight loss). On OBQA/BoolQ/HellaSwag/PIQA/Winogrande, MATES 20% on Pythia 410M provides consistent +0.6 to +2.1 pp gains. On Pythia 1B, the pattern holds: MATES 20% outperforms random 20% on SciQ (67.3% vs. 65.8%), ARC-E (44.9% vs. 43.7%), and OBQA (32.2% vs. 31.8%). **The consistent small-to-moderate gains across diverse tasks suggest that dynamic importance estimation captures generalizable value rather than overfitting to specific evaluation examples.** This is notable because most importance-based methods risk evaluation-set overfitting.

**DSIR on The Pile (RoBERTa-Base 125M).** DSIR uses importance-weighted resampling based on hashed n-gram features. At 51.2M examples: DSIR outperforms random on MNLI (83.1% vs. 82.6%), QNLI (89.1% vs. 86.9%, +2.2 pp — the largest gain), QQP (89.8% vs. 89.6%), and RTE (75.1% vs. 67.4%, +7.7 pp — the largest relative improvement). On SST-2 (90.5% vs. 90.1%), MRPC (87.7% vs. 87.4%), CoLA (54.0% vs. 49.4%, +4.6 pp), and STS-B (89.2% vs. 88.6%). **The RTE gain (+7.7 pp) is the single largest improvement over random selection in the importance-based results table**, suggesting that DSIR's distribution-matching approach is particularly effective when the evaluation set has a markedly different distribution from the overall training corpus.

**Skill-it on RedPajama (GPT-Neo 3B).** Skill-it selects data by modeling skill dependencies and prerequisites. At 1B tokens: Skill-it outperforms uniform selection on HellaSwag (63.9% vs. 63.9% — tied), LAMBADA (67.0% vs. 64.4%, +2.6 pp), PIQA (75.0% vs. 74.8%), and Winogrande (63.9% vs. 62.8%). On ARC-C and ARC-E, results are mixed across three random seeds (Table 4 reports three Skill-it runs and three uniform runs): Skill-it shows 34.6–34.9% ARC-C vs. 34.6–35.4% uniform (overlapping ranges), and 61.2–62.0% ARC-E vs. 62.4–65.2% uniform (Skill-it is worse). **The multi-seed reporting reveals variance that single-run results in other tables obscure:** even with the same budget and same method, performance can vary by 1–3 pp depending on random seed. This is a reminder that most results in Tables 2–4 are single-point estimates without confidence intervals.

**LESS on mixed dataset (LLaMA2 7B, LLaMA2 13B, Mistral 7B).** LESS uses low-rank gradient similarity to select training data that matches validation set gradients. On LLaMA2 7B: LESS 5% achieves 50.2% MMLU (vs. 51.6% full, vs. 46.5% random 5%), 56.2% TYDIQA (vs. 54.0% full — **exceeds full-dataset performance by +2.2 pp**), and 41.5% BBH (vs. 43.2% full, vs. 38.9% random 5%). On LLaMA2 13B: LESS 5% achieves 54.0% MMLU (vs. 54.5% full, vs. 53.4% random 5%), 54.6% TYDIQA (vs. 54.3% full), and 50.6% BBH (vs. 50.8% full). On Mistral 7B: LESS 5% achieves 61.8% MMLU (vs. 60.4% full — **exceeds full-dataset performance by +1.4 pp**), 60.3% TYDIQA (vs. 57.7% full — **exceeds full by +2.6 pp**), and 56.0% BBH (vs. 53.0% full — **exceeds full by +3.0 pp**). **The Mistral 7B results are striking: LESS with 5% data outperforms full-dataset training on all three benchmarks.** This suggests that for Mistral 7B specifically, the mixed instruction tuning dataset contains a substantial fraction of unhelpful or counterproductive examples that gradient-based importance estimation successfully excludes.

**Cross-method comparison within importance-based methods.** LESS shows the strongest results (exceeding full-dataset performance in multiple cases), followed by DSIR (large gains on specific tasks like RTE). DsDm and MATES show more modest, task-dependent gains. Skill-it's results overlap substantially with uniform sampling, suggesting that skill-dependency modeling may not provide benefits beyond simpler diversity or importance approaches for the evaluated tasks. The common thread: **importance-based methods work best when there is a clear distribution gap between the training and evaluation data** (DSIR's RTE gain) or when the training data contains actively harmful examples (LESS exceeding full-set performance). When the training and evaluation distributions are already well-aligned, importance-based selection provides only marginal benefits over random sampling (DsDm's mixed results, Skill-it's overlapping ranges).

### Ablation Studies and Robustness Checks

The survey does not conduct its own ablation studies. It reports ablations from the cited papers, primarily in the narrative text of Sections 3–5 rather than in dedicated ablation tables. Key findings on robustness from the cited works include:

- **Perplexity percentile ablation (PPL, Ankner et al., 2024, Section 3.2):** Comparing Low (0–50th percentile), Mid (50th), and High (50th) perplexity-based selection on The Pile and Dolma reveals that **medium-to-high perplexity data is most valuable, while low-perplexity data is least valuable**. On The Pile, Low 50% achieves 11.1% World Knowledge vs. 16.1% for Mid 50% and 18.2% for High 50%. This ablation establishes that the relationship between perplexity and training value is non-monotonic — extremely predictable text is easy for the model and provides little learning signal. The survey notes this as a key finding that generalizes beyond the specific PPL method.

- **PRM aggregation strategy (not applicable — this survey does not cover PRM-based methods; the ablation context is from the reference example, not the present paper).**

- **Proxy model size ablation (multiple papers, Section 3.2 and 7.4):** Li et al. (2024b) demonstrate that GPT2-125M perplexity and IFD scores select data that is effective for training LLaMA2-7B and LLaMA2-13B — models 56×–104× larger. Ankner et al. (2024) use MPT 125M to filter data for MPT 1B (8× larger). The survey compiles these results as evidence for the proxy model strategy but does not report systematic size scaling (e.g., does a 350M proxy outperform a 125M proxy for a 7B target?). The survey's claim that proxy models work is supported by the existence of positive results at specific scale ratios, but the **limits of proxy model transfer** (at what ratio does the signal degrade?) are not characterized.

- **GPT pairwise vs. individual scoring (QuRator, Wettig et al., 2024, Section 3.3):** The QuRator paper's ablation (cited but not tabulated in the survey) compares GPT-3.5 individual scoring against pairwise comparison. The finding that pairwise comparison is more reliable and consistent motivates QuRator's use of pairwise judgments to train a scoring model. The survey mentions this as an important design choice without quantifying the reliability difference.

- **UNCERTAINTY-BASED METHODS UNDERPERFORM RANDOM (Wu et al., 2023, Section 3.2):** A critical negative result reported in the survey: Wu et al. (2023) find that uncertainty-based data sampling methods (entropy, least confidence, mean margin, min margin) **perform worse than random sampling** on Databricks-Dolly, SelfInstruct-Davinci, and SelfInstruct-GPT4. The survey does not provide specific numbers but flags this as evidence that "for instruction tuning specifically, model uncertainty is not a reliable quality signal." This is one of the most important ablations in the survey because it establishes a boundary condition on model-based quality indicators.

- **Sequential vs. parallel hybrid selection (Section 6):** The survey compares the two dominant hybrid paradigms across multiple methods: **sequential** (quality first, then diversity — DEITA) and **parallel** (simultaneous weighting — QDIT). The survey argues that sequential methods "fail to retrieve the candidates that are filtered out in the preceding quality control steps even if those candidates are of high importance or variety." However, this claim is not supported by a direct experimental comparison; DEITA and QDIT are evaluated on different datasets with different models, so the superiority of parallel over sequential is conjectural based on the taxonomy, not empirically demonstrated.

- **Oracle vs. predicted difficulty bins (not applicable — this is a concept from the reference example paper, not the present survey).**

- **Majority voting for revisions (not applicable).**

- **ReST^EM revision model (not applicable).**

- **Datamodel linearity assumption (DsDm, Section 5.2):** DsDm uses linear datamodels (TARK estimator) to predict subset performance. The survey does not report ablation comparing linear vs. non-linear datamodels. This is a significant gap because real training dynamics are non-linear (example interactions, diminishing returns), and the linearity assumption may degrade for larger models or more diverse datasets.

- **Hashed n-gram vs. neural features for importance (DSIR, Section 5.2):** DSIR's use of cheap hashed n-gram features (rather than neural embeddings or trained datamodels) is itself an ablation relative to more expensive methods, but the survey does not report DSIR's own ablation comparing n-gram features against embeddings. The claim that "cheap approximation... achieves similar performance" is based on DSIR's overall results compared to more expensive methods, not a controlled within-method feature comparison.

### Critical Assessment

The survey's central claim is that data assessment and selection methods for instruction tuning can be organized into quality-, diversity-, and importance-based categories, and that understanding these categories enables better selection strategies. The experimental evidence compiled in Tables 2–4 provides **strong support for the existence and distinctiveness of these categories** but **weaker support for the survey's prescriptive claims** about how methods should be combined or compared.

**Does the evidence support that quality-, diversity-, and importance-based methods capture distinct, complementary signals?**

The evidence for complementarity is primarily indirect, drawn from cross-method comparison patterns rather than controlled experiments:

- **Quality-only methods (IFD, Alpagasus)** achieve strong performance at very small data fractions (5–17%) on individual datasets, but show task-dependent degradation (IFD on MMLU, Alpagasus on MMLU). This suggests quality filtering alone biases selection toward certain task types — exactly the failure mode that diversity-aware methods are designed to address.

- **Diversity-only methods (DQ)** maintain performance on some benchmarks (BBH, DROP) at extreme compression (2% data) but lose substantially on knowledge-intensive benchmarks (MMLU drops 5 pp at 2%). This confirms that diversity without quality is insufficient.

- **Hybrid quality+diversity methods (DEITA, QDIT)** consistently outperform both random selection and pure quality or diversity methods at equivalent budgets. The gains are not uniform (DEITA's MMLU gain is 13.2 pp on LLaMA 13B but only 0.1 pp on LLaMA2 13B) but the direction is consistent.

- **Importance-based methods (LESS, DSIR)** achieve the most dramatic improvements on specific benchmarks (LESS exceeds full-dataset performance on Mistral 7B by 1.4–3.0 pp; DSIR improves RTE by 7.7 pp), suggesting they capture signal — alignment with evaluation distribution — that neither quality nor diversity alone captures.

**However, the evidence for complementarity is weaker than it appears because no experiment directly manipulates the three axes orthogonally.** Every hybrid method (DEITA, QDIT, LIFT) bakes in specific choices about *how* quality and diversity are combined (sequential vs. parallel, specific scoring functions, specific thresholds). The fact that DEITA outperforms DQ could be because DEITA adds quality scoring, or because DEITA uses a better diversity mechanism (Repr Filter vs. dataset quantization), or because DEITA's datasets and models differ from DQ's. Without experiments that vary the combination strategy while holding the underlying quality and diversity metrics constant, the survey's claim that the axes are complementary (rather than simply that more sophisticated methods work better) remains an interpretation, not an established fact.

**Does the evidence support that the taxonomy enables better selection in practice?**

The survey provides no head-to-head comparison showing that a taxonomy-informed selection strategy outperforms an ad-hoc one. The practical value of the taxonomy is asserted but not experimentally validated. For a practitioner reading this survey, the taxonomy provides a vocabulary for describing methods and a framework for thinking about tradeoffs, but it does not provide actionable guidance like "for dataset X with model Y at budget Z, use quality assessment method A followed by diversity method B." The survey acknowledges this limitation implicitly in Section 7.2: "the optimal solution to establishing an overall data assessment and selection pipeline still remains an open question."

**Significant weaknesses in the experimental evidence base:**

1. **No standardization across methods.** Every row in Tables 2–4 uses different datasets, different base models, different training configurations, and different evaluation benchmarks. Comparing IFD on Alpaca with LLaMA 7B against DEITA on a mixed corpus with LLaMA 13B tells us about the individual methods but not about whether quality-based selection is "better" or "worse" than diversity-based selection in any general sense. The survey does not attempt to control for these confounds or to re-evaluate methods under standardized conditions.

2. **Selection cost is never accounted for.** The compute required to perform data selection — GPT-4 API calls for Alpagasus (~$100+ for 52K examples), gradient computation for LESS (multiple forward-backward passes through a 7B model), k-means clustering over millions of embeddings for DEITA, pairwise comparisons for QuRator — is excluded from all budget calculations. A method that achieves 95% of full-dataset performance with 5% of the training data but requires 10× the training budget in selection cost may be a net loss. The survey acknowledges this in Section 7.3 but does not quantify it for any method. This makes all efficiency claims (e.g., "$4\times$ improvement") conditional on selection cost being negligible, which is false for many methods discussed.

3. **Single-run results without confidence intervals.** Almost all numbers in Tables 2–4 are single-point estimates. The Skill-it results in Table 4 are a rare exception, showing 3-run ranges that reveal 1–3 pp variance. If this level of variance is typical (which it likely is for small evaluation sets and small training subsets), then many of the reported "gains" of 1–2 pp over baselines are not statistically distinguishable from noise. The survey does not address this.

4. **Missing baselines.** No method is compared against a simple ensemble of quality and diversity indicators (e.g., "take the top 50% by IFD, then run k-center greedy"). This is the most natural baseline for hybrid methods and its absence makes it impossible to assess whether the sophisticated combination strategies in DEITA and QDIT are genuinely necessary.

5. **Missing scale axis.** The survey compiles results from models ranging from 125M (RoBERTa-Base) to 13B (LLaMA2) but does not analyze how selection effectiveness varies with model scale. Section 7.4 hypothesizes that "the same quality measurement and data selection pipeline can achieve similar performance gains on both small and large LLMs" but acknowledges this is unknown. The compiled tables cannot answer this question because no method is evaluated across a wide range of model scales under controlled conditions.

6. **Evaluation benchmark dependency.** The benefits of importance-based methods (LESS, DSIR) are inherently evaluation-set-dependent — they optimize for a specific target distribution. The survey reports LESS's results on the benchmarks it was optimized for but does not report whether LESS-selected data degrades performance on *unseen* benchmarks (a likely failure mode given the mechanism). This is a critical missing experiment for any claim that importance-based selection is generally beneficial.

7. **Test set contamination risk.** Section 7.1 discusses the risk that pre-trained models may have already seen evaluation examples during pre-training, contaminating fine-tuning evaluations. None of the reported results control for this possibility, yet the survey itself flags it as a serious concern. If contamination is present, methods that select training data similar to evaluation data (LESS, DSIR) may simply be selecting contaminated examples, inflating their apparent effectiveness.

**What experiments would have strengthened the survey's claims?**

A standardized benchmark evaluating 5–10 representative methods (one per taxonomic cell) on the same dataset, same base model, same training protocol, and same evaluation suite, across multiple budget levels and with selection cost accounted for, would transform the survey's organizing framework from a plausible interpretation into an empirically validated theory. The survey's authors likely recognized this but chose to publish the taxonomy as a review paper rather than conduct such experiments, which is a defensible choice for a survey but limits the strength of its prescriptive claims. The most important missing experiment is the one the survey itself calls for: a "benchmark for documenting and comparing the statistics of the selected instruction-response pairs in terms of quality, diversity, and importance" (Section 7.1). Until such a benchmark exists and methods are evaluated on it, the taxonomy remains a useful conceptual tool but not a validated engineering framework.

## 6. Limitations and Trade-offs

### The Cost of Selection Is Never Accounted For

**The assumption or constraint.** Every efficiency claim in this survey — that IFD achieves full-dataset performance with 5% of Alpaca data (Table 2), that LESS exceeds full-dataset accuracy with 5% of a mixed corpus (Table 4), that DEITA's 6K examples outperform random 10K (Table 3) — treats the selection process itself as cost-free. The "budget" in every reported experiment counts only the number of training examples retained or the number of fine-tuning tokens consumed. The computational cost of *computing* IFD scores (requiring model warm-up and inference on all examples), GPT-4 API calls (Alpagasus, LIFT, QuRator), gradient computation and Hessian approximations (LESS, influence functions), k-means clustering over millions of embeddings (DEITA, ClusterClip), or training datamodels on hundreds of random subsets (DsDm) is excluded from every budget calculation in Tables 2–4.

The authors acknowledge this in Section 7.3 when discussing the difficulty estimation cost from prior work: generating 2048 samples per question to estimate difficulty is "extraordinarily expensive" and often "exceeds the budget of the actual fine-tuning step." Yet the survey itself never quantifies selection cost for any of the methods it reviews, nor does it note that some methods (GPT-4 scoring, gradient-based influence for large models) have selection costs that may exceed the training cost savings they claim to provide.

**The consequence.** A practitioner reading that "5% of data matches full-dataset performance" may reasonably conclude that using the method will reduce their total compute expenditure by ~95%. This is false for most methods discussed. The true cost is selection cost + training cost, and for many methods — particularly those involving GPT-4 API calls at scale, gradient computation through large models, or iterative retraining for datamodel estimation — the selection cost can dominate. For example, GPT-4 scoring 52K Alpaca examples at current API pricing costs approximately $100–200, which is modest compared to training LLaMA 7B, but scoring millions of examples for pre-training data curation (as QuRator does for 260B tokens) becomes a substantial fraction of the training budget. For gradient-based methods, computing LESS's low-rank gradient similarities across a 270K-example mixed corpus with a 7B model requires multiple forward-backward passes and substantial GPU-hours — costs that are never reported or amortized.

The consequence is that the headline efficiency numbers in Tables 2–4 are **upper bounds on practical efficiency**, not realized savings. A method that appears to offer 20× data reduction may offer only 2× total compute reduction once selection overhead is included, or may even be a net loss. Without selection-cost accounting, practitioners cannot make informed decisions about which method to deploy.

**What evidence exists in the paper.** The survey provides no direct evidence on this limitation — no table or figure reports selection cost for any method. The limitation is discussed qualitatively in Section 7.3: "The optimization of a scalable pipeline for data assessment and selection is of urgent need." The authors note that "cheap approximation of features by bag-of-n-grams achieves similar performance but requires much less computing resources" (Section 7.3, discussing DSIR) and recommend that "one may draw inspiration from the data deduplication and filtering techniques in handling billions of pre-training tokens." These are suggestions for future work, not evidence that the limitation is addressed. Section 7.4 further notes that "the cost-efficiency of data assessment and selection diminishes with larger LLMs involved in the pipeline," but this warning is not reflected in any of the quantitative results.

**Mitigation status.** The limitation is acknowledged but not addressed. The survey identifies cheap proxy models (Section 7.4) and dimensionality reduction (Section 7.3) as potential mitigation strategies, and DSIR's hashed n-gram approach is highlighted as an existence proof that cheap features can work. However, no cost analysis is performed for any method, and the survey does not recommend that future work standardize selection-cost reporting. A practitioner reading this survey has no way to compare the total cost of two methods with different selection overheads.

---

### No Standardized Evaluation Exists, Making Cross-Method Comparison Unreliable

**The assumption or constraint.** The survey compiles results from dozens of independent papers, each of which uses different base models (LLaMA 7B, LLaMA2 7B, LLaMA2 13B, Mistral 7B, MPT 1B, Pythia 410M and 1B, GPT-Neo 3B, Chinchilla-optimal 1.3B, StarCoder 15B, RoBERTa-Base 125M, Sheared-LLaMA 1.3B), different training datasets (Alpaca, WizardLM, FLAN v2, UltraChat, OpenOrca, Dolly, The Pile, Dolma, C4, RedPajama, and several custom mixtures), different training protocols (learning rates, batch sizes, numbers of epochs are uncontrolled across studies), and different evaluation benchmarks (each paper reports results on a different, often non-overlapping subset of the available benchmarks). There is no standardized testbed where multiple methods are evaluated under identical conditions.

The authors acknowledge this directly in Section 7.1: "a benchmark for documenting and comparing the statistics of the selected instruction-response pairs in terms of quality, diversity, and importance needs to be constructed in the future." Until such a benchmark exists, "it is impractical to simply count on losses or gradients to pinpoint the most beneficial data."

**The consequence.** The survey's central organizing claim — that quality, diversity, and importance represent distinct, complementary axes of data assessment — cannot be rigorously validated from the compiled evidence. A result showing that DEITA (hybrid quality+diversity) outperforms DQ (pure diversity) could be because hybrid methods are genuinely superior, or because DEITA used LLaMA 13B with a 206K mixed corpus while DQ used LLaMA 7B with 52K Alpaca. The models differ in scale (13B vs. 7B), training data distribution, and inherent capability; the datasets differ in size, composition, and noise level. Any of these confounds could explain the performance difference independently of the selection method.

This makes the survey **taxonomically useful but prescriptively weak**. A practitioner cannot consult Tables 2–4 to answer: "Which method should I use for my dataset and model?" The tables show that many methods work, but not which works best, or under what conditions. The taxonomy provides a vocabulary for describing methods but not a decision procedure for choosing among them.

The problem is compounded by the absence of confidence intervals or multi-seed results for most entries. The Skill-it results in Table 4 are a rare exception, showing 3-run ranges for ARC-C (34.6–34.9% for Skill-it vs. 34.6–35.4% for uniform) that overlap substantially — meaning the reported differences are within run-to-run variance. If 1–3 pp variance is typical (and it almost certainly is for 500-question test sets with 5%-sized training subsets), many of the 1–2 pp "improvements" reported in Tables 2–4 are not statistically distinguishable from noise. The survey does not address this.

**What evidence exists in the paper.** The evidence for this limitation is the tables themselves. A careful reader will notice that IFD is evaluated on ARC, HellaSwag, MMLU, TruthfulQA, and AlpacaEval (Table 2, Alpaca/WizardLM rows), while DEITA is evaluated on ARC, HellaSwag, MMLU, and TruthfulQA (Table 3), while LESS is evaluated on MMLU, TYDIQA, and BBH (Table 4). The overlap in evaluation benchmarks across methods is partial and inconsistent. The overlap in training datasets is essentially zero — each method uses different source data. This makes any cross-method comparison (e.g., "DEITA achieves 61.9% MMLU vs. LESS's 61.8% MMLU") meaningless because the underlying training data and model families differ. The survey does not attempt to control for these confounds or to perform meta-analytic normalization.

**Mitigation status.** The limitation is acknowledged in Section 7.1 as a call for future benchmarks but is not addressed in the present work. The survey's contribution is the taxonomy and the systematic compilation, not the resolution of this evaluation gap. The authors are transparent about this: Section 7.1 states that "the evaluation of instruction-tuned models should be accompanied by the specialised evaluation of the selected datapoints" and that "a benchmark... needs to be constructed in the future." This is honest but leaves the survey's prescriptive claims unsupported.

---

### Importance-Based Assessment Is Fundamentally Incompatible with Transferability, Yet Hybrid Methods Require It

**The assumption or constraint.** The survey's taxonomy reveals a structural tension that it does not resolve: quality and diversity are (mostly) model-agnostic and task-agnostic assessment dimensions, while importance is inherently model-dependent and evaluation-set-dependent. A datapoint's IFD score or GPT-4 quality rating is a property of the datapoint itself, largely invariant to which model will be fine-tuned on it. A datapoint's embedding-based diversity relative to other datapoints is similarly model-agnostic (assuming a fixed sentence encoder). But a datapoint's importance — measured via gradient matching, influence functions, datamodel-based loss prediction, or necessity scoring — is defined relative to a specific model's current parameters and a specific evaluation distribution.

The authors identify this tension explicitly in Section 7.2: "instructions that resemble the most to the testing set or bring about performance gains are judged as 'good' data. However, such 'good' data cannot be easily transferred to another LLM of completely different architecture and parameters. Each time the entire pipeline has to be enforced for a novel task, making it difficult to accumulate universally-acknowledged high-quality data for archiving."

**The consequence.** The survey's aspiration — a unified framework that integrates quality, diversity, and importance into a single selection pipeline — faces a fundamental obstacle. Quality and diversity assessment can be done once and reused across models and tasks. Importance assessment must be redone for each model-task pair. This means that any hybrid method that incorporates importance is inherently **non-transferable**: the selected subset that optimally trains LLaMA 7B for MMLU performance may be suboptimal (or even harmful) for training Mistral 7B for MMLU, or for training LLaMA 7B for HumanEval, or for archiving as a generally-useful dataset.

This has practical consequences. The survey documents that LESS (Table 4) achieves remarkable results — exceeding full-dataset MMLU performance with 5% of data on Mistral 7B — but these results are specific to (Mistral 7B, the mixed training corpus, the MMLU/TYDIQA/BBH evaluation suite). If a practitioner wants to use LESS-selected data to train a different model or to improve performance on a different benchmark, they cannot reuse the existing selection; they must recompute it from scratch. For gradient-based methods at 7B+ scale, this recomputation cost may be prohibitive.

The survey also documents (Section 6) that "the importance-based assessment is often overlooked in existing hybrid approaches." This is not an oversight by prior work — it reflects the genuine difficulty of integrating a model-dependent, task-dependent assessment dimension with model-agnostic, task-agnostic ones. The survey's taxonomy makes this difficulty visible but does not provide a resolution strategy.

**What evidence exists in the paper.** The evidence is primarily the experimental pattern across Tables 2–4. Quality-based methods (IFD, Alpagasus, PPL) and diversity-based methods (DEITA, QDIT, ClusterClip) report results on standard benchmarks (ARC, HellaSwag, MMLU, TruthfulQA) that are intended to represent general instruction-following capability — they do not claim that the selected data is optimal for any specific task. Importance-based methods (DsDm, LESS, DSIR) report results on specific evaluation suites, and their mechanism explicitly optimizes for those suites. LESS's results on Mistral 7B (61.8% MMLU vs. 60.4% full) are impressive but do not establish that the selected subset would also improve, say, code generation or mathematical reasoning. The DSIR results (Table 4) show the clearest pattern: DSIR achieves the largest single-method gain on RTE (+7.7 pp over random), a specific natural language inference task, but its gains on other GLUE tasks are modest (+0.3–2.2 pp). This is exactly what we would expect from a method that resamples training data to match an evaluation distribution — it helps most on tasks where the evaluation distribution differs most from the training distribution, and less on tasks where they are already aligned.

**Mitigation status.** The survey does not attempt to resolve this tension. Section 7.2 suggests that "more unified, generally applicable definitions on 'good' datapoints in terms of fine-grained aspects" are needed, and Section 7.5 calls for integrating importance with quality and diversity as future work. The survey's contribution is diagnostic: by making the transferability problem visible through the taxonomy, it frames a research challenge. But the challenge itself — developing importance estimation that is model-agnostic or cheaply recomputable — is unsolved and may be fundamentally difficult given that importance is, by definition, a measure of how much a specific model benefits from a specific example.

---

### Hard Problems and Out-of-Distribution Tasks Are Not Addressed: Selection Cannot Create Capability

**The assumption or constraint.** Every method reviewed in this survey operates under a fundamental assumption: the full training dataset contains at least some examples that are useful for the target task, and the selection method's job is to find them while discarding noise, redundancy, and harmful examples. This assumption fails when the evaluation task requires capabilities that are genuinely absent from the training data. If the base model has zero (or near-zero) capability on a task, and the training data contains no examples that would teach that capability, no selection method can help — there are simply no useful examples to select.

The survey acknowledges this in Section 7.4 when discussing the diminishing benefits of instruction tuning for larger models: "if the pre-trained LLMs are in lack of the prerequisite knowledge, the instruction tuning cannot properly activates the parameterized 'memory' for alignment but only causes overfitting of the given prompt. In that case, the benefits of data selection are limited with poor generalizability." This directly parallels a key finding from the compute-optimal test-time scaling literature: test-time compute amplifies existing capability but does not create it.

**The consequence.** The survey's framework is powerful for *refining* existing capabilities — taking a model that already has some instruction-following ability and selecting data that improves it efficiently. But it provides no guidance for *acquiring new capabilities* — teaching a model to perform tasks for which no training data exists or for which the base model has near-zero pass@1. This is a critical boundary condition for practitioners. If you are fine-tuning a model on a genuinely novel domain (e.g., a specialized scientific reasoning task, a low-resource language, a novel output format), the training data may contain few high-quality examples, and selection methods that filter for quality, diversity, or importance may inadvertently discard the rare examples that actually provide the target capability.

The survey's own evidence shows this boundary indirectly. In Table 2, IFD-based selection on Alpaca with LLaMA 7B improves ARC from 42.7% (full) to 53.9% (5% selected) — a task the model already performs reasonably well on — but degrades MMLU from 41.7% to 36.5%. On WizardLM with LLaMA 7B, 10% IFD selection achieves 52.9% ARC (near full-dataset 53.1%), but MMLU drops from 37.8% to 33.1%. This pattern — some benchmarks improve while others degrade — is consistent across quality-based methods and suggests that what constitutes "high-quality" data is not uniform across tasks. If the model struggles with a particular task type, filtering for general quality may remove the specific examples that would help on that task, because those examples may appear lower-quality by surface metrics (e.g., more complex, less fluent, or from a different distribution).

For importance-based methods, the consequence is more subtle. LESS can exceed full-dataset performance on MMLU by selecting examples that are specifically influential for MMLU (Table 4). But this success comes at the cost of specialization: the same selected subset is unlikely to be optimal for other tasks. If the target task distribution changes (e.g., you later need to add code generation capability), the LESS selection must be redone. There is no general-purpose "important" subset; importance is always importance-for-something.

**What evidence exists in the paper.** The survey itself is a review and does not conduct experiments on out-of-distribution generalization or capability acquisition. The evidence for this limitation is the observed benchmark-level tradeoffs in Tables 2–4: methods that improve some benchmarks often degrade others (IFD on MMLU, Alpagasus on MMLU, DQ on MMLU, DSIR's uneven GLUE gains). The survey does not analyze these tradeoffs in detail, nor does it discuss whether they represent a fundamental limitation or an artifact of specific method implementations. Section 7.4 discusses the related problem of "forgetting" during continual fine-tuning but does not address the more fundamental issue of whether selection can help when the target capability is entirely absent from the training data.

**Mitigation status.** The limitation is not addressed. The survey's framework is explicitly designed for data selection from existing instruction tuning datasets and does not claim to solve the problem of generating or acquiring new training data for novel capabilities. This is a scope limitation rather than a flaw: the survey studies how to select from available data, not how to create data for capabilities the model lacks. However, practitioners should understand this boundary — the survey's methods are tools for efficient alignment, not for expanding the frontier of what a model can do.

---

### The Survey's Own Comparative Claims Are Not Empirically Validated

**The assumption or constraint.** The survey makes several prescriptive claims that go beyond taxonomy:

- "Hybrid methods that combine multiple aspects of data assessment... consistently outperform single-dimensional methods" (Section 6 and the Executive Summary)
- "The sequential setups... fail to retrieve the candidates that are filtered out in the preceding quality control steps even if those candidates are of high importance or variety. Therefore, it would be preferred to develop hybrid assessment techniques that simultaneously weigh quality, diversity, and importance" (Section 6)
- "Importance-based assessment is often overlooked in existing hybrid approaches, implying that the investigation of integrating importance with quality and diversity is of high potentials for future studies" (Section 6)

These claims are plausible interpretations of the evidence compiled in Tables 2–4, but they are not experimentally validated by the survey itself. The survey does not conduct a controlled experiment comparing a quality-only method, a diversity-only method, a quality+diversity method, and a quality+diversity+importance method on the same dataset with the same model. No such experiment exists in the literature that the survey cites, either — the hybrid methods (DEITA, QDIT) were never compared head-to-head with importance-based methods (LESS, DSIR) under standardized conditions.

**The consequence.** The survey's most actionable claim — that practitioners should use hybrid methods combining multiple assessment dimensions — rests on cross-study comparison rather than controlled evidence. The observed pattern that DEITA > Alpagasus > random could reflect better engineering, more favorable datasets, or more optimized hyperparameters rather than the fundamental superiority of combining quality with diversity. Because the underlying quality metrics, diversity mechanisms, and training protocols all vary simultaneously, the causal attribution is ambiguous.

This is a methodological limitation common to survey papers — they can observe patterns in the literature but cannot establish causal relationships. The survey's authors are generally careful about this, using hedging language like "the importance-based assessment is often overlooked... implying that the investigation... is of high potentials." But the Executive Summary's claim that "hybrid quality-and-diversity approaches... consistently outperform single-dimensional methods" is stated without qualification, overstating what the evidence supports.

A related issue: the survey does not examine negative results or failure modes systematically. The only prominent negative result reported is Wu et al. (2023)'s finding that uncertainty-based sampling underperforms random — a critically important boundary condition. But other plausible negative results (e.g., does combining quality and diversity ever *hurt* compared to quality alone? does importance-based selection ever overfit catastrophically to the evaluation set?) are not surveyed, either because they don't exist in the literature or because the survey chose not to highlight them. This gives an asymmetric picture of method reliability.

**What evidence exists in the paper.** The evidence for this limitation is the structure of the paper itself. Tables 2–4 present results organized by assessment dimension, but the comparisons across tables are left to the reader's interpretation. Section 6 provides qualitative discussion of cross-method patterns but no quantitative meta-analysis. The survey does not report effect sizes, statistical significance, or confidence intervals for its comparative claims. A reader who wants to know *how much* better DEITA is than Alpagasus, or whether the difference is statistically reliable, cannot determine this from the survey.

**Mitigation status.** The limitation is inherent to the survey format and is partially mitigated by the authors' honesty about open challenges. Section 7.1 explicitly calls for standardized benchmarks, and Section 7.2 acknowledges that "there exists no unified criteria on discriminating 'good' instructions from 'bad' ones." The survey's primary contribution is the taxonomy and the systematic compilation, not the empirical validation of comparative claims. The hedging in the body text is more careful than in the Executive Summary. However, the gap between what the evidence supports and what the summary claims remains a weakness that a practitioner should note — the taxonomy is a useful framework, but the claim that hybrid methods "consistently outperform" single-dimensional ones should be treated as a hypothesis to be tested under controlled conditions, not as an established finding.

---

### The Taxonomy Does Not Address Dynamic or Curriculum-Based Selection

**The assumption or constraint.** Every method in the survey performs **static, one-shot selection**: all data assessment happens before training begins, and the selected subset `S_b` is fixed for the entire fine-tuning process. The scoring function `q(x_i)` is computed once per example, and the selection mechanism `π` produces a single subset that does not change during training. This is explicit in the survey's formalization (Section 2, Eq. 2–4) and in every method described in Sections 3–5.

The survey acknowledges that this is a limitation when discussing MATES (Section 5.2), which is the one exception: "MATES where a small datamodel continuously selects the most effective subset for the current training of the LLM. The datamodel, like a partner, is updated alternatively to adapt to the constantly changing preferences of the model under development." But MATES is presented as a specialized importance-based method; the survey does not generalize from it to a broader critique of static selection, nor does it discuss how quality- or diversity-based methods might be adapted to dynamic selection.

**The consequence.** Static selection ignores a well-established finding in curriculum learning and training dynamics: the optimal data for a model changes as the model learns. Examples that are confusing or harmful early in training (when the model has weak capabilities) may become valuable later (when the model needs challenging examples to refine its understanding). Conversely, examples that are essential early (to establish basic instruction-following) may become redundant later. A fixed subset `S_b` selected before training cannot adapt to these shifting needs.

This limitation interacts with the survey's own taxonomy in subtle ways. The finding from Jiang et al. (2024c) that "in a data-poor regime, easy datapoints are more informative and should be kept first. On the contrary, in a data-rich regime, hard datapoints should be treasured" (Section 3.2) suggests that the optimal selection strategy depends on how much data the model will see. But it also implies that within a single training run, the model transitions from a data-poor regime (early training, limited effective capacity) to a data-rich regime (late training, saturated on easy examples). A static subset cannot capture this transition — selecting mostly easy examples may help early but leave the model undertrained on hard cases later; selecting mostly hard examples may overwhelm the model early and slow convergence.

For importance-based methods, the consequence is more direct. LESS (Section 5.4) selects training data whose gradients align with a validation set's gradients — but gradients change as the model is trained. The gradients computed from the *initial* model (which is what LESS uses) may not reflect which examples would be influential for the *partially trained* model. The survey does not discuss whether this mismatch degrades importance-based selection, or whether iterative recomputation (a la MATES) would help.

**What evidence exists in the paper.** The survey provides no experimental evidence on dynamic vs. static selection — this is a conceptual limitation of the framework, not an empirical finding. The MATES results (Table 4) show that dynamic datamodel-based selection modestly outperforms random selection (by 0.6–2.1 pp on Pythia 410M, similar margins on Pythia 1B), but these results are not compared against a static version of the same method, so the marginal benefit of dynamism is unknown. The survey does not discuss whether the gains from MATES (over random) are due to the datamodel's importance estimation, the dynamic updating, or both.

Section 7.3 briefly touches on curriculum effects when discussing continual fine-tuning: "the forgetting becomes a severe problem when more instruction datasets are introduced without setting a proper re-playing schedule." But this discussion is about catastrophic forgetting across datasets, not about within-dataset curriculum optimization. The survey does not propose dynamic selection as a solution to curriculum challenges, nor does it analyze why most methods treat selection as static.

**Mitigation status.** The limitation is not addressed. The survey's formalism (`S_b = π(S, b, q)`) is inherently static — it produces a single subset, not a schedule of subsets. Extending the formalism to handle dynamic, curriculum-based selection (where `S_b` changes at each training step `t`, and `q` may depend on the model's current state) would require significant changes to the framework. The survey identifies this implicitly by highlighting MATES as an outlier, but does not discuss how the quality and diversity axes could be adapted to dynamic selection. Given that the paper's primary audiences are practitioners building selection pipelines, the absence of discussion on whether static selection is sufficient — or when dynamic selection becomes necessary — is a meaningful gap.

## 7. Implications and Future Directions
- How this work changes the landscape
  - Provides a common language and toolkit: practitioners can design selection pipelines by choosing metrics on one or more axes (quality/diversity/importance) and plugging them into selection mechanisms (greedy, probabilistic, clustering, bilevel) with clear equations (Eqs. 2–5, 7–8, 42, 45).
  - Encourages moving beyond “more data is better” toward “the right data is better,” with evidence that 5–10% selected subsets can rival or beat full datasets (Tables 2–4).

- Follow‑up research enabled
  - Better hybrid objectives: learn task‑specific weights that combine `quality`, `diversity`, and `importance` end‑to‑end (not just sequential filters), possibly via bilevel methods that optimize downstream metrics directly (Eq. 45; §6).
  - Contamination‑aware selection: automated detection of leakage and decoupled evaluation protocols (§7.1).
  - Scalable proxies: lightweight, well‑calibrated scorers (e.g., small LMs, random projections, hashed features) that approach the fidelity of heavy metrics (§7.4; DSIR Eq. 52).
  - Fairness‑aware selection: integrate WEAT/SEAT, DisCo, and generation‑bias measures into the ‘quality’ axis and report bias‑aware diversity (§7.5).

- Practical applications
  - Cost‑efficient fine‑tuning for enterprise copilots: Use perplexity/IFD to filter noisy instruction logs; cluster to ensure coverage; add importance signals (forgetting or gradient similarity) for target tasks.
  - Domain specialization: For code/math/medical assistants, select high‑necessity items (Eq. 48) where the current model underperforms; enforce semantic diversity to avoid prompt overfitting (§7.3).
  - Continuous data operations (“DataOps” for LLMs): Apply datamodel‑guided selection (Eqs. 49–50) or MATES‑style online selection to new data streams with periodic retraining (Table 4).

> Overall message evidenced by Figures 1–2 and Tables 2–4: thoughtful data assessment and selection, grounded in explicit metrics and implemented with principled selection mechanisms, consistently delivers better alignment and generalization than indiscriminate scaling—often at a fraction of the data and compute.
