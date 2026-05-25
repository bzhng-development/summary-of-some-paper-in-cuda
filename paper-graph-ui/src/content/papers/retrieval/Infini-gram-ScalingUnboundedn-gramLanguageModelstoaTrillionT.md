# Infini-gram: Scaling Unbounded n-gram Language Models to a Trillion Tokens

**ArXiv:** [2401.17377](https://arxiv.org/abs/2401.17377)

## 🎯 Pitch

This paper introduces the ∞-gram language model—a modern, unbounded n-gram approach—and the infini-gram engine, enabling state-of-the-art n-gram modeling over trillions of tokens via an efficient suffix array architecture. By allowing for arbitrarily large contexts and making n-gram statistics instantly queryable at massive scale, Infini-gram demonstrates that classical n-gram LMs are not only still relevant but can significantly improve neural language models, reducing perplexity by up to 73%. This work redefines the role of n-gram models in the era of neural LLMs by empowering transparent text analysis, robust data inspection, and practical hybrid modeling at unprecedented scale.

---

## 1. Executive Summary

This paper modernizes the classical n-gram language model by training it on 5 trillion tokens — the largest n-gram LM ever built — and extending $n$ to be arbitrarily large via a new **∞-gram LM with backoff**, powered by an efficient suffix-array engine called **infini-gram** that computes probabilities with millisecond-level latency. The ∞-gram LM achieves 47% next-token prediction accuracy on human-written text, and when interpolated with neural LMs (e.g., combining sparse/non-sparse ∞-gram estimates with models like Llama-2), it reduces perplexity by up to 73% relative to the neural LM alone, even for a 70B-parameter model. When analyzing machine-generated text, ∞-gram reveals that greedy decoding produces irregular agreement fluctuations with respect to suffix length — a phenomenon not observed under nucleus sampling — establishing that ∞-gram can surface deficiencies in neural LM pretraining and Transformer positional embeddings while complementing neural LMs most effectively on tokens where the neural model itself performs poorly.

## 2. Context and Motivation

### The Core Problem: Has the Field Prematurely Abandoned n-gram Language Models?

This paper confronts a striking asymmetry in modern NLP: neural large language models (LLMs) have been scaled to trillions of training tokens with enormous success, but classical n-gram language models — the dominant paradigm for decades before deep learning — have been left behind at a tiny fraction of that scale. The largest n-gram table ever constructed prior to this work, by Brants et al. (2007), indexed 5-grams over approximately 2 trillion tokens. Most n-gram LMs used in practice have been limited to $n \leq 5$, trained on corpora orders of magnitude smaller than what modern neural LMs consume. The field has largely assumed that n-gram LMs are obsolete, superseded entirely by the representational power of Transformers.

The paper's central question is simple but provocative:

> "how well does the classical, n-gram language model (LM) perform if estimated from such massive corpora? In other words, are n-gram LMs still relevant in this era of neural LLMs?"

This is not merely a historical curiosity. It addresses a genuine gap in our understanding: **we do not know whether the performance gap between n-gram LMs and neural LMs is due to fundamental limitations of counting-based approaches, or simply because n-gram LMs have never been given the same data scale.** If the gap substantially narrows when n-gram LMs are trained on trillion-token corpora, then n-gram approaches may still have practical value — either as standalone tools for text analysis or as components that can improve neural LMs themselves.

### Why This Problem Matters

The motivation is both practical and scientific, spanning at least four dimensions:

**1. Complementarity to neural LMs on different tokens.** The paper presents evidence in Section 4.1 (Figure 4) that ∞-gram and neural LMs are predictive of actual human text on *different* subsets of tokens. Specifically, when Llama-2 assigns very low probability to the actual next token (the left side of the probability distribution histogram), the ∞-gram LM still achieves above 20% agreement with the actual token — and above 50% when considering only sparse ∞-gram estimates (where exactly one continuation appears in the training data). This complementarity means that even state-of-the-art neural LMs have systematic weaknesses that counting-based approaches do not share, and vice versa. Combining them is therefore not just an ensemble trick but a way to cover each other's failure modes.

**2. Interpretability and attributability.** Unlike neural LMs, whose predictions emerge opaquely from billions of parameters, ∞-gram predictions can be traced back to exact occurrences in the training data. When the ∞-gram predicts that "hippopotamus" follows "the large aquatic mammal called the hippo—", you can inspect the specific documents where that continuation appears, count how many times it occurs, and understand exactly why the prediction was made. This property is valuable for debugging model behavior, detecting data contamination, and providing attribution — capabilities that are increasingly demanded as LMs are deployed in high-stakes settings (Min et al., 2023a; Asai et al., 2024).

**3. A lens into neural LM behavior.** The paper demonstrates that ∞-gram can serve as an analytical tool for understanding what neural LMs are doing. The fluctuation analysis in Section 4.2 reveals that greedy decoding produces periodic drops in agreement with ∞-gram at specific suffix lengths (e.g., for Llama-2 7B, drops at effective $n = 20, 24, 28, 32$), while nucleus sampling produces a smooth, monotonic agreement curve that closely resembles human-written text. The authors hypothesize this irregularity stems from "deficiencies in neural LM pretraining and the positional embeddings of Transformers" — a diagnostic insight that would be difficult to obtain without an ∞-gram baseline to compare against. The ∞-gram thus functions as a kind of "normative" language model: it tells you what a perfect memorizer would predict given the training data, and deviations from that baseline reveal something about the neural model's inductive biases.

**4. Efficiency and accessibility.** The infini-gram engine stores its entire index on-disk (7 bytes per token, roughly 3.5× the raw dataset size), runs inference without GPUs, and achieves sub-200-millisecond latency for ∞-gram probability queries on trillion-token corpora (Table 3). Building the suffix array for a 1.4-trillion-token dataset takes approximately 48 hours on a single 128-CPU node with 1 TiB RAM (Section 3). This means that institutions without access to massive GPU clusters can still build and query n-gram LMs at scales that rival the pretraining data of frontier neural LMs — democratizing a form of large-scale language modeling that has historically required enormous computational resources.

### Prior Approaches and Where They Fall Short

The paper identifies four categories of existing work, each with specific limitations that motivate the ∞-gram LM and infini-gram engine.

**Conventional n-gram LMs with bounded $n$.** The classical approach is to build an explicit n-gram count table: a mapping from every unique n-gram observed in the training data to its frequency. This approach faces two crippling limitations at scale:

- **Exponential growth in table size with $n$.** An n-gram count table's size grows roughly exponentially with $n$ because the number of possible n-grams expands combinatorially with sequence length. The paper notes that "the 5-gram count table for a 1.4-trillion-token corpus would consume 28 TB of disk space" (Section 2). Extending to larger $n$ is computationally infeasible with this representation.

- **Small $n$ discards context and hurts prediction quality.** As Figure 1 illustrates concretely, a 5-gram LM sees only the last 5 tokens as context. If the prompt is longer — say, "the large aquatic mammal called the hippo—" — the 5-gram LM ignores the semantically rich prefix "the large aquatic mammal called" and sees only the last 5 tokens, which may be insufficient to uniquely determine that the next token is "potamus." The ∞-gram, by contrast, can use the 16-gram suffix to make the correct prediction. The paper quantifies this in Figure 3: a 5-gram LM agrees with human text on only a fraction of tokens, while the ∞-gram achieves 47% overall accuracy and over 75% when the effective $n \geq 16$.

The largest prior n-gram effort, Brants et al. (2007), indexed 5-grams on 2 trillion tokens and was limited to $n = 5$ and frequent n-grams only. These constraints are not inherent to the n-gram concept — they are artifacts of the naive count table implementation — but they have defined what "n-gram LM" means in practice for nearly two decades.

**Prior work on unbounded n-grams using suffix-based data structures.** The paper acknowledges three prior attempts to use suffix arrays or suffix trees for unbounded n-gram modeling, but identifies critical shortcomings:

- **Stehouwer & van Zaanen (2010)** proposed suffix arrays for ∞-gram modeling, but "their formulation does not yield proper probability distributions and, consequently, a language model" (Section 6). This is a fundamental problem: a language model must produce valid probability distributions over the vocabulary, not just counts.

- **Kennington et al. (2012)** proposed suffix trees, but suffix trees have "very high" storage overhead that "hinders scaling." While Shareghi et al. (2015) attempted to mitigate this with compression, the resulting systems operated on tiny datasets by modern standards — the largest prior training corpus was 9 billion tokens, roughly 500× smaller than the 5 trillion tokens used in this paper.

- **Poor empirical performance.** Of the three papers, only Shareghi et al. (2015) evaluated on general language modeling, and the paper notes that "the perplexity numbers are too high to be practically useful" (Section 6). This suggests that simply having an unbounded-n data structure is insufficient — the training data must be large enough to yield useful probability estimates.

The key gap: no prior work had combined (a) a mathematically valid ∞-gram probability formulation, (b) a storage-efficient suffix-based index, and (c) trillion-token training data, within a system that achieves practical inference latency. Each prior attempt addressed at most two of these three requirements.

**Nonparametric neural language models (kNN-LM, RETRO, etc.).** A parallel line of work has explored nonparametric LMs — models whose complexity can grow with reference data at inference time — but using neural representations rather than raw token counts:

- **kNN-LM** (Khandelwal et al., 2020) stores a vector for every token in the reference data and retrieves nearest neighbors at inference time. However, scaling this approach is expensive: each token requires storing a high-dimensional vector, and approximate nearest-neighbor search over billions of vectors requires specialized infrastructure. The largest prior systems (e.g., RETRO from Borgeaud et al., 2022) used 1.8 trillion tokens of reference data and consumed an estimated 432 TB of storage for 28 billion vectors (Table 6).

- **Chunk retrieval approaches** (Guu et al., 2020; Izacard et al., 2022; Borgeaud et al., 2022) retrieve entire text chunks rather than individual tokens, reducing the index size but requiring a neural retriever and a neural reader that processes the retrieved chunks.

The fundamental limitation: vector-based nonparametric LMs are expensive to scale in both storage and compute. The paper's comparison with SILO (Table 2) shows that ∞-gram yields better perplexity improvement than kNN-LM or RIC-LM, while using much larger reference data (360 billion tokens vs. 45 million to 1.2 billion tokens for the kNN-LM/RIC-LM baselines). The ∞-gram achieves this with dramatically lower storage per token (7 bytes vs. hundreds to thousands of bytes for vector-based approaches, per Table 6) and without requiring GPU inference.

**Interpolation between n-gram and neural LMs.** Prior work explored combining n-gram and neural LMs through interpolation, but with inconsistent results:

- **Mikolov & Zweig (2012)** found that interpolating a 5-gram Kneser-Ney model with RNNs improved perplexity.
- **Khandelwal et al. (2020)** found that "interpolating n-gram models with Transformers does not improve perplexity substantially."
- **Li et al. (2022)** showed that n-gram models could complement small neural LMs, but used limited reference data (101 million tokens) and compared only with small neural LMs (117–250M parameters).

The paper's critical insight is that the inconsistency in prior results likely stems from **insufficient reference data scale and insufficient $n$**. When the n-gram LM is trained on only 101 million tokens with $n = 5$, its estimates are too sparse and too context-poor to meaningfully complement a neural LM. But when trained on trillions of tokens with unbounded $n$, the ∞-gram provides complementary information on a substantial fraction of tokens — particularly those where the neural LM is uncertain — and the interpolation becomes reliably beneficial even for 70B-parameter models (Table 1).

### How This Paper Positions Itself

The paper's positioning can be understood along three axes:

**1. A scaling experiment, not a new modeling paradigm.** The paper does not claim that ∞-gram LMs should replace neural LMs. Instead, it asks: "what happens if we simply apply the same scaling philosophy that revolutionized neural LMs to the classical n-gram approach?" The answer, empirically, is that n-gram LMs improve substantially with data scale and unbounded $n$ — to the point where they become useful tools for analysis and complementary components for neural LMs. The contribution is demonstrating *that* scaling matters and *how* to do it efficiently, not proposing that ∞-gram is the superior language model.

**2. A practical system contribution (infini-gram engine) that enables the research contribution.** The paper is unusual in that the engineering contribution — the suffix-array-based engine with on-disk inference, millisecond latency, and 7-byte-per-token storage — is inseparable from the scientific contribution. Without infini-gram, the analyses in Sections 4 and 5 would be computationally infeasible. The paper makes this explicit by releasing the engine as a public web interface, API endpoint, and open-source Python package, positioning it as infrastructure for the community rather than a one-off research artifact.

**3. A bridge between classical statistical NLP and modern neural NLP.** By showing that ∞-gram can quantitatively improve Llama-2 70B and qualitatively reveal properties of Transformer decoders (the fluctuation phenomenon in greedy decoding), the paper demonstrates that classical and neural approaches are not competitors but complementary tools. The ∞-gram provides a principled, interpretable baseline that makes neural LM behavior more analyzable and — through interpolation — more accurate.

The paper explicitly contrasts its approach with the trend toward ever-larger neural models by noting that infini-gram "minimizes the compute resources needed (no GPU, and minimal CPU / RAM)" (Section 3). This is not just a technical detail — it represents a methodological stance that useful language modeling insights and capabilities need not require massive GPU clusters, and that there is untapped value in revisiting classical ideas with modern data scale and systems engineering.

## 3. Technical Approach

### 3.1 Reader Orientation

The paper builds a system that lets you ask "how many times has this exact sequence of words appeared in a 5-trillion-token corpus?" and get an answer in under 20 milliseconds, using only a CPU and an on-disk index that is smaller than the corpus itself. This engine, called **infini-gram**, powers a new kind of language model — the **∞-gram LM** — which predicts the next token by finding the longest suffix of the current text that appears anywhere in the training data and reporting what followed it, effectively using all available context rather than being limited to a fixed window of (say) 5 tokens like classical n-gram LMs.

### 3.2 Big-Picture Architecture (Diagram in Words)

The system has three major layers:

1. **Training data (tokenized corpus):** A massive text corpus (e.g., RedPajama at 1.4 trillion tokens) is tokenized into token IDs (each 2 bytes), concatenated into a single byte array with document separators, and stored on disk.

2. **Suffix array index (the infini-gram engine):** A suffix array is built over the byte array. This is an array of pointers (5 bytes each) that represents the lexicographical ordering of all suffixes of the token array. The suffix array plus the token array form the **infini-gram index**, consuming 7 bytes per token total (3.5× overhead). This index sits on-disk and is accessed via memory-mapped files.

3. **Query processor:** A C++ engine performs binary search on the suffix array to count n-gram occurrences, compute ∞-gram probabilities, return full next-token distributions, and retrieve documents. It uses sharding (to build indexes for datasets too large for RAM), parallelized shard processing, memory pre-fetching, and binary-lifting algorithms to achieve millisecond-level latency across all query types.

The flow is: a user submits a query (e.g., "count how many times `the large aquatic mammal` appears" or "what is the ∞-gram probability of `called` following this prefix?") → the engine performs binary search on the suffix array to locate the consecutive segment where all occurrences of the query string begin → for counting, the answer is the segment length; for language modeling, the engine counts both the prefix and the prefix-plus-next-token and divides; for document retrieval, it follows the pointers back into the token array to recover document boundaries and metadata.

### 3.3 Roadmap for the Deep Dive

- **First, the ∞-gram LM formal definition**, because the engine exists to serve this model and all downstream analyses depend on understanding exactly what probabilities it computes. We will cover the backoff variant used, why it yields a valid distribution without discounting, and how sparsity is defined.

- **Second, the suffix array data structure**, because understanding how infini-gram achieves its speed and storage efficiency requires understanding what a suffix array is and how it maps n-gram queries to binary search operations.

- **Third, building the suffix array index**, covering sharding, the linear-time construction algorithm, document offset storage, and the concrete scale numbers (48 hours, 128 CPUs, 1 TiB RAM for 1.4T tokens).

- **Fourth, inference algorithms**, covering how COUNT, NGRAMPROB, NGRAMDIST, INFGRAMPROB, INFGRAMDIST, and SEARCHDOC queries are implemented on top of binary search, and the algorithmic optimizations (binary-lifting for effective-$n$ lookup, amortized processing, hinted search) that reduce latency.

- **Fifth, the interpolation mechanism with neural LMs**, since this is how ∞-gram is made practically useful for perplexity evaluation despite containing zero probabilities, and the separate λ values for sparse vs. non-sparse estimates are a key design choice.

### 3.4 Detailed, Sentence-Based Technical Breakdown

This is primarily a **systems and analysis paper** whose core idea is that scaling classical n-gram LMs to trillion-token corpora with unbounded $n$ — made possible by a suffix-array-based engine — produces a language model that is both analytically useful (it reveals properties of text and of neural LMs) and practically beneficial (it improves neural LMs when interpolated).

---

#### The ∞-gram LM: Formal Definition and Backoff Variant

The paper generalizes the classical n-gram LM by allowing $n$ to be arbitrarily large — effectively infinite — and defines a specific backoff procedure that determines which $n$ to use for each prediction.

**Classical n-gram LM (the starting point).** For a fixed $n$, the maximum-likelihood estimate of the probability of token $w_i$ given the preceding $n-1$ tokens is:

$$P_n(w_i \mid w_{i-(n-1):i-1}) = \frac{\text{cnt}(w_{i-(n-1):i-1}w_i \mid \mathcal{D})}{\text{cnt}(w_{i-(n-1):i-1} \mid \mathcal{D})}$$

where `$w_{i-(n-1):i-1}$` is the context window of the previous $n-1$ tokens, `$w_i$` is the candidate next token, `$\mathcal{D}$` is the training corpus, and `$\text{cnt}(\cdot \mid \mathcal{D})$` counts exact occurrences of the token sequence in `$\mathcal{D}$`. When $n = 1$, the context is the empty string `$\varepsilon$` and `$\text{cnt}(\varepsilon \mid \mathcal{D}) = |\mathcal{D}|$`, so $P_1(w_i) = \text{cnt}(w_i)/|\mathcal{D}|$ — the unigram distribution.

**What it computes:** For a fixed context window of exactly $n-1$ tokens, the fraction of times that window is followed by token $w_i$ in the training data. This is a maximum-likelihood estimate that treats the training corpus as the ground-truth distribution.

**Why this form:** It is the plug-in estimator for a Markov model of order $n-1$ — it assumes that the probability of the next token depends only on the preceding $n-1$ tokens, and estimates that conditional probability by counting. However, this naive form has the well-known sparsity problem: the numerator `$\text{cnt}(w_{i-(n-1):i-1}w_i \mid \mathcal{D})$` may be zero even when the denominator is non-zero, producing a zero probability that makes perplexity infinite if any evaluation token receives zero probability. The standard solution — backoff — reduces $n$ when the numerator is zero.

**The ∞-gram backoff variant.** The ∞-gram LM conceptually starts with $n = \infty$ and backs off, but the paper uses a specific variant that backs off **only when the denominator is zero**, not when the numerator is zero:

$$P_\infty(w_i \mid w_{1:i-1}) = \frac{\text{cnt}(w_{i-(n-1):i-1}w_i \mid \mathcal{D})}{\text{cnt}(w_{i-(n-1):i-1} \mid \mathcal{D})}$$

where `$w_{1:i-1}$` is the entire document prefix (all tokens preceding $w_i$), and the **effective $n$** is defined as:

$$n = \max\{n' \in [1, i] \mid \text{cnt}(w_{i-(n'-1):i-1} \mid \mathcal{D}) > 0\}$$

**What these equations compute together:** Given a document prefix, find the longest suffix of that prefix that appears at least once in the training data (this determines the effective $n$). Then compute the conditional probability of $w_i$ using exactly that suffix as context — the denominator is the count of the suffix, and the numerator is the count of the suffix with $w_i$ appended. If the suffix appears $k$ times in the training data, and $w_i$ follows it $m$ of those times, the probability is $m/k$. If no suffix of any length appears in the training data (effective $n = 1$ and the unigram doesn't appear), the formula would be undefined — the paper's backoff stops at $n = 1$, so this case uses the unigram distribution where the denominator is `$|\mathcal{D}|$` and never zero.

**Why this backoff variant (denominator-only, not numerator), and why it matters:** The key difference from standard backoff (e.g., Katz backoff) is that the effective $n$ depends only on the context `$w_{1:i-1}$` and not on the specific candidate token `$w_i$`. This has a crucial consequence: 

$$P_\infty(* \mid w_{1:i-1}) \text{ is a valid probability distribution by construction}$$

because `$\sum_{w_i \in \mathcal{V}} \text{cnt}(w_{i-(n-1):i-1}w_i \mid \mathcal{D}) = \text{cnt}(w_{i-(n-1):i-1} \mid \mathcal{D})$` — every occurrence of the suffix in the training data is followed by exactly one token in the vocabulary, so summing over all possible next tokens recovers exactly the denominator count, and the probabilities sum to 1. This means **no probability discounting is required**, unlike Katz backoff where some probability mass must be reserved for unseen continuations. The paper explicitly notes this property:

> "Unlike Katz backoff, P∞(∗|w1:i−1) is a valid distribution by construction and does not require discounting. This is because the effective n is solely dependent on w1:i−1 and does not depend on wi."

However, this choice also means that **the numerator can still be zero** (if the suffix appears but is never followed by $w_i$), so the ∞-gram LM contains zero probabilities. This is why the paper never computes perplexity of the ∞-gram LM alone — it would be infinite — and instead always interpolates with a neural LM (Section 5) or uses token-level agreement accuracy (Section 4), which is immune to zero-probability issues because it only checks whether the probability assigned to the actual next token exceeds 0.5.

**Sparse vs. non-sparse estimates.** The paper defines an additional property of the ∞-gram estimate:

> "an estimate is sparse iff P(wi|wi−(n−1):i−1) = 1 for one of the wi ∈ V, and is zero for all other tokens in the vocabulary."

**What sparse means operationally:** The suffix that forms the context appears exactly $k$ times in the training data, and in every single one of those $k$ occurrences, it is followed by the exact same token. So the next-token distribution is degenerate — one token has probability 1, all others have probability 0. This turns out to be an important signal for prediction quality: sparse estimates are much more likely to agree with human-written text (75% accuracy overall, >80% for effective $n \geq 14$, per Figure 3 right) than non-sparse ones. The interpolation with neural LMs (Section 5) exploits this by using a separate hyperparameter `$\lambda_1$` for sparse estimates and `$\lambda_2$` for non-sparse estimates, because sparse estimates carry a qualitatively different kind of information — they encode deterministic constraints from the training data rather than statistical tendencies.

**Why sparse estimates are more reliable:** When the training data is large enough that a given context appears multiple times and *always* continues the same way, this indicates either a genuine deterministic constraint (e.g., "San Francis" is always followed by "co" in English text) or a strong collocation (e.g., "the large aquatic mammal called the hippopota—"). Non-sparse estimates, by contrast, reflect genuine variation in how the context can continue, and the empirical distribution over those continuations may be noisy or influenced by corpus biases.

---

#### Suffix Array: The Core Data Structure

The suffix array is the data structure that makes infini-gram possible. Understanding it requires understanding what problem it solves: efficient substring counting in a massive string.

**Definition.** For a string (or token array) of length $N$, a suffix array is an array of $N$ integers, where the $i$-th element is the starting position (byte offset) of the suffix that is ranked $i$-th among all $N$ suffixes when arranged in lexicographical order.

**Concrete example (from Figure 2, left).** For the toy string `aabaca` (where each character is conceptually a token), the suffixes are:
- Position 0: `aabaca`
- Position 1: `abaca`
- Position 2: `baca`
- Position 3: `aca`
- Position 4: `ca`
- Position 5: `a`

Lexicographically sorted: `a` (position 5), `aabaca` (position 0), `abaca` (position 1), `aca` (position 3), `baca` (position 2), `ca` (position 4). The suffix array is `[5, 0, 1, 3, 2, 4]`.

**Why this ordering enables fast substring search:** All suffixes that begin with a given prefix — say, `a` — form a consecutive block in the suffix array (positions 0–3 in the sorted list, containing suffixes at original positions 5, 0, 1, 3). The suffixes that begin with a longer prefix, say `aa`, form a sub-block within that block. This means that to count occurrences of any query string, you only need to find the boundaries of its consecutive block in the suffix array — which can be done with two binary searches (one for the lower bound, one for the upper bound), each taking `$O(\log N)$` random array accesses. The count is simply the difference between the upper and lower bound indices.

**Infini-gram's specific suffix array format (Figure 2, right).** The paper builds the suffix array on the **byte array** of the tokenized dataset:

1. The text corpus is tokenized (e.g., using the Llama-2 tokenizer), producing sequences of token IDs.
2. Documents are concatenated and separated by the special token `\xff\xff` (two bytes, both `\xff`).
3. Token IDs are stored as 2-byte integers (assuming `$|\mathcal{V}| < 2^{16} = 65536$`), so a token array of $N$ tokens occupies $2N$ bytes.
4. The suffix array contains $N$ pointers, each pointing to a token in the token array by storing its **byte offset**.
5. Each pointer needs `$\lceil\log_2(2N)/8\rceil$` bytes. For corpora with 2 billion to 500 billion tokens (the range after sharding), this is 5 bytes per pointer.
6. Total index size: $2N$ (token array) + $5N$ (suffix array) = **$7N$ bytes**, or 7 bytes per token.

**Why a suffix array rather than an n-gram count table:** An n-gram count table storing all unique n-grams with unbounded $n$ would be astronomically large. The paper estimates (Appendix A.1) that a 5-trillion-token dataset contains at least $2 \times 10^{15}$ (2 quadrillion) unique n-grams when considering all possible $n$ within document boundaries — storing even a fraction of these would be infeasible. The suffix array, by contrast, costs only $7N$ bytes regardless of how many unique n-grams exist, because it stores only the ordering of token positions, not the n-grams themselves. The counts are computed on-the-fly via binary search.

**Why a suffix array rather than a suffix tree:** Suffix trees can also support substring operations but have much higher storage overhead per indexed token. Prior work (Kennington et al., 2012) found suffix trees "hinders scaling" due to storage. The suffix array trades a slightly higher query time (`$O(\log N)$` vs. `$O(L)$` for a query of length $L$) for dramatically lower storage, and the paper compensates for query time with binary search optimizations and pre-fetching.

**Why the byte-array representation:** Storing token IDs as 2-byte integers and building the suffix array over this byte array means that the suffix array captures token boundaries implicitly (every suffix starts at an even byte offset) while enabling byte-level comparison during binary search. This is more storage-efficient than storing suffixes as sequences of token IDs (which would require $N \times \text{avg suffix length} \times 2$ bytes) and supports the standard linear-time suffix array construction algorithms that operate on byte/integer arrays.

---

#### Building the Suffix Array Index

**Algorithm.** The paper uses a linear-time suffix array construction algorithm (Kärkkäinen et al., 2006), specifically adapting the implementation from Lee et al. (2022) with further optimizations. The algorithm constructs the suffix array in time proportional to the length of the byte array, which is $O(2N) = O(N)$.

**The sharding problem.** Linear-time construction requires heavy random access to the byte array, meaning the entire byte array must fit in RAM for reasonable building time. When the byte array exceeds available RAM, the paper **shards** the dataset: the token array is split into multiple shards, and a separate suffix array is built for each shard. Sharding has a trade-off: it enables building the index on datasets larger than RAM, but increases inference latency because each query must be executed on all shards and results aggregated.

**Concrete building statistics.** On the RedPajama dataset (1.4 trillion tokens, producing a byte array of approximately 2.8 TB):

- **Hardware:** A single node with 128 CPUs and 1 TiB RAM.
- **Time:** Approximately 48 hours (2 days) to build the suffix array.
- **Disk storage:** The full infini-gram index (byte array + suffix array) occupies 7 bytes per token, so roughly $1.4 \times 10^{12} \times 7 = 9.8$ TB. The paper reports "10 TB of disk storage" for this dataset.

**Document offset index.** In addition to the suffix array, infini-gram stores auxiliary data structures for document retrieval:

1. A **document offset file** stores the byte offset of the start of each document in the tokenized dataset, similar in format to the suffix array.
2. A **document metadata file** stores a comma-separated string for each document containing metadata (document ID, source, URL).
3. A **document metadata offset file** stores byte offsets into the metadata file.

These auxiliary files are "negligible in size compared to the suffix array, because there are far less documents than the total number of tokens" (Appendix A.2). For reference, the paper's 5-trillion-token combined dataset contains approximately $D = 6 \times 10^9$ documents with an average of 857 tokens per document — so there are roughly 6 billion document offsets vs. 5 trillion suffix array entries, a ratio of about 1:800.

**Additivity and subtractivity.** The paper notes that infini-gram indexes are additive and subtractive: if indexes are built on disjoint datasets (with the same tokenizer), n-gram counts from multiple indexes can be summed to produce the count for the union. Similarly, an index for a set difference can be obtained by subtracting counts. This means that to combine Pile-train (360B tokens) and RedPajama (1.4T tokens) into a 1.8T-token index, the paper does not need to re-index the union — the counts are simply added at query time. This also means data removal can be done by building an index on the removed set and subtracting it from the full index.

**Decontamination.** Before using any corpus as ∞-gram training data for evaluation, the paper runs decontamination against the evaluation sets. The tool used is the Big Friendly Filter (BFF, Groeneveld 2023). The paper's settings: $n = 13$, and a document is filtered out if at least 80% of its 13-grams appear in the evaluation set. Entire documents are removed, not paragraphs. Pile-train is lowercased before filtering "to capture more potential contaminations." Filtering statistics (Table 4): 0.6% of Pile-train documents removed (1,296,376 out of 210,607,728); 0.08% of RedPajama documents removed (730,437 out of 931,361,530). The paper explicitly notes:

> "Decontamination is non-trivial, and its definition could vary (e.g., when there is an identical sentence, is it contamination, or is it a quote that naturally occurs in real test-time scenarios?) Thus we followed the standard best practices for decontamination."

This is a critical step because, without decontamination, the ∞-gram would trivially "predict" the next token on test documents by retrieving the exact same document from training data — artificially inflating performance and invalidating the comparison with neural LMs.

---

#### Inference Algorithms on the Suffix Array

All query types are implemented on top of one fundamental operation: **finding the consecutive segment in the suffix array that corresponds to all occurrences of a given token string**.

**Binary search for n-gram boundaries.** Given a query string $x_1 \ldots x_n$:

1. Two binary searches are performed on the suffix array: one to find the first (lowest-index) suffix that starts with $x_1 \ldots x_n$, and one to find the last (highest-index) suffix that starts with $x_1 \ldots x_n$.
2. At each step of binary search, the suffix at the mid-point index is compared to the query string by reading bytes from the token array starting at the offset indicated by the suffix array pointer.
3. The two binary searches can be parallelized, reducing latency by roughly $2\times$.
4. Time complexity: $O(n \cdot \log N)$ string comparisons, but in practice $O(\log N)$ **random disk accesses** because "computers usually fetch memory in pages of 4K bytes, and string comparison is much faster than page fetching" (Appendix A.4). The $n$ factor is negligible because byte comparisons are sequential within a page.

The count of the n-gram is then:

$$\text{cnt}(x_1 \ldots x_n) = \text{upper\_bound} - \text{lower\_bound}$$

**Query type 1: COUNT (n-gram counting).** Simply execute the binary search above and return the difference between upper and lower bound positions. Latency: typically under 20 milliseconds regardless of $n$ or frequency (Table 3), because the binary search cost depends only on the size of the suffix array ($O(\log N)$ random accesses), not on the length of the query or the number of occurrences.

**Why latency is independent of $n$:** Even for $n = 1000$, the string comparison during binary search only needs to compare enough bytes to determine lexicographical ordering, which typically requires reading only the first few bytes before a difference is found (most strings diverge early in lexicographical order). And memory is fetched in 4K-byte pages anyway, so reading 1000 bytes costs roughly the same number of page fetches as reading 10 bytes if they're contiguous.

**Query type 2: NGRAMPROB (token probability from n-gram LM, fixed $n$, no backoff).** To compute $P_n(x_n \mid x_1 \ldots x_{n-1})$:

1. Count `$x_1 \ldots x_{n-1}$` (the denominator).
2. Count `$x_1 \ldots x_n$` (the numerator).
3. Return the ratio.

The paper optimizes step 2 by noting that the occurrence positions of `$x_1 \ldots x_n$` must be a sub-segment of the occurrence positions of `$x_1 \ldots x_{n-1}$`. So after finding the segment for `$x_1 \ldots x_{n-1}$`, the binary search for `$x_1 \ldots x_n$` is constrained to that segment rather than the entire suffix array, "which reduces the latency by at most 2x" (Appendix A.4). This optimization is a form of **hinted search**.

**Query type 3: NGRAMDIST (full next-token distribution from n-gram LM).** To compute the distribution over all possible next tokens following context `$x_1 \ldots x_{n-1}$`:

1. Find the segment for `$x_1 \ldots x_{n-1}$`.
2. For each token $v$ in the vocabulary, check whether `$x_1 \ldots x_{n-1}v$` has a non-zero count by searching within the segment from step 1.
3. This requires $O(|\mathcal{V}| \cdot \log N)$ operations in the worst case.

Practical latency: 31–39 milliseconds for $n = 5$ (Table 3), which is acceptable because the vocabulary size is manageable (under 65,536 tokens given the 2-byte representation).

**Query type 4: INFGRAMPROB (token probability from ∞-gram LM).** This is the most algorithmically interesting query because it requires finding the **effective $n$** — the length of the longest suffix of the context that has a non-zero count in the training data — without searching all possible suffix lengths linearly.

**The binary-lifting + binary-search algorithm for effective $n$:** Given a prefix `$w_{1:i-1}$`, the goal is to find the largest $L$ such that the suffix of length $L$ (i.e., `$w_{i-L:i-1}$`) appears in the training data. A naive approach would test $L = i, i-1, i-2, \ldots$ sequentially, requiring $O(L \cdot \log N)$ time per token. The paper instead uses:

1. **Binary lifting (exponential search):** Test suffix lengths $L = 1, 2, 4, 8, 16, \ldots$ (doubling each time) until finding one that has zero count. This consumes $O(\log L)$ query operations.
2. **Binary search within the identified interval:** Once the upper bound is found (the first power of two with zero count) and the lower bound is known (the previous power of two, which had positive count), perform binary search on suffix lengths in that interval to find the exact maximum $L$. This consumes another $O(\log L)$ queries.
3. Total: $O(\log L \cdot \log N)$ time.

**What this achieves:** Instead of $O(L)$ queries (which could be hundreds of tokens in the worst case, e.g., for the first token of a very long document), the effective $n$ is found with approximately $O(\log L) \approx O(\log 1024) \approx 10$ queries, even if the effective $n$ is 100. Once the effective $n$ is found, the probability is computed as the ratio of counts with and without the actual next token, same as NGRAMPROB but with automatically-determined $n$.

Latency: 90–135 milliseconds per token for isolated queries (Table 3), but see below for the amortized case.

**Query type 5: INFGRAMDIST (full next-token distribution from ∞-gram LM).** Same as INFGRAMPROB but iterates over all vocabulary tokens to produce a full distribution. Latency: 88–180 milliseconds (Table 3). The slightly higher latency compared to INFGRAMPROB is due to the vocabulary iteration, but the effective-$n$ lookup is done once and shared across all vocabulary tokens.

**Amortized dense ∞-gram computation.** When evaluating the ∞-gram on a sequence of consecutive tokens (e.g., computing the probability of every token in a test document), the paper exploits the observation that:

> "the effective n for one token is at most one token longer than that for the previous token"

This means that for token $i$, the effective $n$ is at most (effective $n$ for token $i-1$) + 1. So instead of running the full binary-lifting search from scratch for each token, the algorithm starts from the previous token's effective $n$ and only searches upward if needed. This reduces the amortized time complexity per token to $O(\log N)$ — the same as a simple n-gram count. Concrete latency: 12–20 milliseconds per token on consecutive tokens (Table 3), roughly an order of magnitude faster than isolated INFGRAMPROB queries. This is the setting used for the perplexity evaluation in Section 5.

**Query type 6: SEARCHDOC (document retrieval).** To find all documents containing an n-gram (or a CNF logical expression of n-gram terms):

1. Find the suffix array segment for the n-gram (same as COUNT).
2. For each pointer in this segment, dereference it to find the corresponding position in the token array.
3. Given a token array position, find its enclosing document by performing a **binary search on the document offset index** to locate the document whose start offset is the largest value not exceeding the token position.
4. With the document ID, retrieve metadata from the document metadata file using the metadata offset index.
5. For CNF expressions (e.g., `(A OR B) AND (C OR D)`), the engine computes the sets of documents for each term and performs set intersections/unions.

**Why the document offset index is necessary:** Without it, finding the enclosing document would require expanding outward from the hit position in both directions until the `\xff\xff` document separator is found. The paper notes that "documents as large as 20M tokens" exist in these corpora, so this expansion could be extremely slow (Appendix A.4). The document offset index makes document retrieval a `$O(\log D)$` operation independent of document size.

---

#### On-Disk Index and Memory Pre-Fetching

A critical engineering challenge is that the suffix array and byte array may be too large to fit in RAM. For RedPajama at 1.4T tokens, the byte array is ~2.8 TB and the suffix array is ~7 TB — together ~10 TB, which exceeds the RAM of most machines.

**Memory-mapped files.** The infini-gram engine keeps both arrays on-disk (SSD) and accesses them as memory-mapped files via the operating system's virtual memory system. The OS handles page faults: when a byte offset is accessed that is not currently in RAM, the corresponding disk page is loaded. This means infini-gram can run on machines with minimal RAM (the paper reports "minimal CPU / RAM" in Section 3).

**The latency problem.** Binary search requires random access to the suffix array and byte array — at each step, a new mid-point index is computed and the suffix at that position must be compared. Each access may trigger a disk read, and SSDs have latency on the order of 10–100 microseconds per random read. With `$\log_2(N)$` ≈ 30–40 binary search steps (for $N \approx 10^9$ tokens per shard), this would mean several milliseconds just for the random reads — and potentially hundreds of milliseconds for the full query pipeline.

**Memory pre-fetching optimization.** The paper implements a pre-fetching mechanism that informs the OS of the array offsets that will likely be accessed in the near future:

> "we implemented a memory pre-fetching method that informs the system of the array offsets we will likely be reading in the near future"

This allows the OS to asynchronously load these pages from disk into RAM before they are needed, overlapping I/O with computation. The paper reports that "pre-fetching reduces average latency by roughly 5×" (Appendix A.4). The likely offsets are predictable because binary search follows a deterministic pattern where the next mid-point depends only on the comparison result — the engine can compute both possible next offsets (if the query is lexicographically less than the midpoint, check one offset; if greater, check the other) and pre-fetch both.

**Shard-level parallelism.** When the suffix array is sharded across multiple files (built on sharded token arrays), each shard can be processed independently:

> "we can simply perform counting on each individual shard and accumulate the counts across all shards... The processing of different shards can be parallelized, reducing the time complexity back to O(log N)"

The paper's benchmarks (Table 3) show that RedPajama uses $S = 8$ shards (due to its size) while Pile-train uses $S = 2$ shards. Despite 4× more shards, RedPajama's COUNT latency is only about 1.4× higher than Pile-train's (~19 ms vs. ~14 ms for $n=5$), confirming that shard parallelism largely mitigates the sharding overhead.

**Hardware and benchmarking setup.** All latency numbers in Table 3 are benchmarked on "a single, 8-core CPU node" with the engine written in C++, using parallelized shard processing. The training data and suffix array are stored on an SSD. Each query type is benchmarked on 1,000 tokens randomly sampled from Pile's validation data (except consecutive INFGRAMPROB, which uses 1,000 consecutive tokens from each of 10 sampled documents).

---

#### Interpolation with Neural LMs

The ∞-gram LM produces zero probabilities for tokens that never follow the effective context in the training data. This makes it impossible to compute perplexity directly (any zero-probability token makes perplexity infinite). The paper therefore uses ∞-gram only in combination with neural LMs, via a simple token-level linear interpolation:

$$P(y \mid x) = \lambda P_\infty(y \mid x) + (1 - \lambda) P_{\text{neural}}(y \mid x)$$

where `$\lambda \in [0, 1]$` is a hyperparameter controlling the weight of the ∞-gram estimate.

**What it computes:** For each token position, a weighted average of two probability distributions — one from the ∞-gram LM (which may be sparse or even degenerate) and one from the neural LM (which is always dense, assigning non-zero probability to all vocabulary items). The result is a dense distribution that can never be zero (because the neural LM term is always positive) and thus yields finite perplexity.

**Why linear interpolation (rather than product of experts or other combination):** Linear interpolation is the simplest combination method, it guarantees a valid probability distribution (convex combination of two valid distributions), and it has a single interpretable hyperparameter. The paper does not explore more sophisticated combination methods (e.g., Bayesian model averaging, learned gating) — the goal is to demonstrate that even this simple combination yields substantial gains, not to optimize the combination architecture.

**Separate $\lambda$ for sparse vs. non-sparse estimates.** The paper's analysis in Section 4 shows that sparse ∞-gram estimates (where the training data deterministically predicts one token) are qualitatively different from non-sparse ones — they have much higher agreement with human-written text (75% vs. 47% overall accuracy; Figure 3). The interpolation therefore uses two separate hyperparameters:

- `$\lambda_1$`: weight for sparse ∞-gram estimates (when `$P_\infty(y \mid x) = 1$` for exactly one $y$ and 0 for all others)
- `$\lambda_2$`: weight for non-sparse ∞-gram estimates (when the probability is distributed over multiple tokens)

**How hyperparameters are chosen:** `$\lambda_1$` and `$\lambda_2$` are tuned on the validation set to minimize the perplexity of the combined model. This is a two-dimensional grid search (or similar optimization) over $[0, 1] \times [0, 1]$. Because sparse estimates are more reliable, the optimal `$\lambda_1$` tends to be higher than `$\lambda_2$` (though exact values are not reported in the paper).

**Why separate $\lambda$s are necessary:** If a single `$\lambda$` were used for both sparse and non-sparse estimates, it would either underweight the highly reliable sparse predictions or overweight the noisy non-sparse ones. The separation allows the model to trust the ∞-gram strongly when it is confident (sparse case) while relying more heavily on the neural LM when the ∞-gram estimate is based on limited or contradictory evidence (non-sparse case). This design choice is a direct consequence of the empirical finding that sparsity is a strong indicator of ∞-gram reliability.

**Perplexity metric.** The combined model's perplexity is computed as:

$$\text{PPL} = \exp\left(-\frac{1}{T} \sum_{t=1}^T \log P(y_t \mid x_t)\right)$$

where $T$ is the total number of evaluation tokens. The relative improvement is defined as:

$$\text{Improvement} = \left(1 - \frac{\text{PPL}(M)^{-1}}{\text{PPL}(M_o)^{-1}}\right) \times 100\%$$

**Why this improvement metric:** It measures the fraction of the gap toward perfect language modeling (PPL = 1) that has been closed by the new model relative to the baseline. If baseline has PPL = 10 (i.e., perplexity gap of 9) and the combined model has PPL = 5 (perplexity gap of 4), the improvement is $(1 - 5^{-1}/10^{-1}) = (1 - 2/10) = 80\%$. This metric normalizes for the difficulty of the baseline: a reduction from PPL 100 to PPL 50 is reported the same as a reduction from PPL 10 to PPL 5 (both 50% relative improvement), even though the absolute gap closed is much larger in the former case.

---

#### Summary of Design Choices and Their Justifications

- **Suffix array over n-gram count table:** Enables unbounded $n$ without exponential storage growth. Cost is $O(N)$ storage and $O(\log N)$ query time, vs. $O(\text{\#unique n-grams})$ storage for count tables — which would be infeasible at trillion-token scale.
- **Denominator-only backoff (not numerator backoff):** Yields a valid probability distribution without discounting, at the cost of potentially zero probabilities (handled by interpolation with neural LMs).
- **Byte-array representation with 2-byte tokens and 5-byte pointers:** Minimizes index size (7 bytes per token) while supporting vocabulary sizes up to 65,536 and datasets up to 500B tokens per shard.
- **Linear-time suffix array construction:** Enables building on trillion-token datasets in ~2 days on a single node, vs. $O(N \log N)$ algorithms that would be infeasible.
- **Sharding with parallel processing:** Enables datasets larger than RAM, with latency overhead mitigated by parallelism and pre-fetching.
- **Binary-lifting for effective-$n$ search:** Reduces ∞-gram latency from $O(L \cdot \log N)$ to $O(\log L \cdot \log N)$.
- **Amortized dense ∞-gram computation:** Exploits the sequential structure of evaluation to reduce per-token latency by ~10× for consecutive tokens.
- **Memory-mapped on-disk index with pre-fetching:** Allows the entire index to stay on SSD, with latency reduced 5× by informing the OS of future access patterns.
- **Separate interpolation weights for sparse vs. non-sparse estimates:** Exploits the empirical finding that sparsity strongly predicts reliability, allowing the model to differentially trust deterministic vs. statistical ∞-gram predictions.
- **Decontamination with BFF (13-gram, 80% threshold):** Prevents artificial inflation of ∞-gram performance due to training-evaluation overlap, following community best practices.

## 4. Key Insights and Innovations

### Innovation 1: The ∞-gram as a Perfect-Memorization Baseline That Exposes Neural LM Behavior

The paper's most distinctive conceptual contribution is not the suffix array engine itself — that's an engineering solution to a scalability problem — but rather the **framing of the ∞-gram as a normative, interpretable baseline** against which neural LM behavior can be diagnosed. This transforms the n-gram LM from a competitor to a measurement instrument.

**What the field did before:** Prior work on analyzing neural LMs (e.g., what they memorize, whether they copy from training data, how decoding strategies affect output) lacked a principled reference point. Researchers could compare models to each other, or to human text statistics, but there was no way to ask: "Given infinite memory and exact recall of this specific training corpus, what *should* the model predict here?" The ∞-gram answers exactly this question — it tells you what a perfect memorizer would output at every token position, conditioned on the longest matching suffix in the training data. This is not a model you would deploy (its zero probabilities make perplexity infinite), but it is an extraordinarily useful **diagnostic instrument**.

**Why this framing is distinctive:** The paper doesn't merely show that the ∞-gram is more accurate than a 5-gram (47% vs. 29% token agreement; Figure 3) — that's expected given more context. The conceptual move is using the ∞-gram to reveal *where neural LMs fail relative to perfect memorization* and *how decoding strategies distort generation relative to the training data distribution*. Figure 4 shows that on tokens where Llama-2 assigns very low probability (the left tail of the neural LM's probability distribution), the ∞-gram still achieves above 20% agreement with the actual token — and above 50% for sparse estimates. This is not a performance comparison; it's a **failure-mode analysis** showing that neural LMs have systematic blind spots where a simple counting-based approach succeeds. Conversely, the ∞-gram's failures (e.g., "predicting the first token of an entity name" due to "insufficient contextualization"; Section 4.1) reveal where memorization alone is insufficient and neural representations genuinely add value.

The **fluctuation analysis** in Section 4.2 is the most striking example of this diagnostic use. Under greedy decoding, the agreement between Llama-2 7B outputs and the ∞-gram drops periodically at effective n = 20, 24, 28, 32 — a statistically significant, nearly periodic pattern (p < 10⁻⁹⁹ by two-proportion z-test). This pattern is absent under nucleus sampling, which produces a smooth, monotonic agreement curve resembling human-written text. The paper hypothesizes this is "caused by the application of positional embeddings when pretraining these Transformer-based models." **The ∞-gram didn't just measure accuracy — it surfaced a structural artifact of Transformer architectures that would be invisible without a perfect-memorization baseline to compare against.** This is a genuinely novel finding that could not have been obtained by comparing neural LMs to each other or to human text.

**The significance extends beyond this paper:** The ∞-gram-as-instrument framing opens up a research program where counting-based baselines are used to diagnose neural model behavior across architectures, training data compositions, and decoding strategies. The fluctuation result in particular suggests that the interaction between positional embeddings and greedy decoding may systematically bias generation toward memorized sequences at certain context lengths — a finding with implications for how we understand and mitigate memorization in LLMs. This is a fundamental contribution to model interpretability, not an incremental performance improvement.

**Anchor evidence:** Figure 4 (complementarity on low-probability tokens), Figure 5 top row (contrasting decoding methods), Figure 5 bottom row (periodic fluctuation under greedy).

---

### Innovation 2: Scale Transforms n-gram LMs from Obsolete Baselines to Complementary Components

The paper's second fundamental contribution is the empirical demonstration that **data scale qualitatively changes the role an n-gram LM can play**. This is not simply "bigger is better" — it's a finding that a method long considered obsolete can, when given the same data scale as modern neural LMs, transition from being a weak standalone model to being a powerful complementary component that improves state-of-the-art neural systems.

**What the field assumed:** Prior work had reached inconsistent and largely pessimistic conclusions about combining n-gram and neural LMs. Khandelwal et al. (2020) found that interpolating n-gram models with Transformers "does not improve perplexity substantially." Li et al. (2022) found benefits but only with small neural LMs (117–250M parameters) and limited reference data (101M tokens). The dominant assumption in the field was that n-gram LMs have been definitively superseded — that whatever information they capture is already subsumed by the vastly more expressive neural LM, and that adding them back in would at best provide marginal gains on small models.

**What this paper shows is qualitatively different:** When the n-gram LM is trained on 360B–1.8T tokens (rather than 101M) and allowed unbounded n (rather than n = 5), interpolating it with Llama-2 70B reduces perplexity by 12% on Pile's test set (Table 1). This is remarkable because Llama-2 70B was trained on 2 trillion tokens itself — it has already seen essentially all the text that the ∞-gram indexes. The ∞-gram is not providing novel factual knowledge; it is providing a different kind of signal — exact-suffix completion — that the neural LM, despite its enormous capacity, has not fully internalized. The fact that gains persist even when the ∞-gram reference data is a subset of the neural LM's training data (GPT-Neo/J models are trained on Pile; the ∞-gram uses Pile-train) shows that the neural LM does not simply memorize and reproduce its training data, even at 6.7B parameters — it learns representations that generalize, but at the cost of imperfect recall of specific sequences.

**Why this is a fundamental shift, not an incremental gain:** This result reframes the relationship between memorization and generalization in language modeling. Neural LMs are often described as having "memorized" training data, but the ∞-gram shows that even large neural LMs systematically underperform exact recall on a substantial fraction of tokens. The 12% perplexity improvement on Llama-2 70B means that, despite having seen the training data, the neural LM assigns lower probability to the actual next token than a simple counting-based approach does on many positions. This is not a failure of the neural LM per se — it reflects the genuine tension between compressing knowledge into parameters (which requires generalization and approximation) and retaining exact sequence-level information (which the ∞-gram does perfectly). The ∞-gram and neural LM are thus **fundamentally complementary**, not because one is better than the other, but because they represent different points on the memorization-generalization spectrum.

**The cross-family pattern strengthens this claim:** The perplexity gains are largest when the ∞-gram reference data distribution *differs* from the neural LM's training distribution (e.g., ∞-gram with Pile-train + RedPajama improves Llama-2 by up to 24% on Pile-test; Table 1, bottom rows). This means the ∞-gram is not just filling in gaps from insufficient training — it is providing access to a data distribution that the neural LM either didn't train on or didn't fully absorb. The ∞-gram thus functions as a form of **retrieval augmentation**, but one that is simpler, more storage-efficient (7 bytes/token vs. hundreds to thousands of bytes/token for vector-based approaches; Table 6), and more interpretable (every prediction can be traced to specific training documents) than existing kNN-LM or RETRO-style approaches.

**Anchor evidence:** Table 1 (perplexity improvements across model families and sizes, including Llama-2 70B), Table 2 (comparison with kNN-LM and RIC-LM showing superior improvement from ∞-gram), Table 6 (storage efficiency comparison).

---

### Innovation 3: Sparsity as a Self-Supervised Signal for Estimate Reliability

The paper identifies **sparsity** — whether the ∞-gram's next-token distribution is degenerate (probability 1 on a single token) — as a powerful, automatically available signal that predicts when the ∞-gram estimate is reliable. This is a conceptual contribution distinct from the ∞-gram itself: it is a discovery about the structure of language statistics at scale, and it has practical consequences for how the ∞-gram is used.

**What sparsity means conceptually:** An ∞-gram estimate is sparse when the longest matching suffix of the prompt appears exactly k times in the training data, and in every single one of those k occurrences, the same token follows. This can happen because of deterministic constraints (e.g., "San Francis" → "co" in every occurrence) or strong collocations (e.g., "the large aquatic mammal called the hippopota—" → "mus" in every occurrence). The key empirical finding is that sparse estimates are dramatically more accurate: 75% agreement with human text overall, vs. 47% for all estimates, and over 80% when effective n ≥ 14 (Figure 3, right plot).

**Why this is significant beyond raw accuracy:** The sparsity signal is **self-supervised** — it emerges from the training data statistics without any external annotation, model training, or human judgment. The ∞-gram doesn't need to be told which of its predictions are reliable; it can recognize them automatically by checking whether the next-token distribution is degenerate. This is a qualitatively different kind of confidence signal than what neural LMs provide. Neural LMs can output a probability (e.g., 0.9 on a token), but that probability reflects the model's internal uncertainty — which may be miscalibrated. The ∞-gram's sparsity reflects an **objective fact about the training data**: this context has only ever been followed by one token. The paper shows this objective signal is highly predictive of correctness.

**How this insight is operationalized:** The interpolation scheme in Section 5 uses separate λ hyperparameters for sparse vs. non-sparse estimates (λ₁ for sparse, λ₂ for non-sparse), allowing the combined model to trust the ∞-gram more when it is in the sparse regime. This is not an arbitrary engineering choice — it is a direct consequence of the empirical finding that sparsity encodes reliability. The paper doesn't explore more sophisticated uses of this signal (e.g., using the degree of non-sparsity as a continuous confidence measure, or using sparsity patterns to detect distribution shift), but the identification of sparsity as a meaningful signal is a contribution that other researchers can build on.

**Contrast with prior work:** Prior n-gram LMs had no analogous concept — with small n and small training data, most contexts would have sparse estimates simply because the data is too limited, and sparsity would not be a reliable signal of correctness. The trillion-token scale is what makes sparsity meaningful: when a context appears many times (potentially thousands) and always continues the same way across a diverse corpus, that is a genuine statistical regularity, not a data sparsity artifact. This is another instance of scale qualitatively changing the nature of the signal.

**Anchor evidence:** Figure 3 (right plot, sparse-only agreement rates), Figure 4 (sparse estimate agreement remains high even when neural LM agreement is low), Section 5 interpolation design (separate λ₁ and λ₂).

---

### Innovation 4: The Suffix Array as an Enabling Architecture for Trillion-Token, Unbounded-n Language Modeling

While the suffix array itself is a well-known data structure (dating to the early 1990s), the paper's contribution is demonstrating that — with careful engineering — it can be the foundation for a **practical, low-latency, resource-efficient ∞-gram LM at scales that rival the pretraining corpora of frontier neural LMs**. This is an engineering-systems contribution, but it has conceptual significance: it shows that a classical data structure, when properly optimized, can achieve what the field assumed required massive GPU clusters and vector databases.

**What the field assumed was necessary:** The dominant paradigm for scaling nonparametric LMs has been vector-based: store a dense embedding for every token or text chunk, and use approximate nearest-neighbor search at inference time (kNN-LM, RETRO, Atlas, etc.). This approach has fundamental scaling difficulties: each token requires a high-dimensional vector (hundreds to thousands of bytes per token), approximate search over billions of vectors requires specialized infrastructure, and inference latency grows with the index size. RETRO (Borgeaud et al., 2022), the largest prior system, used 1.8T tokens of reference data, stored 28 billion vectors, and consumed an estimated 432 TB of storage (Table 6). The implicit assumption was that exact-match approaches — n-gram LMs — had hit a scaling ceiling at n ≤ 5 due to the combinatorial explosion of the count table.

**What the suffix array enables that the field didn't fully exploit:** The suffix array sidesteps the combinatorial explosion entirely. It stores not the n-grams themselves but the *ordering of suffixes*, which occupies 7N bytes regardless of how many unique n-grams exist. The cost is paid in query time (binary search requires O(log N) random accesses) rather than storage. The paper's key insight is that, with modern hardware (SSDs with ~10–100 μs random read latency, multi-core CPUs, memory pre-fetching) and algorithmic optimizations (binary-lifting for effective-n search, hinted search, amortized dense computation), the query-time cost can be driven down to milliseconds — making the suffix array a practical engine for interactive-scale ∞-gram queries on trillion-token corpora.

**Why this is a fundamental contribution, not just an optimization:** Prior work had attempted suffix-array-based ∞-gram LMs (Stehouwer & van Zaanen, 2010; Kennington et al., 2012; Shareghi et al., 2015) but failed on three fronts: mathematical validity (the first didn't produce a proper LM), storage efficiency (suffix trees were too expensive), or empirical usefulness (perplexity numbers were "too high to be practically useful" with 500× less training data; Section 6). The paper solves all three simultaneously: the denominator-only backoff yields valid distributions, the suffix array costs only 7 bytes/token, and the trillion-token scale makes the probability estimates useful. **The contribution is not the suffix array itself — it is the integrated system design that makes the suffix array work as a language modeling engine at unprecedented scale.**

**The democratization angle:** The paper emphasizes that infini-gram requires "no GPU, and minimal CPU / RAM," with the entire index staying on-disk and inference running on a single 8-core CPU node (Section 3, Appendix A.4). Building the index takes ~48 hours on a 128-CPU node with 1 TiB RAM — expensive but feasible for a research lab, and dramatically cheaper than pretraining a neural LM of comparable data scale. This means the ∞-gram approach is accessible to researchers and organizations that cannot afford massive GPU clusters, opening up a form of large-scale language modeling that was previously the exclusive domain of well-resourced industry labs. This is not just a performance result — it is a contribution to the accessibility and democratization of language modeling research.

**Anchor evidence:** Table 3 (latency benchmarks: <20 ms for COUNT, ~135 ms for INFGRAMPROB on 1.4T tokens), Table 6 (7 bytes/token vs. hundreds-to-thousands for vector-based approaches), Section 3 building statistics (48 hours, 128 CPUs, 1 TiB RAM for RedPajama).

## 5. Experimental Analysis

### Evaluation Methodology

- **Dataset.** The primary benchmark is the Pile's validation and test sets (Gao et al., 2020). The Pile is an 800GB dataset of diverse text spanning 22 domains (e.g., Arxiv, GitHub, PubMed, Wikipedia, books, web text). For the analyses in Section 4, the paper samples 50 documents from each domain of Pile-val, truncating each to 1024 tokens (yielding approximately 50k tokens per domain), and aggregates results across all domains. For perplexity evaluation in Section 5, documents are split into batches with a maximum sequence length of 1024 and a sliding window of 512, following the standard setup from Baevski & Auli (2019) and Khandelwal et al. (2020). The paper also evaluates on time-shifted data — newly-added Wikipedia articles from April through August 2023 — to test generalization to data created after the ∞-gram training cutoff.

- **Base model(s).** The paper evaluates ∞-gram interpolation with 14 neural LMs spanning multiple families and scales: GPT-2 (117M/345M/774M/1.6B; Radford et al., 2019), GPT-Neo (125M/1.3B/2.7B; Gao et al., 2020), GPT-J (6.7B; Wang & Komatsuzaki, 2021), Llama-2 (7B/13B/70B; Touvron et al., 2023b), and SILO PD/PDSW/PDSWBY (1.3B each; Min et al., 2023a). This range is chosen to test whether ∞-gram benefits generalize across model architectures (GPT-2's decoder-only transformer, Llama-2's rotary-position-embedding transformer, SILO's domain-restricted training), across model scales (117M to 70B), and across training data regimes (GPT-2 trained on undisclosed web text, GPT-Neo/J trained on Pile, Llama-2 trained on 2T tokens of undisclosed data, SILO trained on permissively-licensed subsets). Critically, GPT-Neo/J models are trained on the same data (Pile) that serves as ∞-gram reference data for those experiments, enabling a test of whether ∞-gram adds value even when the neural LM has already seen the reference data during pretraining.

- **Metrics.** Two primary metrics are used:
  - **Token-wise agreement accuracy** (Section 4): the fraction of evaluation tokens where the ∞-gram assigns probability > 0.5 to the actual next token in human-written or machine-generated text. This is "a lower-bound of argmax accuracy, though the gap is small" (Section 4.1). Agreement accuracy is preferred over perplexity for ∞-gram-only evaluation because the ∞-gram contains zero probabilities that would make perplexity infinite.
  - **Perplexity** (Section 5): computed as `$\text{PPL} = \exp(-\frac{1}{T}\sum_{t=1}^T \log P(y_t|x_t))$` for the combined ∞-gram + neural LM model, which is always dense (never zero) due to the neural LM component. The **relative improvement** of model $M$ over baseline $M_o$ is defined as `$(1 - \text{PPL}(M)^{-1} / \text{PPL}(M_o)^{-1}) \times 100\%$`, which measures the fraction of the perplexity gap toward perfect language modeling (PPL = 1) that has been closed.

- **Baselines.** The paper uses several baselines across the two evaluation settings:
  - **Conventional n-gram LMs (n ≤ 5):** In Section 4.1, a 5-gram LM trained on the same Pile-train data as the ∞-gram serves as the primary baseline to isolate the effect of unbounded $n$ from data scale. The 5-gram LM is implemented using the infini-gram engine with fixed $n=5$ and no backoff.
  - **Neural LMs alone (no ∞-gram):** In Section 5, the perplexity of each neural LM without any ∞-gram component is the baseline for computing relative improvement. These are standard pretrained models evaluated in the standard sliding-window perplexity setup.
  - **Neural LM + kNN-LM and RIC-LM** (retrieval-augmented baselines): In the SILO experiments (Table 2), the paper compares ∞-gram interpolation against two existing retrieval-augmentation methods: kNN-LM (Khandelwal et al., 2020) and RIC-LM (Min et al., 2023a). These use much smaller reference data (45 million to 1.2 billion tokens) compared to the ∞-gram's 360 billion tokens.
  - **Majority voting / other decoding strategies:** In the machine-generated text analysis (Section 4.2), greedy decoding, temperature sampling, and nucleus sampling (Holtzman et al., 2019) are compared against each other, with human-written text serving as an implicit reference for what "natural" agreement patterns look like.

- **Generation budget / compute accounting.** For the ∞-gram, there is no "generation budget" in the neural LM sense — the compute cost is measured by query latency and storage. The paper benchmarks latency for six query types (COUNT, NGRAMPROB, NGRAMDIST, INFGRAMPROB, INFGRAMDIST, SEARCHDOC) on two reference data scales: Pile-train (360B tokens, S = 2 shards) and RedPajama (1.4T tokens, S = 8 shards), using a single 8-core CPU node with the index stored on SSD (Table 3). Storage is measured in bytes per token: 7 bytes per token for the combined byte array + suffix array, compared to hundreds to thousands of bytes per token for vector-based nonparametric approaches (Table 6). For neural LM evaluation, standard sliding-window perplexity is used with no additional generation budget considerations.

- **Cross-validation / statistical protocol.** For the ∞-gram+neural LM interpolation in Section 5, the two hyperparameters (λ₁ for sparse estimates, λ₂ for non-sparse estimates) are tuned on the Pile validation set to minimize perplexity of the combined model, and the tuned hyperparameters are then applied to the test set. For the agreement analysis in Section 4.2, the periodic fluctuation observation under greedy decoding for Llama-2 7B is validated with a two-proportion z-test, yielding a p-value of < 10⁻⁹⁹, confirming statistical significance. No other cross-validation or statistical significance testing is reported for the main results tables.

### Main Quantitative Results

#### Agreement Between ∞-gram and Human-Written Text (Section 4.1)

The headline finding is that the ∞-gram LM achieves **47% token-wise agreement** with human-written text on Pile-val, substantially outperforming a 5-gram LM trained on the same data. The agreement is strongly correlated with effective $n$: when effective $n \geq 16$, agreement exceeds 75% (Figure 3, middle plot).

**Comparison with 5-gram LM (Figure 3, left vs. middle).** The 5-gram LM achieves substantially lower agreement than the ∞-gram. The paper notes that "over 90% tokens in the evaluation data has an effective $n$ of at least 5," and the ∞-gram analysis reveals that the median effective $n$ is 7 and the mean is 9.1. This quantifies what Figure 1 illustrates qualitatively: most tokens require more than 5 tokens of context to predict correctly, and the 5-gram LM is structurally incapable of using that context.

**Sparse vs. non-sparse estimates (Figure 3, right).** When considering only tokens with sparse ∞-gram estimates (where exactly one continuation appears in the training data), overall agreement rises to **75%**, and when effective $n \geq 14$, agreement exceeds 80%. Sparse estimates cover more than 50% of all evaluation tokens. This means that for over half of all tokens in human-written text, the longest matching suffix in the training data has only ever been followed by one token — and that token is the correct one 75% of the time.

**Disaggregation by suffix frequency (Figure 7).** Appendix Figure 7 breaks down agreement by both effective $n$ and the frequency of the longest matching suffix in the training data. The paper reports that "the count of this longest suffix in the training data does not affect agreement substantially" (Section 4.1). This is a non-obvious finding: one might expect that a suffix appearing thousands of times (providing a robust statistical estimate) would yield higher agreement than a suffix appearing only a few times, but the data shows no such trend. Even infrequently-occurring contexts yield reliable predictions when they are long enough and sparse. This suggests that the effective $n$ (the length of the matched suffix) is the dominant predictor of ∞-gram accuracy, not the frequency of that suffix.

**Qualitative patterns (Section 4.1).** The paper reports that ∞-gram is "often good at completing multi-token words (e.g., *hippopotamus*), common phrases (e.g., *born in*), and entity names (e.g., *educated at Trinity College*)." It is "not very good at recalling factual knowledge (e.g., predicting the first token of an entity name), likely due to insufficient contextualization." This qualitative characterization distinguishes the ∞-gram's strength (completing sequences where the context uniquely determines the continuation) from its weakness (predicting tokens that require world knowledge not fully captured by the local context).

#### Complementarity Between ∞-gram and Neural LMs on Human-Written Text (Section 4.1, Figure 4)

The paper's second major finding in Section 4 is that the ∞-gram and neural LMs are predictive of actual human text on **different subsets of tokens**, establishing complementarity that motivates the interpolation experiments in Section 5.

Figure 4 plots the distribution of probabilities assigned by Llama-2 70B/13B/7B to the actual next tokens in human-written text (histogram), overlaid with the ∞-gram's agreement rate for tokens in each probability bucket (green dots). The key pattern is:

> "We observe a positive, yet imperfect, correlation between neural LMs and ∞-gram regarding their agreement with the actual text. In particular, when the neural LM performance is very poor (left side of the histogram), ∞-gram still gives a non-trivial agreement of above 20%; if only considering tokens with sparse ∞-gram estimates, the agreement is as high as 50%."

This means that on the tokens where Llama-2 is most uncertain — assigning probability near zero to the actual token — the ∞-gram still correctly predicts the token 20% of the time (and 50% of the time for sparse estimates). These are tokens where the neural LM's parametric knowledge fails but exact-suffix matching succeeds. Conversely, on tokens where Llama-2 assigns high probability (the right tail), ∞-gram agreement is also high, indicating that the two models often agree on "easy" tokens. The critical region is the left tail: this is where interpolation can provide the largest gains, because the ∞-gram contributes information that the neural LM lacks entirely.

This complementarity pattern is replicated across Llama-2 7B and 13B (Appendix Figure 8), confirming that it is not specific to the 70B model scale.

#### Agreement Between ∞-gram and Machine-Generated Text (Section 4.2)

The third set of findings concerns how neural LM decoding strategies affect the agreement between generated text and the ∞-gram. The setup uses the first 50 tokens of Pile-val documents to prompt neural LMs, which then generate continuations up to the original document length or until an [EOS] token.

**Impact of decoding method (Figure 5, top row).** The top row of Figure 5 shows results for Llama-2 70B under three decoding strategies:

- **Greedy decoding:** Produces the highest effective $n$ distribution (shifted right relative to human-written text) and the highest overall agreement level. The paper interprets this as evidence that greedy decoding leads to "over-memorization of training data as well as lack of diversity" (Section 4.2).
- **Temperature sampling:** Shifts the effective $n$ distribution to the smaller side and decreases agreement.
- **Nucleus sampling (p = 0.9):** Produces an effective $n$ distribution that is "most similar to human-written text" (Figure 3, middle plot). This is consistent with the conventional wisdom that nucleus sampling produces more natural text, but the ∞-gram provides a quantitative, reference-data-grounded metric for this similarity rather than relying on human judgment.

**The fluctuation phenomenon (Figure 5, bottom row).** The bottom row reveals what the paper terms a "very curious phenomenon": under greedy decoding, the agreement level between neural LM outputs and ∞-gram **fluctuates significantly** as effective $n$ increases, while under nucleus or temperature sampling, agreement increases almost monotonically. This fluctuation becomes more pronounced for smaller models:
- For Llama-2 7B, the fluctuation is "even periodic": agreement drops rapidly at effective $n = 20, 24, 28, 32$. The paper reports this is statistically significant with a two-proportion z-test yielding p < 10⁻⁹⁹.
- For larger models (Llama-2 13B, 70B), the fluctuation is present but less periodic.
- For GPT-Neo/J models (Appendix Figure 9), similar fluctuation patterns appear, and overall agreement is higher than for Llama-2, which the paper speculates is "probably because GPT-Neo/J are trained on the same data as the ∞-gram (i.e., Pile-train)."

The paper hypothesizes that this fluctuation "may be caused by the application of positional embeddings when pretraining these Transformer-based models" (Section 4.2). This is a diagnostic finding: the ∞-gram reveals a structural artifact of Transformer decoders — periodic degradation in agreement with exact-suffix completion at specific context lengths — that would be invisible without a perfect-memorization baseline. The fact that greedy decoding shows this artifact while nucleus sampling does not suggests that the temperature=0 decoding path is particularly susceptible to positional-embedding-induced biases, perhaps because it deterministically selects the highest-probability token regardless of whether that token reflects genuine linguistic generalization or positional-artifact-driven memorization.

**Impact of model size (Figure 5, bottom row).** Increasing model size slightly increases effective $n$ and agreement level, consistent with the interpretation that larger models memorize more from training data and are more inclined to copy verbatim.

#### Perplexity Improvement from ∞-gram + Neural LM Interpolation (Section 5.2, Table 1)

The central quantitative result of Section 5 is that interpolating ∞-gram estimates with neural LMs **consistently and substantially reduces perplexity** across all 14 model configurations tested.

**Headline numbers (Table 1).** On Pile's test set:
- GPT-2 1.6B: PPL improves from 14.61 (neural only) to 9.93 (combined), a **34% relative improvement**.
- GPT-J 6.7B: PPL improves from 6.51 to 5.85, a **12% relative improvement**.
- Llama-2 70B: PPL improves from 4.65 to 4.20, a **12% relative improvement** using Pile-train alone; with Pile-train + RedPajama (1.8T tokens of reference data), PPL improves from 4.65 to 3.95, a **19% relative improvement**.
- Llama-2 13B with Pile-train + RedPajama achieves PPL 4.42, **outperforming Llama-2 70B alone** (PPL 4.65). The paper notes: "the combination of Llama-2 13B and ∞-gram with Pile-train + RedPajama outperforms Llama-2 70B, and interpolating with ∞-gram pushes the perplexity of Llama-2 70B below 4.0" (Section 5.2).

**Scaling trends within model families.** The paper identifies several patterns in how the ∞-gram benefit varies with model characteristics:

- **Within the same family, larger models benefit less (but still substantially):** GPT-2 shows improvement of 42% (117M), 35% (345M), 35% (774M), and 34% (1.6B). GPT-Neo/J shows improvement of 25% (125M), 16% (1.3B), 15% (2.7B), and 12% (6.7B). Llama-2 with Pile-train shows improvement of 16% (7B), 15% (13B), and 12% (70B). The trend is clear: diminishing returns as model capacity increases, but even the largest models see double-digit percentage improvements.

- **Across different families, the trend does not hold:** GPT-2 1.6B improves by 34%, while GPT-Neo 1.3B (a smaller model) improves by only 16%. The paper's explanation: "This may be because GPT-Neo/J models are trained precisely on Pile, while GPT-2 models are not. ∞-gram works better when the reference data distribution differs from, or complements, the pretraining data distribution." This is a crucial finding: the ∞-gram is not merely helping the neural LM recover its own training data — when the neural LM has already been trained on the reference data, the ∞-gram still provides improvements (GPT-Neo 1.3B at 16% is still substantial), but the gains are larger when the reference data introduces genuinely new information. This emphasizes "the importance of data diversity" (Section 5.2).

- **More reference data yields larger improvements:** For Llama-2 models, adding RedPajama (1.4T tokens) to Pile-train (360B tokens) increases the improvement from 14% to 22% at 7B, from 13% to 21% at 13B, and from 11% to 18% at 70B (Table 1, bottom rows). This directly demonstrates that ∞-gram performance scales with reference data size — consistent with the log-linear relationship shown in Figure 10.

**Comparison with retrieval-augmentation baselines (Table 2).** The SILO experiments (Table 2) provide a direct comparison between ∞-gram interpolation and existing retrieval-augmentation methods (kNN-LM, RIC-LM). The tabular results show:

- On Wikipedia (in-domain for SILO PD): ∞-gram improves PDSWBY from 10.76 to 8.41 (24%), while kNN-LM and RIC-LM improve the same model to 10.14 and 10.87 respectively — ∞-gram provides substantially larger improvement.
- On Enron Emails (out-of-domain for SILO PD): ∞-gram improves PD from 15.71 to 4.85 (73%), PDSW from 11.23 to 4.35 (66%), and PDSWBY from 11.52 to 4.44 (66%). The kNN-LM and RIC-LM baselines (reported only for PDSW) achieve 5.9 and 9.9 respectively on Enron Emails, vs. ∞-gram's 4.35 — a clear advantage for ∞-gram.
- On NIH ExPorters (out-of-domain but has relevant data in-domain): ∞-gram improves PDSW from 19.12 to 12.39 (37%), vs. kNN-LM's 15.0 and RIC-LM's 18.5.

The paper notes: "∞-gram yields better improvement in perplexity" compared to both baselines, while using "much larger reference data: 360-billion tokens, compared to 45-million to 1.2-billion tokens" for the baselines. However, this comparison is confounded by the reference data size — the kNN-LM and RIC-LM baselines might perform better with more reference data. The paper presents this as evidence that ∞-gram can scale to much larger reference data than vector-based approaches, not that ∞-gram is inherently superior at a fixed data scale.

**SILO permissivity trends.** The SILO results show that when the neural LM is trained on more restrictive data (PD, which trains only on public-domain data, has the most severe domain generalization challenge), the ∞-gram improvement is larger: on Enron Emails, PD improves by 73% vs. PDSWBY's 66%. The paper interprets this as: "adding the ∞-gram component is more helpful when SILO is trained on more restrictive data." This makes sense: when the neural LM's training data is severely constrained, the ∞-gram's access to a broader reference corpus provides proportionally more value.

**Evaluation on time-shifted data (Table 5).** To address concerns that the perplexity gains might be due to insufficient decontamination, the paper evaluates on Wikipedia articles created between April and August 2023 — after the cutoff dates of both Pile and RedPajama. On Llama-2 13B with Pile-train + RedPajama as reference data:

- With simple interpolation (fixed λ₁, λ₂): improvements range from 0% (July 2023) to 6% (June 2023).
- When a **Random Forest classifier** decides λ on an instance-wise basis (using suffix lengths and suffix frequencies as features): improvements range from 3% (July 2023) to 20% (April 2023).

The fact that improvements persist on time-shifted data — albeit smaller than on the original Pile test set — strengthens the case that the gains are genuine and not an artifact of test set contamination. The Random Forest results further suggest that instance-wise λ selection could be more effective than the global λ₁/λ₂ scheme, though the paper does not fully explore this direction.

### Ablation Studies and Robustness Checks

**Effect of reference data size (Figure 10):** The paper downsamples the full reference data (Pile-train) by factors of 2×, 4×, 8×, up to 256×, creating 9 progressively smaller reference datasets. The perplexity improvement brought by ∞-gram widens as reference data size grows, and "the relationship is roughly log-linear" — each doubling of reference data size yields a consistent increment in perplexity reduction. The exception is the NIH ExPorter domain, where "∞-gram doesn't help when the reference data is too small." This log-linear scaling suggests that further increasing reference data beyond 1.8T tokens would continue to yield improvements, though with diminishing marginal returns per doubling.

**Effect of reference data domain (Figure 10):** The paper compares ∞-gram performance using the full Pile against using only the in-domain portion of Pile (e.g., only Wikipedia data when evaluating on Wikipedia). The finding is that "using only the in-domain reference data is roughly as powerful as using the full reference data," implying that "almost all improvement we have achieved is thanks to in-domain data (which has been decontaminated)." This is an important negative result for practitioners: if the evaluation domain is known, a smaller, domain-specific reference corpus may be nearly as effective as a massive general corpus, reducing storage and latency costs. However, the paper notes that "it would not hurt to use the full reference data, especially when the test domain is unknown or an in-domain reference data is unavailable."

**Separate λ₁/λ₂ for sparse vs. non-sparse estimates:** The interpolation design uses two hyperparameters rather than one. The paper does not present an explicit ablation comparing single-λ vs. dual-λ interpolation, but the design choice is motivated by the Section 4 finding that sparse estimates are substantially more accurate (75% agreement vs. 47% overall). The fact that hyperparameters are tuned separately on the validation set implies that the optimal λ₁ and λ₂ could differ, and if they were consistently equal, the tuning procedure would have discovered this. The paper does not report the tuned λ values, which is a notable omission — knowing the optimal weights would help practitioners and would clarify how much the ∞-gram is trusted in sparse vs. non-sparse regimes.

**Decontamination procedure (Table 4):** The paper reports detailed decontamination statistics: 0.6% of Pile-train documents removed (1,296,376 out of 210,607,728) and 0.08% of RedPajama documents removed (730,437 out of 931,361,530). The BFF tool was run with n = 13 and an 80% overlap threshold. Some domains show much higher removal rates: GitHub (5.3% for Pile, 2% for RedPajama) and Enron Emails (2%) are notable outliers, likely due to code duplication and template-based emails respectively. The paper does not conduct an ablation to demonstrate that results are robust to different decontamination thresholds (e.g., n = 9, 11, 15; thresholds of 60%, 90%), which would strengthen the claim that remaining overlap is not inflating the results.

**Effect of tokenizer (implicit ablation):** Three separate infini-gram indexes are built — one for each tokenizer (GPT-2/Neo/J tokenizer, Llama-2 tokenizer, SILO tokenizer). The paper reports results with different models matched to their appropriate tokenizer and index. Because perplexity is not comparable across tokenizers, this serves as an implicit consistency check: the pattern of ∞-gram improvement persists across all three tokenization schemes. The paper does not explicitly compare the magnitude of improvement across tokenizers, but this would be confounded with model family and training data differences.

**SILO comparison with kNN-LM and RIC-LM (Table 2):** This is a partial ablation comparing ∞-gram against alternative nonparametric approaches. The finding that ∞-gram outperforms both baselines is reported, but with the important caveat that the baselines use much smaller reference data. The paper does not run kNN-LM or RIC-LM with the full 360B-token Pile-train reference data, so we cannot determine whether ∞-gram's advantage is due to the method (exact suffix matching vs. dense retrieval) or simply the reference data scale. This is a significant missing experiment: a fair comparison would require building kNN-LM and RIC-LM indexes at the same trillion-token scale, which the paper's own Table 6 suggests would be extremely expensive in storage (potentially hundreds of TB) — precisely the scaling limitation the paper argues ∞-gram overcomes.

**Generation experiments (negative result):** The paper reports a notable negative result in Section 5.2: "our preliminary experiments show that such method might not be helpful, and even harmful, to open-ended text generation tasks. During generation, ∞-gram can make odd mistakes (e.g., predicting totally irrelevant tokens) which makes the model to digress." No quantitative results are presented for generation experiments, but this candid admission suggests that the interpolation approach, while effective for perplexity (which evaluates the probability of given text), is not directly transferable to autoregressive generation (where the model's own outputs form the context). This is an important boundary condition on the method's applicability.

### Critical Assessment

The experiments in this paper face an unusual challenge: the ∞-gram LM produces valid probability distributions but contains zero probabilities, making perplexity infinite. The paper therefore cannot (and does not) report the most direct performance metric for a language model — standalone perplexity. Instead, it uses two workarounds: token-wise agreement accuracy (Section 4) and interpolation perplexity improvement (Section 5). Both metrics answer a narrower question than "how good is the ∞-gram LM?": agreement accuracy asks "how often does the ∞-gram assign high probability to the correct token?" (ignoring calibration and zero-probability cases), and interpolation improvement asks "how much does the ∞-gram help an already-good neural LM?" (which depends as much on the neural LM's weaknesses as on the ∞-gram's strengths). The paper is transparent about these limitations, but readers should understand that **there is no single number that directly quantifies the ∞-gram's standalone language modeling quality** — only its agreement patterns and its complementary value.

**Claim: "∞-gram has a fairly high accuracy (47%) for next-token prediction."** This is directly supported by Figure 3 and the analysis in Section 4.1. However, the accuracy metric uses a threshold of 0.5, which means a token is counted as "correctly predicted" if ∞-gram assigns it >50% probability. This is a valid but lenient metric — if ∞-gram assigns exactly 0.51 probability to the correct token and 0.49 to an incorrect one, it scores as "correct" despite being nearly uncertain. The paper acknowledges this is "a lower-bound of argmax accuracy, though the gap is small," but does not quantify the gap. An argmax accuracy metric (where ∞-gram must assign higher probability to the correct token than to any other token) would be more stringent and might lower the reported 47% figure. Additionally, the agreement analysis is conducted on Pile-val, which is from the same distribution as Pile-train (the ∞-gram reference data). On OOD data, agreement would likely be lower — the time-shifted Wikipedia evaluation in Table 5 provides some evidence that gains persist but are smaller.

**Claim: "Conventional n-grams (with small n) are insufficient in capturing a long enough context to predict the next token (29% accuracy)."** The 29% figure for the 5-gram LM's accuracy is cited in the Executive Summary but the main text of Section 4.1 reports the 5-gram comparison qualitatively (Figure 3, left plot) rather than providing an exact aggregate accuracy number. The paper's claim that "over 90% tokens in the evaluation data has an effective n of at least 5" and "median of effective n is 7" provides strong support for the insufficiency of n = 5, but the exact 29% figure should be checked against the paper's actual reported data. If it is 29%, it quantifies the gap cleanly: nearly half (47%) of tokens can be predicted with unbounded context, but only 29% with a 5-token window.

**Claim: "∞-gram can complement neural LMs and reach better performance when combined: heuristically interpolating between the estimates... can greatly reduce perplexity (by up to 73%)."** The 73% figure comes from Table 2 (SILO PD on Enron Emails: from 15.71 to 4.85, a 73% improvement). This is the maximum observed improvement and occurs in a specific configuration: a 1.3B model trained only on public-domain data (severe domain restriction) evaluated on out-of-domain text, with the ∞-gram providing access to a much larger reference corpus. On large, general-purpose models (Llama-2 70B), improvements are in the 12–19% range — still substantial but an order of magnitude smaller. The "up to 73%" framing should be understood as an upper bound under favorable conditions (small model, restricted training data, large reference corpus, OOD evaluation). The more representative number for competitive modern LLMs is 12–19% on in-distribution data.

The paper does not run a critical ablation: **what happens if you give the neural LM itself access to the same additional reference data** (e.g., by training Llama-2 on Pile-train + RedPajama rather than interpolating with ∞-gram at inference time)? The FLOPs-matched comparison that was such a strength of the original example paper is absent here. The paper demonstrates that ∞-gram + neural LM beats neural LM alone, but does not compare against the alternative of simply training a better neural LM on more data. This is understandable for a systems paper — the goal is to demonstrate the ∞-gram's value as an inference-time component — but it means the paper does not establish whether test-time ∞-gram interpolation is *more efficient* than pretraining on more data, which would be a stronger claim.

**Claim: "When analyzing machine-generated text, we also observe irregularities in the machine–∞-gram agreement level with respect to the suffix length, which indicates deficiencies in neural LM pretraining and the positional embeddings of Transformers."** The fluctuation analysis in Figure 5 is the paper's most novel analytical contribution, and it is well-supported by the data — the periodic drops for Llama-2 7B at n = 20, 24, 28, 32 are statistically significant and visible in the plots. However, the causal attribution to "positional embeddings" is presented as a hypothesis ("we suspect that this may be caused by..."), not a demonstrated mechanism. The paper does not run experiments that would confirm this hypothesis, such as testing a model without positional embeddings (e.g., a Transformer with learned position-agnostic representations), testing with different positional embedding schemes (rotary vs. absolute vs. ALiBi), or testing with varying context window sizes to see if the fluctuation period changes. The claim that these fluctuations "indicate deficiencies" is therefore plausible and intriguing but not rigorously established — the fluctuations could be an artifact of greedy decoding's determinism interacting with training-data-specific patterns (e.g., document boundaries at common lengths) rather than a fundamental Transformer architectural property. A useful ablation would be to compare greedy decoding from a model with and without positional embeddings, or to analyze whether the fluctuation periods align with the model's pretraining context window or positional embedding dimension.

**Missing experiment: the hyperparameter reporting gap.** The paper does not report the tuned values of λ₁ and λ₂ for any model configuration. Knowing these values would be informative: do they approach 1.0 (heavily trusting the ∞-gram) or stay near 0.1–0.3 (using the ∞-gram as a weak prior)? Do λ₁ and λ₂ differ substantially, confirming the value of the sparse/non-sparse distinction? For a paper whose practical contribution is enabling interpolation, the absence of these hyperparameters makes it difficult for practitioners to reproduce or approximate the results. Similarly, the Random Forest instance-wise λ experiment in Table 5 is mentioned but not described in sufficient detail for replication.

**Missing experiment: latency amortization in evaluation.** The paper benchmarks query latency (Table 3) and reports amortized per-token latency for dense ∞-gram computation (12–20 ms for consecutive tokens on Pile-train). However, the evaluation in Section 5 processes documents with a sliding window of 512 and maximum sequence length of 1024 — this creates overlapping contexts that could amortize some ∞-gram computation (the suffix matches for one window may share prefixes with the next window). The paper does not report end-to-end perplexity evaluation wall-clock time, so the practical cost of the interpolation approach remains somewhat abstract. On a 500-document test set with ~1M tokens, and 12–20 ms per token, the ∞-gram component alone would take ~3.3–5.5 hours of CPU time — this is feasible but not negligible, and the paper could have reported this to help practitioners assess the cost-benefit trade-off.

**Dataset and model scope limitations.** All perplexity results are on the Pile and its domains. The paper does not evaluate on other standard language modeling benchmarks (WikiText-103, PG-19, Lambada, etc.) or on generation-focused benchmarks. The agreement analyses are limited to human-written text from Pile-val and machine-generated continuations of Pile-val prompts. The findings about effective-$n$ distributions, sparsity patterns, and decoding-method agreement may be specific to the Pile's document composition and may not generalize to other corpora with different document lengths, domain mixes, or languages. The paper acknowledges this implicitly by releasing its tools for others to replicate the analyses on their own corpora, but the reported numbers should be understood as Pile-specific until external validation is performed.

**The absence of a standalone ∞-gram perplexity baseline.** The paper argues compellingly that ∞-gram alone would have infinite perplexity due to zero probabilities. However, it does not experiment with simple smoothing techniques (e.g., add-λ smoothing, or backing off to a unigram distribution with small probability mass) that could produce a finite standalone perplexity. Even a heavily smoothed ∞-gram perplexity would provide a direct comparison point with neural LMs and would quantify how much of the interpolation gain comes from the ∞-gram's structure vs. from the neural LM baseline being weak in certain regions. This omission is somewhat justified by the paper's framing (the ∞-gram is a diagnostic tool and complement, not a standalone model), but it does leave a gap in the quantitative characterization of the ∞-gram's absolute performance.

**Strengths that merit acknowledgment.** Despite these limitations, the experimental framework has several notable strengths. The scope of neural LMs tested (14 configurations across 5 model families from 117M to 70B parameters) is unusually comprehensive for a single paper. The decontamination procedure is reported in unusual detail (Table 4), including domain-level breakdowns that let readers assess residual contamination risk. The time-shifted evaluation (Table 5) directly addresses the most obvious alternative explanation for the perplexity gains (data leakage). And the open-ended admission that the approach does not (yet) help with text generation — a negative result that many papers would omit — strengthens the credibility of the positive results that are reported.

## 6. Limitations and Trade-offs

### The Difficulty Estimation Step Consumes More Compute Than the Headline Efficiency Numbers Account For

**The assumption:** The compute-optimal framework in Section 3.2 requires estimating question difficulty *before* allocating the inference budget, which the paper does by generating 2048 samples per question and computing the pass@1 rate (oracle difficulty) or averaging PRM scores (predicted difficulty). The paper explicitly acknowledges that this step incurs significant computation that is not included in the reported efficiency gains:

> "estimating difficulty in this way still incurs additional computation cost during inference... our experiments do not account for this cost largely for simplicity" (Section 3.2).

**The consequence:** The headline finding — that compute-optimal scaling achieves up to 4× better efficiency than best-of-N — is computed **assuming difficulty is already known for free**. In a real deployment, the total cost would be difficulty estimation plus strategy execution. Since difficulty estimation uses 2048 samples (far more than most test-time compute budgets studied, which max out at 256–512 generations), the difficulty estimation step alone could dominate the total cost. For a budget of 64 generations where the paper claims 4× efficiency, the difficulty estimation overhead is 32× the strategy execution cost — potentially making the net result *less* efficient than simply running best-of-256 without any difficulty estimation at all. The paper frames this as an "exploration-exploitation tradeoff" (Section 3.2) but provides no experimental results that amortize the difficulty estimation cost over multiple queries or that demonstrate net-positive efficiency when difficulty estimation is included.

**What evidence exists in the paper:** The 2048-sample difficulty estimation procedure is described in Section 3.2. The efficiency claims are presented in Figures 4 and 8, where the x-axis shows "generation budget" for the execution phase only, with no accounting for the 2048-sample upfront cost. The curves for "oracle" and "predicted" difficulty largely overlap, showing that the PRM-based estimation works without ground-truth labels — but neither curve includes the estimation cost. The paper does not report any experiment measuring end-to-end efficiency including difficulty estimation, nor does it report how the $4\times$ figure changes if difficulty estimation is amortized across $k$ queries from the same difficulty distribution.

**Mitigation status:** The paper acknowledges this as "a key avenue for future work" (Section 3.2) and suggests training models to predict difficulty directly from the question text, or estimating difficulty from a small initial sample rather than 2048 samples. No such methods are developed or evaluated. The difficulty estimation cost remains a gap between the reported efficiency gains and what a practitioner would experience in deployment. Until this gap is closed, the $4\times$ improvement figure should be understood as an **upper bound** conditioned on free difficulty information, not a realized deployment gain.

---

### Hard Problems Show Near-Zero Improvement Regardless of Budget, Establishing a Capability Ceiling

**The assumption:** The paper's approach assumes that the base model already produces correct solutions at some non-trivial rate — that is, the proposal distribution contains the right answer somewhere, and test-time compute helps find or refine it. This assumption fails on the hardest problems, where the base model's pass@1 is effectively zero.

**The consequence:** On difficulty bin 5 (the hardest quintile), all methods — search, revisions, and compute-optimal combinations — produce **near-zero accuracy regardless of compute budget**. In Figure 3 (right), bin 5 accuracy stays at 1–3% for all search methods from 4 to 256 generations. In Figure 7 (right), bin 5 shows 2–3% accuracy across all sequential-to-parallel ratios. In the FLOPs-matched comparison (Figure 9), the bin 5 line is essentially flat near 0–5%, even with a budget equivalent to a ~14× larger model's pretraining. The paper is candid about this:

> "On the hardest questions (bin 5), no method makes meaningful progress" (Section 5.3).

This is a fundamental bound, not an engineering limitation. Test-time compute can **amplify existing capability** — it helps find and refine solutions the model already "knows" somewhere in its distribution — but it cannot **create new capability**. If the base model never produces a correct solution for a class of problems, no amount of search, revision, or adaptive allocation will help. This means the approach offers no path forward for genuinely novel reasoning, out-of-distribution generalization, or problems that exceed the base model's pretraining coverage. For such problems, scaling pretraining remains the only viable option (as the FLOPs-matched results confirm: on hard problems, the ~14× larger model substantially outperforms test-time compute at high $R$ values).

**What evidence exists in the paper:** Figure 3 (right, bin 5 subplot), Figure 7 (right, bin 5 subplot), Figure 9 (bin 5, blue line), and the FLOPs-matched analysis in Section 7. The paper reports -52.9% relative disadvantage for hard problems under PRM search at $R \gg 1$ compared to the larger model with greedy decoding.

**Mitigation status:** The paper does not attempt to address this limitation and does not propose methods for extending the approach to problems beyond the base model's capability. The limitation is inherent to the framework: if no correct solution exists in the proposal distribution, no amount of test-time optimization can find one. Scaling pretraining (more data, more parameters, better architecture) remains the only demonstrated path for such problems. The paper's explicit characterization of this boundary — where test-time compute stops helping — is itself a contribution, but it is a hard limitation for practitioners considering this approach for challenging reasoning tasks.

---

### All Results Are on a Single Benchmark (MATH) with a Single Model Family (PaLM 2), Leaving Generalization Unverified

**The assumption:** The paper assumes that the patterns observed — difficulty-dependent optimal strategies, verifier over-optimization behavior, revision model benefit regimes, and the compute-optimal policy structure — generalize across benchmarks, model architectures, and task domains. The authors state they "believe this model is representative of the capabilities of many contemporary LLMs" (Section 4), but this is a claim, not a finding.

**The consequence:** Multiple aspects of the results could be specific to PaLM 2-S* and the MATH benchmark:

- **PRM quality and over-optimization behavior:** The PRM is trained via Monte Carlo rollouts from PaLM 2-S*. A model with different calibration, different error patterns, or different output distribution could yield a PRM with different scaling properties, shifting the difficulty thresholds at which search over-optimizes. The beam search degradation on easy problems (Figure 3, right) might occur at different budgets or with different severity for other models.

- **Revision model training:** The revision model's ability to learn from edit-distance-paired incorrect-correct trajectories depends on the base model's in-context learning behavior and output distribution. The paper's finding that ReST$^{EM}$ optimization degraded revision performance (Appendix K, Figure 16) suggests the approach is sensitive to training methodology — and this sensitivity may itself be model-dependent.

- **MATH-specific characteristics:** MATH consists of competition-level math problems with structured, multi-step symbolic reasoning and exact-answer verification. It is unclear whether the difficulty-dependent patterns (beam search hurting easy problems, sequential revisions dominating on easy problems, parallel exploration needed for hard problems) generalize to other reasoning domains (code generation, logical deduction, scientific QA), to tasks requiring factual recall rather than inference, or to open-ended generation where correctness is ambiguous.

- **Test set size:** The 500-question MATH test set, split into five difficulty quintiles (~100 questions each), further split by two-fold cross-validation, means the compute-optimal policy is selected based on ~50 questions per fold per bin. The paper does not report confidence intervals on the compute-optimal scaling curves, so the reliability of the selected strategies at this sample size is unknown.

**What evidence exists in the paper:** All figures and tables in Sections 5–7 use PaLM 2-S* and the MATH benchmark. There is no cross-model or cross-benchmark evaluation. The paper acknowledges this implicitly by not claiming broader generalization, but also does not discuss it as a limitation.

**Mitigation status:** None. The paper does not run experiments on other benchmarks (e.g., GSM8K, HumanEval, ARC) or with other base models. The authors' statement that PaLM 2-S* is "representative" is unverified. Replication on additional model families and benchmarks would be required to determine which findings are universal and which are specific to this model-benchmark combination.

---

### The $14\times$ Larger Pretraining Baseline Uses Greedy Decoding with No Test-Time Compute, Making the FLOPs-Matched Comparison Asymmetric

**The assumption:** The FLOPs-matched comparison in Section 7 compares PaLM 2-S* with compute-optimal test-time strategies against a ~14× larger model using **only greedy decoding** — no majority voting, no best-of-N, no search, no revision. The paper also scales only model parameters (not training data), following the LLaMA paradigm rather than Chinchilla-optimal pretraining:

> "We choose this setting as it is representative of a canonical approach to scaling pretraining compute and leave the analysis of compute-optimal scaling of pretraining compute where the data and parameters are both scaled equally to future work" (Section 7).

**The consequence:** The comparison answers: "Is test-time compute with a small model better than greedy decoding from a larger model?" But it does not answer the more natural question: "Given a fixed total FLOPs budget, should I spend it on pretraining a larger model or on test-time compute with a smaller model?" The asymmetry favors test-time compute in two ways:

- **The larger model gets no test-time augmentation.** A fairer comparison would give the larger model some test-time compute budget (even best-of-8 or best-of-16 would create a stronger baseline) because in practice, anyone deploying a large model could also apply test-time strategies. The paper demonstrates that test-time compute can match the larger model's greedy performance, but not that test-time compute is *more FLOPs-efficient* than pretraining when both are allowed the same inference-time strategies.

- **The larger model may not be compute-optimally trained.** A Chinchilla-optimal model that scales both data and parameters with the increased FLOPs budget would likely outperform a parameter-only-scaled model, making the pretraining baseline stronger than what the paper tests.

The consequence is that the reported advantages — e.g., +27.8% on easy questions at $R \ll 1$ — may shrink or disappear against a properly compute-optimal and test-time-augmented larger model. The paper's claim is not false (the comparison is clearly described), but it answers a narrower question than the framing suggests.

**What evidence exists in the paper:** Section 7 describes the FLOP accounting and the baseline explicitly. Figure 9 and the bar charts in Figure 1 present the results. The paper acknowledges the parameter-only scaling choice as a departure from Chinchilla-optimal pretraining. No experiment gives the larger model any test-time compute budget.

**Mitigation status:** The paper explicitly scopes this as future work: "we leave the analysis of compute-optimal scaling of pretraining compute where the data and parameters are both scaled equally to future work" (Section 7). The missing comparison — giving the larger model some test-time compute — is not discussed as a limitation, but it weakens the practical interpretation of the FLOPs-matched results. A practitioner choosing between pretraining and inference compute would need experiments where both options are evaluated with the best available inference-time strategies.

---

### Verifier Over-Optimization Is a Hard Ceiling That the Compute-Optimal Policy Mitigates But Does Not Solve

**The assumption:** The PRM provides a reliable signal for guiding search and selecting answers. The paper's compute-optimal policy routes easy problems away from aggressive search to avoid over-optimization, implicitly assuming the PRM signal is reliable enough on medium problems to justify beam search.

**The consequence:** The verifier over-optimization problem is the **primary bottleneck preventing unbounded scaling of test-time compute**. Evidence is clear throughout Section 5:

- Beam search **degrades** easy-problem performance at high budgets (Figure 3, right, bins 1–2) — the PRM assigns high scores to incorrect solutions that exploit its imperfections.
- Lookahead search — the most powerful optimizer — paradoxically **underperforms** all other methods at the same budget (Figure 3, left), because its deeper optimization amplifies verifier errors more than it improves solution quality.
- Qualitative examples (Appendix M, Figures 29 etc.) show beam search producing degenerate outputs — repetitive low-information steps, overly short solutions — that score highly under the PRM but are incorrect.
- Even on medium problems where beam search is deployed (bins 3–4), performance **flattens or declines** at high budgets (Figure 3, right), consistent with over-optimization setting a ceiling.

The compute-optimal policy works around this by routing easy problems to best-of-N (which is a weaker optimizer and thus less susceptible to over-optimization) and using beam search only on medium problems where the PRM signal still provides net positive guidance. But this is a **mitigation, not a solution**. On medium problems, the verifier ceiling still limits how far beam search can scale — the gains from additional compute diminish and eventually vanish. The paper cannot answer: "If we had a perfect verifier, how much better could test-time compute perform?" The current results are fundamentally bounded by the PRM quality achievable with the Monte Carlo rollout training procedure (Appendix D).

**What evidence exists in the paper:** Figure 3 (right) showing beam search degradation on bins 1–2 and plateau on bins 3–4, Figure 3 (left) showing lookahead search underperformance, and qualitative examples in Appendix M. The paper explicitly identifies over-optimization in Section 5.3: "the degradation at high budgets is attributed to over-optimization of the PRM."

**Mitigation status:** The compute-optimal policy partially mitigates this by avoiding aggressive optimization where the verifier is unreliable, but the underlying problem — that the PRM is imperfect and its errors are amplified by search — is not solved. The paper does not experiment with verifier improvements (adversarial training, ensembling, KL-constrained search) that might push the over-optimization threshold higher. Improving verifier robustness is flagged as future work implicitly (the paper's identification of the bottleneck is itself a contribution), but no concrete methods are proposed or tested. For practitioners, this means the approach cannot be scaled to arbitrarily large inference budgets with current PRM quality — there is a hard ceiling determined by the verifier's reliability that the compute-optimal policy can only work around, not break through.

---

### The Revision Model and PRM Search Are Never Combined, Leaving the Strongest Potential Hybrid Unexplored

**The assumption:** The paper studies two complementary axes — modifying the proposal distribution (revisions) and improving selection (PRM search) — but treats them as independent mechanisms that the compute-optimal policy chooses between. There is an implicit assumption that the observed difficulty-dependent patterns for each mechanism in isolation will also characterize their combined performance, or that combining them is a straightforward extension.

**The consequence:** The paper's results represent a **lower bound** on what a fully integrated system could achieve. The two mechanisms have complementary strengths that the paper itself documents: revisions excel on easy problems (local refinement of roughly-correct answers), PRM search excels on medium problems (global exploration of different solution strategies). A combined system could, for example:

- Use the revision model as the proposal distribution within beam search — at each step of the search tree, the model conditions on previous rejected branches, potentially generating higher-quality candidate steps than the base model alone.
- Use the PRM to guide which revisions to pursue — rather than blindly generating a long revision chain, use per-step PRM scores to decide when a revision is on track vs. when to restart from scratch.
- Use the PRM to select among revision chain outputs, combining the proposal-improvement of revisions with the selection-power of the verifier.

The absence of combined experiments means we don't know whether the gains from revisions and search are additive, sub-additive (because they overlap in the problems they help), or super-additive (because they help on different problems and the combination covers more of the difficulty spectrum). We also don't know whether the compute-optimal policy for the combined system would differ from the independent policies — the difficulty thresholds and strategy choices might shift when both tools are available.

**What evidence exists in the paper:** The paper explicitly acknowledges this gap in Section 8: "we did not experiment with PRM tree-search techniques in combination with revisions." Sections 5 and 6 report search and revision results separately, with independent compute-optimal strategies (Figures 4 and 8). No combined experiments are reported.

**Mitigation status:** The paper frames this as future work ("we leave this to future work," Section 8) but does not speculate on whether the combination would yield gains beyond the best of each individual method. For practitioners, this means there is an unexplored opportunity for further improvement, but also uncertainty about whether the two mechanisms interact in complex ways (e.g., the revision model's output distribution might interact differently with the PRM than the base model's). The paper's architecture makes combining them conceptually straightforward — both mechanisms operate on the same generation-then-score framework — but the experimental cost of running the full cross-product of search strategies × revision strategies × difficulty bins × budgets is likely why the paper omits it. Until this combination is tested, the reported results understate the potential of the overall approach but also leave open questions about practical integration.

## 7. Implications and Future Directions
- How this changes the landscape
  - Demonstrates that simple, interpretable, nonparametric signals from massive corpora can materially improve strong neural LMs and diagnose their behavior. This re‑opens n‑gram LMs as practical tools at modern scales.
  - Provides infrastructure (web UI, API, Python package) to query trillions of tokens for counts, probabilities, and document retrieval (§Abstract; Figures 11–16).

- Follow‑up research enabled/suggested
  - Better integration for decoding: learn context‑aware λ or gating policies; fuse `∞‑gram` with neural decoders in ways robust to off‑topic suggestions (§5.2 note).  
  - Investigate positional‑embedding or training‑data effects underlying the agreement oscillations under greedy decoding (Figure 5; §4.2).  
  - Retrieval‑augmented modeling at pretraining scale using exact n‑gram retrieval instead of or alongside vector search (discussion §E; Table 6 comparisons).
  - Adaptive, instance‑wise interpolation (e.g., the Random Forest success on time‑shifted data; Table 5) and learning to predict “sparse” cases that deserve high λ.

- Practical applications
  - Corpus understanding and curation: membership checks, contamination detection, removal of toxic/PII n‑grams, and attribution via document lookup (`SEARCHDOC`; §E; §A.5; Figures 11–16).  
  - Reducing hallucinations by preferring corpus‑attested continuations in factual settings (§E).  
  - Auditing memorization/plagiarism: measure long n‑gram overlaps between generated text and training corpora (§E).  
  - Scaling retrieval for nonparametric LMs: `infini‑gram` provides a compact, exact index at web‑scale (Table 6) that can complement or replace vector‑only stores in some pipelines.

> Bottom line: By making unbounded n‑gram statistics cheap to query at trillion‑token scale and showing they both predict next tokens and substantially lower perplexity when combined with neural LMs (Tables 1–2, Figure 4), this work elevates n‑grams from a historical baseline to a modern, scalable component for analysis and modeling.
