# Scaling Retrieval-Based Language Models with a Trillion-Token Datastore

**ArXiv:** [2407.12854](https://arxiv.org/abs/2407.12854)

## 🎯 Pitch

This paper introduces datastore size as a crucial and previously underexplored scaling dimension for language models, complementing model size and pretraining data. By developing MASSIVEDS—the largest and most diverse open-source retrieval datastore at 1.4 trillion tokens—and a compute-efficient pipeline, the authors show that increasing datastore size yields monotonic gains in language modeling and broad downstream tasks, allowing smaller models with large datastores to outperform much larger LM-only models for the same training compute. This work fundamentally expands the roadmap for language model scaling and paves the way for more efficient, knowledge-rich, and broadly applicable AI systems.

---

## 1. Executive Summary

This paper studies how scaling the size of an inference-time datastore improves retrieval-based language models across language modeling and diverse downstream tasks, constructing **MassiveDS**, a 1.4-trillion-token open-source datastore spanning eight domains, and designing an efficient pipeline that reduces the cost of datastore scaling experiments by more than an order of magnitude. The authors systematically evaluate two complementary mechanisms—**datastore scaling** (varying the number of tokens available at inference time for document retrieval) and **compute-optimal scaling** (jointly optimizing pretraining FLOPs, model size, and datastore size)—finding that datastore scaling monotonically improves performance without saturation, enabling a smaller retrieval-based model to outperform a larger LM-only counterpart on knowledge-intensive tasks. The paper demonstrates that retrieval-based LMs achieve superior compute-optimal scaling trends, with a small model augmented by a large datastore matching or exceeding the performance of a much larger LM-only model at the same training cost, while establishing that these gains are task-dependent—substantial on factoid QA but limited on reasoning-heavy benchmarks when the base model lacks sufficient capability to utilize retrieved knowledge.

## 2. Context and Motivation

### The Core Problem: Retrieval-Based LMs Have No Scaling Law for Datastore Size

The fundamental question this paper tackles is deceptively straightforward: **if you give a retrieval-based language model a larger datastore to draw from at inference time, how much does performance improve, and does it ever stop improving?** This matters because, while the field has developed reasonably mature scaling laws for pretraining LMs—the Chinchilla laws (Hoffmann et al., 2022) tell us how to optimally allocate compute between model parameters and training tokens—there is no analogous understanding of how performance scales with the size of the retrieval datastore used during inference.

This gap is significant for several reasons the paper surfaces throughout the introduction and related work:

- **Cost asymmetry between training and indexing.** Indexing a datastore (running one forward pass with a small retriever model to generate embeddings) is dramatically cheaper than pretraining on the same tokens (which requires forward and backward passes with a much larger model). This means that "offloading" knowledge from model parameters into a datastore could be far more FLO P-efficient than memorizing that knowledge during pretraining—but only if we can quantify how much datastore size is needed to achieve equivalent or better performance.
- **Model deployment economics.** If small models augmented with large datastores can match or exceed the performance of much larger LM-only models, organizations could deploy smaller, cheaper models backed by datastores, shifting compute from expensive pretraining to cheaper indexing and retrieval. This has direct implications for on-device deployment, cloud API costs, and inference latency.
- **General-purpose vs. task-specific retrieval.** Most prior work uses small, single-domain datastores (typically Wikipedia, ~5 billion tokens) that are carefully matched to evaluation tasks. This requires knowing the task domain in advance and curating a custom datastore for each use case—defeating the generality that makes LLMs attractive. It remains unknown whether a single broad datastore can simultaneously improve performance across many domains, or whether the retriever simply gets confused by out-of-domain data.

### Prior Work and Its Limitations

**The dominant paradigm: small, task-specific datastores.** The bulk of retrieval-based LM research has converged on a specific pattern: use a datastore constructed from Wikipedia, approximately 5 billion tokens, retrieved via a dense retriever like DPR or Contriever, with documents prepended in context before few-shot examples (Ram et al., 2023; Shi et al., 2023; Asai et al., 2024a). This paradigm works well—retrieval from Wikipedia reliably improves open-domain QA—but it is fundamentally limited in three ways: (1) Wikipedia is small and covers a limited set of topics, so the benefits of retrieval saturate once the model can already access all relevant Wikipedia passages; (2) Wikipedia is exactly in-domain for many benchmark tasks (TriviaQA, Natural Questions are built from Wikipedia), making it unclear whether retrieval helps when the datastore is broader and noisier; (3) there is no scaling curve—with a single fixed datastore size, you cannot study how performance changes with datastore scale.

**The trillion-token exceptions: RETRO and its descendants.** The paper explicitly positions itself against two landmark efforts that did scale retrieval datastoers to the trillion-token range:

- **RETRO** (Borgeaud et al., 2022) is the most directly comparable prior work. It introduced a custom transformer architecture with chunked cross-attention to retrieved neighbors and built a 1.7-trillion-token datastore from MassiveText (Rae et al., 2022), a proprietary corpus. However, RETRO has three critical limitations that this paper aims to overcome. First, RETRO evaluated its trillion-token datastore **only on language modeling perplexity**, not on downstream tasks; for those, it used a small Wikipedia datastore. This leaves open the question of whether scaling the datastore helps on tasks people actually care about. Second, RETRO used a **custom, non-standard architecture** that requires training from scratch—it cannot be used with off-the-shelf LMs or even fine-tuned models, which limits its practical adoption. Third, RETRO's datastore and training data are **proprietary**, making it impossible for the research community to replicate, extend, or systematically ablate.
- **RETRO++ and InstructRETRO** (Wang et al., 2024) extend the RETRO architecture with instruction tuning but inherit the same limitations: proprietary data, custom architecture, and no downstream evaluation at the trillion-token scale.

**SPHERE** (Piktus et al., 2022) is a complementary effort that built an open-source 90-billion-token datastore from CCNet (Wenzek et al., 2020), but their downstream evaluations on KILT (Petroni et al., 2021) showed that SPHERE does not always outperform a small, in-domain Wikipedia datastore. This finding—that more data doesn't necessarily help—highlights precisely the uncertainty that this paper addresses: under what conditions does datastore scaling actually improve downstream performance?

**The missing dimension in scaling laws.** Scaling laws research has focused overwhelmingly on pretraining: Kaplan et al. (2020) established power-law relationships between model size and loss; Hoffmann et al. (2022) refined this to jointly optimize model size and data quantity; Muennighoff et al. (2023) studied scaling under data constraints; Gadre et al. (2024) extended scaling law analysis to downstream tasks. These formulations share a common structure: performance is a function of pretraining compute, which decomposes into model parameters and pretraining tokens. The paper argues that this picture is incomplete because it ignores a third knob: the datastore. Since indexing a datastore requires FLOPs—just far fewer per token than pretraining—there should exist a three-way compute-optimal allocation between model parameters, pretraining tokens, and datastore tokens. Prior to this paper, no such analysis existed.

### The Computational Barrier to Studying Datastore Scaling

Beyond the intellectual gap, there is a **practical barrier** that has prevented the community from studying datastore scaling, and the paper frames overcoming this barrier as a major part of its contribution. Conducting a scaling study requires building many datastores at different sizes with different compositions, random seeds, and data preprocessing configurations—and then evaluating all of them on multiple tasks with multiple models. A naive approach (subsample raw data → filter → index → retrieve → evaluate, repeated for every configuration) would require re-indexing trillions of tokens hundreds of times, which is prohibitively expensive even for well-resourced research groups.

The paper's novel **MassiveDS pipeline** (detailed in Section 3.2 and Appendix A) is motivated by this computational reality. The key insight is that indexing and retrieval are the expensive operations, while filtering and subsampling are cheap. By reordering the pipeline—indexing the full datastore once, retrieving a large number of documents (K=1000) for each query, and then applying filtering, deduplication, and subsampling to only these pre-retrieved sets—the authors reduce compute by more than an order of magnitude. They provide a theoretical proof (Lemma A.3) that this reordering is equivalent to the naive pipeline with high probability, where the failure probability is exponentially small in K. This pipeline innovation is what makes the scaling study accessible on an academic budget and is positioned as an enabling contribution of independent value.

### How This Paper Positions Itself

The paper positions itself not as proposing a new retrieval architecture or training algorithm, but as **performing the first systematic study of datastore scaling on downstream tasks using open-source, off-the-shelf components**. The deliberate choices reflect this:

- **Retrieve-in-context (RIC-LM) architecture**: Rather than a custom architecture like RETRO, the paper uses the simple approach of prepending retrieved documents to the input context, which works with any existing LM (even black-box API models). This choice is explicitly made to ensure the findings apply broadly rather than being tied to a specific architecture.
- **Off-the-shelf retriever (Contriever-MSMARCO)**: A standard dense retriever with 177M parameters, chosen after ablation showing it performs comparably to newer retrievers (Appendix E.1). This demonstrates that the scaling benefits are not dependent on an exotic retriever.
- **MassiveDS: the largest open-source datastore for retrieval-based LMs**: At 1.4 trillion tokens spanning eight domains (including general web data from CommonCrawl at five time periods, plus specialized domains like scientific papers, code, math, and biomedicine), MassiveDS is deliberately diverse to test whether a single broad datastore can serve multiple tasks.
- **Compute-optimal scaling curves**: The paper directly reframes retrieval-based LMs in the language of scaling laws, plotting Pareto frontiers over pretraining FLOPs that include datastore construction as a third axis alongside model size and pretraining tokens. This connects the work to the broader scaling laws literature while arguing that datastore size should be "considered as an integral part of LM efficiency and performance trade-offs."

The paper's framing also serves to reconcile conflicting signals in prior work. RETRO showed language modeling gains from datastore scaling but didn't test downstream tasks. SPHERE showed that a larger datastore doesn't always beat Wikipedia on downstream tasks. The paper argues that a systematic study—with careful decontamination, data quality controls, and a diverse datastore—is needed to resolve when and why scaling helps. The answer, previewed in the paper's findings, is nuanced: scaling helps substantially on knowledge-intensive tasks (TriviaQA, NaturalQuestions) but is limited on reasoning-heavy tasks where either the model lacks the capacity to use retrieved knowledge or the datastore lacks task-relevant content.

### Addressing Common Misconceptions

Several intuitions about retrieval-based LMs might lead readers to underestimate the significance of a datastore scaling study. The paper addresses these implicitly:

**Misconception 1: "Just make the datastore as big as the entire web—why study scaling?"** The paper's results show that performance continues to improve without obvious saturation even at 1.4 trillion tokens, suggesting there is still headroom. Moreover, building and serving a web-scale datastore with a retriever is computationally expensive, and the scaling curves inform when the marginal benefit of more data is worth the marginal cost.

**Misconception 2: "The retriever is the bottleneck—just improve that."** While the paper's reranking analysis (Section 5.2) shows that better retrieval would improve performance, the datastore scaling curves themselves demonstrate that even with a fixed off-the-shelf retriever, more data helps. Improving both retriever and datastore are complementary directions.

**Misconception 3: "Retrieval-based LMs just cheat by retrieving the exact test data."** The paper tackles this head-on with its decontamination analysis (Section 5.3), showing that even after aggressive decontamination (removing documents with 8-gram overlap with the answer), retrieval continues to benefit language modeling—indicating that semantically similar documents, not just verbatim copies, drive the improvements.

The paper's central framing is that the field has been implicitly treating datastore size as a fixed hyperparameter rather than a scaling dimension, and that this has obscured the full picture of what retrieval-based LMs can achieve. By constructing a trillion-token open datastore, building a computationally efficient pipeline for scaling experiments, and evaluating across models, tasks, and compute budgets, the paper aims to establish datastore scaling as a first-class axis of LM scaling alongside model size and pretraining data.

## 3. Technical Approach

### 3.1 Reader Orientation

This paper builds an **experimental infrastructure**—a trillion-token datastore paired with a computationally efficient pipeline—to systematically measure how varying the amount of text available for retrieval at inference time affects the performance of retrieval-based language models on both language modeling and diverse downstream tasks. The system solves the problem that **studying datastore scaling is prohibitively expensive with a naive approach** because it requires re-indexing and re-retrieving from trillions of tokens for every experimental variation (datastore size, data composition, random seed for subsampling, preprocessing filters); the solution is a **pipeline that reorders operations so that the most expensive steps (indexing and retrieval) run once and are shared across all subsequent variants**, reducing compute by more than an order of magnitude while being provably equivalent to the naive approach with high probability.

### 3.2 Big-Picture Architecture (Diagram in Words)

The system has five major components, organized as a linear pipeline that processes raw text into evaluation-ready retrieval-augmented prompts:

1. **Raw Data Pool** — 1.4 trillion tokens across eight domains (Common Crawl at five time periods, C4, scientific papers, books, encyclopedias, StackExchange, GitHub code, mathematical web pages, biomedical articles) stored as fixed-size 256-word chunks. This is the full universe of text from which datastores of varying sizes will be constructed.

2. **Distributed Indexing and Retrieval Module** — uses a Contriever-MSMARCO dense retriever (177M parameters) to embed every chunk in the raw data pool and build searchable indices, then for each evaluation query retrieves the top-K=1000 documents from each domain shard in parallel. This step runs exactly once for the full datastore.

3. **Domain Merging and Post-Hoc Processing Module** — merges retrieved documents across domains, then applies operations that would normally happen to the raw data (decontamination, deduplication, quality filtering) to only the pre-retrieved top-1000 documents per query rather than the entire trillion-token corpus.

4. **Data Subsampling Module** — for each experimental configuration (combination of subsampling ratio `$p$` and random seed), selects documents i.i.d. with probability `$p$` from the processed top-1000 pool, then takes the top-k=3 from the survivors. This emulates having retrieved from a datastore of size `$p \times \text{total tokens}$` without re-indexing.

5. **Evaluation Module** — prepends the final top-3 retrieved documents (in reverse order, so higher-ranked documents are closer to the query) before few-shot examples and the question, then measures perplexity (for language modeling) or exact match/accuracy (for downstream tasks) using off-the-shelf LMs from the Llama-2, Llama-3, Pythia, and OLMo model families.

Information flows: raw text → [one-time indexing] → document embeddings → [per-query retrieval] → top-1000 per domain → [domain merging] → merged top-1000 per query → [filtering/deduplication/decontamination] → cleaned top-1000 → [subsampling with probability p] → subsampled pool → [take top-3] → prepended to prompt → LM evaluation.

### 3.3 Roadmap for the Deep Dive

- **First**, the formal equivalence between the MassiveDS pipeline and the naive pipeline—why reordering operations works, and when it fails—because the entire experimental methodology rests on this reordering being valid.
- **Second**, the datastore construction process (chunking, indexing, distributed retrieval, domain merging) with concrete details on the raw data composition, chunking granularity, retriever choice, and sharding strategy, since these design decisions affect what scaling trends mean in practice.
- **Third**, the post-hoc processing operations (deduplication, decontamination, quality filtering) and their implementation on pre-retrieved documents rather than the raw corpus, including the specific thresholds and algorithms used, because these filters critically affect whether scaling trends reflect genuine retrieval benefit or data leakage.
- **Fourth**, the subsampling procedure and its statistical justification (the tail-bound analysis proving equivalence), since this is the mechanism that enables exploring many datastore sizes without re-indexing.
- **Fifth**, the evaluation setup including prompt format, few-shot configuration, metric computation, and compute-optimal scaling FLOPs accounting, since the paper's primary claims depend on comparing models under a unified compute budget.

### 3.4 Detailed, Sentence-Based Technical Breakdown

This is primarily an **empirical methodology and infrastructure paper** whose core idea is that by building the right pipeline, the research community can study datastore scaling as systematically as we study model and data scaling—and that doing so reveals consistent, unsaturated improvements from larger datastores that have been invisible in prior work limited to single-domain, few-billion-token datastores.

---

#### The MassiveDS Pipeline: Why Reordering Operations Is Valid

The central computational insight enabling this work is that several operations in the standard datastore construction pipeline can be reordered without changing results—and that this reordering shifts expensive operations from being repeated (once per experimental configuration) to being run once and cached.

**The naive pipeline** would proceed in this order for every experimental variation (each combination of subsampling ratio `$p$`, random seed `$s$`, filter configuration, data composition):

1. Subsample the raw data (select each chunk i.i.d. with probability `$p$` using seed `$s$`)
2. Apply data filtering (deduplication, decontamination, quality filters) to the subsampled raw data
3. Build an index over the subsampled, filtered data
4. For each query, retrieve the top-k documents from this index
5. Evaluate

The computational bottleneck is **Step 3 (indexing)**, which requires running a forward pass through the retriever model for every token in the subsampled datastore. For a trillion-token datastore with a 177M-parameter retriever, this step dominates all others. In the naive pipeline, this step must be repeated for every single `$(p, s)$` pair—with, say, 7 subsampling ratios and 3 random seeds, that is 21 full re-indexing operations, each processing up to 1.4 trillion tokens.

**The MassiveDS pipeline** reorders these operations:

1. Build ONE index over the FULL raw data (all 1.4 trillion tokens, `$p=1$`)
2. For each query, retrieve a LARGE number of documents (`$K=1000$`) from this full index
3. Apply filtering (decontamination, deduplication, quality filters) to only the top-1000 per query
4. Subsample from these pre-retrieved, filtered top-1000 with probability `$p$` and seed `$s$`
5. Take the top-k=3 from the subsampled pool
6. Evaluate

**The equivalence argument (Lemma A.3 in the paper)** states that this reordering produces the same final top-3 documents as the naive pipeline with high probability, as long as `$K$` is chosen large enough that the subsampling step yields at least 3 surviving documents. Formally:

> "Subsampling from the retrieved top-K documents with probability p and then taking the top-k (k ≪K) from the subsampled documents (Algorithm 2) is equivalent with high probability to directly retrieving top-k documents from a datastore that is subsampled from the entire raw text datastore with probability p (Algorithm 1). The equivalence holds as long as there are enough k documents left after subsampling."

The number of documents surviving subsampling follows a binomial distribution `$\text{Binomial}(K, p)$`. With `$K=1000$` and even the smallest subsampling ratio `$p=0.01$`, the probability of having fewer than `$m=3$` documents remaining is approximately `$1 - 0.9973 = 0.0027$` (Table 4 in the paper). This failure probability is exponentially small in `$K$`, so the pipeline is effectively equivalent to the naive approach.

**Why this form:** The insight is that individual document retrieval scores are computed independently (the Contriever uses element-wise inner-product scoring—each document's relevance to the query is computed without dependence on other documents). This means whether a document would appear in the top-k of a subsampled datastore depends only on (a) whether it was subsampled and (b) its retrieval score relative to other subsampled documents. If we retrieve many more documents (`$K \gg k$`) from the full datastore and then subsample, documents that would have been in the top-k from the subsampled full index are guaranteed to be among the pre-retrieved pool as long as enough documents survive. The element-wise nature of the scoring (Definition A.1 and Lemma A.1 in the paper) ensures that any document that ranks in the global top-k of a subsampled index must also rank in the top-K of its individual shard—and therefore will be captured in the pre-retrieval step.

The same commutativity argument extends to other element-level operations like reranking and data quality filtering (Lemma A.4): operations that are conditioned on a single document independently of others can be freely reordered with subsampling. Set-level operations like deduplication (which depends on pairs of documents) are not strictly commutable, but the paper shows that running deduplication on the pre-retrieved top-K is equivalent to deduplication on the full index as long as no deduplication pair straddles the top-K boundary—and with `$K=1000$`, documents outside the top-1000 have retrieval scores low enough that they would not affect the top-3 final selection anyway.

The computational savings are dramatic: the complexity of the naive pipeline is `$O((1 + M + |Q|)|P||S|N)$` where `$M=177\text{M}$` (retriever parameters), `$|Q|$` is the number of evaluation queries, `$|P|$` is the number of subsampling ratios, `$|S|$` is the number of random seeds, and `$N$` is the number of documents. The dominant term is `$O(M|P||S|N)$`—the cost of repeatedly indexing the subsampled datastore. The MassiveDS pipeline complexity is `$O(N(M + |Q|) + K|P||Q||S|)$`, dominated by `$O(MN)$`—the cost of one-time indexing of the full datastore. Since `$|P||S| \approx 21$` in the paper's experiments, this represents more than an order-of-magnitude reduction.

---

#### Datastore Construction: Raw Data, Chunking, and Indexing

**Raw data composition.** MassiveDS is assembled from 1,441.2 billion tokens of text spanning eight domains, sourced from both general web data and specialized corpora (Table 2 in the paper):

| Domain | Datasets | Size (Billions of tokens) |
|---|---|---|
| BOOKS | RedPajama Books | 26.3 |
| STEM | peS2o, RedPajama ArXiv | 97.7 |
| ENCYCLOPEDIA | DPR 2018 Wiki, RedPajama 2022 Wiki | 31.9 |
| FORUM (Q&A) | RedPajama StackExchange | 20.2 |
| CODE | RedPajama Github | 52.8 |
| MATH | OpenWebMath, NaturalProofs | 14.1 |
| BIOMEDICAL | PubMed | 6.5 |
| GENERAL WEB | RedPajama CC (2019–2023), RedPajama C4 | 1191.7 |
| **Total** | | **1441.2** |

The token count is measured using the Llama-2 tokenizer (Touvron et al., 2023), providing a consistent unit across all experiments.

**Domain-specific sources** are individually smaller but generally higher in quality: peS2o (Soldaini & Lo, 2023) provides open-access scientific papers processed specifically for pretraining; OpenWebMath (Paster et al., 2023) contains mathematical web pages; NaturalProofs (Welleck et al., 2021) provides mathematical language for theorem proving; PubMed (National Library of Medicine, 2023) provides biomedical abstracts and articles. The **general web data** is sourced from Common Crawl snapshots at five distinct time periods (July 2019, May 2020, April 2021, May 2022, June 2023) processed through the RedPajama pipeline (Computer, 2023), plus C4 (Raffel et al., 2020). This temporal diversity is deliberate: it tests whether the retriever can find relevant information across different web snapshots rather than relying on a single static crawl.

**Chunking.** Before indexing, all raw data is split into fixed-size passages of at most 256 words each. This granularity is chosen to balance two competing needs: chunks must be small enough that retrieved documents contain focused, relevant information (a 1000-word chunk might contain only one relevant sentence buried in noise), but large enough that the retriever has sufficient context to assess relevance (a 10-word chunk provides almost no semantic signal). The 256-word level is standard in dense retrieval work (Karpukhin et al., 2020) and roughly corresponds to a short paragraph.

**Indexing.** Each 256-word chunk is passed through the Contriever-MSMARCO dense retriever (Izacard et al., 2022), a dual-encoder model with 177M parameters trained with contrastive learning on the MS MARCO passage ranking dataset. The final-layer representation of the [CLS] token is taken as the document embedding. These embeddings are stored without compression (flat index via FAISS IndexFlatIP), meaning retrieval computes the inner product between the query embedding and every document embedding—an exact but computationally intensive search.

The paper deliberately chooses a flat index rather than an approximate nearest neighbor index (like IVFADC, Jégou et al., 2011) for two reasons. First, exact search eliminates approximation noise from the scaling study, ensuring that observed trends reflect genuine retrieval quality rather than index approximation artifacts. Second, the FLOPs accounting for datastore construction simplifies: flat indexing requires zero additional operations beyond the initial embedding forward pass, so the FLOPs cost of datastore construction equals the FLOPs cost of embedding all tokens, which is:

> $$\text{FLOPs}_{\text{datastore}} \approx 2 N_{\text{retriever}} D_{\text{datastore}}$$

where `$N_{\text{retriever}} = 177 \times 10^6$` is the number of retriever parameters and `$D_{\text{datastore}}$` is the number of tokens in the datastore.

**What this computes:** the total floating-point operations needed to build the datastore index. The factor of 2 comes from the standard approximation that one forward pass through a transformer requires approximately `$2ND$` FLOPs (one multiply-add per parameter per token). This is in contrast to pretraining FLOPs, which require:

> $$\text{FLOPs}_{\text{pretrain}} \approx 6 N_{\text{LM}} D_{\text{pretrain}}$$

where the factor of 6 accounts for one forward pass (`$2ND$`) plus one backward pass (`$4ND$`, since backpropagation requires roughly twice the computation of the forward pass).

**Why this form:** The ratio `$N_{\text{retriever}} / N_{\text{LM}}$` drives the cost asymmetry between indexing and pretraining. For a Llama-2 7B model, `$N_{\text{LM}} = 7 \times 10^9$`, so `$N_{\text{retriever}}/N_{\text{LM}} \approx 0.025$`. Including the factor of 3 difference from the forward-backward ratio, indexing one token costs approximately `$0.025 \times 3 = 0.075$` times what pretraining on that same token would cost. This cost asymmetry is the economic foundation for the paper's central argument: knowledge is cheaper to store in a datastore than to memorize in parameters.

**Sharding.** To parallelize indexing and retrieval, the raw data from each domain is split into `$m$` shards. The number of shards is determined by domain size: `$m=32$` for each time slice of Common Crawl and C4 (the largest domains), and `$m=8$` for smaller domains. Each shard is embedded independently and stored as a separate FAISS index. At retrieval time, the top-K=1000 documents are fetched from each shard in parallel, then merged across shards within each domain.

**Lemma A.1** in the paper proves that distributed retrieval across `$m$` shards is exactly equivalent to retrieving from a single unsharded index. The proof observes that any document ranking in the global top-K across all documents must rank in the top-K within its own shard (since its global rank bounds its within-shard rank), and conversely, any document in the merged pool of `$mK$` shard-level top-K results that ranks in the overall top-K must have fewer than K documents with higher scores globally. This means the sharding strategy introduces no approximation error—it is an implementation detail for parallelism, not a methodological compromise.

---

#### Distributed Retrieval and Domain Merging

**Per-query retrieval.** For each evaluation query (whether a language modeling prefix or a downstream task question), the query text is encoded with the same Contriever model to produce a query embedding. The inner product between this query embedding and every document embedding in a shard is computed, and the 1000 documents with the highest scores are returned from each shard. This is done in parallel across all shards.

**Domain merging.** After per-shard retrieval, the results are first merged within each domain (taking the union of top-1000 results from all shards in that domain and re-ranking by retrieval score to keep the top-1000 per domain). Then, for a chosen combination of domains, the per-domain top-1000 pools are merged into a single pool of `$D \times 1000$` documents (where `$D$` is the number of included domains), and the top-1000 are selected from this merged pool by retrieval score.

The merged top-1000 documents per query are cached. This caching is a crucial efficiency decision: it allows the data composition to be varied (e.g., comparing performance with all domains vs. only Wikipedia vs. only Common Crawl) without re-running retrieval. The per-domain cached results are simply merged in different combinations, which is computationally trivial.

---

#### Post-Hoc Processing: Deduplication, Decontamination, and Quality Filtering

With the top-1000 documents per query retrieved and cached, the pipeline applies three processing operations that would normally be run on the raw data before indexing. By applying them to only the pre-retrieved documents per query, the pipeline avoids processing trillion-token-scale corpora repeatedly.

**Global deduplication.** Although MassiveDS's source corpora (particularly RedPajama) have undergone local deduplication within shards, the paper observes that many duplicates persist across shards—a document may appear in multiple Common Crawl snapshots or in both Common Crawl and C4. These duplicates are problematic for retrieval because they waste the top-k budget on redundant information.

The deduplication method uses **MinHash with 13-gram Jaccard similarity** following Computer (2023). For each pair of documents in the top-1000 pool, the 13-gram Jaccard similarity is computed (the size of the intersection of their 13-gram sets divided by the size of the union). If the similarity exceeds 80%, the pair is considered a duplicate, and the document with the lower retrieval score is removed. The 13-gram granularity and 80% threshold are standard choices from the deduplication literature (Lee et al., 2022) that balance removing near-duplicates against accidentally removing topically related but distinct documents.

The paper explicitly notes that the original MinHash implementation skips chunks with fewer than 13 grams (about 13 words). Qualitative inspection revealed that many short, nonsensical phrases under 13 words remained in the deduplicated pool, so the paper additionally removes all documents with fewer than 13 words at this stage. Figures 14 and 15 in the paper demonstrate that without this short-chunk removal, the retriever frequently retrieves documents with high lexical overlap with the query (e.g., a 5-word phrase matching a question substring) but containing no useful information for answering—and that removing short chunks significantly improves downstream QA performance.

**Decontamination.** Test-set leakage is a particularly acute concern for retrieval-based LMs because the model can retrieve the exact test data verbatim at inference time (Borgeaud et al., 2022). The paper implements decontamination at two levels of stringency depending on the evaluation type.

For **downstream tasks**, the standard decontamination method is 13-gram Jaccard similarity between the question text and the retrieved document, with removal at a threshold of 80% similarity. This catches near-verbatim copies of test questions in the datastore.

For **language modeling evaluation**, where the model must predict a continuation from a prefix, a stricter two-pronged approach is used. The target sequence (the second half of each 1024-token evaluation chunk, i.e., 512 tokens) is compared against each retrieved document:
1. **13-gram Jaccard similarity** with 80% threshold (same as downstream tasks)
2. **Longest continuous n-gram overlap**: if any contiguous sequence of at least 32 tokens appears in both the document and the target, the document is removed

The 32-gram threshold corresponds to approximately 6.25% of the target sequence length (32/512). This continuous-overlap filter catches cases where a document contains a substantial verbatim excerpt from the target without having high overall Jaccard similarity (e.g., a document that is 10% identical to the target but where that 10% is a single unbroken quote).

The aggressive decontamination variant used in the Section 5.3 ablation reduces the continuous-overlap threshold to 8 grams (approximately 1.56% of target length), creating a much stricter filter.

**Equivalence of post-hoc decontamination (Lemma A.2).** The paper provides a formal justification that running decontamination on the pre-retrieved top-1000 is equivalent to retrieving from a pre-decontaminated datastore. The proof observes that whether a document is removed by decontamination is a deterministic function of its content and the evaluation data—it does not depend on other documents in the datastore. Therefore, any document that survives decontamination in the pre-retrieved pool would also survive if the entire datastore were decontaminated first, and vice versa. The document's retrieval score relative to other surviving documents is unchanged. As long as the pre-retrieval step captures enough documents that the final top-k can be formed from the survivors, the results are identical.

**Quality filtering.** The paper additionally experiments with three quality filters adapted from the Dolma corpus preprocessing pipeline (Soldaini et al., 2024):
1. **Whitespace filter**: removes documents with fewer than a manually defined threshold of whitespace-separated tokens (filtering out documents that are essentially empty or contain only markup)
2. **Language filter**: uses a FastText language identification model (Bojanowski et al., 2017) to detect the document's language and removes documents with low model confidence in the primary language
3. **Alphanumeric filter**: removes documents containing no alphanumeric characters or containing spans of all-punctuation characters exceeding a threshold length (filtering out documents that are purely formatting, tables of symbols, or machine-generated garbage)

The paper finds that these quality filters have relatively limited effect (Figure 13), hypothesizing that the source datasets (particularly RedPajama) have already undergone similar preprocessing, so additional filtering removes few documents beyond what the original processing already handled.

---

#### Data Subsampling: Emulating Variable-Size Datastores

The critical operation that enables studying datastore scaling without re-indexing is **post-hoc subsampling** of the pre-retrieved documents. The core idea is that retrieving from a datastore of size `$p \times \text{total}$` (where `$p \in \{0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0\}$`) is statistically equivalent to retrieving a larger pool from the full datastore and then randomly discarding documents with probability `$1-p$` before taking the top-k.

**Algorithm 2 (the paper's efficient subsampling):**

For each query `$q$`, subsampling ratio `$p$`, and random seed `$s$`:

1. Take the pre-retrieved, filtered top-1000 documents for query `$q$`
2. For each document, draw an independent Bernoulli(`$p$`) random variable using seed `$s$` to determine inclusion
3. From the surviving documents, select the top-k=3 by original retrieval score

The function `$\text{SUBSAMPLE}(D, p, s)$` is defined such that for a fixed `$(p, s)$`, the same document always receives the same inclusion decision—enabling reproducibility across experiments.

**Statistical justification (Lemma A.3).** The paper provides a tail-bound analysis showing that the probability of failure (having fewer than `$k=3$` documents survive subsampling) is:

> $$P(\text{failure}) = P(\text{Binomial}(K, p) < k)$$

For the smallest subsampling ratio `$p=0.01$` and `$K=1000$`, the expected number of surviving documents is `$1000 \times 0.01 = 10$`. The probability of getting fewer than 3 is `$P(\text{Binomial}(1000, 0.01) \leq 2) \approx 1 - 0.9973 = 0.0027$`. For larger `$p$`, the failure probability is effectively zero (Table 4 shows `$\geq 1.0$` for all `$p \geq 0.05$`).

**What this computes:** the probability that a particular `$(p, s)$` configuration fails to produce the minimum 3 documents needed for evaluation. The binomial distribution arises because each of the `$K$` pre-retrieved documents is independently included or excluded. The paper does not implement a fallback mechanism because the failure rate is "very low in our experiments," but notes that a fallback (re-running with a larger `$K$` on failure) could be added for safety.

**Why this form:** The i.i.d. Bernoulli subsampling of pre-retrieved documents exactly mirrors the process of subsampling the raw datastore and then retrieving, because document retrieval scores are element-wise (computed independently per document). Whether a document would appear in the top-k of a subsampled datastore depends on its original retrieval score and which other documents happen to be sampled—and the i.i.d. Bernoulli process on the pre-retrieved pool reproduces exactly this random subset selection. The only approximation is that the pre-retrieval step might miss a document that would have ranked in the top-k from a particular subsampled datastore because it was not in the top-1000 of the full datastore—but for `$K \gg k$` and reasonable `$p$`, such documents have such low retrieval scores that they would almost certainly be outranked by surviving documents from the top-1000 anyway.

**Random seeds and confidence intervals.** For each subsampling ratio `$p$`, the pipeline is run with three different random seeds (100, 101, 102), producing three independent subsamples of the pre-retrieved pool. The final performance metric (perplexity or accuracy) is computed for each seed, and the mean and confidence interval across seeds are reported in the scaling plots. This captures the variance introduced by which specific documents are included in a datastore of a given size—an important source of uncertainty that naive single-seed subsampling would mask.

**The x-axis representation.** The datastore size plotted on the x-axis of all scaling figures is computed as `$p \times \text{total raw tokens}$` (e.g., at `$p=0.01$`, the plotted size is `$0.01 \times 1441.2\text{B} \approx 14.4\text{B}$` tokens). The paper notes that this is technically an overestimate: since some documents are filtered out during deduplication, decontamination, and quality filtering, the actual number of usable tokens is smaller by a factor `$p_f$` (the fraction of data surviving filtering, where `$0 < p_f \leq 1$`). However, because the x-axis is plotted on a log scale and the filtering fraction `$p_f$` is approximately constant across subsampling ratios, this corresponds to a constant horizontal shift in the plotted curves and does not change the observed scaling trends—the slopes and saturation behavior are unaffected.

---

#### Evaluation Setup: Prompt Format, Few-Shot Configuration, and Metrics

**Language modeling evaluation.** Following Baevski & Auli (2019), Khandelwal et al. (2020), and Min et al. (2023a), evaluation data is split into fixed-length chunks of 1,024 tokens with a stride of 512 tokens. For each chunk:
- The first half (512 tokens) serves as both the **retrieval query** and the **prefix** for the LM
- The second half (512 tokens) serves as the **target sequence** for perplexity computation

This sliding-window approach with 50% overlap ensures that target tokens are always preceded by 512 tokens of context that the model has access to for both retrieval and generation. The retrieved top-3 documents are prepended before the full chunk.

Perplexity is computed as:

> $$\text{PPL} = \exp\left(-\frac{1}{T}\sum_{t=1}^{T}\log P(y_t | x, r_1, r_2, r_3, y_{<t})\right)$$

where `$T=512$` is the number of target tokens, `$y_t$` is the t-th target token, `$x$` is the 512-token prefix, and `$r_1, r_2, r_3$` are the three retrieved documents prepended before the prefix.

**What this computes:** the exponentiated average negative log-likelihood of the target tokens given the prefix and retrieved documents. Lower perplexity means the model assigns higher probability to the actual continuation, indicating that the retrieved documents are helping the model predict the text.

**Why this form:** perplexity is the standard intrinsic metric for language modeling quality and is directly comparable across models of different sizes. The exponential of the average log-loss ensures that perplexity has a natural interpretation as the "effective branching factor"—a perplexity of 4.0 means the model is as uncertain as if it were choosing uniformly among 4 equally likely next tokens on average.

The paper evaluates language modeling on data from two domains: general web data sampled from RedPajama (Computer, 2023) and scientific paper data sampled from S2ORC (Lo et al., 2020). These represent different types of text with different predictability characteristics, testing whether datastore scaling benefits transfer across domains.

**Downstream task evaluation.** For five downstream tasks (TriviaQA, Natural Questions, MMLU, MedQA, and—in the complete appendix results—additional tasks), the evaluation uses 5-shot prompting. The exact prompt format is:

```
[Retrieved Document 3 (lowest-ranked but still in top-3)]
[Retrieved Document 2]
[Retrieved Document 1 (highest-ranked)]
[5 Few-Shot Examples with questions and answers]
[Target Question]

[Model generates answer]
```

Documents are concatenated in **reverse order** of retrieval rank, so that the highest-ranked document (with the best retrieval score) appears closest to the target question. This design choice is motivated by the "lost in the middle" phenomenon (Liu et al., 2023), where language models attend more strongly to information at the beginning and end of the context and tend to ignore information in the middle. By placing the best document closest to the question, the model is more likely to use it.

The paper explicitly tested two prompt formats:
1. Few-shot examples first, then retrieved documents, then the question
2. Retrieved documents first, then few-shot examples, then the question

The second format performed better because "the LM can learn the few-shot pattern better when the few-shot examples are closer to the question" (Appendix B.3). This is a practical finding: the model needs the few-shot format close to where it generates the answer to follow the pattern, while the retrieved documents serve as background knowledge that can be placed further away.

**Downstream metrics:**
- **TriviaQA and Natural Questions**: exact match (EM) between the generated answer and the ground-truth answer string, following standard evaluation protocols (Joshi et al., 2017; Kwiatkowski et al., 2019)
- **MMLU and MedQA**: accuracy on multiple-choice selection (the model's highest-probability choice among A/B/C/D options must match the ground-truth), following Hendrycks et al. (2021) and Jin et al. (2020)

The paper uses the lm-evaluation-harness framework (a widely used open-source evaluation suite) adapted to support prepending retrieved documents before the few-shot examples.

---

#### Compute-Optimal Scaling: FLOPs Accounting and Pareto Frontiers

The compute-optimal scaling analysis in Section 4.3 requires a unified accounting of FLOPs spent on both pretraining and datastore construction, so that retrieval-based LMs and LM-only models can be compared under the same total compute budget.

**Pretraining FLOPs.** Following the standard approximation from the scaling laws literature (Kaplan et al., 2020; Hoffmann et al., 2022):

> $$\text{FLOPs}_{\text{pretrain}} \approx 6 N_{\text{LM}} D_{\text{pretrain}}$$

where `$N_{\text{LM}}$` is the number of non-embedding parameters in the language model and `$D_{\text{pretrain}}$` is the number of pretraining tokens. The factor 6 comes from `$2$` (forward pass) `$+$` `$4$` (backward pass), where the backward pass coefficient of 4 (rather than 2) accounts for the additional computation of gradients with respect to activations in backpropagation.

**Datastore construction FLOPs:**

> $$\text{FLOPs}_{\text{datastore}} \approx 2 N_{\text{retriever}} D_{\text{datastore}}$$

where `$N_{\text{retriever}} = 177 \times 10^6$` and `$D_{\text{datastore}} = p \times 1.441 \times 10^{12}$` tokens (the subsampled datastore size). The factor 2 represents one forward pass only—datastore construction does not require backward passes.

**Total training-time FLOPs for retrieval-based LMs:**

> $$\text{FLOPs}_{\text{total}} = 6 N_{\text{LM}} D_{\text{pretrain}} + 2 N_{\text{retriever}} D_{\text{datastore}}$$

**What this computes:** the total floating-point operations invested in creating the model and its datastore, excluding inference-time retrieval cost (which the paper discusses separately). For LM-only models, the second term is zero.

**Why this form:** This decomposition captures the core economic tradeoff. The cost ratio per token between pretraining and indexing is `$3 \times (N_{\text{LM}} / N_{\text{retriever}})$`. For a 7B-parameter LM with a 177M-parameter retriever, this ratio is `$3 \times (7000/177) \approx 119$`—pretraining on one token costs approximately 119 times as many FLOPs as indexing that same token. This means that for the FLOPs cost of pretraining on a single additional token, you could index approximately 119 tokens in the datastore. If those 119 tokens contain knowledge that improves the model's effective performance, indexing is a vastly more FLOP-efficient way to incorporate that knowledge.

**Intermediate checkpoints as proxies for training budget.** The paper uses intermediate checkpoints released by the Pythia (Biderman et al., 2023) and OLMo (Groeneveld et al., 2024) projects as approximations of models trained with different numbers of tokens. For Pythia, checkpoints at `$1/30, 1/15, 1/10, 1/5, 1/4, 1/3, 1/2,$` and all of the full 300B-token corpus are used for models of 1B, 2.8B, 6.9B, and 12B parameters. For OLMo, checkpoints at `$1/50, 1/40, 1/20, 1/9, 1/8, 1/7, 1/6,$` and all of the full corpus are used for the 1B (trained on 3T tokens) and 7B (trained on 2T tokens) models.

The paper acknowledges a limitation of this approach: these intermediate checkpoints share the same learning rate scheduler with a fixed maximum number of training steps that equals or exceeds what they have actually been trained for. As a result, "the performance of these intermediate checkpoints (with or without retrieval) might be lower than otherwise attainable with the same amount of compute"—a model trained from scratch with a scheduler designed for the specific token budget would likely perform better. However, pretraining LMs from scratch at all model sizes and token budgets is "prohibitively expensive for an academic budget."

**Pareto frontier construction.** For each total training FLOPs budget (plotted on the x-axis in log scale in Figure 4), the Pareto-optimal point is the model configuration (pretraining tokens + datastore size, for retrieval-based LMs) or (pretraining tokens only, for LM-only models) that achieves the highest downstream accuracy. The Pareto frontier connects these points, showing the best achievable performance at each budget level. The paper highlights these Pareto-optimal points in red for retrieval-based LMs and blue for LM-only models, enabling direct visual comparison.

**Inference cost discussion.** The paper explicitly notes that the compute-optimal analysis above focuses only on training-time compute, and that inference costs differ between approaches. Prepending retrieved documents increases inference cost due to the extended context length (the model must process retrieved documents in addition to the prompt) and the retrieval search itself (computing query-document similarities at inference time). On the other hand, retrieval-based LMs can use smaller models that are cheaper per inference token. The paper does not compute an inference-compute-optimal scaling curve, noting that "there is emerging work on accelerating retrieval search and designing efficient serving strategies for retrieval-based LMs" and leaving this to future work.

---

#### Model and Retriever Configuration Details

**Language models evaluated.** The paper uses four model families to ensure the scaling trends are not specific to a particular architecture or training recipe:

- **Llama-2** (Touvron et al., 2023): 7B and 13B parameter models
- **Llama-3** (Touvron et al., 2023): 8B parameter model
- **Pythia** (Biderman et al., 2023): 1B, 2.8B, 6.9B, and 12B parameter models, with intermediate checkpoints available throughout training
- **OLMo** (Groeneveld et al., 2024): 1B and 7B parameter models, with intermediate checkpoints available

The diversity of model families is important because different pretraining recipes (data mixture, training duration, learning rate schedule) can affect how well a model utilizes retrieved context. Finding consistent scaling trends across families strengthens the claim that datastore scaling is a general phenomenon.

**Retriever choice justification.** The paper uses Contriever-MSMARCO (Izacard et al., 2022), a 177M-parameter dense dual-encoder trained with contrastive learning, as the primary retriever. An ablation in Appendix E.1 compares Contriever against two alternatives on a 10% subsample of MassiveDS:
- **DRAGON-RoBERTa** (Lin et al., 2023): 110M parameters
- **GTR-T5-Base** (Ni et al., 2021): 110M parameters

All three retrievers perform similarly on language modeling perplexity, Natural Questions, and MMLU (Table 6 in the paper). The paper selects Contriever for the full-scale study because its implementation (from Facebook Research) runs "much faster than the sentence-transformer implementations" of the alternatives—a practical engineering consideration for trillion-token indexing.

**Retrieval hyperparameters.** Across all main experiments:
- Number of retrieved documents for final evaluation: `$k=3$`
- Number of pre-retrieved documents for subsampling: `$K=1000$`
- Subsampling ratios: `$p \in \{0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0\}$`
- Random seeds for subsampling: `$\{100, 101, 102\}$`
- Reranking: not applied in main experiments (studied separately in Section 5.2)

The choice of `$k=3$` is standard in retrieval-augmented generation work; the choice of `$K=1000$` is governed by the tail-bound analysis showing it provides sufficient margin for subsampling to the smallest ratios with negligible failure probability.

---

#### Reranking: Improving Retrieval Quality Post-Hoc

The Section 5.2 analysis studies how improving retrieval quality through reranking affects the datastore scaling trends. Reranking is a two-stage retrieval process: a fast, approximate retriever (Contriever) first fetches a larger pool of candidates, then a slower but more accurate model reranks these candidates.

**Cross-encoder reranker.** The paper uses `MINI-LM-L12 V2`, a BERT-based cross-encoder with approximately 33M parameters trained on MS MARCO for passage ranking. Unlike the dual-encoder Contriever (which encodes the query and document independently and compares them via dot product), a cross-encoder processes the concatenation of query and document together through self-attention, allowing the model to model fine-grained interactions between query and document tokens. This makes cross-encoders significantly more accurate but also much slower—processing every query-document pair requires a full transformer forward pass, whereas the dual-encoder pre-computes document embeddings and only encodes the query at inference time.

In the reranking pipeline, Contriever first retrieves 500 documents, the cross-encoder scores each by encoding the concatenation `[CLS] query [SEP] document [SEP]` and taking the scalar similarity output, and the documents are reordered by cross-encoder score. The top-3 from the reranked list are used for final evaluation.

**Lexical oracle reranker.** To establish an upper bound on what improved retrieval could achieve, the paper implements an oracle reranker that has access to the ground-truth answer. For knowledge-intensive QA tasks, this oracle scores each document by:
1. If the gold answer string appears verbatim in the document, assign the maximum score
2. Otherwise, compute the fraction of unigram overlap between the document and the answer string

This is a simple heuristic but captures the intrinsic retrievability of answer-containing documents—if a document contains the answer but the cross-encoder fails to rank it highly, that indicates room for retriever improvement. Conversely, if no document in the top-500 contains the answer even approximately, the problem is in the datastore coverage, not the retriever.

**Key finding from reranking experiments.** Figure 6 shows that the cross-encoder reranker consistently improves performance on TriviaQA and Natural Questions across all datastore sizes, but a "notable gap persists between the oracle reranker and the cross-encoder-based reranker." This suggests that improving retrieval quality (whether through better dense retrievers, improved rerankers, or hybrid retrieval strategies) is a complementary direction that could further steepen the datastore scaling curves, pushing performance higher at each datastore size.

---

#### Decontamination and Filtering Ablation: Separating Real Gains from Data Leakage

Section 5.3 systematically varies the decontamination and filtering pipeline to verify that the observed scaling benefits are not artifacts of test-set contamination.

**Decontamination levels compared (Figure 7):**
1. **No decontamination**: all pre-retrieved documents are used regardless of overlap with evaluation data
2. **Standard decontamination** (default): 13-gram Jaccard similarity at 80% threshold for downstream tasks; 13-gram Jaccard + 32-gram longest continuous overlap for language modeling
3. **Aggressive decontamination**: 8-gram longest continuous overlap for all tasks. This is described as "a strict filter, as 8-gram overlap occurs frequently even when documents are not nearly identical."

**Results for language modeling (Figure 7, RedPajama PPL):** Without decontamination, perplexity is substantially lower (better) across all datastore sizes, which the paper interprets as evidence that "the benefits in language modeling primarily arise from lexical overlap." However, even after aggressive decontamination, retrieval continues to improve perplexity over the LM-only baseline at all datastore sizes—"indicating that semantically similar retrieved documents with minimal lexical overlap can still enhance language modeling." This is a critical finding: it distinguishes genuine semantic retrieval benefit from shallow pattern matching of verbatim text.

**Results for downstream tasks (Figure 7, NQ):** Decontamination level does not significantly affect NQ performance, which the paper attributes to "less contamination of NQ in the datastore" (the Natural Questions test set was created from Google search queries mapped to Wikipedia answers, and since Wikipedia is a small fraction of MassiveDS, few test questions appear verbatim in the datastore). Interestingly, decontamination actually decreases performance at small datastore sizes but improves final performance at larger scales—a non-monotonic effect that the paper observes but does not fully explain, possibly related to the removal of marginally helpful documents that at small scales outnumbers the removal of harmful contaminated ones.

**Deduplication and quality filtering ablation (Figure 13).** Comparing with and without global MinHash deduplication:
- Language modeling: negligible impact
- Natural Questions: deduplication helps minimize saturation as the datastore scales; without deduplication, performance plateaus earlier, presumably because subsampling with higher `$p$` increases the chance of including duplicate documents that waste retrieval slots without adding new information

Dolma quality filters (whitespace + language + alphanumeric) show limited effect, which the paper attributes to the source corpora already having undergone similar filtering.

---

#### Summary of Key Design Choices and Their Justifications

- **Flat index over approximate index**: eliminates approximation noise from the scaling study, enabling clean measurement of datastore size effects. The tradeoff (higher inference-time search cost) is acceptable because the paper's primary contribution is about scaling trends, not production efficiency, and exact search guarantees that observed trends are not artifacts of index approximation.
- **Pre-retrieval with K=1000 followed by subsampling**: the core enabling innovation that makes the study computationally feasible. The equivalence proof (Lemma A.3) and tail-bound analysis provide theoretical guarantees that make this more than a heuristic approximation.
- **Contriever over newer alternatives**: empirically equivalent performance on a 10% subsample, with faster implementation—a practical choice for trillion-token indexing.
- **Post-hoc decontamination over pre-indexing decontamination**: avoids processing the full corpus against every possible evaluation set, which at 1.4 trillion tokens would be extremely expensive. The equivalence proof (Lemma A.2) ensures this is not a methodological compromise.
- **Reverse-order document concatenation**: places the highest-ranked document closest to the query, exploiting the "lost in the middle" effect to maximize the chance the model attends to the best retrieval.
- **Retrieved documents before few-shot examples**: empirically better than the reverse order, as the model better learns the few-shot pattern when the examples are close to the generation point.
- **512-token prefix for perplexity with 256-word document chunks**: the prefix length is substantially longer than individual document chunks, meaning the retrieved documents serve as additional context beyond what the prefix alone provides, rather than simply contributing one more data point.
- **Three random seeds per subsampling ratio**: captures the variance from random document inclusion, producing error bars in scaling plots that reflect genuine uncertainty rather than measurement noise.
- **Token count computed with Llama-2 tokenizer**: provides a consistent, reproducible unit of datastore size that is independent of the model being evaluated.

## 4. Key Insights and Innovations

### Innovation 1: Datastore Size as a First-Class Scaling Dimension Alongside Model Size and Pretraining Data

The paper's most fundamental contribution is not any single method or architecture, but rather a **conceptual reframing** of what counts as a scaling axis for language models. Prior to this work, the scaling laws literature (Kaplan et al., 2020; Hoffmann et al., 2022; Muennighoff et al., 2023; Gadre et al., 2024) treated LM performance as a function of two variables: model parameters and pretraining tokens. The retrieval-based LM community, working in parallel, treated datastore size as a fixed implementation detail—typically Wikipedia at ~5 billion tokens—rather than as a variable to be systematically varied and optimized.

This paper's core intellectual move is to **elevate datastore size from a fixed hyperparameter to a third scaling axis** and to demonstrate that it behaves with the same regularity as the other two: performance improves monotonically with datastore size, the improvements do not saturate within the range studied (1.4 trillion tokens), and there exist compute-optimal tradeoffs between investing FLOPs in pretraining versus investing them in datastore construction. The authors make this framing explicit in their abstract and introduction, but the real weight of the contribution comes from the compute-optimal scaling curves in Figure 4, which plot Pareto frontiers over total training FLOPs that include datastore construction cost—directly analogous to how Hoffmann et al. (2022) plotted frontiers over pretraining FLOPs.

What makes this a **fundamental shift** rather than an incremental observation is that it changes the decision space for anyone building or deploying LMs. Before this work, the question was "how large a model should I train?" or perhaps "how much data should I train on?" After this work, the question becomes "given a fixed total compute budget, how should I allocate it among model parameters, pretraining tokens, and datastore tokens?" The answer—visible in Figure 4's consistently higher Pareto frontiers for retrieval-based LMs versus LM-only models—is that offloading knowledge to a datastore is often more FLOP-efficient than memorizing it in parameters. This is an inference-time analog of the Chinchilla insight that for a given compute budget, you should train smaller models on more data rather than larger models on less data. Except here, the "data" is not consumed during training but indexed once and reused across all queries.

The significance of this reframing extends beyond the paper's specific numbers. Once datastore size is recognized as a scaling dimension, it opens an entire research program: What is the datastore scaling equivalent of the Chinchilla optimal point? How does the optimal allocation between pretraining and datastore tokens change with model capability and task type? Do datastore scaling laws follow power-law or different functional forms? The paper provides initial empirical answers but the conceptual move is what enables the questions to be asked systematically.

### Innovation 2: The MassiveDS Pipeline as an Enabling Methodology — Efficient Datastore Scaling Studies Through Operation Reordering

The paper's second major contribution is **methodological infrastructure**: a pipeline that reduces the computational cost of datastore scaling experiments by more than an order of magnitude through a principled reordering of operations. The dominant assumption in prior work—implicit, because no one had tried to run a comprehensive scaling study—was that studying datastore scaling would require building and indexing separate datastores for every experimental variation, making trillion-token-scale studies prohibitive. This is why prior work either used a single fixed datastore size (nearly all retrieval-based LM papers) or conducted only limited scaling analysis with proprietary infrastructure (RETRO, Borgeaud et al., 2022, which did not release its pipeline).

The MassiveDS pipeline's key insight is that **element-wise operations commute with subsampling**, meaning you can index the full datastore once, retrieve a large candidate pool for each query, and then apply all experimental variations (subsampling ratio, random seed, filtering, data composition) to these pre-retrieved sets rather than to the raw trillion-token corpus. The paper provides formal proofs (Lemmas A.1-A.4 in Appendix A) that this reordering is equivalent to the naive approach with high probability, with tail-bound analysis (Table 4) showing the failure probability is exponentially small in the pre-retrieval buffer size K.

This contribution is **fundamental in an infrastructural sense**: it transforms datastore scaling from a capability accessible only to industrial labs with massive compute budgets (RETRO required proprietary infrastructure; SPHERE at 90B tokens was the largest open effort) into something reproducible on an academic budget. The paper demonstrates this by releasing not just the final datastore but the full pipeline code, enabling other researchers to run their own scaling studies with different retrievers, models, or data compositions.

What makes this more than an engineering optimization is that it is **provably correct**: the equivalence proofs establish that the pipeline is not an approximation or a heuristic but produces results identical (with controlled, exponentially small failure probability) to the expensive naive approach. This theoretical grounding matters because it means researchers can trust the scaling trends they observe—they are not artifacts of pipeline shortcuts. The methodology also generalizes: any future work studying how other variables (retriever architecture, chunking strategy, indexing method) affect datastore scaling can adopt the same reordering principle.

The significance is amplified by the paper's transparency about computational costs. By quantifying the FLOPs for datastore construction versus pretraining (the `$6 N_{\text{LM}} D_{\text{pretrain}}$` versus `$2 N_{\text{retriever}} D_{\text{datastore}}$` decomposition in Section 4.3) and explicitly discussing inference costs, the paper provides a template for future work to do cost-benefit analyses of retrieval versus pretraining in their own settings.

### Innovation 3: Task-Dependent Scaling — Knowledge-Intensive Tasks Benefit Disproportionately While Reasoning-Heavy Tasks Are Limited by Model Capability and Data Coverage

The paper's cleanest empirical insight is that **datastore scaling benefits are highly task-dependent in a way that reveals fundamental constraints on retrieval-based LMs**. The dominant assumption in prior work—encouraged by the fact that most retrieval evaluations were on open-domain QA—was that retrieval helps uniformly or at least broadly across tasks. The paper's systematic evaluation across four downstream tasks with multiple model families produces a more nuanced picture:

- On **knowledge-intensive factoid QA** (TriviaQA, Natural Questions), datastore scaling provides dramatic gains that continue without saturation. A Llama-2 7B with a large datastore outperforms both Llama-2 13B and Llama-3 8B without retrieval (Figure 3). This suggests that for tasks where the primary challenge is **locating a fact**, storing that fact in a datastore and retrieving it at inference time is more FLOP-efficient than memorizing it during pretraining.

- On **reasoning-heavy benchmarks** (MMLU, MedQA), the benefits are more limited and conditional on model quality. Pythia models show marginal gains on MMLU and MedQA even at 12B parameters, with performance staying near random (Figure 4, right columns). OLMo models, trained on more and better data, show consistent improvement. This pattern suggests that retrieval can supply knowledge, but **the model must already possess the reasoning capability to use that knowledge**—retrieval amplifies existing reasoning ability but cannot create it.

- On **language modeling**, retrieval helps consistently and substantially, with benefits that persist even after aggressive decontamination (Figure 7, top row). The fact that decontamination reduces but does not eliminate the benefit indicates that semantically similar documents (not just verbatim copies) drive the improvement, and that retrieval is doing genuine language modeling work rather than trivial copying.

This finding is **diagnostically important** because it clarifies when to invest in datastore scaling versus other approaches. If your task is knowledge-intensive and factual (open-domain QA, entity linking, fact verification), scaling the datastore is likely a high-return investment. If your task requires complex reasoning (multi-step math, logical deduction, synthesis), you should prioritize improving the base model's reasoning capabilities and ensuring the datastore contains the right kind of reference material (textbooks for MMLU, medical literature for MedQA)—and even then, the gains may be modest. This task-dependence also explains the apparently contradictory findings in prior work: SPHERE (Piktus et al., 2022) found a large datastore didn't always beat Wikipedia on KILT tasks, possibly because those tasks were reasoning-heavy or because the datastore lacked relevant in-domain data. The paper's framework makes this outcome predictable rather than surprising.

### Innovation 4: The Small-Model-Plus-Datastore Regime — A 119× Cost Asymmetry That Makes Knowledge Storage More Efficient Than Memorization

The paper's FLOPs analysis reveals a **quantitative asymmetry with profound economic implications**: indexing a token in the datastore costs approximately 119× fewer FLOPs than pretraining on that same token (for a 7B-parameter model with a 177M-parameter retriever). This asymmetry arises from two multiplicative factors: the retriever is ~40× smaller than the LM, and indexing requires only a forward pass (2× multiplier) while pretraining requires forward + backward (6× multiplier). Combined: `$3 \times (7000/177) \approx 119\times$`.

This number is not novel math—the FLOPs formula is standard—but the paper's contribution is to **make this asymmetry the centerpiece of an argument about where knowledge should be stored**. The dominant paradigm in LLM development has been to train ever-larger models that memorize ever more knowledge in their parameters, with retrieval used as an optional add-on for factuality or attribution. The paper's compute-optimal scaling curves (Figure 4) inverts this logic: for a fixed total training FLOPs budget, you achieve better downstream performance by training a smaller model and investing the savings in a large datastore than by training a larger model alone.

This finding is **incremental in its mechanism** (the asymmetry was always latent in the FLOPs) but **fundamental in its practical implications**. It suggests that the default architecture for knowledge-intensive applications should not be "train the largest model you can afford" but rather "train a model just large enough to reason with retrieved information, then invest in the datastore." The paper demonstrates this concretely: on TriviaQA and Natural Questions, even Pythia 1B with a datastore matches Pythia 12B without one (Figure 9 in Appendix C)—a ~12× parameter reduction achieved through datastore scaling rather than model scaling.

The finding is bounded in important ways that the paper acknowledges: it applies primarily to knowledge-intensive tasks where the model can extract answers from retrieved text; it requires the base model to have sufficient reasoning ability to use the retrieved information (Pythia fails on MMLU regardless of datastore size); and it depends on having a datastore that contains the relevant knowledge. But within those bounds, the implication is clear: knowledge is cheaper to store than to memorize, and future LM systems should be designed with this cost structure in mind.

### Innovation 5: Retrieval Robustness to Out-of-Domain Data — Broad Datastores Work Because Retrievers Self-Organize by Domain

A persistent concern in retrieval-based LM research is that broad, multi-domain datastores would confuse the retriever, causing it to retrieve irrelevant documents from wrong domains and degrading performance compared to carefully curated in-domain datastores. The paper's data composition analysis (Section 5.1, Table 3) provides **direct counterevidence to this concern**, showing that MassiveDS (a broad 8-domain datastore) either matches or outperforms single-domain datastores across every task evaluated.

The mechanism behind this robustness is visualized in Figure 5: when retrieving for Natural Questions (a Wikipedia-derived benchmark), the retriever disproportionately retrieves from Wikipedia and general web sources; when retrieving for MedQA (a medical exam benchmark), the retriever disproportionately retrieves from scientific papers (peS2o) and biomedical articles (PubMed). The retriever is not confused by the presence of out-of-domain data—it effectively ignores it, allocating retrieval probability mass to the relevant domains.

This finding is **significant because it resolves a tension in the literature**. SPHERE (Piktus et al., 2022) found that a 90B-token web-scale datastore did not always outperform a small Wikipedia datastore on KILT benchmarks, which could be interpreted as evidence that broad datastores are not worth the cost. The paper's results suggest that SPHERE's mixed findings may have been due to factors other than domain breadth—possibly data quality, retriever capability, or benchmark construction—and that with a sufficiently large and diverse datastore processed through proper deduplication and decontamination, broad coverage is strictly beneficial.

The practical implication is that **a single general-purpose datastore can serve multiple tasks simultaneously**, removing the need to curate task-specific datastores. This is what makes retrieval-based LMs viable as general-purpose systems rather than task-specific tools: you build one datastore, and the retriever's implicit domain routing handles specialization. The paper's demonstration that this works across knowledge-intensive QA (TriviaQA, NQ), multi-task reasoning (MMLU), and specialized domain QA (MedQA) with the same underlying datastore is a substantial step toward general-purpose retrieval-based LMs.

## 5. Experimental Analysis

### Evaluation Methodology

- **Dataset.** The paper evaluates on five downstream tasks: TriviaQA (TQA; Joshi et al., 2017) — 17,944 trivia questions with answers sourced from Wikipedia and the web, evaluated via exact match; Natural Questions (NQ; Kwiatkowski et al., 2019; Lee et al., 2019) — 3,610 search engine queries with human-annotated Wikipedia answers, evaluated via exact match; MMLU (Hendrycks et al., 2021) — 14,042 multiple-choice questions across 57 subjects measuring general-purpose reasoning, evaluated via accuracy; MedQA (Jin et al., 2020) — 1,273 medical multiple-choice questions from professional medical exams, evaluated via accuracy. For language modeling, evaluation data is sampled from RedPajama (general web domain; Computer, 2023) and S2ORC (scientific papers; Lo et al., 2020), split into 1,024-token chunks with 512-token stride, where the first half serves as retrieval query and prefix and the second half as the target for perplexity computation.

- **Base model(s).** The paper uses four model families: Llama-2 (7B, 13B; Touvron et al., 2023), Llama-3 (8B; Touvron et al., 2023), Pythia (1B, 2.8B, 6.9B, 12B; Biderman et al., 2023), and OLMo (1B, 7B; Groeneveld et al., 2024). Llamas are chosen as representative of "the capabilities of many contemporary LLMs" and for their strong few-shot performance. Pythia and OLMo are selected because they release intermediate checkpoints throughout training, enabling the compute-optimal scaling analysis that requires models trained on varying numbers of tokens without pretraining from scratch. OLMo serves as a higher-quality training data contrast to Pythia (trained on Dolma vs. the Pile).

- **Metrics.** For downstream tasks, the primary metric is either exact match accuracy (TQA, NQ) or multiple-choice accuracy (MMLU, MedQA). For language modeling, perplexity is computed as the exponentiated average negative log-likelihood of target tokens given the prefix and retrieved documents: `$\text{PPL} = \exp(-\frac{1}{T}\sum_{t=1}^{T}\log P(y_t | x, r_1, r_2, r_3, y_{<t}))$` where `$T=512$` target tokens, `$x$` is the 512-token prefix, and `$r_1, r_2, r_3$` are the three retrieved documents. Lower perplexity indicates the model assigns higher probability to the actual continuation.

- **Baselines.** The paper compares against two primary baselines: **(1) LM-only models** — the same language models evaluated without any retrieved documents, using only their parametric knowledge in a standard few-shot setting (5-shot for downstream tasks, pure language modeling for perplexity). This establishes the performance floor that retrieval must improve upon. **(2) Single-domain datastores** — individual domains from MassiveDS used in isolation (e.g., Wikipedia-only, PubMed-only, RedPajama-only) to test whether a broad multi-domain datastore outperforms task-specific curated datastores. The comparison covers DPR Wikipedia, RedPajama Books, ArXiv, StackExchange, GitHub, PubMed, MATH, peS2o, and the full RedPajama corpus.

- **Generation budget / compute accounting.** For the compute-optimal scaling analysis in Section 4.3, total training-time FLOPs are computed as the sum of pretraining FLOPs (`$6 N_{\text{LM}} D_{\text{pretrain}}$`) and datastore construction FLOPs (`$2 N_{\text{retriever}} D_{\text{datastore}}$`), where `$N_{\text{LM}}$` is the number of language model parameters, `$D_{\text{pretrain}}$` is the number of pretraining tokens, `$N_{\text{retriever}} = 177 \times 10^6$` parameters, and `$D_{\text{datastore}} = p \times 1.441 \times 10^{12}$` tokens for subsampling ratio `$p$`. The factor of 6 for pretraining accounts for one forward pass (`$2ND$`) plus one backward pass (`$4ND$`); the factor of 2 for datastore construction accounts for one forward pass only. This combined metric enables direct comparison of retrieval-based LMs and LM-only models under a unified compute budget. For the main scaling results in Section 4.2, the x-axis is datastore size in tokens (plotted as `$p \times$` total raw tokens on a log scale) rather than FLOPs.

- **Cross-validation / statistical protocol.** Each subsampling ratio `$p$` is evaluated with three different random seeds (100, 101, 102) to capture the variance from which specific documents are included in the datastore. The scaling plots report the mean across seeds with confidence intervals (visible as shaded regions or error bars in Figures 3, 4, and Appendix C Figures 9–12). For the language modeling decontamination analysis (Figure 7), the paper compares three decontamination levels: none, standard (13-gram Jaccard + 32-gram longest overlap for LM; 13-gram Jaccard for downstream), and aggressive (8-gram longest overlap). For the data composition analysis (Table 3), single-domain datastores are compared against the full MassiveDS at equivalent retrieval settings. The reranking analysis (Figure 6) compares no reranking, cross-encoder reranking, and lexical oracle reranking on the same retrieval pipeline.

### Main Quantitative Results

#### Datastore Scaling on Language Modeling (Finding 1)

**Headline: Datastore scaling reduces perplexity monotonically without saturation, enabling a small retrieval-based model to outperform a larger LM-only model.**

Figure 3(a) and 3(b) report perplexity on RedPajama (general web) and S2ORC (scientific papers) as a function of datastore size for Llama-2 7B, Llama-2 13B, and Llama-3 8B. Several quantitative patterns emerge:

At the smallest datastore size (~14B tokens, `$p=0.01$`), Llama-2 7B achieves approximately 4.06 perplexity on RedPajama compared to the LM-only baseline of 4.09 (dashed line). At the full datastore size (~1.4T tokens), perplexity drops to approximately 3.50 — a reduction of approximately 0.59 perplexity points, or roughly 14.5% relative improvement over the LM-only baseline. Crucially, the curve shows **no visible signs of saturation** — the downward trend continues even at the largest datastore size, suggesting further scaling would yield additional gains.

The cross-model comparison reveals a striking finding: **Llama-2 7B with the full MassiveDS datastore (~3.50 PPL) outperforms the LM-only performance of its larger counterpart Llama-2 13B (~3.55 PPL, shown as the dashed line)**. This demonstrates that retrieval from a sufficiently large datastore can compensate for a ~1.86× reduction in model parameters on language modeling quality.

On S2ORC (Figure 3b), the pattern is consistent but with higher absolute perplexity values (scientific text is less predictable than general web text). Llama-2 7B drops from approximately 7.18 (LM-only) to approximately 6.57 (full datastore), again crossing below the Llama-2 13B LM-only baseline.

A surprising finding visible in Figure 3(a) is that **Llama-3 8B underperforms Llama-2 7B on RedPajama perplexity both with and without retrieval**. The paper discusses this in Appendix D, attributing it possibly to Llama-3's post-training process using instruction-tuning data that shifts performance away from simple perplexity evaluation, or to RedPajama being down-weighted in Llama-3's larger training corpus leading to less memorization of this specific domain. This is an important caution: scaling trends can be sensitive to model-specific training details.

The Pythia and OLMo scaling curves in Appendix C (Figure 9 onwards) show qualitatively similar patterns — monotonic improvement with datastore size — though with higher absolute perplexity values due to their smaller model sizes and weaker pretraining.

#### Datastore Scaling on Downstream Tasks (Finding 2)

**Headline: Datastore scaling substantially improves knowledge-intensive QA tasks but shows limited benefits on reasoning-heavy benchmarks, where gains depend on base model capability and datastore coverage.**

Figures 3(c)–(f) and Appendix C Figures 9–12 present scaling results on four downstream tasks. The results reveal a clear **task-dependence** that constitutes one of the paper's most important empirical findings.

**Knowledge-intensive factoid QA (TriviaQA and Natural Questions):** These tasks show the strongest scaling benefits. On TriviaQA (Figure 3c), Llama-2 7B starts at approximately 64.1% (LM-only) and reaches approximately 77.0% at the full datastore — a gain of roughly 12.9 percentage points. At fewer than 100B tokens, Llama-2 7B with retrieval already outperforms both Llama-2 13B (approximately 72.6%) and Llama-3 8B (approximately 72.9%) without retrieval. This is a concrete demonstration that **datastore scaling can substitute for model scaling** on knowledge-intensive tasks: a 7B model with retrieval matches or exceeds a ~13B model without it, at a fraction of the training cost when the FLOPs savings from not pretraining the larger model are accounted for.

On Natural Questions (Figure 3d), the pattern is similar but with overall lower accuracy (NQ is generally a harder benchmark). Llama-2 7B improves from approximately 26.6% (LM-only) to approximately 34.6% at the full datastore. Interestingly, the scaling curve appears **less steep than TriviaQA** and may be approaching saturation — the improvement from ~100B tokens to ~1.4T tokens is relatively modest.

The Pythia results for these tasks (Appendix C, Figures 9 and 10) are particularly striking: **Pythia-1B with retrieval matches or exceeds Pythia-12B without retrieval** on both TriviaQA and Natural Questions. For example, Pythia-1B reaches approximately 45% on TriviaQA at the full datastore, roughly matching the LM-only performance of Pythia-12B (~44%). On Natural Questions, Pythia-1B with the full datastore (~28%) outperforms Pythia-12B LM-only (~25%). This is a ~12× parameter reduction compensated entirely by retrieval.

**Reasoning-heavy tasks (MMLU and MedQA):** The scaling benefits here are more nuanced and conditional. On MMLU (Figure 3e), Llama-2 7B improves from approximately 45.8% (LM-only) to approximately 49.3% (full datastore) — a modest 3.5-percentage-point gain. Llama-2 13B and Llama-3 8B show similar absolute improvements but start from higher baselines, reaching approximately 50.4% and 52.1% respectively at the full datastore. Critically, **on MMLU, a smaller model with retrieval does not outperform a larger model without retrieval** — the retrieval gains are additive but insufficient to close the gap between model sizes. Llama-2 7B with full retrieval (~49.3%) still lags behind Llama-2 13B LM-only (~50.4%? — the paper does not report the exact LM-only number for Llama-2 13B on MMLU separately from the scaling plot, but the dashed lines in Figure 3e indicate this).

On MedQA (Figure 3f), the pattern is even more conditional. Llama-2 7B shows a modest improvement from approximately 36.6% (LM-only) to approximately 39.4% (full datastore), but **Llama-2 13B and Llama-3 8B show negligible or inconsistent gains**. In fact, Llama-3 8B with retrieval at some datastore sizes performs slightly worse than its LM-only baseline, suggesting that out-of-domain retrieval (MedQA requires specialized medical knowledge that MassiveDS may not adequately cover) can sometimes confuse stronger models. The paper notes that only the "weaker Llama-2 7B benefits more from datastore scaling" on MedQA.

The Pythia results for these reasoning-heavy tasks (Appendix C, Figures 11 and 12) are starkly negative: **Pythia models show essentially no benefit from retrieval on MMLU and MedQA at any scale**. On MMLU, all Pythia models (1B through 12B) hover around 25–28% accuracy (near random for a 4-choice task) regardless of datastore size. On MedQA, performance stays similarly flat across all datastore scales. The OLMo results (Figure 4, right columns) provide a crucial contrast: **OLMo models do benefit from retrieval on MMLU and MedQA**, with OLMo-7B improving from approximately 52% to 55% on MMLU at the full datastore. The paper attributes this divergence to OLMo being "trained on more and better data" (Dolma vs. the Pile), which may help the model better utilize retrieved context for reasoning.

**Language modeling across domains:** Table 3 shows that on RedPajama perplexity, the full MassiveDS (3.50 PPL) outperforms every single-domain datastore, including RedPajama-only retrieval (3.87 PPL). This is notable because RedPajama is exactly the evaluation domain — yet retrieving from the broader MassiveDS (which includes scientific papers, code, books, etc.) helps more than retrieving only from in-domain text. On S2ORC scientific text perplexity, MassiveDS (6.57 PPL) again outperforms all single-domain datastores including peS2o (6.71 PPL), the in-domain scientific paper corpus. This demonstrates that **cross-domain retrieval provides useful signal even for domain-specific language modeling**, possibly because related concepts appear across domains with different surface realizations.

#### Compute-Optimal Scaling: Retrieval-Based LMs vs. LM-Only (Findings 3–5)

**Headline: Retrieval-based LMs achieve superior compute-optimal scaling, with Pareto frontiers that lie consistently above LM-only frontiers for the same training FLOPs budget.**

Figure 4 presents the paper's central scaling analysis: Pareto frontiers over total training FLOPs for retrieval-based LMs (red) and LM-only models (blue) on four downstream tasks. Each crossmark represents a model configuration (model size × pretraining tokens), with darker colors indicating larger model sizes for Pythia (green) and OLMo (pink). For retrieval-based LMs, each crossmark is connected to a datastore scaling curve of lined dots representing performance at different datastore sizes (from `$p=0.01$` to 1.0).

**TriviaQA and Natural Questions (left columns of Figure 4):** The Pareto frontiers show a clear and consistent advantage for retrieval-based LMs. At a training budget of approximately `$10^{21}$` FLOPs, the best retrieval-based configuration achieves roughly 38% on TriviaQA (red Pareto point, likely corresponding to Pythia-1B or OLMo-1B with a large datastore), while the best LM-only configuration achieves roughly 28% (blue Pareto point) — a ~10-percentage-point gap. At higher budgets (~`$10^{22}$` FLOPs), the gap narrows but retrieval-based LMs maintain superiority, achieving roughly 52% vs. 42% for LM-only on TriviaQA.

The OLMo and Pythia families show **surprisingly similar compute-optimal trajectories** on these factoid QA tasks despite their different pretraining data quality. Figure 4 shows that Pythia-1B (trained on up to 300B tokens from the Pile) and OLMo-1B (trained on up to 3T tokens from Dolma) fall on similar scaling curves when augmented with retrieval. The paper interprets this as evidence that "the ability to extract factual knowledge for simple factual question answering is obtained early in training" (Finding 4). When the LM only needs to locate and extract an answer from retrieved text — without complex reasoning — even relatively weak pretraining suffices.

**MMLU and MedQA (right columns of Figure 4):** The picture changes dramatically. On MMLU, the Pythia family shows **essentially no improvement from retrieval**: the red points sit almost directly on top of the blue LM-only points, with accuracy near random (~25%) regardless of FLOPs budget. The OLMo family, in contrast, shows a clear retrieval benefit, with the red Pareto frontier rising above the blue one, reaching roughly 55% accuracy at ~`$10^{22}$` FLOPs compared to roughly 47% for LM-only.

On MedQA, the pattern is similar but even more extreme: Pythia models show negligible improvement, while OLMo models show a more modest but consistent benefit from retrieval. At the highest compute budgets (~`$10^{22}$` FLOPs), OLMo retrieval-based LMs reach approximately 40% accuracy vs. approximately 35% for LM-only.

The paper draws two conclusions from this divergence (Finding 5): (1) retrieval benefits for reasoning-heavy tasks are **conditional on the base model having sufficient reasoning capability** to use the retrieved information — Pythia models trained on the Pile apparently lack this, while OLMo models trained on Dolma possess it; (2) **datastore coverage may be insufficient** for MMLU and MedQA — these tasks require "specific data, such as relevant textbooks for MMLU and biomedical literature for MedQA, which are currently not included in MassiveDS." The improvement in OLMo with retrieval suggests that even partial coverage helps when the model can reason with what it retrieves.

**Data composition and domain routing (Finding 6).** Table 3 quantifies the performance of Llama-2 7B with retrieval from MassiveDS versus single-domain datastores. On TriviaQA, MassiveDS (77.0%) substantially outperforms the best single-domain datastore (RedPajama at 70.5%, or DPR Wikipedia at 64.5%). On NQ, MassiveDS (34.6%) matches the best single-domain result (RedPajama, also 34.6%) — unsurprising given that NQ answers are Wikipedia-based, and the Wikipedia datastore achieves only 26.9%. On MedQA, MassiveDS (39.4%) matches PubMed (37.8%) within error bars. On MMLU, MassiveDS (49.3%) modestly outperforms the best single-domain result (StackExchange at 48.3%).

Figure 5 provides the mechanistic explanation: when retrieving for NQ, 85.2% of the top-1 retrieved documents come from Common Crawl and an additional 7.4% from Wikipedia — the retriever routes to web and encyclopedia sources. For MedQA, 51.5% come from peS2o (scientific papers), 6.6% from PubMed, and 39.2% from Common Crawl — the retriever routes to scientific and medical sources. Crucially, the retriever does not retrieve uniformly from all domains; it **implicitly performs domain routing** based on query content, ignoring out-of-domain data. The paper notes this aligns with findings on kNN-LM (Khandelwal et al., 2020) in Shao et al. (2023).

#### Reranking Analysis (Section 5.2)

**Headline: Better retrieval through reranking improves scaling trends, but a substantial gap remains between practical rerankers and oracle performance.**

Figure 6 shows TriviaQA and NQ scaling curves for Llama-2 7B under three retrieval quality conditions: no reranking (Contriever only), cross-encoder reranking (MINI-LM-L12 V2), and lexical oracle reranking (using ground-truth answers to select the best documents).

On TriviaQA at the full datastore (~1.4T tokens), the cross-encoder reranker achieves approximately 79–80% accuracy, up from approximately 77% without reranking. The oracle reranker reaches approximately 84–85%. The gap between the cross-encoder (~80%) and the oracle (~85%) is roughly 5 percentage points at the full datastore and **grows with datastore size** — at ~100B tokens, the gap is approximately 3–4 points, while at ~1.4T tokens, it widens to approximately 5 points. This suggests that as the datastore grows, the challenge of selecting the right documents from a larger pool becomes harder, and improved retrieval would yield proportionally larger benefits at larger scales.

On Natural Questions, the pattern is similar: cross-encoder reranking improves over no reranking by approximately 1–2 percentage points across most datastore sizes, and the oracle reranker provides an additional 2–3 percentage points of improvement. The overall lower performance on NQ (maximum ~37% with cross-encoder vs. ~80% on TriviaQA) reflects the greater difficulty of NQ questions and potentially less complete coverage of NQ answers in MassiveDS.

The paper interprets the persistent oracle gap as evidence that **"improving either retrieval or reranking can further boost the scaling performance of retrieval datastores"** — both the scaling curves themselves and the retrieval quality are complementary levers for improving end-task performance.

### Ablation Studies and Robustness Checks

**Data decontamination level (Figure 7):** Comparing no decontamination, standard decontamination (13-gram Jaccard + 32-gram longest overlap for LM), and aggressive decontamination (8-gram longest overlap) reveals a nuanced picture. On language modeling (RedPajama PPL), no decontamination produces substantially lower perplexity across all datastore sizes — at the full datastore, approximately 3.2 PPL vs. 3.5 PPL with standard decontamination — indicating that "the benefits in language modeling primarily arise from lexical overlap" with test data. However, even under aggressive decontamination, retrieval still improves over the LM-only baseline (approximately 3.6 PPL vs. 4.09 PPL), demonstrating that semantically similar documents with minimal lexical overlap contribute genuine benefit. On NQ, decontamination level has minimal effect, with the aggressive decontamination curve mostly overlapping the standard decontamination curve. Notably, decontamination slightly decreases performance at small datastore sizes (~20B tokens) but improves final performance at the largest scale — a non-monotonic effect the paper observes but does not fully explain.

**Data deduplication (Figure 13, Appendix E.2):** On language modeling perplexity, global MinHash deduplication shows negligible impact — the curves with and without deduplication largely overlap. On Natural Questions, deduplication is crucial for maintaining scaling behavior: without deduplication, performance **saturates earlier**, plateauing around 33.5% at ~400B tokens, while with deduplication, performance continues to improve to approximately 34.5% at the full datastore. The paper attributes this to subsampling with higher `$p$` increasing the chance of including duplicate documents that waste retrieval slots, and deduplication mitigating this effect.

**Data quality filtering (Dolma filters, Figure 13):** Applying whitespace, language, and alphanumeric filters adapted from Dolma (Soldaini et al., 2024) has "a relatively limited effect" on both language modeling perplexity and NQ accuracy. The curves with and without quality filtering are nearly indistinguishable. The paper hypothesizes that the source datasets in MassiveDS (particularly RedPajama) have already undergone similar filtering, making additional filters redundant. This is a practically useful negative result: it suggests that for datastores built from already-curated sources, expensive additional quality filtering may not be worth the cost.

**Short chunk removal (Figures 14 and 15, Appendix E.2):** Before removing documents with fewer than 13 words, the retriever frequently retrieves short, high-lexical-overlap chunks that match the query string but contain no answer-relevant information. Figure 14 shows examples: for an NQ question about "how many episodes of teen wolf season 4," the top-1 retrieved document without short-chunk removal is simply the phrase "there are 12 episodes" with no context about the show. Removing short chunks improves NQ performance (Figure 15), with the benefit being most pronounced at larger datastore sizes — at ~1.4T tokens, the removal improves accuracy by roughly 1.5–2 percentage points. This demonstrates that simple data quality heuristics can have meaningful downstream impact on retrieval quality.

**Retriever choice (Table 6, Appendix E.1):** On a 10% subsample of MassiveDS evaluated with Llama-2 7B, three retrievers perform similarly: Contriever-MSMARCO achieves 4.2210 PPL on RedPajama, 0.3321 exact match on NQ, and 0.4922 accuracy on MMLU; DRAGON-RoBERTa achieves 4.2373 PPL, 0.3399 EM, and 0.4875 accuracy; GTR-T5-Base achieves 4.2146 PPL, 0.3080 EM, and 0.4934 accuracy. Contriever is selected for the full-scale study because its implementation runs faster than the alternatives — a practical choice that does not sacrifice performance. This ablation supports the claim that the observed scaling trends are not specific to a particular retriever architecture.

**Prompt format (Appendix B.3):** Two formats were tested: (1) few-shot examples first, then retrieved documents, then the question; (2) retrieved documents first, then few-shot examples, then the question. Format (2) performs better because "the LM can learn the few-shot pattern better when the few-shot examples are closer to the question." This is consistent with the "lost in the middle" phenomenon (Liu et al., 2023) — the model attends more strongly to information near where it generates, so placing few-shot examples adjacent to the question helps the model follow the task format.

### Critical Assessment

#### Does datastore scaling monotonically improve language modeling without saturation?

**What the experiments show:** Figure 3(a) and 3(b) demonstrate a clear negative slope for perplexity vs. datastore size across the range ~14B to ~1.4T tokens, with no visible flattening. The curves appear to be well-modeled by a power law on a log-log plot — the paper does not fit functional forms, but the linear-ish trend on log axes is consistent with unsaturated power-law scaling.

**What is not shown:** The paper only studies up to 1.4T tokens — a single order of magnitude above the largest prior open datastores (~90B for SPHERE) but still modest compared to the scale of web data that could be indexed (Common Crawl alone contains hundreds of trillions of tokens). The claim of "no saturation" is therefore bounded by the range studied. It is entirely possible that saturation would appear at 10T or 100T tokens. The paper also does not fit quantitative scaling laws (e.g., `$\text{PPL}(D) = a \cdot D^{-\alpha} + b$`) that would enable extrapolation. Additionally, the decontamination analysis (Figure 7) shows that a non-trivial fraction of the language modeling gain comes from lexical overlap with test data — the "genuine" semantic benefit after aggressive decontamination is smaller, though still present. The magnitude of the "real" retrieval benefit versus shallow lexical matching benefit is not cleanly disentangled.

#### Does a smaller model augmented with a large datastore outperform a larger LM-only model on knowledge-intensive tasks?

**What the experiments show:** This claim is well-supported on TriviaQA and Natural Questions. Llama-2 7B with MassiveDS outperforms Llama-2 13B without retrieval on both tasks (Figure 3c,d). Pythia-1B with retrieval matches Pythia-12B without retrieval on TriviaQA and outperforms it on NQ (Appendix C, Figures 9-10). The compute-optimal scaling curves (Figure 4) show that at most FLOPs budgets, the best retrieval-based configuration outperforms the best LM-only configuration.

**What is not shown:** The comparison is always between a retrieval-based smaller model and an LM-only larger model — the larger model never gets its own datastore. A fairer comparison would ask: at equivalent total FLOPs, is it better to give a small model a large datastore or a medium model a medium datastore? The three-way allocation problem (model size, pretraining tokens, datastore size) is not fully solved. Additionally, the specific claim that "a small model outperforms a larger model" holds only for knowledge-intensive factoid QA — on MMLU, it does not hold for Llama models (though it does for OLMo vs. Pythia comparisons). The generality of the substitution claim is therefore task-limited.

#### Do retrieval-based LMs achieve superior compute-optimal scaling compared to LM-only models?

**What the experiments show:** Figure 4's Pareto frontiers consistently show retrieval-based LMs achieving higher accuracy at the same FLOPs budget. The gap is largest on TriviaQA and NQ, smaller on MMLU (for OLMo), and marginal on MedQA. The FLOPs accounting correctly captures pretraining and datastore construction costs.

**What is not shown:** This analysis has several important caveats that the paper acknowledges but that limit the strength of the conclusion. First, the intermediate checkpoints from Pythia and OLMo are imperfect proxies — models trained from scratch with compute-optimal schedules for each token budget might perform better, potentially shifting the LM-only frontier upward relative to the retrieval frontier. Second, the datastore is constructed once at the full scale and subsampled — the actual cost of building a 100B-token datastore (rather than building a 1.4T-token datastore and pretending only 100B tokens are used) might be lower, making the comparison slightly unfair to retrieval at small datastore sizes (since the FLOPs plotted include the cost of embedding the full 1.4T tokens). Third, **inference costs are not included in the FLOPs accounting**, and retrieval-based models have higher inference costs due to longer context lengths and the retrieval search itself. For deployment scenarios where inference dominates training (high `$D_{\text{inference}}/D_{\text{pretrain}}$` ratios), the compute-optimal picture could shift in favor of LM-only models. The paper explicitly defers inference-cost-included scaling to future work.

#### Do improvements on reasoning-heavy tasks depend on base model capability and data coverage?

**What the experiments show:** The Pythia-vs-OLMo contrast on MMLU and MedQA (Figure 4, right columns) supports this claim: Pythia models show negligible retrieval benefit, OLMo models show consistent benefit. The paper's interpretation — that Pythia lacks the reasoning capability to use retrieved knowledge — is plausible and consistent with the known quality differences between the Pile and Dolma.

**What is not shown:** The paper does not establish **causality** — it observes a correlation between training data quality (Pile vs. Dolma) and retrieval benefit, but does not run controlled experiments varying only data quality while holding model architecture constant. It is possible that other differences between Pythia and OLMo (architecture details, training hyperparameters, tokenizer) contribute to the divergence. The paper also acknowledges that datastore coverage is a confound: MMLU and MedQA may simply lack relevant content in MassiveDS regardless of model capability. An experiment adding MMLU-relevant textbooks or MedQA-relevant medical literature to MassiveDS and observing whether Pythia then benefits would strengthen the causal claim, but this is not done.

#### Is the MassiveDS pipeline truly equivalent to the naive pipeline?

**What the experiments show:** The paper provides formal proofs (Lemmas A.1-A.4) that the reordering is equivalent with high probability, with tail-bound analysis (Table 4) showing failure probability <0.3% for the smallest subsampling ratio. No empirical comparison between the naive and MassiveDS pipelines is presented (doing so would defeat the purpose — it would require running the naive pipeline, which is exactly what the MassiveDS pipeline avoids).

**What is not shown:** The proofs assume element-wise retrieval (each document's score is independent of others), which holds for the Contriever flat index used. If a different retrieval method were used — e.g., approximate nearest neighbor search, learned sparse retrieval, or cross-attention scoring — the equivalence might break. The paper's proofs also assume that filtering operations are deterministic given document content, which is true for deduplication and decontamination thresholds but might not hold for learned quality filters that depend on corpus-level statistics (e.g., a filter that removes documents with below-average length relative to the corpus would behave differently on a subsample). The paper does not discuss these edge cases.

#### Cross-task and cross-model robustness

**Strengths:** The paper evaluates 4 model families (Llama-2, Llama-3, Pythia, OLMo) spanning 7 sizes (1B through 13B) on 4 downstream tasks plus 2 language modeling domains. This is substantially broader than prior work. The consistent qualitative patterns across model families on knowledge-intensive tasks strengthen the claim that datastore scaling is a general phenomenon.

**Weaknesses:** All downstream tasks are multiple-choice or short-answer QA. The paper does not evaluate on open-ended generation tasks (summarization, translation, dialogue), code generation, or mathematical reasoning — all areas where retrieval-based LMs could theoretically benefit but might show different scaling behavior. The test sets are also relatively small for some tasks (MedQA: 1,273 questions; Natural Questions: 3,610 questions), which limits the statistical precision of the scaling curves — the confidence intervals in the plots are visible but not huge, suggesting that the main trends are robust, but precise quantification of slope and saturation points would benefit from larger test sets.

**Missing baselines:** The paper compares retrieval-based LMs against LM-only models but not against other test-time compute methods — e.g., chain-of-thought prompting, self-consistency, or majority voting over multiple samples. While this is not a flaw (the paper's scope is retrieval scaling, not test-time compute in general), it means the claim that retrieval is the most FLOP-efficient way to improve performance at inference time is not directly tested against alternatives.

#### Compute-optimal scaling limitations

The intermediate checkpoint approach means that the LM-only Pareto frontier may be artificially depressed. If the Pythia-6.9B checkpoint at 1/3 of training has worse performance than a model trained from scratch with a scheduler optimized for exactly that token budget, the LM-only frontier would shift upward, potentially reducing the gap with retrieval-based LMs. The paper acknowledges this but does not quantify the potential bias. Given that the gap on knowledge-intensive tasks is large (e.g., ~10 percentage points on TriviaQA at some FLOPs budgets), it is unlikely to be entirely explained by the checkpoint approximation, but the precise magnitude of the advantage may be overstated.

The FLOPs accounting also assumes that the datastore is built once and used for all evaluations — which is fair for the compute-optimal analysis but does not account for the fact that in practice, different tasks might require different datastores (or at least different subsets of a large datastore). The cost of maintaining and serving a 1.4T-token datastore is not amortized over queries in the FLOPs calculation.

Overall, the experiments genuinely support the paper's central qualitative claims — datastore scaling improves performance, the improvements are task-dependent, and retrieval can substitute for model size on knowledge-intensive tasks — but the quantitative claims about compute-optimal tradeoffs should be treated as conditional on the specific models, tasks, and accounting conventions used. The paper is appropriately transparent about these limitations in Section 6.

## 6. Limitations and Trade-offs

### The Difficulty Estimation Cost — Required to Enable Compute-Optimal Allocation — Is Excluded from All Headline Efficiency Numbers

**The assumption or constraint.** The entire compute-optimal framework rests on the ability to estimate the difficulty of each prompt *before* allocating the inference budget. The paper's method for estimating difficulty — generating 2,048 samples per question and computing either ground-truth pass@1 (oracle difficulty) or the average PRM final-answer score (predicted difficulty) — is extraordinarily expensive. The authors acknowledge this explicitly in Section 3.2:

> "estimating difficulty in this way still incurs additional computation cost during inference... our experiments do not account for this cost largely for simplicity"

At 2,048 samples per question, the difficulty estimation step alone consumes more compute than the largest test-time budgets studied (256–512 generations). This means that in a real deployment, the total cost would be **difficulty estimation + strategy execution**, and the difficulty estimation cost could easily dominate the strategy execution cost, particularly for easy questions that would otherwise require only 4–8 generations.

**The consequence.** The paper's headline claim of 4× efficiency gains over best-of-N (e.g., "16 generations matching 64" for search in Figure 4, or "64 generations matching 256" for revisions in Figure 8) is computed *after* difficulty is known, without amortizing the cost of learning it. In a realistic deployment where difficulty must be estimated for each new prompt:
- If the difficulty estimation cost is included, the actual efficiency gain over best-of-N would be substantially smaller — potentially even negative for low-budget regimes where the estimation cost exceeds the strategy execution cost.
- The difficulty estimation process requires generating 2,048 samples per question, which is **serial and high-latency** if done online (2,048 sequential or batched generations), or **high-throughput but offline** if pre-computed — neither fits a real-time interactive deployment scenario.
- For the predicted-difficulty variant, the PRM must score all 2,048 samples, adding the PRM's forward-pass cost on top of the base model's generation cost.

**What evidence exists in the paper.** The paper shows that predicted difficulty bins perform nearly as well as oracle bins (Figures 4 and 8 show "the two curves largely overlap"), demonstrating that ground-truth answers are not strictly required — the PRM's own score distribution can serve as a proxy. However, the paper does **not** report a scaling curve that amortizes the difficulty estimation cost into the total compute budget, nor does it report an ablation where a smaller number of initial samples (e.g., 16 or 64) is used for difficulty estimation and the remaining budget is allocated according to the estimate. There is no experiment measuring how few samples are sufficient to estimate difficulty reliably enough to preserve the 4× gains. The 2,048-sample count appears to be chosen for statistical reliability of the per-question pass@1 estimate, not optimized for deployment efficiency.

**Mitigation status.** The paper acknowledges this as "a key avenue for future work" (Section 3.2) and suggests that future models could be trained to predict difficulty directly from the question text, or that adaptive (online) difficulty estimation could subsume the cost into the problem-solving process. However, no such model is developed or evaluated. The difficulty estimation cost remains an **unaddressed practical barrier** that separates the paper's analytical results from deployable efficiency gains. A practitioner reading this paper should understand the 4× figure as an **analytic upper bound**, not a realized deployment gain.

---

### Hard Problems Remain Unsolved — Test-Time Compute Provides Zero Benefit When the Base Model's Pass@1 Is Near Zero

**The assumption or constraint.** The paper's entire framework assumes that the base model can generate correct solutions at some non-trivial rate — that is, the proposal distribution contains correct answers that search can find or revisions can refine. In Section 1, the authors frame this as studying problems where "the model already possesses the necessary knowledge" but the challenge is drawing complex inferences. However, difficulty bin 5 (the hardest quintile) reveals what happens when this assumption breaks: pass@1 rates are near zero (roughly 1–3% — see Appendix C, Figure 11–12, or Figure 3 right panel), and no amount of test-time compute helps.

**The consequence.** On difficulty bin 5 across all methods:

- **Search against the PRM** (Figure 3, right): Both best-of-N weighted and beam search achieve 1–3% accuracy regardless of budget (4 through 256 generations). The PRM cannot guide search toward correct solutions because there are essentially no correct solutions in the proposal distribution to find.

- **Iterative revisions** (Figure 7, right): All sequential-to-parallel ratios produce roughly 2–3% accuracy regardless of budget (up to 256 generations). The revision model cannot refine an incorrect answer into a correct one when the base model never produces even a rough approximation of the correct answer.

- **FLOPs-matched comparison** (Figure 9): The bin 5 scaling line (blue, bottommost) is essentially flat near 0–5% and falls below all three `$14\times$` larger model stars. The paper's own takeaway box in Section 7 states that on hard problems, "pretraining is almost always more effective."

This means that **test-time compute can amplify existing capability but cannot create it from nothing.** For problems genuinely outside the base model's capability range — novel reasoning patterns, out-of-distribution problem structures, or knowledge gaps that prevent even approximate solutions — neither search nor revisions provide any path to success. The compute-optimal policy correctly identifies these problems (routing them away from expensive search) but offers no solution for them.

**What evidence exists in the paper.** The difficulty-bin breakdowns in Figures 3 (right), 7 (right), and 9 show this limitation clearly and consistently across search methods, revision strategies, and FLOPs-matched comparisons. The paper is transparent about this, but the limitation is fundamental: it establishes a **hard ceiling** on what test-time compute can achieve that is determined entirely by the base model's pretraining quality. The paper does not measure what fraction of real-world problem distributions fall into bin 5 — this would depend on the base model and task — but the MATH benchmark with PaLM 2-S* suggests roughly 20% of competition math problems are in this "impossible to solve with test-time compute" regime.

**Mitigation status.** The paper does not propose or explore any method for handling bin 5 problems. The only viable path identified is scaling pretraining (larger model or more data), which is explicitly presented as the preferred approach for hard problems in the FLOPs-matched comparison. This is not a flaw — the paper is honest about the boundary — but it means that test-time compute scaling is not a substitute for pretraining in general, only for a specific difficulty range. A practitioner must determine whether their problem distribution contains a substantial fraction of bin 5-equivalent problems before deciding whether to invest in test-time compute infrastructure versus larger model training.

---

### The Revision Model Has a 38% Correct-to-Incorrect Reversion Rate — Sequential Refinement Is Inherently Unstable

**The assumption or constraint.** The revision model is fine-tuned **only on sequences where all in-context answers are incorrect** — the training data construction (Section 6.1) pairs 0–4 incorrect answers with a correct answer as the target, with the last incorrect answer selected to be the one with smallest edit distance to the correct answer. This means the model is never trained on sequences where the in-context answer is already correct. At inference time, the revision chain may produce a correct answer at step `$t$`, and the model — having never seen a correct answer in context during training — does not know what to do with it.

**The consequence.** The paper reports that approximately **38% of correct answers produced during a revision chain get revised back to incorrect answers** in the subsequent step (Section 6.1). This creates a **fundamental instability** in sequential revision: each revision step is a double-edged sword that can improve incorrect answers but also corrupt correct ones. The result is that performance does not monotonically increase with chain length — the pass@1 trajectory in Figure 6 (left) shows that per-step accuracy rises from ~18.2% initially to ~24–25% by step 15–20 but then plateaus and fluctuates around 23–25% with further steps. Longer chains do not yield monotonically better answers.

The paper's mitigation — majority voting or verifier-based selection across the entire chain of revisions (rather than always taking the last revision) — is an imperfect patch:

- It requires generating the full chain and then selecting the best answer post-hoc, which means the compute budget for subsequent revisions is partially wasted when the model has already found a correct answer at an early step.
- The selection mechanism itself can fail — the verifier or majority vote may not correctly identify the correct answer among the chain, especially for hard problems where the verifier signal is unreliable.
- It introduces an additional hyperparameter (the chain length vs. number of parallel chains) that the compute-optimal policy must optimize over, adding to the already expensive difficulty estimation burden.

**What evidence exists in the paper.** The 38% reversion rate is reported in Section 6.1 (though the exact figure may be in an appendix — the main text states the problem and the solution, but the specific number appears in the paper's description of the issue). The pass@1 trajectory in Figure 6 (left) shows the non-monotonic plateau, demonstrating that longer chains do not solve the problem. The mitigation (selection across chain) is tested in Figure 6 (right), showing that sequential with verifier selection still outperforms parallel, but the gap is modest (~2.5 percentage points at 64 generations) compared to what monotonic improvement would yield.

**Mitigation status.** The paper mitigates the reversion problem with chain-wide answer selection but does not solve it at its root. A principled solution — training the revision model to recognize when the current answer is correct and produce a "no revision needed" token — is not explored. The paper acknowledges this as an architectural limitation but treats the current mitigation as sufficient for the scaling study. For practitioners, this means that **revision chains must be significantly over-generated** (producing more revisions than the point where the best answer likely appears) and then filtered, which wastes compute and adds latency. The net effect is that the 4× efficiency gains claimed for revisions over parallel best-of-N likely overstate what would be achievable if the revision model could reliably preserve correct answers.

---

### All Results Are on a Single Benchmark (MATH) with a Single Model Family (PaLM 2-S*) — Generality of Findings Is Unverified

**The assumption or constraint.** Every experiment in the paper uses the MATH benchmark (Hendrycks et al., 2021) with PaLM 2-S* (Anil et al., 2023) as the base model. MATH consists of 12,000 training and 500 test questions drawn from high-school mathematics competitions. The authors state in Section 4 that they "believe this model is representative of the capabilities of many contemporary LLMs," but this is an assertion, not a demonstrated fact.

**The consequence.** Several aspects of the findings could be specific to math reasoning or to PaLM 2-S*:

- **The PRM's quality and over-optimization behavior.** The PRM is trained on PaLM 2-S*'s output distribution using Monte Carlo rollouts from the same model. A model with different error patterns — e.g., one that makes different types of reasoning mistakes or has different calibration — might yield a PRM with different over-optimization thresholds, changing the difficulty bins where beam search helps vs. hurts.

- **The revision model's learning dynamics.** PaLM 2-S* may have specific in-context learning properties that make it amenable to revision training. The paper found that off-the-shelf prompting for self-correction "is largely ineffective" on math reasoning (Section 1, citing Huang et al., 2023), which means the revision capability was entirely induced through fine-tuning. Whether other model families (e.g., GPT-series, Claude, open-weight alternatives) would learn qualitatively different revision behaviors from the same training procedure is unknown.

- **The MATH benchmark's structure.** Competition math problems have well-defined, verifiable answers (exact match grading) and clear step-by-step solution structures that make them amenable to PRM training via Monte Carlo rollouts. Tasks without clean correctness signals — open-ended generation, dialogue, creative writing, code generation (where partial correctness is common), or multi-hop reasoning — would require fundamentally different verifier training approaches that the paper does not address.

- **The difficulty distribution.** MATH may have a difficulty distribution (the five quintiles) that is unusual relative to real-world deployment task distributions. The fraction of "bin 5" problems (~20%) and the fraction where test-time compute dominates pretraining may not generalize.

- **The test set size (500 questions).** Split into five difficulty quintiles of ~100 each and further split by two-fold cross-validation, the compute-optimal policy is selected based on ~50 questions per fold per bin. The paper does not report confidence intervals on the compute-optimal scaling curves themselves, making it difficult to assess whether the selected strategies are robust to this small sample size or would change significantly with more data.

**What evidence exists in the paper.** The paper contains no cross-benchmark or cross-model-family comparisons. The only "transfer" experiment is the FLOPs-matched comparison in Section 7, which uses a second PaLM 2 model with ~14× more parameters — but this is still within the same model family and training paradigm. The difficulty estimation protocol (2,048 samples per question to estimate pass@1) is benchmark-specific and does not demonstrate transfer to other task types.

**Mitigation status.** The authors do not claim generality beyond MATH, and they acknowledge the limitation implicitly by focusing only on math reasoning. Section 8 (Discussion) does not explicitly call for cross-domain replication, but the findings are framed as general principles ("test-time compute can substitute for pretraining," "difficulty-dependent allocation is optimal") rather than math-specific observations. A practitioner working in code generation, scientific QA, or factual reasoning should view these results as **motivating hypotheses** rather than established facts about their domain. Replication on additional benchmarks (code generation, logical reasoning, multi-hop QA, summarization) with additional model families would be needed to establish generality.

---

### The `$14\times$` Larger Model Baseline Is Not Compute-Optimally Trained, and the FLOPs-Matched Comparison Excludes Inference Costs

**The assumption or constraint.** The FLOPs-matched comparison in Section 7 makes two methodological choices that systematically favor test-time compute over pretraining:

**First, the larger model baseline is parameter-scaled only, not compute-optimally trained.** The paper scales only model parameters (by ~14×) while holding training data fixed, following the LLaMA paradigm (Touvron et al., 2023) rather than Chinchilla-optimal scaling (Hoffmann et al., 2022), where both parameters and data are scaled equally for a given compute budget. The authors state in Section 7:

> "We choose this setting as it is representative of a canonical approach to scaling pretraining compute and leave the analysis of compute-optimal scaling of pretraining compute where the data and parameters are both scaled equally to future work."

This matters because a Chinchilla-optimal model trained with 14× more total FLOPs (scaling data and parameters in roughly equal proportion) would likely outperform a model that only scales parameters. The reported advantages of test-time compute over pretraining — e.g., +27.8% on easy questions at `$R \ll 1$` (Figure 1, top-right bar chart) — would likely shrink or potentially reverse against a properly compute-optimal larger model trained on both more parameters and more data.

**Second, the FLOPs accounting excludes inference costs entirely.** The comparison in Section 7 uses only training-time FLOPs: pretraining FLOPs for the larger model vs. pretraining + datastore-construction FLOPs for the smaller model augmented with retrieval. However, inference cost — which is the entire point of the `$R = D_{\text{inference}}/D_{\text{pretrain}}$` analysis — is not included in the FLOPs budget. The paper computes different `$R$` values to determine how much test-time compute the smaller model can "afford" given the pretraining FLOPs savings, but the actual inference FLOPs for both approaches are not added to the total. For high-`$R$` regimes (e.g., `$R=22$`, corresponding to production deployments with large inference volumes), the inference cost of the larger model is substantial — but since it is excluded from the comparison, the FLOPs-matched analysis potentially **overstates** the advantage of test-time compute in these regimes. The paper acknowledges this in the Section 4.3 discussion on inference cost but does not integrate it into the FLOPs-matched comparison.

**The consequence.** The paper's finding that "test-time compute can outperform a 14× larger model" (Section 7) — while genuinely supported for the specific comparison made — may not generalize to a fairer comparison where:

1. The larger model is properly compute-optimally trained (scaling both parameters and data), making it a stronger baseline.
2. Inference costs are included in the total FLOPs budget, penalizing the retrieval-based model for its higher per-query inference cost (longer context length, retrieval search).
3. The larger model is given some modest test-time compute budget of its own (e.g., best-of-8 majority voting), rather than being limited to greedy decoding.

The relative magnitudes of these biases are unknown — the paper does not include ablations that vary the pretraining recipe of the larger model or include inference costs. The +27.8% advantage on easy questions at `$R \ll 1$` is likely robust (the gap is large), but the near-zero or negative advantages on harder questions (e.g., −52.9% on hard questions for PRM search at `$R \gg 1$`) might be **understated** — the disadvantage of test-time compute relative to a compute-optimally trained larger model could be even more severe.

**What evidence exists in the paper.** The paper explicitly acknowledges the Chinchilla-vs-LLaMA training recipe issue and the exclusion of inference costs, and frames both as scope limitations rather than oversights. However, no sensitivity analysis is performed — there is no estimate of how much the results would change under a Chinchilla-optimal baseline or with inference costs included.

**Mitigation status.** The paper frames the parameter-only scaling as "representative of a canonical approach" and defers the compute-optimal pretraining comparison to future work. The inference cost discussion is similarly deferred with a note about "emerging work on accelerating retrieval search and designing efficient serving strategies." For practitioners, this means the FLOPs-matched comparison should be interpreted as an **analytic comparison under specific assumptions that favor test-time compute**, not as a definitive answer to "should I spend my compute on pretraining or test-time compute?" The broad qualitative finding — that test-time compute can substitute for pretraining on problems within the model's capability range — is likely robust, but the specific magnitudes and crossing points (where pretraining becomes preferable) are uncertain.

## 7. Implications and Future Directions
- How this changes the landscape
  - Datastore size emerges as a first-class scaling axis. Practitioners can target better cost–performance by moving factual knowledge into a large non-parametric memory while keeping LMs smaller.
  - Open, trillion-scale retrieval is now practically accessible and reproducible, thanks to the released datastore, embeddings, indices, and the compute-efficient pipeline.
- Follow-up research enabled
  - Retrieval quality: The sizable gap between cross-encoder and oracle reranking (Figure 6) motivates better dense retrievers, hybrid lexical–dense strategies, or task-aware rerankers.
  - Datastore curation: Add high-quality, reasoning-focused sources (textbooks, verified medical literature) and explore more advanced filtering (semantic deduplication, topic balancing).
  - Training strategies: Jointly train LMs to better use retrieved context (e.g., retrieval-aware pretraining or instruction tuning), and study inference-compute-optimal trade-offs (Section 4.3, future work).
  - Systems and serving: Efficient search, caching, and selective augmentation (e.g., compress/retrieve only when helpful) to reduce inference latency and cost.
- Practical applications
  - Knowledge-intensive assistants: Legal, biomedical, and enterprise QA with stronger attribution and freshness via datastore updates.
  - Domain adaptation: Swap or augment domain-specific slices of MASSIVEDS without retraining the LM.
  - Governance and attribution: Easier provenance tracking and credit assignment by pointing to retrieved sources (Section 2 discussion; related works).

Overall, the study demonstrates that enlarging a retrieval datastore systematically improves performance and can be a more compute-efficient way to add knowledge than growing model size or pretraining tokens. The combination of an open trillion-token datastore and a carefully engineered, provably equivalent pipeline makes this direction immediately actionable for both research and production.
