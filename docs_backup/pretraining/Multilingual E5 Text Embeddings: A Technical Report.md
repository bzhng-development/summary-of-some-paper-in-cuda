# Multilingual E5 Text Embeddings: A Technical Report

**ArXiv:** [2402.05672](https://arxiv.org/abs/2402.05672)
**Authors:** Liang Wang, Nan Yang, Xiaolong Huang, Linjun Yang, Rangan Majumder, Furu Wei
**Institutions:** 

## 🎯 Pitch

The "mE5" models introduce a multilingual, instruction-tuned text embedding solution that outperforms traditional English-only models in cross-lingual retrieval and aligns seamlessly with diverse language pairs. This innovation significantly enhances real-world multilingual applications by bridging language gaps, ensuring high-quality retrieval and similarity tasks across global contexts.

---

## 1. Executive Summary (2–3 sentences)
This report introduces `mE5` (multilingual E5) text embedding models—`mE5-small`, `mE5-base`, `mE5-large`—and an instruction-tuned variant `mE5-large-instruct`, trained with a two-stage pipeline: large-scale contrastive pre-training (~1B multilingual text pairs) followed by supervised fine-tuning on high-quality datasets. They deliver strong multilingual retrieval and similarity performance, matching or surpassing leading English-only models on English benchmarks while offering broad cross-lingual coverage and practical efficiency trade-offs (Tables 3–5).

## 2. Context and Motivation
- Problem/gap addressed:
  - High-quality text embeddings are central to search, retrieval-augmented generation (RAG), clustering, and similarity. Yet “most existing embedding models are trained exclusively on English text” (Introduction; citing Sentence-BERT, Sentence-T5, dual encoders). This limits multilingual applications where users query and retrieve across many languages.
- Why it matters:
  - Real-world systems (global search, multilingual assistants, cross-lingual QA) require embeddings that represent meaning consistently across languages. Without multilingual embeddings, accuracy drops and systems cannot reliably bridge languages (e.g., retrieve Spanish passages from English queries).
- Prior approaches and shortcomings:
  - English-only encoders (e.g., `BGElarge-en-v1.5`) perform well on English but do not generalize across languages.
  - Multilingual models such as `LaBSE` emphasize translation-pair training and excel at sentence alignment (bitext mining) but are not tuned to the breadth of retrieval/similarity tasks now common in MTEB.
  - Earlier multilingual retrievers (e.g., `mDPR`) require task-specific fine-tuning and typically underperform large-scale contrastively trained encoders in general-purpose use.
- Positioning of this work:
  - Builds directly on the E5 recipe (two-stage contrastive pre-train + supervised fine-tune) and scales it to multilingual settings with a billion-pair pre-training mixture (Table 1) and a diverse supervised mixture (Table 2).
  - Adds an instruction-tuned embedding variant (`mE5-large-instruct`) that uses natural language task descriptions (“instructions”) to condition embeddings, aiming to generalize across tasks and languages.

## 3. Technical Approach
The approach is a two-stage training pipeline plus an instruction-tuned variant.

- Core concept: `text embeddings` are vectors representing the meaning of text spans (queries, sentences, passages) so that semantically related texts map to nearby vectors. These embeddings can power retrieval (find nearest neighbors), clustering, and similarity scoring.

Stage 1: Weakly-supervised contrastive pre-training (Table 1; “Training Methodology”)
- Objective/mechanism:
  - Use `contrastive learning` with `InfoNCE` loss to bring “positive” pairs closer and push “negative” pairs apart in embedding space. Each training step sees many examples and treats all other examples in the same batch as negatives (`in-batch negatives`).
  - Intuition: If the model consistently brings correlated text pairs together (e.g., Wikipedia title and the corresponding passage) and pushes unrelated texts apart, it learns a geometry where meaning is preserved across varied sources and languages.
- Scale and optimization:
  - Large batch size `32k` and `30k` steps:
    > “The models are trained with a large batch size 32k for a total of 30k steps, which approximately goes over ∼ 1 billion text pairs.” (Training Methodology; Weakly-supervised Contrastive Pre-training)
  - This large batch enables a strong set of negatives each step, which is known to stabilize and strengthen contrastive learning.
- Data mixture and positive-pair construction (Table 1, Appendix A):
  - Diverse multilingual sources: Wikipedia (150M), mC4 (160M), CC News (160M), NLLB translations (160M), Reddit (160M), S2ORC (50M), StackExchange (50M), xP3 (80M), SBERT misc (10M); total ~1B pairs.
  - How positives are formed (Appendix A):
    - Wikipedia: `(section title, section passage)`
    - mC4: `(page title, page content)`
    - CCNews: `(title, news content)`
    - NLLB: translation pairs (cross-lingual positives)
    - Reddit: `(comment, response)`
    - S2ORC: `(title, abstract)` and citation pairs
    - StackExchange: `(question, answer)`
    - xP3: `(input prompt, response)`
    - SBERT misc: a grab-bag of QA, summarization, code, and other datasets.
  - Why these choices:
    - They reflect many ways two texts naturally co-occur or correspond, teaching the model general semantic alignment. Including NLLB translation pairs injects explicit cross-lingual alignment.

Stage 2: Supervised fine-tuning (Table 2; “Supervised Fine-tuning”)
- Objective/mechanism:
  - Further refine the model for retrieval/similarity setups using labeled data. The training uses:
    - `in-batch negatives` (easy negatives),
    - `mined hard negatives` (look-alikes that are incorrect),
    - `knowledge distillation from a cross-encoder` (a stronger model that jointly encodes query+document to produce a fine-grained score; the bi-encoder learns to mimic its judgments), which sharpens decision boundaries for dense retrieval.
- Data mixture and scope (Table 2):
  - ~1.6M supervised pairs sampled from: MS MARCO (Passage, Document), QA datasets (NQ, TriviaQA, SQuAD, ELI5, HotpotQA), NLI, MIRACL and Mr. TyDi (multilingual retrieval), DuReader (Chinese), Fever, Quora duplicate questions, NLLB (100k), etc.
  - Coverage spans English and multilingual retrieval, QA-style matching, entailment, and deduplication—broadening generalization.
- Hyperparameters and initialization (Appendix A):
  - Initial checkpoints:
    - `mE5small` from multilingual `MiniLM`,
    - `mE5base` from `xlm-roberta-base`,
    - `mE5large` from `xlm-roberta-large`.
  - Fine-tuning with batch size 512 for 2 epochs; learning rates {3e-5, 2e-5, 1e-5} for {small, base, large}.
  - Pre-training learning rates {3e-4, 2e-4, 1e-4} for {small, base, large}.

Instruction-tuned variant: `mE5-large-instruct`
- What is instruction tuning?
  - The model receives a natural language `instruction` describing the task (e.g., “Given a question and passage, embed so that relevant answers are close”), then encodes inputs conditioned on that instruction. This can guide the embedding to better reflect task-specific similarity without changing the retrieval pipeline.
- Data and procedure (Section 2; “For the mE5-large-instruct model…”):
  - Uses the data mixture from Wang et al. (2023), adding 500k synthetic examples generated by GPT-3.5/4.
  - Includes “150k unique instructions and covers 93 languages,” and reuses the instruction templates from Wang et al. (2023) for both training and evaluation.
- Why this helps:
  - Instructions explicitly communicate the task definition to the encoder; for multi-task and multilingual use, this reduces ambiguity and can uplift performance across diverse settings.

How the pieces fit together
- Pre-training aligns general semantics across languages and domains at massive scale (including translation pairs).
- Supervised fine-tuning sharpens the model for retrieval-like tasks using curated positives, hard negatives, and a cross-encoder teacher.
- Instruction tuning adds a “task lens” the model can apply at inference time, improving flexibility and performance on many tasks.

## 4. Key Insights and Innovations
- Scaled multilingual contrastive pre-training with heterogeneous positives
  - Novelty: A production-scale mixture of multilingual sources (Table 1) with explicit translation pairs (NLLB) and varied co-occurrence signals (titles–content, Q–A, comments–replies).
  - Significance: Produces strong cross-lingual alignment while preserving general retrieval/similarity capabilities beyond bitext mining.
- Two-stage recipe ported from English E5 to multilingual with minimal changes
  - Novelty: Directly applying the E5 training recipe (contrastive pre-train + supervised fine-tune) but with multilingual coverage and task diversity.
  - Significance: Demonstrates that the E5 recipe scales to multilingual settings and yields competitive or SOTA results on English and multilingual benchmarks (Tables 3–5).
- Instruction-tuned multilingual embeddings (`mE5-large-instruct`)
  - Novelty: Adds multilingual, instruction-conditioned embedding capability using 500k synthetic examples and 150k unique instructions over 93 languages.
  - Significance: Reaches or surpasses English-only models on English benchmarks while retaining strong multilingual performance. For example:
    > “mE5large-instruct 64.4” average on MTEB-English vs “BGElarge-en-v1.5 64.2” (Table 3).
- Efficient contrastive setup with only in-batch negatives at massive batch size
  - Novelty: Uses only `in-batch negatives` (no memory bank or cross-batch tricks) but at batch size 32k, which supplies abundant negatives.
  - Significance: Simplicity and stability at scale; yields strong results, especially when combined with supervised hard negatives and distillation.

Overall, the work is an incremental but thorough engineering advance—scaling a proven recipe to multilingual settings and adding a practical instruction-tuned variant—rather than a new theoretical framework.

## 5. Experimental Analysis
Evaluation methodology
- Benchmarks and tasks:
  - English portion of MTEB (56 datasets; Table 3; Appendix Table 7), covering semantic similarity (STS), classification, clustering, and retrieval/reranking.
  - Multilingual retrieval on MIRACL dev set (16 languages; Table 4; Appendix Table 6) with `nDCG@10` and `R@100`.
    - `nDCG@10` (normalized Discounted Cumulative Gain): measures ranking quality at the top 10, rewarding correct items more if ranked higher.
    - `R@100` (Recall@100): fraction of relevant items found among the top 100 retrieved.
  - Bitext mining on BUCC 2018 (4 languages) and Tatoeba (112 languages) (Table 5), a cross-lingual similarity task matching translated sentence pairs.
- Baselines:
  - For MTEB-English: `LaBSE`, `Coheremultilingual-v3`, `BGElarge-en-v1.5`.
  - For MIRACL: BM25 (lexical), `mDPR` (a multilingual dense retriever fine-tuned on MIRACL).
  - For bitext: `mContrievermsmarco` and `LaBSE`.

Main quantitative results
- MTEB (English portion; Table 3; averages over 56 datasets):
  > LaBSE 45.2; Coheremultilingual-v3 64.0; BGElarge-en-v1.5 64.2; mE5small 57.9; mE5base 59.5; mE5large 61.5; mE5large-instruct 64.4.
  - `mE5-large-instruct` edges the strong English-only `BGElarge-en-v1.5` by +0.2, and the best non-instruct multilingual E5 by +2.9.
  - Model scaling helps: small → base → large improves from 57.9 → 59.5 → 61.5.
  - Per-dataset details (Appendix Table 7) show where instruction tuning helps or hurts (see trade-offs below).
- Multilingual retrieval on MIRACL (Table 4):
  > nDCG@10: BM25 39.3; mDPR 41.5; mE5small 60.8; mE5base 62.3; mE5large 66.5; mE5large-instruct 65.7.  
  > R@100: BM25 78.7; mDPR 78.8; mE5small 92.4; mE5base 93.1; mE5large 94.3; mE5large-instruct 94.6.
  - All mE5 variants far surpass BM25 and mDPR.
  - `mE5-large` has the best nDCG@10 (66.5), while `mE5-large-instruct` has the best R@100 (94.6), indicating instruction-tuned embeddings retrieve more relevant items in the top 100 but rank them slightly worse at the very top-10 on average.
  - Per-language results (Appendix Table 6) show strong cross-language consistency; for example:
    > `zh` nDCG@10 rises from 45.9 (small) → 56.0 (large) → 56.2 (instruct); `te` reaches 84.6 (large) and 83.4 (instruct); `ru` 67.4 (large) and 67.9 (instruct).
- Bitext mining (Table 5):
  > BUCC: mContrievermsmarco 93.7; LaBSE 98.8; mE5small 93.2; mE5base 98.1; mE5large 98.6; mE5large-instruct 99.0.  
  > Tatoeba: mContrievermsmarco 37.7; LaBSE 81.1; mE5small 64.2; mE5base 68.1; mE5large 75.7; mE5large-instruct 83.8.
  - `mE5-large-instruct` slightly surpasses LaBSE on BUCC (99.0 vs 98.8) and notably on Tatoeba (83.8 vs 81.1), despite LaBSE being tailored for bitext mining. The report attributes this to broader language coverage from synthetic instruction data.

Support for claims and trade-offs
- Do the experiments substantiate the core claims?
  - Yes for multilingual retrieval and bitext mining: substantial improvements over strong baselines (BM25, mDPR; Table 4) and competitive with a specialized bitext model (LaBSE; Table 5).
  - On English MTEB, instruction tuning lifts average performance to match a strong English-only model (Table 3). This is notable for a multilingual encoder.
- Mixed results at the task level (Appendix Table 7):
  - Instruction tuning improves many classification and STS tasks (e.g., `AmazonPolarity`: 96.3 vs 93.5; `STS13`: 87.2 vs 81.5; `RedditClustering`: 56.6 vs 46.5; `TwentyNewsgroupsClustering`: 51.3 vs 38.9).
  - But it can hurt some retrieval datasets: `MSMARCO` (40.4 vs 43.7), `NQ` (57.8 vs 64.1), `FEVER` (78.0 vs 82.8), `HotpotQA` (69.3 vs 71.2), `DBPedia` (38.4 vs 41.3).
  - Interpretation: instructions likely steer representations toward the instructed task framing. When evaluation aligns with those instructions, performance rises; when it diverges (or when instructions are suboptimal), retrieval may slightly degrade. This is consistent with the MIRACL pattern—higher recall@100 but slightly lower nDCG@10 for the instruction-tuned model.

Ablations, failure cases, robustness
- The report does not include ablations isolating the contributions of mined hard negatives, distillation, or specific data subsets. However, it does provide size-scaling comparisons (small/base/large) and instruct vs non-instruct variants.
- No explicit error analyses or robustness checks (e.g., adversarial negatives, domain shift) are reported.

## 6. Limitations and Trade-offs
- Compute and scale assumptions:
  - Pre-training uses batch size 32k and 30k steps over ~1B pairs; this presupposes substantial compute and memory. The choice to rely solely on `in-batch negatives` becomes viable because of the massive batch, which may be impractical for some users to reproduce.
- Data mixture and coverage:
  - While multilingual, the paper does not quantify per-language data proportions in pre-training beyond sample counts per source (Table 1). Performance for very low-resource languages may depend heavily on NLLB translation coverage and synthetic instruction data; coverage breadth is high (93 languages in instruction data; 112 in Tatoeba) but depth per language may vary.
- Evaluation scope:
  - English MTEB results are comprehensive (Table 3; Appendix Table 7). For non-English tasks beyond MIRACL and bitext mining, broader multilingual MTEB evaluations are not reported, leaving some uncertainty about non-retrieval multilingual tasks at large scale.
- Instruction-tuning trade-offs:
  - While `mE5-large-instruct` improves average English MTEB and several multilingual metrics, it can slightly degrade performance on certain retrieval datasets (Appendix Table 7; Table 4). This suggests instructions may bias embeddings toward specific notions of similarity, which is not universally optimal for all retrieval tasks.
- Transparency on training details:
  - The report provides key hyperparameters and datasets (Appendix A; Tables 1–2) but does not detail hardware, total FLOPs, or the precise cross-encoder used for distillation. Reproducing results end-to-end may require additional engineering knowledge.
- Potential biases:
  - Like all web-scale training, the model inherits biases from sources (Reddit, CCNews, mC4). The multilingual instruction data also relies on GPT-3.5/4 generation, whose coverage and biases may vary by language and domain.

## 7. Implications and Future Directions
- Impact on the field:
  - This work demonstrates that a scalable, two-stage E5 recipe extends well to multilingual settings without specialized architecture—delivering strong cross-lingual retrieval and sentence alignment while remaining competitive on English tasks. It helps close the gap between English-only and multilingual embedding quality, especially valuable for global applications.
- Practical applications:
  - Multilingual search and retrieval across websites and knowledge bases.
  - Cross-lingual RAG: query in one language, retrieve context in another.
  - Bitext mining for machine translation data creation and quality checks.
  - Multilingual clustering, deduplication, and semantic similarity in analytics pipelines.
  - Instruction-conditioned embedding for multi-task systems where the desired similarity notion can be specified at inference time.
- Suggested follow-ups:
  - Ablations to disentangle the effects of hard negatives, distillation, and individual pre-training sources, especially NLLB translation pairs.
  - Broader multilingual evaluations beyond MIRACL (e.g., full multilingual MTEB variants), and deeper analyses for low-resource languages.
  - Methods to dynamically choose or learn instructions at inference time, mitigating the observed retrieval trade-offs while preserving gains.
  - Efficient training strategies to relax the need for extremely large batches (e.g., cross-batch memory, curriculum negatives).
  - Safety, fairness, and bias audits across languages and domains; targeted data augmentation for underrepresented languages.
  - Domain adaptation recipes (e.g., biomedical, legal) leveraging the strong multilingual base, possibly with lightweight adapters.

In sum, the report operationalizes a practical, high-performance recipe for multilingual text embeddings at scale, quantifies strong cross-lingual retrieval and alignment performance (Tables 4–5), and shows that instruction-tuning can make a multilingual encoder competitive with the best English-only models on English benchmarks (Table 3), albeit with task-specific trade-offs visible in per-dataset results (Appendix Table 7).
