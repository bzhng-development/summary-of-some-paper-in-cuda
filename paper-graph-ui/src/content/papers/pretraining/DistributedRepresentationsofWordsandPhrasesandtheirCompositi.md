# Distributed Representations of Words and Phrases and their Compositionality

**ArXiv:** [1310.4546](https://arxiv.org/abs/1310.4546)

## 🎯 Pitch

This seminal paper introduces dramatic improvements to the Skip-gram model by proposing negative sampling for faster, more scalable training, and subsampling of frequent words to enhance vector quality for rare terms. Crucially, it presents an effective method for learning high-quality representations not only for words, but also for multi-word phrases, enabling the model to capture nuanced linguistic patterns, linear analogies, and compositional meanings. These innovations have transformed large-scale natural language processing by making it possible to efficiently train on billions of words, unlocking richer and more generalizable word and phrase embeddings foundational to modern NLP applications.

---

## 1. Executive Summary
This paper extends the `Skip-gram` approach for learning word vectors by introducing two training improvements—`negative sampling` and `subsampling of frequent words`—and by showing how to learn vectors for multi‑word `phrases`. These changes make training dramatically faster and yield vectors that encode precise linear relationships (including phrase–level analogies and meaningful vector addition), which matters for scalable language understanding from large text corpora.

## 2. Context and Motivation
- Problem addressed
  - How to learn high‑quality, low‑dimensional vector representations for words—and crucially, for multi‑word phrases—that (a) capture syntactic and semantic regularities and (b) can be trained efficiently on billions of tokens. See Introduction (Section 1) and the Skip‑gram objective (Section 2, Eq. (1)).

- Why this matters
  - Many NLP systems benefit from reusable word/phrase vectors that encode meaning and relationships. Efficient training enables using very large corpora, which improves coverage of rare words and entities (Section 1). The paper also targets representations that support algebraic reasoning, useful for analogies and compositional semantics (Figure 2; Sections 1 and 5).

- Prior approaches and shortcomings
  - Earlier neural language models learned useful vectors but required heavy computations (dense matrix multiplications) that limited corpus size (Section 1).
  - The original `Skip-gram` model [8] was efficient but still had two key gaps:
    - Training with a full softmax was impractical for large vocabularies (Section 2, Eq. (2)).
    - Word‑level models ignore word order and cannot represent non‑compositional phrases like “New York Times” (Section 1).
  - Existing approximations (e.g., hierarchical softmax; NCE) helped with speed but had trade‑offs (Sections 2.1 and 2.2).

- Positioning
  - The paper keeps the efficient Skip‑gram framing but alters how probabilities are trained/approximated and how the text is tokenized:
    - A new training loss (`negative sampling`) simplifies NCE (Section 2.2, Eq. (4)).
    - A simple frequency‑based `subsampling` scheme speeds up learning and improves rare‑word quality (Section 2.3, Eq. (5)).
    - A data‑driven `phrase discovery` step lets the same training method learn vectors for millions of phrases (Section 4, Eq. (6)).

## 3. Technical Approach
The core idea remains: learn a vector for each token so that the token predicts its nearby tokens.

- Base Skip‑gram objective (Section 2)
  - For each position t in a text sequence, with center word `w_t`, predict the words in a context window of size `c` around it.
  - Objective (Eq. (1)): maximize the average log probability of observing each context word `w_{t+j}` given `w_t`.
  - Naively using softmax for `p(w_{O} | w_{I})` (Eq. (2)) requires computing a normalization over the whole vocabulary `W` for each training step, which is expensive when `W` is large.

- Hierarchical softmax (Section 2.1; Eq. (3))
  - What it is: Replace the full softmax with a binary tree whose leaves are words. Predict a path of binary decisions from the root to a target word.
  - How it reduces cost: Each prediction traverses `~log2(W)` nodes instead of summing over `W` outputs. Each inner node has its own vector `v'_{n}`; each word has an input vector `v_w`. The probability of a word is the product of logistic (`σ`) probabilities along its path (Eq. (3)).
  - Design choice: use a `Huffman tree` so frequent words have shorter paths, further speeding training (Section 2.1).

- Negative sampling (Section 2.2; Eq. (4))
  - What it is: A simplified objective that turns each training pair `(w_I, w_O)` into a binary classification task—make the dot product `v'_{w_O}^T v_{w_I}` large for true pairs and small for `k` randomly sampled “noise” words `w_i ~ P_n(w)`.
  - Loss for one pair (Eq. (4)):
    - Reward the positive pair with `log σ(v'_{w_O}^T v_{w_I})`.
    - For each of `k` negatives from noise distribution `P_n(w)`, add `log σ(- v'_{w_i}^T v_{w_I})`.
  - Why this helps:
    - Complexity depends on `k`, not vocabulary size.
    - Empirically learns better vectors for frequent words than hierarchical softmax (Table 1).
  - Critical choices:
    - `k` (number of negatives): small datasets benefit from `k=5–20`; large datasets can use `k=2–5` (Section 2.2).
    - Noise distribution `P_n(w)`: best results use the unigram distribution raised to the `3/4` power and renormalized, i.e., `U(w)^{3/4}/Z` (Section 2.2).

- Subsampling of frequent words (Section 2.3; Eq. (5))
  - Problem: very frequent tokens (e.g., “the”) dominate training but carry little information for predicting semantics; they also change slowly after many updates.
  - Mechanism: Discard each token `w_i` with probability `P(w_i) = 1 - sqrt(t / f(w_i))`, where `f(w_i)` is its frequency and `t ≈ 10^-5`.
  - Effect:
    - Favors informative, rarer words.
    - Yields 2×–10× speedups and improves accuracy for less frequent words (Section 2.3; Table 1 shows both speed and accuracy gains).

- Learning phrase vectors (Section 4; Eq. (6))
  - Issue: Many meanings are non‑compositional at the word level (e.g., “Boston Globe” is a newspaper).
  - Pipeline:
    1. Detect candidate bigram phrases using counts and the score:
       `score(w_i, w_j) = (count(w_i w_j) - δ) / (count(w_i) × count(w_j))`, where `δ` discounts accidental bigrams (Eq. (6)).
    2. Keep bigrams whose score exceeds a threshold; run 2–4 passes with decreasing thresholds to form longer multi‑word phrases (Section 4).
    3. Replace accepted phrases by single tokens in the text and train exactly as for words.
  - Evaluation: A new phrase analogy dataset is created (Table 2) to test whether phrase vectors support analogy reasoning.

- Why analogies and vector arithmetic work (Sections 1 and 5)
  - Linear regularities: many relationships correspond to linear offsets; for instance, `vec("Madrid") - vec("Spain") + vec("France") ≈ vec("Paris")` (Section 1).
  - Additive compositionality (Section 5): Because vectors are trained to predict context via a log‑linear model, summing two vectors roughly corresponds to multiplying their context distributions (an “AND” effect). Example outcomes are given in Table 5 (e.g., `Germany + capital → Berlin`).

## 4. Key Insights and Innovations
- Negative sampling as a simple, effective training objective (Section 2.2; Eq. (4))
  - Novelty: A streamlined alternative to NCE that only requires sampling negatives (not computing their probabilities), turning multinomial prediction into multiple binary classifications.
  - Significance: Achieves higher analogy accuracy than hierarchical softmax and NCE on word analogies while being efficient (Table 1).

- Subsampling of frequent words (Section 2.3; Eq. (5))
  - Novelty: A simple frequency‑based discard rule that preserves relative frequency ranking but aggressively prunes high‑frequency tokens.
  - Significance: Delivers 2×–10× speedups and better rare‑word vectors; the effect is visible in both training time and accuracy (Table 1).

- Phrase discovery and phrase vector training at scale (Section 4; Eq. (6))
  - Novelty: A lightweight, count‑based scoring scheme to identify phrases and fold them into Skip‑gram training as single tokens, enabling vectors for millions of phrases.
  - Significance: Produces strong phrase analogy performance; with larger data (33B words), phrase analogy accuracy reaches 72% (Section 4.1).

- Demonstration and explanation of additive compositionality (Section 5)
  - Novelty: Clear empirical and conceptual link between the Skip‑gram objective and meaningful vector addition (Table 5).
  - Significance: Enables simple composition of meanings (e.g., `Russia + river → Volga River`) without complex architectures.

These are primarily fundamental innovations in training objective design (negative sampling and subsampling) and in representational scope (phrases), rather than minor parameter tweaks.

## 5. Experimental Analysis
- Evaluation methodology
  - Tasks:
    - Word analogy reasoning (countries–capitals, morphological relations, etc.), following [8] (Section 3).
    - New phrase analogy reasoning task covering multiple categories (Table 2; Section 4).
  - Metrics: Accuracy on analogical questions; nearest‑neighbor qualitative probes for semantics (Tables 4–6). Training time in minutes is also reported (Table 1).
  - Data:
    - Primary word‑level experiments: ~1B words from news; vocabulary trimmed to words appearing ≥5 times (vocab size 692K) (Section 3).
    - Phrase‑level scaling: up to ~33B words; also a 6B‑word subset to study data size effects (Section 4.1).
  - Models and settings:
    - `Skip-gram` with 300 dimensions and context size 5 for many comparisons (Section 4.1).
    - Training variants: hierarchical softmax (Huffman tree), NCE, negative sampling with different `k`, with/without subsampling (Sections 2–4).

- Main quantitative results
  - Word analogies (Table 1; 300‑dim; 1B words):
    > Without subsampling, `NEG-15` achieves 61% total accuracy (63% syntactic, 58% semantic) in 97 minutes; hierarchical softmax (HS) achieves 47% in 41 minutes; NCE‑5 achieves 53% in 38 minutes.  
    > With `t=10^-5` subsampling, `NEG-15` remains at 61% but runs in 36 minutes; HS rises to 55% and runs in 21 minutes; `NEG-5` improves slightly to 60% and runs in 14 minutes.
    - Takeaway: Negative sampling is best for word analogies; subsampling speeds up all methods and particularly improves HS.

  - Phrase analogies (Table 3; 300‑dim; 1B words):
    > Without subsampling: `NEG-15` = 27%, `NEG-5` = 24%, HS = 19%.  
    > With `t=10^-5` subsampling: HS = 47% (best), `NEG-15` = 42%, `NEG-5` = 27%.
    - Takeaway: For phrases, hierarchical softmax benefits most from subsampling and becomes the top performer in this setting.

  - Scaling with more data (Section 4.1):
    > Using ~33B words, hierarchical softmax with 1000 dimensions and full‑sentence context achieves 72% accuracy on phrase analogies; reducing data to 6B words drops accuracy to 66%.
    - Takeaway: More data substantially improves phrase representations.

- Qualitative evidence
  - Linear structure: PCA projection shows countries and their capitals aligned (Figure 2).
  - Phrase neighbors: Examples illustrate semantic coherence (Table 4).
  - Additive compositionality: Element‑wise sums yield correct targets like “koruna” for `Czech + currency` (Table 5).
  - Comparison to prior released vectors: Nearest‑neighbor lists show the large Skip‑gram model offers more precise semantics for rare words (Table 6).

- Do the experiments support the claims?
  - Efficiency: Training time reductions with subsampling (Table 1) and feasibility on 33B tokens (Section 4.1) demonstrate scalability.
  - Quality: Higher analogy accuracy for negative sampling (words) and for HS with subsampling (phrases), plus compositional examples, support the representational claims.
  - Ablations/trade‑offs:
    - Varying `k` in negative sampling (Table 1, Table 3) shows gains from larger `k`, especially on phrases.
    - With/without subsampling is carefully compared, showing both speed and accuracy effects.

- Missing or limited evaluations
  - No extrinsic task benchmarks (e.g., downstream NLP tasks) in this paper; focus is on analogy tests and qualitative nearest neighbors.
  - Robustness to domain shifts or low‑resource settings is not studied.

## 6. Limitations and Trade-offs
- Modeling assumptions and scope
  - The Skip‑gram context model discards word order within the window; phrase tokenization partly addresses idioms but not general compositional syntax (Sections 1 and 4).
  - Phrase discovery relies on frequency‑based bigram scoring (Eq. (6)) and thresholds; this may miss low‑frequency but important phrases or mistakenly merge frequent but non‑idiomatic bigrams.

- Training objective trade‑offs
  - Negative sampling requires choosing `k` and a noise distribution `P_n(w)`; the recommended `U(w)^{3/4}` is empirically chosen (Section 2.2) without a theoretical optimality guarantee.
  - Hierarchical softmax vs negative sampling:
    - Negative sampling performs best on word analogies (Table 1).
    - Hierarchical softmax, when combined with subsampling, performs best on phrase analogies (Table 3).
    - Thus, the “best” method can be task‑dependent.

- Data and compute
  - Best phrase results (72%) require ~33B tokens and 1000‑dimensional vectors (Section 4.1), which presumes access to very large corpora and non‑trivial compute—even though the model is efficient relative to prior work.

- Evaluation limits
  - Analogy accuracy captures certain linear relations but not all aspects of meaning; the paper notes that even non‑linear models can develop linear structures with enough data (Section 3 discussion), hinting that analogy tests may not fully discriminate model capabilities.

## 7. Implications and Future Directions
- How this work changes the landscape
  - Establishes a practical recipe—`Skip-gram + negative sampling + subsampling`—that scales to billions of tokens and yields vectors with strong linear semantics. The phrase‑token approach widens coverage to names, entities, and idioms, enabling vector operations over multi‑word concepts.

- Follow‑up research enabled/suggested
  - Better phrase discovery:
    - Move from count‑based bigrams to context‑ or dependency‑based segmentation to capture low‑frequency but semantically tight phrases (building on Section 4’s multi‑pass thresholding idea).
  - Adaptive objectives:
    - Explore principled choices of `P_n(w)` and dynamic `k` to balance rare vs frequent token learning (motivated by Section 2.2 observations).
  - Beyond analogies:
    - Systematic extrinsic evaluations on tasks like machine translation, information retrieval, and question answering using the learned phrase vectors.
  - Richer compositionality:
    - Combine the demonstrated additive property (Section 5) with syntactic information (e.g., dependency contexts) to model more complex compositions.
  - Low‑resource and multilingual settings:
    - Investigate whether subsampling and negative sampling remain optimal when data is scarce; extend phrase discovery to languages with different tokenization or morphology.

- Practical applications
  - Entity‑centric search and recommendation: phrase vectors for organizations, products, teams (Table 2 categories).
  - Knowledge base completion and relation extraction: linear offsets and vector addition facilitate relation discovery (Figure 2; Tables 5–6).
  - Fast pretraining for downstream NLP models: efficient training on massive text corpora yields strong general‑purpose embeddings.

Overall, the paper delivers a compact, scalable toolkit for learning both word and phrase vectors and clarifies why simple vector arithmetic can reflect meaningful language composition, supported by targeted experiments across training strategies (Tables 1 and 3), scaling (Section 4.1), and qualitative probes (Figures 2; Tables 4–6).
