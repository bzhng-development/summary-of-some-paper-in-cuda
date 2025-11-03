# Infini-gram: Scaling Unbounded n-gram Language Models to a Trillion Tokens

**ArXiv:** [2401.17377](https://arxiv.org/abs/2401.17377)

## 🎯 Pitch

This paper introduces the ∞-gram language model—a modern, unbounded n-gram approach—and the infini-gram engine, enabling state-of-the-art n-gram modeling over trillions of tokens via an efficient suffix array architecture. By allowing for arbitrarily large contexts and making n-gram statistics instantly queryable at massive scale, Infini-gram demonstrates that classical n-gram LMs are not only still relevant but can significantly improve neural language models, reducing perplexity by up to 73%. This work redefines the role of n-gram models in the era of neural LLMs by empowering transparent text analysis, robust data inspection, and practical hybrid modeling at unprecedented scale.

---

## 1. Executive Summary
This paper introduces an “unbounded” n‑gram language model (`∞‑gram`) and a high‑throughput engine (`infini‑gram`) that serves it over trillions of tokens using suffix arrays. It shows that (a) long‑context n‑gram statistics are surprisingly predictive by themselves and (b) when linearly interpolated with neural LMs, they reduce perplexity substantially—up to 73% relative improvement on certain domains (Table 2) and consistent gains even for large models like `Llama‑2‑70B` (Table 1).

## 2. Context and Motivation
- Problem addressed
  - Can classical n‑gram language models still matter at modern data scales, and if so, how can we make them both accurate and practical? (§1)
  - Two long‑standing obstacles:
    1) Data scale: prior n‑gram tables were built for small n (typically n ≤ 5) and relatively smaller corpora, because count tables grow explosively with n and corpus size (Introduction; §2).  
    2) Context length: small n discards most of the prompt, which hurts next‑token prediction (Figure 1; §4.1).

- Importance
  - Practical: fast, easily interpretable counts enable analysis of corpora (contamination checks, PII filtering, retrieval) and can augment neural LMs (§E).
  - Scientific: reveals where modern neural LMs agree/disagree with corpus statistics; surfaces training irregularities (e.g., positional‑embedding artifacts shown by periodic agreement drops under greedy decoding; §4.2, Figure 5).

- Prior approaches and their gaps
  - Classical n‑gram LMs: usually n ≤ 5; huge count tables; smoothing/backoff to handle sparsity; limited long‑context modeling (Brants et al., 2007; §2).
  - Nonparametric neural augmentations (e.g., kNN‑LM, RETRO): powerful but storage/computation‑heavy (Table 6; §F). Often store vectors for billions of items (hundreds of TB).
  - Suffix trees/arrays for unbounded n were explored but at limited scales or without a proper probability model (§6; §F).

- Positioning of this work
  - Scales unbounded n‑gram modeling to 5T tokens across multiple corpora (Dolma, RedPajama, The Pile, C4)—largest of its kind (§3; A.1–A.2).
  - Introduces a valid probability formulation for `∞‑gram` that uses as much context as exists in the data and backs off only when necessary (§2), and an engine that answers n‑gram/∞‑gram queries in tens to hundreds of milliseconds from disk (Table 3; §A.5).

## 3. Technical Approach
There are two main pieces: the language model (`∞‑gram`) and the serving engine (`infini‑gram`).

- Background: n‑gram LM (brief refresher)
  - An n‑gram LM estimates `P(w_i | w_{i−(n−1):i−1})` by counts: `cnt(context + token) / cnt(context)`. Classic models smooth or back off to shorter contexts when counts are zero (§2). Small n loses long‑range information (Figure 1).

- The `∞‑gram` LM (Section §2)
  - Key idea: use the longest suffix of the entire prompt that appears at least once in the training data (“effective n”). If a longer suffix has zero count (i.e., never appears), back off by one token, repeatedly, until the denominator (`cnt(context)`) is positive.
  - Probability definition
    - For position i with preceding tokens `w1:i−1`, choose `n` equal to 1 + length of the longest suffix of `w1:i−1` that appears in the data.  
    - Then compute `P∞(w_i | w1:i−1) = cnt(suffix + w_i) / cnt(suffix)` where the suffix length is `n−1`.  
    - Because `n` depends only on the context (not on the candidate `w_i`), `P∞(· | context)` forms a valid distribution without extra discounting (§2).
  - “Sparse” estimates
    - If only one next token was ever seen after the chosen suffix, the estimate is “sparse”: probability 1 for that token and 0 for all others. Sparse cases turn out to be highly predictive (§4.1; right panel of Figure 3).

- Interpolating with neural LMs (Section §2; §5)
  - Combined model: `P(y | x) = λ · P∞(y | x) + (1 − λ) · P_neural(y | x)`.  
  - Practical variant: use two weights, `λ1` for sparse ∞‑gram events and `λ2` for non‑sparse ones, tuned on validation to minimize perplexity (§5).

- The `infini‑gram` engine (Section §3; Appendix A)
  - Representation
    - Tokenize the corpus and store it as a byte array (“token array”), using 2 bytes per token ID (assumes vocabulary size < 65,536). Insert a document separator token `\xff\xff`.  
    - Build a suffix array over the token array. A suffix array stores the starting positions of all suffixes, sorted lexicographically. It supports substring counting by binary search (Figure 2).
  - Space and build cost
    - Each pointer in the suffix array needs about 5 bytes for shards of size 2B–500B tokens; the token array uses 2 bytes per token. Total index size ≈ 7 bytes/token (§3).  
    - For RedPajama (1.4T tokens), building took ~48 hours on a single 128‑core CPU node with 1 TiB RAM and ~10 TB disk (§3; §A.3).
  - Counting and probability queries
    - Counting `cnt(x1...xn)`: in the suffix array, all occurrences form one contiguous segment; binary search finds the segment’s bounds, so runtime is O(log N) random accesses (§3; §A.4).  
    - n‑gram LM probability: compute `cnt(context + token) / cnt(context)` using two counts.  
    - `∞‑gram` probability: find the longest matching suffix length by a binary‑lifting + binary‑search procedure (O(log L) counts), then compute the ratio (§A.4 “Speeding up ∞‑gram computation”).
  - Optimizations for low latency (Appendix §A.4)
    - Parallel shard processing; hinted search; memory prefetching; reusing previous search segments across adjacent n’s; amortized processing across consecutive tokens (effective n can increase by at most 1).
  - Latency (Table 3; §A.5)
    - Count an n‑gram (any n up to 1000 tested): ~13–20 ms.  
    - n‑gram LM next‑token distribution: ~31–39 ms/token.  
    - `∞‑gram` probability: ~90–135 ms; full `∞‑gram` distribution: ~88–180 ms/token.  
    - All while keeping indexes on disk; no GPU required.
  - Extra features
    - Indexes are additive/subtractive across datasets and shards (§A.2–A.3).  
    - Document‑level retrieval, including CNF queries over multiple n‑grams (`SEARCHDOC`; §A.5 and Figures 11–16).

- Decontamination of reference data (Appendix §B)
  - To avoid evaluating on training duplicates, they remove documents in the ∞‑gram reference sets that have high overlap with The Pile’s validation/test sets using the “Big Friendly Filter” with 13‑gram overlap at an 80% threshold (Table 4; §4).

## 4. Key Insights and Innovations
- Unbounded n‑gram with a valid distribution (fundamental)
  - Starting backoff from “infinite” context and backing off only when the denominator is zero produces a proper probability distribution without Katz‑style discounting (§2). This is conceptually simple yet crucial for using long contexts reliably.

- Suffix‑array engine that makes trillion‑token, unbounded‑n modeling practical (systems innovation)
  - 7 bytes per token, on‑disk operation, millisecond‑level latencies, with a reproducible single‑node build for 1.4T tokens in ~2 days (§3; Table 3; §A). Previous unbounded approaches (suffix trees) were too large or were not true probability models (§6; §F).

- Empirical finding: long‑context corpus statistics are strong predictors and complement neural LMs (scientific insight)
  - For human text, `∞‑gram` matches the true next token 47% of the time overall, exceeding small n‑gram models (Figure 3; §4.1).  
  - When `∞‑gram` is sparse, accuracy jumps to 75% overall and >80% for effective n ≥ 14 (Figure 3, right).  
  - Interpolating with `∞‑gram` reduces perplexity across model families and sizes, including large `Llama‑2‑70B` (Table 1) and `SILO` models (Table 2).

- Diagnostic lens on decoding and model internals (analytical capability)
  - Agreement profiles vs. suffix length reveal that nucleus sampling best resembles human text; greedy decoding shows strong fluctuations and even periodic accuracy drops (e.g., at effective n = 20, 24, 28, 32 for `Llama‑2‑7B`, p < 1e−99; Figure 5; §4.2), pointing to potential positional‑embedding issues.

## 5. Experimental Analysis
- Reference data and tokenizers
  - Built `infini‑gram` indexes for Dolma (3T), RedPajama (1.4T), The Pile (≈380B), C4 (≈200B), and often evaluate with Pile‑train as the reference data for analyses (§3; §4).  
  - Separate indexes for different tokenizers: GPT‑2/Neo/J share one; Llama‑2 another; SILO another (§5.1 “Tokenizers”).

- Human‑written text analysis (Section §4.1; Figure 3; Figure 4)
  - Setup: Sample 50 docs per Pile‑val domain (≈50k tokens/domain), truncate to 1024 tokens; compute `∞‑gram` probability for the true next token and call it “accurate” if >0.5 (lower bound on argmax accuracy) (§4.1).  
  - Findings:
    - Overall `∞‑gram` accuracy = 47%. Accuracy increases with effective n; for n ≥ 16, >75% (Figure 3, middle).  
    - Small 5‑gram LM has much lower accuracy (left of Figure 3), reflecting insufficient context (the Abstract quantifies 29%).  
    - Sparse `∞‑gram` covers >50% tokens and yields 75% overall accuracy; for effective n ≥ 14, >80% (Figure 3, right).  
    - Complementarity with neural LMs: when `Llama‑2` assigns very low probability to the true token, `∞‑gram` still agrees >20%; if sparse only, ≈50% (Figure 4). This motivates interpolation.

- Machine‑generated text analysis (Section §4.2; Figure 5; Figure 9)
  - Setup: Prompt neural LMs with first 50 tokens of each Pile‑val doc; generate until original length/EOS using greedy, temperature, or nucleus sampling. Models include Llama‑2 70B/13B/7B and GPT‑J/Neo variants (§4.2).  
  - Findings:
    - Decoding matters: nucleus sampling yields an effective‑n distribution most similar to human text; greedy decoding uses longer effective contexts and shows large agreement oscillations with suffix length (Figure 5 top; §4.2).  
    - Model size: larger models show slightly higher effective n and higher agreement (Figure 5 bottom). GPT‑Neo/J agree more with `∞‑gram` likely because they trained on The Pile (which is also the `∞‑gram` reference set there).  
    - Notable irregularity: periodic agreement drops for `Llama‑2‑7B` under greedy decoding (effective n = 20,24,28,32; p < 1e−99), hypothesized to relate to positional embeddings (§4.2).

- Perplexity improvements by interpolation (Section §5; Table 1; Table 2; Table 5; Figure 10)
  - Across GPT‑2, GPT‑Neo/J, and Llama‑2 on Pile‑val/test (Table 1):
    - Consistent gains, decreasing with model size within a family, but still sizeable for large models.  
    - With Pile‑train as reference data, `Llama‑2‑70B` improves from 4.59 to 4.21 on validation and 4.65 to 4.20 on test (≈11–12% relative improvement).  
    - With Pile‑train + RedPajama (1.8T tokens), `Llama‑2‑70B` drops below perplexity 4.0 (to ~3.96–3.95), and `Llama‑2‑13B + ∞‑gram` beats `Llama‑2‑70B` (Table 1 bottom block).  
    - GPT‑2 family benefits more than GPT‑Neo/J, consistent with GPT‑Neo/J training on The Pile (closer to the `∞‑gram` reference), highlighting the value of complementary data distributions (§5.2).
  - On SILO (permissively licensed data; 1.3B) across Wikipedia, Enron Emails, NIH ExPorters (Table 2):
    - Very large relative gains, especially in‑domain or related domains. Example: on Enron Emails test, `SILO‑PD` improves from 20.62 to 4.85 (73% improvement).  
    - Outperforms alternative nonparametric augmentations kNN‑LM and RIC‑LM that use much smaller reference sets (45M–1.2B tokens) (Table 2 notes).
  - Time‑shifted evaluation (new Wikipedia pages after Pile/RPJ cutoff; Table 5):
    - Simple global interpolation improves in 4/5 months by 3–6% (sometimes 0%).  
    - Instance‑wise λ via a Random Forest (features: suffix lengths and counts) yields stronger gains, 3–20% (Table 5, right column), supporting generalization beyond the reference‑data time window (§D.2).
  - Scaling ablation (Figure 10; §D.3):
    - Gains increase roughly log‑linearly as the reference datastore grows (downsampling study).  
    - Using only in‑domain reference data is about as strong as using the full Pile, indicating most benefit comes from in‑domain text—even after decontamination.

- Engine performance (Table 3; §A.5)
  - Sub‑200 ms per token for `∞‑gram` distributions on trillion‑token corpora; counting any n‑gram in ~20 ms with latency largely independent of n (tested up to n = 1000).

- Note on generation quality (Section “A note on text generation” within §5.2)
  - Despite better perplexity, naive interpolation can harm open‑ended generation because `∞‑gram` occasionally suggests off‑topic tokens and can derail the output. This combined model is not yet a drop‑in decoder replacement.

Overall, the experiments are broad (multiple model families, sizes, domains, and time‑shifted data) and include careful decontamination (Table 4), making the case that unbounded corpus statistics are both predictive and complementary. The diagnostics (Figures 3–5) also add qualitative insight beyond mere perplexity numbers.

## 6. Limitations and Trade-offs
- Zero probabilities and sparsity
  - `∞‑gram` assigns zero to unseen continuations; its standalone perplexity is undefined on typical test sets. The paper always evaluates perplexity for the interpolated model (§2; §5). This design requires tuning λ (and λ1/λ2 for sparse/non‑sparse).

- Generation vs. scoring
  - While perplexity improves, naive decoding with `∞‑gram` interpolation can degrade open‑ended generation (“odd mistakes” that cause digressions; §5.2 note). More sophisticated integration is needed for generation.

- Data and tokenizer assumptions
  - The efficient token storage assumes vocabulary size < 65,536 (2 bytes/token). Separate indexes are needed per tokenizer (GPT‑2/Neo/J vs. Llama‑2 vs. SILO; §5.1). Cross‑tokenizer use is not covered.

- Build resources and storage
  - Building trillion‑token suffix arrays needs large RAM during construction (e.g., 1 TiB for 1.4T tokens) and sizable disk (≈7 bytes/token; 35 TB for 5T tokens; §3; A.1–A.3). Although serving is CPU‑only and on‑disk, organizations still need significant storage.

- Domain/language coverage
  - Most analyses center on English web‑style corpora (The Pile subsets, RedPajama). Generalization to low‑resource languages or very different tokenization regimes is not studied.

- Diagnostic claims about model internals
  - The observed periodic fluctuations under greedy decoding (Figure 5) are hypothesized to relate to positional embeddings, but a causal link is not proven (§4.2).

- Decontamination nuances
  - Even with BFF (13‑gram, 80% threshold), defining and achieving perfect decontamination is hard (§4; §B). The paper follows accepted practice but acknowledges ambiguity (quotes vs. contamination).

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
