# EmbeddingGemma: Powerful and Lightweight Text Representations

**ArXiv:** [2509.20354](https://arxiv.org/abs/2509.20354)
**Authors:** Henrique Schechter Vera, Sahil Dua, Biao Zhang, Daniel Salz, Ryan Mullins, Sindhu Raghuram Panyam, Sara Smoot, Iftekhar Naim, Joe Zou, Feiyang Chen, Daniel Cer, Alice Lisak, Min Choi, Lucas Gonzalez, Omar Sanseviero, Glenn Cameron, Ian Ballantyne, Kat Black, Kaifeng Chen, Weiyi Wang, Zhe Li, Gus Martins, Jinhyuk Lee, Mark Sherwood, Juyeong Ji, Renjie Wu, Jingxiao Zheng, Jyotinder Singh, Abheesht Sharma, Divya Sreepat, Aashi Jain, Adham Elarabawy, AJ Co, Andreas Doumanoglou, Babak Samari, Ben Hora, Brian Potetz, Dahun Kim, Enrique Alfonseca, Fedor Moiseev, Feng Han, Frank Palma Gomez, Gustavo Hernández Ábrego, Hesen Zhang, Hui Hui, Jay Han, Karan Gill, Ke Chen, Koert Chen, Madhuri Shanbhogue, Michael Boratko, Paul Suganthan, Sai Meher Karthik Duddu, Sandeep Mariserla, Setareh Ariafar, Shanfeng Zhang, Shijie Zhang, Simon Baumgartner, Sonam Goenka, Steve Qiu, Tanmaya Dabral, Trevor Walker, Vikram Rao, Waleed Khawaja, Wenlei Zhou, Xiaoqi Ren, Ye Xia, Yichang Chen, Yi‑Ting Chen, Zhe Dong, Zhongli Ding, Francesco Visin, Gaël Liu, Jiageng Zhang, Kathleen Kenealy, Michelle Casbon, Ravin Kumar, Thomas Mesnard, Zach Gleicher, Cormac Brick, Olivier Lacombe, Adam Roberts, Yunhsuan Sung, Raphael Hoffmann, Tris Warkentin, Armand Joulin, Tom Duerig, Mojtaba Seyedhosseini
**Institutions:** Google DeepMind

Reasoning upfront
- I read the paper holistically and then rebuilt the model’s pipeline in plain language: how the architecture is created, how training objectives interact, and why each choice matters. I cross-referenced claims with the paper’s specific sections, equations, and tables (e.g., Section 2.1–2.3, Equations (1)–(5), Tables 1–9).
- I also looked for what’s truly novel vs. standard practice, checked how ablations support the narrative, and considered where evidence is strong or incomplete (e.g., quantization robustness, pooling choices, initialization strategy).
- Finally, I thought through practical implications (latency, on-device use), hidden assumptions (distillation dependence, data scale), and realistic future directions.

## 1. Executive Summary
EmbeddingGemma is a 308M-parameter, encoder-only text embedding model that hits state-of-the-art performance for sub-500M models across multilingual, English, and code tasks on MTEB, while staying fast and cheap to run. The core trick is a training recipe that combines encoder-decoder initialization from Gemma 3, geometric embedding distillation from a stronger teacher model, a spread-out regularizer for robustness, and “model souping” across diverse finetuning mixtures—yielding competitive quality even after quantization or embedding truncation (Table 1, Tables 5–8).

## 2. Context and Motivation
- The problem it addresses
  - Many top-performing embedding models have several billion parameters (e.g., NV-Embed, GritLM-7B, E5-Mistral; see Introduction), making them too heavy for low-latency, high-throughput, or on-device workloads.
  - There’s a gap for lightweight (<500M) models that still perform well across diverse tasks and languages (MTEB spans 250+ languages, 20 domains, and 9+ task types; Section 1 and Section 4.1).

- Why this matters
  - Real-world: On-device retrieval, private search, and offline applications need small, efficient models with strong quality.
  - Theoretical: Can smaller encoders rival big ones by transferring knowledge (“distillation”) and better representation learning?

- Prior approaches and shortcomings
  - Larger LLMs have helped via synthetic data, hard negative mining, or initializing embedding models (Introduction).
  - But small models often lag in generalization, robustness (quantization), and multilingual or code domains.

- How this work positions itself
  - EmbeddingGemma proposes a compact model that consistently beats other sub-500M models and even rivals ~600M models (Table 5, Tables 6–8).
  - Core method: use an encoder-decoder adapted from Gemma 3 for initialization (strong input understanding), then train with contrastive + geometric distillation + spread-out regularization + Matryoshka representation learning (MRL), and finish with model souping (Sections 2.1–2.3).

## 3. Technical Approach
Step-by-step pipeline:

1) Architecture and initialization (Section 2.1)
- Start with `Gemma 3` (decoder-only LLM) and adapt it into an `encoder-decoder` using the UL2 objective (a training setup that mixes multiple denoising tasks to improve versatility; Tay et al., 2023).
- Initialize EmbeddingGemma from the encoder part of this encoder-decoder (so the encoder already “knows” how to represent inputs contextually).
- Encoder-only transformer:
  - `n = 24` layers, model dimension `d_M = 768` (Section 2.1).
  - Pooling: `mean pooling` over token representations.
  - Two linear projections: first to intermediate dim `d_U = 3072`, then to final embedding `d = 768` (randomly initialized; trained later).
  - Why this design: encoder-decoder encoders specialize in input understanding and use bidirectional attention, which yields stronger contextual features than decoder-only weights (supported by Table 2).

2) Embedding mapping (Equation (1), Section 2.2)
- Each training example has `query q_i`, `positive passage p_i^+`, optional `hard negative p_i^-`, and task strings `t_q`, `t_p` that describe the task (e.g., retrieval prompts; Section 2.2 Input).
- Embedding formula:
  - Tokenize “task prompt + content” → pass through transformer → mean pool → project to `d_U` → project to `d`.
  - Notation (Equation (1)):
    - `q_i = f(g(P(M_n(t_q ⊕ q_i))))`, `p_i^± = f(g(P(M_n(t_p ⊕ p_i^±))))`.

3) Training objectives (Section 2.2)
- Contrastive loss (`LC`, Equation (2)):
  - Standard InfoNCE with in-batch negatives. Cosine similarity `sim(x, y)` is used, with temperature `τ`.
  - Hard negatives are upweighted with `w_i = exp(α * sg(sim(q_i, p_i^-)))`, where `sg` is the `stop-gradient`. Intuition: if a negative is too similar to the query, push it apart harder. They set `α = 5.0`.
  - Duplicate masking `1_TN(i,j)` prevents false negatives when queries or positives repeat (Equation (3)).
- Spread-out regularizer (`LS`, Equation (4)):
  - Based on the Global Orthogonal Regularizer (GOR; Zhang et al., 2017). It minimizes the squared inner product between different embeddings in the batch (for queries and for positives), encouraging embeddings to be “uniformly spread” on the unit sphere.
  - Why: this improves expressiveness, quantization robustness, and vector-database efficiency (Section 2.2).
- Embedding matching (distillation) (`LD`, Equation (5)):
  - Geometric distillation aligns EmbeddingGemma’s embeddings to a teacher model (`Gemini Embedding`) directly in embedding space (Kim et al., 2023). This goes beyond using only relevance scores.
  - Crucially, they match embeddings for queries, positives, AND hard negatives—a novel expansion that strengthens discrimination (Section 2.2).

4) Matryoshka Representation Learning (MRL) (Section 2.2)
- MRL splits losses across overlapping sub-dimensions so that smaller “slices” of the embedding remain useful.
- Result: EmbeddingGemma natively supports `768`, `512`, `256`, and `128` dims with reasonable quality (Tables 6–8 show performance at different dims).

5) Training recipe (Section 2.3)
- Data scale: ~2.1T tokens total ( encoder-decoder adaptation + pre-finetuning + finetuning).
- Encoder-decoder training:
  - Adapt Gemma 3 to encoder-decoder, train with UL2 on Gemma 3’s pretraining data (multilingual; 100+ languages).
- Pre-finetuning:
  - Train on massive, noisy unsupervised pairs (query, target) without hard negatives; large batch size for stable gradients and richer in-batch negatives.
  - Diverse task types (QA, sentence similarity, code retrieval, web search) and multiple languages/programming languages; includes a large title–body web corpus similar to prior work (Section 2.3).
- Finetuning:
  - Smaller, higher-quality task-specific mix with hard negatives.
  - Use three groups of tasks to cover task diversity, language diversity, coding capability (as in Gecko and Gemini Embedding).
  - Mixture selection: seed from prior grid search → Bayesian optimization + 10 random Dirichlet samples → multiple specialized mixtures (“experts”).
- Model souping (Section 2.3):
  - Average weights across checkpoints trained on different mixtures to improve generalization.
  - They specifically soup across mixtures (not just across hyperparameters), which creates complementary experts (Table 4).
- Quantization-aware training (Section 2.3; Table 1):
  - Provide int4 per-block, int8 per-block, and mixed per-channel variants. Train with QAT to minimize quality loss after quantization.

Design choices summary (with rationale):
- `Encoder-decoder initialization` over decoder-only: yields stronger input representations via bidirectional attention and encoder specialization (Table 2 shows consistent gains).
- `Mean pooling` over attention pooling: simpler, parameter-free pooling avoids overfitting and performed better in ablations (Table 3).
- `Spread-out regularizer`: makes embeddings robust to compression and better for ANN search (Equation (4), Section 2.2).
- `Embedding matching for queries, positives, negatives`: aligns global geometry more faithfully than ranking-only signals (Equation (5)).
- `MRL` for flexible embedding sizes without retraining.
- `Model souping across mixtures`: leverages specialization diversity (Table 4).

Helpful analogies:
- Spread-out regularization is like inflating a balloon: points repel each other to uniformly cover the sphere, so no region is overcrowded. This helps both quantization (less redundancy) and approximate nearest neighbor search (clearer geometric structure).
- Model souping is blending different “experts”—each trained on a distinct mixture—so the final model inherits strengths across domains, not just hyperparameter optimizations.

Definitions (selective):
- `UL2`: A training objective mixing multiple denoising and span-corruption patterns to improve generalization across tasks.
- `MRL (Matryoshka)`: A way to train embeddings so that smaller prefix dimensions are high-quality, enabling easy truncation (Kusupati et al., 2022).
- `Spread-out regularizer (GOR)`: A constraint that encourages embeddings to be orthogonal-ish on average, producing uniform coverage on a unit sphere (Zhang et al., 2017).
- `Model souping`: Averaging the weights of multiple finetuned checkpoints to improve accuracy without extra inference cost (Wortsman et al., 2022).
- `Hard negatives`: Non-relevant passages that are very similar to the query; the model must learn to separate them.
- `Stop-gradient`: Prevents gradients from flowing through certain operations; used here so hardness weights don’t affect parameter gradients directly.
- `Per-block vs per-channel quantization`: Quantize weights in contiguous blocks vs. per output channel; per-channel often preserves quality better but is more complex.

## 4. Key Insights and Innovations
- Encoder-decoder initialization is a big win for small encoders
  - What’s new: Instead of starting from decoder-only LLM weights, they adapt Gemma 3 to encoder-decoder and initialize from the encoder.
  - Evidence: Table 2 shows `Encoder-Decoder` init outperforms `Decoder-only` across MTEB(Multi v2) task-type means (53.6 vs 52.6) and task means (60.4 vs 59.7). This is a consistent, cross-task gain.
  - Why it matters: Bidirectional attention + encoder specialization improves contextual understanding, especially useful for embeddings.

- Geometric distillation that matches embeddings, not just rankings—and includes hard negatives
  - What’s different: They align the student’s `query`, `positive`, and `hard negative` embeddings with a strong teacher (`Gemini Embedding`)—beyond prior distillation that used only relevance scores or only queries/positives (Section 2.2; Equation (5)).
  - Why it matters: It transfers global geometry more faithfully, making the small model behave like the big one in embedding space, which boosts retrieval and similarity tasks.

- Spread-out regularization for expressiveness and quantization robustness
  - Novelty: Use the second-moment component of GOR across queries and positives to push embeddings toward uniform sphere patterns (Equation (4)).
  - Why it matters: Improves ANN search efficiency and quantization tolerance. Table 1 shows quality barely drops under int4/int8 quantization.

- Model souping across diversified mixtures, not just hyperparameters
  - What’s new: Soup checkpoints trained on different finetuning mixtures discovered via Bayesian optimization—yielding “experts” that complement each other (Section 2.3).
  - Evidence: Table 4 shows the `Souped` model (Mean(Task)=61.2, Mean(Type)=54.3) outperforms any single mixture across all task types.
  - Why it matters: Generalization improves without extra inference cost.

- Mean pooling beats attention pooling in this embedding setting
  - Counterintuitive finding: Despite adding learnable parameters, attention pooling underperformed mean pooling (Table 3).
  - Significance: Suggests simple aggregation can be more robust for encoder-only embedding tasks, consistent with recent findings for classification/regression (Suganthan et al., 2025).

Incremental vs. fundamental:
- Incremental: MRL, QAT, in-batch contrastive loss—these are established but smartly combined.
- Fundamental upgrade (for small models): The joint recipe (encoder-decoder init + geometric distillation + spread-out regularizer + souping across mixtures) materially shifts the quality–efficiency frontier for sub-500M models.

## 5. Experimental Analysis
- Evaluation setup (Section 4)
  - Benchmarks:
    - `MTEB(Multilingual, v2)`: 131 tasks, 250+ languages, 20 domains, 9 task types.
    - `MTEB(English, v2)`: 41 tasks.
    - `MTEB(Code)`: 12 code retrieval tasks.
    - `XOR-Retrieve`: Cross-lingual retrieval (queries in 7 languages → English passages).
    - `XTREME-UP`: Cross-lingual retrieval for 20 underrepresented Indo-European languages.
  - Metrics:
    - Aggregate: `Mean(Task)`, `Mean(Type)`, `Borda count` rank (leaderboards).
    - Retrieval: `Recall@5kt` for XOR-Retrieve; `MRR@10` for XTREME-UP (Table 5).
  - Setup details:
    - Half-precision (`bf16`) weights in core evaluations (Section 4.2).
    - Prompt instructions per model card; typical context length 512, extended to 1024/2048 for long-context tasks (Section 4.2).
    - Comparators: Major open models <1B params and commercial APIs; for <500M analyses they exclude models trained on >25% of MTEB data to limit overfitting (Figure 1, Table 6–8).

- Main quantitative results
  - Overall (Table 5):
    - `MTEB(Multilingual, v2)`: EmbeddingGemma Mean(Task)=61.15, Mean(Type)=54.31; competitive even vs larger ~600M models and APIs; ranks #1 among <500M models across aggregates in the paper’s comparisons.
    - `MTEB(English, v2)`: Mean(Task)=69.67, Mean(Type)=65.11; tops sub-500M peer set in Table 7 and is competitive under 1B.
    - `MTEB(Code)`: Mean=68.14; strong on AppsRetrieval (84.39) and CosQA (43.60) (Table 11).
    - `XOR-Retrieve`: Recall@5kt=84.14; strong, second only to Gemini Embedding (90.42).
    - `XTREME-UP`: MRR@10=47.72; remarkably higher than many big open models (e.g., 7.11B Linq-Embed-Mistral=24.6) and certain APIs (e.g., text-embedding-3-large=18.8) (Table 9).
  - Sub-500M MTEB leaders:
    - Multilingual (Table 6): EmbeddingGemma (768d) Mean(Task)=61.2, Mean(Type)=54.3; beats `gte-multilingual-base` (58.2/51.4) and `KaLM mini` baselines (~57–56 / ~50).
    - English (Table 7): EmbeddingGemma leads Mean(Task)=69.7, Mean(Type)=65.1; large margins in classification (+8.5 vs next best), clustering (+7.8), summarization (+4.4).
    - Code (Table 8): Best across aggregates including Mean -COIR; standout improvements on AppsRetrieval (+37.6) and CosQA (+10.0) over second-best.
  - Dimensionality truncation via MRL (Tables 6–8):
    - Multilingual (Mean Task): 768d=61.2; 512d=60.7; 256d=59.7; 128d=58.2.
    - English (Mean Task): 768d=69.7; 128d=66.7.
    - Code (Mean All): 768d=68.8; 128d=63.0.
    - Insight: Graceful degradation—still top-tier among sub-500M even at 128d in multilingual/English.
  - Quantization-aware training (Table 1):
    - Multilingual Mean(Task): bf16=61.15 → int8 per-block=60.93 → int4 per-block=60.62.
    - English Mean(Task): bf16=69.67 → int8=69.49 → int4=69.31.
    - Code Mean(Task): bf16=68.76 → int4=67.99.
    - Insight: Small drops (<1 point), confirming quantization robustness.

- Ablations and robustness checks
  - Initialization (Table 2): Encoder-decoder > Decoder-only > Random; confirms the init choice matters across all task types.
  - Pooling (Table 3): Mean pooling best; attention pooling underperforms despite extra parameters.
  - Model souping (Table 4): Souped model beats each individual mixture across all task types, showing complementary specialization.
  - Quantization: They compare per-block and per-channel QAT variants (Table 1), both holding quality well.

- Do experiments support claims?
  - Yes, broadly. The model consistently tops sub-500M leaderboards across multilingual, English, and code aggregates and remains competitive under 1B (Table 5). Ablations show each design choice contributes.
  - Cross-lingual strength is particularly convincing on XTREME-UP (Table 9), where small models usually struggle; EmbeddingGemma’s MRR is unusually high.
  - Caveat: While they exclude overfitted comparators (>25% MTEB data), it would be ideal to know exactly how much of MTEB their own mixtures include (see “Limitations”).

- Notable failure cases or weak spots
  - Some instruction retrieval tasks have very low or negative scores for many models (e.g., Table 6 “Inst. Retrieval” values); EmbeddingGemma improves to 5.6 (souped) but this task class remains challenging.
  - Distillation target availability: performance depends on access to a strong teacher (Gemini Embedding), which may limit reproducibility for others.

## 6. Limitations and Trade-offs
- Dependence on a strong teacher
  - The geometric distillation relies on `Gemini Embedding`. If you don’t have a teacher of similar strength, reproducing the gains may be hard.

- Data and compute scale
  - Training saw ~2.1T tokens across stages (Section 2.3). The final model is small, but the recipe is compute-intensive—this isn’t a “train-on-a-laptop” setup.

- Transparency of training mixtures vs. evaluation overlap
  - They carefully exclude other models trained on >25% of MTEB, but it’s not fully detailed what portion EmbeddingGemma itself overlaps with (potential data leakage risks). The paper mentions mixtures derived from Gecko and Gemini Embedding datasets (Section 2.3), which likely intersect with MTEB tasks.

- Narrow focus on text-only
  - Future work mentions multimodal, but current evidence is strictly text. If your application needs text–image or audio retrieval now, this model won’t cover it yet.

- Limits of quantization variants
  - They provide int4/int8 and mixed precision, which is great, but practitioners might want even more aggressive compression (2-bit, extreme sparsity). Not covered here.

- Pooling generality
  - Mean pooling wins here, but attention pooling could shine in specialized domains or with different pretraining. The ablation (Table 3) is convincing for MTEB-style tasks, not necessarily universal.

## 7. Implications and Future Directions
- Field impact
  - EmbeddingGemma resets expectations for small encoders: with the right initialization and distillation, ~300M-parameter models can rival ~600M and sometimes sidestep billion-scale models—especially in multilingual and code domains (Tables 5–8, 9).
  - It encourages a design shift: start with specialized encoder-decoder encoders, rely on geometric distillation, add spread-out regularization, and finish with souping across diverse mixtures.

- Practical applications
  - On-device semantic search, clustering, and retrieval where latency and privacy are critical (the model’s robustness under quantization/truncation makes it a fit; Table 1 and MRL results in Tables 6–8).
  - Cross-lingual retrieval in low-resource languages (XTREME-UP results; Table 9) for global apps (customer support, multilingual QA, knowledge bases).
  - Code search and developer tooling (AppsRetrieval, CosQA gains in Table 8/Table 11).

- Follow-up research
  - Multimodal embeddings at small scale: extend this recipe to image/audio/video with encoder-decoder init from multimodal LLMs (Section 5).
  - More transparent mixture design: release exact mixture proportions and overlap analyses with MTEB to set a stronger norm for fair comparisons.
  - Advanced compression: explore 2–3 bit quantization, structured sparsity, and on-device graph-friendly formats (e.g., HNSW calibrations with spread-out embeddings).
  - Distillation without proprietary teachers: replicate geometric matching using open teachers, or self-distillation via large checkpoints within the same family.
  - Pooling exploration: test mean pooling vs. attention pooling with longer inputs, domain-specific prompts, and adaptive masking.

Block-cited specifics for quick reference
- Architecture: “In EmbeddingGemma, we use `n = 24`, `d_M = 768`, `d_U = 3072`, `d = 768`.” (Section 2.1)
- Training objective equations: see Equations (1)–(5), Section 2.2.
- Quantization results: “bf16 Mean(Task) 61.15 vs int4 60.62 on MTEB(Multi, v2)” (Table 1).
- Initialization ablation: “Encoder-Decoder Mean(Type)=53.6 vs Decoder-only 52.6” (Table 2).
- Pooling ablation: “Mean pooling Mean(Task)=60.4 vs Attention=60.2” (Table 3).
- Souping: “Souped Mean(Task)=61.2 beats Mix 1–3” (Table 4).
- Overall comparison: “EmbeddingGemma beats sub-500M models across MTEB multilingual/English/code aggregates” (Tables 6–8).
- Cross-lingual: “XTREME-UP MRR@10=47.72; outperforms many big open and API models listed” (Table 5, expanded in Table 9).
- Dim truncation: “Multilingual Mean(Task) stays high down to 128d (58.2)” (Table 6).
