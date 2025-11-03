# The Common Pile v0.1: An 8TB Dataset of Public Domain and Openly Licensed Text

**ArXiv:** [2506.05209](https://arxiv.org/abs/2506.05209)

## 🎯 Pitch

The Common Pile v0.1 presents an unprecedented 8-terabyte dataset comprising only public domain and openly licensed texts from 30 diverse sources, plus a rigorously curated data mixture and openly released LLMs trained on 1 and 2 trillion tokens. This work proves, for the first time at scale, that state-of-the-art language models can achieve competitive performance with models trained on unlicensed web data—solving a major legal and ethical challenge and unlocking a reproducible, transparent path forward for open, auditable, and equitable LLM research and development.

---

## 1. Executive Summary (2–3 sentences)
This paper introduces the Common Pile v0.1, an 8‑terabyte corpus of public‑domain and openly licensed text collected from 30 sources, plus a filtered, reweighted training mixture (“Comma dataset”) and two 7B‑parameter language models (`Comma v0.1-1T` and `Comma v0.1-2T`) trained on 1T and 2T tokens. It demonstrates—empirically and at scale—that high‑performing LLMs can be trained without unlicensed web text, addressing legal and ethical concerns while maintaining competitive performance with similarly budgeted models trained on unlicensed data (Figures 3–4; Tables 10–11).

## 2. Context and Motivation
- Problem/gap:
  - Most LLMs are pretrained on massive web scrapes that include copyrighted material used without permission, raising legal risk (e.g., infringement claims) and ethical concerns about consent and compensation (Section 1, page 1). Datasets like The Pile have even faced takedowns (Section 1 citing [57]).
  - Prior attempts to build “clean” text corpora were either too small, too narrow, or lacked per‑document license fidelity to train performant LLMs (Section 2.2).
- Importance:
  - Practical: Reduces legal exposure and enables redistribution of training data, mixtures, and model checkpoints. This supports reproducibility, auditing, and safety research (Section 1; Section 2.1 Caveats).
  - Scientific: Enables controlled study of how open‑license data composition, filtering, and mixing affect learning dynamics, memorization, and domain coverage (Sections 4.1–4.3).
- Prior approaches and their shortcomings (Section 2.2):
  - Open License Corpus (OLC) [119]: smaller (0.85 TB) and includes sources with unclear license status (e.g., Hacker News).
  - Common Corpus [74, 91]: similar scale but less English focus and limited per‑document licensing fidelity; includes OpenAlex whose license metadata is known to be noisy (Appendix C; [80]).
  - KL3M [78]: largely government documents and excludes CC BY‑SA; smaller and less diverse (≈3 TB).
  - High‑quality unlicensed web mixtures (e.g., FineWeb [132]) remain legally encumbered and cannot be redistributed document‑by‑document.
- Positioning:
  - The Common Pile v0.1 is, to the authors’ knowledge, the largest all‑open‑license text set to date (Figure 1) and is paired with due‑diligence practices to minimize “license laundering” (Section 2.1) and a full training stack that stays within the open‑data boundary (custom tokenizer on open data; Section 4.4).

## 3. Technical Approach
The work has three pillars: (A) data collection with license due diligence, (B) preprocessing/quality control and mixture design, and (C) model training and evaluation.

A) Data collection and license due diligence (Sections 2–3; Appendix B)
- What “openly licensed” means:
  - Follows the Open Knowledge Foundation “Open Definition 2.1”: content may be freely used, modified, and shared for any purpose. Included licenses: `CC BY`, `CC BY-SA`, `CC0`, and software licenses approved by the Blue Oak Council (e.g., MIT, Apache 2.0). Excluded: `CC-NC` and `CC-ND` (Section 2).
- Avoiding “license laundering” (Section 2.1):
  - Only ingest sources where license information is supplied by the rightsholder or by official distribution channels (e.g., ArXiv S3, Wikimedia dumps).
  - Exclude sources with unreliable/ambiguous licensing metadata (e.g., OpenAlex, “YouTube Commons” datasets, and Hacker News on Kaggle).
- Sources (30 in total; Figure 1; Appendix B):
  - Scholarly/technical: `peS2o` (filtered to open licenses), PubMed Central (open‑license subset), ArXiv full texts (open‑license) and CC0 abstracts (Section 3; Appendix B.1; Appendix H for peS2o license and field stats).
  - Forums and code‑adjacent discussion: StackExchange (community dumps), GitHub issues/PRs/comments (only from repos with Blue Oak‑approved licenses), Ubuntu IRC (public domain) (Appendix B.2).
  - Government/legal: USGPO, Regulations.gov, USPTO patents (public domain), UK Hansard under the Open Parliament License, Caselaw Access Project and CourtListener (public domain) (Appendix B.3).
  - Books: Pre‑1929 U.S. books, Library of Congress public‑domain books, Project Gutenberg (public‑domain, English), Biodiversity Heritage Library (Appendix B.5).
  - Open Educational Resources (OER): DOAB, PressBooks, LibreTexts, OERCommons (Appendix B.6).
  - Wikis: Wikimedia projects and non‑Wikimedia MediaWiki sites via wikiteam, filtered for open licenses (Appendix B.7).
  - Code: Openly licensed subset of `The Stack v2` (license‑screened), Python Enhancement Proposals (public domain) (Appendix B.8).
  - Speech transcripts: Manually curated CC‑BY YouTube channels, transcribed with Whisper; >1.1M videos, >470k hours (Appendix B.9).
  - Open web text (`CCCC`): CC‑marked pages identified across 52 Common Crawl snapshots using a Creative Commons regex; then domain‑level manual verification of the top 1000 domains to keep only sites where all text is under a CC license; extraction and heavy filtering (Appendix B.10 and Appendix G for per‑snapshot stats).

B) Preprocessing, deduplication, and mixture design (Sections 4.1–4.2; Appendix J–L)
- Filtering pipeline (Section 4.1; Table 5; Table 6):
  - Language ID: FastText filter to retain primarily English.
  - Web quality (for `CCCC`): a DataComp‑LM‑adapted classifier with a very low threshold to remove noisy pages.
  - OCR error removal: unigram language‑model likelihood filter (Trillion Word Corpus) to drop texts with pervasive OCR artifacts.
  - Toxicity reduction: two FastText classifiers trained on the Jigsaw dataset; thresholding varies by source.
  - PII redaction: regex removal of emails, phone numbers, and IPs, replaced with placeholders `<EMAIL_ADDRESS>`, `<PHONE_NUMBER>`, `<IP_ADDRESS>`.
  - Source‑specific regex: boilerplate/license notices, page numbers, etc., removed.
  - Outcome: substantial cleaning—e.g., `Stack V2` shrinks from 4774.7 GB raw to 259.9 GB filtered; `CCCC` drops from 260 GB to 58 GB; `Wikiteam` from 437.5 GB to 13.7 GB (Table 6).
- Global fuzzy deduplication (Section 4.1):
  - Bloom‑filter‑based near‑duplicate removal (via Dolma [167]) across all sources; two documents are duplicates if they share >90% of their 20‑grams (a “20‑gram” is a sequence of 20 consecutive tokens).
- Code‑specific filtering (Section 4.1):
  - Start with RedPajama‑V1 code heuristics (line length, character ratios).
  - Keep specific languages (Python, C/C++, Java, JS/TS, Rust, Go, SQL, etc.) and apply language‑specific quality classifiers emphasizing educational and well‑documented code (following SmolLM2, but with a lower threshold to retain a larger high‑quality set).
- Mixture design (Section 4.2; Appendix K–L):
  - Goal: upweight higher‑quality sources (not necessarily the largest) and limit repetitions in a 1T‑token budget.
  - Method:
    - Train per‑source 1.7B models for 28B tokens (Section 4.3 setup) to get quality “signals.”
    - Heuristically assign mixing weights to repeat stronger sources up to six times by 1T tokens and give small but high‑quality sources enough weight to be seen six times (Table 7).
    - Also build a high‑quality “cool‑down” mixture for the final 37.7B tokens with a linear LR decay (Table 8).
    - Tried `MixMin` [178] to learn weights; it did not outperform the heuristic mix.
  - Terminology: The filtered and reweighted training set is called the `Comma dataset` (to distinguish it from the raw “Common Pile”).

C) Model training and evaluation (Section 4.4; Appendix O)
- Tokenizer (Section 4.4):
  - 64k‑vocab byte‑level BPE trained from scratch on a 600 GB sample of the Comma dataset (to stay within open‑data), using the Llama‑3.2 splitting regex and no Unicode normalization.
- Architecture and training (Section 4.4):
  - `Comma v0.1-1T`: Llama‑style 7B parameter decoder‑only Transformer (trained in Meta’s `lingua` library). Batch: 512×4096 tokens; AdamW with weight decay 0.2; cosine LR schedule with 2k warmup and 460k steps total in phase 1, followed by a “cool‑down” phase (linear decay to 0) over 18k steps on a high‑quality subset (Table 8). Final model averages 10 evenly spaced checkpoints from cool‑down.
  - `Comma v0.1-2T`: Same architecture; batch 2048×4096; LR doubled (2e‑3 peak), proportionally shortened steps per phase, similar cool‑down averaging. Data is the 1T mix repeated once; some sources repeat up to 16× (Section 4.4).
- Evaluation setup (Section 4.4):
  - Benchmarks: ARC (Challenge/Easy), MMLU (5‑shot), BoolQ, HellaSwag, OpenBookQA, CommonsenseQA, PIQA, SIQA; coding via HumanEval and MBPP (report pass@10).
  - Protocol: OLMES evaluation stack; zero‑shot except 5‑shot for MMLU (Section 4.4).

## 4. Key Insights and Innovations
- A legally redistributable, diverse, large‑scale corpus (Figure 1; Section 3):
  - Novelty: 8 TB spanning 30 sources across research, code, government, books, OERs, wikis, web, forums, and speech transcripts. Previous open‑license corpora are either much smaller or less diverse (Section 2.2).
  - Significance: Enables sharing the actual documents, mixture weights (Table 7), cool‑down mix (Table 8), and model checkpoints—crucial for reproducibility and audits.
- License due‑diligence workflow tailored for LLM pretraining (Section 2.1; Appendix C; Appendix B.10 for web):
  - Novelty: Conservative source selection (exclude noisy license metadata), channel‑level curation for CC‑BY YouTube, and manual domain‑level verification for CC web pages. The paper also clarifies pitfalls such as compilation licenses (ODC‑By) not granting rights to individual documents (Section 2.1 “Use of collection licenses”).
  - Significance: Mitigates license laundering and supports redistribution that others can trust.
- A quality‑driven mixture design validated by controlled per‑source models (Sections 4.2–4.3; Table 7):
  - Novelty: Train small models on each source to gauge quality and set repetition caps/mix weights accordingly (instead of size‑proportional mixing). Add a “cool‑down” phase on a curated high‑quality subset (Table 8).
  - Significance: Demonstrably lifts downstream results for the same token budget (Figure 2; Table 9) and narrows the gap to state‑of‑the‑art unlicensed datasets.
- Full open‑data training pipeline, including tokenizer (Section 4.4):
  - Incremental but important: Avoids subtle dependence on unlicensed data even in tokenizer training, keeping the entire artifact “clean” for legal reuse.

## 5. Experimental Analysis
Evaluation methodology
- Controlled dataset quality test (Section 4.3; Figure 2; Table 9):
  - Setup: Identical 1.7B Llama‑style models trained 28B tokens on each dataset; GPT‑2 tokenizer; early‑signal tasks—ARC, MMLU, HellaSwag, OpenBookQA, CommonsenseQA, PIQA, SIQA; no Winogrande because it appears in their supervised sources (DPI) to avoid contamination (Section 4.3).
  - Baselines: OLC, Common Corpus, KL3M (open‑license datasets), The Pile (mixed), OSCAR and FineWeb (unlicensed web).
  - Results (Table 9):
    - Average accuracy: `Comma` 40.8 vs OLC 37.3, Common Corpus 37.6, KL3M 36.2, The Pile 39.6, OSCAR 40.9, FineWeb 43.7.
    - Per‑task highlights:
      - Comma leads open‑license peers on all tasks and beats The Pile on five of seven. It excels on knowledge tasks: MMLU 29.5, ARC 38.0, whereas it trails on commonsense web tasks like HellaSwag (39.9 vs FineWeb 48.2 and The Pile 35.8).
    - Ablation: Removing supervised DPI data barely changes the average (40.0 vs 40.8), indicating the gains are not artifactually driven by task‑like data (Table 9).
- Full‑scale model validation (Section 4.4; Figures 3–4; Tables 10–11):
  - `Comma v0.1-1T` (7B, 1T tokens) vs compute‑matched baselines trained on unlicensed data:
    - Benchmarks (Table 10):
      - Knowledge: ARC‑C 52.8 (best among compute‑matched), MMLU 42.4 (beats LLaMA‑7B 34.8; StableLM 45.2 is higher on MMLU but lower elsewhere), BoolQ 75.7 (near top).
      - Commonsense: HellaSwag 62.6 (below LLaMA 76.2 and MPT 77.6).
      - Coding: HumanEval 36.5, MBPP 35.5—substantially above LLaMA‑7B (19.9, 27.9) and RPJ‑INCITE‑7B (11.1, 15.9).
    - Takeaway: Best‑in‑class on several knowledge tasks and markedly strong coding for a 1T‑token budget; weaker on HellaSwag/PIQA (Figure 3).
  - `Comma v0.1-2T` (7B, 2T tokens) vs compute‑matched baselines:
    - Benchmarks (Table 11):
      - Knowledge: MMLU 49.8 (beats Llama‑2‑7B at 45.8 and OLMo‑Twin at 28.2), ARC‑E 71.8 (top among budget‑matched).
      - Commonsense: HellaSwag 64.4 (below Llama‑2 76.2; DeepSeekLLM 74.1).
      - Coding: HumanEval 44.2, MBPP 41.5—competitive with DeepSeekLLM (43.1, 43.8).
    - Caveat (Section 4.4): The 2T run repeats some sources up to 16×; the authors expect a better 2T‑specific mix could perform higher.
- Robustness checks (Appendix O):
  - Alternative hyperparameters (larger batch; three‑stage curriculum) yield similar average scores to `Comma v0.1-1T`, with slightly better coding (Table 12). This suggests the main conclusions are not brittle.

Do the experiments support the claims?
- Yes, with nuance:
  - The controlled 28B‑token experiment shows the Comma dataset is the highest‑quality openly licensed corpus among tested options and is competitive with unlicensed baselines (Figure 2; Table 9).
  - The 7B models trained to 1T–2T tokens are often competitive with compute‑matched unlicensed‑data models on broad knowledge and coding tasks (Figures 3–4; Tables 10–11).
  - The main weakness (commonsense benchmarks like HellaSwag, PIQA) aligns with domain coverage analysis that such tasks correlate with data like blogs, hobbies, and sports—underrepresented in open‑license sources (Section 4.3 citing [188]).

## 6. Limitations and Trade-offs
- Residual licensing risk (Section 2.1 “Caveats”):
  - License laundering is hard to detect exhaustively; rightsholders may change license terms; public‑domain/open documents may contain quoted in‑copyright snippets. Attribution at generation time remains an open research problem (citations [129, 28]).
- Domain coverage gaps:
  - Openly licensed data underrepresents certain web genres (personal blogs, tutorials, sports, hobbies) that strongly influence commonsense benchmarks like HellaSwag and PIQA (Section 4.3).
- Repetition and data efficiency at larger budgets:
  - The 2T run repeats some sources up to 16× (Section 4.4), which may give diminishing returns and risk memorization (references [94, 121]).
- Language coverage:
  - Focus is “primarily English” (Section 4.1); non‑English text is filtered out by default. This limits multilingual generalization.
- Filtering trade‑offs:
  - Aggressive filtering (toxicity, OCR likelihood, boilerplate regex) improves cleanliness but can drop non‑standard yet valuable texts; thresholds vary by source (Table 5).
- Computing cost:
  - Training 7B models to 1T–2T tokens remains expensive; however, the contribution is to show such training can be done within open‑data constraints rather than to reduce compute.

## 7. Implications and Future Directions
- How this changes the landscape:
  - Provides a redistributable, auditable foundation for LLM pretraining, mixing, and evaluation. Researchers and practitioners can inspect, reweight, and legally share the exact documents and mixtures (Tables 7–8), enabling rigorous data‑centric science (Sections 4.2–4.4).
- Follow‑up research enabled/suggested:
  - 2T‑specific and larger‑budget mixtures with less extreme repetition; automatic mixture optimization that beats heuristics (MixMin underperformed here; Section 4.2).
  - Expanding underrepresented domains (e.g., curated open‑license blogs/how‑to sites) to improve commonsense reasoning benchmarks (Section 4.3).
  - Multilingual expansion with strong license provenance; the growth trend of open data (Figure 6) suggests feasibility.
  - Better license‑signal detection and validation tools (Appendix C.1) to reduce manual verification burden while avoiding false positives.
  - Stronger PII and safety filters; improved techniques for attribution/traceability of generations to sources (citations [129, 28]).
  - Studying memorization and long‑tail learning dynamics on a fully shareable corpus (e.g., leveraging [17, 22, 84]).
- Practical applications:
  - Organizations needing low‑risk, transparent training data for internal or commercial LLMs.
  - Education, government, and legal domains where provenance and public accessibility of sources matter.
  - Code models and scientific assistants—the mixture is particularly strong in open‑source code and scholarly text, reflected in superior coding (HumanEval/MBPP) and knowledge‑heavy benchmarks (MMLU, ARC in Tables 10–11).

> Bottom line: With the Common Pile v0.1 (Figure 1) and the validated Comma mixtures (Tables 7–8), this work shows that performant LLMs can be trained on fully open data, narrowing the performance gap with unlicensed corpora while enabling transparent, redistributable, and legally safer research and deployment.
