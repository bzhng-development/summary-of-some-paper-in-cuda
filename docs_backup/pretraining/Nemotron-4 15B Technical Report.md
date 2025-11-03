# Nemotron-4 15B Technical Report

**ArXiv:** [2402.16819](https://arxiv.org/abs/2402.16819)
**Authors:** Jupinder Parmar, Shrimai Prabhumoye, Joseph Jennings, Mostofa Patwary, Sandeep Subramanian, Dan Su, Chen Zhu, Deepak Narayanan, Aastha Jhunjhunwala, Ayush Dattagupta, Vibhu Jawa, Jiwei Liu, Ameya Mahabaleshwarkar, Osvald Nitski, Annika Brundyn, James Maki, Miguel Martinez, Jiaxuan You, John Kamalu, Patrick LeGresley, Denys Fridman, Jared Casper, Ashwath Aithal, Oleksii Kuchaiev, Mohammad Shoeybi, Jonathan Cohen, Bryan Catanzaro
**Institutions:** NVIDIA (presumed)

## 🎯 Pitch

Nemotron-4 15B revolutionizes multilingual language processing by leveraging a massive 8 trillion token dataset to achieve exceptional performance while being deployable on a single NVIDIA A100/H100 GPU. This innovative approach challenges the traditional model-size paradigm, proving that more data per parameter can outperform larger models, making it a game-changer for cost-efficient, real-time multilingual applications and research scalability.

---

## 1. Executive Summary
Nemotron-4 15B is a 15-billion-parameter multilingual language model trained on an unusually large 8 trillion tokens, designed to run on a single NVIDIA A100/H100 GPU while delivering state-of-the-art multilingual performance in its size class. It combines a standard decoder-only Transformer with targeted data curation (70% English, 15% non‑English, 15% code), a very large tokenizer (`256k` vocabulary), grouped-query attention for efficient inference, and a brief “continued training” phase to sharpen performance on weak areas (Sections 2–3, Table 1, Figures 2–4).

## 2. Context and Motivation
- Problem gap
  - How to get strong multilingual and coding capabilities in a model small enough to deploy on a single high-end GPU, without sacrificing broad general-purpose performance. The paper targets the compute/latency sweet spot: fewer parameters but far more training tokens (Abstract; Section 1).
- Why it matters
  - Practical deployment: a 15B model reduces inference cost/latency versus 30–70B models, enabling broader on‑prem and real‑time use.
  - Research significance: it tests the “data‑over‑parameters” regime suggested by the Chinchilla scaling law—i.e., for a fixed compute budget, train smaller models on more data to reach higher accuracy.
- Prior approaches and limits
  - Larger parameter counts with less data (e.g., historical trend before Chinchilla) can underperform compared to smaller models trained on more tokens (Section 1 cites the Chinchilla result: a 65B model on 1.4T tokens outperforms a 280B model on 300B tokens).
  - Specialized multilingual models (e.g., `mGPT 13B`, `XGLM 7.5B`) improve non‑English, but may compromise generality or require larger sizes; many general-purpose open models at similar scale have weaker non‑English and low-resource code performance (Section 3.4 and Tables 7–10).
- Positioning
  - A general-purpose 15B model trained on 8T tokens with an explicit multilingual and multi-language code mix. It aims to outperform similarly sized open models across English, code, and multilingual while remaining deployable on a single A100/H100 (Abstract; Section 1; Table 1).

Definitions of less-common terms used throughout:
- `Chinchilla scaling law`: Empirical rule that, for a fixed training compute budget (same total FLOPs), accuracy increases by allocating more compute to data tokens rather than to parameters alone.
- `IsoFLOP`: Two training runs consume roughly the same number of floating-point operations; used to compare compute-optimal choices.
- `Grouped Query Attention (GQA)`: An attention variant where many query heads share a smaller number of key/value heads, reducing memory and speeding inference with minimal accuracy loss.
- `Continued training`: A short extra training phase after pretraining that changes data sampling weights and the learning-rate schedule to emphasize high-quality or weak areas (Section “Continued Training”).

## 3. Technical Approach
Nemotron-4 uses a standard decoder-only Transformer, but several implementation choices align it with the data-rich, efficiency-first design goals.

- Model architecture (Table 1; Section 2)
  - Layers and sizes: `32` Transformer layers, hidden size `6144`, `48` attention heads, `8` KV heads (due to `GQA`), sequence length `4096`, vocabulary size `256,000`.
  - Parameter split: `3.2B` embedding parameters + `12.5B` non-embedding parameters (Section 2).
  - Components and defaults:
    - `RoPE` rotary position embeddings (stabilizes long-context attention rotations).
    - Tokenization: SentencePiece BPE, `256k` vocab, preserves whitespace, splits numbers into digits, and has byte-level backoff for unseen character sequences (Section 2).
    - MLP activation: `squared ReLU`. No bias terms, dropout set to zero, and input/output embeddings are untied (Section 2).
    - `GQA` with 8 KV heads to reduce inference memory/time while retaining 48 query heads (Section 2).
  - Why these choices?
    - Large vocabulary (`256k`) and digit splitting improve coverage of low-resource languages and scripts (Section 2).
    - `GQA` decreases KV cache size at inference, enabling a 15B model to better fit on a single A100/H100 GPU with 4096 context length (Section 2).
    - Untied embeddings and squared ReLU are common recent choices found to improve performance per parameter in large decoders.

- Data curation and composition (Section 2; Figures 2–4)
  - Scale and mix: `8T` tokens total. Split: `70%` English natural language, `15%` multilingual (53 languages), `15%` code (43 languages).
  - English sources (Figure 2): web crawl dominates (66.3%), plus books (4.6%), academic papers (5.6%), encyclopedias (4.9%), news (2.7%), legal (0.5%), finance (0.5%), other (13.1%).
  - Code languages (Figure 3): diverse distribution across 43 languages; careful sampling to strengthen low-resource languages (e.g., Scala, Julia, R are included and analyzed later).
  - Multilingual languages (Figure 4): 53 non‑English languages with explicit distribution control; tokenizer training up-samples non‑English data to ensure better subword coverage (Section 2).
  - Cleaning: document-level exact and near-deduplication; LLM-based quality filtering plus heuristics inspired by prior work (Section 2 cites Wenzek et al., Rae et al., Raffel et al.).

- Training setup (Pre-training; Table 2)
  - Hardware: 384 DGX H100 nodes, each with 8 H100 80GB GPUs connected via NVLink/NVSwitch; inter-node 400 Gbps InfiniBand (Section “Pre-training”).
  - Parallelism: 8-way tensor parallelism + data parallelism with distributed optimizer; data parallel increased from 96→384 during batch ramp-up (Table 2).
  - Throughput/efficiency: iteration times ~0.57–0.64 s; Model FLOPs Utilization (`MFU`) ~30–34% (Table 2). Training completes in ~13 calendar days.
    - `MFU`: fraction of theoretical peak FLOP/s achieved during training; helps quantify hardware efficiency.
  - Quote for timing:
    > “Training was completed in approximately 13 calendar days.” (Table 2; Pre-training)

- Continued training (Section “Continued Training”)
  - After the full 8T pass, a brief extra phase keeps the same loss objective but changes:
    - Data distributions: (1) emphasizes higher-quality sources already seen; (2) adds a small set of “benchmark‑style alignment” examples and upweights areas where the model was weak during pretraining (e.g., specific tasks or languages).
    - Learning-rate schedule: uses a decay schedule with a “steeper slope of decay than magnitude of learning rate” to gently transition and learn the reweighted areas.
  - Rationale: align the model to evaluation styles and improve weak domains without overhauling the training recipe.

How it all works together:
- A moderate-sized model (15B) is paired with massive, carefully rebalanced data and a tokenizer that covers rare scripts. GQA reduces inference memory, the long `4096` context supports realistic tasks, and the short continued-training phase sharpens few-shot behavior on standardized benchmarks. The training infrastructure ensures rapid iteration to 8T tokens.

## 4. Key Insights and Innovations
- Data-at-scale for smaller models (fundamental)
  - Training a 15B model on `8T` tokens is unusually data-rich. This tests and supports the Chinchilla hypothesis at this scale: more data per parameter can beat bigger but under-trained models (Section 1).
  - Significance: enables single‑GPU deployment while retaining accuracy competitive with or better than 30–70B models on several tasks (Figure 1; Tables 3–4).

- Explicit multilingual and code distributions with tokenizer support (substantial)
  - The blend explicitly allocates `15%` to multilingual and `15%` to code, across 53 and 43 languages respectively (Figures 3–4). The tokenizer is trained with upsampled non‑English text to ensure subword coverage (Section 2).
  - Significance: state-of-the-art multilingual results among similarly sized general-purpose models and improved performance in low-resource programming languages (Tables 7–10; Table 6).

- Efficient inference via `GQA` while keeping many query heads (incremental but impactful)
  - Reducing KV heads to 8 while retaining 48 query heads cuts memory usage and speeds inference with minimal loss (Table 1; Section 2).
  - Significance: helps the 15B model fit on a single A100/H100 at 4096 context—directly addressing deployment constraints.

- Short “continued training” with targeted data/learning-rate schedule (incremental, strategically important)
  - A brief post‑pretraining phase reweights high‑quality data and adds “benchmark‑style alignment examples” plus a steeper LR decay (Section “Continued Training”).
  - Significance: boosts few-shot alignment to standardized evaluation formats and shores up weak areas without full-scale retraining.

## 5. Experimental Analysis
- Evaluation protocol (Section 3)
  - Categories and datasets:
    - Commonsense reasoning (0‑shot): SIQA, ARC‑Easy/Challenge, PIQA, Winogrande, HellaSwag (Table 3).
    - Aggregated: `MMLU` (5‑shot) and `BBH` (3‑shot) (Table 4).
    - Math: `GSM8K` (8‑shot, maj@1) (Table 5). Note: `maj@1` indicates one sample per question; accuracy is simply whether the single answer matches the gold.
    - Code: `HumanEval` (0‑shot), `MBPP` (3‑shot), `MultiPL‑E` (0‑shot, 11 languages reported) (Tables 5–6).
    - Multilingual: `XCOPA` (0 and 4‑shot), `TyDiQA-GoldP` (1‑shot), `MGSM` (8‑shot, English chain‑of‑thought), `FLORES‑101` (8‑shot; spBLEU via sacreBLEU with spm-flores-101) (Tables 7–10).
  - Baselines: LLaMA‑2 13B/34B/70B, Baichuan‑2 13B, QWEN 14B, Mistral 7B, Gemma 7B; and for multilingual comparisons: mGPT 13B, XGLM 7.5B, PaLM 62B/62B‑cont (Section 3; Tables 3–10).
  - Harness: LM‑Evaluation Harness for many English tasks (Section 3.1).

- Main quantitative results
  - Commonsense (0‑shot; Table 3):
    - `Nemotron‑4 15B` average 73.4 vs LLaMA‑2 34B 71.1 and Mistral 7B 70.4. Individual tasks include ARC-c 55.5, ARC-e 80.9, Winogrande 78.0, HellaSwag 82.4.
  - Aggregated (Table 4):
    - `BBH`: 58.7—best among similar-size open models; the paper also notes it exceeds LLaMA‑2 70B’s 51.2 in this benchmark.
    - `MMLU`: 64.2—competitive with Gemma 7B (64.3) and QWEN 14B (66.3). Per‑category breakdown in Table 11 shows STEM at 53.4 being the weakest area.
  - Math and Python‑centric code (Table 5):
    - `GSM8K` 46.0 (behind QWEN 14B 60.1; near Gemma 7B 46.4).
    - Code: `HumanEval` 31.6 and `MBPP` 40.6—close to QWEN 14B (32.2/40.8) and below Gemma 7B (32.3/44.4), but above Mistral 7B and LLaMA‑2 13B/34B.
  - Polyglot code (MultiPL‑E; Table 6):
    - Average across 11 languages: 24.5, slightly above Starcoder 15B (24.2) and Mistral 7B (23.6). Highlights include stronger results on lower‑resource languages like Scala (27.3 vs Starcoder 27.6; vs Mistral 22.2), Julia (24.8 vs 30.2/26.0), and R (18.6 vs 15.5/11.8); trade‑offs are language‑dependent.
  - Multilingual reasoning and QA:
    - `XCOPA` (Table 7): 0‑shot avg 59.5 vs mGPT 56.1 and XGLM 55.6; 4‑shot avg 68.9 vs XGLM 61.4 and mGPT 56.0.
    - `TyDiQA-GoldP` (Table 8; 1‑shot): avg 50.5—higher than PaLM 62B‑cont (45.7), PaLM 62B (40.5), and smaller baselines (QWEN 14B 39.8; LLaMA‑2 13B 33.2; Baichuan‑2 13B 30.8).
    - `MGSM` (Table 9; 8‑shot, English CoT): avg 41.3—substantially above PaLM 62B‑cont (32.0) and Mistral 7B (21.8).
  - Machine translation (FLORES‑101; Table 10):
    - From Chinese to several languages: average spBLEU 23.2 vs LLaMA‑2 13B 12.2 and Baichuan‑2 13B 16.1—gains of +90.2% and +44.1% respectively. Not just ZH→EN (34.0) but also ZH→JA (23.1) and ZH→FR (28.1).

- Do experiments support the claims?
  - Yes, for multilingual and general reasoning: Tables 7–10 and Table 4 (BBH) show clear improvements versus similarly sized baselines, and even versus some much larger models in selected multilingual tasks (e.g., TyDiQA and MGSM against PaLM 62B‑cont).
  - Code results are competitive: near or slightly below the best 7–14B models on Python-centric tasks (Table 5) but stronger on average across multiple programming languages vs Starcoder 15B and Mistral 7B (Table 6).
  - Commonsense and aggregated benchmarks are consistently strong vs LLaMA‑2 34B and Mistral 7B (Tables 3–4).

- Missing analyses and robustness checks
  - No ablation isolating the effect of `continued training` vs the 8T pretraining.
  - No study of the impact of tokenizer vocabulary size (`256k`) or the English/multilingual/code mix ratios.
  - Limited error analysis or failure case discussion; STEM on MMLU is relatively weak (Table 11), but causes are not probed.
  - Using “benchmark‑style alignment examples” during continued training (Section “Continued Training”) raises the possibility of stylistic overfitting to evaluation formats; the paper does not quantify sensitivity to prompt variants or unseen task styles.

## 6. Limitations and Trade-offs
- Assumptions and scope
  - The approach assumes access to very large, high-quality multilingual and code corpora and strong data filtering at web scale (Section 2). Reproducing the 8T pipeline may be challenging for many organizations.
  - It assumes 4096 context is sufficient; long-document tasks beyond this window are not addressed.
- Computational constraints
  - While inference is single‑GPU friendly, training requires extensive compute: 384 DGX H100 nodes for nearly two weeks (Pre‑training; Table 2). This limits research reproducibility.
  - `MFU` of ~30–34% (Table 2) suggests significant remaining headroom in hardware utilization; optimizing kernels could reduce training time further, but that is out of scope.
- Architectural trade-offs
  - Large tokenizer (`256k` vocab) improves multilingual coverage but increases embedding parameters to 3.2B (Section 2), consuming memory and potentially slowing embedding lookups.
  - `GQA` reduces inference KV memory but may slightly reduce attention expressivity compared to full multi-head KV; the paper does not quantify any accuracy trade-off.
- Methodological gaps
  - No ablation for `continued training` or data mixture; this makes it hard to attribute which component drives which gain.
  - Use of “benchmark-style alignment examples” (Section “Continued Training”) blurs the line between pure pretraining generality and format learning; more transparency about content and quantity would help assess overfitting risk.

## 7. Implications and Future Directions
- Field impact
  - Strong evidence that “more data per parameter” scales well even for models as small as 15B, particularly for multilingual and cross‑language reasoning. A well-mixed, high-quality multilingual/code corpus plus a tokenizer designed for low-resource coverage can close much of the gap to much larger models in key tasks (Tables 7–10, 4).
- Practical applications
  - Single‑GPU deployment for multilingual customer support, search, and analytics; code assistance across many languages (Table 6); and cross‑lingual QA/translation pipelines (Tables 8–10).
- Research directions
  - Controlled ablations:
    - Quantify contributions of `continued training`, data mixture ratios (70/15/15), and tokenizer size; measure prompt and format robustness.
  - Data governance and quality:
    - Explore more systematic active sampling toward weak categories (e.g., STEM on MMLU; Table 11); study bias/fairness and domain coverage across the 53 languages.
  - Longer-context and memory:
    - Extend context beyond 4096 and evaluate on long-range multilingual tasks; combine with retrieval or memory-efficient attention.
  - Efficiency:
    - Improve training `MFU`, experiment with sparsity/quantization to further reduce inference cost; investigate KV cache policies with `GQA` for streaming settings.
  - Safety and alignment:
    - The technical report focuses on pretraining performance; downstream instruction tuning, safety alignment, and multilingual safety evaluations are natural next steps.

Key citations to results and details:
- Architecture and tokenizer: Table 1; Section 2.
- Data composition: Figures 2–4; Section 2.
- Training setup and efficiency: Table 2; “Pre‑training”.
- Continued training method: “Continued Training”.
- Commonsense: Table 3.
- Aggregated (BBH, MMLU): Table 4; Table 11 (per-category MMLU).
- Math and Python‑centric code: Table 5.
- Polyglot code: Table 6.
- Multilingual reasoning/QA: Tables 7–9.
- Translation: Table 10.

Representative quotes grounding key points:
- Deployment target and scope:
  > “…developed to be the best general-purpose large language model (LLM) that can fit on a single NVIDIA A100 or H100 GPU.” (Abstract)
- Training completion:
  > “Training was completed in approximately 13 calendar days.” (Pre-training; Table 2)
- Multilingual breadth:
  > “We present Nemotron‑4 15B … spanning English, 53 additional natural languages as well as 43 programming languages.” (Conclusion)
- Multilingual strength:
  > “Nemotron‑4 15B exhibits the strongest multilingual performance of any general purpose language model at its scale…” (Conclusion)
- FLORES‑101 gains:
  > “Nemotron‑4 15B heftily outperforms both LLaMA‑2 13B and Baichuan‑2 13B…” (Section 3.4; Table 10)

Overall, Nemotron‑4 15B demonstrates that meticulous data scaling, multilingual/code balance, tokenizer coverage, and a lightweight alignment pass can push a 15B model to top-tier multilingual performance and competitive general ability—while remaining practical for single‑GPU deployment.
