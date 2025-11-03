# Apertus: Democratizing Open and Compliant LLMs for Global Language Environments

**ArXiv:** [2509.14233](https://arxiv.org/abs/2509.14233)
**Authors:** Alejandro Hernández‑Cano, Alexander Hägele, Allen Hao Huang, Angelika Romanou, Antoni‑Joan Solergibert, Barna Pasztor, Bettina Messmer, Dhia Garbaya, Eduard Frank Ďurech, Ido Hakimi, Juan García Giraldo, Mete Ismayilzada, Negar Foroutan, Skander Moalla, Tiancheng Chen, Vinko Sabolčec, Yixuan Xu, Michael Aerni, Badr AlKhamissi, Ines Altemir Marinas, Mohammad Hossein Amani, Matin Ansaripour, Ilia Badanin, Harold Benoit, Emanuela Boros, Nicholas Browning, Fabian Bösch, Maximilian Böther, Niklas Canova, Camille Challier, Clement Charmillot, Jonathan Coles, Jan Deriu, Arnout Devos, Lukas Drescher, Daniil Dzenhaliou, Maud Ehrmann, Dongyang Fan, Simin Fan, Silin Gao, Miguel Gila, María Grandury, Diba Hashemi, Alexander Hoyle, Jiaming Jiang, Mark Klein, Andrei Kucharavy, Anastasiia Kucherenko, Frederike Lübeck, Roman Machacek, Theofilos Manitaras, Andreas Marfurt, Kyle Matoba, Simon Matrenok, Henrique Mendoncça, Fawzi Roberto Mohamed, Syrielle Montariol, Luca Mouchel, Sven Najem‑Meyer, Jingwei Ni, Gennaro Oliva, Matteo Pagliardini, Elia Palme, Andrei Panferov, Léo Paoletti, Marco Passerini, Ivan Pavlov, Auguste Poiroux, Kaustubh Ponkshe, Nathan Ranchin, Javi Rando, Mathieu Sauser, Jakhongir Saydaliev, Muhammad Ali Sayfiddinov, Marian Schneider, Stefano Schuppli, Marco Scialanga, Andrei Semenov, Kumar Shridhar, Raghav Singhal, Anna Sotnikova, Alexander Sternfeld, Ayush Kumar Tarun, Paul Teiletche, Jannis Vamvas, Xiaozhe Yao, Hao Zhao, Alexander Ilic, Ana Klimovic, Andreas Krause, Caglar Gulcehre, David Rosenthal, Elliott Ash, Florian Tramèr, Joost VandeVondele, Livio Veraldi, Martin Rajman, Thomas Schulthess, Torsten Hoefler, Antoine Bosselut, Martin Jaggi, Imanol Schlag
**Institutions:** EPFL, ETH Zurich, Swiss National Supercomputing Centre (CSCS)

## 🎯 Pitch

Apertus introduces a fully transparent and legally compliant suite of large language models, uniquely pre-trained on 15 trillion tokens with comprehensive multilingual support across 1,811 languages. By prioritizing data transparency and releasing all training artifacts, it sets a new standard in verifiable and equitable AI development, enabling safer deployment and fostering inclusivity for diverse linguistic communities.

---

## 1. Executive Summary
Apertus is a fully open suite of large language models (LLMs) at 8B and 70B parameters trained on 15 trillion tokens with two priorities: legally compliant, transparent data use and broad multilingual coverage. It introduces a reproducible pretraining pipeline that retroactively honors content-owner opt‑outs, reduces verbatim memorization via a modified training loss, and releases all artifacts (data preparation, code, checkpoints, evals), enabling verifiable, compliant development at scale (see Section 1).

## 2. Context and Motivation
- Problem/gap:
  - Most “open” LLMs release weights without disclosing or reproducing the data pipeline; many include content that is non‑permissive or legally risky, and they often lack robust multilingual representation (Introduction, p. 6–7).
  - Models can regurgitate training text verbatim, creating copyright and privacy risks.
- Importance:
  - Real-world impact: organizations need models that comply with laws like the EU AI Act and content-owner rights to deploy safely at scale (Section 1; Section 3.1). 
  - Societal inclusion: many communities use non‑English languages; improving these models’ coverage reduces inequity in access (Introduction; Section 3.2.2).
- Prior approaches and shortcomings:
  - Open-weight releases (e.g., Llama, Qwen) provide checkpoints but not the full, replicable data pipeline. A few projects release more, but typically with fewer languages or incomplete compliance processes (Introduction, p. 6).
  - Memorization mitigation has mostly been tackled with post‑hoc alignment or filtering, which can be brittle (Appendix F).
- Positioning:
  - Apertus trains only on openly available data, respects robots.txt opt-outs retroactively, applies PII and toxicity filters, and evaluates memorization at scale. Multilingually, it trains on content from 1,811 languages with ∼40% non‑English tokens (Introduction; Section 3, Table 6). It also releases complete scientific artifacts (Introduction, p. 7–8).

## 3. Technical Approach
This section unpacks the model design, training recipe, data pipeline, and post‑training alignment.

- Model architecture (Section 2.1; Table 1)
  - Dense decoder-only Transformer with:
    - `xIELU` activation (a trainable, piecewise function combining squared and ELU-like behavior) for the MLP, improving efficiency and handling negative inputs (Section 2.1).
    - `QK-Norm` to normalize queries/keys and stabilize attention logits (Section 2.1).
    - `GQA` (grouped-query attention) to reduce KV heads for faster inference without quality loss (Section 2.1).
    - `RoPE` positional embeddings with NTK‑aware scaling; context during pretraining is 4,096 tokens, later extended to 65,536 (Sections 2.1 and 2.5).
    - Pre‑norm with `RMSNorm`, untied embeddings/output, and attention masks to prevent cross‑document attention (Section 2.1).
  - Scales:
    - `Apertus‑8B`: 32 layers, 4096 hidden size, 32/8 Q/KV heads, MLP=21,504 (Table 1).
    - `Apertus‑70B`: 80 layers, 4096 hidden, 64/8 Q/KV heads, MLP=43,008 (Table 1).

- Tokenizer (Section 2.2; Figure 1)
  - Uses Mistral‑Nemo v3 (“tekken”) byte‑level BPE, 131,072 vocab, with 47 customized special tokens. Chosen for cross‑lingual fairness and efficiency: lowest Gini (more equal tokenization cost across languages) and strong compression/fertility among tested tokenizers (Figure 1).

- Training recipe (Section 2.3; Table 2)
  - Optimizer: `AdEMAMix`—extends Adam with an additional, slow EMA of gradients to better leverage long‑run training signals, scaling favorably with model size and batch (Section 2.3; references to optimizer benchmark in Appendix).
  - Learning rate: `Warmup‑Stable‑Decay (WSD)` schedule; permits extending training without re‑warming. Final decay uses `1−sqrt` shape for stable cooldown (Section 2.3).
  - Memorization mitigation: `Goldfish loss`. Instead of computing cross‑entropy on all tokens, it masks a small proportion (2%) of tokens chosen by hashing local context windows (`h = 50`) and keeping only tokens with hash below a threshold related to `k = 50`. This discourages learning exact token‑to‑context mappings that cause regurgitation (Section 2.3; Algorithm in Appendix F).
    - Intuition: By “forgetting” sparse, deterministically selected tokens during training, the model cannot easily memorize continuous spans verbatim even after many exposures. The hash is recomputed per batch, so masked positions vary (Appendix F).
  - Batch scaling: doubles the global batch mid‑run without changing LR to improve hardware efficiency; empirically helpful late in training (Section 2.3; Table 2).

- Long‑context extension to 65k (Section 2.5; Table 5)
  - Training is staged at 8k, 16k, 32k, 64k contexts with increasing RoPE base (`Θ` grows from 1M to 12M). Uses context parallelism across nodes with tensor/pipeline/data parallelism to fit memory (Table 5). Data mixture biases toward longer documents while maintaining the final pretraining distribution (Section 3.4; Table 8).

- Data pipeline emphasizing compliance (Section 3)
  - `robots.txt with hindsight`: For the top ∼2M domains in FineWeb/2, current robots.txt (Jan 2025) is retrieved; if any AI crawler is disallowed, all past snapshots from 2013–2024 are removed. Estimated token loss: 8% (English) and 4% (multilingual) (Appendix B; Table B.1–B.3).
  - `PII stripping`: regex-based anonymization of emails, IPs, IBANs (Section 3.1.2).
  - `Toxicity filtering`: train per‑language binary classifiers (9 languages) using XLM‑R embeddings + MLP; drop the top 5% most toxic docs per language (Section 3.1.3; Figure 4).
  - Sources (Section 3.2):
    - General web: FineWeb‑2 (1,811 languages), with quality‑filtered FineWeb‑2‑HQ for top languages; English: FineWeb‑HQ, FineWeb‑Edu, DCLM‑Edu.
    - Code: StarCoderData (+ educationally scored subsets), CommonPile/Stack‑v2‑Edu.
    - Math: FineMath subsets; MegaMath and WebMath (Section 3.2.3).
    - Parallel text: EuroParl, ParaDocs (Section 3.2.2).
    - Canary/memorization probes: Project Gutenberg sequences injected at controlled frequencies (Section 3.2.4).
  - Five‑stage curriculum (Section 3.3; Table 6): Starts with broad NL + edu‑quality documents; gradually increases math/code; final cooldown (Stage 5) shifts toward highest‑quality subsets and adds parallel text, Clean Wikipedia, and limited task data.

- Post‑training (Section 4)
  - `SFT` (supervised finetuning) on ~4.18M examples (Table 12) spanning general instructions (WildChat, SciRiff), multilingual chat (SmolTalk2, EuroBlocks), math (NuminaMath extraction; Llama‑Nemotron), code/function calling (Glaive, xLAM), and regional data (Romansh, Swiss‑German, African languages). License filtering and decontamination ensure compatibility and fair evaluations (Sections 4.1.1–4.1.2; Tables 10–11).
  - `Alignment` with `QRPO` (Quantile Reward Policy Optimization) (Section 4.3):
    - Definition: An offline, direct alignment method that optimizes an absolute reward via quantile ranking against reference completions; avoids instability and compute overhead of online RL (Section 4.3; equations).
    - Two regimes:
      1) Standard topics: use a trained reward model (`Skywork‑Reward‑V2‑Llama‑3.1‑8B`) to score completions (Section 4.3.1).
      2) Controversial topics: use an LLM‑as‑judge with a constitutional prompt based on the newly proposed `Swiss AI Charter` of 11 principles (Appendix O; Section 4.3.2). A persona‑based generation stage provides diverse responses; Qwen3‑32B scores them using probability‑weighted 1–9 ratings (Section 4.3.2).
    - A length‑normalized QRPO variant is adopted to stabilize objectives across varying completion lengths (equations in Section 4.3).

- Infrastructure and efficiency (Section 6)
  - Trained on Alps supercomputer (CSCS), up to 4,096 GH200 GPUs. Estimated 6.74×10^24 FLOPs for 70B/15T training; ~6M GPU‑hours; ~5 GWh compute energy (Section 6.2; Appendix E).
  - Stability: 70B/8B pretraining ran stably with few rollbacks; gradient clipping at 0.1 was applied essentially every step (Section 2.6; Figure 3).

## 4. Key Insights and Innovations
1) Compliance-first pretraining with retroactive consent
- What’s new: The data pipeline queries current robots.txt for top domains and removes all historical crawls where AI crawling is disallowed, not just at crawl time. This “with hindsight” enforcement is unusual and quantifiably impacts data: −8% English tokens and −4% multilingual (Appendix B; Table B.1), with detailed bot blocks (Tables B.2–B.3).
- Why it matters: It provides a verifiable compliance story aligned with evolving legal norms and content-owner expectations (Section 3.1.1), lowering downstream legal risk.

2) Goldfish loss at scale for memorization mitigation
- What’s new: Use of a selective‑masking training objective calibrated to 2% masked tokens (`k=50`, `h=50`) that demonstrably keeps verbatim recall near baseline even with up to 128 exposures and 5k‑token prefixes (Section 5.4; Figure 8; Appendix F).
- Why it matters: It addresses copyright/privacy concerns without the brittleness of purely post‑hoc methods; the team shows robustness across decoding strategies (Table 25).

3) Broad multilingual coverage paired with open artifacts
- What’s new: Training on 15T tokens across 1,811 languages (FineWeb‑2) with ~40% non‑English, and ~4.18M post‑training SFT examples covering 149+ languages (Sections 3.2, 4.1.3; Table 12).
- Why it matters: Stronger equitable performance, including support for Romansh (six idioms) and African languages; plus full release of data pipelines and evals enables reproducibility and community extension (Introduction, p. 7–8; Sections 4.1 and K).

4) Training‑efficiency recipe (xIELU + QK‑Norm + AdEMAMix + WSD)
- What’s new: A set of architectural and optimization choices validated by ablations and controlled replays (Section 2.4; Table 3; Table 4; Figure 2).
  - At 3B scale, the merged recipe reaches the baseline loss with 30–40% fewer tokens (Figure 2).
  - When replaying OLMo2 data/hyperparameters, Apertus‑style recipe matches OLMo2 loss with 46% fewer tokens at 1B and 30% fewer at 7B (Table 4).
- Why it matters: Lower training cost and better stability at large scale.

5) QRPO alignment with a democratic “Swiss AI Charter”
- What’s new: A practical, offline alignment method that accepts absolute reward signals—allowing both reward‑model scores and constitutional LLM‑as‑judge ratings grounded in a publicly surveyed charter (Section 4.3; Appendix O; Table 13).
- Why it matters: It makes values‑guided alignment transparent and auditable, supporting pluralistic norms for controversial topics (Section 4.3.2).

## 5. Experimental Analysis
- Evaluation methodology
  - Pretraining quality is tracked with probabilistic scoring on classic NLP benchmarks (ARC, HellaSwag, WinoGrande, XNLI, PIQA; MMLU; Global‑MMLU; culture‑oriented INCLUDE, BLEnD, CulturalBench; SwitzerlandQA) using lm‑evaluation‑harness (Section 5.1; Tables 14–15).
  - Post‑training uses open generation for knowledge, reasoning, math, coding, instruction following, culture (Section 5.2; Tables 17–21). Long‑context is assessed with RULER up to 64k (Table 23).
  - Memorization is measured on injected Gutenberg probes at varied repetition and offsets, using ROUGE‑L and LCCS, and checked against TTR to avoid low‑entropy false positives (Section 5.4; Figures 8–10; Table 25).
  - Safety uses BBQ, HarmBench variants, ToxiGen, and an in‑house RealToxicityPrompts‑Llama‑Guard‑subsample (Section 5.5; Table 26), plus LinguaSafe severity‑weighted scoring across 12 languages (Tables 27–28).

- Main quantitative results
  - Pretraining capability (Tables 14–15; Figure 7):
    - General language understanding (Table 14, average across tasks):
      - `Apertus‑8B`: 65.8; `Apertus‑70B`: 67.5. Comparable to `Llama‑3.1‑8B` (65.4) and `Llama‑3.1‑70B` (67.3); behind `Qwen2.5‑72B` (69.8). Among fully open models, they are state‑of‑the‑art.
    - Factual knowledge (Table 15, averages):
      - `Apertus‑70B`: 58.9 (MMLU 65.2, Global‑MMLU 58.2); `Apertus‑8B`: 56.9. Close to `OLMo2‑32B` average 62.0; below `Llama‑3.1‑70B` average 66.7 and `Qwen2.5‑72B` average 72.5. On regional cultural knowledge (INCLUDE V1/V2, CulturalBench, BLEnD, SwitzerlandQA), Apertus models are competitive; e.g., `Apertus‑70B` scores 75.0 on BLEnD (Table 15).
    - Progress over training is smooth and stable (Figure 7, macros for Global/English/EU/Swiss).
  - Post‑training (Tables 17–21):
    - Knowledge & commonsense (Table 17, averages):
      - `Apertus‑70B‑Instruct`: macro average 63.4 (MMLU 69.6, HellaSwag 78.1).
      - `Apertus‑8B‑Instruct`: 58.8.
      - These trail top open‑weight baselines (`Llama‑3.3‑70B‑Instruct`: 68.4; `Qwen2.5‑72B‑Instruct`: 68.8) but are competitive among fully open models (e.g., `OLMo‑2‑0325‑32B‑Instruct` 68.0).
    - Coding & math (Table 18):
      - `Apertus‑70B‑Instruct`: average 54.4 (HumanEval@10 73.0, GSM8K 77.6, MathQA 33.9). This lags GPT‑class open‑weights (`Llama‑3.3‑70B‑Instruct` average 74.3; `Qwen3‑32B` average 76.3).
      - The team attributes this to not using heavier RL‑with‑verifiers pipelines yet (Section 5.2).
    - Reasoning & instruction following (Table 19):
      - `Apertus‑70B‑Instruct`: average 61.8 (BBH 64.2; IFEval 75.2; Multi‑IFEval 74.7), respectable among fully open models.
    - Cultural knowledge (Table 20):
      - `Apertus‑70B‑Instruct`: average 61.5; strong on SwitzerlandQA (67.2) and CulturalBench (74.2). Again competitive among fully open models.
    - Held‑out test suite (Table 21):
      - `Apertus‑70B‑Instruct`: average 51.4 (AGIeval 40.5; ARC‑Challenge‑Chat 85.0; GSM8K‑Platinum 74.6), below the strongest open‑weights but solid for a fully open stack.
    - Long‑context RULER (Table 23):
      - `Apertus‑70B‑Instruct`: 94.8 (4k), 89.9 (8k), 85.7 (16k), 81.9 (32k). 64k not reported due to runtime limits. These are good but trail `Llama‑3.3‑70B‑Instruct` at 93.7 (32k) and `Qwen3` models overall.
    - Low‑resource translation (Table 24):
      - Germans↔Romansh (six idioms): `Apertus‑70B‑Instruct` consistently beats `Llama‑3.3‑70B‑Instruct`; e.g., Rumantsch Grischun DE→RM 27.8 vs 21.6 BLEU, RM→DE 44.7 vs 35.6.
  - Memorization robustness (Section 5.4; Figures 8–10; Table 25):
    - Across repetitions 1–128 and prefixes up to 5,000 tokens, ROUGE‑L stays near baseline ≈0.18 (Figure 8). Under nucleus sampling, TTR remains high (~0.5) while ROUGE‑L/LCCS stay low (Table 25), showing mitigation isn’t an artifact of degenerate greedy decoding.
    - Observed “primacy effect”: probes inserted earlier (0–9T) sometimes exhibit slightly higher recall than those added later (9–12T), potentially due to text complexity differences (Figure 9).
    - Failure cases: highly duplicated public texts (Bible, Shakespeare, US Constitution) can defeat deterministic hashing when near‑duplicates exist with different formatting/tokenization, revealing an inherent limitation of the scheme (Section 5.4.2).
  - Ablations and curriculum studies:
    - Architecture/optimizer ablations: AdEMAMix and xIELU each improve loss at 1.5B scale; merging all changes at 3B reaches target loss with 30–40% fewer tokens (Table 3, Figure 2).
    - OLMo2 replay: Apertus recipe achieves similar loss with 46%/30% fewer tokens at 1B/7B (Table 4).
    - Data cooldown ablations: DCLM‑Edu gives strongest gains; replacing FineWeb‑Edu with FineWeb‑HQ helps; results in Table 7.

- Safety and robustness (Section 5.5; Tables 26–28)
  - Mixed picture typical for open models:
    - BBQ (bias) is moderate (e.g., `Apertus‑70B‑Instruct` 67.4).
    - ToxiGen classification trails top open‑weights (70.3 vs 86.7 for `gemma‑3‑12b‑it`).
    - HarmBench: direct requests relatively low (10.3% harmful for `A70B‑I`), but human jailbreaks remain a challenge (36.2%).
    - RealToxicityPrompts‑Llama‑Guard subsample: very low rates (0.2), but note non‑comparability to original RTP (Section 5.5.2).
  - Multilingual safety with LinguaSafe: severity‑weighted scores reported across 12 languages and 5 harm categories (Tables 27–28) to encourage cross‑lingual safety evaluation.

- Do the experiments support the claims?
  - Compliance: pipeline is fully specified and quantified (Appendix B; Section 3.1), meeting the transparency goal.
  - Multilingual: broad coverage and culture‑specific benchmarks back the claim (Tables 20, 24).
  - Memorization: extensive probe study with multiple controls and failure analysis supports reduced verbatim recall (Section 5.4).
  - Capability: apertus is competitive among fully open models, though open‑weight SOTA still leads on several fronts (Tables 17–21).

## 6. Limitations and Trade-offs
- Compliance filtering reduces available data:
  - robots.txt hindsight removes 4–8% of tokens; toxicity filtering only covers nine languages (Figure 4), and PII removal uses regex heuristics, which can miss or over‑remove in code/math (Sections 3.1.2–3.1.3).
- Memorization mitigation limits:
  - Goldfish masking can fail on highly duplicated public texts with formatting/tokenization differences; deterministic hashing can be brittle to near‑duplicates (Section 5.4.2).
- Capability trade‑offs:
  - Math/coding lag behind the best open‑weights (Table 18), likely because the project prioritized compliance/multilinguality and did not yet use heavy RL‑with‑verifiers pipelines.
  - Long‑context quality is good but not top‑tier compared to Qwen/Llama at ≥32k (Table 23).
- Engineering constraints:
  - FP8 training increased throughput by ~26% but destabilized loss; rolled back to BF16 (Appendix D).
  - Aggressive gradient clipping (0.1 each step) may limit some learning dynamics (Section 2.6).
- Safety scope:
  - Jailbreak resistance is not the focus; authors argue practical deployments must add guardrails (Section 5.5.1). Some safety benchmarks remain challenging (Table 26).

## 7. Implications and Future Directions
- How this changes the landscape:
  - Sets a precedent for fully transparent, compliance‑aware LLM training at large scale—data scripts, filters, checkpoints, and evals are all released (Introduction, p. 7–8). This raises the bar for what “open” means and provides a template for legally cautious organizations.
  - Demonstrates at‑scale memorization mitigation during pretraining, moving beyond brittle post‑hoc approaches (Section 5.4; Appendix F).
  - Advances multilingual inclusion by training and evaluating on hundreds of languages (Sections 3.2.2, 5.2; Table 20), including low‑resource Romansh (Table 24; Appendix J, K).

- Follow‑up research enabled/suggested (Section 7):
  - `Scaling & Distillation`: Train larger or MoE variants; distill 70B into smaller models while preserving multilingual and safety properties.
  - `RL with Verifiers (RLVR)`: Integrate verifiers for math/code to close capability gaps (Section 7).
  - `Reasoning with adaptive compute`: Explore variable-depth/routing or CoT token budgeting (Section 7).
  - `Data‑to‑performance mapping`: Quantify how each data slice and governance choice affects capability, fairness, and memorization (Section 7).
  - `Multimodality`: Extend compliance pipeline to vision/audio while keeping transparent data governance (Section 7).
  - `Societal alignment`: Broaden value elicitation beyond Swiss charter to pluralistic, multilingual communities; study constitutional evaluation at scale (Section 4.3.2; Appendix O).
  - `Field evaluation`: Deploy with domain professionals and public users to audit reliability across languages and tasks (Section 7).

- Practical applications
  - Public‑sector chatbots and civic services where compliance and multilinguality are mandatory (Swiss context, Section 4.4; Appendix P).
  - Enterprises needing legally defendable data provenance and reduced regurgitation risk.
  - Research platform for studying memorization, alignment, and multilingual fairness with full reproducibility.

> In sum, Apertus offers a compliance‑first, multilingual, and fully transparent alternative in the open LLM ecosystem. It does not yet dethrone top open‑weights on math/coding or long‑context, but it decisively advances legal robustness, reproducibility, and memorization safety while remaining competitive among fully open models.
