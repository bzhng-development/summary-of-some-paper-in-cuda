# OPT: Open Pre-trained Transformer Language Models

**ArXiv:** [2205.01068](https://arxiv.org/abs/2205.01068)

## 🎯 Pitch

OPT delivers an unprecedented open suite of decoder-only transformer models ranging from 125M to 175B parameters, matching GPT-3's performance while releasing comprehensive training code, model weights (≤66B openly, 175B via request), and detailed training documentation. This openness empowers the research community to systematically study, reproduce, and improve large language models—making progress on robustness, bias, and efficiency possible for many, and ensuring the next generation of language AI is both scientifically rigorous and socially responsible.

---

## 1. Executive Summary
OPT introduces an open suite of decoder‑only transformer language models from 125M to 175B parameters that closely tracks GPT‑3 performance while releasing training code, models (up to 66B openly, 175B by request), and a detailed training logbook. It matters because it turns a largely closed research area into one that many groups can reproduce and scrutinize, and it shows that a GPT‑3‑class model can be trained with substantially lower energy and compute overhead.

## 2. Context and Motivation
- Problem addressed:
  - The strongest large language models (LLMs) require vast compute and are largely inaccessible; researchers typically cannot inspect weights or training procedures. This hinders reproducibility and limits community understanding of robustness, bias, and safety issues.
- Importance:
  - Real‑world: LLMs are central to search, assistants, dialogue, and downstream NLP. Responsible deployment needs community auditing, which requires access to models and training details.
  - Scientific: Reproducible baselines at multiple scales enable controlled studies of scaling laws, data curation, safety interventions, and decoding strategies.
- Prior approaches and gaps:
  - GPT‑3 and successors are available primarily through paid APIs with hidden weights and safety layers.
  - Some open initiatives exist (e.g., EleutherAI up to 20B parameters, BigScience/BLOOM; see Introduction and Related Work) but not a GPT‑3‑class suite with matching evaluation setup plus training process transparency.
- Positioning:
  - OPT offers a GPT‑3‑style suite (125M→175B) with near‑parity performance on standard prompts, full training details, code (“metaseq”), a release of all models ≤66B, and research access to 175B (Abstract; Sections 1, 2). It emphasizes efficient training and full documentation of mid‑flight interventions and failures.

## 3. Technical Approach
This is a practical recipe for training GPT‑3‑style models efficiently and transparently.

- Model family and architecture (Section 2.1; Table 1):
  - Nine decoder‑only transformers from `125M` to `175B` parameters.
  - For each size, Table 1 specifies layers (`#L`), attention heads (`#H`), hidden size (`dmodel`), peak learning rate, and global batch (tokens). Example: the `175B` model uses `96` layers, `96` heads, `dmodel=12288`, peak LR `1.2e−4`, batch `2M` tokens.
  - Design choices largely mirror GPT‑3 to reduce instability risk but adjust batch sizes for throughput.
  - Activation: ReLU (not GELU); sequence length 2048; dropout 0.1 except embeddings (Section 2.2).

- Optimization and initialization (Section 2.2):
  - Weight init: normal with std 0.006; final output layer std scaled by `1/√(2L)` where `L` is layers.
  - Optimizer: AdamW with β1=0.9, β2=0.95, weight decay 0.1.
  - LR schedule: linear warm‑up then decay; 175B warms up over first 2000 steps, smaller models over 375M tokens; decay to 10% of peak by 300B tokens. Mid‑flight manual LR reductions are applied for stability (Section 2.5; Figure 1).
  - Gradient clipping initially at 1.0, later lowered to 0.3 to stabilize training spikes (Section 2.5).
  - Gradient pre‑divide factor: split global gradient division across two operations by `√N` to reduce numerical under/overflows when aggregating across `N` processes (Section 2.2).
  - Dynamic loss scaling: scale loss during mixed‑precision training to avoid underflow; a “loss scalar” is adaptively adjusted (Section 2.4).

- Training data and preprocessing (Section 2.3):
  - Corpus combines: RoBERTa components (BookCorpus, Stories, CCNews v2), a curated subset of The Pile (CommonCrawl, OpenWebText2, Wikipedia, Project Gutenberg, DM Mathematics, HackerNews, OpenSubtitles, USPTO), and a subset of Pushshift Reddit transformed to linear threads by selecting each thread’s longest comment chain.
  - Total size: ~180B tokens, tokenized with GPT‑2 byte‑level BPE.
  - De‑duplication with `MinHashLSH` using a Jaccard similarity threshold ≥0.95 to remove near‑duplicate documents across datasets. MinHashLSH is a locality‑sensitive hashing technique that quickly estimates set similarity; here it helps prevent the model from overfitting repeated text.

- Systems and parallelism (Section 2.4):
  - Uses `Fully Sharded Data Parallel (FSDP)` combined with `Megatron-LM Tensor Parallelism`.
    - FSDP shards model parameters, gradients, and optimizer state across GPUs to fit larger models in memory.
    - Tensor parallelism splits large matrix multiplications within layers across GPUs to increase compute throughput.
  - Mixed precision: parameters in FP16, optimizer states in FP32 (for numerical stability).
  - Scale and efficiency:
    > “We trained OPT‑175B on 992 80GB A100 GPUs … reaching 147 TFLOP/s utilization per GPU.” (Section 2.4)
  - Failure handling and restarts:
    > “at least 35 manual restarts and an estimated 70+ automatic restarts due to hardware failures … over 2 months” (Section 2.5). Nodes were diagnosed and cordoned; training resumed from checkpoints.

- Stabilizing instabilities (Section 2.5; Figures 1–2):
  - Observed correlation between divergences, the dynamic loss scalar collapsing to 0, and spikes in final‑layer activation L2 norms.
  - Strategy:
    - Reduce LR and restart from a checkpoint where the loss scalar is “healthy” (≥1.0).
    - Lower gradient clip threshold from 1.0 to 0.3 early in training.
    - Reset loss scalar; test switching to SGD (did not help); upgrade Megatron version (reduced activation norm pressure).
  - Evidence:
    > Figure 1 shows the “Empirical Learning Rate” that was lowered multiple times mid‑flight, and Figure 2 shows the corresponding effects on validation perplexity curves.

- Decoding and evaluation protocols:
  - For standard NLP tasks, reuse GPT‑3 prompts and settings; formulate WSC as multiple choice as in GPT‑3 (Section 3.1).
  - For dialogue generation, use greedy decoding up to 32 tokens and a minimal dialogue prompt structure (“Person 1:” / “Person 2:”) for OPT‑175B; compare against supervised BlenderBot variants that use tuned decoding (Section 3.2).

## 4. Key Insights and Innovations
- Opening a GPT‑3‑class suite with process transparency (fundamental contribution):
  - OPT provides models across scales, code (“metaseq”), and a unique training logbook with day‑to‑day decisions and failures (Sections 1, 6). This kind of transparency—e.g., explicit LR changes, gradient clipping adjustments, and hardware failure rates (Section 2.5)—has been missing in prior mega‑model releases.
- Efficient large‑model training without pipeline parallelism (systems innovation):
  - Demonstrates that combining FSDP with tensor parallelism suffices to train a 175B decoder‑only model on Nvidia GPUs without pipeline parallelism (Section 6). This simplifies training orchestration and is the “only known open‑source implementation … ≥175B … without the use of pipeline parallelism” at publication time.
- Lower energy footprint for GPT‑3‑class training (practical and environmental significance):
  - OPT‑175B’s estimated CO2eq is 75 tons versus GPT‑3’s reported ~500 tons and Gopher’s ~380 tons (Section 6), roughly 1/7th the footprint.
- Thorough, mixed-result safety/ethics evaluation at 175B scale (important capability and caution):
  - OPT‑175B shows stronger hate‑speech detection in ETHOS few‑shot settings (Table 3) but higher toxicity generation on RealToxicityPrompts and more stereotypical bias on CrowS‑Pairs (Table 4; Figure 5). The paper connects these outcomes to data composition choices (e.g., Reddit inclusion).

These are more than incremental tweaks: they change who can study LLMs at scale and how such models are trained and evaluated in the open.

## 5. Experimental Analysis
- Evaluation methodology (Section 3):
  - Standard NLP prompting on 16 tasks: HellaSwag, StoryCloze, PIQA, ARC‑E, ARC‑C, OpenBookQA, Winograd, WinoGrande, and SuperGLUE tasks (BoolQ, CB, COPA, WSC, MultiRC, WiC, RTE, ReCoRD). Accuracy is the primary metric. WSC is multiple‑choice formatted as in GPT‑3 (which affects scores).
  - Averages in Figures 3–4 exclude MultiRC and WiC because they “systematically favor” one family.
  - Dialogue evaluation on ConvAI2, Wizard of Wikipedia (WoW), Empathetic Dialogues (ED), Blended Skill Talk (BST), and Wizard of Internet (WoI), reporting perplexity and Unigram F1 (UF1) token overlap (Section 3.2; Table 2).
    - Unigram F1: harmonic mean of precision/recall on unigrams; a measure of lexical overlap with reference responses.
  - Safety/bias evaluations: ETHOS hate‑speech detection (F1), CrowS‑Pairs (percentage preferring stereotypical sentences; lower is better), StereoSet with LMS, SS, and ICAT (higher ICAT is better), RealToxicityPrompts (toxicity probability vs prompt toxicity), and Dialogue Safety Benchmarks (Section 4; Tables 3–6; Figure 5).
    - LMS (Language Modeling Score) captures how well a model assigns probability to sensible sentences; SS (Stereotype Score) measures preference for stereotypical content; ICAT balances both.

- Main quantitative results:
  - NLP zero‑shot averages (Figure 3):
    > “Across a variety of tasks and model sizes, OPT largely matches the reported averages of GPT‑3.” The trendlines for OPT and GPT are very close across 125M→175B.
  - One‑shot and 32‑shot (Figure 4):
    > OPT lags slightly behind GPT‑3 on average few‑shot accuracy, with substantial per‑task variance (Appendix A, Figures 6–7). Performance is “similar” on 10 tasks but consistently underperforms GPT‑3 on MultiRC; several tasks exhibit scale‑instability (BoolQ, CB, WSC, RTE).
  - Dialogue (Table 2):
    - On ConvAI2, unsupervised OPT‑175B achieves ppl 10.8 and UF1 0.185 vs supervised BlenderBot1 ppl 10.2, UF1 0.183; Reddit 2.7B unsupervised is much worse (ppl 18.9, UF1 0.126).
    - On WoI (unsupervised for all models), OPT has the best perplexity (12.0) but lower UF1 (0.147) than supervised models (0.154–0.160).
    - Generalization check: on ConvAI2 hidden test (not in pretraining), OPT‑175B reaches ppl 10.7, UF1 0.185; on MSC (ConvAI2‑like), ppl 9.7, UF1 0.177—suggesting genuine skill rather than leakage (Section 3.2).
  - Hate‑speech detection (Table 3):
    > OPT‑175B outperforms the Davinci API across setups: zero‑shot F1 0.667 vs 0.628; one‑shot 0.713 vs 0.616; few‑shot binary 0.759 vs 0.354; few‑shot multiclass 0.812 vs 0.672.
  - CrowS‑Pairs (Table 4):
    > Overall stereotypical preference is higher (worse) for OPT‑175B (69.5) than GPT‑3 (67.2); worse in most categories except religion.
  - StereoSet (Table 5):
    > Overall ICAT is similar (60.0 OPT vs 60.8 Davinci); OPT shows lower SS (less stereotypical preference) but also lower LMS (worse language modeling score), balancing out.
  - RealToxicityPrompts (Figure 5):
    > OPT‑175B has higher toxicity probability than Davinci and PaLM across prompt‑toxicity bins, and toxicity rises with prompt toxicity for all models.
  - Dialogue safety (Table 6):
    > OPT‑175B’s unit‑test safety scores are comparable to Reddit 2.7B and worse than supervised BlenderBot variants, especially on “Unsafe” prompts (0.567 vs 0.250–0.289; lower is better).

- Do the experiments support the claims?
  - Matching GPT‑3 class: Figures 3–4 and Appendix plots show OPT roughly tracks GPT‑3 averages across sizes; per‑task deviations and the MultiRC discrepancy are acknowledged.
  - Efficient training: Sections 2.4–2.5 document utilization (147 TFLOP/s/GPU), large‑scale training without pipeline parallelism, and careful handling of instabilities; Section 6 contrasts estimated carbon footprints.
  - Safety and bias: Mixed results support the conclusion that training on broad, minimally moderated corpora yields strong awareness of toxic language (good detection) but higher propensity to generate it and to encode some stereotypes (Tables 3–4; Figure 5).

- Ablations/robustness/failure cases:
  - Mid‑flight interventions (Section 2.5) function as “ablation‑like” observations: LR reductions and clip‑norm changes quell divergence; switching to vanilla SGD did not help; upgrading Megatron improved activation‑norm behavior.
  - Known pitfalls: inconsistent scale effects on small validation‑set tasks (CB, BoolQ, WSC), and underperformance on MultiRC despite attempts to replicate GPT‑3’s setup (Section 3.1).
  - Prompt‑sensitivity remains; WSC reformulation is known to alter difficulty (Section 3.1).

## 6. Limitations and Trade-offs
- Data composition and safety (Sections 4, 5 Limitations):
  - Inclusion of Reddit and other web corpora likely increases exposure to toxic and stereotyped text; this correlates with higher CrowS‑Pairs bias and RealToxicityPrompts toxicity generation.
  - The models can produce incorrect facts, repetitive loops, and fail at direct instruction following; instruction tuning or RLHF is not applied.
- Prompting sensitivity (Section 5 Limitations):
  - “Declarative instructions or point‑blank interrogatives” often elicit meta‑dialogue rather than task execution; performance varies with prompt wording and few‑shot ordering.
- Compute and engineering assumptions (Sections 2.4–2.5):
  - Requires access to ~1,000 A100‑80GB GPUs, robust checkpointing, and the metaseq stack; training is non‑trivial due to frequent hardware failures and numerical instabilities.
- Evaluation ambiguities (Section 3.1; Appendix A):
  - Some tasks have very small validation sets; OPT cannot replicate GPT‑3’s MultiRC and WiC results under the same public prompts; Davinci API results may reflect undocumented safety layers.
- Not production‑ready (Section 5 and D.2):
  - The release intentionally avoids safety fine‑tuning; the license is non‑commercial; the authors explicitly caution against deployment without mitigations.

## 7. Implications and Future Directions
- How this changes the field:
  - Provides a reproducible, GPT‑3‑class baseline with code and weights for the research community; enables systematic studies of scaling, data curation, safety mitigations, and decoding beyond API‑bound experiments.
  - Demonstrates that FSDP+tensor parallelism can train 175B models efficiently without pipeline parallelism, simplifying future large‑model training stacks.
- Follow‑up research enabled:
  - Instruction tuning and RLHF on OPT‑175B to address instruction‑following weaknesses; Section 5 suggests InstructGPT‑style approaches.
  - Retrieval‑augmented generation to improve factuality (Section 5 cites several retrieval‑based methods).
  - Safety mitigations and audits: targeted data filtering, debiasing methods (e.g., self‑debiasing, unlikelihood training), and post‑training safety layers; dialogue‑specific safety finetuning improves toxicity (Table 6).
  - Data governance and carbon accounting: the logbook and footprint estimates (Section 6) invite standardized reporting, failure‑overhead accounting, and embodied‑carbon analysis of hardware.
- Practical applications and use cases:
  - Research‑only use in: dialogue, few‑shot prompt design, safety‑tech evaluation, model analysis (e.g., scaling laws, activation norm monitoring), and systems research on distributed training.
  - As an open baseline, OPT’s smaller models (≤66B) are directly usable for downstream finetuning studies; the 175B research access supports controlled experiments at state‑of‑the‑art scale.

> Key numbers and artifacts to remember:
> - Training: 992 A100‑80GB GPUs; 147 TFLOP/s/GPU; dynamic loss scaling; numerous restarts (Sections 2.4–2.5).
> - Data: ~180B tokens; GPT‑2 BPE; heavy de‑dup with MinHashLSH at Jaccard ≥0.95; Reddit longest‑chain transformation (Section 2.3).
> - Performance: OPT ≈ GPT‑3 averages (Figures 3–4); competitive unsupervised dialogue perplexity and UF1 vs supervised models on ConvAI2 (Table 2).
> - Safety: Higher hate‑speech detection F1 in few‑shot (Table 3) but higher toxicity continuation rates (Figure 5) and more bias on CrowS‑Pairs (Table 4).
> - Footprint: ~75 tons CO2eq vs ~500 for GPT‑3 (Section 6).
