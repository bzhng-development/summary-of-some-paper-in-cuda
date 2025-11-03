# OPT: Open Pre-trained Transformer Language Models

**ArXiv:** [2205.01068](https://arxiv.org/abs/2205.01068)

## 🎯 Pitch

OPT introduces a fully open suite of GPT-3-comparable, large-scale language models (ranging from 125M to 175B parameters), along with their model weights, training code, and a detailed logbook documenting the process. By enabling reproducible research at unprecedented scale, OPT removes critical barriers to transparency, responsible AI study, and innovation in large language models—while achieving comparable performance to GPT-3 with just one-seventh of its carbon footprint.

---

## 1. Executive Summary
OPT (“Open Pre‑trained Transformers”) is a family of decoder‑only language models spanning 125M to 175B parameters trained to closely match GPT‑3‑class performance while making nearly all model weights, code, and a training “logbook” publicly available. The work matters because it removes a major barrier to reproducible, large‑scale LLM research and documents how to efficiently and stably train a 175B model (e.g., 147 TFLOP/s per GPU on 992 A100s) with roughly one‑seventh the carbon footprint of GPT‑3 (75 vs. ~500 tons CO2eq; Section 6).

## 2. Context and Motivation
- Problem/gap:
  - State‑of‑the‑art large language models (LLMs) show strong zero‑ and few‑shot abilities, but most are closed (API‑only) and extremely expensive to reproduce. This limits rigorous study of robustness, bias, toxicity, and safety (Introduction).
- Why it matters:
  - Practical: LLMs power many applications; without access to weights and training details, mitigation research (e.g., on bias/toxicity) is hamstrung.
  - Scientific: Lack of open baselines at GPT‑3 scale prevents reproducibility and ablation of training/data choices.
- Prior approaches and shortcomings:
  - Some open initiatives existed (e.g., EleutherAI up to 20B, Salesforce’s 20B, BigScience; Introduction, footnote 2), but none provided a GPT‑3‑scale dense model with full training details and code needed to replicate training at 175B.
- Positioning:
  - OPT is a replication‑oriented suite: architectures and hyperparameters intentionally mirror GPT‑3 (Table 1; Section 2.1–2.2) to provide comparable baselines; the novelty is transparency, efficient large‑scale training without pipeline parallelism, and breadth of Responsible AI evaluation.

## 3. Technical Approach
This is an empirical engineering effort designed to (a) reproduce GPT‑3‑class models, (b) document large‑scale training mechanics, and (c) release models/code.

- Model family and architecture (Section 2.1; Table 1)
  - Decoder‑only Transformers at nine scales: 125M → 175B parameters.
  - Example specs:
    - `OPT-175B`: `#L=96` transformer layers, `#H=96` attention heads, model dim `d_model=12288`, peak LR `1.2e‑4`, global batch size `2M` tokens (Table 1).
  - Design choice: closely follow GPT‑3 to reduce instability risks and allow direct comparison, varying batch sizes mainly for computational efficiency.

- Training setup (Section 2.2)
  - Initialization: normal(0, 0.006); output layer std scaled by `1/√(2L)`; biases zero.
  - Optimizer: AdamW (β1=0.9, β2=0.95, weight decay 0.1).
  - LR schedule: linear warmup (first 2000 steps for 175B; 375M tokens for smaller) to a max LR, then decay to 10% of max over 300B tokens. Several mid‑flight adjustments were necessary (Section 2.5; Figure 1).
  - Regularization and stability: dropout 0.1 (not on embeddings), gradient clipping at 1.0, later reduced to 0.3 to quell instabilities (Section 2.5), and a “gradient predivide” trick to avoid numeric over/underflows when aggregating gradients across many workers.
  - Sequence length 2048; activation function ReLU.
  - Mixed precision: parameters in FP16; Adam states in FP32; dynamic loss scaling to avoid FP16 underflows.

  Definitions for nonstandard terms:
  - `dynamic loss scaling`: automatically scales the loss upward during training to keep gradients in representable FP16 range; if overflow occurs, the scale is reduced.
  - `gradient clipping`: caps the gradient norm to prevent exploding updates.

- Data pipeline (Section 2.3)
  - 180B tokens created from a union of:
    - RoBERTa data (BookCorpus, Stories, CCNews v2 up to Sep 28, 2021),
    - Subset of The Pile (CommonCrawl, OpenWebText2, USPTO, Gutenberg, OpenSubtitles, Wikipedia, DM Mathematics, HackerNews),
    - Pushshift.io Reddit (processed to extract the longest comment chains).
  - Tokenization: GPT‑2 byte‑level BPE.
  - De‑duplication: MinHash LSH with Jaccard ≥ 0.95 across all sources to remove near‑duplicate documents (Section 2.3). Note: heavy duplication found in The Pile; extra dedup advised.

  Definitions:
  - `MinHash LSH`: an efficient approximate method to detect near‑duplicate documents by hashing shingles of text; `Jaccard similarity` measures overlap between sets.

- Systems and parallelism (Section 2.4)
  - Hardware: 992× 80GB NVIDIA A100 GPUs.
  - Parallelization: `Fully Sharded Data Parallel (FSDP)` + Megatron‑LM `Tensor Parallelism`.
    - FSDP shards parameters, gradients, and optimizer states across data‑parallel workers, reducing per‑GPU memory.
    - Tensor Parallelism splits large matrix multiplications across GPUs within a layer.
  - Throughput: “up to 147 TFLOP/s per GPU” (Section 2.4).
  - Memory/precision: Adam states kept in FP32 (sharded across hosts), model weights in FP16.

- Training process realities (Section 2.5)
  - Hardware faults: 35 manual restarts and 70+ automatic restarts; >100 hosts cycled over ~2 months (Hardware Failures).
  - Loss divergence stabilization:
    - Empirical triggers: when dynamic loss scaler crashed to 0 and final‑layer activation L2 norm spiked.
    - Recovery: roll back to a checkpoint where the scaler was “healthy” (≥1.0), reduce LR, and continue; also tightened gradient clipping to 0.3 early on (Loss Divergences).
    - Figure 1 shows the effective LR schedule; Figure 2 shows validation perplexity reacting to mid‑flight LR changes.
    - Tried and reverted: switching to vanilla SGD (plateaued), reset loss scaler (partial help), upgraded Megatron (reduced activation pressure; improved throughput).

- Release plan and carbon accounting (Section 6)
  - Weights released for 125M–66B openly; `OPT‑175B` by request (research‑restricted, non‑commercial).
  - Codebase `metaseq` released; the logbook documents day‑to‑day issues and changes.
  - Estimated CO2eq to develop OPT‑175B: ~75 tons; GPT‑3 estimate ~500 tons; Gopher ~380 tons (Section 6).

## 4. Key Insights and Innovations
- Open, reproducible GPT‑3‑class suite with code and training “logbook”
  - What’s new: not a novel architecture, but a complete replication recipe, weights (125M–66B), code, and a candid log of mid‑flight changes and failures—details usually omitted. This enables transparent, large‑scale LLM research (Section 6).
  - Why it matters: reproducibility and community scrutiny of safety, bias, and stability are now practical.

- Efficient 175B training without pipeline parallelism
  - What’s different: combining `FSDP` with `Tensor Parallelism` to train a dense 175B decoder‑only model on NVIDIA GPUs “without the use of pipeline parallelism” (Section 6) while achieving 147 TFLOP/s/GPU (Section 2.4).
  - Significance: reduces engineering complexity and improves utilization; documents a high‑throughput recipe many can adopt.

- Process transparency on stability at scale
  - Novelty: explicit links between divergence events, dynamic loss scaler collapses, and final‑layer activation norm spikes; concrete remedies (LR reductions, stricter clipping) with their effect on validation perplexity shown in Figures 1–2 (Section 2.5).
  - Significance: pragmatic know‑how for others training 100B‑scale models.

- Broad, side‑by‑side evaluation—including Responsible AI
  - Scope: 16 NLP tasks (Section 3.1; Appendix A), multiple dialogue sets (Table 2), and bias/toxicity metrics (Tables 3–5; Figure 5; Table 6).
  - Value: shows where a GPT‑3‑class open model is comparable, where it lags, and where training data choices (e.g., Reddit) shift behavior (better hate‑speech detection but higher toxicity).

These are primarily infrastructural and openness innovations rather than algorithmic advances.

## 5. Experimental Analysis
- Evaluation design (Section 3; Appendix A)
  - Prompted NLP tasks (16 total): HellaSwag, StoryCloze, PIQA, ARC (Easy/Challenge), OpenBookQA, Winograd/Winogrande, SuperGLUE tasks (BoolQ, CB, COPA, WIC, WSC, MultiRC, RTE, ReCoRD).
  - Protocol: reuse GPT‑3 prompts; accuracy as the main metric; WSC cast as multiple choice (impacts scores; Section 3.1).
  - Averages: MultiRC and WIC excluded from some averages due to anomalies favoring one model (Section 3.1).

- Main NLP results
  - Zero‑shot averages (Figure 3):
    > “Across a variety of tasks and model sizes, OPT largely matches the reported averages of GPT‑3.”  
    Variation is task‑dependent; per‑task curves are in Appendix A, Figure 6.
  - 1‑ and 32‑shot averages (Figure 4):
    > “OPT performance for one‑ and few‑shot lags behind GPT‑3 models, but performance depends heavily per task.”  
    Detailed curves are in Appendix A, Figure 7.
  - Notable per‑task observations (Section 3.1):
    - Rough parity on 10 tasks; underperformance on ARC‑Challenge and MultiRC.
    - In CB, BoolQ, and WSC, both model families show inconsistent scaling, likely due to small validation sets (56, 277, 104 examples).
    - WIC: OPT consistently above the GPT‑3 numbers reported in Brown et al. (2020), whose 0% accuracy figure is suspect for a binary task (footnote 5).
    - MultiRC: attempted replication of GPT‑3 API results did not match reported numbers (footnote 6).

- Dialogue results (Section 3.2; Table 2)
  - Datasets: ConvAI2, Wizard of Wikipedia (WoW), Empathetic Dialogues (ED), Blended Skill Talk (BST), Wizard of Internet (WoI).
  - Metrics:
    - `perplexity` (lower is better): exponentiated average negative log‑likelihood; measures how well the model predicts the next token.
    - `Unigram F1` (higher is better): word‑overlap F1 between system and reference responses.
  - Highlights from Table 2 (unsupervised `OPT‑175B` vs supervised BlenderBot/R2C2):
    > ConvAI2: `OPT‑175B` 10.8 ppl / 0.185 UF1 vs BlenderBot 10.2 / 0.183; Reddit‑2.7B 18.9 / 0.126.  
    > WoI (unsupervised across models): `OPT‑175B` has lowest ppl 12.0 but UF1 0.147 vs R2C2 0.160.
  - Leakage check: they verified no overlap with ConvAI2 and matched performance on the hidden test set (10.7 ppl, 0.185 UF1) and on MultiSessionChat (9.7 ppl, 0.177 UF1), indicating generalization (Section 3.2).

- Responsible AI evaluations (Section 4)
  - Hate speech detection on ETHOS (Table 3; F1, higher is better):
    > Zero‑shot: Davinci 0.628 vs `OPT‑175B` 0.667.  
    > Few‑shot (multiclass): Davinci 0.672 vs `OPT‑175B` 0.812.  
    Interpretation: `OPT‑175B` performs better, possibly because pretraining on unmoderated Reddit improved recognition of problematic language (Section 4.1).
  - CrowS‑Pairs (Table 4; lower is better—less stereotypical bias):
    > Overall: Davinci 67.2 vs `OPT‑175B` 69.5 (worse).  
    `OPT‑175B` is more biased in most categories except religion. The Reddit component likely increases exposure to stereotypes (Section 4.2).
  - StereoSet (Table 5; higher `LMS` better language modeling, lower `SS` less bias, higher `ICAT` better trade‑off):
    > Overall: Davinci ICAT 60.8 vs `OPT‑175B` 60.0 (very close).  
    `OPT‑175B` has slightly lower LMS and slightly better SS in most categories.
  - RealToxicityPrompts (Figure 5; toxicity probability of continuation):
    > `OPT‑175B` produces more toxic continuations than Davinci and PaLM across prompt‑toxicity bins; all models become more toxic as prompt toxicity increases.
  - Dialogue safety tests (Table 6; lower is safer in Unit Tests):
    > On Unit Tests, `OPT‑175B` is safer than Reddit‑2.7B in Safe (0.033 vs 0.300) and Adversarial (0.283 vs 0.439), but worse than supervised BlenderBot/R2C2 (e.g., Unsafe: 0.567 for `OPT‑175B` vs 0.250 for BlenderBot).  
    This underscores the value of fine‑tuning on curated safety data (Section 4.5).

- Do the experiments support the claims?
  - Yes for comparability: zero‑shot and multi‑shot averages (Figures 3–4) and many per‑task curves demonstrate similar scaling trends. Dialogue results show competitive unsupervised performance.
  - Nuance: OPT often matches GPT‑3 but is not uniformly better; it lags on some tasks and shows higher toxicity on RTP (Figure 5). The paper is careful to present mixed outcomes and likely causes (Sections 3–4).

- Additional evidence and failure cases
  - Training stability: Figures 1–2 show clear effects of LR adjustments on validation perplexity.
  - Sample generations (Appendix E): show rhyme/meter issues in poetry, instruction‑following limitations, arithmetic mistakes, and occasional repetitiveness (Figures 8–13).

## 6. Limitations and Trade-offs
- Methodological and data assumptions (Section 5; Appendix C)
  - Replication‑oriented design: mirrors GPT‑3; no instruction‑tuning, no retrieval augmentation—so instruction following is weak.
  - Data mixture includes unmoderated Reddit and CommonCrawl; beneficial for hate‑speech detection but increases toxicity and stereotypical associations (Sections 4.1–4.4).
  - Prompt sensitivity: results can hinge on prompt phrasing/order and shot count; some reported GPT‑3 numbers are hard to reproduce exactly (Section 3.1; footnote 6).

- Behavioral limitations (Section 5; Appendix E)
  - Instruction following: tends to role‑play the instruction rather than execute it; instruction‑tuning (e.g., InstructGPT) is suggested as future mitigation.
  - Repetition and looping: even with sampling; could be reduced with specialized training/decoding (e.g., unlikelihood training).
  - Factuality: can hallucinate; retrieval‑augmented methods likely beneficial.

- Safety/bias trade‑offs (Section 4)
  - Better at recognizing hate speech (Table 3) but more likely to generate toxicity (Figure 5) and more stereotypical on CrowS‑Pairs (Table 4), reflecting data trade‑offs.

- Compute and access constraints
  - Training required 992 A100s and months of wall‑clock time; despite a lower carbon footprint than GPT‑3, 75 tons CO2eq is still substantial (Section 6).
  - `OPT‑175B` weights are research‑restricted (non‑commercial license), not fully open for production use.

## 7. Implications and Future Directions
- How this changes the landscape
  - Provides the community with a GPT‑3‑class, reproducible open baseline suite (125M–66B openly; 175B by request) plus code and a training logbook. This substantially lowers the barrier to:
    - studying safety/bias at scale,
    - testing new training/regularization methods on realistic models,
    - benchmarking instruction‑tuning or retrieval‑augmentation recipes without API constraints.

- Research avenues enabled or suggested
  - Instruction‑tuning and alignment: apply methods like RLHF to `OPT‑175B` and quantify gains in instruction following and safety (Section 5).
  - Retrieval‑augmented generation: integrate retrieval to improve factuality (Section 5; Related Work).
  - Data curation studies: experiment with different mixes/filters (e.g., less Reddit; better deduplication) and measure effects on bias, toxicity, and stability (Sections 2.3, 4).
  - Training stability at scale: formalize diagnostics based on activation norms and loss‑scaler health; compare alternative optimizers/clipping schemes (Section 2.5).
  - Environmental reporting: standardize carbon accounting across training, ablations, and downtime, building on OPT’s detailed logbook (Section 6).

- Practical applications
  - As‑is: research prototypes for NLP tasks, analysis of LLM behavior, and as a pre‑training base for fine‑tuned systems.
  - With fine‑tuning/safety layers: dialogue agents (Table 2 suggests strong unsupervised starting points), classification (e.g., content moderation), and knowledge‑intensive tasks with retrieval.
  - Caution: the paper explicitly advises against commercial deployment in its current form due to safety, bias, and reliability issues (Section 5; Model Card D.2).

Overall, OPT’s key contribution is infrastructural: a well‑documented, performant, and largely open replication of GPT‑3‑class models that equips the community to do rigorous, large‑scale research on capabilities, risks, and mitigations.
