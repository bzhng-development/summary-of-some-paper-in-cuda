# BLOOM: A 176B-Parameter Open-Access Multilingual Language Model

**ArXiv:** [2211.05100](https://arxiv.org/abs/2211.05100)

## 🎯 Pitch

BLOOM introduces a 176-billion-parameter multilingual large language model, openly releasing both the model and its training data sourced from 46 natural and 13 programming languages. By providing state-of-the-art performance, transparent documentation, and a responsible data governance process, BLOOM democratizes access to cutting-edge language AI, enabling the global research community to innovate and adapt large language models for diverse and equitable applications beyond the English-speaking world.

---

## 1. Executive Summary (2-3 sentences)
BLOOM is a 176-billion-parameter, open-access, multilingual large language model trained on the 1.6TB ROOTS corpus spanning 46 natural and 13 programming languages (Section 3; Table 1; Figure 3). Its core contribution is to democratize access to state-of-the-art scale LLMs through an open, carefully governed data pipeline (Figure 2), transparent engineering/training recipe, and multilingual instruction-tuned variants (BLOOMZ) that deliver strong zero-shot performance (Section 4.7; Figure 11).

## 2. Context and Motivation
- Problem/gap:
  - Most large LLMs have been proprietary and overwhelmingly English-centric, limiting who can study, improve, and deploy them (Introduction; Section 2.1 “Social Limitations of LLM Development”).
  - High training costs and opaque datasets complicate reproducibility and responsible governance (Section 2.1).
- Why it matters:
  - Real-world: Multilingual capability is essential for equitable access and global applications; open availability enables local ecosystems to adapt models responsibly.
  - Scientific: Community reproducibility and ablation-backed architecture choices accelerate understanding and progress at very large scales (Sections 3.2.1–3.2.3).
- Prior approaches & limits:
  - GPT-3/PaLM/Gopher/OPT scale well but are predominantly closed or English-focused; datasets were often crawled and filtered with minimal provenance, reinforcing biases (Section 3.1 Motivation; Section 2.1).
  - Earlier work on instruction-tuning (e.g., T0; Section 3.1.4) showed zero-shot gains, but resources were not multilingual at similar scale or openly released.
- This paper’s positioning:
  - Creates a full-stack, open alternative: data governance (Section 3.1.1), multilingual corpus (ROOTS: Section 3.1; Table 1), scalable training recipe (Section 3.4), architecture tuned via ablations (Sections 3.2.1–3.2.3; Figure 5), multilingual prompting/finetuning (xP3 → BLOOMZ; Section 3.1.4; Section 4.7), and transparent evaluation across tasks (Section 4).

## 3. Technical Approach
Step-by-step overview from data to deployment:

- Data governance and curation (ROOTS; 1.61TB; 498 datasets)
  - Governance: Structured agreements with data providers/hosts, maintaining source-level traceability to respect privacy/IP and enable reproducibility (Section 3.1.1).
  - Composition: 46 natural languages and 13 programming languages; detailed makeup in Table 1; macro distribution shown in Figure 3.
  - Sources and processes:
    - Community-curated catalogues and hackathons (bottom-up sourcing; Section 3.1.2).
    - Pseudocrawl from CommonCrawl via OSCAR 21.09 (38% of corpus; Section 3.1.2).
    - GitHub code via BigQuery, deduplicated (Section 3.1.2).
  - Processing pipeline (Figure 2):
    - Quality filtering targeting “written by humans for humans,” with language-specific thresholds decided by fluent speakers (Section 3.1.3).
    - Near-duplicate removal and regex-based PII redaction, especially for OSCAR due to higher privacy risks (Section 3.1.3).

- Tokenization
  - Byte-level BPE with a large multilingual vocabulary of 250,680 tokens to avoid over-segmentation and support code/whitespace faithfully (Section 3.3).
  - No Unicode normalization to preserve generality (Section 3.3).
  - Pre-tokenizer regex avoids English-specific splitting and preserves whitespace/line breaks needed for code (Section 3.3).

- Model architecture and design choices
  - Architecture: Decoder-only Transformer trained with a causal language modeling objective (Equation (1) in Section 2.1; Section 3.2.2), as ablations showed best zero-shot generalization immediately after pretraining (Wang et al., 2022a; Section 3.2.2).
  - Scale: `BLOOM` (176B) with 70 layers, hidden size 14,336, 112 attention heads (Table 3); smaller variants (560M–7.1B) trained with the same recipe.
  - Two notable architectural deviations (Section 3.2.3; Figure 5):
    - `ALiBi` positional embeddings (“Attention with Linear Biases”): add a distance-based bias directly to attention scores; empirically smoother training and better downstream performance than learned or rotary embeddings, even at the same context length (Section 3.2.3).
    - Extra `Embedding LayerNorm`: layer norm immediately after embeddings to improve training stability in large-scale training (Section 3.2.3).
- Training and engineering at scale
  - Hardware: 384 NVIDIA A100 80GB GPUs on the Jean Zay supercomputer; ~3.5 months; 1,082,990 compute hours (Section 3.4.1).
  - Distributed training: `Megatron-DeepSpeed` with 3D parallelism—data (DP), tensor (TP), and pipeline parallelism (PP)—plus `ZeRO` optimizer (Stage 1) to shard optimizer states (Section 3.4.2; Figure 6).
    - 3D parallelism definition: combining DP (replicate model across devices), TP (split each layer across devices), PP (split layers vertically across devices).
    - Fused CUDA kernels for LayerNorm, Softmax chains, and fused bias+GeLU to reduce memory-bound overheads (Section 3.4.4).
  - Precision and stability: `bfloat16` mixed precision to avoid float16 overflow instabilities; some operations in float32 for stability (Section 3.4.3).
  - Throughput: up to 156 TFLOPs per A100—roughly 50% of the theoretical 312 TFLOPs peak (Section 3.4.2).
  - Training schedule: cosine LR decay with warmup; ~341B tokens (ROOTS size) + 25B repeated tokens based on updated scaling laws (Table 3; Section 3.5).
- Multitask prompted finetuning (BLOOMZ)
  - `xP3` dataset: multilingual prompts for 83 datasets across 46 languages and 16 tasks; matches ROOTS’ language distribution (Section 3.1.4; Figure 4).
  - Also create `xP3mt` via machine-translated prompts to test prompt-language importance (Section 3.1.4).
  - BLOOM variants instruction-tuned on xP3 → `BLOOMZ` (Section 3.5; Section 4.7).

- Licensing and release
  - Responsible AI License (RAIL) separating code (Apache 2.0) and model weights; includes explicit behavioral-use restrictions aligned with the model card (Section 3.6).

- Carbon accounting
  - Life Cycle Assessment (LCA)-style estimate: ~81 tons CO2eq including 14% manufacturing, 30% training energy, 55% idle cluster consumption (Section 3.5.1).
  - Training energy-emissions comparison (Table 4): BLOOM 25 tons CO2eq vs OPT 70 tons largely due to cleaner energy grid (France ~57 gCO2/kWh).

Definitions of select terms:
- `ALiBi` positional embeddings: add a linear distance penalty directly to attention scores, allowing better extrapolation and stable training (Section 3.2.3; Figure 5).
- `3D parallelism`: joint use of DP, TP, PP for scaling across hundreds of GPUs (Section 3.4.2; Figure 6).
- `ZeRO` optimizer (Stage 1): shards optimizer states across processes to reduce memory footprint (Section 3.4.2).
- `ROOTS`: the curated, multilingual pretraining corpus built under governance constraints (Section 3.1).
- `xP3`: multilingual prompt collection for instruction-tuning across tasks (Section 3.1.4).
- `BLOOMZ`: BLOOM models after multilingual instruction-tuning on xP3 (Section 4.7).

## 4. Key Insights and Innovations
- Open, governed, multilingual data pipeline at unprecedented scale
  - What’s new: Explicit governance agreements, source separation, and composite release to respect licensing/privacy while enabling reproducibility (Section 3.1.1; Figure 2).
  - Why it matters: Sets a blueprint for ethical large-scale data curation; ROOTS’ breadth (46 languages; 13 programming languages; Table 1) improves multilingual coverage beyond English-centric corpora.

- Architecture choices validated by ablations for zero-shot generalization
  - What’s new: Systematic comparison of encoder/decoder variants and objectives at smaller scales showed causal decoder-only is best for immediate zero-shot (Section 3.2.2), plus `ALiBi` and a post-embedding LayerNorm for stability (Section 3.2.3).
  - Why it matters: Reconciles research vs. practice for 100B+ scaling and informs future LLM training beyond replicating prior blueprints.

- Scalable, stable training recipe using bfloat16 + 3D parallelism
  - What’s new: End-to-end, openly documented engineering on 384 A100s, mixed precision with bfloat16 to avoid instabilities, and fused kernels to keep GPUs compute-bound (Sections 3.4.1–3.4.5).
  - Why it matters: Practical instructions and pitfalls (e.g., kernel launches, parameter group splits) lower barriers for future large-scale training.

- Multilingual instruction tuning (`BLOOMZ`) with `xP3` for robust zero-shot across languages
  - What’s new: xP3 extends English-focused prompt collections to 46 languages and 16 tasks (Section 3.1.4), enabling BLOOMZ to generalize zero-shot on non-English tasks (Figure 11).
  - Why it matters: Demonstrates that instruction-tuning gains transfer multilingually; closes gap with much larger monolingual instruction-tuned models on many tasks.

- Carbon footprint transparency and grid-aware emissions
  - What’s new: LCA-style accounting including idle emissions and equipment manufacturing (Section 3.5.1).
  - Why it matters: Highlights geographic grid intensity as a major factor; shows comparable energy but substantially lower emissions than OPT due to cleaner power (Table 4).

## 5. Experimental Analysis
- Evaluation setup (Section 4.1)
  - Prompt-based zero-/one-shot evaluation with multiple human-authored prompts per task using PromptSource; prompts not tuned using model feedback to simulate realistic user behavior.
  - Infrastructure: Prompted LM Evaluation Harness integrated with PromptSource (Section 4.1.2).
- Tasks, datasets, and metrics
  - NLU: SuperGLUE subset (Ax-b, Ax-g, BoolQ, CB, WiC, WSC); accuracy (Section 4.1.3).
  - MT: WMT’14 en↔fr, en↔hi; FLoRes-101 (spBLEU, SPM tokenization); DiaBLa dialog MT (BLEU, COMET) (Sections 4.3.1–4.3.3; Tables 6–7; 8a–8d).
  - Summarization: WikiLingua (ROUGE-2/L with multilingual SPM; Levenshtein); one-shot generation in source language (Section 4.4; Figure 9).
  - Code generation: HumanEval pass@k (k∈{1,10,100}) (Section 4.5; Table 9).
  - HELM 5-shot: English-only benchmarks for broader positioning (Section 4.6; Figure 10).
  - Instruction-tuning: BLOOMZ vs pretrained models on multilingual NLI, coreference, and sentence completion (Section 4.7; Figure 11).
  - Embeddings: MTEB tasks with SGPT-BLOOM bi-encoders; accuracy/spearman (Section 4.8; Table 10).
  - Probing: Morphosyntactic feature classification across 17 languages; F1; correlations with language family/script and dataset size (Section 4.9; Table 12; Figure 12; Table 13).
  - Bias: Multilingual CrowS-Pairs; model preference for stereotyped vs. non-stereotyped statements (Section 4.10; Figure 13; Table 14).

- Main quantitative results and comparisons
  - SuperGLUE (zero-/one-shot; Figure 7; Figure 8):
    - Zero-shot: BLOOM competitive with OPT-175B on some tasks (e.g., BoolQ/CB) and improves notably moving to 1-shot, matching or exceeding OPT on Ax-b, CB, WSC, and WiC in 1-shot.
    - Scaling: Meaningful signal emerges above ~2B parameters; BLOOM-176B ≈ OPT-175B across tasks (Figure 8).
  - Machine Translation:
    - WMT’14 (Table 6), BLOOM-176B, 1-shot, best prompts:
      - en→fr: 36.39 BLEU (prompt “a_good_translation-source+target”); fr→en: 36.56 BLEU.
      - en→hi: up to 14.49 BLEU; hi→en: up to 25.80 BLEU. Zero-shot performance is much lower due to language/output drift and overgeneration, which 1-shot examples mitigate (Section 4.3.1).
    - DiaBLa dialog MT (Table 7): Using the previous utterance as the 1-shot example slightly improves BLEU (e.g., en→fr 38.5 vs 37.6) but COMET mixed; custom truncation helps isolate translation quality from overgeneration (Section 4.3.2).
    - Flores-101 (Table 8):
      - High-resource pairs: BLOOM 1-shot often comparable to or better than M2M-100 (e.g., fr→en 45.6 spBLEU vs 37.2 for M2M).
      - High→mid-resource: strong (e.g., en→fr 45.0; en→vi 28.5; id→en 43.2).
      - Low-resource: mixed; strong for some (bn→en 29.9), weak for Swahili↔Yoruba where ROOTS has <50k tokens each.
  - Summarization (Figure 9): Across nine languages, BLOOM family outperforms OPT-175B in one-shot ROUGE-2; performance increases with parameter count.
  - Code generation (Table 9):
    - BLOOM-176B: pass@1/10/100 = 15.52% / 32.20% / 55.45%.
    - Comparable to GPT-NeoX-20B at pass@1 but behind code-specialized Codex-12B (28.81/46.81/72.31).
    - BLOOMZ does not improve HumanEval notably—xP3 contains little pure code completion.
  - Embeddings (Table 10):
    - SGPT-BLOOM-7.1B-msmarco achieves highest or near-highest scores on many languages for classification (e.g., Arabic 59.25% acc; French 66.95%) and STS (e.g., French 80.38 spearman), but the model is large (7.1B) compared to popular multilingual MiniLM/MPNet.
  - HELM (Figure 10): BLOOM is roughly on par with previous-generation English-only models (e.g., GPT-3 davinci v1, J1-Grande v1) but behind newer instruction-tuned monolingual models; fairness is relatively good, calibration mediocre, toxicity slightly above average.
  - Probing (Table 12; Figure 12; Table 13):
    - BLOOM and BLOOM-1.7B outperform TF-IDF baselines; strongest on Mood/Person across languages.
    - BLOOM-1.7B shows higher correlation with pretraining dataset sizes (less general on under-resourced languages) whereas BLOOM-176B appears more stable across languages (Table 13).
  - Bias (Figure 13; Table 14):
    - Overall accuracy near 50% on CrowS-Pairs in English (49.78*) and French (50.61*); distribution across categories is relatively homogeneous, with some significant deviations from 50% in both directions (asterisks mark p<.05).

- Convincingness and robustness
  - The evaluation suite is broad, multilingual, and includes generation (MT, summarization), discrimination (SuperGLUE), code, embeddings, probing, and bias—consistent with the paper’s goals.
  - Key ablations (architecture/objective) were done at 1.3B–6.7B scales (Sections 3.2.1–3.2.2), a practical compromise; results plausibly transfer but phase transitions beyond 6.7B may introduce differences.
  - Failure cases documented: overgeneration and wrong-language outputs in zero-shot MT; mixed DiaBLa COMET vs BLEU; limited gains on code after instruction-tuning; weak performance for very low-resource languages absent enough pretraining data.

> “The two major problems [in zero-shot MT]… are (i) over-generation and (ii) not producing the correct language… both… greatly improved as the number of few-shot examples is increased.” (Section 4.3.1; Table 6)

## 6. Limitations and Trade-offs
- Data balance and low-resource languages
  - ROOTS coverage is broad but uneven; extreme low-resource languages (e.g., Swahili, Yoruba at <50k tokens) show weak MT performance (Table 8a; Section 4.3.3).
  - Despite governance and PII redaction, regex-based redaction for OSCAR may have false positives/negatives (Section 3.1.3).
- Prompt sensitivity and zero-shot variability
  - Zero-shot performance depends strongly on prompt wording; average-over-prompts often hovers near chance on some SuperGLUE tasks (Figure 7), underscoring the need for either prompt engineering or instruction-tuning.
- Compute and engineering complexity
  - Training required ~1.08M GPU-hours on 384 A100s for months (Section 3.4.1); although open, reproducing the full 176B model remains out of reach for many groups.
  - Engineering details (e.g., fused kernels, deadlock fixes, parameter group splitting) add operational complexity (Section 3.4.5).
- Objective and architecture scope
  - The study does not explore mixture-of-experts or state-space models at scale (Section 3.2.1: “Out-of-scope Architectures”); choices are justified but leave open whether these alternatives could yield better multilingual scaling.
- Evaluation constraints
  - HELM is English-only; instruction-tuned results stress multilingual tasks but code instruction-tuning is limited; summarization uses ROUGE which is known to under-represent quality, especially across languages (Section 4.4).
- Ethical scope
  - Bias evaluation (CrowS-Pairs) is preliminary and limited to English and French; broader multilingual harms (dialect variation, marginalized varieties) need dedicated resources (Section 4.10).

## 7. Implications and Future Directions
- Landscape impact
  - Provides the first fully open, multilingual 100B+ LLM with transparent data pipeline, training recipe, and instruction-tuned variants—setting a new standard for reproducibility and responsible release.
  - Demonstrates that multilingual instruction-tuning (xP3 → BLOOMZ) yields strong zero-shot gains across many non-English tasks (Figure 11), moving beyond English-centric instruction-tuning.
- Research avenues
  - Data: Expand governed, high-quality data for under-represented languages; targeted augmentation and active curation could address weak low-resource performance (Table 8a).
  - Methods: Explore alternative architectures (MoE; state-space) and retrieval-augmented training for multilingual settings; study phase transitions beyond 6.7B in ablations.
  - Training: Longer contexts with `ALiBi` extrapolation; investigate if the embedding LayerNorm remains necessary with bfloat16 across different initializations (Section 3.2.3).
  - Evaluation: Develop multilingual generation metrics beyond ROUGE and broader bias/harms audits across more languages and varieties (Section 4.10).
  - Instruction tuning: Enrich xP3 with code-completion-style tasks to improve code generation after finetuning (Table 9); study prompt-language vs task-language interactions (xP3 vs xP3mt, Section 3.1.4).
- Practical applications
  - Multilingual assistants and content generation, cross-lingual retrieval and summarization, low-resource MT bootstrapping, and domain adaptation via the open weights and license—especially for public sector, academia, and SMEs who previously lacked access.

> “We publicly release our models and code under the Responsible AI License.” (Abstract; Section 3.6)

> “BLOOM achieves competitive performance on a wide variety of benchmarks, with stronger results after undergoing multitask prompted finetuning.” (Abstract; Section 4.7; Figure 11)

Overall, BLOOM provides a comprehensive, reproducible blueprint—data governance, model/training recipe, instruction-tuning, and evaluations—that materially broadens who can build and study very large multilingual language models, while clearly documenting limitations and trade-offs to guide future work.
