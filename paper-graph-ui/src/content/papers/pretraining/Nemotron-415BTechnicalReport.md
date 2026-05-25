# Nemotron-4 15B Technical Report

**ArXiv:** [2402.16819](https://arxiv.org/abs/2402.16819)

## 🎯 Pitch

Nemotron-4 15B introduces a 15-billion-parameter multilingual language model trained on a massive 8 trillion-token dataset, designed to fit on a single NVIDIA A100 or H100 GPU. By prioritizing data scale over parameter count, it achieves state-of-the-art multilingual performance in its size class—outperforming even larger or specialized models—while maintaining competitive English and code abilities. This marks an important advance in practical, deployable AI: Nemotron-4 15B substantiates that data scaling can unlock broad capabilities without the serving costs and complexity of massive models, making high-quality language AI more accessible for research and real-world applications.

---

## 1. Executive Summary
Nemotron-4 15B is a 15‑billion‑parameter multilingual language model trained on an unusually large 8 trillion tokens that aims to maximize capability while remaining small enough to run on a single NVIDIA A100/H100 GPU. Its main significance is a data-first recipe that delivers state-of-the-art multilingual accuracy in its size class and competitive English and code performance, validated across diverse benchmarks (Figures 1; Tables 3–10).

## 2. Context and Motivation
- Problem addressed
  - How to get strong general-purpose and multilingual performance from a model small enough for practical single‑GPU deployment. The work targets the compute–data trade-off highlighted by Chinchilla-style scaling: for a fixed compute budget, training on more tokens can beat simply increasing parameters.
- Why this matters
  - Practical impact: A 15B model is easier to deploy with lower latency and cost than larger models, making on‑prem and edge inference more feasible.
  - Scientific impact: Tests whether “data scaling” (8T tokens) can compensate for smaller parameter counts and unlock multilingual and coding competence.
- Prior approaches and gaps
  - Parameter-first scaling (e.g., >30B models) improves accuracy but raises serving cost.
  - Recent models (e.g., LLaMA‑2 13B/34B, Mistral 7B, Qwen 14B, Gemma 7B) do well in English/code but multilingual strength typically requires much larger models or multilingual-specialized training.
  - Gap: Is it possible for a mid-size general-purpose model to reach or beat the multilingual performance of larger or specialized systems?
- Positioning relative to existing work
  - The model is trained on vastly more tokens than similarly sized public models (8T tokens total; Section “Data”). It claims best-in-class multilingual ability, outperforming even some multilingual-specialized or much larger models (XGLM 7.5B, mGPT 13B, PaLM 62B-Cont) on key benchmarks (Tables 7–10).

## 3. Technical Approach
This is an end-to-end recipe that combines (1) a standard but efficiency-minded architecture, (2) very large, carefully filtered and balanced training data, (3) large-scale distributed training with high utilization, and (4) a short “continued training” phase that shifts data emphasis and learning rate schedule.

- Model architecture (Section 2; Table 1)
  - Decoder-only Transformer with causal attention. Key hyperparameters: 32 layers, hidden size 6144, 48 attention heads, 8 key–value heads, context length 4096, vocabulary size 256k. Total parameters: ~15.7B (3.2B embeddings, 12.5B non-embedding).
  - `RoPE` (rotary position embeddings): encodes relative positions by rotating query/key vectors so the model extrapolates better with long contexts.
  - `GQA` (grouped query attention): reduces the number of key–value (KV) heads while keeping many query heads, cutting memory and speeding decoding by sharing KV across queries (Section 2).
  - MLP uses squared ReLU activation; no bias terms; dropout=0; untied input–output embeddings. These choices trade simplicity and throughput for regularization; the low dropout is compensated by massive training data.
  - Tokenizer: SentencePiece BPE with byte-level fallback; preserves whitespace and splits numbers into digits; vocabulary 256k. During tokenizer training, non‑English data is upsampled to improve coverage of lower-resource languages (Section “Data”).

- Training data and curation (Section “Data”; Figures 2–4)
  - 8T tokens total, split: 70% English natural text, 15% multilingual (53 languages), 15% code (43 programming languages).
  - English sources include web crawl, books, news, academic papers, encyclopedias; Figure 2 shows a heavy “web crawl” component plus other curated categories.
  - Code mix spans popular and long‑tail languages; Figure 3 lists 43 languages and their sampling rates (percent of code tokens).
  - Multilingual mix spans 53 languages with careful sampling weights (Figure 4).
  - Quality controls: exact and near‑deduplication at document level; language‑model scoring and heuristics to remove low‑quality text (Wenzek et al., 2019; Rae et al., 2022; Raffel et al., 2020).

- Distributed training implementation (Pre-training section; Table 2)
  - Hardware: 384 DGX H100 nodes (8× H100 80GB per node) with NVLink/NVSwitch (900 GB/s GPU–GPU aggregate) and 400 Gbps InfiniBand per node.
  - Parallelism: 8‑way tensor parallelism + data parallelism; optimizer sharded across data-parallel replicas.
  - Batch ramp schedule (Table 2):
    - Stage 1: DP=96, 768 GPUs, batch=384, MFU=34.3%, 200B tokens, 0.8 d.
    - Stage 2: DP=192, 1,536 GPUs, batch=768, MFU=33.3%, 200B tokens, 0.4 d.
    - Stage 3: DP=288, 2,304 GPUs, batch=1,152, MFU=30.5%, 7.6T tokens, 11.9 d.
  - Total training time ≈13 days. `MFU` (Model FLOP/s Utilization) measures how close the training run is to hardware peak throughput.

- Continued training (Section “Continued Training”)
  - Short extra phase (small token count relative to 8T) with the same loss but altered inputs and schedule:
    - Data distribution 1: reweights toward higher‑quality versions of sources already used in pretraining.
    - Data distribution 2: introduces a small number of “benchmark-style” alignment examples and upweights data for areas where the model underperforms.
  - Learning rate schedule is adjusted so decay is steeper (larger slope) even if the absolute LR magnitude is low, helping the model “pivot” toward newly emphasized content without catastrophic forgetting. This is closer to “curriculum at the end” than to standard fine-tuning.

Why these choices?
- High token count vs. parameter count follows Chinchilla: with a fixed compute budget, more data can beat a bigger model (Section 1).
- GQA and 256k vocab cut inference memory and improve multilingual/code tokenization coverage—both important for single‑GPU serving and multi‑script text.
- Continued training with benchmark-style examples specifically targets measured weaknesses and evaluation formats (Section “Continued Training”).

## 4. Key Insights and Innovations
- Data-first scaling at mid-size yields outsized multilingual gains (fundamental insight)
  - Training a 15B model on 8T tokens leads to multilingual performance that surpasses larger general models and some multilingual-specialized ones:
    - XCOPA 4-shot average 68.9 vs XGLM 7.5B (61.4) and mGPT 13B (56.0) (Table 7).
    - TyDiQA 1-shot average 50.5 vs PaLM 62B-cont 45.7 (Table 8).
    - MGSM 8-shot English-CoT 41.3 vs PaLM 62B-cont 32.0 and Mistral 7B 21.8 (Table 9).
    - FLORES Chinese→X average 23.2 spBLEU vs LLaMA‑2 13B 12.2 and Baichuan‑2 13B 16.1 (Table 10).
  - Significance: Shifts the efficiency frontier—strong multilingual capabilities no longer require very large models if the training corpus is scaled carefully.

- Balanced multilingual and code sampling with tokenizer upsampling for low-resource languages (incremental but impactful)
  - The training mix (Figures 3–4) and tokenizer choices (Section “Data”) improve coverage for less common scripts and languages, reducing fragmentation and unknown tokens, which is key for XCOPA/MGSM/TyDiQA improvements.

- Efficient training + inference design (incremental engineering advance)
  - Use of GQA, large-vocab tokenizer, and no-bias/zero-dropout layers to reduce memory footprints and simplify serving, while large-batch distributed training achieves days‑scale pretraining (Table 2). Practical impact: a highly capable 15B model that fits on a single A100/H100 GPU for inference.

- Continued training curriculum with data distribution and LR schedule shift (novel recipe detail)
  - The two‑distribution strategy plus steeper LR decay (Section “Continued Training”) is a concrete mechanism to translate raw pretraining ability into higher benchmark readiness without full instruction/RLHF pipelines. This is not standard for all open models and likely contributes to the strong few‑shot/zero‑shot results.

## 5. Experimental Analysis
- Evaluation setup (Section 3)
  - English reasoning (zero‑shot): SIQA, ARC‑easy/challenge, PIQA, Winogrande, HellaSwag (Table 3).
  - Aggregated benchmarks: MMLU (5‑shot) and BigBench Hard (3‑shot) (Table 4).
  - Math: GSM8K (8‑shot, maj@1—majority vote selection among samples) (Table 5).
  - Code: HumanEval (0‑shot), MBPP (3‑shot), MultiPL‑E (0‑shot, 11 languages) (Tables 5–6).
  - Multilingual:
    - Classification: XCOPA (0‑shot and 4‑shot) (Table 7).
    - QA generation: TyDiQA‑GoldP (1‑shot) (Table 8).
    - Reasoning: MGSM (8‑shot, “English chain‑of‑thought” prompts) (Table 9).
    - Translation: FLORES‑101 (8‑shot, spBLEU) for Chinese→{EN, FR, ES, AR, RU, JA, DE} (Table 10).
  - Tooling: LM‑Evaluation Harness for English commonsense tasks (Section 3.1).

- Main quantitative results
  - English commonsense (0‑shot, Table 3):
    - Average 73.4 for `Nemotron-4 15B`, beating LLaMA‑2 34B (71.1), Mistral 7B (70.4), Gemma 7B (70.2).
    - Per‑task highlights: PIQA 82.4, HellaSwag 82.4, Winogrande 78.0.
  - Aggregated benchmarks (Table 4):
    - BBH: 58.7 (best at its scale; higher than Qwen 14B 53.4, Gemma 7B 55.1, LLaMA‑2 34B 44.1). The report notes this even exceeds LLaMA‑2 70B (51.2).
    - MMLU: 64.2, competitive with Gemma 7B (64.3) and Qwen 14B (66.3). Category breakdown (Table 11): Humanities 69.2, Social Sciences 74.1, STEM 53.4, Other 67.5.
  - Math + Code (Table 5):
    - GSM8K: 46.0—comparable to Gemma 7B (46.4) but behind Qwen 14B (60.1) and Baichuan‑2 13B (52.8).
    - HumanEval: 31.6; MBPP: 40.6—on par with Qwen 14B (32.2/40.8), below Gemma 7B (32.3/44.4), above LLaMA‑2 13B/34B.
    - MultiPL‑E (Table 6): average across 11 languages is 24.5, slightly higher than StarCoder 15B (24.2) and Mistral 7B (23.6). Stronger on low‑resource languages such as Scala (27.3 vs StarCoder 27.6 vs Mistral 22.2) and R (18.6 vs StarCoder 15.5 vs Mistral 11.8); competitive in C++ (35.4).
  - Multilingual (Tables 7–10):
    - XCOPA (classification): 0‑shot avg 59.5 (vs XGLM 55.6; mGPT 56.1); 4‑shot avg 68.9 (vs XGLM 61.4; mGPT 56.0). Notable high scores include ID=79.6 and IT=79.2 in 4‑shot (Table 7).
    - TyDiQA 1‑shot: avg 50.5, beating PaLM 62B‑cont 45.7 (Table 8).
    - MGSM 8‑shot English‑CoT: avg 41.3, well above PaLM 62B‑cont 32.0 and Mistral 7B 21.8; high performance in ES=50.0, DE=46.8, FR=46.0; lower in SW=16.0 (Table 9).
    - FLORES Chinese→X: avg 23.2 spBLEU (ZH‑EN=34.0) vs LLaMA‑2 13B 12.2 and Baichuan‑2 13B 16.1 (Table 10).

- Do the experiments support the claims?
  - The assertion of “best multilingual at its scale” is strongly supported: on XCOPA, TyDiQA, MGSM, and FLORES, Nemotron‑4 15B wins convincingly versus similarly sized general models and even against larger or multilingual‑focused baselines (Tables 7–10).
  - English and code: results are competitive but not dominant. It leads or ties on some English reasoning tasks (Table 3) and BBH (Table 4), but trails top math (GSM8K) and some code baselines (Table 5). The MultiPL‑E results (Table 6) suggest breadth across languages and relative strength on long-tail languages.
  - The claim of being superior to LLaMA‑2 34B across English evaluations is broadly reflected in Tables 3–5, especially BBH (58.7 vs 44.1).

- Ablations and robustness
  - There are no formal ablation tables isolating the effects of data mixture choices, tokenizer size, or the continued‑training phase. However, Section “Continued Training” details the mechanism (two distributions + steeper LR decay) intended to improve benchmark performance and weak areas.
  - Failure cases are not cataloged; some weaknesses can be inferred: STEM MMLU (53.4; Table 11) and GSM8K (46.0; Table 5) lag the top of class, and MGSM shows variability (e.g., Swedish 16.0; Table 9).

- Caveats in comparisons
  - Some baseline numbers are taken from external reports, which can introduce differences in prompt templates or decoding settings (noted in Sections 3 and specific table captions; e.g., Mistral MBPP and GSM8K settings differ—Table 5 caption). This does not overturn the multilingual leadership but is worth noting for English/code head‑to‑heads.

## 6. Limitations and Trade-offs
- Compute and data demands
  - Training uses 2,304 GPUs for the main phase over ~12 days (Table 2) and requires an 8T-token curated corpus. This is beyond the reach of most practitioners, even if inference is affordable.
- Limited transparency on ablations
  - The report does not provide controlled ablations for: token count vs. model size, effect sizes of GQA, tokenizer vocabulary changes, or the two‑phase continued training. Without these, it’s hard to attribute gains precisely.
- Benchmark‑style data in continued training
  - Section “Continued Training” includes “a small number of benchmark‑style alignment examples.” While this is clearly disclosed, it risks tailoring to evaluation formats. Generalization to unseen task formats is not directly measured in the report.
- Math/STEM headroom
  - STEM category in MMLU (53.4; Table 11) and GSM8K (46.0; Table 5) trail top open models (e.g., Qwen 14B at 60.1 on GSM8K), suggesting room for domain‑specific math data or reasoning enhancements.
- Context length and tokenizer trade-offs
  - Context is 4,096 tokens (Table 1). For tasks needing longer contexts, the model may need extrapolation or fine‑tuned extensions. The 256k vocabulary improves multilingual coverage but increases embedding parameters (3.2B), a large fraction of total size, which impacts memory and training speed.

## 7. Implications and Future Directions
- Field impact
  - This work strengthens the case for data‑centric scaling at mid‑size. It demonstrates that with careful multilingual and code mixtures plus end‑phase curriculum, a 15B model can outperform larger models on multilingual understanding and generation. This offers a practical alternative to building only very large models for broad capability.
- Follow-up research enabled/suggested
  - Controlled ablations on the continued training recipe: quantify the impact of high-quality reweighting vs. benchmark-style examples vs. LR schedule.
  - Math/STEM enhancements: introduce targeted datasets and reasoning techniques (e.g., verifier training or tool use) to close the GSM8K/STEM gap.
  - Longer-context adaptations: evaluate and extend context beyond 4k tokens with scalable positional methods (e.g., RoPE extrapolation or learned interpolation).
  - Safety and alignment: the report focuses on capability; further investigation into safety, toxicity, and bias across languages is necessary.
  - Data mixture optimization: formalize sampling strategies (Figures 3–4) as an optimization problem and study cross-lingual transfer effects.
- Practical applications
  - Multilingual assistants and knowledge workers that must run on a single H100/A100 GPU.
  - Cross‑lingual QA and reasoning (TyDiQA, MGSM) in international products.
  - Broad code assistance across many languages, especially low‑resource ones (Table 6), in enterprise developer tooling.

> Core takeaway: a carefully curated 8T‑token corpus, plus an efficiency‑oriented 15B architecture and a short targeted continued‑training phase, produces a general-purpose model with best‑in‑class multilingual performance for its size (Tables 7–10) and competitive English/code capability (Tables 3–6), all while remaining deployable on a single high‑end GPU.
