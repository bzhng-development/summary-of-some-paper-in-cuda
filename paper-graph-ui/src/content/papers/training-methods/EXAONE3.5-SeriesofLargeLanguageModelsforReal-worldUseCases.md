# EXAONE 3.5: Series of Large Language Models for Real-world Use Cases

**ArXiv:** [2412.04862](https://arxiv.org/abs/2412.04862)

## 🎯 Pitch

EXAONE 3.5 introduces a family of instruction-tuned bilingual (Korean–English) large language models, available in three scalable sizes (32B, 7.8B, 2.4B) and optimized for exceptional real-world instruction following and robust long-context (32K tokens) performance. By achieving state-of-the-art or competitive results across real-world, long-context, and general-domain benchmarks—while maintaining lower training compute than many peers—EXAONE 3.5 provides a uniquely versatile and efficient backbone for both research and deployment, meeting the practical needs of modern AI assistants in diverse environments. This makes it especially impactful for scenarios requiring strong instruction adherence, retrieval-augmented workflows, and deployment flexibility from edge devices to servers.

---

## 1. Executive Summary
EXAONE 3.5 is a family of instruction-tuned bilingual (Korean–English) large language models in three sizes (`32B`, `7.8B`, `2.4B`) optimized for real-world instruction following and long-context use (32K tokens). With a comparatively efficient training budget, the models deliver state-of-the-art or competitive results on seven real-world-use benchmarks and four long-context/RAG benchmarks, while remaining competitive on nine general-domain tasks (Tables 5–8; Sec. 3).

## 2. Context and Motivation
- Problem addressed
  - Many users need models that (a) follow complex instructions in real settings, (b) process long contexts (e.g., full documents, large RAG contexts), and (c) are available at multiple sizes to fit deployment constraints (on-device to server-scale). This work targets all three needs at once (Sec. 1).
- Why it matters
  - Real-world assistants must handle diverse instructions, integrate retrieved evidence, and respect user constraints (e.g., output format, language). They often operate over long contexts in RAG systems. Providing strong small models is also important where GPUs are limited (Sec. 1).
- Prior approaches and gaps
  - Many open models either emphasize general benchmarks or lack long-context reliability; some have limited bilingual strength, and few provide strong small-size models tuned for long-context use. Moreover, compute cost can be prohibitive (Sec. 2.2.3; Table 3).
- Positioning relative to existing work
  - EXAONE 3.5 extends the EXAONE 3.0 line with:
    - Three sizes, all supporting 32K tokens (Table 1).
    - Emphasis on real-world instruction following (seven benchmarks) and long-context usage (four benchmarks).
    - Competitive performance at lower training compute than some peers; e.g., Qwen 2.5 32B uses roughly 2.77× more training compute by the size×tokens proxy (Sec. 2.2.3; Table 3).

## 3. Technical Approach
This section unpacks the architecture, training pipeline, data safeguards, and alignment process.

- Model architecture (Table 1; Sec. 2.1)
  - Transformer, decoder-only, pre-norm, `SwiGLU` feed-forward.
    - `SwiGLU`: a gated activation function designed to improve optimization and expressivity in feed-forward layers.
  - `GQA` (Grouped Query Attention): shares key/value projections among groups of attention heads to reduce memory and improve efficiency while retaining most multi-head benefits.
  - `RoPE theta = 1,000,000`: RoPE is rotary positional embedding; the `theta` controls the base frequency. Larger `theta` helps stabilize very long positions, enabling 32K-token contexts without heavy degradation.
  - Tokenizer: `BBPE` (byte-level BPE) with 102,400 vocab items, roughly 50% Korean and 50% English coverage, improving bilingual tokenization efficiency.
  - Sizes
    - `32B`: 64 layers, d_model 5,120, 40 heads (8 KV heads), max length 32,768.
    - `7.8B`: 32 layers, d_model 4,096.
    - `2.4B`: 30 layers, d_model 2,560; uses tied embeddings to save parameters.

- Two-stage pre-training with long-context extension (Sec. 2.2; 2.2.1)
  - Stage 1: train on a large, diverse web corpus to build general capability.
  - Stage 2: target weaknesses revealed by evaluation—especially long-context skills.
    - Long-context fine-tuning uses the positional-interpolation style method (cited as [7]) and trains on full-length documents without chunking.
    - To avoid catastrophic forgetting (forgetting prior knowledge during long-context fine-tuning), a `replay-based method` is used: Stage-1 data is replayed (mixed back) during Stage-2.
      - Replay means interleaving a portion of original short-context data to maintain competencies learned previously.

- Training budget (Table 2; Sec. 2.2)
  - Training tokens: `32B` uses 6.5T tokens; `7.8B` uses 9T; `2.4B` uses 6.5T.
  - The 32B model’s compute (by size×tokens) is lower than some peers (Table 3), yet it remains competitive, especially on real-world and long-context tasks.

- Decontamination (Sec. 2.2.2; Appendix C, Figure 4; Table 10)
  - To reduce benchmark leakage, a strict substring-based filter is applied.
  - Process:
    - Normalize benchmark test items; extract all unique substrings using a sliding window `S=50`, stride 1.
    - For each training example, randomly sample `N=10` substrings and check membership in the test substrings pool.
    - If matches appear, remove the contaminated training item.
  - The method is simple but strict; examples of removed items are shown in Table 10.

- Supervised fine-tuning (SFT) with instruction evolution (Sec. 2.3.1; Figure 1)
  - From ~8M web pages, a domain taxonomy is extracted (Math/Arts/Sciences, etc.). The team generates instruction–response pairs guided by this taxonomy.
  - An instruction evolution (akin to Evol-Instruct) increases difficulty and variety, producing a curriculum that spans simple to complex tasks (Figure 1).

- Preference optimization with DAAs (direct alignment algorithms) (Sec. 2.3.2; Figure 2)
  - `DAA` refers to algorithms that train directly from preference pairs rather than scalar rewards. Two specific methods are cited:
    - `DPO` (Direct Preference Optimization).
    - `SimPO` (Simple Preference Optimization), which removes the need for a fixed reference model in the reward.
  - Preference data creation (Figure 2, top):
    - For each prompt `x`, sample multiple responses from multiple models.
    - Rank them using a reward model, pick best `y_w` and worst `y_l`.
    - Validate rankings by a second reward model; discard low-agreement pairs.
  - Staged alignment (Figure 2, bottom):
    - Start from the SFT model `M0`, then apply DAAs in multiple stages to obtain `M1`, `M2`.
    - Staging mitigates over-optimization (reward hacking) observed in preference-training regimes.

- Ethics/compliance and license (Sec. 2.4; Appendix B)
  - Data undergoes compliance reviews (copyright, privacy).
  - License is research-only (non-commercial). Notably, Appendix B Sec. 4.2 assigns ownership of generated `Output` to the Licensor, permitting Licensee use and distribution only for research purposes. This affects practical deployment (discussed in Sec. 6).

## 4. Key Insights and Innovations
- Long-context capability across all sizes with simple but effective ingredients
  - What’s new: all three models support 32K tokens via a combination of RoPE scaling (`theta=1e6`) and Stage-2 long-context fine-tuning on full documents (Sec. 2.1–2.2.1).
  - Why it matters: long-context strength is central to modern RAG systems. Figure 3 shows near-perfect “Needle-in-a-Haystack” retrieval across lengths and positions in both English and Korean.

- Replay-based long-context extension to reduce forgetting
  - What’s different: rather than fine-tuning only on long sequences, the team mixes Stage-1 data back in (replay) during Stage-2 (Sec. 2.2.1), explicitly targeting catastrophic forgetting.
  - Significance: helps preserve general skills, contributing to strong real-world and general results despite the focus on long context.

- Instruction data built from a knowledge taxonomy plus instruction “evolution”
  - What’s different: SFT data is derived from taxonomy-extracted knowledge, then “evolved” to systematically grow complexity (Figure 1; Sec. 2.3.1).
  - Significance: produces robust instruction following, reflected in top-tier MT-Bench/Arena-Hard and Korean benchmarks (Table 6).

- Staged preference optimization with dual reward-model validation
  - What’s different: preference pairs are filtered by agreement of two reward models, and alignment proceeds in stages (M0→M1→M2) to avoid over-optimization (Sec. 2.3.2; Figure 2).
  - Significance: improves real-world instruction adherence (e.g., IFEval prompt-strict) and win rates (Arena-Hard) without collapsing style diversity.

- Efficient training relative to peers with competitive outcomes
  - Evidence: Table 3 shows the 32B model’s size×tokens compute is 1.00 (baseline), while Qwen 2.5 32B is 2.77 and Gemma 2 27B is 1.69. Despite this, EXAONE 32B wins the Real-world and Long-context category averages (Table 5).

## 5. Experimental Analysis
- Evaluation setup (Sec. 3; Table 4)
  - Three categories:
    - Real-world Use: MT-Bench, LiveBench (2024-08-31), Arena-Hard v0.1, AlpacaEval 2.0 Length-Controlled (LC), IFEval (prompt-level strict), plus Korean KoMT-Bench and LogicKor.
    - Long Context: Needle-in-a-Haystack (NIAH) in English/Korean; LongBench (English); LongRAG (extended with unanswerable cases) and Korean Ko-LongRAG (in-house) plus Ko-WebRAG (in-house real web RAG).
    - General Domain: nine benchmarks, all in zero-shot, with explicit prompts shared in Appendix D.3. Greedy decoding with max generation length 2,048 for these tasks.
  - LLM-as-a-judge is used for several benchmarks with GPT-4 variants (e.g., GPT-4o-2024-08-06, GPT-4-1106-preview). A footnote notes replacement of the original judge for better separability (Table 4, footnote; Sec. 3.3).

- Main quantitative results
  - Category-level macro averages (Table 5):
    > Real-world Use Cases: `EXAONE 32B: 74.3` vs `Qwen 2.5 32B: 69.8`.  
    > Long Context: `EXAONE 32B: 71.1` vs `Qwen 2.5 32B: 66.9`.  
    > General Domain: `EXAONE 32B: 74.8` vs `Qwen 2.5 32B: 78.7`.
    - Takeaway: EXAONE wins Real-world and Long-context, slightly trails on General Domain.
  - Real-world breakdown (Table 6):
    - `EXAONE 32B` tops MT-Bench `8.51`, Arena-Hard win rate `78.6%`, IFEval `81.7%`, KoMT-Bench `8.05/10`, LogicKor `9.06/10`.
    - It trails Qwen 2.5 32B on LiveBench (43.0 vs 50.6). On AlpacaEval 2.0 LC, EXAONE 32B wins 60.6% vs Qwen 41.0%.
    - The `2.4B` model is notably strong: average `61.1`, beating Qwen 2.5 `3B` (`44.5`) and Llama 3.2 `3B` (`36.7`).
  - Long-context tasks (Figure 3; Table 7; Appendix D)
    - NIAH heatmaps show near-perfect retrieval over up to 32K tokens in both languages.
    - LongBench: EXAONE 32B is comparable to Qwen 2.5 32B (49.2 vs 49.1), behind Command R 32B (50.9).
    - Extended LongRAG (Table 14):
      > `EXAONE 32B` shows high “answerable” accuracy (NQ 73.6, Hotpot 81.8) but low “unanswerable” accuracy (NQ 35.3, Hotpot 26.4), yielding totals 68.3 and 66.9.  
      > `Qwen 2.5 32B` is better on unanswerable (NQ 61.2, Hotpot 70.6) but lower on answerable (NQ 62.3, Hotpot 62.9), totals 62.1 and 65.0.  
      - Net effect: EXAONE 32B averages 67.6 vs Qwen 63.6 on LongRAG.
    - Ko-LongRAG (Table 15): EXAONE leads strongly (e.g., `32B` average 85.3 vs Qwen 73.5).
    - Ko-WebRAG (Table 7): EXAONE 32B 82.3 vs Qwen 81.3.
  - General-domain (Table 8):
    - `EXAONE 32B` average `74.8` vs `Qwen 2.5 32B` `78.7`. On math GSM8K CoT: both ≈92; on MATH CoT: 70.5 vs 76.5; coding HumanEval: 87.2 vs 89.0; MBPP: 81.8 vs 88.9; knowledge MMLU CoT: 78.3 vs 81.4; KMMLU CoT (Korean): 57.0 vs 62.1.
    - Remarkably, the `2.4B` model tops its size class average (`63.3`) beating Qwen 2.5 `3B` (`62.1`) and Llama 3.2 `3B` (`54.9`).
  - Safety/harmlessness (Table 9; Sec. 4.3)
    - On the Korean Trustworthiness Benchmark (10,000 items), EXAONE 32B reaches `87.1%` overall. Subcategory accuracies are detailed (e.g., Hate-related: `90.0%` for 32B).

- Do the experiments support the claims?
  - Real-world instruction following: Yes; strong wins on MT-Bench, Arena-Hard, IFEval, and Korean LLM-as-a-judge tasks (Table 6).
  - Long-context capability: Yes; NIAH heatmaps (Fig. 3) are near-perfect; EXAONE leads category averages (Table 7) and dominates Korean long-context RAG.
  - Efficiency: Table 3’s compute comparison plus outcomes in Tables 5–8 support competitive performance at lower training compute for the 32B.
  - Caveat: several results rely on LLM-as-a-judge (MT-Bench, Arena-Hard, LongRAG variants, Ko-*RAG). The paper partially addresses judge sensitivity (e.g., note on judge separability in Table 4), and provides ground-truth metrics where possible (LiveBench, IFEval, NIAH, portions of LongBench). Still, judge choice can affect rankings.

- Ablations and robustness checks
  - The report presents detailed benchmark coverage and decontamination details (Appendix C) but does not include ablations isolating the effect of replay, instruction evolution, or staged preference optimization. LongRAG is extended with “unanswerable” cases and includes explicit instructions to output “Unanswerable” (Appendix D.2.3), which is a useful robustness check. Safety is assessed using a third-party dataset (Table 9).

- Mixed/conditional results and trade-offs
  - Unanswerable detection: Qwen 32B outperforms EXAONE 32B on unanswerable cases (Table 14), while EXAONE excels when the answer is present. This suggests different calibration/priors about when to abstain.
  - LiveBench: EXAONE 32B lags Qwen 2.5 32B (43.0 vs 50.6), despite winning the real-world average (Table 6).

## 6. Limitations and Trade-offs
- Reliance on LLM-as-a-judge
  - Many key comparisons (e.g., MT-Bench, Arena-Hard, LongRAG, Ko-LongRAG, Ko-WebRAG) depend on GPT-4 variants as judges (Table 4; Figures 5 and 8). Although common in the field, this introduces potential bias and variance, and may favor certain stylistic tendencies.

- Unanswerable-case weakness
  - The extended LongRAG analysis (Table 14) shows notably lower accuracy for EXAONE 32B on unanswerable detection (e.g., Hotpot unanswerable 26.4%). In abstention-critical applications, this calibration may need adjustment.

- Limited ablations
  - The report does not include controlled ablations quantifying:
    - Impact of long-context replay vs. no replay.
    - Contribution of instruction evolution vs. standard SFT data.
    - Effect of staged preference optimization vs. single-stage.
  - Without these, it is hard to attribute gains to specific pipeline choices.

- Compute reporting granularity
  - Training compute is approximated by size×tokens (Table 3), a useful but coarse proxy. Details like optimizer settings, training duration, batch sizes, and exact context distributions are not provided.

- Licensing constraints (Appendix B)
  - Research-only license (“NC”); Section 4.2 assigns ownership of `Output` to the Licensor, with use/distribution permitted solely for research. This materially limits commercial deployment and even some open research workflows that expect permissive output rights.

- Data and coverage assumptions
  - While the tokenizer is balanced across Korean and English (Table 1), the performance gap on KMMLU (Table 8) suggests room to further strengthen Korean expert-knowledge coverage.
  - Safety evaluations show progress (Table 9) but the paper acknowledges “room for improvement,” and broader multilingual safety beyond Korean is not reported.

- Scalability trade-offs
  - All sizes support 32K context, which is valuable but can raise memory and latency costs at inference. The paper does not provide throughput/latency benchmarks.

## 7. Implications and Future Directions
- How this work changes the landscape
  - Demonstrates that a coherent pipeline (taxonomy-driven SFT, staged preference optimization, and replay-assisted long-context finetuning) can produce models that excel on instruction following and long-context RAG—even at smaller scales (`2.4B` with 32K context) and under moderate training compute (Tables 3, 5–7).
  - Provides a strong bilingual long-context baseline in Korean and English, including in-house Korean long-context RAG benchmarks with “unanswerable” handling (Tables 7, 15).

- Follow-up research enabled/suggested
  - Calibration and abstention: Improve unanswerable detection without sacrificing answerable performance—e.g., selective generation, confidence modeling, or “evidence required” prompts.
  - Ablation studies: Quantify the marginal effect of replay, instruction evolution, and staged DAA training on both real-world and long-context metrics.
  - Safety across languages: Extend harmlessness assessments beyond Korean and explore alignment techniques that preserve performance while increasing refusal accuracy in edge cases.
  - Efficiency studies: Benchmark latency/throughput at 32K contexts, explore memory-saving attention variants or retrieval compression strategies.

- Practical applications
  - Long-document support systems (contract analysis, scientific literature assistants, meeting-minute QA).
  - RAG-based enterprise assistants that must ground answers in large evidence sets and follow strict output formats (IFEval results suggest strong spec-following).
  - On-device or resource-constrained deployments using the `2.4B` model with full 32K context support (Table 1) for local privacy-preserving tasks—subject to license restrictions.

- Deployment considerations
  - For commercial use, a separate license is required; research-only output-rights constraints (Appendix B Sec. 4.2) will influence adoption strategies (e.g., prototyping with EXAONE 3.5, productionizing with licensed or alternative models).
  - Given the unanswerable-case findings (Table 14), production RAG systems should include explicit abstention prompts and possibly a verification layer (e.g., evidence highlighting or retrieval confidence thresholds).

Overall, EXAONE 3.5 presents a carefully engineered pipeline that prioritizes real-world instruction following and long-context competence, validated by broad benchmarks, detailed long-context tests, and competitive performance under a relatively modest training budget. The research-only license and some calibration gaps (unanswerable cases) are the main caveats; both are actionable for future releases or downstream system designs.
