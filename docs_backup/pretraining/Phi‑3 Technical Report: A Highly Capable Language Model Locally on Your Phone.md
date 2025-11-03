# Phi‑3 Technical Report: A Highly Capable Language Model Locally on Your Phone

**ArXiv:** [2404.14219](https://arxiv.org/abs/2404.14219)
**Authors:** Marah I. Abdin, Sam Ade Jacobs, Ammar Ahmad Awan, Jyoti Aneja, Ahmed Awadallah, Hany Awadalla, Nguyen Bach, Amit Bahree, Arash Bakhtiari, Harkirat Behl, Alon Benhaim, Misha Bilenko, Johan Bjorck, Sébastien Bubeck, Martin Cai, Caio César Teodoro Mendes, Weizhu Chen, Vishrav Chaudhary, Parul Chopra, Allie Del Giorno, Gustavo de Rosa, Matthew Dixon, Ronen Eldan, Dan Iter, Amit Garg, Abhishek Goswami, Suriya Gunasekar, Emman Haider, Junheng Hao, Russell J. Hewett, Jamie Huynh, Mojan Javaheripi, Xin Jin, Piero Kauffmann, Nikos Karampatziakis, Dongwoo Kim, Mahmoud Khademi, Lev Kurilenko, James R. Lee, Yin Tat Lee, Yuanzhi Li, Chen Liang, Weishung Liu, Eric (Xihui) Lin, Zeqi Lin, Piyush Madan, Arindam Mitra, Hardik Modi, Anh Nguyen, Brandon Norick, Barun Patra, Daniel Perez‑Becker, Thomas Portet, Reid Pryzant, Heyang Qin, Marko Radmilac, Corby Rosset, Sambudha Roy, Olli Saarikivi, Amin Saied, Adil Salim, Michael Santacroce, Shital Shah, Ning Shang, Hiteshi Sharma, Xia Song, Masahiro Tanaka, Xin Wang, Rachel Ward, Guanhua Wang, Philipp Witte, Michael Wyatt, Can Xu, Jiahang Xu, Sonali Yadav, Fan Yang, Ziyi Yang, Donghan Yu, Chengruidong Zhang, Cyril Zhang, Jianwen Zhang, Li Lyna Zhang, Yi Zhang, Yeyey Zhang, Yunan Zhang, Xiren Zhou
**Institutions:** Microsoft Research

## 🎯 Pitch

The `Phi-3` series revolutionizes compact language models by leveraging a novel data-centric training approach that combines scaled web data and targeted synthetic input, enabling a 3.8B parameter model to achieve near-GPT-3.5 accuracy on smartphones. This advancement not only challenges conventional scaling laws but also empowers on-device applications with enhanced privacy, reduced latency, and offline capabilities, redefining efficiency in AI deployment.

---

## 1. Executive Summary (2–3 sentences)
This report introduces the `Phi-3` family of small language models that reach the quality of much larger models while being compact enough to run locally on a smartphone. The central advance is a data-centric training recipe (“data optimal regime”)—scaled, tightly filtered web data plus targeted synthetic data—that lets a 3.8B-parameter model (`phi-3-mini`) trained on 3.3T tokens achieve competitive accuracy with models like Mixtral 8x7B and GPT-3.5, and enables multilingual, long-context, MoE, and vision variants in the `phi-3.5` series.

## 2. Context and Motivation
- Problem addressed
  - How to build highly capable language models that are small enough for on-device use without the typical accuracy drop of small models.
  - Conventional “bigger is better” scaling laws assume fixed data quality; once data quality changes, those laws no longer dictate optimal model size.

- Why it matters
  - Real-world impact: on-device LLMs yield privacy (no data leaves device), lower latency, offline functionality, and lower serving costs. The paper demonstrates a quantized 4-bit `phi-3-mini` occupying ≈1.8 GB and running “more than 12 tokens per second” on an iPhone 14 (Figure 2; see “Highly capable language model running locally on a cell-phone.”).
  - Theoretical significance: it challenges fixed-data scaling laws by showing how improving data quality can let smaller models rival much larger ones (Figure 3).

- Prior approaches and their limits
  - Parameter scaling (e.g., Llama, Mixtral) improves performance but increases memory and compute, making on-device inference impractical.
  - Earlier `phi` models (phi-1.5, phi-2) showed promise that small models can punch above their weight with curated data, but did not reach the breadth of capability of current large models.

- Positioning
  - This work operationalizes a data-first training strategy at scale (“data optimal regime”) and augments it with lightweight architectural and systems improvements—rather than relying on sheer parameter count. It then extends the recipe to multilingual/long-context (`phi-3.5-mini`), efficient MoE routing (`phi-3.5-MoE`), and a compact multimodal model (`phi-3.5-Vision`).

## 3. Technical Approach
This section explains the architecture and training pipeline step-by-step, introducing unfamiliar terms on first use.

- Model family at a glance
  - `phi-3-mini` (3.8B params): Llama-2–style decoder-only transformer with 32 layers, 32 heads, hidden size 3072, Llama-2 tokenizer (vocab 32,064), default 4K context; a LongRope variant extends to 128K (`phi-3-mini-128K`) (Section “Technical Specifications”).
  - `phi-3-small` (7B) and `phi-3-medium` (14B): same general architecture class, trained for 4.8T tokens to study scaling with the same data (Section “Technical Specifications”).
  - `phi-3.5-mini`, `phi-3.5-MoE`, `phi-3.5-Vision`: add multilingual/long-context training, a mixture-of-experts (MoE) feed-forward design, and an image-text pipeline, respectively (Sections “Multilingual and Long Context” and “Phi-3.5-Vision”).

- Training recipe (“data optimal regime”)
  - Two-phase pretraining (Section “Training Methodology”):
    - Phase 1: “mostly web sources aimed at teaching the model general knowledge and language understanding.”
    - Phase 2: “even more heavily filtered webdata (a subset used in Phase-1) with some synthetic data that teach the model logical reasoning and various niche skills.”
  - What is “data optimal regime”? The paper focuses on curating data “to contain the correct level of knowledge” and retaining pages that likely improve reasoning. Example policy: remove low-value facts like “the result of a game in premier league in a particular day” to preserve capacity for reasoning (Section “Data Optimal Regime”; Figure 3 contrasts this approach to Llama-2 trained on a fixed dataset).

- Post-training alignment
  - Supervised finetuning (SFT) on carefully curated data across math, coding, reasoning, conversation, identity, and safety; followed by Direct Preference Optimization (DPO) using preference pairs to steer helpful/harm-free behaviors (Section “Post-training”).
  - Safety alignment includes red teaming and targeted dataset augmentation, which reduces harmful response rates (Figure 5).

- Systems and architectural choices that enable efficiency
  - Quantization: `phi-3-mini` can be quantized to 4 bits, reducing memory footprint to ≈1.8 GB; the iPhone 14 demo runs >12 tokens/s (Figure 2).
  - `phi-3-small` speed-ups (Section describing `phi-3-small`):
    - `GEGLU` activation and `muP` (Maximal Update Parametrization) for stable, transferable hyperparameters.
    - `Grouped-Query Attention (GQA)`: 4 queries share 1 key group; reduces KV-state duplication while keeping quality.
    - `Blocksparse attention` over the `KV cache`: each head attends to a different sparsity pattern so, collectively, all tokens are covered. This reduces memory without losing coverage. Figure 1 shows a toy layout: blue local blocks plus orange “remote/vertical” blocks, skipping gray blocks.
      - `KV cache` is the stored attention keys/values that let the model attend to earlier tokens without recomputing them. Making it blocksparse lowers memory and speeds up long-context inference.
    - Custom kernels: Triton kernels based on FlashAttention for training; a prefilling kernel plus an extended vLLM paged-attention kernel for decoding (Section on implementation in `phi-3-small`).
    - They alternate dense attention layers with blocksparse ones to balance retrieval quality and KV savings.
  - Long context with `LongRope` (rope scaling): extends context length from 4K to 128K in `phi-3-mini-128K` and `phi-3.5` models (Section “Multilingual and Long Context”). LongRope resizes rotary positional embeddings to preserve relative-position behavior across longer sequences.

- Mixture-of-Experts (`phi-3.5-MoE`)
  - The MoE replaces dense feed-forward blocks with 16 expert networks; a learned router activates the top-2 experts per token (“top2 routing”), so only a subset runs per token (Section “The phi-3.5-MoE adopts an Mixture-of-Experts…”).
  - Each expert is a GLU-style FFN; total parameters ≈42B, but only 6.6B are active per token—yielding high quality at manageable active compute.
  - Router training uses `SparseMixer` (a method for training sparse routers efficiently).

- Multimodal model (`phi-3.5-Vision`)
  - Architecture: CLIP ViT-L/14 image encoder + `phi-3.5-mini` text decoder. Visual tokens interleave with text tokens without special ordering (Section “Phi-3.5-Vision: Technical Specifications”).
  - Dynamic cropping: splits high-resolution images into blocks to cover large/varied aspect ratios (Section “Architecture”).
  - Pretraining: 0.5T tokens across interleaved documents, image-text pairs (e.g., FLD-5B), OCR of PDFs, tables/charts, and text-only; loss computed on text tokens only; max train resolution 1344×1344 (Section “Pre-training”).
  - Post-training: multimodal SFT (~33B tokens) and smaller-scale multimodal DPO; co-trained with text tasks to retain language skill (Section “Post-training”).

## 4. Key Insights and Innovations
- Data-first small models can rival much larger ones
  - Novelty: the “data optimal regime”—strong filtering for educational value and synthetic reasoning data—lets a 3.8B model trained on 3.3T tokens achieve benchmark results near GPT-3.5 and Mixtral 8x7B (Section “Abstract”; Table block under “Academic benchmarks”).
  - Evidence: `phi-3-mini` averages 69.7 across benchmarks vs GPT-3.5 at 72.8 (Table under “Academic benchmarks”). Figure 3 visualizes improved scaling when optimizing data quality rather than only increasing parameters.

- Efficient attention for long contexts on small hardware
  - Innovation: per-head `blocksparse attention` over the KV cache divides the context among heads with complementary sparsity patterns (Figure 1), enabling memory reductions and speedups without fully dense attention at every layer. This is paired with custom kernels for both training and inference (Section on `phi-3-small`).
  - Significance: supports long contexts and on-device inference viability.

- Strong small-scale MoE with low active parameters
  - Innovation: `phi-3.5-MoE` (16×3.8B experts, 6.6B active) achieves language/math/code performance on par with mid-size frontier assistants (Table 3).
  - Significance: shows MoE can deliver near-frontier quality with limited active compute.

- Compact multimodal model that competes with larger systems
  - Innovation: `phi-3.5-Vision` (4.2B total) matches or outperforms larger open models on several single- and multi-image benchmarks, with a simple “text loss only” pretraining design and dynamic cropping (Table 5 and Table 6).
  - Significance: lowers the barrier for practical multimodal assistants.

- On-device practicality demonstrated
  - Quote: “phi-3-mini can be quantized to 4-bits so that it only occupies ≈ 1.8GB of memory… deploying phi-3-mini on iPhone 14… fully offline achieving more than 12 tokens per second.” (Figure 2 caption and surrounding paragraph).

## 5. Experimental Analysis
- Evaluation methodology
  - Language benchmarks use a single internal pipeline: few-shot prompts at temperature 0, same across all compared models; numbers may differ from other publications due to prompt choices (Section “Academic benchmarks”). Footnote 4 notes that adding “##” before a question boosts `phi-3-mini`, but that tweak was not used for fairness.
  - Post-training safety is tested via Microsoft’s internal multi-turn RAI framework using GPT-4 as a simulator plus human red teaming (Section “Safety”).

- Main quantitative results (selected highlights)
  - Overall language quality (Table under “Academic benchmarks”):
    - `phi-3-mini` averages 69.7 across tasks; `phi-3-small` 73.6; `phi-3-medium` 76.7; GPT-3.5 (1106) is 72.8.
    - Notable task scores:
      - `MMLU` (5-shot): 68.8 (`mini`), 75.7 (`small`), 78.0 (`medium`), vs 71.4 (GPT-3.5).
      - `GSM8K` (8-shot CoT): 82.5 (`mini`), 89.6 (`small`), 91.0 (`medium`), vs 78.1 (GPT-3.5).
      - `MATH` (0-shot CoT): 41.3 (`mini`), 34.6 (`small`), 53.1 (`medium`), vs 45.3 (GPT-3.5).
      - `TriviaQA` (5-shot): 64.0 (`mini`), 58.1 (`small`), 73.9 (`medium`), vs 85.8 (GPT-3.5) — shows small-model knowledge limits.
      - `MT-Bench` (2-round avg): 8.38 (`mini`), 8.70 (`small`), 8.91 (`medium`), vs 8.35 (GPT-3.5).
  - Multilingual and long-context (Section “Multilingual and Long Context”):
    - Multilingual MMLU (Figure 4): `phi-3.5-mini` avg 55.4 vs `phi-3-mini` 47.3; `phi-3.5-MoE` reaches 69.9.
    - RepoQA (code long-context; Table 1): `phi-3.5-MoE` avg 85, beating open-source Llama-3.1-8B (71) and Mixtral series (68–67.8), close to Gemini-1.5-Flash (90) and GPT-4o-mini (90.6).
    - RULER (long-context retrieval; Table 2): Llama-3.1-8B avg 88.3; `phi-3.5-MoE` 87.1; `phi-3.5-mini` 84.1. Performance drops at 128K, which the paper attributes to lack of high-quality long-context training data.
  - Safety and robustness
    - Red teaming reductions: Figure 5 shows decreased harmful responses after safety alignment across categories like “current events,” “fairness/bias,” and “violence.”
    - Internal RAI benchmark (Table 4; lower is better): `Ungroundedness` improves from 0.603 (`phi-3-mini`) to 0.213 (`phi-3-medium`) and 0.228 (`phi-3.5-MoE`), versus 0.328 (Llama-3-Instruct-8B) and 0.935 (Mistral-7B).
  - Vision results (Table 5 and Table 6):
    - Single-image: `phi-3.5-Vision` achieves 91.3 on ScienceQA (vs GPT-4O at 88.5), 81.9 on MMBench-dev-en (vs 88.4 for GPT-4O; 79.0–75.9 for MM1-7B/MM1-3B), and 43.9 on MathVista testmini (vs 54.4 GPT-4O; 31–36 for open baselines).
    - Multi-image/video: On BLINK, 57.0 (`phi-3.5-Vision`) vs 61.0 (Gemini 1.5 Pro) and 63.2 (GPT-4O); on VideoMME, 50.8 (`phi-3.5-Vision`) vs 62.6 (Gemini 1.5 Pro) and 68.4 (GPT-4O). Despite smaller size, it competes with or exceeds open models of similar scale.

- Do the experiments support the claims?
  - Yes, with caveats:
    - Data-centric training clearly elevates small models: `phi-3-mini` and `phi-3-small` surpass or match GPT-3.5 on many tasks (MMLU, GSM8K, MT-Bench), consistent with the “data optimal regime” hypothesis (Table under “Academic benchmarks” and Figure 3).
    - Long-context: strong but not SOTA; performance declines at 128K indicate the method needs better long-context training data (Table 2).
    - Knowledge-heavy QA (e.g., TriviaQA) remains a weakness for the smallest models, and the paper demonstrates retrieval augmentation as a remedy (Figure 6).
    - Evaluation fairness: same internal pipeline across models, but prompt choices differ from public leaderboards, so numbers aren’t directly comparable outside this report.

- Ablations and robustness checks
  - There is an implicit scaling ablation (`mini`→`small`→`medium`) on the same data; gains from 7B→14B are smaller than 3.8B→7B, suggesting the data mix is closer to “optimal” at smaller scales (Section “Data Optimal Regime”).
  - No explicit ablation separates the effect of synthetic vs filtered web data or blocksparse attention vs dense.
  - Safety evaluation includes both automated and adversarial human testing (Section “Safety”), with category-wise improvements.

- Qualitative failure and mitigation
  - Quote: “The model simply does not have the capacity to store too much ‘factual knowledge’, which can be seen for example with low performance on TriviaQA. However, we believe such weakness can be resolved by augmentation with a search engine.” (Section “Weakness”; Figure 6 shows RAG fixing an Olympics question).

## 6. Limitations and Trade-offs
- Assumptions and dependencies
  - Heavy reliance on data curation quality and synthetic data generation. The exact filters and synthetic sources aren’t fully enumerated, which can affect reproducibility and bias analysis (Sections “Training Methodology” and “Data Optimal Regime”).
  - The internal evaluation pipeline may advantage certain prompting styles (footnote 4 identifies prompt choices that change outcomes).

- Capability limits of small models
  - Factual recall remains weaker for small models; this surfaces on `TriviaQA` and suggests dependency on retrieval for knowledge-intensive tasks (Table under “Academic benchmarks”; Section “Weakness”).
  - Long-context degradation at 128K indicates current mid-training lacks sufficiently rich long-context data (Table 2).

- Scope limits
  - English-centric training for `phi-3-mini`; multilingual capability is added later in `phi-3.5`, not core to `phi-3` (Section “Weakness”).
  - `phi-3.5-Vision` can struggle with “high-level reasoning” and may output ungrounded content in sensitive domains like finance (Section “Phi-3.5-Vision: Weakness”).

- Safety trade-offs
  - Even with safety tuning, occasional failures remain (e.g., describing scam images, certain CAPTCHAs), reflecting the classic helpfulness–harmlessness tension (Section “Phi-3.5-Vision: Safety” and “Weakness”).

- Engineering trade-offs
  - Blocksparse attention introduces kernel complexity; gains depend on support in inference stacks (the paper extends vLLM and implements custom Triton kernels; Section on `phi-3-small`).
  - MoE introduces routing variability and potential load imbalance; the paper uses SparseMixer to stabilize training, but operational complexity is higher than a purely dense model.

## 7. Implications and Future Directions
- How this changes the landscape
  - Establishes a credible data-centric path to high-quality, small LLMs that run on consumer devices. This can shift deployment strategies from cloud-only to hybrid or offline-first for many applications (privacy-critical messaging, assistive tools, embedded devices).
  - Demonstrates that thoughtfully curated data can bend scaling-law expectations (Figure 3), encouraging the community to invest in data pipelines as much as in parameters or FLOPs.

- Practical applications
  - On-device assistants, keyboards, offline translation/summarization, code copilots in IDEs without network access, privacy-preserving note-taking, and multimodal mobile apps (`phi-3.5-Vision`) for forms, charts, or UI understanding.

- Research directions enabled or suggested
  - Data-centric science
    - Systematic ablations of data filters and synthetic curricula: what mixture is “data-optimal” for a given size? How does that mixture evolve from 3.8B to 14B+ (the report notes diminishing returns 7B→14B)?
    - High-quality long-context corpora to close the 128K drop on RULER (Table 2).
  - Retrieval and tool use
    - Tight integration with retrieval/search to offset factual recall limits (demonstrated in Figure 6), plus program-of-thought/tools for math and code to push beyond `phi-3.5-MoE` results (Table 3).
  - Efficient architectures
    - Further exploration of blocksparse patterns and dynamic sparsity to reduce KV memory while maintaining recall; standardizing kernels across inference stacks.
    - MoE routing improvements (load balancing, latency predictability) and expert specialization diagnostics.
  - Multilingual and multimodal robustness
    - Broader multilingual coverage at small scales (beyond the `phi-3.5` additions), with careful safety tuning to prevent content risks across languages.
    - For `phi-3.5-Vision`, more reasoning-focused and hallucination-targeted DPO (Section “Phi-3.5-Vision: Weakness”) and unified treatment of multi-image/video contexts.

- Bottom line
  - The `Phi-3/3.5` series shows that with the right data and lightweight engineering, small models can deliver near-frontier assistant quality while running on ubiquitous hardware. The clearest next win is to pair such models with retrieval and to continue refining long-context and multilingual training data, turning small, private, low-latency assistants into a practical default rather than an exception.
