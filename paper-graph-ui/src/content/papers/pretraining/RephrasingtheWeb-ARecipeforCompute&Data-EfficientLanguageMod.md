# Rephrasing the Web: A Recipe for Compute & Data-Efficient Language Modeling

**ArXiv:** [2401.16380](https://arxiv.org/abs/2401.16380)

## 🎯 Pitch

This paper introduces WRAP (Web Rephrase Augmented Pre-training), a straightforward yet powerful approach that leverages an instruction-tuned language model to automatically rephrase noisy web data into cleaner, evaluation-aligned styles (like Wikipedia or QA format), and then pre-trains large language models on a balanced mix of real and rephrased text. WRAP dramatically boosts training efficiency—cutting the compute and data needed by up to 3× and improving both generalization and zero-shot performance—without sacrificing the web’s broad knowledge or incurring the cost and bias risks of fully-synthetic corpora. This signals a practical path toward faster, cheaper, and more robust language model pre-training as quality data becomes scarce and compute budgets tighten.

---

## 1. Executive Summary
This paper introduces WRAP (Web Rephrase Augmented Pre‑training), a simple data recipe: use an instruction‑tuned language model to rephrase web pages into cleaner, task‑aligned styles (e.g., “Wikipedia‑like” or “question–answer”), then pre‑train a model on a 1:1 mixture of original and rephrased text. Across 1.3B‑parameter models trained up to 300B tokens, WRAP cuts the compute or data needed for a given quality by roughly 3×, reduces perplexity on out‑of‑distribution corpora by up to ~50%, and improves zero‑shot QA accuracy by ~2–3% on average (Figures 1b–c, 2; Tables 1–2).

## 2. Context and Motivation
- Problem addressed
  - Web scrapes (e.g., Common Crawl) are noisy, unstructured, and rarely formatted like downstream evaluations (e.g., QA). Standard practice is to pre‑train on this data, often with handcrafted filters, which still leaves style and quality mismatches.
  - Scaling laws (Chinchilla) suggest linearly increasing both compute and data with model size. This is becoming impractical due to cost and the scarcity of high‑quality data (Introduction; Related Work).
- Why it matters
  - Pre‑training dominates the cost of building LLMs. If the same performance can be achieved with fewer tokens or steps, the savings are substantial.
  - Matching pre‑training data style to evaluation (e.g., QA format) could improve zero‑shot generalization without expensive instruction‑tuning (Introduction; Section 3.1).
- Prior approaches and gaps
  - Data filtering and curated mixtures help but are opaque and often proprietary (Related Work).
  - Synthetic corpora (e.g., “textbook‑quality” or TinyStories) can work well for small models but are expensive to generate with large closed LLMs, may introduce topic/knowledge biases, and do not scale transparently (Related Work; Introduction).
- Positioning of this work
  - WRAP reuses the web’s knowledge but modifies its style/clarity by rephrasing, avoiding topic selection bias and expensive “knowledge generation.” It uses smaller, open instruction models (e.g., `Mistral‑7B‑Instruct`) to create synthetic data at lower cost while keeping the original information content (Sections 3.1, 6.2, 7.1).

## 3. Technical Approach
Step‑by‑step overview (Section 3; Appendices B–C, G):
1. Source corpus
   - Use C4 (a cleaned Common Crawl subset; ~170B English tokens) as the main corpus to rephrase (Appendix A.1).
2. Rephrasing model and styles
   - Freeze an instruction‑tuned model (default: `Mistral‑7B‑Instruct`) and prompt it to rephrase each document chunk (≤300 tokens) into one of four `styles` (Section 3.1; Appendix G):
     - `Easy`: very simple sentences and small vocabulary.
     - `Medium`: “high‑quality English like Wikipedia.”
     - `Hard`: “terse and abstruse” scholarly language.
     - `Q/A`: convert to “Question: … Answer: …” conversational format.
   - 300‑token chunks are used because longer inputs led to information loss during rephrasing (Section 3.1).
   - Outputs are lightly filtered to remove generic prefaces (e.g., “Here’s a paraphrase…”), using sentence heuristics (Appendix B).
3. Training mixture (“WRAP”)
   - Pre‑train target LLMs on a 1:1 mixture of original C4 and synthetic rephrases sampled online (Section 3.1).
   - Rationale: retain exposure to “messy” real‑web tokens (URLs, markup, typos) while gaining style diversity and clarity from rephrases (Section 3.1; Figures 2–3).
4. Models and optimization
   - Decoder‑only GPT‑style models at three sizes (Section 3.2):
     - `128M` (12 layers, 12 heads, d=768), `350M` (24L, 16H, d=1024), `1.3B` (24L, 16H, d=2048).
   - Train with Megatron‑LM, context 1024, cosine LR schedule, Adam (β1=0.9, β2=0.999), weight decay 0.01, grad clip 1.0; batch size 1M tokens (Section 3.2).
   - Standard runs: 300k steps (=~300B tokens), unless otherwise noted (Section 3.2).
5. Evaluations and why not evaluate on C4 itself
   - Main generalization metric: perplexity on 21 Pile domains (first 10k documents per domain) rather than C4 (Section 4; Appendix A.2).
   - Reason: training on WRAP optimizes a different distribution from C4 alone (Equations (1) vs (2)), so C4 perplexity can increase slightly while generalization improves.
   - Zero‑shot QA: 13 tasks via the LM‑Evaluation‑Harness (Section 5.1; Tables 1–2).
6. Cost model (Section 7.1)
   - Rephrasing with `Mistral‑7B` via vLLM yields ~3M tokens/hour per A100. Rephrasing 85B tokens ≈ 25k GPU hours. Model training (1.3B for 300B tokens) on 64×A100 at 0.5M tok/s ≈ 6k GPU hours; for 13B, ≈30k GPU hours.
   - Throughput and speculative decoding can reduce generation cost; smaller rephrasers (e.g., `Qwen‑1.8B`) run ~3× faster with similar quality (Figure 5; Section 7.1).

How WRAP “works” conceptually
- It replaces “noisy” or mismatched styles with clear, evaluation‑aligned styles while preserving the original information (semantic content). Evidence:
  - Cosine similarity of sentence embeddings between original and rephrased text is high (Figure 8a–b; Appendix C.2), indicating content preservation.
  - Rephrases change readability and syntactic profiles (e.g., “Medium” increases reading level and dependency depth closer to Wikipedia/academic text; “Q/A” lowers reading level, aligning with QA corpora) (Figures 10–11; Appendix C).

## 4. Key Insights and Innovations
- Rephrasing for style alignment, not knowledge synthesis
  - Novelty: instead of asking a large LLM to “generate new knowledge,” WRAP uses a smaller LLM to “rephrase” existing web text into target styles. This keeps the topic distribution of the web while making the text cleaner and closer to evaluation formats (Section 3.1; Appendix G).
  - Significance: avoids expensive and opaque synthetic‑corpus design and mitigates bias from topic curation. Leads to faster learning and better zero‑shot QA (Figures 1b–c; Tables 1–2).
- Style is a powerful lever
  - The `Q/A` style consistently boosts zero‑shot QA accuracy; `Medium` (“Wikipedia‑like”) improves perplexity over diverse domains (Figures 2–4; Tables 3, 6, 10–11).
  - No single style dominates everywhere; an oracle mixing styles per domain could reduce perplexity by 16% for small models (Figure 7), suggesting style‑diverse pre‑training is valuable.
- Synthetic data is not just “augmentation”
  - Basic augmentations (synonym replacement, random deletion) fail to match rephrasing benefits (Figure 6). Gains come from style and structural changes that better match downstream distributions, not from trivial lexical variants.
- The rephraser can be small and open
  - High‑quality paraphrases from `Qwen‑1.8B` or `Mistral‑7B` match or beat those from a larger `Vicuna‑13B` on downstream perplexity; a weak `T5‑base` rephraser hurts (Figure 5). This enables cheaper, reproducible pipelines.
- Compute/data efficiency without new knowledge
  - On “Specialized Knowledge” tasks (e.g., MMLU, PubMedQA), performance scales with more real data exposure; WRAP helps but does not “create” knowledge (Table 2). This clarifies the role of rephrasing: faster learning, not knowledge injection.

## 5. Experimental Analysis
- Setup
  - Datasets: Pre‑training on C4 or RefinedWeb (RW); Pile for perplexity evaluation across 21 domains (Section 4; Appendix A.2); 13 zero‑shot tasks split into “General Understanding” and “Specialized Knowledge” (Section 5.1; Appendix A.3).
  - Metrics: macro token‑level perplexity (Equation (3)); task accuracy for QA datasets (Section 5; Appendix D).
  - Baselines: C4 (85B or 170B tokens), RW (160B/320B tokens), `Pythia‑1.4B` (trained on Pile), `TinyLlama` (1T tokens) (Tables 1–2).
- Main results
  - Faster, better pre‑training
    - Zero‑shot learning curves (Figure 1b): WRAP reaches a target average accuracy with ~3× fewer tokens; early checkpoints show up to ~15× faster progress on Pile perplexity (Section 4).
    - Perplexity: For a 1.3B model trained 300B tokens, `C4 + QA‑85B` beats `C4‑170B` across most Pile domains (Figure 2). The average Pile perplexity reduction is up to ~50% (Section 4).
    - Data efficiency: With 150B tokens total, 1.3B models trained with WRAP outperform models trained 300B tokens on C4 alone (Figure 1c; Figure 13).
  - Zero‑shot QA gains (Table 1)
    - On “General Understanding” tasks, `Synthetic (85B)` reaches an average 49.4% vs ~47% for strong real‑data baselines; `Synthetic + C4 (85B)` also averages 49.4%.
    - Biggest win: TruthfulQA improves to 44.0% (`Synthetic`) vs 33.5–39% (real‑data baselines). When mixing in real data, TruthfulQA drops slightly to 40.6%, indicating a trade‑off between style benefits and real‑web noise exposure.
  - Specialized Knowledge tasks (Table 2)
    - Averages: WRAP variants (45.0–45.5%) are comparable to stronger real‑data baselines (`Pythia‑Pile 300B` at 44.6%, `TinyLlama 1T` at 45.6%). Improvements saturate as total real tokens increase (RW‑320B only +0.2% over RW‑160B).
    - Interpretation: WRAP speeds learning but does not inject new domain facts; exposure to more real tokens still matters.
- Ablations and robustness
  - Real data is necessary: Using synthetic only degrades perplexity on web‑like domains (e.g., OpenWebText2, HackerNews) that contain special tokens and noise (Figure 3). Adding real data improves both perplexity and QA in most settings (Tables 3–4).
  - Combining styles: Mixing `Q/A` and `Medium` yields small perplexity gains but does not outperform `Q/A` alone on zero‑shot QA (Tables 5–6; Figure 4).
  - Rephraser quality: `Qwen‑1.8B` and `Mistral‑7B` produce strong rephrases; a weak `T5‑base` hurts (Figure 5).
  - Not just augmentation: WRAP significantly outperforms synonym replacement or deletion (Figure 6).
  - Style–domain match: For 128M models, training with a style that matches a domain (e.g., `Q/A` for StackExchange‑like text) helps, but no single style dominates across all Pile domains (Figure 7).
  - Semantic fidelity and leakage check: High SimCSE cosine similarity between real and rephrased pairs shows content preservation, while differences with random real pairs/halves indicate rephrases do not add new content (Figure 8; Appendix C.2). Additional MRPC paraphrase analysis shows synthetic rephrases behave similarly to true paraphrases (Figure 9).
- Smaller‑budget regimes
  - 350M models, 75B tokens: WRAP improves average QA by ~1.4 points (44.1% vs 42.7%) and Specialized Knowledge by ~2.2 points (41.0% vs 38.8%) when adding `Q/A` rephrases to 15B real tokens (Tables 8–9; Figure 12).
  - 1.3B models, 150B tokens: WRAP (`QA+C4‑35B`) improves “Specialized Knowledge” average to 45.0% vs 42.4–42.8% for C4 alone, and “General Understanding” to 48.4% vs 46.0–46.7% (Tables 10–11; Figure 13).
- Few‑shot leaderboard snapshot
  - On six OpenLLM‑Leaderboard tasks, a 1.3B WRAP model matches or beats `Falcon‑RW 1.3B` and `Pythia‑1.4B` (e.g., ARC‑C 36.4 vs 35.1; TruthfulQA 40.6 vs 36.0 and 38.7) (Table 12).

Do the experiments support the claims?
- Yes, for data/compute efficiency and zero‑shot QA. The paper controls for real‑token exposure (85B vs 170B vs 300B), compares against competitive baselines, and provides detailed ablations (Figures 2–7; Tables 1–6, 8–12). The “no new knowledge” caveat is consistent with results on MMLU/PubMedQA (Table 2).

## 6. Limitations and Trade-offs
- Assumptions and scope
  - Rephrasing preserves semantics. Similarity analyses (Figure 8; Appendix C) support this, but subtle factual drift or hallucinations in rephrases are not deeply audited at scale.
  - The benefits are shown primarily for models up to 1.3B parameters and English‑language web corpora. Behavior at 7B–70B scale and multilingual settings remains untested (implicit limitation).
- What WRAP does not address
  - It does not “create” domain knowledge. On knowledge‑heavy benchmarks, performance still depends on exposure to more real tokens (Table 2).
  - Diversity of synthetic outputs: paraphrasers can reduce content diversity (Section 7.2). The pipeline does not explicitly enforce topical diversity or novelty.
- Computational and data costs
  - Rephrasing 85B tokens costs ≈25k GPU hours with `Mistral‑7B`, though smaller/faster rephrasers and speculative decoding can reduce this (Section 7.1). This is a nontrivial upfront cost, offset when training multiple models.
  - Synthetic‑only training hurts robustness to real‑web artifacts (Figure 3); hence a real+synthetic mixture is required, which means managing two distributions during pre‑training.
- Open questions
  - Optimal style mixing schedules over training.
  - How many rephrases per document are beneficial?
  - Interaction with later instruction‑tuning or RLHF.
  - Risk of long‑term “model collapse” when repeatedly training on model‑generated text (Related Work), even if WRAP reuses web semantics.

## 7. Implications and Future Directions
- How this changes the landscape
  - WRAP reframes “data scaling” as “style scaling”: preserving web knowledge while shaping text to match evaluation formats. This gives a practical, transparent alternative to opaque data curation or large‑LLM synthetic corpora.
  - It provides a compute‑efficient path to better zero‑shot QA: train once on a real+`Q/A` mixture, then evaluate without extra instruction‑tuning (Figures 1b, 2; Tables 1, 10–11).
- Enabled follow‑up research
  - Style‑aware curriculum schedules: start with “Medium” for fast perplexity gains; blend in “Q/A” as training progresses for QA transfer.
  - Automatic style selection: learn per‑domain or per‑batch style weights (cf. DoReMi‑like mixture optimization), leveraging the domain‑style sensitivity in Figure 7.
  - Lightweight rephrasers: systematically search the smallest model that preserves semantics but improves style (Figure 5 suggests the bar can be low).
  - Robustness checks: factual drift detection between original and rephrase; semantic equivalence metrics beyond cosine similarity; human audits on sensitive domains.
  - Low‑resource and multilingual settings: rephrase small, noisy corpora into clearer registers to bootstrap strong base models (Section 7.1 motivation).
- Practical applications
  - Cost‑effective pre‑training for organizations with limited high‑quality data access.
  - Targeted pre‑training for assistants (e.g., customer support) where `Q/A` style alignment is crucial.
  - Data governance: rephrasing can filter harmful formatting without discarding knowledge, potentially aiding safety and bias mitigation (discussed qualitatively in Conclusion).

> Core takeaway: Rather than hunting for ever larger or “cleaner” web dumps, WRAP shows that rephrasing the web you already have—into styles that match how you evaluate and use models—yields faster learning, better zero‑shot transfer, and comparable knowledge performance, all with smaller, open paraphrase models (Figures 1–6; Tables 1–2, 10–11).
