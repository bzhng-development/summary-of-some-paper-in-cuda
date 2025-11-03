# Scaling Synthetic Data Creation with 1,000,000,000 Personas

**ArXiv:** [2406.20094](https://arxiv.org/abs/2406.20094)
**Authors:** Xin Chan, Xiaoyang Wang, Dian Yu, Haitao Mi, Dong Yu
**Institutions:** Tencent AI Lab (Seattle)

## 🎯 Pitch

Introducing persona-driven data synthesis, this paper presents a novel approach to generating diverse synthetic data via a colossal Persona Hub of over 1 billion unique personas. By leveraging these personas, the paper demonstrates the potential to enhance the diversity and quality of datasets used to train large language models, as evidenced by impressive performance on the MATH benchmark, while also addressing scalability challenges in synthetic data generation.

---

## 1. Executive Summary (2–3 sentences)
This paper introduces persona-driven data synthesis: adding a short, specific “persona” description to prompts so a large language model (LLM) generates new data from that persona’s perspective. To scale this idea, the authors automatically build Persona Hub, a collection of roughly 1.016 billion diverse personas, and show that these personas can drive large-scale creation of math and logic problems, instructions, knowledge-rich texts, game NPCs, and tools—culminating in a 7B open model fine-tuned on 1.07M persona-synthesized math problems achieving 64.9% on the MATH benchmark (Table 2).

## 2. Context and Motivation
- Problem addressed
  - Synthetic data generation with LLMs is easy to scale in quantity but hard to scale in diversity. A single “make X” prompt tends to produce similar outputs; getting 1B diverse instances (e.g., math problems) requires 1B diverse prompts (Section 1).
- Why important
  - Diverse, high-quality synthetic data is increasingly used to train and improve LLMs. Diversity matters both for model capability coverage and for avoiding “model collapse” when models train on their own outputs (Section 5.1 and 5.2).
- Prior approaches and shortcomings (Section 1)
  - Instance-driven: remix existing “seed” examples to make new ones (e.g., Self-Instruct). Limitation: fundamentally bounded by the seed corpus diversity and hard to scale beyond it.
  - Key-point-driven: enumerate topics or key concepts and prompt the model to cover them. Limitation: curation is prohibitive outside narrow domains (like mathematics).
- Positioning of this work
  - Introduces a general mechanism—attach a persona to the prompt—to steer the LLM to many perspectives without manually curating topics. Personas can be harvested at web scale, then combined with virtually any synthesis task prompt (Figure 1). Conceptually, Persona Hub compresses world knowledge into “distributed carriers” (Figure 2), which can be “decompressed” as diverse texts.

## 3. Technical Approach
The pipeline has two phases: build Persona Hub at scale, then use personas to drive diverse data synthesis.

A) Building Persona Hub (Section 2)
1) Text-to-Persona (Section 2.1; Figures 3–4)
   - Idea: Any web text implies a likely reader/writer persona. Prompt an LLM with a text and ask “Who is likely to read/write/like/dislike this text?” to generate a fine-grained persona description.
   - How granularity is controlled:
     - Prompt encourages specificity (“as specifically as possible”).
     - Input text itself sets granularity: technical texts produce finer personas (Figure 4 shows linear algebra and superconductivity leading to expert personas).
   - Source data: large public corpora; the paper runs on RedPajama v2 (Section 2.3).

2) Persona-to-Persona (Section 2.2; Figure 5)
   - Motivation: Text-to-Persona misses low-visibility personas (e.g., a child).
   - Mechanism: Given a seed persona, ask the LLM to list closely related personas via interpersonal relations (e.g., patient, colleague, supplier) and iterate. Inspired by “six degrees of separation,” the expansion runs for six iterations (Figure 5).

3) Deduplication (Section 2.3)
   - Why: Billions of generated personas will include near-duplicates.
   - How (two-stage):
     - MinHash-based dedup using 1-gram features, signature size 128, threshold 0.9 (Broder, 1997). MinHash is a fast, approximate method to estimate text similarity by hashing subsets of features.
     - Embedding-based dedup using a text embedding model (e.g., `text-embedding-3-small`) and cosine similarity threshold 0.9. Embeddings map sentences to vectors so semantically similar descriptions are close.
   - Optional stricter dedup (e.g., 0.5 threshold) when fewer but more diverse personas are desired.
   - Outcome: After heuristic quality filters, Persona Hub contains 1,015,863,523 personas (end of Section 2.3).

B) Persona-driven data synthesis (Section 3; Figure 6)
- Core mechanism: Insert a persona into a synthesis prompt to steer the LLM’s “roleplay” and perspective.
- Prompting modes (Figure 6):
  1) Zero-shot: “Create X with persona: {…}.” Maximizes creativity; no demonstrations.
  2) Few-shot: Provide examples to control format/quality.
  3) Persona-enhanced few-shot: Each demonstration explicitly lists its persona and output, which strengthens persona adherence but requires inferring personas for the demo set up front.
- Why this design: Personas are reusable across tasks and easy to combine with arbitrary prompts, unlike curated key-point lists. Roleplay is a strong, generalizable capability of modern LLMs (Section 3).

C) Use-case recipes (Section 4)
- Math problems (Section 4.1; Figures 7–10; Tables 1–2)
  - Data creation: pick personas; zero-shot prompt GPT-4 to write new problems (no MATH instances used). Generate solutions with `gpt-4o` (assistant), optionally cross-check with `gpt-4o` (PoT) and `gpt-4-turbo`.
  - Training: fine-tune an open 7B model (`Qwen2-7B`) on 1.07M synthesized problems; hold out 20k for a synthetic test set, filtered down to 11.6k by solution agreement (Section 4.1.2).
  - Evaluation: ID on the synthetic test; OOD on MATH.
- Logical reasoning problems (Section 4.2; Figures 11–12)
  - Prompts generate standard logical puzzles and culturally specific “Ruozhiba-style” problems once the style is defined in the prompt.
- Instructions (user prompts) (Section 4.3; Figure 13)
  - Zero-shot: guess what a persona would ask an assistant.
  - Persona-enhanced few-shot: use existing instruction datasets (e.g., WildChat), infer their personas via Text-to-Persona, then condition generation on those demos.
- Knowledge-rich texts (Section 4.4; Figure 14)
  - Example: write a Quora-style article “as” a given persona to elicit domain-specific knowledge.
- Game NPCs (Section 4.5; Figures 15–16)
  - Map real-world personas into NPCs consistent with game lore/world (e.g., World of Warcraft, Moonlight Blade).
- Tools (functions) (Section 4.6; Figures 17–18)
  - Predict what tools a persona would need (e.g., traffic condition checker for a cab driver), define high-level interfaces, and auto-implement code (Figure 18 shows a TensorFlow species identifier from an interface spec).

D) Diversity analysis mechanism (Section 4.1.2; Figure 10)
- Measure whether different personas actually yield different outputs:
  - Sample persona pairs with cosine similarities 0.4, 0.6, 0.8 using `text-embedding-3-small` (dim=512).
  - For each pair, generate math problems with greedy decoding (`temperature=0`) and compute problem-pair similarity.
  - Repeat under generic prompts and under more constrained prompts (finance & probability).
  - Also compare cross-model creation (`gpt-4o` vs `gpt-35-turbo`) using highly similar personas (0.9).

## 4. Key Insights and Innovations
- Persona as a universal steering handle (Figures 1 and 6; Sections 1 and 3)
  - What’s new: Rather than curating topics or remixing seeds, the paper shows that a short persona description robustly steers LLM outputs for many data types.
  - Why it matters: It decouples “diversity generation” from “topic enumeration.” A single persona inventory can power diverse synthesis tasks without re-curation.

- Web-scale Persona Hub (Section 2; Figures 3–5)
  - What’s new: A practical recipe to automatically derive over 1B personas from web text (Text-to-Persona) and social graphs (Persona-to-Persona), with dedup at scale.
  - Why it matters: Diversity bottleneck shifts from manual lists to automated mining; the “distributed carrier” view (Figure 2) offers a compression/decompression lens for world knowledge.

- Evidence that persona-driven diversity is real and controllable (Figure 10)
  - What’s new: An empirical link between persona similarity and output similarity, but with output similarity consistently lower than persona similarity; stronger prompt constraints increase output similarity.
  - Why it matters: Supports the claim that personas can scale diversity even at massive volumes, and that specificity can dial similarity up when wanted.

- Large-scale math performance from purely synthesized problems (Section 4.1.2; Tables 1–2; Figure 9)
  - What’s new: A 7B model fine-tuned on 1.07M synthesized math problems reaches 64.9% on MATH (greedy), comparable to `gpt-4-turbo-preview`(1106/0125) in their table.
  - Why it matters: Suggests persona-driven synthesis can train strong reasoning without directly using benchmark training data; supports the “paradigm shift” claim in Section 5.1.1.

## 5. Experimental Analysis
- Evaluation design (Section 4.1.2)
  - Datasets
    - Training: 1.07M GPT-4-written math problems created from 1.09M sampled personas via zero-shot prompting; solutions generated by `gpt-4o` (assistant). No MATH training instances used.
    - Synthetic test (ID): 20k held out; then filtered to 11.6k where at least two of three solutions match across `gpt-4o`(assistant), `gpt-4o`(PoT), and `gpt-4-turbo`.
    - OOD: MATH test set (5,000 competition-level problems).
  - Model and decoding
    - Fine-tune `Qwen2-7B` on 1.07M instances; greedy decoding at test time.
  - Metrics
    - Accuracy via exact-equivalence checking (uses OpenAI’s standard for MATH; for the synthetic test, uses `Llama-3-70B-Instruct` as the equality checker).
- Main quantitative results
  - In-distribution synthetic test (Table 1)
    - Fine-tuned `Qwen2-7B`: 79.4% accuracy on 11.6k synthetic items.
    - Baselines span 39.8%–77.2% for open-source instruction-tuned models; the fine-tuned 7B beats all listed open-source baselines on this ID set.
    - Caveat acknowledged: ID test answers are not “absolutely reliable,” and only their model is trained on that distribution.
  - Out-of-distribution MATH (Table 2)
    - > “Qwen2-7B (fine-tuned w/ 1.07M synthesized instances): 64.9%.”
    - For context: `gpt-4o-2024-05-13`: 76.6%; `gpt-4-turbo-2024-04-09`: 73.4%. Some numbers with asterisks may not use the same evaluation checker.
    - The fine-tuned 7B surpasses several strong open-source 70B–72B baselines listed in the table.
  - Scaling trend (Figure 9)
    - Accuracy on MATH increases as the number of synthetic training instances grows; the curve qualitatively follows known scaling-law behavior.
  - Problem validity spot-check (Section 4.1.2)
    - > “200 challenging problems were reviewed by two math experts; 7/200 invalid → 96.5% validity.”
- Diversity analysis (Figure 10)
  - Persona similarity vs. output similarity:
    - With persona similarities 0.4/0.6/0.8, the resulting problem similarities are lower and positively correlated.
    - Adding prompt constraints (finance & probability) raises output similarity (Figure 10b).
    - Across models (`gpt-4o` vs `gpt-35-turbo`) with persona similarity 0.9, most output similarities fall in 0.6–0.75 (Figure 10c), again lower than persona similarity.
- Other use-case demonstrations (Sections 4.2–4.6)
  - Logical reasoning (Figures 11–12), instructions (Figure 13), knowledge articles (Figure 14), game NPCs (Figures 15–16), and tool interfaces with code (Figures 17–18) are qualitative; they showcase versatility rather than provide quantitative benchmarks.
- Do the experiments support the claims?
  - For math: yes, there is compelling OOD evidence (MATH 64.9% for a 7B model with greedy decoding) that persona-synthesized data is useful. The diversity study is thoughtful and methodical.
  - For other domains: demonstrations are persuasive but not quantitatively evaluated.
- Missing/limited analyses
  - No controlled ablation on persona quality, dedup thresholds, or relationship-expansion depth.
  - Limited human validation (200-problem sample).
  - Limited reporting on cost/compute or failure modes beyond invalid-problem rate.

## 6. Limitations and Trade-offs
- Assumptions (Sections 2–3, 5)
  - Assumes LLMs can reliably infer plausible personas from arbitrary web text and that these personas meaningfully steer generation.
  - Assumes roleplay adherence is strong enough that a short persona string consistently shifts outputs (partly validated in Figures 7–8, 10).
- Scope and edge cases
  - Low-visibility or sensitive personas may still be missed or inaccurately represented; Persona-to-Persona helps but relies on LLM judgments (Figure 5).
  - Persona descriptions are currently short; they plan to enrich them with more fine-grained attributes in future versions (Conclusion).
- Data quality and bias
  - The LLMs used for synthesis may hallucinate or embed biases from pretraining; the 96.5% validity check is small (n=200) and only for “challenging” math problems (Section 4.1.2).
  - Some MATH gains might stem from pretraining contamination inside the generator models, even if no MATH instances were used in fine-tuning data; the paper cannot fully exclude this.
- Dedup and diversity guarantees
  - Dedup thresholds (0.9) are conservative; near-duplicates can remain, and semantic “coverage” is not quantified beyond similarity boxplots (Section 2.3; Figure 10).
- Compute and cost
  - Building and filtering 1B personas and generating >1M problems is compute-heavy; exact resource costs are not reported.
- Safety and ethics (Section 5.2)
  - The paper itself raises significant concerns: persona-driven querying at scale can “dump” a target LLM’s knowledge and capabilities via outputs, threatening data security and competitive advantages.
- Generalization beyond math
  - While use-case breadth is shown, only math receives rigorous, quantitative evaluation; transferability of the performance story to logic, instructions, or tool-use is still an open question.

## 7. Implications and Future Directions
- How this changes the landscape (Section 5.1)
  - Signals a potential shift from “humans create data, LLMs process” to “LLMs create data too.” If persona-driven synthesis can scale diversity and quality, the bottleneck of human-curated data may loosen (Section 5.1.1).
  - Introduces a “compression” viewpoint: Persona Hub (~10^10 tokens) as a compact set of carriers that can “decompress” world knowledge from LLMs into new text (Figure 2). This frames large-scale synthetic data creation as targeted memory elicitation.
- Follow-up research enabled
  - Persona enrichment: turn short blurbs into Wikipedia-level profiles (Conclusion) to increase uniqueness and control over style, background, and constraints.
  - Quantitative studies beyond math: rigorously evaluate logic problems, instructions, knowledge articles, NPCs, and tool specs to establish broad utility.
  - Ablations: measure the impact of dedup thresholds, relationship expansion depth, persona granularity, and prompt formats (zero-shot vs persona-enhanced few-shot).
  - Safety and governance: design rate limits, watermarking, and auditing for persona-driven mass querying, given the “full memory access” risk outlined in Section 5.1.3 and the data-security concerns in Section 5.2.1.
  - Multimodal personas: extend to images, audio, and video (Conclusion), enabling persona-driven multi-modal datasets (e.g., synthetic annotated images from an art historian persona).
- Practical applications
  - Data generation at scale for pretraining and post-training across domains; controllable diversification via personas.
  - Product simulation: model how different user types respond to new features or policies (Section 5.1.2).
  - Agent societies and sandbox testing: populate virtual worlds (games, simulations) with rich, consistent NPCs and agents (Sections 4.5 and 5.1.2).
  - Tool ecosystem bootstrapping: anticipate tool needs of user groups and prebuild function interfaces and implementations (Section 4.6; Figures 17–18).
  - Conversational systems: richer, persona-grounded dialogues and user modeling; the paper also notes a release of 370M “expert personas” (Section 1, release note).

> Key takeaway: By turning “who is speaking” into a first-class control knob, Persona Hub shows that scalable, diverse synthetic data creation is both feasible and useful—in one concrete case, enough to train a 7B model that scores 64.9% on MATH (Table 2)—while also surfacing serious implications for data security and model governance (Section 5.2).
