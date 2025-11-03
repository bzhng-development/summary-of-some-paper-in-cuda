# LMSYS‑Chat‑1M: A Large‑Scale Real‑World LLM Conversation Dataset

**ArXiv:** [2309.11998](https://arxiv.org/abs/2309.11998)
**Authors:** Lianmin Zheng, Wei‑Lin Chiang, Ying Sheng, Tianle Li, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zhuohan Li, Zi Lin, Eric P. Xing, Joseph E. Gonzalez, Ion Stoica, Hao Zhang
**Institutions:** 

## 🎯 Pitch

LMSYS-Chat-1M introduces a groundbreaking dataset of one million real-world conversations with 25 large language models, enabling practical applications like content moderation, jailbreak safety benchmarks, and instruction fine-tuning. This unprecedented scale and diversity offer critical insights into user interactions, enhancing model safety, alignment, and evaluation benchmarks, thereby transforming how we approach LLM deployment and improvement in real-world scenarios.

---

## 1. Executive Summary (2-3 sentences)
LMSYS-Chat-1M is a publicly available dataset of one million real-world conversations between users and 25 large language models (LLMs), collected over five months from the Vicuna demo and the Chatbot Arena platform. The paper shows how this scale and diversity enable four concrete applications—training content moderation models, building a jailbreak-based safety benchmark, instruction fine-tuning, and constructing a challenging evaluation set—demonstrating practical value for safety, alignment, and benchmarking research.

## 2. Context and Motivation
- Problem addressed
  - There is no large, public, multi-model dataset of real user–LLM conversations at scale. Section 1 highlights three barriers: high serving costs, proprietary data held by commercial providers, and difficulty drawing users to open models.
- Why this matters
  - Practical impact: Real-world interactions contain distributional quirks—safety edge cases, adversarial prompts, multi-turn context—that lab-curated instruction datasets often miss. This matters for training content moderation, improving alignment, and building realistic benchmarks (Section 1).
  - Scientific impact: Understanding how people query LLMs informs model design, safety, and evaluation (Section 1).
- Prior approaches and gaps
  - Existing datasets are smaller, narrow in scope, or synthetic. Table 1 contrasts LMSYS-Chat-1M with Anthropic HH (338k samples, single model), OpenAssistant (66k), and Chatbot Arena Conversations (33k). Many synthetic datasets (e.g., Alpaca, UltraChat) lack real-user diversity (Section 6).
- Positioning
  - The dataset provides one million raw multi-LLM conversations from 210,479 distinct IPs across 154 languages, with per-message OpenAI moderation tags and basic PII filtering (Sections 2–3; Table 1). The release purposefully includes unsafe content to enable safety research (Section 2; Section 3.3).

## 3. Technical Approach
This is an empirical data-collection and analysis paper with four downstream demonstrations. The core pipeline has three parts: collection, analysis, and use cases.

- Data collection (Section 2)
  - Platform
    - A free website with three interfaces: `Single model`, `Chatbot Arena (battle)`, and `Chatbot Arena (side-by-side)`. Users either chat with one model or compare two anonymously or by choice (Section 2; Appendix A, Figures 7–8).
  - Models and infrastructure
    - 25 open and proprietary LLMs (e.g., `Vicuna`, `Koala`, `Llama-2`, `GPT-3.5`, `GPT-4`, `Claude-2`) were served over “dozens of A100 GPUs” for five months (Section 2).
  - Consent and safety
    - Users accept “Terms of Use” permitting data release. The team made “best efforts” to remove PII, kept unsafe content intact for safety research, and tagged messages with the OpenAI Moderation API (Section 2).
  - Data format
    - Each sample contains: conversation ID, model name, full conversation in OpenAI-style JSON, detected language (Polyglot), and moderation tags (Section 3.1).
- Dataset composition and analysis
  - Scale and diversity
    - “1,000,000” conversations, “25” models, “210,479” users, “154” languages; average “2.0” turns; average “69.5” tokens per prompt and “214.5” tokens per response (Table 1).
    - Model usage skew: `Vicuna` is most used; top-5 models by count include `Vicuna`, `Koala`, `Alpaca`, `ChatGLM`, and `Llama` (Figure 1).
    - Language diversity: top-5 languages are English, Portuguese, Russian, Chinese, and Spanish (Figure 2).
  - Topic distribution (Section 3.2; Figure 3)
    - Method: sample 100k English conversations, extract user prompts (initial and follow-ups), filter by length (32–1536 chars), embed with SentenceTransformers `all-mpnet-base-v2`, cluster via k-means into 20 clusters, then summarize each cluster’s center examples using GPT-4.
    - Result: most prompts are programming/software (Clusters 1,2,6,16,18), with notable unsafe/explicit clusters (9, 15, 17) and general knowledge/business/writing support clusters.
- Safety tagging (Section 3.3)
  - Definition
    - A conversation is “unsafe” if any message is flagged by the OpenAI Moderation API across categories: sexual, harassment, violence, hate, self-harm (Section 3.3).
  - Counts (Table 2)
    - > “Total flagged conversations: 54,427,” including “Sexual: 33,968,” “Harassment: 21,167,” “Violence: 9,499,” “Hate: 3,591,” “Self-harm: 863.”
- Four downstream use cases (Section 4)
  1) Content moderation model (Section 4.1)
     - Task: classify messages into five moderation categories via natural language rationales rather than rigid labels.
     - Training data: For each category, select top 1k flagged messages from LMSYS-Chat-1M; add 1k normal messages; GPT-4 generates explanations per message; add 3k ShareGPT conversations to diversify (Section 4.1).
     - Model: Fine-tune `Vicuna-7B` into `Vicuna-moderator-7B` using a system prompt defining categories and output format (Appendix B.2).
     - Evaluation: Construct a challenging test set of 110 toxic messages missed by the older `text-moderation-005` API; multi-label micro-F1 in zero-shot and one-shot, using same prompt scaffold for all models (Section 4.1; Table 3).
  2) Jailbreak safety benchmark (Section 4.2)
     - Definitions
       - `Jailbreak`: A technique that coaxes an aligned LLM to produce content it is trained to avoid (unsafe/harmful content).
       - For aggregate counts: a conversation is a jailbreak “attempt” if any user message is flagged; a “success” if any model message is flagged (Table 4; Section 4.2).
     - Analysis: Count attempts/successes across models (Table 4). Build a 50-prompt benchmark by picking the top 5 attempts for each of 10 models. Measure success by whether the model’s outputs are flagged by the OpenAI Moderation API (Table 5).
  3) Instruction fine-tuning (Section 4.3)
     - Subsets
       - `HighQuality`: 45k conversations from OpenAI/Anthropic models.
       - `Upvote`: 39k conversations from open models selected by user votes (no proprietary outputs).
     - Models
       - Fine-tune `Llama2-7B` to obtain `HighQuality-7B` and `Upvote-7B`.
     - Evaluation
       - MMLU (5-shot via InstructEval) and MT-Bench (LLM-as-judge), compared to `Llama2-7B`, `Llama2-7B-chat`, and `Vicuna-7B-v1.5` (Table 6).
  4) Arena-Hard-200 benchmark (Section 4.4)
     - Source: Chatbot Arena conversations (users vote on which of two models answered better).
     - Prompt triage method
       - Use GPT-3.5 with a strict rubric (Appendix B.7) to score prompts 1–10 on “benchmark potential” (ability to test problem-solving, creativity, and truthfulness). Figure 4 shows the score distribution.
       - Validate filtering: In a head-to-head on GPT-4 vs GPT-3.5, GPT-4 wins “52%” of top-50 prompts (scores > 8) but only “22%” of bottom-50 (< 2), indicating the top-scored prompts better separate stronger models (Figure 5).
     - Final benchmark
       - Select 200 “hard” prompts that all three graders (GPT-3.5, Claude-2, GPT-4) scored 9+. Evaluate models by GPT-4-as-judge. Arena-Hard-200 yields larger gaps between proprietary and open models than MT-Bench (Figure 6).

## 4. Key Insights and Innovations
- A first-of-its-kind, large-scale, multi-model, real-user conversation dataset
  - Novelty: Combines scale (1M conversations), multi-model coverage (25 models), and multi-lingual breadth (154 languages), with per-message moderation tags (Table 1; Section 3.1). Prior public datasets lack one or more of these dimensions.
  - Significance: Enables realistic safety analysis and instruction tuning grounded in in-the-wild behavior rather than synthetic prompts.
- Language-model-based moderation with rationales (Section 4.1; Table 3)
  - Difference: Instead of training a classifier, the approach fine-tunes a conversational model (`Vicuna-7B`) to produce a short explanation plus a category decision (Appendix B.2). This leverages the model’s generative capabilities and instruction-following, improving adaptability across categories and edge cases.
  - Outcome: `Vicuna-moderator-7B` achieves micro-F1 comparable to GPT-4 on a hard set of toxic messages missed by the previous OpenAI moderation model.
- A jailbreak benchmark sourced from real failures (Section 4.2; Tables 4–5; Appendix B.4)
  - Difference: Uses naturally occurring jailbreaks in live traffic rather than synthetic red-team prompts only.
  - Significance: More representative of actual adversarial behavior. The resulting “50 jailbreaks” stress-test even strong proprietary models (e.g., GPT-4 at 34% success; Table 5).
- Arena-Hard-200: A curated, human-in-the-loop-and-LLM-verified hard benchmark (Section 4.4; Figures 4–6)
  - Difference: Scores prompts for “benchmark potential,” cross-checks scores with multiple LLMs, and validates by head-to-head outcomes (Figure 5). Produces a compact benchmark that better separates models than MT-Bench (Figure 6).
  - Significance: Addresses the saturation of existing benchmarks and the challenge of evaluating complex, real-world tasks.

## 5. Experimental Analysis
- Evaluation methodology
  - Dataset characterization
    - Basic statistics and distributions: Table 1 (size, turns, tokens), Figure 1 (model counts), Figure 2 (languages), Figure 3 (topics).
    - Safety prevalence: OpenAI moderation tags show “5%” of conversations flagged (Section 3.3; Table 2).
  - Use case 1: Content moderation (Section 4.1; Table 3)
    - Setup: 5-category classification with explanations, trained on flagged LMSYS-Chat-1M messages + normal messages + ShareGPT augmentation; system prompt defines categories (Appendix B.2).
    - Test set: 110 manually labeled toxic messages not flagged by the older moderation model (`005`).
    - Metrics: micro-F1 in zero-shot and one-shot settings using the same prompt.
    - Main results (Table 3):
      > Zero-shot micro-F1 — `GPT-4: 0.71`, `Vicuna-moderator-7B: 0.65`, `GPT-3.5-Turbo: 0.45`, `OpenAI text-moderation-latest (006): 0.36`.
      > One-shot micro-F1 — `Vicuna-moderator-7B: 0.70`, `GPT-4: 0.69`, `GPT-3.5-Turbo: 0.64`.
    - Observations
      - Fine-tuning improves `Vicuna-7B` by ~30 points (0.35 → 0.65 zero-shot), and one-shot brings it on par with GPT-4 (Table 3).
      - Some models underperform due to refusals (e.g., `Llama-2-7B-chat` near 0); Appendix B.3 shows refusal examples even with a neutral moderation instruction.
    - Caveats
      - Small test set (110 messages) and reliance on the older moderation model for constructing the “missed toxic” pool could bias the evaluation toward specific weaknesses of `005`.
  - Use case 2: Jailbreak safety (Section 4.2; Tables 4–5)
    - Aggregate counts (Table 4): e.g., on `Vicuna-13B`, “15,925” attempts and “13,463” successes; on `GPT-4`, “368” attempts and “109” successes.
    - Benchmark construction: 50 jailbreak prompts (top-5 per 10 models), judge success by whether outputs are flagged by the moderation API.
    - Success rates (Table 5):
      > “`Llama-2-13B-chat: 16%`, `Claude-2: 18%`, `GPT-3.5-Turbo: 34%`, `GPT-4: 34%`, `Vicuna-13B-v1.5: 66%`, `Alpaca-13B: 74%`.”
    - Interpretation: Safety-tuned proprietary models are harder to jailbreak, but still vulnerable. Open models without safety training are much more vulnerable.
    - Qualitative evidence: Appendix B.4 shows multi-turn jailbreak strategies such as “content warnings,” “educational framing,” “rewrite more explicit,” and “token replacement,” illustrating realistic attack patterns.
  - Use case 3: Instruction fine-tuning (Section 4.3; Table 6)
    - Metrics: MMLU (5-shot) and MT-Bench.
    - Results (Table 6):
      > `HighQuality-7B`: “MMLU 47.7; MT-Bench 6.03” vs `Vicuna-7B-v1.5` “49.8; 6.17”.
      > `Upvote-7B`: “45.0; 5.86”. Baselines: `Llama2-7B` “42.4; 3.95”, `Llama2-7B-chat` “45.8; 6.27”.
    - Takeaways
      - With only “33M” fine-tuning tokens (Table 6), `HighQuality-7B` approaches `Vicuna-7B-v1.5`, suggesting LMSYS-Chat-1M contains high-quality instruction dialogues.
      - `Upvote-7B`, distilled only from open-model outputs, lags behind `HighQuality-7B`, indicating answer quality still matters more than prompt variety alone.
    - Caveats: Possible contamination (training might include questions similar to MMLU/MT-Bench) noted in Section 4.3.
  - Use case 4: Arena-Hard-200 (Section 4.4; Figures 4–6)
    - Prompt scoring distribution (Figure 4) demonstrates effective filtering; head-to-head ablation (Figure 5):
      > “Top-50: GPT-4 won 52%, Tie 40%, GPT-3.5 won 8%” vs “Bottom-50: GPT-4 won 22%, Tie 54%, GPT-3.5 won 24%.”
    - Final benchmark results (Figure 6): Arena-Hard-200 shows larger separation between open and proprietary models than MT-Bench, better reflecting real-world difficulty.
- Overall assessment
  - The experiments are carefully designed to demonstrate utility across safety, training, and benchmarking. While some evaluations rely on small or heuristic test sets (e.g., moderation 110 samples; the 50 jailbreaks), they are grounded in real traffic and supported by qualitative examples (Appendix B). The use of moderation API as the judge is a pragmatic but imperfect proxy for “harm.”

## 6. Limitations and Trade-offs
- Dataset representativeness (Section 5)
  - User bias: Many users are LLM enthusiasts/researchers; behavior may not represent the general population.
  - Model skew: `Vicuna` is the default model (Figure 1), leading to overrepresentation of its conversations.
- Data quality and duplication (Section 5; Figure 3 note)
  - No user registration and no heavy filtering introduce low-quality and duplicate content; some clusters appear templated/automated (Figure 3 caption).
- Annotations and labeling
  - No human preference labels are released as part of LMSYS-Chat-1M (Section 5). Moderation tags come from an automated API with known recall limitations (Section 3.3; Section 4.1).
- Safety and privacy
  - PII removal is “best effort,” not guaranteed (Section 2). Unsafe content is retained to support safety research; users must handle it responsibly (Section 3.3).
- Evaluation choices
  - Moderation evaluation uses a small set and automated judges; jailbreak success is defined by moderation flags, which may misclassify borderline cases.
  - Instruction-tuning results may be affected by benchmark contamination (Section 4.3).
- Operational constraints
  - Reproducing collection at this scale requires substantial compute and engineering (Section 2; Section 7 mentions need for sponsors for future dumps).

## 7. Implications and Future Directions
- How this work changes the landscape
  - Provides a shared, large-scale, multi-model, real-user corpus that complements synthetic instruction and preference datasets. This opens a path to study real-world safety, robustness, multilingual behavior, and emergent usage patterns at scale.
- Enabled follow-up research (Section 4.5; Section 7)
  - Safety and moderation
    - Train open moderation LMs with rationales and multi-label taxonomies.
    - Build richer jailbreak datasets, categorize attack patterns (e.g., “content warning,” “educational framing,” “token replacement”; Appendix B.4), and evaluate defenses.
  - Alignment and training
    - Curate instruction data via smart prompt selection and regenerate high-quality answers; pursue RLHF/RLAIF using human votes from Arena (future release) or synthetic preference data (Section 4.5).
  - Evaluation science
    - Study LLM-as-judge biases; develop prompt scoring and selection methods beyond simple difficulty heuristics; expand Arena-Hard beyond 200 prompts and across languages (Figures 4–6; Section 4.4).
  - Systems research
    - Model selection and request caching on real traffic traces; investigate cost/latency/quality trade-offs in multi-model routing (Section 4.5).
  - Privacy and ethics
    - Assess PII leakage, train privacy filters, and explore safe data release practices at scale (Section 4.5).
- Practical applications
  - Deployable content moderators on par with frontier models for organizations reluctant to share data with third-party moderation APIs (Table 3).
  - Robust safety evaluations for productizing chat systems (Tables 4–5; Appendix B.4).
  - Hard, diverse benchmarks (Arena-Hard-200) to track progress on complex real-world tasks (Figure 6).
- Next steps suggested by the paper
  - Quarterly dataset dumps contingent on resources and privacy management; community contributions of models, conversations, and votes (Section 7).
  - More rigorous de-duplication, calibration of topic distributions, broader multilingual analysis, and release/validation of preference signals.

In short, LMSYS-Chat-1M is both a data resource and a methodological blueprint: it shows how to turn real, messy user–LLM interactions into actionable assets for safety, training, and evaluation. The initial demonstrations—moderation training with rationales, a jailbreak stress test, instruction fine-tuning, and a hard benchmark—are credible starting points, with ample room for deeper, more controlled follow-up studies.
