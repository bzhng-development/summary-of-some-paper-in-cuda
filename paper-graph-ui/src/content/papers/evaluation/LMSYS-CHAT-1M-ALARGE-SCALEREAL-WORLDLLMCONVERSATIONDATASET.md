# LMSYS-CHAT-1M: A LARGE-SCALE REAL-WORLD LLM CONVERSATION DATASET

**ArXiv:** [2309.11998](https://arxiv.org/abs/2309.11998)

## 🎯 Pitch

LMSYS-Chat-1M introduces the first openly available, large-scale dataset of one million real-world conversations between users and 25 different large language models, representing over 210,000 users across 154 languages. This resource empowers research in LLM safety, alignment, benchmarking, and instruction tuning—enabling advances like content moderation on par with GPT-4, robust jailbreak evaluation, and more realistic model comparisons. Its unprecedented scale and diversity unlock new opportunities for understanding and improving how LLMs interact with humans in real applications.

---

## 1. Executive Summary
LMSYS-Chat-1M introduces a public, million-conversation dataset of real human interactions with 25 large language models (LLMs) gathered “in the wild” from a free website (April–August 2023). Beyond releasing the data, the paper demonstrates four concrete uses: training a content moderation model that performs on par with GPT-4, building a jailbreak safety benchmark, instruction-tuning LLMs to near-Vicuna quality, and creating a new challenging evaluation, Arena-Hard-200, that better separates strong from weak models.

## 2. Context and Motivation
- Problem/gap: Research on how people actually interact with LLMs needs large, diverse, real-user conversations across multiple models. Such datasets are scarce because:
  - Running multi-model services is expensive.
  - Companies with user data rarely release it.
  - It’s hard to attract sustained traffic to open models (Section 1).

- Importance:
  - Real-world value: Understanding user behavior (what they ask, how they adapt to model behavior) helps improve safety, alignment, and usefulness (Section 1).
  - Research value: Enables training and evaluating content moderation, instruction-following, and robust benchmarks grounded in real usage (Sections 1, 4).

- Prior datasets and shortcomings:
  - Human–LLM or human preference datasets exist (e.g., Anthropic HH, OpenAssistant, Chatbot Arena Conversations) but are smaller, single-model, or lack the scale/diversity required for multi-LLM analysis (Table 1; Section 6).
  - Synthetic datasets (e.g., Alpaca, UltraChat) contain model-generated queries, not real user prompts; pre-LLM dialogue datasets (e.g., UbuntuDialogue, DailyDialog) don’t reflect modern LLM interactions (Section 6).

- Positioning:
  - LMSYS-Chat-1M provides 1,000,000 real conversations with 25 models, 210K users, and 154 languages, with moderation tags and minimal filtering to preserve safety-relevant content (Sections 2–3; Table 1). It also demonstrates four downstream uses (Section 4).

## 3. Technical Approach
A. Data collection pipeline (Section 2)
- Service and interfaces:
  - A public website offers three chat modes: single model, “Chatbot Arena (battle)” with two anonymous models, and “Chatbot Arena (side-by-side)” with user-selected models (Appendix A; Figures 7–8).
- Scope and infrastructure:
  - Timeframe: April–August 2023.
  - Models served: 25 open-source and proprietary LLMs, hosted on dozens of A100 GPUs (Section 2).
- Consent and safety preprocessing:
  - Users accept Terms of Use allowing data release.
  - Personal Identifiable Information (PII) removed “to the best effort.”
  - OpenAI Moderation API tags every message; unsafe content is retained to support safety research (Section 2).
- Data format and fields:
  - Each sample includes `conversation_id`, `model`, the full multi-turn conversation in OpenAI-style JSON, detected language tag, and moderation tags (Section 3.1).
- Scale and diversity (Table 1; Figures 1–2):
  - “1,000,000 conversations; 25 models; 210,479 users; 154 languages; 2.0 turns per sample; 69.5 avg tokens per prompt; 214.5 avg tokens per response.”
  - The most-chatted models include `vicuna-13b`, `koala-13b`, `alpaca-13b`, `chatglm-6b`, `llama-13b` (Figure 1).
  - Top languages: English, Portuguese, Russian, Chinese, Spanish (Figure 2).

B. Topic analysis to characterize prompts (Section 3.2; Figure 3)
- Sampling and filtering: 100K English conversations; extract all user prompts (including follow-ups); keep prompts 32–1536 characters.
- Embedding and clustering:
  - Compute sentence embeddings with `all-mpnet-base-v2` (SentenceTransformers).
  - Use k-means to form 20 clusters; select 100 prompts nearest each centroid.
  - Summarize clusters with GPT-4 to assign interpretable topic labels.
- Observations (Figure 3):
  - Heavy coding/software themes; significant unsafe content clusters; general writing, business, and QA.

C. Unsafe content tagging and prevalence (Section 3.3; Table 2)
- Definition: A conversation is unsafe if any message in it is flagged by the OpenAI Moderation API (latest set used for dataset tags).
- Aggregate counts:
  > Table 2: “#Flagged conversations: 54,427 … Sexual: 33,968; Harassment: 21,167; Violence: 9,499; Hate: 3,591; Self-harm: 863.”
- Caveat: The paper notes the API’s recall can be low, so the true unsafe rate is likely higher (Sections 3.3, 4.1).

D. Four downstream use cases (Section 4)
1) Content moderation model (Section 4.1; Table 3; Appendix B.2–B.3)
- Training data:
  - For five categories: `hate`, `self-harm`, `sexual`, `violence`, `harassment`, select top 1K flagged messages per category from LMSYS-Chat-1M.
  - Add 1K non-toxic messages for balance plus ~3K ShareGPT conversations to diversify (Section 4.1).
  - Generate “explanations” for labels with GPT-4. Rather than train a classifier, fine-tune a 7B LLM (`Vicuna-7B`) to output category with rationale. This leverages the LLM’s generative strengths for explainable moderation.
- Model:
  - `Vicuna-moderator-7B` = `Vicuna-7B` fine-tuned on moderation instructions and examples (Appendix B.2 system prompt).
- Evaluation:
  - Construct a challenging set: 110 toxic messages not flagged by an older OpenAI moderation model (v005), plus 25 non-toxic; multi-label; measure micro-F1 (harmful/non-harmful across categories) in 0-shot and 1-shot settings (Section 4.1).
- Results (Table 3):
  > Zero-shot micro-F1: “GPT-4: 0.71; Vicuna-moderator-7B: 0.65; GPT-3.5-Turbo: 0.45; … Llama-2-7B-chat: 0.00.”  
  > One-shot micro-F1: “Vicuna-moderator-7B: 0.70 ≈ GPT-4: 0.69; GPT-3.5-Turbo: 0.64; … Llama-2-7B-chat: 0.01.”
- Qualitative finding: Some chat-aligned models (e.g., `Llama-2-7B-chat`, `Claude-2`) refuse the moderation task itself—leading to poor scores (Appendix B.3). The fine-tuned moderator avoids refusals and explains decisions.

2) Safety benchmark from “jailbreak” conversations (Section 4.2; Tables 4–5; Appendix B.4)
- Terminology: `jailbreak` = a prompt that circumvents a model’s safety guardrails to elicit disallowed content.
- Mining attempts and successes:
  - Count, per model, conversations with flagged “user messages” (attempts) and flagged “model messages” (successes), using the moderation API (Table 4).
  - Example counts:
    > Vicuna-13B: “All Convos 490,712; Attempt 15,925; Success 13,463.”  
    > GPT-4: “All Convos 7,304; Attempt 368; Success 109.”  
    > Claude-2: “All Convos 2,241; Attempt 78; Success 18.”
- Constructed benchmark:
  - From 10 representative models, take top-5 jailbreak attempts each → 50 conversations. Score whether a model’s response is flagged (“success”) by the latest moderation API (Table 5).
  - Success rates:
    > “Llama-2-13B-chat: 16%; Claude-2: 18%; GPT-3.5-Turbo: 34%; GPT-4: 34%; Vicuna-13B-v1.5: 66%; Alpaca-13B: 74%.”
- Interpretation: Safer chat-aligned models (Llama-2-chat, Claude-2) resist more, while unaligned open models jailbreak more easily (Section 4.2). Appendix B.4 catalogs real jailbreak techniques found in the data (e.g., content warnings, “educational purpose” framing, token replacement of sensitive words).

3) Instruction fine-tuning with real conversations (Section 4.3; Table 6)
- Datasets:
  - `HighQuality` (45K conversations): from OpenAI/Anthropic models (answers presumed high-quality).
  - `Upvote` (39K): from open models, selected by user upvotes (no proprietary-model data).
- Training:
  - Fine-tune `Llama-2-7B` to create `HighQuality-7B` and `Upvote-7B` (Section 4.3).
- Evaluation:
  - Benchmarks: `MMLU` (5-shot; general knowledge & reasoning) and `MT-Bench` (multi-turn conversation quality judged by GPT-4) (Table 6).
- Results (Table 6):
  > “Llama2-7B: MMLU 42.4, MT-Bench 3.95.”  
  > “Llama2-7B-chat: 45.8, 6.27.”  
  > “Vicuna-7B-v1.5: 49.8, 6.17.”  
  > “HighQuality-7B: 47.7, 6.03 (near Vicuna-7B).”  
  > “Upvote-7B: 45.0, 5.86.”
- Takeaway: With careful selection from LMSYS-Chat-1M, small models approach Vicuna and Llama-2-chat performance. Quality of answers matters—`Upvote-7B` lags the proprietary-distilled `HighQuality-7B`.

4) Creating challenging benchmarks from real prompts (Section 4.4; Figures 4–6; Appendix B.7–B.8)
- Goal: Identify real-user prompts that best discriminate strong vs. weak models in creativity, problem-solving, and factuality.
- Method:
  - Start from Arena data (head-to-head prompts with human votes).
  - Use an LLM-based “prompt rater” pipeline:
    - Provide a rubric (Appendix B.7 system prompt).
    - Ask `GPT-3.5-Turbo` to score prompts from 1–10 for benchmarking potential (Figure 4 shows the resulting score distribution).
  - Validate the scoring:
    - Compare `GPT-4` vs `GPT-3.5` on two sets: Top-50 prompts (score > 8) and Bottom-50 (< 2).
    - Figure 5: On Top-50, GPT-4 wins 52% vs GPT-3.5’s 8% (40% ties); on Bottom-50, GPT-4 wins only 22% (24% GPT-3.5; 54% ties).
- Arena-Hard-200 benchmark:
  - Select 200 prompts with score ≥ 9 with agreement from `GPT-3.5-Turbo`, `Claude-2`, and `GPT-4` (Appendix B.8 lists examples).
  - Evaluate models with GPT-4-as-judge (as in MT-Bench) (Figure 6).
  - Observation: Arena-Hard-200 “reveals larger performance gaps between open and proprietary models than MT-Bench,” making it a tougher, more discriminative benchmark (Figure 6).

## 4. Key Insights and Innovations
- A million-scale, multi-LLM, real-world conversation dataset (Sections 2–3; Table 1)
  - Novelty: Prior public datasets lack this simultaneous combination of scale (1M), diversity (25 models, 154 languages), and real usage without heavy curation.
  - Why it matters: Enables research on actual user behavior, not synthetic or scripted interactions.

- Moderation-by-generation: fine-tuning a small LLM to both explain and classify toxicity (Section 4.1; Appendix B.2)
  - Difference vs. standard classifiers: The model outputs a short rationale plus a category, which improves interpretability and can reduce refusal behavior seen in chat-tuned models.
  - Significance: On a hard set the OpenAI moderation (v005) missed, `Vicuna-moderator-7B` reaches GPT-4-level micro-F1 in a 1-shot setting (Table 3).

- Jailbreak mining from organic conversations to build a safety benchmark (Sections 4.2; Tables 4–5; Appendix B.4)
  - Difference: Instead of synthetic red teaming, the benchmark uses “found” jailbreak attempts and successes in the wild, capturing tactics people actually use.
  - Significance: Produces a pragmatic safety yardstick where safer models (Claude-2, Llama-2-chat) indeed show lower jailbreak rates (Table 5).

- LLM-assisted prompt selection to create a harder real-world evaluation (Section 4.4; Figures 4–6; Appendix B.7)
  - Difference: Uses LLM scoring plus human-vote grounding to filter for high-discriminative prompts; then cross-validates with Top-50 vs Bottom-50 win rates (Figure 5).
  - Significance: Yields Arena-Hard-200, a compact benchmark that better separates top proprietary models from open models than MT-Bench (Figure 6).

## 5. Experimental Analysis
- Datasets and setup:
  - Moderation: Train on flagged/unflagged LMSYS-Chat-1M messages with GPT-4 explanations; evaluate on 110 toxic messages missed by prior moderation (v005) plus 25 benign (Section 4.1).
  - Safety: Count attempts/successes via moderation tags across models (Table 4). Construct 50-case jailbreak suite and measure success rates by whether responses trigger `text-moderation-006` (Table 5; Section 4.2).
  - Instruction-tuning: Fine-tune Llama-2-7B on `HighQuality` and `Upvote` subsets; evaluate with MMLU (5-shot) and MT-Bench (GPT-4 judge) (Section 4.3; Table 6).
  - Benchmarking: Score prompts with GPT-3.5 using a rubric (Appendix B.7), validate discriminativeness (Figure 5), finalize Arena-Hard-200, and compare models with GPT-4 judge (Figure 6; Section 4.4).

- Principal quantitative results:
  - Moderation (Table 3):
    > 0-shot micro-F1: GPT-4 0.71; Vicuna-moderator-7B 0.65; GPT-3.5 0.45.  
    > 1-shot micro-F1: Vicuna-moderator-7B 0.70 ≈ GPT-4 0.69; GPT-3.5 0.64.  
    The fine-tuned moderator closes most of the gap to GPT-4 in 1-shot and avoids refusals that hurt `Llama-2-7B-chat` (0.00/0.01).
  - Safety robustness (Tables 4–5):
    > On the 50-case benchmark: “Llama-2-13B-chat: 16%; Claude-2: 18%; GPT-3.5-Turbo: 34%; GPT-4: 34%; Vicuna-13B-v1.5: 66%; Alpaca-13B: 74%.”  
    This ordering aligns with expectations that safety-aligned chat models resist jailbreaks better.
  - Instruction-tuning (Table 6):
    > `HighQuality-7B`: MMLU 47.7 vs Vicuna-7B’s 49.8; MT-Bench 6.03 vs 6.17.  
    > `Upvote-7B`: MMLU 45.0; MT-Bench 5.86.  
    Shows competitive results with a fraction of the tokens used for Vicuna (33M vs 370M fine-tuning tokens).
  - Benchmarking difficulty (Figures 4–6):
    > Figure 5: On Top-50 prompts, GPT-4 wins 52% vs GPT-3.5’s 8% (40% ties); on Bottom-50, GPT-4 wins 22%.  
    > Figure 6: Arena-Hard-200 widens gaps between open and proprietary models compared with MT-Bench.

- Do the experiments support the claims?
  - Moderation: Yes, on the specific hard set missed by v005, the generative moderator is competitive with GPT-4 (Table 3). However, the evaluation set is small (135 items total) and focused on v005 misses; broader generalization would require larger, diverse test sets and human adjudication.
  - Safety: The 50-case benchmark is clear and interpretable, but small; success criteria rely on the same moderation API used for tagging, which can mislabel edge cases (Section 3.3). The large-scale counts in Table 4 provide additional context that aligns with known safety alignment differences.
  - Instruction-tuning: Results on MMLU/MT-Bench are standard and credible; the paper explicitly notes possible contamination from overlapping questions (Section 4.3).
  - Arena-Hard-200: The top-vs-bottom validation (Figure 5) is a smart sanity check. Scoring/selection still depends on LLM judges, which can have biases; nonetheless, the enlarged performance gaps observed in Figure 6 are consistent with broader community experience.

- Ablations, failure cases, robustness:
  - The paper includes 0-shot vs 1-shot results (Table 3) and refusal analyses (Appendix B.3). It does not deeply ablate data selection strategies (e.g., how many samples per category for moderation, or different selection heuristics for instruction-tuning). Jailbreak evaluation is concise but not exhaustive. Good qualitative coverage of jailbreak techniques is provided (Appendix B.4).

## 6. Limitations and Trade-offs
- Data representativeness and quality (Section 5):
  - User base skew: Many users are hobbyists/researchers; behavior may not generalize to everyday users or enterprise settings.
  - No deduplication/filtering by design: Includes low-quality or script-generated prompts (Figure 3 note on clusters 14 and 20 showing templated/scripted inputs).
  - No human preference annotations in the release, though some votes exist in Arena data; quality yet to be validated for release (Section 5).

- Safety labeling constraints:
  - OpenAI Moderation API is used for tagging and some evaluations; its recall can be low (Sections 3.3, 4.1), and it can influence outcomes (e.g., jailbreak “success” is defined by whether it flags the output).

- Imbalance across models and languages:
  - Heavy skew toward `vicuna` (default model) and English (Figures 1–2). Some models have far fewer conversations, which can bias per-model analyses.

- Possible benchmark contamination:
  - Instruction-tuning experiments may contain questions overlapping with MMLU/MT-Bench (Section 4.3).

- Ethical and legal considerations:
  - Dataset includes unfiltered unsafe content (Section 3.3). PII was removed to the best effort, but residual risks remain. Researchers must apply responsible-use practices.

- Scale/cost constraints:
  - Operating such a service requires substantial GPU time and careful privacy/safety handling; future quarterly releases depend on sponsorship (Section 7).

## 7. Implications and Future Directions
- How this work shifts the landscape:
  - Provides a widely accessible, large-scale, multi-LLM conversation corpus that reflects real user behavior. This unlocks empirical research across alignment, safety, evaluation, and data curation that previously depended on proprietary data.

- Follow-up research enabled:
  - Safety and red teaming:
    - Train and compare alternative moderation paradigms (classifier vs. generative rationale models).
    - Study generalization across moderation taxonomies and APIs; build human-validated safety sets.
    - Expand the jailbreak benchmark with more cases and standardized success criteria beyond a single moderation API.
  - Instruction tuning and data selection:
    - Systematically explore prompt-only selection plus regeneration with a target model to create high-quality instruction datasets (Section 4.3 suggestion).
    - Study data filtering/deduplication strategies and their effect on downstream capability (Section 4.5 mentions data selection work).
  - Evaluation science:
    - Improve LLM-as-a-judge reliability and calibration; analyze selection bias of LLM graders (Section 4.4 notes this as future study).
    - Extend Arena-Hard-200 methodology to domain-specific “hard” sets and multilingual prompts.
  - Systems and applications:
    - Model selection and request caching (Section 4.5), where real request distributions matter.
    - Privacy research using realistic conversation traces (Section 4.5).
    - RLHF/RLAIF pipelines grounded in crowdsourced, multi-LLM conversational data (Section 4.5).

- Practical applications:
  - Safer deployments: Fine-tune lightweight, on-prem moderation models with explanations (Table 3), avoiding sending sensitive data to third-party APIs.
  - Benchmarking: Adopt Arena-Hard-200 to stress-test general assistants; use method to curate “hard” sets from an organization’s own chat logs.
  - Data curation: Mine high-yield prompts and regenerate answers with a target LLM to construct compact, high-quality instruction datasets that rival larger curated sets (Table 6).

In sum, LMSYS-Chat-1M supplies both a valuable dataset and concrete, reproducible methodologies—moderation-by-generation, jailbreak mining, instruction data selection, and LLM-assisted prompt curation—that together advance practical, safety-focused, and evaluation-centric LLM research.
