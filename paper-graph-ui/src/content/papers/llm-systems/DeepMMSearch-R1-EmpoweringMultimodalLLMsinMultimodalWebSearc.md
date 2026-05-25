# DeepMMSearch-R1: Empowering Multimodal LLMs in Multimodal Web Search

**ArXiv:** [2510.12801](https://arxiv.org/abs/2510.12801)

## 🎯 Pitch

DeepMMSearch-R1 introduces a pioneering multimodal large language model that dynamically performs multi-turn, on-demand web searches—both image- and text-based—while answering knowledge-intensive, information-seeking visual questions. By enabling iterative, self-reflective query refinement and precise, crop-based image search, the model overcomes the limitations of static knowledge and rigid retrieval pipelines, delivering superior real-world performance and up-to-date reasoning. This advance positions DeepMMSearch-R1 as a major step toward truly intelligent, adaptable AI assistants capable of robust factual reasoning in ever-changing environments.

---

## 1. Executive Summary (2-3 sentences)
DeepMMSearch-R1 is a multimodal large language model (MLLM) that can decide, in the middle of a conversation about an image, when and how to search the web and then reason over the retrieved results to answer knowledge-intensive questions. It introduces two core abilities that prior systems lack: (a) iterative, multi-turn refinement of text queries based on retrieved evidence (self-reflection and self-correction) and (b) targeted, crop-based image search that focuses the search on the exact visual entity referenced in the question (Figure 1). Across six benchmarks, it outperforms prior open-source search-equipped MLLMs and is competitive with strong closed models (Table 1).

## 2. Context and Motivation
- Problem addressed:
  - MLLMs often fail on knowledge-intensive, information-seeking visual QA (VQA) because many questions require up-to-date, long-tail knowledge that is not contained in a static training corpus (Section 1). The paper illustrates this with a question whose answer depends on a recent event: “Where is the boat race happening?”—a model with a January 2025 knowledge cutoff fails (Section 1; Appendix F.1 shows the image).
- Why it matters:
  - Real-world assistants must handle questions that blend visual recognition (what is this logo/species/building?) with external facts (speed, date, biography, regulation, etc.). The web changes constantly, so static training data becomes stale; systems must search live information (Section 1).
- Shortcomings of prior approaches:
  - Retrieval-Augmented Generation (RAG): assumes a static knowledge base and uses a rigid retrieve-then-generate pipeline. This leads to unnecessary or noisy retrieval and misses up-to-date, open-world facts (Section 1; Related Works B.2).
  - Prompted Search Agents: rely on instructions to use tools but are not optimized to handle noisy web results; they struggle to reason over retrieved evidence and generalize to open-world settings (Section 1; Related Works B.3).
  - Prior search-equipped MLLMs: mostly text-only retrieval; first multimodal extension (MMSearch-R1) restricts each tool to a single call and performs image search on the whole image, which is brittle when background clutter misleads retrieval (Figure 1-left; Section 1; Related Works B.4).
- Positioning of this work:
  - DeepMMSearch-R1 is a web-search-equipped MLLM that:
    - Performs multi-turn search with self-reflection: it can revise text queries after reading retrieved evidence (Figure 1-right).
    - Performs crop-aware image search: it first grounds a model-generated referring expression with Grounding DINO to crop the relevant region, then searches with that crop to improve retrieval precision (Section 1; Figure 1-right).
    - Is trained in two stages—supervised finetuning (SFT) and online reinforcement learning (RL via GRPO)—to learn when to search, which tool to call, how many times to call it, and how to integrate results (Section 3).

## 3. Technical Approach
- Overview of the system (Figure 1; Sections 2–4):
  - Base model: `Qwen2.5-VL-7B-Instruct` (a 7B-parameter MLLM).
  - Tools:
    - `text search tool`: runs model-generated queries against an in-house web index; an LLM summarizes the top-5 results to fit context limits (Section 4.1, “Multimodal Search Tools”).
    - `grounding tool`: Grounding DINO, an open-set detector that takes a natural-language “referring expression” and returns a bounding box (Section 1; 4.1).
    - `image search tool`: reverse-image search (via in-house API) using either the whole image or the crop plus top-5 result summaries (Section 4.1).
  - Dialogue schema:
    - The model writes structured tags in its output:
      - `<reason>…</reason>`: transparent rationale before any action.
      - `<img_search>…</img_search>`: either `<img>` for whole-image search or a referring expression describing the object to crop-and-search.
      - `<text_search>…</text_search>`: a query string for web search.
      - `<information>…</information>`: place where the system injects summarized search results.
      - `<answer>…</answer>`: final concise answer (Section 2; Figure 2-top).
- Stage 1: Supervised Finetuning (SFT) (Section 3.1)
  - Data: `DeepMMSearchVQA`, a 10,000-sample multi-turn dataset the authors constructed by distilling tool-use behavior and reasoning from a strong proprietary MLLM (Gemini-2.5-Pro) over InfoSeek examples, then filtering to keep only samples whose final answers match ground truth (Section 2; Figure 2).
    - Composition: balanced across knowledge types; explicitly mixes search-free and search-required examples to teach “when to search” (Figure 2-bottom; Figure 3-right).
    - Each sample records the entire interaction: reasons, tool calls, retrieved information, and final answer (Figure 1; Appendix G).
  - Training objective: standard causal language modeling on the assistant’s outputs, but with a critical twist—tokens inside `<information>…</information>` (web snippets) are masked from loss so the model learns to reason over, not mimic, retrieved text (Section 3.1).
  - Parameterization choices:
    - Freeze vision encoder and projection layers to preserve strong pretrained image features.
    - Finetune LLM with LoRA adapters (`r=8`) to add the smallest number of new parameters necessary for tool use (Section 3.1; Implementation Details 4.1).
- Stage 2: Online Reinforcement Learning with GRPO (Section 3.2)
  - What is GRPO? Group-Relative Policy Optimization compares multiple candidate rollouts for the same input; each rollout receives a reward, and the advantage is computed relative to the group average. This stabilizes updates in noisy reward settings (Section 3.2).
  - Rollout generation:
    - Start from the SFT model; let it reason, call tools, and incorporate tool outputs until it answers or hits limits on turns/tokens (Section 3.2).
    - The system constrains the total number of tool calls and response length to encourage efficiency (Section 3.2; Appendix D notes: image search may be called once, text search multiple times, total tool calls ≤ 10; max response length 8192).
  - Reward design:
    - `s` = correctness score, binary {0,1}, judged by `gpt-5-chat-latest` (LLM-as-judge).
    - `s_fmt` = format compliance (correct tags, valid tool-call structure).
    - Final reward: `R_total = (1 − λ_fmt) * s + λ_fmt * s_fmt`, with `λ_fmt = 0.1` (Section 3.2).
    - A KL penalty to a frozen reference constrains the policy away from drifting too far from the SFT distribution (Section 3.2).
- Why these design choices?
  - Crop-based image search: Whole-image reverse search is easily misled by irrelevant background; cropping around “the white bird flying over the water,” for example, yields relevant evidence about that bird (Figure 1-right), avoiding search results about other objects (Figure 1-left).
  - Multi-turn text queries: Web pages are noisy; a single fixed query often returns partial facts. Iterative refinement guided by retrieved snippets lets the model home in on the missing detail (Figure 1-right, text-search turns culminating in “highest recorded speed of egret”).
  - SFT + RL: SFT teaches the mechanics of tool use and schema compliance; RL teaches policy-level decisions—when to search at all, which tool to call, when to stop—to maximize accuracy with minimal calls (Sections 3, 4.2; Figure 4; Figure 5).
- Implementation footprint (Section 4.1, Implementation Details):
  - SFT: 3 epochs, LR 1e-4, LoRA rank 8, bf16, 1 node × 8 H100 GPUs, global batch 8.
  - RL: GRPO in veRL; 30 epochs, 4 nodes × 8 H100 GPUs, batch 512, rollout number 8, LR warmup 45 steps with LR 2e-6, KL penalty 0.001, clip ratio 0.2, max 8192 tokens (Section 4.1).

## 4. Key Insights and Innovations
- Crop-aware image search (fundamental innovation)
  - What’s new: Before image search, the model writes a referring expression (e.g., “the white bird flying over the water”) that Grounding DINO uses to crop the question-relevant region. The crop is then submitted to reverse image search (Sections 1, 4.1; Figure 1-right).
  - Why it matters: It reduces background-induced noise that often derails whole-image searches (Figure 1-left). Ablation shows a mean gain of +1.75 points across six datasets when cropped image search is enabled (Figure 3-left).
- Self-reflection and self-correction in web search (fundamental innovation)
  - What’s new: The model can issue multiple text searches, read the summaries, then refine the query if evidence is incomplete (“highest recorded speed of egret” after noting that “25 mph” is a cruising speed; Figure 1-right).
  - Why it matters: Real web results are noisy and partial. Enabling iterative refinement produces a consistent accuracy lift over a single fixed query baseline (Figure 3-left).
- Training recipe that separates “reasoning over information” from “absorbing information” (important design)
  - Masking retrieved tokens during SFT (Section 3.1) forces the model to learn when to call tools and how to reason with the summaries rather than memorize Web snippets. This is complemented by RL with a correctness/format reward and KL regularization (Section 3.2).
- Balanced and structured SFT data (important design)
  - `DeepMMSearchVQA` mixes search-free and search-required questions 50:50 and balances knowledge categories (Figure 2-bottom; Figure 3-right). Ablation shows imbalance leads to over-searching on simple datasets or under-searching on hard ones; 50:50 yields the best average (Figure 3-right).
- Policy refinement through RL that reduces unnecessary tool use (incremental but impactful)
  - RL increases selective use of image search, encourages multiple text searches when justified, and substantially reduces unnecessary cropped searches (−36.81% on DynVQA and −34.86% on OK-VQA) while improving accuracy (Figure 4; Figure 5; Section 4.2).

## 5. Experimental Analysis
- Evaluation setup (Section 4):
  - Datasets:
    - Training: SFT uses `DeepMMSearchVQA` (10k balanced samples derived from InfoSeek). RL uses FVQA-train, which has more image-search questions to reinforce multimodal retrieval (Section 4.1).
    - Testing: InfoSeek (2k sampled), Encyclopedic-VQA (Enc-VQA; 2k sampled), SimpleVQA, DynVQA, OKVQA, A-OKVQA (Section 4.1).
  - Workflows compared (Section 4.1):
    - Direct Answer (no retrieval).
    - RAG Workflow: exactly two fixed steps—1) image search results fed in, 2) text search results fed in.
    - Prompt-based Search Agent: prompt the base model to use tools at test time (no finetuning).
    - Web-search-equipped MLLMs: models trained to use tools (DeepMMSearch-R1 vs. MMSearch-R1).
  - Metric and judge:
    - LLM-as-Judge using `gpt-5-chat-latest`: binary correctness vs. ground truth (Section 4.1 “Evaluation Metric”; Appendix E.5 provides the prompt).
- Main quantitative results (Table 1; Section 4.2):
  - Overall accuracy (average across six datasets):
    - `DeepMMSearch-R1-7B (RL)`: 57.13
    - `DeepMMSearch-R1-7B (SFT)`: 56.23
    - `MMSearch-R1-7B`: 50.56
    - `GPT-4o` (Direct Answer): 50.16
    - `o3` (Direct Answer): 60.38
  - Per dataset for `DeepMMSearch-R1-7B (RL)` (Table 1):
    - InfoSeek: 47.51
    - Enc-VQA: 52.25
    - SimpleVQA: 55.87
    - DynVQA: 45.87
    - OKVQA: 67.80
    - A-OKVQA: 73.45
  - Against other workflows:
    - Compared to the best RAG workflow baseline (o3 at 47.49 average), DeepMMSearch-R1-7B (RL) is +9.64 points; compared to the Qwen2.5-VL-7B RAG workflow (36.00), the gain is +21.13 (Section 4.2).
    - Compared to prompt-based search agents (best 50.94), it is +6.19 (Section 4.2), with the text noting an average gain of +8.89 vs prompt-based baselines (depends on baseline aggregation).
- Do the experiments support the claims?
  - Claim: Training the model to use tools beats fixed pipelines and prompting. Supported by consistent gains over RAG and prompt-agent workflows in Table 1 and by ablations showing improved performance with self-reflection and crop search (Figure 3-left).
  - Claim: Crop-based image search and multi-turn text search improve robustness. Supported by an average +1.75 boost from cropping and additional gains from allowing multiple text searches (Figure 3-left).
  - Claim: RL refines tool use. Supported by tool-usage shifts—more selective cropping, more frequent multiple text searches, better alignment with dataset difficulty (Figure 4; Figure 5; Section 4.2).
- Ablations and diagnostics:
  - Search-balance ablation: shows 50:50 search-required vs. search-free in SFT is best on average; skewed data causes over/under-searching depending on the benchmark (Figure 3-right).
  - Knowledge taxonomy sampling: uniformly sampling across categories outperforms random sampling (Figure 3-right).
  - General VQA “no-search” capability: Minimal changes vs the base Qwen2.5-VL-7B on OCRBench, MMVet, AI2D, MathVista MINI, MMBench, DocVQA, InfoVQA (Table 2), indicating SFT+RL did not harm general skills notably—consistent with LoRA and KL regularization choices (Section 4.2).
- Limitations of the evaluation to keep in mind:
  - LLM-as-Judge (`gpt-5-chat-latest`) is also used as the reward model and as the summarizer for search results (Section 4.1 “Multimodal Search Tools”; “We employ gpt-5-chat-latest as the LLM summarizer”; Section 3.2—reward uses `gpt-5-chat-latest`). This could bias evaluation toward stylistic patterns of the trained model and its summaries.
  - Web APIs are in-house; although the authors claim no changes were made to accommodate the model (Section 4.1), full replication depends on access to similar search infrastructure.

## 6. Limitations and Trade-offs
- Assumptions baked into the system:
  - One-call limit on image search per question (Appendix D); while text search can be repeated, image search cannot. This may hurt in cases where the first crop is imperfect and a second crop could help.
  - Reliance on high-quality summaries of web results; an LLM summarizer filters the top-5 results (Section 4.1). If summaries omit crucial detail or introduce errors, downstream reasoning is affected.
  - Binary correctness reward from an LLM judge simplifies supervision but may be coarse; borderline or partially correct answers are either 1 or 0 (Section 3.2).
- Scenarios not fully addressed:
  - Very long-horizon research tasks that require browsing inside retrieved pages, following links, or reading figures/tables. Current tools retrieve and summarize search results but do not support page-level navigation.
  - Cases requiring more than one image search (e.g., multiple entities or cross-image disambiguation).
  - Multilingual retrieval beyond English and broader multimodal inputs (e.g., video) are not tackled; future extensions are suggested (Conclusion).
- Computational cost and scalability:
  - Training is resource-intensive (SFT on 8×H100; RL on 32×H100; Section 4.1). Real-time deployment will also incur search API calls plus LLM summarization overhead for both text and image search results.
  - Context-length limits (8192 tokens; Section 4.1) motivate aggressive summarization; complex questions may still hit context ceilings.
- Methodological weaknesses or open questions:
  - Evaluation and reward rely on the same LLM family (`gpt-5-chat-latest`), which risks circularity: models may learn to optimize for judge preferences rather than true factuality. Human evaluation or cross-judge validation would strengthen claims.
  - Grounding errors propagate: if Grounding DINO mis-localizes the entity described by the referring expression, the subsequent image search will be off-target. RL reduces unnecessary cropping (Figure 5), but there is no explicit robustness check for grounding errors.
  - Dataset scope: SFT uses a distilled 10k set (Section 2). While carefully balanced, this size may limit coverage of rare knowledge types; the paper mitigates with RL on FVQA-train (Section 4.1) but broader coverage is an open avenue.

## 7. Implications and Future Directions
- How this changes the landscape:
  - Demonstrates that MLLMs can function as practical, multimodal “search-then-reason” agents when trained end-to-end with the right interfaces—structured tool tags, crop-based image search, and RL that teaches search efficiency (Figures 1–5; Table 1). This moves beyond rigid RAG pipelines toward adaptive, on-demand retrieval behavior.
- Follow-up research enabled or suggested:
  - Tool diversity and deeper browsing: add page navigation, citation extraction, image-text co-reading of retrieved pages, and structured verification to reduce hallucinations.
  - Learning to calibrate uncertainty: combine answer confidence with “decide to search” policies and termination criteria—all learnable with outcome-based rewards.
  - Multi-image and video settings: extend crop-based search and query refinement to temporal domains; integrate tracking and event localization.
  - Multilingual retrieval and cross-market search tooling: evaluate across languages and search engines; study robustness to differing indexes and ranking algorithms.
  - Evaluation rigor: incorporate human judgments and cross-judge LLM evaluations; measure transparency of tool-use logs and source attribution.
- Practical applications:
  - Consumer assistants that can identify unfamiliar visual entities (e.g., plants, landmarks, products) and report current, specific facts (price, speed, regulations) with sources.
  - Education and journalism: cross-check facts from images with up-to-date references; produce concise summaries with explicit tool call traces.
  - Enterprise search: internal-knowledge plus external-web retrieval with grounded, auditable tool-use steps and minimal unnecessary queries, reducing cost.

> Representative result: Table 1 shows `DeepMMSearch-R1-7B (RL)` achieves an average accuracy of 57.13 across six benchmarks, surpassing the prior multimodal search baseline (`MMSearch-R1-7B` at 50.56) and all RAG and prompt-agent configurations tested, while approaching the performance of `o3` direct-answer at 60.38.

> Representative insight: Figure 3-left quantifies the benefit of the two signature capabilities—multi-turn self-reflection in text search and cropped image search—showing consistent average gains, while Figure 4 and Figure 5 demonstrate that RL reduces unnecessary tool calls (especially overuse of cropping) without sacrificing accuracy.
