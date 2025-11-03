# DeepMMSearch‑R1: Empowering Multimodal LLMs in Multimodal Web Search

**ArXiv:** [2510.12801](https://arxiv.org/abs/2510.12801)
**Authors:** Kartik Narayan, Yang Xu, Tian Cao, Kavya Nerella, Vishal M. Patel, Navid Shiee, Peter Grasch, Chao Jia, Yinfei Yang, Zhe Gan
**Institutions:** Apple, Johns Hopkins University

## 1. Executive Summary (2–3 sentences)
Reasoning: To compress the paper’s essence, I identified what is technically new (cropped image search + multi-turn text search) and why it matters (open-world, up-to-date, knowledge-intensive multimodal Q&A). I cross-checked claims with the method (Sections 2–3) and results (Table 1; Figures 1–5).

`DeepMMSearch-R1` is a multimodal large language model that performs on-demand, multi-turn web searches using both image and text tools, enhanced by a targeted cropped-image search mechanism and iterative, self-correcting text queries. Trained via a two-stage pipeline (supervised finetuning on a new dataset `DeepMMSearchVQA`, then online reinforcement learning with `GRPO`), it achieves strong gains over prior multimodal web-search baselines and outperforms prompt-only and rigid RAG workflows across six knowledge-intensive VQA benchmarks (Table 1).

## 2. Context and Motivation
Reasoning: I reconstructed the problem the model targets, examined prior approaches, and identified where they fall short using the Introduction and comparative framing around Figure 1 and Section 1.

- The specific problem:
  - Multimodal VQA often requires information not present in the model’s internal knowledge. Questions may be long-tail, current events, or require external background knowledge (Section 1). Static training corpora quickly become outdated, making purely parametric knowledge insufficient for real-world assistants.
  - Example of failure due to stale knowledge: `Qwen2.5-VL` failing to identify an event location tied to a specific image (Appendix F.1 referenced in Section 1).

- Why it matters:
  - Practical assistants need up-to-date answers grounded in real-world, dynamic information (Section 1). This is both a real-world and methodological gap: models must know when and how to search, process noisy retrieved signals, and integrate multimodal cues.

- Prior approaches and their gaps:
  - `RAG` (retrieval-augmented generation) methods rely on static corpora and rigid “retrieve-then-generate” pipelines, often causing unnecessary or noisy retrieval (Section 1; related works B.2).
  - Prompt-based search agents “wrap” search around an untrained base model; they are brittle with noisy web content and lack learned tool-use policies (Section 1; B.3).
  - Early “web-search-equipped MLLMs” primarily support text search; a first multimodal attempt (`MMSearch-R1`) allows image search but is limited to a single tool call per tool and whole-image queries, which suffer from background clutter and distractors (Section 1; Figure 1 left; B.4).

- Positioning of this work:
  - `DeepMMSearch-R1` addresses two bottlenecks:
    1) It enables self-reflection and self-correction via iterative text-search refinement (multiple turns), rather than a single rigid query (Figure 1 right; Sections 3.2 and 4.2).
    2) It improves image search by generating a referring expression for the relevant entity and using `Grounding DINO` to crop that region, reducing noise from backgrounds (Section 1; detailed in Sections 3 and 4.1; Figure 1 right).

## 3. Technical Approach
Reasoning: I decomposed the system into components—data, tools, training, and inference—and explained their interactions using Sections 2, 3, and 4.1, with the tag schema and constraints grounded in the prompts (Appendix E) and implementation (Appendix D).

- System overview:
  - Base model: `Qwen2.5-VL-7B-Instruct` (Section 3.1).
  - Tools integrated (Section 4.1, “Multimodal Search Tools”):
    - `Text search tool`: takes a model-generated query; an in-house web search API retrieves top-5 results; an LLM summarizer condenses them (Appendix E.7).
    - `Grounding tool`: `Grounding DINO` grounds a model-generated referring expression to a bounding box (object/region) and produces a crop (Section 4.1).
    - `Image search tool`: given either the whole image or the crop, retrieves visually similar web images plus context; an LLM summarizer condenses top results.
  - Structured output schema (Section 2 and Appendix E):
    - Reasoning: `<reason>...</reason>`
    - Text search: `<text_search>...</text_search>`
    - Image search: `<img_search>...</img_search>` or `<img_search><img></img_search>` for full image
    - Interim retrieved content: `<information>...</information>`
    - Final answer: `<answer>...</answer>`
    - This schema lets the model decide when to search, which tool to use, and how to incorporate retrieved information.

- Cropped image search (how it works and why):
  - The model writes a “referring expression” for the entity relevant to the question (e.g., “the white bird flying over the water”) inside `<img_search>...</img_search>`.
  - `Grounding DINO` (an open-set phrase grounding/detection model) maps the expression to a bounding box and crops the image (Section 4.1).
  - The crop is sent to the image search tool, which reduces background distractors and returns more relevant context (Section 1; Figure 1 right; Section 4.2 reports the gains).

- Self-reflection and self-correction via multi-turn text search:
  - If retrieved information is insufficient or off-target, the model refines the text query in subsequent turns (Section 1; Figure 1 right).
  - This supports exploration in noisy web environments (e.g., changing “speed of egret” to “highest recorded speed of egret”).

- Two-stage training (Section 3):
  - Terminology:
    - `SFT` (Supervised Finetuning): fine-tuning on labeled conversations.
    - `LoRA`: low-rank adaptation modules that add a small number of trainable parameters to a frozen base; here rank r=8 (Section 3.1).
    - `GRPO` (Group-Relative Policy Optimization): a reinforcement learning algorithm that computes rewards relative to the group average of sampled rollouts (Section 3.2), stabilizing optimization under noisy rewards.

  1) Stage 1: `SFT` on `DeepMMSearchVQA` (Section 3.1; Section 2; Figure 2).
     - Data creation (Section 2; Figure 2 top & bottom):
       - Start from 200k `InfoSeek` training examples; generate multi-turn, tool-tagged conversations with distilled reasoning and web-retrieved info; keep only examples where predicted answers match ground-truth answers, yielding ~47k; sample 10k examples to balance a knowledge taxonomy and enforce 50:50 search-required vs search-free (Figure 2 bottom).
       - The tag schema enforces when to call which tool, and what to search; image search can be whole image or cropped (via referring expression).
       - The conversation includes `<information>...</information>` blocks with summarized search results, which are fed back to the model in subsequent turns.
     - Objective:
       - Standard causal language modeling loss over the target sequence (reasoning, tool calls, final answer) while masking the tokens from web-retrieved `<information>` so the model learns to reason and use tools, not to overfit web snippets (Section 3.1, training objective paragraph).
     - Implementation: only the LLM is fine-tuned; vision encoder and projection layers are frozen; LoRA rank 8 across transformer blocks (Section 3.1).

  2) Stage 2: Online RL with `GRPO` (Section 3.2).
     - Rollouts: the SFT model interacts with tools to produce multi-turn trajectories up to tool-call/turn limits (10 total calls; image search tools at most once; multiple text searches allowed; Appendix D).
     - Reward (Section 3.2):
       - Composite: accuracy score `s` (binary correctness vs ground truth) plus a format score `sfmt` for tag adherence; weighted as `R_total = (1 − λ_fmt) s + λ_fmt sfmt` with λ_fmt=0.1.
       - Correctness is judged by an LLM (`gpt-5-chat-latest`).
       - KL penalty to keep the policy close to a frozen reference (Section 3.2; Appendix D).
     - `GRPO` detail:
       - K rollouts per prompt; advantage for each rollout is reward minus the group mean; optimized with a PPO-style clipped objective plus KL regularization (Section 3.2, equation).

- Inference-time behavior (Sections 3.2 and 4.1; Appendix E.1/E.2):
  - The model runs a multi-turn loop, deciding when to call image search (whole vs cropped) or text search, incorporating `<information>` to either answer `<answer>...</answer>` or refine queries. It is explicitly trained to follow this schema and to be selective about tool use.

- Implementation details and constraints (Section 4.1; Appendix D):
  - Search APIs are in-house; top-5 results are summarized by `gpt-5-chat-latest` to fit context limits (Appendix E.7).
  - Max response length 8192 tokens; per-rollout tool calls capped (Appendix D).
  - SFT: 3 epochs, LoRA rank=8; RL: 30 epochs of `GRPO`, with KL penalty 0.001 and clip ratio 0.2 (Section 4.1).

## 4. Key Insights and Innovations
Reasoning: I separated incremental vs fundamental contributions by checking what is novel relative to B.2–B.4 and connecting to empirical evidence (Table 1; Figures 3–5).

- Cropped image search via referring expressions + `Grounding DINO` (fundamental):
  - What’s new: moves from whole-image reverse image search to targeted, question-conditioned crops extracted by grounding a model-generated description (Section 4.1; Figure 1 right).
  - Why it matters: reduces background noise and improves retrieval relevance, particularly for questions about a specific entity in a cluttered scene. Empirically, cropped image search yields an average +1.75 improvement across six datasets (Figure 3 left).

- Multi-turn, self-correcting text search (fundamental):
  - What’s new: the model systematically refines queries based on retrieved evidence, beyond single-shot queries (Figure 1 right; Sections 3.2 and 4.2).
  - Why it matters: real web data is noisy; iterative refinement enables the model to course-correct and converge on accurate facts. RL increases cases where the model issues multiple text searches (+1.54% on DynVQA and +2.64% on OK-VQA; Section 4.2).

- `DeepMMSearchVQA` dataset with structured tool traces (enabling; substantial):
  - What’s new: 10k multi-turn, multimodal VQA samples with balanced knowledge taxonomy and a 50:50 split of search-required vs search-free, including explicit `<reason>`, tool calls, and `<information>` blocks (Section 2; Figure 2).
  - Why it matters: teaches the model when/what to search and how to reason over retrieved content. Ablations show that balanced data and uniform sampling across knowledge categories improve average performance (Figure 3 right).

- Two-stage SFT + RL (`GRPO`) with careful masking and KL regularization (methodological; important):
  - What’s new: mask web tokens during SFT to prevent overfitting on retrieved content; use `GRPO` with a composite reward (accuracy + format) and a KL penalty to refine tool selection and efficiency (Sections 3.1–3.2).
  - Why it matters: SFT teaches the schema and capabilities; RL reduces unnecessary tool calls, increases judicious query refinement, and improves performance on search-heavy datasets (Figures 4–5; Section 4.2).

## 5. Experimental Analysis
Reasoning: I parsed the evaluation setups and baselines (Section 4.1), then inspected main results (Table 1), ablations (Figure 3), tool-use behavior (Figure 4), and general VQA retention (Table 2). I checked for consistency (e.g., RAG underperformance on search-free datasets) and looked for trade-offs (e.g., A-OKVQA drop after RL).

- Evaluation setup (Section 4.1):
  - Tools: same in-house text/image search APIs and summarizer for all relevant runs; top-5 results summarized by `gpt-5-chat-latest`.
  - Baseline workflows:
    1) Direct Answer (no retrieval).
    2) RAG Workflow (exactly two retrieval steps: image search then text search; rigid pipeline).
    3) Prompt-based Search Agent (no finetuning; prompt the model to use tools).
    4) Web-search-equipped MLLMs (models trained to use tools; includes `MMSearch-R1` and `DeepMMSearch-R1`).
  - Datasets (Section 4.1):
    - Training SFT: `DeepMMSearchVQA` (10k from `InfoSeek`).
    - RL: `FVQA` train split (more image-search-heavy).
    - Evaluation: `InfoSeek` (2k sampled), `Enc-VQA` (2k sampled), `SimpleVQA`, `DynVQA`, `OKVQA`, `A-OKVQA`.
  - Metric: LLM-as-judge with `gpt-5-chat-latest` (binary correct/incorrect), using a prompt that allows semantic equivalence (Appendix E.5).

- Main results (Table 1; averages across six datasets; higher is better):
  - Web-search-equipped MLLMs outperform RAG and prompt-only agents:
    - Quote from Table 1:
      > `DeepMMSearch-R1-7B (RL)` Average: 57.13  
      > `MMSearch-R1-7B` Average: 50.56  
      > Prompt-based: `Qwen2.5-VL-32B` 50.94; `Qwen2.5-VL-7B` 48.24  
      > RAG: `GPT-4o` 41.50; `Qwen2.5-VL-7B` 36.00; `Qwen2.5-VL-32B` 35.50
    - Interpretation: Training the model to use tools (and to decide when/how) is substantially better than plugging tools into an untrained model or enforcing a rigid retrieve-then-generate pipeline (Section 4.2).
  - Competitive with `GPT-o3` on this suite:
    - Quote from Table 1:
      > `GPT-o3` Direct Answer Average: 60.38; `DeepMMSearch-R1-7B (RL)` Average: 57.13
    - Interpretation: Despite being a 7B open model plus tools, it approaches the performance of a much larger closed model in direct answering settings across these benchmarks.

- Per-dataset highlights (Table 1):
  - Search-heavy datasets see notable gains with `DeepMMSearch-R1`:
    - `DynVQA`: 55.87 vs 53.90 (`MMSearch-R1`); prompt baselines: 48.67 (Qwen-32B agent).
    - `SimpleVQA`: 52.25 vs 36.85 (`MMSearch-R1`).
    - `Enc-VQA`: 47.51 vs 41.33 (`MMSearch-R1`).
  - RAG often hurts on search-light datasets:
    - Quote from Table 1:
      > On `OKVQA`, `Qwen2.5-VL-7B` drops from 63.10 (Direct) to 39.02 (RAG); `GPT-4o` drops from 71.96 (Direct) to 43.22 (RAG).
    - Interpretation: For datasets where many questions are answerable without search (OKVQA, A-OKVQA), forcing retrieval introduces noise (Section 4.2).

- Ablations (Figure 3):
  - Self-reflection/self-correction (allowing multiple text searches) increases average performance vs a single text search call baseline (Figure 3 left).
  - Cropped image search adds an average +1.75 across six datasets (Figure 3 left), with larger gains on newer datasets `SimpleVQA` and `DynVQA` (Section 4.2).
  - Data curation matters (Figure 3 right):
    - A 50:50 mix of search-required vs search-free is best in aggregate; too much search-required causes over-retrieval and hurts OKVQA/A-OKVQA, while too little harms InfoSeek/Enc-VQA/DynVQA/SimpleVQA.
    - Uniform sampling across knowledge taxonomy outperforms random sampling.

- Tool-use diagnostics (Figure 4; Section 4.2; Figure 5):
  - Dataset-specific tool use:
    - Quote (Section 4.2):
      > The model leverages tools for 87.7% of `DynVQA` samples vs 43.5% for `OKVQA`.
  - RL effects:
    - More image searches and “mixed” searches (image + text), reflecting multimodal evidence needs (Section 4.2).
    - More multi-text-search cases after RL (+1.54% `DynVQA`; +2.64% `OKVQA`).
    - Fewer cropped searches after RL (−36.81% `DynVQA`; −34.86% `OKVQA`) but higher accuracy—RL made cropping more selective, avoiding unnecessary crops (Section 4.2; Figure 5 shows SFT over-crops while RL avoids it).

- Retention of general VQA capability (Table 2):
  - Quote from Table 2:
    > `Qwen2.5-VL-7B-Instruct`: OCRBench 88.30, MMVet 68.30, AI2D 83.74, MathVista 68.20, MMBench 83.84, DocVQA 94.97, InfoVQA 82.58  
    > `DeepMMSearch-R1-7B (RL)`: OCRBench 87.60, MMVet 69.81, AI2D 82.57, MathVista 66.80, MMBench 83.76, DocVQA 94.63, InfoVQA 81.63
  - Interpretation: Minimal changes, suggesting SFT with LoRA and RL with a KL penalty preserve core VQA skills (Section 4.2).

- Are the experiments convincing?
  - Strengths:
    - Consistent improvements across multiple datasets and baseline families (Table 1).
    - Mechanism-level ablations (Figure 3) and behavior traces (Figures 4–5) tie gains to the claimed innovations (cropped search; iterative text search; RL selectivity).
  - Caveats:
    - Evaluation uses an LLM-as-judge (`gpt-5-chat-latest`) that also serves as reward model and summarizer; this could introduce correlated biases (Section 4.1; Appendix E.5–E.7).
    - In-house search APIs and summarization mean exact reproduction outside their environment may vary.

- Mixed/conditional results and trade-offs:
  - `A-OKVQA` sees a drop after RL (76.94 → 73.45; Table 1), consistent with RL prioritizing tool-use behaviors that are more beneficial on search-heavy datasets (Section 4.2).
  - RAG underperforms on search-light datasets; prompt-only agents are more stable but weaker than trained tool-users (Table 1).

## 6. Limitations and Trade-offs
Reasoning: I contrasted claims with constraints from Appendices and Section 4.1, and looked for design choices that could constrain generality or reproducibility.

- Image search can only be called once per question in their training/eval setup:
  - The prompts and implementation constrain the image search (whole or cropped) to a single invocation, while text search can be multi-turn (Appendix E.1; Appendix D). This partially addresses prior “single call per tool” limitations by enabling multi-turn text search, but image search remains single-shot; cropping helps, but re-cropping is not explored.

- Dependence on closed infrastructure:
  - Text/image search uses in-house APIs and `gpt-5-chat-latest` for summarization; reproducing identical results may be difficult without similar indices and summarizers (Section 4.1; Appendix E.7).

- LLM-as-judge + reward-model coupling:
  - The same family of LLMs (`gpt-5-chat-latest`) is used as reward model, summarizer, and final judge (Sections 3.2 and 4.1; Appendix E.5–E.7). While convenient, this raises the risk of overfitting to evaluator preferences or shared biases.

- Summarization layer may blur retrieval boundaries:
  - The summarizer compresses retrieved content (Appendix E.7). If the summarizer injects external knowledge not present in the retrieved snippets, it could confound whether improvements come from retrieval vs summarizer priors.

- Computational and engineering cost:
  - Online RL with `GRPO` (4 nodes × 8 H100 GPUs for 30 epochs; Section 4.1) is expensive; multi-turn web search incurs API costs and latency.

- Visual grounding reliance:
  - Cropping quality depends on generating an accurate referring expression and on `Grounding DINO`’s ability to localize it. Failures in grounding can lead to irrelevant image search results.

- Formatting schema rigidity:
  - The strict tag protocol is essential for tool use but may constrain general conversational flexibility or require careful orchestration in downstream systems.

- Scope:
  - Language coverage, multi-image reasoning, and temporal/video scenarios are not addressed. The maximum of 10 tool calls and 8192-token responses constrain very deep research chains (Appendix D).

## 7. Implications and Future Directions
Reasoning: I projected the practical and scientific impact based on demonstrated gains and identified constraints, then scoped concrete follow-ups that flow from the method and results.

- How this changes the landscape:
  - Demonstrates that multimodal web search can be made substantially more effective by (a) conditioning image search on grounded, question-specific crops and (b) training models to iteratively refine text queries. This shifts best practice away from rigid RAG pipelines toward trained tool-use policies with multimodal feedback loops (Table 1; Figures 3–5).

- Practical applications:
  - Digital assistants that identify unfamiliar objects, logos, places, or species in user images and fetch up-to-date facts (Figure 1 right).
  - Education and journalism tools that corroborate visual claims via image+text search, with transparent reasoning traces (`<reason>...</reason>`) and source summaries (`<information>...</information>`).
  - E-commerce and cultural-heritage scenarios where specific visual entities must be recognized and cross-referenced with current catalogs or archives.

- Research directions:
  - Multiple, iterative image searches: Allow re-cropping and re-querying (e.g., progressive zoom-in, multi-entity disambiguation) rather than a single image-search call (Appendix D/E.1).
  - Richer visual tools: Replace boxes with segmentation masks or keypoint-based referring expression grounding; explore detectors specialized for logos, text-in-image (OCR), or fine-grained species.
  - Better rewards and evaluation:
    - Use diverse, independent judges (human and automated) to reduce evaluator coupling.
    - Introduce verifiability rewards (e.g., citation groundedness, source quality) and penalize hallucinations.
  - Open, reproducible retrieval stack:
    - Pair the method with public search indices and lightweight, auditable summarizers to facilitate replication and reduce reliance on closed LLMs during both training and evaluation.
  - Cost and latency optimization:
    - Learn explicit tool-use policies that factor in token and API costs; cache and reuse retrievals across related turns or tasks.
  - Broader scope:
    - Extend to multilingual settings, multi-image/photo albums, and video frames; integrate temporal reasoning for time-sensitive questions (especially relevant for datasets like `DynVQA`).
  - Safety and provenance:
    - Add source attribution in outputs; integrate credibility signals and content filtering to mitigate misinformation risks called out in the Ethics Statement.

By unifying grounded image search with learned, iterative text retrieval—and proving the value of training for tool-use rather than relying on static pipelines—`DeepMMSearch-R1` provides a practical path to more capable, up-to-date multimodal assistants. The next leaps will likely come from iterative visual search, verifiability-aware rewards, and fully open retrieval stacks that make these systems easier to replicate, trust, and deploy.