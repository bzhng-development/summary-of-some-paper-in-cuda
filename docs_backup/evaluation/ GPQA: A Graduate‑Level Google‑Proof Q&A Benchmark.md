# GPQA: A Graduate‑Level Google‑Proof Q&A Benchmark

**ArXiv:** [2311.12022](https://arxiv.org/abs/2311.12022)
**Authors:** David Rein, Betty Li Hou, Asa Cooper Stickland, Jackson Petty, Richard Yuanzhe Pang, Julien Dirani, Julian Michael, Samuel R. Bowman
**Institutions:** Anthropic, Cohere, New York University

## 🎯 Pitch

GPQA creates a rigorous benchmark with graduate-level questions in science fields to tackle 'scalable oversight,' highlighting how non-experts can supervise advanced AI systems effectively. By ensuring questions remain challenging even for empowered non-experts and frontier models, this dataset becomes a critical testbed for developing oversight protocols that can prevent AI hallucinations and improve alignment techniques like RLHF, crucial as AI systems approach superhuman performance.

---

## 1. Executive Summary
GPQA introduces a hard, expert-written, graduate-level multiple-choice benchmark in biology, physics, and chemistry for studying `scalable oversight`—how non-experts can reliably supervise stronger AI systems. Through a carefully incentivized two-expert validation plus non-expert testing pipeline, the dataset ensures both objectivity (experts mostly agree on a single correct answer) and difficulty (skilled non-experts with unrestricted web access still perform poorly), creating a realistic testbed where oversight methods matter.

## 2. Context and Motivation
- Problem addressed:
  - As AI systems begin to outperform humans on niche scientific questions, it becomes difficult for human supervisors—especially non-experts—to verify correctness. This is the `scalable oversight` problem: supervising tasks where the supervisor cannot easily produce or check the true answer themselves (Introduction).
- Why it matters:
  - Alignment techniques like RLHF depend on the supervisor identifying correct outputs. If supervisors cannot judge correctness on frontier problems, systems may exhibit hallucinations or sycophancy with little recourse (Introduction).
- Gaps in prior work:
  - Existing QA benchmarks are typically easy for web-enabled non-experts or current models, or they draw answers directly from readily accessible resources. They do not simulate the “supervisor-can’t-tell” setting needed to test oversight protocols (Related Work).
- Positioning of this work:
  - GPQA constructs a dataset where:
    - Ground-truth answers exist and are validated by experts.
    - Non-experts with web access and ample time still struggle.
    - Frontier models also struggle, leaving room for oversight protocols to improve outcomes.
  - The dataset is designed explicitly for oversight experiments such as debate, market making, and recursive reward modeling (Sec. 2.1; Related Work).

Key definitions used throughout:
- `Scalable oversight`: methods that let weaker evaluators (often non-experts) reliably assess answers produced by stronger agents (agents or models that may know more).
- `Google-proof`: questions that remain difficult for skilled non-experts even with unrestricted internet search time.
- `Expert validator`: a domain expert (PhD or PhD-track in the relevant subdomain) who answers and critiques a question.
- `Non-expert validator`: an expert from a different domain who attempts the question with full web access (no LLMs) and reports resources used.
- `Post-hoc agreement`: after seeing the intended answer and detailed explanation, an expert indicates whether they now agree the question has a single, uncontroversially correct answer (Sec. 3.1).

## 3. Technical Approach
Step-by-step pipeline (Figure 1; Sec. 2.1):
1. Question writing by experts
   - Experts author graduate-level MCQs in their subdomain, plus detailed explanations for why the correct choice is right and distractors are plausible but wrong.
   - Requirements include clarity, objectivity, minimal pure calculation, and search resistance; questions should also be answerable even without options (Appendix A.5.1).
   - Writers label subdomain and estimate time invested. A canary string is embedded in the dataset to help future filtering from training corpora.
2. First expert validation
   - A second domain expert answers the question and provides detailed feedback for improving objectivity and difficulty. Bonus incentives reward feedback that leads to better second-expert outcomes and harder questions for non-experts (Sec. 2.1).
3. Question revision
   - The original writer revises the question based on feedback (or defends the original, if appropriate) (Sec. 2.1; A.5.3).
4. Second expert validation
   - A third domain expert answers the revised question and provides feedback. Their outcome and commentary are used to assess objectivity and classify mistakes (Sec. 3.1; Table 4).
5. Non-expert validation (difficulty check)
   - Three non-expert validators (experts in other domains) attempt each question. They must spend at least 15 minutes (average ~37 minutes; median 30) and can use any online resources except LLM assistants; many read papers or run code (Sec. 3.2).
   - This step verifies that questions remain hard even with strong search capabilities.

Incentive design (Sec. 2.1; A.4):
- Writers are rewarded when both experts agree and when non-experts fail; validators earn bonuses for correctness and feedback that increases difficulty. This directly aligns incentives with objectivity and hardness.
- Estimated average payment ≈ $95/hour (Sec. 2.1), supporting careful work.

Domains and subdomains (Sec. 2.2; Table 3):
- High-level domains: Biology (Molecular Biology, Genetics), Physics (nine subdomains), and Chemistry (Organic, General, etc.).
- Coverage in the extended set: Biology 105, Physics 227, Chemistry 214 (Table 3).

Dataset splits (Sec. 2.3):
- `GPQA Extended` (546 questions): all post-filtering questions except a small held-out set.
- `GPQA` main (448 questions): removes items likely non-objective or too easy for non-experts. It can include items where an expert initially missed but later demonstrated understanding, indicating the question is objective (Sec. 2.3; 3.1).
- `GPQA Diamond` (198 questions): the highest-confidence subset requiring two-of-two expert agreement (allowing inclusion when the second expert clearly explains their mistake) and ≤ 1/3 non-experts correct (Sec. 2.3).

Objectivity measurement (Sec. 3.1; Table 4):
- Beyond raw expert accuracy, objectivity is assessed through `post-hoc agreement` and careful manual categorization of expert mistakes (e.g., “clear mistake identified” vs. “answer is arguable”).
- This reduces the confound where a competent expert may still err on a very hard, specialized question.

Model baselines and prompting (Sec. 4; Table 5–7; Appendix A.3.1):
- Closed-book evaluations: `Llama-2-70B-chat`, `GPT-3.5-turbo-16k`, and `GPT-4` under zero-shot, few-shot, zero-shot chain-of-thought, and few-shot chain-of-thought (`CoT`) prompting. For few-shot CoT, the examples use the human-written explanations as reasoning traces (Sec. 4).
- Open-book evaluation: `GPT-4 with search` via a `self-ask`-style multi-hop search framework (Appendix A.3.1). A “backoff” variant uses the few-shot CoT answer when the search-using model abstains (Table 5 vs. Table 6; Table 7 on abstention).

Quality checks (Appendix A.2):
- Answer-only classifiers (T5-small/base span labeler and a single-layer CBOW with RoBERTa embeddings) cannot exceed chance accuracy, suggesting absence of exploitable spurious patterns in answer strings.
- Calibration: both experts and non-experts provide discrete confidences; non-experts are notably overconfident at most confidence levels, though both show modest ECE (expert 0.1259; non-expert 0.1188; Figure 5).
- Question length: median ~561 characters, median 146 tokens by tiktoken (Figure 4).

## 4. Key Insights and Innovations
- Rigorous, multi-actor validation pipeline purpose-built for oversight
  - Two independent expert validators with a revision stage, plus three web-enabled non-experts per question, is unusually stringent for QA datasets (Figure 1; Sec. 2.1). This yields both high objectivity and verified difficulty—exactly the oversight setting needed.
- Measuring objectivity beyond accuracy
  - The paper introduces a nuanced objectivity analysis using `post-hoc agreement` and manual categorization of expert errors (Sec. 3.1; Table 4). This distinguishes “expert mistake on a hard item” from “ambiguous question,” producing a conservative objectivity estimate of 73.6%–76.4% on the extended set.
- A “Google-proof” difficulty criterion operationalized
  - Non-experts who are themselves PhD-level in other fields and who spend ~37 minutes on average with full web access still achieve low accuracy, making the task suitable for realistic oversight experiments (Sec. 3.2; Table 2).
- Baselines that leave room for oversight research
  - GPT-4 with strong prompting achieves only ~39% accuracy (Table 5), well below expert performance (~65–81%, depending on split; Table 2), ensuring that non-expert supervisors cannot simply defer to the model but may benefit from oversight protocols.
- Dataset splits for different confidence levels
  - The `GPQA Diamond` split creates a high-certainty subset where models and methods can be compared with strong confidence in label correctness (Sec. 2.3; Table 2).

These are more than incremental tweaks; the end-to-end design specifically targets the supervisory challenge that typical benchmarks do not address.

## 5. Experimental Analysis
Evaluation design:
- Human evaluation:
  - Two domain experts per question (pre- and post-revision), plus three non-experts with full web access and time requirements (Sec. 2.1–2.3).
  - Experts self-report whether they have sufficient expertise; ≥90% do, rising to 97% on the Diamond split (Table 2).
  - Difficulty ratings by experts predict non-expert accuracy (Figure 3).
- Model baselines:
  - Closed-book: multiple prompting strategies, especially few-shot `CoT` using writer explanations as exemplars (Sec. 4; Appendix A.3.1).
  - Open-book: `GPT-4 with search` using a self-ask-style tool; “backoff to CoT” when the tool-using model abstains (Sec. 4; Table 5–7).

Main quantitative results:
- Objectivity and difficulty
  - Expert accuracy on the extended set: 64.8% (Table 2). Accounting for clearly identified expert mistakes yields 73.6%–76.4% objectivity (Sec. 3.1; Table 4).
  - Non-expert accuracy on the extended set: 34.1% (Table 2). They spend a median of 30 minutes and average 37 minutes per question (Sec. 3.2).
  - Expert “sufficient expertise” self-report: 90.7% extended; 93.5% main; 97.0% diamond (Table 2).
- Domain breakdown (extended set; Table 3):
  - Expert accuracies: Biology 66.7%, Physics 57.3%, Chemistry 72.0%.
  - Non-expert accuracies: Biology 43.2%, Physics 32.5%, Chemistry 31.4%.
  - Largest expertise gap in Chemistry (40.6 percentage points), suggesting chemistry questions were especially resistant to web-enabled non-experts.
- Model baselines (Table 5; Table 6; Table 7):
  - Best closed-book result: `GPT-4` with few-shot CoT ≈ 38.7% (extended), ≈ 39.7% (main), ≈ 38.8% (diamond).
  - `GPT-4 with search` has two faces:
    - Without backoff, accuracy ≈ 28% (Table 6).
    - With backoff to CoT when abstaining, accuracy ≈ 39.4% (extended), 41.0% (main), 38.8% (diamond) (Table 5).
    - High abstention is the key: >37% abstention for the search model vs. ~4% for GPT-4 few-shot CoT (Table 7).
  - Human comparison (Table 5):
    - Experts: ~65% (extended), ~72% (main), ~81% (diamond).
    - Non-experts: ~34% (extended), ~31% (main), ~22% (diamond).
- Additional analyses:
  - Calibration: experts and non-experts show similar overall ECE on average, but non-experts are overconfident at most confidence levels (Figure 5).
  - Spurious features: answer-only models perform at chance (Appendix A.2), reducing concern that trivial option-format artifacts drive performance.

Do the experiments support the claims?
- Yes, on objectivity: Dual expert validation, high “sufficient expertise” rates, and post-hoc agreement analysis with manual categorization provide a credible lower-bound estimate of objectivity (Sec. 3.1; Table 4).
- Yes, on difficulty: Non-experts’ low accuracy despite significant time and open web access demonstrates questions are indeed “Google-proof” (Sec. 3.2; Table 2).
- Yes, on model challenge: GPT-4 falls far short of expert accuracy, leaving room for oversight protocols to add value (Table 5).
- Nuances:
  - Non-expert accuracies on main/diamond are biased downward because split construction uses non-expert results as a filter (Sec. 3.2, Discussion under Sec. 4).
  - Tool-use limitations are evident: `GPT-4 with search` abstains frequently (Table 7), consistent with known difficulties in effective tool usage.

## 6. Limitations and Trade-offs
- Dataset size and statistical power:
  - The main set has 448 items; the diamond set has 198 (Sec. 2.3). This is sufficient for oversight experiments but small for fine-grained comparisons and training (Sec. 6).
- Selection effects:
  - Because subsets are filtered based on expert and non-expert outcomes, reported accuracies on those subsets are not unbiased estimates of general difficulty (Tables 2 and 5 note this).
- Domain coverage and annotator pool:
  - The dataset focuses on three natural-science domains with the largest subdomain being organic chemistry; other fields (e.g., engineering, law) were piloted but not retained due to quality challenges or contractor availability (Sec. 2.2).
  - Hiring via Upwork may introduce demographic or topical biases (Sec. 6).
- Non-expert definition:
  - Non-experts are highly skilled (often PhD-level in other domains), which may not represent typical end-users; results may not directly translate to general populations (Sec. 6).
- Format constraints:
  - Multiple-choice format simplifies supervision relative to open-ended reasoning; although writers ensured questions are answerable without options, most results are reported in MCQ form (Sec. 2.1; 2.3).
- Superhuman model assumption:
  - The dataset is an imperfect proxy for supervising truly superhuman systems on unknown problems; it uses hard, currently answerable questions, not unsolved scientific questions (Sec. 6).

## 7. Implications and Future Directions
- Field impact:
  - GPQA creates a concrete testbed for `scalable oversight` research where both non-experts and frontier models struggle, but experts can still verify answers. This enables controlled studies of oversight protocols (e.g., debate, market making, recursive reward modeling) with ground-truth labels (Introduction; Sec. 2.1; Related Work).
- What it enables:
  - Head-to-head comparisons of oversight strategies where a non-expert must interrogate and adjudicate between competing model arguments on a truly hard task.
  - Systematic testing of assistance features (retrieval, tool-use) and how they interact with oversight protocols, given the high abstention of tool-using GPT-4 (Table 7).
  - Study of annotator calibration and confidence elicitation, as calibration plots (Figure 5) show substantial overconfidence in non-experts—a key oversight challenge.
- Practical applications:
  - Training and evaluating model-assisted review workflows in science: peer review triage, literature synthesis under expert supervision, and experimental design assistance—contexts where non-expert oversight often occurs.
  - Development and evaluation of tool-augmented models and protocols that reduce abstention without sacrificing accuracy (Table 6–7).
- Future research directions:
  - Scale and breadth: expand to more domains (e.g., engineering, medicine, law) and to open-ended responses with rubric-based expert grading.
  - Long-form arguments: pair GPQA items with structured debates or adjudication interfaces to test whether protocols reliably elevate non-expert performance toward expert levels.
  - Dynamic, time-stamped questions: include items whose answers become known after a delay to approximate “unknown today, known tomorrow” supervision (Sec. 6).
  - Better tool use: investigate prompts, planners, and interfaces that reduce `GPT-4 with search` abstention and improve retrieval quality (Sec. 4; Table 7).
  - Leakage mitigation: keep strengthening canary-string and filtering practices to minimize contamination and preserve benchmark validity as models train on larger corpora.

Selected citations to figures and tables for quick reference:
- Pipeline: Figure 1 (Sec. 2.1)
- Domain counts and expertise gaps: Table 3 (Sec. 2.2)
- Split statistics and expert/non-expert accuracies: Table 2 (Sec. 2.3; Sec. 3)
- Objectivity categorization of expert disagreements: Table 4 (Sec. 3.1)
- Model baselines and human comparisons: Tables 5–6 (Sec. 4)
- Abstention rates: Table 7 (Sec. 4)
- Difficulty vs. non-expert performance: Figure 3 (Appendix A.2)
- Question length distribution: Figure 4 (Appendix A.2)
- Calibration plots: Figure 5 (Appendix A.2)

Representative quoted results:
- > “Expert accuracy [extended set]: 64.8%; Non-expert accuracy: 34.1%; Sufficient expertise reported by experts: 90.7%” (Table 2).
- > “GPQA [main set]: 448 examples; GPQA Diamond: 198 examples” (Sec. 2.3; Table 2).
- > “Few-Shot CoT GPT-4: 38.7% (Extended), 39.7% (Main), 38.8% (Diamond)” and “GPT-4 with search (backoff): 39.4%, 41.0%, 38.8%” (Table 5).
- > “GPT-4 with search abstains on 37.2% of questions (main set), vs. 4.0% for GPT-4 few-shot CoT” (Table 7).
- > “Chemistry shows the largest expertise gap (40.6 points): Expert 72.0% vs. Non-expert 31.4%” (Table 3).
- > “Conservative objectivity estimate: 73.6%, or 76.4% including demonstrations of understanding” (Sec. 3.1; Table 4).
