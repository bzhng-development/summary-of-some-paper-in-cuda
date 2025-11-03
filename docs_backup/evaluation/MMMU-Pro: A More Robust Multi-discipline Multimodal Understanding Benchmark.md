# MMMU-Pro: A More Robust Multi-discipline Multimodal Understanding Benchmark

**ArXiv:** [2409.02813](https://arxiv.org/abs/2409.02813)
**Authors:** Xiang Yue, Tianyu Zheng, Yuansheng Ni, Yubo Wang, Kai Zhang, Shengbang Tong, Yuxuan Sun, Botao Yu, Ge Zhang, Huan Sun, Yu Su, Wenhu Chen, Graham Neubig
**Institutions:** 

## 🎯 Pitch

MMMU-Pro introduces a stringent benchmark for multimodal AI, eliminating text-only shortcuts and requiring models to integrate visual and textual inputs effectively. By expanding choice options and embedding text within images, it highlights the significant challenge of achieving true multimodal comprehension, pushing the field closer to creating AI capable of real-world reasoning and analysis.

---

## 1. Executive Summary (2–3 sentences)
MMMU‑Pro is a new benchmark for evaluating multimodal AI that removes text-only shortcuts and forces models to both “see” and “read” at the same time. It builds on the widely used MMMU benchmark by (1) filtering out questions solvable by text-only models, (2) expanding multiple-choice options from 4 to up to 10, and (3) adding a vision-only input mode where the entire question and options are embedded in screenshots or photos. Across state-of-the-art systems, accuracy drops sharply compared to the original MMMU—revealing that true multimodal comprehension and reasoning remain unsolved (Table 1).

## 2. Context and Motivation
- Problem/gap addressed
  - Many high-profile multimodal benchmarks (including MMMU) contain items that can be solved without looking at the image, or by exploiting statistical patterns in the options. Text-only large language models (LLMs) can answer a nontrivial fraction of questions by using background knowledge or option-based cues (Figure 2 and Section 2.1).
  - Multiple-choice formats with only four options are vulnerable to guessing or “shortcuts,” overstating model competence.
  - Real-world usage often involves screenshots or photos where text and graphics are interleaved; most benchmarks feed clean text separately from images, under-testing the ability to integrate modalities.

- Why this matters
  - Overestimating multimodal capability can lead to brittle systems that fail in novel or realistic scenarios—e.g., reading GUI screenshots, interpreting text-heavy diagrams, or documents with embedded figures (Introduction; Figures 1 and 4).
  - A rigorous benchmark that resists text-only shortcuts and enforces visual-textual integration is necessary to measure real progress and guide model development.

- Prior approaches and their shortfalls
  - Earlier benchmarks (e.g., VQA, OK-VQA, MS COCO; Section 5) primarily test perception or short-form QA, not expert-level, cross-discipline reasoning.
  - MMMU improved coverage and difficulty (11.5K college-level questions across 30 subjects), but some items remained answerable via text-only correlations or option biases (Sections 2.1–2.2; Figure 2).
  - Related benchmarks and studies (e.g., MMBench, MM-Vet, BLINK) highlight perception gaps or hallucinations but do not specifically neutralize text-only shortcuts while enforcing joint reading and seeing.

- Positioning
  - MMMU‑Pro is a “stress test” version of MMMU: it removes text-only solutions, makes guessing harder, and adds a vision-only mode to test the core skill of integrating text inside images (Figure 1). It aims to be closer to real-world scenarios and to reveal where modern models still fail (Abstract; Sections 2–3).

## 3. Technical Approach
MMMU‑Pro’s construction has three sequential stages (Figure 1), followed by a multi-setting evaluation protocol. Below is the process and its rationale, step by step.

1) LLM filtering to remove text-only solvable questions (Section 2.2; Figure 2; Figure 3)
- Objective: keep only items that truly require visual input.
- Procedure:
  - Select four strong text-only LLMs: `Llama-3-70B-Instruct`, `Qwen2-72B-Instruct`, `Yi-1.5-34B-Chat`, and `Mixtral-8×22B-Instruct` (Section 2.2).
  - For every original MMMU question, strip the image and force each model to answer 10 times (sampling variability is important because LLMs are stochastic).
  - A question is marked “answerable by text only” if a model gets it right in more than 5 out of 10 attempts.
  - Remove any question where at least 3 of the 4 models meet that criterion “across the majority of trials.”
  - Randomly sample 1,800 questions from the remaining pool, uniformly across 30 subjects (60 per subject).
- Why this design:
  - Repeated trials prevent one-off lucky guesses from excluding a question.
  - The “3-out-of-4 models” rule is conservative: only broadly text-solvable items are removed.
- Evidence it works:
  - Figure 3 shows text-only LLM accuracy drops markedly after filtering and drops further after the next stage (option augmentation).

2) Option augmentation to up to 10 choices (Section 2.2)
- Objective: reduce option-based guessing and subtle option-pattern shortcuts.
- Procedure:
  - Expand each item from 4 choices to up to 10 choices.
  - Generation: GPT‑4o produces candidates; Claude 3.5 performs a first-pass filter; two rounds of human review ensure option quality, diversity, and non-ambiguity (Appendix C).
  - During this review, annotators also remove questions whose text no longer clearly ties to the image; this eliminates 70 more items, leaving 1,730 in the “Standard” set (Section 2.2).
- Why this design:
  - More options make “educated guessing” less effective and force models to reason precisely.
- Evidence it works:
  - Figure 3 shows text-only LLM accuracy declines further when options are augmented.

3) Vision-only input setting via screenshots/photos (Section 2.2; Figure 4)
- Objective: test whether models can “see and read simultaneously” when the question and options appear only inside an image.
- Procedure:
  - Annotators create photos or screenshots of each question (varying fonts, backgrounds, and display conditions) so that the entire prompt appears visually.
  - This yields a parallel “Vision” set of 1,730 items; combined with the 1,730 Standard items, MMMU‑Pro totals 3,460 items (Section 2.2).
- Why this design:
  - Mirrors real usage (e.g., users upload screenshots).
  - Forces OCR-like extraction and context understanding inside the same visual scene, increasing cognitive load and integration demands.

Evaluation protocol (Section 3.1)
- Three settings:
  1) “Standard (4 options)”—original-style 4 options, kept only for comparison/baseline difficulty.
  2) “Standard (10 options)”—the new MMMU‑Pro format used for scoring.
  3) “Vision Input”—the screenshot/photo version.
- The overall MMMU‑Pro score is the average of (2) and (3) (Section 3.1).
- Baselines include leading proprietary models (GPT‑4o, Claude 3.5 Sonnet, Gemini 1.5 Pro) and strong open-source multimodal models (LLaVA variants, InternVL2, Qwen2‑VL, VILA‑1.5, Pixtral/12B, Phi‑3.5‑Vision, Idefics3; Table 1).
- Prompting variants: “Direct” vs `Chain-of-Thought (CoT)` reasoning prompts; the better score per model is reported (Section 3.1; Figure 5; Appendix A).
  - CoT: an instruction that requests step-by-step reasoning before answering (defined in Appendix A).

Human performance approximation (Section 3.1; Appendix B)
- Rationale: running a new large-scale human study is costly; the underlying question content/difficulty is unchanged.
- Method:
  - Reuse MMMU’s original 90-expert annotations and solution traces for the overlapping items.
  - For questions lacking a written solution, simulate random choice among the expanded options.
  - Report “low/medium/high” expert bands (Table 1; Appendix B, Table 4 and Table 5; Equation (2) in Appendix B).
- Result: estimated human accuracy on MMMU‑Pro remains high (e.g., “High”: 85.4%), providing an upper bound reference (Table 1).

OCR capability and prompting (Section 3.4)
- Measure `OCR Accuracy` via normalized Levenshtein similarity between the model-extracted text and the ground-truth question text (equation in Section 3.4).
- Test whether an explicit “OCR-then-solve” prompt helps in the Vision setting (Appendix A).
- Finding: explicit OCR prompting changes accuracy very little (Table 2); OCR accuracy and Vision performance are not tightly correlated (Figure 6).

Additional diagnostics (Sections 3.5–3.7)
- Qualitative analyses of failure modes (Section 3.5; Figures 10–12, 21, 33).
- Error categorization on 60 GPT‑4o Vision errors: reasoning errors dominate (46%), perception 27%, knowledge 25%, annotation 2%, OCR 0% (Figure 7).
- Response-length analysis: in Vision, GPT‑4o outputs fewer analytical tokens but more “descriptive” tokens, and shorter overall responses (Figure 8).

Vision encoder comparison (Section 4; Table 3; Appendix E)
- Same Multimodal LLM trained with two vision encoders: self-supervised `DINOv2 ViT‑G‑14` vs language-supervised `SigLIP ViT‑SO400M‑14`.
- Observation: SigLIP is slightly better on MMMU (Val), but DINOv2 is better on MMMU‑Pro Vision (Table 3), hinting that strong visual features (not just language-aligned ones) help in text-in-image scenarios.

## 4. Key Insights and Innovations
1) Three-stage benchmark hardening that neutralizes text-only shortcuts
- What’s new: a systematic pipeline—text-only filtering, option augmentation, and vision-only input (Figure 1).
- Why it matters: Figure 3 demonstrates that filtering + more options reduce text-only LLM success rates substantially; Table 1 shows large drops for multimodal models compared to MMMU, indicating the new benchmark defeats earlier shortcuts.

2) A realistic “vision-only” mode that tests simultaneous reading and seeing
- What’s new: all text (question + options) is embedded in a photo/screenshot (Figure 4).
- Why it matters: This stresses the integrated skill users actually need from multimodal systems. Table 1 shows an additional accuracy decline from “Standard (10 options)” to “Vision,” sometimes dramatic (e.g., LLaVA‑OneVision‑72B: −14.0 points across those two settings; and −32.8 points relative to MMMU Val).

3) Evidence that OCR is not the bottleneck—reasoning is
- What’s new: explicit OCR prompting barely helps (Table 2). Models already read text decently (e.g., GPT‑4o OCR accuracy 92.3%), yet Vision accuracy remains much lower.
- Why it matters: Figure 6 shows weak correlation between OCR accuracy and Vision performance; error analysis (Figure 7) attributes the majority of failures to reasoning, not text recognition. The “hard part” is integrating text with diagrams, layouts, and relevant context.

4) Chain-of-Thought helps, but unevenly across models and disciplines
- What’s new: CoT usually improves results, sometimes substantially (Figure 5; Table 6). For example, in Vision, GPT‑4o gains +14.49 points in Tech & Engineering and +8.22 in Science (Table 6; Appendix D/Figure 9).
- Why it matters: CoT especially boosts structured, computation-heavy domains but can hurt in subjective areas (e.g., LLaVA‑OneVision‑72B drops −17.12 points in Art & Design; Table 6), highlighting instruction-following and format-stability issues in some models.

5) Vision encoders tuned for robust visual features may be preferable for text-in-image tasks
- What’s new: with the same MLLM and data, `DINOv2 ViT‑G‑14` edges out `SigLIP SO400M‑14` in the Vision setting (17.4 vs 16.7), while the reverse holds on MMMU Val (Table 3).
- Why it matters: text-rich, layout-heavy images may benefit more from strong general visual features than from language-aligned encoders alone.

## 5. Experimental Analysis
- Datasets and settings (Section 3.1)
  - Items: 3,460 total—1,730 “Standard (10 options)” + 1,730 “Vision.”
  - Subjects: 30 across 6 major disciplines; questions curated from college-level assessments (Section 2.1).
  - Metrics: accuracy (percentage correct). Overall MMMU‑Pro score is the average of “Standard (10 options)” and “Vision.”
  - Baselines: proprietary (GPT‑4o, GPT‑4o mini, Claude 3.5 Sonnet, Gemini 1.5 Pro) and open-source (InternVL2 family, LLaVA family, VILA‑1.5‑40B, Qwen2‑VL family, Pixtral‑12B, Phi‑3.5‑Vision, Idefics3‑8B; Table 1).
  - Prompting: both Direct and CoT prompts; best per-model used in aggregate reporting (Appendix A; Figure 5).
  - Human approximation: estimated low/medium/high bands (Table 1; Appendix B/Table 4).

- Main quantitative results (Table 1)
  - Accuracy drops when moving from MMMU (Val) to MMMU‑Pro:
    - GPT‑4o (0513): 69.1% (MMMU Val) → 54.0% (Standard‑10) and 49.7% (Vision). This is −15.1 and −19.4 points, respectively.
    - Claude 3.5 Sonnet: 68.3 → 55.0 (−13.3) and 48.0 (−20.3).
    - Gemini 1.5 Pro (0801): 65.8 → 49.4 (−16.4) and 44.4 (−21.4).
    - Qwen2‑VL‑72B: 64.5 → 49.2 (−15.3) and 43.3 (−21.2).
    - LLaVA‑OneVision‑72B: 56.8 → 38.0 (−18.8) and 24.0 (−32.8).
  - Human performance (approximated, Appendix B):
    - “High” experts: ~85.4% on MMMU‑Pro (both Standard‑10 and Vision columns in Table 1 show 85.4%), close to 88.6% on MMMU Val.
  - Random/Frequent choice baselines:
    - With 10 options, random is ~12–13% (Table 1: 12.8% Standard‑10; 12.4% Vision), setting a low baseline for chance performance.

- CoT prompting (Figure 5; Table 6; Appendix D/Figure 9)
  - CoT improves many models, but effects vary widely:
    - Example: Claude 3.5 Sonnet in Standard increases from 42.7% (Direct) to 55.0% (CoT).
    - By discipline in Vision (Table 6): GPT‑4o gains are large in Tech & Engineering (+14.49) and Business (+14.66), moderate in Science (+8.22), small in Art & Design (+1.58), and mixed elsewhere. LLaVA‑OneVision‑72B sees minimal or even negative gains in several areas.

- OCR accuracy vs performance (Table 2; Figure 6)
  - OCR similarity is high for many models (e.g., GPT‑4o 92.3%, Gemini 89.7%, LLaVA‑OneVision‑72B 87.8%).
  - However, explicit OCR prompting barely changes Vision accuracy (e.g., GPT‑4o: 49.7% with OCR prompt vs 49.4% without; Table 2).
  - Figure 6: models with similar OCR (e.g., LLaVA‑OneVision‑72B vs InternVL2‑Llama3‑76B) can have very different Vision accuracies, reinforcing that reasoning, not OCR, is the bottleneck.

- Failure analyses and robustness checks
  - Increasing options reveals “nearest-match” behavior: models often pick the closest plausible answer when more distractors exist (Section 3.5; Figure 11).
  - Vision-text integration raises cognitive load: even when text is extracted correctly, models miss the answer (Section 3.5; Figure 10, Figure 21).
  - Visual cue overemphasis: models latch onto salient imagery and miss textual context (Section 3.5; Figure 33).
  - Error taxonomy on GPT‑4o Vision (60 cases): 46% reasoning, 27% perception, 25% knowledge, 2% annotation, 0% OCR (Figure 7).
  - Output style shift: Vision responses contain more description but less analysis; total tokens also drop (Standard: 409 tokens total with 360 analytical vs Vision: 366 total with 258 analytical and 108 descriptive; Figure 8).

- Do the experiments support the claims?
  - Yes. The staged drops from MMMU Val → Standard‑10 → Vision (Table 1), the small impact of explicit OCR prompts (Table 2), and the error analysis (Figure 7) jointly support the central claim: current models struggle mainly with integrated visual-text reasoning rather than text extraction.

## 6. Limitations and Trade-offs
- Assumptions and scope
  - Multiple-choice format: even with 10 options, problems are framed as single-shot selection; this cannot capture interactive reasoning or open-ended generation.
  - The “vision-only” setting relies on static images; it does not test time-based or multi-step visual navigation (e.g., interactive GUIs).
  - Human performance is approximated from MMMU’s prior study rather than re-measured on MMMU‑Pro (Section 3.1; Appendix B), which may introduce estimation bias.

- Dataset construction trade-offs
  - LLM-based filtering may exclude some valid but borderline text-solvable items and keep some edge cases; thresholds (e.g., >5/10 and the 3-of-4 rule) are principled but not perfect (Section 2.2).
  - Option augmentation to 10 choices increases difficulty. While realistic for exams with many distractors, some tasks in the wild may not present so many options, so the setting intentionally stresses models.
  - Photos/screenshots vary in fonts/backgrounds (Figure 4), but do not cover all real-world degradations (e.g., severe blur, occlusion, handwritten notes).

- Computational considerations
  - Running text-only filtering with 4 large LLMs, 10 trials per question, is costly. Reproducibility may depend on access to comparable models and inference budgets.

- Open questions
  - Could models find new “vision-only” shortcuts (e.g., positional biases of options inside images)?
  - How to best disentangle pure perception errors from multimodal reasoning errors at scale, beyond manual analysis?

## 7. Implications and Future Directions
- How this work changes the landscape
  - MMMU‑Pro provides a stronger, more realistic bar for multimodal reasoning: top systems drop 16.8–26.9 points from MMMU Val to MMMU‑Pro Vision for several models (Table 1). Benchmarks that do not neutralize text-only shortcuts risk overstating capabilities.
  - The finding that OCR is largely “solved enough” for these tasks shifts attention toward cross-modal reasoning and context integration (Table 2; Figure 6; Figure 7).

- Practical applications
  - Evaluating readiness for tasks like:
    - Reading textbook pages with interleaved figures and captions.
    - Interpreting scientific diagrams, charts, or medical images with embedded labels (Figure 4; Appendix J).
    - Understanding screenshots of GUIs, dashboards, or forms where text and graphics coexist.

- Research directions (Section 4)
  - Scale language backbones judiciously: larger LLMs consistently help, but are not sufficient alone (Table 1).
  - Improve vision encoders and visual representation learning:
    - The DINOv2 vs SigLIP result (Table 3) suggests robust self-supervised visual features can benefit text-in-image tasks.
  - Better fusion architectures:
    - Develop cross-modal attention and feature fusion that preserve fine-grained layout and semantic relations between text and graphics.
  - Stronger reasoning data and inference-time methods:
    - Generate CoT data tailored to reasoning-heavy domains (Table 6; Appendix D).
    - Explore compute-efficient inference-time algorithms that reliably produce structured reasoning without breaking answer formats.
  - Train on text-rich images:
    - Use tooling that synthesizes screenshots embedding questions and options (Section 4), scaling exposure to complex layouts and mixed modalities.

> Key quantitative takeaways
> - Table 1: GPT‑4o (0513) drops from 69.1% (MMMU Val) to 54.0% (Standard‑10) and 49.7% (Vision). Claude 3.5 Sonnet: 68.3 → 55.0 and 48.0. LLaVA‑OneVision‑72B: 56.8 → 38.0 and 24.0.
> - Table 2: OCR prompt vs no‑OCR prompt differs by ≤0.8 points for most models; GPT‑4o OCR accuracy is 92.3% but Vision accuracy is 49.7%.
> - Figure 7: On GPT‑4o Vision errors, reasoning is the largest category (46%), OCR errors are effectively 0%.
> - Table 6: CoT gains for GPT‑4o in Vision are large in Tech & Engineering (+14.49) and Business (+14.66), smaller in Art & Design (+1.58).

Overall, MMMU‑Pro is a careful stress test that forces multimodal models to demonstrate the capability humans take for granted: reading, seeing, and reasoning jointly in diverse, text-rich visual contexts.
