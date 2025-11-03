# MMMU-Pro: A More Robust Multi-discipline Multimodal Understanding Benchmark

**ArXiv:** [2409.02813](https://arxiv.org/abs/2409.02813)

## 🎯 Pitch

MMMU-Pro presents a new standard for evaluating multimodal AI by rigorously filtering out questions solvable by text alone, expanding multiple-choice options, and embedding questions within images to enforce true vision-language reasoning. This benchmark reveals that prominent models perform far worse when deprived of text-only shortcuts, exposing significant weaknesses in genuine multimodal understanding and shifting the challenge from superficial perception to deep reasoning. By highlighting these critical gaps, MMMU-Pro drives progress toward more robust, human-like AI systems ready for complex, real-world tasks.

---

## 1. Executive Summary (2–3 sentences)
MMMU‑Pro is a redesigned evaluation suite for multimodal AI that removes text‑only shortcuts, expands multiple‑choice options, and forces models to read questions embedded inside images (“vision‑only input”). It matters because many high scores on prior benchmarks (e.g., MMMU) can be achieved without real visual reasoning; MMMU‑Pro exposes this gap, showing large accuracy drops of 16–27 percentage points across state‑of‑the‑art systems (Table 1), and shifting the main failure mode from perception to reasoning (Figure 7).

## 2. Context and Motivation
- Specific gap addressed:
  - The original MMMU benchmark evaluates college‑level multimodal reasoning, but many questions can be answered by large language models (LLMs) using text alone (Section 2.1, Figure 2). Two root causes are identified:
    - “Text‑only dependency”: the image is redundant for some items.
    - “Shortcut exploitation”: models infer the answer via statistical cues in the options without integrating the image.
- Why this is important:
  - Overestimating multimodal understanding risks brittle real‑world deployments—e.g., misreading forms, lab results, or diagrams when the model must both “see” and reason (Introduction).
  - Human cognition smoothly integrates text and graphics; current models often don’t. The paper aims to test that core skill (Figures 1 and 4).
- Prior approaches and shortcomings:
  - Existing benchmarks (e.g., VQA, OK‑VQA, MMBench, MM‑Vet) cover perception and some reasoning but are less rigorous on expert‑level multimodal integration or can be gamed by superficial patterns (Related Work).
  - Original MMMU raised the bar with college‑level, multi‑discipline questions, but still contained text-only‑solvable items and small option sets (4 choices), which inflate scores (Section 2.1, Figure 3).
- Positioning:
  - MMMU‑Pro builds on MMMU with three interventions to suppress shortcuts and stress true vision‑language reasoning: (1) text‑only LLM filtering, (2) option augmentation, and (3) a vision‑only input setting (Section 2.2; Figure 1).

## 3. Technical Approach
MMMU‑Pro is constructed in three sequential stages designed to strip out shortcuts and increase the need for integrated reasoning.

1) Filter out text‑only‑solvable questions (Section 2.2; Figure 1, left)
- How it works:
  - Four strong text‑only LLMs—`Llama‑3‑70B‑Instruct`, `Qwen2‑72B‑Instruct`, `Yi‑1.5‑34B‑Chat`, `Mixtral‑8×22B‑Instruct`—answer every MMMU question without images, 10 times each.
  - A question is flagged as “answerable” if any model gets it right in >5/10 trials. If at least 3 of 4 models answer it correctly by this criterion, the question is removed.
  - From the remainder, 1,800 questions are sampled (60 per subject across 30 subjects), then further refined to 1,730 after human review (details below).
- Why this design:
  - Repeated trials reduce stochastic “luck.” Requiring 3/4 models to succeed makes the filter conservative.
  - The process explicitly identifies items solvable from text patterns alone (Figure 2 shows concrete examples where a text‑only LLM answers correctly by using general knowledge and option cues).

2) Augment candidate options up to 10 (Section 2.2; middle of Figure 1)
- How it works:
  - Human experts, aided by `GPT‑4o` for generation and `Claude 3.5` for filtering, expand 4 options to as many as 10.
  - Two rounds of human validation ensure new distractors are plausible, non‑ambiguous, and require reasoning tied to the image. During validation, questions that still lack coherent image–text linkage are removed (70 items filtered, yielding 1,730).
- Why this design:
  - With more options, guessing is harder and “option‑pattern” shortcuts are less effective. Figure 3 shows accuracy of strong text‑only LLMs drops approximately by half after filtering and option augmentation.

3) Create a “vision‑only input” version (Section 2.2; Figure 4)
- How it works:
  - Every retained question is turned into a screenshot/photo where the question text and the choices are baked into the image itself (humans manually capture screens with varied fonts, backgrounds, and conditions).
  - Final benchmark: 3,460 items, half as standard (text + image files) and half as vision‑only (just a single image containing everything).
- Why this design:
  - Models must perform OCR‑like text reading and visual reasoning together, as humans naturally do when reading diagrams, screenshots, or posters. This simulates realistic inputs users actually send to AI systems.

Evaluation protocol (Section 3.1)
- Three settings per model:
  1) Standard, 4 options (for comparability with MMMU),
  2) Standard, 10 options (the robust variant),
  3) Vision‑only input.
- The benchmark also studies:
  - `CoT` (Chain‑of‑Thought) prompting: ask models to reason step‑by‑step.
  - An `OCR prompt`: first extract the text from the image, then solve (Appendix A).
- Human performance is approximated using original MMMU expert annotations and a conservative adjustment for questions lacking written solutions (Appendix B; Table 4 and Table 5).

## 4. Key Insights and Innovations
1) A principled, data‑driven filter for text‑only solvability
- What’s new:
  - Rather than manual judgments, text‑only solvability is empirically measured via repeated trials on multiple strong LLMs with a clear acceptance criterion (Section 2.2).
- Why it matters:
  - It operationalizes “image‑dependence” and removes items that inflate multimodal scores (Figure 3 shows sizable drops in text‑only LLM accuracy after filtering).

2) Option augmentation to 10 choices
- What’s new:
  - Systematic expansion and human‑validated distractors that are specifically designed to resist guessing and force reasoning (Section 2.2; Appendix C).
- Why it matters:
  - It reduces option‑based shortcuts. In Table 1, all models lose 10–20+ points when moving from MMMU (Val) to MMMU‑Pro’s 10‑option standard; for `GPT‑4o`, −15.1 points (69.1 → 54.0).

3) Vision‑only input setting
- What’s new:
  - Questions and options are embedded inside images (Figure 4). Models receive only the image, so they must simultaneously read, perceive, and reason.
- Why it matters:
  - This setting exposes weaknesses in integrating text and graphics. Table 1 shows further drops—e.g., `LLaVA‑OneVision‑72B` loses −32.8 points versus MMMU (Val) in Vision, far beyond the −18.8 in 10‑option standard.

4) Rigorous analysis of OCR vs reasoning and CoT effects
- OCR prompt: Table 2 reports negligible gains from explicitly prompting “first extract text, then solve” (e.g., `GPT‑4o` 49.4 → 49.7 in Vision). Figure 6 shows that high OCR accuracy does not guarantee strong reasoning.
- CoT prompting: Figure 5 shows CoT improves many models in both settings (e.g., `Claude 3.5 Sonnet` 42.7% → 55.0% in Standard). Table 6 and Figure 9 reveal CoT helps most in structured, calculation‑heavy disciplines (e.g., Tech & Engineering: +14.49 points for `GPT‑4o` in Vision), but can hurt in subjective domains (e.g., Art & Design for `LLaVA‑OneVision‑72B`: −17.12).

Collectively, these are more than incremental tweaks: they reshape evaluation to target true multimodal integration and reasoning, not just pattern matching.

## 5. Experimental Analysis
- Datasets, metrics, and setup (Section 3.1):
  - Dataset: 3,460 questions across 6 disciplines, 30 subjects, 183 subfields.
  - Settings: Standard‑4 options (for reference), Standard‑10 options, and Vision‑only input.
  - Models: Proprietary (e.g., `GPT‑4o`, `Claude 3.5 Sonnet`, `Gemini 1.5 Pro`) and open‑source (e.g., `Qwen2‑VL`, `InternVL2`, `LLaVA`, `VILA`, `MiniCPM‑V2.6`, `Phi‑3.5‑Vision`).
  - Scoring: Accuracy; MMMU‑Pro headline score is the average of the 10‑option standard and Vision scores (used elsewhere in the paper; Table 1 reports per‑setting scores side‑by‑side for clarity).

- Main quantitative results (Table 1):
  - Performance drops when increasing options from 4 to 10:
    > “`GPT‑4o (0513)`: 69.1 (MMMU Val) → 54.0 (Standard‑10), Δ1 = −15.1; `Claude 3.5 Sonnet`: 68.3 → 55.0, Δ1 = −13.3; `Gemini 1.5 Pro (0801)`: 65.8 → 49.4, Δ1 = −16.4.”
  - Additional drops in the Vision‑only setting:
    > “`GPT‑4o (0513)`: 49.7 in Vision, Δ2 = −19.4 vs MMMU Val; `LLaVA‑OneVision‑72B`: 24.0 in Vision, Δ2 = −32.8.”
  - Humans (approximated) remain far ahead:
    > “Estimated Human Expert (High): 85.4–88.6 across settings (Table 1 and Appendix B Table 4).”
  - Baselines bound the task difficulty:
    > “Random choice ≈ 12–13% on MMMU‑Pro’s 10‑option and Vision settings; Frequent choice baseline ≈ 12% as well (Table 1).”

- Do the experiments support the core claims?
  - Yes, across all families of models, the two MMMU‑Pro interventions (10 options, vision‑only) consistently depress performance relative to MMMU (Val), indicating MMMU‑Pro effectively reduces shortcuts (Section 3.2, Table 1).
  - Figure 3 further corroborates the construction pipeline by showing text‑only LLM accuracy steadily declines after filtering and option augmentation.

- OCR vs multimodal reasoning (Section 3.4):
  - OCR accuracy is high for top models (e.g., `GPT‑4o` 92.3%, `Gemini 1.5 Pro` 89.7%), yet Vision accuracy remains much lower (Table 2).
  - Explicit OCR prompting barely changes Vision accuracy (e.g., `GPT‑4o` 49.4 → 49.7).
  - Figure 6: models with comparable OCR can diverge widely in Vision reasoning (e.g., `LLaVA‑OneVision‑72B` has OCR ≈ `InternVL2‑Llama3‑76B`/`GPT‑4o‑mini` but is much worse on Vision accuracy), signaling reasoning—not text extraction—is the bottleneck.

- CoT prompting (Section 3.3):
  - Figure 5: CoT generally helps in both standard and vision settings; exceptions exist (e.g., `VILA‑1.5‑40B` drops).
  - Table 6 and Figure 9: CoT gains are largest in structured domains (Tech & Engineering: +14.49 for `GPT‑4o` in Vision; Science: +8.22). In Art & Design, CoT has minimal or negative impact.

- Qualitative failure analysis (Sections 3.5 and 3.6):
  - Reasoning is the dominant error in Vision (46% of annotated `GPT‑4o` errors), followed by perceptual (27%) and knowledge errors (25%) (Figure 7). OCR errors are effectively 0%.
  - Increased cognitive load in Vision (needing to read and see) shortens analytical reasoning: GPT‑4o outputs shift toward shorter “analytical” content and more “descriptive” tokens (Figure 8).
  - Common error modes documented with examples (Figures 10–43): picking “close” options when there are 10 distractors; misbalancing visual vs textual cues (e.g., WWI/WWII posters → mispredicting League of Nations vs United Nations, Figure 33).

- Ablations on vision encoders (Section 4; Table 3):
  > “On a fixed MLLM (Cambrian‑1 with Llama 3.1 8B), `DINOv2 ViT‑G‑14` yields 17.4% on MMMU‑Pro Vision vs `SigLIP ViT‑SO400M‑14` at 16.7%, despite SigLIP doing slightly better on MMMU (Val).”
  - Suggests that self‑supervised visual features may transfer better to text‑rich vision settings.

Overall, the experiments are broad (many models), controlled (three settings, consistent prompts), and include robustness checks (OCR, CoT, encoder ablation), giving credence to the central thesis: MMMU‑Pro is a tougher, more faithful test of multimodal understanding.

## 6. Limitations and Trade-offs
- Assumptions and scope:
  - Multiple‑choice format remains: even with 10 options, it may still allow residual heuristics or near‑miss reasoning (Appendix J examples show “closest answer” confusions).
  - The benchmark focuses on college‑level academic tasks; it may not capture all real‑world multimodal tasks (Limitations section).
- Vision‑only realism vs control:
  - Photos and screenshots add realism (fonts, lighting, backgrounds) but introduce variability that can conflate perception noise with reasoning difficulty (Section 2.2; Figure 4).
- Human performance estimation:
  - Human scores are approximated from the original MMMU experts rather than re‑collected on MMMU‑Pro (Appendix B). Although the method is conservative and justified (Equation 2; Table 5), it is still an extrapolation.
- Compute and evaluation cost:
  - Creating two versions (standard and vision) doubles evaluation effort; option augmentation and multi‑round human reviews increase curation cost (Section 2.2; Appendix C).
- Remaining shortcuts:
  - Even after filtering, some items may contain subtle statistical cues (Limitations); multi‑discipline breadth makes total removal difficult.

## 7. Implications and Future Directions
- How this changes the landscape:
  - MMMU‑Pro reframes “multimodal understanding” to require simultaneous reading and seeing, not just answering questions where the text is given separately. Result: headline accuracies shrink (Table 1), and error profiles shift toward reasoning (Figure 7), giving a more realistic picture of model capability.
- What research it enables or suggests (Section 4):
  - Model scaling and backbone choice:
    - Larger language backbones consistently help (Table 1; e.g., `GPT‑4o` > `GPT‑4o‑mini`), but Vision remains a bottleneck.
  - Vision encoder research:
    - Self‑supervised encoders (`DINOv2`) may offer better features for text‑rich, reasoning‑dependent scenes than language‑aligned encoders (`SigLIP`) in Vision‑only inputs (Table 3).
  - Better cross‑modal integration:
    - Architectures that deeply fuse visual and textual tokens (e.g., improved cross‑modal attention and feature fusion) are needed; OCR alone is insufficient (Table 2, Figure 6).
  - Reasoning data and algorithms:
    - Targeted Chain‑of‑Thought data generation for structured domains can deliver large gains (Figure 5; Table 6). Methods that allocate inference compute effectively (e.g., inference‑time algorithms) are promising avenues (Section 4).
  - Data generation tools:
    - The paper develops a tool that converts text‑image pairs into screenshot‑style vision‑only items, enabling scalable creation of “text‑rich, reasoning” training/eval data (Section 4).
- Practical applications:
  - More reliable assistants for reading scientific plots, medical images with embedded text, lab reports, math/engineering diagrams, and GUI screenshots (Figures 4 and 10–43).
  - Safer deployment: benchmarks like MMMU‑Pro act as a stress‑test before models are used in critical domains (Introduction; Sections 3.5–3.6).

---

Selected citations to figures and tables for quick reference:
- Construction pipeline and motivation: Figure 1, Figure 4, Section 2.2.
- Text‑only LLM solvability: Figure 2; filtering + option augmentation effects: Figure 3.
- Main results across models/settings: Table 1; per‑discipline CoT effects: Table 6, Figure 9; CoT overall: Figure 5.
- OCR analysis: Table 2; OCR vs Vision accuracy correlation: Figure 6.
- Error taxonomy and response length: Figure 7 and Figure 8.
- Vision encoder ablation: Table 3.
- Human performance approximation: Appendix B (Table 4, Table 5).
