# OCR-input & thinking A/B — artifacts for review

**Question:** does the LLM input source (raw **PyMuPDF** text vs **GLM-OCR** markdown) or the
**thinking** mode (high vs none) materially change DeepSeek-V4-Pro summary quality? This decides
whether to re-summarize the ~6k papers that already have summaries built from the old
(HF-OCR-or-PyMuPDF) input.

## What to look for
- **Correctness/completeness**: are equations, numbers, table values right under each variant?
- **PyMuPDF vs GLM-OCR**: does GLM-OCR's structured markdown win where math/tables live (section 3)?
- **think-high vs none**: is the lift real or ~constant polish (more table/figure refs)?

## Files per paper dir
- `00_INPUT_glm-ocr-markdown.md` — the GLM-OCR text fed to the model
- `01_section1_executive-summary__AB_input-x-thinking.md` — 4 cells (input × thinking)
- `03_section3_technical-approach__AB_input-x-thinking.md` — the discriminator section
- `10_GLM_full_sections-1-4__think-high.md` — full pipeline reference

## Papers

| arXiv | title | dir |
|---|---|---|
| 2205.10625 | Least-to-Most Prompting Enables Complex Reasoning in Large L | `2205.10625__least-to-most-prompting-enables-complex-reasoning/` |
| 2405.13729 | ComboStoc: Combinatorial Stochasticity for Diffusion Generat | `2405.13729__combostoc-combinatorial-stochasticity-for-diffusio/` |
| 2411.18966 | SVGS: Enhancing Gaussian Splatting Using Primitives with Spa | `2411.18966__svgs-enhancing-gaussian-splatting-using-primitives/` |
| 2501.01046 | SEDD: Scalable and Efficient Dataset Deduplication with GPUs | `2501.01046__sedd-scalable-and-efficient-dataset-deduplication/` |