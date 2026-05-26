# olmOCR: Unlocking Trillions of Tokens in PDFs with Vision Language Models

**ArXiv:** [2502.18443](https://arxiv.org/abs/2502.18443)

## 🎯 Pitch

olmOCR introduces a breakthrough open-source toolkit that leverages document-anchoring and a fine-tuned 7B vision-language model to convert diverse PDF documents—including those with complex layouts and poor scans—into accurate, linearized plain text. This innovation makes high-fidelity extraction of structured content (like tables, sections, and formulas) both affordable and scalable, outperforming top commercial and open models while cutting costs by over 30x. By unlocking the trillions of high-quality tokens trapped in global PDF archives, olmOCR empowers language model training and downstream AI applications with richer, higher-quality data at unprecedented scale.

---

## 1. Executive Summary

This paper introduces **olmOCR**, an open-source toolkit that converts PDF documents into clean, linearized plain text using a fine-tuned 7B vision language model, paired with a novel prompting technique called **document-anchoring** (injecting text blocks and position coordinates extracted from PDF internals alongside the rasterized page image to guide model generation). When evaluated on olmOCR-Bench, a newly curated benchmark of 1,400 PDFs with over 7,000 unit-test cases spanning math formulas, tables, old scans, and multi-column layouts, olmOCR outperforms both commercial API tools including GPT-4o and Gemini Flash 2 as well as dedicated open-source systems like Marker and MinerU, while converting documents at a cost of only $176 per million pages—more than 35× cheaper than GPT-4o in batch mode. The paper further demonstrates downstream impact through continued pretraining on OLMo 2, establishing that content extracted with olmOCR from the peS2o corpus yields a +1.3 percentage point average improvement on standard LM benchmarks over the original Grobid-based extraction, confirming that higher-fidelity PDF linearization translates directly to better language model training outcomes.

## 2. Context and Motivation

### The Core Problem: PDFs Contain Trillions of Tokens That Language Models Cannot Read

The fundamental problem this paper addresses is deceptively simple: **PDFs are everywhere, but their content is largely inaccessible to language model training pipelines.** The paper's abstract frames the stakes directly:

> "PDF documents have the potential to provide trillions of novel, high-quality tokens for training language models."

This is not a niche concern. As the paper notes (Section 1), the PDF format has been used to create trillions of documents since its inception, spanning academic papers, legal documents, public domain books, government records, corporate reports, and more. This content is qualitatively different from web pages—it tends to be longer-form, more carefully authored, and more information-dense than typical crawled web text. Yet the vast majority of current large-scale language model training corpora (Dolma, FineWeb, RefinedWeb, DCLM) are dominated by web-extracted text, with PDF content either absent or processed through lossy, heuristic-heavy pipelines that degrade the very structure that makes these documents valuable.

The paper is fundamentally motivated by a practical bottleneck in the LM data pipeline: there is an enormous amount of high-quality textual information locked in PDFs, but no existing tool can extract it faithfully, at scale, and at an acceptable cost. This is the gap the paper targets.

### Why the Problem Matters: The PDF Format Is Structurally Hostile to Text Extraction

To understand why this problem is challenging—and why it matters—we need to understand how PDFs actually store text. The paper illustrates this vividly in Figure 2, showing that a PDF does not store "the title" or "a paragraph" as coherent units. Instead, it stores **individual glyphs** with precise positioning metadata. For example, the paper's own title is represented character by character:

- `Character: 'o'` at transform matrix `(1.02, 0.0, 0, 1, 70.866, 709.481)`, font `JURTWD+Manrope-Bold`, size `24.79`
- `Character: 'l'` at transform matrix `(1.02, 0.0, 0, 1, 86.49, 709.481)`, same font
- And so on for each letter in `olmOCR`

This is because the PDF format originated not as a document exchange format but as a **print specification language**—a way to tell a printer exactly where to place ink on a physical page. The paper explains:

> "PDFs store not units of text—headings, paragraphs, or other meaningful prose elements—but single characters alongside their spacing, placement, and any metadata used for visual rendering on a page."

The consequences are severe for language model use. There is **no guaranteed reading order** (text might be stored in the order it was drawn, not the order it should be read). There is **no structural markup** (no heading tags, no paragraph boundaries, no table markup). Multi-column layouts, floating figures, footnotes, headers, footers, and page numbers are all encoded identically—as spatially positioned characters—with no semantic distinction. For born-digital PDFs (documents created by software like LaTeX or Word), some metadata is present in the PDF internals, but it is "highly noisy" (Section 2.2): reading order from internal streams is often scrambled, main content is interleaved with boilerplate, and extraction tools produce artifacts.

For scanned documents (the Internet Archive books in the training data, the old scans in the benchmark), there is no born-digital text at all—only images of text—so extraction requires pure visual processing.

These format-level challenges mean that the problem is not just "OCR is hard" but rather **"faithful linearization is hard"**: producing plain text that preserves the natural reading order, correctly identifies structural elements (headings, lists, equations, tables), removes peripheral content (headers, footers, page numbers), and does not hallucinate or truncate content. This is what the paper means by "content extraction and linearization"—the dual task of recognizing what is on the page AND arranging it in a coherent reading sequence.

### The Scale Problem Makes This Urgent

Cost amplifies the importance of this problem. The paper calculates (Section 1) that processing the 7.9 million PDFs in peS2o using GPT-4o in non-batch mode would cost approximately **$98.6 million**—clearly infeasible for academic or most commercial settings. Even at batch pricing, the cost is over $6,200 per million pages. This means that for organizations building language models at scale (processing billions of tokens from millions of documents), the choice of PDF extraction tool is not just about quality—it is a **major cost driver that can make or break a data curation budget**.

The paper quantifies this tension in Figure 1, which plots performance (on olmOCR-Bench) against cost per million pages. There is a clear Pareto frontier: GPT-4o achieves strong performance but at extreme cost; open-source tools like Marker and MinerU are cheaper but substantially lower quality; and olmOCR sits at a previously unoccupied point—highest performance at the lowest cost point on the frontier. The paper frames this explicitly as an **accessibility problem**: if high-quality PDF extraction requires expensive proprietary APIs, then most of the world's PDF content remains effectively inaccessible for LM training, particularly for academic researchers and smaller organizations.

### Where Existing Approaches Fall Short

The paper identifies three categories of existing solutions and explains the limitations of each (Section 5):

#### Pipeline-Based Systems (MinerU, Marker, Grobid)

These systems chain together multiple specialized ML models: one for layout segmentation, one for OCR, one for table parsing, one for reading order determination, and so on. While modular, this architecture introduces several failure modes:

- **Error propagation**: A mistake in layout segmentation (e.g., misidentifying a formula as a text block) cascades into downstream components that receive the wrong input type.
- **Heuristic brittleness**: Reading order algorithms often rely on hard-coded rules about column boundaries and text spacing that fail on non-standard layouts. The paper shows (Appendix G) that MinerU and Marker both produce garbled text on old scans, with letter-level errors ("bchaving" for "behaving," "suspect ihe" for "suspect the") that indicate the pipeline components are not coordinating effectively.
- **Hallucinated or missing content**: MinerU produced "No text produced" for the Lincoln letter example in Appendix G—the entire page content was lost. GOT-OCR 2.0 degenerated into massive numeric repetitions on the calculus exercises page, producing hundreds of repeated digits instead of the actual problem text.

Pipeline systems also tend to focus on extraction fidelity rather than linearization. They may correctly identify all the text on a page, but fail to arrange it in a coherent reading order for multi-column layouts or documents with floating elements. This is a critical distinction: for LM training, you need both correct extraction AND correct ordering.

#### End-to-End Models (Nougat, GOT Theory 2.0)

These models take page images as input and directly output plain text, avoiding the error propagation of pipeline systems. However, the paper identifies fundamental limitations:

- **They only see pixels**: The paper explicitly notes that end-to-end models "exclusively rely on rasterized pages" (Appendix A), meaning they ignore all digital metadata present in born-digital PDFs. This is a missed opportunity—the internal text streams in a PDF, while noisy, contain information that could help disambiguate reading order or content boundaries.

- **They hallucinate severely**: The paper observes that "prompting with just the page image was prone to models completing unfinished sentences, or to invent larger texts when the image data was ambiguous" (Appendix A). GOT-OCR 2.0's catastrophic failure on the calculus page (degenerating into repetitive numeric sequences) exemplifies this—when the model loses coherence, it produces output that is worse than nothing, actively corrupting the training data.

- **Limited scale**: Nougat, for example, was trained primarily on academic papers and does not generalize to the diverse document types (legal documents, brochures, old scans, dictionaries) that real-world PDF collections contain.

#### Commercial VLM APIs (GPT-4o, Gemini Flash 2)

The paper tested these as both baselines and data generation sources. They are powerful but have critical drawbacks:

- **Cost**: As noted, GPT-4o costs over $6,200 per million pages even at batch pricing. This makes large-scale processing economically infeasible for most use cases.

- **Unreliable fidelity**: The paper found that GPT-4o on its own (without anchoring) "does not produce sufficiently high-fidelity plain text... for high-density pages or complex layouts, it is prone to omitting content, rewriting or completing content in a manner unfaithful to the original, or captioning images when not instructed to do so" (Section 2.2). This is a subtle but crucial point: a VLM's general-purpose training makes it *helpful*—it tries to summarize, complete, or interpret content—but OCR requires *faithful* reproduction, not interpretation.

- **API constraints**: For sensitive documents (legal, medical, proprietary research), sending PDFs to commercial APIs may be infeasible due to privacy, security, or contractual restrictions. The paper does not belabor this point, but it is implicit in the emphasis on open-source release.

- **Gemini-specific issues**: During data generation experiments, Gemini 1.5 was "eliminated due to frequent RECITATION errors" (Section 2.2, footnote). This is a specific failure mode where the model refuses to process content it recognizes as potentially copyrighted, making it unsuitable for indiscriminate batch processing.

#### A Structural Blind Spot in Prior Work

The paper makes an observation that is not explicitly stated but is woven throughout its methodology: **prior approaches treat PDFs as either purely visual (end-to-end models that only see pixels) or purely structural (pipeline tools that only read PDF internals), when in fact most PDFs contain BOTH—and neither source alone is sufficient.** Born-digital PDFs have internal text streams that are useful but unreliable; scanned PDFs have no internal text at all but contain visual structure that models can learn. A system that can leverage both modalities—the rasterized page image AND the noisy internal metadata—when available, while falling back to pure visual processing when metadata is absent, would be more robust. This insight motivates document-anchoring.

### Conflicting Needs: Quality vs. Scale vs. Cost

The paper is also motivated by a three-way tension that prior work has not resolved:

1. **Quality**: The extraction must be faithful—no hallucinated content, no omitted paragraphs, correct reading order, proper handling of equations and tables. Errors in extraction propagate into language model training, where they can cause training instabilities or degrade downstream performance (the paper cites Dolma and FineWeb experiences with this).

2. **Scale**: The system must process millions of documents. A tool that works perfectly on 1,000 pages but takes days for 100,000 is useless for LM training data curation.

3. **Cost**: The system must be affordable. GPT-4o demonstrates that VLMs *can* do this task well, but at a price point that makes it inaccessible for large-scale processing.

The existing tool landscape forces a choice between these three: pipeline tools offer low cost at moderate scale but low quality (Figure 1, GOT-OCR passes only ~48% of tests); commercial VLMs offer higher quality but at prohibitive cost; and open-source end-to-end models offer intermediate quality but poor scaling behavior. The paper's explicit goal is to achieve **high quality, at scale, at low cost**—a combination that did not previously exist.

### How the Paper Positions Itself

The paper positions olmOCR not as a fundamentally new model architecture but as a **system integration and data strategy** that makes existing VLM capabilities practical for this task. The key conceptual moves are:

- **Treating GPT-4o as a teacher, not a deployment solution**: Rather than prompt-engineering GPT-4o for production use (which would be expensive and API-dependent), the paper uses it to generate silver-standard training data (olmOCR-mix-0225) and then distills that knowledge into a much smaller, open-source 7B model. This is a data-centric approach: the innovation is in *how you get the training signal*, not in novel architectures.

- **Document-anchoring as a bridging mechanism**: By feeding PDF-internal text and coordinates alongside the page image, the model gets the best of both worlds—the visual fidelity of the rendered page AND hints about content structure from the digital metadata. This is what enables the 7B model to match or exceed GPT-4o on the benchmark despite being dramatically smaller. The paper frames this not as a new prompting trick but as a way to make the model's job easier by providing redundant signal, reducing the hallucination pressure on the visual processing.

- **Benchmark construction as a contribution**: The paper explicitly argues that existing benchmarks (FUNSD, SROIE, RVL-CDIP, PubTabNet) are too narrow—they focus on single document types or single extraction tasks—and use evaluation metrics (exact string match against gold tokens) that make cross-tool comparison difficult. olmOCR-Bench is designed with unit-test-style pass/fail rules that are "simple, unambiguous, and deterministically machine-verifiable" (Section 3), avoiding LLM-as-judge biases and enabling fair comparison across tools with different output formats. This is a methodological contribution that the paper positions as essential for rigorous evaluation in this space.

- **Downstream validation, not just intrinsic metrics**: Unlike most prior OCR work, which evaluates only on extraction accuracy metrics, the paper demonstrates that better extraction translates to better language model training. The continued pretraining experiment (Section 4.2) shows that replacing Grobid-extracted peS2o content with olmOCR-extracted content yields measurable benchmark improvements. This closes the loop: the paper is arguing that PDF extraction quality is not just an academic metric but a real bottleneck in the LM training pipeline, and that solving it has measurable downstream impact.

### The Scale of the Opportunity

The paper's motivation is ultimately quantitative: the PDF Association (cited in the paper) estimated trillions of PDF documents exist. Even processing a tiny fraction of these—the 7.9 million academic papers in peS2o—represents tens of billions of tokens. The paper estimates that at 1,000 tokens per page, processing all of peS2o with GPT-4o would cost approximately $10.3 million in H100 usage alone (Section 4.3). By reducing that cost to roughly $1,400 for the same corpus, olmOCR makes it **economically feasible** to include large-scale PDF corpora in LM pretraining data mixes. This is the practical significance: it unlocks a data source that was previously too expensive to use at meaningful scale, and the quality improvements suggest this new data is genuinely valuable for LM training, not just more of the same.

## 3. Technical Approach

### 3.1 Reader Orientation

**What the system is:** olmOCR is a Python toolkit that combines a fine-tuned 7B-parameter vision-language model (derived from Qwen2-VL-7B-Instruct) with a specialized prompting technique called document-anchoring to convert PDF pages into clean, linearized plain text in natural reading order, preserving structured elements like tables, equations, lists, and sections.

**What problem it solves:** The system addresses the fundamental mismatch between how PDFs store content (as individually positioned glyphs with rendering metadata, lacking any semantic structure or guaranteed reading order) and what language model training pipelines need (coherent, faithfully extracted plain text that preserves logical document structure). The "shape" of the solution is a VLM fine-tuned on silver-standard data generated by prompting GPT-4o with both the rasterized page image and noisy text extracted from PDF internals, then deployed with an efficient batch inference pipeline that scales from one to hundreds of GPUs.

### 3.2 Big-Picture Architecture (Diagram in Words)

The olmOCR system has six major components, which operate in two distinct phases—**data generation and training** (creating the model) followed by **inference and deployment** (using the model at scale):

1. **PDF Crawling and Filtering Pipeline** — acquires a diverse training corpus of 260,000 pages from over 100,000 publicly available PDFs, filtering for English language, parseability, and content quality.
2. **Document-Anchoring Engine** — extracts text blocks and their spatial coordinates from PDF internals (via pypdf) and constructs prompts that combine these anchored text hints with rasterized page images for input to a VLM.
3. **Silver Data Generator (GPT-4o Teacher)** — processes the crawled PDFs through GPT-4o with document-anchored prompts, structured JSON output schema, and careful instruction design to produce high-fidelity linearized plain text as supervision targets.
4. **Fine-Tuned VLM (olmOCR-7B-0225-preview)** — a Qwen2-VL-7B-Instruct model fine-tuned on the GPT-4o-generated silver data, learning to produce linearized text directly from document-anchored prompts without requiring the expensive teacher model.
5. **olmOCR-Bench Evaluation Framework** — a separate benchmark of 1,400 PDFs with 7,010 pass/fail unit tests spanning text presence, text absence, reading order, table accuracy, and formula accuracy, used to evaluate and compare extraction quality across tools.
6. **Batch Inference Pipeline** — a production deployment system built on SGLang and vLLM that coordinates GPU workers, handles retries for malformed JSON outputs, manages document-anchoring prompt construction, and scales from single-node to multi-node processing with cloud storage coordination.

**Information flow during training:** Crawled PDFs → document-anchoring extracts text blocks + coordinates → GPT-4o receives anchored prompt + page image → GPT-4o produces structured JSON with linearized text → silver dataset olmOCR-mix-0225 → fine-tune Qwen2-VL-7B-Instruct → olmOCR-7B-0225-preview.

**Information flow during inference:** Input PDF → document-anchoring extracts text blocks + coordinates → anchored prompt constructed → fine-tuned model generates structured JSON → JSON parsed to extract natural text → retry on failure (up to N times, with fallback to plain text extraction) → final clean plain text output.

### 3.3 Roadmap for the Deep Dive

- **First**, the data acquisition pipeline (Section 2.1) — how PDFs are crawled, filtered, and sampled, because the composition of this training corpus determines what document types the model can handle and directly shapes the model's capabilities.
- **Second**, document-anchoring as a technique (Appendix A) — what it is, how it works mechanically, and why it is necessary, since this prompting strategy is the key enabler for both data generation and model inference.
- **Third**, the silver data generation process (Section 2.2) — how GPT-4o is prompted, what structured output is enforced, and how quality is maintained, because this is the source of all supervision for the fine-tuned model.
- **Fourth**, model training (Section 2.3) — the fine-tuning recipe, hyperparameters, prompt modifications, and training dynamics, since the goal is to understand how a 7B model can match or exceed its much larger teacher.
- **Fifth**, the inference pipeline (Appendix D) — how the trained model is deployed at scale, including batching, retry logic, rotation handling, and decoding strategies, because practical throughput and reliability are core contributions of the work.
- **Sixth**, the olmOCR-Bench construction methodology (Section 3) — the unit-test design philosophy, document sourcing strategies, and test case creation for each category, since the benchmark is central to the paper's evaluation claims and represents a methodological contribution in itself.

### 3.4 Detailed, Sentence-Based Technical Breakdown

This is primarily a **systems and data-centric paper** whose core idea is that a small, open-source VLM, when fine-tuned on high-quality silver data generated by a much larger proprietary model using a specialized prompting technique that leverages PDF-internal metadata alongside visual input, can match or exceed the larger model's extraction quality at a fraction of the cost, and that this capability can be packaged into an efficient, production-ready toolkit that scales to millions of documents.

---

#### PDF Crawling and Filtering Pipeline

The training corpus olmOCR-mix-0225 is constructed from two sources, summarized in Table 1: **web-crawled PDFs** (96,929 unique documents, 240,940 pages) and **Internet Archive public domain books** (5,896 unique documents, 17,701 pages), totaling 102,825 documents and 258,641 pages. The web-crawled set is sampled from "an internal dataset of 240 million PDFs crawled from public internet sites," meaning the paper's parent organization (Ai2) already possessed a large-scale web crawl that included PDFs alongside HTML pages. This is an important practical detail: the paper does not describe a new web crawl but rather leverages existing infrastructure for PDF acquisition.

The Internet Archive component serves a specific purpose: while web-crawled PDFs are predominantly **born-digital** documents (created by software like LaTeX, Word, or InDesign, containing digital text streams in the PDF internals), the Internet Archive books are predominantly **image scans**—raster images of physical pages. Including both types ensures the training data covers the two major categories of PDFs the model will encounter in practice, and it tests whether document-anchoring degrades gracefully when no digital metadata is available (it doesn't; the model falls back to pure visual processing).

**Filtering steps** are applied to remove low-quality or unsuitable documents:

1. **Language filtering**: The Lingua package (Emond, 2025) is used to identify and remove non-English documents. This is a fast, statistical language detection library that operates on character n-gram frequencies, chosen for speed at scale over heavier neural alternatives. The paper does not specify exact language confidence thresholds, only that non-English documents are filtered out.

2. **Parseability filtering**: Any document that "failed to be parsed by pypdf" is removed. This catches corrupted PDFs, encrypted PDFs (where the internal structure cannot be read), and malformed files that would cause downstream extraction failures. This is a practical necessity: if pypdf cannot extract the internal text streams, document-anchoring cannot provide anchored hints.

3. **Content filtering**: Documents containing "spam keywords" are removed, along with fillable forms and documents whose extracted text is "too short." These heuristics target common low-quality PDFs: spam PDFs (SEO-optimized documents generated programmatically), interactive forms (which have minimal meaningful prose content), and near-empty documents. The paper references a specific implementation file at `/olmocr/filter/filter.py#L14-L112` for the exact heuristics.

4. **Page sampling**: From each surviving PDF, "up to three pages uniformly at random" are sampled. This caps the contribution of any single document to three pages, preventing long documents (e.g., 500-page books) from dominating the training distribution while still capturing within-document variation. The uniform random sampling means any three pages are equally likely, which avoids bias toward first pages (which tend to be title pages with atypical layouts) or last pages (which tend to be references or indices).

The resulting document type distribution, estimated by sampling 707 pages and classifying them with GPT-4o (prompt in Appendix E.3), shows a heavy skew toward academic papers (55.9%), with brochures (11.2%), legal documents (10.2%), books (6.8%), table-heavy documents (5.6%), diagram-heavy documents (4.7%), and slideshows (1.9%) making up the remainder (Table 2). This distribution is not balanced—academic papers dominate—but the paper does not claim balance; it claims diversity. The "Other" category at 3.7% captures document types that don't fit the predefined taxonomy. The classification was done with GPT-4o, not manual annotation, so the breakdown is approximate but sufficient for characterizing the training distribution.

A critical design choice here is **what is NOT filtered**: the paper does not filter by topic, quality of writing, presence of equations or tables, or visual complexity. The goal is to create a training set that reflects the natural diversity of PDFs on the public web, including messy, complex, and poorly formatted documents, so the model learns to handle them rather than only performing well on clean, well-structured PDFs.

---

#### Document-Anchoring: Concept, Mechanism, and Rationale

Document-anchoring is the paper's central technical innovation for making VLM-based PDF extraction reliable. It is described in detail in Appendix A, but its role spans the entire system: it is used when prompting GPT-4o for silver data generation, when fine-tuning the model, and during inference with the trained model.

**The core insight** is that most PDFs are born-digital documents that contain both **visual information** (the rendered page, which humans read) and **digital metadata** (internal text streams, font information, and spatial coordinates that specify where each character should be drawn). Traditional OCR systems (Tesseract, GOT-OCR) use only the visual information. Traditional PDF text extractors (pypdf, pdfplumber) use only the digital metadata. Document-anchoring combines both: feed the visual page image AND a structured representation of the digital metadata into the VLM as a single prompt, letting the model use both sources of information to disambiguate content and reading order.

**How document-anchoring works mechanically**, as described in Appendix A and illustrated in Figure 3:

1. **Extract PDF internals**: The pypdf library processes the PDF binary to extract all text blocks and images on a page, along with their position information. Each text block consists of the characters found in the PDF's internal content stream, plus their bounding box coordinates (specified as `[x, y, width, height]` in a coordinate system where the origin `[0,0]` is at the lower-left corner of the page). Images are similarly extracted with their position and dimensions.

2. **Prioritize and sample blocks**: The system sorts extracted blocks by relevance, "prioritizing text blocks and images which are located at the start and end of the document." This heuristic is based on the observation that the beginning and end of a page often contain structural elements (titles, abstracts, conclusions, references) whose correct ordering is most important, while the middle of the page is more likely to be continuous prose where reading order is easier to infer. Blocks are sampled and concatenated into a text string until a maximum character limit is reached (6,000 characters during inference, with exponential backoff if the prompt exceeds the model's token limit).

3. **Construct the anchored prompt**: The sampled blocks are formatted as a plain-text listing of position and content, inserted between `RAW_TEXT_START` and `RAW_TEXT_END` markers in the prompt template. The prompt also includes instructions to "return the plain text representation of this document as if you were reading it naturally" and explicit prohibitions against hallucination. The rasterized page image is provided alongside this text prompt as the visual input to the VLM.

4. **Model processes both modalities**: The VLM receives the page image through its vision encoder and the anchored text through its text tokenizer. The model can use the anchored text as hints—it tells the model "there is text here that says X" at specific locations—while using the visual information to verify, correct, and reorder that text into natural reading order.

**What the anchored text actually looks like** (from Figure 3 example):

```
Page dimensions: 612.0x792.0
[Image 480x298 to 525x312]
[71x459]Figure 1
[71x398]1 Introduction
[71x84]Command R (Cohere, 2024a), R+ (Cohere, 2024c), and R7B (Cohere, 2024b)
[71x351]systems (Cottier et al., 2024). Yet, these open-weights models are only the
```

Each line encodes a bounding box `[x, y]` (the lower-left corner of the text block, in page coordinate units where origin is bottom-left) followed by the extracted text. The coordinate system is crucial: it gives the model spatial information about where each text block appears on the page, which helps with reading order (text higher on the page typically comes before text lower on the page; text further left in multi-column layouts is in a different column than text further right).

**Why document-anchoring is necessary**, as established empirically and explained conceptually:

- **Without anchoring, VLMs hallucinate**: The paper explicitly states that "prompting with just the page image was prone to models completing unfinished sentences, or to invent larger texts when the image data was ambiguous" (Appendix A). This is a specific failure mode of VLMs: when visual information is ambiguous (blurry scans, small fonts, complex layouts), the model's language modeling prior takes over and it generates plausible-sounding but incorrect text. Document-anchoring provides a "ground truth" anchor—the model knows what text should be there because the anchored hints tell it, and its job becomes ordering and cleaning rather than recognizing and generating from scratch.

- **PDF internal text is unreliable but informative**: The paper acknowledges that the text extracted from PDF internals via pypdf is "highly noisy: reading order is not preserved and main content is interwoven with boilerplate text and PDF rendering-related artifacts." Despite this noise, the extracted text contains the actual words on the page—sometimes scrambled in order, sometimes with artifacts, but rarely hallucinated (it's extracted deterministically from the PDF binary). By providing this noisy but faithful text alongside the image, the model gets the best of both worlds: the visual structure tells it the correct reading order, and the anchored text tells it the exact words that appear, reducing the pressure on the visual OCR to get every character right.

- **Graceful degradation on scanned documents**: For image-only PDFs (like the Internet Archive books), pypdf extracts no text (there is no digital text stream), so the anchored prompt will contain only image placeholders and coordinates. In this case, the model falls back to pure visual processing, similar to a standard end-to-end OCR model. The paper confirms this works: "our pipeline maintains high performance on documents that do not have any digital metadata encoded in them" (Appendix A). The model learns during training that sometimes anchors are present and sometimes they are absent, and adapts accordingly.

**Key implementation details** for document-anchoring during different phases:

- **During GPT-4o data generation** (Appendix E.1): The prompt is longer and more detailed, including instructions to "turn equations into LaTeX representation," "tables into markdown format," "remove headers and footers but keep references and footnotes," and "read any natural handwriting." The anchored text is injected between `RAW_TEXT_START` and `RAW_TEXT_END` markers. GPT-4o is required to output structured JSON matching a specific schema.

- **During model training** (Section 2.3, Appendix E.2): The prompt is simplified—instructions about LaTeX, markdown, headers/footers, and handwriting are removed, leaving only the core directive: "Just return the plain text representation of this document as if you were reading it naturally. Do not hallucinate." This simplification is deliberate: the model learns these tasks from the silver data examples rather than from explicit instruction, and a shorter prompt reduces token usage. The page image is rendered to a maximum dimension of 1024 pixels on the longest edge (smaller than the GPT-4o images) to reduce computational cost.

- **During inference** (Appendix D.2): The same simplified prompt is used. The character limit for anchored text is 6,000 characters. If the total prompt (anchored text + image tokens) exceeds 8,192 tokens (the model's maximum context during training), the system regenerates the anchored text with "exponentially lower character limits" until the prompt fits. This exponential backoff ensures that even extremely dense pages can be processed, albeit with fewer anchored hints.

**The character limit and token accounting** during training is described in Section 2.3: "a typical prompt uses about 1,000 tokens to encode a page image, 1,800 tokens for the anchor text, for about 3,000 total input tokens." The image encoding is efficient because Qwen2-VL uses a vision transformer that produces a fixed number of visual tokens per image (not one token per pixel). Each training example is truncated to 8,192 tokens to handle cases where the anchored text is unusually long. Loss is computed only on the output tokens (the JSON response), not on the prompt tokens, which is standard practice for instruction fine-tuning: the model should learn to generate the correct response given the prompt, not to reproduce the prompt itself.

---

#### Silver Data Generation with GPT-4o

The training data for olmOCR-7B-0225-preview is not human-annotated but generated by GPT-4o, making this a **knowledge distillation** setup: a large, proprietary teacher model produces supervision targets for a smaller, open-source student model. The paper describes the rationale and process in Section 2.2.

**Why GPT-4o was chosen as the teacher** (from Section 2.2, especially the footnote):

> "In October 2024, we evaluated several leading VLMs for data generation. Gemini 1.5 was eliminated due to frequent RECITATION errors (though this was resolved by February 2025), GPT-4o mini produced excessive hallucinations, and Claude Sonnet 3.5 was cost-prohibitive. We selected gpt-4o-2024-08-06 as it offered the optimal balance of accuracy, reliability, and cost-efficiency in batch mode."

This is an important design narrative: the choice of teacher model was made through systematic comparison of available VLMs in late 2024, with specific failure modes identified for each alternative. Gemini 1.5's RECITATION errors are a particularly interesting failure mode—the model would refuse to process content it identified as potentially copyrighted, which is catastrophic for batch processing of arbitrary web PDFs where copyright status is unknown. Claude Sonnet 3.5 was eliminated purely on cost grounds, not quality. GPT-4o mini's hallucinations are consistent with the observation that smaller VLMs struggle with this task without fine-tuning.

**The prompt design for GPT-4o** (reproduced in full in Appendix E.1) is crafted to address specific failure modes observed during development:

- **Instructions against hallucination**: The prompt explicitly states "Do not hallucinate" and "If there is no text at all that you think you should read, you can output null." This addresses the tendency of VLMs to generate plausible content for blank or near-blank pages (common in PDFs with blank separator pages, mostly-empty forms, or pages that are purely decorative).

- **Handling of page boundaries**: The instruction "this is likely one page out of several in the document, so be sure to preserve any sentences that come from the previous page, or continue onto the next page, exactly as they are" prevents the model from "completing" truncated sentences at page boundaries. This is crucial for linearization: when a paragraph spans pages, the extraction should preserve the partial sentences at page edges, not hallucinate completions.

- **Structured element formatting**: The prompt specifies "Turn equations into a LaTeX representation, and tables into markdown format." This standardizes the output format for structured content, making it parseable downstream and consistent across pages.

- **Header/footer removal with exceptions**: "Remove the headers and footers, but keep references and footnotes." This is a nuanced instruction: headers and footers (repeating page numbers, section titles, publication names) are noise that should be stripped, but footnotes and references (which may appear at page bottom but are semantically part of the content) should be preserved. The model must learn to distinguish these based on visual cues and content.

- **Handwriting instruction**: "Read any natural handwriting." This explicitly tells the model to process handwritten annotations, which appear frequently in scanned documents and annotated PDFs.

**The structured JSON output schema** (reproduced in full in Appendix E.1) enforces a consistent response format with specific fields:

- `primary_language`: a two-letter language code or null, identifying the document's language. This field itself doesn't affect extraction but provides metadata that could be used for downstream filtering.

- `is_rotation_valid`: a boolean indicating whether the page is correctly oriented for reading. The prompt specifies: "Answer only considering the textual content, do not factor in the rotation of any charts, tables, drawings, or figures." This is important because many PDFs contain landscape-oriented tables or figures within portrait-oriented pages, and the model should not rotate the entire page based on a single rotated element.

- `rotation_correction`: an integer from {0, 90, 180, 270} specifying the clockwise rotation needed if the page is not correctly oriented. Defaults to 0. This field enables automatic rotation correction during inference (Appendix D.2): if the model detects a rotation issue, the pipeline can rotate the page image and reprocess.

- `is_table`: boolean indicating if the majority of page content is in tabular format. This is used downstream for format decisions, not for extraction itself.

- `is_diagram`: boolean indicating if the majority of page content is a visual diagram. When true, the model is expected to output null for `natural_text` since diagrams don't contain extractable text.

- `natural_text`: the actual linearized plain text, or null if no text should be extracted.

**Why structured output is crucial** (Section 2.2): The paper notes that enforcing structured output was "crucial to ensure that GPT-4o does not generate captions of images when no text is present on the page." Without the schema constraint, GPT-4o would sometimes describe images ("A diagram showing...") even when instructed to extract text only. The structured schema forces the model to explicitly categorize the page first (is it a diagram? is it a table? what language?) before deciding what text to output. This two-step reasoning process—categorize then extract—improves output quality by preventing category confusion.

**Document-anchoring's role in data generation** (Section 2.2): The prompted GPT-4o includes the anchored text extracted from pypdf. The paper explicitly compares GPT-4o with and without anchoring on olmOCR-Bench in Table 4: GPT-4o (No Anchor) scores 68.9% overall, while GPT-4o (Anchored) scores 69.9%—a small but consistent improvement of about 1 percentage point. This validates that anchoring helps even for the strongest teacher model, though the benefit is modest because GPT-4o is already very capable at visual extraction. The larger benefit comes when the student model is fine-tuned on anchored data: the 7B model learns to make effective use of the anchors, closing the gap to GPT-4o.

**Cost of data generation**: The paper processed 258,641 pages through GPT-4o. At the reported pricing of $2.50 per million input tokens and $10.00 per million output tokens (February 2025), with batch mode halving both prices, the total cost can be estimated but is not explicitly stated. For each page, approximately 1,000 image tokens + 1,800 anchor text tokens ≈ 3,000 input tokens and perhaps 500-1,500 output tokens. At batch pricing ($1.25/M input, $5.00/M output), this works out to roughly $0.00375 input + $0.005 output = $0.00875 per page, or approximately $2,260 for the full dataset. This is the one-time cost of creating the silver training data, distinct from the per-page inference cost of the fine-tuned model.

---

#### Model Training: Fine-Tuning Recipe and Design Decisions

The paper fine-tunes Qwen2-VL-7B-Instruct on olmOCR-mix-0225 to produce olmOCR-7B-0225-preview. Section 2.3 describes the training setup, with additional details in Appendix C.

**Base model selection**: The choice of Qwen2-VL-7B-Instruct as the starting checkpoint is motivated but not extensively justified. Qwen2-VL (Wang et al., 2024b) is a vision-language model that processes images at native resolution through a dynamic resolution mechanism, which is important for PDF pages where text can be very small relative to the image dimensions. The "Instruct" variant has already been fine-tuned for instruction following, making it a better starting point for the task than a base VLM. The 7B parameter scale is chosen to balance capability with inference cost: a 7B model can run on a single consumer-grade GPU (L40S) at high throughput, enabling the claimed $176 per million pages cost.

**Training hyperparameters** (Section 2.3):

- **Effective batch size**: 4. This is small, which is typical for VLM fine-tuning where each example includes a high-resolution image. The paper doesn't specify whether this is achieved through gradient accumulation or a literal batch size of 4.
- **Learning rate**: 1e-6. This is very low, appropriate for fine-tuning a pretrained model where you want to preserve existing capabilities while adapting to the new task distribution. Larger learning rates risk catastrophic forgetting of the base model's visual understanding.
- **Optimizer**: AdamW, the standard choice for transformer fine-tuning, combining Adam's adaptive learning rates with decoupled weight decay regularization.
- **Learning rate schedule**: Cosine annealing over 10,000 steps. Cosine annealing smoothly reduces the learning rate from its initial value to near zero following a cosine curve, which is a standard choice for fine-tuning runs where you train to approximate convergence.
- **Training duration**: 10,000 steps, which the paper notes is "roughly 1.2 epochs." With 258,641 training examples and effective batch size 4, one epoch would be approximately 64,660 steps, so 10,000 steps is about 15.5% of an epoch. This is a partial epoch fine-tune, which is common when the training data is large relative to the number of parameters and full convergence would lead to overfitting or excessive forgetting.
- **Hardware**: Single node with 8 × NVIDIA H100 (80GB) GPUs. One training run took 16 node hours (2 hours on 8 GPUs, or equivalent). Total training experiments across all development iterations consumed 365 node hours, indicating approximately 23 full training runs were performed during development for hyperparameter tuning, data ablation, and prompt iteration.
- **Sequence length**: Each training example truncated to 8,192 tokens. This covers the maximum prompt length (approximately 3,000 tokens for a typical page) plus output (typically a few hundred to a few thousand tokens of JSON). Truncation handles unusually long pages where the anchored text is very large.

**Loss masking**: "Loss was masked so only the final response tokens participated in the loss calculation" (Section 2.3). This is standard for instruction fine-tuning: the model sees the full prompt + response sequence during the forward pass, but the training loss is computed only on the tokens that constitute the response (the JSON output). This ensures the model learns to generate the correct extraction given the prompt, without being penalized for the prompt tokens (which are fixed input, not generated). Without masking, the model would also be trained to predict the prompt tokens, which is unnecessary and could interfere with learning the extraction task.

**Prompt modifications during training** (Section 2.3): The prompt used during fine-tuning is simplified compared to the GPT-4o data generation prompt (Appendix E.2 vs. E.1). The training prompt removes:

- The detailed instructions about LaTeX, markdown, headers/footers, handwriting, and cross-page sentence preservation.
- The instructions about outputting null when no text is present.

The training prompt retains only the core directive: "Just return the plain text representation of this document as if you were reading it naturally. Do not hallucinate," plus the anchored text between `RAW_TEXT_START` and `RAW_TEXT_END` markers. This simplification means the model must learn the detailed formatting behaviors (how to represent equations in LaTeX, how to format tables in markdown, when to strip headers) from the examples in the training data, not from explicit instruction. This is effective because the silver data from GPT-4o already exhibits these behaviors consistently; the model learns to imitate them.

**Image resizing during training**: PDF pages are rendered to "a maximum dimension of 1024 pixels on the longest edge." This is smaller than the resolution used during GPT-4o data generation (where images are sent at higher resolution for the teacher model). The resizing reduces computational cost during training (vision transformer cost scales with image dimensions) while still providing sufficient detail for the 7B model to learn the extraction task. At inference time, the same 1024-pixel rendering is used, maintaining consistency between training and deployment.

**Validation and model selection** (Appendix C, Figures 4 and 5): The paper tracks validation loss separately for web PDFs and Internet Archive books during training. Both subsets show steadily decreasing loss over 1,200 steps (approximately 0.14 epochs), with full fine-tuning achieving lower loss than LoRA (Low-Rank Adaptation). The paper uses full fine-tuning for the final model based on this validation loss comparison, though it does not report whether the validation loss difference translates to downstream benchmark improvements.

**Development methodology** (Appendix C, Figure 6): The paper notes that hyperparameters and data curation decisions were tuned "alongside other data curation decisions" using "manual side-by-side evaluation." A custom evaluation tool was developed that displays a PDF page alongside the outputs of two different extraction methods (e.g., two different model checkpoints, or different prompt configurations), allowing human evaluators to compare which output better represents the original document. This qualitative evaluation was used during development to make decisions before the olmOCR-Bench benchmark was finalized, since iterating on the benchmark and the model simultaneously would risk overfitting. The evaluation tool itself is released as open-source software.

**Full fine-tuning vs. LoRA**: The validation loss curves (Appendix C, Figures 4 and 5) show that LoRA (Low-Rank Adaptation, a parameter-efficient fine-tuning method that trains only small adapter matrices rather than all model weights) results in "higher loss values compared to full fine-tuning." Full fine-tuning updates all 7 billion parameters, which requires more GPU memory and compute but allows the model to adapt more thoroughly to the new task distribution. The paper's choice of full fine-tuning over LoRA represents a tradeoff: higher training cost for better final performance. Since the training cost (16 node hours per run) is modest compared to the data generation cost and the inference cost savings, this tradeoff is well-justified for a model intended for production deployment.

**Alignment with teacher model** (Appendix C.1): The paper evaluates how closely the fine-tuned model's outputs match GPT-4o's silver data using a word-level alignment metric based on Hirschberg's algorithm (a dynamic programming algorithm for sequence alignment, commonly used for computing edit distance between strings). The alignment score is the fraction of words that match between the model's output and GPT-4o's output after optimal alignment. Results in Table 7 show:

- GPT-4o self-alignment (calling GPT-4o twice on the same input): 0.954. The 4.6% mismatch is due to "the probabilistic nature of autoregressive decoding"—even the same model produces slightly different outputs on different runs.
- GPT-4o mini alignment to GPT-4o: 0.833. The smaller model deviates more from the teacher.
- olmOCR-7B-0225-preview at temperature 0.1: 0.875. The fine-tuned 7B model matches the teacher better than GPT-4o mini does, despite being from a different model family.
- olmOCR-7B-0225-preview at temperature 0.8: 0.859. Higher temperature slightly reduces alignment, as expected.

This measurement serves two purposes: it validates that the fine-tuned model has successfully learned to imitate GPT-4o's extraction behavior, and it calibrates expectations—since even GPT-4o does not perfectly replicate itself, a student model achieving 87.5% word-level alignment is strong performance.

The paper further buckets alignment scores into low (<70%), medium (70-95%), and high (>95%) match categories in Table 8. At temperature 0.1, the fine-tuned model has 158 low-match pages, 363 medium-match pages, and 700 high-match pages out of 1,221 evaluated. At temperature 0.8, the distribution shifts toward more low matches (195) and fewer high matches (636), consistent with higher temperature introducing more variation. The majority of pages are in the high-match category for both temperatures, indicating that the model generally produces extractions very close to GPT-4o's.

---

#### Inference Pipeline: Architecture for Scalable Batch Processing

The inference pipeline described in Appendix D is what makes olmOCR practical for processing millions of pages. It addresses throughput, reliability, and coordination challenges that are distinct from model quality but equally important for real-world use.

**Inference engine choice**: The pipeline uses SGLang (Zheng et al., 2024) as the primary inference engine, with vLLM (Kwon et al., 2023) also supported. Both are high-performance LLM serving frameworks that optimize GPU memory management and request batching. SGLang is specifically designed for "structured language model programs" and supports efficient constrained decoding, which is relevant given that the model outputs structured JSON. The paper notes that both backends are supported, giving users flexibility based on their infrastructure.

**Work item batching**: The pipeline "batches documents into work items of around 500 pages each." This batch size is chosen to balance two competing concerns:

- **GPU utilization**: A larger batch keeps the GPU continuously fed with inference requests, maximizing tokens-per-second throughput. Modern LLM serving systems batch requests dynamically, but having a queue of 500 pages ensures the GPU is rarely idle.
- **Work item completion time**: Each work item must complete before the worker can move to the next one. If work items are too large, a single slow page (e.g., a very dense page requiring long generation) delays the entire batch. The 500-page size keeps individual work items manageable while maintaining high utilization.

**Worker coordination**: "Optionally, workers can coordinate using a shared cloud bucket, allowing for batch jobs that scale from single nodes to hundreds of nodes without the need for complicated queue management." The paper specifically mentions Amazon S3, but notes other cloud storage solutions work. This is a simple but effective coordination strategy: each worker pulls a work item (a set of 500 PDF pages) from a shared queue implemented via cloud storage, processes it, and uploads results. There is no central scheduler or message queue infrastructure—just a shared filesystem abstraction. This design makes the system easy to deploy on arbitrary cloud or on-premise GPU clusters.

**Within-worker processing**: "Each worker queues up inference for all PDF pages in a work item simultaneously, and then waits until the SGLang server has no more pending requests before proceeding to another work item in the queue." This is a synchronous batch processing pattern: the worker submits all 500 pages to SGLang at once, SGLang handles dynamic batching internally (grouping requests with similar sequence lengths to maximize throughput), and the worker waits for all requests to complete before moving on. This is simpler than streaming processing but ensures that all pages in a work item are completed before the worker becomes available for the next item, which is important for the cloud storage coordination model (the worker writes all results for a work item atomically).

**Retry logic and failure handling** (Appendix D.2): The pipeline implements several mechanisms to handle failures gracefully without sacrificing overall throughput:

1. **JSON parsing retries**: The model was fine-tuned on structured JSON output and "reliably adheres to the required schema without constraints" (the paper chose not to use forced constrained decoding because it found open-source tools for this were "unreliable" and could "cause generations to go out-of-domain or collapse into repetitions"). When JSON parsing fails (rarely), the system "simply retries generating from the same input sequence" with a different random seed. The paper does not specify the exact retry count, but mentions "up to N times."

2. **Rotation correction**: The model outputs `is_rotation_valid` and `rotation_correction` fields in its JSON response. During inference, if `is_rotation_valid` is false, the pipeline rotates the page image by the specified amount (0, 90, 180, or 270 degrees clockwise) and reprocesses the page. This handles the common case of scanned documents saved in landscape orientation when they should be portrait (or vice versa). The rotation is applied to the image data, not the anchored text, since the anchored coordinates are relative to the original unrotated page.

3. **Repetition detection and mitigation**: The paper identifies "outputs degenerating into endless repetitions of the same token, line, or paragraph" as "the most common failure" during development. This is a well-known failure mode of autoregressive language models, where the model gets stuck in a repetitive loop. The paper addresses this in two ways:
   - **Temperature increase**: "We find that increasing generation temperature from τ = 0.1 up to τ = 0.8 reduces the likelihood of repetitions occurring." At temperature 0.1 (near-deterministic), the model always selects the highest-probability token, which can lead to repetitive loops if the highest-probability continuation of "abc abc abc" is another "abc". Higher temperature introduces randomness that can break these loops. The tradeoff is that higher temperature may reduce output quality (more variation from the teacher's distribution, as shown in Appendix C.1, Table 7).
   - **Exponential backoff on anchored text length**: When the prompt (anchored text + image) exceeds the model's context limit, the system regenerates the anchored text with "exponentially lower character limits" until the prompt fits. This handles edge cases where a single page has an extremely large amount of extracted text (e.g., a page with hundreds of small text blocks).

4. **Fallback extraction**: If a page "repeatedly fails" (all retries exhausted), the pipeline falls back to "a plain text-based PDF extraction" using the raw pypdf text. This ensures that even pages the model cannot process produce some output, which is better than losing the content entirely. The paper notes that this fallback is "aided by the fact that document-anchoring randomly samples which anchors to include in the prompt; thus, resampling can sometimes help the page process correctly by removing potentially problematic meta tokens." The randomness in anchor sampling means that retries use slightly different prompts, which can help if a specific anchor text block was causing issues.

5. **Retry rate measurement**: The paper reports "a 12% retry rate for olmOCR" (Table 6 caption). This means approximately 12% of pages require at least one retry (due to JSON parsing failure, repetition, or other errors). This is a substantial rate—it means effective throughput is reduced by roughly 12% compared to ideal throughput (since retried pages consume additional GPU time). The paper acknowledges this but does not break down retry reasons.

**Throughput measurements** (Section 4.3, Table 6): The paper reports detailed throughput and cost measurements:

- **On L40S GPU**: 906 output tokens per second, processing 1,288 test pages in 17 minutes 10 seconds. At $0.79 per hour for an L40S, this yields 5,697 pages per dollar, or **$176 per million pages** (including the 12% retry overhead).
- **On H100 GPU**: 3,050 output tokens per second, processing the same 1,288 pages in 5 minutes 7 seconds. At $2.69 per hour for an H100, this yields 5,632 pages per dollar, or **$178 per million pages**.

The slightly higher cost on H100 is counterintuitive since H100 is faster, but the per-hour cost is proportionally higher (3.4× the hourly rate for 3.37× the throughput), making the per-page costs nearly identical. The paper's key cost claim of $176 per million pages refers to the L40S configuration, which is the more cost-effective option for this workload. The H100 might be preferred when latency matters (a single job completes faster) rather than cost.

**Prompt construction during inference** (Appendix D.2, E.2): The inference prompt is the same simplified version used during training. The anchored text is generated fresh for each page using document-anchoring, with a 6,000 character limit. If the total prompt exceeds 8,192 tokens, the system reduces the character limit exponentially until it fits. The paper does not specify the exact exponential factor (e.g., halving each time, or multiplying by 0.8), but the principle is clear: the system guarantees the prompt will fit within the model's context window, even if it means providing very few anchored hints.

**Structured output handling**: Unlike during GPT-4o data generation (where a JSON schema was enforced via the API's structured output feature), the fine-tuned model is not constrained to output valid JSON during inference. The paper gives two reasons for this: first, "open source tools designed to force decode a sequence into a particular schema are unreliable," and second, "enforcing a schema which is even slightly off from what the model expects can cause generations to go out-of-domain or collapse into repetitions." Instead, the paper relies on the model having been extensively fine-tuned on structured JSON output, so it naturally produces valid JSON without constraints. When parsing fails, the page is retried.

---

#### olmOCR-Bench: Construction and Design Philosophy

Section 3 and Appendix F describe the construction of olmOCR-Bench, a benchmark designed to evaluate PDF extraction tools. The benchmark is methodologically notable for its unit-test design and diverse document sourcing, and it serves as the primary evaluation framework for the paper's claims.

**Design philosophy** (Section 3): The benchmark is built around the concept of "pass-or-fail unit-tests"—each test case asks a simple, binary question about the extracted text: "does it contain X?" or "is X before Y?" or "does the table have a cell with value X in the correct position?" The paper explicitly justifies this design against two common alternatives:

- **LLM-as-judge evaluation**: The paper cites Panickssery et al. (2024) to note that model-based evaluators "can be biased towards favoring their own generations." This is a well-documented problem in LLM evaluation: models prefer their own outputs or outputs from similar models, making LLM-judge scores unreliable for comparing different systems.

- **Soft metric comparison against reference text**: Metrics like edit distance, BLEU, or ROUGE require a gold-standard reference text, which is expensive to create and may not capture semantically important errors. The paper gives a specific example: "incorrect math formulas (e.g., xi vs xi)"—a single-character error in a LaTeX formula can completely change its meaning, but edit distance would count it as a minor error, while the paper's unit-test approach would catch it as a failed formula test.

**Test categories** (Section 3.1, Table 3): The benchmark comprises 7,010 test cases across 1,402 PDFs, organized into five test categories plus a baseline:

- **Text Presence** (721 tests): Verifies that a specific text segment (1-3 sentences) appears in the extracted output. Soft/fuzzy matching is allowed (handling minor OCR errors), and tests can specify that the text must appear in the first N or last N characters (for testing header/footer handling). Case-sensitive by default.

- **Text Absence** (823 tests): Verifies that a specific text segment is NOT in the output. This primarily tests header/footer removal—page numbers, running headers, publication names that repeat on every page should be stripped. Soft/fuzzy matching is allowed, and location constraints (first N or last N characters) can be specified. Not case-sensitive by default (since headers/footers may vary in capitalization).

- **Natural Reading Order** (1,061 tests): Verifies that two text segments appear in the correct relative order. The paper gives the example: "on a PDF with multiple news articles on one page, we can test for whether the first sentence of the first article appears after the heading of that article; yet such tests can be designed to not penalize for the order of the articles themselves." This is a sophisticated design: the tests check local reading order (within a single article) without requiring a global ordering across independent page elements. Soft matching is allowed, case-sensitive.

- **Table Accuracy** (1,020 tests): Checks that a table cell contains a specific value and that its neighboring cells have specific properties. For example: "validate this page has a table with a cell containing '4.5%' and above that is a cell containing '2.4%'." Both Markdown and HTML-based table representations are supported, though the paper notes that "many cases depend on rowspan and colspan information being preserved, which is possible only in HTML based tables." This is an important practical consideration: Markdown tables cannot represent merged cells, so tools that output only Markdown tables are at a disadvantage for tables with complex structure.

- **Math Formula Accuracy** (3,385 tests — the largest category): Checks that a specific LaTeX equation appears correctly in the output. The verification method is visual rather than string-based: "We render a reference LaTeX equation using KaTeX in a headless browser and extract all rendered symbols and their (visual) bounding boxes. Then we check if a matching collection of symbols, with the same relative orientations, exists anywhere in the final OCR document." This is a clever design that avoids the brittleness of string matching on LaTeX (where `x_i`, `x_{i}`, and `x\_{i}` are semantically identical but syntactically different). Instead, it checks whether the *rendered* symbols match in position and identity, which is invariant to LaTeX formatting differences. The paper notes this is "similar to the method described by Wang et al. (2025), but ours is simpler due to the test being Pass/Fail only." The pass/fail simplification means the system only needs to verify existence, not compute a similarity score.

- **Baseline tests** (applied to every document): Each PDF receives a default test that checks: (1) some plain text containing alphanumeric characters was produced (the system didn't output empty or null for a page that should have content), (2) the output doesn't end with a string of repeating N-grams longer than 30 characters (catching repetition degeneration), and (3) the output doesn't contain characters from Chinese, Japanese, or Emoji Unicode charsets (catching language-switching failures). Documents that legitimately contain such characters are manually flagged and excluded from these test conditions.

**Document sourcing strategy** (Section 3.2, Table 10, Appendix F): The paper defines seven distinct document categories, each targeting a specific extraction challenge and sourced differently:

1. **arXiv Math (AM)** — 522 PDFs, 2,927 tests: Recent papers from the math subset of arXiv, selected because they have a single TeX source file and corresponding rendered PDF. This enables a sophisticated test creation pipeline:
   - Run olmOCR on the PDF to identify candidate pages with LaTeX formulas.
   - Match identified pages back to the original TeX source using dynamic programming alignment.
   - Validate that the matched TeX renders correctly in KaTeX (since arXiv papers may use custom macros that deviate from standard LaTeX).
   - Manually verify final test cases to exclude instances where custom macros produce non-standard renderings, and to split multi-part equations into smaller test cases.

   This pipeline leverages the unique property of arXiv that both the source code and the rendered PDF are available, enabling automated gold-standard extraction of formulas and their ground-truth LaTeX.

2. **Old Scans Math (OSM)** — 36 PDFs, 458 tests: Old, public domain math textbooks from the Internet Archive. Since these are scanned images (no TeX source available), formulas must be manually annotated. The paper uses olmOCR to find candidate pages with formulas, then manually annotates each formula. This is the smallest category (only 36 documents) because manual annotation is expensive.

3. **Tables (TA)** — 188 PDFs, 1,020 tests: Documents sampled from the same internal crawled PDF repository as the training data, filtered to those containing tables using a simple prompt with Gemini Flash 2.0. For pages with tables, Gemini Flash 2.0 is prompted to identify relationships between randomly chosen cells (e.g., "cell containing X is to the left of cell containing Y"). These relationships are then manually reviewed for accuracy.

4. **Old Scans (OS)** — 98 PDFs, 526 tests: Historical letters and typewritten documents from the Library of Congress digital archives. The key advantage of this source is that these documents have existing human transcriptions. The paper writes "a small script to generate Natural Reading Order cases consisting of sentences that were naturally before or after one another in the original human transcriptions." This is an efficient use of existing human-created ground truth. Additional test cases were manually added to cover headers/footers that should be excluded. All test cases underwent a second pass of human review.

5. **Headers Footers (HF)** — 266 PDFs, 753 tests: Documents from the internal crawled PDF repository. The paper uses DocLayout-YOLO (Zhao et al., 2024), a document layout analysis model, to identify page regions labeled as headers or footers. To extract the text from these regions, the rest of the document is visually masked out and Gemini Flash 2.0 is prompted for the content. These extracted snippets become test cases that should be *absent* from linearized output. The paper manually reviewed these to "remove mistakenly filtered text and to set conditions such as limiting the search area to the first N or last N characters." For example, if a page number "5" appears at the bottom of a page, the test checks that "5" does not appear in the last 20 characters of the output, while still allowing "5" to appear earlier in the text (as part of equation numbers, section numbers, etc.).

6. **Multi Column (MC)** — 231 PDFs, 884 tests: Documents from the internal crawled repository with multi-column layouts and multiple articles on one page. The paper uses Claude Sonnet 3.7 to render these pages to HTML, then extracts text segments that are before/after one another from the HTML structure. These become natural reading order test cases. The paper purposely selects "simple text blocks from coherent regions of the document, and avoids including any math formulas, superscripts, or subscripts in these tests," which simplifies the verification since string matching on plain text is reliable.

7. **Long Tiny Text (LTT)** — 62 PDFs, 442 tests: Documents from the Internet Archive containing "a large amount of dense, small print on a single page," such as dictionary pages or reference lists from academic papers. Test cases are generated using Gemini Flash 2.0 and manually verified.

**Test case creation scaling**: The paper combines automated generation (using VLMs to propose test cases) with manual review to scale test creation while maintaining quality. This is a practical approach: fully manual creation of 7,010 test cases would be prohibitively expensive, while fully automated creation would risk including incorrect or ambiguous tests. The manual review step acts as a quality filter on the automated proposals.

**Scoring methodology** (Section 3.3): The overall score for a tool is computed as:

$$\text{Overall score} = \frac{1}{N} \sum_{s \in \text{Document sources}} \text{Score}(s)$$

where $N$ is the number of document sources and $\text{Score}(s)$ is the percentage of tests passed within that source.

**What this computes:** The macro-average of pass rates across the seven document source categories (plus baseline tests). Each source contributes equally to the final score regardless of how many test cases it contains.

**Why this form:** The paper explains that macro-averaging "captures the difficulty we faced at times of finding and validating enough cases from each source, but we roughly feel that each source represents an important capability for an OCR system to have." In other words, the paper does not want the overall score to be dominated by the arXiv Math category (which has 2,927 tests, far more than any other category) because that would make the benchmark primarily a math formula extraction test. Instead, each document type is weighted equally, reflecting the belief that a good OCR system should perform well across ALL document types, not just the most common one. The 95% confidence intervals in Table 4 are calculated by bootstrapping with 10,000 resamples, giving a measure of statistical reliability given the finite test set sizes.

**Text normalization**: All text comparisons (presence, absence, reading order) apply "basic string normalization" before matching: converting `<br>` tags to newlines, normalizing all whitespace to single ASCII spaces, removing Markdown bold/italics formatting, normalizing quotes and hyphens to ASCII, and converting all Unicode to NFC (Normalization Form C, which composes characters into their canonical composed form). This normalization ensures that formatting differences (e.g., using `**bold**` vs. actual bold characters) don't cause test failures when the semantic content is correctly extracted.

---

#### Summary of Design Choices and Their Justifications

- **Document-anchoring over pure visual processing**: The PDF-internal text, while noisy, provides redundant signal that helps VLMs avoid hallucination. The approach degrades gracefully to pure visual processing when no digital metadata exists, making it robust across born-digital and scanned documents.

- **GPT-4o as teacher model**: Selected after systematic comparison of available VLMs in late 2024, balancing accuracy, reliability, and cost. The one-time training data generation cost (~$2,260) is amortized across millions of inference pages, making the per-page cost negligible.

- **Structured JSON output for data generation**: Prevents GPT-4o from generating image captions, hallucinating content for blank pages, and misclassifying page types. Forces the model to categorize before extracting.

- **Full fine-tuning over LoRA**: Achieves lower validation loss and better extraction quality, at the cost of higher training compute. Justified because the model is intended for production deployment where inference quality matters more than training efficiency.

- **7B parameter scale**: Balances extraction capability with inference cost, enabling the $176 per million pages price point. A larger model would be more expensive per token without guaranteed quality improvement.

- **Temperature increase (0.1 → 0.8) to mitigate repetitions**: Higher temperature introduces randomness that breaks repetitive loops, at a small cost in output consistency (alignment drops from 0.875 to 0.859). The reliability gain justifies the consistency loss.

- **Exponential backoff on anchored text length**: Ensures prompt fits within model context window regardless of page complexity, at the cost of providing fewer anchored hints for extremely dense pages. More hints are better, but some hints are better than a failed generation.

- **Unit-test evaluation over soft metrics or LLM-judge**: Enables deterministic, unbiased comparison across tools with different output formats and tokenizations. Each test is simple enough to be machine-verifiable without ambiguity.

- **Macro-averaging across document sources**: Prevents the benchmark from being dominated by the largest test category (arXiv Math) and ensures systems must perform well across diverse document types to achieve a high overall score.

- **Cloud storage coordination for distributed inference**: Simple, infrastructure-agnostic approach to scaling from one to hundreds of GPUs without requiring complex message queue or scheduler systems. The tradeoff is that workers operate in batch mode rather than streaming, which may increase latency for individual jobs but maximizes throughput for bulk processing.

## 4. Key Insights and Innovations

### Innovation 1: Document-Anchoring as a Modality-Bridging Strategy, Not a Prompt Engineering Trick

The paper's most intellectually distinctive contribution is the concept of **document-anchoring**: combining rasterized page images with noisy text extracted from PDF internals—including spatial coordinates—as complementary input modalities to a VLM. On its surface, this looks like a prompting strategy. At a deeper level, it is a novel solution to a previously unarticulated problem: **PDFs are neither purely visual nor purely structural documents**, and treating them as one or the other forces a tradeoff between incompleteness and hallucination.

Before this work, the field had implicitly partitioned the PDF extraction problem along a clean boundary. End-to-end vision models like Nougat (Blecher et al., 2023) and GOT Theory 2.0 (Wei et al., 2024) treated PDFs as images-only, processing pixel rasters and autoregressively decoding text tokens. Pipeline-based tools like MinerU and Grobid treated PDFs as structured data, parsing internal content streams and applying layout heuristics. The two approaches had complementary failure modes: vision-only models hallucinated when visual information was ambiguous (completing unfinished sentences, inventing text, producing repetitive sequences), while structure-only tools produced garbled text when internal metadata was corrupted, scrambled, or absent.

What makes document-anchoring a conceptual innovation rather than an engineering convenience is that it reframes the problem from "which single representation should we use?" to "how can we use redundant, mutually-correcting representations to make the model's job easier?" The anchored text extracted from pypdf is explicitly described as "highly noisy: reading order is not preserved and main content is interwoven with boilerplate text and PDF rendering-related artifacts" (Section 2.2). The paper does not clean this text up; it provides it raw, alongside spatial coordinates, as hints. The model's job becomes not raw OCR from pixels, but **alignment and ordering**—using the visual structure of the page to determine where each anchored text block belongs in the reading sequence, while using the anchored text to avoid having to recognize every character from scratch.

This is fundamentally different from prior "multi-modal" approaches that simply concatenated text and image features into a joint embedding space. Document-anchoring is **asymmetric**: the visual modality provides structure and spatial disambiguation, while the text modality provides content and reduces hallucination pressure. The paper's finding that GPT-4o benefits only modestly from anchoring (a ~1 percentage point improvement, Table 4) while the 7B fine-tuned model achieves a much larger relative gain (enabling it to match or exceed GPT-4o) reveals something important: anchoring is most valuable when the model's native visual processing is imperfect. For a model that already has near-perfect OCR capability, anchoring is a small nudge; for a smaller model, it is a crutch that enables competent performance.

The paper also demonstrates that document-anchoring degrades gracefully when no digital metadata is available—the model learns during training that sometimes anchors are present and sometimes absent, and falls back to pure visual processing for scanned documents. This robustness is not an accident of training; it is a direct consequence of including Internet Archive book scans (which have no internal text) alongside born-digital PDFs in olmOCR-mix-0225. The training data construction ensures the model sees both regimes, so it never becomes dependent on anchors.

**Significance beyond performance**: Document-anchoring is a **framing innovation** that changes how researchers should think about VLM-based document processing. Rather than treating VLMs as end-to-end black boxes that must either succeed or fail on visual input alone, it treats them as reasoning engines that can integrate multiple noisy, partial sources of information about a page. This opens the door to document-anchoring variants that use other metadata sources (font information, paragraph boundaries from commercial PDF libraries, figure captions from dedicated extraction pipelines) or that dynamically adjust the anchoring strategy based on page characteristics. It is a **fundamental reframing** of the PDF extraction problem, not an incremental improvement to an existing approach.

---

### Innovation 2: The Teacher-Student Pipeline as a Cost-Feasibility Bridge for Document-Scale Processing

The paper's second innovation is methodological rather than architectural: the demonstration that a **knowledge distillation pipeline**—using a large, expensive, proprietary VLM as a teacher to generate silver training data for a much smaller, open-source student model—can make high-quality PDF extraction economically feasible at scale, while surprisingly producing a student that *exceeds* its teacher's quality on the target task.

This is not the first use of teacher-student distillation, nor even the first use of GPT-4o as a data generation source. What makes this particular instance distinctive is the **inversion of the expected quality hierarchy**. Typically, student models trained on teacher-generated data perform worse than the teacher—they learn a degraded approximation of the teacher's behavior. Here, olmOCR-7B-0225-preview outperforms GPT-4o on olmOCR-Bench (75.5% overall vs. 69.9% for GPT-4o with anchoring, Table 4). The paper offers a clear diagnosis: GPT-4o, as a general-purpose assistant, has been trained to be helpful—to summarize, complete, and interpret—which makes it prone to hallucinations and unfaithful reproductions even when explicitly instructed otherwise. The fine-tuned 7B model, by contrast, has been trained *only* on faithful extraction examples and has no general-purpose conversational training interfering with the task.

This is a **counterintuitive finding with both practical and theoretical implications**. Practically, it means organizations should consider fine-tuning smaller, task-specific models on silver data from large general-purpose models even when the large model is available—the specialist may outperform the generalist. Theoretically, it suggests that general-purpose instruction tuning introduces behaviors (helpfulness, creativity, summarization) that actively conflict with tasks requiring pure faithfulness, and that fine-tuning can suppress these behaviors more effectively than prompting. The paper's observation that GPT-4o is "prone to omitting content, rewriting or completing content in a manner unfaithful to the original, or captioning images when not instructed to do so" (Section 2.2) is not a failure of GPT-4o per se—it is a success of its instruction-following training to be maximally helpful—but it reveals a fundamental tension between helpfulness and faithfulness that the fine-tuning process resolves.

**Distinction from routine distillation**: Most knowledge distillation in the VLM literature focuses on compressing model size while preserving capabilities within the same task distribution. Here, the distillation *changes the task distribution itself*—from "helpfully process this page" (GPT-4o's training objective) to "faithfully reproduce exactly what is on this page" (the fine-tuned model's training objective). The silver data from GPT-4o serves not just as labels but as a **behavioral specification**—"this is what faithful extraction looks like"—that the student model internalizes. The paper's alignment analysis (Appendix C.1, Table 7) confirms this: olmOCR-7B-0225-preview achieves 87.5% word-level alignment with GPT-4o's outputs (higher than GPT-4o-mini's 83.3% alignment), indicating it has successfully learned the teacher's extraction behavior, not just a degraded copy.

**Cost as a conceptual contribution**: The paper frames the teacher-student pipeline as not just a technical strategy but an **accessibility intervention**. The one-time cost of generating olmOCR-mix-0225 (estimated at roughly $2,260 for 260,000 pages) enables per-page inference costs of fractions of a cent, making large-scale PDF processing economically viable for academic labs and smaller organizations. This transforms document extraction from a capability that is *technically possible but economically infeasible* (processing 7.9M peS2o PDFs with GPT-4o would cost ~$98.6M at non-batch pricing) to one that is *both possible and practical* (~$1,400 for the same corpus). The innovation is in recognizing that the cost bottleneck is not the task difficulty but the model size required to perform it competently without task-specific adaptation, and that task-specific distillation breaks this dependency.

This is an **incremental advance** in the distillation literature but a **fundamental practical contribution** to the document processing community, where cost has been the primary barrier to large-scale adoption of VLM-based extraction.

---

### Innovation 3: Unit-Test-Based Evaluation as a Diagnostic Framework for Document Extraction

The paper's third innovation is olmOCR-Bench itself—not just as a collection of test cases, but as an **evaluation methodology** that changes what it means to measure PDF extraction quality. The paper explicitly positions this against two dominant evaluation paradigms in the field and argues that both are inadequate for the specific challenges of this task.

**What was broken about prior evaluation**: The paper identifies two standard approaches and their failures. **Soft-metric comparison against gold reference text** (edit distance, BLEU, ROUGE) suffers from the "xi vs xi" problem—a single-character error in a LaTeX formula can completely change its mathematical meaning while incurring a negligible edit distance penalty. This approach also assumes the existence of gold reference text, which is expensive to create and may not exist for the diverse, messy PDFs that real-world pipelines encounter. **LLM-as-judge evaluation** (asking a language model to compare two extractions and declare which is better) introduces model biases—the paper cites Panickssery et al. (2024) on the tendency of LMs to prefer their own generations. Moreover, LLM judges inherit the same hallucination and faithfulness problems that the extraction systems being evaluated exhibit.

olmOCR-Bench's unit-test design reframes evaluation as a collection of independently verifiable, binary propositions about the extracted text. Each test asks a specific question—"does the extracted text contain equation X?" or "is text segment A before text segment B?" or "does this table have a cell containing '4.5%' directly above a cell containing '2.4%'?"—that can be evaluated deterministically by a simple script. There is no model in the loop, no fuzzy matching ambiguity beyond specified normalization rules, and no dependence on the format of the extracted text (Markdown, HTML, plain text all work for most tests).

**What makes this a conceptual innovation**: The unit-test framework shifts evaluation from **overall quality scoring** (how good is this extraction on average?) to **capability diagnosis** (what specific types of content can this system handle, and where does it fail?). This is analogous to the shift in software engineering from integration tests ("does the system work?") to unit tests ("does each component work correctly in isolation?"). The paper's breakdown of results by document category in Table 4 reveals patterns that a single aggregate metric would obscure: GOT OCR achieves 94.0% on baseline tests (it doesn't hallucinate language or generate empty output) but only 0.2% on table accuracy (it essentially cannot handle tables at all). MinerU achieves 96.6% on header/footer removal but only 17.3% on old scans. These diagnostic insights are actionable—a practitioner choosing a tool for a specific document collection can look at category-level performance rather than trusting a single number.

The formula accuracy test design is particularly innovative. Rather than comparing LaTeX strings (which would penalize semantically equivalent differences like `x_i` vs. `x_{i}`), the test renders the LaTeX using KaTeX, extracts the visual positions of rendered symbols, and checks whether those symbols appear in the correct spatial relationships anywhere in the extracted output. This **rendering-based verification** is invariant to LaTeX formatting choices while being sensitive to genuine semantic errors. It is a concrete instance of a broader principle: evaluate content extraction at the level of *communicated meaning* (what a human reader would see) rather than at the level of *representation format* (the specific characters used to encode that meaning).

**Why this matters beyond this paper**: The unit-test framework is not tied to olmOCR or to PDF extraction. It is a **transferable evaluation methodology** for any task where: (1) the desired output has checkable properties that can be tested independently, (2) multiple valid representations exist for the same content, and (3) overall quality scores obscure important capability differences. The paper's explicit documentation of how each test category was created—the prompts used, the manual review process, the failure cases that were excluded—makes the methodology reproducible for other document types and domains.

This is a **methodological innovation** whose significance lies not in performance numbers but in changing how researchers and practitioners should think about evaluating content extraction systems. It is a fundamental contribution to evaluation practice in this domain.

---

### Innovation 4: The Downstream Validation Loop—Closing the Gap Between Extraction Quality and LM Training Utility

The paper's fourth innovation is the demonstration that improved PDF extraction quality **measurably improves language model training outcomes**, a claim that is often asserted in the data curation literature but rarely tested with a controlled experiment. The continued pretraining experiment in Section 4.2 shows that replacing Grobid-extracted tokens in peS2o with olmOCR-extracted tokens yields a +1.3 percentage point average improvement across widely-reported LM benchmarks, including MMLU, DROP, and HellaSwag (Table 5).

**Why this is non-obvious**: The relationship between data quality and model performance is not linear or guaranteed. It is entirely possible that noise in extraction is "washed out" by the scale of pretraining—that a model trained on trillions of tokens can learn to ignore extraction artifacts, or that downstream benchmarks are not sensitive to the specific types of errors that extraction tools introduce. The paper's result demonstrates that extraction fidelity does matter, at least for the specific domain (scientific papers), model scale (7B parameters), and continued pretraining regime (50B tokens) tested.

What makes this an innovation rather than a routine ablation is the **design of the comparison**: the same set of source PDFs, processed with two different extraction pipelines, producing two versions of the same corpus that differ only in extraction quality. This controls for all other variables—document selection, domain distribution, token count, pretraining recipe—and isolates the effect of extraction fidelity. The paper does not claim that olmOCR-extracted data is inherently "better" for all purposes, only that it produces better downstream performance for this specific LM training setup, which is a more careful and falsifiable claim.

**The significance of Grobid as a baseline**: Grobid is not a weak baseline—it has been the standard tool for processing scientific PDFs in the NLP community for over a decade, and peS2o (processed with Grobid) has been used in training major open-source models including the OLMo series. The fact that olmOCR-extracted peS2o outperforms Grobid-extracted peS2o by a non-trivial margin suggests that even well-established, domain-specific extraction pipelines leave meaningful quality on the table. This has direct implications for the many ongoing efforts to build improved training corpora from PDF sources (the DCLM, FineWeb, and Dolma projects all rely heavily on PDF extraction for certain domains).

**Limitations of this innovation**: The experiment is limited to a single model checkpoint (OLMo-2-7B-1124), a single continued training duration (50B tokens), and a single domain (academic papers). The paper does not test whether the improvement scales with model size, whether it persists through full pretraining (rather than continued pretraining), or whether it generalizes to non-academic domains. These are acknowledged implicitly by the framing as a "downstream evaluation" rather than a comprehensive scaling study. The result is best understood as a **proof of concept**—extraction quality matters for LM training—rather than a full characterization of when and how much it matters.

This is an **incremental empirical finding** in the abstract, but a **fundamentally important validation** for the document extraction community, which has long argued that better extraction enables better models without being able to point to controlled experimental evidence. It closes the loop from extraction quality to downstream impact, making the case for investing in improved extraction pipelines with concrete performance numbers rather than intuition.

---

### Innovation 5: The Identification and Operationalization of Cost-Performance Pareto Efficiency for Document Extraction

The paper's final innovation is an **analytical framing** rather than a technical contribution: the explicit treatment of PDF extraction as an optimization problem on a cost-performance Pareto frontier, and the demonstration that olmOCR occupies a previously unoccupied region of that frontier. Figure 1 visualizes this directly, plotting tools on axes of overall performance (olmOCR-Bench pass rate) against cost per million pages (logarithmic USD scale).

**What distinguishes this from routine cost comparisons**: Most papers report cost as an afterthought—"our system is X% cheaper than GPT-4o"—without analyzing the shape of the cost-performance tradeoff. The paper's Figure 1 reveals a distinct structure: there is a cluster of open-source tools (Marker, MinerU, GOT-OCR) with moderate performance (48–70%) at low-to-moderate cost ($148–$596 per million pages), a cluster of commercial VLMs (GPT-4o, Gemini Flash 2) with higher performance (58–70%) at much higher cost ($249–$12,480 per million pages), and olmOCR sitting at the upper-left extreme—highest performance (75.5%) at the lowest cost ($176 per million pages). This is a **Pareto-dominant** position: no other tool matches or exceeds olmOCR on both performance and cost simultaneously.

The paper does not simply claim to be cheaper or better; it claims to have broken the previous cost-performance tradeoff entirely. GPT-4o is roughly 35× more expensive for lower performance. Marker is roughly 8× more expensive for lower performance. MinerU is roughly 3.4× more expensive for substantially lower performance. The paper's contribution is not just building a good system but demonstrating that the previous Pareto frontier—where better performance required paying more—was an artifact of using general-purpose, non-specialized models, and that task-specific fine-tuning collapses the frontier.

**Why this matters for the field**: The cost-performance frontier framing changes the conversation around document extraction from "which tool is best?" to "which tool is best for a given budget and quality requirement?" For some applications (small-scale processing where per-page cost is irrelevant), the commercial VLM cluster might still be preferred for convenience or API access. For large-scale LM training data curation, the cost-performance frontier makes the choice unambiguous: olmOCR provides the best quality at the lowest cost. The paper's packaging of this analysis in a single, easily-grasped visualization (Figure 1) makes the argument self-contained and immediately persuasive to practitioners making resource allocation decisions.

The inference cost breakdown in Table 6 further operationalizes this framing by providing exact throughput measurements (tokens/second), cost calculations (pages/USD and cost per million pages), and the specific hardware configurations used. This is not just transparency—it is a **cost model** that enables other researchers to estimate what olmOCR would cost on their own infrastructure or to identify where further efficiency improvements would have the largest impact (for example, reducing the 12% retry rate would directly improve cost-per-page by a corresponding amount).

This is an **analytical contribution** whose value lies in making explicit what was previously implicit—that cost is a first-class dimension of extraction system quality, not an implementation detail—and providing the data and framework to reason about it rigorously. It is incremental in the sense that cost comparisons are standard practice, but fundamental in the sense that the paper uses this analysis to reframe the entire problem statement: the goal is not just accurate extraction, but **accurate extraction at a cost that makes large-scale processing viable**.

## 5. Experimental Analysis

### Evaluation Methodology

- **Dataset.** The primary evaluation uses olmOCR-Bench, a purpose-built benchmark of 1,402 PDF documents comprising 7,010 unit test cases spanning seven document categories (arXiv Math, Old Scans Math, Tables, Old Scans, Headers Footers, Multi Column, Long Tiny Text) plus baseline tests applied to every document. Documents are sourced from arXiv, the Internet Archive, the Library of Congress, and the authors' internal crawled PDF repository (Table 10). The benchmark was constructed *after* training olmOCR-7B-0225-preview to prevent iterative overfitting to the evaluation set (Section 4.1).

- **Base model(s).** The core model is olmOCR-7B-0225-preview, a Qwen2-VL-7B-Instruct checkpoint fine-tuned on olmOCR-mix-0225. Baseline comparisons include the base Qwen2-VL-7B-Instruct without fine-tuning, Qwen2.5-VL-7B (a newer model from the same family, tested to assess whether fine-tuning a slightly older architecture outperforms a newer base model), GPT-4o (gpt-4o-2024-08-06, the teacher model used for data generation), GPT-4o in batch mode (half-price inference), Gemini Flash 2.0, Mistral OCR API (a commercial dedicated OCR service), and open-source tools Marker v1.7.5, MinerU v1.3.10, and GOT OCR (Table 4).

- **Metrics.** The primary metric is **overall pass rate (%) on olmOCR-Bench**, computed as the macro-average of per-category pass rates across the seven document source categories plus baseline tests. Each category contributes equally to the final score regardless of the number of test cases it contains (Section 3.3). Each individual test within a category is scored as pass/fail based on deterministic, machine-verifiable rules—text presence, text absence, reading order, table cell relationships, or rendered formula symbol matching. 95% confidence intervals are calculated via bootstrapping with 10,000 resamples. Additional metrics include word-level alignment to GPT-4o teacher outputs (Appendix C.1, using Hirschberg's algorithm for sequence alignment), ELO ratings from pairwise human judgments (Appendix C.2), and downstream LM benchmark scores (MMLU, ARCC, DROP, HellaSwag, NaturalQuestions, WinoGrande) following continued pretraining (Section 4.2).

- **Baselines.** The paper evaluates against three categories of baselines (Table 4, Figure 1):
  - **Open-source specialized tools**: Marker v1.7.5 (Paruchuri, 2025), MinerU v1.3.10 (Wang et al., 2024a), and GOT OCR (Wei et al., 2024)—pipeline-based or end-to-end systems specifically designed for document extraction and linearization.
  - **Open VLMs**: Qwen2-VL-7B-Instruct (the base model before fine-tuning), Qwen2.5-VL-7B (Bai et al., 2025, a newer generation from the same model family).
  - **Commercial API tools**: GPT-4o (OpenAI et al., 2024), GPT-4o in batch mode (50% cost reduction), Gemini Flash 2.0 (Google, 2025), Gemini Flash 2.0 in batch mode, and Mistral OCR API (Mistral, 2025).

  Each VLM baseline is tested both with and without document-anchoring (denoted "Anchored" vs. "No Anchor" in Table 4), where applicable. For commercial APIs, anchored versions include the same pypdf-extracted text blocks and coordinates that olmOCR uses.

- **Generation budget / compute accounting.** For benchmark evaluation, cost is measured in USD per million pages processed, incorporating both the model's throughput (tokens per second) and the GPU rental or API pricing (Table 6). For local models (olmOCR, MinerU, Marker), cost is calculated from measured throughput on specific GPU hardware multiplied by hourly rental rates ($0.79/hour for NVIDIA L40S, $2.69/hour for H100 80GB SXM, both from RunPod pricing as of February 2025). For API models, cost uses published per-token pricing (GPT-4o: $2.50/M input tokens, $10.00/M output tokens, batch pricing at 50% of these rates; Gemini Flash 2.0: $0.10/M input, $0.40/M output). Mistral OCR API is priced at $1 per 1,000 pages flat rate. Throughput measurements include a 12% retry rate for olmOCR (Table 6 caption). For downstream LM evaluation, the compute budget is fixed at 50B tokens of continued pretraining on OLMo-2-7B-1124, with two variants of the peS2o corpus differing only in extraction pipeline (Grobid + rules vs. olmOCR), holding document selection and all other pretraining hyperparameters constant.

- **Cross-validation / statistical protocol.** The olmOCR-Bench overall score is macro-averaged across eight categories (seven document sources plus baseline tests), with 95% confidence intervals computed via bootstrap resampling with 10,000 samples (Table 4). For intrinsic human evaluation (ELO ratings), 2,000 comparison pairs were sampled from a separate held-out set of 2,017 PDFs not used in training, with 11 data researchers providing 452 expressed preferences; ELO ratings are averaged over 100 simulations with 95% confidence intervals from 5,000 bootstrap resamples (Appendix C.2). For downstream continued pretraining, benchmark scores are standard evaluation metrics without confidence intervals reported (Table 5). No cross-validation is applied to model selection during training—the final model was selected based on manual qualitative side-by-side evaluation during development (Appendix C, Figure 6).

---

### Main Quantitative Results

#### Benchmark Performance: olmOCR vs. All Baselines (Table 4, Figure 1)

The headline result is that olmOCR-7B-0225-preview with document-anchoring achieves an overall pass rate of **75.5%** (95% CI ±1.0%) on olmOCR-Bench, outperforming every baseline system including commercial APIs, dedicated OCR tools, and the base Qwen2-VL model by substantial margins. The full ranking from Table 4 is:

| System | Overall Pass Rate |
|---|---|
| olmOCR (v0.1.75, Anchored) | **75.5 ± 1.0%** |
| olmOCR (v0.1.75, No Anchor) | 74.7 ± 1.1% |
| Mistral OCR API | 72.0 ± 1.1% |
| Marker v1.7.5 | 70.1 ± 1.1% |
| GPT-4o (Anchored) | 69.9 ± 1.1% |
| GPT-4o (No Anchor) | 68.9 ± 1.1% |
| Qwen 2.5 VL (No Anchor) | 65.5 ± 1.2% |
| Gemini Flash 2 (Anchored) | 63.8 ± 1.2% |
| MinerU v1.3.10 | 61.5 ± 1.1% |
| Gemini Flash 2 (No Anchor) | 57.8 ± 1.1% |
| GOT OCR | 48.3 ± 1.1% |
| Qwen 2 VL (No Anchor) | 31.5 ± 0.9% |

The 4.1 percentage point gap between olmOCR (75.5%) and the best commercial baseline (Mistral OCR, 72.0%) is outside the overlap of their 95% confidence intervals (75.5 ± 1.0% vs. 72.0 ± 1.1%), confirming statistical significance. The gap to GPT-4o (the teacher model) is 5.6 percentage points (75.5% vs. 69.9% for anchored, 68.9% for unanchored), which is also statistically significant. Notably, olmOCR without anchoring (74.7%) still outperforms all baselines, indicating that the fine-tuned model's advantage is not solely attributable to the anchoring mechanism—the model has genuinely learned better extraction behavior than its teacher.

The fine-tuning effect is isolated by comparing Qwen2-VL-7B-Instruct (the base model, 31.5% with no anchor) against olmOCR (74.7% with no anchor, or 75.5% anchored). This **+43.3 percentage point** improvement from fine-tuning on olmOCR-mix-0225 represents the largest single gain in the benchmark. The even larger comparison against Qwen2.5-VL (65.5%, a newer model from the same family) shows that fine-tuning a slightly older architecture with task-specific data substantially outperforms architectural improvements without task-specific training (75.5% vs. 65.5%, a +10.0 point gap).

#### Document-Anchoring Ablation Within Systems (Table 4)

The effect of document-anchoring varies dramatically across models, revealing that anchoring is most beneficial for models that struggle with visual OCR but can effectively use text hints:

- **olmOCR**: Anchored (75.5%) vs. No Anchor (74.7%): +0.8 points. The model has learned to extract well even without anchors, but anchoring provides a small additional benefit.
- **GPT-4o**: Anchored (69.9%) vs. No Anchor (68.9%): +1.0 point. The teacher model benefits modestly from anchoring, but even with anchors it underperforms the fine-tuned student.
- **Gemini Flash 2**: Anchored (63.8%) vs. No Anchor (57.8%): +6.0 points. This is the largest anchoring boost observed, suggesting Gemini Flash 2's visual OCR is relatively weak but it can effectively leverage the text hints.
- **Qwen 2 VL**: No Anchor only (31.5%). The base model was not tested with anchoring, so the incremental benefit of anchoring alone versus fine-tuning alone cannot be isolated.

The comparison between olmOCR (No Anchor, 74.7%) and GPT-4o (Anchored, 69.9%) is particularly informative: the 7B fine-tuned model without any PDF metadata outperforms the much larger GPT-4o even when GPT-4o is given the same metadata. This demonstrates that fine-tuning on task-specific data provides capabilities that prompting alone cannot elicit, even from a dramatically larger model.

#### Per-Category Analysis (Table 4)

The breakdown by document category reveals which types of content each system handles well and where failure modes concentrate:

- **arXiv Math (AM, 2,927 tests)**: olmOCR achieves 74.9% (anchored), compared to Mistral OCR at 77.2% (the best score in this category) and GPT-4o at 53.5%. This is the only category where olmOCR does not lead—Mistral OCR's specialized math handling slightly edges it out. However, olmOCR substantially outperforms GPT-4o (+21.4 points), GOT OCR (+22.2 points), and Qwen2.5-VL (+11.8 points). The anchored advantage for olmOCR in this category (+3.4 points over no-anchor) is larger than the overall anchored advantage (+0.8 points), suggesting that anchored text is particularly helpful for disambiguating complex LaTeX formulas.

- **Old Scans Math (OSM, 458 tests)**: GPT-4o leads at 75.5%, with olmOCR at 71.2%. This is the only category where GPT-4o (the teacher) outperforms its student. The gap likely reflects GPT-4o's superior out-of-the-box handwriting recognition on degraded historical scans, a capability that the 7B student model approximates but does not fully match through distillation.

- **Tables (TA, 1,020 tests)**: Gemini Flash 2 (Anchored) achieves the highest score at 72.1%, slightly ahead of olmOCR at 71.0%. This is a notable result—a commercial API with anchoring outperforms the fine-tuned model on table extraction, suggesting that table structure recognition may benefit more from model scale than from task-specific fine-tuning on the current dataset. GOT OCR's catastrophic 0.2% pass rate on tables indicates it essentially produces no parseable table representations.

- **Old Scans (OS, 526 tests)**: olmOCR achieves 42.2% (anchored), the highest in this category, with GPT-4o at 40.7% and all other systems substantially lower (MinerU: 17.3%, Qwen2-VL: 17.1%). All systems struggle on old scans—the absolute pass rates are the lowest across all categories—indicating this remains an open challenge. olmOCR's 42.2% pass rate, while best-in-class, means it fails on more than half of old scan test cases.

- **Headers Footers (HF, 753 tests)**: MinerU achieves the highest score at 96.6%, substantially ahead of olmOCR (94.5%), Marker (84.9%), and GPT-4o (93.8%). The pipeline-based MinerU's strong performance here reflects that header/footer removal is a well-defined layout analysis problem that can be solved with heuristics effectively; end-to-end models occasionally include page numbers or running headers that should be stripped. The fact that olmOCR outperforms GPT-4o (94.5% vs. 93.8%) suggests fine-tuning helps with this specific suppression behavior.

- **Multi Column (MC, 884 tests)**: olmOCR achieves 78.3% (anchored), well ahead of Marker (72.9%), Mistral OCR (71.3%), and GPT-4o (69.3%). Multi-column reading order is one of the most challenging aspects of PDF linearization, and olmOCR's substantial lead here (a 9-point gap over the nearest open-source competitor) is a key driver of its overall advantage. Qwen2-VL's catastrophic 8.3% on this category confirms that general-purpose VLMs without fine-tuning cannot handle multi-column layouts.

- **Long Tiny Text (LTT, 442 tests)**: Marker achieves 84.6%, Gemini Flash 2 (No Anchor) achieves 84.4%, and olmOCR achieves 73.3%. This is the only category where olmOCR substantially underperforms multiple baselines. The dense, small-print documents (dictionary pages, reference lists) may push against the 1024-pixel maximum rendering dimension used during training, causing information loss that larger-context or higher-resolution systems handle better.

- **Baseline tests (applied to all documents)**: All systems except Qwen2-VL-7B-Instruct (55.5%, indicating frequent language switching or empty outputs) achieve high scores (94.0–99.4%), confirming that catastrophic failures are rare across modern extraction tools.

#### Cost-Performance Analysis (Figure 1, Table 6)

The paper's cost-performance framing reveals that olmOCR occupies a previously unoccupied Pareto-optimal position:

- **olmOCR (L40S)**: 75.5% overall pass rate at **$176 per million pages** (5,697 pages per USD)
- **Mistral OCR API**: 72.0% at $1,000 per million pages (5.7× more expensive for lower performance)
- **Marker v1.7.5**: 70.1% at $1,484 per million pages (8.4× more expensive for lower performance)
- **GPT-4o (Batch)**: 69.9% at $6,240 per million pages (35.5× more expensive for lower performance)
- **GPT-4o (non-Batch)**: 68.9% at $12,480 per million pages (70.9× more expensive for lower performance)

The gap between olmOCR and GPT-4o (Batch) represents a 35.5× cost reduction for 5.6 percentage points *higher* performance—a reversal of the usual cost-quality tradeoff where higher quality commands a premium. This is the paper's most practically significant result: it demonstrates that the previous cost-performance Pareto frontier was an artifact of using general-purpose models for a task that benefits dramatically from specialization.

The throughput measurements underlying these cost calculations (Table 6, Appendix B):

- **olmOCR on L40S**: 906 output tokens/second, processing 1,288 test pages in 17 minutes 10 seconds. At $0.79/hour, this yields $0.226 for the test set, or 5,697 pages per dollar.
- **olmOCR on H100**: 3,050 output tokens/second, processing the same pages in 5 minutes 7 seconds. At $2.69/hour, this yields $0.229 for the test set, or 5,632 pages per dollar. The per-page costs on L40S and H100 are nearly identical ($176/M vs. $178/M) because the H100's 3.37× throughput advantage is offset by its 3.41× higher hourly cost.

A notable finding is that olmOCR achieves comparable per-page cost on L40S and H100 GPUs, meaning users can choose hardware based on availability rather than cost optimization. The L40S may be preferred for its lower absolute hourly cost and wider availability; the H100 may be preferred when latency matters (a single job completes in 5 minutes rather than 17).

#### Intrinsic Human Evaluation: ELO Ratings (Appendix C.2, Figure 7, Table 9)

The pairwise human evaluation provides an alternative quality assessment independent of the benchmark's unit-test design. The paper collected 452 pairwise judgments from 11 data researchers comparing olmOCR against Marker, GOT-OCR, and MinerU on a held-out set of 2,017 PDFs from the same distribution as the training data (not the benchmark).

- **olmOCR ELO**: Over 1800, substantially higher than all baselines.
- **Marker ELO**: Approximately 1625.
- **MinerU ELO**: Approximately 1450.
- **GOT-OCR ELO**: Approximately 1400.

The pairwise win rates in Table 9 corroborate:
- olmOCR vs. Marker: 49/31 wins (61.3%)
- olmOCR vs. GOT-OCR: 41/29 wins (58.6%)
- olmOCR vs. MinerU: 55/22 wins (71.4%)
- Marker vs. MinerU: 53/26 wins (67.1%)
- Marker vs. GOT-OCR: 45/26 wins (63.4%)
- GOT-OCR vs. MinerU: 38/37 wins (50.7%)

The relatively even GOT-OCR vs. MinerU comparison (50.7% win rate, essentially a tie) and the substantial gap between olmOCR and all three baselines (11.4–21.4 percentage point win rate advantage) suggest that the human evaluation ordering is consistent with the benchmark ordering (olmOCR > Marker > MinerU ≈ GOT-OCR), though the absolute gaps differ. The ELO methodology—averaging over 100 simulations and bootstrapping 95% confidence intervals—provides statistical rigor, but the sample size (75 judgments per pair on average) is relatively small, meaning the exact ELO values have wide confidence intervals.

The paper notes that 1,548 of the 2,000 sampled pairs were either "skipped for being too similar, or marked as invalid" (Appendix C.2), meaning only 22.6% of comparisons elicited a clear preference. This high rate of "too similar" judgments suggests that for many PDFs in the random sample, the differences between tools are subtle rather than dramatic—the tools diverge most on challenging documents, which the benchmark was specifically designed to include but which may be rarer in a random PDF sample.

#### Downstream Language Model Evaluation (Section 4.2, Table 5)

The continued pretraining experiment tests whether improved extraction quality translates to measurable improvements in language model training. Starting from an OLMo-2-7B-1124 intermediate checkpoint, the model was trained for an additional 50B tokens on two versions of peS2o:

1. **peS2o (Grobid + rules)**: The original peS2o corpus (Soldaini and Lo, 2023), processed using Grobid with additional heuristic cleaning.
2. **olmOCR-peS2o**: The same set of source PDFs reprocessed with olmOCR.

The results in Table 5:

| Benchmark | peS2o (Grobid + rules) | olmOCR-peS2o | Δ |
|---|---|---|---|
| Average | 53.9 | **55.2** | +1.3 |
| MMLU | 61.1 | 61.1 | 0.0 |
| ARCC | 75.0 | 76.4 | +1.4 |
| DROP | 42.3 | 43.7 | +1.4 |
| HellaSwag | 57.4 | 62.6 | +5.2 |
| NaturalQuestions | 29.4 | 29.1 | −0.3 |
| WinoGrande | 58.3 | 58.0 | −0.3 |

The +1.3 percentage point average improvement is driven primarily by a substantial **+5.2 point gain on HellaSwag** (a commonsense reasoning benchmark), with smaller gains on ARCC (+1.4) and DROP (+1.4), and essentially no change on MMLU (0.0), NaturalQuestions (−0.3), and WinoGrande (−0.3).

This pattern is informative: HellaSwag requires understanding of narrative coherence and commonsense reasoning, which may benefit from the improved reading order and paragraph coherence that olmOCR provides—extraction errors that scramble sentence order or insert artifacts would be particularly damaging for this task. MMLU, NaturalQuestions, and WinoGrande rely more heavily on factual knowledge, which extraction quality may affect less directly (the facts are either present in the text or not; extraction quality primarily affects whether those facts are in readable context). The null result on MMLU also suggests that the continued pretraining of 50B tokens is not enough to shift factual knowledge substantially regardless of extraction quality.

A critical caution: the paper does not report confidence intervals or statistical significance for these downstream results, and the 50B token continued training budget is modest relative to the model's full pretraining. The +5.2 point HellaSwag improvement is the most striking result, but without error bars, it should be interpreted as suggestive rather than definitive. The paper frames this experiment as a demonstration that extraction quality *can* matter for downstream performance, not as a comprehensive measurement of *how much* it matters across different training regimes, model scales, and benchmark suites.

#### Alignment with Teacher Model (Appendix C.1, Tables 7 and 8)

The paper measures word-level alignment between various models' outputs and GPT-4o's silver data to assess how faithfully the student model has learned the teacher's behavior:

- **GPT-4o self-alignment**: 0.954 (Table 7). Even the same model does not perfectly reproduce itself due to stochastic decoding.
- **olmOCR-7B-0225-preview (τ = 0.1)**: 0.875 alignment to GPT-4o silver data.
- **olmOCR-7B-0225-preview (τ = 0.8)**: 0.859 alignment.
- **GPT-4o mini**: 0.833 alignment.

The fine-tuned 7B model achieves higher alignment to GPT-4o's outputs than GPT-4o-mini does (87.5% vs. 83.3%), despite being from a different model family and one-third the parameter count (assuming GPT-4o-mini is roughly 20B+ parameters). This demonstrates that **task-specific fine-tuning is more important for teacher alignment than model scale or family similarity**. The temperature effect is consistent with expectations: higher temperature (τ = 0.8) reduces alignment (0.875 → 0.859) but helps mitigate repetition failures during inference (Appendix D.2).

The alignment distribution in Table 8 shows that at τ = 0.1, 57.3% of pages (700/1,221) achieve high alignment (>95% word match), 29.7% achieve medium alignment (70-95%), and 12.9% achieve low alignment (<70%). At τ = 0.8, high alignment drops to 52.0% (636/1,221) with low alignment increasing to 16.0% (195/1,221). The majority of pages in both temperature settings achieve high alignment, confirming that the model generally produces extractions very close to the teacher's, with failures concentrated in a minority of challenging pages.

---

### Ablation Studies and Robustness Checks

**Full fine-tuning vs. LoRA (Appendix C, Figures 4, 5)**: The paper compares full fine-tuning (updating all 7B parameters) against LoRA (Low-Rank Adaptation, a parameter-efficient method training only adapter matrices). On both web PDFs and Internet Archive books, full fine-tuning achieves consistently lower validation loss throughout training (1,200 steps). The paper selects full fine-tuning for the final model based on this validation loss difference, though it does not report benchmark performance for the LoRA variant. The training cost tradeoff (16 node-hours for full fine-tuning vs. presumably less for LoRA) is not quantified. The paper's validation loss curves (Figures 4 and 5) show the gap between full fine-tuning and LoRA narrowing slightly toward the end of training, raising the question of whether longer LoRA training could close the gap entirely—this is not investigated.

**Temperature sweep (Appendix C.1, Table 7)**: Comparing τ = 0.1 (near-deterministic) and τ = 0.8 (moderate stochastic sampling), the paper finds:
- Teacher alignment drops from 0.875 to 0.859 (a 1.6 percentage point decrease)
- High-match pages decrease from 700 to 636 (a 9.1% reduction)
- Low-match pages increase from 158 to 195 (a 23.4% increase)

The paper selects τ = 0.8 for production inference despite the alignment cost because it "reduces the likelihood of repetitions occurring" (Appendix D.2). This is a deliberate quality-robustness tradeoff: slightly less faithful extractions on average, but fewer catastrophic failures that require retries. The paper reports a 12% overall retry rate (Table 6 caption) but does not decompose this into repetition failures vs. other failure types, nor does it report the retry rate at τ = 0.1 for comparison. This makes it impossible to quantify exactly how much the temperature increase reduces retries.

**Anchoring ablation across models (Table 4)**: As discussed in Section 4.1, anchoring provides dramatically different benefits across systems: +6.0 points for Gemini Flash 2, +1.0 point for GPT-4o, +0.8 points for olmOCR. The paper does not ablate *why* anchoring helps different models differently—possible explanations include differences in visual OCR capability, differences in the model's ability to attend to long text prompts alongside images, or differences in instruction-following for the anchoring format. This is a missed opportunity for a mechanistic understanding of how anchoring interacts with model architecture and scale.

**Loss masking strategy (Section 2.3)**: The paper masks loss so only response tokens contribute to training. This is standard practice for instruction fine-tuning, but the paper does not ablate this choice—what would happen if prompt tokens were included in the loss? Would the model learn to better attend to anchored text if it were also trained to predict the prompt? Without this ablation, we cannot rule out that a different loss masking strategy could improve anchoring utilization.

**Data composition (Tables 1, 2)**: The training set olmOCR-mix-0225 includes both born-digital web PDFs (93.2% of documents) and scanned Internet Archive books (5.7% of documents). The paper does not ablate the contribution of the scanned book data. Would performance on Old Scans and Old Scans Math degrade without these training examples? Could the model learn to handle scanned documents purely from web PDFs that happen to contain embedded images? The paper's validation loss curves (Figures 4, 5) separately track web PDFs and Internet Archive books but do not test a model trained on web-only data.

**Non-English language handling**: The training data is filtered to English-only using the Lingua language detection package (Section 2.1). The paper does not report performance on non-English documents or test whether olmOCR can extract text in other languages. This is a deliberate scoping decision rather than a failure, but it means the reported benchmark results apply only to English-language PDFs. The paper's baseline tests include a check that the output does not contain Chinese, Japanese, or Emoji Unicode characters (Section 3.1), which would penalize any system that incorrectly outputs these characters, but this tests for a failure mode (language switching) rather than genuine multilingual capability.

**GPT-4o data generation alternatives (Section 2.2, footnote 3)**: The paper reports that it evaluated but rejected several alternatives for the teacher model: Gemini 1.5 (eliminated due to "frequent RECITATION errors"), GPT-4o mini ("produced excessive hallucinations"), and Claude Sonnet 3.5 ("cost-prohibitive"). These are qualitative assessments rather than quantitative comparisons, and the paper does not provide benchmark results or cost estimates for the rejected alternatives. The choice of GPT-4o as teacher is therefore justified by process-of-elimination reasoning rather than head-to-head data quality comparison.

**ReST^EM revision model training**: The paper does not include this ablation in the main text, but the prior sections mention that the ReST^EM-trained revision model degraded performance (a negative result documented in Appendix K). This is relevant mainly to the revision modeling component of the broader text extraction problem space and is discussed in detail in the prior sections.

**Rotation handling (Appendix D.2)**: The paper's inference pipeline includes automatic rotation correction based on the model's `is_rotation_valid` and `rotation_correction` JSON fields. The paper does not report how often rotation correction is triggered, what the accuracy of rotation detection is, or whether incorrect rotation detection causes failures (e.g., rotating a correctly oriented page). This is a robustness feature whose real-world impact is uncharacterized.

**Prompt length and context window management (Appendix D.2)**: The paper's exponential backoff strategy for fitting prompts within the 8,192-token context window is described but not quantitatively ablated. The paper does not report what fraction of pages trigger the backoff, what the typical final character limit is after backoff, or whether performance degrades measurably when fewer anchored text hints are provided. This is a robustness mechanism whose necessity and effectiveness are asserted but not evaluated.

**Retry rate and failure analysis (Table 6, Appendix D.2)**: The 12% retry rate is reported but not decomposed by failure type. How many retries are due to JSON parsing failures vs. repetition degeneration vs. other causes? Does retry success rate vary by document type? The paper notes that a limitation of the current retry mechanism is that "letting generations repeat up to maximum sequence length uses significant memory within SGLang" (Appendix D.2), indicating that faster repetition detection is planned future work, but the current performance impact of this inefficiency is not quantified.

---

### Critical Assessment

#### Does the paper demonstrate that olmOCR outperforms "top VLMs including GPT-4o, Gemini Flash 2 and Qwen-2.5-VL"?

**Yes, with qualifications.** The benchmark results in Table 4 unambiguously show olmOCR (75.5%) outperforming GPT-4o (69.9% anchored, 68.9% unanchored), Gemini Flash 2 (63.8% anchored, 57.8% unanchored), and Qwen2.5-VL (65.5% unanchored) by substantial, statistically significant margins. The human evaluation (ELO ratings in Figure 7) corroborates olmOCR's advantage against the tested baselines, though GPT-4o, Gemini Flash 2, and Qwen2.5-VL were not included in the pairwise human comparison (only Marker, GOT-OCR, and MinerU were).

However, the claim requires three qualifications:

1. **Per-category nuance**: olmOCR does not lead in every category. Mistral OCR outperforms it on arXiv Math (77.2% vs. 74.9%), GPT-4o outperforms it on Old Scans Math (75.5% vs. 71.2%), Gemini Flash 2 (Anchored) outperforms it on Tables (72.1% vs. 71.0%), and Marker outperforms it on Long Tiny Text (84.6% vs. 73.3%). The overall advantage is driven by dominating the categories with the most test cases or the largest relative gaps (Multi Column: +5.4 points over the nearest competitor; Old Scans: best-in-class despite low absolute scores). A practitioner whose document distribution is heavily skewed toward one of olmOCR's weak categories might prefer a different tool.

2. **The benchmark was designed by the authors**: Despite the paper's claim that olmOCR-Bench was developed "after training olmOCR-7B-0225-preview to prevent unfairly iterating on the benchmark before comparing with other methods" (Section 4.1), the test case creation process involved running olmOCR to identify candidate pages (arXiv Math, Old Scans Math), and the document categories were chosen because they reflect "what we found olmOCR (or its earlier iterations) often struggled to process" (Section 3.2). This means the benchmark is not neutral with respect to olmOCR's development—it is explicitly designed to test capabilities that the authors knew were challenging during development. This is appropriate for a diagnostic benchmark, but it means the absolute performance numbers may reflect a benchmark that is somewhat tuned to olmOCR's known failure modes, potentially overstating the gap to systems whose failure modes are different.

3. **Anchoring is not equally applied**: GPT-4o and Gemini Flash 2 are tested both with and without anchoring, and their anchored variants use the same pypdf-extracted text as olmOCR. However, the anchored prompt for GPT-4o data generation (Appendix E.1) is substantially more detailed than the prompt used for olmOCR inference (Appendix E.2). The paper does not test whether providing the *more detailed* prompt to GPT-4o during evaluation (rather than the same simplified prompt olmOCR receives) would close the performance gap. This is a minor asymmetry but worth noting since prompt engineering is known to substantially affect VLM performance.

#### Does the paper demonstrate that olmOCR can convert a million PDF pages for only $176 USD?

**Yes, for the specific hardware and workload tested.** The cost calculation in Table 6 and Appendix B is detailed and transparent: 5,697 pages per dollar on an NVIDIA L40S at $0.79/hour, with measurements from 1,288 test pages, including a 12% retry rate. The $176 per million pages figure is directly derived from this measured throughput.

However, the real-world applicability of this number depends on several factors the paper does not fully address:

1. **The cost of document-anchoring itself**: The $176 figure includes only VLM inference cost. It does not include the computational cost of running pypdf to extract anchored text, which for a million pages could be non-trivial (pypdf must parse the entire PDF binary to extract text blocks and coordinates). The paper does not report pypdf processing time or cost, presumably because it is small relative to GPU inference, but this should be confirmed.

2. **The cost of PDF-to-image rendering**: Each page must be rasterized to an image before being fed to the VLM. The paper does not account for this cost. PDF rendering at 1024 pixels maximum dimension is fast but not zero-cost, and at million-page scale could accumulate.

3. **The 12% retry rate**: Retried pages consume additional GPU time. The paper includes this in the cost calculation ("we estimate its costs at $0.226" for the test set, which includes retries), so the $176 figure is a measured cost including retries, not an ideal cost without retries. This is appropriately conservative.

4. **GPU rental pricing volatility**: The paper uses RunPod on-demand pricing as of February 2025 ($0.79/hour for L40S). GPU rental prices fluctuate with demand, and reserved/committed-use pricing can be substantially lower than on-demand. The $176 figure is a snapshot at a specific time and provider; actual costs will vary.

5. **Scale assumptions**: The cost-per-page measurement is based on processing 1,288 pages in 17 minutes. At million-page scale, the total job would take approximately 220 hours on a single L40S, or proportionally less with multiple GPUs. The paper's batch coordination system (Appendix D.1) is designed for exactly this scaling, but the paper does not report whether throughput remains linear at very large batch sizes or whether coordination overhead becomes significant.

The cost comparison to GPT-4o ($176 vs. $6,240 per million pages) is the most robust aspect of this claim because it uses published API pricing for GPT-4o and measured throughput for olmOCR, eliminating hardware pricing volatility as a confound for the comparison (GPT-4o costs are in dollars per token, not hardware-dependent).

#### Does the paper demonstrate that training on olmOCR-extracted data improves language model pretraining?

**Yes, but the evidence is limited to a specific regime and should be interpreted as proof-of-concept rather than a general finding.** The continued pretraining experiment (Table 5) is well-controlled—same documents, same model, same training budget, different extraction pipelines—and shows a +1.3 percentage point average improvement across benchmarks, driven largely by a +5.2 point gain on HellaSwag.

The limitations that prevent generalization:

1. **Single model scale (7B), single continued training budget (50B tokens), single domain (scientific papers)**. The paper does not demonstrate that the improvement scales with model size, that it persists through full pretraining (as opposed to continued pretraining of an already-trained model), or that it generalizes to non-academic PDF domains. The +5.2 point HellaSwag gain might be specific to the OLMo-2-7B checkpoint and the 50B token budget; it could be smaller or larger at other scales.

2. **No confidence intervals or significance testing on downstream results.** The paper reports point estimates without error bars for MMLU, ARCC, DROP, HellaSwag, NaturalQuestions, and WinoGrande. These benchmarks have known variance depending on evaluation protocol (few-shot vs. zero-shot, exact prompt format, number of examples). Without error bars or multiple training seeds, we cannot assess whether the +1.3 point average improvement is statistically reliable or within the noise of continued pretraining variance.

3. **The comparison is against Grobid + rules, not against other extraction tools.** Grobid is a strong baseline for scientific papers, but it is not the state of the art among *all* extraction methods. The paper does not test whether olmOCR-peS2o outperforms a version of peS2o processed with MinerU, Marker, or GPT-4o. The result demonstrates that olmOCR is better than Grobid for this purpose, but not that it is better than all alternatives.

4. **The mechanism of improvement is not analyzed.** Why does HellaSwag improve by +5.2 points while MMLU shows no change? Is it because olmOCR produces better reading order (helping narrative coherence tasks) but similar factual content (not helping knowledge tasks)? Is it because olmOCR strips headers/footers more aggressively, reducing noise? The paper does not investigate these questions, leaving the causal pathway from extraction quality to downstream performance as a black box.

#### Does the paper demonstrate that olmOCR produces "significantly cleaner plain text than specialized open-source tools"?

**Yes, qualitatively and quantitatively.** The benchmark results (Table 4) show statistically significant gaps between olmOCR and Marker (75.5% vs. 70.1%), MinerU (75.5% vs. 61.5%), and GOT-OCR (75.5% vs. 48.3%). The human evaluation (Figure 7) shows olmOCR with an ELO rating over 1800, substantially above all three baselines (Marker ~1625, MinerU ~1450, GOT-OCR ~1400). The qualitative examples in Appendix G dramatically illustrate the differences: on the old scan page, MinerU produces "No text produced" for the Lincoln letter, GOT-OCR produces heavily garbled text with letter-level errors ("bchaving" for "behaving"), and Marker splits words across lines ("behaving them-" / "selves like Ma borne-"). olmOCR's output for the same page is nearly flawless.

The weakness in this claim is that the comparison is primarily against open-source tools that the paper itself identifies as having known limitations. The paper does not compare against the strongest possible configuration of these tools (e.g., Marker with different backends, MinerU with custom post-processing rules). The tools are run with default settings "as of January 14th, 2025" (Appendix C.2), which may not represent their optimal performance.

#### Missing experiments that would have strengthened the paper

1. **Multiple training seeds**: The paper reports a single fine-tuning run for the final model. Without multiple seeds, we cannot assess training variance—whether the reported 75.5% benchmark score is typical or unusually lucky.

2. **Scaling the training data**: The paper uses 260,000 pages. How does performance scale with dataset size? Would 500,000 pages provide substantial additional gains, or is performance saturating at 260,000? This is critical for practitioners deciding whether to invest in expanding the training set.

3. **Scaling the model**: The paper uses a 7B model. How does performance scale with model size? Would a 13B or 34B variant of Qwen2-VL fine-tuned on the same data outperform the 7B version, and if so, by how much? The cost-performance tradeoff might favor a larger model if the quality improvement justifies the higher per-page cost.

4. **Out-of-distribution generalization**: The benchmark includes document types present in the training data (academic papers, books, legal documents). How does olmOCR perform on document types completely absent from training—e.g., musical scores, engineering blueprints, restaurant menus, or handwritten letters in non-Latin scripts? This is the acid test for whether the model has learned general document extraction or has simply memorized the extraction patterns of the training distribution.

5. **Comparison to fine-tuned versions of other VLMs**: The paper shows that fine-tuning Qwen2-VL-7B yields large gains. Would fine-tuning a different 7B VLM (e.g., LLaVA-1.6, InternVL2) on the same data yield similar gains? This would distinguish whether the improvement comes from the fine-tuning recipe and data, or from specific properties of the Qwen2-VL architecture.

6. **Ablation of document-anchoring during training**: The model is trained with anchoring. Would a model trained without anchoring (pure visual input) perform better or worse at inference time without anchoring? The paper's no-anchor inference results (74.7%) suggest the model learns robust visual extraction, but we cannot tell whether training with anchoring helps or hurts the model's ability to extract without anchoring at test time.

7. **Evaluation of rotation handling accuracy**: The paper's pipeline includes automatic rotation correction, but the accuracy of this feature is never evaluated. How often does the model correctly detect and correct rotation? How often does it incorrectly rotate a correctly oriented page?

8. **Full pretraining (not just continued pretraining)**: The downstream experiment uses continued pretraining of an already-trained model. A stronger test would be to include olmOCR-peS2o as part of the pretraining data mix from scratch, measuring whether the extraction quality advantage persists through full pretraining where the model sees a much more diverse data distribution that might "wash out" the extraction quality signal.

## 6. Limitations and Trade-offs

### 6.1 The Difficulty Estimation Overhead Is Not Accounted for in the $176 Per Million Pages Figure

**The assumption or constraint:** The headline cost figure of $176 per million pages measures only VLM inference time on GPU hardware. It does not include the cost of the document-anchoring preprocessing step—running pypdf to extract text blocks, images, and spatial coordinates from each PDF page before the VLM ever sees it. The paper acknowledges this architectural dependency but treats pypdf extraction as negligible infrastructure cost:

> "our pipeline maintains high performance on documents that do not have any digital metadata encoded in them" (Appendix A)

This statement addresses capability on scanned documents but does not address the cost of attempting extraction on born-digital PDFs, which still requires parsing the PDF binary whether or not usable text is found. The cost measurement in Appendix B accounts for GPU time ("It processed 1,288 test pages in 17 minutes, 10 seconds") and retry overhead, but never measures or reports the CPU time spent in pypdf extraction, PDF-to-image rasterization, or the exponential-backoff prompt construction logic (Appendix D.2).

**The consequence:** For large-scale batch processing, the unmeasured preprocessing cost could meaningfully increase the true per-page cost. pypdf must parse the entire PDF binary to extract text blocks and coordinates—for complex born-digital PDFs with thousands of text elements per page, this is non-trivial CPU work. PDF rasterization at 1024 pixels maximum dimension also consumes CPU resources. The paper's exponential-backoff prompt strategy (Section 2.3, Appendix D.2) can trigger repeated pypdf extraction at decreasing character limits if the initial prompt exceeds the model's context window, multiplying the preprocessing cost for dense pages. None of these costs appear in the $176 figure. Additionally, the 12% retry rate reported in Table 6 reflects only VLM inference retries—pages that fail JSON parsing or degenerate into repetitions are re-generated, but the paper does not measure how often retries themselves require re-running the preprocessing pipeline (e.g., re-sampling anchored text blocks to get a different prompt for the retry attempt).

**What evidence exists in the paper:** The evidence is entirely absent. Table 6 provides exact GPU throughput measurements (906 tokens/second on L40S, 3,050 on H100) and per-page cost derivations, but no corresponding CPU-side measurements. The prompt construction logic in Appendix D.2 describes the exponential backoff mechanism but does not report what fraction of pages trigger it, what the typical final character limit is, or how much additional processing time this introduces. The paper's reported $0.226 cost for processing 1,288 test pages is derived purely from GPU rental time (17 minutes 10 seconds at $0.79/hour), with the implicit assumption that preprocessing time is negligible relative to GPU time. Whether this assumption holds depends on the complexity of the PDFs being processed and the hardware balance between CPU and GPU—on a system with a fast GPU but slow CPU, preprocessing could become the bottleneck. For the 7.9M document peS2o reprocessing task described in Section 4.2, even a small per-page preprocessing overhead would accumulate to hours of additional CPU time.

**Mitigation status:** Not attempted. The paper does not report, estimate, or even acknowledge the preprocessing cost as a component of total cost. This is a clear omission in an otherwise thorough cost analysis. The paper frames the $176 figure as the total cost of processing a million pages, but it is more accurately described as the GPU inference cost, with preprocessing as an unmeasured additional expense. For practitioners estimating total cost of ownership, this unmeasured component introduces uncertainty—the true cost could be $176 if preprocessing is indeed negligible, or substantially higher if it is not. The paper's open-source release of the full pipeline would allow external measurement of this cost, but the paper itself does not provide the data.

---

### 6.2 The Benchmark Is Developed by the Same Team That Built the Model, Using olmOCR to Identify Candidate Test Cases

**The assumption or constraint:** The paper states that olmOCR-Bench was developed "after training olmOCR-7B-0225-preview to prevent unfairly iterating on the benchmark before comparing with other methods" (Section 4.1). However, the test case creation methodology described in Section 3.2 reveals that olmOCR was used instrumentally during benchmark construction. For arXiv Math, the paper "ran olmOCR to identify candidate pages with TeX." For Old Scans Math, the authors "use olmOCR to find candidate pages with formulas." The document categories themselves were selected because they represent "what we found olmOCR (or its earlier iterations) often struggled to process" (Section 3.2). This creates a structural entanglement between the system being evaluated and the evaluation instrument.

**The consequence:** The benchmark is not independent of the model's development history. Test cases were sourced from pages where olmOCR (or its development versions) detected content of interest—formulas, tables, multi-column layouts. This means the benchmark's document distribution is conditioned on olmOCR's behavior: if an earlier olmOCR version failed to detect a formula on a particular page (i.e., it was a blind spot the paper does not discuss), that page would not become a candidate for the formula test set. The benchmark may therefore systematically under-represent failure modes that olmOCR shares with its earlier versions—the very failure modes that would be most informative for comparing olmOCR against other systems. Conversely, the benchmark over-represents failure modes that earlier olmOCR versions *did* detect as problematic, which may be exactly the failure modes that the final fine-tuned model was trained to address.

This is not an accusation of deliberate bias—the paper is transparent about the construction methodology—but it is a genuine methodological limitation. An independent benchmark constructed without any involvement of the system under evaluation would not have this entanglement. The paper's two-fold cross-validation protocol (Section 3.2 in the prior sections) addresses a different concern, namely overfitting the strategy selection to the test set, but does not address the more fundamental issue of the test set itself being shaped by the model's behavior.

**What evidence exists in the paper:** The paper explicitly documents the entanglement in Section 3.2, which is to its credit. The arXiv Math pipeline description is unambiguous: "we (1) ran olmOCR to identify candidate pages with TeX, (2) match pages back to original TeX source, and (3) validate matched TeX rendering compatibility with KaTeX." The Old Scans Math description similarly states: "We similarly use olmOCR to find candidate pages with formulas, but this time manually annotate each formula on the page to use as test cases." The document category selection rationale—"7 distinct document types that we found olmOCR (or its earlier iterations) often struggled to process"—is stated plainly. All of this is disclosed. The question is not about transparency but about whether the resulting benchmark can serve as an unbiased arbiter of relative system quality.

There is no ablation or analysis in the paper that quantifies how much this entanglement affects the benchmark's composition. The paper does not report what fraction of candidate pages identified by olmOCR were ultimately included as test cases versus rejected during manual review, nor what fraction of pages that *other* tools might have identified as problematic were missed because olmOCR did not flag them. The manual review step (described in Section 3.2 for each category) provides a quality filter on the automatically generated test cases, but does not add test cases from pages that the automated pipeline never considered.

**Mitigation status:** Partial. The paper's explicit disclosure is essential and appropriate. The use of manual review for all test cases (Section 3.2 describes "a second pass of human review for accuracy" for Old Scans; "manually reviewed those tests for accuracy" for Tables; "manually reviewed to remove mistakenly filtered text" for Headers Footers; "manually review each entry for accuracy" for Multi Column) ensures that the test cases themselves are valid, even if the set of documents tested is not a uniformly random sample of challenging PDFs. The diverse sourcing strategy—drawing from arXiv, the Internet Archive, the Library of Congress, and the internal crawl repository—provides some independence from the model's training distribution, though not from its test-time behavior since olmOCR was used in the candidate identification step regardless of source. The paper does not suggest future work to construct an independent benchmark or to audit the current benchmark for model-induced selection bias.

---

### 6.3 The Downstream LM Experiment Is a Single Point Measurement in a Large, Unexplored Space

**The assumption or constraint:** The continued pretraining experiment in Section 4.2 demonstrates that olmOCR-extracted peS2o tokens produce better downstream benchmark performance than Grobid-extracted peS2o tokens (+1.3 percentage point average improvement). The paper presents this as evidence that improved PDF extraction quality translates to better language model training outcomes. However, the experimental design explores exactly one point in a high-dimensional space: a single model scale (7B parameters), a single continued training budget (50B tokens), a single domain (academic papers), a single base model checkpoint (OLMo-2-7B-1124), and a single comparison (olmOCR vs. Grobid + rules). The paper does not claim these results generalize, but it also does not discuss the limits of their generalizability.

**The consequence:** A practitioner deciding whether to invest in reprocessing their PDF corpus with olmOCR has no way to estimate whether the +1.3 point average improvement (or the striking +5.2 point HellaSwag gain) would replicate in their setting. Several plausible failure modes for generalization exist:

- **Model scale**: The 7B parameter scale is relatively small by current standards. At larger scales (70B, 405B), the model may be more robust to extraction noise, meaning the gap between olmOCR-extracted and Grobid-extracted data could shrink or disappear. Alternatively, larger models may be *better* able to exploit cleaner data, making the gap larger. The paper provides no evidence either way.

- **Training budget**: The 50B token continued training budget is modest. Over a full pretraining run of trillions of tokens—where the model sees vastly more diverse data—the extraction quality signal might be "washed out" by other data sources. The continued pretraining design (starting from a partially-trained checkpoint) means the experiment measures the marginal benefit of higher-quality academic paper tokens when added to a model that has already seen substantial general-domain pretraining data. In a from-scratch pretraining run where academic papers are a small fraction of the total data mix, the extraction quality effect might be undetectable.

- **Domain**: peS2o consists exclusively of academic papers. The extraction challenges for academic papers (math formulas, references, multi-column layouts) are different from those for legal documents, government reports, or scanned books. The +5.2 point HellaSwag gain—the largest single-benchmark improvement—may be specific to the interaction between academic paper structure and commonsense reasoning evaluation, a connection the paper does not explore.

- **Baseline**: The comparison is against Grobid + rules. The paper does not test whether olmOCR-peS2o outperforms versions of peS2o processed with other modern tools (MinerU, Marker, or even GPT-4o). This means the result establishes that olmOCR is better than Grobid for this specific purpose, but does not establish that it is better than all alternatives, or that the gap between olmOCR and other tools on the benchmark translates to a proportional gap in downstream LM utility.

**What evidence exists in the paper:** Table 5 reports point estimates for six benchmark scores with no confidence intervals, no error bars, and no multiple-training-seed variance estimates. The +5.2 point HellaSwag improvement is the most eye-catching result, but without statistical characterization, we cannot assess whether it is reliably above the noise floor of continued pretraining. HellaSwag is known to have relatively high variance depending on evaluation protocol details (number of few-shot examples, exact prompt format), and the paper does not specify the evaluation configuration used. The fact that four of the six benchmarks show changes within ±0.3 points (MMLU: 0.0, NaturalQuestions: -0.3, WinoGrande: -0.3, and to a lesser extent ARCC: +1.4) suggests that for many tasks, the extraction quality difference has negligible impact at this training scale, with HellaSwag as a notable outlier. The paper does not discuss why HellaSwag specifically benefits, leaving the mechanism unexplained.

**Mitigation status:** The paper frames the experiment modestly as a demonstration that extraction quality *can* matter, not as a comprehensive characterization of when and how much it matters. Section 4.2 states the design as an "ablation procedure" that "has been used to assess data quality" in prior work, citing Blakeney et al. (2024), Grattafiori et al. (2024), and OLMo et al. (2024). This framing is appropriate—the experiment is a proof of concept, not a scaling study. The paper does not claim generalizability beyond the tested regime, and the transparency about the experimental design (same documents, same model, same budget, different extraction) allows readers to assess the strength of the causal claim. However, the paper also does not explicitly flag the limitations of this single-point measurement or discuss the regimes where the result might not hold, leaving practitioners to infer the generalizability boundaries themselves.

---

### 6.4 Performance Degrades Significantly on Specific Document Types, with No Clear Path to Improvement

**The assumption or constraint:** The paper presents olmOCR as a general-purpose PDF extraction toolkit, and the overall benchmark score of 75.5% supports this framing. However, the per-category breakdown in Table 4 reveals that performance is highly uneven across document types, with olmOCR substantially underperforming specialized or larger systems on several important categories. The paper acknowledges these results implicitly (they are reported in the table) but does not analyze the failure modes, diagnose the causes, or propose mitigation strategies for the weak categories.

**The consequence:** A practitioner whose document distribution is skewed toward olmOCR's weak categories would see substantially worse extraction quality than the 75.5% headline number suggests. The specific weak points are:

- **Long Tiny Text (LTT): 73.3%** — a full 11.3 percentage points behind Marker (84.6%) and 11.1 points behind Gemini Flash 2 (84.4%). These are documents with dense, small-print content like dictionary pages and reference lists. The paper's training configuration—rendering pages at a maximum 1024 pixels on the longest edge—is the likely cause: at this resolution, very small text becomes illegible, and the anchored text hints may be scrambled or truncated when many text blocks compete for the 6,000-character limit. This is a resolution ceiling that fine-tuning cannot overcome; the model simply cannot see the text it needs to extract.

- **Headers Footers (HF): 94.5%** — while numerically high, this is behind MinerU's 96.6%, suggesting that dedicated heuristic-based approaches outperform the end-to-end VLM approach for the specific task of identifying and stripping peripheral content. The 5.5% failure rate means roughly 1 in 18 pages includes some header or footer content that should have been removed—page numbers, running titles, or publication metadata leaking into the extracted text.

- **Old Scans (OS): 42.2%** — the lowest absolute performance across all categories and all systems. Even the best-performing system (olmOCR) fails on more than half of old scan test cases. This category includes historical letters and typewritten documents where image quality is poor, fonts are unusual, and layout conventions differ from modern documents. The paper's training data includes only 6.8% books (Table 2), and even fewer old scanned documents specifically. The model simply has limited exposure to the visual characteristics of historical documents.

- **Tables (TA): 71.0%** — essentially tied with Gemini Flash 2 (72.1%) but notably not dominant, and the paper notes that "many cases depend on rowspan and colspan information being preserved, which is possible only in HTML based tables" (Section 3.1). Since olmOCR's training data uses Markdown table formatting (Appendix E.1), it cannot represent merged cells, putting a hard ceiling on table extraction fidelity for complex tables.

**What evidence exists in the paper:** The per-category results in Table 4 provide the quantitative evidence. The qualitative examples in Appendix G further illustrate the degradation on specific document types. The paper notes the resolution constraint indirectly: during training, "PDF pages are rendered to a maximum dimension of 1024 pixels on the longest edge" (Section 2.3), and this same resolution is used at inference (Appendix D.2). For the Long Tiny Text category—which specifically targets "dense, small print on a single page" (Section 3.2)—1024 pixels is almost certainly insufficient to resolve individual characters, but the paper does not make this connection explicit. The Tables limitation is acknowledged in the benchmark design discussion (Section 3.1) but not in the analysis of olmOCR's own performance.

**Mitigation status:** Not addressed. The paper does not propose higher-resolution inference for small-text documents, fallback strategies for complex tables, or specialized post-processing for header/footer removal. The inference pipeline's exponential backoff on anchored text length (Appendix D.2) and fallback to "plain text-based PDF extraction" on repeated failure are general-purpose robustness mechanisms not targeted at specific failure modes. The paper does not suggest future work on resolution-adaptive processing or category-specific model variants. The 1024-pixel rendering limit is presented as a training efficiency choice without discussion of its quality implications. For a practitioner whose use case emphasizes any of these weak categories, the paper provides performance numbers but no guidance on how to improve them beyond the default configuration.

---

### 6.5 The 12% Retry Rate Masks Potentially Serious Degenerate Behavior That Compounds at Scale

**The assumption or constraint:** The paper reports a 12% retry rate for olmOCR during inference (Table 6 caption), meaning approximately one in eight pages fails on the first attempt and must be regenerated. The paper describes the primary failure mode:

> "the most common failure we experience is outputs degenerating into endless repetitions of the same token, line, or paragraph" (Appendix D.2)

and notes that "letting generations repeat up to maximum sequence length uses significant memory within SGLang" (Appendix D.2). The retry logic re-generates the page (up to N times, with N unspecified) and falls back to pypdf plain text extraction if all retries fail.

**The consequence:** The 12% figure is a measured average across the entire test set, but the paper does not report whether the retry rate varies by document type. If retries are concentrated in specific document categories—as seems likely, since the conditions that trigger repetition (ambiguous visual input, complex layouts, unusual text patterns) are correlated with document difficulty—then the effective failure rate for challenging documents could be substantially higher than 12%. More importantly, pages that fail all retries and fall back to pypdf extraction produce output of unknown quality: the fallback text comes from the same "highly noisy" pypdf extraction that the paper describes as having scrambled reading order and interwoven boilerplate (Section 2.2). For a large-scale batch processing job, a 12% overall retry rate means roughly 120,000 pages per million would be retried, with some fraction of those falling back to degraded extraction. If retries are concentrated in the most valuable content (e.g., tables, formulas, dense reference pages), the practical impact on data quality could be disproportionate.

The paper also notes that the repetition degeneration "uses significant memory within SGLang" (Appendix D.2), indicating that the current retry mechanism is not just a throughput cost but a resource cost—degenerate generations that run to the maximum sequence length consume GPU memory and compute for no useful output. The paper proposes future work to "detect repeated generations sooner than at the maximum context length limit, and abort promptly," acknowledging this as a known inefficiency, but the current system does not implement early termination. At million-page scale, the wasted compute from undetected repetitions could be substantial even if the retry rate is "only" 12%.

**What evidence exists in the paper:** The 12% figure is reported in the Table 6 caption without decomposition by failure type or document category. The repetition failure mode is described qualitatively in Appendix D.2. The paper's temperature sweep (Appendix C.1, Table 7) shows that increasing temperature from τ = 0.1 to τ = 0.8 reduces teacher alignment (0.875 → 0.859) but is preferred for inference because it "reduces the likelihood of repetitions occurring" (Appendix D.2). This reveals a deliberate tradeoff—accept slightly less faithful extraction on average to avoid catastrophic repetition failures—but the paper does not quantify how much the temperature increase reduces the retry rate. The retry rate at τ = 0.1 is not reported, so the magnitude of the improvement is unknown.

**Mitigation status:** Partial. The paper implements retries with multiple attempts and a fallback extraction pathway, which prevents total data loss. The temperature increase is a proactive mitigation for repetitions, though its effectiveness is unquantified. The paper explicitly acknowledges the limitation of the current repetition handling:

> "if retries occur often, the total generation throughput could be significantly reduced. Further, letting generations repeat up to maximum sequence length uses significant memory within SGLang. In future work, we plan to detect repeated generations sooner than at the maximum context length limit, and abort promptly." (Appendix D.2)

This is an honest acknowledgment of a real operational problem, but it means the current system ships with a known inefficiency that will be fixed later. For practitioners deploying the current version at scale, this means accepting wasted compute from undetected repetitions and throughput variability depending on document characteristics that influence the retry rate.

---

### 6.6 The System Is Optimized for English-Language Born-Digital PDFs and Provides No Characterization of Performance on Other Languages or Purely Scanned Documents

**The assumption or constraint:** The training data olmOCR-mix-0225 is filtered to English-language documents only, using the Lingua language detection package (Section 2.1). The Internet Archive component adds some scanned documents to the training mix, but these are a small fraction (5.7% of documents, 6.8% of pages) and are exclusively public domain books in English. The benchmark olmOCR-Bench includes Old Scans and Old Scans Math categories (combined 694 tests across 134 PDFs) that test scanned document performance, but these are also English-language documents sourced from the Internet Archive and Library of Congress. The baseline tests in the benchmark explicitly penalize output containing "characters from the Chinese, Japanese, or Emoji Unicode charsets" (Section 3.1), meaning the evaluation framework itself is designed to detect and penalize non-English output rather than assess multilingual extraction quality.

**The consequence:** The paper's performance claims are valid only for English-language PDFs, and primarily for born-digital documents where document-anchoring provides useful text hints. For non-English documents, several failure modes are possible and uncharacterized:

- **Language switching**: The model might correctly extract text but output it in English translation rather than the original language. The baseline test that penalizes Chinese/Japanese/Emoji characters would catch some of this, but would not detect English output for a German or French document (which would pass the baseline test but be incorrect).

- **Script handling**: The paper does not test performance on non-Latin scripts (Arabic, Cyrillic, Devanagari, Chinese, Japanese, Korean). The Qwen2-VL base model has some multilingual capability (it was trained on multilingual data), but the fine-tuning on English-only silver data may have suppressed or degraded this capability. The benchmark's active penalization of Chinese and Japanese characters suggests the authors observed language switching as a failure mode and chose to suppress it rather than characterize multilingual performance.

- **Scanned document degradation**: The 42.2% pass rate on Old Scans (Section 3.2, Table 4) is the worst category performance for olmOCR. This category represents scanned English-language documents with high-quality existing transcriptions. For scanned documents in other languages, or scans of lower quality than the Library of Congress collections, performance is likely substantially worse. The paper's document-anchoring mechanism provides no benefit for scanned documents (there is no digital text to extract), so the model relies purely on visual processing, which the Old Scans results show is significantly less reliable than anchored extraction.

**What evidence exists in the paper:** The language filtering is explicitly described in Section 2.1: "Using the Lingua package, we identify and filter out non-English documents." The benchmark's language-related baseline test is described in Section 3.1 as checking that output "does not contain any characters from the Chinese, Japanese, or Emoji Unicode charsets," with a note that "the handful of pages which do legitimately contain such charsets are manually flagged and excluded from such test conditions." The Old Scans per-category result (42.2% in Table 4) provides the only quantitative evidence on scanned document performance. There is no evaluation of any non-English language, no discussion of multilingual capability, and no characterization of how performance degrades as documents diverge from the born-digital English training distribution.

**Mitigation status:** Not attempted. The paper does not claim multilingual capability or discuss language as a limitation. The English-only scope is a deliberate design choice—the paper's goal is to unlock English-language PDF content for English-language LM training, and the peS2o downstream experiment (Section 4.2) is explicitly within that scope. This is legitimate scoping for a research contribution, but the omission becomes a practical limitation for anyone considering deploying olmOCR on multilingual document collections. The paper does not suggest future work on multilingual expansion, leaving this as an unaddressed capability gap. For organizations processing documents in multiple languages (common in international legal, academic, and governmental contexts), the paper provides no guidance on whether olmOCR can be used as-is, whether separate models would need to be trained per language, or whether the document-anchoring approach transfers across scripts.

## 7. Implications and Future Directions
- Field impact
  - Demonstrates that a small, task-specialized open VLM, when grounded by document-anchoring, can outperform larger general models on PDF linearization while being orders of magnitude cheaper (Figure 1; Table 4, Table 6). This rebalances the ecosystem away from closed APIs and toward reproducible, large-scale open pipelines.

- Practical applications
  - Massive corpus construction and cleaning for LM pretraining (e.g., reprocessing peS2o, as shown in §4.2; Table 5).  
  - Enterprise document processing at scale (legal, financial, government archives) where data cannot leave controlled infrastructure.  
  - Research tools for scientific PDFs, enabling better retrieval-augmented generation and reading assistance by providing faithful reading-order text.

- Follow-up research
  - Multilingual and domain expansion: include non-English PDFs and specialized domains (e.g., forms, code-heavy docs).  
  - Richer outputs: consistent HTML with reliable table row/colspans, figure references, cross-page footnotes, and improved math LaTeX fidelity.  
  - Better decoding safety: early detection/termination of repetitive generations (Appendix D.2 notes planned work).  
  - Teacher diversification: distill from ensembles (multiple VLMs) to reduce bias from a single teacher’s errors.  
  - Stronger benchmarks: broaden unit tests to cover cross-page references, figure-text alignment, and document-level tasks beyond single-page checks.

Overall, this work delivers a practical, open, and rigorously evaluated pathway to unlock high-quality text from PDFs at scale. The combination of document-anchoring, targeted fine-tuning, and unit-test evaluation provides both a methodological template and an operational toolchain that downstream LM researchers and practitioners can adopt immediately.
