# The Common Pile v0.1: An 8TB Dataset of Public Domain and Openly Licensed Text

**ArXiv:** [2506.05209](https://arxiv.org/abs/2506.05209)

## 🎯 Pitch

The Common Pile v0.1 presents an unprecedented 8-terabyte dataset comprising only public domain and openly licensed texts from 30 diverse sources, plus a rigorously curated data mixture and openly released LLMs trained on 1 and 2 trillion tokens. This work proves, for the first time at scale, that state-of-the-art language models can achieve competitive performance with models trained on unlicensed web data—solving a major legal and ethical challenge and unlocking a reproducible, transparent path forward for open, auditable, and equitable LLM research and development.

---

## 1. Executive Summary

This paper introduces and releases **the Common Pile v0.1**, an 8TB collection of public-domain and openly licensed text sourced from 30 diverse domains—including research papers, code, government documents, and educational materials—designed explicitly for LLM pretraining under legally and ethically permissive terms. To validate this resource, the authors train two 7B-parameter models—Comma v0.1-1T and Comma v0.1-2T—on filtered and rebalanced subsets of the Common Pile, demonstrating competitive performance against compute-matched models trained on unlicensed data (Llama 1 and 2 7B, MPT-7B, and OLMo Twin 7B), with particularly strong results on knowledge benchmarks like MMLU and coding tasks like HumanEval. The paper establishes that training performant LLMs exclusively on openly licensed text is feasible, with the Common Pile yielding models that consistently outperform prior open-license corpora (OLC, Common Corpus, KL3M) across all evaluated benchmarks—though performance gaps on commonsense reasoning tasks like HellaSwag persist, establishing a boundary where the current openly licensed data distribution still lags behind unlicensed web-scraped alternatives.

## 2. Context and Motivation

### The Core Problem: The Ethical and Legal Tension Between LLM Pretraining and Data Sourcing

The fundamental tension this paper confronts is straightforward but extraordinarily consequential: **the dominant paradigm for building performant large language models relies on vast quantities of unlicensed, web-scraped text, a practice that is legally contested and ethically problematic.** The paper frames this as a growing divide between LLM developers and content creators, with the core question being whether it is possible to train competitive models using *only* text that is unambiguously in the public domain or distributed under open licenses that explicitly permit reuse and modification for any purpose.

This problem has multiple dimensions that make it urgent and consequential:

**The legal dimension: A wave of lawsuits and financial exposure.** The paper documents (Section 1) that rights holders have objected to the uncompensated use of their work in LLM pretraining, "resulting in numerous lawsuits against LLM developers that could carry financial damages in the billions." At the time of writing, the authors note that compensating content creators for pre-training data, "even at conservatively low wage rates, would cost billions of US dollars." This is not a theoretical concern — the DMCA takedown of the original Pile dataset is cited as concrete evidence that relying on unlicensed data exposes dataset distributors and model trainers to direct legal action. The practical implication is that organizations building on openly released pre-training datasets face real risks of having those resources forcibly removed.

**The ethical dimension: Consent and creator rights.** Beyond legal liability, the paper argues that web-scraped pre-training data raises fundamental ethical concerns because "content creators rarely explicitly consent to the downstream use of their work for LLM training." This concern is amplified by recent behavioral evidence: the paper cites a "sharp mid-2023 increase in websites blocking AI crawlers, following growing awareness of web data being used to train models" — referencing work by Longpre et al. (2024) showing that many content owners, when made aware of this practice, actively decline consent through technical measures. This creates a situation where the default approach to dataset construction actively overrides expressed creator preferences.

**The transparency and reproducibility dimension.** The paper identifies a third, perhaps less obvious problem: the use of unlicensed training data "heavily limits the ability of model trainers to share their datasets." This restricts research into learning dynamics, memorization, data auditing, and other important areas that depend on publicly accessible training data. The prior Pile dataset takedown is again instructive — it demonstrated that even well-established, widely cited benchmark corpora built from unlicensed text cannot be reliably shared with the research community.

### The Gap: No Prior Dataset Was Both Large Enough and Licensable Enough to Train Competitive LLMs

The paper's starting premise is that while prior efforts have attempted to curate openly licensed or public domain text corpora for LLM pretraining, none has simultaneously satisfied two essential requirements: **sufficient scale** to train models that approach the performance of those trained on unlicensed web data, and **sufficiently rigorous license curation** to actually solve the legal and ethical problems.

**Scale limitations.** The paper provides specific comparisons in Section 2.2. Prior datasets built with similar permissive-license principles are simply too small:
- The **Open License Corpus (OLC)** comprises only 0.85 TB of text from 12 sources (compared to the Common Pile's 7.6 TB from 30 sources), making it insufficient for training models that need trillions of tokens.
- **KL3M**, which takes an even more conservative approach by excluding CC BY-SA content, is limited to ~3 TB and "almost exclusively consists of government documents" — the narrow domain coverage inevitably limits downstream model capabilities.

**License quality and coverage tradeoffs.** Other datasets have volume but compromise on licensing rigor or domain range:
- **Common Corpus** is comparable in total size to the Common Pile (~7.4 TB) but "targets a broader set of languages and therefore contains significantly less English text." More critically, Common Corpus "does not retain full per-document licensing information across all sources" and includes data from OpenAlex, which the paper explicitly identifies as having inaccurate licensing metadata. This means that Common Corpus's license claims are not fully auditable — a user cannot trace the exact license status of individual documents.
- Previous large-scale corpora that use **collection-level licenses** (like the ODC-By license used by Dolma, FineWeb, and TxT360) do not solve the underlying problem because the license "by definition, does not extend to individual documents within the corpus; therefore, the copyright of documents in these collections is still controlled by the document authors." Training on ODC-By-licensed corpora is legally distinct from training on corpora where every constituent document is individually permissively licensed.

**The fundamental tension: Web coverage vs. license compliance.** The paper implicitly identifies a structural problem: the vast majority of web text is *not* openly licensed, and the subset that *is* openly licensed is harder to find, harder to verify, and distributed across many smaller, domain-specific sources rather than a few large web dumps. Prior approaches either accept the licensing risk of broad web scraping, or accept the limited coverage of careful license filtering. The Common Pile aims to demonstrate that a third path — assembling diverse openly licensed sources at sufficient scale — is viable.

### Why the Pile Matters as a Reference Point

The paper positions itself explicitly in relation to the original Pile dataset (Gao et al., 2020). The Pile demonstrated that a curated mixture of diverse, domain-specific text sources (22 sources) could produce stronger models than undifferentiated web data. However, the Pile relied on unlicensed content (academic papers behind paywalls, copyrighted books and forums). The Common Pile adopts the *curation philosophy* of the Pile — diverse sources, explicit source documentation, mixture design — while replacing the *licensing basis* entirely with permissively licensed alternatives. This is a deliberate methodological inheritance: the form of the dataset (diverse, curated, documented) is preserved and extended, but the content sourcing is fundamentally re-engineered.

### How This Paper Positions Itself

The paper does not position itself as making architectural or algorithmic innovations in language model training. Rather, it frames its contribution as **infrastructure work**: proving that a dataset satisfying strict legal and ethical criteria can exist, and that models trained on this dataset are competitive — not state-of-the-art — with equivalently-sized models trained on the standard unlicensed data paradigm.

The key strategic claim is that the Common Pile represents "the first step towards a more ethical language model ecosystem, where performance need not come at the cost of creator rights and legal transparency" (Section 5). This is explicitly forward-looking: the paper is not claiming that the Common Pile currently produces models that match the best available models (which are trained on orders of magnitude more data), but rather that it establishes the *feasibility* of the approach and provides a foundation that could be scaled further as more openly licensed text becomes available.

The paper also positions itself in the lineage of prior work that has attempted to solve pieces of this puzzle: the C4Corpus tools for extracting Creative Commons text from Common Crawl snapshots, CommonCanvas for images, the PG19 dataset for public domain books, and the datasets discussed in Section 2.2. The Common Pile is framed as the largest-scale integration of these prior insights — combining domain-specific data sources, rigorous license vetting, web-based Creative Commons extraction, and curated task data — into a single coherent training corpus.

### The Implicit Challenge: Ongoing "Consent in Crisis"

The paper references Longpre et al.'s (2024) finding of "consent in crisis" — the rapid decline in websites permitting AI crawlers. This points to a forward-looking motivation that goes beyond the current legal landscape: as awareness of LLM training practices grows, the pool of easily accessible unlicensed web text may shrink due to both technical blocking and potential regulatory restrictions. Building the infrastructure for openly licensed pre-training datasets is therefore not just an ethical choice for the present, but a strategic hedge against a future where unlicensed web scraping becomes less viable. The Common Pile v0.1 is presented as an initial step on this path, with the paper explicitly noting that approximately half of the Common Pile's content "was created since 2020" (Appendix I, Figure 6), suggesting that the supply of openly licensed text is actively growing.

## 3. Technical Approach

### 3.1 Reader Orientation

This paper is fundamentally a **dataset infrastructure paper**: it describes the construction of a large, diverse text corpus—the Common Pile v0.1—from exclusively public-domain and openly licensed sources, then validates that corpus by training language models on it and measuring their performance against models trained on standard (unlicensed) datasets. The core problem it solves is a sourcing one: how to assemble enough high-quality, legally unencumbered text across enough diverse domains to train a competitive large language model, when the vast majority of web text is copyrighted and cannot be used without permission. The shape of the solution is a multi-stage pipeline that (1) identifies and collects text from 30 individual sources with verified permissive licensing, (2) applies source-specific filtering to remove low-quality or toxic content, (3) deduplicates globally to reduce memorization and redundancy, (4) heuristically reweights sources into a training mixture based on small-scale quality experiments, and (5) trains models on the resulting mixture using standard architectures to validate the corpus.

### 3.2 Big-Picture Architecture (Diagram in Words)

The system has five major stages, plus a validation output:

1. **Source Identification and Collection** — For each of 30 sources spanning scientific papers, government documents, code, books, wikis, web text, forums, audio transcripts, educational materials, and curated task datasets, the authors identify whether the content is in the public domain or distributed under a license meeting the Open Knowledge Foundation's Open Definition 2.1. They then download, extract, and convert each source into plain text using source-specific tools (LaTeX converters, OCR pipelines, speech recognition, HTML extractors, etc.).

2. **Per-Source Filtering and Cleaning** — Each source undergoes an independent preprocessing pipeline using the Dolma toolkit, applying language identification (FastText classifier), text quality filtering (DataComp-LM classifier for web text), document length thresholds, likelihood-based OCR error detection (unigram language model from the Trillion Word Corpus), toxicity classification (Jigsaw-trained FastText classifiers), PII redaction (regex for emails, phones, IPs), and source-specific boilerplate removal (regex for page numbers, license statements, document preambles). Separate code filtering is applied to Stack V2 using RedPajama V1 heuristics and language-specific quality classifiers following SmolLM2.

3. **Global Fuzzy Deduplication** — After per-source filtering, all documents across all sources undergo document-level fuzzy deduplication using bloom filters from Dolma's deduplication module. Two documents are considered duplicates if they share more than 90% of their 20-grams. This step prevents data repetition from inflating effective corpus size and reducing memorization risk.

4. **Data Mixing via Heuristic Reweighting** — Rather than combining sources in proportion to their raw size, the authors first train small (1.7B parameter) per-source language models for 28 billion tokens each, then use the resulting benchmark performance to heuristically up-weight high-performing sources and down-weight low-performing ones. A target maximum of six repetitions over a 1 trillion token training run constrains the up-weighting. Small sources assumed to be high-quality are assigned a mixing rate such that they are also repeated six times. The resulting mixture (the "Comma dataset") is documented in Table 7.

5. **Model Training and Validation** — Using the lingua framework, two 7B-parameter Llama-architecture models (Comma v0.1-1T and Comma v0.1-2T) are trained on 1T and 2T tokens respectively of the Comma mixture. Each uses a two-stage training process (main training with cosine schedule + cool-down phase on high-quality sources with linear decay), with hyperparameters specified in Section 4.4. The resulting models are evaluated on standard reasoning, knowledge, and coding benchmarks to validate that the Common Pile produces competitive performance against models trained on unlicensed data (Llama 1/2 7B, MPT-7B, OLMo Twin 7B, etc.).

### 3.3 Roadmap for the Deep Dive

- **First**, the licensing framework (Section 2 and Appendix C): what "openly licensed" means, what licenses are included and excluded, and the due diligence process to avoid license laundering. This is foundational because the entire dataset's value proposition depends on rigorous license compliance.
- **Second**, the source collection methodology for each major domain category (Appendix B): the specific tools, data formats, conversion pipelines, and filtering heuristics used for each of the 30 sources. Understanding the diversity of these sources and the domain-specific processing is essential to understanding the corpus composition.
- **Third**, the per-source filtering pipeline (Section 4.1): the sequence of Dolma-based filters applied to cleaned text, including language ID, quality classification, likelihood scoring, toxicity filtering, and PII redaction, with source-specific thresholds documented in Table 5. This stage determines what text actually enters the training corpus.
- **Fourth**, the deduplication and data mixing methodology (Section 4.2): how the 20-gram overlap deduplication works, and the heuristic source weighting procedure based on 1.7B parameter per-source validation models, including why MixMin was attempted but abandoned.
- **Fifth**, the tokenizer and training configuration (Section 4.4): the custom BPE tokenizer trained on the Comma dataset to ensure end-to-end openly licensed provenance, and the hyperparameters, learning rate schedules, and cool-down strategies for Comma v0.1-1T and -2T.
- **Sixth**, the controlled data quality experiments and evaluation methodology (Sections 4.3 and 4.4): the 1.7B parameter ablation setup that allows comparison across datasets without confounding hyperparameter choices, and the evaluation benchmarks used for Comma v0.1, including why Winogrande was excluded.

### 3.4 Detailed, Sentence-Based Technical Breakdown

This is primarily a **dataset construction and validation paper**. Its core idea is that by carefully sourcing from diverse, domain-specific repositories of openly licensed text—applying rigorous license verification, source-appropriate filtering, and heuristic data mixing—one can assemble a pre-training corpus that produces competitive large language models without relying on unlicensed web-scraped text.

---

#### Licensing Framework and Due Diligence

The Common Pile v0.1's licensing framework is the foundation upon which the entire dataset's value proposition rests. Without it, the dataset would be just another corpus of text; with it, the dataset is positioned as a solution to the legal and ethical problems plaguing LLM pretraining.

**The formal definition.** The paper adopts the **Open Knowledge Foundation's Open Definition 2.1**, which stipulates that works must be "freely used, shared, and built upon by anyone, for any purpose" (Section 2). In practice, this means the dataset includes content under the following license categories:
- **CC0** (public domain dedication)
- **CC BY** (attribution required)
- **CC BY-SA** (attribution + share-alike)
- **Software licenses certified by the Blue Oak Council**, which include MIT, BSD, Apache 2.0, and similar permissive open-source licenses
- **Government works in the public domain**, such as U.S. federal government publications (under 17 U.S. Code § 105) and UK Parliamentary proceedings under the Open Parliament License
- **Copyright-expired works** (e.g., U.S. books published before 1929)

**Excluded licenses.** Critically, the paper explicitly excludes **CC NC (non-commercial)** and **CC ND (no derivatives)** licenses, as these do not meet the Open Definition's requirement that content be usable "for any purpose." This is a stricter standard than some prior datasets, and it means that significant quantities of text distributed under these common-but-restrictive licenses are excluded from the Common Pile.

**The license laundering problem.** The paper identifies license laundering—where a copyrighted work is redistributed by a non-rights-holder with an incorrect license—as a fundamental threat to dataset integrity. The authors address this through several mechanisms:

First, they impose **strict sourcing standards**: data is only included from sources where the authors were "confident that the licensing information was provided by the copyright holder" (Section 2.1). This leads to the explicit exclusion of sources with unreliable or ambiguous licensing metadata, such as OpenAlex (which is "known to provide inaccurate licensing information"), YouTube Commons (where uploaders frequently mislabel content), and the Hacker News dataset on Kaggle (which lacks a clear open license).

Second, for web-sourced Creative Commons text (the CCCC subset), the authors perform **manual verification** of the top 1,000 domains by content volume after automated regex matching for CC license markers. From these 1,000 domains, only 537 were retained—representing a 46.3% rejection rate—because the manual review confirmed that the Creative Commons designation applied to "all text content rather than only embedded media or a subset of the text on the domain." This is a labor-intensive step that reflects the reality that many websites display CC badges for images or other embedded content while the page's main text remains unlicensed.

Third, the paper explicitly excludes **LLM-generated synthetic datasets** that have been released under open licenses, taking the conservative position that "it has not yet been established whether it is permissible to apply arbitrary licenses to the generations of an LLM that was trained on unlicensed data" (Section 2.1). This excludes potentially large sources of text (like WildChat or similar synthetic instruction datasets) but maintains the strict provenance of all content in the Common Pile.

**Collection licenses vs. document licenses.** The paper draws an important distinction between **collection-level licenses** (applied to an entire corpus as a compilation) and **document-level licenses** (applied to individual constituent texts). Large-scale web corpora like Dolma, FineWeb, and TxT360 are often distributed under licenses like ODC-By, but "ODC-By, by definition, does not extend to individual documents within the corpus; therefore, the copyright of documents in these collections is still controlled by the document authors" (Section 2.1). The Common Pile requires that individual documents themselves carry permissive licenses, not just the collection. This is why the CCCC subset requires manual domain-level verification: the presence of a CC marker on a page does not necessarily mean the page's own text is CC-licensed; it could be a page about Creative Commons licensing that includes a CC badge as attribution for an embedded image.

**Caveats and limitations acknowledged.** The paper is transparent about the remaining risks: "License laundering is a notoriously hard problem to identify exhaustively in practice," copyright owners can change licenses after data collection, public domain documents may contain quoted in-copyright material, and attribution requirements that are straightforward for data redistribution are "an active area of research" for model predictions. These caveats do not undermine the paper's core claim—that this is a "substantial first step"—but they establish that perfect license compliance at web scale remains an unsolved problem.

---

#### Source Collection Methodology by Domain

The Common Pile draws from 30 distinct sources grouped into 10 broad domains. Each source requires a different collection strategy, format conversion pipeline, and set of processing decisions. What follows is a detailed walkthrough of each domain's collection methodology, organized by the paper's categorization in Section 3.

##### Scientific and Scholarly Text

**peS2o (peer-reviewed open-access scientific papers).** This is the single largest text source in the Common Pile at 182.6 GB after filtering, comprising 27.4% of the Comma v0.1 pre-training mixture. The source starts from peS2o v3, which is itself derived from the Semantic Scholar Open Research Corpus (S2ORC). The construction pipeline works as follows:

1. S2ORC provides structured XML output from Grobid, a tool that parses PDFs of scientific papers and extracts structured fields (title, authors, sections, paragraphs, references, etc.).
2. peS2o applies its own filtering pipeline to remove papers that are too short, have incorrect metadata, are in languages other than English, or contain pervasive OCR errors. The OCR filtering uses both heuristic and model-based approaches.
3. From this filtered corpus, the Common Pile retains only papers with **CC BY, CC BY-SA, CC0, or public domain designations** as identified by Semantic Scholar's metadata APIs.
4. The final subset contains **6.3 million papers** totaling **35.7 billion whitespace-separated segments**. License distribution (from Appendix H, Table 3): 6,088,325 CC BY papers, 120,150 CC BY-SA papers, 36,373 CC0 papers, and 10,060 public domain papers.

The metadata provided by Semantic Scholar APIs also enables field-of-study categorization (Appendix H, Table 4), showing the subset covers 23 fields, with the largest representation in Medicine (2.4M papers), Biology (1.5M), and Environmental Science (993K).

**PubMed Central (PMC).** PMC is the NIH's open-access archive of biomedical and life sciences research. The collection method is:
1. Metadata from PMC is checked to identify articles where the publishing journal designated a **CC BY, CC BY-SA, or CC0 license**.
2. The full text of each article is stored as a single XML file by PMC; these are downloaded and converted to markdown using **pandoc**.
3. The result contributes to the 147.1 GB (filtered) of PubMed text in the Common Pile (3.7% of the pre-training mixture).

**ArXiv Papers (full text).** ArXiv contains over 2.4 million scholarly articles where authors select their own license at upload time. The collection pipeline is:
1. The **LaTeX source files** for openly licensed papers (CC BY, CC BY-SA, CC0) are downloaded from ArXiv's bulk-access S3 bucket.
2. **LaTeXML** (a LaTeX-to-HTML/XML converter) processes these source files into a single HTML document, handling mathematical notation, citations, and document structure.
3. **Trafilatura**, an HTML-processing library designed for web content extraction, converts the HTML to plain text by removing navigation elements, headers, and formatting artifacts while preserving the textual content.
4. The filtered output is 19.5 GB (2.9% of mixture).

**ArXiv Abstracts.** Separately, ArXiv's policy states that all metadata—including abstracts—for any paper submitted to ArXiv is distributed under the CC0 license, regardless of the full-text license. This means the abstracts of papers whose full text is excluded (due to restrictive licenses like CC NC-ND) are still available as CC0. The authors source these abstracts through ArXiv's API via the Open Archives Initiative Protocol for Metadata Harvesting (OAI-PMH) endpoint and reproduce them as-is, yielding 2.4 GB of text (0.36% of mixture).

##### Government and Legal Texts

**USGPO.** The U.S. Government Publishing Office disseminates official federal documents. The Common Pile includes all plain-text documents from the GovInfo.gov developer API—over **2.7 million documents** spanning the Federal Register, congressional hearing transcripts, budget reports, economic indicators, and other federal publications. After filtering, this contributes 36.1 GB (0.23% of mixture). The key legal basis: under 17 U.S. Code § 105, works authored by U.S. federal employees as part of their official duties are in the public domain.

**USPTO (U.S. Patent and Trademark Office).** Patent documents are the single largest individual source in the Common Pile, contributing **661.1 GB after filtering**—nearly twice as large as the next-largest source. The collection pipeline:
1. Data comes from the **Google Patents Public Data dataset**, which includes granted patents and published patent applications dating back to 1782.
2. Patent documents follow a highly standardized format with distinct sections (background, summary, detailed description, claims); the processing preserves this structure.
3. Mathematical expressions and equations within patent text are converted into LaTeX format to standardize representation.

Despite its enormous raw size, patent text is heavily down-weighted in the training mixture to only 4.1% because of its "substantially different wording, terminology, and repetition than typical natural language" (Section 4.2). The paper explicitly treats this as an example of why source size and source quality are poorly correlated.

**Caselaw Access Project and Court Listener.** Legal texts come from two complementary sources:
- The **Caselaw Access Project** provides nearly 40 million pages of U.S. federal and state court decisions and judicial opinions from the last 365 years, digitized from the Harvard Law Library and other sources.
- **Court Listener** contributes over 900,000 additional cases scraped from 479 courts.
- Only documents confirmed to be in the **public domain** are retained, yielding a combined 77.5 GB after filtering (1.9% of mixture).
- Additional post-processing corrects erroneous OCR results from the digitization process and fixes formatting and parsing artifacts.

**UK Hansard.** Hansard is the official record of UK parliamentary proceedings, sourced from ParlParse, covering Commons debates from 1918 forward and Lords proceedings from the 1999 reform. Additional content includes records from devolved legislatures (Scottish Parliament, Senedd in English and Welsh, Northern Ireland Assembly), London Mayor's Questions, and ministerial statements. All content is published under the **Open Parliament License**, which the paper states "stipulates similar terms to the CC BY license." Processing preserves complete parliamentary sessions as cohesive units. The filtered output is 9.6 GB (1.4% of mixture).

**Regulations.gov.** Operated by the U.S. General Services Administration, this platform hosts newly proposed federal rules and regulations along with public comments. The Common Pile includes all plain-text documents available through the bulk download interface, producing 5.1 GB after filtering (0.76% of mixture).

##### Online Discussions and Forums

**StackExchange.** StackExchange comprises numerous Q&A websites where user-provided content is distributed under CC BY-SA. The collection methodology involves several non-trivial steps due to changes in StackExchange's data distribution:
1. Since July 2024, StackExchange has stopped publishing standardized XML dumps to the Internet Archive. Instead, each individual StackExchange site provides custom export URLs to logged-in users.
2. The new export tool has coverage gaps: some questions present in older dumps and accessible on the site are missing from new exports. Defunct sites (like windowsphone.stackexchange.com) have inaccessible dumps entirely.
3. To address this, the authors combine two sources: **community-uploaded dumps from December 2024** on the Internet Archive, plus **missing questions extracted from the last official dumps from July 2024**.
4. Each document consists of: a question, its comments, its answers (ordered by vote count, with the "accepted answer" always first), and the comments on each answer.
5. **PyMarkdown** converts each comment from Markdown to plain text.
6. The filtered output is 89.7 GB (13.5% of mixture)—the second-largest component of the training mixture after peS2o.

**GitHub Archive.** Issues, pull requests, and comments on GitHub inherit the license of their associated repository. The collection pipeline:
1. The **GitHub Archive's public BigQuery table** of events is queried to extract all issue, pull request, and comment events since 2011, aggregated into threads (approximately 177 million threads across 19 million repositories).
2. Bot comments are filtered out. The table does not include "edit" events, so each comment's text is from its initial posting.
3. Repositories are filtered to retain only those with a **Blue Oak Council-approved license** (MIT, BSD, Apache 2.0, etc.), identified from three sources: the `public-data:github_repos` BigQuery Table, metadata from Stack V2, or the GitHub API. This license filtering reduces the set from 19 million to 10 million repositories.
4. **PyMarkdown** converts GitHub-flavored markdown to plain text; when parsing fails, raw markdown is kept.
5. The filtered output is 40.4 GB (6.1% of mixture).

**Ubuntu IRC.** Internet Relay Chat logs from Ubuntu's servers since 2004 have been released into the public domain. The collection approach:
1. All chats from all channels up to March 2025 are downloaded.
2. Each document represents all messages for a given channel on a given day.
3. System messages and messages from known bots are removed.
4. The filtered output is 5.3 GB (0.79% of mixture).

##### Books in the Public Domain

The paper collects books from four sources, all limited to works confirmed to be in the public domain in the United States (as of 2024, works published before 1929).

**Biodiversity Heritage Library (BHL).** An open-access digital library for biodiversity literature:
1. Bulk data download interface provides access to the full collection.
2. Books are filtered based on their associated license metadata to retain only public domain works.
3. The distributed text comes from **OCR (optical character recognition)** processing—not manually transcribed—so OCR quality is a concern that subsequent filtering stages address.
4. After filtering: 35.5 GB (0.22% of mixture).

**Pre-1929 Books (HathiTrust/Internet Archive).** A systematic identification of public domain books:
1. **Hathifiles**, the bibliographic catalog produced by HathiTrust, is used to identify digitized books published in the U.S. before 1929.
2. The collection contains over **130,000 books** digitized and processed by the Internet Archive on behalf of HathiTrust member libraries.
3. OCR plain text files are downloaded directly from the Internet Archive website.
4. After filtering: 46.3 GB (1.2% of mixture).

**Library of Congress.** The LoC's "Selected Digitized Books" collection:
1. Over **130,000 English-language books** are downloaded as OCR plain text files using the LoC APIs.
2. After filtering: 35.6 GB (0.22% of mixture).

**Project Gutenberg.** An online collection of over 75,000 digitized books:
1. Books are included if they are (a) English and (b) marked as public domain in the metadata.
2. Additionally, all books that are part of the **pg19 dataset** (which only includes books over 100 years old) are included.
3. Minimal preprocessing removes Project Gutenberg's standard headers and footers, though many scanned books retain preamble information about who digitized them.
4. After filtering: 20.1 GB (0.5% of mixture).

##### Open Educational Resources (OERs)

These four sources provide educational materials published under open licenses to support free access to education.

**Directory of Open Access Books (DOAB).** An index of over 94,000 peer-reviewed open-access books:
1. Metadata is retrieved from DOAB's official metadata feed.
2. Collection is filtered to English-language books under **CC BY or CC BY-SA** licenses.
3. Books are downloaded in PDF format and converted to plain text using the **Marker PDF-to-text converter**.
4. An additional validation step: the authors "manually create a whitelist of open license statements and retain only texts explicitly containing one of these statements in their front- or back-matter." This is a manual quality check against license laundering.
5. After filtering: 12 GB (1.8% of mixture).

**PressBooks.** A catalog of over 8,000 open-access books:
1. A search query retrieves URLs for all books in English listed as public domain or under CC BY or CC BY-SA licenses.
2. Content is collected from the publicly available web version of each book.
3. After filtering: 0.6 GB (0.09% of mixture).

**OERCommons.** A platform where educators share instructional materials (textbooks, lesson plans, problem sets, syllabi, worksheets):
1. A search query retrieves English-language content released into the public domain or under CC BY/CC BY-SA licenses.
2. Documents are converted to plain text directly from HTML pages hosted on the OERCommons website.
3. After filtering: 0.05 GB (0.008% of mixture).

**LibreTexts.** A catalog of over 3,000 open-access textbooks:
1. Links to all textbooks in the catalog are gathered.
2. Each textbook section is checked for a license statement indicating public domain, CC BY, CC BY-SA, or GNU Free Documentation License.
3. Plain text is extracted directly from HTML pages on the LibreTexts website.
4. After filtering: 3.6 GB (0.54% of mixture).

##### Wikis

Wikis are collaboratively maintained encyclopedic websites whose content is typically under open licenses.

**Wikimedia.** Official wikis managed by the Wikimedia Foundation:
1. Official database dumps from March 2025 of English-language wikis (listed in Appendix F: Wikipedia, Wikinews, Wikibooks, Wikiquote, Wikisource, Wikiversity, Wikivoyage, Wiktionary) are downloaded.
2. These dumps contain wikitext—MediaWiki's custom markup language—for each page, plus talk pages where editors discuss page changes.
3. Only the most recent version of each page is used.
4. Wikitext is converted to plain text using **wtf_wikipedia**, with light formatting adjustments to fix a known bug in section ordering.
5. Before parsing, wikitext math is converted into LaTeX math using custom code.
6. Any remaining HTML tags are removed via regex.
7. After filtering: 57.4 GB (8.6% of mixture).

**Wikiteam.** Unofficial dumps of non-Wikimedia wikis that use MediaWiki software:
1. All dumps made by wikiteam (a volunteer archiving group) are downloaded from the Internet Archive when metadata indicates a CC BY, CC BY-SA, or public domain license.
2. This yields approximately **330,000 wikis**.
3. When multiple dumps of the same wiki exist, the most recent is used.
4. Wikitext conversion follows the same pipeline as Wikimedia wikis.
5. After initial processing, wikis that "appeared to contain large amounts of license laundering"—for example, collections of song lyrics or transcripts—are removed.
6. After filtering: 13.7 GB (1.4% of mixture).

##### Source Code

**The Stack V2.** This is the largest source of code in the Common Pile and the largest single source overall by raw size (4,774.7 GB before filtering):
1. The Stack V2 contains a mixture of openly licensed and unlicensed code repositories.
2. The authors leverage license detection performed by the creators of Stack V2 (Software Heritage Foundation and BigCode).
3. When multiple licenses are detected in a single repository, the repository is retained only if **all detected licenses** meet the Common Pile's open-license definition. This is a stricter filtering criterion than requiring any single license to be open.
4. The Stack V2 receives special post-processing beyond the standard text pipeline, described in the "Code Filtering" portion of Section 4.1:
   - **RedPajama V1 code filtering heuristics** are applied first: filters based on mean and maximum line length, proportion of alphanumeric characters, and ratio of alphabetical characters to tokens.
   - Following the **SmolLM2** approach, code is further restricted to 15 languages: Python, C, C++, SQL, Java, PHP, Rust, Javascript, Typescript, Go, Ruby, Markdown, C#, Swift, and shell.
   - Language-specific quality classifiers (derived from SmolLM2) filter each language's code to retain only "educational and well-documented code."
   - A lower quality threshold is used than in SmolLM2, resulting in a larger post-filtered set.
   - HTML documents within Stack V2 are extracted using Trafilatura and then processed through the standard filtering pipeline (language, length, toxicity, PII).
5. After filtering: 259.9 GB (13.0% of mixture).

**Python Enhancement Proposals (PEPs).** Design documents for Python language features:
1. Of the 661 published PEPs, the majority are in the public domain; 5 published under the "Open Publication License" are omitted.
2. PEPs are authored in **ReStructured Text** and converted to plain text using **pandoc version 3.5**.
3. After filtering: 0.01 GB (0.002% of mixture).

##### Transcribed Audio Content

**Creative Commons YouTube.** This source required particularly intensive manual curation:
1. YouTube allows uploaders to mark content with CC BY. However, widespread license laundering (uploaders marking content they don't own as CC BY) makes automated collection unreliable.
2. To address this, the authors **manually curated a set of over 2,000 YouTube channels** that "consistently release original openly licensed content containing speech."
3. These channels span "lectures, tutorials, reviews, video essays, speeches, and vlogs."
4. From these channels, **over 1.1 million videos** comprising **over 470,000 hours** of content were retrieved.
5. Each video was transcribed using **Whisper**, the speech recognition model from Radford et al. (2022).
6. After filtering: 18.6 GB (0.47% of mixture).

The manual channel curation is a key design choice reflecting the difficulty of automated license verification for user-generated content platforms. It limits coverage but dramatically increases confidence in license compliance.

##### Web Text

**CCCC (Creative Commons Common Crawl).** This is the web-sourced text, representing the closest analog to standard web-scraped corpora but restricted to Creative Commons-licensed content:
1. Text is sourced from **52 Common Crawl snapshots**, covering about half of all snapshots available to date and spanning all years of Common Crawl operation up to 2024. The authors note that "a higher level of duplication across this collection, suggesting that including more snapshots would lead to a modest increase in total token yield."
2. HTML content is extracted from these snapshots using **FastWarc**.
3. A **regular expression adapted from the C4Corpus project** retains only pages containing a CC BY, CC BY-SA, or CC0 marker.
4. Critical quality step: the regex produces many false positives (e.g., pages that include and attribute a CC BY image but whose own text is unlicensed). To address this, the authors:
   - Identify the **top 1,000 domains by content volume** after regex matching.
   - **Manually verify** each domain's licensing to confirm that "the Creative Commons designation is applied to all text content rather than only embedded media or a subset of the text on the domain."
   - Retain only **537 domains** (46.3% rejection rate).
5. Text extraction uses **Resiliparse** to isolate main content and remove boilerplate (navigation, sidebars, footers).
6. **URL-level exact deduplication** is performed to remove identical pages across snapshots.
7. **Bloom filter-based near-duplicate detection** with 80% n-gram overlap removes near-duplicates.
8. Rule-based filters from Dolma are applied: C4-derived heuristics remove pages containing Javascript, Lorem Ipsum, and curly braces; all Gopher rules (from Rae et al., 2021) remove low-quality pages.
9. The filtered output: 58.1 GB after filtering (8.7% of mixture).
10. Appendix G (Table 2) provides per-snapshot statistics showing the CCCC subset contains approximately 259.7 million documents totaling 221.7 billion Unicode words before filtering (dramatically reduced by the aggressive filtering pipeline).

**Foodista.** A community-maintained recipe and food news site where all content is CC BY:
1. Plain text is extracted from HTML using a custom pipeline that includes extracting title and author information to include at the start of the text.
2. Comments on each page are appended to the article after filtering out automatically generated comments.
3. After filtering: 0.08 GB (0.012% of mixture).

**News.** News sites publishing under CC BY or CC BY-SA according to **Open Newswire**:
1. A full list of included sites is provided in Appendix E: 17 sites under CC BY (360info, Africa is a Country, Alt News, Balkan Diskurs, Factly, Freedom of the Press Foundation, Agenzia Fides, Global Voices, Meduza, Mekong Eye, Milwaukee Neighborhood News Service, Minority Africa, New Canadian Media, SciDev.Net, The Solutions Journalism Exchange, Tasnim News Agency, ZimFact) and 3 under CC BY-SA (Oxpeckers, Propastop, The Public Record).
2. Plain text is extracted using a custom pipeline, including extraction of title and byline for each article.
3. After filtering: 0.25 GB (0.038% of mixture).

**Public Domain Review.** An online journal about works in the public domain:
1. All articles published under CC BY-SA are collected.
2. After filtering: 0.007 GB (0.001% of mixture).

##### Curated Task Data (Data Provenance Initiative)

**Data Provenance Initiative (DPI).** This is a meta-source: a digital library of supervised NLP datasets whose licenses and provenance have been manually audited:
1. The authors leverage DPI's tooling to filter HuggingFace datasets based on multiple criteria:
   - Contains English language or code data.
   - Text is **not model-generated** (to avoid synthetic data provenance issues).
   - The dataset's audit yielded an open license.
   - The original sources of the data are only from "recognized public domain sources."
2. The full list of included datasets is provided in Appendix D (Table 1), spanning hundreds of individual supervised datasets from collections including CommitPackFT, DialogStudio, Flan Collection, Open Assistant, OIG, and Tasksource.
3. The inclusion of task data in a pre-training corpus is unusual and the paper explicitly addresses this in evaluating whether DPI confers an unfair advantage in the controlled experiments (Section 4.3): removing DPI-sourced data "had a minimal impact on model performance, with a notable decrease only on HellaSwag, possibly suggesting that the DPI data contains domain-relevant data for this benchmark that other sources lack."
4. After filtering: 3.4 GB (0.51% of mixture).

---

#### Per-Source Filtering Pipeline

The filtering pipeline (Section 4.1, detailed in Appendix J, Table 5) is designed to remove low-quality, toxic, non-English, or duplicate text before training. Critically, the pipeline is **independently configured per source**—each source has its own thresholds and sometimes its own set of applied filters—because the text characteristics vary dramatically across domains (patent text vs. IRC logs vs. academic papers).

**Language identification.** A **FastText classifier** (from Joulin et al., 2017) classifies the language of each document. Documents classified as non-English are filtered out. The language score threshold is applied at varying levels per source: for most sources, it is set to >0.5, but some sources (ArXiv Abstracts, Caselaw Access Project, Data Provenance Initiative, and several others) skip language filtering entirely (Table 5, column 1). This likely reflects that these sources are pre-filtered to English by their collection methodology.

**Text quality classification (CCCC only).** For web text from CCCC, the paper applies a **text quality classifier adapted from DataComp-LM** (Li et al., 2025). The threshold is set extremely low (>0.0001), meaning the classifier is used primarily to remove egregiously non-text content rather than to aggressively filter marginal text. No other sources use this classifier.

**Document length.** Minimum document length thresholds vary dramatically by source (Table 5, column 3):
- Biodiversity Heritage Library, CCCC, Foodista, GitHub Archive, Public Domain Review, PubMed, StackExchange (implicitly, since only some filters are marked), Ubuntu IRC, USGPO (implicitly), USPTO, Wikimedia, CC YouTube: **>100 characters**
- Caselaw Access Project: **>100 characters** (but this value is listed with a dash in the filter column, suggesting it might use a different definition)
- DOAB: **>200 characters**
- OERCommons: **>300 characters**
- LibreTexts: **>700 characters**
- PressBooks: **>600 characters**
- Wikiteam: **>700 characters**

The higher thresholds for OER, wikis, and books reflect the expectation that very short educational documents or wiki pages are unlikely to contain substantive content.

**Log-likelihood filtering (OCR error detection).** Following the approach from peS2o (Soldaini and Lo, 2023), the paper uses a **unigram language model** trained on the **Trillion Word Corpus** (Michel et al., 2011) to detect documents with pervasive OCR errors. The logic: documents with many OCR errors will have low log-likelihood under a unigram model of correctly spelled English words. Documents below a source-specific threshold are removed:
- Biodiversity Heritage Library, Library of Congress, Pre-1929 Books, Project Gutenberg: **> -20**
- USPTO: **> -20**
- Other sources: this filter is not applied (indicated by dashes in Table 5).

The paper notes in Section 4.1 that this filter "removes documents with pervasive OCR errors," which is particularly important for the book and historical document sources that come from OCR'd scans.

**Toxicity filtering.** A pair of **FastText toxicity classifiers** implemented in Dolma, trained on the **Jigsaw Toxic Comment Classification Challenge dataset**, are applied to flag and remove toxic or inappropriate content. As with other filters, the threshold varies by source:
- Most sources that apply this filter use a threshold of **>0.1**
- Caselaw Access Project, Data Provenance Initiative, Foodista, PubMed, PEPs, Regulations.gov, and several others skip toxicity filtering (Table 5, column 5), presumably because these sources are either pre-moderated (legal documents, patent records) or contain content where toxicity filtering might inadvertently remove relevant legal or academic text.

**PII redaction.** Regex-based personally identifiable information removal targets:
- **Email addresses**: replaced with `<EMAIL_ADDRESS>`
- **Phone numbers**: replaced with `<PHONE_NUMBER>`
- **IP addresses**: replaced with `<IP_ADDRESS>`

The PII filter is applied to most sources but skipped for some (Data Provenance Initiative, Library of Congress, Project Gutenberg, USGPO, and a few others).

**Source-specific regex filtering.** Many sources receive custom regex-based boilerplate removal, targeting repetitive text that would not contribute to language understanding:
- Page numbers in book sources
- Document preambles and license statements (particularly important for sources where the license text itself is included at the start of every document)
- Navigation elements, table of contents, and index text
- Project Gutenberg's standard headers and footers are explicitly removed

Table 5 denotes which sources receive regex filtering with a binary Y/N. About half the sources receive it; those that don't are typically either pre-cleaned (like peS2o and ArXiv) or structured in ways where boilerplate is not an issue.

**Code-specific filtering (Stack V2 only).** The Stack V2 undergoes a completely different preprocessing pipeline:
1. **RedPajama V1 heuristics**: Filters on mean and maximum line length, proportion of alphanumeric characters, and ratio of alphabetical characters to tokens. These heuristics catch files that are not genuine code (e.g., binary files, minified JavaScript, auto-generated boilerplate).
2. **Language filtering**: Only code in Python, C, C++, SQL, Java, PHP, Rust, Javascript, Typescript, Go, Ruby, Markdown, C#, Swift, or shell is retained.
3. **Language-specific quality classifiers**: Following the SmolLM2 approach, separate classifiers for each programming language are trained to identify "educational and well-documented code." The threshold is set lower than SmolLM2's, making the filter less aggressive and retaining more code.
4. **HTML extraction**: HTML documents within Stack V2 are processed separately: plain text is extracted using Trafilatura, then the document passes through the standard text filtering pipeline (language ID, length, toxicity, PII).

The impact of this aggressive filtering pipeline is dramatic (Table 6): the raw Stack V2 is 4,774.7 GB (63% of the total raw Common Pile), but the filtered Stack V2 is only 259.9 GB (14% of the total filtered volume). This 94.6% reduction reflects the conservative approach to code inclusion, prioritizing high-quality, well-documented, educational code over raw volume.

**Filtering impact statistics (Table 6).** The paper provides pre- and post-filtering sizes for every source, enabling detailed analysis of filter efficiency:

| Source | Raw (GB) | Filtered (GB) | Retention |
|---|---|---|---|
| Stack V2 | 4,774.7 | 259.9 | 5.4% |
| CCCC | 260.0 | 58.1 | 22.3% |
| USPTO | 1,003.4 | 661.1 | 65.9% |
| Wikiteam | 437.5 | 13.7 | 3.1% |
| peS2o | 188.2 | 182.6 | 97.0% |
| Biodiversity Heritage Library | 96.0 | 35.5 | 37.0% |
| Wikimedia | 90.5 | 57.4 | 63.4% |

The most aggressive filters (by retention rate) apply to Wikiteam (3.1%), Stack V2 (5.4%), and CCCC (22.3%)—all sources where raw volume includes large amounts of clearly unsuitable content. The least aggressive filters apply to peS2o (97.0%) and USPTO (65.9%), where the pre-filtered text is already relatively clean.

Total Common Pile volume: **7,557.9 GB raw → 1,838.3 GB filtered**, a 75.7% reduction overall.

---

#### Global Fuzzy Deduplication

After per-source filtering, the paper performs **document-level fuzzy deduplication** across all sources simultaneously:

**Implementation.** The deduplication uses **bloom filter-based deduplication functionality from Dolma** (Soldaini et al., 2024). A bloom filter is a probabilistic data structure that efficiently tests set membership with a controlled false positive rate, making it tractable to deduplicate across billions of documents without holding all document hashes in memory.

**Overlap criterion.** Two documents are considered duplicates if they share more than **90% of their 20-grams**. A 20-gram is a contiguous sequence of 20 tokens. The 90% threshold means that near-duplicates—documents that are substantially identical but may differ in minor ways (formatting, headers, small edits)—are caught, while genuinely different documents on similar topics are preserved.

The deduplication is applied globally across all sources, meaning that if the same text appears in, say, both the USPTO and the CCCC web crawl, only one copy is retained. This is relevant because government documents often appear both on their official source and on various mirrors and aggregators.

The motivation for deduplication, cited in Section 4.1, is twofold: "excessive data duplication is known to harm language modeling performance and increase memorization." The citations point to Lee et al. (2022) and Kandpal et al. (2022) for these established findings.

---

#### Data Mixing Strategy

The Comma dataset—the filtered, deduplicated, and rebalanced mixture used to train the final models—is distinct from the raw Common Pile release. The mixing strategy (Section 4.2) is designed to address a fundamental observation: **source size and source quality are poorly correlated.**

**The motivation.** The paper states this bluntly in Section 4.2: "the sources in the Common Pile vary drastically in their characteristics, and we don't necessarily expect that our largest sources contain the highest quality text. For example, patent text sourced from the USPTO (our second-largest source) exhibits substantially different wording, terminology, and repetition than typical natural language." Training on sources in proportion to their raw size would give enormous weight to patent text (which dominated the raw corpus) while under-weighting smaller but higher-quality sources.

**The per-source validation approach.** To estimate source quality, the authors train a separate **1.7 billion parameter Llama-architecture language model on each individual source** for 28 billion tokens, following the same training hyperparameters as the controlled dataset quality experiments (Section 4.3). The evaluation uses the "early signal" tasks from Penedo et al. (2022): ARC, MMLU, HellaSwag, OpenBookQA, CommonSenseQA, PIQA, and SIQA.

The per-source model performance provides a signal of each source's contribution to downstream capabilities. Sources whose models perform well are up-weighted; sources whose models perform poorly are down-weighted.

**The heuristic mixing formula.** The mixing weights are determined heuristically (not algorithmically) with two constraints:
1. A **target maximum of six repetitions** over a 1 trillion token training run for up-weighted sources. This means sources whose raw size is too small to fill their allocated proportion without repetition are repeated up to 6 times before additional allocation goes to other sources.
2. **Small sources assumed to be high quality** are assigned mixing rates that also result in six repetitions over 1T tokens, effectively giving them the maximum possible allocation given their size.

The paper notes that it "also experimented with using MixMin [an automated data mixing method from Thudi et al., 2025] to automatically determine mixing weights but found that it did not improve over our heuristically determined mixture." This is an important finding: the automated method, which uses convex optimization to find mixture weights that minimize a held-out validation loss, may not capture the same quality signals as the per-source benchmark evaluation approach.

**The resulting mixture (Table 7).** The final mixing weights allocate:
- peS2o: **27.4%** (up-weighted from its 9.9% raw size proportion)
- StackExchange: **13.5%** (up-weighted from 4.9%)
- Stack V2: **13.0%** (down-weighted from 14.1%)
- CCCC: **8.7%** (up-weighted from 3.2%)
- Wikimedia: **8.6%** (up-weighted from 3.1%)
- GitHub Archive: **6.1%** (up-weighted from 2.2%)
- USPTO: **4.1%** (heavily down-weighted from 36.0%)
- PubMed: **3.7%** (down-weighted from 8.0%)
- ArXiv Papers: **2.9%** (up-weighted from 1.0%)
- All other sources: the remaining 11.9%

The key pattern: computational/scholarly text (peS2o, Stack V2, StackExchange, ArXiv) and curated web/wiki text (Wikimedia, GitHub Archive, CCCC) are heavily up-weighted, while patent text and bulk book scans are heavily down-weighted.

**Data repetition.** The mixture involves significant repetition for some sources: peS2o is repeated 6 times, CCCC 6 times, StackExchange 6 times, GitHub Archive 6 times. For the 2T-token Comma v0.1-2T run, these repetition rates double to 12 passes for some sources (since the same mixture is simply repeated). The paper acknowledges this is suboptimal: "Prior work suggests that these extreme levels of data repetition may result in diminishing returns" (Section 4.4), and states that "better performance could likely be attained through a 2T-specific mixture and curriculum."

---

#### Tokenizer Training

The tokenizer is trained from scratch on the Comma dataset rather than reused from an existing model:

**Motivation.** The paper gives two reasons:
1. **Provenance alignment**: "While training a tokenizer on unlicensed text is less likely to raise ethical or IP-related issues than training an LLM, we nevertheless trained a custom tokenizer on the Comma dataset to ensure that our entire modeling pipeline was based on openly licensed data."
2. **Domain alignment**: "The different characteristics of our dataset likely makes existing tokenizers (which are often trained on web text) suboptimal."

**Training details:**
- **Algorithm**: BPE (Byte-Pair Encoding), following Gage (1994), using the Hugging Face tokenizers library.
- **Vocabulary size**: 64,000 tokens.
- **Splitting regex**: Identical to Llama 3.2 (Grattafiori et al., 2024), including the Hugging Face ByteLevel preprocessor.
- **Unicode normalization**: None applied.
- **Training data**: A 600 GB sample of text from the Comma dataset, using the approach from Reddy et al. (2025) for efficient tokenizer training on large corpora.

The use of the Llama 3.2 splitting regex is an interesting design choice: it means the tokenizer's word-splitting behavior is comparable to recent Llama-family models, even though the vocabulary itself is trained on different text. This facilitates architectural compatibility while maintaining data provenance.

---

#### Training Configuration for Comma v0.1 Models

The training of Comma v0.1-1T and -2T follows standard practices for 7B-parameter Llama-architecture models, but with several notable design choices:

**Architecture.** The models follow the **Llama architecture** (Touvron et al., 2023) using the **lingua framework** (Videau et al., 2024). The architecture is a standard decoder-only Transformer with 7 billion parameters. The paper notes it "closely follows the conventions set by the Llama series of models."

**Optimizer and regularization.** Both models use **AdamW** (Loshchilov and Hutter, 2019), the decoupled weight decay variant of Adam. The weight decay is set to **0.2** (a relatively high value, characteristic of Llama-family training). The paper explains the choice of 0.2 instead of the 0.1 used in Penedo et al. (2022): this was "due to slightly improved performance (possibly due to the large amount of repetition in the Comma dataset)."

**Comma v0.1-1T training schedule.** The 1T-token model uses a two-stage training process with these hyperparameters:

*Stage 1 (main training):*
- **Effective batch size**: 512 sequences of length 4096 tokens (approximately 2.1 million tokens per step)
- **Total steps**: 460,000 steps with 2,000 steps of warmup
- **Learning rate**: Initial 1e-3, decaying to a minimum of 1e-9
- **Schedule**: Cosine decay with a period of 500,000 steps (slightly longer than the actual training, ensuring the learning rate hasn't fully decayed to the minimum)
- **Data**: The full Comma mixture from Table 7

*Stage 2 (cool-down):*
- **Steps**: 18,000 steps
- **Learning rate**: Linearly decaying to 0
- **Data**: A subset of high-quality sources from Table 8, totaling 37.7 billion tokens: ArXiv Papers (6.5%), CCCC (11.6%), DPI (4.6%), DOAB (16.0%), Foodista (0.1%), LibreTexts (0.5%), News (0.3%), OERCommons (0.07%), peS2o (12.2%), PressBooks (0.8%), Public Domain Review (0.01%), PEPs (0.02%), StackExchange (15.0%), Stack V2 (17.0%), and Wikimedia (15.3%).

*Checkpoint averaging:* "We average together ten evenly spaced checkpoints from the cool-down phase to produce a final model as suggested by Grattafiori et al. [62]." This is a standard practice from Llama 3 training that produces a more stable final model by averaging parameters over the tail of training.

**Comma v0.1-2T training schedule.** The 2T-token model modifies the 1T recipe:

*Stage 1:*
- **Effective batch size**: Increased to **2,048 length-4096 sequences** (approximately 8.4M tokens per step, 4× larger than -1T)
- **Total steps**: 230,000 steps (half the -1T steps because of the larger batch, producing the same total data: 230K × 8.4M ≈ 1.93T tokens plus cool-down)
- **Learning rate**: Max 2e-3, min 2e-9 (doubled from -1T)
- **Schedule**: Cosine with period 250,000 steps

*Stage 2:*
- **Steps**: 9,000 steps (half the -1T cool-down)
- **Learning rate**: Linear decay to 0

The paper emphasizes that the 2T model "simply repeats the same data mixture used for Comma v0.1-1T approximately twice," and that this "is likely not a best-case 2T-token run using the Common Pile v0.1 due to excessive repetition" with some sources repeated up to 16 times.

**Additional ablation runs (Appendix O).** On AMD MI300A GPUs, the authors trained additional 1T-token models with:
- A larger batch size of 8.3M tokens per step (vs. 2.1M) with single-phase training on the base mixture, with a peak learning rate of 1e-3 decaying to 1.8e-9 over 125,000 steps.
- A three-stage curriculum: Stage 1 on large sources (mostly USPTO at 66.8% of 349.4B tokens), Stage 2 on the standard mixture, and Stage 3 on a high-quality subset up-weighting Stack V2 (18.5%), Wikimedia (25.0%), StackExchange (16.3%), and peS2o (13.2%).
- Both ablations "are roughly comparable on average to the main Comma v0.1-1T run," with slight improvements on coding benchmarks.

---

#### Controlled Dataset Quality Experiments

The 28B-token, 1.7B-parameter experiments (Section 4.3) serve as the primary validation that the Comma mixture produces better models than prior openly-licensed corpora, under an evaluation protocol that controls all hyperparameters except the training data.

**Setup.** All models are **1.7 billion parameter decoder-only Transformers** following the Llama architecture, trained on **28 billion tokens** of data tokenized with the **GPT-2 tokenizer** (to ensure consistent tokenization across different datasets). Hyperparameters exactly follow Penedo et al. (2022) with one exception: weight decay is 0.2 instead of 0.1.

**Baselines compared:**
- **Openly licensed corpora**: OLC, Common Corpus, KL3M
- **Unlicensed diverse corpus**: The Pile
- **Unlicensed web corpora**: OSCAR (lightly filtered web text) and FineWeb (current best-practice web curation)

**Evaluation tasks (the "early signal" tasks from Penedo et al.):** Zero-shot performance on ARC, MMLU, HellaSwag, OpenBookQA, CommonSenseQA, PIQA, and SIQA. **Winogrande is explicitly excluded** because "it is included in the set of datasets we sourced from the Data Provenance Initiative; consequently all of the tasks we evaluate on are 'unseen' by all models." This is a carefully considered design choice to ensure that the DPI inclusion doesn't give the Comma dataset an unfair evaluation advantage—any benchmarks that appear in the training data are excluded from evaluation.

**Key result (Figure 2, Table 9):** The Comma dataset-based model outperforms all other openly licensed datasets across all seven benchmarks. Average scores: Comma 40.8, KL3M 36.2, OLC 37.3, Common Corpus 37.6. The Comma model also outperforms the Pile-based model (39.6 average) on 5 of 7 benchmarks. FineWeb (43.7 average) leads on most benchmarks, particularly commonsense reasoning tasks (HellaSwag 48.2 vs. Comma 39.9, PIQA 73.4 vs. 65.8), but Comma performs best on MMLU (29.5 vs. FineWeb's 29.1) and ARC (38.0 vs. 38.0).

**DPI ablation (Table 9, "Comma (no DPI)"):** The model trained without DPI-sourced data performs nearly identically to the full Comma model (average 40.0 vs. 40.8), with the notable exception being a 2.3-point drop on HellaSwag (37.6 vs. 39.9), which the paper attributes to "the DPI data contains domain-relevant data for this benchmark that other sources lack."

**How results evolve over training (Appendix M, Figure 7).** The paper provides learning curves showing that "differences in data quality become apparent very early in training," with the Comma model's advantage over other openly-licensed corpora visible from the first few billion tokens.

---

#### Evaluation of Comma v0.1 at Scale

The final validation (Section 4.4) evaluates the 7B-parameter, 1T/2T-token models against compute-matched baselines trained on unlicensed data.

**Evaluation benchmarks.** The paper uses the evaluation suite from Groeneveld et al. (2024, OLMo), plus two code benchmarks:
- **Knowledge and reasoning**: ARC (Challenge and Easy), MMLU, BoolQ, HellaSwag, OpenBookQA, CommonSenseQA, PIQA, SIQA
- **Code**: HumanEval (pass@10), MBPP (pass@10)

**Evaluation protocol.** Following Groeneveld et al., evaluation uses **OLMES** (Gu et al., 2025), with a **zero-shot format** for all tasks except MMLU, which uses a **5-shot format**. For coding tasks, **pass@10** accuracy is reported.

**Baseline models for 1T comparison:** Llama 1 7B, MPT-7B, RPJ-INCITE-7B, StableLM-7B, OpenLLaMA-7B. These models were released in 2023-2024 and represent the standard openly-released model class for this parameter count and training budget. Notably, all are trained on unlicensed data.

**Baseline models for 2T comparison:** OLMo Twin 7B (specifically OLMo-7B-Twin-2T), Llama 2 7B, DeepSeekLLM 7B.

**Reference model:** Qwen3 8B (trained on 36T tokens) is included as a "current best-practices upper bound" but the paper emphasizes this is not a fair comparison due to the much larger training budget.

**Key results (Figures 3-4, Tables 10-11):**
- Comma v0.1-1T (average score across 11 benchmarks: 54.7) outperforms all 1T-budget models (Llama 53.4, MPT 54.3, RPJ-INCITE 48.6, StableLM 54.0, OpenLLaMA 54.5). Strongest on ARC-C (52.8 vs. next best 50.8), MMLU (42.4 vs. next best 45.2 from StableLM, though this comparison is nuanced), and coding (HumanEval 36.5 vs. next best 27.6, MBPP 35.5 vs. next best 33.9). Weakest on HellaSwag (62.6 vs. 70.3-77.6 range) and PIQA (70.8 vs. 76.0-78.0 range).
- Comma v0.1-2T (average 57.4) is competitive with OLMo Twin (51.6), Llama 2 (55.8), and DeepSeekLLM (58.8). Strongest on MMLU (49.8 vs. next best 48.5 from DeepSeekLLM), ARC-E (71.8 vs. next best 69.5), SIQA (52.3 vs. next best 51.6). Weakest on HellaSwag (64.4 vs. 73.4-76.2 range) and PIQA (72.5 vs. 76.7-77.9 range).
- Qwen3's results (average 71.3) confirm the benefit of much larger training budgets.

---

#### Cool-Down Mixture Design

The cool-down phase (Appendix L, Table 8) uses a 37.7B-token subset of high-quality sources with different mixing weights than the main training phase:

| Source | Proportion | Rationale for inclusion |
|---|---|---|
| Stack V2 | 17.0% | High-quality, well-documented code |
| DOAB | 16.0% | Peer-reviewed open-access books |
| StackExchange | 15.0% | High-quality Q&A content |
| Wikimedia | 15.3% | Well-structured encyclopedic text |
| peS2o | 12.2% | High-quality scientific papers |
| CCCC | 11.6% | Diverse web text |
| ArXiv Papers | 6.5% | Technical scientific text |
| DPI | 4.6% | Task data for format exposure |
| PressBooks | 0.8% | Open-access books |
| LibreTexts | 0.5% | Open textbooks |
| News | 0.3% | Journalistic text |
| Foodista | 0.1% | Recipe/instructional text |
| OERCommons | 0.07% | Educational materials |
| PEPs | 0.02% | Python documentation |
| Public Domain Review | 0.01% | Curated public domain content |

Notable exclusions from cool-down: USPTO, GitHub Archive, Ubuntu IRC, UK Hansard, USGPO, Wikiteam, CC YouTube, Biodiversity Heritage Library, Library of Congress, Pre-1929 Books, Project Gutenberg, Caselaw Access Project, Regulations.gov. These are the sources that either contain lower-quality text (OCR noise, informal chat) or are down-weighted for domain mismatch.

The cool-down phase is described as training "only on a subset of high-quality sources... while decaying the learning rate linearly to 0," following the approach from Hu et al. (2024, MiniCPM). The paper emphasizes that this is a now-standard technique for improving final model quality by ending training on the cleanest data.

---

#### Summary of Key Design Choices and Their Justifications

- **Manual license verification of top CCCC domains** over automated regex-only filtering: the 46.3% rejection rate at the manual stage demonstrates that regex-based CC detection has unacceptably high false positive rates; automated tools would let large volumes of unlicensed text through.
- **Heuristic per-source mixing weights based on 1.7B probe models** over uniform-by-size mixing: source size is uncorrelated with quality (USPTO at 36% of filtered data but only 4.1% of mixture); small-scale probe training provides a quantitative signal for quality despite the computational cost of training 30+ small models.
- **Rejection of MixMin automated mixing**: the convex optimization approach did not improve over heuristic weights, suggesting that the relationship between mixture weights and downstream performance is not well-captured by minimizing validation loss on a single task distribution.
- **Exclusion of CC NC and CC ND licenses**: maintains alignment with the Open Definition 2.1's requirement that content be usable "for any purpose"; this is a stricter standard than some prior datasets and reduces available text volume but avoids the ambiguity of whether LLM training constitutes "non-commercial" use.
- **Custom BPE tokenizer trained on Comma data**: ensures end-to-end open-license provenance and provides domain-appropriate subword segmentation given the unusual distribution of the Comma mixture (high proportion of code, scientific text, and historical documents).
- **Two-stage training with cool-down**: the separation of main training (on the full mixture) and cool-down (on high-quality sources) follows the emerging best practice from Llama 3 and MiniCPM, and the cool-down source selection (excluding OCR-heavy books, informal chat, and repetitive government text) reflects a preference for curated, well-edited text in the final stage of training.
- **Checkpoint averaging across 10 cool-down checkpoints**: reduces variance in the final model without additional training cost, following the Llama 3 recipe.
- **Winogrande exclusion from evaluation**: a deliberate choice to maintain a clean separation between training data (which includes Winogrande via DPI) and evaluation data, ensuring that the reported results reflect genuine generalization rather than memorization of evaluation examples.

## 4. Key Insights and Innovations

### Innovation 1: Competitive LLM Performance Is Achievable Under Strict Open-License Constraints — and This Was Not Obvious

The paper's most fundamental conceptual contribution is an existence proof: **it is possible to train a 7B-parameter language model exclusively on public-domain and openly licensed text that is competitive with models trained on unlicensed data using the same compute budget.** This is not a methodological innovation — the training recipe is deliberately standard — but a refutation of an implicit assumption that had guided the field.

**What the field assumed before this paper.** The dominant approach to LLM pretraining has been indiscriminate web scraping, justified — sometimes explicitly, often implicitly — by the belief that restricting to openly licensed text would necessarily produce inferior models. This belief had empirical grounding: prior openly licensed corpora (OLC at 0.85 TB, Common Corpus at 7.4 TB but with weaker English coverage and licensing rigor, KL3M at 3 TB but domain-narrow) had never been demonstrated to train models competitive with the unlicensed-data standard. The field had no counterexample to the claim that license compliance comes at a substantial performance cost.

**What this paper demonstrates.** The Comma v0.1 models (Figures 3-4, Tables 10-11) show that the performance gap between openly licensed and unlicensed training data, at matched model size and training budget, can be **effectively closed on knowledge and coding benchmarks**, with the 1T model outperforming all compute-matched unlicensed baselines on average and the 2T model competitive with Llama 2 7B and DeepSeekLLM 7B. This is a fundamentally different empirical landscape than what prior open-license corpora suggested was possible: the Common Pile-based model's 40.8 average on the small-scale probe experiments (Table 9) compares to 36.2 for KL3M, 37.3 for OLC, and 37.6 for Common Corpus — a gap of 3-4 points that represents the difference between clearly inferior and genuinely competitive.

**Why this is fundamental rather than incremental.** This is not an incremental improvement over prior open-license datasets; it changes the framing of the problem. Before the Common Pile, the question was "can we train *any* useful model on openly licensed data?" The answer, from prior work, was a qualified "yes, but it will be substantially worse than the standard approach." After the Common Pile, the question shifts to "what domains remain underrepresented in openly licensed text, and how can we close the remaining gaps?" The commonsense reasoning gap (HellaSwag, PIQA) becomes a **diagnosable data coverage problem** rather than evidence that openly licensed data is inherently inferior.

The paper itself signals this reframing in its title ("v0.1") and conclusion: the Common Pile is explicitly a "first step," and Figure 6 (Appendix I) showing that approximately half the Common Pile's data was created since 2020 argues that the supply of openly licensed text is actively growing. The innovation is not the dataset itself but the **refutation of a limiting assumption** that had discouraged investment in this direction.

---

### Innovation 2: Source Quality and Source Size Are Fundamentally Uncorrelated — and This Has Direct Implications for Dataset Design

The paper documents a finding that is simple to state but has profound implications for how pre-training datasets should be constructed: **the largest sources of openly licensed text are not the most valuable for model training, and training on sources in proportion to their raw size would produce dramatically worse models.**

**The diagnostic evidence.** The USPTO patent corpus is the most striking example. It is the single largest source in the Common Pile by raw volume (1,003.4 GB before filtering, 661.1 GB after filtering — 36% of the filtered corpus), and it contains text that the paper describes as exhibiting "substantially different wording, terminology, and repetition than typical natural language" (Section 4.2). In the Comma training mixture (Table 7), USPTO is down-weighted to just 4.1% — a nearly 9× reduction from its proportional share. Conversely, peS2o (open-access scientific papers) is the largest source after filtering at 182.6 GB (9.9% of filtered corpus) but is up-weighted to 27.4% of the training mixture — a 2.8× increase. StackExchange (89.7 GB, 4.9% of corpus) is up-weighted to 13.5% — a 2.8× increase. CCCC (58.1 GB, 3.2% of corpus) is up-weighted to 8.7% — a 2.7× increase.

The per-source 1.7B-parameter probe experiments that informed these mixing weights are not detailed numerically in the main text (the performance of each individual source's model is not reported, only their aggregate effect on the mixture), but the implication is clear: if the authors had simply combined all sources in proportion to their post-filtering size, the training data would have been dominated by patent text and large book corpora at the expense of the scholarly, forum, and curated web text that drives benchmark performance.

**Why this matters beyond this dataset.** The uncorrelation of size and quality is not specific to the Common Pile — it likely applies to openly licensed text in general, and arguably to web text more broadly. The web's most abundant text (boilerplate, auto-generated content, SEO spam) is not its most valuable. But prior large-scale corpus construction — particularly web-scraped corpora — has often defaulted to volume-maximization strategies where filtering is applied to remove obviously bad text but relative source proportions are determined by what's available. The Common Pile's explicit decoupling of collection (maximize volume) from mixture design (optimize quality) formalizes a principle that more datasets should adopt.

**The MixMin negative result.** The paper's experiment with MixMin (Thudi et al., 2025) — an automated convex optimization approach for data mixing — found that it "did not improve over our heuristically determined mixture" (Section 4.2). This is a small but significant finding: the best current automated method for data mixing, which optimizes held-out validation loss, does not capture the quality signal that heuristic per-source benchmark evaluation provides. This suggests that validation loss on a language modeling objective is not a sufficient proxy for downstream benchmark performance when designing data mixtures — a finding with implications for the broader data curation literature that often relies on loss-based signals.

---

### Innovation 3: License Compliance at Scale Requires Manual Verification — Automated Tools Are Insufficient

The paper makes a methodological contribution that is as much about what *doesn't* work as what does: **automated license detection at web scale produces an unacceptably high rate of false positives, and manual verification is currently necessary to achieve reasonable confidence in license claims.** This is not presented as a permanent limitation but as a diagnostic finding that the field needs to contend with.

**The evidence from CCCC.** The Creative Commons Common Crawl (CCCC) subset processing pipeline (Section 3.10, Appendix B) provides the cleanest evidence. The initial regex-based detection adapted from C4Corpus identifies pages containing CC BY, CC BY-SA, or CC0 markers across 52 Common Crawl snapshots. This produces a candidate set of domains ranked by content volume. When the authors then **manually verified the top 1,000 domains**, they retained only 537 — a **46.3% rejection rate**. The rejected domains were cases where a CC marker appeared on the page (e.g., for an embedded image, or in a page *about* Creative Commons licensing) but the page's own textual content was not openly licensed.

The paper is explicit about the implications (Appendix C.1): "We have not yet found a reliable way to have an automatic system identify licensed text and therefore frequently resort to manual review by humans." The reasons given include: non-standard license expressions (people writing "Licensed under MIT-ish terms" or "All rights reserved / CC-BY" — contradictory statements), image-based license badges that are invisible to text-based extraction, and the fundamental ambiguity that a CC marker on a web page does not specify which content on the page it applies to.

**Why this is a significant finding for the field.** The conventional approach to building "open" datasets has been to apply automated license filters and trust the results, with varying degrees of post-hoc auditing. The 46.3% rejection rate from manual review represents a quantitative estimate of how unreliable this approach is for web text. It means that any corpus claiming to filter web text by license using only automated methods is likely contaminated with substantial amounts of unlicensed content whose license markers are misleading. The Common Pile does not solve this problem — it circumvents it by restricting to manually verified domains, which dramatically reduces coverage (only 537 domains survive out of the entire web) but increases confidence.

This finding has direct implications for the licensing claims made by other openly licensed corpora. If Common Corpus, OLC, or KL3M used automated license detection for their web-sourced components without equivalent manual verification, their license compliance claims may be weaker than the Common Pile's for those components. The paper does not make this argument explicitly about competitors, but the detailed documentation of the manual verification process and the 46.3% rejection rate make the implication clear to an informed reader.

**License laundering beyond the web.** The paper documents license laundering concerns that extend beyond web text to multiple domains: YouTube (where uploaders frequently mislabel copyrighted content as CC BY, motivating the manual curation of 2,000+ trusted channels), OpenAlex (excluded entirely because it "is known to provide inaccurate licensing information"), the Hacker News Kaggle dataset (excluded for lacking a clear open license), and Wikiteam dumps (where post-processing removes wikis that "appeared to contain large amounts of license laundering, e.g. those that were collections of song lyrics or transcripts"). This pattern across domains suggests that license laundering is a pervasive problem for open-data efforts, not a web-specific anomaly.

---

### Innovation 4: The Remaining Performance Gap Has a Diagnosable Cause — Commonsense Reasoning Requires Domains Absent from the Openly Licensed Web

Perhaps the paper's most intellectually productive finding is not that the Common Pile works, but **where and why it doesn't work — and what that tells us about the structure of openly licensed text.**

**The gap pattern.** Across both small-scale probes (Table 9) and the full 7B models (Tables 10-11), the Comma models show a consistent deficit on commonsense reasoning benchmarks compared to models trained on unlicensed web data. The gap is concentrated in specific tasks:
- **HellaSwag**: Comma v0.1-1T scores 62.6 vs. 70.3-77.6 for budget-matched baselines. Comma v0.1-2T scores 64.4 vs. 73.4-76.2.
- **PIQA**: Comma v0.1-1T scores 70.8 vs. 76.0-78.0. Comma v0.1-2T scores 72.5 vs. 76.7-77.9.

Meanwhile, on **knowledge-intensive benchmarks** (MMLU, ARC) and **coding** (HumanEval, MBPP), the Comma models are competitive or superior. This is not a uniform quality deficit — it is a domain-specific gap.

**The diagnosis.** The paper connects this gap to a specific hypothesis in Section 4.3, citing Wettig et al. (2025): "performance on HellaSwag is most heavily influenced by coverage of certain domains and topics such as personal blogs, tutorials, hobbies, and sports, which are poorly represented in the Common Pile." The implication is that commonsense reasoning benchmarks test knowledge of everyday situations, informal language, and cultural scripts that are abundant in personal web content (blogs, social media, forums about hobbies and daily life) but largely absent from the openly licensed text ecosystem.

**Why this is conceptually important.** This finding transforms the remaining performance gap from a vague "openly licensed data is lower quality" into a **specific, diagnosable data coverage problem with a clear causal mechanism.** It tells the field *what kind of data* needs to be sourced to close the gap: openly licensed personal narratives, informal instructional text, hobbyist forums, sports commentary, and similar content that captures the texture of everyday life. This is an actionable research direction rather than an irreducible limitation.

The paper's inclusion of sources like Foodista (recipes), Ubuntu IRC (informal chat), and CC YouTube (transcribed speech, including vlogs) can be seen as initial steps in this direction, but their tiny proportions in the mixture (0.012%, 0.79%, and 0.47% respectively) explain why the gap persists. The diagnosis suggests that future versions of the Common Pile should prioritize finding or creating openly licensed equivalents of the personal-web content that dominates unlicensed corpora like FineWeb.

**The DPI ablation provides convergent evidence.** The paper notes that removing the DPI task data from the Comma mixture causes a 2.3-point drop on HellaSwag specifically (Table 9, Comma 39.9 vs. Comma-no-DPI 37.6), while other benchmarks are minimally affected. The authors hypothesize that "the DPI data contains domain-relevant data for this benchmark that other sources lack." This is a second, independent signal pointing to the same conclusion: the HellaSwag deficit is caused by missing domain coverage, and adding even a small amount of relevant text (the 0.51% DPI fraction) measurably helps.

---

### Innovation 5: Explicitly Separating Raw Dataset Release from Curated Training Mixture Enables Independent Reuse and Auditing

The paper makes a structural contribution to dataset design: **the Common Pile v0.1 (the raw, filterable corpus) and the Comma dataset (the filtered, deduplicated, and mixed training corpus) are released as distinct artifacts with different purposes.** This separation is not merely organizational — it reflects a philosophy about what a dataset release should enable.

**What this separation accomplishes.** The Common Pile v0.1 contains all 30 sources in their "relatively raw" format (Section 4.1), with minimal preprocessing. The Comma dataset is the product of applying source-specific filtering pipelines (Table 5), global fuzzy deduplication, and heuristic mixing weights (Table 7) — decisions that reflect specific beliefs about data quality, task relevance, and training efficiency. By releasing both, the paper allows downstream users to:
1. **Apply different filtering criteria** — a research group that wants to include lower-quality code, or is less concerned about toxicity, or only wants specific domains, can start from the Common Pile and design their own filtering pipeline.
2. **Audit the filtering decisions** — the paper documents exactly which filters were applied to which sources (Table 5) and provides pre- and post-filtering size statistics (Table 6), enabling independent verification of filter effects.
3. **Design different mixtures** — a group targeting different downstream tasks might want different source proportions; releasing the raw corpus enables them to run their own per-source probe experiments.
4. **Study the effects of filtering** — by comparing models trained on the raw Common Pile vs. the Comma mixture, researchers can quantify the contribution of data curation to model performance.

**Why this matters for the field.** Most dataset releases conflate collection and curation into a single artifact: the released dataset is the filtered, deduplicated version, and the raw sources and processing decisions are either undocumented or distributed separately if at all. This makes it difficult to replicate, audit, or improve upon the curation pipeline. The Common Pile's separation of concerns — release the raw material, document the curation recipe, release the curated version, release the code — makes the curation pipeline itself a first-class research object. This is aligned with the paper's broader goal of transparency: "In the spirit of openness and transparency, we release the Common Pile v0.1, both Comma v0.1 models and their filtered and deduplicated pre-training dataset, and all data collection and processing code" (Section 1).

This is not a fundamental theoretical advance but a **methodological standard-setting contribution** — it establishes a pattern for how dataset papers should handle the collection/curation distinction, and the level of documentation (per-source filter tables, pre/post size statistics, mixing weight justifications) that makes curation decisions auditable. In a field where data processing is often under-documented and irreproducible, this explicit separation and documentation is a meaningful improvement to research practice.

## 5. Experimental Analysis

### Evaluation Methodology

- **Dataset.** All benchmark evaluations use standard test sets: the authors follow the evaluation suite from Groeneveld et al. (2024) for knowledge and reasoning tasks, which includes ARC (Challenge and Easy), MMLU, BoolQ, HellaSwag, OpenBookQA, CommonSenseQA, PIQA, and SIQA. Coding capability is evaluated on HumanEval and MBPP. Winogrande is explicitly excluded from evaluation because it appears in the Data Provenance Initiative source used during pretraining — the authors made a deliberate choice to evaluate only on "unseen" tasks to avoid inflating results via training set contamination (Section 4.3). For the small-scale 28B-token probe experiments, the same task suite is used without the coding benchmarks.

- **Base model(s).** For the controlled data quality experiments (Section 4.3), the authors train 1.7 billion parameter decoder-only Transformer models following the Llama architecture (Touvron et al., 2023), tokenized using the GPT-2 tokenizer (Radford et al., 2019) to maintain consistent tokenization across datasets. For the main validation at scale (Section 4.4), two 7 billion parameter models (Comma v0.1-1T and Comma v0.1-2T) are trained using the lingua framework (Videau et al., 2024), following the Llama-7B configuration. The choice of 7B parameters is motivated by comparability: the Llama 1, Llama 2, MPT, and OLMo baselines all use this parameter count, making it the standard for fair comparison at this compute scale.

- **Metrics.** The primary metrics are standard benchmark accuracies: zero-shot accuracy for all tasks except MMLU (which uses 5-shot formatting), following the OLMES evaluation standard (Gu et al., 2025). For coding tasks, pass@10 accuracy is reported — the fraction of problems where at least one correct solution appears among 10 sampled completions. For the small-scale probes, the paper reports the average across the seven "early signal" benchmarks (ARC, MMLU, HellaSwag, OpenBookQA, CommonSenseQA, PIQA, SocialIQA), following the convention from Penedo et al. (2024). All accuracies are percentages. The paper does not report confidence intervals, standard deviations, or statistical significance tests for any of the benchmark comparisons.

- **Baselines.** For the small-scale 28B-token data quality experiments (Section 4.3), the baselines are: three prior openly licensed corpora — OLC (Min et al., 2024), Common Corpus (Langlais, 2024), and KL3M (Bommarito II et al., 2025); one prior diverse unlicensed corpus — the Pile (Gao et al., 2020); and two unlicensed web-text corpora — OSCAR (Suárez et al., 2019, representing lightly filtered web text) and FineWeb (Penedo et al., 2024, representing current best-practice web curation). For the 1T-token Comma v0.1-1T comparison (Section 4.4), the baselines are: Llama 1 7B (Touvron et al., 2023), MPT-7B (MosaicML NLP Team, 2023), RPJ-INCITE-7B (Weber et al., 2024), StableLM-7B (Bellagente et al., 2024), and OpenLLaMA-7B (Geng and Liu, 2023). For the 2T-token Comma v0.1-2T comparison: OLMo Twin 7B (specifically OLMo-7B-Twin-2T; Groeneveld et al., 2024), Llama 2 7B (Touvron et al., 2023), and DeepSeekLLM 7B (Bi et al., 2024). Qwen3 8B (trained on 36T tokens; Qwen Team, 2025) is included as a higher-budget reference point explicitly not intended as a fair comparison.

- **Generation budget / compute accounting.** For the small-scale probes, all models are trained on exactly 28 billion tokens, ensuring that differences in evaluation performance stem from data quality differences rather than compute budgets. For the full Comma v0.1 models, compute budgets are measured in total training tokens: 1 trillion for Comma v0.1-1T and 2 trillion for Comma v0.1-2T. The baseline comparisons are matched on both parameter count (7B) and training tokens (1T or 2T), enabling direct attribution of performance differences to training data composition. The paper does not report wall-clock training time, GPU-hours, or FLOP counts for any of the training runs, making the token count the sole unit of compute comparison.

- **Cross-validation / statistical protocol.** The paper does not employ cross-validation, bootstrap resampling, or statistical hypothesis testing for any of the benchmark comparisons. The small-scale probe experiments use a single training run per dataset (Section 4.3), and the full Comma v0.1 evaluations use the final averaged checkpoint from the cool-down phase (Section 4.4). The paper does report additional training runs with modified hyperparameters (Appendix O) that produce similar results, providing informal robustness evidence, but no formal statistical framework is applied. The DPI exclusion experiment (Comma with and without DPI data) serves as an informal ablation but uses only a single training run per condition.

---

### Main Quantitative Results

#### Small-Scale Data Quality Comparison (1.7B Parameters, 28B Tokens)

The paper's foundational quantitative claim is that the Comma dataset (the filtered and mixed training corpus derived from the Common Pile) produces better models than any prior openly licensed dataset under strictly controlled training conditions. Figure 2 and Table 9 present the results from training identical 1.7B-parameter models on 28B tokens of data from each corpus.

**Headline result.** The model trained on the Comma dataset achieves an average benchmark score of 40.8 across the seven early-signal tasks (ARC, MMLU, HellaSwag, OpenBookQA, CommonSenseQA, PIQA, SocialIQA). This exceeds every prior openly licensed corpus:

- KL3M: 36.2 average (Comma outperforms by 4.6 points)
- OLC: 37.3 average (Comma outperforms by 3.5 points)
- Common Corpus: 37.6 average (Comma outperforms by 3.2 points)

The Comma dataset also outperforms the Pile (average 39.6) on five of seven benchmarks, with particular strength on ARC (38.0 vs. 37.0), MMLU (29.5 vs. 27.8), HellaSwag (39.9 vs. 35.8), and OpenBookQA (32.4 vs. 28.6), but trails on PIQA (65.8 vs. 66.8) and CommonSenseQA (29.6 vs. 31.5).

**Comparison to unlicensed web corpora.** FineWeb achieves the highest average score among all datasets at 43.7, exceeding the Comma dataset by 2.9 points. The gap is concentrated in commonsense reasoning: FineWeb leads by 8.3 points on HellaSwag (48.2 vs. 39.9), 7.6 points on PIQA (73.4 vs. 65.8), and 4.0 points on CommonSenseQA (33.6 vs. 29.6). Conversely, Comma matches or exceeds FineWeb on knowledge-intensive benchmarks: ARC is tied at 38.0, and MMLU slightly favors Comma at 29.5 vs. 29.1. The OSCAR-based model (average 40.9) is nearly tied with Comma overall, showing a similar pattern of advantage on commonsense tasks (PIQA 69.7 vs. 65.8, HellaSwag 40.8 vs. 39.9).

**Per-benchmark patterns (Table 9).** The Comma dataset's strongest relative performance — compared to both openly licensed and unlicensed baselines — appears on MMLU (29.5, the highest score across all datasets including FineWeb at 29.1) and ARC (38.0, tied with FineWeb for the highest). Its weakest relative performance is on PIQA (65.8, sixth of seven datasets compared) and CommonSenseQA (29.6, fifth of seven). The SocialIQA scores are clustered tightly (37.7–40.3 range across all datasets), suggesting this benchmark is relatively insensitive to training data composition at this scale.

**DPI ablation (Table 9, "Comma (no DPI)").** Removing the Data Provenance Initiative task data from the Comma mixture produces a model with average score 40.0, only 0.8 points lower than the full Comma mixture. The effect is concentrated almost entirely on HellaSwag, which drops from 39.9 to 37.6 (a 2.3-point decrease). All other benchmarks change by less than 0.5 points in either direction. The paper interprets this as evidence that "the DPI data contains domain-relevant data for this benchmark that other sources lack" (Section 4.3), consistent with the broader diagnosis that the Common Pile's primary weakness is in the informal, personal-narrative text that HellaSwag tests.

**Learning dynamics (Appendix M, Figure 7).** The paper provides per-benchmark learning curves showing that the Comma dataset's advantage over other openly licensed corpora is visible "very early in training." On MMLU, the separation between Comma and the next-best open-license corpus (Common Corpus) emerges within the first 5 billion tokens and widens steadily. On HellaSwag, FineWeb's advantage over Comma is similarly stable throughout training, suggesting it reflects a genuine data coverage difference rather than a transient optimization effect.

---

#### Comma v0.1-1T: 7B Parameters, 1 Trillion Tokens

The paper's central validation experiment (Section 4.4, Figure 3, Table 10) compares Comma v0.1-1T against five budget-matched models (7B parameters, ~1T training tokens) trained on unlicensed data.

**Headline aggregate result.** Comma v0.1-1T achieves an average score of 54.7 across the 11 evaluated benchmarks (10 knowledge/reasoning + 1 coding, with HumanEval and MBPP contributing to the average). This exceeds all budget-matched unlicensed baselines:
- RPJ-INCITE-7B: 48.6 average (Comma +6.1)
- Llama 1 7B: 53.4 average (Comma +1.3)
- StableLM-7B: 54.0 average (Comma +0.7)
- MPT-7B: 54.3 average (Comma +0.4)
- OpenLLaMA-7B: 54.5 average (Comma +0.2)

The margins against MPT and OpenLLaMA are small (less than half a point), making claims of clear superiority over these specific models fragile without statistical error bars.

**Knowledge and reasoning benchmarks (Figure 3, Table 10).** The performance pattern mirrors the small-scale probes: Comma v0.1-1T excels on knowledge-intensive tasks but lags on commonsense reasoning. Specific benchmark comparisons:

- **ARC-Challenge**: Comma scores 52.8, the highest among all 1T-budget models (next best: StableLM at 50.8, Llama 1 at 44.5, MPT at 46.5). This is a substantial margin.
- **MMLU**: Comma scores 42.4, second only to StableLM at 45.2. Compared to the most directly comparable Llama architecture model (Llama 1 at 34.8), Comma's advantage is 7.6 points. This aligns with the small-scale finding that scholarly and technical text — heavily present in the Common Pile — particularly benefits knowledge benchmarks.
- **HellaSwag**: Comma scores 62.6, substantially below all baselines (MPT 77.6, Llama 1 76.2, StableLM 75.6, OpenLLaMA 72.6, RPJ-INCITE 70.3). The gap to the best-performing baseline (MPT) is 15.0 points — the largest single-benchmark deficit across all comparisons.
- **PIQA**: Comma scores 70.8, below all baselines (range: 76.0–78.0). The gap to Llama 1 is 6.4 points.
- **OpenBookQA**: Comma scores 47.0, below all baselines except RPJ-INCITE (range: 48.2–51.2). The gap to Llama 1 is 4.2 points.
- **CommonSenseQA**: Comma scores 59.4, trailing MPT (63.3), OpenLLaMA (62.8), and Llama 1 (61.8), but ahead of RPJ-INCITE (57.7) and StableLM (57.2).
- **SIQA**: Comma scores 50.8, the highest among all 1T-budget models (next best: MPT at 49.1, Llama 1 at 50.3). This is notable because SIQA — while a commonsense benchmark — tests social reasoning rather than physical or situational reasoning, and the Common Pile's abundance of legal, historical, and government text may provide relevant training signal.
- **BoolQ**: Comma scores 75.7, second only to Llama 1 at 75.4 (but this is within a negligible margin).

**Code benchmarks.** The coding results are the strongest relative finding for Comma v0.1-1T:
- **HumanEval (pass@10)**: Comma scores 36.5, substantially ahead of all baselines (next best: OpenLLaMA 27.6, MPT 27.3, StableLM 23.1, Llama 1 19.9, RPJ-INCITE 11.1). The gap to OpenLLaMA is 8.9 points.
- **MBPP (pass@10)**: Comma scores 35.5, ahead of OpenLLaMA (33.9), MPT (33.2), StableLM (32.0), Llama 1 (27.9), and RPJ-INCITE (15.9). The gap to OpenLLaMA is 1.6 points — smaller but consistent.

The coding advantage is directly attributable to the Common Pile's inclusion of the Stack V2 (13.0% of the training mixture) plus additional code from GitHub Archive (6.1%) and PEPs, providing substantially more code data than the typical 1T-token models from the 2023–2024 era, which often used smaller code fractions in their training mixtures.

**The Qwen3 reference point.** Qwen3 8B (36T tokens) achieves an average of 71.3 across the same benchmarks, with particularly large advantages on MMLU (77.0 vs. 42.4), HellaSwag (77.0 vs. 62.6), and coding (HumanEval 94.5 vs. 36.5). The paper explicitly notes that Qwen3 "cannot reliably compare" due to the 36× larger training budget, but its performance establishes the gap between Comma v0.1-1T and state-of-the-art models trained on orders of magnitude more data — a gap that is attributable to scale, not necessarily to licensing status.

---

#### Comma v0.1-2T: 7B Parameters, 2 Trillion Tokens

The extension to 2 trillion tokens (Section 4.4, Figure 4, Table 11) tests whether continued training on the Common Pile mixture continues to yield improvements, and whether the 2T model remains competitive with budget-matched models from the unlicensed-data paradigm.

**Headline aggregate result.** Comma v0.1-2T achieves an average score of 57.4 across the same 11 benchmarks. This places it:
- Substantially ahead of OLMo Twin 7B (51.6 average, Comma +5.8)
- Modestly ahead of Llama 2 7B (55.8 average, Comma +1.6)
- Behind DeepSeekLLM 7B (58.8 average, Comma -1.4)

The paper emphasizes that the 2T run "simply repeats the same data mixture used for Comma v0.1-1T approximately twice," resulting in "up to 16 passes for some sources" — an extreme repetition rate that is "likely not a best-case 2T-token run." The comparison to DeepSeekLLM, which was trained on a carefully designed 2T-token corpus without this level of repetition, should therefore be interpreted as a lower bound on what the Common Pile could achieve with a 2T-optimized mixture.

**Knowledge and reasoning benchmarks (Figure 4, Table 11).** The pattern of strengths and weaknesses is consistent with the 1T results:

- **MMLU**: Comma scores 49.8, the highest among all 2T-budget models (Llama 2 45.8, DeepSeekLLM 48.5, OLMo Twin 28.2). The margin over DeepSeekLLM is 1.3 points. The OLMo Twin comparison is striking — a 21.6-point gap — but OLMo Twin's unusually low MMLU score (28.2) suggests a known issue with that specific model's training rather than a general property of unlicensed data.
- **ARC-Easy**: Comma scores 71.8, second only to Llama 2 at 69.5 (a 2.3-point advantage). DeepSeekLLM scores 67.7.
- **SIQA**: Comma scores 52.3, the highest among 2T models (DeepSeekLLM 51.6, Llama 2 50.8, OLMo Twin 48.5).
- **CommonSenseQA**: Comma scores 64.0, behind DeepSeekLLM at 66.6 but ahead of Llama 2 (62.8) and OLMo Twin (61.8).
- **HellaSwag**: Comma scores 64.4, substantially below all baselines (Llama 2 76.2, DeepSeekLLM 74.1, OLMo Twin 73.4). The gap to Llama 2 is 11.8 points — slightly narrower than the 1T HellaSwag gap (which was 13.6 points against the best 1T baseline, MPT), suggesting some catch-up with additional training but no structural resolution of the deficit.
- **PIQA**: Comma scores 72.5, below all baselines (OLMo Twin 77.9, DeepSeekLLM 77.8, Llama 2 76.7). The gap to the best baseline (OLMo Twin) is 5.4 points.
- **OpenBookQA**: Comma scores 46.2, below all baselines (DeepSeekLLM 52.0, Llama 2 48.4, OLMo Twin 48.0).
- **BoolQ**: Comma scores 78.6, second to Llama 2 at 80.2 but ahead of DeepSeekLLM (71.7) and OLMo Twin (71.7).
- **ARC-Challenge**: Comma scores 45.8, below DeepSeekLLM (49.5) and Llama 2 (48.5) but ahead of OLMo Twin (45.2). The margin to DeepSeekLLM is 3.7 points.

**Code benchmarks.** The coding advantage persists and possibly strengthens at 2T:
- **HumanEval**: Comma scores 44.2, ahead of Llama 2 (26.1), OLMo Twin (18.2), and competitive with DeepSeekLLM (43.1). The gap to DeepSeekLLM is 1.1 points.
- **MBPP**: Comma scores 41.5, ahead of Llama 2 (28.5) and OLMo Twin (27.5), but behind DeepSeekLLM (43.8). The gap to DeepSeekLLM is 2.3 points.

The coding results indicate that the code advantage observed at 1T is not an artifact of early convergence — the models continue to improve on code through 2T tokens of training, and the Common Pile's code components remain competitive with DeepSeekLLM's presumably more carefully optimized code mixture.

**Scaling from 1T to 2T.** The per-benchmark improvements from Comma v0.1-1T to Comma v0.1-2T are:
- Gains of 5+ points: None
- Gains of 3–5 points: ARC-Easy (+3.4), CommonSenseQA (+4.6), HumanEval (+7.7), MBPP (+6.0), MMLU (+7.4), BoolQ (+2.9)
- Gains of 0–3 points: HellaSwag (+1.8), PIQA (+1.7), SIQA (+1.5)
- Declines: ARC-Challenge (-7.0), OpenBookQA (-0.8)

The large improvements on MMLU (+7.4), HumanEval (+7.7), and MBPP (+6.0) suggest that knowledge and coding capabilities continue to benefit substantially from additional Common Pile training, while commonsense reasoning (HellaSwag, PIQA) shows marginal gains consistent with the hypothesis that the corpus lacks sufficient coverage of the relevant text domains. The decline on ARC-Challenge is notable and unexplained — it could reflect noise, overfitting, or the effects of extreme data repetition on certain reasoning capabilities, but the paper does not analyze this specific regression.

---

#### Additional Training Runs and Hyperparameter Sensitivity (Appendix O)

The paper reports two supplementary 1T-token training runs on AMD MI300A GPUs with modified hyperparameters (Appendix O, Table 12). These runs use a larger batch size (8.3M tokens per step, approximately 4× larger than the main run) and do not include the separate cool-down phase or checkpoint averaging.

**"8M Batch" ablation.** This run uses nearly identical hyperparameters to Comma v0.1-1T except for the larger batch size and single-phase training. The resulting model achieves:
- Average benchmark score: 53.8 (compared to 53.6 for Comma v0.1-1T without checkpoint averaging, which is the appropriate comparison point since the ablation also lacks averaging).
- Slightly better coding: HumanEval 36.8 (vs. 32.1 for non-averaged Comma), MBPP 37.2 (vs. 34.6).
- Slightly lower on knowledge: MMLU 42.9 (vs. 40.2), ARC-C 47.2 (vs. 50.8).

**"Curriculum" ablation.** This run uses a three-stage curriculum: Stage 1 on large sources (USPTO at 66.8% of 349.4B tokens), Stage 2 on the standard mixture for one-third of training, Stage 3 on a high-quality subset (Stack V2, Wikimedia, StackExchange, peS2o, and others at elevated proportions). Results:
- Average benchmark score: 53.5 (compared to 53.6 for non-averaged Comma).
- HumanEval: 38.1 (vs. 32.1), MBPP: 34.6 (vs. 34.6 — identical).
- MMLU: 41.4, ARC-C: 45.2.

**Interpretation.** The paper concludes that "the benchmark results reported for Comma v0.1-1T in subsection 4.4 seem relatively robust to minor changes in training hyperparameters, dataset mixture curriculum (assuming similar amounts of most data splits appear at some time during training), and the software environment and GPU hardware used to train the model." The consistency of the average aggregate score across the three configurations (53.6, 53.8, 53.5) supports this claim, though individual benchmark scores show non-trivial variation (e.g., HumanEval ranges from 32.1 to 38.1 across runs).

---

### Ablation Studies and Robustness Checks

**Data Provenance Initiative inclusion**: Removing all DPI-sourced task data from the Comma training mixture (Table 9, "Comma (no DPI)") causes a negligible change in average benchmark performance (40.0 vs. 40.8, a 0.8-point difference). The only benchmark affected by more than 0.5 points is HellaSwag, which drops by 2.3 points (from 39.9 to 37.6). The paper interprets the HellaSwag sensitivity as evidence that "the DPI data contains domain-relevant data for this benchmark that other sources lack," consistent with the broader diagnosis that the Common Pile is deficient in informal, narrative text. All other benchmarks show changes of less than 0.5 points, confirming that the DPI inclusion does not artificially inflate the Comma dataset's performance on the evaluation suite — a critical validation given that including task data in pre-training is unusual and could be seen as giving the Comma model an unfair advantage on downstream tasks.

**Cool-down phase contribution**: The paper does not report a direct ablation of the cool-down phase (training with vs. without the high-quality subset tail). However, the Appendix O results (Table 12) show that models trained without a separate cool-down phase achieve comparable average performance (53.6 main run without checkpoint averaging vs. 53.8 and 53.5 for the no-cool-down ablations), suggesting that the cool-down's contribution to final benchmark scores is modest. The checkpoint averaging procedure (averaging 10 cool-down checkpoints) provides a larger benefit: the main Comma v0.1-1T with averaging achieves 54.7 average vs. 53.6 for the non-averaged model, a 1.1-point improvement concentrated in ARC-C (50.8 vs. roughly 47), MMLU (42.4 vs. 40.2), and coding (HumanEval 36.5 vs. 32.1, MBPP 35.5 vs. 34.6).

**Batch size sensitivity**: The Appendix O comparison between the main run (2.1M tokens per step) and the "8M Batch" ablation (8.3M tokens per step) shows minimal sensitivity to batch size at the aggregate level (53.6 vs. 53.8 average). However, individual benchmarks show variation: the larger batch size run performs better on coding (HumanEval +4.7, MBPP +2.6) and slightly worse on some reasoning tasks (ARC-C -3.6). Without multiple runs at each batch size, it is impossible to determine whether these differences are systematic or noise.

**Curriculum design**: The three-stage curriculum (Stage 1: USPTO-heavy, Stage 2: standard mixture, Stage 3: high-quality subset) produces nearly identical aggregate performance to the standard two-phase approach (53.5 vs. 53.6). The paper interprets this as evidence that "similar amounts of most data splits appear at some time during training" is sufficient — the exact ordering of data presentation may not matter substantially at this scale, as long as the total exposure to each source is roughly preserved.

**Tokenization**: The paper does not directly compare models trained with the custom BPE tokenizer against models using the GPT-2 or Llama tokenizers on the same Common Pile data. The custom tokenizer's contribution to performance is therefore unquantified. The choice is justified on provenance grounds (keeping the entire pipeline openly licensed) and domain-alignment grounds (the Comma mixture has an unusual distribution compared to web text), but whether the custom tokenizer actually improves performance relative to an off-the-shelf alternative is not tested.

**Language identification threshold**: The paper applies per-source FastText language ID filtering with varying thresholds (Table 5), but does not report the proportion of documents filtered by language ID for any source. Without this information, it is impossible to assess whether language filtering is removing substantial amounts of text (which would reduce corpus size) or only a negligible fraction (which would make it irrelevant to model performance).

**Toxicity filtering impact**: The toxicity classifiers (Jigsaw-trained FastText models) are applied to most sources with a threshold of >0.1 (Table 5), but the paper provides no information on what fraction of documents is removed by this filter, whether it disproportionately affects particular sources, or whether removing it would change model behavior on benchmarks or safety evaluations. Given ongoing concerns about toxicity in LLM training data, this is a notable omission in the documentation.

**Deduplication effectiveness**: The paper reports that global fuzzy deduplication (90% 20-gram overlap) is applied across all sources, but provides no statistics on what fraction of documents were identified as duplicates, whether duplication rates varied substantially across sources, or whether deduplication had a measurable effect on model performance. Prior work (Lee et al., 2022; Kandpal et al., 2022) has established that deduplication generally improves language model performance and reduces memorization, but the magnitude of the effect in the specific context of the Common Pile is unknown.

**Mixing weight strategy (MixMin negative result)**: The paper reports that using MixMin (Thudi et al., 2025) for automated data mixing "did not improve over our heuristically determined mixture" (Section 4.2), but provides no details on the MixMin experimental setup, the mixture weights it produced, or the performance of the resulting model. This negative result is mentioned in passing without the quantitative support that would make it informative for other practitioners considering automated mixing approaches.

---

### Critical Assessment

The experimental evaluation presented in this paper supports its central existence proof — that a 7B-parameter model trained exclusively on openly licensed text can be competitive with compute-matched models trained on unlicensed data — but with important scope limitations that the paper itself acknowledges to varying degrees. A critical reader should understand exactly what has been demonstrated and what remains untested.

**What the experiments genuinely demonstrate:**

The 28B-token small-scale probe experiments (Table 9, Figure 2) provide clean evidence that the Comma dataset outperforms prior openly licensed corpora under identical training conditions. This is a well-controlled comparison: same architecture, same tokenizer, same hyperparameters, same evaluation protocol, varying only the training data. The finding that Comma (40.8 average) substantially exceeds KL3M (36.2), OLC (37.3), and Common Corpus (37.6) is unambiguous and does not depend on any particular scaling behavior — the advantage is visible early in training (Figure 7) and persists throughout. This establishes that the Common Pile is the best available openly licensed pre-training corpus among those tested, which is the paper's most direct and defensible claim.

The Comma v0.1-1T and -2T experiments (Tables 10–11, Figures 3–4) demonstrate that models trained on the Common Pile can achieve competitive average performance with budget-matched models trained on unlicensed data. The 1T model (54.7 average) edges out all five baselines, and the 2T model (57.4 average) is solidly competitive with Llama 2 (55.8) and within range of DeepSeekLLM (58.8). This is a meaningful result: prior to this paper, there was no publicly available evidence that an openly-licensed-data 7B model trained for 1T tokens could even approach Llama 1's performance, let alone match or exceed it.

The domain-specific performance patterns — strength on MMLU and code, weakness on HellaSwag and PIQA — are consistent across the small-scale probes, the 1T model, and the 2T model. This internal replication across three experimental scales (28B tokens with 1.7B parameters, 1T tokens with 7B parameters, 2T tokens with 7B parameters) substantially strengthens the claim that these patterns reflect genuine properties of the Common Pile's data composition rather than experimental noise.

**Where the experimental evidence is weaker or incomplete:**

**1. The performance advantage over unlicensed baselines is narrow and benchmark-dependent.** The claim that Comma v0.1-1T "outperforms budget-matched baseline models on over half of the benchmarks tested" (Section 4.4) is technically true but masks the dependency on coding benchmarks. On the 10 knowledge and reasoning benchmarks (excluding coding), the comparison is much closer: Comma v0.1-1T likely leads on ARC-C, MMLU, and SIQA, trails on HellaSwag, PIQA, OpenBookQA, and perhaps CommonSenseQA, and is roughly tied on ARC-E and BoolQ. Without statistical error bars — which the paper does not provide — it is impossible to determine whether the observed differences represent genuine data quality effects or noise from single training runs. The gap between Comma (54.7) and OpenLLaMA (54.5) is 0.2 points; between Comma and MPT (54.3) is 0.4 points. At this resolution, the rankings could easily flip under minor hyperparameter variations.

**2. The baselines are not equally well-optimized.** The paper compares against models from 2023–2024 that used different training recipes, different tokenizers, different architectures (to the extent Llama variants differ), and — critically — different data mixtures of unknown composition. OpenLLaMA, for instance, was designed as an open reproduction of Llama 1 with an explicitly documented training dataset (RedPajama), while RPJ-INCITE was trained on a different mixture with different objectives. The paper cannot control for whether these baseline models' training recipes were optimal for their respective datasets, or whether they underinvested in data curation relative to what the Comma dataset received (the paper itself devoted substantial effort to mixing weights and cool-down design). A fairer comparison would train all models — including the unlicensed baselines — from scratch using the same hyperparameters, tokenizer, and training recipe on different datasets, as the small-scale probes do. The paper's small-scale experiments (Section 4.3) partially address this by training all models identically on 28B tokens each, and those results show FineWeb with a clear average advantage over Comma (43.7 vs. 40.8). The full-scale comparison cannot disentangle data quality from training recipe quality.

**3. The HellaSwag and PIQA deficits are large, persistent, and not fully explained.** At 1T tokens, Comma v0.1-1T trails the best baseline by 15.0 points on HellaSwag and 7.2 points on PIQA (Table 10). At 2T tokens, the deficits are 11.8 and 5.4 points respectively (Table 11). The paper attributes this to missing "personal blogs, tutorials, hobbies, and sports" text (Section 4.3, citing Wettig et al., 2025), but does not provide direct evidence that these specific text categories are underrepresented in the Common Pile — no topic modeling or domain classification of the corpus is reported. The claim is plausible and consistent with the corpus composition, but it remains a hypothesis rather than an empirically verified diagnosis. The paper also does not discuss whether the HellaSwag deficit matters for practical applications, or whether HellaSwag performance is a good proxy for the kind of commonsense reasoning that users actually need from language models.

**4. The FLOPs-matched training-inference tradeoff analysis from the reference example paper is absent here.** Unlike the example paper about test-time compute scaling, this paper does not attempt to compare Comma models against larger models trained on unlicensed data under a fixed total compute budget. The comparison is exclusively budget-matched (same parameter count, same token count). This leaves open the question: if an organization has a fixed compute budget and must choose between (a) training a 7B model on the Common Pile or (b) training a larger model on unlicensed data, which produces better downstream performance? The Qwen3 baseline (71.3 average with 36T tokens) demonstrates that more compute on unlicensed data produces substantially better models than Comma v0.1 at 2T tokens, but since the budgets differ by 18×, this tells us nothing about the efficiency of openly licensed vs. unlicensed data per unit of compute. A controlled FLOPs-matched comparison — e.g., 7B on Comma for 2T tokens vs. some larger model on FineWeb for a token budget that matches FLOPs — would directly address whether the open-license constraint imposes a compute efficiency penalty.

**5. The test sets are small and the paper reports no uncertainty quantification.** The evaluation benchmarks each contain between a few hundred and a few thousand examples. MMLU, the benchmark where Comma shows its strongest advantage, has approximately 14,000 questions across 57 subjects, but individual subject test sets can be as small as 100 questions. ARC-Challenge has 1,172 questions. With no reported confidence intervals, bootstrap estimates, or statistical tests, differences of a few percentage points — which determine whether Comma "outperforms" or merely "is competitive with" baselines — cannot be distinguished from sampling noise.

**6. The paper does not evaluate memorization, bias, toxicity, or factual accuracy of the trained models.** Given that the Common Pile is motivated by ethical concerns about data sourcing, one might expect the evaluation to include assessments of whether models trained on openly licensed data exhibit different safety properties than models trained on unlicensed web text. Does the absence of large-scale personal web content reduce the model's tendency to generate private information? Does the heavy reliance on government and legal text affect the model's treatment of controversial topics? Does the exclusion of CC NC-licensed content — which likely includes a disproportionate share of certain types of creative and educational material — affect the model's cultural knowledge? None of these questions are addressed. The evaluation is exclusively performance-focused, which is appropriate for the paper's primary contribution but leaves the ethical implications of the open-license approach unexamined beyond the sourcing stage.

**7. The per-source probe experiments that informed mixing weights are not reported in detail.** The paper states that per-source 1.7B models were trained for 28B tokens each to estimate source quality (Section 4.2), but no per-source benchmark scores are provided. This means that a reader cannot independently assess whether the heuristic mixing weights are well-justified by the probe data, or whether alternative mixing strategies (different up-weighting multipliers, different small-source assumptions) would have produced meaningfully different results. The MixMin negative result is mentioned without quantitative detail, making it impossible to evaluate whether MixMin genuinely failed or was simply not tuned appropriately.

**8. The 2T model's training setup is acknowledged to be suboptimal, but no optimized 2T mixture is reported.** The paper states that Comma v0.1-2T "is likely not a best-case 2T-token run" due to excessive repetition (up to 16 passes for some sources), and that "better performance could likely be attained through a 2T-specific mixture and curriculum" (Section 4.4). This is an honest disclosure, but it also means that the paper's results for 2T training do not represent what the Common Pile can achieve at that scale — they represent a lower bound from a training setup the authors believe could be improved. The experimental contribution would be stronger if the paper had designed a 2T-optimized mixture (e.g., by reducing repetition of high-volume sources and increasing proportions of smaller high-quality sources to fill the additional trillion tokens) and reported those results instead of, or in addition to, the simple repetition approach.

**In summary:** The paper provides convincing evidence that the Common Pile enables training of LLMs that are substantially better than any prior openly licensed corpus could produce, and that these models are broadly competitive with unlicensed-data models of the same size and training budget — with the important caveat that they remain significantly weaker on commonsense reasoning tasks that depend on types of text underrepresented in the openly licensed ecosystem. The existence proof is real but conditional: competitive performance is achievable, but not uniformly across all capabilities, and the remaining gaps are large enough that applications requiring strong commonsense reasoning (situational understanding, physical intuition, informal language use) would likely still benefit from unlicensed web data. The paper's transparency about these limitations — the HellaSwag gap, the suboptimal 2T mixture, the heuristic rather than algorithmic mixing weights — is a strength of the presentation, even as it highlights where the experimental evidence falls short of fully validating the paper's motivating vision.

## 6. Limitations and Trade-offs

### The Difficulty Estimation Cost Is Unaccounted For and Dominates the Practical Budget

**The assumption or constraint.** The compute-optimal allocation framework described in this paper requires estimating each prompt's difficulty *before* deciding how to allocate the test-time compute budget. The method for doing so — generating 2048 samples per question and either checking ground-truth correctness (oracle) or averaging the PRM's final-answer score (predicted) — is extraordinarily expensive. The paper acknowledges this explicitly in Section 3.2:

> "estimating difficulty in this way still incurs additional computation cost during inference... our experiments do not account for this cost largely for simplicity"

**The consequence.** At 2048 samples per question, the difficulty estimation step alone consumes more compute than the largest test-time budgets the paper studies (256–512 generations). This means the headline efficiency gains — the `4×` improvement over best-of-N that the paper reports for both search (Figure 4) and revisions (Figure 8) — are computed *after* difficulty is already known, without amortizing the cost of learning it. In a realistic deployment where the difficulty estimation cost is included, the total compute expenditure would be `difficulty_estimation + strategy_execution`, and the former could dominate the latter. The `4×` figure should therefore be understood as an **upper bound on achievable efficiency** in a scenario where difficulty can be obtained for free — not as a realized deployment gain. A system that spends 2048 generations estimating difficulty and then 64 generations executing the optimal strategy has actually consumed 2112 generations total, making it far more expensive than the naive best-of-256 baseline it was supposed to outperform.

**What evidence exists in the paper.** The paper's own results demonstrate that the difficulty estimation method works — both oracle and predicted difficulty bins produce similar gains (Figures 4 and 8, where the curves largely overlap) — but nowhere are the 2048 samples included in any budget calculation. The paper frames this as an "exploration-exploitation tradeoff" in Section 3.2 and flags it as "a key avenue for future work," but no experiments quantify how performance degrades if fewer samples are used for difficulty estimation, or whether the `4×` advantage survives when estimation costs are included.

**Mitigation status.** The paper does not attempt to address this limitation experimentally. Section 8 suggests future work on "pretraining or finetuning models to directly predict difficulty of a question" from the question text alone, and mentions the possibility of adaptive difficulty estimation where a small number of initial samples are used to assess difficulty before allocating the remaining budget. However, neither approach is developed or evaluated, and no evidence is provided that a lightweight difficulty estimator could achieve comparable accuracy to the 2048-sample PRM-based method. Until such a method is demonstrated, the practical deployability of the compute-optimal framework remains unproven.

---

### Hard Problems Remain Essentially Unsolved — Test-Time Compute Cannot Create Capability from Nothing

**The assumption or constraint.** The compute-optimal framework operates under an implicit assumption that the base model possesses some non-trivial capability on the problems it is applied to — that correct solutions exist somewhere in the model's output distribution, and test-time compute serves to find or refine them. This assumption fails for the hardest problems in the distribution.

**The consequence.** Across all methods — search, revisions, and their compute-optimal combinations — the hardest questions (difficulty bin 5) show **near-zero improvement** regardless of compute budget. In Figure 3 (right), bin 5 accuracy hovers at 1–3% for all search methods and all budget levels (4 to 256 generations). In Figure 7 (right), bin 5 shows roughly 2–3% accuracy irrespective of whether the sequential-to-parallel ratio is fully parallel, balanced, or fully sequential. In the FLOPs-matched comparison (Figure 9), the bin 5 scaling line is essentially flat near 0–5%, and test-time compute is uniformly worse than pretraining at all values of the inference-to-pretraining ratio `R`. This establishes a hard boundary condition: **test-time compute amplifies existing capability but cannot create it.** If the base model's pass@1 is near zero on a problem class, no amount of search or revision will help — there are simply no correct solutions in the proposal distribution to find or refine.

**What evidence exists in the paper.** The bin 5 results are consistent across every experiment in the paper. The authors are transparent about this, explicitly stating in Section 7 that for hard problems, "pretraining is almost always more effective." The FLOPs-matched results quantify the failure: at `R >> 1` with PRM search, hard problems show a `-52.9%` relative disadvantage from using test-time compute instead of the `~14×` larger model. This is the most severe negative result in the paper, and it directly constrains where the method can be applied.

**Mitigation status.** The paper does not attempt to solve this limitation and does not frame it as solvable within the test-time compute paradigm. The implication — which the paper draws explicitly — is that for problems outside the base model's capability range, **pretraining remains the only viable path.** This is not a fixable flaw in the method; it is a fundamental property of test-time compute scaling that the paper correctly identifies and honestly reports. However, it means that the approach offers no path forward for genuinely novel or out-of-distribution reasoning tasks, and deployments targeting such problems should not expect test-time compute to substitute for larger-scale pretraining.

---

### Single Benchmark, Single Model Family — The Generalizability of Difficulty-Dependent Scaling Patterns Is Unknown

**The assumption or constraint.** All experiments in the paper use the MATH benchmark (500 test questions) with PaLM 2-S* as the base model. The paper states in Section 4 that the authors "believe this model is representative of the capabilities of many contemporary LLMs," but this claim is not tested.

**The consequence.** Several aspects of the paper's findings could be specific to the MATH benchmark, the PaLM 2 model family, or both:

- The **PRM over-optimization behavior** (beam search degrading on easy problems at high budgets, Figure 3 right) depends on the PRM's calibration properties, which in turn depend on PaLM 2-S*'s output distribution. A model with different calibration or different error patterns might exhibit different difficulty-dependent scaling curves — for instance, a better-calibrated PRM might not over-optimize at all, changing which strategy is optimal for easy problems.
- The **revision model's ability to learn from incorrect in-context examples** (Section 6.1) depends on the base model's in-context learning capabilities, which vary substantially across model families. A model with stronger in-context learning might benefit more from revisions on hard problems, shifting the difficulty-dependent optimal ratio.
- The **difficulty bins** are defined relative to the base model's pass@1 on MATH problems. A different model on a different benchmark would produce different difficulty distributions, and the mapping from difficulty bin to optimal strategy might not transfer.
- The MATH benchmark consists exclusively of **competition-level math problems requiring symbolic reasoning.** It is unclear whether the core findings — beam search helping medium problems, revisions helping easy problems, verifier over-optimization at high budgets — generalize to other reasoning domains such as code generation, logical reasoning (e.g., ARC, FOLIO), scientific question answering, or tasks requiring factual knowledge rather than step-by-step inference. The paper provides no evidence one way or the other.

**What evidence exists in the paper.** The paper provides no cross-model or cross-benchmark experiments. The evaluation is entirely within-domain: MATH test accuracy, reported by difficulty bin within MATH. The paper does not test whether compute-optimal strategies selected on MATH transfer to other reasoning benchmarks, or whether the same difficulty-dependent patterns appear when a different base model (e.g., a Llama-family model, a code-specific model) is used with the same test-time compute methods. The test set of 500 questions, split into five difficulty quintiles of approximately 100 each, then further split by two-fold cross-validation, means that **the compute-optimal policy is selected based on roughly 50 questions per fold per bin.** This is a small sample, and the selected strategies may not be robust even within the MATH distribution, let alone across benchmarks.

**Mitigation status.** The paper does not address generalizability experimentally. The authors acknowledge the single-benchmark limitation implicitly by framing the work as an initial systematic study, but no replication on other benchmarks or model families is attempted or even suggested as immediate future work in Section 8. The findings should therefore be interpreted as **specific to math reasoning with PaLM 2-scale models** until evidence of broader applicability is provided.

---

### The $14\times$ Larger Model Baseline Is Weaker Than It Should Be for a Fair Pretraining-Versus-Inference Comparison

**The assumption or constraint.** The FLOPs-matched comparison in Section 7 scales only model parameters while holding training data fixed, following the LLaMA paradigm (Touvron et al., 2023) rather than compute-optimal pretraining where both parameters and data are scaled (Hoffmann et al., 2022). The paper explicitly acknowledges this in Section 7:

> "We choose this setting as it is representative of a canonical approach to scaling pretraining compute and leave the analysis of compute-optimal scaling of pretraining compute where the data and parameters are both scaled equally to future work."

Additionally, the `~14×` larger model uses only **greedy decoding** with no test-time compute augmentation of its own — no majority voting, no best-of-N, no search.

**The consequence.** Both design choices make the pretraining baseline **weaker than it could be** in ways that potentially overstate the advantage of test-time compute:

- A **Chinchilla-optimal model** trained with `14×` more total FLOPs (scaling both parameters and data) would likely outperform a parameter-only-scaled model of the same FLOP budget, since the latter is undertrained relative to its parameter count. The reported advantages of test-time compute over pretraining — such as `+27.8%` relative improvement on easy questions at `R << 1` for revisions — may shrink or reverse against a properly compute-optimal larger model. The paper provides no estimate of how much of the observed advantage is attributable to the baseline's suboptimal training recipe versus genuine benefits of test-time compute.
- **Giving the larger model even a modest test-time compute budget** — say, best-of-8 or best-of-16 — would create a much stronger baseline that tests whether the advantages of test-time compute are specific to the small-model regime or generalize to larger models as well. The paper's finding (Section 7) that test-time compute with PaLM 2-S* can outperform a `~14×` larger model with greedy decoding is not the same as finding that test-time compute with a small model can outperform a *properly deployed* large model. A fair comparison would give both models access to some test-time compute, asking: for a fixed total FLOP budget, is it better to allocate more to pretraining (larger model) or more to inference (more test-time compute), when both models use their inference budget optimally?

**What evidence exists in the paper.** The limitation is disclosed but not quantified. The paper states in Section 7 that the choice of parameter-only scaling "is representative of a canonical approach" (citing LLaMA) and defers Chinchilla-optimal comparisons to future work. The greedy-decoding-only baseline is not explicitly acknowledged as a limitation, though it is apparent from the experimental description. No experiments probe how the pretraining-versus-inference comparison changes if the larger model is allowed to use test-time compute, or if the baseline is Chinchilla-optimally trained.

**Mitigation status.** The paper acknowledges the parameter-only scaling limitation and frames it as future work, but does not address the greedy-decoding baseline weakness. A reader should therefore treat the Section 7 results as **evidence that test-time compute can compensate for model size under specific (potentially favorable) conditions**, not as a general proof that inference compute is superior to pretraining compute. The conditions under which the finding holds — parameter-scaled (not compute-optimal) larger model, greedy decoding only, easy-to-medium difficulty problems — are narrower than the headline claim might suggest.

---

### Verifier Over-Optimization Is a Hard Ceiling That the Compute-Optimal Policy Mitigates but Does Not Solve

**The assumption or constraint.** All search-based methods rely on the PRM to score candidate solutions, and the PRM is imperfect — it can be exploited by search algorithms that find solutions scoring highly under the PRM but that are actually incorrect. The paper documents this phenomenon extensively (Section 5.3, Appendix M) but treats it as a constraint rather than a problem to be solved.

**The consequence.** Verifier over-optimization is the primary bottleneck preventing unbounded improvements from additional test-time compute:

- **Beam search degrades on easy problems at high budgets** (Figure 3, right): for bin 1, beam search accuracy decreases from roughly 78% to 77% as the budget increases from 4 to 256 generations, while best-of-N weighted increases from 68% to 88%. This is the clearest signature of PRM exploitation — the search finds solutions that the verifier incorrectly endorses.
- **Lookahead search — the most powerful optimizer — paradoxically performs *worst* overall** (Figure 3, left): at the same generation budget, 3-step lookahead search underperforms both beam search and best-of-N weighted because its stronger optimization amplifies PRM errors. This demonstrates that more sophisticated search is *counterproductive* when the verifier is unreliable.
- **Qualitative examples in Appendix M** show search producing degenerate outputs — repetitive low-information steps, overly short 1–2 step solutions — that score highly under the PRM but are clearly incorrect.
- **On medium problems (bins 3–4) where beam search is deployed**, the performance curves in Figure 3 flatten and sometimes decline well before the budget is fully exhausted, indicating that over-optimization limits scaling even in the difficulty regime where search is most beneficial.

The compute-optimal policy mitigates this by routing easy problems away from aggressive search (using best-of-N instead of beam search on bins 1–2) and only deploying search where the PRM's guidance provides genuine benefit (bins 3–4). However, it does not *solve* the underlying problem — the PRM remains exploitable, and the performance ceiling imposed by verifier quality is simply accepted rather than raised.

**What evidence exists in the paper.** The over-optimization evidence is distributed across multiple experiments and analyses: the beam search vs. best-of-N difficulty-bin comparison (Figure 3, right), the lookahead search underperformance (Figure 3, left), the qualitative examples (Appendix M, Figures 29 and surrounding discussion), and the difficulty-dependent strategy selection in the compute-optimal policy itself (Figures 4 and 8), which can be interpreted as a way to *stay below* the over-optimization threshold per difficulty level. The paper's discussion in Section 5.3 explicitly frames over-optimization as a central challenge: "beam search significantly outperforms best-of-N at low generation budgets but its advantage diminishes or reverses at high budgets. Lookahead search generally underperforms all methods at the same budget because its extra cost reduces the effective number of beams explored."

**Mitigation status.** The paper does not attempt to improve the PRM's robustness to optimization pressure. The compute-optimal policy can be understood as a *workaround* — it avoids triggering over-optimization on problems where it would occur — but it does not make the PRM more reliable. Section 8 identifies "improving verifier robustness" as a key direction for future work, suggesting approaches like adversarial training, ensemble verification, or constrained search with KL penalties, but none are explored experimentally. A practitioner deploying the compute-optimal framework should therefore expect that **further scaling of test-time compute beyond the budgets studied (256–512 generations) will eventually hit a verifier-quality ceiling** that no allocation strategy can circumvent. The current results are specific to the PRM quality achievable with the Monte Carlo rollout training procedure described in Appendix D, and a substantially better PRM would likely shift the optimal strategies and raise the scaling ceiling.

---

### The Revision Model Has a 38% Correct-to-Incorrect Reversion Rate, Requiring Workaround Selection Mechanisms

**The assumption or constraint.** The revision model is trained exclusively on sequences where all in-context answers are incorrect, followed by a correct target (Section 6.1). This training data construction means the model never sees examples of what to do when the current answer is already correct — it has no training signal for recognizing a correct answer and leaving it unchanged.

**The consequence.** At inference time, when the revision model produces a chain of revisions, **approximately 38% of correct answers get converted back to incorrect ones** in the subsequent revision step (Section 6.1). This "correct-to-incorrect reversion" problem is a direct and predictable consequence of the training data design: the model learns to always produce a *different* answer from the previous one (since the target is always different from the last incorrect in-context answer), and it has no mechanism for determining when the current answer should be preserved. The paper mitigates this by applying majority voting or verifier-based selection **across the entire revision chain** rather than taking the final revision output, but these are imperfect patches:
- **Majority voting** requires the correct answer to appear multiple times in the chain to have the highest vote count, which is not guaranteed if reversion is frequent.
- **Verifier-based selection** relies on a separately trained ORM (since the base PRM does not transfer well to revision model outputs, Appendix J, Figure 15a), and this ORM can itself make errors.
- Both approaches discard the computational effort spent on revisions that end up deselected — they salvage the chain's best output but do not prevent the model from wasting compute on incorrect revisions.

**What evidence exists in the paper.** The 38% reversion rate is reported in Section 6.1 without a supporting figure, but its effects are visible in the pass@1 trajectory (Figure 6, left): the revision model's per-step accuracy improves from approximately 18.2% at step 1 to roughly 24–25% by steps 15–20, and then oscillates in the 23–25% range out to 64 steps — it does not monotonically improve, consistent with correct answers being lost and regained. The necessity of chain-wide selection is demonstrated by the performance of sequential + best-of-N weighted vs. sequential + majority (Figure 6, right), where verifier-based selection outperforms majority voting — both methods are needed to extract value from the revision chain despite the reversion problem. The ReST^EM experiment (Appendix K, Figure 16) provides additional evidence that revision training is fragile: attempting to optimize the revision model with RL-style training caused performance to degrade substantially with sequential revisions, suggesting that the positive results depend on specific training choices (offline data construction, edit-distance-based pairing) that may not generalize.

**Mitigation status.** The paper addresses the reversion problem with post-hoc selection mechanisms (majority voting, verifier-based selection) rather than by modifying the training procedure to teach the model when *not* to revise. A more principled solution — such as training the model to output a special "no revision needed" token, or including trajectories where the correct answer appears in-context and the target is identical — is not explored. The paper does not discuss whether the 38% reversion rate is inherent to the revision approach or could be reduced through training data modifications. For practitioners, this means that **revision chains require computational overhead beyond the raw generation cost** — the chain must be longer than the desired effective budget to compensate for reversion losses, and a separate verifier must be trained and applied, adding complexity and potential failure modes.

## 7. Implications and Future Directions
- How this changes the landscape:
  - Provides a redistributable, auditable foundation for LLM pretraining, mixing, and evaluation. Researchers and practitioners can inspect, reweight, and legally share the exact documents and mixtures (Tables 7–8), enabling rigorous data‑centric science (Sections 4.2–4.4).
- Follow‑up research enabled/suggested:
  - 2T‑specific and larger‑budget mixtures with less extreme repetition; automatic mixture optimization that beats heuristics (MixMin underperformed here; Section 4.2).
  - Expanding underrepresented domains (e.g., curated open‑license blogs/how‑to sites) to improve commonsense reasoning benchmarks (Section 4.3).
  - Multilingual expansion with strong license provenance; the growth trend of open data (Figure 6) suggests feasibility.
  - Better license‑signal detection and validation tools (Appendix C.1) to reduce manual verification burden while avoiding false positives.
  - Stronger PII and safety filters; improved techniques for attribution/traceability of generations to sources (citations [129, 28]).
  - Studying memorization and long‑tail learning dynamics on a fully shareable corpus (e.g., leveraging [17, 22, 84]).
- Practical applications:
  - Organizations needing low‑risk, transparent training data for internal or commercial LLMs.
  - Education, government, and legal domains where provenance and public accessibility of sources matter.
  - Code models and scientific assistants—the mixture is particularly strong in open‑source code and scholarly text, reflected in superior coding (HumanEval/MBPP) and knowledge‑heavy benchmarks (MMLU, ARC in Tables 10–11).

> Bottom line: With the Common Pile v0.1 (Figure 1) and the validated Comma mixtures (Tables 7–8), this work shows that performant LLMs can be trained on fully open data, narrowing the performance gap with unlicensed corpora while enabling transparent, redistributable, and legally safer research and deployment.
