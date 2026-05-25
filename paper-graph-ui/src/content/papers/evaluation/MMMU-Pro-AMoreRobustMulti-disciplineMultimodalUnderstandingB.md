# MMMU-Pro: A More Robust Multi-discipline Multimodal Understanding Benchmark

**ArXiv:** [2409.02813](https://arxiv.org/abs/2409.02813)

## 🎯 Pitch

MMMU-Pro presents a new standard for evaluating multimodal AI by rigorously filtering out questions solvable by text alone, expanding multiple-choice options, and embedding questions within images to enforce true vision-language reasoning. This benchmark reveals that prominent models perform far worse when deprived of text-only shortcuts, exposing significant weaknesses in genuine multimodal understanding and shifting the challenge from superficial perception to deep reasoning. By highlighting these critical gaps, MMMU-Pro drives progress toward more robust, human-like AI systems ready for complex, real-world tasks.

---

## 1. Executive Summary

This paper introduces **MMMU-Pro**, a robust version of the Massive Multi-discipline Multimodal Understanding and Reasoning (MMMU) benchmark, designed to more rigorously assess true multimodal understanding in state-of-the-art models like GPT-4o and Claude 3.5 Sonnet. MMMU-Pro applies a three-step construction process—filtering out questions answerable by text-only LLMs, augmenting candidate options from four to ten, and introducing a **vision-only input setting** (where questions and answers are embedded within screenshots or photos, requiring models to "see" and "read" simultaneously)—yielding a benchmark that causes performance drops of 16.8% to 26.9% across all tested models relative to the original MMMU. The paper finds that explicit OCR prompts provide negligible benefit for advanced multimodal models, while Chain of Thought (CoT) reasoning improves performance primarily in structured reasoning domains, establishing that current models' bottleneck lies not in text extraction but in the integrated interpretation of visual and textual information under increased cognitive load.

## 2. Context and Motivation

### The Core Problem: We Don't Know If Multimodal Models Truly Understand

The central question driving this paper is deceptively simple but profoundly consequential: **When multimodal large language models (MLLMs) achieve high scores on benchmark tests, do those scores reflect genuine multimodal understanding, or are models exploiting superficial shortcuts that bear little resemblance to human-like comprehension?**

This question matters because the stakes are high. If models score well on benchmarks through spurious correlations rather than true reasoning, we collectively risk overestimating their capabilities. The paper articulates this concern directly in Section 1:

> "If models rely on superficial cues rather than true multimodal understanding (Du et al., 2023; Yuksekgonul et al., 2023), we risk overestimating their capabilities and potentially deploying systems that fail in unpredictable ways when faced with novel scenarios (Wu and Xie, 2024)."

This is not merely an academic concern. When multimodal models are deployed in real-world applications—interpreting medical images, assisting with scientific research, navigating user interfaces, or analyzing legal documents—failures rooted in shallow pattern matching rather than integrated understanding can have significant consequences. A model that correctly answers a chemistry question by exploiting statistical regularities in the multiple-choice options rather than understanding the underlying reaction mechanism will falter unpredictably when those superficial cues are absent or different. The gap between benchmark performance and true capability is therefore a *safety and reliability* concern as much as a scientific one.

### The Specific Gap: MMMU's Vulnerability to Shortcuts

The paper is motivated by a specific, empirically grounded observation about its predecessor, the **Massive Multi-discipline Multimodal Understanding and Reasoning (MMMU)** benchmark (Yue et al., 2024). MMMU was itself a major step forward—11.5K carefully curated multimodal questions from college exams, quizzes, and textbooks spanning six core disciplines, 30 subjects, and 183 subfields, with 30 diverse image types (charts, diagrams, maps, chemical structures, etc.). It established a new standard for evaluating whether models possess expert-level, college-equivalent multimodal reasoning abilities. GPT-4o achieved 69.1% on MMMU, suggesting substantial progress toward this goal.

However, the paper identifies a critical weakness: **text-only language models can correctly answer some MMMU questions without ever seeing the associated images.**

Section 2.1 provides concrete evidence. The authors deployed four strong open-source LLMs—Llama-3-70B-Instruct, Qwen2-72B-Instruct, Yi-1.5-34B-Chat, and Mixtral-8×22B-Instruct—to answer MMMU questions with text-only input (no images). They ran each model ten times per question and identified questions as "answerable" if a model answered correctly more than five times. A question was flagged for removal if at least three of the four models answered correctly across the majority of trials. Two illustrative examples appear in Figure 2:

**Example 1 (History):** A question about "The Grange" cartoon asks which group supported the organization represented by the standing figure. Without seeing the image, Llama-3-70B-Instruct correctly answers "Western farmers" by exploiting its pre-existing knowledge—the Grange (Patrons of Husbandry) was a late 19th-century farmers' advocacy organization in the Western United States. The model didn't need the image; it just needed to know historical facts.

**Example 2 (Biology):** A question about the five stages of bacteriophage infection presents a diagram with labeled steps A through E. Without seeing the diagram, the model recognizes the standard sequence of viral infection stages (Attachment → Penetration → Biosynthesis → Maturation → Lysis) from its training data and selects the correct multiple-choice option. The visual information was redundant.

The authors identify two distinct failure modes (Section 2.1):

1. **Text-Only Dependency**: Certain questions are fundamentally independent of—or irrelevant to—the accompanying images. The image may be decorative, supplementary, or entirely unnecessary for answering the question if one possesses the right background knowledge.

2. **Shortcut Exploitation**: Even when questions *should* require images for humans to answer correctly, models find statistical shortcuts or correlations within the candidate options, leveraging pre-training knowledge to arrive at correct answers without visual reasoning. The bacteriophage example illustrates this: a standard sequence of steps that is memorizable without seeing any particular diagram.

These aren't minor edge cases. The systematic filtering process—using four models, ten trials each, with a strict threshold (3 of 4 models, majority of trials)—removed enough questions to raise the benchmark's difficulty substantially, suggesting the original MMMU contained a non-trivial fraction of questions that failed to enforce genuine multimodal reasoning.

### Prior Approaches and Where They Fall Short

To understand why MMMU-Pro represents a meaningful advance, we need to situate it within the broader landscape of multimodal benchmark development. The paper's criticisms (Section 5) implicitly and explicitly address several categories of prior work:

**Early Benchmarks Are Too Simple.** Benchmarks like VQA (Antol et al., 2015), OK-VQA (Marino et al., 2019), and MSCOCO (Lin et al., 2014) were foundational but no longer challenge modern MLLMs. These benchmarks primarily test visual recognition and basic question-answering rather than the expert-level, multi-step reasoning that MMMU targets. The paper notes that "earlier benchmarks... no longer suffice to evaluate the full spectrum of LMMs capabilities."

**More Recent Benchmarks Focus on Narrower Skills.** A wave of more advanced benchmarks—LAMM (Yin et al., 2023b), LVLM-eHub (Xu et al., 2023), SEED (Li et al., 2024b), MMBench (Liu et al., 2023d), CV-Bench (Tong et al., 2024a), MM-Vet (Yu et al., 2024), Mantis (Jiang et al., 2024), and BLINK (Fu et al., 2024)—have covered aspects from basic perception to hallucination detection. However, the paper argues these benchmarks "often fall short in evaluating expert-level domain knowledge and complex reasoning." They test specific sub-skills (spatial reasoning, visual perception, factual knowledge) without the integrated, multidisciplinary, college-exam-level complexity that MMMU brought.

**MMMU Made Strides but Left a Gap.** MMMU addressed the expert-knowledge gap by sourcing questions from actual college exams and textbooks across 30 subjects. But as the paper's own filtering experiment demonstrates, MMMU inherited a vulnerability common in multiple-choice benchmarks: the space of four candidate options is small enough that a knowledgeable model can sometimes guess correctly by elimination, pattern recognition, or prior knowledge, without engaging with the visual modality at all. This is a known challenge in benchmark design—Wang et al. (2024) identified similar issues in text-only benchmarks and motivated the MMLU-Pro benchmark with the same concern about limited option spaces enabling guessing strategies.

**Vision-Language Benchmarks Often Fail to Enforce Cross-Modal Integration.** A deeper issue across many multimodal benchmarks is that they present text and images as separate inputs—the model receives a text prompt *alongside* an image. This format, while natural for many applications, does not require the model to *integrate* modalities in the holistic way humans do. A model can process the text question, briefly glance at the image for specific features mentioned in the text, and produce an answer without ever forming a unified representation of the scene that includes both textual and visual elements. The paper draws on the human cognition analogy explicitly: real-world humans "seamlessly integrate and switch between visual and textual information" (Section 1), as when reading a textbook with diagrams, interpreting a scientific figure with embedded labels, or navigating a graphical user interface where instructions and visual elements are co-located.

### How This Paper Positions Itself

MMMU-Pro is not a brand-new dataset from scratch. It is explicitly constructed as a **robustified version** of MMMU, inheriting MMMU's multidisciplinary scope and college-level difficulty while systematically addressing its identified weaknesses. The paper describes itself as providing "a more rigorous evaluation tool, closely mimicking real-world scenarios" (Abstract). Its positioning relative to prior work has several key dimensions:

**It identifies and quantifies a previously recognized but unresolved problem.** The observation that text-only models can solve some multimodal benchmark questions is not entirely new—the paper cites Lu et al. (2023b) and Zhang et al. (2024b) as prior works that noted similar issues. But MMMU-Pro goes beyond noting the problem to (a) systematically filtering using a principled multi-model, multi-trial protocol, (b) quantifying the impact through the reduction in text-only model accuracy shown in Figure 3, and (c) introducing complementary mechanisms (option augmentation, vision-only setting) that address different aspects of the shortcut problem simultaneously.

**It introduces option augmentation as a defense against guessing.** The paper draws on insights from MMLU-Pro (Wang et al., 2024), which addressed a similar issue in text-only benchmarks by expanding multiple-choice options. Increasing from four to ten options significantly reduces the probability of correct-by-chance answers (from 25% to 10% for random guessing) and makes it harder for models to exploit statistical regularities in small option sets. The paper's Figure 3 quantifies this: after filtering and option augmentation, text-only LLM accuracy dropped substantially compared to the original MMMU, confirming the dual interventions are effective.

**It introduces the vision-only input setting as a novel evaluation paradigm.** This is MMMU-Pro's most distinctive contribution to benchmark design. Rather than supplying text and images separately, the model receives a single image—a screenshot or photograph of a question *as it would appear to a human*—with the question text, answer choices, and any reference images all embedded within the same visual field. This is described as testing "a fundamental human cognitive ability: the seamless integration and switching between visual and textual information" (Section 1).

The motivation is both cognitive and practical. Cognitively, humans reading a test or textbook page integrate text, diagrams, labels, and spatial layout into a unified understanding. The vision-only setting demands that models do the same. Practically, the paper argues this "aligns with how users naturally interact with AI systems, often sharing screenshots or photos rather than separating text and images" (Section 1). Users don't typically transcribe a textbook question and upload the diagram separately; they screenshot the whole page. A benchmark that mirrors this interaction pattern tests deployment-relevant capabilities.

**It provides a diagnostic framework, not just a harder test.** The paper's structure—comparing performance across standard (4 options), standard (10 options), and vision-only settings—allows decomposing *why* a model's performance drops. The difference between standard-4 and standard-10 isolates the effect of option augmentation (reducing guessing). The difference between standard-10 and vision-only isolates the effect of the vision-only input format (increasing cognitive load from modality integration). The total drop (∆3) from original MMMU to MMMU-Pro combines both effects plus the question filtering. This decomposition, visible in Table 1, gives researchers diagnostic information about *which* aspects of their models need improvement, rather than just a single harder evaluation score.

**It targets a capability gap that existing benchmarks do not measure.** The paper's error analysis (Section 3.6, Figure 7) reveals that reasoning errors account for 46% of GPT-4o's mistakes in the vision-only setting—a significant increase from 26% in the original MMMU. Perceptual errors (27%) and knowledge errors (25%) remain important, but the shift toward reasoning errors as the dominant failure mode indicates that MMMU-Pro is successfully testing a different capability profile than its predecessor. The vision-only setting does not primarily challenge OCR (text recognition errors account for 0% of annotated errors in Figure 7); it challenges the integration, interpretation, and reasoning across co-located visual and textual information. This is a capability that existing benchmarks, with their separated text-and-image inputs, do not effectively isolate.

## 3. Technical Approach

### 3.1 Reader Orientation

MMMU-Pro is not a new dataset from scratch but rather a **robustification pipeline** applied to the existing MMMU benchmark—a systematic three-step process that filters, expands, and reformats existing questions to eliminate shortcuts and enforce genuine multimodal reasoning. The system solves the problem that original MMMU questions could sometimes be answered correctly without engaging the visual modality at all: it removes questions where text-only reasoning suffices, makes guessing harder by expanding from 4 to 10 answer options, and creates a **vision-only input setting** where the entire question (text, images, and answer choices) appears as a single screenshot or photograph—forcing models to "see" and "read" simultaneously rather than processing text and images through separate input channels.

### 3.2 Big-Picture Architecture (Diagram in Words)

The MMMU-Pro construction pipeline has three sequential stages, each feeding into the next:

1. **LLM-Based Question Filtering** — Four strong open-source text-only LLMs (Llama-3-70B-Instruct, Qwen2-72B-Instruct, Yi-1.5-34B-Chat, Mixtral-8×22B-Instruct) attempt to answer every MMMU question without access to images. Questions that three or more models answer correctly across the majority of 10 trials are marked as "text-only solvable" and removed. This produces a filtered pool of roughly 1800 questions, evenly sampled across 30 subjects (60 per subject).

2. **Human-in-the-Loop Option Augmentation** — For each surviving question, the answer options are expanded from 4 to 10. GPT-4o generates candidate options, Claude 3.5 Sonnet filters out contextually irrelevant or logically inconsistent ones, and then **two rounds of human expert review** refine and validate the expanded set. During this process, experts also eliminate any remaining questions that lack clear image relevance, removing 70 additional questions to yield 1,730 total questions.

3. **Vision-Only Input Construction** — Each question is transformed into a **screenshot or photograph** where the question text, reference images, and answer choices are all embedded within a single visual field. Human annotators manually capture these in simulated display environments with varied backgrounds, font styles, and font sizes. Each of the 1,730 questions produces two versions: one in standard text-plus-image format and one in vision-only format, yielding **3,460 total evaluation instances**.

The final evaluation framework computes MMMU-Pro's overall score as the average of performance on (a) the standard format with 10 options and (b) the vision-only format, with a standard 4-option format also reported solely for comparison against the original MMMU.

### 3.3 Roadmap for the Deep Dive

I'll explain these components in the order they operate during benchmark construction, because each step's output determines what enters the next:

- **First, the question filtering protocol:** how text-only LLMs are used to identify and remove questions answerable without visual information, including the specific models, trial count, voting threshold, and sampling strategy—this is the gate that determines which questions survive.

- **Second, the option augmentation pipeline:** how answer choices are expanded from 4 to 10 using a combination of model generation (GPT-4o), model filtering (Claude 3.5), and two-round human validation—this addresses the guessing vulnerability that persists even after filtering.

- **Third, the vision-only input setting:** the construction process for screenshots and photographs, the motivation from human cognition and real-world usage patterns, and the design choices around visual diversity (backgrounds, fonts, sizes)—this is the novel evaluation paradigm.

- **Fourth, the evaluation protocol:** how models are assessed across the three settings (standard 4-option, standard 10-option, vision-only), how the overall MMMU-Pro score is computed, and how human expert performance is approximated from original MMMU data.

- **Fifth, the diagnostic axes that the design enables:** how comparing performance across the three settings isolates the effects of option augmentation versus modality integration, giving researchers diagnostic information rather than just a single harder score.

### 3.4 Detailed, Sentence-Based Technical Breakdown

This is primarily a **benchmark construction paper** whose core idea is that systematically removing shortcuts—through text-only filtering, option expansion, and forced visual-textual integration—reveals a more accurate picture of multimodal model capabilities than the original MMMU provides.

---

#### LLM-Based Question Filtering

The first stage addresses the problem identified in Section 2.1: some MMMU questions are answerable from text alone, either because the image is irrelevant to the answer (text-only dependency) or because the model's pre-training knowledge contains the answer regardless of the specific visual input (shortcut exploitation). The filtering protocol is designed to identify and remove these questions systematically.

**Model selection.** Four open-source LLMs are chosen for the filtering task: Llama-3-70B-Instruct, Qwen2-72B-Instruct, Yi-1.5-34B-Chat, and Mixtral-8×22B-Instruct. These are strong but not the absolute largest models available, and critically they are all **text-only**—they have no vision modules and cannot process images. The choice of four models, rather than one, provides a more robust signal: if only one model can answer a question without images, it might be a fluke of that model's specific training data, but if three or four independently trained models from different organizations all succeed, the question is genuinely solvable without visual information.

**Trial protocol.** Each model answers each MMMU question ten times independently. A question is considered "answerable" by a given model if that model answers correctly **more than five times** (i.e., in at least 6 of 10 trials). The ten-trial repetition accounts for stochasticity in model outputs: even with greedy decoding or temperature-zero sampling, some variance in answer format or chain-of-thought reasoning can produce different final answers. By taking a majority-vote approach per model, the protocol distinguishes between questions a model can *reliably* answer and those it gets right only occasionally by chance.

**Voting threshold across models.** A question is flagged for removal if **at least three out of the four models** answer correctly across the majority of their trials. This is a conservative threshold—it requires a consensus across independently developed models from different training distributions. If only one or two models succeed, the question might be solvable only by models with specific pre-training data, and it is retained.

**Sampling from the filtered pool.** From the pool of questions that survive filtering, the authors randomly sample 1,800 questions, evenly distributed across the 30 MMMU subjects (60 questions per subject). This ensures balanced disciplinary coverage in the final benchmark rather than over-representing subjects where fewer questions were removed.

**Effectiveness measurement.** Figure 3 quantifies the impact of this filtering: it shows the accuracy of text-only LLMs on the original MMMU questions, on the filtered set, and on the option-augmented set. The accuracy drops substantially from the original to the filtered set, confirming that the removed questions were indeed easier for text-only models. The further drop after option augmentation demonstrates that expanding answer choices compounds the effect—questions that text-only models could occasionally guess correctly with 4 options become much harder with 10.

---

#### Option Augmentation Pipeline

Even after filtering, some questions can still be answered by text-only LLMs because the small space of 4 candidate options makes it possible to guess correctly or to exploit statistical patterns in how answer choices are constructed. The second stage expands options from 4 to 10, reducing the probability of random correct guessing from 25% to 10% and making option-space reasoning significantly harder.

**Why 10 options?** The paper does not provide a theoretical justification for 10 specifically, but the choice is consistent with MMLU-Pro (Wang et al., 2024), which used the same expansion factor for text-only benchmarks. Ten options create a large enough space that elimination strategies (ruling out obviously wrong answers and guessing among the remainder) no longer provide a reliable path to correctness. The paper's comparison baselines in Table 1 include "Random Choice" (expected accuracy 12.8% with 10 options vs. 24.9% with 4 options) and "Frequent Choice" (always picking the most common answer, yielding 12.1% with 10 options vs. 27.8% with 4), quantifying the guessing advantage that option augmentation eliminates.

**The multi-stage generation and validation process.** The pipeline combines automated model assistance with human expert oversight:

1. **GPT-4o generates candidate options.** For each question, GPT-4o receives the question text, the correct answer, and the original 4 options, and is prompted to generate additional plausible but incorrect options to reach a total of 10. The model is instructed to produce options that are diverse, logically distinct, and contextually relevant to the subject matter.

2. **Claude 3.5 Sonnet filters the generated options.** Claude 3.5 receives the question, the original options, and GPT-4o's proposed expansions, and is tasked with identifying any options that are contextually irrelevant, logically inconsistent with the question premises, too obviously wrong (making elimination trivial), or too similar to other options (reducing effective distinctiveness). Options flagged by Claude are removed or sent back for regeneration.

3. **First round of human review.** Individual human experts (the paper specifies "human experts" but does not detail their qualifications beyond being domain-knowledgeable) assess each expanded question. They verify that all 10 options are diverse (testing different misconceptions or partial understandings), logically distinct (no near-duplicates), and free from ambiguity. If any flaws are identified—for example, an option that is factually true but does not answer the question, or an option whose wording could be interpreted multiple ways—reviewers correct the issues or create new options.

4. **Second round of human review.** Two additional human experts cross-validate each question and its options independently. This double-checking eliminates residual inconsistencies or errors that a single reviewer might miss and provides an additional quality assurance layer.

**Question relevance review.** During the option augmentation process, experts also review the original annotated questions themselves to ensure their relevance to the accompanying images. The paper notes that this step "filters out 70 questions" that "lack a clear connection or coherence" between the question text and the visual content. These are questions where even a human would not need to consult the image to answer—for instance, a question about a historical event where the image is purely decorative, or a question whose text contains all necessary information regardless of what the image depicts. The final count after this culling is 1,730 questions.

**The dual-purpose nature of option augmentation.** Beyond making guessing harder, expanding options to 10 serves a diagnostic function: it forces models to engage more deeply with the multimodal content because superficial elimination strategies (e.g., "this answer mentions a number that appears in the image, so it must be correct") become less reliable when more options contain superficially plausible features. The paper's qualitative analysis (Section 3.5, Figure 11) shows an example where a model correctly answers with 4 options by chance—the model's calculation is wrong, but the correct answer happens to be among the limited choices—while with 10 options the same flawed calculation leads to an incorrect selection. This demonstrates that option augmentation doesn't just increase difficulty; it changes the *nature* of the challenge from option-space reasoning toward content understanding.

---

#### Vision-Only Input Setting

The vision-only input setting is MMMU-Pro's most distinctive contribution to benchmark design. It transforms each question from a text-plus-image pair (where the model receives structured text input alongside an image tensor) into a single image—a screenshot or photograph—where the question text, any reference images, and all answer choices are embedded within the same visual field. The model must extract the question text via OCR-like capabilities, interpret any embedded diagrams or images, and select an answer, all from a single visual input.

**Motivation from human cognition.** The paper grounds this design choice in how humans naturally process multimodal information (Section 1, Section 2.2):

> "This setting challenges models to develop the capability to truly 'see' and 'read' simultaneously, mirroring how humans effortlessly process complex scenes where text and images are intertwined."

Humans reading a textbook page, an exam question, or a scientific paper do not receive text and images through separate input channels. They visually scan a page, read text, examine figures, re-read relevant passages, and integrate everything into a unified understanding. The standard multimodal evaluation paradigm—where a model receives a text string and an image tensor as separate inputs—imposes a processing architecture that differs fundamentally from human visual cognition. The vision-only setting brings the evaluation format closer to the human perceptual experience.

**Motivation from real-world usage.** The paper also argues from practical deployment considerations:

> "This approach aligns with how users naturally interact with AI systems, often sharing screenshots or photos rather than separating text and images."

When users interact with multimodal AI assistants, they frequently share screenshots of web pages, photos of textbook pages, or images of whiteboard content. They do not typically transcribe the text and upload diagrams separately. A benchmark that evaluates models in the screenshot-input paradigm tests capabilities that are directly relevant to these deployment scenarios—the model must handle the messy, integrated visual-textual inputs that real users actually provide.

**Construction protocol.** The paper describes the process in Section 2.2:

> "We ask the human annotators to manually capture photos and screenshots over a simulated display environment. This process involves varying the backgrounds, font styles, and font sizes to replicate the diversity of real-world conditions."

The construction process is entirely human-driven (not automated screenshot generation), which introduces several important design properties:

- **Visual diversity through environmental variation.** Different backgrounds (simulating different display environments, lighting conditions, or paper textures), font styles (simulating different document formatting conventions across disciplines), and font sizes (simulating different document layouts) create a range of visual contexts. This prevents models from overfitting to a particular screenshot style—for instance, learning that "questions in this font with this background color always appear in the biology subset."

- **Realistic artifacts.** Manual capture introduces natural artifacts of photography and screenshots: varying resolution, slight blur, uneven lighting, compression artifacts, and other imperfections that automated rendering might not produce. These artifacts reflect what deployed models actually encounter when users upload photos or screenshots.

- **Two versions per question.** Each of the 1,730 questions produces two evaluation instances: one in standard format (text prompt plus separate image) and one in vision-only format (screenshot or photo containing everything). This pairing is crucial for the diagnostic decomposition that the benchmark enables—comparing performance on the same question in both formats isolates the effect of the input modality.

**The cognitive load hypothesis.** The paper's qualitative analysis (Section 3.5) and response length comparison (Section 3.7, Figure 8) provide evidence for a specific mechanism by which the vision-only setting increases difficulty: increased cognitive load from simultaneous visual and textual processing. Figure 8 shows that GPT-4o generates significantly shorter responses in the vision-only setting, and those responses contain proportionally more "Descriptive" tokens (describing what is in the image) and fewer "Analytical" tokens (reasoning about the answer):

> "One possible reason is that the increased cognition workload of the vision inputs requires the model to focus more on visual processing, which distracts the model from generating extensive reasoning chains."

This load is not about OCR difficulty per se—Table 2 shows that most advanced models achieve high OCR accuracy on the vision-only inputs, and Figure 6 shows that OCR accuracy is not the primary bottleneck, since models with comparable OCR accuracy (e.g., LLaVA-OneVision-72B and InternVL2-Llama3-76B, both around 87-88%) have very different MMMU-Pro Vision scores. Rather, the load comes from the *integration* task: the model must simultaneously parse the visual layout, extract text, identify which text is the question versus the options, locate and interpret embedded figures, and reason across all these elements.

---

#### Evaluation Protocol

The paper defines three evaluation settings and a specific formula for computing the overall MMMU-Pro score.

**Setting 1: Standard format with 4 options.** Models receive the question text (as a text input) and the accompanying image (as an image input), with the original 4 multiple-choice options. This setting is included "solely for comparison purposes, to highlight the increased difficulty of MMMU-Pro" (Section 3.1). It corresponds most closely to the original MMMU evaluation format.

**Setting 2: Standard format with 10 options.** Models receive the question text and accompanying image with the augmented 10-option set. This isolates the effect of option augmentation: any performance drop from Setting 1 to Setting 2 is attributable to the reduced effectiveness of guessing and option-space reasoning, since the multimodal content is identical.

**Setting 3: Vision-only input.** Models receive a single image—the screenshot or photograph containing the embedded question, reference images, and all 10 answer choices. No separate text input is provided. This isolates the effect of the vision-only format: any additional performance drop from Setting 2 to Setting 3 is attributable to the increased cognitive load of extracting and integrating text from a visual scene, since the question content and options are identical.

**Overall MMMU-Pro score.** The paper defines (Section 3.1):

> "The overall performance score for MMMU-Pro is calculated as the average of scores from settings (2) and (3)."

That is, the official MMMU-Pro score for a model is the mean of its accuracy on the standard 10-option format and its accuracy on the vision-only format. The standard 4-option accuracy is reported but not included in the MMMU-Pro score. This averaging gives equal weight to the two core challenges that MMMU-Pro introduces: harder option-space reasoning (10 options) and integrated visual-textual processing (vision-only input).

**Prompting protocols.** Models are evaluated with both **Direct** and **Chain of Thought (CoT)** prompts (Appendix A provides the exact prompt texts). The Direct prompt asks the model to "Answer directly with the option letter from the given choices." The CoT prompt asks the model to "Think step by step before answering" and specifies a required output format: the last line must be "Answer: $LETTER". For the overall results in Table 1, the **higher** of the Direct and CoT scores is reported for each model in each setting. The paper separately analyzes the impact of CoT in Section 3.3 (Figure 5) to understand where and when it helps.

**OCR prompt experimentation.** In the vision-only setting specifically, the paper tests whether explicitly prompting models to perform OCR before solving helps. The "w/ OCR Prompt" asks the model to "Write out the multiple-choice question in the image and then solve it," while the "w/o OCR Prompt" simply asks the model to "Answer the following multiple-choice question in the image" (Appendix A shows both prompts). Table 2 reports results with and without this OCR prompt across models, finding minimal differences for most models—a finding discussed in detail in Section 3.4 of the paper.

**Choice of evaluation models.** The paper evaluates a comprehensive set of both proprietary and open-source models (detailed in Section 3.1, Table 1). Proprietary models include GPT-4o (0513 version), GPT-4o mini, Claude 3.5 Sonnet, and Gemini 1.5 Pro (both 0801 and 0523 versions). Open-source models span a range of sizes and architectures: InternVL2 (8B, 40B, Llama3-76B), LLaVA (OneVision-7B, OneVision-72B, NeXT variants at 7B, 13B, 34B, and 72B), VILA-1.5-40B, MiniCPM-V2.6, Phi-3.5-Vision, Idefics3-8B-Llama3, Qwen2-VL (2B, 7B, 72B), and Pixtral-12B. This diversity enables the paper to make claims about general trends across model families rather than properties of a single architecture.

**Model access and reproducibility.** For proprietary models, the paper evaluates via API access (specific model versions are noted, such as GPT-4o "0513"). For open-source models, exact model names are provided, enabling reproduction. The paper does not detail the exact inference infrastructure, batch sizes, or hardware used—these are standard for API-based and released-model evaluations and are not the focus of the benchmark contribution.

---

#### Approximating Human Expert Performance

Rather than conducting a new, expensive human evaluation on MMMU-Pro, the paper develops an approximation method based on the original MMMU human evaluation data. This is described in Section 3.1 and detailed in Appendix B.

**Why approximation rather than new evaluation?** The paper states explicitly:

> "While rigorous human evaluation of MMMU-Pro provides valuable insights, conducting such an assessment is both time-consuming and costly."

The original MMMU evaluation involved 90 human experts across disciplines, each with relevant domain expertise, documenting their problem-solving processes. Replicating this for 3,460 evaluation instances (1,730 standard + 1,730 vision-only) would require substantial resources. The approximation method leverages the existing data to produce reasonable estimates without additional expert effort.

**Validity of the approximation.** The paper justifies the approximation on three grounds:

1. **Core content and difficulty are unchanged.** The questions in MMMU-Pro are a subset of MMMU questions—they have not been modified in their core content or difficulty, only in their option count and input format. Therefore, human performance on the original MMMU provides a valid proxy for how humans would perform on the same questions in MMMU-Pro.

2. **Detailed solution processes reduce guessing.** In the original MMMU evaluation, human experts were required to write out their problem-solving processes. The paper argues this "significantly reduced the likelihood of random guessing." For questions where experts did not provide detailed solution processes (indicating possible guessing), the approximation simulates random selection from the expanded candidate options.

3. **Human experts seamlessly integrate visual and textual information.** The paper posits that "human experts, with their innate ability to seamlessly integrate visual and textual information, are expected to perform similarly in the vision-only input setting as they do in the original format." This is a key assumption—that the vision-only format, which substantially increases difficulty for AI models, does not similarly burden humans because human visual cognition naturally handles integrated text-and-image scenes.

**The estimation procedure (Appendix B).** The original MMMU human evaluation involved 90 experts answering questions, with each response categorized as either having a detailed solution process (indicating genuine problem-solving) or lacking one (indicating possible guessing). For the 577 MMMU-Pro questions that overlap with the original MMMU validation set, the paper extracts the corresponding human evaluation data.

Questions are categorized into two groups:
- **With solution process:** The expert provided a detailed solution, so we can be confident the answer was derived through reasoning, not guessing. These answers are taken at face value.
- **Without solution process:** The expert may have guessed. For these questions, the paper simulates random guessing by assigning the expected number of correct answers under uniform random selection from the expanded options.

The estimation formula (Equation 2 in Appendix B) is:

$$\text{Num}_{\text{Estimate}}(\text{correct}) = \text{Num}_{\text{w/ Solution}}(\text{correct}) + \left\lfloor\frac{\text{Num}_{\text{w/o Solution}}}{\text{Num}_{\text{total}}} \times \text{Num}_{\text{w/o Solution}}\right\rfloor$$

where `$\text{Num}_{\text{w/ Solution}}(\text{correct})$` is the number of correctly answered questions that had detailed solution processes, `$\text{Num}_{\text{w/o Solution}}$` is the total number of questions without detailed solution processes, and `$\text{Num}_{\text{total}}$` is the total number of questions.

**What it computes:** the total estimated number of correct human answers, consisting of (a) all questions where experts provided solutions and got the correct answer, plus (b) a fraction of the non-solution questions that would be expected correct by random chance. The fraction `$\frac{\text{Num}_{\text{w/o Solution}}}{\text{Num}_{\text{total}}}$` represents the proportion of all questions that are in the "no solution" category, multiplied by the number of such questions to estimate how many would be guessed correctly under uniform random selection. The floor function produces a conservative (lower-bound) estimate.

**Why this form:** the formula deliberately produces a conservative estimate by using the floor function and by assuming random guessing for non-solution questions rather than assuming any systematic knowledge. This ensures the human performance benchmark is not inflated by over-crediting guesses. The alternative—treating all non-solution answers as correct—would overestimate human performance; treating them all as incorrect would underestimate it. The random-guessing assumption is the principled middle ground given the information available.

**Resulting estimates.** Table 4 and Table 5 in Appendix B provide the estimated human performance broken down by discipline and by three performance levels:

- **Low:** overall 73.0% accuracy
- **Medium:** overall 80.8% accuracy  
- **High:** overall 85.4% accuracy

These estimates appear in the main results table (Table 1) alongside model performance, serving as a human ceiling reference. The estimates vary by discipline—for instance, Health and Medicine shows lower human performance (65.2% low, 72.8% medium, 84.8% high) than Science (78.5%, 84.9%, 86.0%), reflecting genuine differences in question difficulty across domains.

**Limitation acknowledged.** The paper explicitly notes in its Limitations section:

> "Our reliance on approximated human performance rather than direct evaluation introduces potential biases in reporting accurate human expert performance."

This is an honest acknowledgment that the approximation is not a substitute for direct evaluation—it may over- or under-estimate human performance in ways that are difficult to quantify without new data collection.

---

#### Diagnostic Axes Enabled by the Design

The MMMU-Pro construction pipeline creates a benchmark that is not merely harder than MMMU, but **diagnostically richer**. By reporting performance across three settings (standard 4-option, standard 10-option, vision-only), the paper enables researchers to decompose a model's performance drop into components attributable to different challenges.

**The ∆1 metric: effect of option augmentation.** The difference between standard 4-option accuracy and standard 10-option accuracy (reported as ∆1 in Table 1) isolates the impact of expanding from 4 to 10 options. Since the multimodal content is identical across these two settings, any performance difference reflects the model's reliance on strategies that work in small option spaces—guessing, elimination, pattern matching among options—that become less effective with 10 options.

For GPT-4o, standard-4 accuracy is 64.7% and standard-10 accuracy is 54.0%, giving ∆1 = −10.7 percentage points. This substantial drop suggests that even the strongest model benefits significantly from the smaller option space in the original MMMU. For weaker models, the drop can be larger—for example, LLaVA-NeXT-7B drops from 33.7% to 19.4%, a ∆1 of −14.3 percentage points.

**The ∆2 metric: effect of vision-only format.** The difference between standard 10-option accuracy and vision-only accuracy (reported as ∆2 in Table 1) isolates the impact of the vision-only input format. Since the question content and option count are identical, any performance difference reflects the additional cognitive load of extracting text from a visual scene, integrating it with embedded images, and reasoning in the vision-only paradigm.

For GPT-4o, standard-10 accuracy is 54.0% and vision-only accuracy is 49.7%, giving an additional drop of 4.3 percentage points. For LLaVA-OneVision-72B, the drop is dramatic: 38.0% to 24.0%, a ∆2 of −14.0 percentage points. This large discrepancy between models with similar OCR accuracy (both around 87-88% per Table 2) but very different vision-only performance demonstrates that the vision-only setting tests integration capabilities beyond text extraction.

**The ∆3 metric: total difficulty increase.** The overall difference from original MMMU validation accuracy to the MMMU-Pro score (the average of standard-10 and vision-only) represents the combined effect of question filtering, option augmentation, and the vision-only format. This is the headline number: performance drops ranging from 16.8% (Claude 3.5 Sonnet) to 26.9% (VILA-1.5-40B).

**Rank change analysis.** Table 1 also reports changes in model ranking from MMMU to MMMU-Pro (shown as ↑ or ↓ arrows). Some models maintain their relative positions, while others shift substantially—for instance, LLaVA-OneVision-72B drops 5 ranks in the vision-only setting relative to its MMMU standing, while VILA-1.5-40B drops 9 ranks. These rank changes are diagnostically informative: a model that drops many ranks on the vision-only setting likely has specific weaknesses in integrated visual-textual processing that are masked in standard evaluations. Conversely, the relative stability of GPT-4o and Claude 3.5 Sonnet at the top of the rankings suggests that frontier models handle the vision-only challenge better, though they still show substantial absolute performance drops.

**The challenge isolation is not perfect.** The paper does not claim that the three settings provide perfectly orthogonal measurements. The vision-only setting necessarily also involves 10 options (since all vision-only questions use the augmented option set), so the ∆2 metric actually captures the interaction between the vision-only format and the large option space, not the vision-only format in isolation. A more complete decomposition would require a vision-only setting with 4 options as well, which the paper does not include—likely for practical reasons of annotation cost and benchmark size management.

## 4. Key Insights and Innovations

### Innovation 1: Benchmark Design as Diagnostic Decomposition, Not Just Difficulty Amplification

The dominant paradigm in benchmark development is escalation: identify weaknesses in an existing benchmark, then build a harder one that models score lower on. This is valuable but fundamentally one-dimensional—it tells you *that* models are worse, but not *why*. MMMU-Pro breaks from this pattern by constructing a benchmark whose structure itself enables fine-grained failure analysis. The paper doesn't just make MMMU harder; it builds in a **decompositional architecture** where comparing performance across three settings (standard 4-option, standard 10-option, vision-only) isolates distinct capability deficits.

**Prior work built harder tests.** Benchmarks like MathVista (Lu et al., 2023a), MathVerse (Zhang et al., 2024b), and BLINK (Fu et al., 2024) raised the difficulty ceiling for multimodal evaluation by introducing more complex reasoning, adversarial examples, or perception-focused challenges. But they typically produce a single aggregate score per model. A researcher looking at a low MathVerse score knows their model struggles with visual math, but cannot easily determine whether the failure stems from diagram interpretation, symbolic reasoning, spatial understanding, or text extraction from figures.

**MMMU-Pro builds in diagnostic axes.** The three evaluation settings create a natural decomposition (Section 3.1, Table 1):

- **Standard 4-option → Standard 10-option (∆1):** Isolates sensitivity to option-space size. A large ∆1 (e.g., GPT-4o mini drops 15.4 points) indicates heavy reliance on strategies that only work in small option sets—guessing, elimination based on superficial option features, or exploiting statistical regularities in 4-choice construction. The model is not engaging deeply with content; it's gaming the option structure.

- **Standard 10-option → Vision-only (∆2):** Isolates the cost of integrated visual-textual processing. A large ∆2 (e.g., LLaVA-OneVision-72B drops 14.0 points) indicates weakness in extracting and reasoning over text embedded in visual scenes, above and beyond any OCR capability. Since the question content and option count are identical across these settings, the ∆2 is a relatively clean measurement of the modality-integration tax.

- **Original MMMU → MMMU-Pro overall (∆3):** Captures the combined effect of filtering, option augmentation, and vision-only format, providing the headline difficulty increase (16.8% to 26.9% across models).

**This is not merely a convenience of experimental design—it is the paper's central intellectual contribution to evaluation methodology.** Prior work that tested models across multiple settings (e.g., MMMU itself, which reported accuracy by discipline and image type) provided descriptive breakdowns. MMMU-Pro's settings are *designed for causal attribution*: because the question content is held constant across settings (same questions, same correct answers), performance differences can be attributed to specific properties of the evaluation format rather than to question difficulty differences. This is essentially a within-subject experimental design applied to benchmark construction—a methodological move borrowed from psychology and human-computer interaction but rarely applied with this rigor in AI evaluation.

**Evidence that the decomposition reveals non-obvious patterns.** The rank-change analysis in Table 1 shows that the ∆1 and ∆2 metrics capture different capability dimensions. VILA-1.5-40B drops only 2 ranks on the standard setting (∆1) but 9 ranks on the vision-only setting, suggesting its weakness is specifically in integrated visual-textual processing rather than in handling larger option spaces. Conversely, LLaVA-NeXT-7B drops 3 ranks on ∆1 but only 1 on ∆2, suggesting option-space sensitivity is its larger relative weakness. Without the decompositional design, both models would simply show "large drops on MMMU-Pro" and the differential diagnosis would be lost.

**Significance beyond this paper.** This diagnostic architecture is a template for future benchmark design. Rather than asking "how can we make this benchmark harder?", the more productive question becomes "what independent capability axes should this benchmark decompose?" A benchmark that reports a single number per model is a blunt instrument; a benchmark that reports a vector of diagnostically meaningful sub-scores enables targeted model improvement. MMMU-Pro demonstrates this principle concretely in the multimodal domain, but the methodological contribution generalizes.

This is a **fundamental shift** in benchmark design philosophy, not an incremental refinement. It transforms evaluation from a measurement problem ("how capable is this model?") into a diagnostic problem ("what specific capabilities does this model lack?"). The paper does not frame it in these terms explicitly, but the architecture speaks for itself.

---

### Innovation 2: The Vision-Only Setting as a Forced Integration Paradigm

Multimodal benchmarks have historically presented text and images as **separate input channels**: the model receives a text string (the question, options) and an image tensor (the diagram, photograph, chart) through different processing pathways. This architecture, while natural for current model designs, creates a perverse incentive: models can succeed by processing text and images semi-independently, extracting information from each modality in parallel and combining results at a late stage, without ever forming an integrated representation of the multimodal scene.

**The standard paradigm enables "lazy" multimodal reasoning.** A model facing a chemistry question with a reaction diagram can read the text ("What is the product of this reaction?"), identify key terms, and then query the image only for specific structural features implied by the text. It never needs to understand the diagram as a unified visual-textual artifact—labels, arrows, molecular structures, and reaction conditions are all *in* the same image, but the model processes the text *about* the image separately. This is fundamentally unlike human engagement with such materials, where reading and viewing are simultaneous, interleaved, and mutually informing.

**The vision-only setting makes this lazy strategy impossible.** By embedding the entire question—text, reference images, answer choices—within a single visual field (a screenshot or photograph), the model is forced to (a) locate and extract the question text from the visual scene, (b) identify which visual elements are the reference images versus the question text versus the answer choices, (c) reason across the extracted text and the embedded images in an integrated manner, and (d) select an answer, all from a single visual input. There is no separate text channel to lean on; everything must be derived from the image.

**Why this is intellectually distinctive, not just practically harder.** Prior work has tested OCR-heavy scenarios (text-rich image understanding, document VQA) and visual reasoning scenarios (diagram interpretation, spatial reasoning). But these typically present the *question* as text and only the *content* as an image. The vision-only setting inverts this: the question itself is part of the visual scene. This matters because it forces the model to solve a **scene parsing and information triage problem** before it can even begin reasoning about the answer. The model must decide "which part of this image is the question I need to answer, which part is the reference diagram I need to analyze, and which part is the list of options I need to consider"—a meta-cognitive task that standard multimodal evaluation formats never require.

**Evidence that this changes the nature of model failures.** The paper's error analysis (Section 3.6, Figure 7) shows that in the vision-only setting, reasoning errors account for 46% of GPT-4o's mistakes, compared to 26% in the original MMMU. Critically, **OCR errors account for 0%**—text extraction is not the bottleneck. The increase comes from the *integration and interpretation* of co-located visual and textual information. The qualitative examples (Appendix J) show concrete failure modes: models correctly extract all text from a screenshot but misinterpret which option corresponds to which letter (Figure 27, Physics), or correctly read a diagram's labels but fail to connect them to the question's constraints (Figure 26, Math). These are not perception failures or knowledge gaps; they are failures of scene-level reasoning under increased cognitive load.

**The response length analysis (Section 3.7, Figure 8) provides mechanistic evidence for the increased load hypothesis.** GPT-4o generates significantly shorter responses in the vision-only setting, with proportionally more tokens spent on "Descriptive" content (paraphrasing what's in the image) and fewer on "Analytical" content (reasoning toward the answer). The paper hypothesizes that "the increased cognition workload of the vision inputs requires the model to focus more on visual processing, which distracts the model from generating extensive reasoning chains." This is a specific, testable mechanism for why the vision-only setting is harder: it taxes a shared cognitive resource, leaving less capacity for analytical reasoning.

**Comparison to prior "vision-only" or "screenshot" benchmarks.** Some prior work has evaluated models on screenshot-based tasks—VisualWebBench (Liu et al., 2024b) and VisualWebArena (Koh et al., 2024) test web page understanding from screenshots, and GUI agent benchmarks often use screenshot inputs. But these test *task completion* in specific domains (web navigation, UI interaction), not *expert-level multidisciplinary reasoning*. MMMU-Pro's vision-only setting applies the screenshot paradigm to college-level science, humanities, and engineering questions—domains where the integration challenge is conceptual and inferential, not navigational. The question is not "can you find the button?" but "can you reason about the physics problem when the problem statement, diagram, and answer choices are all part of the same visual scene?"

This is a **fundamental reframing** of what multimodal evaluation should demand. The vision-only setting is not just a harder version of standard multimodal QA; it tests a qualitatively different capability—the ability to parse, triage, and reason over an integrated multimodal scene in the way humans naturally do when reading a textbook, taking an exam, or analyzing a scientific figure. That current models struggle with this despite strong performance on standard multimodal benchmarks reveals a capability gap that the standard paradigm systematically conceals.

---

### Innovation 3: OCR Is Not the Bottleneck—The Real Problem Is Multimodal Integration Under Load

A natural assumption when models perform worse on vision-only inputs is that they're failing at text extraction—that OCR errors are driving the performance drop. This assumption would point toward an engineering solution: better vision encoders, more OCR training data, improved text recognition architectures. The paper's data systematically refutes this assumption and redirects attention to a deeper problem.

**The evidence against the OCR hypothesis.** Table 2 reports OCR accuracy alongside vision-only task accuracy for all evaluated models. GPT-4o achieves 92.3% OCR accuracy but only 49.7% task accuracy in the vision-only setting—a gap of over 42 percentage points. LLaVA-OneVision-72B achieves 87.8% OCR accuracy but only 24.0% task accuracy—a 63.8 point gap. Even models with relatively weak OCR (MiniCPM-V2.6 at 67.0%) show large task-to-OCR gaps. Figure 6 visualizes this relationship: OCR accuracy and task accuracy are correlated at the high level (top models do well on both, bottom models do poorly on both), but among models with comparable OCR performance, task accuracy varies dramatically. InternVL2-Llama3-76B, GPT-4o mini, and LLaVA-OneVision-72B all cluster around 87-89% OCR accuracy, yet their vision-only task accuracies are 38.0%, 35.2%, and 24.0% respectively—a 14-point spread unexplained by text extraction ability.

**Even explicit OCR prompting doesn't help.** Section 3.4 tests whether instructing models to first extract the question text and then solve it (the "OCR prompt" in Appendix A) improves performance. Across models, the differences are negligible: GPT-4o scores 49.7% with the OCR prompt vs. 49.4% without; InternVL2-Llama3-76B scores 38.0% vs. 37.9%. If OCR were the bottleneck, making OCR an explicit subtask should provide substantial gains. The null result is strong evidence that text extraction is not the limiting factor for capable models.

**The error analysis confirms this.** Figure 7 shows that among 60 annotated GPT-4o errors in the vision setting, **OCR errors account for 0%**. The errors are reasoning errors (46%), perceptual errors (27%), and knowledge errors (25%). "Perceptual errors" here do not mean text misreading—the paper clarifies that "text recognition and OCR do not prove to be the primary bottleneck." Rather, perceptual errors involve misinterpreting visual elements: confusing similar lines on a graph, misreading spatial relationships in a diagram, or failing to notice relevant visual features.

**What this means conceptually.** The vision-only setting doesn't challenge models to read better; it challenges them to **think under higher cognitive load**. When a model must simultaneously parse a visual scene, extract text, locate figures, and maintain all this information in working memory while reasoning about a college-level physics or chemistry problem, the bottleneck shifts from perception to integration and reasoning. The paper's response length analysis (Figure 8) supports this: models produce fewer analytical tokens in the vision setting, suggesting the perceptual overhead consumes cognitive resources that would otherwise go toward reasoning.

**Why this is a significant negative result.** In AI research, negative results that refute plausible hypotheses are valuable because they prevent the field from pursuing dead ends. The finding that "better OCR" is not the solution to MMMU-Pro's vision-only challenge redirects research investment away from incremental text recognition improvements and toward the harder problems of **cross-modal attention under cognitive load, efficient visual-textual feature fusion, and architectures that maintain reasoning depth when perceptual demands increase**. If the paper had simply reported lower scores on vision-only inputs without this analysis, the natural interpretation would be "models need better vision encoders." The OCR analysis shows this interpretation is wrong, or at least incomplete, and that the real bottleneck is architectural—how models allocate shared representational capacity between perception and reasoning.

This is an **incremental but diagnostically crucial advance**. It doesn't introduce a new method or architecture, but it changes how the field should think about the multimodal reasoning problem by eliminating the most obvious (and most investable) explanation for poor performance and pointing toward a subtler one.

---

### Innovation 4: Option Augmentation as a Countermeasure Against Shortcut Reasoning in Multimodal Benchmarks

Multiple-choice question answering creates a structural vulnerability in benchmark design: when the option space is small (typically 4 choices), models can achieve non-trivial accuracy through strategies that have nothing to do with understanding the content—eliminating obviously wrong answers, recognizing patterns in how distractors are constructed, exploiting statistical associations between question wording and correct answer position, or simply guessing with a 25% baseline. The original MMMU, by using 4 options per question, inherited this vulnerability from the standard multiple-choice format.

**Prior work recognized this problem but addressed it in text-only settings.** MMLU-Pro (Wang et al., 2024) demonstrated that expanding from 4 to 10 options substantially reduced the effectiveness of shortcut strategies in text-only multiple-choice benchmarks, causing significant performance drops even for strong language models. The intuition is straightforward: with 4 options, ruling out 2 obviously wrong answers leaves a 50% chance of guessing correctly between the remainder; with 10 options, even after ruling out 6, the remaining 4 still leave only a 25% chance. The option space is large enough that content-ignorant strategies are no longer viable.

**MMMU-Pro extends this insight to the multimodal domain with a crucial twist.** In multimodal benchmarks, options don't just test content knowledge—they interact with the visual modality in ways that create additional shortcut opportunities. A model might select an option because it contains a number that appears in the diagram, a term that matches a label in the image, or a concept that is visually depicted, without truly understanding *why* that option is correct. With only 4 options, the probability that the correct answer contains such a superficial visual cue is non-trivial. With 10 carefully constructed options, several may contain superficially plausible visual features, forcing the model to engage in genuine content-level discrimination rather than visual cue-matching.

**The paper provides a concrete example (Figure 11, Appendix H).** In a physics problem about the minimum required diameter of a copper bar, GPT-4o makes a calculation error but happens to select the correct answer (35.7 mm) with 4 options because that option is the only one in a plausible range. With 10 options, the same calculation error leads to an incorrect choice because multiple options fall within the plausible range, and the model's flawed reasoning no longer happens to land on the correct one. This illustrates a general principle: **small option spaces can make models look smarter than they are by increasing the probability that a wrong reasoning process produces the right answer by coincidence.** Option augmentation reduces this "reasoning-process-independent" component of accuracy, making the benchmark a purer measure of content understanding.

**The human-in-the-loop construction is essential to the innovation.** Simply asking a language model to generate 6 additional wrong answers would produce options that are either trivially distinguishable (too obviously wrong, making elimination easy) or systematically biased (sharing statistical patterns that models can learn to recognize). The paper's multi-stage pipeline—GPT-4o generation, Claude 3.5 filtering, two rounds of human expert review—is designed to produce option sets where each distractor is *genuinely plausible* to someone who doesn't fully understand the content, but *clearly wrong* to someone who does. This is the ideal property for a multiple-choice test, and achieving it at scale with 1,730 questions requires the hybrid human-AI approach the paper develops.

**Quantifying the impact.** Table 1 shows that moving from 4 to 10 options (∆1) produces substantial drops across all models—10.7 points for GPT-4o, 16.4 points for InternVL2-Llama3-76B, 19.5 points for GPT-4o mini. These are not small effects; they represent a significant fraction of the total MMMU-Pro difficulty increase. The "Random Choice" baseline drops from 24.9% to 12.8%, and "Frequent Choice" from 27.8% to 12.1%, quantifying the guessing advantage that 4 options provide. The gap between these baselines (12.1-12.8%) and the best model performance (54.0% for GPT-4o) confirms that models are doing far better than chance, but the substantial ∆1 shows they were also benefiting substantially from the small option space in the original MMMU.

This is an **incremental advance** in methodology rather than a fundamental breakthrough—the core idea of option expansion comes from MMLU-Pro—but the paper's adaptation to the multimodal context, the rigorous construction pipeline, and the quantitative demonstration of its impact on model rankings make it a significant contribution to multimodal benchmark design specifically. It establishes that **option-space vulnerability is not just a text-only problem; it afflicts multimodal benchmarks as well, and addressing it requires domain-specific construction methods** (plausible visual-cue-based distractors) that differ from text-only option generation.

## 5. Experimental Analysis

### Evaluation Methodology

- **Dataset.** All experiments use questions drawn from the **Massive Multi-discipline Multimodal Understanding and Reasoning (MMMU)** benchmark (Yue et al., 2024), specifically the validation set. From this pool, the paper constructs MMMU-Pro through its three-step filtering, augmentation, and vision-only conversion process, yielding **1,730 questions** in standard format and **1,730 in vision-only format** (3,460 total evaluation instances), evenly distributed across 30 subjects at 60 questions per subject in the standard format. The original MMMU validation set performance is reported solely for comparison to highlight the increased difficulty of MMMU-Pro.

- **Base model(s).** The paper evaluates a comprehensive set of both **proprietary and open-source multimodal models** spanning a wide range of architectures and scales. Proprietary models include **GPT-4o (0513 version)**, **GPT-4o mini**, **Claude 3.5 Sonnet**, and **Gemini 1.5 Pro** (both 0801 and 0523 versions). Open-source models include **InternVL2** (8B, 40B, Llama3-76B variants), **LLaVA** (OneVision-7B, OneVision-72B, NeXT at 7B, 13B, 34B, and 72B), **VILA-1.5-40B**, **MiniCPM-V2.6**, **Phi-3.5-Vision**, **Idefics3-8B-Llama3**, **Qwen2-VL** (2B, 7B, 72B), and **Pixtral-12B**. The models are chosen to "represent a range of training approaches and capabilities in the field of multimodal AI" (Section 3.1), enabling claims about general trends rather than properties of a single architecture.

- **Metrics.** The primary metric is **accuracy**: the fraction of questions for which the model selects the correct answer letter from the given options. Three settings are evaluated: (1) **Standard 4-option accuracy** — questions in text-plus-image format with original 4 options, reported only for comparison against original MMMU; (2) **Standard 10-option accuracy** — same format with augmented 10 options; (3) **Vision-only accuracy** — questions presented as screenshots/photographs with embedded text and images, with 10 options. The **official MMMU-Pro score** is the average of Standard 10-option and Vision-only accuracies (Section 3.1). Three derived metrics decompose the difficulty increase: **∆1** = Standard 10-option − MMMU (Val), **∆2** = Vision − MMMU (Val), and the overall drop (∆3) is implicit in the total MMMU-Pro score. Additionally, **OCR accuracy** is measured using Levenshtein distance between extracted and original text, computed as `1 − Levenshtein_distance(text1, text2) / max(len(text1), len(text2))`.

- **Baselines.** The paper includes several baselines: (1) **Random Choice** — expected accuracy from uniform random selection among options (24.9% for 4 options, 12.8% for 10 options in the standard setting, 12.4% for vision); (2) **Frequent Choice** — always selecting the most common answer across the dataset (27.8% for 4 options, 12.1% for 10 options, 12.1% for vision); (3) **Human Expert performance** — approximated from original MMMU human evaluation data at three proficiency levels (Low: 73.0%, Medium: 80.8%, High: 85.4% overall, with discipline-specific breakdowns in Appendix B, Table 4). Text-only LLM accuracy on original, filtered, and option-augmented question sets (Figure 3) serves as a validation baseline for the filtering and augmentation steps.

- **Generation budget / compute accounting.** The paper does not frame evaluation in terms of compute budgets or generation counts; it is a **benchmark evaluation paper**, not a compute-scaling study. Models are evaluated via API calls (proprietary) or standard inference (open-source) with prompts provided in Appendix A. Each model receives the same questions under identical prompting protocols, making comparisons fair without requiring compute-matched budgets. For the CoT vs. Direct comparison, each model is evaluated with both prompts and the **higher score** is reported in the main results (Table 1). For the OCR prompt experiment, models are evaluated with and without an explicit OCR instruction prompt and both scores are reported (Table 2).

- **Cross-validation / statistical protocol.** The paper does not employ cross-validation, statistical significance testing, or confidence intervals. Results are reported as point estimates (single accuracy percentages) on the full 1,730-question standard set and 1,730-question vision set. The ten-trial repetition in the text-only LLM filtering step (Section 2.2) provides some robustness against stochastic variation in model outputs, with the "more than five times correctly" threshold ensuring only reliably answerable questions are flagged. Human expert performance is estimated using a conservative formula (Appendix B, Equation 2) that assumes random guessing for questions without detailed solution processes, producing lower-bound estimates rather than confidence intervals. The absence of statistical testing is a limitation discussed in the Critical Assessment below.

### Main Quantitative Results

#### Overall Performance Drop Across All Models

The headline finding from Table 1 is that **all tested models experience substantial performance declines on MMMU-Pro compared to the original MMMU validation set**. The MMMU-Pro score (average of Standard 10-option and Vision-only) ranges from **16.8 to 26.9 percentage points lower** than original MMMU accuracy across models.

The best-performing model is **GPT-4o (0513)**, achieving 54.0% on Standard 10-option and 49.7% on Vision-only, yielding a MMMU-Pro average of approximately 51.9% — a drop of roughly 17.2 points from its 69.1% MMMU validation accuracy. **Claude 3.5 Sonnet** closely follows with 55.0% Standard 10-option and 48.0% Vision-only (average ~51.5%, drop of ~16.8 points from 68.3%). These represent the smallest absolute drops among evaluated models, suggesting frontier proprietary models are relatively more robust to MMMU-Pro's interventions, though they still lose roughly a quarter of their original accuracy.

**Larger drops occur across open-source models.** InternVL2-Llama3-76B drops from 58.3% to an average of approximately 40.0% (a decline of ~18.3 points). LLaVA-OneVision-72B shows one of the most dramatic vision-specific drops: from 56.8% on MMMU to 38.0% Standard 10-option and only 24.0% Vision-only (average ~31.0%, drop of ~25.8 points). The **largest overall decline** is for VILA-1.5-40B, which drops 26.9 percentage points (from 51.9% to an average of ~25.0%), driven primarily by catastrophic Vision-only performance of 14.1%.

**Human performance estimates serve as a ceiling reference.** Even at the "Low" proficiency level, estimated human accuracy (73.0%) substantially exceeds the best model (GPT-4o at ~51.9%). At the "High" level (85.4%), the gap is approximately 33.5 percentage points. This confirms that MMMU-Pro is not impossibly difficult — humans perform well on the same questions — but that current MLLMs have substantial room for improvement, particularly in the vision-only setting where the human-to-model gap is largest.

#### Effect of Option Augmentation (∆1)

The difference between Standard 4-option and Standard 10-option accuracy (∆1 in Table 1) isolates the impact of expanding from 4 to 10 candidate answers. Across all models, **∆1 ranges from −10.7 to −20.0 percentage points**, demonstrating that even the strongest models benefit significantly from smaller option spaces.

**GPT-4o shows a ∆1 of −10.7 points** (64.7% → 54.0%), the smallest drop among all models, indicating relatively strong resistance to option-space expansion. **Claude 3.5 Sonnet** shows a similar ∆1 of −13.3 points (63.7% → 55.0%). At the other extreme, **GPT-4o mini** drops 19.5 points (55.3% → 39.9%) and **Qwen2-VL-7B** drops 20.0 points (46.6% → 34.1%), suggesting these models rely more heavily on strategies that work in small option spaces — guessing, elimination based on superficial features, or exploiting statistical regularities in 4-option construction.

The **Random Choice baseline drops from 24.9% to 12.8%** and **Frequent Choice from 27.8% to 12.1%** when options expand from 4 to 10. These baselines quantify the guessing advantage that option augmentation eliminates. The fact that even the weakest models (Phi-3.5-Vision at 26.3% Standard 10-option) substantially exceed the 12.1-12.8% guessing range confirms that models are doing genuine content-based reasoning, not just guessing — but the large ∆1 values indicate that guessing and shortcut strategies were contributing non-trivially to original MMMU scores.

**Model rankings shift modestly under option augmentation.** In Table 1, ∆1 is accompanied by rank change indicators. GPT-4o mini gains 1 rank, while InternVL2-Llama3-76B drops 1 rank, suggesting that sensitivity to option-space size is not perfectly correlated with overall capability.

#### Impact of the Vision-Only Setting (∆2)

The difference between Standard 10-option and Vision-only accuracy (∆2 in Table 1) isolates the additional cost of the vision-only input format. **∆2 ranges from −4.3 to −21.8 percentage points** across models, revealing dramatically different levels of robustness to integrated visual-textual processing.

**GPT-4o shows the smallest ∆2 (−4.3 points:** 54.0% → 49.7%), followed by **Claude 3.5 Sonnet (−7.0 points:** 55.0% → 48.0%) and **Gemini 1.5 Pro 0801 (−5.0 points:** 49.4% → 44.4%). These small drops — especially when compared to open-source models — suggest that frontier proprietary models have substantially better capabilities for extracting and reasoning over text embedded in visual scenes.

**The most dramatic ∆2 values appear in mid-tier open-source models.** LLaVA-OneVision-72B drops **14.0 points** (38.0% → 24.0%), VILA-1.5-40B drops **21.8 points** (35.9% → 14.1%), and Idefics3-8B-Llama3 drops **14.5 points** (30.1% → 15.6%). These are catastrophic degradations: for VILA-1.5-40B, moving to the vision-only format more than halves its already-modest accuracy. This suggests fundamental architectural weaknesses in how these models handle the simultaneous demands of visual scene parsing and content reasoning, rather than incremental deficits.

**Model rankings shift dramatically in the vision-only setting.** Table 1 reports rank changes relative to original MMMU standings. VILA-1.5-40B drops **9 ranks** (↑9 in ∆2 notation, indicating a large negative rank change), LLaVA-OneVision-72B drops **5 ranks**, and InternVL2-8B drops **3 ranks**. Conversely, some models maintain or improve relative positions: GPT-4o and Claude 3.5 Sonnet maintain their top positions, and GPT-4o mini gains 1 rank. These rank shifts confirm that the vision-only setting measures a capability dimension distinct from standard multimodal QA performance — models that appear competitive on text-plus-image inputs can collapse when forced into the integrated visual-textual paradigm.

#### Combined Effect: MMMU-Pro vs. Original MMMU

The overall difficulty increase from original MMMU to MMMU-Pro combines question filtering, option augmentation, and the vision-only format. The total decline (implicit in Table 1) ranges from **−16.8% (Claude 3.5 Sonnet) to −26.9% (VILA-1.5-40B)**. The best model (GPT-4o at ~51.9% average) achieves barely half the estimated human expert performance at the High level (85.4%), and even the Low human estimate (73.0%) exceeds all models.

**The decomposition of the total drop varies by model.** For GPT-4o, the total decline of approximately 17.2 points is dominated by option augmentation (−10.7 points, 62% of the decline) with a smaller contribution from the vision-only format (−4.3 points, 25%). For LLaVA-OneVision-72B, the total decline of approximately 25.8 points is driven roughly equally by option augmentation (−18.8 points from 4-option to standard 10-option, though this includes filtering effects) and the vision-only format (−14.0 additional points from standard 10-option to vision). For VILA-1.5-40B, the vision-only format (−21.8 points) accounts for more than half of its total 26.9-point decline, indicating a specific catastrophic weakness in integrated visual-textual processing.

#### Impact of Chain of Thought Prompting

Figure 5 examines the effectiveness of CoT prompting compared to Direct answering across models in both the Standard and Vision Input settings. **CoT generally improves performance, but gains vary substantially by model and setting.**

In the **Standard setting**, CoT provides large gains for some models: Claude 3.5 Sonnet rises from 42.7% Direct to 55.0% CoT (a 12.3-point gain), and GPT-4o improves from approximately 50% Direct to 54.0% CoT. However, some models show minimal benefits (LLaVA-OneVision-72B: approximately 37% Direct to 38.0% CoT) or even **degradation** (VILA-1.5-40B: approximately 37% Direct to 35.9% CoT). The paper attributes CoT failures to "challenges in instruction-following abilities" and "boiled response format problems" — when models struggle to follow the required output format while generating reasoning chains, the attempt at step-by-step reasoning backfires.

In the **Vision Input setting**, the pattern is similar but with generally smaller gains. GPT-4o achieves approximately 49-50% with both Direct and CoT (CoT provides negligible benefit here). Claude 3.5 Sonnet improves from approximately 44% to 48.0%. The limited benefit of CoT in the vision setting aligns with the cognitive load hypothesis: if the model's reasoning capacity is already taxed by the demands of visual scene parsing, adding a CoT requirement may not provide additional benefit and could even be counterproductive.

**Discipline-specific CoT analysis (Table 6, Figure 9) reveals that CoT's benefits are concentrated in structured reasoning domains.** GPT-4o shows a +14.49% gain in Tech and Engineering and +14.66% in Business, but only +1.58% in Art and Design and +2.21% in Humanities and Social Science. LLaVA-OneVision-72B actually **declines 17.12% in Art and Design** with CoT, suggesting that step-by-step reasoning is actively harmful in domains requiring subjective interpretation or holistic visual assessment. These findings underscore that CoT is not a universal enhancer — it helps when problems have clear logical structure and hurts when they require gestalt judgments.

#### OCR Analysis: Text Extraction Is Not the Bottleneck

Table 2 and Figure 6 present the relationship between OCR accuracy and Vision-only task performance. **The key finding is that high OCR accuracy does not translate directly to strong multimodal reasoning.**

Most advanced models achieve high OCR similarity scores: GPT-4o (92.3%), Gemini 1.5 Pro 0801 (89.7%), GPT-4o mini (89.6%), InternVL2-Llama3-76B (88.1%), LLaVA-OneVision-72B (87.8%). Yet their Vision-only task accuracies span a wide range: 49.7%, 44.4%, 35.2%, 38.0%, and 24.0% respectively. LLaVA-OneVision-72B matches InternVL2-Llama3-76B and GPT-4o mini in OCR accuracy (~88%) but achieves substantially lower task accuracy (24.0% vs. 38.0% and 35.2%). Figure 6 visualizes this: while there is a positive correlation between OCR accuracy and task accuracy overall, the relationship is far from deterministic, and models with similar OCR capabilities show dramatically different reasoning performance.

**Explicit OCR prompts do not significantly improve performance** (Table 2). Comparing the "w/ OCR Prompt" and "w/o OCR Prompt" columns shows differences of 0.3 percentage points or less for most capable models: GPT-4o 49.7% vs. 49.4%, InternVL2-Llama3-76B 38.0% vs. 37.9%, LLaVA-OneVision-72B 24.0% vs. 23.8%. Only MiniCPM-V2.6 (24.2% vs. 21.1%) and InternVL2-40B (32.1% vs. 28.9%) show non-trivial improvements, suggesting that for models with weaker base OCR, explicit prompting provides some benefit, but for strong models, OCR is already automated and implicit prompting doesn't enhance it.

**OCR errors account for 0% of annotated GPT-4o errors** in the vision setting (Figure 7). Among 60 annotated error cases, the distribution is: Reasoning Error 46%, Perceptual Error 27%, Lack of Knowledge 25%, Annotation Error 2%, OCR Error 0%. This directly refutes the hypothesis that vision-only failures are primarily due to text extraction failures. The bottleneck lies upstream of OCR — in reasoning, visual interpretation, and knowledge retrieval.

#### Qualitative Error Analysis

Section 3.5 provides qualitative analysis of model failures, identifying several recurring patterns:

**Challenges with increased options** (Figure 11): Models "often select the closest answer rather than arriving at a definitive choice," leading to errors when multiple options are numerically or conceptually adjacent. Conceptually similar options in nuanced questions (e.g., historical interpretations, fine-grained scientific distinctions) cause particular difficulty.

**Increased cognitive load in vision-text integration** (Figures 10, 21): Even when models perfectly extract text from screenshots, they fail to correctly reason about the content. Figure 10 shows GPT-4o in the vision setting producing a "more basic" analysis that "lacks in-depth analysis" compared to the standard setting, despite extracting text accurately. Figure 21 shows a case where "the graph's similar lines and overlapping data points may distract the model from distinguishing between the two unemployment categories." These failures are not about perception or knowledge — they are about managing complexity when multiple information sources compete for attention.

**Overemphasis on visual cues** (Figure 33): In a history question about World War I and II posters, the vision setting model incorrectly chose "League of Nations" by focusing on the World War I image while missing the broader context of World War II and the United Nations. The standard setting model integrated both images correctly. This suggests that when visual information is embedded within the same scene as the question, models can over-weight salient visual elements at the expense of textual reasoning.

**Impact of context switching** (Figure 26): In a math optimization problem, the model correctly defined both the objective function and algebraic constraints, but "due to context switching between the textual description and the geometric figure, it misinterpreted the feasible region." This is a specific failure mode of the vision-only format: the model must alternate between reading text, interpreting diagrams, and reasoning mathematically, and transitions between these modes introduce errors that wouldn't occur if text and image information were presented through separate, stable channels.

#### Response Length Comparison

Figure 8 quantifies a phenomenon observed in qualitative examples: **GPT-4o generates significantly shorter responses in the Vision setting, with a shift from analytical to descriptive content.** In the Standard setting, GPT-4o produces approximately 258 "Descriptive" tokens and 366 "Analytical" tokens (total ~624). In the Vision setting, these shift to approximately 108 Descriptive and 258 Analytical (total ~366). The total drops by roughly 41%, and the proportion of Descriptive content increases (from ~41% to ~30% of a smaller total, meaning the absolute reduction in Analytical tokens is severe — from 366 to 258, a 30% drop).

The paper hypothesizes that "the increased cognition workload of the vision inputs requires the model to focus more on visual processing, which distracts the model from generating extensive reasoning chains." This provides a mechanistic explanation for why the vision-only setting is harder: it's not that the model can't perceive the content, but that the perceptual overhead consumes representational capacity that would otherwise be allocated to reasoning. This is consistent with the error analysis showing reasoning errors as the dominant failure mode (46% in Figure 7).

#### Vision Encoder Impact

Table 3 reports a small experiment comparing two vision encoders within the Cambrian-1 architecture (trained on 1M Cambrian data with Llama 3.1 8B as the LLM backbone). **A self-supervised encoder (DINOv2 ViT-G-14) achieves 17.4% on MMMU-Pro Vision, outperforming a language-supervised encoder (Siglip ViT-SO400M-14) at 16.7%, despite the Siglip encoder performing better on original MMMU (37.9% vs. 37.1%).**

This reversal — the encoder that's better on standard multimodal QA is worse on the vision-only setting — suggests that the vision-only format demands different visual representation properties than standard text-plus-image benchmarks. Self-supervised encoders like DINOv2, trained without language supervision, may learn more general visual features that are beneficial for the scene-parsing demands of the vision-only setting (locating text regions, understanding spatial layout, identifying image boundaries within a complex visual field). Language-supervised encoders like Siglip, which are trained to align visual features with text descriptions, may over-emphasize semantic content at the expense of spatial and structural features that matter for parsing integrated visual-textual scenes. The paper frames this as evidence that "further enhancing visual feature learning while exploring the integration of language-based training objectives with self-supervised training objectives" is a promising direction.

### Ablation Studies and Robustness Checks

**OCR prompt vs. no OCR prompt (Section 3.4, Table 2):** Explicitly prompting models to extract text before solving produces negligible accuracy differences for most models (GPT-4o: 49.7% vs. 49.4%; InternVL2-Llama3-76B: 38.0% vs. 37.9%). Only models with weaker OCR (MiniCPM-V2.6, InternVL2-40B) show non-trivial improvements. This confirms that for capable models, text extraction is automated and explicit prompting doesn't enhance it, supporting the paper's claim that OCR is not the primary bottleneck.

**Vision encoder comparison (Section 4, Table 3):** Replacing a language-supervised vision encoder (Siglip) with a self-supervised one (DINOv2) within the same MLLM architecture (Cambrian-1, Llama 3.1 8B) increases MMMU-Pro Vision accuracy from 16.7% to 17.4%, while slightly decreasing original MMMU accuracy (37.9% to 37.1%). The result is directionally interesting but based on a single model architecture and training setup — the modest absolute difference (0.7 percentage points) makes this a suggestive rather than conclusive finding.

**Direct vs. CoT prompting (Section 3.3, Figure 5):** CoT generally improves performance, but with important exceptions. In the Standard setting, some models show large gains (Claude 3.5 Sonnet: +12.3 points), some show minimal gains (LLaVA-OneVision-72B: ~+1 point), and some show degradation (VILA-1.5-40B: ~−1 point). In the Vision setting, CoT gains are smaller on average. These variations demonstrate that CoT is not uniformly beneficial and that its effectiveness interacts with model architecture, setting difficulty, and disciplinary domain, as shown in Table 6 and Figure 9.

**Text-only LLM accuracy across filtering stages (Section 2.2, Figure 3):** The three-stage comparison (original MMMU, after filtering, after option augmentation) shows progressive decreases in text-only LLM accuracy, validating that the filtering and augmentation steps are effective at removing and mitigating text-only answerable questions. This is a robustness check on the benchmark construction pipeline rather than on model evaluation.

**Human performance approximation sensitivity (Appendix B):** The paper provides Low, Medium, and High estimates of human performance based on different handling of questions without detailed solution processes. The variation across these estimates (73.0% to 85.4%) provides a sensitivity range for the human ceiling, though without confidence intervals the precision of these estimates is unclear.

**Negative result: OCR accuracy does not predict task accuracy (Section 3.4, Figure 6):** The weak correlation between OCR accuracy and vision-only performance is an important negative result that redirects attention from text extraction improvements to integration and reasoning improvements. This is not an ablation in the traditional sense but serves the same function of eliminating a plausible alternative explanation for poor performance.

### Critical Assessment

#### Claim 1: MMMU-Pro provides a more robust and challenging evaluation than MMMU, with performance drops of 16.8% to 26.9% across models.

**This claim is directly and convincingly supported by Table 1.** Every tested model, from GPT-4o to LLaVA-NeXT-13B, shows substantial performance declines on MMMU-Pro compared to original MMMU, with drops ranging from 16.8 to 26.9 percentage points. The decomposition into ∆1 (option augmentation effect) and ∆2 (vision-only effect) provides clear evidence that both interventions contribute to the difficulty increase. The baselines (Random Choice, Frequent Choice) confirm that guessing cannot explain the remaining accuracy, and human estimates show that the questions remain answerable by domain experts.

**However, the paper does not isolate the effect of question filtering on overall difficulty.** The "Standard (4 Opts)" column in Table 1 reports accuracy on the filtered, non-augmented questions — 1,730 questions that survived text-only LLM filtering. This accuracy (e.g., GPT-4o at 64.7%) is compared to the original MMMU validation accuracy (69.1%), but these are on *different question sets* — the MMMU validation set contains questions that were filtered out of MMMU-Pro. The 4.4-point difference (69.1% − 64.7% for GPT-4o) partly reflects the removal of easier (text-only-solvable) questions, but the paper does not compute a clean "filtering-only" effect by evaluating models on the original MMMU questions that *survived* filtering. This makes it impossible to precisely attribute how much of the overall 17.2-point GPT-4o drop comes from filtering versus augmentation versus the vision format.

#### Claim 2: The vision-only input setting challenges models to "see" and "read" simultaneously, revealing limitations in integrated multimodal understanding.

**The claim is supported by the ∆2 metric in Table 1 and the qualitative analysis, but with an important caveat.** The vision-only setting demonstrably reduces performance beyond the standard 10-option format (∆2 ranging from −4.3 to −21.8 points). The error analysis (Figure 7) shows reasoning errors dominating (46%) with 0% OCR errors, and the response length analysis (Figure 8) shows reduced analytical content, consistent with increased cognitive load from integrated processing.

**However, the paper does not include a control condition that would isolate "integrated visual-textual processing" from "additional visual complexity."** The vision-only setting differs from the standard setting in multiple ways simultaneously: (a) text must be extracted from an image rather than received as tokens, (b) the visual field contains text, reference images, *and* answer choices co-located, (c) the screenshots and photos introduce visual artifacts (variable backgrounds, fonts, resolutions), and (d) the model must spatially parse which text elements correspond to the question versus options versus image labels. Any of these factors could drive the performance drop independently of "integrated understanding." A control condition — for example, presenting the model with a screenshot of *only* the question text (no reference images, no options embedded) alongside a separate answer-options text prompt — would help isolate whether the difficulty comes from text-extraction overhead, spatial parsing demands, or genuine cross-modal integration.

The paper's strongest evidence for *integration* specifically (as opposed to scene parsing) comes from the qualitative examples: Figure 10 shows a model extracting text perfectly but reasoning poorly; Figure 26 shows correct constraint definition but incorrect region interpretation due to context switching. These are compelling illustrative cases but represent cherry-picked examples rather than systematic quantification of how often integration versus parsing drives failures.

#### Claim 3: OCR is not the primary bottleneck; explicit OCR prompts do not significantly improve performance.

**This claim is well-supported by multiple lines of evidence.** Table 2 shows negligible differences between with-OCR-prompt and without-OCR-prompt conditions for capable models (differences ≤0.3 points for most). Figure 7 shows 0% OCR errors in annotated GPT-4o failures. Figure 6 shows weak correlation between OCR accuracy and task accuracy among high-OCR models.

**A limitation is that OCR accuracy is measured as whole-text similarity using Levenshtein distance**, which treats all extraction errors equally. A model might correctly extract 92% of characters but misread a critical number or symbol (e.g., "2.5°" vs. "2.5°" extracted as "25°") — the similarity score would remain high, but the task accuracy would suffer. The paper does not break down whether OCR errors cluster in numerically or symbolically critical regions (mathematical expressions, chemical formulas, numerical values) where even small errors would be catastrophic. A more fine-grained OCR accuracy metric that weights errors by their task-relevance would strengthen or potentially complicate this conclusion.

#### Claim 4: CoT prompting generally improves performance, with benefits concentrated in structured reasoning domains.

**This is supported by Figure 5 and Table 6 but underreported.** Figure 5 shows that for most models in the standard setting, CoT improves accuracy, but the paper does not provide the underlying Direct and CoT numbers for all models — only representative ones are shown in the figure. Table 6 provides discipline-specific CoT vs. Direct comparisons for only two models (GPT-4o and LLaVA-OneVision-72B), making it unclear whether the domain-specific patterns (e.g., CoT hurting Art and Design) generalize across model families. A systematic CoT ablation across all models and all six disciplines would substantially strengthen this claim but is not provided.

#### Claim 5: Self-supervised vision encoders may be better suited to the vision-only setting than language-supervised ones.

**This claim is based on a single comparison (Table 3) with a 0.7 percentage point difference** (DINOv2 at 17.4% vs. Siglip at 16.7%). The effect is directionally consistent with the paper's hypothesis, but the small magnitude, single model architecture, single training dataset (1M Cambrian samples), and absence of statistical testing or confidence intervals make this a suggestive observation rather than a robust finding. The paper appropriately frames this as a direction for future work rather than a strong conclusion, but it would benefit from testing additional encoder pairs, training data scales, and LLM backbones to establish robustness.

#### Missing Experiments That Would Strengthen the Paper

The paper would benefit from several additional experiments or analyses:

1. **A "filtering-only" effect calculation:** Report model accuracy on the subset of original MMMU questions that *survive* filtering, in their original 4-option format. This would cleanly separate the effect of question removal from the effect of option augmentation.

2. **A vision-only 4-option condition:** Currently, vision-only always uses 10 options. Evaluating vision-only with 4 options would isolate the vision format effect from the option-count effect in the vision setting, completing the 2×2 design (standard/vision × 4-opt/10-opt) and enabling cleaner decomposition.

3. **Statistical significance testing:** All results are reported as point estimates without confidence intervals. On a 1,730-question test set, differences of 1-2 percentage points may not be statistically significant. Reporting 95% confidence intervals or conducting McNemar's test for paired comparisons would clarify which differences are reliable.

4. **Error analysis beyond GPT-4o:** The 60-case error annotation (Figure 7) covers only GPT-4o. Extending this to 2-3 additional models (one strong proprietary, one strong open-source, one weak open-source) would reveal whether error distributions are consistent across model families or whether different architectures fail in qualitatively different ways.

5. **Per-discipline breakdowns for all models:** Table 1 reports only aggregate scores. Per-discipline breakdowns (like those shown for human estimates in Table 4) would reveal whether certain disciplines drive the overall drops or whether difficulty is uniform. The CoT analysis (Table 6) provides this for two models, suggesting the data exists for others but is not reported.

6. **Latency and cost analysis:** The paper describes MMMU-Pro as "more closely mimicking real-world scenarios," but does not discuss the practical costs of evaluation — API costs for proprietary models, inference time for open-source models, or the resources required for the benchmark construction itself (annotator hours, model API calls for filtering and augmentation).

## 6. Limitations and Trade-offs

### Vision-Only Setting Is Not a Fully Clean Isolation of "Integrated Multimodal Understanding"

**The assumption.** The paper frames the vision-only setting as testing "the seamless integration and switching between visual and textual information" (Section 2.2)—a core human cognitive skill that the standard text-plus-image evaluation format fails to measure. The claim is that performance drops from the standard 10-option setting to the vision-only setting (∆2 in Table 1) reflect the additional cognitive load of integrated visual-textual processing.

**The consequence.** The vision-only setting conflates multiple distinct challenges into a single intervention. When a model receives a screenshot rather than separate text and image inputs, it must simultaneously: (a) extract text from an image via OCR, (b) spatially parse which visual elements correspond to the question, answer choices, and reference images, (c) handle variable visual artifacts (backgrounds, fonts, resolutions, lighting conditions from manual photo capture), and (d) integrate the extracted text with embedded diagrams to reason about the answer. Any of these factors—separately or in combination—could drive the observed performance drop without requiring genuine "integrated understanding."

The paper's evidence that OCR specifically is not the bottleneck (Section 3.4, Figure 7 showing 0% OCR errors) rules out factor (a) but does not disentangle (b), (c), and (d). A model might fail in the vision-only setting because it cannot reliably parse which text in a complex visual scene corresponds to the question versus the options (a spatial reasoning and layout understanding problem), not because it cannot integrate text with reference images. This distinction matters for directing research investment: if spatial parsing is the bottleneck, the solution is improved document layout understanding and visual grounding; if integration is the bottleneck, the solution is architectural changes to cross-modal attention and feature fusion under load.

**What evidence exists in the paper.** The paper's strongest integration-specific evidence comes from qualitative examples. Figure 10 shows GPT-4o perfectly extracting text from a screenshot but producing "more basic" analysis with "higher likelihood of errors" compared to the standard setting—suggesting integration rather than parsing failure. Figure 26 shows a math problem where the model "correctly defined both the objective function and the algebraic constraints" but "due to context switching between the textual description and the geometric figure, it misinterpreted the feasible region." These cases are compelling but are cherry-picked illustrative examples, not systematic quantification.

The response length analysis (Section 3.7, Figure 8) provides indirect support: GPT-4o produces fewer analytical tokens in the vision setting, consistent with cognitive load from simultaneous visual and textual processing. However, this could equally be explained by the model spending more tokens on descriptive scene parsing (which Figure 8 confirms: the proportion of descriptive tokens increases), leaving fewer tokens available for reasoning—this is a capacity allocation issue, not necessarily an integration failure.

**Mitigation status.** The paper does not include a control condition that would isolate "integrated understanding" from "visual scene parsing complexity." A natural control would be a vision-only input containing *only* the question text and options (no reference images embedded), compared against the standard text-input format. This would measure the pure overhead of text-from-image extraction plus spatial parsing, without any cross-modal integration demand. The difference between this control and the full vision-only setting would then isolate the integration-specific cost. Without such a control—or any systematic decomposition of which visual-scene factors drive the ∆2 drop—the paper's central claim that the vision-only setting tests "integrated multimodal understanding" specifically, rather than a bundle of vision-in-the-loop challenges generally, remains an interpretation rather than an empirically established fact. The authors do not acknowledge this ambiguity as a limitation; it is left implicit in the experimental design.

---

### Difficulty Estimation Has No Mechanism for Clean Attribution to Underlying Causes

**The assumption.** The paper builds MMMU-Pro through a three-step pipeline (filtering, option augmentation, vision-only conversion) and presents the performance differences across settings (∆1, ∆2) as diagnostically informative—isolating the impacts of option-space size and vision-only format respectively. The assumption is that these settings create clean, independent measurements of distinct capability deficits, enabling researchers to identify *which* aspects of their models need improvement.

**The consequence.** The three evaluation settings are not experimentally orthogonal. The Standard 4-option setting uses the original 4-option questions *after filtering*—it evaluates on a different question set (the 1,730 that survived filtering) than the original MMMU validation set (which included text-only-solvable questions). The ∆1 comparison (Standard 4-option vs. Standard 10-option) is clean because the same 1,730 questions are evaluated with 4 and 10 options respectively. But comparing either of these to the original MMMU validation accuracy (as in Table 1's ∆1 and ∆2 columns) mixes the effect of question filtering with the effect of option augmentation. For GPT-4o, the original MMMU accuracy is 69.1% on the full validation set; the Standard 4-option accuracy is 64.7% on the filtered subset. This 4.4-point gap partly reflects the removal of easier questions, but the paper never reports a "filtering-only" effect—what accuracy models achieve on the subset of original MMMU questions that *survive* filtering, in their original 4-option format. This makes it impossible to cleanly attribute how much of the overall performance drop comes from harder questions versus harder option formats.

Furthermore, the vision-only setting always uses 10 options. There is no vision-only 4-option condition. The ∆2 comparison (Standard 10-option vs. Vision-only) therefore isolates the vision format effect *only in the 10-option regime*. A model's ∆2 might differ substantially if vision-only were tested with 4 options, but the paper provides no way to know. The 2×2 design (standard/vision × 4-opt/10-opt) is incomplete, making the diagnostic decomposition approximate rather than exact.

**What evidence exists in the paper.** Table 1 reports all three settings, and the ∆1 and ∆2 columns document the performance differences. The paper does not acknowledge that the filtering effect is entangled with the augmentation effect in these comparisons, nor does it discuss the missing 2×2 experimental design as a limitation. The text simply presents ∆1 as "Standard (10 options) - MMMU (Val)" and ∆2 as "Vision - MMMU (Val)" without noting that these compare across different question sets (MMMU validation includes filtered-out questions; MMMU-Pro settings do not).

The paper's reporting of rank changes (↑ and ↓ arrows in Table 1) implicitly assumes that ∆1 and ∆2 capture distinct capability dimensions—otherwise, rank-change analysis would be uninformative. The fact that some models drop many ranks on ∆2 but few on ∆1 (e.g., VILA-1.5-40B: 2 ranks on ∆1, 9 ranks on ∆2) is suggestive that the dimensions are partially distinct, but without orthogonal settings this remains an interpretation.

**Mitigation status.** Not addressed. The paper would benefit from: (a) reporting model accuracy on the filtered question subset in its original 4-option format to isolate the filtering effect, (b) including a vision-only 4-option condition to complete the 2×2 design. These are feasible with the existing data (the questions exist; they were simply not evaluated in vision-only 4-option format) and would substantially strengthen the diagnostic claims without requiring new annotation. The paper does not acknowledge the incomplete decomposition as a limitation.

---

### Human Performance Estimates Are Approximated, Not Directly Measured—and the Key Assumption Is Unverified

**The assumption.** Rather than conducting a new human evaluation on MMMU-Pro, the paper approximates human expert performance by adapting data from the original MMMU human evaluation (Section 3.1, Appendix B). The central assumption enabling this approximation is:

> "human experts, with their innate ability to seamlessly integrate visual and textual information, are expected to perform similarly in the vision-only input setting as they do in the original format"

The paper assumes that the vision-only format, which causes substantial performance drops in AI models (∆2 up to −21.8 points), imposes negligible difficulty on humans because human visual cognition naturally handles integrated text-and-image scenes.

**The consequence.** If this assumption is wrong—if human experts also perform worse on vision-only inputs (due to OCR-like overhead, spatial parsing demands, font/background variability, or the additional cognitive load of reading from a photo rather than clean text)—then the reported human-to-model gap is systematically overestimated for the vision-only setting and underestimated for the standard setting. The human performance estimates in Table 1 (73.0% Low, 80.8% Medium, 85.4% High) serve as the ceiling against which model performance is judged. If the true human vision-only accuracy is, say, 5-10 points lower than these estimates, the models' vision-only performance (GPT-4o at 49.7%) would be substantially closer to the human ceiling, and the paper's narrative of "significant drops across all tested models" while humans remain largely unaffected would need to be qualified.

The assumption is not obviously true. Human experts taking a test from a screenshot or photograph—with variable fonts, backgrounds, and potential visual artifacts—may well experience some cognitive overhead compared to reading cleanly typeset text. Visual fatigue from reading extended text on screens, difficulty parsing small fonts in photographs, and the extra step of mentally separating the question text from surrounding visual elements are all plausible sources of human performance degradation. The paper provides no empirical evidence that these effects are negligible for domain experts answering college-level questions.

**What evidence exists in the paper.** None. The paper does not conduct even a small-scale validation of the assumption—for example, having a subset of experts answer a few questions in both standard and vision-only formats to compare. The approximation method (Appendix B, Equation 2) adjusts for guessing on questions without detailed solution processes but does not adjust for any modality-specific difficulty of the vision-only format. The paper explicitly acknowledges this in its Limitations section:

> "our reliance on approximated human performance rather than direct evaluation introduces potential biases in reporting accurate human expert performance"

This acknowledgment is honest but does not address the specific, strongest assumption—that vision-only inputs are no harder for humans than standard inputs—which is more consequential than general "potential biases" from approximation.

**Mitigation status.** The paper treats this as a practical tradeoff ("conducting such an assessment is both time-consuming and costly") and provides conservative estimates (floor function, random guessing for non-solution questions) to avoid overestimating human performance. But conservatism in one direction (avoiding overestimation) does not address the directional bias introduced by the unverified assumption (potential overestimation specifically for the vision-only setting). The paper suggests no mitigation for the assumption itself and does not frame it as a specific limitation beyond the general approximation caveat. A minimal validation—e.g., having 5-10 experts answer 20-30 questions in both formats—would cost little relative to the benchmark construction effort and would either validate or bound the error from this assumption.

---

### The Vision-Only Construction Methodology Introduces Uncontrolled Visual Variability That May Confound Measurement

**The assumption.** Section 2.2 describes the vision-only construction process:

> "We ask the human annotators to manually capture photos and screenshots over a simulated display environment. This process involves varying the backgrounds, font styles, and font sizes to replicate the diversity of real-world conditions."

The paper assumes that this variability is a feature—"ensuring that the models are not only challenged by the integration of text and images but also by the variability in how this content is presented"—and that the effect of visual variability on model performance is (a) consistent with real-world deployment conditions and (b) not confounded with question difficulty or disciplinary domain in ways that distort measurement.

**The consequence.** If visual variability (backgrounds, fonts, resolutions, lighting) is not randomly or systematically distributed across questions, it introduces confounding. For example, if certain disciplines tend to receive particular visual treatments (e.g., History questions captured with parchment-like backgrounds, Computer Science questions with code-friendly monospace fonts), then discipline-specific performance differences may partly reflect visual-presentation difficulty rather than content difficulty. If some questions happen to be captured with higher resolution or better lighting than others (manual photo capture introduces inevitable variability), then question-level difficulty estimates are contaminated by visual-quality noise.

More fundamentally, the paper provides no characterization of how visual variability affects model performance. It may be that particular models are robust to font changes but sensitive to background complexity, or vice versa. Without measuring these effects, the vision-only setting measures an unknown mixture of content difficulty and visual-presentation difficulty, and differences between models may partly reflect sensitivity to visual artifacts rather than differences in multimodal reasoning capability.

The paper's argument that visual variability "replicates the diversity of real-world conditions" is reasonable as a design goal, but real-world conditions introduce variability that is correlated with content: a user uploading a photo of a textbook page captures different visual qualities than a user uploading a screenshot of a web page, and the discipline and question type predict which format is more likely. MMMU-Pro's manual capture process may introduce variability that is *not* ecologically valid—backgrounds and fonts chosen by annotators for diversity rather than representativeness—and that therefore tests model robustness to a distribution of visual presentations that does not match any particular deployment scenario.

**What evidence exists in the paper.** Very little. The paper provides qualitative examples of vision-only inputs (Figure 4) showing varied presentations but does not quantify visual diversity across the dataset, report inter-annotator consistency in capture quality, or analyze whether visual features predict model errors. The OCR accuracy analysis (Table 2) shows that most models extract text well despite visual variability, but this measures text recognition, not the effect of visual presentation on reasoning quality. The paper's error analysis (Figure 7) categorizes errors as reasoning, perceptual, knowledge, annotation, and OCR—but "perceptual error" groups together misinterpretation of diagram content and misinterpretation caused by visual presentation artifacts, making it impossible to separate content-perception failures from presentation-perception failures.

**Mitigation status.** Not addressed. The paper does not report any analysis of how visual variability correlates with model performance, does not test whether models are differentially sensitive to presentation factors, and does not acknowledge the potential confounding as a limitation. This is particularly notable given recent work on MLLMs' sensitivity to visual presentation (Tong et al., 2024b, "Eyes Wide Shut? Exploring the Visual Shortcomings of Multimodal LLMs," which the paper cites in a different context) that demonstrates that seemingly superficial visual changes can significantly affect model outputs. The paper's own qualitative examples hint at this: Figure 21 shows a case where "the graph's similar lines and overlapping data points may distract the model"—a visual presentation issue rather than a content understanding issue. But the paper does not systematically investigate how much of the vision-only performance drop is attributable to visual presentation variability versus genuine multimodal integration difficulty.

---

### Single Benchmark, Single Question Format, and No Cross-Task Generalization Evidence

**The assumption.** All experiments are conducted on MMMU-Pro, which is constructed entirely from MMMU questions—college-level multiple-choice questions across 30 subjects in six disciplines. The paper implicitly assumes that the findings about model behavior (performance drops under option augmentation, vision-only difficulty, CoT effectiveness variation) generalize to other multimodal reasoning tasks and benchmarks. It also assumes that the specific question format (multiple-choice with exactly one correct answer) captures the relevant dimensions of multimodal understanding.

**The consequence.** The paper's central claims—that MMMU-Pro "rigorously assesses multimodal models' true understanding and reasoning capabilities" (Abstract) and that the observed performance drops reveal limitations in "true multimodal understanding"—are supported only for the specific task of answering college-level multiple-choice questions with a single correct answer. This is a narrow slice of multimodal reasoning. Many real-world multimodal tasks require open-ended generation (describing a diagram, explaining a scientific figure), multi-step interactive reasoning (iteratively refining an interpretation based on new visual evidence), or subjective judgment (evaluating design quality, interpreting artistic intent). A model might perform poorly on MMMU-Pro's vision-only multiple-choice questions while excelling at open-ended multimodal explanation tasks, or vice versa.

The paper's findings about CoT effectiveness (Section 3.3) already demonstrate task-format sensitivity within MMMU-Pro: CoT helps in Tech and Engineering (+14.49% for GPT-4o) but hurts in Art and Design (−17.12% for LLaVA-OneVision-72B). This discipline-level variation within the same benchmark suggests that findings are unlikely to transfer uniformly across different task formats and domains—but the paper provides no evidence about transfer at all.

The reliance on multiple-choice format introduces a specific vulnerability that the paper does not discuss: models may learn to exploit structural features of multiple-choice questions (option length, phrasing patterns, position biases) that are orthogonal to multimodal understanding. The option augmentation from 4 to 10 choices reduces but does not eliminate this vulnerability. A model achieving 54.0% on Standard 10-option (GPT-4o) is still operating in a multiple-choice regime where elimination strategies and option-comparison heuristics can contribute to accuracy. Performance on open-ended versions of the same questions—where the model must generate the answer rather than select it—might be substantially lower, but the paper provides no such comparison.

**What evidence exists in the paper.** The paper's main results (Table 1) are from a single benchmark. The discipline-level breakdowns that would reveal cross-domain consistency are not reported for all models—only CoT vs. Direct comparisons for GPT-4o and LLaVA-OneVision-72B appear in Table 6 and Figure 9. The human performance estimates (Table 4) provide discipline-level baselines showing that human accuracy varies by domain (from ~63.6% for Humanities and Social Science to ~84.9% for Science at Medium level), but this variation in human performance could mask or interact with model-level variation in ways not analyzed.

**Mitigation status.** The paper does not claim cross-task generalization and does not attempt to validate findings on other multimodal benchmarks. The Limitations section does not mention the single-benchmark scope as a limitation. The paper's framing in Section 1—that the vision-only setting "tests a fundamental human cognitive ability" and "closely mimics real-world scenarios"—implicitly claims broader relevance than a single benchmark can support, but this is a rhetorical move rather than an empirical claim. The practical consequence is that a practitioner deciding whether MMMU-Pro performance predicts deployment performance in their specific multimodal application has no evidence to guide that decision.

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
