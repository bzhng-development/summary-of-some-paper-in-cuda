# The Prompt Report: A Systematic Survey of Prompt Engineering Techniques

**ArXiv:** [2406.06608](https://arxiv.org/abs/2406.06608)

## 🎯 Pitch

This paper delivers the most comprehensive, evidence-based synthesis of prompt engineering to date by conducting a PRISMA-guided systematic review that maps 58 text-based and 40 multimodal/multilingual prompting techniques into an actionable taxonomy, paired with a unified vocabulary. By benchmarking prompting strategies and providing empirical case studies—including high-stakes domains like suicidality detection—it empowers both practitioners and researchers to navigate and standardize best practices in a field whose inconsistent terminology and scattered methods have previously impeded reliable, secure, and effective AI deployment.

---

## 1. Executive Summary

This paper presents a comprehensive systematic survey of prompt engineering techniques, assembling a taxonomy of 58 text-based prompting techniques and 40 techniques for other modalities through a PRISMA-grounded literature review of 1,565 papers. It provides a standardized vocabulary of 33 terms, a meta-analysis of technique usage across the literature, and two case studies—one benchmarking six prompting techniques on MMLU using GPT-3.5-turbo (finding that Few-Shot CoT achieves the highest accuracy at ~69% while Zero-Shot-CoT unexpectedly drops below the Zero-Shot baseline), and another illustrating the manual prompt engineering process on a real-world suicidal crisis detection task where an expert prompt engineer improved from 0% to an F1 of 0.53 across 47 development steps. The paper further introduces a detailed organizational framework dividing techniques into categories including In-Context Learning, Thought Generation (e.g., Chain-of-Thought prompting appending "Let's think step by step"), Decomposition (e.g., Least-to-Most breaking problems into sub-problems), Ensembling (e.g., Self-Consistency aggregating multiple reasoning paths), Self-Criticism (e.g., Self-Refine iteratively improving answers), and extends the taxonomy to multilingual, multimodal, agent-based, and evaluation prompting, while also cataloging security threats—particularly prompt injection and jailbreaking—and alignment concerns such as prompt sensitivity where minor formatting changes can cause accuracy to vary from near 0 to 0.804 on some tasks, establishing that prompt engineering remains a "black art" characterized by extreme sensitivity to seemingly irrelevant prompt details without any obvious reason those details should matter.

## 2. Context and Motivation

### The Core Problem: Prompt Engineering Is a Fragmented, Poorly Understood Discipline

The fundamental challenge this paper addresses is that prompt engineering—the practice of designing inputs to generative AI models to guide their outputs toward desired behaviors—has emerged as a critically important skill without a correspondingly coherent intellectual foundation. Despite the widespread deployment of transformer-based LLMs across consumer, research, and industrial settings (Bommasani et al., 2021), the field lacks three essential elements that would normally characterize a mature technical discipline: a standardized terminology, a systematic taxonomy of techniques, and empirical guidance on when to use which approach.

The paper frames this gap through a telling observation in Section 1: "prompting is an emerging field, the use of prompts continues to be poorly understood, with only a fraction of existing terminologies and techniques being well-known among practitioners." This is not merely an academic concern about tidy definitions. The absence of shared vocabulary means that researchers and practitioners cannot cleanly communicate about their methods—one paper's "role prompt" is another's "persona prompt" (Section 1.2), and the term "prompt" itself has shifted meaning from referring only to the input text in Brown et al. (2020)—who distinguished between the "prompt" (e.g., "llama") and the "task description" (e.g., "Translate English to French:")—to encompassing the entire string passed to the LLM, including instructions and exemplars.

This fragmentation manifests concretely in several ways the paper identifies:

- **Conflicting definitions**: The authors analyzed definitions from across the literature (Appendix A.1, Table A.1) and found substantial variation in how even the most basic terms—"prompt" and "prompt engineering"—are used. Some papers define prompt engineering as "the practice of designing, refining, and implementing prompts or instructions" (Meskó, 2023), while others emphasize it as "the process of creating a prompting function that results in the most effective performance" (Liu et al., 2023b). These definitional differences are not just semantic quibbles—they reflect fundamentally different conceptions of what the activity entails, which in turn shapes how research is conducted and evaluated.

- **Redundant or overlapping techniques**: Because there was no central taxonomy, researchers often reinvent similar ideas under different names. The paper notes this directly in its Conclusions: "there are sure to be gaps and redundancies." For example, the technique of assigning a specific role to the GenAI is called "Role Prompting" by Wang et al. (2023j) and Zheng et al. (2023d) but "Persona Prompting" by Schmidt et al. (2023) and Wang et al. (2023l). Without a common vocabulary, it becomes difficult to compare results across papers or build cumulatively on prior work.

- **Unknown coverage of the technique space**: Practitioners have no clear map of what techniques exist, let alone which ones work best under what conditions. The paper notes that only a small subset of prompting techniques is commonly used in research and industry (Section 2.3), and that many potentially valuable approaches remain obscure simply because they are hard to discover in a field without systematic organization.

### Why This Problem Matters: Practical and Theoretical Stakes

The paper establishes the importance of addressing this fragmentation along several dimensions that together argue for the urgency of the work.

**The practical economic stakes are enormous.** As the paper notes in Section 1, "Empirically, better prompts lead to improved results across a wide range of tasks." This is not a matter of marginal improvements. The citation to Wei et al. (2022b) references Chain-of-Thought prompting, which demonstrated that simply appending "Let's think step by step" to a prompt substantially improved reasoning performance—without any model retraining, architectural changes, or additional computational cost at training time. Similarly, the paper's own benchmarking case study (Section 6.1) shows that moving from a Zero-Shot baseline (0.627 accuracy on MMLU) to Few-Shot CoT (0.692 accuracy) represents a meaningful gain achieved purely through prompt design, using the same underlying model (gpt-3.5-turbo).

When scaled across the millions of prompts processed daily by deployed systems, such differences translate into substantial real-world impact: better medical advice generation, more accurate code synthesis, more reliable information retrieval, and so on. Yet because the field lacked systematic guidance on which techniques to use when, most practitioners operate well below the achievable performance ceiling for their models—not because the models are incapable, but because the prompts are suboptimal.

**The theoretical significance lies in understanding what LLMs actually are.** Prompting serves as a window into model capabilities. Section 1.3 traces the history of prompts from their precursors—control codes in earlier language models (Pfaff, 1979; Poplack, 1980; Keskar et al., 2019)—through their emergence in the GPT era. The shift from fine-tuning to prompting as the primary interface for model interaction represents a fundamental change in how we relate to AI systems. Rather than retraining models for each task, we now specify tasks through natural language. But this means that understanding prompt engineering is understanding the nature of the interface itself—what models can and cannot do, how they respond to different forms of instruction, and where their capabilities break down.

The paper draws attention to this through its framing of prompting as inheriting "many of the standard challenges of linguistic communication—e.g., ambiguity, the role of context, the need for course correction—while at the same time adding the challenge of communicating with an entity whose 'understanding' of language may not bear any substantial relationship to human understanding" (Section 8). This is not merely a practical engineering problem but a deep question about the nature of these systems.

**The security implications are severe.** Section 5.1 documents a threat landscape where prompt injection and jailbreaking remain fundamentally unsolved problems. The paper cites examples where attackers extracted training data from ChatGPT by prompting it to repeat the word "company" indefinitely (Nasr et al., 2023), where chatbots were manipulated into selling products for $1 (Bakke, 2023), and where an airline chatbot gave legally binding incorrect information about refunds (Garcia, 2024). These are not hypothetical threats—they are documented incidents with real financial and legal consequences. The fact that prompt-based defenses were found to be insufficient in large-scale testing (Schulhoff et al., 2023, which the paper cites as finding that "no prompt-based defense is fully secure") underscores that the field needs systematic understanding to even begin addressing these vulnerabilities.

### Where Prior Approaches Fall Short

The paper positions itself against existing survey and organizational work, identifying specific limitations in each.

**Pre-ChatGPT surveys are now substantially outdated.** Liu et al. (2023b) performed a systematic review of prompt engineering, but the paper notes this was done "in the pre-ChatGPT era." The landscape changed dramatically with the release of ChatGPT and subsequent models, which brought prompting into mainstream usage and spawned an explosion of new techniques. The paper's citation analysis (Figures 2.9, 2.10) shows that the most heavily cited prompting papers and benchmarks are predominantly from 2022-2023, meaning a pre-ChatGPT survey necessarily misses the bulk of current techniques.

**Existing surveys are either narrow in scope or lack systematic methodology.** The paper reviews multiple prior surveys (Section 7) and identifies specific gaps:

- **Chen et al. (2023a)** provide a review of popular prompting techniques like CoT, Tree-of-Thought, and Self-Consistency, but the paper characterizes this as covering only a small subset of the technique space rather than providing comprehensive coverage.

- **White et al. (2023)** and **Schmidt et al. (2023)** offer pattern catalogs organized similarly to software patterns. While useful for practitioners, these are more prescriptive than descriptive—they organize techniques into reusable templates but do not provide a systematic taxonomy grounded in a broad literature review.

- **Santu and Feng (2023)** provide a general taxonomy, but the paper notes this is oriented toward designing prompts with specific properties rather than cataloging the full range of existing techniques.

- **Domain-specific surveys** (Meskó, 2023 for medical; Heston and Khun, 2023 for medical education; Hou et al., 2023 for software engineering; Wang et al., 2023c for visual modality) provide depth in particular application areas but do not attempt cross-domain synthesis.

**No prior survey uses a formal systematic review methodology.** The paper explicitly distinguishes itself by emphasizing its grounding in PRISMA (Page et al., 2021), "the widely used standard for systematic literature reviews" (Section 7). The authors note that "unlike many works that claim to be systematic, we base our work in the widely used standard for systematic literature reviews—PRISMA." This methodological rigor matters because it means the taxonomy is derived from a reproducible, documented process rather than the authors' impressions of the literature. The paper describes a multi-stage pipeline (Section 2.1) involving 44 keywords, 4,247 unique records after deduplication, human annotation of 1,661 papers with 92% inter-annotator agreement (Krippendorff's α = Cohen's κ = 81%), LLM-assisted classification with 89% precision and 75% recall, and a final set of 1,565 papers. This gives the resulting taxonomy an empirical grounding that prior surveys lack.

**Existing terminology efforts are incomplete.** The paper notes that terminology "is rapidly developing" with "many poorly understood definitions (e.g. prompt, prompt engineering) and conflicting ones (e.g. role prompt vs persona prompt)" (Section 1.2). Prior work either used terms inconsistently or focused on narrow subsets. The paper integrates multiple definitions (Appendix A.1) to derive representative ones, creating a vocabulary of 33 terms that spans prompting, prompt engineering, fine-tuning, and orthogonal prompt types.

**The "black art" nature of prompt engineering is undocumented.** Perhaps most significantly, the paper identifies that the actual process of prompt engineering—how an experienced practitioner iteratively improves a prompt on a real problem—had never been documented in the literature. Section 6.2 presents what the authors describe as the first annotated case study of manual prompt engineering, tracking 47 development steps over approximately 20 hours of work on a suicide risk detection task. This addresses a gap where "the literature does not yet include detailed guidance on the process" of prompt engineering itself, as distinct from the techniques that result from that process.

### How This Paper Positions Itself

The paper frames itself not as proposing new techniques but as providing essential infrastructure for the field. Several positioning decisions are explicit:

**Observational, not prescriptive.** The authors state their stance is "primarily observational, and we make no claims to the validity of the presented techniques" (Section 8). This is important because it distinguishes the paper from technique-proposing papers that advocate for specific approaches. The goal is comprehensive cataloging, not ranking or recommending.

**Building a foundation, not a final product.** The paper explicitly describes itself as "an initial attempt to categorize the species of an unfamiliar territory" and "a first iteration of terminologies that will develop over time" (Section 1). This acknowledges the rapid pace of the field and positions the taxonomy as something to be extended rather than as a definitive classification.

**Bridging research and practice.** By providing both a systematic academic review and practical elements like the prompt engineering case study and benchmarking results, the paper positions itself as useful to both researchers (who need the taxonomy and meta-analysis) and practitioners (who need the technique catalog and process guidance). The case study in particular is described as providing "one illustration of how an experienced prompt engineer would approach a task like this, along with lessons learned" (Section 6.2), filling a gap between theoretical knowledge about techniques and practical knowledge about how to apply them.

**Explicitly scoped for manageability.** The paper makes several deliberate scope decisions that distinguish it from prior surveys (Section 1): it focuses on hard (discrete) prompts rather than soft (continuous) prompts, on prefix prompts rather than cloze prompts (since "modern LLM transformer architectures widely employ prefix prompts"), and on task-agnostic rather than task-specific techniques. It also excludes papers that use gradient-based updates (fine-tuning), keeping the scope focused on inference-time techniques only. These decisions keep the work "approachable to less technical readers and maintain a manageable scope" while still covering substantial ground.

**Connecting to broader AI concerns.** The paper embeds prompting within larger discussions of security (Section 5.1), alignment (Section 5.2), evaluation (Section 4.2), and multimodality (Section 3.2), positioning prompt engineering not as an isolated activity but as one that intersects with essentially every concern in deployed AI systems. This is reinforced by the interconnected categories diagram (Figure 1.1), which shows security, safety, and evaluation needs surrounding all prompting activities.

## 3. Technical Approach

This is primarily a **taxonomic and survey paper** whose core idea is that the fragmented landscape of prompt engineering can be organized into a coherent framework through systematic literature review, standardized terminology, and hierarchical categorization of techniques—enabling practitioners and researchers to navigate the technique space, understand relationships between methods, and make informed decisions about which prompting approaches to apply.

### 3.1 Reader Orientation

The paper constructs a **field-level organizational system** for prompt engineering: a standardized vocabulary, a taxonomy of techniques, and empirical guidance derived from a systematic review of 1,565 papers. Unlike a typical methods paper that proposes a single new technique, this work solves the meta-problem of making the entire technique space navigable—providing the intellectual infrastructure (definitions, categories, relationships, benchmarks, and process documentation) that the field has lacked since its emergence.

### 3.2 Big-Picture Architecture (Diagram in Words)

The system described in this paper has six major components, each performing a distinct function in organizing and analyzing the prompt engineering landscape:

1. **Systematic Literature Review Pipeline** (Section 2.1): Ingests papers from arXiv, Semantic Scholar, and ACL using 44 keywords; filters through human annotation (1,661 papers, 92% inter-annotator agreement, Krippendorff's α = Cohen's κ = 81%) and LLM-assisted classification (GPT-4-1106-preview, 89% precision, 75% recall, F1 of 81%); produces a final dataset of 1,565 papers stored on HuggingFace with an accompanying datasheet (Appendix A.3).

2. **Terminology Standardization Module** (Sections 1.2, Appendix A.2): Aggregates conflicting definitions across the literature (Appendix A.1, Table A.1), derives representative consensus definitions for 33 vocabulary terms, and disambiguates terms into orthogonal classification dimensions (originator, hard vs. soft, prediction style).

3. **Taxonomy Construction Engine** (Section 2.2, Figures 2.2, 3.1, 3.2, 4.1, 4.2, 5.1, 5.2): Organizes techniques into hierarchical categories—58 text-based techniques across 6 major categories (In-Context Learning, Thought Generation, Decomposition, Ensembling, Self-Criticism, and additional categories for multilingual, multimodal, agent, evaluation, security, and alignment techniques) based on functional relationships and shared mechanisms.

4. **Meta-Analysis and Visualization System** (Section 2.3, Figures 2.9, 2.10, 2.11): Extracts model names, dataset names, and citation patterns from the paper corpus using GPT-4-1106-preview followed by manual filtering; produces quantitative measures of technique usage, benchmark popularity, and model adoption by counting citations within the dataset.

5. **Benchmarking Harness** (Section 6.1): Runs a controlled comparison of six prompting techniques (Zero-Shot, Zero-Shot-CoT, Zero-Shot-CoT with Self-Consistency, Few-Shot, Few-Shot-CoT, Few-Shot-CoT with Self-Consistency) on 2,800 MMLU questions using GPT-3.5-turbo, systematically varying base instructions, question formats, and thought inducers to measure the impact of technique choice.

6. **Case Study Documentation Framework** (Section 6.2): Tracks the complete prompt engineering process on a real-world suicide risk detection task—47 development steps over ~20 hours, progressing from 0% usable responses to an F1 of 0.53 (0.86 precision, 0.38 recall)—providing the first annotated record of how an expert prompt engineer iteratively improves prompts.

Information flows through these components as follows: the systematic review produces a paper corpus → papers are then classified and organized into the taxonomy → simultaneously, terminology is standardized by aggregating definitions → the taxonomy and terminology feed into the meta-analysis (which quantifies technique usage) and the benchmarking harness (which empirically tests selected techniques) → the case study documents the manual process of applying these techniques to a real problem, illustrating the gap between knowing techniques and effectively deploying them.

### 3.3 Roadmap for the Deep Dive

- **First**, the systematic review pipeline (Section 2.1)—how papers were collected, filtered, and classified—because every technique in the taxonomy derives from this empirical foundation, and understanding the methodology is essential to assessing the taxonomy's coverage.
- **Second**, the terminology standardization framework (Sections 1.1, 1.2, Appendices A.1, A.2)—because clear definitions are prerequisite to meaningful taxonomy, and the paper's resolution of conflicting terminology is itself a technical contribution.
- **Third**, the taxonomy construction logic (Section 2.2)—how the 58 text-based techniques were organized into categories, what defines each category boundary, and why certain techniques map to certain categories—because this is the paper's central organizational contribution.
- **Fourth**, the meta-analysis methodology (Section 2.3)—the specific procedure for extracting model, dataset, and citation data from the corpus and generating the quantitative figures (2.9, 2.10, 2.11)—because these provide empirical grounding for claims about technique prevalence.
- **Fifth**, the benchmarking setup (Section 6.1)—the exact prompt templates, question formats, evaluation logic, and hyperparameter choices used in the MMLU comparison—because this is the paper's primary empirical evidence about technique effectiveness.
- **Sixth**, the case study methodology (Section 6.2)—the step-by-step process the expert prompt engineer followed, including the specific prompts tried, the extractors used, and the AutoDiCoT algorithm—because this illustrates the practical reality of prompt engineering that the taxonomy alone cannot capture.

### 3.4 Detailed, Sentence-Based Technical Breakdown

#### Systematic Literature Review Pipeline (Section 2.1)

The paper's entire taxonomy rests on a systematic review process grounded in the PRISMA 2020 statement (Page et al., 2021), which is "the widely used standard for systematic literature reviews" (Section 7). The pipeline has four stages: querying, deduplication, human review, and LLM-assisted classification.

**Stage 1: Database querying.** The authors query three sources—arXiv, Semantic Scholar, and ACL—using a list of 44 keywords "narrowly related to prompting and prompt engineering" (Appendix A.4). The keywords cover technique names (e.g., "jailbreak prompt," "few-shot learning," "chain-of-thought prompting"), modalities (e.g., "multimodal prompting"), and meta-concepts (e.g., "prompt optimization," "prompt engineering"). The initial retrieval yields 4,797 records: 3,677 from arXiv, 2,087 from Semantic Scholar, and 639 from ACL.

**Design choice—why 44 keywords and these three sources**: The keywords were designed to cast a wide net within the prompt engineering space while excluding adjacent topics (e.g., fine-tuning, soft prompting) through the absence of related terms. arXiv and ACL are the primary repositories for NLP and AI research; Semantic Scholar provides additional coverage through its citation graph. The combination ensures both breadth and academic rigor, though the paper acknowledges the possibility that "some papers were not included due to a translation not being available" (datasheet, Appendix A.3.5), meaning the review is limited to English-language papers.

**Stage 2: Title deduplication.** After initial retrieval, 550 duplicate records are removed, leaving 4,247 unique records. The paper does not specify the deduplication algorithm, suggesting a simple exact-match or near-match on paper titles.

**Stage 3: Human annotation.** From the 4,247 records, human annotators manually label a sample of 1,661 articles from the arXiv set against four inclusion/exclusion criteria:

1. **Include** if the paper proposes a novel prompting technique.
2. **Include** if the paper strictly covers hard prefix prompts.
3. **Exclude** if the paper focuses on training by backpropagating gradients (i.e., fine-tuning, not prompt engineering).
4. **Include** if the paper uses a masked frame and/or window for non-text modalities.

A subset of 300 articles is independently reviewed by two annotators to measure reliability. The paper reports 92% agreement, Krippendorff's α = 0.81, and Cohen's κ = 0.81. The remaining 1,361 articles are reviewed by a single annotator.

**Why these four criteria**: Criterion 1 ensures the paper focuses on prompting techniques (not, e.g., model architecture). Criterion 2 enforces the scope decision to focus on hard prefix prompts (Section 1), excluding soft prompts and cloze prompts. Criterion 3 separates prompting from fine-tuning—a critical distinction because the paper explicitly "refined our focus to hard (discrete) prompts rather than soft (continuous) prompts and leave out papers that make use of techniques using gradient-based updates" (Section 1). Criterion 4 extends coverage to non-text modalities while maintaining the hard-prompt focus.

**Why 92% agreement with α = κ = 0.81**: The inter-annotator agreement metrics serve different purposes. Krippendorff's α corrects for chance agreement and is considered the gold standard for content analysis reliability. Cohen's κ also corrects for chance but assumes only two raters and nominal categories. A value of 0.81 is conventionally interpreted as "almost perfect" agreement (Landis and Koch, 1977), giving confidence that the inclusion criteria are applied consistently. The 92% raw agreement provides an intuitive sense of consistency.

**Stage 4: LLM-assisted classification.** To scale beyond manual annotation, the authors develop a prompt for GPT-4-1106-preview to classify the remaining papers. The prompt is provided in Appendix A.5 and uses the following structure:

- **System prompt**: "You are a lab assistant, helping with a systematic review on prompt engineering. You've been asked to rate the relevance of a paper to the topic of prompt engineering." It then defines hard prefix prompts, distinguishes them from soft prompts and cloze prompts, and instructs the model to classify as "highly relevant," "somewhat relevant," "neutrally relevant," "somewhat irrelevant," or "highly irrelevant."
- **User prompt template**: "Title: '{title}', Abstract: '{abstract}'. Rate its relevance to the topic of prompt engineering as one of the following categories: 'highly relevant', 'somewhat relevant', 'neutrally relevant', 'somewhat irrelevant', 'highly irrelevant', and provide text from the abstract that justifies your reasoning."

The LLM prompt is validated against 100 ground-truth human annotations, achieving **89% precision** and **75% recall**, for an **F1 of 81%**.

**Why these specific metrics**: Precision measures the fraction of papers the LLM classified as relevant that were actually relevant—89% means relatively few false positives. Recall measures the fraction of actually relevant papers that the LLM correctly identified—75% means some relevant papers were missed. The F1 of 81% balances these. The higher precision than recall suggests the LLM is conservative, preferring to miss relevant papers rather than include irrelevant ones—a reasonable tradeoff for building a taxonomy where false inclusions would add noise to the technique catalog.

**Pipeline completion.** After the PRISMA review process, the dataset contains 1,565 records. The paper hosts this dataset on HuggingFace (with a datasheet in Appendix A.3) and provides the code in a GitHub repository.

**Acknowledged limitations.** The paper notes that papers were gathered "in a semi-automated process which introduced the possibility of irrelevant papers being collected and relevant papers not being collected" (datasheet, Appendix A.3.2). The manual reviews were designed to mitigate these errors, but the 75% recall of the LLM classifier means some relevant papers were certainly missed. The paper frames this as acceptable for a first-iteration taxonomy that can be extended over time.

#### Terminology Standardization Methodology (Sections 1.1, 1.2, Appendices A.1, A.2)

The paper's terminology work addresses a core problem: "there are many poorly understood definitions (e.g. prompt, prompt engineering) and conflicting ones (e.g. role prompt vs persona prompt). The lack of a consistent vocabulary hampers the community's ability to clearly describe the various prompting techniques in use" (Section 1.2).

**Definition derivation process.** Rather than inventing new definitions, the authors "integrate many definitions (Appendix A.1) to derive representative definitions." Table A.1 in Appendix A.1 collects definitions for "Prompt" and "Prompt Engineering" from 12 different papers and synthesizes them. For example:

- For **Prompt**: The collected definitions range from "the input of the model" (Chen et al., 2023a) to "instructions given to an LLM to enforce rules, automate processes, and ensure specific qualities of generated output" (White et al., 2023) to "the instructions provided to an LLM to make it follow specified rules" (Hadi et al., 2023). The paper's synthesized definition is: "A prompt is an input to a Generative AI model, that is used to guide its output" (Section 1.1), which captures the common thread while remaining broad enough to encompass text, images, audio, and other modalities.

- For **Prompt Engineering**: Collected definitions include "the practice of designing, refining, and implementing prompts or instructions that guide the output of LLMs" (Meskó, 2023), "the process of structuring input text for LLMs and is a technique integral to optimizing the efficacy of LLMs" (Chen et al., 2023a), and "an increasingly important skill set needed to converse effectively with large language models" (White et al., 2023). The synthesized definition: "Prompt engineering is the iterative process of developing a prompt by modifying or changing the prompting technique that you are using" (Section 1.2.2).

**Why synthesis over selection**: The paper could have simply adopted one existing definition. By synthesizing across multiple sources, it creates definitions that are representative of community usage rather than tied to a single paper's perspective. This is important for a survey aiming to unify the field—adopting a single definition would implicitly endorse one research tradition over others.

**Vocabulary organization (Figure 1.3).** The paper organizes 33 terms into a hierarchical structure with explicit cross-references to where each term is defined in the paper. This serves as both a glossary and a navigational aid. Terms are categorized into:

- **Prompting terms** (e.g., Prompt, Prompting, Prompt Chain, Prompting Technique)
- **Prompt Engineering terms** (e.g., Prompt Engineering, Prompt Engineering Technique, Meta-Prompting, Conversational Prompt Engineering)
- **Fine-Tuning terms** (e.g., Fine-Tuning, Prompt-Based Learning, Prompt Tuning)
- **Orthogonal Prompt Types** (subdivided into Originator—User Prompt, System Prompt, Assistant Prompt; Density—Hard/Discrete vs. Soft/Continuous; Prediction Style—Prefix vs. Cloze)
- **Answer Engineering terms** (e.g., Verbalizer, Extractor, Answer Trigger)
- **Components of a Prompt** (e.g., Directive, Examples/Exemplars, Output Formatting, Style Instructions, Role, Additional Information)

**Design choice—why include fine-tuning terms**: The paper explicitly states in its scope (Section 1) that it excludes papers using gradient-based updates. However, it includes fine-tuning terminology because these terms are frequently confused with prompting terms. For example, "Prompt Tuning" (Lester et al., 2021) refers to "directly optimizing the weights of the prompt itself, usually through some form of gradient-based updates" (Appendix A.2.3), which is distinct from discrete prompt engineering despite the similar name. By including these terms and explicitly distinguishing them, the paper prevents category confusion.

**Critical distinctions established.** Several terminological clarifications are particularly important:

1. **Prompt vs. Prompt Template**: "A prompt template is a function that contains one or more variables which will be replaced by some media (usually text) to create a prompt. This prompt can then be considered to be an instance of the template" (Section 1.1). This distinction is visualized in Figure 1.2, which shows how "Write a poem about trees" is a prompt while "Write a poem about the following topic: {USER_INPUT}" is a prompt template. Confusing these is a common error in the literature.

2. **Prompt Injection vs. Jailbreaking**: "Prompt Injection is the process of overriding original developer instructions in the prompt with user input" while "Jailbreaking is the process of getting a GenAI model to do or say unintended things through prompting" (Section 5.1.1). The key difference: prompt injection involves a prompt template with developer instructions that the user overrides; jailbreaking involves direct prompting without developer instructions. The paper cites Schulhoff (2024) and Willison (2024) for this distinction.

3. **Hard (Discrete) vs. Soft (Continuous) Prompts**: "Hard prompts contain only tokens (vectors) that correspond to words in the model's vocabulary, while soft prompts may contain tokens that have no corresponding word in the vocabulary" (Section 1). This distinction is critical because the entire paper's scope is limited to hard prompts.

4. **Prefix vs. Cloze Prompts**: "In Cloze prompts, the token(s) to be predicted are presented as 'slots to fill', usually somewhere in the middle of the prompt" while "In Prefix prompts, the token to be predicted is at the end of the prompt" (Appendix A.2.4.3). The paper focuses on prefix prompts because "modern LLM transformer architectures widely employ prefix prompts."

5. **Prompt Engineering vs. Prompt Engineering Technique**: "Prompt engineering is the iterative process of developing a prompt" while "A prompt engineering technique is a strategy for iterating on a prompt to improve it" (Section 1.2.2). Meta-Prompting (prompting an LLM to improve a prompt), APE (Automatic Prompt Engineer), and DSPy are all techniques for prompt engineering, not prompting techniques themselves.

**The definition of "Prompt" has shifted historically.** Section 1.3 traces an important semantic shift. In Brown et al. (2020), the "prompt" was only the input text (e.g., "llama") while "Translate English to French:" was the "task description"—they were separate components. The paper notes: "More recent papers, including this one, refer to the entire string passed to the LLM as the prompt." This historical shift is important context for understanding why definitions vary across the literature.

#### Taxonomy Construction Logic (Section 2.2)

The taxonomy organizes 58 text-based prompting techniques into 6 major categories plus additional categories for multilingual, multimodal, agent, evaluation, security, and alignment techniques. The construction follows a functional grouping principle: techniques are placed in "the single category of most relevance" (Section 2.2), with the acknowledgement that "some of the techniques might fit into multiple categories."

**Category 1: In-Context Learning (ICL)—Section 2.2.1**

ICL is defined as "the ability of GenAIs to learn skills and tasks by providing them with exemplars and or relevant instructions within the prompt, without the need for weight updates/retraining" (Section 2.2.1). The paper explicitly acknowledges that this definition is broader than some uses: "Note that the word 'learn' is misleading. ICL can simply be task specification–the skills are not necessarily new, and can have already been included in the training data" (Figure 2.6 and Appendix A.9).

The paper traces a definitional ambiguity in Brown et al. (2020), which seemingly offers two different definitions for ICL: one broad ("using the text input of a pretrained language model as a form of task specification") and one narrow ("few-shot learning, or in-context learning where we allow as many demonstrations as will fit"). The paper adopts the broad definition and cites Brown et al.'s clarification: "These terms are intended to remain agnostic on the question of whether the model learns new tasks from scratch at inference time or simply recognizes patterns seen during training."

**ICL sub-categories:**

**Few-Shot Prompting Design Decisions (Section 2.2.1.1):** The paper identifies six design dimensions that affect few-shot prompt performance, supported by Figure 2.3:

1. **Exemplar Quantity**: "Increasing the quantity of exemplars in the prompt generally improves model performance, particularly in larger models (Brown et al., 2020)." However, the paper notes that "in some cases, the benefits may diminish beyond 20 exemplars (Liu et al., 2021)" and that for long-context LLMs, "additional exemplars continue to increase performance, though efficiency varies depending on task and model (Agarwal et al., 2024; Bertsch et al., 2024; Jiang et al., 2024)."

2. **Exemplar Ordering**: "The order of exemplars affects model behavior (Lu et al., 2021; Kumar and Talukdar, 2021; Liu et al., 2021; Rubin et al., 2022). On some tasks, exemplar order can cause accuracy to vary from sub-50% to 90%+ (Lu et al., 2021)."

3. **Exemplar Label Distribution**: "As in traditional supervised machine learning, the distribution of exemplar labels in the prompt affects behavior. For example, if 10 exemplars from one class and 2 exemplars of another class are included, this may cause the model to be biased toward the first class."

4. **Exemplar Label Quality**: "Some work (Min et al., 2022) suggests that the accuracy of labels is irrelevant—providing models with exemplars with incorrect labels may not negatively diminish performance. However, under certain settings, there is a significant impact on performance (Yoo et al., 2022). Larger models are often better at handling incorrect or unrelated labels (Wei et al., 2023c)."

5. **Exemplar Format**: "One of the most common formats is 'Q: {input}, A: {label}', but the optimal format may vary across tasks; it may be worth trying multiple formats to see which performs best. There is some evidence to suggest that formats that occur commonly in the training data will lead to better performance (Jiang et al., 2020)."

6. **Exemplar Similarity**: "Selecting exemplars that are similar to the test sample is generally beneficial for performance (Liu et al., 2021; Min et al., 2022). However, in some cases, selecting more diverse exemplars can improve performance (Su et al., 2022; Min et al., 2022)."

**Why these six dimensions over others**: The paper identifies these as the factors that "critically influence the output quality" (Section 2.2.1.1), based on the source papers' findings. However, the paper explicitly warns that "recommendations here do not generalize to all tasks; in some cases, each of them could hurt performance" (Figure 2.3 caption), emphasizing the lack of universal rules in prompting.

**Few-Shot Prompting Techniques (Section 2.2.1.2):** Techniques that automate or optimize the few-shot exemplar selection process:

- **K-Nearest Neighbor (KNN)** (Liu et al., 2021): "selects exemplars similar to D_test_xi to boost performance." The paper notes it is "time and resource intensive" because it requires computing similarity between the test instance and all training instances at inference time.

- **Vote-K** (Su et al., 2022): A two-stage method. In stage one, "a model proposes useful unlabeled candidate exemplars for an annotator to label." In stage two, "the labeled pool is used for Few-Shot Prompting." The method "also ensures that newly added exemplars are sufficiently different than existing ones to increase diversity and representativeness."

- **Self-Generated In-Context Learning (SG-ICL)** (Kim et al., 2022): "leverages a GenAI to automatically generate exemplars. While better than zero-shot scenarios when training data is unavailable, the generated samples are not as effective as actual data."

- **Prompt Mining** (Jiang et al., 2020): "the process of discovering optimal 'middle words' in prompts through large corpus analysis." For example, rather than using "Q: A:" format, the method searches for formats that "occur more frequently in the corpus," under the hypothesis that "formats which occur more often in the corpus will likely lead to improved prompt performance."

- **More Complicated Techniques**: The paper briefly mentions LENS (Li and Qiu, 2023a), UDR (Li et al., 2023f), and Active Example Selection (Zhang et al., 2022a), which "leverage iterative filtering, embedding and retrieval, and reinforcement learning, respectively."

**Instruction Selection (Section 2.2.1.1):** A specific design decision about whether to include task-specific instructions before exemplars. "Ajith et al. (2024) show that generic, task-agnostic instructions (i.e., no instruction or 'Complete the following task:') improve classification and question answering accuracy over task-specific ones." The paper concludes that "instruction-following abilities can be achieved via exemplars alone," but notes that instructions in few-shot prompts "can still guide auxiliary output attributes like writing style (Roy et al., 2023)."

**Zero-Shot Prompting Techniques (Section 2.2.1.3):** Techniques that use zero exemplars:

- **Role Prompting** (also known as Persona Prompting): "assigns a specific role to the GenAI in the prompt. For example, the user might prompt it to act like 'Madonna' or a 'travel writer'." The paper cites evidence that this "can create more desirable outputs for open-ended tasks (Reynolds and McDonell, 2021) and in some cases may improve accuracy on benchmarks (Zheng et al., 2023d)."

- **Style Prompting**: "involves specifying the desired style, tone, or genre in the prompt to shape the output." The paper notes that "a similar effect can be achieved using role prompting."

- **Emotion Prompting** (Li et al., 2023a): "incorporates phrases of psychological relevance to humans (e.g., 'This is important to my career') into the prompt, which may lead to improved LLM performance on benchmarks and open-ended text generation."

- **System 2 Attention (S2A)** (Weston and Sukhbaatar, 2023): Two-step process. First, it "asks an LLM to rewrite the prompt and remove any information unrelated to the question therein." Then, "it passes this new prompt into an LLM to retrieve a final response." The name alludes to Kahneman's System 1/System 2 distinction, with the LLM performing a deliberate filtering step.

- **SimToM** (Wilf et al., 2023): Designed for "complicated questions which involve multiple people or objects. Given the question, it attempts to establish the set of facts one person knows, then answer the question based only on those facts." This is a "two prompt process and can help eliminate the effect of irrelevant information in the prompt."

- **Rephrase and Respond (RaR)** (Deng et al., 2023): "instructs the LLM to rephrase and expand the question before generating the final answer." The paper notes this "could all be done in a single pass or the new question could be passed to the LLM separately."

- **Re-reading (RE2)** (Xu et al., 2023): "adds the phrase 'Read the question again:' to the prompt in addition to repeating the question." Despite simplicity, "it has shown improvement in reasoning benchmarks, especially with complex questions."

- **Self-Ask** (Press et al., 2022): "prompts LLMs to first decide if they need to ask follow up questions for a given prompt. If so, the LLM generates these questions, then answers them and finally answers the original question."

**Category 2: Thought Generation (Section 2.2.2)**

"Thought generation encompasses a range of techniques that prompt the LLM to articulate its reasoning while solving a problem" (Section 2.2.2). The paper explicitly acknowledges the anthropomorphizing language ("the paper notes that 'such techniques are often described using words like "think" that anthropomorphize models. We attempt not to use this language, but do use original authors' language where appropriate'"), but uses it for consistency with the cited literature.

**Chain-of-Thought (CoT) Prompting** (Wei et al., 2022b): "leverages few-shot prompting to encourage the LLM to express its thought process before delivering its final answer." The canonical example is Figure 2.8: a one-shot prompt where the exemplar includes a question ("Jack has two baskets, each containing three balls. How many balls does Jack have in total?"), a reasoning path ("One basket contains 3 balls, so two baskets contain 3 * 2 = 6 balls."), and the answer ("6"). The paper notes this technique "has been demonstrated to significantly enhance the LLM's performance in mathematics and reasoning tasks."

**Zero-Shot-CoT (Section 2.2.2.1):** "The most straightforward version of CoT contains zero exemplars. It involves appending a thought inducing phrase like 'Let's think step by step.' (Kojima et al., 2022) to the prompt." Other thought inducers listed:

- "First, let's think about this logically" (Kojima et al., 2022)
- "Let's work this out in a step by step way to be sure we have the right answer" (Zhou et al., 2022b)
- Yang et al. (2023a) "searches for an optimal thought inducer"

The paper notes that "Zero-Shot-CoT approaches are attractive as they don't require exemplars and are generally task agnostic."

**Specific Zero-Shot CoT variants:**

- **Step-Back Prompting** (Zheng et al., 2023c): "a modification of CoT where the LLM is first asked a generic, high-level question about relevant concepts or facts before delving into reasoning." The paper reports this "has improved performance significantly on multiple reasoning benchmarks for both PaLM-2L and GPT-4."

- **Analogical Prompting** (Yasunaga et al., 2023): "similar to SG-ICL, and automatically generates exemplars that include CoTs. It has demonstrated improvements in mathematical reasoning and code generation tasks."

- **Thread-of-Thought (ThoT) Prompting** (Zhou et al., 2023): "consists of an improved thought inducer for CoT reasoning. Instead of 'Let's think step by step,' it uses 'Walk me through this context in manageable parts step by step, summarizing and analyzing as we go.'"

- **Tabular Chain-of-Thought (Tab-CoT)** (Jin and Lu, 2023): "consists of a Zero-Shot CoT prompt that makes the LLM output reasoning as a markdown table. This tabular design enables the LLM to improve the structure and thus the reasoning of its output."

**Few-Shot CoT (Section 2.2.2.2):** This "presents the LLM with multiple exemplars, which include chains-of-thought." The paper notes this "is occasionally referred to as Manual-CoT (Zhang et al., 2022b) or Golden CoT (Del and Fishel, 2023)."

Specific Few-Shot CoT variants:

- **Contrastive CoT Prompting** (Chia et al., 2023): "adds both exemplars with incorrect and correct explanations to the CoT prompt in order to show the LLM how not to reason. This method has shown significant improvement in areas like Arithmetic Reasoning and Factual QA."

- **Uncertainty-Routed CoT Prompting** (Google, 2023): "samples multiple CoT reasoning paths, then selects the majority if it is above a certain threshold (calculated based on validation data). If not, it samples greedily and selects that response."

- **Complexity-based Prompting** (Fu et al., 2023b): Two modifications. First, "it selects complex examples for annotation and inclusion in the prompt, based on factors like question length or reasoning steps required." Second, during inference, "it samples multiple reasoning chains (answers) and uses a majority vote among chains exceeding a certain length threshold, under the premise that longer reasoning indicates higher answer quality."

- **Active Prompting** (Diao et al., 2023): "starts with some training questions/exemplars, asks the LLM to solve them, then calculates uncertainty (disagreement in this case) and asks human annotators to rewrite the exemplars with highest uncertainty."

- **Memory-of-Thought Prompting** (Li and Qiu, 2023b): "leverage unlabeled training exemplars to build Few-Shot CoT prompts at test time. Before test time, it performs inference on the unlabeled training exemplars with CoT. At test time, it retrieves similar instances to the test sample."

- **Automatic Chain-of-Thought (Auto-CoT) Prompting** (Zhang et al., 2022b): "uses Wei et al. (2022b)'s Zero-Shot prompt to automatically generate chains of thought. These are then used to build a Few-Shot CoT prompt for a test sample."

**Category 3: Decomposition (Section 2.2.3)**

"Significant research has focused on decomposing complex problems into simpler sub-questions. This is an effective problem-solving strategy for humans as well as GenAI" (Section 2.2.3). The paper distinguishes decomposition from CoT by noting that "some decomposition techniques are similar to thought-inducing techniques, such as CoT, which often naturally breaks down problems into simpler components. However, explicitly breaking down problems can further improve LLMs' problem solving ability."

**Decomposition technique hierarchy:**

- **Least-to-Most Prompting** (Zhou et al., 2022a): A two-phase approach. First, it "prompts a LLM to break a given problem into sub-problems without solving them." Then, "it solves them sequentially, appending model responses to the prompt each time, until it arrives at a final result." The paper reports improvements in "tasks involving symbolic manipulation, compositional generalization, and mathematical reasoning."

- **Decomposed Prompting (DECOMP)** (Khot et al., 2022): "Few-Shot prompts a LLM to show it how to use certain functions. These might include things like string splitting or internet searching; these are often implemented as separate LLM calls. Given this, the LLM breaks down its original problem into sub-problems which it sends to different functions." The paper reports "improved performance over Least-to-Most prompting on some tasks."

- **Plan-and-Solve Prompting** (Wang et al., 2023f): "consists of an improved Zero-Shot CoT prompt, 'Let's first understand the problem and devise a plan to solve it. Then, let's carry out the plan and solve the problem step by step'." This method "generates more robust reasoning processes than standard Zero-Shot-CoT on multiple reasoning datasets."

- **Tree-of-Thought (ToT)** (Yao et al., 2023b): "creates a tree-like search problem by starting with an initial problem then generating multiple possible steps in the form of thoughts (as from a CoT). It evaluates the progress each step makes towards solving the problem (through prompting) and decides which steps to continue with, then keeps creating more thoughts. ToT is particularly effective for tasks that require search and planning."

- **Recursion-of-Thought** (Lee and Kim, 2023): "similar to regular CoT. However, every time it encounters a complicated problem in the middle of its reasoning chain, it sends this problem into another prompt/LLM call. After this is completed, the answer is inserted into the original prompt." The paper notes that though this was "implemented using fine-tuning to output a special token that sends sub-problem into another prompt, it could also be done only through prompting."

- **Program-of-Thoughts** (Chen et al., 2023d): "uses LLMs like Codex to generate programming code as reasoning steps. A code interpreter executes these steps to obtain the final answer. It excels in mathematical and programming-related tasks but is less effective for semantic reasoning tasks."

- **Faithful Chain-of-Thought** (Lyu et al., 2023): "generates a CoT that has both natural language and symbolic language (e.g. Python) reasoning, just like Program-of-Thoughts. However, it also makes use of different types of symbolic languages in a task-dependent fashion."

- **Skeleton-of-Thought** (Ning et al., 2023): "focuses on accelerating answer speed through parallelization. Given a problem, it prompts an LLM to create a skeleton of the answer, in a sense, sub-problems to be solved. Then, in parallel, it sends these questions to a LLM and concatenates all the outputs to get a final response."

- **Metacognitive Prompting** (Wang and Zhao, 2024): "attempts to make the LLM mirror human metacognitive processes with a five part prompt chain, with steps including clarifying the question, preliminary judgement, evaluation of response, decision confirmation, and confidence assessment."

**Category 4: Ensembling (Section 2.2.4)**

"In GenAI, ensembling is the process of using multiple prompts to solve the same problem, then aggregating these responses into a final output. In many cases, a majority vote—selecting the most frequent response—is used to generate the final output" (Section 2.2.4). The paper notes that these techniques "reduce the variance of LLM outputs and often improving accuracy, but come with the cost of increasing the number of model calls needed to reach a final answer."

**Ensembling techniques:**

- **Demonstration Ensembling (DENSE)** (Khalifa et al., 2023): "creates multiple few-shot prompts, each containing a distinct subset of exemplars from the training set. Next, it aggregates over their outputs to generate a final response."

- **Mixture of Reasoning Experts (MoRE)** (Si et al., 2023d): "creates a set of diverse reasoning experts by using different specialized prompts for different reasoning types (such as retrieval augmentation prompts for factual reasoning, Chain-of-Thought reasoning for multi-hop and math reasoning, and generated knowledge prompting for commonsense reasoning). The best answer from all experts is selected based on an agreement score."

- **Max Mutual Information Method** (Sorensen et al., 2022): "creates multiple prompt templates with varied styles and exemplars, then selects the optimal template as the one that maximizes mutual information between the prompt and the LLM's outputs."

- **Self-Consistency** (Wang et al., 2022): Based on "the intuition that multiple different reasoning paths can lead to the same answer. This method first prompts the LLM multiple times to perform CoT, crucially with a non-zero temperature to elicit diverse reasoning paths. Next, it uses a majority vote over all generated responses to select a final response."

- **Universal Self-Consistency** (Chen et al., 2023e): "similar to Self-Consistency except that rather than selecting the majority response by programmatically counting how often it occurs, it inserts all outputs into a prompt template that selects the majority answer. This is helpful for free-form text generation and cases where the same answer may be output slightly differently by different prompts."

- **Meta-Reasoning over Multiple CoTs** (Yoran et al., 2023): "similar to universal Self-Consistency; it first generates multiple reasoning chains (but not necessarily final answers) for a given problem. Next, it inserts all of these chains in a single prompt template then generates a final answer from them."

- **DiVeRSe** (Li et al., 2023i): "creates multiple prompts for a given problem then performs Self-Consistency for each, generating multiple reasoning paths. They score reasoning paths based on each step in them then select a final response."

- **Consistency-based Self-adaptive Prompting (COSP)** (Wan et al., 2023a): "constructs Few-Shot CoT prompts by running Zero-Shot CoT with Self-Consistency on a set of examples then selecting a high agreement subset of the outputs to be included in the final prompt as exemplars. It again performs Self-Consistency with this final prompt."

- **Universal Self-Adaptive Prompting (USP)** (Wan et al., 2023b): "builds upon the success of COSP, aiming to make it generalizable to all tasks. USP makes use of unlabeled data to generate exemplars and a more complicated scoring function to select them. Additionally, USP does not use Self-Consistency."

- **Prompt Paraphrasing** (Jiang et al., 2020): "transforms an original prompt by changing some of the wording, while still maintaining the overall meaning. It is effectively a data augmentation technique that can be used to generate prompts for an ensemble."

**Category 5: Self-Criticism (Section 2.2.5)**

"When creating GenAI systems, it can be useful to have LLMs criticize their own outputs" (Section 2.2.5). The paper notes this can be "simply a judgement (e.g., is this output correct) or the LLM could be prompted to provide feedback, which is then used to improve the answer."

**Self-criticism techniques:**

- **Self-Calibration** (Kadavath et al., 2022): Two-step process. First, it "prompts an LLM to answer a question. Then, it builds a new prompt that includes the question, the LLM's answer, and an additional instruction asking whether the answer is correct. This can be useful for gauging confidence levels when applying LLMs when deciding when to accept or revise the original answer."

- **Self-Refine** (Madaan et al., 2023): "an iterative framework where, given an initial answer from the LLM, it prompts the same LLM to provide feedback on the answer, and then prompts the LLM to improve the answer based on the feedback. This iterative process continues until a stopping condition is met (e.g., max number of steps reached)."

- **Reversing Chain-of-Thought (RCoT)** (Xue et al., 2023): Three steps. First, it "prompts LLMs to reconstruct the problem based on generated answer. Then, it generates fine-grained comparisons between the original problem and the reconstructed problem as a way to check for any inconsistencies. These inconsistencies are then converted to feedback for the LLM to revise the generated answer."

- **Self-Verification** (Weng et al., 2022): "generates multiple candidate solutions with Chain-of-Thought (CoT). It then scores each solution by masking certain parts of the original question and asking an LLM to predict them based on the rest of the question and the generated solution."

- **Chain-of-Verification (COVE)** (Dhuliawala et al., 2023): Four steps. First, it "uses an LLM to generate an answer to a given question. Then, it creates a list of related questions that would help verify the correctness of the answer. Each question is answered by the LLM, then all the information is given to the LLM to produce the final revised answer."

- **Cumulative Reasoning** (Zhang et al., 2023b): Iterative process. "First generates several potential steps in answering the question. It then has a LLM evaluate them, deciding to either accept or reject these steps. Finally, it checks whether it has arrived at the final answer. If so, it terminates the process, but otherwise it repeats it."

**Category 6: Prompt Engineering Techniques (Section 2.4)**

These are techniques for automatically optimizing prompts, distinct from prompting techniques themselves. The paper notes that it includes "some techniques that use gradient updates, since the set of prompt engineering techniques is much smaller than that of prompting techniques."

- **Meta Prompting**: "the process of prompting a LLM to generate or improve a prompt or prompt template" (Figure 2.12 shows a simple template: "Improve the following prompt: {PROMPT}"). The paper notes this can be "done without any scoring mechanism, using just a simple template," but "other works present more complex uses of meta-prompting, with multiple iterations and scoring mechanisms."

- **AutoPrompt** (Shin et al., 2020b): "uses a frozen LLM as well as a prompt template that includes some 'trigger tokens', whose values are updated via backpropagation at training time. This is a version of soft-prompting." This is included despite the paper's scope being limited to hard prompts because it's a prompt engineering technique rather than a prompting technique.

- **Automatic Prompt Engineer (APE)** (Zhou et al., 2022b): "uses a set of exemplars to generate a Zero-Shot instruction prompt. It generates multiple possible prompts, scores them, then creates variations of the best ones (e.g. by using prompt paraphrasing). It iterates on this process until some desiderata are reached."

- **Gradientfree Instructional Prompt Search (GrIPS)** (Prasad et al., 2023): "similar to APE, but uses a more complex set of operations including deletion, addition, swapping, and paraphrasing in order to create variations of a starting prompt."

- **Prompt Optimization with Textual Gradients (ProTeGi)** (Pryzant et al., 2023): A multi-step approach. First, "it passes a batch of inputs through the template, then passes the output, ground truth, and prompt into another prompt that criticizes the original prompt. It generates new prompts from these criticisms then uses a bandit algorithm (Gabillon et al., 2011) to select one. ProTeGi demonstrates improvements over methods like APE and GRIPS."

- **RLPrompt** (Deng et al., 2022): "uses a frozen LLM with an unfrozen module added. It uses this LLM to generate prompt templates, scores the templates on a dataset, and updates the unfrozen module using Soft Q-Learning (Guo et al., 2022). Interestingly, the method often selects grammatically nonsensical text as the optimal prompt template."

- **Dialogue-comprised Policy-gradient-based Discrete Prompt Optimization (DP2O)** (Li et al., 2023b): "perhaps the most complicated prompt engineering technique, involving reinforcement learning, a custom prompt scoring function, and conversations with an LLM to construct the prompt."

#### Answer Engineering Framework (Section 2.5)

"Answer engineering is the iterative process of developing or selecting among algorithms that extract precise answers from LLM outputs" (Section 2.5). The paper distinguishes three design decisions, illustrated in Figure 2.13:

**Answer Shape**: "The shape of an answer is its physical format. For example, it could be a token, span of tokens, or even an image or video." The paper notes that "it is sometimes useful to restrict the output shape of a LLM to a single token for tasks like binary classification."

**Answer Space**: "The space of an answer is the domain of values that its structure may contain. This may simply be the space of all tokens, or in a binary labeling task, could just be two possible tokens."

**Answer Extractor**: "In cases where it is impossible to entirely control the answer space (e.g. consumer-facing LLMs), or the expected answer may be located somewhere within the model output, a rule can be defined to extract the final answer." Three extractor types are identified:

- **Verbalizer**: "Often used in labeling tasks, a verbalizer maps a token, span, or other type of output to a label and vice-versa (injective). For example, if we wish for a model to predict whether a Tweet is positive or negative, we could prompt it to output either '+' or '-' and a verbalizer would map these token sequences to the appropriate labels."

- **Regex**: "Regexes are often used to extract answers. They are usually used to search for the first instance of a label. However, depending on the output format and whether CoTs are generated, it may be better to search for the last instance."

- **Separate LLM**: "Sometimes outputs are so complicated that regexes won't work consistently. In this case, it can be useful to have a separate LLM evaluate the output and extract an answer. This separate LLM will often use an answer trigger (Kojima et al., 2022), e.g. 'The answer (Yes or No) is', to extract the answer."

**Why answer engineering is distinct from prompt engineering**: The paper explicitly distinguishes these: "We consider answer engineering to be distinct from prompt engineering, but extremely closely related; the processes are often conducted in tandem." This distinction is important because many prompting papers conflate improving the prompt (getting the model to produce better reasoning) with improving the extraction (getting a clean label from messy outputs). The paper's benchmarking case study (Section 6.1.4) illustrates this directly: "Evaluating whether a LLM has properly responded to a question is a difficult task. We marked answers as correct if they followed certain identifiable patterns, such as being the only capitalized letter (A-D) within parentheses or following a phrase like 'The correct answer is'."

#### Meta-Analysis Methodology (Section 2.3)

The paper measures technique prevalence through citation analysis within its own corpus of 1,565 papers.

**Citation counting procedure**: "We measure technique usage by proxy of measuring the number of citations by other papers in our dataset. We do so with the presumption that papers about prompting are more likely to actually use or evaluate the cited technique" (Section 2.3).

**Model and dataset extraction**: To generate Figures 2.9 and 2.10, "we prompted GPT-4-1106-preview to extract any mentioned dataset or model from the body of papers in our dataset. After, we manually filtered out results that were not models or datasets. The citation counts were acquired by searching items from the finalized list on Semantic Scholar."

**Why this extraction methodology**: Using an LLM for extraction scales to 1,565 papers where manual extraction would be prohibitively expensive. The manual filtering step corrects LLM errors. Using Semantic Scholar for citation counts provides standardized, reproducible numbers rather than relying on the paper corpus alone.

**Key results from the meta-analysis:**

- **Model usage** (Figure 2.9): GPT-3 is the most-cited model (~500 mentions), followed by BERT (~480), GPT-4 (~380), and RoBERTa (~280). The paper does not provide exact numbers—these are approximate from the bar chart.

- **Dataset usage** (Figure 2.10): GSM8K is the most-cited dataset (~800 mentions), followed by MMLU (~520), BBH (~250), CommonsenseQA (~200), and HellaSwag (~180).

- **Technique citations** (Figure 2.11): The top 25 cited prompting-related papers within the dataset show Few-Shot Learning and Zero-Shot Reasoning as the most cited (~10^3 range), followed by papers on Good In-Context Examples, Self-Consistency, and Prompt Order Sensitivity (~10^2 range).

**Why these specific metrics matter**: The meta-analysis serves two purposes. First, it provides empirical grounding for claims about technique prevalence—rather than asserting that Few-Shot and CoT are widely used, the paper can point to citation counts. Second, the model and dataset usage data serves as guidance for researchers proposing new techniques: "In order to make it easier for researchers proposing new techniques to know how to benchmark them, we quantitatively examine which models and what benchmark datasets are being used" (Section 2.3.1).

#### Benchmarking Setup (Section 6.1)

The paper's primary empirical comparison tests six prompting techniques on MMLU using GPT-3.5-turbo, providing controlled evidence about relative technique effectiveness.

**Dataset and model selection**: "We choose a subset of prompting techniques and run them on the widely used benchmark MMLU (Hendrycks et al., 2021). We ran on a representative subset of 2,800 MMLU questions (20% of the questions from each category)." The paper excludes the human_sexuality category because "gpt-3.5-turbo refused to answer these questions." All experiments use gpt-3.5-turbo.

**Prompt template structure** (Figure 6.2): The general template is:
```
{BASE_INSTRUCTION}
{EXEMPLARS}
{QUESTION} {THOUGHT_INDUCER}
```
Only base instructions and question exist in every prompt. The base instruction is "a phrase like 'Solve the problem and return (A), (B), (C) or (D).'" The paper varies this across three phrasings.

**Question formats** (Figures 6.3 and 6.4): The paper tests two formatting choices from Sclar et al. (2023b):

- **Format 1**: Structured bullet format:
  ```
  Problem
  {QUESTION}
  Options
  (A)::{A} (B)::{B} (C)::{C} (D)::{D}
  Answer
  ```

- **Format 2**: Compact format:
  ```
  PROBLEM::{QUESTION}, OPTIONS::
  (A): {A}
  (B): {B}
  (C): {C}
  (D): {D}, ANSWER::
  ```

**Why test two formats**: Sclar et al. (2023b) "explored how formatting choices can affect benchmarking results" and found that these two formats "lead to varied results on their task." The paper is testing whether the format sensitivity observed by Sclar et al. generalizes to MMLU with GPT-3.5-turbo.

**Technique variations tested:**

1. **Zero-Shot**: "As a baseline, we ran questions directly through the model without any special prompting technique, only the base instruction and question." Six total variations: 2 question formats × 3 base instruction phrasings. Temperature = 0.

2. **Zero-Shot-CoT**: Three thought inducers tested: "Let's think step by step" (Kojima et al., 2022), ThoT (Zhou et al., 2023), and Plan and Solve (Wang et al., 2023f). For each, the same 2 formats × 3 phrasings. "Then, we selected the best of these, and ran it with Self-Consistency with three iterations, taking the majority response."

3. **Few-Shot**: Exemplars "generated by one of our authors." 2 formats × 3 phrasings. "Then we used the best performing phrasing with Self-Consistency with three iterations, taking the majority response."

4. **Few-Shot-CoT**: Same as Few-Shot but with CoT reasoning in exemplars. Same variation structure.

**Self-Consistency hyperparameters**: "We set temperature to 0.5, following Wang et al. (2022)'s guidelines. For all other prompts, a temperature of 0 was used" (Section 6.1.3).

**Answer extraction for evaluation** (Section 6.1.4): "We marked answers as correct if they followed certain identifiable patterns, such as being the only capitalized letter (A-D) within parentheses or following a phrase like 'The correct answer is'." This is an example of answer engineering (Section 2.5) applied to benchmarking.

**Why these specific techniques**: The paper selects techniques that represent different points on the complexity spectrum: Zero-Shot (simplest) → Few-Shot (intermediate) → CoT variants (more complex) → Self-Consistency variants (most complex). This allows measuring whether "performance generally improved as techniques grew more complex" (Section 6.1.5).

**Key result (Figure 6.1)**: The paper reports:
- Zero-Shot: 0.627 accuracy
- Zero-Shot-CoT: 0.547 accuracy (↓0.080 from Zero-Shot)
- Zero-Shot-CoT with Self-Consistency: 0.574 (above Zero-Shot-CoT but below Zero-Shot)
- Few-Shot: 0.652
- Few-Shot-CoT: 0.692 (highest)
- Few-Shot-CoT with Self-Consistency: 0.691

The paper notes that "Zero-Shot-CoT dropped precipitously from Zero-Shot. Although it had a wide spread, for all variants, Zero-Shot performed better." This is a significant finding because it contradicts the common narrative that CoT always improves performance—on MMLU with GPT-3.5-turbo, the simple Zero-Shot baseline outperformed Zero-Shot-CoT.

#### Case Study Methodology (Section 6.2)

The paper's prompt engineering case study documents the complete process of developing a prompt for a real-world task: detecting entrapment (frantic hopelessness) in text from potentially suicidal individuals.

**Task definition**: Entrapment is defined as "a desire to escape from an unbearable situation, tied with the perception that all escape routes are blocked" (Melzer et al., 2024). The task is binary classification: does a given post exhibit entrapment?

**Dataset**: 221 posts from r/SuicideWatch (from the University of Maryland Reddit Suicidality Dataset), coded by "two coders trained on the recognition of the factors in Suicide Crisis Syndrome" with Krippendorff's α = 0.72. Split into 121 development posts and 100 test posts.

**Prompt engineer profile**: "An expert prompt engineer, who has authored a widely used guide on prompting (Schulhoff, 2022)." The engineer was given "a brief verbal and written summary of Suicide Crisis Syndrome and entrapment, along with 121 development posts and their positive/negative labels." This "mirrors frequent real-life scenarios in which prompts are developed based on a task description and the data" (Section 6.2.2).

**Process documentation**: "The exercise proceeded through 47 recorded development steps, cumulatively about 20 hours of work" (Section 6.2.3).

**Initial exploration phase (2 steps)**: The engineer began by reviewing the description of entrapment (Figure 6.7) and checking whether GPT-4-turbo-preview "knew what entrapment was" (Figure 6.8). Finding that "the LLM's response was not similar to the description that had been given," the engineer included the description in all future prompts.

**Getting a label phase (8 steps)**: The engineer encountered a significant obstacle: "the LLM was giving mental health advice (e.g. Figure 6.9) instead of labeling the input." This was addressed by "switching to the GPT-4-32K model." The key insight: "'guard rails' associated with some large language models may interfere with the ability to make progress on a prompting task, and this could influence the choice of model for reasons other than the LLM's potential quality" (Section 6.2.3.2).

**Prompting techniques phase (32 steps)**: The main development work involved iterating through multiple techniques. The paper provides a quantitative summary (Figure 6.5, Figure 6.6).

**Technique evolution (Figure 6.6):**
- Zero-Shot + Context: 0.40 F1, 1.0 recall, 0.25 precision (first viable prompt)
- 10-Shot + Context: 0.45 F1, ↑0.05 from previous best
- 1-Shot AutoDiCoT + Full Context: 0.36 F1, ↓0.09
- Multiple intermediate steps with varying performance
- 10-Shot AutoDiCoT: 0.53 F1 (final best manual prompt), 0.86 recall, 0.38 precision

**The AutoDiCoT Algorithm** (Figure 6.12): The prompt engineer developed a novel technique during the case study. The algorithm:
1. Takes development items T with n pairs (qi, ai)
2. For each pair, labels qi as entrapment or not using the model
3. If the model labels correctly: prompts with "Why?" to generate reasoning chain ri
4. If incorrect: prompts with "It is actually [is/is not] entrapment, please explain why." to generate ri
5. Stores tuples (qi, ri, ai)

This technique "can be generalized to any labeling task. It combines the automatic generation of CoTs (Zhang et al., 2022b) with showing the LLM examples of bad reasoning, as in the case of Contrastive CoT (Chia et al., 2023)" (Section 6.2.3.3).

**DSPy comparison** (Section 6.2.3.3): As an alternative to manual engineering, the authors tested the DSPy framework (Khattab et al., 2023) "which automatically optimizes LLM prompts for a given target metric." The setup used "a chain-of-thought classification pipeline that uses the definition of entrapment in Figure 6.7. Over 16 iterations, DSPy bootstrapped synthetic LLM-generated demonstrations and randomly sampled training exemplars, with the ultimate objective of maximizing F1 on the same development set." The model used was gpt-4-0125-preview with the BootstrapFewShotWithRandomSearch "teleprompter."

**Key case study findings:**

1. **Performance sensitivity**: F1 scores "could change by as much as 0.04 upon subsequent runs, even with temperature and top p set to zero" (Section 6.2.3.3). This is evidence that LLM outputs are non-deterministic even with temperature=0, likely due to hardware-level nondeterminism in GPU floating-point operations.

2. **Counterintuitive prompt elements**: The prompt engineer accidentally duplicated an email message in the prompt, and "removing the duplicate actually decreased performance." When attempting to de-duplicate intentionally, "removing the duplicate significantly hurt performance, ↓0.07 (0.45) F1." This demonstrates the "black art" nature of prompting where "there being any obvious reason those details should matter" is absent.

3. **Prompt engineering vs. domain expertise divergence**: The engineer's decision to restrict labeling to explicit statements only ("IMPORTANT: Only label the post as entrapment if they explicitly say that they feel trapped") improved F1 but was actually the wrong direction for the real-world goal, because "Entrapment need not be expressed explicitly in order to be present... clinical experts who have looked at the texts found that expressions of entrapment could be implicit and potentially quite nuanced" (Section 6.2.3.3). The paper frames this as a key lesson: "it is easy for the process of prompt development to diverge from the actual goals unless regular engagement is fostered between the prompt engineer and domain experts."

4. **DSPy performance**: The best DSPy prompt "includes 15 exemplars (without CoT reasoning) and one bootstrapped reasoning demonstration. It achieves 0.548 F1 (and 0.385 / 0.952 precision / recall) on the test set, without making any use of the professor's email nor the incorrect instruction about the explicitness of entrapment" (Section 6.2.3.3). This outperformed the human engineer's best manual prompt on the test set (Figure 6.19), demonstrating "the significant promise of automated prompt engineering."

**Why the case study matters**: The paper identifies this as "the first annotated case study of manual prompt engineering" and emphasizes that it "is not intended to be an empirical contribution in terms of actually solving the problem. Rather, it provides one illustration of how an experienced prompt engineer would approach a task like this, along with lessons learned" (Section 6.2). This fills a gap between theoretical technique catalogs and practical application knowledge.

**Take-home lessons from the case study** (Section 6.2.4):

1. "Prompt engineering is fundamentally different from other ways of getting a computer to behave the way you want it to: these systems are being cajoled, not programmed, and, in addition to being quite sensitive to the specific LLM being used, they can be incredibly sensitive to specific details in prompts without there being any obvious reason those details should matter."

2. "It is important to dig into the data (e.g. generating potential explanations for LLM 'reasoning' that leads to incorrect responses)."

3. "Prompt engineering should involve engagement between the prompt engineer, who has expertise in how to coax LLMs to behave in desired ways, and domain experts, who understand what those desired ways are and why."

4. "There was significant promise in an automated method for exploring the prompting space, but also that combining that automation with human prompt engineering/revision was the most successful approach."

## 4. Key Insights and Innovations

### Innovation 1: Reframing Prompt Engineering from an Artisanal Craft to a Systematic Discipline with a Formal Taxonomy and Unified Vocabulary

The paper’s most foundational contribution is not any single technique but the intellectual infrastructure it provides: a standardized vocabulary of 33 terms and a hierarchical taxonomy of 58 text-based prompting techniques, derived from a PRISMA-grounded systematic review of 1,565 papers. Before this work, the field operated without shared definitions—the same technique was called "Role Prompting" by one research group and "Persona Prompting" by another (Section 1.2), and the term "prompt" itself had shifted meaning from Brown et al. (2020)’s narrow usage (where only the input text was the "prompt" while instructions were the "task description") to the modern broader usage. The absence of common vocabulary meant that researchers could not cleanly compare results, practitioners could not discover relevant techniques, and the field accumulated redundant terminology without recognizing it.

What makes this a genuine conceptual contribution rather than mere organizational housekeeping is the *methodology of synthesis* the paper employs. Rather than imposing definitions from a single perspective, the paper collects competing definitions from across the literature (Appendix A.1, Table A.1), integrates them to derive representative consensus definitions, and organizes terms into orthogonal classification dimensions (originator, hard vs. soft, prediction style). This approach—aggregating then reconciling rather than selecting—is itself a methodological contribution that other emerging fields could adopt. The decision to include fine-tuning terminology (e.g., "Prompt Tuning" vs. "Prompt Engineering") alongside prompting terms, explicitly distinguishing them, addresses a common source of confusion where similar names mask fundamentally different procedures.

The taxonomy construction is similarly principled. Rather than organizing techniques by surface features (e.g., alphabetical listing, chronological order), the paper groups them by *functional mechanism*: In-Context Learning techniques modify what the model sees as input (exemplars, instructions), Thought Generation techniques elicit explicit reasoning, Decomposition techniques break problems into sub-problems, Ensembling techniques aggregate multiple outputs, and Self-Criticism techniques have the model evaluate its own outputs. This functional grouping reveals relationships that would be invisible in a flat list—for instance, that Self-Consistency (Ensembling), Self-Refine (Self-Criticism), and Least-to-Most (Decomposition) are distinct responses to the same underlying challenge (LLM output variance and reasoning errors), pursued through different mechanisms. The interconnected categories diagram (Figure 1.1) visually encodes this insight by showing Security, Safety, and Evaluation needs surrounding all prompting activities, positioning prompt engineering as a holistic system design problem rather than an isolated input-crafting exercise.

This contribution is fundamentally a *reframing* of the field’s self-conception. Prior surveys (Liu et al., 2023b; Chen et al., 2023a; White et al., 2023) provided partial catalogs or prescriptive pattern libraries, but none claimed systematic coverage grounded in a reproducible literature review with documented inter-annotator reliability (92% agreement, Krippendorff’s α = 0.81). By providing this infrastructure, the paper enables the field to transition from an era where techniques were "discovered—the result of thorough experimentation, analogies from human reasoning, or pure serendipity" (Section 8) to one where techniques can be systematically compared, related, and built upon with shared language.

---

### Innovation 2: Documenting the "Black Art" of Prompt Engineering Through the First Annotated Case Study, Revealing That Prompt Sensitivity to Seemingly Irrelevant Details Is a Fundamental—Not Incidental—Property of Current LLMs

The paper’s prompt engineering case study (Section 6.2) represents a novel form of empirical contribution for the field: a complete, annotated record of an expert prompt engineer’s 47-step, ~20-hour development process on a real-world suicide risk detection task. This is not an empirical contribution in the traditional benchmarking sense—the goal was never to achieve state-of-the-art entrapment detection—but rather a *process documentation* contribution that reveals phenomena invisible in controlled experiments.

The central finding is what might be called the **sensitivity-without-explanation problem**: LLM performance can be dramatically affected by prompt details for which there is "no obvious reason those details should matter" (Section 6.2.4). The most striking example: the prompt engineer accidentally duplicated an email containing project background information in the prompt. When this duplication was removed, performance dropped significantly (↓0.07 F1) with no clear mechanism for why. This is not a subtle hyperparameter effect—it is a large, replicable performance difference caused by a prompt feature that has no apparent semantic relationship to the task. The paper draws a direct lesson: these systems are "being cajoled, not programmed" (Section 6.2.4).

Prior work had documented prompt sensitivity in controlled settings. Sclar et al. (2023a) showed that minor formatting changes could cause LLaMA2-7B accuracy to range from near 0 to 0.804. The paper’s own benchmarking (Section 6.1) demonstrates sensitivity through the Zero-Shot-CoT underperformance on MMLU. But the case study reveals something beyond what controlled experiments capture: that sensitivity operates through *arbitrary and undiscoverable* mechanisms (duplicating an email) that no systematic ablation could anticipate. This transforms the sensitivity problem from "prompts are sensitive to semantically meaningful variations" to "prompts are sensitive to variations whose meaningfulness is entirely opaque to human reasoning."

A second insight from the case study concerns the **divergence between prompt optimization metrics and real-world goals**. The prompt engineer’s decision to restrict labeling to explicit statements only ("IMPORTANT: Only label the post as entrapment if they explicitly say that they feel trapped") improved F1 on the development set but was actually the wrong direction for the clinical use case, where "expressions of entrapment could be implicit and potentially quite nuanced" (Section 6.2.3.3). This is a documented instance of Goodhart’s Law in prompt engineering: optimizing for the available metric (F1 on labeled data) drove behavior away from the true objective (identifying at-risk individuals). The paper frames the lesson explicitly: "it is easy for the process of prompt development to diverge from the actual goals unless regular engagement is fostered between the prompt engineer and domain experts" (Section 6.2.4).

The AutoDiCoT algorithm (Figure 6.12) that emerged during the case study—automatically generating contrastive reasoning chains by prompting the model to explain both its correct and incorrect classifications—represents a bottom-up technique discovery process. Rather than being proposed in a standalone paper with controlled benchmarks, it was developed organically in response to a specific misclassification, then generalized. This suggests a different model for technique innovation than the dominant "propose → benchmark → publish" pipeline, one grounded in iterative engagement with real data.

The comparison with DSPy (Section 6.2.3.3) adds another layer: automated prompt optimization achieved better test-set performance (0.548 F1 vs. the human engineer’s 0.53) without incorporating the email duplication or the incorrect explicitness instruction. This does not invalidate manual prompt engineering—the paper concludes that "combining that automation with human prompt engineering/revision was the most successful approach"—but it does suggest that automated methods can avoid some of the arbitrary sensitivity traps that human engineers fall into, precisely because they are not susceptible to the same cognitive biases about what "should" matter.

---

### Innovation 3: Providing the First Comprehensive Threat Taxonomy for Prompt-Based Security Attacks, Establishing That Prompt Injection and Jailbreaking Are Distinct Attack Classes Requiring Distinct Defenses, and Documenting That No Known Defense Is Fully Effective

The paper’s security taxonomy (Section 5.1) makes a conceptual contribution by formally distinguishing two classes of prompt-based attacks that are frequently conflated in both research and practice: **prompt injection** (overriding developer instructions embedded in a prompt template by inserting contradictory user input) and **jailbreaking** (getting a model to produce unintended outputs through direct prompting, without developer instructions to override). The distinction is architectural: prompt injection exploits the inability of models to distinguish between developer-provided instructions and user-provided input when both are concatenated into the same prompt; jailbreaking exploits fundamental limitations in the model’s training to refuse harmful requests. The paper’s definitions anchor on the *mechanism* of the attack rather than its effects, creating a taxonomy that explains why defenses effective against one class may fail against the other.

Prior work had documented both phenomena—Carlini et al. (2021) on training data extraction, Goodside (2022) on prompt injection, Perez et al. (2022) on jailbreaking—but treated them as related examples of "prompt hacking" without clear boundaries. Schulhoff et al. (2023) ran "a study with hundreds of thousands of malicious prompts and found that no prompt-based defense is fully secure," but the conceptual framework for organizing the threat landscape was absent. The paper’s contribution is the taxonomy itself, which serves three functions:

1. **Diagnostic**: It allows practitioners to classify novel attacks. An attack that inserts text into a user input field of a template (e.g., "Recommend a book for: {USER_INPUT}") is prompt injection; an attack that directly asks the model for harmful content is jailbreaking. The distinction matters because defenses differ—prompt injection requires architectural separation of instructions from user input, while jailbreaking requires refusal training or output filtering.

2. **Predictive**: The taxonomy implies that prompt injection may be fundamentally unsolvable through prompt engineering alone, because it is "an architectural problem resulting from GenAI models not being able to understand the difference between original developer instructions and user input instructions" (Section 5.1.1). This is a stronger claim than "current defenses are inadequate"—it is a claim about inherent limitations of the prompt-as-concatenated-string abstraction. The paper does not prove this claim, but the taxonomy makes it precise enough to be tested.

3. **Practical**: The paper catalogs real-world consequences across categories: training data reconstruction (Nasr et al., 2023), prompt leaking (Willison, 2022), package hallucination attacks (Lanyado et al., 2023), code vulnerabilities (Pearce et al., 2021, 2022), and customer service manipulation (Bakke, 2023; Garcia, 2024). This evidence base transforms security from a theoretical concern into a documented operational risk, with the Garcia (2024) case—where an airline chatbot gave legally binding incorrect information and the airline lost in court—establishing legal precedent that prompt behavior can create binding commitments.

The hardening measures discussion (Section 5.1.3) introduces a three-tier defense taxonomy (prompt-based defenses, detectors, guardrails) that maps cleanly onto industry practice but had not been systematized in the literature. The finding that "no prompt-based defense is fully secure, though they can mitigate prompt hacking to some extent" (citing Schulhoff et al., 2023) establishes a clear hierarchy: detectors (fine-tuned models trained on malicious prompts) outperform prompt-based defenses, but neither category solves the problem. The paper’s honesty about the unsolved nature of these threats—"prompt hacking (both injection and jailbreaking) remain unsolved problems and likely are impossible to solve entirely"—is itself a contribution, preventing overinvestment in approaches that can only mitigate rather than eliminate risk.

---

### Innovation 4: Demonstrating Through Meta-Analysis That the Field’s Collective Knowledge Is Concentrated on a Small Number of Techniques, Benchmarks, and Models, Creating Implicit Blind Spots That the Taxonomy Partially Addresses

The paper’s meta-analysis (Section 2.3, Figures 2.9–2.11) quantifies something the field had long suspected but never measured: the extreme concentration of research attention on a handful of approaches. The top 25 cited prompting papers within the corpus are dominated by a small set of techniques—Few-Shot Learning, Chain-of-Thought, Self-Consistency—while the other 50+ techniques in the paper’s taxonomy receive disproportionately less attention. Similarly, model usage concentrates on GPT-3, BERT, GPT-4, and RoBERTa (Figure 2.9), and benchmark usage concentrates on GSM8K, MMLU, and a few others (Figure 2.10).

This is not merely a descriptive finding. It has structural implications for the field:

- **Technique evaluation is confounded with benchmark-model pairs**. If most techniques are evaluated on a small set of benchmarks using a small set of models, claims about technique effectiveness may not generalize. The paper’s own benchmarking (Section 6.1) provides evidence for this: the finding that Zero-Shot-CoT underperforms Zero-Shot on MMLU with GPT-3.5-turbo contradicts the common narrative that CoT universally improves reasoning. This suggests that technique effectiveness is benchmark- and model-dependent, and the concentration of evaluation on popular benchmarks creates a distorted picture of technique value.

- **Technique discovery is path-dependent**. The techniques that receive attention are those that were proposed early (Few-Shot, CoT) or that built on those early successes (Self-Consistency, Tree-of-Thought). Techniques in less-explored categories—Self-Criticism, Decomposition beyond Least-to-Most—may be underappreciated not because they are less effective but because they are harder to discover, benchmark, and cite within the existing citation network. The taxonomy partially addresses this by surfacing techniques that the meta-analysis shows are under-cited relative to their potential.

- **Model diversity matters for technique claims**. If most prompting research uses GPT-family models, techniques optimized for GPT-3/4’s specific behaviors may not transfer to models with different pretraining distributions, tokenization schemes, or instruction-tuning procedures. The paper’s decision to benchmark on GPT-3.5-turbo while acknowledging this limitation (Section 6.1) models a self-aware approach to model-dependence.

The meta-analysis methodology itself—using GPT-4-1106-preview to extract model and dataset mentions from 1,565 papers, followed by manual filtering and Semantic Scholar citation counting—represents a scalable approach to field-level bibliometrics that could be applied to other rapidly growing research areas. The validation against human annotations (89% precision, 75% recall) provides a calibration baseline for future meta-analyses using LLM-assisted extraction.

---

### Innovation 5: Resolving the Longstanding Definitional Ambiguity Around "In-Context Learning" by Tracing the Contradiction to Brown et al. (2020)’s Two Incompatible Definitions and Explicitly Adopting the Broader Interpretation for the Taxonomy

This is a subtle but important conceptual contribution. Brown et al. (2020)—the foundational paper that introduced in-context learning—contains two definitions that the paper shows are in tension: a broad definition ("using the text input of a pretrained language model as a form of task specification") and a narrow definition ("few-shot learning, or in-context learning where we allow as many demonstrations as will fit"). The narrow definition excludes zero-shot prompting (no demonstrations) and instruction-only prompting; the broad definition includes both. The field has operated with this ambiguity unresolved, with different papers implicitly adopting different definitions.

The paper makes three moves to resolve this (Appendix A.9):

1. **Surfaces the contradiction explicitly**, quoting both definitions from Brown et al. and showing that they are incompatible. This is valuable because many practitioners are unaware that the canonical source is internally inconsistent.

2. **Locates Brown et al.’s own attempted resolution**, quoting their clarification that the terms are "intended to remain agnostic on the question of whether the model learns new tasks from scratch at inference time or simply recognizes patterns seen during training." This establishes that the broader interpretation is closer to the original authors’ intent.

3. **Adopts the broad definition for the taxonomy** while explicitly noting that "practitioners often use ICL to refer to situations in which the model appears to be learning new tasks from the prompt." This is a pragmatic choice: the broad definition allows the taxonomy to include zero-shot and instruction-only techniques under the ICL umbrella while still flagging the narrower usage for readers who may expect it.

The significance of this resolution extends beyond terminology. By adopting the broader definition, the taxonomy groups Few-Shot Prompting, Zero-Shot Prompting (with its sub-techniques like Role Prompting and Emotion Prompting), and Instruction Selection under a single category. This grouping reflects a functional commonality: all these techniques specify tasks to the model through the prompt content rather than through weight updates. The alternative—treating zero-shot as fundamentally different from few-shot—would obscure this commonality and create a misleading boundary in the technique space. The paper’s resolution thus has downstream effects on the entire taxonomy structure, and the explicit documentation of the reasoning behind the choice allows future work to adopt the same definition with full awareness of the alternatives.

## 5. Experimental Analysis

### Evaluation Methodology

- **Dataset.** The primary benchmarking in Section 6.1 uses **MMLU** (Hendrycks et al., 2021), specifically "a representative subset of 2,800 MMLU questions (20% of the questions from each category)" with the `human_sexuality` category excluded because "gpt-3.5-turbo refused to answer these questions." The case study in Section 6.2 uses a labeled dataset of 221 posts from r/SuicideWatch (from the University of Maryland Reddit Suicidality Dataset), split into 121 development posts and 100 test posts, coded for entrapment by two trained annotators with Krippendorff's α = 0.72.

- **Base model.** The benchmarking uses **gpt-3.5-turbo** for all experiments (Section 6.1). The choice is pragmatic—it represents a widely accessible, commonly used model—but the paper acknowledges this limits generalizability since technique effectiveness is known to be model-dependent. The case study uses multiple models: initially gpt-4-turbo-preview (for checking entrapment knowledge), then **GPT-4-32K** (after gpt-1106-preview refused to label, instead offering mental health advice), and finally gpt-4-0125-preview for the DSPy comparison. The meta-analysis (Section 2.3) uses **GPT-4-1106-preview** for paper classification and model/dataset extraction from the corpus.

- **Metrics.** The benchmarking uses **accuracy** (% of correctly answered MMLU questions), with answer correctness determined by pattern matching: answers are marked correct "if they followed certain identifiable patterns, such as being the only capitalized letter (A-D) within parentheses or following a phrase like 'The correct answer is'" (Section 6.1.4). The case study uses **F1 score** (harmonic mean of precision and recall), with precision and recall also reported separately. The meta-analysis uses **citation counts** within the 1,565-paper corpus as a proxy for technique usage, and **counts** of model and dataset mentions extracted by GPT-4-1106-preview followed by manual filtering (Section 2.3).

- **Baselines.** The benchmarking uses several baselines compared against each other: **Zero-Shot** (no exemplars, no thought inducer—just the base instruction and question), **Zero-Shot-CoT** (with three thought inducers tested: "Let's think step by step" from Kojima et al. (2022), ThoT from Zhou et al. (2023), and Plan and Solve from Wang et al. (2023f)), **Few-Shot** (exemplars generated by one of the authors), and **Few-Shot-CoT** (exemplars with chain-of-thought reasoning). Self-Consistency variants are treated as augmentations of these baselines. The case study has no formal baseline in the traditional sense—the starting point is a prompt that "wouldn't return properly structured responses" (0% usable), effectively a zero baseline.

- **Generation budget / compute accounting.** The benchmarking uses **number of generations**: 1 for Zero-Shot and standard Few-Shot, 1 for standard CoT variants, and 3 for Self-Consistency variants ("three iterations, taking the majority response"). Temperature is set to 0 for all non-Self-Consistency experiments and 0.5 for Self-Consistency (following Wang et al. (2022)'s guidelines). The case study does not track generation budget systematically—it's a process documentation rather than a compute-efficiency study—but implicitly the 47 development steps represent thousands of individual LLM calls across different prompt variations.

- **Cross-validation / statistical protocol.** The benchmarking uses **no formal cross-validation** on MMLU—the 2,800-question subset is evaluated once per technique variant, with six total variations for most techniques (2 question formats × 3 base instruction phrasings). The case study uses a **fixed train-test split** (121 development, 100 test) with no cross-validation, and the prompt engineer reports that "F1 scores could change by as much as 0.04 upon subsequent runs, even with temperature and top p set to zero" (Section 6.2.3.3), acknowledging non-determinism. The meta-analysis validates the LLM-based paper classifier against "100 ground-truth annotations, achieving 89% precision and 75% recall (for an F1 of 81%)" (Section 2.1.1), and human annotator reliability is measured on 300 papers with 92% agreement, Krippendorff's α = 0.81, Cohen's κ = 0.81.

### Main Quantitative Results

#### Benchmarking Prompting Techniques on MMLU (Section 6.1)

The benchmarking experiment compares six prompting technique configurations on 2,800 MMLU questions using gpt-3.5-turbo. Results are presented in Figure 6.1:

- **Zero-Shot**: 0.627 accuracy, with purple error bars showing the minimum and maximum across the six variations (2 question formats × 3 base instruction phrasings).
- **Zero-Shot-CoT**: 0.547 accuracy, representing a **0.080 absolute drop** from Zero-Shot. The paper notes this "dropped precipitously from Zero-Shot" and that "for all variants, Zero-Shot performed better." The spread is wide, indicating substantial format sensitivity.
- **Zero-Shot-CoT with Self-Consistency**: 0.574 accuracy, recovering some of the CoT drop but still below Zero-Shot. As expected for Self-Consistency, the spread is narrower since it uses a single technique repeated.
- **Few-Shot**: 0.652 accuracy, a 0.025 absolute improvement over Zero-Shot.
- **Few-Shot-CoT**: 0.692 accuracy, the highest overall. This represents a 0.065 improvement over Zero-Shot and a 0.040 improvement over Few-Shot without CoT.
- **Few-Shot-CoT with Self-Consistency**: 0.691 accuracy, essentially identical to Few-Shot-CoT without Self-Consistency. The paper notes Self-Consistency "only improved accuracy for Zero-Shot prompts" (Section 6.1.5).

The paper explicitly flags the counterintuitive result: **Zero-Shot-CoT underperforms Zero-Shot**. This contradicts the common narrative that chain-of-thought prompting universally improves reasoning performance. The finding is specific to GPT-3.5-turbo on MMLU, and the paper does not investigate why this occurs, but notes that "unexplained performance drops from certain techniques need further research" (Section 6.1.5).

#### Prompt Engineering Case Study: Entrapment Detection (Section 6.2)

The case study tracks 47 development steps over approximately 20 hours. Performance is measured on the 121-instance development set (since exemplars are drawn from the first 10 or 20 instances, evaluation is on the remaining development instances). Key quantitative results (Figure 6.6):

- **Zero-Shot + Context** (first viable prompt, Figure 6.10): 0.40 F1, with 1.0 recall and 0.25 precision. The perfect recall indicates the model labeled everything as entrapment—a degenerate classifier that achieves high recall by never saying "no."
- **10-Shot + Context** (Figure 6.11): 0.45 F1 (↑0.05 from previous best), 0.91 recall (↓0.09), 0.30 precision (↑0.05).
- **1-Shot AutoDiCoT + Full Context** (Figures 6.12, 6.13): 0.36 F1 (↓0.09 from best). The paper notes this "did not improve F1" but "led the prompt engineer in a direction that did."
- **10-Shot AutoDiCoT** (Figure 6.16, the best performing manual prompt): 0.53 F1, 0.86 recall, 0.38 precision. This was achieved by generating 10 AutoDiCoT exemplars using the algorithm in Figure 6.12 and including them in the prompt with full context (entrapment definition + professor's email + explicitness instruction).

Key intermediate findings during the development process include:

- **Removing the email context** from the best prompt caused a drop to 0.18 F1 (↓0.27 from 0.45), with recall collapsing to 0.17 (↓0.75). The paper describes this as performance falling off "a cliff" (Appendix A.7.3).
- **De-duplicating the accidentally duplicated email** caused a drop to 0.45 F1 (↓0.07 from the 10-Shot AutoDiCoT's 0.53 with the duplication intact). The paper explicitly notes this is counterintuitive: "it seemed reasonable that removing the duplication of the email would perform as well or better than the prompt with the unintentional duplication. As it turned out, however, removing the duplicate significantly hurt performance" (Section 6.2.3.3).
- **Anonymizing the email** (replacing real names with random ones) decreased performance to 0.45 F1 (↓0.08).
- **Triplicating the context** (after de-duplicating the email) did not improve performance: 0.47 F1 (↓0.06).

On the test set (Figure 6.19), the comparison between manual and automated methods shows:

- **10-Shot AutoDiCoT** (best manual prompt): approximately 0.35 F1, with precision around 0.15 and recall around 0.55 (approximate from bar chart).
- **20-Shot AutoDiCoT**: worse than 10-Shot on the test set, with approximately 0.27 F1.
- **DSPy Default**: approximately 0.45 F1, with precision around 0.25 and recall around 0.85.
- **DSPy Default + Small Modifications**: 0.548 F1, with 0.385 precision and 0.952 recall.

The DSPy-optimized prompt substantially outperforms the human engineer's best manual prompt on the held-out test set (0.548 vs. ~0.35 F1), achieving this "without making any use of the professor's email nor the incorrect instruction about the explicitness of entrapment" (Section 6.2.3.3).

#### Meta-Analysis: Technique, Model, and Dataset Usage (Section 2.3)

The paper's citation analysis within its 1,565-paper corpus yields quantitative measures of research attention:

- **Model usage** (Figure 2.9): GPT-3 is the most-cited model (approximately 500 mentions), followed by BERT (~480), GPT-4 (~380), RoBERTa (~280), PaLM (~250), LLaMA (~240), and BART (~200). The distribution is heavily concentrated on transformer-based models, with GPT-family and BERT-family dominating.

- **Dataset usage** (Figure 2.10): GSM8K is the most-cited benchmark (approximately 800 mentions), followed by MMLU (~520), BBH (~250), CommonsenseQA (~200), HellaSwag (~180), BIG-bench (~170), WinoGrande (~160), QASC (~150), AQUA-RAT (~140), and TruthfulQA (~130). Math and reasoning benchmarks dominate, with knowledge-intensive benchmarks (MMLU, CommonsenseQA) also heavily represented.

- **Technique citations** (Figure 2.11): The top 25 cited prompting-related papers within the dataset show Few-Shot Learning and Zero-Shot Reasoning in the ~10^3 citation range, followed by papers on Good In-Context Examples, Self-Consistency, Prompt Order Sensitivity, Least-to-Most Prompting, and Prompt Retrieval in the ~10^2 range. The paper notes that "most of them propose new prompting techniques" and that "the prevalence of citations for Few-Shot and Chain-of-Thought prompting is unsurprising and helps to establish a baseline for understanding the prevalence of other techniques" (Section 2.3).

The paper does not provide exact numerical values for these citation counts in the text—they must be inferred from the bar charts—which limits precise interpretation. The key qualitative finding is the steep concentration gradient: a small number of techniques and benchmarks receive the vast majority of research attention.

### Ablation Studies and Robustness Checks

The paper's ablation structure differs from a typical methods paper because the primary contribution is taxonomic rather than algorithmic. Ablations take the form of technique variations within the benchmarking and case study, and validation of the systematic review methodology.

**Question format variation (Section 6.1.2):** The benchmarking tests two question formats from Sclar et al. (2023b)—a structured bullet format (Figure 6.3) and a compact format (Figure 6.4)—across all six technique configurations. The wide error bars in Figure 6.1 for non-Self-Consistency techniques reflect this variation, confirming format sensitivity on MMLU with GPT-3.5-turbo. The paper does not report per-format breakdowns, so the direction and magnitude of format effects for individual techniques cannot be assessed from the reported data.

**Base instruction phrasing variation (Section 6.1.1):** Three variations of the base instruction are tested for each technique. The base instruction is "a phrase like 'Solve the problem and return (A), (B), (C) or (D).'" The paper does not specify the exact three phrasings. Combined with the two question formats, this yields six total variations per (non-Self-Consistency) technique. The error bars in Figure 6.1 capture this variation, showing that Zero-Shot-CoT has the widest spread, indicating high sensitivity to exact phrasing.

**Thought inducer variation (Section 6.1.1):** For Zero-Shot-CoT, three thought inducers are compared: the standard "Let's think step by step" (Kojima et al., 2022), ThoT ("Walk me through this context in manageable parts step by step, summarizing and analyzing as we go," from Zhou et al., 2023), and Plan and Solve ("Let's first understand the problem and devise a plan to solve it. Then, let's carry out the plan and solve the problem step by step," from Wang et al., 2023f). The paper selects the best of these for the Self-Consistency run but does not report per-inducer performance, making it impossible to determine which inducer performed best or by how much.

**Self-Consistency iteration count (Section 6.1.3):** Self-Consistency uses three iterations with temperature 0.5 (following Wang et al., 2022). The paper does not ablate the number of iterations—this is a fixed hyperparameter choice rather than a tested variable. Given the finding that Few-Shot-CoT with Self-Consistency performs nearly identically to Few-Shot-CoT without (0.691 vs. 0.692), the paper implicitly suggests that three iterations of Self-Consistency may not be beneficial for this model-task combination, but does not test whether more or fewer iterations would change this.

**Explicit vs. implicit entrapment instruction (Section 6.2.3.3):** The case study contains an implicit ablation through the prompt engineer's decision to add "IMPORTANT: Only label the post as entrapment if they explicitly say that they feel trapped." This instruction improved F1 on the development set but diverged from the clinical goal of detecting implicit entrapment. The DSPy prompt, which did not include this instruction, achieved better test-set performance (0.548 vs. ~0.35 F1), suggesting the explicitness constraint was counterproductive. However, this is confounded with other differences between the prompts and cannot be isolated as a clean ablation.

**Email content variations (Section 6.2.3.3):** The case study serendipitously ablates several email-related factors. Removing the email entirely: ↓0.27 F1 (from 0.45 to 0.18). De-duplicating the accidentally doubled email: ↓0.07 F1 (from 0.53 to 0.45). Anonymizing names in the email: ↓0.08 F1 (from ~0.53 to 0.45). Each of these is a large effect by the standards of the task (where the best F1 achieved is 0.53). The paper does not explain why the email matters—it contains "richer background information about the goals of the labeling"—but the magnitude and consistency of the effects suggest the email is providing substantive task specification that the entrapment definition alone does not capture.

**DSPy vs. manual prompt engineering (Section 6.2.3.3, Figure 6.19):** The comparison between the best manual prompt (10-Shot AutoDiCoT) and DSPy-optimized prompts on the test set serves as an implicit ablation of the prompt engineering methodology itself. DSPy achieves 0.548 F1 vs. the manual prompt's ~0.35, suggesting automated optimization outperforms manual engineering for this task. However, the paper does not report whether DSPy was given the same 20-hour time budget, the same access to the development set, or the same model—DSPy used gpt-4-0125-preview while the manual prompts used GPT-4-32K, making this a confounded comparison between both methodology and model.

**LLM-based paper classifier validation (Section 2.1.1):** The GPT-4-1106-preview classifier used in the systematic review is validated against 100 ground-truth human annotations, achieving 89% precision and 75% recall (F1 = 81%). The higher precision than recall indicates conservative classification—fewer false positives than false negatives—which the paper implicitly accepts as the correct tradeoff for taxonomy construction, since including irrelevant papers would add noise to the technique catalog while missing relevant papers only means the taxonomy is incomplete rather than incorrect.

**Human inter-annotator reliability (Section 2.1.1):** On a set of 300 papers independently reviewed by two annotators, agreement is 92%, with Krippendorff's α = 0.81 and Cohen's κ = 0.81. These values are in the "almost perfect" range (Landis and Koch, 1977), providing confidence that the inclusion/exclusion criteria are applied consistently. The paper does not report per-criterion agreement, so it is unknown whether certain criteria (e.g., "proposes a novel prompting technique" vs. "focuses on training by backpropagating gradients") are more ambiguous than others.

**Non-determinism despite temperature=0 (Section 6.2.3.3):** The case study observes that "F1 scores could change by as much as 0.04 upon subsequent runs, even with temperature and top p set to zero." This is not presented as a formal ablation but serves as an important robustness observation: it means the exact performance numbers reported in the case study have an inherent variability of approximately ±0.04 F1, making some of the smaller reported improvements (e.g., 0.05 F1 gains) potentially within the noise floor. The paper does not report multiple runs or confidence intervals to account for this variability.

### Critical Assessment

The experiments in this paper serve a fundamentally different purpose from those in a typical methods paper. Rather than testing whether a proposed technique outperforms baselines, the experiments demonstrate the utility of the paper's organizational framework, reveal properties of prompt engineering as a practice, and validate the methodology used to construct the taxonomy. This means the standard evaluation rubric—"do the experiments support the claimed improvements?"—must be adapted. Instead, we should ask: do the experiments demonstrate what the paper claims they demonstrate?

**On the claim that the taxonomy enables systematic comparison of techniques:** The benchmarking experiment (Section 6.1) partially supports this. By testing six techniques on the same model, same dataset, and same evaluation protocol, the paper demonstrates that technique choice matters substantially—accuracy ranges from 0.547 (Zero-Shot-CoT) to 0.692 (Few-Shot-CoT), a 0.145 absolute difference. This validates the premise that a taxonomy helping practitioners navigate technique choices has practical value. However, the experiment is limited in ways that undermine stronger claims: (a) only six techniques are tested, out of 58 in the taxonomy—the vast majority of the technique space is unexplored; (b) only one model (GPT-3.5-turbo) is used, despite the meta-analysis showing techniques are evaluated on many models; (c) only one benchmark (MMLU) is used, despite the meta-analysis showing GSM8K is actually more common; (d) the 2,800-question subset (20% of MMLU) is evaluated once per configuration, without the multiple runs needed to account for the non-determinism the case study documents. The experiment is better understood as an illustration of the taxonomy's potential rather than a validation of its completeness.

**On the claim that the case study documents the prompt engineering process:** This claim is well-supported by the evidence presented. The 47-step annotated record, with specific prompts, extractor choices, and quantitative performance tracking, genuinely captures a process that had not been documented in the literature. However, the case study has several limitations as evidence: (a) it involves a single prompt engineer on a single task—the extent to which the observed patterns (sensitivity to irrelevant details, divergence from domain goals, accidental discoveries) generalize to other engineers, tasks, and domains is unknown; (b) the prompt engineer is also the lead author of the paper, raising the possibility of hindsight bias in the documentation (the annotations were likely written after the fact, not contemporaneously); (c) the ~20-hour time investment limits replicability—other researchers cannot easily verify the findings by repeating the process; (d) the domain (suicide risk detection) is unusually sensitive, which may have exaggerated certain behaviors (e.g., the model's refusal to label, the engineer's caution about explicit vs. implicit expressions) that would not appear in less charged domains.

**On the claim that the meta-analysis reveals concentration in the research literature:** The citation analysis (Figures 2.9-2.11) supports this claim, but with important caveats. First, the analysis measures citations *within the 1,565-paper corpus*, not citations in the broader literature. A paper that is heavily cited in the prompting literature but not in the wider NLP literature would appear prominent in Figure 2.11, while a paper with broad impact beyond prompting would be underrepresented. Second, the model and dataset extraction uses GPT-4-1106-preview at 75% recall, meaning approximately 25% of mentions were missed, and the paper does not analyze whether the misses are systematic (e.g., disproportionately affecting less common models). Third, the paper does not control for paper age—older papers have had more time to accumulate citations, which could explain some of the concentration without implying that newer techniques are less valuable. Fourth, the paper does not distinguish between papers that *use* a technique and papers that *propose* a technique, collapsing two very different forms of influence into a single citation count.

**On the implicit claim that the systematic review methodology is reproducible:** The paper provides detailed documentation of the PRISMA process (Section 2.1, Figure 2.1), the keyword list (Appendix A.4), the LLM classification prompt (Appendix A.5), and the datasheet (Appendix A.3). The inter-annotator reliability metrics (92% agreement, α = 0.81) and classifier validation (89% precision, 75% recall) provide quantitative evidence that the review was systematic rather than impressionistic. However, several aspects limit reproducibility: (a) the 1,661 human-reviewed papers were drawn "from the arXiv set" specifically, not from all three sources, which means the distribution of inclusion criteria across Semantic Scholar and ACL papers may differ; (b) the remaining 1,361 human-reviewed papers (after the 300 double-annotated) were reviewed by a single annotator, so their reliability is assumed rather than measured; (c) the LLM classifier was validated on only 100 papers, and its performance may vary across different parts of the paper distribution (e.g., it may be less accurate on papers with unusual abstracts); (d) the final dataset of 1,565 papers is not stratified by source, publication year, or technique category, so it is unknown whether certain subfields of prompting are over- or under-represented.

**Critical missing experiments:** Several experiments would have substantially strengthened the paper's contributions:

1. **Technique benchmarking across multiple models.** Running the six MMLU techniques on at least GPT-4 and one open-source model (e.g., LLaMA) would test whether the observed rankings (Few-Shot-CoT > Few-Shot > Zero-Shot > Zero-Shot-CoT) generalize. The case study shows that model choice critically affects behavior (GPT-4-32K gave one-word labels while gpt-1106-preview gave mental health advice), suggesting model dependence may be large.

2. **Multi-run reliability in benchmarking.** Given the case study's finding of ±0.04 F1 variation at temperature=0, the benchmarking results should include multiple runs with error bars representing real variability rather than just variation across phrasings. A single run per configuration conflates prompt sensitivity with inherent stochasticity.

3. **Difficulty-stratified analysis in benchmarking.** The MMLU categories vary in difficulty, and the case study shows that technique effectiveness depends on task characteristics (explicit vs. implicit entrapment). Stratifying benchmarking results by MMLU category or question difficulty could reveal whether Zero-Shot-CoT's underperformance is uniform or concentrated in certain categories.

4. **Controlled ablation of the email effect.** The case study's most striking finding—that duplicating an email substantially affects performance—is purely observational. A controlled experiment systematically varying the content, length, position, and duplication of auxiliary text in prompts could determine whether this is a general phenomenon or specific to this task and model.

5. **Inter-engineer reliability in prompt engineering.** Having a second prompt engineer independently develop prompts for the entrapment task would test whether different engineers converge on similar strategies or discover different "black art" tricks—directly testing the paper's characterization of prompt engineering as lacking systematic principles.

6. **Coverage analysis of the taxonomy.** The paper acknowledges "there are sure to be gaps and redundancies" (Section 8) but provides no quantitative measure of coverage. Comparing the taxonomy to techniques discovered after the review cutoff date, or having independent experts assess completeness, would provide evidence about how comprehensive the taxonomy actually is.

**What the experiments demonstrate vs. what they claim:** The paper's experiments demonstrate that (a) technique choice matters for performance, (b) prompt engineering is a non-trivial process characterized by sensitivity to opaque factors, and (c) the research literature is concentrated on a subset of techniques and benchmarks. These are genuine findings. However, the experiments do not demonstrate that the taxonomy is *correct* in its functional groupings (as opposed to other possible organizations), that the terminology will be adopted by the community, or that the "compute-optimal" allocation concept from the context paper applies to prompt engineering—the paper makes no such claims. The experiments are best understood as an initial empirical exploration accompanying a primarily conceptual and organizational contribution, providing evidence that the organizational work is needed and plausible, not that it is complete or definitively validated.

## 6. Limitations and Trade-offs

### 6.1 The 1,565-Paper Corpus Underlying the Entire Taxonomy Has Unmeasured Coverage Gaps Due to a Conservative LLM-Based Classifier with 75% Recall

**The assumption or constraint.** The systematic review pipeline (Section 2.1) uses GPT-4-1106-preview to classify papers for inclusion after an initial human annotation phase. This classifier was validated against 100 ground-truth human annotations and achieved 89% precision but only **75% recall** (F1 = 81%). The paper acknowledges this explicitly: "The higher precision than recall suggests the LLM is conservative, preferring to miss relevant papers rather than include irrelevant ones."

**The consequence.** At 75% recall, approximately **one in four actually relevant papers in the unannotated portion of the corpus was excluded**. Since the initial human review covered only 1,661 papers (from the arXiv set specifically) and 1,071 papers were reviewed by the LLM alone (Figure 2.1), approximately 250+ relevant papers may have been missed. This means the taxonomy of 58 text-based prompting techniques is **a lower bound on the true technique space**—techniques present in the missed papers are absent from the taxonomy by construction. The paper's claim to provide "the most comprehensive survey on prompt engineering to date" (Abstract) must be qualified by this systematic exclusion. More importantly, techniques concentrated in certain subfields, publication venues, or time periods that the classifier systematically misjudges would be disproportionately undercounted, potentially creating structural blind spots in the taxonomy. A practitioner consulting the taxonomy as an authoritative map of the technique space would be unaware of which regions are sparsely covered due to classifier error rather than genuine absence of techniques.

**What evidence exists in the paper.** The paper reports the classifier validation metrics directly (89% precision, 75% recall, F1 of 81%) in Section 2.1.1 and acknowledges in the datasheet (Appendix A.3.2) that "papers were gathered in a semi-automated process which introduced the possibility of irrelevant papers being collected and relevant papers not being collected." However, the paper provides **no analysis of whether the classifier's errors are systematic**—for example, whether it disproportionately misses papers on certain technique categories, from certain publication years, or with certain abstract structures. The 100-paper validation set is small relative to the 1,071 papers the classifier processed, and no breakdown of errors by paper characteristics is reported.

**Mitigation status.** The paper partially acknowledges the limitation through the datasheet statement and the transparency about recall, but does not attempt to estimate coverage or characterize the distribution of missed papers. The authors frame the taxonomy as "a first iteration of terminologies that will develop over time" (Section 1), suggesting that future work can fill gaps, but no concrete process for identifying or filling those gaps is specified. The paper suggests no method for auditing which regions of the technique space are most likely undercovered.

---

### 6.2 The Difficulty Estimation Oracle Required by Test-Time Compute-Optimal Strategies Is Prohibitively Expensive, Making the 4× Efficiency Gains an Upper Bound Unrealized in Practice

**The assumption or constraint.** The paper draws an explicit parallel to the context paper's compute-optimal test-time scaling framework (the reference example describes estimating difficulty by generating 2,048 samples per question and computing pass@1). The prompt engineering case study (Section 6.2) implicitly confronts a version of this problem: determining which prompting technique will work best for a given task requires **extensive empirical exploration**—the prompt engineer spent approximately 20 hours and 47 development steps to reach an F1 of 0.53. The paper does not frame this as a difficulty estimation problem explicitly, but the parallel is direct: the "difficulty" of a prompting task (how hard it is to achieve good performance, which techniques will work) is unknown in advance and can only be discovered through costly experimentation. The paper's benchmarking (Section 6.1) further demonstrates this: the finding that Zero-Shot-CoT underperforms Zero-Shot on MMLU with GPT-3.5-turbo (0.547 vs. 0.627) was not predictable from theory—it had to be discovered empirically.

**The consequence.** A practitioner consulting the taxonomy to decide which technique to use faces the same exploration-exploitation dilemma the context paper identifies: they must **spend substantial computation (and human time) discovering which techniques work for their specific task-model combination before they can benefit from the taxonomy's guidance**. The paper provides no mechanism for predicting technique effectiveness without running the experiments—no transfer learning between tasks, no meta-learning over technique performance, and no cheap proxy for difficulty estimation. The headline efficiency gains that the context paper reports (4× over best-of-N) assume difficulty is known; in the prompt engineering setting, difficulty estimation cost can dominate the total budget. The case study's 20-hour investment for a single task on a small dataset (221 posts) suggests that scaling this approach across many tasks, models, or domains—as the taxonomy's comprehensive scope implies—would be economically infeasible for most practitioners. The gap between knowing that 58 techniques exist and knowing which one to use remains unbridged.

**What evidence exists in the paper.** The case study (Section 6.2) provides direct evidence of the exploration cost: 47 development steps, approximately 20 hours of expert time, for a single binary classification task. The benchmarking (Section 6.1) shows that technique ranking is non-obvious (Zero-Shot-CoT underperforms Zero-Shot) and model-dependent (the paper notes this finding contradicts the common narrative about CoT, implying ranking would differ for other models). The paper's meta-analysis (Section 2.3) shows technique evaluation is concentrated on a small set of benchmarks, meaning the literature provides limited guidance about technique transfer to new tasks. Figures 6.5 and 6.6 in the case study show that most attempted prompt variations did not improve over the current best—the exploration process is largely wasteful, with only a small fraction of attempts yielding gains. The paper does not measure the total computation cost of the case study in FLOPs or API calls, making the cost unquantified but clearly substantial from the time investment alone.

**Mitigation status.** The paper makes no attempt to address this limitation systematically. The taxonomy itself is a partial mitigation—by organizing techniques, it reduces the search space from "try random things" to "try techniques in relevant functional categories"—but it provides no guidance on which category to try first for a given task. The DSPy comparison (Section 6.2.3.3) suggests automated prompt optimization as a potential solution (DSPy achieved 0.548 F1 vs. the human engineer's 0.53 in ~20 hours), but the paper does not report DSPy's computational cost or wall-clock time, making the tradeoff unquantified. The paper's conclusion that "combining that automation with human prompt engineering/revision was the most successful approach" (Section 6.2.4) acknowledges the value of automation without specifying when or how to combine it with manual effort.

---

### 6.3 All Empirical Findings Are Derived from a Single Model Family (GPT-3.5/GPT-4) on a Narrow Set of English-Language Benchmarks, with No Evidence That Technique Rankings, Sensitivity Patterns, or the Taxonomy's Categories Generalize Across Model Architectures, Languages, or Modalities

**The assumption or constraint.** The paper's primary empirical contributions—the MMLU benchmarking (Section 6.1) and the entrapment detection case study (Section 6.2)—use exclusively GPT-family models: GPT-3.5-turbo for benchmarking, GPT-4-32K and GPT-4-turbo-preview for the case study, GPT-4-1106-preview for the systematic review classifier, and GPT-4-0125-preview for DSPy. The meta-analysis (Section 2.3) shows that GPT-3, BERT, and GPT-4 are the most-cited models in the prompting literature, but the paper's own experiments never test on BERT-family or open-source models. The benchmarking uses a single dataset (MMLU, English-only) on a single model. The paper acknowledges that "techniques may not transfer to other models, problems, or datasets" (Section 8) but provides no empirical evidence about the magnitude or nature of this non-transfer.

**The consequence.** The paper's headline benchmarking result—that Few-Shot-CoT achieves the highest accuracy (0.692) while Zero-Shot-CoT underperforms Zero-Shot (0.547 vs. 0.627)—may be **entirely specific to GPT-3.5-turbo on MMLU**. A practitioner using a different model (e.g., Claude, Gemini, LLaMA, Mistral) or a different task (e.g., code generation, summarization, translation) cannot assume these rankings hold. More fundamentally, the taxonomy's functional categories (In-Context Learning, Thought Generation, Decomposition, etc.) were derived from the full literature but validated only on this narrow empirical base. If certain categories of techniques behave qualitatively differently on non-GPT models—for example, if Self-Criticism techniques that work well on GPT-4 fail entirely on LLaMA due to different instruction-following behaviors—the taxonomy's claim to be model-agnostic is unsupported. The paper's meta-analysis (Figure 2.9) shows that BERT is the second most-cited model in the prompting literature (~480 citations), yet the paper provides no evidence about whether prompting techniques developed for autoregressive models (like the GPT family) even apply to masked language models like BERT, which use cloze rather than prefix prompts—a distinction the paper itself makes (Appendix A.2.4.3).

**What evidence exists in the paper.** The paper's own evidence for model-dependence is strong but confined to the case study observations: (a) gpt-4-turbo-preview gave mental health advice instead of labels, requiring a switch to GPT-4-32K (Section 6.2.3.2), demonstrating that model choice critically affects whether prompting is even possible for a task; (b) the observation that "F1 scores could change by as much as 0.04 upon subsequent runs, even with temperature and top p set to zero" (Section 6.2.3.3) demonstrates sensitivity to model internals that the paper does not attempt to characterize or explain; (c) the meta-analysis (Figure 2.9) shows research is concentrated on a few model families, but the paper does not test whether this concentration reflects genuine superiority or path-dependence in the research community. The paper's systematic review explicitly excluded non-English papers (datasheet, Appendix A.3.5: "All of the papers we collected were written in English. It is possible some papers were not included due to a translation not being available"), meaning multilingual technique coverage is limited to English-language publications about multilingual prompting—a significant constraint for a taxonomy claiming to cover multilingual techniques (Section 3.1).

**Mitigation status.** The paper partially acknowledges this limitation in its conclusions: "we encourage the reader to avoid taking any claims at face value and to recognize that techniques may not transfer to other models, problems, or datasets" (Section 8). The case study explicitly notes that the "guard rails" issue with gpt-4-turbo-preview "could influence the choice of model for reasons other than the LLM's potential quality" (Section 6.2.3.2), implicitly recognizing model-dependence. However, the paper makes no attempt to characterize the scope of model-dependence systematically—it provides no multi-model benchmarking, no analysis of which taxonomy categories might be model-specific, and no guidance on how practitioners should assess technique transfer to their specific model. The meta-analysis's concentration findings (Figures 2.9-2.11) could have been leveraged to identify gaps but are instead presented as descriptive statistics without prescriptive implications.

---

### 6.4 The Taxonomy's Functional Categories Are Not Empirically Validated—No Experiment Tests Whether Grouping Techniques by Mechanism (Rather Than by Performance Characteristics, Compute Cost, or Robustness) Produces Meaningful or Useful Clusters

**The assumption or constraint.** The paper organizes 58 text-based techniques into six major categories (In-Context Learning, Thought Generation, Decomposition, Ensembling, Self-Criticism, and additional categories for multilingual, multimodal, agent, evaluation, security, and alignment) based on the authors' judgment about functional mechanisms—"the single category of most relevance" (Section 2.2). The paper acknowledges that "some of the techniques might fit into multiple categories" but makes no attempt to validate these assignments empirically. There is no experiment testing whether techniques within a category behave similarly (e.g., similar sensitivity to exemplar ordering), whether categories predict technique effectiveness for particular task types, or whether alternative organizations (e.g., by compute cost, by robustness to prompt variation, by required user expertise) would be more useful for practitioners.

**The consequence.** A practitioner using the taxonomy to select techniques may be misled if the functional categories do not correspond to practically meaningful distinctions. For example, the paper groups Self-Consistency under Ensembling (Section 2.2.4) while it groups Self-Refine under Self-Criticism (Section 2.2.5). Both involve generating multiple outputs and selecting or refining them—a practitioner looking for techniques to improve answer quality might reasonably expect them to be in the same category. If the functional distinction (aggregating independent outputs vs. iterative self-improvement) does not predict which technique works better for which task, the taxonomy's organization adds cognitive overhead without practical benefit. More broadly, the paper provides no evidence that its categorization scheme is *useful*—that it helps practitioners find techniques faster, select better techniques for their tasks, or understand technique relationships more accurately than a flat alphabetical list or a simpler organization (e.g., by complexity: basic → intermediate → advanced). The paper's benchmarking (Section 6.1) tests techniques from different categories (Zero-Shot from ICL, CoT from Thought Generation, Self-Consistency from Ensembling) but does not analyze whether within-category performance variance is smaller than between-category variance—the standard test for whether a clustering is meaningful. Without such evidence, the taxonomy remains a plausible organization rather than a validated one.

**What evidence exists in the paper.** The benchmarking experiment (Figure 6.1) provides the only quantitative comparison of techniques from different categories, but it tests too few techniques (6 out of 58) and too few categories (4 out of 6 major text categories—no Decomposition or Self-Criticism techniques are benchmarked) to validate the category structure. The meta-analysis (Figure 2.11) shows citation patterns but citation counts do not validate functional groupings—a technique could be heavily cited and belong to the wrong category. The paper does not report any experiment where category membership is used to predict technique behavior, nor any user study where practitioners are asked to navigate the taxonomy to find techniques for a given task. The case study (Section 6.2) shows the prompt engineer using techniques from multiple categories (Zero-Shot, Few-Shot, CoT, Contrastive CoT, AutoDiCoT, Ensembling) but does not analyze whether the categorical organization influenced technique selection—the engineer appears to select techniques based on diagnosing specific failure modes rather than following category-based navigation.

**Mitigation status.** The paper does not attempt to validate the taxonomy's categories empirically and does not acknowledge this as a limitation. The Conclusions (Section 8) frame the taxonomy as "an initial attempt to categorize the species of an unfamiliar territory" and acknowledge "there are sure to be gaps and redundancies," but this refers to coverage (missing techniques) rather than validity (whether the categories are correct or useful). The paper's emphasis on functional mechanisms as the organizing principle is a design choice presented without justification or alternatives. Future work is implicitly invited to extend the taxonomy, but no framework is provided for testing whether the extensions improve its utility.

---

### 6.5 The Case Study's Key Finding—Extreme Sensitivity to Seemingly Irrelevant Prompt Details—Is Documented but Neither Explained Nor Characterized, Leaving Practitioners with a Warning Rather Than Actionable Guidance

**The assumption or constraint.** The paper's central empirical contribution from the case study is the demonstration that prompt engineering is characterized by "incredible sensitivity to specific details in prompts without there being any obvious reason those details should matter" (Section 6.2.4). The most striking evidence: accidentally duplicating an email in the prompt improved performance, and removing the duplication caused a 0.07 F1 drop—with no clear mechanism. However, the paper provides **no characterization of the scope, causes, or predictability of this sensitivity**. It does not test whether the sensitivity is specific to long-context prompts (where attention might dilute), to the GPT-4 architecture, to emotionally charged domains (suicide risk), or to particular prompt components (instructions vs. exemplars vs. auxiliary context). It does not measure the variance of this sensitivity across multiple runs with different random seeds or across different prompt engineers. It does not test whether the sensitivity is reduced by techniques like Self-Consistency or ensembling.

**The consequence.** A practitioner reading the case study learns that prompts are unpredictably sensitive to arbitrary details—but learns nothing about how to manage this sensitivity. Should they try random variations of their prompts? Duplicate elements intentionally? Avoid long prompts? Use automated optimization to escape human biases about what "should" matter? The paper's conclusion that these systems are "being cajoled, not programmed" (Section 6.2.4) is evocative but provides no engineering principles. The gap between diagnostic (demonstrating the problem exists) and prescriptive (showing how to address it) is particularly consequential because the paper positions itself as a resource for practitioners, yet its most vivid empirical finding offers only a warning, not a solution. Worse, the paper's own evidence shows that automated methods (DSPy) achieved better test-set performance (0.548 F1) without incorporating the apparently crucial email duplication, suggesting that the sensitivity the human engineer discovered and exploited on the development set did **not transfer to the test set**—making the sensitivity not only opaque but potentially misleading as a signal for prompt improvement.

**What evidence exists in the paper.** The case study documents multiple instances of unexplained sensitivity: email removal (↓0.27 F1), email de-duplication (↓0.07 F1), email anonymization (↓0.08 F1), triplicating context (↓0.06 F1). However, each of these is a single observation on a single prompt variant, without systematic variation of the manipulated factors. The paper does not report whether the email effect replicates on other tasks, with other models, or with other "auxiliary context" text that is semantically unrelated to the task. The DSPy comparison (Figure 6.19) provides a critical counterpoint—DSPy achieved superior performance without the email—that the paper does not fully analyze: if the "crucial" email was actually irrelevant to the underlying task (as DSPy's success suggests), then the human engineer's 20 hours of development were partly spent optimizing an artifact that did not generalize. The paper presents this as evidence for combining automated and manual methods but does not provide a framework for distinguishing generalizable prompt improvements from dataset-specific overfitting.

**Mitigation status.** The paper partially addresses this through its recommendation to combine automated and manual prompt engineering: "there was significant promise in an automated method for exploring the prompting space, but also that combining that automation with human prompt engineering/revision was the most successful approach" (Section 6.2.4). However, this is a general recommendation without specific guidance—when should a practitioner trust their manual discoveries vs. defer to automated search? The paper's benchmarking (Section 6.1) demonstrates sensitivity to question format and instruction phrasing through the error bars in Figure 6.1, but these represent sensitivity to factors the experimenters *varied systematically*, not the kind of arbitrary sensitivity (email duplication) the case study revealed. The paper does not propose diagnostic experiments that practitioners could run to assess whether their prompts are overfit to arbitrary details.

---

### 6.6 The Taxonomy Provides No Guidance on Computational Cost, Latency, or Practical Deployment Constraints—Techniques Are Presented as Equivalent Options Despite Orders-of-Magnitude Differences in API Calls, Wall-Clock Time, and Monetary Cost

**The assumption or constraint.** The taxonomy organizes techniques by functional mechanism without any explicit consideration of their **computational cost**. Self-Consistency (Section 2.2.4) requires multiple model calls (3+ for the paper's benchmarking, typically more in practice) and is categorized alongside single-call techniques like Role Prompting (Section 2.2.1.3). Tree-of-Thought (Section 2.2.3) involves building and searching a tree of reasoning paths, potentially requiring dozens or hundreds of LLM calls, yet sits in the same Decomposition category as Plan-and-Solve Prompting, which requires two calls. The paper acknowledges that ensembling techniques "come with the cost of increasing the number of model calls needed to reach a final answer" (Section 2.2.4) but provides no systematic quantification of these costs across techniques. The case study extensively uses GPT-4-level models (GPT-4-32K, GPT-4-turbo-preview) without reporting the total API cost, which for ~20 hours of development work could easily reach hundreds of dollars.

**The consequence.** A practitioner selecting techniques from the taxonomy has **no information about the compute budget required** for each option. A startup building a cost-sensitive application might inadvertently select Tree-of-Thought (potentially 100+ API calls per query) when Least-to-Most (sequential sub-problem solving, fewer calls) would achieve adequate performance at a fraction of the cost. Conversely, a research lab with abundant compute might use simple Zero-Shot when Self-Consistency with 10 samples would substantially improve accuracy. More subtly, techniques with similar functional descriptions can have dramatically different latency profiles: Skeleton-of-Thought (Section 2.2.3) achieves parallelism by generating sub-questions and answering them simultaneously, reducing wall-clock time, while Recursion-of-Thought (Section 2.2.3) is inherently sequential, with each sub-problem depending on the previous answer. A latency-sensitive application (e.g., interactive chatbot) might find the functionally similar Recursion-of-Thought unusable while Skeleton-of-Thought is viable, but the taxonomy provides no signal about this distinction. The paper's benchmarking uses model generations as the unit of compute (1 for Zero-Shot, 3 for Self-Consistency), but this accounting is not extended to the full taxonomy, and even for the benchmarked techniques, the cost-per-generation varies substantially depending on prompt length, output length (CoT generates longer outputs), and model pricing tier—none of which are discussed.

**What evidence exists in the paper.** The paper provides generation counts for the benchmarked techniques (1, 1, 3 for Zero-Shot/Few-Shot, CoT variants, and Self-Consistency respectively) in Section 6.1.3. The meta-analysis (Section 2.3) provides no cost analysis—citation counts do not correlate with computational efficiency. The case study reports wall-clock time (~20 hours) but not API cost, model calls, or tokens processed. The paper's description of individual techniques occasionally implies cost differences (e.g., Tree-of-Thought "creates a tree-like search problem," implying many calls; Skeleton-of-Thought "focuses on accelerating answer speed through parallelization," explicitly mentioning efficiency), but these are qualitative descriptions rather than quantified comparisons. No figure or table in the paper maps techniques to their computational requirements.

**Mitigation status.** The paper does not attempt to address this limitation and does not acknowledge it as a gap. The scope (Section 1) focuses on "task-agnostic techniques" and excludes gradient-based methods, but cost is never mentioned as an organizing principle or a constraint. The paper's stated goal is to "create a broad directory of prompting techniques, that can be quickly understood and easily implemented for rapid experimentation by developers and researchers" (Section 1), which implies practical deployability, but cost—arguably the most important practical constraint for deployment—is absent from the taxonomy's design. Future work might annotate the taxonomy with compute cost estimates, but the paper provides no framework or methodology for doing so.

## 7. Implications and Future Directions
- How this work changes the landscape
  - Provides a shared map and vocabulary for the field (Figures 1.3, 2.2, 3.1, 3.2), making it easier to reason about design choices, compare techniques, and teach best practices.
  - Bridges prompting research with safety and evaluation, promoting end‑to‑end thinking: prompts, answer extraction, evaluation pipelines, and defenses (Sections 2.5, 4.2, 5).

- Follow‑up research enabled
  - Multi‑model, multi‑dataset replications of the benchmark to test interaction effects (e.g., when `Zero‑Shot CoT` helps vs. hurts).
  - Programmatic methods for `answer engineering` (learned extractors, structured decoding) that reduce formatting brittleness (Section 2.5).
  - Generalized `Directed CoT` methods: algorithmic selection of “what not to do” exemplars, with uncertainty‑aware sampling or RL.
  - Robust prompting under adversarial or noisy inputs; formal safety metrics for guards/detectors (Section 5.1.3).

- Practical applications and downstream use cases
  - Enterprise prompt design playbooks: exemplar selection (Figure 2.3), role/style instructions (Section 2.2.1.3), ensembling with cost control (Section 2.2.4), and built‑in answer extraction (Section 2.5).
  - Safety‑critical screening workflows (healthcare, trust & safety): prioritize high‑recall prompts, add calibration prompts (Section 5.2.2), and include human‑in‑the‑loop confirmation.
  - Agentic systems: tool routing and retrieval‑augmented reasoning patterns (`MRKL`, `ReAct`, `IRCoT`) for tasks needing factuality and planning (Section 4.1; Figure 4.1).
  - LLM‑as‑evaluator: adopt `G‑EVAL`/`LLM‑EVAL` styles with CoT and model‑generated guidelines for consistent, auditable assessments (Section 4.2).

> Overall, the work offers a coherent blueprint for designing, evaluating, and securing prompt‑driven systems—from vocabulary and taxonomy (Figures 1.3, 2.2) through empirical guidance (Figure 6.1) and real‑world procedure (Figures 6.12–6.16)—while candidly surfacing sensitivity and safety pitfalls that practitioners must manage.
