# The Prompt Report: A Systematic Survey of Prompting Techniques

**ArXiv:** [2406.06608](https://arxiv.org/abs/2406.06608)
**Authors:** Sander Schulhoff, Michael Ilie, Nishant Balepur, Konstantine Kahadze, Amanda Liu, Chenglei Si, Yinheng Li, Aayush Gupta, HyoJung Han, Sevien Schulhoff, Pranav Sandeep Dulepet, Saurav Vidyadhara, Dayeon Ki, Sweta Agrawal, Chau Pham, Gerson Kroiz, Feileen Li, Hudson Tao, Ashay Srivastava, Hevander Da Costa, Saloni Gupta, Megan L. Rogers, Inna Goncearenco, Giuseppe Sarli, Igor Galynker, Denis Peskoff, Marine Carpuat, Jules White, Shyamal Anadkat, Alexander Hoyle, Philip Resnik
**Institutions:** University of Maryland, OpenAI, Stanford University, Microsoft, Vanderbilt University, Princeton University, ...

## 🎯 Pitch

This paper offers a landmark PRISMA-grounded survey of prompt engineering, establishing a shared vocabulary, a versatile taxonomy of 58 text prompting techniques, and comprehensive evaluation and security guidelines. By clarifying fragmented concepts and empirically evaluating techniques like zero-shot chain-of-thought, this study enhances both the practical application and theoretical understanding of prompting, yielding more effective and secure AI systems.

---

## 1. Executive Summary (2–3 sentences)
This paper delivers a comprehensive, PRISMA-grounded systematic survey of prompt engineering, assembling a standardized vocabulary, a taxonomy of 58 text prompting techniques (plus 40 beyond text), security and evaluation guidance, and two empirical studies. Its significance is twofold: it clarifies a fragmented field with consistent terminology and design patterns, and it tests how widely used techniques actually perform, showing, for example, that zero‑shot chain‑of‑thought can underperform simple zero‑shot on MMLU (Section 6.1, Figure 6.1).

## 2. Context and Motivation
- Problem and gap
  - Prompting has exploded in use but suffers from conflicting terminology, overlapping concepts, and unclear best practices, making it hard for practitioners to select, compose, and evaluate techniques (Section 1; Figure 1.3).
  - Most surveys focus on subsets (e.g., English text only, or specific techniques) and lack a unified ontology or cross-cutting guidance on evaluation, security, and multimodal/agent extensions (Sections 1, 3, 4, 5).

- Why it matters
  - Practically, better prompts yield large quality gains across tasks (Section 1; citations therein). Organizations need shared language, repeatable processes (Figure 1.4), and security hygiene (Section 5.1).
  - Theoretically, clarifying what constitutes a “prompt,” “prompt chain,” “in‑context learning (ICL),” and “answer engineering” helps structure future research (Section 1.2; Section 2.5).

- Prior approaches and shortcomings
  - Existing works often treat prompting techniques in isolation, lack consistent definitions (e.g., role vs. persona prompts), and rarely assess robustness across models or prompt formats (Sections 1.2, 2.3.1, 6.1.2).
  - Standard benchmarks exist, but studies seldom compare techniques head-to-head in controlled settings (Section 6.1).

- Positioning
  - This paper offers: a machine‑assisted, PRISMA-style literature review (Section 2.1; Figure 2.1), a taxonomy spanning text, multilingual, multimodal, and agentic prompting (Figures 2.2, 3.1, 3.2, 4.1), a standardized vocabulary (Figure 1.3), guidance on evaluation (Section 4.2) and security (Section 5.1), and two empirical studies: an MMLU technique comparison and a real‑world case study in suicide-crisis signal detection (Section 6).

## 3. Technical Approach
This section explains what the survey covers, how the corpus was constructed, and how the paper structures the space of prompting, then details the two empirical studies.

- Scope and formalization
  - Focus: hard (discrete) prefix prompts (tokens corresponding to vocabulary words and predicted at the end of the input), not cloze or soft prompts, and task‑agnostic techniques (Section 1; Appendix A.2.4).
  - Prompt engineering is defined as iteratively modifying a prompt template or technique to optimize a utility function on a dataset (Figure 1.4; Appendix A.8). The process explicitly includes an `extractor` (for answer engineering) converting raw model output into final answers (Figure 1.4; Section 2.5).

- Systematic review pipeline (Section 2.1; Figure 2.1; Appendix A.4, A.5)
  - Data sources: arXiv, Semantic Scholar, ACL.
  - Retrieval: 4,247 unique records after de‑duplication.
  - Human screening: 1,661 papers sampled; dual annotation with 92% agreement (Krippendorff’s α ≈ Cohen’s κ = 0.81).
  - LLM screening: a GPT‑4–based classifier tuned on 100 annotated samples (precision 89%, recall 75%, F1 81%) then applied to remaining papers.
  - Final corpus: 1,565 papers meeting inclusion criteria (hard, prefix prompts; non‑gradient methods; allowance for masked frames in non‑text modalities).

- Organizing the field
  - Vocabulary and components of a prompt (Section 1.2.1; Figure 1.3):
    - `Directive` (instruction or question), `Examples` (exemplars/shots), `Output formatting` (e.g., CSV, XML), `Style instructions`, `Role/persona`, and `Additional information`.
  - Answer engineering (Section 2.5; Figure 2.13): distinguish `answer shape` (format, e.g., token/span), `answer space` (allowable values), and `extractor` (rule/regex/LLM/verbalizer).
  - Taxonomy of techniques (Section 2.2; Figure 2.2): six families—ICL, Thought Generation (e.g., CoT), Decomposition (e.g., Least‑to‑Most, Tree‑of‑Thought), Ensembling (e.g., Self‑Consistency), Self‑Criticism (e.g., Self‑Refine, Chain‑of‑Verification), and Zero‑shot prompting variants (e.g., role, style, emotion prompting).
  - Beyond English text: multilingual prompting (Section 3.1; Figure 3.1), including translate-first strategies, cross‑lingual CoT (XLT, CLSP), exemplar selection (X‑InSTA), and prompt language choice trade‑offs.
  - Multimodal prompting (Section 3.2; Figure 3.2): image (prompt modifiers, negative prompts, multimodal ICL, CoT extensions), audio (early-stage ICL), video (generation/editing), segmentation, and 3D prompting.
  - Agents and tools (Section 4.1; Figure 4.1): tool‑use agents (MRKL, CRITIC), code‑generation agents (PAL, ToRA, TaskWeaver), observation‑based agents (ReAct, Reflexion; lifelong learning with Voyager, GITM), and RAG variants (IRCoT, FLARE, Verify‑and‑Edit, DSP).
  - Evaluation with LLMs as judges (Section 4.2; Figure 4.2): prompt designs (ICL, role, CoT), output formats (binary, Likert, structured JSON/XML), and prompting frameworks (LLM‑EVAL, G‑EVAL, ChatEval).

- Empirical Study 1 — MMLU technique comparison (Section 6.1)
  - Setup: GPT‑3.5‑turbo; 2,800 questions (20% from each MMLU subdomain; human_sexuality excluded due to refusals), two multiple‑choice formats (Figures 6.3, 6.4), three instruction phrasings, self‑consistency at temperature 0.5, others at 0 (Sections 6.1.1–6.1.3).
  - Techniques compared: Zero‑Shot; Zero‑Shot‑CoT (with three thought inducers, best one then used with Self‑Consistency); Few‑Shot; Few‑Shot‑CoT; and Self‑Consistency variants (Section 6.1.1).

- Empirical Study 2 — Real‑world case study on suicidal crisis signal (“entrapment”) (Section 6.2)
  - Task: label presence of “entrapment/frantic hopelessness”, a core signal in Suicide Crisis Syndrome (SCS), in Reddit r/SuicideWatch posts (Section 6.2.1).
  - Data: 221 posts; two clinically trained coders; Krippendorff’s alpha = 0.72; split: 121 development, 100 test (Section 6.2.2).
  - Prompt engineering process:
    - Initial exploration revealed guardrails causing advice rather than labels; switching to `gpt‑4‑32k` mitigated this (Section 6.2.3.2).
    - Iterative techniques tested: Zero‑Shot + context (definition); 10‑Shot + context; several CoT and contrastive variants; an “Automatic Directed CoT (AutoDiCoT)” pipeline that generates/extracts reasoning from correct/incorrect cases, then includes the “how not to reason” exemplar (Algorithm in Figure 6.12; Section 6.2.3.3; prompt in Figure 6.13).
    - Extraction logic evolved from first‑token to last‑token matchers and LLM‑based extractors (Section 6.2.3.3; Figure 6.18).
  - Final additions: full‑context snippets (including a project email that, when duplicated by accident, unexpectedly improved performance), and evaluation on both dev and test (Figures 6.5, 6.6, 6.19).

## 4. Key Insights and Innovations
- A unified, practical vocabulary and process model (Figure 1.3; Figure 1.4; Section 2.5)
  - What’s new: clear separation of “prompt components,” “prompt engineering,” and “answer engineering,” with an explicit `extractor` and `verbalizer` concept. Earlier works often conflated these or omitted the extraction step.
  - Why it matters: encourages reproducible pipelines (template → inference → extractor → utility) and makes evaluation choices explicit (Section 2.5).

- A comprehensive taxonomy that cuts across text, multilingual, multimodal, agents, and evaluation (Figures 2.2, 3.1, 3.2, 4.1, 4.2)
  - What’s new: 58 text techniques organized into families; 40 non‑text techniques; agent and RAG prompting patterns; standardized evaluator prompt designs.
  - Why it matters: helps practitioners map needs to techniques (e.g., decomposition vs. ensembling), and reveals extension paths (e.g., CoT → Chain‑of‑Images for reasoning with SVGs; Section 3.2.1.2).

- Evidence that technique effects are conditional and formatting‑sensitive (Sections 6.1.2, 6.1.5; Figures 6.1, 6.3, 6.4)
  - What’s new: head‑to‑head result showing Zero‑Shot‑CoT can underperform Zero‑Shot on MMLU with GPT‑3.5‑turbo (0.547 vs. 0.627), while Few‑Shot‑CoT performs best (0.692) (Figure 6.1).
  - Why it matters: challenges the assumption that CoT is always better; highlights the importance of prompt format and instruction phrasing (Sections 6.1.1–6.1.2).

- AutoDiCoT: a practical, contrastive way to steer CoT (Figure 6.12)
  - What’s new: automatically generate “reasoning” both for correct labels and for corrections of wrong labels; insert a “don’t do this” CoT exemplar to steer the model away from typical mistakes (Figures 6.12–6.16).
  - Why it matters: in the case study it achieved the best dev F1 = 0.53 (precision 0.86, recall 0.38) with 10 exemplars (Section 6.2.3.3; Figures 6.5–6.6).

- Security framing that cleanly distinguishes prompt injection vs. jailbreaking and maps defenses to practice (Section 5.1; Figure 5.1)
  - What’s new: concise definitions and concrete risks (e.g., package hallucination) and three defense layers: prompt‑based heuristics, detectors, and guardrail frameworks (Section 5.1.3).
  - Why it matters: operationalizes safety for prompt‑driven systems.

## 5. Experimental Analysis
- Evaluation methodology
  - MMLU comparison (Section 6.1):
    - Model: `gpt‑3.5‑turbo`.
    - Data: 2,800 questions (20% per category), two formats (Figures 6.3 & 6.4), multiple instruction phrasings; Self‑Consistency at temperature 0.5 (Section 6.1.3), others at 0.
    - Metrics: accuracy; responses parsed by pattern rules (Section 6.1.4).
  - Case study (Sections 6.2.1–6.2.3):
    - Data: 221 r/SuicideWatch posts; 121 dev/100 test; α = 0.72 inter‑coder (Section 6.2.2).
    - Metrics: F1, precision, recall.
    - Baselines and ablations: zero‑shot+context; few‑shot; CoT variants; AutoDiCoT; extraction strategies; ensemble; context duplication and anonymization; DSPy automated prompt optimization (Section 6.2.3; Figures 6.5–6.6, 6.19).

- Main quantitative results
  - MMLU (Figure 6.1):
    > Zero‑Shot: 0.627; Zero‑Shot‑CoT: 0.547; Zero‑Shot‑CoT + Self‑Consistency: 0.574; Few‑Shot: 0.652; Few‑Shot‑CoT: 0.692; Few‑Shot‑CoT + Self‑Consistency: 0.691.
    - Self‑Consistency aided Zero‑Shot‑CoT (+0.027) but not Few‑Shot‑CoT (−0.001), and overall variance across phrasing/format is non‑trivial (purple error bars, Figure 6.1). The better performance of Few‑Shot‑CoT vs. Zero‑Shot‑CoT underscores the value of demonstrations for this model and benchmark (Section 6.1.5).

  - Case study (dev set unless noted; Section 6.2.3; Figures 6.5–6.6):
    > Zero‑Shot + Context: F1 0.40; recall 1.00; precision 0.25.  
    > 10‑Shot + Context: F1 0.45; recall 0.91; precision 0.30.  
    > 1‑Shot AutoDiCoT + Full Context: F1 0.36; recall 0.33; precision 0.39.  
    > Full Context Only (duplicated email inadvertently): F1 0.44; recall 0.92; precision 0.29.  
    > 10‑Shot AutoDiCoT (best dev): F1 0.53; recall 0.86; precision 0.38.  
    > 20‑Shot AutoDiCoT: F1 0.49; recall 0.94; precision 0.33.  
    > De‑dupe email: F1 0.45; recall 0.74; precision 0.33.  
    > Remove email entirely: F1 0.39 (earlier ablation also shows a sharp drop to F1 0.18 with a different configuration).  
    > Ensemble + extraction: F1 0.36; recall 0.64; precision 0.26.
    - On the test set (Figure 6.19):
      > DSPy (default) yields F1 0.548; precision 0.385; recall 0.952. A slightly modified DSPy prompt performs similarly.

- Do the experiments support the claims?
  - Yes, for three core claims:
    - Technique outcomes are conditional: Zero‑Shot‑CoT can underperform; Few‑Shot‑CoT generally helps on MMLU (Section 6.1.5).
    - Prompt formatting and wording matter: two MMLU formats and varied base instructions produced spread in accuracy (Figures 6.3–6.4, 6.1). In the case study, duplicating an email in context unexpectedly improved results, while anonymizing it hurt (Sections 6.2.3.3; “Full Context Only,” “Anonymize Email”).
    - Automated search can rival or surpass manual engineering: DSPy outperforms the manually tuned best prompt on the test set (F1 0.548 vs. best manual dev 0.53; Figure 6.19).

- Robustness and failure analysis
  - The paper reports sensitivity to model guardrails (initial advice‑giving rather than labeling), fixed by switching models (Section 6.2.3.2).
  - Extraction errors: moving from exact/first‑token matching to last‑token and LLM‑based extractors shifted precision/recall balances (Figure 6.18; Section 6.2.3.3).
  - Self‑Consistency benefits are not uniform; they depend on the base prompting regime and temperature (Figure 6.1; Section 6.1.3).

- Mixed or conditional results
  - CoT: helps most when paired with good exemplars (Few‑Shot‑CoT), less so or even harmful in zero‑shot for GPT‑3.5‑turbo on MMLU (Figure 6.1).
  - Exemplars: More is not always better—20‑Shot AutoDiCoT underperforms 10‑Shot on dev (Section 6.2.3.3).

## 6. Limitations and Trade-offs
- Scope constraints
  - Focuses on hard prefix prompts; excludes soft/gradient‑based prompt tuning and full fine‑tuning (Section 1; Appendix A.2.4). Some modern systems combine these approaches, so findings may not transfer.
  - English‑centric emphasis in review and experiments; multilingual section is survey‑style, not experimentally validated in this work (Section 3.1).

- Methodological assumptions
  - PRISMA labeling relies partly on GPT‑4 classification (precision 89%, recall 75%), which can introduce selection bias in included papers (Section 2.1.1).
  - MMLU study uses a single frontier model (`gpt‑3.5‑turbo`) and specific formatting choices; results may vary across models or newer versions (“prompt drift,” Section 5.2.1).

- Data and evaluation constraints
  - Case study dataset is small (221 posts) and domain‑specific; F1 ~0.53 suggests usefulness but not production readiness; precision/recall trade‑offs matter critically in this safety domain (Sections 6.2.3–6.2.4).
  - Some performance increases hinge on incidental cues (duplicating an email), raising concerns about reliance on unintended artifacts (Section 6.2.3.3).

- Practical trade‑offs
  - Ensembling and self‑consistency improve robustness at the cost of latency and API spend (Section 2.2.4).
  - Structured outputs (XML/JSON) can help evaluators but may reduce task performance in some settings; findings are mixed in the literature reviewed (Section 1.2.1 “Output Formatting”; Section 4.2.2).

## 7. Implications and Future Directions
- How this work shifts the field
  - Provides a common map and language for practitioners and researchers to reason about prompting end‑to‑end—from template design to answer extraction, from single‑turn prompts to agentic workflows (Figures 1.3, 2.2, 4.1).
  - Surfaces a critical practical insight: prompting is akin to hyperparameter search where format, phrasing, and extraction must be co‑designed and empirically validated (Section 6; Section 5.2.1).

- Follow‑up research enabled
  - Standardized benchmarking: replicate the MMLU study across models (e.g., GPT‑4‑class, open LLMs) and tasks with common prompt libraries and extraction protocols.
  - Systematic ablation of “thought inducers” and exemplar properties (quantity, order, diversity) across tasks (Sections 2.2.1.1, 2.2.2).
  - Automated prompt optimization: extend DSPy‑style pipelines with answer engineering and safety constraints; compare against RL‑based methods (APE, GRIPS, RLPrompt; Section 2.4).
  - Security hardening: combine detectors, guardrails, and meta‑prompting to counter injection/jailbreaks; publish stress‑test suites (Section 5.1.3).

- Practical applications
  - Enterprise prompt stacks can adopt the survey’s process model (Figure 1.4), including explicit answer extraction and LLM‑judge evaluators (Section 4.2).
  - Agentic systems can use the cataloged planning and tool‑use patterns (MRKL, ReAct, PAL, ToRA, Reflexion) to build reliable multi‑step workflows (Section 4.1).
  - Multilingual and multimodal teams can leverage the surveyed analogs of text techniques (e.g., XLT/CLSP for CoT; Chain‑of‑Images/Tab‑CoT for structure) to close quality gaps (Sections 3.1–3.2).

Definitions used on first mention (selective):
- `In‑Context Learning (ICL)`: guiding a model to perform a task by placing instructions and/or examples directly in the prompt, without updating model weights (Section 2.2.1).
- `Chain‑of‑Thought (CoT)`: prompting the model to verbalize intermediate reasoning steps before giving an answer (Section 2.2.2).
- `Self‑Consistency`: sampling multiple reasoning paths and taking a majority answer to reduce variance (Section 2.2.4).
- `Retrieval‑Augmented Generation (RAG)`: fetching external documents and inserting them into the prompt to aid knowledge‑intensive tasks (Section 4.1.4).
- `Answer engineering`: constraining/extracting model outputs via defined shapes, spaces, and extractors to obtain reliable final labels/answers (Section 2.5).

Overall, this paper is most useful as a field guide: it codifies the moving parts of prompt‑centric systems, shows where techniques help (and where they don’t), and outlines how to evaluate and secure them. Its empirical sections underscore a practical lesson: validate techniques in your exact setting—with your formats, extractors, and costs—before standardizing them in production.
