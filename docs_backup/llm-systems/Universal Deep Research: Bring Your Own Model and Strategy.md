# Universal Deep Research: Bring Your Own Model and Strategy

**ArXiv:** [2509.00244](https://arxiv.org/abs/2509.00244)
**Authors:** Peter Belcak, Pavlo Molchanov
**Institutions:** NVIDIA Research

## 🎯 Pitch

Universal Deep Research (UDR) revolutionizes research tools by transforming user-defined strategies into executable code, offering a flexible, model-agnostic system that enhances transparency, efficiency, and control. This advancement enables both individuals and enterprises to tailor workflows for various domains, breaking free from the constraints of hard-coded research systems and ensuring auditability and cost-effectiveness—a significant leap in adapting AI to diverse, specialized needs.

---

## 1. Executive Summary (2-3 sentences)
Universal Deep Research (UDR) is a general-purpose, “bring-your-own-model” agentic system that converts a user’s natural‑language research strategy into executable code which orchestrates tools, controls a language model, and streams structured progress notifications (Section 2; Figure 2). It addresses the rigidity of existing deep research tools by letting users fully define, edit, and refine research workflows without training or finetuning, enabling transparent, auditable, and cost‑efficient research across consumer and enterprise settings (Introduction; Problems P1–P3; Conclusions).

## 2. Context and Motivation
- Problem addressed
  - Most deep research tools (DRTs) hard‑code a single research strategy and a fixed model/tool stack, leaving users little control beyond the prompt (Introduction; “Problem statement,” P1–P3).
  - Three concrete gaps are identified:
    - P1: Users cannot enforce source hierarchies, automate cross‑validation against trusted sources, or manage cost/latency trade‑offs (Problem statement).
    - P2: Specialized strategies needed in high‑value domains (finance, legal, healthcare) are not expressible in current DRTs (Problem statement).
    - P3: Models are not interchangeable; one cannot pair the best model with a preferred deep‑research agent (Problem statement).

- Why this matters
  - Closing P1 raises report quality for individuals and narrows the gap between consumer and enterprise DRTs.
  - Solving P2 enables automation of specialized, labor‑intensive research workflows in high‑value industries.
  - Addressing P3 allows independent competition and pairing between the most competitive models and research agents (Problem importance and impact).

- Prior approaches and shortcomings
  - Consumer‑oriented tools such as Gemini, Perplexity, and OpenAI Deep Research iteratively browse and expand searches via LM‑driven chains of queries (Introduction; “General landscape”).
  - Grok 3 DeepSearch adds a two‑tier crawling infrastructure with chain‑of‑thought reasoning for credibility checks (Introduction).
  - Enterprise systems often use bespoke, rigid pipelines:
    - NVIDIA AI‑Q Research Assistant: five‑step prompt-to-report plan inside curated corpora (Introduction; “Enterprise landscape”).
    - SambaNova: document‑oriented multi‑agent pipeline with section‑level planning (Introduction).
    - ERP AI Deep Research: graph‑based data access via knowledge graphs/GNNs (Introduction).
  - Across these, users cannot swap the underlying model freely, nor “program” the agentic behavior in natural language to enforce policies like source prioritization, validation steps, or budget limits (P1–P3).

- How this paper positions itself
  - UDR is a generalist wrapper around any language model that lets the user specify the end‑to‑end research strategy in natural language, which UDR compiles into executable code with a fixed tool API and deterministic control flow (Section 2; Figure 2).
  - It claims a general resolution to P1–P3 by making strategy and model interchangeable and user‑defined without additional training (Novelty; Contribution).

## 3. Technical Approach
UDR’s core idea is to transform a human‑written strategy into a single callable program that runs the entire research workflow deterministically, calls tools synchronously, uses the language model only where explicitly instructed, and streams structured notifications.

Step‑by‑step:

1) Inputs (Section 2.1)
   - `Research Strategy` (free‑form natural language): a list or bullet sequence of steps that fully specifies behavior (Appendix A for examples: Minimal, Expansive, Intensive).
   - `Research Prompt`: the user’s topic/task with any content/formatting requirements (Appendix B for examples).
   - Design intent: The strategy—not the model—controls the flow. There are “no implicit restrictions”; any condition must be checked in the strategy logic.

2) Strategy processing → code generation (Section 2.2, “Phase 1 – Strategy processing”)
   - UDR prompts a language model to “compile” the strategy into source code that:
     - is “a single callable function that accepts the research prompt as the input and continuously returns output notifications” via a generator (yielding dictionaries) (Section 2.2).
     - uses only permitted functions and control structures; tools are documented via a docstring so the generated code knows how to call them.
   - Reliability design choices:
     - The system enforces a one‑shot end‑to‑end generation of the whole function and requires that each code segment be preceded by a comment quoting the corresponding strategy step. This curbs the tendency to “take shortcuts, skip steps, or impose constraints not stipulated by the user.” The paper reports that this approach “all but eradicate[s]” such behavior across models (Section 2.2, Reliability).
     - Quote for grounding:
       > “We prompted the model to generate code that corresponded to the strategy step by step, explicitly prepending every segment of the generated code by comments laying out the strategy step it corresponds to.” (Section 2.2)

   - Why not chain smaller code snippets? Earlier prototypes that decomposed the strategy into isolated fragments or embedded it directly into a reasoning‑oriented LM were “fragile and error‑prone,” leading to step skipping and synchronization failures. End‑to‑end code generation improved coherence (Section 2.2, Reliability).

3) Strategy execution (Section 2.2, “Phase 2 – Strategy execution”)
   - The generated function executes in an isolated environment (sandbox) and:
     - Maintains state in named variables, not in the LM context window. This lets the system reuse information across steps without inflating prompts.
       - Grounding quote:
         > “UDR stores all intermediate information and text fragments as named variables in the code execution state… In our experiments, a context length of 8k tokens was sufficient to carry out full research workflows, regardless of their complexity.” (Section 2.2, State modifications)
     - Calls tools synchronously through explicit function calls (e.g., a `search(...)` API), ensuring deterministic behavior (Section 2.2, Tool use).
     - Uses the language model as a local utility (for summarization, ranking, extraction) when the strategy demands it, rather than letting the LM orchestrate the whole process (Section 2.2, LM reasoning):
       > “Language model reasoning is treated as a callable utility rather than a controlling entity.”
     - Emits structured progress notifications as `yield`ed dictionaries (with fields like `type`, `timestamp`, `description`). The final report is returned as a last notification with a distinctive type such as `"final_report"` (Sections 2.2 Notifications; 2.3 Outputs).

4) Outputs (Section 2.3)
   - Notifications: a stream of event dictionaries suitable for real‑time UI updates.
   - Final Research Report: structured text/Markdown built from accumulated state (not from an ever‑growing LM context), enabling traceability and reproducibility.

5) Security and isolation (Section 2.2, Security)
   - Because UDR executes generated code, it must run in a sandbox that blocks access to the host system and prevents side effects; the paper suggests leveraging engines such as Piston and emphasizes that isolation is a “strict requirement” for non‑trusted deployments.

6) Efficiency rationale (Section 2.2, Efficiency)
   - Orchestration is CPU‑only code; expensive LM inference happens only where, and on exactly the text, the strategy requests. This “dual‑level efficiency” (code for control; LM for local text tasks) cuts GPU cost and latency.

7) User interface (Section 3; Figures 3–4)
   - Includes: search bar, strategy selection, “edit strategy” panel, streaming progress notifications, stop button, “generate report” for partial results, and a Markdown viewer for the final report.

8) Example strategies (Appendix A)
   - Minimal: generate 3 search phrases → search → aggregate context → one LM call to write report (Appendix A.1; clear step‑indexed logic).
   - Expansive: first produce 2 topics → per topic, generate up to 2 phrases → search and append to a shared `context` → final synthesis (Appendix A.2).
   - Intensive: iterative refinement over two rounds; uses both a `subcontext` per round and a `supercontext` for all sources; expands phrases after each round based on newly gathered text (Appendix A.3).

9) Example outputs (Appendix B)
   - Demonstrations using `Llama 3.3 70B` with the Minimal strategy, covering varied prompts (culture query, events on a date, market movements, historical figure). They show structured Markdown with sectioning and simple reference lists.

Concepts defined briefly:
- `DRT` (Deep Research Tool): an agent that executes searches and compiles a long‑form, referenced report with progress updates (Introduction; Figure 1).
- `Prompt injection`: malicious content inducing the agent to run unintended actions; UDR mitigates via sandboxing (Section 2.2, Security).
- `Generator`/`yield`: a program function that emits incremental results/event messages over time—used for progress notifications (Sections 2.2–2.3).

## 4. Key Insights and Innovations
1) Strategy‑to‑code compilation with step‑aligned comments
   - What’s new: The system converts a free‑form strategy into a single, fully executable function whose code segments are explicitly aligned with each written step (Section 2.2, Phase 1).
   - Why it matters: This “disciplined structure” greatly reduces failure modes seen in LM‑orchestrated agents (skipping steps, imposing unstated constraints) and in fragmented code generation. The paper reports that such failures were “rarely” observed after adopting this method (Section 2.2, Reliability).
   - Type: Fundamental innovation in agent specification and enforcement.

2) LM as a local utility, not the global controller
   - What’s new: The LM performs bounded tasks (summarize/rank/extract) when explicitly invoked by the code, instead of free‑running agentic control (Section 2.2, LM reasoning).
   - Why it matters: Improves determinism, traceability, and cost control; reduces susceptibility to prompt drift. This is a notable reframing compared to typical LM‑first agent designs (Figure 1 vs. Figure 2).
   - Type: Conceptual/design innovation with practical cost and reliability benefits.

3) Externalized state enables small context windows and reproducibility
   - What’s new: All intermediate text lives in code variables rather than the LM’s context; 8k tokens sufficed in experiments “regardless of complexity” (Section 2.2, State modifications).
   - Why it matters: Makes the approach model‑agnostic and resource‑efficient; facilitates long workflows without context bloat and supports auditing by inspecting state.
   - Type: Practical systems innovation.

4) Bring‑Your‑Own‑Model and Bring‑Your‑Own‑Strategy
   - What’s new: Users can pair any compatible model with any strategy, edit strategies live, and share a library of strategies (Introduction; Sections 2–3; Conclusions R1).
   - Why it matters: Addresses P1–P3 head‑on—users can impose source policies and budgets (P1), craft domain‑specific strategies (P2), and swap models at will (P3).
   - Type: Capability innovation that unlocks new enterprise and consumer workflows.

## 5. Experimental Analysis
- Evaluation setup
  - Demonstrations only; no large‑scale benchmarks. The paper presents example runs using `Llama 3.3 70B` with the Minimal strategy (Appendix B.1–B.4).
  - The examples include:
    - A cultural trivia query (“unladen swallow”) producing a three‑section report with references (Appendix B.1).
    - “Significant events” on a specific date, outputting structured sections and a reference list (Appendix B.2).
    - “US stock movements” on a specific date with opening/closing summaries and broader context (Appendix B.3).
    - A historical figure (“Ulysses S. Grant”), five‑section report with citations (Appendix B.4).
  - The UI supports streaming notifications and early stopping with partial report generation (Section 3; Figure 4).

- Methodological claims vs. evidence
  - Reliability: Section 2.2 claims the end‑to‑end code‑generation approach “rarely” shows earlier failure modes. This is qualitative; there is no error‑rate metric or ablation table.
  - Efficiency: Section 2.2 argues for “dual‑level efficiency” and notes that 8k tokens sufficed in their experiments. Again, no runtime/throughput comparison or cost accounting is provided.
  - Security: The system design mandates sandboxed execution (Section 2.2). No penetration tests or red‑team experiments are reported.

- Quantitative results
  - None in tables/figures; all reported evidence is descriptive. The example outputs do illustrate that the generated code follows the specified strategy steps (e.g., notifications like “search_started,” “report_building” in Appendix A logic), and the final reports are well‑structured Markdown.

- Robustness checks and ablations
  - The only “ablation‑like” narrative is in Reliability (Section 2.2): earlier prototypes tried (a) embedding the strategy into a reasoning prompt and (b) per‑step code generation; both were “fragile.” The paper does not include systematic measurements or user studies.

- Overall assessment
  - The demonstrations substantiate feasibility and illustrate UDR’s determinism/transparency, but they do not quantify advantages over existing DRTs. Claims about reliability and efficiency are plausible given the architecture, yet remain to be validated with controlled benchmarks (e.g., step‑adherence rates, cost/latency vs. LM‑orchestrated agents, success under prompt‑injection attempts).

## 6. Limitations and Trade-offs
- Assumptions and dependencies
  - Faithfulness depends on the code generation capability of the chosen language model (Section 4: “Reliance on language model code generation”). Ambiguity in strategies can still induce “semantic drift or hallucinated logic” despite comment‑aligned step enforcement.
  - The system trusts that the user‑authored strategy is coherent and safe; beyond syntax/execution checks, it does not validate overall logic or quality (Section 4: “Trust in user-defined strategies”).

- Interactivity constraints
  - Mid‑execution user steering is limited: beyond stopping the run or generating a partial report, decisions must be pre-encoded in the strategy (Section 4: “Limited real-time interactivity”).

- Security and deployment
  - Safe operation requires sandboxing. Any lapse in isolation exposes risks from executing generated code or tool calls (Section 2.2, Security: isolation is a “strict requirement”).

- Practicality and user burden
  - Devising robust strategies is “tedious” for end users—even those who want fine control (Conclusions). The paper recommends shipping with a strategy library (R1).

- Scope
  - No coverage of asynchronous tool execution (design allows a “future upgrade”) or distributed crawling; the current emphasis is on determinism and simplicity (Section 2.2, Tool use).

- Evidence limits
  - No quantitative evaluations; no cross‑model comparatives to validate the BYOM advantage; no domain‑specific case studies (e.g., legal/finance) demonstrating P2 at scale.

## 7. Implications and Future Directions
- How this changes the landscape
  - UDR reframes agentic research from “LM decides everything” to “code decides; LM assists.” This separation of concerns can standardize agent design around auditable, deterministic control logic while keeping models plug‑and‑play (Figure 2; Sections 2–3).
  - For enterprises, it enables codifying compliance policies, source priorities, and validation rules directly in the strategy, making research workflows auditable and reproducible—key for regulated domains (P1–P2).

- Follow‑up research enabled
  - Benchmarks for “strategy faithfulness”: measure step‑adherence, tool‑use correctness, and report traceability across models and strategies.
  - Cost/latency studies comparing UDR to LM‑orchestrated agents under identical tasks and datasets.
  - Automatic strategy synthesis/tuning: converting classes of prompts into deterministic agents (Recommendation R3).
  - Richer user control of model reasoning (“thinking”) beyond local utilities—investigating safe, bounded CoT under explicit code governance (Recommendation R2).
  - Asynchronous tool orchestration and distributed crawling under the same strategy‑to‑code paradigm.

- Practical applications
  - Enterprise research assistants with enforceable source hierarchies, budget caps, and cross‑validation routines encoded in strategies (P1).
  - Domain‑specialized workflows (e.g., earnings‑call analysis, regulatory document synthesis, legal precedent mapping) by composing tool APIs and step logic without finetuning (P2).
  - Consumer research with “strategy presets” (Minimal/Expansive/Intensive) and editable templates; marketplaces for vetted strategies (R1).
  - Auditable government/public‑administration research where deterministic steps and traceable sources are required.

In sum, UDR contributes a method and prototype for turning natural‑language research procedures into reliable, tool‑calling programs that treat language models as bounded subroutines. The concept is compelling and addresses real gaps (P1–P3), but it requires systematic evaluation, richer interactivity, and packaged strategy libraries to reach production‑grade impact.
