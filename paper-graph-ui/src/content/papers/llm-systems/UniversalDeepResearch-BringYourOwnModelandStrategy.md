# Universal Deep Research: Bring Your Own Model and Strategy

**ArXiv:** [2509.00244](https://arxiv.org/abs/2509.00244)

## 🎯 Pitch

Universal Deep Research (UDR) introduces a model-agnostic system that lets users pair any language model with fully customizable, natural-language 'research strategies'—which UDR automatically converts into executable research workflows. This empowers users with deep, auditable control over tool usage, process logic, and report structure, breaking the traditional lock-in of research agents to fixed strategies or models and opening new possibilities for tailored research automation in enterprise and consumer settings alike.

---

## 1. Executive Summary (2-3 sentences)
Universal Deep Research (UDR) is a general-purpose deep research system that lets a user bring any language model and specify a natural-language “research strategy,” which UDR compiles into executable code that conducts the research end-to-end (Section 2; Figure 2). Its significance is twofold: it decouples research orchestration from any fixed model or hard-coded agent, and it gives users fine-grained, auditable control over the entire research process, including tools, control flow, progress notifications, and report structure (Sections 1–2.3; Figures 1–2).

## 2. Context and Motivation
- Problem addressed
  - Existing deep research tools (DRTs) are largely rigid: they hard-code one research strategy and one model stack, giving users little control beyond the prompt (Introduction, “Problem statement,” P1–P3).
  - Three specific gaps:
    - P1: Users cannot easily enforce source priorities, automate cross-validation against trusted sources, or control search cost (Introduction, P1).
    - P2: Domain-specialized strategies needed in high-value enterprise settings (finance, legal, healthcare) cannot be authored inside current DRTs (Introduction, P2).
    - P3: Models are not interchangeable—users cannot pair their preferred/latest model with a separate DRT agent of choice (Introduction, P3).

- Why it matters
  - Real-world: More reliable sourcing and cost control matters for enterprise governance and regulated workflows. Special strategies are often essential to unlock automation in high-stakes domains (Introduction, “Problem importance and impact”).
  - Ecosystem: Decoupling “model” and “agent” enables independent competition and innovation among models and DRT frameworks (Introduction, P3 impact).

- Prior approaches and their limits
  - Consumer-facing DRTs browse iteratively using LLM agency (Gemini, Perplexity, OpenAI Deep Research) or custom crawlers plus chain-of-thought (Grok 3 DeepSearch) (Introduction, “General landscape”).
  - Enterprise tools rely on bespoke pipelines (e.g., AI-Q’s five-step plan, SambaNova’s document agents, ERP AI’s graph-based architecture), but are still predefined patterns rather than user-programmable strategies (Introduction, “Enterprise landscape”).
  - Across both, users cannot declaratively author and swap in completely custom strategies, nor freely swap the backbone model.

- UDR’s position
  - UDR is a “generalist wrapper” that compiles a user’s strategy into executable code, runs it in a sandbox with well-scoped tool use, and treats the language model as a local utility rather than as the workflow orchestrator (Sections 2–2.3). It directly targets P1–P3 (Introduction, “Contribution” and “Novelty”).

Definitions (selective):
- Deep research tool (DRT): a tool that takes a research prompt, performs broad search over relevant sources, and produces a structured, cited report with continuous progress updates (Introduction; Figure 1).
- LLM agency vs. code agency: orchestrating the workflow via the model’s reasoning and tool-calling (LLM agency) versus orchestrating via executable code that calls the model as a subroutine (code agency) (Introduction; Figure 1).

## 3. Technical Approach
High-level pipeline (Figure 2; Section 2):
1) Inputs
   - A natural-language `Research Strategy` (a step-by-step plan describing how to search, validate, and build the report).
   - A `Research Prompt` (the user’s research question and desired output format).
2) Phase 1 — Strategy Processing (Section 2.2 “Phase 1 – Strategy processing”)
   - Strategy-to-code compilation: UDR passes the strategy to a language model with strict constraints on the allowed tool functions and control structures.
   - Output: a single callable function that takes the research prompt and yields structured notifications during execution. In implementation, UDR requires a Python-style generator that uses `yield` to emit dictionaries describing progress events.
   - Reliability enforcement: the generated code is forced to align “step-by-step” with the strategy by prepending each code segment with comments that restate the strategy step; this reduces the model’s tendency to “take shortcuts” or skip steps.
     - Quote: “To all but eradicate this behavior, we prompted the model to generate code that corresponded to the strategy step by step, explicitly prepending every segment of the generated code by comments laying out the strategy step it corresponds to.” (Section 2.2, Phase 1)

   How this works in practice:
   - The tool interface (e.g., a `search` function) is provided as a docstring in the prompt that generates the code, constraining how the generated program can call tools.
   - The compiled function’s contract is deterministic: accept `PROMPT`; emit periodic notifications; return a final “report” notification with the full markdown.

3) Phase 2 — Strategy Execution (Section 2.2 “Phase 2 – Strategy execution”)
   - Sandboxed execution: The compiled generator runs in an isolated environment to mitigate risks from untrusted code and prompt-injection (Section “Security”).
     - Quote: “each generated strategy can be … executed within a sandboxed environment that prevents access to the host system… Ready-to-use solutions such as Piston provide a foundation” (Section 2.2, Security).
   - State outside the LLM context: intermediate artifacts (snippets, summaries, tables, lists of sources) are stored as named variables in the execution state, not as an ever-growing LLM prompt.
     - Quote: “UDR stores all intermediate information and text fragments as named variables in the code execution state… a context length of 8k tokens was sufficient” (Section 2.2, “State modifications”).
   - Synchronous, transparent tool calls: tools (e.g., `search`) are called as blocking functions with explicit inputs/outputs, making the dataflow obvious and auditable (Section 2.2, “Tool use”).
   - LLM as a utility, not the “agent brain”: the model is only invoked for local tasks (e.g., generate search phrases, summarize, rank, extract) at steps specified by the strategy, instead of “thinking” through the entire workflow (Section 2.2, “LM reasoning”).
     - Quote: “Language model reasoning is treated as a callable utility rather than a controlling entity.” (Section 2.2, LM reasoning)
   - Structured progress updates: the code emits notifications using `yield {type, timestamp, description, …}` that the UI renders in real time (Section 2.3, “Notifications”).

4) Report Construction and Output (Section 2.3)
   - The final yielded notification has a distinctive `type` (e.g., `final_report`) and contains the assembled markdown report built from the stored state.
   - Because state is explicit and preserved, the output is auditable and reproducible.

Why these design choices?
- Reliability: compiling the entire strategy into one coherent program with step-labeled code reduces drift and step-skipping compared to prompting a generalist agent to act step-wise (Section 2.2, Reliability).
- Cost and speed: CPU executes orchestration; the model is only called on short, focused text (Section 2.2, Efficiency).
  - Quote: “UDR achieves high computational efficiency by separating control logic from language model reasoning… delegating orchestration to CPU-executable logic and limiting LLM use to focused… invocations” (Section 2.2, Efficiency).
- Safety: executable code is sandboxed; tools are sync and predeclared; state is deterministic (Section 2.2, Security).

Concrete examples of strategies (Appendix A; Figure 2 alludes to the mechanism):
- Minimal Strategy (Appendix A.1): generate a few search phrases -> search -> aggregate raw content into `CONTEXT` -> ask the model to write the report from `CONTEXT`.
- Expansive Strategy (Appendix A.2): first generate topics, then per-topic phrases and searches, aggregating into a single `context` before reporting.
- Intensive Strategy (Appendix A.3): iterative refinement using `subcontext` and `supercontext`, where new search phrases are generated based on what the previous round found.

Definitions (selective):
- `generator` (programming): a function that can “yield” intermediate outputs multiple times during execution.
- `context window` (LLM): the maximum token length the model can attend to in one call.
- `sandboxed environment`: an isolated execution container that prevents file/system/network access beyond whitelisted interfaces.

## 4. Key Insights and Innovations
- A. User-authored strategy → executable agent (fundamental innovation)
  - Novelty: a natural-language strategy is compiled into a deterministic, auditable program with fixed tool APIs and explicit control flow (Section 2.2, Phase 1; Figure 2).
  - Why it matters: unlocks BYO strategy and BYO model; enables enterprise teams to encode governance, sourcing policies, and cost limits directly into the agent without fine-tuning.

- B. Treat the LLM as a “local function,” not the orchestrator (conceptual shift)
  - Difference from prior DRTs: typical agents let the model “decide” the next step; UDR’s code decides and calls the model only for scoped tasks (Section 2.2, “LM reasoning”).
  - Impact: improves reliability and cost predictability; reduces “agent drift.” This is a strategic separation of concerns rather than an incremental tweak.

- C. State outside the model’s prompt (practical innovation)
  - Mechanism: keep all intermediate artifacts in program state variables rather than in an ever-growing prompt (Section 2.2, “State modifications”).
  - Significance: keeps each LLM call small; the paper reports that “a context length of 8k tokens was sufficient” even for full workflows (Section 2.2).

- D. Structured, user-authored progress telemetry (usability/ops innovation)
  - Mechanism: notifications are emitted as structured dictionaries via `yield` and can be precisely defined by the strategy (Section 2.3, “Notifications”).
  - Significance: transparent, low-latency progress reporting without exposing raw internals unless the strategy chooses to.

- E. Sandbox-first execution model (safety baseline)
  - Mechanism: run generated code in an isolated environment (e.g., Piston) to prevent host access and side-effects (Section 2.2, “Security”).
  - Importance: recognizes code-generation risks and bakes in a deployment pattern for safer operation.

## 5. Experimental Analysis
Evaluation design in the paper focuses on system behavior and qualitative reliability rather than benchmark numbers.

- What is evaluated and how
  - Reliability comparison (qualitative): The paper compares three orchestration methods during development—(i) prompting an LLM with the strategy inline, (ii) generating code per step and stitching, and (iii) UDR’s single-pass, whole-strategy-to-code approach. It argues (without numeric metrics) that (iii) “yielded significantly more reliable outcomes” with fewer failures like step skipping or out-of-sequence tool calls (Section 2.2, “Reliability”).
    - Quote: “The resulting code is fully interpretable and auditable… and it rarely exhibits the failure modes encountered in our earlier prototypes, such as skipping strategy steps…” (Section 2.2, Reliability).
  - Efficiency characterization: The system-level rationale is described; no latency or cost numbers are reported, but the paper explains why fewer, smaller model calls should reduce cost (Section 2.2, Efficiency).
  - Context sizing: Observational claim that “a context length of 8k tokens was sufficient to carry out full research workflows” due to externalized state (Section 2.2, “State modifications”).
  - Security posture: Architectural description plus recommendation to sand-box with an off-the-shelf engine (Section 2.2, Security), not a penetration test or red-team study.

- Demonstrations
  - UI walkthrough and features (Section 3; Figures 3–4): shows strategy selection, live notifications, and a report viewer.
  - Example strategies and outputs (Appendix A–B): three strategies (Minimal, Expansive, Intensive) and several example prompts with raw markdown outputs produced using `Llama 3.3 70B` (Appendix B).

- Main observations and whether evidence is convincing
  - Reliability: The shift to whole-strategy compilation with step-labeled code is a compelling design hypothesis and the qualitative accounts are consistent (Section 2.2). However, without quantitative measures (e.g., task success rate, step adherence rate, or human evals), this remains suggestive rather than conclusive.
  - Efficiency: The architectural argument is sound, but the paper lacks wall-clock and cost-per-task measurements to substantiate gains across models and workloads (Section 2.2).
  - Capability: Appendix B demonstrates end-to-end functionality (e.g., generating a “swallow airspeed” cultural/technical report, daily stock wrap, and a historical biography). These are instructive examples but do not constitute rigorous benchmarks (Appendix B).

- Ablations, failure cases, robustness
  - No formal ablations or robustness tests are reported. The paper does discuss failure modes it sought to prevent (e.g., skipping steps, spurious constraints) and the prompts/constraints crafted to mitigate them (Section 2.2, Reliability).
  - Limitations section (Section 4) candidly notes potential semantic drift or hallucinated logic when strategies are ambiguous, highlighting an area where quantitative robustness testing would be valuable.

In short, the experimental section primarily provides design rationale, qualitative comparison with earlier internal variants, and demos. It convincingly explains “how and why” the system should work, but does not yet provide standardized, quantitative validation.

## 6. Limitations and Trade-offs
- Dependence on code generation quality (Section 4)
  - Assumption: the underlying model will faithfully implement the strategy and honor the tool/API constraints.
  - Risk: ambiguous or underspecified strategies can lead to “semantic drift or hallucinated logic” despite the step-comment alignment technique.

- Trust in user-authored strategies (Section 4)
  - Assumption: users write logically coherent, safe strategies.
  - Trade-off: UDR does not validate the semantic adequacy of a strategy—bad strategies can produce ineffective or incomplete reports or none at all. In regulated settings, additional linting/verification is needed.

- Limited real-time interactivity (Section 4)
  - Current design: users can stop the workflow and (optionally) generate a partial report, but cannot steer mid-execution or branch dynamically based on live feedback (Section 3, “Generate report button”; Section 4).
  - Implication: exploratory research may require more frequent restarts or preplanning within the strategy.

- Security posture requires careful deployment (Section 2.2, Security)
  - While sandboxing is recommended and feasible (e.g., Piston), safe deployment still hinges on the quality of isolation, network/file restrictions, and defenses against prompt injection via retrieved content.

- Generalizability across tools and complex pipelines
  - The paper discusses synchronous tool calls and notes a “future upgrade to asynchronous tool use” (Section 2.2, Tool use). Highly parallel or streaming data scenarios may need extensions.
  - Enterprise integration (e.g., knowledge graph systems, proprietary APIs) will require robust tool adapters with clear schemas.

- Evaluation depth
  - The paper offers demonstrations but not standardized benchmarks or comparative metrics against existing DRTs. This leaves open questions about scalability, throughput, and adherence under stress (Sections 2–4; Appendix B).

## 7. Implications and Future Directions
- How this changes the landscape
  - Decouples model choice from agent design: users can pair the “most competitive models with the most competitive DRTs,” enabling independent innovation and competition across layers (Introduction, Problem importance; “Novelty”).
  - Shifts agency to the user: strategies become first-class, auditable artifacts—akin to “programming agents in natural language” (Conclusions, Section 5). This opens a path for governance-by-design in enterprises.

- Practical applications
  - Enterprise research with policy-constrained sourcing (finance/legal/compliance) where users must encode source hierarchies, cross-validation steps, and cost caps (Introduction, P1–P2).
  - Internal document research with tailored retrieval/validation loops (cf. Appendix A.2/A.3), where state tracking and iterative expansion are crucial.
  - Education and analysis workflows where transparency and stepwise audit are required (Sections 2.3, 3).

- Recommended next steps (aligning with Section 5, R1–R3)
  - R1: Ship with a library of vetted strategies that users can customize rather than author from scratch—lowers adoption friction (Section 5).
  - R2: Explore interfaces for user control over model “reasoning” style within steps (temperature constraints, few-shot policies, citation enforcement) to further reduce drift (Section 5).
  - R3: Research automatic translation of everyday prompts into deterministic, multi-step strategies (“turn prompts into agents”), including validation/linting of the resulting programs (Section 5).

- Additional research opportunities (beyond the paper’s recommendations)
  - Quantitative evaluation: define metrics for “step adherence,” “tool-call correctness,” cost/latency per task, and human-judged report quality; compare with state-of-the-art DRTs on common benchmarks.
  - Safety & robustness: adversarial tests for prompt injection through retrieved content; sandbox escape attempts; strategy-level static analysis (e.g., verifying tool-call schemas and termination).
  - Asynchronous and parallel tools: extend the runtime to support concurrent searches, caching, and streaming ingestion with deterministic joins.
  - Strategy formalization: evolve from free-form natural language to a lightweight “strategy DSL” that remains readable but enables static checking and reuse across teams.

Quoted anchors and figure references used:
- Figure 1: Typical DRT components and continuous progress updates (Introduction).
- Figure 2: UDR architecture—strategy processing to code; strategy execution with tools and notifications (Section 2).
- Figures 3–4: Demonstration UI with strategy editing, progress visualization, and report viewer (Section 3).
- Section 2.2 (Reliability, Efficiency, Security, State, Tool use, LM reasoning).
- Section 2.3 (Notifications and final report schema).
- Section 4 (Limitations).
- Section 5 (Conclusions & Recommendations; R1–R3).
- Appendix A (Minimal/Expansive/Intensive strategies).
- Appendix B (example prompts and raw markdown outputs).

In essence, UDR’s core advance is architectural: compile user-authored strategies into deterministic, sandboxed programs; treat the model as a local subroutine; keep state outside the prompt; and stream structured telemetry. The current paper demonstrates feasibility and articulates strong design rationale, while leaving quantitative validation and richer developer tooling as the natural next steps.
