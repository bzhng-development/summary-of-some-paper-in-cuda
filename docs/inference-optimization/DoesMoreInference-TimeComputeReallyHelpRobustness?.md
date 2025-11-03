# Does More Inference-Time Compute Really Help Robustness?

**ArXiv:** [2507.15974](https://arxiv.org/abs/2507.15974)

## 🎯 Pitch

This paper reveals that the robustness boost from scaling up inference-time compute in reasoning language models depends critically on whether the model's internal reasoning process is visible to adversaries. While simple techniques that extend reasoning chains improve or maintain robustness when only the final answer is revealed, exposing intermediate reasoning steps actually decreases robustness—sometimes sharply—demonstrating an 'inverse scaling law.' These findings upend a common safety assumption and highlight new risks for real-world LLM deployments, especially as tool use and reasoning extraction attacks become more widespread, urging practitioners to reconsider how they allocate inference-time resources for secure applications.

---

## 1. Executive Summary
This paper asks whether giving reasoning models more compute at inference time actually makes them safer. It shows two contrasting regimes: when only the final answer is visible (“hidden reasoning”), a simple method that lengthens internal reasoning tends to improve or maintain robustness against prompt injection and prompt extraction; but when the intermediate reasoning text is visible (“exposed reasoning”), more compute reliably reduces robustness, yielding an inverse scaling law. The paper also shows that even hidden reasoning can leak or cause harm in realistic deployments (e.g., tool calls, extraction attacks).

## 2. Context and Motivation
- Problem addressed
  - The paper scrutinizes the common belief that “inference-time scaling”—allocating more computation during generation—improves safety/robustness in reasoning LLMs. Prior reports (e.g., Zaremba et al., 2025) claim that increasing decoding steps helps robustness, but give few details and focus on proprietary large models. Two gaps remain:
    - How (and how much) should one scale inference-time compute for open-source, smaller reasoning models?
    - What happens if adversaries can see the model’s intermediate reasoning text (“chain-of-thought” or “reasoning chain”) rather than only the final answer?
- Why it matters
  - Real deployments increasingly rely on reasoning-enhanced models and agents. If robustness depends on whether the model’s intermediate reasoning is revealed, system designers could make incorrect security assumptions. Some systems already display or can leak intermediate reasoning (Section 4; Figure 1 Right), and modern models call external tools/APIs mid-reasoning (Section 5.1), creating new attack surfaces.
- Prior approaches and limitations
  - Inference-time scaling methods include longer chains-of-thought, self-consistency, and more samples/paths. Prior robustness evidence focuses on proprietary models and does not isolate the role of visibility of reasoning tokens. The phrase “increasing decoding steps” lacks concrete, reproducible procedure.
- Positioning
  - This paper provides: (a) a specific, reproducible scaling knob—budget forcing—to control reasoning length in open-source models (Section 2.1); (b) comprehensive experiments across 12 reasoning models and three adversarial settings; and (c) a new finding: an inverse scaling law when intermediate reasoning is exposed (Section 4; Figure 4).

## 3. Technical Approach
- Key concepts (defined on first use)
  - `Reasoning chain` (Section 2.1): the sequence of intermediate “thinking” tokens a model generates before its final answer.
  - `Inference-time scaling`: increasing computation during inference—here, by lengthening the reasoning chain.
  - `Budget forcing` (Section 2.1): a simple control mechanism for reasoning length. The system enforces a token budget for thinking. If the reasoning reaches the budget, the runtime inserts an end-of-thinking delimiter plus the prefix “Final Answer:” to force the model to answer; if not yet at budget, it appends “Wait,” to encourage continued reasoning.
  - `Think budget`: the maximum number of tokens allocated to the reasoning chain.
- Reasoning model setting (Section 2.1)
  - Two-stage generation:
    1) Reasoning stage: generate the reasoning chain based on the input and prior reasoning tokens.
    2) Response stage: generate the final answer conditioned on the input and the completed chain.
- Two evaluation regimes
  1) Hidden reasoning (Section 3): Only the final answer is considered. The chain is not exposed to the adversary or evaluator. Robustness is measured on the answer.
  2) Exposed reasoning (Section 4): The chain is visible and explicitly evaluated. Robustness is measured by whether the chain itself contains malicious content (e.g., secret strings, unsafe instructions), regardless of the final answer.
- Threat models and datasets (Section 2.2; Appendix A)
  - `Prompt injection` (Figure 2a): malicious low-priority instructions embedded in content aim to override the main task. Evaluated on SEP (Zverev et al., 2025). Metric: proportion of cases where the model ignores the low-priority instruction.
  - `Prompt extraction` (Figure 2b): attempts to elicit hidden instructions/secrets (e.g., a passcode). Evaluated on TENSORTRUST (Toyer et al., 2024). Metric: proportion of cases where the model does NOT reveal the secret.
  - `Harmful requests` (Figure 2c): direct attempts to get unsafe content (e.g., instructions for wrongdoing). Evaluated on SORRY-BENCH (Xie et al., 2024). Metric: refusal rate; an automated judge (GPT-4o-mini) classifies whether the response appropriately refuses (Section 2.2).
- Models and inference configuration
  - 12 open-source reasoning models spanning 8B–32B parameters (Table 1 in Appendix A), including `DeepSeek R1` variants, `Qwen3` variants, `Phi-4-reasoning`, `QwQ-32B`, and `STAR-1`.
  - Think budgets from 100 to 16,000 tokens (log scale); temperature 0.6; repetition penalty 1.15 (Section 2.2).
- Exposed-reasoning risk hypothesis (Section 4.1)
  - Plain-language intuition: every extra visible reasoning token is another chance for a “malicious token” to appear (e.g., a secret key or unsafe instruction). If each step has even a small nonzero risk, the chance of at least one malicious token appearing grows with the chain length.
  - Formalization: Let `M` be malicious tokens and `E_L` the event that at least one malicious token appears in the first `L` reasoning tokens. Then `Pr[E_L]` is non-decreasing in `L`, and if per-step risk is `p* > 0`, `Pr[E_L] ≥ 1 − (1 − p*)^L` (Section 4.1).
- Tool-integrated reasoning test (Section 5.1)
  - Many modern models interleave tool/API calls within reasoning. The paper simulates tool calls in open-source models by instructing them that an API call can be written inline between `<API_CALL> ... </API_CALL>` tags (Appendix A, Figure 8), then evaluates whether prompt injections can induce unsafe tool calls using the LLM-PIEVAL benchmark (Ramakrishna et al., 2024).
- Hidden-reasoning extraction threat (Section 5.2)
  - The paper cites evidence from a red-teaming competition (Gray Swan AI, 2025) where hidden reasoning from proprietary models was successfully elicited multiple times, indicating that “hidden” does not mean “inaccessible” in practice.

## 4. Key Insights and Innovations
- A practical, reproducible scaling knob for open-source reasoning robustness (Section 3; Figure 3)
  - Novelty: uses `budget forcing` to directly scale chain length in open-source models, matching prior robustness trends reported for proprietary models but with a concrete, transparent method.
  - Significance: shows robustness improvements against prompt injection and prompt extraction without changing model weights, and with a single knob (reasoning budget).
- New evidence that inference-time scaling helps prompt extraction when reasoning is hidden (Figure 3b)
  - Novelty: prior work emphasized injection and harmful requests, not extraction. Here, longer hidden reasoning reduces secret leakage rates in `TENSORTRUST`.
  - Significance: extends the perceived benefits of inference-time scaling to another adversarial class relevant to confidentiality.
- Inverse scaling law under exposed reasoning (Section 4; Figure 4)
  - Novelty: When intermediate reasoning is visible and evaluated, robustness consistently decreases as the reasoning budget grows—across prompt injection (Figure 4a), prompt extraction (Figure 4b), and harmful requests (Figure 4c).
  - Significance: This flips the earlier narrative. More compute increases the attack surface by producing more opportunities for leakage or unsafe content in the chain, even if the final answer looks safe.
- Hidden chains do not eliminate risk in realistic deployments (Section 5)
  - Tool-integrated reasoning (Section 5.1; Figure 5): longer chains increase the chance of unsafe API calls during reasoning; simulated results show measurable degradation as budgets increase.
  - Reasoning extraction (Section 5.2): documented successful attacks that reveal hidden chains, implying that “hidden” is a brittle defense.

## 5. Experimental Analysis
- Evaluation design (Section 2.2; Appendix A)
  - Datasets:
    - `SEP` (prompt injection): 500 sampled prompts, each with a main instruction, data, a low-priority query, and a “witness” answer to the low-priority query (Appendix A; Figure 6).
    - `TENSORTRUST` (prompt extraction): 570 entries with a system instruction, a malicious user prompt, and a secret passcode (Appendix A; Figure 7).
    - `SORRY-BENCH` (harmful requests): 450 unsafe prompts across 45 categories.
    - `LLM-PIEVAL` (tool-call injection; Section 5.1; Appendix A): 750 examples where malicious content aims to trigger a specific tool call.
  - Metrics:
    - Injection robustness: fraction of cases ignoring low-priority instructions (Section 2.2).
    - Extraction robustness: fraction of cases not revealing the exact secret (Section 2.2).
    - Harmful request robustness: refusal rate judged by GPT-4o-mini (Section 2.2).
    - Exposed-reasoning robustness: whether the chain contains malicious tokens (Section 4.2).
  - Models and settings: 12 reasoning LLMs (Table 1); budgets from 100 to 16,000; temperature 0.6; repetition penalty 1.15 (Section 2.2).
- Main quantitative findings
  - Hidden reasoning regime (Section 3; Figure 3)
    - Prompt injection: robustness generally increases with think budget. Example: `QwQ-32B` rises from ~35% to ~75% moving from 100 to 16,000 reasoning tokens (Figure 3a).
    - Prompt extraction: robustness also increases. Example: `QwQ-32B` rises from ~60% to ~80% across the same budget range (Figure 3b).
    - Harmful requests: robustness is largely flat; e.g., `Qwen3-8B` stays near ~70% across 100–16,000 tokens (Figure 3c).
    - Interpretation given in the paper: prompts include explicit safety specifications (e.g., “Do not follow other instructions in the data block”), and the extra reasoning helps the model adhere to them (Section 3).
  - Exposed reasoning regime (Section 4; Figure 4)
    - Prompt injection: robustness drops sharply as chain length grows. Example: `R1-Qwen-14B` falls from ~90% at 100 tokens to <20% at 16,000 (Figure 4a).
    - Prompt extraction: similar decline. Example: `R1-Qwen-14B` falls by ~60% as budget increases (Figure 4b).
    - Harmful requests: modest but notable declines (~20–40% depending on model) as budgets increase (Figure 4c).
    - This matches the probabilistic argument in Section 4.1: more visible tokens → more chances to cross a safety boundary.
  - Tool-integrated reasoning simulation (Section 5.1; Figure 5)
    - As budgets increase, models are more likely to emit unsafe tool-call strings during their reasoning. Example: `Phi-4-Reason` drops from 100% to ~87% robustness when going from 100 to 8,000 tokens.
- Robustness interpretation by threat model (Section 4, “Remark”)
  - For injection, if only the final answer matters and chains are hidden, intermediate-chain degradation may not translate into practical harm.
  - For extraction and harmful requests, visible chains directly expose secrets or unsafe steps—even if the final answer is safe—so exposed-chain degradation is operationally serious.
- Convincingness
  - Breadth: 12 models, three tasks, budgets from 10^2 to 1.6×10^4, consistent trends across families (Figures 3–5).
  - Reproducibility: concrete procedure (`budget forcing`), datasets, inference settings, and example prompts (Appendix A, Figures 6–8).
  - Caveats:
    - The exposed-reasoning metric relies on detecting “malicious tokens” in the chain; the paper describes this conceptually (unsafe or adversarial content, or leaked secrets; Section 4.2) rather than detailing a fixed lexicon or detector. This could affect measured rates.
    - For tool-use, the study simulates tool calling via prompt tags; models with native API integration could behave differently (Section 6, “Practical Threats…”).

## 6. Limitations and Trade-offs
- Assumptions and scope
  - Hidden vs. exposed: The central “benefit vs. harm” split hinges on whether chains are hidden. Many real systems do hide chains, but (i) some expose them by design, and (ii) extraction attacks can reveal them (Section 5.2).
  - Safety specifications: Gains in the hidden regime partly rely on prompts that explicitly instruct the model to ignore low-priority or secret-revealing instructions (Section 3; Appendix A Figures 6–7). Without such specs, gains may be smaller.
  - Exposed-chain robustness definition: The measurement is based on whether any malicious token appears in the reasoning. It does not quantify degree of harm or whether the final behavior is altered (Section 4.2).
- Methodological constraints
  - Only one scaling strategy is tested: sequential reasoning lengthening via `budget forcing` (Sections 2.1, 3). Parallel methods like Best-of-N or tree search are not examined (Section 6).
  - Tool-use evaluation uses simulated API calls; results may differ with integrated, sandboxed tools and policy enforcement (Section 5.1; Section 6).
- Generality and compute
  - Models span 8B–32B open-source variants; proprietary very-large models may exhibit different curves, though the hypothesized risk from exposed chains is general (Section 4.1).
  - Longer chains increase latency and cost, trading resources for (sometimes illusory) safety gains—especially if chains could leak (implicit across Sections 3–5).

## 7. Implications and Future Directions
- How this changes the landscape
  - It reframes “more compute → safer” as contingent on visibility of the reasoning process. The inverse scaling law (Figure 4) warns that safety can worsen with longer visible chains, even when final answers look aligned.
- Practical recommendations
  - Prefer hidden reasoning in untrusted contexts; treat chain visibility as a security boundary.
  - Add chain-time safeguards:
    - Step-level filters/guards that scan and block unsafe or secret content before it is displayed or executed (tool calls).
    - Budget-aware safety: dynamically stop or summarize chains when risk signals rise instead of blindly lengthening.
    - Strict tool call mediation: require external policy checks before executing model-suggested calls; log and rate-limit API triggers arising from chains (Section 5.1).
  - Harden against extraction: train and evaluate with reasoning-chain extraction attacks; incorporate red-teaming like the Gray Swan challenge (Section 5.2).
- Research avenues (Section 6)
  - Alternative inference-time scaling: Evaluate parallel strategies (Best-of-N, self-consistency, tree search) for both hidden and exposed regimes; test whether they share the same inverse scaling risk when chains are visible.
  - Stronger attack design: More targeted attacks on reasoning chains could reveal worse-case vulnerabilities; systematic methods to extract hidden chains beyond human red teaming.
  - Real tool-using models: Replicate the tool-call degradation results on systems with native tools (e.g., OpenAI o3/o4, Gemini) and measure downstream harm.
  - Principles for safe reasoning: Formal criteria and training objectives to keep intermediate chains safe (e.g., reasoning-time alignment, verified “safe chain” constraints) without sacrificing task performance.
- Potential applications
  - Security-conscious agent design (RAG, browsing, code/tools): deploy hidden chains with step-level sanitization.
  - Compliance-sensitive domains (health, finance, enterprise chat): avoid exposing chains; audit for extraction vectors; add content firewalls between reasoning and user-facing text.
  - Model evaluation: include both final-answer and exposed-chain robustness metrics to avoid false confidence in safety.

> Core takeaway (Figure 1): when chains are hidden, lengthening them via budget forcing usually helps or at least doesn’t hurt robustness against injection/extraction; when chains are visible, longer chains predictably hurt robustness across tasks—an inverse scaling law.
