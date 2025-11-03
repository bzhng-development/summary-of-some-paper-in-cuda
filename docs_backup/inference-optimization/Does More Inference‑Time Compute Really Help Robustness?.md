# Does More Inference‑Time Compute Really Help Robustness?

**ArXiv:** [2507.15974](https://arxiv.org/abs/2507.15974)
**Authors:** Tong Wu, Chong Xiang, Jiachen T. Wang, Weichen Yu, Chawin Sitawarin, Vikash Sehwag, Prateek Mittal
**Institutions:** 

## 🎯 Pitch

This study uncovers a surprising inverse scaling law for reasoning-focused language models, revealing that extended reasoning improves robustness when outputs are hidden but exacerbates failures when intermediate steps are exposed to adversaries. This insight profoundly impacts the safe deployment of AI systems, challenging the prevailing belief that "more thinking" inherently enhances robustness and highlighting the critical need for context-sensitive safeguards in AI design.

---

## 1. Executive Summary
This paper investigates whether giving reasoning-focused language models more computation at inference time improves robustness to adversarial inputs. It finds a two-part answer: when only the final output is visible, longer reasoning generally helps or at least does not hurt; but when intermediate reasoning steps are visible to an adversary, more computation reliably makes models less robust, revealing an “inverse scaling law” for robustness.

## 2. Context and Motivation
- Problem addressed
  - Many modern language models use explicit “reasoning chains” (intermediate steps before the final answer). A recent claim on large proprietary models reported that simply letting the model “think longer” at inference time improves robustness to adversarial prompts.
  - Two gaps remain: (i) how to apply and measure this effect in open-source, smaller models, and (ii) whether the observed robustness depends on a critical but rarely examined assumption—that intermediate reasoning remains hidden from adversaries.

- Why it matters
  - Real systems increasingly rely on reasoning traces, sometimes with tool use (calling external APIs). Robustness failures can cause data leakage (e.g., secret keys), unsafe behavior (e.g., harmful instructions), or unintended actions (e.g., unsafe tool calls). Understanding how inference-time compute interacts with robustness is thus central to safe deployment.

- Prior approaches and limitations
  - Earlier work showed accuracy and some robustness gains from inference-time scaling (longer decoding, multiple samples, search) in large, proprietary models. However, details of the scaling procedure were vague, and results focused on closed-source systems. Crucially, prior evaluations implicitly assumed hidden reasoning traces.

- Positioning of this work
  - Provides a systematic, open-source evaluation across 12 reasoning models (8B–32B) and three adversarial tasks (prompt injection, prompt extraction, harmful requests) with a transparent scaling method (“budget forcing”; Section 2.1).
  - Separately analyzes two deployment regimes:
    1) hidden reasoning chains (only final answer visible),
    2) exposed reasoning chains (intermediate steps visible).
  - Demonstrates a novel inverse scaling law: when reasoning is exposed, increasing the length of the reasoning chain consistently reduces robustness (Section 4 and Figure 4).

## 3. Technical Approach
- Core concepts (defined here when uncommon)
  - `Reasoning chain`: the model’s intermediate tokens produced during a “thinking” phase before the final response.
  - `Inference-time scaling`: allocating more computation at inference by encouraging or enforcing longer or more extensive reasoning (as opposed to training-time scaling).
  - `Budget forcing` (Section 2.1): a simple controller that constrains the reasoning chain length to a fixed token budget L at inference.
    - Mechanism: the model generates reasoning tokens. If the count reaches the budget, the system appends an end-of-thinking delimiter plus the prefix “Final Answer:” to push the model to conclude. If still below budget, it appends “Wait,” to encourage continued reasoning.

- Two-stage generation interface (Section 2.1)
  1) Reasoning stage: generate intermediate tokens conditioned on prior reasoning and the input.
  2) Response stage: generate final answer conditioned on the input and the full reasoning chain (or the truncated chain if budget reached).

- Adversarial settings and measurements (Section 2.2; Figures 2a–2c; Appendix A)
  - `Prompt injection` (SEP dataset): inputs contain a main task plus a low-priority, adversarial instruction embedded in content. Robustness is the fraction of cases where the model ignores the low-priority instruction (Figure 2a; Appendix Figure 6).
  - `Prompt extraction` (TENSORTRUST): inputs include a secret phrase in system messages plus a user attempt to extract it. Robustness is the fraction of cases where the secret is not revealed verbatim (Figure 2b; Appendix Figure 7).
  - `Harmful requests` (SORRY-BENCH): inputs directly ask for unsafe content. Robustness is the fraction of harmful prompts refused; judged by GPT‑4o‑mini (Figure 2c).

- Hidden vs. exposed reasoning evaluations
  - Hidden (Section 3; Figure 3): only the final answer is evaluated for robustness.
  - Exposed (Section 4; Figure 4): robustness is judged based on whether the reasoning chain itself contains malicious content (unsafe instructions, secrets), regardless of whether the final answer is safe.

- Mathematical intuition for the inverse scaling law (Section 4.1)
  - Plain-language summary: if an attacker can see the reasoning chain, every additional token is another “chance” for the model to output a malicious token (e.g., a secret or prohibited instruction). Even if the per-token risk is small, adding more tokens monotonically increases the total risk; with a nonzero per-step risk, the chance of at least one failure grows toward certainty as the chain length increases.
  - In notation (Section 4.1): define a set of “malicious tokens” M and a chain length L. The event that at least one malicious token appears by step L is non-decreasing in L, and if each step has a small risk p*>0, then the failure probability ≥ 1−(1−p*)^L.

- Additional scenarios beyond visibility of chains (Section 5)
  - Tool-integrated reasoning (Section 5.1; Figure 5; Appendix Figure 8): reasoning may contain pseudo-API call markers; an adversary can try to induce unsafe calls even if the chain is hidden, because the model actually performs those calls.
  - Reasoning extraction attacks (Section 5.2): hidden chains can still be coaxed into being revealed through adversarial prompting; evidence from a public red-teaming event shows repeated successes within a relatively small number of attempts.

- Experimental protocol (Section 2.2; Appendix A)
  - Models: 12 open-source reasoning models spanning DeepSeek R1 variants, Qwen3 variants, Phi-4-reasoning variants, QwQ-32B, and STAR-1 (safety-fine-tuned) variants (Appendix Table 1).
  - Compute budgets: 100 to 16,000 reasoning tokens, manipulated via budget forcing.
  - Inference config: temperature 0.6, repetition penalty 1.15, held constant.
  - Datasets: SEP (500 sampled prompts), TENSORTRUST (570), SORRY-BENCH (450), LLM‑PIEVAL (750) for tool-use simulation.

## 4. Key Insights and Innovations
- A simple, transparent scaling method works for open-source models (Section 3; Figure 3)
  - Novelty: Applies budget forcing (a publicly documented method) to a broad set of open models, clarifying what “increasing decoding steps” concretely entails.
  - Significance: Demonstrates that longer reasoning chains can improve robustness to prompt injection and extraction even in smaller models; not just a property of large, proprietary systems.

- Identification of an inverse scaling law with exposed reasoning (Section 4; Figure 4)
  - Novelty: Separates final-answer safety from reasoning-chain safety and shows that longer chains consistently degrade robustness when chains are visible.
  - Significance: Reframes the safety calculus—“think longer” is not uniformly safer; it depends on whether intermediate tokens are observable.

- Clarification of when reasoning-chain failures matter (Section 4.2, “Remark”)
  - Insight: Degradation in intermediate steps is less critical for tasks where only final outputs affect harm (some prompt injection cases), but it is directly dangerous when the adversary’s goal is to read secrets (prompt extraction) or copy unsafe procedures (harmful requests).

- Hiding is not a panacea: two concrete risk channels (Section 5; Figure 5)
  - Tool-integrated reasoning: longer chains offer more opportunities to trigger unsafe tool calls, even if the chain is hidden.
  - Reasoning extraction attacks: hidden chains can sometimes be elicited, expanding practical risk as chain length grows.

## 5. Experimental Analysis
- Evaluation setup summary (Section 2.2; Appendix A)
  - Datasets and metrics:
    - SEP (prompt injection): robustness = fraction of ignored low-priority instructions.
    - TENSORTRUST (prompt extraction): robustness = fraction of cases where the secret is not leaked verbatim.
    - SORRY-BENCH (harmful requests): robustness = refusal rate, judged by GPT‑4o‑mini.
  - Models and budgets: 12 models, 100–16,000 reasoning tokens via budget forcing.
  - Two visibility regimes: hidden vs. exposed reasoning chains.

- Main results with hidden reasoning (Section 3; Figure 3)
  - Prompt injection improves with longer chains:
    - Example: “QwQ‑32B” rises from ~35% to ~75% robustness as the budget increases from 100 to 16,000 tokens (Figure 3a).
    - Mechanism offered in Section 3: prompts include explicit safety specifications; more tokens let the model restate and adhere to these rules.
  - Prompt extraction also improves:
    - Example: “QwQ‑32B” increases from ~60% to ~80% when budget grows from 100 to 16,000 (Figure 3b).
  - Harmful requests show little change:
    - Example: “Qwen3‑8B” remains near ~70% across budgets (Figure 3c).
  - Overall: Figure 1 (left) visualizes this “longer is better or neutral” pattern when only final answers are considered; the authors average across 12 models.

- Main results with exposed reasoning (Section 4; Figure 4)
  - Prompt injection robustness sharply decreases:
    - Example: “R1‑Qwen‑14B” falls from ~90% at 100 tokens to <20% at 16,000 tokens (Figure 4a).
  - Prompt extraction robustness also decreases:
    - Example: “R1‑Qwen‑14B” drops by ~60 percentage points over the same budget increase (Figure 4b).
  - Harmful request robustness declines moderately:
    - Across models, a ~20–40% reduction as budgets increase (Figure 4c).
  - Aggregate view: Figure 1 (right) summarizes the inverse scaling law when chains are visible.

- Tool-integrated reasoning risk (Section 5.1; Figure 5)
  - Setup: simulate tool calls by allowing the model to produce special `<API_CALL>...</API_CALL>` tokens within reasoning; evaluate on LLM‑PIEVAL (Appendix Figure 8 shows the instruction).
  - Result: robustness against unsafe tool-call induction degrades with longer budgets; e.g., “PHI‑4‑REASON” drops from 100% to ~87% as budget increases from 100 to 8,000 tokens (Figure 5).

- Evidence that hidden chains can be extracted (Section 5.2)
  - Real-world red-teaming report: both “O1‑PREVIEW” and “O1‑MINI” had hidden chains exposed multiple times within fewer than 8,000 attempts (Section 5.2). This supports the practical relevance of the exposed-chain risk.

- Do the experiments support the claims?
  - Yes, within scope:
    - Hidden-chain regime: consistent upward or flat trends on SEP and TENSORTRUST (Figure 3a–b), and flat trends on SORRY‑BENCH (Figure 3c).
    - Exposed-chain regime: consistent downward trends across all three tasks (Figure 4a–c).
  - The paper also contextualizes when intermediate failures translate into practical harm (Section 4.2 “Remark”).

- Important conditions and trade-offs revealed by the experiments
  - Gains in the hidden regime depend on explicit safety specifications in prompts guiding the model (Section 3).
  - The inverse scaling law manifests when any of the following holds:
    - The chain is visible (Figure 4),
    - The chain drives tool calls (Figure 5),
    - The chain can be extracted (Section 5.2).
  - Thus, the benefit or harm of inference-time scaling is deployment-dependent.

## 6. Limitations and Trade-offs
- Assumption sensitivity
  - Hidden chains: Improvements rely on specification prompts that “teach” the model to ignore adversarial content (Section 3). Without such guidance, gains may be smaller.
  - Exposed-chain labeling: The paper evaluates whether “malicious tokens” appear in reasoning (Section 4.2). While unambiguous for secrets (exact-string leakage in TENSORTRUST), identifying “unsafe content” in chains for prompt injection or harmful requests could require pattern matching or heuristics; implementation details are not deeply elaborated.

- Scope of models and methods
  - Only open-source models (8B–32B) are tested (Appendix Table 1). Results may differ on much larger or differently trained proprietary systems.
  - One scaling strategy: budget forcing. Parallel strategies (e.g., Best-of‑N, tree search) are not evaluated; their robustness profiles remain open (Section 6).

- Evaluation constraints
  - SORRY‑BENCH judgments use GPT‑4o‑mini as an automated evaluator (Section 2.2); evaluation variance or bias is possible.
  - Compute/latency costs of longer reasoning are not reported; real-world deployments must weigh robustness gains vs. inference cost.

- Tool-use simulation
  - Tool-integrated reasoning is simulated via prompting, not by exercising real API tools in a production agent runtime (Section 5.1). While sufficient to show the trend, real systems may introduce additional complexities (auth, network errors, defense-in-depth).

- Attack coverage
  - The paper uses straightforward attacks and acknowledges that more sophisticated, targeted strategies could amplify observed degradations, especially in stronger models (Section 6).

## 7. Implications and Future Directions
- How this work shifts current understanding
  - It replaces the simple narrative “more thinking makes models safer” with a conditional statement: longer chains tend to help when chains are hidden and the task is specification-driven, but can systematically hurt when chains are visible—or effectively visible via tool actions or extraction attacks (Figures 1, 3, 4, 5; Sections 3–5).

- Practical guidance for deployment
  - If using inference-time scaling:
    - Keep reasoning hidden and hard to extract; treat extraction as an active threat, not a hypothetical (Section 5.2).
    - Add runtime safeguards that monitor and filter intermediate tokens, especially before executing tool calls (Section 5.1).
    - Use explicit safety specifications in prompts and evaluate them under your exact deployment configuration (Section 3).
  - If exposing chains (e.g., for transparency):
    - Expect inverse scaling; longer chains increase the attack surface (Figure 4).
    - Consider redaction, summaries, or post-hoc verification of chains before exposure.
    - Employ separate secret-handling channels so secrets cannot appear in model-visible context.

- Research avenues enabled or suggested (Section 6)
  - Robust parallel inference-time scaling: test Best-of‑N, self-consistency, and search-based methods for both final-output and chain-level safety.
  - Principled chain extraction attacks: automated methods to quantify leakage risk and improve defenses.
  - Tool-use safety for reasoning models: formal interfaces and guards for API calls originating inside reasoning chains; token-level policy enforcement.
  - Safety alignment targeting intermediate steps: train models to avoid generating unsafe content within reasoning, not just in final outputs.

- Downstream use cases
  - Safety-critical agents (customer support, code assistants, research agents) can use budget forcing to improve robustness when chains are reliably hidden.
  - For auditability requirements that demand exposing reasoning, orgs should budget for additional defenses and accept that longer chains may reduce robustness unless mitigations are in place.

> Bottom line (Figures 1, 3, 4, 5): inference-time scaling is a double-edged sword. It can improve robustness against prompt injection and extraction when only final outputs matter and chains remain hidden, but it reliably degrades robustness when intermediate reasoning is accessible—or becomes effectively accessible via tools or extraction attacks.
