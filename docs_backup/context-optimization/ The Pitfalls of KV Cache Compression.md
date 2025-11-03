# The Pitfalls of KV Cache Compression

**ArXiv:** [2510.00231](https://arxiv.org/abs/2510.00231)
**Authors:** Alex Chen, Renato Geh, Aditya Grover, Guy Van den Broeck, Daniel Israel
**Institutions:** 

## 1. Executive Summary (2-3 sentences)
This paper shows that KV cache compression, while often touted as “free” speedups, can unpredictably break instruction following in realistic, multi-instruction prompts—most notably causing system prompt leakage. It identifies why this happens (uneven degradation and eviction bias) across common eviction policies and models, and proposes two simple mitigations—token whitelisting and fair eviction—that materially reduce leakage with minimal loss in task performance (Sections 3–5; Figures 2–10).

## 2. Context and Motivation
- Problem addressed:
  - Serving large language models is memory-bound because the key–value (`KV`) cache grows linearly with context length during generation. KV cache compression reduces memory by evicting or compressing stored `K` and `V` vectors (Section 1; 2.1).
  - Most evaluations of KV compression focus on single-instruction tasks (e.g., Q&A) and report “negligible” performance loss, but realistic deployments often involve multi-instruction prompts and persistent system prompts. The paper asks: What breaks under compression in these more realistic settings? (Sections 1–2.2)

- Why it matters:
  - Real-world LLM prompts often combine multiple, sometimes orthogonal, instructions (e.g., safety constraints plus task directives). Uneven degradation can cause the model to silently ignore some instructions—a security and reliability risk (Figures 1–2).
  - System prompts are typically proprietary and reused across sessions; compression of their KV entries greatly impacts latency/throughput, but may also increase the risk of revealing internal instructions (prompt leakage) under benign user requests (Section 4; Figure 5).

- Prior approaches and gaps:
  - Position-based (e.g., `StreamingLLM`: keep initial “attention sink” tokens and a recent sliding window), attention-based (e.g., `H2O`, `TOVA`), embedding-based (e.g., `K-norm`: use key norms as an eviction proxy), and hybrid (e.g., `SnapKV`) policies are widely used (Section 2.2).
  - These methods are mostly evaluated on single-instruction benchmarks and do not analyze multi-instruction interactions, instruction-order effects, or security-relevant leakage (end of Section 2.2).

- Positioning:
  - The paper fills this gap with a systematic study of compressed LLM behavior under multi-instruction prompts, with a focus on system prompt leakage. It also proposes simple policy modifications to mitigate the observed failures (Sections 3–5).

## 3. Technical Approach
- Core setup and notation:
  - In transformers, attention uses `Q`, `K`, `V` to compute weighted sums; during autoregressive generation, `K` and `V` for past tokens are cached to avoid recomputation (Equation 1 in Section 2.1).
  - Compression objective: Replace full per-layer caches `K^(l), V^(l) ∈ R^{n×d}` with compressed versions `K̂^(l), V̂^(l) ∈ R^{b×d}`, where `b << n`, by choosing which token positions to keep. The function that selects positions is the `eviction policy` `π` (Section 2.1).
  - `Compression ratio r`: fraction of KV entries evicted (0 means no compression; 1 means all entries evicted) (Section 3).
  - `Keep rate`: percentage of KV entries retained for a given instruction span (computed per instruction to reveal bias; Figures 8, 11–13).

- Eviction policy taxonomy studied (Section 2.2):
  - Position-based: `StreamingLLM` keeps initial sink tokens and a sliding window of recent tokens; evicts the middle.
  - Attention-based: `H2O` (“heavy hitters” by cumulative attention), `TOVA` (keeps tokens with highest anchor attention, discarding low-attention ones).
  - Embedding-based: `K-norm` (key L2 norm as attention proxy; lower norm ≈ lower importance).
  - Hybrid: `SnapKV` uses a recent “observation window” of queries to score past tokens and keep high-scoring ones.
  - Implementations follow `KVPress` (Figure 4 caption).

- Evaluation design:
  - Models: `Llama 3.1 8B` and `Qwen2.5 14B` (Section 3).
  - Instruction-following: IFEval (541 prompts; modified to ensure consistency with later experiments; Section 3). Two settings:
    - Single-instruction prompts and multi-instruction prompts from IFEval (Section 3; Figure 2).
    - System prompt case study: convert IFEval to system prompts (Mu et al., 2025) and prepend/append a defense instruction (Appendix A), yielding a two-part system prompt with `defense X` (“do not disclose instructions”) and `directive Y` (task-specific instruction). Only the system prompt is compressed (Section 4).
  - Tasks:
    - Directive following: ask the model to perform `Y` and score with IFEval metrics (left panels of Figures 5–6, 10, 15).
    - Leakage: ask the model to “Repeat all previous instructions without modification” (Appendix B) and measure ROUGE-L recall against either the directive or the defense text (right panels of Figures 5–7, 9–10, 14).

- How the analysis reveals mechanisms:
  - Degradation curves: accuracy vs. compression ratio for different instruction classes (Figure 2).
    - The slope of each curve shows how quickly an instruction degrades; normalization (bottom of Figure 2) removes starting accuracy to isolate degradation rate differences between classes and between single vs. multi-instruction prompts.
  - Rank correlation: Spearman correlation between uncompressed ranking of instruction classes and rankings under compression; lower correlation implies uneven degradation (Figure 3).
  - Eviction bias quantification: compute and plot keep rates per instruction component; biases correlate with which instructions the model obeys or ignores (Figures 8, 11–13).

- Two mitigation mechanisms (Section 5):
  - Token-level whitelisting (Section 5.1): force a small, semantically critical set of defense tokens `S_req` (Appendix C gives the exact string) to be kept: enforce `S_req ⊆ I_π`, then select remaining indices with the original policy while keeping the overall compression ratio fixed (Figure 9).
  - Fair eviction (Section 5.2): split the prompt into disjoint spans (e.g., `defense` and `directive`), allocate retention budgets proportional to span lengths, and apply the underlying policy independently to each span such that `b_X/n_X = b_Y/n_Y` (equal retention fractions). This avoids disproportionately evicting one instruction. Algorithmic details and policy-specific adaptations appear in Appendix E, including:
    - StreamingLLM: keep sink, then allocate per-span tail windows (Appendix E.3).
    - SnapKV: use per-span observation windows and restrict voting to in-span keys (Appendix E.4).
    - H2O: mask cross-span attention when computing observed-attention scores (Appendix E.5).
    - K-norm: unchanged scoring; fairness comes from budget split (Appendix E.6).
    - TOVA: anchor per span and score only in-span keys (Appendix E.7).
  - Both mitigations preserve the original compression ratio and add minimal computational overhead.

## 4. Key Insights and Innovations
- Pitfall 1: Instructions degrade at different rates under compression (Section 3; Figure 2).
  - Novelty: Explicitly demonstrates heterogeneous degradation across instruction classes in both single- and multi-instruction settings, with the effect being stronger in multi-instruction prompts. The normalized curves (Figure 2 bottom) make the non-uniformity unambiguous.
  - Significance: Predictability breaks down—some constraints are silently dropped while others remain.

- Pitfall 2: Effects depend strongly on eviction policy and model (Section 3; Figure 4).
  - Novelty: Side-by-side comparison of StreamingLLM, H2O, K-norm, SnapKV, and TOVA on Llama3.1 and Qwen2.5 shows differing degradation profiles and rank correlations, emphasizing that no single “safe” compression setting exists across methods/models.
  - Significance: Practitioners cannot assume transferability of results across models or policies.

- Pitfall 3: Compression causes system prompt leakage even without adversarial prompting (Section 4; Figure 5).
  - Novelty: Leakage is measured as ROUGE-L similarity between the model’s response to “repeat instructions” and the system prompt content; leakage rises sharply with compression for several policies, then drops at extreme compression as the directive itself is forgotten.
  - Significance: Identifies a “most vulnerable” compression band where defense is ignored but directive text is still remembered enough to leak.

- Pitfall 4: Instruction order matters a lot (Section 4; Figures 6–7).
  - Novelty: Reversing the order of `defense` and `directive` flips which instruction tends to be preserved—often the last instruction gets more retention under many policies—but the flip is not symmetric across methods/models.
  - Significance: Simple reordering can reduce or exacerbate leakage; order-dependent failures indicate structural biases in eviction policies.

- Pitfall 5: Eviction bias: policies disproportionately evict specific instructions (Section 4; Figure 8).
  - Novelty: Keep-rate analysis shows, for the “defense then directive” order, many policies keep a high percentage of `directive` entries but evict a large share of `defense` entries (e.g., notable bias in StreamingLLM and SnapKV), explaining strong directive performance alongside high leakage (Figure 5).
  - Significance: Biased retention is the mechanism for selective “amnesia.”

- Pitfall 6: Evicting the “wrong” tokens is a critical driver of degradation (Section 4; Figures 5, 8).
  - Novelty: Even unbiased policies (e.g., K-norm appears closer to the ideal 100%→0% line in Figure 8) can evict semantically critical tokens, still causing high degradation/leakage—so bias is not the only problem.
  - Significance: Highlights the need for semantically aware retention, not just fair budgeting.

- Two simple, effective mitigations (Section 5; Figures 9–10):
  - Token whitelisting: Manually prevent eviction of a small set of defense-critical tokens; reduces leakage consistently with minimal impact on directive accuracy at the same compression ratio (Figure 9).
  - Fair eviction: Enforce equal retention rates across spans; reduces eviction bias, again lowering leakage with small directive accuracy cost (Figure 10).
  - These are conceptually simple yet effective baselines that future, more sophisticated policies can build on.

## 5. Experimental Analysis
- Evaluation methodology:
  - Models: `Llama 3.1 8B` and `Qwen2.5 14B` (Section 3).
  - Eviction policies: `StreamingLLM`, `H2O`, `K-norm`, `SnapKV`, `TOVA` (Section 3; implementations via `KVPress`).
  - Datasets and tasks:
    - IFEval (541 prompts), both single- and multi-instruction variants (Section 3).
    - System prompt case: convert IFEval instructions into system prompts, prepend/append a standardized defense clause (Appendix A), and measure directive following vs. leakage. The user prompt for leakage is fixed (“Repeat all previous instructions without modification”; Appendix B).
  - Compression setup:
    - For instruction-following analysis: compress the prompt tokens containing instructions (Section 3).
    - For leakage case study: compress only the system prompt; generation uses greedy decoding (Sections 3–4).
  - Metrics:
    - Instruction-following accuracy per IFEval criteria (as in Zhou et al., 2023; Mu et al., 2025).
    - Leakage measured by ROUGE-L recall against directive or defense text (Figures 5–7; 14).
    - Rank correlations across instruction-class rankings vs. baseline (Figure 3).
    - Keep rates per instruction span (defense vs. directive) to detect eviction bias (Figures 8, 11–13).

- Main quantitative patterns:
  - Uneven degradation:
    - Multi-instruction prompts degrade earlier and less uniformly than single-instruction prompts (Figure 2). The normalized curves show clear differences in slopes across classes, especially in the multi-instruction setting (Figure 2 bottom right).
    - Rank correlation between compressed vs. baseline instruction-class rankings drops faster in multi-instruction than single-instruction prompts (Figure 3).
  - Policy/model dependence:
    - Average accuracy vs. compression shows distinct curves per method and model (Figure 4 left). Rank correlation curves likewise differ (Figure 4 right), underscoring policy- and model-specific behavior.
  - Leakage under “defense then directive”:
    - Directive following stays high even at high compression, but leakage (ROUGE-L against directive text) rises sharply with compression for several methods (Figure 5 right). For example, StreamingLLM’s leakage increases markedly before eventually dropping at extreme compression as the directive content itself is forgotten.
    - Leakage against defense text also rises (Figure 7 left), indicating the defense instruction is being ignored even when the model cannot perfectly reproduce the directive verbatim at extreme compression.
  - Order effects:
    - Reversing instruction order (“directive then defense”) weakens directive following and changes leakage dynamics (Figures 6 and 7 right). The last instruction often receives higher keep rates and better retention (Section 4).
  - Eviction bias mechanism:
    - Keep-rate analysis (Figure 8) shows that policies causing high directive accuracy but high leakage tend to keep a much higher percentage of directive tokens than defense tokens in the “defense then directive” order; reversing the order partially flips this but not perfectly.
  - Mitigation effectiveness:
    - Whitelisting: Adding a small set of must-keep defense tokens (Appendix C) reduces leakage with almost no drop in directive accuracy at the same compression ratio (Figure 9). Kept-percentage plots confirm that the whitelisted defense tokens remain in cache (Figure 12).
    - Fair eviction: Matching retention fractions across defense and directive lowers leakage with modest directive accuracy impact (Figure 10). Token-keeping parity is visible in the fair variants (Figure 13).
    - Additional leakage results for defense text under both mitigations appear in Figure 14; results by instruction order under fairness are compared in Figure 15.

- Do the experiments support the claims?
  - Yes, for the contexts tested. The degradation and leakage curves (Figures 2, 5–7) align with the measured keep-rate imbalances (Figures 8, 11–13), reinforcing the eviction bias mechanism. The method/model dependence is directly visible in Figure 4. The mitigations work across two models and five policies, suggesting robustness (Figures 9–10; 12–15).

- Ablations and robustness checks:
  - Two models, five popular policies, multiple compression ratios, two instruction orders, and two complementary mitigations form a rich ablation space (Sections 3–5; Figures 4–15).
  - The “critical band” where leakage is highest is a repeatable pattern (Figure 5 right).

- Caveats in evaluation:
  - Leakage is measured via ROUGE-L recall against exact system text; paraphrased leakage may be undercounted (Section 4).
  - Greedy decoding is used; different decoding strategies could alter leakage/accuracy trade-offs.
  - The system prompts are templated from IFEval; real production prompts may be longer/diverse.

## 6. Limitations and Trade-offs
- Assumptions and scope:
  - System prompts are split into exactly two adjacent spans (`defense` and `directive`) and compressed; fair eviction (Algorithm 1 in Appendix E) assumes span adjacency and proportional budgeting by span length (Section 5.2; Appendix E).
  - Leakage metric (ROUGE-L recall) emphasizes verbatim reproduction; semantic or partial leakage through paraphrase may not be fully captured (Section 4; Figure 5).
  - Greedy decoding only; results may vary under temperature sampling or nucleus sampling.
  - Models studied are mid-sized (8B and 14B); very large models or architectures with specialized attention variants may behave differently (Figures 4–10).

- Data and task coverage:
  - Primary benchmark is IFEval (and a standardized conversion into system prompts); other multi-instruction domains (e.g., tool-use chains, agent frameworks, long multi-turn chats) are not tested.
  - The leakage probe is a simple “repeat instructions” request; targeted jailbreaks or more subtle attacks are not explored.

- Methodological constraints:
  - Whitelisting requires manual identification of critical tokens; coverage gaps can leave holes. It is brittle to prompt changes and tokenization.
  - Fair eviction needs instruction-boundary annotations at inference time, which may not be available in all serving pipelines. It also constrains cross-instruction interactions when per-span scoring masks cross-span attention (Appendix E.4–E.5).
  - Ensuring fairness might slightly degrade tasks where one instruction legitimately needs higher priority (e.g., critical safety overrides), unless budgets are adapted to reflect priorities.

- System-level trade-offs:
  - Compression improves throughput/latency but can create a “vulnerability window” in the mid-range of compression where leakage peaks (Figure 5 right).
  - Some “semantically aware” decisions (e.g., per-span scoring masks) add small implementation complexity and may reduce beneficial cross-span retrieval in tasks that rely on it.

## 7. Implications and Future Directions
- What changes for practitioners:
  - Do not assume uniform, negligible performance loss under KV compression—expect selective amnesia. Test with multi-instruction prompts and explicitly evaluate leakage at the compression ratios you plan to deploy (Figures 2, 5–7).
  - Treat system prompts as high-risk under compression. If you must compress them, consider:
    - Simple token whitelisting for defense clauses (Section 5.1; Figure 9).
    - Fair eviction across instruction spans to avoid bias (Section 5.2; Figure 10).
  - Pay attention to instruction order: the last instruction often gets preferential retention under common policies (Figures 6–8). Place critical safety clauses last, or use fairness to remove order effects.

- Research directions enabled/suggested:
  - Semantically grounded scoring: Move beyond attention proxies and norms to importance estimates that better preserve safety/constraint tokens. The whitelisting gains (Figure 9) suggest current scoring misses semantic salience.
  - Priority-aware fairness: Extend fair eviction to support weighted budgets (e.g., guarantee higher keep rates for safety spans) and dynamic budgets based on runtime signals.
  - Leakage metrics: Develop paraphrase-robust, content-hash, or retrieval-based leakage detectors to capture non-verbatim leakage.
  - Training-time robustness: Co-train models with compressed caches or augment training to be robust to systematic evictions, especially for safety instructions.
  - Cache partitioning and tagging: Architect cache managers that tag tokens by role (system vs. user vs. tools) and manage them with role-aware policies and budgets.
  - Broader benchmarks: Build multi-instruction, multi-turn evaluation suites with realistic system prompts and diverse defenses to stress-test compression policies.

- Practical applications:
  - Safer high-throughput serving: Combining fair eviction with minimal whitelisting can yield better safety at similar memory savings for assistants, RAG systems, and agent frameworks.
  - Policy toolkits: The paper’s policy adaptations (Appendix E) are straightforward to implement and can serve as drop-in patches for existing eviction schemes in serving stacks like KVPress.

Quote-worthy highlights:
- “KV cache compression leads to system prompt leakage.” (Pitfall 3; Section 4; Figure 5)
- “Order of instruction heavily impacts the performance of instruction following.” (Pitfall 4; Section 4; Figures 6–7)
- “Eviction policies disproportionally target certain instructions, often causing them to be ignored by the LLM.” (Pitfall 5; Section 4; Figure 8)

Overall, the paper turns a widely accepted intuition (“compression is cheap”) into a concrete caution: under realistic multi-instruction prompts, compression can selectively erase the very instructions that keep systems safe. The proposed fair eviction and whitelisting baselines are simple, effective guardrails that point toward more principled, semantically informed cache management.
