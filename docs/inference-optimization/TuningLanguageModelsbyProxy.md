# Tuning Language Models by Proxy

**ArXiv:** [2401.08565](https://arxiv.org/abs/2401.08565)

## 🎯 Pitch

This paper unveils 'proxy-tuning,' a novel method to steer large language models at decoding-time using the behavioral difference between a small tuned model and its untuned counterpart—without ever accessing or modifying the large model's weights. By efficiently applying this technique, practitioners can customize powerful, potentially proprietary LMs for instruction-following, domain adaptation, or task specialization, achieving up to 91% of the performance gain of full finetuning at a fraction of the cost and with no access to parameters. This approach democratizes LM customization, enabling safe, up-to-date, and effective model behaviors even for closed-source or resource-constrained scenarios.

---

## 1. Executive Summary
This paper introduces proxy-tuning, a decoding-time method that “tunes” a large language model (LM) without touching its weights by adding a learned logit offset derived from a much smaller tuned model. It matters because it delivers most of the benefits of finetuning (instruction-following, domain adaptation, task-specific behavior) even when the large model’s parameters are private or too costly to train—closing up to 91% of the gap to fully tuned chat models on standard benchmarks (Table 2).

## 2. Context and Motivation
- Problem addressed
  - Finetuning large LMs to follow instructions, adapt to domains, or perform specific tasks usually requires access to model parameters and substantial compute. This is often infeasible when models are proprietary (e.g., GPT-4) or too large to finetune economically (§1).
- Why it matters
  - Practical: Organizations may want to customize closed-source or very large LMs for safety, style, or new knowledge without retraining.  
  - Scientific: Direct finetuning can degrade previously learned knowledge (a.k.a. “alignment tax” or catastrophic forgetting). There is value in methods that alter behavior at inference time while preserving knowledge (Conclusion; Table 3).
- Prior approaches and their shortcomings
  - Parameter-efficient finetuning (e.g., LoRA, adapters) reduces training cost but still needs weight access (Related Work).  
  - Prompt-based steering can be competitive but often uses long prompts that add inference cost and context-window pressure (Related Work).  
  - Other logit-editing methods target specific attributes (toxicity, sentiment) rather than general “do what finetuning would have done” behavior, and may require their own trained classifiers (Related Work).
- Positioning
  - Proxy-tuning reframes “tuning” as purely decoding-time guidance: combine the base model’s logits with the difference between a small tuned model (expert) and its untuned counterpart (anti-expert) so the large model behaves as if it were finetuned—without accessing its weights (§2, Fig. 1, Eq. 1). It generalizes beyond instruction-following to domain and task adaptation (§§3–5) and even works with truly black-box APIs that expose only top-k token probabilities (§7).

## 3. Technical Approach
At a high level, proxy-tuning applies logit arithmetic during decoding to nudge a base model toward the behavior induced by tuning a smaller proxy model.

- Core setup (Fig. 1; §2)
  - Base model `M`: large, untuned (e.g., `Llama2-13B` or `Llama2-70B BASE`).
  - Expert `M+`: small model that has been tuned for the desired behavior (e.g., `Llama2-7B-CHAT` for instruction-following, `CodeLlama-7B-Python` for code).
  - Anti-expert `M-`: the same small model before tuning (e.g., `Llama2-7B BASE`).
  - Requirement: all models share the same tokenizer/vocabulary (§2; footnote 3). When vocabularies differ, one could map tokens using methods like Kasai et al. (2022).

- Mechanism: how the logits are combined at each time step (Eq. 1)
  - Let `sM`, `sM+`, `sM-` be the pre-softmax scores (“logits”) over the vocabulary, computed by conditioning each model on the same generated prefix.
  - Compute new probabilities:
    - pM̃(Xt | x<t) = softmax[sM(Xt | x<t) + sM+(Xt | x<t) − sM−(Xt | x<t)]
  - Plain-language reading: take the base model’s preferences and add the “delta” learned by tuning the small model (expert minus anti-expert). This pushes up tokens favored by tuning and suppresses tokens disfavored by tuning.

- Why expert minus anti-expert?
  - The difference `sM+ − sM−` isolates what changed due to tuning the small model. Adding it to the base model’s logits aims to replicate those changes at larger scale while retaining the base model’s broader knowledge (§2; Fig. 1).

- Optional strength control (α) (§6.2; Fig. 2)
  - A scalar `α` can modulate the influence: `sM + α(sM+ − sM−)`.  
  - Larger `α` applies stronger steering (more like the tuned expert); smaller `α` keeps the base model’s behavior closer to its original.

- Black-box feasibility
  - Only requires access to next-token probabilities or logits of the base model; not its parameters (§2).  
  - Case study shows it still works with extreme restrictions: GPT-3.5 where only top-5 token log-probabilities are exposed and generation must be single-token, by framing tasks as multiple choice over {A, B, C, D} (§7).

- Execution details
  - All three models are run in lockstep at decoding time to produce their logits for the current token; those logits are combined using Eq. (1) to sample the next token, then the process repeats.  
  - Runtime overhead arises because multiple models run sequentially, but can be largely eliminated if run in parallel across GPUs with communication to aggregate logits (§C.1; Table 12).

- Intuition with a toy example
  - Suppose the base model tends to continue harmful content; the tuned expert prefers refusing such prompts. If the expert raises the logit for “I can’t help with that” while the anti-expert does not, the offset `sM+ − sM−` will raise that refusal token in the base model’s distribution, making the large model more likely to respond safely (Table 1, Toxigen example).

## 4. Key Insights and Innovations
- Decoding-time “tuning by proxy” (fundamental innovation)
  - Novelty: treat the small tuned model as an “expert” and use its change from the untuned small model to steer a large base model that you cannot finetune (§2; Fig. 1; Eq. 1).  
  - Significance: delivers most benefits of full finetuning without weight access or training cost; enables customization of proprietary models when they expose logits.

- Knowledge preservation while aligning style and reasoning (insight)
  - Evidence: on TruthfulQA (open-ended), proxy-tuned models are slightly less informative but more truthful than fully tuned chat models (Table 3).  
  - Token-level analysis shows proxy-tuning most boosts “reasoning/stylistic” tokens and myth-debunking phrases rather than factual tokens (§6.1; Table 6; LHS vs RHS deltas on GSM), supporting the idea that alignment primarily affects style and reasoning structure—not core knowledge.

- Generality across adaptation types (capability)
  - Works for instruction-following (§3), domain adaptation to code (§4), and task-specific finetuning (TriviaQA, GSM; §5).  
  - Also works with a truly black-box API (GPT-3.5) with only top-5 logits and single-token outputs (§7).

- Format induction from a small expert (capability)
  - Proxy-tuned large models learn strict output formats seen only by the small expert (e.g., GSM’s “#### final answer”), with 99.7%+ adherence (§5.2; Appendix E examples).

## 5. Experimental Analysis
- Evaluation methodology
  - Instruction-tuning (§3)
    - Base: `Llama2-13B/70B BASE`. Expert: `Llama2-7B-CHAT`. Anti-expert: `Llama2-7B BASE`.  
    - Datasets: AlpacaFarm (open-ended), GSM (math), Toxigen (toxicity avoidance), TruthfulQA (open-ended and MC) (§3.1; Table 2; Table 3).  
    - Zero-shot prompts; greedy decoding (§3.1; Table 8).
  - Code adaptation (§4)
    - Expert: `CodeLlama-7B-Python`; Anti-expert: `Llama2-7B BASE`; Base: `Llama2-13B/70B BASE`.  
    - Benchmarks: CodexEval (HumanEval) and DS-1000; pass@10 using sampled generations with top-p=0.95, temperature=0.8; filtered trivial tokens; max 512 tokens (§4.1–4.2; Table 10).
  - Task finetuning (§5)
    - Authors finetune small task experts on training data: TriviaQA (88K) and GSM (7.5K) (§5.1; §A.3; Table 11).  
    - Proxy-tune large BASE models with the 7B task expert vs 7B BASE as anti-expert (§5.2).
  - Black-box GPT-3.5 (§7)
    - REALTIMEQA; transformed to multiple-choice {A,B,C,D}; only reweights those tokens; small expert is a 7B model continued-pretrained on articles retrieved for each query (§7).

- Main quantitative results
  - Instruction-tuning effectiveness (Table 2; Table 3)
    - 13B: proxy-tuning closes on average 91.1% of the gap to `Llama2-13B-CHAT`.  
      - Example numbers:  
        > AlpacaFarm win-rate: BASE 2.1% → Proxy 83.4% vs CHAT 87.3%  
        > GSM accuracy: BASE 6.6% → Proxy 26.4% vs CHAT 32.4%  
        > Toxigen toxicity: BASE 70.4% → Proxy 0.1% vs CHAT 0.0%  
        > TruthfulQA (Info+True): BASE 49.1% → Proxy 82.0% vs CHAT 80.4%  
    - 70B: closes 88.1% of the gap on average.  
      - Example numbers:  
        > AlpacaFarm: 3.7% → 88.0% vs 90.4%  
        > GSM: 9.6% → 32.0% vs 51.8%  
        > Toxigen: 67.4% → 0.0% vs 0.0%  
        > TruthfulQA (Info+True): 53.9% → 85.1% vs 79.6%
    - Truthfulness vs informativeness (Table 3):  
      > 13B: Proxy is 1.6 points less informative than CHAT (91.4 vs 93.0) but 3.2 points more truthful (90.5 vs 87.3).  
      > 70B: Proxy is 1.0 point less informative (92.8 vs 93.8) but 6.5 points more truthful (92.3 vs 85.8).
  - Code adaptation (Table 4)
    - Strong gains over BASE but typically below fully tuned code models.  
      > 13B CodexEval pass@10: 33.7 → 65.7 (proxy) vs 78.6 (directly tuned).  
      > 70B DS-1000 pass@10: 43.9 → 50.6 (proxy) vs 67.6 (directly tuned).
    - Hypothesized reason (§4.2): larger-scale generic pretraining adds less to already domain-specialized code behavior; contrast term `(large BASE − small BASE)` does not help the 7B code expert as much as in instruction-following.
  - Task-specific finetuning (Table 5)
    - Big absolute improvements vs BASE; near par with fully tuned larger models in TriviaQA.  
      > 13B TriviaQA: 36.8 → 55.9 (proxy) vs 59.5 (full finetune).  
      > 70B TriviaQA: 45.2 → 62.7 (proxy) vs 63.1 (full finetune).  
      > 70B GSM: 9.6 → 53.9 (proxy) vs 67.9 (full finetune).  
    - Formats enforced: >99.7% of proxy-tuned generations follow GSM’s strict “#### final answer” format (§5.2; Appendix E examples).
  - Black-box GPT-3.5 (Table 7)
    - REALTIMEQA Acc.:  
      > GPT-3.5 base 54.2% → proxy-tuned 56.5% (+2.3%, p<0.0001).  
      - Notable because both expert and anti-expert are weaker than GPT‑3.5, yet their contrast still adds signal (§7).
- Analyses and diagnostics
  - Token-level impact (§6.1; Table 6)
    - On GSM, proxy-tuning increases probability of tokens on the left-hand side of equations (reasoning) more than right-hand side (answers):  
      > Δp LHS ≈ 0.131 vs RHS ≈ 0.056; p<1e−4.  
    - On TruthfulQA, most-boosted tokens are stylistic hedges that counter misconceptions, e.g., “There is no scientific…”, “is a common myth” (Table 6).
  - Steering strength trade-off (Fig. 2)
    - Varying α in `sM + α(sM+ − sM−)` yields a smooth trade-off: increasing α raises truthfulness but can lower informativeness, with a peak around α=0.4 for informativeness.
  - Efficiency and when changes happen (§C.1–C.2)
    - Runtime overhead vs fully tuned models: ~2.4× at 13B and ~1.5× at 70B if run sequentially; largely mitigated by parallel execution across GPUs (Table 12).  
    - Fraction of positions changed by proxy-tuning is 13–25% depending on dataset, with strongest influence early in the generation (Fig. 3).

- Comparison to LoRA for task finetuning (Appendix D; Table 15, Table 14)
  - Accuracy: Mixed; LoRA sometimes wins (TriviaQA), sometimes loses (GSM at 13B).  
    > 13B GSM: proxy 43.9 vs LoRA 32.4; 70B GSM: proxy 53.9 vs LoRA 63.0 (Table 15).  
  - Training efficiency: training a 7B expert fully (for proxy-tuning) is far faster than LoRA on 70B (33–39h vs 459h on identical hardware; Table 14).  
  - Takeaway: even with weight access, proxy-tuning can be the more practical route to strong performance quickly.

- Do the experiments support the claims?
  - Yes for instruction-following and task finetuning: large, consistent gains and gap-closing numbers (Table 2, Table 5).  
  - Partially for domain adaptation to code: substantial gains vs BASE but not surpassing fully tuned models or always beating the small code expert (Table 4), clarifying the method’s scope.

## 6. Limitations and Trade-offs
- Access assumptions
  - Needs next-token probabilities/logits from the base model at each step; many proprietary APIs do not expose full logits or allow mid-generation conditioning. The GPT-3.5 case works only by constraining to MC tokens and single-token outputs (§7; footnote 4).
- Tokenizer/vocabulary alignment
  - Method assumes a shared vocabulary between base/expert models or a mapping strategy (§2; footnote 3).
- Computational overhead at inference
  - Naively triples forward passes (base, expert, anti-expert), increasing latency by ~1.5–2.5× if run sequentially (Table 12). Parallelization alleviates this but uses more devices (§C.1).
- When the contrast helps less
  - In tightly specialized domains like code, the `(large BASE – small BASE)` term may not add much to a specialized expert (§4.2), so proxy gains may plateau below fully tuned large models (Table 4).
- No weight updates
  - Behavior only changes at decoding time; there is no persistent change to the base model’s parameters. Applications requiring offline batch inference without multi-model orchestration would not benefit.
- Limited ablations on expert/anti-expert choices
  - The paper fixes a natural choice (same small model before/after tuning), but does not deeply explore mismatched experts, multiple experts, or per-layer contrasts that might further improve results.

## 7. Implications and Future Directions
- Landscape shift
  - Proxy-tuning shows that high-quality alignment and task adaptation can be delivered as a service layer on top of large, even proprietary LMs, provided they expose logits. This reduces the need for repeated, expensive full-model finetunes and encourages providers to expose probability outputs (Conclusion).
- Practical applications
  - Rapidly “upgrade” a base model to:  
    - follow instructions safely (Table 2; Table 1 examples),  
    - adopt strict output formats (GSM; §5.2),  
    - inject new knowledge in multiple-choice settings with black-box models (GPT-3.5; §7),  
    - and adapt to domains (code; Table 4) with minimal engineering.
- Research directions
  - Better expert composition: multiple small experts (safety, style, domain) combined with tunable weights; adaptive α scheduling during a generation (Fig. 2 suggests trade-offs).  
  - Smarter time-step selection: proxy-tuning affects early tokens the most (Fig. 3); selective application could cut cost without sacrificing quality (§C.2).  
  - Robustness to tokenizer mismatch and limited-logit APIs: develop principled token mapping or top-k-only steering beyond MC settings (§7).  
  - Theory of knowledge preservation: the truthfulness gains with slight informativeness declines (Table 3) point to a promising path for alignment without knowledge loss.  
  - Beyond text: extend logit-contrast steering to multimodal and diffusion LMs (Related Work cites early steps).

Overall, proxy-tuning is a simple yet powerful decoding-time technique: add the small tuned model’s “change signal” to the large base model’s logits (Eq. 1). The method delivers striking performance across diverse settings, often rivaling fully tuned large models in instruction-following and task finetuning (Table 2, Table 5), while remaining feasible for black-box systems (§7). The primary trade-off is inference-time compute and reliance on logit access, but the approach opens a practical path to customize large LMs quickly and safely.
