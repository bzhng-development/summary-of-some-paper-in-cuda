## 6. Limitations and Trade-offs

### 6.1 Small Benchmark Sizes and Absence of Confidence Intervals

**The assumption or constraint.** The primary reasoning benchmarks, AIME24 and AIME25, each contain only 30 problems. Accuracy is estimated as the average pass rate over 8 samples per problem, but no confidence intervals, standard errors, or statistical tests are reported. The paper states that on AIME25 TriAttention matches Full Attention exactly at 40.8 % (Table 4), and on MATH 500 it “closely matches” (68.4 % vs. 69.6 %).

**The consequence.** With a 30‑problem test set, an observed accuracy difference of several percentage points can easily arise from sampling noise, especially when the per‑problem pass rate is estimated from only 8 draws. The claim of “matching Full Attention” is thus statistically unvalidated; the true accuracy of TriAttention at the chosen budget could be meaningfully lower or higher. The same uncertainty affects the ablation experiments (Table 3), where performance drops of a few percentage points are interpreted as meaningful but could be within the test‑set variance. Practitioners cannot reliably determine whether a particular deployment will see the reported gains or whether a more expensive configuration is actually necessary.

**What evidence exists in the paper.** The paper does not report any measure of variability (error bars, confidence intervals, or significance tests) in any benchmark or ablation table. The AIME24/AIME25 results in Tables 1 and 4, the budget sweeps in Figure 5, and the ablation in Table 3 all present single‑point estimates without uncertainty.

**Mitigation status.** No mitigation is attempted; the limitation is not discussed. Future work would benefit from reporting 95% binomial confidence intervals (easily computable from the sample sizes) and, ideally, from testing on a larger evaluation corpus, though AIME’s fixed size is inherent.

---

### 6.2 Deployment Difficulty: Requirement of Pre‑RoPE Vector Access

**The assumption or constraint.** TriAttention’s scoring function requires access to the pre‑RoPE query and key vectors—the raw vectors before the positional rotation is applied—as well as the per‑head statistics $\mathbb{E}[\mathbf{q}_f]$, $R_f$, etc. Standard inference engines (vLLM, TensorRT‑LLM, Hugging Face Transformers with FlashAttention‑2) typically compute, store, and compress the *post‑RoPE* keys. The paper does not detail how the pre‑RoPE vectors are extracted during inference, nor does it discuss the engineering effort needed to expose them without slowing down the core attention kernel.

**The consequence.** Adopting TriAttention in existing production stacks is likely to require non‑trivial modifications to the model code, potentially breaking optimised fused attention kernels that consume post‑RoPE keys internally. The calibration step is straightforward (it can be run offline), but the online scoring pass must intercept Q and K before the RoPE rotation, which may not be readily available after the rotation and caching have already happened. Without a well‑engineered solution, the method may be limited to research prototypes that modify the transformer implementation, reducing its practical impact despite the reported throughput improvements.

**What evidence exists in the paper.** Section 4 mentions “we directly use their representations $\mathbf{k}_f$” but does not describe the inference‑time hook. The throughput measurements (Section 5.2.4) do include the overhead of scoring and pruning, but they are obtained in a setting where pre‑RoPE access is presumably implemented; the difficulty of achieving this in other frameworks is not discussed. Appendix J’s OpenClaw demo runs a 32B‑parameter model on an RTX 4090, but again no details about code modifications are given.

**Mitigation status.** The paper does not acknowledge this as a limitation. The future‑work paragraph (Appendix A) only mentions “a dedicated, hardware‑aware inference kernel” for reducing *latency* of the trigonometric series computation, not for the general problem of pre‑RoPE access. A practical mitigation would be to show a reference integration with a popular inference framework and to quantify the total lines of code change required.

---

### 6.3 Unexplored Overhead Scaling and Per‑Step Latency

**The assumption or constraint.** TriAttention prunes the KV cache once every β = 128 tokens, scoring every remaining key independently and retaining the top‑B. The end‑to‑end throughput is measured using this schedule, and the reported speedups (2.5× on AIME25, 6.3× on MATH 500, Tables 4 and 5) incorporate this overhead. However, the paper does **not** separately analyse how the scoring cost grows with the number of cached keys or the sequence length, nor does it report per‑token latency (only overall tokens per second).

**The consequence.** For sequences much longer than the tested maximum (32 K tokens) or with larger KV budgets, the scoring pass could become a noticeable fraction of inference time, potentially eroding the throughput advantage. Moreover, the periodic scoring introduces a burst of compute every 128 tokens; if this burst creates jitter, it could degrade interactive latencies, which matters for real‑time applications even if average throughput remains high. Since the scaling behavior of the scoring cost is not characterised, practitioners cannot extrapolate the reported efficiency gains to contexts of, say, 128 K tokens or to hardware configurations with different compute–memory ratios.

**What evidence exists in the paper.** All throughput numbers are taken on an A100 80 GB with 16 K‑decoding‑length measurements. The maximum batch size is chosen to fill memory after compression; details of how the pruning overhead varies with sequence length, budget, or batch size are absent. The supplementary material (Appendix G) shows that using more future offsets improves accuracy, but it does not report the corresponding increase in latency.

**Mitigation status.** The paper acknowledges that future work could design a dedicated kernel to “further accelerate the computation of trigonometric series and the subsequent cache pruning process” (Appendix A), but it does not provide any latency profiling of the current Python‑level implementation. Without a microbenchmark, the cost‑benefit of the method at extreme scales remains unquantified.

---

### 6.4 Uniform KV Budget Across Layers and Heads

**The assumption or constraint.** TriAttention applies the same fixed memory budget B to every attention head in every layer. The scoring function is computed identically across all heads, and the same number of keys is retained per head. This is a deliberate simplification to facilitate comparison with prior work that also uses uniform budgets (e.g., R‑KV, SnapKV).

**The consequence.** In reality, different heads exhibit very different attention patterns (local, retrieval, sink, etc.) and vary widely in how much compression they can tolerate. Forcing the same budget onto a head that primarily attends to a sliding window and onto a head that must retain long‑range retrieval tokens is likely suboptimal: some heads are allocated memory they do not need, while others are starved. An adaptive, per‑head budget allocation could yield additional accuracy for the same total memory, or the same accuracy with a smaller total budget.

**What evidence exists in the paper.** The paper does not evaluate any per‑head budget variation. The ablation on concentration‑based weighting (Table 3B) adjusts the relative importance of the two scoring terms per band but does not change the number of retained keys per head. The cross‑architecture analysis (Appendix I) shows that concentration varies across heads, but no experiments test whether redistributing the budget accordingly improves performance.

**Mitigation status.** The paper explicitly mentions “head‑specific budgets” as a direction for future work (Appendix A). The limitation is recognised but not addressed in the present study.

---

### 6.5 Degradation Under Extreme Memory‑Retention Pressure

**The assumption or constraint.** TriAttention’s scoring function predicts key importance from pre‑RoPE centres and approximates future queries as their mean. This approximation holds best when the actual queries are tightly concentrated around their centres; i.e., when $R_f$ is very high. On tasks that require maintaining a large number of diverse intermediate states over very long distances, even a small amount of query variance can accumulate errors, causing some critical keys to receive scores that are slightly too low and, eventually, to be evicted.

**The consequence.** The paper demonstrates this limit with the Recursive State Query benchmark (Figure 5D): beyond a recursion depth of 18, TriAttention begins to lag behind Full Attention, while R‑KV already fails catastrophically at depth 16. Although TriAttention is substantially more robust than prior methods, it still loses some information when the memory pressure becomes extreme. For tasks with exceptionally long reasoning chains (e.g., very deep recursive algorithms, multi‑step planning over thousands of tokens), the gap to Full Attention may widen further, though the paper does not test depths beyond 20.

**What evidence exists in the paper.** Figure 5D clearly shows that at depth 20 (the maximum tested), TriAttention’s accuracy drops to approximately 55 % (estimated from the plot), whereas Full Attention remains above 70 %. The text notes that “only beyond depth 18 does TriAttention begin to lag behind,” but the lag is not quantified with a table.

**Mitigation status.** The paper does not suggest a specific mitigation for this regime, beyond the generic future work on refined compression strategies (Appendix A). One could imagine increasing the budget for particularly hard tasks or using a dynamic budget, but those approaches are not explored.

---

### 6.6 Generalisation Beyond Mathematics‑Oriented Long‑Reasoning

**The assumption or constraint.** The strongest improvements are demonstrated on competition‑level mathematical reasoning benchmarks (AIME, MATH 500). While the appendix includes LongBench and RULER results (Appendix F), the accuracy gains over compression baselines are notably smaller on those broader tasks. For example, on LongBench the average improvement over the next‑best compression method (Ada‑KV+SnapKV) is 2.5 points (48.1 vs. 45.6, Table B), whereas on AIME25 the gap over R‑KV is 15.4 percentage points. The paper does not analyse why TriAttention’s distance‑preference scoring is particularly beneficial for mathematical reasoning or whether it transfers equally to tasks like multi‑turn dialogue, factual news QA, or code summarisation where attention patterns may be less distance‑structured.

**The consequence.** A practitioner deploying TriAttention on a general‑purpose assistant that handles a mix of reasoning, retrieval, summarisation, and conversation might see a less dramatic accuracy retention than the headline AIME numbers suggest. The method may be highly effective for certain long‑reasoning workloads but only moderately beneficial for others. Without a systematic investigation of task characteristics that correlate with TriAttention’s advantage, it is hard to predict its performance on an unseen task.

**What evidence exists in the paper.** The LongBench subtask breakdown (Table B) shows that TriAttention wins on 11 of 16 subtasks but that its lead is often narrow, and on some subtasks (e.g., 2Wiki, PaRe, LCC) it slightly trails a competitor. The RULER retrieval result (66.1) is stronger, but the paper does not report TriAttention’s accuracy relative to Full Attention on these long‑context benchmarks, which makes it impossible to gauge the absolute compression cost.

**Mitigation status.** The limitation is not discussed directly, but Section 8 and Appendix A mention future evaluation on “broader domains such as coding and agentic tasks.” Whether the method’s design will prove equally effective for those domains remains an open question.