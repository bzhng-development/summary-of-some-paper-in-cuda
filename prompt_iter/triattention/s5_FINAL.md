## 5. Experimental Analysis

### Evaluation Methodology

- **Datasets.** Primary reasoning evaluation uses **AIME 2024** (30 problems), **AIME 2025** (30 problems), and **MATH 500** (500 problems). AIME benchmarks are competition‑level mathematics requiring multi‑step chain‑of‑thought; MATH 500 spans diverse mathematical reasoning tasks (Hendrycks et al., 2021). Longer‑context and memory‑retention analysis additionally employs the **Recursive State Query** benchmark (DFS simulation) and the **LongBench** (16 subtasks, 50 % KV budget) and **RULER** (retrieval, 4 K context) suites (Appendix E, F); these are used to probe generalisation beyond mathematics.

- **Base models.** Four reasoning‑capable LLMs spanning architectures and scales are evaluated: **Qwen3‑8B** (GQA), **DeepSeek‑R1‑Distill‑Llama‑8B**, **DeepSeek‑R1‑Distill‑Qwen‑7B**, and **GPT‑OSS‑20B**. These models are selected for their strong chain‑of‑thought reasoning and diverse pretraining backgrounds; most experiments are performed on Qwen3‑8B, with the others serving as cross‑model validation. Throughput and deployment tests also use **Qwen3‑32B (INT4)** on a consumer GPU (RTX 4090).

- **Metrics.** For reasoning benchmarks, correctness is measured by **pass rate**: on AIME each problem is sampled 8 times and the average fraction of correct answers is reported; on MATH 500 each problem is sampled once. Throughput is measured as the **average tokens generated per second** over a 16 K‑token decoding run on a single NVIDIA A100 80 GB GPU at the maximum batch size that fits in memory. KV‑cache memory reduction is expressed as the ratio of the original sequence length to the fixed budget $B$.

- **Baselines.** Three primary baselines are compared:
  * **Full Attention** – no KV pruning; the performance upper bound.
  * **SnapKV** (Li et al., 2024b) – an attention‑based method that scores keys from a local observation window.
  * **R‑KV** (Cai et al., 2025) – a state‑of‑the‑art KV‑cache compressor for reasoning models that combines attention‑based importance scoring with redundancy detection.
  Additional comparisons with StreamingLLM, PyramidKV, KnormPress, Ada‑KV+SnapKV, and H2O appear on LongBench and RULER (Appendix E–F), but the core analysis contrasts TriAttention with SnapKV and R‑KV.

- **Generation budget / compute accounting.** The key resource is the **KV‑cache budget** $B$, the number of key–value pairs retained per head. Uniform budgets are enforced across all layers and heads. All methods prune once every $\beta = 128$ generated tokens; when the cache exceeds $B$, it is trimmed back to $B$. For throughput comparisons, the measurement includes the overhead of periodic scoring and pruning, and all methods use FlashAttention‑2 (except GPT‑OSS which uses FlashAttention‑3 on H100). Default budgets are $B = 2048$ (AIME, etc.) and $B = 512$ (MATH 500, DS‑Llama to ensure compression is exercised). Generation length is capped at 32 768 tokens, temperature 0.6, top‑p 0.95.

- **Cross‑validation / statistical protocol.** No explicit cross‑validation on the test benchmarks is reported; the method’s hyper‑parameters (e.g., calibration settings) are fixed based on the calibration dataset. TriAttention’s offline statistics are extracted from a calibration corpus (e.g., ShareGPT, 200 K–960 K tokens) completely disjoint from the evaluation tasks; the stability of the statistics is confirmed by experiments varying the calibration data domain (Table 3C) and quantity (Appendix H). For each AIME problem, 8 samples are drawn to reduce variance; for MATH 500 only one sample is used per problem. No confidence intervals or error bars are provided, so the observed differences should be interpreted with that limitation in mind.

---

### Main Quantitative Results

#### Reasoning Task Accuracy (AIME, MATH 500)

**Tables 1 and 2** report accuracy on AIME24, AIME25, and MATH 500 across all models, using a fixed KV budget (2048 for AIME, 512 for MATH 500 except as noted). The headline finding is that **TriAttention consistently achieves the highest accuracy among compression methods**, often closing most of the gap to Full Attention while substantially outperforming SnapKV and R‑KV.

- On **AIME25** (Qwen3‑8B, budget 2048): Full Attention achieves 40.8 %; TriAttention reaches **32.9 %**, SnapKV 20.0 %, R‑KV 17.5 %. TriAttention thus more than doubles the pass rate of the next‑best competitor (Table 1).
- On **AIME24** (Qwen3‑8B, budget 2048): TriAttention 42.1 % vs. R‑KV 25.4 % and SnapKV 34.6 %. Full Attention is at 57.1 %.
- On **MATH 500** (Qwen3‑8B, budget 512): TriAttention obtains 56.0 %, Full Attention 69.6 %, R‑KV 46.4 %, SnapKV 49.2 % (Table 2). With a budget of 1024, TriAttention reaches 68.4 %, nearly matching Full Attention (69.6 %), as shown in the budget sweep of **Figure 5C**.
- Across the other three models (DeepSeek‑Distill‑Llama‑8B, DeepSeek‑Distill‑Qwen‑7B, GPT‑OSS‑20B), TriAttention also leads all compression baselines on AIME24 and AIME25, with the gap being particularly large on GPT‑OSS‑20B (TriAttention 59.2 % vs. R‑KV 49.6 % on AIME24).

**Figure 5** plots accuracy versus KV budget for Qwen3‑8B on the three reasoning benchmarks. TriAttention dominates R‑KV at every budget, and the advantage is most pronounced at low‑to‑mid budgets. On MATH 500, TriAttention **matches Full Attention at budget 1024 and slightly exceeds it** at higher budgets (Figure 5C). On AIME25, at budget 2048 TriAttention reaches 32.9 % vs. R‑KV’s 17.5 %; with budget 4096 it climbs to 43.3 %, surpassing the Full Attention baseline (40.8 %). This demonstrates that TriAttention not only compresses without severe degradation but can even slightly outperform Full Attention when a modest budget is provided, possibly because pruning occasionally removes noise.

#### Memory Retention (Recursive State Query Benchmark)

The DFS‑based Recursive State Query benchmark (Appendix C) is specifically designed to stress the ability of a KV‑cache compressor to retain intermediate states during backtracking. **Figure 5D** reports accuracy (stack‑exact‑match) on Qwen3‑8B for depths from 6 to 20 steps, comparing Full Attention, R‑KV, and TriAttention (both compression methods use budget 2048).

- At low‑to‑moderate depths (6–14), TriAttention performs **comparably to Full Attention** and slightly outperforms it at depths 8 and 12.
- At depth 16 and beyond, only TriAttention maintains high accuracy; **R‑KV suffers catastrophic degradation**, falling from ∼61 % at depth 14 to ∼31 % at depth 16, while TriAttention remains above 60 %.
- This sharp drop indicates that R‑KV’s observation‑window‑based pruning consistently evicts intermediate states that are essential for backtracking, whereas TriAttention’s distance‑based scoring retains them predictably.

#### Throughput and Efficiency

The paper quantifies the practical benefit by measuring throughput at configurations where TriAttention achieves accuracy comparable to Full Attention (**Table 4**). All measurements are taken on a single A100 80 GB GPU at maximum batch size.

- **MATH 500:** TriAttention with budget 1024 achieves 68.4 % accuracy (Full Attention 69.6 %) while delivering **6.3× higher throughput** (1405.2  vs.  222.8  tokens/s). The KV cache memory is reduced by a factor of roughly 32 K / 1024 ≈ 31 × (the paper does not explicitly state this factor for MATH 500, but the throughput gain includes the memory benefit).
- **AIME25:** TriAttention with budget 3072 matches Full Attention at 40.8 % and yields **2.5× higher throughput** (563.5  vs.  222.8  tokens/s). The KV memory reduction factor is 32 768 / 3072 ≈ **10.7×**, as quoted in the abstract and **Figure 1**.
- **AIME24:** with budget 4096, accuracy is 54.6 % (vs. Full 57.1 %) and throughput improves 1.9× (413.9  vs.  222.8  tokens/s). This configuration is slightly below Full Attention accuracy; the 2.5× throughput achievement is reported for AIME25 where equality is strictly met.

**Table 5** compares TriAttention and R‑KV directly under two regimes.  
- **Comparable accuracy:** TriAttention reaches the same performance as R‑KV with **half the KV budget** (1024 vs. 2048 on MATH 500 and AIME24) while achieving **85 % higher throughput** (1405.2  vs.  760.4  tokens/s on MATH 500).  
- **Comparable memory (same budget):** at budget 1024 (the common setting where both methods get the same memory), TriAttention improves MATH 500 accuracy by **+8.0 %** (68.4 % vs. 60.4 %) and AIME24 accuracy by **+15.4 %** (25.8 % vs. 10.4 %), with throughput being nearly equal (1405.2  vs.  1345.5  tokens/s).  

These numbers demonstrate that TriAttention shifts the entire accuracy–efficiency Pareto front upward relative to R‑KV.

#### Results on LongBench and RULER (Generalisation)

The main paper references additional results on **LongBench** (16 diverse subtasks, 50 % KV budget) and **RULER** (retrieval tasks, 4 K context) to verify that the gains are not specific to math reasoning. In **Appendix F (Table B)** TriAttention achieves the highest average LongBench score (48.1) among compression methods, surpassing Ada‑KV+SnapKV (45.6), SnapKV (45.2), and PyramidKV (42.7). It wins on 11 of 16 subtasks. On RULER (**Table C**), TriAttention scores **66.1** vs. SnapKV’s 55.6 and StreamingLLM’s 61.1. A separate comparison with H2O on the 12 LongBench subtasks where H2O can fit in 48 GB (**Table D**) shows TriAttention wins 10 out of 12 subtasks (average 45.4 vs. 41.4). While these results are in the appendix and not as extensively analysed, they corroborate that TriAttention’s pre‑RoPE scoring generalises beyond the AIME/MATH domain.

---

### Ablation Studies and Robustness Checks

**Removal of the trigonometric series score $S_{\text{trig}}$ (Table 3A):** When the trigonometric series term is omitted and only the norm‑based score $S_{\text{norm}}$ is used, AIME24 accuracy collapses from 42.1 % to **18.8 %** and AIME25 from 32.9 % to **21.2 %**. This confirms that the distance‑preference signal captured by the centres is the dominant source of key importance, not just norm magnitude.

**Removal of the norm‑based score:** In text the paper states that removing $S_{\text{norm}}$ and relying solely on $S_{\text{trig}}$ “drops AIME24 accuracy from 45.8 % to 40.4 %” (this 45.8 % corresponds to a configuration with larger offsets; the exact numbers vary but the drop is ∼5.4 %). Thus the norm term provides a complementary, albeit smaller, benefit.

**Concentration‑based weighting (Table 3B):** Replacing the adaptive weighting $(1-R_f)$ with the base norm score $S_{\text{norm}}^{(0)}$ (which does not down‑weight concentrated bands) reduces AIME24 accuracy from 42.1 % to 41.3 % and AIME25 accuracy from 32.9 % to **28.7 %** (a 4.2‑percentage‑point drop on the harder benchmark). The weighting is especially valuable in harder tasks where noisy norm information in concentrated heads would otherwise hurt the combined score.

**Cross‑domain calibration (Table 3C):** Offline statistics collected on coding data (LiveCodeBench; Jain et al., 2025) instead of reasoning data yield AIME24 accuracy of 44.2 % vs. 42.1 % (reasoning calibration) and AIME25 accuracy of 29.2 % vs. 32.9 %. The small differences (sometimes even favouring coding calibration) confirm that the Q/K centres are model‑intrinsic and not overfit to any specific task domain. This is a practically important result because it eliminates the need for per‑task calibration.

**Future offset design (Appendix G, Table E):**  
- Increasing the maximum offset distance from 128 to 4096 (with denser sampling) raises AIME24 accuracy from 41.7 % to **48.8 %** (+7.1 %), proving that long‑range future queries contribute meaningful importance information.  
- Using **geometric spacing** {1, 2, 4, …} instead of **linear spacing** for the offsets causes a dramatic performance difference: 45.8 % vs. 28.7 % (−17.1 %). Near‑distance regions need finer sampling because the trigonometric series changes more rapidly there; linear spacing severely underestimates the importance of nearby future positions.

**Calibration data quantity and quality (Appendix H, Table F):**  
- Performance is stable across calibration sizes from 50 K to 960 K tokens: 45.4 %, 45.8 %, 45.8 % on AIME24.  
- Calibration on low‑quality data (Google homepage HTML) achieves 46.2 %, comparable to high‑quality chat data (46.7 %). This demonstrates that the Q/K centres are robust to the choice of calibration data, consistent with the claim that they are a model‑intrinsic property.

**MLA architecture validation (Appendix I, Table G):** TriAttention’s assumptions are verified on the GLM‑4.7‑Flash model, which uses Multi‑head Latent Attention (MLA). The Q/K concentration (MRL) is even stronger in MLA: 96.6 % of heads have $R > 0.95$ vs. 84.7 % in Qwen3‑8B (GQA). Reconstruction correlation remains comparable, confirming that the distance‑preference mechanism is architecture‑general.

**Comparisons beyond the primary baselines (Appendix E, Table A):** On AIME24 with DeepSeek‑R1‑Distill‑Qwen‑7B, TriAttention is compared at varying budgets against LazyEviction, H2O, TOVA, and RaaS. TriAttention outperforms all methods at every budget, and at 30 % KV budget it **matches Full Attention** (46.7 %). This additional head‑to‑head reinforces the claim that the method surpasses a wider range of competitors.

**OpenClaw deployment on a single consumer GPU (Appendix J):** A qualitative demonstration shows that TriAttention enables the OpenClaw multi‑turn agent (Qwen3‑32B, INT4) to run on an RTX 4090 (24 GB) without out‑of‑memory errors, whereas Full Attention fails. This validates the practical deployment claim, albeit without a formal accuracy metric for the agent.

---

### Critical Assessment

The experiments collectively provide strong evidence that TriAttention compresses the KV cache with substantially less reasoning degradation than prior methods, and that the underlying Q/K‑concentration and trigonometric‑series mechanism is responsible for the gain. However, several limitations in the experimental design temper the generality and precision of the conclusions.

- **Are the headline throughput and memory claims rigorously supported?** The statements “2.5× higher throughput” and “10.7× KV memory reduction” are specific to the AIME25 setting where TriAttention with budget 3072 exactly matches Full Attention’s 40.8 % accuracy (Table 4, Figure 1). This is a single operating point; the throughput advantage varies across benchmarks (6.3× on MATH 500, 1.9× on AIME24). The paper carefully notes the conditions, so the claims are accurate for those conditions, but one should not extrapolate a universal factor. Moreover, the throughput measurement uses a batch size that fills the GPU; in a deployment with varying batch sizes or lower GPU utilisation, the relative speedup may differ.

- **Is the “matches Full Attention” claim adequately tested?** On AIME25, the accuracy match is exact (both 40.8 %), but with only 240 samples (30 problems × 8) the observed equality could be coincidental. Confidence intervals are not reported, so we cannot assess whether TriAttention’s true accuracy might be slightly lower or higher. On MATH 500, TriAttention at budget 1024 achieves 68.4 % vs. Full 69.6 %—the gap is 1.2 percentage points but is presented as “closely matching”; the statistical significance of this small gap is unknown. The robustness would be stronger if multiple seeds or wider error bars were shown.

- **Do the experiments establish that TriAttention solves the instability of observation‑window methods?** The ablation on Strig (Table 3A) and the memory‑retention benchmark (Figure 5D) are the most direct evidence. Removing Strig (leaving only norm‑based scoring) causes a drastic accuracy drop, and R‑KV’s catastrophic failure on DFS recursion contrasts sharply with TriAttention’s stability. These results strongly support the claim that the trigonometric‑series signal is the critical component and that observation‑window methods suffer from state loss that TriAttention avoids. However, the DFS benchmark is a synthetic stress test, not a natural reasoning task. The paper does not isolate whether the failure of R‑KV on AIME is due specifically to retrieval‑head issues—it is plausible but not proven.

- **Generalisation beyond mathematical reasoning:** The LongBench and RULER results (Appendix F) extend the evidence to summarisation, QA, dialogue, retrieval, and code tasks. TriAttention leads compression methods there, but the margins are smaller than on AIME (e.g., LongBench average 48.1 vs. 45.6 for Ada‑KV+SnapKV). These results indicate that TriAttention still benefits general tasks, but the dramatic improvements on reasoning may partly reflect that reasoning has longer and more structured dependencies where distance‑based scoring is especially advantageous. The paper does not evaluate on extremely long contexts (e.g., 128 K tokens) or on multi‑turn agent benchmarks with quantitative success metrics; the OpenClaw demo is promising but lacks a controlled accuracy comparison.

- **Single‑model family for most ablations:** The core ablations (Table 3, Appendix G–H) are performed solely on Qwen3‑8B. While cross‑model validation (Tables 1–2) shows TriAttention consistently outperforming baselines on three other models, the detailed understanding of how the mechanism behaves (e.g., the relative importance of Strig vs. Snorm across architectures) is only characterised for one model. This is a typical limitation but worth noting.

- **Potential confounding in the budget sweep:** In Figure 5, TriAttention occasionally exceeds Full Attention at high budgets (e.g., AIME25 at budget 4096). This could indicate that pruning acts as a regulariser that removes attention noise, but it could also be statistical noise given the small test set. The paper does not discuss this phenomenon, and it remains unclear whether it is a reliable property.

- **Missing baselines and ablations:** The paper does not compare against a simple “distance‑based” baseline that retains keys uniformly at the positions where the trigonometric series peaks (without using key‑specific norms). Such a baseline would isolate whether the centre‑driven distance preference alone, without key‑content discrimination, is sufficient. Additionally, there is no ablation that replaces the Q centre with a random constant to confirm that the actual direction of the centre matters (though the high reconstruction correlation implicitly does). Finally, the scoring function uses a fixed geometric offset set; sensitivity to the precise offset numbers (e.g., using only powers of 2 vs. a different progression) is only partially explored (linear vs. geometric spacing, but not, say, logarithmic spacing with different base).

- **Limited scale of test sets:** AIME24 and AIME25 each contain 30 problems; even with 8 samples, the confidence interval around accuracy estimates is wide. The performance advantage on AIME25 (TriAttention 32.9 % vs. R‑KV 17.5 %) is large enough to be convincing, but the precise margin has high variance. MATH 500 with 500 problems gives more stable estimates, but TriAttention’s advantage there is smaller relative to R‑KV (56.0 % vs. 46.4 %) and is subject to the same lack of error bars.

In summary, the experiments convincingly demonstrate that TriAttention substantially outperforms prior KV‑cache compression methods on mathematical reasoning benchmarks and on a dedicated memory‑retention test, while achieving practical throughput improvements. The ablation studies firmly tie the gains to the trigonometric‑series component and the concentration‑based weighting. However, the evidence for the precise efficiency factors, the exact magnitude of the advantage on broader tasks, and the statistical robustness of the accuracy equality with Full Attention is less definitive due to the small test sets and unreported uncertainties. These are typical limitations of an initial empirical study and do not undermine the core contribution, but they indicate where more extensive evaluation would strengthen the results.