# LongLoRA: Efficient Adaptive Finetuning of Long-Context LLMs

**ArXiv:** [2402.17764](https://arxiv.org/abs/2402.17764)
**Authors:** Shaohan Huang, Satwik Kottur, Siyuan Feng, Bo Liu, Zhuoyi Yang, Wenqiang Lei, Yujia Xie, Tianyu Gao, Yunfan Li, Yu Wu, Weizhu Chen, Furu Wei
**Institutions:** Microsoft Research Asia, MIT, Shenzhen International Graduate School, Tsinghua University, Peking University

## 🎯 Pitch

LongLoRA innovatively extends the context length of pre-trained large language models up to 100k tokens through a novel combination of Shifted Sparse Attention and an enhanced LoRA fine-tuning, allowing this breakthrough on a single 8×A100 machine. This method preserves high performance during inference, significantly reducing computational requirements and enabling long-context adaptation on accessible hardware, making expansive applications such as document-length analysis and multi-turn conversations feasible.

---

## 1. Executive Summary
LongLoRA introduces a practical way to expand a pre‑trained large language model’s `context length` (how many tokens it can use at once) by fine‑tuning, without the usual prohibitive compute. It combines a training‑time attention variant called `Shifted Sparse Attention (S2‑Attn)` with a small but crucial modification to `LoRA` fine‑tuning—also training the embedding and normalization layers (“LoRA+”). Together, these let LLaMA2‑class models reach up to 100k tokens (7B) and 32k tokens (70B) on a single 8×A100 machine while keeping standard full attention at inference time and achieving perplexity close to full fine‑tuning (Figures 1, 4; Tables 2–4, 12).

## 2. Context and Motivation
- Problem addressed
  - Long contexts are valuable for tasks like summarizing lengthy documents, answering questions over multi‑chapter books, or tracing long conversations. But training or fine‑tuning transformers for long context is costly because self‑attention scales quadratically in sequence length. For example, going from 2k to 8k tokens increases self‑attention compute by 16× (Abstract; Section 1).
- Why it matters
  - Practical: Many real‑world tasks require much longer contexts than the 2k–4k typical in pre‑training (Section 1).
  - Accessibility: Prior strong results required large compute budgets (e.g., 32–128 high‑end GPUs/TPUs), which are out of reach for many labs (Section 1).
- Prior approaches and their limits
  - Full fine‑tuning with positional modifications (e.g., Position Interpolation “PI”): achieves long context but is compute‑heavy (Section 2, “Long-context LLMs”; Section 1 cites PI and Focused Transformer resource usage).
  - Efficient or sparse attention families (Longformer, BigBird, Reformer, etc.): reduce complexity but differ substantially from the full attention used in pre‑training, making them a poor drop‑in for fine‑tuning pre‑trained LLMs; quality often lags when swapped in post‑hoc (Section 2; Table 6).
  - Vanilla `LoRA`: lightweight, but when used alone for long‑context adaptation it underperforms, even with high ranks (Table 2).
- Positioning of this work
  - LongLoRA targets efficient fine‑tuning of existing LLMs to longer contexts while preserving standard dense attention at inference. It offers:
    - Training‑only sparse attention (`S2‑Attn`) that approximates full attention closely (Table 1) but reverts to full attention for inference (Figures 2–3).
    - A revised parameter‑efficient recipe (`LoRA+`), which adds trainable embeddings and layer norms to standard LoRA, closing much of the performance gap to full fine‑tuning with minimal extra parameters (Table 2; Figure 2c).

## 3. Technical Approach
The method has two pillars: S2‑Attn for efficient training, and LoRA+ for effective adaptation to new (long) positional regimes.

1) Shifted Sparse Attention (S2‑Attn) — training‑time only
- Intuition
  - If you split a long sequence into local groups and compute attention only within each group, training becomes much cheaper (linear in group size per position rather than in total length). But naive local attention blocks groups from communicating, harming quality at very long lengths (Table 1, “Short Attn Short ✗” rows).
- Mechanism (Figures 2–3; Algorithm 1)
  - Partition the token sequence of length `N` into groups of size `G` (e.g., `G = N/4` is the default; Appendix B.2).
  - Split attention heads into two halves:
    - Heads in pattern 1: compute local attention within each group (no shift).
    - Heads in pattern 2: “shift” tokens by `G/2` positions before grouping, so each group straddles the boundary between two neighboring groups. This enables information to flow across groups without increasing compute.
  - Implementation (Algorithm 1) boils down to two tensor operations around the standard attention call:
    - Before attention: `chunk` heads into two halves; `roll` the second half by `-G/2` along the token dimension; `view` to treat groups as mini‑batches; call the usual attention on each group.
    - After attention: reverse the `roll` for the second half and `cat` the two halves back together.
  - Optional masking tweak: shifting can, in principle, risk leakage across causal boundaries; Appendix B.3 shows variants that prevent this with negligible impact (Table 8; Figure 6).
- Why this design (vs. other efficient attentions)
  - It preserves the “shape” of local attention seen during pre‑training (short segments), so switching back to full attention at inference works well (Table 1 shows S2‑Attn ≈ full attention under full fine‑tuning; Table 6 shows other patterns degrade when switched to full attention at test time).
  - It provides tangible compute savings while remaining “close” to dense attention in behavior (Figures 2–3; Table 11).

2) LoRA+ — small but critical change to parameter‑efficient fine‑tuning
- Problem with vanilla LoRA for long contexts (Table 2)
  - Only adapting low‑rank updates to attention projections leaves a large gap to full fine‑tuning when extending context. Increasing LoRA rank up to 256 barely helps (perplexity ~11.98 vs 8.08 for full FT at 32k).
- Modification
  - Also unfreeze the input `embedding` layer and `layer normalization` parameters during fine‑tuning. These are a tiny fraction of total parameters (for LLaMA2‑7B, embeddings <2%, norms ≤0.004%; Figure 2c), but they anchor the model’s input statistics and positional handling—both shift when sequences get much longer.
- Effectiveness
  - With `+Norm & +Embed`, LoRA+ reaches perplexity 8.12 at 32k, nearly matching full fine‑tuning 8.08 (Table 2).

3) Training and inference protocol
- Training setup (Section 4.1)
  - Data: RedPajama corpus; next‑token prediction objective.
  - Optimizer/HPs: AdamW; learning rate 2e‑5 (7B/13B) or 1e‑5 (70B); warmup; weight decay 0; 1000 steps; per‑device batch 1 with grad accumulation 8 (global batch 64 on 8 GPUs).
  - Infrastructure: Flash‑Attention2 and DeepSpeed (stage 2/3) (Appendix A.1).
  - Positional scaling: apply `Position Interpolation (PI)` to extend RoPE indices to the target context (Sections 4.1, 2).
- Inference
  - Use standard full attention; S2‑Attn is training‑only (Figures 2–3). This maintains compatibility with existing acceleration like Flash‑Attention2 and avoids accuracy loss from approximate inference attention.

4) Optional SFT for long‑form instruction following
- The paper builds a small long‑context QA dataset `LongAlpaca‑12k` (9k long QAs + 3k short QAs) and applies supervised fine‑tuning to evaluate on long‑context instruction‑following benchmarks (Appendix B.6; Tables 9–10; Figures 8–9).

Simplified example of S2‑Attn
- Suppose `N = 8192` and `G = 2048` (group size is N/4, the default).
  - Heads 1..H/2: attend within [1..2048], [2049..4096], [4097..6144], [6145..8192].
  - Heads H/2+1..H: roll tokens by 1024 (G/2) before grouping, so groups are [−1023..1024]→wraps around, [1025..3072], … Each group now overlaps neighbors, passing information across boundaries at no extra cost.

## 4. Key Insights and Innovations
- Training‑time “short attention” can approximate “long attention” if you introduce a half‑head shift
  - Novelty: the specific two‑pattern, half‑head `G/2` shift (S2‑Attn) balances locality and boundary communication (Figures 2–3).
  - Why it matters: It reduces training compute while keeping the model compatible with full attention at inference, which many efficient attentions fail to do when applied post‑hoc (Table 6).
- Tiny trainable layers (embeddings + norms) unlock long‑context adaptation under LoRA
  - Novelty: `LoRA+` shows that opening just embeddings and norms collapses the large gap between LoRA and full fine‑tuning for long‑context extension (Table 2).
  - Significance: Near‑full‑FT quality with parameter‑efficient training; the extra parameters are negligible (Figure 2c).
- Keep the model architecture unchanged at inference
  - Novelty: Unlike works that alter the attention pattern permanently, LongLoRA reverts to standard dense attention for inference.
  - Payoff: Compatibility with existing accelerations (Flash‑Attention2), inference behavior, and downstream ecosystems (Abstract; Figure 2 caption; Section 3.2 “Consistency to Full Attention”).
- Practical scale on modest hardware
  - Contribution: Extends LLaMA2‑7B to 100k tokens, 13B to 65k, and 70B to 32k on a single 8×A100 node (Table 4; Table 15).
  - Efficiency: With Flash‑Attention2, training hours drop up to ~1.8× versus LoRA at 65k (92.5h → 52.4h; Table 12; Figure 1, right). Without Flash‑Attention2, speedups and memory savings are even larger (Table 13).

## 5. Experimental Analysis
- Evaluation methodology
  - Datasets
    - Language modeling: `PG19` books (validation/test), `proof‑pile` (math proofs) (Sections 4.1, 4.2).
    - Retrieval tasks: LongChat topic retrieval (conversation contexts up to 16k) (Table 5).
    - Synthetic probing: passkey retrieval (Figure 4; Appendix A.2).
    - Long‑form instruction evaluation: `LongBench` and `LEval` after SFT (Tables 9–10; Appendix B.6).
  - Metrics
    - `Perplexity` (lower is better) with a 256‑token sliding window for long documents (Section 4.1).
    - Retrieval accuracy for topic and passkey tasks (Tables 5; Figure 4).
    - GPT‑4 judged win‑rates for open‑ended outputs (LEval, Table 10).
  - Baselines and variants
    - Full attention + full fine‑tuning; vanilla LoRA; S2‑Attn with/without LoRA+; alternative efficient attentions (dilated, block sparse, stride sparse) (Tables 1–3, 6).
    - Ablations on S2‑Attn group size and variants; convergence behavior (Figures 5–6; Table 7–8).
    - Efficiency breakdowns with/without Flash‑Attention2, and FLOPs profiles (Tables 11–13).

- Main quantitative results
  - Long context language modeling quality
    - S2‑Attn quality matches full attention during full FT: at 8k–32k targets, S2‑Attn under full FT yields perplexity ~8.03–8.08 on PG19 validation, essentially identical to full attention (Table 1).
    - LoRA+ recovers full‑FT quality: at 32k target, full FT is 8.08; vanilla LoRA (r=8..256) stays around 11.4–12.0; adding `+Norm & +Embed` yields 8.12 (Table 2).
    - Proof‑pile results show good scaling: for LLaMA2‑7B, training at 32k and evaluating at 32k yields PPL 2.50 with LongLoRA (S2‑Attn + LoRA+) and 2.49 with S2‑Attn alone (Table 3). Similar trends for 13B (2.32 at 32k with LongLoRA).
  - Maximum context on one 8×A100 node (Table 4; Table 15)
    - 7B: trained to 100k, then evaluated across lengths; PPL drops from 3.36 (2k) to 2.52 (100k) on proof‑pile; on PG19 test, 7.04 at 100k.
    - 13B: trained to 65,536; PG19 PPL 6.57 at 65k.
    - 70B: trained to 32k; PG19 PPL 5.27 at 32k.
  - Retrieval tasks
    - Long conversation topic retrieval: Ours‑13B reaches accuracy 1.00/0.98/0.98/0.98/0.94 at 3k/6k/10k/13k/16k, comparable to LongChat‑13B (1.00/1.00/1.00/0.98/0.90) (Table 5).
    - Passkey retrieval: the 32k‑trained 7B maintains near‑perfect accuracy up to 33k–34k; extending only the position interpolation to 48k without retraining yields 60%–90% accuracy from 33k–45k. Baseline LLaMA2‑7B collapses after 4k (Figure 4).
  - Instruction following after SFT
    - LongBench (average across tasks): Ours‑7B scores 36.8, outperforming LLaMA2‑7B chat (31.0) and LongChat‑v1.5‑7B (34.3) (Table 9).
    - LEval win‑rate vs GPT‑3.5‑Turbo judged by GPT‑4: Ours‑7B wins 39.06% (higher than other LLaMA2‑based long‑context baselines in Table 10).
  - Efficiency and compute profiles
    - With Flash‑Attention2: training 7B for 65k context—LoRA takes 92.5h and 71.1 GB peak GPU memory; LongLoRA takes 52.4h and 69.8 GB (Table 12; Figure 1).
    - Without Flash‑Attention2: at 8k, S2‑Attn halves training time (17.5h → 8.2h) and reduces memory by ~1.8× (55.5 GB → 30.3 GB); full attention OOMs at 16k while S2‑Attn trains (Table 13).
    - FLOPs breakdown for 7B: at 65k context, full attention totals 3118T FLOPs with attention 72.2% of total; S2‑Attn cuts total to 1429T and reduces attention share to 39.4% (Table 11).

- Ablations and robustness
  - Group size: using G = N/4 or N/2 works best; smaller groups (N/6, N/8) degrade perplexity (Table 7).
  - S2‑Attn variants and masking: different shift directions or separate shifted groups have similar PPL to the default, alleviating leakage concerns (Table 8; Figure 6).
  - Alternative sparse attentions: when testing with full attention, S2‑Attn best preserves quality (PPL 8.12) vs dilated (9.70), stride (11.78), etc. (Table 6).
  - Convergence: full FT starts faster, but the final gap to LoRA+ is small after ~200 steps (Figure 5).

- Do the experiments support the claims?
  - Yes, on three fronts:
    - Quality: LongLoRA matches or nearly matches full‑FT perplexity at long contexts (Tables 1–3) and performs competitively on long‑context retrieval and instruction tasks (Tables 5, 9, 10; Figure 4).
    - Efficiency: Training‑time savings are substantial, especially at longer lengths and without Flash‑Attention2 (Tables 11–13; Figure 1).
    - Compatibility: Because inference uses standard attention, the method works with existing accelerators (Section 3.2) and does not sacrifice inference quality.

## 6. Limitations and Trade-offs
- Quadratic inference remains
  - S2‑Attn is training‑only; inference still uses full attention with quadratic complexity. Thus, while training is cheaper, very long‑context inference still requires significant memory/compute (Sections 3.2, 5).
- Reliance on positional rescaling
  - The approach depends on Position Interpolation to extend RoPE indices (Sections 4.1–4.2). Known PI behaviors appear (slightly worse PPL at shorter contexts after large‑context training; Section 4.2 notes this).
- Data and objective coverage
  - Main fine‑tuning is next‑token modeling on RedPajama; SFT uses a relatively small custom set (LongAlpaca‑12k). Generalization to diverse real‑world instruction settings may require more varied SFT data (Appendix B.6; Tables 9–10).
- Sensitivity to group size and patterns
  - While robust across variants, S2‑Attn quality depends on reasonable group sizes (Table 7) and the half‑head shift scheme (Table 6 shows other attentions do worse when switched back to full attention).
- Information leakage concerns (mitigated)
  - Shifting can nominally create wrap‑around groups. Appendix B.3 shows masking or architectural variants avoid leakage with similar performance (Table 8), but careful implementation is required.
- Compute is reduced, not eliminated
  - Even with gains, training to 65k+ contexts still consumes tens of hours on 8×A100 (Table 12); practicality for extremely large models or single‑GPU regimes may remain challenging.

## 7. Implications and Future Directions
- Field impact
  - LongLoRA reframes long‑context adaptation as an efficient fine‑tuning problem rather than a heavy re‑training or architecture‑replacement task. Keeping full attention at inference means existing tooling and optimizations remain usable (Figure 2; Section 3.2).
- What this enables
  - Routine long‑context upgrades for existing LLMs on modest hardware budgets (Table 4). This lowers the barrier for research and applied work on document‑length reasoning, legal/medical dossiers, multi‑file codebases, and long conversational agents.
- Practical applications
  - Long‑document QA and summarization, long‑horizon planning within a single prompt, multi‑turn assistants that retain earlier context, code assistants over large repositories, and book‑length analysis (Sections 4.2, B.6; Figures 8–9; Tables 5, 9–10).
- Suggested follow‑ups
  - Inference‑time efficiency: marry LongLoRA with retrieval or compressed memories to mitigate quadratic inference costs.
  - Dynamic group strategies: adapt group size G by layer or head, or learn the shift pattern.
  - Beyond text: extend the training‑time S2‑Attn idea to multimodal transformers (vision‑language) that need long video or document contexts.
  - Broader PEFT recipes: explore unfreezing other small sets of parameters (e.g., output embeddings, select MLP gates) to further close any remaining gap to full FT under domain shift.
  - Positioning alternatives: evaluate LongLoRA with other long‑position strategies (e.g., NTK‑aware RoPE, YaRN), hinted as compatible in the conclusion.

> Representative results
> - “LoRA for long context extension works well under the premise of trainable embedding and normalization” (Table 2: 8.12 PPL vs 8.08 full FT at 32k).
> - “LongLoRA extends LLaMA2‑7B from 4k to 100k and 70B to 32k on a single 8×A100 machine” (Table 4; Table 15).
> - “Training time reduction up to ~1.8× vs LoRA at 65k with Flash‑Attention2; far larger savings without it” (Table 12, Table 13; Figure 1).

Overall, LongLoRA’s key idea—train short, test long, but with a strategic shift that maintains cross‑group information and a minimal unfreezing of embeddings and norms—offers a compelling and practical path to long‑context LLMs without changing inference behavior.
